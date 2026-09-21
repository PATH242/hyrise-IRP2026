#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "benchmark_config.hpp"
#include "hyrise.hpp"
#include "operators/abstract_operator.hpp"
#include "operators/get_table.hpp"
#include "operators/operator_performance_data.hpp"
#include "operators/sort.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "sql/sql_pipeline.hpp"
#include "sql/sql_pipeline_builder.hpp"
#include "sql/sql_pipeline_statement.hpp"
#include "storage/chunk.hpp"
#include "storage/table.hpp"
#include "tpch/tpch_constants.hpp"
#include "tpch/tpch_table_generator.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

// =====================================================================================================================
// Isolated-sort evaluation of Hyrise sort routines, structured as the DuckDB v1.4.0 "Sorting Again" blog analog.
//
// One process handles exactly ONE (routine, scale factor) cell. That is deliberate:
//   * Hyrise caches generated tables in a CWD-relative `tpch_cached_tables/sf-<sf>` directory, so all runs of all
//     routines share one cache and are provably measured on identical input bytes.
//   * Two scale factors in one process would register `lineitem` twice in the StorageManager.
// run_evaluation.sh loops over the scale factors from one shared working directory.
//
// Two harnesses:
//   operator  Sort driven directly (GetTable + column pruning + ForceMaterialization::Yes). This is the payload-cost
//             measurement: the sort produces a RowIDPosList and the payload is gathered afterwards in WriteOutput.
//   sql       `SELECT * FROM lineitem ORDER BY <key set>` through SQLPipeline.
//
// REQUIRED PATCH for the sql harness -- carry it in the evaluation commit so every routine branch gets it:
//   src/lib/logical_query_plan/lqp_translator.cpp, _translate_sort_node(), currently:
//       current_pqp = std::make_shared<Sort>(current_pqp, column_definitions);
//   change to:
//       current_pqp = std::make_shared<Sort>(current_pqp, column_definitions, Chunk::DEFAULT_SIZE,
//                                            Sort::ForceMaterialization::Yes);
//   Both needed headers are already included there. Without this, a query plan's Sort returns a REFERENCE table and
//   never gathers the payload, so WRITE_OUT_US is ~0 and the run is not comparable to the blog's query (whose
//   any_value(COLUMNS(*)) exists precisely to force the payload through). The MATERIALIZED column records, per row,
//   whether the patch was actually in effect -- check it rather than trusting that the branch had it.
// =====================================================================================================================

// ---------------------------------------------------------------------------------------------------------------
// PHASE TABLE -- the one place to edit when a phase is added or a branch renames a step.
// Semantics fixed across all routine branches:
//   materialization  key materialization / normalization of the sort columns
//   sort             the per-thread sort of the runs
//   merge            the merge of sorted runs; reported as 0 by routines that do not merge
//   merge path       merge-path index computation; reported as 0 by routines that do not compute one
//   write out        payload gather into ValueSegments (Hyrise-specific, after the PosList exists)
//
// Two rules, both load-bearing:
//   1. Every branch MUST charge the same work to the same step, or the phase columns stop being comparable across
//      routines even though TOTAL_US stays valid. Verify with: grep set_step_runtime src/lib/operators/sort.cpp
//   2. Sort::OperatorSteps must be IDENTICAL on every branch, baseline included -- this file names the enumerators, so
//      a branch missing one will not compile, and a branch with a different order would shift nothing here but would
//      break anything reading step_runtimes positionally. Add new enumerators everywhere and let routines that do not
//      use them leave the step at 0 (unset steps read back as 0).
// ---------------------------------------------------------------------------------------------------------------
struct PhaseSpec {
  std::string_view csv_column;
  Sort::OperatorSteps step;
};

constexpr auto PHASES = std::array{
    PhaseSpec{"MATERIALIZE_US", Sort::OperatorSteps::MaterializeSortColumns},
    PhaseSpec{"SORT_US", Sort::OperatorSteps::Sort},
    PhaseSpec{"MERGE_US", Sort::OperatorSteps::TemporaryResultWriting},
    // Uncomment once `MergePath` exists in Sort::OperatorSteps on ALL branches (append it to the enum; order is
    // irrelevant here since this table maps by name). Nothing else needs to change -- the CSV header follows.
    // PhaseSpec{"MERGE_PATH_US", Sort::OperatorSteps::MergePath},
    PhaseSpec{"WRITE_OUT_US", Sort::OperatorSteps::WriteOutput},
};
constexpr auto PHASE_COUNT = PHASES.size();

// ---------------------------------------------------------------------------------------------------------------
// Workload definition
// ---------------------------------------------------------------------------------------------------------------
constexpr auto TABLE_NAME = "lineitem";

// Encoding is not an axis: BenchmarkConfig::encoding_config is left at its default, which is Hyrise's own
// "Automatic" setting (the default of hyriseBenchmarkTPCH). Note this is NOT "unencoded" -- with no preferred spec,
// auto_select_segment_encoding_spec picks PER COLUMN: Unencoded for unique-valued columns, FrameOfReference for Int,
// Dictionary for everything else. It is deterministic and identical across routines, so the comparison stays fair,
// but lineitem ends up with a mix, which is worth one sentence in the paper. Recorded in the CSV for the record.
constexpr auto ENCODING_TAG = "automatic";

// ---------------------------------------------------------------------------------------------------------------
// KEY SETS -- add one here and it is immediately selectable with --key; nothing else needs to change.
//
//   shipdate  the DuckDB blog analog: one 10-char SSO string key. This is the key set the payload sweep belongs to.
//   ties      3 leading groups x 2 -> almost all work lands in tie-breaking. The cell where the upstream multi-pass
//             design (one full stable_sort PER sort column) is most exposed against a single-pass composite sort.
//   pk        lineitem's primary key: two int32 columns, unique, and generated in ascending order -- so this doubles
//             as the pre-sorted / adaptivity probe.
//   comment   one key, wide in BYTES rather than columns: 10-44 chars, heap-allocated, straddles the SSO boundary.
//
// CAVEAT for the paper: for a multi-column key the upstream baseline runs N full sorts, so its MATERIALIZE_US and
// SORT_US are sums ACROSS PASSES, not one pass's cost -- the operator exposes only four cumulative step timers and
// cannot separate them. The single-pass routines' bars are one pass. That asymmetry is the result, but it has to be
// stated rather than left for the reader to assume like-for-like.
// ---------------------------------------------------------------------------------------------------------------
struct KeySet {
  std::string_view name;
  std::vector<std::string> columns;
};

const auto KEY_SETS = std::vector<KeySet>{
    {"shipdate", {"l_shipdate"}},
    {"ties", {"l_returnflag", "l_linestatus", "l_shipdate"}},
    {"pk", {"l_orderkey", "l_linenumber"}},
    {"comment", {"l_comment"}},
};

static const KeySet& key_set_by_name(const std::string& name) {
  for (const auto& key_set : KEY_SETS) {
    if (key_set.name == name) {
      return key_set;
    }
  }
  auto known = std::string{};
  for (const auto& key_set : KEY_SETS) {
    known += std::string{key_set.name} + " ";
  }
  Fail("Unknown --key '" + name + "'. Known key sets: " + known);
}

static std::string join(const std::vector<std::string>& parts, const std::string& separator) {
  auto out = std::string{};
  for (auto index = size_t{0}; index < parts.size(); ++index) {
    out += (index ? separator : "") + parts[index];
  }
  return out;
}

// The payload is every lineitem column that is not part of this key set, in table order. Payload width N takes the
// first N. Computed rather than hardcoded, because which columns are available depends on the key set -- which also
// means width N is a DIFFERENT set of columns for different key sets: compare widths within a key set, never across.
static std::vector<std::string> payload_column_order(const std::string& table_name,
                                                     const std::vector<std::string>& key_columns) {
  const auto table = Hyrise::get().storage_manager.get_table(table_name);
  auto payload = std::vector<std::string>{};
  for (const auto& column_name : table->column_names()) {
    if (std::ranges::find(key_columns, column_name) == key_columns.end()) {
      payload.emplace_back(column_name);
    }
  }
  return payload;
}

// ---------------------------------------------------------------------------------------------------------------
// Configuration parsed from the command line
// ---------------------------------------------------------------------------------------------------------------
struct Options {
  std::string routine;
  std::string branch;
  std::string commit;
  std::string machine;
  std::string out_path = "SORT_RESULTS.csv";
  std::vector<std::string> harnesses{"operator", "sql"};
  std::vector<std::string> payload_widths{"1", "4", "8", "all"};
  std::string key_set = "shipdate";
  float scale_factor = 1.0F;
  size_t run_count = 6;  // first run is a discarded warm-up, so 5 measured -- the blog's median-of-5
};

static void print_usage() {
  std::cout << "usage: sort_evaluation --routine <tag> [options]\n"
            << "  --routine  <tag>                 required; identifies the sort implementation in the CSV\n"
            << "  --sf       <float>               scale factor (default 1)\n"
            << "  --key      <shipdate|ties|pk|comment>  sort key set (default shipdate)\n"
            << "  --harness  <operator[,sql]>      which harnesses to run (default operator,sql)\n"
            << "  --payload  <1,4,8,all>           payload widths for the operator harness\n"
            << "  --runs     <n>                   total runs; the first is a discarded warm-up (default 11)\n"
            << "  --out      <path>                CSV to append to (default SORT_RESULTS.csv)\n"
            << "  --branch/--commit/--machine <s>  metadata stamped into every row\n";
}

static std::vector<std::string> split(const std::string& value, const char delimiter) {
  auto parts = std::vector<std::string>{};
  auto stream = std::stringstream{value};
  auto part = std::string{};
  while (std::getline(stream, part, delimiter)) {
    if (!part.empty()) {
      parts.push_back(part);
    }
  }
  return parts;
}

static Options parse_options(int argc, char* argv[]) {
  auto options = Options{};

  for (auto index = 1; index < argc; ++index) {
    const auto flag = std::string{argv[index]};
    const auto next = [&]() {
      Assert(index + 1 < argc, "Missing value for " + flag);
      ++index;
      return std::string{argv[index]};
    };

    if (flag == "--routine") {
      options.routine = next();
    } else if (flag == "--sf") {
      options.scale_factor = std::stof(next());
    } else if (flag == "--key") {
      options.key_set = next();
    } else if (flag == "--harness") {
      options.harnesses = split(next(), ',');
    } else if (flag == "--payload") {
      options.payload_widths = split(next(), ',');
    } else if (flag == "--runs") {
      options.run_count = static_cast<size_t>(std::stoul(next()));
    } else if (flag == "--out") {
      options.out_path = next();
    } else if (flag == "--branch") {
      options.branch = next();
    } else if (flag == "--commit") {
      options.commit = next();
    } else if (flag == "--machine") {
      options.machine = next();
    } else if (flag == "--help" || flag == "-h") {
      print_usage();
      std::exit(0);
    } else {
      std::cerr << "unknown option: " << flag << "\n";
      print_usage();
      std::exit(2);
    }
  }

  Assert(!options.routine.empty(), "--routine is required");
  key_set_by_name(options.key_set);  // validates, throws with the known names
  Assert(options.run_count >= 2, "--runs must be at least 2 (one warm-up plus one measured run)");

  return options;
}

// ---------------------------------------------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------------------------------------------
struct RunSample {
  size_t total_us = 0;
  std::array<size_t, PHASE_COUNT> phase_us{};
};

static std::array<size_t, PHASE_COUNT> extract_phases(const AbstractOperator& sort_operator) {
  auto phases = std::array<size_t, PHASE_COUNT>{};

  const auto* performance_data =
      dynamic_cast<const OperatorPerformanceData<Sort::OperatorSteps>*>(sort_operator.performance_data.get());
  if (!performance_data) {
    return phases;  // all zeros
  }

  for (auto index = size_t{0}; index < PHASE_COUNT; ++index) {
    const auto runtime = performance_data->get_step_runtime(PHASES[index].step);
    phases[index] = static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>(runtime).count());
  }
  return phases;
}

static void silent_tpch_generation(const float scale_factor, const std::shared_ptr<BenchmarkConfig>& config) {
  auto* buffer = std::cout.rdbuf();
  std::cout.rdbuf(nullptr);
  TPCHTableGenerator(scale_factor, ClusteringConfiguration::None, config).generate_and_store();
  std::cout.rdbuf(buffer);
}

// Keeps the sort key plus `payload_columns`, prunes everything else. An empty payload list keeps ALL columns.
// Returns the executed GetTable, the sort definitions resolved against its (pruned) output, and the payload width.
static std::tuple<std::shared_ptr<GetTable>, std::vector<SortColumnDefinition>, size_t> setup_get_table(
    const std::string& table_name, const std::vector<std::string>& key_columns,
    const std::vector<std::string>& payload_columns) {
  const auto table = Hyrise::get().storage_manager.get_table(table_name);
  const auto column_count = table->column_count();

  auto kept = std::vector<ColumnID>{};
  if (payload_columns.empty()) {
    kept.resize(column_count);
    std::iota(kept.begin(), kept.end(), ColumnID{0});
  } else {
    for (const auto& column_name : key_columns) {
      kept.emplace_back(table->column_id_by_name(column_name));
    }
    for (const auto& column_name : payload_columns) {
      kept.emplace_back(table->column_id_by_name(column_name));
    }
    std::ranges::sort(kept);
    kept.erase(std::ranges::unique(kept).begin(), kept.end());
  }

  auto all_columns = std::vector<ColumnID>(column_count);
  std::iota(all_columns.begin(), all_columns.end(), ColumnID{0});
  auto pruned = std::vector<ColumnID>{};
  std::ranges::set_difference(all_columns, kept, std::back_inserter(pruned));

  const auto get_table = std::make_shared<GetTable>(table_name, std::vector<ChunkID>{}, pruned);
  get_table->never_clear_output();
  get_table->execute();

  auto sort_definitions = std::vector<SortColumnDefinition>{};
  sort_definitions.reserve(key_columns.size());
  for (const auto& column_name : key_columns) {
    sort_definitions.emplace_back(get_table->get_output()->column_id_by_name(column_name));
  }

  const auto payload_width = static_cast<size_t>(get_table->get_output()->column_count()) - key_columns.size();
  return {get_table, sort_definitions, payload_width};
}

static std::vector<RunSample> measure_operator(const std::shared_ptr<AbstractOperator>& input,
                                               const std::vector<SortColumnDefinition>& sort_definitions,
                                               const size_t run_count) {
  auto samples = std::vector<RunSample>{};
  samples.reserve(run_count - 1);

  for (auto run_id = size_t{0}; run_id < run_count; ++run_id) {
    const auto start = std::chrono::steady_clock::now();
    auto sort = std::make_shared<Sort>(input, sort_definitions, Chunk::DEFAULT_SIZE, Sort::ForceMaterialization::Yes);
    sort->execute();
    const auto end = std::chrono::steady_clock::now();

    if (run_id == 0) {
      continue;  // warm-up
    }

    auto sample = RunSample{};
    sample.total_us = static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    sample.phase_us = extract_phases(*sort);
    samples.push_back(sample);
  }

  return samples;
}

static std::shared_ptr<const AbstractOperator> find_sort_operator(const std::shared_ptr<const AbstractOperator>& op) {
  if (!op) {
    return nullptr;
  }
  if (op->name() == "Sort") {
    return op;
  }
  if (const auto found = find_sort_operator(op->left_input())) {
    return found;
  }
  return find_sort_operator(op->right_input());
}

struct SqlResult {
  std::vector<RunSample> samples;
  size_t row_count = 0;
  // False means the query plan's Sort returned a reference table, i.e. the LQPTranslator patch documented at the top
  // of this file is NOT in effect on this branch and the payload was never gathered.
  bool materialized = false;
};

static SqlResult measure_sql(const std::string& sql, const size_t run_count) {
  auto result_out = SqlResult{};
  result_out.samples.reserve(run_count - 1);

  for (auto run_id = size_t{0}; run_id < run_count; ++run_id) {
    auto pipeline = SQLPipelineBuilder{sql}.create_pipeline();

    const auto start = std::chrono::steady_clock::now();
    const auto result = pipeline.get_result_table();
    const auto end = std::chrono::steady_clock::now();

    Assert(result.first == SQLPipelineStatus::Success, "SQL pipeline did not succeed: " + sql);
    result_out.row_count = result.second->row_count();
    result_out.materialized = result.second->type() == TableType::Data;

    if (run_id == 0) {
      continue;  // warm-up
    }

    auto sample = RunSample{};
    sample.total_us = static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    const auto& physical_plans = pipeline.get_physical_plans();
    Assert(!physical_plans.empty(), "No physical plan for: " + sql);
    if (const auto sort_operator = find_sort_operator(physical_plans.back())) {
      sample.phase_us = extract_phases(*sort_operator);
    }

    result_out.samples.push_back(sample);
  }

  return result_out;
}

// ---------------------------------------------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------------------------------------------
static void write_header_if_needed(const std::string& path) {
  if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
    return;
  }

  auto out_file = std::ofstream{path};
  out_file << "ROUTINE,BRANCH,COMMIT,MACHINE,HARNESS,MATERIALIZED,TABLE,KEY_SET,KEY_COLS,SORT_KEY,SCALE,"
              "ENCODING,PAYLOAD_COLS,ROW_COUNT,RUN_ID,TOTAL_US";
  for (const auto& phase : PHASES) {
    out_file << ',' << phase.csv_column;
  }
  out_file << '\n';
}

static void append_samples(const Options& options, const std::vector<std::string>& key_columns,
                           const std::string& harness, const bool materialized, const size_t payload_columns,
                           const size_t row_count, const std::vector<RunSample>& samples) {
  auto out_file = std::ofstream(options.out_path, std::ios::app);
  const auto key_columns_joined = join(key_columns, ",");

  for (auto run_id = size_t{0}; run_id < samples.size(); ++run_id) {
    const auto& sample = samples[run_id];
    out_file << std::format(R"("{}","{}","{}","{}","{}",{},"{}","{}",{},"{}",{},"{}",{},{},{},{})", options.routine,
                            options.branch, options.commit, options.machine, harness, materialized ? 1 : 0, TABLE_NAME,
                            options.key_set, key_columns.size(), key_columns_joined, options.scale_factor,
                            ENCODING_TAG, payload_columns, row_count, run_id, sample.total_us);
    for (const auto phase_us : sample.phase_us) {
      out_file << ',' << phase_us;
    }
    out_file << '\n';
  }
}

// ---------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  const auto options = parse_options(argc, argv);

  const auto& key_set = key_set_by_name(options.key_set);
  const auto sql_query = std::format("SELECT * FROM {} ORDER BY {};", TABLE_NAME, join(key_set.columns, ", "));

  std::cout << std::format("routine={} sf={} key={} ({}) encoding={} runs={} (1 warm-up) -> {}\n", options.routine,
                           options.scale_factor, options.key_set, join(key_set.columns, ", "), ENCODING_TAG,
                           options.run_count, options.out_path);

  write_header_if_needed(options.out_path);

  const auto node_queue_scheduler = std::make_shared<NodeQueueScheduler>();
  Hyrise::get().set_scheduler(node_queue_scheduler);

  auto benchmark_config = std::make_shared<BenchmarkConfig>();
  benchmark_config->cache_binary_tables = true;
  silent_tpch_generation(options.scale_factor, benchmark_config);

  // Depends on the generated table, so it cannot be a constant: which columns are payload depends on the key set.
  const auto payload_order = payload_column_order(TABLE_NAME, key_set.columns);

  for (const auto& harness : options.harnesses) {
    if (harness == "operator") {
      for (const auto& width_tag : options.payload_widths) {
        auto payload_columns = std::vector<std::string>{};
        if (width_tag != "all") {
          const auto width = static_cast<size_t>(std::stoul(width_tag));
          Assert(width <= payload_order.size(), "Payload width exceeds the number of non-key columns");
          payload_columns.assign(payload_order.begin(), payload_order.begin() + static_cast<std::ptrdiff_t>(width));
        }

        const auto setup = setup_get_table(TABLE_NAME, key_set.columns, payload_columns);
        const auto get_table = std::get<0>(setup);
        const auto sort_definitions = std::get<1>(setup);
        const auto payload_width = std::get<2>(setup);

        std::cout << std::format("  operator: payload={} ({} columns)\n", width_tag, payload_width) << std::flush;
        const auto samples = measure_operator(get_table, sort_definitions, options.run_count);
        // ForceMaterialization::Yes, so the payload is always gathered in this harness.
        append_samples(options, key_set.columns, "operator", /*materialized*/ true, payload_width,
                       get_table->get_output()->row_count(), samples);
      }
    } else if (harness == "sql") {
      std::cout << "  sql: " << sql_query << '\n' << std::flush;
      const auto sql_result = measure_sql(sql_query, options.run_count);

      if (!sql_result.materialized) {
        std::cerr << "  WARNING: the query plan's Sort returned a reference table -- the LQPTranslator "
                     "ForceMaterialization patch is NOT in effect on this branch. WRITE_OUT_US will be ~0 and these "
                     "rows are not comparable to the blog. See the note at the top of this file.\n"
                  << std::flush;
      }

      // The query is always SELECT *, so payload width is fixed rather than swept here.
      append_samples(options, key_set.columns, "sql", sql_result.materialized, payload_order.size(),
                     sql_result.row_count, sql_result.samples);
    } else {
      Fail("Unknown harness: " + harness);
    }
  }

  node_queue_scheduler->finish();
  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  std::cout << "appended to " << options.out_path << '\n';
  return 0;
}
