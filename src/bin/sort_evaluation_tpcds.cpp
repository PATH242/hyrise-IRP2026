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
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "benchmark_config.hpp"
#include "benchmark_table_encoder.hpp"
#include "encoding_config.hpp"
#include "hyrise.hpp"
#include "import_export/binary/binary_parser.hpp"
#include "import_export/binary/binary_writer.hpp"
#include "operators/abstract_operator.hpp"
#include "operators/operator_performance_data.hpp"
#include "operators/sort.hpp"
#include "operators/table_wrapper.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "storage/chunk.hpp"
#include "storage/table.hpp"
#include "storage/table_column_definition.hpp"
#include "tpcds/tpcds_table_generator.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

// =====================================================================================================================
// Multi-key-column sort evaluation on TPC-DS, reproducing Section VII benchmarks 2 and 3 of
//
//   Kuiper, Raasveldt, Mühleisen. "These Rows Are Made for Sorting and That's Just What We'll Do." ICDE 2023.
//
// That paper sweeps the NUMBER OF KEY COLUMNS with the payload held at one column -- the mirror image of the DuckDB
// "Sorting Again" blog experiment (payload swept, key held at one), which sort_evaluation.cpp covers on TPC-H.
// Together the two files cover both axes.
//
// Paper spec, matched verbatim:
//   catalog_sales   keys cs_warehouse_sk, cs_ship_mode_sk, cs_promo_sk, cs_quantity  (swept 1,2,3,4)
//                   payload cs_item_sk                 SF 10 (14.4M rows) and 100 (144M rows)   [their Fig. 11]
//   customer        keys c_birth_year, c_birth_month, c_birth_day        (integer variant)
//                   keys c_last_name, c_first_name                       (string variant)
//                   payload c_customer_sk              SF 100 (2M rows) and 300 (5M rows)       [their Fig. 12]
//   5 measured repetitions, median.
//
// Why those key columns: all four catalog_sales keys are low-cardinality (a handful of warehouses, ~20 ship modes,
// hundreds of promos, quantity 1-100), so ties are everywhere and each added column does real tie-breaking work. A
// unique leading column would make columns 2-4 unreachable and the sweep flat.
//
// DIFFERENCES FROM THE PAPER, deliberate:
//   * Operator-only. The paper wraps each sort in `SELECT COUNT(*) FROM (... ORDER BY ... OFFSET 1)` to stop the
//     optimizer eliding the sort; driving Sort directly makes that unnecessary, and gives the per-phase breakdown.
//   * Hyrise-internal comparison. The paper compares five systems; absolute seconds are not comparable across
//     machines or engines, so cite its method, not its numbers.
//   * Sub-sort is absent here because it is absent from this evaluation. Note the paper measures sub-sort ONLY
//     single-threaded (Section IV micro-benchmarks); these runs are multi-threaded, so no published sub-sort
//     baseline exists for this setting either way.
//
// MEMORY NOTE: generation builds the FULL table (catalog_sales has 34 columns) before it is projected down to the
// five columns an experiment needs. Peak memory during generation is therefore much higher than steady state. Use
// --generate-only once, unbound, to populate the cache; measured runs then load only the small projected table and
// can be bound to one NUMA node.
// =====================================================================================================================

// ---------------------------------------------------------------------------------------------------------------
// PHASE TABLE -- MUST stay identical to the one in sort_evaluation.cpp, or the two benchmarks' phase columns stop
// being comparable. If you add a phase there, add it here in the same turn.
// ---------------------------------------------------------------------------------------------------------------
struct PhaseSpec {
  std::string_view csv_column;
  Sort::OperatorSteps step;
};

constexpr auto PHASES = std::array{
    PhaseSpec{"MATERIALIZE_US", Sort::OperatorSteps::KeyMaterialization},
    PhaseSpec{"SORT_US", Sort::OperatorSteps::Sort},
    PhaseSpec{"MERGE_US", Sort::OperatorSteps::MergeTime},
    PhaseSpec{"MERGE_PATH_US", Sort::OperatorSteps::MergePath},
    PhaseSpec{"WRITE_OUT_US", Sort::OperatorSteps::ResultMaterialization},
};
constexpr auto PHASE_COUNT = PHASES.size();

// Encoding is not an axis: tables are encoded with a default EncodingConfig, i.e. Hyrise's "Automatic" per-column
// selection (Unencoded for unique columns, FrameOfReference for Int, Dictionary otherwise). Same as the TPC-H side.
constexpr auto ENCODING_TAG = "automatic";
constexpr auto BENCHMARK_TAG = "tpcds";

// ---------------------------------------------------------------------------------------------------------------
// EXPERIMENTS -- add one here and it is immediately selectable with --experiment.
// `default_key_counts` mirrors the paper: catalog_sales is a 1..4 sweep (Fig. 11); the customer variants are single
// points at their full key width (Fig. 12). Override either with --keys.
// ---------------------------------------------------------------------------------------------------------------
struct Experiment {
  std::string_view name;
  std::string_view table;
  std::vector<std::string> key_columns;
  std::string payload_column;
  std::vector<size_t> default_key_counts;
  std::vector<uint32_t> default_scale_factors;
};

const auto EXPERIMENTS = std::vector<Experiment>{
    {"catalog_sales",
     "catalog_sales",
     {"cs_warehouse_sk", "cs_ship_mode_sk", "cs_promo_sk", "cs_quantity"},
     "cs_item_sk",
     {1, 2, 3, 4},
     {10}},  // add 100 once node memory is known to be sufficient -- see the memory note above
    // The paper runs these two at SF 100 and 300. Both are enabled here at SF 10 only; add 100 and 300 to either
    // list when the node has the memory for them. `customer` is small at every one of these -- roughly 500k rows
    // at SF 10, 2M at 100, 5M at 300 -- so none of them is a memory problem the way catalog_sales SF 100 is.
    //
    // Expect SF 10 to be a fast sort: half a million rows of three int columns is a few milliseconds, which is
    // close enough to the operator's own fixed overhead that the phase split will look noisier than at SF 100.
    // Fine for a smoke test of the two key sets; read the headline numbers off the larger scale factors.
    {"customer_int", "customer", {"c_birth_year", "c_birth_month", "c_birth_day"}, "c_customer_sk", {3}, {10}},
    {"customer_str", "customer", {"c_last_name", "c_first_name"}, "c_customer_sk", {2}, {10}},
};

static const Experiment& experiment_by_name(const std::string& name) {
  for (const auto& experiment : EXPERIMENTS) {
    if (experiment.name == name) {
      return experiment;
    }
  }
  auto known = std::string{};
  for (const auto& experiment : EXPERIMENTS) {
    known += std::string{experiment.name} + " ";
  }
  Fail("Unknown --experiment '" + name + "'. Known experiments: " + known);
}

static std::string join(const std::vector<std::string>& parts, const std::string& separator) {
  auto out = std::string{};
  for (auto index = size_t{0}; index < parts.size(); ++index) {
    out += (index ? separator : "") + parts[index];
  }
  return out;
}

static std::vector<std::string> split(const std::string& value, const char delimiter) {
  auto parts = std::vector<std::string>{};
  auto start = size_t{0};
  while (start <= value.size()) {
    const auto end = value.find(delimiter, start);
    const auto piece = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!piece.empty()) {
      parts.push_back(piece);
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return parts;
}

// ---------------------------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------------------------
struct Options {
  std::string routine;
  std::string branch;
  std::string commit;
  std::string machine;
  std::string experiment = "catalog_sales";
  std::string out_path = "SORT_RESULTS.csv";
  std::string cache_dir = "tpcds_sort_cache";
  std::vector<size_t> key_counts;       // empty -> the experiment's default
  uint32_t scale_factor = 0;            // 0 -> loop the experiment's defaults
  size_t run_count = 6;                 // first run is a discarded warm-up: 5 measured, as in the paper
  bool generate_only = false;
};

static void print_usage() {
  std::cout << "usage: sort_evaluation_tpcds --routine <tag> [options]\n"
            << "  --routine    <tag>            required; identifies the sort implementation in the CSV\n"
            << "  --experiment <name>           catalog_sales | customer_int | customer_str (default catalog_sales)\n"
            << "  --sf         <int>            scale factor; omit to use the experiment's paper defaults\n"
            << "  --keys       <1,2,3,4>        how many leading key columns to sort by; omit for the default\n"
            << "  --runs       <n>              total runs; the first is a discarded warm-up (default 6)\n"
            << "  --out        <path>           CSV to append to (default SORT_RESULTS.csv)\n"
            << "  --cache-dir  <path>           projected-table cache (default tpcds_sort_cache)\n"
            << "  --generate-only               build and cache the projected table, then exit\n"
            << "  --branch/--commit/--machine <s>  metadata stamped into every row\n";
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
    } else if (flag == "--experiment") {
      options.experiment = next();
    } else if (flag == "--sf") {
      options.scale_factor = static_cast<uint32_t>(std::stoul(next()));
    } else if (flag == "--keys") {
      for (const auto& piece : split(next(), ',')) {
        options.key_counts.push_back(static_cast<size_t>(std::stoul(piece)));
      }
    } else if (flag == "--runs") {
      options.run_count = static_cast<size_t>(std::stoul(next()));
    } else if (flag == "--out") {
      options.out_path = next();
    } else if (flag == "--cache-dir") {
      options.cache_dir = next();
    } else if (flag == "--generate-only") {
      options.generate_only = true;
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

  Assert(!options.routine.empty() || options.generate_only, "--routine is required");
  Assert(options.run_count >= 2, "--runs must be at least 2 (one warm-up plus one measured run)");

  const auto& experiment = experiment_by_name(options.experiment);
  for (const auto count : options.key_counts) {
    Assert(count >= 1 && count <= experiment.key_columns.size(),
           std::format("--keys value {} is outside 1..{} for experiment '{}'", count, experiment.key_columns.size(),
                       options.experiment));
  }

  return options;
}

// ---------------------------------------------------------------------------------------------------------------
// Table preparation: generate -> project to the columns this experiment needs -> encode -> cache
//
// Projecting before caching is what makes the large scale factors tractable: catalog_sales has 34 columns and the
// experiment needs five, so the cached table is roughly a sixth of the generated one.
// ---------------------------------------------------------------------------------------------------------------
static std::shared_ptr<Table> project_columns(const std::shared_ptr<const Table>& source,
                                              const std::vector<std::string>& column_names) {
  auto column_ids = std::vector<ColumnID>{};
  auto definitions = TableColumnDefinitions{};
  for (const auto& column_name : column_names) {
    const auto column_id = source->column_id_by_name(column_name);
    column_ids.push_back(column_id);
    definitions.emplace_back(source->column_name(column_id), source->column_data_type(column_id),
                             source->column_is_nullable(column_id));
  }

  auto projected = std::make_shared<Table>(definitions, TableType::Data, source->target_chunk_size(), UseMvcc::No);

  const auto chunk_count = source->chunk_count();
  for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
    const auto chunk = source->get_chunk(chunk_id);
    if (!chunk) {
      continue;
    }
    auto segments = Segments{};
    segments.reserve(column_ids.size());
    for (const auto column_id : column_ids) {
      segments.emplace_back(chunk->get_segment(column_id));
    }
    projected->append_chunk(segments);
    projected->last_chunk()->set_immutable();
  }

  return projected;
}

// Hyrise's init_tpcds_tools hardcodes this as a RELATIVE path, so dsdgen resolves it against the process working
// directory. Checking it here turns a bare "open of distributions failed" into something actionable.
constexpr auto TPCDS_DISTRIBUTIONS = "resources/benchmark/tpcds/tpcds.idx";

static void assert_distributions_reachable() {
  if (std::filesystem::exists(TPCDS_DISTRIBUTIONS)) {
    return;
  }
  Fail(std::format(
      "dsdgen needs '{}' relative to the working directory, which is currently '{}'. Hyrise hardcodes that relative "
      "path in init_tpcds_tools, so either run from the Hyrise repo root or symlink its 'resources' directory into "
      "the working directory (run_evaluation.sh does the latter for you). Note this is only needed to GENERATE: once "
      "the cache exists, runs work from anywhere.",
      TPCDS_DISTRIBUTIONS, std::filesystem::current_path().string()));
}

static std::shared_ptr<Table> generate_source_table(const std::string& table_name, const uint32_t scale_factor) {
  assert_distributions_reachable();

  // rng_seed defaults to dsdgen's own 19620718, so generation is deterministic across routines and machines.
  const auto generator = TPCDSTableGenerator{scale_factor, Chunk::DEFAULT_SIZE};

  if (table_name == "customer") {
    return generator.generate_customer();
  }
  if (table_name == "catalog_sales") {
    // Returns {sales, returns}; the returns table is generated alongside and discarded here.
    return generator.generate_catalog_sales_and_returns().first;
  }
  Fail("No generator wired up for table '" + table_name + "'");
}

static std::shared_ptr<Table> prepare_table(const Experiment& experiment, const uint32_t scale_factor,
                                            const std::string& cache_dir, const bool force_regenerate) {
  auto columns = experiment.key_columns;
  columns.push_back(experiment.payload_column);

  // The experiment name is in the path because two experiments on the same table need different projections.
  const auto cache_path = std::format("{}/{}-sf{}-{}.bin", cache_dir, experiment.table, scale_factor, experiment.name);
  // cache_dir may be absolute (run_evaluation.sh passes it that way) so the cache never depends on the CWD that
  // dsdgen forces us into.

  if (!force_regenerate && std::filesystem::exists(cache_path)) {
    std::cout << std::format("  loading cached table {}\n", cache_path) << std::flush;
    return BinaryParser::parse(cache_path);
  }

  std::cout << std::format("  generating {} at SF {} (all columns, then projecting to {})\n", experiment.table,
                           scale_factor, join(columns, ", "))
            << std::flush;

  auto projected = std::shared_ptr<Table>{};
  {
    const auto source = generate_source_table(std::string{experiment.table}, scale_factor);
    std::cout << std::format("  generated {} rows, {:.1f} GB (all columns)\n", source->row_count(),
                             static_cast<double>(source->memory_usage(MemoryUsageCalculationMode::Sampled)) / 1e9)
              << std::flush;
    projected = project_columns(source, columns);
  }
  // `source` is gone here. `projected` shares its segments, so only the unreferenced columns were freed -- which is
  // the point: it drops the high-water mark before encoding, rather than holding both tables at once.

  // Encode exactly as the TPC-H side does: default EncodingConfig == Hyrise's "Automatic" per-column choice.
  BenchmarkTableEncoder::encode(std::string{experiment.table}, projected, EncodingConfig{});
  std::cout << std::format("  projected + encoded to {} columns, {:.2f} GB\n", columns.size(),
                           static_cast<double>(projected->memory_usage(MemoryUsageCalculationMode::Sampled)) / 1e9)
            << std::flush;

  std::filesystem::create_directories(cache_dir);
  BinaryWriter::write(*projected, cache_path);
  std::cout << std::format("  cached {} -> {}\n", experiment.table, cache_path)
            << std::flush;

  return projected;
}

// ---------------------------------------------------------------------------------------------------------------
// Measurement -- identical in shape to sort_evaluation.cpp
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
    return phases;
  }

  for (auto index = size_t{0}; index < PHASE_COUNT; ++index) {
    const auto runtime = performance_data->get_step_runtime(PHASES[index].step);
    phases[index] = static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>(runtime).count());
  }
  return phases;
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

// ---------------------------------------------------------------------------------------------------------------
// CSV output -- same schema as sort_evaluation.cpp plus a BENCHMARK column, so both benchmarks merge and plot
// together. ROUTINE stays first: merge_results.py identifies a results file by that header prefix.
// ---------------------------------------------------------------------------------------------------------------
static void write_header_if_needed(const std::string& path) {
  if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
    return;
  }

  auto out_file = std::ofstream{path};
  out_file << "ROUTINE,BRANCH,COMMIT,MACHINE,BENCHMARK,HARNESS,MATERIALIZED,TABLE,KEY_SET,KEY_COLS,SORT_KEY,SCALE,"
              "ENCODING,PAYLOAD_COLS,ROW_COUNT,RUN_ID,TOTAL_US";
  for (const auto& phase : PHASES) {
    out_file << ',' << phase.csv_column;
  }
  out_file << '\n';
}

static void append_samples(const Options& options, const Experiment& experiment,
                           const std::vector<std::string>& key_columns, const uint32_t scale_factor,
                           const size_t row_count, const std::vector<RunSample>& samples) {
  auto out_file = std::ofstream(options.out_path, std::ios::app);
  const auto key_columns_joined = join(key_columns, ",");

  for (auto run_id = size_t{0}; run_id < samples.size(); ++run_id) {
    const auto& sample = samples[run_id];
    out_file << std::format(R"("{}","{}","{}","{}","{}","{}",{},"{}","{}",{},"{}",{},"{}",{},{},{},{})",
                            options.routine, options.branch, options.commit, options.machine, BENCHMARK_TAG,
                            "operator", 1, experiment.table, experiment.name, key_columns.size(), key_columns_joined,
                            scale_factor, ENCODING_TAG, 1, row_count, run_id, sample.total_us);
    for (const auto phase_us : sample.phase_us) {
      out_file << ',' << phase_us;
    }
    out_file << '\n';
  }
}

// ---------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  const auto options = parse_options(argc, argv);
  const auto& experiment = experiment_by_name(options.experiment);

  const auto key_counts = options.key_counts.empty() ? experiment.default_key_counts : options.key_counts;
  const auto scale_factors = options.scale_factor != 0 ? std::vector<uint32_t>{options.scale_factor}
                                                       : experiment.default_scale_factors;

  std::cout << std::format("benchmark=tpcds experiment={} table={} keys={} payload={} runs={} (1 warm-up)\n",
                           options.experiment, experiment.table, join(experiment.key_columns, ", "),
                           experiment.payload_column, options.run_count);

  const auto node_queue_scheduler = std::make_shared<NodeQueueScheduler>();
  Hyrise::get().set_scheduler(node_queue_scheduler);

  for (const auto scale_factor : scale_factors) {
    std::cout << std::format("scale factor {}\n", scale_factor) << std::flush;
    const auto table = prepare_table(experiment, scale_factor, options.cache_dir, false);

    if (options.generate_only) {
      continue;
    }

    write_header_if_needed(options.out_path);

    const auto table_wrapper = std::make_shared<TableWrapper>(table);
    table_wrapper->never_clear_output();
    table_wrapper->execute();

    for (const auto key_count : key_counts) {
      Assert(key_count <= experiment.key_columns.size(), "key count exceeds this experiment's key columns");
      const auto key_columns =
          std::vector<std::string>(experiment.key_columns.begin(),
                                   experiment.key_columns.begin() + static_cast<std::ptrdiff_t>(key_count));

      auto sort_definitions = std::vector<SortColumnDefinition>{};
      sort_definitions.reserve(key_columns.size());
      for (const auto& column_name : key_columns) {
        sort_definitions.emplace_back(table->column_id_by_name(column_name));
      }

      std::cout << std::format("  {} key column(s): {}\n", key_count, join(key_columns, ", ")) << std::flush;
      const auto samples = measure_operator(table_wrapper, sort_definitions, options.run_count);
      append_samples(options, experiment, key_columns, scale_factor, table->row_count(), samples);
    }
  }

  node_queue_scheduler->finish();
  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  if (options.generate_only) {
    std::cout << "generation complete; cache is populated\n";
  } else {
    std::cout << "appended to " << options.out_path << '\n';
  }
  return 0;
}
