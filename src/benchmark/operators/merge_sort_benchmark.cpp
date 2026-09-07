#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyrise.hpp"
#include "micro_benchmark_utils.hpp"
#include "operators/operator_performance_data.hpp"
#include "operators/sort.hpp"
#include "operators/table_wrapper.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "storage/chunk.hpp"
#include "storage/segment_iterate.hpp"
#include "storage/table.hpp"
#include "storage/value_segment.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

// Micro-benchmark for the Sort operator's run-generation and merge strategies (built per configuration via the
// compile-time toggles: USE_RADIX_SORT, and the merge implementation in the chosen sort.cpp variant).
//
// The matrix is organised as four experiments. They are not separate run campaigns -- they are overlapping slices of
// one matrix, and a single invocation of this binary produces all four. Each registered point carries in_e1..in_e4
// counters so the JSON can be sliced per experiment without re-parsing benchmark names.
//
//   E1 -- single-threaded, integer keys
//         int32 / {random, ascending, dup128, uniform, zipf099} / {1M, 10M, 50M} / cols1 / mt0
//         Widest data sweep. random-uniform-dup128 is the cardinality ladder (~distinct, 2^20, 128 distinct);
//         uniform and zipf099 share one domain and differ in skew alone; ascending is the sorted input.
//
//   E2 -- multi-threaded, integer keys
//         int32 / {random, uniform, zipf099} / {1M, 10M, 50M} / cols1 / mt0 and mt1, paired
//         Skew is the axis that matters: zipf099 is where partition-based routines should show bucket imbalance.
//
//   E3 -- wide and composite keys
//         int32 / {random, zipf099} / 10M / cols {1, 5} / mt0 and mt1
//         zipf099 x cols5 is the cell that matters: ties in c0 push comparison work into the tie-breaking columns.
//
//   E4 -- string keys
//         {int32, str_sso, str_var} / {random, zipf099} / {1M, 10M, 50M} / cols1 / mt0 and mt1
//         int32 is included so the ns/tuple curves are directly comparable across key types -- that is the plot this
//         experiment turns on. 50M runs mt1 only: a single-threaded 50M str_var sort costs minutes and the
//         comparability plot only needs one threading mode.
//
// Within each experiment the design is one-factor-at-a-time around the baseline (10M rows, 1 key column, int32,
// random), so cost is additive in the axes rather than multiplicative. 30 unique points; overlapping points are
// registered once and tagged with every experiment they belong to.
//
// Counters: the four Sort::OperatorSteps runtimes (materialize_s, run_gen_s, merge_s, write_out_s), the configuration
// echoed numerically (rows, cols, mt, key_type, distribution, theta), and the experiment membership flags. Run once
// per build config with --benchmark_out=<label>.json.

namespace {

using namespace hyrise;

constexpr auto SEED = uint32_t{42};

// Baseline for the OFAT sweep. Every axis below varies exactly one field of this point.
constexpr auto BASELINE_ROWS = size_t{10'000'000};
constexpr auto BASELINE_COLS = size_t{1};
constexpr auto WIDE_COLS = size_t{5};

// Row-count ladder, uniform across every key type so the ns/tuple curves are directly comparable at all three points.
// 50M rather than 100M at the top: 100M variable-length keys are roughly 4-6 GB of string objects plus ~100M
// individual heap allocations, and holding the int32 ladder to the same ceiling keeps the curves on one scale.
constexpr auto SMALL_ROWS = size_t{1'000'000};
constexpr auto LARGE_ROWS = size_t{50'000'000};

// Iterations per reported measurement, reduced for the large points (see iterations_for). Repetitions (for
// mean/median/stddev) come from the command line via --benchmark_repetitions=N so that changing the statistical
// budget does not require a rebuild.
constexpr auto BENCHMARK_ITERATIONS = int64_t{3};
constexpr auto LARGE_BENCHMARK_ITERATIONS = int64_t{1};

// Smoke test: after one untimed sort, assert the output is non-decreasing in the first sort column. Catches a broken
// merge before it produces publishable numbers. Skipped above BASELINE_ROWS so it does not cost a full extra sort of
// the 50M tables.
constexpr auto VERIFY_OUTPUT = true;

enum class Distribution : uint8_t { Random, Ascending, Dup128, Uniform, Zipf090, Zipf099 };
enum class KeyType : uint8_t { Int32, StringSso, StringVar };

// Zipf090 is retained in the generator but is not part of the matrix: uniform (theta = 0) and zipf099 (theta = 0.99)
// already bracket the skew axis, and a third point costs a full sweep per build. Add it back to E1/E2 if the
// uniform -> zipf099 gap turns out to be large enough to be worth resolving as a curve.

// ---------------------------------------------------------------------------------------------------------------
// Value domain
// ---------------------------------------------------------------------------------------------------------------
//
// All distributions emit a *rank* in [0, domain). The rank is then mapped to the concrete key type by a strictly
// order-preserving encoding, so that a given distribution produces the same value-frequency histogram and the same
// relative order for every key type. Differences between Int32 and the string variants are therefore attributable to
// the encoding, not to a different input permutation.

constexpr auto DOMAIN_BITS = uint64_t{20};
constexpr auto DOMAIN = uint64_t{1} << DOMAIN_BITS;  // 1,048,576 distinct values for the Uniform/Zipf family.
constexpr auto DOMAIN_MASK = DOMAIN - 1;

uint64_t mix64(uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

// Bijection on [0, DOMAIN). Zipf assigns the highest probability to rank 0, so feeding ranks through directly would
// make the hot values also be the *smallest* values -- skew in frequency would come bundled with skew in value
// locality, and a sorted result would carry a huge dense prefix. Scrambling keeps the frequency histogram exactly
// Zipfian while spreading the hot values across the domain. Both `x ^ (x >> k)` and multiplication by an odd constant
// are invertible modulo 2^DOMAIN_BITS, so no two ranks collide and the histogram is preserved exactly.
//
// Set SCRAMBLE_ZIPF_RANKS to false to measure the "skewed and clustered" variant instead.
constexpr auto SCRAMBLE_ZIPF_RANKS = true;

uint64_t scramble(uint64_t value) {
  if constexpr (!SCRAMBLE_ZIPF_RANKS) {
    return value & DOMAIN_MASK;
  }
  value = (value ^ (value >> 11U)) & DOMAIN_MASK;
  value = (value * 0x2545F49ULL) & DOMAIN_MASK;
  value = (value ^ (value >> 7U)) & DOMAIN_MASK;
  value = (value * 0x9E3779BULL) & DOMAIN_MASK;
  return value & DOMAIN_MASK;
}

// ---------------------------------------------------------------------------------------------------------------
// Zipf generator (Gray et al., "Quickly Generating Billion-Record Synthetic Databases", SIGMOD 1994)
// ---------------------------------------------------------------------------------------------------------------
//
// P(rank = i) is proportional to 1 / (i + 1)^theta over i in [0, domain). The zeta sum is computed once per (domain,
// theta); each draw is then O(1) via the closed-form inverse from the paper -- important because an inverse-CDF table
// lookup would cost a binary search over a multi-megabyte table for every one of up to 10^8 generated values.
//
// theta = 0 degenerates to the exact uniform distribution over the same domain, which is what Distribution::Uniform
// uses: it is the controlled theta = 0 point of the skew series, so Uniform and Zipf099 differ in skew alone and not
// in cardinality. (Distribution::Random, in contrast, draws over the full int32 range and is effectively
// distinct-key; it is kept for continuity with earlier measurements and is the OFAT baseline.)
class ZipfGenerator {
 public:
  ZipfGenerator(const uint64_t domain, const double theta)
      : _domain{domain},
        _theta{theta},
        _zetan{_zeta(domain, theta)},
        _zeta2{_zeta(2, theta)},
        _alpha{1.0 / (1.0 - theta)},
        _eta{(1.0 - std::pow(2.0 / static_cast<double>(domain), 1.0 - theta)) / (1.0 - (_zeta2 / _zetan))} {
    Assert(theta >= 0.0 && theta < 1.0, "ZipfGenerator requires 0 <= theta < 1.");
  }

  uint64_t draw(std::mt19937_64& rng) const {
    auto uniform = std::uniform_real_distribution<double>{0.0, 1.0};
    const auto random_value = uniform(rng);
    const auto scaled = random_value * _zetan;

    if (scaled < 1.0) {
      return 0;
    }
    if (scaled < 1.0 + std::pow(0.5, _theta)) {
      return 1;
    }

    const auto rank = static_cast<uint64_t>(static_cast<double>(_domain) *
                                            std::pow((_eta * random_value) - _eta + 1.0, _alpha));
    return std::min(rank, _domain - 1);
  }

 private:
  static double _zeta(const uint64_t count, const double theta) {
    auto sum = double{0};
    for (auto index = uint64_t{1}; index <= count; ++index) {
      sum += 1.0 / std::pow(static_cast<double>(index), theta);
    }
    return sum;
  }

  uint64_t _domain;
  double _theta;
  double _zetan;
  double _zeta2;
  double _alpha;
  double _eta;
};

double theta_of(const Distribution distribution) {
  switch (distribution) {
    case Distribution::Uniform:
      return 0.0;
    case Distribution::Zipf090:
      return 0.9;
    case Distribution::Zipf099:
      return 0.99;
    default:
      return -1.0;  // Not a member of the skew family.
  }
}

// Cached so that the O(domain) zeta sum is paid once per theta and not once per key column per table.
std::shared_ptr<const ZipfGenerator> zipf_generator_for(const Distribution distribution) {
  const auto theta = theta_of(distribution);
  if (theta < 0.0) {
    return nullptr;
  }

  static auto cache = std::vector<std::pair<double, std::shared_ptr<const ZipfGenerator>>>{};
  for (const auto& [cached_theta, generator] : cache) {
    if (cached_theta == theta) {
      return generator;
    }
  }
  auto generator = std::make_shared<const ZipfGenerator>(DOMAIN, theta);
  cache.emplace_back(theta, generator);
  return generator;
}

// ---------------------------------------------------------------------------------------------------------------
// Rank -> key encoding
// ---------------------------------------------------------------------------------------------------------------

// ASCII-ascending alphabet: '0' < '9' < 'A' < 'Z' < 'a' < 'z'. A fixed-width base-62 numeral over this alphabet is
// order-preserving, i.e. lexicographic comparison of the encodings equals numeric comparison of the ranks. Same idea
// as SyntheticTableGenerator::generate_value(), but with the varying part at the *front* rather than right-aligned
// behind padding, so that this benchmark does not silently become a shared-prefix workload.
constexpr auto ALPHABET = std::string_view{"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
constexpr auto PREFIX_LENGTH = size_t{6};  // 62^6 = 5.68e10 representable ranks.
constexpr auto MAX_ENCODABLE_RANK = uint64_t{56'800'235'584};

// 12 chars fits libstdc++'s 15-char small-string buffer, so the characters live inside the string object and no heap
// indirection is involved in a comparison.
constexpr auto SSO_STRING_LENGTH = size_t{12};
// 8..24 chars straddles the boundary: roughly half the keys are inline, half are heap-allocated.
constexpr auto VAR_MIN_LENGTH = size_t{8};
constexpr auto VAR_MAX_LENGTH = size_t{24};

pmr_string encode_string_key(const uint64_t rank, const KeyType key_type) {
  DebugAssert(rank < MAX_ENCODABLE_RANK, "Rank exceeds what PREFIX_LENGTH base-62 characters can represent.");

  // The length is a deterministic function of the rank, so two rows with the same rank still produce byte-identical
  // strings and the duplicate structure of the distribution is preserved.
  auto length = SSO_STRING_LENGTH;
  if (key_type == KeyType::StringVar) {
    length = VAR_MIN_LENGTH + static_cast<size_t>(mix64(rank) % (VAR_MAX_LENGTH - VAR_MIN_LENGTH + 1));
  }

  auto result = pmr_string(length, ALPHABET.front());

  const auto base = static_cast<uint64_t>(ALPHABET.size());
  auto remainder = rank;
  for (auto position = size_t{0}; position < PREFIX_LENGTH; ++position) {
    result[PREFIX_LENGTH - 1 - position] = ALPHABET[remainder % base];
    remainder /= base;
  }

  // Filler beyond the ordering prefix. Distinct ranks always differ within the first PREFIX_LENGTH characters, so the
  // filler can never affect the ordering; it exists so that the bytes past the prefix are not constant (a constant
  // tail would make every long key compress and prefetch unrealistically well).
  auto filler_state = mix64(rank ^ 0x5DEECE66DULL);
  for (auto position = PREFIX_LENGTH; position < length; ++position) {
    result[position] = ALPHABET[filler_state % base];
    filler_state = mix64(filler_state);
  }

  return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Table generation
// ---------------------------------------------------------------------------------------------------------------

class RankSource {
 public:
  RankSource(const Distribution distribution, const uint32_t seed, std::shared_ptr<const ZipfGenerator> zipf)
      : _distribution{distribution}, _rng{seed}, _zipf{std::move(zipf)} {}

  uint64_t next(const size_t global_row) {
    switch (_distribution) {
      case Distribution::Random:
        return _random_dist(_rng);
      case Distribution::Ascending:
        return static_cast<uint64_t>(global_row);
      case Distribution::Dup128:
        return _dup128_dist(_rng);
      case Distribution::Uniform:
      case Distribution::Zipf090:
      case Distribution::Zipf099:
        return scramble(_zipf->draw(_rng));
    }
    Fail("Unhandled distribution.");
  }

 private:
  Distribution _distribution;
  std::mt19937_64 _rng;
  std::shared_ptr<const ZipfGenerator> _zipf;
  std::uniform_int_distribution<uint64_t> _random_dist{
      0, static_cast<uint64_t>(std::numeric_limits<int32_t>::max())};
  std::uniform_int_distribution<uint64_t> _dup128_dist{0, 127};
};

struct TableSpec {
  size_t row_count{};
  size_t key_column_count{};
  Distribution distribution{};
  KeyType key_type{};

  bool operator==(const TableSpec& other) const = default;
};

// Build a Data table with `key_column_count` columns, chunked at Chunk::DEFAULT_SIZE (so #chunks = #runs =
// ceil(row_count / DEFAULT_SIZE), and output_chunk_count == input_chunk_count). Each column draws independently from
// the requested distribution (distinct seeds), so with low-cardinality data the extra key columns break ties.
// Ascending fills every column with the global row index, yielding an already-sorted table.
template <typename T>
std::shared_ptr<Table> generate_table_impl(const TableSpec& spec) {
  constexpr auto COLUMN_DATA_TYPE = std::is_same_v<T, pmr_string> ? DataType::String : DataType::Int;

  auto column_definitions = TableColumnDefinitions{};
  for (auto column = size_t{0}; column < spec.key_column_count; ++column) {
    column_definitions.emplace_back("c" + std::to_string(column), COLUMN_DATA_TYPE, false);
  }
  auto table = std::make_shared<Table>(column_definitions, TableType::Data, Chunk::DEFAULT_SIZE);

  const auto zipf = zipf_generator_for(spec.distribution);
  auto rank_sources = std::vector<RankSource>{};
  rank_sources.reserve(spec.key_column_count);
  for (auto column = size_t{0}; column < spec.key_column_count; ++column) {
    rank_sources.emplace_back(spec.distribution, SEED + static_cast<uint32_t>(column), zipf);
  }

  const auto chunk_size = size_t{Chunk::DEFAULT_SIZE};
  auto row = size_t{0};
  while (row < spec.row_count) {
    const auto rows_in_chunk = std::min(chunk_size, spec.row_count - row);
    auto segments = Segments{};
    segments.reserve(spec.key_column_count);
    for (auto column = size_t{0}; column < spec.key_column_count; ++column) {
      auto values = pmr_vector<T>();
      values.reserve(rows_in_chunk);
      for (auto offset = size_t{0}; offset < rows_in_chunk; ++offset) {
        const auto rank = rank_sources[column].next(row + offset);
        if constexpr (std::is_same_v<T, pmr_string>) {
          values.push_back(encode_string_key(rank, spec.key_type));
        } else {
          values.push_back(static_cast<int32_t>(rank));
        }
      }
      segments.push_back(std::make_shared<ValueSegment<T>>(std::move(values)));
    }
    table->append_chunk(segments);
    row += rows_in_chunk;
  }

  return table;
}

// Google Benchmark invokes a benchmark function repeatedly (iteration-count estimation, --benchmark_repetitions), and
// regenerating a 50M-row table on every invocation dominates the measurement session. Cache the most recent table:
// the matrix is sorted so that every point sharing a TableSpec is registered consecutively, so a single-entry cache
// captures all of the reuse while never holding two large tables at once.
std::shared_ptr<Table> get_table(const TableSpec& spec) {
  static auto cached_spec = std::optional<TableSpec>{};
  static auto cached_table = std::shared_ptr<Table>{};

  if (cached_spec && *cached_spec == spec) {
    return cached_table;
  }

  cached_spec.reset();
  cached_table.reset();  // Release before allocating the replacement.

  auto table = spec.key_type == KeyType::Int32 ? generate_table_impl<int32_t>(spec)
                                               : generate_table_impl<pmr_string>(spec);
  cached_table = table;
  cached_spec = spec;
  return table;
}

// ---------------------------------------------------------------------------------------------------------------
// Output verification
// ---------------------------------------------------------------------------------------------------------------

// Necessary condition for a correct sort: the leading sort column is non-decreasing across the whole output table.
// Not a full stability or tie-breaking check -- that needs a reference permutation -- but it is O(n), needs no
// reference run, and it fails loudly on the failure mode that actually matters here (a merge that drops or reorders a
// run). Nulls sort first under AscendingNullsFirst and are skipped.
template <typename T>
void assert_leading_column_sorted(const Table& table) {
  auto previous = std::optional<T>{};
  const auto chunk_count = table.chunk_count();
  for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
    const auto& chunk = table.get_chunk(chunk_id);
    if (!chunk) {
      continue;
    }
    segment_iterate<T>(*chunk->get_segment(ColumnID{0}), [&](const auto& position) {
      if (position.is_null()) {
        return;
      }
      const auto value = position.value();
      Assert(!previous || !(value < *previous), "Sort output is not non-decreasing in the leading sort column.");
      previous = value;
    });
  }
}

void verify_output(const Table& table, const KeyType key_type) {
  if (key_type == KeyType::Int32) {
    assert_leading_column_sorted<int32_t>(table);
  } else {
    assert_leading_column_sorted<pmr_string>(table);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Experiment membership
// ---------------------------------------------------------------------------------------------------------------

enum class Experiment : uint8_t {
  E1SingleThreadedInt = 1U << 0U,
  E2MultiThreadedInt = 1U << 1U,
  E3WideKeys = 1U << 2U,
  E4Strings = 1U << 3U,
};

struct BenchmarkConfig {
  KeyType key_type{};
  Distribution distribution{};
  size_t row_count{};
  size_t key_column_count{};
  bool multithreaded{};

  // Bitmask over Experiment. Not part of the point's identity: overlapping points are registered once and carry every
  // experiment they belong to, so a point shared by E1 and E2 is measured once and appears in both slices.
  uint8_t experiments{};

  bool same_point(const BenchmarkConfig& other) const {
    return std::tie(key_type, distribution, row_count, key_column_count, multithreaded) ==
           std::tie(other.key_type, other.distribution, other.row_count, other.key_column_count,
                    other.multithreaded);
  }

  TableSpec table_spec() const {
    return TableSpec{row_count, key_column_count, distribution, key_type};
  }

  bool in(const Experiment experiment) const {
    return (experiments & static_cast<uint8_t>(experiment)) != 0U;
  }
};

// ---------------------------------------------------------------------------------------------------------------
// Benchmark body
// ---------------------------------------------------------------------------------------------------------------

std::string name_of(const KeyType key_type) {
  switch (key_type) {
    case KeyType::Int32:
      return "int32";
    case KeyType::StringSso:
      return "str_sso";
    case KeyType::StringVar:
      return "str_var";
  }
  Fail("Unhandled key type.");
}

std::string name_of(const Distribution distribution) {
  switch (distribution) {
    case Distribution::Random:
      return "random";
    case Distribution::Ascending:
      return "ascending";
    case Distribution::Dup128:
      return "dup128";
    case Distribution::Uniform:
      return "uniform";
    case Distribution::Zipf090:
      return "zipf090";
    case Distribution::Zipf099:
      return "zipf099";
  }
  Fail("Unhandled distribution.");
}

// Name format: Sort/<key type>/<distribution>/rows<N>/cols<N>/mt<0|1>. Slash-separated so that --benchmark_filter
// regexes can select whole axes, e.g. --benchmark_filter='Sort/str_.*/zipf099/'. Experiment membership is a counter
// rather than part of the name, because a point can belong to several experiments.
std::string name_of(const BenchmarkConfig& config) {
  return "Sort/" + name_of(config.key_type) + "/" + name_of(config.distribution) + "/rows" +
         std::to_string(config.row_count) + "/cols" + std::to_string(config.key_column_count) + "/mt" +
         (config.multithreaded ? "1" : "0");
}

int64_t iterations_for(const BenchmarkConfig& config) {
  // One iteration for the 50M points: three sorts of a 50M-row table per repetition is not a better use of the budget
  // than more repetitions, which is what actually gives a variance estimate.
  return config.row_count >= LARGE_ROWS ? LARGE_BENCHMARK_ITERATIONS : BENCHMARK_ITERATIONS;
}

void run_sort_benchmark(benchmark::State& state, const BenchmarkConfig& config) {
  const auto input_table = get_table(config.table_spec());

  const auto input_operator = std::make_shared<TableWrapper>(input_table);
  input_operator->never_clear_output();
  input_operator->execute();

  auto sort_definitions = std::vector<SortColumnDefinition>{};
  sort_definitions.reserve(config.key_column_count);
  for (auto column = size_t{0}; column < config.key_column_count; ++column) {
    sort_definitions.emplace_back(ColumnID{static_cast<uint16_t>(column)}, SortMode::AscendingNullsFirst);
  }

  // Threading: single-threaded runs every JobTask inline via the immediate scheduler; multithreaded uses a
  // NodeQueueScheduler on all cores. Set up once, outside the timed loop.
  //
  // NOTE: under `numactl --cpunodebind=1` the process has 30 logical CPUs, but use_default_topology() enumerates the
  // machine topology. If the reported CPU count is not 30, the scheduler is oversubscribing and every multithreaded
  // number is invalid -- pin the topology explicitly instead.
  if (config.multithreaded) {
    Hyrise::get().topology.use_default_topology();
    Hyrise::get().set_scheduler(std::make_shared<NodeQueueScheduler>());

    static auto topology_reported = false;
    if (!topology_reported) {
      std::cerr << "[sort_benchmark] scheduler topology reports " << Hyrise::get().topology.num_cpus()
                << " CPUs -- confirm this matches the numactl binding.\n";
      topology_reported = true;
    }
  } else {
    Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());
  }

  // Untimed correctness pass. Skipped above the baseline row count so it does not cost a full extra sort of the
  // 50M tables.
  if constexpr (VERIFY_OUTPUT) {
    if (config.row_count <= BASELINE_ROWS) {
      auto verification_sort = std::make_shared<Sort>(input_operator, sort_definitions);
      verification_sort->execute();
      verify_output(*verification_sort->get_output(), config.key_type);
    }
  }

  auto materialize_seconds = double{0};
  auto run_gen_seconds = double{0};
  auto merge_seconds = double{0};
  auto write_out_seconds = double{0};
  auto iterations = size_t{0};

  for (auto _ : state) {
    // Cleared per iteration rather than once per benchmark, so every iteration starts cold rather than only the
    // first. Untimed.
    state.PauseTiming();
    micro_benchmark_clear_cache();
    state.ResumeTiming();

    auto sort = std::make_shared<Sort>(input_operator, sort_definitions);
    sort->execute();

    // Per-step runtimes. MaterializeSortColumns and WriteOutput are negligible for int32 keys but are a first-order
    // cost for string keys (the sort column is copied into and out of the temporary representation), so all four
    // steps are recorded rather than only run generation and the merge.
    //
    // NOTE: merge_s reads OperatorSteps::TemporaryResultWriting, which in stock Hyrise is intermediate-result
    // writing, not merging -- there is no merge step in the baseline build. This mapping is only correct if the
    // sort.cpp variant repurposes that enum slot. Verify before trusting the phase breakdown.
    const auto& performance_data =
        dynamic_cast<const OperatorPerformanceData<Sort::OperatorSteps>&>(*sort->performance_data);
    const auto step_seconds = [&](const Sort::OperatorSteps step) {
      return std::chrono::duration<double>(performance_data.get_step_runtime(step)).count();
    };
    materialize_seconds += step_seconds(Sort::OperatorSteps::MaterializeSortColumns);
    run_gen_seconds += step_seconds(Sort::OperatorSteps::Sort);
    merge_seconds += step_seconds(Sort::OperatorSteps::TemporaryResultWriting);
    write_out_seconds += step_seconds(Sort::OperatorSteps::WriteOutput);
    ++iterations;
  }

  const auto divisor = static_cast<double>(std::max(iterations, size_t{1}));
  state.counters["materialize_s"] = materialize_seconds / divisor;
  state.counters["run_gen_s"] = run_gen_seconds / divisor;
  state.counters["merge_s"] = merge_seconds / divisor;
  state.counters["write_out_s"] = write_out_seconds / divisor;

  // Configuration echoed numerically so the JSON can be grouped without re-parsing benchmark names.
  state.counters["rows"] = static_cast<double>(config.row_count);
  state.counters["cols"] = static_cast<double>(config.key_column_count);
  state.counters["mt"] = config.multithreaded ? 1.0 : 0.0;
  state.counters["key_type"] = static_cast<double>(static_cast<uint8_t>(config.key_type));
  state.counters["distribution"] = static_cast<double>(static_cast<uint8_t>(config.distribution));
  state.counters["theta"] = theta_of(config.distribution);

  // Experiment membership. A point can belong to several, so these are independent flags rather than one label.
  state.counters["in_e1"] = config.in(Experiment::E1SingleThreadedInt) ? 1.0 : 0.0;
  state.counters["in_e2"] = config.in(Experiment::E2MultiThreadedInt) ? 1.0 : 0.0;
  state.counters["in_e3"] = config.in(Experiment::E3WideKeys) ? 1.0 : 0.0;
  state.counters["in_e4"] = config.in(Experiment::E4Strings) ? 1.0 : 0.0;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * config.row_count));

  // Reset to the immediate scheduler so a running NodeQueueScheduler does not leak into other benchmarks.
  if (config.multithreaded) {
    Hyrise::get().scheduler()->finish();
  }
  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());
}

// ---------------------------------------------------------------------------------------------------------------
// Experiment matrix
// ---------------------------------------------------------------------------------------------------------------

class MatrixBuilder {
 public:
  void add(const KeyType key_type, const Distribution distribution, const size_t row_count,
           const size_t key_column_count, const bool multithreaded, const Experiment experiment) {
    auto candidate = BenchmarkConfig{key_type, distribution, row_count, key_column_count, multithreaded, 0};
    const auto existing = std::find_if(_configs.begin(), _configs.end(),
                                       [&](const auto& config) { return config.same_point(candidate); });
    if (existing == _configs.end()) {
      candidate.experiments = static_cast<uint8_t>(experiment);
      _configs.push_back(candidate);
    } else {
      existing->experiments |= static_cast<uint8_t>(experiment);
    }
  }

  // Registration order is execution order in Google Benchmark. Sorting by TableSpec puts every point sharing a table
  // consecutively, so the single-entry table cache never misses on a spec it has already built (previously each spec
  // was generated twice, once per threading mode). Row count leads the ordering so the cheap points run first and a
  // failure at 50M still leaves the smaller results on disk.
  std::vector<BenchmarkConfig> take() {
    std::sort(_configs.begin(), _configs.end(), [](const auto& lhs, const auto& rhs) {
      return std::tie(lhs.row_count, lhs.key_type, lhs.distribution, lhs.key_column_count, lhs.multithreaded) <
             std::tie(rhs.row_count, rhs.key_type, rhs.distribution, rhs.key_column_count, rhs.multithreaded);
    });
    return std::move(_configs);
  }

 private:
  std::vector<BenchmarkConfig> _configs;
};

// E1 -- single-threaded, integer keys. OFAT: the distribution axis at the baseline row count, the row-count axis at
// the baseline distribution.
void add_e1(MatrixBuilder& builder) {
  constexpr auto EXPERIMENT = Experiment::E1SingleThreadedInt;

  for (const auto distribution : {Distribution::Random, Distribution::Ascending, Distribution::Dup128,
                                  Distribution::Uniform, Distribution::Zipf099}) {
    builder.add(KeyType::Int32, distribution, BASELINE_ROWS, BASELINE_COLS, false, EXPERIMENT);
  }
  for (const auto row_count : {SMALL_ROWS, LARGE_ROWS}) {
    builder.add(KeyType::Int32, Distribution::Random, row_count, BASELINE_COLS, false, EXPERIMENT);
  }
}

// E2 -- multi-threaded, integer keys. Same shape as E1 but paired across both threading modes, and restricted to the
// distributions where skew is the question. The mt0 half overlaps E1 and is not re-registered.
void add_e2(MatrixBuilder& builder) {
  constexpr auto EXPERIMENT = Experiment::E2MultiThreadedInt;

  for (const auto multithreaded : {false, true}) {
    for (const auto distribution : {Distribution::Random, Distribution::Uniform, Distribution::Zipf099}) {
      builder.add(KeyType::Int32, distribution, BASELINE_ROWS, BASELINE_COLS, multithreaded, EXPERIMENT);
    }
    for (const auto row_count : {SMALL_ROWS, LARGE_ROWS}) {
      builder.add(KeyType::Int32, Distribution::Random, row_count, BASELINE_COLS, multithreaded, EXPERIMENT);
    }
  }
}

// E3 -- wide and composite keys. Full factorial over {distribution} x {key columns} x {threading}: the cells are the
// point, so OFAT would defeat the purpose. zipf099 x cols5 is where ties in c0 push work into the later columns.
void add_e3(MatrixBuilder& builder) {
  constexpr auto EXPERIMENT = Experiment::E3WideKeys;

  for (const auto multithreaded : {false, true}) {
    for (const auto distribution : {Distribution::Random, Distribution::Zipf099}) {
      for (const auto key_column_count : {BASELINE_COLS, WIDE_COLS}) {
        builder.add(KeyType::Int32, distribution, BASELINE_ROWS, key_column_count, multithreaded, EXPERIMENT);
      }
    }
  }
}

// E4 -- string keys. The row-count axis is crossed with the key type (including int32) rather than run OFAT, because
// the deliverable is a ns/tuple-against-n curve per key type on one plot. 50M runs multithreaded only.
void add_e4(MatrixBuilder& builder) {
  constexpr auto EXPERIMENT = Experiment::E4Strings;

  for (const auto key_type : {KeyType::Int32, KeyType::StringSso, KeyType::StringVar}) {
    for (const auto multithreaded : {false, true}) {
      for (const auto row_count : {SMALL_ROWS, BASELINE_ROWS}) {
        builder.add(key_type, Distribution::Random, row_count, BASELINE_COLS, multithreaded, EXPERIMENT);
      }
    }
    builder.add(key_type, Distribution::Random, LARGE_ROWS, BASELINE_COLS, true, EXPERIMENT);
  }

  // Skew for the string types. Skewed duplicates concentrate the expensive string comparisons on the hot values; the
  // int32 counterparts are already covered by E2.
  for (const auto key_type : {KeyType::StringSso, KeyType::StringVar}) {
    for (const auto multithreaded : {false, true}) {
      builder.add(key_type, Distribution::Zipf099, BASELINE_ROWS, BASELINE_COLS, multithreaded, EXPERIMENT);
    }
  }
}

std::vector<BenchmarkConfig> build_matrix() {
  auto builder = MatrixBuilder{};
  add_e1(builder);
  add_e2(builder);
  add_e3(builder);
  add_e4(builder);
  return builder.take();
}

// Registered at static-initialisation time, i.e. before BENCHMARK_MAIN()'s call to benchmark::Initialize(). Runtime
// registration (rather than BENCHMARK(...)->Apply(...)) is what allows the matrix to be an explicit list of points
// instead of a cross product, which is the whole point of the OFAT design.
struct BenchmarkRegistrar {
  BenchmarkRegistrar() {
    for (const auto& config : build_matrix()) {
      const auto name = name_of(config);
      auto* registered = benchmark::RegisterBenchmark(name.c_str(), [config](benchmark::State& state) {
        run_sort_benchmark(state, config);
      });
      registered->UseRealTime()->Iterations(iterations_for(config));
    }
  }
};

[[maybe_unused]] const auto benchmark_registrar = BenchmarkRegistrar{};

}  // namespace