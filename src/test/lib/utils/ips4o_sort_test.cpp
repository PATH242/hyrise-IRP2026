#include <algorithm>
#include <cstdint>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "base_test.hpp"
#include "hyrise.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "utils/ips4o_sort.hpp"

namespace hyrise {

namespace {

using Value = uint64_t;

const auto less = [](const Value& lhs, const Value& rhs) {
  return lhs < rhs;
};

enum class Distribution { Uniform, FewDistinct, AllEqual, Sorted, Reverse, Skewed };

std::string distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::Uniform:
      return "Uniform";
    case Distribution::FewDistinct:
      return "FewDistinct";
    case Distribution::AllEqual:
      return "AllEqual";
    case Distribution::Sorted:
      return "Sorted";
    case Distribution::Reverse:
      return "Reverse";
    case Distribution::Skewed:
      return "Skewed";
  }
  return "";
}

std::vector<Value> generate(Distribution distribution, size_t size, uint64_t seed) {
  auto generator = std::mt19937_64{seed};
  auto values = std::vector<Value>(size);
  switch (distribution) {
    case Distribution::Uniform: {
      auto dist = std::uniform_int_distribution<Value>{};
      std::ranges::generate(values, [&]() {
        return dist(generator);
      });
    } break;
    case Distribution::FewDistinct: {
      auto dist = std::uniform_int_distribution<Value>{0, 15};
      std::ranges::generate(values, [&]() {
        return dist(generator);
      });
    } break;
    case Distribution::AllEqual:
      std::ranges::fill(values, 42);
      break;
    case Distribution::Sorted:
      for (auto index = size_t{0}; index < size; ++index) {
        values[index] = index;
      }
      break;
    case Distribution::Reverse:
      for (auto index = size_t{0}; index < size; ++index) {
        values[index] = size - index;
      }
      break;
    case Distribution::Skewed: {
      // 90% of the values are a single key, the rest is uniform. Produces very unbalanced buckets.
      auto coin = std::uniform_int_distribution<int>{0, 9};
      auto dist = std::uniform_int_distribution<Value>{};
      std::ranges::generate(values, [&]() {
        return coin(generator) == 0 ? dist(generator) : Value{1'000'000};
      });
    } break;
  }
  return values;
}

// Returns a description of the first violation of the sort order, or an empty string if sorted.
std::string first_unsorted_position(const std::vector<Value>& values) {
  const auto it = std::ranges::is_sorted_until(values);
  if (it == values.end()) {
    return "";
  }
  const auto index = std::distance(values.begin(), it);
  auto num_violations = size_t{0};
  for (auto pos = size_t{1}; pos < values.size(); ++pos) {
    num_violations += values[pos] < values[pos - 1];
  }
  return std::format("first violation at index {} of {} (values[{}]={} > values[{}]={}), {} violations in total",
                     index, values.size(), index - 1, values[index - 1], index, values[index], num_violations);
}

struct SortParameters {
  size_t size;
  size_t max_parallelism;
  size_t min_blocks_per_stripe;
  size_t num_classifiers;
  size_t samples_per_classifier;
  size_t block_size;
};

std::string parameters_name(const SortParameters& params) {
  return std::format("n{}_p{}_mbps{}_c{}_s{}_b{}", params.size, params.max_parallelism, params.min_blocks_per_stripe,
                     params.num_classifiers, params.samples_per_classifier, params.block_size);
}

// The release configuration of the Sort operator (see Sort::Config) for a machine with `threads` hardware threads.
SortParameters release_config(size_t size, size_t threads) {
  return {size, threads, 32, (threads * 2) - 1, 4, 256};
}

}  // namespace

class IPS4oSortTest : public BaseTest {
 protected:
  void use_multithreaded_scheduler() {
    Hyrise::get().topology.use_fake_numa_topology(8, 4);
    Hyrise::get().set_scheduler(std::make_shared<NodeQueueScheduler>());
  }

  // Sorts the values with ips4o and compares the result to std::sort.
  void check_sort(const SortParameters& params, Distribution distribution, uint64_t seed) {
    SCOPED_TRACE(parameters_name(params) + "_" + distribution_name(distribution) + "_seed" + std::to_string(seed));
    auto values = generate(distribution, params.size, seed);
    auto expected = values;
    std::ranges::sort(expected);

    ips4o::sort<Value>(std::views::all(values), less, params.max_parallelism, params.min_blocks_per_stripe,
                       params.num_classifiers, params.samples_per_classifier, params.block_size);

    const auto unsorted = first_unsorted_position(values);
    EXPECT_TRUE(unsorted.empty()) << "Result is not sorted: " << unsorted;

    // Check that the result is a permutation of the input (no lost or duplicated elements).
    auto sorted_copy = values;
    std::ranges::sort(sorted_copy);
    EXPECT_TRUE(sorted_copy == expected) << "Result is not a permutation of the input";
  }

  // Runs a single ips4o pass and checks its invariants: the result is a permutation of the input and all elements of
  // bucket i are less than or equal to all elements of bucket i + 1.
  void check_pass(size_t size, size_t num_buckets, size_t samples_per_classifier, size_t block_size,
                  size_t num_stripes, Distribution distribution, uint64_t seed) {
    SCOPED_TRACE(std::format("pass_n{}_buckets{}_s{}_b{}_stripes{}_{}_seed{}", size, num_buckets,
                             samples_per_classifier, block_size, num_stripes, distribution_name(distribution), seed));
    auto values = generate(distribution, size, seed);
    auto expected = values;
    std::ranges::sort(expected);

    auto range = std::views::all(values);
    const auto delimiter =
        ips4o::ips4o_pass<Value>(range, num_buckets, samples_per_classifier, block_size, num_stripes, less);

    ASSERT_EQ(delimiter.size(), num_buckets);
    ASSERT_EQ(delimiter.back(), size) << "Bucket delimiters do not cover the whole input";
    ASSERT_TRUE(std::ranges::is_sorted(delimiter)) << "Bucket delimiters are not monotonic";

    auto sorted_copy = values;
    std::ranges::sort(sorted_copy);
    EXPECT_TRUE(sorted_copy == expected) << "Pass result is not a permutation of the input";

    // Every bucket's maximum must not exceed the next non-empty bucket's minimum.
    auto bucket_begin = size_t{0};
    auto previous_max = std::optional<Value>{};
    auto previous_bucket = size_t{0};
    for (auto bucket = size_t{0}; bucket < num_buckets; ++bucket) {
      const auto bucket_end = delimiter[bucket];
      if (bucket_end == bucket_begin) {
        continue;
      }
      const auto [min_it, max_it] =
          std::minmax_element(values.begin() + bucket_begin, values.begin() + bucket_end);
      if (previous_max) {
        EXPECT_LE(*previous_max, *min_it) << "Bucket " << previous_bucket << " (max " << *previous_max
                                          << ") overlaps bucket " << bucket << " [" << bucket_begin << ", "
                                          << bucket_end << ") (min " << *min_it << ")";
      }
      previous_max = *max_it;
      previous_bucket = bucket;
      bucket_begin = bucket_end;
    }
  }
};

constexpr auto ALL_DISTRIBUTIONS = std::array{Distribution::Uniform,  Distribution::FewDistinct, Distribution::AllEqual,
                                              Distribution::Sorted,   Distribution::Reverse,     Distribution::Skewed};

// --- Single pass, single threaded (ImmediateExecutionScheduler). Isolates logic bugs from races. ---

TEST_F(IPS4oSortTest, SinglePassSingleThreaded) {
  // Sizes that are and are not multiples of the block size.
  for (const auto size : {size_t{4096}, size_t{4097}, size_t{5000}, size_t{100'003}}) {
    for (const auto num_buckets : {size_t{2}, size_t{4}, size_t{32}}) {
      for (const auto num_stripes : {size_t{2}, size_t{3}, size_t{7}, size_t{16}}) {
        for (const auto distribution : ALL_DISTRIBUTIONS) {
          for (auto seed = uint64_t{0}; seed < 3; ++seed) {
            check_pass(size, num_buckets, 4, 16, num_stripes, distribution, seed);
          }
        }
      }
    }
  }
}

TEST_F(IPS4oSortTest, SinglePassMultiThreaded) {
  use_multithreaded_scheduler();
  for (const auto size : {size_t{100'000}, size_t{100'003}, size_t{1'000'000}}) {
    for (const auto num_buckets : {size_t{4}, size_t{32}, size_t{256}}) {
      for (const auto num_stripes : {size_t{2}, size_t{8}, size_t{32}}) {
        for (const auto distribution : ALL_DISTRIBUTIONS) {
          check_pass(size, num_buckets, 4, 64, num_stripes, distribution, 0);
        }
      }
    }
  }
}

// --- Full sort. Inputs must be larger than 16 * block_size blocks, otherwise pdqsort is used directly. ---

TEST_F(IPS4oSortTest, FullSortSmallBlocksSingleThreaded) {
  for (const auto size : {size_t{300'000}, size_t{300'001}, size_t{1'000'003}}) {
    for (const auto distribution : ALL_DISTRIBUTIONS) {
      check_sort({size, 8, 4, 15, 4, 16}, distribution, 0);
      check_sort({size, 16, 4, 63, 4, 16}, distribution, 1);
    }
  }
}

TEST_F(IPS4oSortTest, FullSortSmallBlocksMultiThreaded) {
  use_multithreaded_scheduler();
  for (const auto size : {size_t{300'000}, size_t{300'001}, size_t{1'000'003}}) {
    for (const auto distribution : ALL_DISTRIBUTIONS) {
      check_sort({size, 8, 4, 15, 4, 16}, distribution, 0);
      check_sort({size, 16, 4, 63, 4, 16}, distribution, 1);
    }
  }
}

// Same parameters the Sort operator uses in release builds. Requires > 16 * 256 * 256 ~ 1M elements to not fall
// back to pdqsort.
TEST_F(IPS4oSortTest, FullSortReleaseConfig) {
  use_multithreaded_scheduler();
  for (const auto size : {size_t{2'000'000}, size_t{5'000'001}, size_t{20'000'000}}) {
    for (const auto distribution : ALL_DISTRIBUTIONS) {
      check_sort(release_config(size, 16), distribution, 0);
      check_sort(release_config(size, 64), distribution, 0);
    }
  }
}

}  // namespace hyrise
