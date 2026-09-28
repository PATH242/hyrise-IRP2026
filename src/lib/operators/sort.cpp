#include "sort.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/sort/pdqsort/pdqsort.hpp>

#include "all_type_variant.hpp"
#include "operators/abstract_operator.hpp"
#include "operators/abstract_read_only_operator.hpp"
#include "operators/operator_performance_data.hpp"
#include "resolve_type.hpp"
#include "storage/abstract_segment.hpp"
#include "storage/base_segment_accessor.hpp"
#include "storage/chunk.hpp"
#include "storage/pos_lists/row_id_pos_list.hpp"
#include "storage/reference_segment.hpp"
#include "storage/segment_accessor.hpp"
#include "storage/segment_iterate.hpp"
#include "storage/table.hpp"
#include "storage/value_segment.hpp"
#include "types.hpp"
#include "utils/assert.hpp"
#include "utils/timer.hpp"

/**
 * Subsort: column-at-a-time sort with tie resolution, single-threaded.
 *
 * This is the subsort approach of Kuiper & Muehleisen, "These Rows Are Made for Sorting and That's Just What We'll
 * Do" (ICDE 2023), Section IV, applied to a columnar source: sort everything by the first key column, identify the
 * tuples that are tied in that column, sort those by the second key column, and repeat until no ties remain or all
 * key columns are done. As in the paper's columnar variant, rows are represented by their RowID and the values of
 * one key column at a time, so each pass causes random access in a single column and the comparison function has no
 * branches for further columns.
 *
 * The first sort column is materialized as (RowID, is_null, value) tuples and sorted with pdqsort; every tie run is
 * then sorted by the next sort column, whose values are fetched only for the rows of that run.
 *
 * There is no parallelism anywhere in this operator: no tasks, no scheduler, no k-way merge. The MergeTime and
 * MergePath steps therefore stay unset (they read back as 0), and SORT_US covers the first-column sort plus all tie
 * resolution.
 *
 * Derived from the subsort in hyrise/dyod2025_group1717 (src/lib/operators/sort.cpp). Changes relative to it:
 *   B1  NULLs in a subsorted column formed no tie run, so rows NULL in column k were never ordered by column k + 1.
 *   B2  With nulls last, the final tie run of a subsorted column extended over the trailing NULLs.
 *   B3  The first-column tie scan compared the placeholder value of NULLs (T{}), merging NULLs into an adjacent run
 *       of T{} (e.g., 0 or "") when nulls sort last.
 *   B4  The first element of each scan was counted twice, so singleton ranges were subsorted (harmless, wasted work).
 *   -   Comparators are functors dispatched once per sort instead of std::function calls per comparison, which is
 *       the static- vs. dynamic-comparator difference measured in Section V-B of the paper.
 *   -   All task scheduling removed (materialization, tie resolution, and output writing are plain loops).
 * B1-B3 only affect NULLs; without NULLs the reference and this implementation produce equally valid orders.
 *
 * Neither variant is stable: rows that tie on all sort columns may appear in any order.
 */

namespace {

using namespace hyrise;  // NOLINT(build/namespaces)

// ==== BEGIN SUBSORT CORE ============================================================================================
// Everything between the BEGIN/END markers is Hyrise-independent (it only needs RowID, SortMode, Fail and boost
// pdqsort). The standalone fuzz harness extracts exactly this block and compiles it against minimal shims, so the
// tested code is this code. Nothing in here spawns tasks or threads.

// Tuple comparator on (id, is_null, value) tuples. Replaces the std::function-based get_comparator() of the
// reference: the sort mode is dispatched once per sort instead of once per comparison, which is the static- vs.
// dynamic-comparator effect measured in Section V-B of Kuiper & Muehleisen (ICDE 2023).
template <bool ASCENDING, bool NULLS_FIRST>
struct TupleLess {
  template <typename Tuple>
  bool operator()(const Tuple& lhs, const Tuple& rhs) const {
    const auto lhs_is_null = std::get<1>(lhs);
    const auto rhs_is_null = std::get<1>(rhs);
    if (lhs_is_null || rhs_is_null) {
      return NULLS_FIRST ? (lhs_is_null && !rhs_is_null) : (!lhs_is_null && rhs_is_null);
    }
    if constexpr (ASCENDING) {
      return std::get<2>(lhs) < std::get<2>(rhs);
    } else {
      return std::get<2>(rhs) < std::get<2>(lhs);
    }
  }
};

template <typename Functor>
void with_comparator(const SortMode sort_mode, Functor&& functor) {
  switch (sort_mode) {
    case SortMode::AscendingNullsFirst:
      functor(TupleLess<true, true>{});
      return;
    case SortMode::AscendingNullsLast:
      functor(TupleLess<true, false>{});
      return;
    case SortMode::DescendingNullsFirst:
      functor(TupleLess<false, true>{});
      return;
    case SortMode::DescendingNullsLast:
      functor(TupleLess<false, false>{});
      return;
  }
  Fail("Unsupported sort mode.");
}

// Two tuples tie on the current column iff both are NULL or both are non-NULL with equal values. The NULL check
// comes first: the value slot of a NULL is a placeholder (T{}) and must not be compared (fixes B3 of the reference).
template <typename Tuple>
bool same_key(const Tuple& lhs, const Tuple& rhs) {
  const auto lhs_is_null = std::get<1>(lhs);
  if (lhs_is_null != std::get<1>(rhs)) {
    return false;
  }
  return lhs_is_null || std::get<2>(lhs) == std::get<2>(rhs);
}

// Calls on_run(run_begin, run_end) for every maximal range of length > 1 with equal keys in the sorted range
// [begin, end) -- the "identify the tuples that have an equal value in this column" step of the subsort approach.
// NULLs form a tie run like any other value (fixes B1/B2); singleton ranges are never reported (B4).
template <typename Iterator, typename OnRun>
void for_each_tie_run(const Iterator begin, const Iterator end, OnRun&& on_run) {
  auto run_begin = begin;
  while (run_begin != end) {
    auto run_end = std::next(run_begin);
    while (run_end != end && same_key(*run_begin, *run_end)) {
      ++run_end;
    }
    if (std::distance(run_begin, run_end) > 1) {
      on_run(run_begin, run_end);
    }
    run_begin = run_end;
  }
}

// ---- Subsort (column-at-a-time tie resolution, Section IV of Kuiper & Muehleisen) ----------------------------------
//
// `Source` abstracts the table: column_count(), sort_mode(k), and visit(k, f), which calls f(fetch) with a callable
// fetch(RowID) -> std::optional<T> for sort column k.

// Sorts the rows in [begin, end), which tie on sort columns 0 .. column_offset - 1, by sort column column_offset and
// recurses into the ties of that column. Only the values of the rows in this range are read, so later key columns
// are only ever touched for rows that tie on all previous ones.
template <typename Row, typename Source>
void subsort_range(Row* begin, Row* end, const size_t column_offset, const Source& source) {
  source.visit(column_offset, [&](auto&& fetch) {
    using OptionalValue = std::decay_t<decltype(fetch(std::get<0>(*begin)))>;
    using T = typename OptionalValue::value_type;

    const auto range_length = static_cast<size_t>(end - begin);
    auto values = std::vector<std::tuple<size_t, bool, T>>{};
    values.reserve(range_length);
    for (auto offset = size_t{0}; offset < range_length; ++offset) {
      auto value = fetch(std::get<0>(begin[offset]));
      if (value) {
        values.emplace_back(offset, false, std::move(*value));
      } else {
        values.emplace_back(offset, true, T{});
      }
    }

    with_comparator(source.sort_mode(column_offset), [&](const auto less) {
      boost::sort::pdqsort(values.begin(), values.end(), less);
    });

    // Apply the permutation to the first-column rows.
    auto rows = std::vector<Row>(std::make_move_iterator(begin), std::make_move_iterator(end));
    for (auto offset = size_t{0}; offset < range_length; ++offset) {
      begin[offset] = std::move(rows[std::get<0>(values[offset])]);
    }

    if (column_offset + 1 == source.column_count()) {
      return;
    }

    for_each_tie_run(values.begin(), values.end(), [&](const auto run_begin, const auto run_end) {
      subsort_range(begin + (run_begin - values.begin()), begin + (run_end - values.begin()), column_offset + 1,
                    source);
    });
  });
}

// Sorts the materialized first sort column and resolves all its ties. Single-threaded throughout.
struct SubsortTimings {
  std::chrono::nanoseconds first_column_sort{};
  std::chrono::nanoseconds sub_sort{};
};

template <typename Row, typename Source>
void subsort_sort(std::vector<Row>& rows, const Source& source, SubsortTimings& timings) {
  const auto clock_start = std::chrono::steady_clock::now();
  with_comparator(source.sort_mode(0), [&](const auto less) {
    boost::sort::pdqsort(rows.begin(), rows.end(), less);
  });
  const auto clock_sorted = std::chrono::steady_clock::now();

  if (source.column_count() > 1 && !rows.empty()) {
    for_each_tie_run(rows.begin(), rows.end(), [&](const auto run_begin, const auto run_end) {
      auto* range_begin = &*run_begin;
      subsort_range(range_begin, range_begin + (run_end - run_begin), 1, source);
    });
  }

  timings.first_column_sort = clock_sorted - clock_start;
  timings.sub_sort = std::chrono::steady_clock::now() - clock_sorted;
}
// ==== END SUBSORT CORE ==============================================================================================

// Ceiling of integer division.
size_t div_ceil(const auto lhs, const auto rhs) {
  DebugAssert(rhs > 0, "Divisor must be larger than 0.");
  return (lhs + rhs - 1u) / rhs;
}

// Table access for subsort_range(): sort column k of `table`, fetched through per-visit segment accessors.
struct TableColumnSource {
  const Table& table;
  const std::vector<SortColumnDefinition>& sort_definitions;

  size_t column_count() const {
    return sort_definitions.size();
  }

  SortMode sort_mode(const size_t sort_column_offset) const {
    return sort_definitions[sort_column_offset].sort_mode;
  }

  template <typename Functor>
  void visit(const size_t sort_column_offset, Functor&& functor) const {
    const auto column_id = sort_definitions[sort_column_offset].column;
    resolve_data_type(table.column_data_type(column_id), [&](auto type) {
      using ColumnDataType = typename decltype(type)::type;

      auto accessor_by_chunk_id = std::unordered_map<ChunkID, std::unique_ptr<AbstractSegmentAccessor<ColumnDataType>>>{};
      const auto fetch = [&](const RowID& row_id) -> std::optional<ColumnDataType> {
        auto iter = accessor_by_chunk_id.find(row_id.chunk_id);
        if (iter == accessor_by_chunk_id.end()) {
          const auto& segment = table.get_chunk(row_id.chunk_id)->get_segment(column_id);
          iter = accessor_by_chunk_id.emplace(row_id.chunk_id, create_segment_accessor<ColumnDataType>(segment)).first;
        }
        return iter->second->access(row_id.chunk_offset);
      };
      functor(fetch);
    });
  }
};

// Given an unsorted_table and a pos_list that defines the output order, this materializes all columns in the table,
// creating chunks of output_chunk_size rows at maximum.
std::shared_ptr<Table> write_materialized_output_table(const std::shared_ptr<const Table>& unsorted_table,
                                                       std::vector<RowIDPosList> pos_lists,
                                                       const ChunkOffset output_chunk_size) {
  // We have decided against duplicating MVCC data in https://github.com/hyrise/hyrise/issues/408
  auto output = std::make_shared<Table>(unsorted_table->column_definitions(), TableType::Data, output_chunk_size);

  auto output_segments_by_chunk = std::vector<Segments>(pos_lists.size());
  const auto output_column_count = unsorted_table->column_count();

  const auto pos_list_count = pos_lists.size();
  for (auto pos_list_index = size_t{0}; pos_list_index < pos_list_count; ++pos_list_index) {
    {
      const auto& pos_list = pos_lists[pos_list_index];
      const auto row_count = pos_list.size();
      for (auto column_id = ColumnID{0}; column_id < output_column_count; ++column_id) {
        const auto column_data_type = output->column_data_type(column_id);
        const auto column_is_nullable = unsorted_table->column_is_nullable(column_id);

        resolve_data_type(column_data_type, [&](auto type) {
          using ColumnDataType = typename decltype(type)::type;

          auto& chunk_it = output_segments_by_chunk[pos_list_index];

          auto value_segment_value_vector = pmr_vector<ColumnDataType>{};
          auto value_segment_null_vector = pmr_vector<bool>{};

          value_segment_value_vector.reserve(row_count);
          if (column_is_nullable) {
            value_segment_null_vector.reserve(row_count);
          }

          auto accessor_by_chunk_id =
              std::unordered_map<ChunkID, std::unique_ptr<AbstractSegmentAccessor<ColumnDataType>>>();
          for (auto row_index = size_t{0}; row_index < row_count; ++row_index) {
            const auto [chunk_id, chunk_offset] = pos_list[row_index];

            auto iter = accessor_by_chunk_id.find(chunk_id);
            if (iter == accessor_by_chunk_id.end()) {
              const auto& abstract_segment = unsorted_table->get_chunk(chunk_id)->get_segment(column_id);
              auto iter_and_inserted =
                  accessor_by_chunk_id.emplace(chunk_id, create_segment_accessor<ColumnDataType>(abstract_segment));
              DebugAssert(iter_and_inserted.second, "Unexpected non-insert.");
              iter = iter_and_inserted.first;
            }

            auto& accessor = iter->second;
            const auto typed_value = accessor->access(chunk_offset);
            const auto is_null = !typed_value;
            value_segment_value_vector.push_back(is_null ? ColumnDataType{} : typed_value.value());
            if (column_is_nullable) {
              value_segment_null_vector.push_back(is_null);
            }
          }

          auto value_segment = std::shared_ptr<ValueSegment<ColumnDataType>>{};
          if (column_is_nullable) {
            value_segment = std::make_shared<ValueSegment<ColumnDataType>>(std::move(value_segment_value_vector),
                                                                           std::move(value_segment_null_vector));
          } else {
            value_segment = std::make_shared<ValueSegment<ColumnDataType>>(std::move(value_segment_value_vector));
          }
          chunk_it.push_back(value_segment);
        });
      }
    }
  }

  for (auto& segments : output_segments_by_chunk) {
    output->append_chunk(segments);
  }

  Assert(output->row_count() == unsorted_table->row_count(), "Mismatching size of input and output table.");

  return output;
}

// Given an unsorted_table and an input_pos_list that defines the output order, this writes the output table as a
// reference table. This is usually faster, but can only be done if a single column in the input table does not
// reference multiple tables. An example where this restriction applies is the sorted result of a union between two
// tables. The restriction is needed because a ReferenceSegment can only reference a single table. It does, however,
// not necessarily apply to joined tables, so two tables referenced in different columns is fine.
//
// If unsorted_table is of TableType::Data, this is trivial and the input_pos_list is used to create the output
// reference table. If the input is already a reference table, the double indirection needs to be resolved.
std::shared_ptr<Table> write_reference_output_table(const std::shared_ptr<const Table>& unsorted_table,
                                                    std::vector<RowIDPosList> input_pos_lists,
                                                    const ChunkOffset /*output_chunk_size*/) {
  // We have decided against duplicating MVCC data in https://github.com/hyrise/hyrise/issues/408
  auto output_table = std::make_shared<Table>(unsorted_table->column_definitions(), TableType::References);

  const auto resolve_indirection = unsorted_table->type() == TableType::References;
  const auto column_count = output_table->column_count();

  auto output_segments_by_chunk = std::vector<Segments>(input_pos_lists.size(), Segments(column_count));

  const auto pos_list_count = input_pos_lists.size();
  for (auto pos_list_index = size_t{0}; pos_list_index < pos_list_count; ++pos_list_index) {
    {
      auto& input_pos_list = input_pos_lists[pos_list_index];
      auto& output_segments = output_segments_by_chunk[pos_list_index];
      if (!resolve_indirection) {
        const auto output_pos_list = std::make_shared<RowIDPosList>(std::move(input_pos_list));
        for (auto column_id = ColumnID{0}; column_id < column_count; ++column_id) {
          output_segments[column_id] = std::make_shared<ReferenceSegment>(unsorted_table, column_id, output_pos_list);
        }
        continue;
      }

      for (auto column_id = ColumnID{0}; column_id < column_count; ++column_id) {
        // To keep the implementation simple, we write the output ReferenceSegments column by column. This means that
        // even if input ReferenceSegments share a PosList, the output will contain independent PosLists.
        auto output_pos_list = std::make_shared<RowIDPosList>();
        output_pos_list->reserve(input_pos_list.size());

        const auto input_chunk_count = unsorted_table->chunk_count();
        auto input_segments = std::vector<std::shared_ptr<AbstractSegment>>(input_chunk_count);
        for (auto input_chunk_id = ChunkID{0}; input_chunk_id < input_chunk_count; ++input_chunk_id) {
          input_segments[input_chunk_id] = unsorted_table->get_chunk(input_chunk_id)->get_segment(column_id);
        }

        const auto first_reference_segment = std::dynamic_pointer_cast<ReferenceSegment>(input_segments.at(0));
        const auto referenced_table = first_reference_segment->referenced_table();
        const auto referenced_column_id = first_reference_segment->referenced_column_id();

        for (const auto& row_id : input_pos_list) {
          const auto& input_reference_segment = static_cast<ReferenceSegment&>(*input_segments[row_id.chunk_id]);
          DebugAssert(input_reference_segment.referenced_table() == referenced_table,
                      "Input column references more than one table");
          DebugAssert(input_reference_segment.referenced_column_id() == referenced_column_id,
                      "Input column references more than one column");
          const auto& input_reference_pos_list = input_reference_segment.pos_list();
          output_pos_list->emplace_back((*input_reference_pos_list)[row_id.chunk_offset]);
        }

        DebugAssert(!output_pos_list->empty(), "Asked to write empty output_pos_list.");
        output_segments[column_id] =
            std::make_shared<ReferenceSegment>(referenced_table, referenced_column_id, output_pos_list);
      }
    }
  }

  for (auto& segments : output_segments_by_chunk) {
    output_table->append_chunk(segments);
  }

  Assert(output_table->row_count() == unsorted_table->row_count(), "Mismatching size of input and output table.");

  return output_table;
}

}  // namespace

namespace hyrise {

template <typename SortColumnType>
class Sort::SortImpl {
 public:
  using Row = std::tuple<RowID, bool, SortColumnType>;

  std::chrono::nanoseconds key_materialization_time{};
  std::chrono::nanoseconds sort_time{};
  std::chrono::nanoseconds result_materialization_time{};

  SortImpl(const std::shared_ptr<const Table>& table_in, const std::vector<SortColumnDefinition>& sort_definitions,
           const ChunkOffset output_chunk_size)
      : _table_in(table_in), _sort_definitions(sort_definitions), _output_chunk_size(output_chunk_size) {}

  std::vector<RowIDPosList> sort() {
    auto timer = Timer{};
    _materialize_first_column();
    key_materialization_time = timer.lap();

    auto timings = SubsortTimings{};
    subsort_sort(_rows, TableColumnSource{*_table_in, _sort_definitions}, timings);
    timer.lap();
    // SORT_US covers the first-column sort and the tie resolution of all further key columns. The values of those
    // columns are materialized inside the tie resolution (only for tied rows), so that cost is part of it.
    sort_time = timings.first_column_sort + timings.sub_sort;

    auto pos_lists = _write_pos_lists();
    result_materialization_time = timer.lap();
    return pos_lists;
  }

 protected:
  // Materializes the first sort column in chunk order.
  void _materialize_first_column() {
    const auto column_id = _sort_definitions[0].column;
    const auto chunk_count = _table_in->chunk_count();

    auto row_count = size_t{0};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto chunk = _table_in->get_chunk(chunk_id);
      Assert(chunk, "Did not expect deleted chunk here.");  // see https://github.com/hyrise/hyrise/issues/1686
      row_count += chunk->size();
    }
    _rows.resize(row_count);

    auto write_offset = size_t{0};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& segment = _table_in->get_chunk(chunk_id)->get_segment(column_id);
      segment_iterate<SortColumnType>(*segment, [&](const auto& position) {
        _rows[write_offset] = Row{RowID{chunk_id, position.chunk_offset()}, position.is_null(), position.value()};
        ++write_offset;
      });
    }
  }

  std::vector<RowIDPosList> _write_pos_lists() const {
    const auto row_count = _rows.size();
    const auto chunk_size = static_cast<size_t>(_output_chunk_size);
    auto pos_lists = std::vector<RowIDPosList>(div_ceil(row_count, chunk_size));
    for (auto pos_list_index = size_t{0}; pos_list_index < pos_lists.size(); ++pos_list_index) {
      const auto begin = pos_list_index * chunk_size;
      const auto end = std::min(begin + chunk_size, row_count);
      auto& pos_list = pos_lists[pos_list_index];
      pos_list.reserve(end - begin);
      for (auto index = begin; index < end; ++index) {
        pos_list.emplace_back(std::get<0>(_rows[index]));
      }
    }
    return pos_lists;
  }

  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::shared_ptr<const Table> _table_in;
  const std::vector<SortColumnDefinition>& _sort_definitions;
  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ChunkOffset _output_chunk_size;
  std::vector<Row> _rows;
};

Sort::Sort(const std::shared_ptr<const AbstractOperator>& input_operator,
           const std::vector<SortColumnDefinition>& sort_definitions, const ChunkOffset output_chunk_size,
           const ForceMaterialization force_materialization)
    : AbstractReadOnlyOperator(OperatorType::Sort, input_operator, nullptr,
                               std::make_unique<OperatorPerformanceData<OperatorSteps>>()),
      _sort_definitions(sort_definitions),
      _output_chunk_size(output_chunk_size),
      _force_materialization(force_materialization) {
  DebugAssert(!_sort_definitions.empty(), "Expected at least one sort criterion");
}

const std::vector<SortColumnDefinition>& Sort::sort_definitions() const {
  return _sort_definitions;
}

const std::string& Sort::name() const {
  static const auto name = std::string{"Sort"};
  return name;
}

std::shared_ptr<AbstractOperator> Sort::_on_deep_copy(
    const std::shared_ptr<AbstractOperator>& copied_left_input,
    const std::shared_ptr<AbstractOperator>& /*copied_right_input*/,
    std::unordered_map<const AbstractOperator*, std::shared_ptr<AbstractOperator>>& /*copied_ops*/) const {
  return std::make_shared<Sort>(copied_left_input, _sort_definitions, _output_chunk_size, _force_materialization);
}

void Sort::_on_set_parameters(const std::unordered_map<ParameterID, AllTypeVariant>& /*parameters*/) {}

std::shared_ptr<const Table> Sort::_on_execute() {
  const auto& input_table = left_input_table();

  for (const auto& column_sort_definition : _sort_definitions) {
    Assert(column_sort_definition.column != INVALID_COLUMN_ID, "Sort: Invalid column in sort definition");
    Assert(column_sort_definition.column < input_table->column_count(),
           "Sort: Column ID is greater than table's column count");
  }

  if (input_table->row_count() == 0) {
    if (_force_materialization == ForceMaterialization::Yes && input_table->type() == TableType::References) {
      return Table::create_dummy_table(input_table->column_definitions());
    }
    return input_table;
  }

  auto pos_lists = std::vector<RowIDPosList>{};
  auto key_materialization_time = std::chrono::nanoseconds{};
  auto sort_time = std::chrono::nanoseconds{};
  auto result_materialization_time = std::chrono::nanoseconds{};

  const auto first_column_data_type = input_table->column_data_type(_sort_definitions[0].column);
  resolve_data_type(first_column_data_type, [&](auto type) {
    using ColumnDataType = typename decltype(type)::type;

    auto sort_impl = SortImpl<ColumnDataType>(input_table, _sort_definitions, _output_chunk_size);
    pos_lists = sort_impl.sort();

    key_materialization_time = sort_impl.key_materialization_time;
    sort_time = sort_impl.sort_time;
    result_materialization_time = sort_impl.result_materialization_time;
  });

  // MergeTime and MergePath are left unset: subsort neither generates runs nor merges them. Unset steps read back
  // as 0, which is the intended reporting for routines that do not have these phases.
  auto& step_performance_data = dynamic_cast<OperatorPerformanceData<OperatorSteps>&>(*performance_data);
  step_performance_data.set_step_runtime(OperatorSteps::KeyMaterialization, key_materialization_time);
  step_performance_data.set_step_runtime(OperatorSteps::Sort, sort_time);

  // We have to materialize the output (i.e., write ValueSegments) if
  //  (a) it is requested by the user,
  //  (b) a column in the table references multiple tables (see write_reference_output_table for details), or
  //  (c) a column in the table references multiple columns in the same table (which is an unlikely edge case).
  // Cases (b) and (c) can only occur if there is more than one ReferenceSegment in an input chunk.
  auto timer = Timer{};
  auto must_materialize = _force_materialization == ForceMaterialization::Yes;
  const auto input_chunk_count = input_table->chunk_count();
  if (!must_materialize && input_table->type() == TableType::References && input_chunk_count > 1) {
    const auto input_column_count = input_table->column_count();

    for (auto input_column_id = ColumnID{0}; input_column_id < input_column_count; ++input_column_id) {
      const auto& first_segment = input_table->get_chunk(ChunkID{0})->get_segment(input_column_id);
      const auto& first_reference_segment = static_cast<ReferenceSegment&>(*first_segment);

      const auto& common_referenced_table = first_reference_segment.referenced_table();
      const auto& common_referenced_column_id = first_reference_segment.referenced_column_id();

      for (auto input_chunk_id = ChunkID{1}; input_chunk_id < input_chunk_count; ++input_chunk_id) {
        const auto& segment = input_table->get_chunk(input_chunk_id)->get_segment(input_column_id);
        const auto& referenced_table = static_cast<ReferenceSegment&>(*segment).referenced_table();
        const auto& referenced_column_id = static_cast<ReferenceSegment&>(*segment).referenced_column_id();

        if (common_referenced_table != referenced_table || common_referenced_column_id != referenced_column_id) {
          must_materialize = true;
          break;
        }
      }
      if (must_materialize) {
        break;
      }
    }
  }

  auto sorted_table = must_materialize
                          ? write_materialized_output_table(input_table, std::move(pos_lists), _output_chunk_size)
                          : write_reference_output_table(input_table, std::move(pos_lists), _output_chunk_size);

  // The first sort definition is the most significant one.
  const auto& final_sort_definition = _sort_definitions[0];
  const auto output_chunk_count = sorted_table->chunk_count();
  for (auto output_chunk_id = ChunkID{0}; output_chunk_id < output_chunk_count; ++output_chunk_id) {
    const auto& output_chunk = sorted_table->get_chunk(output_chunk_id);
    output_chunk->set_immutable();
    output_chunk->set_individually_sorted_by(final_sort_definition);
  }

  // ResultMaterialization covers the position lists and the output table (the payload gather).
  step_performance_data.set_step_runtime(OperatorSteps::ResultMaterialization,
                                         result_materialization_time + timer.lap());
  return sorted_table;
}

}  // namespace hyrise