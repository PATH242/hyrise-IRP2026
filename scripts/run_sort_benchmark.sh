#!/usr/bin/env bash
# Run the merge-sort micro-benchmarks and save the JSON tagged with a config label.
#
# The merge strategy is fixed per sort.cpp variant (sort_kway_merge.cpp / sort_binary_merge.cpp) and the run-generation
# strategy (radix vs. pdqsort) is the USE_RADIX_SORT toggle, so the workflow is:
#   1. pick the sort.cpp variant + set USE_RADIX_SORT, rebuild hyriseMicroBenchmarks
#   2. ./run_benchmark.sh <label> [path-to-binary]
#   3. repeat for each config: kway_radix, kway_pdq, binary_radix, binary_pdq
#   4. python3 plot_results.py
#


set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
  echo "usage: $0 <label> [path-to-hyriseMicroBenchmarks]" >&2
  echo "  <label> tags the output file, e.g. kway_radix / binary_pdq" >&2
  exit 1
fi

BENCH_BIN="${2:-./hyriseMicroBenchmarks}"
FILTER="${FILTER:-BM_MergeSort}"
REPETITIONS="${REPETITIONS:-5}"
OUT_DIR="${OUT_DIR:-results}"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "error: benchmark binary not found or not executable: $BENCH_BIN" >&2
  echo "       pass the path as the 2nd argument, e.g. ./run_benchmark.sh $LABEL build/hyriseMicroBenchmarks" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/$LABEL.json"

echo "running filter='$FILTER' repetitions=$REPETITIONS -> $OUT_FILE"
"$BENCH_BIN" \
  --benchmark_filter="$FILTER" \
  --benchmark_repetitions="$REPETITIONS" \
  --benchmark_out="$OUT_FILE" \
  --benchmark_out_format=json

echo "done: $OUT_FILE"