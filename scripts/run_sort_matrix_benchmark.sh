#!/usr/bin/env bash
# Run the SortMatrix micro-benchmarks (src/benchmark/operators/sort_matrix_benchmark.cpp) and save the JSON tagged
# with a label. Workflow: check out a sort implementation, rebuild hyriseMicroBenchmarks, run
#   ./run_sort_matrix_benchmark.sh <label> [path-to-binary]
# and repeat per implementation.

set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
  echo "usage: $0 <label> [path-to-hyriseMicroBenchmarks]" >&2
  echo "  <label> tags the output file, e.g. kway_radix / ips4o" >&2
  exit 1
fi

BENCH_BIN="${2:-./hyriseMicroBenchmarks}"
FILTER="${FILTER:-SortMatrix/}"
REPETITIONS="${REPETITIONS:-5}"
OUT_DIR="${OUT_DIR:-results}"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "error: benchmark binary not found or not executable: $BENCH_BIN" >&2
  echo "       pass the path as the 2nd argument, e.g. $0 $LABEL build/hyriseMicroBenchmarks" >&2
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
