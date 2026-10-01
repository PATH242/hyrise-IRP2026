#!/usr/bin/env bash
# =====================================================================================================================
# Drives BOTH sort evaluations for ONE routine branch and files the output.
#
#   ./run_evaluation.sh --routine kway                      # TPC-H + TPC-DS
#   ./run_evaluation.sh --routine kway --benchmark tpcds    # one benchmark
#   ./run_evaluation.sh --routine kway --experiment catalog_sales
#   ./run_evaluation.sh --prepare                           # build the TPC-DS table caches, unbound, then exit
#   ./run_evaluation.sh --routine baseline --dry-run
#
# Two binaries, two axes:
#   SortEvaluationTPCH        TPC-H lineitem, ORDER BY l_shipdate, PAYLOAD 1 and 4 columns (DuckDB "Sorting Again" analog)
#   SortEvaluationTPCDS  TPC-DS, KEY column count swept, payload fixed at 1 column  (Kuiper et al., ICDE 2023)
#
# One invocation = one checked-out branch = one routine. Run it once per branch; results accumulate in a combined CSV.
#
# Table caches live in a SHARED working directory (both binaries resolve their caches relative to CWD). The first run
# for a given scale factor generates; every later run -- including runs of the other routines -- loads the identical
# cached bytes. That is what makes the input provably the same across routines.
#
# --prepare keeps the CPU binding but RELAXES the memory binding. Generation materialises the full 34-column
# catalog_sales before projecting it to the five columns an experiment needs, so its peak memory is far above steady
# state; under a strict --membind a node-local OOM is possible even with hundreds of GB free elsewhere. Generation is
# not measured, so letting its pages land on other nodes costs nothing -- and --cpunodebind still keeps every thread
# off the other users cores. Set PREPARE_MEMBIND=1 to make it strict anyway.
#
# Encoding is not an axis in either binary: both leave BenchmarkConfig at its default, i.e. Hyrise's "Automatic"
# per-column selection.
# =====================================================================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- configuration ---------------------------------------------------------------------------------------------
HYRISE_ROOT="${HYRISE_ROOT:-${HOME}/hyrise-IRP2026}"
BUILD_DIR="${BUILD_DIR:-${HYRISE_ROOT}/cmake-build-release}"
BINARY_TPCH="${BINARY_TPCH:-${BUILD_DIR}/SortEvaluationTPCH}"
BINARY_TPCDS="${BINARY_TPCDS:-${BUILD_DIR}/SortEvaluationTPCDS}"

NUMA_NODE="${NUMA_NODE:-1}"
RUNS="${RUNS:-6}"                        # 1 discarded warm-up + 5 measured, as in both reference papers
HARNESS="${HARNESS:-operator,sql}"       # TPC-H only; TPC-DS is operator-only by design

BENCHMARKS=(tpch tpcds)

# Memory binding for --prepare. Empty means "CPU stays on NUMA_NODE, memory may spill" -- see the note above.
PREPARE_MEMBIND="${PREPARE_MEMBIND:-}"

# TPC-H: narrow vs. wide payload on lineitem.
TPCH_SCALE_FACTORS=(1 10)
TPCH_PAYLOAD="${TPCH_PAYLOAD:-1,4}"   # narrow + wide, not a full sweep; 8/all only interpolate

# TPC-DS: key-count sweep. Each experiment carries its own paper-matched scale factors and key counts inside the
# binary, so the script does not duplicate them -- one invocation per experiment.
TPCDS_EXPERIMENTS=(catalog_sales customer_int customer_str)

WORK_ROOT="${WORK_ROOT:-${HERE}/work}"          # shared table caches (TPC-H binary cache + TPC-DS projected cache)
RESULTS_ROOT="${RESULTS_ROOT:-${HERE}/results}"

# --- argument parsing ------------------------------------------------------------------------------------------
ROUTINE=""
DRY_RUN=0
ALLOW_DIRTY=0
PREPARE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --routine)        ROUTINE="$2"; shift 2 ;;
    --benchmark)      IFS=',' read -r -a BENCHMARKS <<< "$2"; shift 2 ;;
    --sf)             IFS=',' read -r -a TPCH_SCALE_FACTORS <<< "$2"; shift 2 ;;
    --experiment)     IFS=',' read -r -a TPCDS_EXPERIMENTS <<< "$2"; shift 2 ;;
    --payload)        TPCH_PAYLOAD="$2"; shift 2 ;;
    --harness)        HARNESS="$2"; shift 2 ;;
    --runs)           RUNS="$2"; shift 2 ;;
    --binary-tpch)    BINARY_TPCH="$2"; shift 2 ;;
    --binary-tpcds)   BINARY_TPCDS="$2"; shift 2 ;;
    --prepare)        PREPARE=1; shift ;;
    --allow-dirty)    ALLOW_DIRTY=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        sed -n '3,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[eval]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[eval]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[eval]\033[0m %s\n' "$*" >&2; exit 1; }

wants() { local needle="$1"; local item; for item in "${BENCHMARKS[@]}"; do [[ "$item" == "$needle" ]] && return 0; done; return 1; }

command -v numactl >/dev/null || die "numactl not found"
mkdir -p "$WORK_ROOT"

# Hyrise hardcodes dsdgen's distribution file as the RELATIVE path resources/benchmark/tpcds/tpcds.idx, resolved
# against the process working directory. We deliberately run from WORK_ROOT (outside the repo) so nothing is written
# into it, so link the repo's read-only resources directory in. TPC-H does not need this -- its dbgen is vendored.
link_tpcds_resources() {
  local target="${HYRISE_ROOT}/resources"
  [[ -d "$target" ]] || die "no resources directory at ${target} -- is HYRISE_ROOT correct?"
  [[ -f "${target}/benchmark/tpcds/tpcds.idx" ]] || die "missing ${target}/benchmark/tpcds/tpcds.idx"
  ln -sfn "$target" "${WORK_ROOT}/resources"
}

# --- prepare mode: populate the TPC-DS caches, unbound, then exit ------------------------------------------------
if [[ $PREPARE -eq 1 ]]; then
  [[ -x "$BINARY_TPCDS" ]] || die "binary not found or not executable: $BINARY_TPCDS"
  link_tpcds_resources
  PREPARE_NUMACTL=(numactl "--cpunodebind=${NUMA_NODE}")
  if [[ -n "$PREPARE_MEMBIND" ]]; then
    PREPARE_NUMACTL+=("--membind=${PREPARE_MEMBIND}")
    log "preparing TPC-DS caches: CPU on node ${NUMA_NODE}, memory STRICTLY on ${PREPARE_MEMBIND}"
  else
    log "preparing TPC-DS caches: CPU on node ${NUMA_NODE}, memory may spill to other nodes (generation is not measured)"
  fi
  for experiment in "${TPCDS_EXPERIMENTS[@]}"; do
    log "prepare: ${experiment}"
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  DRY: (cd ${WORK_ROOT} && ${PREPARE_NUMACTL[*]} ${BINARY_TPCDS} --generate-only --experiment ${experiment} --cache-dir ${WORK_ROOT}/tpcds_sort_cache)" >&2
      continue
    fi
    ( cd "$WORK_ROOT" && "${PREPARE_NUMACTL[@]}" "$BINARY_TPCDS" --generate-only --experiment "$experiment" \
        --cache-dir "${WORK_ROOT}/tpcds_sort_cache" ) \
      || die "generation failed for ${experiment}"
  done
  log "caches ready under ${WORK_ROOT}/tpcds_sort_cache"
  exit 0
fi

# --- preflight ---------------------------------------------------------------------------------------------------
[[ -n "$ROUTINE" ]] || die "--routine is required (it is what identifies this implementation in the CSV)"
[[ -d "$HYRISE_ROOT/.git" ]] || die "not a git repo: $HYRISE_ROOT"
wants tpch  && { [[ -x "$BINARY_TPCH"  ]] || die "binary not found or not executable: $BINARY_TPCH"; }
wants tpcds && { [[ -x "$BINARY_TPCDS" ]] || die "binary not found or not executable: $BINARY_TPCDS"; }

# --- provenance ----------------------------------------------------------------------------------------------------
BRANCH="$(git -C "$HYRISE_ROOT" rev-parse --abbrev-ref HEAD)"
COMMIT="$(git -C "$HYRISE_ROOT" rev-parse --short HEAD)"
MACHINE="$(hostname -s)"

if [[ -n "$(git -C "$HYRISE_ROOT" status --porcelain)" ]]; then
  if [[ $ALLOW_DIRTY -eq 0 ]]; then
    die "working tree is dirty -- results would not be reproducible from ${BRANCH}@${COMMIT}. Commit, stash, or pass --allow-dirty."
  fi
  warn "working tree is dirty; rows will be tagged ${COMMIT}-dirty"
  COMMIT="${COMMIT}-dirty"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_ROOT}/${ROUTINE}/${STAMP}"
# One CSV per benchmark: the two binaries emit different headers (the TPC-DS one adds BENCHMARK, KEY_SET, KEY_COLS),
# and each only writes a header into an EMPTY file -- so sharing one file would silently append 17-field rows under a
# 16-field header. merge_results.py aligns them by column name afterwards.
RUN_CSV_TPCH="${RUN_DIR}/sort_results_tpch.csv"
RUN_CSV_TPCDS="${RUN_DIR}/sort_results_tpcds.csv"
MANIFEST="${RUN_DIR}/manifest.txt"

log "routine    : ${ROUTINE}"
log "source     : ${BRANCH} @ ${COMMIT}"
log "machine    : ${MACHINE}, NUMA node ${NUMA_NODE}"
log "benchmarks : ${BENCHMARKS[*]}"
wants tpch  && log "tpch       : sf ${TPCH_SCALE_FACTORS[*]}  payload ${TPCH_PAYLOAD}  harness ${HARNESS}"
wants tpcds && log "tpcds      : ${TPCDS_EXPERIMENTS[*]}  (scale factors and key counts come from the binary)"
log "runs       : ${RUNS} (1 warm-up)"
wants tpch  && log "out (tpch) : ${RUN_CSV_TPCH}"
wants tpcds && log "out (tpcds): ${RUN_CSV_TPCDS}"

NUMACTL=(numactl "--cpunodebind=${NUMA_NODE}" "--membind=${NUMA_NODE}")

if [[ $DRY_RUN -eq 1 ]]; then
  if wants tpch; then
    for sf in "${TPCH_SCALE_FACTORS[@]}"; do
      echo "  DRY: (cd ${WORK_ROOT} && ${NUMACTL[*]} ${BINARY_TPCH} --routine ${ROUTINE} --sf ${sf}" \
           "--payload ${TPCH_PAYLOAD} --harness ${HARNESS} --runs ${RUNS} --out ${RUN_CSV_TPCH}" \
           "--branch ${BRANCH} --commit ${COMMIT} --machine ${MACHINE})" >&2
    done
  fi
  if wants tpcds; then
    for experiment in "${TPCDS_EXPERIMENTS[@]}"; do
      echo "  DRY: (cd ${WORK_ROOT} && ${NUMACTL[*]} ${BINARY_TPCDS} --routine ${ROUTINE}" \
           "--experiment ${experiment} --cache-dir ${WORK_ROOT}/tpcds_sort_cache --runs ${RUNS} --out ${RUN_CSV_TPCDS}" \
           "--branch ${BRANCH} --commit ${COMMIT} --machine ${MACHINE})" >&2
    done
  fi
  exit 0
fi

mkdir -p "$RUN_DIR" "$RESULTS_ROOT"

# --- manifest --------------------------------------------------------------------------------------------------
{
  echo "routine        : ${ROUTINE}"
  echo "branch         : ${BRANCH}"
  echo "commit         : ${COMMIT}"
  echo "commit_subject : $(git -C "$HYRISE_ROOT" log -1 --pretty=%s)"
  echo "machine        : ${MACHINE}"
  echo "numa_node      : ${NUMA_NODE}"
  echo "started_utc    : ${STAMP}"
  echo "benchmarks     : ${BENCHMARKS[*]}"
  wants tpch  && echo "binary_tpch    : ${BINARY_TPCH}"
  wants tpch  && echo "sha256_tpch    : $(sha256sum "$BINARY_TPCH" | cut -d' ' -f1)"
  wants tpcds && echo "binary_tpcds   : ${BINARY_TPCDS}"
  wants tpcds && echo "sha256_tpcds   : $(sha256sum "$BINARY_TPCDS" | cut -d' ' -f1)"
  wants tpch  && echo "tpch_scale     : ${TPCH_SCALE_FACTORS[*]}"
  wants tpch  && echo "tpch_payload   : ${TPCH_PAYLOAD}"
  wants tpch  && echo "tpch_harness   : ${HARNESS}"
  wants tpcds && echo "tpcds_experim. : ${TPCDS_EXPERIMENTS[*]}"
  echo "runs_per_cell  : ${RUNS} (first discarded as warm-up)"
  echo "work_root      : ${WORK_ROOT}"
  echo
  echo "--- sort phase attribution on this branch ---"
  git -C "$HYRISE_ROOT" grep -n 'set_step_runtime' -- src/lib/operators/sort.cpp || echo "(no set_step_runtime found)"
} > "$MANIFEST"

log "manifest   : ${MANIFEST}"

# --- run -----------------------------------------------------------------------------------------------------------
failures=0

run_cell() {  # run_cell <log name> <command...>
  local cell_log="${RUN_DIR}/$1.log"; shift
  ( cd "$WORK_ROOT" && "${NUMACTL[@]}" "$@" ) > "$cell_log" 2>&1
  if [[ $? -ne 0 ]]; then
    warn "FAILED: $(basename "$cell_log" .log) -- see ${cell_log}"
    failures=$((failures + 1))
  fi
}

if wants tpch; then
  for sf in "${TPCH_SCALE_FACTORS[@]}"; do
    log "tpch  sf=${sf} payload=${TPCH_PAYLOAD}"
    run_cell "tpch_sf${sf}" "$BINARY_TPCH" \
      --routine "$ROUTINE" --sf "$sf" --payload "$TPCH_PAYLOAD" --harness "$HARNESS" \
      --runs "$RUNS" --out "$RUN_CSV_TPCH" --branch "$BRANCH" --commit "$COMMIT" --machine "$MACHINE"
  done
fi

if wants tpcds; then
  link_tpcds_resources
  for experiment in "${TPCDS_EXPERIMENTS[@]}"; do
    log "tpcds ${experiment}"
    run_cell "tpcds_${experiment}" "$BINARY_TPCDS" \
      --routine "$ROUTINE" --experiment "$experiment" --cache-dir "${WORK_ROOT}/tpcds_sort_cache" \
      --runs "$RUNS" --out "$RUN_CSV_TPCDS" --branch "$BRANCH" --commit "$COMMIT" --machine "$MACHINE"
  done
fi

# --- combine ---------------------------------------------------------------------------------------------------
# Per benchmark, because the headers differ. merge_results.py turns the two into one all_results.csv, aligning by
# column name and filling what the other benchmark does not report.
combine() {  # combine <run csv> <combined csv> <label>
  [[ -s "$1" ]] || return 0
  local rows=$(( $(wc -l < "$1") - 1 ))
  if [[ ! -s "$2" ]]; then
    head -1 "$1" > "$2"
  fi
  if [[ "$(head -1 "$1")" == "$(head -1 "$2")" ]]; then
    tail -n +2 "$1" >> "$2"
    log "appended ${rows} ${3} rows to ${2}"
  else
    warn "header changed since ${2} was created; left ${rows} ${3} rows in ${1} for merge_results.py"
  fi
}

wants tpch  && combine "$RUN_CSV_TPCH"  "${RESULTS_ROOT}/all_results_tpch.csv"  "tpch"
wants tpcds && combine "$RUN_CSV_TPCDS" "${RESULTS_ROOT}/all_results_tpcds.csv" "tpcds"

log "combine both with:  python3 merge_results.py ${RESULTS_ROOT}"

if [[ $failures -gt 0 ]]; then
  die "${failures} cell(s) failed"
fi

log "done."