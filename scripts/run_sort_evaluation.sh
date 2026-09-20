\#!/usr/bin/env bash
# =====================================================================================================================
# Drives sort_evaluation over the run matrix for ONE routine branch and files the output.
#
#   ./run_evaluation.sh --routine kway
#   ./run_evaluation.sh --routine baseline --dry-run
#   ./run_evaluation.sh --routine kway --sf 10                      # single scale factor
#
# One invocation = one checked-out branch = one routine. Run it once per branch; results accumulate in a combined CSV.
#
# Table caches live in a SHARED working directory (Hyrise's table cache is CWD-relative). The first run for a given
# scale factor generates the tables; every later run -- including runs of other routines -- loads the identical cached
# bytes. That is what makes the input provably the same across routines.
#
# Encoding is not an axis: the binary leaves BenchmarkConfig at its default, i.e. Hyrise's "Automatic" per-column
# selection, the same thing hyriseBenchmarkTPCH does by default.
# =====================================================================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- configuration ---------------------------------------------------------------------------------------------
HYRISE_ROOT="${HYRISE_ROOT:-${HOME}/hyrise-IRP2026}"
BINARY="${BINARY:-${HYRISE_ROOT}/cmake-build-release/SortEvaluation}"

NUMA_NODE="${NUMA_NODE:-1}"
RUNS="${RUNS:-11}"                       # first run is a discarded warm-up
PAYLOAD="${PAYLOAD:-1,4,8,all}"
HARNESS="${HARNESS:-operator,sql}"

# Run matrix: scale factors. Add 100 here when SF 100 becomes feasible.
SCALE_FACTORS=(1 10)

WORK_ROOT="${WORK_ROOT:-${HERE}/work}"          # shared table cache directory
RESULTS_ROOT="${RESULTS_ROOT:-${HERE}/results}"
COMBINED_CSV="${RESULTS_ROOT}/all_results.csv"

# --- argument parsing ------------------------------------------------------------------------------------------
ROUTINE=""
DRY_RUN=0
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --routine)     ROUTINE="$2"; shift 2 ;;
    --sf)          IFS=',' read -r -a SCALE_FACTORS <<< "$2"; shift 2 ;;
    --binary)      BINARY="$2"; shift 2 ;;
    --runs)        RUNS="$2"; shift 2 ;;
    --payload)     PAYLOAD="$2"; shift 2 ;;
    --harness)     HARNESS="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     sed -n '3,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[eval]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[eval]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[eval]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$ROUTINE" ]] || die "--routine is required (it is what identifies this implementation in the CSV)"
[[ -x "$BINARY" ]]  || die "binary not found or not executable: $BINARY"
[[ -d "$HYRISE_ROOT/.git" ]] || die "not a git repo: $HYRISE_ROOT"
command -v numactl >/dev/null || die "numactl not found"

# --- provenance ------------------------------------------------------------------------------------------------
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
RUN_CSV="${RUN_DIR}/sort_results.csv"
MANIFEST="${RUN_DIR}/manifest.txt"

log "routine   : ${ROUTINE}"
log "source    : ${BRANCH} @ ${COMMIT}"
log "binary    : ${BINARY}"
log "machine   : ${MACHINE}, NUMA node ${NUMA_NODE}"
log "scale f.  : ${SCALE_FACTORS[*]}"
log "payload   : ${PAYLOAD}   harness: ${HARNESS}   runs: ${RUNS}"
log "output    : ${RUN_CSV}"

if [[ $DRY_RUN -eq 1 ]]; then
  for sf in "${SCALE_FACTORS[@]}"; do
    echo "  DRY: (cd ${WORK_ROOT} && numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} ${BINARY}" \
         "--routine ${ROUTINE} --sf ${sf} --payload ${PAYLOAD} --harness ${HARNESS}" \
         "--runs ${RUNS} --out ${RUN_CSV} --branch ${BRANCH} --commit ${COMMIT} --machine ${MACHINE})"
  done
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
  echo "binary         : ${BINARY}"
  echo "binary_sha256  : $(sha256sum "$BINARY" | cut -d' ' -f1)"
  echo "binary_mtime   : $(date -u -r "$BINARY" +%Y-%m-%dT%H:%M:%SZ)"
  echo "scale_factors  : ${SCALE_FACTORS[*]}"
  echo "payload_widths : ${PAYLOAD}"
  echo "harnesses      : ${HARNESS}"
  echo "runs_per_cell  : ${RUNS} (first discarded as warm-up)"
  echo "work_root      : ${WORK_ROOT}"
  echo
  echo "--- sort phase attribution on this branch ---"
  git -C "$HYRISE_ROOT" grep -n 'set_step_runtime' -- src/lib/operators/sort.cpp || echo "(no set_step_runtime found)"
} > "$MANIFEST"

log "manifest  : ${MANIFEST}"

# --- run -------------------------------------------------------------------------------------------------------
failures=0
for sf in "${SCALE_FACTORS[@]}"; do
  cell_log="${RUN_DIR}/cell_sf${sf}.log"

  mkdir -p "$WORK_ROOT"
  log "cell sf=${sf}  (cwd ${WORK_ROOT})"

  ( cd "$WORK_ROOT" && numactl "--cpunodebind=${NUMA_NODE}" "--membind=${NUMA_NODE}" "$BINARY" \
      --routine "$ROUTINE" \
      --sf "$sf" \
      --payload "$PAYLOAD" \
      --harness "$HARNESS" \
      --runs "$RUNS" \
      --out "$RUN_CSV" \
      --branch "$BRANCH" \
      --commit "$COMMIT" \
      --machine "$MACHINE" ) > "$cell_log" 2>&1

  if [[ $? -ne 0 ]]; then
    warn "FAILED: sf=${sf} -- see ${cell_log}"
    failures=$((failures + 1))
  fi
done

# --- combine ---------------------------------------------------------------------------------------------------
if [[ -s "$RUN_CSV" ]]; then
  if [[ ! -s "$COMBINED_CSV" ]]; then
    head -1 "$RUN_CSV" > "$COMBINED_CSV"
  fi
  tail -n +2 "$RUN_CSV" >> "$COMBINED_CSV"
  log "appended $(( $(wc -l < "$RUN_CSV") - 1 )) rows to ${COMBINED_CSV}"
else
  warn "no rows produced"
fi

if [[ $failures -gt 0 ]]; then
  die "${failures} cell(s) failed"
fi

log "done."