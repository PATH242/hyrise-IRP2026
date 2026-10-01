#!/usr/bin/env python3
"""Figures and summary tables for sort_evaluation results.

    python3 plot_results.py --csv results/all_results.csv --outdir figures/

Produces, for every scale factor present in the data:

    fig1_payload_sweep      runtime vs payload width, one line per routine    [the headline figure]
    fig2_phase_breakdown    stacked phase composition per routine
    fig3_speedup            speedup over the baseline routine
    fig5_operator_vs_sql    does the operator-level win survive a real plan  [if sql rows present]

  TPC-DS -- key-count sweep, payload fixed at one column (Kuiper et al., ICDE 2023):
    fig6_tpcds_experiments  every experiment at its full key width            [their Fig. 12]
    fig7_key_sweep          runtime vs number of key columns                  [their Fig. 11]
    fig8_key_phases         where the growth goes: MATERIALIZE vs SORT
    fig9_key_speedup        speedup over baseline across the key sweep
    summary.csv / summary.md   median, spread and speedup per configuration  [the table view]

Phases are self-adapting. Columns are discovered from the CSV header (every `*_US` except TOTAL_US),
so adding MERGE_PATH_US on the C++ side needs no change here; a phase that no run in the data
reports is dropped rather than drawn as an empty legend entry, and a missing value for one run is
treated as absent rather than breaking the stack.

The two benchmarks sweep opposite axes -- TPC-H varies the payload with a one-column key, TPC-DS varies the key
count with a one-column payload -- so no figure mixes them. Rows without BENCHMARK / KEY_SET / KEY_COLS (runs from
the TPC-H binary, which predates those columns) are read as tpch / shipdate / 1 key column.

Encoding is not an axis and is ignored even if the column is present.

Requires: pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}. Install with: pip install pandas matplotlib")


# =====================================================================================================================
# Style. Palette is the validated categorical order (light mode): all-pairs clean for the three routine slots,
# adjacent-pairs clean for the five phase slots. Do not re-order or substitute without re-validating.
# =====================================================================================================================
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2df"

# Colour follows the ENTITY: a routine keeps its hue whether or not the others are in the plot, and figure order
# follows this dict, so the sequence below is also the adjacency the palette was validated against.
#
# Seaborn / matplotlib tab10, minus the two slots this project rejected (orange and pink), reordered for CVD.
# tab10 AS SHIPPED does not survive the validator -- its orange and green are dE 0.7 apart under protanopia, i.e.
# the same colour to a red-blind reader, and orange and pink are its two faintest slots. Dropping those two and
# ordering the rest fixes it:
#   blue, red, cyan, olive, purple, green
#   lightness band PASS · chroma floor PASS · worst adjacent CVD dE 16.6 · worst normal-vision dE 18.2
#   contrast vs surface PASS -- all six clear 3:1 against the page.
# Two hexes are tab10 darkened, NOT tab10 as shipped: cyan #17becf (2.2:1) and olive #bcbd22 (1.96:1) are the
# faintest colours in the set and wash out as bars on a white page. Darkened just far enough to clear 3:1; revert
# to the originals here if exact tab10 matters more than legibility.
# Blue is reserved for a baseline so it never collides. Add new branch tags HERE rather than leaning on the
# fallback, and re-run the dataviz palette validator if you change the order -- the ORDER is the CVD mechanism.
ROUTINE_COLORS = {
    "baseline": "#1f77b4",                # tab10 blue -- reserved
    "baseline-master": "#1f77b4",
    "parallel-binary-merge": "#d62728",   # tab10 red
    "pmerge": "#d62728",
    "k-way": "#1199a6",                   # tab10 cyan, darkened from #17becf
    "kway": "#1199a6",
    "k-way-merge": "#1199a6",
    "k-way-merge-path": "#1199a6",
    "reduckdb-2": "#969718",              # tab10 olive, darkened from #bcbd22
    "reduckdb-V1.4": "#9467bd",           # tab10 purple
    "duckdb-1": "#2ca02c",                # tab10 green
    # Slots 7 and 8 are tab10's remaining hues. They are BELOW the validated margin -- brown sits close to both red
    # and purple under deuteranopia -- so a figure carrying more than six routines should be split, not recoloured.
    "subsort": "#8c564b",                 # tab10 brown
    "ips4o": "#7f7f7f",                   # tab10 grey
}
FALLBACK_COLORS = ["#8c564b", "#7f7f7f", "#9467bd"]  # tab10 brown, grey, purple

# ---------------------------------------------------------------------------------------------------------------
# Phases. The order here is THE order the stacks are drawn in and the order the legend lists -- it is the pipeline
# order, not whatever order the columns happen to sit in the CSV, so a branch that writes its columns differently
# still produces the same figure.
#
# Colour is keyed by PHASE, not by position in the stack. That matters: MERGE_PATH_US only exists on branches that
# have the enumerator, and with position-indexed colours its absence would silently repaint Write Out.
PHASE_ORDER = ["MATERIALIZE_US", "KEY_NORMALIZE_US", "SORT_US", "MERGE_US", "MERGE_PATH_US", "WRITE_OUT_US"]

# MATERIALIZE_US is the column every CSV collected so far uses; KEY_NORMALIZE_US is accepted as a synonym so the
# C++ column can be renamed later without invalidating the results already on disk. Both carry the same label.
PHASE_LABELS = {
    "MATERIALIZE_US": "Key Normalization",
    "KEY_NORMALIZE_US": "Key Normalization",
    "SORT_US": "Sort",
    "MERGE_US": "Merge",
    "MERGE_PATH_US": "Merge Path",
    "WRITE_OUT_US": "Write Out",
}

# blue, green, yellow, grey, red -- in pipeline order, all exact tab10. Validated light-mode, adjacent pairlist:
#   worst adjacent CVD dE 11.1 · worst normal-vision dE 18.3 · lightness band PASS.
# Two deliberate exceptions to the validator's defaults. The grey slot fails the chroma floor because it IS a
# neutral (Merge Path is the odd step out). And olive stays at tab10's own #bcbd22 rather than the darkened step
# the routines use: darkening it walks it straight into tab10 green (dE 0.6 under protanopia, a hard fail), and
# unlike a routine bar a phase segment sits between two other coloured segments rather than on the white page, so
# its contrast against the page is not what makes it readable. Pass --hatch for the photocopy case.
PHASE_COLORS = {
    "MATERIALIZE_US": "#1f77b4",    # tab10 blue
    "KEY_NORMALIZE_US": "#1f77b4",
    "SORT_US": "#2ca02c",           # tab10 green
    "MERGE_US": "#bcbd22",          # tab10 olive -- the yellow slot
    "MERGE_PATH_US": "#7f7f7f",     # tab10 grey
    "WRITE_OUT_US": "#d62728",      # tab10 red
}
PHASE_FALLBACK = ["#9467bd", "#8c564b", "#17becf"]  # a phase column nobody has named yet
RESIDUAL_COLOR = "#c7c7c7"          # tab20's light grey: paler than Merge Path, so "Other" never reads as a phase

# Print relief: papers get photocopied. Hatches give the stacked segments a second, non-color channel.
PHASE_HATCHES = ["", "///", "...", "\\\\\\", "xxx", "---"]


def phase_color(column: str) -> str:
    """Colour follows the phase, never its height in the stack."""
    if column in PHASE_COLORS:
        return PHASE_COLORS[column]
    digest = sum((index + 1) * byte for index, byte in enumerate(column.encode()))
    return PHASE_FALLBACK[digest % len(PHASE_FALLBACK)]

BASE_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",  # never dashed: dashing reads as "threshold", it is just a grid
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "text.color": INK,
    "legend.frameon": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
}


def apply_style(font: str) -> None:
    rc = dict(BASE_RC)
    if font == "serif":
        # Matches a Times-set paper body. Sans is the default; this is opt-in.
        rc["font.family"] = "serif"
        rc["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif"]
    else:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams.update(rc)


def clean_axes(ax) -> None:
    # No frame at all: the y-grid carries the scale, so the box around the plot is redundant ink.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, width=0, pad=3)


# =====================================================================================================================
# Loading
# =====================================================================================================================
def phase_columns(frame: pd.DataFrame) -> list[str]:
    """Every *_US column except TOTAL_US, in PIPELINE order.

    Known phases come first in the order the operator runs them (PHASE_ORDER); anything new keeps its CSV position
    after them, so adding a phase still needs no change here -- it just lands at the end until it is named above.
    """
    found = [c for c in frame.columns if c.endswith("_US") and c != "TOTAL_US"]
    known = [c for c in PHASE_ORDER if c in found]
    return known + [c for c in found if c not in known]


def phase_label(column: str) -> str:
    return PHASE_LABELS.get(column, column[:-3].replace("_", " ").title())


def is_results_csv(path: Path) -> bool:
    try:
        with path.open() as handle:
            return handle.readline().startswith("ROUTINE,")
    except (OSError, UnicodeDecodeError):
        return False


def resolve_sources(paths: list[Path]) -> list[Path]:
    """Expand each argument into result CSVs.

    A directory expands to the result CSVs directly inside it -- NOT recursively. run_evaluation.sh keeps the
    accumulated all_results_<benchmark>.csv files at the top of results/ and each run's own copy in a per-routine
    subdirectory, so recursing would read the same measurements twice.
    """
    sources: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.glob("*.csv") if is_results_csv(p))
            if not found:
                sys.exit(f"{path}: no result CSVs directly inside (looked for a 'ROUTINE,' header)")
            sources.extend(found)
        elif path.exists():
            sources.append(path)
        else:
            sys.exit(f"no such file or directory: {path}")
    return sources


def load(csv_paths: list[Path], routine_order: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    # The two benchmarks write different headers, so concat aligns on column names rather than position.
    frames = []
    for path in csv_paths:
        piece = pd.read_csv(path)
        piece["__source"] = path.name          # kept only long enough to name files in the checks below
        print(f"  {path}: {len(piece)} rows")
        frames.append(piece)
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    before = len(frame)
    frame = frame.drop_duplicates(ignore_index=True)
    if before != len(frame):
        print(f"  dropped {before - len(frame)} duplicate row(s) -- the same measurements appeared in more than one "
              f"source")

    required = {"ROUTINE", "HARNESS", "SCALE", "PAYLOAD_COLS", "RUN_ID", "TOTAL_US"}
    missing = required - set(frame.columns)
    if missing:
        sys.exit(f"missing columns {sorted(missing)}")

    # Rows with no ROUTINE or no TOTAL_US are malformed -- a truncated write, a blank trailing line, or a stray
    # header. Drop them, but say where they came from: a run killed mid-write is worth knowing about rather than
    # quietly averaging around.
    routine_blank = frame["ROUTINE"].isna() | (frame["ROUTINE"].astype(str).str.strip().isin(["", "nan", "ROUTINE"]))
    total_blank = pd.to_numeric(frame["TOTAL_US"], errors="coerce").isna()
    broken = routine_blank | total_blank
    if broken.any():
        print(f"  ! {int(broken.sum())} malformed row(s) dropped -- no ROUTINE or no TOTAL_US:", file=sys.stderr)
        for source, count in frame[broken]["__source"].value_counts().items():
            print(f"      {source}: {count}", file=sys.stderr)
        print("    Check the tail of those files; a run killed mid-write leaves a partial line.", file=sys.stderr)
        frame = frame[~broken].reset_index(drop=True)
    if frame.empty:
        sys.exit("no usable rows left")

    frame = frame.drop(columns="__source")
    frame["ROUTINE"] = frame["ROUTINE"].astype(str)

    # One bad value makes pandas read the whole column as object, and dropping the row does not undo that -- the
    # arithmetic below would still fail on strings. Coerce every numeric column once, here.
    numeric_columns = [c for c in frame.columns if c.endswith("_US")] + [
        c for c in ("SCALE", "PAYLOAD_COLS", "RUN_ID", "KEY_COLS", "ROW_COUNT", "MATERIALIZED") if c in frame.columns
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # The TPC-H binary predates these columns; its rows are one key column (l_shipdate) on the tpch benchmark.
    # Backfilling matters beyond cosmetics: groupby drops rows with NaN keys, so a missing KEY_COLS would silently
    # delete every TPC-H row from the summary table.
    if "KEY_SET" not in frame.columns:
        frame["KEY_SET"] = "shipdate"
    frame["KEY_SET"] = frame["KEY_SET"].fillna("shipdate")

    if "BENCHMARK" not in frame.columns:
        frame["BENCHMARK"] = "tpch"
    frame["BENCHMARK"] = frame["BENCHMARK"].fillna("tpch")

    if "KEY_COLS" not in frame.columns:
        frame["KEY_COLS"] = 1
    frame["KEY_COLS"] = pd.to_numeric(frame["KEY_COLS"], errors="coerce").fillna(1).astype(int)

    # SORT_KEY holds the ordered key columns for the run, comma-joined inside one quoted field. It is what the
    # figures name so a reader never has to guess what "3 key cols" means.
    if "SORT_KEY" not in frame.columns:
        frame["SORT_KEY"] = ""
    frame["SORT_KEY"] = frame["SORT_KEY"].fillna("").astype(str)

    frame["TOTAL_MS"] = frame["TOTAL_US"] / 1000.0

    # If the C++ column is ever renamed MATERIALIZE_US -> KEY_NORMALIZE_US, a results directory will hold CSVs from
    # both sides of the rename. Concatenated, each row fills one column and gets 0 in the other -- and the stack
    # would draw BOTH, double-counting the phase. Fold them into one column instead, and say so.
    if "KEY_NORMALIZE_US" in frame.columns and "MATERIALIZE_US" in frame.columns:
        old = pd.to_numeric(frame["MATERIALIZE_US"], errors="coerce").fillna(0)
        new = pd.to_numeric(frame["KEY_NORMALIZE_US"], errors="coerce").fillna(0)
        both = int(((old > 0) & (new > 0)).sum())
        if both:
            print(f"  ! {both} row(s) report BOTH MATERIALIZE_US and KEY_NORMALIZE_US -- they are the same phase "
                  f"under two names, so one of those runs is mislabelled. Using the larger of the two.",
                  file=sys.stderr)
        frame["KEY_NORMALIZE_US"] = pd.concat([old, new], axis=1).max(axis=1)
        frame = frame.drop(columns="MATERIALIZE_US")
        print("  note: folded MATERIALIZE_US into KEY_NORMALIZE_US -- the same phase before and after the rename")

    # Flexible phases: keep only the ones some run in this data actually reports. A column that is absent,
    # all-NaN or all-zero belongs to a phase no routine here has -- drop it instead of drawing an empty
    # legend entry. Missing values within a kept column become 0 so one gap cannot break a stack.
    phases, dropped = [], []
    for column in phase_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.fillna(0).abs().sum() > 0:
            frame[column] = values.fillna(0)
            frame[column[:-3] + "_MS"] = frame[column] / 1000.0
            phases.append(column)
        else:
            dropped.append(column)

    if dropped:
        print(f"  note: no run reports {', '.join(phase_label(c) for c in dropped)}; dropped from the figures")
    if not phases:
        print("  note: no phase columns with data; skipping the phase breakdown figure", file=sys.stderr)

    if routine_order:
        known = [r for r in routine_order if r in set(frame["ROUTINE"])]
        unknown = sorted(set(frame["ROUTINE"]) - set(routine_order))
        order = known + unknown
    else:
        found = sorted(set(frame["ROUTINE"]))
        order = [r for r in ROUTINE_COLORS if r in found] + [r for r in found if r not in ROUTINE_COLORS]

    unmapped = [r for r in order if r not in ROUTINE_COLORS]
    if unmapped:
        print(f"  note: no colour mapped for {unmapped} -- using the hash fallback, which can collide. Add them to "
              f"ROUTINE_COLORS.")

    frame["ROUTINE"] = pd.Categorical(frame["ROUTINE"], categories=order, ordered=True)
    return frame, phases


def color_for(routine: str, order: list[str]) -> str:
    """Colour follows the routine, never its position.

    An unknown name is hashed rather than indexed by where it sits in `order`: indexing would repaint the survivors
    whenever a routine is missing from a run, which is exactly the failure this rule exists to prevent. Add real
    branch tags to ROUTINE_COLORS above -- the fallback can collide.
    """
    if routine in ROUTINE_COLORS:
        return ROUTINE_COLORS[routine]
    digest = sum((index + 1) * byte for index, byte in enumerate(routine.encode()))
    return FALLBACK_COLORS[digest % len(FALLBACK_COLORS)]


# =====================================================================================================================
# Naming the sort key -- "3 key cols" is meaningless on a slide unless the columns are on the figure
# =====================================================================================================================
# The sort columns' data types, as Hyrise stores them. TPC-H and TPC-DS schemas are fixed by their specs, so this
# cannot drift the way a derived table could -- but it IS a table in a plotting script, so a name that is not here
# is reported rather than silently labelled. The authoritative alternative is to have the binaries emit
# "name:type" into SORT_KEY from `table->column_data_type(...)`; the splitter below already prefers that when the
# CSV carries it, so adding it on the C++ side needs no change here.
#
# "int" rather than "int32": the width is an implementation detail of the generator, the signedness and the fact
# that it is not a string are what the figure is telling the reader.
COLUMN_TYPES = {
    # TPC-DS catalog_sales -- identifiers and a quantity, all integer in the spec
    "cs_warehouse_sk": "int", "cs_ship_mode_sk": "int", "cs_promo_sk": "int", "cs_quantity": "int",
    "cs_item_sk": "int",
    # TPC-DS customer -- the integer key set, the string key set, and the payload
    "c_birth_year": "int", "c_birth_month": "int", "c_birth_day": "int", "c_customer_sk": "int",
    "c_last_name": "string", "c_first_name": "string",
    # TPC-H lineitem -- Hyrise stores DATE columns as strings
    "l_shipdate": "string", "l_orderkey": "int", "l_linenumber": "int", "l_comment": "string",
    "l_returnflag": "string", "l_linestatus": "string", "l_quantity": "float", "l_extendedprice": "float",
}
_UNTYPED_REPORTED: set[str] = set()

# The CSV records how MANY payload columns a run carried, not which ones. Only the experiment matrix wants the
# name, and only for TPC-DS where it is one fixed column per experiment, so it lives here rather than in the
# schema. Everything else in that figure is read from the data.
PAYLOAD_COLUMNS = {
    "catalog_sales": "cs_item_sk",
    "customer_int": "c_customer_sk",
    "customer_str": "c_customer_sk",
}


def split_key_column(entry: str) -> tuple[str, str]:
    """A SORT_KEY entry as (name, type). Accepts "name" or "name:type"; the CSV wins over the table above."""
    name, _, declared = entry.partition(":")
    name = name.strip()
    data_type = declared.strip() or COLUMN_TYPES.get(name, "")
    if not data_type and name and name not in _UNTYPED_REPORTED:
        _UNTYPED_REPORTED.add(name)
        print(f"  note: no data type known for sort column {name!r}; add it to COLUMN_TYPES in this script, or "
              f"have the binary write it into SORT_KEY as {name}:<type>", file=sys.stderr)
    return name, data_type


def key_columns_of(panel: pd.DataFrame, with_types: bool = False) -> list[str]:
    """The ordered key columns of an experiment, read from the widest run in it.

    The sweep is a prefix sweep -- 1 key is the first column, 2 keys the first two -- so the widest run's SORT_KEY
    is the full list and every narrower run is a prefix of it. Taking the widest therefore names every tick at once.
    """
    if "SORT_KEY" not in panel.columns or panel.empty:
        return []
    widest = panel[panel["KEY_COLS"] == panel["KEY_COLS"].max()]
    values = [v for v in widest["SORT_KEY"].unique() if v]
    if not values:
        return []
    pairs = [split_key_column(part) for part in max(values, key=len).split(",") if part.strip()]
    if not with_types:
        return [name for name, _ in pairs]
    return [f"{name} ({data_type})" if data_type else name for name, data_type in pairs]


# Width of one 6.2pt monospace character, in inches, measured off a rendered figure. Used to wrap the caption to
# the panel it belongs to rather than to a fixed column count -- "cs_warehouse_sk (int)" is more than twice the
# width of "c_birth_day", so a fixed count overruns some panels and wastes space on others.
CAPTION_CHAR_INCHES = 0.087


def key_caption(panel: pd.DataFrame, numbered: bool = True, budget: int = 40) -> str:
    """Names the key columns in sort order, with their data types, for a panel subtitle.

    The type is on the figure because it is half the result: an int key normalizes to a fixed-width integer and a
    string key does not, which is most of why the two customer experiments behave differently at all.

    Numbered when the panel sweeps the key count, because the numbers are exactly the x-axis ticks. Packed greedily
    into lines of at most `budget` characters, so a caption never runs into the panel beside it.
    """
    columns = key_columns_of(panel, with_types=True)
    if not columns:
        return ""
    numbering = numbered and len(columns) > 1
    parts = [f"{index + 1} {name}" for index, name in enumerate(columns)] if numbering else list(columns)
    separator = "   " if numbering else ", "
    trailer = "" if numbering else ","

    lines, current = [], ""
    for part in parts:
        candidate = part if not current else current + separator + part
        if current and len(candidate) > budget:
            lines.append(current + trailer)
            current = part
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def annotate_keys(axis, panel: pd.DataFrame, numbered: bool = True, y: float = 1.02) -> int:
    """Put the key-column caption between the panel title and the plot area, flush with the y-axis.

    Returns the title pad this panel needs, so the caller can level every title in the figure to the widest one --
    per-panel pads would step the titles up and down across the row.
    """
    figure = axis.get_figure()
    width_inches = axis.get_position().width * figure.get_figwidth()
    budget = max(int(width_inches / CAPTION_CHAR_INCHES), 18)
    caption = key_caption(panel, numbered, budget)
    if not caption:
        return 6
    axis.annotate(caption, xy=(0, y), xycoords="axes fraction", ha="left", va="bottom",
                  fontsize=6.2, color=INK_MUTED, family="monospace", annotation_clip=False)
    return int(8 + 8.5 * (caption.count("\n") + 1))


def level_titles(axes_row, pads: list[int]) -> None:
    """Re-apply the largest pad to every title in the row so they share one baseline."""
    if not pads:
        return
    for axis in axes_row:
        axis.set_title(axis.get_title(), color=INK, pad=max(pads))


# =====================================================================================================================
# Data quality -- surfaced before any plotting, because a pretty plot of noisy data is worse than no plot
# =====================================================================================================================
def check_data(frame: pd.DataFrame, phases: list[str], cv_threshold: float) -> None:
    print("data checks")

    runs = frame.groupby(["ROUTINE", "HARNESS", "KEY_SET", "KEY_COLS", "SCALE", "PAYLOAD_COLS"],
                         observed=True)["TOTAL_MS"]
    stats = runs.agg(["count", "mean", "std", "median"]).reset_index()
    stats["cv_pct"] = 100 * stats["std"] / stats["mean"]

    thin = stats[stats["count"] < 3]
    if not thin.empty:
        print(f"  ! {len(thin)} configuration(s) with fewer than 3 measured runs")

    noisy = stats[stats["cv_pct"] > cv_threshold].sort_values("cv_pct", ascending=False)
    if noisy.empty:
        print(f"  ok  all configurations within {cv_threshold:.0f}% CV")
    else:
        print(f"  ! {len(noisy)} configuration(s) above {cv_threshold:.0f}% CV -- check the machine was quiet:")
        for _, row in noisy.head(8).iterrows():
            print(f"      {row['ROUTINE']:<10} {row['HARNESS']:<9} {row['KEY_SET']:<9} sf={row['SCALE']:<5}"
                  f" payload={row['PAYLOAD_COLS']:<3} CV={row['cv_pct']:.1f}%")

    if "MATERIALIZED" in frame.columns:
        unmaterialized = frame[(frame["HARNESS"] == "sql") & (frame["MATERIALIZED"] == 0)]
        if not unmaterialized.empty:
            routines = sorted(set(unmaterialized["ROUTINE"].astype(str)))
            print(f"  ! sql rows with MATERIALIZED=0 for {routines} -- the LQPTranslator ForceMaterialization patch")
            print("    was not in effect there; those rows are NOT comparable to the blog and WRITE_OUT_US is ~0")

    if phases:
        print(f"  ok  phases in use: {', '.join(phase_label(c) for c in phases)}")
        for column in phases:
            silent = sorted({str(r) for r in frame[frame[column] == 0]["ROUTINE"]}
                            - {str(r) for r in frame[frame[column] > 0]["ROUTINE"]})
            if silent:
                print(f"        {phase_label(column)}: 0 for {silent}")

        phase_ms = [c[:-3] + "_MS" for c in phases]
        residual = frame["TOTAL_MS"] - frame[phase_ms].sum(axis=1)
        share = (residual / frame["TOTAL_MS"]).median()
        if (residual < 0).any():
            print("  ! phases sum to MORE than TOTAL_US in some rows -- check for double-counted steps")
        print(f"  ok  unattributed time (operator overhead outside the timed phases): {100 * share:.1f}% median")
    print()


# =====================================================================================================================
# Aggregation
# =====================================================================================================================
GROUP = ["ROUTINE", "BENCHMARK", "HARNESS", "KEY_SET", "KEY_COLS", "SCALE", "PAYLOAD_COLS"]


def summarize(frame: pd.DataFrame, phases: list[str], baseline: str) -> pd.DataFrame:
    aggregations = {"TOTAL_MS": ["count", "median", "min", "max", "mean", "std"]}
    for column in phases:
        aggregations[column[:-3] + "_MS"] = ["median"]
    if "ROW_COUNT" in frame.columns:
        aggregations["ROW_COUNT"] = ["max"]

    table = frame.groupby(GROUP, observed=True).agg(aggregations)
    table.columns = ["_".join(c).rstrip("_") for c in table.columns]
    table = table.reset_index().rename(columns={
        "TOTAL_MS_count": "n",
        "TOTAL_MS_median": "median_ms",
        "TOTAL_MS_min": "min_ms",
        "TOTAL_MS_max": "max_ms",
        "TOTAL_MS_mean": "mean_ms",
        "ROW_COUNT_max": "rows",
    })
    table["cv_pct"] = (100 * table["TOTAL_MS_std"] / table["mean_ms"]).round(2)
    table = table.drop(columns=["TOTAL_MS_std"])

    if "rows" in table.columns:
        table["ns_per_tuple"] = (1e6 * table["median_ms"] / table["rows"]).round(1)

    key = ["BENCHMARK", "HARNESS", "KEY_SET", "KEY_COLS", "SCALE", "PAYLOAD_COLS"]
    base = table[table["ROUTINE"] == baseline].set_index(key)["median_ms"]
    table["speedup_vs_base"] = table.apply(
        lambda row: round(base.get(tuple(row[k] for k in key), float("nan")) / row["median_ms"], 3)
        if row["median_ms"] else None,
        axis=1,
    )

    numeric = ["median_ms", "min_ms", "max_ms", "mean_ms"] + [c for c in table.columns if c.endswith("_MS_median")]
    table[numeric] = table[numeric].round(2)
    return table.sort_values(GROUP)


def figure_legend(fig, axis, ncol: int | None = None, y: float = 1.0) -> None:
    """A frameless legend above the axes. Never inside them -- an in-axes legend collides with the data."""
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(labels), handlelength=1.4, columnspacing=1.6)


def plural_cols(width: int) -> str:
    return f"{width} col" if width == 1 else f"{width} cols"


def save(fig, outdir: Path, name: str, formats: list[str]) -> None:
    for extension in formats:
        path = outdir / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"  wrote {path}")
    plt.close(fig)


def scales(frame: pd.DataFrame, benchmark: str = "tpch") -> list[float]:
    """Scale factors present in one benchmark's operator data, ascending."""
    data = of_benchmark(frame, benchmark)
    return sorted(set(data[data["HARNESS"] == "operator"]["SCALE"]))


def of_benchmark(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows from one benchmark. The two have different swept axes, so no figure may mix them."""
    return frame[frame["BENCHMARK"] == name]


def sweep_experiments(frame: pd.DataFrame) -> list[str]:
    """TPC-DS key sets that actually sweep the key count (more than one KEY_COLS value)."""
    tpcds = of_benchmark(frame, "tpcds")
    if tpcds.empty:
        return []
    counts = tpcds.groupby("KEY_SET", observed=True)["KEY_COLS"].nunique()
    return sorted(counts[counts > 1].index)


def sweep_key_set(frame: pd.DataFrame) -> str:
    """The key set the payload sweep belongs to: whichever has the most payload widths, 'shipdate' breaking ties."""
    operator = of_benchmark(frame, "tpch")
    operator = operator[operator["HARNESS"] == "operator"]
    widths = operator.groupby("KEY_SET", observed=True)["PAYLOAD_COLS"].nunique()
    if widths.empty:
        return "shipdate"
    best = widths.max()
    candidates = sorted(widths[widths == best].index)
    return "shipdate" if "shipdate" in candidates else candidates[0]


# =====================================================================================================================
# Figure 1 -- runtime vs payload width. The headline: the blog's single wide-table point, as a curve.
# =====================================================================================================================
def fig_payload_sweep(frame, order, outdir, formats):
    key = sweep_key_set(frame)
    tpch = of_benchmark(frame, "tpch")
    data = tpch[(tpch["HARNESS"] == "operator") & (tpch["KEY_SET"] == key)]
    if data.empty:
        return
    panels = scales(frame)
    all_widths = sorted(set(data["PAYLOAD_COLS"]))
    # Two payload widths is a comparison of two cases, not a curve -- a two-point line invites the reader to
    # interpolate a trend that was never measured. Bars below three widths, the sweep curve at three or more.
    as_bars = len(all_widths) <= 2
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.7), squeeze=False)
    bar_width = 0.8 / max(len(order), 1)

    for axis, scale in zip(axes[0], panels):
        clean_axes(axis)
        panel = data[data["SCALE"] == scale]
        for slot, routine in enumerate(order):
            series = panel[panel["ROUTINE"] == routine]
            if series.empty:
                continue
            grouped = series.groupby("PAYLOAD_COLS", observed=True)["TOTAL_MS"]
            widths = sorted(grouped.groups)
            medians = [grouped.get_group(w).median() for w in widths]
            lows = [grouped.get_group(w).min() for w in widths]
            highs = [grouped.get_group(w).max() for w in widths]
            colour = color_for(routine, order)
            if as_bars:
                offsets = [all_widths.index(w) + (slot - (len(order) - 1) / 2) * bar_width for w in widths]
                axis.bar(offsets, medians, width=bar_width * 0.88, color=colour, edgecolor="none", linewidth=0,
                         label=routine, zorder=3)
                axis.errorbar(offsets, medians, yerr=[
                    [m - lo for m, lo in zip(medians, lows)], [hi - m for m, hi in zip(medians, highs)]],
                    fmt="none", ecolor=INK_MUTED, elinewidth=0.8, capsize=2, zorder=4)
            else:
                axis.fill_between(widths, lows, highs, color=colour, alpha=0.13, linewidth=0)
                axis.plot(widths, medians, color=colour, linewidth=2, marker="o", markersize=4.5,
                          markeredgecolor=SURFACE, markeredgewidth=1.2, label=routine, zorder=3)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_xlabel("payload columns")
        if as_bars:
            axis.set_xticks(range(len(all_widths)))
            axis.set_xticklabels([f"{w:g}" for w in all_widths])
        else:
            axis.set_xticks(all_widths)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("sort time (ms, median)")
    figure_legend(fig, axes[0][0], y=1.06)
    columns = ", ".join(key_columns_of(data)) or f"{key} key"
    fig.suptitle(f"Sort runtime vs. payload width — lineitem ORDER BY {columns}",
                 y=1.17, fontsize=9, color=INK, ha="center")
    save(fig, outdir, "fig1_payload_sweep", formats)


# =====================================================================================================================
# Figure 2 -- where the time goes. This is what the blog does not have.
# =====================================================================================================================
def fig_phase_breakdown(frame, phases, order, outdir, formats, hatch):
    if not phases:
        return
    key = sweep_key_set(frame)
    tpch = of_benchmark(frame, "tpch")
    data = tpch[(tpch["HARNESS"] == "operator") & (tpch["KEY_SET"] == key)]
    if data.empty:
        return

    widths = sorted(set(data["PAYLOAD_COLS"]))
    panels = scales(frame)
    bars_per_panel = len(widths) * max(len(order), 1)
    panel_width = max(3.4, 0.22 * bars_per_panel)
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_width * len(panels), 3.2), squeeze=False)
    phase_ms = [c[:-3] + "_MS" for c in phases]

    for axis, scale in zip(axes[0], panels):
        clean_axes(axis)
        panel = data[data["SCALE"] == scale]

        positions, labels, group_centres = [], [], []
        position = 0.0
        for width in widths:
            start = position
            for routine in order:
                series = panel[(panel["ROUTINE"] == routine) & (panel["PAYLOAD_COLS"] == width)]
                if series.empty:
                    continue
                bottom = 0.0
                for index, column in enumerate(phase_ms):
                    value = series[column].median()
                    value = 0.0 if pd.isna(value) else value
                    axis.bar(position, value, bottom=bottom, width=0.78,
                             color=phase_color(phases[index]),
                             edgecolor=SURFACE if hatch else "none", linewidth=0,
                             hatch=PHASE_HATCHES[index % len(PHASE_HATCHES)] if hatch else None,
                             label=phase_label(phases[index]) if not positions and index < len(phases) else None)
                    bottom += value
                residual = max(series["TOTAL_MS"].median() - bottom, 0.0)
                axis.bar(position, residual, bottom=bottom, width=0.78, color=RESIDUAL_COLOR,
                         edgecolor='none', linewidth=0,
                         label="Other (operator overhead)" if not positions else None)
                positions.append(position)
                labels.append(routine)
                position += 1
            group_centres.append((start + position - 1) / 2)
            position += 0.7

        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=6.5)
        for centre, width in zip(group_centres, widths):
            axis.annotate(plural_cols(width), xy=(centre, -0.44), xycoords=("data", "axes fraction"),
                          ha="center", va="top", fontsize=7, color=INK_MUTED, annotation_clip=False)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.14)
    fig.suptitle(f"Where the time goes — lineitem ORDER BY {', '.join(key_columns_of(data)) or key}",
                 y=1.24, fontsize=9, color=INK)
    save(fig, outdir, "fig2_phase_breakdown", formats)


# =====================================================================================================================
# Figure 3 -- speedup over baseline. Mirrors the blog's speedup tables.
# =====================================================================================================================
def fig_speedup(frame, order, outdir, formats, baseline):
    key = sweep_key_set(frame)
    tpch = of_benchmark(frame, "tpch")
    data = tpch[(tpch["HARNESS"] == "operator") & (tpch["KEY_SET"] == key)]
    others = [r for r in order if r != baseline]
    if data.empty or not others:
        return

    widths = sorted(set(data["PAYLOAD_COLS"]))
    panels = scales(frame)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.6), squeeze=False)
    bar_width = 0.8 / len(others)

    for axis, scale in zip(axes[0], panels):
        clean_axes(axis)
        panel = data[data["SCALE"] == scale]

        for slot, routine in enumerate(others):
            heights, offsets = [], []
            for index, width in enumerate(widths):
                cell = panel[panel["PAYLOAD_COLS"] == width]
                base = cell[cell["ROUTINE"] == baseline]["TOTAL_MS"].median()
                mine = cell[cell["ROUTINE"] == routine]["TOTAL_MS"].median()
                if pd.isna(base) or pd.isna(mine) or not mine:
                    continue
                heights.append(base / mine)
                offsets.append(index + (slot - (len(others) - 1) / 2) * bar_width)
            axis.bar(offsets, heights, width=bar_width * 0.88, color=color_for(routine, order),
                     edgecolor='none', linewidth=0, label=routine, zorder=3)
            # Selective direct labels: the value is the whole point of this figure, and there are few bars.
            for x, height in zip(offsets, heights):
                axis.annotate(f"{height:.2f}×", xy=(x, height), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)

        axis.axhline(1.0, color=INK_MUTED, linewidth=0.9, zorder=2)
        axis.set_xticks(range(len(widths)))
        axis.set_xticklabels([str(w) for w in widths])
        axis.set_xlabel("payload columns")
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel(f"speedup over {baseline} (×)")
    figure_legend(fig, axes[0][0], y=1.07)
    fig.suptitle(f"Speedup over {baseline} — higher is faster", y=1.18, fontsize=9, color=INK)
    save(fig, outdir, "fig3_speedup", formats)


# =====================================================================================================================
# Figure 5 -- does the operator-level win survive a real query plan?
# =====================================================================================================================
def fig_operator_vs_sql(frame, order, outdir, formats, baseline):
    if "sql" not in set(of_benchmark(frame, "tpch")["HARNESS"]):
        return
    key = sweep_key_set(frame)
    tpch = of_benchmark(frame, "tpch")
    sql = tpch[(tpch["HARNESS"] == "sql") & (tpch["KEY_SET"] == key)]
    operator = tpch[(tpch["HARNESS"] == "operator") & (tpch["KEY_SET"] == key)]
    if operator.empty:
        return
    operator = operator[operator["PAYLOAD_COLS"] == operator["PAYLOAD_COLS"].max()]

    scales = sorted(set(sql["SCALE"]) & set(operator["SCALE"]))
    if not scales:
        return

    # Faceted by scale factor, not grouped on one axis: SF 1 and SF 10 differ ~10x, so a shared linear axis
    # flattens SF 1 into unreadable stubs.
    harnesses = [("operator (full payload)", operator), ("sql (SELECT *)", sql)]
    fig, axes = plt.subplots(1, len(scales), figsize=(3.3 * len(scales), 2.6), squeeze=False)
    bar_width = 0.8 / max(len(order), 1)

    for axis, scale in zip(axes[0], scales):
        clean_axes(axis)
        for slot, routine in enumerate(order):
            heights, offsets = [], []
            for index, (_, data) in enumerate(harnesses):
                value = data[(data["ROUTINE"] == routine) & (data["SCALE"] == scale)]["TOTAL_MS"].median()
                if pd.isna(value):
                    continue
                heights.append(value)
                offsets.append(index + (slot - (len(order) - 1) / 2) * bar_width)
            axis.bar(offsets, heights, width=bar_width * 0.88, color=color_for(routine, order),
                     edgecolor='none', linewidth=0, label=routine, zorder=3)
        axis.set_xticks(range(len(harnesses)))
        axis.set_xticklabels([name for name, _ in harnesses], fontsize=7)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.07)
    fig.suptitle(f"Isolated operator vs. end-to-end query — ORDER BY {', '.join(key_columns_of(data)) or key}",
                 y=1.18, fontsize=9, color=INK)
    save(fig, outdir, "fig5_operator_vs_sql", formats)


# =====================================================================================================================
# TPC-DS figures. The swept axis here is the NUMBER OF KEY COLUMNS with the payload fixed at one column -- the mirror
# image of the TPC-H payload sweep above, so the two must never share a chart.
# =====================================================================================================================
def fig_key_sweep(frame, order, outdir, formats):
    """Runtime vs. number of key columns. The ICDE 2023 Figure 11 analogue, and where a per-column re-sort fans out."""
    experiments = sweep_experiments(frame)
    if not experiments:
        return
    data = of_benchmark(frame, "tpcds")
    data = data[(data["HARNESS"] == "operator") & (data["KEY_SET"].isin(experiments))]

    fig, axes = plt.subplots(1, len(experiments), figsize=(max(3.3 * len(experiments), 5.4), 2.7),
                             squeeze=False)

    pads = []
    for axis, experiment in zip(axes[0], experiments):
        clean_axes(axis)
        panel = data[data["KEY_SET"] == experiment]
        key_counts = sorted(set(panel["KEY_COLS"]))
        for routine in order:
            series = panel[panel["ROUTINE"] == routine]
            if series.empty:
                continue
            grouped = series.groupby("KEY_COLS", observed=True)["TOTAL_MS"]
            counts = [c for c in key_counts if c in grouped.groups]
            medians = [grouped.get_group(c).median() for c in counts]
            lows = [grouped.get_group(c).min() for c in counts]
            highs = [grouped.get_group(c).max() for c in counts]
            colour = color_for(routine, order)
            axis.fill_between(counts, lows, highs, color=colour, alpha=0.13, linewidth=0)
            axis.plot(counts, medians, color=colour, linewidth=2, marker="o", markersize=4.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label=routine, zorder=3)
        sf = sorted(set(panel["SCALE"]))
        axis.set_title(f"{experiment}" + (f" · SF {sf[0]:g}" if len(sf) == 1 else ""), color=INK, pad=16)
        pads.append(annotate_keys(axis, panel))
        axis.set_xlabel("sort key columns")
        axis.set_xticks(key_counts)
        axis.set_ylim(bottom=0)

    level_titles(axes[0], pads)
    axes[0][0].set_ylabel("sort time (ms, median)")
    figure_legend(fig, axes[0][0], y=1.22)
    fig.suptitle("Sort runtime vs. number of key columns — TPC-DS", y=1.34, fontsize=9, color=INK)
    save(fig, outdir, "fig7_key_sweep", formats)


def fig_key_speedup(frame, order, outdir, formats, baseline):
    """Speedup over baseline across the key sweep -- the single-pass vs. per-column-re-sort claim, as a ratio."""
    experiments = sweep_experiments(frame)
    others = [r for r in order if r != baseline]
    if not experiments or not others:
        return
    data = of_benchmark(frame, "tpcds")
    data = data[(data["HARNESS"] == "operator") & (data["KEY_SET"].isin(experiments))]

    fig, axes = plt.subplots(1, len(experiments), figsize=(max(3.3 * len(experiments), 5.4), 2.6),
                             squeeze=False)
    bar_width = 0.8 / len(others)

    pads = []
    for axis, experiment in zip(axes[0], experiments):
        clean_axes(axis)
        panel = data[data["KEY_SET"] == experiment]
        key_counts = sorted(set(panel["KEY_COLS"]))

        for slot, routine in enumerate(others):
            heights, offsets = [], []
            for index, count in enumerate(key_counts):
                cell = panel[panel["KEY_COLS"] == count]
                base = cell[cell["ROUTINE"] == baseline]["TOTAL_MS"].median()
                mine = cell[cell["ROUTINE"] == routine]["TOTAL_MS"].median()
                if pd.isna(base) or pd.isna(mine) or not mine:
                    continue
                heights.append(base / mine)
                offsets.append(index + (slot - (len(others) - 1) / 2) * bar_width)
            axis.bar(offsets, heights, width=bar_width * 0.88, color=color_for(routine, order),
                     edgecolor='none', linewidth=0, label=routine, zorder=3)
            for x, height in zip(offsets, heights):
                axis.annotate(f"{height:.2f}×", xy=(x, height), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)

        axis.axhline(1.0, color=INK_MUTED, linewidth=0.9, zorder=2)
        axis.set_xticks(range(len(key_counts)))
        axis.set_xticklabels([str(c) for c in key_counts])
        axis.set_xlabel("sort key columns")
        axis.set_title(experiment, color=INK, pad=16)
        pads.append(annotate_keys(axis, panel))
        axis.set_ylim(bottom=0)

    level_titles(axes[0], pads)
    axes[0][0].set_ylabel(f"speedup over {baseline} (×)")
    figure_legend(fig, axes[0][0], y=1.23)
    fig.suptitle(f"Speedup over {baseline} across the key sweep — TPC-DS", y=1.35, fontsize=9, color=INK)
    save(fig, outdir, "fig9_key_speedup", formats)


def fig_key_phases(frame, phases, order, outdir, formats, hatch):
    """Where the growth goes as key columns are added: MATERIALIZE (more key bytes) or SORT (more passes)."""
    experiments = sweep_experiments(frame)
    if not experiments or not phases:
        return
    data = of_benchmark(frame, "tpcds")
    data = data[(data["HARNESS"] == "operator") & (data["KEY_SET"].isin(experiments))]

    bars_per_panel = max(len(set(data["KEY_COLS"])) * max(len(order), 1), 1)
    panel_width = max(3.6, 0.22 * bars_per_panel)
    fig, axes = plt.subplots(1, len(experiments), figsize=(max(panel_width * len(experiments), 5.8), 3.2),
                             squeeze=False)
    phase_ms = [c[:-3] + "_MS" for c in phases]

    pads = []
    for axis, experiment in zip(axes[0], experiments):
        clean_axes(axis)
        panel = data[data["KEY_SET"] == experiment]
        key_counts = sorted(set(panel["KEY_COLS"]))

        positions, labels, group_centres = [], [], []
        position = 0.0
        for count in key_counts:
            start = position
            for routine in order:
                series = panel[(panel["ROUTINE"] == routine) & (panel["KEY_COLS"] == count)]
                if series.empty:
                    continue
                bottom = 0.0
                for index, column in enumerate(phase_ms):
                    value = series[column].median()
                    value = 0.0 if pd.isna(value) else value
                    axis.bar(position, value, bottom=bottom, width=0.78,
                             color=phase_color(phases[index]),
                             edgecolor=SURFACE if hatch else "none", linewidth=0,
                             hatch=PHASE_HATCHES[index % len(PHASE_HATCHES)] if hatch else None,
                             label=phase_label(phases[index]) if not positions else None)
                    bottom += value
                residual = max(series["TOTAL_MS"].median() - bottom, 0.0)
                axis.bar(position, residual, bottom=bottom, width=0.78, color=RESIDUAL_COLOR,
                         edgecolor='none', linewidth=0,
                         label="Other (operator overhead)" if not positions else None)
                positions.append(position)
                labels.append(routine)
                position += 1
            group_centres.append((start + position - 1) / 2)
            position += 0.7

        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=6.5)
        for centre, count in zip(group_centres, key_counts):
            axis.annotate(f"{count} key col" + ("" if count == 1 else "s"), xy=(centre, -0.44),
                          xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7,
                          color=INK_MUTED, annotation_clip=False)
        axis.set_title(experiment, color=INK, pad=16)
        pads.append(annotate_keys(axis, panel))
        axis.set_ylim(bottom=0)

    level_titles(axes[0], pads)
    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.26)
    fig.suptitle("Where the time goes as key columns are added — TPC-DS", y=1.37, fontsize=9, color=INK)
    save(fig, outdir, "fig8_key_phases", formats)


def fig_experiment_matrix(frame, outdir, formats):
    """The slide-ready run matrix: which TPC-DS experiment sorts which columns, of which type, at which scale.

    Every cell except the payload NAME is read from the measured rows, so this table cannot claim a configuration
    that was never run -- which is exactly the failure mode of keeping the matrix in a separate hand-written script.
    """
    data = of_benchmark(frame, "tpcds")
    data = data[data["HARNESS"] == "operator"]
    if data.empty:
        return
    experiments = sorted(set(data["KEY_SET"]))

    rows = []
    for experiment in experiments:
        panel = data[data["KEY_SET"] == experiment]
        pairs = [split_key_column(part)
                 for part in max((v for v in panel["SORT_KEY"].unique() if v), key=len, default="").split(",")
                 if part.strip()]
        names = [name for name, _ in pairs]
        types = sorted({data_type for _, data_type in pairs if data_type})
        counts = sorted(set(panel["KEY_COLS"]))
        scale_factors = sorted(set(panel["SCALE"]))
        row_counts = sorted(set(pd.to_numeric(panel.get("ROW_COUNT", pd.Series(dtype=float)), errors="coerce")
                                .dropna()))
        payload_width = int(panel["PAYLOAD_COLS"].max()) if "PAYLOAD_COLS" in panel else 1
        rows.append({
            "name": experiment,
            "table": str(panel["TABLE"].iloc[0]) if "TABLE" in panel else "",
            "keys": names,
            "type": ", ".join(types) if types else "",
            "count": f"{counts[0]} to {counts[-1]}" if len(counts) > 1 else str(counts[0]),
            "payload": PAYLOAD_COLUMNS.get(experiment, f"{payload_width} col"),
            "scale": ", ".join(f"{s:g}" for s in scale_factors),
            "rows": ", ".join(f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.0f} k" for n in row_counts),
        })

    # The one experiment that sweeps gets the prefix block underneath -- it is what makes "3 key cols" concrete.
    sweep_rows = []
    for experiment in sweep_experiments(frame):
        panel = data[data["KEY_SET"] == experiment]
        labelled = key_columns_of(panel, with_types=True)
        sweep_rows = [(count, labelled[:count]) for count in sorted(set(panel["KEY_COLS"]))]
        sweep_name = experiment
        break

    chips = [color_for(r, []) for r in ("baseline", "reduckdb-V1.4", "duckdb-1")]
    columns = [("Experiment", 0.035), ("Table", 0.180), ("Key columns, in order", 0.283), ("Key type", 0.500),
               ("Key cols", 0.580), ("Payload", 0.655), ("SF", 0.770), ("Rows", 0.855)]
    key_slots = [0.283, 0.428, 0.573, 0.718]
    row_height, top = 0.108, 0.815
    height = 6.2 if sweep_rows else 4.2

    fig = plt.figure(figsize=(12.0, height))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0.03, 1.0)
    axis.axis("off")

    axis.text(0.035, 0.965, "TPC-DS multiple key column experiments", fontsize=16, color=INK, va="top")
    axis.text(0.035, 0.912, "Key count swept, payload fixed at one column. Every cell below is read from the "
                            "measured runs.", fontsize=10.5, color=INK_SECONDARY, va="top")

    for label, x in columns:
        axis.text(x, top + 0.022, label.upper(), fontsize=8.5, color=INK_MUTED, va="bottom", fontweight="bold")
    axis.plot([0.035, 0.965], [top, top], color=INK_MUTED, linewidth=1.0, solid_capstyle="butt")

    for index, row in enumerate(rows):
        y = top - (index + 1) * row_height + row_height / 2
        if index % 2 == 1:
            axis.add_patch(Rectangle((0.035, y - row_height / 2 + 0.005), 0.93, row_height - 0.01,
                                     facecolor="#f3f2ef", edgecolor="none", zorder=0))
        axis.add_patch(Rectangle((0.038, y - 0.015), 0.009, 0.030, facecolor=chips[index % len(chips)],
                                 edgecolor="none", zorder=2))
        axis.text(0.056, y, row["name"], fontsize=10.5, color=INK, va="center", fontweight="bold")
        axis.text(0.180, y, row["table"], fontsize=8.5, color=INK_SECONDARY, va="center", family="monospace")
        wrapped = ",\n".join(", ".join(row["keys"][start:start + 2]) for start in range(0, len(row["keys"]), 2))
        axis.text(0.283, y, wrapped, fontsize=8.5, color=INK_SECONDARY, va="center", family="monospace")
        axis.text(0.500, y, row["type"], fontsize=9, color=INK, va="center", family="monospace")
        axis.text(0.580, y, row["count"], fontsize=10, color=INK, va="center")
        axis.text(0.655, y, row["payload"], fontsize=8.5, color=INK_SECONDARY, va="center", family="monospace")
        axis.text(0.770, y, row["scale"], fontsize=9.5, color=INK, va="center")
        axis.text(0.855, y, row["rows"], fontsize=9.5, color=INK, va="center")

    if sweep_rows:
        sweep_top = top - len(rows) * row_height - 0.075
        axis.plot([0.035, 0.965], [sweep_top + 0.052, sweep_top + 0.052], color=GRID, linewidth=0.9)
        axis.text(0.035, sweep_top + 0.030, f"HOW THE {sweep_name} KEY GROWS", fontsize=8.5, color=INK_MUTED,
                  va="bottom", fontweight="bold")
        for index, (count, names) in enumerate(sweep_rows):
            y = sweep_top - index * 0.072
            axis.text(0.035, y, f"{count} key column" + ("" if count == 1 else "s"), fontsize=9.5,
                      color=INK if index == len(sweep_rows) - 1 else INK_SECONDARY, va="center")
            for position, name in enumerate(names[:len(key_slots)]):
                is_new = position == count - 1
                axis.text(key_slots[position], y, name, fontsize=8, va="center", family="monospace",
                          color=PHASE_COLORS["WRITE_OUT_US"] if is_new else INK_SECONDARY,
                          fontweight="bold" if is_new else "normal")
        axis.text(0.035, sweep_top - len(sweep_rows) * 0.072 - 0.010,
                  "Each step adds the next column, shown in red. All of them are low cardinality, so ties are "
                  "everywhere and every added column does real\ntie breaking work: a unique leading column would "
                  "leave the later ones unreachable and the sweep flat.",
                  fontsize=8.5, color=INK_MUTED, va="top", linespacing=1.6)

    save(fig, outdir, "fig0_experiment_matrix", formats)


def fig_tpcds_experiments(frame, order, outdir, formats):
    """Every TPC-DS experiment at its full key width -- including the ones with no sweep (ICDE Figure 12)."""
    data = of_benchmark(frame, "tpcds")
    data = data[data["HARNESS"] == "operator"]
    if data.empty:
        return
    widest = data.groupby("KEY_SET", observed=True)["KEY_COLS"].transform("max")
    data = data[data["KEY_COLS"] == widest]

    experiments = sorted(set(data["KEY_SET"]))
    if not experiments:
        return

    fig, axes = plt.subplots(1, len(experiments), figsize=(max(2.9 * len(experiments), 5.4), 2.6),
                             squeeze=False)
    bar_width = 0.8 / max(len(order), 1)

    pads = []
    for axis, experiment in zip(axes[0], experiments):
        clean_axes(axis)
        panel = data[data["KEY_SET"] == experiment]
        scale_factors = sorted(set(panel["SCALE"]))
        for slot, routine in enumerate(order):
            heights, offsets = [], []
            for index, scale in enumerate(scale_factors):
                value = panel[(panel["ROUTINE"] == routine) & (panel["SCALE"] == scale)]["TOTAL_MS"].median()
                if pd.isna(value):
                    continue
                heights.append(value)
                offsets.append(index + (slot - (len(order) - 1) / 2) * bar_width)
            axis.bar(offsets, heights, width=bar_width * 0.88, color=color_for(routine, order),
                     edgecolor='none', linewidth=0, label=routine, zorder=3)
        axis.set_xticks(range(len(scale_factors)))
        axis.set_xticklabels([f"SF {s:g}" for s in scale_factors])
        keys = int(panel["KEY_COLS"].max())
        axis.set_title(f"{experiment} · {keys} key col" + ("" if keys == 1 else "s"), color=INK, pad=16)
        pads.append(annotate_keys(axis, panel, numbered=False))
        axis.set_ylim(bottom=0)

    level_titles(axes[0], pads)
    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.23)
    fig.suptitle("TPC-DS experiments at full key width", y=1.35, fontsize=9, color=INK)
    save(fig, outdir, "fig6_tpcds_experiments", formats)


# =====================================================================================================================
def write_tables(table: pd.DataFrame, outdir: Path) -> None:
    csv_path = outdir / "summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    columns = [c for c in ["ROUTINE", "BENCHMARK", "HARNESS", "KEY_SET", "KEY_COLS", "SCALE", "PAYLOAD_COLS", "n",
                           "median_ms", "min_ms", "max_ms", "cv_pct", "ns_per_tuple", "speedup_vs_base"]
               if c in table.columns]
    md_path = outdir / "summary.md"
    md_path.write_text(table[columns].to_markdown(index=False))
    print(f"  wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, nargs="+", default=[Path("results")],
                        help="result CSVs, or a directory holding them (default: results/). The TPC-H and TPC-DS "
                             "files are separate -- pass both, or just the directory.")
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--baseline", default="baseline", help="routine used as the speedup reference")
    parser.add_argument("--routine-order", default=None,
                        help="comma-separated routine order for legends (colors are fixed per routine regardless)")
    parser.add_argument("--formats", default="pdf,png", help="comma-separated output formats")
    parser.add_argument("--font", choices=["sans", "serif"], default="sans",
                        help="serif matches a Times-set paper body")
    parser.add_argument("--hatch", action="store_true",
                        help="add hatching to stacked phases so the figure survives greyscale printing")
    parser.add_argument("--keys", default=None,
                        help="comma-separated KEY_SET filter (default: all present)")
    parser.add_argument("--exclude", default="subsort",
                        help="comma-separated routines to leave out (default: subsort). Pass --exclude '' for none.")
    parser.add_argument("--cv-threshold", type=float, default=5.0, help="warn above this coefficient of variation")
    args = parser.parse_args()

    routine_order = args.routine_order.split(",") if args.routine_order else None
    sources = resolve_sources(args.csv)
    print("reading")
    frame, phases = load(sources, routine_order)

    excluded = [r.strip() for r in args.exclude.split(",") if r.strip()]
    if excluded:
        present = [r for r in excluded if r in set(frame["ROUTINE"].astype(str))]
        dropped = int(frame["ROUTINE"].astype(str).isin(present).sum()) if present else 0
        if present:
            frame = frame[~frame["ROUTINE"].astype(str).isin(present)]
            # Re-derive the categorical order so the dropped routine leaves no empty legend slot.
            found = sorted(set(frame["ROUTINE"].astype(str)))
            order_now = [r for r in ROUTINE_COLORS if r in found] + [r for r in found if r not in ROUTINE_COLORS]
            frame["ROUTINE"] = pd.Categorical(frame["ROUTINE"].astype(str), categories=order_now, ordered=True)
            print(f"  excluding {present}: {dropped} row(s) dropped")
        missing = [r for r in excluded if r not in present]
        if missing:
            print(f"  (nothing to exclude for {missing} -- not in the data)")
    if frame.empty:
        sys.exit("every row was excluded")

    key_order = args.keys.split(",") if args.keys else None
    if key_order:
        frame = frame[frame["KEY_SET"].isin(key_order)]
        if frame.empty:
            sys.exit(f"no rows with KEY_SET in {key_order}")

    order = list(frame["ROUTINE"].cat.categories)

    print(f"\nloaded {len(frame)} rows from {len(sources)} file(s)")
    print(f"  routines : {', '.join(order)}")
    print(f"  phases   : {', '.join(phase_label(p) for p in phases) or '(none)'}")
    benchmarks = sorted(set(frame["BENCHMARK"]))
    print(f"  benchmark: {', '.join(benchmarks)}")
    print(f"  key sets : {', '.join(sorted(set(frame['KEY_SET'])))}")
    if "tpch" in benchmarks:
        print(f"  tpch     : payload sweep on '{sweep_key_set(frame)}'")
    if "tpcds" in benchmarks:
        swept = sweep_experiments(frame)
        print(f"  tpcds    : key sweep on {', '.join(swept) if swept else '(none -- no experiment varies KEY_COLS)'}")
    print(f"  harnesses: {', '.join(sorted(set(frame['HARNESS'])))}\n")

    check_data(frame, phases, args.cv_threshold)

    if args.baseline not in order:
        # Branch tags are longer than the short name ("baseline-master"), so resolve an unambiguous prefix rather
        # than silently emitting empty speedup columns.
        candidates = [r for r in order if r.startswith(args.baseline)]
        if len(candidates) == 1:
            print(f"  baseline '{args.baseline}' resolved to '{candidates[0]}'\n")
            args.baseline = candidates[0]
        else:
            print(f"  ! baseline routine '{args.baseline}' not in the data; speedups and fig3/fig9 will be empty.\n"
                  f"    Available: {', '.join(order)}\n"
                  f"    Pass --baseline <one of those>.\n", file=sys.stderr)

    args.outdir.mkdir(parents=True, exist_ok=True)
    apply_style(args.font)
    formats = args.formats.split(",")

    print("figures")
    fig_experiment_matrix(frame, args.outdir, formats)
    fig_payload_sweep(frame, order, args.outdir, formats)
    fig_phase_breakdown(frame, phases, order, args.outdir, formats, args.hatch)
    fig_speedup(frame, order, args.outdir, formats, args.baseline)
    fig_operator_vs_sql(frame, order, args.outdir, formats, args.baseline)

    fig_tpcds_experiments(frame, order, args.outdir, formats)
    fig_key_sweep(frame, order, args.outdir, formats)
    fig_key_phases(frame, phases, order, args.outdir, formats, args.hatch)
    fig_key_speedup(frame, order, args.outdir, formats, args.baseline)

    print("\ntables")
    write_tables(summarize(frame, phases, args.baseline), args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
