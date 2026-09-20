#!/usr/bin/env python3
"""Figures and summary tables for sort_evaluation results.

    python3 plot_results.py --csv results/all_results.csv --outdir figures/

Produces, for every scale factor present in the data:

    fig1_payload_sweep      runtime vs payload width, one line per routine    [the headline figure]
    fig2_phase_breakdown    stacked phase composition per routine
    fig3_speedup            speedup over the baseline routine
    fig5_operator_vs_sql    does the operator-level win survive a real plan  [if sql rows present]
    summary.csv / summary.md   median, spread and speedup per configuration  [the table view]

Phases are self-adapting. Columns are discovered from the CSV header (every `*_US` except TOTAL_US),
so adding MERGE_PATH_US on the C++ side needs no change here; a phase that no run in the data
reports is dropped rather than drawn as an empty legend entry, and a missing value for one run is
treated as absent rather than breaking the stack.

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

# Color follows the ENTITY, not its rank: a routine keeps its hue whether or not the others are in the plot.
ROUTINE_COLORS = {
    "baseline": "#2a78d6",  # blue
    "pmerge": "#eb6834",    # orange
    "kway": "#1baf7a",      # aqua
    "ips4o": "#eda100",     # yellow
    "subsort": "#e87ba4",   # magenta
}
FALLBACK_COLORS = ["#008300", "#4a3aa7", "#e34948"]  # green, violet, red

PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RESIDUAL_COLOR = "#b9b8b4"

# Print relief: papers get photocopied. Hatches give the stacked segments a second, non-color channel.
PHASE_HATCHES = ["", "///", "...", "\\\\\\", "xxx", "---"]

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=2, width=0.8)


# =====================================================================================================================
# Loading
# =====================================================================================================================
def phase_columns(frame: pd.DataFrame) -> list[str]:
    """Every *_US column except TOTAL_US, in CSV order -- so a new phase needs no change here."""
    return [c for c in frame.columns if c.endswith("_US") and c != "TOTAL_US"]


def phase_label(column: str) -> str:
    return column[:-3].replace("_", " ").title()


def load(csv_path: Path, routine_order: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(csv_path)

    required = {"ROUTINE", "HARNESS", "SCALE", "PAYLOAD_COLS", "RUN_ID", "TOTAL_US"}
    missing = required - set(frame.columns)
    if missing:
        sys.exit(f"{csv_path}: missing columns {sorted(missing)}")

    frame["TOTAL_MS"] = frame["TOTAL_US"] / 1000.0

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

    frame["ROUTINE"] = pd.Categorical(frame["ROUTINE"], categories=order, ordered=True)
    return frame, phases


def color_for(routine: str, order: list[str]) -> str:
    if routine in ROUTINE_COLORS:
        return ROUTINE_COLORS[routine]
    index = [r for r in order if r not in ROUTINE_COLORS].index(routine)
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


# =====================================================================================================================
# Data quality -- surfaced before any plotting, because a pretty plot of noisy data is worse than no plot
# =====================================================================================================================
def check_data(frame: pd.DataFrame, phases: list[str], cv_threshold: float) -> None:
    print("data checks")

    runs = frame.groupby(["ROUTINE", "HARNESS", "SCALE", "PAYLOAD_COLS"], observed=True)["TOTAL_MS"]
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
            print(f"      {row['ROUTINE']:<10} {row['HARNESS']:<9} sf={row['SCALE']:<5}"
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
GROUP = ["ROUTINE", "HARNESS", "SCALE", "PAYLOAD_COLS"]


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

    key = ["HARNESS", "SCALE", "PAYLOAD_COLS"]
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


def scales(frame: pd.DataFrame) -> list[float]:
    """Scale factors present in the operator data, ascending."""
    return sorted(set(frame[frame["HARNESS"] == "operator"]["SCALE"]))


# =====================================================================================================================
# Figure 1 -- runtime vs payload width. The headline: the blog's single wide-table point, as a curve.
# =====================================================================================================================
def fig_payload_sweep(frame, order, outdir, formats):
    data = frame[frame["HARNESS"] == "operator"]
    if data.empty:
        return
    panels = scales(frame)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.7), squeeze=False)

    for axis, scale in zip(axes[0], panels):
        clean_axes(axis)
        panel = data[data["SCALE"] == scale]
        for routine in order:
            series = panel[panel["ROUTINE"] == routine]
            if series.empty:
                continue
            grouped = series.groupby("PAYLOAD_COLS", observed=True)["TOTAL_MS"]
            widths = sorted(grouped.groups)
            medians = [grouped.get_group(w).median() for w in widths]
            lows = [grouped.get_group(w).min() for w in widths]
            highs = [grouped.get_group(w).max() for w in widths]
            colour = color_for(routine, order)
            axis.fill_between(widths, lows, highs, color=colour, alpha=0.13, linewidth=0)
            axis.plot(widths, medians, color=colour, linewidth=2, marker="o", markersize=4.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label=routine, zorder=3)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_xlabel("payload columns")
        axis.set_xticks(sorted(set(data["PAYLOAD_COLS"])))
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("sort time (ms, median)")
    figure_legend(fig, axes[0][0], y=1.06)
    fig.suptitle("Sort runtime vs. payload width — lineitem ORDER BY l_shipdate",
                 y=1.17, fontsize=9, color=INK, ha="center")
    save(fig, outdir, "fig1_payload_sweep", formats)


# =====================================================================================================================
# Figure 2 -- where the time goes. This is what the blog does not have.
# =====================================================================================================================
def fig_phase_breakdown(frame, phases, order, outdir, formats, hatch):
    if not phases:
        return
    data = frame[frame["HARNESS"] == "operator"]
    if data.empty:
        return

    widths = sorted(set(data["PAYLOAD_COLS"]))
    panels = scales(frame)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.9), squeeze=False)
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
                             color=PHASE_COLORS[index % len(PHASE_COLORS)],
                             edgecolor=SURFACE, linewidth=1.2,  # 2px-equivalent surface gap, not a border
                             hatch=PHASE_HATCHES[index % len(PHASE_HATCHES)] if hatch else None,
                             label=phase_label(phases[index]) if not positions and index < len(phases) else None)
                    bottom += value
                residual = max(series["TOTAL_MS"].median() - bottom, 0.0)
                axis.bar(position, residual, bottom=bottom, width=0.78, color=RESIDUAL_COLOR,
                         edgecolor=SURFACE, linewidth=1.2,
                         label="Other (operator overhead)" if not positions else None)
                positions.append(position)
                labels.append(routine)
                position += 1
            group_centres.append((start + position - 1) / 2)
            position += 0.7

        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        for centre, width in zip(group_centres, widths):
            axis.annotate(plural_cols(width), xy=(centre, -0.28), xycoords=("data", "axes fraction"),
                          ha="center", va="top", fontsize=7, color=INK_MUTED, annotation_clip=False)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.14)
    fig.suptitle("Where the time goes", y=1.24, fontsize=9, color=INK)
    save(fig, outdir, "fig2_phase_breakdown", formats)


# =====================================================================================================================
# Figure 3 -- speedup over baseline. Mirrors the blog's speedup tables.
# =====================================================================================================================
def fig_speedup(frame, order, outdir, formats, baseline):
    data = frame[frame["HARNESS"] == "operator"]
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
                     edgecolor=SURFACE, linewidth=1.2, label=routine, zorder=3)
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
    if "sql" not in set(frame["HARNESS"]):
        return
    sql = frame[frame["HARNESS"] == "sql"]
    widest = frame[frame["HARNESS"] == "operator"]["PAYLOAD_COLS"].max()
    operator = frame[(frame["HARNESS"] == "operator") & (frame["PAYLOAD_COLS"] == widest)]

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
                     edgecolor=SURFACE, linewidth=1.2, label=routine, zorder=3)
        axis.set_xticks(range(len(harnesses)))
        axis.set_xticklabels([name for name, _ in harnesses], fontsize=7)
        axis.set_title(f"SF {scale:g}", color=INK, pad=6)
        axis.set_ylim(bottom=0)

    axes[0][0].set_ylabel("median time (ms)")
    figure_legend(fig, axes[0][0], y=1.07)
    fig.suptitle("Isolated operator vs. end-to-end query", y=1.18, fontsize=9, color=INK)
    save(fig, outdir, "fig5_operator_vs_sql", formats)


# =====================================================================================================================
def write_tables(table: pd.DataFrame, outdir: Path) -> None:
    csv_path = outdir / "summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    columns = [c for c in ["ROUTINE", "HARNESS", "SCALE", "PAYLOAD_COLS", "n",
                           "median_ms", "min_ms", "max_ms", "cv_pct", "ns_per_tuple", "speedup_vs_base"]
               if c in table.columns]
    md_path = outdir / "summary.md"
    md_path.write_text(table[columns].to_markdown(index=False))
    print(f"  wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("results/all_results.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--baseline", default="baseline", help="routine used as the speedup reference")
    parser.add_argument("--routine-order", default=None,
                        help="comma-separated routine order for legends (colors are fixed per routine regardless)")
    parser.add_argument("--formats", default="pdf,png", help="comma-separated output formats")
    parser.add_argument("--font", choices=["sans", "serif"], default="sans",
                        help="serif matches a Times-set paper body")
    parser.add_argument("--hatch", action="store_true",
                        help="add hatching to stacked phases so the figure survives greyscale printing")
    parser.add_argument("--cv-threshold", type=float, default=5.0, help="warn above this coefficient of variation")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"no such file: {args.csv}")

    routine_order = args.routine_order.split(",") if args.routine_order else None
    frame, phases = load(args.csv, routine_order)
    order = list(frame["ROUTINE"].cat.categories)

    print(f"loaded {len(frame)} rows from {args.csv}")
    print(f"  routines : {', '.join(order)}")
    print(f"  phases   : {', '.join(phase_label(p) for p in phases) or '(none)'}")
    print(f"  harnesses: {', '.join(sorted(set(frame['HARNESS'])))}\n")

    check_data(frame, phases, args.cv_threshold)

    if args.baseline not in order:
        print(f"  ! baseline routine '{args.baseline}' not in the data; speedups will be empty\n", file=sys.stderr)

    args.outdir.mkdir(parents=True, exist_ok=True)
    apply_style(args.font)
    formats = args.formats.split(",")

    print("figures")
    fig_payload_sweep(frame, order, args.outdir, formats)
    fig_phase_breakdown(frame, phases, order, args.outdir, formats, args.hatch)
    fig_speedup(frame, order, args.outdir, formats, args.baseline)
    fig_operator_vs_sql(frame, order, args.outdir, formats, args.baseline)

    print("\ntables")
    write_tables(summarize(frame, phases, args.baseline), args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())