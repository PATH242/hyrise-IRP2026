#!/usr/bin/env python3
"""Read Google Benchmark JSON files (one per build configuration) and plot the sort results.

Each results/<label>.json is one build configuration -- one sorting routine, since the routine is fixed at compile
time (label = the tag passed to the run script, e.g. base / binary_radix / kway_radix).

The benchmark matrix is one-factor-at-a-time (OFAT) around a fixed baseline, so the per-metric figures are organised
by *axis*: for each axis, the points that differ from the baseline in that field alone are plotted together.

Metrics reported (write_out_s is deliberately not among them -- it is a fraction of a percent of the operator and
carries only noise):
    total time        real_time_s   whole operator, end to end
    in-thread sort    run_gen_s     Sort::OperatorSteps::Sort
    merge             merge_s       Sort::OperatorSteps::TemporaryResultWriting

Figures written to <results_dir>/plots/:
    <metric>__axes.png        per-axis panels, one row per threading mode
    step_share__<mt>.png      step composition against row count, one panel per implementation
    threading__total_time.png single-threaded beside multithreaded, per row count

Usage:
    python3 plot_results.py [results_dir]         # default: results/
    python3 plot_results.py a.json b.json ...     # explicit files
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import textwrap
from collections import defaultdict, namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------
# Appearance
# ------------------------------------------------------------------------------------------------------------------

# Fixed categorical order, assigned by sorted config label and never cycled: a config keeps its colour when another
# config is added or removed, so figures across drafts stay comparable. The first six are the validated set; the last
# two are Okabe-Ito additions for runs with more than six build variants.
CONFIG_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#7B5AA6", "#8A5A00", "#B03A6E", "#56B4E9", "#E69F00"]

# The three recorded steps are ordered stages of one pipeline, so a single-hue light-to-dark ramp is the right
# encoding for the stacked breakdown -- it also survives greyscale printing.
STEP_RAMP = ["#cde3f0", "#6badd6", "#14567f"]

# Threading is the comparison in the ST-vs-MT figure, so colour encodes mode there, not implementation.
# A light/dark pair of one hue, deliberately outside CONFIG_PALETTE so it cannot be misread as a config.
THREADING_COLORS = {0: "#A9C7DC", 1: "#14567f"}

INK = "#222222"
INK_MUTED = "#666666"

# write_out_s is excluded throughout: it is consistently under one percent of the operator and does not vary between
# implementations, so it adds a sliver of noise to every stack and a figure nobody reads.
STEPS = ["materialize_s", "run_gen_s", "merge_s"]
STEP_LABELS = {
    "materialize_s": "materialize sort columns",
    "run_gen_s": "in-thread sort",
    "merge_s": "merge",
}

METRICS = {
    "real_time_s": "total time [s]",
    "run_gen_s": "in-thread sort time [s]",
    "merge_s": "merge time [s]",
}

TIME_UNIT_TO_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
MT_NAME = {0: "single-threaded", 1: "multithreaded"}
MT_SHORT = {0: "st", 1: "mt"}

# ------------------------------------------------------------------------------------------------------------------
# The experiment matrix -- must match build_matrix() in the benchmark
# ------------------------------------------------------------------------------------------------------------------

Point = namedtuple("Point", ["key_type", "distribution", "rows", "cols", "mt"])

BASELINE = {"key_type": "int32", "distribution": "random", "rows": 10_000_000, "cols": 1}
AXES = ["distribution", "key_type", "rows", "cols"]

# Display order along each categorical axis. Anything unrecognised is appended alphabetically, so adding a
# distribution to the benchmark does not require editing this file first.
AXIS_ORDER = {
    "distribution": ["random", "ascending", "dup128", "uniform", "zipf090", "zipf099"],
    "key_type": ["int32", "str_sso", "str_var"],
}


def human(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}K"
    return str(n)


def axis_tick_label(axis, value):
    return human(value) if axis == "rows" else str(value)


def ordered_axis_values(axis, values):
    preferred = AXIS_ORDER.get(axis)
    if preferred is None:
        return sorted(values)
    known = [v for v in preferred if v in values]
    return known + sorted(set(values) - set(known))


# ------------------------------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------------------------------

# Deliberately not anchored at the end: Google Benchmark appends _mean / _median / _stddev / _cv to aggregate rows.
NAME_RE = re.compile(
    r"^Sort/(?P<key_type>[^/]+)/(?P<distribution>[^/]+)/rows(?P<rows>\d+)/cols(?P<cols>\d+)/mt(?P<mt>[01])")


def parse_point(name):
    match = NAME_RE.match(name)
    if not match:
        return None
    return Point(match["key_type"], match["distribution"], int(match["rows"]), int(match["cols"]), int(match["mt"]))


def load_file(path):
    """Return ({Point: {metric: value}}, context_dict) for one config file."""
    with open(path) as handle:
        data = json.load(handle)

    median_rows, iteration_rows = {}, defaultdict(list)
    for entry in data.get("benchmarks", []):
        point = parse_point(entry.get("name", ""))
        if point is None:
            continue

        unit = TIME_UNIT_TO_S.get(entry.get("time_unit", "ns"), 1e-9)
        record = {}
        for metric in set(METRICS) | set(STEPS):
            if metric == "real_time_s":
                record[metric] = float(entry.get("real_time", float("nan"))) * unit
            else:
                record[metric] = float(entry.get(metric, float("nan")))

        run_type = entry.get("run_type")
        if run_type == "aggregate" and entry.get("aggregate_name") == "median":
            median_rows[point] = record
        elif run_type in ("iteration", None):
            iteration_rows[point].append(record)

    resolved = {}
    for point in set(median_rows) | set(iteration_rows):
        if point in median_rows:
            resolved[point] = median_rows[point]
        else:
            runs = iteration_rows[point]
            resolved[point] = {m: statistics.median(r[m] for r in runs) for m in runs[0]}
    return resolved, data.get("context", {})


def collect(paths):
    """Load every file, merging those that share a label.

    Later files win per point, so listing a patch directory after the main results backfills missing points and
    replaces re-measured ones without editing the original JSONs. Provenance stays honest: nothing is overwritten on
    disk, and the merge is visible in the load summary."""
    configs, contexts, sources = defaultdict(dict), {}, defaultdict(list)
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        points, context = load_file(path)
        if not points:
            continue
        overlap = len(set(points) & set(configs[label]))
        configs[label].update(points)
        if context:
            contexts[label] = context
        sources[label].append((path, len(points), overlap))

    for label, entries in sources.items():
        if len(entries) > 1:
            print(f"merged {len(entries)} files into '{label}':")
            for path, count, overlap in entries:
                replaced = f", {overlap} replacing earlier point(s)" if overlap else ""
                print(f"    {path}: {count} points{replaced}")
    return dict(configs), contexts


def binding_note(contexts):
    nodes = {ctx.get("numa_node") for ctx in contexts.values() if ctx.get("numa_node")}
    if not nodes:
        return "NUMA binding not recorded in the result files"
    if len(nodes) == 1:
        return f"bound to NUMA node {nodes.pop()}"
    return f"WARNING: result files mix NUMA nodes ({', '.join(sorted(nodes))}) -- not comparable"


# ------------------------------------------------------------------------------------------------------------------
# Axis decomposition
# ------------------------------------------------------------------------------------------------------------------

def points_on_axis(configs, axis):
    """Points that match the baseline in every field except `axis` (and mt, which is its own dimension)."""
    fixed = [f for f in BASELINE if f != axis]
    found = set()
    for cfg in configs.values():
        for point in cfg:
            if all(getattr(point, field) == BASELINE[field] for field in fixed):
                found.add(getattr(point, axis))
    return ordered_axis_values(axis, found)


def point_on_axis(axis, value, mt):
    fields = dict(BASELINE)
    fields[axis] = value
    return Point(fields["key_type"], fields["distribution"], fields["rows"], fields["cols"], mt)


def baseline_row_counts(configs, mt):
    """Row counts measured at the baseline key type / distribution / column count."""
    found = set()
    for cfg in configs.values():
        for point in cfg:
            if (point.key_type == BASELINE["key_type"] and point.distribution == BASELINE["distribution"]
                    and point.cols == BASELINE["cols"] and point.mt == mt):
                found.add(point.rows)
    return sorted(found)


def config_color(label, all_labels):
    index = sorted(all_labels).index(label)
    if index >= len(CONFIG_PALETTE):
        raise SystemExit(
            f"{len(all_labels)} configs but only {len(CONFIG_PALETTE)} validated colours. Extend CONFIG_PALETTE and "
            f"re-run the palette validator rather than letting matplotlib cycle hues.")
    return CONFIG_PALETTE[index]


# ------------------------------------------------------------------------------------------------------------------
# Drawing primitives
# ------------------------------------------------------------------------------------------------------------------

def _style_axis(ax):
    ax.grid(True, which="both", linestyle=":", alpha=0.4, color=INK_MUTED)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _label_line_ends(ax, endpoints, min_gap_pts=8.5):
    """Direct labels at the end of each line, anchored at the line's own y and displaced only where labels would
    collide. A fixed evenly-spaced stack is as tall as the number of configs regardless of how far apart the lines
    run, which overflows the panel once there are more than a few."""
    figure = ax.figure
    figure.canvas.draw()
    dpi = figure.dpi
    min_gap_px = min_gap_pts * dpi / 72.0

    placed = sorted(([ax.transData.transform(xy)[1], label, xy] for xy, label in endpoints), key=lambda i: i[0])
    for index in range(1, len(placed)):
        if placed[index][0] - placed[index - 1][0] < min_gap_px:
            placed[index][0] = placed[index - 1][0] + min_gap_px

    for target_y, label, xy in placed:
        offset_pts = (target_y - ax.transData.transform(xy)[1]) * 72.0 / dpi
        ax.annotate(label, xy=xy, xytext=(5, offset_pts), textcoords="offset points", fontsize=6.5,
                    color=INK_MUTED, va="center", annotation_clip=False)


def _fit_exponent(points):
    """Least-squares slope of log(time) against log(rows). 1.0 is linear scaling; above that the routine is degrading
    faster than the input grows, which is the single most useful number on the rows axis."""
    usable = [(x, y) for x, y in points if x > 0 and y > 0 and y == y]
    if len(usable) < 2:
        return None
    xs = [math.log10(x) for x, _ in usable]
    ys = [math.log10(y) for _, y in usable]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _bar_panel(ax, configs, axis, values, mt, metric):
    labels = sorted(configs)
    width = 0.8 / max(len(labels), 1)
    for label_index, label in enumerate(labels):
        heights, positions = [], []
        for value_index, value in enumerate(values):
            record = configs[label].get(point_on_axis(axis, value, mt))
            if record is None:
                continue
            positions.append(value_index - 0.4 + width * (label_index + 0.5))
            heights.append(record.get(metric, float("nan")))
        if positions:
            ax.bar(positions, heights, width=width * 0.88, color=config_color(label, labels), label=label,
                   linewidth=0)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([axis_tick_label(axis, v) for v in values], rotation=30, ha="right")


def _line_panel(ax, configs, axis, values, mt, metric):
    labels = sorted(configs)
    endpoints, exponents = [], []
    for label in labels:
        series = []
        for value in values:
            record = configs[label].get(point_on_axis(axis, value, mt))
            if record is not None:
                measurement = record.get(metric, float("nan"))
                if measurement == measurement:
                    series.append((value, measurement))
        if not series:
            continue
        ax.plot([s[0] for s in series], [s[1] for s in series], marker="o", markersize=5, linewidth=2,
                color=config_color(label, labels), label=label)
        endpoints.append((series[-1], label))
        exponent = _fit_exponent(series)
        if exponent is not None:
            exponents.append(f"{label} n^{exponent:.2f}")

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    if endpoints:
        ax.set_xlim(min(values) / 1.25, max(values) * 3.4)
        _label_line_ends(ax, endpoints)
    ax.set_xticks(values)
    ax.set_xticklabels([human(v) for v in values])
    ax.minorticks_off()

    # Fitted exponents in the corner: the crossover between two routines is invisible in absolute seconds on a log
    # axis, but the slopes state it outright.
    if exponents:
        ax.text(0.02, 0.98, "\n".join(exponents), transform=ax.transAxes, fontsize=6, color=INK_MUTED,
                va="top", ha="left")


def finish(fig, ax_for_legend, title, note, out_path, legend_anchor=0.955, rect_top=0.90, footnote=None):
    handles, labels = ax_for_legend.get_legend_handles_labels()
    fig.suptitle(title, color=INK, fontsize=11)
    # Wrapped, and the provenance note sits a line above it: at 7pt a long single-line footnote runs straight into
    # the right-hand note and the two overprint.
    if footnote:
        wrapped = textwrap.fill(footnote, width=int(fig.get_figwidth() * 15))
        fig.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=7, color=INK_MUTED)
        fig.text(0.99, 0.005 + 0.022 * wrapped.count("\n"), note, ha="right", va="bottom", fontsize=7,
                 color=INK_MUTED)
    else:
        fig.text(0.99, 0.005, note, ha="right", fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.055 if footnote else 0.03, 1, rect_top))
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), frameon=False,
                   bbox_to_anchor=(0.5, legend_anchor), fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------------------------

def plot_metric_axes(configs, metric, mt_values, axis_values, note, out_path):
    fig, axes = plt.subplots(len(mt_values), len(AXES),
                             figsize=(3.6 * len(AXES), 3.2 * len(mt_values)), squeeze=False)
    for row_index, mt in enumerate(mt_values):
        for col_index, axis in enumerate(AXES):
            ax = axes[row_index][col_index]
            values = axis_values[axis]
            if not values:
                ax.set_visible(False)
                continue
            if axis == "rows":
                _line_panel(ax, configs, axis, values, mt, metric)
            else:
                _bar_panel(ax, configs, axis, values, mt, metric)
            _style_axis(ax)
            if row_index == 0:
                ax.set_title(f"{axis} axis", color=INK, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{MT_NAME.get(mt, mt)}\n{METRICS[metric]}", fontsize=9)
    finish(fig, axes[0][0],
           f"{METRICS[metric]} -- OFAT around baseline ({BASELINE['key_type']}, {BASELINE['distribution']}, "
           f"{human(BASELINE['rows'])} rows, {BASELINE['cols']} key col)",
           note, out_path, legend_anchor=0.965, rect_top=0.93,
           footnote="Every axis passes through the baseline, so the baseline bar is the same measurement in all four "
                    "panels.")


def plot_step_share(configs, mt, note, out_path):
    """Step composition against row count, one panel per implementation, int32 only.

    One panel each rather than grouped bars: with a stacked bar the colours already encode the step, so grouping
    implementations side by side leaves nothing to identify them except position, and a reader has to count bars
    against a footnote. A panel title names the implementation outright.

    The question this answers: as n grows, does the merge take a larger share of the operator? That is the shape of a
    routine whose merge scales worse than linearly, and it is invisible in absolute seconds because everything grows
    with n at once."""
    labels = sorted(configs)
    row_counts = baseline_row_counts(configs, mt)
    if not row_counts or not labels:
        return

    fig, axes = plt.subplots(1, len(labels), figsize=(2.6 * len(labels) + 1.0, 4.0), sharey=True, squeeze=False)
    drew_any = False

    for label_index, label in enumerate(labels):
        ax = axes[0][label_index]
        for row_index, rows in enumerate(row_counts):
            point = Point(BASELINE["key_type"], BASELINE["distribution"], rows, BASELINE["cols"], mt)
            record = configs[label].get(point)
            if record is None:
                continue
            heights = [record.get(step, 0.0) for step in STEPS]
            heights = [0.0 if h != h else h for h in heights]  # NaN -> 0
            total = sum(heights)
            if total <= 0:
                continue

            bottom = 0.0
            for step_index, height in enumerate(h / total for h in heights):
                ax.bar(row_index, height, width=0.62, bottom=bottom, color=STEP_RAMP[step_index], linewidth=0,
                       label=STEP_LABELS[STEPS[step_index]] if not drew_any else None)
                bottom += height
            drew_any = True

        ax.set_title(label, color=INK, fontsize=10)
        ax.set_xticks(range(len(row_counts)))
        ax.set_xticklabels([human(r) for r in row_counts])
        ax.set_xlabel("rows", fontsize=9)
        ax.set_ylim(0, 1)
        if label_index == 0:
            ax.set_ylabel("share of recorded step time", fontsize=9)
        _style_axis(ax)

    finish(fig, axes[0][0],
           f"Step composition against row count -- {MT_NAME[mt]}, "
           f"{BASELINE['key_type']} / {BASELINE['distribution']} / {BASELINE['cols']} key column",
           note, out_path,
           footnote="Shares are of the three recorded steps; output writing is excluded, as is any operator time not "
                    "attributed to a step. An empty panel means that implementation has no data in this mode.")


def plot_threading_comparison(configs, note, out_path, metric="real_time_s"):
    """Single-threaded beside multithreaded, one panel per row count.

    Replaces the earlier speedup-ratio figure. A ratio is a derived number: a bar of height 2.9 tells you nothing
    about whether the sort took 40 milliseconds or 40 seconds, and it silently hides which of the two runs moved. Here
    both measurements are drawn at their real magnitude, colour means threading mode and nothing else, and the ratio
    is printed above each pair -- so the same question is answered without giving up the underlying numbers.

    One panel per row count rather than one per implementation: within a panel every bar is on the same scale, and
    total time spans roughly fifty-fold from 1M to 50M, which would flatten the small sizes into invisibility on a
    shared axis."""
    # Only implementations measured in both modes. A config swept single-threaded only would contribute a lone bar
    # with nothing to compare it against.
    labels = sorted(label for label, cfg in configs.items()
                    if any(p.mt == 0 for p in cfg) and any(p.mt == 1 for p in cfg))
    omitted = sorted(set(configs) - set(labels))
    row_counts = baseline_row_counts(configs, 0)
    if not labels or not row_counts:
        return

    fig, axes = plt.subplots(1, len(row_counts), figsize=(2.9 * len(row_counts) + 1.0, 4.2), squeeze=False)

    for row_index, rows in enumerate(row_counts):
        ax = axes[0][row_index]
        for label_index, label in enumerate(labels):
            single = configs[label].get(
                Point(BASELINE["key_type"], BASELINE["distribution"], rows, BASELINE["cols"], 0), {}).get(metric)
            multi = configs[label].get(
                Point(BASELINE["key_type"], BASELINE["distribution"], rows, BASELINE["cols"], 1), {}).get(metric)

            for offset, value, mt in ((-0.19, single, 0), (0.19, multi, 1)):
                if value is None or value != value:
                    continue
                ax.bar(label_index + offset, value, width=0.34, color=THREADING_COLORS[mt], linewidth=0,
                       label=MT_NAME[mt] if (row_index, label_index) == (0, 0) else None)

            if single and multi and multi > 0 and single == single and multi == multi:
                ax.annotate(f"{single / multi:.1f}x", xy=(label_index, max(single, multi)),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK_MUTED)

        ax.set_title(f"{human(rows)} rows", color=INK, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.margins(y=0.16)
        if row_index == 0:
            ax.set_ylabel(METRICS[metric], fontsize=9)
        _style_axis(ax)

    footnote = ("Number above each pair is the speedup, single-threaded time divided by multithreaded time. "
                f"{BASELINE['key_type']} / {BASELINE['distribution']} / {BASELINE['cols']} key column.")
    if omitted:
        footnote += f" Not shown (measured in one threading mode only): {', '.join(omitted)}."
    finish(fig, axes[0][0], "Single-threaded vs multithreaded -- total operator time", note, out_path,
           footnote=footnote)


# ------------------------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------------------------

def print_table(configs, metric="real_time_s"):
    print(f"\n{METRICS[metric]}:")
    header = ["key type", "distribution", "rows", "cols", "mt"] + sorted(configs)
    print("  " + "  ".join(f"{h:>13}" for h in header))
    for point in sorted({p for cfg in configs.values() for p in cfg}):
        cells = [point.key_type, point.distribution, human(point.rows), str(point.cols),
                 MT_SHORT.get(point.mt, point.mt)]
        for label in sorted(configs):
            value = configs[label].get(point, {}).get(metric, float("nan"))
            cells.append(f"{value:.6f}" if value == value else "-")
        print("  " + "  ".join(f"{c:>13}" for c in cells))


def report_coverage(configs):
    """Per-config point inventory against the union of all configs.

    A point present in one JSON and absent from another silently vanishes from the figures, where it reads as a
    measurement rather than an omission. The usual cause is that the variants were built from branches whose
    build_matrix() differed, so the configs are not comparable point for point."""
    union = set()
    for cfg in configs.values():
        union |= set(cfg)

    print(f"\npoint coverage ({len(union)} distinct points across all configs):")
    for label in sorted(configs):
        missing = sorted(union - set(configs[label]))
        suffix = f"  MISSING {len(missing)}" if missing else ""
        print(f"  {label:>16}: {len(configs[label])}/{len(union)}{suffix}")
        for point in missing[:6]:
            print(f"                    - {point.key_type}/{point.distribution}/{human(point.rows)}/"
                  f"cols{point.cols}/mt{point.mt}")
        if len(missing) > 6:
            print(f"                    ... and {len(missing) - 6} more")


def write_tidy_csv(configs, out_path):
    columns = list(dict.fromkeys(list(METRICS) + STEPS))
    with open(out_path, "w") as handle:
        handle.write("config,key_type,distribution,rows,cols,mt," + ",".join(columns) + "\n")
        for label in sorted(configs):
            for point in sorted(configs[label]):
                record = configs[label][point]
                values = ",".join(f"{record.get(m, float('nan')):.9g}" for m in columns)
                handle.write(f"{label},{point.key_type},{point.distribution},{point.rows},{point.cols},"
                             f"{point.mt},{values}\n")
    print(f"wrote {out_path}")


# ------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", default=["results"])
    parser.add_argument("--max-rows", type=int, default=None,
                        help="drop points above this row count (e.g. 50000000 to exclude a 100M ladder point that "
                             "only some configs have). Filters at load time; the JSONs are left alone.")
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        if os.path.isdir(item):
            # Skip the .meta.json provenance sidecars written alongside each result file.
            paths.extend(sorted(p for p in glob.glob(os.path.join(item, "*.json"))
                                if not p.endswith(".meta.json")))
        else:
            paths.append(item)
    if not paths:
        parser.error("no .json files found")

    results_dir = args.inputs[0] if os.path.isdir(args.inputs[0]) else (os.path.dirname(paths[0]) or ".")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    configs, contexts = collect(paths)
    if args.max_rows is not None:
        configs = {label: {p: r for p, r in cfg.items() if p.rows <= args.max_rows}
                   for label, cfg in configs.items()}
        print(f"dropping points above {human(args.max_rows)} rows")
    configs = {label: cfg for label, cfg in configs.items() if cfg}
    if not configs:
        parser.error("no benchmark entries parsed from the JSON -- check the benchmark names")

    note = binding_note(contexts)
    if note.startswith("WARNING"):
        print(f"!! {note}")

    mt_values = sorted({p.mt for cfg in configs.values() for p in cfg})
    axis_values = {axis: points_on_axis(configs, axis) for axis in AXES}

    print(f"loaded configs: {', '.join(sorted(configs))}")
    print(f"provenance: {note}")
    for axis in AXES:
        print(f"  {axis:>13} axis: {[axis_tick_label(axis, v) for v in axis_values[axis]]}")
    report_coverage(configs)
    if not any(axis_values.values()):
        parser.error(f"no points matched the baseline {BASELINE} -- does BASELINE still match the benchmark?")

    for metric in METRICS:
        plot_metric_axes(configs, metric, mt_values, axis_values, note,
                         os.path.join(plots_dir, f"{metric}__axes.png"))

    for mt in mt_values:
        plot_step_share(configs, mt, note, os.path.join(plots_dir, f"step_share__{MT_SHORT[mt]}.png"))

    if set(mt_values) >= {0, 1}:
        plot_threading_comparison(configs, note, os.path.join(plots_dir, "threading__total_time.png"))

    write_tidy_csv(configs, os.path.join(results_dir, "results.csv"))
    print_table(configs, "real_time_s")


if __name__ == "__main__":
    main()