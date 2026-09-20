#!/usr/bin/env python3
"""Merge per-branch SortEvaluation CSVs into one file for plot_results.py.

    python3 merge_results.py ~/sort-results/raw -o ~/sort-results/all_results.csv

Filenames are irrelevant -- ROUTINE, BRANCH and COMMIT are read from inside each row. Drop every
branch's sort_results.csv into one directory under any unique name and run this.

Why not `cat`: if one branch has MERGE_PATH_US and another doesn't, concatenating shifts every
column after it and silently corrupts the data. This aligns by column name.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: pandas. Install with: pip install pandas")


def phase_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.endswith("_US") and c != "TOTAL_US"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indir", type=Path, help="directory holding the per-branch CSVs (searched recursively)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output CSV (default: <indir>/../all_results.csv)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="keep byte-identical input files instead of dropping the copies")
    args = parser.parse_args()

    output = args.output or args.indir.parent / "all_results.csv"

    # Any filename works, with or without a .csv extension -- identity lives in the rows, not the name.
    # A file only counts if its header actually looks like SortEvaluation output.
    def is_results_file(path: Path) -> bool:
        if not path.is_file() or path.suffix not in ("", ".csv") or path.resolve() == output.resolve():
            return False
        try:
            with path.open() as handle:
                return handle.readline().startswith("ROUTINE,")
        except (OSError, UnicodeDecodeError):
            return False

    files = sorted(p for p in args.indir.rglob("*") if is_results_file(p))
    if not files:
        sys.exit(f"no SortEvaluation CSVs under {args.indir} "
                 "(expected files whose first line starts with 'ROUTINE,')")

    # --- read, dropping files that are byte-identical copies of one already seen -------------------------------
    frames, headers, seen = [], set(), {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen and not args.keep_duplicates:
            print(f"  duplicate of {seen[digest].name}, skipping: {path.name}")
            continue
        seen[digest] = path

        frame = pd.read_csv(path)
        if frame.empty:
            print(f"  empty, skipping: {path.name}")
            continue
        headers.add(tuple(frame.columns))
        frames.append(frame)
        print(f"  {path.name:<28} {len(frame):>5} rows  "
              f"{sorted(set(frame['ROUTINE'])) if 'ROUTINE' in frame else '(no ROUTINE column)'}")

    if not frames:
        sys.exit("nothing to merge")

    merged = pd.concat(frames, ignore_index=True)

    # --- column mismatch between branches --------------------------------------------------------------------
    phases = phase_columns(merged)
    if len(headers) > 1:
        print("\n! branches produced different columns:")
        absent = merged[phases].isna().any()
        for column in absent[absent].index:
            routines = sorted(set(merged[merged[column].isna()]["ROUTINE"]))
            print(f"    {column} absent for {routines}")
        merged[phases] = merged[phases].fillna(0)
        print("  filled with 0 -- correct if that phase does not exist in those routines;")
        print("  re-run the branch if it simply predates the column.")

    # --- provenance checks -----------------------------------------------------------------------------------
    print("\nprovenance")
    if {"ROUTINE", "COMMIT"} <= set(merged.columns):
        per_routine = merged.groupby("ROUTINE").agg(
            rows=("ROUTINE", "size"),
            commits=("COMMIT", lambda s: sorted(set(s))),
            branches=("BRANCH", lambda s: sorted(set(s))) if "BRANCH" in merged else ("COMMIT", "size"),
        )
        for routine, row in per_routine.iterrows():
            print(f"  {routine:<12} {row['rows']:>5} rows   {row['branches']} @ {row['commits']}")

        mixed = per_routine[per_routine["commits"].apply(len) > 1]
        if not mixed.empty:
            print(f"  ! {list(mixed.index)} span more than one commit -- results from different code are mixed")

        owners: dict[str, list[str]] = {}
        for routine, row in per_routine.iterrows():
            for commit in row["commits"]:
                owners.setdefault(commit, []).append(str(routine))
        for commit, routines in owners.items():
            if len(routines) > 1:
                print(f"  ! commit {commit} is tagged as {routines} -- one of these is mislabelled")

    if "MATERIALIZED" in merged.columns and "HARNESS" in merged.columns:
        bad = merged[(merged["HARNESS"] == "sql") & (merged["MATERIALIZED"] == 0)]
        if not bad.empty:
            print(f"  ! sql rows with MATERIALIZED=0 for {sorted(set(bad['ROUTINE']))} -- "
                  "the ForceMaterialization patch was missing there")

    # --- the one check the CSV cannot make -------------------------------------------------------------------
    manifests = sorted(args.indir.rglob("manifest.txt"))
    if manifests:
        digests: dict[str, list[str]] = {}
        for manifest in manifests:
            for line in manifest.read_text().splitlines():
                if line.startswith("binary_sha256"):
                    digests.setdefault(line.split(":", 1)[1].strip(), []).append(manifest.parent.name)
        repeated = {d: p for d, p in digests.items() if len(p) > 1}
        if repeated:
            for digest, places in repeated.items():
                print(f"  ! same binary ({digest[:12]}) used for {places} -- a checkout without a rebuild")
        else:
            print(f"  ok  {len(digests)} distinct binaries across {len(manifests)} manifests")
    else:
        print("  note: no manifest.txt files found, so a forgotten rebuild cannot be detected here")
        print("        (BRANCH/COMMIT come from git at run time and look correct even with a stale binary)")

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"\nwrote {output}: {len(merged)} rows, {merged['ROUTINE'].nunique()} routines")
    print(f"next: python3 plot_results.py --csv {output} --outdir figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())