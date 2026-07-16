"""Explore a parsed HDFS log: print a data-quality report and write EDA plots.

    python scripts/explore_hdfs.py --input data/interim/HDFS_2k.log_parsed.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from logtriage.config import PROJECT_ROOT
from logtriage.eda import data_quality_report, make_plots, print_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="parsed CSV")
    ap.add_argument(
        "--figdir", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"], keep_default_na=False)
    prefix = args.input.stem.split(".")[0]

    print_report(data_quality_report(df))
    figs = make_plots(df, args.figdir, prefix=prefix)
    print(f"\nwrote {len(figs)} figures to {args.figdir}:")
    for p in figs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
