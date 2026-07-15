"""Parse a raw HDFS log with Drain into the canonical event CSV.

    python scripts/parse_hdfs.py --input data/raw/HDFS_2k.log
    python scripts/parse_hdfs.py --input data/raw/HDFS.log
"""

import argparse
from pathlib import Path

from logtriage.config import DATA_INTERIM, HDFS
from logtriage.parsing.drain_parser import parse_log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="raw log file")
    ap.add_argument("--outdir", type=Path, default=DATA_INTERIM)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found — run scripts/download_hdfs.py first")
    parse_log(args.input, HDFS, outdir=args.outdir)


if __name__ == "__main__":
    main()
