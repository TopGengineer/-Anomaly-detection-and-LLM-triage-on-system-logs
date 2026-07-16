"""Parse a raw log with Drain into the dataset's canonical event CSV.

    python scripts/parse.py --dataset HDFS --input data/raw/HDFS_2k.log
    python scripts/parse.py --dataset BGL  --input data/raw/BGL_2k.log
    python scripts/parse.py --dataset BGL  --input data/raw/BGL.log
"""

import argparse
from pathlib import Path

from logtriage.config import DATA_INTERIM, DATASETS
from logtriage.parsing.drain_parser import parse_log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--input", type=Path, required=True, help="raw log file")
    ap.add_argument("--outdir", type=Path, default=DATA_INTERIM)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found — run scripts/download.py first")
    parse_log(args.input, DATASETS[args.dataset], outdir=args.outdir)


if __name__ == "__main__":
    main()
