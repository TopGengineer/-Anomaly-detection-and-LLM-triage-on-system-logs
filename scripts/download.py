"""Download a loghub dataset (sample or full).

    python scripts/download.py --dataset HDFS --sample
    python scripts/download.py --dataset HDFS --full
    python scripts/download.py --dataset BGL  --sample
    python scripts/download.py --dataset BGL  --full
"""

import argparse

from logtriage.config import DATASETS
from logtriage.data.download import download_full, download_sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="2k-line GitHub sample")
    group.add_argument("--full", action="store_true", help="full dataset from Zenodo")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    if args.sample:
        download_sample(cfg, force=args.force)
    else:
        download_full(cfg, force=args.force)


if __name__ == "__main__":
    main()
