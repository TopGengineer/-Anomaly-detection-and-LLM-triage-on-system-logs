"""Download the HDFS dataset.

    python scripts/download_hdfs.py --sample   # 2k-line sample (fast, no labels)
    python scripts/download_hdfs.py --full     # HDFS_v1 from Zenodo (~1.5 GB) + labels
"""

import argparse

from logtriage.config import HDFS
from logtriage.data.download import download_full, download_sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="2k-line GitHub sample")
    group.add_argument("--full", action="store_true", help="full HDFS_v1 from Zenodo")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.sample:
        download_sample(HDFS, force=args.force)
    else:
        download_full(HDFS, force=args.force)


if __name__ == "__main__":
    main()
