"""Download loghub datasets: 2k-line samples (GitHub) or full archives (Zenodo)."""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from logtriage.config import DATA_RAW, DatasetConfig

CHUNK = 1 << 20  # 1 MiB


def _stream_to_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                bar.update(len(chunk))


def download_sample(cfg: DatasetConfig, force: bool = False) -> Path:
    """Fetch the 2k-line sample log. Returns the local path."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW / cfg.sample_url.rsplit("/", 1)[-1]
    if dest.exists() and not force:
        print(f"already present: {dest}")
        return dest
    _stream_to_file(cfg.sample_url, dest)
    return dest


def download_full(cfg: DatasetConfig, force: bool = False) -> list[Path]:
    """Fetch the full dataset zip from Zenodo and extract the needed members.

    The archive is deleted after extraction (HDFS_v1 alone is ~1.5 GB zipped).
    Returns the extracted file paths.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    wanted = {m: DATA_RAW / Path(m).name for m in cfg.full_members}
    if all(p.exists() for p in wanted.values()) and not force:
        print("already present: " + ", ".join(str(p) for p in wanted.values()))
        return list(wanted.values())

    archive = DATA_RAW / f"{cfg.name}_full.zip"
    if not archive.exists() or force:
        _stream_to_file(cfg.full_url, archive)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member, dest in wanted.items():
            if member not in names:
                raise KeyError(
                    f"{member!r} not found in {archive.name}; archive contains: "
                    + ", ".join(sorted(names)[:20])
                )
            with zf.open(member) as src, open(dest, "wb") as out:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    out.write(chunk)
            print(f"extracted {member} -> {dest}")

    archive.unlink()
    return list(wanted.values())
