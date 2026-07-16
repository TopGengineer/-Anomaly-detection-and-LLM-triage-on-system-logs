"""Dataset definitions: sources, log formats, and Drain parameters.

Log formats and Drain hyperparameters (depth, similarity threshold, masking
regexes) follow the loghub benchmark settings published by the logparser
authors, so parsed templates are comparable to published results.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    # Drain settings
    log_format: str
    masking_regexes: list[str]
    drain_depth: int
    drain_sim_threshold: float
    # Sources
    sample_url: str
    full_url: str
    full_members: list[str] = field(default_factory=list)  # files to extract from the zip


HDFS = DatasetConfig(
    name="HDFS",
    log_format="<Date> <Time> <Pid> <Level> <Component>: <Content>",
    masking_regexes=[
        r"blk_(|-)[0-9]+",            # block ids
        r"(\d+\.){3}\d+(:\d+)?",      # IP[:port]
    ],
    drain_depth=4,
    drain_sim_threshold=0.5,
    sample_url="https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    full_url="https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
    full_members=["HDFS.log", "preprocessed/anomaly_label.csv"],
)

BGL = DatasetConfig(
    name="BGL",
    # Label is the first field: '-' = non-alert, anything else = an alert tag.
    log_format=(
        "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> "
        "<Type> <Component> <Level> <Content>"
    ),
    masking_regexes=[
        r"core\.\d+",
        r"(0x)?[0-9a-fA-F]{8,}",      # hex addresses
        r"(\d+\.){3}\d+(:\d+)?",      # IPs
        r"\b\d+\b",                   # bare integers
    ],
    drain_depth=4,
    drain_sim_threshold=0.5,
    sample_url="https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
    full_url="https://zenodo.org/records/8196385/files/BGL.zip?download=1",
    full_members=["BGL.log"],  # labels are inline (Label field), no separate file
)

DATASETS = {"HDFS": HDFS, "BGL": BGL}
