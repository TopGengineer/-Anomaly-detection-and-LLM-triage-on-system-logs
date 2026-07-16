import pandas as pd

from logtriage.data.hdfs import build_block_sessions, explode_blocks
from logtriage.eda import data_quality_report
from logtriage.parsing.drain_parser import BLOCK_ID_RE, _canonical_bgl, _hdfs_timestamp


def test_block_id_regex():
    content = (
        "10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864"
    )
    assert BLOCK_ID_RE.findall(content) == ["blk_7128370237687728475"]
    assert BLOCK_ID_RE.findall("delete blk_-1608999687919862906") == [
        "blk_-1608999687919862906"
    ]
    assert BLOCK_ID_RE.findall("no block here") == []


def test_bgl_canonical_timestamp_and_label():
    # mimics Drain's structured CSV columns for BGL
    df = pd.DataFrame(
        {
            "Label": ["-", "KERNDTLB"],  # '-' normal, tag = alert
            "Time": ["2005-06-03-15.42.50.675872", "2005-06-04-08.01.02.000123"],
            "EventId": ["e1", "e2"],
            "EventTemplate": ["t1", "t2"],
            "Node": ["R02", "R03"],
            "Component": ["KERNEL", "KERNEL"],
            "Level": ["INFO", "FATAL"],
            "Content": ["ok", "boom"],
        }
    )
    out = _canonical_bgl(df)
    assert out["label"].tolist() == [0, 1]
    # microsecond resolution preserved
    assert out["timestamp"][0] == pd.Timestamp("2005-06-03 15:42:50.675872")
    assert out["timestamp"][0].microsecond == 675872


def test_bgl_canonical_drops_unparseable_timestamps():
    df = pd.DataFrame(
        {
            "Label": ["-", "-"],
            "Time": ["2005-06-03-15.42.50.675872", "GARBAGE"],
            "EventId": ["e1", "e2"],
            "EventTemplate": ["t1", "t2"],
            "Node": ["R02", "R03"],
            "Component": ["KERNEL", "KERNEL"],
            "Level": ["INFO", "INFO"],
            "Content": ["ok", "bad"],
        }
    )
    out = _canonical_bgl(df)
    assert len(out) == 1  # the garbage-timestamp row is dropped


def test_hdfs_timestamp():
    ts = _hdfs_timestamp(pd.Series(["081109"]), pd.Series(["203515"]))
    assert ts[0] == pd.Timestamp("2008-11-09 20:35:15")
    # loghub stores fields as integers; a leading zero in Time can be lost
    ts = _hdfs_timestamp(pd.Series(["081109"]), pd.Series(["35515"]))
    assert ts[0] == pd.Timestamp("2008-11-09 03:55:15")


def _events():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2008-11-09 20:35:15", "2008-11-09 20:35:16", "2008-11-09 20:35:18"]
            ),
            "event_id": ["E1", "E2", "E1"],
            "block_ids": ["blk_1", "blk_1 blk_2", ""],
        }
    )


def test_explode_blocks():
    per_block = explode_blocks(_events())
    # line 2 belongs to two sessions, line 3 (no block) is dropped
    assert len(per_block) == 3
    assert sorted(per_block["block_id"].unique()) == ["blk_1", "blk_2"]


def test_build_block_sessions_with_labels():
    labels = pd.Series([0, 1], index=["blk_1", "blk_2"], name="label")
    sessions = build_block_sessions(_events(), labels)
    sessions = sessions.set_index("block_id")
    assert sessions.loc["blk_1", "event_seq"] == ["E1", "E2"]
    assert sessions.loc["blk_1", "n_events"] == 2
    assert sessions.loc["blk_2", "label"] == 1


def test_data_quality_report_detects_ties_and_cleanliness():
    ev = _events()  # ts at :15, :16, :18 -> no ties, sorted, no nulls
    ev = ev.assign(
        template=["t1", "t2", "t1"],
        level=["INFO", "INFO", "WARN"],
        content=["a", "b", "c"],
    )
    rep = data_quality_report(ev)
    assert rep["null_cells"] == 0
    assert rep["exact_duplicate_rows"] == 0
    assert rep["time_monotonic_nondecreasing"] is True
    assert rep["same_timestamp_tie_fraction"] == 0.0
    assert rep["n_templates"] == 2
