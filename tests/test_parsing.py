import pandas as pd

from logtriage.data.hdfs import build_block_sessions, explode_blocks
from logtriage.parsing.drain_parser import BLOCK_ID_RE, _hdfs_timestamp


def test_block_id_regex():
    content = (
        "10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864"
    )
    assert BLOCK_ID_RE.findall(content) == ["blk_7128370237687728475"]
    assert BLOCK_ID_RE.findall("delete blk_-1608999687919862906") == [
        "blk_-1608999687919862906"
    ]
    assert BLOCK_ID_RE.findall("no block here") == []


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
