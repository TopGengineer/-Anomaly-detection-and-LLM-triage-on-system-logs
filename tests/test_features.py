import pandas as pd

from logtriage.features.sessions import build_count_matrix
from logtriage.features.split import temporal_split


def _events():
    # 3 blocks with distinct start times; blk_c starts latest
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2008-11-09 20:00:00",  # a
                    "2008-11-09 20:00:02",  # a
                    "2008-11-09 20:05:00",  # b
                    "2008-11-09 20:10:00",  # c
                    "2008-11-09 20:10:01",  # c
                ]
            ),
            "event_id": ["E1", "E2", "E1", "E3", "E3"],
            "block_ids": ["blk_a", "blk_a", "blk_b", "blk_c", "blk_c"],
        }
    )


def test_count_matrix_shape_and_values():
    labels = pd.Series([0, 1, 0], index=["blk_a", "blk_b", "blk_c"], name="label")
    X, meta = build_count_matrix(_events(), labels)
    assert list(X.columns) == ["E1", "E2", "E3"]  # sorted template order
    # rows are ordered by session start: a, b, c
    assert list(X.index) == ["blk_a", "blk_b", "blk_c"]
    assert X.loc["blk_a"].tolist() == [1, 1, 0]
    assert X.loc["blk_c"].tolist() == [0, 0, 2]
    assert meta.loc["blk_a", "n_events"] == 2
    assert meta.loc["blk_c", "label"] == 0


def test_temporal_split_is_ordered_not_shuffled():
    labels = pd.Series([0, 1, 0], index=["blk_a", "blk_b", "blk_c"], name="label")
    _, meta = build_count_matrix(_events(), labels)
    split, summary = temporal_split(meta, train_frac=2 / 3)
    # earliest 2 of 3 blocks -> train; latest -> test
    assert split["blk_a"] == "train"
    assert split["blk_b"] == "train"
    assert split["blk_c"] == "test"
    assert summary.n_train == 2 and summary.n_test == 1
    # boundary is the start time of the last train block
    assert summary.boundary == pd.Timestamp("2008-11-09 20:05:00")


def test_split_rejects_bad_fraction():
    _, meta = build_count_matrix(_events())
    for bad in (0.0, 1.0, 1.5):
        try:
            temporal_split(meta, train_frac=bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
