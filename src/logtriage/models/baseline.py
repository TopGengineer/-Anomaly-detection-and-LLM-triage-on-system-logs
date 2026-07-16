"""Isolation Forest baseline — the unsupervised detector the autoencoder must beat.

Fit on training block-count vectors with no labels (labels are for evaluation
only). The anomaly score is `-score_samples`, so higher = more anomalous, the
same orientation the autoencoder's reconstruction error will use — evaluation
code stays model-agnostic.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    def __init__(self, n_estimators: int = 100, random_state: int = 0, log1p: bool = True):
        # log1p tames the heavy-tailed count columns (a few templates dominate)
        self.log1p = log1p
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.log1p(X) if self.log1p else X

    def fit(self, X_train: np.ndarray) -> "IsolationForestBaseline":
        self.clf.fit(self._prep(X_train))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores, higher = more anomalous."""
        return -self.clf.score_samples(self._prep(X))
