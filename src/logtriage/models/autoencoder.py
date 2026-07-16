"""Autoencoder anomaly detector — the core detection model.

Trained on **normal** block-count vectors only; it learns to reconstruct normal
sessions, and the per-sample reconstruction error becomes the anomaly score.
No attack labels are used in training (they are for evaluation only). The score
is oriented so higher = more anomalous — identical to the Isolation Forest
baseline — so the same metrics harness evaluates both.

Preprocessing (log1p + standardisation) is fit on the normal training data only,
so nothing about the test period leaks into the model or its scaling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _AE(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...], latent: int):
        super().__init__()
        dims = [d_in, *hidden, latent]
        enc: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc[:-1])  # no ReLU on the latent layer
        dec: list[nn.Module] = []
        rdims = dims[::-1]
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        self.decoder = nn.Sequential(*dec[:-1])  # linear output layer

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderDetector:
    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16),
        latent: int = 8,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_frac: float = 0.1,
        patience: int = 6,
        seed: int = 0,
    ):
        self.hidden, self.latent = hidden, latent
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.val_frac, self.patience, self.seed = val_frac, patience, seed
        self.model: _AE | None = None
        self.mean_ = self.std_ = None
        self.history_: list[tuple[float, float]] = []

    # -- preprocessing fit on normal training data only --
    def _fit_scaler(self, X: np.ndarray) -> None:
        Z = np.log1p(np.asarray(X, dtype=np.float32))
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-6
        self._d_in = Z.shape[1]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.log1p(np.asarray(X, dtype=np.float32))
        return (Z - self.mean_) / self.std_

    def fit(self, X_normal: np.ndarray) -> "AutoEncoderDetector":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self._fit_scaler(X_normal)
        Z = self._transform(X_normal)

        # random train/val split *within the normal set* for early stopping
        idx = rng.permutation(len(Z))
        n_val = max(1, int(len(Z) * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr = torch.from_numpy(Z[tr_idx])
        Xval = torch.from_numpy(Z[val_idx])

        self.model = _AE(self._d_in, self.hidden, self.latent)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr))
            for s in range(0, len(Xtr), self.batch_size):
                b = Xtr[perm[s : s + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(b), b)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vtr = loss_fn(self.model(Xtr), Xtr).item()
                vval = loss_fn(self.model(Xval), Xval).item()
            self.history_.append((vtr, vval))
            if vval < best_val - 1e-6:
                best_val, best_state, bad = vval, {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample reconstruction MSE; higher = more anomalous."""
        assert self.model is not None, "call fit() first"
        Z = torch.from_numpy(self._transform(X))
        self.model.eval()
        with torch.no_grad():
            recon = self.model(Z)
            err = ((recon - Z) ** 2).mean(dim=1).numpy()
        return err
