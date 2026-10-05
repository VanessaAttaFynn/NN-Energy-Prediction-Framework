"""CAL-GATE: applicability check and pilot recalibration (Algorithm 2).

Implements Eqs. (10)-(12):
  (10) d(x*) = min_i || S^-1 (x* - x_i) ||_2 over the calibration (training) set,
       with S = diag(feature standard deviations of the training set);
  (11) delta = 95th percentile of leave-one-out nearest-neighbour distances of
       the calibration set (computed over unique configurations, so repeated
       runs of the same configuration do not shrink the threshold to zero);
  (12) alpha = median_k (E_k / E_hat_k) over a labelled pilot set P,
       E_hat_pilot = alpha * E_hat.

Status rules (Algorithm 2, lines 4-7):
  within support      : family and hardware seen in training and d(x*) <= delta
  recalibration req.  : otherwise, when no pilot set is available
  pilot calibrated    : otherwise, when a pilot set is available
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class CalGate:
    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile

    def fit(self, X_train: np.ndarray, families=None, hardware=None) -> "CalGate":
        X = np.asarray(X_train, dtype=float)
        s = X.std(axis=0)
        s[s == 0] = 1.0                       # constant features carry no distance
        self.scale_ = s
        Z = np.unique(X / s, axis=0)          # unique configurations
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(Z)
        if len(Z) > 1:
            d_loo = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
            self.delta_ = float(np.quantile(d_loo, self.quantile))
        else:
            self.delta_ = 0.0
        self.families_ = set(families) if families is not None else None
        self.hardware_ = set(hardware) if hardware is not None else None
        return self

    def distance(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=float) / self.scale_
        return self.nn_.kneighbors(Z)[0][:, 0]

    def check(self, X: np.ndarray, families=None, hardware=None) -> dict:
        """Return per-row support distance and the two flag components."""
        d = self.distance(X)
        out_dist = d > self.delta_
        n = len(d)
        out_label = np.zeros(n, dtype=bool)
        if families is not None and self.families_ is not None:
            out_label |= ~np.isin(np.asarray(families), list(self.families_))
        if hardware is not None and self.hardware_ is not None:
            out_label |= ~np.isin(np.asarray(hardware), list(self.hardware_))
        return {"distance": d, "flag_distance": out_dist,
                "flag_label": out_label, "flag": out_dist | out_label}


def pilot_alpha(E_pilot: np.ndarray, E_hat_pilot: np.ndarray) -> float:
    """Eq. (12): robust multiplicative factor from a labelled pilot set."""
    return float(np.median(np.asarray(E_pilot, float) / np.asarray(E_hat_pilot, float)))
