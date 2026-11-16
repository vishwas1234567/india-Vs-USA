"""Trajectory forecasting for tracked road users.

* LSTMForecaster  - single-layer LSTM encoder + linear multi-horizon head, implemented in NumPy with exact
                    back-propagation through time (no framework needed; trains on CPU in seconds). Gradients are verified
                    against finite differences in the tests.
* Baselines       - zero velocity, constant velocity (last / mean of observed steps), ridge regression.
* build_windows   - turns trajectories.csv (from irp.static) into (observation, future) windows.
* ADE / FDE       - reported in body widths (scale free) and in pixels at 960 px image width.

Usage:  python -m irp.predict --runs results/b_main results/a_kalman --out results/_forecast
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from iscvd.metrics import ade_fde


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMForecaster:
    """X (B, T, D) -> future displacements (B, horizon, 2)."""

    def __init__(self, input_dim: int = 4, hidden: int = 32, horizon: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        H, D = hidden, input_dim
        self.D, self.H, self.horizon = D, H, horizon
        sc = 1.0 / np.sqrt(H)
        b = np.zeros(4 * H)
        b[H:2 * H] = 1.0                                     # forget-gate bias
        self.p: Dict[str, np.ndarray] = {
            "Wx": rng.normal(0, sc, (4 * H, D)), "Wh": rng.normal(0, sc, (4 * H, H)), "b": b,
            "Wo": rng.normal(0, sc, (horizon * 2, H)), "bo": np.zeros(horizon * 2)}

    # ---- forward / backward
    def forward(self, X: np.ndarray):
        p, H = self.p, self.H
        B, T, _ = X.shape
        h, c = np.zeros((B, H)), np.zeros((B, H))
        cache = []
        for t in range(T):
            z = X[:, t] @ p["Wx"].T + h @ p["Wh"].T + p["b"]
            i, f, o, g = _sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H]), _sigmoid(z[:, 2 * H:3 * H]), np.tanh(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            cache.append((X[:, t], h, c, i, f, o, g, tc))
            h, c = o * tc, c_new
        y = h @ p["Wo"].T + p["bo"]
        return y.reshape(B, self.horizon, 2), (cache, h)

    def backward(self, dY: np.ndarray, pack) -> Dict[str, np.ndarray]:
        cache, h_last = pack
        p, H = self.p, self.H
        B = dY.shape[0]
        dy = dY.reshape(B, -1)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"], g["bo"] = dy.T @ h_last, dy.sum(0)
        dh, dc = dy @ p["Wo"], np.zeros((B, H))
        for x, h_prev, c_prev, i, f, o, gg, tc in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, df, dg, dc_prev = dc * gg, dc * c_prev, dc * i, dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - gg ** 2)], 1)
            g["Wx"] += dz.T @ x
            g["Wh"] += dz.T @ h_prev
            g["b"] += dz.sum(0)
            dh, dc = dz @ p["Wh"], dc_prev
        return g

    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray):
        pred, pack = self.forward(X)
        diff = pred - Y
        loss = float(np.mean(diff ** 2))
        return loss, self.backward(2.0 * diff / diff.size, pack)

    # ---- training
    def fit(self, X: np.ndarray, Y: np.ndarray, Xv: Optional[np.ndarray] = None, Yv: Optional[np.ndarray] = None,
            epochs: int = 150, batch: int = 64, lr: float = 3e-3, clip: float = 5.0, weight_decay: float = 1e-5,
            patience: int = 25, seed: int = 0, verbose: bool = False) -> Dict[str, List[float]]:
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        step, best, best_p, bad = 0, np.inf, None, 0
        hist: Dict[str, List[float]] = {"train": [], "val": []}
        for ep in range(epochs):
            order = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), batch):
                idx = order[s:s + batch]
                loss, grads = self.loss_and_grads(X[idx], Y[idx])
                tot += loss * len(idx)
                norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
                scale = min(1.0, clip / (norm + 1e-12))
                step += 1
                for k in self.p:
                    gk = grads[k] * scale + weight_decay * self.p[k]
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v[k] = 0.999 * v[k] + 0.001 * gk ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
            hist["train"].append(tot / len(X))
            if Xv is not None and len(Xv):
                val = float(np.mean((self.predict(Xv) - Yv) ** 2))
                hist["val"].append(val)
                if val < best - 1e-6:
                    best, best_p, bad = val, {k: x.copy() for k, x in self.p.items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if verbose and ep % 10 == 0:
                print(ep, hist["train"][-1], hist["val"][-1] if hist["val"] else None)
        if best_p is not None:
            self.p = best_p
        return hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def save(self, path: str) -> None:
        np.savez(path, D=self.D, H=self.H, horizon=self.horizon, **self.p)

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        z = np.load(path)
        m = cls(int(z["D"]), int(z["H"]), int(z["horizon"]))
        m.p = {k: z[k] for k in m.p}
        return m


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_predict(X: np.ndarray, horizon: int, kind: str) -> np.ndarray:
    """Baselines from the first two features (dx, dy per step, in body widths)."""
    d = X[:, :, :2]
    B = len(X)
    if kind == "zero":
        v = np.zeros((B, 2))
    elif kind == "cv_last":
        v = d[:, -1]
    elif kind == "cv_mean":
        v = d.mean(1)
    else:
        raise ValueError(kind)
    return v[:, None, :] * np.arange(1, horizon + 1)[None, :, None]


class RidgeForecaster:
    def __init__(self, alpha: float = 1.0):
        self.alpha, self.W = alpha, None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeForecaster":
        A = np.c_[X.reshape(len(X), -1), np.ones(len(X))]
        self.W = np.linalg.solve(A.T @ A + self.alpha * np.eye(A.shape[1]), A.T @ Y.reshape(len(Y), -1))
        self.h = Y.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.c_[X.reshape(len(X), -1), np.ones(len(X))] @ self.W).reshape(len(X), self.h, 2)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_trajectories(path: str) -> Dict[int, np.ndarray]:
    """track_id -> array (n, 5) of [frame, cx, cy, w, h] (image coordinates at 960 px width)."""
    tr: Dict[int, List[List[float]]] = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            tr[int(r["track_id"])].append([float(r[k]) for k in ("frame", "cx", "cy", "w", "h")])
    return {k: np.array(sorted(v)) for k, v in tr.items()}


def resample(traj: np.ndarray, step: int) -> np.ndarray:
    """Linear resampling to every `step` frames: array (m, 4) of [cx, cy, w, h]."""
    f = traj[:, 0]
    grid = np.arange(f[0], f[-1] + 1e-9, step)
    return np.stack([np.interp(grid, f, traj[:, j]) for j in (1, 2, 3, 4)], 1)


def build_windows(trajs: Dict[int, np.ndarray], obs: int = 4, pred: int = 5, step: int = 3, image_h: float = 540.0,
                  min_motion_bw: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, obs, 4), Y (N, pred, 2), ref width (N,), track id (N,).

    Features per observed step: [dx/w_ref, dy/w_ref, w/w_ref, cy/image_h]; targets: future displacement / w_ref.
    """
    X, Y, W, T = [], [], [], []
    for tid, tr in trajs.items():
        s = resample(tr, step)
        for i in range(0, len(s) - (obs + 1 + pred) + 1):
            win = s[i:i + obs + 1 + pred]
            past, fut = win[:obs + 1], win[obs + 1:]
            wr = max(float(past[-1, 2]), 4.0)
            d = np.diff(past[:, :2], axis=0) / wr
            feats = np.c_[d, past[1:, 2] / wr, past[1:, 1] / image_h]
            target = (fut[:, :2] - past[-1, :2]) / wr
            if np.abs(target).max() < min_motion_bw:
                continue
            X.append(feats); Y.append(target); W.append(wr); T.append(tid)
    if not X:
        return (np.zeros((0, obs, 4)), np.zeros((0, pred, 2)), np.zeros(0), np.zeros(0, int))
    return np.array(X), np.array(Y), np.array(W), np.array(T)


def split_by_track(T: np.ndarray, fractions=(0.7, 0.15, 0.15), seed: int = 0):
    ids = np.unique(T)
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    parts = [set(ids[:a]), set(ids[a:b]), set(ids[b:])]
    return [np.nonzero([t in p for t in T])[0] for p in parts]


def score(pred: np.ndarray, Y: np.ndarray, W: np.ndarray) -> Dict[str, float]:
    bw = ade_fde(pred, Y)
    px = ade_fde(pred * W[:, None, None], Y * W[:, None, None])
    return {"ADE_bw": bw["ADE"], "FDE_bw": bw["FDE"], "ADE_px960": px["ADE"], "FDE_px960": px["FDE"]}


def bootstrap_ci(err: np.ndarray, groups: Optional[np.ndarray] = None, n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap CI of the mean. With `groups` (track ids) whole tracks are resampled, because windows of one
    track are strongly correlated and window-level resampling would understate the uncertainty."""
    rng = np.random.default_rng(seed)
    if groups is None:
        means = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(n)]
    else:
        ids = np.unique(groups)
        sums = np.array([err[groups == g].sum() for g in ids])
        cnt = np.array([(groups == g).sum() for g in ids], float)
        means = []
        for _ in range(n):
            pick = rng.integers(0, len(ids), len(ids))
            means.append(sums[pick].sum() / cnt[pick].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_experiment(train_runs: Sequence[str], test_runs: Sequence[str] = (), obs: int = 4, pred: int = 5, step: int = 3,
                   hidden: int = 32, seeds: int = 3) -> Dict[str, object]:
    def windows(runs):
        parts = []
        for k, r in enumerate(runs):
            x, y, w, t = build_windows(load_trajectories(str(Path(r) / "trajectories.csv")), obs, pred, step)
            parts.append((x, y, w, t + 100000 * k))          # keep track ids unique across runs
        return tuple(np.concatenate([p[i] for p in parts]) if parts else None for i in range(4))

    X, Y, W, T = windows(train_runs)
    tr, va, te = split_by_track(T)
    out: Dict[str, object] = {"obs_steps": obs, "pred_steps": pred, "step_frames": step, "windows": int(len(X)),
                              "tracks": int(len(np.unique(T))), "split_windows": [int(len(tr)), int(len(va)), int(len(te))]}
    sets = {"held_out_tracks": (X[te], Y[te], W[te], T[te])}
    if test_runs:
        Xa, Ya, Wa, Ta = windows(test_runs)
        sets["cross_scene"] = (Xa, Ya, Wa, Ta + 10 ** 7)
    models: Dict[str, object] = {}
    ridge = RidgeForecaster(alpha=1.0).fit(X[tr], Y[tr])
    lstm_runs = []
    for s in range(seeds):
        m = LSTMForecaster(4, hidden, pred, seed=s)
        m.fit(X[tr], Y[tr], X[va], Y[va], seed=s)
        lstm_runs.append(m)
    results: Dict[str, Dict[str, object]] = {}
    for name, (Xs, Ys, Ws, Ts) in sets.items():
        res: Dict[str, object] = {"n_windows": int(len(Xs))}
        for kind in ("zero", "cv_last", "cv_mean"):
            res[kind] = score(baseline_predict(Xs, pred, kind), Ys, Ws)
        res["ridge"] = score(ridge.predict(Xs), Ys, Ws)
        preds = [m.predict(Xs) for m in lstm_runs]
        res["lstm"] = score(np.mean(preds, 0), Ys, Ws)                     # seed ensemble
        res["lstm_single_seed_ADE_bw"] = [score(p, Ys, Ws)["ADE_bw"] for p in preds]
        e_cv = np.linalg.norm(baseline_predict(Xs, pred, "cv_mean") - Ys, axis=-1).mean(1)
        e_ls = np.linalg.norm(np.mean(preds, 0) - Ys, axis=-1).mean(1)
        res["n_tracks"] = int(len(np.unique(Ts)))
        res["ADE_bw_ci95_track_bootstrap"] = {"cv_mean": bootstrap_ci(e_cv, Ts), "lstm": bootstrap_ci(e_ls, Ts),
                                              "lstm_minus_cv_mean": bootstrap_ci(e_ls - e_cv, Ts)}
        results[name] = res
    out["results"] = results
    out["models"] = {"lstm_hidden": hidden, "lstm_seeds": seeds}
    out["_lstm0"] = lstm_runs[0]
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="training runs (folders with trajectories.csv)")
    ap.add_argument("--test-runs", nargs="*", default=[], help="additional scenes for cross-scene testing")
    ap.add_argument("--out", default="results/_forecast")
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--pred", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    res = run_experiment(a.runs, a.test_runs, a.obs, a.pred, hidden=a.hidden, seeds=a.seeds)
    model = res.pop("_lstm0")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "lstm.npz"))
    (out / "forecast.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
