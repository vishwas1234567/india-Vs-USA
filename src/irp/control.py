"""Vehicle models and controllers.

* BicycleModel  - kinematic bicycle (x, y, psi, v) with inputs (acceleration, steering angle)
* Path          - dense polyline reference with nearest-point projection (cross-track error, heading error, curvature)
* LQRLateral    - discrete-time LQR on the error dynamics [e_y, e_psi], gain-scheduled over speed, with curvature feed-forward
* MPC           - nonlinear model-predictive controller: shooting over N steps, L-BFGS-B with batched finite-difference
                  gradients, obstacle-avoidance penalty on predicted obstacle positions, actuator bounds, warm start
Sign convention: e_y > 0 means the vehicle is left of the path; steering delta > 0 turns left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class BicycleModel:
    L: float = 2.5
    max_steer: float = 0.5
    max_accel: float = 2.5
    max_brake: float = 6.0

    def step(self, s: np.ndarray, a: float, delta: float, dt: float) -> np.ndarray:
        x, y, psi, v = s
        a = float(np.clip(a, -self.max_brake, self.max_accel))
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        return np.array([x + v * math.cos(psi) * dt, y + v * math.sin(psi) * dt, psi + v / self.L * math.tan(delta) * dt,
                         max(0.0, v + a * dt)])


class Path:
    def __init__(self, xy: np.ndarray, spacing: float = 0.25):
        xy = np.asarray(xy, float)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        grid = np.arange(0, s[-1], spacing)
        self.x, self.y = np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])
        dx, dy = np.gradient(self.x), np.gradient(self.y)
        self.psi = np.unwrap(np.arctan2(dy, dx))
        ds = np.maximum(np.hypot(dx, dy), 1e-9)
        self.kappa = np.gradient(self.psi) / ds
        self.s = grid
        self.length = float(s[-1])

    @classmethod
    def straight(cls, length: float = 200.0, y: float = 0.0) -> "Path":
        return cls(np.array([[0.0, y], [length, y]]))

    def project(self, x, y):
        """Nearest dense point for scalars or arrays; returns (index, e_y, e_psi_ref, kappa, s) with e_y left-positive."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        flat_x, flat_y = x.reshape(-1), y.reshape(-1)
        d2 = (flat_x[:, None] - self.x[None]) ** 2 + (flat_y[:, None] - self.y[None]) ** 2
        i = d2.argmin(1)
        dx, dy = flat_x - self.x[i], flat_y - self.y[i]
        e_y = -np.sin(self.psi[i]) * dx + np.cos(self.psi[i]) * dy
        shape = x.shape
        return i.reshape(shape), e_y.reshape(shape), self.psi[i].reshape(shape), self.kappa[i].reshape(shape), self.s[i].reshape(shape)

    def errors(self, state: np.ndarray):
        i, e_y, psi_ref, kappa, s = self.project(state[0], state[1])
        return float(e_y), float(wrap(state[2] - psi_ref)), float(kappa), float(s)


class LQRLateral:
    def __init__(self, L: float = 2.5, dt: float = 0.05, q: Tuple[float, float] = (1.0, 0.5), r: float = 2.0,
                 max_steer: float = 0.5):
        self.L, self.dt, self.Q, self.R, self.max_steer = L, dt, np.diag(q), np.array([[r]]), max_steer
        self._gains = {}

    def gain(self, v: float) -> np.ndarray:
        key = round(max(v, 0.5) * 2) / 2.0                      # cache per 0.5 m/s
        if key not in self._gains:
            A = np.array([[1.0, key * self.dt], [0.0, 1.0]])
            B = np.array([[0.0], [key / self.L * self.dt]])
            P = solve_discrete_are(A, B, self.Q, self.R)
            self._gains[key] = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        return self._gains[key]

    def control(self, e_y: float, e_psi: float, kappa: float, v: float) -> float:
        ff = math.atan(self.L * kappa)
        fb = -float((self.gain(v) @ np.array([e_y, e_psi])).item())
        return float(np.clip(ff + fb, -self.max_steer, self.max_steer))


class SpeedController:
    """Proportional speed tracking with acceleration limits (longitudinal baseline)."""

    def __init__(self, kp: float = 1.2, a_max: float = 2.0, a_min: float = -5.0):
        self.kp, self.a_max, self.a_min = kp, a_max, a_min

    def control(self, v: float, v_ref: float) -> float:
        return float(np.clip(self.kp * (v_ref - v), self.a_min, self.a_max))


@dataclass
class Obstacle:
    """Predicted obstacle positions (N+1, 2) at the MPC time steps and a collision radius (m)."""
    traj: np.ndarray
    radius: float = 0.8
    kind: str = "obstacle"


class MPC:
    def __init__(self, N: int = 15, dt: float = 0.2, L: float = 2.5, a_bounds=(-6.0, 2.5), delta_bounds=(-0.45, 0.45),
                 w_y: float = 2.0, w_psi: float = 1.0, w_v: float = 0.4, w_a: float = 0.05, w_d: float = 0.5,
                 w_da: float = 0.1, w_dd: float = 2.0, w_obs: float = 120.0, w_lim: float = 300.0, ego_radius: float = 1.0, margin: float = 1.2,
                 max_iter: int = 25):
        self.N, self.dt, self.L = N, dt, L
        self.ab, self.db = a_bounds, delta_bounds
        self.w = dict(y=w_y, psi=w_psi, v=w_v, a=w_a, d=w_d, da=w_da, dd=w_dd, obs=w_obs, lim=w_lim)
        self.ego_r, self.margin, self.max_iter = ego_radius, margin, max_iter
        self.u_prev = np.zeros((N, 2))
        self.last_ms = 0.0

    # ---- batched rollout and cost: U has shape (B, N, 2)
    def _rollout(self, s0: np.ndarray, U: np.ndarray) -> np.ndarray:
        B = U.shape[0]
        S = np.empty((B, self.N + 1, 4))
        S[:, 0] = s0
        for k in range(self.N):
            x, y, psi, v = S[:, k].T
            a, d = U[:, k, 0], U[:, k, 1]
            S[:, k + 1, 0] = x + v * np.cos(psi) * self.dt
            S[:, k + 1, 1] = y + v * np.sin(psi) * self.dt
            S[:, k + 1, 2] = psi + v / self.L * np.tan(d) * self.dt
            S[:, k + 1, 3] = np.maximum(0.0, v + a * self.dt)
        return S

    def _cost(self, s0, U, path: Path, obstacles: Sequence[Obstacle], v_ref: np.ndarray, a_last: float, d_last: float,
              lateral_limits: Optional[Tuple[float, float]] = None) -> np.ndarray:
        S = self._rollout(s0, U)
        w = self.w
        _, e_y, psi_ref, _, _ = path.project(S[:, 1:, 0], S[:, 1:, 1])
        e_psi = wrap(S[:, 1:, 2] - psi_ref)
        cost = (w["y"] * e_y ** 2 + w["psi"] * e_psi ** 2 + w["v"] * (S[:, 1:, 3] - v_ref[None, :]) ** 2).sum(1)
        if lateral_limits is not None:   # soft drivable-corridor limits on e_y (lo, hi): forces braking when swerving is impossible
            cost += w["lim"] * (np.maximum(0.0, e_y - lateral_limits[1]) ** 2 + np.maximum(0.0, lateral_limits[0] - e_y) ** 2).sum(1)
        cost += (w["a"] * U[:, :, 0] ** 2 + w["d"] * U[:, :, 1] ** 2).sum(1)
        da = np.diff(np.concatenate([np.full((U.shape[0], 1), a_last), U[:, :, 0]], 1), axis=1)
        dd = np.diff(np.concatenate([np.full((U.shape[0], 1), d_last), U[:, :, 1]], 1), axis=1)
        cost += (w["da"] * da ** 2 + w["dd"] * dd ** 2).sum(1)
        for ob in obstacles:
            dist = np.hypot(S[:, 1:, 0] - ob.traj[None, 1:, 0], S[:, 1:, 1] - ob.traj[None, 1:, 1])
            safe = ob.radius + self.ego_r + self.margin
            cost += w["obs"] * (np.maximum(0.0, safe - dist) ** 2).sum(1)
        return cost

    def solve(self, state: np.ndarray, path: Path, obstacles: Sequence[Obstacle] = (), v_ref: float = 8.0,
              a_last: float = 0.0, d_last: float = 0.0, lateral_limits: Optional[Tuple[float, float]] = None) -> Tuple[float, float, np.ndarray, dict]:
        """Return (accel, steering, predicted states (N+1,4), info)."""
        import time
        t0 = time.perf_counter()
        N = self.N
        vr = np.full(N, v_ref)
        u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]]) if np.any(self.u_prev) else np.zeros((N, 2))
        u0[:, 0] = np.clip(u0[:, 0], *self.ab)
        u0[:, 1] = np.clip(u0[:, 1], *self.db)
        eps = 1e-4
        n = 2 * N

        def fun(z):
            U = z.reshape(1, N, 2)
            pert = np.repeat(U, n + 1, axis=0)
            for j in range(n):
                pert[j + 1].reshape(-1)[j] += eps
            c = self._cost(state, pert, path, obstacles, vr, a_last, d_last, lateral_limits)
            return float(c[0]), (c[1:] - c[0]) / eps

        bounds = [self.ab if j % 2 == 0 else self.db for j in range(n)]
        res = minimize(fun, u0.reshape(-1), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "ftol": 1e-6})
        U = res.x.reshape(N, 2)
        self.u_prev = U
        S = self._rollout(state, U[None])[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return float(U[0, 0]), float(U[0, 1]), S, {"cost": float(res.fun), "iters": int(res.nit), "ms": self.last_ms}
