"""Intent estimation with a discrete Hidden Markov Model.

Hidden states (per road user, relative to the ego corridor):
    0 passive      not interacting with the ego path
    1 approaching  moving toward the corridor from outside (about to cross / cut in)
    2 entering     inside the corridor or crossing it

Observation symbol = (position class, motion class, speed class), computed from the track's lateral offset e normalised by
the corridor half-width, its rate de/dt and its speed. The HMM is fitted by counting (supervised) or Baum-Welch (EM) and used online
as a forward filter. Validated on simulated agents with known intent (tests, results/_sim); not validated on labelled real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STATES = ("passive", "approaching", "entering")
N_OBS = 3 * 3 * 2


def encode(e: float, de_dt: float, speed: float, v_thr: float = 0.15, fast: float = 1.0) -> int:
    """Discretise (normalised lateral offset, its rate per second, speed in body widths or m/s) to one of 18 symbols."""
    a = abs(e)
    pos = 0 if a < 0.5 else 1 if a < 1.2 else 2
    toward = -np.sign(e) * de_dt                                     # >0: moving toward the corridor centre
    motion = 0 if toward > v_thr else 1 if toward < -v_thr else 2    # toward / away / parallel
    return (pos * 3 + motion) * 2 + int(speed > fast)


class IntentHMM:
    def __init__(self, n_states: int = 3, n_obs: int = N_OBS):
        self.S, self.O = n_states, n_obs
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.B = np.full((n_states, n_obs), 1.0 / n_obs)

    def fit_supervised(self, obs_seqs: Sequence[Sequence[int]], state_seqs: Sequence[Sequence[int]], smoothing: float = 1.0):
        pi, A, B = (np.full(self.S, smoothing), np.full((self.S, self.S), smoothing), np.full((self.S, self.O), smoothing))
        for o, s in zip(obs_seqs, state_seqs):
            pi[s[0]] += 1
            for t in range(len(s)):
                B[s[t], o[t]] += 1
                if t:
                    A[s[t - 1], s[t]] += 1
        self.pi, self.A, self.B = pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)
        return self

    def fit_em(self, obs_seqs: Sequence[Sequence[int]], iters: int = 20):
        """Baum-Welch refinement (unsupervised); state labels are only identifiable up to permutation."""
        for _ in range(iters):
            pi_n, A_n, B_n = np.zeros(self.S), np.zeros((self.S, self.S)), np.zeros((self.S, self.O))
            for o in obs_seqs:
                T = len(o)
                alpha, c = np.zeros((T, self.S)), np.zeros(T)
                alpha[0] = self.pi * self.B[:, o[0]]
                c[0] = alpha[0].sum(); alpha[0] /= c[0]
                for t in range(1, T):
                    alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, o[t]]
                    c[t] = alpha[t].sum(); alpha[t] /= c[t]
                beta = np.ones((T, self.S))
                for t in range(T - 2, -1, -1):
                    beta[t] = (self.A @ (self.B[:, o[t + 1]] * beta[t + 1])) / c[t + 1]
                gamma = alpha * beta
                gamma /= gamma.sum(1, keepdims=True)
                pi_n += gamma[0]
                for t in range(T - 1):
                    xi = alpha[t][:, None] * self.A * (self.B[:, o[t + 1]] * beta[t + 1])[None, :]
                    A_n += xi / xi.sum()
                for t in range(T):
                    B_n[:, o[t]] += gamma[t]
            self.pi = (pi_n + 1e-6) / (pi_n + 1e-6).sum()
            self.A = (A_n + 1e-6) / (A_n + 1e-6).sum(1, keepdims=True)
            self.B = (B_n + 1e-6) / (B_n + 1e-6).sum(1, keepdims=True)
        return self

    def step(self, belief: Optional[np.ndarray], obs: int) -> np.ndarray:
        """One forward-filter update; belief=None starts a new sequence."""
        b = self.pi * self.B[:, obs] if belief is None else (belief @ self.A) * self.B[:, obs]
        return b / max(b.sum(), 1e-300)

    def filter(self, obs: Sequence[int]) -> np.ndarray:
        out, b = [], None
        for o in obs:
            b = self.step(b, o)
            out.append(b)
        return np.array(out)

    def viterbi(self, obs: Sequence[int]) -> List[int]:
        T = len(obs)
        logA, logB = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        d = np.log(self.pi + 1e-300) + logB[:, obs[0]]
        back = np.zeros((T, self.S), int)
        for t in range(1, T):
            sc = d[:, None] + logA
            back[t], d = sc.argmax(0), sc.max(0) + logB[:, obs[t]]
        path = [int(d.argmax())]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]


class IntentTracker:
    """Online per-track intent beliefs; call update(track_id, e, de_dt, speed) every frame."""

    def __init__(self, hmm: IntentHMM):
        self.hmm, self.belief = hmm, {}

    def update(self, track_id: int, e: float, de_dt: float, speed: float) -> Tuple[str, np.ndarray]:
        b = self.hmm.step(self.belief.get(track_id), encode(e, de_dt, speed))
        self.belief[track_id] = b
        return STATES[int(b.argmax())], b

    def p_enter(self, track_id: int) -> float:
        b = self.belief.get(track_id)
        return 0.0 if b is None else float(b[1] * 0.5 + b[2])    # entering, plus half of approaching


# --------------------------------------------------------------------------- #
# Simulation of agents with known intent (for fitting and validation)
# --------------------------------------------------------------------------- #
def simulate_agents(n: int, T: int = 60, dt: float = 0.1, seed: int = 0, noise: float = 0.05):
    """Return observation and state sequences. Intent follows a Markov chain; motion is generated physically."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.96, 0.04, 0.0], [0.05, 0.88, 0.07], [0.04, 0.0, 0.96]])
    obs_seqs, state_seqs = [], []
    for _ in range(n):
        side = rng.choice([-1.0, 1.0])
        e = side * rng.uniform(1.4, 2.5)
        s = int(rng.choice([0, 0, 1]))
        o, st = [], []
        speed_lat = 0.0
        for t in range(T):
            if t:
                s = int(rng.choice(3, p=P[s]))
            target = {0: rng.normal(0.0, 0.1), 1: -side * rng.uniform(0.5, 0.9), 2: -side * rng.uniform(1.0, 1.6)}[s]
            speed_lat += 0.4 * (target - speed_lat)
            e_new = e + speed_lat * dt + rng.normal(0, noise * dt)
            if s == 2:
                e_new = e + (-side) * abs(speed_lat) * dt                # keeps crossing
            if s == 0:
                e_new = e + 0.2 * (side * rng.uniform(1.4, 2.5) - e) * dt  # drifts around outside the corridor
            de = (e_new - e) / dt + rng.normal(0, noise)
            o.append(encode(e_new, de, abs(de) + rng.uniform(0, 0.6)))
            st.append(s)
            e = e_new
        obs_seqs.append(o)
        state_seqs.append(st)
    return obs_seqs, state_seqs


def rule_baseline(obs: Sequence[int]) -> List[int]:
    """Memoryless classifier from the instantaneous symbol only."""
    out = []
    for o in obs:
        pos, motion = (o // 2) // 3, (o // 2) % 3
        out.append(2 if pos == 0 else 1 if motion == 0 and pos >= 1 else 0)
    return out
