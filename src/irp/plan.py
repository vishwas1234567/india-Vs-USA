"""Behavioural planning: a hierarchical state machine and a learned (Q-learning) alternative.

Both map a `WorldView` (ego state, predicted agents, lane availability) to a `Decision` (mode, target speed, target lane).
The state machine is deterministic and auditable; the Q-learning planner is a prototype trained in the closed-loop simulator
(irp.sim). Deep RL can replace the Q-table behind the same interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CRUISE, FOLLOW, YIELD, STOP, LANE_CHANGE, EMERGENCY = "CRUISE", "FOLLOW", "YIELD", "STOP", "LANE_CHANGE", "EMERGENCY"


@dataclass
class AgentView:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    kind: str = "car"
    p_enter: float = 0.0           # intent probability of entering the ego corridor


@dataclass
class WorldView:
    t: float
    ego_x: float
    ego_y: float
    ego_v: float
    lane_y: float                  # centre of the current target lane
    other_lane_y: Optional[float]  # centre of the adjacent lane (None if absent)
    agents: List[AgentView] = field(default_factory=list)
    corridor_half: float = 1.6     # half-width of the ego corridor (m)


@dataclass
class Decision:
    mode: str
    v_ref: float
    lane_y: float
    reason: str = ""


def ahead_in_corridor(w: WorldView, lane_y: float, horizon: float = 3.0, lookahead: float = 80.0) -> List[Tuple[AgentView, float, float]]:
    """Agents ahead whose position now or within `horizon` s overlaps the corridor of lane_y: (agent, gap, closing speed)."""
    out = []
    for a in w.agents:
        gap = a.x - w.ego_x - a.radius - 1.0
        if gap < -1.0 or gap > lookahead:
            continue
        hit = False
        for tt in np.linspace(0.0, horizon, 7):
            if abs((a.y + a.vy * tt) - lane_y) < w.corridor_half + a.radius * 0.5:
                hit = True
                break
        if hit:
            out.append((a, max(gap, 0.0), w.ego_v - a.vx))
    return sorted(out, key=lambda r: r[1])


def lane_free(w: WorldView, lane_y: float, back: float = 10.0, front: float = 30.0) -> bool:
    for a in w.agents:
        for tt in (0.0, 1.0, 2.0):
            if abs((a.y + a.vy * tt) - lane_y) < 2.2 and (w.ego_x - back) < (a.x + a.vx * tt) < (w.ego_x + front + w.ego_v * tt * 0.2):
                return False
    return True


class BehaviorFSM:
    """CRUISE / FOLLOW / YIELD / STOP / LANE_CHANGE / EMERGENCY with hysteresis and minimum dwell times."""

    def __init__(self, v_cruise: float = 10.0, allow_lane_change: bool = True, min_dwell: float = 1.0):
        self.v_c, self.allow_lc, self.dwell = v_cruise, allow_lane_change, min_dwell
        self.mode, self.t_enter, self.lane_y, self.changed = CRUISE, -1e9, None, False

    def reset(self):
        self.mode, self.t_enter, self.lane_y, self.changed = CRUISE, -1e9, None, False

    def decide(self, w: WorldView) -> Decision:
        if self.lane_y is None:
            self.lane_y = w.lane_y
        lead = ahead_in_corridor(w, self.lane_y)
        gap, closing, agent = (lead[0][1], lead[0][2], lead[0][0]) if lead else (math.inf, 0.0, None)
        ttc = gap / closing if closing > 0.1 else math.inf
        stop_dist = w.ego_v ** 2 / (2 * 5.0) + 4.0                      # braking distance at 5 m/s^2 + margin
        mode, v_ref, reason = CRUISE, self.v_c, "free road"
        lane_target, lane_change = self.lane_y, False
        if ttc < 1.8 or gap < 0.6 * stop_dist:
            mode, v_ref, reason = EMERGENCY, 0.0, f"ttc {ttc:.1f}s gap {gap:.1f}m"
        elif any(a.p_enter > 0.5 and a.kind in ("pedestrian", "animal") and 0 < a.x - w.ego_x < 40 + 2 * w.ego_v for a in w.agents):
            near = min(a.x - w.ego_x for a in w.agents if a.p_enter > 0.5 and a.kind in ("pedestrian", "animal") and a.x > w.ego_x)
            mode, v_ref, reason = YIELD, (0.0 if near < 14 else 3.0), "vulnerable user intends to enter the corridor: yield"
        elif agent is not None and (agent.vx < 1.0 and gap < 2.2 * stop_dist) or (agent is not None and agent.vx < 0.5 * self.v_c and gap < 35):
            if self.allow_lc and not self.changed and w.other_lane_y is not None and lane_free(w, w.other_lane_y):
                mode, reason = LANE_CHANGE, "slow/stopped object ahead, adjacent lane free"
                lane_target, lane_change = w.other_lane_y, True
                v_ref = self.v_c * 0.8
            elif agent.vx < 1.0:
                mode, v_ref, reason = STOP, 0.0, "stopped object ahead, no free lane"
            else:
                mode, v_ref, reason = FOLLOW, max(0.0, agent.vx + 0.3 * (gap - (6 + 1.2 * w.ego_v))), "slow lead"
        elif agent is not None and gap < 30 + 2 * w.ego_v and agent.p_enter < 0.5 and agent.kind not in ("pedestrian",):
            mode, v_ref, reason = FOLLOW, min(self.v_c, max(0.0, agent.vx + 0.3 * (gap - (6 + 1.2 * w.ego_v)))), "following"
        elif any(a.p_enter > 0.5 and 0 < a.x - w.ego_x < 40 + 2 * w.ego_v for a in w.agents):
            near = min(a.x - w.ego_x for a in w.agents if a.p_enter > 0.5 and a.x > w.ego_x)
            mode, v_ref, reason = YIELD, (0.0 if near < 14 else 3.0), "agent intends to enter the corridor"
        elif self.mode == LANE_CHANGE and abs(w.ego_y - self.lane_y) > 0.5:
            mode, v_ref, reason = LANE_CHANGE, self.v_c * 0.8, "completing lane change"
        if mode != self.mode and mode != EMERGENCY and self.mode != EMERGENCY and (w.t - self.t_enter) < self.dwell:
            mode, v_ref, reason = self.mode, min(v_ref, self.v_c), "dwell"      # hysteresis against chattering
        if mode == LANE_CHANGE and lane_change:      # commit the lane change only if the transition was not suppressed
            self.lane_y, self.changed = lane_target, True
        if mode != self.mode:
            self.mode, self.t_enter = mode, w.t
        return Decision(mode, float(v_ref), float(self.lane_y), reason)


ACTIONS = ("cruise", "slow", "stop", "lane_change")


class QLearningPlanner:
    """Tabular Q-learning over a discretised world: gap bin x closing speed x intent flag x adjacent lane free x lane changed."""

    def __init__(self, v_cruise: float = 10.0, epsilon: float = 0.15, alpha: float = 0.2, gamma: float = 0.95, seed: int = 0):
        self.v_c, self.eps, self.alpha, self.gamma = v_cruise, epsilon, alpha, gamma
        self.Q = np.zeros((5, 3, 2, 2, 2, len(ACTIONS)))
        self.rng = np.random.default_rng(seed)
        self.lane_y, self.changed, self.train = None, False, False
        self.last: Optional[Tuple[tuple, int]] = None

    def reset(self):
        self.lane_y, self.changed, self.last = None, False, None

    def state(self, w: WorldView) -> tuple:
        lead = ahead_in_corridor(w, self.lane_y if self.lane_y is not None else w.lane_y)
        gap, closing = (lead[0][1], lead[0][2]) if lead else (math.inf, 0.0)
        g = 0 if gap > 50 else 1 if gap > 30 else 2 if gap > 15 else 3 if gap > 7 else 4
        c = 0 if closing < 0.5 else 1 if closing < 4 else 2
        intent = int(any(a.p_enter > 0.5 and 0 < a.x - w.ego_x < 40 for a in w.agents))
        free = int(w.other_lane_y is not None and lane_free(w, w.other_lane_y))
        return (g, c, intent, free, int(self.changed))

    def decide(self, w: WorldView) -> Decision:
        if self.lane_y is None:
            self.lane_y = w.lane_y
        s = self.state(w)
        if self.train and self.rng.random() < self.eps:
            a = int(self.rng.integers(len(ACTIONS)))
        else:
            q = self.Q[s]
            a = int(self.rng.choice(np.flatnonzero(q == q.max())))
        self.last = (s, a)
        if ACTIONS[a] == "lane_change" and w.other_lane_y is not None and not self.changed:
            self.lane_y, self.changed = w.other_lane_y, True
        v = {"cruise": self.v_c, "slow": 0.4 * self.v_c, "stop": 0.0, "lane_change": 0.8 * self.v_c}[ACTIONS[a]]
        return Decision(ACTIONS[a].upper(), v, float(self.lane_y), "learned policy")

    def learn(self, reward: float, next_state: Optional[tuple]) -> None:
        if self.last is None:
            return
        s, a = self.last
        target = reward + (0.0 if next_state is None else self.gamma * self.Q[next_state].max())
        self.Q[s + (a,)] += self.alpha * (target - self.Q[s + (a,)])
