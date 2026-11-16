"""Closed-loop 2-D simulator for the autonomy stack (perception -> prediction/intent -> behaviour -> control -> safety).

Scenarios are scripted, Indian-flavoured interactions on a two-lane road: a pedestrian crossing mid-block, an auto-rickshaw cutting
in, a weaving two-wheeler, a stopped vehicle, a braking lead car. Agent parameters are randomised by seed.
Everything here is SIMULATION: it tests decision and control logic and fault handling, not real-road performance.

Faults injected into the primary pipeline: dropout (silently empty world), freeze (stale world), nan, crash, stall (latency).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .control import BicycleModel, LQRLateral, MPC, Obstacle, Path, SpeedController, wrap
from .intent import IntentHMM, IntentTracker, simulate_agents
from .plan import AgentView, BehaviorFSM, Decision, QLearningPlanner, WorldView
from .safety import PrimaryOutput, RangeReading, SafetySupervisor

LANES = (0.0, 3.5)
RADIUS = {"pedestrian": 0.4, "two_wheeler": 0.6, "auto": 1.0, "car": 1.1}
DT = 0.05
SCENARIOS = ("ped_cross", "auto_cut_in", "two_wheeler_weave", "stopped_vehicle", "lead_brake")
FAULTS = ("none", "dropout", "freeze", "nan", "crash", "stall")


@dataclass
class Agent:
    id: int
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    script: Optional[Callable[["Agent", float, float], None]] = None

    @property
    def radius(self) -> float:
        return RADIUS[self.kind]

    def step(self, t: float, dt: float) -> None:
        if self.script:
            self.script(self, t, dt)
        self.x += self.vx * dt
        self.y += self.vy * dt


def make_scenario(name: str, seed: int) -> List[Agent]:
    rng = np.random.default_rng(seed)
    U = lambda a, b: float(rng.uniform(a, b))
    if name == "ped_cross":
        t0, sp = U(0.5, 2.0), U(1.2, 1.7)

        def script(a, t, dt, t0=t0, sp=sp):
            a.vy = sp if (t >= t0 and a.y < 6.0) else 0.0
        return [Agent(1, "pedestrian", U(40, 52), -4.0, 0.0, 0.0, script)]
    if name == "auto_cut_in":
        t0, vy = U(1.0, 2.5), U(0.9, 1.4)

        def script(a, t, dt, t0=t0, vy=vy):
            a.vy = -vy if (t >= t0 and a.y > 0.0) else 0.0
            if a.y <= 0.0:
                a.y = 0.0
        return [Agent(1, "auto", U(18, 26), 3.5, 5.5, 0.0, script)]
    if name == "two_wheeler_weave":
        ph, amp, per = U(0, 6.28), U(0.7, 1.1), U(2.5, 3.5)

        def script(a, t, dt, ph=ph, amp=amp, per=per):
            a.vy = amp * (2 * math.pi / per) * math.cos(2 * math.pi * t / per + ph)
        return [Agent(1, "two_wheeler", U(32, 45), 0.0, 5.0, 0.0, script)]
    if name == "stopped_vehicle":
        return [Agent(1, "car", U(48, 62), 0.0, 0.0, 0.0), Agent(2, "car", U(25, 40), 3.5, U(9, 11), 0.0)]
    if name == "lead_brake":
        tb, dec = U(2.5, 4.0), U(3.0, 5.0)

        def script(a, t, dt, tb=tb, dec=dec):
            if t >= tb:
                a.vx = max(0.0, a.vx - dec * dt)
        return [Agent(1, "car", U(24, 32), 0.0, 9.0, 0.0, script)]
    raise ValueError(name)


class Perception:
    """Noisy, delayed detections of the true agents plus injectable faults."""

    def __init__(self, seed: int, fault: str = "none", t_fault: float = 3.0, duration: float = 3.0, noise: float = 0.15,
                 miss: float = 0.05, latency: float = 0.1):
        self.rng, self.fault, self.t0, self.dur, self.noise, self.miss = np.random.default_rng(seed + 7), fault, t_fault, duration, noise, miss
        self.delay = int(round(latency / DT))
        self.hist: List[List[dict]] = []
        self.frozen: Optional[List[dict]] = None

    def in_fault(self, t: float) -> bool:
        return self.fault != "none" and self.t0 <= t < self.t0 + self.dur

    def observe(self, t: float, agents: Sequence[Agent]) -> List[dict]:
        self.hist.append([dict(id=a.id, kind=a.kind, x=a.x, y=a.y) for a in agents])
        past = self.hist[max(0, len(self.hist) - 1 - self.delay)]
        obs = [dict(o, x=o["x"] + self.rng.normal(0, self.noise), y=o["y"] + self.rng.normal(0, self.noise))
               for o in past if self.rng.random() > self.miss]
        if self.in_fault(t):
            if self.fault == "dropout":
                return []
            if self.fault == "freeze":
                if self.frozen is None:
                    self.frozen = obs
                return self.frozen
            if self.fault == "nan":
                return [dict(o, x=float("nan"), y=float("nan")) for o in obs]
        return obs

    def latency_s(self, t: float) -> float:
        return 0.5 if (self.fault == "stall" and self.in_fault(t)) else 0.02

    def crashed(self, t: float) -> bool:
        return self.fault == "crash" and self.in_fault(t)


def radar(ego: np.ndarray, agents: Sequence[Agent], rng: np.random.Generator, half_width: float = 1.7) -> RangeReading:
    """Independent range sensor: nearest object in the ego corridor ahead (noise 0.1 m), closing speed."""
    best, closing = math.inf, 0.0
    for a in agents:
        gap = a.x - ego[0] - a.radius - 1.0
        if gap > -0.5 and abs(a.y - ego[1]) < half_width + a.radius * 0.5 and gap < best:
            best, closing = gap, ego[3] - a.vx
    return RangeReading(best + (rng.normal(0, 0.1) if math.isfinite(best) else 0.0), closing)


class _TrackBook:
    """Finite-difference velocity with exponential smoothing per agent id."""

    def __init__(self, alpha: float = 0.5):
        self.state: Dict[int, Tuple[float, float, float, float, float]] = {}
        self.alpha = alpha

    def update(self, t: float, obs: Sequence[dict]):
        out = []
        for o in obs:
            if not (math.isfinite(o["x"]) and math.isfinite(o["y"])):
                continue
            prev = self.state.get(o["id"])
            vx = vy = 0.0
            if prev is not None and t > prev[4]:
                dt = t - prev[4]
                vx = (1 - self.alpha) * prev[2] + self.alpha * (o["x"] - prev[0]) / dt
                vy = (1 - self.alpha) * prev[3] + self.alpha * (o["y"] - prev[1]) / dt
            self.state[o["id"]] = (o["x"], o["y"], vx, vy, t)
            out.append((o, vx, vy))
        return out


_HMM_CACHE: Dict[str, IntentHMM] = {}


def default_hmm() -> IntentHMM:
    if "h" not in _HMM_CACHE:
        o, s = simulate_agents(300, seed=0)
        _HMM_CACHE["h"] = IntentHMM().fit_supervised(o, s)
    return _HMM_CACHE["h"]


class PrimaryStack:
    """The AI pipeline: tracking -> prediction (+ intent) -> behaviour planner -> controller."""

    def __init__(self, planner="fsm", controller="mpc", use_intent: bool = True, rl: Optional[QLearningPlanner] = None,
                 mpc_horizon: int = 10):
        self.planner_kind, self.controller_kind, self.use_intent = planner, controller, use_intent
        self.fsm = BehaviorFSM() if planner == "fsm" else None
        self.rl = rl if planner == "rl" else None
        self.mpc = MPC(N=mpc_horizon, max_iter=15) if controller == "mpc" else None
        self.lqr, self.spd, self.book = LQRLateral(dt=DT), SpeedController(), _TrackBook()
        self.intent = IntentTracker(default_hmm())
        self.path, self.lane_y = Path.straight(400, 0.0), 0.0
        self.last_decision: Optional[Decision] = None
        self.a_last = self.d_last = 0.0
        self.solve_ms: List[float] = []

    def _world(self, t: float, ego: np.ndarray, obs: Sequence[dict]) -> WorldView:
        agents = []
        for o, vx, vy in self.book.update(t, obs):
            e = (o["y"] - self.lane_y) / 1.75
            _, belief = self.intent.update(o["id"], e, vy / 1.75, math.hypot(vx, vy) / 1.75) if self.use_intent else (None, np.zeros(3))
            p_enter = self.intent.p_enter(o["id"]) if self.use_intent else 0.0
            if abs(o["y"] - self.lane_y) < 1.0:   # already inside the corridor: intent is moot
                p_enter = 0.0
            agents.append(AgentView(o["id"], o["x"], o["y"], vx, vy, RADIUS.get(o["kind"], 1.0), o["kind"], p_enter))
        other = LANES[1] if self.lane_y < 1.75 else LANES[0]
        return WorldView(t, ego[0], ego[1], ego[3], self.lane_y, other, agents)

    def _path_for(self, ego: np.ndarray, lane_y: float) -> Path:
        if abs(lane_y - self.lane_y) < 1e-6 and self.path is not None:
            return self.path
        x0, y0 = ego[0], ego[1]
        xs = np.linspace(x0, x0 + 400, 400)
        blend = np.clip((xs - x0) / 30.0, 0, 1)
        ys = y0 + (lane_y - y0) * 0.5 * (1 - np.cos(np.pi * blend))
        return Path(np.c_[xs, ys])

    def plan(self, t: float, ego: np.ndarray, obs: Sequence[dict]) -> Tuple[float, float, Decision]:
        w = self._world(t, ego, obs)
        decision = (self.fsm or self.rl).decide(w)
        if decision.lane_y != self.lane_y:
            self.path, self.lane_y = self._path_for(ego, decision.lane_y), decision.lane_y
        self.last_decision = decision
        if self.mpc is not None:
            obstacles = []
            for a in w.agents:
                vy = a.vy
                if self.use_intent and a.p_enter > 0.5 and abs(a.y - self.lane_y) >= 1.0:
                    vy = -math.copysign(1.3, a.y - self.lane_y)       # intent prior: crossing toward the corridor
                ts = np.arange(self.mpc.N + 1) * self.mpc.dt
                obstacles.append(Obstacle(np.c_[a.x + a.vx * ts, a.y + vy * ts], a.radius, a.kind))
            a_cmd, d_cmd, _, info = self.mpc.solve(ego, self.path, obstacles, decision.v_ref, self.a_last, self.d_last, (-0.8, 0.8))
            self.solve_ms.append(info["ms"])
        else:
            ey, epsi, kappa, _ = self.path.errors(ego)
            a_cmd, d_cmd = self.spd.control(ego[3], decision.v_ref), self.lqr.control(ey, epsi, kappa, ego[3])
        self.a_last, self.d_last = a_cmd, d_cmd
        return a_cmd, d_cmd, decision

    def lqr_steer(self, ego: np.ndarray) -> float:
        ey, epsi, kappa, _ = self.path.errors(ego)
        return self.lqr.control(ey, epsi, kappa, ego[3])


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    collision: bool
    min_gap: float
    min_ttc: float
    mean_jerk: float
    progress: float
    max_lateral: float
    solve_ms_mean: float
    degraded_frac: float
    fault: str = "none"


def run_episode(scenario: str, seed: int, planner: str = "fsm", controller: str = "mpc", supervisor: bool = False,
                fault: str = "none", use_intent: bool = True, rl: Optional[QLearningPlanner] = None, T: float = 14.0,
                train: bool = False, mpc_horizon: int = 10) -> EpisodeResult:
    agents = make_scenario(scenario, seed)
    car, ego = BicycleModel(), np.array([0.0, 0.0, 0.0, 8.0])
    stack = PrimaryStack(planner, controller, use_intent, rl, mpc_horizon)
    if rl is not None:
        rl.reset()
    sup = SafetySupervisor() if supervisor else None
    per = Perception(seed, fault, t_fault=3.0 + (seed % 5) * 0.2)
    rng = np.random.default_rng(seed + 99)
    a_cmd = d_cmd = 0.0
    hold_lane = False
    a_hist: List[float] = []
    min_gap, min_ttc, max_lat, collided, degraded, steps = math.inf, math.inf, 0.0, False, 0, 0
    for k in range(int(T / DT)):
        t = k * DT
        for a in agents:
            a.step(t, DT)
        if k % 4 == 0:   # 5 Hz decision loop
            obs = per.observe(t, agents)
            primary = None
            if per.crashed(t):
                primary = PrimaryOutput(0.0, 0.0, ok=False)
            else:
                try:
                    p_a, p_d, _ = stack.plan(t, ego, obs)
                    primary = PrimaryOutput(p_a, p_d, per.latency_s(t))
                except (ValueError, FloatingPointError):
                    primary = PrimaryOutput(0.0, 0.0, ok=False)
            rr = radar(ego, agents, rng)
            hold_lane = False
            if sup is not None:
                acc, steer, why = sup.arbitrate(t, ego[3], primary, rr)
                a_cmd, hold_lane = acc, steer is None
                if steer is not None:
                    d_cmd = steer
                degraded += why != "primary"
            elif primary.ok and primary.latency_s < 0.3 and math.isfinite(primary.accel):
                a_cmd, d_cmd = primary.accel, primary.steer       # unsupervised: failures keep the last command
        steer = stack.lqr_steer(ego) if (controller == "lqr" or hold_lane) else d_cmd
        ego = car.step(ego, a_cmd, steer, DT)
        a_hist.append(float(np.clip(a_cmd, -car.max_brake, car.max_accel)))
        steps += 1
        max_lat = max(max_lat, abs(ego[1] - stack.lane_y))
        for a in agents:
            gap = math.hypot(a.x - ego[0], a.y - ego[1]) - a.radius - 1.0
            min_gap = min(min_gap, gap)
            if gap < 0:
                collided = True
            ahead = a.x - ego[0] - a.radius - 1.0
            if ahead > 0 and abs(a.y - ego[1]) < 1.7 + a.radius * 0.5 and ego[3] - a.vx > 0.1:
                min_ttc = min(min_ttc, ahead / (ego[3] - a.vx))
        if collided:
            break
    jerk = float(np.mean(np.abs(np.diff(a_hist)) / DT)) if len(a_hist) > 1 else 0.0
    return EpisodeResult(scenario, seed, collided, float(min_gap), float(min_ttc) if math.isfinite(min_ttc) else 99.0, jerk,
                         float(ego[0]), float(max_lat), float(np.mean(stack.solve_ms)) if stack.solve_ms else 0.0,
                         degraded / max(steps / 4, 1), fault)


def train_rl(episodes: int = 1500, seed: int = 0) -> QLearningPlanner:
    """Train the Q-learning planner in the simulator (fast LQR low-level control). Reward: progress, comfort, safety."""
    rl = QLearningPlanner(seed=seed)
    rl.train = True
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        scenario = SCENARIOS[int(rng.integers(len(SCENARIOS)))]
        sd = 1000 + int(rng.integers(100000))
        agents = make_scenario(scenario, sd)
        car, ego = BicycleModel(), np.array([0.0, 0.0, 0.0, 8.0])
        stack = PrimaryStack("rl", "lqr", True, rl)
        rl.reset()
        per = Perception(sd)
        a_cmd = 0.0
        prev = None
        collided = False
        for k in range(int(14.0 / DT)):
            t = k * DT
            for a in agents:
                a.step(t, DT)
            if k % 4 == 0:
                obs = per.observe(t, agents)
                w = stack._world(t, ego, obs)
                if prev is not None:
                    gaps = [math.hypot(a.x - ego[0], a.y - ego[1]) - a.radius - 1.0 for a in agents]
                    r = 0.05 * ego[3] - 0.02 * abs(a_cmd) - (1.0 if min(gaps) < 2.5 else 0.0)
                    rl.learn(r, rl.state(w))
                a_cmd, d_cmd, _ = stack.plan(t, ego, obs)
            ego = car.step(ego, a_cmd, stack.lqr_steer(ego), DT)
            if any(math.hypot(a.x - ego[0], a.y - ego[1]) - a.radius - 1.0 < 0 for a in agents):
                rl.learn(-50.0, None)
                collided = True
                break
            prev = True
        rl.eps = max(0.03, rl.eps * 0.998)
    rl.train, rl.eps = False, 0.0
    return rl


def summarize(results: Sequence[EpisodeResult]) -> Dict[str, float]:
    n = len(results)
    ttc = [r.min_ttc for r in results if r.min_ttc < 99]
    return {"episodes": n, "collision_rate": sum(r.collision for r in results) / max(n, 1),
            "min_gap_median_m": float(np.median([r.min_gap for r in results])) if n else float("nan"),
            "min_ttc_median_s": float(np.median(ttc)) if ttc else float("nan"),
            "mean_jerk": float(np.mean([r.mean_jerk for r in results])) if n else float("nan"),
            "progress_mean_m": float(np.mean([r.progress for r in results])) if n else float("nan"),
            "solve_ms_mean": float(np.mean([r.solve_ms_mean for r in results])) if n else float("nan"),
            "degraded_frac": float(np.mean([r.degraded_frac for r in results])) if n else float("nan")}


def benchmark(config: Dict[str, dict], seeds: Sequence[int] = tuple(range(10)), scenarios: Sequence[str] = SCENARIOS,
              rl: Optional[QLearningPlanner] = None) -> Dict[str, dict]:
    """Run every named configuration over scenarios x seeds; returns per-config summaries and per-scenario collision rates."""
    out: Dict[str, dict] = {}
    for name, kw in config.items():
        res = [run_episode(sc, sd, rl=rl if kw.get("planner") == "rl" else None, **kw) for sc in scenarios for sd in seeds]
        s = summarize(res)
        s["per_scenario_collision"] = {sc: float(np.mean([r.collision for r in res if r.scenario == sc])) for sc in scenarios}
        out[name] = s
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Closed-loop benchmark suites (simulation only).")
    ap.add_argument("--suite", choices=["rl", "stacks", "faults"], required=True)
    ap.add_argument("--out", default="results/_sim")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--configs", default="", help="comma list of stack names to run (stacks suite)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=1500, help="RL training episodes")
    a = ap.parse_args(argv)
    out = FsPath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = a.scenarios.split(",")
    seeds = list(range(a.seeds))
    if a.suite == "rl":
        t0 = time.perf_counter()
        rl = train_rl(a.episodes)
        np.save(out / "qtable.npy", rl.Q)
        (out / "rl_training.json").write_text(json.dumps({"episodes": a.episodes, "seconds": round(time.perf_counter() - t0, 1),
                                                          "visited_states": int((rl.Q != 0).any(-1).sum())}, indent=2))
        print("trained", a.episodes, "episodes in", round(time.perf_counter() - t0, 1), "s")
        return
    rl = None
    if (out / "qtable.npy").exists():
        rl = QLearningPlanner()
        rl.Q = np.load(out / "qtable.npy")
        rl.train, rl.eps = False, 0.0
    if a.suite == "stacks":
        all_cfg = {"fsm+lqr (no intent)": dict(planner="fsm", controller="lqr", use_intent=False),
                   "fsm+lqr+intent": dict(planner="fsm", controller="lqr", use_intent=True),
                   "fsm+mpc+intent": dict(planner="fsm", controller="mpc", use_intent=True),
                   "rl+lqr+intent": dict(planner="rl", controller="lqr", use_intent=True)}
        cfg = {k: v for k, v in all_cfg.items() if not a.configs or k in a.configs.split(",")}
        res = benchmark(cfg, seeds, scenarios, rl=rl)
        path = out / "stacks.json"
        merged = json.loads(path.read_text()) if path.exists() else {}
        for name, r in res.items():                         # chunked runs: merge scenario subsets by episode-weighted mean
            if name in merged and set(merged[name].get("scenarios", [])).isdisjoint(scenarios):
                old = merged[name]
                n0, n1 = old["episodes"], r["episodes"]
                comb = {k: (old[k] * n0 + r[k] * n1) / (n0 + n1) for k in r if isinstance(r[k], float)}
                comb["episodes"] = n0 + n1
                comb["per_scenario_collision"] = {**old["per_scenario_collision"], **r["per_scenario_collision"]}
                comb["scenarios"] = old["scenarios"] + scenarios
                merged[name] = comb
            else:
                r["scenarios"] = list(scenarios)
                merged[name] = r
        path.write_text(json.dumps(merged, indent=2))
        print(json.dumps(merged, indent=2))
    else:
        rows = {}
        for fault in FAULTS:
            for sup in (False, True):
                res = [run_episode(sc, sd, "fsm", "lqr", supervisor=sup, fault=fault) for sc in scenarios for sd in seeds]
                s = summarize(res)
                rows[f"{fault}|{'supervisor' if sup else 'primary only'}"] = s
        (out / "faults.json").write_text(json.dumps(rows, indent=2))
        for k, v in rows.items():
            print(f"{k:28s} collision {v['collision_rate']:.2f}  min gap median {v['min_gap_median_m']:.2f}  degraded {v['degraded_frac']:.2f}")


if __name__ == "__main__":
    main()
