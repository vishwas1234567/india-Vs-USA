"""Static-camera junction analysis (classical CV; OpenCV + NumPy, SciPy optional).

Pipeline: [stabilise] -> [enhance] -> median background -> road mask -> MOG2 motion
segmentation with shadow suppression -> blob filtering -> tracking (greedy or Kalman +
Hungarian) -> scene metrics, flow gates, headways, traffic bursts, traffic-police and
zebra-crossing cues. All pixel parameters scale with the processing width so the same
settings work from 320 px to 960 px wide input.

Usage:
    python -m irp.static --source data/junction_clip.mp4 --out results/main --label "Baseline"
    python -m irp.static --source data/junction2_360p.mp4 --out results/b_main --tracker kalman --enhance full
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import __version__
from .common import frame_range, iter_frames, open_video, put_text, resolve_source, write_csv
from .flow import default_gates, detect_bursts, headway_stats, per_bin_counts, track_crossings
from .quality import ENHANCE_MODES, Stabilizer, enhance_frame
from .tracking import KalmanTracker

log = logging.getLogger("irp.static")


@dataclass
class StaticConfig:
    source: str
    out: str = "results/static"
    label: str = ""
    horizon: float = 0.43            # far road edge as a fraction of image height
    min_track_frames: int = 12       # minimum track length (frames) to confirm a track
    min_displacement_px: float = 40.0  # at 960 px width; scaled with resolution
    warmup_frames: int = 30          # background-model warm-up (excluded from metrics)
    mog_history: int = 250
    mog_var_threshold: float = 36.0
    shadow_suppression: bool = True
    start_s: float = 0.0
    duration_s: float = 0.0          # 0 = until the end
    max_dist: float = 80.0           # tracker association radius (px at 960 px width)
    max_miss: int = 10               # tracker miss tolerance (frames)
    background_samples: int = 60
    proc_width: int = 0              # 0 = min(source width, 960)
    proc_height: int = 0             # 0 = keep aspect ratio
    tracker: str = "greedy"          # greedy | kalman
    enhance: str = "none"            # none | clahe | full
    stabilize: bool = False
    vest_persistence_s: float = 2.0  # a vest counts as police only if it stays in place this long
    write_video: bool = True


def odd(n: float, scale: float) -> int:
    """Odd kernel size from a size defined at 960 px width."""
    return max(1, int(round(n * scale))) | 1


# --------------------------------------------------------------------------- #
# Scene cues
# --------------------------------------------------------------------------- #
def build_background(cap: cv2.VideoCapture, start: int, end: int, size: Tuple[int, int], n: int = 60) -> np.ndarray:
    """Temporal median over n frames sampled uniformly in [start, end)."""
    frames = []
    for i in np.unique(np.linspace(start, end - 1, n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, size))
    if not frames:
        raise RuntimeError("Could not read frames to build the background.")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def road_from_background(bg: np.ndarray, top_frac: float = 0.43, scale: float = 1.0) -> np.ndarray:
    """Road = low-saturation, non-dark pixels below the far road edge (largest component)."""
    h = bg.shape[0]
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 35)).astype(np.uint8)
    m[: int(h * top_frac)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((odd(15, scale),) * 2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((odd(9, scale),) * 2, np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (comp == k).astype(np.uint8)


def zebra_stripes(bg: np.ndarray, road: np.ndarray, scale: float = 1.0) -> List[Tuple[int, int, int, int]]:
    """Elongated bright stripes on the road (zebra-crossing candidates) via white top-hat."""
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    k = odd(41, scale)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    road_d = cv2.dilate(road, np.ones((odd(9, scale),) * 2, np.uint8))
    s = ((th > 35) & (road_d > 0)).astype(np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 150 * scale * scale and w > 2.0 * h:
            boxes.append((x, y, w, h))
    return boxes


def detect_vests(frame_bgr: np.ndarray, scale: float = 1.0) -> List[Tuple[int, int, int, int]]:
    """Orange hi-vis vests (traffic police) as (x, y, w, h) boxes."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 150, 120), (20, 255, 255))
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((odd(9, scale),) * 2, np.uint8))
    cnts, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 400 * scale * scale and 0.7 < h / max(w, 1) < 3.0:
            out.append((x, y, w, h))
    return out


# --------------------------------------------------------------------------- #
# Baseline tracker
# --------------------------------------------------------------------------- #
class CentroidTracker:
    """Greedy nearest-centroid tracker for foreground blobs."""

    def __init__(self, max_dist: float = 80.0, max_miss: int = 10):
        self.max_dist, self.max_miss = max_dist, max_miss
        self.tracks = {}
        self.finished: List[dict] = []
        self._next_id = 1

    def update(self, boxes: Sequence[Tuple[int, int, int, int]], frame_idx: int) -> None:
        used = set()
        for t in self.tracks.values():
            t["matched"] = False
            best, best_d = None, self.max_dist
            for j, b in enumerate(boxes):
                if j in used:
                    continue
                d = math.hypot((b[0] + b[2]) / 2 - t["c"][0], (b[1] + b[3]) / 2 - t["c"][1])
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                b = boxes[best]
                t.update(box=b, c=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), miss=0, matched=True)
                t["hist"].append((frame_idx, t["c"], b))
        for j, b in enumerate(boxes):
            if j not in used:
                c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                self.tracks[self._next_id] = {"id": self._next_id, "box": b, "c": c, "miss": 0,
                                              "hist": [(frame_idx, c, b)], "matched": True}
                self._next_id += 1
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if not t["matched"]:
                t["miss"] += 1
            if t["miss"] > self.max_miss:
                self.finished.append(self.tracks.pop(tid))

    def all_tracks(self) -> List[dict]:
        return self.finished + list(self.tracks.values())


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def _percentiles(values: Sequence[float], qs: Sequence[int]) -> Optional[List[float]]:
    return [round(float(np.percentile(values, q)), 3) for q in qs] if len(values) else None


def analyze(cfg: StaticConfig) -> dict:
    """Run the static-camera pipeline; write outputs to cfg.out and return the summary dict."""
    if cfg.enhance not in ENHANCE_MODES:
        raise ValueError(f"enhance must be one of {ENHANCE_MODES}")
    if cfg.tracker not in ("greedy", "kalman"):
        raise ValueError("tracker must be 'greedy' or 'kalman'")
    t_start = time.perf_counter()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    path = resolve_source(cfg.source, out)
    cap, info = open_video(path)
    start, end = frame_range(info, cfg.start_s, cfg.duration_s)
    PW = cfg.proc_width or min(info.width, 960)
    PH = cfg.proc_height or int(round(PW * info.height / info.width))
    s = PW / 960.0
    fps = info.fps
    n_frames = end - start
    if n_frames <= cfg.warmup_frames:
        raise ValueError(f"Segment has {n_frames} frames; need more than warm-up ({cfg.warmup_frames}).")

    bg = build_background(cap, start, end, (PW, PH), cfg.background_samples)
    road = road_from_background(bg, cfg.horizon, s)
    road_d = cv2.dilate(road, np.ones((odd(25, s),) * 2, np.uint8))
    road_px = max(int(road.sum()), 1)
    zebra = zebra_stripes(bg, road, s)
    horizon_y = int(PH * cfg.horizon)
    k_open, k_close = odd(3, s), odd(11, s)
    min_h, min_w = max(4, int(round(18 * s))), max(4, int(round(14 * s)))
    min_disp = cfg.min_displacement_px * s

    mog = cv2.createBackgroundSubtractorMOG2(
        history=cfg.mog_history, varThreshold=cfg.mog_var_threshold, detectShadows=cfg.shadow_suppression)
    tracker_cls = KalmanTracker if cfg.tracker == "kalman" else CentroidTracker
    tracker = tracker_cls(cfg.max_dist * s, cfg.max_miss)
    stabilizer = Stabilizer() if cfg.stabilize else None
    gates = default_gates()
    writer = None
    if cfg.write_video:
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (PW, PH))

    def is_police(t: dict) -> bool:
        return t.get("police_hits", 0) >= 5

    # A vest is accepted as police only if orange blobs keep appearing around the same place
    # (people on duty stand still; orange cars, autos and scooters move on).
    cell_px = max(8, int(round(32 * s)))
    vest_thr = max(1, int(round(cfg.vest_persistence_s * fps)))
    vest_hits: Counter = Counter()

    def cell_of(x: float, y: float) -> Tuple[int, int]:
        return int(x // cell_px), int(y // cell_px)

    def persistent(cell: Tuple[int, int]) -> bool:
        return sum(vest_hits.get((cell[0] + i, cell[1] + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1)) >= vest_thr

    row_cells: List[List[Tuple[int, int]]] = []
    rows: List[dict] = []
    for k, (idx, frame) in enumerate(iter_frames(cap, start, end), start=1):
        if stabilizer is not None:
            frame = stabilizer(frame)
        f = cv2.resize(frame, (PW, PH))
        fe = enhance_frame(f, cfg.enhance)
        fg = (mog.apply(fe) == 255).astype(np.uint8)  # label 127 (shadow) is dropped
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
        occ = float((fg & road_d).sum()) / road_px
        warm = k <= cfg.warmup_frames

        raw_vests = detect_vests(f, s)
        vcells = [cell_of(x + w / 2, y + h / 2) for (x, y, w, h) in raw_vests]
        vest_hits.update(vcells)
        row_cells.append(vcells)
        vests = [v for v, c in zip(raw_vests, vcells) if persistent(c)]
        n_live = 0
        if not warm:
            cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in cs:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < 0.0008 * PW * PH or h < min_h or w < min_w:
                    continue
                if road_d[min(y + h - 1, PH - 1), min(x + w // 2, PW - 1)] == 0:
                    continue  # feet/wheels must touch the road
                boxes.append((x, y, x + w, y + h))
            tracker.update(boxes, idx)
            for t in tracker.tracks.values():
                bx1, by1, bx2, by2 = t["box"]
                if t["miss"] == 0:
                    for (x, y, w, h), c in zip(raw_vests, vcells):
                        if bx1 <= x + w / 2 <= bx2 and by1 <= y + h / 2 <= by2:
                            t.setdefault("vest_cells", []).append(c)
                            if persistent(c):
                                t["police_hits"] = t.get("police_hits", 0) + 1
            n_live = sum(1 for t in tracker.tracks.values()
                         if len(t["hist"]) >= cfg.min_track_frames and t["miss"] == 0 and not is_police(t))

        if writer is not None:
            vis = f.copy()
            tint = np.zeros_like(vis)
            tint[road > 0] = (0, 140, 0)
            vis = cv2.addWeighted(vis, 1, tint, 0.25, 0)
            for (x, y, w, h) in zebra:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
            for g in gates:
                cv2.line(vis, (int(g.p1[0] * PW), int(g.p1[1] * PH)), (int(g.p2[0] * PW), int(g.p2[1] * PH)),
                         (255, 255, 0), 1)
            for (x, y, w, h) in vests:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(vis, "police vest", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(s, 0.7),
                            (0, 140, 255), 1, cv2.LINE_AA)
            for tid, t in tracker.tracks.items():
                if len(t["hist"]) < cfg.min_track_frames or t["miss"] > 0 or is_police(t):
                    continue
                x1, y1, x2, y2 = t["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis, f"#{tid}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(s, 0.7),
                            (0, 255, 255), 1, cv2.LINE_AA)
                pts = np.array([hh[1] for hh in t["hist"][-30:]], np.int32)
                cv2.polylines(vis, [pts], False, (0, 255, 255), 1)
            hud = f"t={idx / fps:5.1f}s users:{n_live} occ:{occ * 100:.1f}%"
            put_text(vis, hud, (10, 22), 0.6 * max(s, 0.8))
            writer.write(vis)

        rows.append({"frame": idx, "t_s": round(idx / fps, 2), "warmup": int(warm),
                     "road_occupancy": round(occ, 4), "moving_users": n_live, "police_vests": len(vests)})
    cap.release()
    if writer is not None:
        writer.release()

    # ---- final police decision (persistence judged with the whole clip's evidence)
    for t in tracker.all_tracks():
        t["police_hits"] = sum(1 for c in t.get("vest_cells", []) if persistent(c))
    for r, cells in zip(rows, row_cells):
        r["police_vests"] = sum(1 for c in cells if persistent(c))

    # ---- per-track statistics (confirmed, non-police, non-stationary)
    stats: List[dict] = []
    kept: List[dict] = []
    n_police = 0
    for t in tracker.all_tracks():
        h = t["hist"]
        if len(h) < cfg.min_track_frames:
            continue
        if is_police(t):
            n_police += 1
            continue
        (f0, c0, _), (f1, c1, _) = h[0], h[-1]
        dx, dy = c1[0] - c0[0], c1[1] - c0[1]
        disp = math.hypot(dx, dy)
        if disp < min_disp:
            continue
        if abs(dx) > 1.2 * abs(dy):
            direction = "left->right" if dx > 0 else "right->left"
        else:
            direction = "toward camera" if dy > 0 else "away from camera"
        sizes = [math.sqrt((b[2] - b[0]) * (b[3] - b[1])) / max(b[3] - horizon_y, 1) for _, _, b in h]
        widths = [b[2] - b[0] for _, _, b in h]
        dur = (f1 - f0) / fps
        stats.append({
            "id": t["id"], "t_start": round(f0 / fps, 1), "t_end": round(f1 / fps, 1),
            "direction": direction, "disp_px": round(disp / s),  # normalised to 960 px width
            "rel_size": round(float(np.median(sizes)), 3),
            "body_widths_per_s": round(disp / max(float(np.median(widths)), 1) / max(dur, 0.1), 2)})
        kept.append(t)

    # ---- flow analytics
    events: List[dict] = []
    for t in kept:
        base = [(f, (c[0], b[3])) for f, c, b in t["hist"]]
        for g in gates:
            for frame_idx, label in track_crossings(base, g, (PW, PH)):
                events.append({"gate": g.name, "direction": label, "t_s": round(frame_idx / fps, 2), "track_id": t["id"]})
    flow_counts = {g.name: dict(Counter(e["direction"] for e in events if e["gate"] == g.name)) for g in gates}
    flow_headway = {g.name: headway_stats([e["t_s"] for e in events if e["gate"] == g.name]) for g in gates}
    seg_dur = len(rows) / fps
    flow_bins = {g.name: per_bin_counts([e["t_s"] - start / fps for e in events if e["gate"] == g.name], seg_dur)
                 for g in gates}

    post = [r for r in rows if not r["warmup"]]
    occ_arr = np.array([r["road_occupancy"] for r in post])
    bursts = detect_bursts([r["t_s"] for r in post], occ_arr, fps)
    peak = post[int(np.argmax(occ_arr))]
    wall = time.perf_counter() - t_start
    sizes_all = [x["rel_size"] for x in stats]
    summary = {
        "frames_total": len(rows), "frames_analysed": len(post),
        "duration_s": round(len(rows) / fps, 1), "fps": fps,
        "source_size": [info.width, info.height], "proc_size": [PW, PH],
        "tracker": cfg.tracker, "enhance": cfg.enhance, "stabilize": cfg.stabilize,
        "road_area_frac": round(road_px / (PW * PH), 3),
        "zebra_stripe_candidates": len(zebra),
        "moving_tracks_confirmed": len(stats),
        "police_tracks_excluded": n_police,
        "direction_counts": dict(Counter(x["direction"] for x in stats)),
        "occupancy_mean": round(float(occ_arr.mean()), 4),
        "occupancy_p95": round(float(np.percentile(occ_arr, 95)), 4),
        "occupancy_peak": {"t_s": peak["t_s"], "value": peak["road_occupancy"]},
        "mean_tracked_users": round(float(np.mean([r["moving_users"] for r in post])), 2),
        "police_vest_frame_frac": round(sum(1 for r in rows if r["police_vests"]) / len(rows), 3),
        "median_rel_size": round(float(np.median(sizes_all)), 3) if sizes_all else None,
        "rel_size_p10_p90": _percentiles(sizes_all, (10, 90)),
        "median_speed_body_widths_per_s":
            round(float(np.median([x["body_widths_per_s"] for x in stats])), 2) if stats else None,
        "track_duration_median_s":
            round(float(np.median([x["t_end"] - x["t_start"] for x in stats])), 2) if stats else None,
        "crossings": flow_counts, "headway": flow_headway,
        "bursts": {k: v for k, v in bursts.items() if k != "bursts"},
        "wall_time_s": round(wall, 1), "processing_fps": round(len(rows) / wall, 1),
    }
    if stabilizer is not None:
        summary["stabilizer"] = stabilizer.summary()
    traj = [{"track_id": t["id"], "frame": f, "cx": round(c[0] / s, 2), "cy": round(c[1] / s, 2),
             "w": round((b[2] - b[0]) / s, 2), "h": round((b[3] - b[1]) / s, 2)}
            for t in kept for f, c, b in t["hist"]]   # image coordinates normalised to 960 px width
    write_csv(out / "trajectories.csv", traj)
    write_csv(out / "frames.csv", rows)
    write_csv(out / "tracks.csv", stats)
    write_csv(out / "crossings.csv", events)
    (out / "flow.json").write_text(json.dumps({"gates": [g.__dict__ for g in gates], "counts": flow_counts,
                                               "headway": flow_headway, "bin_counts_10s": flow_bins,
                                               "bursts": bursts}, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    params = asdict(cfg)
    params.update(version=__version__, opencv=cv2.__version__, numpy=np.__version__,
                  start_frame=start, end_frame=end)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    cv2.imwrite(str(out / "background.jpg"), bg)
    cv2.imwrite(str(out / "road_mask.png"), road * 255)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> StaticConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or URL (URLs need yt-dlp)")
    p.add_argument("--out", default="results/static")
    p.add_argument("--label", default="", help="short description stored with the run")
    p.add_argument("--horizon", type=float, default=0.43)
    p.add_argument("--min-track-frames", type=int, default=12)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--start", type=float, default=0.0, dest="start_s", help="start time (s)")
    p.add_argument("--duration", type=float, default=0.0, dest="duration_s", help="duration (s); 0 = to the end")
    p.add_argument("--tracker", choices=["greedy", "kalman"], default="greedy")
    p.add_argument("--enhance", choices=list(ENHANCE_MODES), default="none")
    p.add_argument("--stabilize", action="store_true", help="remove camera translation using the static upper region")
    p.add_argument("--vest-persistence", type=float, default=2.0, dest="vest_persistence_s",
                   help="seconds a vest must stay in place to count as police (0 = any orange blob)")
    p.add_argument("--no-shadow-suppression", action="store_true", help="ablation: shadows count as foreground")
    p.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
    a = p.parse_args(argv)
    return StaticConfig(
        source=a.source, out=a.out, label=a.label, horizon=a.horizon, min_track_frames=a.min_track_frames,
        warmup_frames=a.warmup_frames, start_s=a.start_s, duration_s=a.duration_s, tracker=a.tracker,
        enhance=a.enhance, stabilize=a.stabilize, vest_persistence_s=a.vest_persistence_s,
        shadow_suppression=not a.no_shadow_suppression,
        write_video=not a.no_video)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = analyze(parse_args(argv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
