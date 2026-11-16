"""Tests for the irp package. Run with:  PYTHONPATH=src python -m unittest discover -s tests -v

All tests use synthetic data and stub models; no deep-learning libraries are needed.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from irp.common import (ClassicalRoad, Detection, MotionAnalyzer, category, corridor, est_dist, frame_range,
                        lane_marking_state, open_video, pothole_candidates, VideoInfo)
from irp.flow import Gate, detect_bursts, headway_stats, per_bin_counts, track_crossings
from irp.quality import Stabilizer, blockiness, enhance_frame, noise_sigma, assess_quality
from irp.robustness import agreement, degrade
from irp.static import CentroidTracker, StaticConfig, analyze
from irp.tracking import KalmanTracker
from irp.vehicle import VehicleConfig, assess_risk, run
import shutil


def road_scene(h=480, w=640, paint=True, pothole=True):
    img = np.full((h, w, 3), (60, 200, 60), np.uint8)
    pts = np.array([[w * 0.1, h], [w * 0.9, h], [w * 0.55, h * 0.45], [w * 0.45, h * 0.45]], np.int32)
    cv2.fillPoly(img, [pts], (105, 105, 110))
    if paint:
        for y in range(260, h, 60):
            cv2.line(img, (w // 2, y), (w // 2, y + 30), (240, 240, 240), 6)
    if pothole:
        cv2.ellipse(img, (300, 400), (35, 18), 0, 0, 360, (40, 40, 45), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def write_video(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


class CommonTests(unittest.TestCase):
    def test_category_mapping(self):
        self.assertEqual(category("motorcycle"), "two_wheeler")
        self.assertEqual(category("Person"), "pedestrian")
        self.assertEqual(category("auto_rickshaw"), "auto")
        self.assertEqual(category("cow"), "animal")
        self.assertEqual(category("traffic light"), "sign")
        self.assertIsNone(category("banana"))

    def test_distance_decreases_toward_camera(self):
        near = est_dist(340, 162, 457, 1.4)
        far = est_dist(220, 162, 457, 1.4)
        self.assertLess(near, far)
        self.assertTrue(math.isinf(est_dist(160, 162, 457, 1.4)))

    def test_frame_range(self):
        info = VideoInfo(fps=30.0, width=640, height=360, frames=300)
        self.assertEqual(frame_range(info), (0, 300))
        self.assertEqual(frame_range(info, 1.0, 2.0), (30, 90))
        with self.assertRaises(ValueError):
            frame_range(info, 20.0, 1.0)

    def test_lane_score_higher_with_paint(self):
        seg = ClassicalRoad()
        with_paint, no_paint = road_scene(paint=True), road_scene(paint=False)
        s1 = lane_marking_state(with_paint, seg(with_paint))
        s0 = lane_marking_state(no_paint, seg(no_paint))
        self.assertGreater(s1[1], s0[1])
        self.assertEqual(s0[0], "absent")

    def test_pothole_candidate_found(self):
        img = road_scene()
        found = pothole_candidates(img, ClassicalRoad()(img))
        self.assertTrue(any(abs(x + w / 2 - 300) < 30 and abs(y + h / 2 - 400) < 30 for x, y, w, h in found))

    def test_motion_analyzer_flags_weaving_only(self):
        ma = MotionAnalyzer(ego_compensation=True)
        for t in range(14):
            dets = [Detection((300 + (25 if t % 2 else -25) - 20, 100, 300 + (25 if t % 2 else -25) + 20, 160),
                              "motorcycle", track_id=1)]
            for tid, x0 in ((2, 100), (3, 500), (4, 200)):
                dets.append(Detection((x0 + 2 * t, 100, x0 + 2 * t + 60, 160), "car", track_id=tid))
            ma.update(dets)
        self.assertTrue(dets[0].erratic)
        self.assertFalse(any(d.erratic for d in dets[1:]))

    def test_risk_levels(self):
        w, h, hy = 640, 360, 162
        poly, wide = corridor(w, h, hy), corridor(w, h, hy, 1.5)
        near_ped = Detection((290, 240, 350, 340), "person", dist=3.6)
        far_ped = Detection((300, 190, 320, 230), "person", dist=15.0)
        side_car = Detection((20, 250, 120, 330), "car", dist=5.0)
        self.assertEqual(assess_risk([near_ped], poly, wide)[0], "HIGH")
        self.assertEqual(assess_risk([far_ped], poly, wide)[0], "MED")
        self.assertEqual(assess_risk([side_car], poly, wide)[0], "NONE")


class PipelineTests(unittest.TestCase):
    def test_static_pipeline_tracks_a_crossing_object(self):
        frames = []
        for t in range(90):
            f = np.full((180, 320, 3), (60, 200, 60), np.uint8)
            f[77:] = (105, 105, 110)
            x = int(3.3 * t)
            cv2.rectangle(f, (x, 110), (x + 40, 138), (200, 60, 60), -1)
            frames.append(f)
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "clip.mp4"
            write_video(clip, frames)
            cfg = StaticConfig(source=str(clip), out=str(Path(td) / "run"), proc_width=320, proc_height=180,
                               write_video=False, label="synthetic")
            summary = analyze(cfg)
            for name in ("frames.csv", "tracks.csv", "summary.json", "params.json", "road_mask.png"):
                self.assertTrue((Path(td) / "run" / name).exists(), name)
        self.assertGreaterEqual(summary["moving_tracks_confirmed"], 1)
        self.assertGreaterEqual(summary["direction_counts"].get("left->right", 0), 1)
        self.assertEqual(summary["frames_total"], 90)

    def test_vehicle_pipeline_with_stub_models(self):
        w, h = 640, 360
        base = np.full((h, w, 3), (60, 200, 60), np.uint8)
        cv2.fillPoly(base, [np.array([[40, h], [600, h], [380, 170], [260, 170]], np.int32)], (105, 105, 110))
        frames = [base.copy() for _ in range(40)]
        counter = {"n": 0}

        def stub_detector(_frame):
            n = counter["n"]
            counter["n"] += 1
            wob = 30 if n % 2 else -30
            dets = [Detection((290, 240, 350, 340), "person", track_id=1),
                    Detection((150 + wob - 20, 250, 150 + wob + 20, 300), "motorcycle", track_id=2)]
            for tid, x0 in ((3, 60), (4, 420), (5, 500)):
                dets.append(Detection((x0 + 2 * n, 200, x0 + 2 * n + 60, 260), "car", track_id=tid))
            return dets

        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "drive.mp4"
            write_video(clip, frames)
            cfg = VehicleConfig(source=str(clip), out=str(Path(td) / "run"), stride=1, seg_backend="classical")
            summary = run(cfg, stub_detector, ClassicalRoad())
            for name in ("frames.csv", "summary.json", "report.md", "annotated.mp4"):
                self.assertTrue((Path(td) / "run" / name).exists(), name)
            json.loads((Path(td) / "run" / "summary.json").read_text())
        self.assertGreater(summary["risk_frac"]["HIGH"], 0.5)
        self.assertGreaterEqual(summary["erratic_tracks"].get("two_wheeler", 0), 1)
        self.assertEqual(summary["unique_tracks"]["pedestrian"], 1)

    def test_open_video_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            open_video("/nonexistent/video.mp4")


def crossing_video(path, n=90, w=320, h=180, color=(200, 60, 60), size=(40, 28), y0=110, speed=3.3, still=False):
    frames = []
    for t in range(n):
        f = np.full((h, w, 3), (60, 200, 60), np.uint8)
        f[77:] = (105, 105, 110)
        x = 150 if still else int(speed * t)
        cv2.rectangle(f, (x, y0), (x + size[0], y0 + size[1]), color, -1)
        frames.append(f)
    write_video(path, frames)


class AdvancedTests(unittest.TestCase):
    def test_kalman_bridges_gap_that_greedy_cannot(self):
        def run_tracker(tracker):
            for t in range(21):
                if t in (10, 11):
                    tracker.update([], t)
                    continue
                x = 30 * t
                tracker.update([(x, 100, x + 40, 140)], t)
            return len(tracker.all_tracks())
        self.assertEqual(run_tracker(KalmanTracker(max_dist=40, max_miss=10)), 1)
        self.assertEqual(run_tracker(CentroidTracker(max_dist=40, max_miss=10)), 2)

    def test_gate_crossing_direction(self):
        gate = Gate("v", (0.5, 0.4), (0.5, 1.0), ("L->R", "R->L"))
        left_to_right = [(i, (100 + 20 * i, 120)) for i in range(15)]
        right_to_left = [(i, (400 - 20 * i, 120)) for i in range(15)]
        self.assertEqual([d for _, d in track_crossings(left_to_right, gate, (320, 180))], ["L->R"])
        self.assertEqual([d for _, d in track_crossings(right_to_left, gate, (320, 180))], ["R->L"])
        self.assertEqual(track_crossings([(0, (10, 120)), (1, (20, 120))], gate, (320, 180)), [])

    def test_headway_bins_and_bursts(self):
        self.assertAlmostEqual(headway_stats([0, 1, 3, 6])["mean_s"], 2.0)
        self.assertEqual(sum(per_bin_counts([1, 2, 11, 25], 30.0)), 4)
        fps = 10.0
        t = np.arange(0, 60, 1 / fps)
        occ = np.zeros_like(t)
        for start in (5, 25, 45):
            occ[(t >= start) & (t < start + 6)] = 0.2
        res = detect_bursts(t, occ, fps)
        self.assertEqual(res["n"], 3)
        self.assertAlmostEqual(res["mean_interval_s"], 20.0, delta=1.0)

    def test_quality_measures_and_enhance(self):
        rng = np.random.default_rng(0)
        base = cv2.GaussianBlur((rng.random((180, 320)) * 255).astype(np.uint8), (0, 0), 3)
        noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
        self.assertGreater(noise_sigma(noisy), noise_sigma(base) + 3)
        self.assertAlmostEqual(blockiness(base), 1.0, delta=0.1)
        frame = road_scene()
        for mode in ("none", "clahe", "full"):
            out = enhance_frame(frame, mode)
            self.assertEqual(out.shape, frame.shape)
        with self.assertRaises(ValueError):
            enhance_frame(frame, "bogus")

    def test_stabilizer_reduces_shift(self):
        rng = np.random.default_rng(1)
        tex = cv2.GaussianBlur((rng.random((200, 320, 3)) * 255).astype(np.uint8), (0, 0), 2)
        stab = Stabilizer()
        ref, errs_raw, errs_fix = None, [], []
        for t in range(6):
            m = np.float32([[1, 0, 2 * t], [0, 1, t]])
            shifted = cv2.warpAffine(tex, m, (320, 200), borderMode=cv2.BORDER_REFLECT)
            fixed = stab(shifted)
            if ref is None:
                ref = fixed.copy()
                continue
            errs_raw.append(float(np.abs(shifted.astype(int) - ref).mean()))
            errs_fix.append(float(np.abs(fixed.astype(int) - ref).mean()))
        self.assertLess(np.mean(errs_fix), 0.6 * np.mean(errs_raw))

    def test_vest_persistence_separates_police_from_moving_orange(self):
        orange = (0, 128, 255)
        with tempfile.TemporaryDirectory() as td:
            still, moving = Path(td) / "still.mp4", Path(td) / "moving.mp4"
            crossing_video(still, color=orange, size=(14, 26), y0=100, still=True)
            crossing_video(moving, color=orange, size=(14, 26), y0=100)
            cfg = dict(proc_width=320, proc_height=180, write_video=False)
            s_still = analyze(StaticConfig(source=str(still), out=str(Path(td) / "a"), **cfg))
            s_move = analyze(StaticConfig(source=str(moving), out=str(Path(td) / "b"), **cfg))
        self.assertGreater(s_still["police_vest_frame_frac"], 0.8)
        self.assertLess(s_move["police_vest_frame_frac"], 0.2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not available")
    def test_degradation_and_agreement(self):
        with tempfile.TemporaryDirectory() as td:
            clip, low = Path(td) / "clip.mp4", Path(td) / "low.mp4"
            crossing_video(clip)
            degrade(str(clip), str(low), {"scale": (160, 90), "crf": 35})
            cap, info = open_video(str(low))
            self.assertEqual((info.width, info.height, info.frames), (160, 90, 90))
            cap.release()
            cfg = dict(proc_width=320, proc_height=180, write_video=False)
            analyze(StaticConfig(source=str(clip), out=str(Path(td) / "ref"), **cfg))
            analyze(StaticConfig(source=str(clip), out=str(Path(td) / "same"), **cfg))
            res = agreement(Path(td) / "ref", Path(td) / "same")
        self.assertAlmostEqual(res["occ_pearson"], 1.0, places=3)
        self.assertEqual(res["tracks_ratio"], 1.0)
        self.assertEqual(res["direction_l1"], 0.0)

    def test_quality_assessment_on_clip(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "clip.mp4"
            crossing_video(clip)
            rep = assess_quality(str(clip), n_samples=8)
        self.assertEqual((rep["width"], rep["height"]), (320, 180))
        self.assertIn("low_resolution", rep["flags"])


if __name__ == "__main__":
    unittest.main()
