"""Tests for the v3 autonomy stack. Synthetic scenes and simulation only: they verify logic, geometry and fault handling,
not accuracy on real roads."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from irp.camera import CameraModel, EgoMotion
from irp.common import Detection
from irp.compare import compare
from irp.control import BicycleModel, LQRLateral, MPC, Obstacle, Path as RefPath, SpeedController
from irp.detect import MotionDetector, decode_yolov8, letterbox_params
from irp.intent import IntentHMM, rule_baseline, simulate_agents
from irp.lanes import LaneDetector
from irp.pipeline import PipelineConfig, run as run_pipeline
from irp.plan import AgentView, BehaviorFSM, WorldView
from irp.predict import LSTMForecaster, RidgeForecaster, baseline_predict, build_windows
from irp.road import PanopticFuser, RoadSegmenter, free_space_depth, scene_descriptor
from irp.runtime import AdaptiveScale, Profiler, prefetch, select_device
from irp.safety import PrimaryOutput, RangeReading, SafetySupervisor
from irp.sim import make_scenario, run_episode, summarize
from irp.sources import open_source
from irp.synth import SceneObject, moving_camera_sequence, render_road
from iscvd import ISCVDataset
from iscvd.formats import Sample
from iscvd.metrics import PanopticQuality, lane_f1

CAM = CameraModel(640, 360)


def write_video(path, frames, fps=15):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def road_mask(cam, ego=0.0, half=4.5):
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    m = np.zeros((cam.height, cam.width), np.uint8)
    below = yy > cam.horizon_row + 2
    g = cam.image_to_ground(xx[below], yy[below])
    m[below] = np.abs(g[:, 0] + ego) < half
    return m


class InputAndCamera(unittest.TestCase):
    def test_sources_video_folder_glob_image_and_errors(self):
        frames = moving_camera_sequence(CameraModel(320, 180), n=12)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_video(td / "v.mp4", frames)
            src = open_source(str(td / "v.mp4"), stride=2)
            self.assertEqual(src.kind, "video")
            self.assertEqual(len(list(src)), 6)
            self.assertEqual(max(open_source(str(td / "v.mp4"), max_side=160).info.width, 0), 160)
            (td / "imgs").mkdir()
            for i, f in enumerate(frames[:5]):
                cv2.imwrite(str(td / "imgs" / f"f{i:02d}.jpg"), f)
            (td / "imgs" / "broken.jpg").write_bytes(b"not an image")
            self.assertEqual(len(list(open_source(str(td / "imgs")))), 5)            # corrupt file skipped
            self.assertEqual(len(list(open_source(str(td / "imgs" / "f0*.jpg")))), 5)
            self.assertEqual(open_source(str(td / "imgs" / "f00.jpg")).info.n_frames, 1)
            with self.assertRaises(FileNotFoundError):
                open_source(str(td / "nothing_here.mp4"))

    def test_camera_roundtrip_and_bev(self):
        X, Z = np.array([-1.5, 2.0]), np.array([8.0, 25.0])
        back = CAM.image_to_ground(*CAM.ground_to_image(X, Z).T)
        np.testing.assert_allclose(back, np.c_[X, Z], atol=1e-6)
        H, size, _ = CAM.bev_homography()
        self.assertEqual(H.shape, (3, 3))
        self.assertGreater(size[0], 0)

    def test_ego_motion_static_vs_moving(self):
        cam = CameraModel(320, 180)
        ego = EgoMotion()
        base = render_road(cam, noise=0.0)[0]
        for _ in range(15):
            ego.update(base)
        self.assertFalse(ego.moving_camera())
        ego2 = EgoMotion()
        for f in moving_camera_sequence(CameraModel(640, 360), n=30, speed_mps=8):
            ego2.update(f)
        self.assertTrue(ego2.moving_camera())


class LanesRoadDetect(unittest.TestCase):
    def _f1(self, **kw):
        img, gt = render_road(CAM, **kw)
        res = LaneDetector(CAM, track=False).detect(img)
        return lane_f1(res.polylines_image(CAM), gt, 20)["F1"], res

    def test_lane_detection_straight_curved_faded(self):
        for kw in ({}, {"curvature": 0.002}, {"paint": 0.25}, {"glare": 0.5}, {"noise": 8}):
            f1, res = self._f1(**kw)
            self.assertGreaterEqual(f1, 0.9, kw)
            self.assertEqual(res.state, "clear")

    def test_ego_offset_and_freespace_fallback(self):
        _, res = self._f1(ego_offset=0.5)
        self.assertAlmostEqual(res.ego_offset_m, 0.5, delta=0.2)
        img, _ = render_road(CAM, paint=0.0, ego_offset=0.5, road_half=4.5)
        res = LaneDetector(CAM, track=False).detect(img, road_mask(CAM, 0.5, 4.5))
        self.assertEqual(res.source, "freespace")
        self.assertAlmostEqual(res.ego_offset_m, 0.5, delta=0.25)
        self.assertEqual(LaneDetector(CAM, track=False).detect(img).source, "none")

    def test_road_segmenter_freespace_scene_and_panoptic(self):
        cam = CameraModel(320, 180)
        objs = [SceneObject(-1.0, 12.0)]
        img, _ = render_road(cam, objects=objs, road_half=4.5)
        road = RoadSegmenter()(img)
        gt = road_mask(cam)
        self.assertGreater((road & gt).sum() / (road | gt).sum(), 0.85)
        self.assertLess(free_space_depth(road, cam).min(), 15.0)
        self.assertEqual(scene_descriptor(img, road)["lighting"], "day")
        b = cam.ground_to_image(np.array([-1.0]), np.array([12.0]))[0]
        pw, ph = cam.f * 1.8 / 12, cam.f * 1.5 / 12
        det = Detection((int(b[0] - pw / 2), int(b[1] - ph), int(b[0] + pw / 2), int(b[1])), "car", 0.9)
        pan = PanopticFuser().fuse(road, [det])
        self.assertIn(10, set(np.unique(pan) // 1000))
        pq = PanopticQuality()
        pq.update(pan, pan)
        self.assertAlmostEqual(pq.result()["PQ"], 1.0)

    def test_yolo_decode_and_motion_detector(self):
        out = np.zeros((1, 84, 100), np.float32)
        out[0, :4, 3] = [320, 320, 100, 50]
        out[0, 4 + 2, 3] = 0.9
        r, dx, dy = letterbox_params((360, 640), 640)
        dets = decode_yolov8(out, ["a", "b", "car"] + ["x"] * 77, 0.25, 0.45, r, dx, dy, (360, 640))
        self.assertEqual(len(dets), 1)
        self.assertEqual((dets[0].name, dets[0].box), ("car", (270, 155, 370, 205)))
        cam = CameraModel(320, 180)
        frames = moving_camera_sequence(cam, n=30, speed_mps=8, mover=SceneObject(-3.0, 14.0, 1.6, 1.4, (30, 60, 200)))
        ego, det, hits, n = EgoMotion(), MotionDetector(), 0, 0
        for i, f in enumerate(frames):
            d = det(f, ego.update(f))
            if i > 8:
                n += 1
                exp = cam.ground_to_image(np.array([-3.0 + 1.5 * i / 15.0]), np.array([14.0]))[0]
                hits += any(abs((b.box[0] + b.box[2]) / 2 - exp[0]) < 35 and abs(b.box[3] - exp[1]) < 35 for b in d)
        self.assertGreaterEqual(hits / n, 0.8)


class PredictionIntentControl(unittest.TestCase):
    def test_lstm_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        m = LSTMForecaster(4, 6, 3, seed=1)
        X, Y = rng.normal(size=(5, 4, 4)), rng.normal(size=(5, 3, 2))
        _, g = m.loss_and_grads(X, Y)
        worst = 0.0
        for k in m.p:
            flat = m.p[k].reshape(-1)
            for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                old, eps = flat[i], 1e-5
                flat[i] = old + eps
                lp = m.loss_and_grads(X, Y)[0]
                flat[i] = old - eps
                lm = m.loss_and_grads(X, Y)[0]
                flat[i] = old
                num, ana = (lp - lm) / (2 * eps), g[k].reshape(-1)[i]
                worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        self.assertLess(worst, 1e-5)

    def test_lstm_learns_constant_velocity_and_windows(self):
        rng = np.random.default_rng(1)
        v = rng.normal(0, 0.3, (300, 1, 2))
        X = np.concatenate([np.repeat(v, 4, 1), np.ones((300, 4, 1)), np.zeros((300, 4, 1))], 2)
        Y = v * np.arange(1, 4)[None, :, None]
        m = LSTMForecaster(4, 16, 3, seed=0)
        hist = m.fit(X[:240], Y[:240], X[240:], Y[240:], epochs=60)
        self.assertLess(np.mean((m.predict(X[240:]) - Y[240:]) ** 2), 0.5 * np.mean(Y[240:] ** 2))
        self.assertEqual(baseline_predict(X, 3, "cv_last").shape, (300, 3, 2))
        self.assertEqual(RidgeForecaster().fit(X, Y).predict(X).shape, (300, 3, 2))
        traj = {1: np.c_[np.arange(0, 90, 3), np.arange(30) * 5.0, np.full(30, 100.0), np.full(30, 40.0), np.full(30, 30.0)]}
        Xw, Yw, W, T = build_windows(traj, obs=4, pred=5, step=3)
        self.assertEqual(Xw.shape[1:], (4, 4))
        np.testing.assert_allclose(Yw[0, 0], [5.0 / 40.0, 0.0], atol=1e-6)

    def test_intent_hmm_beats_memoryless_rule(self):
        tr_o, tr_s = simulate_agents(200, seed=0)
        te_o, te_s = simulate_agents(60, seed=1)
        h = IntentHMM().fit_supervised(tr_o, tr_s)
        acc = lambda preds: np.mean(np.concatenate([np.array(p) == np.array(t) for p, t in zip(preds, te_s)]))
        self.assertGreater(acc([list(h.filter(o).argmax(1)) for o in te_o]), acc([rule_baseline(o) for o in te_o]) + 0.05)

    def test_lqr_and_mpc(self):
        car, lqr, spd = BicycleModel(), LQRLateral(dt=0.05), SpeedController()
        path, s = RefPath.straight(300), np.array([0.0, 1.0, 0.1, 8.0])
        for _ in range(80):
            ey, epsi, kappa, _ = path.errors(s)
            s = car.step(s, spd.control(s[3], 8.0), lqr.control(ey, epsi, kappa, s[3]), 0.05)
        self.assertLess(abs(path.errors(s)[0]), 0.05)
        mpc, s, ob = MPC(N=10, max_iter=15), np.array([0.0, 0.0, 0.0, 10.0]), np.array([35.0, 0.0])
        a_l = d_l = 0.0
        for _ in range(30):
            a, d, _, _ = mpc.solve(s, path, [Obstacle(np.tile(ob, (11, 1)), 1.0)], 10.0, a_l, d_l, (-1.0, 1.0))
            a_l, d_l = a, d
            s = car.step(s, a, d, 0.2)
        self.assertLess(s[0], 33.0)                       # stopped short of the obstacle inside the lane limits
        self.assertGreater(np.hypot(*(s[:2] - ob)), 2.5)


class PlanningSafetySim(unittest.TestCase):
    def _world(self, agents, v=10.0):
        return WorldView(0.0, 0.0, 0.0, v, 0.0, 3.5, agents)

    def test_fsm_modes(self):
        fsm = BehaviorFSM()
        self.assertEqual(fsm.decide(self._world([])).mode, "CRUISE")
        fsm = BehaviorFSM()
        self.assertEqual(fsm.decide(self._world([AgentView(1, 9.0, 0.0, 0.0, 0.0, 1.1, "car")])).mode, "EMERGENCY")
        fsm = BehaviorFSM()
        far = fsm.decide(self._world([AgentView(1, 45.0, 0.0, 0.0, 0.0, 1.1, "car")]))     # still beyond the lane-change reaction distance
        self.assertEqual(far.mode, "FOLLOW")
        self.assertLess(far.v_ref, 10.0)                                                  # but already easing off
        fsm = BehaviorFSM()
        self.assertEqual(fsm.decide(self._world([AgentView(1, 30.0, 0.0, 0.0, 0.0, 1.1, "car")])).mode, "LANE_CHANGE")
        fsm = BehaviorFSM()
        d = fsm.decide(self._world([AgentView(1, 25.0, -3.0, 0.0, 1.3, 0.4, "pedestrian", p_enter=0.9)]))
        self.assertEqual(d.mode, "YIELD")

    def test_supervisor_overrides_and_falls_back(self):
        sup = SafetySupervisor()
        close = RangeReading(12.0, 10.0)
        acc, steer, why = sup.arbitrate(0.0, 10.0, PrimaryOutput(0.5, 0.0), close)
        self.assertLess(acc, -3.0)
        self.assertIn("override", why)
        sup.reset()
        acc, steer, why = sup.arbitrate(0.2, 10.0, PrimaryOutput(0.0, 0.0, ok=False), RangeReading(float("inf"), 0.0))
        self.assertIsNone(steer)
        self.assertIn("DEGRADED", why)
        acc, steer, why = sup.arbitrate(0.4, 10.0, PrimaryOutput(float("nan"), 0.0), RangeReading(float("inf"), 0.0))
        self.assertIn("non-finite", why)
        acc, steer, why = sup.arbitrate(4.0, 3.0, PrimaryOutput(0.0, 0.0, ok=False), RangeReading(float("inf"), 0.0))
        self.assertIn("SAFE_STOP", why)

    def test_simulator_runs_and_supervisor_never_hurts_under_dropout(self):
        for sc in ("ped_cross", "auto_cut_in", "two_wheeler_weave", "stopped_vehicle", "lead_brake"):
            r = run_episode(sc, 0, "fsm", "lqr")
            self.assertTrue(np.isfinite(r.min_gap) and np.isfinite(r.progress))
        prim = [run_episode(sc, sd, "fsm", "lqr", fault="dropout") for sc in ("ped_cross", "stopped_vehicle") for sd in range(10)]
        sup = [run_episode(sc, sd, "fsm", "lqr", supervisor=True, fault="dropout") for sc in ("ped_cross", "stopped_vehicle") for sd in range(10)]
        self.assertLessEqual(summarize(sup)["collision_rate"], summarize(prim)["collision_rate"])
        self.assertEqual(len(make_scenario("stopped_vehicle", 3)), 2)

    def test_mpc_episode_has_bounded_solve_time(self):
        r = run_episode("ped_cross", 0, "fsm", "mpc")
        self.assertLess(r.solve_ms_mean, 250.0)
        self.assertFalse(r.collision)


class PipelineCompareRuntime(unittest.TestCase):
    def test_pipeline_on_synthetic_drive_and_images(self):
        frames = moving_camera_sequence(CAM, n=24, speed_mps=8, mover=SceneObject(-3.0, 14.0, 1.6, 1.4, (30, 60, 200)))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_video(td / "drive.mp4", frames)
            s = run_pipeline(PipelineConfig(str(td / "drive.mp4"), str(td / "o1"), detector="motion"))
            self.assertEqual(s["mode"], "vehicle")
            self.assertEqual(s["frames"], 24)
            for name in ("annotated.mp4", "frames.csv", "summary.json", "config.json"):
                self.assertTrue((td / "o1" / name).exists(), name)
            (td / "imgs").mkdir()
            for i, f in enumerate(frames[:6]):
                cv2.imwrite(str(td / "imgs" / f"f{i}.jpg"), f)
            s2 = run_pipeline(PipelineConfig(str(td / "imgs"), str(td / "o2"), detector="motion", assume_sequence=False))
            self.assertEqual((s2["mode"], s2["frames"]), ("vehicle", 6))
            self.assertIn("lane_state_frac", s2)

    def test_pipeline_handles_tiny_and_odd_sized_frames(self):
        from irp.pipeline import Pipeline
        for (w, h) in ((64, 48), (160, 90), (333, 211), (1920, 1080)):
            pipe = Pipeline(PipelineConfig("x", detector="motion", write_video=False))
            img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
            for i in range(3):
                vis, st = pipe.process(img, i, i / 10.0)
            self.assertEqual(vis.shape, (h, w, 3))
            self.assertIn("lane_state", st)

    def test_compare_with_labelled_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = moving_camera_sequence(CAM, n=6, speed_mps=5)
            samples = []
            for i, f in enumerate(frames):
                p = td / f"d{i}.png"
                cv2.imwrite(str(p), f)
                samples.append(Sample("kitti", "val", str(p), 640, 360, np.array([[100, 200, 160, 260]], np.float32), np.array([0])))
            ds = ISCVDataset(samples, "kitti-mini")
            write_video(td / "v.mp4", frames)
            res = compare(ds, str(td / "v.mp4"), str(td / "cmp"), "US-style (labelled)", "Video", detector="motion", max_frames=5)
            self.assertTrue((td / "cmp" / "comparison.mp4").exists())
            self.assertIsNotNone(res["US-style (labelled)"]["ap50_labelled_frames"])
            self.assertIsNone(res["Video"]["ap50_labelled_frames"])

    def test_runtime_utilities(self):
        p = Profiler()
        with p.stage("a"):
            pass
        self.assertIn("a", p.summary())
        self.assertEqual(list(prefetch(iter(range(5)))), [0, 1, 2, 3, 4])
        ad = AdaptiveScale(budget_ms=100.0, start=640)
        for _ in range(30):
            ad.update(400.0)
        self.assertLess(ad.width, 640)
        self.assertIn(select_device("auto"), ("cpu", "cuda:0", "mps"))


if __name__ == "__main__":
    unittest.main()
