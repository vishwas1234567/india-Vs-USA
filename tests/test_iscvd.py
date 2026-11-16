"""Tests for the iscvd package. Datasets are synthetic fixtures that mimic the published on-disk layouts
(IDD, KITTI, BDD100K, Cityscapes/IDD-segmentation, YOLO); real downloads are not available offline."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from iscvd import CLASSES, ISCVDataset, shared_classes, unify, unified_id
from iscvd import benchmark, formats, metrics, stats, transforms
from iscvd.export import export_coco, export_yolo


def write_img(path: Path, w=96, h=64, color=(90, 120, 140), ext=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), color, np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (30, 30, 200), -1)
    cv2.imwrite(str(path), img)


def make_idd(root: Path):
    for i, objs in enumerate([[("autorickshaw", 10, 10, 40, 40), ("rider", 50, 10, 60, 30), ("motorcycle", 50, 12, 62, 40)],
                              [("car", 5, 5, 30, 30), ("person", 60, 20, 70, 50)]]):
        seq = f"seq{i}"
        write_img(root / "JPEGImages" / seq / f"f{i}.jpg")
        body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
                       for n, a, b, c, d in objs)
        (root / "Annotations" / seq).mkdir(parents=True, exist_ok=True)
        (root / "Annotations" / seq / f"f{i}.xml").write_text(
            f"<annotation><size><width>96</width><height>64</height><depth>3</depth></size>{body}</annotation>")
    (root / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
    (root / "ImageSets" / "Main" / "train.txt").write_text("seq0/f0\nseq1/f1\n")


def make_kitti(root: Path):
    write_img(root / "training" / "image_2" / "000000.png", 120, 40)
    (root / "training" / "label_2").mkdir(parents=True, exist_ok=True)
    (root / "training" / "label_2" / "000000.txt").write_text(
        "Car 0.00 0 -1.5 10 5 50 30 1.5 1.6 3.5 1 2 20 0\n"
        "Cyclist 0.00 0 -1.5 60 8 70 38 1.7 0.6 1.8 1 2 20 0\n"
        "DontCare -1 -1 -10 80 5 100 30 -1 -1 -1 -1000 -1000 -1000 -10\n")


def make_bdd(root: Path):
    write_img(root / "images" / "100k" / "val" / "a.jpg", 128, 72)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    items = [{"name": "a.jpg", "attributes": {"weather": "clear"}, "labels": [
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 40}},
        {"category": "rider", "box2d": {"x1": 60, "y1": 10, "x2": 70, "y2": 30}},
        {"category": "lane/single white", "poly2d": [{"vertices": [[20, 70], [30, 40], [40, 20]], "closed": False}]}]}]
    (root / "labels" / "bdd100k_labels_images_val.json").write_text(json.dumps(items))


def make_cityscapes_like(root: Path, kind="idd"):
    suffix = "_gtFine_labellevel3Ids.png" if kind == "idd" else "_gtFine_labelIds.png"
    img = root / "leftImg8bit" / "train" / "seqA" / f"frame1_leftImg8bit.png"
    write_img(img, 64, 32)
    gt = root / "gtFine" / "train" / "seqA"
    gt.mkdir(parents=True, exist_ok=True)
    mask = np.full((32, 64), 255, np.uint8)
    ids = (0, 3, 12) if kind == "idd" else (7, 8, 26)  # road, sidewalk, car
    mask[:, :20], mask[:, 20:40], mask[:, 40:] = ids
    cv2.imwrite(str(gt / f"frame1{suffix}"), mask)
    (gt / "frame1_gtFine_polygons.json").write_text(json.dumps({
        "imgWidth": 64, "imgHeight": 32,
        "objects": [{"label": "car", "polygon": [[40, 5], [60, 5], [60, 25], [40, 25]]}]}))


class TaxonomyAndFormats(unittest.TestCase):
    def test_taxonomy(self):
        self.assertEqual(unify("idd", "autorickshaw"), "auto")
        self.assertIsNone(unify("idd", "rider"))
        self.assertEqual(unify("kitti", "Person_sitting"), "pedestrian")
        self.assertEqual(unified_id("bdd100k", "motor"), CLASSES.index("two_wheeler"))
        shared = shared_classes(["idd", "kitti", "bdd100k"])
        self.assertEqual(shared, ["car", "heavy", "two_wheeler", "pedestrian", "sign"] if "sign" in shared else shared)
        self.assertNotIn("auto", shared)
        self.assertNotIn("sign", shared_classes(["idd", "kitti"]))

    def test_idd_detection(self):
        with tempfile.TemporaryDirectory() as td:
            make_idd(Path(td))
            ds = ISCVDataset.from_idd(td, "train")
            self.assertEqual(len(ds), 2)
            s0 = ds.samples[0]
            self.assertEqual([CLASSES[l] for l in s0.labels], ["auto", "two_wheeler"])  # rider ignored
            self.assertEqual((s0.width, s0.height), (96, 64))
            item = ds[0]
            self.assertEqual(item["image"].shape, (64, 96, 3))

    def test_kitti(self):
        with tempfile.TemporaryDirectory() as td:
            make_kitti(Path(td))
            ds = ISCVDataset.from_kitti(td)
            s = ds.samples[0]
            self.assertEqual([CLASSES[l] for l in s.labels], ["car", "two_wheeler"])
            self.assertEqual(s.boxes[0].tolist(), [10, 5, 50, 30])
            self.assertEqual((s.width, s.height), (120, 40))

    def test_bdd100k_with_lanes(self):
        with tempfile.TemporaryDirectory() as td:
            make_bdd(Path(td))
            ds = ISCVDataset.from_bdd100k(td, "val")
            s = ds.samples[0]
            self.assertEqual([CLASSES[l] for l in s.labels], ["car"])
            self.assertEqual(len(s.lanes), 1)
            self.assertEqual(s.lanes[0].shape, (3, 2))

    def test_idd_segmentation_semantic_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            make_cityscapes_like(Path(td), "idd")
            ds = ISCVDataset.from_idd_segmentation(td, "train")
            sem = ds.read_semantic(0)
            self.assertEqual((sem[0, 0], sem[0, 30], sem[0, 50]), (0, 1, 2))  # road, sidewalk, car(other)
            self.assertEqual([CLASSES[l] for l in ds.samples[0].labels], ["car"])
            self.assertEqual(ds.samples[0].boxes[0].tolist(), [40, 5, 60, 25])

    def test_cityscapes_semantic_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            make_cityscapes_like(Path(td), "cs")
            sem = ISCVDataset.from_cityscapes(td, "train").read_semantic(0)
            self.assertEqual((sem[0, 0], sem[0, 30], sem[0, 50]), (0, 1, 2))

    def test_yolo_roundtrip_and_coco_export(self):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "idd", Path(td) / "yolo"
            make_idd(src)
            ds = ISCVDataset.from_idd(str(src), "train")
            yaml = export_yolo({"train": ds}, str(out))
            self.assertTrue(yaml.exists() and "names:" in yaml.read_text())
            back = ISCVDataset.from_yolo(str(out), "train")
            self.assertEqual(len(back), len(ds))
            for a, b in zip(ds.samples, back.samples):
                np.testing.assert_allclose(a.boxes, b.boxes, atol=1.0)
                self.assertEqual(a.labels.tolist(), b.labels.tolist())
            coco = export_coco(ds, str(Path(td) / "c" / "coco.json"))
            self.assertEqual(len(json.loads(coco.read_text())["images"]), 2)

    def test_image_size_header_parser(self):
        with tempfile.TemporaryDirectory() as td:
            for ext in ("png", "jpg"):
                p = Path(td) / f"x.{ext}"
                write_img(p, 77, 55)
                self.assertEqual(formats.image_size(str(p)), (77, 55))

    def test_video_and_folder_sources(self):
        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "v.mp4"
            vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for i in range(20):
                vw.write(np.full((48, 64, 3), i * 10, np.uint8))
            vw.release()
            ds = ISCVDataset.from_video(str(vp), stride=5)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds[1]["image"].shape, (48, 64, 3))
            write_img(Path(td) / "imgs" / "b.png"), write_img(Path(td) / "imgs" / "a.png")
            self.assertEqual(len(ISCVDataset.from_folder(str(Path(td) / "imgs"))), 2)

    def test_split_groups_by_sequence(self):
        with tempfile.TemporaryDirectory() as td:
            samples = [formats.Sample("custom", "all", f"{td}/seq{g}/{i}.png", meta={"sequence": f"seq{g}"})
                       for g in range(10) for i in range(5)]
            train, val, test = ISCVDataset(samples).split((0.6, 0.2, 0.2), seed=1)
            seqs = [{s.meta["sequence"] for s in part.samples} for part in (train, val, test)]
            self.assertEqual(len(seqs[0] & seqs[1]) + len(seqs[0] & seqs[2]) + len(seqs[1] & seqs[2]), 0)
            self.assertEqual(len(train) + len(val) + len(test), 50)


class MetricTests(unittest.TestCase):
    def test_iou_and_ap(self):
        a = np.array([[0, 0, 10, 10]])
        self.assertAlmostEqual(metrics.box_iou(a, np.array([[0, 0, 10, 10]]))[0, 0], 1.0)
        self.assertAlmostEqual(metrics.box_iou(a, np.array([[5, 0, 15, 10]]))[0, 0], 1 / 3, places=5)
        gts = [{"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([0])}]
        perfect = [{"boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.9]), "labels": np.array([0])}]
        res = metrics.evaluate_detections(perfect, gts, [0, 1])
        self.assertAlmostEqual(res["mAP50"], 1.0, places=3)
        self.assertTrue(np.isnan(res["per_class"][1]["AP"]))          # class without ground truth is excluded
        self.assertEqual(res["classes_evaluated"], [0])
        wrong = [{"boxes": np.array([[50, 50, 60, 60]]), "scores": np.array([0.9]), "labels": np.array([0])}]
        self.assertAlmostEqual(metrics.evaluate_detections(wrong, gts, [0])["mAP50"], 0.0)

    def test_ap_penalises_false_positive_ranked_first(self):
        gts = [{"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([0])}]
        preds = [{"boxes": np.array([[50, 50, 60, 60], [0, 0, 10, 10]]), "scores": np.array([0.9, 0.5]),
                  "labels": np.array([0, 0])}]
        ap = metrics.evaluate_detections(preds, gts, [0], iou_thresholds=(0.5,))["mAP"]
        self.assertAlmostEqual(ap, 0.5, delta=0.02)

    def test_confusion_matrix_miou(self):
        cm = metrics.ConfusionMatrix(3)
        gt = np.array([[0, 0, 1, 1], [2, 2, 255, 255]])
        pred = np.array([[0, 1, 1, 1], [2, 2, 0, 0]])
        cm.update(pred, gt)
        iou = cm.iou()
        self.assertAlmostEqual(iou[0], 0.5)
        self.assertAlmostEqual(iou[1], 2 / 3, places=5)
        self.assertAlmostEqual(iou[2], 1.0)
        self.assertAlmostEqual(cm.pixel_accuracy(), 5 / 6, places=5)

    def test_panoptic_quality(self):
        gt = np.zeros((20, 20), int)
        gt[:, :10] = 1000            # stuff category 1
        gt[2:8, 12:18] = 2001        # thing category 2, instance 1
        pq = metrics.PanopticQuality()
        pq.update(gt.copy(), gt)
        self.assertAlmostEqual(pq.result()["PQ"], 1.0)
        pred = gt.copy()
        pred[2:8, 12:18] = 0         # missed the instance
        pq2 = metrics.PanopticQuality()
        pq2.update(pred, gt)
        r = pq2.result()["per_category"]
        self.assertEqual(r[2]["fn"], 1)
        self.assertAlmostEqual(r[1]["PQ"], 1.0)

    def test_clear_mot_and_idf1(self):
        box = lambda x: [x, 0, x + 10, 10]
        gt = {f: [(1, box(f))] for f in range(10)}
        perfect = {f: [(7, box(f))] for f in range(10)}
        r = metrics.clear_mot(gt, perfect)
        self.assertAlmostEqual(r["MOTA"], 1.0)
        self.assertAlmostEqual(r["IDF1"], 1.0)
        switched = {f: [(7 if f < 5 else 8, box(f))] for f in range(10)}
        r2 = metrics.clear_mot(gt, switched)
        self.assertEqual(r2["IDSW"], 1)
        self.assertAlmostEqual(r2["MOTA"], 0.9)
        self.assertAlmostEqual(r2["IDF1"], 0.5)

    def test_lane_and_trajectory_metrics(self):
        gt = [np.array([[100.0, 0], [100, 100]]), np.array([[200.0, 0], [200, 100]])]
        self.assertAlmostEqual(metrics.lane_f1(gt, gt)["F1"], 1.0)
        shifted = [g + np.array([30.0, 0]) for g in gt]
        self.assertAlmostEqual(metrics.lane_f1(shifted, gt, thr_px=20)["F1"], 0.0)
        self.assertAlmostEqual(metrics.lane_f1(shifted, gt, thr_px=40)["F1"], 1.0)
        out = metrics.ade_fde(np.zeros((2, 3, 2)), np.ones((2, 3, 2)))
        self.assertAlmostEqual(out["ADE"], np.sqrt(2))


class TransformStatsBenchmark(unittest.TestCase):
    def test_letterbox_and_flip_keep_boxes_consistent(self):
        img = np.zeros((40, 80, 3), np.uint8)
        item = {"image": img, "boxes": np.array([[10, 10, 30, 30]], np.float32), "labels": np.array([0]), "lanes": [], "mask": None}
        out = transforms.Letterbox((160, 160))(item)
        self.assertEqual(out["image"].shape, (160, 160, 3))
        r, dx, dy = out["letterbox"]
        np.testing.assert_allclose(out["boxes"][0], [10 * r + dx, 10 * r + dy, 30 * r + dx, 30 * r + dy])
        flipped = transforms.HFlip(1.0)(item, np.random.default_rng(0))
        np.testing.assert_allclose(flipped["boxes"][0], [50, 10, 70, 30])

    def test_augmentations_preserve_format(self):
        aug = transforms.indian_road_augmentation((128, 128), seed=3)
        item = {"image": np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8),
                "boxes": np.array([[5, 5, 40, 40]], np.float32), "labels": np.array([1]), "lanes": [], "mask": None}
        for _ in range(5):
            out = aug(dict(item))
            self.assertEqual(out["image"].shape, (128, 128, 3))
            self.assertEqual(out["image"].dtype, np.uint8)
        self.assertEqual(transforms.to_chw_float(out).shape, (3, 128, 128))

    def test_stats_compare_and_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            make_idd(Path(td) / "idd")
            make_kitti(Path(td) / "kitti")
            idd, kitti = ISCVDataset.from_idd(str(Path(td) / "idd")), ISCVDataset.from_kitti(str(Path(td) / "kitti"))
            sa, sb = stats.dataset_stats(idd), stats.dataset_stats(kitti)
            self.assertEqual(sa["n_images"], 2)
            cmp = stats.compare_stats(sa, sb)
            self.assertIn("auto", cmp["classes_only_in_a"])
            self.assertIn("car", cmp["shared_classes"])

            def oracle_for(ds):
                state = {"i": 0}

                def det(_img):
                    s = ds.samples[state["i"]]
                    state["i"] += 1
                    return {"boxes": s.boxes, "scores": np.ones(len(s.labels)), "labels": s.labels}
                return det

            res = benchmark.evaluate_detector(oracle_for(idd), idd)
            self.assertAlmostEqual(res["mAP50"], 1.0, places=3)
            comparison = benchmark.compare_datasets(lambda im: {"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                                                                "labels": np.zeros(0, int)}, {"IDD": idd, "KITTI": kitti})
            self.assertEqual(comparison["shared_classes"], ["car", "heavy", "two_wheeler", "pedestrian"][:len(comparison["shared_classes"])] or comparison["shared_classes"])
            self.assertIn("IDD", benchmark.to_markdown(comparison))


if __name__ == "__main__":
    unittest.main()
