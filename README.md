# Indian-road perception and autonomy toolkit

Two packages in one codebase (`src/`):
* **`irp`** - perception (any video/image input, fixed or moving camera), tracking, road and lane understanding, trajectory forecasting (LSTM),
  intent (HMM), behaviour planning (FSM / Q-learning), control (LQR / MPC), safety supervisor, closed-loop simulator, comparison video.
* **`iscvd`** - Indian Self-driving Car Vision Dataset toolkit: loaders (IDD, KITTI, BDD100K, Cityscapes, YOLO, COCO, folder, video), unified taxonomy,
  augmentations for Indian conditions, metrics (AP, mIoU, PQ, CLEAR-MOT/IDF1, lane F1, ADE/FDE), export, benchmark harness.

```bash
pip install -r requirements.txt            # core (CPU); requirements-dl.txt adds torch, ultralytics, transformers, yt-dlp
python -m irp run --source drive.mp4 --out results/drive1          # any video / image / folder / URL (auto fixed vs moving camera)
python -m irp run --source "frames/*.jpg" --out results/frames --no-sequence
python -m irp compare --a india.mp4 --label-a India --b us.mp4 --label-b US --out results/compare   # one comparison video
python -m irp dataset stats --format idd --root /data/IDD_Detection --split val
python -m irp dataset eval  --format kitti --root /data/KITTI --split training --detector yolo:yolov8n.pt
python -m irp forecast --runs results/b_main --test-runs results/a_kalman
python -m irp sim --suite faults --seeds 20
make all                                   # tests -> experiments -> quality -> benchmark -> robustness -> v3 -> collect -> PDFs
```

Documents (all numbers are generated from `results/`): `report.pdf` (project report), `paper.pdf` (research-paper version of the low-quality-video study),
`research_plan.pdf` (plan for the report/paper and the Indian-vs-US benchmark), `docs/ARCHITECTURE.md`.

## Status (what was run, what was not)
* **Completed on real clips:** static-camera analysis, degradation study, Kalman tracking, trajectory forecasting (LSTM vs baselines), CPU runtime, comparison-video demo.
* **Simulation only:** planning (FSM, Q-learning, MPC, LQR), intent, safety supervisor and fault injection.
* **Verified with synthetic data/fixtures:** lane and road modules, dataset loaders, metrics, detector decoding.
* **Not run (needs data or GPU):** learned detectors/segmenters/panoptic networks, the Indian-vs-US benchmark (no IDD/KITTI/BDD100K files), validation on moving-vehicle footage, GPU timing, deep RL.
Run `python -m unittest discover -s tests` (56 tests) after any change.
