# Architecture

Two packages in one codebase (`src/`): **`iscvd`** (dataset, metrics, benchmark) and **`irp`** (perception, prediction, planning,
control, safety, simulation, reports).

```
                      any video / image / folder / URL / webcam
                                      |
                           irp.sources.FrameSource   (resize, skip bad frames, natural sort)
                                      |
        +-----------------------------+------------------------------+
        | fixed camera (auto-detected)                               | moving camera / images
        v                                                            v
  irp.static (classical, CPU)                              irp.pipeline.Pipeline
  background + MOG2 + Kalman tracks                        ego-motion (LK + homography)
  flow gates, bursts, vest/zebra cues                      detector: YOLO | ONNX(OpenCV DNN) | motion fallback
                                                           tracker: Kalman + Hungarian
                                                           road: colour model + temporal smoothing, free space
                                                           lanes: BEV, sliding windows, quadratic fits, free-space fallback
                                                           scene descriptor, panoptic fusion, intent (HMM), risk
                                      |
              irp.predict (LSTM, NumPy)  ->  irp.plan (FSM | Q-learning)  ->  irp.control (MPC | LQR)
                                                                |
                                   irp.safety.SafetySupervisor (independent radar-like sensor, fallback controller)
                                      |
                              vehicle commands (accel, steering)

  irp.sim: closed-loop simulator (scenarios, perception faults) drives the whole stack for benchmarking
  iscvd:  loaders (IDD, KITTI, BDD100K, Cityscapes, YOLO, COCO, folder, video) -> unified taxonomy -> metrics -> benchmark
  irp.compare: one side-by-side video of two sources with the same algorithm and a summary card
```

## CPU and GPU
* Everything classical runs on CPU with OpenCV/NumPy/SciPy (`requirements.txt`). Measured: 22-40 fps on one CPU core without a deep model.
* Learned detectors: `--detector yolo:<weights> --device cuda:0` (FP16 on GPU) or `--detector onnx:<model.onnx>` (CPU via OpenCV DNN, no PyTorch).
* `irp.runtime`: device selection, per-stage profiler, threaded frame prefetch, ffmpeg H.264 writer, adaptive-resolution controller.
* GPU paths are implemented but were not executed in the build environment (no GPU, no PyTorch).

## Safety design
The supervisor shares no state with the AI pipeline. It uses one independent range sensor and vehicle state to (1) monitor the primary
(exceptions, latency, non-finite output, staleness), (2) enforce a stopping envelope, (3) run a deterministic fallback (brake/limp, hold lane) and latch SAFE_STOP.
Production requires separate hardware and a real independent sensor; the simulator models both.

## Extension points
* New detector: any callable `detector(frame, motion=None) -> List[Detection]` (`irp.detect`).
* New segmenter: any callable `frame -> uint8 road mask` (`irp.road.RoadSegmenter(base=...)`).
* New dataset: add a loader in `iscvd/formats.py` and a mapping in `iscvd/taxonomy.py`.
* New planner/controller: implement `decide(WorldView) -> Decision` (`irp.plan`) or `solve(...)` (`irp.control`).
* New experiment: `python -m irp.static ... --out results/<name>` then `make collect report paper`.
