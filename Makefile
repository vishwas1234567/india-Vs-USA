# Indian-road perception: tests, experiments, quality, robustness study, results collection and reports.
# Usage:  make all            (tests -> experiments -> quality -> robustness -> collect -> PDFs)
#         make experiments    (individual targets: make results/b_main/summary.json)
PY ?= python3
export PYTHONPATH := src
CLIP_A ?= data/junction_clip.mp4
CLIP_B ?= data/junction2_360p.mp4
RUNS_A := main noshadow mt8 mt20 first5s a_kalman
RUNS_B := b_main b_greedy b_enh b_noshadow
COMMON := --no-video

.PHONY: all test experiments quality benchmark robustness v3 collect report paper plan clean-results

all: test experiments quality benchmark robustness v3 collect report paper plan

test:
	mkdir -p results
	$(PY) -m unittest discover -s tests -v > results/tests.log 2>&1 || (cat results/tests.log; exit 1)
	tail -n 3 results/tests.log

experiments: $(addprefix results/,$(addsuffix /summary.json,$(RUNS_A) $(RUNS_B)))

# ---- Junction 1 (1280x720, 24 s): v1 baseline runs and ablations
results/main/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/main --label "J1 baseline (greedy tracker)"
results/noshadow/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/noshadow $(COMMON) --no-shadow-suppression --label "J1 ablation: shadows counted as foreground"
results/mt8/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/mt8 $(COMMON) --min-track-frames 8 --label "J1 min. track length 8 frames"
results/mt20/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/mt20 $(COMMON) --min-track-frames 20 --label "J1 min. track length 20 frames"
results/first5s/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/first5s $(COMMON) --duration 5 --label "J1 first 5 s only"
results/a_kalman/summary.json:
	$(PY) -m irp.static --source $(CLIP_A) --out results/a_kalman $(COMMON) --tracker kalman --label "J1 Kalman + Hungarian tracker"

# ---- Junction 2 (640x360, 118.8 s, low quality)
results/b_main/summary.json:
	$(PY) -m irp.static --source $(CLIP_B) --out results/b_main --tracker kalman --label "J2 (360p) Kalman tracker, no enhancement"
results/b_greedy/summary.json:
	$(PY) -m irp.static --source $(CLIP_B) --out results/b_greedy $(COMMON) --tracker greedy --label "J2 greedy tracker"
results/b_enh/summary.json:
	$(PY) -m irp.static --source $(CLIP_B) --out results/b_enh $(COMMON) --tracker kalman --enhance full --label "J2 Kalman + denoise + CLAHE"
results/b_noshadow/summary.json:
	$(PY) -m irp.static --source $(CLIP_B) --out results/b_noshadow $(COMMON) --tracker kalman --no-shadow-suppression --label "J2 ablation: shadows counted as foreground"

quality: results/_quality/junction1_720p.json results/_quality/junction2_360p.json
results/_quality/junction1_720p.json:
	$(PY) -m irp.quality --source $(CLIP_A) --out $@ > /dev/null
results/_quality/junction2_360p.json:
	$(PY) -m irp.quality --source $(CLIP_B) --out $@ > /dev/null

benchmark: results/_benchmark/tracker_benchmark.csv
results/_benchmark/tracker_benchmark.csv:
	$(PY) -m irp.bench --out results/_benchmark --seeds 20 > /dev/null

robustness: results/_robustness/robustness.csv
results/_robustness/robustness.csv:
	$(PY) -m irp.robustness --source $(CLIP_A) --out results/_robustness --tracker kalman > /dev/null

# ---- v3: forecasting, simulation suites (RL training, stacks, faults), runtime, synthetic checks
v3:
	$(PY) -m irp.predict --runs results/b_main --test-runs results/a_kalman --out results/_forecast > /dev/null
	$(PY) -m irp.sim --suite rl --episodes 1500
	$(PY) -m irp.sim --suite stacks --seeds 20 --configs "fsm+lqr (no intent),fsm+lqr+intent,rl+lqr+intent" > /dev/null
	$(PY) -m irp.sim --suite stacks --seeds 20 --configs "fsm+mpc+intent" --scenarios ped_cross,auto_cut_in > /dev/null
	$(PY) -m irp.sim --suite stacks --seeds 20 --configs "fsm+mpc+intent" --scenarios two_wheeler_weave,stopped_vehicle,lead_brake > /dev/null
	$(PY) -m irp.sim --suite faults --seeds 20 > /dev/null
	$(PY) -m irp.bench_runtime --source $(CLIP_A) --out results/_runtime --frames 60 > /dev/null
	$(PY) -m irp.eval_synth --out results/_synthetic > /dev/null
	$(PY) -m irp.compare --a $(CLIP_A) --label-a "Junction 1 (720p)" --b $(CLIP_B) --label-b "Junction 2 (360p)" --detector motion --max-frames 120 --out results/compare > /dev/null

collect:
	$(PY) -m irp.collect
	$(PY) -m irp.report_v3

report: collect
	latexmk -pdf -interaction=nonstopmode report.tex

paper: collect
	latexmk -pdf -interaction=nonstopmode paper.tex

plan: collect
	latexmk -pdf -interaction=nonstopmode research_plan.tex

clean-results:
	rm -rf results/main results/noshadow results/mt8 results/mt20 results/first5s results/a_kalman results/b_main results/b_greedy results/b_enh results/b_noshadow results/_robustness results/_quality results/_benchmark results/tests.log
