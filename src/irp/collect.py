"""Collect experiment runs into LaTeX tables, macros and figures for the report.

Every folder in results/ that contains summary.json + params.json is one experiment.
Re-run after adding an experiment:   python -m irp.collect

Conventions used for the named macros/figures (other run names still appear in the table):
    main        baseline run on the full clip
    noshadow    shadow-suppression ablation
    mt8, mt20   minimum-track-length sensitivity
    first5s     first five seconds only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger("irp.collect")
Run = Dict[str, object]


def load_runs(results_dir: Path) -> Dict[str, Run]:
    runs: Dict[str, Run] = {}
    dirs = sorted(Path(results_dir).iterdir(), key=lambda d: (d.name != "main", d.name))  # baseline first
    for d in dirs:
        if (d / "summary.json").exists() and (d / "params.json").exists():
            runs[d.name] = {"dir": d,
                            "summary": json.loads((d / "summary.json").read_text()),
                            "params": json.loads((d / "params.json").read_text())}
    return runs


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text


def _dirs(s: dict) -> str:
    d = s.get("direction_counts", {})
    return "/".join(str(d.get(k, 0)) for k in ("right->left", "left->right", "toward camera", "away from camera"))


def write_experiments_table(runs: Dict[str, Run], path: Path) -> None:
    lines = [r"\begin{tabular}{@{}l p{4.6cm} r r c r r l r@{}}", r"\toprule",
             r"Run & Description & Tracks & Police & R$\to$L/L$\to$R/T/A & Occ.\ mean & Occ.\ p95 & Peak (t) & Speed \\",
             r"\midrule"]
    for name, r in runs.items():
        s, p = r["summary"], r["params"]
        pk = s["occupancy_peak"]
        speed = s.get("median_speed_body_widths_per_s")
        lines.append(" & ".join([
            tex_escape(name), tex_escape(p.get("label") or "-"), str(s["moving_tracks_confirmed"]),
            str(s["police_tracks_excluded"]), _dirs(s), f"{s['occupancy_mean'] * 100:.1f}\\,\\%",
            f"{s['occupancy_p95'] * 100:.1f}\\,\\%", f"{pk['value'] * 100:.1f}\\,\\% ({pk['t_s']:.1f}\\,s)",
            "-" if speed is None else f"{speed:.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_macros(runs: Dict[str, Run], tests_log: Optional[Path], path: Path) -> None:
    macros: Dict[str, str] = {}

    def put(name: str, value) -> None:
        macros[name] = str(value)

    main = runs.get("main")
    if main:
        s = main["summary"]
        d = s["direction_counts"]
        put("mDuration", f"{s['duration_s']:.1f}")
        put("mFramesAnalysed", s["frames_analysed"])
        put("mTracks", s["moving_tracks_confirmed"])
        put("mPolice", s["police_tracks_excluded"])
        put("mDirRL", d.get("right->left", 0))
        put("mDirLR", d.get("left->right", 0))
        put("mDirTo", d.get("toward camera", 0))
        put("mDirAway", d.get("away from camera", 0))
        put("mOccMean", f"{s['occupancy_mean'] * 100:.1f}")
        put("mOccPNN", f"{s['occupancy_p95'] * 100:.1f}")
        put("mOccPeak", f"{s['occupancy_peak']['value'] * 100:.1f}")
        put("mOccPeakT", f"{s['occupancy_peak']['t_s']:.1f}")
        put("mVest", f"{s['police_vest_frame_frac'] * 100:.1f}")
        put("mSpeed", s["median_speed_body_widths_per_s"])
        put("mRoadArea", f"{s['road_area_frac'] * 100:.1f}")
        put("mZebra", s["zebra_stripe_candidates"])
        put("mTrackDur", s["track_duration_median_s"])
        put("mMeanUsers", s["mean_tracked_users"])
        put("mWall", s["wall_time_s"])
        put("mProcFps", s["processing_fps"])
        put("mOpenCV", main["params"].get("opencv", "n/a"))
        put("mNumpy", main["params"].get("numpy", "n/a"))
    ns = runs.get("noshadow")
    if main and ns:
        a, b = main["summary"], ns["summary"]
        put("mNSOccMean", f"{b['occupancy_mean'] * 100:.1f}")
        put("mNSOccPNN", f"{b['occupancy_p95'] * 100:.1f}")
        put("mNSOccPeak", f"{b['occupancy_peak']['value'] * 100:.1f}")
        put("mNSTracks", b["moving_tracks_confirmed"])
        put("mNSPolice", b["police_tracks_excluded"])
        gain = (b["occupancy_mean"] - a["occupancy_mean"]) * 100
        put("mShadowGainPP", f"{gain:.1f}")
        put("mShadowGainRel", f"{gain / (a['occupancy_mean'] * 100) * 100:.0f}")
    for key, nm in (("Low", "mt8"), ("High", "mt20")):
        if nm in runs:
            put(f"mMt{key}Tracks", runs[nm]["summary"]["moving_tracks_confirmed"])
            put(f"mMt{key}L", runs[nm]["params"]["min_track_frames"])
    f5 = runs.get("first5s")
    if f5:
        s = f5["summary"]
        put("mFiveDur", f"{s['duration_s']:.1f}")
        put("mFiveTracks", s["moving_tracks_confirmed"])
        put("mFiveOccMean", f"{s['occupancy_mean'] * 100:.1f}")
        put("mFiveOccPeak", f"{s['occupancy_peak']['value'] * 100:.1f}")
        put("mFiveVest", f"{s['police_vest_frame_frac'] * 100:.1f}")
    if tests_log and tests_log.exists():
        text = tests_log.read_text()
        m = re.search(r"Ran (\d+) tests?", text)
        if m:
            put("mTests", m.group(1))
            put("mTestsStatus", "all passed" if re.search(r"^OK\s*$", text, re.M) else "FAILED - see results/tests.log")
    body = ["% Auto-generated by irp.collect - do not edit."]
    for k, v in macros.items():
        body.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    path.write_text("\n".join(body) + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _grab(path: Path, times: Sequence[float], size=(640, 360)):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            out.append(cv2.resize(f, size))
    cap.release()
    return out


def _grid(frames, cols=3):
    if not frames:
        return None
    blank = np.full_like(frames[0], 255)
    frames = list(frames) + [blank] * ((-len(frames)) % cols)
    return np.vstack([np.hstack(frames[i:i + cols]) for i in range(0, len(frames), cols)])


def _read_frames_csv(path: Path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def make_figures(runs: Dict[str, Run], clip: Path, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    main = runs.get("main")
    if not main:
        log.warning("No 'main' run found; skipping figures.")
        return
    mdir, s = Path(main["dir"]), main["summary"]
    dur = s["duration_s"]
    times = np.linspace(0.04, 0.96, 6) * dur
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 88]

    if clip.exists():
        g = _grid(_grab(clip, times))
        if g is not None:
            cv2.imwrite(str(fig_dir / "fig_scene.jpg"), g, jpg)
    ann_path = mdir / "annotated.mp4"
    if ann_path.exists():
        ann = _grab(ann_path, times)
        g = _grid(ann)
        if g is not None:
            cv2.imwrite(str(fig_dir / "fig_annotated.jpg"), g, jpg)
        bg, rm = cv2.imread(str(mdir / "background.jpg")), cv2.imread(str(mdir / "road_mask.png"))
        if ann and bg is not None and rm is not None:
            cv2.imwrite(str(fig_dir / "fig_method.jpg"),
                        np.hstack([cv2.resize(bg, (640, 360)), cv2.resize(rm, (640, 360)), ann[min(2, len(ann) - 1)]]), jpg)

    rows = [r for r in _read_frames_csv(mdir / "frames.csv") if r["warmup"] == "0"]
    t = np.array([float(r["t_s"]) for r in rows])
    occ = np.array([float(r["road_occupancy"]) for r in rows]) * 100
    users = np.array([int(r["moving_users"]) for r in rows])
    kern = np.ones(30) / 30
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax[0].plot(t, occ, color="#9ecae1", lw=0.8, label="per frame")
    ax[0].plot(t, np.convolve(occ, kern, mode="same"), color="#08519c", lw=1.6, label="1 s average")
    ax[0].axhline(s["occupancy_mean"] * 100, color="gray", ls=":", lw=1)
    pk = s["occupancy_peak"]
    ax[0].axvline(pk["t_s"], color="crimson", ls="--", lw=1)
    ax[0].annotate(f"peak {pk['value'] * 100:.1f}% ({pk['t_s']:.1f} s)", (pk["t_s"], pk["value"] * 100),
                   xytext=(max(t.min(), pk["t_s"] - 0.3 * dur), pk["value"] * 100 * 0.85), fontsize=8, color="crimson")
    ax[0].set_ylabel("Road occupancy (%)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(alpha=.3)
    ax[1].plot(t, users, color="#e6550d", lw=1.2)
    ax[1].set_ylabel("Tracked moving users")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_occupancy.pdf")
    plt.close()

    order = ["right->left", "left->right", "toward camera", "away from camera"]
    vals = [s["direction_counts"].get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars = ax.bar(range(4), vals, color=["#3182bd", "#6baed6", "#fd8d3c", "#fdae6b"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["R to L", "L to R", "toward", "away"], fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .3, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Tracks")
    ax.set_ylim(0, max(vals + [1]) + 4)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_directions.pdf")
    plt.close()

    tracks = list(csv.DictReader(open(mdir / "tracks.csv"))) if (mdir / "tracks.csv").stat().st_size else []
    if tracks:
        sz = np.array([float(x["rel_size"]) for x in tracks])
        sp = np.array([float(x["body_widths_per_s"]) for x in tracks])
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        ax[0].hist(np.clip(sz, 0, 2.19), bins=np.arange(0, 2.4, 0.2), color="#3182bd", edgecolor="white")
        ax[0].set_xlabel("Relative size s (values >= 2.2 pooled)")
        ax[0].set_ylabel("Tracks")
        ax[1].hist(np.clip(sp, 0, 8.2), bins=np.arange(0, 9, 0.75), color="#fd8d3c", edgecolor="white")
        ax[1].set_xlabel("Relative speed (body widths/s, >= 8 pooled)")
        for a, v in zip(ax, (sz, sp)):
            a.axvline(np.median(v), color="crimson", ls="--", lw=1)
            a.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_hist.pdf")
        plt.close()

    ns = runs.get("noshadow")
    if ns:
        rows2 = [r for r in _read_frames_csv(Path(ns["dir"]) / "frames.csv") if r["warmup"] == "0"]
        t2 = np.array([float(r["t_s"]) for r in rows2])
        o2 = np.array([float(r["road_occupancy"]) for r in rows2]) * 100
        sm1, sm2 = np.convolve(occ, kern, mode="same"), np.convolve(o2, kern, mode="same")
        n = min(len(t), len(t2))
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.plot(t2[:n], sm2[:n], color="#d95f02", lw=1.6, label="Shadows treated as foreground (ablation)")
        ax.plot(t[:n], sm1[:n], color="#1b6ca8", lw=1.6, label="Shadow suppression on (baseline)")
        ax.fill_between(t[:n], sm1[:n], sm2[:n], color="#d95f02", alpha=.15)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Road occupancy (%), 1 s average")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_ablation.pdf")
        plt.close()

    sens = sorted(((r["params"]["min_track_frames"], r["summary"]["moving_tracks_confirmed"])
                   for r in runs.values()
                   if r["params"].get("shadow_suppression", True) and not r["params"].get("duration_s")), key=lambda x: x[0])
    if len(sens) >= 2:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.plot([x for x, _ in sens], [y for _, y in sens], "o-", color="#3182bd")
        for x, y in sens:
            ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Minimum track length (frames)")
        ax.set_ylabel("Confirmed tracks")
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_sensitivity.pdf")
        plt.close()


# --------------------------------------------------------------------------- #
# Paper-level tables, macros and figures (quality, robustness, benchmark, junction 2)
# --------------------------------------------------------------------------- #
def _fmt(x, nd: int = 2) -> str:
    return "--" if x is None else f"{float(x):.{nd}f}"


def _load_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_quality_table(qdir: Path, path: Path) -> None:
    reports = {f.stem: json.loads(f.read_text()) for f in sorted(qdir.glob("*.json"))} if qdir.exists() else {}
    if not reports:
        return
    rows = [
        ("Resolution", lambda r: f"{r['width']}$\\times${r['height']}"),
        ("Duration (s)", lambda r: _fmt(r["duration_s"], 1)),
        ("Bits per pixel (file size)", lambda r: _fmt(r["bits_per_pixel"], 3)),
        ("Noise $\\sigma$ (grey levels)", lambda r: _fmt(r["noise_sigma"], 2)),
        ("Blockiness (1 = none)", lambda r: _fmt(r["blockiness"], 3)),
        ("Sharpness (Laplacian variance at 640 px)", lambda r: _fmt(r["sharpness_lap_var_640"], 0)),
        ("Overexposed pixels (\\%)", lambda r: _fmt(float(r["overexposed_frac"]) * 100, 1)),
        ("Mean brightness / contrast", lambda r: f"{_fmt(r['brightness_mean'], 0)} / {_fmt(r['contrast_std'], 0)}"),
        ("Inter-frame jitter, RMS (px)", lambda r: _fmt(r["jitter_px_rms"], 3)),
        ("Flags", lambda r: tex_escape(", ".join(r["flags"]) or "none")),
    ]
    names = list(reports)
    head = " & ".join(["Measure"] + [tex_escape(n) for n in names]) + r" \\"
    body = [" & ".join([label] + [fn(reports[n]) for n in names]) + r" \\" for label, fn in rows]
    path.write_text("\n".join([r"\begin{tabular}{@{}l" + "r" * len(names) + r"@{}}", r"\toprule", head, r"\midrule"]
                              + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_robustness_table(csv_path: Path, path: Path) -> None:
    rows = _load_csv(csv_path)
    if not rows:
        return
    conds = []
    for r in rows:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    get = {(r["condition"], r["enhance"]): r for r in rows}
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Occupancy $r$} & \multicolumn{2}{c}{MAE (pp)} & \multicolumn{2}{c}{Mean ratio} & \multicolumn{2}{c}{Track ratio} \\",
             r"Condition & none & enh. & none & enh. & none & enh. & none & enh. \\", r"\midrule"]
    for c in conds:
        a, b = get.get((c, "none")), get.get((c, "full"))
        cells = [tex_escape(c)]
        for key, nd in (("occ_pearson", 3), ("occ_mae_pp", 2), ("occ_mean_ratio", 2), ("tracks_ratio", 2)):
            cells += [_fmt(a[key], nd) if a else "--", _fmt(b[key], nd) if b else "--"]
        lines.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_benchmark_table(csv_path: Path, path: Path) -> None:
    rows = _load_csv(csv_path)
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Scenario & Tracker & Fragments/object & ID switches & Purity & Count ratio & False tracks \\", r"\midrule"]
    last = None
    for r in rows:
        pm = lambda k, nd: f"{float(r[k]):.{nd}f}$\\pm${float(r[k + '_std']):.{nd}f}"
        lines.append(" & ".join([r["scenario"] if r["scenario"] != last else "", r["tracker"],
                                 pm("frag_per_object", 2), pm("id_switches", 1), pm("purity", 3),
                                 pm("count_ratio", 2), pm("fp_tracks", 1)]) + r" \\")
        last = r["scenario"]
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_flow_table(run: Run, path: Path) -> None:
    s = run["summary"]
    lines = [r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
             r"Gate & Direction & Crossings & \multicolumn{4}{c}{Headway, all directions (s)} \\",
             r" & & & n & mean & median & p90 \\", r"\midrule"]
    for gate, counts in s["crossings"].items():
        hw = s["headway"][gate]
        dirs = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "--"
        lines.append(" & ".join([tex_escape(gate), "", tex_escape(dirs), str(hw["n"]), _fmt(hw["mean_s"]),
                                 _fmt(hw["median_s"]), _fmt(hw["p90_s"])]) + r" \\")
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_paper_macros(runs: Dict[str, Run], results: Path, path: Path) -> None:
    from .robustness import agreement

    m: Dict[str, str] = {}
    b = runs.get("b_main")
    if b:
        s = b["summary"]
        d = s["direction_counts"]
        m.update(mBDuration=f"{s['duration_s']:.1f}", mBFrames=s["frames_analysed"], mBTracks=s["moving_tracks_confirmed"],
                 mBPolice=s["police_tracks_excluded"], mBDirRL=d.get("right->left", 0), mBDirLR=d.get("left->right", 0),
                 mBDirTo=d.get("toward camera", 0), mBDirAway=d.get("away from camera", 0),
                 mBOccMean=f"{s['occupancy_mean'] * 100:.1f}", mBOccPNN=f"{s['occupancy_p95'] * 100:.1f}",
                 mBOccPeak=f"{s['occupancy_peak']['value'] * 100:.1f}", mBOccPeakT=f"{s['occupancy_peak']['t_s']:.1f}",
                 mBSpeed=s["median_speed_body_widths_per_s"], mBTrackDur=s["track_duration_median_s"],
                 mBVest=f"{s['police_vest_frame_frac'] * 100:.1f}", mBFps=s["processing_fps"],
                 mBMeanUsers=s["mean_tracked_users"], mBRoadArea=f"{s['road_area_frac'] * 100:.1f}")
        cr = s["crossings"]
        m.update(mBCrossVRL=cr["mid-vertical"].get("R->L", 0), mBCrossVLR=cr["mid-vertical"].get("L->R", 0),
                 mBCrossHToward=cr["near-horizontal"].get("toward", 0), mBCrossHAway=cr["near-horizontal"].get("away", 0),
                 mBHeadMed=_fmt(s["headway"]["mid-vertical"]["median_s"]), mBHeadMean=_fmt(s["headway"]["mid-vertical"]["mean_s"]),
                 mBBurstN=s["bursts"]["n"], mBBurstDur=_fmt(s["bursts"]["mean_duration_s"], 1),
                 mBBurstInt=_fmt(s["bursts"]["mean_interval_s"], 1), mBBurstCV=_fmt(s["bursts"]["interval_cv"], 2))
        for key, nm in (("Greedy", "b_greedy"), ("Enh", "b_enh"), ("NS", "b_noshadow")):
            if nm in runs:
                r = runs[nm]["summary"]
                m[f"mB{key}Tracks"] = r["moving_tracks_confirmed"]
                m[f"mB{key}Occ"] = f"{r['occupancy_mean'] * 100:.1f}"
                if nm == "b_noshadow":
                    m["mBNSGain"] = f"{(r['occupancy_mean'] - s['occupancy_mean']) * 100:.1f}"
                    m["mBNSGainRel"] = f"{(r['occupancy_mean'] - s['occupancy_mean']) / s['occupancy_mean'] * 100:.0f}"
                if nm == "b_enh":
                    ag = agreement(Path(b["dir"]), Path(runs[nm]["dir"]))
                    m.update(mBEnhR=f"{ag['occ_pearson']:.3f}", mBEnhMae=f"{ag['occ_mae_pp']:.2f}")
    rob = _load_csv(results / "_robustness" / "robustness.csv")
    if rob:
        g = {(r["condition"], r["enhance"]): r for r in rob}
        none = [float(r["occ_pearson"]) for r in rob if r["enhance"] == "none"]
        full = [float(r["occ_pearson"]) for r in rob if r["enhance"] == "full"]
        devs = [abs(float(r["tracks_ratio"]) - 1) * 100 for r in rob if r["enhance"] == "none"]
        m.update(mRobMinR=f"{min(none + full):.3f}", mRobMinRNone=f"{min(none):.3f}", mRobMaxR=f"{max(none):.3f}",
                 mRobMaxTrackDev=f"{max(devs):.0f}", mRobMeanR=f"{np.mean(none):.3f}", mRobMeanRFull=f"{np.mean(full):.3f}",
                 mRobNConds=len({r["condition"] for r in rob}),
                 mRobMaeNone=f"{np.mean([float(r['occ_mae_pp']) for r in rob if r['enhance'] == 'none']):.2f}",
                 mRobMaeFull=f"{np.mean([float(r['occ_mae_pp']) for r in rob if r['enhance'] == 'full']):.2f}")
        better = sum(1 for c in {r["condition"] for r in rob}
                     if (c, "none") in g and (c, "full") in g
                     and float(g[(c, "full")]["occ_mae_pp"]) < float(g[(c, "none")]["occ_mae_pp"]))
        m["mRobEnhBetter"] = better
        for key, c in (("Combo", "combo"), ("ResLow", "res180"), ("Crf", "crf45")):
            for mode, tag in (("none", "N"), ("full", "F")):
                if (c, mode) in g:
                    m[f"mRob{key}R{tag}"] = f"{float(g[(c, mode)]['occ_pearson']):.3f}"
                    m[f"mRob{key}Mae{tag}"] = f"{float(g[(c, mode)]['occ_mae_pp']):.2f}"
        ref = results / "_robustness" / "reference.json"
        if ref.exists():
            rj = json.loads(ref.read_text())
            m.update(mRobRefTracks=rj["tracks"], mRobRefOcc=rj["occ_mean_pct"])
    bench = _load_csv(results / "_benchmark" / "tracker_benchmark.csv")
    if bench:
        for r in bench:
            tag = ("G" if r["tracker"] == "greedy" else "K") + r["scenario"].capitalize()
            m[f"mBench{tag}Frag"] = r["frag_per_object"]
            m[f"mBench{tag}Idsw"] = f"{float(r['id_switches']):.0f}"
            m[f"mBench{tag}Pur"] = r["purity"]
        m["mBenchSeeds"] = bench[0]["seeds"]
    q = results / "_quality"
    for key, f in (("A", q / "junction1_720p.json"), ("B", q / "junction2_360p.json")):
        if f.exists():
            r = json.loads(f.read_text())
            m.update({f"mQ{key}Noise": _fmt(r["noise_sigma"], 2), f"mQ{key}Block": _fmt(r["blockiness"], 3),
                      f"mQ{key}Bpp": _fmt(r["bits_per_pixel"], 3), f"mQ{key}Jitter": _fmt(r["jitter_px_rms"], 3),
                      f"mQ{key}Over": _fmt(float(r["overexposed_frac"]) * 100, 1)})
    body = ["% Auto-generated by irp.collect (paper macros) - do not edit."]
    for k, v in m.items():
        body.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    path.write_text("\n".join(body) + "\n")
    (path.with_suffix(".names")).write_text("\n".join(m))  # list of defined macro names (for the report preamble)


def make_paper_figures(runs: Dict[str, Run], results: Path, clip_b: Path, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 88]
    b = runs.get("b_main")
    if b:
        bdir, s = Path(b["dir"]), b["summary"]
        times = np.linspace(0.04, 0.96, 6) * s["duration_s"]
        if clip_b.exists():
            g = _grid(_grab(clip_b, times))
            if g is not None:
                cv2.imwrite(str(fig_dir / "figB_scene.jpg"), g, jpg)
        if (bdir / "annotated.mp4").exists():
            peak_t = [s["occupancy_peak"]["t_s"], 0.25 * s["duration_s"], 0.5 * s["duration_s"],
                      0.65 * s["duration_s"], 0.8 * s["duration_s"], 0.92 * s["duration_s"]]
            g = _grid(_grab(bdir / "annotated.mp4", peak_t))
            if g is not None:
                cv2.imwrite(str(fig_dir / "figB_annotated.jpg"), g, jpg)
        rows = [r for r in _read_frames_csv(bdir / "frames.csv") if r["warmup"] == "0"]
        t = np.array([float(r["t_s"]) for r in rows])
        occ = np.array([float(r["road_occupancy"]) for r in rows]) * 100
        users = np.array([int(r["moving_users"]) for r in rows])
        flow = json.loads((bdir / "flow.json").read_text())
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
        ax[0].plot(t, occ, color="#9ecae1", lw=0.7)
        ax[0].plot(t, np.convolve(occ, np.ones(30) / 30, mode="same"), color="#08519c", lw=1.4, label="1 s average")
        for i, bu in enumerate(flow["bursts"]["bursts"]):
            ax[0].axvspan(bu["start_s"], bu["end_s"], color="#fdae6b", alpha=.35, label="burst" if i == 0 else None)
        ax[0].set_ylabel("Road occupancy (%)")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[0].grid(alpha=.3)
        ax[1].plot(t, users, color="#e6550d", lw=1.0)
        ax[1].set_ylabel("Tracked users")
        ax[1].set_xlabel("Time (s)")
        ax[1].grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "figB_occupancy.pdf")
        plt.close()
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        width = 10.0
        for k, (gate, counts) in enumerate(flow["bin_counts_10s"].items()):
            x = np.arange(len(counts)) * width + width * (0.1 + 0.4 * k)
            ax.bar(x, counts, width=width * 0.4, align="edge", label=gate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Crossings per 10 s")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "figB_flow.pdf")
        plt.close()
    rob = _load_csv(results / "_robustness" / "robustness.csv")
    if rob:
        conds = []
        for r in rob:
            if r["condition"] not in conds:
                conds.append(r["condition"])
        g = {(r["condition"], r["enhance"]): r for r in rob}
        x = np.arange(len(conds))
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
        for k, (mode, label, col) in enumerate((("none", "no enhancement", "#3182bd"), ("full", "denoise + CLAHE", "#fd8d3c"))):
            ax[0].bar(x + (k - 0.5) * 0.38, [float(g[(c, mode)]["occ_mae_pp"]) if (c, mode) in g else 0 for c in conds],
                      width=0.38, label=label, color=col)
            ax[1].bar(x + (k - 0.5) * 0.38, [float(g[(c, mode)]["tracks_ratio"]) if (c, mode) in g else 0 for c in conds],
                      width=0.38, color=col)
        ax[0].set_ylabel("Occupancy MAE vs reference (pp)")
        ax[1].set_ylabel("Track count / reference")
        ax[1].axhline(1.0, color="gray", ls=":", lw=1)
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
            a.grid(alpha=.3, axis="y")
        ax[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_robustness.pdf")
        plt.close()
    bench = _load_csv(results / "_benchmark" / "tracker_benchmark.csv")
    if bench:
        scen = []
        for r in bench:
            if r["scenario"] not in scen:
                scen.append(r["scenario"])
        gb = {(r["scenario"], r["tracker"]): r for r in bench}
        x = np.arange(len(scen))
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
        for k, (tr, col) in enumerate((("greedy", "#d95f02"), ("kalman", "#1b6ca8"))):
            for a, key, ylabel in ((ax[0], "frag_per_object", "Tracks per true object"), (ax[1], "id_switches", "ID switches (log)")):
                vals = [float(gb[(sc, tr)][key]) for sc in scen]
                errs = [float(gb[(sc, tr)][key + "_std"]) for sc in scen]
                a.bar(x + (k - 0.5) * 0.38, np.maximum(vals, 0.3 if key == "id_switches" else 0), width=0.38, yerr=errs,
                      label=tr, color=col, capsize=2)
                a.set_ylabel(ylabel)
        ax[1].set_yscale("log")
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels(scen, fontsize=8)
            a.grid(alpha=.3, axis="y")
        ax[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_benchmark.pdf")
        plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--clip", default="data/junction_clip.mp4")
    ap.add_argument("--clip-b", default="data/junction2_360p.mp4")
    ap.add_argument("--generated", default="generated")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args(argv)
    results, gen = Path(a.results), Path(a.generated)
    gen.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"No experiment runs found in {results}/ (run `make experiments` first).")
    write_experiments_table(runs, gen / "experiments.tex")
    write_macros(runs, results / "tests.log", gen / "macros.tex")
    make_figures(runs, Path(a.clip), Path(a.figures))
    write_quality_table(results / "_quality", gen / "quality.tex")
    write_robustness_table(results / "_robustness" / "robustness.csv", gen / "robustness.tex")
    write_benchmark_table(results / "_benchmark" / "tracker_benchmark.csv", gen / "benchmark.tex")
    if "b_main" in runs:
        write_flow_table(runs["b_main"], gen / "flow_b.tex")
    write_paper_macros(runs, results, gen / "paper_macros.tex")
    make_paper_figures(runs, results, Path(a.clip_b), Path(a.figures))
    log.info("Collected %d runs: %s", len(runs), ", ".join(runs))


if __name__ == "__main__":
    main()
