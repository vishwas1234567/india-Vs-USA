"""Turn the v3 results (forecasting, simulation, runtime, synthetic checks) into LaTeX tables, macros and figures.

    python -m irp.report_v3          # reads results/_forecast, _sim, _runtime, _synthetic, _quality -> generated/ and figures/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .collect import tex_escape


def _load(p: Path) -> Optional[dict]:
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=2):
    return "--" if x is None else f"{float(x):.{nd}f}"


def build(results: str = "results", gen: str = "generated", figs: str = "figures") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, G, F = Path(results), Path(gen), Path(figs)
    G.mkdir(exist_ok=True)
    F.mkdir(exist_ok=True)
    macros: Dict[str, str] = {}

    fc = _load(R / "_forecast" / "forecast.json")
    if fc:
        rows = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
                r"Test set & Model & ADE (bw) & FDE (bw) & ADE (px@960) & FDE (px@960) & $\Delta$ADE vs CV [95\% CI] \\", r"\midrule"]
        for name, label in (("held_out_tracks", "held-out tracks"), ("cross_scene", "cross-scene")):
            r = fc["results"].get(name)
            if not r:
                continue
            ci = r["ADE_bw_ci95_track_bootstrap"]["lstm_minus_cv_mean"]
            for model, mlabel in (("zero", "zero velocity"), ("cv_last", "const. vel. (last)"), ("cv_mean", "const. vel. (mean)"),
                                  ("ridge", "ridge regression"), ("lstm", "LSTM (3-seed ensemble)")):
                m = r[model]
                delta = f"{ci[0]:+.3f}\\,\\ldots\\,{ci[1]:+.3f}" if model == "lstm" else ""
                first = f"{label} ({r['n_tracks']} tracks)" if model == "zero" else ""
                rows.append(" & ".join([first, mlabel, _f(m["ADE_bw"], 3), _f(m["FDE_bw"], 3), _f(m["ADE_px960"], 1),
                                        _f(m["FDE_px960"], 1), delta]) + r" \\")
            rows.append(r"\midrule")
        rows[-1] = r"\bottomrule"
        (G / "v3_forecast.tex").write_text("\n".join(rows + [r"\end{tabular}"]) + "\n")
        h, c = fc["results"]["held_out_tracks"], fc["results"].get("cross_scene")
        macros.update(vFcWindows=fc["windows"], vFcTracks=fc["tracks"], vFcHeldAdeCv=_f(h["cv_mean"]["ADE_bw"], 3),
                      vFcHeldAdeLstm=_f(h["lstm"]["ADE_bw"], 3), vFcHeldGain=f"{100 * (1 - h['lstm']['ADE_bw'] / h['cv_mean']['ADE_bw']):.0f}",
                      vFcHeldTracks=h["n_tracks"])
        lo, hi = h["ADE_bw_ci95_track_bootstrap"]["lstm_minus_cv_mean"]
        macros.update(vFcHeldCiLo=f"{lo:+.3f}", vFcHeldCiHi=f"{hi:+.3f}")
        if c:
            lo2, hi2 = c["ADE_bw_ci95_track_bootstrap"]["lstm_minus_cv_mean"]
            macros.update(vFcCrossAdeCv=_f(c["cv_mean"]["ADE_bw"], 3), vFcCrossAdeLstm=_f(c["lstm"]["ADE_bw"], 3),
                          vFcCrossGain=f"{100 * (1 - c['lstm']['ADE_bw'] / c['cv_mean']['ADE_bw']):.0f}", vFcCrossTracks=c["n_tracks"],
                          vFcCrossCiLo=f"{lo2:+.3f}", vFcCrossCiHi=f"{hi2:+.3f}")
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        models = ["zero", "cv_last", "cv_mean", "ridge", "lstm"]
        for k, (name, col) in enumerate((("held_out_tracks", "#3182bd"), ("cross_scene", "#fd8d3c"))):
            if name in fc["results"]:
                ax.bar(np.arange(5) + (k - 0.5) * 0.38, [fc["results"][name][m]["ADE_bw"] for m in models], width=0.38,
                       color=col, label=name.replace("_", " "))
        ax.set_xticks(range(5))
        ax.set_xticklabels(["zero", "CV last", "CV mean", "ridge", "LSTM"])
        ax.set_ylabel("ADE (body widths)")
        ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(F / "fig_v3_forecast.pdf")
        plt.close()

    st = _load(R / "_sim" / "stacks.json")
    if st:
        rows = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
                r"Stack & Episodes & Collision rate & Min gap (m) & Min TTC (s) & Jerk (m/s$^3$) & Progress (m) & MPC solve (ms) \\", r"\midrule"]
        for name, v in st.items():
            rows.append(" & ".join([tex_escape(name), str(v["episodes"]), _f(v["collision_rate"], 2), _f(v["min_gap_median_m"]),
                                    _f(v["min_ttc_median_s"]), _f(v["mean_jerk"]), _f(v["progress_mean_m"], 1),
                                    _f(v["solve_ms_mean"], 0) if v["solve_ms_mean"] > 0 else "--"]) + r" \\")
        (G / "v3_stacks.tex").write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        for tag, key in (("FsmLqrNo", "fsm+lqr (no intent)"), ("FsmLqr", "fsm+lqr+intent"), ("FsmMpc", "fsm+mpc+intent"), ("Rl", "rl+lqr+intent")):
            if key in st:
                macros[f"vSim{tag}Coll"] = _f(st[key]["collision_rate"], 2)
                macros[f"vSim{tag}Prog"] = _f(st[key]["progress_mean_m"], 0)
                macros[f"vSim{tag}Ttc"] = _f(st[key]["min_ttc_median_s"], 1)
        macros["vSimEpisodes"] = st[next(iter(st))]["episodes"]
        if "fsm+mpc+intent" in st:
            macros["vMpcSolve"] = _f(st["fsm+mpc+intent"]["solve_ms_mean"], 0)
    rl = _load(R / "_sim" / "rl_training.json")
    if rl:
        macros.update(vRlEpisodes=rl["episodes"], vRlSeconds=_f(rl["seconds"], 0), vRlStates=rl["visited_states"])

    fa = _load(R / "_sim" / "faults.json")
    if fa:
        faults = []
        for k in fa:
            f = k.split("|")[0]
            if f not in faults:
                faults.append(f)
        rows = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Fault & \multicolumn{2}{c}{Collision rate} & \multicolumn{2}{c}{Time in fallback} \\",
                r" & primary only & + supervisor & primary only & + supervisor \\", r"\midrule"]
        for f in faults:
            a, b = fa[f"{f}|primary only"], fa[f"{f}|supervisor"]
            rows.append(" & ".join([f, _f(a["collision_rate"]), _f(b["collision_rate"]), _f(a["degraded_frac"]), _f(b["degraded_frac"])]) + r" \\")
        (G / "v3_faults.tex").write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        for f in ("dropout", "freeze", "nan", "crash", "stall", "none"):
            if f in faults:
                macros[f"vFault{f.capitalize()}Prim"] = _f(fa[f"{f}|primary only"]["collision_rate"], 2)
                macros[f"vFault{f.capitalize()}Sup"] = _f(fa[f"{f}|supervisor"]["collision_rate"], 2)
        macros["vFaultCrashFallback"] = _f(fa["crash|supervisor"]["degraded_frac"], 2)
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        x = np.arange(len(faults))
        ax.bar(x - 0.19, [fa[f"{f}|primary only"]["collision_rate"] for f in faults], 0.38, color="#d95f02", label="primary pipeline only")
        ax.bar(x + 0.19, [fa[f"{f}|supervisor"]["collision_rate"] for f in faults], 0.38, color="#1b6ca8", label="with safety supervisor")
        ax.set_xticks(x)
        ax.set_xticklabels(faults)
        ax.set_ylabel("Collision rate")
        ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(F / "fig_v3_faults.pdf")
        plt.close()

    rt = _load(R / "_runtime" / "stage_latency.json")
    if rt:
        stages = ["ego", "detect", "track", "road", "lane", "scene", "render"]
        rows = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule", r"Width & ego & detect & track & road & lane & scene & render & total (ms) & fps \\", r"\midrule"]
        for w, r in rt["results"].items():
            rows.append(" & ".join([w] + [_f(r["stages_ms"].get(s), 1) for s in stages] + [_f(r["latency_ms_mean"], 1), _f(r["fps"], 1)]) + r" \\")
        (G / "v3_runtime.tex").write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        ws = list(rt["results"])
        macros.update(vRtFpsLow=_f(rt["results"][ws[0]]["fps"], 0), vRtFpsHigh=_f(rt["results"][ws[-1]]["fps"], 0),
                      vRtWidthHigh=ws[-1], vRtLane=_f(rt["results"][ws[-1]]["stages_ms"]["lane"], 0), vRtDevice=rt["results"][ws[0]]["device"])
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        bottom = np.zeros(len(ws))
        for s in stages:
            vals = np.array([rt["results"][w]["stages_ms"].get(s, 0.0) for w in ws])
            ax.bar(ws, vals, bottom=bottom, label=s)
            bottom += vals
        ax.set_xlabel("Processing width (px)")
        ax.set_ylabel("Latency (ms), one CPU core")
        ax.legend(fontsize=7, ncol=4)
        plt.tight_layout()
        plt.savefig(F / "fig_v3_runtime.pdf")
        plt.close()

    sy = _load(R / "_synthetic" / "synthetic.json")
    if sy:
        rows = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Synthetic lane scene & F1 & Ego-offset error (m) & Path source \\", r"\midrule"]
        for name, v in sy["lanes"].items():
            rows.append(" & ".join([tex_escape(name), _f(v["F1"]), _f(v["ego_offset_abs_err_m"], 3), v["source"]]) + r" \\")
        (G / "v3_synth.tex").write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        mo = sy["motion_detector"]
        macros.update(vSynRoadIou=_f(sy["road_iou_synthetic"], 2), vSynMotionHit=_f(mo["with moving object"]["object_hit_rate"], 2),
                      vSynMotionFp=_f(mo["no moving object"]["false_boxes_per_frame"], 1))
    q = R / "_quality" / "junction2_360p.json"

    body = ["% Auto-generated by irp.report_v3 - do not edit."]
    for k, v in macros.items():
        body.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    (G / "v3_macros.tex").write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    build()
