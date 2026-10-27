"""Paper Figure 6 (staleness sweep, 10 seeds). Run from the repo root after evaluation/staleness_analysis.py:
   PYTHONPATH=. .venv/bin/python scripts/make_fig6_paper.py"""
import csv, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/tables/summary_mean_ci.csv"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
def pm(s):
    m = re.match(r"\s*([-\d.eE]+)\s*±\s*([-\d.eE]+)", s or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
ss = {r["setting"]: r for r in csv.DictReader(open(ROOT / "results/tables/staleness_summary.csv"))}
main = {r["method"]: r for r in csv.DictReader(open(R)) if r["scenario"] == "abilene_replay"}
stat, cent = main["static_shortest"], main["transformer_minmax_lp"]
deltas = [0, 5, 10, 20, 40]
pan = [("mean_mlu", "Mean MLU", False), ("p95_mlu", "95th-percentile MLU", False), ("jfi", "Jain's fairness index", False),
       ("rtt_ms_mean", "Mean RTT (ms)", False), ("reroutes_total", "Reroutes per run", True),
       ("reversal_rate", "Reversal rate (×10$^{-3}$)", True), ("link_drops_total", "Link drops per run", True)]
fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.0)); axl = axes.ravel()
for ax, (k, lab, nonneg) in zip(axl, pan):
    ys, es = [], []
    for dl in deltas:
        v, e = pm(ss["dist_s%d" % dl][k]); ys.append(v); es.append(e)
    sc = 1e3 if k == "reversal_rate" else 1
    ys, es = [y * sc for y in ys], [e * sc for e in es]
    lo = [min(e, y) if nonneg else e for y, e in zip(ys, es)]
    ax.errorbar(deltas, ys, yerr=[lo, es], fmt="o-", color="#1f4e8c", lw=1.3, ms=4, capsize=2.5, elinewidth=0.9)
    for src, st, c in ((cent, "--", "#2e8b57"), (stat, ":", "#c8553d")):
        v, _ = pm(src.get(k, ""))
        if v is not None: ax.axhline(v, ls=st, color=c, lw=1.2)
    ax.set_xticks(deltas); ax.set_xlim(-4, 44)
    ax.set_title(lab, pad=4); ax.set_xlabel("State age Δ (s)")
    ax.yaxis.grid(True, lw=0.3, color="#ddd"); ax.set_axisbelow(True)
    if nonneg: ax.set_ylim(bottom=0)
axl[7].axis("off")
axl[7].legend(handles=[Line2D([], [], color="#1f4e8c", marker="o", lw=1.3, label="Distributed"),
                       Line2D([], [], color="#2e8b57", ls="--", lw=1.2, label="Centralised"),
                       Line2D([], [], color="#c8553d", ls=":", lw=1.2, label="Static shortest")],
              loc="center", frameon=False, fontsize=8)
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.12, wspace=0.42, hspace=0.62)
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results/figures/fig6_staleness_paper.png", dpi=300)
