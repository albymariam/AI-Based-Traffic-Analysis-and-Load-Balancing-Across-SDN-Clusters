"""Staleness study: tables + figure + paired tests vs the centralised controller.
   [RUNS_DIR=<folder with run directories>] T11_TOPOLOGY=abilene12 PYTHONPATH=. .venv/bin/python evaluation/staleness_analysis.py"""
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common.config import ROOT, load_yaml

RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "results/runs"))      # RUNS_DIR=... reads runs from another folder
TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
EST = os.environ.get("DEMAND", "oracle") == "counters"        # DEMAND=counters -> analyse the *_est runs (demand from flow counters)
SFX = "_est" if EST else ""
SCEN, CENTRAL, STATIC = "abilene_replay", "transformer_minmax_lp" + SFX, "static_shortest"
METRICS = ["mean_mlu", "p95_mlu", "jfi", "rtt_ms_mean", "fct_p95_s", "reroutes_total", "reversal_rate",
           "link_drops_total"]
LINK_REC, FLOW_REC = 64, 48          # assumed bytes per synchronised link / flow record (MODELLED, not measured)
WARM = load_yaml("traffic")["warmup_s"]
C = len(load_yaml("topology")["domains"])
N_LINKS = 2 * len(load_yaml("topology")["links"])      # directed links (30 for Abilene)


def ci95(x):
    x = pd.Series(x).dropna()
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def reversal_rate(d):
    """Fraction of reroutes that send a flow back to a path it had already used (ping-pong)."""
    f = d / "reroutes.csv"
    if not f.exists():
        return np.nan
    r = pd.read_csv(f).sort_values("t")
    seen, rev, n = {}, 0, 0
    for _, row in r.iterrows():
        prev = seen.setdefault(row.flow_id, {row.old_path})
        if row.t >= WARM:
            n += 1
            rev += row.new_path in prev
        prev.add(row.new_path)
    return rev / n if n else 0.0


def sync_kBps(d, delta):
    """MODELLED state-sync overhead: every controller sends its link+flow state to the other C-1 every `delta` s."""
    f = d / "plans.csv"
    if delta <= 0 or not f.exists():
        return np.nan
    n_el = pd.read_csv(f).query("t >= @WARM").n_elephants.mean()
    return (C - 1) * (N_LINKS * LINK_REC + C * n_el * FLOW_REC) / delta / 1000.0


rows = []
for d in sorted(RUNS.glob("*/")):
    f = d / "summary.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r["seed"] < 100 and r["scenario"] == SCEN:
            r["dir"] = str(d)
            rows.append(r)
df = pd.DataFrame(rows).sort_values("dir").drop_duplicates(["method", "seed"], keep="last")
m = df.method.str.extract(r"^dist_lp_%ss(\d+)$" % ("est_" if EST else ""))[0]
df["delta"] = m.astype(float)
df["reversal_rate"] = [reversal_rate(Path(d)) for d in df.dir]
df["sync_kBps_modelled"] = [sync_kBps(Path(d), dl) for d, dl in zip(df.dir, df.delta)]
METRICS = [m for m in METRICS if m in df and df[m].notna().any()]      # e.g. no FCT in the elephant-only replay
dist = df[df.delta.notna()]
deltas = sorted(dist.delta.unique())
cent, stat = df[df.method == CENTRAL], df[df.method == STATIC]
if dist.empty or cent.empty:
    raise SystemExit("need dist_lp_%ss* runs and %s runs for %s" % ("est_" if EST else "", CENTRAL, SCEN))

# ---- table
out = []
for label, g in [("static", stat), ("centralised", cent)] + [("dist_s%d" % dl, dist[dist.delta == dl]) for dl in deltas]:
    row = {"setting": label, "n_seeds": len(g)}
    for mt in METRICS + ["sync_kBps_modelled"]:
        row[mt] = "%.3f ± %.3f" % (g[mt].mean(), ci95(g[mt])) if g[mt].notna().any() else ""
    out.append(row)
pd.DataFrame(out).to_csv(TAB / ("staleness_summary%s.csv" % SFX), index=False)

# ---- paired tests (same seeds = same traffic): vs static, vs centralised, vs the fresh-state distributed control (delta 0)
tests = []
refs = [("static", stat), ("centralised", cent), ("dist_s0", dist[dist.delta == 0])]
for dl in deltas:
    for ref, rdf in refs:
        if ref == "dist_s0" and dl == 0:
            continue
        for mt in [m for m in ["mean_mlu", "p95_mlu", "peak_mlu", "jfi", "rtt_ms_mean"] if m in df]:
            a = dist[dist.delta == dl].set_index("seed")[mt]
            b = rdf.set_index("seed")[mt]
            j = pd.concat([a, b], axis=1, keys=["dist", "ref"]).dropna()
            if len(j) >= 3:
                dd = j.dist - j.ref
                tests.append(dict(delta_s=dl, vs=ref, metric=mt, n=len(j), mean_diff=round(dd.mean(), 4),
                                  ci95=round(ci95(dd), 4), pct=round(100 * dd.mean() / j.ref.mean(), 2),
                                  p_value=round(stats.ttest_rel(j.dist, j.ref).pvalue, 4)))
pd.DataFrame(tests).to_csv(TAB / ("staleness_paired%s.csv" % SFX), index=False)

# ---- figure
fig, axs = plt.subplots(2, 4, figsize=(17, 7))
xs = np.arange(len(deltas))
for ax, mt in zip(axs.ravel(), METRICS):
    mu = [dist[dist.delta == dl][mt].mean() for dl in deltas]
    er = [ci95(dist[dist.delta == dl][mt]) for dl in deltas]
    ax.errorbar(xs, mu, yerr=er, marker="o", capsize=3, label="distributed")
    ax.axhline(cent[mt].mean(), ls="--", c="g", label="centralised")
    if stat[mt].notna().any():
        ax.axhline(stat[mt].mean(), ls=":", c="r", label="static shortest")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d" % dl for dl in deltas])
    ax.set(xlabel="state staleness Δ (s)", title=mt)
    ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
for ax in axs.ravel()[len(METRICS):]:
    ax.axis("off")
fig.suptitle("Effect of controller-state staleness (%s, mean ± 95%% CI, n=%d seeds)" % (SCEN, dist.seed.nunique()))
fig.tight_layout()
fig.savefig(FIG / ("staleness_sweep%s.png" % SFX), dpi=200)
print(pd.DataFrame(out).to_string(index=False))
print("\nwrote staleness_summary.csv, staleness_paired.csv, staleness_sweep.png")