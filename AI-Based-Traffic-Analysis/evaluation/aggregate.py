"""Real results: mean ± 95% CI table, paired tests vs static_shortest, MLU-over-time, bars, RTT CDF."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common.config import ROOT

RUNS, TAB, FIG = ROOT / "results/runs", ROOT / "results/tables", ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
METRICS = ["mean_mlu", "p95_mlu", "jfi", "rtt_ms_mean", "rtt_ms_p95", "rtt_jitter_ms", "fct_p95_s",
           "udp_loss_pct", "udp_jitter_ms_mean", "throughput_mbps_aggregate", "link_drops_total",
           "flowmods_per_s", "reroutes_total", "reroute_decision_ms_mean", "flows_failed_pct", "lp_solve_ms_mean", "plans_applied"]
ORDER = ["static_shortest", "static_tracked", "ecmp", "round_robin", "lstm_hysteresis",
         "transformer_minmax_lp", "tcn_minmax_lp"]
BASE = "static_shortest"


def ci95(x):
    x = x.dropna()
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


rows = []
for d in sorted(RUNS.glob("*/")):
    f = d / "summary.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r["seed"] < 100:                       # evaluation seeds only
            r["dir"] = str(d)
            rows.append(r)
df = pd.DataFrame(rows)
df = df.sort_values("dir").drop_duplicates(["scenario", "method", "seed"], keep="last")  # latest run wins
mets = [m for m in METRICS if m in df]

# ---- table: mean ± CI
out = []
for (sc, me), g in df.groupby(["scenario", "method"]):
    row = {"scenario": sc, "method": me, "n_seeds": len(g)}
    for m in mets:
        row[m] = "%.3f ± %.3f" % (g[m].mean(), ci95(g[m])) if g[m].notna().any() else ""
    out.append(row)
pd.DataFrame(out).to_csv(TAB / "summary_mean_ci.csv", index=False)

# ---- paired tests vs baseline (same seeds = same traffic)
tests = []
for sc, g in df.groupby("scenario"):
    for m in ["mean_mlu", "jfi", "rtt_ms_mean", "rtt_ms_p95"]:
        if m not in g:
            continue
        pv = g.pivot_table(index="seed", columns="method", values=m)
        for me in pv.columns:
            if me == BASE or BASE not in pv:
                continue
            p = pv[[BASE, me]].dropna()
            if len(p) >= 3:
                diff = p[me] - p[BASE]
                tests.append(dict(scenario=sc, metric=m, method=me, n=len(p),
                                  mean_diff_vs_static=round(diff.mean(), 4),
                                  p_value=round(stats.ttest_rel(p[me], p[BASE]).pvalue, 4)))
pd.DataFrame(tests).to_csv(TAB / "paired_tests.csv", index=False)

# ---- figures
for sc, g in df.groupby("scenario"):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for me in [m for m in ORDER if m in set(g.method)]:
        series = []
        for d in g[g.method == me].dir:
            t = pd.read_csv(d + "/telemetry.csv")
            t = t[t.t >= 20]
            series.append(t.groupby(t.t.astype(int)).utilization.max())
        s = pd.concat(series, axis=1).mean(axis=1).rolling(5, min_periods=1).mean()
        ax.plot(s.index, s.values, label=me)
    ax.set(xlabel="time (s)", ylabel="max link utilisation", title="MLU over time: " + sc, ylim=(0, 1.05))
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / ("mlu_time_%s.png" % sc), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for me in [m for m in ORDER if m in set(g.method)]:
        rtt = pd.concat([pd.read_csv(d + "/probes.csv").query("t>=20").rtt_ms for d in g[g.method == me].dir])
        xs = np.sort(rtt.values)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=me)
    ax.set(xlabel="probe RTT (ms)", ylabel="CDF", title="RTT CDF: " + sc)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / ("rtt_cdf_%s.png" % sc), dpi=200)
    plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, m in zip(axs, ["mean_mlu", "p95_mlu", "jfi", "rtt_ms_mean"]):
    scs = sorted(df.scenario.unique())
    meths = [x for x in ORDER if x in set(df.method)]
    for j, me in enumerate(meths):
        mu = [df[(df.scenario == s) & (df.method == me)][m].mean() for s in scs]
        er = [ci95(df[(df.scenario == s) & (df.method == me)][m]) for s in scs]
        ax.bar(np.arange(len(scs)) + j * 0.8 / len(meths), mu, 0.8 / len(meths), yerr=er, capsize=2, label=me)
    ax.set_xticks(np.arange(len(scs)) + 0.4)
    ax.set_xticklabels(scs, rotation=15, fontsize=7)
    ax.set_title(m)
axs[0].legend(fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "bars_summary.png", dpi=200)
print("wrote", TAB, FIG)