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
           "flowmods_per_s", "reroutes_total", "reroute_decision_ms_mean", "flows_failed_pct", "lp_solve_ms_mean", "plans_applied",
           "peak_mlu", "mean_util", "congested_link_pct", "carried_over_offered", "elephant_goodput_mbps_mean", "infer_ms_mean", "fct_mean_s"]
ORDER = ["static_shortest", "static_tracked", "ecmp", "round_robin", "lstm_hysteresis",
         "transformer_minmax_lp", "tcn_minmax_lp", "persistence_minmax_lp"]
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
df = df[df.method.isin(ORDER)]                    # staleness (dist_lp_s*) runs are analysed by staleness_analysis.py, not here
mets = [m for m in METRICS if m in df]

# ---- table: mean ± CI
out = []
for (sc, me), g in df.groupby(["scenario", "method"]):
    row = {"scenario": sc, "method": me, "n_seeds": len(g)}
    for m in mets:
        row[m] = "%.3f ± %.3f" % (g[m].mean(), ci95(g[m])) if g[m].notna().any() else ""
    out.append(row)
pd.DataFrame(out).to_csv(TAB / "summary_mean_ci.csv", index=False)

# ---- paired tests (same seed = same traffic): every method vs static_shortest, flow-mods vs static_tracked, and head-to-head
TEST_M = ["mean_mlu", "p95_mlu", "peak_mlu", "jfi", "rtt_ms_mean", "rtt_ms_p95", "rtt_jitter_ms", "udp_loss_pct",
          "udp_jitter_ms_mean", "elephant_goodput_mbps_mean", "flowmods_per_s", "fct_mean_s", "fct_p95_s"]
HEAD2HEAD = [("transformer_minmax_lp", "tcn_minmax_lp"), ("transformer_minmax_lp", "persistence_minmax_lp"),
             ("tcn_minmax_lp", "persistence_minmax_lp"), ("transformer_minmax_lp", "lstm_hysteresis"),
             ("persistence_minmax_lp", "lstm_hysteresis")]


def holm(pv):
    """Holm step-down adjusted p-values (family = the tests of one metric against one baseline)."""
    order = np.argsort(pv)
    adj, run = np.empty(len(pv)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pv) - rank) * pv[i]))
        adj[i] = run
    return adj


tests = []
for sc, g in df.groupby("scenario"):
    for m in [x for x in TEST_M if x in g]:
        pv = g.pivot_table(index="seed", columns="method", values=m)
        pairs = [(me, BASE) for me in pv.columns if me != BASE]
        if "static_tracked" in pv:
            pairs += [(me, "static_tracked") for me in pv.columns if me not in (BASE, "static_tracked")] if m == "flowmods_per_s" else []
        pairs += [(x, y) for x, y in HEAD2HEAD if x in pv and y in pv]
        for me, base in pairs:
            if base not in pv:
                continue
            q = pv[[me, base]].dropna()
            if len(q) < 3 or q[base].mean() == 0 and q[me].mean() == 0:
                continue
            diff = q[me] - q[base]
            sd = diff.std(ddof=1)
            t = stats.ttest_rel(q[me], q[base])
            hw = stats.t.ppf(0.975, len(q) - 1) * sd / np.sqrt(len(q))
            tests.append(dict(scenario=sc, metric=m, method=me, baseline=base, n=len(q),
                              mean_diff=diff.mean(), ci95_lo=diff.mean() - hw, ci95_hi=diff.mean() + hw,
                              pct_change=100 * diff.mean() / q[base].mean() if q[base].mean() else np.nan,
                              cohens_dz=diff.mean() / sd if sd > 0 else np.nan, p_value=t.pvalue))
T = pd.DataFrame(tests)
if len(T):
    T["p_holm"] = np.nan
    for _, idx in T.groupby(["scenario", "metric", "baseline"]).groups.items():
        T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_value"].values)
    T = T.round({"mean_diff": 4, "ci95_lo": 4, "ci95_hi": 4, "pct_change": 2, "cohens_dz": 2, "p_value": 4, "p_holm": 4})
tests = T.to_dict("records")
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