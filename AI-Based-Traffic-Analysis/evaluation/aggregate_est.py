"""Demand from flow counters vs configured-rate oracle (faculty fix #1).
   T11_TOPOLOGY=abilene12 PYTHONPATH=. .venv/bin/python evaluation/aggregate_est.py
Writes results/tables/est_summary.csv, est_paired.csv, est_accuracy.csv.
Paired over seeds (same traffic): est vs static, est vs oracle, and the fair forecaster test (neural_est vs persistence_est)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common.config import ROOT

RUNS, TAB = ROOT / "results/runs", ROOT / "results/tables"
SCEN, STATIC = "abilene_replay", "static_shortest"
ARCH = ["transformer", "tcn", "persistence"]
M = ["mean_mlu", "p95_mlu", "peak_mlu", "jfi", "rtt_ms_mean", "reroutes_total", "plans_applied"]
WARM = 20.0


def ci95(x):
    x = pd.Series(x).dropna()
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def holm(pv):
    o = np.argsort(pv); adj, run = np.empty(len(pv)), 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(pv) - rank) * pv[i])); adj[i] = run
    return adj


rows = []
for d in sorted(RUNS.glob("*/")):
    f = d / "summary.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r["seed"] < 100 and r["scenario"] == SCEN:
            r["dir"] = str(d); rows.append(r)
df = pd.DataFrame(rows).sort_values("dir").drop_duplicates(["method", "seed"], keep="last")
names = [STATIC] + ["%s_minmax_lp" % a for a in ARCH] + ["%s_minmax_lp_est" % a for a in ARCH]
df = df[df.method.isin(names)]
M = [m for m in M if m in df and df[m].notna().any()]

out = []
for me in names:
    g = df[df.method == me]
    if len(g):
        out.append({"method": me, "n_seeds": len(g), **{m: "%.4f ± %.4f" % (g[m].mean(), ci95(g[m])) for m in M}})
pd.DataFrame(out).to_csv(TAB / "est_summary.csv", index=False)

pairs = [("%s_minmax_lp_est" % a, STATIC) for a in ARCH]
pairs += [("%s_minmax_lp_est" % a, "%s_minmax_lp" % a) for a in ARCH]                   # does removing the oracle change anything?
pairs += [("transformer_minmax_lp_est", "persistence_minmax_lp_est"), ("tcn_minmax_lp_est", "persistence_minmax_lp_est")]
tests = []
for m in M:
    pv = df.pivot_table(index="seed", columns="method", values=m)
    for a, b in pairs:
        if a in pv and b in pv:
            q = pv[[a, b]].dropna()
            if len(q) >= 3:
                dd = q[a] - q[b]
                tests.append(dict(metric=m, method=a, baseline=b, n=len(q), mean_diff=dd.mean(), ci95=ci95(dd),
                                  pct=100 * dd.mean() / q[b].mean() if q[b].mean() else np.nan,
                                  dz=dd.mean() / dd.std(ddof=1) if dd.std(ddof=1) > 0 else np.nan,
                                  p_value=stats.ttest_rel(q[a], q[b]).pvalue))
T = pd.DataFrame(tests)
if len(T):
    T["p_holm"] = np.nan
    for _, idx in T.groupby(["metric", "baseline"]).groups.items():
        T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_value"].values)
    T.round(4).to_csv(TAB / "est_paired.csv", index=False)

# estimator accuracy: |est - configured| / configured over flows that have an estimate; share of flow-ticks without one
acc = []
for _, r in df[df.method.str.endswith("_est")].iterrows():
    f = Path(r.dir) / "demand_est.csv"
    if not f.exists():
        continue
    e = pd.read_csv(f).query("t >= @WARM")
    have = e.dropna(subset=["rate_est_mbps"])
    err = (have.rate_est_mbps - have.rate_true_mbps) / have.rate_true_mbps
    acc.append(dict(method=r.method, seed=r.seed, flow_ticks=len(e), no_estimate_pct=100 * (1 - len(have) / max(len(e), 1)),
                    mape_pct=100 * err.abs().mean(), bias_pct=100 * err.mean(), p95_abs_err_pct=100 * err.abs().quantile(0.95)))
A = pd.DataFrame(acc)
if len(A):
    A.round(3).to_csv(TAB / "est_accuracy_per_run.csv", index=False)
    A.groupby("method").mean(numeric_only=True).round(3).to_csv(TAB / "est_accuracy.csv")
    print(A.groupby("method").mean(numeric_only=True).round(2).to_string())
print(pd.DataFrame(out).to_string(index=False))
if len(T):
    print(T.round(4).to_string(index=False))
