"""Per-run gap between what the LP promised and the max link load the network really had.

Uses files every run already writes (plans.csv, telemetry.csv, meta.json, traffic.yaml),
so no controller change and no new runs are needed.

Usage (from the repo folder AI-Based-Traffic-Analysis):
    python scripts/check_realised_gap.py results/runs/*transformer_minmax_lp*abilene_replay_s*  [--out gap.csv]

Per plan tick it compares:
  frac     = LP value z (fractional split over the K candidate paths)
  plan     = same plan after rounding to one path per flow
  promised = plan if the plan was applied, else theta_cur (network stays as it was)
  real     = max over links of raw utilisation, averaged over the next H seconds,
             starting LAG seconds after the tick (LAG covers rule-install time)

Notes: at most max_moves_per_tick (6) flows are moved per tick, so "promised" can be a little
optimistic when a plan needs more moves. Only one folder per (method, seed) is used (the latest).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--H", type=float, default=2.0, help="averaging window, s (= LP period)")
p.add_argument("--lags", type=float, nargs="+", default=[0.0, 6.0], help="delay before the window, s")
p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)), help="seeds to keep (default 1-10)")
p.add_argument("--out", default=None)
a = p.parse_args()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# one folder per (method, seed): keep the latest, so reruns / pilots / training seeds do not double count
best = {}
for d in sorted(map(Path, a.runs)):                       # names start with a timestamp, so sorted = oldest first
    try:
        m = json.loads((d / "meta.json").read_text())
    except Exception as ex:
        print("skip", d.name, repr(ex), file=sys.stderr)
        continue
    if m.get("seed") in a.seeds:
        if (m["method"], m["seed"]) in best:
            print("note: %s replaces %s" % (d.name, best[(m["method"], m["seed"])].name), file=sys.stderr)
        best[(m["method"], m["seed"])] = d

rows = []
for d in best.values():
    try:
        meta = json.loads((d / "meta.json").read_text())
        warm = yaml.safe_load((d / "traffic.yaml").read_text())["warmup_s"]
        pl = pd.read_csv(d / "plans.csv", index_col=False)   # index_col=False: tolerate extra columns (dist_lp)
        te = pd.read_csv(d / "telemetry.csv")
    except Exception as ex:
        print("skip", d.name, repr(ex), file=sys.stderr)
        continue
    pl = pl[pl.t >= warm].copy()
    if pl.empty:
        continue
    # max raw utilisation over links at each telemetry step
    mlu = te.groupby("timestep").agg(t=("t", "first"), m=("utilization_raw", "max")).sort_values("t")
    tt, mm = mlu.t.to_numpy(), mlu.m.to_numpy()
    pl["promised"] = np.where(pl.applied == 1, pl.theta_plan, pl.theta_cur)
    r = dict(run=d.name, method=meta["method"], seed=meta["seed"], plans=len(pl),
             applied_pct=100 * pl.applied.mean(),
             cur=pl.theta_cur.mean(), frac=pl.theta_lp_frac.mean(), plan=pl.theta_plan.mean(),
             promised=pl.promised.mean(),
             rounding_gap_pct=100 * (pl.theta_plan.mean() / pl.theta_lp_frac.mean() - 1))
    for lag in a.lags:
        real = []
        for t in pl.t:
            # a telemetry row stamped ts is the average over (ts-1, ts], so the real-time window
            # [t+lag, t+lag+H] corresponds to rows with ts in (t+lag+1, t+lag+H+1]
            s = (tt > t + lag + 1.0) & (tt <= t + lag + a.H + 1.0)
            real.append(mm[s].mean() if s.any() else np.nan)
        real = np.array(real)
        ok = ~np.isnan(real)
        err = real[ok] - pl.promised.to_numpy()[ok]
        r[f"real_lag{lag:g}"] = real[ok].mean()
        r[f"real_minus_promised_lag{lag:g}"] = err.mean()
        r[f"mae_lag{lag:g}"] = np.abs(err).mean()
        r[f"corr_lag{lag:g}"] = np.corrcoef(real[ok], pl.promised.to_numpy()[ok])[0, 1] if ok.sum() > 2 else np.nan
    rows.append(r)

df = pd.DataFrame(rows).sort_values(["method", "seed"])
pd.set_option("display.width", 250, "display.max_columns", 40)
print(df.round(3).to_string(index=False))
print("\nmean over runs, per method:")
print(df.groupby("method").mean(numeric_only=True).drop(columns=["seed"]).round(3).T.to_string())
if a.out:
    df.to_csv(a.out, index=False)
    print("wrote", a.out)
