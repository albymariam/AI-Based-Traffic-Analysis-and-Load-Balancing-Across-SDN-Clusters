"""Write demo/data.js for the offline HTML demo from results/tables and (if present) the raw Abilene files.

Run from anywhere:  .venv/bin/python scripts/export_demo_data.py
Optional:           --raw path/to/abilene  --week 24
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TABLES = ROOT / "results" / "tables"


def mean_of(v):
    """'0.500 ± 0.157' -> 0.5"""
    try:
        return float(str(v).split("±")[0].strip())
    except (ValueError, TypeError):
        return None


def table(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def summary():
    d = table("summary_mean_ci.csv")
    if d is None:
        return []
    d = d[d["scenario"] == "abilene_replay"]
    return [dict(method=r["method"], mean=mean_of(r["mean_mlu"]), p95=mean_of(r["p95_mlu"]),
                 peak=mean_of(r.get("peak_mlu")), jfi=mean_of(r["jfi"]), rtt=mean_of(r["rtt_ms_mean"]))
            for _, r in d.iterrows()]


def prediction():
    d = table("prediction_error_m2.csv")
    if d is None:
        return []
    d = d[d["horizon"].astype(str) == "all"]
    return [dict(model=r["model"], mae=float(r["MAE"]), rmse=float(r["RMSE"]), smape=float(r["sMAPE_gt5"]))
            for _, r in d.iterrows()]


def staleness(name):
    d = table(name)
    if d is None:
        return []
    return [dict(setting=r["setting"], n=int(r["n_seeds"]), mean=mean_of(r["mean_mlu"]), p95=mean_of(r["p95_mlu"]))
            for _, r in d.iterrows()]


def estimated():
    d = table("est_paired_alby.csv")
    if d is None:
        return []
    d = d[(d["fam"] == "transformer") & (d["vs"] == "static") & d["metric"].isin(["mean_mlu", "p95_mlu", "peak_mlu"])]
    return [dict(metric=r["metric"], est=float(r["est"]), base=float(r["base"]), pct=float(r["pct"]),
                 holm=float(r["holm"])) for _, r in d.iterrows()]


def rounding():
    d = table("est_realised_gap.csv")
    if d is None:
        return None
    d = d[d["method"].str.contains("transformer")]
    g = d["rounding_gap_pct"]
    r0 = d["real_minus_promised_lag0"] / d["promised"] * 100
    r6 = d["real_minus_promised_lag6"] / d["promised"] * 100
    return dict(n=int(len(d)), rounding_mean=float(g.mean()), rounding_sd=float(g.std()),
                real0_mean=float(r0.mean()), real0_sd=float(r0.std()),
                real6_mean=float(r6.mean()), real6_sd=float(r6.std()))


def onos():
    d = table("onos_m5.csv")
    if d is None:
        return dict(rows=[], fit=None)
    rows = [dict(cond=r["condition"], delay=float(r["delay_ms"]), ok=float(r["ok_pct"]),
                 reroute_med=float(r["reroute_rest_ms_median"]), reroute_p95=float(r["reroute_rest_ms_p95"]),
                 write_vis=float(r["write_visible_ms_median"])) for _, r in d.iterrows()]
    fit = None
    cl = d[d["condition"] != "A"]  # 3-node cluster rows only
    if len(cl) >= 3:
        x, y = cl["delay_ms"].astype(float).values, cl["write_visible_ms_median"].astype(float).values
        s, i = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (s * x + i)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fit = dict(slope=float(s), intercept=float(i), r2=float(r2))
    return dict(rows=rows, fit=fit)


def bounds():
    d = table("abilene_te_bounds.csv")
    if d is None:
        return []
    keep = ["mcf_optimal", "kpath_3", "kpath_3_int", "kpath_6", "oblivious_opt", "shortest_path", "vlb"]
    d = d[d["policy"].isin(keep)]
    return [dict(policy=r["policy"], ratio=float(r["mean_ratio_to_opt"])) for _, r in d.iterrows()]


def abilene_series(raw_dir, week, top=30, steps=72, peak_mbps=120.0):
    """Total demand of the top-N OD pairs for the 6 h window with the biggest rise, scaled so its peak = peak_mbps."""
    try:
        from abilene.loaders import week_files, read_week, split_layout, detect_layout
        from abilene.preprocessing import to_mbps, od_columns
    except Exception as e:  # repo modules not importable
        print("  Abilene loaders not importable:", e)
        return None
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print("  raw Abilene folder not found:", raw_dir, "-> series skipped")
        return None
    files = {int("".join(c for c in p.name if c.isdigit())): p for p in week_files(raw_dir)}
    wk = week if week in files else max(files)
    raw = read_week(files[wk])
    layout, _ = detect_layout(raw)
    x = to_mbps(split_layout(raw, layout)[:, :, 0])[:, od_columns()]
    x = np.nan_to_num(x)
    keep = np.argsort(-x.sum(0))[:top]
    tot = x[:, keep].sum(1)
    best, best_score = 0, -1
    for s in range(0, len(tot) - steps):
        w = tot[s:s + steps]
        base = np.median(w[:20])
        score = w.max() / base if base > 0 else 0
        if score > best_score:
            best, best_score = s, score
    w = tot[best:best + steps]
    scale = peak_mbps / w.max()
    return dict(week=int(wk), start_step=int(best), step_seconds=300, top_pairs=top, scale=float(scale),
                peak_mbps=peak_mbps, values=[round(float(v), 2) for v in w * scale],
                note="Aggregate of the top-%d OD pairs, week %d, 6 h window with the largest rise, "
                     "scaled so the peak is %d Mbps." % (top, wk, peak_mbps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "abilene"))
    ap.add_argument("--week", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "demo" / "data.js"))
    a = ap.parse_args()

    data = dict(summary=summary(), prediction=prediction(), stale_known=staleness("staleness_summary.csv"),
                stale_est=staleness("staleness_summary_est.csv"), estimated=estimated(), rounding=rounding(),
                onos=onos(), bounds=bounds(), abilene=abilene_series(a.raw, a.week))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.DEMO_DATA = " + json.dumps(data, indent=1) + ";\n")
    print("wrote", out, "| abilene series:", "yes" if data["abilene"] else "NO (synthetic fallback in the demo)")
    print("rounding:", data["rounding"])
    print("onos fit:", data["onos"]["fit"])


if __name__ == "__main__":
    main()
