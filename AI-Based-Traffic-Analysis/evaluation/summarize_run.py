"""Compute headline metrics for one run directory from its CSVs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _f(x):
    x = float(x)
    return None if np.isnan(x) else round(x, 5)


def _jfi(x):
    s2 = (x ** 2).sum()
    return (x.sum() ** 2) / (len(x) * s2) if s2 > 0 else np.nan


def summarize(run_dir):
    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    tcfg = yaml.safe_load((run / "traffic.yaml").read_text())
    warm = tcfg["warmup_s"]
    out = dict(method=meta["method"], scenario=meta["scenario"], seed=meta["seed"], git=meta["git_hash"])

    tele = pd.read_csv(run / "telemetry.csv")
    tele = tele[tele.t >= warm]
    g = tele.groupby("timestep")["utilization"]
    mlu = g.max()
    out.update(mean_util=_f(g.mean().mean()), mean_mlu=_f(mlu.mean()), p95_mlu=_f(mlu.quantile(0.95)),
               peak_mlu=_f(mlu.max()), jfi=_f(g.apply(_jfi).mean()),
               queue_pkts_mean=_f(tele.queue_length.mean()), link_drops_total=int(tele.packet_loss.sum()),
               link_delay_ms_mean=_f(tele.delay_ms.mean()),
               congested_link_pct=_f(100 * (tele.utilization_raw >= 0.95).mean()),
               congestion_events=int((mlu >= 0.95).astype(int).diff().fillna(0).clip(lower=0).sum()))

    fl = pd.read_csv(run / "flows.csv")
    fl = fl[fl.t_launch >= warm]
    out["flows_total"] = int(len(fl))
    out["flows_failed_pct"] = _f(100 * (fl.status != "ok").mean()) if len(fl) else None
    ok = fl[fl.status == "ok"]
    span = max(1.0, tele.t.max() - warm)
    out["throughput_mbps_aggregate"] = _f(ok.bytes.sum() * 8 / span / 1e6)
    offered = (fl.rate_mbps * fl.dur_s).sum() / span          # elephants only (mice have no rate)
    out["offered_mbps"] = _f(offered)
    out["carried_over_offered"] = _f(ok[ok.kind == "elephant"].bytes.sum() * 8 / span / 1e6 / offered) if offered > 0 else None
    el = ok[ok.kind == "elephant"]
    mice = ok[ok.kind == "mice"]
    out["elephant_goodput_mbps_mean"] = _f(el.bps.mean() / 1e6) if len(el) else None
    if len(mice):
        out.update(fct_mean_s=_f(mice.xfer_s.mean()), fct_p50_s=_f(mice.xfer_s.median()),
                   fct_p95_s=_f(mice.xfer_s.quantile(0.95)))
    udp = ok[ok.proto == "udp"]
    if len(udp) and udp.packets.sum() > 0:
        out["udp_loss_pct"] = _f(100 * udp.lost_packets.sum() / udp.packets.sum())
        out["udp_jitter_ms_mean"] = _f(udp.jitter_ms.mean())
    if "pin_ms" in fl and fl.pin_ms.notna().any():
        out["pin_rest_ms_mean"] = _f(fl.pin_ms.mean())
        out["pin_rest_ms_p95"] = _f(fl.pin_ms.quantile(0.95))

    pr = pd.read_csv(run / "probes.csv")
    pr = pr[pr.t >= warm]
    if len(pr):
        out.update(rtt_ms_mean=_f(pr.rtt_ms.mean()), rtt_ms_p95=_f(pr.rtt_ms.quantile(0.95)))
        out["rtt_jitter_ms"] = _f(np.mean([np.abs(np.diff(x.rtt_ms.values)).mean()
                                           for _, x in pr.groupby(["src", "dst"]) if len(x) > 2]))

    ctrl = pd.read_csv(run / "ctrl.csv")
    c = ctrl[ctrl.t >= warm]
    if len(c) > 1:
        out["flowmods_per_s"] = _f((c.flowmods_total.iloc[-1] - c.flowmods_total.iloc[0]) /
                                   max(1.0, c.t.iloc[-1] - c.t.iloc[0]))
        for col, name in (("cpu_pct", "ctrl_cpu_pct_mean"), ("rss_mb", "ctrl_rss_mb_max")):
            if col in c:                                       # optional: only if the logger records them
                out[name] = _f(c[col].max() if col == "rss_mb" else c[col].mean())
    rr = run / "reroutes.csv"
    if rr.exists():
        r = pd.read_csv(rr)
        r = r[r.t >= warm]
        out["reroutes_total"] = int(len(r))
        if len(r):
            out.update(reroute_decision_ms_mean=_f(r.decision_ms.mean()),
                       reroute_decision_ms_p95=_f(r.decision_ms.quantile(0.95)),
                       infer_ms_mean=_f(r.infer_ms.mean()))

    pl = run / "plans.csv"
    if pl.exists():
        p = pd.read_csv(pl)
        p = p[p.t >= warm]
        if len(p):
            out.update(lp_solves=int(len(p)), plans_applied=int(p.applied.sum()),
                       lp_solve_ms_mean=_f(p.lp_ms.mean()), lp_solve_ms_p95=_f(p.lp_ms.quantile(0.95)),
                       predicted_mlu_gain_mean=_f((p.theta_cur - p.theta_plan).mean()))
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))