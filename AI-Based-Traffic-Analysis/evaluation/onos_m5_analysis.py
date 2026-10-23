"""M5 stage 1 table: one row per ONOS condition (A single node, B cluster, C<ms> cluster + delay).
  .venv/bin/python evaluation/onos_m5_analysis.py          -> results/tables/onos_m5.csv
Sync overhead = bytes seen by the three Atomix (storage) nodes on eth0, both directions (inter-Atomix bytes are counted at
both ends), idle rate subtracted, divided by the replayed operations. Condition A has no cluster and no storage nodes: 0."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "onos_m5"


def rj(p):
    return json.loads(p.read_text()) if p.exists() else None


def total(snap):
    return sum(v["rx"] + v["tx"] for k, v in snap.items() if k != "_t")


def order(c):
    return (0, 0) if c == "A" else (1, 0) if c == "B" else (2, int(re.sub(r"\D", "", c) or 0))


rows = []
for d in sorted((p for p in RES.glob("*") if p.is_dir()), key=lambda p: order(p.name)):
    rs, pr = rj(d / "replay_summary.json"), rj(d / "propagation.json")
    if not rs:
        continue
    row = dict(condition=d.name, delay_ms=0 if d.name in ("A", "B") else int(re.sub(r"\D", "", d.name)),
               ok_pct=rs["ok_pct"], replay_lag_p95_s=rs["replay_lag_p95_s"], ops=rs["n_ops"], rest_per_s=rs["rest_requests_per_s"])
    for op in ("pin", "reroute", "unpin"):
        row["%s_rest_ms_median" % op] = rs[op]["rest_ms_median"]
        row["%s_rest_ms_p95" % op] = rs[op]["rest_ms_p95"]
    row["pin_added_ms_median"], row["pin_added_ms_p95"] = rs["pin_added_ms"]["median"], rs["pin_added_ms"]["p95"]
    if pr:
        st = pr["stats"]
        w = pr["writer"]
        row["write_visible_ms_median"] = 1000 * st["vis|" + w]["median"]
        lags = [v["lag_p95"] for k, v in st.items() if k.startswith("vis|") and v["lag_p95"] is not None]
        row["peer_lag_p95_ms"] = 1000 * max(lags) if lags else None
    b, a, i0, i1, i2 = (rj(d / ("net_%s.json" % n)) for n in ("before", "after", "idle0", "idle1", "idle2"))
    if b and a and i0 and i1:
        idle = (total(i1) - total(i0)) / max(1e-9, i1["_t"] - i0["_t"])
        if i2:                                   # idle traffic drifts: use the mean of the windows before and after the replay
            idle = 0.5 * (idle + (total(i2) - total(a)) / max(1e-9, i2["_t"] - a["_t"]))
        run_s = a["_t"] - b["_t"]
        extra = total(a) - total(b) - idle * run_s
        row["sync_idle_kBps"] = idle / 1e3
        row["sync_replay_kBps"] = (total(a) - total(b)) / run_s / 1e3
        row["sync_bytes_per_op"] = max(0.0, extra) / max(1, rs["n_ops"])
    elif d.name == "A":
        row["sync_bytes_per_op"] = 0.0
    rows.append(row)
df = pd.DataFrame(rows)
out = ROOT / "results" / "tables" / "onos_m5.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.round(1).T.to_string() if len(df) else "no results in %s" % RES)
print("-> %s" % out)
