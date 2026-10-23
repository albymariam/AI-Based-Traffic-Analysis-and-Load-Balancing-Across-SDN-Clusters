"""Build the pin/reroute/unpin event list for the ONOS replay from a finished Mininet run.
  .venv/bin/python scripts/make_onos_events.py results/runs/<run_dir> data/onos_m5/events_s1.csv
Uses flows.csv (elephants: t_launch, wall_s, dport, path) and reroutes.csv (t, flow_id, new_path)."""
import sys
from pathlib import Path

import pandas as pd

run, out = Path(sys.argv[1]), Path(sys.argv[2])
fl = pd.read_csv(run / "flows.csv")
fl = fl[(fl["kind"] == "elephant") & fl["path"].notna() & (fl["path"] != "")]
rr = pd.read_csv(run / "reroutes.csv") if (run / "reroutes.csv").exists() else pd.DataFrame(columns=["t", "flow_id", "new_path"])
rows = []
for _, f in fl.iterrows():
    protos = "6" if f["proto"] == "tcp" else "6-17"
    base = dict(flow_id=int(f["flow_id"]), src=int(f["src"]), dst=int(f["dst"]), dport=int(f["dport"]), protos=protos)
    t0 = float(f["t_launch"])
    t1 = t0 + float(f["wall_s"]) if pd.notna(f.get("wall_s")) else t0 + float(f["dur_s"]) + 5
    rows.append(dict(base, t=t0, op="pin", path=str(f["path"]).split(">")[0]))
    for _, r in rr[rr["flow_id"] == f["flow_id"]].iterrows():
        if t0 < float(r["t"]) < t1:
            rows.append(dict(base, t=float(r["t"]), op="reroute", path=str(r["new_path"])))
    rows.append(dict(base, t=t1, op="unpin", path=""))
ev = pd.DataFrame(rows).sort_values("t")
out.parent.mkdir(parents=True, exist_ok=True)
ev[["t", "op", "flow_id", "src", "dst", "dport", "protos", "path"]].to_csv(out, index=False)
print("%d flows, %d events (%d pin, %d reroute, %d unpin) over %.0f s -> %s"
      % (ev["flow_id"].nunique(), len(ev), (ev.op == "pin").sum(), (ev.op == "reroute").sum(), (ev.op == "unpin").sum(),
         ev["t"].max() - ev["t"].min(), out))
