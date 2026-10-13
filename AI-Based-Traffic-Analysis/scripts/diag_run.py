"""usage: python scripts/diag_run.py results/runs/<run_dir>"""
import sys
import pandas as pd

run = sys.argv[1]
fl = pd.read_csv(f"{run}/flows.csv")
print("== flow status by kind/proto ==")
print(fl.groupby(["kind", "proto", "status"]).size())
if "error" in fl:
    print("\n== iperf errors ==")
    print(fl.error.value_counts().head())
print("\n== busiest destinations ==")
print(fl.groupby("dst").size().sort_values(ascending=False).head(5))
tele = pd.read_csv(f"{run}/telemetry.csv")
u = tele[tele.t >= 20].groupby("link_id").utilization.mean().sort_values(ascending=False)
print("\n== hottest links (mean util) ==")
print(u.head(8))
print(f"links with mean util > 0.05: {(u > 0.05).sum()} / {len(u)}")
if "stderr" in fl:
    print("\n== stderr ==")
    print(fl.stderr.value_counts().head())