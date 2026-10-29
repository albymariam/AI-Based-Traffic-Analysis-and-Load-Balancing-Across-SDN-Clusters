"""Step 0 check, no experiment needed: with Ryu running and ANY flows pinned (e.g. during a run), print the per-flow rates the
estimator would use. Compare with the configured rates in results/abilene/schedules/seed_<n>.csv.
   PYTHONPATH=. .venv/bin/python scripts/check_flowstats.py [seconds]"""
import sys
import time

from controller.demand_estimator import FlowRateEstimator
from controller.rest_client import TEClient

api, est = TEClient(), FlowRateEstimator()
t_end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 15)
while time.time() < t_end:
    rows = api.flowstats()["flows"]
    now = time.time()
    est.update({(r["src"], r["dst"], r["dport"], r["proto"]): (r["bytes"], r["dur_s"]) for r in rows}, now)
    print("t=%.1f  %d rules" % (now % 1000, len(rows)),
          {"%d>%d:%s/%d" % (k[0], k[1], k[2], k[3]): round(est.rate(k, now), 3) for k in sorted(est.good) if est.rate(k, now) is not None})
    time.sleep(1.0)
