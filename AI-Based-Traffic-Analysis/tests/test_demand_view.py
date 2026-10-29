"""The counters demand source must (a) use only counter data, (b) leave unmeasured flows out, (c) log the true rate only."""
import csv

import pytest

pytest.importorskip("torch")
from controller.demand_estimator import FlowRateEstimator
from controller.lp_policy import LPReroute

MBPS = 1e6 / 8


class FakeAPI:
    def __init__(self):
        self.t, self.b, self.rate = 0.0, 0.0, 2.0

    def step(self, dt=0.5):
        self.t += dt
        self.b += self.rate * MBPS * dt

    def flowstats(self):
        return dict(flows=[dict(src=1, dst=2, dport=5201, proto=17, bytes=self.b, dur_s=self.t, age_s=0.1)])


def make(tmp_path, api):
    p = object.__new__(LPReroute)            # skip checkpoint loading: only the demand view is under test
    p.api, p.est, p.min_est = api, FlowRateEstimator(), 0.01
    p.fe = open(tmp_path / "demand_est.csv", "w", newline="")
    p.we = csv.writer(p.fe)
    return p


def els(rate_true):
    return [dict(id=7, src=1, dst=2, port=5201, protos=[6, 17], path=[1, 2], rate=rate_true),
            dict(id=8, src=3, dst=4, port=5202, protos=[6, 17], path=[3, 4], rate=9.9)]   # no counters for this one


def test_estimate_replaces_configured_rate_and_unmeasured_flow_is_dropped(tmp_path):
    api = FakeAPI()
    p = make(tmp_path, api)
    import time
    out = []
    for i in range(8):
        api.step()
        out = p._demand_view(els(rate_true=123.0), now=float(i))      # configured rate is deliberately WRONG
        time.sleep(0.001)
    assert [e["id"] for e in out] == [7]
    assert out[0]["rate"] == pytest.approx(2.0, rel=1e-3)             # taken from counters, not from 123.0
    p.fe.close()
    rows = list(csv.reader(open(tmp_path / "demand_est.csv")))
    assert rows[-2][4] == "123.0" and rows[-2][5] != ""                 # true rate is logged next to the estimate
    assert rows[-1][4] == "9.9" and rows[-1][5] == ""                   # flow without counters has no estimate
