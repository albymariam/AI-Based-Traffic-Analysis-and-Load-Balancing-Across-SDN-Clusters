import pytest

from controller.demand_estimator import FlowRateEstimator

MBPS = 1e6 / 8     # bytes per second at 1 Mbps


def feed(est, key, rate_mbps, t_from, t_to, dt=0.5, start_bytes=0.0, dur0=0.0):
    b, d = start_bytes, dur0
    t = t_from
    while t < t_to - 1e-9:
        t += dt; d += dt; b += rate_mbps * MBPS * dt
        est.update({key: (b, d)}, now=t)
    return b, d, t


def test_constant_rate_is_recovered():
    est, k = FlowRateEstimator(), ("a", 1)
    _, _, t = feed(est, k, 2.5, 0, 4)
    assert est.rate(k, t) == pytest.approx(2.5, rel=1e-6)


def test_no_estimate_before_min_span():
    est, k = FlowRateEstimator(min_span_s=1.0), ("a", 1)
    _, _, t = feed(est, k, 3.0, 0, 0.5)
    assert est.rate(k, t) is None


def test_rate_change_is_tracked_within_window():
    est, k = FlowRateEstimator(window_s=3.0), ("a", 1)
    b, d, t = feed(est, k, 1.0, 0, 6)
    b, d, t = feed(est, k, 4.0, t, t + 4, start_bytes=b, dur0=d)
    assert est.rate(k, t) == pytest.approx(4.0, rel=0.05)


def test_counter_reset_holds_last_estimate_then_recovers():
    est, k = FlowRateEstimator(hold_s=5.0), ("a", 1)
    _, _, t = feed(est, k, 2.0, 0, 3)
    est.update({k: (10.0, 0.5)}, now=t + 0.5)                 # rule re-installed -> counters restart
    assert est.rate(k, t + 0.5) == pytest.approx(2.0, rel=1e-6)     # held
    feed(est, k, 3.0, t + 0.5, t + 3, start_bytes=10.0, dur0=0.5)
    assert est.rate(k, t + 3) == pytest.approx(3.0, rel=0.05)


def test_hold_expires_and_idle_flow_is_forgotten():
    est, k = FlowRateEstimator(hold_s=5.0, window_s=3.0), ("a", 1)
    _, _, t = feed(est, k, 2.0, 0, 3)
    assert est.rate(k, t + 6) is None
    est.update({}, now=t + 20)
    assert k not in est.hist


def test_duplicate_poll_does_not_change_estimate():
    est, k = FlowRateEstimator(), ("a", 1)
    b, d, t = feed(est, k, 2.0, 0, 3)
    est.update({k: (b, d)}, now=t + 0.1)
    assert est.rate(k, t + 0.1) == pytest.approx(2.0, rel=1e-6)
