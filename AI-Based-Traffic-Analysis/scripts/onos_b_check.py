"""Verify the 3-node ONOS cluster (condition B) before any experiment.
  PYTHONPATH=. ~/venvs/ml/bin/python scripts/onos_b_check.py --nodes localhost:8182,localhost:8183,localhost:8184 --expect-devices 4 --flow-test
Checks: membership agreement, required apps, device agreement, and (with --flow-test) cross-node flow-rule visibility
(the measured state-propagation delay between nodes that condition C will later perturb)."""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TEST_IP = "203.0.113.77/32"      # TEST-NET-3: never matches real traffic
TIMEOUT = 5                      # REST read timeout (s); --timeout raises it for slow (high-delay) clusters


def get(node, path, auth, timeout=None):
    r = requests.get("http://%s/onos/v1%s" % (node, path), auth=auth, timeout=timeout or TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError("GET %s%s -> %s %s" % (node, path, r.status_code, r.text[:200]))
    return r.json()


def has_test_flow(js):
    for f in js.get("flows", []):
        for c in f.get("selector", {}).get("criteria", []):
            if c.get("type") == "IPV4_DST" and c.get("ip") == TEST_IP:
                return f
    return None


def poll(node, dev, auth, want, tmax=10.0):
    """seconds until the test flow is visible (want=True) / gone (want=False) on node; None on timeout"""
    t0 = time.time()
    while time.time() - t0 < tmax:
        f = has_test_flow(get(node, "/flows/%s" % dev, auth))
        if bool(f) == want:
            return time.time() - t0, f
        time.sleep(0.02)
    return None, None


def poll_abs(node, dev, auth, want, tmax=15.0):
    """perf_counter() time at which the test flow is first seen (want=True) / gone (want=False) on node; None on timeout"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < tmax:
        f = has_test_flow(get(node, "/flows/%s" % dev, auth))
        t = time.perf_counter()
        if bool(f) == want:
            return t, f
        time.sleep(0.005)
    return None, None


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(p * (len(x) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--user", default="onos")
    ap.add_argument("--password", default="rocks")
    ap.add_argument("--expect-members", type=int, default=3)
    ap.add_argument("--expect-devices", type=int, default=0)
    ap.add_argument("--require", default="org.onosproject.openflow")
    ap.add_argument("--flow-test", action="store_true")
    ap.add_argument("--json", default=None, help="write the propagation statistics to this file")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the flow test N times and report median / p95")
    ap.add_argument("--timeout", type=float, default=5, help="REST read timeout in seconds")
    a = ap.parse_args()
    global TIMEOUT
    TIMEOUT = a.timeout
    nodes, auth, ok = a.nodes.split(","), (a.user, a.password), True

    def fail(msg):
        nonlocal ok
        ok = False
        print("  FAIL", msg)

    print("== 1. membership (each node's own view) ==")
    views = {}
    for n in nodes:
        try:
            cl = get(n, "/cluster", auth)["nodes"]
            views[n] = {x["id"]: x.get("status", "?") for x in cl}
            print("  %-18s %s" % (n, views[n]))
        except Exception as e:
            fail("%s: %s" % (n, e))
    if views:
        sets = [frozenset(v) for v in views.values()]
        if len(set(sets)) != 1 or len(sets[0]) != a.expect_members:
            fail("membership differs between nodes or != %d members" % a.expect_members)
        bad = [(n, i, s) for n, v in views.items() for i, s in v.items() if s not in ("READY", "ACTIVE")]
        if bad:
            fail("members not READY/ACTIVE: %s" % bad)

    print("== 2. applications ==")
    for n in nodes:
        try:
            apps = {x["name"]: x["state"] for x in get(n, "/applications", auth)["applications"]}
            for r in a.require.split(","):
                st = apps.get(r)
                print("  %-18s %-34s %s" % (n, r, st))
                if st != "ACTIVE":
                    fail("%s: %s is %s" % (n, r, st))
            if apps.get("org.onosproject.fwd") == "ACTIVE":
                fail("%s: org.onosproject.fwd (reactive forwarding) is ACTIVE; it will fight our flow rules" % n)
        except Exception as e:
            fail("%s: %s" % (n, e))

    print("== 3. devices (every node should see the same set) ==")
    dev_sets = {}
    for n in nodes:
        try:
            ds = get(n, "/devices", auth)["devices"]
            dev_sets[n] = {d["id"] for d in ds if d.get("available")}
            print("  %-18s %d available devices" % (n, len(dev_sets[n])))
        except Exception as e:
            fail("%s: %s" % (n, e))
    if dev_sets:
        vals = list(dev_sets.values())
        if any(v != vals[0] for v in vals):
            fail("nodes disagree on the device set")
        if len(vals[0]) < a.expect_devices:
            fail("fewer devices than expected (%d): attach Mininet to all controllers first" % a.expect_devices)

    if a.flow_test and dev_sets and any(dev_sets.values()):
        print("== 4. cross-node flow-rule visibility (state propagation, %d test%s) ==" % (a.repeat, "" if a.repeat == 1 else "s"))
        dev = sorted(next(iter(dev_sets.values())))[0]
        writer, readers = nodes[0], nodes[1:]
        body = {"priority": 100, "timeout": 0, "isPermanent": True, "deviceId": dev,
                "treatment": {"instructions": [{"type": "OUTPUT", "port": "1"}]},
                "selector": {"criteria": [{"type": "ETH_TYPE", "ethType": "0x0800"},
                                           {"type": "IPV4_DST", "ip": TEST_IP}]}}
        # All nodes are polled concurrently from the moment of the POST, so each time is "seconds since the POST".
        # lag = reader time - writer time cancels the REST round-trip that every reading shares.
        acc = {k: {n: [] for n in nodes} for k in ("vis", "rem")}

        def watch(want):
            with ThreadPoolExecutor(len(nodes)) as ex:
                futs = {n: ex.submit(poll_abs, n, dev, auth, want) for n in nodes}
                return {n: f.result() for n, f in futs.items()}

        for i in range(a.repeat):
            t_post = time.perf_counter()
            r = requests.post("http://%s/onos/v1/flows/%s?appId=org.onosproject.rest" % (writer, dev),
                              json=body, auth=auth, timeout=5)
            if r.status_code not in (200, 201):
                fail("flow POST failed: %s" % r.text[:300])
                break
            seen = watch(True)
            flow = seen[writer][1]
            for n in nodes:
                if seen[n][0] is None:
                    fail("flow never became visible on %s" % n)
                else:
                    acc["vis"][n].append(seen[n][0] - t_post)
            if not flow:
                fail("writer never listed the flow; manual cleanup may be needed (match IPV4_DST %s)" % TEST_IP)
                break
            t_del = time.perf_counter()
            requests.delete("http://%s/onos/v1/flows/%s/%s" % (writer, dev, flow["id"]), auth=auth, timeout=5)
            gone = watch(False)
            for n in nodes:
                if gone[n][0] is None:
                    fail("test flow not removed on %s; delete it manually" % n)
                else:
                    acc["rem"][n].append(gone[n][0] - t_del)
            time.sleep(0.2)
        stats_out = {}
        for k, label in (("vis", "visible after POST"), ("rem", "gone after DELETE")):
            wr = acc[k][writer]
            for n in nodes:
                v = acc[k][n]
                if not v:
                    continue
                lag = [x - w for x, w in zip(v, wr)] if n != writer else None
                stats_out["%s|%s" % (k, n)] = dict(n=len(v), median=statistics.median(v), p95=q(v, 0.95), max=max(v),
                                                   lag_median=None if lag is None else statistics.median(lag),
                                                   lag_p95=None if lag is None else q(lag, 0.95))
                print("  %-18s %-18s median %.3f s  p95 %.3f s  max %.3f s%s" % (
                    n + (" (writer)" if n == writer else ""), label, statistics.median(v), q(v, 0.95), max(v),
                    "" if lag is None else "   | lag vs writer: median %.3f s  p95 %.3f s" % (statistics.median(lag), q(lag, 0.95))))

    if a.json and a.flow_test:
        with open(a.json, "w") as fh:
            json.dump(dict(ok=ok, writer=nodes[0], stats=locals().get("stats_out", {})), fh, indent=1)
    print("\nRESULT:", "READY" if ok else "NOT READY (see FAIL lines)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()