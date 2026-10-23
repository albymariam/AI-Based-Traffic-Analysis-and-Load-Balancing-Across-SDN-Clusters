"""Verify the 3-node ONOS cluster (condition B) before any experiment.
  PYTHONPATH=. ~/venvs/ml/bin/python scripts/onos_b_check.py --nodes localhost:8182,localhost:8183,localhost:8184 --expect-devices 4 --flow-test
Checks: membership agreement, required apps, device agreement, and (with --flow-test) cross-node flow-rule visibility
(the measured state-propagation delay between nodes that condition C will later perturb)."""
import argparse
import sys
import time

import requests

TEST_IP = "203.0.113.77/32"      # TEST-NET-3: never matches real traffic


def get(node, path, auth, timeout=5):
    r = requests.get("http://%s/onos/v1%s" % (node, path), auth=auth, timeout=timeout)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--user", default="onos")
    ap.add_argument("--password", default="rocks")
    ap.add_argument("--expect-members", type=int, default=3)
    ap.add_argument("--expect-devices", type=int, default=0)
    ap.add_argument("--require", default="org.onosproject.openflow")
    ap.add_argument("--flow-test", action="store_true")
    a = ap.parse_args()
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
        print("== 4. cross-node flow-rule visibility (state propagation) ==")
        dev = sorted(next(iter(dev_sets.values())))[0]
        writer, readers = nodes[0], nodes[1:]
        body = {"priority": 100, "timeout": 0, "isPermanent": True, "deviceId": dev,
                "treatment": {"instructions": [{"type": "OUTPUT", "port": "1"}]},
                "selector": {"criteria": [{"type": "ETH_TYPE", "ethType": "0x0800"},
                                           {"type": "IPV4_DST", "ip": TEST_IP}]}}
        r = requests.post("http://%s/onos/v1/flows/%s?appId=org.onosproject.rest" % (writer, dev),
                          json=body, auth=auth, timeout=5)
        print("  POST on %s -> HTTP %s" % (writer, r.status_code))
        if r.status_code not in (200, 201):
            fail("flow POST failed: %s" % r.text[:300])
        else:
            t_w, flow = poll(writer, dev, auth, True)
            print("  visible on writer after %s s" % (None if t_w is None else round(t_w, 3)))
            for n in readers:
                t, _ = poll(n, dev, auth, True)
                print("  visible on %-18s after %s s" % (n, None if t is None else round(t, 3)))
                if t is None:
                    fail("flow never became visible on %s" % n)
            if flow:
                requests.delete("http://%s/onos/v1/flows/%s/%s" % (writer, dev, flow["id"]), auth=auth, timeout=5)
                for n in readers:
                    t, _ = poll(n, dev, auth, False)
                    print("  removal seen on %-18s after %s s" % (n, None if t is None else round(t, 3)))
                    if t is None:
                        fail("test flow not removed on %s; delete it manually" % n)
            else:
                fail("writer never listed the flow; manual cleanup may be needed (match IPV4_DST %s)" % TEST_IP)

    print("\nRESULT:", "B READY" if ok else "B NOT READY (see FAIL lines)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()