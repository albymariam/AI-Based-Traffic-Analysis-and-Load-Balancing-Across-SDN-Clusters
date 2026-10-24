"""Historical OD demand -> Mininet flow schedule (same CSV schema the repo's FlowRunner already consumes).
Mapping (all documented, all configurable in configs/abilene.yaml -> replay):
  * Mbps_real = file_units * 100 B * 8 / 300 s            (units per readme; confirm)
  * one 5-min TM interval is replayed for `step_s` seconds (time compression 300/step_s)
  * demand is scaled by ONE constant so the busiest emulated link under static shortest path reaches `target_mlu`
    of the emulated capacity (stress scaling; absolute Abilene rates are NOT reproduced)
  * only the `top_pairs` largest OD pairs are generated (iperf3 process cost); dropped volume is reported
  * each (OD, interval) becomes one iperf3 flow of kind 'elephant' at that offered rate for step_s seconds
Offered != carried: compare with iperf3 results / port counters from the run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from topology import portmap as pm

ROOT = Path(__file__).resolve().parents[1]


def build_schedule(demand_mbps, topo_cfg, rc, warmup_s, pairs):
    """demand_mbps: (steps, P) real-scale Mbps; pairs: [(src,dst)] switch ids. Returns flows, info."""
    n, P = demand_mbps.shape
    tot = demand_mbps.sum(0)
    keep = np.argsort(-tot)[:rc["top_pairs"]]
    nh = pm.next_hops(topo_cfg)
    load = {}
    for k in keep:                                   # static shortest path load per link at each step
        p = pm.static_path(nh, *pairs[k])
        for e in zip(p, p[1:]):
            load.setdefault(e, np.zeros(n))
            load[e] += demand_mbps[:, k]
    peak = max(v.max() for v in load.values())
    scale = rc["target_mlu"] * topo_cfg["emulated_capacity_mbps"] / peak
    flows = []
    for t in range(n):
        for k in keep:
            r = demand_mbps[t, k] * scale
            if r < rc["min_rate_mbps"]:
                continue
            t0 = 0.0 if t == 0 else warmup_s + t * rc["step_s"]
            dur = rc["step_s"] + (warmup_s if t == 0 else 0)
            flows.append(dict(t_start=round(t0, 3), src=pairs[k][0], dst=pairs[k][1], kind="elephant",
                              proto=rc["proto"], rate_mbps=round(float(r), 4), dur_s=round(float(dur), 3)))
    flows.sort(key=lambda f: f["t_start"])
    for i, f in enumerate(flows):
        f["id"] = i
    info = dict(scale_to_emulated=float(scale), coverage_top_pairs=float(tot[keep].sum() / tot.sum()),
                n_flows=len(flows), steps=n, step_s=rc["step_s"], time_compression=300.0 / rc["step_s"],
                target_mlu_static=rc["target_mlu"])
    return flows, info


def add_mice(flows, per_s, t_end, seed, size_kb):
    """Short TCP flows (kind 'mice') on the OD pairs of the elephants, Poisson arrivals, never pinned: they follow the default
    routes, so their completion time (FCT) shows what the TE policy does to short flows. Elephants keep their ids."""
    rng = np.random.default_rng(1000 + seed)
    ods = [(f["src"], f["dst"]) for f in flows]
    n = rng.poisson(per_s * (t_end - 2.0))
    nid = max(f["id"] for f in flows) + 1
    for t in np.sort(rng.uniform(2.0, t_end, n)):
        s_, d_ = ods[rng.integers(len(ods))]
        flows.append(dict(t_start=round(float(t), 3), src=s_, dst=d_, kind="mice", proto="tcp",
                          size_bytes=int(rng.uniform(*size_kb) * 1000), id=nid))
        nid += 1
    flows.sort(key=lambda f: (f["t_start"], f["id"]))
    return n


def load_schedule(path, topo_cfg):
    """-> flows, pairs, weights, info  (what experiments/run_baseline.py expects from build_schedule)."""
    flows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            g = {k: (v if v != "" else None) for k, v in r.items()}
            for k in ("t_start", "rate_mbps", "dur_s"):
                g[k] = float(g[k]) if g.get(k) is not None else None
            for k in ("src", "dst", "id"):
                g[k] = int(g[k])
            flows.append(g)
    w = {}
    for fl in flows:
        w[(fl["src"], fl["dst"])] = w.get((fl["src"], fl["dst"]), 0.0) + (fl["rate_mbps"] or 0.0) * (fl["dur_s"] or 0.0)
    pairs = list(w)
    wt = np.array([w[p] for p in pairs]); wt = wt / wt.sum()
    dist = pm.hop_dist(topo_cfg)
    info = dict(avg_hops=float(sum(wi * dist[s][d] for (s, d), wi in zip(pairs, wt))),
                demand_mbps=float(sum((f["rate_mbps"] or 0) for f in flows) / max(1, len({f["t_start"] for f in flows}))),
                lambda0=0.0, n_flows=len(flows))
    return flows, pairs, wt, info


def segment_for(seed, split):
    """Deterministic (week, start) for a seed: seeds < 100 -> TEST weeks (evaluation), seeds >= 100 -> TRAIN weeks
    (telemetry collection). Different seeds give different weeks / times of week; all methods share a seed's segment."""
    pool, i = (split["test"], seed - 1) if seed < 100 else (split["train"], seed - 100)
    return pool[i % len(pool)], 60 + 240 * ((i * 3) % 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="pick week/start from the seed (see segment_for); overrides --week/--start")
    ap.add_argument("--week", type=int, help="test week to replay (default: first test week)")
    ap.add_argument("--start", type=int, default=0, help="first 5-min step inside the week")
    ap.add_argument("--out", default="results/abilene/replay_schedule.csv")
    ap.add_argument("--mice-per-s", type=float, default=0.0, help="add short TCP flows (for FCT); 0 = elephants only (the M4 matrix)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))
    from abilene.forecasting import prepare
    from abilene.preprocessing import to_mbps
    from abilene.loaders import load_real
    from abilene.preprocessing import clean, od_columns
    dc, rc = cfg["data"], cfg["replay"]
    weeks, _ = load_real(ROOT / dc["raw_dir"], dc["layout"])
    weeks, _ = clean(weeks)
    _, _, split, _, cols = prepare(cfg)
    wk, start = segment_for(a.seed, split) if a.seed else (a.week or split["test"][0], a.start)
    for shift in range(0, 600, 60):                                   # nothing is imputed: move on to the next clean segment
        seg = to_mbps(weeks[wk][start + shift:start + shift + rc["n_steps"]][:, cols], dc["bytes_per_unit"], dc["interval_s"])
        if len(seg) == rc["n_steps"] and not np.isnan(seg).any():
            break
    else:
        raise SystemExit("no clean replay segment found for week %d from step %d" % (wk, start))
    a.start = start + shift
    topo = yaml.safe_load(open(ROOT / "configs" / "topology_abilene12.yaml"))
    pairs = [(c // 12 + 1, c % 12 + 1) for c in cols]
    flows, info = build_schedule(seg, topo, rc, cfg_warm(), pairs)
    if a.mice_per_s > 0:
        tcfg = yaml.safe_load(open(ROOT / "configs" / "traffic.yaml"))
        info["n_mice"] = add_mice(flows, a.mice_per_s, cfg_warm() + rc["n_steps"] * rc["step_s"] - 5.0, a.seed or 0,
                                  tcfg["flow_mix"]["mice"]["size_kb"])
        info["n_flows"] = len(flows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t_start", "src", "dst", "kind", "proto", "rate_mbps", "dur_s", "size_bytes", "id"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(flows)
    info.update(week=wk, start=a.start, seed=a.seed)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def cfg_warm():
    return float(yaml.safe_load(open(ROOT / "configs" / "traffic.yaml"))["warmup_s"])


if __name__ == "__main__":
    main()
