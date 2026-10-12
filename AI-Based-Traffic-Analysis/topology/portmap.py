"""Topology helpers shared by the Mininet driver, the Ryu app (py3.9) and the TE service.

Conventions
  * host h<s> (IP 10.0.0.<s>) hangs off switch s on port 1
  * inter-switch ports are numbered 2.. in ascending neighbour-id order
"""
from collections import deque
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HOST_PORT = 1


def load_yaml(name):
    with open(ROOT / "configs" / ("%s.yaml" % name)) as f:
        return yaml.safe_load(f)


def load_topology_cfg():
    return load_yaml("topology")


def host_ip(s):
    return "10.0.0.%d" % s


def host_mac(s):
    return "00:00:00:00:00:%02x" % s


def dpid(s):
    return "%016x" % s


def adjacency(cfg):
    adj = {s: set() for s in cfg["switches"]}
    for u, v in cfg["links"]:
        adj[u].add(v)
        adj[v].add(u)
    return {s: sorted(n) for s, n in adj.items()}


def port_map(cfg):
    """{switch: {neighbour: port}}"""
    return {s: {n: i + 2 for i, n in enumerate(nbrs)} for s, nbrs in adjacency(cfg).items()}


def iface_name(pmap, u, v):
    """OVS interface on switch u that faces switch v (Mininet naming)."""
    return "s%d-eth%d" % (u, pmap[u][v])


def hop_dist(cfg):
    adj = adjacency(cfg)
    dist = {}
    for src in adj:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        dist[src] = d
    return dist


def next_hops(cfg):
    """{(s, d): [neighbours strictly closer to d]}  (equal-cost next hops by hop count)"""
    adj, dist = adjacency(cfg), hop_dist(cfg)
    return {(s, d): [n for n in adj[s] if dist[n][d] == dist[s][d] - 1]
            for s in adj for d in adj if s != d}


def static_path(nh, s, d):
    """Path used by static routing (always the lowest-id next hop)."""
    path = [s]
    while path[-1] != d:
        path.append(nh[(path[-1], d)][0])
    return path