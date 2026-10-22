"""Parse topo-2003-04-10.txt -> networkx graph + Mininet config (configs/topology_abilene12.yaml).
Deliberate differences vs the real backbone (document in the report):
  * Mininet links all use the SAME emulated bandwidth (repo's portmap/LP assume uniform capacity), so the two
    2.48 Gbps ATLAng<->IPLSng links (4x slower than the rest) are NOT slower in emulation. The offline model keeps them.
  * Uniform 1 ms link delay; OSPF weights are used only for candidate-path ranking, not by the emulated forwarding.
  * 2003 topology file; the 2004 traffic may have been carried on a slightly different network."""
import argparse
from pathlib import Path

import networkx as nx
import yaml

DOMAINS = {"A": ["DNVRng", "LOSAng", "SNVAng", "STTLng"],        # west
           "B": ["CHINng", "HSTNng", "IPLSng", "KSCYng"],        # central
           "C": ["ATLA-M5", "ATLAng", "NYCMng", "WASHng"]}       # east


def parse_topology(path):
    """-> routers {name: (lat, lon)}, links [(x, y, kbps, ospf)] directed as listed."""
    routers, links, sec = {}, [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in ("router", "link"):
            sec = line
            continue
        f = line.split()
        if sec == "router":
            routers[f[0]] = (float(f[2]), float(f[3]))
        elif sec == "link":
            links.append((f[0], f[1], int(f[2]), int(f[3])))
    return routers, links


def node_order(routers):
    """Alphabetical router order == assumed OD index order of the X files (VERIFY against the readme/demands file)."""
    return sorted(routers)


def build_digraph(routers, links):
    order = node_order(routers)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, len(order) + 1))
    idx = {n: i + 1 for i, n in enumerate(order)}
    for x, y, kbps, w in links:
        G.add_edge(idx[x], idx[y], capacity_mbps=kbps / 1000.0, ospf=w)
    return G, idx


def mininet_config(routers, links, emu_mbps=10.0, delay_ms=1):
    order = node_order(routers)
    idx = {n: i + 1 for i, n in enumerate(order)}
    und = sorted({tuple(sorted((idx[x], idx[y]))) for x, y, _, _ in links})
    cap = max(k for *_, k, _ in links) / 1000.0
    return {"name": "abilene12", "modelled_capacity_mbps": cap, "emulated_capacity_mbps": emu_mbps,
            "link_delay_ms": delay_ms, "hosts_per_switch": 1, "k_paths": 3, "queue_pkts": 30,
            "node_names": {idx[n]: n for n in order},
            "switches": list(range(1, len(order) + 1)), "links": [list(e) for e in und],
            "domains": {d: sorted(idx[n] for n in ns) for d, ns in DOMAINS.items()}}


def k_paths(G, s, d, k, weight="ospf"):
    """Loop-free (simple) k shortest paths by OSPF weight."""
    out = []
    for p in nx.shortest_simple_paths(G, s, d, weight=weight):
        out.append(p)
        if len(out) == k:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="data/raw/abilene/topo-2003-04-10.txt")
    ap.add_argument("--out", default="configs/topology_abilene12.yaml")
    a = ap.parse_args()
    r, l = parse_topology(a.topo)
    cfg = mininet_config(r, l)
    for d, ns in cfg["domains"].items():
        sub = nx.Graph([tuple(e) for e in cfg["links"] if e[0] in ns and e[1] in ns])
        assert set(ns) <= set(sub.nodes) and nx.is_connected(sub), "domain %s not connected" % d
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote %s: %d switches, %d links" % (a.out, len(cfg["switches"]), len(cfg["links"])))
