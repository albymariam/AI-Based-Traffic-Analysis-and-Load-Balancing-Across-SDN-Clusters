"""
NSFNET topology loaded from configs/topology.yaml.
Graph edge attrs:
  capacity       -> modelled capacity (Mbps), used for reporting
  emu_capacity   -> emulated capacity (Mbps), used for TCLink bw and utilisation ratios
  delay          -> ms
"""
import networkx as nx

from common.config import load_yaml


def create_nsfnet_topology(cfg: dict | None = None) -> nx.Graph:
    cfg = cfg or load_yaml("topology")
    G = nx.Graph()
    G.add_nodes_from(cfg["switches"])
    for u, v in cfg["links"]:
        G.add_edge(
            u, v,
            capacity=float(cfg["modelled_capacity_mbps"]),
            emu_capacity=float(cfg["emulated_capacity_mbps"]),
            delay=float(cfg["link_delay_ms"]),
        )
    assert G.number_of_nodes() == 14 and G.number_of_edges() == 21, "NSFNET must be 14 nodes / 21 links"
    return G


def get_k_shortest_paths(G, source, target, k=None):
    k = k or load_yaml("topology")["k_paths"]
    try:
        gen = nx.shortest_simple_paths(G, source, target, weight="delay")
        return [p for _, p in zip(range(k), gen)]
    except nx.NetworkXNoPath:
        return []


if __name__ == "__main__":
    G = create_nsfnet_topology()
    print(f"Switches: {G.number_of_nodes()}  Links: {G.number_of_edges()}")
    for i, p in enumerate(get_k_shortest_paths(G, 1, 14), 1):
        print(f"  Path {i}: {' -> '.join(map(str, p))}")