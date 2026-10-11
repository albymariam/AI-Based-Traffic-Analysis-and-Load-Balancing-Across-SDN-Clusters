"""
SDN Network Topology Definition & Candidate Path Finder
Academic Research Project RM-T11 (Phase P0)

Defines standard 14-node NSFNET topology and K-Shortest Path routing calculation.
"""

import networkx as nx

def create_nsfnet_topology():
    """
    Constructs the 14-node NSFNET topology (standard benchmark in SDN Traffic Engineering).
    Nodes: 14 switches
    Edges: 21 bidirectional physical links with uniform 100 Mbps capacity.
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, 15))
    
    # 21 bidirectional physical links
    edges = [
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 8),
        (3, 6),
        (4, 5), (4, 9),
        (5, 6), (5, 7),
        (6, 13), (6, 14),
        (7, 8),
        (8, 11),
        (9, 10), (9, 12),
        (10, 11), (10, 13),
        (11, 12), (11, 14),
        (12, 13)
    ]
    
    for u, v in edges:
        G.add_edge(u, v, capacity=100.0, delay=1.0)  # Capacity in Mbps, Delay in ms
        
    return G

def get_k_shortest_paths(G, source, target, k=3):
    """
    Calculates K shortest loop-free candidate paths between source and target switches.
    Uses Yen's / Dijkstra shortest simple paths algorithm based on propagation delay.
    """
    try:
        paths = list(nx.shortest_simple_paths(G, source, target, weight='delay'))
        return paths[:k]
    except nx.NetworkXNoPath:
        return []

if __name__ == "__main__":
    topo = create_nsfnet_topology()
    print(f"[Phase P0] NSFNET Topology Initialized:")
    print(f"  Switches (Nodes): {topo.number_of_nodes()}")
    print(f"  Physical Links:   {topo.number_of_edges()}")
    
    sample_paths = get_k_shortest_paths(topo, source=1, target=14, k=3)
    print(f"\nCandidate Paths (Switch 1 -> Switch 14):")
    for i, path in enumerate(sample_paths, 1):
        print(f"  Path {i}: {' -> '.join(map(str, path))}")
