"""
Baseline Routing Algorithms Implementation for M1 Comparison
Academic Research Project RM-T11 (Section 4.1)

Implements the 4 baseline routing policies:
  1. ECMP (Equal-Cost Multi-Path): Hash-based routing over shortest paths.
  2. Round-Robin (RR): Cyclically distributes flows across candidate paths.
  3. Current-Utilization Heuristic (CUH): Reactive rerouting using instantaneous telemetry (H=0).
  4. Reactive Threshold Rerouting (RTR): Triggers flow re-allocation only when instantaneous link utilization > 85%.
"""

import numpy as np

def route_ecmp(candidate_paths, flow_id=0):
    """
    1. Equal-Cost Multi-Path (ECMP):
       Hashes flow headers (simulated via flow_id % len(candidate_paths)) to select a path deterministically.
    """
    if not candidate_paths:
        return None
    hash_idx = hash(flow_id) % len(candidate_paths)
    return candidate_paths[hash_idx]

def route_round_robin(candidate_paths, call_count=0):
    """
    2. Round-Robin (RR):
       Cyclically distributes incoming flows across candidate paths.
    """
    if not candidate_paths:
        return None
    idx = call_count % len(candidate_paths)
    return candidate_paths[idx]

def route_current_utilization_heuristic(candidate_paths, current_util_dict):
    """
    3. Current-Utilization Heuristic (CUH):
       Reactive policy (H=0) that selects the path with the minimum INSTANTANEOUS bottleneck link utilization.
    """
    if not candidate_paths:
        return None
        
    path_bottlenecks = {}
    for path in candidate_paths:
        max_util = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            util = current_util_dict.get((u, v), current_util_dict.get((v, u), 0.0))
            if util > max_util:
                max_util = util
        path_bottlenecks[tuple(path)] = max_util
        
    best_path = min(path_bottlenecks, key=path_bottlenecks.get)
    return list(best_path)

def route_reactive_threshold(candidate_paths, current_path, current_util_dict, threshold=0.85):
    """
    4. Reactive Threshold Rerouting (RTR):
       Triggers rerouting ONLY IF the current path bottleneck utilization exceeds threshold (85%).
    """
    if not candidate_paths:
        return current_path
        
    # Evaluate current path bottleneck
    curr_bottleneck = 0.0
    if current_path:
        for i in range(len(current_path) - 1):
            u, v = current_path[i], current_path[i+1]
            util = current_util_dict.get((u, v), current_util_dict.get((v, u), 0.0))
            if util > curr_bottleneck:
                curr_bottleneck = util
                
    # If current bottleneck is below 85%, stay on current path
    if curr_bottleneck < threshold and current_path is not None:
        return current_path, False
        
    # Otherwise, reroute to path with lowest current utilization
    best_path = route_current_utilization_heuristic(candidate_paths, current_util_dict)
    return best_path, True
