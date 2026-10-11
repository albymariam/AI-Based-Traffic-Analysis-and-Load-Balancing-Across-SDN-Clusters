"""
M1 Bottleneck-Aware Heuristic Path Selection & Hysteresis Engine
Academic Research Project RM-T11 (Phase P2)

Implements:
  - Equation (4): Path Bottleneck Utilization B(p)
  - Equation (5): Target Optimal Path Selection p*
  - Equation (6): Strict Hysteresis Rule [B(p_curr) - B(p*) > eta]
"""

import numpy as np

def compute_path_bottleneck(path, predicted_util_dict):
    """
    Implements Equation (4):
      B(p) = max_{l in p} ( max_{tau in [1, H]} U_hat_{l, t+tau} )
    Calculates the maximum predicted bottleneck link utilization across candidate path p.
    """
    max_bottleneck = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        link_key = (u, v)
        if link_key in predicted_util_dict:
            link_preds = predicted_util_dict[link_key]
            link_max = float(np.max(link_preds))
            if link_max > max_bottleneck:
                max_bottleneck = link_max
    return max_bottleneck

def select_optimal_path_with_hysteresis(candidate_paths, current_path, predicted_util_dict, eta=0.08):
    """
    Implements Equations (5) and (6):
      p* = arg min_{p in P_{s,d}} B(p)                        (Eq 5)
      Flow Reroute Triggered IF AND ONLY IF:
      B(p_curr) - B(p*) > eta                                 (Eq 6)
      
    eta: Strict hysteresis threshold parameter preventing route oscillation/flapping.
    """
    if not candidate_paths:
        return current_path, False, 0.0
        
    path_bottlenecks = {}
    for path in candidate_paths:
        path_key = tuple(path)
        path_bottlenecks[path_key] = compute_path_bottleneck(path, predicted_util_dict)
        
    # Equation (5): Select path p* minimizing bottleneck utilization
    best_path_key = min(path_bottlenecks, key=path_bottlenecks.get)
    best_bottleneck = path_bottlenecks[best_path_key]
    best_path = list(best_path_key)
    
    # Bottleneck of current active path
    if current_path is not None and tuple(current_path) in path_bottlenecks:
        current_bottleneck = path_bottlenecks[tuple(current_path)]
    else:
        current_bottleneck = 1.0  # Force initial assignment if current_path is None
        
    # Equation (6): Evaluate Hysteresis Threshold
    utilization_savings = current_bottleneck - best_bottleneck
    should_reroute = utilization_savings > eta
    
    selected_path = best_path if should_reroute else (current_path if current_path else best_path)
    return selected_path, should_reroute, utilization_savings

if __name__ == "__main__":
    print("[Phase P2] Testing M1 Bottleneck Rerouting & Hysteresis Engine (Equations 4, 5, 6)...")
    candidate_paths = [
        [1, 2, 8, 11, 14],      # Path 1 (Default ECMP)
        [1, 3, 6, 14],          # Path 2 (Alternative A)
        [1, 4, 9, 12, 13, 14]   # Path 3 (Alternative B)
    ]
    current_path = candidate_paths[0]
    
    # Mock future link utilization forecasts (Horizon H=5)
    mock_preds = {
        (1, 2): [0.88, 0.92, 0.95, 0.94, 0.91],  # Congested bottleneck on current path
        (2, 8): [0.75, 0.80, 0.82, 0.80, 0.78],
        (8, 11): [0.60, 0.65, 0.65, 0.60, 0.60],
        (11, 14): [0.50, 0.55, 0.55, 0.50, 0.50],
        (1, 3): [0.25, 0.30, 0.35, 0.30, 0.28],  # Clear alternative path
        (3, 6): [0.30, 0.35, 0.40, 0.35, 0.32],
        (6, 14): [0.35, 0.40, 0.42, 0.38, 0.35],
        (1, 4): [0.20, 0.22, 0.25, 0.22, 0.20],
        (4, 9): [0.25, 0.28, 0.30, 0.28, 0.25],
        (9, 12): [0.30, 0.32, 0.35, 0.32, 0.30],
        (12, 13): [0.35, 0.38, 0.40, 0.38, 0.35],
        (13, 14): [0.40, 0.42, 0.45, 0.42, 0.40]
    }
    
    selected, rerouted, savings = select_optimal_path_with_hysteresis(
        candidate_paths, current_path, mock_preds, eta=0.08
    )
    
    print(f"  Current Path:               {current_path}")
    print(f"  Selected Path:              {selected}")
    print(f"  Reroute Decision:           {'REROUTE' if rerouted else 'STAY'}")
    print(f"  Predicted Util Savings:     {savings*100:.1f}% (Hysteresis threshold eta=8.0%)")
