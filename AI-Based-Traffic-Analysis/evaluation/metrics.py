"""
Comprehensive Evaluation Metrics Module
Academic Research Project RM-T11 (Section 4.2 / Section IX)

Calculates:
  1. AI Accuracy Metrics: MAE, RMSE, MAPE
  2. Network QoS Metrics: Max Util, Mean Util, Variance, Jain's Fairness Index, Latency, Jitter, Packet Loss, FCT, Reroute Count.
"""

import numpy as np

def calculate_jains_fairness_index(utilization_array):
    """
    Calculates Jain's Fairness Index across link utilization array:
      J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
    Range: [1/n, 1.0]. A value closer to 1.0 means more fair load distribution.
    """
    x = np.array(utilization_array, dtype=np.float64)
    n = len(x)
    if n == 0 or np.sum(x**2) == 0:
        return 1.0
    return float((np.sum(x) ** 2) / (n * np.sum(x ** 2)))

def calculate_ai_accuracy_metrics(y_true, y_pred):
    """
    Calculates AI prediction error metrics:
      - MAE: Mean Absolute Error
      - RMSE: Root Mean Square Error
      - MAPE: Mean Absolute Percentage Error (%)
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    
    # Avoid division by zero in MAPE
    denom = np.where(y_t == 0, 1e-5, y_t)
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def calculate_network_qos_metrics(link_util_dict, delay_list, loss_list, reroute_count=0):
    """
    Calculates Network Quality-of-Service (QoS) and Load Balancing metrics.
    """
    utils = list(link_util_dict.values())
    if isinstance(utils[0], (list, np.ndarray)):
        utils = [np.mean(u) for u in utils]
        
    utils = np.array(utils, dtype=np.float64)
    
    max_util = float(np.max(utils))
    mean_util = float(np.mean(utils))
    util_var = float(np.var(utils))
    fairness = calculate_jains_fairness_index(utils)
    
    avg_delay = float(np.mean(delay_list)) if len(delay_list) > 0 else 1.0
    jitter = float(np.std(delay_list)) if len(delay_list) > 0 else 0.1
    pkt_loss_rate = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
    
    # Flow Completion Time (FCT) model: proportional to delay and queue drops
    base_fct = 10.0  # ms base
    fct = base_fct + (avg_delay * 1.5) + (pkt_loss_rate * 5.0)
    
    return {
        'Max_Utilization': max_util,
        'Mean_Utilization': mean_util,
        'Utilization_Variance': util_var,
        'Jains_Fairness_Index': fairness,
        'End_to_End_Delay_ms': avg_delay,
        'Jitter_ms': jitter,
        'Packet_Loss_Rate': pkt_loss_rate,
        'Flow_Completion_Time_ms': fct,
        'Reroute_Count': reroute_count
    }
