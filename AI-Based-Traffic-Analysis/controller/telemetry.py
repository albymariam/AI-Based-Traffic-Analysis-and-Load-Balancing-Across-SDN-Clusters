"""
SDN Telemetry Extraction & Data Preprocessing Pipeline
Academic Research Project RM-T11 (Phase P1)

Implements Equation (1) [Feature Formulation] and Equation (2) [Sliding Window].
"""

import numpy as np
import pandas as pd
import torch

# Equation (1): 7-Dimensional Feature Vector
# x_{l,t} = [B_{l,t}, P_{l,t}, N_{l,t}, U_{l,t}, Q_{l,t}, D_{l,t}, L_{l,t}]
FEATURE_NAMES = [
    'byte_throughput',  # B_{l,t}: Byte Throughput (Bytes/sec)
    'packet_rate',      # P_{l,t}: Packet Rate (packets/sec)
    'active_flows',     # N_{l,t}: Active Flow Count
    'utilization',      # U_{l,t}: Link Utilization Ratio [0.0 to 1.0]
    'queue_length',     # Q_{l,t}: Egress Buffer Queue Length
    'delay_ms',         # D_{l,t}: Estimated Propagation + Queuing Delay
    'packet_loss'       # L_{l,t}: Packet Drop Count
]

def generate_simulated_telemetry(num_timesteps=500, seed=42):
    """
    Generates synthetic sFlow/OpenFlow link telemetry traces with bursty traffic spikes.
    """
    np.random.seed(seed)
    from topology.mininet_topology import create_nsfnet_topology
    topo = create_nsfnet_topology()
    
    directed_links = []
    for u, v in topo.edges():
        directed_links.append((u, v))
        directed_links.append((v, u))
        
    records = []
    for t in range(num_timesteps):
        base_load = 0.35 + 0.25 * np.sin(2 * np.pi * t / 150)  # Diurnal sinusoidal traffic
        
        for u, v in directed_links:
            is_burst = np.random.rand() < 0.06  # 6% chance of sudden traffic burst
            burst_factor = np.random.uniform(1.8, 2.8) if is_burst else 1.0
            
            raw_util = (base_load + np.random.normal(0, 0.05)) * burst_factor
            utilization = float(np.clip(raw_util, 0.01, 0.98))
            
            byte_throughput = utilization * 100.0 * 1e6 / 8
            packet_rate = byte_throughput / 1000.0
            active_flows = int(np.clip(utilization * 120 + np.random.randint(-5, 5), 2, 200))
            
            if utilization > 0.75:
                queue_length = float((utilization - 0.75) * 80 + np.random.uniform(0, 5))
                packet_loss = int(np.random.poisson((utilization - 0.75) * 15))
            else:
                queue_length = float(np.random.uniform(0, 1.5))
                packet_loss = 0
                
            delay_ms = 1.0 + (queue_length * 0.04)
            
            records.append({
                'timestep': t,
                'link_id': f"L_{u}_{v}",
                'src': u,
                'dst': v,
                'byte_throughput': byte_throughput,
                'packet_rate': packet_rate,
                'active_flows': active_flows,
                'utilization': utilization,
                'queue_length': queue_length,
                'delay_ms': delay_ms,
                'packet_loss': packet_loss
            })
            
    return pd.DataFrame(records)

def create_sliding_window_sequences(df, window_size=10, forecast_horizon=5):
    """
    Implements Equation (2): Converts feature matrix into sliding window sequences X_{l,t}.
    Returns PyTorch tensors for X (Inputs) and Y (Target utilization over horizon H).
    """
    df = df.sort_values(by=['link_id', 'timestep']).reset_index(drop=True)
    X_samples, Y_samples = [], []
    
    for link_id, group in df.groupby('link_id'):
        features = group[FEATURE_NAMES].values
        target_util = group['utilization'].values
        
        # Feature Normalization (Min-Max)
        f_min = np.min(features, axis=0)
        f_max = np.max(features, axis=0)
        f_range = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)
        norm_features = (features - f_min) / f_range
        
        n = len(group)
        for i in range(n - window_size - forecast_horizon + 1):
            X_samples.append(norm_features[i : i + window_size])
            Y_samples.append(target_util[i + window_size : i + window_size + forecast_horizon])
            
    X_tensor = torch.tensor(np.array(X_samples), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_samples), dtype=torch.float32)
    
    return X_tensor, Y_tensor

if __name__ == "__main__":
    print("[Phase P1] Generating SDN Telemetry & Extracting Feature Matrices...")
    df_raw = generate_simulated_telemetry(num_timesteps=100)
    X_tensor, Y_tensor = create_sliding_window_sequences(df_raw, window_size=10, forecast_horizon=5)
    print(f"  7-D Feature Names: {FEATURE_NAMES}")
    print(f"  Input Tensor X (Eq 2):  {X_tensor.shape} (Samples, Window W=10, Features=7)")
    print(f"  Target Tensor Y:        {Y_tensor.shape} (Samples, Horizon H=5)")
