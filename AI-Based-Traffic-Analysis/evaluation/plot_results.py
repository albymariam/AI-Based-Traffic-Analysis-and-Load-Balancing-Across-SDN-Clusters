"""
Publication-Quality Plotting & Visualization Module
Academic Research Project RM-T11

Generates and saves 4 core academic figures into results/figures/:
  1. Max Link Utilization Comparison (ECMP vs RR vs CUH vs RTR vs M1)
  2. End-to-End Latency, Jitter, and Packet Loss Comparison
  3. Jain's Fairness Index & Reroute Count (Hysteresis Flap Prevention)
  4. LSTM Traffic Prediction Error Distribution (MAE / RMSE)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "C:/Users/ALBY MARIAM/AI-Based-Traffic-Analysis/results/figures"

def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_max_utilization(timesteps, util_results_dict):
    """
    Figure 1: Max Link Utilization Over Time
    """
    ensure_dir()
    plt.figure(figsize=(9, 5))
    
    styles = {
        'ECMP': ('#d9534f', '--', 'ECMP (Static)'),
        'Round-Robin': ('#f0ad4e', '-.', 'Round-Robin'),
        'CUH': ('#5bc0de', ':', 'CUH (Reactive H=0)'),
        'RTR': ('#8e44ad', '-.', 'RTR (Threshold 85%)'),
        'M1 (LSTM + Hysteresis)': ('#5cb85c', '-', 'M1 (LSTM Forecast + Hysteresis)')
    }
    
    for key, data in util_results_dict.items():
        color, style, label = styles.get(key, ('#333', '-', key))
        plt.plot(timesteps, data, label=label, color=color, linestyle=style, linewidth=2.0)
        
    plt.axhline(y=0.85, color='gray', linestyle=':', alpha=0.7, label='Congestion Threshold (85%)')
    plt.title("Max Link Utilization Comparison Across Routing Policies", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Timesteps (seconds)", fontsize=11)
    plt.ylabel("Maximum Link Utilization Ratio", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "max_utilization_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_qos_metrics_barchart(qos_summary_df):
    """
    Figure 2: End-to-End Latency, Jitter, and Packet Loss Bar Chart
    """
    ensure_dir()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    methods = qos_summary_df['Method']
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#8e44ad', '#5cb85c']
    
    # 1. Latency
    axes[0].bar(methods, qos_summary_df['End_to_End_Delay_ms'], color=colors, width=0.6)
    axes[0].set_title("End-to-End Latency (ms)", fontweight='bold')
    axes[0].set_ylabel("Delay (ms)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=30)
    
    # 2. Jitter
    axes[1].bar(methods, qos_summary_df['Jitter_ms'], color=colors, width=0.6)
    axes[1].set_title("Jitter (ms)", fontweight='bold')
    axes[1].set_ylabel("Jitter (ms)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=30)
    
    # 3. Packet Loss
    axes[2].bar(methods, qos_summary_df['Packet_Loss_Rate'], color=colors, width=0.6)
    axes[2].set_title("Packet Loss Rate (%)", fontweight='bold')
    axes[2].set_ylabel("Loss Rate (%)")
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.suptitle("Network QoS Performance Comparison", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "qos_latency_loss_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_fairness_and_reroutes(summary_df):
    """
    Figure 3: Jain's Fairness Index & Reroute Count
    """
    ensure_dir()
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    
    methods = summary_df['Method']
    x = np.arange(len(methods))
    width = 0.35
    
    ax2 = ax1.twinx()
    
    rects1 = ax1.bar(x - width/2, summary_df['Jains_Fairness_Index'], width, label="Jain's Fairness Index", color='#2980b9')
    rects2 = ax2.bar(x + width/2, summary_df['Reroute_Count'], width, label="Reroute Count (Control Overhead)", color='#e67e22')
    
    ax1.set_ylabel("Jain's Fairness Index (0 to 1.0)", color='#2980b9', fontweight='bold')
    ax2.set_ylabel("Total Reroute Count", color='#e67e22', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25)
    ax1.set_ylim(0.0, 1.1)
    
    plt.title("Load Balancing Fairness vs. Control Overhead (Reroute Flapping)", fontweight='bold')
    fig.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "fairness_and_reroutes.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath
