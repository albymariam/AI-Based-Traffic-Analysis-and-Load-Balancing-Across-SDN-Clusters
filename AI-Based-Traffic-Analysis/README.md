# RM-T11: AI-Based Traffic Analysis & Load Balancing Across SDN Clusters

## Overview
This project implements a **forecast-driven SDN traffic-engineering framework** that predicts near-future link utilization using deep learning models (LSTM, Transformer/TCN) trained on SDN telemetry and uses these predictions to **proactively reroute flows before congestion occurs**.

**Methodology M1:** LSTM Traffic Predictor + Heuristic Rerouting  
**Methodology M2:** Forecast-then-Optimise (Transformer/TCN Multi-Step Predictor + Min-Max Link-Utilization LP)  
**Methodology M3:** Distributed Controller-Cluster Study (ONOS/ODL with East-West Synchronization & State Staleness Analysis)

The work addresses limitations of traditional static routing (ECMP, round-robin) during traffic spikes and quantifies how distributed controller state staleness affects load-balancing quality.


## Environment
- **OS:** Ubuntu 22.04 LTS (recommended)
- **Python:** 3.10+
- **Mininet:** 2.3+
- **Controller:** Ryu 4.34+ / ONOS 2.7+ / ODL
- **iperf3:** 3.13+
- **sFlow-RT:** latest (optional, for fine-grained telemetry)
- **TensorFlow/Keras:** 2.13+ (or PyTorch 2.0+ if you prefer)
- **scikit-learn:** 1.3+
- **Optimization (for M2):** CVXPY / PuLP / Gurobi (optional)

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/harshitha14-cyber/-AI-Based-Traffic-Analysis.git
cd rm-t11
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Mininet
```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch
```

### 5. Install Controller (choose one)

#### Option A: Ryu (for M1 prototype)
```bash
pip install ryu
```

#### Option B: ONOS / ODL (for M3 multi-controller study)
Follow the official installation guides:
- ONOS: https://docs.onosproject.org/docs/install-guide
- ODL: https://wiki.opendaylight.org/

### 6. Install iperf3
```bash
sudo apt-get install -y iperf3
```

### 7. Install sFlow-RT (optional)
```bash
cd /opt
sudo git clone https://github.com/sflow/sflow-rt.git
cd sflow-rt
./install.sh
```

### 8. Install Optimization Libraries (for M2)
```bash
pip install cvxpy pulp
# Or install Gurobi if you have an academic license
```

### 9. Test Mininet
```bash
sudo mn --test pingall
```

### 10. Run a simple topology
```bash
sudo python3 topology/mininet_topology.py
```

Inside Mininet CLI:
```bash
pingall
iperf3 -c <destination_host_ip>
```

## Project Structure

| Directory / File | Description |
| :--- | :--- |
| `configs/` | Configuration files (topology, traffic, experiments) |
| `topology/` | Mininet topology definitions |
| `controller/` | SDN controller logic (telemetry, routing, baselines) |
| `data/` | Raw and processed datasets |
| `models/` | LSTM/Transformer models, training, prediction |
| `optimization/` | Min-max LP and other optimization methods (M2) |
| `experiments/` | Experiment runners for baselines, M1, M2, M3 |
| `evaluation/` | Metrics and plotting utilities |
| `results/` | Generated figures and tables |
| `logs/` | Experiment logs |
| `scripts/` | Helper scripts for setup and testing |
| `README.md` | Project overview and documentation |
| `.gitignore` | Files and folders ignored by Git |
| `requirements.txt` | Python dependencies |


## Methodologies

### M1: LSTM Traffic Predictor + Heuristic Rerouting
- Collect telemetry from Mininet switches via OpenFlow or sFlow-RT.
- Extract features: bytes, packets, flow-count, utilization, queue, delay, loss.
- Train LSTM to predict link utilization over a short horizon (e.g., 5 seconds).
- Use predicted bottleneck utilization to select paths with hysteresis and cost constraints.
- Compare against ECMP, round-robin, current-utilization, and threshold-based baselines.

### M2: Forecast-then-Optimise (Transformer/TCN + Min-Max LP)
- Replace LSTM with a spatial-temporal Transformer or temporal-CNN multi-step predictor.
- Feed multi-step predictions into a min-max link-utilization linear program.
- Optimize routing globally to minimize maximum predicted link utilization.
- Compare against M1 and static baselines.

### M3: Distributed Controller-Cluster Study
- Deploy ONOS/ODL cluster with east-west state synchronization.
- Quantify how controller-state staleness (sync lag) degrades load-balancing quality vs. a single controller.
- Evaluate M1/M2 policies under realistic multi-controller conditions.

## Evaluation Metrics
- **Link utilization:** mean, max, variance, Jain's fairness index
- **Throughput:** aggregate and per-flow
- **End-to-end delay, jitter, packet loss**
- **Flow-completion time**
- **Traffic-prediction error:** MAE, RMSE, MAPE/sMAPE
- **Reroute count and control overhead**
- **Controller sync overhead and staleness** (for M3)

## Configuration Files

### `configs/topology.yaml`
Define your Mininet topology (switches, hosts, links, bandwidth, delay).

### `configs/traffic.yaml`
Define traffic generation patterns (constant, bursty, TCP/UDP, elephant flows).

### `configs/experiment.yaml`
Define experiment parameters:
- Prediction horizon (e.g., 5 seconds)
- Rerouting threshold
- Hysteresis parameter
- Optimization weights (for M2)
- Synchronization delay settings (for M3)

## Reproducibility
- All experiments should be run with fixed random seeds.
- Record Git commit hash, configuration file, and environment versions for each experiment.
- Store raw telemetry in `data/raw/` and processed datasets in `data/processed/`.
- Save trained models in `models/checkpoints/` (add to `.gitignore`).
- Generate figures and tables in `results/figures/` and `results/tables/`.

## Literature Survey
The project is grounded in four technical pillars:
1. **Traditional SDN & Load Balancing** (ECMP, Hedera, Mahout)
2. **LSTM Traffic Prediction in Networks**
3. **Spatial-Temporal Transformers & Linear Programming Optimization**
4. **Distributed SDN Controllers (ONOS/ODL) & Synchronization Staleness**

A detailed survey of 25+ papers is provided in the paper document.

## License
This project is for academic research purposes.

## Acknowledgments
- Mininet: http://mininet.org/
- Ryu: https://ryu-sdn.org/
- ONOS: https://onosproject.org/
- ODL: https://www.opendaylight.org/
- sFlow-RT: https://sflow.net/rt/
