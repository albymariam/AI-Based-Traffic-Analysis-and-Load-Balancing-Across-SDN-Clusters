#!/usr/bin/env bash
# Ubuntu 22.04 / 24.04 (or WSL2 with systemd). Run from the repo root:  bash scripts/setup_env.sh
set -euo pipefail
sudo apt update
sudo apt install -y mininet openvswitch-switch iperf3 git python3-venv python3-pip software-properties-common \
     openjdk-17-jre-headless docker.io
sudo systemctl enable --now openvswitch-switch || true

# 1) ML / optimisation / Mininet driver env. --system-site-packages exposes the apt `mininet` module.
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build
.venv/bin/pip install -r requirements.txt -r requirements-abilene.txt

# 2) Ryu needs Python 3.9 (eventlet 0.31.1). Ubuntu 22.04 has no 3.9 -> deadsnakes PPA.
if ! command -v python3.9 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
  sudo apt install -y python3.9 python3.9-venv
fi
mkdir -p ~/venvs && python3.9 -m venv ~/venvs/ryu
~/venvs/ryu/bin/pip install "pip<24.1" "setuptools==65.5.0" wheel
~/venvs/ryu/bin/pip install ryu==4.34 -r requirements-ryu.txt

# 3) sanity
sudo mn --test pingall
PYTHONPATH=. .venv/bin/python scripts/check_env.py ml || true
PYTHONPATH=. ~/venvs/ryu/bin/python scripts/check_env.py ryu || true
