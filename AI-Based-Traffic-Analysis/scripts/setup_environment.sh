#!/usr/bin/env bash
set -euo pipefail
sudo apt update
sudo apt install -y mininet openvswitch-switch iperf3 git python3-venv python3-pip \
    openjdk-17-jre-headless docker.io
sudo systemctl enable --now openvswitch-switch || true

# ML venv (Python 3.12 is fine here)
python3 -m venv ~/venvs/ml --system-site-packages
~/venvs/ml/bin/pip install -U pip
~/venvs/ml/bin/pip install -r requirements.txt

echo "Ryu venv already built at ~/venvs/ryu (Python 3.9). Installing extras:"
~/venvs/ryu/bin/pip install -r requirements-ryu.txt

echo "Sanity checks:"
PYTHONPATH=. ~/venvs/ml/bin/python scripts/check_env.py ml || true
PYTHONPATH=. ~/venvs/ryu/bin/python scripts/check_env.py ryu || true