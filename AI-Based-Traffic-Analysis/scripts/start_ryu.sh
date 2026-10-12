#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/venvs/ryu/bin/activate
exec ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 controller/ryu_te_app.py