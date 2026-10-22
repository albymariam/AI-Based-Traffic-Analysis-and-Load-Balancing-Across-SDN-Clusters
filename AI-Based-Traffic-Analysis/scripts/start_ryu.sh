#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export T11_TOPOLOGY=${T11_TOPOLOGY:-abilene12}   # the controller must use the same topology as the Mininet run
source ~/venvs/ryu/bin/activate
exec ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 controller/ryu_te_app.py