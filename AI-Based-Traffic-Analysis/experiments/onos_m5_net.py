"""M5: the 12-switch Abilene Mininet network attached to ONOS (no Ryu). Run as root, leave running.
  sudo env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/onos_m5_net.py --of-ports 6654,6655,6656
  (single node: --of-ports 6657)
Prints READY when all switches are connected; Ctrl+C or SIGTERM stops it."""
import argparse
import signal
import time

from topology import portmap as pm
from topology.mininet_net import build_network

ap = argparse.ArgumentParser()
ap.add_argument("--of-ports", default="6654,6655,6656")
ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()
cfg = pm.load_topology_cfg()
net, _ = build_network(cfg, controllers=[(a.host, int(p)) for p in a.of_ports.split(",")])
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
print("waiting for %d switches to connect ..." % len(cfg["switches"]), flush=True)
net.waitConnected(timeout=90)
print("READY", flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
net.stop()
