"""Spike: a 4-switch Mininet network attached to all three ONOS nodes (OpenFlow ports 6654-6656).
  sudo python3 experiments/onos_cluster_b/spike_topology.py      (leave running; Ctrl+C to stop)
Then, in another terminal:  bash scripts/onos_spike.sh check"""
import time

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import LinearTopo

sw = lambda n, **kw: OVSSwitch(n, protocols="OpenFlow13", **kw)
net = Mininet(topo=LinearTopo(4), switch=sw, controller=None, build=False)
for i, port in enumerate((6654, 6655, 6656)):
    net.addController("c%d" % i, controller=RemoteController, ip="127.0.0.1", port=port)
net.build()
net.start()
print("waiting for the switches to connect to the controllers ...")
net.waitConnected(timeout=60)
print("connected; run: bash scripts/onos_spike.sh check   (Ctrl+C here to stop)")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
net.stop()
