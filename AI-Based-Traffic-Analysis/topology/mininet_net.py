"""Builds the NSFNET emulation in Mininet from configs/topology.yaml. Run as root."""
from mininet.link import Link, TCLink
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

from topology import portmap as pm


def build_network(cfg, controller_ip="127.0.0.1", of_port=6653):
    assert cfg["hosts_per_switch"] == 1, "portmap assumes one host per switch"
    pmap = pm.port_map(cfg)
    net = Mininet(controller=None, switch=OVSKernelSwitch, link=TCLink,
                  autoSetMacs=False, autoStaticArp=False, build=False)
    net.addController("c0", controller=RemoteController, ip=controller_ip, port=of_port)

    sw, hosts = {}, {}
    for s in cfg["switches"]:
        sw[s] = net.addSwitch("s%d" % s, dpid=pm.dpid(s), protocols="OpenFlow13", failMode="secure")
        hosts[s] = net.addHost("h%d" % s, ip="%s/24" % pm.host_ip(s), mac=pm.host_mac(s))
        net.addLink(hosts[s], sw[s], port2=pm.HOST_PORT, cls=Link)   # access link: not a bottleneck

    for u, v in cfg["links"]:
        net.addLink(sw[u], sw[v], port1=pmap[u][v], port2=pmap[v][u], cls=TCLink,
                    bw=cfg["emulated_capacity_mbps"], delay="%gms" % cfg["link_delay_ms"],
                    max_queue_size=cfg.get("queue_pkts", 30))
    net.start()
    net.staticArp()
    return net, pmap