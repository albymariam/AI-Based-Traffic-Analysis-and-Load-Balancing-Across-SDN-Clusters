"""Ryu TE app (OpenFlow 1.3).
  * proactive destination-based routing: 'static' (lowest-id next hop) or 'ecmp' (select groups)
  * per-flow path pinning (priority 100) via REST  -> used by round-robin / LSTM / LP methods
  * 1 s-ish OpenFlow port-stats polling exposed via REST
Run:  ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 controller/ryu_te_app.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from webob import Response

from topology import portmap as pm

PRIO_BASE, PRIO_LOCAL, PRIO_FLOW = 10, 50, 100
ETH_IP = 0x0800


def _json(obj, status=200):
    return Response(content_type="application/json", charset="utf-8",
                    status=status, body=json.dumps(obj).encode())


class TEApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(TEApp, self).__init__(*args, **kwargs)
        self.cfg = pm.load_topology_cfg()
        self.pmap = pm.port_map(self.cfg)
        self.nh = pm.next_hops(self.cfg)
        self.mode = "static"
        self.dps = {}
        self.port_stats = {}
        self.flows = {}        # (src,dst,dport,protos) -> {"path": [...]}
        self.flowmods = 0      # per-flow FlowMods only (base installs not counted)
        poll = float(pm.load_yaml("experiment")["telemetry"]["poll_interval_s"])
        self.poll_interval = max(0.2, poll / 2.0)   # poll faster than the logger -> fresh data
        kwargs["wsgi"].register(TEController, {"app": self})
        hub.spawn(self._poll_loop)

    # ---------------- switch lifecycle ----------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _on_features(self, ev):
        dp = ev.msg.datapath
        self.dps[dp.id] = dp
        self._install_base(dp)
        self.logger.info("switch %s connected (%d/%d)", dp.id, len(self.dps), len(self.cfg["switches"]))

    @set_ev_cls(ofp_event.EventOFPStateChange, [DEAD_DISPATCHER])
    def _on_dead(self, ev):
        dp = ev.datapath
        if self.dps.get(dp.id) is dp:
            del self.dps[dp.id]
            self.port_stats.pop(dp.id, None)

    # ---------------- base routing ----------------
    def _add_flow(self, dp, prio, match, actions):
        p, ofp = dp.ofproto_parser, dp.ofproto
        inst = [p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(p.OFPFlowMod(dp, priority=prio, match=match, instructions=inst))

    def _install_base(self, dp):
        p, ofp, s = dp.ofproto_parser, dp.ofproto, dp.id
        dp.send_msg(p.OFPFlowMod(dp, table_id=ofp.OFPTT_ALL, command=ofp.OFPFC_DELETE,
                                 out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY, match=p.OFPMatch()))
        dp.send_msg(p.OFPGroupMod(dp, ofp.OFPGC_DELETE, ofp.OFPGT_ALL, ofp.OFPG_ALL))
        dp.send_msg(p.OFPBarrierRequest(dp))

        self._add_flow(dp, PRIO_LOCAL, p.OFPMatch(eth_type=ETH_IP, ipv4_dst=pm.host_ip(s)),
                       [p.OFPActionOutput(pm.HOST_PORT)])
        for d in self.cfg["switches"]:
            if d == s:
                continue
            ports = [self.pmap[s][n] for n in self.nh[(s, d)]]
            if self.mode == "ecmp" and len(ports) > 1:
                buckets = [p.OFPBucket(weight=1, watch_port=ofp.OFPP_ANY, watch_group=ofp.OFPG_ANY,
                                       actions=[p.OFPActionOutput(x)]) for x in ports]
                dp.send_msg(p.OFPGroupMod(dp, ofp.OFPGC_ADD, ofp.OFPGT_SELECT, d, buckets))
                actions = [p.OFPActionGroup(d)]
            else:
                actions = [p.OFPActionOutput(ports[0])]
            self._add_flow(dp, PRIO_BASE, p.OFPMatch(eth_type=ETH_IP, ipv4_dst=pm.host_ip(d)), actions)

    def set_mode(self, mode):
        if mode not in ("static", "ecmp"):
            raise ValueError("mode must be static|ecmp")
        self.mode = mode
        self.flows.clear()
        for dp in list(self.dps.values()):
            self._install_base(dp)

    # ---------------- per-flow pinning ----------------
    @staticmethod
    def _match(p, src, dst, proto, dport):
        kw = dict(eth_type=ETH_IP, ipv4_src=pm.host_ip(src), ipv4_dst=pm.host_ip(dst), ip_proto=proto)
        if dport is not None and proto == 6:
            kw["tcp_dst"] = dport
        elif dport is not None and proto == 17:
            kw["udp_dst"] = dport
        return p.OFPMatch(**kw)

    def _pin_add(self, dp, src, dst, proto, dport, out_port):
        p, ofp = dp.ofproto_parser, dp.ofproto
        inst = [p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(out_port)])]
        dp.send_msg(p.OFPFlowMod(dp, priority=PRIO_FLOW, match=self._match(p, src, dst, proto, dport),
                                 instructions=inst))
        self.flowmods += 1

    def _pin_del(self, dp, src, dst, proto, dport):
        p, ofp = dp.ofproto_parser, dp.ofproto
        dp.send_msg(p.OFPFlowMod(dp, command=ofp.OFPFC_DELETE_STRICT, priority=PRIO_FLOW,
                                 out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                                 match=self._match(p, src, dst, proto, dport)))
        self.flowmods += 1

    def pin_flow(self, src, dst, dport, protos, path):
        path = [int(x) for x in path]
        if path[0] != src or path[-1] != dst:
            raise ValueError("path endpoints do not match src/dst")
        for a, b in zip(path, path[1:]):
            if b not in self.pmap[a]:
                raise ValueError("no link %d-%d" % (a, b))
        if any(s not in self.dps for s in path):
            raise ValueError("switch not connected")
        key = (src, dst, dport, tuple(protos))
        old = self.flows.get(key)
        for i, s in enumerate(path):          # add new path first ...
            out = pm.HOST_PORT if i == len(path) - 1 else self.pmap[s][path[i + 1]]
            for proto in protos:
                self._pin_add(self.dps[s], src, dst, proto, dport, out)
        if old:                               # ... then remove switches no longer on the path
            for s in set(old["path"]) - set(path):
                if s in self.dps:
                    for proto in protos:
                        self._pin_del(self.dps[s], src, dst, proto, dport)
        self.flows[key] = {"path": path}

    def unpin_flow(self, src, dst, dport, protos):
        old = self.flows.pop((src, dst, dport, tuple(protos)), None)
        if old:
            for s in old["path"]:
                if s in self.dps:
                    for proto in protos:
                        self._pin_del(self.dps[s], src, dst, proto, dport)

    def link_flows(self):
        cnt = {}
        for (src, dst, dport, protos), v in self.flows.items():
            if protos == (1,):
                continue                       # ignore ICMP probes
            for a, b in zip(v["path"], v["path"][1:]):
                k = "%d-%d" % (a, b)
                cnt[k] = cnt.get(k, 0) + 1
        return cnt

    # ---------------- telemetry ----------------
    def _poll_loop(self):
        while True:
            for dp in list(self.dps.values()):
                dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(self.poll_interval)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _on_port_stats(self, ev):
        ports = {}
        for st in ev.msg.body:
            if st.port_no < 0xffffff00:
                ports[st.port_no] = dict(rx_bytes=st.rx_bytes, tx_bytes=st.tx_bytes,
                                         rx_packets=st.rx_packets, tx_packets=st.tx_packets,
                                         tx_dropped=st.tx_dropped)
        self.port_stats[ev.msg.datapath.id] = {"ts": time.time(), "ports": ports}


class TEController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(TEController, self).__init__(req, link, data, **config)
        self.app = data["app"]

    def _guard(self, fn):
        try:
            return _json(fn())
        except (ValueError, KeyError, TypeError, IndexError) as e:
            return _json({"error": str(e)}, 400)

    @route("te_stats", "/te/stats", methods=["GET"])
    def stats(self, req, **kw):
        a = self.app
        return _json({"mode": a.mode, "switches": len(a.dps), "pinned": len(a.flows),
                      "flowmods": a.flowmods, "ts": time.time(), "topology": a.cfg.get("name")})

    @route("te_ports", "/te/ports", methods=["GET"])
    def ports(self, req, **kw):
        return _json({"ts": time.time(), "switches": self.app.port_stats})

    @route("te_linkflows", "/te/linkflows", methods=["GET"])
    def linkflows(self, req, **kw):
        return _json(self.app.link_flows())

    @route("te_mode", "/te/mode", methods=["POST"])
    def mode(self, req, **kw):
        return self._guard(lambda: (self.app.set_mode(req.json["mode"]), {"ok": True})[1])

    @route("te_pin", "/te/pin", methods=["POST"])
    def pin(self, req, **kw):
        def f():
            b = req.json
            self.app.pin_flow(int(b["src"]), int(b["dst"]), b.get("dport"), b["protos"], b["path"])
            return {"ok": True}
        return self._guard(f)

    @route("te_unpin", "/te/unpin", methods=["POST"])
    def unpin(self, req, **kw):
        def f():
            b = req.json
            self.app.unpin_flow(int(b["src"]), int(b["dst"]), b.get("dport"), b["protos"])
            return {"ok": True}
        return self._guard(f)