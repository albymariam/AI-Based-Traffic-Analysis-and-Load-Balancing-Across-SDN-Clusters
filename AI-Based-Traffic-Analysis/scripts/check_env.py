"""Run in each venv:  python scripts/check_env.py [ml|ryu]"""
import importlib
import shutil
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "ml"
mods = {
    "ml": ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "yaml",
           "networkx", "torch", "cvxpy", "pulp", "matplotlib", "requests"],
    "ryu": ["ryu", "eventlet", "webob", "routes", "requests", "yaml"],
}[mode]
bins = ["ovs-vsctl", "iperf3", "mn"] if mode == "ml" else ["ryu-manager"]

bad = 0
print(f"Python {sys.version.split()[0]}")
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad += 1
        print(f"  FAIL {m:12s} {e}")
for b in bins:
    p = shutil.which(b)
    print(f"  {'OK  ' if p else 'FAIL'} {b:12s} {p or 'not on PATH'}")
    bad += p is None
if mode == "ml":
    from topology.mininet_topology import create_nsfnet_topology
    G = create_nsfnet_topology()
    print(f"  OK   topology      {G.number_of_nodes()} nodes / {G.number_of_edges()} links")
sys.exit(1 if bad else 0)