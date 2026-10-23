"""Print eth0 rx/tx byte counters of Docker containers as JSON (read through a netshoot sidecar, works with Docker Desktop).
  python3 scripts/docker_net_bytes.py atomix-b1 atomix-b2 onos-b1 > snap.json"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def one(c):
    r = subprocess.run(["docker", "run", "--rm", "--net=container:" + c, "nicolaka/netshoot", "sh", "-c",
                        "cat /sys/class/net/eth0/statistics/rx_bytes /sys/class/net/eth0/statistics/tx_bytes"],
                       capture_output=True, text=True, timeout=60)
    rx, tx = r.stdout.split()[:2]
    return c, dict(rx=int(rx), tx=int(tx))


t = time.time()
with ThreadPoolExecutor(8) as ex:
    out = dict(ex.map(one, sys.argv[1:]))
out["_t"] = t
print(json.dumps(out))
