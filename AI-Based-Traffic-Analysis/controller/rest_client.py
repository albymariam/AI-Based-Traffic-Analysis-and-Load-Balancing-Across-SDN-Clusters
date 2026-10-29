"""Thin REST client for the Ryu TE app. Create one instance per thread (requests.Session isn't thread-safe)."""
import time

import requests


class TEClient:
    def __init__(self, host="127.0.0.1", port=8080, timeout=5.0):
        self.base = "http://%s:%d/te" % (host, port)
        self.timeout = timeout
        self.s = requests.Session()

    def _get(self, path):
        r = self.s.get(self.base + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body):
        t = time.perf_counter()
        r = self.s.post(self.base + path, json=body, timeout=self.timeout)
        ms = (time.perf_counter() - t) * 1000.0
        if r.status_code != 200:
            raise RuntimeError("%s -> %s %s" % (path, r.status_code, r.text))
        return ms

    def stats(self):      return self._get("/stats")
    def ports(self):      return self._get("/ports")
    def linkflows(self):  return self._get("/linkflows")
    def flowstats(self):  return self._get("/flowstats")

    def wait_switches(self, n, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.stats()["switches"] >= n:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise TimeoutError("only some switches connected; is ryu-manager running?")

    def set_mode(self, mode):
        return self._post("/mode", {"mode": mode})

    def pin(self, src, dst, dport, protos, path):
        """Returns REST round-trip latency in ms (a lower bound on rule-install latency)."""
        return self._post("/pin", {"src": src, "dst": dst, "dport": dport, "protos": protos, "path": path})

    def unpin(self, src, dst, dport, protos):
        return self._post("/unpin", {"src": src, "dst": dst, "dport": dport, "protos": protos})