"""Flow demand estimated from OpenFlow per-flow byte counters (replaces the configured-rate oracle).

The Ryu app reports, for every pinned flow, the cumulative byte counter of its ingress-switch rule together with the
switch's own rule duration. The estimator turns successive samples into a rate in Mbps:

    rate = 8 * (bytes_now - bytes_oldest_in_window) / (duration_now - duration_oldest_in_window) / 1e6

* Duration comes from the switch, so controller polling jitter does not bias the rate.
* A counter that goes backwards (rule re-installed after a reroute, or removed and re-added) restarts the window;
  until a new window has `min_span_s` of data the last good estimate is held for at most `hold_s` seconds.
* A flow without any estimate is not given a demand (the caller treats its load as background traffic).
"""
from collections import deque


class FlowRateEstimator:
    def __init__(self, window_s=3.0, min_span_s=1.0, hold_s=5.0):
        self.window_s, self.min_span_s, self.hold_s = window_s, min_span_s, hold_s
        self.hist = {}        # key -> deque[(duration_s, bytes)]
        self.good = {}        # key -> (rate_mbps, wall_time_of_estimate)
        self.seen = {}        # key -> wall time of last sample

    def update(self, samples, now):
        """samples: {key: (byte_count, duration_s)} for the rules reported in one poll; now: wall-clock seconds."""
        for key, (nbytes, dur) in samples.items():
            h = self.hist.setdefault(key, deque())
            if h and (nbytes < h[-1][1] or dur < h[-1][0]):
                h.clear()                                   # counters restarted: rule was re-installed
            if h and dur == h[-1][0]:
                continue                                    # no new data since the previous poll
            h.append((dur, nbytes))
            while len(h) > 2 and dur - h[1][0] >= self.window_s:
                h.popleft()                                 # keep the oldest sample that is still >= window old
            self.seen[key] = now
            span = h[-1][0] - h[0][0]
            if span >= self.min_span_s:
                self.good[key] = (8.0 * (h[-1][1] - h[0][1]) / span / 1e6, now)
        for key in [k for k, t in self.seen.items() if now - t > self.hold_s + self.window_s]:
            self.hist.pop(key, None); self.good.pop(key, None); self.seen.pop(key, None)

    def rate(self, key, now):
        g = self.good.get(key)
        if g is None or now - g[1] > self.hold_s:
            return None
        return g[0]
