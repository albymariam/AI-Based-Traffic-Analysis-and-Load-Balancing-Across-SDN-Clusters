"""Path-based min-max link-utilisation LP.

  min  theta
  s.t. ubg_l + sum_{c,j : l in P_cj} (D_c / cap) * x_cj  <=  theta      for every directed link l
       sum_j x_cj = 1,  x_cj >= 0                                        for every commodity c

ubg_l is the predicted background utilisation (traffic we do not control). A tiny per-hop cost breaks ties
toward shorter paths without measurably changing theta.
"""
import time

import numpy as np
from scipy.optimize import linprog


def solve_minmax(links, commodities, ubg, cap, hop_eps=1e-3):
    """commodities: [{'demand': Mbps, 'paths': [[n1, n2, ...], ...]}]. Returns (theta, [x_c arrays], solve_ms) or None."""
    lidx = {l: i for i, l in enumerate(links)}
    cols = [(c, j) for c, com in enumerate(commodities) for j in range(len(com["paths"]))]
    n, nl, nc = len(cols) + 1, len(links), len(commodities)
    cost = np.zeros(n)
    cost[-1] = 1.0
    A_ub, A_eq = np.zeros((nl, n)), np.zeros((nc, n))
    b_ub = -np.array([ubg.get(l, 0.0) for l in links])
    for k, (c, j) in enumerate(cols):
        p, r = commodities[c]["paths"][j], commodities[c]["demand"] / cap
        for a, b in zip(p, p[1:]):
            A_ub[lidx[(a, b)], k] += r
        A_eq[c, k] = 1.0
        cost[k] = hop_eps * (len(p) - 1) * r
    A_ub[:, -1] = -1.0
    t0 = time.perf_counter()
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(nc),
                  bounds=[(0, 1)] * len(cols) + [(0, None)], method="highs")
    ms = (time.perf_counter() - t0) * 1000
    if not res.success:
        return None
    xs, k = [], 0
    for com in commodities:
        m = len(com["paths"])
        xs.append(np.clip(res.x[k:k + m], 0, 1))
        k += m
    return float(res.x[-1]), xs, ms