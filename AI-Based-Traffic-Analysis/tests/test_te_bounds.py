import numpy as np
import pytest

from abilene import topology_mapping as tm
from abilene.te_bounds import MCF, KPath, Net, demand_matrix, flow_paths, oblivious_routing, vlb_fractions

TOPO = "tests/fixtures/topo-2003-04-10.txt"


@pytest.fixture(scope="module")
def net():
    r, l = tm.parse_topology(TOPO)
    G, _ = tm.build_digraph(r, l)
    pairs = [(1, 7), (2, 9), (3, 11), (4, 12), (5, 8), (6, 10), (9, 2), (12, 4), (7, 1), (10, 6)]
    return Net(G, pairs)


def demand(net, seed=0):
    return np.random.default_rng(seed).uniform(200, 900, net.P)


def test_full_mcf_lower_bounds_k_path_which_lower_bounds_shortest_path(net):
    d = demand(net)
    th = MCF(net).solve(demand_matrix(net, d))
    sp, k4 = KPath(net, [tm.k_paths(net.G, s, t, 1) for s, t in net.pairs]), KPath(net, [tm.k_paths(net.G, s, t, 4) for s, t in net.pairs])
    u_sp = sp.util(sp.x_sp, d).max()
    u_k = k4.util(k4.solve(d), d).max()
    assert th <= u_k + 1e-6 <= u_sp + 1e-6
    u_int = k4.util(k4.integral(k4.solve(d)), d).max()
    assert u_int >= u_k - 1e-9                                   # rounding to one path can only hurt


def test_oblivious_routing_is_a_valid_unit_flow_and_never_beats_the_optimum(net):
    f, ratio = oblivious_routing(net, verbose=False)
    for k, (s, t) in enumerate(net.pairs):                       # conservation: unit flow s -> t
        net_out = net.inc @ f[k]
        assert net_out[net.nidx[s]] == pytest.approx(1.0, abs=1e-6) and net_out[net.nidx[t]] == pytest.approx(-1.0, abs=1e-6)
    assert ratio >= 1.0 - 1e-6
    for seed in range(3):
        d = demand(net, seed)
        obl = ((f.T @ d) / net.cap).max()
        assert obl >= MCF(net).solve(demand_matrix(net, d)) - 1e-6
        assert obl <= ratio * MCF(net).solve(demand_matrix(net, d)) + 1e-6   # worst-case guarantee holds
    ps, w = flow_paths(net, 0, f[0], 4)
    assert w.sum() == pytest.approx(1.0) and all(p[0] == net.pairs[0][0] and p[-1] == net.pairs[0][1] for p in ps)


def test_vlb_flow_conservation(net):
    f = vlb_fractions(net)
    for k, (s, t) in enumerate(net.pairs):
        o = net.inc @ f[k]
        assert o[net.nidx[s]] == pytest.approx(1.0, abs=1e-9) and o[net.nidx[t]] == pytest.approx(-1.0, abs=1e-9)
