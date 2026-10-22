import numpy as np
import pytest

from abilene import loaders, preprocessing as pp, topology_mapping as tm
from tests.conftest import make_week

TOPO = "tests/fixtures/topo-2003-04-10.txt"


@pytest.mark.parametrize("layout", ["pair_major", "estimate_major"])
def test_layout_detection(layout):
    raw = make_week(T=60, layout=layout)
    assert loaders.detect_layout(raw)[0] == layout
    real = loaders.split_layout(raw, layout)[:, :, 0]
    assert real.shape == (60, 144) and (real > 0).all()


def test_split_is_chronological_and_scaler_train_only(weeks_dir):
    weeks, _ = loaders.load_real(weeks_dir)
    sp = pp.chrono_split(list(weeks))
    assert sp["train"] + sp["val"] + sp["test"] == sorted(weeks)
    assert max(sp["train"]) < min(sp["val"]) and max(sp["val"]) < min(sp["test"])
    sc = pp.Scaler().fit([weeks[k] for k in sp["train"]])
    z = np.vstack([sc.transform(weeks[k]) for k in sp["train"]])
    assert abs(z.mean()) < 1e-6                                  # fitted on train
    assert abs(np.vstack([sc.transform(weeks[k]) for k in sp["test"]]).mean()) > 1e-9
    assert np.allclose(sc.inverse(sc.transform(weeks[1])), weeks[1], rtol=1e-6)


def test_windows_never_cross_weeks_or_nan():
    a = np.ones((50, 3)); b = np.ones((50, 3)) * 2; a[20, 0] = np.nan
    X, Y, M = pp.windows({1: a, 2: b}, 5, 2)
    assert not np.isnan(X).any() and not np.isnan(Y).any()
    assert (X[M[:, 0] == 2] == 2).all() and (Y[M[:, 0] == 1] == 1).all()
    assert len(X) == (50 - 7 + 1 - 7) + (50 - 7 + 1)             # 7 windows touch the NaN


def test_clean_flags_negative_not_imputes():
    w, bad = pp.clean({1: np.array([[1.0, -1.0], [np.nan, 2.0]])})
    assert bad == 2 and np.isnan(w[1]).sum() == 2


def test_units():
    assert pp.to_mbps(3750, 100, 300) == pytest.approx(0.01)


def test_topology_and_config():
    r, l = tm.parse_topology(TOPO)
    assert len(r) == 12 and len(l) == 30
    cfg = tm.mininet_config(r, l)
    assert len(cfg["switches"]) == 12 and len(cfg["links"]) == 15
    G, idx = tm.build_digraph(r, l)
    ps = tm.k_paths(G, idx["SNVAng"], idx["NYCMng"], 4)
    assert all(len(set(p)) == len(p) for p in ps) and len(ps) == 4   # loop-free


def test_minmax_beats_or_equals_shortest_path_and_conserves_flow():
    pytest.importorskip("cvxpy")
    from abilene.te_offline import TE
    r, l = tm.parse_topology(TOPO)
    G, _ = tm.build_digraph(r, l)
    pairs = [(s, d) for s in range(1, 13) for d in range(1, 13) if s != d]
    te = TE(G, pairs, 4)
    rng = np.random.default_rng(0)
    dem = rng.lognormal(5, 1, len(pairs))
    x, U, ms, ok = te.solve(dem)
    assert ok
    for k in range(len(pairs)):
        assert x[te.off[k]:te.off[k + 1]].sum() == pytest.approx(1.0)
    assert te.mlu(x, dem) == pytest.approx(U, rel=1e-3)
    assert U <= te.mlu(te.x_sp, dem) + 1e-9 and U <= te.mlu(te.x_ecmp, dem) + 1e-9
    assert set(np.unique(te.integral(x))) <= {0.0, 1.0}


def test_replay_schedule_coverage_and_rates():
    import yaml
    from abilene.replay import build_schedule
    cfg = yaml.safe_load(open("configs/abilene.yaml"))["replay"]
    r, l = tm.parse_topology(TOPO)
    topo = tm.mininet_config(r, l)
    pairs = [(s, d) for s in range(1, 13) for d in range(1, 13) if s != d]
    dem = np.random.default_rng(1).lognormal(2, 1, (10, len(pairs)))
    fl, info = build_schedule(dem, topo, cfg, 20.0, pairs)
    assert 0 < info["coverage_top_pairs"] <= 1 and fl and all(f["rate_mbps"] >= cfg["min_rate_mbps"] for f in fl)
