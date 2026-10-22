"""Abilene multi-step OD forecasting: persistence, seasonal-naive (conditional), LSTM, TCN.
   python -m abilene.forecasting --models persistence seasonal lstm tcn --seeds 1 2 3 4 5
All models are scored on IDENTICAL held-out windows. Metrics are in Mbps on the original scale."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from abilene.loaders import load_real
from abilene.preprocessing import Scaler, chrono_split, clean, od_columns, to_mbps, windows

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ data
def prepare(cfg):
    dc, fc = cfg["data"], cfg["forecast"]
    weeks, layout = load_real(ROOT / dc["raw_dir"], dc["layout"])
    weeks, _ = clean(weeks)
    cols = od_columns(dc["drop_diagonal"])
    mb = {k: to_mbps(v[:, cols], dc["bytes_per_unit"], dc["interval_s"]) for k, v in weeks.items()}
    split = chrono_split(list(mb), cfg["split"]["train"], cfg["split"]["val"])
    sc = Scaler().fit([mb[k] for k in split["train"]])          # TRAIN only
    D = {}
    for name, ws in split.items():
        X, Y, M = windows({k: mb[k] for k in ws}, fc["window"], fc["horizon"])
        D[name] = dict(X=X, Y=Y, M=M)
    return D, sc, split, mb, cols


# ------------------------------------------------------------------ metrics
def metrics(Y, P, eps):
    e = P - Y
    m = Y > eps
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_%": float(100 * (np.abs(e[m]) / Y[m]).mean()) if m.any() else float("nan"),   # targets <= eps excluded
            "WAPE_%": float(100 * np.abs(e).sum() / Y.sum())}


def report(name, Y, P, eps, train_total_p90):
    rows = [dict(model=name, scope="all", **metrics(Y, P, eps))]
    for h in range(Y.shape[1]):
        rows.append(dict(model=name, scope="h+%d" % (h + 1), **metrics(Y[:, h], P[:, h], eps)))
    peak = Y.sum(2).max(1) >= train_total_p90                   # windows whose true total demand is in the TRAIN top-10%
    if peak.any():
        rows.append(dict(model=name, scope="peak", **metrics(Y[peak], P[peak], eps)))
    return rows


# ------------------------------------------------------------------ baselines
def persistence(X, H):
    return np.repeat(X[:, -1:, :], H, axis=1)


def seasonal_naive(M, H, W, series, lag):
    """prediction for step t = value at t-lag in the same week; returns (P, valid_mask)."""
    P, ok = np.zeros((len(M), H, series[M[0][0]].shape[1])), np.zeros(len(M), bool)
    for i, (wk, s) in enumerate(M):
        idx = s + W + np.arange(H) - lag
        if idx.min() >= 0:
            P[i], ok[i] = series[wk][idx], True
    return P, ok


# ------------------------------------------------------------------ models
def build(kind, P, W, H, fc):
    import torch.nn as nn

    class LSTM(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.rnn = nn.LSTM(P, c["hidden_dim"], c["num_layers"], batch_first=True,
                            dropout=c["dropout"] if c["num_layers"] > 1 else 0.0)
            s.head = nn.Linear(c["hidden_dim"], H * P)

        def forward(s, x):
            o, _ = s.rnn(x)
            return s.head(o[:, -1]).view(-1, H, P)

    class Block(nn.Module):
        def __init__(s, cin, cout, k, d, p):
            super().__init__()
            s.pad = (k - 1) * d
            s.c1, s.c2 = nn.Conv1d(cin, cout, k, dilation=d), nn.Conv1d(cout, cout, k, dilation=d)
            s.act, s.dr = nn.ReLU(), nn.Dropout(p)
            s.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(s, x):                                   # causal: left-pad only
            f = nn.functional.pad
            y = s.dr(s.act(s.c1(f(x, (s.pad, 0)))))
            y = s.dr(s.act(s.c2(f(y, (s.pad, 0)))))
            return s.act(y + s.skip(x))

    class TCN(nn.Module):
        def __init__(s, c):
            super().__init__()
            ch = [P] + [c["channels"]] * c["levels"]
            s.net = nn.Sequential(*[Block(ch[i], ch[i + 1], c["kernel"], 2 ** i, c["dropout"]) for i in range(c["levels"])])
            s.head = nn.Linear(c["channels"], H * P)

        def forward(s, x):
            y = s.net(x.transpose(1, 2))
            return s.head(y[:, :, -1]).view(-1, H, P)

    return (LSTM if kind == "lstm" else TCN)(fc[kind])


def train_model(kind, D, sc, fc, seed, out_dir):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(max(1, torch.get_num_threads()))
    T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    Xtr, Ytr, Xva, Yva = T(D["train"]["X"]), T(D["train"]["Y"]), T(D["val"]["X"]), T(D["val"]["Y"])
    P, W, H = Xtr.shape[2], Xtr.shape[1], Ytr.shape[1]
    m = build(kind, P, W, H, fc)
    opt, lf = torch.optim.Adam(m.parameters(), lr=fc["lr"]), nn.MSELoss()
    best, state, bad, curve = 1e9, None, 0, []
    for ep in range(fc["epochs"]):
        m.train()
        perm = torch.randperm(len(Xtr))                        # shuffles TRAIN windows only (all precede val/test)
        for i in range(0, len(perm), fc["batch_size"]):
            b = perm[i:i + fc["batch_size"]]
            opt.zero_grad(); l = lf(m(Xtr[b]), Ytr[b]); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            tr = lf(m(Xtr[:4096]), Ytr[:4096]).item(); va = lf(m(Xva), Yva).item()
        curve.append(dict(epoch=ep, train_mse=tr, val_mse=va))
        if va < best - 1e-6:
            best, state, bad = va, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= fc["patience"]:
                break
    m.load_state_dict(state); m.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(out_dir / ("curve_%s_s%d.csv" % (kind, seed)), index=False)
    ck = ROOT / "models" / "checkpoints" / "abilene"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "kind": kind, "cfg": fc[kind], "window": W, "horizon": H, "P": P,
                "scaler": sc.state(), "seed": seed}, ck / ("%s_s%d.pt" % (kind, seed)))
    return m


def predict(m, X, sc):
    import torch
    t0 = time.perf_counter()
    with torch.no_grad():
        Z = m(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy()
    return sc.inverse(Z), (time.perf_counter() - t0) * 1000 / max(1, len(X))


def acf_lag(mb, train_weeks, lag):
    t = np.concatenate([mb[k].sum(1) - mb[k].sum(1).mean() for k in train_weeks if len(mb[k]) > lag])
    return float((t[:-lag] * t[lag:]).sum() / (t * t).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["persistence", "seasonal", "lstm", "tcn"])
    ap.add_argument("--seeds", nargs="+", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))
    fc, eps = cfg["forecast"], cfg["data"]["zero_eps_mbps"]
    D, sc, split, mb, cols = prepare(cfg)
    te, W, H = D["test"], fc["window"], fc["horizon"]
    out = ROOT / "results" / "abilene" / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    p90 = float(np.percentile(np.concatenate([D["train"]["Y"].sum(2).max(1)]), 90))
    keep = np.ones(len(te["X"]), bool)
    lag_ok = False
    if "seasonal" in a.models:
        r = acf_lag(mb, split["train"], fc["seasonal_lag"])
        lag_ok = r > fc["seasonal_min_acf"]
        print("train ACF(total demand, lag=%d) = %.3f -> seasonal naive %s" % (fc["seasonal_lag"], r, "USED" if lag_ok else "SKIPPED"))
        if lag_ok:
            Psn, ok = seasonal_naive(te["M"], H, W, mb, fc["seasonal_lag"])
            keep &= ok                                        # identical windows for every model
    Y = te["Y"][keep]
    rows, preds, lat = [], {}, {}
    if "persistence" in a.models:
        preds["persistence"] = persistence(te["X"][keep], H)
    if "seasonal" in a.models and lag_ok:
        preds["seasonal_naive"] = Psn[keep]
    for kind in [k for k in a.models if k in ("lstm", "tcn")]:
        for seed in a.seeds or fc["seeds"]:
            m = train_model(kind, D, sc, fc, seed, out)
            P, ms = predict(m, te["X"][keep], sc)
            preds["%s_s%d" % (kind, seed)] = P; lat["%s_s%d" % (kind, seed)] = ms
    for name, P in preds.items():
        rows += report(name, Y, P, eps, p90)
        np.save(out / ("pred_test_%s.npy" % name), P.astype(np.float32))
    np.save(out / "actual_test.npy", Y.astype(np.float32)); np.save(out / "meta_test.npy", te["M"][keep])
    df = pd.DataFrame(rows)
    df["infer_ms_per_window"] = df["model"].map(lat)
    df.to_csv(ROOT / "results" / "tables" / "abilene_forecast_metrics.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({"cfg": cfg, "split_weeks": split, "n_test_windows": int(keep.sum()),
                                                      "od_columns": cols.tolist()}, indent=2))
    print(df[df.scope == "all"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
