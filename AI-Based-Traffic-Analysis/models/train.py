"""Train the LSTM on real telemetry; compare with naive baselines on held-out runs.
   PYTHONPATH=. ~/venvs/ml/bin/python models/train.py"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.config import ROOT, load_yaml, set_seed
from common.features import FEATURES
from models.dataset import POOL_MIN_SEED, build_splits, normalize
from models.lstm_predictor import LSTMTrafficPredictor

U = FEATURES.index("utilization")


def metrics(y, p):
    e = p - y
    m = y > 0.05
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                sMAPE=100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-6)),
                MAPE_util_gt_5pct=100 * np.mean(np.abs(e[m]) / y[m]) if m.any() else np.nan)


def main():
    pc = load_yaml("experiment")["predictor"]
    W, H = pc["window"], pc["horizon"]
    set_seed(1)
    S = build_splits(W, H, load_yaml("traffic")["warmup_s"])
    print("split seeds:", S["split_seeds"], "| windows:",
          {k: len(S["X_" + k]) for k in ("train", "val", "test")})
    T = lambda k: torch.from_numpy(normalize(S["X_" + k], S["norm"]))
    Xtr, Xva, Xte = T("train"), T("val"), T("test")
    Ytr, Yva = torch.from_numpy(S["Y_train"]), torch.from_numpy(S["Y_val"])

    model = LSTMTrafficPredictor(input_dim=len(FEATURES), hidden_dim=pc["hidden_dim"],
                                 num_layers=pc["num_layers"], forecast_horizon=H, dropout=pc["dropout"])
    opt, lossf = torch.optim.Adam(model.parameters(), lr=pc["lr"]), nn.MSELoss()
    best, state, bad = 1e9, None, 0
    for ep in range(pc["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), pc["batch_size"]):
            idx = perm[i:i + pc["batch_size"]]
            opt.zero_grad()
            lossf(model(Xtr[idx]), Ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva), Yva).item()
        if v < best - 1e-6:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0:
            print("epoch %3d  val_mse %.5f" % (ep, v))
        if bad >= 8:
            print("early stop at epoch", ep)
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        P = model(Xte).numpy()
    Y, hu = S["Y_test"], S["X_test"][:, :, U]
    preds = {"LSTM": P,
             "Persistence": np.repeat(hu[:, -1:], H, 1),
             "MovingAverage": np.repeat(hu.mean(1, keepdims=True), H, 1)}
    rows = []
    for name, p in preds.items():
        rows.append(dict(model=name, horizon="all", **metrics(Y, p)))
        for h in (0, H - 1):
            rows.append(dict(model=name, horizon="t+%d" % (h + 1), **metrics(Y[:, h], p[:, h])))
    tab = pd.DataFrame(rows).round(4)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / "results" / "tables" / "prediction_error.csv", index=False)
    print(tab.to_string(index=False))

    ck = ROOT / "models" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / "lstm.pt")
    (ck / "norm.json").write_text(json.dumps(dict(norm=S["norm"], features=FEATURES, window=W, horizon=H,
                                                  split_seeds=S["split_seeds"], pool_min_seed=POOL_MIN_SEED)))


if __name__ == "__main__":
    main()