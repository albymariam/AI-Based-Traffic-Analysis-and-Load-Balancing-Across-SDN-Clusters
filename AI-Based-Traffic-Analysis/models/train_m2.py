"""Train Transformer + TCN on real telemetry; compare with LSTM and naive baselines on held-out runs.
   PYTHONPATH=. ~/venvs/ml/bin/python models/train_m2.py"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.config import ROOT, load_yaml, set_seed
from common.features import FEATURES
from models.dataset import build_splits, normalize
from models.lstm_predictor import LSTMTrafficPredictor
from models.transformer_predictor import build_model

U = FEATURES.index("utilization")
CK = ROOT / "models" / "checkpoints"
TAB = ROOT / "results" / "tables"


def metrics(y, p):
    e = p - y
    m, hot = y > 0.05, y > 0.5
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                sMAPE_gt5=100 * np.mean(2 * np.abs(e[m]) / (np.abs(y[m]) + np.abs(p[m]))) if m.any() else np.nan,
                MAPE_gt5=100 * np.mean(np.abs(e[m]) / y[m]) if m.any() else np.nan,
                MAE_hot_gt50=np.abs(e[hot]).mean() if hot.any() else np.nan, n_hot=int(hot.sum()))


def latency_ms(model, W, n_links=42, reps=50):
    torch.set_num_threads(1)
    model.eval()
    x = torch.rand(n_links, W, len(FEATURES))
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t) / reps * 1000


def train_one(arch, S, pc, ec):
    set_seed(1)
    W, H = ec["window"], ec["horizon"]
    T = lambda k: torch.from_numpy(normalize(S["X_" + k], S["norm"]))
    Xtr, Xva = T("train"), T("val")
    Ytr, Yva = torch.from_numpy(S["Y_train"]), torch.from_numpy(S["Y_val"])
    model = build_model(arch, len(FEATURES), W, H, U, pc)
    model.set_norm(S["norm"])
    opt = torch.optim.AdamW(model.parameters(), lr=ec["lr"], weight_decay=pc["weight_decay"])
    lossf = nn.SmoothL1Loss(beta=pc["huber_beta"])
    best, state, bad = 1e9, None, 0
    for ep in range(ec["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), ec["batch_size"]):
            idx = perm[i:i + ec["batch_size"]]
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
            print("[%s] epoch %3d  val_huber %.5f" % (arch, ep, v))
        if bad >= 8:
            print("[%s] early stop at epoch %d" % (arch, ep))
            break
    model.load_state_dict(state)
    model.eval()
    CK.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arch=arch, state=model.state_dict(), pc=pc, norm=S["norm"], window=W, horizon=H,
                    features=FEATURES, split_seeds=S["split_seeds"]), CK / ("%s.pt" % arch))
    return model


def main():
    exp, tr = load_yaml("experiment"), load_yaml("traffic")
    ec, pc = exp["predictor"], exp["predictor_m2"]
    W, H = ec["window"], ec["horizon"]
    S = build_splits(W, H, tr["warmup_s"])
    print("split seeds:", S["split_seeds"], "| windows:", {k: len(S["X_" + k]) for k in ("train", "val", "test")})
    Y, hu = S["Y_test"], S["X_test"][:, :, U]
    Xte = torch.from_numpy(normalize(S["X_test"], S["norm"]))

    preds, lat = {}, []
    for arch in ("transformer", "tcn"):
        m = train_one(arch, S, pc, ec)
        with torch.no_grad():
            preds[arch] = m(Xte).numpy()
        lat.append(dict(model=arch, params=sum(p.numel() for p in m.parameters()),
                        infer_ms_42links=round(latency_ms(m, W), 2)))

    lstm_ck, lstm_norm = CK / "lstm.pt", CK / "norm.json"
    if lstm_ck.exists() and lstm_norm.exists():            # re-evaluate the M1 LSTM on the same test windows
        ln = json.loads(lstm_norm.read_text())
        lm = LSTMTrafficPredictor(input_dim=len(FEATURES), hidden_dim=ec["hidden_dim"],
                                  num_layers=ec["num_layers"], forecast_horizon=H, dropout=ec["dropout"])
        lm.load_state_dict(torch.load(lstm_ck, map_location="cpu"))
        lm.eval()
        with torch.no_grad():
            preds["lstm"] = lm(torch.from_numpy(normalize(S["X_test"], ln["norm"]))).numpy()
        lat.append(dict(model="lstm", params=sum(p.numel() for p in lm.parameters()),
                        infer_ms_42links=round(latency_ms(lm, W), 2)))
    preds["persistence"] = np.repeat(hu[:, -1:], H, 1)
    preds["moving_average"] = np.repeat(hu.mean(1, keepdims=True), H, 1)

    rows = []
    for name, p in preds.items():
        rows.append(dict(model=name, horizon="all", **metrics(Y, p)))
        for h in (0, H - 1):
            rows.append(dict(model=name, horizon="t+%d" % (h + 1), **metrics(Y[:, h], p[:, h])))
    TAB.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(TAB / "prediction_error_m2.csv", index=False)
    pd.DataFrame(lat).to_csv(TAB / "infer_latency.csv", index=False)
    print(tab.to_string(index=False))
    print(pd.DataFrame(lat).to_string(index=False))


if __name__ == "__main__":
    main()