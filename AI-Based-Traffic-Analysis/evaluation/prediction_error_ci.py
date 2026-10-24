"""Forecast error of the three predictors (+ persistence) on the 10 EVALUATION seeds, with 95% CIs across seeds.
Telemetry comes from the static_shortest Abilene runs (no controller action, so the same traffic for every model);
the models were trained on seeds >= 100 only, so this is a held-out test.
   PYTHONPATH=. .venv/bin/python evaluation/prediction_error_ci.py"""
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats

from common.config import ROOT, load_yaml
from common.features import FEATURES
from models.dataset import _link_arrays, _windows, normalize
from models.lstm_predictor import LSTMTrafficPredictor
from models.transformer_predictor import build_model

U = FEATURES.index("utilization")
CK, RUNS, TAB = ROOT / "models/checkpoints", ROOT / "results/runs", ROOT / "results/tables"
SCEN, SRC = "abilene_replay", "static_shortest"


def metrics(y, p):
    e = p - y
    m = y > 0.05                                          # MAPE only where the link carries >5% load
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                MAPE_gt5=100 * np.mean(np.abs(e[m]) / y[m]) if m.any() else np.nan)


def load_models(exp):
    torch.set_num_threads(1)
    pc, pm = exp["predictor"], exp["predictor_m2"]
    meta = json.loads((CK / "norm.json").read_text())
    W, H = meta["window"], meta["horizon"]
    out = {}
    lstm = LSTMTrafficPredictor(input_dim=len(FEATURES), hidden_dim=pc["hidden_dim"], num_layers=pc["num_layers"],
                                forecast_horizon=H, dropout=pc["dropout"])
    lstm.load_state_dict(torch.load(CK / "lstm.pt", map_location="cpu"))
    out["lstm"] = (lstm.eval(), meta["norm"])
    for arch in ("transformer", "tcn"):
        ck = torch.load(CK / ("%s.pt" % arch), map_location="cpu")
        m = build_model(arch, len(FEATURES), ck["window"], ck["horizon"], U, ck["pc"])
        m.load_state_dict(ck["state"])
        out[arch] = (m.eval(), ck["norm"])
    return out, W, H


def ci95(x):
    x = pd.Series(x).dropna()
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    exp, warm = load_yaml("experiment"), load_yaml("traffic")["warmup_s"]
    models, W, H = load_models(exp)
    runs = {}
    for d in sorted(RUNS.glob("*/")):
        f = d / "summary.json"
        if f.exists() and (d / "telemetry.csv").exists():
            r = json.loads(f.read_text())
            if r["method"] == SRC and r["scenario"] == SCEN and r["seed"] < 100:
                runs[r["seed"]] = d                       # latest run wins
    if not runs:
        raise SystemExit("no %s runs for %s in %s" % (SRC, SCEN, RUNS))
    rows = []
    for seed, d in sorted(runs.items()):
        X, Y = _windows(_link_arrays(d, warm), W, H, U)
        preds = {"persistence": np.repeat(X[:, -1, U][:, None], H, axis=1)}
        for name, (m, norm) in models.items():
            with torch.no_grad():
                preds[name] = m(torch.from_numpy(normalize(X, norm))).numpy()
        for name, P in preds.items():
            rows.append(dict(seed=seed, model=name, n_windows=len(X), **metrics(Y, P)))
            for h in range(H):
                rows.append(dict(seed=seed, model=name, horizon="t+%d" % (h + 1), n_windows=len(X), **metrics(Y[:, h], P[:, h])))
    df = pd.DataFrame(rows)
    df["horizon"] = df["horizon"].fillna("all")
    df.to_csv(TAB / "prediction_error_per_seed.csv", index=False)

    out = []
    for (mo, hz), g in df.groupby(["model", "horizon"], sort=False):
        row = dict(model=mo, horizon=hz, n_seeds=g.seed.nunique())
        for mt in ("MAE", "RMSE", "MAPE_gt5"):
            row[mt] = "%.4f ± %.4f" % (g[mt].mean(), ci95(g[mt])) if mt != "MAPE_gt5" else "%.2f ± %.2f" % (g[mt].mean(), ci95(g[mt]))
        out.append(row)
    pd.DataFrame(out).to_csv(TAB / "prediction_error_ci.csv", index=False)
    a = df[df.horizon == "all"]
    print(pd.DataFrame([r for r in out if r["horizon"] == "all"]).to_string(index=False))
    print("\npaired MAE difference vs persistence (negative = model better), all horizons:")
    base = a[a.model == "persistence"].set_index("seed").MAE
    for mo in ("lstm", "transformer", "tcn"):
        d = a[a.model == mo].set_index("seed").MAE - base
        print("  %-12s %+.4f ± %.4f  (%+.1f%%, p=%.4f, n=%d)" % (mo, d.mean(), ci95(d), 100 * d.mean() / base.mean(),
                                                               stats.ttest_rel(a[a.model == mo].set_index("seed").MAE, base).pvalue, len(d)))
    print("\nwrote prediction_error_per_seed.csv, prediction_error_ci.csv")


if __name__ == "__main__":
    main()
