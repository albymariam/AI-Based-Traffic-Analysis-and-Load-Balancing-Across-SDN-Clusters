"""Validate downloaded files against the manifest, DETECT the column layout, and write the data audit.
   python scripts/verify_data.py [--plots]  ->  data/processed/abilene/audit.json, results/tables/abilene_audit.md"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abilene.loaders import N_OD, detect_layout, read_week, split_layout, week_files  # noqa: E402
from abilene.preprocessing import od_columns, to_mbps  # noqa: E402
from scripts.download_data import sha256  # noqa: E402


def acf(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / (x * x).sum()) if len(x) > lag else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))
    dc = cfg["data"]
    raw_dir = ROOT / dc["raw_dir"]
    man = {e["file"]: e for e in json.load(open(ROOT / "data/manifests/dataset_manifest.json"))["files"]}
    problems, weeks, layout, scores = [], {}, None, None
    for p in week_files(raw_dir):
        if p.name not in man or man[p.name]["sha256"] != sha256(p):
            problems.append("%s: checksum missing/mismatch vs manifest" % p.name)
        try:
            raw = read_week(p)
        except Exception as e:                         # truncated gz, ragged rows ...
            problems.append("%s: unreadable (%r)" % (p.name, e))
            continue
        if layout is None:
            layout, scores = detect_layout(raw) if dc["layout"] == "auto" else (dc["layout"], None)
        weeks[int(p.name[1:3])] = split_layout(raw, layout)[:, :, 0]
    if not weeks:
        sys.exit("no readable weeks: " + "; ".join(problems))

    rows = {k: len(v) for k, v in weeks.items()}
    allx = np.vstack([weeks[k] for k in sorted(weeks)])
    cols = od_columns(dc["drop_diagonal"])
    mbps = to_mbps(allx[:, cols], dc["bytes_per_unit"], dc["interval_s"])
    tot = to_mbps(allx[:, cols].sum(1), dc["bytes_per_unit"], dc["interval_s"])      # same pairs as the models (no diagonal)
    diag = to_mbps(allx[:, np.setdiff1d(np.arange(N_OD), cols)], dc["bytes_per_unit"], dc["interval_s"])
    dup = int(sum((np.diff(weeks[k], axis=0) == 0).all(1).sum() for k in weeks))
    q1, q3 = np.nanpercentile(mbps, 25, 0), np.nanpercentile(mbps, 75, 0)
    outl = int((mbps > q3 + 3 * (q3 - q1)).sum())
    wk = sorted(weeks)
    w2w = [float(np.corrcoef(weeks[a_][:, cols].sum(1), weeks[b_][:, cols].sum(1))[0, 1]) for a_, b_ in zip(wk, wk[1:])
           if rows[a_] == rows[b_] and b_ == a_ + 1]
    audit = {
        "weeks_present": wk, "weeks_missing_from_1_to_max": [w for w in range(1, max(wk) + 1) if w not in weeks],
        "rows_per_week": rows, "n_snapshots": int(allx.shape[0]), "columns_per_row": 720,
        "n_nodes": 12, "n_od_pairs": N_OD, "n_od_pairs_used": int(len(cols)), "matrix_dims": "12x12 per snapshot",
        "layout_detected": layout, "layout_rank1_scores": scores,
        "sampling_interval_s": dc["interval_s"],
        "timestamps": "NOT stored in the files. Week index from filename only; calendar dates come from readme.txt "
                      "(NOT verified here). Weeks are treated as separate segments; no contiguity assumed.",
        "missing_values_nan": int(np.isnan(allx).sum()), "negative_values": int((allx < 0).sum()),
        "duplicate_consecutive_snapshots": dup, "duplicate_timestamps": "n/a (no timestamps in files)",
        "zero_demand_fraction": float((mbps == 0).mean()),
        "units": "file units x %d bytes / %ds -> Mbps (unit definition from readme.txt, confirm)" % (dc["bytes_per_unit"], dc["interval_s"]),
        "demand_mbps": {"median": float(np.median(mbps)), "mean": float(mbps.mean()), "p95": float(np.percentile(mbps, 95)),
                        "p99": float(np.percentile(mbps, 99)), "max": float(mbps.max())},
        "total_demand_mbps": {"median": float(np.median(tot)), "p99": float(np.percentile(tot, 99)), "peak": float(tot.max())},
        "diagonal_self_pairs_mbps": ({"median": float(np.median(diag)), "max": float(diag.max())} if diag.size else None),
        "outliers_iqr3_count": outl, "outlier_treatment": "flagged only, NOT removed or clipped",
        "acf_total_demand": {str(l): acf(tot[:rows[wk[0]]], l) for l in (1, 12, 288)},
        "week_over_week_corr_total": {"mean": float(np.mean(w2w)) if w2w else None, "n": len(w2w)},
        "od_order_assumption": "src-major alphabetical routers; UNVERIFIED (swap would transpose OD pairs, diagonal unaffected)",
        "problems": problems,
    }
    out = ROOT / dc["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    md = ["# Abilene data audit (generated)\n"] + ["- **%s**: %s" % (k, json.dumps(v)) for k, v in audit.items()]
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "abilene_audit.md").write_text("\n".join(md))
    print(json.dumps({k: audit[k] for k in ("n_snapshots", "layout_detected", "weeks_missing_from_1_to_max",
                                              "missing_values_nan", "zero_demand_fraction", "problems")}, indent=2))
    if a.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 3)); plt.plot(tot, lw=.4); plt.ylabel("total demand (Mbps)")
        plt.xlabel("snapshot index (weeks concatenated)"); plt.tight_layout()
        plt.savefig(ROOT / "results" / "figures" / "abilene_timeseries.png", dpi=150)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
