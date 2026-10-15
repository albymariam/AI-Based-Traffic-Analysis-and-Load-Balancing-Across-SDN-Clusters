"""Bundle finished runs into compact archives + one flat summary CSV.
   PYTHONPATH=. ~/venvs/ml/bin/python scripts/export_results.py"""
import json
import zipfile

import pandas as pd

from common.config import ROOT

RUNS = ROOT / "results" / "runs"
ARCH, SUMM = ROOT / "results" / "archive", ROOT / "results" / "summaries"
KEEP = ["telemetry.csv", "flows.csv", "probes.csv", "ctrl.csv", "reroutes.csv", "plans.csv",
        "schedule.csv", "summary.json", "meta.json", "traffic.yaml", "topology.yaml", "experiment.yaml"]


def pack(name, dirs):
    p = ARCH / name
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d in dirs:
            for fn in KEEP:
                if (d / fn).exists():
                    z.write(d / fn, arcname="%s/%s" % (d.name, fn))
    mb = p.stat().st_size / 1e6
    print("%-45s %6.1f MB  (%d runs)" % (name, mb, len(dirs)))
    if mb > 90:
        print("  WARNING: GitHub rejects files > 100 MB; split this archive")


def main():
    ARCH.mkdir(parents=True, exist_ok=True)
    SUMM.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(RUNS.glob("*/")):
        f = d / "summary.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["run_dir"] = d.name
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(SUMM / "all_runs_summary.csv", index=False)
    pack("eval_runs_seeds_lt100.zip", [RUNS / r for r in df[df.seed < 100].run_dir])
    pack("training_pool_runs_seeds_ge100.zip", [RUNS / r for r in df[df.seed >= 100].run_dir])

    ev = df[df.seed < 100].sort_values("run_dir")
    dup = ev.groupby(["scenario", "method", "seed"]).size()
    print("\nDuplicate (scenario,method,seed) runs (aggregate keeps the latest):")
    print(dup[dup > 1].to_string() if (dup > 1).any() else "  none")
    cov = ev.drop_duplicates(["scenario", "method", "seed"], keep="last").pivot_table(
        index="scenario", columns="method", values="seed", aggfunc="count").fillna(0).astype(int)
    print("\nCoverage = unique evaluation seeds per cell (want 3 everywhere you report):")
    print(cov.to_string())


if __name__ == "__main__":
    main()