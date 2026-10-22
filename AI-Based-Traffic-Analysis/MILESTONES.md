# T11 — Abilene add-on: milestone guide

Unzip the add-on **into the repo root** (the folder that contains `controller/`, `models/`, `configs/`), then:
`unzip -o t11_abilene_overlay.zip`.  All commands below run from that root.

**Status of this code.** Unit-tested (`pytest`, 9 pass) and run end-to-end on a *synthetic* file set written in the
Abilene format. It has **not** been run on the real Abilene files (the build sandbox cannot reach cs.utexas.edu) or on
Mininet/Ryu. Every number you report must come from your own run. Items marked **CHECK** are things the code verifies or
that you must confirm against `readme.txt`.

| New file | Purpose |
|---|---|
| `scripts/setup_env.sh`, `requirements-abilene.txt` | environment |
| `scripts/download_data.py`, `scripts/verify_data.py` | discover/download/checksum, audit |
| `abilene/{loaders,preprocessing}.py` | parsing, week-wise chronological split, train-only scaler, leak-free windows |
| `abilene/forecasting.py` | persistence, seasonal-naive (only if ACF justifies), LSTM, TCN |
| `abilene/topology_mapping.py`, `configs/topology_abilene12.yaml` | 12-node Abilene → Mininet config |
| `abilene/te_offline.py` | CVXPY min-max + 7 policies on the real Abilene graph (fluid model, **not** Mininet) |
| `abilene/replay.py`, `scripts/apply_abilene_patches.py` | OD demand → Mininet flow schedule; small patches to your repo |

---
## M0 — Environment
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py ml        # PYTHONPATH=. if needed
pytest -q tests                       # 9 passed
```
Done when `check_env` shows no FAIL and `sudo mn --test pingall` gives 0% drop.

## M1 — Data: download, verify, audit
```bash
python scripts/download_data.py            # lists files from the official page, downloads, writes data/manifests/dataset_manifest.json
python scripts/verify_data.py --plots      # → data/processed/abilene/audit.json, results/tables/abilene_audit.md, results/figures/abilene_timeseries.png
cat data/raw/abilene/readme.txt            # CHECK units, column order, citation/reuse terms → copy into the manifest "licence" field
```
Page index lists `X01.gz…X24.gz` (weekly), `A`, `demands`, `links`, `readme.txt`, `topo-2003-04-10.txt`, `AbileneTM-all.tar` (170 MB, not used).
What the audit **detects rather than assumes**: column layout (via rank-1 test of the simple-gravity block), rows/week, NaN/negative,
duplicate consecutive rows, zero fraction, percentiles, IQR outliers (flagged, not removed), autocorrelation, week-over-week correlation, missing week numbers.
**CHECK:** the files carry no timestamps, so dates are *not* reconstructed and weeks are never joined into one continuous series. The 100-byte unit
and the OD index order (source-major, alphabetical routers) come from the readme/assumption — if verify reports otherwise, edit `configs/abilene.yaml`.
Done when `problems: []`.

## M2 — Forecasting (LSTM, TCN vs baselines)
```bash
python -m abilene.forecasting --models persistence seasonal lstm tcn --seeds 1 2 3 4 5     # CPU, ~minutes per model
```
Chronological split by whole weeks (70/15/15), scaler on train only, all models scored on identical test windows, horizon 6×5 min.
Outputs: `results/tables/abilene_forecast_metrics.csv` (MAE, RMSE, MAPE with targets ≤ `zero_eps_mbps` excluded, WAPE; per horizon; peak windows; inference ms),
`results/abilene/forecast/{curve_*.csv, pred_test_*.npy, actual_test.npy}`, checkpoints in `models/checkpoints/abilene/` (git-ignored).
Seasonal-naive runs only if train ACF at lag 288 > 0.5; the script prints the decision. Report mean ± CI over the 5 seeds.

## M3 — Topology mapping + min-max optimisation (offline)
```bash
python -m abilene.topology_mapping        # regenerates configs/topology_abilene12.yaml from the real topo file
python -m abilene.te_offline --forecast lstm_s1 tcn_s1       # add --fractional to score unrounded splits
```
Formulation (`TE.prob`): min U s.t. Σₖₚ A[e,kp]·dₖ·x[kp] ≤ U·cap(e), Σₚ xₖₚ = 1, x ≥ 0 (K=4 loop-free candidate paths by OSPF weight, CVXPY/CLARABEL, DPP so it compiles once).
Solution is **fractional**; the default scoring rounds each OD to its largest-share path (installable as a plain OpenFlow route) and applies hysteresis η.
Policies: shortest_path, ecmp (equal hop count), round_robin (fluid equal split), reactive, `<model>_heuristic`, `<model>_opt`, `oracle_opt` (upper bound).
Decisions use only past/forecast demand; every policy is scored on the actual demand. **Assumption to state in the report:** raw Abilene links are lightly loaded, so demand is
scaled so the *train-set* peak MLU = `te.target_peak_mlu`; results are for that stress level, not for 2004 absolute loads.
Output: `results/tables/abilene_te_offline.csv`. Done when `solver_fallbacks = 0`.

## M4 — Mininet + Ryu replay (measured emulation)
```bash
python scripts/apply_abilene_patches.py            # idempotent; adds T11_TOPOLOGY switch + --schedule to run_baseline.py
export T11_TOPOLOGY=abilene12
python -m abilene.replay                           # → results/abilene/replay_schedule.{csv,json}  (coverage + scale printed)
# terminal 1
T11_TOPOLOGY=abilene12 scripts/start_ryu.sh
# terminal 2 — connectivity check, then one run
sudo -E env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/run_baseline.py --method static_shortest --load abilene --profile replay --seed 1 --check
for m in static_shortest ecmp round_robin; do for s in 1 2 3 4 5; do sudo mn -c >/dev/null 2>&1
  sudo -E env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/run_baseline.py \
       --method $m --load abilene --profile replay --seed $s --duration 300 --schedule results/abilene/replay_schedule.csv \
       || echo "FAILED $m $s" >> results/failed_runs.log; done; done
```
Forecast-driven methods need predictors trained on **this** topology's telemetry (your current checkpoints are NSFNET): collect with `static_tracked` runs at seeds ≥100
(`--method static_tracked --seed 101…`; use the synthetic generator, i.e. omit `--schedule`), then `PYTHONPATH=. .venv/bin/python models/train.py && … models/train_m2.py`,
and run `--method lstm_hysteresis | tcn_minmax_lp | transformer_minmax_lp` with the same `--schedule`. Keep the same trace for every method (paired design).
Mapping to document: Mbps = units×100 B×8/300 s; 300 s → 5 s (60× compression); one global scale to reach `target_mlu` under static shortest path; top-30 OD pairs only (coverage in the JSON);
UDP iperf3 flows (offered rate fixed); **Mininet links are uniform 10 Mbps, so the two 2.48 Gbps ATLAng–IPLSng links are not 4× slower** (offline model keeps them); 1 ms uniform delay; hop-count forwarding.
Offered vs carried: offered = schedule rates; carried = per-flow iperf3 results (`flows.csv`) and port counters (`telemetry.csv`).

## M5 — Controller-state staleness (primary contribution)
```bash
for d in 0 1 2 5 10 20; do for s in 1 2 3 4 5; do sudo mn -c >/dev/null 2>&1
  sudo -E env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/run_baseline.py \
     --method dist_lp_s$d --load abilene --profile replay --seed $s --duration 300 --schedule results/abilene/replay_schedule.csv; done; done
PYTHONPATH=. .venv/bin/python evaluation/staleness_analysis.py
```
**Label honestly:** `controller/dist_lp_policy.py` is a *controller-independent state-staleness emulation* (3 logical domains, delayed telemetry/flow-placement views, staggered independent decisions).
It is **not** a native ONOS/ODL cluster. `controller/onos_adapter.py` is empty, so a native-cluster experiment does not exist yet. Before the report: add per-decision logging of
controller id, state version, generation/availability/decision/install timestamps (your `plans.csv` has `ctrl`, `view_age_s`, not the full versioned record), and sweep ≥ the measured propagation range.

## M6 — Statistics, figures, report
```bash
PYTHONPATH=. .venv/bin/python evaluation/aggregate.py        # mean ± 95% CI, paired tests vs static_shortest, bars, CDFs
PYTHONPATH=. .venv/bin/python scripts/export_results.py
```
Before citing: `results/_mockup_DO_NOT_CITE/`, `experiments/_mockup_run_m1.py`, and `evaluation/plot_results.py` (mock-data plots) must not feed the report. Empty stubs still to fill:
`experiments/run_m2.py`, `models/predict.py`, `controller/onos_adapter.py`, `evaluation/plots.py`. Missing from the brief: Jain/flow-mod rate/FCT tables from the new runs, ablations
(hysteresis η, K, horizon, forecast vs oracle), literature matrix (≥25 papers, ≥70% from the last 5 years — verify each DOI yourself).

## M7 — Optional
* **ONOS (single node, to start an adapter):** `docker run -d --name onos -p 8181:8181 -p 6653:6653 onosproject/onos:2.7-latest` (CHECK tag), REST `http://localhost:8181/onos/v1` (default `onos/rocks`), activate `org.onosproject.openflow`. Cluster mode and flow-install adapter are untested here.
* **GÉANT:** verify provenance/terms at https://totem.info.ucl.ac.be/dataset.html first; add a loader returning the same `{week: (T,P)}` dict and rerun M2/M3 unchanged.
