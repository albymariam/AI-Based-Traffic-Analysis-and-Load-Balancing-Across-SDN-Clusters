# M5: real ONOS measurements (Abilene replay, seed 1)

Replay: `data/onos_m5/events_s1.csv` (3907 flow operations: 1800 pins, 307 reroutes, 1800 unpins; 321 s) driven
through the ONOS REST API by `experiments/onos_m5_replay.py`; run by `scripts/onos_m5.sh`; table by
`evaluation/onos_m5_analysis.py` -> `results/tables/onos_m5.csv`.

| Condition | Setup | Status |
|---|---|---|
| A | single ONOS node | valid (1 of 3907 operations failed: dropped connection) |
| B | 3-node cluster, no added delay | valid |
| C10 / C25 / C50 | 3-node cluster, +10 / +25 / +50 ms per hop (tc netem) | valid; C50 propagation check flagged `ok=false` (one pass/fail check of the 30-sample flow test failed; all 30 samples were recorded) |
| C75 | +75 ms per hop | did not converge: the 12 switches were visible on one node only after a 35 min wait |
| C100 | +100 ms per hop | not attempted |

Peer lag (`peer_lag_p95_ms`) is not reliable (negative values: a non-writer node can see a write before the writer).
Report `write_visible_ms_median` instead. `sync_bytes_per_op` is noisy; use `sync_idle_kBps` / `sync_replay_kBps`.
