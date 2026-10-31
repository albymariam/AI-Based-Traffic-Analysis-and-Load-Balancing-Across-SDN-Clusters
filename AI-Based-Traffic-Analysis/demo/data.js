window.DEMO_DATA = {
 "summary": [
  {
   "method": "ecmp",
   "mean": 0.499,
   "p95": 0.695,
   "peak": 0.786,
   "jfi": 0.556,
   "rtt": 4.891
  },
  {
   "method": "lstm_hysteresis",
   "mean": 0.483,
   "p95": 0.677,
   "peak": 0.801,
   "jfi": 0.541,
   "rtt": 4.953
  },
  {
   "method": "persistence_minmax_lp",
   "mean": 0.427,
   "p95": 0.648,
   "peak": 0.766,
   "jfi": 0.568,
   "rtt": 4.897
  },
  {
   "method": "round_robin",
   "mean": 0.623,
   "p95": 0.855,
   "peak": 0.943,
   "jfi": 0.63,
   "rtt": 5.478
  },
  {
   "method": "static_shortest",
   "mean": 0.5,
   "p95": 0.695,
   "peak": 0.791,
   "jfi": 0.535,
   "rtt": 4.915
  },
  {
   "method": "static_tracked",
   "mean": 0.5,
   "p95": 0.697,
   "peak": 0.795,
   "jfi": 0.535,
   "rtt": 4.931
  },
  {
   "method": "tcn_minmax_lp",
   "mean": 0.429,
   "p95": 0.65,
   "peak": 0.779,
   "jfi": 0.567,
   "rtt": 4.923
  },
  {
   "method": "transformer_minmax_lp",
   "mean": 0.426,
   "p95": 0.644,
   "peak": 0.766,
   "jfi": 0.569,
   "rtt": 4.903
  }
 ],
 "prediction": [
  {
   "model": "transformer",
   "mae": 0.012,
   "rmse": 0.0244,
   "smape": 6.1535
  },
  {
   "model": "tcn",
   "mae": 0.0121,
   "rmse": 0.0247,
   "smape": 6.1365
  },
  {
   "model": "lstm",
   "mae": 0.0134,
   "rmse": 0.0259,
   "smape": 6.6974
  },
  {
   "model": "persistence",
   "mae": 0.0124,
   "rmse": 0.0251,
   "smape": 6.3373
  },
  {
   "model": "moving_average",
   "mae": 0.0155,
   "rmse": 0.0298,
   "smape": 7.4952
  }
 ],
 "stale_known": [
  {
   "setting": "static",
   "n": 5,
   "mean": 0.51,
   "p95": 0.666
  },
  {
   "setting": "centralised",
   "n": 5,
   "mean": 0.433,
   "p95": 0.623
  },
  {
   "setting": "dist_s0",
   "n": 10,
   "mean": 0.435,
   "p95": 0.651
  },
  {
   "setting": "dist_s5",
   "n": 10,
   "mean": 0.436,
   "p95": 0.653
  },
  {
   "setting": "dist_s10",
   "n": 10,
   "mean": 0.44,
   "p95": 0.672
  },
  {
   "setting": "dist_s20",
   "n": 10,
   "mean": 0.44,
   "p95": 0.658
  },
  {
   "setting": "dist_s40",
   "n": 10,
   "mean": 0.445,
   "p95": 0.675
  }
 ],
 "stale_est": [
  {
   "setting": "static",
   "n": 5,
   "mean": 0.51,
   "p95": 0.666
  },
  {
   "setting": "centralised",
   "n": 10,
   "mean": 0.458,
   "p95": 0.723
  },
  {
   "setting": "dist_s0",
   "n": 10,
   "mean": 0.455,
   "p95": 0.697
  },
  {
   "setting": "dist_s5",
   "n": 10,
   "mean": 0.457,
   "p95": 0.699
  },
  {
   "setting": "dist_s10",
   "n": 10,
   "mean": 0.456,
   "p95": 0.702
  },
  {
   "setting": "dist_s20",
   "n": 10,
   "mean": 0.46,
   "p95": 0.708
  },
  {
   "setting": "dist_s40",
   "n": 10,
   "mean": 0.464,
   "p95": 0.705
  }
 ],
 "estimated": [
  {
   "metric": "mean_mlu",
   "est": 0.46211399999999997,
   "base": 0.5001580000000001,
   "pct": -7.60639637874434,
   "holm": 0.012268232580011184
  },
  {
   "metric": "p95_mlu",
   "est": 0.7083069999999999,
   "base": 0.695412,
   "pct": 1.8542964458479139,
   "holm": 0.14144396308873033
  },
  {
   "metric": "peak_mlu",
   "est": 0.926961,
   "base": 0.790891,
   "pct": 17.20464640513042,
   "holm": 0.00032093558546845394
  }
 ],
 "rounding": {
  "n": 10,
  "rounding_mean": 6.05138967059044,
  "rounding_sd": 2.2346191332960657,
  "real0_mean": 9.957205934143442,
  "real0_sd": 6.0583279856830385,
  "real6_mean": 13.10658149093384,
  "real6_sd": 8.869989431438725
 },
 "onos": {
  "rows": [
   {
    "cond": "A",
    "delay": 0.0,
    "ok": 99.97,
    "reroute_med": 37.43,
    "reroute_p95": 58.84,
    "write_vis": 20.94642799988833
   },
   {
    "cond": "B",
    "delay": 0.0,
    "ok": 100.0,
    "reroute_med": 50.39,
    "reroute_p95": 81.23,
    "write_vis": 22.367610499941293
   },
   {
    "cond": "C10",
    "delay": 10.0,
    "ok": 100.0,
    "reroute_med": 273.69,
    "reroute_p95": 401.76,
    "write_vis": 110.27397549992202
   },
   {
    "cond": "C25",
    "delay": 25.0,
    "ok": 100.0,
    "reroute_med": 256.95,
    "reroute_p95": 461.32,
    "write_vis": 268.8873335000608
   },
   {
    "cond": "C50",
    "delay": 50.0,
    "ok": 100.0,
    "reroute_med": 1098.58,
    "reroute_p95": 1657.87,
    "write_vis": 525.4964189998645
   }
  ],
  "fit": {
   "slope": 10.150100862994504,
   "intercept": 16.066691286313855,
   "r2": 0.9993335809990365
  }
 },
 "bounds": [
  {
   "policy": "mcf_optimal",
   "ratio": 1.0
  },
  {
   "policy": "kpath_3",
   "ratio": 1.06038
  },
  {
   "policy": "kpath_3_int",
   "ratio": 1.22019
  },
  {
   "policy": "kpath_6",
   "ratio": 1.0
  },
  {
   "policy": "shortest_path",
   "ratio": 1.54425
  },
  {
   "policy": "oblivious_opt",
   "ratio": 1.32837
  },
  {
   "policy": "vlb",
   "ratio": 3.88617
  }
 ],
 "abilene": {
  "week": 24,
  "start_step": 1318,
  "step_seconds": 300,
  "top_pairs": 30,
  "scale": 0.01510112409865019,
  "peak_mbps": 120.0,
  "values": [
   38.1,
   37.74,
   39.99,
   37.75,
   38.47,
   38.8,
   38.75,
   38.64,
   38.82,
   38.65,
   40.91,
   40.19,
   41.2,
   40.61,
   39.88,
   40.85,
   40.63,
   40.04,
   40.96,
   42.23,
   43.65,
   43.45,
   44.58,
   44.52,
   42.62,
   42.04,
   43.13,
   44.29,
   43.31,
   44.49,
   43.04,
   43.94,
   42.64,
   42.12,
   41.82,
   41.98,
   43.08,
   42.93,
   44.34,
   45.48,
   42.59,
   41.47,
   43.12,
   45.76,
   45.54,
   43.25,
   43.1,
   44.68,
   44.67,
   44.68,
   43.73,
   44.01,
   44.28,
   45.4,
   44.93,
   48.14,
   45.6,
   46.63,
   43.21,
   42.62,
   42.55,
   43.99,
   46.75,
   47.88,
   50.68,
   51.85,
   47.34,
   45.94,
   45.06,
   89.62,
   98.26,
   120.0
  ],
  "note": "Aggregate of the top-30 OD pairs, week 24, 6 h window with the largest rise, scaled so the peak is 120 Mbps."
 }
};
