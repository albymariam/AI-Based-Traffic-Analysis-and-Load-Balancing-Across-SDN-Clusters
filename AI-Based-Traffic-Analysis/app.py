import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="AI-SDN Traffic Management Demo", page_icon="🌐", layout="wide")

st.title("AI-Based Traffic Analysis & Load Balancing Across SDN Clusters")
st.caption("Min-max LP traffic engineering under controller-state staleness | "
           "Tab 1 = simulation, Tab 2 = recorded experiments")

ROOT = Path(__file__).resolve().parent
CAP = 100  # Mbps per path in the simulation


def locate_csv(filename):
    for base in (ROOT, Path.cwd()):
        for sub in ("results/tables", "AI-Based-Traffic-Analysis/results/tables"):
            p = base / sub / filename
            if p.exists():
                return p
    return None


def load_csv(filename):
    p = locate_csv(filename)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def num(value):
    """'0.500 ± 0.157' -> 0.5"""
    try:
        return float(str(value).split("±")[0].strip())
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------- simulation
def make_demand(n, base, spike, noise, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    d = np.full(n, float(base))
    ramp_start, peak, ramp_end = int(n * 0.30), int(n * 0.45), int(n * 0.70)
    d[ramp_start:peak] = np.linspace(base, spike, peak - ramp_start)
    d[peak:ramp_end] = spike
    d[ramp_end:ramp_end + 8] = np.linspace(spike, base, 8)[: len(d[ramp_end:ramp_end + 8])]
    d = d + rng.normal(0, noise, n)
    return t, np.clip(d, 5, None)


def forecast(kind, hist):
    """Predict the next demand value from past values only."""
    if len(hist) == 0:
        return 0.0
    last = hist[-1]
    if kind == "Persistence (last value)":
        return last
    if kind == "Trend (linear)":
        return max(last + (last - hist[-2]), 0) if len(hist) > 1 else last
    if kind == "Moving average (3)":
        return float(np.mean(hist[-3:]))
    return last


def split_fraction(f, bg_a, bg_b):
    """Fraction of new traffic sent on A so both paths end equally loaded (min-max)."""
    if f <= 0:
        return 1.0
    to_a = np.clip((bg_b + f - bg_a) / 2, 0, f)
    return to_a / f


def route(demand, frac_a, n_flows):
    """Return (load_a, load_b). With n_flows, a flow cannot be split."""
    if n_flows <= 1:
        k = round(frac_a)  # one flow: all on A or all on B
        return demand * k, demand * (1 - k)
    k = round(frac_a * n_flows)
    return demand * k / n_flows, demand * (n_flows - k) / n_flows


tab1, tab2, tab3 = st.tabs(["1 · Traffic spike simulation", "2 · Recorded experiment results", "3 · How it works"])

with tab1:
    st.subheader("A traffic spike hits one path. What does each policy do?")
    st.info("**Simulation.** Demand is synthetic. The forecasters here are simple "
            "(they use only past demand). The trained LSTM/TCN/Transformer and the real controller "
            "are **not** run here; see Tab 2 for the measured results.")

    c1, c2, c3, c4 = st.columns(4)
    base = c1.slider("Normal demand (Mbps)", 10, 100, 50, 5)
    spike = c2.slider("Spike demand (Mbps)", 50, 180, 110, 5)
    bg_a = c3.slider("Background on Path A (Mbps)", 0, 60, 30, 5)
    bg_b = c4.slider("Background on Path B (Mbps)", 0, 60, 20, 5)

    c5, c6, c7, c8 = st.columns(4)
    fc_kind = c5.selectbox("Forecaster", ["Persistence (last value)", "Trend (linear)", "Moving average (3)"])
    noise = c6.slider("Traffic noise (Mbps)", 0, 20, 6, 2)
    n_flows = c7.slider("Flows (1 = cannot split)", 1, 20, 20, 1,
                        help="Real flows follow one path each. Few flows means coarse splits.")
    seed = c8.number_input("Random seed", 0, 999, 1)

    N = 60
    t, D = make_demand(N, base, spike, noise, seed)

    static_a = bg_a + D
    static_b = np.full(N, float(bg_b))
    bal_a, bal_b, orc_a, orc_b, fcs = [], [], [], [], []
    for i in range(N):
        f = forecast(fc_kind, D[:i])
        f = f if i > 0 else D[0]
        fcs.append(f)
        # decision uses the forecast; the load that arrives is the REAL demand
        la, lb = route(D[i], split_fraction(f, bg_a, bg_b), n_flows)
        bal_a.append(bg_a + la)
        bal_b.append(bg_b + lb)
        la, lb = route(D[i], split_fraction(D[i], bg_a, bg_b), n_flows)
        orc_a.append(bg_a + la)
        orc_b.append(bg_b + lb)

    mlu_static = np.maximum(static_a, static_b) / CAP
    mlu_bal = np.maximum(bal_a, bal_b) / CAP
    mlu_orc = np.maximum(orc_a, orc_b) / CAP

    fig = go.Figure()
    fig.add_scatter(x=t, y=mlu_static * 100, name="Static shortest path", line=dict(color="#d62728"))
    fig.add_scatter(x=t, y=mlu_bal * 100, name=f"Forecast-informed LP ({fc_kind})", line=dict(color="#1f77b4"))
    fig.add_scatter(x=t, y=mlu_orc * 100, name="Perfect forecast (reference)",
                    line=dict(color="#7f7f7f", dash="dot"))
    fig.add_hline(y=100, line_dash="dash", annotation_text="Link capacity")
    fig.update_layout(title="Maximum link utilization over time (lower is better)",
                      xaxis_title="Time step", yaxis_title="MLU (%)", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    for col, name, mlu in ((m1, "Static", mlu_static), (m2, "Forecast-informed", mlu_bal),
                           (m3, "Perfect forecast", mlu_orc)):
        col.markdown(f"**{name}**")
        col.metric("Mean MLU", f"{mlu.mean()*100:.1f}%")
        col.metric("Peak MLU", f"{mlu.max()*100:.1f}%")
        col.metric("Steps above 100%", f"{int((mlu > 1).sum())} of {N}")

    st.markdown("#### Inspect one moment")
    tick = st.slider("Time step", 0, N - 1, int(N * 0.55))
    a, b = st.columns([1, 2])
    with a:
        st.metric("Real demand", f"{D[tick]:.0f} Mbps")
        st.metric("Forecast used", f"{fcs[tick]:.0f} Mbps", delta=f"{fcs[tick]-D[tick]:+.0f} vs real",
                  delta_color="off")
    with b:
        bar = pd.DataFrame({
            "Path": ["Path A", "Path B"] * 2,
            "Policy": ["Static"] * 2 + ["Forecast-informed"] * 2,
            "Utilization (%)": [static_a[tick], static_b[tick], bal_a[tick], bal_b[tick]],
        })
        fb = px.bar(bar, x="Path", y="Utilization (%)", color="Policy", barmode="group",
                    color_discrete_map={"Static": "#d62728", "Forecast-informed": "#1f77b4"})
        fb.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fb, use_container_width=True)

    st.caption("Try: switch the forecaster (the curves barely move: the optimisation does the work, "
               "not the forecast). Set flows to 1-3 to see the single-path rounding cost. "
               "If total demand exceeds both paths together, routing cannot remove congestion.")

with tab2:
    st.subheader("Measured results (Abilene traffic replayed in Mininet)")
    st.caption("Setup: 60x compressed replay, top-30 OD pairs, 10 Mbps links, 10 test seeds, "
               "min-max LP over K=3 paths every 2 s. Paired t-tests with Holm correction.")

    st.markdown("#### Main result: routing methods on Abilene replay")
    summ = load_csv("summary_mean_ci.csv")
    names = {"static_shortest": "Static shortest path", "ecmp": "ECMP", "round_robin": "Round robin",
             "lstm_hysteresis": "LSTM + hysteresis (M1)", "persistence_minmax_lp": "Persistence + LP",
             "tcn_minmax_lp": "TCN + LP", "transformer_minmax_lp": "Transformer + LP"}
    if summ is not None and "scenario" in summ.columns:
        r = summ[(summ["scenario"] == "abilene_replay") & summ["method"].isin(names)].copy()
        r["Method"] = r["method"].map(names)
        r["Mean MLU"] = r["mean_mlu"].apply(num)
        r["Peak MLU"] = r["peak_mlu"].apply(num) if "peak_mlu" in r.columns else np.nan
        r = r.sort_values("Mean MLU")
        long = r.melt(id_vars="Method", value_vars=["Mean MLU", "Peak MLU"], var_name="Metric", value_name="MLU")
        st.plotly_chart(px.bar(long, x="Method", y="MLU", color="Metric", barmode="group",
                               title="Lower is better (10 seeds). The three LP forecasters are almost identical."),
                        use_container_width=True)
    else:
        main = pd.DataFrame({"Method": ["Static shortest path", "Persistence + LP", "TCN + LP", "Transformer + LP"],
                             "Mean MLU change vs static (%)": [0.0, -14.6, -14.2, -14.8]})
        st.plotly_chart(px.bar(main, x="Method", y="Mean MLU change vs static (%)"), use_container_width=True)
        st.caption("summary_mean_ci.csv not found: showing the headline numbers from the paper.")

    pe = load_csv("prediction_error_m2.csv")
    if pe is not None and "horizon" in pe.columns:
        pa = pe[pe["horizon"].astype(str) == "all"]
        st.plotly_chart(px.bar(pa, x="model", y="MAE", title="Forecast error (MAE, lower is better): "
                               "persistence is almost as good as the neural models"),
                        use_container_width=True)

    st.markdown(
        "- Static **0.500** → Transformer LP **0.426** (-14.8%). Persistence is within 0.2% of Transformer (p = 0.17).\n"
        "- **The LP gives the gain, not the forecaster.** ECMP is no better than static here (0.499).\n"
        "- With demand **estimated from OpenFlow counters**: mean MLU -7.6% (about half the gain), "
        "but peak MLU is 13-17% worse.\n"
        "- Single-path rounding costs about 6% over the fractional LP; the load carried is about 10% above the plan.")

    bnd = load_csv("abilene_te_bounds.csv")
    if bnd is not None and "mean_ratio_to_opt" in bnd.columns:
        keep = ["shortest_path", "oblivious_opt", "vlb", "kpath_3", "kpath_3_int", "kpath_6", "mcf_optimal"]
        bb = bnd[bnd["policy"].isin(keep)]
        st.plotly_chart(px.bar(bb, x="policy", y="mean_ratio_to_opt",
                               title="Offline: mean MLU relative to the optimal (1.0 = optimal). "
                                     "K=3 paths are close; rounding to one path (_int) costs most"),
                        use_container_width=True)

    st.markdown("#### Archived tables from the repository")
    st.caption("Read directly from results/tables. Missing files are skipped.")
    shown = 0
    tables = [
        ("Routing comparison", "summary_mean_ci.csv"),
        ("Prediction error", "prediction_error_m2.csv"),
        ("Offline bounds (ratio to optimal)", "abilene_te_bounds.csv"),
        ("Offline policies", "abilene_te_offline.csv"),
        ("Staleness (known demand)", "staleness_summary.csv"),
        ("Staleness (estimated demand)", "staleness_summary_est.csv"),
        ("Estimated vs known demand, paired", "est_paired_alby.csv"),
        ("Rounding gap and realised load", "est_realised_gap.csv"),
    ]
    for title, fname in tables:
        df = load_csv(fname)
        if df is None:
            continue
        shown += 1
        with st.expander(f"{title}  ·  {fname}", expanded=(fname == "summary_mean_ci.csv")):
            if fname == "summary_mean_ci.csv" and "scenario" in df.columns:
                df = df[df["scenario"] == "abilene_replay"]
            if fname == "prediction_error_m2.csv" and "horizon" in df.columns:
                df = df[df["horizon"].astype(str) == "all"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            # simple chart when a clear label + numeric column exists
            try:
                for col in ("mean_mlu", "mean_MLU", "MAE"):
                    if col in df.columns:
                        label = next(c for c in ("method", "policy", "model", "setting") if c in df.columns)
                        plot = df.assign(value=df[col].apply(num))
                        st.plotly_chart(px.bar(plot, x=label, y="value", title=f"{col} (lower is better)"),
                                        use_container_width=True)
                        break
            except Exception:
                pass
    if shown == 0:
        st.warning("No CSV tables found. Put app.py in the project root (next to results/), "
                   "or run it from there. The summary above is still valid.")

    st.markdown("#### Controller staleness and ONOS cluster")
    st.markdown(
        "- Known demand: no loss up to **20 s** of stale state; at 40 s mean MLU is +2.2% (Holm p = 0.025).\n"
        "- ONOS 3-node cluster: write visibility ≈ **16.1 + 10.15·d ms** (d = added delay per hop, R² = 0.999).")

with tab3:
    st.subheader("Pipeline")
    stages = [
        ("1. Telemetry", "Read link and flow byte counters from the switches (OpenFlow)."),
        ("2. Demand estimate", "Turn flow counters into a traffic matrix (MAPE about 9.5%)."),
        ("3. Forecast", "LSTM / TCN / Transformer predict the next demand; persistence is the baseline."),
        ("4. Optimise", "Min-max LP over K=3 candidate paths minimises the busiest link."),
        ("5. Install", "Ryu controller writes the chosen paths as forwarding rules (hysteresis 0.05, cooldown 5 s)."),
        ("6. Evaluate", "MLU, p95, Jain fairness, RTT, in 10 seeds with paired tests."),
        ("7. Distributed study", "How stale state and delay between controllers (ONOS cluster) affect decisions."),
    ]
    for title, desc in stages:
        st.markdown(f"**{title}**  \n{desc}")

    st.markdown("#### What is simulated and what is measured")
    st.table(pd.DataFrame({
        "Part": ["Tab 1 curves", "Tab 2 numbers", "Mininet + Ryu replay", "ONOS cluster"],
        "Status": ["Synthetic simulation (illustration)", "Recorded from real runs",
                   "Real emulation (run live or shown as recording)", "Real emulation, separate from the LP loop"],
    }))
    st.caption("Limits: the LP loop was not closed inside ONOS; select groups and staleness inside ONOS "
               "are future work.")
