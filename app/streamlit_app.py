"""TimeFlow Streamlit dashboard — forecast viewer with intervals and policy view.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="TimeFlow", layout="wide", page_icon="📈")

st.title("📈 TimeFlow — Demand Forecasting")
st.caption(
    "LightGBM point + quantile forecasts · conformal intervals · newsvendor policy"
)

preds_path = REPORTS / "predictions_test.parquet"
metrics_path = REPORTS / "metrics.csv"
fi_path = REPORTS / "feature_importance.csv"
conf_path = MODELS / "conformal.json"

if not preds_path.exists():
    st.warning(
        "Predictions not found. Run `python -m src.train_all` first to generate "
        f"`{preds_path.relative_to(ROOT)}`."
    )
    st.stop()

preds = pd.read_parquet(preds_path)
preds["date"] = pd.to_datetime(preds["date"])

tabs = st.tabs(["Forecast viewer", "Metrics", "Feature importance", "Newsvendor policy"])

# ---------- Tab 1: Forecast viewer ----------
with tabs[0]:
    c1, c2, c3 = st.columns([1, 1, 2])
    stores = sorted(preds["store"].unique())
    items = sorted(preds["item"].unique())
    store = c1.selectbox("Store", stores, index=0)
    item = c2.selectbox("Item", items, index=0)
    show_interval = c3.radio(
        "Interval", ["Conformal 90%", "Quantile 10–90%", "None"], horizontal=True, index=0
    )

    ser = preds[(preds["store"] == store) & (preds["item"] == item)].sort_values("date")
    if ser.empty:
        st.info("No rows for that store/item.")
    else:
        fig = go.Figure()
        if show_interval == "Conformal 90%":
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([ser["date"], ser["date"][::-1]]),
                    y=pd.concat([ser["conformal_upper"], ser["conformal_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(100, 149, 237, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Conformal 90%",
                    hoverinfo="skip",
                )
            )
        elif show_interval == "Quantile 10–90%":
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([ser["date"], ser["date"][::-1]]),
                    y=pd.concat([ser["q90"], ser["q10"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Quantile 10–90%",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=ser["date"], y=ser["sales"], mode="lines+markers",
                name="Actual", line=dict(color="#111", width=1.5),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ser["date"], y=ser["y_pred"], mode="lines",
                name="Point forecast", line=dict(color="#1f77b4", width=2),
            )
        )
        fig.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Units",
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        residuals = ser["sales"] - ser["y_pred"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{np.mean(np.abs(residuals)):.2f}")
        c2.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.2f}")
        c3.metric(
            "Empirical coverage",
            f"{np.mean((ser['sales'] >= ser['conformal_lower']) & (ser['sales'] <= ser['conformal_upper'])):.1%}",
        )
        c4.metric("Avg interval width", f"{np.mean(ser['conformal_upper'] - ser['conformal_lower']):.2f}")

# ---------- Tab 2: Metrics ----------
with tabs[1]:
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
        if conf_path.exists():
            with open(conf_path) as f:
                cj = json.load(f)
            st.caption(f"Conformal correction q̂ = {cj['q_hat']:.3f} · α = {cj['alpha']}")
    else:
        st.info("No metrics.csv — run training first.")

# ---------- Tab 3: Feature importance ----------
with tabs[2]:
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        fig = go.Figure(
            go.Bar(x=fi["gain"][::-1], y=fi["feature"][::-1], orientation="h",
                   marker_color="#1f77b4")
        )
        fig.update_layout(
            height=max(400, 22 * len(fi)),
            title="Top features by gain",
            xaxis_title="Gain",
            margin=dict(l=140, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature_importance.csv — run training first.")

# ---------- Tab 4: Newsvendor policy ----------
with tabs[3]:
    st.markdown(
        "Pick an ordering policy and see expected profit vs. realized demand. "
        "The **point policy** orders the predicted mean; the **newsvendor policy** "
        "orders the critical-fractile quantile."
    )
    c1, c2, c3, c4 = st.columns(4)
    unit_cost = c1.number_input("Unit cost", 0.0, 100.0, 1.0, 0.1)
    unit_price = c2.number_input("Unit price", 0.0, 100.0, 3.0, 0.1)
    holding_cost = c3.number_input("Holding cost", 0.0, 100.0, 0.2, 0.1)
    stockout_penalty = c4.number_input("Stockout penalty", 0.0, 100.0, 1.5, 0.1)

    margin = unit_price - unit_cost
    if margin + stockout_penalty + holding_cost <= 0:
        st.error("Bad economics: margin + penalty + holding must be > 0.")
    else:
        crit = (margin + stockout_penalty) / (margin + stockout_penalty + holding_cost)
        st.caption(f"Critical fractile = {crit:.2%}")

        qs = np.array([0.1, 0.5, 0.9])
        chosen_q = qs[np.argmin(np.abs(qs - crit))]
        order_nv = np.round(np.clip(preds[f"q{int(chosen_q*100):02d}"].values, 0, None))
        order_pt = np.round(np.clip(preds["y_pred"].values, 0, None))
        y = preds["sales"].values

        def profit(order):
            sold = np.minimum(y, order)
            leftover = np.maximum(order - y, 0)
            short = np.maximum(y - order, 0)
            return (
                unit_price * sold
                - unit_cost * order
                - holding_cost * leftover
                - stockout_penalty * short
            )

        prof_pt = profit(order_pt)
        prof_nv = profit(order_nv)

        c1, c2 = st.columns(2)
        c1.metric(
            "Point policy — mean profit / day-item",
            f"{prof_pt.mean():.2f}",
            f"service {np.mean(y <= order_pt):.1%}",
        )
        c2.metric(
            f"Newsvendor policy (q={chosen_q}) — mean profit",
            f"{prof_nv.mean():.2f}",
            f"service {np.mean(y <= order_nv):.1%}",
        )

        diff = prof_nv.sum() - prof_pt.sum()
        st.metric(
            "Total profit uplift (newsvendor − point)",
            f"{diff:,.0f}",
            f"{(diff / abs(prof_pt.sum()) * 100):.1f}% vs point" if prof_pt.sum() != 0 else None,
        )
