from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import default_sample_data, load_csv, load_from_url
from src.features import build_feature_frame
from src.metrics import directional_accuracy, mae, mape, peak_load_error, rmse
from src.model import scenario_temperature_adjustment, train_and_forecast

st.set_page_config(page_title="PJM Power & Demand Forecasting Dashboard", layout="wide")
st.title("⚡ PJM Power & Demand Forecasting Dashboard")
st.caption("Upload demand/weather data, train a simple model, and explore forecast performance.")

with st.sidebar:
    st.header("Data Input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    url = st.text_input("Optional CSV URL", value="")
    horizon_label = st.selectbox("Forecast Horizon", ["Next 24 hours", "Next 7 days"], index=0)
    horizon_hours = 24 if horizon_label == "Next 24 hours" else 168

@st.cache_data(show_spinner=False)
def load_data(uploaded_file, url_text: str):
    if uploaded_file is not None:
        return load_csv(uploaded_file)
    if url_text.strip():
        return load_from_url(url_text.strip())
    return default_sample_data()


df = load_data(uploaded, url)
if df is None or df.empty:
    st.error("No data loaded. Upload a CSV or place sample_data.csv in project root.")
    st.stop()

if "load_mw" not in df.columns or df["load_mw"].dropna().empty:
    st.error("The dataset must include a usable load column (load_mw or alias load/demand).")
    st.stop()

feat_df = build_feature_frame(df)
if len(feat_df) < 200:
    st.warning("Dataset is quite small after lag features; results may be unstable.")

art = train_and_forecast(feat_df, horizon_hours=horizon_hours)
y_true = art.test_df["load_mw"].values
y_pred = art.y_pred

metrics_model = {
    "MAE": mae(y_true, y_pred),
    "RMSE": rmse(y_true, y_pred),
    "MAPE %": mape(y_true, y_pred),
    "Peak Error MW": peak_load_error(y_true, y_pred),
    "Directional Acc. %": directional_accuracy(y_true, y_pred),
}

tabs = st.tabs(["Data Overview", "Demand Forecast", "Forecast Performance", "Scenario / Weather Sensitivity"])

with tabs[0]:
    st.subheader("Data Overview")
    st.dataframe(df.head(20), use_container_width=True)

    fig_hist = px.line(df, x="datetime", y="load_mw", color="zone", title="Historical Demand (MW)")
    st.plotly_chart(fig_hist, use_container_width=True)

    heat_df = df.copy()
    heat_df["hour"] = heat_df["datetime"].dt.hour
    heat_df["day_of_week"] = heat_df["datetime"].dt.dayofweek
    pivot = heat_df.pivot_table(values="load_mw", index="day_of_week", columns="hour", aggfunc="mean")
    fig_heat = px.imshow(pivot, aspect="auto", title="Average Demand Heatmap (Day x Hour)", labels={"x": "Hour", "y": "Day of Week"})
    st.plotly_chart(fig_heat, use_container_width=True)

with tabs[1]:
    st.subheader("Demand Forecast")
    pred_df = art.test_df[["datetime", "load_mw"]].copy()
    pred_df["forecast_mw"] = y_pred
    pred_df["baseline_24h"] = art.baseline_24h
    pred_df["baseline_168h"] = art.baseline_168h

    fig_fcst = go.Figure()
    fig_fcst.add_trace(go.Scatter(x=pred_df["datetime"], y=pred_df["load_mw"], name="Actual"))
    fig_fcst.add_trace(go.Scatter(x=pred_df["datetime"], y=pred_df["forecast_mw"], name="RF Forecast"))
    fig_fcst.add_trace(go.Scatter(x=pred_df["datetime"], y=pred_df["baseline_24h"], name="Baseline Lag24"))
    fig_fcst.add_trace(go.Scatter(x=pred_df["datetime"], y=pred_df["baseline_168h"], name="Baseline Lag168"))
    fig_fcst.update_layout(title="Actual vs Forecast")
    st.plotly_chart(fig_fcst, use_container_width=True)

    cols = st.columns(5)
    metric_keys = ["MAE", "RMSE", "MAPE %", "Peak Error MW", "Directional Acc. %"]
    for c, k in zip(cols, metric_keys):
        c.metric(k, f"{metrics_model[k]:,.2f}")

with tabs[2]:
    st.subheader("Forecast Performance")
    err_df = art.test_df[["datetime"]].copy()
    err_df["error_mw"] = y_pred - y_true
    err_df["abs_error_mw"] = np.abs(err_df["error_mw"])

    fig_err = px.line(err_df, x="datetime", y="error_mw", title="Forecast Error Over Time (MW)")
    st.plotly_chart(fig_err, use_container_width=True)

    importances = pd.DataFrame({
        "feature": art.train_df.drop(columns=["datetime", "zone", "load_mw"]).columns,
        "importance": art.model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fig_imp = px.bar(importances.head(15), x="importance", y="feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

with tabs[3]:
    st.subheader("Scenario / Weather Sensitivity")
    delta = st.slider("Adjust temperature assumption (°F)", min_value=-15, max_value=15, value=0, step=1)

    scenario_pred = scenario_temperature_adjustment(art.test_df, art.model, delta_temp=float(delta))
    scen_df = art.test_df[["datetime", "load_mw"]].copy()
    scen_df["base_forecast"] = y_pred
    scen_df["scenario_forecast"] = scenario_pred

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=scen_df["datetime"], y=scen_df["load_mw"], name="Actual"))
    fig_scen.add_trace(go.Scatter(x=scen_df["datetime"], y=scen_df["base_forecast"], name="Base Forecast"))
    fig_scen.add_trace(go.Scatter(x=scen_df["datetime"], y=scen_df["scenario_forecast"], name=f"Scenario Forecast ({delta:+}°F)"))
    fig_scen.update_layout(title="Temperature Sensitivity")
    st.plotly_chart(fig_scen, use_container_width=True)

    impact = float(np.mean(scenario_pred - y_pred))
    st.metric("Avg Forecast Change (MW)", f"{impact:,.2f}")
