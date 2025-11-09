import streamlit as st
import pandas as pd
import datetime
import joblib
import os

from AQIAgent import (
    get_dynamic_date_ranges,
    get_sensor_data_over_period,
    get_weather_data,
    get_weather_forecast,
    enrich_features,
    upload_to_hopsworks,
    train_models,
    predict_next_3_days,
    CITY, COUNTRY
)

st.set_page_config(page_title="Karachi AQI Dashboard", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center;">🌍 Karachi Air Quality Dashboard</h1>
    <p style="text-align:center; font-size:18px;">
    End‑to‑end pipeline: feature engineering → training → 3‑day forecast
    </p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Configuration")
    train_days = st.slider("Training window (days)", 7, 60, 30)
    forecast_days = st.slider("Forecast horizon (days)", 1, 7, 3)
    st.write(f"📍 City: {CITY}, Country: {COUNTRY}")

    selected_model = st.selectbox(
        "Choose model for forecast",
        ["RandomForest", "Ridge", "XGBoost"],
        index=0
    )

ranges = get_dynamic_date_ranges(train_days=train_days, forecast_days=forecast_days)

tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Model", "🔮 Forecast"])

if st.button("🚀 Run pipeline"):
    with st.spinner("Fetching pollutant + weather data..."):
        pollutant_df = get_sensor_data_over_period(
            location_id=4837117,
            date_from=ranges["train_start_iso"],
            date_to=ranges["train_end_iso"],
            page_limit=1000,
            max_pages=5
        )
        weather_df = get_weather_data(
            start_date=ranges["train_start_day"],
            end_date=ranges["train_end_day"],
            hourly=True
        )

    if pollutant_df.empty:
        st.error("No pollutant data fetched. Check location_id/date range.")
        st.stop()

    pollutant_df = enrich_features(pollutant_df.copy())
    pollutant_df["timestamp"] = pd.to_datetime(pollutant_df["timestamp"], errors="coerce").dt.tz_localize(None)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce").dt.tz_localize(None)
    pollutant_df.dropna(subset=["timestamp"], inplace=True)
    weather_df.dropna(subset=["timestamp"], inplace=True)

    train_start = pd.to_datetime(ranges["train_start_day"])
    train_end = pd.to_datetime(ranges["train_end_day"]) + pd.Timedelta(days=1)
    pollutant_df = pollutant_df[(pollutant_df["timestamp"] >= train_start) & (pollutant_df["timestamp"] < train_end)]
    pollutant_df["timestamp"] = pollutant_df["timestamp"].dt.floor("H")
    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("H")

    merged_df = pollutant_df.merge(weather_df, on="timestamp", how="left", suffixes=("", "_weather"))
    merged_df = merged_df.dropna(subset=["temperature", "humidity", "pressure", "wind_speed", "weathercode"])

    #Tab 1: Data
    with tab1:
        st.success("✅ Data prepared")
        st.subheader("Sample processed training data")
        st.dataframe(merged_df.head(), width="stretch")

        st.subheader("📈 EDA: AQI Trends")
        st.line_chart(merged_df.set_index("timestamp")["value"])

        st.subheader("🔥 Worst AQI Hours")
        top_aqi = merged_df.sort_values("value", ascending=False).head(5)
        st.dataframe(top_aqi[["timestamp", "value", "parameter"]], width="stretch")
        st.line_chart(top_aqi.set_index("timestamp")["value"])

        st.subheader("📉 Correlation with Weather")
        corr_df = merged_df[["value", "temperature", "humidity", "pressure", "wind_speed"]].corr()
        st.dataframe(corr_df, width="stretch")

        if st.checkbox("Upload to Hopsworks Feature Store", value=False):
            with st.spinner("Uploading features..."):
                upload_to_hopsworks(merged_df)
            st.success("Uploaded to Hopsworks.")

    #Tab 2: Model
    if "pm25" in merged_df["parameter"].unique():
        target_param = "pm25"
    else:
        candidates = [p for p in merged_df["parameter"].unique() if p.lower() in ["pm25", "pm10", "pm1"]]
        target_param = candidates[0] if candidates else merged_df["parameter"].iloc[0]

    with tab2:
        st.subheader("Model Training")
        with st.spinner(f"Training models for target: {target_param}"):
            results = train_models(merged_df, target_param=target_param, selected_model=selected_model)
        st.success("Training complete.")

        metrics_df = pd.DataFrame(results).rename_axis("Metric").reset_index()
        st.dataframe(metrics_df, width="stretch")

        st.subheader("Feature Importance (SHAP)")
        shap_path = f"{selected_model.lower()}_shap_summary.png"
        if os.path.exists(shap_path):
            st.image(shap_path, caption=f"{selected_model} SHAP Summary")
        else:
            st.info("SHAP summary not available for this model.")

    #Tab 3: Forecast
    with tab3:
        st.subheader(f"3‑Day AQI Forecast ({selected_model})")
        with st.spinner("Fetching forecast + predicting..."):
            forecast_df = get_weather_forecast(
                start_day=ranges["forecast_start_day"],
                end_day=ranges["forecast_end_day"]
            )
            hourly_preds, daily_preds = predict_next_3_days(
                models_paths={
                    "randomforest": "randomforest_model.pkl",
                    "ridge": "ridge_model.pkl",
                    "xgboost": "xgboost_model.pkl"
                },
                weather_forecast_df=forecast_df,
                selected_model=selected_model.lower()
            )

        st.subheader("🚨 AQI Alerts")
        hazardous_days = daily_preds[daily_preds["aqi_max"] >= 300]
        if not hazardous_days.empty:
            st.error("⚠️ Hazardous AQI levels detected!")
            st.dataframe(hazardous_days, width="stretch")
        else:
            st.success("✅ No hazardous AQI levels forecasted.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Daily AQI (max)**")
            st.dataframe(daily_preds, width="stretch")
            st.bar_chart(daily_preds.set_index("date")["aqi_max"])
        with col2:
            st.markdown("**Hourly AQI (first 48 hours)**")
            st.dataframe(hourly_preds.head(48), width="stretch")
            st.line_chart(hourly_preds.set_index("timestamp")[["aqi_pred"]])

        st.download_button(
            "⬇️ Download hourly predictions (CSV)",
            data=hourly_preds.to_csv(index=False),
            file_name=f"hourly_aqi_{ranges['forecast_start_day']}_{ranges['forecast_end_day']}.csv",
            mime="text/csv"
        )
        st.download_button(
            "⬇️ Download daily predictions (CSV)",
            data=daily_preds.to_csv(index=False),
            file_name=f"daily_aqi_{ranges['forecast_start_day']}_{ranges['forecast_end_day']}.csv",
            mime="text/csv"
        )

# Footer
st.markdown(
    f"<p style='text-align:center; font-size:14px;'>"
    f"Training window: {ranges['train_start_day']} → {ranges['train_end_day']} | "
    f"Forecast: {ranges['forecast_start_day']} → {ranges['forecast_end_day']}"
    f"</p>",
    unsafe_allow_html=True
)