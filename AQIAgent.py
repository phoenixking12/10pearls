import hopsworks
import pandas as pd
import requests
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time


model_defs = {
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    ),
    "Ridge": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge())
        ]),
        {
            "ridge__alpha": [0.1, 1.0, 10.0]
        }
    ),
    "XGBoost": (
        XGBRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        }
    )
}

API_KEY = "vkQUwBffplAUhyYL.giHjnNqIRRF88JUfLtwjyskbXVuqhO3mHiocVBP5I9OOrEOxVSjLIDQBUw1qpo1Q"
PROJECT_NAME = "AirAi"

OPENAQ_HEADERS = {
    "X-API-Key": "e323e4f35041ef19951799d962bdb5ccc6e878baead60ede3e7e56574ed1ae0f"
}

LAT, LON = 24.8607, 67.0011
CITY = "Karachi"
COUNTRY = "PK"

TRAIN_DAYS = 30   
FORECAST_DAYS = 3

def date_str(dt):
    return dt.strftime("%Y-%m-%d")

def iso_utc(dt):

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def get_dynamic_date_ranges(train_days=TRAIN_DAYS, forecast_days=FORECAST_DAYS):
    now = datetime.datetime.now(datetime.timezone.utc)
    train_end = now
    train_start = now - datetime.timedelta(days=train_days)
    forecast_start = now
    forecast_end = now + datetime.timedelta(days=forecast_days)
    return {
        "train_start_iso": train_start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
        "train_end_iso": train_end.replace(hour=23, minute=59, second=59, microsecond=0).isoformat(),
        "train_start_day": train_start.date().strftime("%Y-%m-%d"),
        "train_end_day": train_end.date().strftime("%Y-%m-%d"),
        "forecast_start_day": forecast_start.date().strftime("%Y-%m-%d"),
        "forecast_end_day": forecast_end.date().strftime("%Y-%m-%d"),
    }

def _extract_ts(m):
    ts = m.get("timestamp", {}).get("utc")
    if ts:
        return ts

    ts = m.get("period", {}).get("datetimeFrom", {}).get("utc")
    if ts:
        return ts
    ts = m.get("period", {}).get("datetimeTo", {}).get("utc")
    return ts

def get_sensor_data_over_period(location_id, date_from, date_to, page_limit=1000, max_pages=10):
    loc_url = f"https://api.openaq.org/v3/locations/{location_id}"
    r = requests.get(loc_url, headers=OPENAQ_HEADERS)
    r.raise_for_status()
    sensors = r.json()["results"][0].get("sensors", [])

    all_data = []
    for s in sensors:
        sensor_id = s["id"]
        page = 1
        while page <= max_pages:
            meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
            meas_params = {
                "date_from": date_from,
                "date_to": date_to,
                "limit": page_limit,
                "page": page,
                "sort": "desc",
            }
            r = requests.get(meas_url, headers=OPENAQ_HEADERS, params=meas_params)
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
            if not results:
                break
            for m in results:
                ts = _extract_ts(m)
                if not ts:
                    continue
                all_data.append({
                    "sensor_id": sensor_id,
                    "parameter": m["parameter"]["name"], 
                    "value": m["value"],
                    "unit": m["parameter"]["units"],
                    "timestamp": ts,
                })
            page += 1
    return pd.DataFrame(all_data)
 
def get_weather_data(start_date, end_date, hourly=True):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    if hourly:
        params["hourly"] = [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "weathercode"
        ]
    # else:
    #     params["daily"] = [
    #         "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    #         "windspeed_10m_max", "weathercode"
    #     ]

    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()

    if hourly:
        df = pd.DataFrame(data["hourly"])
        df.rename(columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "pressure_msl": "pressure",
            "wind_speed_10m": "wind_speed",
            "time": "timestamp",
        }, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["city"] = CITY
    df["country"] = COUNTRY
    return df


def enrich_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    df["day"] = df["timestamp"].dt.day.astype(float)
    df["month"] = df["timestamp"].dt.month.astype(float)
    df["weekday"] = df["timestamp"].dt.dayofweek.astype(float) 

    df = df.sort_values(by=["parameter", "timestamp"])

    if "value" in df.columns:
        df["value_prev"] = df.groupby("parameter")["value"].shift(1)
        df["aqi_change_rate"] = ((df["value"] - df["value_prev"]) / df["value_prev"]).fillna(0)

        # Lag features
        for lag in [3, 6, 12, 24]:
            df[f"value_lag_{lag}h"] = df.groupby("parameter")["value"].shift(lag)

        # Rolling averages
        df["value_roll_mean_6h"] = df.groupby("parameter")["value"].transform(lambda x: x.rolling(6).mean())
        df["value_roll_std_6h"] = df.groupby("parameter")["value"].transform(lambda x: x.rolling(6).std())
    else:
        df["aqi_change_rate"] = 0.0

    return df

def compute_aqi_pm25(pm25):
    # US EPA PM2.5 AQI breakpoints (µg/m³)
    bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm = float(pm25)
    for Clow, Chigh, Ilow, Ihigh in bp:
        if Clow <= pm <= Chigh:
            return Ilow + (Ihigh - Ilow) * (pm - Clow) / (Chigh - Clow)
    return np.nan

def upload_to_hopsworks(df, group_name="environmental_features"):
    try:
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
        fs = project.get_feature_store()
    except Exception as e:
        print("⚠️ Could not connect to Hopsworks:", e)
        return

    df = df.rename(columns=lambda x: x.strip().lower())
    df = df.fillna({"city": "Unknown", "country": "Unknown"})
    df = df.where(pd.notnull(df), None)

    if "sensor_id" in df.columns:
        df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce")

    if "weathercode" in df.columns:
        df["weathercode"] = pd.to_numeric(df["weathercode"], errors="coerce").fillna(0).astype("int64")

    numeric_cols = ["hour", "day", "month", "humidity", "pressure", "wind_speed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fg = fs.get_or_create_feature_group(
        name=group_name,
        version=4,
        description="Environmental data from OpenAQ and Open-Meteo",
        primary_key=["timestamp", "parameter"],
        online_enabled=False
    )

    job, ge_report = fg.insert(df)


    print("✅ Data uploaded to Hopsworks. Materialization job started.")

    # Poll materialization job state
    while True:
        state = fg.materialization_job.get_state()
        print("Materialization state:", state)
        if state in ["FINISHED", "FAILED", "STOPPED"]:
            break
        time.sleep(10)

    print("Final state:", fg.materialization_job.get_final_state())


def evaluate_model(model, X, y, label=""):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{label} Performance -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def train_models(df, target_param="pm25", selected_model="RandomForest"):
    data = df[df["parameter"] == target_param].dropna(subset=["value"])
    features = [
    "hour", "day", "month", "humidity", "pressure", "wind_speed",
    "aqi_change_rate", "value_prev",
    "value_lag_3h", "value_lag_6h", "value_lag_12h", "value_lag_24h",
    "value_roll_mean_6h", "value_roll_std_6h"
    ]
    data = data.dropna(subset=features).sort_values("timestamp")

    if len(data) < 10:
        print("⚠️ Not enough data points after alignment for robust training.")
    split_idx = int(0.8 * len(data)) if len(data) > 5 else len(data)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:] if split_idx < len(data) else pd.DataFrame()

    X_train, y_train = train[features], train["value"]
    X_train = X_train.apply(pd.to_numeric, errors="coerce").dropna()

    try:
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
        mr = project.get_model_registry()
    except Exception as e:
        print("!!! Could not connect to Hopsworks Model Registry:", e)
        mr = None

    results = {}
    for name, (base_model, param_grid) in model_defs.items():
        print(f"\n🔍 Tuning {name}...")
        grid = GridSearchCV(base_model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"!!! Best params for {name}: {grid.best_params_}")

        if not test.empty:
            X_test, y_test = test[features], test["value"]
            preds = model.predict(X_test)
            scope = "(out-of-sample)"
        else:
            preds = model.predict(X_train)
            y_test = y_train
            scope = "(in-sample)"

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        baseline_pred = y_test.shift(1).fillna(y_train.mean())
        print("Baseline RMSE:", np.sqrt(mean_squared_error(y_test, baseline_pred)))

        train_metrics = evaluate_model(model, X_train, y_train, label=f"{name} (train)")
        if not test.empty:
            test_metrics = evaluate_model(model, X_test, y_test, label=f"{name} (test)")
        else:
            test_metrics = None

        results[name] = {"train": train_metrics, "test": test_metrics}

        model_path = f"{name.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"\n{name} {scope} Performance:\nRMSE: {rmse:.3f}\tMAE: {mae:.3f}\tR2: {r2:.3f}")

        if name == selected_model and name != "Ridge":
            try:
                shap_path = f"{name.lower()}_shap_summary.png"
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                shap.summary_plot(shap_values, X_train, show=False)
                plt.savefig(shap_path, bbox_inches="tight")
                plt.close()
                print(f"!!! SHAP summary saved: {shap_path}")
            except Exception as e:
                print(f"!!! SHAP failed for {name}: {e}")


        model_obj = mr.python.create_model(
            name=f"{name}_AQI_Model",
            description=f"{name} model for PM2.5 prediction",
            metrics={"rmse": rmse, "mae": mae, "r2": r2},
            input_example=X_train.head(1)
        )

        model_obj.save(model_path)
    return results


def predict_next_3_days(models_paths, weather_forecast_df, selected_model="XGBoost"):
    df = weather_forecast_df.copy()
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    df["day"] = df["timestamp"].dt.day.astype(float)
    df["month"] = df["timestamp"].dt.month.astype(float)
    df["aqi_change_rate"] = 0.0 

    features = ["hour", "day", "month", "humidity", "pressure", "wind_speed", "aqi_change_rate"]
    X_future = df[features].fillna(method="ffill").fillna(method="bfill")

    model_path = models_paths.get(selected_model)
    if not model_path:
        raise ValueError(f"Model '{selected_model}' not found in models_paths.")
    model = joblib.load(model_path)


    df["pm25_pred"] = model.predict(X_future)
    df["aqi_pred"] = df["pm25_pred"].apply(compute_aqi_pm25)

    daily = df.groupby(df["timestamp"].dt.date).agg(
        pm25_mean=("pm25_pred", "mean"),
        aqi_max=("aqi_pred", "max")
    ).reset_index().rename(columns={"timestamp": "date"})

    return df[["timestamp", "pm25_pred", "aqi_pred"]], daily


def run_feature_pipeline():
    ranges = get_dynamic_date_ranges()

    pollutant_df = get_sensor_data_over_period(
        location_id=4837117,
        date_from=ranges["train_start_iso"],
        date_to=ranges["train_end_iso"]
    )

    weather_df = get_weather_data(
        start_date=ranges["train_start_day"],
        end_date=ranges["train_end_day"],
        hourly=True
    )

    pollutant_df = enrich_features(pollutant_df)
    pollutant_df["timestamp"] = pd.to_datetime(pollutant_df["timestamp"], errors="coerce").dt.tz_localize(None)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce").dt.tz_localize(None)

    pollutant_df = pollutant_df.dropna(subset=["timestamp"])
    weather_df = weather_df.dropna(subset=["timestamp"])

    pollutant_df["timestamp"] = pollutant_df["timestamp"].dt.floor("H")
    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("H")

    merged_df = pollutant_df.merge(weather_df, on="timestamp", how="left")

    if "parameter" not in merged_df.columns:
        merged_df["parameter"] = pollutant_df["parameter"]

    merged_df["parameter"] = merged_df["parameter"].fillna("weather")

    merged_df = merged_df.dropna(subset=["temperature", "humidity", "pressure", "wind_speed", "weathercode"])

    upload_to_hopsworks(merged_df)
    print("!!! Feature pipeline completed.")


def run_training_pipeline():
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name="environmental_features",
        version=4 
    )

    merged_df = feature_group.read() 

    print(f"📥 Loaded {len(merged_df)} rows from feature store")

    target_param = "pm25"
    train_models(merged_df, target_param=target_param)

    print("✅ Training pipeline completed.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "feature":
        run_feature_pipeline()
    elif mode == "train":
        run_training_pipeline()
    else:
        run_feature_pipeline()
        run_training_pipeline()
