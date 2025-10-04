import requests
import pandas as pd
from datetime import datetime

OPENAQ_KEY = "e323e4f35041ef19951799d962bdb5ccc6e878baead60ede3e7e56574ed1ae0f"
OPENWEATHER_KEY = "8f182d0e5e8c48e2a6dc308668938409"

HEADERS = {"X-API-Key": OPENAQ_KEY}

def get_openaq_data(location_id, date_from, date_to, limit=1000):
    """Fetch pollutant data from OpenAQ"""
    loc_url = f"https://api.openaq.org/v3/locations/{location_id}"
    r = requests.get(loc_url, headers=HEADERS)
    r.raise_for_status()
    sensors = r.json()["results"][0].get("sensors", [])
    
    records = []
    for s in sensors:
        sensor_id = s["id"]
        meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
        params = {"date_from": date_from, "date_to": date_to, "limit": limit, "sort": "desc"}
        res = requests.get(meas_url, headers=HEADERS, params=params).json()
        for m in res.get("results", []):
            ts = m.get("period", {}).get("datetimeFrom", {}).get("utc")
            records.append({
                "sensor_id": sensor_id,
                "parameter": m["parameter"]["name"],
                "value": m["value"],
                "unit": m["parameter"]["units"],
                "timestamp": ts
            })
    return pd.DataFrame(records)


def get_openweather_data(lat, lon, start, end):
    """Fetch hourly weather data from OpenWeather One Call History API"""
    weather_records = []

    for ts in pd.date_range(start=start, end=end, freq="1H"):
        unix_time = int(ts.timestamp())
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {"lat": lat, "lon": lon, "dt": unix_time, "appid": OPENWEATHER_KEY, "units": "metric"}
        r = requests.get(url, params=params).json()
        if "hourly" in r:
            for h in r["hourly"]:
                weather_records.append({
                    "timestamp": datetime.utcfromtimestamp(h["dt"]).isoformat() + "Z",
                    "temp": h["temp"],
                    "humidity": h["humidity"],
                    "wind_speed": h["wind_speed"]
                })
    return pd.DataFrame(weather_records)