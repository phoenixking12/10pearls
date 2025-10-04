import requests

HEADERS = {"X-API-Key": "e323e4f35041ef19951799d962bdb5ccc6e878baead60ede3e7e56574ed1ae0f"}

def get_sensor_data_over_period(location_id, date_from, date_to, limit=1000):
    """
    Fetch measurements for all sensors at a given location
    within the specified date range, with timestamps.
    """
    loc_url = f"https://api.openaq.org/v3/locations/{location_id}"
    r = requests.get(loc_url, headers=HEADERS)
    r.raise_for_status()
    location = r.json()["results"][0]   # single location
    sensors = location.get("sensors", [])

    all_data = []

    for s in sensors:
        sensor_id = s["id"]
        meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
        meas_params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "sort": "desc"
        }
        r = requests.get(meas_url, headers=HEADERS, params=meas_params)
        r.raise_for_status()
        results = r.json().get("results", [])

        for m in results:
            ts = (
                m.get("period", {}).get("datetimeFrom", {}).get("utc")
                or m.get("period", {}).get("datetimeTo", {}).get("utc")
            )

            all_data.append({
                "sensor_id": sensor_id,
                "parameter": m["parameter"]["name"],
                "value": m["value"],
                "unit": m["parameter"]["units"],
                "timestamp": ts
            })

            print(all_data[-1])  # for debugging



# Example usage
if __name__ == "__main__":
    data = get_sensor_data_over_period(
        location_id=4837117,   # NED Karachi
        date_from="2025-09-25T00:00:00Z",
        date_to="2025-10-02T23:59:59Z",
        limit=5
    )

    if not data:
        print("⚠️ No data found for given period.")
    else:
        for d in data:
            print(d)
