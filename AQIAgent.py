import requests

API_KEY = "e323e4f35041ef19951799d962bdb5ccc6e878baead60ede3e7e56574ed1ae0f"   # paste full key
url = "https://api.openaq.org/v3/locations?country=PK&limit=5"

headers = {
    "X-API-Key": API_KEY
}

resp = requests.get(url, headers=headers)
print(resp.status_code)
print(resp.json())
