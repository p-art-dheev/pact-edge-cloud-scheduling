"""Fetch measured regional carbon intensity from UK National Grid ESO.

    python scripts/fetch_carbon.py [hours]

Source: https://api.carbonintensity.org.uk - free, unauthenticated, 18 GB
regions at 30-minute resolution. Saves data/carbon/uk_regional_24h.json.

Why this source: it is openly published, needs no API key, and reports genuine
measured/forecast regional intensity. Over a single day its regions span
0 gCO2/kWh (North Scotland, wind) to ~390 (South Wales), which is the spread
that makes tier placement matter at all.
"""
import datetime as dt
import json
import os
import sys
import urllib.request

OUT = "data/carbon/uk_regional_24h.json"


def main(hours=24):
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(hours=hours)
    url = ("https://api.carbonintensity.org.uk/regional/intensity/"
           f"{start:%Y-%m-%dT%H:%MZ}/{end:%Y-%m-%dT%H:%MZ}")
    print("fetching", url)
    raw = urllib.request.urlopen(url, timeout=90).read().decode()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(raw)
    d = json.loads(raw)
    print(f"saved {OUT}: {len(d['data'])} half-hour periods, "
          f"{len(d['data'][0]['regions'])} regions")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
