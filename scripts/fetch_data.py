"""Download the Rezaee DAG benchmark and build the simulation cache.

    python scripts/fetch_data.py

Downloads ~91 MB from Zenodo (CC-BY-4.0) into data/rezaee/, then extracts one
time window into a compact cache the simulator reads.
"""
import os
import urllib.request

FILES = {
    "JobDetails.zip":
        "https://zenodo.org/api/records/4667690/files/JobDetails..zip/content",
    "TaskDetails.zip":
        "https://zenodo.org/api/records/4667690/files/TaskDetails.txt.zip/content",
}

def main():
    os.makedirs("data/rezaee", exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join("data/rezaee", name)
        if os.path.exists(dst):
            print(f"have {dst}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dst)
        print(f"  -> {os.path.getsize(dst):,} bytes")
    from pact.sim.rezaee import build_cache
    build_cache(window_start=0.0, window_len=600.0, max_jobs=4000)

if __name__ == "__main__":
    main()
