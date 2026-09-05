import pandas as pd, requests, json, csv
from datetime import datetime, timezone

EXTERNAL_IP = "34.136.20.97"
URL = f"http://{EXTERNAL_IP}/predict"

df = pd.read_csv("synthetic_100.csv")

with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list(df.columns) + ["prediction", "timestamp"])

    for i, row in df.iterrows():
        payload = row.to_dict()
        try:
            resp = requests.post(URL, json=payload, timeout=10)
            pred = resp.json().get("prediction", "ERROR")
        except Exception as e:
            pred = f"ERROR: {e}"
        timestamp = datetime.now(timezone.utc).isoformat()
        writer.writerow(list(row.values) + [pred, timestamp])
        print(f"{i+1}/100:", payload, "->", pred)

print("Done. Logged to predictions.csv")
