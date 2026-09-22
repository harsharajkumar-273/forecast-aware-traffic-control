"""Offline-safe ingestion and calibration utilities for AI Kosh exports."""
import csv
import json
from collections import defaultdict
from pathlib import Path

REQUIRED={"timestamp","intersection_id","vehicle_count","queue_length"}
def read_traffic_csv(path):
    with Path(path).open(newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("AI Kosh export is empty")
    missing=REQUIRED-set(rows[0])
    if missing: raise ValueError(f"AI Kosh export missing columns: {sorted(missing)}")
    return rows
def build_profile(rows):
    hourly=defaultdict(lambda:{"vehicle_count":0.0,"queue_length":0.0,"records":0}); intersections=set()
    for row in rows:
        key=(str(row["timestamp"])[:13],str(row["intersection_id"])); x=hourly[key]
        x["vehicle_count"]+=float(row["vehicle_count"]); x["queue_length"]+=float(row["queue_length"]); x["records"]+=1; intersections.add(key[1])
    profile=[]
    for (hour,intersection),x in sorted(hourly.items()):
        n=x["records"]; profile.append({"hour":hour,"intersection_id":intersection,"vehicle_count":x["vehicle_count"]/n,"queue_length":x["queue_length"]/n})
    vc=[x["vehicle_count"] for x in profile]; q=[x["queue_length"] for x in profile]
    return {"intersections":sorted(intersections),"rows":profile,"calibration":{"mean_vehicle_count":sum(vc)/len(vc) if vc else 0.0,"mean_queue_length":sum(q)/len(q) if q else 0.0}}
def write_profile(input_path,output_path):
    p=build_profile(read_traffic_csv(input_path)); Path(output_path).parent.mkdir(parents=True,exist_ok=True); Path(output_path).write_text(json.dumps(p,indent=2)); return p
