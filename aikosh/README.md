# AI Kosh data layer

Place an approved aggregate transportation CSV in `aikosh/data/` and run `PYTHONPATH=src python3 -m traffic_control.cli ingest-aikosh --input aikosh/data/traffic.csv --output runs/aikosh_profile.json`. Required columns: `timestamp,intersection_id,vehicle_count,queue_length`.

