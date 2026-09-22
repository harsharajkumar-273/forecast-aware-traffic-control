from traffic_control.aikosh import build_profile


def test_aikosh_profile():
 p=build_profile([{"timestamp":"2026-01-01T10:00:00","intersection_id":"I1","vehicle_count":"10","queue_length":"4"},{"timestamp":"2026-01-01T10:30:00","intersection_id":"I1","vehicle_count":"20","queue_length":"6"}])
 assert p["calibration"]["mean_vehicle_count"]==15 and p["rows"][0]["queue_length"]==5
