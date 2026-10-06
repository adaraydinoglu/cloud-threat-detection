import sqlite3
import pandas as pd
from datetime import datetime, timedelta

connection = sqlite3.connect("cloud_threat.db")
cursor = connection.cursor()

cursor.execute("""
SELECT event_id, event_time, event_name, user, source_ip
FROM cloudtrail_events
ORDER BY event_time
""")

events = cursor.fetchall()
seen_ips = {}
seen_events = {}
features = []

for event in events:
    event_id, event_time, event_name, user, source_ip = event
    current_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    hour = current_time.hour

    if user not in seen_ips:
        seen_ips[user] = set()

    if user not in seen_events:
        seen_events[user] = set()

    is_new_ip = 0 if source_ip in seen_ips[user] else 1
    is_new_event = 0 if event_name in seen_events[user] else 1

    seen_ips[user].add(source_ip)
    seen_events[user].add(event_name)

    window_start = current_time - timedelta(minutes=10)

    event_frequency = sum(
        1
        for previous_event in events
        if previous_event[3] == user
        and window_start <= datetime.fromisoformat(previous_event[1].replace("Z", "+00:00")) < current_time
    )

    features.append({
        "event_id": event_id,
        "hour": hour,
        "is_new_ip": is_new_ip,
        "is_new_event": is_new_event,
        "event_frequency": event_frequency
    })

connection.close()

df = pd.DataFrame(features)

print(df)