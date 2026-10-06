import json
import sqlite3

with open("sample_cloudtrail.json", "r") as file:
    events = json.load(file)

connection = sqlite3.connect("cloud_threat.db")
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS cloudtrail_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    event_time TEXT,
    event_name TEXT,
    event_source TEXT,
    user TEXT,
    source_ip TEXT,
    aws_region TEXT,
    user_identity_type TEXT,
    user_agent TEXT,
    request_parameters TEXT
)
""")

for event in events:
    cursor.execute("""
    INSERT OR IGNORE INTO cloudtrail_events (
        event_id,
        event_time,
        event_name,
        event_source,
        user,
        source_ip,
        aws_region,
        user_identity_type,
        user_agent,
        request_parameters
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event["eventID"],
        event["eventTime"],
        event["eventName"],
        event["eventSource"],
        event["userIdentity"]["userName"],
        event["sourceIPAddress"],
        event["awsRegion"],
        event["userIdentity"]["type"],
        event["userAgent"],
        json.dumps(event["requestParameters"])
    ))

connection.commit()

cursor.execute("SELECT * FROM cloudtrail_events")

for row in cursor.fetchall():
    print(row)

connection.close()