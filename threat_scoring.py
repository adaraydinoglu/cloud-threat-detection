import pandas as pd
from sklearn.ensemble import IsolationForest

EVENT_RISK_WEIGHTS = {
    "DeleteTrail": 10.0,
    "StopLogging": 10.0,
    "DeleteFlowLogs": 9.0,
    "PutBucketPolicy": 8.0,
    "AttachUserPolicy": 8.0,
    "AttachRolePolicy": 8.0,
    "CreateAccessKey": 7.0,
    "CreateUser": 7.0,
    "DeleteDatabase": 6.0,
    "DeleteGroup": 5.0,
    "CreateTable": 2.0,
    "CreateDatabase": 2.0,
    "StartQueryExecution": 1.0,
    "ListNamedQueries": 0.5,
    "GetTable": 0.5,
    "DescribeConfigurationRecorders": 0.5,
    "DescribeConfigurationRecorderStatus": 0.5,
    "GetPolicy20150331v2": 0.5,
    "ListCapacityReservations": 0.5,
    "ListTags": 0.2
}

DEFAULT_RISK_WEIGHT = 1.0

df = pd.read_csv("data/dataset.csv")

df["eventtime"] = pd.to_datetime(df["eventtime"], utc=True)
df["hour_window"] = df["eventtime"].dt.floor("h")

hourly_features = pd.crosstab(
    df["hour_window"],
    df["eventname"]
)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(hourly_features)

predictions = model.predict(hourly_features)
scores = model.decision_function(hourly_features)

baseline_mean = hourly_features.mean(axis=0)
baseline_std = hourly_features.std(axis=0).replace(0, 1)

anomaly_indices = [i for i, label in enumerate(predictions) if label == -1]

risk_records = []

for idx in anomaly_indices:
    hour = hourly_features.index[idx]
    raw_ml_score = scores[idx]
    
    ml_anomaly_magnitude = max(0.0, -raw_ml_score)
    
    hour_counts = hourly_features.loc[hour]
    z_scores = (hour_counts - baseline_mean) / baseline_std
    deviated_events = z_scores[z_scores > 2]
    
    security_weight_sum = 0.0
    top_driver_name = "Yok"
    top_driver_weight = 0.0
    
    if not deviated_events.empty:
        for ev, _ in deviated_events.items():
            w = EVENT_RISK_WEIGHTS.get(ev, DEFAULT_RISK_WEIGHT)
            security_weight_sum += w
            if w > top_driver_weight:
                top_driver_weight = w
                top_driver_name = ev
    else:
        security_weight_sum = DEFAULT_RISK_WEIGHT
        
    composite_risk = ml_anomaly_magnitude * (1.0 + security_weight_sum)
    
    risk_records.append({
        "hour_window": hour,
        "raw_ml_score": raw_ml_score,
        "ml_anomaly_magnitude": ml_anomaly_magnitude,
        "security_weight_sum": security_weight_sum,
        "top_threat_event": top_driver_name,
        "composite_risk_score": composite_risk
    })

risk_df = pd.DataFrame(risk_records)

print("=== TEHDIT ONCELIKLENDIRMESINE GORE ILK 5 SAAT (COMPOSITE RISK) ===")
sorted_by_risk = risk_df.sort_values(by="composite_risk_score", ascending=False).head(5)
for _, r in sorted_by_risk.iterrows():
    print(f"\nSaat: {r['hour_window']}")
    print(f"  ML Sapma Derecesi : {r['ml_anomaly_magnitude']:.4f}")
    print(f"  Guvenlik Agirligi : {r['security_weight_sum']:.1f} (En kritik olay: {r['top_threat_event']})")
    print(f"  Bilesik Risk Skoru: {r['composite_risk_score']:.4f}")
