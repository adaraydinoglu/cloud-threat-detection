import pandas as pd
from sklearn.ensemble import IsolationForest
import networkx as nx
from attack_graph import build_iam_attack_graph, find_attack_paths

EVENT_BASE_SEVERITY = {
    "DeleteTrail": 10.0,
    "StopLogging": 10.0,
    "DeleteFlowLogs": 9.0,
    "PutBucketPolicy": 8.0,
    "AttachRolePolicy": 8.0,
    "CreateAccessKey": 7.0,
    "DeleteDatabase": 6.0,
    "CreateTable": 2.0,
    "StartQueryExecution": 1.0,
    "DescribeConfigurationRecorders": 0.5
}

DEFAULT_SEVERITY = 0.5

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

iam_graph = build_iam_attack_graph()

correlated_alerts = []

for idx, label in enumerate(predictions):
    if label == -1:
        hour = hourly_features.index[idx]
        raw_score = scores[idx]
        ml_magnitude = max(0.0, -raw_score)
        
        counts = hourly_features.loc[hour]
        z_scores = (counts - baseline_mean) / baseline_std
        deviated = z_scores[z_scores > 2].sort_values(ascending=False)
        
        top_action = None
        max_action_severity = 0.0
        
        if not deviated.empty:
            for ev in deviated.index:
                sev = EVENT_BASE_SEVERITY.get(ev, DEFAULT_SEVERITY)
                if sev > max_action_severity:
                    max_action_severity = sev
                    top_action = ev
        else:
            top_action = "Genel Hacim Sapmasi"
            max_action_severity = DEFAULT_SEVERITY
            
        attack_paths = find_attack_paths(iam_graph, top_action)
        has_path = len(attack_paths) > 0
        
        path_multiplier = 2.0 if has_path else 1.0
        risk_score = (ml_magnitude * 10) * max_action_severity * path_multiplier
        
        correlated_alerts.append({
            "hour_window": hour,
            "ml_anomaly_score": round(raw_score, 4),
            "primary_driver": top_action,
            "driver_severity": max_action_severity,
            "has_attack_path": has_path,
            "attack_paths": attack_paths,
            "risk_score": round(risk_score, 2)
        })

alerts_df = pd.DataFrame(correlated_alerts)
alerts_df = alerts_df.sort_values(by="risk_score", ascending=False)

print("=== EN KRITIK TEHDIT VE SALDIRI YOLU KORELASYONLARI (TOP 5) ===")
for _, item in alerts_df.head(5).iterrows():
    print(f"\nZaman Penceresi    : {item['hour_window']}")
    print(f"ML Anomali Skoru   : {item['ml_anomaly_score']}")
    print(f"Baskın Eylem       : {item['primary_driver']} (Tehdit Agirligi: {item['driver_severity']})")
    print(f"Saldiri Yolu Var mi: {'EVET' if item['has_attack_path'] else 'HAYIR'}")
    if item['has_attack_path']:
        for p in item['attack_paths']:
            print(f"  Rota             : {' -> '.join(p)}")
    print(f"Nihai Risk Skoru   : {item['risk_score']} / 100")
