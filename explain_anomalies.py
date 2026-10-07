import pandas as pd
from sklearn.ensemble import IsolationForest

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

results = pd.DataFrame({
    "hour_window": hourly_features.index,
    "anomaly_label": predictions,
    "anomaly_score": scores
})

baseline_mean = hourly_features.mean(axis=0)
baseline_std = hourly_features.std(axis=0).replace(0, 1)

top_anomalies = results.sort_values(by="anomaly_score").head(5)

print("=== EN YUKSEK ANOMALI SAPMASINA SAHIP 5 SAAT VE NEDENLERI ===")

for _, row in top_anomalies.iterrows():
    target_hour = row["hour_window"]
    score = row["anomaly_score"]
    
    hour_counts = hourly_features.loc[target_hour]
    z_scores = (hour_counts - baseline_mean) / baseline_std
    
    significant_events = z_scores[z_scores > 2].sort_values(ascending=False).head(3)
    
    print(f"\nSaat: {target_hour} | Karar Skoru: {score:.4f}")
    if significant_events.empty:
        print("  Belirgin tekil event artisi gorulmedi (genel hacim sapmasi olabilir).")
    else:
        for event, z_val in significant_events.items():
            actual = int(hour_counts[event])
            avg = baseline_mean[event]
            print(f"  - {event}: {actual} adet (Normal ortalama: {avg:.2f}, Sapma: +{z_val:.1f} sigma)")
