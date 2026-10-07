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

print("Anomali Etiket Dağılımı:")
print(results["anomaly_label"].value_counts())
print("\nEn düşük skorlu (en anormal) 5 saat:")
print(results.sort_values(by="anomaly_score").head(5))
