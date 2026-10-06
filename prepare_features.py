import pandas as pd

df = pd.read_csv("data/dataset.csv")

df["eventtime"] = pd.to_datetime(df["eventtime"], utc=True)
df["hour_window"] = df["eventtime"].dt.floor("h")

hourly_features = pd.crosstab(
    df["hour_window"],
    df["eventname"]
)

print(hourly_features.head())
print("Shape:", hourly_features.shape)