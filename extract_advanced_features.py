import pandas as pd
import numpy as np

def build_advanced_features(csv_path="data/dataset.csv"):
    df = pd.read_csv(csv_path)
    
    df["eventtime"] = pd.to_datetime(df["eventtime"], utc=True)
    df["hour_window"] = df["eventtime"].dt.floor("h")
    
    event_counts = pd.crosstab(
        df["hour_window"],
        df["eventname"]
    )
    
    grouped = df.groupby("hour_window")
    
    behavioral_df = pd.DataFrame(index=event_counts.index)
    
    behavioral_df["total_events"] = grouped.size()
    behavioral_df["unique_event_types"] = grouped["eventname"].nunique()
    behavioral_df["error_count"] = grouped["errorcode"].apply(lambda s: s.notna().sum())
    behavioral_df["error_rate"] = behavioral_df["error_count"] / behavioral_df["total_events"]
    
    behavioral_df["write_ratio"] = grouped["readonly"].apply(lambda s: (s == False).mean())
    behavioral_df["management_ratio"] = grouped["managementevent"].apply(lambda s: (s == True).mean())
    
    behavioral_df["unique_regions"] = grouped["awsregion"].nunique()
    behavioral_df["unique_source_ips"] = grouped["sourceipaddress"].nunique()
    behavioral_df["unique_user_agents"] = grouped["useragent"].nunique()
    
    hours = behavioral_df.index.hour
    behavioral_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    behavioral_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    
    combined_features = pd.concat([event_counts, behavioral_df], axis=1)
    
    return combined_features, behavioral_df

if __name__ == "__main__":
    combined, behavioral = build_advanced_features()
    print("Ozellik Matrisi Boyutu (Eventler + Davranissal):", combined.shape)
    print("\nOrnek Davranissal Metrikler (Ilk 5 Saat):")
    print(behavioral.head(5).round(3))
