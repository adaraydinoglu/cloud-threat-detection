import pandas as pd
import numpy as np
from scipy.stats import entropy

def map_mitre_tactic(event_name):
    if not isinstance(event_name, str):
        return "other"
    ev = event_name.lower()
    if ev.startswith(("describe", "list", "get")):
        return "reconnaissance"
    elif any(x in ev for x in ["assumerole", "attach", "putrole", "createaccesskey", "createuser"]):
        return "privilege_escalation"
    elif any(x in ev for x in ["deletetrail", "stoplogging", "deleteflowlogs", "updateflow"]):
        return "defense_evasion"
    elif any(x in ev for x in ["delete", "terminate", "remove"]):
        return "impact"
    else:
        return "other"

def calculate_shannon_entropy(series):
    counts = series.value_counts()
    return entropy(counts)

def build_soc_features(csv_path="data/dataset.csv"):
    df = pd.read_csv(csv_path)
    
    df["eventtime"] = pd.to_datetime(df["eventtime"], utc=True)
    df["hour_window"] = df["eventtime"].dt.floor("h")
    df["mitre_tactic"] = df["eventname"].apply(map_mitre_tactic)
    
    event_counts = pd.crosstab(df["hour_window"], df["eventname"])
    
    grouped = df.groupby("hour_window")
    
    soc_df = pd.DataFrame(index=event_counts.index)
    
    soc_df["total_events"] = grouped.size()
    soc_df["unique_event_types"] = grouped["eventname"].nunique()
    soc_df["error_count"] = grouped["errorcode"].apply(lambda s: s.notna().sum())
    soc_df["error_rate"] = soc_df["error_count"] / soc_df["total_events"]
    soc_df["write_ratio"] = grouped["readonly"].apply(lambda s: (s == False).mean())
    soc_df["management_ratio"] = grouped["managementevent"].apply(lambda s: (s == True).mean())
    soc_df["unique_regions"] = grouped["awsregion"].nunique()
    soc_df["unique_source_ips"] = grouped["sourceipaddress"].nunique()
    soc_df["unique_user_agents"] = grouped["useragent"].nunique()
    
    soc_df["event_entropy"] = grouped["eventname"].apply(calculate_shannon_entropy)
    
    tactic_crosstab = pd.crosstab(df["hour_window"], df["mitre_tactic"])
    for col in ["reconnaissance", "privilege_escalation", "defense_evasion", "impact"]:
        if col in tactic_crosstab.columns:
            soc_df[f"mitre_{col}_count"] = tactic_crosstab[col]
            soc_df[f"mitre_{col}_ratio"] = tactic_crosstab[col] / soc_df["total_events"]
        else:
            soc_df[f"mitre_{col}_count"] = 0
            soc_df[f"mitre_{col}_ratio"] = 0.0
            
    soc_df = soc_df.sort_index()
    soc_df["event_volume_delta"] = soc_df["total_events"].diff().fillna(0)
    soc_df["error_count_delta"] = soc_df["error_count"].diff().fillna(0)
    
    hours = soc_df.index.hour
    soc_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    soc_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    
    combined = pd.concat([event_counts, soc_df], axis=1)
    
    return combined, soc_df

if __name__ == "__main__":
    combined, soc_features = build_soc_features()
    print("Toplam Matris Boyutu (Eventler + SOC Boyutlari):", combined.shape)
    print("\nEklenen Yeni Guvenlik ve Taktik Metrikleri (Ilk 3 Kolon):")
    cols_to_preview = [
        "total_events", "event_entropy", "event_volume_delta",
        "mitre_reconnaissance_ratio", "mitre_defense_evasion_count", "mitre_privilege_escalation_count"
    ]
    print(soc_features[cols_to_preview].head(5).round(3))
