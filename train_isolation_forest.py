from pathlib import Path
import ipaddress
import pandas as pd
from sklearn.ensemble import IsolationForest


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = Path("data/dataset.csv")
OUTPUT_DIR = Path("data")
TARGET_IP = "203.0.113.42"
ANALYSIS_WINDOW_MINUTES = 60
MIN_SOURCE_ACTIVE_HOURS = 20
SOURCE_CONTAMINATION = 0.05
DOCUMENTATION_NETWORKS = [
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
]

DISCOVERY_EVENTS = [
    "DescribeRegions",
    "DescribeInstances",
    "DescribeSecurityGroups",
    "DescribeSubnets",
    "DescribeNetworkAcls",
    "DescribeRouteTables",
    "DescribeInternetGateways",
    "DescribeNatGateways",
    "ListBuckets",
]

CRITICAL_EVENT_POINTS = {
    "AssumeRole": 2,
    "StopLogging": 3,
    "DeleteTrail": 3,
}

REQUIRED_COLUMNS = ["eventtime", "eventname"]
OPTIONAL_COLUMNS = [
    "eventsource",
    "awsregion",
    "readonly",
    "managementevent",
    "sourceipaddress",
    "errorcode",
    "useragent",
]


def print_section(title: str) -> None:
    """Print a consistent section heading."""
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def load_dataset() -> pd.DataFrame:
    """Load and validate the CloudTrail-like CSV dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATA_PATH}\n"
            "Komutu proje klasöründe çalıştırdığından emin ol."
        )

    data = pd.read_csv(DATA_PATH)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Veri setinde zorunlu sütun(lar) eksik: " + ", ".join(missing)
        )

    # Analizde kullanacağımız isteğe bağlı alanlar yoksa boş oluştur.
    for column in OPTIONAL_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data["eventtime"] = pd.to_datetime(
        data["eventtime"], utc=True, errors="coerce"
    )
    data = data.dropna(subset=["eventtime", "eventname"]).copy()
    data["eventname"] = data["eventname"].astype(str)
    data["hour_window"] = data["eventtime"].dt.floor("h")

    if data.empty:
        raise ValueError("Geçerli eventtime ve eventname içeren kayıt bulunamadı.")

    return data


def run_isolation_forest(data: pd.DataFrame):
    """Train Isolation Forest on hourly event-type counts."""
    features = pd.crosstab(data["hour_window"], data["eventname"])

    if len(features) < 2:
        raise ValueError(
            "Isolation Forest için en az iki farklı saatlik zaman penceresi gerekli."
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    model.fit(features)

    results = pd.DataFrame(
        {
            "hour_window": features.index,
            "anomaly_label": model.predict(features),
            "anomaly_score": model.decision_function(features),
            "event_count": features.sum(axis=1).values,
        }
    ).reset_index(drop=True)

    return features, results


def classify_source_value(value: object) -> str:
    """Describe the sourceipaddress column value without assuming its meaning."""
    if pd.isna(value):
        return "missing"
    text_value = str(value).strip()
    try:
        address = ipaddress.ip_address(text_value)
    except ValueError:
        return "non_ip_label_in_source_column"

    if address.version == 4 and any(
        address in network for network in DOCUMENTATION_NETWORKS
    ):
        return "documentation_ipv4_not_real_attribution"
    return "ip_address_format_not_independently_validated"


def run_source_level_isolation_forest(
    data: pd.DataFrame,
    global_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one Isolation Forest per source-field value using active hours only.

    The sourceipaddress field in this dataset also contains service-name labels,
    so the function treats each distinct value as an entity key without asserting
    that every value is a real IP address.
    """
    source_data = data.dropna(subset=["sourceipaddress"]).copy()
    group_columns = ["sourceipaddress", "hour_window"]

    def count_errors(values: pd.Series) -> int:
        cleaned = values.astype("string").str.strip()
        return int((values.notna() & cleaned.ne("")).sum())

    def count_write_events(values: pd.Series) -> int:
        return int(values.astype("string").str.lower().eq("false").sum())

    # One row per source-field value and active hour. Empty hours are excluded so
    # the model evaluates behaviour when an entity is active, not its inactivity.
    base_features = source_data.groupby(
        group_columns, dropna=False
    ).agg(
        source_event_count=("eventname", "size"),
        distinct_event_types=("eventname", "nunique"),
        error_count=("errorcode", count_errors),
        non_readonly_count=("readonly", count_write_events),
        distinct_user_agents=("useragent", "nunique"),
        distinct_regions=("awsregion", "nunique"),
    )

    event_type_counts = pd.crosstab(
        [source_data["sourceipaddress"], source_data["hour_window"]],
        source_data["eventname"],
    )
    event_type_counts.columns = [
        f"event_type__{event_name}" for event_name in event_type_counts.columns
    ]

    source_features = base_features.join(event_type_counts, how="left").fillna(0)
    global_anomaly_map = global_results.set_index("hour_window")["anomaly_label"].to_dict()
    global_score_map = global_results.set_index("hour_window")["anomaly_score"].to_dict()

    hourly_rows = []
    summary_rows = []

    for source_value, entity_features in source_features.groupby(
        level="sourceipaddress", sort=False
    ):
        entity_features = entity_features.droplevel("sourceipaddress").sort_index()
        entity_events = source_data[
            source_data["sourceipaddress"] == source_value
        ]
        active_hours = len(entity_features)
        distinct_types = int(entity_events["eventname"].nunique())
        source_kind = classify_source_value(source_value)

        variable_columns = [
            column
            for column in entity_features.columns
            if entity_features[column].nunique(dropna=False) > 1
        ]

        model_status = "trained"
        anomaly_count = 0
        overlap_count = 0
        feature_count = len(variable_columns)
        top_anomaly_hours = ""
        min_anomaly_score = float("nan")

        if active_hours < MIN_SOURCE_ACTIVE_HOURS:
            model_status = f"skipped_fewer_than_{MIN_SOURCE_ACTIVE_HOURS}_active_hours"
        elif not variable_columns:
            model_status = "skipped_no_varying_features"
        else:
            entity_model = IsolationForest(
                n_estimators=100,
                contamination=SOURCE_CONTAMINATION,
                random_state=42,
            )
            model_input = entity_features[variable_columns].astype(float)
            entity_model.fit(model_input)
            labels = entity_model.predict(model_input)
            scores = entity_model.decision_function(model_input)

            entity_result = pd.DataFrame({
                "sourceipaddress": str(source_value),
                "source_value_type": source_kind,
                "hour_window": entity_features.index,
                "source_event_count": entity_features["source_event_count"].to_numpy(),
                "distinct_event_types": entity_features["distinct_event_types"].to_numpy(),
                "error_count": entity_features["error_count"].to_numpy(),
                "non_readonly_count": entity_features["non_readonly_count"].to_numpy(),
                "source_anomaly_label": labels,
                "source_anomaly_score": scores,
            })
            entity_result["global_anomaly_label"] = (
                entity_result["hour_window"].map(global_anomaly_map).fillna(1).astype(int)
            )
            entity_result["global_anomaly_score"] = entity_result["hour_window"].map(
                global_score_map
            )
            entity_result["global_anomaly_present"] = (
                entity_result["global_anomaly_label"] == -1
            )
            entity_result["source_global_agreement"] = (
                (entity_result["source_anomaly_label"] == -1)
                & entity_result["global_anomaly_present"]
            )
            hourly_rows.append(entity_result)

            anomalies = entity_result[
                entity_result["source_anomaly_label"] == -1
            ].sort_values("source_anomaly_score")
            anomaly_count = len(anomalies)
            overlap_count = int(anomalies["global_anomaly_present"].sum())
            if not anomalies.empty:
                min_anomaly_score = float(anomalies["source_anomaly_score"].min())
                top_anomaly_hours = " | ".join(
                    str(hour) for hour in anomalies["hour_window"].head(3)
                )

        summary_rows.append({
            "sourceipaddress": str(source_value),
            "source_value_type": source_kind,
            "total_events": len(entity_events),
            "active_hours": active_hours,
            "distinct_event_types": distinct_types,
            "features_used": feature_count,
            "model_status": model_status,
            "source_anomaly_hours": anomaly_count,
            "source_anomaly_rate_pct": round(100 * anomaly_count / active_hours, 2)
                if model_status == "trained" and active_hours else float("nan"),
            "source_global_overlap_hours": overlap_count,
            "lowest_anomaly_score": min_anomaly_score,
            "top_anomaly_hours_utc": top_anomaly_hours,
        })

    hourly_columns = [
        "sourceipaddress", "source_value_type", "hour_window",
        "source_event_count", "distinct_event_types", "error_count",
        "non_readonly_count", "source_anomaly_label", "source_anomaly_score",
        "global_anomaly_label", "global_anomaly_score", "global_anomaly_present",
        "source_global_agreement",
    ]
    hourly_results = (
        pd.concat(hourly_rows, ignore_index=True)
        if hourly_rows else pd.DataFrame(columns=hourly_columns)
    )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["source_anomaly_hours", "active_hours"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return hourly_results, summary


def inspect_source_level_results(
    hourly_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print source-level model coverage and the most anomalous active hours."""
    print_section("KAYNAK ALANI BAZLI ISOLATION FOREST")
    if summary.empty:
        print("Kaynak bazlı model için değerlendirilecek kayıt bulunamadı.")
        return

    trained_count = int(summary["model_status"].eq("trained").sum())
    skipped_count = len(summary) - trained_count
    print(f"Kaynak alanındaki farklı değer sayısı: {len(summary)}")
    print(f"Model eğitilen değer sayısı: {trained_count}")
    print(f"Yetersiz/uygun olmayan veri nedeniyle atlanan değer: {skipped_count}")
    print("Not: Bu sütunda gerçek IP biçimleri yanında servis adı etiketleri de bulunuyor.")

    columns = [
        "sourceipaddress", "source_value_type", "total_events", "active_hours",
        "distinct_event_types", "source_anomaly_hours", "source_anomaly_rate_pct",
        "source_global_overlap_hours", "model_status",
    ]
    print("\nKaynak bazlı özet:")
    print(summary[columns].to_string(index=False))

    if not hourly_results.empty:
        anomalies = hourly_results[
            hourly_results["source_anomaly_label"] == -1
        ].sort_values("source_anomaly_score")
        print("\nKaynak bazında en düşük skorlu 10 aktif saat:")
        print(
            anomalies[[
                "sourceipaddress", "hour_window", "source_event_count",
                "source_anomaly_score", "global_anomaly_present",
                "source_global_agreement",
            ]].head(10).to_string(index=False)
        )


def inspect_anomaly_results(
    data: pd.DataFrame,
    features: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Print a concise summary of hourly anomalies."""
    totals = features.sum(axis=1)

    print_section("VERİ SETİ BİLGİLERİ")
    print(f"Toplam olay sayısı: {len(data):,}")
    print(f"Saatlik pencere sayısı: {len(features):,}")
    print(f"Zaman aralığı: {data['eventtime'].min()} — {data['eventtime'].max()}")
    print("Kullanılabilir sütunlar:", ", ".join(data.columns))

    print_section("SAATLİK OLAY İSTATİSTİKLERİ")
    print(totals.describe().round(2).to_string())
    print("\nEn fazla olay içeren 5 saat:")
    print(totals.sort_values(ascending=False).head(5).to_string())

    print_section("ISOLATION FOREST SONUÇLARI")
    print("Anomali etiket dağılımı (-1=anomali, 1=normal):")
    print(results["anomaly_label"].value_counts().sort_index().to_string())
    print("\nEn düşük skorlu (en anormal) 5 saat:")
    print(
        results.sort_values("anomaly_score")[
            ["hour_window", "anomaly_label", "anomaly_score", "event_count"]
        ].head(5).to_string(index=False)
    )

    most_anomalous_hour = results.loc[results["anomaly_score"].idxmin(), "hour_window"]
    events = data[data["hour_window"] == most_anomalous_hour]

    print_section("EN ANORMAL SAATİN DETAYLARI")
    print("Saat:", most_anomalous_hour)
    print("Toplam olay:", len(events))
    print("En sık olay türleri:")
    print(events["eventname"].value_counts().head(10).to_string())
    print("\nAWS servisleri:")
    print(events["eventsource"].value_counts(dropna=False).head(10).to_string())
    print("\nHata kodları:")
    print(events["errorcode"].value_counts(dropna=False).head(10).to_string())
    print("\nKaynak IP adresleri:")
    print(events["sourceipaddress"].value_counts(dropna=False).head(10).to_string())

    # Önceden incelenen iki saat mevcut veri setinde varsa karşılaştır.
    print_section("SEÇİLİ SAATLERİN NORMAL ORTALAMAYLA KARŞILAŞTIRILMASI")
    comparison_hours = [
        pd.Timestamp("2026-02-25 17:00:00", tz="UTC"),
        pd.Timestamp("2026-02-13 04:00:00", tz="UTC"),
    ]
    for hour in comparison_hours:
        if hour not in features.index:
            print(f"\n{hour}: veri setinde bulunamadı.")
            continue

        current = features.loc[hour]
        other_hours = features.loc[features.index != hour]
        baseline = other_hours.mean()
        comparison = pd.DataFrame(
            {"event_count": current, "baseline_average": baseline}
        )
        comparison["difference"] = (
            comparison["event_count"] - comparison["baseline_average"]
        )
        comparison = comparison[comparison["event_count"] > 0]

        print(f"\nSaat: {hour}")
        print(
            comparison.sort_values("difference", ascending=False)
            .head(10)
            .round(2)
            .to_string()
        )


def inspect_general_traffic(data: pd.DataFrame) -> None:
    """Print compact global traffic summaries; never dump all event rows."""
    print_section("GENEL TRAFİK ÖZETİ")
    print("En sık görülen kaynak IP adresleri:")
    print(data["sourceipaddress"].value_counts(dropna=False).head(10).to_string())
    print("\nUser-Agent dağılımı:")
    print(data["useragent"].value_counts(dropna=False).head(5).to_string())
    print("\nBölgelere göre olay sayısı:")
    print(data["awsregion"].value_counts(dropna=False).head(10).to_string())

    selected_hour = pd.Timestamp("2026-02-25 17:00:00", tz="UTC")
    hour_events = data[data["hour_window"] == selected_hour]
    print_section("25 ŞUBAT 2026 - 17:00 UTC IP / USER-AGENT ÖZETİ")
    if hour_events.empty:
        print("Bu saat için olay bulunamadı.")
    else:
        combos = (
            hour_events.groupby(["sourceipaddress", "useragent"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        print(combos.to_string())


def inspect_target_ip(data: pd.DataFrame, target_ip: str) -> None:
    """Show a concise investigation summary for a selected IP."""
    ip_events = data[data["sourceipaddress"] == target_ip]

    print_section(f"HEDEF IP ANALİZİ: {target_ip}")
    print("Toplam olay sayısı:", len(ip_events))

    if ip_events.empty:
        print("Bu IP için kayıt bulunamadı.")
        return

    print("\nEn sık olay türleri:")
    print(ip_events["eventname"].value_counts().head(12).to_string())
    print("\nZaman aralığı:")
    print(ip_events["eventtime"].agg(["min", "max"]).to_string())

    # GetCallerIdentity kayıtları çok sık olabileceğinden tek tek yazdırma.
    identity_count = int((ip_events["eventname"] == "GetCallerIdentity").sum())
    print("\nGetCallerIdentity kayıt sayısı:", identity_count)

    important = ["AssumeRole", "StopLogging", "DeleteTrail"]
    critical_events = ip_events[ip_events["eventname"].isin(important)]

    print("\nHedef IP için kritik olaylar:")
    if critical_events.empty:
        print("AssumeRole, StopLogging veya DeleteTrail kaydı bulunamadı.")
    else:
        print(
            critical_events[
                ["eventtime", "eventname", "eventsource", "errorcode"]
            ].sort_values("eventtime").to_string(index=False)
        )

    # Bu IP, örnek dokümantasyon adres aralığındadır; gerçek saldırgan adresi varsayma.
    try:
        address = ipaddress.ip_address(target_ip)
        if address in ipaddress.ip_network("203.0.113.0/24"):
            print(
                "\nNot: Bu adres 203.0.113.0/24 dokümantasyon/test aralığındadır; "
                "gerçek bir saldırgan IP'si olarak değerlendirilmemelidir."
            )
    except ValueError:
        pass

    start = pd.Timestamp("2026-01-26 03:00:00", tz="UTC")
    end = pd.Timestamp("2026-01-26 04:00:00", tz="UTC")
    timeline = ip_events[
        (ip_events["eventtime"] >= start) & (ip_events["eventtime"] < end)
    ]

    print_section("HEDEF IP - 26 OCAK 2026 03:00-04:00 UTC ZAMAN ÇİZELGESİ")
    if timeline.empty:
        print("Bu zaman aralığında olay bulunamadı.")
    else:
        print(
            timeline[
                ["eventtime", "eventname", "eventsource", "errorcode"]
            ].sort_values("eventtime").to_string(index=False)
        )


def detect_behavior_chains(
    data: pd.DataFrame,
    results: pd.DataFrame,
    source_hourly_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect explainable sequences following GetCallerIdentity."""
    anomaly_hours = set(
        results.loc[results["anomaly_label"] == -1, "hour_window"]
    )
    source_anomaly_hours_by_value: dict[str, set[pd.Timestamp]] = {}
    if source_hourly_results is not None and not source_hourly_results.empty:
        source_anomalies = source_hourly_results[
            source_hourly_results["source_anomaly_label"] == -1
        ]
        for source_value, source_rows in source_anomalies.groupby("sourceipaddress"):
            source_anomaly_hours_by_value[str(source_value)] = set(
                source_rows["hour_window"]
            )

    alerts = []
    seen_signatures = set()

    for ip, ip_events in data.dropna(subset=["sourceipaddress"]).groupby(
        "sourceipaddress"
    ):
        ip_events = ip_events.sort_values("eventtime")
        identity_events = ip_events[
            ip_events["eventname"] == "GetCallerIdentity"
        ]

        for _, identity_row in identity_events.iterrows():
            start_time = identity_row["eventtime"]
            end_time = start_time + pd.Timedelta(minutes=ANALYSIS_WINDOW_MINUTES)
            window = ip_events[
                (ip_events["eventtime"] >= start_time)
                & (ip_events["eventtime"] <= end_time)
            ]

            found_discovery = set(
                window.loc[
                    window["eventname"].isin(DISCOVERY_EVENTS), "eventname"
                ]
            )
            critical = window[
                window["eventname"].isin(CRITICAL_EVENT_POINTS)
            ]

            # Gürültüyü azalt: en az 3 farklı keşif işlemi ve kritik olay gerekli.
            if len(found_discovery) < 3 or critical.empty:
                continue

            # Aynı kritik olay kümesini birden fazla GetCallerIdentity penceresi
            # yakalarsa raporda mükerrer alarm üretme.
            signature = (
                str(ip),
                tuple(
                    sorted(
                        (
                            str(row.eventtime),
                            str(row.eventname),
                            str(row.errorcode),
                        )
                        for row in critical.itertuples(index=False)
                    )
                ),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            rule_score = 1
            reasons = ["GetCallerIdentity kimlik sorgusu tespit edildi."]

            rule_score += 2
            reasons.append(
                f"{ANALYSIS_WINDOW_MINUTES} dakika içinde {len(found_discovery)} "
                "farklı altyapı keşif işlemi görüldü: "
                + ", ".join(sorted(found_discovery))
                + "."
            )

            for event_name, points in CRITICAL_EVENT_POINTS.items():
                rows = critical[critical["eventname"] == event_name]
                if rows.empty:
                    continue

                rule_score += points
                denied = rows["errorcode"].astype("string").str.contains(
                    "AccessDenied", na=False
                )

                if denied.all():
                    reasons.append(
                        f"{event_name}: ilgili tüm kayıtlar AccessDenied ile reddedildi."
                    )
                elif denied.any():
                    reasons.append(
                        f"{event_name}: bazı kayıtlar AccessDenied; kalan sonuçlar ayrıca doğrulanmalı."
                    )
                else:
                    reasons.append(
                        f"{event_name}: AccessDenied kaydı yok; bu tek başına başarı kanıtı değildir."
                    )

            window_hours = set(window["eventtime"].dt.floor("h"))
            matching_global_hours = sorted(window_hours.intersection(anomaly_hours))
            source_anomaly_hours = source_anomaly_hours_by_value.get(str(ip), set())
            matching_source_hours = sorted(window_hours.intersection(source_anomaly_hours))
            global_ml_anomaly_present = bool(matching_global_hours)
            source_ml_anomaly_present = bool(matching_source_hours)
            ml_anomaly_present = global_ml_anomaly_present or source_ml_anomaly_present

            if global_ml_anomaly_present:
                reasons.append(
                    "Global Isolation Forest aynı zaman aralığında en az bir saati anomali işaretledi."
                )
            if source_ml_anomaly_present:
                reasons.append(
                    "Kaynak alanı bazlı Isolation Forest bu kaynak için aynı zaman aralığında "
                    "en az bir aktif saati anomali işaretledi."
                )
            if not ml_anomaly_present:
                reasons.append(
                    "Global veya kaynak bazlı Isolation Forest bu zaman aralığında anomali işaretlemedi."
                )
            if ml_anomaly_present:
                reasons.append(
                    "Makine öğrenmesi işaretleri zamansal destek sinyalidir; tek başına saldırı kanıtı değildir."
                )

            if rule_score >= 7:
                severity = "HIGH"
            elif rule_score >= 4:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            error_text = critical["errorcode"].astype("string")
            denied_count = int(
                error_text.str.contains("AccessDenied", na=False).sum()
            )

            if denied_count == len(critical):
                outcome = "ALL_CRITICAL_ACTIONS_DENIED"
                assessment = "SUSPICIOUS_ATTEMPT_BLOCKED"
            elif denied_count > 0:
                outcome = "MIXED_OR_PARTIAL_RESULTS"
                assessment = "REQUIRES_INVESTIGATION"
            else:
                outcome = "NO_ACCESSDENIED_RECORDED"
                assessment = "REQUIRES_INVESTIGATION"

            alerts.append(
                {
                    "sourceipaddress": ip,
                    "start_time": start_time,
                    "end_time": critical["eventtime"].max(),
                    "rule_score": rule_score,
                    "severity": severity,
                    "assessment": assessment,
                    "discovery_count": len(found_discovery),
                    "discovery_events": ", ".join(sorted(found_discovery)),
                    "critical_events": ", ".join(sorted(set(critical["eventname"]))),
                    "critical_event_count": len(critical),
                    "denied_count": denied_count,
                    "outcome": outcome,
                    "ml_anomaly_present": ml_anomaly_present,
                    "global_ml_anomaly_present": global_ml_anomaly_present,
                    "source_ml_anomaly_present": source_ml_anomaly_present,
                    "global_ml_anomaly_hours": ", ".join(
                        str(hour) for hour in matching_global_hours
                    ),
                    "source_ml_anomaly_hours": ", ".join(
                        str(hour) for hour in matching_source_hours
                    ),
                    "detection_reasons": " ".join(reasons),
                }
            )

    columns = [
        "sourceipaddress",
        "start_time",
        "end_time",
        "rule_score",
        "severity",
        "assessment",
        "discovery_count",
        "discovery_events",
        "critical_events",
        "critical_event_count",
        "denied_count",
        "outcome",
        "ml_anomaly_present",
        "global_ml_anomaly_present",
        "source_ml_anomaly_present",
        "global_ml_anomaly_hours",
        "source_ml_anomaly_hours",
        "detection_reasons",
    ]
    return pd.DataFrame(alerts, columns=columns)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    features, results = run_isolation_forest(data)
    source_hourly_results, source_summary = run_source_level_isolation_forest(
        data, results
    )

    inspect_anomaly_results(data, features, results)
    inspect_source_level_results(source_hourly_results, source_summary)
    inspect_general_traffic(data)
    inspect_target_ip(data, TARGET_IP)

    print_section("AÇIKLANABİLİR TEHDİT TESPİT RAPORU")
    alert_df = detect_behavior_chains(data, results, source_hourly_results)

    if alert_df.empty:
        print("Belirlenen davranış zincirine uyan olay bulunamadı.")
    else:
        alert_df = alert_df.sort_values(
            ["rule_score", "start_time"], ascending=[False, True]
        )
        print("Toplam uyarı:", len(alert_df))
        summary_columns = [
            "sourceipaddress",
            "start_time",
            "rule_score",
            "severity",
            "assessment",
            "outcome",
            "ml_anomaly_present",
            "global_ml_anomaly_present",
            "source_ml_anomaly_present",
        ]
        print(alert_df[summary_columns].to_string(index=False))

        for number, (_, alert) in enumerate(alert_df.iterrows(), start=1):
            print(f"\n--- Uyarı {number} ---")
            print("IP:", alert["sourceipaddress"])
            print("Risk puanı:", alert["rule_score"])
            print("Seviye:", alert["severity"])
            print("Değerlendirme:", alert["assessment"])
            print("Sonuç:", alert["outcome"])
            print("Tespit nedenleri:")
            print(alert["detection_reasons"])

    # CSV outputs: global anomalies, source-aware model and explainable alerts.
    results.to_csv(OUTPUT_DIR / "hourly_anomaly_results.csv", index=False)
    source_hourly_results.to_csv(
        OUTPUT_DIR / "source_hourly_anomaly_results.csv", index=False
    )
    source_summary.to_csv(OUTPUT_DIR / "source_anomaly_summary.csv", index=False)
    alert_df.to_csv(OUTPUT_DIR / "behavior_alerts.csv", index=False)

    print_section("DOSYALAR KAYDEDİLDİ")
    print(OUTPUT_DIR / "hourly_anomaly_results.csv")
    print(OUTPUT_DIR / "source_hourly_anomaly_results.csv")
    print(OUTPUT_DIR / "source_anomaly_summary.csv")
    print(OUTPUT_DIR / "behavior_alerts.csv")
    print("\nAnaliz tamamlandı.")
    print(
        "Not: HIGH bir kural puanı seviyesidir; doğrulanmış saldırı anlamına gelmez. "
        "AccessDenied olmayan kayıtlar da ayrıntılı CloudTrail kanıtı olmadan başarılı kabul edilmez."
    )


if __name__ == "__main__":
    main()
