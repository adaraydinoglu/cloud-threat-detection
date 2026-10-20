"""Synthetic tests for the CloudTrail behavior-chain detector.

Run from the project directory:
    python test_detection_rules.py

These tests use synthetic examples, not the real dataset. They validate rule
logic; they do not measure production precision/recall without labeled data.
"""

import unittest
import pandas as pd

from train_isolation_forest import detect_behavior_chains


BASE_TIME = pd.Timestamp("2026-01-26 03:00:00", tz="UTC")
TEST_IP = "198.51.100.77"  # Reserved documentation IP; synthetic only.
DISCOVERY_EVENTS = [
    "DescribeRegions",
    "DescribeInstances",
    "DescribeSecurityGroups",
    "DescribeSubnets",
]


def make_event(minute: int, event_name: str, error_code=None, source_ip=TEST_IP):
    event_time = BASE_TIME + pd.Timedelta(minutes=minute)
    return {
        "eventtime": event_time,
        "hour_window": event_time.floor("h"),
        "eventname": event_name,
        "sourceipaddress": source_ip,
        "errorcode": error_code,
    }


def make_results(anomaly: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_window": [BASE_TIME.floor("h")],
            "anomaly_label": [-1 if anomaly else 1],
            "anomaly_score": [-0.25 if anomaly else 0.1],
        }
    )


def normal_events() -> pd.DataFrame:
    # Identity check plus discovery-like reads, but no critical action.
    rows = [make_event(0, "GetCallerIdentity")]
    rows.extend(
        make_event(2 + index, event_name)
        for index, event_name in enumerate(DISCOVERY_EVENTS[:3])
    )
    return pd.DataFrame(rows)


def blocked_suspicious_events() -> pd.DataFrame:
    rows = [make_event(0, "GetCallerIdentity")]
    rows.extend(
        make_event(2 + index, event_name)
        for index, event_name in enumerate(DISCOVERY_EVENTS)
    )
    rows.extend(
        [
            make_event(31, "AssumeRole", "AccessDenied"),
            make_event(33, "StopLogging", "AccessDenied"),
            make_event(34, "DeleteTrail", "AccessDenied"),
        ]
    )
    return pd.DataFrame(rows)


def mixed_outcome_events() -> pd.DataFrame:
    rows = [make_event(0, "GetCallerIdentity")]
    rows.extend(
        make_event(2 + index, event_name)
        for index, event_name in enumerate(DISCOVERY_EVENTS)
    )
    rows.extend(
        [
            make_event(31, "AssumeRole", "AccessDenied"),
            make_event(33, "StopLogging", None),
            make_event(34, "DeleteTrail", "AccessDenied"),
        ]
    )
    return pd.DataFrame(rows)


def critical_events_without_identity() -> pd.DataFrame:
    rows = [
        make_event(2 + index, event_name)
        for index, event_name in enumerate(DISCOVERY_EVENTS)
    ]
    rows.extend(
        [
            make_event(31, "AssumeRole", "AccessDenied"),
            make_event(33, "StopLogging", "AccessDenied"),
            make_event(34, "DeleteTrail", "AccessDenied"),
        ]
    )
    return pd.DataFrame(rows)


class BehaviorDetectionTests(unittest.TestCase):
    def test_normal_activity_does_not_alert(self):
        alerts = detect_behavior_chains(normal_events(), make_results(anomaly=False))
        self.assertTrue(alerts.empty, "Normal activity unexpectedly produced an alert")

    def test_suspicious_chain_all_denied_is_detected(self):
        alerts = detect_behavior_chains(
            blocked_suspicious_events(), make_results(anomaly=True)
        )
        self.assertEqual(len(alerts), 1)
        alert = alerts.iloc[0]
        self.assertEqual(alert["severity"], "HIGH")
        self.assertEqual(alert["rule_score"], 11)
        self.assertEqual(alert["outcome"], "ALL_CRITICAL_ACTIONS_DENIED")
        self.assertEqual(alert["assessment"], "SUSPICIOUS_ATTEMPT_BLOCKED")
        self.assertEqual(alert["denied_count"], 3)
        self.assertTrue(alert["ml_anomaly_present"])

    def test_mixed_outcome_requires_investigation(self):
        alerts = detect_behavior_chains(
            mixed_outcome_events(), make_results(anomaly=False)
        )
        self.assertEqual(len(alerts), 1)
        alert = alerts.iloc[0]
        self.assertEqual(alert["outcome"], "MIXED_OR_PARTIAL_RESULTS")
        self.assertEqual(alert["assessment"], "REQUIRES_INVESTIGATION")
        self.assertEqual(alert["denied_count"], 2)
        self.assertFalse(alert["ml_anomaly_present"])

    def test_critical_actions_without_identity_anchor_do_not_alert(self):
        alerts = detect_behavior_chains(
            critical_events_without_identity(), make_results(anomaly=True)
        )
        self.assertTrue(
            alerts.empty,
            "A chain without GetCallerIdentity unexpectedly produced an alert",
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BehaviorDetectionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
