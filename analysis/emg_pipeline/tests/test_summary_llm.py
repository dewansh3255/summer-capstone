"""
Unit tests for Phase D: summary building + LLM interpretation.

All tests run offline with no API key: the LLM path is exercised through
MockLLMClient (which wraps the deterministic rule-based report), and the
summary builders are tested on synthetic features.
"""

import json
import numpy as np
import pytest

from emg_pipeline.delsys_parser import EMGChannel, DelsysRecording
from emg_pipeline.features import extract_channel_features, extract_recording_features
from emg_pipeline.summary import build_sensor_summary, build_recording_summary
from emg_pipeline.llm import (
    build_prompt, rule_based_report, interpret_fatigue,
    MockLLMClient, FATIGUE_SYSTEM_PROMPT, LLMClient,
)

FS = 1000.0


def _channel(sig, sid="1", fs=FS):
    t = np.arange(len(sig)) / fs
    return EMGChannel(sid, f"Sensor {sid}", t, sig, fs)


def _fatiguing_signal(dur=10.0, fs=FS):
    n = int(dur * fs)
    t = np.arange(n) / fs
    freq = np.linspace(150, 80, n)      # falling frequency
    amp = np.linspace(1.0, 2.0, n)      # rising amplitude
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return amp * np.sin(phase)


def _stationary_signal(dur=10.0, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return np.sin(2 * np.pi * 100 * t)


# ──────────────────────────────────────────────────────────────────────────────
# Sensor summary
# ──────────────────────────────────────────────────────────────────────────────

class TestSensorSummary:
    def test_basic_fields(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf)
        assert s["sensor_id"] == "1"
        assert s["label"] == "Sensor 1"
        assert "amplitude" in s and "frequency" in s
        assert "fatigue_detected" in s

    def test_custom_label(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf, label="Biceps Brachii")
        assert s["label"] == "Biceps Brachii"

    def test_fatigue_flag_true_on_fatiguing(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf)
        assert s["fatigue_detected"] is True
        assert s["amplitude"]["rms_pct_change"] > 0
        assert s["frequency"]["mdf_pct_change"] < 0

    def test_fatigue_flag_false_on_stationary(self):
        cf = extract_channel_features(_channel(_stationary_signal()))
        s = build_sensor_summary(cf)
        assert s["fatigue_detected"] is False

    def test_pct_mvc_included_when_available(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()), mvc_reference=2.0)
        s = build_sensor_summary(cf)
        assert "mean_pct_mvc" in s["amplitude"]
        assert "peak_pct_mvc" in s["amplitude"]

    def test_pct_mvc_absent_without_reference(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf)
        assert "mean_pct_mvc" not in s["amplitude"]

    def test_activation_count_optional(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf, activation_count=7)
        assert s["activation_count"] == 7

    def test_json_serialisable(self):
        cf = extract_channel_features(_channel(_fatiguing_signal()))
        s = build_sensor_summary(cf)
        json.dumps(s)   # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# Recording summary
# ──────────────────────────────────────────────────────────────────────────────

class TestRecordingSummary:
    def _recording(self):
        return DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="t", datetime="now", collection_length_s=10.0,
            channels={
                "1": _channel(_fatiguing_signal(), sid="1"),
                "2": _channel(_stationary_signal(), sid="2"),
            },
        )

    def test_counts(self):
        feats = extract_recording_features(self._recording())
        summ = build_recording_summary(feats)
        assert summ["n_sensors"] == 2
        assert summ["n_fatigued"] >= 1
        assert len(summ["sensors"]) == 2

    def test_most_fatigued_is_fatiguing_sensor(self):
        feats = extract_recording_features(self._recording())
        summ = build_recording_summary(feats)
        # Sensor 1 is the fatiguing one
        assert summ["most_fatigued"] == "Sensor 1"

    def test_labels_applied(self):
        feats = extract_recording_features(self._recording())
        summ = build_recording_summary(feats, labels={"1": "Biceps", "2": "Triceps"})
        labels = [s["label"] for s in summ["sensors"]]
        assert "Biceps" in labels and "Triceps" in labels

    def test_task_name(self):
        feats = extract_recording_features(self._recording())
        summ = build_recording_summary(feats, task_name="push task")
        assert summ["task"] == "push task"

    def test_json_serialisable(self):
        feats = extract_recording_features(self._recording())
        summ = build_recording_summary(feats)
        json.dumps(summ)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

class TestPrompt:
    def test_returns_system_and_user(self):
        system, user = build_prompt({"n_sensors": 0, "sensors": []})
        assert system == FATIGUE_SYSTEM_PROMPT
        assert "json" in user.lower()

    def test_user_contains_summary_json(self):
        summary = {"n_sensors": 1, "sensors": [{"label": "Sensor 9"}]}
        _, user = build_prompt(summary)
        assert "Sensor 9" in user

    def test_system_prompt_has_fatigue_rules(self):
        # Guardrail + domain rules must be present
        assert "MDF" in FATIGUE_SYSTEM_PROMPT
        assert "NEVER invent" in FATIGUE_SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based report
# ──────────────────────────────────────────────────────────────────────────────

class TestRuleBasedReport:
    def _summary(self):
        rec = DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="t", datetime="now", collection_length_s=10.0,
            channels={
                "1": _channel(_fatiguing_signal(), sid="1"),
                "2": _channel(_stationary_signal(), sid="2"),
            },
        )
        feats = extract_recording_features(rec)
        return build_recording_summary(feats, task_name="push task")

    def test_report_is_string(self):
        assert isinstance(rule_based_report(self._summary()), str)

    def test_report_mentions_task(self):
        assert "push task" in rule_based_report(self._summary())

    def test_report_mentions_fatigue(self):
        report = rule_based_report(self._summary())
        assert "FATIGUE" in report or "fatigue" in report

    def test_report_lists_all_sensors(self):
        report = rule_based_report(self._summary())
        assert "Sensor 1" in report
        assert "Sensor 2" in report

    def test_report_no_fatigue_case(self):
        rec = DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="t", datetime="now", collection_length_s=10.0,
            channels={"2": _channel(_stationary_signal(), sid="2")},
        )
        feats = extract_recording_features(rec)
        summ = build_recording_summary(feats)
        report = rule_based_report(summ)
        assert "No sensor met the full fatigue signature" in report


# ──────────────────────────────────────────────────────────────────────────────
# interpret_fatigue + MockLLMClient
# ──────────────────────────────────────────────────────────────────────────────

class TestInterpret:
    def _summary(self):
        rec = DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="t", datetime="now", collection_length_s=10.0,
            channels={"1": _channel(_fatiguing_signal(), sid="1")},
        )
        feats = extract_recording_features(rec)
        return build_recording_summary(feats)

    def test_without_client_uses_rule_based(self):
        summ = self._summary()
        out = interpret_fatigue(summ)            # no client
        assert out == rule_based_report(summ)

    def test_with_mock_client(self):
        summ = self._summary()
        client = MockLLMClient(summ)
        out = interpret_fatigue(summ, client)
        assert isinstance(out, str) and len(out) > 0

    def test_mock_client_receives_prompt(self):
        summ = self._summary()
        client = MockLLMClient(summ)
        interpret_fatigue(summ, client)
        # The system prompt and user message were passed through
        assert client.last_system == FATIGUE_SYSTEM_PROMPT
        assert client.last_user is not None
        assert "json" in client.last_user.lower()

    def test_mock_client_satisfies_protocol(self):
        summ = self._summary()
        client = MockLLMClient(summ)
        assert isinstance(client, LLMClient)
