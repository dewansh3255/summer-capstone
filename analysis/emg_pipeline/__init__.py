"""
emg_pipeline
------------
EMG analysis pipeline for Delsys Trigno Discover recordings.

Phase A: parsing + preprocessing   (delsys_parser, preprocess)   [done]
Phase B: feature extraction         (features)                    [done]
Phase C: analysis + visualisation   (analysis)                    [done]
Phase D: LLM interpretation         (summary, llm)                [done]
"""

from .delsys_parser import (
    DelsysRecording,
    EMGChannel,
    parse_delsys_csv,
)
from .preprocess import (
    bandpass_filter,
    notch_filter,
    preprocess_channel,
    preprocess_recording,
)
from .features import (
    ChannelFeatures,
    FatigueMetrics,
    extract_channel_features,
    extract_recording_features,
    features_to_long_frame,
    compute_mvc_reference,
    combine_mvc_references,
    compute_fatigue_metrics,
    fatigue_summary_frame,
)
from .analysis import (
    ActivationEvent,
    teager_kaiser_energy,
    linear_envelope,
    detect_onsets,
    plot_rms_trend,
    plot_mdf_trend,
    plot_fatigue_ranking,
    plot_channel_overview,
)
from .summary import (
    build_sensor_summary,
    build_recording_summary,
)
from .llm import (
    LLMClient,
    OpenAIClient,
    MockLLMClient,
    FATIGUE_SYSTEM_PROMPT,
    build_prompt,
    rule_based_report,
    interpret_fatigue,
)

__all__ = [
    # Phase A
    "DelsysRecording",
    "EMGChannel",
    "parse_delsys_csv",
    "bandpass_filter",
    "notch_filter",
    "preprocess_channel",
    "preprocess_recording",
    # Phase B
    "ChannelFeatures",
    "FatigueMetrics",
    "extract_channel_features",
    "extract_recording_features",
    "features_to_long_frame",
    "compute_mvc_reference",
    "combine_mvc_references",
    "compute_fatigue_metrics",
    "fatigue_summary_frame",
    # Phase C
    "ActivationEvent",
    "teager_kaiser_energy",
    "linear_envelope",
    "detect_onsets",
    "plot_rms_trend",
    "plot_mdf_trend",
    "plot_fatigue_ranking",
    "plot_channel_overview",
    # Phase D
    "build_sensor_summary",
    "build_recording_summary",
    "LLMClient",
    "OpenAIClient",
    "MockLLMClient",
    "FATIGUE_SYSTEM_PROMPT",
    "build_prompt",
    "rule_based_report",
    "interpret_fatigue",
]
