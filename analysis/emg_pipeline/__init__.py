"""
emg_pipeline
------------
EMG analysis pipeline for Delsys Trigno Discover recordings.

Phase A: parsing + preprocessing  (delsys_parser, preprocess)
Phase B: feature extraction        (features)        [later]
Phase C: analysis + visualisation  (analysis)        [later]
Phase D: LLM interpretation        (llm)             [later]
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

__all__ = [
    "DelsysRecording",
    "EMGChannel",
    "parse_delsys_csv",
    "bandpass_filter",
    "notch_filter",
    "preprocess_channel",
    "preprocess_recording",
]
