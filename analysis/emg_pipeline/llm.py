"""
llm.py
------
Phase D (part 2): LLM interpretation of the EMG feature summary.

Pieces
------
- FATIGUE_SYSTEM_PROMPT : domain-grounded system prompt encoding the EMG
  fatigue rules from the literature (RMS up + MDF/MNF down => fatigue), and a
  hard guardrail that the model must ONLY interpret the supplied numbers and
  never invent values.
- build_prompt(summary)  : returns (system, user) messages from a summary dict.
- LLMClient (Protocol)   : minimal interface — .complete(system, user) -> str.
- OpenAIClient           : real OpenAI implementation (lazy import; API key from
                           the OPENAI_API_KEY env var or passed explicitly).
- rule_based_report(summary) : deterministic, no-LLM interpreter. Works fully
                           offline, needs no API key, and serves as both a
                           fallback and the test oracle.
- MockLLMClient          : wraps rule_based_report so the whole interpret path
                           can be exercised in tests without network access.
- interpret_fatigue(summary, client=None) : the top-level entry point. With no
                           client it uses the rule-based report; with a client
                           it calls the LLM.

Design intent: the numeric analysis is done by Phases B/C. The LLM (or the
rule-based fallback) only turns those numbers into readable, contextualised
prose. This separation keeps results reproducible and defensible.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Protocol, runtime_checkable


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


FATIGUE_SYSTEM_PROMPT = """\
You are an expert in surface electromyography (sEMG) signal interpretation,
assisting a biomechanics researcher.

You will be given a JSON summary of pre-computed sEMG features for one or more
sensors from a single recording. Each number has already been computed by a
validated signal-processing pipeline. Your job is to INTERPRET these numbers
into a clear, concise report on muscle activation and fatigue.

Domain rules you must apply (established sEMG fatigue science):
- Muscle FATIGUE during sustained/repeated effort is indicated by:
    * RMS amplitude RISING over time (rms_slope > 0, positive rms_pct_change), AND
    * Median frequency (MDF) and mean frequency (MNF) FALLING over time
      (mdf_slope < 0, negative mdf_pct_change).
  The spectral compression toward lower frequencies is the most reliable
  fatigue indicator; rising amplitude alone can also reflect increased effort.
- Higher RMS / %MVC indicates stronger muscle ACTIVATION.
- A flat MDF with stable RMS indicates little or no fatigue.
- %MVC (percent of maximum voluntary contraction) contextualises effort level:
  higher %MVC means the muscle worked closer to its maximum.

Strict rules:
1. ONLY interpret the numbers provided. NEVER invent or assume values that are
   not in the summary. If a quantity is missing, say so rather than guessing.
2. Refer to each sensor by its "label" field.
3. Be quantitative: cite the actual numbers (e.g. "MDF fell from 56.0 to 44.0 Hz").
4. Be concise and structured. Prefer short paragraphs or bullet points.
5. Do not give medical or clinical advice; this is a biomechanics analysis.

Produce:
- A one-line overall summary.
- A short per-sensor interpretation (activation level + fatigue assessment).
- A closing note on which sensor(s) showed the clearest fatigue and why.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Client interface + implementations
# ──────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM interface: take system+user messages, return the reply text."""
    def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    """
    OpenAI Chat Completions implementation.

    The `openai` package is imported lazily so the rest of the pipeline (and the
    test suite) has no hard dependency on it. The API key is read from the
    OPENAI_API_KEY environment variable unless passed explicitly.

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        report = interpret_fatigue(summary, client)
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        try:
            import openai  # lazy import
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key. Set the OPENAI_API_KEY environment variable "
                "or pass api_key=... to OpenAIClient."
            )

        self._openai = openai
        self.client = openai.OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class MockLLMClient:
    """
    Test/offline client. Ignores the prompt text and produces the deterministic
    rule-based report from the summary it is constructed with. Lets the full
    interpret path run with no network and no API key.
    """
    def __init__(self, summary: dict):
        self._summary = summary
        self.last_system: Optional[str] = None
        self.last_user: Optional[str] = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return rule_based_report(self._summary)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(summary: dict) -> tuple[str, str]:
    """
    Build (system, user) messages for the LLM from a recording summary dict
    (see summary.build_recording_summary).
    """
    user = (
        "Interpret the following sEMG feature summary. Report on muscle "
        "activation and fatigue, citing the actual numbers.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```"
    )
    return FATIGUE_SYSTEM_PROMPT, user


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based interpreter (deterministic, no LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _interpret_sensor(s: dict) -> str:
    """One-sensor textual interpretation from the summary dict."""
    label = s["label"]
    amp = s["amplitude"]
    freq = s["frequency"]
    fatigued = s["fatigue_detected"]

    parts = []

    # Activation level (prefer %MVC if available)
    if "mean_pct_mvc" in amp:
        parts.append(
            f"mean activation {amp['mean_pct_mvc']:.0f}%MVC "
            f"(peak {amp.get('peak_pct_mvc', float('nan')):.0f}%MVC)"
        )
    else:
        parts.append(f"RMS {amp['rms_start_mv']:.4f}->{amp['rms_end_mv']:.4f} mV")

    # Fatigue assessment
    if fatigued:
        assessment = (
            f"FATIGUE detected: amplitude rose {amp['rms_pct_change']:+.0f}% while "
            f"MDF fell from {freq['mdf_start_hz']:.0f} to {freq['mdf_end_hz']:.0f} Hz "
            f"({freq['mdf_pct_change']:+.0f}%, slope {freq['mdf_slope_hz_per_s']:+.4f} Hz/s)"
        )
    else:
        if freq["mdf_slope_hz_per_s"] < 0:
            assessment = (
                f"partial/uncertain: MDF declined "
                f"({freq['mdf_start_hz']:.0f}->{freq['mdf_end_hz']:.0f} Hz) but the "
                f"amplitude trend did not rise (rms {amp['rms_pct_change']:+.0f}%)"
            )
        else:
            assessment = (
                f"no clear fatigue: MDF stable/rising "
                f"({freq['mdf_start_hz']:.0f}->{freq['mdf_end_hz']:.0f} Hz)"
            )

    line = f"- {label}: {', '.join(parts)}; {assessment}."
    if "activation_count" in s:
        line += f" {s['activation_count']} activations detected."
    return line


def rule_based_report(summary: dict) -> str:
    """
    Deterministic natural-language report generated directly from the summary
    dict — no LLM required. Used as an offline fallback and as the test oracle.
    """
    lines: list[str] = []

    task = summary.get("task")
    header = f"EMG fatigue analysis{f' — {task}' if task else ''}"
    lines.append(header)
    lines.append("=" * len(header))

    n = summary.get("n_sensors", 0)
    nf = summary.get("n_fatigued", 0)
    lines.append(
        f"Overall: {nf} of {n} sensor(s) show the fatigue signature "
        f"(rising amplitude + falling median frequency)."
    )
    if summary.get("most_fatigued"):
        lines.append(f"Most fatigued: {summary['most_fatigued']}.")

    lines.append("")
    lines.append("Per-sensor:")
    for s in summary.get("sensors", []):
        lines.append(_interpret_sensor(s))

    fatigued = summary.get("fatigued_sensors", [])
    lines.append("")
    if fatigued:
        lines.append(
            "Clearest fatigue in: " + ", ".join(fatigued) +
            " — these show both rising RMS and falling MDF, the canonical sEMG "
            "fatigue signature."
        )
    else:
        lines.append(
            "No sensor met the full fatigue signature (rising RMS AND falling MDF)."
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def interpret_fatigue(
    summary: dict,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Produce a natural-language fatigue/activation report from a recording
    summary.

    Parameters
    ----------
    summary : dict
        Output of summary.build_recording_summary.
    client : LLMClient, optional
        An LLM client (e.g. OpenAIClient). If None, the deterministic
        rule_based_report is used — no API key or network required.

    Returns
    -------
    str
        The interpretation report.
    """
    if client is None:
        return rule_based_report(summary)
    system, user = build_prompt(summary)
    return client.complete(system, user)
