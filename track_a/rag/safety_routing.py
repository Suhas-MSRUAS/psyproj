"""Fixture-based support/review/urgent routing (spec section A3).

No clinician-reviewed organizer protocol was supplied for this build, so per
the spec this demonstrates **software routing only**, using a fixed keyword
fixture list with explicitly artificial labels. This is NOT clinical triage
and must never be presented as one. Immediate safety concerns are routed to
a distinct human/emergency escalation category rather than left to free-form
LLM judgement.
"""
from __future__ import annotations

import re

from common.schemas import RoutingResult

# Unambiguous, explicit risk wording -> "urgent" unless locally negated.
# Phrases that carry their own negation ("don't want to be alive") are listed
# whole, so the local-negation check below only looks *before* the phrase.
_RISK_PHRASES = [
    "kill myself", "killing myself", "end my life", "ending my life",
    "take my own life", "taking my own life", "end it all",
    "want to die", "want to be dead", "better off dead", "better off without me",
    "don't want to be alive", "dont want to be alive", "do not want to be alive",
    "don't want to live", "dont want to live", "do not want to live",
    "don't want to be here anymore", "dont want to be here anymore",
    "not worth living",
    "hurt myself", "hurting myself", "harm myself", "harming myself",
    "self-harm", "self harm", "no reason to live",
    "can't go on", "cant go on", "can't take it anymore", "cant take it anymore",
    "suicide", "suicidal",
]

# Distress wording that is concerning but not an explicit risk statement -> "review".
_MODERATE_PHRASES = ["hopeless", "worthless", "giving up", "overwhelmed", "can't cope", "cant cope"]

_NEGATION_CUES = ["not ", "n't ", "never ", "no longer ", "without ", "don't ", "dont "]

_WINDOW_CHARS = 30


def classify(text: str) -> RoutingResult:
    lower = text.lower()

    for phrase in _RISK_PHRASES:
        idx = lower.find(phrase)
        if idx == -1:
            continue
        window = lower[max(0, idx - _WINDOW_CHARS): idx]
        negated = any(cue in window for cue in _NEGATION_CUES)
        if negated:
            return RoutingResult(
                category="review",
                rationale=(
                    f"Risk phrase '{phrase}' detected but immediately preceded by a "
                    "negation cue; routed to human review rather than auto-cleared or "
                    "auto-escalated, since negated risk mentions still warrant a careful "
                    "look rather than being dismissed by keyword matching alone."
                ),
            )
        return RoutingResult(
            category="urgent",
            rationale=(
                f"Unambiguous risk phrase '{phrase}' detected with no negation nearby; "
                "routed to the distinct human/emergency escalation path, not a "
                "free-form LLM severity judgement."
            ),
        )

    for phrase in _MODERATE_PHRASES:
        if phrase in lower:
            return RoutingResult(
                category="review",
                rationale=f"Moderate distress phrase '{phrase}' matched the fixed fixture list; routed for human review.",
            )

    return RoutingResult(
        category="support",
        rationale="No risk or distress phrase in the fixed fixture list matched; routed to general support/information path.",
    )
