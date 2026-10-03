"""Rule-based statement extraction (spec section A1 / A2).

Deterministic, no training: regex and keyword heuristics only. Operates on
client turns (AnnoMI) or an ASR transcript (fictional audio) and produces a
`StatementRecord` that traces every extracted item back to its source turn,
preserves negation/uncertainty, and never labels anything as a depression
diagnosis or severity score.
"""
from __future__ import annotations

import re

from common.schemas import StatementRecord
from track_a.ingest_annomi import Conversation, Turn

_SELF_REPORT_RE = re.compile(
    r"\bi\b[^.!?]*\b(feel|felt|feeling|am|m|have|ve|had|don't|dont|can't|cant|"
    r"couldn't|couldnt|was|been|get|got|think|guess|know|sleep|sleeping|eat|eating)\b",
    re.IGNORECASE,
)
# "n't" sits inside a word ("don't"), so it can't use a leading \b.
_NEGATION_RE = re.compile(r"\b(not|never|no|nothing|none|nobody|neither|without)\b|n['’]t\b", re.IGNORECASE)
_UNCERTAINTY_RE = re.compile(
    r"\b(maybe|perhaps|might|possibly|kind of|sort of|not sure|i guess|i think|probably)\b",
    re.IGNORECASE,
)
_QUANTITY = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|a few|few|several|many|a couple(?: of)?|couple(?: of)?)"
)
_UNIT = r"(?:hour|day|week|month|year|decade)s?(?! old)"
_AMOUNT = rf"(?:{_QUANTITY}\s+(?:or (?:so|more|two|three)\s+)?(?:to {_QUANTITY}\s+)?{_UNIT}|{_UNIT})"
# Time-span phrases only. Rates ("a pack a day", "twice a week") and ages
# ("17 years old") are deliberately not durations.
_DURATION_RE = re.compile(
    r"\b("
    rf"for (?:the (?:past|last) |about |almost |nearly |around |over |like |maybe )?{_AMOUNT}(?: or so)?"
    rf"|(?:about|almost|nearly|around|over) {_QUANTITY}\s+{_UNIT}(?: or so)?(?! a\b)"
    rf"|(?:over|in|during|for) the (?:past|last) (?:{_QUANTITY}\s+)?{_UNIT}(?: or so)?"
    rf"|{_QUANTITY}\s+(?:to {_QUANTITY}\s+)?{_UNIT}\s+(?:or so\s+)?ago"
    r"|(?:ever )?since (?:i |I |last |the |my |high school|college|then|childhood)[\w' ]{0,30}?(?=[,.;!?]|$| and | but )"
    r"|for (?:a (?:long )?while|a long time|ages|years|months|weeks|so long)"
    r"|lately|recently|these days|these past (?:few )?\w+"
    r")",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"\b(stopped|can't sleep|cant sleep|trouble sleeping|lost interest|less energy|"
    r"more tired|can't concentrate|cant concentrate|stopped eating|eating more|eating less|"
    r"sleeping more|sleeping less|used to)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?\s*$")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_from_client_turns(turns: list[Turn]) -> StatementRecord:
    """Extract only from turns spoken by the client — a therapist's question
    or hypothesis is never treated as the person's own report."""
    record = StatementRecord()
    for turn in turns:
        if turn.interlocutor != "client":
            continue
        for sentence in _split_sentences(turn.text):
            if _QUESTION_RE.search(sentence):
                continue  # the client asking something back is not a self-report
            used = False

            # Duration/change are checked on every client sentence, not only
            # first-person ones: a bare answer like "About a month." to a
            # therapist's "how long?" is still the client's own report.
            if record.duration_if_said is None:
                m = _DURATION_RE.search(sentence)
                if m:
                    record.duration_if_said = m.group(0).strip()
                    used = True

            if record.changes_in_functioning_if_said is None:
                m = _CHANGE_RE.search(sentence)
                if m:
                    record.changes_in_functioning_if_said = m.group(0)
                    used = True

            if _SELF_REPORT_RE.search(sentence):
                is_negated = bool(_NEGATION_RE.search(sentence))
                is_uncertain = bool(_UNCERTAINTY_RE.search(sentence))
                if is_negated:
                    record.explicit_negations.append(sentence)
                else:
                    label = sentence + (" [uncertain]" if is_uncertain else "")
                    record.reported_concerns.append(label)
                used = True

            if used and turn.utterance_id not in record.source_turn_ids:
                record.source_turn_ids.append(turn.utterance_id)

    if record.duration_if_said is None:
        record.unknown_fields.append("duration")
    if record.changes_in_functioning_if_said is None:
        record.unknown_fields.append("changes_in_functioning")
    return record


def extract_from_conversation(conversation: Conversation) -> StatementRecord:
    return extract_from_client_turns(conversation.turns)


def extract_from_transcript(transcript: str) -> StatementRecord:
    """Same heuristics applied to a single-speaker ASR transcript (no
    therapist/client roles to filter, so the whole transcript is treated as
    the speaker's own statement)."""
    fake_turn = Turn(utterance_id=0, interlocutor="client", timestamp="00:00:00", text=transcript)
    return extract_from_client_turns([fake_turn])
