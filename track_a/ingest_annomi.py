"""AnnoMI dialogue ingestion (spec section A1).

Loads AnnoMI-simple.csv (or AnnoMI-full.csv), validates it, and groups
utterances into per-conversation turn sequences ordered by utterance_id.
AnnoMI's mi_quality / client_talk_type labels are kept as-is and are never
reinterpreted as depression diagnoses or severity here or downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "transcript_id",
    "utterance_id",
    "interlocutor",
    "timestamp",
    "utterance_text",
]

_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


@dataclass
class IngestionReport:
    n_rows: int
    n_conversations: int
    missing_field_rows: list[int] = field(default_factory=list)
    duplicate_turns: list[tuple[int, int]] = field(default_factory=list)
    unparsable_timestamps: list[tuple[int, int, str]] = field(default_factory=list)
    non_monotonic_timestamps: list[tuple[int, int]] = field(default_factory=list)
    unknown_interlocutor_rows: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class Turn:
    utterance_id: int
    interlocutor: str  # "therapist" | "client"
    timestamp: str
    text: str
    main_therapist_behaviour: str | None = None
    client_talk_type: str | None = None


@dataclass
class Conversation:
    transcript_id: int
    mi_quality: str
    topic: str
    video_url: str
    turns: list[Turn]

    def history_up_to(self, utterance_id: int) -> list[Turn]:
        """Turns strictly before `utterance_id`, in order — used to resolve
        references to earlier turns (e.g. a question callback)."""
        return [t for t in self.turns if t.utterance_id < utterance_id]


def _timestamp_to_seconds(ts: str) -> int | None:
    if not _TIMESTAMP_RE.match(ts or ""):
        return None
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def load_annomi(csv_path: str | Path) -> tuple[pd.DataFrame, IngestionReport]:
    df = pd.read_csv(csv_path, dtype={"transcript_id": int, "utterance_id": int})

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"AnnoMI CSV missing required columns: {missing_cols}")

    report = IngestionReport(n_rows=len(df), n_conversations=df["transcript_id"].nunique())

    # Missing fields in required columns
    missing_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)
    report.missing_field_rows = df.index[missing_mask].tolist()

    # Duplicate (transcript_id, utterance_id) pairs
    dup_mask = df.duplicated(subset=["transcript_id", "utterance_id"], keep=False)
    report.duplicate_turns = sorted(
        set(
            (int(r.transcript_id), int(r.utterance_id))
            for r in df.loc[dup_mask, ["transcript_id", "utterance_id"]].itertuples()
        )
    )

    # Unknown interlocutor values
    valid_roles = {"therapist", "client"}
    bad_role_mask = ~df["interlocutor"].isin(valid_roles)
    for r in df.loc[bad_role_mask, ["transcript_id", "utterance_id", "interlocutor"]].itertuples():
        report.unknown_interlocutor_rows.append((int(r.transcript_id), int(r.utterance_id), str(r.interlocutor)))

    # Timestamp checks, per conversation, in utterance_id order
    for transcript_id, group in df.sort_values("utterance_id").groupby("transcript_id"):
        prev_seconds = None
        for row in group.itertuples():
            seconds = _timestamp_to_seconds(str(row.timestamp))
            if seconds is None:
                report.unparsable_timestamps.append((int(transcript_id), int(row.utterance_id), str(row.timestamp)))
                continue
            if prev_seconds is not None and seconds < prev_seconds:
                report.non_monotonic_timestamps.append((int(transcript_id), int(row.utterance_id)))
            prev_seconds = seconds

    return df, report


def build_conversations(df: pd.DataFrame) -> dict[int, Conversation]:
    conversations: dict[int, Conversation] = {}
    for transcript_id, group in df.groupby("transcript_id"):
        ordered = group.sort_values("utterance_id")  # order by utterance_id, not row order
        first = ordered.iloc[0]
        turns = [
            Turn(
                utterance_id=int(r.utterance_id),
                interlocutor=str(r.interlocutor),
                timestamp=str(r.timestamp),
                text=str(r.utterance_text),
                main_therapist_behaviour=getattr(r, "main_therapist_behaviour", None),
                client_talk_type=getattr(r, "client_talk_type", None),
            )
            for r in ordered.itertuples()
        ]
        conversations[int(transcript_id)] = Conversation(
            transcript_id=int(transcript_id),
            mi_quality=str(first["mi_quality"]),
            topic=str(first["topic"]),
            video_url=str(first["video_url"]),
            turns=turns,
        )
    return conversations
