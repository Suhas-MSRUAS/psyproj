"""Output contracts shared across Track A and Track B.

Section 10 of the spec requires independent, machine-readable results with
distinct identifier namespaces, so a Track A id always starts with
``annomi-`` or ``audio-`` and a Track B id always starts with ``wisdm-`` or
``ucihar-``. Neither schema references the other track's fields.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------


class SourceTurn(BaseModel):
    utterance_id: int
    interlocutor: Literal["therapist", "client"]
    text: str


class StatementRecord(BaseModel):
    """A structured, source-traceable extraction of what a person reported."""

    reported_concerns: list[str] = Field(default_factory=list)
    duration_if_said: Optional[str] = None
    changes_in_functioning_if_said: Optional[str] = None
    explicit_negations: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    source_turn_ids: list[int] = Field(default_factory=list)


class ConversationOutput(BaseModel):
    """Track A output for one AnnoMI conversation. Id namespace: annomi-*"""

    conversation_id: str
    mi_quality: str
    topic: str
    turns: list[SourceTurn]
    statements: list[StatementRecord]


class ASREvaluation(BaseModel):
    case_id: str
    reference_transcript: str
    hypothesis_transcript: str
    word_error_rate: float
    low_confidence_spans: list[str] = Field(default_factory=list)
    critical_word_flags: list[str] = Field(default_factory=list)
    negation_preserved: bool


class AudioCaseOutput(BaseModel):
    """Track A output for one fictional-audio case. Id namespace: audio-*"""

    case_id: str
    transcript: str
    asr_evaluation: ASREvaluation
    statement: StatementRecord


class AudioInfo(BaseModel):
    filename: str
    format: str
    sample_rate_hz: Optional[int] = None
    n_channels: Optional[int] = None
    duration_s: float
    quality_flags: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    low_confidence: bool
    reasons: list[str] = Field(default_factory=list)


class UploadedAudioOutput(BaseModel):
    """Track A output for one user-uploaded clip. Id namespace: audio-*"""

    case_id: str
    asr_model: str
    audio: AudioInfo
    transcript: str
    segments: list[TranscriptSegment]
    low_confidence_spans: list[str] = Field(default_factory=list)
    asr_evaluation: Optional[ASREvaluation] = None  # only when a reference transcript is supplied
    statement: StatementRecord


class RAGPassage(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    section: str = ""  # heading of the source-page section the passage paraphrases; used for retrieval only
    url: str
    population: str
    retrieval_date: str
    text: str


class RetrievedPassage(BaseModel):
    chunk_id: str
    similarity: float


class RoutingResult(BaseModel):
    category: Literal["support", "review", "urgent"]
    rationale: str
    is_artificial_label: bool = True
    source: Literal["organizer_policy", "software_routing_demo"] = "software_routing_demo"


class ChatResponse(BaseModel):
    conversation_or_case_id: str
    question: str
    answer: str
    abstained: bool
    cited_passage_ids: list[str] = Field(default_factory=list)
    routing: RoutingResult


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------


class QualityFlags(BaseModel):
    total_readings: int
    missing_count: int
    duplicate_timestamp_count: int
    gap_count: int
    max_gap_seconds: float
    non_monotonic_count: int
    coverage_ratio: float
    stationary_phone_suspected: bool


class WindowFeature(BaseModel):
    window_index: int
    start_time_s: float
    end_time_s: float
    n_samples: int
    mean_magnitude: float
    std_magnitude: float
    movement_intensity: float
    state: Literal["active", "stationary", "uncertain"]


class RecordingSummary(BaseModel):
    """Whole-recording roll-up of the per-window states."""

    duration_s: float
    n_windows: int
    active_windows: int
    stationary_windows: int
    uncertain_windows: int
    active_fraction: float
    stationary_fraction: float
    uncertain_fraction: float
    overall_state: Literal["active", "stationary", "uncertain"]
    overall_state_rule: str


class ActivityReport(BaseModel):
    """Track B output for one recording. Id namespace: wisdm-*/ucihar-*"""

    clip_id: str
    source_dataset: Literal["wisdm", "ucihar"]
    device: str
    sensor: str
    units: str
    sampling_hz_declared: float
    sampling_hz_estimated: float
    windows: list[WindowFeature]
    summary: RecordingSummary
    quality: QualityFlags
    ground_truth_activity_label: Optional[str] = None
    supported_activity_states: list[str]
    recording_notes: list[str]  # clip-specific, built from this clip's own data
    limitations: list[str]  # fixed, apply to every clip
