"""Speech-to-text on independent, consented fictional audio (spec section A2).

Never touches AnnoMI. Each case is a `<case_id>.wav` + `<case_id>.json` pair
(see data/audio/README.md). Output is traceable to the case id only — never
attributed to an AnnoMI participant.
"""
from __future__ import annotations

import json
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import jiwer

from common.backends import ASRBackend
from common.schemas import ASREvaluation, AudioCaseOutput, AudioInfo, TranscriptSegment, UploadedAudioOutput
from track_a.extraction import extract_from_transcript

_NEGATION_WORDS = {"not", "no", "never", "n't", "without", "nothing", "none", "nobody"}

WER_LOW_CONFIDENCE_THRESHOLD = 0.3


@dataclass
class AudioValidation:
    case_id: str
    sample_rate_hz: int
    n_channels: int
    duration_s: float
    flags: list[str]


def validate_audio(wav_path: Path) -> AudioValidation:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        duration = n_frames / float(sample_rate) if sample_rate else 0.0

    flags = []
    if sample_rate < 8000 or sample_rate > 48000:
        flags.append(f"unusual_sample_rate:{sample_rate}")
    if duration < 0.3:
        flags.append(f"very_short_duration:{duration:.2f}s")
    if duration > 120:
        flags.append(f"very_long_duration:{duration:.2f}s")
    if n_channels > 2:
        flags.append(f"unusual_channel_count:{n_channels}")

    return AudioValidation(
        case_id=wav_path.stem,
        sample_rate_hz=sample_rate,
        n_channels=n_channels,
        duration_s=duration,
        flags=flags,
    )


_CONTRACTIONS = [
    (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon't\b", "will not"),
    (r"\bshan't\b", "shall not"), (r"\bain't\b", "am not"), (r"n't\b", " not"),
]


def normalize_for_comparison(text: str) -> str:
    """Lowercase, expand negative contractions, drop punctuation. Applied to
    both reference and prediction so "don't" vs "do not" is neither a WER
    error nor a negation change, while the stored transcripts stay verbatim."""
    t = text.lower().replace("’", "'")
    for pattern, repl in _CONTRACTIONS:
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return " ".join(t.split())


def _critical_word_flags(reference: str, hypothesis: str) -> list[str]:
    # Whole-token counts, so "know" is not "no" and a second "not" lost in a
    # sentence that still has one is still caught.
    ref_tokens = normalize_for_comparison(reference).split()
    hyp_tokens = normalize_for_comparison(hypothesis).split()
    flags = []
    for w in sorted(_NEGATION_WORDS - {"n't"}):
        diff = ref_tokens.count(w) - hyp_tokens.count(w)
        if diff > 0:
            flags.append(f"negation_dropped:{w}")
        elif diff < 0:
            flags.append(f"negation_added:{w}")
    return flags


def evaluate_case(wav_path: Path, asr: ASRBackend) -> AudioCaseOutput:
    case_id = wav_path.stem
    meta = json.loads(wav_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not meta.get("speaker_consent", False):
        raise ValueError(f"Case {case_id} is missing recorded speaker consent; refusing to process.")

    reference = meta["reference_transcript"]
    hypothesis = asr.transcribe(wav_path)
    asr_eval = _compare_to_reference(case_id, reference, hypothesis)

    statement = extract_from_transcript(hypothesis)

    return AudioCaseOutput(
        case_id=f"audio-{case_id}",
        transcript=hypothesis,
        asr_evaluation=asr_eval,
        statement=statement,
    )


def _compare_to_reference(case_id: str, reference: str, hypothesis: str) -> ASREvaluation:
    wer = jiwer.wer(normalize_for_comparison(reference), normalize_for_comparison(hypothesis))
    critical_flags = _critical_word_flags(reference, hypothesis)
    return ASREvaluation(
        case_id=case_id,
        reference_transcript=reference,
        hypothesis_transcript=hypothesis,
        word_error_rate=wer,
        low_confidence_spans=[hypothesis] if wer >= WER_LOW_CONFIDENCE_THRESHOLD else [],
        critical_word_flags=critical_flags,
        negation_preserved=not any(f.startswith("negation_") for f in critical_flags),
    )


# ---------------------------------------------------------------------------
# User-uploaded clips (any format ffmpeg can read: wav, mp3, m4a, ogg, flac...)
# ---------------------------------------------------------------------------

# Whisper's own fallback thresholds (openai-whisper transcribe() defaults):
# below/above these a segment is treated as unreliable rather than repaired.
_LOGPROB_THRESHOLD = -1.0
_NO_SPEECH_THRESHOLD = 0.6
_COMPRESSION_RATIO_THRESHOLD = 2.4


def probe_audio(path: Path, filename: str) -> AudioInfo:
    """Inspect sample rate / channels / duration with ffprobe (ships with ffmpeg,
    which Whisper already requires)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"Could not read '{filename}' as audio: {proc.stderr.strip()[:200]}")
    info = json.loads(proc.stdout)
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError(f"'{filename}' contains no audio stream.")

    sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
    channels = stream.get("channels")
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)

    flags = []
    if sample_rate and (sample_rate < 8000 or sample_rate > 48000):
        flags.append(f"unusual_sample_rate:{sample_rate}")
    if sample_rate and sample_rate < 16000:
        flags.append("below_16kHz:Whisper upsamples to 16 kHz, accuracy may drop")
    if duration < 0.3:
        flags.append(f"very_short_duration:{duration:.2f}s")
    if duration > 600:
        flags.append(f"very_long_duration:{duration:.0f}s")
    if channels and channels > 2:
        flags.append(f"unusual_channel_count:{channels}")

    return AudioInfo(
        filename=filename,
        format=info.get("format", {}).get("format_name", "unknown"),
        sample_rate_hz=sample_rate,
        n_channels=channels,
        duration_s=duration,
        quality_flags=flags,
    )


def evaluate_upload(audio_path: Path, filename: str, asr, reference: str | None = None) -> UploadedAudioOutput:
    """Transcribe one uploaded clip with Whisper. Low-confidence segments are
    flagged, never rewritten. WER + negation audit run only if the user gives
    a reference transcript; the reference is kept separate from the prediction."""
    audio = probe_audio(audio_path, filename)
    result = asr.transcribe_detailed(audio_path)

    segments = []
    for s in result["segments"]:
        reasons = []
        if s["avg_logprob"] < _LOGPROB_THRESHOLD:
            reasons.append(f"low average log-probability ({s['avg_logprob']:.2f})")
        if s["no_speech_prob"] > _NO_SPEECH_THRESHOLD:
            reasons.append(f"may be silence/noise (no-speech prob {s['no_speech_prob']:.2f})")
        if s["compression_ratio"] > _COMPRESSION_RATIO_THRESHOLD:
            reasons.append(f"repetitive output (compression ratio {s['compression_ratio']:.2f})")
        segments.append(TranscriptSegment(
            start_s=s["start_s"], end_s=s["end_s"], text=s["text"],
            avg_logprob=s["avg_logprob"], no_speech_prob=s["no_speech_prob"],
            low_confidence=bool(reasons), reasons=reasons,
        ))

    transcript = result["text"]
    if not transcript:
        audio.quality_flags.append("no_speech_detected")

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem)[:60] or "clip"
    evaluation = None
    if reference and reference.strip():
        evaluation = _compare_to_reference(stem, reference.strip(), transcript)

    return UploadedAudioOutput(
        case_id=f"audio-upload-{stem}",
        asr_model=getattr(asr, "checkpoint", "whisper"),
        audio=audio,
        transcript=transcript,
        segments=segments,
        low_confidence_spans=[s.text for s in segments if s.low_confidence],
        asr_evaluation=evaluation,
        statement=extract_from_transcript(transcript),
    )


def list_cases(audio_dir: Path) -> list[Path]:
    return sorted(audio_dir.glob("*.wav"))
