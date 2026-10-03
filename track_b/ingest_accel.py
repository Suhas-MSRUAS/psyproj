"""Raw WISDM phone-accelerometer ingestion and validation (spec section 4).

Only raw accelerometer channels (subject-id, activity-label, timestamp, x, y, z)
are read. This module never imports anything from track_a and never sees an
AnnoMI id or audio case id — the activity report is independent by
construction, not just by convention.

WISDM raw file format (one line per reading, no header, trailing `;`):
    subject-id, activity-label, timestamp(ns, unix), x, y, z;
Nominal sampling rate is ~20 Hz; each subject file is 18 back-to-back
~3-minute activity recordings (one contiguous run per activity label).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.schemas import QualityFlags

_LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([A-Za-z])\s*,\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*;?\s*$"
)


@dataclass
class ParseReport:
    total_lines: int
    parsed_lines: int
    malformed_lines: int


def parse_wisdm_file(path: str | Path) -> tuple[pd.DataFrame, ParseReport]:
    rows: list[tuple] = []
    total = 0
    malformed = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            m = _LINE_RE.match(line)
            if not m:
                malformed += 1
                continue
            subject_id, activity, ts, x, y, z = m.groups()
            rows.append((int(subject_id), activity, int(ts), float(x), float(y), float(z)))

    df = pd.DataFrame(rows, columns=["subject_id", "activity_label", "timestamp_ns", "x", "y", "z"])
    return df, ParseReport(total_lines=total, parsed_lines=len(df), malformed_lines=malformed)


def split_into_clips(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Each WISDM subject file is several activities recorded back-to-back;
    split on contiguous runs of the same activity label into separate clips."""
    if df.empty:
        return []
    seg_id = (df["activity_label"] != df["activity_label"].shift()).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(seg_id)]


def clip_id_for(subject_id: int, activity_label: str, segment_index: int) -> str:
    return f"wisdm-{subject_id}-{activity_label}-{segment_index}"


def validate_clip(clip: pd.DataFrame, expected_hz: float, max_gap_s: float) -> QualityFlags:
    total = len(clip)
    timestamps_s = clip["timestamp_ns"].to_numpy(dtype=np.float64) / 1e9

    missing = int(clip[["x", "y", "z"]].isna().any(axis=1).sum())
    duplicate_ts = int(clip["timestamp_ns"].duplicated().sum())

    diffs = np.diff(timestamps_s) if total > 1 else np.array([])
    non_monotonic = int((diffs < 0).sum())
    gap_count = int((diffs > max_gap_s).sum())
    max_gap = float(diffs.max()) if diffs.size else 0.0

    duration = float(timestamps_s[-1] - timestamps_s[0]) if total > 1 else 0.0
    expected_samples = duration * expected_hz
    coverage_ratio = float(min(total / expected_samples, 1.0)) if expected_samples > 0 else 0.0

    magnitude = np.sqrt(clip["x"] ** 2 + clip["y"] ** 2 + clip["z"] ** 2)
    # Whole-clip screen: a near-zero-variance magnitude over the full clip
    # suggests a stationary/unattended phone. The per-window state in
    # track_b/features.py is the one that actually drives the report.
    stationary_suspected = bool(magnitude.std() < 0.05) if total > 1 else False

    return QualityFlags(
        total_readings=total,
        missing_count=missing,
        duplicate_timestamp_count=duplicate_ts,
        gap_count=gap_count,
        max_gap_seconds=max_gap,
        non_monotonic_count=non_monotonic,
        coverage_ratio=coverage_ratio,
        stationary_phone_suspected=stationary_suspected,
    )


def estimate_sampling_hz(clip: pd.DataFrame) -> float:
    if len(clip) < 2:
        return 0.0
    timestamps_s = clip["timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
    diffs = np.diff(timestamps_s)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(1.0 / np.median(diffs))


def list_subject_files(wisdm_accel_dir: str | Path) -> list[Path]:
    return sorted(Path(wisdm_accel_dir).glob("data_*_accel_phone.txt"))
