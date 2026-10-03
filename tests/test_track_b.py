"""Required test scenarios, spec section 8: B: clean movement, B: sensor failure.

The clean-movement scenario uses the real downloaded WISDM data (skipped if
not present) since the point is to compare against a genuine ground-truth
activity label. The sensor-failure scenario uses the synthetic fixture,
which deterministically encodes a gap, a duplicate timestamp, and a
stationary phone.
"""
from pathlib import Path

import pytest

from common.config import get_config, resolve_path
from track_b.activity import GROUND_TRUTH_STATE_BY_LABEL
from track_b.ingest_accel import parse_wisdm_file, split_into_clips
from track_b.report import build_activity_report

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

_WISDM_DIR = resolve_path(get_config()["paths"]["wisdm_raw_phone_accel_dir"])
_WISDM_SUBJECT_1600 = _WISDM_DIR / "data_1600_accel_phone.txt"


@pytest.mark.skipif(not _WISDM_SUBJECT_1600.exists(), reason="real WISDM data not downloaded")
def test_clean_movement_matches_ground_truth_activity():
    df, parse_report = parse_wisdm_file(_WISDM_SUBJECT_1600)
    assert parse_report.parsed_lines > 0
    clips = split_into_clips(df)

    walking_clip = next(c for c in clips if c.activity_label.iloc[0] == "A")  # walking
    sitting_clip = next(c for c in clips if c.activity_label.iloc[0] == "D")  # sitting

    walking_report = build_activity_report(
        walking_clip, subject_id=1600, activity_label="A", segment_index=0,
        expected_hz=20, window_seconds=10, window_overlap=0.0,
        stationary_variance_threshold=0.6, max_gap_s=2.0,
    )
    sitting_report = build_activity_report(
        sitting_clip, subject_id=1600, activity_label="D", segment_index=3,
        expected_hz=20, window_seconds=10, window_overlap=0.0,
        stationary_variance_threshold=0.6, max_gap_s=2.0,
    )

    assert walking_report.ground_truth_activity_label == GROUND_TRUTH_STATE_BY_LABEL["A"] == "active"
    assert sitting_report.ground_truth_activity_label == GROUND_TRUTH_STATE_BY_LABEL["D"] == "stationary"

    walking_states = [w.state for w in walking_report.windows]
    sitting_states = [w.state for w in sitting_report.windows]
    assert walking_states.count("active") >= len(walking_states) * 0.8
    assert sitting_states.count("stationary") >= len(sitting_states) * 0.8


def test_sensor_failure_flags_are_raised_without_asserting_daily_activity():
    df, parse_report = parse_wisdm_file(FIXTURES / "accel" / "sensor_failure_clip.txt")
    assert parse_report.malformed_lines == 0
    clip = split_into_clips(df)[0]

    report = build_activity_report(
        clip, subject_id=int(clip.subject_id.iloc[0]), activity_label=clip.activity_label.iloc[0], segment_index=0,
        expected_hz=20, window_seconds=2, window_overlap=0.0,
        stationary_variance_threshold=0.6, max_gap_s=2.0,
    )

    assert report.quality.gap_count >= 1
    assert report.quality.max_gap_seconds >= 2.0
    assert report.quality.duplicate_timestamp_count >= 1
    assert report.quality.stationary_phone_suspected is True
    # No activity window state asserts anything beyond active/stationary/uncertain,
    # and the report always carries explicit non-extrapolation limitations.
    assert all(w.state in {"active", "stationary", "uncertain"} for w in report.windows)
    assert any("daily activity" in limitation for limitation in report.limitations)
