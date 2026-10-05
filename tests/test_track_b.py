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
from common.schemas import QualityFlags, WindowFeature
from track_b.activity import GROUND_TRUTH_STATE_BY_LABEL
from track_b.ingest_accel import parse_wisdm_file, split_into_clips
from track_b.report import build_activity_report, recording_notes, summarize_windows

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


def _window(i: int, state: str) -> WindowFeature:
    return WindowFeature(window_index=i, start_time_s=i * 10.0, end_time_s=(i + 1) * 10.0, n_samples=200,
                         mean_magnitude=9.8, std_magnitude=1.0, movement_intensity=0.1, state=state)


def test_recording_summary_rolls_up_windows_with_fixed_rule():
    summary = summarize_windows([_window(i, s) for i, s in enumerate(["active"] * 6 + ["stationary"] * 3 + ["uncertain"])])
    assert (summary.n_windows, summary.duration_s) == (10, 100.0)
    assert (summary.active_windows, summary.stationary_windows, summary.uncertain_windows) == (6, 3, 1)
    assert summary.active_fraction + summary.stationary_fraction + summary.uncertain_fraction == pytest.approx(1.0)
    assert summary.overall_state == "active"

    # Mostly uncertain, or a tie, is never guessed.
    assert summarize_windows([_window(i, s) for i, s in enumerate(["uncertain"] * 3 + ["active"] * 2)]).overall_state == "uncertain"
    assert summarize_windows([_window(0, "active"), _window(1, "stationary")]).overall_state == "uncertain"
    assert summarize_windows([]).overall_state == "uncertain"

    # The report built from a real clip carries the summary, consistent with its windows.
    df, _ = parse_wisdm_file(FIXTURES / "accel" / "sensor_failure_clip.txt")
    clip = split_into_clips(df)[0]
    report = build_activity_report(
        clip, subject_id=int(clip.subject_id.iloc[0]), activity_label=clip.activity_label.iloc[0], segment_index=0,
        expected_hz=20, window_seconds=2, window_overlap=0.0,
        stationary_variance_threshold=0.6, max_gap_s=2.0,
    )
    assert report.summary.n_windows == len(report.windows)
    assert report.summary.active_windows + report.summary.stationary_windows + report.summary.uncertain_windows == len(report.windows)


def test_recording_notes_are_clip_specific():
    df, _ = parse_wisdm_file(FIXTURES / "accel" / "sensor_failure_clip.txt")
    clip = split_into_clips(df)[0]
    report = build_activity_report(
        clip, subject_id=int(clip.subject_id.iloc[0]), activity_label=clip.activity_label.iloc[0], segment_index=0,
        expected_hz=20, window_seconds=2, window_overlap=0.0,
        stationary_variance_threshold=0.6, max_gap_s=2.0,
    )
    notes = " ".join(report.recording_notes)
    assert "lying still" in notes and "gap" in notes and "duplicate" in notes

    clean = QualityFlags(total_readings=3600, missing_count=0, duplicate_timestamp_count=0, gap_count=0,
                         max_gap_seconds=0.05, non_monotonic_count=0, coverage_ratio=1.0,
                         stationary_phone_suspected=False)
    clean_summary = summarize_windows([_window(i, "active") for i in range(18)])
    assert recording_notes(clean_summary, clean, 20, 20.0) == ["No data problems found in this recording."]
