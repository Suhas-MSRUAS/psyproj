"""Recording-level activity report assembly (spec section 4, "Required
separate output"). Builds an independent JSON/CSV report; nothing here reads
from or writes to Track A state.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from common.schemas import ActivityReport, QualityFlags, RecordingSummary, WindowFeature
from track_b.activity import GROUND_TRUTH_STATE_BY_LABEL, classify_window
from track_b.features import make_windows, window_stats
from track_b.ingest_accel import clip_id_for, estimate_sampling_hz, validate_clip

LIMITATIONS = [
    "This is a short, controlled recording and must not be extrapolated to daily "
    "activity, longitudinal decline, or depression severity.",
    "A low-variance window may mean the phone was set down on a table, not that "
    "its owner was inactive.",
    "The accelerometer signal includes the gravity component; orientation changes "
    "can shift readings even without much movement.",
    "Windows with insufficient sample coverage are marked 'uncertain' rather than guessed.",
]

OVERALL_STATE_RULE = (
    "Fixed rule, not learned: 'uncertain' if more than half the windows are uncertain; "
    "otherwise whichever of active/stationary has more windows; a tie is 'uncertain'."
)


def summarize_windows(windows: list[WindowFeature]) -> RecordingSummary:
    n = len(windows)
    counts = {s: sum(w.state == s for w in windows) for s in ("active", "stationary", "uncertain")}
    if n == 0 or counts["uncertain"] > n / 2 or counts["active"] == counts["stationary"]:
        overall = "uncertain"
    else:
        overall = "active" if counts["active"] > counts["stationary"] else "stationary"
    frac = (lambda k: counts[k] / n) if n else (lambda k: 0.0)
    return RecordingSummary(
        duration_s=(windows[-1].end_time_s - windows[0].start_time_s) if n else 0.0,
        n_windows=n,
        active_windows=counts["active"],
        stationary_windows=counts["stationary"],
        uncertain_windows=counts["uncertain"],
        active_fraction=frac("active"),
        stationary_fraction=frac("stationary"),
        uncertain_fraction=frac("uncertain"),
        overall_state=overall,
        overall_state_rule=OVERALL_STATE_RULE,
    )


def recording_notes(
    summary: RecordingSummary, quality: QualityFlags, declared_hz: float, estimated_hz: float
) -> list[str]:
    """Plain-language notes about this particular clip, derived from its own data."""
    notes = []
    if summary.uncertain_windows:
        notes.append(
            f"{summary.uncertain_windows} of {summary.n_windows} windows "
            f"({summary.uncertain_fraction:.0%}) are uncertain because too few readings were recorded."
        )
    if quality.stationary_phone_suspected:
        notes.append("The phone may have been lying still for this whole recording.")
    if quality.gap_count:
        notes.append(f"{quality.gap_count} gap(s) in the data; the longest is {quality.max_gap_seconds:.1f} s.")
    if quality.coverage_ratio < 0.9:
        notes.append(f"Only {quality.coverage_ratio:.0%} of the expected readings are present.")
    if quality.missing_count:
        notes.append(f"{quality.missing_count} reading(s) have missing values.")
    if quality.duplicate_timestamp_count:
        notes.append(f"{quality.duplicate_timestamp_count} duplicate timestamp(s).")
    if quality.non_monotonic_count:
        notes.append(f"{quality.non_monotonic_count} out-of-order timestamp(s).")
    if declared_hz and abs(estimated_hz - declared_hz) / declared_hz > 0.1:
        notes.append(f"Sampling rate estimated at {estimated_hz:.1f} Hz against {declared_hz:.0f} Hz declared.")
    return notes or ["No data problems found in this recording."]


def build_activity_report(
    clip: pd.DataFrame,
    subject_id: int,
    activity_label: str,
    segment_index: int,
    expected_hz: float,
    window_seconds: float,
    window_overlap: float,
    stationary_variance_threshold: float,
    max_gap_s: float,
) -> ActivityReport:
    clip_id = clip_id_for(subject_id, activity_label, segment_index)
    quality: QualityFlags = validate_clip(clip, expected_hz=expected_hz, max_gap_s=max_gap_s)
    sampling_hz_estimated = estimate_sampling_hz(clip)

    raw_windows = make_windows(clip, window_seconds=window_seconds, overlap=window_overlap)
    window_features: list[WindowFeature] = []
    for rw in raw_windows:
        stats = window_stats(rw)
        state = classify_window(rw, stats, expected_hz, stationary_variance_threshold)
        window_features.append(
            WindowFeature(
                window_index=rw.window_index,
                start_time_s=rw.start_time_s,
                end_time_s=rw.end_time_s,
                n_samples=rw.n_samples,
                mean_magnitude=stats.mean_magnitude,
                std_magnitude=stats.std_magnitude,
                movement_intensity=stats.movement_intensity,
                state=state,
            )
        )

    summary = summarize_windows(window_features)

    return ActivityReport(
        clip_id=clip_id,
        source_dataset="wisdm",
        device="phone",
        sensor="accelerometer",
        units="m/s^2",
        sampling_hz_declared=expected_hz,
        sampling_hz_estimated=sampling_hz_estimated,
        windows=window_features,
        summary=summary,
        quality=quality,
        ground_truth_activity_label=GROUND_TRUTH_STATE_BY_LABEL.get(activity_label),
        supported_activity_states=["active", "stationary", "uncertain"],
        recording_notes=recording_notes(summary, quality, expected_hz, sampling_hz_estimated),
        limitations=list(LIMITATIONS),
    )


def report_to_json(report: ActivityReport) -> str:
    return report.model_dump_json(indent=2)


def windows_to_csv(report: ActivityReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["clip_id", "window_index", "start_time_s", "end_time_s", "n_samples",
                      "mean_magnitude", "std_magnitude", "movement_intensity", "state"])
    for w in report.windows:
        writer.writerow([report.clip_id, w.window_index, w.start_time_s, w.end_time_s, w.n_samples,
                          w.mean_magnitude, w.std_magnitude, w.movement_intensity, w.state])
    return buf.getvalue()


def write_report(report: ActivityReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{report.clip_id}.json"
    csv_path = out_dir / f"{report.clip_id}_windows.csv"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    csv_path.write_text(windows_to_csv(report), encoding="utf-8")
    return json_path, csv_path
