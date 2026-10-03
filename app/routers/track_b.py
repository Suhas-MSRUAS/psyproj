"""Track B endpoints: independent WISDM accelerometer validation + activity report.

This router never imports from track_a and never reads AnnoMI or fictional
audio data — see tests/test_separation.py for the audit that checks this
holds. Nothing here sends data to Track A's chatbot, retriever, or routing.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from common.config import get_config, resolve_path
from common.schemas import ActivityReport
from track_b.ingest_accel import list_subject_files, parse_wisdm_file, split_into_clips, clip_id_for
from track_b.report import build_activity_report, windows_to_csv

router = APIRouter(prefix="/track-b", tags=["track-b"])


@lru_cache(maxsize=1)
def _accel_dir():
    return resolve_path(get_config()["paths"]["wisdm_raw_phone_accel_dir"])


@lru_cache(maxsize=64)
def _clips_for_subject(subject_id: int):
    path = _accel_dir() / f"data_{subject_id}_accel_phone.txt"
    if not path.exists():
        return None
    df, _parse_report = parse_wisdm_file(path)
    return split_into_clips(df)


def _subject_ids() -> list[int]:
    ids = []
    for p in list_subject_files(_accel_dir()):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


@router.get("/subjects")
def list_subjects():
    return _subject_ids()


@router.get("/clips/{subject_id}")
def list_clips(subject_id: int):
    clips = _clips_for_subject(subject_id)
    if clips is None:
        raise HTTPException(status_code=404, detail=f"No WISDM file for subject_id={subject_id}")
    return [
        {"clip_id": clip_id_for(subject_id, c.activity_label.iloc[0], i), "activity_label": c.activity_label.iloc[0], "n_readings": len(c)}
        for i, c in enumerate(clips)
    ]


def _get_clip(subject_id: int, activity_label: str, segment_index: int):
    clips = _clips_for_subject(subject_id)
    if clips is None:
        raise HTTPException(status_code=404, detail=f"No WISDM file for subject_id={subject_id}")
    if segment_index < 0 or segment_index >= len(clips):
        raise HTTPException(status_code=404, detail=f"No segment_index={segment_index} for subject_id={subject_id}")
    clip = clips[segment_index]
    if clip.activity_label.iloc[0] != activity_label:
        raise HTTPException(
            status_code=404,
            detail=f"segment_index={segment_index} is activity '{clip.activity_label.iloc[0]}', not '{activity_label}'",
        )
    return clip


@router.get("/report/{subject_id}/{activity_label}/{segment_index}", response_model=ActivityReport)
def get_report(subject_id: int, activity_label: str, segment_index: int):
    clip = _get_clip(subject_id, activity_label, segment_index)
    cfg = get_config()["track_b"]
    return build_activity_report(
        clip,
        subject_id=subject_id,
        activity_label=activity_label,
        segment_index=segment_index,
        expected_hz=cfg["sampling_hz_expected"],
        window_seconds=cfg["window_seconds"],
        window_overlap=cfg["window_overlap"],
        stationary_variance_threshold=cfg["stationary_variance_threshold"],
        max_gap_s=cfg["max_gap_seconds"],
    )


@router.get("/report/{subject_id}/{activity_label}/{segment_index}/csv", response_class=PlainTextResponse)
def get_report_csv(subject_id: int, activity_label: str, segment_index: int):
    report = get_report(subject_id, activity_label, segment_index)
    return windows_to_csv(report)
