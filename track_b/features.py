"""Signal magnitude and windowed feature extraction (spec section 4).

a(t) = sqrt(x(t)^2 + y(t)^2 + z(t)^2). WISDM accelerometer values include the
gravity component (the sensor is not gravity-compensated), so a stationary,
flat phone reads a magnitude near 9.8 m/s^2, not 0 — this is expected and is
why "low variance" rather than "low magnitude" is what the stationary
heuristic below actually keys on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_magnitude(clip: pd.DataFrame) -> np.ndarray:
    return np.sqrt(clip["x"] ** 2 + clip["y"] ** 2 + clip["z"] ** 2).to_numpy(dtype=np.float64)


@dataclass
class RawWindow:
    window_index: int
    start_time_s: float
    end_time_s: float
    n_samples: int
    magnitudes: np.ndarray


def make_windows(clip: pd.DataFrame, window_seconds: float, overlap: float = 0.0) -> list[RawWindow]:
    """Time-based (not sample-count-based) windowing, so gaps or rate jitter
    don't silently shift window boundaries."""
    if clip.empty:
        return []

    timestamps_s = clip["timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
    magnitude = compute_magnitude(clip)
    t0 = timestamps_s[0]
    t_end = timestamps_s[-1]
    step = window_seconds * (1.0 - overlap) if overlap < 1.0 else window_seconds
    if step <= 0:
        step = window_seconds

    windows: list[RawWindow] = []
    start = t0
    idx = 0
    while start < t_end:
        end = start + window_seconds
        mask = (timestamps_s >= start) & (timestamps_s < end)
        n = int(mask.sum())
        if n > 0:
            windows.append(
                RawWindow(
                    window_index=idx,
                    start_time_s=float(start - t0),
                    end_time_s=float(end - t0),
                    n_samples=n,
                    magnitudes=magnitude[mask],
                )
            )
            idx += 1
        start += step
    return windows


@dataclass
class WindowStats:
    mean_magnitude: float
    std_magnitude: float
    movement_intensity: float  # mean absolute sample-to-sample change, a jerk-like proxy


def window_stats(raw_window: RawWindow) -> WindowStats:
    mags = raw_window.magnitudes
    mean_mag = float(np.mean(mags)) if mags.size else 0.0
    std_mag = float(np.std(mags)) if mags.size else 0.0
    if mags.size > 1:
        movement_intensity = float(np.mean(np.abs(np.diff(mags))))
    else:
        movement_intensity = 0.0
    return WindowStats(mean_magnitude=mean_mag, std_magnitude=std_mag, movement_intensity=movement_intensity)
