"""Transparent, deterministic active/stationary heuristic (spec section 4).

No training: a fixed variance threshold on the magnitude signal within each
time window. Windows with too few samples (gaps, non-wear, truncated
recordings) are marked "uncertain" rather than guessed, per the sensor-failure
requirement in section 8 of the spec.

The spec's other permitted option — "an existing compatible pretrained
activity model without updating its weights" — is intentionally not wired up
here: no specific checkpoint was named for this task, and inventing one would
misrepresent what was actually evaluated. This module is the fully-specified
heuristic path; swapping in a real pretrained classifier later only requires
implementing the same `classify_window` signature.
"""
from __future__ import annotations

from typing import Literal

from track_b.features import RawWindow, WindowStats, window_stats

State = Literal["active", "stationary", "uncertain"]

MIN_WINDOW_COVERAGE = 0.5  # fraction of expected samples required to trust a window's label


def classify_window(
    raw_window: RawWindow,
    stats: WindowStats,
    expected_hz: float,
    stationary_variance_threshold: float,
) -> State:
    expected_samples = expected_hz * (raw_window.end_time_s - raw_window.start_time_s)
    if expected_samples <= 0 or (raw_window.n_samples / expected_samples) < MIN_WINDOW_COVERAGE:
        return "uncertain"
    return "stationary" if stats.std_magnitude < stationary_variance_threshold else "active"


# For evaluation/agreement reporting only (spec: "Evaluate output against
# activity labels where appropriate") — never fed into the classifier above,
# and never used to claim anything about daily activity or depression.
GROUND_TRUTH_STATE_BY_LABEL: dict[str, State] = {
    "A": "active",  # walking
    "B": "active",  # jogging
    "C": "active",  # stairs
    "D": "stationary",  # sitting
    "E": "stationary",  # standing
    "F": "stationary",  # typing
    "G": "stationary",  # brushing teeth
    "H": "stationary",  # eating soup
    "I": "stationary",  # eating chips
    "J": "stationary",  # eating pasta
    "K": "stationary",  # drinking
    "L": "stationary",  # eating sandwich
    "M": "active",  # kicking
    "O": "active",  # catch
    "P": "active",  # dribbling
    "Q": "stationary",  # writing
    "R": "active",  # clapping
    "S": "stationary",  # folding clothes
}
