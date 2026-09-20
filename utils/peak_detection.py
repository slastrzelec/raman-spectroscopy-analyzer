"""
Peak detection on a (already preprocessed) Raman spectrum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


class PeakDetectionError(ValueError):
    """Raised when peak detection parameters are invalid for the given spectrum."""


def resolve_prominence(df: pd.DataFrame, prominence_pct: float) -> float:
    """Convert a relative prominence (percent of the spectrum's own
    intensity range) into the absolute value `detect_peaks` needs.

    Peak prominence is naturally measured in the data's own intensity
    units, which vary hugely depending on preprocessing -- a raw spectrum
    might range 0-300, an area-normalized one 0-0.01. A single absolute
    default therefore silently finds zero peaks on some spectra. Expressing
    the UI control as "% of intensity range" keeps one setting meaningful
    regardless of what preprocessing was applied beforehand.
    """
    y = df[INTENSITY_COL].to_numpy(dtype=float)
    intensity_range = float(y.max() - y.min()) if len(y) > 0 else 0.0
    if intensity_range <= 0:
        return 1e-9  # constant/degenerate spectrum -- detect_peaks will just find nothing
    return (prominence_pct / 100.0) * intensity_range


def detect_peaks(
    df: pd.DataFrame, prominence: float = 5.0, min_distance: int = 5
) -> tuple[pd.DataFrame, dict]:
    """Find peaks in the spectrum's intensity signal.

    `prominence` here is an ABSOLUTE value in the spectrum's own intensity
    units -- callers driven by a UI should compute it with
    `resolve_prominence()` first so the same relative setting works
    regardless of preprocessing.

    Returns a DataFrame (one row per peak, sorted by Wavenumber ascending)
    with columns: Wavenumber, Intensity, prominence, width -- plus the
    params dict describing what was used, for display/export.
    """
    y = df[INTENSITY_COL].to_numpy(dtype=float)
    x = df[WAVENUMBER_COL].to_numpy(dtype=float)
    n = len(y)

    if prominence <= 0:
        raise PeakDetectionError("prominence must be positive.")
    if min_distance < 1:
        raise PeakDetectionError("min_distance must be >= 1.")
    if min_distance > n:
        raise PeakDetectionError(
            f"min_distance ({min_distance}) is larger than the number of spectrum points ({n})."
        )

    indices, properties = find_peaks(
        y, prominence=prominence, distance=min_distance, width=0
    )

    peak_df = pd.DataFrame(
        {
            "peak_index": indices,
            WAVENUMBER_COL: x[indices],
            INTENSITY_COL: y[indices],
            "prominence": properties["prominences"],
            "width": properties["widths"],
        }
    ).sort_values(WAVENUMBER_COL).reset_index(drop=True)

    params = {"prominence": prominence, "min_distance": min_distance, "n_peaks": len(peak_df)}
    return peak_df, params
