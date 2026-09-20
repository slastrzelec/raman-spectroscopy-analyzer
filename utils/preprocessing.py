"""
Preprocessing steps for Raman spectra: baseline correction, normalization,
smoothing.

Every function takes a DataFrame with columns ("Wavenumber", "Intensity")
and returns a new DataFrame with the same columns plus a params dict
describing exactly what was applied (used for the UI and for exported
reports). Functions never touch st.session_state -- that wiring lives in
app.py -- which is what makes these directly unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


class PreprocessingError(ValueError):
    """Raised when a preprocessing step is called with invalid parameters
    for the given spectrum (e.g. a smoothing window larger than the data)."""


def _intensity(df: pd.DataFrame) -> np.ndarray:
    return df[INTENSITY_COL].to_numpy(dtype=float)


def _with_intensity(df: pd.DataFrame, new_intensity: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out[INTENSITY_COL] = new_intensity
    return out


# --- Baseline correction ----------------------------------------------------


def baseline_linear_endpoints(df: pd.DataFrame, n_points: int = 5):
    """Fit a straight line through the average of the first/last `n_points`
    and subtract it. Simple and robust when the spectrum's edges are
    genuinely baseline (no peaks near the boundaries).
    """
    y = _intensity(df)
    n = len(y)
    if n_points < 1:
        raise PreprocessingError("n_points must be >= 1.")
    if n_points * 2 > n:
        raise PreprocessingError(
            f"n_points ({n_points}) is too large for a spectrum with {n} points."
        )

    x = np.arange(n)
    left_x, left_y = x[:n_points], y[:n_points]
    right_x, right_y = x[-n_points:], y[-n_points:]

    x_fit = np.concatenate([left_x, right_x])
    y_fit = np.concatenate([left_y, right_y])
    slope, intercept = np.polyfit(x_fit, y_fit, 1)

    baseline = slope * x + intercept
    corrected = y - baseline

    params = {"method": "Linear (Endpoints)", "n_points": n_points}
    return _with_intensity(df, corrected), baseline, params


def baseline_als(df: pd.DataFrame, lam: float = 1e5, p: float = 0.01, n_iter: int = 10):
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    `lam` controls smoothness of the estimated baseline, `p` controls
    asymmetry (how much more the fit is pulled toward points below the
    current estimate vs. above it -- appropriate since real peaks point up).
    """
    y = _intensity(df)
    n = len(y)
    if n < 3:
        raise PreprocessingError("The spectrum must have at least 3 points for ALS.")
    if not (0 < p < 1):
        raise PreprocessingError("p must be in the range (0, 1).")
    if lam <= 0:
        raise PreprocessingError("lambda must be positive.")

    diag_matrix = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2), dtype=float)
    smoothness_penalty = lam * diag_matrix.dot(diag_matrix.transpose())
    weights = np.ones(n)

    baseline = np.zeros(n)
    for _ in range(n_iter):
        weight_matrix = sparse.diags(weights, 0)
        system_matrix = weight_matrix + smoothness_penalty
        baseline = spsolve(system_matrix.tocsc(), weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)

    corrected = y - baseline
    params = {"method": "ALS (Asymmetric Least Squares)", "lam": lam, "p": p, "n_iter": n_iter}
    return _with_intensity(df, corrected), baseline, params


# --- Normalization -----------------------------------------------------------


def normalize(df: pd.DataFrame, method: str = "Min-Max"):
    y = _intensity(df)

    if method == "Min-Max":
        y_min, y_max = y.min(), y.max()
        if y_max == y_min:
            raise PreprocessingError("Cannot normalize a spectrum with constant intensity (Min-Max).")
        normalized = (y - y_min) / (y_max - y_min)
    elif method == "Area (integral)":
        wavenumber = df[WAVENUMBER_COL].to_numpy(dtype=float)
        area = np.trapezoid(y, wavenumber)  # np.trapz was removed in numpy 2.0
        if area == 0:
            raise PreprocessingError("Cannot normalize a spectrum with zero area under the curve.")
        normalized = y / abs(area)
    elif method == "Max intensity":
        y_max = y.max()
        if y_max == 0:
            raise PreprocessingError("Cannot normalize a spectrum with zero maximum intensity.")
        normalized = y / y_max
    else:
        raise PreprocessingError(f"Unknown normalization method: '{method}'.")

    params = {"method": method}
    return _with_intensity(df, normalized), params


# --- Smoothing -----------------------------------------------------------------


def smooth(df: pd.DataFrame, window_length: int = 11, polyorder: int = 3):
    """Savitzky-Golay smoothing."""
    y = _intensity(df)
    n = len(y)

    if window_length % 2 == 0:
        raise PreprocessingError("window_length must be an odd number.")
    if window_length > n:
        raise PreprocessingError(
            f"window_length ({window_length}) is larger than the number of spectrum points ({n})."
        )
    if polyorder >= window_length:
        raise PreprocessingError("polyorder must be smaller than window_length.")

    smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    params = {"window_length": window_length, "polyorder": polyorder}
    return _with_intensity(df, smoothed), params
