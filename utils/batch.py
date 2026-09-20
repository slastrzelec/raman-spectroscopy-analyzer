"""
Running the full analysis pipeline (baseline correction -> normalization ->
smoothing -> peak detection -> peak fitting) on one spectrum, given a single
settings dict -- this is what the app's batch-export mode calls once per
uploaded file, reusing the exact settings chosen for the "active" spectrum.

Kept separate from app.py per SPEC.md's architecture rule: app.py only
orchestrates and renders, it does not compute. Each step is best-effort: if
a step's parameters don't suit one particular file in the batch (e.g. a
baseline window wider than that file's spectrum), that step is skipped and
recorded in `errors` rather than dropping the whole file from the batch.
"""

from __future__ import annotations

import pandas as pd

from utils.peak_detection import PeakDetectionError, detect_peaks, resolve_prominence
from utils.peak_fitting import PeakFittingError, fit_peaks
from utils.preprocessing import (
    PreprocessingError,
    baseline_als,
    baseline_linear_endpoints,
    normalize,
    smooth,
)


def run_pipeline(df: pd.DataFrame, settings: dict) -> dict:
    """Apply `settings` to one spectrum.

    settings keys (all optional -- a missing/None key means "skip this step"):
      baseline_method: None | "Linear (Endpoints)" | "ALS (Asymmetric Least Squares)"
      baseline_params: dict of kwargs for the chosen baseline function
      normalization_method: None | one of config.settings.NORMALIZATION_METHODS
      smoothing_params: dict(window_length=, polyorder=) | None
      peak_detection_params: dict(prominence_pct=, min_distance=) | None --
        prominence_pct is relative to THIS file's own intensity range (see
        utils.peak_detection.resolve_prominence), so it stays meaningful
        across files with very different preprocessing/scales
      fit_profile: None | one of config.settings.PEAK_FITTING_PROFILES
      fit_window_half_width: float

    Returns a dict: spectrum, processing_steps, errors, peaks, fit,
    fit_profile, n_points (of the ORIGINAL, unprocessed spectrum).
    """
    processing_steps: list[str] = []
    errors: list[str] = []
    working = df

    baseline_method = settings.get("baseline_method")
    if baseline_method in ("Linear (Endpoints)", "ALS (Asymmetric Least Squares)"):
        try:
            params = settings.get("baseline_params") or {}
            if baseline_method == "Linear (Endpoints)":
                working, _baseline, p = baseline_linear_endpoints(working, **params)
            else:
                working, _baseline, p = baseline_als(working, **params)
            processing_steps.append(f"Baseline correction: {p['method']}")
        except PreprocessingError as exc:
            errors.append(f"Baseline correction skipped: {exc}")

    normalization_method = settings.get("normalization_method")
    if normalization_method:
        try:
            working, p = normalize(working, method=normalization_method)
            processing_steps.append(f"Normalization: {p['method']}")
        except PreprocessingError as exc:
            errors.append(f"Normalization skipped: {exc}")

    smoothing_params = settings.get("smoothing_params")
    if smoothing_params:
        try:
            working, p = smooth(working, **smoothing_params)
            processing_steps.append(
                f"Smoothing: Savitzky-Golay "
                f"(window_length={p['window_length']}, polyorder={p['polyorder']})"
            )
        except PreprocessingError as exc:
            errors.append(f"Smoothing skipped: {exc}")

    result: dict = {
        "spectrum": working,
        "processing_steps": processing_steps,
        "errors": errors,
        "peaks": None,
        "fit": None,
        "fit_profile": None,
        "n_points": len(df),
    }

    peak_detection_params = settings.get("peak_detection_params")
    if peak_detection_params:
        try:
            prominence_pct = peak_detection_params.get("prominence_pct", 5.0)
            min_distance = peak_detection_params.get("min_distance", 5)
            prominence = resolve_prominence(working, prominence_pct)
            peak_df, _ = detect_peaks(working, prominence=prominence, min_distance=min_distance)
            result["peaks"] = peak_df
        except PeakDetectionError as exc:
            errors.append(f"Peak detection skipped: {exc}")
            peak_df = None

        fit_profile = settings.get("fit_profile")
        if fit_profile and result["peaks"] is not None and len(result["peaks"]) > 0:
            try:
                fit_df, _ = fit_peaks(
                    working,
                    result["peaks"],
                    profile=fit_profile,
                    window_half_width=settings.get("fit_window_half_width", 15.0),
                )
                result["fit"] = fit_df
                result["fit_profile"] = fit_profile
            except PeakFittingError as exc:
                errors.append(f"Peak fitting skipped: {exc}")

    return result
