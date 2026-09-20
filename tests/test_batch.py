import numpy as np
import pandas as pd
import pytest

from utils.batch import run_pipeline
from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


@pytest.fixture
def spectrum_with_peak():
    x = np.linspace(0, 200, 2000)
    baseline = 0.05 * x + 2.0
    y = baseline + 5.0 + 100 * np.exp(-((x - 100) ** 2) / (2 * 4**2))
    return pd.DataFrame({WAVENUMBER_COL: x, INTENSITY_COL: y})


def test_run_pipeline_empty_settings_returns_spectrum_unchanged(spectrum_with_peak):
    result = run_pipeline(spectrum_with_peak, {})
    pd.testing.assert_frame_equal(result["spectrum"], spectrum_with_peak)
    assert result["processing_steps"] == []
    assert result["errors"] == []
    assert result["peaks"] is None
    assert result["fit"] is None
    assert result["n_points"] == len(spectrum_with_peak)


def test_run_pipeline_full_settings_runs_every_stage(spectrum_with_peak):
    settings = {
        "baseline_method": "ALS (Asymmetric Least Squares)",
        "baseline_params": {"lam": 1e5, "p": 0.01, "n_iter": 5},
        "normalization_method": "Min-Max",
        "smoothing_params": {"window_length": 11, "polyorder": 3},
        "peak_detection_params": {"prominence_pct": 0.1, "min_distance": 5},
        "fit_profile": "Gaussian",
        "fit_window_half_width": 15.0,
    }
    result = run_pipeline(spectrum_with_peak, settings)
    assert len(result["processing_steps"]) == 3
    assert result["errors"] == []
    assert result["peaks"] is not None
    assert len(result["peaks"]) >= 1
    assert result["fit"] is not None
    assert result["fit_profile"] == "Gaussian"
    assert result["fit"]["converged"].any()


def test_run_pipeline_baseline_only(spectrum_with_peak):
    settings = {
        "baseline_method": "Linear (Endpoints)",
        "baseline_params": {"n_points": 5},
    }
    result = run_pipeline(spectrum_with_peak, settings)
    assert result["processing_steps"] == ["Baseline correction: Linear (Endpoints)"]
    assert result["peaks"] is None


def test_run_pipeline_invalid_baseline_params_recorded_as_error_not_raised(spectrum_with_peak):
    # n_points far larger than the spectrum -> PreprocessingError inside baseline_linear_endpoints
    settings = {
        "baseline_method": "Linear (Endpoints)",
        "baseline_params": {"n_points": 100_000},
        "normalization_method": "Min-Max",
    }
    result = run_pipeline(spectrum_with_peak, settings)
    assert any("Baseline correction skipped" in e for e in result["errors"])
    # later steps still ran on the un-baseline-corrected spectrum
    assert result["processing_steps"] == ["Normalization: Min-Max"]


def test_run_pipeline_peak_detection_without_fit_profile_only_detects(spectrum_with_peak):
    settings = {"peak_detection_params": {"prominence_pct": 0.5, "min_distance": 5}}
    result = run_pipeline(spectrum_with_peak, settings)
    assert result["peaks"] is not None
    assert result["fit"] is None


def test_run_pipeline_no_peaks_found_skips_fit_without_error(spectrum_with_peak):
    settings = {
        "peak_detection_params": {"prominence_pct": 1000.0, "min_distance": 5},
        "fit_profile": "Gaussian",
    }
    result = run_pipeline(spectrum_with_peak, settings)
    assert len(result["peaks"]) == 0
    assert result["fit"] is None
    assert result["errors"] == []
