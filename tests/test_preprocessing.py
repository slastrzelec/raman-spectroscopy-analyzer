import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    PreprocessingError,
    baseline_als,
    baseline_linear_endpoints,
    normalize,
    smooth,
)
from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


@pytest.fixture
def synthetic_spectrum():
    """A simple peak on a linear baseline -- easy to reason about analytically."""
    x = np.linspace(0, 100, 200)
    baseline = 2.0 * x + 10.0
    peak = 50 * np.exp(-((x - 50) ** 2) / (2 * 3**2))
    y = baseline + peak
    return pd.DataFrame({WAVENUMBER_COL: x, INTENSITY_COL: y})


# --- baseline_linear_endpoints -----------------------------------------------


def test_baseline_linear_removes_linear_trend(synthetic_spectrum):
    corrected_df, baseline, params = baseline_linear_endpoints(synthetic_spectrum, n_points=5)
    # far from the peak, corrected intensity should be close to zero
    edge_values = corrected_df[INTENSITY_COL].to_numpy()[:10]
    assert np.allclose(edge_values, 0, atol=1.0)
    assert params["method"] == "Linear (Endpoints)"
    assert len(baseline) == len(synthetic_spectrum)


def test_baseline_linear_rejects_too_large_n_points(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        baseline_linear_endpoints(synthetic_spectrum, n_points=1000)


def test_baseline_linear_rejects_zero_n_points(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        baseline_linear_endpoints(synthetic_spectrum, n_points=0)


def test_baseline_linear_preserves_wavenumber(synthetic_spectrum):
    corrected_df, _, _ = baseline_linear_endpoints(synthetic_spectrum, n_points=5)
    assert np.array_equal(
        corrected_df[WAVENUMBER_COL].to_numpy(), synthetic_spectrum[WAVENUMBER_COL].to_numpy()
    )


# --- baseline_als -------------------------------------------------------------


def test_baseline_als_reduces_edge_intensity(synthetic_spectrum):
    corrected_df, baseline, params = baseline_als(synthetic_spectrum, lam=1e4, p=0.01, n_iter=10)
    edge_values = corrected_df[INTENSITY_COL].to_numpy()[:10]
    assert np.mean(np.abs(edge_values)) < np.mean(np.abs(synthetic_spectrum[INTENSITY_COL].to_numpy()[:10]))
    assert params["method"] == "ALS (Asymmetric Least Squares)"


def test_baseline_als_rejects_invalid_p(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        baseline_als(synthetic_spectrum, p=1.5)
    with pytest.raises(PreprocessingError):
        baseline_als(synthetic_spectrum, p=0)


def test_baseline_als_rejects_negative_lambda(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        baseline_als(synthetic_spectrum, lam=-1)


def test_baseline_als_on_real_example_spectrum_does_not_raise(example_spectrum_df):
    baseline_als(example_spectrum_df, lam=1e5, p=0.01, n_iter=5)  # should not raise


# --- normalize -----------------------------------------------------------------


def test_normalize_min_max_range(synthetic_spectrum):
    normalized_df, params = normalize(synthetic_spectrum, method="Min-Max")
    y = normalized_df[INTENSITY_COL].to_numpy()
    assert np.isclose(y.min(), 0.0)
    assert np.isclose(y.max(), 1.0)
    assert params["method"] == "Min-Max"


def test_normalize_max_intensity(synthetic_spectrum):
    normalized_df, _ = normalize(synthetic_spectrum, method="Max intensity")
    assert np.isclose(normalized_df[INTENSITY_COL].max(), 1.0)


def test_normalize_area(synthetic_spectrum):
    normalized_df, _ = normalize(synthetic_spectrum, method="Area (integral)")
    x = normalized_df[WAVENUMBER_COL].to_numpy()
    y = normalized_df[INTENSITY_COL].to_numpy()
    assert np.isclose(np.trapezoid(y, x), 1.0, atol=1e-6)


def test_normalize_unknown_method_raises(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        normalize(synthetic_spectrum, method="not-a-real-method")


def test_normalize_constant_spectrum_raises():
    df = pd.DataFrame({WAVENUMBER_COL: np.arange(10), INTENSITY_COL: np.full(10, 5.0)})
    with pytest.raises(PreprocessingError):
        normalize(df, method="Min-Max")


# --- smooth --------------------------------------------------------------------


def test_smooth_reduces_noise():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 100, 300)
    clean = np.sin(x / 10)
    noisy = clean + rng.normal(0, 0.3, size=x.shape)
    df = pd.DataFrame({WAVENUMBER_COL: x, INTENSITY_COL: noisy})

    smoothed_df, params = smooth(df, window_length=11, polyorder=3)
    noisy_residual = np.std(noisy - clean)
    smoothed_residual = np.std(smoothed_df[INTENSITY_COL].to_numpy() - clean)
    assert smoothed_residual < noisy_residual
    assert params == {"window_length": 11, "polyorder": 3}


def test_smooth_rejects_even_window_length(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        smooth(synthetic_spectrum, window_length=10, polyorder=3)


def test_smooth_rejects_polyorder_too_large(synthetic_spectrum):
    with pytest.raises(PreprocessingError):
        smooth(synthetic_spectrum, window_length=5, polyorder=5)


def test_smooth_rejects_window_larger_than_data():
    df = pd.DataFrame({WAVENUMBER_COL: np.arange(5), INTENSITY_COL: np.arange(5, dtype=float)})
    with pytest.raises(PreprocessingError):
        smooth(df, window_length=11, polyorder=3)
