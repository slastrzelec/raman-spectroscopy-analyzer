import numpy as np
import pandas as pd
import pytest

from utils.peak_fitting import (
    PeakFittingError,
    fit_peaks,
    gaussian,
    lorentzian,
    pseudo_voigt,
)
from utils.peak_detection import detect_peaks
from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


@pytest.fixture
def single_gaussian_peak():
    x = np.linspace(0, 200, 2000)
    y = 5.0 + gaussian(x, amplitude=100, center=100, sigma=4)
    df = pd.DataFrame({WAVENUMBER_COL: x, INTENSITY_COL: y})
    peak_df, _ = detect_peaks(df, prominence=5.0, min_distance=5)
    return df, peak_df


def test_gaussian_peak_at_zero_offset_equals_amplitude():
    assert np.isclose(gaussian(np.array([5.0]), amplitude=10, center=5, sigma=2)[0], 10.0)


def test_lorentzian_peak_at_center_equals_amplitude():
    assert np.isclose(lorentzian(np.array([5.0]), amplitude=10, center=5, gamma=2)[0], 10.0)


def test_pseudo_voigt_at_center_equals_amplitude():
    assert np.isclose(
        pseudo_voigt(np.array([5.0]), amplitude=10, center=5, width=2, eta=0.5)[0], 10.0
    )


def test_fit_peaks_recovers_known_gaussian_center(single_gaussian_peak):
    df, peak_df = single_gaussian_peak
    assert len(peak_df) == 1

    fit_df, params = fit_peaks(df, peak_df, profile="Gaussian", window_half_width=15)
    assert len(fit_df) == 1
    assert fit_df.iloc[0]["converged"]
    assert np.isclose(fit_df.iloc[0]["center"], 100, atol=0.5)
    assert np.isclose(fit_df.iloc[0]["amplitude"], 100, atol=5)
    assert fit_df.iloc[0]["r_squared"] > 0.99
    assert params["n_converged"] == 1


def test_fit_peaks_lorentzian_profile_has_gamma_column(single_gaussian_peak):
    df, peak_df = single_gaussian_peak
    fit_df, _ = fit_peaks(df, peak_df, profile="Lorentzian", window_half_width=15)
    assert "gamma" in fit_df.columns
    assert "unc_gamma" in fit_df.columns


def test_fit_peaks_pseudo_voigt_has_eta_column(single_gaussian_peak):
    df, peak_df = single_gaussian_peak
    fit_df, _ = fit_peaks(df, peak_df, profile="Pseudo-Voigt", window_half_width=15)
    assert "eta" in fit_df.columns
    assert 0 <= fit_df.iloc[0]["eta"] <= 1


def test_fit_peaks_rejects_unknown_profile(single_gaussian_peak):
    df, peak_df = single_gaussian_peak
    with pytest.raises(PeakFittingError):
        fit_peaks(df, peak_df, profile="not-a-profile")


def test_fit_peaks_rejects_empty_peak_list(single_gaussian_peak):
    df, _ = single_gaussian_peak
    empty_peaks = pd.DataFrame(columns=[WAVENUMBER_COL, INTENSITY_COL, "width"])
    with pytest.raises(PeakFittingError):
        fit_peaks(df, empty_peaks, profile="Gaussian")


def test_fit_peaks_rejects_nonpositive_window(single_gaussian_peak):
    df, peak_df = single_gaussian_peak
    with pytest.raises(PeakFittingError):
        fit_peaks(df, peak_df, profile="Gaussian", window_half_width=0)


def test_fit_peaks_on_real_example_spectrum(example_spectrum_df):
    peak_df, _ = detect_peaks(example_spectrum_df, prominence=5.0, min_distance=5)
    if len(peak_df) == 0:
        pytest.skip("no peaks detected in raw example spectrum at these settings")
    fit_df, params = fit_peaks(example_spectrum_df, peak_df, profile="Lorentzian")
    assert len(fit_df) == len(peak_df)
    assert "converged" in fit_df.columns
