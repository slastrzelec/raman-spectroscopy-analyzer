import numpy as np
import pandas as pd
import pytest

from utils.peak_detection import PeakDetectionError, detect_peaks
from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


@pytest.fixture
def two_peak_spectrum():
    x = np.linspace(0, 200, 1000)
    y = (
        10.0
        + 80 * np.exp(-((x - 60) ** 2) / (2 * 2**2))
        + 50 * np.exp(-((x - 140) ** 2) / (2 * 2**2))
    )
    return pd.DataFrame({WAVENUMBER_COL: x, INTENSITY_COL: y})


def test_detect_peaks_finds_both_peaks(two_peak_spectrum):
    peak_df, params = detect_peaks(two_peak_spectrum, prominence=5.0, min_distance=5)
    assert len(peak_df) == 2
    assert params["n_peaks"] == 2
    wavenumbers = peak_df[WAVENUMBER_COL].to_numpy()
    assert np.isclose(wavenumbers[0], 60, atol=1)
    assert np.isclose(wavenumbers[1], 140, atol=1)


def test_detect_peaks_sorted_by_wavenumber(two_peak_spectrum):
    peak_df, _ = detect_peaks(two_peak_spectrum, prominence=5.0, min_distance=5)
    assert peak_df[WAVENUMBER_COL].is_monotonic_increasing


def test_detect_peaks_high_prominence_filters_small_peaks(two_peak_spectrum):
    peak_df, _ = detect_peaks(two_peak_spectrum, prominence=1000.0, min_distance=5)
    assert len(peak_df) == 0


def test_detect_peaks_rejects_zero_prominence(two_peak_spectrum):
    with pytest.raises(PeakDetectionError):
        detect_peaks(two_peak_spectrum, prominence=0, min_distance=5)


def test_detect_peaks_rejects_min_distance_too_large(two_peak_spectrum):
    with pytest.raises(PeakDetectionError):
        detect_peaks(two_peak_spectrum, prominence=5.0, min_distance=10_000)


def test_detect_peaks_on_real_example_spectrum_returns_dataframe(example_spectrum_df):
    peak_df, params = detect_peaks(example_spectrum_df, prominence=5.0, min_distance=5)
    assert set(["peak_index", WAVENUMBER_COL, INTENSITY_COL, "prominence", "width"]) <= set(
        peak_df.columns
    )
    assert isinstance(params["n_peaks"], int)
