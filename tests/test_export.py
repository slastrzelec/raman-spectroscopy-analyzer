import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from utils.export import (
    build_batch_zip,
    build_summary_report,
    dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes,
)
from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


@pytest.fixture
def spectrum_df():
    return pd.DataFrame({WAVENUMBER_COL: [100.0, 200.0], INTENSITY_COL: [5.0, 10.0]})


def test_dataframe_to_csv_bytes_roundtrips(spectrum_df):
    raw = dataframe_to_csv_bytes(spectrum_df)
    assert isinstance(raw, bytes)
    import io

    roundtrip = pd.read_csv(io.BytesIO(raw))
    pd.testing.assert_frame_equal(roundtrip, spectrum_df)


def test_dataframe_to_csv_bytes_empty_df_has_header_only():
    empty = pd.DataFrame(columns=[WAVENUMBER_COL, INTENSITY_COL])
    raw = dataframe_to_csv_bytes(empty)
    text = raw.decode("utf-8").strip()
    assert text == f"{WAVENUMBER_COL},{INTENSITY_COL}"


def test_build_summary_report_minimal_contains_filename_and_points():
    report = build_summary_report(
        source_filename="example.txt", n_points=500, processing_steps=[]
    )
    assert "example.txt" in report
    assert "500" in report
    assert "none -- analysis on the raw spectrum" in report


def test_build_summary_report_includes_processing_steps():
    report = build_summary_report(
        source_filename="a.txt",
        n_points=10,
        processing_steps=["Baseline correction: ALS (lambda=1e5, p=0.01)", "Smoothing: Savitzky-Golay"],
    )
    assert "1. Baseline correction: ALS (lambda=1e5, p=0.01)" in report
    assert "2. Smoothing: Savitzky-Golay" in report


def test_build_summary_report_includes_peaks():
    peak_df = pd.DataFrame(
        {WAVENUMBER_COL: [150.0], INTENSITY_COL: [42.0], "prominence": [8.5]}
    )
    report = build_summary_report(
        source_filename="a.txt", n_points=10, processing_steps=[], peak_df=peak_df
    )
    assert "Detected peaks (1)" in report
    assert "150.00 cm" in report
    assert "42.000" in report


def test_build_summary_report_includes_fit_results_with_failed_and_converged():
    fit_df = pd.DataFrame(
        {
            WAVENUMBER_COL: [100.0, 200.0],
            "center": [100.5, np.nan],
            "r_squared": [0.995, np.nan],
            "converged": [True, False],
        }
    )
    report = build_summary_report(
        source_filename="a.txt",
        n_points=10,
        processing_steps=[],
        fit_df=fit_df,
        fit_profile="Gaussian",
    )
    assert "profile: Gaussian" in report
    assert "Converged: 1/2" in report
    assert "100.50 cm" in report
    assert "DID NOT CONVERGE" in report


# --- dataframe_to_xlsx_bytes ------------------------------------------------------


def test_dataframe_to_xlsx_bytes_roundtrips_single_sheet(spectrum_df):
    raw = dataframe_to_xlsx_bytes({"Spectrum": spectrum_df})
    assert isinstance(raw, bytes)
    roundtrip = pd.read_excel(io.BytesIO(raw), sheet_name="Spectrum")
    # xlsx has no distinct int/float type, so whole-number floats round-trip
    # as int64 -- compare values, not dtype.
    pd.testing.assert_frame_equal(roundtrip, spectrum_df, check_dtype=False)


def test_dataframe_to_xlsx_bytes_multiple_sheets():
    peaks = pd.DataFrame({WAVENUMBER_COL: [150.0], INTENSITY_COL: [42.0]})
    raw = dataframe_to_xlsx_bytes({"Spectrum": peaks, "Peaks": peaks})
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    assert set(sheets.keys()) == {"Spectrum", "Peaks"}


# --- build_batch_zip ---------------------------------------------------------------


@pytest.fixture
def batch_results(spectrum_df):
    peaks = pd.DataFrame({WAVENUMBER_COL: [150.0], INTENSITY_COL: [42.0], "prominence": [8.5]})
    return {
        "sample_a.txt": {
            "spectrum": spectrum_df,
            "peaks": peaks,
            "fit": None,
            "fit_profile": None,
            "n_points": len(spectrum_df),
            "processing_steps": ["Normalization: Min-Max"],
        },
        "sample_b.txt": {
            "spectrum": spectrum_df,
            "peaks": None,
            "fit": None,
            "fit_profile": None,
            "n_points": len(spectrum_df),
            "processing_steps": [],
        },
    }


def test_build_batch_zip_contains_one_xlsx_and_one_report_per_file(batch_results):
    raw = build_batch_zip(batch_results)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())
    assert names == {
        "sample_a/sample_a.xlsx",
        "sample_a/sample_a_report.txt",
        "sample_b/sample_b.xlsx",
        "sample_b/sample_b_report.txt",
    }


def test_build_batch_zip_xlsx_has_peaks_sheet_only_when_peaks_present(batch_results):
    raw = build_batch_zip(batch_results)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        a_xlsx = zf.read("sample_a/sample_a.xlsx")
        b_xlsx = zf.read("sample_b/sample_b.xlsx")
    a_sheets = pd.read_excel(io.BytesIO(a_xlsx), sheet_name=None)
    b_sheets = pd.read_excel(io.BytesIO(b_xlsx), sheet_name=None)
    assert "Peaks" in a_sheets
    assert "Peaks" not in b_sheets


def test_build_batch_zip_report_mentions_source_filename(batch_results):
    raw = build_batch_zip(batch_results)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        report = zf.read("sample_a/sample_a_report.txt").decode("utf-8")
    assert "sample_a.txt" in report
