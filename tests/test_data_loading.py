import io

import pytest

from utils.data_loading import (
    SpectrumParseError,
    list_example_spectra,
    load_example_spectrum,
    load_uploaded_spectrum,
    validate_filename,
    validate_size,
)
from config.settings import EXAMPLE_SPECTRA, REQUIRED_COLUMNS


def test_list_example_spectra_matches_config():
    listed = list_example_spectra()
    assert len(listed) == len(EXAMPLE_SPECTRA)
    ids = {example_id for example_id, _label in listed}
    assert ids == set(EXAMPLE_SPECTRA.keys())


@pytest.mark.parametrize("example_id", list(EXAMPLE_SPECTRA.keys()))
def test_load_each_example_spectrum(example_id):
    df, metadata = load_example_spectrum(example_id)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) == metadata.n_points
    assert metadata.n_points > 1000  # real spectra are large
    assert metadata.wavenumber_range[0] < metadata.wavenumber_range[1]


def test_load_example_spectrum_unknown_id_raises():
    with pytest.raises(SpectrumParseError):
        load_example_spectrum("does_not_exist")


def test_load_uploaded_spectrum_matches_example(example_spectrum_file_obj, example_spectrum_path):
    size = example_spectrum_path.stat().st_size
    df, metadata = load_uploaded_spectrum(
        example_spectrum_file_obj, example_spectrum_path.name, size
    )
    assert metadata.filename == example_spectrum_path.name
    assert len(df) == metadata.n_points


def test_load_uploaded_spectrum_sorted_by_wavenumber(example_spectrum_file_obj, example_spectrum_path):
    size = example_spectrum_path.stat().st_size
    df, _ = load_uploaded_spectrum(example_spectrum_file_obj, example_spectrum_path.name, size)
    assert df[REQUIRED_COLUMNS[0]].is_monotonic_increasing


def test_validate_filename_rejects_bad_extension():
    with pytest.raises(SpectrumParseError):
        validate_filename("spectrum.exe")


def test_validate_filename_accepts_txt():
    validate_filename("spectrum.txt")  # should not raise


def test_validate_size_rejects_oversized_file():
    with pytest.raises(SpectrumParseError):
        validate_size(500 * 1024 * 1024, "big.txt")  # 500 MB > 200 MB limit


def test_load_uploaded_spectrum_rejects_non_txt_extension():
    fake = io.BytesIO(b"100 1\n200 2\n")
    with pytest.raises(SpectrumParseError):
        load_uploaded_spectrum(fake, "malicious.exe", fake.getbuffer().nbytes)


def test_parse_rejects_non_numeric_content():
    fake = io.BytesIO(b"this is not a spectrum at all\nneither is this\n")
    with pytest.raises(SpectrumParseError):
        load_uploaded_spectrum(fake, "bad.txt", fake.getbuffer().nbytes)


def test_parse_rejects_single_line_file():
    fake = io.BytesIO(b"100.0 5.0\n")
    with pytest.raises(SpectrumParseError):
        load_uploaded_spectrum(fake, "tiny.txt", fake.getbuffer().nbytes)


def test_parse_rejects_empty_file():
    fake = io.BytesIO(b"")
    with pytest.raises(SpectrumParseError):
        load_uploaded_spectrum(fake, "empty.txt", 0)


def test_parse_skips_blank_lines():
    fake = io.BytesIO(b"100.0 5.0\n\n200.0 6.0\n\n\n300.0 7.0\n")
    df, metadata = load_uploaded_spectrum(fake, "spaced.txt", fake.getbuffer().nbytes)
    assert metadata.n_points == 3
