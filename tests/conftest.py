"""
Shared pytest fixtures. Fixtures use the real example spectra bundled in
data/raw/ instead of synthetic data, so tests exercise the exact files the
app ships with.
"""

import io
import sys
from pathlib import Path

import pytest

# Make the project root importable (so `import config` / `import utils` work
# regardless of where pytest is invoked from).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import EXAMPLE_DATA_DIR, EXAMPLE_SPECTRA  # noqa: E402


@pytest.fixture
def example_spectrum_path() -> Path:
    """Path to the first bundled example spectrum file."""
    filename, _label = next(iter(EXAMPLE_SPECTRA.values()))
    return EXAMPLE_DATA_DIR / filename


@pytest.fixture
def example_spectrum_text(example_spectrum_path) -> str:
    return example_spectrum_path.read_text(encoding="utf-8")


@pytest.fixture
def example_spectrum_file_obj(example_spectrum_text) -> io.BytesIO:
    """An in-memory binary file-like object, as Streamlit's uploader provides."""
    return io.BytesIO(example_spectrum_text.encode("utf-8"))


@pytest.fixture
def example_spectrum_df(example_spectrum_path):
    """The first example spectrum already parsed into a DataFrame."""
    from utils.data_loading import _parse_spectrum_text

    text = example_spectrum_path.read_text(encoding="utf-8")
    return _parse_spectrum_text(text, example_spectrum_path.name)
