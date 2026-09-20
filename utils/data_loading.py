"""
Loading and validating Raman spectra.

Two entry points converge on the same parsing logic (`_parse_spectrum_text`)
so an example spectrum and an uploaded file are treated identically by the
rest of the app -- there is exactly one code path that turns raw text into a
validated DataFrame.

Security note: files are parsed as plain numeric text (str -> float), never
executed, evaluated or unpickled. Nothing here writes uploaded content to
disk -- callers keep it in memory (e.g. Streamlit's session_state).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from config.settings import (
    ALLOWED_UPLOAD_EXTENSIONS,
    EXAMPLE_DATA_DIR,
    EXAMPLE_SPECTRA,
    MAX_UPLOAD_SIZE_MB,
    REQUIRED_COLUMNS,
)


class SpectrumParseError(ValueError):
    """Raised when a file cannot be parsed as a valid Raman spectrum.

    The message is written to be shown directly to the user (Streamlit
    st.error), so it never leaks a raw traceback.
    """


@dataclass
class SpectrumMetadata:
    filename: str
    n_points: int
    wavenumber_range: tuple[float, float]
    intensity_range: tuple[float, float]

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "n_points": self.n_points,
            "wavenumber_range": self.wavenumber_range,
            "intensity_range": self.intensity_range,
        }


def list_example_spectra() -> list[tuple[str, str]]:
    """Return [(example_id, human_readable_label), ...] in a stable order."""
    return [(example_id, label) for example_id, (_, label) in EXAMPLE_SPECTRA.items()]


def validate_filename(filename: str) -> None:
    """Raise SpectrumParseError if the filename's extension isn't allowed."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(ALLOWED_UPLOAD_EXTENSIONS)
        raise SpectrumParseError(
            f"Unsupported file format '{suffix}'. Allowed extensions: {allowed}."
        )


def validate_size(size_bytes: int, filename: str) -> None:
    """Raise SpectrumParseError if the file exceeds MAX_UPLOAD_SIZE_MB."""
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise SpectrumParseError(
            f"File '{filename}' is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"The limit is {MAX_UPLOAD_SIZE_MB} MB."
        )


def _parse_spectrum_text(text: str, filename: str) -> pd.DataFrame:
    """Parse whitespace/tab-separated 'wavenumber intensity' lines into a DataFrame.

    Raises SpectrumParseError with a human-readable message on any malformed
    or empty content -- never lets a raw ValueError/IndexError escape.
    """
    wavenumbers: list[float] = []
    intensities: list[float] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SpectrumParseError(
                f"File '{filename}': line {line_no} doesn't have two columns "
                f"(Wavenumber, Intensity) separated by a space/tab."
            )
        try:
            wavenumber = float(parts[0])
            intensity = float(parts[1])
        except ValueError as exc:
            raise SpectrumParseError(
                f"File '{filename}': line {line_no} contains non-numeric values "
                f"('{parts[0]}', '{parts[1]}')."
            ) from exc
        wavenumbers.append(wavenumber)
        intensities.append(intensity)

    if len(wavenumbers) == 0:
        raise SpectrumParseError(
            f"File '{filename}' contains no valid data rows."
        )
    if len(wavenumbers) < 2:
        raise SpectrumParseError(
            f"File '{filename}' contains only {len(wavenumbers)} point(s) -- "
            f"a spectrum needs at least 2 points."
        )

    df = pd.DataFrame(
        {REQUIRED_COLUMNS[0]: wavenumbers, REQUIRED_COLUMNS[1]: intensities}
    )
    df = df.sort_values(REQUIRED_COLUMNS[0]).reset_index(drop=True)
    return df


def _build_metadata(df: pd.DataFrame, filename: str) -> SpectrumMetadata:
    return SpectrumMetadata(
        filename=filename,
        n_points=len(df),
        wavenumber_range=(
            float(df[REQUIRED_COLUMNS[0]].min()),
            float(df[REQUIRED_COLUMNS[0]].max()),
        ),
        intensity_range=(
            float(df[REQUIRED_COLUMNS[1]].min()),
            float(df[REQUIRED_COLUMNS[1]].max()),
        ),
    )


def load_uploaded_spectrum(
    file_obj: Union[BinaryIO, io.BytesIO], filename: str, size_bytes: int
) -> tuple[pd.DataFrame, SpectrumMetadata]:
    """Validate and parse a user-uploaded spectrum file.

    `file_obj` is any binary file-like object (e.g. Streamlit's UploadedFile).
    Nothing here persists the upload to disk.
    """
    validate_filename(filename)
    validate_size(size_bytes, filename)

    raw_bytes = file_obj.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    df = _parse_spectrum_text(text, filename)
    metadata = _build_metadata(df, filename)
    return df, metadata


def load_example_spectrum(example_id: str) -> tuple[pd.DataFrame, SpectrumMetadata]:
    """Load one of the bundled example spectra by its internal id.

    Example spectra live in the repo (data/raw/) and are treated as
    read-only, trusted assets -- separate from anything a user uploads.
    """
    if example_id not in EXAMPLE_SPECTRA:
        raise SpectrumParseError(f"Unknown example id: '{example_id}'.")

    filename, _label = EXAMPLE_SPECTRA[example_id]
    path = EXAMPLE_DATA_DIR / filename
    if not path.exists():
        raise SpectrumParseError(
            f"Missing example file '{filename}' in {EXAMPLE_DATA_DIR}."
        )

    text = path.read_text(encoding="utf-8")
    df = _parse_spectrum_text(text, filename)
    metadata = _build_metadata(df, filename)
    return df, metadata
