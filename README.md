# Raman Spectroscopy Analyzer

An interactive Streamlit app for analyzing Raman spectra: baseline correction,
normalization, smoothing, peak detection and fitting, and result export.

Full project specification (architecture, dependency version rationale, data
security): see [`SPEC.md`](SPEC.md).

🚀 **Live demo:** [raman-spectroscopy-analyzer.streamlit.app](https://raman-spectroscopy-analyzer.streamlit.app/)

## Features

- Spectrum loading: one of 3 sample CNT-COOH spectra, or your own `.txt` file
  (two columns: Wavenumber, Intensity) — upload also supports multiple files at once
- Preprocessing: baseline correction (Linear / ALS), normalization, smoothing
- Peak detection and fitting (Lorentzian / Gaussian / Pseudo-Voigt)
- Interactive Plotly charts (with a Matplotlib fallback)
- Result export: CSV (spectrum/peaks/fit), XLSX (everything in one file),
  a TXT report — plus **batch export**: the same settings applied to every
  uploaded file at once, packaged into a single ZIP

## Screenshots

**Data Overview** — metadata of the loaded file and a preview of the raw spectrum:

![Data Overview](screenshots/data-overview.png)

**Preprocessing — baseline correction** — the raw spectrum with the fitted
baseline overlaid (top), and the corrected spectrum below, with the D, G and 2D
bands clearly visible:

![Baseline correction](screenshots/baseline-correction.png)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11 (see `runtime.txt`).

## Tests

```bash
pytest
```

## Project structure

```
app.py              # Streamlit UI (orchestration, no business logic)
config/settings.py  # constants and default values
utils/               # logic: data loading, preprocessing, peak detection/fitting,
                     # visualization, export, batch processing
data/raw/            # sample spectra
tests/               # pytest tests for each module in utils/
```
