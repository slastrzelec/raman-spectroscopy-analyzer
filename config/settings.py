"""
Central configuration and default values for the Raman Spectroscopy Analyzer.

Keeping these in one place means every module (and every test) reads the
same defaults instead of duplicating magic numbers.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# --- Example spectra ---------------------------------------------------------
# Maps a stable internal id -> (filename, human-readable label).
# The label is what the user sees in the "choose an example" selector.

EXAMPLE_SPECTRA = {
    "cnt_cooh_a": ("CNT-COOH_a.txt", "Example 1 — CNT-COOH (a)"),
    "cnt_cooh_b": ("CNT-COOH_b.txt", "Example 2 — CNT-COOH (b)"),
    "cnt_cooh_c": ("CNT-COOH_c.txt", "Example 3 — CNT-COOH (c)"),
}

# --- File upload -------------------------------------------------------------

ALLOWED_UPLOAD_EXTENSIONS = (".txt",)
MAX_UPLOAD_SIZE_MB = 200
REQUIRED_COLUMNS = ("Wavenumber", "Intensity")

# --- Baseline correction ------------------------------------------------------

BASELINE_DEFAULTS = {
    "linear_n_points": 5,
    "linear_n_points_min": 3,
    "linear_n_points_max": 20,
    "als_lambda": 1e5,
    "als_lambda_min": 1e2,
    "als_lambda_max": 1e8,
    "als_p": 0.01,
    "als_p_min": 0.001,
    "als_p_max": 0.1,
    "als_n_iter": 10,
}

# --- Normalization -------------------------------------------------------------

NORMALIZATION_METHODS = ("Min-Max", "Area (integral)", "Max intensity")
NORMALIZATION_DEFAULT_METHOD = "Min-Max"

# --- Smoothing -----------------------------------------------------------------

SMOOTHING_DEFAULTS = {
    "window_length": 11,
    "window_length_min": 5,
    "window_length_max": 51,
    "polyorder": 3,
    "polyorder_min": 1,
    "polyorder_max": 5,
}

# --- Peak detection --------------------------------------------------------------

PEAK_DETECTION_DEFAULTS = {
    # Prominence is expressed as a % of the spectrum's own intensity range
    # (max - min), not an absolute value -- an absolute default tuned for a
    # raw spectrum (values in the tens/hundreds) silently finds zero peaks
    # on an area-normalized spectrum (values ~1e-3), which is confusing.
    # utils.peak_detection.resolve_prominence() converts % -> absolute.
    "prominence_pct": 5.0,
    "prominence_pct_min": 0.1,
    "prominence_pct_max": 50.0,
    "min_distance": 5,
    "min_distance_min": 1,
    "min_distance_max": 50,
}

# --- Peak fitting ------------------------------------------------------------------

PEAK_FITTING_DEFAULTS = {
    "profile": "Lorentzian",
    "window_half_width": 15,  # cm^-1 window around each detected peak used for fitting
}
PEAK_FITTING_PROFILES = ("Lorentzian", "Gaussian", "Pseudo-Voigt")

# --- Visual theme (matches the portfolio site: MkDocs Material, deep orange) -------

THEME = {
    "primary_color": "#FF5722",     # Material "deep orange 500"
    "accent_color": "#FF9800",      # Material "orange 500"
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F5F5F5",
    "text_color": "#262730",
    "font": "Roboto, sans-serif",
    "grid_color": "#E0E0E0",
}

# --- Chart colors --------------------------------------------------------------
# Validated with the dataviz skill's scripts/validate_palette.js (light mode):
# all pairs pass lightness band, chroma floor, CVD separation (protan/deutan/
# tritan >= 12.6 dE, worst case) and normal-vision separation (>= 13 dE).
# CHART_BASELINE is deliberately muted/gray -- it marks an *informational*
# curve (the fitted background), not a data series competing for identity.

CHART_COLORS = {
    "primary": "#FF5722",     # raw / main spectrum series (matches THEME)
    "secondary": "#1565C0",   # corrected / fitted / processed series
    "tertiary": "#00897B",    # 3rd series, e.g. comparing example spectra
    "baseline": "#9E9E9E",    # muted gray -- baseline curve (informational)
    "grid": "#E0E0E0",
    "surface": "#FFFFFF",
}
