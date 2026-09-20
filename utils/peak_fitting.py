"""
Fitting analytical peak profiles (Gaussian / Lorentzian / Pseudo-Voigt) to
the peaks found by utils.peak_detection.

Each detected peak is fit independently in a small window around it. A
single peak failing to converge does not abort the whole batch -- it's
recorded as a failed fit (NaN row) so the UI can show "3 of 4 peaks fit
successfully" instead of crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


class PeakFittingError(ValueError):
    """Raised for invalid fitting parameters (not for a single peak's fit
    failing to converge -- that is recorded per-row instead)."""


# --- Profile functions --------------------------------------------------------


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, amplitude, center, width, eta):
    """eta in [0, 1] mixes Lorentzian (eta=1) and Gaussian (eta=0) sharing
    the same `width` (a simplified, widely-used pseudo-Voigt formulation)."""
    g = gaussian(x, amplitude, center, width)
    l_ = lorentzian(x, amplitude, center, width)
    return eta * l_ + (1 - eta) * g


_PROFILE_FUNCS = {
    "Gaussian": gaussian,
    "Lorentzian": lorentzian,
    "Pseudo-Voigt": pseudo_voigt,
}

_PROFILE_PARAM_NAMES = {
    "Gaussian": ("amplitude", "center", "sigma"),
    "Lorentzian": ("amplitude", "center", "gamma"),
    "Pseudo-Voigt": ("amplitude", "center", "width", "eta"),
}


def _initial_guess(profile: str, amplitude: float, center: float, width_guess: float):
    if profile in ("Gaussian", "Lorentzian"):
        return [amplitude, center, max(width_guess, 1e-3)]
    if profile == "Pseudo-Voigt":
        return [amplitude, center, max(width_guess, 1e-3), 0.5]
    raise PeakFittingError(f"Unknown profile: '{profile}'.")


def _bounds(profile: str, x_window: np.ndarray):
    x_min, x_max = x_window.min(), x_window.max()
    if profile in ("Gaussian", "Lorentzian"):
        lower = [0, x_min, 1e-6]
        upper = [np.inf, x_max, (x_max - x_min)]
    elif profile == "Pseudo-Voigt":
        lower = [0, x_min, 1e-6, 0]
        upper = [np.inf, x_max, (x_max - x_min), 1]
    else:
        raise PeakFittingError(f"Unknown profile: '{profile}'.")
    return lower, upper


def fit_peaks(
    df: pd.DataFrame,
    peak_df: pd.DataFrame,
    profile: str = "Lorentzian",
    window_half_width: float = 15.0,
) -> tuple[pd.DataFrame, dict]:
    """Fit `profile` to each row of `peak_df` in a window around it.

    Returns (fit_results_df, params). fit_results_df has one row per input
    peak with the fitted parameters, their 1-sigma uncertainties (prefixed
    `unc_`), an R^2 goodness-of-fit, and a `converged` flag -- a failed fit
    is NaN-filled rather than raising, so one bad peak doesn't lose the rest.
    """
    if profile not in _PROFILE_FUNCS:
        raise PeakFittingError(
            f"Unknown profile '{profile}'. Available: {list(_PROFILE_FUNCS)}."
        )
    if window_half_width <= 0:
        raise PeakFittingError("window_half_width must be positive.")
    if len(peak_df) == 0:
        raise PeakFittingError("No detected peaks to fit.")

    func = _PROFILE_FUNCS[profile]
    param_names = _PROFILE_PARAM_NAMES[profile]

    x_full = df[WAVENUMBER_COL].to_numpy(dtype=float)
    y_full = df[INTENSITY_COL].to_numpy(dtype=float)
    # utils.peak_detection reports "width" in samples (scipy.signal.find_peaks'
    # convention), not in Wavenumber units -- convert using the spectrum's
    # own sample spacing so the initial guess is on the right scale.
    dx = float(np.median(np.diff(x_full))) if len(x_full) > 1 else 1.0

    rows = []
    n_converged = 0

    for _, peak in peak_df.iterrows():
        center_guess = float(peak[WAVENUMBER_COL])
        amplitude_guess = float(peak[INTENSITY_COL])
        width_guess_samples = peak.get("width", None)
        if width_guess_samples is not None and width_guess_samples > 0:
            width_guess = width_guess_samples * dx
        else:
            width_guess = window_half_width / 3
        # keep the guess safely inside the fit window's bounds
        width_guess = min(width_guess, 0.9 * window_half_width)

        mask = (x_full >= center_guess - window_half_width) & (
            x_full <= center_guess + window_half_width
        )
        x_window = x_full[mask]
        y_window = y_full[mask]

        row = {WAVENUMBER_COL: center_guess, "converged": False, "r_squared": np.nan}
        for name in param_names:
            row[name] = np.nan
            row[f"unc_{name}"] = np.nan

        if len(x_window) < len(param_names) + 1:
            rows.append(row)
            continue

        p0 = _initial_guess(profile, amplitude_guess, center_guess, width_guess)
        lower, upper = _bounds(profile, x_window)

        try:
            popt, pcov = curve_fit(
                func, x_window, y_window, p0=p0, bounds=(lower, upper), maxfev=5000
            )
            perr = np.sqrt(np.diag(pcov))

            y_fit = func(x_window, *popt)
            ss_res = np.sum((y_window - y_fit) ** 2)
            ss_tot = np.sum((y_window - np.mean(y_window)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            for name, value, error in zip(param_names, popt, perr):
                row[name] = value
                row[f"unc_{name}"] = error
            row["r_squared"] = r_squared
            row["converged"] = True
            n_converged += 1
        except (RuntimeError, ValueError):
            pass  # leave the NaN-filled row -- this peak just didn't converge

        rows.append(row)

    fit_results_df = pd.DataFrame(rows)
    params = {
        "profile": profile,
        "window_half_width": window_half_width,
        "n_peaks": len(peak_df),
        "n_converged": n_converged,
    }
    return fit_results_df, params
