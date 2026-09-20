"""
Chart builders for the Raman Spectroscopy Analyzer.

Every function returns a Plotly figure (interactive, the app's default) via
the `_plotly` variants, or a Matplotlib figure (static fallback, for the
"Interactive Plots" toggle and for export-to-image) via the `_static`
variants. Both read colors from `config.settings.CHART_COLORS`, a palette
validated with the dataviz skill's `validate_palette.js` for colorblind
safety and contrast -- see the comment above CHART_COLORS for the numbers.

Mark conventions used throughout (per the dataviz skill):
  * thin 2px lines; markers >= 8px
  * a legend is shown whenever a figure has >= 2 series, omitted for one
  * gridlines are recessive (light gray, only major ticks)
  * the baseline curve is muted gray + dashed -- it's informational, not a
    series competing for identity with the real data
  * peak/fit labels are selective (top peaks only), never one per point
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, REQUIRED_COLUMNS
from utils.peak_fitting import _PROFILE_FUNCS, _PROFILE_PARAM_NAMES

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS

_AXIS_LABELS = {"x": "Wavenumber (cm⁻¹)", "y": "Intensity (a.u.)"}
_MAX_PEAK_LABELS = 8  # selective labeling: annotate at most this many peaks


# --- shared layout helpers ------------------------------------------------------


def _plotly_layout(fig: go.Figure, title: str, show_legend: bool) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=_AXIS_LABELS["x"],
        yaxis_title=_AXIS_LABELS["y"],
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=CHART_COLORS["surface"],
        paper_bgcolor=CHART_COLORS["surface"],
        font=dict(family="Roboto, sans-serif", color="#262730"),
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )
    grid_kwargs = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], gridwidth=1, zeroline=False)
    fig.update_xaxes(**grid_kwargs)
    fig.update_yaxes(**grid_kwargs)
    return fig


def _mpl_style(ax, title: str, show_legend: bool) -> None:
    ax.set_title(title)
    ax.set_xlabel(_AXIS_LABELS["x"])
    ax.set_ylabel(_AXIS_LABELS["y"])
    ax.grid(True, color=CHART_COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(frameon=False)


# --- raw / processed spectrum ---------------------------------------------------


def plot_spectrum_plotly(df: pd.DataFrame, title: str = "Spectrum", color_key: str = "primary") -> go.Figure:
    """A single-series line plot of any spectrum (raw or processed)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[WAVENUMBER_COL],
            y=df[INTENSITY_COL],
            mode="lines",
            line=dict(color=CHART_COLORS[color_key], width=2),
            name=title,
        )
    )
    return _plotly_layout(fig, title, show_legend=False)


def plot_spectrum_static(df: pd.DataFrame, title: str = "Spectrum", color_key: str = "primary") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df[WAVENUMBER_COL], df[INTENSITY_COL], color=CHART_COLORS[color_key], linewidth=1.5)
    _mpl_style(ax, title, show_legend=False)
    fig.tight_layout()
    return fig


# --- baseline correction (raw + baseline overlay, and the corrected result) -----


def plot_baseline_plotly(
    df: pd.DataFrame, baseline: np.ndarray, corrected_df: pd.DataFrame, method_label: str
) -> go.Figure:
    """Raw spectrum with the fitted baseline overlaid, and the corrected
    spectrum below it, sharing the x-axis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"Raw spectrum + baseline ({method_label})", "Corrected spectrum"),
    )
    fig.add_trace(
        go.Scatter(
            x=df[WAVENUMBER_COL],
            y=df[INTENSITY_COL],
            mode="lines",
            line=dict(color=CHART_COLORS["primary"], width=2),
            name="Raw spectrum",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df[WAVENUMBER_COL],
            y=baseline,
            mode="lines",
            line=dict(color=CHART_COLORS["baseline"], width=2, dash="dash"),
            name="Baseline",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=corrected_df[WAVENUMBER_COL],
            y=corrected_df[INTENSITY_COL],
            mode="lines",
            line=dict(color=CHART_COLORS["secondary"], width=2),
            name="Corrected spectrum",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        plot_bgcolor=CHART_COLORS["surface"],
        paper_bgcolor=CHART_COLORS["surface"],
        font=dict(family="Roboto, sans-serif", color="#262730"),
        margin=dict(l=60, r=30, t=70, b=50),
        height=650,
        hovermode="x unified",
    )
    grid_kwargs = dict(showgrid=True, gridcolor=CHART_COLORS["grid"], gridwidth=1, zeroline=False)
    fig.update_xaxes(**grid_kwargs)
    fig.update_yaxes(**grid_kwargs)
    fig.update_xaxes(title_text=_AXIS_LABELS["x"], row=2, col=1)
    fig.update_yaxes(title_text=_AXIS_LABELS["y"], row=1, col=1)
    fig.update_yaxes(title_text=_AXIS_LABELS["y"], row=2, col=1)
    return fig


def plot_baseline_static(
    df: pd.DataFrame, baseline: np.ndarray, corrected_df: pd.DataFrame, method_label: str
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    ax1.plot(df[WAVENUMBER_COL], df[INTENSITY_COL], color=CHART_COLORS["primary"], linewidth=1.5, label="Raw spectrum")
    ax1.plot(
        df[WAVENUMBER_COL], baseline, color=CHART_COLORS["baseline"], linewidth=1.5, linestyle="--", label="Baseline"
    )
    _mpl_style(ax1, f"Raw spectrum + baseline ({method_label})", show_legend=True)
    ax1.set_xlabel("")

    ax2.plot(corrected_df[WAVENUMBER_COL], corrected_df[INTENSITY_COL], color=CHART_COLORS["secondary"], linewidth=1.5)
    _mpl_style(ax2, "Corrected spectrum", show_legend=False)
    fig.tight_layout()
    return fig


# --- peak detection ---------------------------------------------------------------


def _top_peak_indices(peak_df: pd.DataFrame, n: int) -> set:
    if len(peak_df) <= n:
        return set(peak_df.index)
    return set(peak_df.nlargest(n, INTENSITY_COL).index)


def plot_peaks_plotly(df: pd.DataFrame, peak_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[WAVENUMBER_COL],
            y=df[INTENSITY_COL],
            mode="lines",
            line=dict(color=CHART_COLORS["primary"], width=2),
            name="Spectrum",
        )
    )
    if len(peak_df) > 0:
        labeled = _top_peak_indices(peak_df, _MAX_PEAK_LABELS)
        fig.add_trace(
            go.Scatter(
                x=peak_df[WAVENUMBER_COL],
                y=peak_df[INTENSITY_COL],
                mode="markers+text",
                marker=dict(color=CHART_COLORS["secondary"], size=10, symbol="triangle-down"),
                text=[
                    f"{row[WAVENUMBER_COL]:.0f}" if idx in labeled else ""
                    for idx, row in peak_df.iterrows()
                ],
                textposition="top center",
                textfont=dict(size=11, color=CHART_COLORS["secondary"]),
                name="Detected peaks",
            )
        )
    return _plotly_layout(fig, "Detected peaks", show_legend=True)


def plot_peaks_static(df: pd.DataFrame, peak_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df[WAVENUMBER_COL], df[INTENSITY_COL], color=CHART_COLORS["primary"], linewidth=1.5, label="Spectrum")
    if len(peak_df) > 0:
        ax.scatter(
            peak_df[WAVENUMBER_COL],
            peak_df[INTENSITY_COL],
            color=CHART_COLORS["secondary"],
            marker="v",
            s=60,
            zorder=3,
            label="Detected peaks",
        )
        labeled = _top_peak_indices(peak_df, _MAX_PEAK_LABELS)
        for idx, row in peak_df.iterrows():
            if idx in labeled:
                ax.annotate(
                    f"{row[WAVENUMBER_COL]:.0f}",
                    (row[WAVENUMBER_COL], row[INTENSITY_COL]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=CHART_COLORS["secondary"],
                )
    _mpl_style(ax, "Detected peaks", show_legend=True)
    fig.tight_layout()
    return fig


# --- peak fitting -------------------------------------------------------------------


def _fit_curve_points(peak_row: pd.Series, profile: str, window_half_width: float, n: int = 200):
    func = _PROFILE_FUNCS[profile]
    param_names = _PROFILE_PARAM_NAMES[profile]
    center = peak_row[WAVENUMBER_COL]
    x_curve = np.linspace(center - window_half_width, center + window_half_width, n)
    params = [peak_row[name] for name in param_names]
    y_curve = func(x_curve, *params)
    return x_curve, y_curve


def plot_fit_plotly(
    df: pd.DataFrame, fit_df: pd.DataFrame, profile: str, window_half_width: float
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[WAVENUMBER_COL],
            y=df[INTENSITY_COL],
            mode="lines",
            line=dict(color=CHART_COLORS["primary"], width=2),
            name="Spectrum",
        )
    )
    converged = fit_df[fit_df["converged"]]
    for i, (_, row) in enumerate(converged.iterrows()):
        x_curve, y_curve = _fit_curve_points(row, profile, window_half_width)
        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                line=dict(color=CHART_COLORS["secondary"], width=2, dash="dot"),
                name="Fit" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup="fit",
            )
        )
    if len(converged) > 0:
        fig.add_trace(
            go.Scatter(
                x=converged[WAVENUMBER_COL],
                y=converged["amplitude"] if "amplitude" in converged else converged[WAVENUMBER_COL] * 0,
                mode="markers",
                marker=dict(color=CHART_COLORS["secondary"], size=9, symbol="x"),
                name="Peak centers",
            )
        )
    return _plotly_layout(fig, f"Peak fitting ({profile})", show_legend=True)


def plot_fit_static(df: pd.DataFrame, fit_df: pd.DataFrame, profile: str, window_half_width: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df[WAVENUMBER_COL], df[INTENSITY_COL], color=CHART_COLORS["primary"], linewidth=1.5, label="Spectrum")
    converged = fit_df[fit_df["converged"]]
    for i, (_, row) in enumerate(converged.iterrows()):
        x_curve, y_curve = _fit_curve_points(row, profile, window_half_width)
        ax.plot(
            x_curve,
            y_curve,
            color=CHART_COLORS["secondary"],
            linewidth=1.5,
            linestyle=":",
            label="Fit" if i == 0 else None,
        )
    _mpl_style(ax, f"Peak fitting ({profile})", show_legend=True)
    fig.tight_layout()
    return fig


# --- comparison across example spectra (bonus: overlay up to 3 series) -----------


def plot_comparison_plotly(spectra: dict[str, pd.DataFrame]) -> go.Figure:
    """Overlay up to 3 spectra by name -- e.g. comparing the 3 bundled
    example files. Colors are assigned in the fixed categorical order
    (primary, secondary, tertiary), never cycled or reassigned."""
    color_order = ["primary", "secondary", "tertiary"]
    fig = go.Figure()
    for (label, df), color_key in zip(spectra.items(), color_order):
        fig.add_trace(
            go.Scatter(
                x=df[WAVENUMBER_COL],
                y=df[INTENSITY_COL],
                mode="lines",
                line=dict(color=CHART_COLORS[color_key], width=2),
                name=label,
            )
        )
    return _plotly_layout(fig, "Spectrum comparison", show_legend=True)
