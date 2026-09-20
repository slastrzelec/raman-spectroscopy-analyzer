"""
Exporting results (processed spectrum, peak table, fit results, a plain-text
summary report) as downloadable bytes.

Everything here returns `bytes` (or `str` for the summary) rather than
writing to disk -- the app hands these straight to `st.download_button`, so
nothing about a user's data ever touches the server's filesystem.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import REQUIRED_COLUMNS

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Generic CSV export used for the spectrum, peak table, and fit table."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Multi-sheet .xlsx export (openpyxl engine, in-memory -- nothing
    touches disk). `sheets` maps sheet name -> DataFrame; names are
    truncated to Excel's 31-character sheet-name limit."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def build_summary_report(
    *,
    source_filename: str,
    n_points: int,
    processing_steps: list[str],
    peak_df: pd.DataFrame | None = None,
    fit_df: pd.DataFrame | None = None,
    fit_profile: str | None = None,
) -> str:
    """Build a plain-text analysis summary for download.

    `processing_steps` is a list of human-readable strings describing each
    step applied, in order (e.g. "Baseline correction: ALS (lambda=1e5, p=0.01)").
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Raman Spectroscopy Analyzer -- Analysis Report")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Source file: {source_filename}")
    lines.append(f"Spectrum points: {n_points}")
    lines.append("")

    lines.append("-- Processing steps applied --")
    if processing_steps:
        for i, step in enumerate(processing_steps, start=1):
            lines.append(f"  {i}. {step}")
    else:
        lines.append("  (none -- analysis on the raw spectrum)")
    lines.append("")

    if peak_df is not None and len(peak_df) > 0:
        lines.append(f"-- Detected peaks ({len(peak_df)}) --")
        for _, row in peak_df.iterrows():
            lines.append(
                f"  {row[WAVENUMBER_COL]:.2f} cm⁻¹  |  intensity {row[INTENSITY_COL]:.3f}"
                f"  |  prominence {row.get('prominence', float('nan')):.3f}"
            )
        lines.append("")

    if fit_df is not None and len(fit_df) > 0:
        n_converged = int(fit_df["converged"].sum())
        lines.append(f"-- Peak fitting (profile: {fit_profile}) --")
        lines.append(f"  Converged: {n_converged}/{len(fit_df)} peaks")
        for _, row in fit_df.iterrows():
            if not row["converged"]:
                lines.append(f"  center ~{row[WAVENUMBER_COL]:.2f} cm⁻¹  |  DID NOT CONVERGE")
                continue
            center = row.get("center", row[WAVENUMBER_COL])
            r2 = row.get("r_squared", float("nan"))
            lines.append(f"  center {center:.2f} cm⁻¹  |  R² {r2:.4f}")
        lines.append("")

    lines.append("=" * 70)
    lines.append(
        "Note: this report was generated automatically. The processing and "
        "fitting parameters above describe exactly the settings that were applied."
    )
    return "\n".join(lines)


def build_batch_zip(results: dict[str, dict]) -> bytes:
    """Package per-file batch results into one downloadable ZIP.

    `results` maps filename -> {
        "spectrum": DataFrame, "peaks": DataFrame | None, "fit": DataFrame | None,
        "fit_profile": str | None, "n_points": int, "processing_steps": list[str],
    } -- the shape produced by utils.batch.run_pipeline. Each file gets its own
    "<stem>/<stem>.xlsx" (Spectrum/Peaks/Fit sheets) and "<stem>/<stem>_report.txt"
    inside the archive, built entirely in memory.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, result in results.items():
            stem = Path(filename).stem

            sheets = {"Spectrum": result["spectrum"]}
            if result.get("peaks") is not None and len(result["peaks"]) > 0:
                sheets["Peaks"] = result["peaks"]
            if result.get("fit") is not None and len(result["fit"]) > 0:
                sheets["Fit"] = result["fit"]
            zf.writestr(f"{stem}/{stem}.xlsx", dataframe_to_xlsx_bytes(sheets))

            summary = build_summary_report(
                source_filename=filename,
                n_points=result["n_points"],
                processing_steps=result["processing_steps"],
                peak_df=result.get("peaks"),
                fit_df=result.get("fit"),
                fit_profile=result.get("fit_profile"),
            )
            zf.writestr(f"{stem}/{stem}_report.txt", summary)
    return buffer.getvalue()
