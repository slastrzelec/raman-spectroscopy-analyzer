"""
Raman Spectroscopy Analyzer -- Streamlit app.

Flow: choose a spectrum (bundled example or your own upload) in the sidebar
-> Data Overview -> Preprocessing (baseline / normalization / smoothing,
applied in that order, each optional) -> Peak Detection -> Peak Fitting ->
Analysis & Results -> Export.

Security note (see SPEC.md "Bezpieczeństwo danych"): uploaded files are
parsed as plain numeric text and kept only in st.session_state (memory) --
never written to disk, never eval/exec/pickle'd. No outbound network calls
are made anywhere in this app.
"""

from __future__ import annotations

import streamlit as st

from config.settings import (
    BASELINE_DEFAULTS,
    MAX_UPLOAD_SIZE_MB,
    NORMALIZATION_METHODS,
    PEAK_DETECTION_DEFAULTS,
    PEAK_FITTING_DEFAULTS,
    PEAK_FITTING_PROFILES,
    REQUIRED_COLUMNS,
    SMOOTHING_DEFAULTS,
    THEME,
)
from utils.data_loading import (
    SpectrumParseError,
    list_example_spectra,
    load_example_spectrum,
    load_uploaded_spectrum,
)
from utils.batch import run_pipeline
from utils.export import (
    build_batch_zip,
    build_summary_report,
    dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes,
)
from utils.peak_detection import PeakDetectionError, detect_peaks, resolve_prominence
from utils.peak_fitting import PeakFittingError, fit_peaks
from utils.preprocessing import (
    PreprocessingError,
    baseline_als,
    baseline_linear_endpoints,
    normalize,
    smooth,
)
from utils.visualization import (
    plot_baseline_plotly,
    plot_baseline_static,
    plot_fit_plotly,
    plot_fit_static,
    plot_peaks_plotly,
    plot_peaks_static,
    plot_spectrum_plotly,
    plot_spectrum_static,
)

WAVENUMBER_COL, INTENSITY_COL = REQUIRED_COLUMNS


# --- page setup -----------------------------------------------------------------

st.set_page_config(
    page_title="Raman Spectroscopy Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    h1 {{ color: {THEME['primary_color']}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 500; }}
    div[data-testid="stMetricValue"] {{ color: {THEME['primary_color']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- session state ----------------------------------------------------------------

_DEFAULTS = {
    "raw_df": None,
    "metadata": None,
    "source_filename": None,
    "processed_df": None,
    "processing_steps": [],
    "last_baseline": None,
    "last_baseline_method": None,
    "last_normalization_method": None,
    "last_smoothing_params": None,
    "peak_df": None,
    "peak_params": None,
    "fit_df": None,
    "fit_params": None,
    "uploaded_spectra": {},  # {filename: (df, metadata)} -- populated in "multi-file upload" mode
    "_batch_zip": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_preprocessing() -> None:
    """Wipe the preprocessing pipeline back to the raw spectrum. Used when a
    new spectrum is loaded, or when the user explicitly hits Reset."""
    st.session_state.processed_df = st.session_state.raw_df
    st.session_state.processing_steps = []
    st.session_state.last_baseline = None
    st.session_state.last_baseline_method = None
    st.session_state.last_normalization_method = None
    st.session_state.last_smoothing_params = None
    _invalidate_from("detect")


def _invalidate_from(stage: str) -> None:
    """Clear results that depend on an earlier stage, so a changed spectrum
    or processing step never leaves stale peaks/fits displayed. Does NOT
    touch the preprocessing pipeline itself -- call after applying a
    preprocessing step, not before."""
    order = ["detect", "fit"]
    idx = order.index(stage)
    if idx <= order.index("detect"):
        st.session_state.peak_df = None
        st.session_state.peak_params = None
    if idx <= order.index("fit"):
        st.session_state.fit_df = None
        st.session_state.fit_params = None


def _load_spectrum(df, metadata, filename: str) -> None:
    st.session_state.raw_df = df
    st.session_state.metadata = metadata
    st.session_state.source_filename = filename
    _reset_preprocessing()


# --- sidebar: spectrum source + display options ------------------------------------

with st.sidebar:
    st.header("📁 Spectrum source")
    source_mode = st.radio(
        "Choose source", ["Example spectrum", "Upload your own file"], label_visibility="collapsed"
    )

    if source_mode == "Example spectrum":
        examples = list_example_spectra()
        example_labels = {label: example_id for example_id, label in examples}
        chosen_label = st.selectbox("Example spectrum", list(example_labels.keys()))
        if st.button("Load example", width="stretch"):
            example_id = example_labels[chosen_label]
            try:
                df, metadata = load_example_spectrum(example_id)
                _load_spectrum(df, metadata, metadata.filename)
                st.success(f"Loaded: {metadata.filename}")
            except SpectrumParseError as exc:
                st.error(str(exc))
    else:
        uploaded_list = st.file_uploader(
            f"Spectrum file(s) (.txt, max {MAX_UPLOAD_SIZE_MB} MB each) -- select several at "
            "once for batch mode",
            type=["txt"],
            accept_multiple_files=True,
        )
        if uploaded_list and st.button("Load file(s)", width="stretch"):
            loaded: dict = {}
            for uf in uploaded_list:
                try:
                    df, metadata = load_uploaded_spectrum(uf, uf.name, uf.size)
                    loaded[uf.name] = (df, metadata)
                except SpectrumParseError as exc:
                    st.error(str(exc))
            if loaded:
                st.session_state.uploaded_spectra = loaded
                first_name = next(iter(loaded))
                df, metadata = loaded[first_name]
                _load_spectrum(df, metadata, metadata.filename)
                if len(loaded) == 1:
                    st.success(f"Loaded: {metadata.filename}")
                else:
                    st.success(f"Loaded {len(loaded)} files. Active: {metadata.filename}")

        if len(st.session_state.uploaded_spectra) > 1:
            names = list(st.session_state.uploaded_spectra.keys())
            active_name = st.selectbox(
                "Active spectrum (for the analysis tabs)", names, key="active_upload_name"
            )
            if active_name != st.session_state.source_filename:
                df, metadata = st.session_state.uploaded_spectra[active_name]
                _load_spectrum(df, metadata, metadata.filename)
                st.rerun()
            st.caption(
                f"💡 {len(names)} files uploaded -- batch export is available in the "
                "'💾 Export' tab."
            )

    st.divider()
    st.header("⚙️ Display")
    interactive = st.checkbox("Interactive charts (Plotly)", value=True)

    if st.session_state.raw_df is not None:
        st.divider()
        st.caption(f"Active spectrum: **{st.session_state.source_filename}**")


# --- main: guard clause if nothing loaded yet -----------------------------------

st.title("📈 Raman Spectroscopy Analyzer")
st.caption(
    "Raman spectrum analysis: baseline correction, normalization, smoothing, "
    "peak detection and fitting."
)

if st.session_state.raw_df is None:
    st.info("👈 Choose an example spectrum or upload your own file in the sidebar to begin.")
    st.stop()

raw_df = st.session_state.raw_df
metadata = st.session_state.metadata
if st.session_state.processed_df is None:
    st.session_state.processed_df = raw_df


def _render_spectrum(df, title: str, color_key: str = "primary"):
    if interactive:
        st.plotly_chart(plot_spectrum_plotly(df, title, color_key), width="stretch")
    else:
        st.pyplot(plot_spectrum_static(df, title, color_key))


tab_overview, tab_preprocess, tab_detect, tab_fit, tab_results, tab_export = st.tabs(
    ["📊 Data Overview", "🧹 Preprocessing", "🔍 Peak Detection", "📐 Peak Fitting", "📋 Analysis & Results", "💾 Export"]
)


# --- Data Overview -----------------------------------------------------------------

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", metadata.filename)
    col2.metric("Number of points", metadata.n_points)
    col3.metric(
        "Wavenumber range", f"{metadata.wavenumber_range[0]:.0f}–{metadata.wavenumber_range[1]:.0f} cm⁻¹"
    )
    col4.metric(
        "Intensity range", f"{metadata.intensity_range[0]:.1f}–{metadata.intensity_range[1]:.1f}"
    )
    _render_spectrum(raw_df, "Raw spectrum")
    with st.expander("Data preview (table)"):
        st.dataframe(raw_df, width="stretch", height=300)


# --- Preprocessing -------------------------------------------------------------------

with tab_preprocess:
    working_df = st.session_state.processed_df

    st.subheader("1. Baseline correction")
    baseline_method = st.selectbox(
        "Method", ["None", "Linear (Endpoints)", "ALS (Asymmetric Least Squares)"], key="baseline_method"
    )
    if baseline_method == "Linear (Endpoints)":
        n_points = st.slider(
            "Number of edge points",
            BASELINE_DEFAULTS["linear_n_points_min"],
            BASELINE_DEFAULTS["linear_n_points_max"],
            BASELINE_DEFAULTS["linear_n_points"],
            key="baseline_n_points",
        )
        if st.button("Apply linear correction"):
            try:
                corrected, baseline, params = baseline_linear_endpoints(working_df, n_points=n_points)
                st.session_state.processed_df = corrected
                st.session_state.last_baseline = baseline
                st.session_state.last_baseline_method = params["method"]
                st.session_state.processing_steps.append(
                    f"Baseline correction: {params['method']} (n_points={n_points})"
                )
                _invalidate_from("detect")
                st.rerun()
            except PreprocessingError as exc:
                st.error(str(exc))
    elif baseline_method == "ALS (Asymmetric Least Squares)":
        c1, c2, c3 = st.columns(3)
        lam = c1.number_input(
            "lambda (smoothness)", BASELINE_DEFAULTS["als_lambda_min"], BASELINE_DEFAULTS["als_lambda_max"],
            BASELINE_DEFAULTS["als_lambda"], format="%.0f", key="als_lambda",
        )
        p = c2.number_input(
            "p (asymmetry)", BASELINE_DEFAULTS["als_p_min"], BASELINE_DEFAULTS["als_p_max"],
            BASELINE_DEFAULTS["als_p"], format="%.3f", key="als_p",
        )
        n_iter = c3.number_input(
            "number of iterations", 1, 50, BASELINE_DEFAULTS["als_n_iter"], key="als_n_iter"
        )
        if st.button("Apply ALS correction"):
            try:
                corrected, baseline, params = baseline_als(working_df, lam=lam, p=p, n_iter=int(n_iter))
                st.session_state.processed_df = corrected
                st.session_state.last_baseline = baseline
                st.session_state.last_baseline_method = params["method"]
                st.session_state.processing_steps.append(
                    f"Baseline correction: {params['method']} (lambda={lam:.0e}, p={p})"
                )
                _invalidate_from("detect")
                st.rerun()
            except PreprocessingError as exc:
                st.error(str(exc))

    if st.session_state.last_baseline is not None:
        if interactive:
            st.plotly_chart(
                plot_baseline_plotly(
                    raw_df, st.session_state.last_baseline, st.session_state.processed_df,
                    st.session_state.last_baseline_method,
                ),
                width="stretch",
            )
        else:
            st.pyplot(
                plot_baseline_static(
                    raw_df, st.session_state.last_baseline, st.session_state.processed_df,
                    st.session_state.last_baseline_method,
                )
            )

    st.divider()
    st.subheader("2. Normalization")
    norm_method = st.selectbox(
        "Normalization method", ["None"] + list(NORMALIZATION_METHODS), index=0, key="norm_method"
    )
    if norm_method != "None" and st.button("Apply normalization"):
        try:
            normalized, params = normalize(st.session_state.processed_df, method=norm_method)
            st.session_state.processed_df = normalized
            st.session_state.last_normalization_method = params["method"]
            st.session_state.processing_steps.append(f"Normalization: {params['method']}")
            _invalidate_from("detect")
            st.rerun()
        except PreprocessingError as exc:
            st.error(str(exc))

    if st.session_state.last_normalization_method is not None:
        st.success(f"Applied: {st.session_state.last_normalization_method} normalization.")

    st.divider()
    st.subheader("3. Smoothing (Savitzky-Golay)")
    c1, c2 = st.columns(2)
    window_length = c1.slider(
        "Window length (must be odd)",
        SMOOTHING_DEFAULTS["window_length_min"], SMOOTHING_DEFAULTS["window_length_max"],
        SMOOTHING_DEFAULTS["window_length"], step=2, key="smooth_window_length",
    )
    polyorder = c2.slider(
        "Polynomial order", SMOOTHING_DEFAULTS["polyorder_min"], SMOOTHING_DEFAULTS["polyorder_max"],
        SMOOTHING_DEFAULTS["polyorder"], key="smooth_polyorder",
    )
    if st.button("Apply smoothing"):
        try:
            smoothed, params = smooth(
                st.session_state.processed_df, window_length=window_length, polyorder=polyorder
            )
            st.session_state.processed_df = smoothed
            st.session_state.last_smoothing_params = {"window_length": window_length, "polyorder": polyorder}
            st.session_state.processing_steps.append(
                f"Smoothing: Savitzky-Golay (window_length={window_length}, polyorder={polyorder})"
            )
            _invalidate_from("detect")
            st.rerun()
        except PreprocessingError as exc:
            st.error(str(exc))

    if st.session_state.last_smoothing_params is not None:
        p = st.session_state.last_smoothing_params
        st.success(
            f"Applied: Savitzky-Golay smoothing (window_length={p['window_length']}, "
            f"polyorder={p['polyorder']})."
        )

    st.divider()
    if st.button("↩️ Reset preprocessing (back to raw spectrum)"):
        _reset_preprocessing()
        st.rerun()

    if st.session_state.processing_steps:
        st.caption("Steps applied: " + " → ".join(st.session_state.processing_steps))
    st.subheader("Processed spectrum")
    _render_spectrum(st.session_state.processed_df, "Processed spectrum", color_key="secondary")


# --- Peak Detection --------------------------------------------------------------------

with tab_detect:
    working_df = st.session_state.processed_df
    c1, c2 = st.columns(2)
    prominence_pct = c1.slider(
        "Prominence (% of intensity range)",
        PEAK_DETECTION_DEFAULTS["prominence_pct_min"], PEAK_DETECTION_DEFAULTS["prominence_pct_max"],
        PEAK_DETECTION_DEFAULTS["prominence_pct"], key="pk_prominence_pct",
    )
    min_distance = c2.slider(
        "Minimum distance between peaks (points)",
        PEAK_DETECTION_DEFAULTS["min_distance_min"], PEAK_DETECTION_DEFAULTS["min_distance_max"],
        PEAK_DETECTION_DEFAULTS["min_distance"], key="pk_min_distance",
    )
    prominence = resolve_prominence(working_df, prominence_pct)
    st.caption(
        f"= absolute prominence {prominence:.4g} on this spectrum's own intensity scale "
        f"(range {working_df[INTENSITY_COL].max() - working_df[INTENSITY_COL].min():.4g})"
    )
    if st.button("🔍 Detect peaks", type="primary"):
        try:
            peak_df, params = detect_peaks(working_df, prominence=prominence, min_distance=min_distance)
            _invalidate_from("fit")
            st.session_state.peak_df = peak_df
            st.session_state.peak_params = params
        except PeakDetectionError as exc:
            st.error(str(exc))

    if st.session_state.peak_df is not None:
        st.success(f"Detected {st.session_state.peak_params['n_peaks']} peaks.")
        if interactive:
            st.plotly_chart(plot_peaks_plotly(working_df, st.session_state.peak_df), width="stretch")
        else:
            st.pyplot(plot_peaks_static(working_df, st.session_state.peak_df))
        st.dataframe(st.session_state.peak_df, width="stretch")


# --- Peak Fitting -----------------------------------------------------------------------

with tab_fit:
    if st.session_state.peak_df is None or len(st.session_state.peak_df) == 0:
        st.info("First detect peaks in the 'Peak Detection' tab.")
    else:
        c1, c2 = st.columns(2)
        profile = c1.selectbox("Fitting profile", PEAK_FITTING_PROFILES, key="fit_profile_select")
        window_half_width = c2.number_input(
            "Window half-width (cm⁻¹)", min_value=1.0,
            value=float(PEAK_FITTING_DEFAULTS["window_half_width"]), key="fit_window_half_width",
        )
        if st.button("📐 Fit peaks", type="primary"):
            try:
                fit_df, params = fit_peaks(
                    st.session_state.processed_df, st.session_state.peak_df,
                    profile=profile, window_half_width=window_half_width,
                )
                st.session_state.fit_df = fit_df
                st.session_state.fit_params = params
            except PeakFittingError as exc:
                st.error(str(exc))

        if st.session_state.fit_df is not None:
            n_conv = st.session_state.fit_params["n_converged"]
            n_total = st.session_state.fit_params["n_peaks"]
            if n_conv == n_total:
                st.success(f"Fitted {n_conv}/{n_total} peaks.")
            else:
                st.warning(f"Fitted {n_conv}/{n_total} peaks -- the rest did not converge.")
            if interactive:
                st.plotly_chart(
                    plot_fit_plotly(
                        st.session_state.processed_df, st.session_state.fit_df,
                        st.session_state.fit_params["profile"], st.session_state.fit_params["window_half_width"],
                    ),
                    width="stretch",
                )
            else:
                st.pyplot(
                    plot_fit_static(
                        st.session_state.processed_df, st.session_state.fit_df,
                        st.session_state.fit_params["profile"], st.session_state.fit_params["window_half_width"],
                    )
                )
            st.dataframe(st.session_state.fit_df, width="stretch")


# --- Analysis & Results ------------------------------------------------------------------

with tab_results:
    st.subheader("Analysis summary")
    st.markdown(f"**Source file:** {metadata.filename}  \n**Number of points:** {metadata.n_points}")

    if st.session_state.processing_steps:
        st.markdown("**Processing steps:**")
        for i, step in enumerate(st.session_state.processing_steps, start=1):
            st.markdown(f"{i}. {step}")
    else:
        st.caption("No processing steps applied (analysis on the raw spectrum).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Detected peaks**")
        if st.session_state.peak_df is not None:
            st.dataframe(st.session_state.peak_df, width="stretch")
        else:
            st.caption("None -- run peak detection.")
    with col2:
        st.markdown("**Fit results**")
        if st.session_state.fit_df is not None:
            st.dataframe(st.session_state.fit_df, width="stretch")
        else:
            st.caption("None -- run peak fitting.")


# --- Export -----------------------------------------------------------------------------

with tab_export:
    st.subheader("Export results")
    st.caption("Every export is generated in memory -- nothing is written to the server's disk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Processed spectrum (CSV)",
            data=dataframe_to_csv_bytes(st.session_state.processed_df),
            file_name=f"{metadata.filename}_processed.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        if st.session_state.peak_df is not None:
            st.download_button(
                "⬇️ Peak table (CSV)",
                data=dataframe_to_csv_bytes(st.session_state.peak_df),
                file_name=f"{metadata.filename}_peaks.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.caption("No peak table to export.")
    with col3:
        if st.session_state.fit_df is not None:
            st.download_button(
                "⬇️ Fit results (CSV)",
                data=dataframe_to_csv_bytes(st.session_state.fit_df),
                file_name=f"{metadata.filename}_fit_results.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.caption("No fit results to export.")

    xlsx_sheets = {"Spectrum": st.session_state.processed_df}
    if st.session_state.peak_df is not None and len(st.session_state.peak_df) > 0:
        xlsx_sheets["Peaks"] = st.session_state.peak_df
    if st.session_state.fit_df is not None and len(st.session_state.fit_df) > 0:
        xlsx_sheets["Fit"] = st.session_state.fit_df
    st.download_button(
        "⬇️ Everything in one file (XLSX)",
        data=dataframe_to_xlsx_bytes(xlsx_sheets),
        file_name=f"{metadata.filename}_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    st.divider()
    summary = build_summary_report(
        source_filename=metadata.filename,
        n_points=metadata.n_points,
        processing_steps=st.session_state.processing_steps,
        peak_df=st.session_state.peak_df,
        fit_df=st.session_state.fit_df,
        fit_profile=st.session_state.fit_params["profile"] if st.session_state.fit_params else None,
    )
    st.download_button(
        "📄 Full analysis report (TXT)",
        data=summary.encode("utf-8"),
        file_name=f"{metadata.filename}_report.txt",
        mime="text/plain",
        width="stretch",
    )
    with st.expander("Report preview"):
        st.text(summary)

    # --- batch export: same settings applied to every uploaded file ---------------
    if len(st.session_state.uploaded_spectra) > 1:
        st.divider()
        st.subheader("📦 Batch export (all uploaded files)")
        st.caption(
            f"Applies the current processing, detection and fitting settings "
            f"(as in the tabs above, for the active spectrum) to all "
            f"{len(st.session_state.uploaded_spectra)} uploaded files and packages the results "
            f"(XLSX + TXT report per file) into a single ZIP archive."
        )
        if st.button("📦 Run batch processing and download ZIP"):
            baseline_method = st.session_state.get("baseline_method", "None")
            baseline_params = (
                {"n_points": st.session_state.get("baseline_n_points", BASELINE_DEFAULTS["linear_n_points"])}
                if baseline_method == "Linear (Endpoints)"
                else {
                    "lam": st.session_state.get("als_lambda", BASELINE_DEFAULTS["als_lambda"]),
                    "p": st.session_state.get("als_p", BASELINE_DEFAULTS["als_p"]),
                    "n_iter": int(st.session_state.get("als_n_iter", BASELINE_DEFAULTS["als_n_iter"])),
                }
            )
            batch_settings = {
                "baseline_method": baseline_method if baseline_method != "None" else None,
                "baseline_params": baseline_params,
                "normalization_method": (
                    st.session_state.get("norm_method")
                    if st.session_state.get("norm_method", "None") != "None"
                    else None
                ),
                "smoothing_params": {
                    "window_length": st.session_state.get(
                        "smooth_window_length", SMOOTHING_DEFAULTS["window_length"]
                    ),
                    "polyorder": st.session_state.get("smooth_polyorder", SMOOTHING_DEFAULTS["polyorder"]),
                },
                "peak_detection_params": {
                    "prominence_pct": st.session_state.get(
                        "pk_prominence_pct", PEAK_DETECTION_DEFAULTS["prominence_pct"]
                    ),
                    "min_distance": st.session_state.get(
                        "pk_min_distance", PEAK_DETECTION_DEFAULTS["min_distance"]
                    ),
                },
                "fit_profile": st.session_state.get("fit_profile_select"),
                "fit_window_half_width": st.session_state.get(
                    "fit_window_half_width", PEAK_FITTING_DEFAULTS["window_half_width"]
                ),
            }
            with st.spinner("Processing all files..."):
                batch_results = {}
                batch_errors = {}
                for filename, (df, meta) in st.session_state.uploaded_spectra.items():
                    result = run_pipeline(df, batch_settings)
                    batch_results[filename] = result
                    if result["errors"]:
                        batch_errors[filename] = result["errors"]
                zip_bytes = build_batch_zip(batch_results)
            st.session_state["_batch_zip"] = zip_bytes
            if batch_errors:
                for filename, errs in batch_errors.items():
                    for err in errs:
                        st.warning(f"{filename}: {err}")
            st.success(f"Processed {len(batch_results)} files.")

        if st.session_state.get("_batch_zip"):
            st.download_button(
                "⬇️ Download batch results (ZIP)",
                data=st.session_state["_batch_zip"],
                file_name="raman_batch_results.zip",
                mime="application/zip",
                width="stretch",
            )
