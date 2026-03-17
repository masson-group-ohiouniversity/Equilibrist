# -*- coding: utf-8 -*-
"""
app.py
------
Equilibrist – Chemically rigorous equilibrium and kinetics simulator.
Entry point for Streamlit. Imports all backend modules.
"""
import sys
# ── Windows UTF-8 fix ──────────────────────────────────────────────────────
# On Windows, Python defaults stdout/stderr to cp1252, which cannot encode
# Unicode characters used in sidebar labels (⁻¹, ₀, ±, →, —, etc.).
# Streamlit pipes stdout, so a UnicodeEncodeError mid-sidebar silently kills
# the rest of the render.  Force UTF-8 before any Streamlit call.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# ──────────────────────────────────────────────────────────────────────────
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from equilibrist_parser import *
from equilibrist_network import *
from equilibrist_kinetics import *
from equilibrist_kinetics_nmr import *
from equilibrist_kinetics_spectra import *
from equilibrist_curve import *
from equilibrist_fit_conc import *
from equilibrist_fit_nmr import *
from equilibrist_fit_spectra import *
from equilibrist_io import *


st.set_page_config(page_title="Equilibrist", layout="wide")

# Custom CSS
st.markdown("""
<style>
/* Style for blue snapshot button */
div.stButton > button:has([aria-label*="snapshot"]) {
    background-color: #1f77b4 !important;
    color: white !important;
    border: none !important;
}
div.stButton > button:has([aria-label*="snapshot"]):hover {
    background-color: #1565c0 !important;
    color: white !important;
}
/* Force sidebar subheader to match main panel subheader size */
[data-testid="stSidebar"] h3 {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

def _pub_download_button(plotly_fig, key: str,
                          x_label: str = "", y_label: str = "") -> None:
    """Render a publication-figure download button."""
    from datetime import datetime as _dt
    try:
        xlim, ylim = _pub_axis_range(plotly_fig)
        _ts  = _dt.now().strftime("%Y%m%d_%H%M%S")
        _png = _pub_figure_bytes(plotly_fig, x_label, y_label,
                                 xlim=xlim, ylim=ylim)
        st.download_button(
            "📐 Publication figure (pdf)",
            data=_png,
            file_name=f"{key}_{_ts}.pdf",
            mime="application/pdf",
            key=f"_pubfig_{key}",
        )
    except Exception as _e:
        st.caption(f"Export failed: {_e}")





def _num_input(label, key, default, **kwargs):
    """Number input that seeds from default only on first encounter.
    Also applies any pending override (e.g. _pending_<key>) before rendering."""
    pending_key = f"_pending_{key}"
    shadow_key  = f"_shadow_{key}"
    if pending_key in st.session_state:
        st.session_state[key] = st.session_state.pop(pending_key)
    elif key not in st.session_state:
        # Restore from shadow snapshot if available (survives st.rerun() cleanup)
        if shadow_key in st.session_state:
            st.session_state[key] = st.session_state[shadow_key]
        else:
            st.session_state[key] = default
    return st.number_input(label, key=key, **kwargs)


def _logk_input_with_fit(label, key, default):
    """
    One row: [log K number_input  |  fit checkbox]
    Uses Streamlit's native number_input arrows for +/- increments (step=0.1).
    The fit checkbox is enabled when experimental data is loaded.
    """
    if key not in st.session_state:
        st.session_state[key] = float(default)
    fit_key = f"fit_{key}"
    if fit_key not in st.session_state:
        st.session_state[fit_key] = False

    # Enable fit checkbox if experimental data or NMR data or spectra data is available
    has_exp_data = (bool(st.session_state.get("_exp_data", {})) or
                    bool(st.session_state.get("_nmr_data", {})) or
                    bool(st.session_state.get("_spectra_data", {})))

    c1, c2 = st.columns([0.86, 0.14])
    with c1:
        val = st.number_input(label, key=key, step=0.1, format="%.2f")
    with c2:
        st.checkbox("fit", key=fit_key, disabled=not has_exp_data)
    return float(val)


def _conc_input_with_fit(label, key, default):
    """
    One row: [concentration number_input (mM)  |  fit checkbox]
    Works in mM space. Fit checkbox enabled when experimental data is loaded,
    but disabled when only UV-Vis spectra data is present (concentration is
    not identifiable from spectra with analytical extinction coefficients).
    Respects _pending_{key} set by soft-apply or post-fit updates.
    """
    pending_key = f"_pending_{key}"
    if pending_key in st.session_state:
        st.session_state[key] = float(st.session_state.pop(pending_key))
    elif key not in st.session_state:
        st.session_state[key] = float(default)

    fit_key = f"fit_{key}"
    if fit_key not in st.session_state:
        st.session_state[fit_key] = False

    has_exp_data = (bool(st.session_state.get("_exp_data", {})) or
                    bool(st.session_state.get("_nmr_data", {})) or
                    bool(st.session_state.get("_spectra_data", {})))

    # Concentration fitting from UV-Vis spectra only is not supported:
    # cage0 is unidentifiable when E is solved analytically (A = C@E is
    # scale-invariant). Disable the checkbox in spectra-only mode.
    spectra_only = (bool(st.session_state.get("_spectra_data", {})) and
                    not bool(st.session_state.get("_exp_data", {})) and
                    not bool(st.session_state.get("_nmr_data", {})))
    if spectra_only and st.session_state.get(fit_key, False):
        st.session_state[fit_key] = False   # uncheck silently if was checked

    c1, c2 = st.columns([0.86, 0.14])
    with c1:
        val = st.number_input(label, key=key, step=0.1, format="%.4f", min_value=0.0)
    with c2:
        st.checkbox("fit", key=fit_key, disabled=(not has_exp_data) or spectra_only)
    return float(val)


def _k_input_with_fit(label, key, default_log):
    """
    Like _logk_input_with_fit but displays and accepts LINEAR rate constant values.
    Internally stores log₁₀(k) in session_state[key] for solver compatibility.

    Display format: scientific notation with 1 decimal (e.g. 9.9e+03).
    Step = 0.1 × 10^floor(log10(v)) — correct for going up.
    Going down across a decade boundary the native step would be 10× too
    large (1.0e4 − 1e3 = 9.0e3 instead of 9.9e3).  We fix this by keeping a
    shadow copy of the previous value and correcting lin_key BEFORE the widget
    renders (writing to a widget key before instantiation is always allowed):
        9.7e3 → 9.8e3 → 9.9e3 → 1.0e4 → 1.1e4   (up, no correction needed)
        1.1e4 → 1.0e4 → 9.9e3 → 9.8e3 → 9.7e3   (down, corrected pre-render)
    """
    import math

    # Session state holds log10 value; convert to linear for display seed
    if key not in st.session_state:
        st.session_state[key] = float(default_log)
    fit_key = f"fit_{key}"
    if fit_key not in st.session_state:
        st.session_state[fit_key] = False

    has_exp_data = (bool(st.session_state.get("_exp_data", {})) or
                    bool(st.session_state.get("_nmr_data", {})) or
                    bool(st.session_state.get("_spectra_data", {})))

    # Linear display key (separate from the log-scale storage key)
    lin_key = f"_lin_{key}"
    if lin_key not in st.session_state:
        st.session_state[lin_key] = float(10.0 ** st.session_state[key])

    # Shadow key holds the value from the previous render.
    # If the user just clicked − and crossed a decade boundary, lin_key
    # already holds the wrong value (e.g. 9.0e3).  We detect and correct it
    # here, before st.number_input is called, so the widget sees the right value.
    shadow_key = f"_shdlin_{key}"
    _cur = max(float(st.session_state[lin_key]), 1e-300)
    if shadow_key in st.session_state:
        _prev = max(float(st.session_state[shadow_key]), 1e-300)
        if _cur < _prev:                                  # value decreased
            _exp_cur  = math.floor(math.log10(_cur))
            _exp_prev = math.floor(math.log10(_prev))
            if _exp_cur < _exp_prev:                      # crossed a decade
                # A − button press applies the coarse step = 0.1 × 10^_exp_prev.
                # Only correct if the actual delta matches that step within 10%.
                # A directly typed value will differ by far more (or far less).
                _coarse_step = 0.1 * (10.0 ** _exp_prev)
                _delta = _prev - _cur
                if abs(_delta - _coarse_step) < _coarse_step * 0.1:
                    _fine_step = 0.1 * (10.0 ** (_exp_prev - 1))
                    _cur = _prev - _fine_step             # correct the value
                    st.session_state[lin_key] = _cur      # write before widget
    st.session_state[shadow_key] = _cur                   # update shadow

    _step = 0.1 * (10.0 ** math.floor(math.log10(_cur)))

    c1, c2 = st.columns([0.86, 0.14])
    with c1:
        lin_val = st.number_input(label, key=lin_key,
                                  min_value=0.0, step=_step,
                                  format="%.1e")
    with c2:
        st.checkbox("fit", key=fit_key, disabled=not has_exp_data)

    # Write back as log10 so the solver always sees log-scale
    log_val = math.log10(max(lin_val, 1e-300))
    st.session_state[key] = log_val
    return log_val   # return log10 so caller can use directly



# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

# ── Apply any pending logK values BEFORE any widget is created ──────────
# Writing to a widget-owned key after it renders raises StreamlitAPIException.
# This loop runs before any st.* widget call, so it is always safe.
for _pk in [k for k in st.session_state if k.startswith("_pending_logK_")]:
    _real = _pk[len("_pending_"):]   # "_pending_logK_Kdimer" → "logK_Kdimer"
    st.session_state[_real] = st.session_state.pop(_pk)
    st.session_state.pop(f"_lin_{_real}", None)    # clear stale linear display cache
    st.session_state.pop(f"_shdlin_{_real}", None) # clear stale shadow so correction logic doesn't misfire

# ── Restore fit-checkbox states persisted across kinetics fit rerun ──────
# In kinetics mode the fit block runs BEFORE the sidebar, so Streamlit
# garbage-collects fit_logK_* keys (never rendered in the fit run).
# We snapshot them just before st.rerun() and restore them here.
for _pk in [k for k in st.session_state if k.startswith("_persist_fit_logK_")]:
    _fk = _pk[len("_persist_"):]   # "_persist_fit_logK_k1" → "fit_logK_k1"
    st.session_state[_fk] = st.session_state.pop(_pk)
for _pk in [k for k in st.session_state if k.startswith("_persist_fit_conc_")]:
    _fk = _pk[len("_persist_"):]   # "_persist_fit_conc_G" → "fit_conc_G"
    st.session_state[_fk] = st.session_state.pop(_pk)

# ── Script upload ─────────────────────────────
with st.sidebar:
    st.subheader("Equilibrist Script")

    # A "Clear & reload" button that wipes everything and resets the uploader
    if st.button("↺ Reset / load new script"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # Nonce makes the uploader widget unique each session so re-uploading
    # the same filename is always treated as a new upload by the browser.
    if "_uploader_nonce" not in st.session_state:
        st.session_state["_uploader_nonce"] = 0

    uploaded = st.file_uploader(
        "Upload .txt script", type=["txt"],
        key=f"_uploader_{st.session_state['_uploader_nonce']}",
    )
    if uploaded is not None:
        new_text = uploaded.read().decode("utf-8", errors="replace")
        # Full reset — same as clicking "Reset / load new script"
        # but preserve fit preferences and increment the uploader nonce
        _fit_prefs = {k: st.session_state[k] for k in ("fit_tolerance_log", "fit_timeout")
                      if k in st.session_state}
        _nonce = st.session_state.get("_uploader_nonce", 0)
        st.session_state.clear()
        st.session_state.update(_fit_prefs)
        st.session_state["_uploader_nonce"] = _nonce + 1
        st.session_state["_script_text"] = new_text
        st.session_state["_script_filename"] = uploaded.name
        st.rerun()

    if st.session_state.get("_script_filename"):
        st.markdown(f"<span style='background:#1a6bbf;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:0.82rem'>📄 {st.session_state['_script_filename']}</span>", unsafe_allow_html=True)

script_text = st.session_state.get("_script_text", None)

# ── Skeleton script shown when nothing is loaded yet ─────────────────────
_SKELETON = """\
$concentrations
G0 = 1.0 mM

$volumes
V0 = 500 uL

$titrant
Ht = 10 mM

$reactions
G + H = GH;  log K1 = 4.0

$plot
xmax = 3.0
x = H0/G0
y = G, H, GH\
"""

if script_text is None:
    st.info("📂 Upload a script (.txt) in the sidebar — or write one directly below and click **▶ Apply**.")
    _draft = st.text_area(
        "Script editor",
        value=_SKELETON,
        height=320,
        key="_welcome_editor",
        label_visibility="collapsed",
    )
    _c1, _c2 = st.columns([1, 5])
    with _c1:
        if st.button("▶ Apply", key="_welcome_apply"):
            st.session_state["_script_text"] = _draft
            st.session_state["_script_filename"] = "untitled.txt"
            st.rerun()
    with _c2:
        st.download_button(
            "💾 Save script",
            data=_draft,
            file_name=f"equilibrist_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="_welcome_save",
        )
    st.stop()

# ── Shared script editor widget ───────────────────────────────────────────
# Called at the bottom of both the kinetics and equilibrium branches.
# Shows the original uploaded script in an editable text area with
# Apply and Save buttons.  "Apply" re-parses; if the parameter names are
# unchanged it keeps session state, otherwise it does a full reset.

def _render_script_editor():
    """Editable script expander rendered below the main plot."""
    # Use the uploader nonce in the widget key so that whenever a new script
    # is uploaded (nonce increments), the text_area gets a fresh key and
    # Streamlit cannot restore the previous widget value from its cache.
    _nonce = st.session_state.get("_uploader_nonce", 0)
    with st.expander("📝 Edit script", expanded=False):
        _edited = st.text_area(
            "script_editor_area",
            value=st.session_state.get("_script_text", ""),
            height=320,
            key=f"_script_editor_{_nonce}",
            label_visibility="collapsed",
        )
        _ec1, _ec2, _ec3 = st.columns([2, 2, 3])
        with _ec1:
            _apply = st.button("Apply", icon=":material/play_arrow:",
                               key=f"_script_apply_{_nonce}")
        with _ec2:
            _fname_orig = st.session_state.get("_script_filename", "equilibrist_script.txt")
            _fname_stem = _fname_orig.rsplit(".", 1)[0]  # strip .txt
            _fname = f"{_fname_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button(
                "Save",
                icon=":material/save:",
                data=_edited,
                file_name=_fname,
                mime="text/plain",
                key=f"_script_save_{_nonce}",
            )

        if _apply:
            _new_text = _edited
            # Smart reset: only wipe session state if parameter names changed
            try:
                _new_parsed  = parse_script(_new_text)
                _old_parsed  = parse_script(st.session_state.get("_script_text", ""))
                _new_params  = (
                    {e["kname"] for e in _new_parsed.get("equilibria", [])} |
                    {r["kname"] for r in _new_parsed.get("kinetics", [])} |
                    {r["krname"] for r in _new_parsed.get("kinetics", []) if "krname" in r}
                )
                _old_params  = (
                    {e["kname"] for e in _old_parsed.get("equilibria", [])} |
                    {r["kname"] for r in _old_parsed.get("kinetics", [])} |
                    {r["krname"] for r in _old_parsed.get("kinetics", []) if "krname" in r}
                )
                _params_same = (_new_params == _old_params)
            except Exception:
                _params_same = False

            if _params_same:
                # Soft apply: push new concentrations, volumes, titrant, xmax,
                # AND logK/rate-constant values via _pending_ so all widgets
                # reflect the updated script on the next render.
                st.session_state["_script_text"] = _new_text
                # logK values — covers the case where log/linear prefix changed
                # (e.g. "log k2 = 10.0" → "k2 = 10.0") with the same param name
                for _eq in _new_parsed.get("equilibria", []):
                    st.session_state[f"_pending_logK_{_eq['kname']}"] = float(_eq["logK"])
                for _rxn in _new_parsed.get("kinetics", []):
                    st.session_state[f"_pending_logK_{_rxn['kname']}"] = float(_rxn["log_k"])
                    if "krname" in _rxn:
                        st.session_state[f"_pending_logK_{_rxn['krname']}"] = float(_rxn["log_kr"])
                # Concentrations
                for _cname, _cval in _new_parsed.get("concentrations", {}).items():
                    _root = _cname[:-1] if _cname.endswith("0") else _cname
                    st.session_state[f"_pending_conc_{_root}"] = float(_cval)
                # Volume
                _vols = _new_parsed.get("volumes", {})
                if _vols:
                    st.session_state["_pending_V0_mL"] = float(list(_vols.values())[0])
                # Titrant concentrations
                for _tkey, _tval in _new_parsed.get("titrant", {}).items():
                    st.session_state[f"_pending_titrant_mM_{_tkey}"] = float(_tval)
                # xmax / plot settings
                if _new_parsed.get("plot_xmax") is not None:
                    st.session_state["_pending_xmax"] = float(_new_parsed["plot_xmax"])
            else:
                # Hard reset: new/renamed parameters → rebuild everything
                _fit_prefs = {k: st.session_state[k]
                              for k in ("fit_tolerance_log", "fit_timeout")
                              if k in st.session_state}
                _nonce    = st.session_state.get("_uploader_nonce", 0)
                _fname    = st.session_state.get("_script_filename", "untitled.txt")
                st.session_state.clear()
                st.session_state.update(_fit_prefs)
                st.session_state["_uploader_nonce"]  = _nonce
                st.session_state["_script_text"]     = _new_text
                st.session_state["_script_filename"] = _fname
            st.rerun()


# ── Parse & validate ──────────────────────────
try:
    parsed = parse_script(script_text)
except Exception as e:
    st.warning(f"⚠️ Script error — please revise: {e}")
    st.stop()

# ── Temperature warning (soft parse failure) ──
_temp_warn = parsed.pop("_temperature_warning", None)
if _temp_warn:
    st.warning(f"⚠️ {_temp_warn}")

# ── Syntax check — show all warnings before proceeding ────────────
_syntax_errors = check_script_syntax(script_text)
if _syntax_errors and not st.session_state.get("_override_syntax", False):
    st.warning(
        "⚠️ **Syntax error in Equilibrist script — please revise!**\n\n"
        + "\n\n".join(
            f"**Line {ln}** — `{raw.strip()}`\n> {msg}"
            for ln, raw, msg in _syntax_errors
        )
    )
    if st.button("▶ Override warning and launch anyway"):
        st.session_state["_override_syntax"] = True
        st.rerun()
    st.stop()

if not parsed["equilibria"] and not parsed["kinetics"]:
    st.warning("⚠️ No reactions found — please add a $reactions section to your script.")
    st.stop()

# ── Parser warnings (non-fatal) ───────────────
for _w in parsed.get("warnings", []):
    st.warning(f"⚠️ {_w}")

# ── Detect mode ───────────────────────────────
IS_KINETICS = parsed["is_kinetics"]


# ═══════════════════════════════════════════════════════════════════
# KINETICS BRANCH
# ═══════════════════════════════════════════════════════════════════
if IS_KINETICS:
    all_kin_species = _collect_all_kinetic_species(parsed)
    logk_dict       = build_kinetics_logk_dict(parsed)
    t_max           = float(parsed["plot_xmax"])
    plot_y_names    = parsed["plot_y"] if parsed["plot_y"] else all_kin_species[:6]

    # ── Handle kinetics fitting (before widgets) ──────────────────
    if st.session_state.pop("_fit_requested", False):
        _nmr_data_fit     = st.session_state.get("_nmr_data", {})
        exp_data_fit      = st.session_state.get("_exp_data", {})
        _spectra_data_fit = st.session_state.get("_spectra_data", {})
        _nmr_cfg_fit      = parsed.get("nmr")
        _use_nmr_fit      = (_nmr_cfg_fit is not None and
                             _nmr_cfg_fit["mode"] in ("shift", "integration", "mixed") and
                             bool(_nmr_data_fit))
        _use_spectra_fit  = parsed.get("spectra") is not None and bool(_spectra_data_fit)
        fit_keys_k        = [name for name in logk_dict
                             if st.session_state.get(f"fit_logK_{name}", False)]
        fit_conc_keys_k   = [root for root in
                             (cname[:-1] if cname.endswith("0") else cname
                              for cname in parsed["concentrations"])
                             if st.session_state.get(f"fit_conc_{root}", False)]
        _has_data_fit     = _use_nmr_fit or _use_spectra_fit or bool(exp_data_fit)
        if _has_data_fit and (fit_keys_k or fit_conc_keys_k):
            # Build parsed_kin with current sidebar concentrations from session state
            _fit_concs = {}
            for cname, cval in parsed["concentrations"].items():
                root = cname[:-1] if cname.endswith("0") else cname
                _ckey = f"conc_{root}"
                # Prefer shadow (snapshotted at button click) over live session state
                _fit_concs[cname] = float(
                    st.session_state.get(f"_shadow_{_ckey}",
                    st.session_state.get(_ckey, cval)))
            parsed_fit = dict(parsed)
            parsed_fit["concentrations"] = _fit_concs
            # Use current sidebar t_max, not the script default
            t_max = float(st.session_state.get("_shadow_xmax",
                          st.session_state.get("xmax", t_max)))

            current_logk = {
                name: st.session_state.get(f"_shadow_logK_{name}",
                       st.session_state.get(f"logK_{name}", logk_dict[name]))
                for name in logk_dict
            }
            tol_log    = st.session_state.get("_shadow_fit_tolerance_log",
                         st.session_state.get("fit_tolerance_log", 6.0))
            tol        = 10.0 ** (-float(tol_log))
            _timeout_s = float(st.session_state.get("_shadow_fit_timeout",
                               st.session_state.get("fit_timeout", 30)))
            maxiter    = 100_000
            try:
                # Constraints are only passed if the toggle is active
                _this_fit_constrained = st.session_state.get("fit_use_constraints", False)
                _last_fit_constrained = st.session_state.get("_last_fit_was_constrained", False)
                _active_constraints = (
                    parsed_fit.get("constraints", []) if _this_fit_constrained else []
                )

                # If constraints changed state OR fitting concentrations, reset to
                # script defaults: sidebar logK values may be far from the optimum
                # at the script concentration, causing L-BFGS-B to push the
                # concentration to its boundary before logK can adjust.
                _script_defaults_k = {**{e["kname"]: e["logK"]   for e in parsed_fit["equilibria"]},
                                      **{r["kname"]: r["log_k"]  for r in parsed_fit.get("kinetics", [])},
                                      **{r["krname"]: r["log_kr"] for r in parsed_fit.get("kinetics", []) if "krname" in r}}
                if _this_fit_constrained != _last_fit_constrained:
                    _start_logk = {k: _script_defaults_k.get(k, v) for k, v in current_logk.items()}
                else:
                    _start_logk = current_logk

                st.session_state["_last_fit_was_constrained"] = _this_fit_constrained

                if _use_spectra_fit:
                    wl_min_k = float(st.session_state.get("spectra_wl_min",
                                     _spectra_data_fit["wavelengths"][0]))
                    wl_max_k = float(st.session_state.get("spectra_wl_max",
                                     _spectra_data_fit["wavelengths"][-1]))
                    _auto_range_k   = bool(st.session_state.get("spectra_auto_range", False))
                    _allow_neg_k    = bool(st.session_state.get("spectra_allow_neg", False))
                    with st.spinner("Fitting parameters…"):
                        success, fitted, stats, msg = fit_kinetics_spectra(
                            parsed_fit, _start_logk, _spectra_data_fit, fit_keys_k,
                            t_max, 200, wl_min_k, wl_max_k, tol, maxiter,
                            timeout_s=_timeout_s, auto_range=_auto_range_k, allow_negative_eps=_allow_neg_k,
                            use_lbfgsb=st.session_state.get("fit_use_lbfgsb", True),
                            use_neldermead=st.session_state.get("fit_use_neldermead", True),
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k)
                    if _auto_range_k and "opt_wl_min" in stats:
                        st.session_state["_pending_spectra_wl_min"] = stats["opt_wl_min"]
                        st.session_state["_pending_spectra_wl_max"] = stats["opt_wl_max"]
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "shift":
                    with st.spinner("Fitting parameters…"):
                        success, fitted, stats, msg = fit_kinetics_nmr_shifts(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k)
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "integration":
                    with st.spinner("Fitting parameters…"):
                        success, fitted, stats, msg = fit_kinetics_nmr_integration(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k)
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "mixed":
                    with st.spinner("Fitting parameters…"):
                        success, fitted, stats, msg = fit_kinetics_nmr_mixed(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k)
                else:
                    with st.spinner("Fitting parameters…"):
                        success, fitted, stats, msg = fit_kinetics(
                            parsed_fit, exp_data_fit, _start_logk, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k)
                for name, val in fitted.items():
                    st.session_state[f"_pending_logK_{name}"] = float(val)
                # Push fitted concentration results back to sidebar
                for _root, _mM_val in stats.get("fitted_concs", {}).items():
                    st.session_state[f"_pending_conc_{_root}"] = float(_mM_val)
                st.session_state["_fit_stats"] = stats
                _n_evals = stats.get("n_iter", 0)
                if stats.get("timed_out"):
                    st.session_state["_fit_message"] = ("warning",
                        f"⏱️ Fit timed out after {_n_evals} evaluations ({_timeout_s:.0f} s limit) — "
                        "showing best result found. Consider increasing Timeout.")
                elif success:
                    _n_fitted_k = len(fitted) + len(stats.get("fitted_concs", {}))
                    st.session_state["_fit_message"] = ("success",
                        f"Fit completed! Updated {_n_fitted_k} parameters. "
                        f"({_n_evals} evaluations, tol=1e-{tol_log:.0f})")
                else:
                    st.session_state["_fit_message"] = ("warning",
                        f"Fit did not fully converge — {msg}")
            except Exception as _fit_exc:
                import traceback as _tb
                st.session_state["_fit_message"] = ("warning",
                    f"Fitting error: {_fit_exc}\n{_tb.format_exc()}")
            # ── Persist fit-checkbox states across rerun ──────────────────────
            # The sidebar (which renders fit_logK_* widgets) runs AFTER this
            # block, so Streamlit GC will remove unrendered fit_logK_* keys.
            # Snapshot them now; the top-of-script loop restores them on Run B.
            for _fname in list(logk_dict.keys()):
                _fk = f"fit_logK_{_fname}"
                if _fk in st.session_state:
                    st.session_state[f"_persist_{_fk}"] = st.session_state[_fk]
            # Persist concentration fit-checkbox states across rerun
            for _cname in parsed.get("concentrations", {}):
                _cr = _cname[:-1] if _cname.endswith("0") else _cname
                _cfk = f"fit_conc_{_cr}"
                if _cfk in st.session_state:
                    st.session_state[f"_persist_{_cfk}"] = st.session_state[_cfk]
            st.rerun()

    # ── Apply pending fitted values ──────────────────────────────
    for _k in list(st.session_state.keys()):
        if _k.startswith("_pending_logK_"):
            _kname = _k[len("_pending_logK_"):]
            st.session_state[f"logK_{_kname}"] = st.session_state.pop(_k)
            st.session_state.pop(f"_lin_logK_{_kname}", None)
            st.session_state.pop(f"_shdlin_logK_{_kname}", None)

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("Initial concentrations (mM)")
        conc_vals_kin = {}
        for cname, cval in parsed["concentrations"].items():
            root = cname[:-1] if cname.endswith("0") else cname
            conc_vals_kin[root] = _conc_input_with_fit(
                f"{root}₀", key=f"conc_{root}", default=float(cval))

        st.header("Plot settings")
        t_max_ui = _num_input("Time max (s)", key="xmax",
                               default=t_max, step=0.1, format="%.4f")
        nPts_kin = st.number_input("# points", value=200, step=10, min_value=20)

        # Wavelength range (only shown once spectra data is loaded — mirrors equilibrium mode)
        _kin_sp_wl_sb = st.session_state.get("_spectra_data", {})
        if parsed.get("spectra") is not None and _kin_sp_wl_sb:
            _wl_lo_k = float(_kin_sp_wl_sb["wavelengths"][0])
            _wl_hi_k = float(_kin_sp_wl_sb["wavelengths"][-1])
            st.caption("Wavelength range (nm)")
            _wl_c1k, _wl_c2k = st.columns(2)
            with _wl_c1k:
                _num_input("Min", key="spectra_wl_min", default=_wl_lo_k, step=1.0, format="%.0f")
            with _wl_c2k:
                _num_input("Max", key="spectra_wl_max", default=_wl_hi_k, step=1.0, format="%.0f")
            st.checkbox("Auto-optimize range", key="spectra_auto_range")
            st.checkbox("Allow negative absorbances", key="spectra_allow_neg")

        st.header("Rate constants")
        logk_ui = {}
        for rxn in parsed["kinetics"]:
            kn   = rxn["kname"]
            lbl  = _kinetics_reaction_label(rxn)
            n_r  = sum(c for c, _ in rxn["reactants"])
            n_p  = sum(c for c, _ in rxn["products"])
            u_fwd = _rate_constant_units(n_r)
            st.caption(f"**{lbl}**  —  {kn}: {u_fwd}")
            logk_ui[kn] = _k_input_with_fit(
                kn, key=f"logK_{kn}", default_log=logk_dict[kn])
            if "krname" in rxn:
                krn   = rxn["krname"]
                u_rev = _rate_constant_units(n_p, is_reverse=True, n_products=n_p)
                st.caption(f"{krn}: {u_rev}")
                logk_ui[krn] = _k_input_with_fit(
                    krn, key=f"logK_{krn}", default_log=logk_dict[krn])

        if parsed["equilibria"]:
            st.header("Pre-equilibria (= reactions)")
            for eq in parsed["equilibria"]:
                kn  = eq["kname"]
                n_r = sum(c for c, _ in eq["reactants"])
                n_p = sum(c for c, _ in eq["products"])
                lbl = _kinetics_reaction_label({**eq, "type": "equilibrium"})
                units = _equilibrium_constant_units(n_r, n_p)
                st.caption(f"**{lbl}**  —  {kn}: {units}")
                logk_ui[kn] = _logk_input_with_fit(
                    f"log {kn}", key=f"logK_{kn}", default=eq["logK"])

        # Merge UI values into logk_dict
        logk_dict.update(logk_ui)

        # ── Constraints toggle (only shown when $constraints section present) ──
        if parsed.get("constraints"):
            st.checkbox(
                "Activate constraints",
                key="fit_use_constraints",
                value=False,
                help=f"{len(parsed['constraints'])} constraint(s) defined in $constraints section.",
            )

    # ── Simulate ─────────────────────────────────────────────────
    # Patch initial concentrations from sidebar into parsed copy
    parsed_kin = dict(parsed)
    parsed_kin["concentrations"] = {
        (k + "0" if not k.endswith("0") else k): v
        for k, v in conc_vals_kin.items()
    }

    with st.spinner("Integrating ODEs…"):
        try:
            kin_curve = compute_kinetics_curve(parsed_kin, logk_dict, t_max_ui, int(nPts_kin))
        except Exception as e:
            st.error(f"ODE solver error: {e}")
            st.stop()

    t_vals = kin_curve["t"]

    # ── Plot ─────────────────────────────────────────────────────
    COLORS = ["#636EFA","#EF553B","#00CC96","#AB63FA",
              "#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]

    col1, col2 = st.columns([2.2, 1.0], gap="large")
    with col1:
        fig = go.Figure()
        trace_colors = {}
        variables = parsed.get("variables", {})
        for idx, sp in enumerate(plot_y_names):
            color = COLORS[idx % len(COLORS)]
            trace_colors[sp] = color
            if sp in variables:
                # Evaluate variable expression
                ns = {s: kin_curve.get(s, np.zeros_like(t_vals)) for s in all_kin_species}
                try:
                    y_vals = np.array(eval(variables[sp], {"__builtins__": {}, "np": np}, ns))
                except Exception:
                    y_vals = np.zeros_like(t_vals)
            else:
                y_vals = kin_curve.get(sp, np.zeros_like(t_vals))
            fig.add_trace(go.Scatter(
                x=t_vals, y=y_vals, mode="lines", name=sp,
                line=dict(color=color, width=2)))

        # Experimental overlay (column A = time in seconds)
        exp_data = st.session_state.get("_exp_data", {})
        for col_name, col_data in exp_data.items():
            if col_name.startswith("_"):
                continue
            color = trace_colors.get(col_name, "#FFFFFF")
            fig.add_trace(go.Scatter(
                x=col_data["v_add_mL"],   # time in seconds
                y=col_data["y"],
                mode="markers", name=f"{col_name} (exp)",
                marker=dict(color=color, size=7, symbol="circle",
                            line=dict(width=1, color="white")),
                showlegend=True,
            ))

        # ── NMR back-calculated concentrations on main plot ─────────────────
        _kin_fit_stats   = st.session_state.get("_fit_stats", {})
        _kin_nmr_data    = st.session_state.get("_nmr_data", {})
        _kin_sp_concs    = _kin_fit_stats.get("sp_concs", {})
        _kin_nmr_cfg_plt = parsed.get("nmr")
        _kin_all_sp_plt  = _collect_all_kinetic_species(parsed)

        # Integration / mixed — post-fit: sp_concs from fit stats
        if _kin_sp_concs and _kin_nmr_data:
            _bc_kin_int = {}
            for sp, arr_list in _kin_sp_concs.items():
                t_ref = arr_list[0][0]
                c_avg = np.mean([np.interp(t_ref, ta, ca) for ta, ca in arr_list], axis=0)
                _bc_kin_int[sp] = (t_ref, c_avg)
            _kin_sfx_integ = "(NMR/integration)" if _kin_nmr_cfg_plt and _kin_nmr_cfg_plt.get("mode") == "mixed" else "(NMR)"
            _plot_backcalc_dots(fig, _bc_kin_int, plot_y_names,
                                parsed.get("variables", {}),
                                _kin_all_sp_plt, trace_colors,
                                label_suffix=_kin_sfx_integ)

        # Integration / mixed — pre-fit: back-calc from raw integrals (no k needed)
        elif (_kin_nmr_cfg_plt is not None and
              _kin_nmr_cfg_plt["mode"] in ("integration", "mixed") and
              _kin_nmr_data and not _kin_sp_concs):
            _nmr_n_H_pre  = _kin_nmr_cfg_plt.get("n_H_list", [])
            _n_integ_pre  = _kin_nmr_cfg_plt.get("n_integ", len(_nmr_n_H_pre))
            _all_cols_pre = [c for c in _kin_nmr_data if not c.startswith("_")]
            _integ_cols_p = _all_cols_pre[:_n_integ_pre]
            _integ_data_p = {c: _kin_nmr_data[c] for c in _integ_cols_p}
            _bc_pre_kin   = _kinetics_nmr_integration_backCalc(
                _integ_data_p, _nmr_n_H_pre[:_n_integ_pre], parsed, _kin_all_sp_plt)
            _kin_sfx_pre = "(NMR/integration)" if _kin_nmr_cfg_plt and _kin_nmr_cfg_plt.get("mode") == "mixed" else "(NMR)"
            _plot_backcalc_dots(fig, _bc_pre_kin, plot_y_names,
                                parsed.get("variables", {}),
                                _kin_all_sp_plt, trace_colors,
                                label_suffix=_kin_sfx_pre)

        # Shift mode: invert M-matrix to back-calculate concentrations from Δδ
        # (same algorithm as equilibrium branch)
        # NOTE: in mixed mode, integration back-calc already covers species on main
        # plot — skip the shift M-matrix overlay to avoid duplicates.
        _kin_dv_all  = _kin_fit_stats.get("delta_vecs_all", {})
        _kin_df_all  = _kin_fit_stats.get("delta_free", {})
        _kin_t_free  = _kin_fit_stats.get("x_free_val", {})
        _kin_c2tgt   = _kin_fit_stats.get("col_to_target", {})
        _kin_ref_cor = _kin_fit_stats.get("ref_corrections", {})
        _kin_nmr_cfg_plt = parsed.get("nmr")
        if (_kin_nmr_cfg_plt is not None and
                _kin_nmr_cfg_plt["mode"] in ("shift", "mixed") and
                _kin_nmr_data and _kin_dv_all):
            _kin_all_sp_plt  = _collect_all_kinetic_species(parsed)
            _kin_fake_net_plt = {"all_species": _kin_all_sp_plt}
            _fitted_cols_plt = [col for col in _kin_nmr_data
                                if not col.startswith("_") and col in _kin_dv_all]
            if _fitted_cols_plt:
                # Use the time points from the first fitted signal as common grid
                _t_bc = _kin_nmr_data[_fitted_cols_plt[0]]["v_add_mL"]
                _n_pts_bc = len(_t_bc)
                # Unique non-free species across all targets
                _nf_sp = []
                for _tgt_bc in _kin_nmr_cfg_plt["targets"]:
                    for _, _sp in _get_species_for_target(_tgt_bc, parsed, _kin_fake_net_plt)[1:]:
                        if _sp not in _nf_sp:
                            _nf_sp.append(_sp)
                _n_nf = len(_nf_sp); _n_sig_bc = len(_fitted_cols_plt)
                M_bc  = np.zeros((_n_sig_bc, _n_nf))
                rhs_bc = np.zeros((_n_sig_bc, _n_pts_bc))
                for k_bc, col_bc in enumerate(_fitted_cols_plt):
                    _tgt_bc   = _kin_c2tgt_plt = _kin_c2tgt.get(col_bc)
                    if _tgt_bc is None: continue
                    _spc_bc   = _get_species_for_target(_tgt_bc, parsed, _kin_fake_net_plt)
                    _sp_dd_bc = _kin_dv_all[col_bc]
                    _ref_c_bc = _kin_ref_cor.get(col_bc, 0.0)
                    for coeff_bc, sp_bc in _spc_bc[1:]:
                        if sp_bc in _nf_sp:
                            M_bc[k_bc, _nf_sp.index(sp_bc)] = coeff_bc * _sp_dd_bc.get(sp_bc, 0.0)
                    # Ctot at the experimental time points (from theoretical curve)
                    _ctot_bc = np.maximum(
                        sum(coeff_bc * np.interp(_t_bc, t_vals, kin_curve.get(sp_bc, np.zeros_like(t_vals)))
                            for coeff_bc, sp_bc in _spc_bc), 1e-20)
                    _df0_bc   = float(_kin_df_all.get(col_bc, _kin_nmr_data[col_bc]["y"][0]))
                    _t_col_bc = _kin_nmr_data[col_bc]["v_add_mL"]
                    _dobs_bc  = np.interp(_t_bc, _t_col_bc, _kin_nmr_data[col_bc]["y"] - _df0_bc)
                    rhs_bc[k_bc, :] = (_dobs_bc + _ref_c_bc) * _ctot_bc
                # Solve M @ c_nonfree(t) = rhs(t) at each point
                _rank_bc = np.linalg.matrix_rank(M_bc)
                _c_nf_bc = np.zeros((_n_nf, _n_pts_bc))
                if _n_nf > 0 and _rank_bc >= min(_n_sig_bc, _n_nf):
                    for j_bc in range(_n_pts_bc):
                        _sol = np.linalg.lstsq(M_bc, rhs_bc[:, j_bc], rcond=None)[0]
                        _c_nf_bc[:, j_bc] = np.clip(_sol, 0.0, None)
                # Back-calc free species from mass balance, then plot via helper
                _bc_kin_shift = {}
                for _tgt_bc in _kin_nmr_cfg_plt["targets"]:
                    _spc_bc = _get_species_for_target(_tgt_bc, parsed, _kin_fake_net_plt)
                    if not _spc_bc: continue
                    _free_sp_bc = _spc_bc[0][1]
                    _ctot_uw = np.maximum(
                        sum(np.interp(_t_bc, t_vals, kin_curve.get(sp_bc, np.zeros_like(t_vals)))
                            for _, sp_bc in _spc_bc), 1e-20)
                    _sum_nf = np.zeros(_n_pts_bc)
                    for _, sp_bc in _spc_bc[1:]:
                        if sp_bc in _nf_sp:
                            _sum_nf += _c_nf_bc[_nf_sp.index(sp_bc), :]
                    _c_free_bc = np.clip(_ctot_uw - _sum_nf, 0.0, None)
                    for _, sp_bc in _spc_bc:
                        if sp_bc in _bc_kin_shift: continue
                        _c_arr = _c_free_bc if sp_bc == _free_sp_bc else (
                            _c_nf_bc[_nf_sp.index(sp_bc), :] if sp_bc in _nf_sp else None)
                        if _c_arr is None: continue
                        _bc_kin_shift[sp_bc] = (_t_bc, _c_arr)
                _kin_sfx_shift = "(NMR/shift)" if _kin_nmr_cfg_plt and _kin_nmr_cfg_plt.get("mode") == "mixed" else "(NMR)"
                _plot_backcalc_dots(fig, _bc_kin_shift, plot_y_names,
                                    parsed.get("variables", {}),
                                    _collect_all_kinetic_species(parsed), trace_colors,
                                    label_suffix=_kin_sfx_shift)

        # ── UV-Vis back-calc dots on kinetics main plot ──────────────────────────
        _kin_fit_stats_sp = st.session_state.get("_fit_stats", {})
        if _kin_fit_stats_sp.get("fit_mode") == "kinetics_spectra":
            _kin_absorbers = _kin_fit_stats_sp.get("absorbers", [])
            _kin_t_exp_sp  = _kin_fit_stats_sp.get("x_exp", np.array([]))
            _kin_C_back    = _kin_fit_stats_sp.get("C_back", None)
            if _kin_C_back is not None and len(_kin_t_exp_sp) == _kin_C_back.shape[0]:
                _bc_kin_sp = {sp: (_kin_t_exp_sp, _kin_C_back[:, j])
                              for j, sp in enumerate(_kin_absorbers)}
                _plot_backcalc_dots(fig, _bc_kin_sp, plot_y_names,
                                    parsed.get("variables", {}),
                                    all_kin_species, trace_colors,
                                    label_suffix="(UV-Vis)")

        if not kin_curve.get("success", True):
            st.warning("⚠️ ODE integrator did not fully converge.")

        fig.update_layout(
            height=700,
            margin=dict(l=40, r=20, t=40, b=120),
            xaxis=dict(title="Time [s]"),
            yaxis=dict(title=_infer_y_label(plot_y_names,
                                            parsed,
                                            {"all_species": _collect_all_kinetic_species(parsed)}),
                       rangemode="tozero"),
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width='stretch')
        _pub_download_button(fig, "kinetics_main")
        st.session_state["_current_figure"] = fig
        st.session_state.pop("_kin_snapshot_data", None)

        # ── NMR chemical shift plot ──────────────────────────────────────────
        _kin_nmr_cfg = parsed.get("nmr")
        if _kin_nmr_cfg is not None and _kin_nmr_cfg["mode"] in ("shift", "mixed") and _kin_nmr_data:
            _kin_delta_free    = _kin_fit_stats.get("delta_free", {})
            _kin_delta_vecs    = _kin_fit_stats.get("delta_vecs_all", {})
            _kin_t_free_val    = _kin_fit_stats.get("x_free_val", {})
            _kin_col_to_tgt    = _kin_fit_stats.get("col_to_target", {})
            _kin_all_sp        = _collect_all_kinetic_species(parsed)
            _kin_fake_net      = {"all_species": _kin_all_sp}

            fig_nmr_k = go.Figure()
            _NMR_PAL  = ["#5B9BD5", "#ED7D31", "#70AD47", "#FF5C5C", "#9966CC", "#00B0CC"]
            _kin_tgt_colors = {tgt: _NMR_PAL[i % len(_NMR_PAL)]
                               for i, tgt in enumerate(_kin_nmr_cfg["targets"])}
            _n_integ_k   = _kin_nmr_cfg.get("n_integ", 0)
            _all_cols_k  = [c for c in _kin_nmr_data if not c.startswith("_")]
            _shift_cols_k = _all_cols_k[_n_integ_k:]

            # Build col→target for shift columns
            _kin_c2t = {}
            for _col in _shift_cols_k:
                for _tgt in _kin_nmr_cfg["targets"]:
                    if _col == _tgt or _col.startswith(_tgt + ".") or _col.startswith(_tgt + "_"):
                        _kin_c2t[_col] = _tgt; break
                else:
                    _kin_c2t[_col] = _kin_nmr_cfg["targets"][0] if _kin_nmr_cfg["targets"] else _col

            _shown_tgt = set()
            for _col in _shift_cols_k:
                _col_data = _kin_nmr_data[_col]
                _t_exp    = _col_data["v_add_mL"]
                _df0      = float(_kin_delta_free.get(_col, _col_data["y"][0]))
                _dobs_rel = _col_data["y"] - _df0
                _tgt      = _kin_c2t.get(_col, _col)
                _color    = _kin_tgt_colors.get(_tgt, "#888888")
                _show_leg = _tgt not in _shown_tgt; _shown_tgt.add(_tgt)
                fig_nmr_k.add_trace(go.Scatter(
                    x=_t_exp, y=_dobs_rel, mode="markers",
                    name=_tgt if _show_leg else _col,
                    legendgroup=_tgt, showlegend=_show_leg,
                    marker=dict(color=_color, size=6, symbol="circle"),
                ))

            # Theoretical Δδ curves — only shown after a fit has been run
            _kin_nmr_fitted = bool(_kin_delta_vecs)
            if _kin_nmr_fitted:
                for _col in _shift_cols_k:
                    _col_data  = _kin_nmr_data[_col]
                    _tgt       = _kin_c2t.get(_col, _col)
                    _sp_coeffs = _get_species_for_target(_tgt, parsed, _kin_fake_net)
                    if not _sp_coeffs: continue
                    _df0       = float(_kin_delta_free.get(_col, _col_data["y"][0]))
                    _t_free_c  = _kin_t_free_val.get(_col, 0.0)
                    _t_sim     = t_vals

                    _denom_full = np.maximum(
                        sum(coeff * kin_curve.get(sp, np.zeros_like(_t_sim))
                            for coeff, sp in _sp_coeffs), 1e-20)
                    _denom_ref  = float(np.maximum(
                        sum(coeff * float(np.interp(_t_free_c, _t_sim,
                                          kin_curve.get(sp, np.zeros_like(_t_sim))))
                            for coeff, sp in _sp_coeffs), 1e-20))
                    _non_free  = _sp_coeffs[1:]

                    if _col not in _kin_delta_vecs:
                        continue  # post-fit only
                    _sp_dd = _kin_delta_vecs[_col]
                    _calc_rel = np.zeros_like(_t_sim)
                    for coeff, sp in _non_free:
                        _F_full = coeff * kin_curve.get(sp, np.zeros_like(_t_sim)) / _denom_full
                        _F_ref  = coeff * float(np.interp(_t_free_c, _t_sim,
                                      kin_curve.get(sp, np.zeros_like(_t_sim)))) / _denom_ref
                        _calc_rel += (_F_full - _F_ref) * _sp_dd.get(sp, 0.0)

                    fig_nmr_k.add_trace(go.Scatter(
                        x=_t_sim, y=_calc_rel, mode="lines",
                        name=f"{_col} (calc)", legendgroup=_tgt, showlegend=False,
                        line=dict(color=_kin_tgt_colors.get(_tgt, "#888888"), width=2),
                    ))

            fig_nmr_k.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=40, b=80),
                xaxis=dict(title="Time [s]"),
                yaxis=dict(title="Δδ [ppm]"),
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.2,
                            xanchor="center", x=0.5),
                title=dict(text="NMR Chemical Shift Changes", x=0.5,
                           font=dict(size=13)),
            )
            st.plotly_chart(fig_nmr_k, width='stretch')
            _pub_download_button(fig_nmr_k, "kinetics_nmr", y_label="Δδ [ppm]")

        # ── UV-Vis spectra subplot (inside col1) ──────────────────────────────
        _kin_sd_plot = st.session_state.get("_spectra_data", {})
        if parsed.get("spectra") is not None and _kin_sd_plot:
            _kin_wl_all  = _kin_sd_plot["wavelengths"]
            _kin_t_raw   = _kin_sd_plot["x_vals"]
            _kin_A_sp    = _kin_sd_plot["A"]
            _kin_n_sp    = len(_kin_t_raw)
            _kin_wl_lo_p = float(st.session_state.get("spectra_wl_min", _kin_wl_all[0]))
            _kin_wl_hi_p = float(st.session_state.get("spectra_wl_max", _kin_wl_all[-1]))
            _kin_wl_mask = (_kin_wl_all >= _kin_wl_lo_p) & (_kin_wl_all <= _kin_wl_hi_p)
            _kin_wl_plot = _kin_wl_all[_kin_wl_mask]
            _kin_A_plot  = _kin_A_sp[:, _kin_wl_mask]

            import colorsys as _cs
            def _kin_rainbow(i, n):
                hue = (270 - (270 * i / max(n - 1, 1))) / 360.0
                r, g, b = _cs.hls_to_rgb(hue, 0.55, 0.85)
                return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            fig_kin_sp = go.Figure()
            for _i in range(_kin_n_sp):
                _col_c = _kin_rainbow(_i, _kin_n_sp)
                _lbl = f"t={_kin_t_raw[_i]:.3g} s"
                fig_kin_sp.add_trace(go.Scatter(
                    x=_kin_wl_plot, y=_kin_A_plot[_i], mode="lines",
                    line=dict(color=_col_c, width=1.5), name=_lbl, showlegend=False,
                    hovertemplate=f"{_lbl}<br>λ=%{{x:.0f}} nm<br>A=%{{y:.4f}}<extra></extra>",
                ))
            fig_kin_sp.add_annotation(x=0.01, y=1.02, xref="paper", yref="paper",
                                      text=f"t={_kin_t_raw[0]:.3g} s", showarrow=False,
                                      font=dict(color=_kin_rainbow(0, _kin_n_sp), size=11),
                                      xanchor="left")
            fig_kin_sp.add_annotation(x=0.99, y=1.02, xref="paper", yref="paper",
                                      text=f"t={_kin_t_raw[-1]:.3g} s", showarrow=False,
                                      font=dict(color=_kin_rainbow(_kin_n_sp-1, _kin_n_sp), size=11),
                                      xanchor="right")
            fig_kin_sp.update_layout(
                height=350, margin=dict(l=40, r=20, t=40, b=60),
                xaxis=dict(title="Wavelength [nm]"),
                yaxis=dict(title="Absorbance", rangemode="tozero"),
                template="plotly_dark", showlegend=False,
                title=dict(text="UV-Vis spectra (kinetics)", font=dict(size=13), x=0.5),
            )
            st.plotly_chart(fig_kin_sp, width='stretch')
            _pub_download_button(fig_kin_sp, "kinetics_spectra",
                                 x_label="Wavelength [nm]", y_label="Absorbance")

            # Pure-species spectra (only after fit)
            _kin_fit_sp = st.session_state.get("_fit_stats", {})
            if _kin_fit_sp.get("fit_mode") == "kinetics_spectra":
                _kin_E      = _kin_fit_sp.get("E_final")
                _kin_wl_fit = _kin_fit_sp.get("wavelengths_fit")
                _kin_abs    = _kin_fit_sp.get("absorbers", [])
                if _kin_E is not None and len(_kin_abs) > 0:
                    _PALETTE = ["#636EFA","#EF553B","#00CC96","#AB63FA",
                                "#FFA15A","#19D3F3","#FF6692","#B6E880"]
                    fig_kin_pure = go.Figure()
                    for _j, _sp in enumerate(_kin_abs):
                        fig_kin_pure.add_trace(go.Scatter(
                            x=_kin_wl_fit, y=_kin_E[_j], mode="lines",
                            line=dict(color=_PALETTE[_j % len(_PALETTE)], width=2),
                            name=_sp,
                            hovertemplate=f"{_sp}<br>λ=%{{x:.0f}} nm<br>ε=%{{y:.4f}} mM⁻¹<extra></extra>",
                        ))
                    fig_kin_pure.update_layout(
                        height=350, margin=dict(l=40, r=20, t=40, b=60),
                        xaxis=dict(title="Wavelength [nm]"),
                        yaxis=dict(title="ε [mM⁻¹, path length absorbed]", rangemode="tozero"),
                        template="plotly_dark", showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        title=dict(text="Pure species spectra", font=dict(size=13), x=0.5),
                    )
                    st.plotly_chart(fig_kin_pure, width='stretch')
                    _pub_download_button(fig_kin_pure, "kinetics_spectra_species",
                                         x_label="Wavelength [nm]", y_label="ε [mM⁻¹]")

    with col2:
        # ── Fit message ──────────────────────────────────────────
        _fit_msg = st.session_state.pop("_fit_message", None)
        if _fit_msg:
            level, text = _fit_msg
            if level == "success":
                st.success(text)
            else:
                st.warning(text)

        # ── Fitting controls (identical layout to equilibrium branch) ──
        exp_data      = st.session_state.get("_exp_data", {})
        _nmr_data_k     = st.session_state.get("_nmr_data", {})
        _spectra_data_k = st.session_state.get("_spectra_data", {})
        has_exp         = bool(exp_data) or bool(_nmr_data_k) or bool(_spectra_data_k)
        fit_keys_k    = [n for n in logk_dict
                         if st.session_state.get(f"fit_logK_{n}", False)]
        fit_conc_keys_k_btn = [root for root in
                             (cname[:-1] if cname.endswith("0") else cname
                              for cname in parsed["concentrations"])
                             if st.session_state.get(f"fit_conc_{root}", False)]
        fit_ok        = has_exp and (len(fit_keys_k) > 0 or len(fit_conc_keys_k_btn) > 0)

        # Render Tol/Timeout BEFORE the button so they are always rendered
        # before any st.rerun() call
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            fit_tolerance_log = _num_input(
                "Tol (−log)", key="fit_tolerance_log", default=6.0, step=0.5, format="%.1f"
            )
        with _fc2:
            fit_timeout = _num_input(
                "Timeout (s)", key="fit_timeout", default=30, step=5, format="%d"
            )

        # ── Optimizer selection ──────────────────────────────────
        _oc1, _oc2 = st.columns(2)
        with _oc1:
            use_lbfgsb_k = st.checkbox("L-BFGS-B", key="fit_use_lbfgsb", value=True)
        with _oc2:
            use_neldermead_k = st.checkbox("Nelder-Mead", key="fit_use_neldermead", value=True)
        if not use_lbfgsb_k and not use_neldermead_k:
            st.warning("At least one optimizer must be selected.")
            use_lbfgsb_k = True

        if st.button("🔧 Fit Parameters", disabled=not fit_ok):
            if fit_ok:
                for _pref in ("fit_tolerance_log", "fit_timeout"):
                    if _pref in st.session_state:
                        st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                # Snapshot current concentration values before rerun clears widget state
                for _cname in parsed["concentrations"]:
                    _root = _cname[:-1] if _cname.endswith("0") else _cname
                    _ckey = f"conc_{_root}"
                    if _ckey in st.session_state:
                        st.session_state[f"_shadow_{_ckey}"] = st.session_state[_ckey]
                if "xmax" in st.session_state:
                    st.session_state["_shadow_xmax"] = st.session_state["xmax"]
                st.session_state["_fit_requested"] = True
                st.rerun()
            else:
                st.info("Load experimental data and check parameters to fit.")

        # ── Export ───────────────────────────────────────────────
        st.subheader("Data Export")
        col_exp, col_snap = st.columns(2)
        with col_exp:
            try:
                buf = _export_kinetics_excel(kin_curve, t_vals, plot_y_names,
                                              parsed_kin, logk_dict, script_text,
                                              variables,
                                              script_path=st.session_state.get("_script_filename"),
                                              input_path=st.session_state.get("_input_filename"))
                fname = f"Equilibrist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="💾 Export data",
                    data=buf,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch',
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
        with col_snap:
            try:
                _kin_fig = st.session_state.get("_current_figure")
                if _kin_fig is not None:
                    _kin_ptxt = generate_kinetics_parameters_text(
                        parsed_kin, logk_dict, script_text, xmax=float(t_max_ui))
                    _kin_snap_bytes, _kin_snap_fname = create_snapshot(
                        _kin_fig, parsed, {}, {},
                        xmax=float(t_max_ui),
                        x_label="Time [s]",
                        y_label="Concentration [mM]",
                        params_text=_kin_ptxt)
                    st.download_button(
                        label="📸 Snapshot",
                        data=_kin_snap_bytes,
                        file_name=_kin_snap_fname,
                        mime="application/pdf",
                        width='stretch',
                    )
            except Exception as _se:
                st.error(f"Snapshot failed: {_se}")
        # ── Experimental data upload ──────────────────────────────
        st.subheader("Experimental data")
        _kin_nmr_cfg_up = parsed.get("nmr")
        if _kin_nmr_cfg_up is not None:
            _kin_mode_up = _kin_nmr_cfg_up["mode"]
            if _kin_mode_up == "integration":
                st.caption(f"NMR mode: **integration** (slow exchange) — "
                           f"{len(_kin_nmr_cfg_up.get('n_H_list', []))} signals")
                _kin_hint = "💧 column A = time (s); columns B+ = species names (normalised integrals)"
            elif _kin_mode_up == "shift":
                st.caption(f"NMR mode: **shift** — targets: {', '.join(_kin_nmr_cfg_up['targets'])}")
                _kin_hint = "💧 column A = time (s); columns B+ = NMR observables"
            else:  # mixed
                st.caption(f"NMR mode: **mixed** — targets: {', '.join(_kin_nmr_cfg_up['targets'])}")
                _kin_hint = "💧 column A = time (s); columns B+ = NMR observables"
        else:
            _kin_hint = "💧 column A = time (s); other columns = species (mM)"
        st.caption(_kin_hint)
        if st.button("↺ Reset experimental data"):
            for k in list(st.session_state.keys()):
                if (k.startswith("_exp") or k.startswith("_nmr") or
                        k.startswith("_spectra") or k.startswith("_fit") or
                        k.startswith("_pending_logK") or k == "_input_filename"):
                    del st.session_state[k]
            st.rerun()

        if "_exp_uploader_nonce" not in st.session_state:
            st.session_state["_exp_uploader_nonce"] = 0
        _kin_has_spectra = parsed.get("spectra") is not None
        if _kin_has_spectra:
            st.caption("column A = time (s); row 1 = wavelengths (nm); body = absorbance")
        exp_uploaded = st.file_uploader(
            "Upload experimental data (.xlsx)", type=["xlsx"],
            key=f"_exp_uploader_{st.session_state['_exp_uploader_nonce']}")
        if exp_uploaded is not None:
            if _kin_has_spectra:
                _loaded_kin = load_spectra_data(exp_uploaded.read())
                st.session_state["_spectra_data"] = _loaded_kin
                # Seed wavelength range from actual data on first load
                if _loaded_kin and len(_loaded_kin.get("wavelengths", [])) > 0:
                    st.session_state["_pending_spectra_wl_min"] = float(_loaded_kin["wavelengths"][0])
                    st.session_state["_pending_spectra_wl_max"] = float(_loaded_kin["wavelengths"][-1])
                # Update xmax from last data point
                if _loaded_kin and len(_loaded_kin.get("x_vals", [])) > 0:
                    _x_last_kin = float(_loaded_kin["x_vals"][-1])
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_kin * 10) / 10)
            else:
                _loaded_kin = load_experimental_data(exp_uploaded.read())
                if _kin_nmr_cfg_up is not None:
                    st.session_state["_nmr_data"] = _loaded_kin
                else:
                    st.session_state["_exp_data"] = _loaded_kin
                # Update xmax from last data point (time axis stored in v_add_mL)
                _kin_cols = [c for c in _loaded_kin if not c.startswith("_")]
                if _kin_cols:
                    _x_last_kin = float(_loaded_kin[_kin_cols[0]]["v_add_mL"][-1])
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_kin * 10) / 10)
            st.session_state["_input_filename"] = exp_uploaded.name
            st.session_state["_exp_uploader_nonce"] += 1
            st.rerun()
        if st.session_state.get("_input_filename"):
            st.markdown(f"<span style='background:#1a6bbf;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:0.82rem'>📄 {st.session_state['_input_filename']}</span>", unsafe_allow_html=True)
        # Show summary of loaded data
        if _kin_has_spectra:
            _kin_sp_loaded = st.session_state.get("_spectra_data", {})
            if _kin_sp_loaded:
                st.caption(f"Loaded: {len(_kin_sp_loaded['x_vals'])} spectra × "
                           f"{len(_kin_sp_loaded['wavelengths'])} wavelengths "
                           f"({_kin_sp_loaded['wavelengths'][0]:.0f}–"
                           f"{_kin_sp_loaded['wavelengths'][-1]:.0f} nm)")
        elif _kin_nmr_cfg_up is not None:
            _kin_nmr_loaded = st.session_state.get("_nmr_data", {})
            if _kin_nmr_loaded:
                _n_sig = sum(1 for k in _kin_nmr_loaded if not k.startswith("_"))
                _n_pts = next((len(v["y"]) for k, v in _kin_nmr_loaded.items()
                               if not k.startswith("_")), 0)
                st.caption(f"Loaded: {_n_sig} signals × {_n_pts} points")
        else:
            _kin_exp_loaded = st.session_state.get("_exp_data", {})
            if _kin_exp_loaded:
                _total_pts = sum(len(v["v_add_mL"]) for k, v in _kin_exp_loaded.items()
                                 if not k.startswith("_"))
                st.caption(f"Loaded: {', '.join(k for k in _kin_exp_loaded if not k.startswith('_'))} ({_total_pts} pts)")

    # sidebar closes here — script editor goes below the plot in col1
        # ── Fit stats ────────────────────────────────────────────
        fit_stats = st.session_state.get("_fit_stats", {})
        if fit_stats:
            _kin_fit_mode = fit_stats.get("fit_mode", "kinetics")
            st.write("**Fit Statistics:**")
            if fit_stats.get("timed_out"):
                _to_s = float(st.session_state.get("fit_timeout", 30))
                _n_ev = fit_stats.get("n_iter", 0)
                st.warning(f"⏱️ Fit timed out after {_n_ev} evaluations ({_to_s:.0f} s limit) — "
                           "showing best result found. Consider increasing Timeout.")
            if _kin_fit_mode == "mixed":
                r2_i = fit_stats.get("r2_integ", 0.0); rmse_i = fit_stats.get("rmse_integ", 0.0)
                n_i  = fit_stats.get("n_integ_pts", 0)
                r2_s = fit_stats.get("r2_shift",  0.0); rmse_s = fit_stats.get("rmse_shift",  0.0)
                n_s  = fit_stats.get("n_shift_pts", 0)
                st.write(f"**Integration fit** ({n_i} points)")
                st.write(f"• R² = {r2_i:.4f}")
                st.write(f"• RMSE = {rmse_i:.2e} mM")
                st.write(f"**Shift fit** ({n_s} points)")
                st.write(f"• R² = {r2_s:.4f}")
                st.write(f"• RMSE = {rmse_s:.2e} ppm")
                n_total_k = fit_stats.get("n_points", n_i + n_s)
                st.write(f"• Total data points: {n_total_k}")
                if n_total_k > fit_stats.get("n_params", 0):
                    st.write(f"• Parameters fitted: {fit_stats.get('n_params', '?')}")
            else:
                st.write(f"• R² = {fit_stats['r_squared']:.4f}")
                _rmse_unit = " ppm" if _kin_fit_mode == "shift" else " mM"
                st.write(f"• RMSE = {fit_stats['rmse']:.2e}{_rmse_unit}")
                st.write(f"• Data points: {fit_stats['n_points']}")
                if fit_stats['n_points'] > fit_stats.get('n_params', 0):
                    _rchi2 = fit_stats['ssr'] / (fit_stats['n_points'] - fit_stats['n_params'])
                    st.write(f"• Reduced χ² = {_rchi2:.2e}")
            if "n_iter" in fit_stats:
                st.write(f"• Objective evaluations: {fit_stats['n_iter']}")

            # ── Fitted rate constants (k and ±k only, no log) ────────────
            param_values     = fit_stats.get("param_values", {})
            param_errors     = fit_stats.get("param_errors", {})
            fitted_concs_kin = fit_stats.get("fitted_concs", {})
            if param_values or fitted_concs_kin:
                st.write("**Fitted constants:**")
                rows = []
                # Equilibrium constants: show log + linear; rate constants: linear only
                _eq_knames_disp = {eq["kname"] for eq in parsed.get("equilibria", [])}
                for kname, val in param_values.items():
                    err     = param_errors.get(kname)
                    k_lin   = 10.0 ** val
                    err_lin = k_lin * 2.302585 * err if err is not None else None
                    if kname in _eq_knames_disp:
                        rows.append({"Parameter": kname,
                                     "log P":  f"{val:.2f}",
                                     "±log P": f"± {err:.2f}" if err is not None else "n/a",
                                     "P":      f"{k_lin:.2e}",
                                     "±P":     f"± {err_lin:.2e}" if err_lin is not None else "n/a"})
                    else:
                        rows.append({"Parameter": kname,
                                     "log P":  "—",
                                     "±log P": "—",
                                     "P":      f"{k_lin:.2e}",
                                     "±P":     f"± {err_lin:.2e}" if err_lin is not None else "n/a"})
                # linear-space concentration parameters (mM)
                for root, mM_val in fitted_concs_kin.items():
                    err_c = param_errors.get(root)
                    rows.append({"Parameter": root,
                                 "log P":  "—",
                                 "±log P": "—",
                                 "P":      f"{mM_val:.2e} mM",
                                 "±P":     f"± {err_c:.2e} mM" if err_c is not None else "n/a"})
                if rows:
                    st.dataframe(pd.DataFrame(rows).set_index("Parameter"), width='stretch')

                # ── Free energy table ─────────────────────────────────────────
                if param_values:
                    import math as _math
                    _T    = float(parsed.get("temperature_K", 298.15))
                    _R    = 1.987e-3   # kcal/(mol·K)
                    _lnKT = _math.log(1.380649e-23 * _T / 6.62607015e-34)  # ln(k_BT/h)
                    _eq_knames_set  = {eq["kname"] for eq in parsed.get("equilibria", [])}
                    _kin_knames_set = ({r["kname"]  for r in parsed.get("kinetics", [])} |
                                       {r["krname"] for r in parsed.get("kinetics", []) if "krname" in r})
                    _energy_rows = []
                    _has_act = False
                    for _kn, _val in param_values.items():
                        _err = param_errors.get(_kn)
                        if _kn in _kin_knames_set:
                            _has_act = True
                            _dG = _R * _T * (_lnKT - _math.log(10) * _val)
                            _dG_err = _R * _T * _math.log(10) * _err if _err is not None else None
                            _energy_rows.append({"Parameter": _kn + " *",
                                                 "ΔG (kcal/mol)": f"{_dG:.2f}",
                                                 "±ΔG": f"± {_dG_err:.2f}" if _dG_err is not None else "n/a"})
                        else:
                            _dG = -_R * _T * _math.log(10) * _val
                            _dG_err = _R * _T * _math.log(10) * _err if _err is not None else None
                            _energy_rows.append({"Parameter": _kn,
                                                 "ΔG (kcal/mol)": f"{_dG:.2f}",
                                                 "±ΔG": f"± {_dG_err:.2f}" if _dG_err is not None else "n/a"})
                    if _energy_rows:
                        st.write(f"**Free energies ({_T:.2f} K):**")
                        st.dataframe(pd.DataFrame(_energy_rows).set_index("Parameter"), width='stretch')
                        if _has_act:
                            st.caption("* ΔG‡: assuming elementary steps")

            # ── NMR signal assignment (integration / mixed) ───────────────
            sp_concs_disp  = fit_stats.get("sp_concs", {})
            col_to_sp_disp = fit_stats.get("col_to_sp", {})
            col_to_nH_disp = fit_stats.get("col_to_nH", {})
            if sp_concs_disp:
                st.write("**NMR signal assignment:**")
                rows_bc = []
                for sp, arr_list in sp_concs_disp.items():
                    n_signals = sum(1 for col, csp in col_to_sp_disp.items() if csp == sp)
                    n_H_vals  = sorted(set(col_to_nH_disp.get(col, "?")
                                          for col, csp in col_to_sp_disp.items() if csp == sp))
                    rows_bc.append({
                        "Species":        sp,
                        "# signals used": max(n_signals, 1),
                        "n_H per signal": ", ".join(
                            str(int(v) if isinstance(v, float) and v == int(v) else v)
                            for v in n_H_vals) or "1",
                    })
                if rows_bc:
                    st.dataframe(pd.DataFrame(rows_bc).set_index("Species"),
                                 width='stretch')
                    st.caption("Concentrations averaged from all signals of each species")

            # ── Pure-species chemical shifts (shift / mixed) ──────────────
            pure_shifts_disp = fit_stats.get("pure_shifts", {})
            if pure_shifts_disp:
                st.write("**Pure-species chemical shifts (ppm):**")
                all_sp_cols = []; rows_ps = []
                for tgt, col_dict in pure_shifts_disp.items():
                    for col, sp_dict in col_dict.items():
                        row = {"Signal": col}
                        for sp, delta in sp_dict.items():
                            row[sp] = f"{delta:.4f}"
                            if sp not in all_sp_cols:
                                all_sp_cols.append(sp)
                        rows_ps.append(row)
                if rows_ps:
                    df_ps = pd.DataFrame(rows_ps).set_index("Signal")
                    ordered = [c for c in all_sp_cols if c in df_ps.columns]
                    st.dataframe(df_ps[ordered], width='stretch')
                    st.caption("Each row = one NMR signal; columns = δ of each pure species (ppm)")

            if _kin_fit_mode == "mixed":
                integ_sp   = list(sp_concs_disp.keys())
                shift_tgts = list(pure_shifts_disp.keys())
                st.caption(
                    f"Mixed fit: slow-exchange integrations [{', '.join(integ_sp)}] + "
                    f"fast-exchange shifts [{', '.join(shift_tgts)}] fitted simultaneously.")

    with col1:
        _render_script_editor()
    st.stop()   # ← kinetics branch ends here; nothing below runs


# ═══════════════════════════════════════════════════════════════════
# TITRATION / EQUILIBRIUM BRANCH  (unchanged)
# ═══════════════════════════════════════════════════════════════════

try:
    network = build_network(parsed)
except Exception as e:
    st.warning(f"⚠️ Script error (network build) — please revise: {e}")
    st.stop()

titrant_key        = network["titrant_key"]         # e.g. 'Mt'
titrant_name       = network["titrant_name"]        # e.g. 'M'
titrant_free_names = network["titrant_free_names"]  # e.g. ['M', 'Q']
titrant_keys       = network["titrant_keys"]        # e.g. ['Mt', 'Qt']

# Map $concentrations entries to root species (strip trailing "0")
conc_roots = {}
for cname, cval in parsed["concentrations"].items():
    root = cname[:-1] if cname.endswith("0") else cname
    conc_roots[root] = cval

primary_component = list(conc_roots.keys())[0] if conc_roots else network["free_species"][0]

# ── Sidebar: concentrations, volume, titrant ──
with st.sidebar:

    st.header("Initial concentrations (mM)")
    conc_vals = {}
    for root, default in conc_roots.items():
        conc_vals[root] = _conc_input_with_fit(
            f"{root}₀", key=f"conc_{root}", default=float(default),
        )

    st.header("Volume & Titrant")
    V0_default  = list(parsed["volumes"].values())[0]  if parsed["volumes"]  else 0.5
    V0_mL = _num_input("V₀ (mL)", key="V0_mL", default=float(V0_default), step=0.01, format="%.4f")

    titrant_mMs    = {}
    titrant_ratios = {}
    if parsed["titrant_is_solid"]:
        st.caption("🪨 Solid titrant — volume fixed at V₀")
        for tkey in titrant_keys:
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            raw_ratio = parsed["titrant_ratios"].get(tkey, 1.0)
            titrant_mMs[tfree]    = 0.0        # not used for solid
            titrant_ratios[tfree] = raw_ratio
        if len(titrant_keys) > 1:
            total_r = sum(titrant_ratios.values())
            for tfree, r in titrant_ratios.items():
                st.caption(f"  {tfree}: fraction = {r/total_r:.4f}")
    else:
        for tkey in titrant_keys:
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            tit_default = parsed["titrant"].get(tkey, 10.0)
            titrant_mMs[tfree] = _conc_input_with_fit(
                f"{tkey} stock (mM)", key=f"titrant_mM_{tkey}",
                default=float(tit_default),
            )
            titrant_ratios[tfree] = 1.0

    st.header("Plot settings")
    # Default x-expression: H0/G0 (equivalents using X0 convention)
    _ref_cname = list(parsed['concentrations'].keys())[0]  # e.g. 'G0'
    _tit_free  = network['titrant_free_names'][0] if network.get('titrant_free_names') else titrant_key[:-1] if titrant_key.endswith('t') else titrant_key
    x_expr_default = parsed["plot_x_expr"] if parsed["plot_x_expr"] else f"{_tit_free}0/{_ref_cname}"
    st.caption(f"x-axis expression: `{x_expr_default}`")
    xmax = _num_input("X-axis max (sweep range)", key="xmax", default=float(parsed["plot_xmax"]), step=0.1, format="%.3f")
    nPts = st.number_input("# points", value=100, step=1, min_value=5)

    # ── Wavelength range (only shown once spectra data is loaded) ────
    _sd = st.session_state.get("_spectra_data", {})
    if parsed.get("spectra") is not None and _sd:
        _wl_lo = float(_sd["wavelengths"][0])
        _wl_hi = float(_sd["wavelengths"][-1])
        st.caption("Wavelength range (nm)")
        _wl_c1, _wl_c2 = st.columns(2)
        with _wl_c1:
            _num_input("Min", key="spectra_wl_min", default=_wl_lo, step=1.0, format="%.0f")
        with _wl_c2:
            _num_input("Max", key="spectra_wl_max", default=_wl_hi, step=1.0, format="%.0f")
        st.checkbox("Auto-optimize range", key="spectra_auto_range")
        st.checkbox("Allow negative absorbances", key="spectra_allow_neg")

    st.header("Equilibrium constants (log₁₀ K)")
    logK_vals = {}
    for eq in parsed["equilibria"]:
        kname = eq["kname"]
        
        # Format reactants: (coeff, species) → "coeff species" or just "species" if coeff=1
        reactants_str = []
        for coeff, species in eq["reactants"]:
            if coeff == 1:
                reactants_str.append(species)
            else:
                reactants_str.append(f"{coeff}{species}")
        
        # Format product(s): handle both single and multiple products
        products_display = []
        if "products" in eq:
            for prod_coeff, prod_species in eq["products"]:
                if prod_coeff == 1:
                    products_display.append(prod_species)
                else:
                    products_display.append(f"{prod_coeff}{prod_species}")
        elif "product" in eq:
            # Backwards compatibility
            prod_coeff, prod_species = eq["product"]
            if prod_coeff == 1:
                products_display.append(prod_species)
            else:
                products_display.append(f"{prod_coeff}{prod_species}")
        
        products_str = " + ".join(products_display)
        
        label = f"{kname}  ({' + '.join(reactants_str)} ⇌ {products_str})"
        logK_vals[kname] = _logk_input_with_fit(label, key=f"logK_{kname}", default=eq["logK"])

# ── Constraints toggle (only shown when $constraints section present) ──
if parsed.get("constraints"):
    with st.sidebar:
        st.checkbox(
            "Activate constraints",
            key="fit_use_constraints",
            value=False,
            help=f"{len(parsed['constraints'])} constraint(s) defined in $constraints section.",
        )

with st.sidebar:
    st.markdown("---")
    st.caption("Equilibrist © Eric Masson, Ohio University, 2026")

# ── Thermodynamic cycle detection ────────────
cycle_warnings = detect_thermodynamic_cycles(parsed, logK_vals)
for w in cycle_warnings:
    st.warning(w)

# ── Compute ───────────────────────────────────
# xmax is always in x-axis units — compute how much x we get per equivalent
# so that the sweep exactly covers [0, xmax].
# _x_per_equiv is defined earlier in the file (near find_equiv_for_x).

_ref_cname_m = list(parsed['concentrations'].keys())[0]
_tit_free_m  = network['titrant_free_names'][0] if network.get('titrant_free_names') else titrant_key[:-1] if titrant_key.endswith('t') else titrant_key
_x_expr_main = parsed.get("plot_x_expr") or f"{_tit_free_m}0/{_ref_cname_m}"
_xpe = _x_per_equiv(
    _x_expr_main, parsed, conc_vals, float(V0_mL),
    titrant_free_names, titrant_keys, titrant_mMs, titrant_ratios,
    parsed["titrant_is_solid"], primary_component,
)
_maxEquiv = float(xmax) / _xpe

params = {
    "conc0":               conc_vals,
    "V0_mL":               float(V0_mL),
    "titrant_name":        titrant_name,
    "titrant_key":         titrant_key,
    "titrant_free_names":  titrant_free_names,
    "titrant_keys":        titrant_keys,
    "titrant_mMs":         titrant_mMs,
    "titrant_ratios":      titrant_ratios,
    "titrant_is_solid":    parsed["titrant_is_solid"],
    "maxEquiv":            _maxEquiv,
    "nPts":                int(nPts),
    "primary_component":   primary_component,
}


with st.spinner("Solving mass balances…"):
    try:
        curve = compute_curve(parsed, network, logK_vals, params)
    except Exception as e:
        st.warning(f"⚠️ Solver error — check concentrations and reaction definitions: {e}")
        st.stop()

# ── Evaluate x-axis expression (needed for fitting) ────
x_expr = parsed["plot_x_expr"]
if not x_expr:
    ref_key = list(parsed["concentrations"].keys())[0]
    _tit_free_fb = network['titrant_free_names'][0] if network.get('titrant_free_names') else titrant_key[:-1] if titrant_key.endswith('t') else titrant_key
    x_expr  = f"{_tit_free_fb}0/{ref_key}"   # e.g. H0/G0 (X0 convention)

# Patch parsed concentrations with sidebar values so x-axis expression
# (e.g. Gt/cage0) uses the same cage0 as convert_exp_x does for the dots.
_parsed_for_x = dict(parsed)
_parsed_for_x["concentrations"] = dict(parsed["concentrations"])
for _root, _cval in conc_vals.items():
    _cname = _root + "0" if not _root.endswith("0") else _root
    if _cname in _parsed_for_x["concentrations"]:
        _parsed_for_x["concentrations"][_cname] = float(_cval)

try:
    x_vals, x_label = evaluate_x_expression(x_expr, curve, _parsed_for_x)
except ValueError as e:
    st.warning(f"⚠️ x-axis expression error — please revise $plot x: {e}")
    st.stop()

# ── Clip to xmax in x-axis units (purely cosmetic) ───────────────────
x_mask = x_vals <= float(xmax)
x_vals = x_vals[x_mask]
for key in curve:
    if isinstance(curve[key], np.ndarray) and len(curve[key]) == len(x_mask):
        curve[key] = curve[key][x_mask]

# ── Resolve $plot y targets ───────────────────
def resolve_plot_targets(plot_y_names, variables, all_species, x0_keys=None):
    """x0_keys: set of X0 variable names available in curve (e.g. {'G0','H0'})."""
    resolved  = []
    plot_warns = []
    _x0 = x0_keys or set()
    for name in plot_y_names:
        if name in variables:
            expr = variables[name]
            
            # Check if it's a simple sum (old style: "G + GM + GM2")
            if all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+ ' for c in expr):
                # Old-style sum variable - parse as before
                members = [p.strip() for p in expr.split("+") if p.strip()]
                valid   = [sp for sp in members if sp in all_species or sp in _x0]
                invalid = [sp for sp in members if sp not in all_species and sp not in _x0]
                if invalid:
                    plot_warns.append(f"Variable '{name}': species {invalid} not found, ignored.")
                if valid:
                    resolved.append((name, valid))
            else:
                # New-style expression variable - treat as single computed target
                resolved.append((name, [name]))  # Will be computed via expression evaluation
                
        elif name in all_species or name in _x0:
            resolved.append((name, [name]))
        else:
            plot_warns.append(f"Plot target '{name}' is not a species or variable — skipped.")
    return resolved, plot_warns

plot_y_names = parsed["plot_y"] if parsed["plot_y"] else network["all_species"][:6]
_x0_curve_keys = {k for k in curve if k.endswith("0") and k not in {"V0_mL", "totals_mM"}}
plot_targets, plot_warns = resolve_plot_targets(plot_y_names, parsed["variables"], network["all_species"], _x0_curve_keys)

for w in plot_warns:
    st.warning(w)

# ── Fit dispatch — runs BEFORE plots so spinner appears at top ────────
if st.session_state.pop("_fit_requested", False):
    exp_data_fit     = st.session_state.get("_exp_data", {})
    nmr_data_fit     = st.session_state.get("_nmr_data", {})
    spectra_data_fit = st.session_state.get("_spectra_data", {})
    nmr_cfg_fit      = parsed.get("nmr")
    use_nmr_fit      = (nmr_cfg_fit is not None and
                        nmr_cfg_fit["mode"] in ("shift", "integration", "mixed") and
                        bool(nmr_data_fit))
    use_spectra_fit  = parsed.get("spectra") is not None and bool(spectra_data_fit)

    fit_keys_end = [eq["kname"] for eq in parsed["equilibria"]
                    if st.session_state.get(f"fit_logK_{eq['kname']}", False)]
    has_data_end = use_nmr_fit or use_spectra_fit or bool(exp_data_fit)

    # Concentrations and titrant to fit
    fit_conc_keys_end = [root for root in conc_roots
                         if st.session_state.get(f"fit_conc_{root}", False)]
    fit_titrant_keys_end = [tkey for tkey in titrant_keys
                            if st.session_state.get(f"fit_titrant_mM_{tkey}", False)]

    if has_data_end and (fit_keys_end or fit_conc_keys_end or fit_titrant_keys_end):
        try:
            tolerance_log   = st.session_state.get("fit_tolerance_log", 6.0)
            tolerance       = 10.0 ** (-tolerance_log)
            maxiter         = 100_000
            _use_lbfgsb     = st.session_state.get("fit_use_lbfgsb", True)
            _use_neldermead = st.session_state.get("fit_use_neldermead", True)
            if not _use_lbfgsb and not _use_neldermead:
                _use_lbfgsb = True

            # Constraints are only passed if the toggle is active
            _this_fit_constrained = st.session_state.get("fit_use_constraints", False)
            _last_fit_constrained = st.session_state.get("_last_fit_was_constrained", False)
            _active_constraints = (
                parsed.get("constraints", []) if _this_fit_constrained else []
            )

            # If constraints changed state, reset to script defaults so the
            # optimizer starts from a point that satisfies (or at least doesn't
            # badly violate) the constraints.
            # Also reset to script defaults when fitting concentrations: the sidebar
            # logK values may be far from the true optimum at the script A0, causing
            # L-BFGS-B to push A0 to its boundary before logK can adjust.
            _script_defaults_eq = {eq["kname"]: eq["logK"] for eq in parsed["equilibria"]}
            if _this_fit_constrained != _last_fit_constrained:
                _start_logK = {k: _script_defaults_eq.get(k, v) for k, v in logK_vals.items()}
            else:
                _start_logK = logK_vals

            st.session_state["_last_fit_was_constrained"] = _this_fit_constrained

            if use_spectra_fit:
                wl_min_fit = float(st.session_state.get("spectra_wl_min",
                                   spectra_data_fit["wavelengths"][0]))
                wl_max_fit = float(st.session_state.get("spectra_wl_max",
                                   spectra_data_fit["wavelengths"][-1]))
                auto_range    = bool(st.session_state.get("spectra_auto_range", False))
                allow_neg_eps = bool(st.session_state.get("spectra_allow_neg", False))
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                with st.spinner("Fitting parameters…"):
                    success, fitted_logKs, stats, message = fit_spectra(
                        parsed, network, spectra_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, wl_min_fit, wl_max_fit,
                        tolerance, maxiter, auto_range=auto_range, timeout_s=_timeout_s, allow_negative_eps=allow_neg_eps,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end)
                if auto_range and "opt_wl_min" in stats:
                    st.session_state["_pending_spectra_wl_min"] = stats["opt_wl_min"]
                    st.session_state["_pending_spectra_wl_max"] = stats["opt_wl_max"]
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "shift":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                with st.spinner("Fitting parameters…"):
                    success, fitted_logKs, stats, message = fit_nmr_shifts(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end)
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "integration":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                with st.spinner("Fitting parameters…"):
                    success, fitted_logKs, stats, message = fit_nmr_integration(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end)
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "mixed":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                with st.spinner("Fitting parameters…"):
                    success, fitted_logKs, stats, message = fit_nmr_mixed(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end)
            else:
                with st.spinner("Fitting parameters…"):
                    success, fitted_logKs, stats, message = fit_parameters(
                        parsed, network, exp_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        use_lbfgsb=_use_lbfgsb, use_neldermead=_use_neldermead,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end)

            for kname, fitted_val in fitted_logKs.items():
                st.session_state[f"_pending_logK_{kname}"] = float(fitted_val)
            # Push fitted concentrations back to sidebar widgets
            for root, mM_val in stats.get("fitted_concs", {}).items():
                st.session_state[f"_pending_conc_{root}"] = float(mM_val)
            for tkey, mM_val in stats.get("fitted_titrants", {}).items():
                st.session_state[f"_pending_titrant_mM_{tkey}"] = float(mM_val)
            st.session_state["_fit_stats"] = stats
            _n_evals = stats.get("n_iter", 0)
            _n_fitted = len(fitted_logKs) + len(stats.get("fitted_concs", {})) + len(stats.get("fitted_titrants", {}))
            if stats.get("timed_out"):
                _to_s = float(st.session_state.get("fit_timeout", 30))
                st.session_state["_fit_message"] = ("warning",
                    f"⏱️ Fit timed out after {_n_evals} evaluations ({_to_s:.0f} s limit) — "
                    f"showing best result found. Consider increasing Timeout.")
            elif success:
                st.session_state["_fit_message"] = ("success",
                    f"Fit completed! Updated {_n_fitted} parameters. "
                    f"(tol=1e-{tolerance_log:.0f})")
            else:
                st.session_state["_fit_message"] = ("warning",
                    f"Fit did not fully converge — {message}")
            st.rerun()

        except Exception as e:
            import traceback
            st.error(f"Fitting error: {str(e)}")
            st.caption(traceback.format_exc())

# ── Plot ──────────────────────────────────────
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52",
]

col1, col2 = st.columns([2.2, 1.0], gap="large")

# Initialize pure_shifts here so it is always defined for both col1 and col2.
# It is only populated during NMR shift fitting (not integration mode).
pure_shifts = {}

with col1:
    fig = go.Figure()

    # Build a color map: label → color (for matching exp dots to traces)
    trace_colors = {}
    for idx, (label, species_list) in enumerate(plot_targets):
        color  = COLORS[idx % len(COLORS)]
        trace_colors[label] = color
        
        # Check if this is an expression variable vs species sum
        variables = parsed.get("variables", {})
        if len(species_list) == 1 and species_list[0] == label and label in variables:
            y_vals = compute_variable_curve(label, variables, curve, network, x_vals)
        else:
            y_vals = sum(curve.get(sp, np.zeros_like(x_vals)) for sp in species_list)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    warn = curve.get("warn", np.zeros(len(x_vals), dtype=bool))
    if np.any(warn) and plot_targets:
        first_y = sum(curve.get(sp, np.zeros_like(x_vals)) for sp in plot_targets[0][1])
        fig.add_trace(go.Scatter(
            x=x_vals[warn], y=first_y[warn],
            mode="markers",
            marker=dict(symbol="x", size=10, color="yellow"),
            name="solver warning",
        ))

    # ── Regular experimental data overlay (non-NMR) ──────────────
    exp_data = st.session_state.get("_exp_data", {})
    if exp_data:
        # ── Pass 1: resolve direct species → variable matches ────────────
        # For each raw exp column (e.g. "A"), find the plotted variable it
        # directly feeds (e.g. "%A" because A appears in "%A = A/S").
        # Transform the y-values through that variable expression and record
        # the resulting exp series keyed by variable name.
        # Also collect a shared x-axis from the first column that has data.
        exp_series = {}   # {var_name: (exp_x, exp_y_transformed)}

        for col_name, col_vals in exp_data.items():
            if col_name.startswith("_"): continue
            matched_var = find_variable_for_exp_col(
                col_name, variables, plot_y_names, network["all_species"])
            exp_x = convert_exp_x(
                col_vals["v_add_mL"], x_expr, parsed, params, network)
            if matched_var and matched_var in trace_colors:
                exp_y_transformed = transform_exp_via_variable(
                    matched_var, variables,
                    col_name, exp_x, col_vals["y"],
                    curve, x_vals, network)
                exp_series[matched_var] = (exp_x, exp_y_transformed)
            else:
                # No variable match — plot raw, match color by name
                if col_name in trace_colors:
                    display_name = col_name
                elif "%" + col_name in trace_colors:
                    display_name = "%" + col_name
                elif col_name.lstrip("%") in trace_colors:
                    display_name = col_name.lstrip("%")
                else:
                    display_name = col_name
                exp_series[display_name] = (exp_x, col_vals["y"])

        # ── Pass 2: propagate through chained variables ──────────────────
        # If "%AB = %A + %B" and both "%A" and "%B" have exp series,
        # compute "%AB (exp)" by evaluating the expression using the
        # already-resolved exp values (same logic as transform_exp_via_variable
        # but substituting all available exp series, not just one species).
        dep_order = resolve_variable_dependencies(variables)
        for var_name in dep_order:
            if var_name in exp_series:
                continue                        # already computed in pass 1
            if var_name not in plot_y_names:
                continue                        # not plotted — skip
            expr_deps = extract_identifiers_from_expression(variables[var_name])
            # Check every dep that is itself a variable is covered by exp_series
            var_deps = [d for d in expr_deps if d in variables]
            if not var_deps:
                continue                        # depends only on species, handled in pass 1
            if not all(d in exp_series for d in var_deps):
                continue                        # some dep has no exp data — skip
            # All variable deps are available: evaluate the chained expression
            # Use x-axis from the first dep (all should be on the same x-grid)
            ref_x, _ = exp_series[var_deps[0]]
            n = len(ref_x)
            chained_y = np.full(n, np.nan)
            for i in range(n):
                # Build namespace from exp_series values at this point
                var_vals = {}
                for dep in var_deps:
                    dep_x, dep_y = exp_series[dep]
                    # Interpolate dep's exp series at this x point
                    var_vals[dep] = float(np.interp(ref_x[i], dep_x, dep_y))
                # Also interpolate theoretical species for any species deps
                species_vals = {sp: float(np.interp(ref_x[i], x_vals,
                                curve.get(sp, np.zeros_like(x_vals))))
                                for sp in network["all_species"]}
                chained_y[i] = evaluate_variable_expression(
                    variables[var_name], species_vals, var_vals)
            exp_series[var_name] = (ref_x, chained_y)

        # ── Render all exp series ────────────────────────────────────────
        for display_name, (exp_x, plot_y_vals) in exp_series.items():
            color = trace_colors.get(display_name, "#888888")
            fig.add_trace(go.Scatter(
                x=exp_x, y=plot_y_vals,
                mode="markers", name=f"{display_name} (exp)",
                marker=dict(color=color, size=7, symbol="circle",
                            line=dict(width=1, color="white")),
                showlegend=True,
            ))

    # ── NMR back-calculated concentration dots on main plot ───────
    # Correct formula: at each exp point j, for each signal k:
    #   Σᵢ nᵢ·[Sᵢ](j) × Δδᵢ_k  =  (Δδ_obs_rel_k(j) + ref_correction_k) × Ctot_k_theory(j)
    #
    # where Δδ_obs_rel_k = δ_obs - δ_free_k (data relative to first point),
    #       ref_correction_k = Σᵢ Fᵢ(x_ref) × Δδᵢ_k  (from fit stats),
    #       Ctot_k_theory(j) = weighted Σ nᵢ·[Sᵢ] from the theoretical curve.
    #
    # Build M[k,i] = nᵢ × Δδᵢ_k  and solve at each point.
    fit_stats_main  = st.session_state.get("_fit_stats", {})
    nmr_data_main   = st.session_state.get("_nmr_data", {})
    delta_vecs_main = fit_stats_main.get("delta_vecs_all", {})
    delta_f_main    = fit_stats_main.get("delta_free", {})
    x_free_main     = fit_stats_main.get("x_free_val", {})
    col_to_tgt_main = fit_stats_main.get("col_to_target", {})
    ref_corr_main   = fit_stats_main.get("ref_corrections", {})

    if parsed.get("nmr") is not None and nmr_data_main and delta_vecs_main:
        nmr_cfg_main = parsed["nmr"]
        fitted_cols  = [col for col in nmr_data_main
                        if not col.startswith("_") and col in delta_vecs_main]

        if fitted_cols:
            # Common x-grid: use first fitted signal's volume points
            ref_v    = nmr_data_main[fitted_cols[0]]["v_add_mL"]
            exp_x_bc = convert_exp_x(ref_v, x_expr, parsed, params, network)
            n_pts    = len(exp_x_bc)

            # Unique non-free species across all targets (ordered)
            non_free_sp = []
            for tgt in nmr_cfg_main["targets"]:
                for _, sp in _get_species_for_target(tgt, parsed, network)[1:]:
                    if sp not in non_free_sp:
                        non_free_sp.append(sp)
            n_nfree = len(non_free_sp)
            n_sig   = len(fitted_cols)

            # Build M[k,i] = nᵢ × Δδᵢ_k  and RHS[k,j]
            M_mat   = np.zeros((n_sig, n_nfree))
            rhs_mat = np.zeros((n_sig, n_pts))

            for k, col in enumerate(fitted_cols):
                tgt       = col_to_tgt_main.get(col)
                if tgt is None: continue
                sp_coeffs = _get_species_for_target(tgt, parsed, network)
                sp_dd     = delta_vecs_main[col]
                ref_corr  = ref_corr_main.get(col, 0.0)

                # M[k,i] = nᵢ × Δδᵢ_k
                for coeff, sp in sp_coeffs[1:]:
                    if sp in non_free_sp:
                        M_mat[k, non_free_sp.index(sp)] = coeff * sp_dd.get(sp, 0.0)

                # Ctot_k_theory(j) = Σᵢ nᵢ·[Sᵢ](j) from theoretical curve at exp_x_bc
                ctot_k = np.maximum(
                    sum(coeff * np.interp(exp_x_bc, x_vals,
                                          curve.get(sp, np.zeros_like(x_vals)))
                        for coeff, sp in sp_coeffs), 1e-20)

                # Δδ_obs_rel interpolated onto exp_x_bc
                x_col       = convert_exp_x(nmr_data_main[col]["v_add_mL"], x_expr, parsed, params, network)
                df0         = float(delta_f_main.get(col, nmr_data_main[col]["y"][0]))
                dobs_rel    = np.interp(exp_x_bc, x_col, nmr_data_main[col]["y"] - df0)

                # RHS[k,j] = (Δδ_obs_rel + ref_correction) × Ctot_theory
                rhs_mat[k, :] = (dobs_rel + ref_corr) * ctot_k

            # Solve M @ c_nonfree(j) = rhs(j) at each point
            conc_nonfree = np.zeros((n_nfree, n_pts))
            rank = np.linalg.matrix_rank(M_mat)
            if n_nfree > 0 and rank >= min(n_sig, n_nfree):
                for j in range(n_pts):
                    sol = np.linalg.lstsq(M_mat, rhs_mat[:, j], rcond=None)[0]
                    conc_nonfree[:, j] = np.clip(sol, 0.0, None)

            # Recover free-species concentration from mass balance of each target
            c_bc_eq_shift = {}
            for tgt in nmr_cfg_main["targets"]:
                sp_coeffs = _get_species_for_target(tgt, parsed, network)
                if not sp_coeffs: continue
                free_sp = sp_coeffs[0][1]
                ctot_unweighted = np.maximum(
                    sum(np.interp(exp_x_bc, x_vals, curve.get(sp, np.zeros_like(x_vals)))
                        for _, sp in sp_coeffs), 1e-20)
                sum_nonfree = np.zeros(n_pts)
                for _, sp in sp_coeffs[1:]:
                    if sp in non_free_sp:
                        sum_nonfree += conc_nonfree[non_free_sp.index(sp), :]
                c_free_bc = np.clip(ctot_unweighted - sum_nonfree, 0.0, None)
                for _, sp in sp_coeffs:
                    if sp in c_bc_eq_shift: continue
                    if sp == free_sp:
                        c_bc_eq_shift[sp] = (exp_x_bc, c_free_bc)
                    elif sp in non_free_sp:
                        c_bc_eq_shift[sp] = (exp_x_bc, conc_nonfree[non_free_sp.index(sp), :])
            _nmr_suffix_shift = "(NMR/shift)" if nmr_cfg_main.get("mode") == "mixed" else "(NMR)"
            _plot_backcalc_dots(fig, c_bc_eq_shift, plot_y_names,
                                parsed.get("variables", {}), network["all_species"],
                                trace_colors, label_suffix=_nmr_suffix_shift)

    # ── NMR integration back-calculated concentrations ───────────
    # After fitting in integration mode, sp_concs contains per-signal
    # back-calculated [Sp] arrays.  Average across signals of same species
    # and plot as dots on the main concentration plot.
    fit_stats_integ = st.session_state.get("_fit_stats", {})
    nmr_data_integ  = st.session_state.get("_nmr_data", {})
    nmr_cfg_integ   = parsed.get("nmr")
    sp_concs_integ  = fit_stats_integ.get("sp_concs", {})

    if (nmr_cfg_integ is not None and
            nmr_cfg_integ["mode"] in ("integration", "mixed") and
            nmr_data_integ and sp_concs_integ):
        # ── Post-fit: show back-calculated concentrations ──────────
        _bc_post = {}
        for sp, arr_list in sp_concs_integ.items():
            x_ref = arr_list[0][0]
            c_avg = np.mean([np.interp(x_ref, xa, ca) for xa, ca in arr_list], axis=0)
            _bc_post[sp] = (x_ref, c_avg)
        _nmr_suffix_integ = "(NMR/integration)" if nmr_cfg_integ.get("mode") == "mixed" else "(NMR)"
        _plot_backcalc_dots(fig, _bc_post, plot_y_names,
                            parsed.get("variables", {}), network["all_species"],
                            trace_colors, label_suffix=_nmr_suffix_integ)

    elif (nmr_cfg_integ is not None and
            nmr_cfg_integ["mode"] in ("integration", "mixed") and
            nmr_data_integ and not sp_concs_integ):
        # ── Pre-fit: back-calculate concentrations from integrals ─────────────
        # Uses the same physical formula as the fit function (no K needed).
        # For mixed mode, only the first n_integ columns are integrations.
        n_H_list_pre = nmr_cfg_integ.get("n_H_list", [])
        n_integ_pre  = nmr_cfg_integ.get("n_integ", len(n_H_list_pre))
        all_cols_pre = [c for c in nmr_data_integ if not c.startswith("_")]
        integ_cols_pre = all_cols_pre[:n_integ_pre]
        integ_data_pre = {c: nmr_data_integ[c] for c in integ_cols_pre}
        bc_pre = _nmr_integration_backCalc(
            integ_data_pre, n_H_list_pre[:n_integ_pre], params, network, x_expr, parsed)
        _bc_pre_pairs = {sp_p: (x_p, c_bc_p) for sp_p, (x_p, c_bc_p) in bc_pre.items()}
        _nmr_suffix_pre = "(NMR/integration)" if nmr_cfg_integ.get("mode") == "mixed" else "(NMR)"
        _plot_backcalc_dots(fig, _bc_pre_pairs, plot_y_names,
                            parsed.get("variables", {}), network["all_species"],
                            trace_colors, label_suffix=_nmr_suffix_pre)

    # ── UV-Vis spectra back-calculated concentrations ───────────
    fit_stats_sp = st.session_state.get("_fit_stats", {})
    if fit_stats_sp.get("fit_mode") == "spectra":
        absorbers_sp = fit_stats_sp.get("absorbers", [])
        _sd_bc       = st.session_state.get("_spectra_data", {})
        _x_raw_sp_bc = _sd_bc.get("x_vals", np.array([]))
        E_final_sp   = fit_stats_sp.get("E_final", None)
        C_back_sp    = fit_stats_sp.get("C_back", None)
        if E_final_sp is not None and len(_x_raw_sp_bc) > 0:
            try:
                # Recompute C_back from A using current E_final so y-values
                # are consistent with current cage0 (sidebar value).
                _A_sp_bc   = _sd_bc.get("A", None)
                _wl_fit_sp = fit_stats_sp.get("wavelengths_fit", np.array([]))
                if _A_sp_bc is not None and len(_wl_fit_sp) > 0:
                    _wl_all_bc = _sd_bc.get("wavelengths", np.array([]))
                    _wl_mask_bc = (_wl_all_bc >= _wl_fit_sp[0]) & (_wl_all_bc <= _wl_fit_sp[-1])
                    _A_fit_bc   = _A_sp_bc[:, _wl_mask_bc]
                    _Cb_raw, _, _, _ = np.linalg.lstsq(E_final_sp.T, _A_fit_bc.T, rcond=None)
                    C_back_sp = np.clip(_Cb_raw.T, 0.0, None)
            except Exception:
                pass  # fall back to stored C_back
        if C_back_sp is not None and len(_x_raw_sp_bc) == C_back_sp.shape[0]:
            # x positions use current params so dots align with traces
            x_exp_sp = convert_exp_x(_x_raw_sp_bc, x_expr, parsed, params, network)
            _bc_uvvis = {sp: (x_exp_sp, C_back_sp[:, j]) for j, sp in enumerate(absorbers_sp)}
            _plot_backcalc_dots(fig, _bc_uvvis, plot_y_names,
                                parsed.get("variables", {}), network["all_species"],
                                trace_colors, label_suffix="(UV-Vis)")

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=40, b=80),
        xaxis=dict(title=x_label, range=[0, xmax]),
        yaxis=dict(title=_infer_y_label(plot_y_names, parsed, network), rangemode="tozero"),
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, width='stretch')
    _pub_download_button(fig, "equilibrium_main", y_label=_infer_y_label(plot_y_names, parsed, network))
    st.session_state["_current_figure"] = fig
    st.session_state.pop("_eq_snapshot_data", None)

    # ── UV-Vis secondary spectra plot ────────────────────────────
    if parsed.get("spectra") is not None:
        _sd_plot = st.session_state.get("_spectra_data", {})
        if _sd_plot:
            _wl_all  = _sd_plot["wavelengths"]
            _x_raw_sp = _sd_plot["x_vals"]     # mL (liquid) or x-axis values (solid)
            _A_sp    = _sd_plot["A"]            # (n_spectra, n_wl)
            _n_sp    = len(_x_raw_sp)

            # Convert to x-axis expression values for labels
            _x_sp = convert_exp_x(_x_raw_sp, x_expr, parsed, params, network)
            _x_unit = x_label   # e.g. "Ht/G0" or "V [mL]"

            # Wavelength filter from sidebar
            _wl_lo_p = float(st.session_state.get("spectra_wl_min", _wl_all[0]))
            _wl_hi_p = float(st.session_state.get("spectra_wl_max", _wl_all[-1]))
            _wl_mask_p = (_wl_all >= _wl_lo_p) & (_wl_all <= _wl_hi_p)
            _wl_plot   = _wl_all[_wl_mask_p]
            _A_plot    = _A_sp[:, _wl_mask_p]

            # Rainbow: violet (270°) → red (0°) mapped first → last spectrum
            def _rainbow(i, n):
                """HSL hue from 270 (violet) down to 0 (red), converted to hex."""
                import colorsys
                hue = (270 - (270 * i / max(n - 1, 1))) / 360.0
                r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.85)
                return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            fig_sp = go.Figure()
            for _i in range(_n_sp):
                _col = _rainbow(_i, _n_sp)
                _lbl = f"{_x_unit}={_x_sp[_i]:.3g}"
                fig_sp.add_trace(go.Scatter(
                    x=_wl_plot,
                    y=_A_plot[_i],
                    mode="lines",
                    line=dict(color=_col, width=1.5),
                    name=_lbl,
                    showlegend=False,
                    hovertemplate=f"{_lbl}<br>λ=%{{x:.0f}} nm<br>A=%{{y:.4f}}<extra></extra>",
                ))

            # Colorbar-style annotation: first and last label
            fig_sp.add_annotation(x=0.01, y=1.02, xref="paper", yref="paper",
                                  text=f"{_x_unit}={_x_sp[0]:.3g}",
                                  showarrow=False, font=dict(color=_rainbow(0, _n_sp), size=11),
                                  xanchor="left")
            fig_sp.add_annotation(x=0.99, y=1.02, xref="paper", yref="paper",
                                  text=f"{_x_unit}={_x_sp[-1]:.3g}",
                                  showarrow=False, font=dict(color=_rainbow(_n_sp-1, _n_sp), size=11),
                                  xanchor="right")

            fig_sp.update_layout(
                height=350,
                margin=dict(l=40, r=20, t=40, b=60),
                xaxis=dict(title="Wavelength [nm]"),
                yaxis=dict(title="Absorbance", rangemode="tozero"),
                template="plotly_dark",
                showlegend=False,
                title=dict(text="UV-Vis spectra", font=dict(size=13), x=0.5),
            )
            st.plotly_chart(fig_sp, width='stretch')
            _pub_download_button(fig_sp, "equilibrium_spectra", x_label="Wavelength [nm]", y_label="Absorbance")

            # ── Pure species spectra (only shown after a successful fit) ──
            _fit_stats_sp = st.session_state.get("_fit_stats", {})
            if _fit_stats_sp.get("fit_mode") == "spectra":
                _E_final   = _fit_stats_sp.get("E_final")        # (n_absorbers, n_wl_fit)
                _wl_fit_sp = _fit_stats_sp.get("wavelengths_fit")  # (n_wl_fit,)
                _absorbers = _fit_stats_sp.get("absorbers", [])

                if _E_final is not None and len(_absorbers) > 0:
                    # Assign a consistent color per species (match main plot palette)
                    _PALETTE = [
                        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
                        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
                        "#FF97FF", "#FECB52",
                    ]
                    fig_pure = go.Figure()
                    for _j, _sp in enumerate(_absorbers):
                        _col_p = _PALETTE[_j % len(_PALETTE)]
                        fig_pure.add_trace(go.Scatter(
                            x=_wl_fit_sp,
                            y=_E_final[_j],
                            mode="lines",
                            line=dict(color=_col_p, width=2),
                            name=_sp,
                            hovertemplate=f"{_sp}<br>λ=%{{x:.0f}} nm<br>ε=%{{y:.4f}} mM⁻¹<extra></extra>",
                        ))

                    fig_pure.update_layout(
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=60),
                        xaxis=dict(title="Wavelength [nm]"),
                        yaxis=dict(title="ε [mM⁻¹, path length absorbed]", rangemode="tozero"),
                        template="plotly_dark",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="left", x=0),
                        title=dict(text="Pure species spectra", font=dict(size=13), x=0.5),
                    )
                    st.plotly_chart(fig_pure, width='stretch')
                    _pub_download_button(fig_pure, "equilibrium_spectra_species", x_label="Wavelength [nm]", y_label="ε [mM⁻¹]")

    # ── NMR chemical shift plot ───────────────────────────────────
    nmr_cfg  = parsed.get("nmr")
    nmr_data = st.session_state.get("_nmr_data", {})

    if nmr_cfg is not None and nmr_cfg["mode"] in ("shift", "mixed") and nmr_data:
        fit_stats   = st.session_state.get("_fit_stats", {})
        delta_free  = fit_stats.get("delta_free", {})
        delta_b_all = fit_stats.get("delta_bound_all", {})
        delta_vecs_all = fit_stats.get("delta_vecs_all", {})

        fig_nmr = go.Figure()

        # Distinct colors per target (dots and lines share the same color per target)
        _NMR_PALETTE = ["#5B9BD5", "#ED7D31", "#70AD47", "#FF5C5C", "#9966CC", "#00B0CC"]
        nmr_target_colors = {tgt: _NMR_PALETTE[i % len(_NMR_PALETTE)]
                             for i, tgt in enumerate(nmr_cfg["targets"])}

        # For mixed mode, only columns after the first n_integ are shift columns
        _n_integ_plot = nmr_cfg.get("n_integ", 0)
        _all_cols_plot = [c for c in nmr_data if not c.startswith("_")]
        _shift_cols_plot = _all_cols_plot[_n_integ_plot:]

        # Map each shift column to its target variable
        col_to_target = {}
        for col in _shift_cols_plot:
            for tgt in nmr_cfg["targets"]:
                if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                    col_to_target[col] = tgt
                    break
            else:
                col_to_target[col] = nmr_cfg["targets"][0] if nmr_cfg["targets"] else col

        # ── Experimental Δδ dots ─────────────────────────────────────
        shown_tgt_legend = set()
        for col in _shift_cols_plot:
            col_data = nmr_data[col]
            exp_x  = convert_exp_x(col_data["v_add_mL"], x_expr, parsed, params, network)
            df0    = float(delta_free.get(col, col_data["y"][0]))
            delta_obs_rel = col_data["y"] - df0
            tgt    = col_to_target.get(col, col)
            color  = nmr_target_colors.get(tgt, "#888888")
            show_in_legend = tgt not in shown_tgt_legend
            shown_tgt_legend.add(tgt)
            fig_nmr.add_trace(go.Scatter(
                x=exp_x, y=delta_obs_rel,
                mode="markers",
                name=tgt if show_in_legend else col,
                legendgroup=tgt,
                showlegend=show_in_legend,
                marker=dict(color=color, size=6, symbol="circle"),
            ))

        # ── Theoretical Δδ curves — only shown after a fit has been run ─────
        # Pre-fit curves are meaningless (they reflect the current K slider value,
        # not a fit to the data) so we suppress them entirely.
        _nmr_has_been_fitted = bool(fit_stats.get("delta_vecs_all") or fit_stats.get("delta_free"))
        if _nmr_has_been_fitted:
            x_free_val_plot = fit_stats.get("x_free_val", {})
            for col in _shift_cols_plot:
                col_data  = nmr_data[col]
                tgt       = col_to_target.get(col, col)
                sp_coeffs = _get_species_for_target(tgt, parsed, network)
                if not sp_coeffs: continue

                df0      = float(delta_free.get(col, col_data["y"][0]))
                x_free_c = x_free_val_plot.get(col, 0.0)

                denom_full = np.maximum(
                    sum(coeff * curve.get(sp, np.zeros_like(x_vals))
                        for coeff, sp in sp_coeffs), 1e-20)
                denom_ref = float(np.maximum(
                    sum(coeff * float(np.interp(x_free_c, x_vals, curve.get(sp, np.zeros_like(x_vals))))
                        for coeff, sp in sp_coeffs), 1e-20))

                free_sp  = sp_coeffs[0][1]
                non_free = sp_coeffs[1:]

                if col in delta_vecs_all:
                    sp_dd = delta_vecs_all[col]
                    delta_calc_rel = np.zeros_like(x_vals)
                    for coeff, sp in non_free:
                        F_full = coeff * curve.get(sp, np.zeros_like(x_vals)) / denom_full
                        F_ref  = coeff * float(np.interp(x_free_c, x_vals,
                                    curve.get(sp, np.zeros_like(x_vals)))) / denom_ref
                        delta_calc_rel += (F_full - F_ref) * sp_dd.get(sp, 0.0)
                else:
                    continue  # post-fit only — skip if no fitted Δδ available

                fig_nmr.add_trace(go.Scatter(
                    x=x_vals, y=delta_calc_rel,
                    mode="lines",
                    name=f"{col} (calc)",
                    legendgroup=tgt,
                    showlegend=False,
                    line=dict(color=nmr_target_colors.get(tgt, "#444444"), width=2),
                ))

        fig_nmr.update_layout(
            height=350,
            margin=dict(l=40, r=20, t=30, b=80),
            xaxis=dict(title=x_label, range=[0, xmax]),
            yaxis=dict(title="Δδ [ppm]"),
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            title=dict(text="NMR Chemical Shift Changes", x=0.5, font=dict(size=13)),
        )
        st.plotly_chart(fig_nmr, width='stretch')
        _pub_download_button(fig_nmr, "equilibrium_nmr", y_label="Δδ [ppm]")


with col2:
    # ── Show fit message from previous run ───────
    _fit_msg = st.session_state.pop("_fit_message", None)
    if _fit_msg:
        level, text = _fit_msg
        if level == "success":
            st.success(text)
        else:
            st.warning(text)

    # ── Fitting ──────────────────────────────────
    nmr_cfg  = parsed.get("nmr")
    nmr_data = st.session_state.get("_nmr_data", {})
    exp_data = st.session_state.get("_exp_data", {})
    spectra_data = st.session_state.get("_spectra_data", {})
    has_nmr  = nmr_cfg is not None and bool(nmr_data)
    has_spectra = parsed.get("spectra") is not None and bool(spectra_data)
    has_exp_data = bool(exp_data) or has_nmr or has_spectra

    # Get which parameters are marked for fitting
    fit_keys = []
    for eq in parsed["equilibria"]:
        kname = eq["kname"]
        fit_key = f"fit_logK_{kname}"
        if st.session_state.get(fit_key, False):
            fit_keys.append(kname)

    # Render Tol/Timeout BEFORE the button so they are always rendered
    # before any st.rerun() call — prevents Streamlit from garbage-collecting
    # their session state when the button aborts the run mid-script.
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        fit_tolerance_log = _num_input(
            "Tol (−log)", key="fit_tolerance_log", default=6.0, step=0.5, format="%.1f"
        )
    with _fc2:
        fit_timeout = _num_input(
            "Timeout (s)", key="fit_timeout", default=30, step=5, format="%d"
        )

    # ── Optimizer selection ───────────────────────
    # Default: both checked → L-BFGS-B first, Nelder-Mead fallback.
    # User can uncheck one; at least one must remain checked.
    _oc1, _oc2 = st.columns(2)
    with _oc1:
        use_lbfgsb = st.checkbox("L-BFGS-B", key="fit_use_lbfgsb", value=True)
    with _oc2:
        use_neldermead = st.checkbox("Nelder-Mead", key="fit_use_neldermead", value=True)
    # Prevent unchecking both
    if not use_lbfgsb and not use_neldermead:
        st.warning("At least one optimizer must be selected.")
        use_lbfgsb = True

    _fit_conc_keys_btn = [root for root in
                          (cname[:-1] if cname.endswith("0") else cname
                           for cname in parsed["concentrations"])
                          if st.session_state.get(f"fit_conc_{root}", False)]
    _fit_titrant_keys_btn = [tkey for tkey in titrant_keys
                             if st.session_state.get(f"fit_titrant_mM_{tkey}", False)]
    fit_enabled = has_exp_data and (len(fit_keys) > 0 or
                                    len(_fit_conc_keys_btn) > 0 or
                                    len(_fit_titrant_keys_btn) > 0)
    if st.button("🔧 Fit Parameters", disabled=not fit_enabled):
        if fit_enabled:
            # Snapshot user-set fit prefs into shadow keys so st.rerun()
            # cannot lose them via Streamlit's widget-state cleanup
            for _pref in ("fit_tolerance_log", "fit_timeout"):
                if _pref in st.session_state:
                    st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
            st.session_state["_fit_requested"] = True
            st.rerun()
        else:
            st.info("Load experimental data and check parameters to fit.")

    # ── Export Data Button ──────────────────────
    st.subheader("Data Export")
    col_export, col_snapshot = st.columns(2)

    with col_export:
        try:
            script_text = st.session_state.get("_script_text", "")
            excel_data, filename = export_to_excel(curve, x_vals, parsed, params, network, script_text, logK_vals,
                                                    script_path=st.session_state.get("_script_filename"),
                                                    input_path=st.session_state.get("_input_filename"))
            st.download_button(
                label="💾 Export data",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    with col_snapshot:
        try:
            current_fig = st.session_state.get("_current_figure")
            if current_fig is not None:
                _eq_snap_bytes, _eq_snap_fname = create_snapshot(
                    current_fig, parsed, params, logK_vals,
                    xmax=float(x_vals[-1]) if len(x_vals) > 0 else None,
                    x_label=x_label,
                    y_label=_infer_y_label(plot_y_names, parsed, network))
                st.download_button(
                    label="📸 Snapshot",
                    data=_eq_snap_bytes,
                    file_name=_eq_snap_fname,
                    mime="application/pdf",
                    width='stretch',
                )
        except Exception as _se:
            st.error(f"Snapshot failed: {_se}")
    # ── Warning Popup for Solver Issues ────────
    n_warn = int(np.sum(warn))
    if n_warn > 0:
        max_rn = float(np.max(curve["resid_norm"]))
        if max_rn > 1e-6:
            st.warning(f"⚠️ **Solver Warning**: {n_warn}/{len(x_vals)} points failed to converge properly (max residual: {max_rn:.2e})")
        else:
            st.warning(f"⚠️ {n_warn}/{len(x_vals)} points had minor solver warnings (max residual: {max_rn:.2e})")

    # ── Experimental data ──────────────────────
    st.subheader("Experimental data")

    if st.button("↺ Reset experimental data"):
        for k in list(st.session_state.keys()):
            if (k.startswith("_exp") or k.startswith("_nmr") or
                    k.startswith("_spectra") or k.startswith("_fit") or
                    k.startswith("_pending_logK") or k == "_fit_requested" or
                    k == "_input_filename"):
                del st.session_state[k]
        st.rerun()

    if "_exp_uploader_nonce" not in st.session_state:
        st.session_state["_exp_uploader_nonce"] = 0

    # ── Single experimental data uploader ────────────────────────
    # Routes to _nmr_data when $nmr is present,
    #           _spectra_data when $spectra is present,
    #           _exp_data otherwise.
    _is_solid_up = parsed.get("titrant_is_solid", False)
    _col_a_hint  = "column A = x-axis values (equivalents)" if _is_solid_up else "column A = volume added (mL)"

    if parsed.get("spectra") is not None:
        _uploader_label = "Upload experimental data (.xlsx)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; row 1 = wavelengths (nm); body = absorbance"
    elif nmr_cfg is not None:
        if nmr_cfg["mode"] == "integration":
            st.caption(f"NMR mode: **integration** (slow exchange) — "
                       f"{len(nmr_cfg.get('n_H_list', []))} signals")
        else:
            st.caption(f"NMR mode: **{nmr_cfg['mode']}** — targets: {', '.join(nmr_cfg['targets'])}")
        _uploader_label = "Upload experimental data (.xlsx)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; columns B+ = NMR observables"
    else:
        _uploader_label = "Upload experimental data (.xlsx)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; columns B+ = species concentrations (mM)"

    st.caption(_uploader_hint)
    _uploaded = st.file_uploader(
        _uploader_label, type=["xlsx"],
        key=f"_exp_uploader_{st.session_state['_exp_uploader_nonce']}",
    )
    if _uploaded is not None:
        try:
            if parsed.get("spectra") is not None:
                _loaded = load_spectra_data(_uploaded.read())
                st.session_state["_spectra_data"] = _loaded
                # Override xmax with the x-value of the last data point
                if _loaded:
                    _x_last_mL = float(_loaded["x_vals"][-1])
                    _x_last    = convert_exp_x(
                        np.array([_x_last_mL]), x_expr_default,
                        parsed, params, network)[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last * 10) / 10)
            elif nmr_cfg is not None:
                _loaded = load_experimental_data(_uploaded.read())
                st.session_state["_nmr_data"] = _loaded
                # Update xmax from last data point (x-axis in mL, converted via x_expr)
                _eq_nmr_cols = [c for c in _loaded if not c.startswith("_")]
                if _eq_nmr_cols:
                    _x_last_mL_nmr = float(_loaded[_eq_nmr_cols[0]]["v_add_mL"][-1])
                    _x_last_nmr = convert_exp_x(
                        np.array([_x_last_mL_nmr]), x_expr_default,
                        parsed, params, network)[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_nmr * 10) / 10)
            else:
                _loaded = load_experimental_data(_uploaded.read())
                st.session_state["_exp_data"] = _loaded
                # Update xmax from last data point
                _eq_cols = [c for c in _loaded if not c.startswith("_")]
                if _eq_cols:
                    _x_last_mL_eq = float(_loaded[_eq_cols[0]]["v_add_mL"][-1])
                    _x_last_eq = convert_exp_x(
                        np.array([_x_last_mL_eq]), x_expr_default,
                        parsed, params, network)[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_eq * 10) / 10)
            st.session_state["_input_filename"] = _uploaded.name
            st.session_state["_exp_uploader_nonce"] += 1
            st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.get("_input_filename"):
        st.markdown(f"<span style='background:#1a6bbf;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:0.82rem'>📄 {st.session_state['_input_filename']}</span>", unsafe_allow_html=True)

    # Show summary of what's loaded
    if parsed.get("spectra") is not None:
        _sd_loaded = st.session_state.get("_spectra_data", {})
        if _sd_loaded:
            st.caption(f"Loaded: {len(_sd_loaded['x_vals'])} spectra × {len(_sd_loaded['wavelengths'])} wavelengths "
                       f"({_sd_loaded['wavelengths'][0]:.0f}–{_sd_loaded['wavelengths'][-1]:.0f} nm)")
    elif nmr_cfg is not None:
        nmr_data_loaded = st.session_state.get("_nmr_data", {})
        if nmr_data_loaded:
            n_sig = sum(1 for k in nmr_data_loaded if not k.startswith("_"))
            n_pts = next((len(v["y"]) for k, v in nmr_data_loaded.items() if not k.startswith("_")), 0)
            st.caption(f"Loaded: {n_sig} signals × {n_pts} points")
    else:
        exp_data_loaded = st.session_state.get("_exp_data", {})
        if exp_data_loaded:
            total_pts = sum(len(v["v_add_mL"]) for k, v in exp_data_loaded.items() if not k.startswith("_"))
            st.caption(f"Loaded: {', '.join(k for k in exp_data_loaded if not k.startswith('_'))} ({total_pts} pts)")

    # ── Fit statistics ─────────────────────────
    fit_stats = st.session_state.get("_fit_stats", {})
    if fit_stats:
        fit_mode_disp = fit_stats.get("fit_mode", "")
        st.write("**Fit Statistics:**")

        if fit_mode_disp == "mixed":
            # Show per-component stats — never mix mM and ppm into one number
            r2_i    = fit_stats.get("r2_integ", 0.0)
            rmse_i  = fit_stats.get("rmse_integ", 0.0)
            n_i     = fit_stats.get("n_integ_pts", 0)
            r2_s    = fit_stats.get("r2_shift", 0.0)
            rmse_s  = fit_stats.get("rmse_shift", 0.0)
            n_s     = fit_stats.get("n_shift_pts", 0)
            st.write(f"**Integration fit** ({n_i} points)")
            st.write(f"• R² = {r2_i:.4f}")
            st.write(f"• RMSE = {rmse_i:.2e} mM")
            st.write(f"**Shift fit** ({n_s} points)")
            st.write(f"• R² = {r2_s:.4f}")
            st.write(f"• RMSE = {rmse_s:.2e} ppm")
            n_total_disp = fit_stats.get("n_points", n_i + n_s)
            st.write(f"• Total data points: {n_total_disp}")
            if n_total_disp > fit_stats.get("n_params", 0):
                st.write(f"• Parameters fitted: {fit_stats.get('n_params', '?')}")
            if "n_iter" in fit_stats:
                _evals = fit_stats["n_iter"]
                st.write(f"• Objective evaluations: {_evals}")
        elif fit_mode_disp == "spectra":
            # ── Timeout / correlation warnings ─────────────────
            if fit_stats.get("timed_out"):
                st.warning(
                    "⏱️ **Fit timed out** — showing best parameters found within "
                    f"{int(st.session_state.get('fit_timeout', 30))} s. "
                    "Results may not be fully converged.")
            if fit_stats.get("is_correlated"):
                cond = fit_stats.get("cond_number", 0)
                combos = fit_stats.get("identifiable", [])
                warn_lines = [
                    f"⚠️ **Parameters are highly correlated** (condition number = {cond:.1e}). "
                    "Individual K values cannot be reliably extracted from these data. "
                    "The spectra of the complexes are too similar to distinguish them."
                ]
                if combos:
                    warn_lines.append("\n**What IS well-determined:**")
                    for label, val, se, ratio in combos:
                        warn_lines.append(f"- **{label}** = {val:.3f} ± {se:.3f}  "
                                          f"(eigenvalue ratio {ratio:.2e})")
                st.warning("\n".join(warn_lines))
            # ── Standard stats ──────────────────────────────────
            st.write(f"• RMSE = {fit_stats['rmse']:.2e} AU")
            if fit_stats.get("r2_conc") is not None:
                st.write(f"• Concentration R² = {fit_stats['r2_conc']:.4f}  "
                         f"*(data points vs theoretical traces)*")
                st.write(f"• Concentration RMSE = {fit_stats['rmse_conc']:.2e} mM")
            st.write(f"• Data points: {fit_stats['n_points']} (spectra × wavelengths)")
            if fit_stats['n_points'] > fit_stats['n_params']:
                reduced_chi2 = fit_stats['ssr'] / (fit_stats['n_points'] - fit_stats['n_params'])
                st.write(f"• Reduced χ² = {reduced_chi2:.2e}")
            _wl_fit = fit_stats.get("wavelengths_fit", np.array([]))
            if len(_wl_fit) > 0:
                if fit_stats.get("auto_range"):
                    st.write(f"• Auto-optimized range: {fit_stats['opt_wl_min']:.0f}–{fit_stats['opt_wl_max']:.0f} nm")
                else:
                    st.write(f"• Fit range: {_wl_fit[0]:.0f}–{_wl_fit[-1]:.0f} nm")
            if "n_iter" in fit_stats:
                _evals = fit_stats["n_iter"]
                st.write(f"• Objective evaluations: {_evals}")
        else:
            st.write(f"• R² = {fit_stats['r_squared']:.4f}")
            rmse_unit = " ppm" if fit_mode_disp == "shift" else " mM" if fit_mode_disp == "integration" else ""
            st.write(f"• RMSE = {fit_stats['rmse']:.2e}{rmse_unit}")
            st.write(f"• Data points: {fit_stats['n_points']}")
            if fit_stats['n_points'] > fit_stats['n_params']:
                reduced_chi2 = fit_stats['ssr'] / (fit_stats['n_points'] - fit_stats['n_params'])
                st.write(f"• Reduced χ² = {reduced_chi2:.2e}")
            if "n_iter" in fit_stats:
                _evals = fit_stats["n_iter"]
                st.write(f"• Objective evaluations: {_evals}")

        # ── Fitted equilibrium constants ─────────────────────────
        param_values    = fit_stats.get("param_values", {})
        param_errors    = fit_stats.get("param_errors", {})
        fitted_concs_eq = fit_stats.get("fitted_concs", {})
        fitted_tits_eq  = fit_stats.get("fitted_titrants", {})
        if param_values or fitted_concs_eq or fitted_tits_eq:
            st.write("**Fitted constants:**")
            rows = []
            # log-space parameters (K values)
            for kname, val in param_values.items():
                err     = param_errors.get(kname)
                k_lin   = 10.0 ** val
                err_lin = k_lin * 2.302585 * err if err is not None else None
                rows.append({"Parameter": kname,
                             "log P":  f"{val:.2f}",
                             "±log P": f"± {err:.2f}" if err is not None else "n/a",
                             "P":      f"{k_lin:.2e}",
                             "±P":     f"± {err_lin:.2e}" if err_lin is not None else "n/a"})
            # linear-space concentration parameters (mM)
            for root, mM_val in fitted_concs_eq.items():
                err_c = param_errors.get(root)
                rows.append({"Parameter": root,
                             "log P":  "—",
                             "±log P": "—",
                             "P":      f"{mM_val:.2e} mM",
                             "±P":     f"± {err_c:.2e} mM" if err_c is not None else "n/a"})
            # linear-space titrant concentration parameters (mM)
            for tkey, mM_val in fitted_tits_eq.items():
                err_t = param_errors.get(tkey)
                rows.append({"Parameter": tkey,
                             "log P":  "—",
                             "±log P": "—",
                             "P":      f"{mM_val:.2e} mM",
                             "±P":     f"± {err_t:.2e} mM" if err_t is not None else "n/a"})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Parameter"), width='stretch')

            # ── Free energy table ─────────────────────────────────────────
            if param_values:
                import math as _math
                _T    = float(parsed.get("temperature_K", 298.15))
                _R    = 1.987e-3   # kcal/(mol·K)
                _energy_rows_eq = []
                for _kn, _val in param_values.items():
                    _err = param_errors.get(_kn)
                    _dG = -_R * _T * _math.log(10) * _val
                    _dG_err = _R * _T * _math.log(10) * _err if _err is not None else None
                    _energy_rows_eq.append({"Parameter": _kn,
                                            "ΔG° (kcal/mol)": f"{_dG:.2f}",
                                            "±ΔG°": f"± {_dG_err:.2f}" if _dG_err is not None else "n/a"})
                if _energy_rows_eq:
                    st.write(f"**Free energies ({_T:.2f} K):**")
                    st.dataframe(pd.DataFrame(_energy_rows_eq).set_index("Parameter"), width='stretch')

        # ── Read fit-mode-aware stats ─────────────────────────────────────
        fit_mode         = fit_stats.get("fit_mode", "")
        sp_concs_disp    = fit_stats.get("sp_concs", {})
        col_to_sp_disp   = fit_stats.get("col_to_sp", {})
        col_to_nH_disp   = fit_stats.get("col_to_nH", {})
        pure_shifts_disp = fit_stats.get("pure_shifts", {})  # from fit, not page var

        # ── Integration back-calculated concentrations ────────────────────────
        # Show whenever integration data was fitted (integration or mixed mode)
        if sp_concs_disp:
            st.write("**NMR signal assignment:**")
            rows_bc = []
            for sp, arr_list in sp_concs_disp.items():
                # Count how many raw signals contributed to this species average
                n_signals = sum(1 for col, csp in col_to_sp_disp.items() if csp == sp)
                n_H_vals  = sorted(set(
                    col_to_nH_disp.get(col, "?")
                    for col, csp in col_to_sp_disp.items() if csp == sp))
                rows_bc.append({
                    "Species":        sp,
                    "# signals used": max(n_signals, 1),
                    "n_H per signal": ", ".join(
                        str(int(v) if isinstance(v, float) and v == int(v) else v)
                        for v in n_H_vals) or "1",
                })
            if rows_bc:
                st.dataframe(pd.DataFrame(rows_bc).set_index("Species"),
                             width='stretch')
                st.caption("Concentrations averaged from all signals of each species")

        # ── Pure-species chemical shifts ──────────────────────────────────────
        # Show whenever shift data was fitted (shift or mixed mode)
        if pure_shifts_disp:
            st.write("**Pure-species chemical shifts (ppm):**")
            all_sp_cols = []
            rows_ps = []
            for tgt, col_dict in pure_shifts_disp.items():
                for col, sp_dict in col_dict.items():
                    row = {"Signal": col}
                    for sp, delta in sp_dict.items():
                        row[sp] = f"{delta:.4f}"
                        if sp not in all_sp_cols:
                            all_sp_cols.append(sp)
                    rows_ps.append(row)
            if rows_ps:
                df_ps = pd.DataFrame(rows_ps).set_index("Signal")
                ordered = [c for c in all_sp_cols if c in df_ps.columns]
                st.dataframe(df_ps[ordered], width='stretch')
                st.caption("Each row = one NMR signal; columns = δ of each pure species (ppm)")

        # ── Mixed mode: summary ──────────────────────────────────────────────
        if fit_mode == "mixed":
            integ_sp = list(sp_concs_disp.keys())
            shift_tgts = list(pure_shifts_disp.keys())
            st.caption(
                f"Mixed fit: slow-exchange integrations [{', '.join(integ_sp)}] + "
                f"fast-exchange shifts [{', '.join(shift_tgts)}] fitted simultaneously.")

with col1:
    _render_script_editor()

# ── Execute fit AFTER all widgets have rendered ───────────────────────────
# This is the ONLY correct place to run the fit and call st.rerun():
# every widget (logK inputs, checkboxes, tol/iter) has been rendered above,
