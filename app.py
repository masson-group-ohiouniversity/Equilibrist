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
from equilibrist_curve import _find_maxEquiv, _x_per_equiv
from equilibrist_fit_conc import *
from equilibrist_fit_nmr import *
from equilibrist_fit_nmr import shift_backcalc_resolved   # not in the module's __all__
from equilibrist_fit_spectra import *
from equilibrist_io import *
from equilibrist_io import _itc_pub_figure_bytes   # underscore-prefixed: not picked up by import *
from equilibrist_itc import *
# ── Statistical diagnostics and bootstrap CIs (added in v2) ───────────────
# Drop-in modules.  See INTEGRATION_PATCHES.md for the rationale.
from equilibrist_diagnostics import (augment_stats, render_diagnostics_panel,
                                     collect_residuals_from_stats,
                                     collect_residuals_conc,
                                     collect_residuals_nmr_shift,
                                     collect_residuals_spectra,
                                     collect_residuals_kinetics,
                                     compare_models, make_diagnostics_figure,
                                     compute_rmse_profile,
                                     compute_rmse_profile_2d,
                                     make_2d_profile_figure,
                                     compute_profile_trace,
                                     make_profile_trace_figure,
                                     render_identifiability_panel,
                                     render_rank_analysis_panel,
                                     compute_local_sensitivity)
import equilibrist_bootstrap as ebs
import equilibrist_hpc_ui as hpc_ui   # supercomputer (SLURM) submission panel
import equilibrist_pyscript as pyscript   # user's custom .py script (sidebar buttons)
import equilibrist_tabs as eqtabs           # several workspaces ("tabs") in one session
import equilibrist_global as eqglobal       # one fit across several tabs
import equilibrist_live as eqlive           # live view of a running fit


# ── Progress bar shown while a fit runs ─────────────────────────────────────
# The optimizer has no known number of steps, so the bar shows elapsed time
# against the Timeout, which is what bounds a fit; it stops early on
# convergence.  The bar is advanced by a small helper thread (the fit blocks
# the script thread), attached to Streamlit's run context so it may draw.
import contextlib as _contextlib
import threading as _threading
import time as _time


def _pinned_shift_mismatch(nmr_cfg, nmr_data, tol_ppm=0.004):
    """When a script pins intrinsic shifts (``read:`` + sheet 2), compare them
    with the shifts observed at the first titration point.  Returns
    ``(species, median_offset_ppm, n_cols)`` when the pinned values of the
    species closest to the data are still further than ``tol_ppm`` away
    (a referencing mismatch between the data and the pinned sheet), else None."""
    try:
        if not nmr_cfg or not nmr_cfg.get("read") or not nmr_data:
            return None
        known = nmr_data.get("_known_shifts") or {}
        if not known:
            return None
        cols = [c for c in nmr_data if not str(c).startswith("_")]
        mode = nmr_cfg.get("mode")
        if mode == "integration":
            return None
        if mode == "mixed":
            cols = cols[int(nmr_cfg.get("n_integ", len(nmr_cfg.get("n_H_list", []) or [])) or 0):]
        obs0 = []
        for c in cols:
            v = np.asarray(nmr_data[c].get("v_add_mL", []), dtype=float)
            y = np.asarray(nmr_data[c].get("y", []), dtype=float)
            ok = np.isfinite(v) & np.isfinite(y)
            obs0.append(float(y[ok][np.argmin(v[ok])]) if ok.any() else np.nan)
        obs0 = np.asarray(obs0, dtype=float)
        best = None
        for sp in nmr_cfg["read"]:
            kv = np.asarray(known.get(sp, []), dtype=float)
            n = min(len(kv), len(obs0))
            if n == 0:
                continue
            off = obs0[:n] - kv[:n]
            off = off[np.isfinite(off)]
            if off.size == 0:
                continue
            med = float(np.median(off))
            if best is None or abs(med) < abs(best[1]):
                best = (sp, med, int(off.size), bool(np.all(np.sign(off) == np.sign(med))))
        if best is None or abs(best[1]) <= tol_ppm:
            return None
        return best
    except Exception:
        return None


def _fit_end_message(stats, success, message, n_fitted, tol_log, timeout_s):
    """(level, text) describing how a fit ended, from the stats the NMR
    fitters record (stop_reason / n_evals / final_spread / elapsed_s).
    Falls back to the older wording for fitters that do not record them."""
    level, text = _fit_end_message_core(stats, success, message, n_fitted, tol_log, timeout_s)
    ab = stats.get("at_bound") or {}
    if ab:
        parts = [f"{n} at the {side} bound of its range (log = {b:g})" for n, (side, b) in ab.items()]
        text += ("  ⚠️ " + "; ".join(parts) + " — the fit wanted to go further; widen the range with "
                 "'from X to Y' on that reaction line if that is not intended.")
    return level, text


def _fit_end_message_core(stats, success, message, n_fitted, tol_log, timeout_s):
    reason = stats.get("stop_reason")
    if reason is None:
        n_ev = stats.get("n_iter", 0)
        if stats.get("timed_out"):
            return ("warning", f"⏱️ Fit timed out after {n_ev} evaluations ({timeout_s:.0f} s limit) — "
                               "showing best result found. Consider increasing Timeout.")
        if success:
            return ("success", f"Fit completed! Updated {n_fitted} parameters. (tol=1e-{tol_log:.0f})")
        return ("warning", f"Fit did not fully converge — {message}")
    n_ev = stats.get("n_evals") or stats.get("n_iter", 0)
    el = float(stats.get("elapsed_s") or 0.0)
    tol = 10.0 ** (-float(tol_log))
    sp = stats.get("final_spread")
    moving = (f"the constants were still moving by up to {sp:.2g} log units in the last iterations"
              if sp is not None else "the constants were still moving")
    if reason == "timeout":
        return ("warning",
                f"⏱️ Stopped by the {timeout_s:.0f} s Timeout after {n_ev} evaluations — not converged: "
                f"{moving} (Tol asks for {tol:.0e}).  Press Fit again to continue from these values, "
                f"raise the Timeout, or relax Tol.")
    if reason == "tolerance":
        return ("success",
                f"✅ Converged after {n_ev} evaluations in {el:.0f} s (Tol = {tol:.0e}); the constants "
                f"moved by less than {sp:.1e} log units over the last iterations.  Updated {n_fitted} parameters."
                if sp is not None else
                f"✅ Converged after {n_ev} evaluations in {el:.0f} s (Tol = {tol:.0e}).  "
                f"Updated {n_fitted} parameters.")
    return ("warning", f"Stopped at the iteration cap after {n_ev} evaluations; {moving}.")


@_contextlib.contextmanager
def _fit_progress(slot, timeout_s, label="Fitting parameters…"):
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx as _add_ctx
    except Exception:                                   # very old Streamlit: plain spinner
        with slot, st.spinner(label):
            yield
        return
    timeout_s = max(float(timeout_s or 0), 1.0)
    t0 = _time.time()
    stop = _threading.Event()
    try:
        bar = slot.progress(0.0, text=f"{label}  0 s of {timeout_s:.0f} s")
    except Exception:
        bar = None

    def _tick():
        while bar is not None and not stop.wait(0.5):
            el = _time.time() - t0
            try:
                bar.progress(min(el / timeout_s, 0.99),
                             text=f"{label}  {el:.0f} s of {timeout_s:.0f} s "
                                  f"(ends early when converged)")
            except Exception:
                break

    th = _threading.Thread(target=_tick, daemon=True)
    _add_ctx(th)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1.0)
        try:
            slot.empty()
        except Exception:
            pass


# ── Fits run in the background; the page stays live ─────────────────────
def _live_refresh():
    """The 'Refresh (s)' setting (0 = no intermediate redraws)."""
    try:
        return float(st.session_state.get("_shadow_fit_refresh", st.session_state.get("fit_refresh", 60)) or 0)
    except Exception:
        return 0.0


def _fit_turn(slot, flag="_fit_requested"):
    """Whether the fit code behind ``flag`` runs on this script run.

    A fit started by :func:`_fit_async` runs in a background thread.  While it
    is alive the request stays pending, a small fragment in ``slot`` keeps the
    progress bar moving and, every 'Refresh (s)' seconds, writes the best
    constants found so far into the sidebar and reruns the page -- the normal
    plots update in place, nothing else changes -- and False is returned so the
    rest of the page renders as usual.  Once the thread has finished (or when
    no fit is running) the request is consumed and True returned: the fit code
    then collects the result (or starts the fit)."""
    ss = st.session_state
    for k, v in (ss.pop("_bg_prefs_restore", None) or {}).items():   # after a fit's final rerun
        ss.setdefault(k, v)
    bg = ss.get("_bg_fit")
    if bg is not None:
        # the fit boxes of the sidebar: in the kinetics branch that sidebar renders
        # AFTER the fit code, so the early reruns of a background fit drop their
        # widget state and the fit block would find nothing ticked
        for k, v in (bg.get("ticks") or {}).items():
            ss.setdefault(k, v)
    if bg is not None and bg["thread"].is_alive():
        if bg.get("tab_idx") is not None and ss.get("_tab_idx") != bg["tab_idx"]:
            ss["_tab_switch_to"] = bg["tab_idx"]          # the fit belongs to the tab it started in
            st.toast("Tabs are locked while a fit is running.")
            st.rerun()
        for k, v in (bg.get("prefs") or {}).items():       # checkboxes below the fit code keep their state
            ss.setdefault(k, v)
        if bg["flag"] == flag:
            _bg_fit_watch(slot, bg)
        elif ss.get(flag):
            ss.pop(flag, None)
            st.toast("A fit is already running; wait for it to finish.")
        return False
    if bg is not None and bg["flag"] == flag and ss.get(flag):
        bg["collect_at"] = _time.time()           # this run collects the finished fit
    return bool(ss.pop(flag, False))


def _bg_fit_watch(slot, bg):
    """Progress bar of the background fit, refreshed every second by a fragment
    (the rest of the page is untouched); triggers the periodic page reruns."""
    @st.fragment(run_every=1.0)
    def _tick():
        bg = st.session_state.get("_bg_fit")
        if bg is None:
            return
        el = _time.time() - bg["t0"]
        rec = bg["rec"]
        info = f"  ·  {rec['n']} evaluations"
        if rec.get("detail"):
            info += f", {rec['detail']}"
        elif np.isfinite(rec["best_f"]):
            info += f", {rec.get('label', 'objective')} {rec['best_f']:.4g}"
        if bg["refresh"] > 0:
            info += f"  ·  plots refresh every {bg['refresh']:.0f} s"
        st.progress(min(el / max(bg["timeout"], 1.0), 0.99),
                    text=f"{bg['label']}  {el:.0f} s of {bg['timeout']:.0f} s (ends early when converged){info}")
        if not bg["thread"].is_alive():
            st.rerun(scope="app")                         # collect the result
        elif bg["refresh"] > 0 and _time.time() >= bg["next"]:
            bg["next"] = _time.time() + bg["refresh"]
            best = {k: float(v) for k, v in (rec.get("best") or {}).items()}
            for k, v in best.items():
                if k in bg["start_logK"]:
                    st.session_state[f"_pending_logK_{k}"] = v
            # the NMR shift and spectra plots draw their model from the fit statistics
            # (pure-species shifts, absorptivities): evaluate them at the best point
            fa = st.session_state.get("_fit_args_live")
            if isinstance(fa, dict) and best:
                try:
                    lk = dict(fa.get("start_logK") or fa.get("start_logk") or {})
                    lk.update({k: v for k, v in best.items() if k in lk})
                    keys = list(fa.get("fit_keys") or []) or ([next(iter(lk))] if lk else [])
                    out = eqglobal._call_fit(fa, lk, keys, tolerance=1e30, maxiter=1, timeout_s=1e9,
                                             compute_hessian=False, objective_only=False)
                    if isinstance(out, tuple) and len(out) == 4 and isinstance(out[2], dict) and out[2]:
                        st.session_state["_fit_stats"] = out[2]
                except Exception:
                    pass
            st.rerun(scope="app")                         # redraw the page at the best point so far
    with slot.container():
        _tick()


def _fit_async(slot, timeout_s, run_fit, start_logK, label="Fitting parameters…", flag="_fit_requested"):
    """Run ``run_fit()`` -- a fit call returning (ok, fitted, stats, message) --
    in a background thread.  First call: start the thread, keep ``flag`` set so
    the call site is entered again, and rerun; the page then renders normally
    while :func:`_fit_turn` keeps the request pending and redraws the plots at
    the best constants every 'Refresh (s)' seconds.  Final call (thread
    finished): return the fit's result to the call site, which handles it
    exactly as it did when the fit was synchronous."""
    ss = st.session_state
    bg = ss.get("_bg_fit")
    if bg is not None and not bg["thread"].is_alive() and not (
            bg.get("flag") == flag and _time.time() - float(bg.get("collect_at") or 0.0) < 60.0):
        ss.pop("_bg_fit", None)                   # a stale, never collected fit: start afresh
        st.toast("The previous fit's result was not collected; starting a new fit.")
        bg = None
    if bg is not None and not bg["thread"].is_alive():
        ss.pop("_bg_fit", None)
        # the call site ends with an early st.rerun(): widgets rendered after it
        # (Tol, Timeout, Refresh, the global-fit checkboxes) would lose their state
        for k in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
            if k in ss:
                ss[f"_shadow_{k}"] = ss[k]
        ss["_bg_prefs_restore"] = {k: ss[k] for k in ("_global_weighted", "_global_share_shifts",
                                                       "_global_share_shifts_touched") if k in ss}
        # the constants and concentrations as they stand now: those the fit did not
        # change come back after the final rerun; the fitted ones arrive through
        # _pending_logK_ and take precedence (consumed first, at the top of the run)
        ss["_bg_prefs_restore"].update({k: ss[k] for k in (bg.get("ticks") or {}) if k in ss})
        if bg["box"].get("error") is not None:
            raise bg["box"]["error"]
        return bg["box"]["result"]
    if bg is not None:                                    # still running: not our turn
        ss[flag] = True
        st.rerun()
    rec = {"best": None, "best_f": float("inf"), "n": 0, "label": "objective"}
    box = {"result": None, "error": None}
    ident = {"id": None}

    def recorder(values, f, authoritative=False, detail=None):   # called by the fit's objective (fit thread only)
        if _threading.get_ident() != ident["id"]:
            return
        rec["n"] += 1
        if authoritative:                                 # the global fit reports its own current best
            rec["best_f"], rec["best"], rec["label"] = float(f), dict(values), "rms objective (plain sum of the tabs)"
            rec["detail"] = detail
        elif f < rec["best_f"]:
            rec["best_f"], rec["best"] = float(f), dict(values)

    def worker():
        ident["id"] = _threading.get_ident()
        try:
            with eqlive.live_hook(recorder):
                box["result"] = run_fit()
        except BaseException as ex:                       # handed back to the call site
            box["error"] = ex

    refresh = _live_refresh()
    # the sidebar state the early reruns drop (kinetics: the sidebar renders after
    # the fit code): fit boxes, and the constants / concentrations themselves
    _ticks = {k: ss[k] for k in list(ss.keys())
              if isinstance(k, str) and (k.startswith(("fit_logK_", "fit_conc_", "fit_titrant_mM_",
                                                       "logK_", "conc_", "titrant_mM_")) or k == "xmax")}
    bg = {"thread": _threading.Thread(target=worker, daemon=True), "box": box, "rec": rec, "ticks": _ticks,
          "t0": _time.time(), "timeout": max(float(timeout_s or 0), 1.0), "label": label, "flag": flag,
          "refresh": refresh, "next": (_time.time() + refresh) if refresh > 0 else float("inf"),
          "tab_idx": ss.get("_tab_idx"),
          "start_logK": {k: float(v) for k, v in (start_logK or {}).items()},
          "prefs": {k: ss[k] for k in ("_global_weighted", "_global_share_shifts", "_global_share_shifts_touched") if k in ss}}
    ss["_bg_fit"] = bg
    bg["thread"].start()
    ss[flag] = True
    st.rerun()

# ── v2: fragment wrapper ──────────────────────────────────────────────────
# Use st.fragment where available so interacting with widgets inside the
# diagnostics / bootstrap / model-comparison panels reruns ONLY that panel
# and preserves the page scroll position.  Falls back to a no-op decorator
# on Streamlit < 1.33 so the app still runs (just with the old behaviour).
if hasattr(st, 'fragment'):
    _fragment = st.fragment
elif hasattr(st, 'experimental_fragment'):
    _fragment = st.experimental_fragment
else:
    _fragment = lambda f: f


def _fit_ticks_crossed_zero(fit_key: str) -> None:
    """on_change of a fit box.  The Fit button's state, the global-fit tab list and
    the live fit args are computed by the main script, which a fragment click does
    not rerun; one full rerun is requested when this click makes the tab fittable
    (first box ticked) or unfittable (last box unticked).  Other clicks keep the
    scroll position, which is what the fragment is for."""
    ss = st.session_state
    n = sum(1 for k, v in ss.items() if isinstance(k, str) and v is True
            and k.startswith(("fit_logK_", "fit_conc_", "fit_titrant_mM_")))
    if (ss.get(fit_key) and n == 1) or (not ss.get(fit_key) and n == 0):
        ss["_fit_ticks_rerun"] = True


@_fragment
def _fit_toggle(fit_key: str, *, disabled: bool = False):
    """Render a sidebar "fit" checkbox inside its own fragment.

    Streamlit reruns the whole script on any widget change, which re-renders
    the main column and drops the reader back to the top of the page - so
    every fit-checkbox click meant scrolling down to the thermogram again.

    A fragment reruns ONLY itself, leaving the main column's DOM untouched,
    so the scroll position simply never moves. This is safe precisely for
    these checkboxes because they do not affect the plotted curve: they only
    record which parameters to float, and that is read on the next full
    rerun, which is exactly when the Fit button is pressed.

    Number inputs cannot be treated this way - changing a concentration must
    redraw the curve, which requires a full rerun.
    """
    st.checkbox("fit", key=fit_key, disabled=disabled,
                on_change=_fit_ticks_crossed_zero, args=(fit_key,))
    if st.session_state.pop("_fit_ticks_rerun", False):
        try:
            st.rerun(scope="app")
        except TypeError:                     # older Streamlit without fragments/scope
            st.rerun()


def _indented_checkbox(label, *, indent_frac: float = 0.035, **kwargs):
    """Render an st.checkbox visually indented under a parent expander.

    Used for child toggles inside "Show residual diagnostics" (and any
    similar parent) so the child boxes read as children, not as siblings
    of the parent.  Eric's feedback from the tutorial 1-11 debugging
    pass (debugging.docx item 0): "the 3 tests inside 'show residual
    diagnostics' should have a tab, so that their boxes are underneath
    the icon of 'show residual diagnostics'."

    Implementation: split the row into a thin empty left column and a
    wide right column holding the checkbox.  Streamlit doesn't allow
    per-widget CSS targeting by key, so this column-pair pattern is the
    cleanest minimum-invasive way to indent only the checkbox itself
    without restructuring the body code that lives under
    ``if _show_…:``.

    Returns the checkbox value (matching ``st.checkbox``'s contract).
    """
    _spacer, _content = st.columns([indent_frac, 1.0 - indent_frac])
    with _content:
        return st.checkbox(label, **kwargs)


def _write_tuple_cache(key: tuple, value) -> None:
    """Store a value under a tuple key in session_state AND record the
    key in a registry list so ``_serialize_tuple_caches`` can find it
    at save time.

    Background: ``st.session_state[tuple_key] = value`` works (direct
    lookup via ``.get(tuple_key)`` returns it), but
    ``st.session_state.keys()`` does NOT reliably yield tuple keys to
    external iteration in Streamlit's ``SessionStateProxy``.  So the
    Prepare-bundle code can't enumerate the caches by walking
    ``session_state``; instead it walks this registry list (stored
    under a plain string key, which iteration *does* expose).

    Every cache write site (``_prof_cache_*``, ``_p2d_cache_*``)
    must go through this helper — otherwise the cache is alive
    during the session but invisible to the JSON serialiser, and
    figures fail to re-render on reload (the user sees
    ``profile caches: 0`` in the Prepare-bundle caption).
    """
    st.session_state[key] = value
    _reg = st.session_state.setdefault("_tuple_cache_registry", [])
    if key not in _reg:
        _reg.append(key)



def _hpc_store_tuple_cache(_res, _ctx):
    """Store a result fetched from the cluster where the local 1D / 2D
    profile would have cached it."""
    if isinstance(_res, dict):
        for _k in ("p1_fixed", "p2_fixed", "x_fixed"):
            if _k in _ctx:
                _res[_k] = bool(_ctx[_k])
    _write_tuple_cache(tuple(_ctx.get("ck", [])), _res)


def _hpc_store_state(_key):
    """Handler that stores a fetched cluster result under a session key."""
    def _h(_res, _ctx):
        st.session_state[_key] = _res
    return _h


def _profile_param_picker(site: str, fit_stats: dict, fa: dict):
    """Multi-select of the parameters to scan in the 1D RMSE profile.

    Offers every parameter fitted in the last fit (constants, concentrations,
    titrants) plus the constants that were held FIXED, labelled "(fixed)".
    Fixed constants are scanned around their fixed value while every fitted
    parameter is re-optimised at each grid point -- the same engine the 2D
    map uses.  Returns ``(logK_values, concs_mM, titrants_mM, fixed_selected,
    tag, is_default)``; ``is_default`` is True when the selection equals
    "all fitted parameters" (the historical behaviour -- the cache key then
    stays the historical 3-tuple).
    """
    def _f(_v):
        try:
            return float(_v)
        except Exception:
            return None
    _K = {n: _f(v) for n, v in (fit_stats.get("param_values", {}) or {}).items()}
    _K = {n: v for n, v in _K.items() if v is not None}
    _c = {n: _f(v) for n, v in (fit_stats.get("fitted_concs", {}) or {}).items()}
    _c = {n: v for n, v in _c.items() if v is not None and v > 0}
    _t = {n: _f(v) for n, v in (fit_stats.get("fitted_titrants", {}) or {}).items()}
    _t = {n: v for n, v in _t.items() if v is not None and v > 0}
    _fixed = {}
    for n, v in ((fa or {}).get("start_logK") or (fa or {}).get("start_logk") or {}).items():
        if n in _K or n == "Kw":
            continue
        v = _f(v)
        if v is not None:
            _fixed[n] = v
    _opts = list(_K) + list(_c) + list(_t) + list(_fixed)
    _default = list(_K) + list(_c) + list(_t)
    _key, _okey = f"_prof_sel_{site}", f"_prof_sel_opts_{site}"
    if st.session_state.get(_okey) != _opts or _key not in st.session_state:
        # new fit / new model: back to the default -- also when the widget was
        # off-screen for a run (Streamlit then drops its value and the picker
        # came back empty, e.g. after a change of the diagnostics scope)
        st.session_state[_okey] = _opts
        st.session_state[_key] = list(_default)
    _sel = st.multiselect(
        "Parameters to profile", _opts, key=_key,
        format_func=lambda n: f"{n} (fixed)" if n in _fixed else n,
        help="Each selected parameter is pinned on a grid around its value "
             "while all fitted parameters are re-optimised at every point.  "
             "Constants held fixed in the last fit can be added: they are "
             "scanned around their fixed value.  Cost grows with the number "
             "of parameters selected.")
    _selK = {n: (_K[n] if n in _K else _fixed[n]) for n in _sel if n in _K or n in _fixed}
    _selc = {n: _c[n] for n in _sel if n in _c}
    _selt = {n: _t[n] for n in _sel if n in _t}
    return (_selK, _selc, _selt, {n: _fixed[n] for n in _sel if n in _fixed},
            ",".join(sorted(_sel)), sorted(_sel) == sorted(_default))


def _profile_display(prof, fixed_values):
    """Label the profiles of constants that were held fixed in the last fit."""
    if not prof:
        return prof
    _fx = fixed_values or {}
    return {(f"{n} (fixed at {_fx[n]:.2f})" if n in _fx else n): v for n, v in prof.items()}


def _profile_trace_block(site: str, fit_stats: dict) -> None:
    """Profile trace: scan one parameter, refit a chosen subset, plot the
    fitted values (left axis) and RMSE/RMSE_min (right axis, log) against the
    scanned value.  ``site`` is "eq" or "kin".  Shares the cache, parallel and
    supercomputer machinery of the 1D / 2D profiles."""
    _show = _indented_checkbox(
        "📈 Show profile trace (fitted constants vs one scanned parameter)",
        value=False, key=f"_show_ptr_{site}",
        help="Walk one parameter across a range; at every step refit only the "
             "parameters you tick (the others stay at their fitted values) and "
             "plot their fitted values, with RMSE / RMSE_min on the right axis.  "
             "Shows directly how one constant is compensated by the others.")
    if not _show:
        return
    _fa = st.session_state.get(f"_fit_args_{site}")
    if not _fa:
        st.info("Click **Fit** first to enable the profile trace.")
        return
    _fitted = {}
    for _n, _v in (fit_stats.get("param_values", {}) or {}).items():
        try: _fitted[_n] = float(_v)
        except Exception: pass
    _conc_like = {}
    for _key in ("fitted_concs", "fitted_titrants"):
        for _n, _mM in (fit_stats.get(_key, {}) or {}).items():
            try:
                _v = float(_mM)
                if _v > 0: _conc_like[_n] = float(np.log10(_v))
            except Exception: pass
    _fixed = {}
    for _n, _v in ((_fa.get("start_logK") or _fa.get("start_logk") or {})).items():
        if _n in _fitted or _n == "Kw": continue
        try: _fixed[_n] = float(_v)
        except Exception: pass
    _x_cands = {**_fitted, **_conc_like, **_fixed}
    _y_cands = {**_fitted, **_conc_like}
    if not _x_cands or not _y_cands:
        st.info("Need at least one fitted parameter.")
        return
    _lbl = lambda _n: (f"{_n} (fixed)" if _n in _fixed else _n)
    c1, _ = st.columns(2)
    _x = c1.selectbox("X-axis parameter", list(_x_cands), key=f"_ptr_x_{site}",
                      format_func=_lbl)
    c_sp, c_np, _ = st.columns(3)
    _span = c_sp.number_input("Span (log ±)", min_value=0.1, max_value=4.0, value=1.0,
                              step=0.1, key=f"_ptr_span_{site}")
    _np = c_np.number_input("Grid (odd)", min_value=3, max_value=201, value=11, step=2,
                            key=f"_ptr_np_{site}")
    _y_opts = [n for n in _y_cands if n != _x]
    _okey, _skey = f"_ptr_sel_opts_{site}", f"_ptr_sel_{site}"
    if st.session_state.get(_okey) != _y_opts:
        st.session_state[_okey] = _y_opts
        st.session_state[_skey] = list(_y_opts)
    _sel = st.multiselect("Parameters refit at each step (and plotted)", _y_opts, key=_skey,
                          help="Ticked parameters are re-optimised at every grid point; the "
                               "other fitted parameters are held at their fitted values.")
    if _x in _fixed:
        st.caption(f"**{_x}** was held fixed at {_fixed[_x]:.2f} in the last fit; the scan "
                   "is centred on that value.")
    _par = st.checkbox("Parallel (use all cores)", value=False, key=f"_ptr_par_{site}")
    _hpc_go = hpc_ui.controls(f"ptr_{site}", label="the profile trace")
    _ck = (f"_ptr_cache_{site}", _x, float(_span), int(_np), ",".join(sorted(_sel)))
    _res = st.session_state.get(_ck)
    if _res is None:
        if st.button("Compute profile trace", key=f"_ptr_btn_{site}",
                     disabled=not _sel) or _hpc_go:
            _bar = st.progress(0.0, text="Computing profile trace…")
            def _cb(f, m):
                try: _bar.progress(min(1.0, f), text=m)
                except Exception: pass
            try:
                _fa2 = dict(_fa)
                if _fa2.get("kind") != "global":
                    _fa2["kind"] = site
                _seed = dict(_fa2.get("start_logK") or _fa2.get("start_logk") or {})
                _seed.update(_fitted)
                _fa2["start_logK"] = _seed
                _fa2["start_logk"] = dict(_seed)
                _fa2["simplex_step"] = 0.15
                _res = compute_profile_trace(
                    _fa2, _x, _x_cands[_x], span=float(_span), n_pts=int(_np),
                    fit_names=list(_sel), n_jobs=(-1 if _par else 1),
                    progress_callback=_cb)
                _res["x_fixed"] = bool(_x in _fixed)
                _write_tuple_cache(_ck, _res)
                _bar.empty()
            except hpc_ui.CapturedCall as _hpc_cc:
                _bar.empty()
                hpc_ui.handle_capture(_hpc_cc, f"ptr_{site}", "profile trace",
                                      context={"ck": list(_ck), "x_fixed": bool(_x in _fixed)})
            except Exception as _e:
                _bar.empty()
                st.error(f"Profile trace failed: {_e}")
    hpc_ui.jobs_panel(f"ptr_{site}", on_result=_hpc_store_tuple_cache)
    if _res is None:
        return
    _noleg = st.checkbox("No legend", value=False, key=f"_ptr_noleg_{site}")
    try:
        _fig = make_profile_trace_figure(_res, show_legend=not _noleg)
        st.pyplot(_fig, clear_figure=False)
        try:
            _rm = np.asarray(_res["rmse"], float)
            _i0 = int(np.nanargmin(_rm))
            _msg = (f"Lowest RMSE on the scan: **{_rm[_i0]:.5g}** at log {_x} = "
                    f"{float(_res['x_grid'][_i0]):.2f}")
            _fit_r = fit_stats.get("rmse")
            if _fit_r is not None and np.isfinite(float(_fit_r)):
                _ref = _x_cands.get(_x)
                _msg += (f" — last fit: **{float(_fit_r):.5g}** at "
                         f"{'' if _ref is None else f'{_ref:.2f}'}"
                         f" ({float(_fit_r) / _rm[_i0]:.3f} × the scan minimum)")
            st.caption(_msg)
        except Exception:
            pass
        import io as _io
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=200, bbox_inches="tight")
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download profile trace PNG", data=_buf.getvalue(),
                           file_name=f"Equilibrist_trace_{_x}_{_ts}.png", mime="image/png",
                           key=f"_ptr_dl_{site}")
        if _res.get("error_samples"):
            st.caption("Refit errors (first few): " + "; ".join(str(e) for e in _res["error_samples"]))
    except Exception as _re:
        st.error(f"Could not render the profile trace: {_re}")

def _diag_fa(fa):
    """The fit record the diagnostics work on.  After a global fit every tab holds
    the global record; the "Diagnostics on" switch (above the fit statistics)
    chooses between it ("all tabs") and this tab alone at the global solution
    ("this tab only", see equilibrist_global.tab_fit_args)."""
    if (isinstance(fa, dict) and fa.get("kind") == "global"
            and st.session_state.get("_diag_scope") == "this tab only"):
        return eqglobal.tab_fit_args(fa)
    return fa


def _diag_scope_switch(fa_key):
    """The 'Diagnostics on' radio, shown only while this tab holds a global fit."""
    _fa_scope = st.session_state.get(fa_key)
    if not (isinstance(_fa_scope, dict) and _fa_scope.get("kind") == "global"):
        return
    st.session_state.setdefault("_diag_scope", "all tabs (global)")
    def _scope_changed():
        st.session_state["_diag_scope_changed"] = True
    st.radio("Diagnostics on", ["all tabs (global)", "this tab only"], key="_diag_scope",
             horizontal=True, on_change=_scope_changed,
             help="After a global fit the profiles, trace, sensitivity, bootstrap, jackknife, "
                  "Monte Carlo and model comparison either refit all tabs together (the global "
                  "objective) or this tab alone, starting from the global constants.  Changing "
                  "it clears the diagnostics shown.")
    if st.session_state.pop("_diag_scope_changed", False):
        _reset_diagnostic_toggles()
        st.rerun(scope="app")


def _share_shifts_checkbox(_ready):
    """The 'Share pure-species shifts across tabs' box, below 'Weight tabs by
    their own fit'.  Ticked by default when the tabs' shift columns line up
    (same number of columns, same target variables; they are paired by
    position) and unticked otherwise; once the user touches it, their choice
    stays (both are PREFS, kept across tabs)."""
    _fas = eqtabs.fit_args_with_data(st.session_state)      # every tab with data, ticked or not
    _ok, _why = eqglobal.shift_columns_line_up(_fas) if len(_fas) >= 2 else (False, "")
    if not st.session_state.get("_global_share_shifts_touched"):
        st.session_state["_global_share_shifts"] = bool(_ok)
    st.checkbox("Share pure-species shifts across tabs", key="_global_share_shifts",
                on_change=lambda: st.session_state.__setitem__("_global_share_shifts_touched", True),
                help="The chemical shift of each species (GHM, GH2M, ...) is solved once for all tabs "
                     "instead of separately in each: right when the tabs follow the same protons of the "
                     "same compound under the same conditions.  Shift columns are paired by position.  "
                     "Ticked by default when the tabs' shift columns line up.")
    if st.session_state.get("_global_share_shifts") and _why:
        st.caption(f"⚠️ Shared shifts will be refused: {_why}.")


def _render_pure_shift_comparison(fit_stats):
    """Pure-species shifts of every fitted tab side by side, protons matched by
    their free shift (see equilibrist_global.match_shift_columns), with the
    spread across tabs: what 'Share pure-species shifts across tabs' has to
    reconcile.  Shown under the pure-species table when at least two tabs
    carry NMR shift results."""
    ss = st.session_state
    tabs = ss.get("_tabs") or []
    if len(tabs) < 2:
        return
    names = eqtabs.labels(ss)
    per_tab = []
    for i, tab in enumerate(tabs):
        stt = fit_stats if i == ss.get("_tab_idx") else ((tab.get("state") or {}).get("_fit_stats"))
        if isinstance(stt, dict) and stt.get("pure_shifts"):
            per_tab.append((names[i], stt))
    if len(per_tab) < 2:
        return
    cols_by_tab, fps = [], []
    for name, stt in per_tab:
        df = stt.get("delta_free") or {}
        cols = [(tgt, c) for tgt in stt["pure_shifts"] for c in stt["pure_shifts"][tgt]]
        cols_by_tab.append(cols)
        fps.append([float(df.get(c, 0.0)) for _, c in cols])
    try:
        groups, how = eqglobal.match_shift_columns(fps)
    except ValueError as ex:
        st.caption(f"Pure-species shifts across tabs: {ex}.")
        return
    rows = []
    for grp in groups:
        present = [(t, j) for t, j in enumerate(grp) if j is not None]
        if len(present) < 2:
            continue
        species = []
        for t, j in present:
            tgt, c = cols_by_tab[t][j]
            for sp in per_tab[t][1]["pure_shifts"][tgt][c]:
                if sp not in species:
                    species.append(sp)
        free0 = fps[present[0][0]][present[0][1]]
        for sp in species:
            row = {"Proton (free δ)": f"{free0:.3f}", "Species": sp}
            vals = []
            for t, j in present:
                name, stt = per_tab[t]
                tgt, c = cols_by_tab[t][j]
                dd = stt["pure_shifts"][tgt][c].get(sp)
                if dd is None:
                    row[name] = "—"
                    continue
                show_abs = (not stt.get("nmr_noref", False)) or stt.get("pure_shifts_anchored", False)
                v = (float((stt.get("delta_free") or {}).get(c, 0.0)) if show_abs else 0.0) + float(dd)
                vals.append(v)
                und = sp in ((((stt.get("pure_shift_undetermined") or {}).get(tgt) or {}).get(c)) or [])
                row[name] = f"({v:.3f})" if und else f"{v:.3f}"
            row["Δ (max − min)"] = f"{(max(vals) - min(vals)):.3f}" if len(vals) >= 2 else ""
            rows.append(row)
    if not rows:
        return
    st.write("**Pure-species shifts across tabs (ppm):**")
    st.dataframe(pd.DataFrame(rows).set_index(["Proton (free δ)", "Species"]), width="stretch")
    st.caption(("Protons matched by their free shift" if how == "fingerprint" else "Columns paired by position")
               + "; ( ) = not determined in that tab; Δ = spread across tabs, i.e. what "
                 "'Share pure-species shifts across tabs' has to reconcile.")


def _global_fit_dispatch(_fit_slot):
    """Run one fit across every registered tab (any mode), hand each tab its
    results and the GLOBAL fit args, then rerun.  Called from the equilibrium
    and the kinetics branches when the Global fit button was pressed."""
    _gexps = eqtabs.collect_experiments(st.session_state)
    if len(_gexps) < 2:
        st.session_state["_fit_message"] = ("warning", "A Global fit needs at least two tabs with data "
                                            "loaded and a 'fit' box ticked.")
    else:
        try:
            _g_tol_log = float(st.session_state.get("fit_tolerance_log", 3.0))
            _g_timeout = float(st.session_state.get("fit_timeout", 30))
            _g_weighted = bool(st.session_state.get("_global_weighted", True))
            _g_share = bool(st.session_state.get("_global_share_shifts", False))
            _g_in = [{"label": _lab, "fa": _fa, "start_logK": dict(_fa.get("start_logK") or {}),
                      "ticked": list(_fa.get("fit_keys") or [])} for _i, _lab, _fa in _gexps]
            # the ACTIVE tab's sidebar values win as starting point for shared constants
            _g_active = [x for x in _g_in if x["label"] == eqtabs.labels(st.session_state)[st.session_state["_tab_idx"]]]
            if _g_active:
                _g_in.remove(_g_active[0]); _g_in.insert(0, _g_active[0])
            _hpc_site = st.session_state.pop("_global_fit_hpc", None)
            if _hpc_site:
                # the call is captured and shipped to the cluster; its result comes
                # back through _hpc_store_fit -> _apply_global_pending.  On the
                # cluster the fit runs for the job's wall time, not the sidebar Timeout.
                _g_timeout = hpc_ui.fit_timeout_seconds(_g_timeout)
                hpc_ui.hpc.arm_capture(_hpc_site)
                try:
                    eqglobal.global_fit(_g_in, tolerance=10.0 ** (-_g_tol_log), maxiter=100_000,
                                        timeout_s=_g_timeout, weighted=_g_weighted, share_shifts=_g_share)
                except hpc_ui.CapturedCall as _cc:
                    hpc_ui.handle_capture(_cc, _hpc_site, "Global fit",
                                          context={"tool": "global", "labels": [x["label"] for x in _g_in],
                                                   "tol_log": _g_tol_log, "timeout": _g_timeout},
                                          settings_override=hpc_ui.fit_settings())
                    return
                hpc_ui.hpc.disarm_capture()
                st.error("The global fit could not be captured for the cluster.")
                return
            _gres = _fit_async(_fit_slot, _g_timeout, lambda: eqglobal.global_fit(_g_in, tolerance=10.0 ** (-_g_tol_log), maxiter=100_000,
                                            timeout_s=_g_timeout, weighted=_g_weighted, share_shifts=_g_share),
                _g_in[0]["start_logK"], "Global fit across tabs…", "_global_fit_requested")
            _apply_global_result(_gres, _g_in, _gexps, _g_tol_log, _g_timeout)
            st.rerun()
        except Exception as _ge:
            import traceback as _gtb
            st.error(f"Global fit error: {_ge}")
            st.caption(_gtb.format_exc())


def _apply_global_result(_gres, _g_in, _gexps, _g_tol_log, _g_timeout):
    """Hand every tab its share of a global fit result (computed here or fetched
    from the cluster): fitted values, stats, the global fit-args for the diagnostics."""
    _g_weighted = bool(_gres.get("weighted", True))
    _g_parts = ", ".join(f"{_lab}: {_r:.3g}" if _r is not None else f"{_lab}: n/a"
                         for _lab, _r in _gres["rmse_by_label"].items())
    if _g_weighted:
        _g_parts += ";  tabs rescaled to their own fit, relative weights " + ", ".join(
            f"{_lab}: {_w:.2g}" for _lab, _w in (_gres.get("weights") or {}).items())
    else:
        _g_parts += ";  plain sum"
    _g_share = _gres.get("info_share") or {}
    if _g_share:
        def _share_txt(_k):
            return " / ".join(f"{_lab} {100 * _d[_k]:.0f} %" for _lab, _d in _g_share.items()
                              if _d.get(_k) is not None)
        _g_parts += ";  share of the information on " + ", ".join(
            f"{_k}: {_share_txt(_k)}" for _k in (_gres.get("shared") or []) if _share_txt(_k))
    if _gres.get("share_shifts"):
        _sp = _gres.get("shared_pure_shifts") or []
        _common = [s for s in _sp if sum(c is not None for c in s.get("cols") or []) >= 2]
        _how = (_common[0].get("how") if _common else None)
        _g_parts += (";  pure-species shifts shared across tabs ("
                     + f"{len(_common)} proton{'s' if len(_common) != 1 else ''} "
                     + ("matched by their V = 0 shifts: " + ", ".join(f"{s['free_shift']:.3f}" for s in _common) + " ppm"
                        if _how == "fingerprint" else "paired by column position")
                     + (f"; {len(_sp) - len(_common)} tab-specific" if len(_sp) > len(_common) else "") + ")")
    if _gres["summary"].get("n_restarts"):
        _g_parts += f";  {_gres['summary']['n_restarts']} restart(s)"
    _g_lvl = "success" if _gres["success"] else "warning"
    _parsed_list = [_fa.get("parsed_fit") for _i, _lab, _fa in _gexps]
    _g_txt = (f"{'✅' if _gres['success'] else '⏱️'} Global fit over {len(_g_in)} tabs "
              f"{_gres['message']} after {_gres['summary'].get('n_evals')} evaluations "
              f"in {_gres['elapsed_s']:.0f} s.  Shared constants: "
              + ", ".join(eqglobal.format_shared_constant(k, _gres['values'][k], _parsed_list) for k in _gres["shared"])
              + f".  Global RMSE {(_gres['rmse'] or float('nan')):.3g}  (per tab — {_g_parts}).")
    _names_now = eqtabs.labels(st.session_state)
    for _i, _lab, _fa in _gexps:
        _st_i = _gres["stats"].get(_lab) or {}
        # this tab's residuals at the global solution: without them the residual
        # diagnostics (and their heatmap / sloppy spectrum / t-test / profile
        # children) stay hidden, or show the residuals of an older single fit
        try:
            _x_expr_i = (_fa.get("x_expr") or (_fa.get("parsed_fit") or {}).get("plot_x_expr")
                         or ("t" if _fa.get("kind") == "kin" else ""))
            _res_i = collect_residuals_from_stats(_st_i, x_expr=_x_expr_i)
            augment_stats(_st_i, _res_i)
        except Exception:
            _res_i = {}
        # the diagnostics of every tab now run on the GLOBAL objective
        _fa_i = eqglobal.make_global_fit_args(
            _fa, _g_in, _gres, tolerance=10.0 ** (-_g_tol_log), maxiter=100_000,
            timeout_s=_g_timeout, label=_lab)
        _fa_key = "_fit_args_kin" if (_fa.get("kind") == "kin") else "_fit_args_eq"
        if _i == st.session_state["_tab_idx"]:
            for _k, _v in (_st_i.get("param_values") or {}).items():
                st.session_state[f"_pending_logK_{_k}"] = float(_v)
            for _root, _mM in (_st_i.get("fitted_concs") or {}).items():
                st.session_state[f"_pending_conc_{_root}"] = float(_mM)
            for _tk, _mM in (_st_i.get("fitted_titrants") or {}).items():
                st.session_state[f"_pending_titrant_mM_{_tk}"] = float(_mM)
            st.session_state["_fit_counter"] = int(st.session_state.get("_fit_counter", 0)) + 1
            _reset_diagnostic_toggles()
            st.session_state["_fit_stats"] = _st_i
            st.session_state["_fit_residuals"] = _res_i
            st.session_state[_fa_key] = _fa_i
            st.session_state["_fit_message"] = (_g_lvl, _g_txt)
        else:
            eqtabs.write_fit_into(st.session_state["_tabs"][_i]["state"], _st_i, _fa_i, (_g_lvl, _g_txt), key=_fa_key)
            st.session_state["_tabs"][_i]["state"]["_fit_residuals"] = _res_i


def _hpc_store_fit(_res, _ctx):
    """A fit fetched from the cluster (this tab's or the global one) is applied at
    the top of the next run, before any widget exists (fitted values go into
    widget keys)."""
    if (_ctx or {}).get("tool") == "global":
        st.session_state["_global_fit_pending"] = (_res, _ctx)
    else:
        st.session_state["_single_fit_pending"] = (_res, _ctx)


def _submit_single_fit_to_cluster(_site):
    """Capture this tab's fit (from its live fit record) and ship it to the cluster."""
    _fa = st.session_state.get("_fit_args_live")
    if not isinstance(_fa, dict) or not _fa.get("has_data") or not (
            _fa.get("fit_keys") or _fa.get("fit_conc_keys") or _fa.get("fit_titrant_keys")):
        st.session_state["_fit_message"] = ("warning", "Load experimental data and tick at least "
                                            "one 'fit' box before submitting the fit.")
        return
    _tol_log = float(st.session_state.get("fit_tolerance_log", 3.0))
    # on the cluster the fit runs for the job's wall time, not the sidebar Timeout
    _timeout = hpc_ui.fit_timeout_seconds(float(st.session_state.get("fit_timeout", 30)))
    _label = eqtabs.labels(st.session_state)[st.session_state["_tab_idx"]]
    hpc_ui.hpc.arm_capture(_site)
    try:
        eqglobal.single_fit(_fa, tolerance=10.0 ** (-_tol_log), maxiter=100_000, timeout_s=_timeout)
    except hpc_ui.CapturedCall as _cc:
        hpc_ui.handle_capture(_cc, _site, "Fit", context={"tool": "single", "label": _label,
                                                        "tol_log": _tol_log, "timeout": _timeout},
                              settings_override=hpc_ui.fit_settings())
        return
    hpc_ui.hpc.disarm_capture()
    st.error("The fit could not be captured for the cluster.")


def _apply_single_pending(_sp):
    """Apply a single-tab fit fetched from the cluster to the tab it was submitted from."""
    _res, _ctx = _sp
    _label = _ctx.get("label")
    _names = eqtabs.labels(st.session_state)
    if _label not in _names:
        st.session_state["_fit_message"] = ("warning", f"The cluster's fit was submitted from tab "
                                            f"{_label}, which no longer exists; its result was not applied.")
        return
    _i = _names.index(_label)
    _active = (_i == st.session_state["_tab_idx"])
    _state = st.session_state if _active else (st.session_state["_tabs"][_i].get("state") or {})
    _fa = _state.get("_fit_args_live")
    if not isinstance(_fa, dict):
        st.session_state["_fit_message"] = ("warning", f"Tab {_label} has no fit record any more; "
                                            "the cluster's fit was not applied.")
        return
    _stats = dict(_res.get("stats") or {})
    _fitted = {k: float(v) for k, v in (_res.get("fitted") or {}).items()}
    _stats.setdefault("param_values", dict(_fitted))
    _fa = dict(_fa); _fa.pop("has_data", None)
    _fa["n_data"] = int(_stats.get("n_points", 0) or 0)
    _fa["tolerance"] = 10.0 ** (-float(_ctx.get("tol_log", 3.0)))
    _fa["timeout_s"] = float(_ctx.get("timeout", 30))
    _key = "_fit_args_kin" if _fa.get("kind") == "kin" else "_fit_args_eq"
    _lvl = "success" if _res.get("ok") else "warning"
    _txt = f"{'✅' if _res.get('ok') else '⏱️'} {_res.get('message', 'Fit')} — computed on the supercomputer."
    if _active:
        _acid = bool((_fa.get("parsed_fit") or {}).get("is_acid_base"))
        for _k, _v in _fitted.items():
            if _acid and _k != "Kw":
                st.session_state[f"_pending_pKa_{_k}"] = -float(_v)
            else:
                st.session_state[f"_pending_logK_{_k}"] = float(_v)
        for _root, _mM in (_stats.get("fitted_concs") or {}).items():
            st.session_state[f"_pending_conc_{_root}"] = float(_mM)
        for _tk, _mM in (_stats.get("fitted_titrants") or {}).items():
            st.session_state[f"_pending_titrant_mM_{_tk}"] = float(_mM)
        st.session_state["_fit_counter"] = int(st.session_state.get("_fit_counter", 0)) + 1
        _reset_diagnostic_toggles()
        st.session_state["_fit_stats"] = _stats
        st.session_state[_key] = _fa
        st.session_state["_fit_message"] = (_lvl, _txt)
    else:
        eqtabs.write_fit_into(st.session_state["_tabs"][_i]["state"], _stats, _fa, (_lvl, _txt), key=_key)
    st.rerun()


def _apply_global_pending(_gp):
    _res, _ctx = _gp
    _gexps = eqtabs.collect_experiments(st.session_state)
    _labels = [_lab for _i, _lab, _fa in _gexps]
    _want = list(_ctx.get("labels") or [])
    if sorted(_want) != sorted(_labels):
        st.session_state["_fit_message"] = ("warning", "The cluster's global fit was over the tabs "
                                            + ", ".join(_want) + " but the tabs are now " + ", ".join(_labels)
                                            + "; its result was not applied.")
        return
    try:
        _g_in = [{"label": _lab, "fa": _fa, "start_logK": dict(_fa.get("start_logK") or {}),
                  "ticked": list(_fa.get("fit_keys") or [])} for _i, _lab, _fa in _gexps]
        _apply_global_result(_res, _g_in, _gexps, float(_ctx.get("tol_log", 3.0)), float(_ctx.get("timeout", 30)))
        st.rerun()
    except Exception as _ge:
        import traceback as _gtb
        st.error(f"Could not apply the cluster's global fit: {_ge}")
        st.caption(_gtb.format_exc())


def _reset_diagnostic_toggles():
    """Clear all "Show X" toggle states and cached diagnostic results.

    Called immediately before a new fit's ``_fit_stats`` is written into
    session_state.  Without this, every diagnostic checkbox the user had
    enabled for fit #N stays enabled for fit #N+1 — so the user sees a
    stale parameter-profile, 2D-RMSE colormap, jackknife, bootstrap,
    Monte-Carlo, or LST result generated from the *previous* fit's
    optimum.  Resetting all toggles to False makes the user opt back in
    to each diagnostic for the new fit, which is the correct semantics:
    every diagnostic must reflect the *current* fit.

    For the toggle (``_show_*``) keys we SET to ``False`` rather than
    delete.  Streamlit's ``@st.fragment`` decorator caches widget state
    independently of the public ``st.session_state`` dict; deleting a
    fragment-hosted widget's key on its own doesn't always reach back
    into the fragment's internal cache, so the next fragment render
    can show the stale ``True``.  Explicitly writing ``False`` to the
    session-state key tells the widget to render unchecked on its next
    pass, which is what we want.

    For the cache (``_bs_result_*``, ``_jk_result_*`` etc.) keys we
    delete — those are data caches, not widget states, so removal is
    correct.

    Special case: JSON-load auto-refit.  When the user loads a session
    JSON that contains UQ data, the load handler restores those UQ
    dicts into the canonical ``_bs_result_*``/``_jk_result_*``/
    ``_mc_result_*`` session_state keys, then sets ``_fit_requested``
    so the fit re-runs on the next rerun (to reconstruct the per-fit
    plots and diagnostic figures).  That auto-refit calls this
    function — which would otherwise blindly wipe the just-restored
    UQ data as "stale".  The load handler now sets a one-shot
    ``_preserve_uq_on_next_reset`` flag in that case; when present,
    this function skips the UQ-cache deletion (toggles are still
    reset; only the UQ data dicts are preserved) and clears the flag.
    """
    _preserve_uq = bool(st.session_state.pop(
        "_preserve_uq_on_next_reset", False))
    _str_toggle_prefixes = (
        "_show_",
        "_lst_eq_chk_", "_lst_kin_chk_",
        "_var_pick_",
    )
    _str_cache_prefixes = (
        "_bs_result_", "_jk_result_", "_mc_result_", "_lst_result_",
        "_ttest_names_", "_ttest_nulls_",
    )
    # Tuple cache keys: first element string starts with one of these
    _tuple_prefixes = (
        "_p2d_cache_", "_lst_cache_", "_jk_cache_",
        "_mc_cache_", "_bs_cache_", "_prof_cache_",
    )
    # When preserving diagnostic state on a JSON-load auto-refit,
    # exclude:
    #   • the UQ + LST data-cache string prefixes (_bs_result_,
    #     _jk_result_, _mc_result_, _lst_result_) so the restored
    #     bootstrap / jackknife / Monte-Carlo / Masson-ξ panels
    #     survive the reset;
    #   • the 1-D / 2-D / LST profile-cache tuple prefixes
    #     (_prof_cache_, _p2d_cache_, _lst_cache_) so the restored
    #     RMSE-profile and colormap panels also survive; the
    #     bootstrap / jackknife / MC tuple caches are preserved too
    #     for symmetry;
    #   • the LST variable-selection toggle prefixes
    #     (_lst_eq_chk_, _lst_kin_chk_, _var_pick_) so the
    #     variable-selection boxes redisplay the variables that were
    #     selected when the saved Masson-ξ test was run.  ``_show_*``
    #     panel-expand toggles still reset (so panels render
    #     collapsed and the user clicks to reveal them — but the
    #     data is there when they do).
    if _preserve_uq:
        _uq_str_prefixes  = ("_bs_result_", "_jk_result_",
                             "_mc_result_", "_lst_result_")
        _uq_tuple_prefixes = ("_bs_cache_", "_jk_cache_", "_mc_cache_",
                              "_prof_cache_", "_p2d_cache_",
                              "_lst_cache_")
        # Toggle prefixes that gate figure rendering — the user wants
        # the captured state to win (figures the user had open at
        # save time should re-open after JSON load, ditto for the
        # LST variable-selection checkboxes).  ``_show_*`` covers the
        # outer panel-expand boxes AND the inner figure-gating boxes
        # (``_show_jk_influence_*``, ``_show_prof_*``, ``_show_p2d_*``,
        # ``_show_corr_*``); preserving them means after the JSON
        # load + auto-refit, the user lands exactly where they were
        # at save time — tables AND figures both rendered.
        _uq_toggle_prefixes = ("_lst_eq_chk_", "_lst_kin_chk_",
                               "_var_pick_", "_show_")
        _str_cache_prefixes = tuple(p for p in _str_cache_prefixes
                                     if p not in _uq_str_prefixes)
        _tuple_prefixes = tuple(p for p in _tuple_prefixes
                                 if p not in _uq_tuple_prefixes)
        _str_toggle_prefixes = tuple(
            p for p in _str_toggle_prefixes
            if p not in _uq_toggle_prefixes)
    for _k in list(st.session_state.keys()):
        try:
            if isinstance(_k, str):
                if any(_k.startswith(_p) for _p in _str_toggle_prefixes):
                    st.session_state[_k] = False
                elif any(_k.startswith(_p) for _p in _str_cache_prefixes):
                    del st.session_state[_k]
            elif isinstance(_k, tuple) and _k and isinstance(_k[0], str) \
                    and any(_k[0].startswith(_p) for _p in _tuple_prefixes):
                del st.session_state[_k]
        except Exception:
            pass
    # ``st.session_state.keys()`` does NOT yield tuple keys (Streamlit 1.54; see
    # _write_tuple_cache), so the loop above never reached the profile / 2D-map
    # caches and a stale profile survived every new fit and every change of the
    # diagnostics scope (caught 2026-11-07).  Walk the registry as well.
    _reg = st.session_state.get("_tuple_cache_registry")
    if isinstance(_reg, list):
        _keep = []
        for _k in list(_reg):
            try:
                if isinstance(_k, tuple) and _k and isinstance(_k[0], str) \
                        and any(_k[0].startswith(_p) for _p in _tuple_prefixes):
                    if _k in st.session_state:
                        del st.session_state[_k]
                else:
                    _keep.append(_k)
            except Exception:
                _keep.append(_k)
        st.session_state["_tuple_cache_registry"] = _keep


st.set_page_config(page_title="Equilibrist", layout="wide")

# ── Tabs: a pending new / close / switch request is honoured here, before any
# widget exists (the only moment widget-owned keys may be rewritten freely).
_tab_notice = eqtabs.swap_if_requested(st.session_state)
if _tab_notice:
    st.session_state["_tab_notice"] = _tab_notice

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





# ── Outlier helpers ──────────────────────────────────────────────────────────
# Outlier state lives in three session-state keys:
#   _outliers_main    : dict  col_name → set of int indices  (exp_data / kinetics)
#   _outliers_nmr     : dict  col_name → set of int indices  (nmr_data)
#   _outliers_spectra : set   of int row indices              (spectra_data)

def _bump_outlier_ver(state_key: str) -> None:
    k = f"_outlier_ver_{state_key}"
    st.session_state[k] = st.session_state.get(k, 0) + 1

def _toggle_outlier(state_key: str, col: str, idx: int) -> None:
    """Toggle a single point in session_state[state_key][col]."""
    d = st.session_state.setdefault(state_key, {})
    s = d.setdefault(col, set())
    s.discard(idx) if idx in s else s.add(idx)
    _bump_outlier_ver(state_key)

def _toggle_spectra_outlier(state_key: str, idx: int) -> None:
    """Toggle a single row in session_state[state_key] (plain set)."""
    s = st.session_state.setdefault(state_key, set())
    s.discard(idx) if idx in s else s.add(idx)
    _bump_outlier_ver(state_key)

def _filter_exp_outliers(data: dict, state_key: str) -> dict:
    """Return a copy of an exp/nmr data dict with excluded rows removed."""
    outliers = st.session_state.get(state_key, {})
    if not outliers:
        return data
    out = {}
    for k, v in data.items():
        if k.startswith("_"):
            out[k] = v; continue
        excl = outliers.get(k, set())
        if not excl:
            out[k] = v; continue
        mask = np.array([i not in excl for i in range(len(v["y"]))])
        out[k] = {"v_add_mL": v["v_add_mL"][mask], "y": v["y"][mask]}
    return out

def _filter_spectra_outliers(data: dict, state_key: str) -> dict:
    """Return a copy of spectra_data with excluded rows removed."""
    excl = st.session_state.get(state_key, set())
    if not excl or not data:
        return data
    mask = np.array([i not in excl for i in range(len(data["x_vals"]))])
    return {**data, "x_vals": data["x_vals"][mask], "A": data["A"][mask]}

def _n_outliers(state_key: str) -> int:
    v = st.session_state.get(state_key)
    if isinstance(v, dict):
        return sum(len(s) for s in v.values())
    if isinstance(v, set):
        return len(v)
    return 0

def _nmr_excl_union(nmr_data: dict) -> set:
    """Union of excluded row indices across all NMR columns — used to hollow
    back-calc concentration dots that share the same row indices as NMR data."""
    outliers = st.session_state.get("_outliers_nmr", {})
    if not outliers:
        return set()
    all_cols = [c for c in nmr_data if not c.startswith("_")]
    merged = set()
    for col in all_cols:
        merged |= outliers.get(col, set())
    return merged

def _nmr_excl_intersection(nmr_data: dict) -> set:
    """Intersection of excluded row indices across all NMR columns.

    A back-calc concentration dot at row i should only be hollowed when ALL
    NMR columns have row i excluded — meaning the whole row was flagged via
    a main-plot click.  A single-column exclusion (from a shift-plot click)
    leaves the concentration dot solid because the concentration is still
    derivable from the remaining signals.
    """
    outliers = st.session_state.get("_outliers_nmr", {})
    if not outliers:
        return set()
    all_cols = [c for c in nmr_data if not c.startswith("_")]
    if not all_cols:
        return set()
    result = outliers.get(all_cols[0], set()).copy()
    for col in all_cols[1:]:
        result &= outliers.get(col, set())
    return result

def _outlier_bar(chart_id: str, *state_keys: str) -> None:
    """Always render an outlier hint below a chart; add count + clear when points are excluded."""
    total = sum(_n_outliers(k) for k in state_keys)
    if total == 0:
        st.caption("🖱️ Click a data point to flag it as an outlier and exclude it from fitting")
        return
    _ob1, _ob2 = st.columns([5, 1])
    with _ob1:
        st.caption(
            f"⚠️ {total} point{'s' if total != 1 else ''} excluded from fitting — "
            "click a hollow marker to restore it"
        )
    with _ob2:
        if st.button("✕ Clear all", key=f"_clr_{chart_id}"):
            for k in state_keys:
                st.session_state.pop(k, None)
                _bump_outlier_ver(k)
            st.rerun()

def _process_outlier_event(event, state_key: str, is_spectra: bool = False,
                             nmr_bc_cols: list = None) -> bool:
    """
    Process a plotly_chart on_select event.

    Handles three customdata shapes:
      ["__nmr_bc__", idx]   — back-calc NMR dot on main plot; toggles idx in
                              ALL nmr_bc_cols inside _outliers_nmr so the
                              secondary NMR plot hollows the same row.
      [col_name, idx]       — regular exp/NMR dot; toggles in state_key.
      [[row_idx]]           — spectra line (is_spectra=True); toggles in state_key.

    Always bumps the version key on any click (even on non-customdata traces)
    so the chart gets a fresh widget key, clearing Plotly selection highlighting
    and breaking the on_select -> rerun -> on_select loop.
    """
    if not event or not event.selection or not event.selection.points:
        return False
    for _pt in event.selection.points:
        _cd = _pt.get("customdata")
        if not _cd:
            continue
        if is_spectra:
            _toggle_spectra_outlier(state_key, int(_cd[0]))
        elif str(_cd[0]) == "__nmr_bc__" and nmr_bc_cols and len(_cd) >= 2:
            # Back-calc NMR dot: mirror exclusion across all NMR columns so
            # the secondary NMR shift plot hollows the matching row too.
            for _col in nmr_bc_cols:
                _toggle_outlier("_outliers_nmr", _col, int(_cd[1]))
            _bump_outlier_ver("_outliers_nmr")
        elif str(_cd[0]) == "__uvvis_bc__" and len(_cd) >= 2:
            # Back-calc UV-Vis dot: toggle the row index in _outliers_spectra
            # so the spectra plot dashes the matching spectrum.
            _toggle_spectra_outlier("_outliers_spectra", int(_cd[1]))
        else:
            if len(_cd) < 2:
                continue
            _toggle_outlier(state_key, str(_cd[0]), int(_cd[1]))
    # MUST always bump here: _toggle_* only bumps when customdata is present.
    # Without this, clicking a non-customdata trace returns True but leaves the
    # version unchanged -> same chart key -> Plotly restores selection -> fires
    # on_select again -> infinite rerun loop.
    _bump_outlier_ver(state_key)
    return True

# ─────────────────────────────────────────────────────────────────────────────

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
                    bool(st.session_state.get("_spectra_data", {})) or
                    bool(st.session_state.get("_itc_raw")))

    c1, c2 = st.columns([0.86, 0.14])
    with c1:
        val = st.number_input(label, key=key, step=0.1, format="%.2f")
    with c2:
        _fit_toggle(fit_key, disabled=not has_exp_data)
    return float(val)


def _pka_input_with_fit(label, pka_key, logk_key, default_pka):
    """
    pKa widget (positive number) with fit checkbox — acid-base mode only.
    Stores pKa in session_state[pka_key]; logK = -pKa is returned for the solver.
    Handles _pending_logK_ writeback from the fitting engine by converting to pKa.
    """
    # Convert any pending logK (from fitter) → pKa.
    # The top-of-file loop may have already applied _pending_logK_ → logK_{key},
    # so check both the pending key and the logK key.
    kname_suffix    = logk_key[len('logK_'):]
    pending_pka_key = f"_pending_pKa_{kname_suffix}"
    pending_logk_key = f"_pending_logK_{kname_suffix}"
    if pending_pka_key in st.session_state:
        # Soft-apply or direct pKa pending — takes priority
        st.session_state[pka_key] = float(st.session_state.pop(pending_pka_key))
    elif pending_logk_key in st.session_state:
        # Fitter wrote _pending_logK_ before the top-of-file loop consumed it
        st.session_state[pka_key] = -float(st.session_state.pop(pending_logk_key))
    elif logk_key in st.session_state and pka_key not in st.session_state:
        # top-of-file loop already moved pending → logK_{kname}; convert to pKa
        st.session_state[pka_key] = -float(st.session_state[logk_key])
    elif pka_key not in st.session_state:
        st.session_state[pka_key] = float(default_pka)

    fit_key = f"fit_{logk_key}"   # keep same fit key as normal mode so fitter finds it
    if fit_key not in st.session_state:
        st.session_state[fit_key] = False

    has_exp_data = (bool(st.session_state.get("_exp_data", {})) or
                    bool(st.session_state.get("_nmr_data", {})) or
                    bool(st.session_state.get("_spectra_data", {})) or
                    bool(st.session_state.get("_itc_raw")))

    c1, c2 = st.columns([0.86, 0.14])
    with c1:
        pka_val = st.number_input(label, key=pka_key, step=0.1, format="%.2f")
    with c2:
        _fit_toggle(fit_key, disabled=not has_exp_data)
    return -float(pka_val)   # return logK for the solver


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
                    bool(st.session_state.get("_spectra_data", {})) or
                    bool(st.session_state.get("_itc_raw")))

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
        _fit_toggle(fit_key, disabled=(not has_exp_data) or spectra_only)
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
                    bool(st.session_state.get("_spectra_data", {})) or
                    bool(st.session_state.get("_itc_raw")))

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
        _fit_toggle(fit_key, disabled=not has_exp_data)

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

# ── Restore what a background fit's early reruns drop ────────────────────
# In kinetics mode the sidebar renders AFTER the fit code, so the rerun that
# starts a background fit, and the one that collects it, drop the widget state
# of the constants and concentrations: they came back at the script values
# (pre-equilibria set to 2.0 by hand showed 3.0 again after Global fit).
# _fit_async keeps them (bg["ticks"] while the fit runs, _bg_prefs_restore for
# the final rerun); put them back here, before _fit_args_live is built and
# before any widget renders.  setdefault: a value the fit wrote back (above)
# or that the user changed meanwhile is not overridden.
_bg_now = st.session_state.get("_bg_fit")
if _bg_now is not None:
    for _bk, _bv in (_bg_now.get("ticks") or {}).items():
        st.session_state.setdefault(_bk, _bv)
for _bk, _bv in (st.session_state.pop("_bg_prefs_restore", None) or {}).items():
    st.session_state.setdefault(_bk, _bv)

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
# Restore xmax across the kinetics fit's rerun.  Set by the persist
# block at line ~1346 just before its st.rerun(); must be restored here,
# before any widget renders, so the sidebar save block (which reads
# ``session_state["xmax"]``) sees the correct value on Run B.
if "_persist_xmax" in st.session_state:
    st.session_state["xmax"] = st.session_state.pop("_persist_xmax")

# ── Script upload ─────────────────────────────
with st.sidebar:
    st.subheader("Equilibrist Script")
    st.caption("Equilibrist v3.0 beta")

    # A "Clear & reload" button that wipes everything and resets the uploader
    if st.button("↺ Reset / load new script"):
        for k in list(st.session_state.keys()):
            if k not in eqtabs.META:
                del st.session_state[k]
        st.rerun()
    if st.button("➕ New tab", help="Open another workspace with its own script and data.  "
                                    "Constants with the same name are shared by the Global fit."):
        # snapshot now: after st.rerun() the widgets not rendered in this run lose their state
        st.session_state["_tab_new_snapshot"] = eqtabs.snapshot(st.session_state)
        st.session_state["_tab_new"] = True
        st.rerun()

    # Nonce makes the uploader widget unique each session so re-uploading
    # the same filename is always treated as a new upload by the browser.
    if "_uploader_nonce" not in st.session_state:
        st.session_state["_uploader_nonce"] = 0

    uploaded = st.file_uploader(
        "Upload .txt script or .json session bundle",
        type=["txt", "json"],
        key=f"_uploader_{st.session_state['_uploader_nonce']}",
        help="Drop a .txt Equilibrist script to start a new analysis, "
             "or a .json session bundle (from the Save button below) "
             "to restore a previous one — including the data file and "
             "last fit result."
    )
    if uploaded is not None:
        _is_session = uploaded.name.lower().endswith(".json")
        if _is_session:
            # ── Session restore from JSON ─────────────────────────────
            # Parse the bundle (SHA-256 verified inside) before
            # touching anything; a bad file is surfaced as an error
            # without clobbering current state.
            try:
                from equilibrist_session import parse_session_json
                _raw_json = uploaded.read().decode("utf-8", errors="replace")
                _obj = parse_session_json(_raw_json)
            except Exception as _le:
                st.error(
                    f"This .json doesn't look like an Equilibrist "
                    f"session bundle: {_le}"
                )
                _obj = None

            if _obj is not None:
                # Mirror the .txt branch's reset pattern: preserve fit
                # preferences, bump BOTH uploader nonces (so the file
                # uploader widgets get fresh keys and don't replay the
                # JSON on the next rerun — without this, the JSON would
                # re-trigger the restore on every rerun, an infinite
                # loop), clear state, then re-populate from the bundle.
                _data_in   = _obj.get("data") or {}
                _fit_prefs = {k: st.session_state[k]
                              for k in ("fit_tolerance_log", "fit_timeout", "fit_refresh")
                              if k in st.session_state}
                _nonce     = st.session_state.get("_uploader_nonce", 0)
                _exp_nonce = st.session_state.get("_exp_uploader_nonce", 0)
                _tab_meta  = {k: st.session_state[k] for k in eqtabs.META if k in st.session_state}
                st.session_state.clear()
                st.session_state.update(_fit_prefs)
                st.session_state.update(_tab_meta)
                st.session_state["_uploader_nonce"]     = _nonce + 1
                st.session_state["_exp_uploader_nonce"] = _exp_nonce + 1
                st.session_state["_script_text"]        = _obj.get("script", "")
                st.session_state["_script_filename"]    = (
                    _obj.get("script_filename") or "restored.txt"
                )
                # Re-parse the embedded data into the mode-appropriate
                # session dict (_nmr_data / _exp_data / _spectra_data)
                # — the rest of the app looks at THOSE keys, not the
                # raw bytes — by re-parsing the script to detect mode.
                # ALSO: set ``_input_filename`` (the key that drives the
                # "📄 file loaded" badge in the data-uploader area) and
                # ``_pending_xmax`` (the auto-fit x-axis end), because
                # the live uploader sets both of those alongside the
                # data dict.  Without _input_filename the badge stays
                # empty and the user can't tell that the data is
                # actually loaded — they assume it isn't and re-upload.
                if _data_in and "bytes" in _data_in:
                    st.session_state["_exp_data_bytes"]    = _data_in["bytes"]
                    st.session_state["_exp_data_filename"] = _data_in.get(
                        "filename", "restored_data.xlsx")
                    st.session_state["_input_filename"]    = _data_in.get(
                        "filename", "restored_data.xlsx")
                    try:
                        _r_parsed = parse_script(_obj.get("script", ""))
                        _raw = _data_in["bytes"]
                        if _r_parsed.get("spectra") is not None:
                            _loaded_r = load_spectra_data(_raw)
                            st.session_state["_spectra_data"] = _loaded_r
                            # Default wavelength range to the spectrum extent
                            if _loaded_r and len(_loaded_r.get("wavelengths", [])) > 0:
                                st.session_state["_pending_spectra_wl_min"] = \
                                    float(_loaded_r["wavelengths"][0])
                                st.session_state["_pending_spectra_wl_max"] = \
                                    float(_loaded_r["wavelengths"][-1])
                        elif _r_parsed.get("nmr") is not None:
                            _loaded_r = load_experimental_data(_raw)
                            st.session_state["_nmr_data"] = _loaded_r
                        else:
                            _loaded_r = load_experimental_data(_raw)
                            st.session_state["_exp_data"] = _loaded_r
                        # ── xmax restore (2026-05-22 audit) ────────────
                        # Two-tier priority:
                        #
                        #   1. The widget value snapshotted into
                        #      ``fit_state["xmax"]`` at save time (this is
                        #      the on-screen value the user actually saw —
                        #      may be data-derived and differ from the
                        #      script's ``$plot xmax = …``).
                        #
                        #   2. Fallback to the script's ``plot_xmax``,
                        #      which handles older JSONs saved before the
                        #      widget-snapshot was added, and is the most
                        #      reasonable default given the JSON has no
                        #      other record of xmax.
                        #
                        # NB.  The previous behaviour wrote the raw
                        # ``v_add_mL`` value (or x_vals[-1] for spectra)
                        # directly into ``_pending_xmax``, producing a
                        # mL-vs-script-units mismatch — tutorial 5: 0.15
                        # mL appeared as xmax = 0.2 instead of H0/G0 = 3.0,
                        # off by ~15× (caught by the unit conversion in
                        # the live data-upload path at line ~4972-4984,
                        # which routes through ``convert_exp_x``; that
                        # helper isn't usable here because ``params`` /
                        # ``network`` / ``x_expr_default`` aren't built
                        # until later in the rerun).
                        _xmax_saved = (_obj.get("fit_state") or {}).get("xmax")
                        _pending_xmax_set = False
                        if _xmax_saved is not None:
                            try:
                                st.session_state["_pending_xmax"] = float(_xmax_saved)
                                _pending_xmax_set = True
                            except (TypeError, ValueError):
                                pass
                        if not _pending_xmax_set:
                            _px = _r_parsed.get("plot_xmax")
                            if _px is not None:
                                st.session_state["_pending_xmax"] = float(_px)
                    except Exception as _pe:
                        st.session_state["_restore_data_warning"] = str(_pe)
                # Push fitted values into the _pending_* keys so the
                # sidebar widgets show them as starting values on the
                # next rerun — same channel the fit engine uses after
                # a successful fit.
                _summary = _obj.get("fit_result_summary") or {}
                if _summary:
                    for _name, _val in (_summary.get("param_values") or {}).items():
                        try:
                            st.session_state[f"_pending_logK_{_name}"] = float(_val)
                        except Exception:
                            pass
                    for _root, _mM in (_summary.get("fitted_concs") or {}).items():
                        try:
                            st.session_state[f"_pending_conc_{_root}"] = float(_mM)
                        except Exception:
                            pass
                    # fitted_titrants: parallel to fitted_concs.  Missing
                    # pre-audit (2026-05-22) — without this, titrant
                    # stock concentrations saved as "free parameters"
                    # came back at their script default values, then the
                    # auto-refit re-fit them from scratch instead of
                    # warm-starting at the previously-converged value.
                    for _tkey, _mM in (_summary.get("fitted_titrants") or {}).items():
                        try:
                            st.session_state[f"_pending_titrant_mM_{_tkey}"] = float(_mM)
                        except Exception:
                            pass
                    st.session_state["_restored_fit_summary"] = _summary

                # ── Restore _fit_stats so the diagnostic panels
                # (correlation heatmap, identifiability panel,
                # residual-diagnostic figure, AIC/BIC/F-test) render
                # immediately, without waiting for the auto-refit
                # below to complete (or in the case where there's
                # nothing to refit because the JSON was saved with
                # all parameters frozen).  Re-inflate list-shaped
                # fields back to numpy arrays so downstream consumers
                # don't need to know whether they came from a fresh
                # fit or a JSON restore.
                if _summary:
                    _fs = dict(_summary)
                    _pc = _fs.get("param_cov")
                    if _pc is not None:
                        try:
                            _fs["param_cov"] = np.asarray(_pc, dtype=float)
                        except Exception:
                            _fs["param_cov"] = None
                    for _arr_key in ("residuals", "y_calc", "y_obs"):
                        if _fs.get(_arr_key) is not None:
                            try:
                                _fs[_arr_key] = np.asarray(_fs[_arr_key],
                                                            dtype=float)
                            except Exception:
                                pass
                    st.session_state["_fit_stats"] = _fs

                # ── Restore the bootstrap / jackknife / Monte Carlo
                # session-state keys so those UQ panels also render
                # without requiring the user to click "Run bootstrap"
                # (etc.) again after a JSON reload.  The keys are
                # mode-suffixed (`_eq` for equilibrium fits, `_kin`
                # for kinetics fits) — pick the right one based on
                # the `is_kinetics` flag captured in fit_result_summary.
                _uq_in = _obj.get("uq") or {}
                _ws_in = _obj.get("widget_state")
                _pc_in = _obj.get("profile_caches")
                # Set the one-shot preserve flag iff at least one
                # diagnostic-state field is going to be restored.
                # Without this, the upcoming auto-refit's
                # ``_reset_diagnostic_toggles()`` will wipe the
                # restored UQ / LST / profile-cache / widget-toggle
                # entries before any panel can render them.
                if _uq_in or isinstance(_ws_in, dict) or (
                        isinstance(_pc_in, list) and _pc_in):
                    st.session_state["_preserve_uq_on_next_reset"] = True
                if _uq_in:
                    # Determine equilibrium vs kinetics so we can write
                    # the UQ data to the session_state suffix the UI
                    # actually reads.  Prefer the explicit `is_kinetics`
                    # flag introduced in v2 of the session schema; for
                    # legacy JSONs that lack it, fall back to inferring
                    # from `fit_mode` (the two non-NMR kinetics modes
                    # use uniquely kinetic mode strings).
                    _fit_mode_str = str(_summary.get("fit_mode", ""))
                    _is_kin = (
                        bool(_summary.get("is_kinetics", False))
                        or _fit_mode_str in ("kinetics", "kinetics_spectra")
                    )
                    _suf = "_kin" if _is_kin else "_eq"
                    if isinstance(_uq_in.get("bootstrap"), dict):
                        st.session_state[f"_bs_result{_suf}"] = _uq_in["bootstrap"]
                    if isinstance(_uq_in.get("jackknife"), dict):
                        st.session_state[f"_jk_result{_suf}"] = _uq_in["jackknife"]
                    if isinstance(_uq_in.get("monte_carlo"), dict):
                        st.session_state[f"_mc_result{_suf}"] = _uq_in["monte_carlo"]
                    # Masson ξ (LST) result — same kin/eq suffix
                    # convention.  Stored under ``_lst_result_*`` so
                    # the LST UI panel re-renders the saved table
                    # and per-paired-difference transparency view
                    # without recomputation.
                    if isinstance(_uq_in.get("lst"), dict):
                        st.session_state[f"_lst_result{_suf}"] = _uq_in["lst"]
                # Restore the diagnostic-widget input values so the
                # boxes display what was actually used during the
                # saved diagnostics rather than their defaults.  Done
                # before the widgets render in the current rerun;
                # ``st.rerun()`` below propagates these into the
                # widget cache on the next pass.
                if isinstance(_ws_in, dict):
                    for _wk, _wv in _ws_in.items():
                        if isinstance(_wk, str):
                            st.session_state[_wk] = _wv
                # Restore tuple-keyed profile caches (1-D & 2-D
                # RMSE).  Encoded keys are reconstructed into tuples
                # via ``_from_json_key``.  Older JSONs saved before
                # the cache-key was simplified carry an extra
                # trailing element — a sorted tuple of
                # ``(param_name, fitted_value)`` pairs — that the
                # current UI no longer includes when looking up.  We
                # strip that trailing element off old-format keys so
                # legacy JSONs still light up the panels on reload
                # without forcing the user to re-save.
                if isinstance(_pc_in, list) and _pc_in:
                    try:
                        from equilibrist_session import _from_json_key
                        # Expected key shapes (post-strip):
                        #   ("_prof_cache_*", span, npts)              len 3
                        #   ("_p2d_cache_*", p1, p2, sx, sy, np)       len 6
                        _expected_lens = {"_prof_cache_": 3,
                                          "_p2d_cache_":  6,
                                          "_lst_cache_":  None}
                        _registry = st.session_state.setdefault(
                            "_tuple_cache_registry", [])
                        for _entry in _pc_in:
                            if not (isinstance(_entry, list)
                                    and len(_entry) == 2):
                                continue
                            _k_decoded = _from_json_key(_entry[0])
                            if not (isinstance(_k_decoded, tuple)
                                    and _k_decoded
                                    and isinstance(_k_decoded[0], str)):
                                continue
                            _prefix_match = next(
                                (p for p in _expected_lens
                                 if _k_decoded[0].startswith(p)), None)
                            if (_prefix_match
                                    and _expected_lens[_prefix_match]
                                    is not None
                                    and len(_k_decoded)
                                    > _expected_lens[_prefix_match]
                                    and isinstance(_k_decoded[-1], tuple)):
                                # Old-format key — trim the trailing
                                # param_values tuple.
                                _k_decoded = _k_decoded[:_expected_lens[
                                    _prefix_match]]
                            st.session_state[_k_decoded] = _entry[1]
                            # Seed the registry so a subsequent save
                            # (without an intervening recompute) can
                            # still find and persist this cache entry.
                            if _k_decoded not in _registry:
                                _registry.append(_k_decoded)
                    except Exception:
                        pass

                # ── Restore which parameters were being fitted ────────
                # The "fit" checkbox next to each K / concentration /
                # titrant input is what tells the fitter "this is a
                # free parameter".  Their session_state keys are
                # ``fit_logK_<kname>``, ``fit_conc_<root>``, and
                # ``fit_titrant_mM_<tkey>``.  Without flipping these
                # on, the restored UI shows all the fitted values but
                # the fit engine has no parameters to fit — the auto-
                # rerun completes instantly (with nothing to do) and
                # the "Re-running the fit" banner appears stuck.
                _fit_state_in = _obj.get("fit_state") or {}
                for _kname in (_fit_state_in.get("fit_keys") or []):
                    st.session_state[f"fit_logK_{_kname}"] = True
                for _croot in (_fit_state_in.get("fit_conc_keys") or []):
                    st.session_state[f"fit_conc_{_croot}"] = True
                for _tkey in (_fit_state_in.get("fit_titrant_keys") or []):
                    st.session_state[f"fit_titrant_mM_{_tkey}"] = True

                # Auto-trigger the fit so the plot, the fitted-curve
                # overlay, and the diagnostic panel reconstruct
                # themselves automatically — but only if there was a
                # fit to begin with (skip when the bundle was saved
                # before any fit had run).
                if _summary and (
                    _fit_state_in.get("fit_keys") or
                    _fit_state_in.get("fit_conc_keys") or
                    _fit_state_in.get("fit_titrant_keys")
                ):
                    st.session_state["_fit_requested"] = True

                # One-shot success banner shown by the block below.
                st.session_state["_just_restored"] = {
                    "notes":      _obj.get("notes", ""),
                    "saved_at":   _obj.get("saved_at_utc", ""),
                    "data_file":  _data_in.get("filename", ""),
                    "will_refit": bool(_summary) and bool(
                        _fit_state_in.get("fit_keys") or
                        _fit_state_in.get("fit_conc_keys") or
                        _fit_state_in.get("fit_titrant_keys")
                    ),
                }
                st.rerun()
        else:
            # ── .txt script upload (original behavior) ────────────────
            new_text = uploaded.read().decode("utf-8", errors="replace")
            _fit_prefs = {k: st.session_state[k]
                          for k in ("fit_tolerance_log", "fit_timeout", "fit_refresh")
                          if k in st.session_state}
            _nonce = st.session_state.get("_uploader_nonce", 0)
            _tab_meta = {k: st.session_state[k] for k in eqtabs.META if k in st.session_state}
            st.session_state.clear()
            st.session_state.update(_fit_prefs)
            st.session_state.update(_tab_meta)
            st.session_state["_uploader_nonce"] = _nonce + 1
            st.session_state["_script_text"] = new_text
            st.session_state["_script_filename"] = uploaded.name
            st.rerun()

    # Brief success banner after a restore
    _jr = st.session_state.pop("_just_restored", None)
    if _jr:
        if _jr.get("will_refit"):
            st.success("Session restored — re-running the fit to "
                       "reconstruct the plot and diagnostics…")
        else:
            st.success("Session restored.  Click **🔧 Fit Parameters** "
                       "to compute the result.")
    if st.session_state.pop("_restore_data_warning", None):
        st.warning("Embedded data could not be re-parsed.  "
                   "Re-upload the data file manually.")

    if st.session_state.get("_script_filename"):
        st.markdown(f"<span style='background:#1a6bbf;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:0.82rem'>📄 {st.session_state['_script_filename']}</span>", unsafe_allow_html=True)

    # ── Tabs (workspaces) ────────────────────────────────────────────────
    if st.session_state.pop("_tab_notice", None):
        pass    # the swap already happened at the top of this run
    _tab_labels = eqtabs.labels(st.session_state)
    if len(_tab_labels) > 1:
        _tab_shares = [eqtabs.global_share(st.session_state, _i) for _i in range(len(_tab_labels))]
        _tab_disp = [_l if _s is None else f"{_l}  ·  {100 * _s:.0f} %"
                     for _l, _s in zip(_tab_labels, _tab_shares)]
        st.radio("Tabs", list(range(len(_tab_labels))), key="_tab_radio",
                 format_func=lambda i: _tab_disp[i],
                 help="Each tab is an independent workspace (script, data, constants, fit).  "
                      "Switch by clicking; the Global fit button fits all tabs at once.  "
                      "After a global fit the percentage next to a tab is its share of the "
                      "information on the shared constants: how much of the curvature that pins "
                      "them comes from this tab, averaged over the constants (the per-constant "
                      "shares are in the fit message).")
        if st.button("✖ Close this tab", key="_tab_close_btn"):
            st.session_state["_tab_close"] = True
            st.rerun()

    # ── Session save (FAIR-compliance) ───────────────────────────────────
    # Bundle the script, the uploaded experimental data (verbatim, with
    # a SHA-256 integrity check), the per-fit configuration, and a
    # compact summary of the last fit result into a single JSON file.
    # The file is human-inspectable, self-contained, and reloadable via
    # the main script uploader above (which now accepts .json too).
    with st.expander("💾 Save current session", expanded=False):
        st.caption(
            "Download a self-contained JSON bundle (script + data file "
            "+ last fit result).  Restore later by dropping the .json "
            "into the uploader above."
        )
        _script_now = st.session_state.get("_script_text", "")
        if _script_now:
            try:
                from equilibrist_session import (make_session_json,
                                                   summarize_stats)
                _data_bytes = st.session_state.get("_exp_data_bytes")
                _data_name  = st.session_state.get("_exp_data_filename")
                _fit_stats  = st.session_state.get("_fit_stats") or {}
                _fa_eq      = st.session_state.get("_fit_args_eq")
                _fa_kin     = st.session_state.get("_fit_args_kin")
                # Pick whichever fit_args dict is populated — kinetics
                # branch wins if both exist (because the equilibrium
                # block runs only after kinetics returns).
                _fa_save = _fa_kin or _fa_eq or {}
                _fit_state_save = {
                    k: _fa_save.get(k) for k in (
                        "kind", "fit_keys", "fit_conc_keys",
                        "fit_titrant_keys", "tolerance", "maxiter",
                        "timeout_s", "wl_min", "wl_max",
                        "use_spectra_fit", "use_nmr_fit", "nmr_mode",
                        "allow_neg_eps", "constraints",
                    ) if k in _fa_save
                }
                _start = _fa_save.get("start_logK") or _fa_save.get("start_logk")
                if _start:
                    _fit_state_save["start_logK"] = dict(_start)
                # ── xmax snapshot (2026-05-22 audit) ────────────────────
                # The widget value of ``xmax`` can have drifted from the
                # script's ``$plot xmax = …`` after a data upload, because
                # the live data-load path overrides xmax with the
                # data-derived value via ``convert_exp_x`` (tutorial 13:
                # script says ``xmax = 3``, data covers H0/G0 only to
                # ~0.8 → widget snaps to 0.8).  The script text is NOT
                # rewritten when this happens, so without snapshotting
                # the widget value here the JSON loses the user's actual
                # on-screen xmax.  See the restore block for the read
                # side, which prefers this value over ``plot_xmax``.
                #
                # Priority: ``_pending_xmax`` (queued by a recent data
                # upload or restore but NOT yet consumed by the xmax
                # widget's ``_num_input`` call — the save block runs
                # BEFORE the kinetics sidebar that owns the widget, so
                # the pending value is the most up-to-date intent) →
                # ``session_state["xmax"]`` (the current widget value,
                # already a post-consume snapshot).  Without consulting
                # the pending key first, a save immediately after a
                # data upload captures the stale pre-upload widget
                # value (tutorial 11 pre-fit save: script xmax=3, data
                # range 0-20 → JSON would capture 3 instead of 20).
                _xmax_widget = st.session_state.get(
                    "_pending_xmax", st.session_state.get("xmax"))
                if _xmax_widget is not None:
                    try:
                        _fit_state_save["xmax"] = float(_xmax_widget)
                    except (TypeError, ValueError):
                        pass
                _notes = st.text_input(
                    "Notes (optional)", value="",
                    key="_session_notes",
                    placeholder="e.g. 1:2 binding, run 3, repeat of Fig 4a"
                )
                # One-click save.  Why a click-triggered button +
                # auto-download rather than the obvious always-on
                # ``st.download_button``: the latter bakes its bytes
                # at *render* time, but Streamlit fragments (which
                # host every diagnostic — bootstrap, jackknife, MC,
                # 1-D / 2-D RMSE, Masson ξ) re-run independently of
                # the main script, so any interaction inside a
                # diagnostic panel (e.g. clicking "Compute profile"
                # with new span/npts) DOES NOT re-render the save
                # expander.  An eager download_button at the top of
                # the page would therefore keep serving the stale
                # JSON baked at the previous full rerun — the user's
                # most-recent UQ work would be missing.
                #
                # Clicking a regular ``st.button`` outside any
                # fragment forces a full script rerun, on which
                # ``make_session_json`` reads the current
                # session_state (including whatever the fragments
                # populated on their internal reruns) and builds a
                # fresh JSON.  The bytes are then handed to the
                # browser via a ``<a download>`` whose click is
                # immediately fired by an injected ``<script>``,
                # so the file downloads without a second user
                # action.
                if st.button("💾 Save session JSON",
                              key="_session_save_btn",
                              type="primary"):
                    from equilibrist_session import (
                        _capture_widget_state,
                        _serialize_tuple_caches,
                    )
                    _lst_for_save = (
                        st.session_state.get("_lst_result_kin")
                        or st.session_state.get("_lst_result_eq")
                    )
                    _session_json = make_session_json(
                        script=_script_now,
                        script_filename=st.session_state.get("_script_filename", ""),
                        data_bytes=_data_bytes,
                        data_filename=_data_name,
                        fit_state=_fit_state_save,
                        fit_result_summary=summarize_stats(_fit_stats),
                        bootstrap_result=(st.session_state.get("_bs_result_kin")
                                           or st.session_state.get("_bs_result_eq")),
                        jackknife_result=(st.session_state.get("_jk_result_kin")
                                           or st.session_state.get("_jk_result_eq")),
                        monte_carlo_result=(st.session_state.get("_mc_result_kin")
                                             or st.session_state.get("_mc_result_eq")),
                        lst_result=_lst_for_save,
                        widget_state=_capture_widget_state(st.session_state),
                        profile_caches=_serialize_tuple_caches(st.session_state),
                        seed=42,
                        notes=_notes,
                        app_version="equilibrist-v3.0-beta",
                    )
                    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _stem = "session"
                    if _data_name:
                        import os as _os
                        _stem = _os.path.splitext(_data_name)[0] or _stem
                    _fname = f"Equilibrist_{_stem}_{_ts}.json"
                    # Trigger the browser download by injecting a
                    # hidden anchor + a script that clicks it.  The
                    # data is encoded inline as a base64 data URL so
                    # no server round-trip is needed.  Streamlit
                    # sandboxes the injected HTML inside an iframe
                    # but ``<a download>`` works inside the sandbox
                    # because Streamlit's iframe sandbox includes
                    # ``allow-downloads``.
                    import base64 as _base64
                    from streamlit.components.v1 import html as _components_html
                    _b64 = _base64.b64encode(
                        _session_json.encode("utf-8")).decode("ascii")
                    _components_html(
                        f"""
                        <a id="_eq_autodl"
                           href="data:application/json;base64,{_b64}"
                           download="{_fname}"
                           style="display:none">.</a>
                        <script>
                          document.getElementById("_eq_autodl").click();
                        </script>
                        """,
                        height=0,
                    )
                    st.success(f"Saved as `{_fname}`")
            except Exception as _se:
                st.warning(f"Could not build session bundle: {_se}")
        else:
            st.caption("Load or write a script above to enable saving.")
    # ── End session save ────────────────────────────────────────────────


# Live progress of supercomputer jobs: a small panel fixed to the bottom-right
# corner of the window, so it stays visible however the page or the sidebar
# is scrolled.  Top level on purpose (see hpc_ui.follower).
hpc_ui.follower()

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
                    if _new_parsed.get("is_acid_base") and _eq["kname"] != "Kw":
                        # acid-base mode: push as pKa so the pKa widget picks it up
                        st.session_state[f"_pending_pKa_{_eq['kname']}"] = -float(_eq["logK"])
                    else:
                        st.session_state[f"_pending_logK_{_eq['kname']}"] = float(_eq["logK"])
                for _rxn in _new_parsed.get("kinetics", []):
                    st.session_state[f"_pending_logK_{_rxn['kname']}"] = float(_rxn["log_k"])
                    if "krname" in _rxn:
                        st.session_state[f"_pending_logK_{_rxn['krname']}"] = float(_rxn["log_kr"])
                # Concentrations.
                # A bare name in an ITC script ("G0" with no value) parses to
                # None: it is a placeholder to be filled from the .itc header,
                # not a value to push into a widget. Skipping it leaves the
                # value already loaded from the file untouched; float(None)
                # used to raise TypeError here.
                for _cname, _cval in _new_parsed.get("concentrations", {}).items():
                    if _cval is None:
                        continue
                    _root = _cname[:-1] if _cname.endswith("0") else _cname
                    st.session_state[f"_pending_conc_{_root}"] = float(_cval)
                # Volume
                _vols = _new_parsed.get("volumes", {})
                if _vols:
                    _v0_new = list(_vols.values())[0]
                    if _v0_new is not None:
                        st.session_state["_pending_V0_mL"] = float(_v0_new)
                # Titrant concentrations
                for _tkey, _tval in _new_parsed.get("titrant", {}).items():
                    if _tval is None:
                        continue
                    st.session_state[f"_pending_titrant_mM_{_tkey}"] = float(_tval)
                # xmax / plot settings
                if _new_parsed.get("plot_xmax") is not None:
                    st.session_state["_pending_xmax"] = float(_new_parsed["plot_xmax"])
            else:
                # Hard reset: new/renamed parameters → rebuild everything
                _fit_prefs = {k: st.session_state[k]
                              for k in ("fit_tolerance_log", "fit_timeout", "fit_refresh")
                              if k in st.session_state}
                _nonce    = st.session_state.get("_uploader_nonce", 0)
                _fname    = st.session_state.get("_script_filename", "untitled.txt")
                _tab_meta = {k: st.session_state[k] for k in eqtabs.META if k in st.session_state}
                st.session_state.clear()
                st.session_state.update(_fit_prefs)
                st.session_state.update(_tab_meta)     # the other tabs survive a hard reset
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

# ── Clear outliers when the script changes ────
_script_hash_now = hash(script_text)
if st.session_state.get("_last_script_hash") != _script_hash_now:
    for _ok in ("_outliers_main", "_outliers_nmr", "_outliers_spectra"):
        st.session_state.pop(_ok, None)
    st.session_state["_last_script_hash"] = _script_hash_now

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
    # Reserved slot for the 'Fitting…' message.  Emitted on every run so the
    # element that follows (the two-column layout) keeps the same position;
    # otherwise the spinner, appearing only on fitting runs, took the columns'
    # place and the right-hand panel vanished until the fit returned.
    _fit_slot = st.empty()
    # ── What a Global fit would use from THIS (kinetics) tab, refreshed every run ──
    try:
        _nmr_live_k = _filter_exp_outliers(st.session_state.get("_nmr_data", {}), "_outliers_nmr")
        _exp_live_k = _filter_exp_outliers(st.session_state.get("_exp_data", {}), "_outliers_main")
        _sp_live_k  = _filter_spectra_outliers(st.session_state.get("_spectra_data", {}), "_outliers_spectra")
        _nmr_cfg_live_k = parsed.get("nmr")
        _use_nmr_live_k = (_nmr_cfg_live_k is not None and _nmr_cfg_live_k["mode"] in ("shift", "integration", "mixed")
                           and bool(_nmr_live_k))
        _use_sp_live_k = parsed.get("spectra") is not None and bool(_sp_live_k)
        _concs_live_k = {}
        for _cname, _cval in parsed["concentrations"].items():
            _root = _cname[:-1] if _cname.endswith("0") else _cname
            _concs_live_k[_cname] = float(st.session_state.get(f"conc_{_root}", _cval))
        _parsed_live_k = dict(parsed); _parsed_live_k["concentrations"] = _concs_live_k
        _logk_live = {n: float(st.session_state.get(f"logK_{n}", logk_dict[n])) for n in logk_dict}
        _fa_live_k = {
            "kind": "kin",
            "fit_keys": [n for n in logk_dict if st.session_state.get(f"fit_logK_{n}", False)],
            "fit_conc_keys": [root for root in (c[:-1] if c.endswith("0") else c for c in parsed["concentrations"])
                              if st.session_state.get(f"fit_conc_{root}", False)],
            "fit_titrant_keys": [],
            "use_spectra_fit": bool(_use_sp_live_k), "use_nmr_fit": bool(_use_nmr_live_k),
            "nmr_mode": (_nmr_cfg_live_k["mode"] if _use_nmr_live_k else None),
            "t_max": float(st.session_state.get("xmax", t_max)),
            "tolerance": 10.0 ** (-float(st.session_state.get("fit_tolerance_log", 3.0))),
            "maxiter": 100_000, "timeout_s": float(st.session_state.get("fit_timeout", 30)),
            "constraints": (_parsed_live_k.get("constraints", []) if st.session_state.get("fit_use_constraints", False) else []),
            "start_logk": dict(_logk_live), "start_logK": dict(_logk_live),
            "exp_data_fit": (_exp_live_k if not (_use_nmr_live_k or _use_sp_live_k) else None),
            "nmr_data_fit": (_nmr_live_k if _use_nmr_live_k else None),
            "spectra_data_fit": (_sp_live_k if _use_sp_live_k else None),
            "parsed_fit": _parsed_live_k, "x_expr": _parsed_live_k.get("plot_x_expr", "t"),
            "wl_min": (float(st.session_state.get("spectra_wl_min", _sp_live_k["wavelengths"][0])) if _use_sp_live_k else None),
            "wl_max": (float(st.session_state.get("spectra_wl_max", _sp_live_k["wavelengths"][-1])) if _use_sp_live_k else None),
            "allow_neg_eps": bool(st.session_state.get("spectra_allow_neg", False)),
            "has_data": bool(_use_nmr_live_k or _use_sp_live_k or _exp_live_k),
        }
        if _use_nmr_live_k:
            try:
                from equilibrist_network import build_network as _bn_live
                _fa_live_k["network"] = _bn_live(_parsed_live_k)
            except Exception:
                _fa_live_k["network"] = None
            _fa_live_k["params"] = {"conc0": {k[:-1] if k.endswith("0") else k: float(v) for k, v in _concs_live_k.items()},
                                    "V0_mL": float(list(_parsed_live_k.get("volumes", {1: 1}).values())[0])}
        else:
            _fa_live_k["params"] = {"conc0": {k[:-1] if k.endswith("0") else k: float(v) for k, v in _concs_live_k.items()}}
        st.session_state["_fit_args_live"] = _fa_live_k
    except Exception:
        st.session_state.pop("_fit_args_live", None)
    if _fit_turn(_fit_slot, "_global_fit_requested"):
        _global_fit_dispatch(_fit_slot)
    if st.session_state.get("_global_fit_pending"):
        _apply_global_pending(st.session_state.pop("_global_fit_pending"))
    if st.session_state.get("_single_fit_pending"):
        _apply_single_pending(st.session_state.pop("_single_fit_pending"))
    if st.session_state.get("_fit_requested") and st.session_state.get("_single_fit_hpc"):
        st.session_state.pop("_fit_requested", None)
        _submit_single_fit_to_cluster(st.session_state.pop("_single_fit_hpc"))
    if _fit_turn(_fit_slot):
        _nmr_data_fit     = _filter_exp_outliers(st.session_state.get("_nmr_data", {}),     "_outliers_nmr")
        exp_data_fit      = _filter_exp_outliers(st.session_state.get("_exp_data", {}),     "_outliers_main")
        _spectra_data_fit = _filter_spectra_outliers(st.session_state.get("_spectra_data", {}), "_outliers_spectra")
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
                         st.session_state.get("fit_tolerance_log", 3.0))
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

                # The constraint toggle flipping does NOT reset the K's to
                # script defaults — the sidebar K's are typically a much better
                # starting point.  Soft constraint violations are absorbed by
                # the penalty term in the first few iterations.
                _script_defaults_k = {**{e["kname"]: e["logK"]   for e in parsed_fit["equilibria"]},
                                      **{r["kname"]: r["log_k"]  for r in parsed_fit.get("kinetics", [])},
                                      **{r["krname"]: r["log_kr"] for r in parsed_fit.get("kinetics", []) if "krname" in r}}
                _start_logk = current_logk

                st.session_state["_last_fit_was_constrained"] = _this_fit_constrained

                if _use_spectra_fit:
                    wl_min_k = float(st.session_state.get("spectra_wl_min",
                                     _spectra_data_fit["wavelengths"][0]))
                    wl_max_k = float(st.session_state.get("spectra_wl_max",
                                     _spectra_data_fit["wavelengths"][-1]))
                    _auto_range_k   = bool(st.session_state.get("spectra_auto_range", False))
                    _allow_neg_k    = bool(st.session_state.get("spectra_allow_neg", False))
                    success, fitted, stats, msg = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_kinetics_spectra(
                            parsed_fit, _start_logk, _spectra_data_fit, fit_keys_k,
                            t_max, 200, wl_min_k, wl_max_k, tol, maxiter,
                            timeout_s=_timeout_s, auto_range=_auto_range_k, allow_negative_eps=_allow_neg_k,
                            use_lbfgsb=st.session_state.get("fit_use_lbfgsb", True),
                            use_neldermead=st.session_state.get("fit_use_neldermead", True),
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k),
                        _start_logk, "Fitting parameters…", "_fit_requested")
                    if _auto_range_k and "opt_wl_min" in stats:
                        st.session_state["_pending_spectra_wl_min"] = stats["opt_wl_min"]
                        st.session_state["_pending_spectra_wl_max"] = stats["opt_wl_max"]
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "shift":
                    success, fitted, stats, msg = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_kinetics_nmr_shifts(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k),
                        _start_logk, "Fitting parameters…", "_fit_requested")
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "integration":
                    success, fitted, stats, msg = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_kinetics_nmr_integration(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k),
                        _start_logk, "Fitting parameters…", "_fit_requested")
                elif _use_nmr_fit and _nmr_cfg_fit["mode"] == "mixed":
                    success, fitted, stats, msg = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_kinetics_nmr_mixed(
                            parsed_fit, _start_logk, _nmr_data_fit, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k),
                        _start_logk, "Fitting parameters…", "_fit_requested")
                else:
                    success, fitted, stats, msg = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_kinetics(
                            parsed_fit, exp_data_fit, _start_logk, fit_keys_k,
                            t_max, 200, tol, maxiter, timeout_s=_timeout_s,
                            constraints=_active_constraints,
                            fit_conc_keys=fit_conc_keys_k),
                        _start_logk, "Fitting parameters…", "_fit_requested")
                for name, val in fitted.items():
                    st.session_state[f"_pending_logK_{name}"] = float(val)
                # Push fitted concentration results back to sidebar
                for _root, _mM_val in stats.get("fitted_concs", {}).items():
                    st.session_state[f"_pending_conc_{_root}"] = float(_mM_val)
                # ── v2: canonicalise param_values across all kin fit modes ─
                # Same logic as the equilibrium branch: ensure
                # fit_stats["param_values"] is populated regardless of
                # which kinetics module ran, so the Local sensitivity
                # test (and any future diagnostic that needs the
                # optimum) has one consistent place to read from.
                if isinstance(stats, dict):
                    stats.setdefault("param_values", dict(fitted))
                # Reset diagnostic toggles + cached results so the user
                # opts back in for the new fit (stale "show ..." plots
                # were a real-world confusion from earlier sessions).
                # Also bump the fit-counter — the parent diag-panel
                # checkbox uses this as a key suffix so Streamlit's
                # fragment-internal widget cache can't keep it stuck
                # checked across fits (a quirk where programmatic
                # writes to st.session_state don't always reach the
                # fragment's cached widget state for the top-level
                # widget in the fragment).
                st.session_state["_fit_counter"] = int(
                    st.session_state.get("_fit_counter", 0)) + 1
                _reset_diagnostic_toggles()
                st.session_state["_fit_stats"] = stats
                # ── v2: save fit args so the bootstrap UI can find them after rerun ──
                # Wrapped in try/except so a missing local (kinetics scope does
                # not define `network`, `params`, or `x_expr` — those live in the
                # equilibrium branch) cannot break the fit pipeline.  Pure
                # kinetics bootstrap does not need any of them; only the
                # kinetics+NMR variants do, and we build them lazily there.
                try:
                    _fa_kin = {
                        "kind":              "kin",
                        "fit_keys":          list(fit_keys_k),
                        "fit_conc_keys":     list(fit_conc_keys_k),
                        "use_spectra_fit":   bool(_use_spectra_fit),
                        "use_nmr_fit":       bool(_use_nmr_fit),
                        "nmr_mode":          (_nmr_cfg_fit["mode"] if _use_nmr_fit and _nmr_cfg_fit else None),
                        "t_max":             float(t_max),
                        "tolerance":         float(tol),
                        "maxiter":           int(maxiter),
                        "timeout_s":         float(_timeout_s),
                        "constraints":       _active_constraints,
                        "start_logk":        dict(_start_logk),
                        "exp_data_fit":      exp_data_fit if not (_use_nmr_fit or _use_spectra_fit) else None,
                        "nmr_data_fit":      _nmr_data_fit     if _use_nmr_fit     else None,
                        "spectra_data_fit":  _spectra_data_fit if _use_spectra_fit else None,
                        "parsed_fit":        parsed_fit,
                    }
                    # x_expr / network / params are only needed for the
                    # kinetics+NMR bootstrap variants; build them lazily and
                    # only when applicable so missing variables in pure-
                    # kinetics scope cannot break this assignment.
                    _fa_kin["x_expr"] = parsed_fit.get("plot_x_expr", "t")
                    if _use_nmr_fit:
                        try:
                            from equilibrist_network import build_network as _bn_kin
                            _fa_kin["network"] = _bn_kin(parsed_fit)
                        except Exception:
                            _fa_kin["network"] = None
                        try:
                            _cv_k = {k[:-1] if k.endswith('0') else k: float(v)
                                     for k, v in parsed_fit.get("concentrations", {}).items()}
                            _V0_k = float(list(parsed_fit.get("volumes", {1: 1}).values())[0])
                            _fa_kin["params"] = {"conc0": _cv_k, "V0_mL": _V0_k}
                        except Exception:
                            _fa_kin["params"] = None
                    # n_data for local-sensitivity test (needed by σ_cond formula)
                    _fa_kin["n_data"] = int(stats.get("n_points", 0))
                    st.session_state["_fit_args_kin"] = _fa_kin
                except Exception:
                    st.session_state["_fit_args_kin"] = None
                # ── Augment stats with AIC/BIC + residuals (v2 diagnostics) ──
                try:
                    # Unified collector reads y_obs/y_calc/residuals + per-col
                    # arrays populated by all v2-patched fit modes.  Passing
                    # ``x_expr`` lets it label the residuals-vs-predictor
                    # axis with the script's actual x expression (e.g.
                    # "H0/G0") instead of generic "Data-point index".
                    _res = collect_residuals_from_stats(
                        stats, x_expr=parsed_fit.get("plot_x_expr", "t"))
                    augment_stats(stats, _res)
                    st.session_state["_fit_residuals"] = _res
                except Exception:
                    st.session_state["_fit_residuals"] = {}
                _n_evals = stats.get("n_iter", 0)
                _n_fitted_k = len(fitted) + len(stats.get("fitted_concs", {}))
                st.session_state["_fit_message"] = _fit_end_message(
                    stats, success, msg, _n_fitted_k, tol_log, _timeout_s)
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
            # Persist the xmax widget value across the rerun.  Same reason
            # as fit_logK_* above: the kinetics sidebar (which renders
            # ``xmax`` via ``_num_input``) runs AFTER this block, so
            # Streamlit GC removes ``session_state['xmax']`` between runs.
            # Without this snapshot, the sidebar save block in the next
            # run (which runs BEFORE the kinetics sidebar) sees no widget
            # value and silently omits ``xmax`` from the saved JSON — so a
            # session saved immediately after a kinetics fit loses the
            # data-derived xmax override (caught 2026-05-22, tutorial 13).
            if "xmax" in st.session_state:
                st.session_state["_persist_xmax"] = st.session_state["xmax"]
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
        _rendered_knames_rate = set()
        for rxn in parsed["kinetics"]:
            kn   = rxn["kname"]
            lbl  = _kinetics_reaction_label(rxn)
            n_r  = sum(c for c, _ in rxn["reactants"])
            n_p  = sum(c for c, _ in rxn["products"])
            u_fwd = _rate_constant_units(n_r)
            if kn in _rendered_knames_rate:
                st.caption(f"**{lbl}**  —  {kn}: {u_fwd}  *(shared with above)*")
                logk_ui[kn] = float(st.session_state.get(f"logK_{kn}", logk_dict[kn]))
            else:
                st.caption(f"**{lbl}**  —  {kn}: {u_fwd}")
                logk_ui[kn] = _k_input_with_fit(
                    kn, key=f"logK_{kn}", default_log=logk_dict[kn])
                _rendered_knames_rate.add(kn)
            if "krname" in rxn:
                krn   = rxn["krname"]
                u_rev = _rate_constant_units(n_p, is_reverse=True, n_products=n_p)
                if krn in _rendered_knames_rate:
                    st.caption(f"{krn}: {u_rev}  *(shared with above)*")
                    logk_ui[krn] = float(st.session_state.get(f"logK_{krn}", logk_dict[krn]))
                else:
                    st.caption(f"{krn}: {u_rev}")
                    logk_ui[krn] = _k_input_with_fit(
                        krn, key=f"logK_{krn}", default_log=logk_dict[krn])
                    _rendered_knames_rate.add(krn)

        if parsed["equilibria"]:
            st.header("Pre-equilibria (= reactions)")
            _rendered_knames_kin = set()
            for eq in parsed["equilibria"]:
                kn  = eq["kname"]
                n_r = sum(c for c, _ in eq["reactants"])
                n_p = sum(c for c, _ in eq["products"])
                lbl = _kinetics_reaction_label({**eq, "type": "equilibrium"})
                units = _equilibrium_constant_units(n_r, n_p)
                if kn in _rendered_knames_kin:
                    # Shared kname — show reaction label as caption only, no duplicate widget
                    st.caption(f"**{lbl}**  —  {kn}: {units}  *(shared with above)*")
                    logk_ui[kn] = float(st.session_state.get(f"logK_{kn}", eq["logK"]))
                else:
                    # Check whether other equilibria share this kname
                    _sharing_kin = [e for e in parsed["equilibria"] if e["kname"] == kn]
                    if len(_sharing_kin) > 1:
                        _all_lbl = " = ".join(
                            _kinetics_reaction_label({**e, "type": "equilibrium"})
                            for e in _sharing_kin)
                        st.caption(f"**{_all_lbl}**  —  {kn}: {units}  *(shared)*")
                    else:
                        st.caption(f"**{lbl}**  —  {kn}: {units}")
                    logk_ui[kn] = _logk_input_with_fit(
                        f"log {kn}", key=f"logK_{kn}", default=eq["logK"])
                    _rendered_knames_kin.add(kn)

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
                # Evaluate the expression with its dependencies on other
                # variables resolved, as in the equilibrium branch; a
                # variable that cannot be evaluated plots as a gap, not as 0.
                y_vals = compute_variable_curve(
                    sp, variables, kin_curve, {"all_species": all_kin_species}, t_vals)
            else:
                y_vals = kin_curve.get(sp, np.zeros_like(t_vals))
            fig.add_trace(go.Scatter(
                x=t_vals, y=y_vals, mode="lines", name=sp,
                line=dict(color=color, width=2)))

        # Experimental overlay — only for quantities listed in $plot y
        exp_data = st.session_state.get("_exp_data", {})
        _kin_plot_y_set = set(plot_y_names) if plot_y_names else None
        _kin_main_excl = st.session_state.get("_outliers_main", {})
        for col_name, col_data in exp_data.items():
            if col_name.startswith("_"):
                continue
            if _kin_plot_y_set is not None and col_name not in _kin_plot_y_set:
                continue
            color = trace_colors.get(col_name, "#FFFFFF")
            _xs_all = col_data["v_add_mL"]
            _ys_all = col_data["y"]
            _excl_set = _kin_main_excl.get(col_name, set())
            _inc = [i for i in range(len(_ys_all)) if i not in _excl_set]
            _exc = [i for i in range(len(_ys_all)) if i in _excl_set]
            if _inc:
                fig.add_trace(go.Scatter(
                    x=_xs_all[_inc], y=_ys_all[_inc],
                    mode="markers", name=f"{col_name} (exp)",
                    customdata=[[col_name, i] for i in _inc],
                    marker=dict(color=color, size=7, symbol="circle",
                                line=dict(width=1, color="white")),
                    showlegend=True,
                ))
            if _exc:
                fig.add_trace(go.Scatter(
                    x=_xs_all[_exc], y=_ys_all[_exc],
                    mode="markers", name=f"_outlier_{col_name}",
                    customdata=[[col_name, i] for i in _exc],
                    marker=dict(color=color, size=7, symbol="circle-open",
                                line=dict(width=1.5, color=color)),
                    showlegend=False,
                ))

        # ── NMR back-calculated concentrations on main plot ─────────────────
        _kin_fit_stats   = st.session_state.get("_fit_stats", {})
        _kin_nmr_data    = st.session_state.get("_nmr_data", {})
        _kin_sp_concs    = _kin_fit_stats.get("sp_concs", {})
        _kin_nmr_cfg_plt = parsed.get("nmr")
        _kin_all_sp_plt  = _collect_all_kinetic_species(parsed)

        # Integration / mixed — post-fit: sp_concs from fit stats
        if _kin_sp_concs and _kin_nmr_data:
            _kin_sfx_integ = "(NMR/integration)" if _kin_nmr_cfg_plt and _kin_nmr_cfg_plt.get("mode") == "mixed" else "(NMR)"
            # Integration back-calc is K-independent — always render from the full
            # unfiltered data so original row indices are correct in customdata.
            _kni_nh = _kin_nmr_cfg_plt.get("n_H_list", [])
            _kni_ni = _kin_nmr_cfg_plt.get("n_integ", len(_kni_nh))
            _kni_cols = [c for c in _kin_nmr_data if not c.startswith("_")][:_kni_ni]
            _kni_data = {c: _kin_nmr_data[c] for c in _kni_cols}
            _kni_full = _kinetics_nmr_integration_backCalc(
                _kni_data, _kni_nh[:_kni_ni], parsed, _kin_all_sp_plt)
            _plot_backcalc_dots(fig, _kni_full, plot_y_names,
                                parsed.get("variables", {}),
                                _kin_all_sp_plt, trace_colors,
                                label_suffix=_kin_sfx_integ, symbol="circle-open",
                                excl_rows=_nmr_excl_intersection(_kin_nmr_data))

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
                                label_suffix=_kin_sfx_pre, symbol="circle-open",
                                excl_rows=_nmr_excl_intersection(_kin_nmr_data))

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
                _c_nf_bc = np.zeros((_n_nf, _n_pts_bc))
                if _n_nf > 0:
                    # model concentrations at the data times, corrected only along
                    # the directions the shift signals resolve
                    _c_model_nf = np.array([np.interp(_t_bc, t_vals, kin_curve.get(sp, np.zeros_like(t_vals)))
                                            for sp in _nf_sp])
                    _sol_bc, _k_res = shift_backcalc_resolved(M_bc, rhs_bc, _c_model_nf)
                    _c_nf_bc = np.clip(_sol_bc, 0.0, None)
                    if _k_res < _n_nf:
                        st.session_state["_shift_bc_note_kin"] = (
                            f"Shift-derived points: the {_n_sig_bc} shift signal{'s' if _n_sig_bc != 1 else ''} resolve "
                            f"{_k_res} combination{'s' if _k_res != 1 else ''} of the {_n_nf} species of the target "
                            f"({', '.join(_nf_sp)}); along the others the points follow the model.")
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
                                    label_suffix=_kin_sfx_shift,
                                    excl_rows=_nmr_excl_intersection(_kin_nmr_data))

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
                                    label_suffix="(UV-Vis)",
                                    excl_rows=st.session_state.get("_outliers_spectra", set()),
                                    bc_tag="__uvvis_bc__")

        if not kin_curve.get("success", True):
            st.warning("⚠️ ODE integrator did not fully converge.")

        _kin_net_fake = {"all_species": _collect_all_kinetic_species(parsed)}
        _kin_y_label  = _infer_y_label(plot_y_names, parsed, _kin_net_fake)
        _kin_rangemode = None if _has_log_units(plot_y_names, parsed, _kin_net_fake) else "tozero"
        _kin_yaxis = dict(title=_kin_y_label)
        if _kin_rangemode is None:
            _kin_yaxis["autorange"] = True
        else:
            _kin_yaxis["rangemode"] = _kin_rangemode
        fig.update_layout(
            height=700,
            margin=dict(l=40, r=20, t=40, b=120),
            xaxis=dict(title="Time [s]"),
            yaxis=_kin_yaxis,
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        )
        _kin_ver_main = st.session_state.get("_outlier_ver__outliers_main", 0)
        # Suppress Plotly's built-in selection dimming so only our hollow markers
        # communicate exclusion state, not Plotly's opacity reduction.
        fig.update_traces(
            unselected=dict(marker=dict(opacity=1.0)),
            selected=dict(marker=dict(opacity=1.0, size=10)),
            selector=dict(mode="markers"),
        )
        _kin_main_event = st.plotly_chart(
            fig, width='stretch',
            on_select="rerun", selection_mode="points",
            key=f"_kin_main_chart_{_kin_ver_main}",
        )
        if _process_outlier_event(_kin_main_event, "_outliers_main",
                                   nmr_bc_cols=[c for c in _kin_nmr_data if not c.startswith("_")]):
            st.rerun()
        _outlier_bar("kin_main", "_outliers_main", "_outliers_nmr", "_outliers_spectra")
        if st.session_state.get("_shift_bc_note_kin"):
            st.caption(st.session_state.pop("_shift_bc_note_kin"))
        _pub_download_button(fig, "kinetics_main", y_label=_kin_y_label)
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
            _kin_nmr_excl = st.session_state.get("_outliers_nmr", {})
            for _col in _shift_cols_k:
                _col_data = _kin_nmr_data[_col]
                _t_exp    = _col_data["v_add_mL"]
                _df0      = float(_kin_delta_free.get(_col, _col_data["y"][0]))
                _dobs_rel = _col_data["y"] - _df0
                _tgt      = _kin_c2t.get(_col, _col)
                _color    = _kin_tgt_colors.get(_tgt, "#888888")
                _show_leg = _tgt not in _shown_tgt; _shown_tgt.add(_tgt)
                _excl_set_n = _kin_nmr_excl.get(_col, set())
                _inc_n = [i for i in range(len(_dobs_rel)) if i not in _excl_set_n]
                _exc_n = [i for i in range(len(_dobs_rel)) if i in _excl_set_n]
                if _inc_n:
                    fig_nmr_k.add_trace(go.Scatter(
                        x=_t_exp[_inc_n], y=_dobs_rel[_inc_n], mode="markers",
                        name=_tgt if _show_leg else _col,
                        legendgroup=_tgt, showlegend=_show_leg,
                        customdata=[[_col, i] for i in _inc_n],
                        marker=dict(color=_color, size=6, symbol="circle"),
                    ))
                if _exc_n:
                    fig_nmr_k.add_trace(go.Scatter(
                        x=_t_exp[_exc_n], y=_dobs_rel[_exc_n], mode="markers",
                        name=f"_outlier_{_col}",
                        legendgroup=_tgt, showlegend=False,
                        customdata=[[_col, i] for i in _exc_n],
                        marker=dict(color=_color, size=6, symbol="circle-open",
                                    line=dict(width=1.5, color=_color)),
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
            _kin_ver_nmr = st.session_state.get("_outlier_ver__outliers_nmr", 0)
            fig_nmr_k.update_traces(
                unselected=dict(marker=dict(opacity=1.0)),
                selected=dict(marker=dict(opacity=1.0, size=9)),
                selector=dict(mode="markers"),
            )
            _kin_nmr_event = st.plotly_chart(
                fig_nmr_k, width='stretch',
                on_select="rerun", selection_mode="points",
                key=f"_kin_nmr_chart_{_kin_ver_nmr}",
            )
            if _process_outlier_event(_kin_nmr_event, "_outliers_nmr"):
                st.rerun()
            _outlier_bar("kin_nmr", "_outliers_nmr")
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
            _kin_sp_excl = st.session_state.get("_outliers_spectra", set())
            for _i in range(_kin_n_sp):
                _col_c = _kin_rainbow(_i, _kin_n_sp)
                _lbl = f"t={_kin_t_raw[_i]:.3g} s"
                _is_excl = _i in _kin_sp_excl
                # Visual line — dashed+dim when excluded
                fig_kin_sp.add_trace(go.Scatter(
                    x=_kin_wl_plot, y=_kin_A_plot[_i], mode="lines",
                    line=dict(color=_col_c, width=1.5,
                              dash="dash" if _is_excl else "solid"),
                    opacity=0.75 if _is_excl else 1.0,
                    name=f"_outlier_{_i}" if _is_excl else _lbl,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Full-line click surface: markers at every wavelength point, near-invisible.
                # This makes the entire line clickable, not just the peak.
                _n_wl = len(_kin_wl_plot)
                fig_kin_sp.add_trace(go.Scatter(
                    x=_kin_wl_plot, y=_kin_A_plot[_i], mode="markers",
                    name=f"_outlier_{_i}" if _is_excl else f"_sp_sel_{_i}",
                    showlegend=False,
                    customdata=[[_i]] * _n_wl,
                    marker=dict(
                        color=_col_c,
                        size=6,
                        opacity=0.01,   # near-invisible but still hit-testable by Plotly
                        symbol="circle-open" if _is_excl else "circle",
                        line=dict(width=0),
                    ),
                    hovertemplate=f"{'[excluded — click to restore] ' if _is_excl else '[click to exclude] '}{_lbl}<br>λ=%{{x:.0f}} nm<br>A=%{{y:.4f}}<extra></extra>",
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
                title=dict(text="UV-Vis spectra (kinetics) — click anywhere on a spectrum to exclude/restore", font=dict(size=13), x=0.5),
            )
            _kin_ver_sp = st.session_state.get("_outlier_ver__outliers_spectra", 0)
            # Suppress selection highlighting on marker traces only
            fig_kin_sp.update_traces(
                unselected=dict(marker=dict(opacity=1.0)),
                selected=dict(marker=dict(opacity=1.0)),
                selector=dict(mode="markers"),
            )
            _kin_sp_event = st.plotly_chart(
                fig_kin_sp, width='stretch',
                on_select="rerun", selection_mode="points",
                key=f"_kin_sp_chart_{_kin_ver_sp}",
            )
            if _process_outlier_event(_kin_sp_event, "_outliers_spectra", is_spectra=True):
                st.rerun()
            _outlier_bar("kin_sp", "_outliers_spectra")
            _pub_download_button(fig_kin_sp, "kinetics_spectra",
                                 x_label="Wavelength [nm]", y_label="Absorbance")

            # Pure-species spectra (only after fit)
            _kin_fit_sp = st.session_state.get("_fit_stats", {})
            if _kin_fit_sp.get("fit_mode") == "kinetics_spectra":
                _kin_E      = _kin_fit_sp.get("E_final")
                _kin_wl_fit = _kin_fit_sp.get("wavelengths_fit")
                _kin_abs    = _kin_fit_sp.get("absorbers", [])
                if _kin_E is not None and len(_kin_abs) > 0:
                    _kin_path_disp = float(_kin_fit_sp.get("path_cm", 1.0))
                    _PALETTE = ["#636EFA","#EF553B","#00CC96","#AB63FA",
                                "#FFA15A","#19D3F3","#FF6692","#B6E880"]
                    fig_kin_pure = go.Figure()
                    for _j, _sp in enumerate(_kin_abs):
                        fig_kin_pure.add_trace(go.Scatter(
                            x=_kin_wl_fit, y=_kin_E[_j], mode="lines",
                            line=dict(color=_PALETTE[_j % len(_PALETTE)], width=2),
                            name=_sp,
                            hovertemplate=f"{_sp}<br>λ=%{{x:.0f}} nm<br>ε=%{{y:.4f}} mM⁻¹ cm⁻¹<extra></extra>",
                        ))
                    fig_kin_pure.update_layout(
                        height=350, margin=dict(l=40, r=20, t=40, b=60),
                        xaxis=dict(title="Wavelength [nm]"),
                        yaxis=dict(title="ε [mM⁻¹ cm⁻¹]", rangemode="tozero"),
                        template="plotly_dark", showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        title=dict(text=f"Pure species spectra  (path = {_kin_path_disp} cm)", font=dict(size=13), x=0.5),
                    )
                    st.plotly_chart(fig_kin_pure, width='stretch')
                    _pub_download_button(fig_kin_pure, "kinetics_spectra_species",
                                         x_label="Wavelength [nm]", y_label="ε [mM⁻¹ cm⁻¹]")

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
                "Tol (−log)", key="fit_tolerance_log", default=3.0, step=0.5, format="%.1f",
                help="Convergence target for the constants, in log units: the fit is "
                     "declared converged when the search has narrowed to 10^−Tol in every "
                     "constant.  Errors on constants are typically 0.01–0.1, so 3–4 is "
                     "usually enough; 6 is very tight and rarely reached before the Timeout.  "
                     "The message after each fit says how it ended."
            )
        with _fc2:
            fit_timeout = _num_input(
                "Timeout (s)", key="fit_timeout", default=30, step=5, format="%d"
            )
        fit_refresh = _num_input(
            "Refresh (s)", key="fit_refresh", default=60, step=5, format="%d",
            help="While a fit runs, redraw data and model at the best constants found so far "
                 "every this many seconds (0 = off).  NMR and global fits for now."
        )

        _g_why_k = []
        _g_ready_k = eqtabs.collect_experiments(st.session_state, _g_why_k)
        _multi_k = len(st.session_state.get("_tabs") or []) > 1
        if _multi_k:
            st.session_state.setdefault("_global_weighted", True)
            st.checkbox("Weight tabs by their own fit", key="_global_weighted",
                        help="Each tab's objective is rescaled to its own misfit (re-estimated after each "
                             "optimiser pass), so a titration the model fits worse cannot outvote a clean "
                             "one.  Untick for a plain sum of the tabs' objectives.")
            _share_shifts_checkbox(_g_ready_k)
        _fb1, _fb2 = st.columns(2)
        with _fb1:
            # never disabled: the fit boxes live in fragments, so a tick does not rerun the
            # main script and a button greyed out here would stay greyed out (a fresh tab
            # with boxes ticked could not be fitted).  The ticks are read on the press.
            if st.button("🔧 Fit Parameters", width="stretch"):
                if fit_ok:
                    for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
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
        with _fb2:
            if _multi_k and st.button("🌐 Global fit (all tabs)", key="_global_fit_btn_kin", width="stretch",
                                      help="One fit across every tab that has data and a ticked parameter.  "
                                           "Constants with the same name share one value; concentrations stay "
                                           "per tab; each tab's bounds and constraints apply."):
                if len(_g_ready_k) >= 2:
                    for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                        if _pref in st.session_state:
                            st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                    st.session_state["_global_fit_requested"] = True
                    st.rerun()
                else:
                    st.info("The Global fit needs at least two tabs with data loaded and a 'fit' "
                            "box ticked.  " + "  ".join(_g_why_k))
        if _multi_k:
            st.caption("Tabs in the global fit: " + (", ".join(_lab for _i, _lab, _fa in _g_ready_k) or "none yet")
                       + ((".  Left out — " + "; ".join(_g_why_k)) if _g_why_k else ""))
        _hpc_go_k = hpc_ui.controls("hpc_kin", label="the fit",
                                    submits=[("fit", "this tab's fit")] + ([("global", "the global fit")] if _multi_k else []))
        if _hpc_go_k == "global":
            if len(_g_ready_k) >= 2:
                for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                    if _pref in st.session_state:
                        st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                st.session_state["_global_fit_hpc"] = "hpc_kin"
                st.session_state["_global_fit_requested"] = True
                st.rerun()
            else:
                st.info("The Global fit needs at least two tabs with data loaded and a 'fit' "
                        "box ticked.  " + "  ".join(_g_why_k))
        elif _hpc_go_k == "fit":
            if fit_ok:
                for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                    if _pref in st.session_state:
                        st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                st.session_state["_single_fit_hpc"] = "hpc_kin"
                st.session_state["_fit_requested"] = True
                st.rerun()
            else:
                st.info("Load experimental data and check parameters to fit.")
        hpc_ui.jobs_panel("hpc_kin", on_result=_hpc_store_fit)
        if _multi_k:
            if st.button("📤 Propagate ticked constants", key="_propagate_btn_kin",
                         help="Copy the constants ticked here, at the values in their boxes, into every "
                              "other tab whose script has a constant of the same name.  Only the values "
                              "move; the other tabs' ticks, bounds, data and scripts stay as they are."):
                st.info(eqtabs.propagate_ticked(st.session_state, parse_script))

        # ── Export ───────────────────────────────────────────────
        st.subheader("Data Export")
        col_exp, col_snap = st.columns(2)
        with col_exp:
            try:
                buf = _export_kinetics_excel(kin_curve, t_vals, plot_y_names,
                                              parsed_kin, logk_dict, script_text,
                                              variables,
                                              script_path=st.session_state.get("_script_filename"),
                                              input_path=st.session_state.get("_input_filename"),
                                              fit_stats=st.session_state.get("_fit_stats", {}))
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
            _raw_bytes_kin = exp_uploaded.read()
            # Capture raw bytes + filename in session_state so the
            # "Save session" feature can bundle the data verbatim.
            st.session_state["_exp_data_bytes"]    = _raw_bytes_kin
            st.session_state["_exp_data_filename"] = exp_uploaded.name
            if _kin_has_spectra:
                _loaded_kin = load_spectra_data(_raw_bytes_kin)
                st.session_state["_spectra_data"] = _loaded_kin
                st.session_state.pop("_outliers_spectra", None)
                if _loaded_kin and len(_loaded_kin.get("wavelengths", [])) > 0:
                    st.session_state["_pending_spectra_wl_min"] = float(_loaded_kin["wavelengths"][0])
                    st.session_state["_pending_spectra_wl_max"] = float(_loaded_kin["wavelengths"][-1])
                # Update xmax from last data point
                if _loaded_kin and len(_loaded_kin.get("x_vals", [])) > 0:
                    _x_last_kin = float(_loaded_kin["x_vals"][-1])
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_kin * 10) / 10)
            else:
                _loaded_kin = load_experimental_data(_raw_bytes_kin)
                if _kin_nmr_cfg_up is not None:
                    st.session_state["_nmr_data"] = _loaded_kin
                    st.session_state.pop("_outliers_nmr", None)
                else:
                    st.session_state["_exp_data"] = _loaded_kin
                    st.session_state.pop("_outliers_main", None)
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
        if fit_stats and fit_stats.get("fit_mode", "kinetics") != "itc":
            _kin_fit_mode = fit_stats.get("fit_mode", "kinetics")
            if fit_stats.get("timed_out"):
                _to_s = float(st.session_state.get("fit_timeout", 30))
                _n_ev = fit_stats.get("n_iter", 0)
                st.warning(f"⏱️ Fit timed out after {_n_ev} evaluations ({_to_s:.0f} s limit) — "
                           "showing best result found. Consider increasing Timeout.")

            # ── Basic fit statistics (collapsible) ───────────────────────
            # All the simple goodness-of-fit numbers (R², RMSE, χ²,
            # AIC/BIC, DW/Shapiro) tucked behind a checkbox so the
            # default post-fit view shows just the fitted-constants
            # table.  Same hierarchy as "Show residual diagnostics".
            @_fragment
            def _stats_kin_frag():
                _diag_scope_switch("_fit_args_kin")
                if st.checkbox("🔢 Show fit statistics",
                               value=False, key="_show_stats_kin",
                               help="R², RMSE, reduced χ², information "
                                    "criteria (AIC/BIC), and residual "
                                    "tests (Durbin–Watson, Shapiro–Wilk)."):
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

                    # ── v2 diagnostics: AIC / BIC / Durbin–Watson / Shapiro ──
                    if "aic" in fit_stats and np.isfinite(fit_stats.get("aic", float("nan"))):
                        st.write(f"• AIC = {fit_stats['aic']:.2f}   |   "
                                 f"AICc = {fit_stats['aicc']:.2f}   |   "
                                 f"BIC = {fit_stats['bic']:.2f}")
                    if "durbin_watson" in fit_stats and np.isfinite(fit_stats.get("durbin_watson", float("nan"))):
                        _dw = fit_stats["durbin_watson"]; _sp = fit_stats.get("shapiro_p")
                        _dw_note = "✓" if 1.5 <= _dw <= 2.5 else "⚠ autocorrelation suspected"
                        _sp_note = ("normal" if (_sp is not None and np.isfinite(_sp) and _sp >= 0.05)
                                    else "non-normal" if (_sp is not None and np.isfinite(_sp)) else "")
                        _sp_disp = f"; Shapiro p = {_sp:.3f} ({_sp_note})" if (_sp is not None and np.isfinite(_sp)) else ""
                        st.write(f"• Durbin–Watson = {_dw:.2f} ({_dw_note}){_sp_disp}")
                        # Spectra-mode caveat (kinetics-spectra fits — same
                        # rationale as the equilibrium branch: residuals
                        # are wavelength-correlated, so DW / SW lose
                        # their diagnostic meaning).
                        if "spectra" in str(fit_stats.get("fit_mode", "")):
                            st.caption(
                                "ℹ Spectra mode: residuals are wavelength-correlated by "
                                "spectrometer bandpass, baseline drift, and spectral "
                                "smoothness, so Durbin–Watson and Shapiro–Wilk are not "
                                "diagnostically meaningful (DW will read low and Shapiro "
                                "p will be tiny regardless of fit quality). The "
                                "informative view is the residuals-vs-predictor panel."
                            )
            _stats_kin_frag()

            # ── v2 diagnostics: residual panel (expander) ────────────────
            _res_kin = st.session_state.get("_fit_residuals")
            if _res_kin and len(_res_kin.get("y_obs", [])) > 0:
                @_fragment
                def _diag_kin_frag():
                    # Same rationale as eq branch: fit-counter-bound
                    # key forces a fresh widget on each fit.
                    _n_kin = int(st.session_state.get("_fit_counter", 0))
                    if st.checkbox("📊 Show residual diagnostics",
                                   value=False,
                                   key=f"_show_diag_kin_{_n_kin}"):
                        # ── Parameter correlation heatmap (linear/Hessian) ──
                        # Cheap (Hessian already computed at fit time), so
                        # placed FIRST in the diagnostic toggle stack —
                        # it's the linear/local view that should be checked
                        # before paying for the nonlinear 1D and 2D profile
                        # scans below.  When the heatmap shows weak
                        # correlations but the 2D RMSE profile shows a
                        # banana valley, the Hessian SE is misleading and
                        # the bootstrap CIs are the trustworthy estimate.
                        _show_corr_k = _indented_checkbox(
                            "🔗 Show parameter correlation heatmap",
                            value=False, key="_show_corr_kin",
                            help="Pearson ρ derived from the Hessian "
                                 "covariance.  Strong off-diagonal "
                                 "entries (|ρ| > 0.9) mean the Hessian "
                                 "standard errors understate the true "
                                 "uncertainty.  Compare against the 2D "
                                 "RMSE profile for the nonlinear view.")
                        if _show_corr_k:
                            try:
                                from equilibrist_diagnostics import \
                                    make_correlation_heatmap
                                _figh = make_correlation_heatmap(
                                    fit_stats,
                                    title="Parameter correlation matrix")
                                st.pyplot(_figh, clear_figure=False)
                                import io as _io
                                _bufh = _io.BytesIO()
                                _figh.savefig(_bufh, format="png", dpi=200,
                                              bbox_inches="tight")
                                _bufh.seek(0)
                                _tsh = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    "Download correlation heatmap PNG",
                                    data=_bufh.getvalue(),
                                    file_name=f"Equilibrist_corr_"
                                              f"{_tsh}.png",
                                    mime="image/png",
                                    key="_corr_dl_kin")
                            except Exception as _ce:
                                st.error(f"Correlation heatmap failed: {_ce}")

                        # ── Parameter identifiability — sloppy spectrum ──
                        # Eigenanalysis of the same Hessian covariance,
                        # giving the GLOBAL view: which combinations of
                        # parameters are well/poorly determined.  The
                        # correlation heatmap above is the pairwise
                        # picture, and the 2D RMSE profile below is the
                        # nonlinear pairwise picture; this panel is the
                        # multi-parameter generalization — eigenvalues
                        # sorted ascending, with each eigendirection
                        # labelled by its dominant linear combination
                        # and tagged stiff / intermediate / sloppy.
                        # Condition number κ summarises the degeneracy
                        # of the fit.  Especially useful for models
                        # with many coupled parameters (≥ 4 K's or
                        # rate constants).
                        _show_ident_k = _indented_checkbox(
                            "🔬 Show parameter identifiability "
                            "(sloppy spectrum)",
                            value=False, key="_show_ident_kin",
                            help="Eigendecomposition of the Hessian "
                                 "covariance.  Spread of σᵢ across many "
                                 "decades = sloppy model — some parameter "
                                 "combinations are unconstrained by the "
                                 "data.  Brown & Sethna 2003; Gutenkunst "
                                 "et al. 2007; Hibbert & Thordarson 2016.")
                        if _show_ident_k:
                            try:
                                render_identifiability_panel(
                                    fit_stats, key="ident_kin")
                            except Exception as _ie:
                                st.error(f"Identifiability analysis failed: {_ie}")

                        # ── Parameter significance — t-test ──────────────
                        # Per-parameter (θ̂ − θ₀) / SE, p-value from
                        # Student's t with df = n − p.  The user can
                        # override θ₀ per row via the editable column
                        # (default 0, which for log K means K = 1, and
                        # for log conc means c = 1 mM — handy sanity
                        # check).  Cheap to compute (just reads
                        # param_errors that were already populated for
                        # the diagnostics panel) and complements the
                        # heatmap by reporting the *significance* of
                        # each estimate rather than its correlation
                        # with others.
                        _show_ttest_k = _indented_checkbox(
                            "📐 Show parameter significance (t-test)",
                            value=False, key="_show_ttest_kin",
                            help="t = (θ̂ − θ₀) / SE, two-tailed p-value "
                                 "from Student's t with df = n − p.  "
                                 "Estimates and SE are in log10 units; "
                                 "edit the Null column to test against "
                                 "literature values (e.g. log K = 4.5).")
                        if _show_ttest_k:
                            try:
                                from equilibrist_diagnostics import \
                                    compute_param_t_tests
                                _nul_key_k = "_ttest_nulls_kin"
                                _nul_overrides_k = st.session_state.get(
                                    _nul_key_k, {})
                                _rows_k = compute_param_t_tests(
                                    fit_stats, _nul_overrides_k)
                                if not _rows_k:
                                    st.info("No fitted parameters with "
                                            "standard errors available.")
                                else:
                                    import pandas as _pd
                                    # Stash name order so the on_change
                                    # callback can map row indices →
                                    # parameter names without us having
                                    # to retrieve the editor's internal
                                    # state in the wrong order.
                                    _names_k = [r["name"] for r in _rows_k]
                                    st.session_state["_ttest_names_kin"] = _names_k

                                    def _on_ttest_kin_change():
                                        # Runs BEFORE the next rerun's
                                        # script body, so by the time
                                        # the fragment re-executes the
                                        # nulls are already updated and
                                        # t/p recompute correctly — no
                                        # explicit st.rerun() needed
                                        # (which would scroll the page
                                        # back to the top).
                                        _state = st.session_state.get(
                                            "_ttest_editor_kin", {}) or {}
                                        _ed_rows = _state.get(
                                            "edited_rows", {}) or {}
                                        if not _ed_rows:
                                            return
                                        _names = st.session_state.get(
                                            "_ttest_names_kin", [])
                                        _nulls = dict(st.session_state.get(
                                            "_ttest_nulls_kin", {}))
                                        for _ri, _ch in _ed_rows.items():
                                            try:
                                                _i = int(_ri)
                                                if (_i < len(_names)
                                                        and "Null θ₀" in _ch):
                                                    _nulls[_names[_i]] = float(_ch["Null θ₀"])
                                            except (ValueError, TypeError, KeyError):
                                                pass
                                        st.session_state["_ttest_nulls_kin"] = _nulls

                                    _df_k = _pd.DataFrame(_rows_k)
                                    _df_disp_k = _pd.DataFrame({
                                        "Parameter":      _df_k["name"],
                                        "Kind":           _df_k["kind"],
                                        "Estimate (log)": _df_k["value"].round(4),
                                        "SE (log)":       _df_k["se"].round(4),
                                        "Null θ₀":        _df_k["null"].round(4),
                                        "t":              _df_k["t"].round(2),
                                        "p":              _df_k["p"].map(
                                            lambda v: f"{v:.2e}" if _pd.notna(v) else "n/a"),
                                        "Sig":            _df_k["stars"],
                                    })
                                    st.data_editor(
                                        _df_disp_k,
                                        column_config={
                                            "Parameter":      st.column_config.TextColumn(disabled=True),
                                            "Kind":           st.column_config.TextColumn(disabled=True),
                                            "Estimate (log)": st.column_config.NumberColumn(disabled=True, format="%.4f"),
                                            "SE (log)":       st.column_config.NumberColumn(disabled=True, format="%.4f"),
                                            "Null θ₀":        st.column_config.NumberColumn(
                                                                  required=True,
                                                                  format="%.4f",
                                                                  help="θ₀ used for the t-test.  "
                                                                       "Edit to compare against a "
                                                                       "literature value (in log "
                                                                       "units).  Default 0."),
                                            "t":              st.column_config.NumberColumn(disabled=True, format="%.2f"),
                                            "p":              st.column_config.TextColumn(disabled=True),
                                            "Sig":            st.column_config.TextColumn(disabled=True,
                                                                  help="*** p<0.001, ** p<0.01, "
                                                                       "* p<0.05, n.s. otherwise"),
                                        },
                                        hide_index=True,
                                        use_container_width=True,
                                        on_change=_on_ttest_kin_change,
                                        key="_ttest_editor_kin")
                                    if _rows_k and _rows_k[0]["df"] > 0:
                                        st.caption(
                                            f"Degrees of freedom: "
                                            f"df = {_rows_k[0]['df']} "
                                            f"(n − p). Estimates and SE "
                                            f"shown in log10 units.")
                            except Exception as _te:
                                st.error(f"t-test failed: {_te}")

                        # ── Rank analysis (EFA / scree / TFA) ────────
                        # Data-space diagnostic — see equilibrium-side
                        # comment for rationale.  Only visible when the
                        # fit produced a multi-channel data matrix
                        # (kinetics-with-spectra; multi-signal kinetics
                        # NMR shift/integ).
                        if fit_stats.get("data_matrix") is not None:
                            _show_rank_k = _indented_checkbox(
                                "📊 Show data-rank analysis "
                                "(EFA / scree / TFA)",
                                value=False, key="_show_rank_kin",
                                help="Evolving Factor Analysis + "
                                     "Malinowski IND + Target Factor "
                                     "Analysis on the data matrix "
                                     "D = C·E^T.  EFA reveals when "
                                     "each species' contribution rises "
                                     "above the noise floor; IND "
                                     "recommends a rank; TFA projects "
                                     "each fitted pure spectrum onto "
                                     "the data's factor space to check "
                                     "that the fitted ε is actually "
                                     "supported by the data.  Maeder "
                                     "& Zuberbühler 1986; "
                                     "Malinowski 1991.")
                            if _show_rank_k:
                                try:
                                    render_rank_analysis_panel(
                                        fit_stats, key="rank_kin")
                                except Exception as _re:
                                    st.error(f"Rank analysis failed: {_re}")

                        # Optional RMSE parameter profile (Musketeer §4.4
                        # style).  Expensive — refits at every grid point
                        # for every fitted parameter — so it's behind a
                        # checkbox and cached in session_state.
                        _show_prof = _indented_checkbox(
                            "🔍 Show parameter profile (RMSE vs each "
                            "parameter)",
                            value=False, key="_show_prof_kin",
                            help="For each fitted parameter, pins it at a "
                                 "grid of values around the optimum and "
                                 "refits the others.  Sharp valleys = "
                                 "well-determined parameter; flat regions "
                                 "= unidentifiable.")
                        _prof = None
                        _prof_fixed = {}
                        if _show_prof:
                            _fa = _diag_fa(st.session_state.get("_fit_args_kin"))
                            if not _fa:
                                st.info("Click **Fit** first to enable "
                                        "the parameter profile.")
                            else:
                                col_a, col_b = st.columns(2)
                                _p_span = col_a.number_input(
                                    "Span (log units, ±)", min_value=0.1,
                                    max_value=3.0, value=0.5, step=0.1,
                                    key="_prof_span_kin")
                                _p_npts = col_b.number_input(
                                    "Grid points (odd)", min_value=5,
                                    max_value=201, value=11, step=2,
                                    key="_prof_npts_kin")
                                (_pk_K, _pk_c, _pk_t, _prof_fixed, _pk_tag,
                                 _pk_default) = _profile_param_picker("kin", fit_stats, _fa)
                                _prof_par_kin = st.checkbox(
                                    "Parallel (use all cores)",
                                    value=False, key="_prof_par_kin",
                                    help="Run independent grid-point "
                                         "refits in parallel via joblib. "
                                         "Speeds up large profiles "
                                         "roughly linearly with core count.")
                                _hpc_go_prof_kin = hpc_ui.controls("prof_kin", label="the 1D profile")
                                _prof_jobs_kin = -1 if _prof_par_kin else 1
                                # Cache key intentionally OMITS the
                                # fitted ``param_values`` tuple: every
                                # new fit clears all ``_prof_cache_*``
                                # entries via ``_reset_diagnostic_toggles``,
                                # so staleness is impossible while the
                                # cache is alive.  Including the
                                # parameter-value tuple in the key
                                # made the cache fragile across a JSON
                                # save / reload because any bit-level
                                # difference between save-time and
                                # post-reload-refit parameter values
                                # caused a cache miss and the saved
                                # profile failed to re-render.
                                _cache_key = ("_prof_cache_kin",
                                              float(_p_span), int(_p_npts))
                                if not _pk_default:      # a custom selection gets its own cache entry
                                    _cache_key = _cache_key + (_pk_tag,)
                                _cached = st.session_state.get(_cache_key)
                                if _cached is not None:
                                    _prof = _cached
                                else:
                                    if st.button("Compute profile",
                                                 key="_prof_btn_kin",
                                                 disabled=not (_pk_K or _pk_c or _pk_t)) or _hpc_go_prof_kin:
                                        _bar = st.progress(0.0,
                                            text="Computing profile…")
                                        def _cb(f, m):
                                            try:
                                                _bar.progress(min(1.0, f),
                                                              text=m)
                                            except Exception:
                                                pass
                                        try:
                                            _fa_kin_prof = dict(_fa)
                                            if _fa_kin_prof.get("kind") != "global":
                                                _fa_kin_prof["kind"] = "kin"
                                            # Seed every refit from the FITTED values and use a small
                                            # initial simplex: the refits start next to the optimum.
                                            _seed1d = dict(_fa_kin_prof.get("start_logK") or _fa_kin_prof.get("start_logk") or {})
                                            for _n, _v in (fit_stats.get("param_values", {}) or {}).items():
                                                try: _seed1d[_n] = float(_v)
                                                except Exception: pass
                                            _fa_kin_prof["start_logK"] = _seed1d
                                            _fa_kin_prof["start_logk"] = dict(_seed1d)
                                            _fa_kin_prof["simplex_step"] = 0.15
                                            _prof = compute_rmse_profile(
                                                _fa_kin_prof,
                                                _pk_K,
                                                span=float(_p_span),
                                                n_pts=int(_p_npts),
                                                fitted_concs_mM=_pk_c,
                                                fitted_titrants_mM=_pk_t,
                                                n_jobs=_prof_jobs_kin,
                                                progress_callback=_cb)
                                            _write_tuple_cache(_cache_key, _prof)
                                            _bar.empty()
                                        except hpc_ui.CapturedCall as _hpc_cc:
                                            _bar.empty()
                                            hpc_ui.handle_capture(_hpc_cc, "prof_kin", "1D RMSE profile",
                                                                  context={"ck": list(_cache_key)})
                                        except Exception as _pe:
                                            _bar.empty()
                                            st.error(f"Profile failed: {_pe}")
                                hpc_ui.jobs_panel("prof_kin", on_result=_hpc_store_tuple_cache)
                        render_diagnostics_panel(fit_stats, _res_kin,
                                                 key="diag_kin",
                                                 rmse_profile=_profile_display(_prof, _prof_fixed))

                        # ── 2D RMSE profile (pairwise correlation diagnostic) ──
                        # Lives as a peer to the 1D profile.  Different
                        # output channel (its own colormap panel below the
                        # 2×2 figure) and a much higher compute cost
                        # (n_pts² refits per pair, vs n_params×n_pts for
                        # 1D) — so it gets its own toggle and its own
                        # cache.  Enabled when at least 2 parameters
                        # were fitted; auto-selects the pair when
                        # exactly 2, otherwise asks via dropdowns.
                        _show_p2d_k = _indented_checkbox(
                            "🌐 Show 2D parameter profile "
                            "(RMSE colormap for a pair)",
                            value=False, key="_show_p2d_kin",
                            help="At each (X, Y) grid point both "
                                 "parameters are pinned and every "
                                 "OTHER free parameter is refit. "
                                 "Long diagonal valleys = strong "
                                 "pairwise correlation; round wells "
                                 "= independently identified.")
                        if _show_p2d_k:
                            _fa_k_for2d = _diag_fa(st.session_state.get("_fit_args_kin"))
                            if not _fa_k_for2d:
                                st.info("Click **Fit** first to enable "
                                        "the 2D profile.")
                            else:
                                # Build the candidate parameter list
                                # in log10 units (K's natively; concs
                                # and titrants converted from mM)
                                _cands = {}
                                for _n, _v in (fit_stats.get(
                                        "param_values", {}) or {}).items():
                                    try:
                                        _cands[_n] = float(_v)
                                    except Exception: pass
                                for _n, _mM in (fit_stats.get(
                                        "fitted_concs", {}) or {}).items():
                                    try:
                                        _v = float(_mM)
                                        if _v > 0:
                                            _cands[_n] = float(np.log10(_v))
                                    except Exception: pass
                                # Constants that were held FIXED in the last fit are offered too:
                                # the refit engine pins both scanned parameters anyway, so a
                                # fixed constant can be scanned against a fitted one while every
                                # other fitted parameter is re-optimised at each grid point.
                                _fixed_cands = {}
                                _slk2d = (_fa_k_for2d.get("start_logK")
                                          or _fa_k_for2d.get("start_logk") or {})
                                for _n, _v in _slk2d.items():
                                    if _n in _cands or _n == "Kw": continue
                                    try: _fixed_cands[_n] = float(_v)
                                    except Exception: pass
                                _cands.update(_fixed_cands)
                                _lbl2d = lambda _n: (f"{_n} (fixed)" if _n in _fixed_cands else _n)
                                _names = list(_cands.keys())
                                if len(_names) < 2:
                                    st.info("Need at least 2 "
                                            "parameters to build a 2D "
                                            "profile.")
                                else:
                                    if len(_names) == 2:
                                        _p1_k, _p2_k = _names[0], _names[1]
                                        st.caption(f"Auto-selected: "
                                                   f"**{_p1_k}** vs **{_p2_k}** "
                                                   "(only two available).")
                                    else:
                                        c1, c2 = st.columns(2)
                                        _p1_k = c1.selectbox(
                                            "X-axis parameter", _names,
                                            index=0, key="_p2d_p1_kin",
                                            format_func=_lbl2d)
                                        _p2_k = c2.selectbox(
                                            "Y-axis parameter", _names,
                                            index=1, key="_p2d_p2_kin",
                                            format_func=_lbl2d)
                                    _fx_sel = [_n for _n in (_p1_k, _p2_k) if _n in _fixed_cands]
                                    if _fx_sel:
                                        st.caption("Held fixed in the last fit: " + ", ".join(
                                            f"**{_n}** = {_fixed_cands[_n]:.2f}" for _n in _fx_sel) +
                                            ". The map is centred on that value, which is not an "
                                            "optimum — look for where the RMSE minimum actually "
                                            "falls. All other fitted parameters are refit at each "
                                            "grid point.")
                                    c_sx, c_sy, c_np = st.columns(3)
                                    _sx_k = c_sx.number_input(
                                        "Span X (log ±)", min_value=0.1,
                                        max_value=3.0, value=0.5, step=0.1,
                                        key="_p2d_sx_kin")
                                    _sy_k = c_sy.number_input(
                                        "Span Y (log ±)", min_value=0.1,
                                        max_value=3.0, value=0.5, step=0.1,
                                        key="_p2d_sy_kin")
                                    _np_k = c_np.number_input(
                                        "Grid (per axis, odd)",
                                        min_value=5, max_value=201,
                                        value=11, step=2,
                                        key="_p2d_np_kin")
                                    _par2_k = st.checkbox(
                                        "Parallel (use all cores)",
                                        value=True, key="_p2d_par_kin",
                                        help="Strongly recommended — "
                                             "2D scans run n_pts² refits.")
                                    _hpc_go_p2d_kin = hpc_ui.controls("p2d_kin", label="the 2D profile")
                                    _p2d_jobs = -1 if _par2_k else 1
                                    if _p1_k == _p2_k:
                                        st.warning("Pick two **different** "
                                                   "parameters.")
                                    else:
                                        # Cache key omits the per-fit
                                        # parameter-value tuple — see
                                        # the 1-D profile cache for the
                                        # rationale (invariant across
                                        # JSON save/reload now that the
                                        # reset function handles
                                        # invalidation on new fits).
                                        _ck = ("_p2d_cache_kin", _p1_k, _p2_k,
                                               float(_sx_k), float(_sy_k),
                                               int(_np_k))
                                        _cached2 = st.session_state.get(_ck)
                                        if _cached2 is not None:
                                            _scan2 = _cached2
                                        else:
                                            _scan2 = None
                                            if st.button(
                                                    "Compute 2D profile",
                                                    key="_p2d_btn_kin") or _hpc_go_p2d_kin:
                                                _bar2 = st.progress(0.0,
                                                    text="Computing 2D profile…")
                                                def _cb2(f, m):
                                                    try: _bar2.progress(min(1.0, f), text=m)
                                                    except Exception: pass
                                                try:
                                                    _fa_k_prof2 = dict(_fa_k_for2d)
                                                    if _fa_k_prof2.get("kind") != "global":
                                                        _fa_k_prof2["kind"] = "kin"
                                                    _seed2k = dict(_fa_k_prof2.get("start_logk") or _fa_k_prof2.get("start_logK") or {})
                                                    for _n, _v in (fit_stats.get("param_values", {}) or {}).items():
                                                        try: _seed2k[_n] = float(_v)
                                                        except Exception: pass
                                                    _fa_k_prof2["start_logk"] = _seed2k
                                                    _fa_k_prof2["start_logK"] = dict(_seed2k)
                                                    _fa_k_prof2["simplex_step"] = 0.15
                                                    _scan2 = compute_rmse_profile_2d(
                                                        _fa_k_prof2,
                                                        _p1_k, _cands[_p1_k],
                                                        _p2_k, _cands[_p2_k],
                                                        span_x=float(_sx_k),
                                                        span_y=float(_sy_k),
                                                        n_pts_x=int(_np_k),
                                                        n_pts_y=int(_np_k),
                                                        n_jobs=_p2d_jobs,
                                                        progress_callback=_cb2)
                                                    _scan2["p1_fixed"] = bool(_p1_k in _fixed_cands)
                                                    _scan2["p2_fixed"] = bool(_p2_k in _fixed_cands)
                                                    _write_tuple_cache(_ck, _scan2)
                                                    _bar2.empty()
                                                except hpc_ui.CapturedCall as _hpc_cc:
                                                    _bar2.empty()
                                                    hpc_ui.handle_capture(_hpc_cc, "p2d_kin", "2D RMSE profile",
                                                                          context={"ck": list(_ck), "p1_fixed": bool(_p1_k in _fixed_cands), "p2_fixed": bool(_p2_k in _fixed_cands)})
                                                except Exception as _e2:
                                                    _bar2.empty()
                                                    st.error(f"2D profile failed: {_e2}")
                                        hpc_ui.jobs_panel("p2d_kin", on_result=_hpc_store_tuple_cache)
                                        if _scan2 is not None:
                                            _p2d_noleg = st.checkbox(
                                                "No legend", value=False, key="_p2d_noleg_kin",
                                                help="Draw the map without the legend box and the "
                                                     "note in the corner — for publication, where "
                                                     "that text goes in the figure caption.")
                                            try:
                                                _fig2 = make_2d_profile_figure(
                                                    _scan2,
                                                    title=f"2D RMSE profile: "
                                                          f"{_p1_k} vs {_p2_k}",
                                                    param_cov=fit_stats.get("param_cov"),
                                                    cov_names=fit_stats.get("param_cov_names"),
                                                    constraints=_fa_k_for2d.get("constraints"),
                                                    show_legend=not _p2d_noleg)
                                                st.pyplot(_fig2, clear_figure=False)
                                                import io as _io
                                                _buf2 = _io.BytesIO()
                                                _fig2.savefig(_buf2, format="png",
                                                              dpi=200,
                                                              bbox_inches="tight")
                                                _buf2.seek(0)
                                                _ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                st.download_button(
                                                    "Download 2D profile PNG",
                                                    data=_buf2.getvalue(),
                                                    file_name=f"Equilibrist_2Dprofile_"
                                                              f"{_p1_k}_{_p2_k}_{_ts2}.png",
                                                    mime="image/png",
                                                    key="_p2d_dl_kin")
                                            except Exception as _re:
                                                st.error(f"Could not render 2D "
                                                         f"figure: {_re}")
                        _profile_trace_block("kin", fit_stats)
                _diag_kin_frag()

            # ── v2 bootstrap CIs (toggle, in fragment) ───────────────────
            @_fragment
            def _bs_kin_frag():
                if st.checkbox("🔁 Show bootstrap confidence intervals",
                               value=False, key="_show_bs_kin"):
                    st.caption("Resamples fit residuals to estimate CIs without "
                               "assuming Gaussian errors. Recommended when "
                               "Shapiro p < 0.05 above. Each iteration is a full refit.")
                    _fa = _diag_fa(st.session_state.get("_fit_args_kin"))
                    if not _fa:
                        st.info("Click **Fit** once with the current script and data "
                                "to enable bootstrap.  (The bootstrap reuses the exact "
                                "configuration of the most recent fit.)")
                    else:
                        _bs_n   = st.number_input("Iterations", min_value=20, max_value=10000,
                                                  value=200, step=50, key="_bs_n_kin")
                        _bs_mth = st.selectbox("Resampling method",
                                               ["residual", "parametric", "wild"],
                                               index=0, key="_bs_mth_kin")
                        _bs_par = st.checkbox("Parallel (use all cores)",
                                              value=False, key="_bs_par_kin")
                        _hpc_go_bs_kin = hpc_ui.controls("bs_kin", label="the bootstrap")
                        if st.button("Run bootstrap", key="_bs_run_kin") or _hpc_go_bs_kin:
                            _pb = st.progress(0.0, text="Bootstrap running…")
                            _cb = lambda i, N: _pb.progress(min(i / max(N,1), 1.0),
                                                            text=f"Bootstrap {i}/{N}")
                            _jobs = -1 if _bs_par else 1
                            try:
                                if _fa.get("kind") == "global":
                                    _r = eqglobal.bootstrap_global(
                                        _fa, n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42, progress_callback=_cb)
                                elif _fa["use_spectra_fit"]:
                                    _r = ebs.bootstrap_kinetics_spectra(
                                        _fa["parsed_fit"], _fa["start_logk"],
                                        _fa["spectra_data_fit"], _fa["fit_keys"],
                                        t_max=_fa["t_max"], n_pts=200,
                                        wl_min=_fa.get("wl_min", 200.0),
                                        wl_max=_fa.get("wl_max", 800.0),
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=_fa["timeout_s"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "shift":
                                    _r = ebs.bootstrap_kinetics_nmr_shifts(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logk"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        t_max=_fa["t_max"], n_pts=200,
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=_fa["timeout_s"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "integration":
                                    _r = ebs.bootstrap_kinetics_nmr_integration(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logk"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        t_max=_fa["t_max"], n_pts=200,
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=_fa["timeout_s"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "mixed":
                                    _r = ebs.bootstrap_kinetics_nmr_mixed(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logk"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        t_max=_fa["t_max"], n_pts=200,
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=_fa["timeout_s"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        progress_callback=_cb)
                                else:
                                    _r = ebs.bootstrap_kinetics(
                                        _fa["parsed_fit"], _fa["exp_data_fit"] or {},
                                        _fa["start_logk"], _fa["fit_keys"],
                                        t_max=_fa["t_max"], n_pts=200,
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=_fa["timeout_s"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        progress_callback=_cb)
                                _pb.empty()
                                st.session_state["_bs_result_kin"] = _r
                            except hpc_ui.CapturedCall as _hpc_cc:
                                _pb.empty()
                                hpc_ui.handle_capture(_hpc_cc, "bs_kin", "Bootstrap",
                                                      context={})
                            except Exception as _bse:
                                _pb.empty()
                                import traceback as _tb_bs
                                st.error(f"Bootstrap failed: {_bse}")
                                st.caption(_tb_bs.format_exc())
                        hpc_ui.jobs_panel("bs_kin", on_result=_hpc_store_state("_bs_result_kin"))
                        _br = st.session_state.get("_bs_result_kin")
                        if _br and _br.get("ci"):
                            st.caption(f"{_br['n_success']}/{_br['n_bootstrap']} bootstrap "
                                       f"fits converged ({_br['wall_seconds']:.1f}s; "
                                       f"method = {_br['method']})")
                            # ── Multi-level CIs always shown ──
                            # The bootstrap stored the raw samples
                            # ({param: ndarray}) so all four standard
                            # percentile intervals (50/80/95/99 %, the
                            # SupraFit / SIVVU convention) are computed
                            # on the fly without re-resampling.  Columns
                            # are ordered widest → narrowest so the eye
                            # moves from the conservative interval to
                            # the tight one.
                            _levels_kin = [99, 95, 80, 50]
                            _samples_kin = _br.get("samples") or {}
                            _rows = []
                            for _n in _br["ci"].keys():
                                _pt = _br["best_fit"].get(_n, float("nan"))
                                _s = np.asarray(_samples_kin.get(_n, []),
                                                dtype=float)
                                _s = _s[np.isfinite(_s)]
                                _row = {"parameter": _n,
                                        "point":  f"{_pt:+.4f}",
                                        "median": (f"{np.median(_s):+.4f}"
                                                    if _s.size else "—"),
                                        "σ":      (f"{np.std(_s, ddof=1):.4f}"
                                                    if _s.size > 1 else "—")}
                                for _lvl in _levels_kin:
                                    if _s.size:
                                        _alpha = 1.0 - _lvl / 100.0
                                        _lo, _hi = np.quantile(
                                            _s, [_alpha/2, 1 - _alpha/2])
                                        _row[f"{_lvl} % CI"] = \
                                            f"[{_lo:+.4f}, {_hi:+.4f}]"
                                    else:
                                        _row[f"{_lvl} % CI"] = "—"
                                _rows.append(_row)
                            st.dataframe(_rows, hide_index=True,
                                          use_container_width=True)
            _bs_kin_frag()

            # ── v2 jackknife (leave-one-out) — kinetics branch ────────
            if fit_stats:
                @_fragment
                def _jk_kin_frag():
                    if st.checkbox("🪛 Show jackknife (leave-one-out) analysis",
                                   value=False, key="_show_jk_kin",
                                   help="For each refit the jackknife "
                                        "removes one full titration "
                                        "step — that is, all signal "
                                        "columns (or all wavelengths "
                                        "for UV\u2013Vis spectra; all "
                                        "observed species at one time "
                                        "for kinetics) at one x-value, "
                                        "simultaneously."):
                        _fa_jk_k = _diag_fa(st.session_state.get("_fit_args_kin"))
                        if not _fa_jk_k:
                            st.info("Click **Fit** once with the current "
                                    "script and data to enable jackknife.")
                        else:
                            _jk_par_k = st.checkbox(
                                "Parallel (use all cores)",
                                value=False, key="_jk_par_kin")
                            _hpc_go_jk_kin = hpc_ui.controls("jk_kin", label="the jackknife")
                            if st.button("Run jackknife",
                                          key="_jk_run_kin") or _hpc_go_jk_kin:
                                _pb_jk_k = st.progress(
                                    0.0, text="Jackknife running…")
                                _cb_jk_k = lambda i, N: _pb_jk_k.progress(
                                    min(i / max(N, 1), 1.0),
                                    text=f"Jackknife {i}/{N}")
                                _jobs_jk_k = -1 if _jk_par_k else 1
                                _to_s_k = float(st.session_state.get(
                                    "fit_timeout", 30))
                                try:
                                    _jr_k = None
                                    if _fa_jk_k.get("kind") == "global":
                                        _jr_k = eqglobal.jackknife_global(
                                            _fa_jk_k, n_jobs=_jobs_jk_k, progress_callback=_cb_jk_k)
                                    elif _fa_jk_k.get("use_spectra_fit"):
                                        # kinetics + spectra
                                        _jr_k = ebs.jackknife_kinetics_spectra(
                                            _fa_jk_k["parsed_fit"],
                                            _fa_jk_k["start_logk"],
                                            _fa_jk_k["spectra_data_fit"],
                                            _fa_jk_k["fit_keys"],
                                            _fa_jk_k["t_max"], 100,
                                            _fa_jk_k.get("wl_min", 200),
                                            _fa_jk_k.get("wl_max", 800),
                                            n_jobs=_jobs_jk_k,
                                            tolerance=_fa_jk_k["tolerance"],
                                            maxiter=_fa_jk_k["maxiter"],
                                            timeout_s=_to_s_k,
                                            constraints=_fa_jk_k["constraints"],
                                            fit_conc_keys=_fa_jk_k["fit_conc_keys"],
                                            allow_negative_eps=_fa_jk_k.get("allow_neg_eps", False),
                                            progress_callback=_cb_jk_k)
                                    elif _fa_jk_k.get("use_nmr_fit") and _fa_jk_k.get("nmr_mode") == "shift":
                                        _jr_k = ebs.jackknife_kinetics_nmr_shift(
                                            _fa_jk_k["parsed_fit"],
                                            _fa_jk_k["start_logk"],
                                            _fa_jk_k["nmr_data_fit"],
                                            _fa_jk_k["fit_keys"],
                                            _fa_jk_k["t_max"], 100,
                                            n_jobs=_jobs_jk_k,
                                            tolerance=_fa_jk_k["tolerance"],
                                            maxiter=_fa_jk_k["maxiter"],
                                            timeout_s=_to_s_k,
                                            constraints=_fa_jk_k["constraints"],
                                            fit_conc_keys=_fa_jk_k["fit_conc_keys"],
                                            progress_callback=_cb_jk_k)
                                    elif _fa_jk_k.get("use_nmr_fit") and _fa_jk_k.get("nmr_mode") == "integration":
                                        _jr_k = ebs.jackknife_kinetics_nmr_integration(
                                            _fa_jk_k["parsed_fit"],
                                            _fa_jk_k["start_logk"],
                                            _fa_jk_k["nmr_data_fit"],
                                            _fa_jk_k["fit_keys"],
                                            _fa_jk_k["t_max"], 100,
                                            n_jobs=_jobs_jk_k,
                                            tolerance=_fa_jk_k["tolerance"],
                                            maxiter=_fa_jk_k["maxiter"],
                                            timeout_s=_to_s_k,
                                            constraints=_fa_jk_k["constraints"],
                                            fit_conc_keys=_fa_jk_k["fit_conc_keys"],
                                            progress_callback=_cb_jk_k)
                                    elif _fa_jk_k.get("use_nmr_fit") and _fa_jk_k.get("nmr_mode") == "mixed":
                                        _jr_k = ebs.jackknife_kinetics_nmr_mixed(
                                            _fa_jk_k["parsed_fit"],
                                            _fa_jk_k["start_logk"],
                                            _fa_jk_k["nmr_data_fit"],
                                            _fa_jk_k["fit_keys"],
                                            _fa_jk_k["t_max"], 100,
                                            n_jobs=_jobs_jk_k,
                                            tolerance=_fa_jk_k["tolerance"],
                                            maxiter=_fa_jk_k["maxiter"],
                                            timeout_s=_to_s_k,
                                            constraints=_fa_jk_k["constraints"],
                                            fit_conc_keys=_fa_jk_k["fit_conc_keys"],
                                            progress_callback=_cb_jk_k)
                                    else:
                                        # pure kinetics
                                        _jr_k = ebs.jackknife_kinetics(
                                            _fa_jk_k["parsed_fit"],
                                            _fa_jk_k["exp_data_fit"],
                                            _fa_jk_k["start_logk"],
                                            _fa_jk_k["fit_keys"],
                                            _fa_jk_k["t_max"], 100,
                                            n_jobs=_jobs_jk_k,
                                            tolerance=_fa_jk_k["tolerance"],
                                            maxiter=_fa_jk_k["maxiter"],
                                            timeout_s=_to_s_k,
                                            constraints=_fa_jk_k["constraints"],
                                            fit_conc_keys=_fa_jk_k["fit_conc_keys"],
                                            progress_callback=_cb_jk_k)
                                    if _jr_k is not None:
                                        st.session_state["_jk_result_kin"] = _jr_k
                                except hpc_ui.CapturedCall as _hpc_cc:
                                    _pb_jk_k.empty()
                                    hpc_ui.handle_capture(_hpc_cc, "jk_kin", "Jackknife",
                                                          context={})
                                except Exception as _jke_k:
                                    _pb_jk_k.empty()
                                    import traceback as _tb_jk_k
                                    st.error(f"Jackknife failed: {_jke_k}")
                                    st.caption(_tb_jk_k.format_exc())
                                else:
                                    _pb_jk_k.empty()

                            hpc_ui.jobs_panel("jk_kin", on_result=_hpc_store_state("_jk_result_kin"))
                            _jr_k = st.session_state.get("_jk_result_kin")
                            if _jr_k is not None and not _jr_k.get("jack_se"):
                                _msg = (_jr_k.get("best_message")
                                         or "Jackknife produced no "
                                            "estimates (0 successful refits).")
                                st.warning(f"Jackknife: {_msg}")
                            if _jr_k and _jr_k.get("jack_se"):
                                st.caption(
                                    f"{_jr_k['n_success']}/{_jr_k['n_jack']} "
                                    f"refits successful "
                                    f"({_jr_k['wall_seconds']:.1f}s)")
                                _rows_jk_k = []
                                _kinds_kin = _jr_k.get("param_kinds", {}) or {}
                                for _n, _se in _jr_k["jack_se"].items():
                                    _fv = _jr_k["best_fit"].get(
                                        _n, float("nan"))
                                    _kind = _kinds_kin.get(_n, "log")
                                    if   _kind == "log":       _unit_str = "log10"
                                    elif _kind == "linear_k":  _unit_str = "linear"
                                    else:                       _unit_str = "mM"
                                    _rows_jk_k.append({
                                        "parameter":  _n,
                                        "unit":       _unit_str,
                                        "θ_full":     f"{_fv:+.2e}",
                                        "SE_jack":    (f"{_se:.2e}"
                                                        if np.isfinite(_se)
                                                        else "—"),
                                        "95 % CI (≈ ±1.96·SE)": (
                                            f"[{_fv - 1.96*_se:+.2e}, "
                                            f"{_fv + 1.96*_se:+.2e}]"
                                            if np.isfinite(_se) else "—"),
                                    })
                                st.dataframe(_rows_jk_k, hide_index=True,
                                              use_container_width=True)

                                _show_inf_k = st.checkbox(
                                    "Show per-step influence plot",
                                    value=True,
                                    key="_show_jk_influence_kin")
                                if _show_inf_k:
                                    from equilibrist_diagnostics import \
                                        make_jackknife_figure
                                    _x_kind_k = st.radio(
                                        "Plot x-axis",
                                        options=["x_value", "index"],
                                        horizontal=True,
                                        key="_jk_xkind_kin")
                                    _fig_jk_k = make_jackknife_figure(
                                        _jr_k, x_kind=_x_kind_k,
                                        x_label="Time [s]",
                                        title="Jackknife (leave-one-out) "
                                              "influence")
                                    st.pyplot(_fig_jk_k, clear_figure=False)
                                    import io as _io_jk
                                    _buf_jk_k = _io_jk.BytesIO()
                                    _fig_jk_k.savefig(_buf_jk_k,
                                                       format="png", dpi=200,
                                                       bbox_inches="tight")
                                    _buf_jk_k.seek(0)
                                    _ts_jk_k = datetime.now().strftime(
                                        "%Y%m%d_%H%M%S")
                                    st.download_button(
                                        "Download influence plot PNG",
                                        data=_buf_jk_k.getvalue(),
                                        file_name=f"Equilibrist_jackknife_"
                                                  f"{_ts_jk_k}.png",
                                        mime="image/png",
                                        key="_jk_dl_kin")
                _jk_kin_frag()

            # ── v2 Monte Carlo (nuisance-parameter uncertainty, kin) ─────
            # Parallel to the equilibrium-branch MC: samples relative
            # uncertainties on initial concentrations and V₀, refits with
            # the perturbed (assumed-known) values, and reports the σ
            # contribution that bootstrap / Hessian don't see.
            if fit_stats:
                @_fragment
                def _mc_kin_frag():
                    if st.checkbox("🎲 Show Monte Carlo "
                                   "(experimental-uncertainty propagation)",
                                   value=False, key="_show_mc_kin",
                                   help="For each iteration the MC samples "
                                        "user-specified relative uncertainties "
                                        "on initial concentrations and V₀, "
                                        "then refits the same observed data "
                                        "with the perturbed (assumed-known) "
                                        "values.  The σ on the fitted rate "
                                        "constants captures contributions from "
                                        "experimental-prep uncertainty — "
                                        "invisible to the residual-noise "
                                        "bootstrap."):
                        _fa_mc_k = _diag_fa(st.session_state.get("_fit_args_kin"))
                        if not _fa_mc_k:
                            st.info("Click **Fit** once with the current "
                                    "script and data to enable Monte Carlo.")
                        else:
                            # Nuisance parameters for kinetics: anything in
                            # parsed["concentrations"] not in fit_conc_keys,
                            # plus anything in parsed["volumes"].
                            _parsed_mc_k = _fa_mc_k.get("parsed_fit") or {}
                            _concs_k = _parsed_mc_k.get("concentrations", {}) or {}
                            _vols_k  = _parsed_mc_k.get("volumes",        {}) or {}
                            _fck_k   = set(_fa_mc_k.get("fit_conc_keys", []) or [])
                            _conc_nuis_k = [n for n in _concs_k
                                             if (n[:-1] if n.endswith("0") else n)
                                                 not in _fck_k]

                            st.caption("Default σ = 0 % (no contribution). "
                                       "Set realistic relative uncertainties "
                                       "for each prep-error source. Typical "
                                       "values: 2–3 % for gravimetric stock "
                                       "concentrations, 0.5–1 % for "
                                       "volumetric V₀.")
                            _specs_k: dict = {}
                            _all_inputs_k: list = []
                            for _n in _conc_nuis_k:
                                _all_inputs_k.append(("conc",
                                                       f"{_n} (nominal "
                                                       f"{_concs_k[_n]:.3g} mM)",
                                                       _n))
                            for _vn, _vv in _vols_k.items():
                                _all_inputs_k.append(("vol",
                                                       f"{_vn} (nominal "
                                                       f"{_vv:.3g} mL)",
                                                       _vn))

                            _mc_is_nmr_k = bool(_fa_mc_k.get("use_nmr_fit")) or (
                                _fa_mc_k.get("kind") == "global" and any(
                                    (e.get("fa") or {}).get("use_nmr_fit") for e in _fa_mc_k.get("experiments", [])))
                            if not _all_inputs_k and not _mc_is_nmr_k:
                                st.info("No nuisance parameters to sample "
                                        "— every experimental quantity is "
                                        "already a fit variable.")
                            else:
                                _ncols_inp = 2
                                _cols_inp_k = st.columns(_ncols_inp)
                                for _ii, (_kind, _label, _name) in enumerate(_all_inputs_k):
                                    _col = _cols_inp_k[_ii % _ncols_inp]
                                    _val = _col.number_input(
                                        f"σ on {_label} (%)",
                                        min_value=0.0, max_value=50.0,
                                        value=0.0, step=0.1,
                                        key=f"_mc_sig_kin_{_kind}_{_name}",
                                        format="%.2f")
                                    if _val > 0:
                                        if _kind == "conc":
                                            _specs_k[f"conc:{_name}"] = _val / 100.0
                                        elif _kind == "vol":
                                            _specs_k[f"vol:{_name}"]  = _val / 100.0
                                if _mc_is_nmr_k:
                                    _mn1, _mn2 = st.columns(2)
                                    _mc_integ_pct_k = _mn1.number_input(
                                        "σ on NMR integrations (%)", min_value=0.0, max_value=50.0,
                                        value=0.0, step=0.5, key="_mc_sig_kin_nmr_integ", format="%.2f",
                                        help="Relative error of the peak integrals (per point).")
                                    _mc_shift_ppm_k = _mn2.number_input(
                                        "σ on chemical shifts (ppm)", min_value=0.0, max_value=1.0,
                                        value=0.0, step=0.001, key="_mc_sig_kin_nmr_shift", format="%.4f",
                                        help="Absolute error of the observed shifts (per point, ppm).")
                                    if _mc_integ_pct_k > 0:
                                        _specs_k["nmr_integ_rel"] = float(_mc_integ_pct_k) / 100.0
                                    if _mc_shift_ppm_k > 0:
                                        _specs_k["nmr_shift_ppm"] = float(_mc_shift_ppm_k)

                                _ci_n, _ci_p = st.columns([1, 1])
                                _n_mc_k = _ci_n.number_input(
                                    "MC iterations",
                                    min_value=50, max_value=5000,
                                    value=500, step=50,
                                    key="_mc_iter_kin")
                                _mc_par_k = _ci_p.checkbox(
                                    "Parallel (use all cores)",
                                    value=False, key="_mc_par_kin")
                                _hpc_go_mc_kin = hpc_ui.controls("mc_kin", label="the Monte Carlo run")

                                if not _specs_k:
                                    st.warning("All uncertainties are 0 % — "
                                               "MC would be degenerate. "
                                               "Set at least one σ > 0 "
                                               "before running.")

                                if st.button("Run Monte Carlo",
                                              key="_mc_run_kin",
                                              disabled=(not _specs_k)) or (_hpc_go_mc_kin and bool(_specs_k)):
                                    _pb_mc = st.progress(0.0, text="MC running…")
                                    _cb_mc = lambda i, N: _pb_mc.progress(
                                        min(i / max(N, 1), 1.0),
                                        text=f"MC {i}/{N}")
                                    _jobs_mc = -1 if _mc_par_k else 1
                                    try:
                                        _mr_k = None
                                        if _fa_mc_k.get("kind") == "global":
                                            _mr_k = eqglobal.monte_carlo_global(
                                                _fa_mc_k, nuisance_specs=_specs_k, n_iter=int(_n_mc_k),
                                                n_jobs=_jobs_mc, progress_callback=_cb_mc)
                                        elif _fa_mc_k.get("use_spectra_fit"):
                                            _mr_k = ebs.monte_carlo_kinetics_spectra(
                                                _fa_mc_k["parsed_fit"], _fa_mc_k["start_logk"],
                                                _fa_mc_k["spectra_data_fit"],
                                                _fa_mc_k["fit_keys"], _fa_mc_k["t_max"],
                                                float(_fa_mc_k.get("wl_min") or 0.0),
                                                float(_fa_mc_k.get("wl_max") or 1e9),
                                                nuisance_specs=_specs_k,
                                                n_iter=int(_n_mc_k), n_jobs=_jobs_mc,
                                                tolerance=_fa_mc_k["tolerance"], maxiter=_fa_mc_k["maxiter"],
                                                constraints=_fa_mc_k["constraints"],
                                                fit_conc_keys=_fa_mc_k["fit_conc_keys"],
                                                allow_negative_eps=_fa_mc_k.get(
                                                    "allow_neg_eps", False),
                                                progress_callback=_cb_mc)
                                        elif _fa_mc_k.get("use_nmr_fit") and _fa_mc_k.get("nmr_mode") == "shift":
                                            _mr_k = ebs.monte_carlo_kinetics_nmr_shifts(
                                                _fa_mc_k["parsed_fit"], _fa_mc_k["start_logk"],
                                                _fa_mc_k["nmr_data_fit"],
                                                _fa_mc_k["fit_keys"], _fa_mc_k["t_max"],
                                                nuisance_specs=_specs_k,
                                                n_iter=int(_n_mc_k), n_jobs=_jobs_mc,
                                                tolerance=_fa_mc_k["tolerance"], maxiter=_fa_mc_k["maxiter"],
                                                constraints=_fa_mc_k["constraints"],
                                                fit_conc_keys=_fa_mc_k["fit_conc_keys"],
                                                progress_callback=_cb_mc)
                                        elif _fa_mc_k.get("use_nmr_fit") and _fa_mc_k.get("nmr_mode") == "integration":
                                            _mr_k = ebs.monte_carlo_kinetics_nmr_integration(
                                                _fa_mc_k["parsed_fit"], _fa_mc_k["start_logk"],
                                                _fa_mc_k["nmr_data_fit"],
                                                _fa_mc_k["fit_keys"], _fa_mc_k["t_max"],
                                                nuisance_specs=_specs_k,
                                                n_iter=int(_n_mc_k), n_jobs=_jobs_mc,
                                                tolerance=_fa_mc_k["tolerance"], maxiter=_fa_mc_k["maxiter"],
                                                constraints=_fa_mc_k["constraints"],
                                                fit_conc_keys=_fa_mc_k["fit_conc_keys"],
                                                progress_callback=_cb_mc)
                                        elif _fa_mc_k.get("use_nmr_fit") and _fa_mc_k.get("nmr_mode") == "mixed":
                                            _mr_k = ebs.monte_carlo_kinetics_nmr_mixed(
                                                _fa_mc_k["parsed_fit"], _fa_mc_k["start_logk"],
                                                _fa_mc_k["nmr_data_fit"],
                                                _fa_mc_k["fit_keys"], _fa_mc_k["t_max"],
                                                nuisance_specs=_specs_k,
                                                n_iter=int(_n_mc_k), n_jobs=_jobs_mc,
                                                tolerance=_fa_mc_k["tolerance"], maxiter=_fa_mc_k["maxiter"],
                                                constraints=_fa_mc_k["constraints"],
                                                fit_conc_keys=_fa_mc_k["fit_conc_keys"],
                                                progress_callback=_cb_mc)
                                        else:
                                            _mr_k = ebs.monte_carlo_kinetics(
                                                _fa_mc_k["parsed_fit"], _fa_mc_k["exp_data_fit"],
                                                _fa_mc_k["start_logk"],
                                                _fa_mc_k["fit_keys"], _fa_mc_k["t_max"],
                                                nuisance_specs=_specs_k,
                                                n_iter=int(_n_mc_k), n_jobs=_jobs_mc,
                                                tolerance=_fa_mc_k["tolerance"], maxiter=_fa_mc_k["maxiter"],
                                                constraints=_fa_mc_k["constraints"],
                                                fit_conc_keys=_fa_mc_k["fit_conc_keys"],
                                                progress_callback=_cb_mc)
                                        if _mr_k is not None:
                                            st.session_state["_mc_result_kin"] = _mr_k
                                    except hpc_ui.CapturedCall as _hpc_cc:
                                        _pb_mc.empty()
                                        hpc_ui.handle_capture(_hpc_cc, "mc_kin", "Monte Carlo",
                                                              context={})
                                    except Exception as _mce_k:
                                        _pb_mc.empty()
                                        import traceback as _tb_mc
                                        st.error(f"Monte Carlo failed: {_mce_k}")
                                        st.caption(_tb_mc.format_exc())
                                    else:
                                        _pb_mc.empty()

                                hpc_ui.jobs_panel("mc_kin", on_result=_hpc_store_state("_mc_result_kin"))
                            _mr_k = st.session_state.get("_mc_result_kin")
                            if _mr_k is not None and not _mr_k.get("mc_se"):
                                st.warning(f"Monte Carlo: "
                                            f"{_mr_k.get('best_message', 'no result')}")
                            if _mr_k and _mr_k.get("mc_se"):
                                st.caption(
                                    f"{_mr_k['n_success']}/{_mr_k['n_iter']} "
                                    f"iterations succeeded "
                                    f"({_mr_k['wall_seconds']:.1f} s)")
                                _rows_mc_k = []
                                _kinds_mc_k = _mr_k.get("param_kinds", {}) or {}
                                for _n, _se in _mr_k["mc_se"].items():
                                    _fv   = _mr_k["best_fit"].get(_n, float("nan"))
                                    _kind = _kinds_mc_k.get(_n, "log")
                                    if   _kind == "log":       _ust = "log10"
                                    elif _kind == "linear_k":  _ust = "linear"
                                    else:                       _ust = "mM"
                                    _lo = _mr_k["mc_ci_lo"].get(_n, float("nan"))
                                    _hi = _mr_k["mc_ci_hi"].get(_n, float("nan"))
                                    _rows_mc_k.append({
                                        "parameter": _n,
                                        "unit":      _ust,
                                        "θ_nominal": f"{_fv:+.2e}",
                                        "MC σ":      (f"{_se:.2e}"
                                                       if np.isfinite(_se) else "—"),
                                        "95 % CI (from samples)":
                                            (f"[{_lo:+.2e}, {_hi:+.2e}]"
                                             if np.isfinite(_lo) and np.isfinite(_hi)
                                             else "—"),
                                    })
                                st.dataframe(_rows_mc_k, hide_index=True,
                                              use_container_width=True)
                                st.caption("**Comparing with bootstrap σ above** "
                                           "tells you which uncertainty source "
                                           "dominates: residual noise (bootstrap) "
                                           "or experimental prep error (MC).  If "
                                           "MC σ ≳ bootstrap σ, tightening stock "
                                           "concentrations / volume measurements "
                                           "will improve precision more than "
                                           "noise reduction.")
                _mc_kin_frag()

            # ── v2 Local sensitivity test (Masson/beta) — kinetics ─────
            if fit_stats:
                @_fragment
                def _lst_kin_frag():
                    if st.checkbox("🎚️ Show Local sensitivity test (Masson)",
                                   value=False, key="_show_lst_kin",
                                   help="Probes the local SSR surface "
                                        "around the optimum by perturbing "
                                        "each selected variable by ±δ "
                                        "in log10 space.  The paired-"
                                        "difference construction cancels "
                                        "cross-coupling terms exactly, "
                                        "isolating each variable's "
                                        "intrinsic stiffness.  Compared "
                                        "with the marginal Hessian σ, "
                                        "the ratio diagnoses parameter "
                                        "coupling.  3^N grid evaluations — "
                                        "tractable up to ~6–7 selected "
                                        "variables."):
                        _fa_lst_k = _diag_fa(st.session_state.get("_fit_args_kin"))
                        if not _fa_lst_k:
                            st.info("Click **Fit** once with the current "
                                    "script and data to enable the local "
                                    "sensitivity test.")
                        else:
                            # Variable list for kinetics: rate constants
                            # AND equilibrium constants are both in
                            # start_logk (build_kinetics_logk_dict
                            # includes both).  Initial concentrations
                            # live in parsed["concentrations"], volumes
                            # in parsed["volumes"].
                            _all_K_k = list(_fa_lst_k.get("start_logk", {}).keys())
                            _parsed_lst = _fa_lst_k.get("parsed_fit") or {}
                            _all_C_k = list((_parsed_lst.get("concentrations", {}) or {}).keys())
                            _all_V_k = list((_parsed_lst.get("volumes",        {}) or {}).keys())

                            _fit_K_k = set(_fa_lst_k.get("fit_keys",      []) or [])
                            _fit_C_k = set()
                            for _root in (_fa_lst_k.get("fit_conc_keys", []) or []):
                                # fit_conc_keys uses ROOT names (no
                                # trailing 0); match against parsed
                                # entries by stripping the suffix
                                for _full in _all_C_k:
                                    if (_full[:-1] if _full.endswith("0") else _full) == _root:
                                        _fit_C_k.add(_full)

                            st.caption("Pick the variables to include "
                                       "in the test. Fitted parameters "
                                       "are checked by default; any "
                                       "held-fixed quantity can also be "
                                       "tested.")

                            _selected_k: list = []
                            if _all_K_k:
                                st.write("**Rate / equilibrium constants**")
                                _cKk = st.columns(min(3, max(1, len(_all_K_k))))
                                for _ii, _n in enumerate(_all_K_k):
                                    if _cKk[_ii % len(_cKk)].checkbox(
                                            _n, value=(_n in _fit_K_k),
                                            key=f"_lst_kin_chk_K_{_n}"):
                                        _selected_k.append(_n)
                            if _all_C_k:
                                st.write("**Initial concentrations**")
                                _cCk = st.columns(min(3, max(1, len(_all_C_k))))
                                for _ii, _n in enumerate(_all_C_k):
                                    if _cCk[_ii % len(_cCk)].checkbox(
                                            _n, value=(_n in _fit_C_k),
                                            key=f"_lst_kin_chk_C_{_n}"):
                                        _selected_k.append(_n)
                            if _all_V_k:
                                st.write("**Volumes**")
                                _cVk = st.columns(min(3, max(1, len(_all_V_k))))
                                for _ii, _n in enumerate(_all_V_k):
                                    if _cVk[_ii % len(_cVk)].checkbox(
                                            _n, value=False,
                                            key=f"_lst_kin_chk_V_{_n}"):
                                        _selected_k.append(_n)

                            _csz_k, _cpar_k = st.columns([1, 1])
                            _step_lst_k = _csz_k.number_input(
                                "Step δ (log10 units)",
                                min_value=0.001, max_value=1.0,
                                value=0.10, step=0.01, format="%.3f",
                                key="_lst_kin_step")
                            _par_lst_k = _cpar_k.checkbox(
                                "Parallel (use all cores)",
                                value=True, key="_lst_kin_par")
                            _hpc_go_lst_kin = hpc_ui.controls("lst_kin", label="the local sensitivity test")

                            _N = len(_selected_k)
                            if _N == 0:
                                st.warning("Select at least one variable "
                                           "to include in the test.")
                                _can_run = False
                            else:
                                n_grid = 3 ** _N
                                if _N >= 8:
                                    st.error(f"⚠ {_N} variables → 3^{_N} = "
                                              f"{n_grid:,} evaluations. "
                                              "That's likely too slow "
                                              "even parallel. Consider "
                                              "deselecting some.")
                                elif _N >= 6:
                                    st.warning(f"⚠ {_N} variables → 3^{_N} "
                                                f"= {n_grid:,} grid "
                                                "points — expect several "
                                                "minutes.")
                                else:
                                    st.caption(f"Grid: 3^{_N} = {n_grid:,} "
                                                "RMSE evaluations.")
                                _can_run = True

                            if st.button("Run local sensitivity test",
                                          key="_lst_kin_run",
                                          disabled=not _can_run) or (_hpc_go_lst_kin and _can_run):
                                _pb = st.progress(0.0, text="Local sensitivity running…")
                                _cb = lambda i, N: _pb.progress(
                                    min(i / max(N, 1), 1.0),
                                    text=f"Local sensitivity {i}/{N}")
                                _jobs = -1 if _par_lst_k else 1
                                _pe = fit_stats.get("param_errors", {}) or {}
                                _sigma_marg = {k: float(v) for k, v in _pe.items()
                                                if v is not None and np.isfinite(v)}
                                try:
                                    _lr = compute_local_sensitivity(
                                        _fa_lst_k, _selected_k,
                                        step=float(_step_lst_k),
                                        n_jobs=_jobs,
                                        sigma_marg=_sigma_marg,
                                        fit_stats=fit_stats,
                                        progress_callback=_cb)
                                    if _lr is not None:
                                        st.session_state["_lst_result_kin"] = _lr
                                except hpc_ui.CapturedCall as _hpc_cc:
                                    _pb.empty()
                                    hpc_ui.handle_capture(_hpc_cc, "lst_kin", "Local sensitivity",
                                                          context={})
                                except Exception as _le:
                                    _pb.empty()
                                    import traceback as _tb_lst
                                    st.error(f"Local sensitivity failed: {_le}")
                                    st.caption(_tb_lst.format_exc())
                                else:
                                    _pb.empty()

                            hpc_ui.jobs_panel("lst_kin", on_result=_hpc_store_state("_lst_result_kin"))
                            _lr = st.session_state.get("_lst_result_kin")
                            if _lr is not None and not _lr.get("f") and _lr.get("best_message"):
                                st.warning(f"Local sensitivity test: {_lr['best_message']}")
                            if _lr is not None and _lr.get("f"):
                                st.caption(
                                    f"{_lr['n_success']}/{_lr['n_grid']} "
                                    f"grid points evaluated "
                                    f"({_lr['wall_seconds']:.1f} s, "
                                    f"N_data = {_lr['n_data']}, "
                                    f"RMSE_opt = {_lr['rmse_opt']:.3e})")
                                _rows = []
                                for _v in _lr["selected_vars"]:
                                    _f  = _lr["f"].get(_v, float("nan"))
                                    _sc = _lr["sigma_cond"].get(_v, float("nan"))
                                    _sm = _lr["sigma_marg"].get(_v, float("nan"))
                                    _cp = _lr["coupling"].get(_v, float("nan"))
                                    if   np.isnan(_cp):    _cp_tag = "n/a"
                                    elif _cp <= 1.5:       _cp_tag = f"{_cp:.2f} ✓"
                                    elif _cp <= 3.0:       _cp_tag = f"{_cp:.2f} ⚠"
                                    else:                  _cp_tag = f"{_cp:.2f} ⚠⚠"
                                    _rows.append({
                                        "variable": _v,
                                        "ξ (% RMSE per ±δ)":
                                            (f"{_f*100:+.2f}%"
                                              if np.isfinite(_f) else "—"),
                                        "σ_cond (log10)":
                                            (f"{_sc:.2e}" if np.isfinite(_sc) else "—"),
                                        "σ_marg (Hessian, log10)":
                                            (f"{_sm:.2e}" if np.isfinite(_sm) else "—"),
                                        "Coupling factor":
                                            _cp_tag,
                                    })
                                st.dataframe(_rows, hide_index=True,
                                              use_container_width=True)
                                st.caption(
                                    "**σ_cond** is the conditional CI "
                                    "half-width (other variables held "
                                    "fixed); **σ_marg** is the marginal "
                                    "Hessian σ (other variables free to "
                                    "refit).  **Coupling factor** = "
                                    "σ_marg / σ_cond ≥ 1: ≤ 1.5 means "
                                    "the variable is independently "
                                    "constrained by the data; 1.5–3.0 "
                                    "indicates mild coupling; > 3 "
                                    "signals that the variable is "
                                    "mostly identified through coupling "
                                    "with another, and its individual "
                                    "value should be interpreted "
                                    "cautiously.")
                                # Per-variable expandable diff table
                                _drecs = _lr.get("diff_records", {})
                                if _drecs:
                                    with st.expander("Show individual paired differences"):
                                        for _v in _lr["selected_vars"]:
                                            _records = _drecs.get(_v, [])
                                            if not _records: continue
                                            st.markdown(f"**{_v}** — "
                                                        f"{len(_records)} paired diffs, "
                                                        f"mean = {_lr['f'].get(_v, float('nan'))*100:+.3f}%")
                                            _drows = []
                                            for _rec in _records:
                                                _sign_lbl = ("+δ" if _rec["sign"] > 0 else "−δ")
                                                _other_str = ", ".join(
                                                    f"{k}={v}" for k, v in _rec["others"].items())
                                                _drows.append({
                                                    f"sign of δ_{_v}": _sign_lbl,
                                                    "other variables": _other_str or "—",
                                                    "RMSE (unp.)":  f"{_rec.get('rmse_unp', float('nan')):.6e}",
                                                    "RMSE (pert.)": f"{_rec.get('rmse_pert', float('nan')):.6e}",
                                                    "diff r (%)": f"{_rec['diff']*100:+.4f}%",
                                                })
                                            st.dataframe(_drows, hide_index=True,
                                                          use_container_width=True)
                _lst_kin_frag()

            # ── v2 model comparison (AIC / BIC / F-test / Akaike weights) ────
            if fit_stats:
                @_fragment
                def _cmp_kin_frag():
                    if st.checkbox("⚖️ Show model comparison (AIC / BIC / F-test)",
                                   value=False, key="_show_cmp_kin"):
                        st.caption(
                            "Fit an alternative kinetic model on the same "
                            "experimental data and rank the two by information "
                            "criteria.  Akaike weights translate ΔAIC into an "
                            "evidence ratio (w_A / w_B = exp(ΔAIC / 2)).  Tick "
                            "**Nested?** when the simpler model is a special "
                            "case of the more complex one (e.g. single-step vs "
                            "two-step) to enable the F-test.")
                        _fa_k = _diag_fa(st.session_state.get("_fit_args_kin"))
                        _alt_script_k = st.text_area(
                            "Alternative kinetic model script",
                            height=180, key="_alt_script_text_kin",
                            placeholder=(
                                "$concentrations\n"
                                "A0 = 1.00 mM\n\n"
                                "$volumes\nV0 = 500 uL\n\n"
                                "$reactions\n"
                                "A → B;   log k1 = -2.0\n"
                                "B → C;   log k2 = -3.0\n\n"
                                "$plot\nxmax = 1000\nx = t\ny = A, B, C"))
                        _nested_k = st.checkbox(
                            "Nested? (enables F-test)", value=True,
                            key="_alt_nested_kin")
                        if st.button("Fit alternative & compare",
                                     key="_alt_compare_btn_kin"):
                            if not _fa_k:
                                st.warning("Click **Fit** with the current "
                                           "script first.")
                            elif not _alt_script_k.strip():
                                st.warning("Paste an alternative kinetic "
                                           "model above.")
                            else:
                                try:
                                    _alt_p_k = parse_script(_alt_script_k)
                                    _alt_logK_k = {
                                        **{e["kname"]: float(e["logK"])
                                           for e in _alt_p_k.get("equilibria", [])},
                                        **{r["kname"]: float(r["log_k"])
                                           for r in _alt_p_k.get("kinetics", [])},
                                        **{r["krname"]: float(r["log_kr"])
                                           for r in _alt_p_k.get("kinetics", [])
                                           if "krname" in r},
                                    }
                                    _alt_keys_k = list(_alt_logK_k.keys())
                                    _t_max_k    = float(_fa_k["t_max"])
                                    _tol_k      = float(_fa_k["tolerance"])
                                    _mxi_k      = int(_fa_k["maxiter"])
                                    _cstr_k     = _alt_p_k.get("constraints", [])
                                    with st.spinner("Fitting alternative kinetic model…"):
                                        if _fa_k["use_spectra_fit"]:
                                            _ok2, _f2, _s2, _m2 = fit_kinetics_spectra(
                                                _alt_p_k, _alt_logK_k,
                                                _fa_k["spectra_data_fit"],
                                                _alt_keys_k,
                                                t_max=_t_max_k, n_pts=200,
                                                tolerance=_tol_k, maxiter=_mxi_k,
                                                constraints=_cstr_k)
                                        elif _fa_k["use_nmr_fit"] and _fa_k["nmr_mode"] == "shift":
                                            _ok2, _f2, _s2, _m2 = fit_kinetics_nmr_shifts(
                                                _alt_p_k, _alt_logK_k,
                                                _fa_k["nmr_data_fit"], _alt_keys_k,
                                                _fa_k["x_expr"],
                                                t_max=_t_max_k, n_pts=200,
                                                tolerance=_tol_k, maxiter=_mxi_k,
                                                constraints=_cstr_k)
                                        elif _fa_k["use_nmr_fit"] and _fa_k["nmr_mode"] == "integration":
                                            _ok2, _f2, _s2, _m2 = fit_kinetics_nmr_integration(
                                                _alt_p_k, _alt_logK_k,
                                                _fa_k["nmr_data_fit"], _alt_keys_k,
                                                _fa_k["x_expr"],
                                                t_max=_t_max_k, n_pts=200,
                                                tolerance=_tol_k, maxiter=_mxi_k,
                                                constraints=_cstr_k)
                                        elif _fa_k["use_nmr_fit"] and _fa_k["nmr_mode"] == "mixed":
                                            _ok2, _f2, _s2, _m2 = fit_kinetics_nmr_mixed(
                                                _alt_p_k, _alt_logK_k,
                                                _fa_k["nmr_data_fit"], _alt_keys_k,
                                                _fa_k["x_expr"],
                                                t_max=_t_max_k, n_pts=200,
                                                tolerance=_tol_k, maxiter=_mxi_k,
                                                constraints=_cstr_k)
                                        else:
                                            _ok2, _f2, _s2, _m2 = fit_kinetics(
                                                _alt_p_k, _fa_k["exp_data_fit"] or {},
                                                _alt_logK_k, _alt_keys_k,
                                                _t_max_k, 200, _tol_k, _mxi_k,
                                                constraints=_cstr_k)
                                    if not _ok2:
                                        st.warning(f"Alternative model did not "
                                                   f"converge: {_m2}")
                                    else:
                                        augment_stats(_s2)
                                        st.session_state["_alt_stats_kin"]  = _s2
                                        st.session_state["_alt_fitted_kin"] = dict(_f2)
                                except Exception as _ce:
                                    import traceback as _tb_c
                                    st.error(f"Model comparison failed: {_ce}")
                                    st.caption(_tb_c.format_exc())
                        # Render last comparison if available
                        _s2 = st.session_state.get("_alt_stats_kin")
                        _f2 = st.session_state.get("_alt_fitted_kin")
                        if _s2 and fit_stats.get("aic") is not None and _s2.get("aic") is not None:
                            cmp = compare_models(fit_stats, _s2, nested=_nested_k,
                                                 label_a="Current", label_b="Alternative")
                            st.markdown(f"**{cmp['interpretation']}**")
                            _kA = fit_stats.get('n_params', 0) or 0   # = p (manuscript convention)
                            _kB = _s2.get('n_params', 0) or 0         # = p (manuscript convention)
                            _crows = [
                                {"Metric": "ℓ_max",
                                 "Current":     f"{fit_stats.get('log_likelihood', float('nan')):.3f}",
                                 "Alternative": f"{_s2.get('log_likelihood',  float('nan')):.3f}",
                                 "Δ (Alt − Cur)": f"{_s2.get('log_likelihood', 0)-fit_stats.get('log_likelihood', 0):+.3f}"},
                                {"Metric": "SSR",
                                 "Current":     f"{fit_stats.get('ssr', float('nan')):.5g}",
                                 "Alternative": f"{_s2.get('ssr',  float('nan')):.5g}",
                                 "Δ (Alt − Cur)": f"{_s2.get('ssr', 0)-fit_stats.get('ssr', 0):+.5g}"},
                                {"Metric": "p (fitted)",
                                 "Current": f"{_kA}", "Alternative": f"{_kB}",
                                 "Δ (Alt − Cur)": f"{_kB - _kA:+d}"},
                                {"Metric": "AIC",
                                 "Current":     f"{cmp['aic_a']:.2f}",
                                 "Alternative": f"{cmp['aic_b']:.2f}",
                                 "Δ (Alt − Cur)": f"{cmp['delta_aic']:+.2f}"},
                                {"Metric": "AICc",
                                 "Current":     f"{cmp['aicc_a']:.2f}",
                                 "Alternative": f"{cmp['aicc_b']:.2f}",
                                 "Δ (Alt − Cur)": f"{cmp['delta_aicc']:+.2f}"},
                                {"Metric": "BIC",
                                 "Current":     f"{cmp['bic_a']:.2f}",
                                 "Alternative": f"{cmp['bic_b']:.2f}",
                                 "Δ (Alt − Cur)": f"{cmp['delta_bic']:+.2f}"},
                            ]
                            st.dataframe(_crows, hide_index=True)
                            st.markdown(
                                f"**Akaike weights**: "
                                f"Current = **{cmp['weight_a']:.3f}** · "
                                f"Alternative = **{cmp['weight_b']:.3f}**  \n"
                                f"_(weight ≈ probability this model is the better of the two)_")
                            if _nested_k and np.isfinite(cmp['f_statistic']):
                                _dp = abs(_kB - _kA)
                                if cmp['f_statistic'] == 0.0 and cmp['f_p_value'] == 1.0:
                                    st.markdown(
                                        f"**F-test** (Δp = {_dp}): the more "
                                        "complex model does *not* reduce SSR "
                                        "→ no evidence whatsoever for the "
                                        "additional parameter(s).")
                                else:
                                    st.markdown(
                                        f"**F-test** (Δp = {_dp}): "
                                        f"F = {cmp['f_statistic']:.3f}, "
                                        f"p = {cmp['f_p_value']:.4g}")
                            # Show alternative model's fitted parameters
                            if _f2:
                                st.markdown("**Alternative model fitted constants:**")
                                _alt_rows = []
                                for n_alt, v_alt in _f2.items():
                                    _e_alt = _s2.get('param_errors', {}).get(n_alt)
                                    _alt_rows.append({
                                        "Parameter": n_alt,
                                        "log K/k": f"{v_alt:+.4f}",
                                        "± σ (Hessian)": (f"{_e_alt:.4f}"
                                                          if _e_alt is not None and np.isfinite(_e_alt)
                                                          else "—"),
                                    })
                                st.dataframe(_alt_rows, hide_index=True)

                _cmp_kin_frag()

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
                                     "±log P": f"± {err:.2f}" if err is not None else ("n/d" if kname in param_errors else "n/a"),
                                     "P":      f"{k_lin:.2e}",
                                     "±P":     f"± {err_lin:.2e}" if err_lin is not None else ("n/d" if kname in param_errors else "n/a")})
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
            # Display rule:
            #   • Default (no noref):       show absolute δ = δ_obs(t_free) + dd
            #     (same behavior as legacy app; dd_ref = 0 by auto-pin so the
            #     reference column shows the V=0 observed shift).
            #   • noref WITH read: anchor:  show absolute δ (scale anchored).
            #   • noref WITHOUT read::      show Δδ relative to math reference;
            #     absolute scale is undetermined (uniform-shift symmetry).
            _nmr_noref   = bool(fit_stats.get("nmr_noref", False))
            _ps_anchored = bool(fit_stats.get("pure_shifts_anchored", False))
            _show_abs    = (not _nmr_noref) or _ps_anchored
            _ps_dfree    = fit_stats.get("delta_free", {}) or {}
            _ps_und      = fit_stats.get("pure_shift_undetermined", {}) or {}
            _ps_mf       = fit_stats.get("pure_shift_maxfrac", {}) or {}
            _ps_pin      = fit_stats.get("pure_shift_pinned", {}) or {}
            if pure_shifts_disp:
                if _show_abs:
                    st.write("**Pure-species chemical shifts (ppm):**")
                else:
                    st.write("**Pure-species chemical shifts — Δδ relative to math reference (ppm):**")
                all_sp_cols = []; rows_ps = []
                for tgt, col_dict in pure_shifts_disp.items():
                    for col, sp_dict in col_dict.items():
                        row = {"Signal": col}
                        df0 = float(_ps_dfree.get(col, 0.0)) if _show_abs else 0.0
                        for sp, dd_val in sp_dict.items():
                            row[sp] = f"{df0 + dd_val:.4f}"
                            if sp in ((_ps_und.get(tgt) or {}).get(col) or []):
                                row[sp] = f"({row[sp]})"
                            elif (sp not in ((_ps_pin.get(tgt) or {}).get(col) or [])
                                  and float(((_ps_mf.get(tgt) or {}).get(col) or {}).get(sp, 1.0)) < 0.10):
                                row[sp] = f"{row[sp]} *"
                            if sp not in all_sp_cols:
                                all_sp_cols.append(sp)
                        rows_ps.append(row)
                if rows_ps:
                    df_ps = pd.DataFrame(rows_ps).set_index("Signal")
                    ordered = [c for c in all_sp_cols if c in df_ps.columns]
                    st.dataframe(df_ps[ordered], width='stretch')
                    if _show_abs:
                        st.caption("Each row = one NMR signal; columns = δ of each pure species (ppm). "
                                   "( ) = species below 2 % of the pool over this titration: its shift is not "
                                   "determined by these data and is held at the free shift (or at the value "
                                   "shared from the other tabs).  * = species below 10 % of the pool: its shift "
                                   "is poorly determined and may absorb misfit; tie it to a related species in "
                                   "$constraints if its value is implausible.")
                    else:
                        st.caption("Each row = one NMR signal; columns = Δδ relative to math reference (ppm). "
                                   "Add a `read:` anchor in $nmr to recover absolute δ.")

            _render_pure_shift_comparison(fit_stats)

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
# Sanitise conc_roots: replace None values with 1.0 mM default
conc_roots = {k: (v if v is not None else 1.0) for k, v in conc_roots.items()}



# ITC placeholder: all concentrations/volumes/titrant are bare names (no values)
# and no ITC file has been loaded yet. Show a friendly message instead of crashing.
_itc_placeholder = (
    all(v is None for v in parsed["concentrations"].values()) and
    all(v is None for v in parsed["volumes"].values()) and
    all(v is None for v in parsed["titrant"].values()) and
    st.session_state.get("_itc_raw") is None
)
if _itc_placeholder:
    st.warning("⬆️ Upload an ITC file to set concentrations and volumes automatically.")
    _ph_uploaded = st.file_uploader(
        "Upload ITC file (.itc)", type=["itc"],
        key=f"_itc_uploader_{st.session_state.get('_itc_uploader_nonce', 0)}",
    )
    if _ph_uploaded is not None:
        try:
            from equilibrist_itc import parse_itc_file as _parse_itc
            _itc_raw_ph = _parse_itc(_ph_uploaded.read())
            st.session_state["_itc_raw"] = _itc_raw_ph
            # Fill in concentrations/volumes/titrant from file header
            _cell_conc_ph   = _itc_raw_ph.get("cell_conc_mM")
            _syr_conc_ph    = _itc_raw_ph.get("syringe_conc_mM")
            _cell_vol_ph    = _itc_raw_ph.get("cell_vol_mL")
            _cname_ph = list(parsed["concentrations"].keys())[0]
            _tkey_ph  = list(parsed["titrant"].keys())[0]
            _vkey_ph  = list(parsed["volumes"].keys())[0]
            _root_ph  = _cname_ph[:-1] if _cname_ph.endswith("0") else _cname_ph
            _tfree_ph = _tkey_ph[:-1]  if _tkey_ph.endswith("t") else _tkey_ph
            if _cell_conc_ph:
                st.session_state[f"conc_{_root_ph}"]        = float(_cell_conc_ph)
                st.session_state[f"_pending_conc_{_root_ph}"] = float(_cell_conc_ph)
            if _syr_conc_ph:
                st.session_state[f"titrant_mM_{_tkey_ph}"]            = float(_syr_conc_ph)
                st.session_state[f"_pending_titrant_mM_{_tkey_ph}"]   = float(_syr_conc_ph)
            if _cell_vol_ph:
                st.session_state["V0_mL"] = float(_cell_vol_ph)
            st.rerun()
        except Exception as _e:
            st.error(f"Could not read ITC file: {_e}")
    st.stop()

# Sanitise parsed dicts: replace any remaining None with session state values.
# This runs only when _itc_placeholder is False (ITC file has been loaded).
for _k in list(parsed["concentrations"]):
    if parsed["concentrations"][_k] is None:
        _root = _k[:-1] if _k.endswith("0") else _k
        parsed["concentrations"][_k] = float(st.session_state.get(f"conc_{_root}", 1.0))
for _k in list(parsed["titrant"]):
    if parsed["titrant"][_k] is None:
        parsed["titrant"][_k] = float(st.session_state.get(f"titrant_mM_{_k}", 10.0))
for _k in list(parsed["volumes"]):
    if parsed["volumes"][_k] is None:
        parsed["volumes"][_k] = float(st.session_state.get("V0_mL", 0.1991))


# ── Sidebar: concentrations, volume, titrant ──
with st.sidebar:

    st.header("Initial concentrations (mM)")
    conc_vals = {}
    _is_acid_base = parsed.get("is_acid_base", False)
    for root, default in conc_roots.items():
        if _is_acid_base and root == "H2O":
            # H2O is implicit in acid-base mode — keep value but hide widget
            conc_vals[root] = float(st.session_state.get(f"conc_{root}", float(default) if default is not None else 1.0))
            continue
        _def_c = float(default) if default is not None else \
                 float(st.session_state.get(f"conc_{root}", 1.0))
        conc_vals[root] = _conc_input_with_fit(
            f"{root}₀", key=f"conc_{root}", default=_def_c,
        )

    st.header("Volume & Titrant")
    _v0_raw     = list(parsed["volumes"].values())[0]  if parsed["volumes"]  else 0.5
    V0_default  = float(_v0_raw) if _v0_raw is not None else \
                  float(st.session_state.get("V0_mL", ITC200_CELL_VOLUME_ML))
    V0_mL = _num_input("V₀ (mL)", key="V0_mL", default=V0_default, step=0.01, format="%.4f")

    titrant_mMs    = {}
    titrant_ratios = {}
    if parsed["titrant_is_solid"]:
        st.caption("🪨 Solid titrant — volume fixed at V₀")
        for tkey in titrant_keys:
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            raw_ratio = float(parsed["titrant_ratios"].get(tkey, 1.0))
            titrant_mMs[tfree] = 0.0  # not used for solid
            titrant_ratios[tfree] = _num_input(
                f"{tkey} ratio", key=f"titrant_ratio_{tkey}",
                default=raw_ratio, step=0.1, format="%.4f"
            )
        if len(titrant_keys) > 1:
            total_r = sum(titrant_ratios.values()) or 1.0
            for tfree, r in titrant_ratios.items():
                st.caption(f"  {tfree}: mole fraction = {r/total_r:.4f}")
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

    if _is_acid_base:
        st.header("pKa values")
    else:
        st.header("Equilibrium constants (log₁₀ K)")
    logK_vals = {}
    _rendered_knames = set()
    for eq in parsed["equilibria"]:
        kname = eq["kname"]

        # In acid-base mode: hide Kw (implicit) silently
        if _is_acid_base and kname == "Kw":
            logK_vals[kname] = eq["logK"]
            continue

        # Build reaction label for this equilibrium
        reactants_str = []
        for coeff, species in eq["reactants"]:
            if coeff == 1:
                reactants_str.append(species)
            else:
                reactants_str.append(f"{coeff}{species}")

        products_display = []
        if "products" in eq:
            for prod_coeff, prod_species in eq["products"]:
                if prod_coeff == 1:
                    products_display.append(prod_species)
                else:
                    products_display.append(f"{prod_coeff}{prod_species}")
        elif "product" in eq:
            prod_coeff, prod_species = eq["product"]
            if prod_coeff == 1:
                products_display.append(prod_species)
            else:
                products_display.append(f"{prod_coeff}{prod_species}")

        products_str = " + ".join(products_display)
        rxn_str = f"{' + '.join(reactants_str)} ⇌ {products_str}"

        if kname in _rendered_knames:
            logK_vals[kname] = float(st.session_state.get(f"logK_{kname}",
                                     logK_vals.get(kname, eq["logK"])))
        else:
            _sharing = [e for e in parsed["equilibria"] if e["kname"] == kname]
            if len(_sharing) > 1:
                all_rxn_strs = []
                for _e in _sharing:
                    _r = " + ".join(
                        sp if c == 1 else f"{c}{sp}"
                        for c, sp in _e["reactants"])
                    _p_list = []
                    if "products" in _e:
                        _p_list = [sp if c == 1 else f"{c}{sp}"
                                   for c, sp in _e["products"]]
                    elif "product" in _e:
                        c, sp = _e["product"]
                        _p_list = [sp if c == 1 else f"{c}{sp}"]
                    all_rxn_strs.append(f"{_r} ⇌ {' + '.join(_p_list)}")
                label = f"{kname}  ({' = '.join(all_rxn_strs)})"
            else:
                label = f"{kname}  ({rxn_str})"

            if _is_acid_base:
                # Show as pKa (positive number); logK returned for solver
                _pka_default = -eq["logK"]
                _pka_key  = f"pKa_{kname}"
                _logk_key = f"logK_{kname}"
                logK_vals[kname] = _pka_input_with_fit(
                    label, pka_key=_pka_key, logk_key=_logk_key,
                    default_pka=_pka_default)
            else:
                logK_vals[kname] = _logk_input_with_fit(label, key=f"logK_{kname}", default=eq["logK"])
            _rendered_knames.add(kname)

# ── ITC q_offset widget (only shown when ITC data is loaded) ──────────
_itc_raw_sb = st.session_state.get("_itc_raw")
if _itc_raw_sb is not None:
    with st.sidebar:
        st.header("ITC offset")
        # Apply pending writeback from fitter (same pattern as logK/conc widgets)
        if "_pending_itc_q_offset" in st.session_state:
            st.session_state["itc_q_offset"] = float(st.session_state.pop("_pending_itc_q_offset"))
        _qoffset_default = float(st.session_state.get("itc_q_offset", 0.0))
        # Identical layout to K1/conc widgets: narrow fit checkbox column
        _qoff_c1, _qoff_c2 = st.columns([0.86, 0.14])
        with _qoff_c1:
            _itc_q_offset = _num_input(
                "q_offset (µcal)", key="itc_q_offset",
                default=_qoffset_default, step=0.01, format="%.4f")
        with _qoff_c2:
            # No value= : defaults to False, consistent with every other
            # "fit" checkbox. Forcing True here meant q_offset was silently
            # floated the moment an .itc file was loaded.
            _fit_toggle("fit_itc_q_offset")

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
    pyscript.sidebar_controls()          # "Load custom .py script" / "Run py script"
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
if _itc_placeholder:
    _maxEquiv = 2.4  # dummy — not used for rendering
else:
    # Sanitise: replace any remaining None values before calling _find_maxEquiv
    conc_vals   = {k: (v if v is not None else 1.0)  for k, v in conc_vals.items()}
    titrant_mMs = {k: (v if v is not None else 10.0) for k, v in titrant_mMs.items()}
    # _find_maxEquiv reads parsed["concentrations"] directly — patch it too
    _parsed_maxequiv = dict(parsed)
    _parsed_maxequiv["concentrations"] = dict(parsed["concentrations"])
    for _cn, _cv in conc_vals.items():
        _k0 = _cn + "0"
        if _k0 in _parsed_maxequiv["concentrations"]:
            _parsed_maxequiv["concentrations"][_k0] = float(_cv)
        elif _cn in _parsed_maxequiv["concentrations"]:
            _parsed_maxequiv["concentrations"][_cn] = float(_cv)
    # Non-ITC behaviour must be untouched: only ITC placeholders need the
    # widget-patched copy, because their script concentrations were None.
    _pm_arg = _parsed_maxequiv if st.session_state.get("_itc_raw") is not None else parsed
    _maxEquiv = _find_maxEquiv(
        _x_expr_main, float(xmax), _pm_arg, conc_vals, float(V0_mL),
        titrant_free_names, titrant_keys, titrant_mMs, titrant_ratios,
        parsed["titrant_is_solid"], primary_component,
    )

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


if _itc_placeholder:
    curve = {}   # empty dict — downstream .get() and iteration are safe
else:
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

# ── ITC Fit dispatch ─────────────────────────────────────────────────
_itc_raw_disp = st.session_state.get("_itc_raw")
_itc_q_exp    = st.session_state.get("_itc_q_exp")
_use_itc_fit  = (_itc_raw_disp is not None and
                 _itc_q_exp is not None and
                 len(_itc_q_exp) > 0)

# Reserved slot for the 'Fitting…' message.  Emitted on every run so the
# element that follows (the two-column layout) keeps the same position;
# otherwise the spinner, appearing only on fitting runs, took the columns'
# place and the right-hand panel vanished until the fit returned.
_fit_slot = st.empty()
if st.session_state.get("_single_fit_pending"):
    _apply_single_pending(st.session_state.pop("_single_fit_pending"))
if st.session_state.get("_fit_requested") and st.session_state.get("_single_fit_hpc"):
    st.session_state.pop("_fit_requested", None)
    _submit_single_fit_to_cluster(st.session_state.pop("_single_fit_hpc"))
if _use_itc_fit and _fit_turn(_fit_slot):
    _itc_fit_keys  = list(dict.fromkeys(
        eq["kname"] for eq in parsed["equilibria"]
        if st.session_state.get(f"fit_logK_{eq['kname']}", False)
    ))
    _itc_conc_keys = [root for root in conc_roots
                      if st.session_state.get(f"fit_conc_{root}", False)]
    # Titrant stock: the checkbox key uses the titrant KEY ('Ht') while
    # params["titrant_mMs"] is keyed by the FREE name ('H'), so map across
    # the two parallel lists. Previously this was not collected at all, so
    # ticking "fit" on the syringe stock silently did nothing.
    _itc_tit_keys  = [_fname for _tkey, _fname in zip(titrant_keys,
                                                       titrant_free_names)
                      if st.session_state.get(f"fit_titrant_mM_{_tkey}", False)]
    _itc_any_fit   = bool(_itc_fit_keys or _itc_conc_keys or _itc_tit_keys)
    if _itc_any_fit:
        try:
            _itc_tol_log  = st.session_state.get("fit_tolerance_log", 3.0)
            _itc_tol      = 10.0 ** (-_itc_tol_log)
            _itc_timeout  = float(st.session_state.get("fit_timeout", 30))
            _itc_constr   = (parsed.get("constraints", [])
                             if st.session_state.get("fit_use_constraints", False) else [])
            # q_offset: free (lstsq) when fit checkbox is ticked, fixed otherwise
            _qoff_fit = bool(st.session_state.get("fit_itc_q_offset", True))
            _qoff_arg = None if _qoff_fit else float(st.session_state.get("itc_q_offset", 0.0))

            # Pass the full raw dict and excluded set to fit_itc.
            # fit_itc computes dC for ALL injections then masks — this ensures
            # dC[i] = C[i] - C[i-1] is correct even when early injections are excluded.
            _itc_excl = st.session_state.get("_outliers_itc", set())
            # q_exp[0] is always the dummy injection — skip it; fit only real injections
            _itc_q_fit = np.array([q for i, q in enumerate(_itc_q_exp[1:])
                                   if i not in _itc_excl])

            _itc_success, _itc_fitted, _itc_stats, _itc_msg = _fit_async(_fit_slot, _itc_timeout, lambda: fit_itc(
                    parsed, network, _itc_q_fit, _itc_raw_disp, params, logK_vals,
                    _itc_fit_keys, x_expr,
                    fit_titrant_keys=_itc_tit_keys,
                    tolerance=_itc_tol, maxiter=100_000,
                    timeout_s=_itc_timeout,
                    fit_conc_keys=_itc_conc_keys,
                    q_offset_fixed=_qoff_arg,
                    constraints=_itc_constr,
                    excl_set=_itc_excl,
                ),
                logK_vals, "Fitting ITC parameters…", "_fit_requested")

            # Write back fitted K values
            for kname, val in _itc_fitted.items():
                st.session_state[f"_pending_logK_{kname}"] = float(val)
            for root, mM in _itc_stats.get("fitted_concs", {}).items():
                st.session_state[f"_pending_conc_{root}"] = float(mM)
            # Fitted titrant stocks: the widget key uses the titrant KEY
            # ('Ht') while the fit returns the FREE name ('H'), so map back.
            _free_to_tkey = {_f: _t for _t, _f in zip(titrant_keys,
                                                       titrant_free_names)}
            for _fname, mM in _itc_stats.get("fitted_titrants", {}).items():
                _tk = _free_to_tkey.get(_fname, _fname)
                st.session_state[f"_pending_titrant_mM_{_tk}"] = float(mM)
            # Write back q_offset only when it was solved (not fixed)
            if _qoff_fit:
                st.session_state["_pending_itc_q_offset"] = float(_itc_stats.get("q_offset", 0.0))

            # Residual diagnostics (residuals vs predictor, Q-Q, DW, Shapiro,
            # AIC/BIC) need the unified collector, exactly as the equilibrium
            # dispatch does. Without this the panel renders empty for ITC.
            try:
                _itc_res = collect_residuals_from_stats(_itc_stats, x_expr=x_expr)
                augment_stats(_itc_stats, _itc_res)
                # The diagnostics panel is gated on this session key, not on
                # the stats dict: it renders only when
                # st.session_state["_fit_residuals"] has a non-empty y_obs.
                # Computing the residuals without storing them here left the
                # whole panel invisible for ITC.
                st.session_state["_fit_residuals"] = _itc_res
            except Exception:
                st.session_state["_fit_residuals"] = {}
            st.session_state["_fit_stats"] = _itc_stats
            st.session_state["_fit_counter"] = int(st.session_state.get("_fit_counter", 0)) + 1
            # Statistics panels (profiles, bootstrap, jackknife, MC) read
            # _fit_args_eq. The equilibrium dispatch builds it; ITC has its
            # own dispatch, so build the ITC flavour here or none of the
            # v2.0 diagnostics can reach an ITC fit.
            st.session_state["_fit_args_eq"] = {
                "kind":             "eq",
                "use_itc_fit":      True,
                "use_spectra_fit":  False,
                "use_nmr_fit":      False,
                "fit_keys":         list(_itc_fit_keys),
                "fit_conc_keys":    list(_itc_conc_keys),
                "fit_titrant_keys": list(_itc_tit_keys),
                "tolerance":        float(_itc_tol),
                "maxiter":          100_000,
                "timeout_s":        float(_itc_timeout),
                "constraints":      _itc_constr,
                "start_logK":       dict(logK_vals),
                "start_logk":       dict(logK_vals),
                "x_expr":           x_expr,
                "parsed_fit":       parsed,
                "network":          network,
                "params":           params,
                "itc_q_exp":        _itc_q_fit,
                "itc_raw":          _itc_raw_disp,
                "q_offset_fixed":   None,
                "excl_set":         set(_itc_excl or set()),
                "exp_data_fit":     None,
                "nmr_data_fit":     None,
                "spectra_data_fit": None,
                "n_data":           int(_itc_stats.get("n_points", 0)),
            }
            _itc_n_par = (len(_itc_fitted)
                          + len(_itc_stats.get("fitted_concs", {}))
                          + len(_itc_stats.get("fitted_titrants", {})))
            st.session_state["_fit_message"] = _fit_end_message(
                _itc_stats, _itc_success, _itc_msg, _itc_n_par, _itc_tol_log, _itc_timeout)
            st.rerun()
        except Exception as _itc_exc:
            import traceback as _itc_tb
            st.error(f"ITC fitting error: {_itc_exc}")
            st.caption(_itc_tb.format_exc())

# ── Fit dispatch — runs BEFORE plots so spinner appears at top ────────
# ── What a Global fit would use from THIS tab: refreshed every run so the
# snapshot of an inactive tab always carries its current script, data,
# ticks, values and constraints.
if not _use_itc_fit:
    try:
        _nmr_cfg_live = parsed.get("nmr")
        _nmr_live = _filter_exp_outliers(st.session_state.get("_nmr_data", {}), "_outliers_nmr")
        _exp_live = _filter_exp_outliers(st.session_state.get("_exp_data", {}), "_outliers_main")
        _sp_live  = _filter_spectra_outliers(st.session_state.get("_spectra_data", {}), "_outliers_spectra")
        _use_nmr_live = (_nmr_cfg_live is not None and _nmr_cfg_live["mode"] in ("shift", "integration", "mixed")
                         and bool(_nmr_live))
        _use_sp_live = parsed.get("spectra") is not None and bool(_sp_live)
        _tol_live = 10.0 ** (-float(st.session_state.get("fit_tolerance_log", 3.0)))
        st.session_state["_fit_args_live"] = {
            "kind": "eq",
            "fit_keys": list(dict.fromkeys(eq["kname"] for eq in parsed["equilibria"]
                                           if st.session_state.get(f"fit_logK_{eq['kname']}", False))),
            "fit_conc_keys": [root for root in conc_roots if st.session_state.get(f"fit_conc_{root}", False)],
            "fit_titrant_keys": [tkey for tkey in titrant_keys if st.session_state.get(f"fit_titrant_mM_{tkey}", False)],
            "use_spectra_fit": bool(_use_sp_live), "use_nmr_fit": bool(_use_nmr_live),
            "nmr_mode": (_nmr_cfg_live["mode"] if _use_nmr_live else None),
            "wl_min": (float(st.session_state.get("spectra_wl_min", _sp_live["wavelengths"][0])) if _use_sp_live else None),
            "wl_max": (float(st.session_state.get("spectra_wl_max", _sp_live["wavelengths"][-1])) if _use_sp_live else None),
            "allow_neg_eps": bool(st.session_state.get("spectra_allow_neg", False)),
            "tolerance": float(_tol_live), "maxiter": 100_000,
            "timeout_s": float(st.session_state.get("fit_timeout", 30)),
            "constraints": (parsed.get("constraints", []) if st.session_state.get("fit_use_constraints", False) else []),
            "start_logK": dict(logK_vals), "start_logk": dict(logK_vals),
            "x_expr": x_expr,
            "exp_data_fit": (_exp_live if not (_use_nmr_live or _use_sp_live) else None),
            "nmr_data_fit": (_nmr_live if _use_nmr_live else None),
            "spectra_data_fit": (_sp_live if _use_sp_live else None),
            "parsed_fit": parsed, "network": network, "params": params,
            "has_data": bool(_use_nmr_live or _use_sp_live or _exp_live),
        }
    except Exception:
        st.session_state.pop("_fit_args_live", None)
else:
    # ITC tab: what a Global fit would use from it (mirrors the ITC fit dispatch)
    try:
        _itc_excl_live = set(st.session_state.get("_outliers_itc", set()) or set())
        _itc_q_all = st.session_state.get("_itc_q_exp")
        _itc_q_live = (np.array([q for i, q in enumerate(_itc_q_all[1:]) if i not in _itc_excl_live])
                       if _itc_q_all is not None and len(_itc_q_all) > 1 else None)
        _qoff_fit_live = bool(st.session_state.get("fit_itc_q_offset", True))
        st.session_state["_fit_args_live"] = {
            "kind": "eq", "use_itc_fit": True, "use_spectra_fit": False, "use_nmr_fit": False,
            "fit_keys": list(dict.fromkeys(eq["kname"] for eq in parsed["equilibria"]
                                           if st.session_state.get(f"fit_logK_{eq['kname']}", False))),
            "fit_conc_keys": [root for root in conc_roots if st.session_state.get(f"fit_conc_{root}", False)],
            "fit_titrant_keys": [_fname for _tkey, _fname in zip(titrant_keys, titrant_free_names)
                                 if st.session_state.get(f"fit_titrant_mM_{_tkey}", False)],
            "tolerance": 10.0 ** (-float(st.session_state.get("fit_tolerance_log", 3.0))),
            "maxiter": 100_000, "timeout_s": float(st.session_state.get("fit_timeout", 30)),
            "constraints": (parsed.get("constraints", []) if st.session_state.get("fit_use_constraints", False) else []),
            "start_logK": dict(logK_vals), "start_logk": dict(logK_vals),
            "x_expr": x_expr, "parsed_fit": parsed, "network": network, "params": params,
            "itc_q_exp": _itc_q_live, "itc_raw": _itc_raw_disp,
            "q_offset_fixed": (None if _qoff_fit_live else float(st.session_state.get("itc_q_offset", 0.0))),
            "excl_set": _itc_excl_live,
            "exp_data_fit": None, "nmr_data_fit": None, "spectra_data_fit": None,
            "has_data": bool(_itc_q_live is not None and len(_itc_q_live) > 0),
        }
    except Exception:
        st.session_state.pop("_fit_args_live", None)

# ── Global fit dispatch (all tabs at once) ────────────────────────────
if _fit_turn(_fit_slot, "_global_fit_requested"):
    _global_fit_dispatch(_fit_slot)
if st.session_state.get("_global_fit_pending"):
    _apply_global_pending(st.session_state.pop("_global_fit_pending"))

if not _use_itc_fit and _fit_turn(_fit_slot):
    exp_data_fit     = _filter_exp_outliers(st.session_state.get("_exp_data", {}),      "_outliers_main")
    nmr_data_fit     = _filter_exp_outliers(st.session_state.get("_nmr_data", {}),      "_outliers_nmr")
    spectra_data_fit = _filter_spectra_outliers(st.session_state.get("_spectra_data", {}), "_outliers_spectra")
    nmr_cfg_fit      = parsed.get("nmr")
    use_nmr_fit      = (nmr_cfg_fit is not None and
                        nmr_cfg_fit["mode"] in ("shift", "integration", "mixed") and
                        bool(nmr_data_fit))
    use_spectra_fit  = parsed.get("spectra") is not None and bool(spectra_data_fit)

    fit_keys_end = list(dict.fromkeys(
        eq["kname"] for eq in parsed["equilibria"]
        if st.session_state.get(f"fit_logK_{eq['kname']}", False)
    ))
    has_data_end = use_nmr_fit or use_spectra_fit or bool(exp_data_fit)

    # Concentrations and titrant to fit
    fit_conc_keys_end = [root for root in conc_roots
                         if st.session_state.get(f"fit_conc_{root}", False)]
    fit_titrant_keys_end = [tkey for tkey in titrant_keys
                            if st.session_state.get(f"fit_titrant_mM_{tkey}", False)]

    if has_data_end and (fit_keys_end or fit_conc_keys_end or fit_titrant_keys_end):
        try:
            tolerance_log   = st.session_state.get("fit_tolerance_log", 3.0)
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

            # When fitting concentrations, reset to script defaults: the sidebar
            # logK values may be far from the true optimum at the script A0,
            # causing L-BFGS-B to push A0 to its boundary before logK can adjust.
            #
            # The constraint toggle flipping does NOT trigger a reset.  The
            # sidebar K's (typically the previous fit's output) are usually a
            # much better starting point than script defaults, even if they
            # violate the soft constraint slightly — the penalty term will
            # pull them onto the constraint surface in the first few iterations.
            # Resetting to script defaults forced the optimizer to re-discover
            # the basin from scratch, often hitting the timeout.
            _script_defaults_eq = {eq["kname"]: eq["logK"] for eq in parsed["equilibria"]}
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
                success, fitted_logKs, stats, message = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_spectra(
                        parsed, network, spectra_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, wl_min_fit, wl_max_fit,
                        tolerance, maxiter, auto_range=auto_range, timeout_s=_timeout_s, allow_negative_eps=allow_neg_eps,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end),
                    _start_logK, "Fitting parameters…", "_fit_requested")
                if auto_range and "opt_wl_min" in stats:
                    st.session_state["_pending_spectra_wl_min"] = stats["opt_wl_min"]
                    st.session_state["_pending_spectra_wl_max"] = stats["opt_wl_max"]
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "shift":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                success, fitted_logKs, stats, message = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_nmr_shifts(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end),
                    _start_logK, "Fitting parameters…", "_fit_requested")
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "integration":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                success, fitted_logKs, stats, message = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_nmr_integration(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end),
                    _start_logK, "Fitting parameters…", "_fit_requested")
            elif use_nmr_fit and nmr_cfg_fit["mode"] == "mixed":
                _timeout_s = float(st.session_state.get("fit_timeout", 30))
                success, fitted_logKs, stats, message = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_nmr_mixed(
                        parsed, network, nmr_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        timeout_s=_timeout_s,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end),
                    _start_logK, "Fitting parameters…", "_fit_requested")
            else:
                success, fitted_logKs, stats, message = _fit_async(_fit_slot, st.session_state.get("_shadow_fit_timeout", st.session_state.get("fit_timeout", 30)), lambda: fit_parameters(
                        parsed, network, exp_data_fit, params, _start_logK,
                        fit_keys_end, x_expr, tolerance, maxiter,
                        use_lbfgsb=_use_lbfgsb, use_neldermead=_use_neldermead,
                        constraints=_active_constraints,
                        fit_conc_keys=fit_conc_keys_end,
                        fit_titrant_keys=fit_titrant_keys_end),
                    _start_logK, "Fitting parameters…", "_fit_requested")

            for kname, fitted_val in fitted_logKs.items():
                if parsed.get("is_acid_base") and kname != "Kw":
                    st.session_state[f"_pending_pKa_{kname}"] = -float(fitted_val)
                else:
                    st.session_state[f"_pending_logK_{kname}"] = float(fitted_val)
            # Push fitted concentrations back to sidebar widgets
            for root, mM_val in stats.get("fitted_concs", {}).items():
                st.session_state[f"_pending_conc_{root}"] = float(mM_val)
            for tkey, mM_val in stats.get("fitted_titrants", {}).items():
                st.session_state[f"_pending_titrant_mM_{tkey}"] = float(mM_val)
            # ── v2: canonicalise param_values across ALL fit modes ──────
            # fit_conc populates stats["param_values"] internally, but
            # the NMR / spectra modules don't.  Mirror fitted_logKs into
            # stats["param_values"] so downstream diagnostics (Local
            # sensitivity test in particular) have a single source of
            # truth for the optimum, regardless of which fit module ran.
            if isinstance(stats, dict):
                stats.setdefault("param_values", dict(fitted_logKs))
            # Reset diagnostic toggles + cached results so the user opts
            # back in for the new fit (stale "show ..." plots were a
            # real-world confusion from earlier sessions).
            # Also bump the fit-counter — the parent diag-panel
            # checkbox uses this as a key suffix so Streamlit's
            # fragment-internal widget cache can't keep it stuck
            # checked across fits.
            st.session_state["_fit_counter"] = int(
                st.session_state.get("_fit_counter", 0)) + 1
            _reset_diagnostic_toggles()
            st.session_state["_fit_stats"] = stats
            # ── v2: save fit args so the bootstrap UI can find them after rerun ──
            st.session_state["_fit_args_eq"] = {
                "kind":              "eq",
                "fit_keys":          list(fit_keys_end),
                "fit_conc_keys":     list(fit_conc_keys_end),
                "fit_titrant_keys":  list(fit_titrant_keys_end),
                "use_spectra_fit":   bool(use_spectra_fit),
                "use_nmr_fit":       bool(use_nmr_fit),
                "nmr_mode":          (nmr_cfg_fit["mode"] if use_nmr_fit and nmr_cfg_fit else None),
                "wl_min":            (float(wl_min_fit) if use_spectra_fit else None),
                "wl_max":            (float(wl_max_fit) if use_spectra_fit else None),
                "tolerance":         float(tolerance),
                "maxiter":           int(maxiter),
                # The profile / bootstrap refits reuse the user's Timeout —
                # without this key they silently fell back to 30 s per refit.
                "timeout_s":         float(st.session_state.get("fit_timeout", 30)),
                "constraints":       _active_constraints,
                "start_logK":        dict(_start_logK),
                "start_logk":        dict(_start_logK),  # alias for profile dispatcher
                "x_expr":            x_expr,
                "exp_data_fit":      exp_data_fit     if not (use_nmr_fit or use_spectra_fit) else None,
                "nmr_data_fit":      nmr_data_fit     if use_nmr_fit     else None,
                "spectra_data_fit":  spectra_data_fit if use_spectra_fit else None,
                "allow_neg_eps":     bool(allow_neg_eps) if use_spectra_fit else False,
                "parsed_fit":        parsed,
                "network":           network,
                "params":            params,
                # n_data: needed by the local-sensitivity test to
                # turn f-values into conditional σ via δ/√(2·N·f).
                "n_data":            int(stats.get("n_points", 0)),
            }
            # ── Augment stats with AIC/BIC + residuals (v2 diagnostics) ──
            try:
                _final_logK = dict(_start_logK); _final_logK.update(fitted_logKs)
                # Unified collector reads y_obs/y_calc/residuals + per-col arrays
                # populated by all v2-patched fit modes — works uniformly for
                # conc, NMR-shift, NMR-integration, NMR-mixed, and spectra fits.
                # ``x_expr`` is passed so the residuals-vs-predictor axis label
                # matches the script's plot directive (e.g. "H0/G0").
                _res = collect_residuals_from_stats(stats, x_expr=x_expr_default)
                augment_stats(stats, _res)
                st.session_state["_fit_residuals"] = _res
            except Exception:
                st.session_state["_fit_residuals"] = {}
            _n_evals = stats.get("n_iter", 0)
            _n_fitted = len(fitted_logKs) + len(stats.get("fitted_concs", {})) + len(stats.get("fitted_titrants", {}))
            st.session_state["_fit_message"] = _fit_end_message(
                stats, success, message, _n_fitted, tolerance_log,
                float(st.session_state.get("fit_timeout", 30)))
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

    if _itc_placeholder:
        fig.add_annotation(
            text="📂 Upload an ITC file to compute the curve",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#888888"),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=350)
        st.plotly_chart(fig, use_container_width=True)
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
        # exp_series values are now 4-tuples: (exp_x, exp_y, col_name, orig_indices)
        # col_name + orig_indices let us look up exclusion state at render time.
        # For chained variables (pass 2) col_name=None, orig_indices=None.
        exp_series = {}   # {var_name: (exp_x, exp_y_transformed, col_name, orig_indices)}

        for col_name, col_vals in exp_data.items():
            if col_name.startswith("_"): continue
            matched_var = find_variable_for_exp_col(
                col_name, variables, plot_y_names, network["all_species"])
            _exp_hdr = exp_data.get("_x_col_header", "")
            exp_x = convert_exp_x(
                col_vals["v_add_mL"], x_expr, parsed, params, network,
                x_col_header=_exp_hdr)
            orig_indices = list(range(len(col_vals["y"])))
            if matched_var and matched_var in trace_colors:
                exp_y_transformed = transform_exp_via_variable(
                    matched_var, variables,
                    col_name, exp_x, col_vals["y"],
                    curve, x_vals, network)
                exp_series[matched_var] = (exp_x, exp_y_transformed, col_name, orig_indices)
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
                exp_series[display_name] = (exp_x, col_vals["y"], col_name, orig_indices)

        # ── Pass 2: propagate through chained variables ──────────────────
        dep_order = resolve_variable_dependencies(variables)
        for var_name in dep_order:
            if var_name in exp_series:
                continue
            if var_name not in plot_y_names:
                continue
            expr_deps = extract_identifiers_from_expression(variables[var_name])
            var_deps = [d for d in expr_deps if d in variables]
            if not var_deps:
                continue
            if not all(d in exp_series for d in var_deps):
                continue
            ref_x, _, _, _ = exp_series[var_deps[0]]
            n = len(ref_x)
            chained_y = np.full(n, np.nan)
            for i in range(n):
                var_vals = {}
                for dep in var_deps:
                    dep_x, dep_y, _, _ = exp_series[dep]
                    var_vals[dep] = float(np.interp(ref_x[i], dep_x, dep_y))
                species_vals = {sp: float(np.interp(ref_x[i], x_vals,
                                curve.get(sp, np.zeros_like(x_vals))))
                                for sp in network["all_species"]}
                chained_y[i] = evaluate_variable_expression(
                    variables[var_name], species_vals, var_vals)
            exp_series[var_name] = (ref_x, chained_y, None, None)

        # ── Render exp series — only for quantities listed in $plot y ───────
        _plot_y_set = set(plot_y_names) if plot_y_names else None
        _eq_main_excl = st.session_state.get("_outliers_main", {})
        for display_name, (exp_x, plot_y_vals, src_col, src_indices) in exp_series.items():
            if _plot_y_set is not None and display_name not in _plot_y_set:
                continue
            color = trace_colors.get(display_name, "#888888")
            # Chained / derived series: no per-point exclusion, render as before
            if src_col is None or src_indices is None:
                fig.add_trace(go.Scatter(
                    x=exp_x, y=plot_y_vals,
                    mode="markers", name=f"{display_name} (exp)",
                    marker=dict(color=color, size=7, symbol="circle",
                                line=dict(width=1, color="white")),
                    showlegend=True,
                ))
                continue
            _excl_set = _eq_main_excl.get(src_col, set())
            _inc = [j for j, oi in enumerate(src_indices) if oi not in _excl_set]
            _exc = [j for j, oi in enumerate(src_indices) if oi in _excl_set]
            if _inc:
                fig.add_trace(go.Scatter(
                    x=exp_x[_inc], y=np.asarray(plot_y_vals)[_inc],
                    mode="markers", name=f"{display_name} (exp)",
                    customdata=[[src_col, src_indices[j]] for j in _inc],
                    marker=dict(color=color, size=7, symbol="circle",
                                line=dict(width=1, color="white")),
                    showlegend=True,
                ))
            if _exc:
                fig.add_trace(go.Scatter(
                    x=exp_x[_exc], y=np.asarray(plot_y_vals)[_exc],
                    mode="markers", name=f"_outlier_{src_col}",
                    customdata=[[src_col, src_indices[j]] for j in _exc],
                    marker=dict(color=color, size=7, symbol="circle-open",
                                line=dict(width=1.5, color=color)),
                    showlegend=False,
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
            _nmr_hdr_main = nmr_data_main.get("_x_col_header", "")
            ref_v    = nmr_data_main[fitted_cols[0]]["v_add_mL"]
            exp_x_bc = convert_exp_x(ref_v, x_expr, parsed, params, network,
                                     x_col_header=_nmr_hdr_main)
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
                x_col       = convert_exp_x(nmr_data_main[col]["v_add_mL"], x_expr, parsed, params, network,
                                            x_col_header=_nmr_hdr_main)
                df0         = float(delta_f_main.get(col, nmr_data_main[col]["y"][0]))
                dobs_rel    = np.interp(exp_x_bc, x_col, nmr_data_main[col]["y"] - df0)

                # RHS[k,j] = (Δδ_obs_rel + ref_correction) × Ctot_theory
                rhs_mat[k, :] = (dobs_rel + ref_corr) * ctot_k

            # Solve M @ c_nonfree(j) = rhs(j) at each point
            conc_nonfree = np.zeros((n_nfree, n_pts))
            if n_nfree > 0:
                # the model's concentrations at the data points; the shifts correct
                # them only along the directions they resolve (shift_backcalc_resolved)
                c_model_nf = np.array([np.interp(exp_x_bc, x_vals, curve.get(sp, np.zeros_like(x_vals)))
                                       for sp in non_free_sp])
                sol, k_res = shift_backcalc_resolved(M_mat, rhs_mat, c_model_nf)
                conc_nonfree = np.clip(sol, 0.0, None)
                if k_res < n_nfree:
                    st.session_state["_shift_bc_note_eq"] = (
                        f"Shift-derived points: the {n_sig} shift signal{'s' if n_sig != 1 else ''} resolve "
                        f"{k_res} combination{'s' if k_res != 1 else ''} of the {n_nfree} species of the target "
                        f"({', '.join(non_free_sp)}); along the others the points follow the model.")

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
                                trace_colors, label_suffix=_nmr_suffix_shift,
                                excl_rows=_nmr_excl_intersection(nmr_data_main))

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
        # ── Post-fit: back-calc concentrations from raw integrals (no K needed).
        # The integration back-calc is K-independent — pre-fit and post-fit values
        # are identical. We therefore always render from the FULL unfiltered data
        # so original row indices are preserved in customdata, allowing hollow
        # markers to be toggled correctly after fitting.
        _nmr_suffix_integ = "(NMR/integration)" if nmr_cfg_integ.get("mode") == "mixed" else "(NMR)"
        _eqi_nh   = nmr_cfg_integ.get("n_H_list", [])
        _eqi_ni   = nmr_cfg_integ.get("n_integ", len(_eqi_nh))
        _eqi_cols = [c for c in nmr_data_integ if not c.startswith("_")][:_eqi_ni]
        _eqi_data = {c: nmr_data_integ[c] for c in _eqi_cols}
        _eqi_full = _nmr_integration_backCalc(
            _eqi_data, _eqi_nh[:_eqi_ni], params, network, x_expr, parsed,
            x_col_header=nmr_data_integ.get("_x_col_header", ""))
        _plot_backcalc_dots(fig, _eqi_full, plot_y_names,
                            parsed.get("variables", {}), network["all_species"],
                            trace_colors, label_suffix=_nmr_suffix_integ, symbol="circle-open",
                            excl_rows=_nmr_excl_intersection(nmr_data_integ))

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
            integ_data_pre, n_H_list_pre[:n_integ_pre], params, network, x_expr, parsed,
            x_col_header=nmr_data_integ.get("_x_col_header", ""))
        _bc_pre_pairs = bc_pre        # as returned: it carries its common grid for the plot
        _nmr_suffix_pre = "(NMR/integration)" if nmr_cfg_integ.get("mode") == "mixed" else "(NMR)"
        _plot_backcalc_dots(fig, _bc_pre_pairs, plot_y_names,
                            parsed.get("variables", {}), network["all_species"],
                            trace_colors, label_suffix=_nmr_suffix_pre, symbol="circle-open",
                            excl_rows=_nmr_excl_intersection(nmr_data_integ))

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
                # Recompute C_back from A using E_absorbed = E_final * path_cm so
                # concentrations are correct regardless of path length.
                _A_sp_bc   = _sd_bc.get("A", None)
                _wl_fit_sp = fit_stats_sp.get("wavelengths_fit", np.array([]))
                _path_sp   = float(fit_stats_sp.get("path_cm", 1.0))
                if _A_sp_bc is not None and len(_wl_fit_sp) > 0:
                    _wl_all_bc  = _sd_bc.get("wavelengths", np.array([]))
                    _wl_mask_bc = (_wl_all_bc >= _wl_fit_sp[0]) & (_wl_all_bc <= _wl_fit_sp[-1])
                    _A_fit_bc   = _A_sp_bc[:, _wl_mask_bc]
                    _E_absorbed = E_final_sp * max(_path_sp, 1e-12)
                    _Cb_raw, _, _, _ = np.linalg.lstsq(_E_absorbed.T, _A_fit_bc.T, rcond=None)
                    C_back_sp = np.clip(_Cb_raw.T, 0.0, None)
            except Exception:
                pass  # fall back to stored C_back
        if C_back_sp is not None and len(_x_raw_sp_bc) == C_back_sp.shape[0]:
            # x positions use current params so dots align with traces
            _sp_hdr_bc = _sd_bc.get("x_col_header", "")
            x_exp_sp = convert_exp_x(_x_raw_sp_bc, x_expr, parsed, params, network,
                                     x_col_header=_sp_hdr_bc)
            _bc_uvvis = {sp: (x_exp_sp, C_back_sp[:, j]) for j, sp in enumerate(absorbers_sp)}
            _plot_backcalc_dots(fig, _bc_uvvis, plot_y_names,
                                parsed.get("variables", {}), network["all_species"],
                                trace_colors, label_suffix="(UV-Vis)",
                                excl_rows=st.session_state.get("_outliers_spectra", set()),
                                bc_tag="__uvvis_bc__")

    # ── ITC back-calculated concentration dots on main plot ─────
    _itc_stats_mp = st.session_state.get("_fit_stats", {})
    _itc_raw_mp   = st.session_state.get("_itc_raw")
    if _itc_stats_mp.get("fit_mode") == "itc" and _itc_raw_mp is not None:
        _x_inj_mp  = _itc_stats_mp.get("x_inj", np.array([]))
        _C_back_mp = _itc_stats_mp.get("C_back", None)
        _proxy_sp  = _itc_stats_mp.get("rxn_proxy_sp", [])
        _itc_excl_mp = st.session_state.get("_outliers_itc", set())
        if _C_back_mp is not None and len(_x_inj_mp) > 0:
            for _j, _sp in enumerate(_proxy_sp):
                if _sp not in trace_colors:
                    continue
                _col_bc = trace_colors[_sp]
                _x_bc = [_x_inj_mp[i] for i in range(len(_x_inj_mp)) if i not in _itc_excl_mp]
                _y_bc = [float(_C_back_mp[i, _j]) for i in range(len(_x_inj_mp)) if i not in _itc_excl_mp]
                _x_excl = [_x_inj_mp[i] for i in range(len(_x_inj_mp)) if i in _itc_excl_mp]
                _y_excl = [float(_C_back_mp[i, _j]) for i in range(len(_x_inj_mp)) if i in _itc_excl_mp]
                if _x_bc:
                    fig.add_trace(go.Scatter(
                        x=_x_bc, y=_y_bc, mode="markers",
                        name=f"{_sp} (ITC)",
                        customdata=[[f"_itc_{_sp}", i] for i, _ in enumerate(_x_bc)],
                        marker=dict(color=_col_bc, size=7, symbol="circle",
                                    line=dict(width=1, color="white")),
                    ))
                if _x_excl:
                    fig.add_trace(go.Scatter(
                        x=_x_excl, y=_y_excl, mode="markers",
                        name=f"_outlier_itc_{_sp}", showlegend=False,
                        marker=dict(color=_col_bc, size=7, symbol="circle-open",
                                    line=dict(width=1.5, color=_col_bc)),
                    ))

    _eq_y_label   = _infer_y_label(plot_y_names, parsed, network)
    _eq_rangemode = None if _has_log_units(plot_y_names, parsed, network) else "tozero"
    _eq_yaxis = dict(title=_eq_y_label)
    if _eq_rangemode is None:
        _eq_yaxis["autorange"] = True   # log quantities: let Plotly range freely
    else:
        _eq_yaxis["rangemode"] = _eq_rangemode
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=40, b=80),
        xaxis=dict(title=x_label, range=[0, xmax]),
        yaxis=_eq_yaxis,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    _eq_ver_main = st.session_state.get("_outlier_ver__outliers_main", 0)
    fig.update_traces(
        unselected=dict(marker=dict(opacity=1.0)),
        selected=dict(marker=dict(opacity=1.0, size=10)),
        selector=dict(mode="markers"),
    )
    _eq_main_event = st.plotly_chart(
        fig, width='stretch',
        on_select="rerun", selection_mode="points",
        key=f"_eq_main_chart_{_eq_ver_main}",
    )
    if _process_outlier_event(_eq_main_event, "_outliers_main",
                               nmr_bc_cols=[c for c in st.session_state.get("_nmr_data", {})
                                            if not c.startswith("_")]):
        st.rerun()
    _outlier_bar("eq_main", "_outliers_main", "_outliers_nmr", "_outliers_spectra")
    if st.session_state.get("_shift_bc_note_eq"):
        st.caption(st.session_state.pop("_shift_bc_note_eq"))
    _pub_download_button(fig, "equilibrium_main", y_label=_eq_y_label)
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
            _sp_hdr_plot = _sd_plot.get("x_col_header", "")
            _x_sp = convert_exp_x(_x_raw_sp, x_expr, parsed, params, network,
                                  x_col_header=_sp_hdr_plot)
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
            _eq_sp_excl = st.session_state.get("_outliers_spectra", set())
            for _i in range(_n_sp):
                _col = _rainbow(_i, _n_sp)
                _lbl = f"{_x_unit}={_x_sp[_i]:.3g}"
                _is_excl = _i in _eq_sp_excl
                # Visual line — dashed+dim when excluded
                fig_sp.add_trace(go.Scatter(
                    x=_wl_plot,
                    y=_A_plot[_i],
                    mode="lines",
                    line=dict(color=_col, width=1.5,
                              dash="dash" if _is_excl else "solid"),
                    opacity=0.75 if _is_excl else 1.0,
                    name=f"_outlier_{_i}" if _is_excl else _lbl,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Full-line click surface: markers at every wavelength point, near-invisible.
                # This makes the entire line clickable, not just the peak.
                _n_wl_sp = len(_wl_plot)
                fig_sp.add_trace(go.Scatter(
                    x=_wl_plot, y=_A_plot[_i], mode="markers",
                    name=f"_outlier_{_i}" if _is_excl else f"_sp_sel_{_i}",
                    showlegend=False,
                    customdata=[[_i]] * _n_wl_sp,
                    marker=dict(
                        color=_col,
                        size=6,
                        opacity=0.01,   # near-invisible but still hit-testable by Plotly
                        symbol="circle-open" if _is_excl else "circle",
                        line=dict(width=0),
                    ),
                    hovertemplate=f"{'[excluded — click to restore] ' if _is_excl else '[click to exclude] '}{_lbl}<br>λ=%{{x:.0f}} nm<br>A=%{{y:.4f}}<extra></extra>",
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
                title=dict(text="UV-Vis spectra — click anywhere on a spectrum to exclude/restore", font=dict(size=13), x=0.5),
            )
            _eq_ver_sp = st.session_state.get("_outlier_ver__outliers_spectra", 0)
            # Suppress selection highlighting on marker traces only
            fig_sp.update_traces(
                unselected=dict(marker=dict(opacity=1.0)),
                selected=dict(marker=dict(opacity=1.0)),
                selector=dict(mode="markers"),
            )
            _eq_sp_event = st.plotly_chart(
                fig_sp, width='stretch',
                on_select="rerun", selection_mode="points",
                key=f"_eq_sp_chart_{_eq_ver_sp}",
            )
            if _process_outlier_event(_eq_sp_event, "_outliers_spectra", is_spectra=True):
                st.rerun()
            _outlier_bar("eq_sp", "_outliers_spectra")
            _pub_download_button(fig_sp, "equilibrium_spectra", x_label="Wavelength [nm]", y_label="Absorbance")

            # ── Pure species spectra (only shown after a successful fit) ──
            _fit_stats_sp = st.session_state.get("_fit_stats", {})
            if _fit_stats_sp.get("fit_mode") == "spectra":
                _E_final   = _fit_stats_sp.get("E_final")        # (n_absorbers, n_wl_fit)
                _wl_fit_sp = _fit_stats_sp.get("wavelengths_fit")  # (n_wl_fit,)
                _absorbers = _fit_stats_sp.get("absorbers", [])

                if _E_final is not None and len(_absorbers) > 0:
                    _path_disp = float(_fit_stats_sp.get("path_cm", 1.0))
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
                            hovertemplate=f"{_sp}<br>λ=%{{x:.0f}} nm<br>ε=%{{y:.4f}} mM⁻¹ cm⁻¹<extra></extra>",
                        ))

                    fig_pure.update_layout(
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=60),
                        xaxis=dict(title="Wavelength [nm]"),
                        yaxis=dict(title="ε [mM⁻¹ cm⁻¹]", rangemode="tozero"),
                        template="plotly_dark",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="left", x=0),
                        title=dict(text=f"Pure species spectra  (path = {_path_disp} cm)", font=dict(size=13), x=0.5),
                    )
                    st.plotly_chart(fig_pure, width='stretch')
                    _pub_download_button(fig_pure, "equilibrium_spectra_species", x_label="Wavelength [nm]", y_label="ε [mM⁻¹ cm⁻¹]")

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
        _nmr_hdr_plot = nmr_data.get("_x_col_header", "")
        shown_tgt_legend = set()
        _eq_nmr_excl = st.session_state.get("_outliers_nmr", {})
        for col in _shift_cols_plot:
            col_data = nmr_data[col]
            exp_x  = convert_exp_x(col_data["v_add_mL"], x_expr, parsed, params, network,
                                   x_col_header=_nmr_hdr_plot)
            df0    = float(delta_free.get(col, col_data["y"][0]))
            delta_obs_rel = col_data["y"] - df0
            tgt    = col_to_target.get(col, col)
            color  = nmr_target_colors.get(tgt, "#888888")
            show_in_legend = tgt not in shown_tgt_legend
            shown_tgt_legend.add(tgt)
            _excl_set_e = _eq_nmr_excl.get(col, set())
            _inc_e = [i for i in range(len(delta_obs_rel)) if i not in _excl_set_e]
            _exc_e = [i for i in range(len(delta_obs_rel)) if i in _excl_set_e]
            if _inc_e:
                fig_nmr.add_trace(go.Scatter(
                    x=exp_x[_inc_e], y=delta_obs_rel[_inc_e],
                    mode="markers",
                    name=tgt if show_in_legend else col,
                    legendgroup=tgt,
                    showlegend=show_in_legend,
                    customdata=[[col, i] for i in _inc_e],
                    marker=dict(color=color, size=6, symbol="circle"),
                ))
            if _exc_e:
                fig_nmr.add_trace(go.Scatter(
                    x=exp_x[_exc_e], y=delta_obs_rel[_exc_e],
                    mode="markers",
                    name=f"_outlier_{col}",
                    legendgroup=tgt,
                    showlegend=False,
                    customdata=[[col, i] for i in _exc_e],
                    marker=dict(color=color, size=6, symbol="circle-open",
                                line=dict(width=1.5, color=color)),
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
        _eq_ver_nmr = st.session_state.get("_outlier_ver__outliers_nmr", 0)
        fig_nmr.update_traces(
            unselected=dict(marker=dict(opacity=1.0)),
            selected=dict(marker=dict(opacity=1.0, size=9)),
            selector=dict(mode="markers"),
        )
        _eq_nmr_event = st.plotly_chart(
            fig_nmr, width='stretch',
            on_select="rerun", selection_mode="points",
            key=f"_eq_nmr_chart_{_eq_ver_nmr}",
        )
        if _process_outlier_event(_eq_nmr_event, "_outliers_nmr"):
            st.rerun()
        _outlier_bar("eq_nmr", "_outliers_nmr")
        _pub_download_button(fig_nmr, "equilibrium_nmr", y_label="Δδ [ppm]")


    # ── ITC Thermogram plot ───────────────────────────────────────
    _itc_raw_plot = st.session_state.get("_itc_raw")
    if _itc_raw_plot is not None:
        _itc_power = _itc_raw_plot.get("power_ucal_s")
        _itc_time  = _itc_raw_plot.get("time_s")
        _itc_q_exp_plot = st.session_state.get("_itc_q_exp", np.array([]))
        _itc_starts_plot = _itc_raw_plot.get("inj_start_idx", [])
        _itc_bl_plot     = st.session_state.get("_itc_baselines", np.zeros(len(_itc_starts_plot)))
        _itc_vols_plot   = _itc_raw_plot.get("inj_vols_uL", [])
        _itc_excl_plot   = st.session_state.get("_outliers_itc", set())
        _itc_stats_plot  = st.session_state.get("_fit_stats", {})

        if _itc_power is not None and len(_itc_time) > 0:
            fig_itc = go.Figure()

            # Raw power trace
            fig_itc.add_trace(go.Scatter(
                x=_itc_time, y=_itc_power, mode="lines",
                name="Power (µcal/s)",
                line=dict(color="#7c9eff", width=1.2),
                showlegend=True,
            ))

            # Injection markers (vertical lines)
            for _im, _si in enumerate(_itc_starts_plot):
                if _si < len(_itc_time):
                    _t_inj = float(_itc_time[_si])
                    fig_itc.add_vline(
                        x=_t_inj, line_width=1,
                        line_dash="dot", line_color="#555570",
                    )

            # Linear baseline segments per injection
            _itc_post_seg = st.session_state.get(
                "_itc_post_baselines", np.zeros(len(_itc_starts_plot)))
            for _im in range(len(_itc_starts_plot)):
                _s = _itc_starts_plot[_im]
                _e = (_itc_starts_plot[_im + 1] if _im + 1 < len(_itc_starts_plot)
                      else len(_itc_time))
                if _s < len(_itc_time) and _e <= len(_itc_time) and _e > _s:
                    _pre_v  = float(_itc_bl_plot[_im]) if _im < len(_itc_bl_plot) else 0.0
                    _post_v = float(_itc_post_seg[_im]) if _im < len(_itc_post_seg) else _pre_v
                    fig_itc.add_trace(go.Scatter(
                        x=[_itc_time[_s], _itc_time[_e - 1]],
                        y=[_pre_v, _post_v],
                        mode="lines",
                        line=dict(color="#fbbf24", width=1, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))

            # Integrated heats as dots on a secondary y-axis
            # Place dots at injection midpoints, x = time midpoint of window
            if len(_itc_q_exp_plot) > 0:
                _itc_dot_t = []
                _itc_dot_q = []
                _itc_dot_orig_idx = []  # original injection indices for customdata
                _itc_dot_q_excl = []
                _itc_dot_t_excl = []
                for _ii in range(len(_itc_q_exp_plot)):
                    # _ii=0 is the dummy injection (@1); real injections start at _ii=1 (@2+)
                    _si2 = _itc_starts_plot[_ii + 1] if (_ii + 1) < len(_itc_starts_plot) else None
                    _ei2 = (_itc_starts_plot[_ii + 2] if (_ii + 2) < len(_itc_starts_plot)
                            else len(_itc_time))
                    if _si2 is not None and _si2 < len(_itc_time):
                        _t_mid = float((_itc_time[_si2] + _itc_time[min(_ei2 - 1, len(_itc_time)-1)]) / 2)
                    else:
                        _t_mid = 0.0
                    _real_idx = _ii - 1  # index into real injections (q_exp[1:])
                    if _ii == 0 or _real_idx in _itc_excl_plot:
                        # dummy (_ii==0) always hollow; user-excluded real injections also hollow
                        _itc_dot_t_excl.append(_t_mid)
                        _itc_dot_q_excl.append(float(_itc_q_exp_plot[_ii]))
                    else:
                        _itc_dot_t.append(_t_mid)
                        _itc_dot_q.append(float(_itc_q_exp_plot[_ii]))
                        _itc_dot_orig_idx.append(_real_idx)  # index into real injections

                if _itc_dot_t:
                    fig_itc.add_trace(go.Scatter(
                        x=_itc_dot_t, y=_itc_dot_q, mode="markers",
                        name="q (µcal)", yaxis="y2",
                        customdata=[[_itc_dot_orig_idx[i]] for i in range(len(_itc_dot_t))],
                        marker=dict(color="#00CC96", size=8, symbol="circle",
                                    line=dict(width=1, color="white")),
                        hovertemplate="injection %{customdata[0]}<br>q = %{y:.3f} µcal<extra></extra>",
                    ))
                if _itc_dot_t_excl:
                    fig_itc.add_trace(go.Scatter(
                        x=_itc_dot_t_excl, y=_itc_dot_q_excl, mode="markers",
                        name="_itc_excl", yaxis="y2", showlegend=False,
                        marker=dict(color="#00CC96", size=8, symbol="circle-open",
                                    line=dict(width=1.5, color="#00CC96")),
                    ))

                # Theoretical q(i) line after fitting
                if _itc_stats_plot.get("fit_mode") == "itc":
                    _q_theo_plot = _itc_stats_plot.get("q_theo", np.array([]))
                    _x_inj_t    = _itc_stats_plot.get("x_inj", np.array([]))
                    if len(_q_theo_plot) > 0 and len(_x_inj_t) > 0:
                        # Re-use same time midpoints for non-excluded injections
                        _all_t_mids = []
                        for _ii in range(1, len(_itc_q_exp_plot)):  # skip dummy (_ii=0)
                            _si2 = (_itc_starts_plot[_ii + 1]
                                    if (_ii + 1) < len(_itc_starts_plot) else None)
                            _ei2 = (_itc_starts_plot[_ii + 2]
                                    if (_ii + 2) < len(_itc_starts_plot) else len(_itc_time))
                            if _si2 is not None and _si2 < len(_itc_time):
                                _all_t_mids.append(float(
                                    (_itc_time[_si2] + _itc_time[min(_ei2 - 1, len(_itc_time)-1)]) / 2))
                            else:
                                _all_t_mids.append(0.0)
                        # _all_t_mids[j] = time of real injection j (0-indexed)
                        _incl_idx = [i for i in range(len(_itc_q_exp_plot) - 1) if i not in _itc_excl_plot]
                        _theo_t   = [_all_t_mids[i] for i in _incl_idx if i < len(_all_t_mids)]
                        _theo_q   = [float(_q_theo_plot[j]) for j, i in enumerate(_incl_idx)
                                     if j < len(_q_theo_plot)]
                        if _theo_t:
                            fig_itc.add_trace(go.Scatter(
                                x=_theo_t, y=_theo_q, mode="lines+markers",
                                name="q (calc)", yaxis="y2",
                                line=dict(color="#EF553B", width=2),
                                marker=dict(color="#EF553B", size=5),
                            ))

            fig_itc.update_layout(
                height=380,
                margin=dict(l=40, r=60, t=40, b=60),
                xaxis=dict(title="Time [s]"),
                yaxis=dict(title="Power [µcal/s]"),
                yaxis2=dict(title="q [µcal]", overlaying="y", side="right",
                            showgrid=False),
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.15,
                            xanchor="center", x=0.5),
                title=dict(text="ITC Thermogram — click an injection dot to exclude/restore",
                           font=dict(size=13), x=0.5),
            )
            _itc_ver_plot = st.session_state.get("_outlier_ver__outliers_itc", 0)
            _itc_event = st.plotly_chart(
                fig_itc, width="stretch",
                on_select="rerun", selection_mode="points",
                key=f"_itc_chart_{_itc_ver_plot}",
            )
            # Outlier toggle for ITC injection dots
            if _itc_event and _itc_event.get("selection", {}).get("points"):
                for _pt in _itc_event["selection"]["points"]:
                    _cd = _pt.get("customdata")
                    if _cd and len(_cd) >= 1:
                        _inj_i = int(_cd[0])
                        _s_itc = st.session_state.setdefault("_outliers_itc", set())
                        _s_itc.discard(_inj_i) if _inj_i in _s_itc else _s_itc.add(_inj_i)
                        _bump_outlier_ver("_outliers_itc")
                        # Re-integrate with current baselines
                        _bl_now = st.session_state.get("_itc_baselines",
                                  np.zeros(len(_itc_starts_plot)))
                        _post_now = st.session_state.get("_itc_post_baselines",
                                             _bl_now)
                        st.session_state["_itc_q_exp"] = integrate_with_baselines(
                            _itc_raw_plot, _bl_now, _post_now)
                        st.rerun()

            # Outlier bar
            _itc_n_excl = len(st.session_state.get("_outliers_itc", set()))
            if _itc_n_excl == 0:
                st.caption("🖱️ Click an injection dot to flag it as an outlier and exclude it from fitting")
            else:
                _col_obar1, _col_obar2 = st.columns([4, 1])
                with _col_obar1:
                    st.caption(f"⚠️ {_itc_n_excl} injection(s) excluded from fitting "
                               "— click a hollow dot to restore it")
                with _col_obar2:
                    if st.button("✕ Clear all", key="_itc_clear_outliers"):
                        st.session_state.pop("_outliers_itc", None)
                        _bump_outlier_ver("_outliers_itc")
                        st.rerun()

            # ── Baseline adjustment UI ────────────────────────────
            with st.expander("⚙️ Adjust per-injection baselines", expanded=False):
                st.caption("**Pre** = power level before injection; "
                           "**Post** = power level after peak recovery. "
                           "A linear baseline is drawn between them. "
                           "Adjust and click **Re-integrate**.")
                _bl_edit      = list(_itc_bl_plot)
                _itc_post_plot = st.session_state.get(
                    "_itc_post_baselines",
                    np.zeros(len(_itc_starts_plot)))
                _bl_post_edit = list(_itc_post_plot)
                _n_inj_ui = len(_itc_starts_plot)

                # Header row
                _hc = st.columns([1, 3, 3])
                _hc[0].caption("**Inj**")
                _hc[1].caption("**Pre (µcal/s)**")
                _hc[2].caption("**Post (µcal/s)**")

                for _im in range(1, _n_inj_ui):  # skip dummy (index 0)
                    _rc = st.columns([1, 3, 3])
                    _rc[0].caption(f"#{_im}")
                    with _rc[1]:
                        _pre_new = st.number_input(
                            f"pre_{_im}", label_visibility="collapsed",
                            value=float(_bl_edit[_im]) if _im < len(_bl_edit) else 0.0,
                            step=0.001, format="%.4f",
                            key=f"_itc_bl_pre_{_im}",
                        )
                        _bl_edit[_im] = _pre_new
                    with _rc[2]:
                        _post_new = st.number_input(
                            f"post_{_im}", label_visibility="collapsed",
                            value=float(_bl_post_edit[_im]) if _im < len(_bl_post_edit) else 0.0,
                            step=0.001, format="%.4f",
                            key=f"_itc_bl_post_{_im}",
                        )
                        _bl_post_edit[_im] = _post_new

                if st.button("Re-integrate", key="_itc_reintegrate"):
                    _bl_arr      = np.array(_bl_edit)
                    _post_arr    = np.array(_bl_post_edit)
                    st.session_state["_itc_baselines"]      = _bl_arr
                    st.session_state["_itc_post_baselines"] = _post_arr
                    st.session_state["_itc_q_exp"] = integrate_with_baselines(
                        _itc_raw_plot, _bl_arr, _post_arr)
                    # Clear old fit stats since data changed
                    st.session_state.pop("_fit_stats", None)
                    st.rerun()

            # ITC publication figure: custom two-panel layout
            try:
                from datetime import datetime as _dt2
                _itc_stats_pub = st.session_state.get("_fit_stats", {})
                _itc_pre_pub   = st.session_state.get("_itc_baselines",
                                   np.zeros(len(_itc_starts_plot)))
                _itc_post_pub  = st.session_state.get("_itc_post_baselines",
                                   _itc_pre_pub)
                _itc_eq_pub    = st.session_state.get("_itc_eq_times",
                                   np.zeros(len(_itc_q_exp_plot)))

                # x_inj_all: molar-ratio values for ALL injections (len = n_q)
                # x_inj: only included injections — wrong length, use x_inj_all
                _itc_x_pub = _itc_stats_pub.get("x_inj_all")
                if _itc_x_pub is None:
                    _itc_x_pub = _itc_stats_pub.get("x_inj")  # fallback

                # dH per mol for each injection (from fit stats)
                _itc_dH_pub   = _itc_stats_pub.get("dH_per_inj")
                _itc_qcalc    = _itc_stats_pub.get("q_theo")

                # Fitted curve for the isotherm
                _itc_curve_x  = _itc_stats_pub.get("curve_x")
                _itc_curve_y  = _itc_stats_pub.get("curve_dH")

                # Compute dH_norm from real injections (q_exp[1:], skip dummy at [0])
                # so it aligns with x_inj_all (real injections only).
                _itc_vols_pub = np.array(_itc_raw_plot.get("inj_vols_uL", []))
                _itc_Hst_pub  = float(_itc_raw_plot.get("syringe_conc_mM", 5.0))
                _itc_dH_data_pub = np.array([
                    float(_itc_q_exp_plot[i+1]) /
                    max(_itc_Hst_pub * _itc_vols_pub[i+1]/1000.0, 1e-12) / 1000.0
                    if (i+1) < len(_itc_vols_pub) and (i+1) < len(_itc_q_exp_plot) else float("nan")
                    for i in range(len(_itc_q_exp_plot) - 1)
                ])
                _itc_dH_theo_pub  = _itc_stats_pub.get("dH_theo_norm")     # kcal/mol fit

                _itc_pub_bytes = _itc_pub_figure_bytes(
                    raw=_itc_raw_plot,
                    q_exp=_itc_q_exp_plot[1:],  # skip dummy
                    pre_baselines=_itc_pre_pub,
                    post_baselines=_itc_post_pub,
                    eq_times_s=_itc_eq_pub,
                    x_vals=_itc_x_pub,
                    q_calc=_itc_qcalc,
                    excl_set=_itc_excl_plot,
                    dH_per_mol=_itc_dH_data_pub,
                    dH_calc_curve_x=_itc_x_pub,
                    dH_calc_curve_y=_itc_dH_theo_pub,
                    x_label=x_expr,
                )
                _ts2 = _dt2.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📐 Publication figure (pdf)",
                    data=_itc_pub_bytes,
                    file_name=f"itc_pub_{_ts2}.pdf",
                    mime="application/pdf",
                    key="_pubfig_itc",
                )
            except Exception as _epub:
                st.caption(f"Export failed: {_epub}")

    # ── Custom Python script: run on request, show the latest output ──────
    def _py_context():
        return pyscript.EquilibristContext(
            parsed=parsed, network=network, params=params, logK=logK_vals,
            curve=curve, x=x_vals, x_label=x_label, xmax=float(xmax),
            data={"nmr": st.session_state.get("_nmr_data", {}) or {},
                  "conc": st.session_state.get("_exp_data", {}) or {},
                  "spectra": st.session_state.get("_spectra_data", {}) or {}},
            fit_stats=st.session_state.get("_fit_stats", {}) or {}, st_module=st,
            main_fig=fig)
    pyscript.render_output(_py_context)


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

    # Get which parameters are marked for fitting (deduplicated for shared knames)
    fit_keys = list(dict.fromkeys(
        eq["kname"] for eq in parsed["equilibria"]
        if st.session_state.get(f"fit_logK_{eq['kname']}", False)
    ))

    # Render Tol/Timeout BEFORE the button so they are always rendered
    # before any st.rerun() call — prevents Streamlit from garbage-collecting
    # their session state when the button aborts the run mid-script.
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        fit_tolerance_log = _num_input(
            "Tol (−log)", key="fit_tolerance_log", default=3.0, step=0.5, format="%.1f",
            help="Convergence target for the constants, in log K units: the fit is "
                 "declared converged when the search has narrowed to 10^−Tol in every "
                 "constant.  The errors on constants are typically 0.01–0.1, so 3–4 is "
                 "usually enough and much faster; 6 is very tight and is rarely reached "
                 "before the Timeout.  The message after each fit says how it ended."
        )
    with _fc2:
        fit_timeout = _num_input(
            "Timeout (s)", key="fit_timeout", default=30, step=5, format="%d",
            help="Wall-clock limit for one fit.  When it expires the best point found "
                 "so far is kept and the message says the fit was not converged."
        )
    fit_refresh = _num_input(
        "Refresh (s)", key="fit_refresh", default=60, step=5, format="%d",
        help="While a fit runs, redraw data and model at the best constants found so far "
             "every this many seconds, under the progress bar (0 = off).  Each redraw costs "
             "one model evaluation.  NMR fits and the global fit for now."
    )

    _fit_conc_keys_btn = [root for root in
                          (cname[:-1] if cname.endswith("0") else cname
                           for cname in parsed["concentrations"])
                          if st.session_state.get(f"fit_conc_{root}", False)]
    _fit_titrant_keys_btn = [tkey for tkey in titrant_keys
                             if st.session_state.get(f"fit_titrant_mM_{tkey}", False)]
    _has_itc_data_btn = (st.session_state.get("_itc_raw") is not None and
                          len(st.session_state.get("_itc_q_exp", [])) > 0)
    fit_enabled = (has_exp_data or _has_itc_data_btn) and (len(fit_keys) > 0 or
                                    len(_fit_conc_keys_btn) > 0 or
                                    len(_fit_titrant_keys_btn) > 0)
    _g_why = []
    _g_ready = eqtabs.collect_experiments(st.session_state, _g_why)
    _multi = len(st.session_state.get("_tabs") or []) > 1
    if _multi:
        st.session_state.setdefault("_global_weighted", True)
        st.checkbox("Weight tabs by their own fit", key="_global_weighted",
                    help="Each tab's objective is rescaled to its own misfit (re-estimated after each "
                         "optimiser pass), so a titration the model fits worse cannot outvote a clean "
                         "one.  Untick for a plain sum of the tabs' objectives.")
        _share_shifts_checkbox(_g_ready)
    # NOT disabled=not fit_enabled. The fit checkboxes live in fragments, so
    # ticking one does not rerun the main script - fit_enabled would still
    # hold its previous value and the button would stay greyed out, with no
    # way to press it to trigger the rerun that would enable it. Clicking a
    # button always reruns the whole app, so fit_enabled is recomputed from
    # session_state at that moment and the guard below is authoritative.
    _fb1, _fb2 = st.columns(2)
    with _fb1:
        if st.button("🔧 Fit Parameters", width="stretch"):
            if fit_enabled:
                # Snapshot user-set fit prefs into shadow keys so st.rerun()
                # cannot lose them via Streamlit's widget-state cleanup
                for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                    if _pref in st.session_state:
                        st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                st.session_state["_fit_requested"] = True
                st.rerun()
            else:
                st.info("Tick at least one 'fit' box (K, concentration, titrant "
                        "or q_offset) and load experimental data first.")
    with _fb2:
        if _multi and st.button("🌐 Global fit (all tabs)", width="stretch",
                                help="One fit across every tab that has data and a ticked constant.  "
                                     "Constants with the same name share one value; concentrations stay "
                                     "per tab; each tab's bounds and constraints apply."):
            if len(_g_ready) >= 2 and not _use_itc_fit:
                for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                    if _pref in st.session_state:
                        st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
                st.session_state["_global_fit_requested"] = True
                st.rerun()
            else:
                st.info("The Global fit needs at least two tabs with data loaded and a 'fit' "
                        "box ticked.  " + ("  ".join(_g_why) if _g_why else ""))
    if _multi:
        st.caption("Tabs in the global fit: " + (", ".join(_lab for _i, _lab, _fa in _g_ready) or "none yet")
                   + ((".  Left out — " + "; ".join(_g_why)) if _g_why else ""))
    _hpc_go = hpc_ui.controls("hpc_eq", label="the fit",
                              submits=[("fit", "this tab's fit")] + ([("global", "the global fit")] if _multi else []))
    if _hpc_go == "global":
        if len(_g_ready) >= 2 and not _use_itc_fit:
            for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                if _pref in st.session_state:
                    st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
            st.session_state["_global_fit_hpc"] = "hpc_eq"
            st.session_state["_global_fit_requested"] = True
            st.rerun()
        else:
            st.info("The Global fit needs at least two tabs with data loaded and a 'fit' "
                    "box ticked.  " + ("  ".join(_g_why) if _g_why else ""))
    elif _hpc_go == "fit":
        if fit_enabled:
            for _pref in ("fit_tolerance_log", "fit_timeout", "fit_refresh"):
                if _pref in st.session_state:
                    st.session_state[f"_shadow_{_pref}"] = st.session_state[_pref]
            st.session_state["_single_fit_hpc"] = "hpc_eq"
            st.session_state["_fit_requested"] = True
            st.rerun()
        else:
            st.info("Tick at least one 'fit' box (K, concentration, titrant "
                    "or q_offset) and load experimental data first.")
    hpc_ui.jobs_panel("hpc_eq", on_result=_hpc_store_fit)
    if _multi:
        if st.button("📤 Propagate ticked constants", key="_propagate_btn",
                     help="Copy the constants ticked here, at the values in their boxes, into every "
                          "other tab whose script has a constant of the same name.  Only the values "
                          "move; the other tabs' ticks, bounds, data and scripts stay as they are."):
            st.info(eqtabs.propagate_ticked(st.session_state, parse_script))

    # ── Export Data Button ──────────────────────
    st.subheader("Data Export")
    col_export, col_snapshot = st.columns(2)

    with col_export:
        try:
            script_text = st.session_state.get("_script_text", "")
            excel_data, filename = export_to_excel(curve, x_vals, parsed, params, network, script_text, logK_vals,
                                                    script_path=st.session_state.get("_script_filename"),
                                                    input_path=st.session_state.get("_input_filename"),
                                                    fit_stats=st.session_state.get("_fit_stats", {}))
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

    # ── ITC: the experimental-data uploader below also accepts .itc files ──
    if "_itc_uploader_nonce" not in st.session_state:
        st.session_state["_itc_uploader_nonce"] = 0
    if not IS_KINETICS and _itc_placeholder:
        st.info("📂 Upload an ITC file below — concentrations and volumes will be set "
                "automatically from the file header.")

    def _handle_itc_upload(_file):
        """Read a MicroCal .itc file dropped on the experimental-data uploader."""
        try:
            _itc_content = _file.read()
            _itc_raw_new = parse_itc_file(_itc_content)
            _itc_q_new, _itc_pre, _itc_post, _itc_eq = auto_integrate(_itc_raw_new)
            # auto_integrate returns q_ucal (dummy at [0]) and pre_bl[1:]
            # (n_total-1 values). Prepend the pre-eq window value so
            # _itc_pre_full has n_total elements — needed for the baseline
            # editor which loops over range(1, len(starts)).
            _itc_pre_full  = np.concatenate([[_itc_pre[0]  if len(_itc_pre)  > 0 else 0.0], _itc_pre])
            _itc_post_full = np.concatenate([[_itc_post[0] if len(_itc_post) > 0 else 0.0], _itc_post])
            st.session_state["_itc_raw"]            = _itc_raw_new
            st.session_state["_itc_q_exp"]          = _itc_q_new
            st.session_state["_itc_baselines"]      = _itc_pre_full   # pre (for thermogram display)
            st.session_state["_itc_post_baselines"] = _itc_post_full
            st.session_state["_itc_eq_times"]       = _itc_eq
            st.session_state.pop("_outliers_itc", None)
            st.session_state.pop("_fit_stats", None)
            # Pre-fill concentrations from file header
            _cell_mM = _itc_raw_new.get("cell_conc_mM")
            _syr_mM  = _itc_raw_new.get("syringe_conc_mM")
            _temp_c  = _itc_raw_new.get("temperature_C", 25.0)
            # Write cell conc to first $concentrations entry
            _first_cname = (list(parsed.get("concentrations", {}).keys()) or [None])[0]
            if _first_cname is not None and _cell_mM is not None:
                _first_root = _first_cname[:-1] if _first_cname.endswith("0") else _first_cname
                st.session_state[f"_pending_conc_{_first_root}"] = float(_cell_mM)
            # Write syringe conc to first $titrant entry
            _first_tkey = (list(parsed.get("titrant", {}).keys()) or [None])[0]
            if _first_tkey is not None and _syr_mM is not None:
                st.session_state[f"_pending_titrant_mM_{_first_tkey}"] = float(_syr_mM)
            # Auto-set xmax to last injection equivalents (approx)
            if _cell_mM and _syr_mM and _cell_mM > 0:
                _n_real = max(0, len(_itc_q_new) - 1)  # exclude dummy
                _total_vol_mL = sum(v / 1000.0 for v in _itc_raw_new["inj_vols_uL"][1:])
                _approx_equiv = (_syr_mM * _total_vol_mL) / (
                    _cell_mM * float(_itc_raw_new.get("cell_vol_mL", ITC200_CELL_VOLUME_ML)))
                st.session_state["_pending_xmax"] = float(
                    np.ceil(_approx_equiv * 10) / 10)
            st.session_state["_itc_filename"] = _file.name
            st.session_state["_exp_uploader_nonce"] += 1
            st.rerun()
        except Exception as _itc_e:
            st.error(f"Could not read ITC file: {_itc_e}")


    if st.button("↺ Reset experimental data"):
        for k in list(st.session_state.keys()):
            if (k.startswith("_exp") or k.startswith("_nmr") or
                    k.startswith("_spectra") or k.startswith("_fit") or
                    k.startswith("_itc_raw") or k.startswith("_itc_q_exp") or
                    k.startswith("_itc_baselines") or k.startswith("_itc_post_baselines") or
                    k.startswith("_itc_eq_times") or k.startswith("_itc_filename") or
                    k.startswith("_outliers_itc") or
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
    if _is_solid_up:
        # Build a human-readable list of accepted col-A headers
        _tit_keys_up  = parsed.get("titrant", {}).keys()
        _tit_x0_up    = [k[:-1] + "0" if k.endswith("t") else k + "0" for k in _tit_keys_up]
        _conc_keys_up = list(parsed.get("concentrations", {}).keys())
        _ratio_ex = f"{_tit_x0_up[0]}/{_conc_keys_up[0]}" if _tit_x0_up and _conc_keys_up else "H0/G0"
        _col_a_hint = (
            f"column A header = titrant conc (e.g. **{_tit_x0_up[0] if _tit_x0_up else 'H0'}** [mM]) "
            f"or equivalency ratio (e.g. **{_ratio_ex}**)"
        )
    else:
        _col_a_hint = "column A = volume added (mL)"

    if parsed.get("spectra") is not None:
        _uploader_label = "Upload experimental data (.xlsx, or a MicroCal .itc file)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; row 1 = wavelengths (nm); body = absorbance"
    elif nmr_cfg is not None:
        if nmr_cfg["mode"] == "integration":
            st.caption(f"NMR mode: **integration** (slow exchange) — "
                       f"{len(nmr_cfg.get('n_H_list', []))} signals")
        else:
            st.caption(f"NMR mode: **{nmr_cfg['mode']}** — targets: {', '.join(nmr_cfg['targets'])}")
        _mm = _pinned_shift_mismatch(nmr_cfg, st.session_state.get("_nmr_data"))
        if _mm:
            _sp_mm, _off_mm, _n_mm, _same_sign = _mm
            st.warning(f"The shifts pinned for **{_sp_mm}** (sheet 2) differ from the shifts observed at the "
                       f"first titration point by {_off_mm:+.3f} ppm (median over {_n_mm} columns"
                       f"{', same sign on every column' if _same_sign else ''}), i.e. the data and the "
                       "pinned sheet are not referenced alike.  The fit works with shifts relative to the "
                       "first point, so the result may not change, but the pinned values are then off by "
                       "that amount: re-reference the data or the sheet.")
        _uploader_label = "Upload experimental data (.xlsx, or a MicroCal .itc file)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; columns B+ = NMR observables"
    else:
        _uploader_label = "Upload experimental data (.xlsx, or a MicroCal .itc file)"
        _uploader_hint  = f"{'🪨' if _is_solid_up else '💧'} {_col_a_hint}; columns B+ = species concentrations (mM)"

    if not IS_KINETICS:
        _uploader_hint += " — or drop a MicroCal .itc thermogram for ITC mode"
    st.caption(_uploader_hint)
    _uploaded = st.file_uploader(
        _uploader_label, type=(["xlsx"] if IS_KINETICS else ["xlsx", "itc"]),
        key=f"_exp_uploader_{st.session_state['_exp_uploader_nonce']}",
    )
    if _uploaded is not None and _uploaded.name.lower().endswith(".itc"):
        _handle_itc_upload(_uploaded)           # routes by file type
    elif _uploaded is not None:
        try:
            _raw_bytes = _uploaded.read()
            # Capture raw bytes + filename in session_state so the
            # "Save session" feature can bundle the data verbatim.
            st.session_state["_exp_data_bytes"]    = _raw_bytes
            st.session_state["_exp_data_filename"] = _uploaded.name
            if parsed.get("spectra") is not None:
                _loaded = load_spectra_data(_raw_bytes)
                st.session_state["_spectra_data"] = _loaded
                st.session_state.pop("_outliers_spectra", None)
                # Override xmax with the x-value of the last data point
                if _loaded:
                    _x_last_mL = float(_loaded["x_vals"][-1])
                    _x_last    = convert_exp_x(
                        np.array([_x_last_mL]), x_expr_default,
                        parsed, params, network,
                        x_col_header=_loaded.get("x_col_header", ""))[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last * 10) / 10)
            elif nmr_cfg is not None:
                _loaded = load_experimental_data(_raw_bytes)
                st.session_state["_nmr_data"] = _loaded
                st.session_state.pop("_outliers_nmr", None)
                # Update xmax from last data point (x-axis in mL, converted via x_expr)
                _eq_nmr_cols = [c for c in _loaded if not c.startswith("_")]
                if _eq_nmr_cols:
                    _x_last_mL_nmr = float(_loaded[_eq_nmr_cols[0]]["v_add_mL"][-1])
                    _x_last_nmr = convert_exp_x(
                        np.array([_x_last_mL_nmr]), x_expr_default,
                        parsed, params, network,
                        x_col_header=_loaded.get("_x_col_header", ""))[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_nmr * 10) / 10)
            else:
                _loaded = load_experimental_data(_raw_bytes)
                st.session_state["_exp_data"] = _loaded
                st.session_state.pop("_outliers_main", None)
                _eq_cols = [c for c in _loaded if not c.startswith("_")]
                if _eq_cols:
                    _x_last_mL_eq = float(_loaded[_eq_cols[0]]["v_add_mL"][-1])
                    _x_last_eq = convert_exp_x(
                        np.array([_x_last_mL_eq]), x_expr_default,
                        parsed, params, network,
                        x_col_header=_loaded.get("_x_col_header", ""))[0]
                    st.session_state["_pending_xmax"] = float(np.ceil(_x_last_eq * 10) / 10)
            st.session_state["_input_filename"] = _uploaded.name
            st.session_state["_exp_uploader_nonce"] += 1
            st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.get("_input_filename"):
        st.markdown(f"<span style='background:#1a6bbf;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:0.82rem'>📄 {st.session_state['_input_filename']}</span>", unsafe_allow_html=True)
    if not IS_KINETICS:
        _itc_loaded = st.session_state.get("_itc_raw")
        if _itc_loaded is not None:
            _itc_fname = st.session_state.get("_itc_filename", "")
            st.markdown(f"<span style='background:#2d6a4f;color:white;font-weight:bold;"
                        f"padding:3px 10px;border-radius:12px;font-size:0.82rem'>"
                        f"📡 {_itc_fname}</span>", unsafe_allow_html=True)
            _itc_n_inj = max(0, len(st.session_state.get("_itc_q_exp", [])) - 1)  # exclude dummy
            _itc_temp  = _itc_loaded.get("temperature_C", 25.0)
            st.caption(f"Loaded: {_itc_n_inj} injections · {_itc_temp:.1f} °C · "
                       f"cell {_itc_loaded.get('cell_conc_mM', '?'):.4f} mM · "
                       f"syringe {_itc_loaded.get('syringe_conc_mM', '?'):.4f} mM")
            if st.button("↺ Clear ITC data", key="_itc_clear"):
                for _k in ["_itc_raw","_itc_q_exp","_itc_baselines","_itc_filename",
                           "_outliers_itc","_fit_stats"]:
                    st.session_state.pop(_k, None)
                st.rerun()

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

        # ── Warnings stay OUTSIDE the collapsible (always visible) ────
        if fit_mode_disp == "itc":
            # ── ITC statistics ──────────────────────────────────
            if fit_stats.get("timed_out"):
                st.warning("⏱️ **ITC fit timed out** — showing best result found.")
            st.write(f"• R² = {fit_stats.get('r_squared', 0):.4f}")
            st.write(f"• RMSE = {fit_stats.get('rmse', 0):.3f} µcal")
            st.write(f"• Data points: {fit_stats.get('n_points', 0)} injections")
            if fit_stats.get("n_points", 0) > fit_stats.get("n_params", 0):
                _rchi2_itc = fit_stats["ssr"] / (fit_stats["n_points"] - fit_stats["n_params"])
                st.write(f"• Reduced χ² = {_rchi2_itc:.3e}")
            st.write(f"• q_offset = {fit_stats.get('q_offset', 0):.4f} µcal")
            if fit_stats.get("n_iter"):
                st.write(f"• Evaluations: {fit_stats['n_iter']}")

            # ── Fitted constants table ───────────────────────────────
            _dH_vals = fit_stats.get("dH_values", {})
            _dH_errs = fit_stats.get("dH_errors", {})
            _pv_itc  = fit_stats.get("param_values", {})
            _pe_itc  = fit_stats.get("param_errors", {})
            if _pv_itc or _dH_vals:
                st.write("**Fitted constants:**")
                _rows_itc = []
                for kn, val in _pv_itc.items():
                    err = _pe_itc.get(kn)
                    k_lin   = 10.0 ** val
                    err_lin = k_lin * 2.302585 * err if err is not None else None
                    _rows_itc.append({
                        "Parameter": kn,
                        "log P":  f"{val:.2f}",
                        "±log P": f"± {err:.2f}" if err is not None else "n/a",
                        "P":      f"{k_lin:.2e}",
                        "±P":     f"± {err_lin:.2e}" if err_lin is not None else "n/a",
                    })
                # ΔH omitted here — shown in thermodynamics table below
                for root, mM in fit_stats.get("fitted_concs", {}).items():
                    err = _pe_itc.get(root)
                    _rows_itc.append({
                        "Parameter": root,
                        "log P": "—", "±log P": "—",
                        "P": f"{mM:.2e} mM",
                        "±P": f"± {err:.2e} mM" if err is not None else "n/a",
                    })
                if _rows_itc:
                    st.dataframe(pd.DataFrame(_rows_itc).set_index("Parameter"),
                                 width="stretch")

            # ── Thermodynamics table ─────────────────────────────────────
            if _pv_itc and _dH_vals:
                import math as _math
                _T_itc = float(parsed.get("temperature_K", 298.15))
                _R_itc = 1.987e-3  # kcal/(mol·K)
                _thermo_rows = []
                for kn, val in _pv_itc.items():
                    _err_K  = _pe_itc.get(kn)
                    _dH     = _dH_vals.get(kn, float("nan"))
                    _dH_err = _dH_errs.get(kn)
                    _dG     = -_R_itc * _T_itc * _math.log(10) * val
                    _dG_err = _R_itc * _T_itc * _math.log(10) * _err_K if _err_K is not None else None
                    _dS     = (_dH - _dG) / _T_itc if _T_itc > 0 else float("nan")
                    # error on ΔS = sqrt(σ_ΔH² + σ_ΔG²) / T
                    _dS_err = (_math.sqrt(_dH_err**2 + _dG_err**2) / _T_itc
                               if (_dH_err is not None and _dG_err is not None) else None)
                    _TdS     = _T_itc * _dS
                    _TdS_err = _T_itc * _dS_err if _dS_err is not None else None
                    _thermo_rows.append({
                        "Reaction": kn,
                        "ΔG°":  f"{_dG:.2f}",
                        "±ΔG°": f"± {_dG_err:.2f}" if _dG_err is not None else "n/a",
                        "ΔH°":  f"{_dH:.2f}",
                        "±ΔH°": f"± {_dH_err:.2f}" if _dH_err is not None else "n/a",
                        "ΔS°":  f"{_dS * 1000:.2f}",
                        "±ΔS°": f"± {_dS_err * 1000:.2f}" if _dS_err is not None else "n/a",
                        "TΔS°": f"{_TdS:.2f}",
                        "±TΔS°":f"± {_TdS_err:.2f}" if _TdS_err is not None else "n/a",
                    })
                if _thermo_rows:
                    st.write(f"**Thermodynamics ({_T_itc:.2f} K):**")
                    st.caption("Energy terms ΔG, ΔH and TΔS in kcal/mol, ΔS in cal/mol·K")
                    st.dataframe(pd.DataFrame(_thermo_rows).set_index("Reaction"),
                                 width="stretch")
        elif fit_mode_disp == "spectra":
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

        # ── Basic fit statistics (collapsible) ────────────────────────
        # All the simple goodness-of-fit numbers (R², RMSE, χ²,
        # AIC/BIC, DW/Shapiro) tucked behind a checkbox so the default
        # post-fit view shows just the fitted-constants table.  Same
        # hierarchy as "Show residual diagnostics".
        @_fragment
        def _stats_eq_frag():
            _diag_scope_switch("_fit_args_eq")
            if st.checkbox("🔢 Show fit statistics",
                           value=False, key="_show_stats_eq",
                           help="R², RMSE, reduced χ², information "
                                "criteria (AIC/BIC), and residual "
                                "tests (Durbin–Watson, Shapiro–Wilk)."):
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
                    # Warnings already shown above the collapsible — only stats here
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

                # ── v2 diagnostics: AIC / BIC / Durbin–Watson / Shapiro ──
                if "aic" in fit_stats and np.isfinite(fit_stats.get("aic", float("nan"))):
                    st.write(f"• AIC = {fit_stats['aic']:.2f}   |   "
                             f"AICc = {fit_stats['aicc']:.2f}   |   "
                             f"BIC = {fit_stats['bic']:.2f}")
                if "durbin_watson" in fit_stats and np.isfinite(fit_stats.get("durbin_watson", float("nan"))):
                    _dw = fit_stats["durbin_watson"]; _sp = fit_stats.get("shapiro_p")
                    _dw_note = "✓" if 1.5 <= _dw <= 2.5 else "⚠ autocorrelation suspected"
                    _sp_note = ("normal" if (_sp is not None and np.isfinite(_sp) and _sp >= 0.05)
                                else "non-normal" if (_sp is not None and np.isfinite(_sp)) else "")
                    _sp_disp = f"; Shapiro p = {_sp:.3f} ({_sp_note})" if (_sp is not None and np.isfinite(_sp)) else ""
                    st.write(f"• Durbin–Watson = {_dw:.2f} ({_dw_note}){_sp_disp}")
                    # Spectra-mode caveat: residuals are concatenated across
                    # many spectra × many wavelengths, and adjacent entries
                    # are the same spectrum at adjacent wavelengths —
                    # correlated by bandpass / baseline drift / spectral
                    # smoothness.  DW (which assumes independent samples)
                    # will read as autocorrelated and Shapiro p will be
                    # tiny essentially every time, regardless of fit
                    # quality.  Show the numbers (they're still computed),
                    # but flag that they are not diagnostically meaningful
                    # for this mode — the informative view is the
                    # residual-vs-predictor panel, not the DW / SW values.
                    if "spectra" in str(fit_stats.get("fit_mode", "")):
                        st.caption(
                            "ℹ Spectra mode: residuals are wavelength-correlated by "
                            "spectrometer bandpass, baseline drift, and spectral "
                            "smoothness, so Durbin–Watson and Shapiro–Wilk are not "
                            "diagnostically meaningful (DW will read low and Shapiro "
                            "p will be tiny regardless of fit quality). The "
                            "informative view is the residuals-vs-predictor panel."
                        )
        _stats_eq_frag()

        # ── v2 diagnostics: residual panel (expander) ────────────────────
        _res_eq = st.session_state.get("_fit_residuals")
        if fit_stats and _res_eq and len(_res_eq.get("y_obs", [])) > 0:
            @_fragment
            def _diag_eq_frag():
                # The parent checkbox uses a key that incorporates the
                # fit-counter so Streamlit treats it as a fresh widget
                # on each new fit, side-stepping the fragment's
                # internal widget-state cache which otherwise keeps
                # the parent stuck "checked" even after the public
                # session_state entry has been cleared.
                _n_eq = int(st.session_state.get("_fit_counter", 0))
                if st.checkbox("📊 Show residual diagnostics",
                               value=False,
                               key=f"_show_diag_eq_{_n_eq}"):
                    # ── Parameter correlation heatmap (linear/Hessian) ──
                    _show_corr_e = _indented_checkbox(
                        "🔗 Show parameter correlation heatmap",
                        value=False, key="_show_corr_eq",
                        help="Pearson ρ derived from the Hessian "
                             "covariance.  Strong off-diagonal entries "
                             "(|ρ| > 0.9) mean the Hessian standard "
                             "errors understate the true uncertainty.  "
                             "Compare against the 2D RMSE profile for "
                             "the nonlinear view.")
                    if _show_corr_e:
                        try:
                            from equilibrist_diagnostics import \
                                make_correlation_heatmap
                            _figh = make_correlation_heatmap(
                                fit_stats,
                                title="Parameter correlation matrix")
                            st.pyplot(_figh, clear_figure=False)
                            import io as _io
                            _bufh = _io.BytesIO()
                            _figh.savefig(_bufh, format="png", dpi=200,
                                          bbox_inches="tight")
                            _bufh.seek(0)
                            _tsh = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                "Download correlation heatmap PNG",
                                data=_bufh.getvalue(),
                                file_name=f"Equilibrist_corr_{_tsh}.png",
                                mime="image/png",
                                key="_corr_dl_eq")
                        except Exception as _ce:
                            st.error(f"Correlation heatmap failed: {_ce}")

                    # ── Parameter identifiability — sloppy spectrum ──
                    # Multi-parameter generalisation of the correlation
                    # heatmap.  See companion comment in the kinetics
                    # block (line ~2287) for the rationale.
                    _show_ident_e = _indented_checkbox(
                        "🔬 Show parameter identifiability "
                        "(sloppy spectrum)",
                        value=False, key="_show_ident_eq",
                        help="Eigendecomposition of the Hessian "
                             "covariance.  Spread of σᵢ across many "
                             "decades = sloppy model — some parameter "
                             "combinations are unconstrained by the "
                             "data.  Brown & Sethna 2003; Gutenkunst "
                             "et al. 2007; Hibbert & Thordarson 2016.")
                    if _show_ident_e:
                        try:
                            render_identifiability_panel(
                                fit_stats, key="ident_eq")
                        except Exception as _ie:
                            st.error(f"Identifiability analysis failed: {_ie}")

                    # ── Parameter significance — t-test ──────────────
                    # Per-parameter (θ̂ − θ₀) / SE, p-value from
                    # Student's t with df = n − p.  The user can
                    # override θ₀ per row via the editable column
                    # (default 0, which for log K means K = 1, and
                    # for log conc means c = 1 mM — handy sanity
                    # check).  Cheap to compute and complements the
                    # heatmap by reporting the *significance* of each
                    # estimate rather than its correlation with others.
                    _show_ttest_e = _indented_checkbox(
                        "📐 Show parameter significance (t-test)",
                        value=False, key="_show_ttest_eq",
                        help="t = (θ̂ − θ₀) / SE, two-tailed p-value "
                             "from Student's t with df = n − p.  "
                             "Estimates and SE are in log10 units; "
                             "edit the Null column to test against "
                             "literature values (e.g. log K = 4.5).")
                    if _show_ttest_e:
                        try:
                            from equilibrist_diagnostics import \
                                compute_param_t_tests
                            _nul_key_e = "_ttest_nulls_eq"
                            _nul_overrides_e = st.session_state.get(
                                _nul_key_e, {})
                            _rows_e = compute_param_t_tests(
                                fit_stats, _nul_overrides_e)
                            if not _rows_e:
                                st.info("No fitted parameters with "
                                        "standard errors available.")
                            else:
                                import pandas as _pd
                                _names_e = [r["name"] for r in _rows_e]
                                st.session_state["_ttest_names_eq"] = _names_e

                                def _on_ttest_eq_change():
                                    # Runs BEFORE the next rerun's
                                    # script body — see kinetics branch
                                    # for the rationale.
                                    _state = st.session_state.get(
                                        "_ttest_editor_eq", {}) or {}
                                    _ed_rows = _state.get(
                                        "edited_rows", {}) or {}
                                    if not _ed_rows:
                                        return
                                    _names = st.session_state.get(
                                        "_ttest_names_eq", [])
                                    _nulls = dict(st.session_state.get(
                                        "_ttest_nulls_eq", {}))
                                    for _ri, _ch in _ed_rows.items():
                                        try:
                                            _i = int(_ri)
                                            if (_i < len(_names)
                                                    and "Null θ₀" in _ch):
                                                _nulls[_names[_i]] = float(_ch["Null θ₀"])
                                        except (ValueError, TypeError, KeyError):
                                            pass
                                    st.session_state["_ttest_nulls_eq"] = _nulls

                                _df_e = _pd.DataFrame(_rows_e)
                                _df_disp_e = _pd.DataFrame({
                                    "Parameter":      _df_e["name"],
                                    "Kind":           _df_e["kind"],
                                    "Estimate (log)": _df_e["value"].round(4),
                                    "SE (log)":       _df_e["se"].round(4),
                                    "Null θ₀":        _df_e["null"].round(4),
                                    "t":              _df_e["t"].round(2),
                                    "p":              _df_e["p"].map(
                                        lambda v: f"{v:.2e}" if _pd.notna(v) else "n/a"),
                                    "Sig":            _df_e["stars"],
                                })
                                st.data_editor(
                                    _df_disp_e,
                                    column_config={
                                        "Parameter":      st.column_config.TextColumn(disabled=True),
                                        "Kind":           st.column_config.TextColumn(disabled=True),
                                        "Estimate (log)": st.column_config.NumberColumn(disabled=True, format="%.4f"),
                                        "SE (log)":       st.column_config.NumberColumn(disabled=True, format="%.4f"),
                                        "Null θ₀":        st.column_config.NumberColumn(
                                                              required=True,
                                                              format="%.4f",
                                                              help="θ₀ used for the t-test.  "
                                                                   "Edit to compare against a "
                                                                   "literature value (in log "
                                                                   "units).  Default 0."),
                                        "t":              st.column_config.NumberColumn(disabled=True, format="%.2f"),
                                        "p":              st.column_config.TextColumn(disabled=True),
                                        "Sig":            st.column_config.TextColumn(disabled=True,
                                                              help="*** p<0.001, ** p<0.01, "
                                                                   "* p<0.05, n.s. otherwise"),
                                    },
                                    hide_index=True,
                                    use_container_width=True,
                                    on_change=_on_ttest_eq_change,
                                    key="_ttest_editor_eq")
                                if _rows_e and _rows_e[0]["df"] > 0:
                                    st.caption(
                                        f"Degrees of freedom: "
                                        f"df = {_rows_e[0]['df']} "
                                        f"(n − p). Estimates and SE "
                                        f"shown in log10 units.")
                        except Exception as _te:
                            st.error(f"t-test failed: {_te}")

                    # ── Rank analysis (EFA / scree / TFA) ────────────
                    # Data-space diagnostic: where the three checkboxes
                    # above ask whether the *fitted parameters* are
                    # well-determined, this one asks whether the *data*
                    # supports the rank (= number of distinguishable
                    # species) implied by the chemical model.  Only
                    # available for fit modes that produce a multi-
                    # channel data matrix D (UV-Vis spectra; multi-
                    # signal NMR shift / integration).  For modes
                    # without D the checkbox is hidden.
                    if fit_stats.get("data_matrix") is not None:
                        _show_rank_e = _indented_checkbox(
                            "📊 Show data-rank analysis "
                            "(EFA / scree / TFA)",
                            value=False, key="_show_rank_eq",
                            help="Evolving Factor Analysis + Malinowski "
                                 "IND + Target Factor Analysis on the "
                                 "data matrix D = C·E^T.  EFA reveals "
                                 "when each species' contribution "
                                 "rises above the noise floor; IND "
                                 "recommends a rank; TFA projects each "
                                 "fitted pure spectrum onto the data's "
                                 "factor space to check that the "
                                 "fitted ε is actually supported by the "
                                 "data.  Maeder & Zuberbühler 1986; "
                                 "Malinowski 1991.")
                        if _show_rank_e:
                            try:
                                render_rank_analysis_panel(
                                    fit_stats, key="rank_eq")
                            except Exception as _re:
                                st.error(f"Rank analysis failed: {_re}")

                    # Optional RMSE parameter profile (Musketeer §4.4
                    # style).  Expensive — refits at every grid point
                    # for every fitted parameter — so it's behind a
                    # checkbox and cached in session_state.
                    _show_prof = _indented_checkbox(
                        "🔍 Show parameter profile (RMSE vs each "
                        "parameter)",
                        value=False, key="_show_prof_eq",
                        help="For each fitted parameter, pins it at a "
                             "grid of values around the optimum and "
                             "refits the others.  Sharp valleys = "
                             "well-determined parameter; flat regions "
                             "= unidentifiable.")
                    _prof = None
                    _prof_fixed = {}
                    if _show_prof:
                        _fa = _diag_fa(st.session_state.get("_fit_args_eq"))
                        if not _fa:
                            st.info("Click **Fit** first to enable "
                                    "the parameter profile.")
                        else:
                            col_a, col_b = st.columns(2)
                            _p_span = col_a.number_input(
                                "Span (log units, ±)", min_value=0.1,
                                max_value=3.0, value=0.5, step=0.1,
                                key="_prof_span_eq")
                            _p_npts = col_b.number_input(
                                "Grid points (odd)", min_value=5,
                                max_value=201, value=11, step=2,
                                key="_prof_npts_eq")
                            (_pk_K, _pk_c, _pk_t, _prof_fixed, _pk_tag,
                             _pk_default) = _profile_param_picker("eq", fit_stats, _fa)
                            _prof_par = st.checkbox(
                                "Parallel (use all cores)",
                                value=False, key="_prof_par_eq",
                                help="Run independent grid-point refits "
                                     "in parallel via joblib.  Speeds up "
                                     "large profiles roughly linearly "
                                     "with core count.")
                            _hpc_go_prof_eq = hpc_ui.controls("prof_eq", label="the 1D profile")
                            _prof_jobs = -1 if _prof_par else 1
                            # Cache key intentionally omits the fitted
                            # ``param_values`` tuple — see kinetics
                            # branch for the rationale (every new fit
                            # already invalidates this cache, and the
                            # tuple was fragile across JSON save/load).
                            _cache_key = ("_prof_cache_eq",
                                          float(_p_span), int(_p_npts))
                            if not _pk_default:      # a custom selection gets its own cache entry
                                _cache_key = _cache_key + (_pk_tag,)
                            _cached = st.session_state.get(_cache_key)
                            if _cached is not None:
                                _prof = _cached
                            else:
                                if st.button("Compute profile",
                                             key="_prof_btn_eq",
                                             disabled=not (_pk_K or _pk_c or _pk_t)) or _hpc_go_prof_eq:
                                    _bar = st.progress(0.0,
                                        text="Computing profile…")
                                    def _cb(f, m):
                                        try:
                                            _bar.progress(min(1.0, f),
                                                          text=m)
                                        except Exception:
                                            pass
                                    try:
                                        _fa_eq_prof = dict(_fa)
                                        if _fa_eq_prof.get("kind") != "global":
                                            _fa_eq_prof["kind"] = "eq"
                                        # Seed every refit from the FITTED values and use a small
                                        # initial simplex: the refits start next to the optimum.
                                        _seed1d = dict(_fa_eq_prof.get("start_logK") or _fa_eq_prof.get("start_logk") or {})
                                        for _n, _v in (fit_stats.get("param_values", {}) or {}).items():
                                            try: _seed1d[_n] = float(_v)
                                            except Exception: pass
                                        _fa_eq_prof["start_logK"] = _seed1d
                                        _fa_eq_prof["start_logk"] = dict(_seed1d)
                                        _fa_eq_prof["simplex_step"] = 0.15
                                        _prof = compute_rmse_profile(
                                            _fa_eq_prof,
                                            _pk_K,
                                            span=float(_p_span),
                                            n_pts=int(_p_npts),
                                            fitted_concs_mM=_pk_c,
                                            fitted_titrants_mM=_pk_t,
                                            n_jobs=_prof_jobs,
                                            progress_callback=_cb)
                                        _write_tuple_cache(_cache_key, _prof)
                                        _bar.empty()
                                    except hpc_ui.CapturedCall as _hpc_cc:
                                        _bar.empty()
                                        hpc_ui.handle_capture(_hpc_cc, "prof_eq", "1D RMSE profile",
                                                              context={"ck": list(_cache_key)})
                                    except Exception as _pe:
                                        _bar.empty()
                                        st.error(f"Profile failed: {_pe}")
                            hpc_ui.jobs_panel("prof_eq", on_result=_hpc_store_tuple_cache)
                    render_diagnostics_panel(fit_stats, _res_eq,
                                             key="diag_eq",
                                             rmse_profile=_profile_display(_prof, _prof_fixed))

                    # ── 2D RMSE profile (pairwise) — equilibrium ─────
                    _show_p2d_e = _indented_checkbox(
                        "🌐 Show 2D parameter profile "
                        "(RMSE colormap for a pair)",
                        value=False, key="_show_p2d_eq",
                        help="At each (X, Y) grid point both "
                             "parameters are pinned and every OTHER "
                             "free parameter is refit.  Long diagonal "
                             "valleys = strong pairwise correlation; "
                             "round wells = independently identified.")
                    if _show_p2d_e:
                        _fa_e_for2d = _diag_fa(st.session_state.get("_fit_args_eq"))
                        if not _fa_e_for2d:
                            st.info("Click **Fit** first to enable "
                                    "the 2D profile.")
                        else:
                            _cands = {}
                            for _n, _v in (fit_stats.get(
                                    "param_values", {}) or {}).items():
                                try: _cands[_n] = float(_v)
                                except Exception: pass
                            for _n, _mM in (fit_stats.get(
                                    "fitted_concs", {}) or {}).items():
                                try:
                                    _v = float(_mM)
                                    if _v > 0:
                                        _cands[_n] = float(np.log10(_v))
                                except Exception: pass
                            for _n, _mM in (fit_stats.get(
                                    "fitted_titrants", {}) or {}).items():
                                try:
                                    _v = float(_mM)
                                    if _v > 0:
                                        _cands[_n] = float(np.log10(_v))
                                except Exception: pass
                            # Constants that were held FIXED in the last fit are offered too:
                            # the refit engine pins both scanned parameters anyway, so a
                            # fixed constant can be scanned against a fitted one while every
                            # other fitted parameter is re-optimised at each grid point.
                            _fixed_cands = {}
                            _slk2d = (_fa_e_for2d.get("start_logK")
                                      or _fa_e_for2d.get("start_logk") or {})
                            for _n, _v in _slk2d.items():
                                if _n in _cands or _n == "Kw": continue
                                try: _fixed_cands[_n] = float(_v)
                                except Exception: pass
                            _cands.update(_fixed_cands)
                            _lbl2d = lambda _n: (f"{_n} (fixed)" if _n in _fixed_cands else _n)
                            _names = list(_cands.keys())
                            if len(_names) < 2:
                                st.info("Need at least 2 "
                                        "parameters to build a 2D "
                                        "profile.")
                            else:
                                if len(_names) == 2:
                                    _p1_e, _p2_e = _names[0], _names[1]
                                    st.caption(f"Auto-selected: "
                                               f"**{_p1_e}** vs **{_p2_e}** "
                                               "(only two available).")
                                else:
                                    c1, c2 = st.columns(2)
                                    _p1_e = c1.selectbox(
                                        "X-axis parameter", _names,
                                        index=0, key="_p2d_p1_eq",
                                        format_func=_lbl2d)
                                    _p2_e = c2.selectbox(
                                        "Y-axis parameter", _names,
                                        index=1, key="_p2d_p2_eq",
                                        format_func=_lbl2d)
                                _fx_sel = [_n for _n in (_p1_e, _p2_e) if _n in _fixed_cands]
                                if _fx_sel:
                                    st.caption("Held fixed in the last fit: " + ", ".join(
                                        f"**{_n}** = {_fixed_cands[_n]:.2f}" for _n in _fx_sel) +
                                        ". The map is centred on that value, which is not an "
                                        "optimum — look for where the RMSE minimum actually "
                                        "falls. All other fitted parameters are refit at each "
                                        "grid point.")
                                c_sx, c_sy, c_np = st.columns(3)
                                _sx_e = c_sx.number_input(
                                    "Span X (log ±)", min_value=0.1,
                                    max_value=3.0, value=0.5, step=0.1,
                                    key="_p2d_sx_eq")
                                _sy_e = c_sy.number_input(
                                    "Span Y (log ±)", min_value=0.1,
                                    max_value=3.0, value=0.5, step=0.1,
                                    key="_p2d_sy_eq")
                                _np_e = c_np.number_input(
                                    "Grid (per axis, odd)",
                                    min_value=5, max_value=201,
                                    value=11, step=2,
                                    key="_p2d_np_eq")
                                _par2_e = st.checkbox(
                                    "Parallel (use all cores)",
                                    value=True, key="_p2d_par_eq",
                                    help="Strongly recommended — "
                                         "2D scans run n_pts² refits.")
                                _hpc_go_p2d_eq = hpc_ui.controls("p2d_eq", label="the 2D profile")
                                _p2d_jobs_e = -1 if _par2_e else 1
                                if _p1_e == _p2_e:
                                    st.warning("Pick two **different** "
                                               "parameters.")
                                else:
                                    # Cache key omits the per-fit
                                    # parameter-value tuple — see
                                    # kinetics branch for rationale.
                                    _ck = ("_p2d_cache_eq", _p1_e, _p2_e,
                                           float(_sx_e), float(_sy_e),
                                           int(_np_e))
                                    _cached2 = st.session_state.get(_ck)
                                    if _cached2 is not None:
                                        _scan2 = _cached2
                                    else:
                                        _scan2 = None
                                        if st.button(
                                                "Compute 2D profile",
                                                key="_p2d_btn_eq") or _hpc_go_p2d_eq:
                                            _bar2 = st.progress(0.0,
                                                text="Computing 2D profile…")
                                            def _cb2(f, m):
                                                try: _bar2.progress(min(1.0, f), text=m)
                                                except Exception: pass
                                            try:
                                                _fa_e_prof2 = dict(_fa_e_for2d)
                                                if _fa_e_prof2.get("kind") != "global":
                                                    _fa_e_prof2["kind"] = "eq"
                                                # Seed every grid refit from the FITTED
                                                # constants (not the pre-fit start) so the
                                                # remaining parameters converge quickly.
                                                _seed2d = dict(_fa_e_for2d.get("start_logK")
                                                               or _fa_e_for2d.get("start_logk") or {})
                                                for _n, _v in (fit_stats.get("param_values", {}) or {}).items():
                                                    try: _seed2d[_n] = float(_v)
                                                    except Exception: pass
                                                _fa_e_prof2["start_logK"] = _seed2d
                                                _fa_e_prof2["start_logk"] = dict(_seed2d)
                                                _fa_e_prof2["simplex_step"] = 0.15
                                                _scan2 = compute_rmse_profile_2d(
                                                    _fa_e_prof2,
                                                    _p1_e, _cands[_p1_e],
                                                    _p2_e, _cands[_p2_e],
                                                    span_x=float(_sx_e),
                                                    span_y=float(_sy_e),
                                                    n_pts_x=int(_np_e),
                                                    n_pts_y=int(_np_e),
                                                    n_jobs=_p2d_jobs_e,
                                                    progress_callback=_cb2)
                                                _scan2["p1_fixed"] = bool(_p1_e in _fixed_cands)
                                                _scan2["p2_fixed"] = bool(_p2_e in _fixed_cands)
                                                _write_tuple_cache(_ck, _scan2)
                                                _bar2.empty()
                                            except hpc_ui.CapturedCall as _hpc_cc:
                                                _bar2.empty()
                                                hpc_ui.handle_capture(_hpc_cc, "p2d_eq", "2D RMSE profile",
                                                                      context={"ck": list(_ck), "p1_fixed": bool(_p1_e in _fixed_cands), "p2_fixed": bool(_p2_e in _fixed_cands)})
                                            except Exception as _e2:
                                                _bar2.empty()
                                                st.error(f"2D profile failed: {_e2}")
                                    hpc_ui.jobs_panel("p2d_eq", on_result=_hpc_store_tuple_cache)
                                    if _scan2 is not None:
                                        _p2d_noleg = st.checkbox(
                                            "No legend", value=False, key="_p2d_noleg_eq",
                                            help="Draw the map without the legend box and the "
                                                 "note in the corner — for publication, where "
                                                 "that text goes in the figure caption.")
                                        try:
                                            _fig2 = make_2d_profile_figure(
                                                _scan2,
                                                title=f"2D RMSE profile: "
                                                      f"{_p1_e} vs {_p2_e}",
                                                param_cov=fit_stats.get("param_cov"),
                                                cov_names=fit_stats.get("param_cov_names"),
                                                constraints=_fa_e_for2d.get("constraints"),
                                                show_legend=not _p2d_noleg)
                                            st.pyplot(_fig2, clear_figure=False)
                                            import io as _io
                                            _buf2 = _io.BytesIO()
                                            _fig2.savefig(_buf2, format="png",
                                                          dpi=200,
                                                          bbox_inches="tight")
                                            _buf2.seek(0)
                                            _ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            st.download_button(
                                                "Download 2D profile PNG",
                                                data=_buf2.getvalue(),
                                                file_name=f"Equilibrist_2Dprofile_"
                                                          f"{_p1_e}_{_p2_e}_{_ts2}.png",
                                                mime="image/png",
                                                key="_p2d_dl_eq")
                                        except Exception as _re:
                                            st.error(f"Could not render 2D "
                                                     f"figure: {_re}")
                    _profile_trace_block("eq", fit_stats)
            _diag_eq_frag()

        # ── v2 bootstrap CIs (toggle, in fragment) ───────────────────────
        if fit_stats:
            @_fragment
            def _bs_eq_frag():
                if st.checkbox("🔁 Show bootstrap confidence intervals",
                               value=False, key="_show_bs_eq"):
                    st.caption("Resamples fit residuals to estimate CIs without "
                               "assuming Gaussian errors. Recommended when "
                               "Shapiro p < 0.05 above. Each iteration is a full refit.")
                    _fa = _diag_fa(st.session_state.get("_fit_args_eq"))
                    if not _fa:
                        st.info("Click **Fit** once with the current script and data "
                                "to enable bootstrap.  (The bootstrap reuses the exact "
                                "configuration of the most recent fit.)")
                    else:
                        _bs_n   = st.number_input("Iterations", min_value=20, max_value=10000,
                                                  value=200, step=50, key="_bs_n_eq")
                        _bs_mth = st.selectbox("Resampling method",
                                               ["residual", "parametric", "wild"],
                                               index=0, key="_bs_mth_eq")
                        _bs_par = st.checkbox("Parallel (use all cores)",
                                              value=False, key="_bs_par_eq")
                        _hpc_go_bs_eq = hpc_ui.controls("bs_eq", label="the bootstrap")
                        if st.button("Run bootstrap", key="_bs_run_eq") or _hpc_go_bs_eq:
                            _pb = st.progress(0.0, text="Bootstrap running…")
                            _cb = lambda i, N: _pb.progress(min(i / max(N,1), 1.0),
                                                            text=f"Bootstrap {i}/{N}")
                            _jobs = -1 if _bs_par else 1
                            try:
                                if _fa.get("kind") == "global":
                                    _r = eqglobal.bootstrap_global(
                                        _fa, n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42, progress_callback=_cb)
                                elif _fa["use_spectra_fit"]:
                                    _r = ebs.bootstrap_spectra(
                                        _fa["parsed_fit"], _fa["network"], _fa["spectra_data_fit"],
                                        _fa["params"], _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        wl_min=_fa["wl_min"], wl_max=_fa["wl_max"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        timeout_s=float(st.session_state.get("fit_timeout", 30)),
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        fit_titrant_keys=_fa["fit_titrant_keys"],
                                        allow_negative_eps=_fa.get("allow_neg_eps", False),
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "shift":
                                    _r = ebs.bootstrap_nmr_shift(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        fit_titrant_keys=_fa["fit_titrant_keys"],
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "integration":
                                    _r = ebs.bootstrap_nmr_integ(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        fit_titrant_keys=_fa["fit_titrant_keys"],
                                        progress_callback=_cb)
                                elif _fa["use_nmr_fit"] and _fa["nmr_mode"] == "mixed":
                                    _r = ebs.bootstrap_nmr_mixed(
                                        _fa["parsed_fit"], _fa["network"], _fa["nmr_data_fit"],
                                        _fa["params"], _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        fit_titrant_keys=_fa["fit_titrant_keys"],
                                        progress_callback=_cb)
                                elif _fa.get("use_itc_fit"):
                                    _r = ebs.bootstrap_itc(
                                        _fa["parsed_fit"], _fa["network"],
                                        _fa["itc_q_exp"], _fa["itc_raw"],
                                        _fa["params"], _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"],
                                        maxiter=_fa["maxiter"],
                                        timeout_s=float(_fa.get("timeout_s", 30.0)),
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        q_offset_fixed=_fa.get("q_offset_fixed"),
                                        excl_set=_fa.get("excl_set") or set(),
                                        progress_callback=_cb)
                                else:
                                    _exp = _fa["exp_data_fit"] or {}
                                    _xcol = _exp.get("_x_col_header", "")
                                    _r = ebs.bootstrap_conc(
                                        _fa["parsed_fit"], _fa["network"], _exp, _fa["params"],
                                        _fa["start_logK"],
                                        _fa["fit_keys"], _fa["x_expr"],
                                        n_bootstrap=int(_bs_n), method=_bs_mth,
                                        n_jobs=_jobs, seed=42,
                                        tolerance=_fa["tolerance"], maxiter=_fa["maxiter"],
                                        constraints=_fa["constraints"],
                                        fit_conc_keys=_fa["fit_conc_keys"],
                                        fit_titrant_keys=_fa["fit_titrant_keys"],
                                        x_col_header=_xcol,
                                        progress_callback=_cb)
                                _pb.empty()
                                st.session_state["_bs_result_eq"] = _r
                            except hpc_ui.CapturedCall as _hpc_cc:
                                _pb.empty()
                                hpc_ui.handle_capture(_hpc_cc, "bs_eq", "Bootstrap",
                                                      context={})
                            except Exception as _bse:
                                _pb.empty()
                                import traceback as _tb_bs
                                st.error(f"Bootstrap failed: {_bse}")
                                st.caption(_tb_bs.format_exc())
                        hpc_ui.jobs_panel("bs_eq", on_result=_hpc_store_state("_bs_result_eq"))
                        _br = st.session_state.get("_bs_result_eq")
                        if _br and _br.get("ci"):
                            st.caption(f"{_br['n_success']}/{_br['n_bootstrap']} bootstrap "
                                       f"fits converged ({_br['wall_seconds']:.1f}s; "
                                       f"method = {_br['method']})")
                            # ── Multi-level CIs always shown ──
                            # Bootstrap samples (raw resamples) live in
                            # _br["samples"] so the four standard
                            # percentile intervals (50/80/95/99 %, the
                            # SupraFit / SIVVU convention) are computed
                            # on the fly.  Columns ordered widest →
                            # narrowest so the eye moves from the
                            # conservative interval to the tight one.
                            _levels_eq = [99, 95, 80, 50]
                            _samples_eq = _br.get("samples") or {}
                            _rows = []
                            for _n in _br["ci"].keys():
                                _pt = _br["best_fit"].get(_n, float("nan"))
                                _s = np.asarray(_samples_eq.get(_n, []),
                                                dtype=float)
                                _s = _s[np.isfinite(_s)]
                                _row = {"parameter": _n,
                                        "point":  f"{_pt:+.4f}",
                                        "median": (f"{np.median(_s):+.4f}"
                                                    if _s.size else "—"),
                                        "σ":      (f"{np.std(_s, ddof=1):.4f}"
                                                    if _s.size > 1 else "—")}
                                for _lvl in _levels_eq:
                                    if _s.size:
                                        _alpha = 1.0 - _lvl / 100.0
                                        _lo, _hi = np.quantile(
                                            _s, [_alpha/2, 1 - _alpha/2])
                                        _row[f"{_lvl} % CI"] = \
                                            f"[{_lo:+.4f}, {_hi:+.4f}]"
                                    else:
                                        _row[f"{_lvl} % CI"] = "—"
                                _rows.append(_row)
                            st.dataframe(_rows, hide_index=True,
                                          use_container_width=True)
            _bs_eq_frag()

        # ── v2 jackknife (leave-one-out) ─────────────────────────────────
        # Refits the model N times each dropping one data point in turn.
        # Yields a jackknife SE per parameter and a per-point influence
        # signal that reveals which observations the fit is most sensitive
        # to.  Complementary to the bootstrap: bootstrap quantifies how
        # uncertainty in the data propagates; jackknife identifies *which*
        # individual observations are doing the most work.  Currently
        # prototyped for NMR-shift fits; extending to the other modes is
        # mechanical once the UX is approved.
        if fit_stats:
            @_fragment
            def _jk_eq_frag():
                if st.checkbox("🪛 Show jackknife (leave-one-out) analysis",
                               value=False, key="_show_jk_eq",
                               help="For each refit the jackknife "
                                    "removes one full titration step "
                                    "— that is, all signal columns "
                                    "(or all wavelengths for UV\u2013Vis "
                                    "spectra; all observed species at "
                                    "one time for kinetics) at one "
                                    "x-value, simultaneously."):
                    _fa_jk = _diag_fa(st.session_state.get("_fit_args_eq"))
                    if not _fa_jk:
                        st.info("Click **Fit** once with the current "
                                "script and data to enable jackknife.")
                    else:
                        _jk_par = st.checkbox("Parallel (use all cores)",
                                               value=False, key="_jk_par_eq")
                        _hpc_go_jk_eq = hpc_ui.controls("jk_eq", label="the jackknife")
                        if st.button("Run jackknife", key="_jk_run_eq") or _hpc_go_jk_eq:
                            _pb_jk = st.progress(0.0,
                                                  text="Jackknife running…")
                            _cb_jk = lambda i, N: _pb_jk.progress(
                                min(i / max(N, 1), 1.0),
                                text=f"Jackknife {i}/{N}")
                            _jobs_jk = -1 if _jk_par else 1
                            _to_s = float(st.session_state.get(
                                "fit_timeout", 30))
                            try:
                                _jr = None
                                if _fa_jk.get("kind") == "global":
                                    _jr = eqglobal.jackknife_global(
                                        _fa_jk, n_jobs=_jobs_jk, progress_callback=_cb_jk)
                                elif _fa_jk.get("use_spectra_fit"):
                                    _jr = ebs.jackknife_spectra(
                                        _fa_jk["parsed_fit"], _fa_jk["network"],
                                        _fa_jk["spectra_data_fit"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"],
                                        _fa_jk["x_expr"],
                                        wl_min=_fa_jk["wl_min"],
                                        wl_max=_fa_jk["wl_max"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        timeout_s=_to_s,
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        fit_titrant_keys=_fa_jk["fit_titrant_keys"],
                                        allow_negative_eps=_fa_jk.get(
                                            "allow_neg_eps", False),
                                        progress_callback=_cb_jk)
                                elif (_fa_jk.get("use_nmr_fit")
                                       and _fa_jk.get("nmr_mode") == "shift"):
                                    _jr = ebs.jackknife_nmr_shift(
                                        _fa_jk["parsed_fit"], _fa_jk["network"], _fa_jk["nmr_data_fit"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"], _fa_jk["x_expr"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        fit_titrant_keys=_fa_jk["fit_titrant_keys"],
                                        progress_callback=_cb_jk)
                                elif (_fa_jk.get("use_nmr_fit")
                                       and _fa_jk.get("nmr_mode") == "integration"):
                                    _jr = ebs.jackknife_nmr_integration(
                                        _fa_jk["parsed_fit"], _fa_jk["network"], _fa_jk["nmr_data_fit"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"], _fa_jk["x_expr"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        fit_titrant_keys=_fa_jk["fit_titrant_keys"],
                                        progress_callback=_cb_jk)
                                elif (_fa_jk.get("use_nmr_fit")
                                       and _fa_jk.get("nmr_mode") == "mixed"):
                                    _jr = ebs.jackknife_nmr_mixed(
                                        _fa_jk["parsed_fit"], _fa_jk["network"], _fa_jk["nmr_data_fit"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"], _fa_jk["x_expr"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        fit_titrant_keys=_fa_jk["fit_titrant_keys"],
                                        progress_callback=_cb_jk)
                                elif _fa_jk.get("use_itc_fit"):
                                    _jr = ebs.jackknife_itc(
                                        _fa_jk["parsed_fit"], _fa_jk["network"],
                                        _fa_jk["itc_q_exp"], _fa_jk["itc_raw"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"], _fa_jk["x_expr"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        timeout_s=float(_fa_jk.get("timeout_s", 30.0)),
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        q_offset_fixed=_fa_jk.get("q_offset_fixed"),
                                        excl_set=_fa_jk.get("excl_set") or set(),
                                        progress_callback=_cb_jk)
                                else:
                                    # Default to CONC mode for equilibrium fits
                                    _jr = ebs.jackknife_conc(
                                        _fa_jk["parsed_fit"], _fa_jk["network"], _fa_jk["exp_data_fit"],
                                        _fa_jk["params"], _fa_jk["start_logK"],
                                        _fa_jk["fit_keys"], _fa_jk["x_expr"],
                                        n_jobs=_jobs_jk,
                                        tolerance=_fa_jk["tolerance"],
                                        maxiter=_fa_jk["maxiter"],
                                        use_lbfgsb=_fa_jk.get("use_lbfgsb", True),
                                        use_neldermead=_fa_jk.get("use_neldermead", True),
                                        constraints=_fa_jk["constraints"],
                                        fit_conc_keys=_fa_jk["fit_conc_keys"],
                                        fit_titrant_keys=_fa_jk["fit_titrant_keys"],
                                        progress_callback=_cb_jk)
                                if _jr is not None:
                                    st.session_state["_jk_result_eq"] = _jr
                            except hpc_ui.CapturedCall as _hpc_cc:
                                _pb_jk.empty()
                                hpc_ui.handle_capture(_hpc_cc, "jk_eq", "Jackknife",
                                                      context={})
                            except Exception as _jke:
                                _pb_jk.empty()
                                import traceback as _tb_jk
                                st.error(f"Jackknife failed: {_jke}")
                                st.caption(_tb_jk.format_exc())
                            else:
                                _pb_jk.empty()

                        hpc_ui.jobs_panel("jk_eq", on_result=_hpc_store_state("_jk_result_eq"))
                        _jr = st.session_state.get("_jk_result_eq")
                        if _jr is not None and not _jr.get("jack_se"):
                            # Result exists but produced no usable
                            # estimates — surface why instead of staying
                            # silent.
                            _msg = (_jr.get("best_message")
                                     or "Jackknife produced no estimates "
                                        "(0 successful refits).")
                            st.warning(f"Jackknife: {_msg}")
                        if _jr and _jr.get("jack_se"):
                            st.caption(f"{_jr['n_success']}/{_jr['n_jack']} "
                                       f"refits successful "
                                       f"({_jr['wall_seconds']:.1f}s)")
                            # Summary table: param | unit | full | SE_jack
                            # The unit column makes clear that θ_full and
                            # SE_jack are in log10 space for K/k parameters
                            # and linear mM for fitted concentrations and
                            # titrants — the same convention as the
                            # influence plot below.
                            _rows_jk = []
                            _kinds_eq = _jr.get("param_kinds", {}) or {}
                            for _n, _se in _jr["jack_se"].items():
                                _fv = _jr["best_fit"].get(_n, float("nan"))
                                _kind = _kinds_eq.get(_n, "log")
                                if   _kind == "log":       _unit_str = "log10"
                                elif _kind == "linear_k":  _unit_str = "linear"
                                else:                       _unit_str = "mM"
                                _rows_jk.append({
                                    "parameter":  _n,
                                    "unit":       _unit_str,
                                    "θ_full":     f"{_fv:+.2e}",
                                    "SE_jack":    (f"{_se:.2e}"
                                                    if np.isfinite(_se)
                                                    else "—"),
                                    "95 % CI (≈ ±1.96·SE)": (
                                        f"[{_fv - 1.96*_se:+.2e}, "
                                        f"{_fv + 1.96*_se:+.2e}]"
                                        if np.isfinite(_se)
                                        else "—"),
                                })
                            st.dataframe(_rows_jk, hide_index=True,
                                          use_container_width=True)

                            # Influence plot toggle
                            _show_inf = st.checkbox(
                                "Show per-point influence plot",
                                value=True, key="_show_jk_influence_eq",
                                help="Δ = θ_full − θ_drop for each data "
                                     "point.  Dashed lines at ±2·SE_jack "
                                     "flag influential observations.")
                            if _show_inf:
                                from equilibrist_diagnostics import \
                                    make_jackknife_figure
                                _x_kind = st.radio(
                                    "Plot x-axis",
                                    options=["x_value", "index"],
                                    horizontal=True,
                                    key="_jk_xkind_eq",
                                    help="x_value: actual x-coordinate "
                                         "(useful when most influential "
                                         "points cluster around the "
                                         "equivalence region).  index: "
                                         "iteration order (useful when "
                                         "signals overlap on x).")
                                _fig_jk = make_jackknife_figure(
                                    _jr, x_kind=_x_kind,
                                    x_label=_fa_jk.get("x_expr", "x"),
                                    title="Jackknife (leave-one-out) "
                                          "influence")
                                st.pyplot(_fig_jk, clear_figure=False)
                                import io as _io
                                _buf_jk = _io.BytesIO()
                                _fig_jk.savefig(_buf_jk, format="png",
                                                 dpi=200,
                                                 bbox_inches="tight")
                                _buf_jk.seek(0)
                                _ts_jk = datetime.now().strftime(
                                    "%Y%m%d_%H%M%S")
                                st.download_button(
                                    "Download influence plot PNG",
                                    data=_buf_jk.getvalue(),
                                    file_name=f"Equilibrist_jackknife_"
                                              f"{_ts_jk}.png",
                                    mime="image/png",
                                    key="_jk_dl_eq")
            _jk_eq_frag()

        # ── v2 Monte Carlo (nuisance-parameter uncertainty) ──────────────
        # Unlike bootstrap (which captures noise sensitivity), Monte Carlo
        # here samples uncertainty in the *experimentally-fixed* values —
        # stock concentrations, V₀ — and propagates them through the fit.
        # The MC σ shows how much of the final precision on each fitted
        # constant is attributable to those experimental-prep errors, a
        # contribution invisible to the Hessian, bootstrap, or jackknife.
        if fit_stats:
            @_fragment
            def _mc_eq_frag():
                if st.checkbox("🎲 Show Monte Carlo "
                               "(experimental-uncertainty propagation)",
                               value=False, key="_show_mc_eq",
                               help="For each iteration the MC samples "
                                    "user-specified relative uncertainties "
                                    "on stock concentrations and V₀, then "
                                    "refits the same observed data with "
                                    "the perturbed (assumed-known) values. "
                                    "The σ on the fitted constants is the "
                                    "contribution from experimental-prep "
                                    "uncertainty — invisible to the "
                                    "residual-noise bootstrap."):
                    _fa_mc = _diag_fa(st.session_state.get("_fit_args_eq"))
                    if not _fa_mc:
                        st.info("Click **Fit** once with the current "
                                "script and data to enable Monte Carlo.")
                    else:
                        # Build the list of NUISANCE parameters: anything
                        # in conc0 not in fit_conc_keys, anything in
                        # titrant_mMs not in fit_titrant_keys, plus V0.
                        _params_mc = _fa_mc.get("params") or {}
                        _conc0   = _params_mc.get("conc0", {}) or {}
                        _titmMs  = _params_mc.get("titrant_mMs", {}) or {}
                        _fck     = set(_fa_mc.get("fit_conc_keys", []) or [])
                        _ftk     = set(_fa_mc.get("fit_titrant_keys", []) or [])
                        _conc_nuis    = [n for n in _conc0  if n not in _fck]
                        _titrant_nuis = [n for n in _titmMs if n not in _ftk]

                        st.caption("Default σ = 0 % (no contribution). "
                                   "Set realistic relative uncertainties "
                                   "for each prep-error source. Typical "
                                   "values: 2–3 % for gravimetric stock "
                                   "concentrations, 0.5–1 % for "
                                   "volumetric V₀.")
                        # Input grid: per-component relative σ
                        _specs: dict = {}
                        _all_inputs: list = []
                        for _n in _conc_nuis:
                            _all_inputs.append(("conc0",
                                                 f"conc0[{_n}] (nominal "
                                                 f"{_conc0[_n]:.3g} mM)",
                                                 _n))
                        for _n in _titrant_nuis:
                            _all_inputs.append(("titrant",
                                                 f"titrant[{_n}] "
                                                 f"(nominal "
                                                 f"{_titmMs[_n]:.3g} mM)",
                                                 _n))
                        if "V0_mL" in _params_mc:
                            _all_inputs.append(("V0_mL",
                                                 f"V₀ (nominal "
                                                 f"{_params_mc['V0_mL']:.3g} mL)",
                                                 None))

                        _mc_is_nmr = bool(_fa_mc.get("use_nmr_fit")) or (
                            _fa_mc.get("kind") == "global" and any(
                                (e.get("fa") or {}).get("use_nmr_fit") for e in _fa_mc.get("experiments", [])))
                        if not _all_inputs and not _mc_is_nmr:
                            st.info("No nuisance parameters to sample "
                                    "— every experimental quantity is "
                                    "already a fit variable.")
                        else:
                            # Render in 2-column grid
                            _ncols_inp = 2
                            _cols_inp  = st.columns(_ncols_inp)
                            for _ii, (_kind, _label, _name) in enumerate(_all_inputs):
                                _col = _cols_inp[_ii % _ncols_inp]
                                _val = _col.number_input(
                                    f"σ on {_label} (%)",
                                    min_value=0.0, max_value=50.0,
                                    value=0.0, step=0.1,
                                    key=f"_mc_sig_eq_{_kind}_{_name}",
                                    format="%.2f")
                                if _val > 0:
                                    if _kind == "conc0":
                                        _specs[f"conc0:{_name}"]   = _val / 100.0
                                    elif _kind == "titrant":
                                        _specs[f"titrant:{_name}"] = _val / 100.0
                                    elif _kind == "V0_mL":
                                        _specs["V0_mL"]           = _val / 100.0
                            if _mc_is_nmr:
                                # Measurement noise on the NMR data itself: each
                                # iteration also fits a perturbed copy of the data.
                                _mn1, _mn2 = st.columns(2)
                                _mc_integ_pct = _mn1.number_input(
                                    "σ on NMR integrations (%)", min_value=0.0, max_value=50.0,
                                    value=0.0, step=0.5, key="_mc_sig_eq_nmr_integ", format="%.2f",
                                    help="Relative error of the peak integrals (per point).  "
                                         "0 leaves the integrals untouched.")
                                _mc_shift_ppm = _mn2.number_input(
                                    "σ on chemical shifts (ppm)", min_value=0.0, max_value=1.0,
                                    value=0.0, step=0.001, key="_mc_sig_eq_nmr_shift", format="%.4f",
                                    help="Absolute error of the observed shifts (per point, ppm).  "
                                         "0 leaves the shifts untouched.")
                                if _mc_integ_pct > 0:
                                    _specs["nmr_integ_rel"] = float(_mc_integ_pct) / 100.0
                                if _mc_shift_ppm > 0:
                                    _specs["nmr_shift_ppm"] = float(_mc_shift_ppm)

                            _ci_n, _ci_p = st.columns([1, 1])
                            _n_mc = _ci_n.number_input(
                                "MC iterations",
                                min_value=50, max_value=5000, value=500,
                                step=50, key="_mc_iter_eq")
                            _mc_par = _ci_p.checkbox(
                                "Parallel (use all cores)",
                                value=False, key="_mc_par_eq")
                            _hpc_go_mc_eq = hpc_ui.controls("mc_eq", label="the Monte Carlo run")

                            if not _specs:
                                st.warning("All uncertainties are 0 % — "
                                           "MC would be degenerate. Set "
                                           "at least one σ > 0 before "
                                           "running.")

                            if st.button("Run Monte Carlo",
                                          key="_mc_run_eq",
                                          disabled=(not _specs)) or (_hpc_go_mc_eq and bool(_specs)):
                                _pb_mc = st.progress(0.0, text="MC running…")
                                _cb_mc = lambda i, N: _pb_mc.progress(
                                    min(i / max(N, 1), 1.0),
                                    text=f"MC {i}/{N}")
                                _jobs_mc = -1 if _mc_par else 1
                                try:
                                    _mr = None
                                    if _fa_mc.get("kind") == "global":
                                        _mr = eqglobal.monte_carlo_global(
                                            _fa_mc, nuisance_specs=_specs, n_iter=int(_n_mc),
                                            n_jobs=_jobs_mc, progress_callback=_cb_mc)
                                    elif _fa_mc.get("use_spectra_fit"):
                                        _mr = ebs.monte_carlo_spectra(
                                            _fa_mc["parsed_fit"], _fa_mc["network"],
                                            _fa_mc["spectra_data_fit"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"],
                                            _fa_mc["x_expr"],
                                            wl_min=_fa_mc["wl_min"],
                                            wl_max=_fa_mc["wl_max"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc),
                                            n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            fit_titrant_keys=_fa_mc["fit_titrant_keys"],
                                            allow_negative_eps=_fa_mc.get("allow_neg_eps", False),
                                            progress_callback=_cb_mc)
                                    elif _fa_mc.get("use_nmr_fit") and _fa_mc.get("nmr_mode") == "shift":
                                        _mr = ebs.monte_carlo_nmr_shift(
                                            _fa_mc["parsed_fit"], _fa_mc["network"], _fa_mc["nmr_data_fit"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"], _fa_mc["x_expr"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc), n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            fit_titrant_keys=_fa_mc["fit_titrant_keys"],
                                            progress_callback=_cb_mc)
                                    elif _fa_mc.get("use_nmr_fit") and _fa_mc.get("nmr_mode") == "integration":
                                        _mr = ebs.monte_carlo_nmr_integration(
                                            _fa_mc["parsed_fit"], _fa_mc["network"], _fa_mc["nmr_data_fit"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"], _fa_mc["x_expr"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc), n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            fit_titrant_keys=_fa_mc["fit_titrant_keys"],
                                            progress_callback=_cb_mc)
                                    elif _fa_mc.get("use_nmr_fit") and _fa_mc.get("nmr_mode") == "mixed":
                                        _mr = ebs.monte_carlo_nmr_mixed(
                                            _fa_mc["parsed_fit"], _fa_mc["network"], _fa_mc["nmr_data_fit"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"], _fa_mc["x_expr"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc), n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            fit_titrant_keys=_fa_mc["fit_titrant_keys"],
                                            progress_callback=_cb_mc)
                                    elif _fa_mc.get("use_itc_fit"):
                                        _mr = ebs.monte_carlo_itc(
                                            _fa_mc["parsed_fit"], _fa_mc["network"],
                                            _fa_mc["itc_q_exp"], _fa_mc["itc_raw"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"], _fa_mc["x_expr"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc), n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            timeout_s=float(_fa_mc.get("timeout_s", 30.0)),
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            q_offset_fixed=_fa_mc.get("q_offset_fixed"),
                                            excl_set=_fa_mc.get("excl_set") or set(),
                                            seed=42, progress_callback=_cb_mc)
                                    else:
                                        _mr = ebs.monte_carlo_conc(
                                            _fa_mc["parsed_fit"], _fa_mc["network"], _fa_mc["exp_data_fit"],
                                            _fa_mc["params"], _fa_mc["start_logK"],
                                            _fa_mc["fit_keys"], _fa_mc["x_expr"],
                                            nuisance_specs=_specs,
                                            n_iter=int(_n_mc), n_jobs=_jobs_mc,
                                            tolerance=_fa_mc["tolerance"],
                                            maxiter=_fa_mc["maxiter"],
                                            use_lbfgsb=_fa_mc.get("use_lbfgsb", True),
                                            use_neldermead=_fa_mc.get("use_neldermead", True),
                                            constraints=_fa_mc["constraints"],
                                            fit_conc_keys=_fa_mc["fit_conc_keys"],
                                            fit_titrant_keys=_fa_mc["fit_titrant_keys"],
                                            progress_callback=_cb_mc)
                                    if _mr is not None:
                                        st.session_state["_mc_result_eq"] = _mr
                                except hpc_ui.CapturedCall as _hpc_cc:
                                    _pb_mc.empty()
                                    hpc_ui.handle_capture(_hpc_cc, "mc_eq", "Monte Carlo",
                                                          context={})
                                except Exception as _mce:
                                    _pb_mc.empty()
                                    import traceback as _tb_mc
                                    st.error(f"Monte Carlo failed: {_mce}")
                                    st.caption(_tb_mc.format_exc())
                                else:
                                    _pb_mc.empty()

                            hpc_ui.jobs_panel("mc_eq", on_result=_hpc_store_state("_mc_result_eq"))
                        _mr = st.session_state.get("_mc_result_eq")
                        if _mr is not None and not _mr.get("mc_se"):
                            st.warning(f"Monte Carlo: "
                                        f"{_mr.get('best_message', 'no result')}")
                        if _mr and _mr.get("mc_se"):
                            st.caption(
                                f"{_mr['n_success']}/{_mr['n_iter']} "
                                f"iterations succeeded "
                                f"({_mr['wall_seconds']:.1f} s)")
                            # Result table: param | unit | nominal | MC σ
                            # | 95 % CI; comparable to the bootstrap table
                            _rows_mc = []
                            _kinds_mc = _mr.get("param_kinds", {}) or {}
                            for _n, _se in _mr["mc_se"].items():
                                _fv   = _mr["best_fit"].get(_n, float("nan"))
                                _kind = _kinds_mc.get(_n, "log")
                                if   _kind == "log":       _ust = "log10"
                                elif _kind == "linear_k":  _ust = "linear"
                                else:                       _ust = "mM"
                                _lo = _mr["mc_ci_lo"].get(_n, float("nan"))
                                _hi = _mr["mc_ci_hi"].get(_n, float("nan"))
                                _rows_mc.append({
                                    "parameter": _n,
                                    "unit":      _ust,
                                    "θ_nominal": f"{_fv:+.2e}",
                                    "MC σ":      (f"{_se:.2e}"
                                                   if np.isfinite(_se) else "—"),
                                    "95 % CI (from samples)":
                                        (f"[{_lo:+.2e}, {_hi:+.2e}]"
                                         if np.isfinite(_lo) and np.isfinite(_hi)
                                         else "—"),
                                })
                            st.dataframe(_rows_mc, hide_index=True,
                                          use_container_width=True)
                            st.caption("**Comparing with bootstrap σ above** "
                                       "tells you which uncertainty source "
                                       "dominates: residual noise (bootstrap) "
                                       "or experimental prep error (MC).  If "
                                       "MC σ ≳ bootstrap σ, tightening stock "
                                       "concentrations / volume measurements "
                                       "will improve precision more than "
                                       "noise reduction.")
            _mc_eq_frag()

        # ── v2 Local sensitivity test (Masson/beta) ─────────────────────
        # Paired-difference, no-refit sensitivity probe over a user-
        # selected subset of variables.  Yields f, σ_cond, σ_marg, and
        # the coupling ratio σ_marg/σ_cond (square-root of the VIF) —
        # the latter is the diagnostic answer to "is parameter X
        # individually identified or only through coupling?"
        if fit_stats:
            @_fragment
            def _lst_eq_frag():
                if st.checkbox("🎚️ Show Local sensitivity test (Masson)",
                               value=False, key="_show_lst_eq",
                               help="Probes the local SSR surface around "
                                    "the optimum by perturbing each "
                                    "selected variable by ±δ in log10 "
                                    "space.  The paired-difference "
                                    "construction cancels cross-coupling "
                                    "terms exactly, isolating each "
                                    "variable's intrinsic stiffness.  "
                                    "Compared with the marginal Hessian σ, "
                                    "the ratio diagnoses parameter "
                                    "coupling.  3^N grid evaluations — "
                                    "tractable up to ~6–7 selected "
                                    "variables."):
                    _fa_lst = _diag_fa(st.session_state.get("_fit_args_eq"))
                    if not _fa_lst:
                        st.info("Click **Fit** once with the current "
                                "script and data to enable the local "
                                "sensitivity test.")
                    else:
                        # ── Build the full variable list ──────────────
                        _all_K = list(_fa_lst.get("start_logK", {}).keys())
                        _params_lst = _fa_lst.get("params") or {}
                        _all_C = list((_params_lst.get("conc0",       {}) or {}).keys())
                        _all_T = list((_params_lst.get("titrant_mMs", {}) or {}).keys())
                        _has_V0 = "V0_mL" in _params_lst

                        # Sets of fitted parameter names (CHECKED by default)
                        _fit_K = set(_fa_lst.get("fit_keys",         []) or [])
                        _fit_C = set(_fa_lst.get("fit_conc_keys",    []) or [])
                        _fit_T = set(_fa_lst.get("fit_titrant_keys", []) or [])

                        st.caption("Pick the variables to include in the "
                                   "test. Fitted parameters are checked "
                                   "by default; any held-fixed quantity "
                                   "can also be tested.")

                        # 3-column checkbox grid, sectioned by type
                        _selected: list = []
                        if _all_K:
                            st.write("**Equilibrium constants**")
                            _cK = st.columns(min(3, max(1, len(_all_K))))
                            for _ii, _n in enumerate(_all_K):
                                if _cK[_ii % len(_cK)].checkbox(
                                        _n, value=(_n in _fit_K),
                                        key=f"_lst_eq_chk_K_{_n}"):
                                    _selected.append(_n)
                        if _all_C:
                            st.write("**Initial concentrations**")
                            _cC = st.columns(min(3, max(1, len(_all_C))))
                            for _ii, _n in enumerate(_all_C):
                                if _cC[_ii % len(_cC)].checkbox(
                                        _n, value=(_n in _fit_C),
                                        key=f"_lst_eq_chk_C_{_n}"):
                                    _selected.append(_n)
                        if _all_T:
                            st.write("**Titrant stock concentrations**")
                            _cT = st.columns(min(3, max(1, len(_all_T))))
                            for _ii, _n in enumerate(_all_T):
                                if _cT[_ii % len(_cT)].checkbox(
                                        f"{_n} (titrant)",
                                        value=(_n in _fit_T),
                                        key=f"_lst_eq_chk_T_{_n}"):
                                    _selected.append(_n)
                        if _has_V0:
                            if st.checkbox("V₀ (initial volume)",
                                            value=False, key="_lst_eq_chk_V0"):
                                _selected.append("V0_mL")

                        # Step + parallel controls
                        _csz, _cpar = st.columns([1, 1])
                        _step_lst = _csz.number_input(
                            "Step δ (log10 units)",
                            min_value=0.001, max_value=1.0,
                            value=0.10, step=0.01, format="%.3f",
                            key="_lst_eq_step")
                        _par_lst = _cpar.checkbox(
                            "Parallel (use all cores)",
                            value=True, key="_lst_eq_par")
                        _hpc_go_lst_eq = hpc_ui.controls("lst_eq", label="the local sensitivity test")

                        # Scaling warning + run button
                        _N = len(_selected)
                        if _N == 0:
                            st.warning("Select at least one variable to "
                                       "include in the test.")
                            _can_run = False
                        else:
                            n_grid = 3 ** _N
                            if _N >= 8:
                                st.error(f"⚠ {_N} variables → 3^{_N} = "
                                          f"{n_grid:,} evaluations. "
                                          "That's likely too slow even "
                                          "parallel. Consider "
                                          "deselecting some.")
                            elif _N >= 6:
                                st.warning(f"⚠ {_N} variables → 3^{_N} = "
                                            f"{n_grid:,} grid points — "
                                            "expect several minutes.")
                            else:
                                st.caption(f"Grid: 3^{_N} = {n_grid:,} "
                                            "RMSE evaluations.")
                            _can_run = True

                        if st.button("Run local sensitivity test",
                                      key="_lst_eq_run", disabled=not _can_run) or (_hpc_go_lst_eq and _can_run):
                            _pb = st.progress(0.0, text="Local sensitivity running…")
                            _cb = lambda i, N: _pb.progress(
                                min(i / max(N, 1), 1.0),
                                text=f"Local sensitivity {i}/{N}")
                            _jobs = -1 if _par_lst else 1
                            # Build sigma_marg dict from Hessian param_errors
                            _pe = fit_stats.get("param_errors", {}) or {}
                            _sigma_marg = {k: float(v) for k, v in _pe.items()
                                            if v is not None and np.isfinite(v)}
                            try:
                                _lr = compute_local_sensitivity(
                                    _fa_lst, _selected,
                                    step=float(_step_lst), n_jobs=_jobs,
                                    sigma_marg=_sigma_marg,
                                    fit_stats=fit_stats,
                                    progress_callback=_cb)
                                if _lr is not None:
                                    st.session_state["_lst_result_eq"] = _lr
                            except hpc_ui.CapturedCall as _hpc_cc:
                                _pb.empty()
                                hpc_ui.handle_capture(_hpc_cc, "lst_eq", "Local sensitivity",
                                                      context={})
                            except Exception as _le:
                                _pb.empty()
                                import traceback as _tb_lst
                                st.error(f"Local sensitivity failed: {_le}")
                                st.caption(_tb_lst.format_exc())
                            else:
                                _pb.empty()

                        hpc_ui.jobs_panel("lst_eq", on_result=_hpc_store_state("_lst_result_eq"))
                        _lr = st.session_state.get("_lst_result_eq")
                        if _lr is not None and not _lr.get("f") and _lr.get("best_message"):
                            st.warning(f"Local sensitivity test: {_lr['best_message']}")
                        if _lr is not None and _lr.get("f"):
                            st.caption(
                                f"{_lr['n_success']}/{_lr['n_grid']} "
                                f"grid points evaluated ({_lr['wall_seconds']:.1f} s, "
                                f"N_data = {_lr['n_data']}, "
                                f"RMSE_opt = {_lr['rmse_opt']:.3e})")
                            _rows = []
                            for _v in _lr["selected_vars"]:
                                _f  = _lr["f"].get(_v, float("nan"))
                                _sc = _lr["sigma_cond"].get(_v, float("nan"))
                                _sm = _lr["sigma_marg"].get(_v, float("nan"))
                                _cp = _lr["coupling"].get(_v, float("nan"))
                                if   np.isnan(_cp):     _cp_tag = "n/a"
                                elif _cp <= 1.5:        _cp_tag = f"{_cp:.2f} ✓"
                                elif _cp <= 3.0:        _cp_tag = f"{_cp:.2f} ⚠"
                                else:                   _cp_tag = f"{_cp:.2f} ⚠⚠"
                                _rows.append({
                                    "variable": _v,
                                    "ξ (% RMSE per ±δ)":
                                        (f"{_f*100:+.2f}%"
                                          if np.isfinite(_f) else "—"),
                                    "σ_cond (log10)":
                                        (f"{_sc:.2e}" if np.isfinite(_sc) else "—"),
                                    "σ_marg (Hessian, log10)":
                                        (f"{_sm:.2e}" if np.isfinite(_sm) else "—"),
                                    "Coupling factor":
                                        _cp_tag,
                                })
                            st.dataframe(_rows, hide_index=True,
                                          use_container_width=True)
                            st.caption(
                                "**σ_cond** is the conditional CI half-width "
                                "(other variables held fixed); **σ_marg** is "
                                "the marginal Hessian σ (other variables "
                                "free to refit).  **Coupling factor** = "
                                "σ_marg / σ_cond ≥ 1: ≤ 1.5 means the "
                                "variable is independently constrained by "
                                "the data; 1.5–3.0 indicates mild coupling; "
                                "> 3 signals that the variable is mostly "
                                "identified through coupling with another, "
                                "and its individual value should be "
                                "interpreted cautiously.")
                            # Per-variable expandable: show each of the
                            # 2·3^(N-1) paired differences so the user
                            # can see whether the mean f comes from
                            # uniformly small diffs (truly insensitive)
                            # or from sign-flipping diffs averaging to
                            # zero (non-quadratic / off-optimum anchor).
                            _drecs = _lr.get("diff_records", {})
                            if _drecs:
                                with st.expander("Show individual paired differences"):
                                    for _v in _lr["selected_vars"]:
                                        _records = _drecs.get(_v, [])
                                        if not _records: continue
                                        st.markdown(f"**{_v}** — "
                                                    f"{len(_records)} paired diffs, "
                                                    f"mean = {_lr['f'].get(_v, float('nan'))*100:+.3f}%")
                                        _drows = []
                                        for _rec in _records:
                                            _sign_lbl = ("+δ" if _rec["sign"] > 0 else "−δ")
                                            _other_str = ", ".join(
                                                f"{k}={v}" for k, v in _rec["others"].items())
                                            _drows.append({
                                                f"sign of δ_{_v}": _sign_lbl,
                                                "other variables": _other_str or "—",
                                                "RMSE (unp.)":  f"{_rec.get('rmse_unp', float('nan')):.6e}",
                                                "RMSE (pert.)": f"{_rec.get('rmse_pert', float('nan')):.6e}",
                                                "diff r (%)": f"{_rec['diff']*100:+.4f}%",
                                            })
                                        st.dataframe(_drows, hide_index=True,
                                                      use_container_width=True)
            _lst_eq_frag()

        # ── v2 model comparison (AIC / BIC / F-test / Akaike weights) ────
        if fit_stats:
            @_fragment
            def _cmp_eq_frag():
                if st.checkbox("⚖️ Show model comparison (AIC / BIC / F-test)",
                               value=False, key="_show_cmp_eq"):
                    st.caption(
                        "Fit an alternative reaction model on the same experimental "
                        "data and rank the two by information criteria.  Akaike "
                        "weights translate ΔAIC into an evidence ratio "
                        "(w_A / w_B = exp(ΔAIC / 2)).  Tick **Nested?** when the "
                        "simpler model is a special case of the more complex one "
                        "(e.g. 1:1 vs 1:2 binding) to enable the F-test.")
                    _fa_eq = _diag_fa(st.session_state.get("_fit_args_eq"))
                    _alt_script = st.text_area(
                        "Alternative model script",
                        height=180, key="_alt_script_text",
                        placeholder=(
                            "$concentrations\n"
                            "G0 = 1.00 mM\n\n"
                            "$volumes\nV0 = 500 uL\n\n"
                            "$titrant\nHt = 10.0 mM\n\n"
                            "$reactions\n"
                            "G + H = GH;     log K1 = 4.0\n"
                            "GH + H = GH2;   log K2 = 2.0\n\n"
                            "$plot\nxmax = 3.00\nx = H0/G0\ny = G, H, GH, GH2"))
                    _nested = st.checkbox(
                        "Nested? (enables F-test)", value=True, key="_alt_nested")
                    if st.button("Fit alternative & compare", key="_alt_compare_btn"):
                        if not _fa_eq:
                            st.warning("Click **Fit** with the current script first.")
                        elif not _alt_script.strip():
                            st.warning("Paste an alternative reaction model above.")
                        else:
                            try:
                                from equilibrist_curve import _find_maxEquiv as _fmE
                                _alt_p = parse_script(_alt_script)
                                _alt_net = build_network(_alt_p)
                                _alt_cv = {k[:-1] if k.endswith('0') else k: float(v)
                                           for k, v in _alt_p['concentrations'].items()}
                                _alt_tm = {k[:-1] if (k.endswith('t') or k.endswith('0')) else k: float(v)
                                           for k, v in _alt_p['titrant'].items()}
                                _alt_V0 = list(_alt_p['volumes'].values())[0]
                                _alt_xmax = float(_alt_p.get('plot_xmax', 3.0))
                                _alt_pri  = list(_alt_p['concentrations'].keys())[0]
                                _alt_pri_r = _alt_pri[:-1] if _alt_pri.endswith('0') else _alt_pri
                                _alt_xexpr = _alt_p['plot_x_expr']
                                _alt_maxEq = _fmE(_alt_xexpr, _alt_xmax, _alt_p, _alt_cv, float(_alt_V0),
                                                  _alt_net['titrant_free_names'],
                                                  _alt_net['titrant_keys'], _alt_tm,
                                                  _alt_p.get('titrant_ratios', {}) or {},
                                                  _alt_p['titrant_is_solid'], _alt_pri_r)
                                _alt_params = {
                                    "conc0": _alt_cv, "V0_mL": float(_alt_V0),
                                    "titrant_name": _alt_net.get('titrant_name', ''),
                                    "titrant_key":  _alt_net.get('titrant_key',  ''),
                                    "titrant_free_names": _alt_net['titrant_free_names'],
                                    "titrant_keys": _alt_net['titrant_keys'],
                                    "titrant_mMs": _alt_tm,
                                    "titrant_ratios": _alt_p.get('titrant_ratios', {}) or {},
                                    "titrant_is_solid": _alt_p["titrant_is_solid"],
                                    "maxEquiv": _alt_maxEq, "nPts": 100,
                                    "primary_component": _alt_pri_r,
                                }
                                _alt_logK = {e['kname']: float(e['logK'])
                                             for e in _alt_p['equilibria']}
                                _alt_keys = list(_alt_logK.keys())
                                with st.spinner("Fitting alternative model…"):
                                    if _fa_eq["use_spectra_fit"]:
                                        _ok2, _f2, _s2, _m2 = fit_spectra(
                                            _alt_p, _alt_net, _fa_eq["spectra_data_fit"],
                                            _alt_params, _alt_logK, _alt_keys, _alt_xexpr,
                                            _fa_eq["wl_min"], _fa_eq["wl_max"],
                                            _fa_eq["tolerance"], _fa_eq["maxiter"],
                                            constraints=_alt_p.get('constraints', []),
                                            allow_negative_eps=_fa_eq.get("allow_neg_eps", False))
                                    elif _fa_eq["use_nmr_fit"] and _fa_eq["nmr_mode"] == "shift":
                                        _ok2, _f2, _s2, _m2 = fit_nmr_shifts(
                                            _alt_p, _alt_net, _fa_eq["nmr_data_fit"],
                                            _alt_params, _alt_logK, _alt_keys, _alt_xexpr,
                                            _fa_eq["tolerance"], _fa_eq["maxiter"],
                                            constraints=_alt_p.get('constraints', []))
                                    elif _fa_eq["use_nmr_fit"] and _fa_eq["nmr_mode"] == "integration":
                                        _ok2, _f2, _s2, _m2 = fit_nmr_integration(
                                            _alt_p, _alt_net, _fa_eq["nmr_data_fit"],
                                            _alt_params, _alt_logK, _alt_keys, _alt_xexpr,
                                            _fa_eq["tolerance"], _fa_eq["maxiter"],
                                            constraints=_alt_p.get('constraints', []))
                                    elif _fa_eq["use_nmr_fit"] and _fa_eq["nmr_mode"] == "mixed":
                                        _ok2, _f2, _s2, _m2 = fit_nmr_mixed(
                                            _alt_p, _alt_net, _fa_eq["nmr_data_fit"],
                                            _alt_params, _alt_logK, _alt_keys, _alt_xexpr,
                                            _fa_eq["tolerance"], _fa_eq["maxiter"],
                                            constraints=_alt_p.get('constraints', []))
                                    else:
                                        _ok2, _f2, _s2, _m2 = fit_parameters(
                                            _alt_p, _alt_net, _fa_eq["exp_data_fit"] or {},
                                            _alt_params, _alt_logK, _alt_keys, _alt_xexpr,
                                            tolerance=_fa_eq["tolerance"],
                                            maxiter=_fa_eq["maxiter"],
                                            constraints=_alt_p.get('constraints', []))
                                if not _ok2:
                                    st.warning(f"Alternative model did not converge: {_m2}")
                                else:
                                    augment_stats(_s2)
                                    st.session_state["_alt_stats"]  = _s2
                                    st.session_state["_alt_fitted"] = dict(_f2)
                            except Exception as _ce:
                                import traceback as _tb_c
                                st.error(f"Model comparison failed: {_ce}")
                                st.caption(_tb_c.format_exc())
                    # Render last comparison if available
                    _s2 = st.session_state.get("_alt_stats")
                    _f2 = st.session_state.get("_alt_fitted")
                    if _s2 and fit_stats.get("aic") is not None and _s2.get("aic") is not None:
                        cmp = compare_models(fit_stats, _s2, nested=_nested,
                                             label_a="Current", label_b="Alternative")
                        st.markdown(f"**{cmp['interpretation']}**")
                        _kA = fit_stats.get('n_params', 0) or 0   # = p (manuscript convention)
                        _kB = _s2.get('n_params', 0) or 0         # = p (manuscript convention)
                        _crows = [
                            {"Metric": "ℓ_max",
                             "Current":     f"{fit_stats.get('log_likelihood', float('nan')):.3f}",
                             "Alternative": f"{_s2.get('log_likelihood',  float('nan')):.3f}",
                             "Δ (Alt − Cur)": f"{_s2.get('log_likelihood', 0)-fit_stats.get('log_likelihood', 0):+.3f}"},
                            {"Metric": "SSR",
                             "Current":     f"{fit_stats.get('ssr', float('nan')):.5g}",
                             "Alternative": f"{_s2.get('ssr',  float('nan')):.5g}",
                             "Δ (Alt − Cur)": f"{_s2.get('ssr', 0)-fit_stats.get('ssr', 0):+.5g}"},
                            {"Metric": "p (fitted)",
                             "Current": f"{_kA}", "Alternative": f"{_kB}",
                             "Δ (Alt − Cur)": f"{_kB - _kA:+d}"},
                            {"Metric": "AIC",
                             "Current":     f"{cmp['aic_a']:.2f}",
                             "Alternative": f"{cmp['aic_b']:.2f}",
                             "Δ (Alt − Cur)": f"{cmp['delta_aic']:+.2f}"},
                            {"Metric": "AICc",
                             "Current":     f"{cmp['aicc_a']:.2f}",
                             "Alternative": f"{cmp['aicc_b']:.2f}",
                             "Δ (Alt − Cur)": f"{cmp['delta_aicc']:+.2f}"},
                            {"Metric": "BIC",
                             "Current":     f"{cmp['bic_a']:.2f}",
                             "Alternative": f"{cmp['bic_b']:.2f}",
                             "Δ (Alt − Cur)": f"{cmp['delta_bic']:+.2f}"},
                        ]
                        st.dataframe(_crows, hide_index=True)
                        st.markdown(
                            f"**Akaike weights**: "
                            f"Current = **{cmp['weight_a']:.3f}** · "
                            f"Alternative = **{cmp['weight_b']:.3f}**  \n"
                            f"_(weight ≈ probability this model is the better of the two)_")
                        if _nested and np.isfinite(cmp['f_statistic']):
                            _dp = abs(_kB - _kA)
                            if cmp['f_statistic'] == 0.0 and cmp['f_p_value'] == 1.0:
                                st.markdown(
                                    f"**F-test** (Δp = {_dp}): the more complex model "
                                    "does *not* reduce SSR → no evidence whatsoever "
                                    "for the additional parameter(s).")
                            else:
                                st.markdown(
                                    f"**F-test** (Δp = {_dp}): "
                                    f"F = {cmp['f_statistic']:.3f}, "
                                    f"p = {cmp['f_p_value']:.4g}")
                        # Show alternative model's fitted parameters
                        if _f2:
                            st.markdown("**Alternative model fitted constants:**")
                            _alt_rows = []
                            for n_alt, v_alt in _f2.items():
                                _e_alt = _s2.get('param_errors', {}).get(n_alt)
                                _alt_rows.append({
                                    "Parameter": n_alt,
                                    "log K": f"{v_alt:+.4f}",
                                    "± σ (Hessian)": (f"{_e_alt:.4f}"
                                                      if _e_alt is not None and np.isfinite(_e_alt)
                                                      else "—"),
                                })
                            st.dataframe(_alt_rows, hide_index=True)

            _cmp_eq_frag()

        # ── Fitted equilibrium constants ─────────────────────────
        param_values    = fit_stats.get("param_values", {})
        param_errors    = fit_stats.get("param_errors", {})
        fitted_concs_eq = fit_stats.get("fitted_concs", {})
        fitted_tits_eq  = fit_stats.get("fitted_titrants", {})
        if param_values or fitted_concs_eq or fitted_tits_eq:
            st.write("**Fitted constants:**")
            rows = []
            # log-space parameters (K values)
            _err_notes = fit_stats.get("param_error_notes", {}) or {}
            def _nd_text(kname, val, log_space=True):
                """Cell text for an undetermined error."""
                if kname not in param_errors:
                    return "n/a"
                _nt = _err_notes.get(kname) or {}
                _kind = _nt.get("kind") if isinstance(_nt, dict) else _nt
                _sg   = _nt.get("sigma") if isinstance(_nt, dict) else None
                if _kind == "upper-limit":
                    if _sg is None: return "n/d (upper limit only)"
                    return (f"≲ {val + _sg:.2f} (1σ)" if log_space
                            else f"≲ {10.0 ** (val + _sg):.2e} (1σ)")
                if _kind == "lower-limit":
                    if _sg is None: return "n/d (lower limit only)"
                    return (f"≳ {val - _sg:.2f} (1σ)" if log_space
                            else f"≳ {10.0 ** (val - _sg):.2e} (1σ)")
                return "n/d"
            for kname, val in param_values.items():
                err     = param_errors.get(kname)
                k_lin   = 10.0 ** val
                err_lin = k_lin * 2.302585 * err if err is not None else None
                _nd = _nd_text(kname, val)
                rows.append({"Parameter": kname,
                             "log P":  f"{val:.2f}",
                             "±log P": f"± {err:.2f}" if err is not None else _nd,
                             "P":      f"{k_lin:.2e}",
                             "±P":     f"± {err_lin:.2e}" if err_lin is not None else _nd_text(kname, val, log_space=False)})
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
                _undet = [k for k, v in param_errors.items() if v is None]
                if _undet:
                    _kind_of = lambda k: ((_err_notes.get(k) or {}).get("kind")
                                          if isinstance(_err_notes.get(k), dict) else _err_notes.get(k))
                    _flat  = [k for k in _undet if _kind_of(k) in (None, "flat", "large")]
                    _upper = [k for k in _undet if _kind_of(k) == "upper-limit"]
                    _lower = [k for k in _undet if _kind_of(k) == "lower-limit"]
                    _parts = []
                    if _flat:
                        _parts.append("the objective is flat along " + ", ".join(_flat) +
                                      " (error > 3 log units) — the fitted value is not "
                                      "meaningful on its own")
                    if _upper:
                        _parts.append(", ".join(_upper) + ": the fit does not get worse when the "
                                      "constant is lowered (it sits at the lower edge of its "
                                      "search range, or the objective is flat there) — the data "
                                      "give only an upper limit; the quoted value is the bound, "
                                      "not a measurement")
                    if _lower:
                        _parts.append(", ".join(_lower) + ": the fit does not get worse when the "
                                      "constant is raised (it sits at the upper edge of its "
                                      "search range, or the objective is flat there) — the data "
                                      "give only a lower limit; the quoted value is the bound, "
                                      "not a measurement")
                    st.caption("n/d = not determined by the data. " + "; ".join(_parts) + ".")

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
        # Show whenever shift data was fitted (shift or mixed mode).
        # Display rule (matches legacy behavior when noref is absent):
        #   • Default (no noref):       show absolute δ = δ_obs(V=0) + dd.
        #   • noref WITH read::         show absolute δ (anchored).
        #   • noref WITHOUT read::      show Δδ relative to math reference;
        #     absolute scale is undetermined (uniform-shift symmetry).
        _nmr_noref   = bool(fit_stats.get("nmr_noref", False))
        _ps_anchored = bool(fit_stats.get("pure_shifts_anchored", False))
        _show_abs    = (not _nmr_noref) or _ps_anchored
        _ps_dfree    = fit_stats.get("delta_free", {}) or {}
        _ps_und      = fit_stats.get("pure_shift_undetermined", {}) or {}
        _ps_mf       = fit_stats.get("pure_shift_maxfrac", {}) or {}
        _ps_pin      = fit_stats.get("pure_shift_pinned", {}) or {}
        if pure_shifts_disp:
            if _show_abs:
                st.write("**Pure-species chemical shifts (ppm):**")
            else:
                st.write("**Pure-species chemical shifts — Δδ relative to math reference (ppm):**")
            all_sp_cols = []
            rows_ps = []
            for tgt, col_dict in pure_shifts_disp.items():
                for col, sp_dict in col_dict.items():
                    row = {"Signal": col}
                    df0 = float(_ps_dfree.get(col, 0.0)) if _show_abs else 0.0
                    for sp, dd_val in sp_dict.items():
                        row[sp] = f"{df0 + dd_val:.4f}"
                        if sp in ((_ps_und.get(tgt) or {}).get(col) or []):
                            row[sp] = f"({row[sp]})"
                        elif (sp not in ((_ps_pin.get(tgt) or {}).get(col) or [])
                              and float(((_ps_mf.get(tgt) or {}).get(col) or {}).get(sp, 1.0)) < 0.10):
                            row[sp] = f"{row[sp]} *"
                        if sp not in all_sp_cols:
                            all_sp_cols.append(sp)
                    rows_ps.append(row)
            if rows_ps:
                df_ps = pd.DataFrame(rows_ps).set_index("Signal")
                ordered = [c for c in all_sp_cols if c in df_ps.columns]
                st.dataframe(df_ps[ordered], width='stretch')
                if _show_abs:
                    st.caption("Each row = one NMR signal; columns = δ of each pure species (ppm). "
                               "( ) = species below 2 % of the pool over this titration: its shift is not "
                               "determined by these data and is held at the free shift (or at the value "
                               "shared from the other tabs).  * = species below 10 % of the pool: its shift "
                               "is poorly determined and may absorb misfit; tie it to a related species in "
                               "$constraints if its value is implausible.")
                else:
                    st.caption("Each row = one NMR signal; columns = Δδ relative to math reference (ppm). "
                               "Add a `read:` anchor in $nmr to recover absolute δ.")

        _render_pure_shift_comparison(fit_stats)

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
