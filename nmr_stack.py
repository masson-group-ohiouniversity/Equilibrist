"""
NMR stack -- from a tree of Bruker 1D experiments to an Equilibrist input file.

    streamlit run nmr_stack.py

Give a folder (any depth) or a zip of it.  Every experiment holding ``fid`` and
``acqus`` is transformed, phased (TopSpin's own phases when present), baseline-
corrected and drawn as a stack in a browser-side canvas (``nmr_workbench/``)
with five modes: zoom, calibrate (click one reference peak, all spectra are
aligned on it), drag a spectrum (manual calibration), signals and phase (live
while you drag).  A signal is one thing followed through the series and gives
both a shift column and an integration column: click it on each spectrum
where it is (or drag a box to click many at once); the multiplet nearest the
click is found, its pattern and J read, its centre is the shift, and its integral
comes from a Lorentzian fitted to each line on a baseline that includes the
solvent tail (a numerical area between extent limits is the fallback).  Tables
below the plot hold the calibration offsets, the signals with their shifts,
patterns and integrals, and the export: the ``.xlsx`` in Equilibrist's $nmr
layout.  Undo steps back
one action; Reset clears them all.  Measurements are made in Python at full
resolution; the browser only picks.
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import nmr_stack_core as core

st.set_page_config(page_title="NMR stack", page_icon="📚", layout="wide")

_WB = components.declare_component(
    "nmr_workbench", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nmr_workbench"))

def _rerun_later():
    """Ask for a rerun at the END of this run.  An immediate st.rerun() would
    stop the script before the widgets further down are drawn, and Streamlit
    forgets the state of every keyed widget a run does not draw (the ppm
    window, the export settings ...)."""
    st.session_state["_rerun_req"] = True


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948", "#7f7f7f", "#17becf", "#bcbd22", "#8c564b"]
# signals: 24 colours before any repeats (the first twelve as they have always been,
# then a darker shade of each), so that two signals on one peak are told apart
SIG_COLORS = ["#e34948", "#7f7f7f", "#17becf", "#bcbd22", "#8c564b", "#2a78d6",
              "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
              "#8f1414", "#2f2f2f", "#0a6f7a", "#6c6d0e", "#4d2a23", "#143f78",
              "#93380f", "#0b6244", "#855a00", "#9a3468", "#004707", "#22195c"]


# ── helpers ─────────────────────────────────────────────────────────────────
def _decimate(x: np.ndarray, y: np.ndarray, n_max: int = 4000):
    """Block min/max decimation that keeps peak tops and dips."""
    n = x.size
    if n <= n_max:
        return x, y
    nb = n_max // 2
    edges = np.linspace(0, n, nb + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i_lo, i_hi = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for i in sorted((i_lo, i_hi)):
            xs.append(x[i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


@st.cache_data(show_spinner=False)
def _scan(root: str, stamp: float):
    return core.find_experiments(root)


def _core_stamp() -> float:
    """Modification time of the processing module: part of every cache key, so
    that a change to the processing can never be served from the cache."""
    try:
        return os.path.getmtime(core.__file__)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=64)
def _process(path: str, settings_key: tuple, stamp: float, core_stamp: float):
    exps = core.find_experiments(path)
    if not exps:
        return None
    s = dict(settings_key)
    sp = core.process_experiment(exps[0], s)
    return {"ppm": sp.ppm, "real": sp.real, "imag": sp.imag, "phc0": sp.phc0,
            "phc1": sp.phc1, "source": sp.source, "raw": sp.raw}


def _spectrum(exp, settings: dict):
    d = _process(exp.path, tuple(sorted(settings.items())), _stamp(exp.path), _core_stamp())
    if d is None:
        return None
    return core.Spectrum(ppm=d["ppm"], real=d["real"], imag=d["imag"], exp=exp,
                         phc0=d["phc0"], phc1=d["phc1"], source=d["source"], raw=d.get("raw"))


def _stamp(path: str) -> float:
    try:
        return os.path.getmtime(os.path.join(path, "fid"))
    except OSError:
        return 0.0


def _fmt_time(h):
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return ""
    if h < 1:
        return f"{h * 60:.0f} min"
    return f"{h:g} h"


# ── session save / load ─────────────────────────────────────────────────────
# Everything except the spectra themselves: calibration, signals with their
# measurements, phase tweaks, the experiments table (labels, x, ticks), the
# processing and display settings, the ppm window.  The spectra are found
# again from the folder (or the zip) the session names.
import json as _json
import datetime as _dt
_SESSION_STATE = ("_offsets", "_signals", "_sig_current", "_tweaks", "_g0", "_g1", "_calib_click", "_sig_cfg",
                  "_next_id", "_fitk_spec")
_SESSION_WIDGETS = ("_use_1r", "_lb", "_zf", "_phsrc", "_bl", "_blm", "_blo", "_bll", "_order", "_norm", "_view",
                    "_offset", "_skew", "_clip", "_flip", "_x_kind", "_stem", "_int_norm_in", "_sig_hw",
                    "_calib_ref_in", "_calib_win_in", "_win_lo_in", "_win_hi_in", "_sig_method_in")

def _jsonable(v):
    """Plain JSON types only: numpy scalars, tuples, NaN, timestamps converted."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)

def _session_payload(root, src_mode, zip_name):
    t = st.session_state.get("_tbl")
    rows = []
    if t is not None:
        for _, r in t.iterrows():
            rows.append({"relpath": os.path.relpath(str(r["path"]), root), "show": bool(r["show"]),
                         "label": str(r["label"]), "time_h": r["time_h"], "x": r["x"]})
    return _jsonable({
        "app": "nmr_stack", "version": 1, "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"mode": "folder" if src_mode == "Folder on this computer" else "zip",
                   "root": root if src_mode == "Folder on this computer" else None, "zip_name": zip_name},
        "state": {k: st.session_state.get(k) for k in _SESSION_STATE if k in st.session_state},
        "widgets": {k: st.session_state.get(k) for k in _SESSION_WIDGETS if k in st.session_state},
        "int_ref": st.session_state.get("_int_ref_in"),
        "table": rows})

# a loaded session is applied in two steps: the settings and the source before any
# widget is drawn (here), the data once the experiments table exists (below)
if "_sess_pending" in st.session_state:
    _p = st.session_state.pop("_sess_pending")
    for _k, _v in (_p.get("widgets") or {}).items():
        if _k in _SESSION_WIDGETS:
            st.session_state[_k] = _v
    for _k in ("_g0", "_g1"):                       # the global phase: sidebar sliders, drawn before step two
        if _k in (_p.get("state") or {}):
            st.session_state[_k] = float(_p["state"][_k])
    _src = _p.get("source") or {}
    if _src.get("mode") == "folder" and _src.get("root") and os.path.isdir(str(_src["root"])):
        st.session_state["_src_mode"] = "Folder on this computer"
        st.session_state["_root_in"] = str(_src["root"])
    st.session_state["_sess_state"] = _p

# ── sidebar: source ─────────────────────────────────────────────────────────
st.sidebar.title("📚 NMR stack")
st.sidebar.caption("Bruker 1D experiments, any folder depth.")

src_mode = st.sidebar.radio("Source", ["Folder on this computer", "Upload a zip"],
                            horizontal=True, key="_src_mode")
root = None
if src_mode == "Folder on this computer":
    root_in = st.sidebar.text_input(
        "Folder", value=st.session_state.get("_root_in", ""), key="_root_in",
        placeholder="/Users/.../MKG02005 - Jar,PdNO3,NaCl_120C_D2O",
        help="Paste the path of a folder; every experiment below it (any depth) is found.")
    if root_in.strip():
        p = os.path.expanduser(root_in.strip().strip('"').strip("'"))
        if os.path.isdir(p):
            root = p
        else:
            st.sidebar.error("That folder does not exist.")
else:
    up = st.sidebar.file_uploader("Zip of the folder", type=["zip"], key="_zip_up")
    if up is not None:
        key = ("_zip_root", up.name, up.size)
        if st.session_state.get("_zip_key") != key:
            st.session_state["_zip_root"] = core.extract_zip(up.getvalue())
            st.session_state["_zip_key"] = key
        root = st.session_state.get("_zip_root")

if not root:
    st.title("NMR stack")
    st.info("Point the sidebar at a folder of Bruker experiments, or upload a zip of one.")
    st.stop()

exps = _scan(root, os.path.getmtime(root))
if not exps:
    st.error("No 1D experiment (a folder holding `fid` + `acqus`) found under that folder.")
    st.stop()

# ── workbench events that touch widgets (phase sliders, ppm window) ─────────
# Data-dependent events (calibration, signals) are handled further down,
# once the spectra exist.  Every event carries a unique ``seq`` and is applied once.
def _wrap180(v):
    return ((float(v) + 180.0) % 360.0) - 180.0

# ── history: every action on the data is preceded by a snapshot, so Undo can
# step back one action per click and Reset returns to the untouched spectra.
import copy as _copy
_STATE_KEYS = ("_offsets", "_signals", "_sig_current", "_tweaks", "_g0", "_g1", "_calib_click", "_tbl", "_tbl_ver",
               "_win_lo_in", "_win_hi_in", "_sig_cfg", "_fitk_spec")

def _snapshot():
    return {k: _copy.deepcopy(st.session_state.get(k)) for k in _STATE_KEYS}

def _snap_equal(a, b):
    """True when two snapshots hold the same state (so that going from one to the
    other would change nothing on screen)."""
    for k in _STATE_KEYS:
        x, y = a.get(k), b.get(k)
        if isinstance(x, pd.DataFrame) or isinstance(y, pd.DataFrame):
            if not (isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame) and x.equals(y)):
                return False
        elif k in ("_win_lo_in", "_win_hi_in") and x is not None and y is not None:
            if abs(float(x) - float(y)) > 1e-6:
                return False
        elif x != y:
            return False
    return True

def _push(kind="edit"):
    """Remember the state before an action.  Consecutive view changes (zoom,
    pan) collapse into one step, so Undo after a scroll goes back to the view
    before the scrolling started rather than one wheel notch."""
    h = st.session_state.setdefault("_hist", [])
    st.session_state["_redo"] = []               # a new action: what was undone can no longer be redone
    if kind == "zoom" and h and h[-1].get("_kind") == "zoom":
        return
    snap = _snapshot(); snap["_kind"] = kind
    h.append(snap)
    del h[:-60]

def _restore(snap):
    """Put back what a snapshot holds.  A key the snapshot does not mention is
    left alone (the experiments table, for one, must survive a Reset)."""
    for k in _STATE_KEYS:
        if k not in snap:
            continue
        v = snap[k]
        if v is None:
            st.session_state.pop(k, None)
        else:
            st.session_state[k] = v
    st.session_state.setdefault("_g0", 0.0); st.session_state.setdefault("_g1", 0.0)
    # the signal settings came back with the snapshot: the widgets that edit them must
    # show the same values, or they would put their own back on the next run
    _c = st.session_state.get("_sig_cfg") or {}
    if "hw" in _c:
        st.session_state["_sig_hw"] = float(_c["hw"])
    st.session_state.pop("_sig_method_in", None)
    _tw = st.session_state.get("_tweaks") or {}
    _cur = _tw.get(st.session_state.get("_tw_sel"), (0.0, 0.0))
    st.session_state["_tw0"], st.session_state["_tw1"] = float(_cur[0]), float(_cur[1])

# undo and redo come from the two buttons in the plot's toolbar (or Ctrl/Cmd-Z there)
_ev0 = st.session_state.get("_wb")
if (isinstance(_ev0, dict) and _ev0.get("seq") and _ev0.get("seq") != st.session_state.get("_wb_done")
        and _ev0.get("type") in ("undo", "redo")):
    st.session_state["_wb_done"] = _ev0["seq"]
    st.session_state["_undo_req" if _ev0["type"] == "undo" else "_redo_req"] = True
if st.session_state.pop("_undo_req", False):
    h = st.session_state.get("_hist") or []
    _cur = _snapshot(); _cur["_kind"] = "edit"
    # a step that would change nothing is skipped: one press of Undo always undoes something
    while h and _snap_equal(h[-1], _cur):
        h.pop()
    if h:
        r = st.session_state.setdefault("_redo", [])
        r.append(_cur); del r[:-60]              # where we are now, to come back to with Redo
        _restore(h.pop())
if st.session_state.pop("_redo_req", False):
    r = st.session_state.get("_redo") or []
    _cur = _snapshot(); _cur["_kind"] = "edit"
    while r and _snap_equal(r[-1], _cur):
        r.pop()
    if r:
        h = st.session_state.setdefault("_hist", [])
        h.append(_cur); del h[:-60]
        _restore(r.pop())
if st.session_state.pop("_reset_req", False):
    _push()
    _restore({"_offsets": {}, "_signals": [], "_sig_current": -1, "_tweaks": {}, "_g0": 0.0, "_g1": 0.0,
              "_calib_click": None, "_fitk_spec": {}})

_ev = st.session_state.get("_wb")
_ev_new = isinstance(_ev, dict) and _ev.get("seq") and _ev.get("seq") != st.session_state.get("_wb_done")
if _ev_new and _ev.get("type") in ("phase", "zoom", "zoom_reset"):
    st.session_state["_wb_done"] = _ev["seq"]
    _ev_new = False
    if _ev["type"] == "phase":
        _push()
        _d0, _d1 = float(_ev.get("p0", 0.0)), float(_ev.get("p1", 0.0))
        _tgt = _ev.get("target", "__all__")
        if _tgt == "__all__":
            st.session_state["_g0"] = _wrap180(st.session_state.get("_g0", 0.0) + _d0)
            st.session_state["_g1"] = float(np.clip(st.session_state.get("_g1", 0.0) + _d1, -360, 360))
        else:
            _tw = st.session_state.setdefault("_tweaks", {})
            _cur = _tw.get(_tgt, (0.0, 0.0))
            _tw[_tgt] = (_wrap180(_cur[0] + _d0), float(np.clip(_cur[1] + _d1, -360, 360)))
            if st.session_state.get("_tw_sel") == _tgt:
                st.session_state["_tw0"], st.session_state["_tw1"] = _tw[_tgt]
    elif _ev["type"] == "zoom":
        # a deliberate change of view is an undo step; a nudge too small to see (a trackpad
        # twitch after a click) is not, or Undo would spend a press on taking it back
        _l0, _h0 = st.session_state.get("_win_lo_in"), st.session_state.get("_win_hi_in")
        _nl, _nh = round(float(_ev["lo"]), 3), round(float(_ev["hi"]), 3)
        if _l0 is None or _h0 is None or max(abs(_nl - float(_l0)), abs(_nh - float(_h0))) > 0.02 * abs(float(_h0) - float(_l0)):
            _push("zoom")
        st.session_state["_win_lo_in"] = _nl
        st.session_state["_win_hi_in"] = _nh
    elif _ev["type"] == "zoom_reset":
        _push("zoom")
        st.session_state["_win_reset"] = True

# ── sidebar: processing ─────────────────────────────────────────────────────
st.sidebar.subheader("Processing")
use_1r = st.sidebar.checkbox("Use TopSpin's processed spectra (1r) as they are", value=False,
                             key="_use_1r",
                             help="Skip the FID processing and read the spectrum TopSpin "
                                  "wrote.  Phasing cannot be changed in that case.")
lb = st.sidebar.number_input("Line broadening LB [Hz]", 0.0, 20.0, 0.3, 0.1, key="_lb",
                             disabled=use_1r)
zf = st.sidebar.select_slider("Zero-fill factor", [1, 2, 4], value=2, key="_zf", disabled=use_1r)
phase_source = st.sidebar.radio(
    "Phase", ["TopSpin values", "Automatic", "Manual only"], key="_phsrc", disabled=use_1r,
    help="TopSpin values: the PHC0/PHC1 saved with each experiment (automatic when "
         "an experiment has none).  Automatic: negative-area minimisation -- with a "
         "dominant solvent peak the first-order term is poorly defined, finish with "
         "the sliders.  Manual only: start from zero and use the sliders.")
_src_map = {"TopSpin values": "topspin", "Automatic": "auto", "Manual only": "manual"}
st.sidebar.markdown("**Additional phase, all spectra**")
st.session_state.setdefault("_g0", 0.0); st.session_state.setdefault("_g1", 0.0)
g0 = st.sidebar.slider("PHC0 (°)", -180.0, 180.0, step=0.5, key="_g0", disabled=use_1r)
g1 = st.sidebar.slider("PHC1 (°)", -360.0, 360.0, step=1.0, key="_g1", disabled=use_1r)
baseline = st.sidebar.checkbox("Baseline correction", value=True, key="_bl")
bl_method = st.sidebar.radio("Baseline method", ["asymmetric least squares", "polynomial"], key="_blm",
                             disabled=not baseline, horizontal=True,
                             help="Asymmetric least squares hugs the floor of the spectrum and follows a "
                                  "rolling baseline while leaving broad humps alone (recommended). The "
                                  "polynomial is the classic whole-spectrum fit.")
if bl_method == "polynomial":
    bl_order = st.sidebar.slider("Polynomial order", 0, 8, 3, key="_blo", disabled=not baseline)
    bl_lam = 1e4
else:
    bl_order = 3
    bl_lam = 10.0 ** st.sidebar.slider("Stiffness (log10)", 3.0, 9.0, 4.0, 0.5, key="_bll", disabled=not baseline,
                                       help="4 follows the curved foot of a huge solvent signal and leaves "
                                            "humps a tenth of a ppm wide alone. Lower follows faster wiggles "
                                            "and starts eating broad signals; higher is closer to a gently "
                                            "curved line.")
_blm = "poly" if bl_method == "polynomial" else "als"

with st.sidebar.expander("Per-spectrum phase tweak"):
    tweak_names = [e.label for e in exps]
    tw_sel = st.selectbox("Spectrum", ["(none)"] + tweak_names, key="_tw_sel")
    tweaks = st.session_state.setdefault("_tweaks", {})
    if tw_sel != "(none)":
        if st.session_state.get("_tw_sel_prev") != tw_sel:      # selection changed: load its tweak
            cur = tweaks.get(tw_sel, (0.0, 0.0))
            st.session_state["_tw0"], st.session_state["_tw1"] = float(cur[0]), float(cur[1])
            st.session_state["_tw_sel_prev"] = tw_sel
        t0 = st.slider("PHC0 tweak (°)", -180.0, 180.0, step=0.5, key="_tw0")
        t1 = st.slider("PHC1 tweak (°)", -360.0, 360.0, step=1.0, key="_tw1")
        tweaks[tw_sel] = (t0, t1)
        if st.button("Reset this spectrum", key="_tw_reset"):
            tweaks.pop(tw_sel, None)
            st.session_state["_tw0"], st.session_state["_tw1"] = 0.0, 0.0
            _rerun_later()
    if tweaks:
        st.caption("Tweaked: " + ", ".join(f"{k} ({v[0]:+.1f}°, {v[1]:+.1f}°)"
                                           for k, v in tweaks.items() if v != (0.0, 0.0)))

settings = {"lb_hz": float(lb), "zero_fill": int(zf), "phase_source": _src_map[phase_source],
            "phc0": 0.0, "phc1": 0.0, "baseline": False, "baseline_order": int(bl_order),
            "use_1r": bool(use_1r)}


# ── experiments table ───────────────────────────────────────────────────────
st.title("NMR stack")
st.caption(f"{len(exps)} experiments under `{root}`")

tbl_key = ("_tbl", root, tuple(e.path for e in exps))
if st.session_state.get("_tbl_key") != tbl_key or "_tbl" not in st.session_state:
    rows = []
    for e in exps:
        t_h = core.reaction_time_h(e.title)
        x_t = core.x_from_title(e.title)
        rows.append({"show": True, "label": e.label, "time_h": t_h,
                     "x": x_t if x_t is not None else (t_h * 3600.0 if t_h is not None else None),
                     "folder": os.path.dirname(e.relpath), "expno": e.expno,
                     "NS": e.ns, "RG": e.rg, "acquired": e.date,
                     "title": e.title, "path": e.path})
    st.session_state["_tbl"] = pd.DataFrame(rows)
    st.session_state["_tbl_key"] = tbl_key
    # a new folder: forget the workbench state of the previous one
    for k in ("_offsets", "_signals", "_sig_current", "_calib_ref", "_calib_win", "_hist", "_redo", "_calib_click", "_fitk_spec"):
        st.session_state.pop(k, None)

# second step of a loaded session: the data, now that the table exists
if "_sess_state" in st.session_state:
    _p = st.session_state.pop("_sess_state")
    for _k in _SESSION_STATE:
        if _k in (_p.get("state") or {}) and _k not in ("_g0", "_g1"):
            st.session_state[_k] = _p["state"][_k]
    _names = [sg.get("name") for sg in (st.session_state.get("_signals") or [])]
    if _p.get("int_ref") in _names:
        st.session_state["_int_ref_in"] = _p["int_ref"]
    _t = st.session_state["_tbl"].copy()
    _rows = {r.get("relpath"): r for r in (_p.get("table") or [])}
    _hit = 0
    for _i in range(len(_t)):
        _r = _rows.get(os.path.relpath(str(_t.at[_i, "path"]), root))
        if _r is None:
            continue
        _hit += 1
        _t.at[_i, "show"] = bool(_r.get("show", True))
        _t.at[_i, "label"] = str(_r.get("label", _t.at[_i, "label"]))
        _t.at[_i, "time_h"] = _r.get("time_h") if _r.get("time_h") is not None else np.nan
        _t.at[_i, "x"] = _r.get("x") if _r.get("x") is not None else np.nan
    st.session_state["_tbl"] = _t
    st.session_state["_tbl_ver"] = st.session_state.get("_tbl_ver", 0) + 1
    st.session_state["_hist"] = []
    st.session_state["_redo"] = []
    st.session_state["_sess_msg"] = (f"Session loaded: {len(st.session_state.get('_signals') or [])} signal(s), "
                                     f"{len(st.session_state.get('_offsets') or {})} calibrated spectra, "
                                     f"{_hit} of {len(_t)} table rows matched.")

with st.expander("Experiments (tick to show, edit labels, reaction times and the x column)", expanded=False):
    st.caption("**x** is what goes into column A of the Equilibrist data file: time in seconds "
               "for kinetics (filled from the reaction time parsed from the title), or the volume "
               "added in mL for a titration (filled when the title holds a value like 0.10mL).")
    edited = st.data_editor(
        st.session_state["_tbl"],
        column_config={
            "show": st.column_config.CheckboxColumn("show"),
            "label": st.column_config.TextColumn("label"),
            "time_h": st.column_config.NumberColumn("time [h]", format="%.3f"),
            "x": st.column_config.NumberColumn("x (s or mL)", format="%.4f"),
            "acquired": st.column_config.DatetimeColumn("acquired", format="YYYY-MM-DD HH:mm"),
            "path": None,
        },
        disabled=["folder", "expno", "NS", "RG", "acquired", "title"],
        hide_index=True, width="stretch", key=f"_tbl_editor_{st.session_state.get('_tbl_ver', 0)}")
    st.session_state["_tbl"] = edited
    if st.button("Fill x from the reaction times (hours → seconds)", key="_fill_x"):
        _push()
        t = st.session_state["_tbl"].copy()
        t["x"] = [float(v) * 3600.0 if pd.notna(v) else None for v in t["time_h"]]
        st.session_state["_tbl"] = t
        st.session_state["_tbl_ver"] = st.session_state.get("_tbl_ver", 0) + 1
        _rerun_later()

tbl = st.session_state["_tbl"]
by_path = {e.path: e for e in exps}

# ── display controls ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1, 1, 1, 1, 0.8])
c6.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)      # level with the boxes beside it
if c6.button("⟲ Reset all", key="_reset_btn", width="stretch",
             help="Remove every action on the data: calibration, signals, phase tweaks. "
                  "Processing settings stay. Undo, in the plot's toolbar, brings it back."):
    st.session_state["_reset_req"] = True
    _rerun_later()
order = c1.selectbox("Order", ["acquisition time", "reaction time", "folder / expno", "table order"],
                     key="_order")
norm = c2.selectbox("Normalise", ["each to its maximum", "per scan and receiver gain", "none"],
                    key="_norm")
_norm_map = {"each to its maximum": "max", "per scan and receiver gain": "ns_rg", "none": "none"}
view = c3.selectbox("View", ["stack", "overlay", "map (ppm × spectrum)"], key="_view")
offset = c4.slider("Stack offset", 0.0, 2.0, 0.6, 0.05, key="_offset",
                   help="Vertical spacing between traces.")
skew = c5.slider("Skew (ppm per trace)", -0.5, 0.5, 0.0, 0.01, key="_skew")

sel_rows = tbl[tbl["show"]]
if sel_rows.empty:
    st.warning("No experiment ticked in the table.")
    st.stop()

# processed spectra (cached per experiment and settings)
tweaks = st.session_state.setdefault("_tweaks", {})
specs = []
for _, r in sel_rows.iterrows():
    e = by_path.get(r["path"])
    if e is None:
        continue
    sp = _spectrum(e, settings)
    if sp is None:
        continue
    d0, d1 = float(g0), float(g1)
    t = tweaks.get(e.label)
    if t:
        d0 += t[0]; d1 += t[1]
    if sp.source == "fid" and (d0 or d1 or baseline):
        sp = core.rephase(sp, d0, d1, baseline=bool(baseline), order=int(bl_order), method=_blm, lam=float(bl_lam))
    elif sp.source == "1r" and baseline:
        sp = core.Spectrum(ppm=sp.ppm, real=core.apply_baseline(sp.real, _blm, int(bl_order), float(bl_lam)),
                           imag=sp.imag, exp=sp.exp, phc0=sp.phc0, phc1=sp.phc1, source=sp.source)
    specs.append((r, sp))

if not specs:
    st.error("Nothing could be processed.")
    st.stop()

def _sort_key(item):
    r, sp = item
    if order == "acquisition time":
        return (sp.exp.date or datetime.max)
    if order == "reaction time":
        t = r["time_h"]
        return (1, 0.0) if t is None or (isinstance(t, float) and np.isnan(t)) else (0, float(t))
    if order == "folder / expno":
        return core._natural_key(sp.exp.relpath)
    return 0
if order != "table order":
    specs.sort(key=_sort_key)
labels = [str(r["label"]) for r, _ in specs]

# ── workbench state ─────────────────────────────────────────────────────────
offsets = st.session_state.setdefault("_offsets", {})      # label -> ppm shift (calibration)
signals = st.session_state.setdefault("_signals", [])      # see _new_signal()
calib_ref = st.session_state.setdefault("_calib_ref", 4.79)
calib_win = st.session_state.setdefault("_calib_win", 0.15)
sig_cfg = st.session_state.setdefault("_sig_cfg", {"hw": 0.05, "frac": 1.0, "margin": 0.0, "fit_k": 6.0,
                                                    "method": "sum"})
if float(sig_cfg.get("frac", 1.0)) >= 3.0:          # sessions saved with the old 3 % default
    sig_cfg["frac"] = 1.0
if "method" not in sig_cfg:
    # work begun (or a session saved) before the integral became a plain sum: it used a
    # lineshape fit and a local baseline under every signal.  Bring it to the new default.
    sig_cfg["method"] = "sum"
    for _sg in signals:
        _sg["baseline"] = False
    if signals:
        st.session_state["_sig_notice"] = (
            "Integrals are now the plain sum of the points between the limits, on the baseline-corrected "
            "spectrum, as in TopSpin and Mnova. The lineshape fit and the local baseline that this work was "
            "using have been switched off; both remain available below, per signal or for all.")
fitk_spec = st.session_state.setdefault("_fitk_spec", {})   # label -> window of that spectrum alone
st.session_state.setdefault("_next_id", 1)
st.session_state.setdefault("_sig_current", -1)            # index of the signal the next click adds to

def _new_id():
    # never an id already in use, whatever the counter says (a loaded session
    # used to restart the counter and hand out an id twice)
    used = [int(g["id"]) for g in signals if g.get("id") is not None]
    st.session_state["_next_id"] = max([int(st.session_state.get("_next_id", 1))] + used) + 1
    return int(st.session_state["_next_id"])

def _heal_signals():
    """Every signal has its own id and its own name.  Two signals sharing an id
    cannot be told apart by the signals table, and two sharing a name give two
    identical columns in the export."""
    seen = set()
    for sg in signals:
        if sg.get("id") is None or int(sg["id"]) in seen:
            sg["id"] = _new_id()
        seen.add(int(sg["id"]))
    names = set()
    for sg in signals:
        base = str(sg.get("name") or "").strip() or "S"
        nm, k = base, 2
        while nm in names:
            nm = f"{base}_{k}"; k += 1
        sg["name"] = nm; names.add(nm)

def _new_signal():
    """A signal = one thing followed through the series: a shift column AND an
    integration column.  ``meas`` holds one measurement per spectrum, in the
    UNcalibrated frame (so a later calibration moves everything consistently):
    {pos, pattern, J_hz, lines, lo, hi}."""
    _used = {g.get("name") for g in signals}
    _n = len(signals) + 1
    while f"S{_n}" in _used:
        _n += 1
    sg = {"id": _new_id(), "name": f"S{_n}", "nH": 1, "use_shift": True,
          "use_int": True, "baseline": False, "measure": "multiplet", "integration": "auto", "meas": {}}
    signals.append(sg)
    return sg

_heal_signals()

def _cal(sp):
    return core.shifted(sp, float(offsets.get(sp.exp.label, 0.0)))

cal_specs = [(r, _cal(sp)) for r, sp in specs]
by_label = {sp.exp.label: sp for _, sp in cal_specs}

def _shift_model(mod, d):
    """The fitted model of a signal moved by ``d`` ppm (calibration frame <-> raw frame)."""
    if not mod:
        return None
    out_free = {"free": [[L[0], L[1] + d, L[2], L[3]] for L in mod["free"]]} if mod.get("free") is not None else {}
    return {**out_free,
            "lines": [[L[0], L[1] + d, L[2], L[3]] for L in mod.get("lines", [])],
            "nuis": [[L[0], L[1] + d, L[2], L[3]] for L in mod.get("nuis", [])],
            "x0": mod["x0"] + d, "c": list(mod.get("c", [])),
            "xs": (mod["xs"] + d) if mod.get("xs") is not None else None,
            "lo": mod["lo"] + d, "hi": mod["hi"] + d}

def _measure(sg, label, lo, hi, x0=None):
    """Measure signal ``sg`` on one spectrum inside [lo, hi] (calibrated ppm)."""
    sp = by_label.get(label)
    if sp is None:
        return
    m = core.measure_signal(sp, lo, hi, x0=x0, measure=sg.get("measure", "multiplet"),
                            frac=float(sig_cfg.get("frac", 3.0)) / 100.0, margin=float(sig_cfg.get("margin", 0.0)),
                            fit_k=_fit_k(label))
    off = float(offsets.get(label, 0.0))
    if m is None:
        sg["meas"].pop(label, None)
        return
    old = sg["meas"].get(label) or {}
    sg["meas"][label] = {"pos": m["pos"] - off, "pattern": m["pattern"], "J_hz": m["J_hz"],
                         "lines": [float(v - off) for v in m["lines"]], "fwhm_hz": m.get("fwhm_hz"),
                         "lo": m["lo"] - off, "hi": m["hi"] - off, "x0": (x0 - off) if x0 is not None else None,
                         "box": (lo - off, hi - off),
                         "fit_area": m.get("fit_area"), "fit_fwhm_hz": m.get("fit_fwhm_hz"),
                         "fit_rmse": m.get("fit_rmse"), "fit_eta": m.get("fit_eta"),
                         "fit_rejected": int(m.get("fit_rejected") or 0),
                         "fit_nuisance": int(m.get("fit_nuisance") or 0), "fit_frac": m.get("fit_frac"),
                         "fit_num_expected": m.get("fit_num_expected"),
                         "fit_model": _shift_model(m.get("fit_model"), -off),
                         # manual limits / shift and a dropped integral (raw frame) survive a re-measure
                         "mlo": old.get("mlo"), "mhi": old.get("mhi"), "mpos": old.get("mpos"),
                         "noint": bool(old.get("noint", False)), "nopos": bool(old.get("nopos", False))}

def _fit_k(label):
    """Integration window (± linewidths) for one spectrum: its own value when it was
    set in the single-spectrum view, else the value for all spectra."""
    return float(fitk_spec.get(label, sig_cfg.get("fit_k", 6.0)))

def _remeasure(labels=None, auto_limits=False):
    """Measure again every signal on the given spectra (all when None).  With
    ``auto_limits`` the limits dragged by hand on them are forgotten first."""
    for sg in signals:
        for l, mm in list(sg["meas"].items()):
            if labels is not None and l not in labels:
                continue
            if auto_limits:
                mm["mlo"] = mm["mhi"] = None
            off = float(offsets.get(l, 0.0))
            if mm.get("x0") is not None:
                hw = float(sig_cfg.get("hw", 0.05))
                _measure(sg, l, mm["x0"] + off - hw, mm["x0"] + off + hw, x0=mm["x0"] + off)
            else:
                _measure(sg, l, mm["box"][0] + off, mm["box"][1] + off)

def _remeasure_all():
    for sg in signals:
        for l, mm in list(sg["meas"].items()):
            off = float(offsets.get(l, 0.0))
            if mm.get("x0") is not None:
                hw = float(sig_cfg.get("hw", 0.05))
                _measure(sg, l, mm["x0"] + off - hw, mm["x0"] + off + hw, x0=mm["x0"] + off)
            else:
                _measure(sg, l, mm["box"][0] + off, mm["box"][1] + off)

def _method(sg, mm):
    """How this cell's integral is obtained: 'manual', 'fit' or 'area'."""
    if mm is None:
        return None
    if mm.get("mlo") is not None and mm.get("mhi") is not None:
        return "manual"
    if mm.get("dcv"):
        return "deconv"                          # deconvoluted with the tool in the plot
    choice = sg.get("integration", "auto")
    if choice == "auto":
        # the setting for all signals: the sum of the points between the limits (TopSpin,
        # Mnova), or the lineshape fit when the user asked for it
        choice = "fit" if sig_cfg.get("method", "sum") == "fit" else "area"
    if choice == "fit" and (mm.get("fit_area") is None
                            or (mm.get("fit_rmse") is not None and float(mm["fit_rmse"]) > 0.05)):
        choice = "area"                          # no fit, or one that does not reproduce the data
    return choice

def _sig_value(sg, label, key):
    """A stored value in the calibrated frame; 'lo'/'hi' are the limits actually
    used for the integral (manual ones when set)."""
    mm = sg["meas"].get(label)
    if mm is None:
        return None
    off = float(offsets.get(label, 0.0))
    if key in ("lo", "hi") and (mm.get("noint") or mm.get(key) is None):
        return None
    if key in ("lo", "hi") and _method(sg, mm) == "manual":
        return float(mm["m" + key] + off)
    if key == "pos" and mm.get("nopos"):
        return None                              # the shift was erased, or never picked, on this spectrum
    if key == "pos" and mm.get("mpos") is not None:
        return float(mm["mpos"] + off)
    if key in ("pos", "lo", "hi"):
        return float(mm[key] + off)
    return mm.get(key)

def _sig_area(sg, label):
    """Numerical area over the limits in use (window or manual), flank baseline optional."""
    mm = sg["meas"].get(label)
    if mm is None or label not in by_label or mm.get("noint") or _sig_value(sg, label, "lo") is None:
        return np.nan
    return core.integrate_window(by_label[label], _sig_value(sg, label, "lo"), _sig_value(sg, label, "hi"),
                                 baseline=sg.get("baseline", False))

def _sig_integral(sg, label):
    mm = sg["meas"].get(label)
    if mm is None or label not in by_label or mm.get("noint"):
        return np.nan
    if _method(sg, mm) == "fit":
        return float(mm["fit_area"])
    if _method(sg, mm) == "deconv":
        return float(mm["dcv"]["area"])
    return _sig_area(sg, label)

def _sig_check(sg, label):
    """Relative disagreement between the fit and the numerical area (None when
    either is unavailable); above 15 % the cell deserves a look."""
    mm = sg["meas"].get(label)
    if mm is None or mm.get("fit_area") is None or _method(sg, mm) == "manual":
        return None
    a = _sig_area(sg, label)
    if not np.isfinite(a) or a == 0:
        return None
    expected = mm.get("fit_num_expected")           # the fitted curve integrated like the data
    if not expected or not np.isfinite(expected) or not sg.get("baseline", False) or _method(sg, mm) != "fit":
        return None
    return float(a / float(expected) - 1.0)

def _int_norm_factor(label):
    """Per-spectrum factor applied to every integral: 1 (raw), 1/I(reference
    signal) or 1/max(I) so that the chosen signal, or the largest one, reads 1.0."""
    mode = st.session_state.get("_int_norm", "raw")
    if mode == "raw":
        return 1.0
    vals = {sg["name"]: _sig_integral(sg, label) for sg in signals if sg.get("use_int", True)}
    if mode == "reference":
        v = vals.get(st.session_state.get("_int_ref"))
    else:
        finite = [x for x in vals.values() if x is not None and np.isfinite(x)]
        v = max(finite) if finite else None
    return (1.0 / v) if v is not None and np.isfinite(v) and v != 0 else np.nan

# ── data-dependent workbench events ─────────────────────────────────────────
if _ev_new:
    st.session_state["_wb_done"] = _ev["seq"]
    typ = _ev.get("type")
    if typ != "sig_new":
        _push()  # sig_new pushes itself once the signal exists
    try:
        if typ == "calib":
            lab, x0 = _ev["spec"], float(_ev["ppm"])
            start = labels.index(lab) if lab in labels else 0
            found = core.track_peak([sp for _, sp in cal_specs], start, x0, float(calib_win))
            for lab2, pos in found.items():
                if pos is not None:
                    offsets[lab2] = float(offsets.get(lab2, 0.0) + float(calib_ref) - pos)
            st.session_state["_calib_click"] = float(calib_ref)
        elif typ == "drag_align":
            lab = _ev["spec"]
            offsets[lab] = float(offsets.get(lab, 0.0) + float(_ev["delta"]))
        elif typ == "sig_noint":
            sg = signals[int(_ev["signal"])]
            if _ev["spec"] in sg["meas"]:
                sg["meas"][_ev["spec"]]["noint"] = True
        elif typ == "sig_erase":
            # the eraser: a click on the vertical bar removes the shift of that signal on that
            # spectrum, a click on the range bar its integral; with both gone, nothing is left
            _j, _lab = int(_ev.get("signal", -1)), _ev.get("spec")
            if 0 <= _j < len(signals) and _lab in signals[_j]["meas"]:
                _mm = signals[_j]["meas"][_lab]
                if _ev.get("what") == "int":
                    _mm["noint"] = True; _mm["mlo"] = _mm["mhi"] = None
                else:
                    _mm["nopos"] = True; _mm["mpos"] = None
                if _mm.get("noint") and _mm.get("nopos"):
                    signals[_j]["meas"].pop(_lab, None)
        elif typ == "sig_limits":
            sg = signals[int(_ev["signal"])]
            lab = _ev["spec"]
            if lab in sg["meas"]:
                off = float(offsets.get(lab, 0.0))
                sg["meas"][lab]["mlo"] = float(min(_ev["lo"], _ev["hi"])) - off
                sg["meas"][lab]["mhi"] = float(max(_ev["lo"], _ev["hi"])) - off
        elif typ == "set_x":
            t = st.session_state["_tbl"].copy()
            v = _ev.get("value")
            t.loc[t["label"] == _ev["spec"], "x"] = (None if v is None else float(v))
            st.session_state["_tbl"] = t
            st.session_state["_tbl_ver"] = st.session_state.get("_tbl_ver", 0) + 1
            for r_, _sp in cal_specs:                       # the rows in hand for this run
                if str(r_["label"]) == _ev["spec"]:
                    r_["x"] = None if v is None else float(v)
        elif typ == "sig_deconv":
            # the deconvolute tool: everything in the box is fitted together as a set of
            # lines, on every spectrum the box covers; each signal already measured there
            # takes the lines that are its own, the rest (a shoulder, a neighbour that is
            # not a signal) is fitted too and left out
            blo, bhi = float(min(_ev["lo"], _ev["hi"])), float(max(_ev["lo"], _ev["hi"]))
            _done, _empty = 0, 0
            for lab in _ev.get("specs", []):
                sp_ = by_label.get(lab)
                if sp_ is None:
                    continue
                off = float(offsets.get(lab, 0.0))
                sf_ = sp_.exp.sfo1 or 500.0
                cands = []                                  # (signal, measurement, its reference positions, tolerance)
                for sg in signals:
                    mm = sg["meas"].get(lab)
                    if mm is None:
                        continue
                    refs = [float(v) + off for v in (mm.get("lines") or [])]
                    if mm.get("mpos") is not None:
                        refs.append(float(mm["mpos"]) + off)
                    elif mm.get("pos") is not None:
                        refs.append(float(mm["pos"]) + off)
                    refs = [v for v in refs if blo <= v <= bhi]
                    if not refs:
                        continue
                    tol = max(0.75 * float(mm.get("fwhm_hz") or 0.0) / sf_, 0.005)
                    cands.append((sg, mm, refs, tol))
                if not cands:
                    _empty += 1
                    continue
                res = core.deconvolve_region(sp_, blo, bhi, free_baseline=any(c_[0].get("baseline", False) for c_ in cands))
                if res is None:
                    _empty += 1
                    continue
                owner = []
                for A_, c_, g_, e_ in res["comps"]:
                    best, bd = None, None
                    for k_, (sg, mm, refs, tol) in enumerate(cands):
                        d_ = min(abs(c_ - v) for v in refs)
                        if d_ <= tol and (bd is None or d_ < bd):
                            best, bd = k_, d_
                    owner.append(best)
                kfit = _fit_k(lab)
                for k_, (sg, mm, refs, tol) in enumerate(cands):
                    mine = [q for q, o in zip(res["comps"], owner) if o == k_]
                    if not mine:
                        continue
                    rest = [q for q, o in zip(res["comps"], owner) if o is not None and o != k_]   # other signals' lines
                    free = [q for q, o in zip(res["comps"], owner) if o is None]                     # lines of no signal
                    w_lo = min(q[1] - kfit * 2.0 * q[2] for q in mine)
                    w_hi = max(q[1] + kfit * 2.0 * q[2] for q in mine)
                    # the area of its lines inside its window: what a plain sum over that
                    # window would give if the other signals were not there
                    area = sum(core.pv_area(q[0], q[2], q[3], w_lo, w_hi, q[1]) for q in mine)
                    mm["dcv"] = {"area": float(area), "rmse": float(res["rmse"]), "n": len(mine), "n_all": len(res["comps"]),
                                 "model": _shift_model({"lines": mine, "nuis": rest, "free": free, "x0": res["x0"],
                                                        "c": res["c"], "xs": None, "lo": res["lo"], "hi": res["hi"]}, -off)}
                    mm["lo"], mm["hi"] = float(w_lo - off), float(w_hi - off)
                    mm["mlo"] = mm["mhi"] = None
                    mm["noint"] = False
                    _done += 1
            if not _done:
                st.toast("Nothing to deconvolute there: the box must contain at least one signal that is already "
                         "measured on the spectra it covers (pick it first with a click or a box).", icon="ℹ️")
        elif typ == "fit_window":
            # the window setting in the plot's toolbar.  In the single-spectrum view it
            # applies to that spectrum alone; on the stack to everything.  Either way the
            # limits dragged by hand on the spectra concerned go back to automatic.
            _k = float(np.clip(float(_ev.get("k", 6.0)), 0.5, 30.0))
            _sp = _ev.get("spec")
            if _sp:
                fitk_spec[_sp] = _k
                _remeasure({_sp}, auto_limits=True)
            else:
                sig_cfg["fit_k"] = _k
                fitk_spec.clear()
                _remeasure(None, auto_limits=True)
        elif typ == "sig_rename":
            j = int(_ev.get("signal", -1)); nm = str(_ev.get("name", "")).strip()
            if 0 <= j < len(signals) and nm:
                signals[j]["name"] = nm
        elif typ == "sig_new":
            _push()
            sg = _new_signal()
            st.session_state["_sig_current"] = signals.index(sg)
        elif typ in ("sig_click", "sig_box", "sig_lasso", "sig_line"):
            j = int(_ev.get("signal", -1))
            if 0 <= j < len(signals):
                sg = signals[j]
            else:
                sg = _new_signal()
            st.session_state["_sig_current"] = signals.index(sg)
            # what this selection is for: the shift, the integral, or both (the two toggles
            # in the plot's toolbar).  What it is not for stays as it was on a spectrum
            # already measured, and is left out on a new one.
            _pick, _integ = bool(_ev.get("pick", True)), bool(_ev.get("integ", True))
            _touched = ([_ev.get("spec")] if typ == "sig_click" else
                        list(_ev.get("specs", [])) if typ == "sig_box" else
                        [it.get("spec") for it in _ev.get("items", [])])
            _before = {l_: (bool(sg["meas"][l_].get("noint")), bool(sg["meas"][l_].get("nopos")))
                       for l_ in _touched if l_ in sg["meas"]}
            _dcv_before = {l_: (sg["meas"][l_]["dcv"], sg["meas"][l_].get("lo"), sg["meas"][l_].get("hi"))
                           for l_ in _touched if l_ in sg["meas"] and sg["meas"][l_].get("dcv")}
            if typ == "sig_click":
                lab, x0 = _ev["spec"], float(_ev["ppm"])
                off = float(offsets.get(lab, 0.0))
                if _ev.get("remove"):
                    sg["meas"].pop(lab, None)
                elif _ev.get("exact"):
                    # the shift is exactly where the user clicked; measure the integral if not done yet
                    if lab not in sg["meas"]:
                        hw = float(sig_cfg.get("hw", 0.05))
                        _measure(sg, lab, x0 - hw, x0 + hw, x0=x0)
                    if lab not in sg["meas"]:                       # nothing measurable: shift only
                        sg["meas"][lab] = {"pos": x0 - off, "pattern": "", "J_hz": None, "lines": [],
                                           "lo": None, "hi": None, "x0": x0 - off, "box": (x0 - off, x0 - off),
                                           "fit_area": None, "noint": True, "mlo": None, "mhi": None}
                    sg["meas"][lab]["mpos"] = x0 - off
                    sg["meas"][lab]["nopos"] = False
                else:
                    hw = float(sig_cfg.get("hw", 0.05))
                    _measure(sg, lab, x0 - hw, x0 + hw, x0=x0)
                    sg["meas"][lab].pop("mpos", None) if lab in sg["meas"] else None
                    if lab in sg["meas"]:
                        sg["meas"][lab]["noint"] = False
            elif typ == "sig_line":
                # a straight line between the signal on two spectra: a click on every
                # spectrum in between.  The line gives the slope; the signal is
                # followed from one spectrum to the next (the prediction starts from
                # where it was actually found), so a drift that bends away from the
                # straight line is still tracked.
                hw = float(sig_cfg.get("hw", 0.05))
                items = _ev.get("items", [])
                last = None
                for k, it in enumerate(items):
                    lab = it.get("spec")
                    pred = float(it["ppm"])
                    if last is not None and k > 0:
                        pred = last + (float(it["ppm"]) - float(items[k - 1]["ppm"]))
                    if lab in by_label:
                        _measure(sg, lab, pred - hw, pred + hw, x0=pred)
                        if lab in sg["meas"]:
                            sg["meas"][lab]["noint"] = False
                            last = float(sg["meas"][lab]["pos"]) + float(offsets.get(lab, 0.0))
                            continue
                    last = pred
            elif typ == "sig_lasso":
                # a free shape: every spectrum it covers comes with its own ppm range
                for it in _ev.get("items", []):
                    lab = it.get("spec")
                    if lab in by_label:
                        _measure(sg, lab, float(min(it["lo"], it["hi"])), float(max(it["lo"], it["hi"])))
            else:
                blo, bhi = float(min(_ev["lo"], _ev["hi"])), float(max(_ev["lo"], _ev["hi"]))
                for lab in _ev.get("specs", []):
                    if lab in by_label:
                        _measure(sg, lab, blo, bhi)
            if not _ev.get("remove") and not _ev.get("exact"):
                for l_ in _touched:
                    _mm = sg["meas"].get(l_)
                    if _mm is None:
                        continue
                    _was = _before.get(l_)                     # (noint, nopos) before, None when new
                    _mm["nopos"] = False if _pick else (_was[1] if _was else True)
                    _mm["noint"] = False if _integ else (_was[0] if _was else True)
                    if not _integ and l_ in _dcv_before:       # only the shift was picked: the integral stays as it was
                        _mm["dcv"], _mm["lo"], _mm["hi"] = _dcv_before[l_]
                    if _mm["nopos"] and _mm["noint"]:
                        sg["meas"].pop(l_, None)
    except Exception as _e:
        st.warning(f"Could not apply that action: {_e}")
    _heal_signals()
    cal_specs = [(r, _cal(sp)) for r, sp in specs]
    by_label = {sp.exp.label: sp for _, sp in cal_specs}

# ── ppm window ──────────────────────────────────────────────────────────────
ppm_lo_all = float(min(sp.ppm.min() for _, sp in cal_specs))
ppm_hi_all = float(max(sp.ppm.max() for _, sp in cal_specs))
w1, w2, w3, w4 = st.columns([1, 1, 1, 3.4])
if st.session_state.pop("_win_reset", False) or "_win_lo_in" not in st.session_state:
    st.session_state["_win_lo_in"] = float(max(ppm_lo_all, -0.5))
    st.session_state["_win_hi_in"] = float(min(ppm_hi_all, 10.0))
lo = w1.number_input("ppm from", step=0.05, format="%.3f", key="_win_lo_in")
hi = w2.number_input("ppm to", step=0.05, format="%.3f", key="_win_hi_in")
if hi <= lo:
    lo, hi = min(lo, hi), max(lo, hi) + 0.01
clip_on = w3.checkbox("Clip tall peaks", value=True, key="_clip",
                      help="In the exported figure, traces are cut at 1.2 × the tallest non-solvent peak.")
flip = w3.checkbox("First spectrum on top", value=False, key="_flip",
                   help="Off: the first spectrum of the chosen order sits at the bottom of the stack "
                        "(TopSpin style). On: it sits at the top and the series reads downwards.")
hide_names = w3.checkbox("Hide spectrum names", value=False, key="_hide_names",
                         help="Leave out the title written on each trace; the order and the "
                              "experiments table still identify them.")
w4.caption("Zoom by dragging in 🔍 mode, pan with a two-finger swipe or a right-button drag, "
           "pinch to zoom; double-click to see everything. Press **s** over a spectrum to see it "
           "alone with its integrals drawn, **s** again to return to the stack.")

# ── the workbench ───────────────────────────────────────────────────────────
def _pick_indices(sp, lo, hi, budget):
    idx = np.flatnonzero((sp.ppm >= lo) & (sp.ppm <= hi))
    if idx.size <= budget:
        return idx
    nb = max(1, budget // 2)
    edges = np.linspace(0, idx.size, nb + 1).astype(int)
    out = []
    y = sp.real
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = idx[a:b]
        s = y[seg]
        out.extend(sorted({int(seg[int(np.argmin(s))]), int(seg[int(np.argmax(s))])}))
    return np.asarray(sorted(set(out)), int)

if view == "map (ppm × spectrum)":
    xg = np.linspace(lo, hi, 3000)
    Z = []
    for _, sp in cal_specs:
        y = np.interp(xg, sp.ppm[::-1], sp.real[::-1])
        m = float(np.nanmax(np.abs(y))) or 1.0
        Z.append(y / m)
    fig = go.Figure(go.Heatmap(x=xg, y=labels, z=np.vstack(Z), colorscale="Viridis",
                               colorbar=dict(title="intensity")))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="δ [ppm]", autorange="reversed", range=[hi, lo])
    fig.update_layout(height=max(360, 28 * len(cal_specs) + 120), margin=dict(l=40, r=20, t=30, b=50),
                      template="simple_white")
    st.plotly_chart(fig, width="stretch", key="_map_fig")
else:
    budget = int(min(6000, max(1500, 60000 // max(1, len(cal_specs)))))
    wb_data = []
    for r, sp in cal_specs:
        if sp.raw is None and sp.source == "1r":
            re_, im_ = sp.real, np.zeros_like(sp.real)
        else:
            re_, im_ = (sp.raw if sp.raw is not None else sp.real), sp.imag
        _m = 0.5 * (hi - lo)                    # margin: live panning within it needs no round trip
        idx = _pick_indices(sp, lo - _m, hi + _m, int(budget * 2))
        if idx.size < 2:
            continue
        off = float(offsets.get(sp.exp.label, 0.0))
        wb_data.append({
            "label": sp.exp.label, "off": off,
            "x": [round(float(v), 4) for v in sp.ppm[idx]],
            "re": [float(f"{v:.5g}") for v in sp.real[idx]],        # what is drawn: baseline-corrected
            "re0": [float(f"{v:.5g}") for v in re_[idx]],           # raw real part, for the live phasing preview
            "im": [float(f"{v:.5g}") for v in im_[idx]],
            "ppmFirst": float(sp.ppm[0] - off), "ppmSpan": float((sp.ppm[0] - sp.ppm[-1]) or 1.0),
            "xval": (None if r["x"] is None or (isinstance(r["x"], float) and np.isnan(r["x"])) else float(r["x"]))})
    wb_signals = []
    for i, sg in enumerate(signals):
        per = {}
        for l in sg["meas"]:
            _iv = _sig_integral(sg, l) * _int_norm_factor(l)
            # the straight baseline through the two limits, from the full-resolution data:
            # the canvas only holds a thinned copy and cannot estimate it when zoomed out
            _bhi = _blo = None
            _lo_, _hi_ = _sig_value(sg, l, "lo"), _sig_value(sg, l, "hi")
            if _lo_ is not None and _hi_ is not None and l in by_label:
                if sg.get("baseline", False):
                    _xb, _yb, _bb = core.window_baseline(by_label[l], _lo_, _hi_)
                    if _bb.size >= 2:
                        _bhi, _blo = float(_bb[0]), float(_bb[-1])
                else:
                    _bhi = _blo = 0.0               # the plain sum counts from zero
            per[l] = {"pos": _sig_value(sg, l, "pos"), "lo": _lo_, "hi": _hi_, "bhi": _bhi, "blo": _blo,
                      "manual": _method(sg, sg["meas"][l]) == "manual", "noint": bool(sg["meas"][l].get("noint")),
                      "value": (None if _iv is None or not np.isfinite(_iv) else float(_iv)),
                      "method": {"area": "sum, local baseline" if sg.get("baseline", False) else "sum",
                                 "manual": "manual, local baseline" if sg.get("baseline", False) else "manual"
                                 }.get(_method(sg, sg["meas"][l]), _method(sg, sg["meas"][l])),
                      # the fitted lineshape, drawn in the single-spectrum view when the fit is what is reported
                      "model": (_shift_model(sg["meas"][l].get("fit_model"), float(offsets.get(l, 0.0)))
                                if _method(sg, sg["meas"][l]) == "fit" else
                                _shift_model(sg["meas"][l]["dcv"].get("model"), float(offsets.get(l, 0.0)))
                                if _method(sg, sg["meas"][l]) == "deconv" else None),
                      "lines": [float(v + offsets.get(l, 0.0)) for v in sg["meas"][l].get("lines", [])]}
        wb_signals.append({"name": sg["name"], "color": SIG_COLORS[i % len(SIG_COLORS)], "meas": per})
    _WB(spectra=wb_data, lo=float(lo), hi=float(hi), offset=float(offset) if view == "stack" else 0.0,
        skew=float(skew), flip=bool(flip), names=not bool(hide_names), signals=wb_signals,
        current=int(st.session_state.get("_sig_current", -1)),
        colors=COLORS, full=[ppm_lo_all, ppm_hi_all],
        fit_k=float(sig_cfg.get("fit_k", 6.0)), fit_k_spec={l: float(v) for l, v in fitk_spec.items()},
        n_undo=len(st.session_state.get("_hist") or []), n_redo=len(st.session_state.get("_redo") or []),
        xlabel=("V [mL]" if str(st.session_state.get("_x_kind", "")).startswith("titr") else "t [s]"),
        calib_ppm=st.session_state.get("_calib_click"), key="_wb", default=None)

# ── tables: calibration, signals, export ───────────────────────────────────
tab_cal, tab_sig, tab_exp, tab_proc = st.tabs(
    ["🎯 Calibration", "〰 Signals (shift + integration columns)", "📤 Equilibrist export", "Processing details"])

with tab_cal:
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    st.session_state["_calib_ref"] = cc1.number_input("Reference peak is at [ppm]", value=float(calib_ref),
                                                      step=0.01, format="%.3f", key="_calib_ref_in",
                                                      help="HDO 4.79, TMS/DSS 0.00, CHCl3 7.26 ...")
    st.session_state["_calib_win"] = cc2.number_input("Search window (± ppm)", value=float(calib_win),
                                                      min_value=0.01, max_value=2.0, step=0.01,
                                                      format="%.2f", key="_calib_win_in")
    cc3.caption("Switch the plot to 🎯 mode and click the reference peak on any spectrum. Every "
                "spectrum is aligned on the most prominent peak within the window, following the "
                "peak from one spectrum to the next. Correct an individual spectrum in ↔ mode or "
                "in the table.")
    odf = pd.DataFrame({"spectrum": labels, "offset [ppm]": [float(offsets.get(l, 0.0)) for l in labels]})
    oed = st.data_editor(odf, hide_index=True, width="stretch", key="_off_editor",
                         disabled=["spectrum"],
                         column_config={"offset [ppm]": st.column_config.NumberColumn(format="%.4f")})
    for l, v in zip(oed["spectrum"], oed["offset [ppm]"]):
        v = float(v) if pd.notna(v) else 0.0
        if abs(v - float(offsets.get(l, 0.0))) > 1e-9:
            _push(); offsets[l] = v
            _rerun_later()
    if st.button("Clear calibration", key="_cal_clear"):
        _push(); offsets.clear(); st.session_state.pop("_calib_click", None); _rerun_later()

with tab_sig:
    st.caption("In 〰 mode, click a signal on a spectrum: the multiplet nearest the click is found, its "
               "pattern and J are read, its centre becomes the shift, and the points between the limits "
               "drawn under it are summed to give the integral. Clicks add spectra to the signal shown "
               "in the plot's bar; press **new signal** there to start another. A box drag clicks every "
               "spectrum it covers at once. Drag an end of the bar under a signal to set its integration "
               "limits by hand. Alt/Option-click (Ctrl or Cmd also work) puts the shift exactly at the "
               "cursor; the same on the bar drops the integral and keeps the shift. Shift-click a marker "
               "to remove that spectrum from its signal, whichever signal is selected.")
    _note = st.session_state.pop("_sig_notice", None)
    if _note:
        st.info(_note)
    _mlab = {"sum": "sum of the points between the limits (TopSpin, Mnova)", "fit": "lineshape fit (deconvolution)"}
    if st.session_state.get("_sig_method_in") not in _mlab.values():
        st.session_state["_sig_method_in"] = _mlab["fit" if sig_cfg.get("method", "sum") == "fit" else "sum"]
    _msel = st.radio("Integral", list(_mlab.values()), horizontal=True, key="_sig_method_in",
                     help="Sum: every point between the two limits of a signal is added up, on the "
                          "baseline-corrected spectrum, exactly what TopSpin's integration and Mnova's Sum "
                          "method do. Nothing is modelled, so what the single-spectrum view shades is what is "
                          "counted. Lineshape fit: a pseudo-Voigt per line; useful to separate overlapping "
                          "signals, and to be checked by eye in the single-spectrum view (press s). A signal can "
                          "be set apart from this choice in its row of the table (column integration).")
    _mnew = "fit" if _msel == _mlab["fit"] else "sum"
    if _mnew != sig_cfg.get("method", "sum"):
        _push(); sig_cfg["method"] = _mnew; _rerun_later()
    s1, s5 = st.columns([1, 2.4])
    _hw = s1.number_input("Search half-width (± ppm)", value=float(sig_cfg.get("hw", 0.05)), min_value=0.005,
                          max_value=2.0, step=0.005, format="%.3f", key="_sig_hw",
                          help="How far from a click the signal is looked for.")
    _own = {l: v for l, v in fitk_spec.items() if abs(v - float(sig_cfg.get("fit_k", 6.0))) > 1e-9}
    s5.caption(f"Integration window: from the outer lines of each signal outwards to where the trace is back at "
               f"the baseline ({float(sig_cfg.get('frac', 3.0)):g} % of the signal height, or the noise), at most "
               f"± {float(sig_cfg.get('fit_k', 6.0)):g} linewidths"
               + (f"; {len(_own)} spectr{'um has' if len(_own) == 1 else 'a have'} their own cap" if _own else "")
               + ". Set the cap in the plot's toolbar (**window ±**, in 〰 mode): on the stack it applies to every "
                 "spectrum, in the single-spectrum view (press **s**) to that spectrum alone. A side stops "
                 "earlier only in the valley before a neighbouring signal.")
    if abs(_hw - sig_cfg.get("hw", 0.05)) > 1e-12:
        _push()
        sig_cfg["hw"] = float(_hw)
        if signals:
            _remeasure_all()
        _rerun_later()
    if not signals:
        st.info("No signal yet.")
    else:
        sdf = pd.DataFrame([{"id": sg["id"], "name": sg["name"], "nH": int(sg.get("nH", 1) or 1),
                             "shift": bool(sg.get("use_shift", True)), "integral": bool(sg.get("use_int", True)),
                             "baseline": bool(sg.get("baseline", False)), "measure": sg.get("measure", "multiplet"),
                             "integration": sg.get("integration", "fit"),
                             "spectra": len(sg["meas"])} for sg in signals])
        sed = st.data_editor(sdf, hide_index=True, width="stretch", key="_sig_editor", num_rows="dynamic",
                             disabled=["spectra"],
                             column_config={
                                 "id": None,
                                 "nH": st.column_config.NumberColumn(min_value=1, step=1, help="Proton count, for the integration: line"),
                                 "shift": st.column_config.CheckboxColumn("shift column", help="Export the shift of this signal"),
                                 "integral": st.column_config.CheckboxColumn("integration column", help="Export the integral of this signal"),
                                 "baseline": st.column_config.CheckboxColumn(
                                     "local baseline",
                                     help="Off (default): the plain sum, counted from zero of the baseline-corrected "
                                          "spectrum. On: a straight line through the two limits is subtracted first "
                                          "(TopSpin's bias and slope), for a signal sitting on the tail of a much "
                                          "larger one such as the solvent."),
                                 "integration": st.column_config.SelectboxColumn(
                                     options=["auto", "area", "fit"],
                                     help="auto: what is chosen above for all signals. area: the sum of the points "
                                          "between the limits. fit: a pseudo-Voigt per line, neighbours inside the "
                                          "window fitted too; falls back to the sum where the fit does not reproduce "
                                          "the data. Limits dragged on the plot always give the sum."),
                                 "measure": st.column_config.SelectboxColumn(
                                     options=["multiplet", "apex", "centroid"],
                                     help="multiplet: centre of the pattern (centroid of the lines when they are not "
                                          "evenly spaced). apex: the tallest line. centroid: intensity-weighted centre "
                                          "over the integration limits, for broad humps.")})
        kept = set(int(v) for v in sed["id"] if pd.notna(v))
        _pend = [("del", sg, None) for sg in signals if sg["id"] not in kept]
        for _, row in sed.iterrows():
            if pd.isna(row["id"]):
                continue
            sg = next((g for g in signals if g["id"] == int(row["id"])), None)
            if sg is None:
                continue
            new = {"name": str(row["name"]).strip() or sg["name"],
                   "nH": int(row["nH"]) if pd.notna(row["nH"]) else sg.get("nH", 1),
                   "use_shift": bool(row["shift"]) if pd.notna(row["shift"]) else True,
                   "use_int": bool(row["integral"]) if pd.notna(row["integral"]) else True,
                   "baseline": bool(row["baseline"]) if pd.notna(row["baseline"]) else False,
                   "measure": str(row["measure"]) if pd.notna(row["measure"]) else "multiplet",
                   "integration": str(row["integration"]) if pd.notna(row["integration"]) else "auto"}
            if any(sg.get(k) != v for k, v in new.items()):
                _pend.append(("upd", sg, new))
        if _pend:
            _push()
            for what, sg, val in _pend:
                if what == "del":
                    signals.remove(sg)
                    st.session_state["_sig_current"] = -1
                else:
                    remeasure = val["measure"] != sg.get("measure")
                    sg.update(val)
                    if remeasure:
                        for l, mm in list(sg["meas"].items()):
                            off = float(offsets.get(l, 0.0))
                            if mm.get("x0") is not None:
                                hw = float(sig_cfg.get("hw", 0.05))
                                _measure(sg, l, mm["x0"] + off - hw, mm["x0"] + off + hw, x0=mm["x0"] + off)
                            else:
                                _measure(sg, l, mm["box"][0] + off, mm["box"][1] + off)
            _rerun_later()

        def _fmt_pat(sg, l):
            mm = sg["meas"].get(l)
            if not mm:
                return ""
            j = mm.get("J_hz")
            return (mm.get("pattern") or "") + (f", J = {j:.1f} Hz" if j else "")

        st.markdown("**Chemical shifts [ppm]**")
        st.dataframe(pd.DataFrame({"spectrum": labels, **{
            sg["name"]: [None if _sig_value(sg, l, "pos") is None else round(_sig_value(sg, l, "pos"), 4)
                         for l in labels] for sg in signals}}), hide_index=True, width="stretch")
        st.markdown("**Pattern and coupling** (m = lines not evenly spaced)")
        st.dataframe(pd.DataFrame({"spectrum": labels, **{sg["name"]: [_fmt_pat(sg, l) for l in labels]
                                                          for sg in signals}}), hide_index=True, width="stretch")
        _nm = st.session_state.get("_int_norm", "raw")
        st.markdown("**Integrals** (over the limits drawn at each baseline"
                    + {"raw": ", raw", "reference": f", {st.session_state.get('_int_ref')} = 1.0 in every spectrum",
                       "largest": ", largest = 1.0 in every spectrum"}.get(_nm, "") + ")")
        idf = pd.DataFrame({"spectrum": labels, **{sg["name"]: [_sig_integral(sg, l) * _int_norm_factor(l) for l in labels]
                                                   for sg in signals}})
        if len(signals) > 1:
            first = signals[0]["name"]
            for sg in signals[1:]:
                idf[f"{sg['name']}/{first}"] = idf[sg["name"]] / idf[first].replace(0, np.nan)
        st.dataframe(idf, hide_index=True, width="stretch")
        st.markdown("**Integration window [ppm]** and how each value was obtained: fit (linewidth, residual "
                    "as % of the maximum), area, or manual (limits dragged on the plot). ⚠ marks a cell where "
                    "the fit and the numerical area disagree by more than 15 %: look at it.")
        if st.button("Clear manual limits", key="_sig_clear_manual",
                     help="Forget every limit dragged on the plot; the measured windows are used again."):
            _push()
            for sg_ in signals:
                for mm_ in sg_["meas"].values():
                    mm_["mlo"] = mm_["mhi"] = None
            _rerun_later()
        def _fmt_lim(sg, l):
            mm = sg["meas"].get(l)
            if mm is None:
                return ""
            if mm.get("noint") or _sig_value(sg, l, "lo") is None:
                return "shift only" + (" (δ set by hand)" if mm.get("mpos") is not None else "")
            s_ = f"{_sig_value(sg, l, 'lo'):.3f} – {_sig_value(sg, l, 'hi'):.3f}  "
            if mm.get("mpos") is not None:
                s_ = "δ set by hand; " + s_
            meth = _method(sg, mm)
            if meth == "deconv":
                _d = mm["dcv"]
                s_ += (f"deconvoluted ({_d['n']} line{'s' if _d['n'] != 1 else ''} of {_d['n_all']} fitted, "
                       f"residual {100 * _d['rmse']:.1f} %)")
            elif meth == "manual":
                s_ += "manual limits, sum" + (", local baseline" if sg.get("baseline", False) else "")
            elif meth == "fit":
                s_ += f"fit ({mm['fit_fwhm_hz']:.1f} Hz, {100 * mm['fit_rmse']:.1f} %"
                if mm.get("fit_nuisance"):
                    s_ += f", {mm['fit_nuisance']} neighbour{'s' if mm['fit_nuisance'] > 1 else ''} fitted"
                if mm.get("fit_rejected"):
                    s_ += f", {mm['fit_rejected']} pts dropped"
                s_ += ")"
            else:
                s_ += "sum" + (", local baseline" if sg.get("baseline", False) else "")
            chk = _sig_check(sg, l)
            if chk is not None and abs(chk) > 0.15:
                s_ += f"  ⚠ data/fit in window {1 + chk:.2f}"
            return s_
        st.dataframe(pd.DataFrame({"spectrum": labels, **{sg["name"]: [_fmt_lim(sg, l) for l in labels]
                                                          for sg in signals}}), hide_index=True, width="stretch")

with tab_exp:
    ex_kind = st.radio("Column A", ["kinetics: time in seconds", "titration: volume added in mL"],
                       horizontal=True, key="_x_kind")
    sample = st.text_input("Sample / file stem", value=os.path.basename(root.rstrip("/")) or "nmr", key="_stem")
    n1, n2 = st.columns([1.6, 1])
    _int_names = [sg["name"] for sg in signals if sg.get("use_int", True) and sg["meas"]]
    n1.radio("Integrals", ["raw", "reference signal = 1.0", "largest = 1.0"], horizontal=True, key="_int_norm_in",
             help="Per spectrum: divide every integral by that of the reference signal, or by the largest one, "
                  "so the numbers read like TopSpin's calibrated integrals. Equilibrist only uses ratios "
                  "within a spectrum, so this changes readability, not the fit.")
    _ref_in = n2.selectbox("Reference signal", _int_names or ["(none)"], key="_int_ref_in",
                           disabled=not _int_names or not st.session_state.get("_int_norm_in", "raw").startswith("ref"))
    st.session_state["_int_norm"] = {"raw": "raw", "reference signal = 1.0": "reference", "largest = 1.0": "largest"}[
        st.session_state.get("_int_norm_in", "raw")]
    st.session_state["_int_ref"] = _ref_in if _int_names else None
    xs = []
    for r, _ in cal_specs:
        v = r["x"]
        xs.append(None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
    ex_peaks = [{"name": sg["name"], "pos": {l: _sig_value(sg, l, "pos") for l in labels}}
                for sg in signals if sg.get("use_shift", True) and sg["meas"]]
    ex_regions = [{"name": sg["name"], "nH": sg.get("nH", 1), "baseline": sg.get("baseline", False),
                   "values": {l: _sig_integral(sg, l) for l in labels if l in sg["meas"]},
                   "limits": {l: ((_sig_value(sg, l, "lo"), _sig_value(sg, l, "hi"))
                                  if _sig_value(sg, l, "lo") is not None else None) for l in labels}}
                  for sg in signals if sg.get("use_int", True) and sg["meas"]]
    if not ex_peaks and not ex_regions:
        st.info("Define at least one signal (〰 mode), with its shift or integral ticked.")
    else:
        # only spectra on which at least one exported signal was measured make a row
        _has = {l for pk in ex_peaks for l, p in pk["pos"].items() if p is not None} | \
               {l for rg in ex_regions for l, lim in rg["limits"].items() if lim is not None}
        rows = [{"label": l, "x": v, "spec": by_label[l]} for l, v in zip(labels, xs) if l in _has]
        missing = [r["label"] for r in rows if r["x"] is None]
        if missing:
            st.warning("Column A is empty for: " + ", ".join(missing) + " (type the value in the box beside the spectrum).")
        df = core.build_equilibrist_table(rows, ex_peaks, ex_regions)
        if st.session_state.get("_int_norm", "raw") != "raw" and ex_regions:
            _f = np.array([_int_norm_factor(r["label"]) for r in rows], float)
            for k in range(1, 1 + len(ex_regions)):           # integration columns follow column A
                df.iloc[:, k] = df.iloc[:, k].to_numpy(float) * _f
        df = df.rename(columns={"x": "t" if ex_kind.startswith("kin") else "V"})
        st.caption("Integration columns first, then shift columns, as Equilibrist's mixed mode expects. "
                   "A signal exported as both appears twice under its name in the file; Equilibrist "
                   "tells the two apart by position (the count in `integration:`). The preview marks "
                   "the shift columns with (δ).")
        _prev = df.copy()
        _cols, _seen = [], {}
        for k, c in enumerate(df.columns):
            c = c if k <= len(ex_regions) else f"{c} (δ)"
            _seen[c] = _seen.get(c, 0) + 1
            _cols.append(c if _seen[c] == 1 else f"{c} [{_seen[c]}]")     # a table cannot show two equal headers
        _prev.columns = _cols
        st.dataframe(_prev, hide_index=True, width="stretch")
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="data")
        st.download_button("⬇ data file (.xlsx)", xbuf.getvalue(), file_name=f"{sample}_input.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="_dl_xlsx")

with tab_proc:
    rep = pd.DataFrame([{
        "label": sp.exp.label, "source": sp.source, "PHC0": round(sp.phc0, 2), "PHC1": round(sp.phc1, 2),
        "TopSpin PHC0": sp.exp.phc0, "TopSpin PHC1": sp.exp.phc1, "offset [ppm]": round(offsets.get(sp.exp.label, 0.0), 4),
        "NS": sp.exp.ns, "RG": sp.exp.rg, "pulse program": sp.exp.pulprog, "SFO1 [MHz]": sp.exp.sfo1,
        "points": sp.real.size} for _, sp in cal_specs])
    st.dataframe(rep, hide_index=True, width="stretch")

# ── publication figure ──────────────────────────────────────────────────────
st.subheader("Figure")
ex1, ex2, ex3 = st.columns(3)
if ex1.button("Build publication figure (PDF + PNG)", key="_build_fig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stack = view != "overlay"
    hfig = 2.2 + 0.28 * len(cal_specs) if stack else 4.0
    mfig, ax = plt.subplots(figsize=(7.2, min(hfig, 10)))
    scaled = []
    for r, sp in cal_specs:
        m = (sp.ppm >= lo) & (sp.ppm <= hi)
        x, y = sp.ppm[m], sp.real[m]
        if x.size == 0:
            continue
        top = float(x[int(np.nanargmax(y))])
        mm = ~((x > top - 0.15) & (x < top + 0.15))
        ref_max = float(np.nanmax(np.abs(y[mm]))) if mm.any() else float(np.nanmax(np.abs(y)))
        mode_ = _norm_map[norm]
        k = (1.0 / ref_max if ref_max > 0 else 1.0) if mode_ == "max" else core.scale_factor(sp, mode_)
        ys = y * k
        if clip_on:
            ys = np.minimum(ys, 1.2 * ref_max * k)
        scaled.append((r, x, ys))
    tallest = max(float(np.nanmax(y)) for _, _, y in scaled) or 1.0
    step = offset * tallest if stack else 0.0
    if flip:
        scaled = scaled[::-1]
    for i, (r, x, y) in enumerate(scaled):
        t = r["time_h"]
        name = _fmt_time(float(t)) if t is not None and not (isinstance(t, float) and np.isnan(t)) else str(r["label"])
        ax.plot(x + i * skew, y + i * step, lw=0.6, color="black" if stack else COLORS[i % len(COLORS)])
        if stack:
            ax.text(hi + 0.02 * (hi - lo), i * step, name, fontsize=6.5, va="bottom", ha="right")
    ax.set_xlim(hi, lo); ax.set_xlabel("δ [ppm]"); ax.set_yticks([])
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    mfig.tight_layout()
    pdf = io.BytesIO(); mfig.savefig(pdf, format="pdf"); png = io.BytesIO(); mfig.savefig(png, format="png", dpi=300)
    st.session_state["_export"] = (pdf.getvalue(), png.getvalue())
    plt.close(mfig)
if "_export" in st.session_state:
    pdf_b, png_b = st.session_state["_export"]
    ex2.download_button("PDF", pdf_b, file_name="nmr_stack.pdf", mime="application/pdf", key="_dl_pdf")
    ex3.download_button("PNG", png_b, file_name="nmr_stack.png", mime="image/png", key="_dl_png")

# ── sidebar: session save / load ────────────────────────────────────────────
with st.sidebar:
    st.subheader("Session")
    _msg = st.session_state.pop("_sess_msg", None)
    if _msg:
        st.success(_msg)
    _zip_name = (st.session_state.get("_zip_key") or (None, None, None))[1]
    _payload = _session_payload(root, src_mode, _zip_name)
    st.session_state["_sess_json"] = _json.dumps(_payload, indent=1)
    st.download_button("💾 Save session", st.session_state["_sess_json"].encode("utf-8"),
                       file_name=f"{st.session_state.get('_stem') or 'nmr_stack'}_session.json",
                       mime="application/json", key="_sess_dl", width="stretch",
                       help="Everything except the spectra: calibration, signals and their limits, phase "
                            "tweaks, the experiments table, processing and display settings, the ppm window. "
                            "The spectra are read again from the folder (or the zip) when the session is loaded.")
    _up = st.file_uploader("Load session", type=["json"], key="_sess_up",
                           help="Open the folder (or the zip) of that session first if it is not the current one; "
                                "a folder path saved in the session is opened automatically when it exists.")
    if _up is not None:
        _k = (_up.name, _up.size)
        if st.session_state.get("_sess_key") != _k:
            st.session_state["_sess_key"] = _k
            try:
                _p = _json.loads(_up.getvalue().decode("utf-8"))
                if _p.get("app") != "nmr_stack":
                    raise ValueError("not an NMR stack session file")
                st.session_state["_sess_pending"] = _p
                _rerun_later()
            except Exception as _e:
                st.error(f"Could not read that session: {_e}")

# ── deferred rerun (see _rerun_later) ───────────────────────────────────────
if st.session_state.pop("_rerun_req", False):
    st.rerun()
