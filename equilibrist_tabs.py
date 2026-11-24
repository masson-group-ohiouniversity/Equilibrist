"""
equilibrist_tabs -- several independent workspaces ("tabs") in one Streamlit
session, each holding its own script, data, constants, fit and diagnostics.

A tab is a snapshot of ``st.session_state``.  Switching tabs = snapshot the
current state into the tab being left, wipe the session, restore the other
tab's snapshot.  This happens at the very top of the script run (before any
widget exists), which is the only moment Streamlit allows widget-owned keys
to be written freely.

Keys listed in ``META`` describe the tab set itself and are never part of a
snapshot.  Uploaded-file objects are never snapshotted either: Streamlit does
not allow a file_uploader's value to be set from session state, and the
uploader nonces are bumped on every restore so each tab gets fresh uploaders.
"""
from __future__ import annotations

from typing import Optional

META = ("_tabs", "_tab_idx", "_tab_radio", "_tab_close", "_tab_new", "_tab_new_snapshot", "_tab_new_state",
        "_tab_switch_to", "_tab_notice", "_tab_nonce_seq", "_restore_tour", "_welcome_nonce",
        "_py_code")          # the loaded custom .py script: one for the session, like its uploader
PREFS = ("fit_tolerance_log", "fit_timeout", "fit_refresh", "_global_weighted", "_global_share_shifts", "_global_share_shifts_touched")   # kept across tabs like across resets
_NONCES = ("_uploader_nonce", "_exp_uploader_nonce")

# diagnostics that belong to ONE fit and must go when a tab receives a new fit
_SCRUB_STR = ("_show_", "_bs_result_", "_jk_result_", "_mc_result_", "_lst_result_",
              "_ttest_names_", "_ttest_nulls_")
_SCRUB_TUPLE = ("_p2d_cache_", "_lst_cache_", "_jk_cache_", "_mc_cache_", "_bs_cache_",
                "_prof_cache_", "_ptr_cache_")


_READONLY_CACHE = None
_READONLY_WIDGETS = ("button", "download_button", "form_submit_button", "file_uploader",
                     "plotly_chart", "data_editor", "camera_input", "chat_input")


def _readonly_keys():
    """Keys of widgets whose value Streamlit refuses to set from session state
    (buttons, uploaders, chart selections, data editors).  Learned once by
    scanning the app's own sources for ``key=`` arguments of those calls.
    Returns ``(exact: set, prefixes: tuple)``; f-string keys contribute their
    literal prefix."""
    global _READONLY_CACHE
    if _READONLY_CACHE is not None:
        return _READONLY_CACHE
    import os, re
    exact, prefixes = set(), set()
    here = os.path.dirname(os.path.abspath(__file__))
    pat = re.compile(r"(?:\.(?:%s)|\b_btn)\(" % "|".join(_READONLY_WIDGETS))
    for fn in os.listdir(here):
        if not fn.endswith(".py"):
            continue
        try:
            src = open(os.path.join(here, fn), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in pat.finditer(src):
            i, depth = m.end(), 1
            while i < len(src) and depth:
                c = src[i]
                depth += (c == "(") - (c == ")")
                i += 1
            call = src[m.end():i]
            for km in re.finditer(r"key\s*=\s*(f?)(?:\"([^\"]*)\"|'([^']*)')", call):
                is_f, k = km.group(1), (km.group(2) if km.group(2) is not None else km.group(3))
                if not k:
                    continue
                if is_f and "{" in k:
                    pre = k.split("{", 1)[0]
                    if len(pre) >= 3:
                        prefixes.add(pre)
                else:
                    exact.add(k)
    _READONLY_CACHE = (exact, tuple(sorted(prefixes)))
    return _READONLY_CACHE


def _is_readonly_key(k) -> bool:
    if not isinstance(k, str):
        return False
    exact, prefixes = _readonly_keys()
    return k in exact or (bool(prefixes) and k.startswith(prefixes))


def _is_upload(v) -> bool:
    try:
        from streamlit.runtime.uploaded_file_manager import UploadedFile
    except Exception:                                   # pragma: no cover
        return False
    if isinstance(v, UploadedFile):
        return True
    return isinstance(v, (list, tuple)) and bool(v) and all(isinstance(x, UploadedFile) for x in v)


class StateView(dict):
    """A tab's state addressed the way ``st.session_state`` is.

    Streamlit stores every key as ``str(key)``, so the tuple keys of the
    profile caches sit in a snapshot under their string form.  Code written
    against ``st.session_state`` (the session save / restore) can read and
    write a tab that is not on screen through this wrapper."""

    @staticmethod
    def _k(k):
        return k if isinstance(k, str) else str(k)

    def __getitem__(self, k):
        return dict.__getitem__(self, self._k(k))

    def __setitem__(self, k, v):
        dict.__setitem__(self, self._k(k), v)

    def __delitem__(self, k):
        dict.__delitem__(self, self._k(k))

    def __contains__(self, k):
        return dict.__contains__(self, self._k(k))

    def get(self, k, default=None):
        return dict.get(self, self._k(k), default)

    def pop(self, k, *default):
        return dict.pop(self, self._k(k), *default)

    def setdefault(self, k, default=None):
        return dict.setdefault(self, self._k(k), default)


def forget(ss, k) -> None:
    """Delete ``k`` from the session, skipping a key the session lists but no
    longer holds.  A chart's widget id depends on what it draws (Streamlit
    1.6x), so a chart key can stay listed under an id of an earlier drawing
    after its current entry is gone; deleting it then raises KeyError
    (seen as a crash on a second "Reset" in a kinetics tab)."""
    try:
        del ss[k]
    except KeyError:
        pass


def ensure(ss) -> None:
    """Make sure the tab registry exists."""
    if not isinstance(ss.get("_tabs"), list) or not ss["_tabs"]:
        ss["_tabs"] = [{"name": "Tab 1", "state": {}}]
    if not isinstance(ss.get("_tab_idx"), int) or not (0 <= ss["_tab_idx"] < len(ss["_tabs"])):
        ss["_tab_idx"] = 0


def live_items(ss) -> list:
    """``[(key, value)]`` of every entry the session really holds: a key that is
    listed but can no longer be read (see :func:`forget`) is skipped, where
    ``ss.items()`` would raise KeyError on it."""
    out = []
    for k in list(ss.keys()):
        try:
            out.append((k, ss[k]))
        except KeyError:
            continue
    return out


def snapshot(ss) -> dict:
    """The current workspace as a plain dict (no meta keys, no uploads)."""
    out = {}
    for k, v in live_items(ss):
        if k in META or _is_upload(v) or _is_readonly_key(k):
            continue
        out[k] = v
    return out


def restore(ss, state: dict) -> None:
    """Replace the workspace by ``state`` (meta keys untouched, fresh uploaders)."""
    prefs = {k: ss[k] for k in PREFS if k in ss}
    # Nonces must never repeat across tabs: a widget keyed by the nonce (the
    # script editor, the uploaders) keeps its last frontend value when the same
    # key comes back, so a tab that landed on a nonce another tab had used
    # showed that tab's script.  Each restore therefore takes a value above
    # every nonce seen so far in this session, not just above the tab's own.
    seq = int(ss.get("_tab_nonce_seq", 0) or 0)
    for n in _NONCES:
        seq = max(seq, int(ss.get(n, 0) or 0), int(state.get(n, 0) or 0))
    for k in list(ss.keys()):
        if k in META:
            continue
        forget(ss, k)
    for k, v in state.items():
        if k in META or _is_upload(v) or _is_readonly_key(k):
            continue
        ss[k] = v
    for k, v in prefs.items():
        ss.setdefault(k, v)
    for n in _NONCES:
        seq += 1
        ss[n] = seq
    ss["_tab_nonce_seq"] = seq


def reset_all(ss) -> None:
    """A full reset: the session as it is when the app is launched.  One empty
    tab; every tab's script, data, constants and results are dropped and the
    settings return to their defaults.  Only two counters survive, and both
    move on, so that no uploader or script editor comes back under a key it
    had before (the browser would show its old content again): the uploader
    nonces (see :func:`restore`) and the one of the welcome-screen editor,
    which therefore shows the skeleton script again and not an old draft.
    The same counter keys the custom-script uploader, which comes back empty."""
    welcome = int(ss.get("_welcome_nonce", 0) or 0) + 1
    seq = int(ss.get("_tab_nonce_seq", 0) or 0)
    for n in _NONCES:
        seq = max(seq, int(ss.get(n, 0) or 0))
    for tab in (ss.get("_tabs") or []):
        state = (tab.get("state") or {}) if isinstance(tab, dict) else {}
        for n in _NONCES:
            seq = max(seq, int(state.get(n, 0) or 0))
    for k in list(ss.keys()):
        if k == "_tab_radio":             # the Tabs widget: reassigned below
            continue
        forget(ss, k)
    ss["_tabs"] = [{"name": "Tab 1", "state": {}}]
    ss["_tab_idx"] = 0
    ss["_tab_radio"] = 0
    for n in _NONCES:
        seq += 1
        ss[n] = seq
    ss["_tab_nonce_seq"] = seq
    ss["_welcome_nonce"] = welcome


def tab_name(ss, i: int) -> str:
    """Display name of tab ``i`` (the active one is read live)."""
    tabs = ss["_tabs"]
    if i == ss["_tab_idx"]:
        fn = ss.get("_script_filename")
    else:
        fn = (tabs[i].get("state") or {}).get("_script_filename")
    return str(fn) if fn else f"Tab {i + 1} (empty)"


def global_share(ss, i: int):
    """Share of the information on the shared constants that tab ``i`` brought to
    the global fit it last took part in (mean over the constants, 0..1), or None
    when the tab's latest fit is not a global one."""
    if i == ss["_tab_idx"]:
        st_ = ss.get("_fit_stats")
    else:
        st_ = (ss["_tabs"][i].get("state") or {}).get("_fit_stats")
    g = st_.get("global_fit") if isinstance(st_, dict) else None
    own = (g or {}).get("own_share") if isinstance(g, dict) else None
    vals = [float(v) for v in (own or {}).values() if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def labels(ss) -> list:
    """Unique labels for all tabs (duplicates get a numeric suffix)."""
    out, seen = [], {}
    for i in range(len(ss["_tabs"])):
        n = tab_name(ss, i)
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} #{seen[n]}")
    return out


def swap_if_requested(ss) -> Optional[str]:
    """Run at the top of the script.  Handles a pending new-tab, close-tab or
    switch request.  Returns a short notice for the UI, or None."""
    ensure(ss)
    tabs, cur = ss["_tabs"], ss["_tab_idx"]
    notice = None

    if ss.pop("_tab_new", False):
        # The button that asks for a new tab calls st.rerun() before most
        # widgets have rendered, and Streamlit then drops the state of every
        # widget absent from that run (constants, ticks, concentrations).  The
        # button therefore snapshots the tab itself, while the values are still
        # there; the snapshot taken here is only the fallback.
        # "Duplicate tab" hands over what the new tab starts with (the script of
        # the tab being left, as a freshly loaded script: no data, no results)
        init = ss.pop("_tab_new_state", None)
        init = dict(init) if isinstance(init, dict) else {}
        tabs[cur] = {"name": tab_name(ss, cur), "state": ss.pop("_tab_new_snapshot", None) or snapshot(ss)}
        tabs.append({"name": f"Tab {len(tabs) + 1}", "state": dict(init)})
        restore(ss, init)
        ss["_tab_idx"] = len(tabs) - 1
        notice = ("Tab duplicated: same script; load its data." if init.get("_script_text")
                  else "New tab opened. Load a script and data for it.")

    elif ss.pop("_tab_close", False):
        if len(tabs) <= 1:
            tabs[0] = {"name": "Tab 1", "state": {}}
            restore(ss, {})
            ss["_tab_idx"] = 0
            notice = "Tab cleared."
        else:
            tabs.pop(cur)
            new = max(0, cur - 1)
            restore(ss, tabs[new].get("state") or {})
            ss["_tab_idx"] = new
            notice = f"Tab closed; now on {tab_name(ss, new)}."

    else:
        target = ss.pop("_tab_switch_to", None)
        if target is None:
            r = ss.get("_tab_radio")
            if isinstance(r, int) and r != cur and 0 <= r < len(tabs):
                target = r
        if target is not None and 0 <= target < len(tabs) and target != cur:
            tabs[cur] = {"name": tab_name(ss, cur), "state": snapshot(ss)}
            restore(ss, tabs[target].get("state") or {})
            ss["_tab_idx"] = target

    ss["_tab_radio"] = ss["_tab_idx"]
    return notice


def open_tabs(ss, states: list, active: int = 0, run_global: bool = False, note: str = "",
              info: Optional[dict] = None) -> None:
    """Replace every open tab by ``states`` (one state per tab, in order) and
    show the first one.  The app then walks through the tabs once
    (:func:`tour_step`): a tab's fit record only exists after the tab has been
    drawn, and a global fit needs the record of every tab."""
    states = [s if isinstance(s, dict) else {} for s in states] or [{}]
    ss["_tabs"] = [{"name": f"Tab {i + 1}", "state": s} for i, s in enumerate(states)]
    ss["_tab_idx"] = 0
    for k in ("_tab_close", "_tab_new", "_tab_new_snapshot", "_tab_new_state", "_tab_switch_to", "_tab_notice"):
        ss.pop(k, None)
    restore(ss, states[0])
    ss["_tab_radio"] = 0
    active = active if isinstance(active, int) and 0 <= active < len(states) else 0
    ss["_restore_tour"] = {"todo": list(range(1, len(states))), "final": active,
                           "global": bool(run_global), "n": len(states), "note": str(note or ""),
                           "info": dict(info or {}),      # the saved global fit (weights, ticks per tab)
                           "seen": 0, "stuck": 0, "seq": 0}


def _fit_thread_alive(ss) -> bool:
    bg = ss.get("_bg_fit")
    try:
        return bg is not None and bg["thread"].is_alive()
    except Exception:
        return False


def tour_step(ss) -> Optional[str]:
    """Run once at the very top of every script run, before
    :func:`swap_if_requested`.  While an all-tabs file is being reopened: a
    tab is drawn twice (the first pass puts the restored values into the
    boxes, the second builds the fit record from them), any fit it asked for
    is left to finish, then the next tab is shown; after the last one the
    tab that was active when the file was saved comes back.

    Returns "global" on the run that ends the walk when the saved result was
    a global fit (the app then restores it), "done" when the walk ends
    without one, else None."""
    tour = ss.get("_restore_tour")
    if not isinstance(tour, dict):
        return None
    ensure(ss)
    tour["seq"] = int(tour.get("seq", 0)) + 1          # one per full run (see the app's ticker)
    if len(ss["_tabs"]) != int(tour.get("n", -1)):
        ss.pop("_restore_tour", None)                 # tabs were added or closed meanwhile
        return "done"
    if _fit_thread_alive(ss):
        tour["stuck"] = 0
        return None
    if ss.get("_bg_fit") is not None or ss.get("_fit_requested") or ss.get("_global_fit_requested"):
        # a fit about to start or about to be collected; if it never does
        # (a script or data problem), stop waiting for it
        tour["stuck"] = int(tour.get("stuck", 0)) + 1
        if tour["stuck"] < 5:
            return None
        ss.pop("_fit_requested", None)
        ss.pop("_global_fit_requested", None)
    tour["stuck"] = 0
    seen = int(tour.get("seen", 0))
    drawn = (not ss.get("_script_text")) and seen >= 1          # an empty tab: one pass is enough
    drawn = drawn or (seen >= 2 and isinstance(ss.get("_fit_args_live"), dict)) or seen >= 4
    if not drawn:
        tour["seen"] = seen + 1
        return None
    cur = ss["_tab_idx"]
    todo = [i for i in tour.get("todo", []) if i != cur and 0 <= i < len(ss["_tabs"])]
    if todo:
        tour["todo"], tour["seen"] = todo[1:], 1      # the next tab is drawn in this very run
        ss["_tab_switch_to"] = todo[0]
        return None
    tour["todo"] = []
    if cur != tour.get("final", cur):
        tour["seen"] = 2                              # already drawn twice earlier in the walk
        ss["_tab_switch_to"] = tour["final"]
        return None
    ss.pop("_restore_tour", None)
    return "global" if tour.get("global") else "done"


def scrub_diagnostics(state: dict) -> None:
    """Drop one-fit diagnostics from a snapshot that is about to receive a new fit."""
    for k in list(state.keys()):
        if isinstance(k, str) and k.startswith(_SCRUB_STR):
            del state[k]
        elif isinstance(k, tuple) and k and isinstance(k[0], str) and k[0].startswith(_SCRUB_TUPLE):
            del state[k]


def write_fit_into(state: dict, stats: dict, fit_args: dict, message, key: str = "_fit_args_eq") -> None:
    """Give a (non-active) tab's snapshot the result of a global fit, the way
    the app does for the active tab after a fit."""
    if not state.pop("_preserve_uq_on_next_reset", False):   # set by a session reload
        scrub_diagnostics(state)
    for k, v in (stats.get("param_values") or {}).items():
        state[f"_pending_logK_{k}"] = float(v)
    for root, mM in (stats.get("fitted_concs") or {}).items():
        state[f"_pending_conc_{root}"] = float(mM)
    for tkey, mM in (stats.get("fitted_titrants") or {}).items():
        state[f"_pending_titrant_mM_{tkey}"] = float(mM)
    state["_fit_stats"] = stats
    state[key] = fit_args
    state["_fit_counter"] = int(state.get("_fit_counter", 0)) + 1
    state["_fit_message"] = message


def refresh_fit_args(fa: dict, state: dict) -> dict:
    """Bring a tab's recorded fit args up to date with the widget values held
    in its snapshot.  The fit ticks and the constant / concentration inputs
    live in Streamlit fragments, so changing them does not rerun the main
    script; the recorded ``_fit_args_live`` can therefore be older than the
    boxes the user sees.  The widget keys themselves are always current."""
    fa = dict(fa)
    parsed = fa.get("parsed_fit") or {}
    # every constant the script defines: equilibrium constants AND the rate
    # constants of a kinetics script (a mixed script with pre-equilibria lost its
    # ticked rate constants here and was reported as "no 'fit' box ticked")
    knames = [eq.get("kname") for eq in parsed.get("equilibria", []) if eq.get("kname")]
    for r in parsed.get("kinetics", []) or []:
        for key in ("kname", "krname"):
            if r.get(key) and r[key] not in knames:
                knames.append(r[key])
    if knames and not parsed.get("is_acid_base"):
        # widget keys absent from the snapshot carry no information: keep the recorded ticks
        if any(f"fit_logK_{k}" in state for k in knames):
            fa["fit_keys"] = [k for k in knames if state.get(f"fit_logK_{k}")]
        start = dict(fa.get("start_logK") or {})
        for k in knames:
            v = state.get(f"logK_{k}")
            if isinstance(v, (int, float)):
                start[k] = float(v)
        fa["start_logK"], fa["start_logk"] = start, dict(start)
    params = fa.get("params") or {}
    if isinstance(params, dict):
        conc0 = dict(params.get("conc0") or {})
        changed = False
        for root in list(conc0):
            v = state.get(f"conc_{root}")
            if isinstance(v, (int, float)) and float(v) > 0 and float(v) != conc0[root]:
                conc0[root] = float(v); changed = True
        tit = dict(params.get("titrant_mMs") or {})
        for tkey in list(params.get("titrant_keys") or []):
            v = state.get(f"titrant_mM_{tkey}")
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            if isinstance(v, (int, float)) and float(v) > 0 and tfree in tit and float(v) != tit[tfree]:
                tit[tfree] = float(v); changed = True
        if changed:
            params = dict(params); params["conc0"] = conc0; params["titrant_mMs"] = tit
            fa["params"] = params
        if any(f"fit_conc_{r}" in state for r in conc0):
            fa["fit_conc_keys"] = [r for r in conc0 if state.get(f"fit_conc_{r}")]
        if any(f"fit_titrant_mM_{t}" in state for t in (params.get("titrant_keys") or [])):
            fa["fit_titrant_keys"] = [t for t in (params.get("titrant_keys") or []) if state.get(f"fit_titrant_mM_{t}")]
    if "fit_use_constraints" in state:
        fa["constraints"] = (parsed.get("constraints", []) if state.get("fit_use_constraints") else [])
    return fa


def _constant_names(text, parse_script) -> tuple:
    """(names, is_acid_base) of the constants a script defines: equilibrium
    constants plus forward / reverse rate constants."""
    try:
        p = parse_script(text or "")
    except Exception:
        return [], False
    names = [e.get("kname") for e in p.get("equilibria", []) if e.get("kname")]
    for r in p.get("kinetics", []):
        names += [r[k] for k in ("kname", "krname") if r.get(k)]
    return names, bool(p.get("is_acid_base"))


def propagate_ticked(ss, parse_script) -> str:
    """Copy the active tab's ticked constants, at the values in its boxes, into
    every other tab whose script has a constant of the same name.  Only the
    values move, the way a fit writes them back (``_pending_logK_`` consumed
    before the widgets exist, plus the widget key itself so the global fit
    sees the new value without the tab being opened); the other tabs' ticks,
    bounds, data and scripts are untouched.  Returns a one-line notice."""
    ensure(ss)
    names, acid_base = _constant_names(ss.get("_script_text"), parse_script)
    if acid_base:
        return "Propagation is not available for acid-base (pKa) scripts."
    ticked = {}
    for n in names:
        v = ss.get(f"logK_{n}")
        if ss.get(f"fit_logK_{n}") and isinstance(v, (int, float)):
            ticked[n] = float(v)
    if not ticked:
        return "Nothing to propagate: no constant is ticked for fitting in this tab."
    idx, done, unmatched = ss["_tab_idx"], [], set(ticked)
    names_shown = labels(ss)
    for i, tab in enumerate(ss["_tabs"]):
        if i == idx:
            continue
        state = tab.get("state") or {}
        hit = [n for n in _constant_names(state.get("_script_text"), parse_script)[0] if n in ticked]
        for n in hit:
            state[f"_pending_logK_{n}"] = ticked[n]
            state[f"logK_{n}"] = ticked[n]
            for pre in ("_lin_logK_", "_shdlin_logK_", "_shadow_logK_"):
                state.pop(f"{pre}{n}", None)
        tab["state"] = state
        unmatched -= set(hit)
        if hit:
            done.append(f"{names_shown[i]} ← " + ", ".join(f"{n} = {ticked[n]:.3f}" for n in hit))
    if not done:
        return "No other tab has a constant named " + ", ".join(sorted(ticked)) + "."
    msg = "Propagated: " + "; ".join(done) + "."
    if unmatched:
        msg += "  No match anywhere for " + ", ".join(sorted(unmatched)) + "."
    return msg


def fit_args_with_data(ss) -> list:
    """The fit-args record of every tab that has data, ticked or not (what the
    'Share pure-species shifts' default is judged on)."""
    ensure(ss)
    out = []
    for i, tab in enumerate(ss["_tabs"]):
        fa = ss.get("_fit_args_live") if i == ss["_tab_idx"] else (tab.get("state") or {}).get("_fit_args_live")
        if isinstance(fa, dict) and fa.get("has_data"):
            out.append(fa)
    return out


def collect_experiments(ss, why: Optional[list] = None) -> list:
    """Every tab that can take part in a global fit: ``[(tab_index, label, fa)]``.
    ``why`` (a list) receives one line per tab left out, with the reason."""
    ensure(ss)
    out = []
    names = labels(ss)
    for i, tab in enumerate(ss["_tabs"]):
        if i == ss["_tab_idx"]:
            fa = ss.get("_fit_args_live")
        else:
            state = tab.get("state") or {}
            fa = state.get("_fit_args_live")
            if isinstance(fa, dict):
                fa = refresh_fit_args(fa, state)
        if not isinstance(fa, dict):
            if why is not None:
                why.append(f"{names[i]}: no script with data has been shown in this tab yet")
            continue
        if not fa.get("has_data"):
            if why is not None:
                why.append(f"{names[i]}: no experimental data loaded")
            continue
        if not (fa.get("fit_keys") or fa.get("fit_conc_keys") or fa.get("fit_titrant_keys")):
            if why is not None:
                why.append(f"{names[i]}: no 'fit' box ticked")
            continue
        out.append((i, names[i], fa))
    return out
