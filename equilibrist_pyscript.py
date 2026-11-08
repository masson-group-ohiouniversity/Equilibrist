"""
equilibrist_pyscript.py — run a custom Python script inside Equilibrist.

Two sidebar controls ("Load custom .py script", "Run py script") let the user
execute their own Python code against the job that is currently loaded: the
constants, the simulated concentration curves, the variables, the data, and
the last fit.  The script draws its own figure (or table, or text) and
Equilibrist shows it under the main plots, with download buttons.

Inside the script one object is in scope, ``eq``:

    eq.logK            dict  {name: log10 K}         current sidebar values
    eq.K               dict  {name: K}                 linear
    eq.x               array x-axis of the simulated curve (as in $plot x)
    eq.x_label         str
    eq.conc            dict  {species: mM array}       every species
    eq.species         list  species names
    eq.var(name)       array a $variables expression evaluated along x
    eq.variables       dict  {name: expression string}
    eq.parsed          dict  the parsed script      eq.network   dict
    eq.params          dict  concentrations, V0, titrant …  (sidebar values)
    eq.data            dict  {"nmr": …, "conc": …, "spectra": …} as loaded
    eq.fit             dict  statistics of the last fit ({} if none)
    eq.errors          dict  {name: ± log K} from the last fit
    eq.simulate(**logK)  (x, conc) recomputed with some constants changed
    eq.show(obj)       show a matplotlib / plotly figure, DataFrame or text

A matplotlib figure left in a variable named ``fig`` (or a list ``figs``) is
shown automatically.  ``numpy`` (np), ``pandas`` (pd), ``matplotlib.pyplot``
(plt) and ``streamlit`` (st) are pre-imported.

The script runs with the same permissions as Equilibrist itself: only run
scripts you wrote or trust.

© Eric Masson, Ohio University, 2026
"""
from __future__ import annotations

import contextlib
import io
import traceback
from datetime import datetime

import numpy as np

__all__ = ["EquilibristContext", "sidebar_controls", "render_output", "EXAMPLE_SCRIPT"]

_OUT = "_py_output"          # session_state: last run's outputs
_CODE = "_py_code"           # session_state: (filename, source)
_RUN = "_py_run_requested"   # session_state: run on this pass


# ═══════════════════════════════════════════════════════════════════════════
# The object handed to the script
# ═══════════════════════════════════════════════════════════════════════════
def harvest_plot(fig) -> list:
    """The main plot's traces as plain dicts: name, kind ('line' | 'points'),
    x, y, color, excluded (hollow outlier markers).  Empty list if unavailable."""
    out = []
    for tr in (getattr(fig, "data", None) or []):
        try:
            x = np.asarray(tr.x, dtype=float); y = np.asarray(tr.y, dtype=float)
        except Exception:
            continue
        if x.size == 0 or x.shape != y.shape:
            continue
        name = str(tr.name or "")
        mode = str(getattr(tr, "mode", "") or "lines")
        kind = "line" if "lines" in mode else "points"
        color = (getattr(tr.line, "color", None) if kind == "line"
                 else getattr(tr.marker, "color", None))
        out.append({"name": name, "kind": kind, "x": x, "y": y,
                    "color": color if isinstance(color, str) else None,
                    "excluded": name.startswith("_outlier_")})
    return out


class EquilibristContext:
    def __init__(self, *, parsed, network, params, logK, curve, x, x_label,
                 xmax=None, data=None, fit_stats=None, st_module=None, main_fig=None):
        self.parsed, self.network, self.params = parsed, network, params
        self.plot = harvest_plot(main_fig)   # traces of the main plot, as drawn
        self.logK = {k: float(v) for k, v in dict(logK).items()}
        self.x = np.asarray(x, dtype=float)
        self.x_label = str(x_label or "")
        self.xmax = xmax
        self.conc = {k: np.asarray(v, dtype=float) for k, v in dict(curve).items()
                     if isinstance(v, np.ndarray) and not str(k).startswith("_")
                     and len(v) == len(self.x)}
        self.species = list(network.get("all_species", []))
        self.variables = dict(parsed.get("variables", {}) or {})
        self.data = dict(data or {})
        self.fit = dict(fit_stats or {})
        self.errors = dict(self.fit.get("param_errors", {}) or {})
        self._st = st_module
        self._shown = []          # objects displayed through show()

    # ── convenience ──────────────────────────────────────────────────────
    @property
    def K(self) -> dict:
        return {k: 10.0 ** v for k, v in self.logK.items()}

    def var(self, name: str) -> np.ndarray:
        """Evaluate a $variables expression (or a species name) along x."""
        from equilibrist_network import compute_variable_curve
        if name in self.conc:
            return self.conc[name]
        if name not in self.variables:
            raise KeyError(f"'{name}' is neither a species nor a $variables name. "
                           f"Variables: {', '.join(self.variables) or 'none'}")
        return np.asarray(compute_variable_curve(name, self.variables, self.conc,
                                                 self.network, self.x), dtype=float)

    def simulate(self, **logK_overrides):
        """Recompute the concentration curves with some constants changed,
        e.g. ``x, c = eq.simulate(KGHM=7.0)``.  Returns ``(x, conc)`` on the
        same x grid (clipped to xmax like the app's own plot)."""
        from equilibrist_curve import compute_curve, evaluate_x_expression
        lk = dict(self.logK)
        for k, v in logK_overrides.items():
            if k not in lk:
                raise KeyError(f"unknown constant {k!r}; known: {', '.join(lk)}")
            lk[k] = float(v)
        c = compute_curve(self.parsed, self.network, lk, self.params)
        x, _ = evaluate_x_expression(self.parsed.get("plot_x_expr") or "", c, self.parsed)
        x = np.asarray(x, dtype=float)
        mask = x <= float(self.xmax) if self.xmax is not None else np.ones_like(x, bool)
        conc = {k: np.asarray(v, dtype=float)[mask] for k, v in c.items()
                if isinstance(v, np.ndarray) and not str(k).startswith("_") and len(v) == len(mask)}
        return x[mask], conc

    def show(self, obj, caption: str = ""):
        """Display a figure (matplotlib or plotly), a DataFrame, or text.  The
        object is kept so it is still shown after the page refreshes."""
        self._shown.append((obj, caption))
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# Running a script
# ═══════════════════════════════════════════════════════════════════════════
def _kind(obj) -> str:
    n = type(obj).__module__ + "." + type(obj).__name__
    if n.startswith("matplotlib.figure"):
        return "mpl"
    if n.startswith("plotly"):
        return "plotly"
    if n.endswith("DataFrame") or n.endswith("Series"):
        return "table"
    return "text"


def run_script(code: str, name: str, ctx: EquilibristContext) -> dict:
    """Execute ``code`` with ``eq`` in scope.  Returns the outputs to show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st
    ns = {"eq": ctx, "np": np, "pd": pd, "plt": plt, "st": st, "__name__": "__equilibrist_script__"}
    out = {"name": name, "when": datetime.now().strftime("%H:%M:%S"),
           "items": [], "stdout": "", "error": ""}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, name or "<custom script>", "exec"), ns)
    except SystemExit as e:                     # a script may stop itself with a message
        if e.code not in (None, 0):
            out["error"] = str(e.code)
    except Exception:
        out["error"] = traceback.format_exc()
    out["stdout"] = buf.getvalue()
    items = list(ctx._shown)
    shown_ids = {id(o) for o, _ in items}
    # figures / tables left in conventional variable names
    for var in ("fig", "figs", "table", "df"):
        v = ns.get(var)
        if v is None:
            continue
        for o in (v if isinstance(v, (list, tuple)) else [v]):
            if id(o) not in shown_ids and _kind(o) != "text":
                items.append((o, "")); shown_ids.add(id(o))
    out["items"] = items
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit pieces
# ═══════════════════════════════════════════════════════════════════════════
EXAMPLE_SCRIPT = '''# Custom Equilibrist script -- `eq` holds everything about the job shown in the app.
# Leave a matplotlib figure in a variable named `fig` (or call eq.show) and it is
# displayed under the main plots, with PDF / PNG downloads.
#
# What `eq` offers:
#   eq.logK, eq.K          every constant (log10 K / K) at its sidebar value
#   eq.x, eq.x_label       x axis of the simulated curve ($plot x)
#   eq.conc[species]       concentration (mM) of any species along eq.x
#   eq.var("S1")           a $variables expression evaluated along eq.x
#   eq.simulate(KGHM=7.0)  (x, conc) with some constants changed
#   eq.plot                the main plot as drawn: list of {name, kind, x, y, color, excluded}
#   eq.data, eq.fit, eq.errors, eq.parsed, eq.network, eq.params
#   eq.show(obj)           show a figure, a DataFrame or text; print() works too

import numpy as np
import matplotlib.pyplot as plt

# ── Example: fractions of the protonation states within the S012 pool ──────
# This is what the S012 NMR signal reports (a population-weighted average),
# as opposed to the absolute concentrations drawn by Equilibrist.
needed = ["S0", "S1", "S2"]
missing = [n for n in needed if n not in eq.variables and n not in eq.conc]
if missing:
    raise SystemExit(f"This example needs $variables named {needed}; "
                     f"missing: {missing}.  Defined variables: {list(eq.variables)}")

S0, S1, S2 = (eq.var(n) for n in needed)
pool = S0 + S1 + S2

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(eq.x, S0 / pool, label="S0 / S012")
ax.plot(eq.x, S1 / pool, label="S1 / S012")
ax.plot(eq.x, S2 / pool, label="S2 / S012")
ax.set_xlabel(eq.x_label)
ax.set_ylabel("fraction of the S012 pool")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
fig.tight_layout()

print("constants:", {k: round(v, 2) for k, v in eq.logK.items()})
'''


def sidebar_controls() -> None:
    """The two sidebar buttons (call inside ``with st.sidebar:``)."""
    import streamlit as st
    st.markdown("**Custom Python script**")
    up = st.file_uploader("Load custom .py script", type=["py"], key="_py_uploader",
                          help="A Python file that receives the current job as `eq` "
                               "and produces a figure, a table or text.  Download "
                               "the example to see what is available.")
    if up is not None:
        try:
            src = up.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            src = up.getvalue().decode("latin-1")
        st.session_state[_CODE] = (up.name, src)
    code = st.session_state.get(_CODE)
    if code:
        st.caption(f"📄 {code[0]}")
    if st.button("▶ Run py script", key="_py_run_btn", disabled=not code,
                 help="Runs the loaded script against the job shown now.  Its "
                      "output appears under the main plots.  A commented "
                      "starting point ships with Equilibrist as "
                      "custom_script_example.py."):
        st.session_state[_RUN] = True


def render_output(build_context) -> None:
    """Run the script if requested, then show the latest outputs.
    ``build_context`` is a zero-argument function returning an
    ``EquilibristContext`` (called only when a run is requested)."""
    import streamlit as st
    if st.session_state.pop(_RUN, False):
        code = st.session_state.get(_CODE)
        if code:
            with st.spinner("Running your script…"):
                try:
                    ctx = build_context()
                    st.session_state[_OUT] = run_script(code[1], code[0], ctx)
                except Exception:
                    st.session_state[_OUT] = {"name": code[0], "when": datetime.now().strftime("%H:%M:%S"),
                                              "items": [], "stdout": "", "error": traceback.format_exc()}
        try:
            st.toast("Script finished — its output is below the main plots.")
        except Exception:
            pass
    out = st.session_state.get(_OUT)
    if not out:
        return
    st.markdown("---")
    st.markdown(f"**Custom script output** — `{out['name']}` · run at {out['when']}")
    if out["error"]:
        st.error("The script raised an error:")
        st.code(out["error"])
    elif not out["items"] and not out["stdout"].strip():
        st.info("The script ran without errors but produced nothing to show.  "
                "Leave a matplotlib figure in a variable named `fig`, call "
                "`eq.show(...)` for a figure, table or text, or `print(...)`.  "
                "`custom_script_example.py` in the Equilibrist folder is a "
                "working starting point.")
    if out["stdout"].strip():
        st.code(out["stdout"].rstrip())
    for i, (obj, caption) in enumerate(out["items"]):
        kind = _kind(obj)
        if kind == "mpl":
            st.pyplot(obj, clear_figure=False)
            b1, b2 = st.columns(2)
            for col, fmt, mime in ((b1, "pdf", "application/pdf"), (b2, "png", "image/png")):
                buf = io.BytesIO()
                try:
                    obj.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
                    col.download_button(f"Download {fmt.upper()}", data=buf.getvalue(),
                                        file_name=f"custom_figure_{i + 1}.{fmt}", mime=mime,
                                        key=f"_py_dl_{fmt}_{i}")
                except Exception as e:
                    col.caption(f"{fmt.upper()} export failed: {e}")
        elif kind == "plotly":
            st.plotly_chart(obj, use_container_width=True, key=f"_py_plotly_{i}")
        elif kind == "table":
            st.dataframe(obj, use_container_width=True)
        else:
            st.write(obj)
        if caption:
            st.caption(caption)
