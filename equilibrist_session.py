# -*- coding: utf-8 -*-
"""equilibrist_session.py

Save and restore a complete Equilibrist analysis as a single JSON file.

The intent is FAIR-compliance: one self-contained file that bundles the
script (which encodes the model), the experimental data (verbatim, with
a checksum for integrity), the per-fit settings (tolerance, maxiter,
constraints, fitted-parameter selection), and a summary of the last
fit result (so the file documents the analysis even if the user does
not re-run the fit).  Inspired by Musketeer's .fit files but stored as
plain JSON so the file is human-inspectable and language-portable.

Public API
----------

``make_session_json(...)`` — returns a JSON string suitable for download.
``parse_session_json(text)`` — returns a dict with decoded data bytes.

Both functions are stateless and have no Streamlit dependency, so the
module can be imported in headless contexts and unit-tested directly.
"""

from __future__ import annotations
import json
import sys
import base64
import hashlib
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# Bumping this string is the only supported way to break backward-compat.
# v2 (2026-05-23) adds three FAIR-reproducibility fields to the payload:
#   * ``env``  — Python / NumPy / SciPy / joblib version strings
#   * ``seed`` — RNG seed actually used by the bootstrap / MC routines
#   * ``uq``   — bootstrap / jackknife / Monte Carlo summary dicts
# Older v1 files are still loadable (parse_session_json accepts both),
# they simply lack these fields.
SESSION_FORMAT_VERSION = "equilibrist.session.v2"
_ACCEPTED_SESSION_FORMATS = frozenset({
    "equilibrist.session.v1",   # legacy, loaded but lacks env/seed/uq
    "equilibrist.session.v2",
})


def _scrub(obj):
    """Recursively make an arbitrary object JSON-serializable.

    Converts numpy scalars to Python scalars, numpy arrays to lists,
    tuples to lists, and drops anything else with a best-effort
    ``repr`` fallback (so saving never crashes on a stray object).
    """
    # numpy scalars / arrays — handled without importing numpy at top
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return [_scrub(x) for x in obj.tolist()]
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_scrub(x) for x in obj]
    try:
        return repr(obj)
    except Exception:
        return None


def _capture_env() -> dict:
    """Capture the runtime environment that affects numerical reproducibility.

    The Python interpreter version, plus the three numerical-stack
    libraries that drive every fit and UQ routine in Equilibrist
    (NumPy for arrays, SciPy for the optimiser and special functions,
    joblib for the parallel iteration in bootstrap/jackknife/MC).
    Each import is wrapped in try/except so a missing optional library
    yields an empty string rather than crashing the session save.
    """
    out = {"python": sys.version.split()[0]}
    for _mod in ("numpy", "scipy", "joblib"):
        try:
            _m = __import__(_mod)
            out[_mod] = str(getattr(_m, "__version__", "") or "")
        except Exception:
            out[_mod] = ""
    return out


def _summarize_uq(uq: Optional[dict]) -> Optional[dict]:
    """Extract a JSON-friendly scalar summary of a UQ-or-LST result dict.

    Accepts the return shape of any ``bootstrap_*`` / ``jackknife_*`` /
    ``monte_carlo_*`` routine in ``equilibrist_bootstrap.py`` AND the
    return shape of ``compute_local_sensitivity`` (the Masson ξ test)
    in ``equilibrist_diagnostics.py``.  The bulky per-parameter
    ``samples`` array is dropped only when above the per-parameter
    cap; everything else that documents the result — the percentile
    CI block, the per-parameter SE / median dicts, the iteration
    counters, the wall-time, the method label, the LST per-variable
    f / σ_cond / σ_marg / coupling dicts and the per-paired-difference
    transparency records — is retained.

    Returns ``None`` for an empty or non-dict input so the caller can
    test truthiness before adding the entry to the payload.
    """
    if not isinstance(uq, dict) or not uq:
        return None
    # Scalar counters / labels — union across bootstrap, jackknife,
    # Monte Carlo, and LST schemas.
    keep_scalars = (
        # Common to bootstrap / jackknife / MC
        "n_iter", "n_bootstrap", "n_jack",
        "n_success", "n_failed",
        "wall_seconds", "method", "best_message",
        "ci_levels",
        # LST (Masson ξ) specific scalars
        "rmse_opt", "n_data", "n_grid", "step",
    )
    out: dict = {k: uq.get(k) for k in keep_scalars if k in uq}
    # Per-parameter SE / median / CI dicts.  Names differ by family
    # (bootstrap uses ``ci`` as {param: (lo, med, hi)}; MC uses flat
    # ``mc_se`` / ``mc_median`` / ``mc_ci_lo`` / ``mc_ci_hi``;
    # jackknife uses ``jack_se``; LST uses ``nominal`` / ``f`` /
    # ``sigma_cond`` / ``sigma_marg`` / ``coupling``).  Copy whichever
    # are present.
    #
    # ``best_fit`` is the per-parameter point estimate from the
    # original (unresampled) fit, used by the UI to overlay the
    # optimum in the bootstrap / jackknife / MC histograms.  It must
    # round-trip through the JSON because the histograms read it
    # directly via ``_br["best_fit"][param_name]``.
    for d_key in ("ci",                                         # bootstrap
                  "bs_se",                                       # bootstrap
                  "best_fit",                                    # B/J/MC
                  "param_kinds",                                 # J + MC
                  "mc_se", "mc_median", "mc_ci_lo", "mc_ci_hi", # MC
                  "jack_se", "influence",                       # jackknife
                  "nominal", "f",                               # LST
                  "sigma_cond", "sigma_marg", "coupling"):      # LST
        v = uq.get(d_key)
        if isinstance(v, dict) and v:
            out[d_key] = _scrub(v)
    # nuisance_specs (MC input-uncertainty configuration) is small
    # and critical for documenting an MC run's reproducibility.
    ns = uq.get("nuisance_specs")
    if isinstance(ns, dict) and ns:
        out["nuisance_specs"] = _scrub(ns)
    # ``selected_vars`` (LST): list of variable names actually probed.
    # Small, always kept verbatim.
    sv = uq.get("selected_vars")
    if isinstance(sv, (list, tuple)) and sv:
        out["selected_vars"] = list(sv)
    # ``steps`` (jackknife): per-observation step identifiers (list of
    # ints or strings).  ``make_jackknife_figure`` reads this to label
    # the x-axis ticks of the per-step influence plot, so the figure
    # cannot render without it.
    stps = uq.get("steps")
    if isinstance(stps, (list, tuple)) and stps:
        out["steps"] = _scrub(list(stps))
    # ``diff_records`` (LST): per-variable list of paired-difference
    # records used by the transparency view.  Structure is
    # {var: [{"sign": ±1, "others": {...}, "diff": float}, ...]}.
    # Small enough to keep in full (2·3^(N-1) entries per variable;
    # for typical N≤4 this is ≤54 records).
    dr = uq.get("diff_records")
    if isinstance(dr, dict) and dr:
        out["diff_records"] = _scrub(dr)
    # Bootstrap / MC / LST sample distributions — needed if the user
    # wants to re-render UQ histograms after a JSON reload without re-
    # running the resampler.  Capped at MAX_SAMPLES_PER_PARAM ×
    # number of parameters to keep typical JSON files in the low-MB
    # range; if the cap is hit we drop the samples (the SE / CI
    # scalars above are always sufficient for the tabular summary).
    MAX_SAMPLES_PER_PARAM = 20000     # ~160 KB per parameter at 8 B/float
    samples = uq.get("samples")
    if isinstance(samples, dict) and samples:
        try:
            n_params = len(samples)
            n_per    = max(len(np.asarray(v).ravel())
                           for v in samples.values()) if n_params else 0
            if n_params and n_per and n_per <= MAX_SAMPLES_PER_PARAM:
                out["samples"] = _scrub(samples)
        except Exception:
            pass
    return _scrub(out) if out else None


# ─── Helpers for diagnostic-state persistence (widgets + caches) ───

def _to_json_key(k):
    """Encode a tuple-keyed session_state key as a JSON-friendly list.

    Tuples are flagged with a sentinel string ``"@t"`` so they survive
    the JSON round-trip and can be reconstructed via ``_from_json_key``.
    Plain values pass through unchanged.
    """
    if isinstance(k, tuple):
        return ["@t"] + [_to_json_key(e) for e in k]
    if isinstance(k, list):
        return [_to_json_key(e) for e in k]
    return k


def _from_json_key(j):
    """Inverse of ``_to_json_key`` — reconstruct tuples from the sentinel."""
    if isinstance(j, list) and len(j) >= 1 and j[0] == "@t":
        return tuple(_from_json_key(e) for e in j[1:])
    if isinstance(j, list):
        return [_from_json_key(e) for e in j]
    return j


def _capture_widget_state(session_state) -> dict:
    """Snapshot diagnostic-widget values for JSON serialisation.

    Walks ``session_state`` and captures the JSON-serialisable values
    of every key matching a diagnostic-widget prefix (bootstrap /
    jackknife / Monte-Carlo / 1-D & 2-D RMSE profile / Masson ξ
    controls + the LST variable-selection checkboxes).  Restored
    on JSON load before the widgets render, so the boxes display
    the values that were actually used to produce the saved
    diagnostics rather than their defaults.
    """
    prefixes = (
        # Bootstrap
        "_bs_n_", "_bs_mth_", "_bs_par_",
        # Jackknife
        "_jk_par_", "_jk_xkind_",
        # Monte Carlo
        "_mc_iter_", "_mc_par_",
        # Monte Carlo per-component input σ (relative uncertainty %
        # on initial concentrations / titrants / V₀).  Keys follow
        # ``_mc_sig_{eq|kin}_{conc0|titrant|V0_mL}_{name}``.  Without
        # capturing these, every input box on the MC panel resets to
        # 0.0 % after reload, the user thinks MC is mis-restored,
        # and the "Run Monte Carlo" button degenerates because
        # ``_specs`` is empty.
        "_mc_sig_",
        # 1-D RMSE profile
        "_prof_span_", "_prof_npts_", "_prof_par_", "_prof_sel_",
        # 2-D RMSE colormap
        "_p2d_p1_", "_p2d_p2_", "_p2d_sx_", "_p2d_sy_",
        "_p2d_np_", "_p2d_par_",
        # Profile trace
        "_ptr_x_", "_ptr_span_", "_ptr_np_", "_ptr_sel_", "_ptr_par_",
        # Masson ξ (LST): step, parallel, per-variable checkboxes,
        # variable multiselect
        "_lst_kin_step", "_lst_eq_step",
        "_lst_kin_par",  "_lst_eq_par",
        "_lst_kin_chk_", "_lst_eq_chk_",
        "_var_pick_",
        # Figure-gating toggles (``_show_*``).  Captured so the
        # JSON-load auto-refit can put the user back where they
        # were — outer panel expanded, inner figure checkboxes in
        # the state the user had them.  Without this, the inner
        # checkboxes (``_show_jk_influence_*``,
        # ``_show_prof_*``, ``_show_p2d_*``, ``_show_corr_*``)
        # default to False every reload and the figures stay
        # hidden behind another click — tables visible (they're
        # not gated by these toggles), figures invisible.
        "_show_",
    )
    out: dict = {}
    for k in list(session_state.keys()):
        if not isinstance(k, str):
            continue
        if not any(k.startswith(p) for p in prefixes):
            continue
        v = session_state[k]
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            try:
                # Only keep lists of JSON-friendly scalars
                out[k] = [x for x in v
                          if isinstance(x, (str, int, float, bool))]
            except Exception:
                pass
    return out


def _serialize_tuple_caches(session_state) -> list:
    """Capture tuple-keyed diagnostic caches (1-D and 2-D RMSE profiles).

    The 1-D RMSE profile (``_prof_cache_*``) and 2-D RMSE colormap
    (``_p2d_cache_*``) modules cache their results under tuple keys.
    Walking ``session_state.keys()`` to find them does NOT work —
    Streamlit's ``SessionStateProxy.keys()`` filters out non-string
    keys from external iteration (the cache values are still
    reachable via ``session_state.get(tuple_key)`` because that's a
    direct lookup, not an iteration).  So we instead read an
    explicit registry list — maintained as a string-keyed entry by
    the cache-write sites in ``app.py`` — that records every tuple
    cache key the app has populated this session.

    Returns a list of ``[encoded_key, value]`` pairs; consumed by
    the load handler in ``app.py``.
    """
    prefixes = ("_prof_cache_", "_p2d_cache_", "_ptr_cache_", "_lst_cache_")
    registry = session_state.get("_tuple_cache_registry")
    if not isinstance(registry, (list, tuple)):
        return []
    out: list = []
    for k in registry:
        if not (isinstance(k, tuple) and k
                and isinstance(k[0], str)
                and any(k[0].startswith(p) for p in prefixes)):
            continue
        try:
            # Direct lookup works for tuple keys even though
            # iteration doesn't.
            v = session_state.get(k)
            if v is None:
                continue
            v_clean = _scrub(v) if not isinstance(
                v, (int, float, str, bool)) else v
            out.append([_to_json_key(k), v_clean])
        except Exception:
            continue
    return out



def make_session_json(*,
                      script: str,
                      script_filename: Optional[str] = None,
                      data_bytes: Optional[bytes] = None,
                      data_filename: Optional[str] = None,
                      data_kind: str = "experimental",
                      fit_state: Optional[dict] = None,
                      fit_result_summary: Optional[dict] = None,
                      bootstrap_result: Optional[dict] = None,
                      jackknife_result: Optional[dict] = None,
                      monte_carlo_result: Optional[dict] = None,
                      lst_result: Optional[dict] = None,
                      widget_state: Optional[dict] = None,
                      profile_caches: Optional[list] = None,
                      seed: Optional[int] = None,
                      notes: str = "",
                      app_version: str = "") -> str:
    """Build a self-contained JSON session string.

    Parameters
    ----------
    script : str
        The full Equilibrist script (concentrations / volumes / titrant
        / reactions / equilibria / kinetics / plot / nmr / spectra
        sections).  This is the human-editable model definition.
    script_filename : str, optional
        Original filename of the .txt script the user uploaded, so the
        restore step can put the same name back in the sidebar badge.
        Optional — falls back to ``"restored.txt"`` on load.
    data_bytes : bytes, optional
        Raw bytes of the experimental data file (xlsx / csv / etc.) as
        uploaded by the user.  Stored base64-encoded with a SHA-256
        checksum for integrity verification on restore.
    data_filename : str, optional
        Original filename, preserved so the restore step can pick the
        right loader.  Defaults to ``"experimental_data.xlsx"``.
    data_kind : str
        Free-text label for the data type (``"experimental"``,
        ``"nmr"``, ``"spectra"``, ...).  Not used by the loader; just
        for the JSON file to be self-documenting.
    fit_state : dict, optional
        Whatever per-fit configuration the UI exposes (selected
        parameters, tolerance, maxiter, constraints, ...).  Stored
        verbatim after scrubbing for JSON compatibility.
    fit_result_summary : dict, optional
        Compact summary of the last fit's outputs — fitted parameters,
        R², RMSE, AIC, etc.  Excludes large arrays (raw residuals,
        spectra matrices) to keep the file small.
    bootstrap_result, jackknife_result, monte_carlo_result : dict, optional
        Result dicts returned by the ``bootstrap_*`` / ``jackknife_*``
        / ``monte_carlo_*`` routines in ``equilibrist_bootstrap.py``.
        Each is reduced to its scalar / per-parameter summary via
        ``_summarize_uq`` (drops the bulky ``samples`` arrays).  Stored
        under the ``uq`` key in the payload.  Pass ``None`` for any
        family that was not run.
    seed : int, optional
        Random-number-generator seed actually used by the bootstrap /
        MC routines for this analysis.  Recorded verbatim under the
        ``seed`` payload key so the bootstrap and Monte Carlo numbers
        can be reproduced bit-for-bit on reload.
    notes : str
        Free-text user notes.
    app_version : str
        Equilibrist version string, for traceability.
    """
    payload = {
        "format":              SESSION_FORMAT_VERSION,
        "saved_at_utc":        datetime.now(timezone.utc).isoformat(),
        "app_version":         app_version,
        "env":                 _capture_env(),
        "notes":               notes,
        "script":              script,
        "script_filename":     script_filename or "",
        "fit_state":           _scrub(fit_state or {}),
        "fit_result_summary":  _scrub(fit_result_summary or {}),
    }
    if seed is not None:
        try:
            payload["seed"] = int(seed)
        except (TypeError, ValueError):
            payload["seed"] = None
    # UQ + LST summaries — only emitted for families actually run.
    _uq = {}
    _bs = _summarize_uq(bootstrap_result)
    if _bs is not None:
        _uq["bootstrap"] = _bs
    _jk = _summarize_uq(jackknife_result)
    if _jk is not None:
        _uq["jackknife"] = _jk
    _mc = _summarize_uq(monte_carlo_result)
    if _mc is not None:
        _uq["monte_carlo"] = _mc
    _lst = _summarize_uq(lst_result)
    if _lst is not None:
        _uq["lst"] = _lst
    if _uq:
        payload["uq"] = _uq
    # Widget state — restored before widgets render so the boxes
    # display the values used during the saved diagnostics rather
    # than their defaults.
    if isinstance(widget_state, dict) and widget_state:
        payload["widget_state"] = _scrub(widget_state)
    # 1-D and 2-D RMSE profile tuple-keyed caches.  Restored on
    # load so the panels re-display the saved profiles without
    # requiring a recompute.
    if isinstance(profile_caches, list) and profile_caches:
        payload["profile_caches"] = profile_caches
    if data_bytes is not None and len(data_bytes) > 0:
        payload["data"] = {
            "filename":    data_filename or "experimental_data.xlsx",
            "kind":        data_kind,
            "encoding":    "base64",
            "sha256":      hashlib.sha256(data_bytes).hexdigest(),
            "size_bytes":  len(data_bytes),
            "content":     base64.b64encode(data_bytes).decode("ascii"),
        }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse_session_json(text: str) -> dict:
    """Decode a session JSON string.

    Returns a dict with the same shape as the saved JSON, except the
    ``data`` sub-dict (if present) carries an extra ``bytes`` field
    containing the decoded raw bytes.  Raises ``ValueError`` on a
    format-version mismatch or a SHA-256 integrity-check failure.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse session JSON: {e}") from e

    fmt = obj.get("format", "")
    if fmt not in _ACCEPTED_SESSION_FORMATS:
        raise ValueError(
            f"Unrecognized session format: {fmt!r}.  "
            f"This file is from a different Equilibrist version "
            f"(expected one of {sorted(_ACCEPTED_SESSION_FORMATS)})."
        )

    data = obj.get("data")
    if isinstance(data, dict) and data.get("encoding") == "base64":
        content = data.get("content", "")
        try:
            decoded = base64.b64decode(content, validate=False)
        except Exception as e:
            raise ValueError(f"Embedded data is not valid base64: {e}") from e
        expected = data.get("sha256", "")
        if expected:
            actual = hashlib.sha256(decoded).hexdigest()
            if actual != expected:
                raise ValueError(
                    "Embedded data failed SHA-256 integrity check "
                    "(saved file appears to be corrupted)."
                )
        data["bytes"] = decoded
    return obj


def summarize_stats(stats: dict) -> dict:
    """Extract a small, JSON-friendly summary of a fit result.

    Large arrays (residuals, A_obs/A_calc, per_col_*, sp_concs,
    delta_vecs_all, pure_shifts) are deliberately excluded so the
    session file stays compact.  Scalar metrics, per-parameter point
    estimates and errors, the full Hessian covariance, and the
    mode-specific split-component diagnostics are kept.

    The covariance matrix is preserved because downstream FAIR-reuse
    of the JSON (parameter correlation heatmap, t-test p-values,
    Hessian 1σ / 95 % ellipse overlay on a 2D RMSE profile) all
    consume it; without it those diagnostics would have to be
    recomputed even when the user only wants to inspect the saved
    fit result.

    See ``equilibrist_session.py`` audit (Eric Masson, 2026-05-22):
    expanded from 14 scalars to ~30 to cover the union of stats
    populated by the 10 fit modules — `chi2_rmse` (mixed-mode NMR),
    `fitted_titrants`, `param_cov` / `param_cov_names`, residual
    moments, integ/shift/conc split metrics, spectra wavelength
    window, and a few mode flags.
    """
    if not isinstance(stats, dict):
        return {}
    keep_scalars = (
        # Core fit metrics
        "n_points", "n_params", "ssr", "rmse", "chi2_rmse", "r_squared",
        "log_likelihood", "aic", "aicc", "bic",
        # Residual diagnostics (DW, Shapiro, moments)
        "durbin_watson", "shapiro_p",
        "res_mean", "res_std", "res_skew", "res_kurtosis",
        # Convergence / mode flags
        "n_iter", "timed_out", "fit_mode", "is_kinetics",
        # Mixed-mode NMR — integration / shift split
        "rmse_integ", "rmse_shift", "r2_integ", "r2_shift",
        "n_integ_pts", "n_shift_pts",
        # Spectra-mode scalars
        "rmse_conc", "r2_conc",
        "opt_wl_min", "opt_wl_max", "path_cm", "auto_range",
        # NMR flags
        "nmr_noref", "pure_shifts_anchored",
    )
    out = {k: stats.get(k) for k in keep_scalars if k in stats}

    # Per-parameter dicts (point estimates, errors, fitted nuisance
    # concentrations and titrants).  fitted_titrants was missing pre-
    # audit and meant titrant fits couldn't be properly re-seeded from
    # the JSON on restore.
    for key in ("param_values", "param_errors",
                "fitted_concs", "fitted_titrants"):
        v = stats.get(key)
        if isinstance(v, dict):
            cleaned = {}
            for k, vv in v.items():
                if vv is None: continue
                try:
                    cleaned[k] = float(vv)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                out[key] = cleaned

    # Full Hessian covariance matrix + parameter ordering.  _scrub
    # at the end of this function turns the ndarray into a nested
    # list, so it round-trips through JSON cleanly.
    pcov  = stats.get("param_cov")
    pnames = stats.get("param_cov_names")
    if pcov is not None:
        out["param_cov"] = pcov
        if pnames is not None:
            out["param_cov_names"] = list(pnames)

    # Residual / fitted-value arrays.  Storing these lets the load
    # path re-render the full four-panel residual diagnostic (residuals
    # vs index, residuals vs fitted, histogram, normal Q-Q) without
    # requiring a re-fit.  Cap on size keeps the JSON file in the
    # low-MB range even for spectra fits with n ≳ 10⁴; if any array
    # is over the cap we drop ALL three (mixing kept-and-dropped arrays
    # would break the panels that need both).
    MAX_RES_LEN = 50000
    res_keys = ("residuals", "y_calc", "y_obs")
    try:
        _arrs = {k: np.asarray(stats.get(k), dtype=float).ravel()
                 for k in res_keys if stats.get(k) is not None}
        if _arrs and all(a.size > 0 and a.size <= MAX_RES_LEN
                          for a in _arrs.values()):
            for k, a in _arrs.items():
                out[k] = a
    except Exception:
        pass

    return _scrub(out)
