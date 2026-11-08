# -*- coding: utf-8 -*-
"""equilibrist_diagnostics.py

Statistical diagnostics for Equilibrist fits.

This module is stateless: every public function takes plain numerical inputs
(an SSR, residual arrays, two stats dicts, ...) and returns a scalar / dict /
matplotlib Figure.  It does not modify the fit modules; it consumes their
``stats`` dicts after the fact.

Layers
------

1.  Pure-Python information criteria
        ``model_selection_metrics(ssr, n_points, n_params)``
        Returns AIC, AICc, BIC and the Gaussian log-likelihood.  Designed
        to be merged into the ``stats`` dict produced by any fit function.

2.  Residual analysis
        ``residual_diagnostics(y_obs, y_calc)``
        Returns the residual vector, standardised residuals, SSR, residual
        variance, RMSE, R², Durbin-Watson statistic and (when SciPy is
        available) a Shapiro–Wilk p-value.

3.  Residual collection — one helper per fit mode
        ``collect_residuals_conc(parsed, network, exp_data, params,
                                 final_logKs, x_expr, ...)``
        ``collect_residuals_nmr_shift(stats, parsed, network, nmr_data,
                                      params, final_logKs)``
        ``collect_residuals_spectra(stats)``
        ``collect_residuals_kinetics(stats, parsed, exp_data, logk_dict,
                                     t_max, n_pts)``
        Each returns a dict::

            {
              'y_obs':  ndarray,    # 1-D observations
              'y_calc': ndarray,    # 1-D model predictions
              'x':      ndarray,    # 1-D abscissa for plotting
              'labels': list[str],  # per-point group label (e.g. column name)
              'xlabel': str,
              'ylabel': str,
            }

4.  Model comparison
        ``compare_models(stats_a, stats_b, *, nested=False,
                         label_a='Model A', label_b='Model B')``
        Returns ΔAIC, ΔAICc, ΔBIC, Akaike weights and, when ``nested=True``,
        the F-statistic and its p-value.

5.  Augmentation helper
        ``augment_stats(stats, residuals_dict=None)``
        Adds AIC, AICc, BIC, log-likelihood and (if residuals supplied) the
        Durbin-Watson statistic and normality p-value to an existing stats
        dict in-place.  Safe to call multiple times.

6.  Streamlit rendering
        ``render_diagnostics_panel(stats, residuals_dict, *, key='diag')``
        Draws a four-panel diagnostic display: residual-vs-predictor plot,
        residual histogram, normal Q-Q plot, and a compact metrics table
        including AIC/AICc/BIC/DW.  Uses matplotlib so the figures can also
        be embedded in the Excel/PDF export pipeline.

All public functions are pure-Python + NumPy.  SciPy is optional and only
gates the Shapiro–Wilk p-value and the Q-Q plot quantile theoretical line.
Streamlit and matplotlib are optional — the module imports them lazily so
the diagnostics layer can also be called from headless scripts and tests.

References
----------
Burnham, K. P.; Anderson, D. R.  *Model Selection and Multimodel
Inference*, 2nd ed.; Springer, 2002.
Hibbert, D. B.; Thordarson, P.  *Chem. Commun.* **2016**, *52*, 12792.
Durbin, J.; Watson, G. S.  *Biometrika* **1950**, *37*, 409.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# SciPy is optional; we gate the Shapiro–Wilk and F-test p-values on it.
try:
    from scipy import stats as _sp_stats     # type: ignore
    _HAS_SCIPY = True
except Exception:                            # pragma: no cover
    _HAS_SCIPY = False


__all__ = [
    "model_selection_metrics",
    "residual_diagnostics",
    "compare_models",
    "augment_stats",
    "collect_residuals_conc",
    "collect_residuals_nmr_shift",
    "collect_residuals_spectra",
    "collect_residuals_kinetics",
    "render_diagnostics_panel",
]


# ---------------------------------------------------------------------------
# 1. Information criteria
# ---------------------------------------------------------------------------

def model_selection_metrics(ssr: float,
                            n_points: int,
                            n_params: int) -> dict:
    """Compute AIC, AICc and BIC under the Gaussian-residuals likelihood.

    Equilibrist uses the *truncated* log-likelihood convention that
    matches the manuscript equations (10)–(12).  Under independent
    Gaussian residuals with variance σ² estimated by σ̂² = SSR/n, the
    Gaussian log-likelihood is

        ℓ_full = − (n/2) · [ ln(2π) + ln(σ̂²) + 1 ]

    The two additive constants − (n/2)·ln(2π) and − n/2 cancel in any
    pairwise comparison of models fitted to the same data, so we report

        ℓ_max = − (n/2) · ln(SSR/n)            (manuscript eq. 9-equiv.)
        AIC   = − 2 ℓ_max + 2 p
              = n·ln(SSR/n) + 2 p              (manuscript eq. 10)
        AICc  = AIC + 2 p (p + 1) / (n − p − 1) (manuscript eq. 11)
        BIC   = − 2 ℓ_max + p · ln(n)
              = n·ln(SSR/n) + p · ln(n)        (manuscript eq. 12)

    where p = number of free fitted parameters (σ² is treated as a
    nuisance parameter and is NOT counted in p; the σ²-as-parameter
    convention adds an irrelevant constant to AIC/BIC and shifts the
    AICc denominator by one, which would just rename our (n − p − 1)
    to (n − p − 2)).  Differences ΔAIC, ΔAICc, ΔBIC between models
    fitted to the same data are identical under either convention; only
    absolute values differ.  AICc is the small-sample correction
    recommended whenever n / p < 40.

    Parameters
    ----------
    ssr :
        Sum of squared residuals from the converged fit (matches
        ``stats['ssr']``).
    n_points :
        Number of experimental data points used to form the SSR (matches
        ``stats['n_points']``).
    n_params :
        Number of free parameters that were fitted to the data (matches
        ``stats['n_params']``).  Does *not* include species fully determined
        by mass balance or the residual variance σ².

    Returns
    -------
    dict
        ``{'log_likelihood', 'aic', 'aicc', 'bic', 'k_effective',
        'n_points', 'n_params'}``.  Returns NaNs for the criteria when the
        inputs are degenerate (SSR ≤ 0, n ≤ p+1).  ``k_effective`` is
        retained as a key for backward compatibility with downstream
        consumers and is set equal to ``n_params`` (i.e. p).

    Notes
    -----
    The absolute values of AIC/BIC are not meaningful — only differences
    (ΔAIC, ΔBIC) between models fitted to the same data are interpretable.
    Use :func:`compare_models` for that.
    """
    n = int(n_points)
    p = int(n_params)

    out = {
        "log_likelihood": float("nan"),
        "aic":            float("nan"),
        "aicc":           float("nan"),
        "bic":            float("nan"),
        "k_effective":    p,
        "n_points":       n,
        "n_params":       p,
    }

    if ssr is None or not np.isfinite(ssr) or ssr <= 0.0:
        return out
    if n <= 0 or p < 0:
        return out

    sigma2 = ssr / n
    # Truncated log-likelihood: ℓ_max = −(n/2)·ln(SSR/n).  Additive
    # Gaussian constants are absorbed; see the docstring.
    log_lik = -0.5 * n * math.log(sigma2)
    aic = -2.0 * log_lik + 2.0 * p              # = n·ln(SSR/n) + 2p
    bic = -2.0 * log_lik + p * math.log(n)      # = n·ln(SSR/n) + p·ln(n)
    # AICc undefined when n − p − 1 ≤ 0; report inf so the caller can flag.
    denom = n - p - 1
    aicc = aic + (2.0 * p * (p + 1)) / denom if denom > 0 else float("inf")

    out.update({
        "log_likelihood": float(log_lik),
        "aic":            float(aic),
        "aicc":           float(aicc),
        "bic":            float(bic),
    })
    return out


# ---------------------------------------------------------------------------
# 2. Residual analysis
# ---------------------------------------------------------------------------

def residual_diagnostics(y_obs, y_calc) -> dict:
    """Compute residual-based diagnostics from observed/calculated arrays.

    All arrays are coerced to 1-D float64.  Non-finite pairs are silently
    dropped (so callers do not have to pre-filter).

    Returns
    -------
    dict containing
        - ``residuals``       : 1-D ndarray, ``y_obs − y_calc``
        - ``standardised``    : residuals divided by ``s = sqrt(SSR/(n−p))``
                                with ``p = 1`` placeholder; for proper
                                standardised residuals the caller should
                                recompute using their own dof
        - ``n``               : number of finite residuals
        - ``ssr``             : sum of squared residuals
        - ``mean``            : residual mean
        - ``std``             : residual standard deviation
        - ``rmse``            : root mean squared residual
        - ``r_squared``       : 1 − SSR/SST (only meaningful for arrays
                                with non-trivial variance in y_obs)
        - ``durbin_watson``   : DW statistic (≈ 2 → no autocorrelation,
                                ≈ 0 → positive autocorrelation,
                                ≈ 4 → negative autocorrelation)
        - ``shapiro_p``       : Shapiro–Wilk p-value if SciPy is available
                                and n ≥ 3, else NaN
        - ``skew``, ``kurtosis``: sample skewness and excess kurtosis

    Notes
    -----
    The Durbin-Watson statistic only makes sense when residuals have a
    natural ordering (e.g. by titrant volume or by time).  The caller is
    responsible for passing arrays in that order; if the order is arbitrary
    DW is uninformative.
    """
    y_obs  = np.asarray(y_obs,  dtype=float).ravel()
    y_calc = np.asarray(y_calc, dtype=float).ravel()
    n_in = min(len(y_obs), len(y_calc))
    y_obs, y_calc = y_obs[:n_in], y_calc[:n_in]
    mask = np.isfinite(y_obs) & np.isfinite(y_calc)
    y_obs, y_calc = y_obs[mask], y_calc[mask]
    r = y_obs - y_calc
    n = len(r)

    out = {
        "residuals":     r,
        "standardised":  np.zeros_like(r),
        "n":             n,
        "ssr":           float("nan"),
        "mean":          float("nan"),
        "std":           float("nan"),
        "rmse":          float("nan"),
        "r_squared":     float("nan"),
        "durbin_watson": float("nan"),
        "shapiro_p":     float("nan"),
        "skew":          float("nan"),
        "kurtosis":      float("nan"),
    }
    if n == 0:
        return out

    ssr = float(np.sum(r * r))
    mean = float(np.mean(r))
    std  = float(np.std(r, ddof=1)) if n > 1 else 0.0
    rmse = float(math.sqrt(ssr / n))
    sst  = float(np.sum((y_obs - y_obs.mean()) ** 2))
    r2   = 1.0 - ssr / sst if sst > 1e-30 else float("nan")

    # Durbin–Watson
    if n > 1:
        dr = np.diff(r)
        dw = float(np.sum(dr * dr) / max(ssr, 1e-30))
    else:
        dw = float("nan")

    # Standardised residuals: σ̂ = sqrt(SSR/(n−1)) as a denominator that
    # does not require the caller to know p.  This is conservative; the
    # plotting code only uses these to label points as |z| > 3 outliers.
    if n > 1 and ssr > 0:
        sigma_hat = math.sqrt(ssr / max(n - 1, 1))
        z = r / sigma_hat
    else:
        z = np.zeros_like(r)

    # Sample skewness and excess kurtosis (without SciPy)
    if n > 2 and std > 0:
        m3 = float(np.mean((r - mean) ** 3))
        m4 = float(np.mean((r - mean) ** 4))
        skew = m3 / (std ** 3)
        kurt = m4 / (std ** 4) - 3.0
    else:
        skew = float("nan")
        kurt = float("nan")

    shapiro_p = float("nan")
    if _HAS_SCIPY and 3 <= n <= 5000:
        try:
            _, shapiro_p = _sp_stats.shapiro(r)
            shapiro_p = float(shapiro_p)
        except Exception:
            pass

    out.update({
        "residuals":     r,
        "standardised":  z,
        "n":             n,
        "ssr":           ssr,
        "mean":          mean,
        "std":           std,
        "rmse":          rmse,
        "r_squared":     float(r2),
        "durbin_watson": dw,
        "shapiro_p":     shapiro_p,
        "skew":          float(skew),
        "kurtosis":      float(kurt),
    })
    return out


# ---------------------------------------------------------------------------
# 3. Residual collection by fit mode
# ---------------------------------------------------------------------------
#
# These helpers reproduce the forward pass of each fit module so we can
# expose (y_obs, y_calc) pairs.  We do *not* import from the fit modules
# directly to avoid circular imports — instead we accept the dependencies
# the caller already has on hand (parsed, network, params, final logKs).
#
# All four return the same dict shape, listed at the top of this file.

def collect_residuals_from_stats(stats: dict, *,
                                  xlabel: str = "",
                                  ylabel: str = "",
                                  x_expr: str = "") -> dict:
    """Unified residual collector for v2-patched fit modules.

    Every fit function in the v2 stack stores flat ``y_obs``, ``y_calc``,
    and ``residuals`` arrays directly in its stats dict.  This function
    just reads them and packages them in the dict layout expected by
    ``augment_stats`` / ``make_diagnostics_figure``.

    Works for every fit mode: concentration, NMR-shift, NMR-integration,
    NMR-mixed, spectra, kinetics, kinetics+NMR (all variants), and
    kinetics+spectra.  Returns empty arrays gracefully if stats lacks
    the v2 keys (older snapshots).

    For spectra modes (``spectra`` / ``kinetics_spectra``), the residual
    matrix ``A_obs`` / ``A_calc`` (n_spectra × n_wavelengths) is used to
    place wavelength on the x-axis and colour residuals by spectrum.

    For NMR modes (``shift`` / ``integration`` / ``mixed``) and the
    kinetics-NMR variants, the per-column arrays ``per_col_y_obs`` /
    ``per_col_y_calc`` / ``per_col_x`` are preferred when present, so
    residuals are plotted against the actual titration coordinate or
    time (rather than the data-point index) and coloured by column
    label.  The ``x_expr`` kwarg supplies a human-readable axis label
    (e.g. ``"H0/G0"`` from the script's ``$plot x = H0/G0`` directive).
    """
    fit_mode = (stats.get("fit_mode") or "").lower()

    # ── Spectra-mode special path: λ on x-axis, label by spectrum ────────
    A_obs  = stats.get("A_obs")
    A_calc = stats.get("A_calc")
    if (fit_mode in ("spectra", "kinetics_spectra")
            and A_obs is not None and A_calc is not None):
        try:
            A_obs  = np.asarray(A_obs,  dtype=float)
            A_calc = np.asarray(A_calc, dtype=float)
            if A_obs.ndim == 2 and A_obs.shape == A_calc.shape:
                n_spectra, n_wl = A_obs.shape
                wl = np.asarray(stats.get("wavelengths_fit",
                                          np.arange(n_wl)), dtype=float)
                yo = A_obs.ravel()
                yc = A_calc.ravel()
                x  = np.tile(wl, n_spectra)
                lb = []
                for i in range(n_spectra):
                    lb.extend([f"spectrum {i+1}"] * n_wl)
                _xlabel = (xlabel if xlabel else
                           ("Wavelength / nm" if len(wl) and wl[-1] > 100
                            else "Wavelength index"))
                _ylabel = ylabel if ylabel else "Absorbance"
                return {"y_obs": yo, "y_calc": yc, "x": x, "labels": lb,
                        "xlabel": _xlabel, "ylabel": _ylabel}
        except Exception:
            pass

    # ── NMR per-column path: use per_col_x for real titration / time x ───
    # NMR-shift, NMR-integration, NMR-mixed (equilibrium) and the
    # kinetics+NMR variants all store per-column observed/calc/x arrays
    # in stats.  Using them lets the residual scatter put residuals at
    # the right x-position (H0/G0, time, etc.) and colour them by NMR
    # signal — much more diagnostic than a flat 0..N-1 index.
    pc_x  = stats.get("per_col_x")  or {}
    pc_yo = stats.get("per_col_y_obs")  or {}
    pc_yc = stats.get("per_col_y_calc") or {}
    if pc_x and pc_yo and pc_yc:
        try:
            x_list, yo_list, yc_list, lab_list = [], [], [], []
            # Iterate in insertion order so columns stay grouped
            for col in pc_x.keys():
                _x  = np.asarray(pc_x.get(col,  []), dtype=float).ravel()
                _yo = np.asarray(pc_yo.get(col, []), dtype=float).ravel()
                _yc = np.asarray(pc_yc.get(col, []), dtype=float).ravel()
                n_col = min(_x.size, _yo.size, _yc.size)
                if n_col == 0:
                    continue
                x_list.extend(_x[:n_col].tolist())
                yo_list.extend(_yo[:n_col].tolist())
                yc_list.extend(_yc[:n_col].tolist())
                lab_list.extend([str(col)] * n_col)
            if x_list:
                # Disambiguate kinetics+NMR (fit_mode = "shift"/"integration"
                # /"mixed" same as equilibrium NMR) by checking the
                # ``is_kinetics`` flag — populated by every kinetics+NMR
                # fit module — so the time axis isn't mislabeled as
                # the script's titration coordinate.
                _is_kin = (fit_mode == "kinetics"
                           or fit_mode.startswith("kinetics")
                           or bool(stats.get("is_kinetics")))
                # Caller hint > kinetics detection > x_expr from script > generic
                if not xlabel:
                    if _is_kin:
                        xlabel = "Time / s"
                    elif x_expr:
                        xlabel = x_expr
                    else:
                        xlabel = "Titration coordinate"
                if not ylabel:
                    if fit_mode in ("shift", "mixed"):
                        ylabel = "Δδ / ppm"
                    elif fit_mode == "integration":
                        ylabel = "Concentration / mM"
                    elif _is_kin:
                        ylabel = "Concentration / mM"
                    else:
                        ylabel = "Observed signal"
                return {
                    "y_obs":  np.asarray(yo_list, dtype=float),
                    "y_calc": np.asarray(yc_list, dtype=float),
                    "x":      np.asarray(x_list,  dtype=float),
                    "labels": lab_list,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                }
        except Exception:
            pass

    # ── Fallback (e.g. older snapshots without per_col arrays): index x ──
    # NOTE: we deliberately do NOT use ``x_expr`` for the xlabel here,
    # because the x VALUES in this path are 0..N-1 indices (per_col_x
    # was missing).  Labeling an index axis with "H0/G0" would be a
    # lie.  Every v2-patched fit module DOES store per_col_x, so this
    # path should only fire for legacy/custom fit modules.
    _yo = stats.get("y_obs")
    _yc = stats.get("y_calc")
    _rs = stats.get("residuals")
    yo = np.asarray(_yo if _yo is not None else [], dtype=float).ravel()
    yc = np.asarray(_yc if _yc is not None else [], dtype=float).ravel()
    rs = np.asarray(_rs if _rs is not None else [], dtype=float).ravel()
    if yo.size == 0 and rs.size:
        # Some fits may store residuals but not y_obs (rare); reconstruct
        if yc.size == rs.size:
            yo = yc + rs
    if yc.size == 0 and yo.size and rs.size == yo.size:
        yc = yo - rs
    n = min(yo.size, yc.size)
    yo, yc = yo[:n], yc[:n]
    x = np.arange(n, dtype=float)
    if not xlabel:
        if   fit_mode == "kinetics":           xlabel = "Time-point index"
        elif fit_mode.startswith("kinetics"):  xlabel = "Data-point index"
        else:                                   xlabel = "Data-point index"
    if not ylabel:
        if   fit_mode in ("shift", "mixed"):  ylabel = "Δδ / ppm"
        elif fit_mode == "integration":       ylabel = "Concentration / mM"
        elif fit_mode == "kinetics":          ylabel = "Concentration / mM"
        else:                                  ylabel = "Observed signal"
    return {
        "y_obs":  yo, "y_calc": yc,
        "x":      x,
        "labels": [""] * n,
        "xlabel": xlabel, "ylabel": ylabel,
    }


def collect_residuals_conc(parsed, network, exp_data, params, final_logKs,
                           x_expr, *, x_col_header: str = "") -> dict:
    """Return residuals from a concentration-mode fit.

    Fast path (v2): read flat ``y_obs`` / ``y_calc`` arrays straight from
    ``stats`` when present.  Legacy path: reproduce the forward pass via
    ``compute_single_point``.  Kept for backward compatibility with code
    that doesn't pass ``stats``.
    """
    # The fast path requires a stats dict, which this signature doesn't take.
    # Caller can use ``collect_residuals_from_stats(stats)`` directly if the
    # fit module has already stored y_obs/y_calc (true for v2-patched fits).
    from equilibrist_curve import (compute_single_point, find_equiv_for_x,
                                   convert_exp_x)

    y_obs_list, y_calc_list, x_list, lab_list = [], [], [], []
    for exp_col, col_data in exp_data.items():
        if exp_col.startswith("_"):
            continue
        try:
            v_add = col_data["v_add_mL"]
            y_arr = col_data["y"]
        except Exception:
            continue
        for i in range(len(v_add)):
            v_i, y_i = float(v_add[i]), float(y_arr[i])
            if not (np.isfinite(v_i) and np.isfinite(y_i)):
                continue
            try:
                x_val = convert_exp_x(np.array([v_i]), x_expr, parsed, params,
                                      network, x_col_header=x_col_header)[0]
                equiv = find_equiv_for_x(x_val, parsed, params)
                theo  = compute_single_point(equiv, parsed, network,
                                             final_logKs, params, exp_col)
            except Exception:
                continue
            if not np.isfinite(theo):
                continue
            y_obs_list.append(y_i)
            y_calc_list.append(float(theo))
            x_list.append(float(x_val))
            lab_list.append(str(exp_col))

    return {
        "y_obs":  np.asarray(y_obs_list,  dtype=float),
        "y_calc": np.asarray(y_calc_list, dtype=float),
        "x":      np.asarray(x_list,      dtype=float),
        "labels": lab_list,
        "xlabel": "Titration coordinate",
        "ylabel": "Concentration / mM",
    }


def collect_residuals_nmr_shift(stats: dict, nmr_data: dict) -> dict:
    """Extract (δ_obs, δ_calc) from the NMR shift fit ``stats`` dict.

    The fit_nmr module stores per-column observed and calculated shift
    vectors in ``stats['delta_vecs_all']``.  We unpack them here.  When
    that key is unavailable (older snapshots), we fall back to whatever
    ``stats`` exposes and raise a quiet warning by returning an empty dict.
    """
    delta_vecs = stats.get("delta_vecs_all") or {}
    delta_bound = stats.get("delta_bound_all") or {}
    # Each delta_vecs[col] is (x_arr, y_obs_arr); delta_bound[col] is the
    # calculated bound-shift trajectory at the same x.  Variations exist
    # depending on how fit_nmr stored them; we handle both layouts.
    y_obs_list, y_calc_list, x_list, lab_list = [], [], [], []
    for col, entry in delta_vecs.items():
        try:
            if isinstance(entry, tuple) and len(entry) >= 2:
                x_arr, y_obs_arr = np.asarray(entry[0]), np.asarray(entry[1])
            else:
                continue
            calc_entry = delta_bound.get(col)
            if calc_entry is None:
                continue
            if isinstance(calc_entry, tuple) and len(calc_entry) >= 2:
                x_calc, y_calc_arr = (np.asarray(calc_entry[0]),
                                      np.asarray(calc_entry[1]))
                # Interpolate calc onto obs x if grids differ
                if not (len(x_calc) == len(x_arr) and
                        np.allclose(x_calc, x_arr)):
                    order = np.argsort(x_calc)
                    y_calc_arr = np.interp(x_arr, x_calc[order],
                                           y_calc_arr[order])
            else:
                y_calc_arr = np.asarray(calc_entry)
                if len(y_calc_arr) != len(x_arr):
                    continue
        except Exception:
            continue
        for x_i, yo, yc in zip(x_arr, y_obs_arr, y_calc_arr):
            if not (np.isfinite(yo) and np.isfinite(yc)):
                continue
            y_obs_list.append(float(yo))
            y_calc_list.append(float(yc))
            x_list.append(float(x_i))
            lab_list.append(str(col))

    return {
        "y_obs":  np.asarray(y_obs_list,  dtype=float),
        "y_calc": np.asarray(y_calc_list, dtype=float),
        "x":      np.asarray(x_list,      dtype=float),
        "labels": lab_list,
        "xlabel": "Titration coordinate",
        "ylabel": "δ / ppm",
    }


def collect_residuals_spectra(stats: dict) -> dict:
    """Extract residuals from a spectra fit using the ``stats`` snapshot.

    fit_spectra stores the wavelength grid (``wavelengths_fit``), the
    fitted concentration matrix (``C_back``) and the extinction matrix
    (``E_final``) — we reconstruct ``A_calc`` and pair it with ``A_obs``
    which the caller supplies as ``stats['A_obs']`` if available; if not,
    we operate on the residual matrix already present.
    """
    # Try the cleanest path first: reconstruct from C_back and E_final.
    # If unavailable, fall back to whatever is in stats.
    y_obs_list, y_calc_list, x_list, lab_list = [], [], [], []
    A_obs  = stats.get("A_obs")
    A_calc = stats.get("A_calc")
    wl     = stats.get("wavelengths_fit")
    x_exp  = stats.get("x_exp")
    if A_obs is None or A_calc is None:
        # Fall back: empty result; caller should add A_obs/A_calc to stats
        # at fit time using the integration patch (see app.py patch).
        return {
            "y_obs":  np.array([], dtype=float),
            "y_calc": np.array([], dtype=float),
            "x":      np.array([], dtype=float),
            "labels": [],
            "xlabel": "Wavelength / nm",
            "ylabel": "Absorbance",
        }
    A_obs  = np.asarray(A_obs,  dtype=float)
    A_calc = np.asarray(A_calc, dtype=float)
    wl     = np.asarray(wl,     dtype=float) if wl is not None else None
    x_exp  = np.asarray(x_exp,  dtype=float) if x_exp is not None else None
    n_spectra, n_wl = A_obs.shape
    for s in range(n_spectra):
        for j in range(n_wl):
            yo = A_obs[s, j]; yc = A_calc[s, j]
            if not (np.isfinite(yo) and np.isfinite(yc)):
                continue
            y_obs_list.append(float(yo))
            y_calc_list.append(float(yc))
            x_list.append(float(wl[j]) if wl is not None else float(j))
            lab_list.append(f"spec_{s:03d}")

    return {
        "y_obs":  np.asarray(y_obs_list,  dtype=float),
        "y_calc": np.asarray(y_calc_list, dtype=float),
        "x":      np.asarray(x_list,      dtype=float),
        "labels": lab_list,
        "xlabel": "Wavelength / nm",
        "ylabel": "Absorbance",
    }


def collect_residuals_kinetics(stats: dict, exp_data: dict,
                               sim_curve: dict) -> dict:
    """Extract (y_obs, y_calc) from a kinetics fit.

    ``sim_curve`` is the final integrated trajectory dict returned by
    ``compute_kinetics_curve``: ``{'t': ndarray, sp1: ndarray, ...}``.
    ``exp_data`` has the same shape as in the fit:
    ``{species_name: {'v_add_mL': t_array, 'y': y_array}}``.
    """
    y_obs_list, y_calc_list, x_list, lab_list = [], [], [], []
    t_grid = np.asarray(sim_curve.get("t", []), dtype=float)
    for sp, col_data in exp_data.items():
        if sp.startswith("_"):
            continue
        if sp not in sim_curve:
            continue
        try:
            t_obs = np.asarray(col_data["v_add_mL"], dtype=float)
            y_obs = np.asarray(col_data["y"],        dtype=float)
        except Exception:
            continue
        c_sp = np.asarray(sim_curve[sp], dtype=float)
        order = np.argsort(t_grid)
        y_calc = np.interp(t_obs, t_grid[order], c_sp[order])
        for t_i, yo, yc in zip(t_obs, y_obs, y_calc):
            if not (np.isfinite(yo) and np.isfinite(yc)):
                continue
            y_obs_list.append(float(yo))
            y_calc_list.append(float(yc))
            x_list.append(float(t_i))
            lab_list.append(str(sp))

    return {
        "y_obs":  np.asarray(y_obs_list,  dtype=float),
        "y_calc": np.asarray(y_calc_list, dtype=float),
        "x":      np.asarray(x_list,      dtype=float),
        "labels": lab_list,
        "xlabel": "Time / s",
        "ylabel": "Concentration / mM",
    }


# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------

def compare_models(stats_a: dict, stats_b: dict, *,
                   nested: bool = False,
                   label_a: str = "Model A",
                   label_b: str = "Model B") -> dict:
    """Compare two converged fits of the same data.

    Both ``stats_a`` and ``stats_b`` should have AIC/AICc/BIC populated
    (either by passing them through :func:`augment_stats`, or because the
    integration patch in this module has been applied to the fit-driver
    code in ``app.py``).

    Parameters
    ----------
    stats_a, stats_b :
        Stats dicts from two converged fits to the *same* experimental
        data.  Comparing fits to different data is meaningless.
    nested :
        If True, the simpler model (the one with fewer parameters) is
        assumed to be a special case of the more complex one, and an
        F-test for the additional parameters is reported.  The default
        is False because nested-ness is not always obvious from a script
        alone (e.g. a 1:1 vs 1:2 model is nested only if logK₂ → −∞
        recovers the 1:1 model exactly — typically true for stepwise
        binding models, not always for kinetic schemes).
    label_a, label_b :
        Human-readable names for the comparison report.

    Returns
    -------
    dict with keys
        - ``best_by_aic``      : 'A' or 'B'
        - ``best_by_aicc``     : 'A' or 'B'
        - ``best_by_bic``      : 'A' or 'B'
        - ``delta_aic``        : AIC_B − AIC_A    (positive ⇒ A preferred)
        - ``delta_aicc``       : AICc_B − AICc_A
        - ``delta_bic``        : BIC_B − BIC_A
        - ``weight_a``         : exp(−ΔAIC_A / 2) normalised (Akaike weight)
        - ``weight_b``         : exp(−ΔAIC_B / 2) normalised
        - ``f_statistic``      : F-statistic when ``nested=True`` (else NaN)
        - ``f_p_value``        : F-test p-value when ``nested=True`` and
                                 SciPy is available (else NaN)
        - ``interpretation``   : one-line plain-English summary
        - ``label_a``, ``label_b``
    """
    def _aic_aicc_bic(s):
        a = s.get("aic");  ac = s.get("aicc");  b = s.get("bic")
        if a is None or ac is None or b is None:
            m = model_selection_metrics(s.get("ssr", float("nan")),
                                        s.get("n_points", 0),
                                        s.get("n_params", 0))
            a, ac, b = m["aic"], m["aicc"], m["bic"]
        return float(a), float(ac), float(b)

    aic_a, aicc_a, bic_a = _aic_aicc_bic(stats_a)
    aic_b, aicc_b, bic_b = _aic_aicc_bic(stats_b)

    d_aic  = aic_b  - aic_a
    d_aicc = aicc_b - aicc_a
    d_bic  = bic_b  - bic_a

    # Akaike weights
    aic_min = min(aic_a, aic_b)
    wa = math.exp(-0.5 * (aic_a - aic_min))
    wb = math.exp(-0.5 * (aic_b - aic_min))
    wsum = wa + wb if (wa + wb) > 0 else 1.0
    wa /= wsum; wb /= wsum

    # F-test for nested models
    f_stat = float("nan"); f_p = float("nan")
    if nested:
        # Identify simpler vs more complex
        if stats_a.get("n_params", 0) <= stats_b.get("n_params", 0):
            s_simple, s_full = stats_a, stats_b
        else:
            s_simple, s_full = stats_b, stats_a
        ssr_s = float(s_simple.get("ssr",      float("nan")))
        ssr_f = float(s_full.get  ("ssr",      float("nan")))
        p_s   = int  (s_simple.get("n_params", 0))
        p_f   = int  (s_full.get  ("n_params", 0))
        n     = int  (s_full.get  ("n_points", 0))
        if (np.isfinite(ssr_s) and np.isfinite(ssr_f)
                and ssr_f > 0 and p_f > p_s and n - p_f > 0):
            num = (ssr_s - ssr_f) / (p_f - p_s)
            den = ssr_f / (n - p_f)
            if den > 0:
                if num <= 0:
                    # Full model didn't actually reduce SSR — the extra
                    # parameter explains zero variance.  Report F = 0,
                    # p = 1 ("no evidence whatsoever for the extra
                    # parameter") rather than NaN.
                    f_stat = 0.0
                    f_p    = 1.0
                else:
                    f_stat = float(num / den)
                    if _HAS_SCIPY:
                        try:
                            f_p = float(_sp_stats.f.sf(f_stat,
                                                       p_f - p_s,
                                                       n - p_f))
                        except Exception:
                            pass

    best_aic  = label_a if d_aic  > 0 else label_b
    best_aicc = label_a if d_aicc > 0 else label_b
    best_bic  = label_a if d_bic  > 0 else label_b

    # Plain-English interpretation following Burnham & Anderson rules of
    # thumb: |ΔAIC| < 2 ≈ comparable; 4–7 considerably less support;
    # > 10 essentially no support for the worse model.
    abs_da = abs(d_aic)
    if abs_da < 2:
        interp = (f"{best_aic} is preferred by AIC but the support is weak "
                  f"(|ΔAIC| = {abs_da:.2f} < 2).")
    elif abs_da < 7:
        interp = (f"{best_aic} is preferred by AIC (|ΔAIC| = {abs_da:.2f}; "
                  "moderate support).")
    else:
        interp = (f"{best_aic} is strongly preferred by AIC "
                  f"(|ΔAIC| = {abs_da:.2f}; the other model has "
                  "essentially no empirical support).")

    return {
        "label_a": label_a,
        "label_b": label_b,
        "aic_a":   aic_a,  "aic_b":   aic_b,
        "aicc_a":  aicc_a, "aicc_b":  aicc_b,
        "bic_a":   bic_a,  "bic_b":   bic_b,
        "delta_aic":  d_aic,
        "delta_aicc": d_aicc,
        "delta_bic":  d_bic,
        "best_by_aic":  best_aic,
        "best_by_aicc": best_aicc,
        "best_by_bic":  best_bic,
        "weight_a":     wa,
        "weight_b":     wb,
        "f_statistic":  f_stat,
        "f_p_value":    f_p,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 5. Augmentation
# ---------------------------------------------------------------------------

def augment_stats(stats: dict, residuals_dict: Optional[dict] = None) -> dict:
    """Add AIC/AICc/BIC and (optionally) residual diagnostics to ``stats``.

    Modifies ``stats`` in-place *and* returns it for convenience.  Safe to
    call repeatedly — keys are overwritten with fresh values each time.

    Parameters
    ----------
    stats :
        A stats dict as returned by any Equilibrist fit function.  Must
        contain ``ssr``, ``n_points`` and ``n_params``.
    residuals_dict :
        Output of one of the ``collect_residuals_*`` helpers.  When
        supplied, Durbin-Watson and normality diagnostics are added.
    """
    m = model_selection_metrics(stats.get("ssr", float("nan")),
                                stats.get("n_points", 0),
                                stats.get("n_params", 0))
    stats["log_likelihood"] = m["log_likelihood"]
    stats["aic"]            = m["aic"]
    stats["aicc"]           = m["aicc"]
    stats["bic"]            = m["bic"]

    if residuals_dict is not None:
        d = residual_diagnostics(residuals_dict.get("y_obs"),
                                 residuals_dict.get("y_calc"))
        # Don't overwrite r_squared / rmse from the fit (they are mode-
        # specific in spectra and NMR), but expose the residual-array
        # statistics under distinct keys.
        stats["durbin_watson"] = d["durbin_watson"]
        stats["shapiro_p"]     = d["shapiro_p"]
        stats["res_mean"]      = d["mean"]
        stats["res_std"]       = d["std"]
        stats["res_skew"]      = d["skew"]
        stats["res_kurtosis"]  = d["kurtosis"]
    return stats


# ---------------------------------------------------------------------------
# 6. Streamlit / matplotlib rendering
# ---------------------------------------------------------------------------

def _qq_theoretical_quantiles(n: int):
    """Approximate normal quantiles for a Q-Q plot without SciPy.

    Uses the Acklam algorithm — a rational approximation accurate to
    about 1.15 × 10⁻⁹ across the central distribution.  Works for n ≤
    50 000; beyond that the SciPy path is preferable.
    """
    if _HAS_SCIPY:
        try:
            from scipy.stats import norm  # type: ignore
            ps = (np.arange(1, n + 1) - 0.5) / n
            return norm.ppf(ps)
        except Exception:
            pass
    # Acklam coefficients
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    out = np.empty(n)
    ps = (np.arange(1, n + 1) - 0.5) / n
    for i, p in enumerate(ps):
        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            out[i] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) \
                     / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        elif p <= p_high:
            q = p - 0.5; r = q * q
            out[i] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q \
                     / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            out[i] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) \
                      / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return out


def make_diagnostics_figure(stats: dict, residuals_dict: dict,
                            *, title: Optional[str] = None,
                            rmse_profile: Optional[dict] = None):
    """Return a matplotlib Figure with the four diagnostic panels.

    Self-contained — does not import or touch Streamlit.  Suitable for
    embedding into the PDF/Excel export pipeline.

    If ``rmse_profile`` is provided (mapping ``param_name`` → ``(values,
    rmse_at_value)`` arrays), the SE panel is used for the parameter
    profile plot and the fit-metrics summary moves to a strip below the
    figure.  When ``rmse_profile`` is ``None`` (default) the SE panel
    contains the fit-metrics table, preserving the original layout.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})

    y_obs   = np.asarray(residuals_dict.get("y_obs",  []), dtype=float)
    y_calc  = np.asarray(residuals_dict.get("y_calc", []), dtype=float)
    x_vals  = np.asarray(residuals_dict.get("x",      []), dtype=float)
    labels  = list(residuals_dict.get("labels", []))
    xlabel  = residuals_dict.get("xlabel", "x")
    ylabel  = residuals_dict.get("ylabel", "y")

    r = y_obs - y_calc if len(y_obs) and len(y_calc) else np.array([])
    n = len(r)

    # When the parameter profile is shown in the SE panel, reserve a bottom
    # strip of the figure for the fit-metrics summary via figtext.
    _has_profile = bool(rmse_profile)
    _fig_h = 7.7 if _has_profile else 6.5
    fig, axes = plt.subplots(2, 2, figsize=(8, _fig_h),
                             constrained_layout=False)
    if _has_profile:
        # Hand-tuned spacing so the metrics strip at the bottom does not
        # overlap with the x-axis tick labels of the lower-row plots.
        fig.subplots_adjust(left=0.085, right=0.97,
                            top=0.93 if title else 0.97,
                            bottom=0.22, wspace=0.30, hspace=0.42)
    else:
        fig.set_layout_engine("constrained")
    if title:
        fig.suptitle(title, fontsize=11)

    # ── (1) Residual vs predictor ────────────────────────────────────────
    ax = axes[0, 0]
    if n:
        # Colour per label so the user sees which dataset contributes
        unique = sorted(set(labels))
        for lab in unique:
            idx = [i for i, L in enumerate(labels) if L == lab]
            ax.scatter(x_vals[idx], r[idx], s=22, alpha=0.75,
                       edgecolor="none", label=lab if len(unique) <= 8 else None)
        ax.axhline(0.0, color="0.4", lw=0.8)
        # ±2σ guide
        if stats.get("ssr") and stats.get("n_points") and stats.get("n_params"):
            try:
                sigma = math.sqrt(stats["ssr"]
                                  / max(stats["n_points"] - stats["n_params"], 1))
                ax.axhline( 2 * sigma, color="0.6", lw=0.6, ls="--")
                ax.axhline(-2 * sigma, color="0.6", lw=0.6, ls="--")
            except Exception:
                pass
        if len(set(labels)) <= 8 and labels:
            ax.legend(loc="best", fontsize=7, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (obs − calc)")
    ax.set_title("Residuals vs predictor")

    # ── (2) Residual histogram ───────────────────────────────────────────
    ax = axes[0, 1]
    if n:
        bins = max(8, int(round(math.sqrt(n))))
        ax.hist(r, bins=bins, edgecolor="0.2", color="0.7")
        ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")

    # ── (3) Normal Q-Q ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if n >= 2:
        s = np.sort(r)
        tq = _qq_theoretical_quantiles(n)
        ax.scatter(tq, s, s=18, alpha=0.8, edgecolor="none")
        if n > 1:
            # Reference line through 1st and 3rd quartiles
            q1, q3 = np.quantile(s, [0.25, 0.75])
            tq1, tq3 = np.quantile(tq, [0.25, 0.75])
            if tq3 > tq1:
                slope = (q3 - q1) / (tq3 - tq1)
                intercept = q1 - slope * tq1
                xs = np.array([tq.min(), tq.max()])
                ax.plot(xs, slope * xs + intercept, color="0.4", lw=0.8)
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Sample quantile")
    ax.set_title("Normal Q-Q")

    # ── (4) SE quadrant: metrics table OR parameter profile ──────────────
    ax = axes[1, 1]
    _fit_mode_low = (stats.get("fit_mode") or "").lower()

    # Always build the metrics rows — they go either in the SE quadrant
    # (no profile) or as a figtext strip below the figure (with profile).
    rows = []
    def _row(label, val, fmt="{:.4g}"):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            rows.append((label, "—"))
        else:
            rows.append((label, fmt.format(val)))
    _row("n",         stats.get("n_points"))
    _row("p (fitted)", stats.get("n_params"))
    _row("SSR",       stats.get("ssr"))
    _row("RMSE",      stats.get("rmse"))
    _row("R²",        stats.get("r_squared"))
    _row("ℓ_max",     stats.get("log_likelihood"))
    _row("AIC",       stats.get("aic"))
    _row("AICc",      stats.get("aicc"))
    _row("BIC",       stats.get("bic"))
    if _fit_mode_low not in ("spectra", "kinetics_spectra"):
        _row("DW",        stats.get("durbin_watson"))
        _row("Shapiro p", stats.get("shapiro_p"))

    if _has_profile:
        # ── SE: parameter profile plot ───────────────────────────────────
        # Overlay all fitted parameters as Δlog vs RMSE/RMSE_min, centered
        # so all curves pass through (0, 1) at their respective optima.
        # A sharp valley = well-determined parameter; flat = unidentifiable.
        ax.axis("on")
        # Reference lines FIRST and BEHIND the data (low zorder).  A flat
        # profile (parameter unidentifiable given the others) sits at
        # y=1.0; if the dashed reference line were drawn on top, it
        # would visually erase the curve and the parameter would look
        # absent from the plot.  Now the colored marker dots always
        # show through.  Same idea for the x=0 reference.
        ax.axhline(1.0, color="0.6", lw=0.7, ls="--", zorder=0.5)
        ax.axvline(0.0, color="0.6", lw=0.7, ls="--", zorder=0.5)
        rmse_min_global = float(stats.get("rmse") or np.nan)
        _palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
                    "#9467BD", "#8C564B", "#E377C2", "#17BECF"]
        _legend_handles = []
        _flat_lines = []   # names whose profile is essentially flat (≤1%)
        for _i, (_pname, _data) in enumerate(rmse_profile.items()):
            try:
                _vals, _rmses = _data
                _vals  = np.asarray(_vals,  dtype=float)
                _rmses = np.asarray(_rmses, dtype=float)
                if _vals.size < 2: continue
                # x is measured from the grid CENTRE -- the fitted (or fixed)
                # value the scan was built around -- not from the grid point
                # with the lowest RMSE: on a flat profile that point is
                # decided by numerical noise and would shift the whole curve.
                _r_opt = float(np.nanmin(_rmses))
                _v_opt = float(_vals[len(_vals) // 2])
                if not (np.isfinite(_r_opt) and _r_opt > 0): continue
                _dx = _vals - _v_opt
                _dy = _rmses / _r_opt
                _c  = _palette[_i % len(_palette)]
                _h, = ax.plot(_dx, _dy, "-o", ms=2.2, lw=1.2,
                              color=_c, label=_pname, zorder=3)
                _legend_handles.append(_h)
                # Flag near-flat profiles for an inline annotation —
                # the user otherwise might think the parameter is
                # missing when its line overlaps y=1.0.
                _peak_ratio = float(np.nanmax(_dy))
                if _peak_ratio < 1.01:
                    _flat_lines.append(_pname)
            except Exception:
                continue
        ax.set_yscale("log")
        ax.set_xlabel("Δ log parameter from its fitted (or fixed) value")
        ax.set_ylabel("RMSE / RMSE$_{min}$  (log scale)")
        ax.set_title("Parameter profile")
        if _legend_handles:
            ax.legend(loc="best", fontsize=7, frameon=False, ncol=1)
        if _flat_lines:
            # Caption inside the panel so reviewers immediately see why
            # one or more curves lie on top of the reference.
            ax.text(0.5, 0.97,
                    f"⚑ Flat (≤1 %): {', '.join(_flat_lines)} — "
                    "not identifiable",
                    transform=ax.transAxes, fontsize=7,
                    ha="center", va="top", color="#8B4513",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#FFF8E1",
                              edgecolor="#C0A060", lw=0.6))
        # Move the metrics table to a figtext strip below the figure
        # (two columns of compact label = value pairs).  We render labels
        # and "= value" as SEPARATE text objects with explicit alignment
        # so that the "=" column lands at a fixed horizontal position
        # regardless of glyph-width quirks (e.g. ² and ℓ are slightly
        # narrower than other characters even in monospace fonts).
        _half = (len(rows) + 1) // 2
        _col1 = rows[:_half]
        _col2 = rows[_half:]
        for _x_label, _x_eq, _col in ((0.18, 0.19, _col1),
                                       (0.63, 0.64, _col2)):
            fig.text(_x_label, 0.015,
                     "\n".join(k for k, _ in _col),
                     family="sans-serif", fontsize=8.5,
                     va="bottom", ha="right")
            fig.text(_x_eq, 0.015,
                     "\n".join(f"=  {v}" for _, v in _col),
                     family="sans-serif", fontsize=8.5,
                     va="bottom", ha="left")
    else:
        # ── SE: classic metrics-text panel ───────────────────────────────
        # Render labels and "= value" as two text objects so the "="
        # column stays aligned even for labels containing narrow glyphs
        # like ² or ℓ.
        ax.axis("off")
        ax.text(0.45, 1.0,
                "\n".join(k for k, _ in rows),
                family="sans-serif", fontsize=9,
                va="top", ha="right", transform=ax.transAxes)
        ax.text(0.47, 1.0,
                "\n".join(f"=  {v}" for _, v in rows),
                family="sans-serif", fontsize=9,
                va="top", ha="left", transform=ax.transAxes)
        ax.set_title("Fit metrics")

    # ── Noise-floor flag ─────────────────────────────────────────────────
    # When residuals are at solver precision (RMSE much smaller than the
    # natural scale of y_obs), DW and Shapiro statistics are measuring
    # numerical-integration artifacts rather than experimental noise.
    # Flag this regime so the user / reviewer does not over-interpret.
    # Only show the flag in the SE quadrant when the metrics table lives
    # there; with the profile shown, the metrics strip below already
    # captures the relevant statistics and an in-axes box would collide
    # with the profile plot.
    if not _has_profile:
        try:
            rmse = stats.get("rmse")
            y_range = float(np.nanmax(y_obs) - np.nanmin(y_obs)) if len(y_obs) else 0.0
            if (rmse is not None and np.isfinite(rmse) and rmse > 0
                    and y_range > 0 and rmse / y_range < 1e-4):
                ratio = rmse / y_range
                note = ("⚠ Residuals at solver precision\n"
                        f"   (RMSE / data range ≈ {ratio:.1e}).\n"
                        "   DW & Shapiro tests reflect\n"
                        "   numerical artifacts, not\n"
                        "   experimental noise.")
                ax.text(0.0, 0.05, note, family="monospace", fontsize=8,
                        va="bottom", ha="left", transform=ax.transAxes,
                        color="#C00000",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  edgecolor="#C00000",
                                  facecolor="#FFF6F6",
                                  linewidth=0.8))
        except Exception:
            pass

    # ── Spectra-mode independence-violation note ─────────────────────────
    # See above for the DW/Shapiro rationale.  Same suppression logic
    # applies when the profile is shown.
    if not _has_profile:
        try:
            fit_mode = (stats.get("fit_mode") or "").lower()
            if fit_mode in ("spectra", "kinetics_spectra"):
                note_sp = ("Note: Spectra-mode residuals are\n"
                           "      correlated across wavelength by\n"
                           "      spectrometer physics.  DW and\n"
                           "      Shapiro assume independent samples\n"
                           "      and are not meaningful here.\n"
                           "      Inspect residuals vs λ per spectrum\n"
                           "      for model quality instead.")
                ax.text(0.0, 0.05, note_sp, family="monospace", fontsize=8,
                        va="bottom", ha="left", transform=ax.transAxes,
                        color="#1F4E79",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  edgecolor="#1F4E79",
                                  facecolor="#F0F6FB",
                                  linewidth=0.8))
        except Exception:
            pass

    return fig


def render_diagnostics_panel(stats: dict, residuals_dict: dict, *,
                             key: str = "diag",
                             rmse_profile: Optional[dict] = None) -> None:
    """Streamlit renderer: figure + interpretive captions + download button.

    Imports Streamlit and matplotlib lazily so the module is importable in
    headless contexts.  Quietly does nothing if Streamlit isn't installed.

    Pass ``rmse_profile`` (the dict returned by ``compute_rmse_profile``)
    to render the parameter-profile plot in the SE quadrant instead of
    the fit-metrics table.  When ``None`` (default), the original layout
    is used.
    """
    try:
        import streamlit as st
    except Exception:
        return

    fig = make_diagnostics_figure(stats, residuals_dict,
                                  title="Residual diagnostics",
                                  rmse_profile=rmse_profile)
    st.pyplot(fig, clear_figure=False)

    # Noise-floor flag — shown in-app as a prominent warning so users do not
    # over-interpret DW / Shapiro statistics when residuals are at solver
    # precision rather than experimental noise.
    try:
        y_obs = np.asarray(residuals_dict.get("y_obs", []), dtype=float)
        rmse  = stats.get("rmse")
        y_range = (float(np.nanmax(y_obs) - np.nanmin(y_obs))
                   if len(y_obs) else 0.0)
        if (rmse is not None and np.isfinite(rmse) and rmse > 0
                and y_range > 0 and rmse / y_range < 1e-4):
            ratio = rmse / y_range
            st.warning(
                f"**Residuals at solver precision** (RMSE / data range "
                f"≈ {ratio:.1e}).  The input data have essentially no "
                f"experimental noise — for example, idealised simulated "
                f"data — so the residuals reflect the finite precision of "
                f"the equilibrium / kinetic solver rather than measurement "
                f"error.  The Durbin–Watson and Shapiro–Wilk statistics "
                f"are not meaningful in this regime; interpret R², SSR and "
                f"the information criteria with that context in mind."
            )
    except Exception:
        pass

    # Spectra-mode independence-violation note — shown in-app so users do
    # not panic at DW ≈ 0 and Shapiro p ≈ 0 for UV-Vis spectra fits, where
    # the residual vector is naturally correlated by wavelength.
    try:
        fit_mode = (stats.get("fit_mode") or "").lower()
        if fit_mode in ("spectra", "kinetics_spectra"):
            st.info(
                "**Spectra-mode note.**  The residual vector concatenates "
                "many spectra × many wavelengths, and adjacent entries are "
                "the *same* spectrum at *adjacent* wavelengths — correlated "
                "by spectrometer bandpass, baseline drift, and spectral "
                "smoothness.  Durbin–Watson and Shapiro–Wilk both assume "
                "*independent* samples and are therefore not meaningful "
                "here (DW will be near zero, Shapiro p tiny).  The "
                "diagnostically informative view is the *shape* of "
                "residuals vs wavelength per spectrum, which surfaces "
                "regions where the pure-species ε(λ) is slightly "
                "imperfect."
            )
    except Exception:
        pass

    # Interpretive captions
    dw  = stats.get("durbin_watson")
    sp  = stats.get("shapiro_p")
    aic = stats.get("aic")
    _fit_mode_low = (stats.get("fit_mode") or "").lower()
    _skip_dw_sp = _fit_mode_low in ("spectra", "kinetics_spectra")
    notes = []
    if not _skip_dw_sp and dw is not None and np.isfinite(dw):
        if 1.5 <= dw <= 2.5:
            notes.append(f"Durbin–Watson = {dw:.2f} — no evidence of "
                         "autocorrelation among ordered residuals.")
        elif dw < 1.5:
            notes.append(f"Durbin–Watson = {dw:.2f} — suggests positive "
                         "autocorrelation; the model may be missing "
                         "structure in the data ordering.")
        else:
            notes.append(f"Durbin–Watson = {dw:.2f} — suggests negative "
                         "autocorrelation, often an oversampling or "
                         "differencing artefact.")
    if not _skip_dw_sp and sp is not None and np.isfinite(sp):
        if sp >= 0.05:
            notes.append(f"Shapiro–Wilk p = {sp:.3f} — residuals are "
                         "consistent with a normal distribution.")
        else:
            notes.append(f"Shapiro–Wilk p = {sp:.3f} — residuals are "
                         "*not* consistent with a normal distribution; "
                         "Hessian-based standard errors may be optimistic. "
                         "Consider the bootstrap CI panel for a more "
                         "robust uncertainty estimate.")
    if aic is not None and np.isfinite(aic):
        notes.append("AIC/AICc/BIC are interpretable only in *differences* "
                     "between models fitted to the same data.  Use the "
                     "Model Comparison tool to evaluate competing "
                     "stoichiometries or mechanisms.")
    for n in notes:
        st.caption("• " + n)

    # PNG download
    try:
        import io
        from datetime import datetime as _dt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download diagnostics PNG",
                           data=buf.getvalue(),
                           file_name=f"Equilibrist_diagnostics_{_ts}.png",
                           mime="image/png",
                           key=f"{key}_dl_png")
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
# RMSE parameter profile (Musketeer §4.4 style)
# ════════════════════════════════════════════════════════════════════════

def _refit_with_pinned(fa: dict, *, pinned_key: str, pinned_val: float):
    """See :func:`_refit_with_pinned_impl`.  ``fa["simplex_step"]`` (log
    units), when present, sets the optimiser's initial step for the
    refit -- the profiles use a small one because they start next to the
    optimum."""
    from equilibrist_parser import simplex_step_override
    with simplex_step_override(fa.get("simplex_step")):
        return _refit_with_pinned_impl(fa, pinned_key=pinned_key, pinned_val=pinned_val)


def _refit_with_pinned_impl(fa: dict, *, pinned_key: str, pinned_val: float):
    """Run the appropriate fit function with one parameter pinned.

    Reads the same ``_fit_args_*`` dict the bootstrap UI uses, so the
    profile sees exactly the same fit pipeline that produced the headline
    result.  The pinned parameter is removed from ``fit_keys`` (or
    ``fit_conc_keys`` if it is a fitted concentration) and substituted
    with the requested value, so the underlying fit's pin-then-refit
    semantics produce a proper *conditional* profile — every other
    fitted parameter (including the un-pinned K's and concentrations)
    is free to compensate.

    ``pinned_val`` is always a log10 value.  For K's it is the log10 K
    (the fit's native parameterization).  For concentrations and
    titrants it is log10(mM), and ``_refit_with_pinned`` converts back
    to linear when writing into ``params["conc0"]`` / ``params["titrant_mMs"]``.

    Returns the resulting ``stats`` dict, or ``None`` on failure.
    """
    if fa.get("kind") == "global":              # one fit across several tabs
        from equilibrist_global import refit_global
        return refit_global(fa, {pinned_key: float(pinned_val)})
    try:
        # Identify which kind of parameter is being pinned by looking at
        # the fit_args lists.  K's live in fit_keys; concentrations live
        # in fit_conc_keys; titrants live in fit_titrant_keys.
        _is_K     = pinned_key in (fa.get("fit_keys") or [])
        _is_conc  = pinned_key in (fa.get("fit_conc_keys") or [])
        _is_tit   = pinned_key in (fa.get("fit_titrant_keys") or [])
        # Location parameters (currently ITC's q_offset) are pinned on a
        # LINEAR chart: they are additive, may be negative, and log10 is
        # undefined for them. The profile is invariant to how the pinned
        # axis is labelled, so mixing charts across parameters is sound
        # provided each axis is reported in its own units.
        _is_lin   = pinned_key in (fa.get("linear_pin_keys") or [])
        _lin_val  = float(pinned_val) if _is_lin else None

        _fk  = list(fa.get("fit_keys", []) or [])
        _fck = list(fa.get("fit_conc_keys", []) or [])
        _ftk = list(fa.get("fit_titrant_keys", []) or [])
        if _is_lin:
            pass                      # nothing removed: not a free key
        elif _is_K:
            _fk = [k for k in _fk if k != pinned_key]
        elif _is_conc:
            _fck = [k for k in _fck if k != pinned_key]
        elif _is_tit:
            _ftk = [k for k in _ftk if k != pinned_key]
        elif not _is_lin:
            # Not a recognized parameter — try treating it as a K for
            # backward compatibility with older callers.
            _fk = [k for k in _fk if k != pinned_key]
            _is_K = True

        # K's pinned via the start_log{k,K} dict (read by the fit
        # function's logK key-filter machinery).
        _start = dict(fa.get("start_logk", fa.get("start_logK", {})))
        if _is_K and not _is_lin:
            _start[pinned_key] = float(pinned_val)

        # Concentrations and titrants are linear-space mM values that
        # live in the params dict.  Copy the dict so we don't mutate the
        # caller's params; convert log10 → linear when writing.
        _params = fa.get("params")
        if (_is_conc or _is_tit) and _params is not None:
            _params = dict(_params)
            if _is_conc:
                _conc0 = dict(_params.get("conc0", {}))
                _conc0[pinned_key] = float(10.0 ** float(pinned_val))
                _params["conc0"] = _conc0
            elif _is_tit:
                _tmMs = dict(_params.get("titrant_mMs", {}))
                _tmMs[pinned_key] = float(10.0 ** float(pinned_val))
                _params["titrant_mMs"] = _tmMs

        if not _fk and not _fck and not _ftk:
            # Only one parameter exists in the whole fit and the user
            # asked to profile it.  There is nothing left to "refit"
            # while it is pinned, so a strict refit isn't possible —
            # but we can still evaluate RMSE at the pinned grid point,
            # which is all the profile actually needs.  Reuse the same
            # maxiter=1 + tiny-simplex pattern that _compute_rmse_at
            # employs for the local sensitivity test: put the pinned
            # parameter back into its keys list, set maxiter=1, and the
            # fit function (with its `1e-9 if maxiter <= 1 else 1.5`
            # simplex-step rule) returns RMSE at exactly the pinned
            # value without iterating.
            if _is_K:
                _fk = [pinned_key]
            elif _is_conc:
                _fck = [pinned_key]
            elif _is_tit:
                _ftk = [pinned_key]
            else:
                _fk = [pinned_key]
            _force_oneshot = True
        else:
            _force_oneshot = False

        _kind = fa.get("kind", "eq")  # "eq" or "kin"
        _tol  = float(fa.get("tolerance", 1e-6))
        _mxi  = int(fa.get("maxiter", 100000))
        if _force_oneshot:
            # Match the _refit_with_two_pinned one-shot path: huge tol +
            # maxiter=1 so the optimizer accepts the starting (= pinned)
            # point immediately and the returned RMSE is exactly the
            # value at the pinned configuration.
            _tol = 1e30
            _mxi = 1
        _cstr = fa.get("constraints", []) or []
        # NB: _fck and _params are already set above with the pinned
        # parameter removed / overridden — do NOT reread from fa here.

        # Equilibrium branch
        if _kind == "eq":
            _p   = fa["parsed_fit"]
            _net = fa["network"]
            _par = _params if _params is not None else fa["params"]
            _xe  = fa["x_expr"]
            if fa.get("use_itc_fit"):
                # ITC. q_offset is a linear nuisance parameter eliminated by
                # lstsq at every objective evaluation and is routinely
                # negative, so it is not profiled by this log10-based
                # machinery; its trade-off with dH is reported by the
                # correlation matrix instead. Pass it through if fixed.
                from equilibrist_itc import fit_itc as _f
                _ok, _, _s, _ = _f(_p, _net, fa["itc_q_exp"], fa["itc_raw"],
                                    _par, _start, _fk, _xe,
                                    tolerance=_tol, maxiter=_mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    fit_conc_keys=_fck,
                                    q_offset_fixed=(_lin_val if _is_lin
                                                    else fa.get("q_offset_fixed")),
                                    constraints=_cstr,
                                    excl_set=fa.get("excl_set") or set())
                if isinstance(_s, dict):
                    _r = _s.get("rmse")
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_spectra_fit"):
                from equilibrist_fit_spectra import fit_spectra as _f
                _ok, _, _s, _ = _f(_p, _net, fa["spectra_data_fit"],
                                    _par, _start, _fk, _xe,
                                    fa.get("wl_min"), fa.get("wl_max"),
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    fit_titrant_keys=_ftk,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if _mode == "shift":
                    from equilibrist_fit_nmr import fit_nmr_shifts as _f
                elif _mode == "integration":
                    from equilibrist_fit_nmr import fit_nmr_integration as _f
                elif _mode == "mixed":
                    from equilibrist_fit_nmr import fit_nmr_mixed as _f
                else:
                    return None
                # IMPORTANT: pass fit_conc_keys and fit_titrant_keys
                # through.  Without them, a profile run that PINS one
                # K and leaves a fitted concentration to compensate
                # would call this with fit_keys=[K_remaining] and no
                # conc keys — the fit's "free concentrations" set is
                # empty, the concentration stays pinned at its
                # original value, and the refit collapses to "nothing
                # to do" (returning the original RMSE).  Worse, when
                # the user is fitting ONLY a concentration plus one
                # K, the K-profile branch produces an empty fit_keys
                # list AND an empty conc_keys list, the fit function
                # returns "nothing to fit", and the K curve in the
                # profile plot comes back all-NaN and invisible.
                _ok, _, _s, _ = _f(_p, _net, fa["nmr_data_fit"],
                                    _par, _start, _fk, _xe,
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    fit_titrant_keys=_ftk)
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            from equilibrist_fit_conc import fit_parameters as _f
            _ok, _, _s, _ = _f(_p, _net, fa.get("exp_data_fit") or {},
                                _par, _start, _fk, _xe,
                                tolerance=_tol, maxiter=_mxi,
                                constraints=_cstr,
                                fit_conc_keys=_fck,
                                fit_titrant_keys=_ftk)
            # Accept the result whenever stats carry a finite rmse — the
            # optimizer's formal convergence flag is irrelevant for the
            # profile (force-oneshot mode uses maxiter=1 which never
            # declares convergence even though the resulting rmse is
            # exactly what we want at the pinned point).
            if isinstance(_s, dict):
                _r = _s.get('rmse')
                if _r is not None and np.isfinite(_r):
                    return _s
            return None

        # Kinetics branch
        if _kind == "kin":
            _p = fa["parsed_fit"]
            _t_max = float(fa["t_max"])
            # Kinetics fit functions use positional args (parsed, start_logk,
            # data, fit_keys, t_max, n_pts_sim, tolerance, maxiter, ...) —
            # see how app.py invokes them at the main fit call sites.
            # The earlier dispatcher mistakenly passed x_expr as the 5th
            # positional argument (which is reserved for t_max in these
            # functions) AND used the kwarg name n_pts (actual: n_pts_sim),
            # which caused all kinetics+NMR profile refits to raise
            # TypeError and the whole profile plot to come back empty.
            if fa.get("use_spectra_fit"):
                from equilibrist_kinetics_spectra import fit_kinetics_spectra as _f
                _ok, _, _s, _ = _f(_p, _start, fa["spectra_data_fit"], _fk,
                                    _t_max, 200,
                                    float(fa.get("wl_min") or 0.0),
                                    float(fa.get("wl_max") or 1e9),
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if _mode == "shift":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_shifts as _f
                elif _mode == "integration":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_integration as _f
                elif _mode == "mixed":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_mixed as _f
                else:
                    return None
                _ok, _, _s, _ = _f(_p, _start, fa["nmr_data_fit"], _fk,
                                    _t_max, 200, _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck)
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            from equilibrist_kinetics import fit_kinetics as _f
            _ok, _, _s, _ = _f(_p, fa.get("exp_data_fit") or {},
                                _start, _fk, _t_max, 200, _tol, _mxi,
                                timeout_s=float(fa.get("timeout_s", 30.0)),
                                constraints=_cstr,
                                fit_conc_keys=_fck)
            # Accept the result whenever stats carry a finite rmse — the
            # optimizer's formal convergence flag is irrelevant for the
            # profile (force-oneshot mode uses maxiter=1 which never
            # declares convergence even though the resulting rmse is
            # exactly what we want at the pinned point).
            if isinstance(_s, dict):
                _r = _s.get('rmse')
                if _r is not None and np.isfinite(_r):
                    return _s
            return None
    except Exception:
        return None
    return None


def compute_rmse_profile(fit_args: dict, fitted_values: dict,
                         *, span: float = 1.0, n_pts: int = 11,
                         progress_callback=None,
                         fitted_concs_mM: Optional[dict] = None,
                         fitted_titrants_mM: Optional[dict] = None,
                         n_jobs: int = 1,
                         conc_span_mult: tuple = (0.7, 1.3),
                         linear_params: Optional[dict] = None,
                         lin_span_frac: float = 1.0) -> dict:
    """Profile RMSE around each fitted parameter's optimum.

    For every key in ``fitted_values`` (mapping ``name`` → ``log10 K``),
    sweep that parameter from ``opt − span`` to ``opt + span`` in
    ``n_pts`` log-spaced steps, refit the other parameters at each step
    via ``_refit_with_pinned`` and record the resulting RMSE.

    To also profile fitted concentration parameters, pass them as
    ``fitted_concs_mM`` (mapping ``species_root`` → ``mM_value``); the
    function converts to log10(mM) internally so the same Δ log axis
    applies.  Concentrations remain free during K-profile refits and
    vice versa, giving a proper conditional / profile-likelihood
    interpretation: pin one parameter, marginalize over the rest.

    Returns ``{name: (values_array, rmse_array)}``.  Values where the
    pinned refit fails are recorded as ``np.nan`` so the plot can show
    gaps without crashing.

    ``progress_callback(frac, msg)`` is invoked between refits when
    provided.  ``span`` is in log10 units; ``n_pts`` should be odd so
    the optimum sits at the centre of the scan.

    ``n_jobs`` mirrors the bootstrap convention: ``n_jobs=1`` runs
    refits serially (default, safest on Streamlit Cloud).  ``n_jobs=-1``
    uses all cores via ``joblib.Parallel`` with process workers; any
    other positive integer caps the worker count.  Falls back to serial
    if joblib is unavailable.
    """
    if n_pts < 3:
        n_pts = 3
    if n_pts % 2 == 0:
        n_pts += 1  # ensure the optimum lies exactly on the grid

    # Merge K's, concs, and titrants into a single scan dict.  All values
    # stored as log10 so the Δ-log axis is consistent.  Conc/titrant
    # names get a distinguishing prefix to avoid collisions with K
    # names — and the dispatcher uses fit_conc_keys / fit_titrant_keys
    # to recognize them.
    scan = dict(fitted_values or {})
    if fitted_concs_mM:
        for _name, _mM in fitted_concs_mM.items():
            try:
                _v = float(_mM)
                if _v > 0:
                    scan[_name] = float(np.log10(_v))
            except Exception:
                continue
    if fitted_titrants_mM:
        for _name, _mM in fitted_titrants_mM.items():
            try:
                _v = float(_mM)
                if _v > 0:
                    scan[_name] = float(np.log10(_v))
            except Exception:
                continue

    # Flatten the (parameter × grid-point) work list so workers can be
    # dispatched in parallel.  Each task is a (param_name, pinned_value)
    # tuple; the result is the RMSE at that point (or NaN on failure).
    tasks = []         # list of (param_name, value)
    grid_per_name = {} # param_name -> grid array (preserves order)

    # Concentrations get a MULTIPLICATIVE window (default 0.7x - 1.3x of the
    # fitted value) instead of the K's +/- span decades. A full decade either
    # side drives the pinned refit into the degenerate "no binding" basin
    # (K -> 0, q_offset absorbs everything) where RMSE saturates at a constant
    # identical for every concentration, which is uninformative.
    _conc_names = set(fitted_concs_mM or {}) | set(fitted_titrants_mM or {})
    _lo_m, _hi_m = float(conc_span_mult[0]), float(conc_span_mult[1])
    _half = max(1, (n_pts - 1) // 2)

    for _name, _opt in scan.items():
        try:
            _opt_f = float(_opt)
        except Exception:
            continue
        if _name in _conc_names and _lo_m > 0 and _hi_m > 0:
            # Two half-grids sharing the centre, so the optimum stays exactly
            # on the grid even though the window is asymmetric in log space.
            _lo = _opt_f + float(np.log10(_lo_m))
            _hi = _opt_f + float(np.log10(_hi_m))
            _vals = np.concatenate([
                np.linspace(_lo, _opt_f, _half + 1)[:-1],
                np.linspace(_opt_f, _hi, _half + 1)])
        else:
            _vals = np.linspace(_opt_f - span, _opt_f + span, n_pts)
        grid_per_name[_name] = _vals
        for _v in _vals:
            tasks.append((_name, float(_v)))
    # Location parameters (ITC q_offset) profile on a LINEAR chart in their
    # own units. They may be negative, so log10 is undefined; and the profile
    # is invariant to how the pinned axis is labelled, so this is exact, not
    # an approximation. The dispatcher recognises them via
    # fit_args["linear_pin_keys"] and routes them to q_offset_fixed.
    _lin = dict(linear_params or {})
    if _lin:
        _lin_keys = set(fit_args.get("linear_pin_keys") or set())
        _lin_keys |= set(_lin)
        fit_args["linear_pin_keys"] = _lin_keys
        for _name, _val in _lin.items():
            try:
                _v = float(_val)
            except Exception:
                continue
            _w = max(abs(_v) * float(lin_span_frac), 1e-9)
            _vals = np.linspace(_v - _w, _v + _w, n_pts)
            grid_per_name[_name] = _vals
            for _g in _vals:
                tasks.append((_name, float(_g)))

    total = max(1, len(tasks))

    def _one(name_val):
        _name, _v = name_val
        _s = _refit_with_pinned(fit_args, pinned_key=_name, pinned_val=_v)
        if _s is None:
            return float("nan")
        _r = _s.get("rmse")
        return float(_r) if _r is not None and np.isfinite(_r) else float("nan")

    # Serial path (also the fallback if joblib is missing)
    if n_jobs == 1:
        results = [None] * len(tasks)
        for _i, _t in enumerate(tasks):
            if progress_callback is not None:
                try:
                    progress_callback(_i / total,
                                      f"Profiling {_t[0]}: "
                                      f"{_i + 1}/{total}")
                except Exception:
                    pass
            results[_i] = _one(_t)
    else:
        # Parallel path — mirror the bootstrap convention.  joblib is
        # process-based ("prefer=processes") to bypass the GIL; each
        # worker re-imports the fit modules and re-runs _refit_with_pinned.
        # The fit_args dict is picklable (plain dicts/lists/arrays), so
        # this round-trips cleanly.
        try:
            from joblib import Parallel, delayed
        except Exception:
            # joblib unavailable — fall back to serial without losing
            # progress reporting.
            return compute_rmse_profile(
                fit_args, fitted_values, span=span, n_pts=n_pts,
                progress_callback=progress_callback,
                fitted_concs_mM=fitted_concs_mM,
                fitted_titrants_mM=fitted_titrants_mM,
                n_jobs=1,
            )
        try:
            gen = Parallel(n_jobs=n_jobs, prefer="processes",
                           return_as="generator")(
                delayed(_one)(_t) for _t in tasks
            )
            results = []
            for _r in gen:
                results.append(_r)
                if progress_callback is not None:
                    try:
                        progress_callback(len(results) / total,
                                          f"Profiling: "
                                          f"{len(results)}/{total}")
                    except Exception:
                        pass
        except TypeError:
            # Old joblib without return_as — silent parallel batch
            results = Parallel(n_jobs=n_jobs, prefer="processes",
                               batch_size="auto")(
                delayed(_one)(_t) for _t in tasks
            )

    # Re-group flat results back into the per-parameter (vals, rmses) tuples
    profile = {}
    idx = 0
    for _name, _vals in grid_per_name.items():
        _rmses = np.asarray(results[idx:idx + n_pts], dtype=float)
        profile[_name] = (_vals, _rmses)
        idx += n_pts

    if progress_callback is not None:
        try:
            progress_callback(1.0, "Profile complete.")
        except Exception:
            pass
    return profile


# ════════════════════════════════════════════════════════════════════════
# 2D RMSE parameter profile (Musketeer §4.4 pairwise diagnostic)
# ════════════════════════════════════════════════════════════════════════
#
# The 1D profile above answers "is parameter X individually identifiable?"
# The 2D profile answers "are X and Y identifiable jointly?" — long
# diagonal valleys in the colormap reveal strong pairwise correlations
# that 1D profiles only hint at indirectly (e.g. the tutorial-5 K2 vs G
# anti-correlation when K1 is fixed at a wrong value).
#
# Math: same conditional / profile-likelihood semantics as 1D — at each
# (X_val, Y_val) grid point, both parameters are pinned and EVERY OTHER
# free parameter is refit.  So a 2D scan over (X, Y) when 4 parameters
# are free still costs a full nonlinear fit over the remaining 2 at
# each grid point.

def _refit_with_two_pinned(fa: dict, *,
                           pin1_key: str, pin1_val: float,
                           pin2_key: str, pin2_val: float):
    """See :func:`_refit_with_two_pinned_impl`; honours ``fa["simplex_step"]``."""
    from equilibrist_parser import simplex_step_override
    with simplex_step_override(fa.get("simplex_step")):
        return _refit_with_two_pinned_impl(fa, pin1_key=pin1_key, pin1_val=pin1_val,
                                           pin2_key=pin2_key, pin2_val=pin2_val)


def _refit_with_two_pinned_impl(fa: dict, *,
                           pin1_key: str, pin1_val: float,
                           pin2_key: str, pin2_val: float):
    """Run the appropriate fit function with TWO parameters pinned.

    Mechanically identical to ``_refit_with_pinned`` but handles both
    pins simultaneously: each pin is independently classified as a K
    (start_logK), concentration (params["conc0"] in linear mM), or
    titrant (params["titrant_mMs"]), and the matching free-parameter
    list has its entry removed.  Any other fitted parameter remains
    free to compensate during the refit — that's what makes the
    resulting RMSE surface a *conditional* profile.

    Pin values are always in log10 units (K's natively, concs and
    titrants as log10 mM converted to linear when written into
    ``params``).  Returns the resulting ``stats`` dict or ``None`` on
    dispatch failure.
    """
    if fa.get("kind") == "global":
        from equilibrist_global import refit_global
        return refit_global(fa, {pin1_key: float(pin1_val), pin2_key: float(pin2_val)})
    # Linear-chart pinning is handled only by _refit_with_pinned;
    # define the flags here so the shared ITC branch is safe.
    _is_lin, _lin_val = False, None
    try:
        _all_K = list(fa.get("fit_keys", []) or [])
        _all_C = list(fa.get("fit_conc_keys", []) or [])
        _all_T = list(fa.get("fit_titrant_keys", []) or [])

        def _classify(name):
            if name in _all_K: return "K"
            if name in _all_C: return "C"
            if name in _all_T: return "T"
            return "K"  # fallback — treat as K for backward compat

        _t1 = _classify(pin1_key)
        _t2 = _classify(pin2_key)

        # Drop both pinned names from every list (defensive: a name
        # should only appear in one list, but we strip from all three
        # in case a user names a K the same as a concentration root).
        _drop = {pin1_key, pin2_key}
        _fk  = [k for k in _all_K if k not in _drop]
        _fck = [k for k in _all_C if k not in _drop]
        _ftk = [k for k in _all_T if k not in _drop]

        # K-pins → start_logK; conc/titrant-pins → params dict
        _start = dict(fa.get("start_logk", fa.get("start_logK", {})))
        for _name, _val, _t in ((pin1_key, pin1_val, _t1),
                                 (pin2_key, pin2_val, _t2)):
            if _t == "K":
                _start[_name] = float(_val)

        _params = fa.get("params")
        if any(_t in ("C", "T") for _t in (_t1, _t2)) and _params is not None:
            _params = dict(_params)
            _params["conc0"]       = dict(_params.get("conc0", {}))
            _params["titrant_mMs"] = dict(_params.get("titrant_mMs", {}))
            for _name, _val, _t in ((pin1_key, pin1_val, _t1),
                                     (pin2_key, pin2_val, _t2)):
                if _t == "C":
                    _params["conc0"][_name] = float(10.0 ** float(_val))
                elif _t == "T":
                    _params["titrant_mMs"][_name] = float(10.0 ** float(_val))

        # ── Handle the "all pinned, nothing left to refit" case ──────
        # When both pinned parameters were the ONLY free parameters
        # (e.g. tutorial 5 fitting just K1 + K2), the lists above
        # become empty.  scipy.optimize.minimize with Nelder-Mead
        # crashes on dim=0 (ZeroDivisionError on ``1 + 2/dim``), so we
        # can't just call the fit function with empty lists.
        #
        # Workaround: leave ONE of the pinned parameters in fit_keys
        # (or fit_conc_keys / fit_titrant_keys) so the optimizer has
        # a dimension to work with, then set maxiter=1 and a huge
        # tolerance so the optimizer immediately accepts the starting
        # point (which is the pinned value) as converged.  The
        # function then reports the RMSE at the pinned configuration
        # — exactly what we want for the 2D profile.  This adds at
        # most one evaluation of the objective function per cell
        # versus a true direct evaluation, which is negligible.
        _force_oneshot = (not _fk and not _fck and not _ftk)
        if _force_oneshot:
            # Pick whichever pinned param can be put back in its
            # appropriate list.  If pin1 is a K, restore it to _fk;
            # similarly for C / T.  start_logK / params still pin
            # both to the requested values, so the optimizer starts
            # at the pinned point in all dimensions.
            if _t1 == "K":
                _fk = [pin1_key]
            elif _t1 == "C":
                _fck = [pin1_key]
            elif _t1 == "T":
                _ftk = [pin1_key]
            else:
                _fk = [pin1_key]  # fallback

        _kind = fa.get("kind", "eq")
        _tol  = float(fa.get("tolerance", 1e-6))
        _mxi  = int(fa.get("maxiter", 100000))
        # Override tolerance / maxiter in the one-shot path so the
        # optimizer doesn't actually move away from the pinned values.
        if _force_oneshot:
            _tol = 1e30
            _mxi = 1
        _cstr = fa.get("constraints", []) or []

        if _kind == "eq":
            _p   = fa["parsed_fit"]
            _net = fa["network"]
            _par = _params if _params is not None else fa["params"]
            _xe  = fa["x_expr"]
            if fa.get("use_itc_fit"):
                # ITC. q_offset is a linear nuisance parameter eliminated by
                # lstsq at every objective evaluation and is routinely
                # negative, so it is not profiled by this log10-based
                # machinery; its trade-off with dH is reported by the
                # correlation matrix instead. Pass it through if fixed.
                from equilibrist_itc import fit_itc as _f
                _ok, _, _s, _ = _f(_p, _net, fa["itc_q_exp"], fa["itc_raw"],
                                    _par, _start, _fk, _xe,
                                    tolerance=_tol, maxiter=_mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    fit_conc_keys=_fck,
                                    q_offset_fixed=(_lin_val if _is_lin
                                                    else fa.get("q_offset_fixed")),
                                    constraints=_cstr,
                                    excl_set=fa.get("excl_set") or set())
                if isinstance(_s, dict):
                    _r = _s.get("rmse")
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_spectra_fit"):
                from equilibrist_fit_spectra import fit_spectra as _f
                _ok, _, _s, _ = _f(_p, _net, fa["spectra_data_fit"],
                                    _par, _start, _fk, _xe,
                                    fa.get("wl_min"), fa.get("wl_max"),
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    fit_titrant_keys=_ftk,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if _mode == "shift":
                    from equilibrist_fit_nmr import fit_nmr_shifts as _f
                elif _mode == "integration":
                    from equilibrist_fit_nmr import fit_nmr_integration as _f
                elif _mode == "mixed":
                    from equilibrist_fit_nmr import fit_nmr_mixed as _f
                else:
                    return None
                _ok, _, _s, _ = _f(_p, _net, fa["nmr_data_fit"],
                                    _par, _start, _fk, _xe,
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    fit_titrant_keys=_ftk)
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            from equilibrist_fit_conc import fit_parameters as _f
            _ok, _, _s, _ = _f(_p, _net, fa.get("exp_data_fit") or {},
                                _par, _start, _fk, _xe,
                                tolerance=_tol, maxiter=_mxi,
                                constraints=_cstr,
                                fit_conc_keys=_fck,
                                fit_titrant_keys=_ftk)
            # Accept the result whenever stats carry a finite rmse — the
            # optimizer's formal convergence flag is irrelevant for the
            # profile (force-oneshot mode uses maxiter=1 which never
            # declares convergence even though the resulting rmse is
            # exactly what we want at the pinned point).
            if isinstance(_s, dict):
                _r = _s.get('rmse')
                if _r is not None and np.isfinite(_r):
                    return _s
            return None

        if _kind == "kin":
            _p     = fa["parsed_fit"]
            _t_max = float(fa["t_max"])
            if fa.get("use_spectra_fit"):
                from equilibrist_kinetics_spectra import fit_kinetics_spectra as _f
                _ok, _, _s, _ = _f(_p, _start, fa["spectra_data_fit"], _fk,
                                    _t_max, 200,
                                    float(fa.get("wl_min") or 0.0),
                                    float(fa.get("wl_max") or 1e9),
                                    _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            if fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if _mode == "shift":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_shifts as _f
                elif _mode == "integration":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_integration as _f
                elif _mode == "mixed":
                    from equilibrist_kinetics_nmr import fit_kinetics_nmr_mixed as _f
                else:
                    return None
                _ok, _, _s, _ = _f(_p, _start, fa["nmr_data_fit"], _fk,
                                    _t_max, 200, _tol, _mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    constraints=_cstr,
                                    fit_conc_keys=_fck)
                # Accept the result whenever stats carry a finite rmse — the
                # optimizer's formal convergence flag is irrelevant for the
                # profile (force-oneshot mode uses maxiter=1 which never
                # declares convergence even though the resulting rmse is
                # exactly what we want at the pinned point).
                if isinstance(_s, dict):
                    _r = _s.get('rmse')
                    if _r is not None and np.isfinite(_r):
                        return _s
                return None
            from equilibrist_kinetics import fit_kinetics as _f
            _ok, _, _s, _ = _f(_p, fa.get("exp_data_fit") or {},
                                _start, _fk, _t_max, 200, _tol, _mxi,
                                timeout_s=float(fa.get("timeout_s", 30.0)),
                                constraints=_cstr,
                                fit_conc_keys=_fck)
            # Accept the result whenever stats carry a finite rmse — the
            # optimizer's formal convergence flag is irrelevant for the
            # profile (force-oneshot mode uses maxiter=1 which never
            # declares convergence even though the resulting rmse is
            # exactly what we want at the pinned point).
            if isinstance(_s, dict):
                _r = _s.get('rmse')
                if _r is not None and np.isfinite(_r):
                    return _s
            return None
    except Exception:
        return None
    return None


def compute_rmse_profile_2d(fit_args: dict,
                            p1_name: str, p1_opt: float,
                            p2_name: str, p2_opt: float,
                            *, span_x: float = 0.5, span_y: float = 0.5,
                            n_pts_x: int = 11, n_pts_y: int = 11,
                            progress_callback=None,
                            n_jobs: int = 1) -> dict:
    """Compute a 2D RMSE profile over (p1, p2) with the rest refit.

    Sweeps ``p1`` from ``p1_opt ± span_x`` and ``p2`` from ``p2_opt ±
    span_y`` over an ``n_pts_x × n_pts_y`` grid (both arrays forced
    odd so the optima sit exactly on grid lines).  At each grid point
    both parameters are pinned and any other free parameter is refit.

    Returns ``{"p1_name", "p2_name", "p1_grid" (1D), "p2_grid" (1D),
    "rmse" (2D, shape ``(n_pts_y, n_pts_x)``), "rmse_min"}``.

    ``p1_opt`` and ``p2_opt`` are log10 — for K's, log10 K directly;
    for concentrations and titrants, log10(mM).  The function does NOT
    convert; the caller is responsible for passing log10 already (the
    1D ``compute_rmse_profile`` does the same).

    ``n_jobs`` mirrors the 1D convention: 1 = serial, -1 = all cores
    via joblib.Parallel.  Falls back to serial if joblib is missing.
    """
    # Force odd so the optimum sits on the grid
    if n_pts_x < 3: n_pts_x = 3
    if n_pts_y < 3: n_pts_y = 3
    if n_pts_x % 2 == 0: n_pts_x += 1
    if n_pts_y % 2 == 0: n_pts_y += 1

    p1_grid = np.linspace(p1_opt - span_x, p1_opt + span_x, n_pts_x)
    p2_grid = np.linspace(p2_opt - span_y, p2_opt + span_y, n_pts_y)

    # Build the full (j, i) -> (p2_val, p1_val) task list.  Row j ↔ p2,
    # column i ↔ p1, so the resulting matrix can be passed straight to
    # imshow / pcolormesh without transposing.
    tasks = []
    for j, _p2v in enumerate(p2_grid):
        for i, _p1v in enumerate(p1_grid):
            tasks.append((j, i, float(_p1v), float(_p2v)))
    total = max(1, len(tasks))

    # Collect a sample of exception messages from failed cells.  When
    # every cell returns NaN, the rendered figure currently just says
    # "all refits returned NaN" — useless for diagnosis.  Capturing the
    # FIRST few exceptions lets the UI surface what actually went
    # wrong (parallel-pickle error? ODE divergence? key error?).
    _error_samples: list = []

    def _one(task):
        _j, _i, _v1, _v2 = task
        try:
            _s = _refit_with_two_pinned(fit_args,
                                        pin1_key=p1_name, pin1_val=_v1,
                                        pin2_key=p2_name, pin2_val=_v2)
        except Exception as _e:
            return (_j, _i, float("nan"), f"{type(_e).__name__}: {_e}")
        if _s is None:
            return (_j, _i, float("nan"), "refit_returned_None")
        _r = _s.get("rmse")
        if _r is None:
            return (_j, _i, float("nan"), "stats_missing_rmse_key")
        if not np.isfinite(_r):
            return (_j, _i, float("nan"), f"rmse_non_finite ({_r!r})")
        return (_j, _i, float(_r), None)

    rmse_mat = np.full((n_pts_y, n_pts_x), np.nan, dtype=float)

    def _record_err(msg):
        # Keep at most ~5 distinct error messages; further duplicates
        # are dropped to keep the figure annotation short.
        if msg and len(_error_samples) < 5 and msg not in _error_samples:
            _error_samples.append(msg)

    if n_jobs == 1:
        for _idx, _t in enumerate(tasks):
            if progress_callback is not None:
                try:
                    progress_callback(_idx / total,
                                      f"2D profile: {_idx + 1}/{total}")
                except Exception:
                    pass
            _j, _i, _r, _err = _one(_t)
            rmse_mat[_j, _i] = _r
            _record_err(_err)
    else:
        try:
            from joblib import Parallel, delayed
        except Exception:
            return compute_rmse_profile_2d(
                fit_args, p1_name, p1_opt, p2_name, p2_opt,
                span_x=span_x, span_y=span_y,
                n_pts_x=n_pts_x, n_pts_y=n_pts_y,
                progress_callback=progress_callback, n_jobs=1)
        try:
            gen = Parallel(n_jobs=n_jobs, prefer="processes",
                           return_as="generator")(
                delayed(_one)(_t) for _t in tasks
            )
            _done = 0
            for _j, _i, _r, _err in gen:
                rmse_mat[_j, _i] = _r
                _record_err(_err)
                _done += 1
                if progress_callback is not None:
                    try:
                        progress_callback(_done / total,
                                          f"2D profile: {_done}/{total}")
                    except Exception:
                        pass
        except TypeError:
            for _j, _i, _r, _err in Parallel(n_jobs=n_jobs, prefer="processes",
                                        batch_size="auto")(
                    delayed(_one)(_t) for _t in tasks):
                rmse_mat[_j, _i] = _r
                _record_err(_err)

    if progress_callback is not None:
        try:
            progress_callback(1.0, "2D profile complete.")
        except Exception:
            pass

    _rmin = float(np.nanmin(rmse_mat)) if np.any(np.isfinite(rmse_mat)) else np.nan
    _n_fail = int(np.sum(~np.isfinite(rmse_mat)))
    return {
        "p1_name":  p1_name, "p2_name":  p2_name,
        "p1_opt":   float(p1_opt), "p2_opt": float(p2_opt),
        "p1_grid":  p1_grid, "p2_grid": p2_grid,
        "rmse":     rmse_mat,
        "rmse_min": _rmin,
        "n_failed": _n_fail,
        "n_total":  int(rmse_mat.size),
        "error_samples": list(_error_samples),
    }


def make_2d_profile_figure(scan_2d: dict, *,
                           title: Optional[str] = None,
                           param_cov: Optional[np.ndarray] = None,
                           cov_names: Optional[list] = None,
                           constraints: Optional[list] = None,
                           show_legend: bool = True):
    """Render a 2D RMSE-profile colormap.

    ``show_legend=False`` omits the legend box and the interpretive note in
    the top-left corner — for publication figures, where that text belongs
    in the caption.  Markers, ellipses and contours are drawn either way.

    Plots log10(RMSE / RMSE_min) over the (log p1, log p2) grid — axes in
    ABSOLUTE log10 units, e.g. 6 … 9 for a constant at 7.5 scanned ± 1.5 —
    with a perceptually-uniform colormap, a star at the optimum, and
    contour lines at RMSE-ratio = 1.1, 1.5, 2× (=  log10 ≈ 0.041,
    0.176, 0.301).  Long diagonal valleys (high color contrast running
    diagonally across the panel) reveal strong correlations between
    the two parameters; round valleys mean they're independently
    identified.

    If a full-fit covariance matrix is supplied via ``param_cov`` (with
    ``cov_names`` listing the row/column order), the function overlays
    the linearized Hessian confidence ellipses for the two parameters
    at 68 % and 95 % levels.  The ellipse is the quadratic
    approximation of the SSR surface at the optimum: when it tracks
    the nonlinear contours, the Hessian SE is trustworthy; when the
    contours bend into a banana while the ellipse stays straight, the
    linear approximation is failing and the reported SE understates
    the true uncertainty.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    p1_grid = np.asarray(scan_2d["p1_grid"], dtype=float)
    p2_grid = np.asarray(scan_2d["p2_grid"], dtype=float)
    rmse_mat = np.asarray(scan_2d["rmse"], dtype=float)
    _rmin = float(scan_2d.get("rmse_min", np.nan))
    _p1 = scan_2d["p1_name"]; _p2 = scan_2d["p2_name"]
    _p1_opt = float(scan_2d["p1_opt"]); _p2_opt = float(scan_2d["p2_opt"])

    # Plot Δ-from-optimum so both axes are centered at 0 — keeps the
    # visual convention identical to the 1D profile.
    # Axes carry ABSOLUTE log10 values (the scanned range itself), so the
    # map reads directly as "log K from 6 to 9" rather than as an offset.
    # The names dx / dy are kept for the grid coordinates used below.
    dx = p1_grid
    dy = p2_grid

    # Color scale: RMSE / RMSE_min on log axis (log_norm), so a 2× rise
    # spans the same color range whether the optimum was 1e-3 or 1e-1.
    if not (np.isfinite(_rmin) and _rmin > 0):
        # All NaN — render an informative diagnostic instead of the
        # generic "all refits returned NaN" so the user can act on it.
        _samples = scan_2d.get("error_samples", []) or []
        _n_fail  = scan_2d.get("n_failed", 0)
        _n_tot   = scan_2d.get("n_total",  0)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _lines = [f"2D profile failed: {_n_fail}/{_n_tot} cells "
                  "returned no finite RMSE."]
        if _samples:
            _lines.append("")
            _lines.append("First few failure reasons:")
            for _m in _samples[:5]:
                _lines.append(f"  • {_m}")
        else:
            _lines.append("(no error samples captured — check the "
                          "Streamlit console for tracebacks)")
        ax.text(0.04, 0.96, "\n".join(_lines),
                ha="left", va="top", transform=ax.transAxes,
                color="#A03030", fontsize=10, family="monospace")
        ax.set_axis_off()
        if title: fig.suptitle(title, fontsize=11)
        return fig

    ratio = rmse_mat / _rmin
    # Mask any non-finite entries so the colormap shows them as
    # background rather than as misleading 0s
    masked = np.ma.masked_invalid(ratio)
    _vmax = float(np.nanmax(ratio))
    _vmax = min(max(_vmax, 1.05), 100.0)  # cap for the LogNorm to be sensible

    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    # pcolormesh with cell edges centered on the grid points
    _dx_e = np.concatenate(([dx[0] - (dx[1]-dx[0])/2],
                            (dx[:-1]+dx[1:])/2,
                            [dx[-1] + (dx[-1]-dx[-2])/2]))
    _dy_e = np.concatenate(([dy[0] - (dy[1]-dy[0])/2],
                            (dy[:-1]+dy[1:])/2,
                            [dy[-1] + (dy[-1]-dy[-2])/2]))
    pcm = ax.pcolormesh(_dx_e, _dy_e, masked,
                        norm=LogNorm(vmin=1.0, vmax=_vmax),
                        cmap="viridis_r", shading="flat", zorder=1)

    # Contour overlay at meaningful RMSE-ratio levels
    try:
        _contours = [c for c in (1.1, 1.5, 2.0, 5.0)
                     if c <= _vmax * 0.98]
        if _contours:
            X, Y = np.meshgrid(dx, dy)
            cs = ax.contour(X, Y, masked, levels=_contours,
                            colors="white", linewidths=0.7,
                            linestyles="--", zorder=2)
            ax.clabel(cs, fmt=lambda v: f"{v:g}×", fontsize=7,
                      inline=True)
    except Exception:
        pass

    # ── Linearized (Hessian) confidence ellipses ──────────────────
    # Draw the local quadratic approximation of the SSR surface as
    # confidence ellipses derived from the 2×2 sub-covariance of the
    # two scanned parameters.  These are the "what the Hessian thinks
    # the joint distribution looks like" curves; comparing them to the
    # nonlinear white contours is the whole point — a Hessian SE is
    # trustworthy only when these ellipses lie on the same level sets
    # as the nonlinear contours.
    _ellipse_drawn = False
    if param_cov is not None and cov_names:
        try:
            cov_arr = np.asarray(param_cov, dtype=float)
            names = list(cov_names)
            if _p1 in names and _p2 in names and cov_arr.shape[0] == len(names):
                i1, i2 = names.index(_p1), names.index(_p2)
                sub = np.array([[cov_arr[i1, i1], cov_arr[i1, i2]],
                                [cov_arr[i2, i1], cov_arr[i2, i2]]],
                               dtype=float)
                # Eigendecompose: positive-semidefinite, eigh returns
                # ascending eigenvalues + orthonormal eigenvectors.
                if (np.all(np.isfinite(sub))
                        and sub[0, 0] > 0 and sub[1, 1] > 0):
                    eigvals, eigvecs = np.linalg.eigh(sub)
                    # Guard against tiny-negative eigenvalues from
                    # numerical roundoff
                    eigvals = np.clip(eigvals, 1e-30, None)
                    # Sort descending so eigvecs[:, 0] is the major axis
                    order   = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    # χ² thresholds for 2 d.o.f.:
                    #   68.3 % (1σ) → 2.30,  95 % → 5.99
                    _chi2_levels = [(2.30,  "1σ (68 %)"),
                                    (5.991, "95 %")]
                    _t = np.linspace(0.0, 2.0 * np.pi, 360)
                    for _chi2, _lbl in _chi2_levels:
                        _a = float(np.sqrt(eigvals[0] * _chi2))
                        _b = float(np.sqrt(eigvals[1] * _chi2))
                        # Parametric ellipse in principal frame, then
                        # rotate via eigvecs into the (Δp1, Δp2) frame.
                        _xy_p = np.stack([_a * np.cos(_t),
                                          _b * np.sin(_t)], axis=0)
                        _xy = eigvecs @ _xy_p
                        ax.plot(_xy[0] + _p1_opt, _xy[1] + _p2_opt,
                                color="#FF7A00", lw=1.4,
                                ls=("--" if _lbl.startswith("1σ") else "-"),
                                zorder=3, label=f"Hessian {_lbl}")
                    _ellipse_drawn = True
        except Exception:
            pass

    # Optimum marker (the fitted optimum, or — when one of the scanned
    # constants was held FIXED in the last fit — just the reference point
    # the map is centred on, which is generally not the minimum).
    _p1_fixed = bool(scan_2d.get("p1_fixed", False))
    _p2_fixed = bool(scan_2d.get("p2_fixed", False))
    _any_fixed = _p1_fixed or _p2_fixed
    ax.plot(_p1_opt, _p2_opt, marker="*", ms=14,
            mec="white", mew=1.0, mfc="#FFD700", zorder=4,
            label=("reference point (last fit)" if _any_fixed else None))
    # Mark the grid minimum when it is not the central cell — with a
    # fixed constant on an axis this is the whole point of the map.
    _min_drawn = False
    try:
        if np.any(np.isfinite(rmse_mat)):
            _jm, _im = np.unravel_index(np.nanargmin(rmse_mat), rmse_mat.shape)
            if (abs(dx[_im] - _p1_opt) > 1e-9) or (abs(dy[_jm] - _p2_opt) > 1e-9):
                ax.plot(dx[_im], dy[_jm], marker="X", ms=10,
                        mec="black", mew=0.8, mfc="white", zorder=5,
                        label=(f"grid minimum: {_p1} = {p1_grid[_im]:.2f}, "
                               f"{_p2} = {p2_grid[_jm]:.2f}"))
                _min_drawn = True
    except Exception:
        pass
    ax.axhline(_p2_opt, color="white", lw=0.4, ls=":", alpha=0.5, zorder=2)
    ax.axvline(_p1_opt, color="white", lw=0.4, ls=":", alpha=0.5, zorder=2)

    ax.set_xlabel(f"log {_p1}")
    ax.set_ylabel(f"log {_p2}")
    ax.set_title(title or f"2D RMSE profile: {_p1} vs {_p2}")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("RMSE / RMSE$_{min}$  (log scale)")

    # Lock axes to the scan grid extent — the Hessian ellipse can in
    # principle extend far past it (loose-direction parameters have
    # huge variance), in which case we want the ellipse clipped at
    # the plot edge rather than the colormap shrinking into one
    # corner.  The very fact that the ellipse runs off-axis is
    # diagnostic: it means the Hessian predicts the joint CI is
    # wider than the scan covered.
    ax.set_xlim(_dx_e[0], _dx_e[-1])
    ax.set_ylim(_dy_e[0], _dy_e[-1])

    # If Hessian ellipses were overlaid, attach a small legend so the
    # reader knows what the orange curves mean.
    if show_legend and (_ellipse_drawn or _min_drawn or _any_fixed):
        ax.legend(loc="lower right", fontsize=7,
                  facecolor="black", edgecolor="none",
                  labelcolor="white", framealpha=0.45)

    # Footnote: the grid axes are in the optimizer's native log10 space
    # for all parameter kinds (binding constants, rate constants, fitted
    # concentrations), not the linear display units used elsewhere.
    # This is technically accurate — "Δ log k₂ = 0.5" means "k₂ changes
    # by a factor of 10^0.5 ≈ 3.2" — but worth flagging because the
    # jackknife panel reports Δ for rate constants in *linear* units
    # (matching the main "Fitted constants" table) while the 2D scan
    # has to stay in log to be physically meaningful.
    # (Caption removed per Eric's review — the axis labels themselves
    # already say "Δ log K1", "Δ log K2" etc., so the footnote was
    # redundant.)
    # Interpretive subtitle: diagonal vs round?
    try:
        # First check whether a constraint links the two plotted
        # parameters.  If so, the Hessian-based eigenvalue heuristic
        # below is misleading — the constraint pins the relationship
        # between p1 and p2, so apparent "round well" or "elongated
        # valley" reads from the eigenvalues reflect the constraint
        # geometry rather than data identifiability.  Override the
        # annotation in that case.
        _p1_name = scan_2d.get("p1_name", "")
        _p2_name = scan_2d.get("p2_name", "")
        _constraint_active = False
        if constraints and _p1_name and _p2_name:
            for _c in constraints:
                # Constraints from the parser are dicts with either
                # `param_names` (list) or `coeffs` (dict) listing the
                # variables they touch.  A constraint "links" the two
                # plotted parameters iff BOTH names appear in its
                # variable list.
                _names = set()
                _pn = _c.get("param_names")
                if _pn:
                    _names.update(_pn)
                _co = _c.get("coeffs")
                if isinstance(_co, dict):
                    _names.update(_co.keys())
                if _p1_name in _names and _p2_name in _names:
                    _constraint_active = True
                    break

        if not show_legend:
            pass
        elif _constraint_active:
            _msg = (f"constraint links {_p1_name} ↔ {_p2_name} — "
                    f"shape reflects the constrained landscape, "
                    f"not free identifiability")
            ax.text(0.02, 0.98, _msg, transform=ax.transAxes,
                    fontsize=7, ha="left", va="top", color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#7a3b00", edgecolor="white",
                              lw=0.5, alpha=0.9), zorder=5)
        else:
            # Heuristic: compare the principal axes of the iso-1.5 region.
            # If the iso-region is much longer along one diagonal than the
            # perpendicular, flag a correlation.
            _mask15 = (ratio <= 1.5) & np.isfinite(ratio)
            if _mask15.any():
                _ys, _xs = np.where(_mask15)
                _xy = np.column_stack([dx[_xs] / max(span_or_1(dx), 1e-9),
                                       dy[_ys] / max(span_or_1(dy), 1e-9)])
                if _xy.shape[0] >= 4:
                    _cov = np.cov(_xy.T)
                    _eig = np.linalg.eigvalsh(_cov)
                    _ratio = float(_eig.max() / max(_eig.min(), 1e-12))
                    if _ratio > 9:
                        _msg = "long diagonal valley → strong correlation"
                    elif _ratio > 3:
                        _msg = "elongated valley → moderate correlation"
                    else:
                        _msg = "round well → parameters independently identified"
                    ax.text(0.02, 0.98, _msg, transform=ax.transAxes,
                            fontsize=7, ha="left", va="top", color="white",
                            bbox=dict(boxstyle="round,pad=0.25",
                                      facecolor="#222", edgecolor="white",
                                      lw=0.5, alpha=0.85), zorder=5)
    except Exception:
        pass
    return fig


def span_or_1(arr):
    """Tiny helper: range of arr or 1.0 fallback (used for axis-scaling)."""
    try:
        a = np.asarray(arr, dtype=float)
        s = float(np.nanmax(a) - np.nanmin(a))
        return s if s > 0 else 1.0
    except Exception:
        return 1.0


# ════════════════════════════════════════════════════════════════════════
# Parameter correlation matrix (linear / Hessian-based)
# ════════════════════════════════════════════════════════════════════════
#
# The Hessian covariance matrix stored in stats by every fit module
# carries the full pairwise covariance — not just the diagonal variance
# (= square of the standard error).  Normalising to correlation:
#
#     ρ_ij  =  cov_ij  /  √(cov_ii · cov_jj)
#
# gives a value in [-1, +1] that measures pairwise linear dependence.
# This is the LINEAR diagnostic — complementary to the nonlinear 2D
# RMSE profile.  When ρ_ij ≈ ±1, the Hessian-derived standard errors
# are misleading because the surface is a ridge.  When ρ_ij ≈ 0 the
# parameters are independently identified at the optimum.
#
# Both views are useful: the Hessian correlation is FAST and global;
# the 2D RMSE profile is SLOWER and exposes nonlinear structure (e.g.
# bananas where the linear approximation says "well-conditioned" but
# the truth is a curved valley).  SupraFit and SIVVU both report this
# linear matrix; Musketeer doesn't, and neither did Equilibrist until
# now.

def compute_correlation_matrix(stats: dict):
    """Return (ρ_matrix, names) from the stored Hessian covariance.

    ``stats`` must carry both ``param_cov`` (the n × n covariance
    matrix returned by ``_hessian_errors``) and ``param_cov_names``
    (the row/column order).  Returns ``(None, [])`` if the covariance
    is missing or numerically degenerate (zero-variance row).
    """
    cov   = stats.get("param_cov")
    names = stats.get("param_cov_names") or []
    if cov is None or not names:
        return None, []
    try:
        cov = np.asarray(cov, dtype=float)
        n   = cov.shape[0]
        if n == 0 or n != len(names):
            return None, []
        # Standard deviations from the diagonal; protect against
        # numerical zero or negative (clipping retains a defined ρ).
        sd = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        denom = np.outer(sd, sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(denom > 0, cov / denom, 0.0)
        # Clip to [-1, 1] to absorb tiny numerical overshoot
        rho = np.clip(rho, -1.0, 1.0)
        np.fill_diagonal(rho, 1.0)
        return rho, list(names)
    except Exception:
        return None, []


def make_correlation_heatmap(stats: dict, *,
                              title: Optional[str] = None,
                              annotate: bool = True):
    """Render the parameter correlation heatmap as a matplotlib Figure.

    Cells coloured on a diverging RdBu scale, value range fixed to
    [-1, +1] so colour intensity is directly comparable across fits.
    Off-diagonal cells optionally annotated with ρ to two decimal
    places — switch off for very large parameter sets where the text
    becomes illegible.
    """
    import matplotlib.pyplot as plt

    rho, names = compute_correlation_matrix(stats)
    if rho is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.text(0.5, 0.5,
                "Parameter covariance is unavailable\n"
                "(Hessian was not computed for this fit).",
                ha="center", va="center", transform=ax.transAxes,
                color="#808080", fontsize=10)
        ax.set_axis_off()
        if title: fig.suptitle(title)
        return fig

    # A 1×1 matrix carries no information (ρ = 1 by definition).
    # Tell the user explicitly rather than rendering a single red cell.
    if len(names) < 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.text(0.5, 0.5,
                "Correlation matrix is N/A —\n"
                "only one parameter was fitted.",
                ha="center", va="center", transform=ax.transAxes,
                color="#808080", fontsize=10)
        ax.set_axis_off()
        if title: fig.suptitle(title)
        return fig

    n = len(names)
    # Size scales gently with n; cap so very large fits don't blow up
    side = max(3.2, min(0.55 * n + 2.0, 8.0))
    fig, ax = plt.subplots(figsize=(side, side * 0.88),
                           constrained_layout=True)
    im = ax.imshow(rho, vmin=-1.0, vmax=1.0, cmap="RdBu_r",
                   aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    if annotate and n <= 12:
        # Choose text colour for legibility against the cell colour
        for i in range(n):
            for j in range(n):
                v = float(rho[i, j])
                # Dark text on light cells, white on dark cells
                color = "white" if abs(v) > 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cb.set_label("Pearson ρ (Hessian)", fontsize=9)
    ax.set_title(title or "Parameter correlation matrix",
                 fontsize=10)

    # Interpretive corner note: count off-diagonal |ρ| > 0.9
    if n >= 2:
        off = rho[np.triu_indices(n, k=1)]
        n_strong   = int(np.sum(np.abs(off) > 0.9))
        n_moderate = int(np.sum((np.abs(off) > 0.5) & (np.abs(off) <= 0.9)))
        if n_strong > 0:
            msg = (f"⚑ {n_strong} pair(s) with |ρ| > 0.9 — Hessian "
                   "errors likely understate the true uncertainty")
            color = "#A03000"
        elif n_moderate > 0:
            msg = (f"{n_moderate} pair(s) with 0.5 < |ρ| ≤ 0.9 — "
                   "moderate correlation, worth checking with the "
                   "2D profile")
            color = "#806000"
        else:
            msg = "All |ρ| ≤ 0.5 — parameters effectively independent"
            color = "#206020"
        fig.text(0.5, -0.02, msg, ha="center", va="top",
                 fontsize=8, color=color)

    return fig


# ════════════════════════════════════════════════════════════════════════
# Parameter identifiability — eigenanalysis of the Hessian covariance
# ════════════════════════════════════════════════════════════════════════
#
# For coupled multi-parameter fits, the per-parameter 1σ printed in
# the stats block is local and one-dimensional; it cannot reveal that
# two parameters are tightly correlated and only their sum or ratio is
# actually constrained by the data.  The 2D RMSE profile gives the
# nonlinear pairwise picture, but it only covers one pair at a time.
# The compact global view is the **eigendecomposition of the
# parameter covariance matrix**:
#
#     C = V · diag(λ₁ ≤ λ₂ ≤ … ≤ λ_p) · Vᵀ
#
# Each eigendirection vᵢ is a linear combination of the fitted
# parameters; σᵢ = √λᵢ is the standard deviation of the fit along
# that direction.  Sorted ascending, the FIRST direction is the
# **stiffest** (smallest variance ⇒ best determined by the data),
# the LAST is the **sloppiest**.  The spread of eigenvalues across
# many orders of magnitude is the signature of a "sloppy model" in
# the sense of Brown & Sethna (Phys. Rev. E 2003) and Gutenkunst et
# al. (PLoS Comput. Biol. 2007).  The condition number κ = σ_max /
# σ_min summarises the degeneracy: κ > 10⁶ typically signals that
# at least one parameter combination is non-identifiable.
#
# Equilibrist context: parameters in equilibrium mode are log K's;
# in kinetics mode they are log k's (when the script uses ``log k =
# …``) or linear rate constants (``k = …``).  σ is reported in the
# same units as the fitted parameter — for log-space parameters that
# means σ < 0.1 is ≲25 % relative error in K (well determined) and
# σ > 1 means the parameter combination can flex by an order of
# magnitude without moving SSR (unconstrained).  Reference for the
# binding-constant context: Hibbert & Thordarson, Chem. Commun.
# 2016, 52, 12792 — which already advocates moving beyond local σ
# for coupled equilibria.

def compute_identifiability(stats: dict) -> Optional[dict]:
    """Eigenanalysis of the parameter covariance matrix.

    ``stats`` must carry ``param_cov`` (the p × p covariance returned
    by ``_hessian_errors``) and ``param_cov_names`` (the row/column
    parameter ordering).  Eigenvalues are sorted ascending so
    eigendirection 0 is the stiffest.

    Returns ``None`` on missing or numerically degenerate covariance.
    Otherwise returns a dict:

    ``eigenvalues``       (p,)         λ₁ ≤ … ≤ λ_p
    ``eigenvectors``      (p, p)       columns are vᵢ
    ``sigmas``            (p,)         √λᵢ
    ``names``             list[str]    parameter names
    ``condition_number``  float        σ_max / σ_min
    ``decades``           float        log10(condition_number)
    ``directions``        list[dict]   one entry per eigendirection,
                                       each with ``index``, ``sigma``,
                                       ``weights`` (name → component),
                                       ``dominant`` (sorted list), and
                                       ``tag`` ∈ {stiff, intermediate,
                                       sloppy}
    """
    cov = stats.get("param_cov")
    names = stats.get("param_cov_names") or []
    if cov is None or not names:
        return None
    try:
        C = np.asarray(cov, dtype=float)
        p = C.shape[0]
        if p == 0 or p != len(names):
            return None
        # ``eigh`` returns eigenvalues in ascending order for a
        # symmetric matrix — perfectly aligned with our convention.
        # Symmetrize defensively in case of round-off asymmetry.
        C_sym = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C_sym)
        # Clip tiny numerical negatives (covariance is positive
        # semidefinite in theory; rounding can dip just below zero).
        eigvals = np.clip(eigvals, 0.0, None)
        sigmas = np.sqrt(eigvals)
        # Condition number on σ (one square root, more intuitive than
        # on λ).  Skip zero eigenvalues — they represent perfectly
        # unconstrained directions and would make κ = ∞ uninformative.
        pos = sigmas[sigmas > 0]
        if len(pos) >= 2:
            kappa = float(pos[-1] / pos[0])
            decades = float(np.log10(kappa))
        elif len(pos) == 1:
            kappa = 1.0
            decades = 0.0
        else:
            kappa = float("inf")
            decades = float("inf")
        # Per-direction breakdown
        directions = []
        for i in range(p):
            sigma_i = float(sigmas[i])
            vec = eigvecs[:, i]
            weights = {names[j]: float(vec[j]) for j in range(p)}
            dominant = sorted(weights.items(),
                              key=lambda kv: -abs(kv[1]))
            # Tag thresholds calibrated for log-space parameters
            # (the common case in Equilibrist).  In linear units the
            # interpretation has to be made by the user, but the
            # ordering and the κ are still meaningful.
            if sigma_i < 0.1:
                tag = "stiff"
            elif sigma_i > 1.0:
                tag = "sloppy"
            else:
                tag = "intermediate"
            directions.append({
                "index":    i,
                "sigma":    sigma_i,
                "weights":  weights,
                "dominant": dominant,
                "tag":      tag,
            })
        # ── Per-parameter ranking (Hessian diagonal projection) ─────
        # H = C⁻¹.  σ_conditional(k) = 1 / √H_kk is the uncertainty
        # in parameter k IF ALL OTHERS WERE HELD FIXED — the raw
        # stiffness of the fit along that single axis.  Contrast
        # with σ_marginal(k) = √C_kk, the uncertainty WITH the
        # others free to vary.  Their ratio is the per-K correlation
        # dilution factor.  Computed here so both the figure and the
        # ranking table read from one source.
        diag_C = np.diag(C_sym)
        sigma_marg = np.sqrt(np.clip(diag_C, 0.0, None))
        try:
            H = np.linalg.inv(C_sym)
        except np.linalg.LinAlgError:
            H = np.linalg.pinv(C_sym)
        diag_H = np.diag(H)
        sigma_cond = np.where(diag_H > 0,
                               1.0 / np.sqrt(np.clip(diag_H, 1e-30, None)),
                               np.inf)
        # Per-parameter rows, sorted by σ_cond ascending = highest
        # impact first.
        order_k = np.argsort(sigma_cond)
        per_param = []
        for rank_k, idx in enumerate(order_k, start=1):
            sc = float(sigma_cond[idx])
            sm = float(sigma_marg[idx])
            dilution = (sm / sc) if (sc > 0 and np.isfinite(sc)) else float("nan")
            if sc < 0.1:
                tag_k = "stiff"
            elif sc > 1.0:
                tag_k = "sloppy"
            else:
                tag_k = "intermediate"
            per_param.append({
                "rank":       rank_k,
                "name":       names[idx],
                "sigma_cond": sc,
                "sigma_marg": sm,
                "dilution":   dilution,
                "tag":        tag_k,
            })
        return {
            "eigenvalues":      eigvals,
            "eigenvectors":     eigvecs,
            "sigmas":           sigmas,
            "names":            list(names),
            "condition_number": kappa,
            "decades":          decades,
            "directions":       directions,
            "per_param":        per_param,
        }
    except Exception:
        return None


def make_identifiability_figure(stats: dict, *,
                                 title: Optional[str] = "Parameter identifiability",
                                 figsize: tuple = (11.0, 5.5)):
    """Three-panel matplotlib figure for the sloppy-models analysis.

    All three panels render side-by-side in a single row:

      Left:   sloppy spectrum — σᵢ as horizontal bars on a log x-axis,
              colour-coded by tag.  Stiffest direction at top, sloppiest
              at bottom.  Dotted reference lines at σ = 0.1 and σ = 1.0
              delineate the stiff / intermediate / sloppy regions.

      Middle: eigendirection composition heatmap — rows are
              eigendirections (stiffest at top), columns are parameter
              names.  Diverging RdBu palette centred on zero.  Cell
              values annotated when there are at most 8 parameters.

      Right:  per-parameter impact — same visual language as the
              sloppy spectrum, but on individual parameters: σ (alone)
              = 1/√H_kk as horizontal bars on a log x-axis, with the
              highest-impact parameter at the top.  Same colour map
              and threshold lines.  This is the per-K projection of
              the eigenanalysis on the left.

    Returns a Figure with a placeholder if the covariance is missing
    or the problem is trivial (single parameter).
    """
    import matplotlib.pyplot as plt
    res = compute_identifiability(stats)
    if res is None:
        fig, ax = plt.subplots(figsize=(figsize[0], 3.5))
        ax.text(0.5, 0.5,
                "Parameter covariance is unavailable\n"
                "(the optimizer did not return a usable Hessian).",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#777")
        ax.set_axis_off()
        return fig
    eigvecs = res["eigenvectors"]
    sigmas  = res["sigmas"]
    names   = res["names"]
    p       = len(names)

    if p == 1:
        fig, ax = plt.subplots(figsize=(figsize[0], 3.5))
        ax.text(0.5, 0.5,
                f"Only 1 fitted parameter — eigenanalysis is trivial.\n"
                f"σ = {sigmas[0]:.2e}  (= Hessian σ in stats panel).",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11)
        ax.set_axis_off()
        return fig

    color_map = {"stiff":         "#2ca02c",   # green
                 "intermediate":  "#7f7f7f",   # grey
                 "sloppy":        "#d62728"}   # red
    s_floor = 1e-6                              # log-axis safety floor

    fig = plt.figure(figsize=figsize)
    # Manual axes positioning in figure-coordinate fractions.  Using
    # explicit fig.add_axes rather than a GridSpec because gridspec +
    # set_box_aspect + colorbar-steal-width interact in ways that
    # produced subtle height mismatches between the heatmap and the
    # bar panels.  The four axes below all share identical `bottom`
    # and `height`, so their tops/bottoms line up exactly.
    #
    # Width budget (all in figure fractions, summing to 1.0):
    #   left_margin + 2·bar_w + 2·G + heat_w + cax_pad + cax_w + right_margin = 1
    #   with bar_w = 2u and heat_w = 3u, so 7u + 2G + everything else = 1.
    # Vertical: bar panel "shape" is 2:3 width:height in *data* terms;
    # heatmap is 3:3 (square).  Both have height = 3u in horizontal-u
    # units → converted to vertical figure fraction via fig_w / fig_h.
    fig_w, fig_h = figsize
    left_margin  = 0.06
    right_margin = 0.04
    cax_w        = 0.012
    cax_pad      = 0.005
    G            = 0.05                       # inter-panel gap
    u = (1 - left_margin - right_margin
         - 2*G - cax_pad - cax_w) / 7
    bar_w   = 2 * u
    heat_w  = 3 * u
    bottom  = 0.22                            # room for x-labels below
    panel_h = 3 * u * (fig_w / fig_h)
    # Defensive: if the caller passes a very short figsize, clamp the
    # panel height so the panel doesn't run off the top of the figure.
    max_panel_h = 1 - bottom - 0.12           # 0.12 ≈ suptitle + title
    if panel_h > max_panel_h:
        panel_h = max_panel_h

    x_spec = left_margin
    x_imp  = x_spec + bar_w + G
    x_heat = x_imp  + bar_w + G
    x_cax  = x_heat + heat_w + cax_pad

    ax_spec = fig.add_axes([x_spec, bottom, bar_w,  panel_h])
    ax_imp  = fig.add_axes([x_imp,  bottom, bar_w,  panel_h])
    ax_heat = fig.add_axes([x_heat, bottom, heat_w, panel_h])
    cax     = fig.add_axes([x_cax,  bottom, cax_w,  panel_h])

    # ── Left: sloppy spectrum ──────────────────────────────────────
    colors_v = [color_map[d["tag"]] for d in res["directions"]]
    s_show = np.clip(sigmas, s_floor, None)
    y_pos = np.arange(p)
    ax_spec.barh(y_pos, s_show, color=colors_v, alpha=0.85, height=0.6)
    ax_spec.set_xscale("log")
    ax_spec.set_yticks(y_pos)
    ax_spec.set_yticklabels([f"v{i+1}" for i in range(p)])
    ax_spec.set_xlabel("σᵢ  (parameter units)")
    ax_spec.set_title("Sloppy spectrum")
    ax_spec.axvline(0.1, color="#666", linestyle=":",
                     linewidth=0.8, alpha=0.6)
    ax_spec.axvline(1.0, color="#666", linestyle=":",
                     linewidth=0.8, alpha=0.6)
    ax_spec.invert_yaxis()
    ax_spec.grid(True, axis="x", linestyle="--", alpha=0.3)

    # ── Middle: eigenvector composition heatmap ────────────────────
    V = eigvecs.T
    vmax = max(abs(V.min()), abs(V.max()), 0.1)
    # aspect='auto' lets the imshow fill the manually-positioned
    # square box.  Cells stay square because the box is square and
    # the data is p × p.
    im = ax_heat.imshow(V, cmap="RdBu_r", aspect="auto",
                         vmin=-vmax, vmax=vmax)
    ax_heat.set_xticks(np.arange(p))
    ax_heat.set_xticklabels(names)            # horizontal — short names
    ax_heat.set_yticks(y_pos)
    ax_heat.set_yticklabels([f"v{i+1}" for i in range(p)])
    ax_heat.set_title("Eigendirection composition")
    # Annotate every cell with its value at the same font size as the
    # axis tick labels.  No suppression threshold — small entries are
    # informative too (e.g., a near-zero weight means a parameter
    # makes no contribution to that eigendirection).
    for i in range(p):
        for j in range(p):
            val = V[i, j]
            ax_heat.text(j, i, f"{val:+.2f}",
                          ha="center", va="center",
                          color="white" if abs(val) > vmax / 2
                          else "black",
                          fontsize=10)
    plt.colorbar(im, cax=cax, label="weight")

    # ── Right: per-parameter impact (mirror of the sloppy spectrum) ─
    per_param = res.get("per_param") or []
    if per_param:
        sorted_names  = [r["name"]       for r in per_param]
        sc_sorted     = [r["sigma_cond"] for r in per_param]
        bar_colors_k  = [color_map[r["tag"]] for r in per_param]
        y_pos_k = np.arange(p)
        sc_show = np.clip(sc_sorted, s_floor, None)
        ax_imp.barh(y_pos_k, sc_show, color=bar_colors_k,
                     alpha=0.85, height=0.6)
        ax_imp.set_xscale("log")
        ax_imp.set_yticks(y_pos_k)
        ax_imp.set_yticklabels(sorted_names)
        ax_imp.set_xlabel("σ (alone)  (parameter units)")
        ax_imp.set_title("Per-parameter impact")
        ax_imp.axvline(0.1, color="#666", linestyle=":",
                        linewidth=0.8, alpha=0.6)
        ax_imp.axvline(1.0, color="#666", linestyle=":",
                        linewidth=0.8, alpha=0.6)
        ax_imp.invert_yaxis()       # highest-impact at top
        ax_imp.grid(True, axis="x", linestyle="--", alpha=0.3)

    if title:
        fig.suptitle(title, y=0.97, fontsize=11)
    return fig


def render_identifiability_panel(stats: dict, *,
                                  key: str = "ident") -> None:
    """Streamlit panel: spectrum figure, κ banner, eigendirection
    table, PNG download.  No-op if Streamlit isn't available.
    """
    try:
        import streamlit as st
    except Exception:
        return
    res = compute_identifiability(stats)
    if res is None:
        st.caption(
            "Parameter identifiability analysis requires the Hessian "
            "covariance matrix.  Run a fit first, or check that the "
            "optimizer returned a usable Hessian."
        )
        return
    p = len(res["names"])
    if p < 2:
        st.caption(
            "Only one fitted parameter — the eigenanalysis is trivial.  "
            "Use the Hessian σ from the **Fit Statistics** block above."
        )
        return

    # Figure
    fig = make_identifiability_figure(stats)
    st.pyplot(fig, clear_figure=False)

    # Condition number banner — tag thresholds chosen for log-space
    # parameters but the qualitative meaning carries to linear units.
    kappa = res["condition_number"]
    decades = res["decades"]
    if not np.isfinite(kappa) or kappa > 1e6:
        st.error(
            f"**κ = {kappa:.1e}** — severely ill-conditioned "
            f"(≥6 decades of sloppiness).  At least one parameter "
            f"combination is non-identifiable.  The bottom "
            f"eigendirection(s) below are essentially unconstrained "
            f"by the data and would benefit from a constraint, a "
            f"fixed value, or removal from the fit."
        )
    elif kappa > 1e4:
        st.warning(
            f"**κ = {kappa:.1e}** — ill-conditioned "
            f"({decades:.1f} decades).  Some parameter combinations "
            f"are poorly constrained; bootstrap CIs will be wide for "
            f"those, and the Hessian σ for individual parameters "
            f"likely understates the true uncertainty."
        )
    elif kappa > 1e2:
        st.info(
            f"**κ = {kappa:.1e}** ({decades:.1f} decades of sloppiness)."
        )
    else:
        st.success(f"**κ = {kappa:.1e}** — well-conditioned.")
    st.caption(
        "**About κ.**  The *condition number* is the ratio of the "
        "loosest direction's σ to the stiffest direction's σ "
        "(σ_max / σ_min) — geometrically, the aspect ratio of the "
        "confidence ellipsoid.  κ = 1 = a sphere (every direction "
        "equally tight); κ = 100 = a cigar 100× longer than wide; "
        "κ = 10⁶ = essentially a pancake in parameter space, with "
        "one direction the data can't see."
    )

    # Eigendirection table
    rows = []
    for d in res["directions"]:
        # Render dominant terms (|w| > 0.1) as a combination string
        terms = []
        for name, w in d["dominant"]:
            if abs(w) < 0.1:
                break
            sign = "+" if w > 0 else "−"
            terms.append(f"{sign} {abs(w):.2f}·{name}")
        combo = " ".join(terms) if terms else "(all weights < 0.1)"
        tag_label = {"stiff":        "✓ stiff",
                     "intermediate": "—",
                     "sloppy":       "⚠ sloppy"}[d["tag"]]
        rows.append({
            "direction":   f"v{d['index'] + 1}",
            "σ":           f"{d['sigma']:.2e}",
            "tag":         tag_label,
            "combination": combo,
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "σᵢ is in the units of the fitted parameter — log units for "
        "log K / log k parameters, linear units for direct rate "
        "constants.  Threshold convention for log-space parameters: "
        "σ < 0.1 ≈ ≤25 % relative error in K (stiff, well-determined); "
        "σ > 1 means the parameter combination can flex by an order "
        "of magnitude without changing SSR (sloppy, unconstrained).  "
        "References: Brown, K. S.; Sethna, J. P. *Phys. Rev. E* "
        "**2003**, *68*, 021904.  "
        "Gutenkunst, R. N.; Waterfall, J. J.; Casey, F. P.; Brown, "
        "K. S.; Myers, C. R.; Sethna, J. P. *PLoS Comput. Biol.* "
        "**2007**, *3*, 1871–1878.  "
        "Hibbert, D. B.; Thordarson, P. *Chem. Commun.* "
        "**2016**, *52*, 12792–12805."
    )

    # ── Per-parameter impact ranking ─────────────────────────────────
    # The eigenanalysis above ranks combinations.  ``compute_identifi-
    # ability`` also computes the per-parameter projection via the
    # Hessian H = C⁻¹: σ_conditional(k) = 1/√H_kk is the standard
    # deviation in parameter k IF ALL OTHERS WERE HELD FIXED at their
    # best-fit values — the "raw stiffness" of the fit along the K_k
    # axis.  σ_marginal(k) = √C_kk (= the Hessian σ shown in the
    # Fit Statistics panel) allows the OTHER parameters to also move;
    # the ratio σ_marginal / σ_conditional is the per-K correlation
    # dilution factor.  Bar chart of σ_conditional lives in the
    # third panel of the main figure above; this table adds the
    # numeric ranking and the dilution column.
    per_param = res.get("per_param") or []
    if per_param:
        impact_rows = []
        for entry in per_param:
            sc = entry["sigma_cond"]
            sm = entry["sigma_marg"]
            dil = entry["dilution"]
            tag_label = {"stiff":        "✓ stiff",
                         "intermediate": "—",
                         "sloppy":       "⚠ sloppy"}[entry["tag"]]
            dilution_str = (f"{dil:.2f}×"
                            if np.isfinite(dil) else "—")
            impact_rows.append({
                "rank":                  entry["rank"],
                "parameter":             entry["name"],
                "σ (alone)":             f"{sc:.2e}",
                "σ (in practice)":       f"{sm:.2e}",
                "correlation dilution":  dilution_str,
                "tag":                   tag_label,
            })
        st.markdown("**Per-parameter impact ranking**")
        st.dataframe(impact_rows, hide_index=True,
                     use_container_width=True)
        st.caption(
            "**σ (alone)** = conditional uncertainty in this "
            "parameter if all others were held exactly at their "
            "best-fit values (= 1/√H_kk).  This is the *intrinsic* "
            "stiffness the fit assigns to the parameter — smaller "
            "= higher impact, ranked here from biggest to smallest "
            "impact.  **σ (in practice)** = marginal uncertainty "
            "with the others free to vary (= √C_kk, same value as "
            "the Hessian σ in the Fit Statistics block).  "
            "**Correlation dilution** = ratio σ-in-practice ÷ "
            "σ-alone: ≈ 1× means the parameter is essentially "
            "independent; ≫ 1× means its real-world uncertainty "
            "is dominated by trade-offs with other parameters, "
            "and the eigendirection table above identifies which."
        )

    # PNG download
    try:
        import io as _io
        from datetime import datetime as _dt
        _buf = _io.BytesIO()
        fig.savefig(_buf, format="png", dpi=200, bbox_inches="tight")
        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download identifiability PNG",
            data=_buf.getvalue(),
            file_name=f"Equilibrist_identifiability_{_ts}.png",
            mime="image/png",
            key=f"_ident_dl_{key}",
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
# Parameter significance — t-test on each fitted parameter
# ════════════════════════════════════════════════════════════════════════
#
# For every fitted parameter θ̂ᵢ with standard error SE(θ̂ᵢ) (from the
# Hessian), the t-statistic against a null θ₀ᵢ is:
#
#     tᵢ  =  (θ̂ᵢ − θ₀ᵢ) / SE(θ̂ᵢ)
#
# Under H₀ this follows a t-distribution with df = n − p (residual
# degrees of freedom).  Two-tailed p-value:
#
#     pᵢ  =  2 · [1 − F_t(|tᵢ|, df)]
#
# All quantities live in LOG10 SPACE because that is the space the
# optimizer + Hessian work in:
#   * For binding constants the parameter is log K directly.
#   * For fitted concentrations and titrants the parameter is
#     log10(c / mM), so the SE in stats is in log10 units; the value
#     used for the test is log10(fitted_concs[name]).
#
# Default null is θ₀ = 0 for every parameter:
#   * For log K: K = 1 (no binding preference).
#   * For log10(c): c = 1 mM (the default starting concentration in
#     many tutorial scripts; convenient as a sanity check).
# The caller can override the default per-parameter via the
# ``null_overrides`` dict — that's what the UI exposes as an editable
# column so the user can test against literature values (e.g. is my
# log K significantly different from a previously-published 4.5?).

def compute_param_t_tests(stats: dict,
                          null_overrides: Optional[dict] = None
                          ) -> list:
    """Compute per-parameter t-statistics + p-values against given nulls.

    Returns a list of row dicts, one per fitted parameter, with keys:
    ``name``, ``value``, ``se``, ``null``, ``t``, ``p``, ``df``,
    ``stars`` (significance stars), ``kind`` ("log K" / "log conc" /
    "log titrant").  Rows with no SE (Hessian failure) carry NaN in
    ``t``/``p`` but are still emitted so the UI can flag them.
    """
    try:
        from scipy.stats import t as _student_t
    except Exception:
        _student_t = None

    rows: list = []
    nulls = null_overrides or {}

    pv = stats.get("param_values") or {}
    pe = stats.get("param_errors") or {}
    fc = stats.get("fitted_concs") or {}
    ft = stats.get("fitted_titrants") or {}

    n_data = int(stats.get("n_points") or 0)
    # after a global fit the SE comes from all tabs: use their total point count
    _gtot = (stats.get("global_fit") or {}).get("n_points_total") if isinstance(stats.get("global_fit"), dict) else None
    if _gtot:
        n_data = int(_gtot)
    n_par  = int(stats.get("n_params") or 0)
    df_resid = max(n_data - n_par, 1)

    def _emit(name, value, se, kind):
        null = float(nulls.get(name, 0.0))
        if se is None or not np.isfinite(se) or se <= 0:
            rows.append({"name": name, "value": float(value), "se": None,
                         "null": null, "t": float("nan"),
                         "p": float("nan"), "df": df_resid,
                         "stars": "", "kind": kind})
            return
        t_val = (float(value) - null) / float(se)
        if _student_t is not None and np.isfinite(t_val):
            try:
                p_val = float(2.0 * (1.0 - _student_t.cdf(abs(t_val), df_resid)))
            except Exception:
                p_val = float("nan")
        else:
            p_val = float("nan")
        if   np.isnan(p_val): stars = ""
        elif p_val < 0.001:   stars = "***"
        elif p_val < 0.01:    stars = "**"
        elif p_val < 0.05:    stars = "*"
        else:                  stars = "n.s."
        rows.append({"name": name, "value": float(value),
                     "se": float(se), "null": null,
                     "t": float(t_val), "p": p_val,
                     "df": df_resid, "stars": stars, "kind": kind})

    # log K's (binding constants) — value is already in log10 space
    for name, val in pv.items():
        try:
            _emit(name, float(val), pe.get(name), "log K")
        except Exception:
            continue

    # Fitted concentrations — stored as linear mM in fitted_concs,
    # but the optimizer (and the Hessian SE) work in log10 space.
    # So the t-test value is log10(c) and the SE is what's in
    # param_errors[name].
    for name, mM in fc.items():
        try:
            if mM is None or mM <= 0:
                continue
            _emit(name, float(np.log10(float(mM))),
                  pe.get(name), "log conc")
        except Exception:
            continue

    # Fitted titrants — same convention as fitted concentrations
    for name, mM in ft.items():
        try:
            if mM is None or mM <= 0:
                continue
            _emit(name, float(np.log10(float(mM))),
                  pe.get(name), "log titrant")
        except Exception:
            continue

    return rows


# ════════════════════════════════════════════════════════════════════════
# Jackknife influence plot
# ════════════════════════════════════════════════════════════════════════
#
# For each fitted parameter, draws a scatter of (point index, Δᵢ) where
# Δᵢ = θ_full − θ_drop_i.  Coloured per signal column so the user sees
# which signal the influential observations belong to.  Horizontal
# dashed lines at ± 2·SE_jack give a quick visual threshold.

def make_jackknife_figure(jack_result: dict,
                          *,
                          title: Optional[str] = None,
                          x_kind: str = "x_value",
                          x_label: Optional[str] = None):
    """Render the per-step jackknife influence plot.

    Each scatter point is one titration step (or one spectrum / one
    time slice for spectra and kinetics fits).  Δᵢ = θ_full − θ_drop_i
    quantifies how much that step's removal moves the parameter.  Steps
    where |Δᵢ| exceeds ±2·MAD of the influence distribution are likely
    leverage / outlier observations.

    Parameters
    ----------
    jack_result : dict
        Output of any ``jackknife_<mode>`` function.  Must carry
        ``steps`` (list of ``(row_idx, x_display, n_obs)`` tuples),
        ``influence``, ``jack_se``, and ``best_fit``.
    title : str, optional
        Figure suptitle.
    x_kind : "x_value" or "index"
        ``"x_value"`` plots Δ vs the step's x-coordinate.  ``"index"``
        plots vs the step order.
    x_label : str, optional
        The exact label to put on the x-axis when ``x_kind="x_value"``.
        Pass the user's plot x-expression (``"H0/G0"``, ``"t"``, …) so
        the jackknife panel reads the same as the main residual /
        observation plot above.
    """
    import matplotlib.pyplot as plt

    if jack_result is None or not jack_result.get("influence"):
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.text(0.5, 0.5, "Jackknife result is unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                color="#808080", fontsize=10)
        ax.set_axis_off()
        if title: fig.suptitle(title)
        return fig

    steps     = jack_result.get("steps", []) or []
    influence = jack_result["influence"]
    jack_se   = jack_result.get("jack_se", {}) or {}
    full_fit  = jack_result.get("best_fit", {}) or {}

    # ``influence`` values are np.ndarray on a freshly-computed
    # jackknife but plain Python lists after a JSON restore (the
    # ``_scrub`` pass that prepares the result for json.dumps
    # converts numpy arrays to lists).  Test array-likeness rather
    # than ``isinstance(inf, np.ndarray)`` so reloaded sessions
    # don't fall through to the "no finite estimates" branch
    # with the figure area present but empty.
    plot_params = []
    for name, inf in influence.items():
        try:
            arr = np.asarray(inf, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.size and np.any(np.isfinite(arr)):
            plot_params.append(name)
    n_p = len(plot_params)
    if n_p == 0:
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.text(0.5, 0.5, "Jackknife produced no finite estimates.",
                ha="center", va="center", transform=ax.transAxes,
                color="#808080", fontsize=10)
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(n_p, 1,
                              figsize=(8.0, 2.4 * n_p + 0.6),
                              sharex=True, constrained_layout=True)
    if n_p == 1:
        axes = [axes]

    if x_kind == "index":
        x_all = np.arange(len(steps), dtype=float)
        xlabel = "Step index (one refit per row dropped)"
    else:
        x_all = np.array([s[1] for s in steps], dtype=float)
        xlabel = x_label if x_label else "x"

    # Per-parameter unit map — populated by every jackknife_<mode>
    # function via _build_param_kinds.  "log" → optimizer worked in
    # log10 space, so Δᵢ is in log units (Δ log K, Δ log k); axis
    # label needs to reflect that.  "linear_mM" → fitted_concs /
    # fitted_titrants are returned to the caller in mM, so Δᵢ is in
    # mM.  Missing entries fall back to "log" (the safer default
    # for parameters whose kind is unknown).
    param_kinds = jack_result.get("param_kinds", {}) or {}

    for ax, name in zip(axes, plot_params):
        inf = np.asarray(influence[name], dtype=float)
        se = float(jack_se.get(name, float("nan")))
        finite = np.isfinite(inf)
        inf_v = inf[finite]

        # Robust threshold: ±2·MAD(Δᵢ) — flags ~5 % of a normal
        # influence distribution.  Calibrated to the spread of the
        # individual Δᵢ values, NOT to SE_jack (which is on the scale
        # of the full-parameter uncertainty and is far wider).
        thr = float("nan")
        if len(inf_v) >= 4:
            med = float(np.median(inf_v))
            mad = float(np.median(np.abs(inf_v - med)))
            if mad > 0:
                thr = 2.0 * 1.4826 * mad

        # Single-colour scatter — one dot per step
        ax.scatter(x_all[finite], inf[finite],
                   s=26, alpha=0.78, edgecolor="white", linewidths=0.4,
                   color="#1f6fb4")

        if np.isfinite(thr) and thr > 0:
            ax.axhline(+thr, color="#A03000", ls=":", lw=0.9)
            ax.axhline(-thr, color="#A03000", ls=":", lw=0.9)

        # Y-limit: fit the data + a margin; only honour the threshold
        # if it isn't grotesquely far from the data spread.
        if len(inf_v):
            y_data_max = float(np.max(np.abs(inf_v))) * 1.25
            y_max = y_data_max
            if np.isfinite(thr) and thr < y_data_max * 2.5:
                y_max = max(y_max, thr * 1.15)
            if y_max > 0:
                ax.set_ylim(-y_max, y_max)

        ax.text(0.99, 0.96,
                f"SE_jack = {se:.4f}\nθ_full = {full_fit.get(name, float('nan')):.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#404040",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.25", alpha=0.85))
        ax.axhline(0, color="black", lw=0.4)

        # Per-parameter y-axis label respecting the parameter's unit.
        # Three kinds are supported:
        #   "log"       → optimizer space, e.g. log K for binding constants.
        #   "linear_k"  → rate constant in linear units (s⁻¹ or higher
        #                 order); we already converted from log space.
        #   "linear_mM" → fitted concentration or titrant, linear mM.
        kind = param_kinds.get(name, "log")
        if kind == "log":
            ax.set_ylabel(f"Δ log {name}\n(θ_full − θ_drop)", fontsize=9)
        elif kind == "linear_k":
            ax.set_ylabel(f"Δ {name}\n(θ_full − θ_drop)", fontsize=9)
        else:  # linear_mM
            ax.set_ylabel(f"Δ {name} / mM\n(θ_full − θ_drop)", fontsize=9)
        ax.set_title(f"Parameter: {name}", fontsize=10, loc="left")

    axes[-1].set_xlabel(xlabel, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig


# ════════════════════════════════════════════════════════════════════════
# Local sensitivity test (Masson/beta)
# ════════════════════════════════════════════════════════════════════════
#
# A paired-difference, no-refit sensitivity probe.  For each user-selected
# variable (whether or not it was a fitted parameter in the underlying
# fit), we evaluate RMSE on a 3^N grid of perturbations around the
# optimum (each variable at {-δ, 0, +δ} in log space), then for each
# variable take the average of the 2·3^(N-1) PAIRED differences:
#
#     diff_X(δX, others) = r(X*+δX, others) − r(X*, others)
#
# where r = (RMSE_new − RMSE_opt)/RMSE_opt.  By symmetric construction
# the cross-coupling H_AB terms cancel exactly, isolating the diagonal
# Hessian element:
#
#     f_X ≈ H_XX · δ² / (4·N_data·RMSE_opt²)
#
# This yields the CONDITIONAL standard error
#
#     σ_X_cond = δ / √(2·N_data·f_X)
#
# i.e., X's uncertainty if every other variable were held perfectly
# fixed.  Compared against the MARGINAL Hessian σ_X (which lets other
# parameters refit to compensate during the fit), the ratio
#
#     coupling_X = σ_X_marg / σ_X_cond  ≥ 1
#
# is the square root of the classical Variance Inflation Factor — a
# direct measure of how much of X's practical uncertainty is from
# parameter coupling vs. intrinsic data sensitivity.


def _compute_rmse_at(fa: dict, param_values: dict) -> Optional[float]:
    """Evaluate RMSE at a specific parameter configuration without
    actually re-fitting.

    ``param_values`` is a dict mapping a parameter name to its log10
    value:
        - log K          (for equilibrium constants, fitted or fixed)
        - log k          (for rate constants — optimizer-native log10)
        - log10(mM)      (for concentrations / titrant stocks)
        - log10(mL)      (for V0)

    Implementation: pins every supplied name into the appropriate slot
    of a deep-copied ``fa`` (start_logK / params / parsed), then calls
    the underlying fit function with ``maxiter=1`` and ``tol=1e30``.
    Combined with the maxiter<=1 → 1e-9 simplex-step guard inside each
    fit module, the optimizer's single allowed iteration cannot move
    away from the pinned configuration, so the reported RMSE is the
    forward-model value at exactly the requested parameter set.

    Returns None on dispatch failure (mode mismatch, missing data, etc.)
    """
    if fa.get("kind") == "global":
        from equilibrist_global import evaluate_global
        return evaluate_global(fa, param_values)
    # Linear-chart pinning is handled only by _refit_with_pinned;
    # define the flags here so the shared ITC branch is safe.
    _is_lin, _lin_val = False, None
    try:
        _kind = fa.get("kind", "eq")
        _tol  = 1e30
        _mxi  = 1
        _cstr = fa.get("constraints", []) or []

        if _kind == "eq":
            # Pin via start_logK and params copies
            _start = dict(fa.get("start_logK", fa.get("start_logk", {})))
            _params = fa.get("params")
            if _params is not None:
                _params = dict(_params)
                _params["conc0"]       = dict(_params.get("conc0", {}))
                _params["titrant_mMs"] = dict(_params.get("titrant_mMs", {}))

            for _name, _val in param_values.items():
                if _name in _start:
                    _start[_name] = float(_val)
                    continue
                if _params is None:
                    continue
                if _name in _params["conc0"]:
                    _params["conc0"][_name] = float(10.0 ** float(_val))
                elif _name in _params["titrant_mMs"]:
                    _params["titrant_mMs"][_name] = float(10.0 ** float(_val))
                elif _name == "V0_mL":
                    _params["V0_mL"] = float(10.0 ** float(_val))

            # Force-oneshot: empty fit_keys/conc_keys/titrant_keys would
            # crash Nelder-Mead with dim=0, so put one item back from
            # whichever list is non-empty in the original fa.
            _all_K = list(fa.get("fit_keys", []) or [])
            _all_C = list(fa.get("fit_conc_keys", []) or [])
            _all_T = list(fa.get("fit_titrant_keys", []) or [])
            if _all_K:    _fk, _fck, _ftk = [_all_K[0]], [], []
            elif _all_C:  _fk, _fck, _ftk = [], [_all_C[0]], []
            elif _all_T:  _fk, _fck, _ftk = [], [], [_all_T[0]]
            else:         _fk, _fck, _ftk = list(_start.keys())[:1], [], []

            _p   = fa["parsed_fit"]
            _net = fa["network"]
            _par = _params if _params is not None else fa.get("params")
            _xe  = fa["x_expr"]
            if fa.get("use_itc_fit"):
                # ITC. q_offset is a linear nuisance parameter eliminated by
                # lstsq at every objective evaluation and is routinely
                # negative, so it is not profiled by this log10-based
                # machinery; its trade-off with dH is reported by the
                # correlation matrix instead. Pass it through if fixed.
                from equilibrist_itc import fit_itc as _f
                _ok, _, _s, _ = _f(_p, _net, fa["itc_q_exp"], fa["itc_raw"],
                                    _par, _start, _fk, _xe,
                                    tolerance=_tol, maxiter=_mxi,
                                    timeout_s=float(fa.get("timeout_s", 30.0)),
                                    fit_conc_keys=_fck,
                                    q_offset_fixed=(_lin_val if _is_lin
                                                    else fa.get("q_offset_fixed")),
                                    constraints=_cstr,
                                    excl_set=fa.get("excl_set") or set())
                if isinstance(_s, dict):
                    _r = _s.get("rmse")
                    if _r is not None and np.isfinite(_r):
                        return float(_r)   # this function returns a float, not stats
                return None
            if fa.get("use_spectra_fit"):
                from equilibrist_fit_spectra import fit_spectra as _f
                _ok, _, _s, _ = _f(_p, _net, fa["spectra_data_fit"],
                                    _par, _start, _fk, _xe,
                                    fa.get("wl_min"), fa.get("wl_max"),
                                    _tol, _mxi, constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    fit_titrant_keys=_ftk,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
            elif fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if   _mode == "shift":       from equilibrist_fit_nmr import fit_nmr_shifts      as _f
                elif _mode == "integration": from equilibrist_fit_nmr import fit_nmr_integration as _f
                elif _mode == "mixed":       from equilibrist_fit_nmr import fit_nmr_mixed       as _f
                else: return None
                _ok, _, _s, _ = _f(_p, _net, fa["nmr_data_fit"],
                                    _par, _start, _fk, _xe,
                                    _tol, _mxi, constraints=_cstr,
                                    fit_conc_keys=_fck, fit_titrant_keys=_ftk)
            else:
                from equilibrist_fit_conc import fit_parameters as _f
                _ok, _, _s, _ = _f(_p, _net, fa.get("exp_data_fit") or {},
                                    _par, _start, _fk, _xe,
                                    tolerance=_tol, maxiter=_mxi,
                                    constraints=_cstr,
                                    fit_conc_keys=_fck, fit_titrant_keys=_ftk)
            if isinstance(_s, dict):
                # For NMR mixed mode, the optimizer minimized
                # variance-normalized chi² (the only way to combine
                # mM-valued integrations with ppm-valued shifts on a
                # comparable footing).  Raw rmse has its minimum at a
                # different point — so the LST must probe chi² to land
                # at the same anchor the fit converged on.  Other
                # modes' data_objective IS raw SSR, so chi2_rmse is
                # absent and we use rmse as before.
                if fa.get("use_nmr_fit") and fa.get("nmr_mode") == "mixed":
                    _r = _s.get("chi2_rmse", _s.get("rmse"))
                else:
                    _r = _s.get("rmse")
                if _r is not None and np.isfinite(_r):
                    return float(_r)
            return None

        if _kind == "kin":
            # Kinetics: pin via start_logk (rate constants AND eq
            # constants for hybrid fits) plus parsed concentrations and
            # volumes.
            _start  = dict(fa.get("start_logk", fa.get("start_logK", {})))
            _parsed = fa["parsed_fit"]
            _p_perturbed = dict(_parsed)
            _p_perturbed["concentrations"] = dict(_parsed.get("concentrations", {}))
            _p_perturbed["volumes"]        = dict(_parsed.get("volumes",        {}))

            for _name, _val in param_values.items():
                if _name in _start:
                    _start[_name] = float(_val)
                    continue
                if _name in _p_perturbed["concentrations"]:
                    _p_perturbed["concentrations"][_name] = float(10.0 ** float(_val))
                elif _name in _p_perturbed["volumes"]:
                    _p_perturbed["volumes"][_name] = float(10.0 ** float(_val))

            _all_K = list(fa.get("fit_keys", []) or [])
            _all_C = list(fa.get("fit_conc_keys", []) or [])
            if _all_K:   _fk, _fck = [_all_K[0]], []
            elif _all_C: _fk, _fck = [], [_all_C[0]]
            else:        _fk, _fck = list(_start.keys())[:1], []

            _t_max = float(fa["t_max"])
            if fa.get("use_spectra_fit"):
                from equilibrist_kinetics_spectra import fit_kinetics_spectra as _f
                _ok, _, _s, _ = _f(_p_perturbed, _start, fa["spectra_data_fit"], _fk,
                                    _t_max, 200,
                                    float(fa.get("wl_min") or 0.0),
                                    float(fa.get("wl_max") or 1e9),
                                    _tol, _mxi, constraints=_cstr,
                                    fit_conc_keys=_fck,
                                    allow_negative_eps=fa.get("allow_neg_eps", False))
            elif fa.get("use_nmr_fit"):
                _mode = fa.get("nmr_mode")
                if   _mode == "shift":       from equilibrist_kinetics_nmr import fit_kinetics_nmr_shifts      as _f
                elif _mode == "integration": from equilibrist_kinetics_nmr import fit_kinetics_nmr_integration as _f
                elif _mode == "mixed":       from equilibrist_kinetics_nmr import fit_kinetics_nmr_mixed       as _f
                else: return None
                _ok, _, _s, _ = _f(_p_perturbed, _start, fa["nmr_data_fit"], _fk,
                                    _t_max, 200, _tol, _mxi,
                                    constraints=_cstr, fit_conc_keys=_fck)
            else:
                from equilibrist_kinetics import fit_kinetics as _f
                _ok, _, _s, _ = _f(_p_perturbed, fa.get("exp_data_fit") or {},
                                    _start, _fk, _t_max, 200, _tol, _mxi,
                                    constraints=_cstr, fit_conc_keys=_fck)
            if isinstance(_s, dict):
                # Mixed-mode NMR (kinetics): same rationale as eq —
                # the LST must probe chi² rather than raw rmse so the
                # anchor coincides with the fit optimum.
                if fa.get("use_nmr_fit") and fa.get("nmr_mode") == "mixed":
                    _r = _s.get("chi2_rmse", _s.get("rmse"))
                else:
                    _r = _s.get("rmse")
                if _r is not None and np.isfinite(_r):
                    return float(_r)
            return None

    except Exception:
        return None
    return None


def compute_local_sensitivity(fa: dict, selected_vars: list, *,
                               step: float = 0.1,
                               n_jobs: int = 1,
                               sigma_marg: Optional[dict] = None,
                               fit_stats: Optional[dict] = None,
                               progress_callback=None) -> dict:
    """Compute the Masson local sensitivity test.

    ``selected_vars``: list of variable names (any subset of the full
    parameter space — fitted parameters AND held-fixed quantities).
    Each is perturbed by ±``step`` in log10 space; the 3^N grid is
    swept and per-variable f, σ_cond, σ_marg, coupling ratio are
    derived.

    ``sigma_marg``: optional dict mapping variable name → Hessian σ
    (marginal SE in log10 space).  Names not in the dict get a
    "n/a" coupling ratio (typical for held-fixed variables).

    ``fit_stats``: the post-fit statistics dict.  CRITICAL for correct
    anchoring — the optimum lives at the FITTED parameter values, not
    the user's sidebar starting values that ``fa["start_logK"]`` was
    built from.  If omitted, the test silently anchors on the starting
    values, which is wrong whenever the fit moved any parameter away
    from its starting point (f can come out negative, coupling ratio
    can drop below 1 — both physically impossible at a true optimum).

    Returns a result dict with keys:
        "rmse_opt"     — baseline RMSE at the optimum
        "n_data"       — number of data points (from fit_stats)
        "nominal"      — {name: log10-space nominal value}
        "f"            — {name: average paired-difference of r}
        "sigma_cond"   — {name: δ/√(2·N·f)} (conditional SE)
        "sigma_marg"   — {name: σ from Hessian} or None where unknown
        "coupling"     — {name: σ_marg/σ_cond} or None where unknown
        "n_grid"       — total 3^N grid evaluations
        "n_success"    — number of evaluations that returned finite
        "wall_seconds" — total time
        "step"         — step value used
        "selected_vars"— echoed selection
    """
    import time as _time
    t0 = _time.perf_counter()
    sel = list(selected_vars)
    Nsel = len(sel)
    if Nsel == 0:
        return {"selected_vars": [], "wall_seconds": 0.0,
                "best_message": "No variables selected."}

    # ── Build a CORRECTLY-ANCHORED baseline ──────────────────────────
    # The fit_stats dict carries the fitted parameter values that
    # define the optimum.  Starting from fa's sidebar values, we
    # overlay them so that the local-sensitivity grid (and the nominal
    # lookup for the picker UI) is centred on the actual optimum
    # rather than wherever the user typed in the sidebar.
    _kind = fa.get("kind", "eq")
    _baseline_logK   = dict(fa.get("start_logK", fa.get("start_logk", {})))
    _baseline_params = dict(fa.get("params") or {})
    if _baseline_params:
        _baseline_params["conc0"]       = dict(_baseline_params.get("conc0",       {}))
        _baseline_params["titrant_mMs"] = dict(_baseline_params.get("titrant_mMs", {}))
    _baseline_parsed = dict(fa.get("parsed_fit") or {})
    if _baseline_parsed:
        _baseline_parsed["concentrations"] = dict(_baseline_parsed.get("concentrations", {}))
        _baseline_parsed["volumes"]        = dict(_baseline_parsed.get("volumes",        {}))

    if fit_stats:
        # Fitted log K / log k values — directly overlay (same log10 space)
        for _k, _v in (fit_stats.get("param_values") or {}).items():
            if _k in _baseline_logK:
                try: _baseline_logK[_k] = float(_v)
                except Exception: pass
        # Fitted concentrations (stored as linear mM in fit_stats)
        for _k, _v in (fit_stats.get("fitted_concs") or {}).items():
            try: _vf = float(_v)
            except Exception: continue
            # Equilibrium-mode concs live in params["conc0"]
            if _baseline_params and _k in _baseline_params.get("conc0", {}):
                _baseline_params["conc0"][_k] = _vf
            # Kinetics-mode concs live in parsed["concentrations"];
            # fit_stats uses the ROOT name ("G") while parsed uses the
            # "$concentrations" form ("G0").  Try both lookups.
            elif _baseline_parsed and _baseline_parsed.get("concentrations"):
                _d = _baseline_parsed["concentrations"]
                if _k in _d:
                    _d[_k] = _vf
                elif (_k + "0") in _d:
                    _d[_k + "0"] = _vf
        # Fitted titrant stocks (linear mM)
        for _k, _v in (fit_stats.get("fitted_titrants") or {}).items():
            try: _vf = float(_v)
            except Exception: continue
            if _baseline_params and _k in _baseline_params.get("titrant_mMs", {}):
                _baseline_params["titrant_mMs"][_k] = _vf

    # ── Patch fa with the corrected baseline for the grid sweep ──────
    # _compute_rmse_at reads ``start_logK`` / ``start_logk`` / ``params``
    # / ``parsed_fit`` from fa to build its pinned configuration.  We
    # make a SHALLOW-but-segregated copy with the corrected baselines
    # so the original fa (used by other diagnostics) isn't mutated.
    _fa_corrected = dict(fa)
    _fa_corrected["start_logK"] = _baseline_logK
    _fa_corrected["start_logk"] = _baseline_logK  # alias used by some paths
    if _baseline_params:
        _fa_corrected["params"] = _baseline_params
    if _baseline_parsed and _kind == "kin":
        _fa_corrected["parsed_fit"] = _baseline_parsed

    # ── Gather nominal (log10) values for the SELECTED variables ─────
    nominal = {}
    for v in sel:
        if v in _baseline_logK:
            nominal[v] = float(_baseline_logK[v])
            continue
        if _kind == "eq":
            _conc0 = _baseline_params.get("conc0",      {}) if _baseline_params else {}
            _titm  = _baseline_params.get("titrant_mMs", {}) if _baseline_params else {}
            if v in _conc0:
                nominal[v] = float(np.log10(max(float(_conc0[v]),  1e-12)))
            elif v in _titm:
                nominal[v] = float(np.log10(max(float(_titm[v]),   1e-12)))
            elif v == "V0_mL" and _baseline_params and "V0_mL" in _baseline_params:
                nominal[v] = float(np.log10(max(float(_baseline_params["V0_mL"]), 1e-12)))
        else:
            _concs  = _baseline_parsed.get("concentrations", {}) if _baseline_parsed else {}
            _vols   = _baseline_parsed.get("volumes",        {}) if _baseline_parsed else {}
            if v in _concs:
                nominal[v] = float(np.log10(max(float(_concs[v]), 1e-12)))
            elif v in _vols:
                nominal[v] = float(np.log10(max(float(_vols[v]),  1e-12)))

    # Any selected variable we couldn't anchor → drop it with a warning
    missing = [v for v in sel if v not in nominal]
    sel = [v for v in sel if v in nominal]
    Nsel = len(sel)
    if Nsel == 0:
        return {"selected_vars": [], "wall_seconds": 0.0,
                "best_message": f"None of the selected variables "
                                f"could be resolved: {missing}"}

    # ── Build the 3^N grid ────────────────────────────────────────────
    # Each variable takes 3 values: nominal-step, nominal, nominal+step
    # Tasks indexed by a tuple of {0, 1, 2}^N (0 = -δ, 1 = 0, 2 = +δ).
    import itertools
    levels = [-step, 0.0, +step]
    tasks = list(itertools.product([0, 1, 2], repeat=Nsel))
    n_grid = len(tasks)

    def _build_param_values(task_tuple):
        pv = {}
        for i, lvl in enumerate(task_tuple):
            pv[sel[i]] = nominal[sel[i]] + levels[lvl]
        return pv

    # Evaluate RMSE at each grid point (using the corrected baseline fa)
    def _eval(task_tuple):
        return _compute_rmse_at(_fa_corrected, _build_param_values(task_tuple))

    rmse_grid: dict = {}  # task_tuple → rmse or NaN
    n_total = max(1, n_grid)

    if n_jobs == 1:
        for idx, t in enumerate(tasks):
            if progress_callback is not None:
                try: progress_callback(idx, n_total)
                except Exception: pass
            r = _eval(t)
            rmse_grid[t] = r if (r is not None and np.isfinite(r)) else float("nan")
    else:
        try:
            from joblib import Parallel, delayed
        except Exception:
            return compute_local_sensitivity(fa, sel, step=step, n_jobs=1,
                                              sigma_marg=sigma_marg,
                                              fit_stats=fit_stats,
                                              progress_callback=progress_callback)
        try:
            gen = Parallel(n_jobs=n_jobs, prefer="processes",
                            return_as="generator")(
                delayed(_eval)(t) for t in tasks)
            done = 0
            for t, r in zip(tasks, gen):
                rmse_grid[t] = r if (r is not None and np.isfinite(r)) else float("nan")
                done += 1
                if progress_callback is not None:
                    try: progress_callback(done, n_total)
                    except Exception: pass
        except TypeError:
            for t, r in zip(tasks, Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_eval)(t) for t in tasks)):
                rmse_grid[t] = r if (r is not None and np.isfinite(r)) else float("nan")

    # ── Baseline (center grid point) ──────────────────────────────────
    center = tuple([1] * Nsel)
    rmse_opt = rmse_grid.get(center, float("nan"))
    if not np.isfinite(rmse_opt) or rmse_opt <= 0:
        # Fall back to nanmin if center failed
        finite_vals = [v for v in rmse_grid.values() if np.isfinite(v) and v > 0]
        if not finite_vals:
            return {"selected_vars": sel, "wall_seconds": _time.perf_counter() - t0,
                    "best_message": "All grid evaluations returned NaN/0.",
                    "n_grid": n_grid, "n_success": 0}
        rmse_opt = float(min(finite_vals))

    # Prefer fit_stats["n_points"] (the canonical post-fit value),
    # fall back to whatever fa carries, finally to 1 to avoid /0.
    n_data = 0
    if fa.get("kind") == "global":
        # all tabs together: the sum of the per-tab point counts of the fit
        try:
            n_data = int(sum(int((st_ or {}).get("n_points") or 0)
                             for st_ in (fa.get("baseline_stats") or {}).values()))
        except Exception:
            n_data = 0
    if n_data <= 0 and fit_stats:
        try: n_data = int(fit_stats.get("n_points") or 0)
        except Exception: n_data = 0
    if n_data <= 0:
        n_data = int(fa.get("n_data") or fa.get("n_points") or 0)
    if n_data <= 0:
        n_data = 1

    # ── Compute f for each variable via paired differences ────────────
    # Also retain the full per-diff record so the UI can show each of
    # the 2·3^(N-1) paired diffs individually — useful for diagnosing
    # whether a small mean f comes from clean tiny diffs (truly
    # insensitive parameter) or from a sign-flipping mix that averages
    # toward zero (non-quadratic surface, asymmetric response).
    f_vals: dict = {}
    diff_records: dict = {}   # {var: [{"sign": -1|+1, "others": dict, "diff": float}, …]}
    for ii, var in enumerate(sel):
        diffs = []
        per_var_records = []
        # Build list of "other" indices
        others = [j for j in range(Nsel) if j != ii]
        if not others:
            other_combos = [()]
        else:
            other_combos = list(itertools.product([0, 1, 2], repeat=len(others)))
        for sign_idx in (0, 2):  # -δ and +δ
            sign_val = -1 if sign_idx == 0 else +1
            for oc in other_combos:
                # Build task tuples for "perturbed" and "unperturbed"
                t_p = list((1,) * Nsel)  # all at center
                t_u = list((1,) * Nsel)
                t_p[ii] = sign_idx
                # Fill in "others" indices
                for k, j in enumerate(others):
                    t_p[j] = oc[k]
                    t_u[j] = oc[k]
                r_p = rmse_grid.get(tuple(t_p), float("nan"))
                r_u = rmse_grid.get(tuple(t_u), float("nan"))
                if np.isfinite(r_p) and np.isfinite(r_u):
                    _d = (r_p - r_u) / rmse_opt
                    diffs.append(_d)
                    # Label "other" levels as -, 0, +
                    _level_to_sym = {0: "-", 1: "0", 2: "+"}
                    _other_label = {sel[j]: _level_to_sym[oc[k]]
                                    for k, j in enumerate(others)}
                    per_var_records.append({
                        "sign":      sign_val,
                        "others":    _other_label,
                        "diff":      float(_d),
                        # Absolute RMSE values at the two grid points
                        # being differenced — let the user see whether
                        # the numerical change in RMSE is meaningful or
                        # at the noise floor of the speciation solver.
                        "rmse_pert": float(r_p),
                        "rmse_unp":  float(r_u),
                    })
        if diffs:
            f_vals[var] = float(np.mean(diffs))
        else:
            f_vals[var] = float("nan")
        diff_records[var] = per_var_records

    # ── Derive σ_cond, look up σ_marg, coupling ratio ────────────────
    sigma_cond: dict = {}
    sigma_m:    dict = {}
    coupling:   dict = {}
    sigma_marg = sigma_marg or {}
    for var in sel:
        f_v = f_vals.get(var, float("nan"))
        if np.isfinite(f_v) and f_v > 0 and n_data > 0:
            sigma_cond[var] = float(step / np.sqrt(2.0 * n_data * f_v))
        else:
            sigma_cond[var] = float("nan")
        # σ_marg from Hessian (caller supplies)
        if var in sigma_marg and np.isfinite(sigma_marg[var]):
            sigma_m[var] = float(sigma_marg[var])
            if np.isfinite(sigma_cond[var]) and sigma_cond[var] > 0:
                coupling[var] = float(sigma_m[var] / sigma_cond[var])
            else:
                coupling[var] = float("nan")
        else:
            sigma_m[var]   = float("nan")
            coupling[var]  = float("nan")

    n_success = int(sum(1 for v in rmse_grid.values() if np.isfinite(v)))
    return {
        "selected_vars": sel,
        "rmse_opt": float(rmse_opt),
        "n_data":   int(n_data),
        "nominal":  nominal,
        "f":        f_vals,
        "diff_records": diff_records,  # per-variable list of all paired diffs
        "sigma_cond": sigma_cond,
        "sigma_marg": sigma_m,
        "coupling":   coupling,
        "n_grid":   n_grid,
        "n_success": n_success,
        "n_failed":  n_grid - n_success,
        "wall_seconds": _time.perf_counter() - t0,
        "step":     float(step),
    }


# ════════════════════════════════════════════════════════════════════
#  Spectral rank analysis  (EFA, scree+IND, TFA)
# ════════════════════════════════════════════════════════════════════
#
#  Where the parameter-identifiability tools above (correlation heatmap,
#  sloppy spectrum, 1D/2D RMSE profiles, Masson/β test) ask whether the
#  *fitted parameters* are well-determined by the data, the diagnostics
#  in this section ask the complementary question: whether the *data
#  itself* supports the rank (= number of distinguishable chemical
#  species) implied by the chemical model.
#
#  This is the standard chemometrics toolkit for multivariate
#  spectroscopic data — Evolving Factor Analysis (Maeder & Zuberbühler
#  1986), Malinowski's IE / RE / IND functions (Factor Analysis in
#  Chemistry, 2nd ed. 1991), and Target Factor Analysis (Malinowski
#  1978).  They apply to any fit mode whose data matrix D = C·E^T has
#  a bilinear mixing law — that is, UV-Vis (Beer-Lambert in absorbance),
#  fast-exchange NMR shift (population-weighted shifts), and NMR
#  integration (number-of-nuclei-weighted concentrations).  They do
#  not apply to pure-concentration fits, where there are typically too
#  few measured channels for a rank analysis to be informative.
#
#  Conventions:
#     D shape   : (n_obs, n_ch) where n_obs = titration / time points,
#                 n_ch = wavelengths or signal channels.
#     c         : min(n_obs, n_ch) — the maximum possible rank.
#     r         : max(n_obs, n_ch) — used in Malinowski's RE formula.
#
#  The diagnostic is fundamentally a sanity check on the chemical
#  model: if the user fits a 1:2 binding model (3 species) but EFA
#  shows the data supports rank 2, something is wrong — either the
#  model is over-parameterised or one of the species is spectrally
#  indistinguishable from a linear combination of the others.
# ════════════════════════════════════════════════════════════════════

def compute_efa(D, direction: str = "forward",
                max_factors: Optional[int] = None) -> dict:
    """
    Evolving Factor Analysis on the data matrix D.

    For each k = 1, ..., n_obs, computes the SVD of the leading k rows
    (direction='forward') or trailing k rows (direction='backward') of
    D, and records the resulting singular values.  Plotting σ vs k
    reveals how each chemical species' contribution rises above the
    noise floor as observations are added (forward) or removed
    (backward); the combined view defines the concentration window of
    each species.

    Parameters
    ----------
    D            : (n_obs, n_ch) array-like
    direction    : "forward" or "backward"
    max_factors  : truncate to the first ``max_factors`` singular
                   values per SVD; default min(n_obs, n_ch).

    Returns
    -------
    dict with keys:
        row_indices  : (n_obs,) ints, the k value for each row of sigma
        sigma        : (n_obs, p) matrix; sigma[i, j] is the j-th
                       singular value of the SVD of D[:i+1] (forward)
                       or D[n_obs-i-1:] (backward).
        direction    : input direction
        n_obs, n_ch  : data shape
        max_factors  : number of factor curves retained
    """
    import numpy as _np
    D = _np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError(f"D must be 2-D, got shape {D.shape}")
    n_obs, n_ch = D.shape
    p_max = min(n_obs, n_ch)
    p = p_max if max_factors is None else min(max_factors, p_max)
    sigma_grid = _np.full((n_obs, p), _np.nan, dtype=float)

    if direction == "forward":
        for k in range(1, n_obs + 1):
            s = _np.linalg.svd(D[:k, :], compute_uv=False)
            sigma_grid[k - 1, : min(len(s), p)] = s[: min(len(s), p)]
        row_indices = _np.arange(1, n_obs + 1)
    elif direction == "backward":
        for k in range(1, n_obs + 1):
            s = _np.linalg.svd(D[n_obs - k:, :], compute_uv=False)
            sigma_grid[k - 1, : min(len(s), p)] = s[: min(len(s), p)]
        row_indices = _np.arange(n_obs - 1, -1, -1)
    else:
        raise ValueError(f"direction must be 'forward' or 'backward', "
                         f"got {direction!r}")

    return dict(row_indices=row_indices, sigma=sigma_grid,
                direction=direction, n_obs=n_obs, n_ch=n_ch,
                max_factors=p)


def compute_scree_metrics(D) -> dict:
    """
    Full-matrix SVD plus Malinowski's rank-determination statistics.

    For a data matrix D of shape (n_obs, n_ch), let c = min(n_obs, n_ch)
    (maximum possible rank) and r = max(n_obs, n_ch).  Then for each
    candidate rank k = 1, ..., c-1:

        RE(k)  = sqrt( sum_{j>k} σ_j^2 / (r · (c - k)) )
        IE(k)  = RE(k) · sqrt(k / c)
        IND(k) = RE(k) / (c - k)^2

    The recommended rank is the integer k that *minimises* IND.  In
    chemometrics practice IND is read together with the scree plot
    itself: a sharp drop in IND that flattens to a plateau is the
    cleanest signature of the true rank.

    Returns
    -------
    dict with keys:
        sigma           : full SVD singular values, descending
        re, ie, ind     : (c,) arrays; entries at k=0 and k>=c are NaN
        ind_argmin_rank : recommended rank (the k minimising IND)
        snr             : (c,) signal-to-noise ratios, defined as
                          σ_k / σ_noise where σ_noise is estimated
                          from the geometric mean of trailing-half
                          singular values
        noise_floor     : σ_noise estimate
        n_obs, n_ch     : data shape
    """
    import numpy as _np
    D = _np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError(f"D must be 2-D, got shape {D.shape}")
    n_obs, n_ch = D.shape
    c = min(n_obs, n_ch)
    r = max(n_obs, n_ch)
    p_max = c

    s = _np.linalg.svd(D, compute_uv=False)

    re  = _np.full(p_max, _np.nan)
    ie  = _np.full(p_max, _np.nan)
    ind = _np.full(p_max, _np.nan)
    for k in range(1, p_max):
        residual_sumsq = float(_np.sum(s[k:] ** 2))
        if (c - k) > 0:
            re[k]  = _np.sqrt(residual_sumsq / (r * (c - k)))
            ie[k]  = re[k] * _np.sqrt(k / c)
            ind[k] = re[k] / (c - k) ** 2

    valid = ~_np.isnan(ind)
    ind_argmin_rank = int(_np.nanargmin(ind)) if valid.any() else 1

    tail_start = max(p_max // 2, 1)
    tail_vals = s[tail_start:][s[tail_start:] > 0]
    if len(tail_vals) > 0:
        noise_floor = float(_np.exp(_np.mean(_np.log(tail_vals))))
    else:
        noise_floor = 0.0
    snr = s / max(noise_floor, 1e-30)

    return dict(sigma=s, re=re, ie=ie, ind=ind,
                ind_argmin_rank=ind_argmin_rank,
                snr=snr, noise_floor=noise_floor,
                n_obs=n_obs, n_ch=n_ch)


def compute_tfa(D, targets, rank: int, axis: str = "column") -> dict:
    """
    Target Factor Analysis: project each target vector onto the
    factor space of D and report the projection residual.

    Given the SVD D = U Σ V^T, the rank-r column-space projector is
    P_col = V_r V_r^T (where V_r = V[:, :rank]).  A target vector t
    that is consistent with the data's factor space at this rank
    satisfies P_col t ≈ t — its projection residual ||t − P_col t|| /
    ||t|| is small.  A target that is incompatible (a different
    chemical species' spectrum, or a fitted pure spectrum that doesn't
    correspond to a real component) leaves a large residual.

    The fitted pure spectra ε from a constrained-MCR fit (e.g.
    equilibrist_fit_spectra.py) are natural targets: a small residual
    confirms that each fitted ε is reproducible from the data's
    leading singular vectors, regardless of which chemical model
    produced it.

    Parameters
    ----------
    D       : (n_obs, n_ch) data matrix
    targets : (n_ch, K) for axis="column" (each column is a putative
              pure spectrum) or (n_obs, K) for axis="row" (each column
              is a putative concentration profile).  Single 1-D
              vectors are accepted and broadcast as one target.
    rank    : assumed model rank (e.g. ind_argmin_rank from scree)
    axis    : "column" (project onto column space ~ pure spectra) or
              "row"    (project onto row space    ~ conc profiles)

    Returns
    -------
    dict with keys:
        residuals  : (K,) relative projection residuals
        projected  : projected targets, same shape as ``targets``
        rank_used  : input rank
        axis       : input axis
        n_targets  : K
    """
    import numpy as _np
    D = _np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError(f"D must be 2-D, got shape {D.shape}")
    targets = _np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]

    U, s, Vt = _np.linalg.svd(D, full_matrices=False)

    if axis == "column":
        V = Vt.T
        V_r = V[:, :rank]
        if targets.shape[0] != V.shape[0]:
            raise ValueError(
                f"target length {targets.shape[0]} != D.shape[1]={D.shape[1]} "
                f"for axis='column' projection")
        projected = V_r @ (V_r.T @ targets)
    elif axis == "row":
        U_r = U[:, :rank]
        if targets.shape[0] != U.shape[0]:
            raise ValueError(
                f"target length {targets.shape[0]} != D.shape[0]={D.shape[0]} "
                f"for axis='row' projection")
        projected = U_r @ (U_r.T @ targets)
    else:
        raise ValueError(f"axis must be 'column' or 'row', got {axis!r}")

    residuals = _np.zeros(targets.shape[1])
    for j in range(targets.shape[1]):
        t = targets[:, j]
        t_proj = projected[:, j]
        denom = _np.linalg.norm(t)
        if denom > 0:
            residuals[j] = _np.linalg.norm(t - t_proj) / denom

    return dict(residuals=residuals, projected=projected,
                rank_used=rank, axis=axis, n_targets=int(targets.shape[1]))


def compute_rank_analysis(stats: dict) -> Optional[dict]:
    """
    Run the full EFA / scree / TFA pipeline on a fit's data matrix.

    Reads from ``stats`` the following slots that the fit module must
    populate:

        data_matrix             : (n_obs, n_ch) observed D
        data_matrix_obs_labels  : (n_obs,) labels for the rows
        data_matrix_ch_labels   : (n_ch,) labels for the columns
        data_matrix_kind        : string descriptor (e.g. 'absorbance')

    Optional slots used as TFA targets:

        E_final                 : (n_ch, n_species) fitted pure spectra
        species_names           : (n_species,) names for E columns

    Returns ``None`` if the data matrix is missing.  Otherwise returns
    a dict containing the EFA forward, EFA backward, scree, and TFA
    sub-dicts, plus the model rank inferred from ``param_cov_names``
    (a sanity-check comparison to ind_argmin_rank).
    """
    import numpy as _np
    D = stats.get("data_matrix")
    if D is None:
        return None
    D = _np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] < 2 or D.shape[1] < 2:
        return None

    # Limit max_factors to keep the EFA plot readable
    max_factors = min(8, min(D.shape))

    efa_f = compute_efa(D, direction="forward",  max_factors=max_factors)
    efa_b = compute_efa(D, direction="backward", max_factors=max_factors)
    scree = compute_scree_metrics(D)

    # TFA on fitted pure spectra if available
    tfa = None
    E_final = stats.get("E_final")
    if E_final is not None:
        E = _np.asarray(E_final, dtype=float)
        # E has shape (n_ch, n_species) by Equilibrist convention; some
        # callers store it transposed, so we sanity-check.
        if E.shape[0] != D.shape[1] and E.shape[1] == D.shape[1]:
            E = E.T
        if E.shape[0] == D.shape[1]:
            try:
                tfa = compute_tfa(D, targets=E,
                                  rank=scree["ind_argmin_rank"],
                                  axis="column")
                tfa["species_labels"] = stats.get(
                    "species_names",
                    [f"sp_{j+1}" for j in range(E.shape[1])])
            except Exception:
                tfa = None

    # Model rank from the chemical model: number of free parameters
    # is not the same as model rank in general, so we record it as
    # informational only.  The cleanest "model rank" reading is the
    # number of distinct chemical species present in the fit, which
    # the caller may pass as stats["n_species_fit"].
    model_rank = stats.get("n_species_fit")

    return dict(
        efa_forward=efa_f,
        efa_backward=efa_b,
        scree=scree,
        tfa=tfa,
        model_rank=model_rank,
        data_matrix_kind=stats.get("data_matrix_kind", "data"),
        obs_labels=stats.get("data_matrix_obs_labels"),
        ch_labels=stats.get("data_matrix_ch_labels"),
    )


def make_rank_analysis_figure(stats: dict, *,
                              figsize=(13.0, 4.5),
                              title: Optional[str] = None) -> Optional[object]:
    """
    Three-panel figure for the EFA / scree / TFA rank analysis.

    Left panel:   forward EFA (solid) overlaid with backward EFA (dashed),
                  log y-axis, noise floor as horizontal reference line.
                  The "appearance" and "disappearance" rows of each
                  species are visible as the points where each σ rises
                  above the noise floor.

    Middle panel: full-matrix scree (σ_k vs k) with Malinowski IND on a
                  secondary y-axis, plus a vertical dashed line at
                  the IND argmin.  The shape of the IND curve — a
                  sharp drop followed by a plateau — is the canonical
                  chemometrics rank-determination signal.

    Right panel:  Target Factor Analysis residuals, one bar per fitted
                  pure spectrum.  Bars below ~5% confirm that the
                  fitted ε is reproducible from the data's leading
                  singular vectors; bars above ~20% flag a fitted
                  spectrum that the data does not actually support.
                  Omitted when E_final is unavailable.

    Returns ``None`` if no rank analysis is possible (e.g. missing
    data matrix); otherwise a matplotlib Figure.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    result = compute_rank_analysis(stats)
    if result is None:
        return None

    efa_f = result["efa_forward"]
    efa_b = result["efa_backward"]
    scree = result["scree"]
    tfa   = result["tfa"]
    kind  = result.get("data_matrix_kind", "data")

    has_tfa = tfa is not None and len(tfa.get("residuals", [])) > 0
    n_panels = 3 if has_tfa else 2
    if not has_tfa:
        figsize = (figsize[0] * (2/3), figsize[1])

    fig, axes = _plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 2:
        ax_efa, ax_scree = axes
        ax_tfa = None
    else:
        ax_efa, ax_scree, ax_tfa = axes

    # ── Panel A: EFA forward + backward ─────────────────────
    p = efa_f["max_factors"]
    cmap_fwd = _plt.cm.viridis
    cmap_bwd = _plt.cm.plasma
    noise = scree["noise_floor"]

    for j in range(p):
        c_fwd = cmap_fwd(j / max(p - 1, 1))
        ax_efa.plot(efa_f["row_indices"], efa_f["sigma"][:, j],
                    "-", color=c_fwd, lw=1.5,
                    label=f"σ$_{j+1}$" if j < 5 else None)
    for j in range(p):
        c_bwd = cmap_bwd(j / max(p - 1, 1))
        # Backward EFA plotted vs "starting row index" so x axis is shared
        starting_row = efa_b["n_obs"] - _np.arange(1, efa_b["n_obs"] + 1) + 1
        ax_efa.plot(starting_row, efa_b["sigma"][:, j],
                    "--", color=c_bwd, lw=1.0, alpha=0.7)

    ax_efa.axhline(noise, color="0.5", ls=":", lw=1.0)
    ax_efa.text(0.99, noise, "  noise floor", ha="right", va="bottom",
                fontsize=8, color="0.4", transform=ax_efa.get_yaxis_transform())
    ax_efa.set_yscale("log")
    ax_efa.set_xlabel("Row index  (forward → / ← backward)")
    ax_efa.set_ylabel("Singular value")
    ax_efa.set_title("EFA  (forward: solid, backward: dashed)")
    ax_efa.grid(True, alpha=0.3)
    if p <= 5:
        ax_efa.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # ── Panel B: scree + IND ──────────────────────────────
    s = scree["sigma"]
    ks = _np.arange(1, len(s) + 1)
    ax_scree.semilogy(ks, s, "o-", color="steelblue", lw=1.5, markersize=5,
                      label="σ$_k$")
    ax_scree.axhline(noise, color="0.5", ls=":", lw=1.0)
    ax_scree.set_xlabel("Singular-value index k")
    ax_scree.set_ylabel("σ$_k$", color="steelblue")
    ax_scree.tick_params(axis="y", labelcolor="steelblue")
    ax_scree.grid(True, alpha=0.3)
    ax_scree.set_xlim(0, min(20, len(s)) + 0.5)

    ax_ind = ax_scree.twinx()
    valid = ~_np.isnan(scree["ind"])
    ax_ind.semilogy(ks[valid], scree["ind"][valid], "s-",
                    color="crimson", lw=1.0, markersize=4, label="IND")
    ax_ind.set_ylabel("Malinowski IND(k)", color="crimson")
    ax_ind.tick_params(axis="y", labelcolor="crimson")
    k_rec = scree["ind_argmin_rank"]
    ax_ind.axvline(k_rec, color="crimson", ls="--", lw=1.0, alpha=0.6)
    ax_scree.set_title(f"Scree + IND  (argmin → rank ≈ {k_rec})")

    # ── Panel C: TFA per fitted spectrum ─────────────────
    if has_tfa:
        residuals = _np.asarray(tfa["residuals"])
        labels = tfa.get("species_labels",
                         [f"sp_{j+1}" for j in range(len(residuals))])
        # Truncate long labels and trim trailing whitespace
        labels_disp = [str(l).strip()[:14] for l in labels]
        # Colour tag bars by threshold
        colors = []
        for r in residuals:
            if r < 0.05:   colors.append("seagreen")
            elif r < 0.20: colors.append("goldenrod")
            else:          colors.append("crimson")
        bars = ax_tfa.bar(range(len(residuals)), residuals, color=colors,
                          edgecolor="0.3", lw=0.5)
        # Annotate threshold lines
        ax_tfa.axhline(0.05, color="seagreen", ls=":", lw=1.0, alpha=0.6)
        ax_tfa.axhline(0.20, color="crimson",  ls=":", lw=1.0, alpha=0.6)
        ax_tfa.set_xticks(range(len(residuals)))
        ax_tfa.set_xticklabels(labels_disp, rotation=30, ha="right",
                               fontsize=9)
        ax_tfa.set_ylabel("Relative TFA residual")
        ax_tfa.set_title(f"TFA on fitted pure spectra  (rank = {tfa['rank_used']})")
        ax_tfa.set_ylim(0, max(0.3, _np.max(residuals) * 1.2))
        ax_tfa.grid(True, alpha=0.3, axis="y")
        # Annotate residual values on bars
        for bar, val in zip(bars, residuals):
            ax_tfa.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val*100:.2f}%",
                        ha="center", va="bottom", fontsize=8)

    if title is None:
        title = f"Rank analysis of {kind} data matrix " \
                f"({scree['n_obs']} × {scree['n_ch']})"
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


def render_rank_analysis_panel(stats: dict, *, key: str = "rank") -> None:
    """
    Streamlit wrapper: render the rank-analysis figure plus a banner
    comparing the IND-recommended rank to the chemical model rank, an
    eigenvalue table, and a download button.

    No-op if no data matrix is present in ``stats``.
    """
    import io as _io
    from datetime import datetime as _dt
    try:
        import streamlit as st
    except Exception:
        return

    result = compute_rank_analysis(stats)
    if result is None:
        st.caption("Rank analysis is unavailable for this fit "
                   "(no multi-channel data matrix).")
        return

    scree = result["scree"]
    tfa   = result["tfa"]
    k_rec = scree["ind_argmin_rank"]
    model_rank = result.get("model_rank")

    # ── Banner: rank comparison ──────────────────────────
    if model_rank is not None:
        if model_rank == k_rec:
            st.success(
                f"🟢 **Model rank matches data rank.**  "
                f"Malinowski IND recommends rank {k_rec}; the chemical "
                f"model fits {model_rank} species.  The data supports "
                f"the model dimensionality.")
        elif model_rank < k_rec:
            st.warning(
                f"🟡 **Model may be under-parameterised.**  "
                f"IND recommends rank {k_rec} but the model fits only "
                f"{model_rank} species.  The data contains more "
                f"distinguishable signal than the model represents; "
                f"consider adding a species.")
        else:
            st.warning(
                f"🟡 **Model may be over-parameterised.**  "
                f"IND recommends rank {k_rec} but the model fits "
                f"{model_rank} species.  The extra species may be "
                f"spectrally indistinguishable from a combination of "
                f"the others, or may be absent at the concentrations "
                f"used.")
    else:
        st.info(f"Malinowski IND recommends rank {k_rec} for the data matrix.")

    fig = make_rank_analysis_figure(stats)
    if fig is None:
        return
    st.pyplot(fig, clear_figure=False)

    # ── Table: scree singular values + SNR ──────────────
    import pandas as _pd
    s   = scree["sigma"]
    snr = scree["snr"]
    rows = []
    n_show = min(10, len(s))
    for i in range(n_show):
        tag = ("🟢 above noise" if snr[i] > 10 else
               "🟡 borderline"  if snr[i] > 3  else
               "🔴 at noise floor")
        rows.append(dict(
            k=i + 1,
            sigma=f"{s[i]:.3e}",
            SNR=f"{snr[i]:.1f}",
            tag=tag,
        ))
    df = _pd.DataFrame(rows)
    st.caption(f"Top {n_show} singular values  (noise floor "
               f"σ ≈ {scree['noise_floor']:.2e}; SNR = σ / noise)")
    st.dataframe(df, hide_index=True, use_container_width=True)

    if tfa is not None:
        tfa_rows = []
        for j, (lbl, r) in enumerate(zip(
                tfa.get("species_labels", []),
                tfa["residuals"])):
            tag = ("🟢 consistent"   if r < 0.05 else
                   "🟡 marginal"     if r < 0.20 else
                   "🔴 not supported")
            tfa_rows.append(dict(
                species=str(lbl).strip(),
                residual=f"{r*100:.2f}%",
                tag=tag,
            ))
        st.caption(f"Target Factor Analysis of fitted pure spectra "
                   f"(projected onto rank-{tfa['rank_used']} factor space)")
        st.dataframe(_pd.DataFrame(tfa_rows), hide_index=True,
                     use_container_width=True)

    # PNG download
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("Download rank-analysis PNG",
                       data=buf.getvalue(),
                       file_name=f"Equilibrist_rank_{ts}.png",
                       mime="image/png",
                       key=f"_rank_dl_{key}")


# ════════════════════════════════════════════════════════════════════════
# Profile trace: how the other parameters respond when one is scanned
# ════════════════════════════════════════════════════════════════════════

def compute_profile_trace(fit_args: dict, x_name: str, x_center: float,
                          *, span: float = 1.0, n_pts: int = 11,
                          fit_names: Optional[list] = None,
                          n_jobs: int = 1, progress_callback=None) -> dict:
    """Walk ``x_name`` from ``x_center - span`` to ``x_center + span`` and, at
    every grid point, refit ONLY the parameters in ``fit_names`` (everything
    else stays at its value in ``fit_args["start_logK"]``).  Records the
    fitted value of each of them and the RMSE.

    ``x_name`` may be a constant that was held fixed in the fit.  Values are
    log10 throughout (concentrations / titrants as log10 mM).

    Returns ``{x_name, x_center, x_grid, fit_names, values: {name: array},
    rmse: array, rmse_min, n_failed, n_total, error_samples}``.
    """
    n_pts = max(3, int(n_pts))
    if n_pts % 2 == 0:
        n_pts += 1
    x_grid = np.linspace(float(x_center) - span, float(x_center) + span, n_pts)
    fit_names = [n for n in (fit_names or []) if n != x_name]

    fa = dict(fit_args)
    _all_conc = list(fa.get("fit_conc_keys") or [])
    _all_tit = list(fa.get("fit_titrant_keys") or [])
    fa["fit_conc_keys"] = [n for n in fit_names if n in _all_conc]
    fa["fit_titrant_keys"] = [n for n in fit_names if n in _all_tit]
    fa["fit_keys"] = [n for n in fit_names if n not in fa["fit_conc_keys"]
                      and n not in fa["fit_titrant_keys"]]
    # the scanned parameter is pinned by _refit_with_pinned; when it is a K it
    # must be listed for the pin machinery to recognise it as one
    if x_name not in _all_conc and x_name not in _all_tit and x_name not in fa["fit_keys"]:
        fa["fit_keys"] = fa["fit_keys"] + [x_name]

    def _one(idx_val):
        _i, _v = idx_val
        try:
            _s = _refit_with_pinned(fa, pinned_key=x_name, pinned_val=float(_v))
        except Exception as _e:                              # pragma: no cover
            return _i, float("nan"), {}, f"{type(_e).__name__}: {_e}"
        if not isinstance(_s, dict):
            return _i, float("nan"), {}, "refit returned nothing"
        _r = _s.get("rmse")
        _vals = {}
        for n in fit_names:
            v = None
            if n in (_s.get("param_values") or {}):
                v = _s["param_values"][n]
            elif n in (_s.get("fitted_concs") or {}):
                try: v = float(np.log10(float(_s["fitted_concs"][n])))
                except Exception: v = None
            elif n in (_s.get("fitted_titrants") or {}):
                try: v = float(np.log10(float(_s["fitted_titrants"][n])))
                except Exception: v = None
            _vals[n] = float(v) if v is not None and np.isfinite(v) else float("nan")
        return _i, (float(_r) if _r is not None and np.isfinite(_r) else float("nan")), _vals, None

    tasks = list(enumerate(x_grid))
    total = len(tasks)
    rmse = np.full(total, np.nan)
    values = {n: np.full(total, np.nan) for n in fit_names}
    errors: list = []

    def _store(res):
        _i, _r, _vals, _err = res
        rmse[_i] = _r
        for n, v in _vals.items():
            values[n][_i] = v
        if _err and len(errors) < 5:
            errors.append(_err)

    if n_jobs == 1:
        for _k, _t in enumerate(tasks):
            if progress_callback is not None:
                try: progress_callback(_k / total, f"Profile trace: {_k + 1}/{total}")
                except Exception: pass
            _store(_one(_t))
    else:
        try:
            from joblib import Parallel, delayed
        except Exception:
            return compute_profile_trace(fit_args, x_name, x_center, span=span, n_pts=n_pts,
                                         fit_names=fit_names, n_jobs=1,
                                         progress_callback=progress_callback)
        try:
            gen = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
                delayed(_one)(_t) for _t in tasks)
            _done = 0
            for res in gen:
                _store(res); _done += 1
                if progress_callback is not None:
                    try: progress_callback(_done / total, f"Profile trace: {_done}/{total}")
                    except Exception: pass
        except TypeError:
            for res in Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
                    delayed(_one)(_t) for _t in tasks):
                _store(res)
    if progress_callback is not None:
        try: progress_callback(1.0, "Profile trace complete.")
        except Exception: pass
    return {
        "x_name": x_name, "x_center": float(x_center), "x_grid": x_grid,
        "fit_names": list(fit_names), "values": values, "rmse": rmse,
        "rmse_min": float(np.nanmin(rmse)) if np.any(np.isfinite(rmse)) else np.nan,
        "n_failed": int(np.sum(~np.isfinite(rmse))), "n_total": int(total),
        "error_samples": errors,
    }


def make_profile_trace_figure(res: dict, title: Optional[str] = None,
                              show_legend: bool = True):
    """Left axis: fitted log K of every refit parameter vs the scanned one.
    Right axis (log ticks): RMSE / RMSE_min along the scan."""
    import matplotlib
    import matplotlib.pyplot as plt
    x = np.asarray(res["x_grid"], float)
    names = list(res.get("fit_names") or [])
    rmse = np.asarray(res["rmse"], float)
    rmin = float(np.nanmin(rmse)) if np.any(np.isfinite(rmse)) else np.nan
    _palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
                "#9467BD", "#8C564B", "#E377C2", "#17BECF"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    handles = []
    for i, n in enumerate(names):
        y = np.asarray(res["values"].get(n, []), float)
        if y.size != x.size:
            continue
        h, = ax.plot(x, y, "-o", ms=3.2, lw=1.4, color=_palette[i % len(_palette)],
                     label=f"log {n}")
        handles.append(h)
    ax.axvline(res["x_center"], color="0.5", ls=":", lw=1.0)
    x_lbl = f"log {res['x_name']}"
    ax.set_xlabel(x_lbl + ("  (scanned; held fixed in the fit)" if res.get("x_fixed") else "  (scanned)"))
    ax.set_ylabel("fitted log K")
    ax.grid(False)
    ax2 = ax.twinx()
    if np.isfinite(rmin) and rmin > 0:
        h2, = ax2.plot(x, rmse / rmin, "--s", ms=3.0, lw=1.1, color="0.35",
                       label="RMSE / RMSE$_{min}$ (right)")
        handles.append(h2)
        ax2.set_yscale("log")
        ax2.axhline(1.0, color="0.75", ls="--", lw=0.8)
    ax2.set_ylabel("RMSE / RMSE$_{min}$  (log scale)", color="0.35")
    ax2.tick_params(axis="y", colors="0.35")
    ax.set_title(title or f"Profile trace: {res['x_name']}")
    if show_legend and handles:
        ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="best")
    nf = int(res.get("n_failed", 0) or 0)
    if nf:
        ax.text(0.01, 0.01, f"{nf} of {res.get('n_total', '?')} refits failed",
                transform=ax.transAxes, fontsize=7, color="#C0392B", va="bottom")
    fig.tight_layout()
    return fig


# ── Cluster offload hook (see equilibrist_hpc.py) ────────────────────────────
try:
    from equilibrist_hpc import install_offloadable as _install_offloadable
    _install_offloadable(
        globals(),
        ["compute_rmse_profile", "compute_rmse_profile_2d",
         "compute_profile_trace", "compute_local_sensitivity"],
        "equilibrist_diagnostics")
except ImportError:
    pass
