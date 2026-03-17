# -*- coding: utf-8 -*-
"""equilibrist_fit_spectra.py"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls as _nnls
from equilibrist_network import solve_equilibria_general
from equilibrist_fit_nmr import _hessian_errors
from equilibrist_curve import convert_exp_x, _solid_col_header_to_equiv
from equilibrist_parser import constraints_penalty

__all__ = ['_compute_at_volumes', '_optimal_spectral_range', '_identifiability_check', 'fit_spectra']


# ─────────────────────────────────────────────
# UV-VIS SPECTRA ENGINE
# ─────────────────────────────────────────────

def _compute_at_volumes(v_add_mL: np.ndarray, params: dict, network: dict,
                        logK_vals: dict) -> dict:
    """
    Solve equilibrium directly at each experimental volume point.
    Much faster than compute_curve (no grid sweep, no interpolation).

    Uses warm-starting: the log-concentration solution from point i is used as
    the initial guess for point i+1, which typically converges in 1–2 iterations
    vs. 10–50 for a cold start.

    Returns dict {species: np.ndarray (n_pts,)} in mM, plus "x_vals" = v_add_mL.
    """
    conc0_mM   = params["conc0"]
    V0         = params["V0_mL"]
    tit_names  = params["titrant_free_names"]
    tit_keys   = params["titrant_keys"]
    tit_mMs    = params["titrant_mMs"]
    is_solid   = params["titrant_is_solid"]
    tit_name   = tit_names[0]
    tit_mM     = tit_mMs[tit_name]
    primary    = params["primary_component"]

    n0           = {name: conc * V0 for name, conc in conc0_mM.items()}
    all_species  = network["all_species"]
    equilibria   = network["equilibria"]

    if is_solid:
        n_primary    = n0.get(primary, 1.0)
        ratio_sum    = sum(params.get("titrant_ratios", {1: 1.0}).values()) or 1.0
        solid_fractions = {tfree: params.get("titrant_ratios", {}).get(tfree, 1.0) / ratio_sum
                           for tfree in tit_names}
    else:
        solid_fractions = {}

    out    = {sp: np.zeros(len(v_add_mL)) for sp in all_species}
    y_warm = None   # warm-start vector: log-concentrations from previous point

    for i, v_add in enumerate(v_add_mL):
        if is_solid:
            eq     = v_add  # already equivalents in solid mode
            n_primary = n0.get(primary, 1.0)
            n_tit  = eq * n_primary
            V      = V0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree in tit_names:
                frac = solid_fractions.get(tfree, 1.0)
                totals[tfree] = totals.get(tfree, 0.0) + (n_tit * frac / V) * 1e-3
        else:
            n_tit = v_add * tit_mM          # mmol of primary titrant
            V     = V0 + v_add
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree, tkey in zip(tit_names, tit_keys):
                ratio    = tit_mMs[tfree] / max(tit_mM, 1e-12)
                tit_conc = (n_tit * ratio / V) * 1e-3
                totals[tfree] = totals.get(tfree, 0.0) + tit_conc

        # Call solver directly — warm-start from previous point's solution
        final_concs, success, y_warm = solve_equilibria_general(
            totals, equilibria, all_species, logK_vals, y0_warm=y_warm)

        # Store for species_conc_fn (used downstream)
        network["_rigorous_concentrations"] = final_concs

        for sp in all_species:
            out[sp][i] = final_concs.get(sp, 0.0) * 1e3   # M → mM

    return out


def _optimal_spectral_range(wavelengths: np.ndarray, E: np.ndarray,
                             min_width_nm: float = 50.0) -> tuple:
    """
    Find the optimal contiguous wavelength range for fitting.

    Strategy: exclude low-contrast edges (where all absorbers look similar),
    keep everything above a contrast threshold. This preserves the full
    information content of informative regions while dropping dead zones.

    Contrast at λ = variance of E[:, λ] across absorbers.
    Threshold = 10th percentile of contrast across all wavelengths.
    Result = [first λ above threshold, last λ above threshold],
             expanded if needed to meet min_width_nm.
    """
    contrast  = np.var(E, axis=0)          # (n_wl,)
    threshold = np.percentile(contrast, 10)
    above     = np.where(contrast >= threshold)[0]

    if len(above) == 0:
        return float(wavelengths[0]), float(wavelengths[-1])

    i_lo = int(above[0])
    i_hi = int(above[-1])

    # Enforce minimum width
    wl_step = float(np.median(np.diff(wavelengths))) if len(wavelengths) > 1 else 1.0
    min_pts = max(1, int(np.ceil(min_width_nm / wl_step)))
    while (i_hi - i_lo) < min_pts - 1:
        if i_lo > 0:
            i_lo -= 1
        if i_hi < len(wavelengths) - 1:
            i_hi += 1
        if i_lo == 0 and i_hi == len(wavelengths) - 1:
            break

    return float(wavelengths[i_lo]), float(wavelengths[i_hi])


def _identifiability_check(obj_fn, x_best, fit_keys, ssr, n_residuals):
    """
    Analyse parameter identifiability via numerical Hessian eigendecomposition.

    Returns:
        is_correlated (bool)   — True if condition number > 1e5
        cond          (float)  — condition number of Hessian
        combinations  (list)   — [(label, value, stderr), ...] well-determined combos
        timed_out     (bool)   — forwarded flag for display
    """
    n = len(fit_keys)
    if n < 2:
        return False, 1.0, [], False

    # Numerical Hessian via 4-point central differences
    eps = 5e-3
    H = np.zeros((n, n))
    f0 = obj_fn(x_best)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            fpp = obj_fn(x_best + ei + ej)
            fpm = obj_fn(x_best + ei - ej)
            fmp = obj_fn(x_best - ei + ej)
            fmm = obj_fn(x_best - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)

    H = (H + H.T) / 2  # ensure symmetry
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return False, 1.0, [], False

    pos = eigvals[eigvals > 1e-30]
    if len(pos) < 2:
        return False, 1.0, [], False
    cond = float(pos[-1] / pos[0])
    is_correlated = cond > 1e5

    if not is_correlated:
        return False, cond, [], False

    # σ² from SSR / dof; uncertainty on combo v·x is σ/√λ
    dof    = max(n_residuals - n, 1)
    sigma2 = max(ssr / dof, 1e-30)

    # Iterate eigenvalues largest → smallest; collect well-determined combos
    combinations = []
    order = np.argsort(eigvals)[::-1]   # largest eigenvalue first
    max_ev = float(eigvals[order[0]])

    for idx_ev in order:
        ev  = float(eigvals[idx_ev])
        if ev <= 1e-30:
            continue
        ratio = ev / max_ev
        vec   = eigvecs[:, idx_ev]         # (n,) unit eigenvector

        # Only keep combinations where eigenvector is reasonably localised
        sig_mask = np.abs(vec) > 0.15
        if not sig_mask.any():
            continue

        # Value of this linear combination at best-fit point
        raw_val = float(np.dot(vec, x_best))
        stderr  = float(np.sqrt(sigma2 / ev))

        # Check if all significant components have nearly equal coefficients
        sig_coeffs = vec[sig_mask]
        all_keys   = [fit_keys[j] for j in range(n)]
        sig_keys   = [fit_keys[j] for j in range(n) if sig_mask[j]]

        equal_mag = np.std(np.abs(sig_coeffs)) < 0.10 * np.mean(np.abs(sig_coeffs))
        all_same_sign = np.all(sig_coeffs > 0) or np.all(sig_coeffs < 0)

        if equal_mag and all_same_sign and len(sig_keys) == n:
            # Sum combination: v = [c,c,...,c]/norm → represents log(∏ K)
            # actual sum of log K = raw_val / c  (since v·x = c * Σx = raw_val)
            c         = float(np.mean(sig_coeffs))
            sum_logK  = raw_val / c
            se_sum    = stderr / abs(c)
            label     = "log(" + "·".join(fit_keys) + ")"
            combinations.append((label, sum_logK, se_sum, ratio))
        else:
            # General combination: report as weighted sum
            parts = []
            for j in range(n):
                coeff = float(vec[j])
                if abs(coeff) > 0.15:
                    if abs(abs(coeff) - 1.0) < 0.15:
                        parts.append(("+" if coeff > 0 else "-") + f"log {fit_keys[j]}")
                    else:
                        parts.append(f"{coeff:+.2f}·log {fit_keys[j]}")
            label = "".join(parts).lstrip("+")
            combinations.append((label, raw_val, stderr, ratio))

        if len(combinations) >= 2:
            break

    return is_correlated, cond, combinations, False


def fit_spectra(parsed: dict, network: dict, spectra_data: dict,
                params: dict, logK_vals: dict, fit_keys: list,
                x_expr: str, wl_min: float, wl_max: float,
                tolerance: float, maxiter: int,
                auto_range: bool = False, timeout_s: float = 30.0,
                constraints=None,
                fit_conc_keys=None, fit_titrant_keys=None,
                allow_negative_eps: bool = False):
    """
    Fit equilibrium constants to UV-Vis absorbance data using Beer-Lambert law.

    Physical model
    ──────────────
    A(x, λ) = Σⱼ εⱼ(λ) · [Sⱼ](x)           (Beer-Lambert, path length absorbed into ε)

    In matrix form:  A = C @ E
      A  (n_spectra × n_wl)      observed absorbance (wavelength-filtered)
      C  (n_spectra × n_absorbers)  concentrations from equilibrium solver (mM)
      E  (n_absorbers × n_wl)    molar absorptivities (mM⁻¹, path length absorbed)

    For fixed K → C is fully determined → E solved analytically by lstsq.
    Nelder-Mead optimises K only; each objective call solves equilibrium at
    the experimental points directly (no grid sweep, no interpolation).

    If auto_range=True, a two-pass procedure is used:
      Pass 1: fit over [wl_min, wl_max] to get E_init
      Range:  find optimal 50+ nm window by spectral contrast in E_init
      Pass 2: refit over optimal range → final K and stats
      The optimal range is returned in stats as "opt_wl_min" / "opt_wl_max".
    """

    def _solve_E(C_mat, A_mat):
        """Solve C @ E ≈ A for E, respecting allow_negative_eps flag."""
        if allow_negative_eps:
            E, _, _, _ = np.linalg.lstsq(C_mat, A_mat, rcond=None)
            return E
        return np.column_stack([_nnls(C_mat, A_mat[:, _j])[0]
                                for _j in range(A_mat.shape[1])])

    from scipy.optimize import minimize
    from scipy.linalg import null_space as _null_space

    wavelengths = spectra_data["wavelengths"]
    x_raw       = spectra_data["x_vals"].copy()
    A_full      = spectra_data["A"]
    n_pts       = len(x_raw)

    # ── Solid mode: convert col-A header units → equivalents ────────────────
    _sp_x_col_header = spectra_data.get("x_col_header", "")
    if params.get("titrant_is_solid", False) and _sp_x_col_header:
        x_raw = _solid_col_header_to_equiv(x_raw, _sp_x_col_header, parsed, params)

    spectra_cfg = parsed.get("spectra") or {}
    transparent = set(spectra_cfg.get("transparent", []))
    absorbers   = [sp for sp in network["all_species"] if sp not in transparent]

    if not absorbers:
        return False, {}, {}, "All species are transparent — nothing to fit"

    def _mask(lo, hi):
        return (wavelengths >= lo) & (wavelengths <= hi)

    # ── Precompute topology matrices (K-independent) ────────────────────────
    # These are computed ONCE and reused across all Nelder-Mead iterations.
    equilibria  = network["equilibria"]
    all_species = network["all_species"]
    n_sp        = len(all_species)
    sp_idx      = {sp: i for i, sp in enumerate(all_species)}
    absorber_idx = np.array([sp_idx[sp] for sp in absorbers])

    nu = np.zeros((len(equilibria), n_sp))
    for i, eq in enumerate(equilibria):
        for coeff, sp in eq["reactants"]: nu[i, sp_idx[sp]] -= float(coeff)
        for coeff, sp in eq["products"]:  nu[i, sp_idx[sp]] += float(coeff)

    Lambda = _null_space(nu, rcond=1e-10).T      # (n_cons, n_sp)

    # Precompute per-point totals (K-independent)
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)

    V0        = params["V0_mL"]
    tit_names = params["titrant_free_names"]
    tit_keys  = params["titrant_keys"]
    is_solid  = params["titrant_is_solid"]

    def _build_c0_T(cur_params):
        """Compute c0_arr (n_pts × n_sp) and T_arr from current params."""
        p_c     = cur_params["conc0"]
        _tit_mMs = cur_params["titrant_mMs"]
        _tit_mM  = _tit_mMs[tit_names[0]]
        _n0 = {name: conc * V0 for name, conc in p_c.items()}
        _c0 = np.zeros((n_pts, n_sp))
        for ii, v in enumerate(x_raw):
            if is_solid:
                primary   = cur_params["primary_component"]
                n_primary = _n0.get(primary, 1.0)
                ratio_sum = sum(cur_params.get("titrant_ratios", {1: 1.0}).values()) or 1.0
                sf = {tf: cur_params.get("titrant_ratios", {}).get(tf, 1.0) / ratio_sum
                      for tf in tit_names}
                n_tit = v * n_primary
                _V = V0
                for name in _n0:
                    if name in sp_idx:
                        _c0[ii, sp_idx[name]] = (_n0[name] / _V) * 1e-3
                for tf in tit_names:
                    if tf in sp_idx:
                        _c0[ii, sp_idx[tf]] += (n_tit * sf.get(tf, 1.0) / _V) * 1e-3
            else:
                _V = V0 + v
                for name in _n0:
                    if name in sp_idx:
                        _c0[ii, sp_idx[name]] = (_n0[name] / _V) * 1e-3
                for tf, tk in zip(tit_names, tit_keys):
                    if tf in sp_idx:
                        ratio = _tit_mMs[tf] / max(_tit_mM, 1e-12)
                        n_tit = v * _tit_mM
                        _c0[ii, sp_idx[tf]] += (n_tit * ratio / _V) * 1e-3
        _T = (Lambda @ _c0.T).T
        return _c0, _T

    c0_arr, T_arr = _build_c0_T(params)

    # Warm-start cache: y[ii] = log-concentrations from previous NM step at point ii
    # Initialized from analytical totals; updated after every successful NM evaluation.
    y_cache = np.log(np.maximum(c0_arr, 1e-20))

    # kname → index in equilibria list (for fast lnK construction)
    eq_knames = [eq["kname"] for eq in equilibria]

    def _fast_solve_all(logk_vec):
        """Solve equilibrium at all n_pts points; return C (n_pts × n_absorbers).
        Uses warm-start from y_cache and updates it in-place on success."""
        from scipy.optimize import least_squares as _lsq
        lnK = logk_vec * np.log(10.0)
        C   = np.zeros((n_pts, len(absorbers)))
        new_y = y_cache.copy()
        for ii in range(n_pts):
            T = T_arr[ii]
            def res(y, _lnK=lnK, _T=T):
                return np.concatenate([_lnK - nu @ y,
                                       Lambda @ np.exp(np.clip(y, -80, 10)) - _T])
            sol = _lsq(res, y_cache[ii], method='lm',
                       xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
            if np.max(np.abs(sol.fun)) > 1e-4:
                # retry with cold start
                y0_cold = np.log(np.maximum(c0_arr[ii], 1e-20))
                sol2 = _lsq(res, y0_cold, method='lm',
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=500)
                if np.max(np.abs(sol2.fun)) < np.max(np.abs(sol.fun)):
                    sol = sol2
            new_y[ii] = sol.x
            cf = np.exp(np.clip(sol.x, -80, 10)) * 1e3   # M → mM
            C[ii] = cf[absorber_idx]
        y_cache[:] = new_y
        return C

    def _build_lnK(logk_trial):
        lk = np.array([logK_vals[kn] for kn in eq_knames])
        for i, k in enumerate(fit_keys):
            j = eq_knames.index(k)
            lk[j] = logk_trial[i]
        return lk

    def _run_fit(A_fit):
        """Run Nelder-Mead with timeout and best-so-far tracking.
        If fit_conc_keys / fit_titrant_keys are set, concentrations are appended
        to the parameter vector and c0_arr / T_arr are recomputed at each step."""
        import time

        class _Timeout(Exception):
            pass

        # ── Concentration fitting setup ──────────────────────────────────────
        CONC_MIN = 0.0
        _conc_names    = fit_conc_keys
        _titrant_names = fit_titrant_keys
        _n_conc    = len(_conc_names)
        _n_titrant = len(_titrant_names)

        # Map root → original cname for conc_bounds lookup (e.g. 'G' → 'G0')
        _root_to_cname_sp = {}
        for _cn in parsed.get("concentrations", {}):
            _r = _cn[:-1] if _cn.endswith("0") else _cn
            _root_to_cname_sp[_r] = _cn

        def _cb(root, default_val):
            cname = _root_to_cname_sp.get(root, root)
            lo, hi = parsed.get("conc_bounds", {}).get(cname,
                     parsed.get("conc_bounds", {}).get(root, (None, None)))
            script_val = parsed.get("concentrations", {}).get(cname, default_val)
            lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, script_val * 0.80)
            hi = hi if hi is not None else script_val * 1.20
            return (lo, hi)

        def _tb(tkey, default_val):
            lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
            script_val = parsed.get("titrant", {}).get(tkey, default_val)
            lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, script_val * 0.80)
            hi = hi if hi is not None else script_val * 1.20
            return (lo, hi)

        # Seed from script values so optimizer starts inside bounds
        _root_to_cname_sp = {}
        for _cn in parsed.get("concentrations", {}):
            _r = _cn[:-1] if _cn.endswith("0") else _cn
            _root_to_cname_sp[_r] = _cn
        def _script_val_sp(root):
            cname = _root_to_cname_sp.get(root, root)
            sv = parsed.get("concentrations", {}).get(cname)
            return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
        _x0_conc    = [_script_val_sp(cn) for cn in _conc_names]
        _x0_titrant = []
        for tk in _titrant_names:
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            _x0_titrant.append(params["titrant_mMs"].get(tfree, 10.0))
        _bounds_conc    = [_cb(cn, _x0_conc[i])    for i, cn in enumerate(_conc_names)]
        _bounds_titrant = [_tb(tk, _x0_titrant[i]) for i, tk in enumerate(_titrant_names)]

        def _make_params_sp(conc_d, tit_d):
            p = dict(params)
            p["conc0"] = dict(params["conc0"])
            p["titrant_mMs"] = dict(params["titrant_mMs"])
            for root, val in conc_d.items():
                p["conc0"][root] = float(val)
            for tkey, val in tit_d.items():
                tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
                p["titrant_mMs"][tfree] = float(val)
            return p

        y_cache[:] = np.log(np.maximum(c0_arr, 1e-20))   # reset warm-start

        best_tracker = {"x": np.array([logK_vals[k] for k in fit_keys] + _x0_conc + _x0_titrant),
                        "f": np.inf,
                        "start": time.time(),
                        "timed_out": False}

        # Per-parameter soft bounds from parser (fall back to -2/14 if not specified)
        _DEFAULT_LO, _DEFAULT_HI = -2.0, 14.0
        _eq_by_kname = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
        _k_lo = np.array([_eq_by_kname.get(k, {}).get("logK_lo") or _DEFAULT_LO for k in fit_keys])
        _k_hi = np.array([_eq_by_kname.get(k, {}).get("logK_hi") or _DEFAULT_HI for k in fit_keys])

        n_k = len(fit_keys)

        def objective(trial):
            logk_trial = trial[:n_k]
            # Soft-wall penalty: grows quadratically outside per-parameter bounds
            penalty = 0.0
            for kv, lo, hi in zip(logk_trial, _k_lo, _k_hi):
                if kv < lo:
                    penalty += 1e6 * (kv - lo) ** 2
                elif kv > hi:
                    penalty += 1e6 * (kv - hi) ** 2
            lk_dict = {fit_keys[i]: logk_trial[i] for i in range(n_k)}
            if penalty > 0:
                if penalty < best_tracker["f"]:
                    best_tracker["f"] = penalty
                    best_tracker["x"] = trial.copy()
                if time.time() - best_tracker["start"] > timeout_s:
                    best_tracker["timed_out"] = True
                    raise _Timeout()
                return penalty

            # Update c0_arr / T_arr / y_cache when fitting concentrations
            if fitting_concs:
                conc_d = {_conc_names[i]: float(np.clip(trial[n_k+i],
                          _bounds_conc[i][0], _bounds_conc[i][1]))
                          for i in range(_n_conc)}
                tit_d  = {_titrant_names[i]: float(np.clip(trial[n_k+_n_conc+i],
                          _bounds_titrant[i][0], _bounds_titrant[i][1]))
                          for i in range(_n_titrant)}
                cur_params_sp = _make_params_sp(conc_d, tit_d)
                _c0, _T = _build_c0_T(cur_params_sp)
                y_cache[:] = np.log(np.maximum(_c0, 1e-20))
                lk = _build_lnK(logk_trial)
                # Use local _fast_solve with updated arrays
                from scipy.optimize import least_squares as _lsq
                lnK = lk * np.log(10.0)
                C_sp = np.zeros((n_pts, len(absorbers)))
                for ii in range(n_pts):
                    T_ii = _T[ii]
                    def res(y, _lnK=lnK, _T2=T_ii):
                        return np.concatenate([_lnK - nu @ y,
                                               Lambda @ np.exp(np.clip(y, -80, 10)) - _T2])
                    sol = _lsq(res, y_cache[ii], method='lm',
                               xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=300)
                    y_cache[ii] = sol.x
                    cf = np.exp(np.clip(sol.x, -80, 10)) * 1e3
                    C_sp[ii] = cf[absorber_idx]
                C = C_sp
            else:
                lk = _build_lnK(logk_trial)
                C  = _fast_solve_all(lk)

            E = _solve_E(C, A_fit)
            f  = float(np.sum((A_fit - C @ E) ** 2))
            cp = constraints_penalty(constraints or [], {**logK_vals, **lk_dict}, ssr_scale=f)
            f_total = f + cp
            if f_total < best_tracker["f"]:
                best_tracker["f"] = f_total
                best_tracker["x"] = trial.copy()
            if time.time() - best_tracker["start"] > timeout_s:
                best_tracker["timed_out"] = True
                raise _Timeout()
            return f_total

        x0  = np.array([logK_vals[k] for k in fit_keys] + _x0_conc + _x0_titrant)
        n_p = len(fit_keys)
        bounds = [(_k_lo[i], _k_hi[i]) for i in range(n_p)] + _bounds_conc + _bounds_titrant

        # Phase 1: warm-start logK before joint optimisation
        if _n_conc > 0 and n_k > 0:
            def _phase1_obj_spfit(lk_vec):
                full = np.concatenate([lk_vec, x0[n_k:]])
                return objective(full)
            _sp1_sf = np.vstack([x0[:n_k]] +
                                [x0[:n_k] + np.eye(n_k)[i]*1.5 for i in range(n_k)])
            try:
                _r1_sf = minimize(_phase1_obj_spfit, x0[:n_k], method="Nelder-Mead",
                                  options={"maxiter": maxiter//2,
                                           "xatol": tolerance, "fatol": tolerance * 1e-4,
                                           "adaptive": True,
                                           "initial_simplex": _sp1_sf})
                x0 = np.concatenate([_r1_sf.x, x0[n_k:]])
            except Exception:
                pass

        # Timeout-free objective for L-BFGS-B (same logic as objective but no timeout)
        def objective_safe(trial):
            logk_trial = trial[:n_k]
            penalty = sum(1e6*(kv-lo)**2 for kv,lo in zip(logk_trial,_k_lo) if kv < lo) + \
                      sum(1e6*(kv-hi)**2 for kv,hi in zip(logk_trial,_k_hi) if kv > hi)
            lk_dict = {fit_keys[i]: logk_trial[i] for i in range(n_k)}
            if penalty > 0:
                return penalty
            if fitting_concs:
                try:
                    return objective(trial)
                except Exception:
                    return 1e9
            lk = _build_lnK(logk_trial)
            C  = _fast_solve_all(lk)
            E = _solve_E(C, A_fit)
            f_safe = float(np.sum((A_fit - C @ E) ** 2))
            cp_safe = constraints_penalty(constraints or [], {**logK_vals, **lk_dict}, ssr_scale=f_safe)
            return f_safe + cp_safe

        # ── Stage 1: L-BFGS-B ──────────────────────────────────────
        obj_start = objective_safe(x0)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                res_lbfgs = minimize(objective_safe, x0, method="L-BFGS-B",
                                     bounds=bounds,
                                     options={"maxiter": maxiter, "ftol": tolerance,
                                              "gtol": tolerance * 1e-3})
            lbfgs_improved = res_lbfgs.fun < obj_start * 0.999
        except Exception:
            res_lbfgs = None
            lbfgs_improved = False

        if lbfgs_improved:
            x0 = res_lbfgs.x   # warm-start Nelder-Mead from L-BFGS-B result
            best_tracker["x"] = x0.copy()
            best_tracker["f"] = res_lbfgs.fun

        # ── Stage 2: Nelder-Mead (always runs, warm-started if BFGS helped) ──
        init_simplex = np.vstack([x0] +
                                  [x0 + np.eye(n_p)[i] * (
                                      1.5 if i < len(fit_keys)
                                      else max(abs(x0[i]) * 0.1, 0.05))
                                   for i in range(n_p)])
        try:
            result = minimize(objective, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x       = best_tracker["x"]
                success = False
                fun     = best_tracker["f"]
            result = _MockResult()

        result._timed_out = best_tracker["timed_out"]
        result._obj_fn = objective_safe
        return result, objective

    def _get_C_from_logKs(logk_vec):
        """Re-solve at final parameters to get C (bypasses cache reset)."""
        return _fast_solve_all(logk_vec)

    # ── Pass 1 (always) ────────────────────────────────────────────────────
    wl_mask1 = _mask(wl_min, wl_max)
    A_fit1   = A_full[:, wl_mask1]
    if A_fit1.shape[1] == 0:
        return False, {}, {}, "No wavelengths in selected range"

    result1, _obj1 = _run_fit(A_fit1)
    fitted_logKs   = {fit_keys[i]: result1.x[i] for i in range(len(fit_keys))}
    # Extract fitted concentrations/titrant from extended vector
    _n_k = len(fit_keys)
    fitted_concs_sp    = {fit_conc_keys[i]: float(result1.x[_n_k+i])
                          for i in range(len(fit_conc_keys))}
    fitted_titrants_sp = {fit_titrant_keys[i]: float(result1.x[_n_k+len(fit_conc_keys)+i])
                          for i in range(len(fit_titrant_keys))}

    # Compute E from pass-1 result (needed for range optimisation)
    lk1 = _build_lnK(result1.x)
    C1  = _get_C_from_logKs(lk1)
    E1 = _solve_E(C1, A_fit1)

    opt_wl_min, opt_wl_max = wl_min, wl_max

    # ── Pass 2 (only if auto_range) ────────────────────────────────────────
    if auto_range and len(absorbers) > 1:
        wl_fit1    = wavelengths[wl_mask1]
        opt_wl_min, opt_wl_max = _optimal_spectral_range(wl_fit1, E1, min_width_nm=50.0)
        wl_mask2 = _mask(opt_wl_min, opt_wl_max)
        A_fit2   = A_full[:, wl_mask2]
        if A_fit2.shape[1] > 0:
            result2, _obj2 = _run_fit(A_fit2)
            fitted_logKs   = {fit_keys[i]: result2.x[i] for i in range(len(fit_keys))}
            fitted_concs_sp    = {fit_conc_keys[i]: float(result2.x[_n_k+i])
                                  for i in range(len(fit_conc_keys))}
            fitted_titrants_sp = {fit_titrant_keys[i]: float(result2.x[_n_k+len(fit_conc_keys)+i])
                                  for i in range(len(fit_titrant_keys))}
            result1 = result2

    # ── Final statistics ───────────────────────────────────────────────────
    wl_mask_f = _mask(opt_wl_min, opt_wl_max)
    wl_fit_f  = wavelengths[wl_mask_f]
    A_fit_f   = A_full[:, wl_mask_f]

    lk_final     = _build_lnK(result1.x)

    # Build final params with fitted concentrations so x_exp and C_final
    # use the same cage0 scale as the theoretical curve in the main plot.
    _fin_params = dict(params)
    if fitting_concs and (fitted_concs_sp or fitted_titrants_sp):
        _fin_params["conc0"] = dict(params["conc0"])
        _fin_params["titrant_mMs"] = dict(params["titrant_mMs"])
        for _root, _val in fitted_concs_sp.items():
            _fin_params["conc0"][_root] = float(_val)
        for _tkey, _val in fitted_titrants_sp.items():
            _tfree = _tkey[:-1] if (_tkey.endswith("t") or _tkey.endswith("0")) else _tkey
            _fin_params["titrant_mMs"][_tfree] = float(_val)
        c0_arr[:], T_arr[:] = _build_c0_T(_fin_params)
        y_cache[:] = np.log(np.maximum(c0_arr, 1e-20))

    C_final      = _get_C_from_logKs(lk_final)
    E_final = _solve_E(C_final, A_fit_f)
    A_calc_final = C_final @ E_final

    C_back, _, _, _ = np.linalg.lstsq(E_final.T, A_fit_f.T, rcond=None)
    C_back = np.clip(C_back.T, 0.0, None)

    x_exp = convert_exp_x(x_raw, x_expr, parsed, _fin_params, network)

    residuals = (A_fit_f - A_calc_final).ravel()
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((A_fit_f - A_fit_f.mean()) ** 2))
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    # Concentration R²: how well does C_final (equilibrium model) match C_back
    # (concentrations back-projected from absorbance data)?
    _c_res = (C_back - C_final).ravel()
    _c_sst = float(np.sum((C_back - C_back.mean()) ** 2))
    r2_conc = float(1.0 - np.sum(_c_res ** 2) / max(_c_sst, 1e-30))
    rmse_conc = float(np.sqrt(np.sum(_c_res ** 2) / max(len(_c_res), 1)))  # mM

    def _obj_final(logk_trial):
        lk = _build_lnK(logk_trial)
        C  = _get_C_from_logKs(lk)
        E = _solve_E(C, A_fit_f)
        return float(np.sum((A_fit_f - C @ E) ** 2))

    _err_idx     = _hessian_errors(_obj_final, result1.x, ssr, len(residuals), len(fit_keys))
    param_errors = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        len(residuals),
        "n_params":        len(fit_keys),
        "param_values":    fitted_logKs,
        "param_errors":    param_errors,
        "fitted_concs":    fitted_concs_sp,
        "fitted_titrants": fitted_titrants_sp,
        "fit_mode":        "spectra",
        "n_iter":          getattr(result1, "nit", 0),
        "r2_conc":         r2_conc,
        "rmse_conc":       rmse_conc,
        "absorbers":       absorbers,
        "x_exp":           x_exp,
        "C_back":          C_back,
        "E_final":         E_final,
        "wavelengths_fit": wl_fit_f,
        "opt_wl_min":      opt_wl_min,
        "opt_wl_max":      opt_wl_max,
        "auto_range":      auto_range,
        "sp_concs": {}, "col_to_sp": {}, "col_to_nH": {},
        "pure_shifts": {}, "delta_vecs_all": {}, "delta_bound_all": {},
        "delta_free": {}, "x_free_val": {}, "col_to_target": {}, "ref_corrections": {},
    }
    # ── Identifiability analysis via concentration matrix condition ────────
    timed_out = getattr(result1, "_timed_out", False)
    is_corr, cond_num, combos = False, 1.0, []
    if len(fit_keys) >= 2:
        try:
            # Condition number of C: if large, concentration profiles are near-linearly
            # dependent → individual K values cannot be separated from spectral data.
            # This is more reliable than the Hessian approach when SSR ≈ 0.
            _sv = np.linalg.svd(C_final, compute_uv=False)
            cond_num = float(_sv[0] / max(_sv[-1], 1e-30))
            is_corr = cond_num > 1e4

            if is_corr and len(fit_keys) >= 2:
                # Report the sum log K (= log β_n) as the well-determined combination
                sum_logK = float(sum(result1.x))
                # Rough uncertainty: from SSR / (dof * min_sv²)
                dof = max(len(residuals) - len(fit_keys), 1)
                se_sum = float(np.sqrt(ssr / max(dof, 1)) / max(_sv[-1], 1e-30)) * 0.1
                se_sum = min(se_sum, 2.0)  # cap at 2 log units
                label = "log(" + "·".join(fit_keys) + ")"
                combos = [(label, sum_logK, se_sum, 1.0)]
        except Exception:
            pass

    # Sanity-check for escaped bounds (should be prevented by soft penalty, but belt-and-braces)
    unphysical = [k for k, v in zip(fit_keys, result1.x) if v < -2 or v > 14]
    if unphysical:
        is_corr = True
        combos = [("WARNING: unphysical values", 0.0, 0.0, 0.0)]

    stats["timed_out"]       = timed_out
    stats["is_correlated"]   = is_corr
    stats["cond_number"]     = cond_num
    stats["identifiable"]    = combos   # [(label, value, stderr, ratio), ...]

    _r2_spec = stats.get("r_squared", 0.0)
    if timed_out and _r2_spec >= 0.99:
        timed_out = False
        stats["timed_out"] = False
    msg = ("UV-Vis fit timed out — best parameters so far shown"
           if timed_out else "UV-Vis spectra fit complete")
    _conv_spec = timed_out is False and (result1.success or ssr < 1e-6 or _r2_spec >= 0.99)
    return _conv_spec, fitted_logKs, stats, msg



# ─────────────────────────────────────────────────────────────────────────────
