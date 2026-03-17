# -*- coding: utf-8 -*-
"""equilibrist_kinetics_spectra.py"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls as _nnls
from equilibrist_kinetics import compute_kinetics_curve, _collect_all_kinetic_species
from equilibrist_fit_spectra import _optimal_spectral_range
from equilibrist_fit_nmr import _get_species_for_target, _hessian_errors, _resolve_c
from equilibrist_kinetics_nmr import _kinetics_nmr_integration_backCalc
from equilibrist_curve import convert_exp_x
from equilibrist_parser import constraints_penalty

__all__ = ['fit_kinetics_spectra']


def fit_kinetics_spectra(parsed: dict, logk_dict: dict, spectra_data: dict,
                         fit_keys: list, t_max: float, n_pts_sim: int,
                         wl_min: float, wl_max: float,
                         tolerance: float, maxiter: int,
                         timeout_s: float = 30.0, auto_range: bool = False,
                         use_lbfgsb: bool = True, use_neldermead: bool = True,
                         constraints=None, fit_conc_keys=None,
                         allow_negative_eps: bool = False):
    """
    Fit rate constants to UV-Vis kinetics spectra using Beer-Lambert law.

    A(t, λ) = Σⱼ εⱼ(λ) · [Sⱼ](t)

    C(t) from ODE integration; E solved analytically by lstsq; k optimised by Nelder-Mead.
    """

    def _solve_E(C_mat, A_mat):
        """Solve C @ E ≈ A for E, respecting allow_negative_eps flag."""
        if allow_negative_eps:
            E, _, _, _ = np.linalg.lstsq(C_mat, A_mat, rcond=None)
            return E
        return np.column_stack([_nnls(C_mat, A_mat[:, _j])[0]
                                for _j in range(A_mat.shape[1])])

    from scipy.optimize import minimize
    import time

    # ── Fittable concentrations ──────────────────────────────────────────────
    fit_conc_keys = list(fit_conc_keys or [])
    _n_k   = len(fit_keys)
    _n_c   = len(fit_conc_keys)
    CONC_MIN_K = 0.0

    _root_to_cname_k = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_k[_r] = _cn

    def _cb_k(root):
        _cn = _root_to_cname_k.get(root, root)
        sv  = float(parsed.get("concentrations", {}).get(_cn, 1.0))
        lo, hi = parsed.get("conc_bounds", {}).get(_cn,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(CONC_MIN_K, lo) if lo is not None else max(CONC_MIN_K, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    _conc_script_k = {}
    for _cn, _cv in parsed.get("concentrations", {}).items():
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _conc_script_k[_r] = float(_cv)

    _x0_c  = np.array([_conc_script_k.get(r, 1.0) for r in fit_conc_keys])
    _bds_c = [_cb_k(r) for r in fit_conc_keys]

    def _unpack_ck(params_vec):
        lk = params_vec[:_n_k]
        cd = {fit_conc_keys[i]: float(np.clip(params_vec[_n_k + i],
                                               _bds_c[i][0], _bds_c[i][1]))
              for i in range(_n_c)}
        return lk, cd

    def _patched_parsed_k(conc_d):
        if not conc_d:
            return parsed
        p = dict(parsed)
        p["concentrations"] = dict(parsed["concentrations"])
        for root, val in conc_d.items():
            cname = _root_to_cname_k.get(root, root)
            p["concentrations"][cname] = float(val)
        return p

    wavelengths = spectra_data["wavelengths"]
    t_exp       = spectra_data["x_vals"]
    A_full      = spectra_data["A"]

    spectra_cfg = parsed.get("spectra") or {}
    transparent = set(spectra_cfg.get("transparent", []))
    all_kin_sp  = _collect_all_kinetic_species(parsed)
    absorbers   = [sp for sp in all_kin_sp if sp not in transparent]

    if not absorbers:
        return False, {}, {}, "All species are transparent — nothing to fit"

    wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    A_fit   = A_full[:, wl_mask]
    if A_fit.shape[1] == 0:
        return False, {}, {}, "No wavelengths in selected range"

    n_pts  = len(t_exp)

    def _simulate(params_vec):
        lk_v, cd = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        try:
            return compute_kinetics_curve(_patched_parsed_k(cd), lk, t_max, n_pts_sim)
        except Exception:
            return None

    def _build_C(curve):
        t_sim = curve["t"]
        C = np.zeros((n_pts, len(absorbers)))
        for j, sp in enumerate(absorbers):
            c_sim = _resolve_c(curve, sp, parsed, t_sim)
            C[:, j] = np.interp(t_exp, t_sim, c_sim)
        return np.maximum(C, 0.0)

    def objective(params_vec):
        lk_v, _ = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        curve = _simulate(params_vec)
        if curve is None:
            return 1e12
        C = _build_C(curve)
        E = _solve_E(C, A_fit)
        ssr = float(np.sum((A_fit - C @ E) ** 2))
        cp = constraints_penalty(constraints or [], lk, ssr_scale=ssr)
        return ssr + cp

    def data_objective(params_vec):
        """Data-only SSR (no constraint penalty) — used for Hessian / error estimation."""
        curve = _simulate(params_vec)
        if curve is None:
            return 1e12
        C = _build_C(curve)
        E = _solve_E(C, A_fit)
        return float(np.sum((A_fit - C @ E) ** 2))

    class _Timeout(Exception):
        pass

    x0  = np.concatenate([np.array([logk_dict[k] for k in fit_keys]), _x0_c])
    n_p = len(x0)

    # Phase 1: warm-start logK before joint optimisation
    if _n_c > 0 and _n_k > 0:
        def _phase1_obj_sp(lk_vec):
            return objective(np.concatenate([lk_vec, x0[_n_k:]]))
        _sp1_sp = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                            for i in range(_n_k)])
        try:
            _r1_sp = minimize(_phase1_obj_sp, x0[:_n_k], method="Nelder-Mead",
                              options={"maxiter": maxiter//2,
                                       "xatol": tolerance, "fatol": tolerance * 1e-4,
                                       "adaptive": True,
                                       "initial_simplex": _sp1_sp})
            x0 = np.concatenate([_r1_sp.x, x0[_n_k:]])
        except Exception:
            pass

    best_tracker = {"x": x0.copy(), "f": np.inf,
                    "start": time.time(), "timed_out": False}

    # ── Stage 1: L-BFGS-B (timeout-free, warm-starts Nelder-Mead) ────────
    # log k is unbounded in principle; use loose bounds to keep solver stable
    _k_lo = np.concatenate([np.full(_n_k, -6.0), np.array([b[0] for b in _bds_c])])
    _k_hi = np.concatenate([np.full(_n_k,  9.0), np.array([b[1] for b in _bds_c])])
    bounds = list(zip(_k_lo.tolist(), _k_hi.tolist()))

    def objective_safe(logk_trial):
        penalty = sum(1e6*(v-lo)**2 for v,lo in zip(logk_trial,_k_lo) if v < lo) + \
                  sum(1e6*(v-hi)**2 for v,hi in zip(logk_trial,_k_hi) if v > hi)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        penalty += constraints_penalty(constraints or [], lk)
        if penalty > 0:
            return float(penalty)
        return objective(logk_trial)

    if use_lbfgsb:
        obj_start = objective_safe(x0)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                res_lbfgs = minimize(objective_safe, x0, method="L-BFGS-B",
                                     bounds=bounds,
                                     options={"maxiter": maxiter,
                                              "ftol": tolerance,
                                              "gtol": tolerance * 1e-3})
            if res_lbfgs.fun < obj_start * 0.999:
                x0 = res_lbfgs.x
                best_tracker["x"] = x0.copy()
                best_tracker["f"] = res_lbfgs.fun
        except Exception:
            pass   # fall through to Nelder-Mead

    def _obj_timed(logk_trial):
        f = objective(logk_trial)
        if f < best_tracker["f"]:
            best_tracker["f"] = f
            best_tracker["x"] = logk_trial.copy()
        if time.time() - best_tracker["start"] > timeout_s:
            best_tracker["timed_out"] = True
            raise _Timeout()
        return f

    _steps = np.array([1.5] * _n_k + [max(abs(_x0_c[i]) * 0.1, 0.05)
                       for i in range(_n_c)])
    init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _steps[i]
                                     for i in range(n_p)])
    if use_neldermead:
        try:
            result = minimize(_obj_timed, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x       = best_tracker["x"]
                success = False
                fun     = best_tracker["f"]
                nit     = 0
            result = _MockResult()
    else:
        # L-BFGS-B only — use its result directly
        class _MockResult:
            x       = best_tracker["x"]
            success = best_tracker["f"] < np.inf
            fun     = best_tracker["f"]
            nit     = 0
        result = _MockResult()

    # ── Auto-range pass 2 ────────────────────────────────────────────────
    if auto_range and len(absorbers) > 1:
        # Compute E from pass-1 result, find optimal wavelength window
        _curve1 = _simulate(result.x)
        if _curve1 is not None:
            _C1 = _build_C(_curve1)
            _E1 = _solve_E(_C1, A_fit)
            wl_fit_now = wavelengths[wl_mask]
            wl_min, wl_max = _optimal_spectral_range(wl_fit_now, _E1, min_width_nm=50.0)
            wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            A_fit   = A_full[:, wl_mask]
            if A_fit.shape[1] > 0:
                best_tracker["x"] = result.x.copy()
                best_tracker["f"] = np.inf
                best_tracker["start"] = time.time()
                try:
                    result = minimize(_obj_timed, result.x, method="Nelder-Mead",
                                      options={"maxiter": maxiter, "xatol": tolerance,
                                               "fatol": tolerance * 1e-4, "adaptive": True})
                except _Timeout:
                    class _MockResult2:
                        x       = best_tracker["x"]
                        success = False
                        fun     = best_tracker["f"]
                        nit     = 0
                    result = _MockResult2()

    timed_out      = best_tracker["timed_out"]
    fitted_logks   = {fit_keys[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1])) for i in range(_n_c)}

    # ── Final statistics ───────────────────────────────────────────────────
    curve_final = _simulate(result.x)
    if curve_final is None:
        return False, fitted_logks, {}, "ODE failed at fitted parameters"

    C_final = _build_C(curve_final)
    wl_fit  = wavelengths[wl_mask]
    E_final = _solve_E(C_final, A_fit)
    A_calc  = C_final @ E_final

    C_back, _, _, _ = np.linalg.lstsq(E_final.T, A_fit.T, rcond=None)
    C_back = np.clip(C_back.T, 0.0, None)

    residuals = (A_fit - A_calc).ravel()
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((A_fit - A_fit.mean()) ** 2))
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    _c_res    = (C_back - C_final).ravel()
    _c_sst    = float(np.sum((C_back - C_back.mean()) ** 2))
    r2_conc   = float(1.0 - np.sum(_c_res ** 2) / max(_c_sst, 1e-30))
    rmse_conc = float(np.sqrt(np.sum(_c_res ** 2) / max(len(_c_res), 1)))

    _err_idx     = _hessian_errors(data_objective, result.x, ssr, len(residuals), n_p)
    param_errors = {}
    for _i in range(min(len(fit_keys), n_p)):
        if _i in _err_idx: param_errors[fit_keys[_i]] = _err_idx[_i]
    for _i in range(len(fit_conc_keys)):
        _idx = len(fit_keys) + _i
        if _idx in _err_idx: param_errors[fit_conc_keys[_i]] = _err_idx[_idx]

    _r2_ok = r2 >= 0.99
    if timed_out and _r2_ok:
        timed_out = False
    _conv = getattr(result, "success", False) or ssr < 1e-6 or (not timed_out and _r2_ok)

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        len(residuals),
        "n_params":        n_p,
        "param_values":    fitted_logks,
        "param_errors":    param_errors,
        "fit_mode":        "kinetics_spectra",
        "n_iter":          getattr(result, "nit", 0),
        "timed_out":       timed_out,
        "r2_conc":         r2_conc,
        "rmse_conc":       rmse_conc,
        "absorbers":       absorbers,
        "x_exp":           t_exp,
        "C_back":          C_back,
        "E_final":         E_final,
        "wavelengths_fit": wl_fit,
        "opt_wl_min":      wl_min,
        "opt_wl_max":      wl_max,
        "auto_range":      auto_range,
        "sp_concs": {}, "col_to_sp": {}, "col_to_nH": {},
        "pure_shifts": {}, "delta_vecs_all": {}, "delta_bound_all": {},
        "delta_free": {}, "x_free_val": {}, "col_to_target": {}, "ref_corrections": {},
        "fitted_concs":    fitted_concs_k, "fitted_titrants": {},
    }
    return _conv, fitted_logks, stats, "Kinetics UV-Vis spectra fit complete"
