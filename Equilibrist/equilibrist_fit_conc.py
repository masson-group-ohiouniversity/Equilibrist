# -*- coding: utf-8 -*-
"""equilibrist_fit_conc.py"""
import numpy as np
from scipy.optimize import minimize
from equilibrist_curve import compute_single_point, find_equiv_for_x, convert_exp_x
from equilibrist_parser import constraints_penalty

__all__ = ['fit_parameters']


def _make_params_with_concs(params, conc_vals_mM, titrant_vals_mM):
    p = dict(params)
    p["conc0"] = dict(params["conc0"])
    p["titrant_mMs"] = dict(params["titrant_mMs"])
    for root, val in conc_vals_mM.items():
        p["conc0"][root] = float(val)
    for tkey, val in titrant_vals_mM.items():
        tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
        p["titrant_mMs"][tfree] = float(val)
    return p


def _make_parsed_with_concs(parsed, conc_vals_mM, titrant_vals_mM):
    """Update parsed concentrations/titrant.
    conc_vals_mM keys are roots (e.g. 'G'); parsed['concentrations'] keys are
    original names (e.g. 'G0'). Map back before updating.
    """
    p = dict(parsed)
    p["concentrations"] = dict(parsed["concentrations"])
    p["titrant"] = dict(parsed["titrant"])
    # Build root→original_name map from parsed concentrations
    _root_to_cname = {}
    for cname in parsed["concentrations"]:
        root = cname[:-1] if cname.endswith("0") else cname
        _root_to_cname[root] = cname
    for root, val in conc_vals_mM.items():
        cname = _root_to_cname.get(root, root)
        p["concentrations"][cname] = float(val)
    for tkey, val in titrant_vals_mM.items():
        p["titrant"][tkey] = float(val)
    return p


def fit_parameters(parsed, network, exp_data, params, logK_vals, fit_keys, x_expr,
                   tolerance=1e-6, maxiter=100_000,
                   use_lbfgsb=True, use_neldermead=True,
                   constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None):
    """
    Fit selected log K / rate constants (and optionally concentrations/titrant) to
    experimental data.

    fit_conc_keys   : list of concentration root names to fit (e.g. ['G'])
    fit_titrant_keys: list of titrant keys to fit (e.g. ['Mt'])

    Returns (success, fitted_logKs, stats, message).
    stats['fitted_concs']    -> {root: mM}
    stats['fitted_titrants'] -> {tkey: mM}
    """
    try:
        fit_conc_keys    = list(fit_conc_keys    or [])
        fit_titrant_keys = list(fit_titrant_keys or [])
        fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)

        logK_names    = list(fit_keys)
        conc_names    = fit_conc_keys
        titrant_names = fit_titrant_keys

        if not exp_data or (not logK_names and not fitting_concs):
            return False, {}, {}, "No experimental data or parameters to fit"

        # ── Solid mode: retrieve column-A header so convert_exp_x can back-convert ──
        _x_col_header = exp_data.get("_x_col_header", "")

        LOGK_MIN, LOGK_MAX = -15.0, 15.0
        CONC_MIN = 0.0

        x0_logK    = [logK_vals[k] for k in logK_names]
        # Seed concentrations from script values (not sidebar) so the optimizer
        # always starts at a neutral point inside the ±20% bounds, not at
        # whatever the user has typed (which may be at or near a boundary).
        _root_to_cname_x0 = {}
        for _cn in parsed.get("concentrations", {}):
            _r = _cn[:-1] if _cn.endswith("0") else _cn
            _root_to_cname_x0[_r] = _cn
        def _script_conc(root):
            cname = _root_to_cname_x0.get(root, root)
            script_val = parsed.get("concentrations", {}).get(cname)
            if script_val is not None:
                return float(script_val)
            return float(params["conc0"].get(root, 1.0))
        x0_conc    = [_script_conc(cn) for cn in conc_names]
        x0_titrant = []
        for tk in titrant_names:
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            x0_titrant.append(params["titrant_mMs"].get(tfree, 10.0))
        x0 = np.array(x0_logK + x0_conc + x0_titrant, dtype=float)

        # bounds
        bounds_logK = [(LOGK_MIN, LOGK_MAX)] * len(logK_names)

        # Map root → original cname (e.g. 'G' → 'G0') for conc_bounds lookup
        _root_to_cname = {}
        for cname in parsed.get("concentrations", {}):
            root = cname[:-1] if cname.endswith("0") else cname
            _root_to_cname[root] = cname

        def _cb(root, default_val, sv_override=None):
            cname = _root_to_cname.get(root, root)
            sv  = sv_override if sv_override is not None else \
                  parsed.get("concentrations", {}).get(cname, default_val)
            lo, hi = parsed.get("conc_bounds", {}).get(cname,
                     parsed.get("conc_bounds", {}).get(root, (None, None)))
            lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, sv * 0.80)
            hi = hi if hi is not None else sv * 1.20
            return (lo, hi)

        def _tb(tkey, default_val, sv_override=None):
            sv  = sv_override if sv_override is not None else \
                  parsed.get("titrant", {}).get(tkey, default_val)
            lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
            lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, sv * 0.80)
            hi = hi if hi is not None else sv * 1.20
            return (lo, hi)


        bounds_conc    = [_cb(cn, x0_conc[i])    for i, cn in enumerate(conc_names)]
        bounds_titrant = [_tb(tk, x0_titrant[i]) for i, tk in enumerate(titrant_names)]

        def _update_bds(fitted_x):
            """Recentre bounds to ±20% of current fitted values after each pass."""
            for _i, _cn in enumerate(conc_names):
                _sv = float(np.clip(fitted_x[n_logK + _i], CONC_MIN + 1e-12, 1e9))
                bounds_conc[_i] = _cb(_cn, _sv, sv_override=_sv)
            for _i, _tk in enumerate(titrant_names):
                _sv = float(np.clip(fitted_x[n_logK + n_conc + _i], CONC_MIN + 1e-12, 1e9))
                bounds_titrant[_i] = _tb(_tk, _sv, sv_override=_sv)
            # Keep bounds list in sync
            nonlocal bounds
            bounds = bounds_logK + bounds_conc + bounds_titrant

        def _make_simplex_steps_fp():
            k_steps = np.full(n_logK, 1.5)
            c_steps = np.array([bounds_conc[i][1]    - bounds_conc[i][0]
                                for i in range(n_conc)])
            t_steps = np.array([bounds_titrant[i][1] - bounds_titrant[i][0]
                                for i in range(len(titrant_names))])
            return np.concatenate([k_steps, c_steps, t_steps]) \
                   if len(c_steps) or len(t_steps) else k_steps

        bounds = bounds_logK + bounds_conc + bounds_titrant

        n_logK = len(logK_names)
        n_conc = len(conc_names)

        def _unpack(fp):
            logKs_d = {logK_names[i]: float(np.clip(fp[i], LOGK_MIN, LOGK_MAX))
                       for i in range(n_logK)}
            conc_d  = {conc_names[i]: float(np.clip(fp[n_logK+i],
                                                     bounds_conc[i][0], bounds_conc[i][1]))
                       for i in range(n_conc)}
            tit_d   = {titrant_names[i]: float(np.clip(fp[n_logK+n_conc+i],
                                                        bounds_titrant[i][0], bounds_titrant[i][1]))
                       for i in range(len(titrant_names))}
            return logKs_d, conc_d, tit_d

        # collect raw experimental data (store volumes not x-values,
        # so we can recompute x when concentrations change)
        exp_points = []
        for exp_col, col_data in exp_data.items():
            if exp_col.startswith("_"):
                continue
            try:
                v_add_mL = col_data["v_add_mL"]
                exp_y    = col_data["y"]
                for i in range(len(v_add_mL)):
                    if np.isfinite(float(v_add_mL[i])) and np.isfinite(float(exp_y[i])):
                        exp_points.append((float(v_add_mL[i]), float(exp_y[i]), exp_col))
            except Exception:
                continue

        if len(exp_points) < len(x0):
            return False, {}, {}, (f"Too few data points ({len(exp_points)}) "
                                   f"for {len(x0)} parameters")

        # static equiv cache keyed by x_val (NOT by v).
        # Keying by v would cause wrong cache hits when an x_val from one
        # point happens to numerically equal the v-value of a different point.
        _static_cache = {}
        if not fitting_concs:
            for v, _, _ in exp_points:
                x_val = convert_exp_x(np.array([v]), x_expr, parsed, params, network,
                                      x_col_header=_x_col_header)[0]
                if x_val not in _static_cache:
                    _static_cache[x_val] = find_equiv_for_x(x_val, parsed, params)

        _constraints = constraints or []

        def objective(fp):
            try:
                logKs_d, conc_d, tit_d = _unpack(fp)
                cur_logKs = logK_vals.copy()
                cur_logKs.update(logKs_d)
                if fitting_concs:
                    cur_params = _make_params_with_concs(params, conc_d, tit_d)
                    cur_parsed = _make_parsed_with_concs(parsed, conc_d, tit_d)
                else:
                    cur_params = params
                    cur_parsed = parsed
                residuals = []
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    for v_add, exp_y, exp_col in exp_points:
                        try:
                            if fitting_concs:
                                x_val = convert_exp_x(np.array([v_add]), x_expr,
                                                      cur_parsed, cur_params, network,
                                                      x_col_header=_x_col_header)[0]
                                equiv = find_equiv_for_x(x_val, cur_parsed, cur_params)
                            else:
                                x_val = convert_exp_x(np.array([v_add]), x_expr,
                                                      parsed, params, network,
                                                      x_col_header=_x_col_header)[0]
                                equiv = _static_cache.get(x_val,
                                            find_equiv_for_x(x_val, parsed, params))
                            theo = compute_single_point(equiv, cur_parsed, network,
                                                        cur_logKs, cur_params, exp_col)
                            residuals.append((exp_y-theo)**2 if np.isfinite(theo) else 1e6)
                        except Exception:
                            residuals.append(1e6)
                if not residuals:
                    return 1e9
                t = float(np.sum(residuals))
                if not np.isfinite(t):
                    return 1e9
                cp = constraints_penalty(_constraints, cur_logKs, ssr_scale=t)
                return t + cp
            except Exception:
                return 1e9

        import time as _time_fp
        obj_start = objective(x0)
        if not np.isfinite(obj_start):
            return False, {}, {}, "Initial objective evaluation failed"

        def _nm_fp(obj_fn, start, steps, max_it=None):
            simplex = np.vstack([start] + [start + np.eye(len(start))[i] * steps[i]
                                            for i in range(len(start))])
            return minimize(obj_fn, start, method='Nelder-Mead',
                            options={'maxiter': max_it or maxiter,
                                     'xatol': max(tolerance ** 0.5, 1e-6),
                                     'fatol': max(tolerance, 1e-8),
                                     'adaptive': True,
                                     'initial_simplex': simplex})

        _logk_steps = np.full(n_logK, 1.5)
        result      = None
        best_x      = x0.copy()

        if not fitting_concs:
            # ── No concentration fitting: original L-BFGS-B + Nelder-Mead ─────
            improved = False
            if use_lbfgsb:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': maxiter, 'ftol': tolerance,
                                           'gtol': tolerance})
                if np.isfinite(objective(result.x)) and objective(result.x) < obj_start * 0.999:
                    best_x = result.x.copy(); improved = True
            if use_neldermead and not improved:
                result = _nm_fp(objective, x0, np.full(len(x0), 1.5))
                if np.isfinite(objective(result.x)) and objective(result.x) < obj_start * 0.999:
                    best_x = result.x.copy()
            if result is None:
                return False, {}, {}, "No optimizer was enabled"

        else:
            # ── Phase 0: warm-start logK before re-fit loop ───────────────────
            if n_logK > 0:
                def _logk_only_obj(lk_vec):
                    return objective(np.concatenate([lk_vec, x0[n_logK:]]))
                _r_p0 = _nm_fp(_logk_only_obj, x0[:n_logK], _logk_steps)
                if use_lbfgsb:
                    try:
                        _r_lb = minimize(_logk_only_obj, x0[:n_logK], method="L-BFGS-B",
                                         bounds=bounds[:n_logK],
                                         options={"maxiter": maxiter, "ftol": tolerance,
                                                  "gtol": tolerance})
                        if np.isfinite(_r_lb.fun) and _r_lb.fun < _r_p0.fun:
                            _r_p0 = _r_lb
                    except Exception:
                        pass
                x0 = np.concatenate([_r_p0.x, x0[n_logK:]])

            # ── Re-fit loop ───────────────────────────────────────────────────
            for _pass_fp in range(10_000):
                _all_steps_fp = _make_simplex_steps_fp()

                result_pass = _nm_fp(objective, x0, _all_steps_fp)

                x0_prev = x0.copy()
                x0 = result_pass.x.copy()
                best_x = x0.copy()
                result = result_pass
                _update_bds(x0)

                # Convergence: parameters stopped moving
                if np.max(np.abs(x0[n_logK:] - x0_prev[n_logK:])) < tolerance:
                    break

        if result is None:
            return False, {}, {}, "No optimizer was enabled"

        logKs_fit, conc_fit, tit_fit = _unpack(best_x)
        fitted_logKs = {k: logKs_fit[k] for k in logK_names}

        # ── statistics ───────────────────────────────────────────────────────
        try:
            final_logKs = logK_vals.copy()
            final_logKs.update(fitted_logKs)
            if fitting_concs:
                fin_params = _make_params_with_concs(params, conc_fit, tit_fit)
                fin_parsed = _make_parsed_with_concs(parsed, conc_fit, tit_fit)
            else:
                fin_params = params
                fin_parsed = parsed

            final_res, theo_vals, exp_vals = [], [], []
            for v_add, exp_y, exp_col in exp_points:
                try:
                    x_val = convert_exp_x(np.array([v_add]), x_expr,
                                          fin_parsed, fin_params, network,
                                          x_col_header=_x_col_header)[0]
                    equiv = find_equiv_for_x(x_val, fin_parsed, fin_params)
                    theo  = compute_single_point(equiv, fin_parsed, network,
                                                 final_logKs, fin_params, exp_col)
                    if np.isfinite(theo):
                        final_res.append((exp_y - theo)**2)
                        theo_vals.append(theo)
                        exp_vals.append(exp_y)
                except Exception:
                    continue

            if not final_res:
                return False, {}, {}, "No valid residuals for statistics"

            ssr  = float(np.sum(final_res))
            rmse = float(np.sqrt(ssr / len(final_res)))
            n_pts = len(exp_vals)
            n_par = len(best_x)
            sst   = float(np.sum((np.array(exp_vals) - np.mean(exp_vals))**2)) if exp_vals else 1e-12
            r2    = float(1.0 - ssr / sst) if sst > 1e-12 else 0.0

            stats = {
                "r_squared":       r2,
                "rmse":            rmse,
                "n_points":        n_pts,
                "n_params":        n_par,
                "ssr":             ssr,
                "n_iter":          int(getattr(result, '_best_nit', result.nit)),
                "param_errors":    {},
                "param_values":    {},
                "fitted_concs":    conc_fit,
                "fitted_titrants": tit_fit,
            }

            # Hessian-based uncertainties (data-only)
            def raw_rvec(fp):
                lgk, cd, td = _unpack(fp)
                lk = logK_vals.copy(); lk.update(lgk)
                if fitting_concs:
                    cp = _make_params_with_concs(params, cd, td)
                    cpd = _make_parsed_with_concs(parsed, cd, td)
                else:
                    cp = params; cpd = parsed
                res = []
                for v_add, exp_y, exp_col in exp_points:
                    try:
                        xv = convert_exp_x(np.array([v_add]), x_expr, cpd, cp, network,
                                           x_col_header=_x_col_header)[0]
                        eq = find_equiv_for_x(xv, cpd, cp)
                        th = compute_single_point(eq, cpd, network, lk, cp, exp_col)
                        res.append(exp_y - th if np.isfinite(th) else 0.0)
                    except Exception:
                        res.append(0.0)
                return np.array(res)

            try:
                if n_pts > n_par:
                    r0  = raw_rvec(best_x)
                    eps = 1e-4
                    J   = np.zeros((n_pts, n_par))
                    for k in range(n_par):
                        dx = np.zeros(n_par); dx[k] = eps
                        J[:, k] = (raw_rvec(best_x + dx) - r0) / eps
                    sigma2 = float(np.dot(r0, r0)) / (n_pts - n_par)
                    JtJ = J.T @ J
                    try:
                        cov = np.linalg.inv(JtJ) * sigma2
                    except np.linalg.LinAlgError:
                        d = np.diag(JtJ)
                        cov = np.diag(np.where(d > 0, sigma2/d, 0.0))
                    all_names = logK_names + conc_names + titrant_names
                    stats["param_errors"] = {
                        all_names[k]: float(np.sqrt(max(cov[k,k], 0.0)))
                        for k in range(n_par)
                    }
            except Exception:
                pass

            for k, pname in enumerate(logK_names):
                stats["param_values"][pname] = float(best_x[k])

            _conv = bool(result is not None and (getattr(result, "success", False) or (objective(best_x) < obj_start * 0.999)))
            return _conv, fitted_logKs, stats, ("Fit successful" if _conv else "Converged with warnings")

        except Exception as e:
            return False, {}, {}, f"Statistics failed: {e}"

    except Exception as e:
        return False, {}, {}, f"Fitting error: {e}"
