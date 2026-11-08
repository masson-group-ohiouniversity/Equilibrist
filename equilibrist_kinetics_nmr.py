# -*- coding: utf-8 -*-
"""equilibrist_kinetics_nmr.py"""
import re
import time
import numpy as np
import equilibrist_live as _live
from scipy.optimize import minimize
from equilibrist_kinetics import compute_kinetics_curve, _collect_all_kinetic_species, kinetic_param_bounds
from equilibrist_fit_nmr import _get_species_for_target, _hessian_errors, _resolve_c
from equilibrist_parser import constraints_penalty, PenaltyScale, FitTrace, simplex_step, tied_minimize
from equilibrist_shift_constraints import solve_shifts, get_per_column_bounds

__all__ = ['_build_kinetics_stoich', '_kinetics_nmr_integration_backCalc', 'fit_kinetics_nmr_shifts', 'fit_kinetics_nmr_integration', 'fit_kinetics_nmr_mixed']


# ─────────────────────────────────────────────
# KINETICS NMR ENGINE
# ─────────────────────────────────────────────

def _build_kinetics_stoich(parsed: dict) -> dict:
    """
    Build stoich dict {(initial_sp, product_sp): count} by tracing composition
    through kinetics reactions — analogous to build_network's count_free() for
    equilibrium.  Used by the NMR integration back-calculator.

    Example: G + H > GH  →  stoich(G, GH)=1, stoich(H, GH)=1
    """
    # Initial species: keys in $concentrations, strip trailing "0"
    init_names = []
    for cname in parsed.get("concentrations", {}):
        root = cname[:-1] if cname.endswith("0") else cname
        if root not in init_names:
            init_names.append(root)

    # composition[sp] = {init_sp: count}
    composition = {sp: {sp: 1.0} for sp in init_names}

    # Process reactions in order; multi-pass handles chains (A>B; B>C)
    changed = True
    for _pass in range(20):
        if not changed: break
        changed = False
        for rxn in parsed.get("equilibria", []) + parsed.get("kinetics", []):
            for prod_coeff, prod_sp in rxn.get("products", []):
                if prod_sp in init_names:
                    continue
                comp = {}
                for react_coeff, react_sp in rxn.get("reactants", []):
                    src = composition.get(react_sp, {react_sp: 1.0})
                    for k, v in src.items():
                        comp[k] = comp.get(k, 0.0) + v * (react_coeff / prod_coeff)
                if composition.get(prod_sp) != comp:
                    composition[prod_sp] = comp
                    changed = True

    stoich = {}
    for sp, comp in composition.items():
        for init_s, cnt in comp.items():
            if cnt > 0:
                stoich[(init_s, sp)] = cnt
    return stoich


def _kinetics_nmr_integration_backCalc(nmr_data: dict, n_H_list: list,
                                        parsed: dict, all_species: list) -> dict:
    """
    Convert NMR integrations to concentrations for kinetics mode.

    Identical algorithm to equilibrium _nmr_integration_backCalc, but:
      - No volume dilution: G_total(t) = G0  (constant)
      - Stoichiometry derived from kinetics reactions via _build_kinetics_stoich

    Model:
        ratio_sp(t) = mean(I_sp / n_H_sp)   [averaged over duplicate signals]
        denom(t)    = Σ stoich(primary, sp) × ratio_sp(t)  [G-containing sp only]
        [G](t)      = G0 / denom(t)
        [Sp](t)     = ratio_sp(t) × [G](t)
    """
    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return {}

    variables_parsed = parsed.get("variables", {})

    # Primary species = first entry in $concentrations (e.g. G0 → G)
    primary_sp = None
    G0_mM = 1.0
    for cname, cval in parsed.get("concentrations", {}).items():
        root = cname[:-1] if cname.endswith("0") else cname
        primary_sp = root
        G0_mM = float(cval)
        break
    if primary_sp is None:
        return {}

    # Build proper stoichiometry from reactions
    kin_stoich = _build_kinetics_stoich(parsed)

    def _effective_stoich(col_sp):
        """Return stoich of primary_sp in col_sp (0 if none)."""
        s = kin_stoich.get((primary_sp, col_sp), 0.0)
        if s > 0:
            return s
        # Also check if col_sp is a $variable — expand and sum member stoichs
        if col_sp in variables_parsed:
            expr = variables_parsed[col_sp]
            total = 0.0; count = 0.0
            for part in expr.split("+"):
                part = part.strip()
                m = re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$", part)
                coeff_m = float(m.group(1)) if m else 1.0
                sp_m    = m.group(2) if m else part
                s_m = kin_stoich.get((primary_sp, sp_m), 0.0)
                if s_m > 0:
                    total += coeff_m * s_m; count += coeff_m
            if count > 0:
                return total / count
        return 0.0

    # ── Build entries: (col, sp_name, n_H, raw_I, t_arr) ────────────────────
    entries = []
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_species and sp not in variables_parsed:
            sp = col.split(".")[0]
        if sp not in all_species and sp not in variables_parsed:
            sp = col
        n_H   = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0
        raw_I = nmr_data[col]["y"]
        t_arr = nmr_data[col]["v_add_mL"]   # time in seconds
        entries.append((col, sp, n_H, raw_I, t_arr))

    # ── Average I/n_H across duplicate signals of each species ───────────────
    # The duplicate columns of a species need not have the same rows filled
    # (blank cells where a signal could not be integrated): they are averaged
    # row by row, over the columns that have a value there.
    from equilibrist_fit_nmr import _merge_duplicate_integrals, _ratio_on_grid, BackCalc
    _dups = {}
    for col, sp, n_H, raw_I, t_arr in entries:
        _dups.setdefault(sp, []).append((t_arr, np.asarray(raw_I, float) / n_H))
    sp_ratio = {}; sp_t = {}
    for sp, cols_ in _dups.items():
        sp_t[sp], sp_ratio[sp] = _merge_duplicate_integrals(cols_)

    # ── G-containing species only go into the denominator ────────────────────
    g_sp_stoich = {sp: _effective_stoich(sp)
                   for sp in sp_ratio if _effective_stoich(sp) > 0}
    if not g_sp_stoich:
        return {}   # stoich failed — no back-calc possible

    # the common grid: every time at which at least one species has an integral
    t_ref_arr = np.unique(np.concatenate([sp_t[sp] for sp in sp_ratio]))
    n_pts     = len(t_ref_arr)

    denom = np.zeros(n_pts)
    for sp, stoich_coeff in g_sp_stoich.items():
        denom += stoich_coeff * _ratio_on_grid(t_ref_arr, sp_t[sp], sp_ratio[sp])
    denom = np.maximum(denom, 1e-20)

    G_conc = G0_mM / denom   # mM

    # [Sp](t) = ratio_sp(t) × G_conc(t)  for ALL measured species, reported only
    # where that species was measured (a blank cell is not a data point)
    result = BackCalc()
    filled = {}
    for sp, ratio_arr in sp_ratio.items():
        at = np.searchsorted(t_ref_arr, sp_t[sp])
        c_bc = np.clip(ratio_arr * G_conc[at], 0.0, None)
        result[sp] = (t_ref_arr[at], c_bc)
        filled[sp] = np.clip(_ratio_on_grid(t_ref_arr, sp_t[sp], ratio_arr) * G_conc, 0.0, None)
    result.filled = (t_ref_arr, filled)
    return result


def fit_kinetics_nmr_shifts(parsed: dict, logk_dict: dict, nmr_data: dict,
                             fit_keys: list, t_max: float, n_pts_sim: int,
                             tolerance: float, maxiter: int,
                             timeout_s: float = 30.0, constraints=None,
                             fit_conc_keys=None,
                             *, compute_hessian: bool = True, _objective_only: bool = False):
    """
    Fit rate constants to fast-exchange NMR chemical shift data (kinetics mode).

    Identical physics to fit_nmr_shifts but uses compute_kinetics_curve
    and time (s) directly as the x-axis instead of volume + convert_exp_x.
    """
    from scipy.optimize import minimize

    nmr_cfg      = parsed["nmr"]
    all_kin_sp   = _collect_all_kinetic_species(parsed)

    # Soft bounds per constant: pre-equilibria -2..14 (or their "from X to Y"),
    # rate constants unrestricted beyond the generic |log k| <= 12 penalty
    _k_lo_kn, _k_hi_kn = kinetic_param_bounds(parsed, fit_keys)
    fake_network = {"all_species": all_kin_sp}

    # ── Column → target mapping ───────────────────────────────────────────────
    col_to_target = {}
    for col in nmr_data:
        if col.startswith("_"): continue
        for tgt in nmr_cfg["targets"]:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = nmr_cfg["targets"][0] if nmr_cfg["targets"] else col

    first_col = next((c for c in nmr_data if not c.startswith("_")), None)
    if first_col is None:
        return False, {}, {}, "No NMR data columns found"

    # δ_free and t_free: first time point per signal
    delta_free = {}
    t_free_val = {}
    for col, col_data in nmr_data.items():
        if col.startswith("_"): continue
        delta_free[col] = float(col_data["y"][0])
        t_free_val[col] = float(col_data["v_add_mL"][0])

    _t_exp_all = np.unique(np.concatenate([np.asarray(_d["v_add_mL"], dtype=float)
                                           for _c, _d in nmr_data.items() if not _c.startswith("_")]))
    def _simulate(params_vec):
        lk_v, cd = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        try:
            return compute_kinetics_curve(_patched_parsed_k(cd), lk, t_max, n_pts_sim, t_extra=_t_exp_all)
        except Exception:
            return None

    def _build_fraction_matrix(c, t_exp, sp_coeffs, t_free=None, include_ref=False):
        """Build design matrix X of mole fractions.
        include_ref=False (legacy): N-1 columns (non-ref species).
        include_ref=True (noref-aware): N columns (all species, ref first)."""
        t_sim  = c["t"]
        n_pts  = len(t_exp)
        denom  = np.zeros(n_pts)
        for coeff, sp in sp_coeffs:
            denom += coeff * np.interp(t_exp, t_sim, c.get(sp, np.zeros_like(t_sim)))
        denom  = np.maximum(denom, 1e-20)
        cols_sp = list(sp_coeffs) if include_ref else list(sp_coeffs[1:])
        X = np.zeros((n_pts, len(cols_sp)))
        for i, (coeff, sp) in enumerate(cols_sp):
            X[:, i] = coeff * np.interp(t_exp, t_sim, c.get(sp, np.zeros_like(t_sim))) / denom
        if t_free is not None:
            denom_ref = max(sum(
                coeff * float(np.interp(t_free, t_sim, c.get(sp, np.zeros_like(t_sim))))
                for coeff, sp in sp_coeffs), 1e-20)
            X_ref = np.array([
                coeff * float(np.interp(t_free, t_sim,
                              c.get(sp, np.zeros_like(t_sim)))) / denom_ref
                for coeff, sp in cols_sp])
            X = X - X_ref[np.newaxis, :]
        return X

    _shift_constraints = parsed.get("shift_constraints", []) or []

    # Per-column bounds map (the 'shift: -0.8, free, 1.0, ...' syntax)
    _per_col_bounds = get_per_column_bounds(_shift_constraints)
    _col_to_bound = {}
    if _per_col_bounds is not None:
        _shift_cols_in_order = [c for c in nmr_data if not c.startswith("_")]
        if len(_per_col_bounds) != len(_shift_cols_in_order):
            return False, {}, {}, (
                f"Per-column shift bounds list has {len(_per_col_bounds)} entries "
                f"but the data has {len(_shift_cols_in_order)} shift columns. "
                f"Use 'free' for columns you don't want to constrain."
            )
        _col_to_bound = dict(zip(_shift_cols_in_order, _per_col_bounds))

    # Pinned intrinsic shifts ($nmr 'read:' species + sheet-2 known shifts)
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _col_to_pinned = {}
    if _read_species:
        _shift_cols_in_order = [c for c in nmr_data if not c.startswith("_")]
        for sp in _read_species:
            if sp not in _known_shifts:
                return False, {}, {}, (
                    f"$nmr 'read: {sp}' but species '{sp}' not found in sheet 2 of "
                    f"the data file (expected row with '{sp}' in column A and "
                    f"{len(_shift_cols_in_order)} chemical-shift values to its right)."
                )
            vals = _known_shifts[sp]
            if len(vals) != len(_shift_cols_in_order):
                return False, {}, {}, (
                    f"$nmr 'read: {sp}': sheet 2 has {len(vals)} values for '{sp}' "
                    f"but the data has {len(_shift_cols_in_order)} shift columns. "
                    f"Provide one value per shift column, in the same order."
                )
            for k, col in enumerate(_shift_cols_in_order):
                dd_val = float(vals[k]) - float(delta_free[col])
                _col_to_pinned.setdefault(col, {})[sp] = dd_val

    # noref flag (default False = legacy auto-pin to ref)
    _nmr_noref = bool(nmr_cfg.get("noref", False))

    def _analytic_delta(X, dobs_rel, sp_coeffs=None, column_bound=None,
                        pinned_dd=None):
        """Solve per-column shift sub-problem.  X must contain all species
        columns (build with include_ref=True)."""
        if X.shape[1] == 0:
            return np.array([]), np.zeros_like(dobs_rel), float(np.sum(dobs_rel**2))
        if sp_coeffs is None:
            dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((dobs_rel - calc)**2))
        species = [sp for _, sp in sp_coeffs]
        ref_sp  = sp_coeffs[0][1]
        try:
            return solve_shifts(X, dobs_rel, species, ref_sp,
                                _shift_constraints,
                                column_bound=column_bound,
                                pinned_dd=pinned_dd,
                                noref=_nmr_noref)
        except ValueError:
            dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((dobs_rel - calc)**2))

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    _trace = FitTrace()   # how did the fit end? (see equilibrist_parser)
    def objective(params_vec):
        _trace(params_vec)
        lk_v, _ = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        c = _simulate(params_vec)
        if c is None: return 1e12
        total_ssr = 0.0
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, fake_network)
            if not sp_coeffs: continue
            t_exp         = col_data["v_add_mL"]
            delta_obs_rel = col_data["y"] - delta_free[col]
            X = _build_fraction_matrix(c, t_exp, sp_coeffs, t_free=t_free_val[col], include_ref=True)
            _, _, ssr = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            total_ssr += ssr
        # Penalty for log-K outside per-K bounds (from script's 'from X to Y')
        bound_penalty = 0.0
        for kv, _lo, _hi in zip(lk_v, _k_lo_kn, _k_hi_kn):
            if kv < _lo: bound_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: bound_penalty += 1e6 * (kv - _hi) ** 2
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(total_ssr))
        _live.notify(lk, total_ssr + cp + bound_penalty)
        return total_ssr + cp + bound_penalty

    def data_objective(params_vec):
        """Data-only SSR (no constraint penalty) — used for Hessian / error estimation."""
        c = _simulate(params_vec)
        if c is None: return 1e12
        total_ssr = 0.0
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, fake_network)
            if not sp_coeffs: continue
            t_exp         = col_data["v_add_mL"]
            delta_obs_rel = col_data["y"] - delta_free[col]
            X = _build_fraction_matrix(c, t_exp, sp_coeffs, t_free=t_free_val[col], include_ref=True)
            _, _, ssr = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            total_ssr += ssr
        return total_ssr

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

    x0  = np.concatenate([np.array([logk_dict[k] for k in fit_keys]), _x0_c])
    n_p = len(x0)

    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    if _objective_only:
        
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float)))),
            "data_ssr":  None,
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys),
            "n_log_params": len(fit_keys),
            "n_data": int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_'))),
        }
    # Global timer starts here — covers Phase 1 AND main optimisation
    _fit_start = time.time()

    # Phase 1: warm-start logK before joint optimisation
    if _n_c > 0 and _n_k > 0:
        class _TimeoutP1(Exception): pass
        def _phase1_obj_p(lk_vec, _t0=_fit_start, _tlim=timeout_s):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP1()
            return objective(np.concatenate([lk_vec, x0[_n_k:]]))
        _sp1 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                         for i in range(_n_k)])
        try:
            _r1 = tied_minimize(_phase1_obj_p, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logk_dict,
                           options={"maxiter": maxiter//10,
                                    "xatol": tolerance,
                                    "fatol": tolerance * 1e-4,
                                    "adaptive": True, "initial_simplex": _sp1})
            x0 = np.concatenate([_r1.x, x0[_n_k:]])
        except (_TimeoutP1, Exception):
            pass

    _steps = np.array([(1e-9 if maxiter <= 1 else simplex_step(1.5))] * _n_k + [max(abs(_x0_c[i]) * 0.1, 0.05)
                       for i in range(_n_c)])
    init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _steps[i]
                                     for i in range(n_p)])

    class _Timeout(Exception): pass
    best_tracker = {"x": x0.copy(), "f": np.inf, "start": _fit_start, "nit": 0}

    def _obj_timed(logk_trial):
        best_tracker["nit"] += 1
        f = objective(logk_trial)
        if f < best_tracker["f"]: best_tracker["f"] = f; best_tracker["x"] = logk_trial.copy()
        if time.time() - best_tracker["start"] > timeout_s: raise _Timeout()
        return f

    try:
        result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys), constraints=constraints or [], fixed=logk_dict,
                          options={"maxiter": maxiter, "xatol": tolerance,
                                   "fatol": tolerance * 1e-4, "adaptive": True,
                                   "initial_simplex": init_simplex})
    except _Timeout:
        class _MockResult:
            x = best_tracker["x"]; success = False; fun = best_tracker["f"]
            nit = best_tracker["nit"]
        result = _MockResult()
    fitted_logks   = {fit_keys[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1])) for i in range(_n_c)}

    # ── Final pass: Δδ vectors and statistics ────────────────────────────────
    c_final = _simulate(result.x)
    delta_vecs_all  = {}
    delta_bound_all = {}
    ref_corrections = {}
    pure_shifts     = {}
    all_residuals   = []
    all_y_obs       = []
    # v2: per-column arrays for bootstrap resampling
    per_col_y_obs   = {}
    per_col_y_calc  = {}
    per_col_x       = {}

    if c_final is not None:
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, fake_network)
            if not sp_coeffs: continue
            t_exp         = col_data["v_add_mL"]
            dobs_rel      = col_data["y"] - delta_free[col]
            X  = _build_fraction_matrix(c_final, t_exp, sp_coeffs, t_free=t_free_val[col], include_ref=True)
            dd, calc_rel, _ = _analytic_delta(X, dobs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            all_residuals.extend((dobs_rel - calc_rel).tolist())
            all_y_obs.extend(dobs_rel.tolist())
            per_col_y_obs[col]  = np.asarray(dobs_rel, dtype=float)
            per_col_y_calc[col] = np.asarray(calc_rel, dtype=float)
            per_col_x[col]      = np.asarray(t_exp,    dtype=float)
            # dd[i] = δ_i_absolute − δ_obs(t_free) for every species (including ref).
            # Legacy auto-pin sets dd[ref]=0; noref leaves dd[ref] as a free fit param.
            sp_dd = {sp: float(dd[i]) if i < len(dd) else 0.0
                     for i, (_, sp) in enumerate(sp_coeffs)}
            # Normalize when not anchored: subtract dd[ref] so legacy and
            # noref-no-read produce identical "Δδ relative to ref" output.
            if not (_nmr_noref and _read_species):
                ref_sp = sp_coeffs[0][1]
                ref_dd = sp_dd.get(ref_sp, 0.0)
                if abs(ref_dd) > 1e-12:
                    sp_dd = {k: v - ref_dd for k, v in sp_dd.items()}
            delta_vecs_all[col] = sp_dd
            if len(sp_coeffs) == 2:
                non_ref_sp = sp_coeffs[1][1]
                delta_bound_all[col] = sp_dd.get(non_ref_sp, 0.0)
            # Pure-species shifts stored as dd values; display/export adds
            # δ_obs(t_free) when noref+read: anchors the absolute scale.
            # ref_corrections[col] = Σᵢ Fᵢ(t_ref)·ddᵢ over ALL species (incl. ref).
            t_sim_rc = c_final["t"]
            denom_ref_rc = max(sum(
                coeff * float(np.interp(t_free_val[col], t_sim_rc,
                              c_final.get(sp, np.zeros_like(t_sim_rc))))
                for coeff, sp in sp_coeffs), 1e-20)
            ref_corrections[col] = sum(
                coeff * float(np.interp(t_free_val[col], t_sim_rc,
                              c_final.get(sp, np.zeros_like(t_sim_rc)))) / denom_ref_rc
                * sp_dd.get(sp, 0.0)
                for coeff, sp in sp_coeffs   # all species, incl. ref
            )
            if tgt not in pure_shifts:
                pure_shifts[tgt] = {}
            pure_shifts[tgt][col] = dict(sp_dd)

    residuals = np.array(all_residuals)
    y_obs     = np.array(all_y_obs)
    ssr  = float(np.sum(residuals**2))
    sst  = float(np.sum((y_obs - y_obs.mean())**2)) if len(y_obs) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))
    param_errors = {}
    _cov_mat = None
    _cov_names = list(fit_keys) + list(fit_conc_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(data_objective, result.x, ssr, len(residuals), n_p)
        for _i in range(min(len(fit_keys), n_p)):
            if _i in _err_idx: param_errors[fit_keys[_i]] = _err_idx[_i]
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors[fit_conc_keys[_i]] = _err_idx[_idx]

    stats = {
        "r_squared": r2, "rmse": rmse, "ssr": ssr,
        "n_points": len(residuals), "n_params": n_p,
        "param_values": fitted_logks, "param_errors": param_errors,
        "param_cov": _cov_mat, "param_cov_names": _cov_names,
        "delta_vecs_all": delta_vecs_all, "delta_bound_all": delta_bound_all,
        "delta_free": delta_free, "x_free_val": t_free_val,
        "col_to_target": col_to_target, "ref_corrections": ref_corrections,
        "fit_mode": "shift", "is_kinetics": True, "n_iter": getattr(result, "nit", 0),
        "timed_out": not getattr(result, "success", True),
        "sp_concs": {}, "col_to_sp": {}, "col_to_nH": {}, "pure_shifts": pure_shifts,
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "fitted_concs": fitted_concs_k, "fitted_titrants": {},
        # v2 diagnostics — flat residual arrays
        "y_obs":     np.asarray(y_obs,             dtype=float),
        "y_calc":    np.asarray(y_obs - residuals, dtype=float),
        "residuals": np.asarray(residuals,         dtype=float),
        # v2 bootstrap — per-column arrays
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to = stats["timed_out"]
    if _to and r2 >= 0.99:
        stats["timed_out"] = False
    _conv = result.success or ssr < 1e-6 or (_to and r2 >= 0.99)
    stats.update(_trace.summary(result, not getattr(result, "success", True), tolerance))
    return _conv, fitted_logks, stats, "Kinetics NMR shift fit complete"


def fit_kinetics_nmr_integration(parsed: dict, logk_dict: dict, nmr_data: dict,
                                  fit_keys: list, t_max: float, n_pts_sim: int,
                                  tolerance: float, maxiter: int,
                                  timeout_s: float = 30.0, constraints=None,
                                  fit_conc_keys=None,
                                  *, compute_hessian: bool = True, _objective_only: bool = False):
    """Fit rate constants to slow-exchange NMR integration data (kinetics mode)."""
    from scipy.optimize import minimize

    nmr_cfg    = parsed["nmr"]
    n_H_list   = nmr_cfg.get("n_H_list", [])
    all_kin_sp = _collect_all_kinetic_species(parsed)

    # Soft bounds per constant: pre-equilibria -2..14 (or their "from X to Y"),
    # rate constants unrestricted beyond the generic |log k| <= 12 penalty
    _k_lo_kn, _k_hi_kn = kinetic_param_bounds(parsed, fit_keys)

    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return False, {}, {}, "No integration data columns found"

    bc = _kinetics_nmr_integration_backCalc(nmr_data, n_H_list, parsed, all_kin_sp)
    if not bc:
        return False, {}, {}, "Back-calculation failed"

    col_to_sp = {}; col_to_nH = {}
    _vars_kin = parsed.get("variables", {}) or {}
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_kin_sp and sp not in _vars_kin: sp = col.split(".")[0]
        if sp not in all_kin_sp and sp not in _vars_kin: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    _t_exp_all = np.unique(np.concatenate([np.asarray(_d["v_add_mL"], dtype=float)
                                           for _c, _d in nmr_data.items() if not _c.startswith("_")]))
    def _simulate(params_vec):
        lk_v, cd = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        try:
            return compute_kinetics_curve(_patched_parsed_k(cd), lk, t_max, n_pts_sim, t_extra=_t_exp_all)
        except Exception:
            return None

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    _trace = FitTrace()   # how did the fit end? (see equilibrist_parser)
    def objective(params_vec, bc_frozen):
        _trace(params_vec)
        # bc_frozen is fixed for the entire optimizer pass (computed once at the
        # top of each re-fit loop pass from the current x0).  This prevents the
        # moving-target problem: when G0 is free, calling bc(G0_trial) inside the
        # optimizer makes both theory and target shift together, collapsing the
        # objective.  bc_frozen breaks this coupling.
        lk_v2, _ = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v2[i]
        c = _simulate(params_vec)
        if c is None: return 1e12
        t_sim = c["t"]
        ssr = 0.0
        for sp, (t_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(t_bc, t_sim, _resolve_c(c, sp, parsed, t_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        # Penalty for log-K outside per-K bounds
        bound_penalty = 0.0
        for kv, _lo, _hi in zip(lk_v2, _k_lo_kn, _k_hi_kn):
            if kv < _lo: bound_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: bound_penalty += 1e6 * (kv - _hi) ** 2
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(ssr))
        _live.notify(lk, ssr + cp + bound_penalty)
        return ssr + cp + bound_penalty

    def data_objective(params_vec, bc_frozen):
        """Data-only SSR (no constraint penalty) — used for Hessian / error estimation."""
        c = _simulate(params_vec)
        if c is None: return 1e12
        t_sim = c["t"]
        ssr = 0.0
        for sp, (t_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(t_bc, t_sim, _resolve_c(c, sp, parsed, t_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        return ssr

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

    def _cb_k(root, sv_override=None):
        _cn = _root_to_cname_k.get(root, root)
        sv  = sv_override if sv_override is not None \
              else float(parsed.get("concentrations", {}).get(_cn, 1.0))
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

    def _update_bds_c(fitted_x):
        """Recompute bounds from current fitted concentrations — mirrors what
        app.py does between clicks: sidebar updates to fitted value, so bounds
        shift to \u00b120% around the new value (unless explicitly set in script)."""
        for _i, _r in enumerate(fit_conc_keys):
            _sv_now = float(np.clip(fitted_x[_n_k + _i],
                                    CONC_MIN_K + 1e-12, 1e9))
            _bds_c[_i] = _cb_k(_r, sv_override=_sv_now)


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


    def _get_bc(params_vec):
        """Return back-calculated concentrations.
        Recomputed at each iteration when fitting concentrations (G0 changes bc);
        cached otherwise — identical pattern to fit_nmr_integration."""
        if _n_c == 0:
            return bc
        _, cd = _unpack_ck(params_vec)
        p_patched = _patched_parsed_k(cd)
        return _kinetics_nmr_integration_backCalc(
            nmr_data, n_H_list, p_patched, all_kin_sp) or bc

    x0  = np.concatenate([np.array([logk_dict[k] for k in fit_keys]), _x0_c])
    n_p = len(x0)

    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    if _objective_only:
        
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float), _get_bc(np.asarray(t, dtype=float))))),
            "data_ssr":  None,
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys),
            "n_log_params": len(fit_keys),
            "n_data": int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_'))),
        }
    # Global timer starts here — covers Phase 1 AND the re-fit loop
    _global_start  = time.time()

    # Phase 1: warm-start logK before joint optimisation (run once, outside the re-fit loop)
    if _n_c > 0 and _n_k > 0:
        class _TimeoutP1(Exception): pass
        def _phase1_obj_p(lk_vec, _t0=_global_start, _tlim=timeout_s):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP1()
            _bc_ph1 = _get_bc(x0)  # frozen for this warm-start pass
            return objective(np.concatenate([lk_vec, x0[_n_k:]]), _bc_ph1)
        _sp1 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                         for i in range(_n_k)])
        try:
            _r1 = tied_minimize(_phase1_obj_p, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logk_dict,
                           options={"maxiter": maxiter//10,
                                    "xatol": tolerance,
                                    "fatol": tolerance * 1e-4,
                                    "adaptive": True, "initial_simplex": _sp1})
            x0 = np.concatenate([_r1.x, x0[_n_k:]])
        except (_TimeoutP1, Exception):
            pass

    # ── Quick R² helper (no full stats overhead) ──────────────────────────────
    def _quick_r2(params_vec, bc_frozen):
        c = _simulate(params_vec)
        if c is None:
            return -np.inf
        t_sim = c["t"]
        res, obs = [], []
        for sp, (t_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(t_bc, t_sim, _resolve_c(c, sp, parsed, t_sim))
            res.extend((c_bc - c_th).tolist())
            obs.extend(c_bc.tolist())
        if not obs:
            return -np.inf
        r = np.array(res); o = np.array(obs)
        sst_ = float(np.sum((o - o.mean()) ** 2)) if len(o) > 1 else 1.0
        return 1.0 - float(np.sum(r ** 2)) / max(sst_, 1e-30)


    def _make_simplex_steps():
        # Mathematically sound step size: span the full feasible range per dimension.
        # step_i = hi_i - lo_i guarantees the initial simplex vertex in dimension i
        # reaches the far boundary of the search space, so no feasible basin can be
        # missed regardless of where x0 sits.  Any smaller fraction is arbitrary and
        # can leave unreachable basins.
        # logK dims are unbounded here; 1.5 is a practical heuristic for log-space.
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else simplex_step(1.5)))
        c_steps = np.array([_bds_c[i][1] - _bds_c[i][0] for i in range(_n_c)])
        return np.concatenate([k_steps, c_steps]) if _n_c else k_steps

    # ── Re-fit loop: only active when concentrations are being fitted ───────────
    # Each pass: freeze bc from current x0, optimise, update x0 and bounds.
    # Stopping: global timeout OR max passes. No convergence heuristic — each
    # near-converged pass is very fast so running all _MAX_REFIT costs little.
    # When _n_c == 0, n_passes = 1 (constants-only, original behaviour).
    n_passes   = 50 if _n_c > 0 else 1  # timeout is real limit; convergence exits early

    best_x_global  = x0.copy()
    _total_nit     = 0
    _any_timed_out = False
    result         = None
    # _global_start already set before Phase 1 — do not reset here

    for _pass in range(n_passes):
        # ── Time budget ──────────────────────────────────────────────────────
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True
            break

        # ── Freeze bc from current x0 for this entire pass ──────────────────
        bc_pass = _get_bc(x0)

        _base_steps = _make_simplex_steps()
        if _pass > 0 and _n_c > 0:
            _base_steps[:_n_k] = 0.1  # K near optimum after pass 0; stay local
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])

        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0}

        def _obj_timed(params_trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            f = objective(params_trial, _bc)
            if f < _tracker["f"]:
                _tracker["f"] = f
                _tracker["x"] = params_trial.copy()
            if time.time() - _t0 > _tlim:
                raise _Timeout()
            return f

        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys), constraints=constraints or [], fixed=logk_dict,
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult()
            _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])

        if _n_c > 0:
            # Always take the latest result — each pass warm-starts from the
            # previous so results are monotonically improving.  Never compare
            # SSR across passes: each uses a different frozen bc so the values
            # are on different scales and incomparable.
            x0_prev = x0.copy()
            x0 = result.x.copy()   # warm-start next pass
            best_x_global = x0.copy()
            _update_bds_c(x0)      # shift bounds to ±20% of new fitted values
            if _any_timed_out:
                break
            # Stop when concentration parameters no longer move — genuinely converged
            _max_move = np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:]))
            if _max_move < tolerance:
                break
        else:
            best_x_global = result.x.copy()
    # Point result at the global best parameter vector
    _bc_final_for_r2 = _get_bc(best_x_global)
    _r2_best         = _quick_r2(best_x_global, _bc_final_for_r2)
    class _BestResult:
        x       = best_x_global
        success = (_r2_best >= 0.99
                   or (result is not None and getattr(result, "success", False)))
        fun     = getattr(result, 'fun', np.inf)
        nit     = _total_nit
    result = _BestResult()


    fitted_logks   = {fit_keys[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1])) for i in range(_n_c)}

    c_final  = _simulate(result.x)
    bc_final = _get_bc(result.x)
    all_res = []; all_obs = []
    # v2: per-species arrays
    per_col_y_obs   = {}
    per_col_y_calc  = {}
    per_col_x       = {}
    if c_final is not None:
        t_sim_f = c_final["t"]
        for sp, (t_bc, c_bc) in bc_final.items():
            c_th = np.interp(t_bc, t_sim_f, _resolve_c(c_final, sp, parsed, t_sim_f))
            all_res.extend((c_bc - c_th).tolist())
            all_obs.extend(c_bc.tolist())
            per_col_y_obs[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc[sp] = np.asarray(c_th, dtype=float)
            per_col_x[sp]      = np.asarray(t_bc, dtype=float)

    residuals = np.array(all_res); y_obs_arr = np.array(all_obs)
    ssr  = float(np.sum(residuals**2))
    sst  = float(np.sum((y_obs_arr - y_obs_arr.mean())**2)) if len(y_obs_arr) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))
    _bc_final_frozen = bc_final  # frozen for Hessian — consistent with last pass
    param_errors = {}
    _cov_mat = None
    _cov_names = list(fit_keys) + list(fit_conc_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final_frozen),
                                   result.x, ssr, len(residuals), n_p)
        for _i in range(min(len(fit_keys), n_p)):
            if _i in _err_idx: param_errors[fit_keys[_i]] = _err_idx[_i]
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors[fit_conc_keys[_i]] = _err_idx[_idx]
    sp_concs = {sp: [(t_bc, c_bc)] for sp, (t_bc, c_bc) in bc_final.items()}

    stats = {
        "r_squared": r2, "rmse": rmse, "ssr": ssr,
        "n_points": len(residuals), "n_params": n_p,
        "param_values": fitted_logks, "param_errors": param_errors,
        "param_cov": _cov_mat, "param_cov_names": _cov_names,
        "sp_concs": sp_concs, "col_to_sp": col_to_sp, "col_to_nH": col_to_nH,
        "fit_mode": "integration", "is_kinetics": True, "n_iter": getattr(result, "nit", 0),
        "timed_out": not getattr(result, "success", True),
        "pure_shifts": {}, "delta_vecs_all": {}, "delta_bound_all": {},
        "delta_free": {}, "x_free_val": {}, "col_to_target": {}, "ref_corrections": {},
        "fitted_concs": fitted_concs_k, "fitted_titrants": {},
        # v2 diagnostics — flat residual arrays
        "y_obs":     np.asarray(y_obs_arr,             dtype=float),
        "y_calc":    np.asarray(y_obs_arr - residuals, dtype=float),
        "residuals": np.asarray(residuals,             dtype=float),
        # v2 bootstrap — per-species arrays
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to = stats["timed_out"]
    if _to and r2 >= 0.99: stats["timed_out"] = False
    _conv = result.success or ssr < 1e-6 or (_to and r2 >= 0.99)
    stats.update(_trace.summary(result, _any_timed_out, tolerance))
    return _conv, fitted_logks, stats, "Kinetics NMR integration fit complete"



def fit_kinetics_nmr_mixed(parsed: dict, logk_dict: dict, nmr_data: dict,
                            fit_keys: list, t_max: float, n_pts_sim: int,
                            tolerance: float, maxiter: int,
                            timeout_s: float = 30.0, constraints=None,
                            fit_conc_keys=None,
                            *, compute_hessian: bool = True, _objective_only: bool = False):
    """Fit rate constants to mixed slow+fast exchange NMR data (kinetics mode)."""
    from scipy.optimize import minimize

    nmr_cfg       = parsed["nmr"]
    n_H_list      = nmr_cfg.get("n_H_list", [])
    n_integ       = nmr_cfg.get("n_integ", len(n_H_list))
    shift_targets = nmr_cfg.get("targets", [])
    all_kin_sp    = _collect_all_kinetic_species(parsed)

    # Soft bounds per constant: pre-equilibria -2..14 (or their "from X to Y"),
    # rate constants unrestricted beyond the generic |log k| <= 12 penalty
    _k_lo_kn, _k_hi_kn = kinetic_param_bounds(parsed, fit_keys)
    fake_network  = {"all_species": all_kin_sp}
    _shift_constraints = parsed.get("shift_constraints", []) or []

    all_cols   = [col for col in nmr_data if not col.startswith("_")]
    integ_cols = all_cols[:n_integ]
    shift_cols = all_cols[n_integ:]
    integ_data = {col: nmr_data[col] for col in integ_cols}
    shift_data = {col: nmr_data[col] for col in shift_cols}

    if not integ_cols and not shift_cols:
        return False, {}, {}, "No NMR data columns found"

    # ── Per-column shift bounds (the 'shift: -0.8, free, 1.0, ...' syntax) ──
    _per_col_bounds = get_per_column_bounds(_shift_constraints)
    _col_to_bound = {}
    if _per_col_bounds is not None:
        if len(_per_col_bounds) != len(shift_cols):
            return False, {}, {}, (
                f"Per-column shift bounds list has {len(_per_col_bounds)} entries "
                f"but the data has {len(shift_cols)} shift columns. "
                f"Use 'free' for columns you don't want to constrain."
            )
        _col_to_bound = dict(zip(shift_cols, _per_col_bounds))

    bc = {}
    if integ_cols:
        bc = _kinetics_nmr_integration_backCalc(
            integ_data, n_H_list[:n_integ], parsed, all_kin_sp)

    col_to_target = {}
    for col in shift_cols:
        for tgt in shift_targets:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = shift_targets[0] if shift_targets else col

    delta_free = {}; t_free_val = {}
    for col in shift_cols:
        delta_free[col] = float(nmr_data[col]["y"][0])
        t_free_val[col] = float(nmr_data[col]["v_add_mL"][0])

    # ── Pinned intrinsic shifts ($nmr 'read:' + sheet-2 known shifts) ─────
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _col_to_pinned = {}
    if _read_species:
        for sp in _read_species:
            if sp not in _known_shifts:
                return False, {}, {}, (
                    f"$nmr 'read: {sp}' but species '{sp}' not found in sheet 2 of "
                    f"the data file (expected row with '{sp}' in column A and "
                    f"{len(shift_cols)} chemical-shift values to its right)."
                )
            vals = _known_shifts[sp]
            if len(vals) != len(shift_cols):
                return False, {}, {}, (
                    f"$nmr 'read: {sp}': sheet 2 has {len(vals)} values for '{sp}' "
                    f"but the data has {len(shift_cols)} shift columns. "
                    f"Provide one value per shift column, in the same order."
                )
            for k, col in enumerate(shift_cols):
                dd_val = float(vals[k]) - float(delta_free[col])
                _col_to_pinned.setdefault(col, {})[sp] = dd_val

    # noref flag (default False = legacy auto-pin to ref)
    _nmr_noref = bool(nmr_cfg.get("noref", False))

    _t_exp_all = np.unique(np.concatenate([np.asarray(_d["v_add_mL"], dtype=float)
                                           for _c, _d in nmr_data.items() if not _c.startswith("_")]))
    def _simulate(params_vec):
        lk_v, cd = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        try:
            return compute_kinetics_curve(_patched_parsed_k(cd), lk, t_max, n_pts_sim, t_extra=_t_exp_all)
        except Exception:
            return None

    def _shift_ssr(c):
        t_sim  = c["t"]
        total  = 0.0
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, fake_network)
            if not sp_coeffs: continue
            t_exp    = nmr_data[col]["v_add_mL"]
            t_free_c = t_free_val[col]
            denom_full = np.maximum(
                sum(coeff * np.interp(t_exp, t_sim, c.get(sp, np.zeros_like(t_sim)))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(t_free_c, t_sim, c.get(sp, np.zeros_like(t_sim))))
                for coeff, sp in sp_coeffs), 1e-20)
            # Build full design matrix (all species, ref first)
            all_target_sp = list(sp_coeffs)
            n_pts    = len(t_exp)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(t_exp, t_sim, c.get(sp, np.zeros_like(t_sim))) / denom_full
                F_ref  = coeff * float(np.interp(t_free_c, t_sim,
                         c.get(sp, np.zeros_like(t_sim)))) / denom_ref
                X[:, i] = F_full - F_ref
            dobs_rel = nmr_data[col]["y"] - delta_free[col]
            _col_bd = _col_to_bound.get(col)
            _col_pn = _col_to_pinned.get(col)
            if X.shape[1] > 0 and np.any(np.abs(X) > 1e-15):
                try:
                    _, calc, _ssr = solve_shifts(
                        X, dobs_rel,
                        [sp for _, sp in all_target_sp], sp_coeffs[0][1],
                        _shift_constraints,
                        column_bound=_col_bd, pinned_dd=_col_pn,
                        noref=_nmr_noref,
                    )
                    total += float(np.sum((dobs_rel - calc)**2))
                except ValueError:
                    dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
                    total += float(np.sum((dobs_rel - X @ dd)**2))
            else:
                total += float(np.sum(dobs_rel**2))
        return total

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    _trace = FitTrace()   # how did the fit end? (see equilibrist_parser)
    def objective(params_vec, bc_frozen):
        _trace(params_vec)
        lk_v, _ = _unpack_ck(params_vec)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = lk_v[i]
        c = _simulate(params_vec)
        if c is None: return 1e12
        t_sim = c["t"]
        integ_ssr = 0.0
        integ_var = max(sum(float(np.var(c_bc)) for _, (_, c_bc) in bc_frozen.items()), 1e-20)
        for sp, (t_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(t_bc, t_sim, _resolve_c(c, sp, parsed, t_sim))
            integ_ssr += float(np.sum((c_bc - c_th)**2)) / integ_var
        shift_ssr_raw = _shift_ssr(c)
        shift_var = max(sum(float(np.var(nmr_data[col2]["y"]))
                            for col2 in shift_cols), 1e-20) if shift_cols else 1.0
        total_ssr = integ_ssr + shift_ssr_raw / shift_var
        # Penalty for log-K outside per-K bounds (from script's 'from X to Y')
        bound_penalty = 0.0
        for kv, _lo, _hi in zip(lk_v, _k_lo_kn, _k_hi_kn):
            if kv < _lo: bound_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: bound_penalty += 1e6 * (kv - _hi) ** 2
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(total_ssr))
        _live.notify(lk, total_ssr + cp + bound_penalty)
        return total_ssr + cp + bound_penalty

    def data_objective(params_vec, bc_frozen):
        """Data-only SSR (no constraint penalty) — used for Hessian / error estimation."""
        c = _simulate(params_vec)
        if c is None: return 1e12
        t_sim = c["t"]
        integ_ssr = 0.0
        integ_var = max(sum(float(np.var(c_bc)) for _, (_, c_bc) in bc_frozen.items()), 1e-20)
        for sp, (t_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(t_bc, t_sim, _resolve_c(c, sp, parsed, t_sim))
            integ_ssr += float(np.sum((c_bc - c_th)**2)) / integ_var
        shift_ssr_raw = _shift_ssr(c)
        shift_var = max(sum(float(np.var(nmr_data[col2]["y"]))
                            for col2 in shift_cols), 1e-20) if shift_cols else 1.0
        return integ_ssr + shift_ssr_raw / shift_var

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

    def _cb_k(root, sv_override=None):
        _cn = _root_to_cname_k.get(root, root)
        sv  = sv_override if sv_override is not None \
              else float(parsed.get("concentrations", {}).get(_cn, 1.0))
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

    def _update_bds_c(fitted_x):
        for _i, _r in enumerate(fit_conc_keys):
            _sv_now = float(np.clip(fitted_x[_n_k + _i], CONC_MIN_K + 1e-12, 1e9))
            _bds_c[_i] = _cb_k(_r, sv_override=_sv_now)

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

    def _get_bc_mixed(params_vec):
        """Recompute integration bc when fitting concentrations; cached otherwise."""
        if _n_c == 0 or not integ_cols:
            return bc
        _, cd = _unpack_ck(params_vec)
        p_patched = _patched_parsed_k(cd)
        return _kinetics_nmr_integration_backCalc(
            integ_data, n_H_list[:n_integ], p_patched, all_kin_sp) or bc

    def _make_simplex_steps():
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else simplex_step(1.5)))
        c_steps = np.array([_bds_c[i][1] - _bds_c[i][0] for i in range(_n_c)])
        return np.concatenate([k_steps, c_steps]) if _n_c else k_steps

    x0  = np.concatenate([np.array([logk_dict[k] for k in fit_keys]), _x0_c])
    n_p = len(x0)


    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    if _objective_only:
        
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float), _get_bc_mixed(np.asarray(t, dtype=float))))),
            "data_ssr":  None,
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys),
            "n_log_params": len(fit_keys),
            "n_data": int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_'))),
        }
    # Global timer starts here — covers Phase 1 AND the re-fit loop
    _global_start  = time.time()

    # Phase 1: warm-start logK before joint optimisation
    if _n_c > 0 and _n_k > 0:
        _bc_ph1 = _get_bc_mixed(x0)
        class _TimeoutP1(Exception): pass
        def _phase1_obj_p(lk_vec, _t0=_global_start, _tlim=timeout_s, _bc=_bc_ph1):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP1()
            return objective(np.concatenate([lk_vec, x0[_n_k:]]), _bc)
        _sp1 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                         for i in range(_n_k)])
        try:
            _r1 = tied_minimize(_phase1_obj_p, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logk_dict,
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp1})
            x0 = np.concatenate([_r1.x, x0[_n_k:]])
        except (_TimeoutP1, Exception):
            pass

    # ── Re-fit loop ─────────────────────────────────────────────────────────
    n_passes       = 200 if _n_c > 0 else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True
            break

        bc_pass      = _get_bc_mixed(x0)
        _base_steps = _make_simplex_steps()
        if _pass > 0 and _n_c > 0:
            _base_steps[:_n_k] = 0.1  # K near optimum after pass 0; stay local
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])

        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0}

        def _obj_timed(params_trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            f = objective(params_trial, _bc)
            if f < _tracker["f"]:
                _tracker["f"] = f
                _tracker["x"] = params_trial.copy()
            if time.time() - _t0 > _tlim:
                raise _Timeout()
            return f

        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys), constraints=constraints or [], fixed=logk_dict,
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult()
            _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])

        if _n_c > 0:
            x0_prev = x0.copy()
            x0 = result.x.copy()
            best_x_global = x0.copy()
            _update_bds_c(x0)
            if _any_timed_out:
                break
            _max_move = np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:]))
            if _max_move < tolerance:
                break
        else:
            best_x_global = result.x.copy()

    _bc_final  = _get_bc_mixed(best_x_global)
    class _BestResult:
        x       = best_x_global
        success = getattr(result, "success", False)
        fun     = getattr(result, "fun", np.inf)
        nit     = _total_nit
    result = _BestResult()

    timed_out      = _any_timed_out
    fitted_logks   = {fit_keys[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1])) for i in range(_n_c)}
    c_final        = _simulate(result.x)
    bc             = _bc_final
    sp_concs       = {sp: [(t_bc, c_bc)] for sp, (t_bc, c_bc) in bc.items()}

    timed_out      = _any_timed_out
    fitted_logks   = {fit_keys[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1])) for i in range(_n_c)}
    c_final        = _simulate(result.x)
    sp_concs     = {sp: [(t_bc, c_bc)] for sp, (t_bc, c_bc) in bc.items()}

    delta_vecs_all = {}; delta_bound_all = {}; ref_corrections = {}; pure_shifts = {}
    integ_res = []; integ_obs = []
    shift_res = []; shift_obs = []
    # v2: per-source arrays
    per_col_y_obs_integ  = {}; per_col_y_calc_integ  = {}; per_col_x_integ  = {}
    per_col_y_obs_shift  = {}; per_col_y_calc_shift  = {}; per_col_x_shift  = {}

    if c_final is not None:
        t_sim_f = c_final["t"]
        for sp, (t_bc, c_bc) in bc.items():
            c_th = np.interp(t_bc, t_sim_f, _resolve_c(c_final, sp, parsed, t_sim_f))
            integ_res.extend((c_bc - c_th).tolist()); integ_obs.extend(c_bc.tolist())
            per_col_y_obs_integ[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc_integ[sp] = np.asarray(c_th, dtype=float)
            per_col_x_integ[sp]      = np.asarray(t_bc, dtype=float)
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, fake_network)
            if not sp_coeffs: continue
            t_exp    = nmr_data[col]["v_add_mL"]
            t_free_c = t_free_val[col]
            denom_full = np.maximum(
                sum(coeff * np.interp(t_exp, t_sim_f, c_final.get(sp, np.zeros_like(t_sim_f)))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(t_free_c, t_sim_f, c_final.get(sp, np.zeros_like(t_sim_f))))
                for coeff, sp in sp_coeffs), 1e-20)
            # Build full design matrix (all species, ref first)
            all_target_sp = list(sp_coeffs)
            n_pts    = len(t_exp)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(t_exp, t_sim_f, c_final.get(sp, np.zeros_like(t_sim_f))) / denom_full
                F_ref  = coeff * float(np.interp(t_free_c, t_sim_f,
                         c_final.get(sp, np.zeros_like(t_sim_f)))) / denom_ref
                X[:, i] = F_full - F_ref
            dobs_rel = nmr_data[col]["y"] - delta_free[col]
            _col_bd = _col_to_bound.get(col)
            _col_pn = _col_to_pinned.get(col)
            if X.shape[1] > 0:
                try:
                    dd, calc_rel, _ = solve_shifts(
                        X, dobs_rel,
                        [sp for _, sp in all_target_sp], sp_coeffs[0][1],
                        _shift_constraints,
                        column_bound=_col_bd, pinned_dd=_col_pn,
                        noref=_nmr_noref,
                    )
                except ValueError:
                    dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
                    calc_rel = X @ dd
            else:
                dd = np.array([]); calc_rel = np.zeros_like(dobs_rel)
            shift_res.extend((dobs_rel - calc_rel).tolist()); shift_obs.extend(dobs_rel.tolist())
            per_col_y_obs_shift[col]  = np.asarray(dobs_rel, dtype=float)
            per_col_y_calc_shift[col] = np.asarray(calc_rel, dtype=float)
            per_col_x_shift[col]      = np.asarray(t_exp,    dtype=float)
            # dd[i] = δ_i_absolute − δ_obs(t_free); legacy auto-pin sets dd[ref]=0
            sp_dd = {sp: float(dd[i]) if i < len(dd) else 0.0
                     for i, (_, sp) in enumerate(all_target_sp)}
            # Normalize when not anchored (see fit_nmr_shifts for rationale).
            if not (_nmr_noref and _read_species):
                ref_sp = sp_coeffs[0][1]
                ref_dd = sp_dd.get(ref_sp, 0.0)
                if abs(ref_dd) > 1e-12:
                    sp_dd = {k: v - ref_dd for k, v in sp_dd.items()}
            delta_vecs_all[col] = sp_dd
            if len(sp_coeffs) == 2:
                non_ref_sp = sp_coeffs[1][1]
                delta_bound_all[col] = sp_dd.get(non_ref_sp, 0.0)
            # ref_corrections[col] = Σᵢ Fᵢ(t_ref)·ddᵢ over ALL species (incl. ref).
            ref_corrections[col] = sum(
                coeff * float(np.interp(t_free_c, t_sim_f,
                              c_final.get(sp, np.zeros_like(t_sim_f)))) / denom_ref
                * sp_dd.get(sp, 0.0)
                for coeff, sp in sp_coeffs   # all species, incl. ref
            )
            # Pure-species shifts: stored as dd (= δ − δ_obs(t_free)).  Display
            # adds δ_obs(t_free) when noref+read: anchors absolute scale.
            if tgt not in pure_shifts:
                pure_shifts[tgt] = {}
            pure_shifts[tgt][col] = dict(sp_dd)

    def _r2_rmse(res_list, obs_list):
        if not res_list: return 0.0, 0.0, 0.0
        r = np.array(res_list); o = np.array(obs_list)
        ssr_ = float(np.sum(r**2))
        sst_ = float(np.sum((o - o.mean())**2)) if len(o) > 1 else 1.0
        return 1.0 - ssr_/max(sst_, 1e-30), float(np.sqrt(ssr_/max(len(r), 1))), ssr_

    r2_integ, rmse_integ, ssr_integ = _r2_rmse(integ_res, integ_obs)
    r2_shift, rmse_shift, ssr_shift = _r2_rmse(shift_res, shift_obs)
    n_integ_pts = len(integ_res); n_shift_pts = len(shift_res); n_total = n_integ_pts + n_shift_pts
    r2   = (r2_integ * n_integ_pts + r2_shift * n_shift_pts) / max(n_total, 1)
    ssr  = ssr_integ + ssr_shift
    rmse = float(np.sqrt(ssr / max(n_total, 1)))
    # v2: combined flat residuals (integration first, then shift)
    _kmix_yobs = np.asarray(list(integ_obs) + list(shift_obs), dtype=float)
    _kmix_res  = np.asarray(list(integ_res) + list(shift_res), dtype=float)
    _kmix_ycalc = _kmix_yobs - _kmix_res
    ssr_obj = float(data_objective(result.x, _bc_final))
    param_errors = {}
    _cov_mat = None
    _cov_names = list(fit_keys) + list(fit_conc_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final),
                                   result.x, ssr_obj, n_total, n_p)
        for _i in range(min(len(fit_keys), n_p)):
            if _i in _err_idx: param_errors[fit_keys[_i]] = _err_idx[_i]
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors[fit_conc_keys[_i]] = _err_idx[_idx]

    # Variance-normalized chi²-RMSE for the LST (see fit_nmr_mixed for
    # rationale): the optimizer minimized data_objective (chi²) which
    # has its minimum at a DIFFERENT point than raw rmse for mixed
    # units, so the LST must probe chi² to land at the same anchor.
    try:
        _chi2_final = float(data_objective(result.x, _bc_final))
        _chi2_rmse  = float(np.sqrt(max(_chi2_final, 0.0) / max(n_total, 1)))
    except Exception:
        _chi2_rmse = float(rmse)

    col_to_sp = {}; col_to_nH = {}
    _vars_kin2 = parsed.get("variables", {}) or {}
    for idx, col in enumerate(integ_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_kin_sp and sp not in _vars_kin2: sp = col.split(".")[0]
        if sp not in all_kin_sp and sp not in _vars_kin2: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    stats = {
        "r_squared": r2, "rmse": rmse, "chi2_rmse": _chi2_rmse,
        "ssr": ssr, "n_points": n_total, "n_params": n_p,
        "param_values": fitted_logks, "param_errors": param_errors,
        "param_cov": _cov_mat, "param_cov_names": _cov_names,
        "sp_concs": sp_concs, "col_to_sp": col_to_sp, "col_to_nH": col_to_nH,
        "delta_vecs_all": delta_vecs_all, "delta_bound_all": delta_bound_all,
        "delta_free": delta_free, "x_free_val": t_free_val,
        "col_to_target": col_to_target, "ref_corrections": ref_corrections,
        "fit_mode": "mixed", "is_kinetics": True, "n_iter": getattr(result, "nit", 0),
        "timed_out": timed_out, "pure_shifts": pure_shifts,
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "fitted_concs": fitted_concs_k, "fitted_titrants": {},
        "r2_integ": r2_integ, "rmse_integ": rmse_integ, "n_integ_pts": n_integ_pts,
        "r2_shift": r2_shift, "rmse_shift": rmse_shift, "n_shift_pts": n_shift_pts,
        # v2 diagnostics — combined flat residuals (integration first, shift second)
        "y_obs":     _kmix_yobs,
        "y_calc":    _kmix_ycalc,
        "residuals": _kmix_res,
        # v2 bootstrap — per-source arrays (split by data type so the
        # residual-resampling bootstrap can resample within each block)
        "per_col_y_obs_integ":  per_col_y_obs_integ,
        "per_col_y_calc_integ": per_col_y_calc_integ,
        "per_col_x_integ":      per_col_x_integ,
        "per_col_y_obs_shift":  per_col_y_obs_shift,
        "per_col_y_calc_shift": per_col_y_calc_shift,
        "per_col_x_shift":      per_col_x_shift,
        # v2 unified per_col_* — what the residuals-vs-predictor scatter
        # consumes via collect_residuals_from_stats.  Integration and
        # shift columns are tagged so the legend tells them apart even
        # if they happen to share a name (the y-axis is residual in
        # both cases, units differ but both are just (obs - calc)).
        "per_col_x":      {**{f"{c} (int)":   v for c, v in per_col_x_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_x_shift.items()}},
        "per_col_y_obs":  {**{f"{c} (int)":   v for c, v in per_col_y_obs_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_obs_shift.items()}},
        "per_col_y_calc": {**{f"{c} (int)":   v for c, v in per_col_y_calc_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_calc_shift.items()}},
    }
    _r2v = stats["r_squared"]
    if timed_out and _r2v >= 0.99: stats["timed_out"] = False
    _conv = result.success or ssr < 1e-6 or (timed_out and _r2v >= 0.99)
    stats.update(_trace.summary(result, timed_out, tolerance))
    return _conv, fitted_logks, stats, "Kinetics mixed NMR fit complete"
