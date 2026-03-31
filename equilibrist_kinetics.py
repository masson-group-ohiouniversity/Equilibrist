# -*- coding: utf-8 -*-
"""equilibrist_kinetics.py"""
import re
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from equilibrist_network import solve_equilibria_general
from equilibrist_parser import constraints_penalty

__all__ = ['_kinetics_reaction_label', '_rate_constant_units', '_equilibrium_constant_units', '_collect_all_kinetic_species', 'build_kinetics_logk_dict', 'compute_kinetics_curve', 'fit_kinetics']


def _kinetics_reaction_label(rxn: dict) -> str:
    """Build a short human-readable label for a kinetic or equilibrium reaction."""
    def _side(pairs):
        parts = []
        for coeff, sp in pairs:
            parts.append(sp if coeff == 1 else f"{coeff}{sp}")
        return " + ".join(parts)
    t = rxn.get("type", "equilibrium")
    arrow = "<>" if t == "reversible_kinetic" else (">" if t == "irreversible" else "=")
    return f"{_side(rxn['reactants'])} {arrow} {_side(rxn['products'])}"


def _rate_constant_units(n_reactants: int, is_reverse: bool = False, n_products: int = 0) -> str:
    """
    Return the SI unit string for a rate constant given reaction order.
    Forward: order = sum of reactant stoich coefficients.
    Reverse: order = sum of product stoich coefficients.
    """
    order = n_products if is_reverse else n_reactants
    if order == 1:
        return "s⁻¹"
    elif order == 2:
        return "M⁻¹ s⁻¹"
    elif order == 3:
        return "M⁻² s⁻¹"
    else:
        return f"M⁻{order-1} s⁻¹"


def _equilibrium_constant_units(n_reactants: int, n_products: int) -> str:
    """Return units for an equilibrium constant K."""
    delta = n_products - n_reactants
    if delta == 0:
        return "dimensionless"
    elif delta == -1:
        return "M"
    elif delta == 1:
        return "M⁻¹"
    elif delta < 0:
        return f"M{-delta}"
    else:
        return f"M⁻{delta}"



def _collect_all_kinetic_species(parsed: dict) -> list:
    """Return sorted list of all species appearing in kinetic + equilibrium reactions."""
    sp_set = set()
    for rxn in parsed["kinetics"] + parsed["equilibria"]:
        for _, sp in rxn["reactants"]:
            sp_set.add(sp)
        for _, sp in rxn["products"]:
            sp_set.add(sp)
    # Also include anything in $concentrations
    for cname in parsed["concentrations"]:
        root = cname[:-1] if cname.endswith("0") else cname
        sp_set.add(root)
    return sorted(sp_set)


def build_kinetics_logk_dict(parsed: dict) -> dict:
    """
    Return dict of all rate-constant names → log10(k) values.
    Covers both kinetic reactions (log_k, log_kr) and pre-equilibria (logK).
    """
    d = {}
    for rxn in parsed["kinetics"]:
        d[rxn["kname"]] = rxn["log_k"]
        if "krname" in rxn:
            d[rxn["krname"]] = rxn["log_kr"]
    for eq in parsed["equilibria"]:
        d[eq["kname"]] = eq["logK"]
    return d


def compute_kinetics_curve(parsed: dict, logk_dict: dict, t_max: float, n_pts: int) -> dict:
    """
    Integrate the kinetics ODEs from t=0 to t_max.

    All species are tracked in the ODE state vector.
    Pre-equilibria ('=' reactions) are enforced as very fast reversible rates,
    so the stiff Radau solver handles them naturally — no operator splitting needed.
    This guarantees mass conservation even when pre-eq species appear in kinetic reactions.

    State vector: y[i] = concentration (mM) of all_species[i].
    """
    from scipy.integrate import solve_ivp

    all_species  = _collect_all_kinetic_species(parsed)
    n_sp         = len(all_species)
    sp_idx       = {sp: i for i, sp in enumerate(all_species)}
    kinetic_rxns = parsed["kinetics"]
    pre_eq_rxns  = parsed["equilibria"]

    # ── Initial conditions ────────────────────────────────────────────────────
    init_concs = {}
    for cname, cval_mM in parsed["concentrations"].items():
        root = cname[:-1] if cname.endswith("0") else cname
        init_concs[root] = cval_mM

    # If pre-equilibria exist, redistribute initial concentrations so the
    # starting point already satisfies them — this eliminates the t=0 spike.
    if pre_eq_rxns:
        totals_M = {sp: init_concs.get(sp, 0.0) * 1e-3 for sp in all_species}
        pre_eq_logK = {eq["kname"]: logk_dict[eq["kname"]] for eq in pre_eq_rxns}
        try:
            concs_M, _, _ = solve_equilibria_general(
                totals_M, pre_eq_rxns, all_species, pre_eq_logK)
            for sp in all_species:
                init_concs[sp] = concs_M[sp] * 1e3
        except Exception:
            pass  # fall back to raw initial conditions

    y0 = np.array([init_concs.get(sp, 0.0) for sp in all_species], dtype=float)

    # ── Choose k_fast: 1e4 × largest kinetic rate constant ───────────────────
    max_log_k = max((logk_dict.get(r["kname"], 0) for r in kinetic_rxns), default=0)
    k_fast = max(10.0 ** (max_log_k + 4), 1e4)

    # ── Pre-compute M→mM conversion factors for each kinetic reaction ─────────
    # ODE runs in mM; user enters k in M units.
    # rate_mM = k_M * 1000^(1 - n_r) * prod([S]_mM^coeff)
    # So k_eff_mM = k_M * 1000^(1 - n_r_fwd) for forward
    #              k_M * 1000^(1 - n_r_rev) for reverse (n_r_rev = sum of product coeffs)
    rxn_conv_fwd = []
    rxn_conv_rev = []
    for rxn in kinetic_rxns:
        n_r = sum(coeff for coeff, _ in rxn["reactants"])
        n_p = sum(coeff for coeff, _ in rxn["products"])
        rxn_conv_fwd.append(1000.0 ** (1 - n_r))
        rxn_conv_rev.append(1000.0 ** (1 - n_p))

    def ode_rhs(t, y):
        c = np.maximum(y, 0.0)   # concentrations in mM, clamped to ≥ 0

        dydt = np.zeros(n_sp)

        # ── Kinetic reactions (> and <>) ──────────────────────────────────────
        for i_rxn, rxn in enumerate(kinetic_rxns):
            # k entered in M units → convert to mM-compatible rate constant
            k_fwd = 10.0 ** logk_dict[rxn["kname"]] * rxn_conv_fwd[i_rxn]
            r_fwd = k_fwd
            for coeff, sp in rxn["reactants"]:
                r_fwd *= c[sp_idx[sp]] ** coeff

            r_rev = 0.0
            if rxn["type"] == "reversible_kinetic" and "krname" in rxn:
                k_rev = 10.0 ** logk_dict[rxn["krname"]] * rxn_conv_rev[i_rxn]
                r_rev = k_rev
                for coeff, sp in rxn["products"]:
                    r_rev *= c[sp_idx[sp]] ** coeff

            net = r_fwd - r_rev
            for coeff, sp in rxn["reactants"]:
                dydt[sp_idx[sp]] -= coeff * net
            for coeff, sp in rxn["products"]:
                dydt[sp_idx[sp]] += coeff * net

        # ── Pre-equilibria (= reactions): fast reversible rates ───────────────
        # K is stored in M units; ODE runs in mM.
        # K_mM = K_M * 1000^(n_p - n_r)  →  k_rev = k_fast / K_mM
        for eq in pre_eq_rxns:
            K_M     = 10.0 ** logk_dict[eq["kname"]]
            n_r     = sum(coeff for coeff, sp in eq["reactants"])
            n_p     = sum(coeff for coeff, sp in eq["products"])
            K_mM    = K_M * (1000.0 ** (n_p - n_r))
            k_rev_e = k_fast / max(K_mM, 1e-30)

            r_fwd = k_fast
            for coeff, sp in eq["reactants"]:
                r_fwd *= c[sp_idx[sp]] ** coeff

            r_rev = k_rev_e
            for coeff, sp in eq["products"]:
                r_rev *= c[sp_idx[sp]] ** coeff

            net = r_fwd - r_rev
            for coeff, sp in eq["reactants"]:
                dydt[sp_idx[sp]] -= coeff * net
            for coeff, sp in eq["products"]:
                dydt[sp_idx[sp]] += coeff * net

        return dydt

    # ── Integrate ─────────────────────────────────────────────────────────────
    t_eval = np.linspace(0.0, t_max, n_pts)
    sol = solve_ivp(
        ode_rhs, [0.0, t_max], y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10,
        dense_output=False,
    )

    # ── Assemble output dict (mM arrays) ──────────────────────────────────────
    out = {"t": sol.t, "warn": np.zeros(len(sol.t), dtype=bool),
           "success": sol.success}
    for i, sp in enumerate(all_species):
        out[sp] = np.maximum(sol.y[i], 0.0)
    if not sol.success:
        out["warn"][:] = True

    return out


def fit_kinetics(parsed: dict, exp_data: dict, logk_dict: dict, fit_keys: list,
                 t_max: float, n_pts: int, tolerance: float, maxiter: int,
                 timeout_s: float = 30.0, constraints=None, fit_conc_keys=None):
    """
    Fit selected log10(k) values to experimental kinetic data.
    exp_data format: {species_name: {"v_add_mL": t_array, "y": y_array}}
    where "v_add_mL" is reused to hold time in seconds.

    Returns (success, fitted_logks, stats, message).
    """
    import time
    from scipy.optimize import minimize

    fit_names = list(fit_keys)
    x0 = np.array([logk_dict[k] for k in fit_names])
    n_p = len(fit_names)
    # ── Fittable concentrations ──────────────────────────────────────────────
    fit_conc_keys = list(fit_conc_keys or [])
    _n_k   = len(fit_names)
    _n_c   = len(fit_conc_keys)
    CONC_MIN = 0.0

    # root → original cname map (e.g. 'G' → 'G0')
    _root_to_cname_k = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_k[_r] = _cn

    def _cb_k(root, script_val):
        _cn = _root_to_cname_k.get(root, root)
        lo, hi = parsed.get("conc_bounds", {}).get(_cn,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, script_val * 0.80)
        hi = hi if hi is not None else script_val * 1.20
        return (lo, hi)

    # Starting values and bounds for concentration parameters
    _conc_script = {}  # root → script value (mM)
    for _cn, _cv in parsed.get("concentrations", {}).items():
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _conc_script[_r] = float(_cv)

    _x0_c   = np.array([_conc_script.get(r, 1.0) for r in fit_conc_keys])
    _bds_c  = [_cb_k(r, _conc_script.get(r, 1.0)) for r in fit_conc_keys]

    def _unpack_concs(params_vec):
        """Return (logk_vec, conc_dict_for_parsed_update)."""
        lk = params_vec[:_n_k]
        cd = {fit_conc_keys[i]: float(np.clip(params_vec[_n_k + i],
                                               _bds_c[i][0], _bds_c[i][1]))
              for i in range(_n_c)}
        return lk, cd

    def _patched_parsed(conc_d):
        """Return parsed copy with updated concentrations."""
        p = dict(parsed)
        p["concentrations"] = dict(parsed["concentrations"])
        for root, val in conc_d.items():
            cname = _root_to_cname_k.get(root, root)
            p["concentrations"][cname] = float(val)
        return p

    x0 = np.concatenate([x0, _x0_c])
    n_p = len(x0)

    # Pre-collect experimental points: [(t, y, col_name), ...]
    exp_points = []
    for col, data in exp_data.items():
        if col.startswith("_"):
            continue
        t_arr = data["v_add_mL"]   # time in seconds
        y_arr = data["y"]
        for t_val, y_val in zip(t_arr, y_arr):
            if np.isfinite(t_val) and np.isfinite(y_val):
                exp_points.append((float(t_val), float(y_val), col))

    if len(exp_points) < n_p:
        return False, {}, {}, "Too few data points"

    def _simulate(params_vec):
        lk_vec, conc_d = _unpack_concs(params_vec)
        current_logk = dict(logk_dict)
        for i, name in enumerate(fit_names):
            current_logk[name] = lk_vec[i]
        cur_parsed = _patched_parsed(conc_d) if conc_d else parsed
        try:
            curve = compute_kinetics_curve(cur_parsed, current_logk, t_max, n_pts)
            return curve
        except Exception:
            return None

    def objective(params_vec):
        lk_vec, conc_d = _unpack_concs(params_vec)
        lk = dict(logk_dict)
        for i, name in enumerate(fit_names):
            lk[name] = lk_vec[i]
        curve = _simulate(params_vec)
        if curve is None:
            return 1e12
        penalty = sum(max(0, abs(p) - 12) ** 2 for p in params_vec) * 0.1
        residuals = []
        for t_val, y_val, col in exp_points:
            if col in curve:
                y_sim = np.interp(t_val, curve["t"], curve[col])
                residuals.append(y_sim - y_val)
        if not residuals:
            return 1e12
        ssr = sum(r ** 2 for r in residuals) + penalty
        cp = constraints_penalty(constraints or [], lk, ssr_scale=ssr)
        return ssr + cp

    # ── Timeout-aware Nelder-Mead ────────────────────────────────────────
    class _Timeout(Exception): pass
    best_tracker = {"x": x0.copy(), "f": np.inf, "start": time.time(),
                    "timed_out": False, "nit": 0}

    def _obj_timed(params_vec):
        best_tracker["nit"] += 1
        f = objective(params_vec)
        if f < best_tracker["f"]:
            best_tracker["f"] = f
            best_tracker["x"] = params_vec.copy()
        if time.time() - best_tracker["start"] > timeout_s:
            best_tracker["timed_out"] = True
            raise _Timeout()
        return f

    # Phase 1: warm-start logK before joint optimisation
    if _n_c > 0 and _n_k > 0:
        def _phase1_obj(lk_vec):
            return objective(np.concatenate([lk_vec, x0[_n_k:]]))
        _sp1 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                         for i in range(_n_k)])
        try:
            _r1 = minimize(_phase1_obj, x0[:_n_k], method="Nelder-Mead",
                           options={"maxiter": maxiter//2,
                                    "xatol": tolerance,
                                    "fatol": tolerance * 1e-4,
                                    "adaptive": True, "initial_simplex": _sp1})
            x0 = np.concatenate([_r1.x, x0[_n_k:]])
        except Exception:
            pass
        best_tracker["start"] = time.time()  # reset timer after Phase 1

    _steps = np.array([1.5] * _n_k + [max(abs(_x0_c[i]) * 0.1, 0.05)
                       for i in range(_n_c)])
    init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _steps[i]
                                     for i in range(n_p)])
    try:
        result = minimize(_obj_timed, x0, method="Nelder-Mead",
                          options={"maxiter": maxiter, "xatol": tolerance,
                                   "fatol": tolerance * 1e-4, "adaptive": True,
                                   "initial_simplex": init_simplex})
    except _Timeout:
        class _MockResult:
            x = best_tracker["x"]; success = False; fun = best_tracker["f"]
            nit = best_tracker["nit"]
        result = _MockResult()

    timed_out    = best_tracker["timed_out"]
    fitted_logks = {fit_names[i]: result.x[i] for i in range(_n_k)}
    fitted_concs_k = {fit_conc_keys[i]: float(np.clip(result.x[_n_k+i],
                      _bds_c[i][0], _bds_c[i][1]))
                      for i in range(_n_c)}

    # ── Statistics ───────────────────────────────────────────────────────
    curve = _simulate(result.x)
    residuals = []
    y_obs = []
    for t_val, y_val, col in exp_points:
        if curve is not None and col in curve:
            y_sim = np.interp(t_val, curve["t"], curve[col])
            residuals.append(y_sim - y_val)
            y_obs.append(y_val)
    residuals = np.array(residuals)
    y_obs     = np.array(y_obs)
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((y_obs - y_obs.mean()) ** 2)) if len(y_obs) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / len(residuals))) if len(residuals) > 0 else 0.0

    # Jacobian-based parameter errors (relative step, covers logK and conc params)
    n_d = len(residuals)
    param_errors = {}
    try:
        r0 = np.array([np.interp(t, curve["t"], curve[col]) - y
                       for t, y, col in exp_points])
        J  = np.zeros((n_d, n_p))
        for k in range(n_p):
            eps_k = max(abs(result.x[k]) * 1e-4, 1e-6)
            dx = np.zeros(n_p); dx[k] = eps_k
            c2 = _simulate(result.x + dx)
            if c2 is None:
                continue
            r2_vec = np.array([np.interp(t, c2["t"], c2[col]) - y
                                for t, y, col in exp_points])
            J[:, k] = (r2_vec - r0) / eps_k
        sigma2 = ssr / max(n_d - n_p, 1)
        cov    = np.linalg.pinv(J.T @ J) * sigma2
        for k in range(_n_k):
            param_errors[fit_names[k]] = float(np.sqrt(max(cov[k, k], 0)))
        for k in range(_n_c):
            param_errors[fit_conc_keys[k]] = float(np.sqrt(max(cov[_n_k + k, _n_k + k], 0)))
    except Exception:
        pass

    _r2 = r2
    if timed_out and _r2 >= 0.99:
        timed_out = False   # good enough — suppress warning

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        n_d,
        "n_params":        n_p,
        "param_values":    fitted_logks,
        "param_errors":    param_errors,
        "fitted_concs":    fitted_concs_k,
        "fitted_titrants": {},
        "fit_mode":        "kinetics",
        "timed_out":       timed_out,
        "n_iter":          getattr(result, "nit", 0),
    }

    _conv = result.success or ssr < 1e-6 or (not timed_out and _r2 >= 0.99)
    _msg  = "Kinetics fit complete"
    return _conv or not timed_out, fitted_logks, stats, _msg
