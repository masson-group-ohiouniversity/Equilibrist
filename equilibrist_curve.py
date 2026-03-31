"""equilibrist_curve.py"""
import numpy as np
from scipy.optimize import least_squares
from equilibrist_network import solve_free_species, solve_equilibria_general
from equilibrist_network import compute_variable_curve, _sanitise_pct, _LN_LO, _LN_HI
from equilibrist_network import evaluate_variable_expression, resolve_variable_dependencies

__all__ = ['compute_curve', 'evaluate_x_expression', 'convert_exp_x', '_solid_col_header_to_equiv',
           '_x_per_equiv', '_find_maxEquiv', 'find_equiv_for_x', '_solid_frac_for_tkey', 'compute_single_point']


def compute_curve(parsed: dict, network: dict, logK_vals: dict, params: dict) -> dict:
    """
    Sweep equiv from 0 to maxEquiv, solving mass balances at each point.
    Handles both liquid titration (volume changes) and solid addition (volume fixed).
    """
    conc0_mM   = params["conc0"]
    V0         = params["V0_mL"]
    tit_names  = params["titrant_free_names"]
    tit_keys   = params["titrant_keys"]
    tit_mMs    = params["titrant_mMs"]      # {free_name: mM}  (0.0 for solid)
    tit_ratios = params["titrant_ratios"]   # {free_name: ratio} (only solid)
    is_solid   = params["titrant_is_solid"]
    tit_name   = tit_names[0]
    tit_mM     = tit_mMs[tit_name]          # 0.0 for solid
    maxEquiv   = params["maxEquiv"]
    nPts       = params["nPts"]
    primary    = params["primary_component"]

    n0        = {name: conc * V0 for name, conc in conc0_mM.items()}  # mmol
    xs        = np.linspace(0.0, float(maxEquiv), int(nPts))
    n_primary = n0.get(primary, 1.0)

    # Normalise solid ratios so they sum to 1
    if is_solid and tit_ratios:
        ratio_sum = sum(tit_ratios.values())
        solid_fractions = {tfree: tit_ratios[tfree] / ratio_sum for tfree in tit_names}
    else:
        solid_fractions = {}

    free_species = network["free_species"]
    all_species  = network["all_species"]

    out = {sp: np.zeros(len(xs)) for sp in all_species}
    out["equiv"]      = xs
    out["warn"]       = np.zeros(len(xs), dtype=bool)
    out["resid_norm"] = np.zeros(len(xs))

    all_comp_names   = list(n0.keys()) + tit_names
    totals_mM_arrays = {name: np.zeros(len(xs)) for name in all_comp_names}
    for tkey, tfree in zip(tit_keys, tit_names):
        totals_mM_arrays[tkey] = totals_mM_arrays[tfree]

    def safe_x0(totals):
        x0_vals = []
        for fs in free_species:
            total = totals.get(fs, 1e-20)
            free_fraction = 0.1 if total > 1e-6 else 0.5
            init_guess = max(total * free_fraction, 1e-15)
            x0_vals.append(np.log(init_guess))
        return np.clip(np.array(x0_vals), -30.0 + 1e-6, _LN_HI - 1e-6)

    x0 = None

    for i, eq in enumerate(xs):
        n_tit = eq * n_primary   # total mmol of titrant added at this point

        if is_solid:
            # ── Solid: volume is fixed ──────────────────────────────────
            V = V0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree in tit_names:
                frac     = solid_fractions.get(tfree, 1.0)
                n_species = n_tit * frac          # mmol of this solid species
                totals[tfree] = totals.get(tfree, 0.0) + (n_species / V) * 1e-3
        else:
            # ── Liquid: volume grows with titrant added ─────────────────
            V_add = n_tit / max(tit_mM, 1e-12)
            V     = V0 + V_add
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree, tkey in zip(tit_names, tit_keys):
                ratio    = tit_mMs[tfree] / max(tit_mM, 1e-12)
                tit_conc = (n_tit * ratio / V) * 1e-3
                totals[tfree] = totals.get(tfree, 0.0) + tit_conc

        # Store total concentrations in mM
        for name in list(n0.keys()) + tit_names:
            totals_mM_arrays[name][i] = totals.get(name, 0.0) * 1e3

        if x0 is None:
            x0 = safe_x0(totals)

        sol, free_concs = solve_free_species(totals, network, logK_vals, x0)

        residual_norm = float(np.linalg.norm(sol.fun))
        if residual_norm > 1e-6:
            x0_retry = np.clip(
                np.array([np.log(max(totals.get(fs, 1e-20) * 0.5, 1e-15)) for fs in free_species]),
                -30.0, 2.0
            )
            sol, free_concs = solve_free_species(totals, network, logK_vals, x0_retry)

        x0 = sol.x

        rn = float(np.linalg.norm(sol.fun))
        out["resid_norm"][i] = rn
        out["warn"][i]       = (not np.isfinite(rn)) or (rn > 1e-8)

        memo = {}
        for sp in all_species:
            if "_rigorous_concentrations" in network and sp in network["_rigorous_concentrations"]:
                out[sp][i] = network["_rigorous_concentrations"][sp] * 1e3
            else:
                out[sp][i] = network["species_conc_fn"](sp, free_concs, logK_vals, memo) * 1e3

    out["totals_mM"]    = totals_mM_arrays
    out["V0_mL"]        = V0
    out["mmol_titrant"] = xs * n_primary
    for tfree, tkey in zip(tit_names, tit_keys):
        if is_solid:
            frac = solid_fractions.get(tfree, 1.0)
        else:
            frac = tit_mMs[tfree] / max(tit_mM, 1e-12)
        out[f"mmol_{tkey}"] = xs * n_primary * frac

    # ── X0 variables: dilution-corrected analytical concentrations (mM) ──
    # X0[i] = total analytical concentration of species X in the cell
    # at titration point i, BEFORE any equilibrium takes place.
    # In liquid mode this decreases with dilution; in solid mode it is constant.
    n_tit_arr = xs * n_primary  # cumulative mmol of primary titrant
    if is_solid:
        _V_arr = np.full(len(xs), V0)
    else:
        _v_add_arr = n_tit_arr / max(tit_mM, 1e-12)  # mL of titrant added
        _V_arr     = V0 + _v_add_arr

    # Concentrations from $concentrations: dilute as V grows
    for _root, _cval_mM in conc0_mM.items():
        _x0_key = _root + "0"  # e.g. G → G0
        if is_solid:
            out[_x0_key] = np.full(len(xs), _cval_mM)
        else:
            out[_x0_key] = (_cval_mM * V0) / _V_arr

    # Concentrations from $titrant: add titrant contribution to X0
    # If a species also appears in $concentrations, X0 = initial + added (total analytical)
    for _tfree, _tkey in zip(tit_names, tit_keys):
        _h0_key = _tfree + "0"  # e.g. C → C0
        if is_solid:
            _frac = solid_fractions.get(_tfree, 1.0)
            _added = (n_tit_arr * _frac) / V0
        else:
            _ratio = tit_mMs[_tfree] / max(tit_mM, 1e-12)
            _added = (n_tit_arr * _ratio) / _V_arr
        if _h0_key in out:
            out[_h0_key] = out[_h0_key] + _added  # ADD to initial
        else:
            out[_h0_key] = _added

    return out


def evaluate_x_expression(expr: str, curve: dict, parsed: dict) -> tuple:
    """
    Evaluate the x-axis expression at every titration point.

    Namespace:

      Species names  (e.g. G, GH, GH2)
        → post-equilibrium concentration in mM at each titration point (np.ndarray)

      X0 names  (e.g. G0, H0, Q0)
        → analytical concentration in mM BEFORE equilibrium at each point (np.ndarray)
          For $concentrations species: dilution-corrected = (C0*V0) / (V0+v_add)
          For $titrant species:        (mmol_added) / (V0+v_add)
          H0/G0 = equivalents of H added relative to G (dimensionless) ✓

    Arithmetic operators + - * / ( ) are all supported.

    Returns (x_vals array, x_label string).
    Raises ValueError with a helpful message on failure.
    """
    # Species that are not X0 variables or bookkeeping arrays
    _reserved = {"equiv", "warn", "resid_norm", "totals_mM", "V0_mL",
                 "mmol_titrant"}
    all_species = set(k for k in curve.keys()
                      if k not in _reserved
                      and not k.startswith("mmol_")
                      and not k.endswith("0"))  # exclude X0 variables
    n = len(curve["equiv"])

    ns = {}

    # Species concentrations (mM arrays, after equilibrium)
    for sp in all_species:
        ns[sp] = curve[sp]

    # X0 variables: dilution-corrected analytical concentrations (mM arrays)
    # G0, H0, etc. = concentration in cell before equilibrium at each point
    for cname in parsed["concentrations"]:
        if cname in curve:
            ns[cname] = curve[cname]
    for tkey in parsed.get("titrant", {}):
        tfree = tkey[:-1] if tkey.endswith("t") else tkey
        h0_key = tfree + "0"
        if h0_key in curve:
            ns[h0_key] = curve[h0_key]

    try:
        result = eval(expr, {"__builtins__": {}}, ns)
    except Exception as e:
        raise ValueError(f"Could not evaluate x expression '{expr}': {e}")

    result = np.broadcast_to(np.asarray(result, dtype=float), n).copy()
    return result, expr


# ─────────────────────────────────────────────
# SIDEBAR HELPERS
# ─────────────────────────────────────────────


def _solid_col_header_to_equiv(col_values: np.ndarray, col_header: str,
                               parsed: dict, params: dict) -> np.ndarray:
    """
    Convert solid-mode column-A values to equivalents, using the column header
    (cell A1) to know what the user stored there.

    Supported header forms
    ──────────────────────
    "H0"      titrant X analytical concentration (mM)
    "H0/G0"   ratio of titrant-X concentration to a $concentrations species Y (equiv)

    In solid mode at equivalents q:
        H0(q) = init_H0 + q * primary_mM * frac_H
    where
        primary_mM = params["conc0"][primary_component]   (sidebar value)
        frac_H     = titrant_ratios[H] / Σ ratios
        init_H0    = params["conc0"].get(H, 0.0)          (if H also in $concentrations)

    For "H0/Y0" ratio the denominator Y0 is the constant $concentrations value of Y.

    Falls back to treating col_values as equivalents if header is empty or unrecognised.
    """
    import re as _re
    col_values = np.asarray(col_values, dtype=float)
    if not col_header:
        return col_values  # no header → already equivalents (backward compat)

    tit_free_names = params.get("titrant_free_names", [])
    tit_ratios     = params.get("titrant_ratios", {})
    conc0          = params.get("conc0", {})           # {root: mM}  sidebar values
    primary        = params.get("primary_component", "")
    primary_mM     = float(conc0.get(primary, 1.0))
    ratio_sum      = float(sum(tit_ratios.values())) if tit_ratios else 1.0

    # Map "H0" → (tfree, normalised_fraction)
    tit_x0 = {}
    for tfree in tit_free_names:
        frac = tit_ratios.get(tfree, 1.0) / max(ratio_sum, 1e-30)
        tit_x0[tfree + "0"] = (tfree, frac)

    # Map "G0" → mM  for $concentrations species (sidebar-adjusted)
    conc_x0 = {}
    for root, mM in conc0.items():
        conc_x0[root + "0" if not root.endswith("0") else root] = float(mM)
    # Also include parsed canonical names (covers any not in conc0 yet)
    for cname, cval in parsed.get("concentrations", {}).items():
        if cname not in conc_x0:
            conc_x0[cname] = float(cval)

    header = col_header.strip()

    # ── "X0/Y0" form ──────────────────────────────────────────────────────────
    m = _re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*/\s*([A-Za-z_][A-Za-z0-9_]*)$', header)
    if m:
        num_name, den_name = m.group(1), m.group(2)
        if num_name in tit_x0 and den_name in conc_x0:
            tfree, frac = tit_x0[num_name]
            den_mM      = conc_x0[den_name]
            init_H0     = float(conc0.get(tfree, 0.0))   # 0 if pure titrant
            # col = (init_H0 + q * primary_mM * frac) / den_mM
            slope  = primary_mM * frac / max(den_mM, 1e-30)
            offset = init_H0 / max(den_mM, 1e-30)
            return np.maximum((col_values - offset) / max(slope, 1e-30), 0.0)

    # ── "X0" form (bare titrant concentration in mM) ──────────────────────────
    if header in tit_x0:
        tfree, frac = tit_x0[header]
        init_H0     = float(conc0.get(tfree, 0.0))
        slope       = primary_mM * frac            # mM per equiv
        # col = init_H0 + q * slope
        return np.maximum((col_values - init_H0) / max(slope, 1e-30), 0.0)

    # ── Fallback: treat as equivalents ────────────────────────────────────────
    return col_values


def convert_exp_x(v_add_mL: np.ndarray, x_expr: str, parsed: dict,
                  params: dict, network: dict, x_col_header: str = "") -> np.ndarray:
    """
    Convert the experimental x column to the same x-axis expression used by the plot.

    Solid mode:  column A already contains x-axis values → return as-is.
    Liquid mode: column A is volume added (mL) → compute X0 variables from v_add and
                 evaluate the x-axis expression in the same X0 namespace as the plot.
    """
    is_solid = params.get("titrant_is_solid", False)

    if is_solid:
        col_values = np.asarray(v_add_mL, dtype=float)
        # Step 1: convert col-A values → equivalents using the column header
        equiv_arr = _solid_col_header_to_equiv(col_values, x_col_header, parsed, params)

        # Step 2: build X0 namespace at each equivalent point
        tit_free_names = params.get("titrant_free_names", [])
        tit_ratios     = params.get("titrant_ratios", {})
        conc0          = params.get("conc0", {})
        primary        = params.get("primary_component", "")
        primary_mM     = float(conc0.get(primary, 1.0))
        ratio_sum      = float(sum(tit_ratios.values())) if tit_ratios else 1.0

        ns = {}
        # X0 for $concentrations species: constant (no dilution in solid mode)
        for root, cval_mM in conc0.items():
            x0_key = root + "0" if not root.endswith("0") else root
            ns[x0_key] = np.full(len(equiv_arr), float(cval_mM))
        # Also parsed canonical names (script values, only if not already set)
        for cname, cval in parsed.get("concentrations", {}).items():
            if cname not in ns:
                ns[cname] = np.full(len(equiv_arr), float(cval))

        # X0 for titrant species: initial (from $concentrations) + added
        for tfree in tit_free_names:
            h0_key = tfree + "0"
            frac   = tit_ratios.get(tfree, 1.0) / max(ratio_sum, 1e-30)
            added  = equiv_arr * primary_mM * frac
            if h0_key in ns:
                ns[h0_key] = ns[h0_key] + added   # initial + added
            else:
                ns[h0_key] = added                 # titrant-only → starts at 0

        # Step 3: evaluate x_expr in that namespace
        try:
            result = eval(x_expr, {"__builtins__": {}}, ns)
            return np.broadcast_to(np.asarray(result, dtype=float), len(equiv_arr)).copy()
        except Exception:
            return col_values  # fallback: return raw col values

    # ── Liquid mode ──────────────────────────────────────────────────────
    tit_free_names = network["titrant_free_names"]
    tit_keys       = network["titrant_keys"]
    tit_mMs        = params["titrant_mMs"]
    conc0          = params["conc0"]
    V0             = params["V0_mL"]

    v_add     = np.asarray(v_add_mL, dtype=float)
    tit_mM_primary = tit_mMs[tit_free_names[0]]
    V_arr     = V0 + v_add  # cell volume at each experimental point

    ns = {}
    # X0 for $concentrations: dilution-corrected mM
    for root, cval_mM in conc0.items():
        x0_key = root + "0" if not root.endswith("0") else root
        ns[x0_key] = (cval_mM * V0) / V_arr

    # X0 for $titrant: add titrant contribution (total = initial + added)
    for tfree, tkey in zip(tit_free_names, tit_keys):
        h0_key = tfree + "0"
        ratio = tit_mMs[tfree] / max(tit_mM_primary, 1e-12)
        _added = (v_add * tit_mM_primary * ratio) / V_arr
        if h0_key in ns:
            ns[h0_key] = ns[h0_key] + _added  # ADD to initial diluted conc
        else:
            ns[h0_key] = _added

    try:
        result = eval(x_expr, {"__builtins__": {}}, ns)
        result = np.broadcast_to(np.asarray(result, dtype=float), len(v_add)).copy()
    except Exception:
        result = v_add.copy()

    return result



def _x_per_equiv(x_expr, parsed, conc_vals, V0_mL,
                 titrant_free_names, titrant_keys, titrant_mMs, titrant_ratios,
                 is_solid, primary_component):
    """Return x(equiv=1) — the x-axis value at exactly 1 equivalent."""
    V0       = V0_mL
    primary_mM = conc_vals.get(primary_component, 1.0)
    n_prim   = primary_mM * V0   # mmol of primary component

    ns = {}
    if is_solid:
        # Volume is fixed; X0 concentrations don't change with dilution
        for cname, cval in parsed["concentrations"].items():
            ns[cname] = cval   # constant mM (no dilution)
        ratio_sum = sum(titrant_ratios.values()) if titrant_ratios else 1.0
        for tfree in titrant_free_names:
            frac = titrant_ratios.get(tfree, 1.0) / ratio_sum
            h0_key = tfree + "0"
            _added = n_prim * frac / V0  # mM at 1 equiv
            if h0_key in ns:
                ns[h0_key] = ns[h0_key] + _added  # ADD to initial
            else:
                ns[h0_key] = _added
    else:
        tit_mM_prim = titrant_mMs[titrant_free_names[0]]
        v_add_1 = n_prim / max(tit_mM_prim, 1e-12)  # mL added at 1 equiv
        V_1     = V0 + v_add_1
        # X0 for concentrations at 1 equiv (diluted)
        for cname, cval in parsed["concentrations"].items():
            ns[cname] = (cval * V0) / V_1
        # X0 for titrant at 1 equiv: ADD to initial if species also in $concentrations
        for tfree in titrant_free_names:
            h0_key = tfree + "0"
            ratio = titrant_mMs[tfree] / max(tit_mM_prim, 1e-12)
            _added = (v_add_1 * tit_mM_prim * ratio) / V_1
            if h0_key in ns:
                ns[h0_key] = ns[h0_key] + _added  # ADD to initial diluted conc
            else:
                ns[h0_key] = _added

    try:
        val = float(eval(x_expr, {"__builtins__": {}}, ns))
        return val if val > 0 else 1.0
    except Exception:
        return 1.0   # fallback: treat equiv ≡ x




def _find_maxEquiv(x_expr, xmax_target, parsed, conc_vals, V0_mL,
                   titrant_free_names, titrant_keys, titrant_mMs,
                   titrant_ratios, is_solid, primary_component,
                   max_search=1e5):
    """
    Numerically find the equiv value where the x-axis expression equals xmax_target.

    This replaces the linear approximation (xmax / _x_per_equiv) which fails
    for ratio expressions like C0/A0 that are nonlinear in equivalents.

    Uses binary search after bounding the root. If xmax_target is unreachable
    (above the asymptote), returns max_search as a fallback so the simulation
    covers as wide a range as possible.
    """
    import math

    def _x_at_eq(equiv):
        return _x_at_equiv(equiv, x_expr, parsed, conc_vals, V0_mL,
                           titrant_free_names, titrant_mMs, titrant_ratios,
                           is_solid, primary_component)

    x0 = _x_at_eq(0.0)

    # Check if xmax_target is reachable by probing increasing equiv values
    # Double equiv until we exceed xmax or hit max_search
    hi = 1.0
    while hi < max_search:
        x_hi = _x_at_eq(hi)
        if not math.isfinite(x_hi):
            break
        if x_hi >= xmax_target:
            break
        hi = min(hi * 2.0, max_search)

    x_hi = _x_at_eq(hi)
    if not math.isfinite(x_hi) or x_hi < xmax_target:
        # xmax_target is unreachable (above asymptote).
        # Find the plateau value (x at hi, which is effectively the asymptote)
        # and return the equiv needed to reach 99% of that plateau.
        # This ensures the simulation is dense enough to show the full curve.
        x_plateau = _x_at_eq(hi) if math.isfinite(_x_at_eq(hi)) else _x_at_eq(hi / 2)
        if not math.isfinite(x_plateau) or x_plateau <= 0:
            return hi  # fallback
        target_99 = x_plateau * 0.99
        # Binary search for 99% of plateau
        lo99, hi99 = 0.0, hi
        for _ in range(60):
            mid = (lo99 + hi99) * 0.5
            if _x_at_eq(mid) < target_99:
                lo99 = mid
            else:
                hi99 = mid
            if (hi99 - lo99) < 1e-6:
                break
        return (lo99 + hi99) * 0.5

    # Binary search for equiv where x = xmax_target
    lo = 0.0
    for _ in range(60):  # 60 iterations → precision < 1e-15 * range
        mid = (lo + hi) * 0.5
        x_mid = _x_at_eq(mid)
        if x_mid < xmax_target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-9:
            break

    return (lo + hi) * 0.5

def _x_at_equiv(equiv, x_expr, parsed, conc_vals, V0_mL,
                titrant_free_names, titrant_mMs, titrant_ratios,
                is_solid, primary_component):
    """
    Evaluate the x-axis expression at a given equiv, using only X0 namespace
    (no equilibrium solving).  Shared by _find_maxEquiv and find_equiv_for_x.
    """
    V0 = V0_mL
    primary_mM = conc_vals.get(primary_component, 1.0)
    n_prim = primary_mM * V0  # mmol

    ns = {}
    if is_solid:
        for cname, cval in parsed["concentrations"].items():
            ns[cname] = float(cval)
        ratio_sum = sum(titrant_ratios.values()) if titrant_ratios else 1.0
        for tfree in titrant_free_names:
            frac = titrant_ratios.get(tfree, 1.0) / max(ratio_sum, 1e-30)
            added = n_prim * equiv * frac / V0
            h0_key = tfree + "0"
            ns[h0_key] = ns.get(h0_key, 0.0) + added
    else:
        tit_mM_prim = titrant_mMs[titrant_free_names[0]]
        v_add = n_prim * equiv / max(tit_mM_prim, 1e-30)
        V = V0 + v_add
        for cname, cval in parsed["concentrations"].items():
            ns[cname] = (float(cval) * V0) / V
        for tfree in titrant_free_names:
            ratio = titrant_mMs[tfree] / max(tit_mM_prim, 1e-30)
            added = (v_add * tit_mM_prim * ratio) / V
            h0_key = tfree + "0"
            ns[h0_key] = ns.get(h0_key, 0.0) + added

    try:
        return float(eval(x_expr, {"__builtins__": {}}, ns))
    except Exception:
        return float("nan")


def find_equiv_for_x(target_x, parsed, params):
    """
    Convert an x-axis value back to equivalents using binary search.

    The old linear approximation (equiv = target_x / x_per_equiv) is WRONG
    whenever the x-expression is non-zero at equiv=0 — e.g. x = C0 where C0
    has an initial value from $concentrations.  The correct inversion is:

        equiv = (target_x - x_at_0) / slope

    but slope itself may be non-constant for ratio expressions.  Binary search
    via _x_at_equiv handles both cases exactly.
    """
    import math as _math
    x_expr = parsed.get("plot_x_expr") or \
        f"{params['titrant_free_names'][0]}0/{list(parsed['concentrations'].keys())[0]}"

    is_solid   = params.get("titrant_is_solid", False)
    conc_vals  = params["conc0"]
    V0_mL      = params["V0_mL"]
    tit_names  = params["titrant_free_names"]
    tit_keys   = params["titrant_keys"]
    tit_mMs    = params["titrant_mMs"]
    tit_ratios = params.get("titrant_ratios", {})
    primary    = params["primary_component"]

    def _xe(eq):
        return _x_at_equiv(eq, x_expr, parsed, conc_vals, V0_mL,
                           tit_names, tit_mMs, tit_ratios, is_solid, primary)

    x0 = _xe(0.0)
    if not _math.isfinite(x0):
        x0 = 0.0

    # Find upper bracket
    hi = 1.0
    for _ in range(40):
        x_hi = _xe(hi)
        if _math.isfinite(x_hi) and x_hi >= target_x:
            break
        hi = min(hi * 2.0, 1e6)
    else:
        # target_x unreachable — fall back to linear
        xpe = _x_per_equiv(x_expr, parsed, conc_vals, V0_mL,
                           tit_names, tit_keys, tit_mMs, tit_ratios, is_solid, primary)
        return (target_x - x0) / max(xpe - x0, 1e-30) if xpe > x0 else target_x / max(xpe, 1e-30)

    # Binary search
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _xe(mid) < target_x:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-9:
            break
    return (lo + hi) * 0.5


def _solid_frac_for_tkey(tkey_num, params):
    """
    Return the normalised fraction of the solid titrant component whose
    script key is tkey_num (e.g. 'Mt').  Returns 1.0 if not found.
    """
    tit_keys   = params.get("titrant_keys", [])
    tit_names  = params.get("titrant_free_names", [])
    tit_ratios = params.get("titrant_ratios", {})
    ratio_sum  = sum(tit_ratios.values()) if tit_ratios else 1.0
    if ratio_sum == 0:
        return 1.0
    for tkey, tfree in zip(tit_keys, tit_names):
        if tkey == tkey_num:
            return tit_ratios.get(tfree, 1.0) / ratio_sum
    return 1.0


def compute_single_point(equiv, parsed, network, logK_vals, params, target_variable):
    """
    Compute theoretical concentration for a single titration point.
    Uses solve_equilibria_general directly — consistent with compute_curve.
    """
    try:
        conc0_mM  = params["conc0"]
        V0        = params["V0_mL"]
        tit_names = params["titrant_free_names"]
        tit_mMs   = params["titrant_mMs"]
        primary   = params["primary_component"]

        n0        = {name: conc * V0 for name, conc in conc0_mM.items()}
        n_primary = n0.get(primary, 1.0)
        equiv     = max(equiv, 0.0)

        n_tit = equiv * n_primary
        is_solid   = params.get("titrant_is_solid", False)
        tit_ratios = params.get("titrant_ratios", {})

        if is_solid:
            V = V0
            ratio_sum = sum(tit_ratios.values()) if tit_ratios else 1.0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree in tit_names:
                frac = tit_ratios.get(tfree, 1.0) / ratio_sum
                totals[tfree] = totals.get(tfree, 0.0) + (n_tit * frac / V) * 1e-3
        else:
            tit_mM = tit_mMs[tit_names[0]]
            V_add  = n_tit / max(tit_mM, 1e-12)
            V      = V0 + V_add
            if V <= 0:
                return 0.0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree, tkey in zip(tit_names, params["titrant_keys"]):
                ratio = tit_mMs[tfree] / max(tit_mM, 1e-12)
                totals[tfree] = totals.get(tfree, 0.0) + (n_tit * ratio / V) * 1e-3

        # Solve with the general solver
        equilibria  = network.get("equilibria", [])
        all_species = network["all_species"]
        concs_M, _, _ = solve_equilibria_general(totals, equilibria, all_species, logK_vals)

        # Convert to mM and read target
        concs_mM = {sp: concs_M[sp] * 1e3 for sp in all_species}

        # Direct species
        if target_variable in concs_mM:
            val = concs_mM[target_variable]
            return float(val) if np.isfinite(val) else 0.0

        # Expression variable ($variables section)
        variables = parsed.get("variables", {})
        if target_variable in variables:
            var_order = resolve_variable_dependencies(variables)
            variable_values = {}
            for var_name in var_order:
                variable_values[var_name] = evaluate_variable_expression(
                    variables[var_name], concs_mM, variable_values
                )
            val = variable_values.get(target_variable, 0.0)
            return float(val) if np.isfinite(val) else 0.0

        return 0.0

    except Exception:
        return 0.0


# ─────────────────────────────────────────────