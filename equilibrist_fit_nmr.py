# -*- coding: utf-8 -*-
"""equilibrist_fit_nmr.py"""
import re
import time
import collections
import numpy as np
from scipy.optimize import minimize
from equilibrist_network import compute_variable_curve, _sanitise_pct
from equilibrist_curve import convert_exp_x, find_equiv_for_x, compute_single_point, params_covering_data
import equilibrist_live as _live

WEAK_SPECIES_FRACTION = 0.02   # a species never above 2 % of its pool over the data cannot have its shift determined


def _weak_species(X, species, pinned=None, shift_constraints=None, thr=WEAK_SPECIES_FRACTION):
    """Species whose design column never leaves ±``thr``: they never make more
    than ``thr`` of the fast-exchange pool over the data, so their shift is not
    identifiable and the bounded least squares would park it anywhere inside the
    per-column bound (an aromatic proton "at 5.5 ppm").  Such a shift is held at
    the free shift (dd = 0) and flagged.  Species pinned by ``read:`` or named in
    a shift constraint are left alone."""
    skip = set(pinned or {})
    for c in shift_constraints or ():
        if isinstance(c, dict) and c.get("type") in ("shift_eq_group", "shift_order"):
            skip.update(c.get("species", ()) or ())
    out = []
    for i, sp in enumerate(species):
        if sp in skip or i >= X.shape[1]:
            continue
        if X.shape[0] == 0 or float(np.max(np.abs(X[:, i]))) < thr:
            out.append(sp)
    return out


def _pins_with_weak(X, species, pinned, shift_constraints=None):
    """``pinned`` plus a dd = 0 pin for every undetermined species; also returns them."""
    weak = _weak_species(X, species, pinned, shift_constraints)
    if not weak:
        return pinned, weak
    p = dict(pinned or {})
    p.update({sp: 0.0 for sp in weak})
    return p, weak
from equilibrist_curve import compute_curve, evaluate_x_expression
from equilibrist_parser import constraints_penalty, PenaltyScale, simplex_step, tied_minimize
from equilibrist_shift_constraints import solve_shifts, get_per_column_bounds

__all__ = ['shift_backcalc_resolved', 'compute_nmr_curves', '_get_species_for_target', '_hessian_errors', 'fit_nmr_shifts', '_resolve_c', '_nmr_integration_backCalc', 'fit_nmr_integration', 'fit_nmr_mixed']


def compute_nmr_curves(nmr_cfg: dict, parsed: dict, curve: dict,
                        network: dict, x_vals: np.ndarray,
                        delta_bound: dict) -> dict:
    """
    Compute theoretical NMR chemical shift curves for all targets.

    Fast-exchange (shift mode):
        δ_obs(x) = Σ_i  [species_i](x) / target_total(x)  × δ_i

    Parameters
    ----------
    nmr_cfg    : parsed["nmr"]  {"mode": "shift", "targets": ["Gtot"]}
    curve      : concentration dict from compute_curve  (mM arrays)
    delta_bound: {target: {species: δ_value}}   — pure-species shifts
                 e.g. {"Gtot": {"G": 8.0, "GH": 12.5}}

    Returns
    -------
    {"Gtot": np.ndarray of δ_obs values}   (one entry per target)
    """
    variables  = parsed.get("variables", {})
    all_species = network["all_species"]
    out = {}

    for target in nmr_cfg["targets"]:
        shifts = delta_bound.get(target, {})
        if not shifts:
            continue

        # Numerator: sum of [sp](x) * δ_sp
        numerator = np.zeros_like(x_vals)
        # Denominator: total concentration of target species
        denominator = np.zeros_like(x_vals)

        for sp, delta in shifts.items():
            if sp in curve:
                c_sp = curve[sp]
            elif sp in variables:
                # Evaluate variable expression
                c_sp = compute_variable_curve(sp, variables, curve, network, x_vals)
            else:
                c_sp = np.zeros_like(x_vals)
            numerator   += c_sp * delta
            denominator += c_sp

        # Avoid division by zero
        safe_denom = np.where(denominator > 1e-20, denominator, 1e-20)
        out[target] = numerator / safe_denom

    return out


def shift_backcalc_resolved(M, rhs, c_model, rel_cut: float = 0.05):
    """Concentrations of the non-free species of a shift target, from the
    observed shifts, resolved only along the directions the signals determine.

    ``M`` (n_sig x n_sp) holds the fitted pure-species shift changes, ``rhs``
    (n_sig x n_pts) the observed shift changes times the target's total, and
    ``c_model`` (n_sp x n_pts) the model's concentrations at the same points.
    Returns ``(c, k)`` with  c = c_model + M+ (rhs - M c_model), the pseudo-
    inverse truncated to the singular values >= rel_cut x the largest, and k
    the number kept.  With a square, well-conditioned M this is the plain
    inversion; when the signals cannot tell species apart (fewer signals than
    species, or near-identical shift changes), the plain minimum-norm solve
    scattered the points over physically impossible values; here those
    directions follow the model and only what the shifts resolve is read
    from the data."""
    M = np.asarray(M, dtype=float); rhs = np.asarray(rhs, dtype=float)
    c0 = np.asarray(c_model, dtype=float)
    if M.size == 0 or rhs.size == 0:
        return c0.copy(), 0
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s >= float(rel_cut) * (float(s[0]) if s.size else 0.0)
    k = int(np.sum(keep))
    if k == 0:
        return c0.copy(), 0
    Mp = (Vt[keep].T * (1.0 / s[keep])) @ U[:, keep].T          # n_sp x n_sig
    return c0 + Mp @ (rhs - M @ c0), k


def _get_species_for_target(target: str, parsed: dict, network: dict) -> list:
    """
    Return [(coeff, species), ...] for the target variable (or bare species).

    Parses stoichiometric coefficients so that weighted NMR averages are exact:
      Gtot = G + GH + GH2     -> [(1,'G'), (1,'GH'), (1,'GH2')]
      Htot = H + GH + 2*GH2  -> [(1,'H'), (1,'GH'), (2,'GH2')]
      bare species G           -> [(1,'G')]
    """
    variables   = parsed.get("variables", {})
    all_species = network["all_species"]

    if target in variables:
        expr  = variables[target]
        parts = [p.strip() for p in expr.split("+") if p.strip()]
        result = []
        for part in parts:
            m = re.match(r'^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$', part)
            if m:
                coeff, sp = float(m.group(1)), m.group(2)
            else:
                coeff, sp = 1.0, part
            if sp in all_species:
                result.append((coeff, sp))
        return result
    elif target in all_species:
        return [(1.0, target)]
    return []


def _linear_eq_matrix(constraints, param_names):
    """Rows of the linear equality constraints (log-space, '=') expressed
    over the fitted parameter vector ``param_names``.  Coefficients on
    parameters that are not fitted are constants and drop out.  Returns an
    (m × n) array with linearly independent rows, or None if no such
    constraint touches a fitted parameter."""
    if not constraints:
        return None
    rows = []
    for c in constraints:
        if c.get("mode", "log") != "log" or c.get("op", "=") != "=":
            continue
        row = np.array([float(c["coeffs"].get(nm, 0.0)) for nm in param_names])
        if np.any(row != 0.0):
            rows.append(row)
    if not rows:
        return None
    A = np.vstack(rows)
    # Keep an independent subset of rows (drop duplicates / combinations)
    if np.linalg.matrix_rank(A) < A.shape[0]:
        keep = []
        for i in range(A.shape[0]):
            trial = A[keep + [i]]
            if np.linalg.matrix_rank(trial) == len(keep) + 1:
                keep.append(i)
        A = A[keep]
    return A


def _hessian_errors(objective_fn, x_best: np.ndarray, ssr_best: float,
                    n_data: int, n_params: int, step: float = 1e-3,
                    eq_matrix=None, n_log_params=None, flat_delta: float = 1.0,
                    max_sigma: float = 3.0, notes: dict = None):
    """
    Estimate ±log K errors from the finite-difference Hessian of the objective.

    Returns a tuple ``(errors_by_index, cov_matrix)``:
      * ``errors_by_index`` — ``{i: σᵢ}`` mapping the position in
        ``x_best`` to the marginal standard error.  Same as the legacy
        return value used by callers.
      * ``cov_matrix`` — the full ``n × n`` covariance matrix as a
        numpy array (or ``None`` on numerical failure).  Off-diagonal
        elements give the parameter covariances that the diagnostics
        layer normalises to a correlation matrix for the heatmap.

    Method
    ──────
    For a least-squares objective  f(θ) = SSR(θ),  the Hessian at the minimum is
    approximately  H ≈ 2 J'J  (Gauss-Newton), and the covariance matrix is:

        Cov(θ) = σ² × (H/2)⁻¹  =  [SSR/(n-p)] × (J'J)⁻¹

    When the Jacobian is unavailable (Nelder-Mead), we estimate H by central
    finite differences of the scalar objective:

        H[i,i]  ≈  [f(θ+hᵢ) + f(θ-hᵢ) - 2f(θ)] / hᵢ²   (diagonal)
        H[i,j]  ≈  [f(θ+hᵢ+hⱼ) - f(θ+hᵢ-hⱼ)
                    - f(θ-hᵢ+hⱼ) + f(θ-hᵢ-hⱼ)] / (4hᵢhⱼ)  (off-diagonal)

    σ² = SSR / (n - p)  is the reduced chi-squared.
    Error on parameter i = sqrt(Cov[i,i]) in log K units, or ``None`` when
    the data do not determine that parameter.

    Undetermined parameters
    ───────────────────────
    A finite-difference Hessian is unreliable along directions where the
    objective is flat: solver noise of either sign gives a spurious
    curvature and the reported error is then arbitrary (anything from
    "± 0.3" to "± 70").  So, for the first ``n_log_params`` entries (the
    log-space constants; default all), the objective is probed directly at
    ± ``flat_delta`` (1 log unit) along the parameter's admissible direction
    (its axis, or its projection onto the constraint null space so that
    partners move consistently).  If the objective changes by less than
    what a standard error of ``max_sigma`` (3 log units) would imply,

        Δf  <  f0 · flat_delta² / (max_sigma² · (n_data − n_params)),

    the parameter is reported as ``None``.  Errors larger than ``max_sigma``
    are reported as ``None`` as well.

    One-sided cases are recognised too: a parameter that has hit a bound is
    not at a stationary point, so the local curvature is meaningless.  If
    the objective is flat on one side only, the data provide just an upper
    (or lower) limit.  When a ``notes`` dict is supplied it is filled with
    ``{i: {"kind": "flat" | "upper-limit" | "lower-limit" | "large",
    "sigma": <1σ excursion on the rising side, one-sided kinds only>}}``
    for every parameter reported as ``None``.

    Equality constraints
    ────────────────────
    When ``eq_matrix`` (m × n, from ``_linear_eq_matrix``) is given, the
    fitted parameters are confined to the affine subspace A·θ = b.  The
    covariance is then evaluated in the null-space coordinates
    θ = θ* + N·z (N = null_space(A)):

        Cov(θ) = σ² · N (N'·H·N / 2)⁻¹ N'

    so directions removed by the constraints carry zero variance and the
    uncertainty of a constrained parameter is propagated from its partners
    (e.g. for  log KA + log KB = c  both get the same ± error).  The
    effective number of parameters in σ² is reduced by rank(A).
    """
    n = len(x_best)
    if n == 0:
        return {}, None

    N = None
    if eq_matrix is not None:
        A = np.atleast_2d(np.asarray(eq_matrix, dtype=float))
        if A.shape[1] == n and A.shape[0] > 0:
            from scipy.linalg import null_space as _ns
            N = _ns(A)
            n_params = n_params - int(np.linalg.matrix_rank(A))
            if N.shape[1] == 0:          # fully determined by constraints
                return {i: 0.0 for i in range(n)}, np.zeros((n, n))
    if n_data <= n_params:
        return {}, None

    sigma2 = max(ssr_best / (n_data - n_params), 1e-30)
    h = np.abs(x_best) * step
    h[h < 1e-6] = step   # minimum step

    try:
        f0 = float(objective_fn(x_best))   # evaluate at minimum (may differ from ssr_best)
        H  = np.zeros((n, n))

        # Diagonal elements
        for i in range(n):
            ei      = np.zeros(n); ei[i] = h[i]
            fp      = objective_fn(x_best + ei)
            fm      = objective_fn(x_best - ei)
            H[i, i] = (fp + fm - 2 * f0) / (h[i] ** 2)

        # Off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = h[i]
                ej = np.zeros(n); ej[j] = h[j]
                fpp = objective_fn(x_best + ei + ej)
                fpm = objective_fn(x_best + ei - ej)
                fmp = objective_fn(x_best - ei + ej)
                fmm = objective_fn(x_best - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])

        if N is not None:
            # Project onto the constraint null space before inverting
            Hr = N.T @ H @ N
            eigvals = np.linalg.eigvalsh(Hr)
            if np.any(eigvals <= 0):
                Hr += np.eye(Hr.shape[0]) * max(-eigvals.min() * 1.01, 1e-10)
            cov = N @ np.linalg.inv(Hr / 2.0) @ N.T * sigma2
        else:
            # Regularize: ensure H is positive definite
            eigvals = np.linalg.eigvalsh(H)
            if np.any(eigvals <= 0):
                H += np.eye(n) * max(-eigvals.min() * 1.01, 1e-10)
            cov = np.linalg.inv(H / 2.0) * sigma2
        # ── Flatness probe (see docstring) ───────────────────────────────
        n_probe = n if n_log_params is None else min(int(n_log_params), n)
        dof     = max(n_data - n_params, 1)
        f_thr   = abs(f0) * flat_delta ** 2 / (max_sigma ** 2 * dof)
        P = (N @ N.T) if N is not None else None
        flat = set(); _note = {}
        for i in range(n_probe):
            d = np.zeros(n); d[i] = 1.0
            if P is not None:
                d = P @ d
                if abs(d[i]) < 1e-9:
                    continue          # fully fixed by the constraints
                d = d / d[i]          # parameter i moves by exactly flat_delta
            try:
                fp = float(objective_fn(x_best + flat_delta * d))
                fm = float(objective_fn(x_best - flat_delta * d))
            except Exception:
                continue
            # A side counts as "open" when the objective does not RISE by at
            # least f_thr there: flat (no information) or descending (the
            # point is not a minimum along that side — typically a parameter
            # pinned at a bound of its search range, or a fit cut short).
            up_open   = (fp - f0) < f_thr
            down_open = (fm - f0) < f_thr
            if up_open and down_open:
                flat.add(i); _note[i] = {"kind": "flat"}
            elif down_open or up_open:
                # One-sided: the data bound the parameter only from the
                # rising side.  Estimate that side's 1σ excursion from the
                # profile (quadratic extrapolation from the first probe that
                # rises above the 1σ increase f0/dof).
                sgn  = +1.0 if down_open else -1.0
                thr1 = abs(f0) / dof
                sig_side = None
                for off in (0.25, 0.5, flat_delta):
                    try:
                        fo = float(objective_fn(x_best + sgn * off * d))
                    except Exception:
                        break
                    dfo = fo - f0
                    if dfo >= thr1:
                        sig_side = off * np.sqrt(thr1 / dfo)
                        break
                flat.add(i)
                _note[i] = {"kind": "upper-limit" if down_open else "lower-limit",
                            "sigma": (None if sig_side is None else float(sig_side))}

        errors = {}
        for i in range(n):
            sig = float(np.sqrt(max(cov[i, i], 0.0)))
            cap = max(max_sigma, max_sigma * abs(float(x_best[i])))
            undetermined = (i in flat) or (not np.isfinite(sig)) or (sig > cap)
            if undetermined and i not in _note:
                _note[i] = {"kind": "large"}
            errors[i] = None if undetermined else sig
        if notes is not None:
            notes.update(_note)
        return errors, cov
    except Exception:
        return {}, None


def _max_constraint_violation(constraints, logK_dict) -> float:
    """Largest absolute violation (log units, or relative for K-space
    constraints) among the given $constraints at logK_dict.  0.0 if none."""
    if not constraints:
        return 0.0
    from equilibrist_parser import _eval_kspace_expr
    worst = 0.0
    for c in constraints:
        try:
            if c.get("mode", "log") == "K":
                lhs = _eval_kspace_expr(c["lhs_expr"], logK_dict, c["param_names"])
                rhs = _eval_kspace_expr(c["rhs_expr"], logK_dict, c["param_names"])
                val = (lhs - rhs) / max(abs(rhs), 1e-30)
            else:
                val = (sum(coeff * logK_dict.get(name, 0.0)
                           for name, coeff in c["coeffs"].items()) + c["const"])
        except Exception:
            continue
        op = c.get("op", "=")
        if op == '<':
            viol = max(val, 0.0)
        elif op == '>':
            viol = max(-val, 0.0)
        else:
            viol = abs(val)
        worst = max(worst, float(viol))
    return worst


def _build_pinned_shifts(read_species, known_shifts, shift_cols, delta_free, noref,
                         col_to_bound=None):
    """Build {col: {species: dd}} from the $nmr 'read:' list and sheet-2 rows.

    Sheet-2 cells left empty (or NaN) mark that species x column pair as
    still free to fit, so a partial pin is supported (as documented in the
    manual).  A row may be shorter than the number of shift columns; the
    missing trailing cells are treated as empty.  Rows longer than the
    number of shift columns are rejected.

    Returns (col_to_pinned, error_message_or_None).
    """
    col_to_pinned = {}
    n_cols = len(shift_cols)
    for sp in read_species:
        if sp not in known_shifts:
            return {}, (
                f"$nmr 'read: {sp}' but species '{sp}' not found in sheet 2 of "
                f"the data file (expected row with '{sp}' in column A and up to "
                f"{n_cols} chemical-shift values to its right)."
            )
        vals = np.asarray(known_shifts[sp], dtype=float).ravel()
        if len(vals) > n_cols:
            return {}, (
                f"$nmr 'read: {sp}': sheet 2 has {len(vals)} values for '{sp}' "
                f"but the data has only {n_cols} shift columns. "
                f"Provide at most one value per shift column, in the same order."
            )
        n_pinned = 0
        for k, col in enumerate(shift_cols):
            if k >= len(vals) or not np.isfinite(vals[k]):
                continue   # empty / NaN cell -> this column's shift stays free
            col_to_pinned.setdefault(col, {})[sp] = float(vals[k]) - float(delta_free[col])
            n_pinned += 1
        if n_pinned == 0:
            return {}, (
                f"$nmr 'read: {sp}': the sheet-2 row for '{sp}' contains no "
                f"numeric chemical-shift value. Provide at least one value "
                f"(leave cells empty for columns that should stay free)."
            )
    # A pinned dd that violates the per-column magnitude bound would make
    # solve_shifts raise on every iteration; the fitters catch that and fall
    # back to an unconstrained lstsq, silently ignoring the pin.  Catch the
    # conflict here so the user gets a clear message instead of a bad fit.
    for col, pins in col_to_pinned.items():
        bnd = (col_to_bound or {}).get(col)
        if bnd is None:
            continue
        B = abs(float(bnd))
        for sp, dd in pins.items():
            if abs(dd) > B + 1e-9:
                return {}, (
                    f"$nmr 'read: {sp}': the sheet-2 shift for '{sp}' in column "
                    f"'{col}' is {dd:+.3f} ppm from the first data point, which "
                    f"lies outside the per-column bound of ±{B:.2f} ppm set by "
                    f"the 'shift: …' magnitude line in $constraints. Widen that "
                    f"bound for this column (or use 'free'), or leave the "
                    f"sheet-2 cell empty to let that shift float."
                )
    if noref and read_species:
        unanchored = [c for c in shift_cols if not col_to_pinned.get(c)]
        if unanchored:
            return {}, (
                f"$nmr 'noref' requires at least one 'read:' pin in every shift "
                f"column, but column(s) {', '.join(unanchored)} have none "
                f"(the corresponding sheet-2 cells are empty). Fill in a known "
                f"shift for at least one read: species in each column, or "
                f"remove 'noref'."
            )
    return col_to_pinned, None


def fit_nmr_shifts(parsed: dict, network: dict, nmr_data: dict,
                   params: dict, logK_vals: dict, fit_keys: list,
                   x_expr: str, tolerance: float, maxiter: int,
                   timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True,
                   _objective_only: bool = False, shared_dd=None):
    """
    Fit equilibrium constants + per-signal pure-species shifts to NMR data.

    General fast-exchange model with stoichiometric weights:
        δ_obs = Σᵢ (nᵢ·[Sᵢ]) / Σⱼ (nⱼ·[Sⱼ])  × δᵢ
              = δ_free + Σᵢ₌₁..N-1  Fᵢ(x) × Δδᵢ

    where nᵢ are stoichiometric coefficients from the $variables expression
    (e.g. Htot = H + GH + 2*GH2 gives n_GH2=2) and Fᵢ = nᵢ·[Sᵢ] / Σnⱼ·[Sⱼ].

    δ_free is read from the first data point of each signal.  For signals where
    V=0 is missing (e.g. host not present at start), the first available point
    is used as reference and the theoretical curve is anchored there too.

    The Δδ vector is solved analytically by linear least squares for each trial
    of K — only K values are optimised by Nelder-Mead.
    """
    from scipy.optimize import minimize
    nmr_cfg = parsed["nmr"]

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # ── Map columns to target variables ──────────────────────────────────────
    col_to_target = {}
    for col in nmr_data:
        if col.startswith("_"): continue
        for tgt in nmr_cfg["targets"]:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = nmr_cfg["targets"][0] if nmr_cfg["targets"] else col

    # the simulation must reach every data point: a point beyond the plotted xmax
    # used to be compared with the END of the curve (the model at xmax, undiluted)
    params = params_covering_data(params, parsed, network, x_expr,
                                  [d["v_add_mL"] for c, d in nmr_data.items() if not c.startswith("_")],
                                  _x_col_header)

    # ── δ_free: first available data point per signal ─────────────────────────
    # (handles host signals where V=0 row is NaN/missing)
    delta_free = {}
    x_free_val = {}   # x-axis value at which δ_free was measured
    for col, col_data in nmr_data.items():
        if col.startswith("_"): continue
        delta_free[col] = float(col_data["y"][0])
        # x value at first point (used to anchor theoretical curve)
        x_arr = convert_exp_x(col_data["v_add_mL"], x_expr, parsed, params, network,
                               x_col_header=_x_col_header)
        x_free_val[col] = float(x_arr[0])

    first_col = next((c for c in nmr_data if not c.startswith("_")), None)
    if first_col is None:
        return False, {}, {}, "No NMR data columns found"


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else simplex_step(1.5)))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _x_for_col(col_data, _params=None):
        return convert_exp_x(col_data["v_add_mL"], x_expr, parsed, _params or params, network,
                             x_col_header=_x_col_header)

    def _build_fraction_matrix(c, x_exp, sp_coeffs, x_free=None,
                                df_theoretical=None, include_ref=False):
        """
        Build design matrix X of shape (n_pts, N) or (n_pts, N-1).

        Fᵢ(x) = nᵢ·[Sᵢ](x) / Σⱼ nⱼ·[Sⱼ](x)   = mole fraction of species i

        If ``include_ref=False`` (default / legacy): builds X with N−1 columns
        for the non-reference species (sp_coeffs[1:]), used in the dd_ref ≡ 0
        formulation.

        If ``include_ref=True``: builds X with N columns for ALL species
        in the target (including sp_coeffs[0]), used in the noref formulation
        where every species has its own free dd parameter.

        If ``x_free`` is given, subtract the theoretical δ at that reference
        point so the regression is for Δδ relative to the same anchor as the
        data.
        """
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        n_pts = len(x_exp)

        # Weighted denominator: Σ nⱼ·[Sⱼ]
        denom = np.zeros(n_pts)
        for coeff, sp in sp_coeffs:
            denom += coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim)))
        denom = np.maximum(denom, 1e-20)

        # Choose which species get columns
        if include_ref:
            cols_sp = list(sp_coeffs)             # all species
        else:
            cols_sp = list(sp_coeffs[1:])         # non-reference only

        # X[:,i] = nᵢ·[Sᵢ] / denom
        X = np.zeros((n_pts, len(cols_sp)))
        for i, (coeff, sp) in enumerate(cols_sp):
            X[:, i] = coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim))) / denom

        # If a reference x-point is given, subtract the fraction at that point
        # so the model predicts Δδ = 0 at the first measurement (as data does)
        if x_free is not None:
            X_ref = np.zeros(len(cols_sp))
            denom_ref = sum(coeff * float(np.interp(x_free, x_sim, c.get(sp, np.zeros_like(x_sim))))
                            for coeff, sp in sp_coeffs)
            denom_ref = max(denom_ref, 1e-20)
            for i, (coeff, sp) in enumerate(cols_sp):
                X_ref[i] = coeff * float(np.interp(x_free, x_sim,
                                                     c.get(sp, np.zeros_like(x_sim)))) / denom_ref
            X = X - X_ref[np.newaxis, :]

        return X

    # Capture user shift constraints once for this fit (cheap to look up)
    _shift_constraints = parsed.get("shift_constraints", []) or []

    # Per-column bounds (the 'shift: -0.8, free, 1.0, ...' syntax).  Build a
    # {col_name: bound_value_or_None} map for use by the inner solver.
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

    # Pinned intrinsic shifts: $nmr 'read:' species + sheet-2 known shifts.
    # Convert absolute ppm in sheet 2 to dd values via dd_X[col] = δ_X − δ_obs(x_free, col).
    # Built as {col_name: {species: dd_value}} for per-column lookup.
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _shift_cols_in_order = [c for c in nmr_data if not c.startswith("_")]
    _col_to_pinned, _pin_err = _build_pinned_shifts(
        _read_species, _known_shifts, _shift_cols_in_order, delta_free,
        bool(nmr_cfg.get("noref", False)), col_to_bound=_col_to_bound)
    if _pin_err:
        return False, {}, {}, _pin_err
    _col_to_pinned_user = {c: dict(v or {}) for c, v in _col_to_pinned.items()}
    if shared_dd:
        # pure-species shifts solved across tabs by a global fit (see fit_nmr_mixed)
        for _c, _pins in shared_dd.items():
            if _c in _shift_cols_in_order:
                _col_to_pinned[_c] = {**(_col_to_pinned.get(_c) or {}), **{k: float(v) for k, v in _pins.items()}}
                _col_to_bound.pop(_c, None)

    # NMR-level noref flag — when True, every solve_shifts call is run in
    # noref mode (no auto-pin to reference), requiring at least one read:
    # anchor.  Default False preserves legacy "Δδ relative to ref" behavior.
    _nmr_noref = bool(nmr_cfg.get("noref", False))

    def _analytic_delta(X, delta_obs_rel, sp_coeffs=None, column_bound=None,
                        pinned_dd=None):
        """Solve the per-column shift sub-problem.  X must have a column for
        every species in sp_coeffs (build with include_ref=True).
        Returns (dd, calc, ssr) where dd[i] corresponds to sp_coeffs[i]."""
        if X.shape[1] == 0:
            return np.array([]), np.zeros_like(delta_obs_rel), float(np.sum(delta_obs_rel**2))
        if sp_coeffs is None:
            # Legacy fall-through (no species info): bare lstsq
            dd = np.linalg.lstsq(X, delta_obs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((delta_obs_rel - calc)**2))
        species = [sp for _, sp in sp_coeffs]
        ref_sp  = sp_coeffs[0][1]
        pinned_dd, _weak = _pins_with_weak(X, species, pinned_dd, _shift_constraints)
        try:
            return solve_shifts(X, delta_obs_rel, species, ref_sp,
                                _shift_constraints,
                                column_bound=column_bound,
                                pinned_dd=pinned_dd,
                                noref=_nmr_noref)
        except ValueError:
            # Bad constraint topology — fall back to unconstrained lstsq so
            # the fit still runs (the parser warning will alert the user).
            dd = np.linalg.lstsq(X, delta_obs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((delta_obs_rel - calc)**2))

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    def objective(trial):
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return 1e12
        total_ssr = sum(d["ssr"] for d in _shift_designs(c, _p))
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(total_ssr))
        _live.notify(lk, total_ssr + cp + conc_penalty)
        return total_ssr + cp + conc_penalty

    def _shift_designs(c, _p):
        """Per shift column (data order): the linear sub-problem X·dd = Δδ at
        curve ``c`` and its own solution's SSR.  equilibrist_global stacks the
        designs across tabs when pure-species shifts are shared."""
        out = []
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_exp         = _x_for_col(col_data, _p)
            _xfv          = float(x_exp[0])
            delta_obs_rel = col_data["y"] - delta_free[col]
            X = _build_fraction_matrix(c, x_exp, sp_coeffs, x_free=_xfv, include_ref=True)
            _, _, ssr = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            out.append({"col": col, "target": tgt, "X": X, "y": np.asarray(delta_obs_rel, dtype=float),
                        "species": [sp for _, sp in sp_coeffs], "ref": sp_coeffs[0][1],
                        "column_bound": _col_to_bound.get(col), "pinned": _col_to_pinned.get(col),
                        "constraints": _shift_constraints, "noref": _nmr_noref,
                        "delta_free": float(delta_free[col]), "ssr": float(ssr)})
        return out

    def data_objective(full_trial):
        """Data-only SSR for Hessian — accepts full parameter vector (logK + conc)."""
        _p = _make_params_nmr(full_trial) if fitting_concs else params
        c = _simulate(full_trial[:len(fit_keys)], _p)
        if c is None: return 1e12
        return float(sum(d["ssr"] for d in _shift_designs(c, _p)))

    def shift_parts(trial):
        """For a global fit with SHARED pure-species shifts (see fit_nmr_mixed)."""
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return None
        return {"integ_ssr": 0.0, "shift_var": 1.0, "designs": _shift_designs(c, _p),
                "penalty": (lambda total_ssr: float(constraints_penalty(
                    constraints or [], lk, ssr_scale=_pscale(total_ssr)) + conc_penalty))}

    import time
    x0  = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p    = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k   = len(fit_keys)
    _n_c   = len(fit_conc_keys)


    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    # Returns the pieces an outer optimiser needs; nothing is optimised.
    if _objective_only:
        _n_data_hint = int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_')))
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float)))),
            "data_ssr":  (lambda t: float(data_objective(np.asarray(t, dtype=float)))),
            "shift_parts": (lambda t: shift_parts(np.asarray(t, dtype=float))),
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys),
            "n_log_params": len(fit_keys),
            "n_data": _n_data_hint,
        }
    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s):
            if time.time() - _t0 > _tlim * 0.4:  # use at most 40% of budget on warm-start
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]))
        try:
            _r0 = tied_minimize(_ph0_obj, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logK_vals,
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass  # use whatever x0 we have

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _total_evals   = 0
    _last_recent   = []
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        # After the warm-start pass, K is near its optimum. Use a tight K step
        # (0.1 log-units) so subsequent passes refine concentrations without
        # wastefully re-exploring K space from scratch every time.
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0,
               "recent": collections.deque(maxlen=4 * (len(x0) + 1))}   # last trial points

        def _obj_timed(trial, _tracker=_bt, _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            _tracker["recent"].append(np.array(trial, dtype=float))
            f = objective(trial)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys), constraints=constraints or [], fixed=logK_vals,
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])
        _total_evals += _bt["nit"]
        _last_recent = list(_bt["recent"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    # ── How did the fit end?  (reported to the user; see stats below) ──
    # final_spread: how much the constants still moved among the last trial
    # points, in log units — the practical "precision reached", to compare
    # with the tolerance asked for.
    _stop_reason = ("timeout" if _any_timed_out else
                    ("tolerance" if getattr(result, "success", False) else "maxiter"))
    try:
        _final_spread = (float(np.max(np.ptp(np.array(_last_recent, dtype=float), axis=0)))
                         if len(_last_recent) > 1 else None)
    except Exception:
        _final_spread = None
    _elapsed_s = time.time() - _global_start
    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final = _make_params_nmr(result.x) if fitting_concs else params

    # ── Final pass: Δδ vectors and statistics ────────────────────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final)
    pure_shifts     = {}
    pure_shift_undetermined = {}     # {target: {col: [species whose shift the data cannot fix]}}
    pure_shift_maxfrac = {}          # {target: {col: {species: largest change of its pool fraction over the data}}}
    pure_shift_pinned = {}           # {target: {col: [species pinned by the user's read: line]}}
    delta_vecs_all  = {}   # {col: {sp: Δδ_relative_to_free}}
    delta_bound_all = {}   # 1:1 compat
    ref_corrections = {}   # {col: float} — correction at reference x-point
    all_residuals   = []
    all_y_obs       = []
    # v2: per-column observed and calculated arrays for bootstrap resampling
    per_col_y_obs   = {}   # {col: ndarray of Δδ_obs_rel}
    per_col_y_calc  = {}   # {col: ndarray of Δδ_calc_rel}
    per_col_x       = {}   # {col: ndarray of x-axis values}

    if c_final is not None:
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue

            x_exp         = _x_for_col(col_data, _params_final)
            delta_obs     = col_data["y"]
            df_exp        = delta_free[col]
            delta_obs_rel = delta_obs - df_exp

            X  = _build_fraction_matrix(c_final, x_exp, sp_coeffs, x_free=x_free_val[col], include_ref=True)
            dd, delta_calc_rel, _ = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            pure_shift_undetermined.setdefault(tgt, {})[col] = _weak_species(
                X, [sp for _, sp in sp_coeffs], _col_to_pinned_user.get(col), _shift_constraints)
            pure_shift_maxfrac.setdefault(tgt, {})[col] = {
                sp: float(np.max(np.abs(X[:, i]))) if X.shape[0] else 0.0 for i, (_, sp) in enumerate(sp_coeffs)}
            pure_shift_pinned.setdefault(tgt, {})[col] = sorted(_col_to_pinned_user.get(col) or {})

            all_residuals.extend((delta_obs_rel - delta_calc_rel).tolist())
            all_y_obs.extend(delta_obs_rel.tolist())   # use Δδ, not absolute ppm
            per_col_y_obs[col]  = np.asarray(delta_obs_rel,  dtype=float)
            per_col_y_calc[col] = np.asarray(delta_calc_rel, dtype=float)
            per_col_x[col]      = np.asarray(x_exp,          dtype=float)

            # In the new convention: dd[i] = δ_i_absolute − δ_obs(V=0).
            # `dd` is the full per-species vector (same order as sp_coeffs).
            # In legacy mode (auto-pin ref to dd=0) dd[ref]=0; in noref mode
            # dd[ref] is fitted from the data.
            sp_dd = {sp: float(dd[i]) if i < len(dd) else 0.0
                     for i, (_, sp) in enumerate(sp_coeffs)}
            # When the absolute shift scale is NOT anchored by a read: pin,
            # subtract dd[ref] from every species so the math reference reads
            # as 0.  This makes legacy and noref-no-read produce identical
            # "Δδ relative to ref" output.  (When anchored, leave dd alone so
            # we can recover absolute δ as δ_obs(V=0) + dd.)
            if not (_nmr_noref and _read_species):
                ref_sp = sp_coeffs[0][1]
                ref_dd = sp_dd.get(ref_sp, 0.0)
                if abs(ref_dd) > 1e-12:
                    sp_dd = {k: v - ref_dd for k, v in sp_dd.items()}
            delta_vecs_all[col] = sp_dd

            if len(sp_coeffs) == 2:
                # 2-species target: store the non-ref dd in delta_bound_all
                non_ref_sp = sp_coeffs[1][1]
                delta_bound_all[col] = sp_dd.get(non_ref_sp, 0.0)

            # Pure-species shifts:
            # Always stored as dd values (= δ_absolute − δ_obs(V=0)).
            # Display/export adds δ_obs(V=0) back to recover absolute when
            # the shift scale is anchored (noref + at least one read:).
            # ref_corrections[col] is consumed by app.py's back-calc-to-
            # concentrations machinery. It is defined as
            #     ref_correction = Σᵢ Fᵢ(x_ref) × ddᵢ   (over ALL species,
            # including the free / reference species).  In legacy & noref-
            # without-read modes the normalization above pins dd[ref]=0, so
            # the ref-species term contributes zero and the sum can be
            # restricted to non-ref species.  In noref+read mode dd[ref]
            # may be nonzero — we must include it in the sum.
            x_sim_fp, _ = evaluate_x_expression(x_expr, c_final, parsed)
            denom_at_ref = max(sum(
                coeff * float(np.interp(x_free_val[col], x_sim_fp,
                              c_final.get(sp, np.zeros_like(x_sim_fp))))
                for coeff, sp in sp_coeffs), 1e-20)
            ref_corrections[col] = sum(
                coeff * float(np.interp(x_free_val[col], x_sim_fp,
                              c_final.get(sp, np.zeros_like(x_sim_fp)))) / denom_at_ref
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

    # Parameter errors via finite-difference Hessian (data-only objective, never penalised).
    # Skipped when ``compute_hessian=False`` (bootstrap workers).  Defaults
    # to empty dict / None placeholders so downstream code that reads
    # param_errors / param_cov doesn't have to special-case the absence.
    _n_total_params_shift = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    param_errors_shift = {}
    _err_notes = {}
    _cov_mat = None
    _cov_names_shift = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(data_objective, result.x, ssr, len(residuals), _n_total_params_shift,
                                   eq_matrix=_linear_eq_matrix(constraints or [], _cov_names_shift),
                                   n_log_params=len(fit_keys), notes=_err_notes)
        param_errors_shift = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_shift[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_shift[fit_titrant_keys[_i]] = _err_idx[_idx]

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        len(residuals),
        "n_params":        len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys),
        "param_values":    fitted_logKs,
        "param_errors":    param_errors_shift,
        "param_error_notes": {_cov_names_shift[i]: r for i, r in _err_notes.items() if i < len(_cov_names_shift)},
        # Full covariance matrix for the correlation-heatmap and t-test
        # diagnostics.  ``param_cov_names`` is the row/column order.
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_shift,
        "pure_shifts":     pure_shifts,
        "pure_shift_undetermined": pure_shift_undetermined,
        "pure_shift_maxfrac": pure_shift_maxfrac,
        "pure_shift_pinned": pure_shift_pinned,        # dd values (= δ − δ_obs(V=0))
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "delta_vecs_all":  delta_vecs_all,
        "delta_bound_all": delta_bound_all,
        "delta_free":      delta_free,
        "x_free_val":      x_free_val,
        "col_to_target":   col_to_target,
        "ref_corrections": ref_corrections,
        "fit_mode":        "shift",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":          getattr(result, "nit", 0),
        "n_evals":         _total_evals,
        "stop_reason":     _stop_reason,
        "final_spread":    _final_spread,
        "elapsed_s":       _elapsed_s,
        "tolerance":       float(tolerance),
        "timed_out":       not getattr(result, "success", True),
        "sp_concs":        {},      # not used in shift mode; keeps display logic uniform
        "col_to_sp":       {},
        "col_to_nH":       {},
        # v2 diagnostics — flat residual arrays
        "y_obs":           np.asarray(y_obs,     dtype=float),
        "y_calc":          np.asarray(y_obs - residuals, dtype=float),
        "residuals":       np.asarray(residuals, dtype=float),
        # v2 bootstrap — per-column arrays for residual resampling
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to   = stats.get("timed_out", False)
    _r2   = stats.get("r_squared", 0.0)
    # A timed-out fit with R² ≥ 0.99 is normally reported as converged so the
    # user is not nagged — but NOT when an active $constraints line is still
    # violated: the optimizer was cut off before the penalty could pull the
    # parameters onto the constraint surface, and the user must know.
    _cviol = _max_constraint_violation(constraints or [], {**logK_vals, **fitted_logKs})
    _quiet_timeout = bool(_to and _r2 >= 0.99 and _cviol <= 0.02)
    _conv = result.success or ssr < 1e-6 or _quiet_timeout
    if _quiet_timeout:
        stats["timed_out"] = False
    stats["constraint_violation"] = _cviol
    _msg = "NMR shift fit complete"
    if params.get("range_extended_to_x") is not None:
        _msg += (f" (simulated to x = {params['range_extended_to_x']:.3g} to reach the last data point; "
                 f"raise xmax to see the curve there)")
    return _conv, fitted_logKs, stats, _msg



def _resolve_c(c: dict, sp: str, parsed: dict, x_grid: np.ndarray) -> np.ndarray:
    """
    Return the concentration array for `sp` on x_grid.
    Works for both real species (direct lookup) and $variables (sum of members).
    """
    if sp in c:
        return c[sp]
    variables = parsed.get("variables", {})
    if sp in variables:
        expr    = variables[sp]
        total   = np.zeros(len(x_grid))
        for part in expr.split("+"):
            part = part.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$", part)
            coeff, member = (float(m.group(1)), m.group(2)) if m else (1.0, part)
            total += coeff * c.get(member, np.zeros(len(x_grid)))
        return total
    return np.zeros(len(x_grid))


class BackCalc(dict):
    """{species: (x, c)} of back-calculated concentrations, each species at the
    rows where it was MEASURED: this is what a fit compares with the model.

    ``filled`` = (x_grid, {species: c on the whole grid}) carries the same
    species on the common grid, as the mass balance sees them (zero outside a
    species' measured range, interpolated across gaps inside it).  The plot
    uses it to evaluate $variables such as percentages at every row."""
    filled = None


def _merge_duplicate_integrals(cols):
    """Average the duplicate integration columns of one species, row by row.

    ``cols`` is a list of (x, ratio) pairs, one per column, each holding only
    the rows where that column has a value: the columns need not have the
    same rows (a signal that could not be integrated on some spectra leaves
    blank cells).  Returns (x_union, mean_ratio): at every x where at least
    one column has a value, the mean of the columns that do."""
    xs = np.unique(np.concatenate([np.asarray(x, float).ravel() for x, _ in cols]))
    acc = np.zeros(xs.size); cnt = np.zeros(xs.size)
    for x, r in cols:
        x = np.asarray(x, float).ravel(); r = np.asarray(r, float).ravel()
        ok = np.isfinite(x) & np.isfinite(r)
        idx = np.searchsorted(xs, x[ok])
        np.add.at(acc, idx, r[ok]); np.add.at(cnt, idx, 1.0)
    keep = cnt > 0
    return xs[keep], acc[keep] / cnt[keep]


def _ratio_on_grid(grid, x_sp, ratio_sp):
    """A species' ratio on the common grid, for the mass balance: interpolated
    across gaps inside its measured range, and ZERO outside it (before its
    first and after its last measured point the species is taken as absent,
    which is what a signal that has not appeared yet, or has gone, means)."""
    return np.interp(grid, x_sp, ratio_sp, left=0.0, right=0.0)


def _nmr_integration_backCalc(nmr_data: dict, n_H_list: list, params: dict,
                               network: dict, x_expr: str, parsed: dict,
                               x_col_header: str = "") -> dict:
    """
    Convert per-spectrum-normalised NMR integrations to absolute concentrations.

    Physical model
    ──────────────
    The NMR software normalises each spectrum so that the primary component
    signal (e.g. G) integrates to a fixed reference value (typically 1.0 × n_H).
    All other integrals in that spectrum are therefore relative to [G] at that
    titration point.

    For any species Sp:
        I_sp / n_H_sp  =  [Sp](x) / [G](x)     (ratio in the normalised spectrum)

    G mass-balance:
        [G_total](x)  =  Σ_{sp containing G}  stoich(G, sp) × [Sp](x)
                      =  [G](x) × Σ_{sp containing G}  stoich(G, sp) × ratio_sp(x)

    where  ratio_sp(x) = mean over duplicate signals of I_sp/n_H_sp.

    Therefore:
        [G](x)   = [G_total](x) / denom(x)
        [Sp](x)  = ratio_sp(x)  × [G](x)
        [G_total](x) = G0 × V0/(V0+V_add)   (liquid)  or  G0  (solid)

    Returns
    ───────
    {sp_name: (x_arr, c_bc_arr)}  — one entry per unique species.
    x_arr is in the same x-axis units as the plot.
    """
    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return {}

    all_sp       = network["all_species"]
    stoich       = network["stoich"]
    primary_sp   = params.get("primary_component", "")
    G0           = params["conc0"].get(primary_sp, 1.0)   # mM
    V0           = params["V0_mL"]
    is_solid     = params.get("titrant_is_solid", False)

    # ── Build (col, sp_name, n_H, raw_I, v_add_mL) list ─────────────────────
    # Species name extraction: a column header may be a real species (e.g. "GH"),
    # or a $variable name (e.g. "S012"), or a pandas-deduplicated duplicate of
    # either ("GH.1", "S012.2", …).  Strip the trailing ".N"/"_N" suffix first;
    # accept the result if it matches a real species OR a known $variable.
    # If neither, fall back to the column header as a last resort.
    variables_parsed = parsed.get("variables", {}) or {}
    entries = []
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp and sp not in variables_parsed:
            sp = col.split(".")[0]
        if sp not in all_sp and sp not in variables_parsed:
            sp = col
        n_H    = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0
        raw_I  = nmr_data[col]["y"]
        v_add  = nmr_data[col]["v_add_mL"]   # mL (liquid) or x-axis value (solid)
        x_exp  = convert_exp_x(v_add, x_expr, parsed, params, network,
                                x_col_header=x_col_header)
        entries.append((col, sp, n_H, raw_I, v_add, x_exp))

    # ── Internal standard ("$nmr standard: 1.0 mM") ───────────────────────────
    # A column headed Std / Standard holds the standard's integrals and normalises
    # the columns that follow it, up to the next Std column: every integral per
    # nucleus is divided by the standard's integral per nucleus of the same
    # spectrum and multiplied by the standard's concentration.  19F and 1H blocks,
    # each with its own standard column, thus land on one absolute mM scale and
    # may be averaged together as duplicates of the same species.
    _std_mM = (parsed.get("nmr") or {}).get("standard_mM")
    _is_std = lambda name: str(name).lower().startswith(("std", "standard"))
    if _std_mM:
        _scaled, _cur = [], None
        for col, sp, n_H, raw_I, v_add, x_exp in entries:
            if _is_std(sp) or _is_std(col):
                _cur = (np.asarray(v_add, float), np.asarray(raw_I, float) / max(n_H, 1e-12))
                continue
            if _cur is None:
                raise ValueError(f"'{col}': with an internal standard every integral column must "
                                 "follow a Std column (its block's standard).")
            _v = np.asarray(v_add, float)
            _std_here = np.interp(_v, _cur[0], _cur[1])
            _scaled.append((col, sp, 1.0, np.asarray(raw_I, float) / max(n_H, 1e-12) / np.maximum(_std_here, 1e-20) * float(_std_mM), v_add, x_exp))
        entries = _scaled                         # raw_I is now a concentration in mM, n_H = 1
    else:
        entries = [e for e in entries if not (_is_std(e[1]) or _is_std(e[0]))]

    # ── Average I/n_H across duplicate signals of each species ───────────────
    # The duplicate columns of a species need not have the same rows filled
    # (blank cells where a signal could not be integrated): they are averaged
    # row by row, over the columns that have a value there.
    _dups = {}
    for col, sp, n_H, raw_I, v_add, x_exp in entries:
        _dups.setdefault(sp, []).append((v_add, np.asarray(raw_I, float) / n_H))
    sp_ratio  = {}   # {sp: ratio_array}   ratio = mean(I/n_H)
    sp_x      = {}   # {sp: x_array}
    sp_v_add  = {}   # {sp: v_add_array}
    for sp, cols_ in _dups.items():
        v_u, r_u = _merge_duplicate_integrals(cols_)
        sp_v_add[sp] = v_u
        sp_ratio[sp] = r_u
        sp_x[sp] = convert_exp_x(v_u, x_expr, parsed, params, network, x_col_header=x_col_header)

    # ── G-containing species and their stoichiometric coefficients ───────────
    # A column may name a real species (e.g. GH3) OR a $variables expression
    # (e.g. G012 = G + GH + GH2).  For variables we expand the expression and
    # sum the stoich values of the member species — in practice each member
    # contains exactly 1 primary component, so stoich_eff = 1.
    variables_parsed = parsed.get("variables", {})

    def _effective_stoich(col_sp):
        """Return effective stoich of col_sp w.r.t. primary_sp."""
        # Direct species lookup
        s = stoich.get((primary_sp, col_sp), 0)
        if s > 0:
            return s
        # Variable: expand to member species and average their stoich
        if col_sp in variables_parsed:
            expr = variables_parsed[col_sp]
            members = [p.strip() for p in expr.split("+") if p.strip()]
            total = 0.0; count = 0
            for m in members:
                # Handle "2 * GH2" style coefficients
                import re as _re
                mm = _re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$", m)
                if mm:
                    coeff_m, sp_m = float(mm.group(1)), mm.group(2)
                else:
                    coeff_m, sp_m = 1.0, m
                s_m = stoich.get((primary_sp, sp_m), 0)
                if s_m > 0:
                    total += coeff_m * s_m; count += coeff_m
            # effective stoich = total G atoms per molecule summed over members /
            # total molecule count (= 1 when all members have stoich 1 for G)
            if count > 0:
                return total / count
        return 0.0

    g_sp_stoich = {sp: _effective_stoich(sp)
                   for sp in sp_ratio
                   if _effective_stoich(sp) > 0}

    if not g_sp_stoich:
        # fallback: use primary species itself only
        g_sp_stoich = {primary_sp: 1}
    # "$nmr pool: A, B": every integrated species holds one unit of the pool
    _pool = list(((parsed.get("nmr") or {}).get("pool")) or [])
    if _pool and not is_solid:
        g_sp_stoich = {sp: 1.0 for sp in sp_ratio}

    # ── Compute [G](x) and [Sp](x) for each x point ──────────────────────────
    # The common grid: every row of column A where at least one species has an
    # integral.  (Raw column A values: mL in liquid mode, x-axis values in solid.)
    v_add_ref = np.unique(np.concatenate([sp_v_add[sp] for sp in sp_ratio]))
    x_ref_arr = convert_exp_x(v_add_ref, x_expr, parsed, params, network, x_col_header=x_col_header)

    if is_solid:
        # Solid mode: volume is constant at V0, so [G_total] = G0 (no dilution).
        # v_add_ref here contains x-axis (equiv) values — used only for array shape.
        G_total = np.full_like(x_ref_arr, G0, dtype=float)
    elif _pool:
        # Liquid mode, pool normalisation: sum of the pool components' totals at each
        # point -- declared components are diluted, titrant components are added
        _tit_mMs = params.get("titrant_mMs") or {}
        _Vtot = np.maximum(V0 + v_add_ref, 1e-12)
        G_total = np.zeros_like(x_ref_arr, dtype=float)
        for _c in _pool:
            if _c in params["conc0"]:
                G_total = G_total + float(params["conc0"][_c]) * V0 / _Vtot
            if _c in _tit_mMs:
                G_total = G_total + float(_tit_mMs[_c]) * v_add_ref / _Vtot
    else:
        # Liquid mode: v_add_ref is volume of titrant added in mL.
        # G is diluted as V increases: [G_total](x) = G0 * V0 / (V0 + V_add)
        G_total = G0 * V0 / np.maximum(V0 + v_add_ref, 1e-12)

    # denom(x) = Σ stoich(G,sp) × ratio_sp(x)   over G-containing species
    # Interpolate each species ratio onto the shared x-axis grid (x_ref_arr).
    # For both liquid and solid modes, sp_x values are in x-axis (equiv) units.
    denom = np.zeros_like(G_total)
    for sp, stoich_coeff in g_sp_stoich.items():
        if sp not in sp_ratio:
            continue
        denom += stoich_coeff * _ratio_on_grid(v_add_ref, sp_v_add[sp], sp_ratio[sp])
    denom = np.maximum(denom, 1e-20)

    G_conc = G_total / denom   # mM
    if _std_mM and not _pool:
        G_conc = np.ones_like(G_total)            # already absolute (internal standard)

    # [Sp](x) = ratio_sp(x) × G_conc(x)  for ALL species (G-containing or not),
    # reported ONLY where that species was measured: a blank cell is not a data
    # point, so nothing interpolated or assumed is handed to the fit.
    bc = BackCalc()
    filled = {}
    for sp, ratio_arr in sp_ratio.items():
        at = np.searchsorted(v_add_ref, sp_v_add[sp])
        c_bc = np.clip(ratio_arr * G_conc[at], 0.0, None)
        bc[sp] = (x_ref_arr[at], c_bc)   # (x_arr in equiv units, concentration_mM)
        filled[sp] = np.clip(_ratio_on_grid(v_add_ref, sp_v_add[sp], ratio_arr) * G_conc, 0.0, None)
    bc.filled = (x_ref_arr, filled)

    return bc


def fit_nmr_integration(parsed: dict, network: dict, nmr_data: dict,
                        params: dict, logK_vals: dict, fit_keys: list,
                        x_expr: str, tolerance: float, maxiter: int,
                        timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True,
                   _objective_only: bool = False):
    """
    Fit equilibrium constants to slow-exchange NMR integration data.

    Physical model (per-spectrum normalisation):
        Each NMR spectrum is normalised to the primary-component signal = 1.
        Absolute concentrations are recovered via the G mass-balance:

            [Sp](x) = ratio_sp(x) × [G](x)
            [G](x)  = [G_total](x) / Σ stoich(G,sp)·ratio_sp(x)

        where ratio_sp = mean(I_sp/n_H_sp) over duplicate signals and
        [G_total](x) = G0·V0/(V0+V_add)  (liquid) or G0  (solid).

    The back-calculated concentrations are K-independent and computed once.
    The optimizer only moves K until the theoretical curves match them.
    """
    from scipy.optimize import minimize
    nmr_cfg  = parsed["nmr"]
    n_H_list = nmr_cfg.get("n_H_list", [])

    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return False, {}, {}, "No integration data columns found"

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # the simulation must reach every data point: a point beyond the plotted xmax
    # used to be compared with the END of the curve (the model at xmax, undiluted)
    params = params_covering_data(params, parsed, network, x_expr,
                                  [nmr_data[c]["v_add_mL"] for c in signal_cols], _x_col_header)

    # ── Back-calculate concentrations from integrals (K-independent) ─────────
    bc = _nmr_integration_backCalc(nmr_data, n_H_list, params, network, x_expr, parsed,
                                   x_col_header=_x_col_header)
    if not bc:
        return False, {}, {}, "Back-calculation failed"

    # Build col→sp and col→n_H maps for statistics output
    # (Mirrors the species-name extraction in _nmr_integration_backCalc:
    # accept stripped name if it's a real species OR a known $variable.)
    all_sp   = network["all_species"]
    _vars    = parsed.get("variables", {}) or {}
    col_to_sp  = {}
    col_to_nH  = {}
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp and sp not in _vars: sp = col.split(".")[0]
        if sp not in all_sp and sp not in _vars: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    # Unique species that have back-calculated data
    bc_species = list(bc.keys())


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else simplex_step(1.5)))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _get_bc(_p):
        """Return back-calculated concentrations for given params.
        Recomputed when fitting concentrations (G0 changes bc); cached otherwise."""
        if fitting_concs:
            return _nmr_integration_backCalc(nmr_data, n_H_list, _p, network, x_expr, parsed) or bc
        return bc

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    def objective(trial, bc_frozen):
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        ssr = 0.0
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(ssr))
        _live.notify(lk, ssr + cp + conc_penalty)
        return ssr + cp + conc_penalty

    def data_objective(full_trial, bc_frozen):
        """Data-only SSR for Hessian."""
        _p = _make_params_nmr(full_trial) if fitting_concs else params
        c = _simulate(full_trial[:len(fit_keys)], _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        ssr = 0.0
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        return ssr

    import time
    x0  = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p  = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k = len(fit_keys)
    _n_c = len(fit_conc_keys)


    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    # Returns the pieces an outer optimiser needs; nothing is optimised.
    if _objective_only:
        _bc_frz = _get_bc(_make_params_nmr(x0) if fitting_concs else params); _n_data_hint = int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_')))
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float), _get_bc(_make_params_nmr(np.asarray(t, dtype=float))) if fitting_concs else _bc_frz))),
            "data_ssr":  (lambda t: float(data_objective(np.asarray(t, dtype=float), _get_bc(_make_params_nmr(np.asarray(t, dtype=float))) if fitting_concs else _bc_frz))),
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys),
            "n_log_params": len(fit_keys),
            "n_data": _n_data_hint,
        }
    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _bc_ph0 = _get_bc(_make_params_nmr(x0) if fitting_concs else params)
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s, _bc=_bc_ph0):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]), _bc)
        try:
            _r0 = tied_minimize(_ph0_obj, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logK_vals,
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _total_evals   = 0
    _last_recent   = []
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        _p_x0    = _make_params_nmr(x0) if fitting_concs else params
        bc_pass  = _get_bc(_p_x0)
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0,
               "recent": collections.deque(maxlen=4 * (len(x0) + 1))}   # last trial points

        def _obj_timed(trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            _tracker["recent"].append(np.array(trial, dtype=float))
            f = objective(trial, _bc)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys), constraints=constraints or [], fixed=logK_vals,
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])
        _total_evals += _bt["nit"]
        _last_recent = list(_bt["recent"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    # ── How did the fit end?  (reported to the user; see stats below) ──
    # final_spread: how much the constants still moved among the last trial
    # points, in log units — the practical "precision reached", to compare
    # with the tolerance asked for.
    _stop_reason = ("timeout" if _any_timed_out else
                    ("tolerance" if getattr(result, "success", False) else "maxiter"))
    try:
        _final_spread = (float(np.max(np.ptp(np.array(_last_recent, dtype=float), axis=0)))
                         if len(_last_recent) > 1 else None)
    except Exception:
        _final_spread = None
    _elapsed_s = time.time() - _global_start
    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final_integ = _make_params_nmr(result.x) if fitting_concs else params

    # ── Statistics ────────────────────────────────────────────────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final_integ)
    bc_final = _get_bc(_params_final_integ)
    all_res = []
    all_obs = []
    # v2: per-species arrays for bootstrap resampling
    per_col_y_obs   = {}
    per_col_y_calc  = {}
    per_col_x       = {}

    if c_final is not None:
        x_sim_f, _ = evaluate_x_expression(x_expr, c_final, parsed)
        for sp, (x_bc, c_bc) in bc_final.items():
            c_th = np.interp(x_bc, x_sim_f, _resolve_c(c_final, sp, parsed, x_sim_f))
            all_res.extend((c_bc - c_th).tolist())
            all_obs.extend(c_bc.tolist())
            per_col_y_obs[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc[sp] = np.asarray(c_th, dtype=float)
            per_col_x[sp]      = np.asarray(x_bc, dtype=float)

    residuals = np.array(all_res)
    y_obs_arr = np.array(all_obs)
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((y_obs_arr - y_obs_arr.mean()) ** 2)) if len(y_obs_arr) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    # Parameter errors via finite-difference Hessian (skipped if compute_hessian=False)
    _bc_final_h = _get_bc(_params_final_integ)
    _n_total_params_integ = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    param_errors_integ = {}
    _err_notes = {}
    _cov_mat = None
    _cov_names_integ = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final_h),
                                   result.x, ssr, len(residuals), _n_total_params_integ,
                                   eq_matrix=_linear_eq_matrix(constraints or [], _cov_names_integ),
                                   n_log_params=len(fit_keys), notes=_err_notes)
        param_errors_integ = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_integ[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_integ[fit_titrant_keys[_i]] = _err_idx[_idx]

    # sp_concs format: {sp: [(x_arr, c_bc_arr), ...]}  (list for compatibility)
    sp_concs = {sp: [(x_bc, c_bc)] for sp, (x_bc, c_bc) in bc_final.items()}

    stats = {
        "r_squared":    r2,
        "rmse":         rmse,
        "ssr":          ssr,
        "n_points":     len(residuals),
        "n_params":     len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys),
        "param_values": fitted_logKs,
        "param_errors": param_errors_integ,
        "param_error_notes": {_cov_names_integ[i]: r for i, r in _err_notes.items() if i < len(_cov_names_integ)},
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_integ,
        "sp_concs":     sp_concs,
        "col_to_sp":    col_to_sp,
        "col_to_nH":    col_to_nH,
        "fit_mode":     "integration",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":       getattr(result, "nit", 0),
        "n_evals":         _total_evals,
        "stop_reason":     _stop_reason,
        "final_spread":    _final_spread,
        "elapsed_s":       _elapsed_s,
        "tolerance":       float(tolerance),
        "timed_out":    not getattr(result, "success", True),
        "pure_shifts":     {},   # not used in integration mode
        "delta_vecs_all":  {},
        "delta_bound_all": {},
        "delta_free":      {},
        "x_free_val":      {},
        "col_to_target":   {},
        "ref_corrections": {},
        # v2 diagnostics — flat residual arrays
        "y_obs":           np.asarray(y_obs_arr,             dtype=float),
        "y_calc":          np.asarray(y_obs_arr - residuals, dtype=float),
        "residuals":       np.asarray(residuals,             dtype=float),
        # v2 bootstrap — per-species arrays
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to   = stats.get("timed_out", False)
    _r2   = stats.get("r_squared", 0.0)
    # A timed-out fit with R² ≥ 0.99 is normally reported as converged so the
    # user is not nagged — but NOT when an active $constraints line is still
    # violated: the optimizer was cut off before the penalty could pull the
    # parameters onto the constraint surface, and the user must know.
    _cviol = _max_constraint_violation(constraints or [], {**logK_vals, **fitted_logKs})
    _quiet_timeout = bool(_to and _r2 >= 0.99 and _cviol <= 0.02)
    _conv = result.success or ssr < 1e-6 or _quiet_timeout
    if _quiet_timeout:
        stats["timed_out"] = False
    stats["constraint_violation"] = _cviol
    _msg = "NMR integration fit complete"
    if params.get("range_extended_to_x") is not None:
        _msg += (f" (simulated to x = {params['range_extended_to_x']:.3g} to reach the last data point; "
                 f"raise xmax to see the curve there)")
    return _conv, fitted_logKs, stats, _msg


def fit_nmr_mixed(parsed: dict, network: dict, nmr_data: dict,
                  params: dict, logK_vals: dict, fit_keys: list,
                  x_expr: str, tolerance: float, maxiter: int,
                  timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True,
                   _objective_only: bool = False, shared_dd=None):
    """
    Fit equilibrium constants to mixed slow+fast exchange NMR data.

    The $nmr section contains BOTH lines:
        integration: n1, n2, ...   <- first n_integ data columns are slow-exchange peaks
        shift: VarName             <- remaining columns are fast-exchange chemical shifts

    Column layout in the Excel file:
        [V_add] | [integ col 1] ... [integ col n_integ] | [shift col 1] ...

    Integration columns:
        Back-calculated to absolute concentrations via the G mass-balance
        (same as pure integration mode).  These are K-independent and computed once.

    Shift columns:
        Each belongs to the fast-exchange variable named in `shift:` (e.g. G012).
        δ_obs = Σᵢ [Spᵢ]·δᵢ / Σᵢ [Spᵢ]   where the sum runs over member species.
        Δδ values are solved analytically (linear least squares) inside the optimizer.

    Combined objective:
        SSR = Σ (c_bc - c_theory)²   [integration part, in mM²]
            + w · Σ (Δδ_obs - Δδ_calc)²   [shift part, in ppm²]
        where w scales the shift residuals to comparable magnitude.
    """
    from scipy.optimize import minimize

    nmr_cfg  = parsed["nmr"]
    n_H_list = nmr_cfg.get("n_H_list", [])
    n_integ  = nmr_cfg.get("n_integ", len(n_H_list))
    shift_targets = nmr_cfg.get("targets", [])
    _shift_constraints = parsed.get("shift_constraints", []) or []

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # ── Split nmr_data into integration and shift columns ────────────────────
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

    # the simulation must reach every data point: a point beyond the plotted xmax
    # used to be compared with the END of the curve (the model at xmax, undiluted)
    params = params_covering_data(params, parsed, network, x_expr,
                                  [nmr_data[c]["v_add_mL"] for c in all_cols], _x_col_header)

    # ── Back-calculate concentrations from integration columns (K-independent) ─
    bc = {}
    if integ_cols:
        bc = _nmr_integration_backCalc(integ_data, n_H_list[:n_integ],
                                        params, network, x_expr, parsed,
                                        x_col_header=_x_col_header)

    # ── Prepare shift data structures (same as fit_nmr_shifts) ───────────────
    col_to_target = {}
    for col in shift_cols:
        for tgt in shift_targets:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = shift_targets[0] if shift_targets else col

    # δ_free: first data point per shift column
    delta_free  = {}
    x_free_val  = {}
    for col in shift_cols:
        delta_free[col] = float(nmr_data[col]["y"][0])
        x_arr = convert_exp_x(nmr_data[col]["v_add_mL"], x_expr, parsed, params, network,
                               x_col_header=_x_col_header)
        x_free_val[col] = float(x_arr[0])

    # ── Pinned intrinsic shifts ($nmr 'read:' + sheet-2 known shifts) ─────
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _col_to_pinned, _pin_err = _build_pinned_shifts(
        _read_species, _known_shifts, shift_cols, delta_free,
        bool(nmr_cfg.get("noref", False)), col_to_bound=_col_to_bound)
    if _pin_err:
        return False, {}, {}, _pin_err
    _col_to_pinned_user = {c: dict(v or {}) for c, v in _col_to_pinned.items()}
    if shared_dd:
        # pure-species shifts solved across tabs by a global fit: every species of
        # the column is pinned, so the stats below describe THAT solution
        for _c, _pins in shared_dd.items():
            if _c in shift_cols:
                _col_to_pinned[_c] = {**(_col_to_pinned.get(_c) or {}), **{k: float(v) for k, v in _pins.items()}}
                _col_to_bound.pop(_c, None)

    # noref flag: when True, dd_ref is NOT auto-pinned to 0; instead at least
    # one read: anchor is required to fix the absolute scale.  Default False
    # preserves legacy "Δδ relative to ref" behavior so tutorials run unchanged.
    _nmr_noref = bool(nmr_cfg.get("noref", False))


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else simplex_step(1.5)))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _x_for_col(col):
        return convert_exp_x(nmr_data[col]["v_add_mL"], x_expr, parsed, params, network,
                             x_col_header=_x_col_header)

    def _shift_designs(c):
        """Per shift column (data order): the linear sub-problem X·dd = Δδ at
        curve ``c`` -- what solve_shifts consumes.  equilibrist_global stacks
        these across tabs when pure-species shifts are shared."""
        out = []
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
            x_exp     = _x_for_col(col)
            x_free_c  = x_free_val[col]

            # Weighted denominator
            denom_full = np.maximum(
                sum(coeff * c.get(sp, np.zeros_like(x_sim))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(x_free_c, x_sim, c.get(sp, np.zeros_like(x_sim))))
                for coeff, sp in sp_coeffs), 1e-20)

            # Build full design matrix (all species columns including ref)
            n_pts    = len(x_exp)
            all_target_sp = list(sp_coeffs)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim))) /                          np.maximum(np.interp(x_exp, x_sim, denom_full), 1e-20)
                F_ref  = coeff * float(np.interp(x_free_c, x_sim,
                         c.get(sp, np.zeros_like(x_sim)))) / denom_ref
                X[:, i] = F_full - F_ref

            out.append({"col": col, "target": tgt, "X": X,
                        "y": np.asarray(nmr_data[col]["y"] - delta_free[col], dtype=float),
                        "species": [sp for _, sp in all_target_sp], "ref": sp_coeffs[0][1],
                        "column_bound": _col_to_bound.get(col), "pinned": _col_to_pinned.get(col),
                        "constraints": _shift_constraints, "noref": _nmr_noref,
                        "delta_free": float(delta_free[col])})
        return out

    def _design_residuals(d):
        """Residuals of one column's sub-problem solved on its own."""
        X, dobs_rel = d["X"], d["y"]
        if X.shape[1] > 0 and np.any(np.abs(X) > 1e-15):
            pins, _weak = _pins_with_weak(X, d["species"], d["pinned"], d["constraints"])
            try:
                dd, calc, _ssr = solve_shifts(
                    X, dobs_rel, d["species"], d["ref"], d["constraints"],
                    column_bound=d["column_bound"], pinned_dd=pins, noref=d["noref"],
                )
                return dobs_rel - calc
            except ValueError:
                dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
                return dobs_rel - X @ dd
        return dobs_rel

    def _shift_ssr(c):
        """Compute shift SSR with analytic Δδ for current curve c."""
        return float(sum(np.sum(_design_residuals(d) ** 2) for d in _shift_designs(c)))

    def _get_bc_mixed(_p):
        """Recompute bc when fitting concentrations (G0 changes bc); cached otherwise."""
        if fitting_concs and integ_cols:
            return _nmr_integration_backCalc(integ_data, n_H_list[:n_integ],
                                             _p, network, x_expr, parsed) or bc
        return bc

    _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
    _shift_var_all = max(sum(float(np.var(nmr_data[col2]["y"]))
                             for col2 in shift_cols), 1e-20) if shift_cols else 1.0

    def _parts(trial, bc_frozen):
        """Everything of the objective except the shift solve:
        (bound penalty, trial log K dict, curve or None, integration SSR, shift variance)."""
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None:
            return conc_penalty, lk, None, 0.0, _shift_var_all
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        integ_ssr = 0.0
        integ_var = max(sum(float(np.var(c_bc)) for _, (_, c_bc) in bc_frozen.items()), 1e-20)
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            integ_ssr += float(np.sum((c_bc - c_th)**2)) / integ_var
        return conc_penalty, lk, c, integ_ssr, _shift_var_all

    def objective(trial, bc_frozen):
        conc_penalty, lk, c, integ_ssr, shift_var = _parts(trial, bc_frozen)
        if c is None: return 1e12
        total_ssr = integ_ssr + _shift_ssr(c) / shift_var
        cp = constraints_penalty(constraints or [], lk, ssr_scale=_pscale(total_ssr))
        _live.notify(lk, total_ssr + cp + conc_penalty)
        return total_ssr + cp + conc_penalty

    def data_objective(full_trial, bc_frozen):
        """Data-only SSR for Hessian."""
        _, _, c, integ_ssr, shift_var = _parts(full_trial, bc_frozen)
        if c is None: return 1e12
        return integ_ssr + _shift_ssr(c) / shift_var

    def shift_parts(trial, bc_frozen):
        """For a global fit with SHARED pure-species shifts: the per-column
        designs and the rest of the objective, so the outer solver can solve
        the shifts across tabs and finish the objective itself.  None when the
        curve cannot be computed."""
        conc_penalty, lk, c, integ_ssr, shift_var = _parts(trial, bc_frozen)
        if c is None: return None
        return {"integ_ssr": float(integ_ssr), "shift_var": float(shift_var),
                "designs": _shift_designs(c),
                "penalty": (lambda total_ssr: float(constraints_penalty(
                    constraints or [], lk, ssr_scale=_pscale(total_ssr)) + conc_penalty))}

    import time

    x0 = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p  = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k = len(fit_keys)
    _n_c = len(fit_conc_keys)


    # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
    # Returns the pieces an outer optimiser needs; nothing is optimised.
    if _objective_only:
        _bc_frz = _get_bc_mixed(_make_params_nmr(x0) if fitting_concs else params); _n_data_hint = int(sum(len(np.asarray(nmr_data[c]['y'])) for c in nmr_data if not str(c).startswith('_')))
        return {
            "objective": (lambda t: float(objective(np.asarray(t, dtype=float), _get_bc_mixed(_make_params_nmr(np.asarray(t, dtype=float))) if fitting_concs else _bc_frz))),
            "data_ssr":  (lambda t: float(data_objective(np.asarray(t, dtype=float), _get_bc_mixed(_make_params_nmr(np.asarray(t, dtype=float))) if fitting_concs else _bc_frz))),
            "shift_parts": (lambda t: shift_parts(np.asarray(t, dtype=float), _get_bc_mixed(_make_params_nmr(np.asarray(t, dtype=float))) if fitting_concs else _bc_frz)),
            "x0": np.asarray(x0, dtype=float),
            "names": list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys),
            "n_log_params": len(fit_keys),
            "n_data": _n_data_hint,
        }
    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _bc_ph0 = _get_bc_mixed(_make_params_nmr(x0) if fitting_concs else params)
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s, _bc=_bc_ph0):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]), _bc)
        try:
            _r0 = tied_minimize(_ph0_obj, x0[:_n_k], names=list(fit_keys), constraints=constraints or [], fixed=logK_vals,
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _total_evals   = 0
    _last_recent   = []
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        _p_x0    = _make_params_nmr(x0) if fitting_concs else params
        bc_pass  = _get_bc_mixed(_p_x0)
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0,
               "recent": collections.deque(maxlen=4 * (len(x0) + 1))}   # last trial points

        def _obj_timed(trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            _tracker["recent"].append(np.array(trial, dtype=float))
            f = objective(trial, _bc)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys), constraints=constraints or [], fixed=logK_vals,
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])
        _total_evals += _bt["nit"]
        _last_recent = list(_bt["recent"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    # ── How did the fit end?  (reported to the user; see stats below) ──
    # final_spread: how much the constants still moved among the last trial
    # points, in log units — the practical "precision reached", to compare
    # with the tolerance asked for.
    _stop_reason = ("timeout" if _any_timed_out else
                    ("tolerance" if getattr(result, "success", False) else "maxiter"))
    try:
        _final_spread = (float(np.max(np.ptp(np.array(_last_recent, dtype=float), axis=0)))
                         if len(_last_recent) > 1 else None)
    except Exception:
        _final_spread = None
    _elapsed_s = time.time() - _global_start
    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    timed_out = _any_timed_out
    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final_mixed = _make_params_nmr(result.x) if fitting_concs else params

    # ── Final pass: collect all stats and shift Δδ vectors ──────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final_mixed)
    sp_concs = {sp: [(x_bc, c_bc)] for sp, (x_bc, c_bc) in _get_bc_mixed(_params_final_mixed).items()}

    # Re-run shift analytics to get delta_vecs_all, pure_shifts etc.
    # (reuse the machinery from fit_nmr_shifts)
    delta_vecs_all  = {}
    delta_bound_all = {}
    pure_shifts     = {}
    pure_shift_undetermined = {}     # {target: {col: [species whose shift the data cannot fix]}}
    pure_shift_maxfrac = {}          # {target: {col: {species: largest change of its pool fraction over the data}}}
    pure_shift_pinned = {}           # {target: {col: [species pinned by the user's read: line]}}
    ref_corrections = {}
    # Track integration and shift residuals SEPARATELY to avoid unit mixing.
    # Integration residuals are in mM; shift residuals are in ppm (Δδ).
    # Mixing them into one array produces nonsensical R² and RMSE.
    integ_res = []; integ_obs = []
    shift_res = []; shift_obs = []   # stored as Δδ (relative), not absolute ppm
    # v2: per-source arrays for bootstrap (integration species + shift columns)
    per_col_y_obs_integ  = {}; per_col_y_calc_integ  = {}; per_col_x_integ  = {}
    per_col_y_obs_shift  = {}; per_col_y_calc_shift  = {}; per_col_x_shift  = {}

    if c_final is not None:
        x_sim_f, _ = evaluate_x_expression(x_expr, c_final, parsed)
        _bc_final_mixed = _get_bc_mixed(_params_final_mixed)

        # Integration residuals (mM)
        for sp, (x_bc, c_bc) in _bc_final_mixed.items():
            c_th = np.interp(x_bc, x_sim_f, _resolve_c(c_final, sp, parsed, x_sim_f))
            integ_res.extend((c_bc - c_th).tolist())
            integ_obs.extend(c_bc.tolist())
            per_col_y_obs_integ[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc_integ[sp] = np.asarray(c_th, dtype=float)
            per_col_x_integ[sp]      = np.asarray(x_bc, dtype=float)

        # Shift analytics (Δδ ppm)
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_exp     = _x_for_col(col)
            x_free_c  = x_free_val[col]
            df0       = delta_free[col]

            denom_full = np.maximum(
                sum(coeff * c_final.get(sp, np.zeros_like(x_sim_f))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(x_free_c, x_sim_f, c_final.get(sp, np.zeros_like(x_sim_f))))
                for coeff, sp in sp_coeffs), 1e-20)

            # Build full design matrix (all species columns including ref)
            n_pts    = len(x_exp)
            all_target_sp = list(sp_coeffs)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(x_exp, x_sim_f, c_final.get(sp, np.zeros_like(x_sim_f))) /                          np.maximum(np.interp(x_exp, x_sim_f, denom_full), 1e-20)
                F_ref  = coeff * float(np.interp(x_free_c, x_sim_f,
                         c_final.get(sp, np.zeros_like(x_sim_f)))) / denom_ref
                X[:, i] = F_full - F_ref

            dobs_rel = nmr_data[col]["y"] - df0
            _col_bd = _col_to_bound.get(col)
            _sp_names = [sp for _, sp in all_target_sp]
            _col_pn, _ = _pins_with_weak(X, _sp_names, _col_to_pinned.get(col), _shift_constraints)
            pure_shift_undetermined.setdefault(tgt, {})[col] = _weak_species(
                X, _sp_names, _col_to_pinned_user.get(col), _shift_constraints)
            pure_shift_maxfrac.setdefault(tgt, {})[col] = {
                sp: float(np.max(np.abs(X[:, i]))) if X.shape[0] else 0.0 for i, sp in enumerate(_sp_names)}
            pure_shift_pinned.setdefault(tgt, {})[col] = sorted(_col_to_pinned_user.get(col) or {})
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
                dd = np.array([])
                calc_rel = np.zeros_like(dobs_rel)

            # Use Δδ (relative shifts) — NOT absolute ppm — for statistics
            shift_res.extend((dobs_rel - calc_rel).tolist())
            shift_obs.extend(dobs_rel.tolist())
            per_col_y_obs_shift[col]  = np.asarray(dobs_rel, dtype=float)
            per_col_y_calc_shift[col] = np.asarray(calc_rel, dtype=float)
            per_col_x_shift[col]      = np.asarray(x_exp,    dtype=float)

            # In the new convention dd[i] = δ_i_absolute − δ_obs(V=0).
            # Legacy (no noref): dd[ref] = 0 by auto-pin.  Noref: dd[ref] is fitted.
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

            # Pure-species shifts: stored as dd values (= δ_absolute − δ_obs(V=0)).
            # Display/export adds δ_obs(V=0) back when the scale is anchored
            # (noref=True with at least one read: pin).
            # ref_corrections[col] is defined as Σᵢ Fᵢ(x_ref)·ddᵢ over ALL
            # species (incl. reference).  In noref+read mode dd[ref] may
            # be nonzero so the ref species term must be kept.
            ref_corrections[col] = sum(
                coeff * float(np.interp(x_free_c, x_sim_f,
                              c_final.get(sp, np.zeros_like(x_sim_f)))) / denom_ref
                * sp_dd.get(sp, 0.0)
                for coeff, sp in sp_coeffs   # all species, incl. ref
            )
            if tgt not in pure_shifts:
                pure_shifts[tgt] = {}
            pure_shifts[tgt][col] = dict(sp_dd)

    # ── Per-component statistics ─────────────────────────────────────────────
    def _r2_rmse(res_list, obs_list):
        if not res_list: return 0.0, 0.0, 0.0
        r = np.array(res_list); o = np.array(obs_list)
        ssr_ = float(np.sum(r**2))
        sst_ = float(np.sum((o - o.mean())**2)) if len(o) > 1 else 1.0
        return 1.0 - ssr_/max(sst_, 1e-30), float(np.sqrt(ssr_/max(len(r),1))), ssr_

    r2_integ, rmse_integ, ssr_integ = _r2_rmse(integ_res, integ_obs)
    r2_shift, rmse_shift, ssr_shift = _r2_rmse(shift_res, shift_obs)

    # Combined data point count
    n_integ_pts = len(integ_res)
    n_shift_pts = len(shift_res)
    n_total     = n_integ_pts + n_shift_pts

    # Overall R²: weighted average by data point count
    r2   = (r2_integ * n_integ_pts + r2_shift * n_shift_pts) / max(n_total, 1)
    ssr  = ssr_integ + ssr_shift
    rmse = float(np.sqrt(ssr / max(n_total, 1)))   # mixed units — shown separately

    # v2: combined flat residuals (integration first, then shift)
    _mixed_yobs = np.asarray(list(integ_obs) + list(shift_obs), dtype=float)
    _mixed_res  = np.asarray(list(integ_res) + list(shift_res), dtype=float)
    _mixed_ycalc = _mixed_yobs - _mixed_res

    # Parameter errors via finite-difference Hessian (skipped if compute_hessian=False).
    # For mixed mode the objective is dimensionless (normalized chi-squared), so
    # σ² is just 1/(n-p) — we're already working in units of variance.
    # Use data_objective (no penalty) so constraint curvature doesn't dominate.
    _n_total_params_mixed = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _bc_final_mixed_h = _get_bc_mixed(_params_final_mixed)
    param_errors_mixed = {}
    _err_notes = {}
    _cov_mat = None
    _cov_names_mixed = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final_mixed_h),
                                   result.x, ssr, n_total, _n_total_params_mixed,
                                   eq_matrix=_linear_eq_matrix(constraints or [], _cov_names_mixed),
                                   n_log_params=len(fit_keys), notes=_err_notes)
        param_errors_mixed = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_mixed[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_mixed[fit_titrant_keys[_i]] = _err_idx[_idx]

    # ── Variance-normalized chi²-RMSE for the LST ──────────────────────────
    # The optimizer minimized the variance-normalized objective
    # chi² = integ_ssr/integ_var + shift_ssr_raw/shift_var (mixed units
    # are made comparable by the per-component variance weights).  The
    # raw `rmse` reported above mixes mM² and ppm² and has its minimum
    # at a DIFFERENT point than the chi² minimum where the fit lives,
    # so the local-sensitivity test (which uses paired-difference
    # geometry that only makes sense at a local minimum of the probed
    # objective) needs the chi² value, not the raw rmse.  We expose
    # both: `rmse` stays raw for backward-compat / display, and
    # `chi2_rmse` is what the LST reads.
    try:
        _chi2_final = float(data_objective(result.x, _bc_final_mixed_h))
        _chi2_rmse  = float(np.sqrt(max(_chi2_final, 0.0) / max(n_total, 1)))
    except Exception:
        _chi2_rmse = float(rmse)  # fallback

    # Build col_to_sp and col_to_nH for display
    # (Mirrors the species-name extraction in _nmr_integration_backCalc:
    # accept stripped name if it's a real species OR a known $variable.)
    all_sp_net = network["all_species"]
    _vars      = parsed.get("variables", {}) or {}
    col_to_sp = {}; col_to_nH = {}
    for idx, col in enumerate(integ_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp_net and sp not in _vars: sp = col.split(".")[0]
        if sp not in all_sp_net and sp not in _vars: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "chi2_rmse":       _chi2_rmse,
        "ssr":             ssr,
        "n_points":        n_total,
        "n_params":        len(fit_keys),
        "param_values":    fitted_logKs,
        "param_errors":    param_errors_mixed,
        "param_error_notes": {_cov_names_mixed[i]: r for i, r in _err_notes.items() if i < len(_cov_names_mixed)},
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_mixed,
        "sp_concs":        sp_concs,
        "col_to_sp":       col_to_sp,
        "col_to_nH":       col_to_nH,
        "pure_shifts":     pure_shifts,
        "pure_shift_undetermined": pure_shift_undetermined,
        "pure_shift_maxfrac": pure_shift_maxfrac,
        "pure_shift_pinned": pure_shift_pinned,        # dd values (= δ − δ_obs(V=0))
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "delta_vecs_all":  delta_vecs_all,
        "delta_bound_all": delta_bound_all,
        "delta_free":      delta_free,
        "x_free_val":      x_free_val,
        "col_to_target":   col_to_target,
        "ref_corrections": ref_corrections,
        "fit_mode":        "mixed",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":          getattr(result, "nit", 0),
        "n_evals":         _total_evals,
        "stop_reason":     _stop_reason,
        "final_spread":    _final_spread,
        "elapsed_s":       _elapsed_s,
        "tolerance":       float(tolerance),
        "timed_out":       timed_out,
        # Per-component stats (integration in mM, shift in ppm Δδ)
        "r2_integ":        r2_integ,
        "rmse_integ":      rmse_integ,
        "n_integ_pts":     n_integ_pts,
        "r2_shift":        r2_shift,
        "rmse_shift":      rmse_shift,
        "n_shift_pts":     n_shift_pts,
        # v2 diagnostics — combined flat residuals (integration block first, shift block second)
        "y_obs":           _mixed_yobs,
        "y_calc":          _mixed_ycalc,
        "residuals":       _mixed_res,
        # v2 bootstrap — per-source arrays (split for within-block resampling)
        "per_col_y_obs_integ":  per_col_y_obs_integ,
        "per_col_y_calc_integ": per_col_y_calc_integ,
        "per_col_x_integ":      per_col_x_integ,
        "per_col_y_obs_shift":  per_col_y_obs_shift,
        "per_col_y_calc_shift": per_col_y_calc_shift,
        "per_col_x_shift":      per_col_x_shift,
        # v2 unified per_col_* — what the residuals-vs-predictor scatter
        # consumes via collect_residuals_from_stats.  Same tagging
        # scheme as fit_kinetics_nmr_mixed so the legend distinguishes
        # integration from shift columns.
        "per_col_x":      {**{f"{c} (int)":   v for c, v in per_col_x_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_x_shift.items()}},
        "per_col_y_obs":  {**{f"{c} (int)":   v for c, v in per_col_y_obs_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_obs_shift.items()}},
        "per_col_y_calc": {**{f"{c} (int)":   v for c, v in per_col_y_calc_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_calc_shift.items()}},
    }
    _r2   = stats.get("r_squared", 0.0)
    # A timed-out fit with R² ≥ 0.99 is normally reported as converged so the
    # user is not nagged — but NOT when an active $constraints line is still
    # violated: the optimizer was cut off before the penalty could pull the
    # parameters onto the constraint surface, and the user must know.
    _cviol = _max_constraint_violation(constraints or [], {**logK_vals, **fitted_logKs})
    _quiet_timeout = bool(timed_out and _r2 >= 0.99 and _cviol <= 0.02)
    _conv = result.success or ssr < 1e-6 or _quiet_timeout
    if _quiet_timeout:
        stats["timed_out"] = False
    stats["constraint_violation"] = _cviol
    _msg = "Mixed NMR fit complete"
    if params.get("range_extended_to_x") is not None:
        _msg += (f" (simulated to x = {params['range_extended_to_x']:.3g} to reach the last data point; "
                 f"raise xmax to see the curve there)")
    return _conv, fitted_logKs, stats, _msg


# ─────────────────────────────────────────────
