"""
equilibrist_global -- one fit across several experiments (tabs).

Each experiment is described by the same ``fit_args`` dict the single-experiment
tools use (parsed script, network, params, data, fit keys, constraints ...).
The fit functions hand out their objective without optimising
(``_objective_only=True``); this module stitches those objectives together,
runs one Nelder-Mead over the union of parameters, estimates errors from the
Hessian of the summed data-only SSR, and finally evaluates every experiment at
the optimum (one-shot call of its fit function) so each tab receives the usual
``stats`` dict for its plots and diagnostics.

Sharing rule
------------
* an equilibrium constant is shared BY NAME: one value for every tab in which
  it occurs; it is fitted globally when it is ticked "fit" in at least one tab;
* a constant that occurs in one tab only is simply that tab's parameter;
* fitted concentrations / titrant stocks are always local to their tab;
* every tab applies its own bounds and $constraints (so bounds intersect and
  constraints accumulate);
* the per-experiment objectives are each normalised to their own data variance
  (as in the single fits) and summed with unit weight.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import equilibrist_live as _live
from scipy.optimize import minimize

from equilibrist_parser import FitTrace, simplex_step_override


# ── Building one experiment's objective ────────────────────────────────────
def _fit_function(fa: dict):
    """The fit function and a kind tag for one experiment's fit args."""
    if fa.get("kind") == "kin":
        if fa.get("use_spectra_fit"):
            from equilibrist_kinetics_spectra import fit_kinetics_spectra
            return "kin_spectra", fit_kinetics_spectra
        if fa.get("use_nmr_fit"):
            mode = fa.get("nmr_mode")
            import equilibrist_kinetics_nmr as m
            return "kin_" + str(mode), {"shift": m.fit_kinetics_nmr_shifts,
                                        "integration": m.fit_kinetics_nmr_integration,
                                        "mixed": m.fit_kinetics_nmr_mixed}[mode]
        from equilibrist_kinetics import fit_kinetics
        return "kin_conc", fit_kinetics
    if fa.get("use_itc_fit"):
        from equilibrist_itc import fit_itc
        return "itc", fit_itc
    if fa.get("use_spectra_fit"):
        from equilibrist_fit_spectra import fit_spectra
        return "spectra", fit_spectra
    if fa.get("use_nmr_fit"):
        mode = fa.get("nmr_mode")
        import equilibrist_fit_nmr as m
        return mode, {"shift": m.fit_nmr_shifts, "integration": m.fit_nmr_integration,
                      "mixed": m.fit_nmr_mixed}[mode]
    from equilibrist_fit_conc import fit_parameters
    return "conc", fit_parameters


def _call_fit(fa: dict, logK: dict, fit_keys: list, *, tolerance: float, maxiter: int,
              timeout_s: float, compute_hessian: bool, objective_only: bool):
    """Call the experiment's fit function with a uniform argument set."""
    kind, f = _fit_function(fa)
    p = fa["parsed_fit"]
    cstr = fa.get("constraints") or []
    fck = list(fa.get("fit_conc_keys") or [])
    ftk = list(fa.get("fit_titrant_keys") or [])
    if kind.startswith("kin_"):
        t_max = float(fa.get("t_max") or 0.0)
        if kind == "kin_spectra":
            return f(p, logK, fa["spectra_data_fit"], fit_keys, t_max, 200,
                     float(fa.get("wl_min") or 0.0), float(fa.get("wl_max") or 1e9),
                     tolerance, maxiter, timeout_s=timeout_s, constraints=cstr, fit_conc_keys=fck,
                     allow_negative_eps=bool(fa.get("allow_neg_eps", False)),
                     compute_hessian=compute_hessian, _objective_only=objective_only)
        if kind in ("kin_shift", "kin_integration", "kin_mixed"):
            return f(p, logK, fa["nmr_data_fit"], fit_keys, t_max, 200, tolerance, maxiter,
                     timeout_s=timeout_s, constraints=cstr, fit_conc_keys=fck,
                     compute_hessian=compute_hessian, _objective_only=objective_only)
        return f(p, fa.get("exp_data_fit") or {}, logK, fit_keys, t_max, 200, tolerance, maxiter,
                 timeout_s=timeout_s, constraints=cstr, fit_conc_keys=fck,
                 compute_hessian=compute_hessian, _objective_only=objective_only)
    net, par, xe = fa["network"], fa["params"], fa["x_expr"]
    if kind == "itc":
        return f(p, net, fa["itc_q_exp"], fa["itc_raw"], par, logK, fit_keys, xe,
                 tolerance=tolerance, maxiter=maxiter, timeout_s=timeout_s,
                 fit_conc_keys=fck, fit_titrant_keys=ftk,
                 q_offset_fixed=fa.get("q_offset_fixed"), constraints=cstr,
                 excl_set=fa.get("excl_set") or set(), _objective_only=objective_only)
    if kind == "spectra":
        return f(p, net, fa["spectra_data_fit"], par, logK, fit_keys, xe,
                 fa.get("wl_min"), fa.get("wl_max"), tolerance, maxiter,
                 auto_range=False, timeout_s=timeout_s, constraints=cstr,
                 fit_conc_keys=fck, fit_titrant_keys=ftk,
                 allow_negative_eps=bool(fa.get("allow_neg_eps", False)),
                 compute_hessian=compute_hessian, _objective_only=objective_only)
    if kind in ("shift", "integration", "mixed"):
        extra = ({"shared_dd": fa["shared_dd"]}
                 if (fa.get("shared_dd") and kind in ("shift", "mixed")) else {})
        return f(p, net, fa["nmr_data_fit"], par, logK, fit_keys, xe,
                 tolerance, maxiter, timeout_s=timeout_s, constraints=cstr,
                 fit_conc_keys=fck, fit_titrant_keys=ftk,
                 compute_hessian=compute_hessian, _objective_only=objective_only, **extra)
    return f(p, net, fa.get("exp_data_fit") or {}, par, logK, fit_keys, xe,
             tolerance=tolerance, maxiter=maxiter, constraints=cstr,
             fit_conc_keys=fck, fit_titrant_keys=ftk,
             compute_hessian=compute_hessian, _objective_only=objective_only)


def experiment_constants(fa: dict) -> list:
    """Names of the constants that exist in this experiment: equilibrium
    constants and, for kinetics scripts, forward / reverse rate constants."""
    p = fa["parsed_fit"]
    names = [eq["kname"] for eq in p.get("equilibria", []) if eq.get("kname")]
    for r in p.get("kinetics", []) or []:
        for key in ("kname", "krname"):
            if r.get(key) and r[key] not in names:
                names.append(r[key])
    return names


# ── The global fit ─────────────────────────────────────────────────────────
SHIFT_MATCH_TOL = 0.05      # ppm: two columns closer than this at V = 0 are "the same proton"


def match_shift_columns(fingerprints: list, tol: float = SHIFT_MATCH_TOL):
    """Pair the tabs' shift columns proton by proton.

    ``fingerprints[t]`` is the list of δ_obs(V = 0) of tab t's shift columns, in
    data order.  The same proton has the same free shift in every tab (within
    referencing noise), whatever column it was typed in, so columns are matched
    by nearest free shift within ``tol``; a column with no partner stays
    tab-specific.  When a tab holds two protons closer than ``tol`` the
    fingerprints cannot tell them apart: the columns are then paired by position
    if every tab has the same number of them, otherwise a ValueError says so.

    Returns ``(groups, how)``: ``groups`` is a list of per-proton lists giving, for
    every tab, the column index or None; ``how`` is "fingerprint" or "position".
    """
    n_tabs = len(fingerprints)
    ambiguous = any(abs(a - b) < tol for fp in fingerprints
                    for i, a in enumerate(fp) for b in fp[i + 1:])
    if ambiguous:
        counts = {len(fp) for fp in fingerprints}
        if len(counts) != 1:
            raise ValueError("two shift columns of one tab have the same free shift (within "
                             f"{tol:g} ppm) and the tabs have different numbers of shift columns, "
                             "so the protons cannot be paired across tabs")
        return [[j] * n_tabs for j in range(counts.pop())], "position"
    groups = [[j] + [None] * (n_tabs - 1) for j in range(len(fingerprints[0]))]
    for t in range(1, n_tabs):
        for j, v in enumerate(fingerprints[t]):
            best, dist = None, tol
            for gi, grp in enumerate(groups):
                anchors = [fingerprints[u][grp[u]] for u in range(t) if grp[u] is not None]
                if not anchors or grp[t] is not None:
                    continue
                dd = min(abs(v - a) for a in anchors)
                if dd < dist:
                    best, dist = gi, dd
            if best is None:
                grp = [None] * n_tabs; grp[t] = j; groups.append(grp)
            else:
                groups[best][t] = j
    return groups, "fingerprint"


def _shared_shift_solve(parts: list, weights: list):
    """Solve the pure-species shifts ONCE across tabs.

    ``parts[t]`` is what a tab's ``shift_parts`` handle returned: its shift
    columns' designs (X·dd = Δδ, data order), its integration SSR and shift
    variance.  Columns are matched proton by proton (:func:`match_shift_columns`,
    by their free shift); for each proton the tabs' rows are stacked, scaled by
    sqrt(w_t / var_t) so the stacked least squares minimises exactly the weighted
    global objective, and solved by ``solve_shifts`` with the union of the tabs'
    species, the merged ``read:`` pins and the tightest column bound.  A proton
    present in one tab only is solved as before.  Shifts are relative to each
    tab's own δ_obs(V=0); with ``noref`` (absolute anchoring) tabs whose
    reference rows differ are brought to the first tab's gauge,
    dd_t = dd_0 − (δ_free,t − δ_free,0).

    Returns ``(ssr_per_tab, shared)``: each tab's shift SSR with the shared
    solution and, per proton, ``{"species", "dd0", "gauge", "cols", "how"}``
    (``cols[t]`` is None for a tab without that proton).
    """
    from equilibrist_shift_constraints import solve_shifts
    groups, how = match_shift_columns([[float(d["delta_free"]) for d in p["designs"]] for p in parts])
    ssr = [0.0] * len(parts)
    shared = []
    for grp in groups:
        ds = [(t, parts[t]["designs"][j]) for t, j in enumerate(grp) if j is not None]
        t0, d0 = ds[0]
        species = []
        for _, d in ds:
            for sp in d["species"]:
                if sp not in species:
                    species.append(sp)
        gauge = {t: ((float(d["delta_free"]) - float(d0["delta_free"])) if d0.get("noref") else 0.0) for t, d in ds}
        pins = {}
        for t, d in ds:
            for sp, v in (d.get("pinned") or {}).items():
                pins.setdefault(sp, float(v) + gauge[t])
        bounds = [abs(float(d["column_bound"])) for _, d in ds if d.get("column_bound") is not None]
        bound = min(bounds) if bounds else None
        rows_X, rows_y, plain = [], [], {}
        for t, d in ds:
            Xt = np.zeros((d["X"].shape[0], len(species)))
            for i, sp in enumerate(d["species"]):
                Xt[:, species.index(sp)] = d["X"][:, i]
            s = float(np.sqrt(max(float(weights[t]), 0.0) / max(float(parts[t]["shift_var"]), 1e-20)))
            rows_X.append(Xt * s); rows_y.append(np.asarray(d["y"], dtype=float) * s); plain[t] = Xt
        X = np.vstack(rows_X); y = np.concatenate(rows_y)
        # a species below 2 % of the pool in EVERY tab stays at the free shift (see _weak_species)
        from equilibrist_fit_nmr import _weak_species
        for sp in _weak_species(np.vstack([plain[t] for t, _ in ds]), species, pins, d0.get("constraints")):
            pins[sp] = 0.0
        if X.shape[1] and np.any(np.abs(X) > 1e-15):
            try:
                dd, _, _ = solve_shifts(X, y, species, d0["ref"], d0.get("constraints"),
                                        column_bound=bound, pinned_dd=pins or None,
                                        noref=bool(d0.get("noref")))
            except ValueError:
                dd = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            dd = np.zeros(len(species))
        for t, d in ds:
            calc = plain[t] @ dd
            ssr[t] += float(np.sum((np.asarray(d["y"], dtype=float) - calc) ** 2))
        shared.append({"species": species, "dd0": np.asarray(dd, dtype=float),
                       "gauge": [gauge.get(t) for t in range(len(parts))],
                       "cols": [parts[t]["designs"][j]["col"] if j is not None else None for t, j in enumerate(grp)],
                       "delta_free0": float(d0["delta_free"]), "how": how})
    return ssr, shared


def shift_columns_line_up(fit_args_list: list):
    """Can these tabs share pure-species shifts?  ``(ok, why)``: every tab must
    be an NMR shift or mixed fit with the same shift target variables, and the
    protons must be matchable across tabs (see :func:`match_shift_columns`)."""
    fps, targets = [], set()
    for fa in fit_args_list:
        try:
            kind = _fit_function(fa)[0]
        except Exception:
            return False, "a tab's fit could not be set up"
        if kind not in ("shift", "mixed"):
            return False, "every tab needs NMR chemical-shift columns (shift or mixed mode)"
        p = fa.get("parsed_fit") or {}
        cfg = p.get("nmr") or {}
        data = fa.get("nmr_data_fit") or {}
        cols = [c for c in data if not str(c).startswith("_")]
        n_integ = int(cfg.get("n_integ") or 0) if kind == "mixed" else 0
        try:
            fps.append([float(np.asarray(data[c]["y"], dtype=float)[0]) for c in cols[n_integ:]])
        except Exception:
            return False, "a tab's shift columns could not be read"
        targets.add(tuple(cfg.get("targets") or ()))
    if len(fps) < 2:
        return False, "at least two tabs are needed"
    if len(targets) != 1:
        return False, "the tabs' shift columns belong to different target variables"
    try:
        groups, how = match_shift_columns(fps)
    except ValueError as ex:
        return False, str(ex)
    if not any(sum(j is not None for j in grp) >= 2 for grp in groups):
        return False, ("no proton is common to two tabs (free shifts differ by more than "
                       f"{SHIFT_MATCH_TOL:g} ppm everywhere)")
    return True, ""


@_live.suppressed
def global_fit(experiments: list, *, tolerance: float = 1e-3, maxiter: int = 100000,
               timeout_s: float = 300.0, simplex_step: float = 1.5,
               progress: Optional[Callable] = None,
               with_stats: bool = True, with_errors: bool = True,
               weighted: bool = True, fixed_weights: Optional[dict] = None,
               share_shifts: bool = False) -> dict:
    """Fit all ``experiments`` at once.

    ``experiments``: list of ``{"label": str, "fa": fit_args, "start_logK": {name: log K},
    "ticked": [constants ticked "fit" in that tab]}``.

    With ``weighted`` every tab's (dimensionless) objective is rescaled to its own
    misfit, w_i = n_i / obj_i, re-estimated after each optimiser pass, so a tab the
    model fits badly cannot outvote a clean one; the result carries the relative
    weights (``abs_weights``).  Without ``weighted`` the objectives are plainly
    summed.  ``fixed_weights`` {label: w} freezes the weights -- what every
    diagnostic and resampling of a fit must do, so that a profile point is
    scored with the weights of the fit it belongs to and not rescaled to itself.

    With ``share_shifts`` the pure-species chemical shifts are solved ONCE across
    the tabs (same compound, same protons: see :func:`_shared_shift_solve`) instead
    of separately in every tab; every tab's stats then describe that common
    solution (its species shifts are pinned to it in the final one-shot).

    Returns ``{"success", "message", "stats": {label: stats_dict}, "shared": {...},
    "global": {...}}``; every ``stats_dict`` is the usual per-experiment stats
    (rmse, residuals, back-calcs ...) with the global errors written into it.
    """
    t0 = time.time()
    if len(experiments) < 2:
        raise ValueError("A global fit needs at least two experiments.")

    # 1. which constants are fitted globally, and which tabs hold them
    ticked_any = []
    for e in experiments:
        for k in e.get("ticked") or []:
            if k not in ticked_any:
                ticked_any.append(k)
    per_exp_keys = []
    for e in experiments:
        have = set(experiment_constants(e["fa"]))
        per_exp_keys.append([k for k in ticked_any if k in have])
    # nothing to vary at all -> evaluation only (profiles pin everything at times)
    evaluate_only = (not any(per_exp_keys) and not any(
        e["fa"].get("fit_conc_keys") or e["fa"].get("fit_titrant_keys") for e in experiments))

    # 2. one shared starting value per constant (first tab that has it, unless
    #    the caller provided a preferred value in start_logK of a later tab)
    shared_start = {}
    for e in experiments:
        for k, v in (e.get("start_logK") or {}).items():
            if k in ticked_any and k not in shared_start:
                shared_start[k] = float(v)

    # 3. hand-out objectives.  A tab in which nothing varies (no shared constant
    #    present, nothing local ticked) cannot take part; say so instead of failing.
    handles, skipped = [], []
    for e, keys in zip(experiments, per_exp_keys):
        fa = e["fa"]
        if not keys and not (fa.get("fit_conc_keys") or fa.get("fit_titrant_keys")) and not evaluate_only:
            have = experiment_constants(fa)
            skipped.append(f"{e['label']} (its constants are {', '.join(have) or 'none'}; none of the "
                           f"ticked ones -- {', '.join(ticked_any) or 'none'} -- occurs in it)")
            continue
        logK = dict(e.get("start_logK") or {})
        for k in keys:
            logK[k] = shared_start.get(k, logK.get(k, 0.0))
        if evaluate_only:
            # no objective needed: the one-shot below evaluates the point.  The
            # fit function wants at least one "free" key, which the one-shot
            # (maxiter=1, tiny simplex) leaves exactly where it is.
            have = experiment_constants(fa)
            h = {"objective": None, "data_ssr": None, "x0": np.zeros(0), "names": [],
                 "n_log_params": 0, "n_data": 0, "label": e["label"],
                 "keys": [have[0]] if have else [], "logK0": logK, "_eval_only": True}
            # the objective at the pinned point, so the scalar the diagnostics
            # compare (chi_rmse) exists in evaluation-only mode too
            try:
                h2 = _call_fit(fa, logK, [have[0]], tolerance=tolerance, maxiter=maxiter,
                               timeout_s=1e9, compute_hessian=False, objective_only=True) if have else None
                if isinstance(h2, dict) and h2.get("data_ssr") is not None:
                    h["f_const"] = float(h2["data_ssr"](np.asarray(h2["x0"], float)))
                    h["n_data"] = int(h2.get("n_data") or 0)
                    if share_shifts and h2.get("shift_parts") is not None:
                        h["_parts_const"] = h2["shift_parts"](np.asarray(h2["x0"], float))
            except Exception:
                pass
            handles.append(h)
            continue
        h = _call_fit(fa, logK, keys, tolerance=tolerance, maxiter=maxiter,
                      timeout_s=1e9, compute_hessian=False, objective_only=True)
        if not isinstance(h, dict) or "objective" not in h:
            why = ""
            if isinstance(h, tuple) and len(h) >= 4:
                why = f": {h[3]}"
            raise RuntimeError(f"{e['label']}: the fit function could not set up this experiment{why}")
        h["label"] = e["label"]; h["keys"] = keys; h["logK0"] = logK
        handles.append(h)
    if len(handles) < 2 and not evaluate_only:
        raise ValueError("A global fit needs at least two tabs with something to fit.  "
                         + ("Left out: " + "; ".join(skipped) + ".  " if skipped else "")
                         + "Constants are shared by NAME: tick 'fit' on a constant that exists in both "
                           "scripts, or tick one in each tab.")
    experiments = [e for e, keys in zip(experiments, per_exp_keys)
                   if any(h["label"] == e["label"] for h in handles)]
    if share_shifts:
        lacking = [h["label"] for h in handles
                   if (h.get("_parts_const") is None if h.get("_eval_only") else h.get("shift_parts") is None)]
        if lacking:
            raise ValueError("Shared pure-species shifts need NMR chemical-shift columns in every tab "
                             "(shift or mixed mode); not the case for: " + ", ".join(lacking))
        fps = []
        for h in handles:
            pr = h["_parts_const"] if h.get("_eval_only") else h["shift_parts"](np.asarray(h["x0"], float))
            if pr is None:
                raise ValueError(f"{h['label']}: the curve cannot be computed at the starting point")
            fps.append([float(d["delta_free"]) for d in pr["designs"]])
        try:
            groups, _how = match_shift_columns(fps)
        except ValueError as ex:
            raise ValueError("Shared pure-species shifts: " + str(ex))
        if not any(sum(j is not None for j in grp) >= 2 for grp in groups):
            raise ValueError("Shared pure-species shifts: no proton is common to two tabs (the free shifts "
                             f"differ by more than {SHIFT_MATCH_TOL:g} ppm everywhere)")

    # 4. the global parameter vector: shared constants first, then local parameters
    gnames = [k for k in ticked_any if any(k in h["keys"] for h in handles)]
    x0 = [shared_start.get(k, 0.0) for k in gnames]
    maps = []                         # per experiment: indices into the global vector
    for h in handles:
        idx = []
        for j, n in enumerate(h["names"]):
            if j < h["n_log_params"]:
                idx.append(gnames.index(n))
            else:                     # local concentration / titrant
                gn = f"{n}@{h['label']}"
                gnames.append(gn); x0.append(float(h["x0"][j])); idx.append(len(gnames) - 1)
        maps.append(np.asarray(idx, dtype=int))
    x0 = np.asarray(x0, dtype=float)
    n_shared = sum(1 for n in gnames if "@" not in n)

    # 4b. one weight per experiment.  Every fit function minimises a dimensionless
    #     objective (its residuals divided by the variance of that tab's own data),
    #     so in a plain sum a tab the model fits badly weighs in proportion to its
    #     misfit and can drag a shared constant away from what a clean tab says.
    #     With ``weighted`` each tab is rescaled to its own fit, w_i = n_i / obj_i,
    #     re-estimated after each optimiser pass (the first pass is a plain sum),
    #     so at the end every tab contributes about n_i (reduced chi-squared of one).
    weights = [1.0] * len(handles)
    chi2_pp = {}                       # per tab: objective per point at the last reference
    if fixed_weights:
        weights = [float(fixed_weights.get(e["label"], 1.0)) for e in experiments]
        weighted = False               # no re-estimation: the weights belong to the fit being examined

    def per_tab(x, which="objective"):
        """Every tab's objective (or data-only SSR) at the global point ``x``."""
        if not share_shifts:
            out = []
            for h, idx in zip(handles, maps):
                f = h.get(which) or h.get("objective")
                out.append(float(h.get("f_const") or 0.0) if f is None else float(f(x[idx])))
            return out
        parts = [h["_parts_const"] if h.get("_eval_only") else h["shift_parts"](x[idx])
                 for h, idx in zip(handles, maps)]
        if any(p is None for p in parts):
            return [1e12] * len(handles)
        ssr_shift, _ = _shared_shift_solve(parts, weights)
        out = []
        for h, p, s in zip(handles, parts, ssr_shift):
            v = float(p["integ_ssr"]) + float(s) / max(float(p["shift_var"]), 1e-20)
            # evaluation-only tabs score data alone (as f_const does without sharing):
            # the sensitivity test must not see the constraint penalty of a perturbation
            if which == "objective" and not h.get("_eval_only"):
                v += float(p["penalty"](v))
            out.append(v)
        return out

    def reweight(x):
        """Rescale every tab to its objective at ``x``; True if a weight moved by > 5 %."""
        moved = False
        try:
            vals = per_tab(x, "data_ssr")
        except Exception:
            return False
        for j, (h, e) in enumerate(zip(handles, experiments)):
            if h.get("_eval_only"):
                continue
            v = vals[j]
            n = max(int(h.get("n_data") or 1), 1)
            if np.isfinite(v) and v > 0:
                w = n / v
                moved = moved or abs(w - weights[j]) > 0.05 * max(w, weights[j])
                weights[j] = w
                chi2_pp[e["label"]] = v / n
        return moved

    # the first pass runs as a plain sum: weights taken at the starting point would
    # measure how far each tab starts from its optimum, not how well it can be fitted

    def total(x, which="objective"):
        return float(sum(w * v for w, v in zip(weights, per_tab(x, which))))

    # 5. optimise (Nelder-Mead, adaptive, with the fit's tolerance semantics)
    trace = FitTrace()
    best = {"x": x0.copy(), "f": np.inf, "n": 0, "n_improve": 0}
    # Nelder-Mead certifies convergence by the SIZE of the simplex; along a flat
    # direction (two constants whose product alone is determined) the simplex
    # never shrinks although the objective stopped changing long ago, and the
    # fit runs to the Timeout with constants that no longer move.  That is a
    # converged fit: when the best objective improved by less than tolerance/100
    # (relative; 1e-5 for Tol = 3) over the last ``stall_evals`` evaluations,
    # the pass is declared stationary.
    stall = {"evals": max(100, 20 * (x0.size + 1))}
    stall_rel = float(tolerance) * 1e-2
    f_hist = []                                   # best objective after every evaluation

    class _Timeout(Exception):
        pass

    class _Stationary(Exception):
        pass

    _n_total_hint = max(int(sum(int(h.get("n_data") or 0) for h in handles)), 1)

    def obj(x):
        trace(x)
        best["n"] += 1
        vals = per_tab(x)
        f = float(sum(w * v for w, v in zip(weights, vals)))
        if f < best["f"]:
            best["f"], best["x"] = f, np.array(x, dtype=float)
            # the live view shows the PLAIN sum of the tabs' objectives per point: the
            # weighted objective changes scale at every reweighting (it equals the
            # number of points right after one, so its root per point jumped to 1.0)
            best["plain"] = float(np.sqrt(max(float(sum(vals)), 0.0) / _n_total_hint))
            # what the live view shows: the root-mean-square objective of every tab
            # at the current best point (a weighted fit trades one tab's misfit for
            # another's, so no single number is monotone; the pair is what matters)
            best["detail"] = "rms objective per tab: " + ", ".join(
                f"{h['label']} {np.sqrt(max(float(v), 0.0) / max(int(h.get('n_data') or 1), 1)):.3g}"
                for h, v in zip(handles, vals))
        if np.isfinite(best["f"]) and best.get("plain") is not None:
            _live.notify_global({n: float(v) for n, v in zip(gnames, best["x"])}, best["plain"],
                                best.get("detail"))
        f_hist.append(best["f"])
        if len(f_hist) - best["n_improve"] > stall["evals"]:
            f_then = f_hist[-stall["evals"] - 1]
            if np.isfinite(f_then) and f_then - best["f"] <= stall_rel * max(abs(best["f"]), 1e-30):
                raise _Stationary()
        if progress is not None and best["n"] % 10 == 0:
            try:
                progress(min((time.time() - t0) / max(timeout_s, 1.0), 0.99),
                         f"Global fit: {best['n']} evaluations, objective {best['f']:.6g}")
            except Exception:
                pass
        if time.time() - t0 > timeout_s:
            raise _Timeout()
        return f

    timed_out = False
    tied_names = []                       # constants tied exactly by $constraints (4a below)
    if x0.size == 0:                      # evaluation only
        x_best, success, res = x0, True, None
        summary = {"n_evals": 0, "stop_reason": "evaluation", "final_spread": None, "elapsed_s": 0.0}
    else:
        steps = np.empty_like(x0)
        for i, n in enumerate(gnames):
            steps[i] = simplex_step if "@" not in n else max(abs(x0[i]) * 0.1, 1e-3)
        # 4a. linear equality $constraints (log space, '=') between fitted constants
        #     are solved exactly: the simplex moves within the subspace that
        #     satisfies them, x = x_part + N z.  Left to the penalty alone, a vertex
        #     that moves ONE tied constant costs ~1e5 x the misfit, so tied constants
        #     never moved (K_Cage_Cl = K_Cage_Cl2 = ... stayed at their start).
        tie_rows, tie_rhs, seen_ties = [], [], set()
        for e in experiments:
            for c in (e["fa"].get("constraints") or []):
                if c.get("mode", "log") != "log" or c.get("op", "=") != "=":
                    continue
                coeffs = {k: float(v) for k, v in (c.get("coeffs") or {}).items() if float(v) != 0.0}
                key = (tuple(sorted(coeffs.items())), float(c.get("const") or 0.0))
                if not coeffs or key in seen_ties:
                    continue
                seen_ties.add(key)
                row, rhs = np.zeros(x0.size), -float(c.get("const") or 0.0)
                for k, v in coeffs.items():
                    if k in gnames:
                        row[gnames.index(k)] = v
                    else:                        # not fitted: a fixed number in the tie
                        val = next((float(h["logK0"][k]) for h in handles
                                    if k in (h.get("logK0") or {})), None)
                        if val is None:
                            row = None
                            break
                        rhs -= v * val
                if row is not None and np.any(row != 0.0):
                    tie_rows.append(row); tie_rhs.append(rhs)
        expand, z0, steps_z = (lambda z: np.asarray(z, dtype=float)), x0.copy(), steps.copy()
        if tie_rows:
            A, b = np.vstack(tie_rows), np.asarray(tie_rhs, dtype=float)
            keep = []
            for i in range(A.shape[0]):
                if np.linalg.matrix_rank(A[keep + [i]]) == len(keep) + 1:
                    keep.append(i)
            A, b = A[keep], b[keep]
            tied_cols = [j for j in range(x0.size) if np.any(A[:, j] != 0.0)]
            free_cols = [j for j in range(x0.size) if j not in tied_cols]
            A_t = A[:, tied_cols]
            _, s_, Vt = np.linalg.svd(A_t)
            rank = int(np.sum(s_ > 1e-10 * max(float(s_.max()), 1.0)))
            N_t = Vt[rank:].T                                  # null-space basis
            x_part = x0.copy()
            x_part[tied_cols] = x0[tied_cols] - np.linalg.pinv(A_t) @ (A_t @ x0[tied_cols] - b)
            tied_names = [gnames[j] for j in tied_cols]
            def expand(z, _xp=x_part, _N=N_t, _f=free_cols, _t=tied_cols):
                z = np.asarray(z, dtype=float)
                x = _xp.copy()
                x[_f] = z[:len(_f)]
                x[_t] = _xp[_t] + _N @ z[len(_f):]
                return x
            z0 = np.concatenate([x0[free_cols], np.zeros(N_t.shape[1])])
            steps_z = np.concatenate([steps[free_cols], np.full(N_t.shape[1], float(np.max(steps[tied_cols])))])
            if not np.allclose(x_part, x0):
                best["x"] = x_part.copy()          # the start moved onto the constraint surface
        best["z"] = z0.copy()
        def obj_z(z):
            x = expand(z)
            try:
                return obj(x)
            finally:
                if np.array_equal(best["x"], x):
                    best["z"] = np.asarray(z, dtype=float).copy()
        # Nelder-Mead with restarts.  In a flat valley (two constants whose
        # product alone is determined) the simplex collapses along the valley
        # and "converges" while a slope remains in another direction -- the
        # VPIPV/DAB-512 case, where KGH3M2 stalled at its bound although a
        # point 12 % lower in the objective existed.  A fresh full-size
        # simplex around the best point finds that slope; we restart until a
        # restart no longer improves the objective (or time runs out).
        n_runs, n_reweights, f_prev, res, stationary = 0, 0, np.inf, None, False
        f_pass_start = np.inf
        try:
            with simplex_step_override(simplex_step):
                while True:
                    zs = best["z"] if n_runs else z0
                    simplex = np.vstack([zs] + [zs + np.eye(z0.size)[i] * steps_z[i] for i in range(z0.size)])
                    best["n_improve"] = len(f_hist)        # a fresh simplex gets a fresh stall window
                    f_pass_start = best["f"]
                    # while the weights are still being settled, passes are short: the
                    # weights are re-estimated after each and the next pass restarts
                    bounded = bool(weighted and n_reweights < 3)
                    pass_iter = min(maxiter, 8 * (z0.size + 1)) if bounded else maxiter
                    try:
                        res = minimize(obj_z, zs, method="Nelder-Mead",
                                       options={"maxiter": pass_iter, "xatol": tolerance,
                                                "fatol": tolerance * 1e-4, "adaptive": True,
                                                "initial_simplex": simplex})
                        pass_ok = bool(res.success) or bounded
                    except _Stationary:
                        res, pass_ok, stationary = None, True, True
                        stall["evals"] = max(60, stall["evals"] // 2)   # the checking restart needs less
                    n_runs += 1
                    if bounded and reweight(best["x"]):
                        # weights moved: the objective is a new function; one more
                        # pass at least, compared with itself from here on
                        n_reweights += 1
                        best["f"] = total(best["x"])
                        f_prev = np.inf
                        continue
                    if bounded:                            # weights settled (moved < 5 %): full passes from here
                        n_reweights = 3
                        f_prev = np.inf
                        continue
                    gain = f_prev - best["f"]
                    f_prev = best["f"]
                    if (not pass_ok or n_runs >= 6 or not np.isfinite(best["f"])
                            or gain <= max(abs(best["f"]), 1e-30) * tolerance * 1e-3):
                        break
            x_best = (expand(np.asarray(res.x, dtype=float)) if (res is not None and res.fun <= best["f"])
                      else best["x"])
            success = bool(pass_ok)
        except _Timeout:
            x_best, res = best["x"], None
            # the clock ran out during a restart (or the reweighted pass) that had
            # gained nothing a Tol could see: the fit is converged, not timed out
            if n_runs >= 1 and np.isfinite(f_pass_start) and np.isfinite(best["f"]) \
                    and f_pass_start - best["f"] <= 10.0 * stall_rel * max(abs(best["f"]), 1e-30):
                success, timed_out, stationary = True, False, True
            else:
                success, timed_out = False, True
        summary = trace.summary(res, timed_out, tolerance)
        summary["n_restarts"] = max(n_runs - 1, 0)
        summary["stationary"] = bool(stationary and not timed_out)
        if summary["stationary"]:
            summary["stop_reason"] = "stationary"
    summary["tied"] = list(tied_names)
    values = {n: float(x_best[i]) for i, n in enumerate(gnames)}
    objective_best = total(x_best)
    _n_total = int(sum(int(h.get("n_data") or 0) for h in handles))
    chi_rmse = float(np.sqrt(objective_best / max(_n_total, 1))) if np.isfinite(objective_best) else None
    _wmax = max(weights) if weights else 1.0
    _rel_weights = {e["label"]: (w / _wmax if _wmax else 1.0) for e, w in zip(experiments, weights)}
    if not with_stats:
        return {"success": success, "timed_out": timed_out, "message": "light",
                "values": values, "errors": {}, "notes": {}, "shared": gnames[:n_shared],
                "names": gnames, "stats": {}, "rmse": None, "rmse_by_label": {},
                "summary": summary, "objective": objective_best, "skipped": skipped,
                "weights": _rel_weights, "abs_weights": {e["label"]: float(w) for e, w in zip(experiments, weights)},
            "chi2_per_point": dict(chi2_pp), "weighted": bool(weighted), "chi_rmse": chi_rmse,
            "share_shifts": bool(share_shifts), "elapsed_s": time.time() - t0}

    # 6. every experiment evaluated at the optimum -> its usual stats dict.  With
    #    shared shifts every species of every shift column is pinned to the common
    #    solution, so the per-tab stats (pure shifts, curves, residuals) describe it.
    shared_dd_by_tab, shared_pure = [None] * len(handles), []
    if share_shifts:
        try:
            parts = [h["_parts_const"] if h.get("_eval_only") else h["shift_parts"](x_best[idx])
                     for h, idx in zip(handles, maps)]
            _, shared = _shared_shift_solve(parts, weights)
            shared_dd_by_tab = []
            for t, p in enumerate(parts):
                pins = {}
                for sh in shared:
                    col = sh["cols"][t]
                    if col is None:
                        continue                      # this tab has no such proton
                    own = {sp for d in p["designs"] if d["col"] == col for sp in d["species"]}
                    pins[col] = {sp: float(sh["dd0"][i] - sh["gauge"][t])
                                 for i, sp in enumerate(sh["species"]) if sp in own}
                shared_dd_by_tab.append(pins)
            shared_pure = [{"cols": sh["cols"], "how": sh["how"], "free_shift": sh["delta_free0"],
                            "shifts_abs": {sp: float(sh["dd0"][i] + sh["delta_free0"])
                                           for i, sp in enumerate(sh["species"])}}
                           for sh in shared]
        except Exception:
            shared_dd_by_tab = [None] * len(handles)
    stats_out, rmses, n_pts = {}, {}, {}
    for h, idx, e, _sdd in zip(handles, maps, experiments, shared_dd_by_tab):
        logK = dict(h["logK0"])
        for k in h["keys"]:
            if k in values:
                logK[k] = values[k]
        fa_once = dict(e["fa"])
        # local concentrations / titrants: start the one-shot at the fitted value
        # kinetics records may carry no params (their fitted concentrations are
        # patched into parsed_fit by the kinetics fit functions themselves)
        par = dict(fa_once.get("params") or {}); par["conc0"] = dict(par.get("conc0", {}))
        par["titrant_mMs"] = dict(par.get("titrant_mMs", {}))
        for j, n in enumerate(h["names"]):
            if j >= h["n_log_params"]:
                v = float(x_best[idx[j]])
                if n in (fa_once.get("fit_conc_keys") or []):
                    par["conc0"][n] = v
                else:
                    tfree = n[:-1] if (n.endswith("t") or n.endswith("0")) else n
                    par["titrant_mMs"][tfree] = v
        fa_once["params"] = par
        if _sdd:
            fa_once["shared_dd"] = _sdd
        try:
            with simplex_step_override(1e-9):
                _ok, _lk, st_, _msg = _call_fit(fa_once, logK, h["keys"], tolerance=1e30, maxiter=1,
                                                timeout_s=1e9, compute_hessian=False,
                                                objective_only=False)
        except Exception as ex:
            st_ = {"error": f"{type(ex).__name__}: {ex}"}
        if not isinstance(st_, dict):
            st_ = {}
        st_["param_values"] = {k: values.get(k, logK.get(k)) for k in h["keys"]} if not h.get("_eval_only") else {}
        st_["global_fit"] = {"labels": [x["label"] for x in experiments], "shared": gnames[:n_shared],
                             "share_shifts": bool(share_shifts)}
        st_.update({k: v for k, v in summary.items() if k in ("n_evals", "stop_reason", "final_spread", "elapsed_s")})
        st_["timed_out"] = timed_out
        stats_out[e["label"]] = st_
        rmses[e["label"]] = st_.get("rmse")
        n_pts[e["label"]] = st_.get("n_points")

    # 7. errors from the Hessian of the summed data-only SSR (n = real point counts)
    errors, cov, notes, note_by_name = {n: None for n in gnames}, None, {}, {}
    if not with_errors:
        gnames_h = []                     # skip the Hessian below
    else:
        gnames_h = gnames
    from equilibrist_fit_nmr import _hessian_errors, _linear_eq_matrix
    n_data = int(sum(int(n or 0) for n in n_pts.values())) or int(sum(int(h.get("n_data") or 0) for h in handles))
    has_data_ssr = all(h.get("data_ssr") is not None for h in handles)
    which = "data_ssr" if has_data_ssr else "objective"
    ssr_best = total(x_best, which)
    if not gnames_h:
        pass
    else:
      try:
          cstr_all = []
          for e in experiments:
              cstr_all.extend(e["fa"].get("constraints") or [])
          err_idx, cov = _hessian_errors(lambda v: total(v, which), x_best, ssr_best,
                                         max(n_data, len(gnames) + 1), len(gnames),
                                         eq_matrix=_linear_eq_matrix(cstr_all, gnames),
                                         n_log_params=n_shared, notes=notes)
          errors = {gnames[i]: (float(v) if v is not None else None) for i, v in (err_idx or {}).items()}
          for i, n in enumerate(gnames):
              errors.setdefault(n, None)
      except Exception:
          errors = {n: None for n in gnames}
      note_by_name = {gnames[i]: r for i, r in (notes or {}).items() if i < len(gnames)}

    for h in handles:
        st_ = stats_out[h["label"]]
        st_["param_errors"] = {k: errors.get(k) for k in h["keys"]}
        st_["param_error_notes"] = {k: note_by_name[k] for k in h["keys"] if k in note_by_name}
        if cov is not None:
            sel = [gnames.index(k) for k in h["keys"]]
            st_["param_cov"] = np.asarray(cov)[np.ix_(sel, sel)].tolist()
            st_["param_cov_names"] = list(h["keys"])

    # 7b. how much each tab contributes to pinning every shared constant: the
    # diagonal curvature of its WEIGHTED objective at the solution (the global
    # covariance is the inverse of the sum of these), as a share of the total.
    # A tab whose script does not declare the constant contributes nothing.
    info_share = {}
    try:
        _xb = np.asarray(x_best, dtype=float)
        _hfd = 1e-3
        _base = np.asarray(per_tab(_xb, which), dtype=float)
        for k in range(n_shared):
            _xp, _xm = _xb.copy(), _xb.copy()
            _xp[k] += _hfd
            _xm[k] -= _hfd
            _fp = np.asarray(per_tab(_xp, which), dtype=float)
            _fm = np.asarray(per_tab(_xm, which), dtype=float)
            _curv = np.clip(np.asarray(weights, dtype=float) * (_fp + _fm - 2.0 * _base) / _hfd ** 2, 0.0, None)
            _tot = float(np.sum(_curv))
            for j, e in enumerate(experiments):
                info_share.setdefault(e["label"], {})[gnames[k]] = (float(_curv[j] / _tot) if _tot > 0 else None)
    except Exception:
        info_share = {}

    rmse_global = None
    try:
        num = sum(float(r) ** 2 * float(n) for r, n in zip(rmses.values(), n_pts.values())
                  if r is not None and n)
        den = sum(float(n) for n in n_pts.values() if n)
        rmse_global = float(np.sqrt(num / den)) if den else None
    except Exception:
        pass
    for _lab_s, st_ in stats_out.items():
        st_["global_fit"]["rmse"] = rmse_global
        st_["global_fit"]["rmse_by_label"] = dict(rmses)
        st_["global_fit"]["values"] = {n: values[n] for n in gnames}
        st_["global_fit"]["errors"] = dict(errors)
        st_["global_fit"]["info_share"] = info_share
        st_["global_fit"]["n_points_total"] = int(sum(int(n or 0) for n in n_pts.values()))
        st_["global_fit"]["own_share"] = dict(info_share.get(_lab_s) or {})

    msg = ("converged" if success else ("timed out" if timed_out else "stopped"))
    if success and summary.get("stationary"):
        # the stationarity stop also ends well-determined fits (the objective simply
        # stopped improving); claim a flat direction only when the errors say so
        _undet = [n for n in gnames[:n_shared]
                  if with_errors and (errors.get(n) is None or not np.isfinite(errors.get(n)) or errors.get(n) > 1.0)]
        if _undet:
            msg = ("converged (objective stationary: some constants lie along a flat direction "
                   "and their individual values are not determined: " + ", ".join(_undet) + ")")
        else:
            msg = "converged (objective stationary)"
    if tied_names:
        msg += f"; {len(tied_names)} constants tied exactly by $constraints ({', '.join(tied_names)})"
    if skipped:
        msg += "; left out: " + "; ".join(skipped)
    return {"success": success, "timed_out": timed_out, "message": msg,
            "values": values, "errors": errors, "notes": note_by_name,
            "shared": gnames[:n_shared], "names": gnames, "stats": stats_out, "skipped": skipped,
            "objective": objective_best,
            "rmse": rmse_global, "rmse_by_label": rmses, "summary": summary,
            "weights": _rel_weights, "abs_weights": {e["label"]: float(w) for e, w in zip(experiments, weights)},
            "chi2_per_point": dict(chi2_pp), "weighted": bool(weighted), "chi_rmse": chi_rmse,
            "info_share": info_share,
            "share_shifts": bool(share_shifts), "shared_pure_shifts": shared_pure,
            "elapsed_s": time.time() - t0}


# ════════════════════════════════════════════════════════════════════════
# A "global" fit-args dict: what the diagnostics (profiles, trace, bootstrap,
# jackknife, Monte Carlo, sensitivity) receive after a global fit.
# ════════════════════════════════════════════════════════════════════════
def single_fit(fa: dict, *, tolerance: float, maxiter: int = 100000, timeout_s: float = 300.0) -> dict:
    """One tab's own fit from its fit-args record (what the Fit button does),
    callable on the cluster.  Returns ``{"ok", "fitted", "stats", "message"}``."""
    logK = dict(fa.get("start_logK") or fa.get("start_logk") or {})
    keys = list(fa.get("fit_keys") or [])
    out = _call_fit(fa, logK, keys, tolerance=float(tolerance), maxiter=int(maxiter),
                    timeout_s=float(timeout_s), compute_hessian=True, objective_only=False)
    if not (isinstance(out, tuple) and len(out) == 4):
        raise RuntimeError("the fit function did not return (ok, fitted, stats, message)")
    ok, fitted, stats, msg = out
    return {"ok": bool(ok), "fitted": {k: float(v) for k, v in (fitted or {}).items()},
            "stats": stats if isinstance(stats, dict) else {}, "message": str(msg)}


def make_global_fit_args(active_fa: dict, experiments: list, result: dict, *,
                         tolerance: float, maxiter: int, timeout_s: float, label=None) -> dict:
    """``active_fa`` is the active tab's own fit args (kept so UI code that
    reads parsed_fit / params / x_expr keeps working and refers to that tab);
    ``experiments`` is the list given to :func:`global_fit`; ``result`` its
    return value."""
    fa = dict(active_fa)
    fa.pop("has_data", None)
    fa["kind"] = "global"
    fa["experiments"] = [{"label": e["label"], "fa": e["fa"],
                          "start_logK": dict(e.get("start_logK") or {})} for e in experiments]
    fa["weighted"] = bool(result.get("weighted", True))
    fa["share_shifts"] = bool(result.get("share_shifts", False))
    fa["tab_weights"] = dict(result.get("abs_weights") or {})   # frozen for every diagnostic
    fa["own_label"] = label                  # which experiment is this tab (see tab_fit_args)
    fa["labels"] = [e["label"] for e in experiments]
    fa["fit_keys"] = list(result.get("shared") or [])
    fa["fit_conc_keys"], fa["fit_titrant_keys"] = [], []
    start = dict(active_fa.get("start_logK") or {})
    for k in fa["fit_keys"]:
        start[k] = float(result["values"][k])
    fa["start_logK"], fa["start_logk"] = start, dict(start)
    fa["constraints"] = [c for e in experiments for c in (e["fa"].get("constraints") or [])]
    fa["tolerance"], fa["maxiter"], fa["timeout_s"] = float(tolerance), int(maxiter), float(timeout_s)
    fa["simplex_step"] = 0.15
    fa["baseline_stats"] = dict(result.get("stats") or {})
    fa["global_values"] = dict(result.get("values") or {})
    return fa


def rate_constant_info(name: str, parsed_list) -> Optional[tuple]:
    """``(n_reactants, is_reverse, n_products)`` when ``name`` is a rate constant of
    one of the parsed scripts (a ``>`` or ``<>`` reaction), else None."""
    for p in parsed_list or []:
        for r in (p or {}).get("kinetics") or []:
            n_r = sum(float(c) for c, _ in (r.get("reactants") or []))
            n_p = sum(float(c) for c, _ in (r.get("products") or []))
            if r.get("kname") == name:
                return (int(n_r), False, int(n_p))
            if r.get("krname") == name:
                return (int(n_r), True, int(n_p))
    return None


def format_shared_constant(name: str, log_value: float, parsed_list) -> str:
    """How a fitted constant reads in the global-fit message: equilibrium constants
    as log K (the app's convention), rate constants LINEAR with their units (the
    scripts and the kinetics tables never show a log k)."""
    info = rate_constant_info(name, parsed_list)
    if info is None:
        return f"{name} = {float(log_value):.2f}"
    from equilibrist_kinetics import _rate_constant_units
    n_r, rev, n_p = info
    units = (_rate_constant_units(n_p, is_reverse=True, n_products=n_p) if rev
             else _rate_constant_units(n_r))
    return f"{name} = {10.0 ** float(log_value):.3g} {units}".rstrip()


def tab_fit_args(fa: dict) -> dict:
    """This tab's own single-experiment fit args at the global solution -- what the
    diagnostics use when the user asks for "this tab only" after a global fit:
    the tab's data, its own fit ticks, the global constants as starting values."""
    label = fa.get("own_label")
    own = next((e for e in (fa.get("experiments") or []) if e.get("label") == label), None)
    if own is None:
        return fa
    efa = dict(own["fa"])
    have = set(experiment_constants(efa))
    start = dict(efa.get("start_logK") or {})
    for k, v in (fa.get("global_values") or {}).items():
        if k in have:
            start[k] = float(v)
    efa["start_logK"], efa["start_logk"] = start, dict(start)
    for k in ("tolerance", "maxiter", "timeout_s", "simplex_step"):
        if k in fa:
            efa[k] = fa[k]
    efa["from_global"] = label
    return efa


def _experiments_for(fa: dict, pins: Optional[dict] = None, *, overrides: Optional[dict] = None) -> list:
    """Per-experiment inputs for :func:`global_fit` from a global fit-args dict.
    ``pins`` {name: value} are held fixed everywhere they occur; ``overrides``
    {label: {"data": ..., "params": ...}} swap in synthetic data / params."""
    pins = dict(pins or {})
    overrides = overrides or {}
    shared = [k for k in (fa.get("fit_keys") or []) if k not in pins]
    gvals = dict(fa.get("start_logK") or {})
    out = []
    for e in fa["experiments"]:
        efa = dict(e["fa"])
        have = set(experiment_constants(efa))
        start = dict(e.get("start_logK") or efa.get("start_logK") or {})
        for k, v in gvals.items():
            if k in have:
                start[k] = float(v)
        for k, v in pins.items():
            if "@" in k:                               # a local parameter: name@label
                name, lab = k.split("@", 1)
                if lab == e["label"]:
                    par = dict(efa.get("params") or {}); par["conc0"] = dict(par.get("conc0") or {})
                    par["titrant_mMs"] = dict(par.get("titrant_mMs") or {})
                    if name in (efa.get("fit_conc_keys") or []):
                        par["conc0"][name] = 10.0 ** float(v)
                        efa["fit_conc_keys"] = [c for c in efa["fit_conc_keys"] if c != name]
                    elif name in (efa.get("fit_titrant_keys") or []):
                        tfree = name[:-1] if (name.endswith("t") or name.endswith("0")) else name
                        par["titrant_mMs"][tfree] = 10.0 ** float(v)
                        efa["fit_titrant_keys"] = [c for c in efa["fit_titrant_keys"] if c != name]
                    efa["params"] = par
            elif k in have:
                start[k] = float(v)
        ov = overrides.get(e["label"]) or {}
        if "data" in ov:
            k = _data_key(efa)
            if k:
                efa[k] = ov["data"]
        if "excl" in ov:
            efa["excl_set"] = set(ov["excl"])
        if "params" in ov:
            efa["params"] = ov["params"]
        out.append({"label": e["label"], "fa": efa, "start_logK": start,
                    "ticked": [k for k in shared if k in have]})
    return out


def _run(fa: dict, exps: list, *, with_stats: bool, with_errors: bool, timeout_s=None):
    return global_fit(exps, tolerance=float(fa.get("tolerance", 1e-3)),
                      maxiter=int(fa.get("maxiter", 100000)),
                      timeout_s=float(timeout_s if timeout_s is not None else fa.get("timeout_s", 300.0)),
                      simplex_step=float(fa.get("simplex_step", 0.15)),
                      with_stats=with_stats, with_errors=with_errors,
                      weighted=bool(fa.get("weighted", True)), fixed_weights=fa.get("tab_weights") or None,
                      share_shifts=bool(fa.get("share_shifts", False)))


def _as_stats(res: dict, fa: dict) -> dict:
    """A stats-like dict (what ``_refit_with_pinned`` returns) from a global result."""
    # the scalar the profiles / maps / trace compare: the weighted objective (as a
    # root-mean-square per point) when the fit was weighted, so their minimum is the fit's
    st_ = {"rmse": (res.get("chi_rmse") if res.get("chi_rmse") else res.get("rmse")), "rmse_phys": res.get("rmse"),
           "param_values": {k: v for k, v in res["values"].items() if "@" not in k},
           "fitted_concs": {}, "fitted_titrants": {},
           "n_points": int(sum(int(s.get("n_points") or 0) for s in (res.get("stats") or {}).values())),
           "stop_reason": res["summary"].get("stop_reason"), "n_evals": res["summary"].get("n_evals"),
           "timed_out": res.get("timed_out"), "objective": res.get("objective"),
           "rmse_by_label": res.get("rmse_by_label"), "global_fit": {"labels": fa.get("labels")}}
    for k, v in res["values"].items():
        if "@" in k:
            st_["fitted_concs"][k] = 10.0 ** float(v)
    return st_


def refit_global(fa: dict, pins: Optional[dict] = None, *, with_errors: bool = False):
    """Refit all experiments with ``pins`` held fixed (profiles / maps / trace)."""
    try:
        res = _run(fa, _experiments_for(fa, pins), with_stats=True, with_errors=with_errors)
    except Exception:
        return None
    return _as_stats(res, fa)


def evaluate_global(fa: dict, values: dict) -> Optional[float]:
    """Global RMSE with EVERY shared constant pinned at ``values`` (sensitivity test)."""
    pins = dict(fa.get("start_logK") or {})
    pins.update({k: float(v) for k, v in values.items()})
    pins = {k: v for k, v in pins.items() if k in (fa.get("fit_keys") or []) or k in values}
    try:
        exps = _experiments_for(fa, pins)
        for e in exps:                      # nothing varies: evaluation only
            e["ticked"] = []
        res = global_fit(exps, tolerance=1e30, maxiter=1, timeout_s=1e9, with_stats=True, with_errors=False,
                         weighted=bool(fa.get("weighted", True)), fixed_weights=fa.get("tab_weights") or None,
                         share_shifts=bool(fa.get("share_shifts", False)))
    except Exception:
        return None
    r = res.get("chi_rmse") or res.get("rmse")
    return float(r) if r is not None and np.isfinite(r) else None


# ════════════════════════════════════════════════════════════════════════
# Resampling tools on the global fit
# ════════════════════════════════════════════════════════════════════════
def _mode_of(efa: dict) -> str:
    """'shift' | 'integration' | 'mixed' | 'conc' | 'spectra' | 'itc', with a
    'kin_' prefix for kinetics scripts."""
    return _fit_function(efa)[0]


def _data_key(efa: dict):
    if efa.get("use_itc_fit"):
        return "itc_q_exp"
    for key in ("nmr_data_fit", "spectra_data_fit", "exp_data_fit"):
        if efa.get(key) is not None:
            return key
    return None


def _data_of(efa: dict):
    k = _data_key(efa)
    return efa[k] if k else {}


def _synth_experiment(efa: dict, stats: dict, rng, method: str):
    """One synthetic data set for an experiment, by residual resampling
    (same recipes as equilibrist_bootstrap, per mode)."""
    from equilibrist_bootstrap import _resample_residuals, _inject_synth_y, _extract_per_col
    mode, data = _mode_of(efa), _data_of(efa)
    if mode.startswith("kin_"):
        mode = mode[4:]
    if mode == "itc":
        q_exp = np.asarray(data, dtype=float)
        q_th = stats.get("q_theo")
        if q_th is None or len(q_th) != len(q_exp):
            raise ValueError("no calculated heats in the stats of this experiment")
        resid = q_exp - np.asarray(q_th, dtype=float)
        return np.asarray(q_th, dtype=float) + _resample_residuals(resid, method, rng)
    if mode == "mixed":
        per_integ = _extract_per_col(stats, "per_col_y_obs_integ", "per_col_y_calc_integ")
        per_shift = _extract_per_col(stats, "per_col_y_obs_shift", "per_col_y_calc_shift")
        if not per_integ and not per_shift:
            per_shift = _extract_per_col(stats)          # kinetics mixed stores plain per-column arrays
        if not per_integ and not per_shift:
            raise ValueError("no residuals in the stats of this experiment")
        synth = {}
        for col, cd in data.items():
            if str(col).startswith("_"):
                synth[col] = cd
            elif col in per_integ:
                resid, _yo, y_calc = per_integ[col]
                synth[col] = _inject_synth_y(cd, y_calc + _resample_residuals(resid, method, rng))
            elif col in per_shift:
                resid, _yo, y_calc = per_shift[col]
                df = np.asarray((stats.get("delta_free") or {}).get(col, 0.0))
                synth[col] = _inject_synth_y(cd, (y_calc + _resample_residuals(resid, method, rng)) + df)
            else:
                synth[col] = cd
        return synth
    if mode == "integration":
        per_col = _extract_per_col(stats)
        if not per_col:
            raise ValueError("no residuals in the stats of this experiment")
        col_to_sp = stats.get("col_to_sp") or {}
        synth = {col: cd for col, cd in data.items()}
        for sp, (resid, _yobs, y_calc) in per_col.items():
            r_synth = _resample_residuals(resid, method, rng)
            ratio = np.where(np.abs(y_calc) > 1e-20, (y_calc + r_synth) / y_calc, 1.0)
            for col, cd in data.items():
                if str(col).startswith("_") or col_to_sp.get(col, col) != sp:
                    continue
                y = np.asarray(cd["y"], dtype=float)
                v = np.asarray(cd.get("v_add_mL", cd.get("x", [])), dtype=float)
                fin = np.where(np.isfinite(v) & np.isfinite(y))[0]
                k = min(len(fin), len(ratio))
                y2 = y.copy(); y2[fin[:k]] = y[fin[:k]] * ratio[:k]
                synth[col] = {**cd, "y": y2}
        return synth
    if mode in ("shift", "conc"):
        per_col = _extract_per_col(stats)
        if not per_col:
            raise ValueError("no residuals in the stats of this experiment")
        synth = {}
        for col, cd in data.items():
            if str(col).startswith("_") or col not in per_col:
                synth[col] = cd
                continue
            resid, _yo, y_calc = per_col[col]
            synth[col] = _inject_synth_y(cd, y_calc + _resample_residuals(resid, method, rng))
        return synth
    if mode == "spectra":
        A_obs, A_calc = stats.get("A_obs"), stats.get("A_calc")
        if A_obs is None or A_calc is None:
            raise ValueError("no residual matrix in the stats of this experiment")
        A_obs, A_calc = np.asarray(A_obs, float), np.asarray(A_calc, float)
        r = _resample_residuals((A_obs - A_calc).ravel(), method, rng).reshape(A_obs.shape)
        synth = dict(data)
        for k_try in ("A_obs", "A", "matrix", "abs_matrix"):
            if k_try in synth and isinstance(synth[k_try], np.ndarray):
                synth[k_try] = A_calc + r
                break
        return synth
    raise ValueError(f"unsupported mode {mode!r}")


def _baseline(fa: dict) -> dict:
    """Per-experiment stats at the global optimum (from the fit, or recomputed)."""
    bs = fa.get("baseline_stats") or {}
    if bs and all(e["label"] in bs for e in fa["experiments"]):
        return bs
    res = _run(fa, _experiments_for(fa, {}), with_stats=True, with_errors=False)
    return res["stats"]


def _names_of(fa: dict) -> list:
    names = list(fa.get("fit_keys") or [])
    for e in fa["experiments"]:
        for c in list(e["fa"].get("fit_conc_keys") or []) + list(e["fa"].get("fit_titrant_keys") or []):
            names.append(f"{c}@{e['label']}")
    return names


def _light_values(fa: dict, exps: list) -> Optional[dict]:
    res = _run(fa, exps, with_stats=False, with_errors=False)
    if res.get("timed_out") and res["summary"].get("n_evals", 0) < 10:
        return None
    out = {}
    for k, v in res["values"].items():
        out[k] = (10.0 ** float(v)) if "@" in k else float(v)
    return out


def bootstrap_global(fa: dict, *, n_bootstrap: int = 200, method: str = "residual",
                     alpha: float = 0.05, n_jobs: int = 1, seed: Optional[int] = 42,
                     progress_callback: Optional[Callable] = None) -> dict:
    """Residual bootstrap of the global fit: every experiment's data are
    resampled, then all are refit together.  Same result layout as the
    single-experiment bootstraps."""
    from equilibrist_bootstrap import _run_iter, _finalise_samples, _ci_block, _se_block
    t0 = time.time()
    base = _baseline(fa)
    names = _names_of(fa)
    best_fit = {k: float(v) for k, v in (fa.get("global_values") or {}).items()}
    best_fit = {k: (10.0 ** v if "@" in k else v) for k, v in best_fit.items()}

    def _worker(i, rng):
        try:
            overrides = {}
            for e in fa["experiments"]:
                overrides[e["label"]] = {"data": _synth_experiment(e["fa"], base[e["label"]], rng, method)}
            return _light_values(fa, _experiments_for(fa, {}, overrides=overrides))
        except Exception:
            return None

    raw = _run_iter(_worker, (), int(n_bootstrap), n_jobs, progress_callback, np.random.SeedSequence(seed))
    samples, n_ok, n_fail = _finalise_samples(raw, names)
    return {"samples": samples, "ci": _ci_block(samples, alpha), "bs_se": _se_block(samples),
            "ci_levels": (alpha / 2, 0.5, 1 - alpha / 2), "method": method,
            "n_bootstrap": int(n_bootstrap), "n_success": n_ok, "n_failed": n_fail,
            "best_fit": best_fit, "best_stats": {"global": True, "labels": fa.get("labels")},
            "wall_seconds": time.time() - t0, "best_message": "global fit"}


def jackknife_global(fa: dict, *, n_jobs: int = 1,
                     progress_callback: Optional[Callable] = None) -> dict:
    """Leave-one-titration-step-out over ALL experiments, refitting globally."""
    from equilibrist_bootstrap import (_run_iter, _jackknife_finalise, _build_param_kinds,
                                       _enumerate_steps_column_dict, _drop_row_from_column_dict,
                                       _enumerate_steps_spectra, _drop_row_from_spectra, _jackknife_empty)
    t0 = time.time()
    base = _baseline(fa)
    names = _names_of(fa)
    full = {k: (10.0 ** float(v) if "@" in k else float(v)) for k, v in (fa.get("global_values") or {}).items()}
    steps, owners = [], []                              # (row, x, n_obs) ; (label, mode)
    for e in fa["experiments"]:
        mode, data, st_ = _mode_of(e["fa"]), _data_of(e["fa"]), base.get(e["label"]) or {}
        if mode.startswith("kin_"):
            mode = mode[4:]
        if mode == "itc":
            q = np.asarray(data, dtype=float)
            x_ = st_.get("x_inj")
            s_ = [(i, float(x_[i]) if x_ is not None and i < len(x_) else float(i), 1) for i in range(len(q))]
        elif mode == "spectra":
            s_ = _enumerate_steps_spectra(data, st_.get("x_exp"))
        else:
            s_ = _enumerate_steps_column_dict(data, st_.get("per_col_x"))
        for row, x, n in s_:
            steps.append((row, x, n)); owners.append((e["label"], mode))
    if not steps:
        return {**_jackknife_empty("no finite titration steps"), "best_fit": full,
                "wall_seconds": time.time() - t0}

    def _worker(idx, _rng):
        try:
            lab, mode = owners[idx]
            e = next(x for x in fa["experiments"] if x["label"] == lab)
            data = _data_of(e["fa"])
            row = steps[idx][0]
            if mode == "itc":
                # position in the (already filtered) heat vector -> absolute injection index
                base_excl = set(e["fa"].get("excl_set") or set())
                kept = [i for i in range(len(data) + len(base_excl)) if i not in base_excl][:len(data)]
                drop_abs = kept[row] if row < len(kept) else row
                ov = {"data": np.delete(np.asarray(data, dtype=float), row), "excl": base_excl | {drop_abs}}
            elif mode == "spectra":
                ov = {"data": _drop_row_from_spectra(data, row)}
            else:
                ov = {"data": _drop_row_from_column_dict(data, row)}
            return _light_values(fa, _experiments_for(fa, {}, overrides={lab: ov}))
        except Exception:
            return None

    raw = _run_iter(_worker, (), len(steps), n_jobs, progress_callback, np.random.SeedSequence(42))
    samples, jack_se, influence, n_ok, n_fail = _jackknife_finalise(raw, names, full)
    kinds = _build_param_kinds(log_keys=[n for n in names if "@" not in n],
                               fit_conc_keys=[n for n in names if "@" in n], fit_titrant_keys=[])
    return {"steps": steps, "step_labels": [o[0] for o in owners], "samples": samples,
            "influence": influence, "jack_se": jack_se, "best_fit": full,
            "best_stats": {"global": True, "labels": fa.get("labels")}, "param_kinds": kinds,
            "n_jack": len(steps), "n_success": n_ok, "n_failed": n_fail,
            "wall_seconds": time.time() - t0, "best_message": "global fit"}


def monte_carlo_global(fa: dict, *, nuisance_specs: dict, n_iter: int = 200, n_jobs: int = 1,
                       seed: int = 42, progress_callback: Optional[Callable] = None) -> dict:
    """Nuisance Monte Carlo on the global fit: the relative uncertainties in
    ``nuisance_specs`` ("conc0:<name>", "titrant:<name>", "V0_mL") are applied
    to every experiment that has that quantity, sampled independently."""
    from equilibrist_bootstrap import _run_iter, _mc_finalise, _build_param_kinds, _sample_nuisance_params
    t0 = time.time()
    names = _names_of(fa)
    full = {k: (10.0 ** float(v) if "@" in k else float(v)) for k, v in (fa.get("global_values") or {}).items()}

    def _worker(idx, rng):
        try:
            from equilibrist_bootstrap import _mc_nmr_data
            overrides = {}
            for e in fa["experiments"]:
                efa = e["fa"]
                ov = {"params": _sample_nuisance_params(efa.get("params") or {}, nuisance_specs, rng)}
                if efa.get("use_nmr_fit") and efa.get("nmr_data_fit") is not None:
                    ov["data"] = _mc_nmr_data(efa["nmr_data_fit"], efa.get("parsed_fit"), nuisance_specs, rng)
                overrides[e["label"]] = ov
            return _light_values(fa, _experiments_for(fa, {}, overrides=overrides))
        except Exception:
            return None

    raw = _run_iter(_worker, (), int(n_iter), n_jobs, progress_callback, np.random.SeedSequence(seed))
    samples, mc_se, mc_median, lo, hi, n_ok, n_fail = _mc_finalise(raw, names)
    kinds = _build_param_kinds(log_keys=[n for n in names if "@" not in n],
                               fit_conc_keys=[n for n in names if "@" in n], fit_titrant_keys=[])
    return {"samples": samples, "mc_se": mc_se, "mc_median": mc_median, "mc_ci_lo": lo, "mc_ci_hi": hi,
            "best_fit": full, "best_stats": {"global": True, "labels": fa.get("labels")},
            "param_kinds": kinds, "nuisance_specs": dict(nuisance_specs), "n_iter": int(n_iter),
            "n_success": n_ok, "n_failed": n_fail, "wall_seconds": time.time() - t0,
            "best_message": "global fit"}


# ── Cluster offload hook (see equilibrist_hpc.py) ────────────────────────────
try:
    from equilibrist_hpc import install_offloadable as _install_offloadable
    _install_offloadable(globals(), ["single_fit", "global_fit", "bootstrap_global", "jackknife_global", "monte_carlo_global"],
                         "equilibrist_global")
except ImportError:
    pass
