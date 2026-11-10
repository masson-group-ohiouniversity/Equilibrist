# -*- coding: utf-8 -*-
"""equilibrist_itc.py

MicroCal iTC200 file parser, peak integrator, and enthalpogram fitter
for Equilibrist v2.

Physics
-------
For each injection n (1-based, dummy excluded):
  q(n) [µcal] = Σ_rxn  ΔH_rxn [kcal/mol]
                      × Δ[complex_rxn](n) [mM]
                      × V_cell [mL]
              + q_offset [µcal]

Units work out: kcal/mol × mM × mL = kcal/mol × 10⁻³ mol/L × 10⁻³ L
              = 10⁻⁶ kcal = 1 µcal  ✓

ΔH values are solved analytically (lstsq) at fixed K, mirroring how ε is
solved in the UV-Vis fitter.  Only K (and optionally concentrations) are
optimised non-linearly.

The cell is treated as solution-mode (expanding volume) which introduces a
small error (~3%) for typical injection volumes vs cell volume.  This matches
the standard Origin/PEAQ-ITC approximation and is negligible for most ITC
experiments.
"""

import re
import numpy as np
import equilibrist_live as _live
import time
from scipy.optimize import minimize

from equilibrist_curve import compute_single_point, find_equiv_for_x, convert_exp_x, solve_equilibria_general
from equilibrist_network import build_network
from equilibrist_parser import constraints_penalty, PenaltyScale, FitTrace, tied_minimize

__all__ = [
    "ITC200_CELL_VOLUME_ML",
    "parse_itc_file",
    "auto_integrate",
    "integrate_with_baselines",
    "fit_itc",
]

# MicroCal iTC200 standard cell volume [mL]
ITC200_CELL_VOLUME_ML = 1.4245


# ═══════════════════════════════════════════════════════════════════
# FILE PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_itc_file(content: bytes) -> dict:
    """Parse a MicroCal iTC200 .itc raw thermogram file.

    Parameters
    ----------
    content : bytes
        Raw file bytes.

    Returns
    -------
    dict with keys:
        n_injections      : int   — number of injections declared in header
        temperature_C     : float — experiment temperature (°C)
        cell_conc_mM      : float — cell concentration [mM] from # block
        syringe_conc_mM   : float — syringe concentration [mM] from # block
        inj_vols_uL       : list[float]  — per-injection volumes [µL], index 0 = dummy
        inj_rates         : list[float]  — per-injection rates [µL/min]
        time_s            : np.ndarray   — time axis [s]
        power_ucal_s      : np.ndarray   — differential power [µcal/s]
        inj_start_idx     : list[int]    — index in time/power arrays of each @N marker
        inj_start_time    : list[float]  — time [s] of each @N marker
        dt_s              : float        — data interval [s] (typically 5 s)
        software_version  : str
    """
    text = content.decode("latin-1", errors="replace")
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]

    result = {
        "n_injections":    0,
        "temperature_C":   25.0,
        "cell_conc_mM":    None,
        "syringe_conc_mM": None,
        "inj_vols_uL":     [],
        "inj_rates":       [],
        "time_s":          None,
        "power_ucal_s":    None,
        "inj_start_idx":   [],
        "inj_start_time":  [],
        "dt_s":            5.0,
        "software_version": "",
    }

    # ── Parse header sections ─────────────────────────────────────
    dollar_lines = []   # $ block lines (after $ITC)
    hash_lines   = []   # # block lines
    pct_lines    = []   # % block lines
    data_lines   = []   # numeric data rows
    inj_markers  = []   # (@N, line_index_in_data_lines, vol, rate)

    _in_dollar = False
    _in_hash   = False
    _in_pct    = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue

        if stripped.startswith("$ITC"):
            _in_dollar = True
            _in_hash   = False
            _in_pct    = False
            continue

        if stripped.startswith("$"):
            if _in_dollar:
                dollar_lines.append(stripped[1:].strip())
            continue

        if stripped.startswith("#"):
            _in_dollar = False
            _in_hash   = True
            _in_pct    = False
            hash_lines.append(stripped[1:].strip())
            continue

        if stripped == "?":
            _in_hash = False
            _in_pct  = False
            continue

        if stripped.startswith("%"):
            _in_hash   = False
            _in_pct    = True
            pct_lines.append(stripped[1:].strip())
            continue

        # Injection marker: @N  or  @N,vol,rate
        if stripped.startswith("@"):
            _in_dollar = _in_hash = _in_pct = False
            parts = stripped[1:].split(",")
            try:
                n   = int(parts[0].strip())
                vol  = float(parts[1].strip()) if len(parts) > 1 else 0.0
                rate = float(parts[2].strip()) if len(parts) > 2 else 0.0
            except (ValueError, IndexError):
                continue
            inj_markers.append((n, len(data_lines), vol, rate))
            continue

        # Numeric data row
        if re.match(r"^\s*[\d\.\-]", stripped):
            _in_dollar = _in_hash = _in_pct = False
            data_lines.append(stripped)
            continue

    # ── Extract per-injection volumes from $ block ────────────────
    # Each $ line after the first few fixed-header lines is:
    # "vol, rate, spacing, ?"
    # First two lines are n_injections and temperature; we detect them.
    _dollar_inj = []
    _dollar_singles = []
    for dl in dollar_lines:
        # Lines with commas are injection parameter lines
        if "," in dl:
            parts = [p.strip() for p in dl.split(",")]
            try:
                v  = float(parts[0])
                r  = float(parts[1]) if len(parts) > 1 else 1.6
                _dollar_inj.append((v, r))
            except ValueError:
                pass
        else:
            # Single-value lines: first = n_injections, second = temperature.
            # NB: this used to stash the list on `result` guarded by
            # hasattr(result, ...). `result` is a dict, so hasattr tests
            # attributes, never keys — it was always False and the list was
            # re-initialised on every line, leaving only the last value
            # (ADCGainCode: 0), which made n_injections always 0.
            try:
                v = float(dl.replace("ADCGainCode:", "").replace("True", "").replace("False", "").strip())
                _dollar_singles.append(v)
            except (ValueError, AttributeError):
                pass

    # n_injections from $ block (first numeric single value)
    singles = _dollar_singles
    if singles:
        try:
            result["n_injections"] = int(singles[0])
        except (ValueError, IndexError):
            pass
    if len(singles) >= 2:
        try:
            result["temperature_C"] = float(singles[1])
        except (ValueError, IndexError):
            pass

    # Per-injection vols/rates
    if _dollar_inj:
        result["inj_vols_uL"] = [x[0] for x in _dollar_inj]
        result["inj_rates"]   = [x[1] for x in _dollar_inj]

    # ── Extract concentrations from # block ───────────────────────
    # The # block typically has these values (by position):
    # 0: offset/misc
    # 1: misc
    # 2: misc
    # 3: cell concentration [mM]
    # 4: temperature
    # 5: syringe concentration [mM]
    # We identify them by matching against the % block which repeats them.
    _hash_floats = []
    for hl in hash_lines:
        try:
            _hash_floats.append(float(hl.strip()))
        except ValueError:
            pass

    # # block layout (iTC200):
    #   [0] = 0       (offset/misc)
    #   [1] = concentration A [mM]  (order varies between experiments)
    #   [2] = concentration B [mM]
    #   [3] = cell volume [mL]
    #   [4] = temperature [°C]
    # The cell is always at lower concentration than the syringe in a normal
    # ITC experiment (macromolecule in cell, ligand in syringe at 5-10x excess).
    # Assign: smaller value = cell concentration, larger = syringe concentration.
    if len(_hash_floats) >= 4:
        _cA = float(_hash_floats[1])
        _cB = float(_hash_floats[2])
        result["cell_conc_mM"]    = min(_cA, _cB)
        result["syringe_conc_mM"] = max(_cA, _cB)
        result["cell_vol_mL"]     = float(_hash_floats[3])
    if len(_hash_floats) >= 5:
        result["temperature_C"]   = float(_hash_floats[4])

    # Software version from first % line
    if pct_lines:
        sv = pct_lines[0].strip()
        if "ITC" in sv or "Ver" in sv:
            result["software_version"] = sv

    # ── Parse raw data ────────────────────────────────────────────
    times   = []
    powers  = []
    for dl in data_lines:
        parts = dl.split(",")
        if len(parts) < 2:
            continue
        try:
            t   = float(parts[0].strip())
            pwr = float(parts[1].strip())
            times.append(t)
            powers.append(pwr)
        except ValueError:
            continue

    if times:
        result["time_s"]       = np.array(times, dtype=float)
        result["power_ucal_s"] = np.array(powers, dtype=float)
        # Detect dt from first interval
        if len(times) >= 2:
            result["dt_s"] = float(times[1] - times[0])

    # ── Map injection markers to array indices ────────────────────
    for n, data_idx, vol, rate in inj_markers:
        result["inj_start_idx"].append(data_idx)
        t_val = times[data_idx] if data_idx < len(times) else 0.0
        result["inj_start_time"].append(t_val)

    # If injection volumes weren't in $ block, use marker data
    if not result["inj_vols_uL"] and inj_markers:
        result["inj_vols_uL"] = [x[2] for x in inj_markers]
        result["inj_rates"]   = [x[3] for x in inj_markers]

    # ── Trim planned injections down to those that actually ran ───
    # The $ header lists every PLANNED injection, but a run is routinely
    # stopped early once the isotherm saturates. The @ markers record what
    # actually happened. Without this trim the overflow model generates
    # concentrations for injections that never occurred, and the ΔH least-
    # squares step fails outright with "Incompatible dimensions".
    # $-block entry i corresponds to marker @(i+1), so the count we keep is
    # the number of markers carrying a real volume.
    _n_actual = sum(1 for m in inj_markers if m[2] > 0)
    if _n_actual == 0:
        _n_actual = max(len(inj_markers) - 1, 0)
    if 0 < _n_actual < len(result["inj_vols_uL"]):
        result["n_injections_planned"] = len(result["inj_vols_uL"])
        result["inj_vols_uL"] = result["inj_vols_uL"][:_n_actual]
        result["inj_rates"]   = result["inj_rates"][:_n_actual]

    return result


# ═══════════════════════════════════════════════════════════════════
# PEAK INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def _estimate_plateau(power: np.ndarray, start: int, n_pts: int = 12) -> float:
    """Mean of n_pts data points ending at index start (pre-injection plateau)."""
    lo = max(0, start - n_pts)
    hi = start
    if hi > lo:
        return float(np.mean(power[lo:hi]))
    elif start > 0:
        return float(power[max(0, start - 1)])
    return float(power[0]) if len(power) > 0 else 0.0


def _detect_eq_time(corrected: np.ndarray, dt: float,
                     threshold: float = 3.0, n_pts: int = 12,
                     min_pts: int = 3) -> int:
    """Return index (within corrected) where signal has recovered to baseline.

    Uses a peak-normalised tolerance: the signal is deemed recovered when it
    stays within max(threshold% × peak_depth, threshold × noise) of zero.
    This gives consistent ~40-50s EqTimes regardless of peak size
    (threshold=3 → ~2% of peak depth).

    Parameters
    ----------
    corrected   : signal with linear baseline already subtracted
    dt          : time step [s]
    threshold   : controls tolerance: tol ≈ (threshold/150) × peak_depth
                  default 3 → ~2% of peak
    n_pts       : points for absolute noise fallback estimate
    min_pts     : consecutive points within tolerance to call recovery

    Returns index of recovery, or len(corrected) if never recovered.
    """
    n = len(corrected)
    if n == 0:
        return 0

    # Peak: maximum absolute deviation from zero
    peak_depth = float(np.max(np.abs(corrected)))
    peak_idx   = int(np.argmax(np.abs(corrected)))
    if peak_depth < 1e-9:
        return 0

    # Absolute noise floor from post-injection plateau
    _noise_region = corrected[max(0, n - n_pts):]
    noise = float(np.std(_noise_region)) if len(_noise_region) > 1 else 1e-6
    if noise < 1e-9:
        noise = 1e-6

    # Tolerance = max of peak-normalised and absolute-noise criteria
    # threshold/150 ≈ 0.02 at the default threshold=3
    peak_frac = threshold / 150.0
    tol = max(peak_frac * peak_depth, threshold * noise)

    # Scan from after the peak for recovery
    start_scan = peak_idx + 1
    for j in range(start_scan, n - min_pts + 1):
        if all(abs(corrected[j + k]) <= tol for k in range(min_pts)):
            return j
    return n  # never fully recovered within window


def auto_integrate(raw: dict, n_baseline_pts: int = 12,
                   threshold: float = 3.5,
                   tail_pts: int = 8) -> tuple:
    """Integrate ITC injection peaks using linear baseline and peak-picking.

    Algorithm:
    1. For each real injection window:
       a. pre_bl  = mean of last n_baseline_pts points before injection
       b. post_bl = mean of last n_baseline_pts points of window
       c. linear_baseline = linear interp from pre_bl to post_bl over window
       d. corrected = power - linear_baseline
       e. EqTime = first index where corrected stays within threshold×noise
       f. Integrate corrected[0 : EqTime + tail_pts]

    The dummy injection (index 0) is always excluded from the returned arrays.
    Both pre_bl and post_bl are stored; the user can adjust post_bl per injection.

    Parameters
    ----------
    raw            : output of parse_itc_file()
    n_baseline_pts : points used for pre/post plateau estimation (default 12)
    threshold      : noise-multiplier for EqTime detection (default 3)
    tail_pts       : extra points integrated past EqTime to catch the tail
                     (default 8 = 40 s at dt = 5 s)

    Returns
    -------
    q_ucal         : np.ndarray (n_real,) — integrated heats [µcal]
    pre_baselines  : np.ndarray (n_total,) — pre-injection levels [µcal/s]
                     (index 0 = pre-eq window; indices 1..N = real injections)
    post_baselines : np.ndarray (n_total,) — post-injection levels [µcal/s]
    eq_times_s     : np.ndarray (n_total,) — detected EqTime [s] per window
    """
    power  = raw["power_ucal_s"]
    time_s = raw["time_s"]
    dt     = raw["dt_s"]
    starts = raw["inj_start_idx"]

    if power is None or len(starts) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_total = len(starts)
    pre_bl   = np.zeros(n_total)
    post_bl  = np.zeros(n_total)
    eq_t     = np.zeros(n_total)

    # ── Pass 1: compute pre baselines for every window ───────────────────
    for i in range(n_total):
        start = starts[i]
        pre_bl[i] = _estimate_plateau(power, start, n_baseline_pts)

    # Post baseline for window i = pre baseline of window i+1.
    # The post-injection settled level equals the
    # pre-injection plateau of the next injection, giving a linear baseline
    # that bridges two fully-settled segments without residual signal bias.
    for i in range(n_total):
        if (i + 1) < n_total:
            post_bl[i] = pre_bl[i + 1]
        else:
            post_bl[i] = pre_bl[i]   # last window: flat baseline

    # ── Pass 2: integrate each real injection (skip dummy = index 1) ─────
    q_ucal = np.zeros(n_total - 1)   # q_ucal[j] = real injection j+1 (0-indexed)

    for i in range(1, n_total):   # i indexes @N markers; i=1 is dummy, i≥2 are real
        start = starts[i]
        end   = starts[i + 1] if (i + 1) < n_total else len(power)
        n_win = end - start
        if n_win <= 0:
            continue

        # Linear baseline: ramp from pre_bl[i] to post_bl[i]
        t_frac    = np.linspace(0.0, 1.0, n_win)
        lin_bl    = pre_bl[i] + (post_bl[i] - pre_bl[i]) * t_frac
        corrected = power[start:end] - lin_bl

        # Detect equilibration time
        eq_idx = _detect_eq_time(corrected, dt, threshold=threshold,
                                  n_pts=n_baseline_pts)
        int_end = min(eq_idx + tail_pts, n_win)
        eq_t[i] = float(eq_idx * dt)

        # Integrate up to EqTime + tail
        q_ucal[i - 1] = float(np.sum(corrected[:int_end] * dt))

    # Return all injections: q_ucal[0] = dummy, q_ucal[1:] = real injections.
    # Callers must always skip q_ucal[0] when passing to fit_itc.
    return q_ucal, pre_bl[1:], post_bl[1:], eq_t[1:]


def integrate_with_baselines(raw: dict,
                              pre_baselines: np.ndarray,
                              post_baselines: np.ndarray = None,
                              threshold: float = 3.0,
                              tail_pts: int = 8,
                              n_baseline_pts: int = 12) -> np.ndarray:
    """Re-integrate using user-supplied pre and post baselines.

    Called when the user adjusts baselines in the UI.  Uses the same
    linear-baseline + peak-picking algorithm as auto_integrate.

    Parameters
    ----------
    raw            : output of parse_itc_file()
    pre_baselines  : (n_total,) array — pre-injection levels (index 0 = pre-eq)
    post_baselines : (n_total,) array — post-injection levels; if None, uses pre
    threshold      : EqTime detection threshold
    tail_pts       : extra points past EqTime
    n_baseline_pts : points used for noise estimation

    Returns
    -------
    q_ucal : np.ndarray (n_real,) — skips dummy, length = n_total - 2
    """
    power  = raw["power_ucal_s"]
    dt     = raw["dt_s"]
    starts = raw["inj_start_idx"]

    if power is None or len(starts) < 2:
        return np.array([])

    n_total = len(starts)

    if post_baselines is None:
        post_baselines = pre_baselines.copy()

    q_ucal = np.zeros(n_total - 1)

    for i in range(1, n_total):
        start = starts[i]
        end   = starts[i + 1] if (i + 1) < n_total else len(power)
        n_win = end - start
        if n_win <= 0:
            continue

        t_frac    = np.linspace(0.0, 1.0, n_win)
        lin_bl    = pre_baselines[i] + (post_baselines[i] - pre_baselines[i]) * t_frac
        corrected = power[start:end] - lin_bl

        eq_idx = _detect_eq_time(corrected, dt, threshold=threshold,
                                  n_pts=n_baseline_pts)
        int_end = min(eq_idx + tail_pts, n_win)
        q_ucal[i - 1] = float(np.sum(corrected[:int_end] * dt))

    # Return all injections including dummy at [0], matching auto_integrate behaviour.
    return q_ucal


# ═══════════════════════════════════════════════════════════════════
# CONCENTRATION CURVES AT INJECTION POINTS
# ═══════════════════════════════════════════════════════════════════

def _itc_overflow_concentrations(raw: dict, params: dict) -> tuple:
    """Compute G and H total concentrations at each real injection using the
    iTC200 overflow (constant cell volume) model.

        [G]_n = [G]_{n-1} * V0/(V0+dVn)
        [H]_n = [H]_{n-1} * V0/(V0+dVn) + H_stock * dVn/(V0+dVn)

    The dummy injection (index 0) is included in the running totals because it
    physically changes cell concentrations even though its heat is excluded.
    """
    vols_uL = raw["inj_vols_uL"]
    V0      = float(params["V0_mL"])
    G0      = float(list(params["conc0"].values())[0])
    H_stock = float(list(params["titrant_mMs"].values())[0])

    G_conc = G0
    H_conc = 0.0
    G_totals = []
    H_totals = []

    for i, v_uL in enumerate(vols_uL):
        dV = v_uL / 1000.0
        G_conc = G_conc * V0 / (V0 + dV)
        H_conc = H_conc * V0 / (V0 + dV) + H_stock * dV / (V0 + dV)
        if i >= 1:   # skip dummy (i=0), record real injections
            G_totals.append(G_conc)
            H_totals.append(H_conc)

    return np.array(G_totals), np.array(H_totals)


def _itc_injection_x_vals(raw: dict, parsed: dict, params: dict, x_expr: str,
                           network: dict) -> np.ndarray:
    """Return x-axis values (H_total/G_total molar ratio) at each real injection
    using the overflow model."""
    G_tots, H_tots = _itc_overflow_concentrations(raw, params)
    G_safe = np.where(G_tots > 1e-12, G_tots, 1e-12)
    return H_tots / G_safe


def _reaction_usage_matrix(parsed: dict, rxn_valid: list) -> np.ndarray:
    """How many times each reaction is used to build each product species.

    The heat of an injection is sum_k dH_k * d(xi_k), where xi_k is the EXTENT
    of reaction k - not the concentration of its product. For sequential
    binding the distinction matters: forming GH2 via

        G  + H = GH    (reaction 1)
        GH + H = GH2   (reaction 2)

    advances reaction 1 as well as reaction 2, so
        xi_1 = [GH] + [GH2]      xi_2 = [GH2]

    Using [GH] alone as the extent of reaction 1 forces the fitted dH_2 to
    absorb dH_1, i.e. it becomes the CUMULATIVE enthalpy of G + 2H -> GH2
    rather than the stepwise enthalpy of GH + H -> GH2 that the reaction
    notation and the UI label both promise.

    Returns M with M[s, k] = times reaction k is used to form product species
    s, so that  extent_k = sum_s M[s, k] * [s]  and  A = dC @ M.
    For a single reaction M is [[1]], so 1:1 fits are completely unaffected.
    """
    eqs = parsed.get("equilibria", []) or []
    prods = [sp for _, sp in rxn_valid]
    idx = {sp: i for i, sp in enumerate(prods)}
    n = len(rxn_valid)
    M = np.eye(n)
    if n < 2:
        return M

    def _resolve(slot, seen):
        rxn_i = rxn_valid[slot][0]
        v = np.zeros(n)
        v[slot] = 1.0
        if rxn_i >= len(eqs) or slot in seen:
            return v
        seen = seen | {slot}
        for st, sp in eqs[rxn_i].get("reactants", []):
            if sp in idx:                     # reactant is itself a complex
                v = v + float(st) * _resolve(idx[sp], seen)
        return v

    for slot in range(n):
        try:
            M[idx[prods[slot]]] = _resolve(slot, set())
        except Exception:
            return np.eye(n)                  # never worse than the old behaviour
    return M


def _compute_complex_concs_at_injections(
        x_vals: np.ndarray,      # kept for API compatibility (unused)
        parsed: dict,
        network: dict,
        logK_vals: dict,
        params: dict,
        rxn_product_species: list) -> np.ndarray:
    """Compute [complex] [mM] at each real injection using the overflow model.

    Passes overflow-model G_total and H_total directly to the equilibrist
    mass-balance solver, bypassing the growing-volume titration model which
    is incorrect for the constant-cell-volume iTC200.
    """
    raw = params.get("_itc_raw")
    if raw is None:
        raise RuntimeError("_itc_raw missing from params")

    G_tots, H_tots = _itc_overflow_concentrations(raw, params)
    n_inj = len(G_tots)
    n_rxn = len(rxn_product_species)
    C = np.zeros((n_inj, n_rxn))

    equilibria  = network.get("equilibria", [])
    all_species = network["all_species"]
    primary     = params.get("primary_component", list(params["conc0"].keys())[0])
    tit_free    = params["titrant_free_names"][0]
    _conc0_all  = dict(params.get("conc0", {}) or {})
    _primary_c0 = float(_conc0_all.get(primary, 0.0))
    _other_cell = {k: v for k, v in _conc0_all.items()
                   if k != primary and k != tit_free and v is not None
                   and k in all_species}

    for j in range(n_inj):
        try:
            totals = {sp: 0.0 for sp in all_species}
            totals[primary]  = float(G_tots[j]) * 1e-3   # mM → M
            totals[tit_free] = float(H_tots[j]) * 1e-3
            # Any OTHER component present in the cell (a competitor, buffer or
            # chelator declared in $concentrations) must be carried through
            # too. Previously every non-primary, non-titrant species was left
            # at zero, so its equilibria could never proceed: the reaction was
            # silently inert and its ΔH column was all zeros, which made the
            # lstsq singular and produced absurd ΔH values.
            # They are diluted by the same overflow factor as the primary,
            # since neither is present in the syringe.
            if _other_cell:
                _dil = (float(G_tots[j]) / _primary_c0) if _primary_c0 > 0 else 1.0
                for _sp, _c0 in _other_cell.items():
                    totals[_sp] = float(_c0) * _dil * 1e-3

            concs_M, _, _ = solve_equilibria_general(
                totals, equilibria, all_species, logK_vals)

            for k, sp in enumerate(rxn_product_species):
                if sp and sp in concs_M:
                    C[j, k] = max(float(concs_M[sp]) * 1e3, 0.0)  # M → mM
        except Exception:
            pass

    return C


def _delta_C(C: np.ndarray, inj_vols_mL: np.ndarray = None,
             V0_mL: float = None, C_dummy: np.ndarray = None) -> np.ndarray:
    """Compute Δ[complex] between consecutive injections using the correct
    overflow (constant cell volume) model:

        dC[i] = C[i] - C[i-1] × V0 / (V0 + dVi)

    Without volume correction (inj_vols_mL=None), falls back to simple diff.
    The V0/(V0+dVi) factor accounts for the fact that the previous complex
    is diluted by overflow before new binding occurs.  Without it, late-
    injection heats are wrong (dilution drop in C is misread as dissociation).

    Parameters
    ----------
    C            : (n_inj, n_rxn) — complex concentrations at each injection
    inj_vols_mL  : (n_inj,) — injection volumes in mL (real injections, dummy excluded)
    V0_mL        : cell volume in mL

    Returns dC : same shape as C.
    """
    if inj_vols_mL is None or V0_mL is None:
        return np.diff(C, axis=0, prepend=np.zeros((1, C.shape[1])))

    n = C.shape[0]
    dC = np.zeros_like(C)
    for i in range(n):
        if i == 0:
            dVi = float(inj_vols_mL[0])
            dilution = V0_mL / (V0_mL + dVi)
            # C_dummy: complex concentration after dummy injection (initial state)
            # Passed as optional extra row prepended to C, or as keyword arg.
            # Default: assume previous state zero (conservative fallback).
            prev = C_dummy if C_dummy is not None else np.zeros(C.shape[1])
            dC[i] = C[i] - prev * dilution
        else:
            dVi = float(inj_vols_mL[i])
            dilution = V0_mL / (V0_mL + dVi)
            dC[i] = C[i] - C[i-1] * dilution
    return dC


def _rxn_product_species(parsed: dict) -> list:
    """Return the 'proxy' product species for each equilibrium reaction.

    For each equilibrium reaction we track the first product species listed.
    This is the extent-of-reaction proxy used to compute q(i).

    For 1:1 binding (G + H = GH), this returns ['GH'].
    For sequential binding (G + H = GH, GH + H = GH2), returns ['GH', 'GH2'].
    """
    species = []
    for eq in parsed.get("equilibria", []):
        prods = eq.get("products", [])
        if not prods and "product" in eq:
            prods = [eq["product"]]
        if prods:
            species.append(prods[0][1])   # (coeff, name) → name
        else:
            species.append(None)
    return species


# ═══════════════════════════════════════════════════════════════════
# ITC FITTER
# ═══════════════════════════════════════════════════════════════════

def _compute_dH_per_inj(q_exp, A_mat, dH_vec, q_offset, C_mat, V0_mL):
    """Convert q [µcal] to ΔH [kcal/mol] per injection for publication figure.

    ΔH_i = q_i / (dC_complex_i * V0 * 1e-3)  [kcal/mol]
    where dC is in mM and V0 in mL.
    """
    try:
        # A_mat columns: [dC*V0*1e3 per reaction, (optional 1 for offset)]
        # dC column 0 = A_mat[:,0] / (V0*1e3)
        n_inj = len(q_exp)
        n_rxn = C_mat.shape[1] if C_mat.ndim == 2 else 1
        if n_rxn == 1:
            dC_col = A_mat[:, 0] if A_mat.shape[1] >= 1 else np.ones(n_inj)
            q_binding = q_exp - q_offset
            with np.errstate(divide="ignore", invalid="ignore"):
                dH = np.where(np.abs(dC_col) > 1e-9,
                              q_binding / dC_col * dH_vec[0],
                              np.nan)
            # Simpler: q_i / (dC_i * V0 * 1e3) in kcal/mol
            # dC_col is already dC * V0 * 1e3, and dH_vec[0] is in kcal/mol
            # So ΔH_i = q_binding_i / dC_col_i * dH_vec[0] ... but that's circular.
            # Better: ΔH_i = (q_i - q_offset) / dC_col_i
            # where result is in kcal/mol because dC_col_i = dC[mM] * V0[mL] * 1e3
            # and q is in µcal, so µcal / (mM*mL*1e3) = µcal/µmol = cal/mmol = kcal/mol... wait
            # Actually µcal / (mM * mL) = µcal/µmol = µcal/µmol
            # 1 µcal/µmol = 1 cal/mmol = 1e-3 kcal/mmol = 1 kcal/mol * 1e-3
            # dC_col = dC[mM] * V0[mL] * 1000 -- the *1000 converts to µcal/(kcal/mol)
            # So q[µcal] / dC_col[µcal/(kcal/mol)] = ΔH[kcal/mol]  ✓
            with np.errstate(divide="ignore", invalid="ignore"):
                dH_per_inj = np.where(np.abs(dC_col) > 1e-9,
                                      (q_exp - q_offset) / dC_col,
                                      np.nan)
            return dH_per_inj
        else:
            # Multi-reaction: return total q / total dC (approximate)
            dC_total = np.sum(A_mat[:, :n_rxn], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(dC_total) > 1e-9,
                                (q_exp - q_offset) / dC_total,
                                np.nan)
    except Exception:
        return np.full(len(q_exp), np.nan)


def _compute_dH_norm(q_arr: np.ndarray, raw: dict) -> np.ndarray:
    """Normalise q [µcal] to ΔH [kcal/mol] by moles of titrant injected per injection.

    This is the standard ITC isotherm normalisation:
        ΔH_i = q_i [µcal] / (H_stock [mM] × dV_i [mL]) / 1000

    This is the normalised molar heat per injection, in kcal/mol.
    q_arr must correspond to real injections (dummy excluded), aligned with
    raw["inj_vols_uL"][1:].
    """
    try:
        vols_uL  = np.array(raw.get("inj_vols_uL", []))
        H_stock  = float(raw.get("syringe_conc_mM", 5.0))
        n = len(q_arr)
        out = np.full(n, np.nan)
        for i in range(n):
            dV = vols_uL[i + 1] / 1000.0 if (i + 1) < len(vols_uL) else 0.002
            n_H = H_stock * dV   # µmol
            if n_H > 1e-12:
                out[i] = float(q_arr[i]) / n_H / 1000.0  # kcal/mol
        return out
    except Exception:
        return np.full(len(q_arr), np.nan)


def _pad_to_full(arr: np.ndarray, incl_mask: np.ndarray) -> np.ndarray:
    """Expand a values-for-included-only array back to full length with nan at excluded."""
    n_full = len(incl_mask)
    out = np.full(n_full, np.nan)
    out[incl_mask] = arr
    return out


def fit_itc(parsed: dict, network: dict, q_exp: np.ndarray,
            raw: dict, params: dict, logK_vals: dict,
            fit_keys: list, x_expr: str,
            tolerance: float = 1e-6, maxiter: int = 100_000,
            timeout_s: float = 30.0,
            fit_conc_keys: list = None,
            fit_titrant_keys: list = None,
            q_offset_fixed: float = None,
            constraints: list = None,
            excl_set: set = None,
            _objective_only: bool = False) -> tuple:
    """Fit ITC enthalpogram data.

    Optimises K (and optionally concentrations) non-linearly.
    ΔH per reaction and q_offset are solved analytically (lstsq) at each
    optimizer step, identical to the ε-lstsq approach in UV-Vis fitting.

    Parameters
    ----------
    parsed          : Equilibrist parsed script dict
    network         : Equilibrist network dict
    q_exp           : np.ndarray [µcal] — integrated heats, real injections only
    raw             : dict from parse_itc_file()
    params          : Equilibrist params dict (conc0, V0_mL, titrant_mMs, ...)
    logK_vals       : dict  kname → logK (current sidebar values)
    fit_keys        : list of knames to optimise
    x_expr          : x-axis expression string
    tolerance       : Nelder-Mead convergence tolerance
    maxiter         : maximum NM iterations
    timeout_s       : wall-clock timeout [s]; best result returned if exceeded
    fit_conc_keys   : list of concentration root names to optimise (e.g. ['G'])
    q_offset_fixed  : if not None, hold q_offset at this value (not fitted)
    constraints     : list of Equilibrist constraint dicts

    Returns
    -------
    (success, fitted_logKs, stats, message)

    stats keys:
        r_squared, rmse, ssr, n_points, n_params, n_iter,
        param_values, param_errors,
        dH_values     : {rxn_kname: ΔH [kcal/mol]}
        dH_errors     : {rxn_kname: ±ΔH [kcal/mol]}
        q_offset      : float [µcal]
        q_theo        : np.ndarray [µcal] — theoretical heats at best fit
        x_inj         : np.ndarray — x-axis values at each injection
        C_back        : np.ndarray (n_inj, n_rxn) — back-calculated [complex] [mM]
        fitted_concs  : {root: mM}
        timed_out     : bool
    """
    try:
        fit_conc_keys = list(fit_conc_keys or [])
        fit_titrant_keys = list(fit_titrant_keys or [])
        constraints   = constraints or []
        excl_set      = set(excl_set or [])
        # Inject raw into base params so overflow model functions can access it
        params = dict(params)
        params["_itc_raw"] = raw  # indices into full injection array to exclude

        n_logK = len(fit_keys)
        n_conc = len(fit_conc_keys)
        n_inj  = len(q_exp)

        if n_inj < 2:
            return False, {}, {}, "Too few injections for fitting"

        # ── Identify reaction product species ─────────────────────
        rxn_proxy_sp = _rxn_product_species(parsed)
        # Filter to only equilibrium reactions (not None)
        rxn_valid = [(i, sp) for i, sp in enumerate(rxn_proxy_sp) if sp is not None]
        n_rxn = len(rxn_valid)
        _rxn_usage_M = _reaction_usage_matrix(parsed, rxn_valid)
        # Enthalpies the user pinned with "; dH = <value>" in the script.
        # These are removed from the least-squares: their known contribution
        # is subtracted from q and their column dropped, which is what makes
        # displacement/competition ITC identifiable at all.
        _eqs_all = parsed.get("equilibria", []) or []
        _fixed_dH = {}
        for _slot, (_ri, _sp) in enumerate(rxn_valid):
            if _ri < len(_eqs_all):
                _v = _eqs_all[_ri].get("dH_fixed")
                if _v is not None:
                    _fixed_dH[_slot] = float(_v)
        n_rxn_free = n_rxn - len(_fixed_dH)

        if n_rxn == 0:
            return False, {}, {}, "No equilibrium reactions found"

        # Minimum data requirement
        n_params_total = n_logK + n_conc + n_rxn_free + (0 if q_offset_fixed is not None else 1)
        if n_inj < n_params_total:
            return False, {}, {}, (
                f"Too few injections ({n_inj}) for {n_params_total} parameters. "
                f"Need at least {n_params_total} real injections.")

        # ── Concentration bounds ───────────────────────────────────
        CONC_MIN = 0.0
        _root_to_cname = {}
        for cname in parsed.get("concentrations", {}):
            r = cname[:-1] if cname.endswith("0") else cname
            _root_to_cname[r] = cname

        def _cb(root, sv_override=None):
            cname = _root_to_cname.get(root, root)
            sv = sv_override if sv_override is not None else \
                 float(parsed.get("concentrations", {}).get(cname, 1.0))
            lo, hi = parsed.get("conc_bounds", {}).get(cname,
                     parsed.get("conc_bounds", {}).get(root, (None, None)))
            lo = max(CONC_MIN, lo) if lo is not None else max(CONC_MIN, sv * 0.80)
            hi = hi if hi is not None else sv * 1.20
            return (lo, hi)

        # Initial x0 for concentrations: use script values
        _conc_script = {}
        for cn, cv in parsed.get("concentrations", {}).items():
            r = cn[:-1] if cn.endswith("0") else cn
            _conc_script[r] = float(cv)
        x0_conc = np.array([_conc_script.get(r, 1.0) for r in fit_conc_keys])
        bds_conc = [_cb(r) for r in fit_conc_keys]

        # ── Titrant (syringe) stock concentrations ───────────────────
        # Fitting the syringe stock mirrors fitting the cell concentration:
        # both rescale the molar-ratio axis, so fitting either one absorbs
        # a systematic concentration error. Without this the titrant fit
        # checkbox was silently ignored and the stock never moved.
        _tit_nom = dict(params.get("titrant_mMs", {}) or {})

        def _tb(name):
            sv = float(_tit_nom.get(name, 10.0))
            return (max(CONC_MIN, sv * 0.80), sv * 1.20)

        x0_tit  = np.array([float(_tit_nom.get(t, 10.0)) for t in fit_titrant_keys])
        bds_tit = [_tb(t) for t in fit_titrant_keys]
        n_tit   = len(fit_titrant_keys)

        # ── x0 vector ─────────────────────────────────────────────
        x0_logK = np.array([logK_vals[k] for k in fit_keys])
        x0 = np.concatenate([x0_logK, x0_conc, x0_tit])
        n_p = len(x0)  # non-linear parameters only

        def _unpack(fp):
            logKs = {fit_keys[i]: float(np.clip(fp[i], -15, 15))
                     for i in range(n_logK)}
            concs = {fit_conc_keys[i]: float(np.clip(fp[n_logK + i],
                     bds_conc[i][0], bds_conc[i][1]))
                     for i in range(n_conc)}
            tits  = {fit_titrant_keys[i]: float(np.clip(fp[n_logK + n_conc + i],
                     bds_tit[i][0], bds_tit[i][1]))
                     for i in range(n_tit)}
            return logKs, concs, tits

        def _make_params(concs, tits=None):
            p = dict(params)
            p["_itc_raw"] = raw          # needed by _compute_complex_concs
            if concs:
                p["conc0"] = dict(params["conc0"])
                for root, val in concs.items():
                    p["conc0"][root] = float(val)
            if tits:
                p["titrant_mMs"] = dict(params.get("titrant_mMs", {}) or {})
                for name, val in tits.items():
                    p["titrant_mMs"][name] = float(val)
            return p

        def _make_parsed(concs, tits=None):
            if not concs and not tits:
                return parsed
            p = dict(parsed)
            if concs:
                p["concentrations"] = dict(parsed["concentrations"])
                for root, val in concs.items():
                    cname = _root_to_cname.get(root, root)
                    p["concentrations"][cname] = float(val)
            if tits:
                p["titrant"] = dict(parsed.get("titrant", {}) or {})
                for name, val in tits.items():
                    for _tk in list(p["titrant"]):
                        _root = _tk[:-1] if _tk.endswith(("t", "0")) else _tk
                        if _tk == name or _root == name:
                            p["titrant"][_tk] = float(val)
            return p

        def _build_A_matrix_from_dC(dC_masked):
            """Build design matrix from pre-computed, pre-masked dC.

            dC_masked : (n_included, n_rxn) — already excludes outlier rows
            A @ [ΔH₁, ..., ΔH_n, q_offset] ≈ q_exp
            """
            V0 = float(params["V0_mL"])
            # Columns must be reaction EXTENTS, not product concentrations,
            # or dH for every step after the first silently becomes cumulative.
            A = (dC_masked @ _rxn_usage_M) * V0 * 1000.0   # µcal / [kcal/mol]
            if q_offset_fixed is None:
                A = np.column_stack([A, np.ones(A.shape[0])])
            return A

        def _compute_dC_masked(C_all, cur_logKs_=None, rxn_sp_=None):
            """Compute dC for ALL injections using the correct overflow model,
            then mask out excluded rows.
            dC[i] = C[i] - C[i-1] × V0/(V0+dVi)  — prevents dilution artefacts.
            For i=0, the previous state is the equilibrium after the dummy injection
            (not zero), which matters when K is large enough that the dummy binds
            a non-negligible amount.
            """
            _vols_uL = raw.get("inj_vols_uL", [])
            _V0      = float(params["V0_mL"])
            n_all    = C_all.shape[0]
            n_rxn    = C_all.shape[1]
            _vols_mL = np.array([
                _vols_uL[i + 1] / 1000.0 if (i + 1) < len(_vols_uL) else 0.002
                for i in range(n_all)
            ])

            # Compute complex concentration after the dummy injection
            # using the current logK values (recalculated each optimizer step)
            _C_dummy = np.zeros(n_rxn)
            try:
                _dV_dummy = float(_vols_uL[0]) / 1000.0
                _G0       = float(list(params["conc0"].values())[0])
                _H_stock  = float(list(params["titrant_mMs"].values())[0])
                _G_d      = _G0 * _V0 / (_V0 + _dV_dummy)
                _H_d      = _H_stock * _dV_dummy / (_V0 + _dV_dummy)
                _totals_d = {sp: 0.0 for sp in network["all_species"]}
                _totals_d[params.get("primary_component",
                          list(params["conc0"].keys())[0])] = _G_d * 1e-3
                _totals_d[params["titrant_free_names"][0]]  = _H_d * 1e-3
                _c_d, _, _ = solve_equilibria_general(
                    _totals_d, network["equilibria"],
                    network["all_species"], cur_logKs_ or logK_vals)
                for _r, _sp in enumerate(rxn_sp_ or []):
                    _C_dummy[_r] = float(_c_d.get(_sp, 0.0)) * 1e3  # M→mM
            except Exception:
                pass  # fallback: _C_dummy stays zeros

            dC_all = _delta_C(C_all, _vols_mL, _V0, C_dummy=_C_dummy)
            if not excl_set:
                return dC_all
            incl = np.array([i not in excl_set for i in range(len(C_all))])
            return dC_all[incl]

        def _solve_dH(A_mat, q_obs):
            """Solve A @ x ≈ q_obs for [ΔH..., q_offset] by lstsq.

            Columns whose ΔH the user pinned in the script are not solved for:
            their contribution is subtracted from q_obs and the column is
            dropped, then the pinned values are reinserted so the caller
            always receives a full-length vector.
            """
            if not _fixed_dH:
                x, _, _, _ = np.linalg.lstsq(A_mat, q_obs, rcond=None)
                return x
            ncol = A_mat.shape[1]
            fx = sorted(i for i in _fixed_dH if i < ncol)
            fr = [i for i in range(ncol) if i not in fx]
            vals = np.array([_fixed_dH[i] for i in fx], dtype=float)
            q_adj = np.asarray(q_obs, dtype=float) - A_mat[:, fx] @ vals
            if not fr:
                return np.array([_fixed_dH.get(i, 0.0) for i in range(ncol)])
            x_free, _, _, _ = np.linalg.lstsq(A_mat[:, fr], q_adj, rcond=None)
            x = np.empty(ncol, dtype=float)
            x[fx] = vals
            x[fr] = x_free
            return x

        _pscale = PenaltyScale()   # fixed reference for the constraint penalty (see equilibrist_parser)
        _trace = FitTrace()   # how did the fit end? (see equilibrist_parser)
        def objective(fp):
            _trace(fp)
            try:
                logKs, concs, tits = _unpack(fp)
                cur_logKs = logK_vals.copy()
                cur_logKs.update(logKs)
                cur_params = _make_params(concs, tits)
                cur_parsed = _make_parsed(concs, tits)

                # Compute C for ALL injections, then dC for all, then mask.
                # This ensures dC[i] = C[i] - C[i-1] is correct even when
                # early injections are excluded.
                x_vals_all = _itc_injection_x_vals(raw, cur_parsed, cur_params,
                                                    x_expr, network)
                proxy_sp = [sp for _, sp in rxn_valid]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    C_all = _compute_complex_concs_at_injections(
                        x_vals_all, cur_parsed, network, cur_logKs, cur_params, proxy_sp)

                dC_masked = _compute_dC_masked(C_all, cur_logKs_=cur_logKs, rxn_sp_=proxy_sp)
                if dC_masked.shape[0] != n_inj:
                    return 1e9
                A = _build_A_matrix_from_dC(dC_masked)
                if q_offset_fixed is not None:
                    q_adj = q_exp - q_offset_fixed
                    dH_vec = _solve_dH(A, q_adj)
                    q_theo = A @ dH_vec + q_offset_fixed
                else:
                    params_vec = _solve_dH(A, q_exp)
                    q_theo = A @ params_vec

                ssr = float(np.sum((q_exp - q_theo) ** 2))
                if not np.isfinite(ssr):
                    return 1e9

                cp = constraints_penalty(constraints, cur_logKs, ssr_scale=_pscale(ssr))
                _live.notify(cur_logKs, ssr + cp)
                return ssr + cp
            except Exception:
                return 1e9


        # ── Objective handed out for a GLOBAL fit (equilibrist_global) ───────
        if _objective_only:
        
            return {
                "objective": (lambda t: float(objective(np.asarray(t, dtype=float)))),
                "data_ssr":  None,
                "x0": np.asarray(x0, dtype=float),
                "names": list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys),
                "n_log_params": len(fit_keys),
                "n_data": int(len(q_exp)),
            }

        # ── Warm-start Phase 0: warm-start K before joint optimisation ─
        _global_start = time.time()

        if n_conc > 0 and n_logK > 0:
            class _TimeoutP0(Exception): pass
            def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s):
                if time.time() - _t0 > _tlim * 0.4:
                    raise _TimeoutP0()
                return objective(np.concatenate([lk, x0[n_logK:]]))
            _sp0 = np.vstack([x0[:n_logK]] + [x0[:n_logK] + np.eye(n_logK)[i] * 1.5
                              for i in range(n_logK)])
            try:
                _r0 = tied_minimize(_ph0_obj, x0[:n_logK], names=list(fit_keys), constraints=constraints or [], fixed=logK_vals,
                               options={"maxiter": maxiter // 10,
                                        "xatol": tolerance,
                                        "fatol": tolerance * 1e-4,
                                        "adaptive": True,
                                        "initial_simplex": _sp0})
                x0 = np.concatenate([_r0.x, x0[n_logK:]])
            except (_TimeoutP0, Exception):
                pass

        # ── Main optimisation ──────────────────────────────────────
        # One simplex step per fitted parameter: log K's, then cell
        # concentrations, then titrant stocks. Omitting the titrant block
        # left _steps shorter than n_p and indexed off the end.
        _steps = np.concatenate([
            np.full(n_logK, 1.5),
            np.array([bds_conc[i][1] - bds_conc[i][0] for i in range(n_conc)]),
            np.array([bds_tit[i][1] - bds_tit[i][0] for i in range(n_tit)]),
        ]) if n_p > 0 else np.array([1.5])

        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _steps[i]
                                         for i in range(n_p)])

        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "nit": 0}

        def _obj_timed(fp, _t=_bt, _t0=_global_start, _tlim=timeout_s):
            _t["nit"] += 1
            f = objective(fp)
            if f < _t["f"]:
                _t["f"] = f
                _t["x"] = fp.copy()
            if time.time() - _t0 > _tlim:
                raise _Timeout()
            return f

        timed_out = False
        try:
            result = tied_minimize(_obj_timed, x0, names=list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys), constraints=constraints or [], fixed=logK_vals,
                              options={"maxiter": maxiter,
                                       "xatol": tolerance,
                                       "fatol": tolerance * 1e-4,
                                       "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult()
            timed_out = True

        best_x = _bt["x"].copy()

        # ── Final statistics ───────────────────────────────────────
        fitted_logKs, fitted_concs, fitted_tits = _unpack(best_x)
        fin_logKs  = logK_vals.copy(); fin_logKs.update(fitted_logKs)
        fin_params = _make_params(fitted_concs, fitted_tits)
        fin_parsed = _make_parsed(fitted_concs, fitted_tits)

        x_inj_all = _itc_injection_x_vals(raw, fin_parsed, fin_params, x_expr, network)
        proxy_sp  = [sp for _, sp in rxn_valid]
        C_all_f   = _compute_complex_concs_at_injections(
            x_inj_all, fin_parsed, network, fin_logKs, fin_params, proxy_sp)
        # Compute full unmasked dC (all n_all injections) with correct dummy state
        _vols_uL_f = raw.get("inj_vols_uL", [])
        _V0_f      = float(params["V0_mL"])
        _vols_mL_f = np.array([
            _vols_uL_f[i + 1] / 1000.0 if (i + 1) < len(_vols_uL_f) else 0.002
            for i in range(len(C_all_f))
        ])
        _C_dummy_f = np.zeros(C_all_f.shape[1])
        try:
            _dV_d_f  = float(_vols_uL_f[0]) / 1000.0
            _G0_f    = float(list(fin_params["conc0"].values())[0])
            _Hs_f    = float(list(fin_params["titrant_mMs"].values())[0])
            _Gd      = _G0_f * _V0_f / (_V0_f + _dV_d_f)
            _Hd      = _Hs_f * _dV_d_f / (_V0_f + _dV_d_f)
            _td      = {sp: 0.0 for sp in network["all_species"]}
            _td[fin_params.get("primary_component", list(fin_params["conc0"].keys())[0])] = _Gd * 1e-3
            _td[fin_params["titrant_free_names"][0]] = _Hd * 1e-3
            _cd, _, _ = solve_equilibria_general(_td, network["equilibria"],
                                                  network["all_species"], fin_logKs)
            for _r, _sp in enumerate(proxy_sp):
                _C_dummy_f[_r] = float(_cd.get(_sp, 0.0)) * 1e3
        except Exception:
            pass
        dC_all_f  = _delta_C(C_all_f, _vols_mL_f, _V0_f, C_dummy=_C_dummy_f)
        V0_f      = _V0_f
        # Same extent transform as the objective's A matrix. This branch
        # rebuilds A independently for the FINAL dH solve, so without the
        # matrix here the reported dH silently reverted to cumulative even
        # though the optimisation itself used stepwise extents.
        A_all_f   = (dC_all_f @ _rxn_usage_M) * V0_f * 1000.0
        if q_offset_fixed is None:
            A_all_f = np.column_stack([A_all_f, np.ones(len(A_all_f))])
        incl_f  = np.array([i not in excl_set for i in range(len(x_inj_all))])
        A_final = A_all_f[incl_f]
        x_inj   = x_inj_all[incl_f]
        C_final = C_all_f[incl_f]
        if q_offset_fixed is not None:
            params_final = _solve_dH(A_final, q_exp - q_offset_fixed)
            q_offset_val = float(q_offset_fixed)
            dH_vec = params_final
        else:
            params_final = _solve_dH(A_final, q_exp)
            dH_vec = params_final[:n_rxn]
            q_offset_val = float(params_final[n_rxn]) if len(params_final) > n_rxn else 0.0
        q_theo = A_final @ params_final
        if q_offset_fixed is not None:
            q_theo = q_theo + q_offset_fixed

        residuals = q_exp - q_theo
        ssr  = float(np.sum(residuals ** 2))
        sst  = float(np.sum((q_exp - q_exp.mean()) ** 2)) if len(q_exp) > 1 else 1e-12
        r2   = float(1.0 - ssr / max(sst, 1e-30))
        rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

        # ΔH values keyed by reaction kname
        dH_values = {}
        dH_errors = {}
        _rxn_eqs = parsed.get("equilibria", [])
        for (rxn_idx, sp), dh in zip(rxn_valid, dH_vec):
            if rxn_idx < len(_rxn_eqs):
                kn = _rxn_eqs[rxn_idx]["kname"]
            else:
                kn = f"rxn{rxn_idx}"
            dH_values[kn] = float(dh)

        # Hessian-based uncertainties
        _cov_full, _cov_names = None, []
        try:
            n_pts = len(q_exp)
            n_par = n_p + n_rxn + (0 if q_offset_fixed is not None else 1)
            if n_pts > n_par:
                eps = 1e-4

                def _residuals_vec(fp):
                    lk, cn, tt = _unpack(fp)
                    cl = logK_vals.copy(); cl.update(lk)
                    cp2 = _make_params(cn, tt)
                    cp3 = _make_parsed(cn, tt)
                    x2 = _itc_injection_x_vals(raw, cp3, cp2, x_expr, network)
                    C2 = _compute_complex_concs_at_injections(
                        x2, cp3, network, cl, cp2, proxy_sp)
                    dC2 = _compute_dC_masked(C2, cur_logKs_=cl, rxn_sp_=proxy_sp)
                    A2 = _build_A_matrix_from_dC(dC2)
                    if q_offset_fixed is not None:
                        p2 = _solve_dH(A2, q_exp - q_offset_fixed)
                        return q_exp - (A2 @ p2 + q_offset_fixed)
                    p2 = _solve_dH(A2, q_exp)
                    return q_exp - A2 @ p2

                r0 = _residuals_vec(best_x)
                J = np.zeros((n_pts, n_p))
                for k in range(n_p):
                    dx = np.zeros(n_p); dx[k] = eps
                    J[:, k] = (_residuals_vec(best_x + dx) - r0) / eps
                # Degrees of freedom must count EVERY estimated quantity.
                # dH (one per reaction) and q_offset are solved by lstsq
                # inside the objective, but they are still fitted from
                # the data and consume df. Using n_p alone understated
                # sigma^2 and therefore every parameter error.
                _n_lin_est = n_rxn_free + (0 if q_offset_fixed is not None else 1)
                _n_est_tot = n_p + _n_lin_est
                sigma2 = float(np.dot(r0, r0)) / max(n_pts - _n_est_tot, 1)
                JtJ = J.T @ J
                try:
                    cov = np.linalg.inv(JtJ) * sigma2
                except np.linalg.LinAlgError:
                    d = np.diag(JtJ)
                    cov = np.diag(np.where(d > 0, sigma2 / d, 0.0))

                param_errs = {fit_keys[k]: float(np.sqrt(max(cov[k, k], 0.0)))
                              for k in range(n_logK)}
                for k in range(n_conc):
                    param_errs[fit_conc_keys[k]] = float(
                        np.sqrt(max(cov[n_logK + k, n_logK + k], 0.0)))

                # ── Joint covariance over ALL fitted parameters ──────────
                # The block above gives errors for the nonlinear parameters
                # only (log K's and fitted concentrations), because ΔH and
                # q_offset are eliminated analytically by lstsq at every
                # objective evaluation (variable projection).
                #
                # For the diagnostics layer we want the FULL joint
                # covariance, so that the correlation heatmap and the
                # t-tests see the K <-> concentration <-> ΔH <-> q_offset
                # trade-offs that dominate ITC. Build the complete Jacobian
                # by holding the linear parameters fixed and differentiating
                # q_theo directly; the linear columns are exact and analytic
                # (they are the columns of A itself).
                try:
                    def _qtheo_vec(fp):
                        lk, cn, tt = _unpack(fp)
                        cl = logK_vals.copy(); cl.update(lk)
                        cp2 = _make_params(cn, tt)
                        cp3 = _make_parsed(cn, tt)
                        x2 = _itc_injection_x_vals(raw, cp3, cp2, x_expr, network)
                        C2 = _compute_complex_concs_at_injections(
                            x2, cp3, network, cl, cp2, proxy_sp)
                        dC2 = _compute_dC_masked(C2, cur_logKs_=cl, rxn_sp_=proxy_sp)
                        A2 = _build_A_matrix_from_dC(dC2)
                        return A2 @ params_final

                    _q0 = _qtheo_vec(best_x)
                    _Jnl = np.zeros((n_pts, n_p))
                    for k in range(n_p):
                        dx = np.zeros(n_p); dx[k] = eps
                        _Jnl[:, k] = (_qtheo_vec(best_x + dx) - _q0) / eps
                    # Linear columns: exact derivatives of A @ [dH; q_offset]
                    _Jlin = np.asarray(A_final, dtype=float)
                    if q_offset_fixed is not None and _Jlin.shape[1] > n_rxn:
                        _Jlin = _Jlin[:, :n_rxn]      # q_offset not a free parameter
                    _Jfull = np.hstack([_Jnl, _Jlin])
                    _npar_full = _Jfull.shape[1]
                    if n_pts > _npar_full:
                        _s2 = ssr / float(n_pts - _npar_full)
                        _JtJ = _Jfull.T @ _Jfull
                        _cov_full = np.linalg.inv(_JtJ) * _s2
                        _cov_names = list(fit_keys[:n_logK]) + \
                                     list(fit_conc_keys[:n_conc])
                        for (rxn_idx, _sp) in rxn_valid:
                            _kn = (_rxn_eqs[rxn_idx]["kname"]
                                   if rxn_idx < len(_rxn_eqs) else f"rxn{rxn_idx}")
                            _cov_names.append(f"ΔH({_kn})")
                        if q_offset_fixed is None and _Jlin.shape[1] > n_rxn:
                            _cov_names.append("q_offset")
                        if len(_cov_names) != _npar_full:
                            _cov_full, _cov_names = None, []
                except Exception:
                    _cov_full, _cov_names = None, []

                # ΔH errors from covariance of lstsq solution
                # Var(ΔH) = σ² (AᵀA)⁻¹ — using residual sigma
                sigma2_data = ssr / max(n_pts - n_par, 1)
                AtA = A_final.T @ A_final
                try:
                    cov_dH = np.linalg.inv(AtA) * sigma2_data
                    for j, (rxn_idx, _) in enumerate(rxn_valid):
                        if rxn_idx < len(_rxn_eqs):
                            kn = _rxn_eqs[rxn_idx]["kname"]
                        else:
                            kn = f"rxn{rxn_idx}"
                        dH_errors[kn] = float(np.sqrt(max(cov_dH[j, j], 0.0)))
                except np.linalg.LinAlgError:
                    pass
            else:
                param_errs = {}
        except Exception:
            param_errs = {}

        param_values = {k: float(best_x[i]) for i, k in enumerate(fit_keys)}

        _conv = bool(r2 >= 0.99 or (not timed_out and
                     getattr(result, "success", False)))

        stats = {
            "r_squared":    r2,
            "rmse":         rmse,
            "ssr":          ssr,
            "n_points":     len(q_exp),
            # Total estimated quantities, so AIC/BIC/AICc count the
            # lstsq-solved dH and q_offset as the parameters they are.
            "n_params":     n_p + n_rxn_free + (0 if q_offset_fixed is not None else 1),
            "n_params_nonlinear": n_p,
            "n_iter":       getattr(result, "nit", _bt["nit"]),
            "param_values": param_values,
            # v2 diagnostics contract: the unified residual collector reads
            # flat y_obs / y_calc / residuals from stats. Without these the
            # residual-diagnostics panel (residuals vs predictor, Q-Q,
            # Durbin-Watson, Shapiro, AIC/BIC) renders nothing for ITC.
            "y_obs":        np.asarray(q_exp, dtype=float),
            "y_calc":       np.asarray(q_theo, dtype=float),
            "residuals":    np.asarray(q_exp, dtype=float) - np.asarray(q_theo, dtype=float),
            "per_col_y_obs":  {"q [µcal]": np.asarray(q_exp, dtype=float)},
            "per_col_y_calc": {"q [µcal]": np.asarray(q_theo, dtype=float)},
            "per_col_x":      {"q [µcal]": np.asarray(x_inj, dtype=float)},
            "param_errors": param_errs,
            "param_cov":       _cov_full,
            "param_cov_names": _cov_names,
            "dH_values":    dH_values,
            "dH_errors":    dH_errors,
            "q_offset":     q_offset_val,
            "q_theo":       q_theo,
            "x_inj":        x_inj,
            "C_back":       C_final,
            "dH_per_inj":   _compute_dH_per_inj(q_exp, A_final, dH_vec,
                                                  q_offset_val, C_final,
                                                  float(params["V0_mL"])),
            "dH_norm_per_inj": _compute_dH_norm(q_exp, raw),
            "dH_theo_norm":    _pad_to_full(_compute_dH_norm(q_theo, raw), incl_f),
            "x_inj_all":       x_inj_all,   # ALL real injections (length = n_real)
            "rxn_proxy_sp": proxy_sp,
            "fitted_concs": fitted_concs,
            "fitted_titrants": dict(fitted_tits),
            "timed_out":    timed_out,
            "fit_mode":     "itc",
        }

        msg = "ITC fit complete" if _conv else \
              ("Fit timed out — showing best result" if timed_out
               else "Fit converged with warnings")
        stats.update(_trace.summary(result, timed_out, tolerance))
        return _conv, {k: float(best_x[i]) for i, k in enumerate(fit_keys)}, stats, msg

    except Exception as e:
        import traceback
        return False, {}, {}, f"ITC fitting error: {e}\n{traceback.format_exc()}"
