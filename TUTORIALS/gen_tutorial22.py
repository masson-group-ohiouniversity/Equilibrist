"""Tutorial 22 (A/B/C): synthetic slow-exchange NMR kinetics for a three-tab global fit.

    A + B <> C     k1 (M^-1 s^-1), k-1 (s^-1)
    2 C  > D       k2 (M^-1 s^-1)

1.00 mM A in every tab; B0 = 0.50 / 1.00 / 2.00 mM in 22A / 22B / 22C.  Four
integrated signals per spectrum (A, B, C, D), all in slow exchange.  The
concentrations come from Equilibrist's own ODE integrator (compute_kinetics_curve,
Radau, rtol 1e-8), interpolated onto the spectrum times.

Usage:  python3 gen_tutorial22.py [out_dir] [noise_scale] [seed]
        defaults: TUTORIALS/  1.0  1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from openpyxl import Workbook
from equilibrist_parser import parse_script
from equilibrist_kinetics import compute_kinetics_curve

OUT   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "TUTORIALS")
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(SEED)

# ── truth (what the data are made with) and the scripts' starting values ──
TRUE  = {"k1": 5.0, "k-1": 2.0e-3, "k2": 0.30}      # M^-1 s^-1, s^-1, M^-1 s^-1
START = {"k1": 2.0, "k-1": 1.0e-3, "k2": 0.10}
A0_mM = 1.00
B0 = {"22A": 0.50, "22B": 1.00, "22C": 2.00}
T_MAX = 7200.0
N_H   = {"A": 3, "B": 2, "C": 3, "D": 6}             # protons behind each integrated signal
COLS  = ["A", "B", "C", "D"]

def script(tag, k):
    return f"""$concentrations
A0 = {A0_mM:.2f} mM
B0 = {B0[tag]:.2f} mM

$volumes
V0 = 0.5 mL

$reactions
A + B <> C; k1 = {k['k1']:g}; k-1 = {k['k-1']:g}
2 C > D; k2 = {k['k2']:g}

$plot
xmax = {T_MAX:g}
y = A, B, C, D

$nmr
integration: {', '.join(str(N_H[c]) for c in COLS)}
"""

# spectrum times: every 15 s for the first 5 min, then every 2 min to 2 h
T_EXP = np.concatenate([np.arange(0.0, 300.0 + 1e-9, 15.0), np.arange(360.0, T_MAX + 1e-9, 120.0)])

def write_xlsx(path, sheet, header, rows):
    wb = Workbook(); ws = wb.active; ws.title = sheet
    ws.append(list(header))
    for r in rows:
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in r])
    wb.save(path)

def concentrations(tag):
    """Species concentrations (mM) at T_EXP from the truth, via Equilibrist's integrator."""
    p = parse_script(script(tag, TRUE))
    logk = {"k1": np.log10(TRUE["k1"]), "k-1": np.log10(TRUE["k-1"]), "k2": np.log10(TRUE["k2"])}
    curve = compute_kinetics_curve(p, logk, T_MAX, 72001)          # 0.1 s grid
    tkey = next((k for k in ("t", "time", "x") if k in curve), None)
    if tkey is None:
        raise RuntimeError(f"no time key in curve: {list(curve)[:8]}")
    t = np.asarray(curve[tkey], float)
    return {sp: np.interp(T_EXP, t, np.asarray(curve[sp], float)) for sp in COLS}

os.makedirs(OUT, exist_ok=True)
for tag in ("22A", "22B", "22C"):
    conc = concentrations(tag)
    rows = []
    for i, t in enumerate(T_EXP):
        gain = rng.uniform(0.8, 1.2)                       # per-spectrum receiver gain, cancels in ratios
        row = [float(t)]
        for sp in COLS:
            I = gain * N_H[sp] * conc[sp][i]
            I = I * (1.0 + NOISE * 0.02 * rng.standard_normal()) + NOISE * 0.01 * gain * rng.standard_normal()
            row.append(float(max(I, 0.0)))
        rows.append(row)
    with open(os.path.join(OUT, f"tutorial_{tag}.txt"), "w") as fh:
        fh.write(script(tag, START))
    write_xlsx(os.path.join(OUT, f"tutorial_{tag}_input.xlsx"), "data", ["t (s)"] + COLS, rows)
    c_end = {sp: conc[sp][-1] for sp in COLS}
    print(f"{tag}: B0 {B0[tag]:.2f} mM, {len(rows)} spectra, at {T_MAX:.0f} s: " + ", ".join(f"{sp} {v:.3f}" for sp, v in c_end.items()) + " mM")
print("truth:", TRUE, " scripts start at:", START)
