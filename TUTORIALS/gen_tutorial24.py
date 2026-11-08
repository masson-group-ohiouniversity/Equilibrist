"""Tutorial 24: an oscillating reaction (the Brusselator) in Equilibrist's kinetics mode.

    A > X              k1
    B + X > Y + D      k2
    2 X + Y > 3 X      k3   (autocatalysis)
    X > E              k4

A and B are in large excess and act as reservoirs, so the closed batch sustains a limit
cycle (relaxation oscillations) for many periods.  X and Y are "measured" every DT s.

Usage:  python3 gen_tutorial24.py [out_dir] [noise_scale] [seed]
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

TRUE  = {"k1": 1.0e-3, "k2": 1.0, "k3": 1.0e6, "k4": 1.0}      # s^-1, M^-1 s^-1, M^-2 s^-1, s^-1
START = {"k1": 1.1e-3, "k2": 0.9, "k3": 1.1e6, "k4": 1.1}        # within 10 %: see the tutorial header
A0, B0, X0, Y0 = 1000.0, 3000.0, 1.0, 1.0                     # mM
T_MAX, DT = 60.0, 0.2                                         # s

def script(k):
    return f"""# Tutorial 24: an oscillating reaction -- the Brusselator (Prigogine & Lefever 1968)
# A and B are reservoirs (large excess); X and Y oscillate on a limit cycle,
# period about 7 s, while A and B are slowly consumed.
# File: tutorial_24_input.xlsx (t in s, X and Y in mM, one point every {DT:g} s).
# Fitting an oscillator: the objective is rugged in the rate constants (a small
# change of period shifts every later peak), so the starting values must already
# oscillate with roughly the right period. From within about 10 % of the solution
# the fit converges; from 20-50 % away Nelder-Mead collapses to a flat line
# through the mean of the data.
# Adjust the constants until the simulated period matches the data before fitting.

$concentrations
A0 = {A0:g} mM
B0 = {B0:g} mM
X0 = {X0:g} mM
Y0 = {Y0:g} mM

$volumes
V0 = 1 mL

$reactions
A > X; k1 = {k['k1']:g}
B + X > Y + D; k2 = {k['k2']:g}
2 X + Y > 3 X; k3 = {k['k3']:g}
X > E; k4 = {k['k4']:g}

$plot
xmax = {T_MAX:g}
y = X, Y
"""

p = parse_script(script(TRUE))
logk = {k: np.log10(v) for k, v in TRUE.items()}
curve = compute_kinetics_curve(p, logk, T_MAX, int(T_MAX / 0.005) + 1)       # 5 ms grid
t_exp = np.arange(0.0, T_MAX + 1e-9, DT)
X = np.interp(t_exp, curve["t"], curve["X"]); Y = np.interp(t_exp, curve["t"], curve["Y"])
Xn = X * (1 + NOISE * 0.02 * rng.standard_normal(len(X))) + NOISE * 0.03 * rng.standard_normal(len(X))
Yn = Y * (1 + NOISE * 0.02 * rng.standard_normal(len(Y))) + NOISE * 0.03 * rng.standard_normal(len(Y))
Xn, Yn = np.clip(Xn, 0, None), np.clip(Yn, 0, None)

os.makedirs(OUT, exist_ok=True)
wb = Workbook(); ws = wb.active; ws.title = "data"; ws.append(["t (s)", "X", "Y"])
for i in range(len(t_exp)):
    ws.append([float(t_exp[i]), float(Xn[i]), float(Yn[i])])
wb.save(os.path.join(OUT, "tutorial_24_input.xlsx"))
open(os.path.join(OUT, "tutorial_24.txt"), "w").write(script(START))
from scipy.signal import find_peaks
pk, _ = find_peaks(X, prominence=0.3)
print(f"wrote {OUT}/tutorial_24_input.xlsx: {len(t_exp)} points, X {X.min():.2f}..{X.max():.2f} mM, {len(pk)} X peaks, period {np.diff(t_exp[pk]).mean():.2f} s; A left {curve['A'][-1]:.0f} mM, B left {curve['B'][-1]:.0f} mM")
