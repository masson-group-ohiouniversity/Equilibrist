"""Synthetic inputs for Tutorial 21 (A, B, C): one chemistry, three techniques,
meant for the cross-tab global fit.

    A + B = C + D    log K1 = 1.30   (K1 = 20, dimensionless exchange constant)
    2 A   = E        log K2 = 2.50   (K2 = 316 M-1, dimerisation of A)

A is titrated with B in every experiment (x = B0/A0, up to 3 equivalents).
Reading of the letters: A = host.X complex, B = guest Y, C = host.Y, D = free X,
E = (host.X)2 dimer.

  21A  slow-exchange 1H NMR, 2.00 mM A, 500 uL, 20 mM B.  Integrals of two
       signals of A (3H, 1H) and two of C (3H, 1H) against an internal standard
       (Std, 6H, 1.00 mM).  The standard is needed: the network cannot tell that
       C carries one unit of A (two-product reaction), so the usual A mass
       balance cannot turn relative integrals into concentrations.
  21B  fast-exchange 1H NMR, 1.00 mM A, 500 uL, 10 mM B.  Five signals:
       Host (A, C, 2E) x2, Xg (A, 2E, D) x2, Yg (B, C) x1.
  21C  UV-Vis, 0.050 mM A, 2.00 mL, 1.00 mM B, 1 cm, 300-500 nm; D transparent.
       At 50 uM the dimer is ~3 % at most, so this tab alone says little about K2.

Run from the project folder:  python3 gen_tutorial21.py [out_dir] [noise_scale] [seed]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from openpyxl import Workbook
from equilibrist_parser import parse_script
from equilibrist_network import build_network, solve_equilibria_general

LOGK = {"K1": 1.30, "K2": 2.50}
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "TUTORIALS")
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     # 0 -> noise-free data (forward-model checks)
RNG = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)

HEAD = """$concentrations
A0 = {A0} mM

$volumes
V0 = {V0}

$titrant
Bt = {Bt} mM

$equilibria
A + B = C + D; log K1 = 1.0
2 A = E; log K2 = 2.0

$plot
x = B0/A0
xmax = 3.0
y = A, B, C, D, E
"""

SCRIPTS = {
 "21A": HEAD.format(A0="2.00", V0="500 uL", Bt="20.0") + """
$nmr
integration: 6, 3, 1, 3, 1
standard: 1.00 mM
""",
 "21B": HEAD.format(A0="1.00", V0="500 uL", Bt="10.0") + """
$variables
Host = A + C + 2*E
Xg = A + 2*E + D
Yg = B + C

$nmr
shift: Host, Xg, Yg
""",
 "21C": HEAD.format(A0="0.050", V0="2.00 mL", Bt="1.00") + """
$spectra
transparent: D
""",
}

def concentrations(A0_mM, V0_mL, Bt_mM, V_mL):
    """Species concentrations (mM) at each added volume, from Equilibrist's own solver."""
    p = parse_script(SCRIPTS["21A"]); net = build_network(p)
    rows = []
    for V in V_mL:
        Vt = V0_mL + V
        tot = {s: 0.0 for s in net["all_species"]}
        tot["A"] = A0_mM * V0_mL / Vt * 1e-3
        tot["B"] = Bt_mM * V / Vt * 1e-3
        c, *_ = solve_equilibria_general(tot, net["equilibria"], net["all_species"], LOGK)
        rows.append([c[s] * 1e3 for s in "ABCDE"])
    return np.array(rows)          # columns A, B, C, D, E in mM

def write_xlsx(path, sheet, header, rows):
    wb = Workbook(); ws = wb.active; ws.title = sheet
    ws.append(header)
    for r in rows:
        ws.append([None if (isinstance(v, float) and not np.isfinite(v)) else v for v in r])
    wb.save(path)

# ── 21A: slow exchange, integrals vs internal standard ─────────────────────
V = np.round(np.arange(0, 0.1501, 0.01), 4)                       # 16 spectra
C = concentrations(2.00, 0.500, 20.0, V)
A_, C_ = C[:, 0], C[:, 2]
gain = RNG.uniform(0.8, 1.2, len(V))                               # arbitrary per-spectrum scale
def integ(nH, conc):
    I = nH * conc * gain * (1 + NOISE * RNG.normal(0, 0.015, len(V))) + NOISE * RNG.normal(0, 0.01, len(V)) * gain
    return np.clip(I, 0.0, None)
rows = np.column_stack([V, integ(6, np.full(len(V), 1.00)), integ(3, A_), integ(1, A_), integ(3, C_), integ(1, C_)])
write_xlsx(os.path.join(OUT, "tutorial_21A_input.xlsx"), "Slow_exchange_integrals",
           ["V", "Std", "A", "A", "C", "C"], rows.tolist())

# ── 21B: fast exchange, five signals ───────────────────────────────────────
V = np.round(np.arange(0, 0.1501, 0.005), 4)                      # 31 spectra
C = concentrations(1.00, 0.500, 10.0, V)
sp = dict(zip("ABCDE", C.T))
PURE = {   # column: [(coeff, species, delta_pure)]
 "Host.1": [(1, "A", 7.80), (1, "C", 7.52), (2, "E", 8.05)],
 "Host.2": [(1, "A", 6.95), (1, "C", 7.21), (2, "E", 6.62)],
 "Xg.1":   [(1, "A", 1.22), (2, "E", 0.98), (1, "D", 2.12)],
 "Xg.2":   [(1, "A", 3.41), (2, "E", 3.18), (1, "D", 3.79)],
 "Yg":     [(1, "B", 2.65), (1, "C", 1.78)],
}
cols = []
for name, terms in PURE.items():
    num = sum(n * sp[s] * d for n, s, d in terms); den = sum(n * sp[s] for n, s, d in terms)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = np.where(den > 1e-9, num / np.maximum(den, 1e-30), np.nan)
    obs = obs + NOISE * RNG.normal(0, 0.002, len(V))
    cols.append(obs)
rows = np.column_stack([V] + cols)
write_xlsx(os.path.join(OUT, "tutorial_21B_input.xlsx"), "Fast_exchange_shifts",
           ["V", "Host", "Host", "Xg", "Xg", "Yg"], rows.tolist())

# ── 21C: UV-Vis, D transparent ─────────────────────────────────────────────
V = np.round(np.arange(0, 0.3001, 0.0125), 4)                     # 25 spectra
C = concentrations(0.050, 2.00, 1.00, V)
wl = np.arange(300, 501, 2.0)
g = lambda a, mu, s: a * np.exp(-0.5 * ((wl - mu) / s) ** 2)
EPS = {"A": g(12, 380, 28) + g(3, 320, 20), "B": g(5, 330, 22),
       "C": g(9, 425, 32) + g(2, 340, 20), "D": 0 * wl, "E": g(22, 365, 25)}   # mM-1 cm-1
Abs = sum(np.outer(C[:, i], EPS[s]) for i, s in enumerate("ABCDE")) + NOISE * RNG.normal(0, 0.002, (len(V), len(wl)))
rows = [[float(V[i])] + Abs[i].tolist() for i in range(len(V))]
write_xlsx(os.path.join(OUT, "tutorial_21C_input.xlsx"), "Spectra", ["V"] + wl.tolist(), rows)

for k, s in SCRIPTS.items():
    open(os.path.join(OUT, f"tutorial_{k}.txt"), "w").write(s.lstrip("\n"))
print("wrote tutorial_21A/B/C .txt and _input.xlsx to", OUT)
print("truth: log K1 = 1.30, log K2 = 2.50 (scripts start at 1.0 / 2.0)")
