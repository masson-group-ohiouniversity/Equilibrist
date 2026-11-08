"""Tutorial 23: synthetic MicroCal iTC200 thermogram for a three-step network

    G   + H = GH      log K1   dH1 (exothermic)
    GH  + H = GH2     log K2   dH2 (ENDOTHERMIC)
    GH2 + G = G2H2    log K3   dH3 (exothermic)

Host G in the cell, guest H in the syringe.  Heats are generated with Equilibrist's
own forward model (overflow concentrations, reaction EXTENTS from the usage matrix,
stepwise enthalpies), then turned into a power trace with realistic peak shapes,
baseline drift and noise, and written in the MicroCal .itc layout.

Usage:  python3 gen_tutorial23.py [out_dir] [noise_scale] [seed]
        defaults: TUTORIALS/  1.0  1      (noise_scale 0 -> noise-free trace)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from equilibrist_parser import parse_script
from equilibrist_network import build_network, solve_equilibria_general
from equilibrist_itc import (_itc_overflow_concentrations, _delta_C,
                             _rxn_product_species, _reaction_usage_matrix)

OUT   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "TUTORIALS")
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
RNG = np.random.default_rng(SEED)
_trap = getattr(np, "trapezoid", None) or np.trapz      # numpy < 2 has only trapz

# ── truth and starting values ─────────────────────────────────────────────
LOGK = {"K1": 5.5, "K2": 4.3, "K3": 4.8}            # M^-1 (log10)
DH   = {"K1": -10.0, "K2": +3.0, "K3": -6.0}       # kcal/mol, STEPWISE
START = {"K1": 5.0, "K2": 4.0, "K3": 4.0}

G0_mM, V0_mL, H_STOCK_mM, TEMP_C = 0.30, 0.2039, 5.0, 25.0
N_INJ, V_INJ_uL, V_DUMMY_uL = 28, 1.2, 0.4
SPACING_S, DT_S = 180.0, 5.0
BASE_POWER, TAU_D, TAU_R = 9.8471, 12.0, 3.0        # ucal/s; decay and rise times -- fast enough that the
                                                    # auto-integrator's EqTime + 40 s window loses < 0.2 % of a peak
NOISE_SD, DRIFT_TOTAL = 0.0035, 0.045               # ucal/s

def script(k):
    return f"""# ITC experiment: host G (cell) + guest H (syringe), iTC200, T = 25 C
# File: tutorial_23_input.itc
# Three-step network with one ENDOTHERMIC step (GH + H = GH2).
#
# Cell:    G = {G0_mM} mM
# Syringe: H = {H_STOCK_mM} mM
# Cell volume: {V0_mL} mL (from the .itc header)
# {N_INJ} injections of {V_INJ_uL} uL after a {V_DUMMY_uL} uL dummy, 180 s apart.

$concentrations
G0 = {G0_mM} mM

$volumes
V0 = {V0_mL} mL

$titrant
Ht = {H_STOCK_mM} mM

$temperature
T = {TEMP_C:.0f} C

$reactions
G + H = GH ; log K1 = {k['K1']}
GH + H = GH2 ; log K2 = {k['K2']}
GH2 + G = G2H2 ; log K3 = {k['K3']}

$plot
x = H0/G0
xmax = 3.5
y = G, H, GH, GH2, G2H2
"""

parsed = parse_script(script(LOGK))
net = build_network(parsed)
rxn_sp = [s for s in _rxn_product_species(parsed) if s is not None]          # ['GH', 'GH2', 'G2H2']
M = _reaction_usage_matrix(parsed, list(enumerate(rxn_sp)))                  # extents = C @ M
raw_stub = {"inj_vols_uL": [V_DUMMY_uL] + [V_INJ_uL] * N_INJ}
params = {"conc0": {"G": G0_mM}, "V0_mL": V0_mL, "titrant_free_names": ["H"],
          "titrant_mMs": {"H": H_STOCK_mM}, "primary_component": "G"}
G_tot, H_tot = _itc_overflow_concentrations(raw_stub, params)

def _solve(g_mM, h_mM):
    tot = {sp: 0.0 for sp in net["all_species"]}
    tot["G"], tot["H"] = g_mM * 1e-3, h_mM * 1e-3
    c, _, _ = solve_equilibria_general(tot, net["equilibria"], net["all_species"], LOGK)
    return np.array([float(c.get(sp, 0.0)) * 1e3 for sp in rxn_sp])

C = np.vstack([_solve(G_tot[i], H_tot[i]) for i in range(len(G_tot))])
_dV = V_DUMMY_uL / 1000.0
C_dummy = _solve(G0_mM * V0_mL / (V0_mL + _dV), H_STOCK_mM * _dV / (V0_mL + _dV))
C_pre = _solve(G0_mM, 0.0)
dH_vec = np.array([DH[k] for k in ("K1", "K2", "K3")])
q_dummy = float(((C_dummy - C_pre) @ M) @ dH_vec) * V0_mL * 1e3                 # ucal
inj_mL = np.asarray(raw_stub["inj_vols_uL"][1:], float) / 1000.0
dC = _delta_C(C, inj_mL, V0_mL, C_dummy=C_dummy)
q = ((dC @ M) @ dH_vec) * V0_mL * 1e3                                           # ucal per injection

ratio = H_tot / np.maximum(G_tot, 1e-12)
print("usage matrix M (rows GH, GH2, G2H2; cols K1, K2, K3):\n", M)
print(f"c values: c1 = {10**LOGK['K1']*G0_mM*1e-3:.0f}, c2 = {10**LOGK['K2']*G0_mM*1e-3:.1f}, c3 = {10**LOGK['K3']*G0_mM*1e-3:.1f}")
print(f"molar ratio H/G {ratio[0]:.3f} -> {ratio[-1]:.3f} over {N_INJ} injections; q range {q.min():.2f} .. {q.max():.2f} ucal")
imax = int(np.argmax(C[:, 2])); print(f"G2H2 peaks at injection {imax+1} (ratio {ratio[imax]:.2f}): {C[imax,2]:.4f} mM of {G_tot[imax]:.3f} mM G")

# ── power trace ────────────────────────────────────────────────────────────
n_per = int(round(SPACING_S / DT_S)); N_WIN = N_INJ + 2; n_tot = N_WIN * n_per
q_all = np.concatenate([[q_dummy], q])
t = (np.arange(n_tot) + 1) * DT_S
power = BASE_POWER + NOISE * DRIFT_TOTAL * (t / t[-1])
for i in range(1, N_WIN):
    st_i = i * n_per
    tl = np.arange(n_tot - st_i) * DT_S
    shape = np.exp(-tl / TAU_D) - np.exp(-tl / TAU_R)
    power[st_i:] += q_all[i - 1] * shape / _trap(shape, dx=DT_S)
power += RNG.normal(0.0, NOISE * NOISE_SD, n_tot)

# ── MicroCal layout, CRLF ─────────────────────────────────────────────────
L = ["$ITC", f"$ {N_INJ} ", "$NOT", f"$ {TEMP_C:.0f} ", "$ 60 ", "$ 1250 ", "$ 10 ", "$ 2 ",
     "$ADCGainCode:  0 ", "$False,True,True",
     f"$ {V_DUMMY_uL} , {V_DUMMY_uL*2:.1f} , {SPACING_S:.0f} , {DT_S:.0f} "]
L += [f"$ {V_INJ_uL} , {V_INJ_uL*2:.1f} , {SPACING_S:.0f} , {DT_S:.0f} " for _ in range(N_INJ)]
L += ["# 0 ", f"# {G0_mM} ", f"# {H_STOCK_mM:.0f} ", f"# {V0_mL} ", f"# {TEMP_C:.0f} ", "# 3.6697 ", "# 8.96611 ", "?",
      "% ITC200_10.11.570", f"% {V0_mL} ", "% 3.6697 ", "%-0.605691 , 29.79366 ", "%-0.014574 , 29.52486 ",
      "% 0.8333333 ", "% 168960 , 168950 ", "% 0.25 , 81 ",
      "% 0.04257 ,-0.00249 , 0.000047734 ,-0.000000435026 , 0.00000000201917 ", "% ITC200 Ver: 1.25.5"]
jt = TEMP_C + 0.0198
for i in range(N_WIN):
    if i == 0:
        L.append("@0")
    else:
        v = V_DUMMY_uL if i == 1 else V_INJ_uL
        L.append(f"@{i},{v:.4f},{v*2:.1f}")
    for k in range(i * n_per, min((i + 1) * n_per, n_tot)):
        L.append(f"{t[k]:.2f},{power[k]:.5f},{TEMP_C + 0.0190 + 3e-6*k:.6f},{0.000197 + 1e-7*(k % 7):.6f},"
                 f"{jt + 2e-6*k:.6f},{0.4420 + 1e-4*(k % 11):.4f},{0.068 + 1e-4*(k % 5):.3f}")

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "tutorial_23_input.itc"), "w", newline="").write("\r\n".join(L) + "\r\n")
open(os.path.join(OUT, "tutorial_23.txt"), "w").write(script(START))
print(f"wrote {OUT}/tutorial_23_input.itc ({n_tot} points) and tutorial_23.txt; truth log K {LOGK}, dH {DH} kcal/mol; scripts start at {START}")
