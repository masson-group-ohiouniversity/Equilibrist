# Custom Equilibrist script -- three plots in a row.
#
#   a   the main-window plot, as drawn (lines + experimental points)
#   b   total concentration of guest species carrying 0, 1 and 2 metal ions
#   c   the concentration of every species of the model
#
# In panel c the colour tells how many hosts a species carries (same colours as
# S0, S1, S2, S3 on the main plot) and the line style how many metal ions:
# solid = 0, dashed = 1, dotted = 2.  Species without guest are drawn in grey
# (free metal ion: dark, dash-dot).
#
# Load this file with "Load custom .py script" and press "Run py script".

import numpy as np
import matplotlib.pyplot as plt

# ── Settings ────────────────────────────────────────────────────────────────
GUEST, HOST, METAL = "G", "H", "M"  # free components of the model
METAL_NAME = "Ag"                   # label used in the legend of panel b
STATE_NAMES = ["S0", "S1", "S2", "S3"]   # main-plot traces whose colours stand for 0, 1, 2, 3 hosts
GUEST_ONLY = False                  # True: panel c leaves out H, M, HM, HM2 ...
HIDE_BELOW = 0.0                    # panel c: skip species that never exceed this concentration (mM)
ONLY_STATES_PANEL_A = False         # True: hide main-plot traces that are not in STATE_NAMES (e.g. S012)
METAL_COLORS = ["#a6a6a6", "#5c5c5c", "#000000"]          # panel b: 0, 1, 2 metal ions
HOST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # used if the main plot has no S0..S3
LINESTYLES = ["-", "--", ":", "-."]                        # 0, 1, 2, 3 metal ions

# ── 1. Composition of every species, from the reaction network ─────────────
def composition(sp, _memo={}):
    """{free component: count} of a species, e.g. GH3M2 -> {'G': 1, 'H': 3, 'M': 2}."""
    if sp in _memo:
        return _memo[sp]
    rxn = eq.network["prod_to_eq"].get(sp)
    if rxn is None:                                  # a free component
        comp = {sp: 1.0}
    else:
        comp = {}
        for coeff, r in rxn["reactants"]:
            for k, n in composition(r).items():
                comp[k] = comp.get(k, 0.0) + coeff * n
    _memo[sp] = comp
    return comp

def count(sp, component):
    return int(round(composition(sp).get(component, 0)))

species = [s for s in eq.species if s in eq.conc]
guest_species = [s for s in species if count(s, GUEST) > 0]
groups = {}                                          # number of metal ions -> [guest species]
for s in guest_species:
    groups.setdefault(count(s, METAL), []).append(s)
by_metal = {n: sum(eq.conc[s] * count(s, GUEST) for s in groups[n]) for n in sorted(groups)}

main_color = {t["name"]: t["color"] for t in eq.plot if t["kind"] == "line" and t["color"]}
host_color = [main_color.get(n, HOST_COLORS[i % len(HOST_COLORS)]) for i, n in enumerate(STATE_NAMES)]

def quantity_of(trace_name):
    """'S3 (NMR/integration)' -> 'S3';  '_outlier_S3_(NMR)' -> 'S3'."""
    name = trace_name[len("_outlier_"):] if trace_name.startswith("_outlier_") else trace_name
    return name.replace("_", " ").split(" ")[0]

# ── 2. Figure ───────────────────────────────────────────────────────────────
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.5, 4.2))
x = eq.x

# a: the main plot, as drawn
for t in eq.plot:
    if t["name"] == "solver warning":
        continue
    if ONLY_STATES_PANEL_A and quantity_of(t["name"]) not in STATE_NAMES:
        continue
    if t["kind"] == "line":
        ax_a.plot(t["x"], t["y"], color=t["color"], lw=1.6, label=t["name"])
    else:
        hollow = t["excluded"]
        ax_a.plot(t["x"], t["y"], "o", ms=3.6, mew=0.9, color=t["color"] or "0.6",
                  mfc="none" if hollow else (t["color"] or "0.6"),
                  label=None if hollow else t["name"])

# b: guest species grouped by the number of metal ions
for n, col in zip(by_metal, METAL_COLORS * 3):
    ax_b.plot(x, by_metal[n], color=col, lw=1.8, label=f"{n} {METAL_NAME}")

# c: every species -- colour = number of hosts, line style = number of metal ions
shown = [s for s in (guest_species if GUEST_ONLY else species)
         if float(np.nanmax(eq.conc[s])) > HIDE_BELOW]
shown.sort(key=lambda s: (count(s, GUEST) == 0, count(s, HOST), count(s, METAL)))
for s in shown:
    is_guest = count(s, GUEST) > 0
    col = host_color[min(count(s, HOST), len(host_color) - 1)] if is_guest else "0.55"
    ls = LINESTYLES[min(count(s, METAL), len(LINESTYLES) - 1)]
    if not is_guest and count(s, HOST) == 0:         # the free metal ion: its own look
        col, ls = "0.2", "-."
    ax_c.plot(x, eq.conc[s], color=col, lw=1.6 if is_guest else 1.2, ls=ls, label=s)

for ax, tag in zip((ax_a, ax_b, ax_c), "abc"):
    ax.set_xlim(0, float(eq.xmax) if eq.xmax else float(np.nanmax(x)))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(eq.x_label)
    ax.set_ylabel("Concentration [mM]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_title(tag, loc="left", fontweight="bold", fontsize=10, pad=3)
ax_a.legend(frameon=False, fontsize=7, handlelength=1.6, loc="upper right", ncol=2, columnspacing=1.0)
ax_b.legend(frameon=False, fontsize=7.5, handlelength=1.6)
ax_c.legend(frameon=False, fontsize=7, handlelength=2.6, ncol=2, columnspacing=1.0)
fig.tight_layout()

# ── 3. What went into each group ───────────────────────────────────────────
for n in by_metal:
    print(f"{n} {METAL_NAME}: {' + '.join(groups[n])}")
print("panel c:", ", ".join(shown))
