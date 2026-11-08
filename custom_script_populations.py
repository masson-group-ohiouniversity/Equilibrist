# Custom Equilibrist script -- four population plots in a 2 x 2 figure.
#
#   top left     the main-window plot, as drawn (lines + experimental points)
#   top right    molar fractions of the states S0, S1, S2, S3 (they add up to 100 %)
#   bottom left  total concentration of guest species carrying 0, 1 and 2 metal ions
#   bottom right molar fractions of the same three groups (add up to 100 %)
#
# Load this file with "Load custom .py script" and press "Run py script".

import numpy as np
import matplotlib.pyplot as plt

# ── Settings ────────────────────────────────────────────────────────────────
STATES = ["S0", "S1", "S2", "S3"]   # $variables shown as fractions (top right)
GUEST, METAL = "G", "M"             # free components: which species count, what is counted
METAL_NAME = "Ag"                   # label used in the legends
ONLY_STATES_TOP_LEFT = False        # True: hide main-plot traces that are not in STATES (e.g. S012)
METAL_COLORS = ["#a6a6a6", "#5c5c5c", "#000000"]   # 0, 1, 2 metal ions: light to dark

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

guest_species = [s for s in eq.species if s in eq.conc and composition(s).get(GUEST, 0) > 0]
groups = {}                                          # number of metal ions -> [species]
for s in guest_species:
    groups.setdefault(int(round(composition(s).get(METAL, 0))), []).append(s)
n_metal = sorted(groups)
by_metal = {n: sum(eq.conc[s] * composition(s)[GUEST] for s in groups[n]) for n in n_metal}
total_guest = sum(by_metal.values())

missing = [n for n in STATES if n not in eq.variables and n not in eq.conc]
if missing:
    raise SystemExit(f"STATES lists {missing}, which are not $variables of this model. "
                     f"Defined: {list(eq.variables)}")
states = {n: eq.var(n) for n in STATES}
total_states = sum(states.values())

# colours of the states = colours of the main plot
main_color = {t["name"]: t["color"] for t in eq.plot if t["kind"] == "line" and t["color"]}
fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
state_color = {n: main_color.get(n, fallback[i % len(fallback)]) for i, n in enumerate(STATES)}

def quantity_of(trace_name):
    """'S3 (NMR/integration)' -> 'S3';  '_outlier_S3_(NMR)' -> 'S3'."""
    name = trace_name[len("_outlier_"):] if trace_name.startswith("_outlier_") else trace_name
    return name.replace("_", " ").split(" ")[0]

# ── 2. Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
(ax_c, ax_f), (ax_mc, ax_mf) = axes
x = eq.x

# top left: the main plot, as drawn
for t in eq.plot:
    q = quantity_of(t["name"])
    if ONLY_STATES_TOP_LEFT and q not in STATES:
        continue
    if t["kind"] == "line":
        if t["name"] == "solver warning":
            continue
        ax_c.plot(t["x"], t["y"], color=t["color"], lw=1.6, label=t["name"])
    else:
        hollow = t["excluded"]
        ax_c.plot(t["x"], t["y"], "o", ms=3.6, mew=0.9, color=t["color"] or "0.6",
                  mfc="none" if hollow else (t["color"] or "0.6"),
                  label=None if hollow else t["name"])
ax_c.set_ylabel("Concentration [mM]")

# top right: fractions of the states; experimental points divided by the total guest
for n in STATES:
    ax_f.plot(x, 100 * states[n] / total_states, color=state_color[n], lw=1.6, label=n)
for t in eq.plot:
    if t["kind"] == "points" and quantity_of(t["name"]) in STATES:
        q = quantity_of(t["name"])
        tot = np.interp(t["x"], x, total_states)
        ax_f.plot(t["x"], 100 * t["y"] / tot, "o", ms=3.6, mew=0.9, color=state_color[q],
                  mfc="none" if t["excluded"] else state_color[q])
ax_f.set_ylabel("Molar fraction [%]")

# bottom: species grouped by the number of metal ions they carry
for n, col in zip(n_metal, METAL_COLORS * 3):
    lab = f"{n} {METAL_NAME}"
    ax_mc.plot(x, by_metal[n], color=col, lw=1.8, label=lab)
    ax_mf.plot(x, 100 * by_metal[n] / total_guest, color=col, lw=1.8, label=lab)
ax_mc.set_ylabel("Concentration [mM]")
ax_mf.set_ylabel("Molar fraction [%]")

for ax in (ax_f, ax_mf):
    ax.set_ylim(0, 100)
for ax in (ax_c, ax_mc):
    ax.set_ylim(bottom=0)
for ax in axes.ravel():
    ax.set_xlim(0, float(eq.xmax) if eq.xmax else float(np.nanmax(x)))
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=7.5, handlelength=1.6,
              **({"loc": "upper right", "ncol": 2, "columnspacing": 1.0} if ax is ax_c else {}))
for ax in (ax_mc, ax_mf):
    ax.set_xlabel(eq.x_label)
for ax, tag in zip(axes.ravel(), "abcd"):
    ax.set_title(tag, loc="left", fontweight="bold", fontsize=10, pad=3)
fig.tight_layout()

# ── 3. What went into each group ───────────────────────────────────────────
for n in n_metal:
    print(f"{n} {METAL_NAME}: {' + '.join(groups[n])}")
print("states:", ", ".join(f"{n} = {eq.variables.get(n, n)}" for n in STATES))
