# Custom Equilibrist script -- `eq` holds everything about the job shown in the app.
# Leave a matplotlib figure in a variable named `fig` (or call eq.show) and it is
# displayed under the main plots, with PDF / PNG downloads.
#
# What `eq` offers:
#   eq.logK, eq.K          every constant (log10 K / K) at its sidebar value
#   eq.x, eq.x_label       x axis of the simulated curve ($plot x)
#   eq.conc[species]       concentration (mM) of any species along eq.x
#   eq.var("S1")           a $variables expression evaluated along eq.x
#   eq.simulate(KGHM=7.0)  (x, conc) with some constants changed
#   eq.plot                the main plot as drawn: list of {name, kind, x, y, color, excluded}
#   eq.data, eq.fit, eq.errors, eq.parsed, eq.network, eq.params
#   eq.show(obj)           show a figure, a DataFrame or text; print() works too

import numpy as np
import matplotlib.pyplot as plt

# ── Example: fractions of the protonation states within the S012 pool ──────
# This is what the S012 NMR signal reports (a population-weighted average),
# as opposed to the absolute concentrations drawn by Equilibrist.
needed = ["S0", "S1", "S2"]
missing = [n for n in needed if n not in eq.variables and n not in eq.conc]
if missing:
    raise SystemExit(f"This example needs $variables named {needed}; "
                     f"missing: {missing}.  Defined variables: {list(eq.variables)}")

S0, S1, S2 = (eq.var(n) for n in needed)
pool = S0 + S1 + S2

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(eq.x, S0 / pool, label="S0 / S012")
ax.plot(eq.x, S1 / pool, label="S1 / S012")
ax.plot(eq.x, S2 / pool, label="S2 / S012")
ax.set_xlabel(eq.x_label)
ax.set_ylabel("fraction of the S012 pool")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
fig.tight_layout()

print("constants:", {k: round(v, 2) for k, v in eq.logK.items()})
