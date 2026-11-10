# Equilibrist

A browser-based thermodynamic and kinetic solver built with Python and Streamlit.  
Define a chemical system in a plain-text script, simulate titration or time-resolved curves, and fit binding or rate constants to experimental data.  
Runs entirely on your local machine — no data is sent anywhere.  
**Version 3.0.1 beta** adds tabs with a global fit across techniques, isothermal titration calorimetry, supercomputer offload, live fits and 24 tutorials, on top of the statistical-validation, identifiability and uncertainty-quantification layer of 2.0.

---

## Capabilities

- **Equilibrium titrations** — arbitrary reaction networks, any stoichiometry, multiple products per reaction
- **Kinetics simulations** — irreversible (`>`) and reversible (`<>`) steps, mixed equilibrium/kinetics scripts
- **Free-energy input** — reactions defined via ΔG° (van 't Hoff) and ΔG‡ (Eyring) in kcal/mol
- **Acid-base mode** — `$reactions acid-base` accepts pKa values directly; water equilibrium, water concentration, and a `pH` variable are injected automatically; ladder syntax for polyprotic acids
- **Solution and solid titrant modes** — volume-based dilution, or solid addition with fixed cell volume
- **UV-Vis global spectral fitting** — Beer–Lambert analysis; molar absorptivities solved analytically (variable projection); known spectra for individual species can be provided in a second sheet of the data file via `read:` in `$spectra`, pinning those species' ε(λ) during the fit
- **NMR fitting** — fast-exchange chemical shift, slow-exchange integration, and simultaneous slow + fast mixed mode; all three available in both equilibrium and kinetics modes. `read:` in `$nmr` pins per-column intrinsic shifts from a second sheet of the data file; `noref` releases the auto-pinned reference so an externally anchored absolute shift scale is recovered when `read:` is also present
- **Concentration fitting** — direct fitting of measured species concentrations [mM]
- **Tabs and global fit** — several experiments, of the same or different kinds, fitted at once; constants shared by name, each tab weighted by its own misfit, per-tab information shares, diagnostics on all tabs or on one
- **ITC** — MicroCal `.itc` thermograms integrated in the app; binding constants with stepwise enthalpies and a heat offset; ΔG°/ΔH°/ΔS° table; competition experiments with fixed enthalpies
- **Dual optimizer** — L-BFGS-B warm-start followed by Nelder-Mead; user-selectable per run
- **Thermodynamic constraints** — inter-parameter relationships among equilibrium and rate constants enforced as soft or hard penalties; v2.0 adds NMR shift constraints (equality groups `A = B = C`, monotone chains `A > B > C`, per-column magnitude bounds) for fast-exchange fits
- **Thermodynamic cycle detection** — automatic warning on inconsistent K values
- **Outlier flagging** — click any experimental data point to exclude it from fitting; excluded points are absent from all exported figures
- **Statistical validation** — AIC, AICc and BIC printed with every fit; Akaike weights and F-test for nested model comparison; Durbin–Watson and Shapiro–Wilk residual tests; four-panel residual figure (residuals vs predictor, histogram, normal Q–Q, metrics)
- **Identifiability tools** — 1D and 2D RMSE profiles with Hessian-derived confidence ellipses overlaid; parameter correlation heatmap; sloppy-spectrum eigendecomposition with stiff/sloppy ranking and per-parameter impact bars; per-parameter *t*-tests with user-editable null hypotheses
- **Uncertainty quantification** — non-parametric residual bootstrap, parametric Monte Carlo, and Mammen wild bootstrap; leave-one-out jackknife; Monte Carlo propagation of user-specified relative uncertainty on stock concentrations, titrant stocks and V₀. All three layers are available for every fit mode and can run in parallel across all cores
- **Supercomputer submission (SLURM)** — every parallel tool (1D / 2D RMSE profiles, bootstrap, jackknife, Monte Carlo, local sensitivity) can be sent to a SLURM cluster from a panel below its *Parallel* box: connect over SSH, choose nodes × CPUs, submit, close the browser, fetch the result later. Password kept in memory only; optional one-click SSH-key setup; OSC preset included, other clusters via an editable job header
- **Custom Python scripts** — load your own `.py` from the sidebar and run it against the current job: constants, simulated curves, variables, data and fit statistics are handed to the script as `eq`; figures, tables and printed output appear under the main plots with PDF/PNG downloads
- **Local sensitivity (Masson ξ)** — 3<sup>N</sup> paired-difference probe over arbitrary subsets of fitted parameters and held-fixed quantities, returning conditional and marginal standard errors and a coupling ratio
- **FAIR session save/restore** — single self-contained JSON bundle (script + experimental data with SHA-256 checksum + fit state + every uncertainty and sensitivity result + RNG seed + Python / NumPy / SciPy / joblib versions). Drop the `.json` back into the script uploader to reconstruct the analysis exactly
- **Excel export** — timestamped workbook with data, script, and parameters tabs
- **Publication-quality PDF export** — vector text, Arial 9 pt, inward ticks, editable in Illustrator
- **In-app script editor** — edit and re-run without leaving the browser

---

## Requirements

- Python 3.10 or later **or** Miniconda (see below)
- All other dependencies are installed automatically from `requirements.txt` or `environment.yml`

---

## Installation

### Option A — conda (recommended, especially on Windows)

Best if you already have Anaconda or Miniconda installed, or if Option B gives you trouble.  
If you need Miniconda: https://docs.conda.io/en/latest/miniconda.html

```
cd path/to/Equilibrist
conda env create -f environment.yml
conda activate equilibrist
```

Upgrading from 2.0: an existing `equilibrist` environment still runs the app. To add the two packages that the new optional features need (`paramiko` for supercomputer submission, `nmrglue` for the NMR stack tool), run `conda env update -n equilibrist -f environment.yml` instead of `conda env create` (which stops if the environment already exists).

### Option B — pip

Best if you already have Python 3.10+ installed.

```
cd path/to/Equilibrist
pip install -r requirements.txt
```

---

## Running the app

```
conda activate equilibrist    # conda users only — skip if using pip
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.  
Press **Ctrl + C** in the terminal to stop the server.

---

## NMR stack (companion tool)

`nmr_stack.py` is a separate Streamlit app that turns a folder of Bruker 1D
experiments (any depth) into an Equilibrist input file:

```bash
streamlit run nmr_stack.py
```

Point it at the folder (or upload a zip). It processes the FIDs (TopSpin's
phases when present), stacks the spectra in a browser-side canvas where you
zoom, calibrate all spectra on one reference peak, click a signal on each
spectrum where it is (or select many at once with a box, a lasso around a
drifting signal, or a line from a signal on one spectrum to the same signal on
another) to get its shift, pattern, J and integral (the plain sum of the points between the limits, as in TopSpin and Mnova; a lineshape fit is an option, and a deconvolute tool separates overlapping signals where you box them), type the time or volume of
each spectrum beside it, and phase live. The export tab writes the `.xlsx` in
the `$nmr` layout. *Save session* in the sidebar writes everything except the
spectra (calibration, signals and limits, phase tweaks, the experiments table,
settings, ppm window) to a small `.json`; *Load session* brings it back, reading
the spectra again from the folder named in the file.
Needs `nmrglue` (in `requirements.txt`).

## File layout

```
Equilibrist/
├── app.py                              ← main application entry point
├── equilibrist_parser.py
├── equilibrist_network.py
├── equilibrist_kinetics.py
├── equilibrist_kinetics_nmr.py
├── equilibrist_kinetics_spectra.py
├── equilibrist_curve.py
├── equilibrist_fit_conc.py
├── equilibrist_fit_nmr.py
├── equilibrist_fit_spectra.py
├── equilibrist_shift_constraints.py    ← v2.0: NMR shift-constraint solver
├── equilibrist_diagnostics.py          ← v2.0: statistical & identifiability layer
├── equilibrist_bootstrap.py            ← v2.0: bootstrap / jackknife / Monte Carlo
├── equilibrist_session.py              ← v2.0: FAIR JSON session save/restore
├── equilibrist_itc.py                  ← isothermal titration calorimetry
├── equilibrist_hpc.py                  ← SLURM cluster offload (also the driver run on the cluster)
├── equilibrist_hpc_ui.py               ← supercomputer panel
├── equilibrist_pyscript.py             ← custom Python script hook (sidebar buttons)
├── custom_script_example.py            ← starting point for a custom script (two more examples alongside)
├── .streamlit/config.toml              ← Streamlit settings used by the tabs
├── equilibrist_io.py
├── equilibrist_global.py               ← global fit across tabs
├── equilibrist_tabs.py                 ← tabs (workspaces)
├── equilibrist_live.py                 ← live progress of running fits
├── nmr_stack.py, nmr_stack_core.py     ← companion NMR stack tool
├── TUTORIALS/                          ← 24 tutorials (scripts and data) and the generators of the synthetic ones
├── requirements.txt                    ← pip dependency list
├── environment.yml                     ← conda environment definition
├── README.md
└── Equilibrist_manual.html             ← full user manual (open in any browser)
```

A `__pycache__` folder may appear after the first run — this is normal.

---

## Troubleshooting

**`PackagesNotFoundError: nmrglue` during `conda env create`**  
The `environment.yml` of 3.0 beta (8 November 2026) listed `nmrglue` as a conda package; it is not on the conda-forge or default channels, so the environment was not created. This is corrected in 3.0.1 beta. If you still have the 3.0 beta files: replace `environment.yml` with the current one from this repository, or open it in a text editor and move the line `nmrglue>=0.10` under `- pip:` (same indentation as `streamlit`), then run `conda env create -f environment.yml` again.

**"streamlit: command not found"**  
Run `pip install streamlit`, or make sure the conda environment is activated first.

**"pip is not recognized" (Windows)**  
Use `python -m pip install -r requirements.txt` instead.

**Wrong Python version**  
Verify with `python --version` — Equilibrist requires 3.10 or later.

**App won't start after updating files**  
Press **Ctrl + C** to stop any running instance, then relaunch with `streamlit run app.py`.

---

## Citation

If Equilibrist contributes to a publication, please cite: E. Masson, *Equilibrist: A Browser-Based Platform for Fitting Equilibrium and Rate Constants from Optical and NMR Data*, *J. Chem. Inf. Model.* **2026**, *66*, 9976–9988, DOI [10.1021/acs.jcim.6c01004](https://doi.org/10.1021/acs.jcim.6c01004) (open access, CC BY 4.0). The PDF is attached to each GitHub release.

---

## Documentation

Open `Equilibrist_manual.html` in any browser for the full user manual, including all script syntax, data file formats, fitting modes, the complete statistical-validation and uncertainty-quantification layer, the ITC chapter, a tutorial index, and three worked examples (residual diagnostics on a 1:1 binding, 2D RMSE profile + bootstrap + jackknife + local sensitivity on a 2:1 NMR shift fit, and correlation heatmap + sloppy spectrum + Monte Carlo on a kinetics + UV-Vis fit) that mirror the case studies in the accompanying paper.

---

*Equilibrist v3.0.1 beta — built with Python & Streamlit · all computation local, no data transmitted*  
*© Eric Masson, Ohio University, 2026*
