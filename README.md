# Equilibrist

A browser-based thermodynamic and kinetic solver built with Python and Streamlit.  
Define a chemical system in a plain-text script, simulate titration or time-resolved curves, and fit binding or rate constants to experimental data.  
Runs entirely on your local machine — no data is sent anywhere.

---

## Capabilities

- **titrations** — arbitrary reaction networks, any stoichiometry, multiple products per reaction
- **Kinetics simulations** — irreversible (`>`) and reversible (`<>`) steps, mixed equilibrium/kinetics scripts
- **Free-energy input** — reactions defined via free energy terms ΔG° and ΔG‡ in kcal/mol
- **Acid-base mode** — accepts pKa values directly; ion product constant, water activity, and a `pH` variable are injected automatically; ladder syntax for polyprotic acids
- **Solution and solid titrant modes** — volume-based dilution, or solid addition with fixed cell volume
- **UV-Vis global spectral fitting** — Beer–Lambert analysis; molar absorptivities solved analytically (variable projection); known spectra for individual species can be provided in a second sheet of the data file, pinning those species' ε(λ) during the fit
- **NMR fitting** — fast-exchange chemical shift, slow-exchange integration, and simultaneous mixed-mode
- **Concentration fitting** — direct fitting of measured species concentrations (mM)
- **Dual optimizer** — L-BFGS-B warm-start followed by Nelder-Mead; user-selectable per run
- **Thermodynamic constraints** — inter-parameter relationships enforced as soft or hard penalties
- **Thermodynamic cycle detection** — automatic warning on inconsistent K values
- **Outlier flagging** — click any experimental data point to exclude it from fitting; excluded points are absent from all exported figures
- **Excel export** — timestamped workbook with data, script, and parameters tabs, and spectra of pure species
- **Publication-quality PDF export** — vector text, editable in Illustrator
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

## File layout

```
Equilibrist/
├── app.py                          ← main application entry point
├── equilibrist_parser.py
├── equilibrist_network.py
├── equilibrist_kinetics.py
├── equilibrist_kinetics_nmr.py
├── equilibrist_kinetics_spectra.py
├── equilibrist_curve.py
├── equilibrist_fit_conc.py
├── equilibrist_fit_nmr.py
├── equilibrist_fit_spectra.py
├── equilibrist_io.py
├── requirements.txt                ← pip dependency list
├── environment.yml                 ← conda environment definition
├── README.md
└── Equilibrist_manual.html         ← full user manual (open in any browser)
```

A `__pycache__` folder may appear after the first run — this is normal.

---

## Tutorials

18 Tutorials available; tutorials_MMX.txt scripts (MM = 01-18, X = A, B, C...) assess any input files tutorial_MMY_input.xlsx (Y = A, B, C...).

---

## Documentation

Open `Equilibrist_manual.html` in any browser for the full user manual, including all script syntax, data file formats, fitting modes, and worked examples.

---

*Equilibrist — built with Python & Streamlit · all computation local, no data transmitted*  
*© Eric Masson, Ohio University, 2026*
