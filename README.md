# Equilibrist v1.0

Fitting software for complex chemical equilibria and kinetics from NMR and absorption spectroscopy data.  

---

## Requirements

- Python 3.10 or later **or** Miniconda (see below)
- All other dependencies are installed automatically

---

## Installation — choose one method

### Option A: conda (recommended, especially on Windows)

Best if you already have Anaconda or Miniconda, or if Option B gives you trouble.

1. Download the Equilibrist folder
2. Install Miniconda if needed: https://docs.conda.io/en/latest/miniconda.html
3. Open a terminal (Mac/Linux) or Anaconda Prompt (Windows)
4. Navigate to the Equilibrist folder:
   ```
   cd path/to/Equilibrist
   ```
5. Create the environment (once):
   ```
   conda env create -f environment.yml
   ```
6. Activate it:
   ```
   conda activate Equilibrist
   ```

### Option B: pip

Best if you already have Python 3.10+ installed.

1. Download the Equilibrist folder
2. Open a terminal and navigate to the Equilibrist folder:
   ```
   cd path/to/Equilibrist
   ```
3. Install dependencies (once):
   ```
   pip install -r requirements.txt
   ```

---

## Running the app

After installing (Option A or B), launch the app with:
```
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

**Note for conda users:** make sure the environment is activated first (`conda activate Equilibrist`).

---

## Every time you use Equilibrist

```
conda activate Equilibrist        ← conda users only
streamlit run app.py
```

---

## Desktop installer (optional)

Pre-built desktop installers are available for **Windows**, **macOS**, and
**Linux**.  The installer bundles Python, all dependencies, and a native app
window — no separate Python installation required.

**This is the recommended way to run Equilibrist on Windows**, as it avoids
common issues with Python path configuration, missing build tools for compiled
dependencies, and permission errors.

To get the installer:

1. Go to the latest pull request or release on GitHub.
2. Open the **Checks** tab and click the **"Build Electron distributables"**
   workflow.
3. Download the artifact for your platform from the **Artifacts** section:
   - **Windows** — `Equilibrist-Windows` (`.exe` installer)
   - **macOS (Apple Silicon)** — `Equilibrist-macOS-arm64` (`.dmg`)
   - **macOS (Intel)** — `Equilibrist-macOS-x64` (`.dmg`)
   - **Linux** — `Equilibrist-Linux` (`.AppImage`)
4. Run the installer — no additional setup needed.

For more details on the Electron wrapper, see `ELECTRON.md`.

---

## Troubleshooting

**"streamlit: command not found"**
Run `pip install streamlit` or make sure the conda environment is activated.

**Publication figure (PDF) or Snapshot button does nothing**
These features use matplotlib — make sure it is installed: `pip install matplotlib>=3.7`

**Wrong Python version**
Make sure you are using Python 3.10 or later: `python --version`
