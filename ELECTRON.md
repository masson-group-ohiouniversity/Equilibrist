# Electron wrapper for Equilibrist

## What this is

A thin Electron shell that wraps the existing Streamlit-based Equilibrist
application into a native desktop app.  Electron provides the window chrome,
dock/taskbar icon, and distributable installers (`.dmg`, `.exe`, `.AppImage`)
while the Python/Streamlit backend runs unchanged as a local server.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Electron (Chromium renderer)                │
│  ┌────────────────────────────────────────┐  │
│  │  BrowserWindow → http://127.0.0.1:PORT│  │
│  └────────────────────────────────────────┘  │
│         ▲  WebSocket (Streamlit protocol)    │
│         │                                    │
│  ┌──────┴─────────────────────────────────┐  │
│  │  Streamlit server  (child process)     │  │
│  │  └─ app.py + equilibrist_*.py modules  │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### Startup sequence

1. **Electron `main.js`** finds a free TCP port.
2. It locates the **bundled standalone Python** (`resources/python-runtime/`)
   and sets `PYTHONPATH` to the bundled library directory
   (`resources/python-libs/`).  Falls back to system Python only in development
   mode (when no bundled runtime is found).
3. It spawns `<python> -m streamlit run app.py` bound to `127.0.0.1:<port>`.
4. A splash screen (`splash.html`) is shown while the server boots.
5. `main.js` polls the port; once Streamlit responds, the `BrowserWindow` loads
   the Streamlit URL.
6. On quit, Electron sends `SIGTERM` to the Streamlit process (with a 3-second
   `SIGKILL` fallback).

### Files

```
electron/
├── package.json    # Electron + electron-builder config
├── main.js         # App lifecycle, Streamlit child-process management
├── preload.js      # Secure renderer ↔ main bridge (minimal for now)
└── splash.html     # Loading screen shown during Streamlit startup
```

## How this fixes the Kaleido problem

### Background

Equilibrist previously used **Kaleido** for Plotly-to-PNG export.  Kaleido
ships a **headless Chromium binary** that renders Plotly figures off-screen.
This caused:

- **Windows hangs** — Kaleido's Chromium subprocess would freeze indefinitely
  on many Windows machines (see `KALEIDO.md` for the full post-mortem).
- **Bloated bundle** — an extra ~150-250 MB for a second Chromium binary.
- **Signing headaches** — macOS Gatekeeper requires every embedded binary to be
  signed/notarized; a second Chromium doubles the effort.

Kaleido was removed and replaced with a matplotlib-based re-draw, which works
reliably but only *approximates* the Plotly chart (different rendering engine,
slightly different styling).

### Why Electron eliminates the need for Kaleido entirely

Electron **is** Chromium.  The Plotly charts displayed in the app are already
rendered by the same engine Kaleido was trying to provide.  This means:

1. **`Plotly.toImage()` is available in the renderer.**  Plotly.js includes a
   built-in static-export method that converts the chart's `<canvas>` / SVG to
   a PNG or JPEG data-URL.  Inside Electron you can call this via IPC and save
   the result directly to disk — no extra process, no extra Chromium.

2. **`webContents.capturePage()`** can screenshot any region of the window,
   giving you a pixel-perfect capture of exactly what the user sees.

3. **No subprocess** — there is only one Chromium process (Electron's own).
   No child-process hangs, no platform-specific Chromium quirks, no extra
   system-library dependencies.

### Comparison

| Export method          | Fidelity       | Extra deps        | Windows stable | Bundle cost |
|------------------------|----------------|-------------------|----------------|-------------|
| Kaleido                | Pixel-perfect  | Headless Chromium  | No (hangs)     | +150-250 MB |
| Matplotlib re-draw     | Approximate    | None (already dep) | Yes            | +0 MB       |
| Electron `Plotly.toImage()` | Pixel-perfect | None          | Yes            | +0 MB       |

The Electron path gives you **pixel-perfect Plotly fidelity** with **zero
additional dependencies** and **no subprocess stability risk**.

## Bundled Python runtime

Distributable builds include a **standalone Python interpreter** and all
dependencies in a flat library directory.  This means **end users do not need
Python installed** — everything is self-contained inside the app.

> **Why not a venv?**  A standard Python venv hardcodes the absolute path to
> the base interpreter in `pyvenv.cfg` and in script shebangs.  When the venv
> is built on a CI runner (e.g. `C:\hostedtoolcache\…`) and then installed on
> a user's machine, those paths don't exist and Python refuses to start.  The
> standalone approach has no hardcoded paths.

The CI workflow (`.github/workflows/build-electron.yml`) handles this
automatically:

1. Downloads a **portable CPython 3.11** build from
   [python-build-standalone](https://github.com/indygreg/python-build-standalone)
   — a self-contained interpreter with no system dependencies.
2. Extracts it to `electron/python-runtime/`.
3. Runs `pip install --target electron/python-libs` to install all packages
   from `Equilibrist/requirements.txt` into a flat directory.
4. `electron-builder` bundles both `python-runtime/` and `python-libs/` as
   extra resources via the `extraResources` config in `package.json`.
5. At runtime, `main.js` sets `PYTHONPATH` to point at `python-libs/` and
   invokes the bundled interpreter directly.

### Bundled resources layout (installed app)

```
resources/
├── Equilibrist/          # Python source code
│   ├── app.py
│   └── equilibrist_*.py
├── python-runtime/       # Standalone CPython interpreter
│   ├── bin/python3       # (or python.exe on Windows)
│   └── lib/…
└── python-libs/          # pip --target output (all deps)
    ├── streamlit/
    ├── numpy/
    ├── scipy/
    └── …
```

### Python lookup order

1. `EQUILIBRIST_PYTHON` env var (explicit override)
2. Bundled standalone interpreter at `<resources>/python-runtime/`
3. System `python3` / `python` on PATH (development fallback)

## Development quickstart

```bash
# 1. Install Python dependencies (if not already done)
cd Equilibrist
pip install -r requirements.txt

# 2. Install Electron dependencies
cd ../electron
npm install

# 3. Run in development mode (uses system Python)
npm start
```

To use a specific Python interpreter in development:

```bash
EQUILIBRIST_PYTHON=/path/to/venv/bin/python npm start
```

To test with a bundled runtime locally (mirrors what CI does):

```bash
cd electron

# Download standalone Python (example: macOS arm64)
curl -fSL -o python.tar.gz \
  "https://github.com/indygreg/python-build-standalone/releases/download/20241219/cpython-3.11.11+20241219-aarch64-apple-darwin-install_only_stripped.tar.gz"
mkdir -p python-runtime && tar -xzf python.tar.gz --strip-components=1 -C python-runtime
rm python.tar.gz

# Install deps into flat lib dir
python-runtime/bin/python3 -m pip install --target python-libs -r ../Equilibrist/requirements.txt

npm start
```

## Packaging for distribution

```bash
cd electron

# macOS
npm run dist:mac

# Windows
npm run dist:win

# Linux
npm run dist:linux
```

This uses [electron-builder](https://www.electron.build/) to produce platform
installers.  The `Equilibrist/` source, `python-runtime/` interpreter, and
`python-libs/` dependencies are all bundled as extra resources.

## CI / GitHub Actions

The workflow at `.github/workflows/build-electron.yml` builds distributable
packages for all three platforms on every pull request:

| Runner           | Target arch | Output         | Artifact name              |
|------------------|-------------|----------------|----------------------------|
| `macos-latest`   | arm64       | `.dmg`         | Equilibrist-macOS-arm64    |
| `macos-latest`   | x64         | `.dmg`         | Equilibrist-macOS-x64      |
| `windows-latest` | x64         | `.exe` (NSIS)  | Equilibrist-Windows        |
| `ubuntu-latest`  | x64         | `.AppImage`    | Equilibrist-Linux          |

The Intel (x64) macOS build is **cross-compiled** on an ARM runner using
`electron-builder --x64`.  This avoids the need for a dedicated `macos-13`
runner (which is not available in all CI environments).  The bundled Python
runtime for the x64 build is the `x86_64-apple-darwin` variant from
python-build-standalone, so it runs natively on Intel Macs.

Artifacts are uploaded with 14-day retention.

### Downloading installers from CI

Pre-built installers are generated automatically on every pull request.
To download them:

1. Open the pull request on GitHub.
2. Click the **Checks** tab (or scroll to the status checks at the bottom).
3. Click the **"Build Electron distributables"** workflow run.
4. On the workflow summary page, scroll to the **Artifacts** section.
5. Download the artifact for your platform:
   - **Equilibrist-macOS-arm64** — `.dmg` for Apple Silicon Macs (M1/M2/M3/M4)
   - **Equilibrist-macOS-x64** — `.dmg` for Intel Macs
   - **Equilibrist-Windows** — `.exe` installer for Windows
   - **Equilibrist-Linux** — `.AppImage` for Linux
6. Unzip the downloaded archive and run the installer.

## Future enhancements

- **Plotly.toImage() IPC bridge** — wire up `preload.js` to call
  `Plotly.toImage()` in the renderer and save to disk via Node.js `fs`, giving
  pixel-perfect PNG/SVG/PDF export without matplotlib.
- **Auto-updates** — use `electron-updater` for seamless version bumps.
- **Native file dialogs** — replace Streamlit's file upload/download widgets
  with native OS dialogs via Electron IPC for a more desktop-native feel.
- **Slimmer bundle** — strip tests, `__pycache__`, and `.pyc` files from the
  bundled `python-libs/` to reduce installer size.
