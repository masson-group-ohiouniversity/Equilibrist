# Why kaleido was removed from Equilibrist

## What is kaleido?

[Kaleido](https://github.com/plotly/Kaleido) is a cross-platform library for
static image export of web-based visualization libraries (Plotly, Vega, etc.).
Under the hood it ships a **headless Chromium** binary that renders the figure
off-screen and captures the result as PNG, SVG, or PDF.

Equilibrist previously used kaleido to convert interactive Plotly charts into
PNG images for the **Snapshot** export feature.

## The problem on Windows

Kaleido's embedded Chromium subprocess would **hang indefinitely** on many
Windows machines.  The symptoms were:

| Symptom | Cause |
|---------|-------|
| The Streamlit page freezes after the Export / Snapshot button is pressed | `pio.to_image()` blocks the main thread while Chromium is unresponsive |
| No PNG file is produced; the spinner never stops | The Chromium process never returns |
| High CPU usage from a `chromium` child process | Chromium enters an infinite loop or waits on a missing resource |

The root cause is that kaleido's Chromium build depends on several system
libraries and GPU/display abstractions that behave differently across Windows
versions, antivirus configurations, and Docker Desktop back-ends.  Version
`0.2.1` was more stable than later releases, but still failed on a significant
fraction of Windows setups.

### Attempted workarounds (before removal)

1. **Platform-specific version pinning** — `requirements.txt` used
   `kaleido==0.2.1` on Windows and `kaleido>=0.2` elsewhere.  This reduced but
   did not eliminate the hang.

2. **Timeout wrapper with `ThreadPoolExecutor`** — A function
   `_safe_plotly_to_image()` wrapped the kaleido call in a 15-second timeout so
   that at least the UI would not freeze forever.  If the timeout expired the
   export silently failed.

3. **Docker container** — Running Equilibrist inside a Linux Docker container
   avoided the Windows-specific Chromium issues entirely, but required users to
   install Docker Desktop.

None of these were satisfactory: pinning still failed for some users, the
timeout silently swallowed errors, and Docker added a heavy prerequisite for a
lightweight desktop app.

## The solution: matplotlib-based export

Matplotlib was already a dependency (used for publication-quality PDF figures).
The new helper function `_plotly_to_png_bytes()` in `equilibrist_io.py`:

1. Iterates over each trace in the Plotly figure.
2. Extracts x/y data, line/marker colors, and plot mode.
3. Re-draws the traces on a matplotlib `Figure` / `Axes`.
4. Copies axis labels and title from the Plotly layout.
5. Saves the figure to an in-memory PNG buffer via `fig.savefig()`.

This approach:

- Has **zero additional dependencies** (matplotlib + Pillow were already
  required).
- Works identically on **Windows, macOS, and Linux** with no platform-specific
  code paths.
- Does **not spawn any child process** — it runs entirely in the Python
  interpreter.
- Produces comparable image quality for the purpose of quick snapshots.

## What was removed

| Item | Description |
|------|-------------|
| `kaleido` in `requirements.txt` | Dependency line (with platform markers) |
| `kaleido` in `environment.yml` | Conda dependency |
| `_safe_plotly_to_image()` | Timeout wrapper in `equilibrist_io.py` |
| `Dockerfile` | Linux container configuration |
| `docker-compose.yml` | Container orchestration |
| `.dockerignore` | Docker build exclusion list |
| Chromium system libraries in Dockerfile | `libnss3`, `libx11-xcb1`, etc. |
| Docker section in `README.md` | "Option C: Docker" installation guide |
| kaleido callout in `Equilibrist_manual.html` | Windows troubleshooting note |

## If you still need kaleido

If your workflow requires kaleido (e.g. for SVG or EPS export, or for
pixel-perfect Plotly reproduction), you can install it manually:

```bash
pip install kaleido==0.2.1
```

Note that this will **not** re-enable it inside Equilibrist — the app now uses
matplotlib unconditionally.  You would need to call `plotly.io.to_image()`
directly in your own scripts.
