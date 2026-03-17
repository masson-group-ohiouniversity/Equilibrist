const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");
const fs = require("fs");

let mainWindow = null;
let streamlitProcess = null;
let stderrBuffer = "";

// ---------------------------------------------------------------------------
// Port helper – find a free TCP port to avoid collisions
// ---------------------------------------------------------------------------
function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

// ---------------------------------------------------------------------------
// Resolve paths for the bundled Python runtime
// ---------------------------------------------------------------------------
function getResourceBase() {
  return app.isPackaged ? process.resourcesPath : __dirname;
}

function getAppDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "Equilibrist");
  }
  return path.join(__dirname, "..", "Equilibrist");
}

// ---------------------------------------------------------------------------
// Locate the bundled standalone Python interpreter
// ---------------------------------------------------------------------------
function getPython() {
  if (process.env.EQUILIBRIST_PYTHON) {
    return process.env.EQUILIBRIST_PYTHON;
  }

  const base = getResourceBase();
  const pythonDir = path.join(base, "python-runtime");

  // Standalone Python from python-build-standalone
  if (process.platform === "win32") {
    const p = path.join(pythonDir, "python.exe");
    if (fs.existsSync(p)) return p;
  } else {
    const p = path.join(pythonDir, "bin", "python3");
    if (fs.existsSync(p)) return p;
    const p2 = path.join(pythonDir, "bin", "python");
    if (fs.existsSync(p2)) return p2;
  }

  // Development fallback: system Python
  return process.platform === "win32" ? "python" : "python3";
}

// ---------------------------------------------------------------------------
// Build the PYTHONPATH so the bundled site-packages are found
// ---------------------------------------------------------------------------
function getPythonLibDir() {
  const base = getResourceBase();
  return path.join(base, "python-libs");
}

// ---------------------------------------------------------------------------
// Launch the Streamlit server
// ---------------------------------------------------------------------------
function startStreamlit(port) {
  const python = getPython();
  const appDir = getAppDir();
  const appPy = path.join(appDir, "app.py");
  const libDir = getPythonLibDir();

  console.log(`[Equilibrist] Python:  ${python}`);
  console.log(`[Equilibrist] Libs:    ${libDir}`);
  console.log(`[Equilibrist] App dir: ${appDir}`);
  console.log(`[Equilibrist] Port:    ${port}`);

  const args = [
    "-m", "streamlit", "run", appPy,
    "--server.port", String(port),
    "--server.address", "127.0.0.1",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
    "--global.developmentMode", "false",
  ];

  // Prepend bundled libs to PYTHONPATH so imports resolve there first
  const sep = process.platform === "win32" ? ";" : ":";
  const existingPP = process.env.PYTHONPATH || "";
  const pythonPath = existingPP ? `${libDir}${sep}${existingPP}` : libDir;

  streamlitProcess = spawn(python, args, {
    cwd: appDir,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      // Prevent Python from looking for a user site-packages
      PYTHONNOUSERSITE: "1",
      // Bytecode is pre-compiled at build time; skip writing .pyc at runtime
      PYTHONDONTWRITEBYTECODE: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  streamlitProcess.stdout.on("data", (d) => process.stdout.write(d));
  streamlitProcess.stderr.on("data", (d) => {
    const text = d.toString();
    process.stderr.write(d);
    stderrBuffer = (stderrBuffer + text).slice(-2048);
  });

  streamlitProcess.on("error", (err) => {
    dialog.showErrorBox(
      "Equilibrist – failed to start",
      `Could not launch Python.\n\n` +
        `Tried: ${python}\n\n` +
        `${err.message}\n\n` +
        `If running from source, make sure Python 3.10+ is installed ` +
        `and the Equilibrist dependencies are available.`
    );
    app.quit();
  });

  streamlitProcess.on("exit", (code) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const detail = stderrBuffer.trim()
        ? `\nLast output:\n${stderrBuffer.trim()}`
        : "";
      dialog.showErrorBox(
        "Equilibrist – server stopped",
        `The Streamlit process exited unexpectedly (code ${code}).\n` +
          `Python: ${python}\n` +
          `App: ${appPy}${detail}`
      );
      app.quit();
    }
  });
}

// ---------------------------------------------------------------------------
// Wait for Streamlit to be ready (poll the TCP port)
// ---------------------------------------------------------------------------
function waitForServer(port, retries = 150, delayMs = 200) {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    function tryConnect() {
      attempts++;
      const sock = new net.Socket();
      sock
        .once("connect", () => {
          sock.destroy();
          resolve();
        })
        .once("error", () => {
          sock.destroy();
          if (attempts >= retries) {
            reject(new Error(`Streamlit did not start within ${retries * delayMs}ms`));
          } else {
            setTimeout(tryConnect, delayMs);
          }
        });
      sock.connect(port, "127.0.0.1");
    }

    tryConnect();
  });
}

// ---------------------------------------------------------------------------
// Create the application window
// ---------------------------------------------------------------------------
function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 900,
    title: "Equilibrist",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "splash.html"));

  waitForServer(port)
    .then(() => {
      mainWindow.loadURL(`http://127.0.0.1:${port}`);
    })
    .catch((err) => {
      dialog.showErrorBox("Equilibrist – timeout", err.message);
      app.quit();
    });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------
app.whenReady().then(async () => {
  const port = await getFreePort();
  startStreamlit(port);
  createWindow(port);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow(port);
    }
  });
});

app.on("window-all-closed", () => {
  killStreamlit();
  app.quit();
});

app.on("before-quit", () => {
  killStreamlit();
});

function killStreamlit() {
  if (streamlitProcess && !streamlitProcess.killed) {
    streamlitProcess.kill("SIGTERM");
    const forceKill = setTimeout(() => {
      if (streamlitProcess && !streamlitProcess.killed) {
        streamlitProcess.kill("SIGKILL");
      }
    }, 3000);
    streamlitProcess.on("exit", () => clearTimeout(forceKill));
  }
}
