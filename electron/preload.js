// preload.js — runs in the renderer before any page JS.
//
// Currently this is intentionally minimal.  It establishes a secure bridge
// between the Electron renderer process and the main process via
// contextBridge.  Expand this if you later need to expose native file
// dialogs, Plotly.toImage() capture, or other Electron-specific features.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("equilibrist", {
  // Example: request a native "Save File" dialog from the main process
  // saveFile: (data, defaultName) => ipcRenderer.invoke("save-file", data, defaultName),

  platform: process.platform,
});
