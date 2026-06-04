// Pyodide driver: boots Pyodide once under Node, mounts the run's workspace as
// the only host directory it can see (deny-by-default isolation, no network at
// runtime), then processes line-delimited JSON commands on stdin:
//   {"cmd":"run"}                  -> execute /workspace/_run.py
//   {"cmd":"install","package":x}  -> micropip.install(x)
// Each result is emitted as one line prefixed with RESULT_SENTINEL.
//
// Usage: node pyodide_driver.mjs <workspace_dir> <pyodide_node_modules_dir>
import readline from "node:readline";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SENTINEL = "@@RESULT@@";
const [workspace, indexURL] = process.argv.slice(2);

// Import Pyodide by absolute path so resolution doesn't depend on where the
// `pyodide` package lives relative to this driver file.
const { loadPyodide } = await import(pathToFileURL(path.join(indexURL, "pyodide.mjs")).href);
const pyodide = await loadPyodide({ indexURL });
// Mount the host workspace as the only writable host path the code can touch.
pyodide.FS.mkdirTree("/workspace");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: workspace }, "/workspace");
pyodide.FS.chdir("/workspace");
// Pre-load the agent's standard toolkit. The slim npm package fetches these
// wheels from the Pyodide CDN on first use; report status instead of failing.
let pkgStatus = "ok";
try {
  await pyodide.loadPackage(["pandas", "openpyxl", "matplotlib", "micropip"]);
} catch (e) {
  pkgStatus = "unavailable: " + String(e.message || e);
}
await pyodide.runPythonAsync("import os; os.environ['MPLBACKEND']='Agg'");

process.stdout.write(SENTINEL + JSON.stringify({ ready: true, packages: pkgStatus }) + "\n");

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  let req;
  try { req = JSON.parse(line); } catch { continue; }
  let out = "", err = "", code = 0;
  pyodide.setStdout({ batched: (s) => { out += s + "\n"; } });
  pyodide.setStderr({ batched: (s) => { err += s + "\n"; } });
  try {
    if (req.cmd === "run") {
      const src = pyodide.FS.readFile("/workspace/_run.py", { encoding: "utf8" });
      await pyodide.runPythonAsync(src);
    } else if (req.cmd === "install") {
      const mp = pyodide.pyimport("micropip");
      await mp.install(req.package);
    }
  } catch (e) {
    err += String(e.message || e);
    code = 1;
  }
  process.stdout.write(SENTINEL + JSON.stringify({ stdout: out, stderr: err, exit_code: code }) + "\n");
}
