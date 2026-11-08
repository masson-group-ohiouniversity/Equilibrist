"""
equilibrist_hpc.py — run Equilibrist's parallel tools on a SLURM cluster.

Every expensive tool in Equilibrist (1D / 2D RMSE profiles, bootstrap,
jackknife, Monte Carlo, local sensitivity) is a list of independent refits
handed to ``joblib.Parallel``.  This module moves that list to a cluster
without touching the tools themselves:

  1. CAPTURE   (laptop)   The tool functions are wrapped by
     ``install_offloadable``.  When the app "arms" a capture, the next call
     to a wrapped function does not run: it raises ``CapturedCall`` carrying
     the function name and its arguments.
  2. SHIP      (laptop)   The call is written as portable tagged JSON
     (``dumps`` — no pickle crosses machines), the ``equilibrist_*.py``
     sources are synced, and a batch script is submitted with ``sbatch``.
  3. RUN       (cluster)  ``srun`` starts N copies of this file
     (``python equilibrist_hpc.py run job.json``).  Each copy calls the very
     same tool function, but ``joblib.Parallel`` is replaced by
     ``ShardedParallel``: rank r computes tasks r, r+N, r+2N, … , writes
     each result to the shared job directory, then waits until every task
     is present and returns the complete, ordered list.  All ranks therefore
     finish the function with the full result; rank 0 writes ``result.json``.
  4. FETCH     (laptop)   ``result.json`` is downloaded, decoded, and handed
     to the app exactly as if the tool had run locally.

Because the tool's own code path is unchanged, a cluster result is the same
object the local run would have produced.

Nothing here imports Streamlit; the UI lives in ``equilibrist_hpc_ui.py``.
Paramiko is imported lazily, only when an SSH connection is opened.

© Eric Masson, Ohio University, 2026
"""
from __future__ import annotations

import base64
import functools
import hashlib
import importlib
import inspect
import io
import json
import os
import pickle
import shlex
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = [
    "dumps", "loads", "CapturedCall", "arm_capture", "disarm_capture",
    "install_offloadable", "ShardedParallel", "install_sharded_parallel",
    "driver_main", "render_batch_script", "PRESETS", "default_settings",
    "SSHRemote", "LocalRemote", "UnknownHostKey", "HostKeyMismatch",
    "HPCError", "load_config", "save_config", "load_jobs", "save_jobs",
    "build_job", "submit_job", "job_status", "fetch_result", "cancel_job",
    "resume_job", "walltime_seconds",
    "install_ssh_key", "start_env_install", "check_env_install",
    "config_dir",
]

# Modules whose functions may be executed by the cluster driver.  The job
# file names a (module, function) pair; anything outside this list is refused
# so a tampered job file cannot make the driver import arbitrary code.
ALLOWED_MODULES = ("equilibrist_bootstrap", "equilibrist_diagnostics", "equilibrist_global")


class HPCError(RuntimeError):
    """User-facing failure in the cluster layer."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. Portable serialisation (tagged JSON — no pickle between machines)
# ═══════════════════════════════════════════════════════════════════════════
_TAGS = ("__nd__", "__ndobj__", "__tuple__", "__set__", "__frozenset__",
         "__bytes__", "__complex__", "__dictpairs__", "__datetime__",
         "__network__")


def _enc(o, path="root"):
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.complexfloating) or isinstance(o, complex):
        return {"__complex__": [float(o.real), float(o.imag)]}
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            return {"__ndobj__": {"shape": list(o.shape),
                                  "items": [_enc(v, f"{path}[{i}]")
                                            for i, v in enumerate(o.ravel().tolist())]}}
        a = np.ascontiguousarray(o)
        if a.dtype.byteorder == ">":                 # force little-endian
            a = a.astype(a.dtype.newbyteorder("<"))
        return {"__nd__": {"dtype": a.dtype.str, "shape": list(a.shape),
                           "b64": base64.b64encode(a.tobytes()).decode("ascii")}}
    if isinstance(o, dict):
        if callable(o.get("species_conc_fn")) and "prod_to_eq" in o:
            # A reaction network: its evaluator is a Python closure that is
            # fully determined by the parsed script, so it is dropped here and
            # rebuilt by ``loads`` from the parsed script in the same payload.
            return {"__network__": _enc({k: v for k, v in o.items()
                                         if k != "species_conc_fn"}, path)}
        plain = all(isinstance(k, str) for k in o) and not any(k in _TAGS for k in o)
        if plain:
            return {k: _enc(v, f"{path}.{k}") for k, v in o.items()}
        return {"__dictpairs__": [[_enc(k, f"{path}<key>"), _enc(v, f"{path}[{k!r}]")]
                                  for k, v in o.items()]}
    if isinstance(o, list):
        return [_enc(v, f"{path}[{i}]") for i, v in enumerate(o)]
    if isinstance(o, tuple):
        return {"__tuple__": [_enc(v, f"{path}[{i}]") for i, v in enumerate(o)]}
    if isinstance(o, frozenset):
        return {"__frozenset__": [_enc(v, path) for v in sorted(o, key=repr)]}
    if isinstance(o, set):
        return {"__set__": [_enc(v, path) for v in sorted(o, key=repr)]}
    if isinstance(o, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(o)).decode("ascii")}
    if isinstance(o, datetime):
        return {"__datetime__": o.isoformat()}
    raise TypeError(f"Cannot send object of type {type(o).__name__} to the "
                    f"cluster (at {path}).")


def _dec(o):
    if isinstance(o, list):
        return [_dec(v) for v in o]
    if not isinstance(o, dict):
        return o
    if len(o) == 1:
        (k, v), = o.items()
        if k == "__nd__":
            raw = base64.b64decode(v["b64"])
            return np.frombuffer(raw, dtype=np.dtype(v["dtype"])).reshape(v["shape"]).copy()
        if k == "__ndobj__":
            arr = np.empty(len(v["items"]), dtype=object)
            for i, it in enumerate(v["items"]):
                arr[i] = _dec(it)
            return arr.reshape(v["shape"])
        if k == "__tuple__":
            return tuple(_dec(x) for x in v)
        if k == "__set__":
            return set(_dec(x) for x in v)
        if k == "__frozenset__":
            return frozenset(_dec(x) for x in v)
        if k == "__bytes__":
            return base64.b64decode(v)
        if k == "__complex__":
            return complex(v[0], v[1])
        if k == "__datetime__":
            return datetime.fromisoformat(v)
        if k == "__dictpairs__":
            return {_hashable(_dec(kk)): _dec(vv) for kk, vv in v}
        if k == "__network__":
            net = _dec(v)
            net["species_conc_fn"] = None
            _pending_networks.append(net)
            return net
    return {k: _dec(v) for k, v in o.items()}


_pending_networks: list = []


def _find_parsed(o, depth=0):
    """First dict that looks like a parsed Equilibrist script."""
    if depth > 4:
        return None
    if isinstance(o, dict):
        if "equilibria" in o and "concentrations" in o and "titrant" in o:
            return o
        for v in o.values():
            r = _find_parsed(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(o, (list, tuple)):
        for v in o:
            r = _find_parsed(v, depth + 1)
            if r is not None:
                return r
    return None


def _hashable(k):
    return tuple(_hashable(x) for x in k) if isinstance(k, list) else k


def dumps(obj) -> str:
    """Encode ``obj`` (nested dict / list / tuple / set / ndarray / scalars)
    as JSON that round-trips exactly through ``loads``."""
    return json.dumps(_enc(obj), allow_nan=True)


def loads(text: str):
    del _pending_networks[:]
    obj = _dec(json.loads(text))
    if _pending_networks:
        nets = list(_pending_networks); del _pending_networks[:]
        parsed = _find_parsed(obj)
        if parsed is not None:
            from equilibrist_network import build_network
            fn = build_network(parsed)["species_conc_fn"]
            for net in nets:
                net["species_conc_fn"] = fn
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# 2. Capturing a tool call instead of running it
# ═══════════════════════════════════════════════════════════════════════════
class CapturedCall(BaseException):
    """Raised by an offloadable function while a capture is armed.

    Derives from ``BaseException`` so the app's ordinary
    ``except Exception`` handlers around each tool do not swallow it.
    """

    def __init__(self, module: str, function: str, args: tuple, kwargs: dict,
                 site: str = ""):
        super().__init__(f"captured {module}.{function}")
        self.module, self.function = module, function
        self.args_, self.kwargs_ = tuple(args), dict(kwargs)
        self.site = site


_tls = threading.local()


def arm_capture(site: str = "") -> None:
    """The next offloadable call on THIS thread is captured, not executed.
    Thread-local, so concurrent Streamlit sessions cannot affect each other."""
    _tls.armed = True
    _tls.site = site


def disarm_capture() -> None:
    _tls.armed = False
    _tls.site = ""


def _is_armed() -> bool:
    return bool(getattr(_tls, "armed", False))


def _make_offloadable(func: Callable, module_name: str) -> Callable:
    if getattr(func, "_equilibrist_offloadable", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _is_armed():
            site = getattr(_tls, "site", "")
            disarm_capture()                      # one shot
            raise CapturedCall(module_name, func.__name__, args, kwargs, site)
        return func(*args, **kwargs)

    wrapper._equilibrist_offloadable = True
    return wrapper


def install_offloadable(module_globals: dict, names, module_name: str) -> None:
    """Wrap the named functions of a module in place.  Called at the bottom
    of ``equilibrist_bootstrap`` and ``equilibrist_diagnostics``."""
    for n in names:
        f = module_globals.get(n)
        if callable(f):
            module_globals[n] = _make_offloadable(f, module_name)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ShardedParallel — joblib.Parallel across SLURM ranks
# ═══════════════════════════════════════════════════════════════════════════
class _ShardState:
    dir: Optional[Path] = None
    rank: int = 0
    size: int = 1
    call_index: int = 0
    poll_max: float = 5.0
    wait_limit_s: float = 7 * 24 * 3600.0


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


class ShardedParallel:
    """Drop-in for ``joblib.Parallel`` inside the cluster driver.

    ``Parallel(...)(delayed(f)(a) for a in items)`` becomes: materialise the
    task list (identical on every rank), compute the tasks assigned to this
    rank (index % size == rank), store each result in the shared job
    directory, wait until all results exist, return them in order.

    Results are exchanged as pickles: they never leave the cluster, and all
    ranks run the same interpreter and code, so pickle is safe and general.
    """

    def __init__(self, n_jobs=None, return_as: str = "list", **_ignored):
        self.return_as = return_as

    def __call__(self, iterable):
        S = _ShardState
        tasks = list(iterable)
        n = len(tasks)
        k = S.call_index
        S.call_index += 1
        cdir = S.dir / f"call_{k:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        if S.rank == 0:
            _atomic_write_bytes(cdir / "N", str(n).encode())

        for idx in range(S.rank, n, max(S.size, 1)):
            out = cdir / f"r_{idx:07d}.pkl"
            if out.exists():                       # resumed job
                continue
            f, a, kw = tasks[idx]
            try:
                res = f(*a, **kw)
            except Exception:
                _write_error(f"task {idx} of parallel call {k}")
                raise
            _atomic_write_bytes(out, pickle.dumps(res, protocol=4))

        # ── barrier: wait for every rank's results ───────────────────────
        t0, delay = time.time(), 0.5
        while True:
            present = sum(1 for _ in cdir.glob("r_*.pkl"))
            if present >= n:
                break
            if any(S.dir.parent.glob("ERROR*")):
                raise HPCError("another rank failed — see the ERROR file in "
                               "the job directory")
            if time.time() - t0 > S.wait_limit_s:
                raise HPCError(f"timed out waiting for {n - present} task "
                               f"results from other ranks")
            time.sleep(delay)
            delay = min(delay * 1.5, S.poll_max)

        results = []
        for idx in range(n):
            p = cdir / f"r_{idx:07d}.pkl"
            for _try in range(20):                 # tolerate NFS lag
                try:
                    with open(p, "rb") as fh:
                        results.append(pickle.load(fh))
                    break
                except (EOFError, FileNotFoundError, pickle.UnpicklingError):
                    time.sleep(0.5)
            else:
                raise HPCError(f"could not read result {p.name}")
        if str(self.return_as).startswith("generator"):
            return iter(results)
        return results


def _delayed(function):
    """Minimal ``joblib.delayed`` used when joblib is absent on the cluster."""
    def _d(*args, **kwargs):
        return function, args, kwargs
    return _d


def install_sharded_parallel(shard_dir, rank: int, size: int) -> None:
    """Route every ``from joblib import Parallel, delayed`` to the sharded
    implementation (creating a stand-in ``joblib`` module if needed)."""
    S = _ShardState
    S.dir, S.rank, S.size, S.call_index = Path(shard_dir), int(rank), int(size), 0
    S.dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.Parallel = ShardedParallel
        if not hasattr(joblib, "delayed"):
            joblib.delayed = _delayed
    except Exception:
        import types
        fake = types.ModuleType("joblib")
        fake.Parallel, fake.delayed = ShardedParallel, _delayed
        sys.modules["joblib"] = fake


def _write_error(context: str) -> None:
    try:
        S = _ShardState
        p = S.dir.parent / f"ERROR_rank{S.rank}.txt"
        _atomic_write_bytes(p, (f"{context}\n\n" + traceback.format_exc()).encode())
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. The cluster-side driver
# ═══════════════════════════════════════════════════════════════════════════
def _rank_and_size(rank=None, size=None):
    env = os.environ
    if rank is None:
        rank = env.get("EQUILIBRIST_RANK", env.get("SLURM_PROCID", env.get("PMI_RANK", 0)))
    if size is None:
        size = env.get("EQUILIBRIST_SIZE", env.get("SLURM_NTASKS", env.get("PMI_SIZE", 1)))
    return int(rank), int(size)


def driver_main(job_path, rank=None, size=None) -> int:
    job_path = Path(job_path).resolve()
    job_dir = job_path.parent
    rank, size = _rank_and_size(rank, size)
    install_sharded_parallel(job_dir / "shards", rank, size)
    try:
        job = loads(job_path.read_text())
        mod_name, fn_name = job["module"], job["function"]
        if mod_name not in ALLOWED_MODULES:
            raise HPCError(f"module {mod_name!r} is not allowed")
        code_dir = str(Path(__file__).resolve().parent)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        func = getattr(importlib.import_module(mod_name), fn_name)
        args, kwargs = list(job["args"]), dict(job["kwargs"])
        sig = None
        try:
            sig = inspect.signature(func)
            if "n_jobs" in sig.parameters:
                kwargs["n_jobs"] = -1          # take the joblib branch → sharded
            if "progress_callback" in sig.parameters:
                kwargs["progress_callback"] = None
            if "progress" in sig.parameters and rank == 0:
                # a fit reports (fraction, message): written to progress.txt for the
                # follower in the app, at most every few seconds
                _last = [0.0]
                def _report(frac, msg, _dir=job_dir):
                    now = time.time()
                    if now - _last[0] < 5.0:
                        return
                    _last[0] = now
                    try:
                        _atomic_write_bytes(_dir / "progress.txt",
                                            f"{float(frac):.3f}\t{str(msg)[:160]}".encode())
                    except Exception:
                        pass
                kwargs["progress"] = _report
        except (TypeError, ValueError):
            pass
        if "n_jobs" not in getattr(sig, "parameters", {}) and rank != 0:
            return 0                       # a plain fit: one task does the work, the others idle
        t0 = time.time()
        result = func(*args, **kwargs)
        if rank == 0:
            payload = dumps({"result": result,
                             "meta": {"ranks": size,
                                      "wall_seconds": time.time() - t0,
                                      "python": sys.version.split()[0],
                                      "numpy": np.__version__,
                                      "finished": datetime.now().isoformat(timespec="seconds")}})
            _atomic_write_bytes(job_dir / "result.json", payload.encode())
            _atomic_write_bytes(job_dir / "DONE", b"ok\n")
        return 0
    except BaseException:
        _write_error("driver")
        traceback.print_exc()
        return 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Settings, presets, batch script
# ═══════════════════════════════════════════════════════════════════════════
PRESETS = {
    "OSC (Pitzer / Cardinal / Ascend)": {
        "host": "pitzer.osc.edu",
        "conda_cmds": "module load miniconda3/24.1.2-py310",
        "setup_cmds": "module load miniconda3/24.1.2-py310\nsource activate equilibrist",
        "extra_sbatch": "",
        "cpus_per_node": 40,
    },
    "Generic SLURM + conda": {
        "host": "",
        "conda_cmds": "# make `conda` available, e.g.:\n# module load anaconda3",
        "setup_cmds": "# module load anaconda3\nsource activate equilibrist",
        "extra_sbatch": "",
        "cpus_per_node": 32,
    },
}


def default_settings() -> dict:
    s = {"preset": "OSC (Pitzer / Cardinal / Ascend)", "port": 22, "username": "",
         "account": "", "partition": "", "remote_base": "equilibrist_hpc",
         "python_cmd": "python", "env_name": "equilibrist",
         "nodes": 2, "walltime": "01:00:00"}
    s.update(PRESETS[s["preset"]])
    return s


_BATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cpus}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --output=slurm-%j.out
{optional_header}
# -- Equilibrist cluster job {job_id} | {tool} --
# One core per task: keep the numerical libraries single-threaded so the
# {ntasks} refit workers do not oversubscribe the nodes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

{setup_cmds}

cd "{job_dir_abs}"
srun {python_cmd} -u "{code_dir_abs}/equilibrist_hpc.py" run job.json
"""


def render_batch_script(settings: dict, *, job_id: str, tool: str,
                        job_dir_abs: str, code_dir_abs: str) -> str:
    nodes = int(settings.get("nodes", 1))
    cpus = int(settings.get("cpus_per_node", 1))
    header = []
    if str(settings.get("account", "")).strip():
        header.append(f"#SBATCH --account={settings['account'].strip()}")
    if str(settings.get("partition", "")).strip():
        header.append(f"#SBATCH --partition={settings['partition'].strip()}")
    for line in str(settings.get("extra_sbatch", "")).splitlines():
        line = line.strip()
        if line and not line.startswith("#SBATCH"):
            line = "#SBATCH " + line
        if line:
            header.append(line)
    return _BATCH_TEMPLATE.format(
        job_name=f"eqb_{job_id[9:]}", nodes=nodes, cpus=cpus,
        walltime=str(settings.get("walltime", "01:00:00")).strip(),
        optional_header="\n".join(header), job_id=job_id, tool=tool,
        ntasks=nodes * cpus,
        setup_cmds=str(settings.get("setup_cmds", "")).strip(),
        job_dir_abs=job_dir_abs, code_dir_abs=code_dir_abs,
        python_cmd=str(settings.get("python_cmd", "python")).strip() or "python")


# ── local configuration and job records (never contain secrets) ───────────
def config_dir() -> Path:
    d = Path(os.environ.get("EQUILIBRIST_HOME", Path.home() / ".equilibrist"))
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


_SECRET_KEYS = ("password", "otp", "passphrase")


def load_config() -> dict:
    s = default_settings()
    p = config_dir() / "hpc_config.json"
    if p.exists():
        try:
            s.update({k: v for k, v in json.loads(p.read_text()).items()
                      if k not in _SECRET_KEYS})
        except Exception:
            pass
    return s


def save_config(settings: dict) -> None:
    clean = {k: v for k, v in settings.items() if k not in _SECRET_KEYS}
    (config_dir() / "hpc_config.json").write_text(json.dumps(clean, indent=2))


def load_jobs() -> list:
    p = config_dir() / "hpc_jobs.json"
    if not p.exists():
        return []
    try:
        return list(json.loads(p.read_text()))
    except Exception:
        return []


def save_jobs(jobs: list) -> None:
    (config_dir() / "hpc_jobs.json").write_text(json.dumps(jobs[-200:], indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Remote transports
# ═══════════════════════════════════════════════════════════════════════════
class UnknownHostKey(HPCError):
    def __init__(self, host, key_type, fingerprint):
        super().__init__(f"Host {host} is not known yet "
                         f"({key_type} key, fingerprint SHA256:{fingerprint}).")
        self.host, self.key_type, self.fingerprint = host, key_type, fingerprint


class HostKeyMismatch(HPCError):
    pass


def _sha256_fp(key) -> str:
    return base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


class _RemoteBase:
    """Commands run through a login shell so ``module``/``sbatch`` resolve."""

    def run(self, cmd: str, timeout: float = 120.0):
        raise NotImplementedError

    def sh(self, cmd: str, timeout: float = 120.0) -> str:
        rc, out, err = self.run(cmd, timeout=timeout)
        if rc != 0:
            raise HPCError(f"remote command failed ({rc}): {cmd}\n{(err or out).strip()[-2000:]}")
        return out

    def abspath(self, rel: str) -> str:
        if rel.startswith("/"):
            return rel
        home = self.sh("echo $HOME").strip().splitlines()[-1]
        return home.rstrip("/") + "/" + rel


class SSHRemote(_RemoteBase):
    """SSH + SFTP via Paramiko.  Authentication is attempted in this order:
    SSH agent keys, Equilibrist's own key (``~/.equilibrist/id_rsa``),
    unencrypted default keys in ``~/.ssh``, then the password.  If the server
    asks for anything further (a second factor), the prompts are answered
    with ``otp``.  The password is used only inside ``connect`` and is not
    stored on the object."""

    def __init__(self):
        self.t = None
        self._sftp = None
        self.host = self.username = ""
        self.auth_method = ""

    # ── connection ───────────────────────────────────────────────────────
    @staticmethod
    def own_key_path() -> Path:
        return config_dir() / "id_rsa"

    @staticmethod
    def own_known_hosts() -> Path:
        return config_dir() / "known_hosts"

    def connect(self, host, username, *, password=None, otp=None, port=22,
                trust_new_host=False, timeout=20.0):
        try:
            import paramiko
        except ImportError as e:
            raise HPCError("The 'paramiko' package is required for cluster "
                           "access.  Install it with:  pip install paramiko") from e
        self.close()
        try:
            sock = socket.create_connection((host, int(port)), timeout=timeout)
        except OSError as e:
            raise HPCError(f"Cannot reach {host}:{port} — {e}") from e
        t = paramiko.Transport(sock)
        t.banner_timeout = timeout
        try:
            t.start_client(timeout=timeout)
            self._verify_host_key(paramiko, t, host, int(port), trust_new_host)
            self._authenticate(paramiko, t, username, password, otp)
        except BaseException:
            t.close()
            raise
        t.set_keepalive(30)
        self.t, self.host, self.username = t, host, username
        return self

    def _verify_host_key(self, paramiko, t, host, port, trust_new):
        key = t.get_remote_server_key()
        name = host if port == 22 else f"[{host}]:{port}"
        stores = []
        for p in (Path.home() / ".ssh" / "known_hosts", self.own_known_hosts()):
            if p.exists():
                try:
                    hk = paramiko.HostKeys(); hk.load(str(p)); stores.append(hk)
                except Exception:
                    pass
        seen_other = False
        for hk in stores:
            entry = hk.lookup(name)
            if entry:
                known = entry.get(key.get_name())
                if known is not None:
                    if known.asbytes() == key.asbytes():
                        return
                    raise HostKeyMismatch(
                        f"The host key of {host} has CHANGED since it was last "
                        f"trusted.  This can indicate an attack.  Do not enter "
                        f"your password; contact the cluster's support.")
                seen_other = True
        if not trust_new:
            raise UnknownHostKey(host, key.get_name(), _sha256_fp(key))
        own = paramiko.HostKeys()
        if self.own_known_hosts().exists():
            own.load(str(self.own_known_hosts()))
        own.add(name, key.get_name(), key)
        own.save(str(self.own_known_hosts()))

    def _authenticate(self, paramiko, t, username, password, otp):
        def handler(title, instructions, prompts):
            answers = []
            for text, _echo in prompts:
                answers.append((password or "") if "assword" in text else (otp or ""))
            return answers

        def finish(remaining, label):
            """``remaining`` non-empty ⇒ server wants a further factor."""
            if remaining and not t.is_authenticated():
                if "keyboard-interactive" in remaining:
                    t.auth_interactive(username, handler)
                elif "password" in remaining and password:
                    t.auth_password(username, password, fallback=False)
            if t.is_authenticated():
                self.auth_method = label
                return True
            return False

        keys = []
        try:
            keys += [(k, "SSH agent") for k in paramiko.Agent().get_keys()]
        except Exception:
            pass
        candidates = [(self.own_key_path(), "Equilibrist key")]
        for n in ("id_ed25519", "id_ecdsa", "id_rsa"):
            candidates.append((Path.home() / ".ssh" / n, f"~/.ssh/{n}"))
        for p, label in candidates:
            if not p.exists():
                continue
            for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
                try:
                    keys.append((cls.from_private_key_file(str(p)), label)); break
                except Exception:
                    continue                      # wrong type or passphrase-protected
        for k, label in keys:
            try:
                if finish(t.auth_publickey(username, k), label):
                    return
            except paramiko.SSHException:
                continue
        if password:
            try:
                if finish(t.auth_password(username, password, fallback=False), "password"):
                    return
            except paramiko.BadAuthenticationType as e:
                if "keyboard-interactive" in (e.allowed_types or []):
                    try:
                        t.auth_interactive(username, handler)
                        if t.is_authenticated():
                            self.auth_method = "password (interactive)"; return
                    except paramiko.SSHException:
                        pass
            except paramiko.SSHException:
                pass
        raise HPCError("Login failed.  Check the username and password"
                       + ("" if otp else ", and whether the cluster expects a "
                          "second-factor response") + ".")

    @property
    def connected(self) -> bool:
        return bool(self.t is not None and self.t.is_active())

    def close(self):
        for x in (self._sftp, self.t):
            try:
                if x is not None: x.close()
            except Exception:
                pass
        self._sftp = self.t = None

    # ── operations ───────────────────────────────────────────────────────
    def run(self, cmd, timeout=120.0):
        if not self.connected:
            raise HPCError("Not connected to the cluster.")
        ch = self.t.open_session(timeout=timeout)
        ch.settimeout(timeout)
        ch.exec_command("bash -lc " + shlex.quote(cmd))
        out = ch.makefile("rb").read().decode("utf-8", "replace")
        err = ch.makefile_stderr("rb").read().decode("utf-8", "replace")
        rc = ch.recv_exit_status()
        ch.close()
        return rc, out, err

    @property
    def sftp(self):
        import paramiko
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self.t)
        return self._sftp

    def put_bytes(self, remote_path: str, data: bytes, mode: int = 0o644):
        with self.sftp.open(remote_path, "wb") as fh:
            fh.set_pipelined(True)
            fh.write(data)
        self.sftp.chmod(remote_path, mode)

    def get_bytes(self, remote_path: str) -> bytes:
        with self.sftp.open(remote_path, "rb") as fh:
            fh.prefetch()
            return fh.read()

    def exists(self, remote_path: str) -> bool:
        try:
            self.sftp.stat(remote_path); return True
        except IOError:
            return False


class LocalRemote(_RemoteBase):
    """Same interface, but 'remote' is a directory on this machine.  Used by
    the test-suite (with stand-in ``sbatch``/``srun``) and handy for trying
    the whole workflow without a cluster."""

    def __init__(self, home, env: Optional[dict] = None):
        self.home = Path(home).resolve(); self.home.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ); self.env.update(env or {})
        self.env["HOME"] = str(self.home)
        self.host, self.username, self.auth_method = "localhost", "local", "local"
        self.connected = True

    def _p(self, remote_path):
        p = Path(remote_path)
        return p if p.is_absolute() else self.home / p

    def run(self, cmd, timeout=120.0):
        r = subprocess.run(["bash", "-c", cmd], cwd=self.home, env=self.env,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def put_bytes(self, remote_path, data, mode=0o644):
        p = self._p(remote_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data); os.chmod(p, mode)

    def get_bytes(self, remote_path):
        return self._p(remote_path).read_bytes()

    def exists(self, remote_path):
        return self._p(remote_path).exists()

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 7. Job life-cycle
# ═══════════════════════════════════════════════════════════════════════════
def _code_files(app_dir: Path) -> list:
    return sorted(p for p in Path(app_dir).glob("equilibrist_*.py")
                  if not p.name.endswith("_ui.py"))


def _code_digest(files) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


def build_job(captured: CapturedCall, *, tool: str, site: str,
              context: Optional[dict] = None) -> dict:
    """Turn a captured call into a job record plus its JSON payload."""
    kwargs = {k: v for k, v in captured.kwargs_.items() if k != "progress_callback"}
    payload = dumps({"module": captured.module, "function": captured.function,
                     "args": list(captured.args_), "kwargs": kwargs,
                     "tool": tool, "created": datetime.now().isoformat(timespec="seconds")})
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    record = {"id": job_id, "tool": tool, "site": site,
              "function": f"{captured.module}.{captured.function}",
              "context": context or {}, "status": "NEW", "slurm_id": "",
              "created": datetime.now().isoformat(timespec="seconds")}
    return {"record": record, "payload": payload}


def submit_job(remote: _RemoteBase, job: dict, settings: dict, *,
               app_dir=None, dry_run: bool = False) -> dict:
    """Sync the code, stage the job, and ``sbatch`` it.  With ``dry_run`` the
    function returns the batch script without contacting the scheduler."""
    app_dir = Path(app_dir or Path(__file__).resolve().parent)
    rec = job["record"]
    files = _code_files(app_dir)
    digest = _code_digest(files)
    base = str(settings.get("remote_base", "equilibrist_hpc")).strip() or "equilibrist_hpc"
    code_rel = f"{base}/code_{digest}"
    job_rel = f"{base}/jobs/{rec['id']}"

    if dry_run:
        return {"script": render_batch_script(
            settings, job_id=rec["id"], tool=rec["tool"],
            job_dir_abs=f"$HOME/{job_rel}", code_dir_abs=f"$HOME/{code_rel}"),
            "n_code_files": len(files), "payload_bytes": len(job["payload"])}

    base_abs = remote.abspath(base)
    code_abs, job_abs = f"{base_abs}/code_{digest}", f"{base_abs}/jobs/{rec['id']}"
    remote.sh(f"mkdir -p {shlex.quote(code_abs)} {shlex.quote(job_abs)}")
    if not remote.exists(f"{code_abs}/.complete"):
        # One archive instead of one SFTP round trip per file: over a slow
        # link the per-file upload took about a minute after every code change.
        import io as _io, tarfile as _tarfile
        buf = _io.BytesIO()
        with _tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in files:
                tar.add(str(p), arcname=p.name)
        remote.put_bytes(f"{code_abs}/code.tgz", buf.getvalue())
        remote.sh(f"cd {shlex.quote(code_abs)} && tar xzf code.tgz && rm -f code.tgz && echo ok > .complete")
    remote.put_bytes(f"{job_abs}/job.json", job["payload"].encode())
    script = render_batch_script(settings, job_id=rec["id"], tool=rec["tool"],
                                 job_dir_abs=job_abs, code_dir_abs=code_abs)
    remote.put_bytes(f"{job_abs}/run.sh", script.encode(), mode=0o755)
    out = remote.sh(f"cd {shlex.quote(job_abs)} && sbatch --parsable run.sh")
    slurm_id = out.strip().splitlines()[-1].split(";")[0].strip()
    if not slurm_id.isdigit():
        raise HPCError(f"sbatch did not return a job id: {out.strip()[-500:]}")
    rec.update({"slurm_id": slurm_id, "status": "SUBMITTED", "job_dir": job_abs,
                "host": remote.host, "username": remote.username,
                "nodes": int(settings.get("nodes", 1)),
                "cpus_per_node": int(settings.get("cpus_per_node", 1)),
                "walltime": settings.get("walltime", ""),
                "code_digest": digest,
                "submitted": datetime.now().isoformat(timespec="seconds")})
    return rec


_FINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
          "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


def job_status(remote: _RemoteBase, rec: dict, scheduler: bool = True) -> dict:
    """Refresh ``rec`` in place: scheduler state, progress, error text.

    With ``scheduler=False`` only the job folder is looked at (result
    marker, error reports, number of finished refits).  That is a plain file
    listing, cheap enough to repeat every few seconds, whereas ``squeue`` /
    ``sacct`` load the scheduler and should be called sparingly."""
    jd, sid = rec.get("job_dir", ""), rec.get("slurm_id", "")
    q = shlex.quote
    sched = (
        f"echo \"@@Q $(squeue -h -j {q(sid)} -o %T 2>/dev/null | head -1)\"; "
        f"echo \"@@A $(sacct -n -X -j {q(sid)} -o State 2>/dev/null | head -1)\"; "
    ) if scheduler else ""
    script = (
        f"cd {q(jd)} 2>/dev/null || {{ echo '@@NODIR'; exit 0; }}; " + sched +
        # Look at the folder through a LISTING, not through "does file X
        # exist?".  On a network filesystem the login node caches a negative
        # answer to the latter for a while, so a result written by a compute
        # node a moment ago can seem absent; opening the directory forces a
        # fresh look.  result.json is written before DONE and is moved into
        # place atomically, so either one proves the job is complete.
        "LS=$(ls -a 2>/dev/null); ls -a shards >/dev/null 2>&1; "
        "echo \"$LS\" | grep -qx 'DONE' && echo '@@DONE'; "
        "echo \"$LS\" | grep -qx 'result.json' && echo '@@DONE'; "
        "echo \"$LS\" | grep -q '^ERROR' && echo '@@ERROR'; "
        # the scheduler creates the output file when the job starts running
        f"echo \"$LS\" | grep -qx {q('slurm-' + sid + '.out')} && echo '@@STARTED'; "
        "tot=0; for f in shards/call_*/N; do [ -f \"$f\" ] && tot=$((tot+$(cat \"$f\"))); done; "
        "echo \"@@TOT $tot\"; "
        "echo \"@@CNT $(find shards -name 'r_*.pkl' 2>/dev/null | wc -l)\"; "
        "echo \"@@PROG $(head -c 200 progress.txt 2>/dev/null | tr '\\n\\t' '  ')\"")
    rc, out, err = remote.run(script)
    info = {}
    for line in out.splitlines():
        if line.startswith("@@"):
            k, _, v = line[2:].partition(" ")
            info[k] = v.strip()
    if "NODIR" in info:
        rec["status"] = "MISSING"
        return rec
    state = (info.get("Q") or info.get("A") or "").split()[0:1]
    state = state[0].rstrip("+") if state else ""
    if "DONE" in info:
        rec["status"] = "DONE"
        rec["error"] = ""                  # drop any earlier, mistaken diagnosis
    elif "ERROR" in info:
        rec["status"] = "ERROR"
    elif not scheduler:
        # folder-only look: the output file (or refits) appearing means the
        # job has started, without waiting for the next scheduler query
        if ("STARTED" in info or int(info.get("CNT", "0") or 0) > 0) \
                and rec.get("status") in ("SUBMITTED", "PENDING", "CONFIGURING"):
            rec["status"] = "RUNNING"
    elif state:
        if state == "COMPLETED":
            # Finished, yet no result visible.  Before calling that a failure,
            # allow the filesystem time to show files written in the job's
            # last second; keep polling meanwhile.
            first = rec.get("completed_seen")
            if not first:
                rec["completed_seen"] = datetime.now().isoformat(timespec="seconds")
                rec["status"] = "COMPLETING"
            else:
                try:
                    waited = (datetime.now() - datetime.fromisoformat(first)).total_seconds()
                except ValueError:
                    waited = 1e9
                rec["status"] = "COMPLETING" if waited < 90 else "ERROR"
        elif state.startswith("CANCEL"):
            rec["status"] = "CANCELLED"        # by the user or an administrator
        elif state in ("TIMEOUT", "DEADLINE"):
            rec["status"] = "TIMEOUT"          # wall time exhausted
        elif state in _FINAL:
            rec["status"] = "FAILED"
        else:
            rec["status"] = state
    try:
        rec["progress"] = [int(info.get("CNT", "0") or 0), int(info.get("TOT", "0") or 0)]
    except ValueError:
        rec["progress"] = [0, 0]
    rec["progress_text"] = (info.get("PROG") or "").strip()      # "<fraction> <message>" from a fit
    if scheduler:
        rec["scheduler_state"] = state
        rec["checked_scheduler"] = datetime.now().isoformat(timespec="seconds")
    if rec["status"] in ("ERROR", "FAILED", "TIMEOUT", "CANCELLED"):
        # Collect everything needed to diagnose a failure in one round trip:
        # the scheduler's own record (state, exit code, reason), the content
        # of the job folder, the driver's error report and the job output
        # (also looked for in $HOME, where some sites put it).
        diag = (
            f"cd {q(jd)} || exit 0; "
            "echo '--- scheduler record ---'; "
            f"sacct -X -j {q(sid)} -o JobID,State,ExitCode,Elapsed,Start,End,NodeList%24,Reason%40 2>&1 | tail -n 3; "
            f"scontrol show job {q(sid)} 2>/dev/null | tr ' ' '\n' | "
            "grep -E '^(JobState|Reason|ExitCode|RunTime|TimeLimit|Partition|NumNodes|NumCPUs|StdOut|StdErr|WorkDir)=' ; "
            "echo '--- job folder ---'; ls -la | tail -n +2; "
            "echo '--- error report ---'; cat ERROR* 2>/dev/null | tail -n 30; "
            "echo '--- scheduler output ---'; "
            f"for f in slurm-*.out \"$HOME/slurm-{sid}.out\"; do "
            "[ -f \"$f\" ] && { echo \"[$f]\"; tail -n 40 \"$f\"; }; done; true")
        rc, tail, _ = remote.run(diag)
        hint = ""
        if state == "COMPLETED":
            hint = ("The scheduler reports the job as COMPLETED, but no result "
                    "file was written.\n")
        rec["error"] = (hint + tail.strip())[-6000:]
    rec["checked"] = datetime.now().isoformat(timespec="seconds")
    return rec


def fetch_result(remote: _RemoteBase, rec: dict):
    """Download and decode the finished result.  Returns ``(result, meta)``."""
    path = f"{rec['job_dir']}/result.json"
    if not remote.exists(path):
        raise HPCError("The job has not produced a result yet.")
    data = loads(remote.get_bytes(path).decode("utf-8"))
    rec["status"] = "FETCHED"
    rec["meta"] = data.get("meta", {})
    return data["result"], data.get("meta", {})


def resume_job(remote: _RemoteBase, rec: dict, settings: dict) -> dict:
    """Submit a stopped job again in the SAME job folder.

    Every refit already stored under ``shards/`` is skipped by the driver,
    so a job that ran out of wall time (or was cancelled, or lost a node)
    continues where it stopped.  Nodes, CPUs and wall time are taken from
    ``settings`` and may differ from the first submission: tasks are matched
    to results by index, not by which rank computed them."""
    job_abs = rec.get("job_dir", "")
    digest = rec.get("code_digest", "")
    if not job_abs or not digest or "/jobs/" not in job_abs:
        raise HPCError("This job record is too old to be resumed.")
    base_abs = job_abs.rsplit("/jobs/", 1)[0]
    code_abs = f"{base_abs}/code_{digest}"
    if not remote.exists(f"{code_abs}/.complete") or not remote.exists(f"{job_abs}/job.json"):
        raise HPCError("The job's files are no longer on the cluster.")
    script = render_batch_script(settings, job_id=rec["id"], tool=rec.get("tool", ""),
                                 job_dir_abs=job_abs, code_dir_abs=code_abs)
    remote.put_bytes(f"{job_abs}/run.sh", script.encode(), mode=0o755)
    out = remote.sh(f"cd {shlex.quote(job_abs)} && rm -f DONE ERROR* result.json "
                    "&& sbatch --parsable run.sh")
    slurm_id = out.strip().splitlines()[-1].split(";")[0].strip()
    if not slurm_id.isdigit():
        raise HPCError(f"sbatch did not return a job id: {out.strip()[-500:]}")
    rec.setdefault("previous_slurm_ids", []).append(rec.get("slurm_id", ""))
    rec.update({"slurm_id": slurm_id, "status": "SUBMITTED",
                "nodes": int(settings.get("nodes", 1)),
                "cpus_per_node": int(settings.get("cpus_per_node", 1)),
                "walltime": settings.get("walltime", ""),
                "submitted": datetime.now().isoformat(timespec="seconds"),
                "running_since": None, "checked": None, "error": "",
                "scheduler_state": "", "auto_fetched": False})
    return rec


def walltime_seconds(text) -> Optional[int]:
    """Seconds in a SLURM time string (MM, MM:SS, HH:MM:SS, D-HH[:MM[:SS]]);
    ``None`` if it cannot be read."""
    try:
        t = str(text).strip()
        days = 0
        if "-" in t:
            d, t = t.split("-", 1); days = int(d)
            parts = [int(x) for x in t.split(":")] + [0, 0]
            h, m, sec = parts[0], parts[1], parts[2]
        else:
            parts = [int(x) for x in t.split(":")]
            if len(parts) == 1: h, m, sec = 0, parts[0], 0
            elif len(parts) == 2: h, m, sec = 0, parts[0], parts[1]
            else: h, m, sec = parts[0], parts[1], parts[2]
        return ((days * 24 + h) * 60 + m) * 60 + sec
    except Exception:
        return None


def cancel_job(remote: _RemoteBase, rec: dict) -> dict:
    remote.run(f"scancel {shlex.quote(rec.get('slurm_id', ''))}")
    rec["status"] = "CANCELLED"
    return rec


# ── one-time conveniences ─────────────────────────────────────────────────
def install_ssh_key(remote: SSHRemote) -> str:
    """Create ``~/.equilibrist/id_rsa`` (if absent) and authorise it on the
    cluster, so later connections need no password."""
    import paramiko
    kp = SSHRemote.own_key_path()
    if kp.exists():
        key = paramiko.RSAKey.from_private_key_file(str(kp))
    else:
        key = paramiko.RSAKey.generate(3072)
        key.write_private_key_file(str(kp))
        os.chmod(kp, stat.S_IRUSR | stat.S_IWUSR)
    comment = f"equilibrist@{socket.gethostname()}"
    pub = f"{key.get_name()} {key.get_base64()} {comment}"
    q = shlex.quote
    remote.sh("mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys "
              "&& chmod 600 ~/.ssh/authorized_keys && "
              f"(grep -qF {q(key.get_base64())} ~/.ssh/authorized_keys || "
              f"echo {q(pub)} >> ~/.ssh/authorized_keys)")
    return str(kp)


_ENV_YML = """name: {env}
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.24
  - scipy>=1.11
  - pandas>=2.0
  - joblib>=1.3
  - openpyxl>=3.1
  - matplotlib>=3.7
  - pillow>=10.0
  - plotly>=5.18
"""


def start_env_install(remote: _RemoteBase, settings: dict) -> str:
    """Create the conda environment on the cluster, in the background."""
    base_abs = remote.abspath(str(settings.get("remote_base", "equilibrist_hpc")))
    env = str(settings.get("env_name", "equilibrist")).strip() or "equilibrist"
    remote.sh(f"mkdir -p {shlex.quote(base_abs)}")
    remote.put_bytes(f"{base_abs}/environment_hpc.yml", _ENV_YML.format(env=env).encode())
    script = ("#!/bin/bash -l\n"
              f"cd {shlex.quote(base_abs)}\nrm -f INSTALL_DONE INSTALL_FAILED\n"
              f"{str(settings.get('conda_cmds', '')).strip()}\n"
              f"if conda env list | awk '{{print $1}}' | grep -qx {shlex.quote(env)}; then\n"
              f"  echo 'environment {env} already exists'\n"
              "else\n"
              f"  conda env create -n {shlex.quote(env)} -f environment_hpc.yml || {{ touch INSTALL_FAILED; exit 1; }}\n"
              "fi\ntouch INSTALL_DONE\n")
    remote.put_bytes(f"{base_abs}/install_env.sh", script.encode(), mode=0o755)
    remote.sh(f"cd {shlex.quote(base_abs)} && "
              "(nohup ./install_env.sh > install.log 2>&1 < /dev/null &) ; sleep 1; echo started")
    return base_abs


def check_env_install(remote: _RemoteBase, settings: dict) -> dict:
    base_abs = remote.abspath(str(settings.get("remote_base", "equilibrist_hpc")))
    q = shlex.quote(base_abs)
    rc, out, _ = remote.run(
        f"cd {q} 2>/dev/null || {{ echo '@@NONE'; exit 0; }}; "
        "[ -f INSTALL_DONE ] && echo '@@DONE'; [ -f INSTALL_FAILED ] && echo '@@FAILED'; "
        "echo '@@LOG'; tail -n 12 install.log 2>/dev/null")
    state = "running"
    if "@@NONE" in out: state = "not started"
    if "@@FAILED" in out: state = "failed"
    if "@@DONE" in out: state = "ready"
    return {"state": state, "log": out.split("@@LOG", 1)[-1].strip()}


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        # Run through the importable module, not this ``__main__`` copy, so
        # the tool modules and the driver share one module state.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import equilibrist_hpc as _hpc
        sys.exit(_hpc.driver_main(sys.argv[2]))
    print("usage: python equilibrist_hpc.py run <job.json>")
    sys.exit(2)
