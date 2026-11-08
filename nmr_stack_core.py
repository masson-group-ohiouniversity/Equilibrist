"""
nmr_stack_core -- find, process and arrange 1D Bruker NMR experiments.

Standalone for now (no dependency on Equilibrist).  Used by ``nmr_stack.py``.

    experiments = find_experiments(root)                # walk a folder tree
    spec = process_experiment(exp, settings)            # FID -> spectrum
    reaction_time_h(exp.title)                          # "120C15min" -> 0.25

Processing from the FID reproduces TopSpin's own spectrum (correlation
0.9999 on two data sets) when the TopSpin phase values are used: exponential
window LB, zero-fill to SI, FFT of the whole FID, reversed so index 0 is the
high-ppm end, the digital-filter group delay taken out as a linear phase of
360 * GRPDLY degrees (TopSpin's way; truncating the FID instead leaves a
dome in the baseline), then ``exp(-i(PHC0 + PHC1 * k/SI))``, ppm axis from
OFFSET / SF / SW_p.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import nmrglue as ng
except ImportError as _e:                       # pragma: no cover
    ng = None
    _NG_ERR = _e


# ── Data model ──────────────────────────────────────────────────────────────
@dataclass
class Experiment:
    path: str                     # folder holding fid + acqus
    root: str                     # the folder the user gave (for relative names)
    expno: str                    # experiment number (folder name)
    title: str = ""
    date: Optional[datetime] = None
    ns: int = 0
    rg: float = 1.0
    nucleus: str = ""
    pulprog: str = ""
    sfo1: float = 0.0
    sw_h: float = 0.0
    td: int = 0
    temperature_K: float = 0.0
    pdata: Optional[str] = None   # pdata/<n> folder with procs (+ 1r), if any
    phc0: Optional[float] = None  # TopSpin phases, if a procs file exists
    phc1: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def relpath(self) -> str:
        r = os.path.relpath(self.path, self.root)
        return r if r != "." else os.path.basename(self.path)

    @property
    def label(self) -> str:
        return self.title or self.relpath


@dataclass
class Spectrum:
    ppm: np.ndarray               # descending (high ppm first), like TopSpin
    real: np.ndarray
    imag: np.ndarray
    exp: Experiment
    phc0: float
    phc1: float
    source: str                   # "fid" | "1r"
    raw: Optional[np.ndarray] = None   # real part before baseline correction


# ── Discovery ───────────────────────────────────────────────────────────────
def _is_1d_experiment(folder: str) -> bool:
    return (os.path.isfile(os.path.join(folder, "fid"))
            and os.path.isfile(os.path.join(folder, "acqus")))


def _read_jcamp(path: str) -> dict:
    """Minimal JCAMP-DX reader for acqus / procs (``##$KEY= value``)."""
    out = {}
    try:
        with open(path, "r", errors="replace") as fh:
            key = None
            for line in fh:
                line = line.rstrip("\n")
                m = re.match(r"^##\$?([^=]+)=\s*(.*)$", line)
                if m:
                    key, val = m.group(1).strip(), m.group(2).strip()
                    out[key] = val
                elif key is not None and not line.startswith("##") and not line.startswith("$$"):
                    out[key] = (out[key] + " " + line.strip()).strip()
    except OSError:
        pass
    return out


def _num(d: dict, key: str, default=0.0, cast=float):
    try:
        v = d.get(key)
        if v is None:
            return default
        v = str(v).strip("<>")
        return cast(v)
    except Exception:
        return default


def _read_experiment(folder: str, root: str) -> Optional[Experiment]:
    acq = _read_jcamp(os.path.join(folder, "acqus"))
    if not acq:
        return None
    # skip nD experiments (ser file / PARMODE > 0)
    if _num(acq, "PARMODE", 0, int) > 0 or os.path.isfile(os.path.join(folder, "ser")):
        return None
    exp = Experiment(path=folder, root=root, expno=os.path.basename(folder))
    exp.ns = _num(acq, "NS", 0, int)
    exp.rg = _num(acq, "RG", 1.0)
    exp.nucleus = str(acq.get("NUC1", "")).strip("<>")
    exp.pulprog = str(acq.get("PULPROG", "")).strip("<>")
    exp.sfo1 = _num(acq, "SFO1", 0.0)
    exp.sw_h = _num(acq, "SW_h", 0.0)
    exp.td = _num(acq, "TD", 0, int)
    exp.temperature_K = _num(acq, "TE", 0.0)
    ts = _num(acq, "DATE", 0, int)
    if ts:
        try:
            exp.date = datetime.fromtimestamp(ts)
        except Exception:
            exp.date = None
    # processed data folder: lowest-numbered pdata/<n> with a procs file
    pdir = os.path.join(folder, "pdata")
    if os.path.isdir(pdir):
        cands = []
        for n in os.listdir(pdir):
            p = os.path.join(pdir, n)
            if n.isdigit() and os.path.isfile(os.path.join(p, "procs")):
                cands.append((int(n), p))
        if cands:
            exp.pdata = sorted(cands)[0][1]
            procs = _read_jcamp(os.path.join(exp.pdata, "procs"))
            exp.phc0 = _num(procs, "PHC0", None)
            exp.phc1 = _num(procs, "PHC1", None)
            exp.extra["procs"] = procs
            tpath = os.path.join(exp.pdata, "title")
            if os.path.isfile(tpath):
                try:
                    with open(tpath, "r", errors="replace") as fh:
                        exp.title = " ".join(l.strip() for l in fh if l.strip())
                except OSError:
                    pass
    exp.extra["acqus"] = acq
    return exp


def find_experiments(root: str) -> list:
    """Every 1D experiment under ``root`` (any depth), sorted by path."""
    root = os.path.abspath(root)
    found = []
    if _is_1d_experiment(root):
        e = _read_experiment(root, os.path.dirname(root))
        return [e] if e else []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "fid" in filenames and "acqus" in filenames:
            e = _read_experiment(dirpath, root)
            if e is not None:
                found.append(e)
            dirnames[:] = []            # do not descend into pdata
    found.sort(key=lambda e: _natural_key(e.relpath))
    return found


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def extract_zip(data: bytes, dest: Optional[str] = None) -> str:
    """Unpack an uploaded zip into a temporary folder; returns its path."""
    dest = dest or tempfile.mkdtemp(prefix="nmr_stack_")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for m in zf.infolist():
            # guard against path traversal
            target = os.path.abspath(os.path.join(dest, m.filename))
            if not target.startswith(os.path.abspath(dest) + os.sep):
                continue
            if m.filename.startswith("__MACOSX/"):
                continue
            zf.extract(m, dest)
    return dest


# ── Reaction time from the title ────────────────────────────────────────────
_TIME_RE = re.compile(
    r"(?<![0-9.])(\d+(?:[.,]\d+)?)\s*"
    r"(min(?:s|utes?)?|MIN(?:S|UTES?)?|h(?:rs?|ours?)?|d(?:ays?)?|s(?:ec(?:onds?)?)?)"
    r"(?![A-Za-z])")


def reaction_time_h(text: str) -> Optional[float]:
    """Reaction time in hours parsed from a title such as
    '..._120C15min_D2O' or 'reflux 6 h' -- the LAST time-like token wins.
    Returns None when the title holds none."""
    if not text:
        return None
    hits = _TIME_RE.findall(text)
    if not hits:
        return None
    v, u = hits[-1]
    v = float(v.replace(",", "."))
    u = u.lower()
    if u.startswith("min"):
        return v / 60.0
    if u.startswith("h"):
        return v
    if u.startswith("d"):
        return 24.0 * v
    if u.startswith("s"):
        return v / 3600.0
    return None


# ── Processing ──────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "lb_hz": 0.3,              # exponential line broadening (Hz); None -> from procs
    "zero_fill": 2,            # zero-fill factor (spectrum size = zf * TD/2, min SI from procs)
    "phase_source": "topspin", # "topspin" | "auto" | "manual"
    "phc0": 0.0,               # manual / additional phases (degrees)
    "phc1": 0.0,
    "baseline": True,          # baseline correction
    "baseline_method": "als",  # als (asymmetric least squares) or poly
    "baseline_order": 3,       # for poly
    "baseline_lam": 1e4,       # stiffness for als
    "use_1r": False,           # read TopSpin's processed spectrum instead of the FID
}


def _ppm_axis(exp: Experiment, n: int) -> np.ndarray:
    """ppm axis (descending) for a spectrum of ``n`` points."""
    procs = exp.extra.get("procs") or {}
    sf = _num(procs, "SF", 0.0) or exp.sfo1
    sw_p = _num(procs, "SW_p", 0.0) or exp.sw_h
    offset = _num(procs, "OFFSET", None)
    if offset is None:
        # carrier at the centre of the window: O1 (Hz) relative to BF1
        acq = exp.extra.get("acqus") or {}
        o1 = _num(acq, "O1", 0.0)
        bf1 = _num(acq, "BF1", 0.0) or sf
        sf = sf or bf1
        offset = (o1 + sw_p / 2.0) / sf if sf else sw_p / 2.0
    return offset - np.arange(n) * sw_p / sf / n


def _phase(spec: np.ndarray, p0: float, p1: float) -> np.ndarray:
    n = spec.size
    return spec * np.exp(-1j * np.deg2rad(p0 + p1 * np.arange(n) / n))


def autophase(spec: np.ndarray) -> tuple:
    """Zero- and first-order phases (degrees, TopSpin sign convention).

    Objective: negative area of the real part (squared, normalised) minus a
    small reward for total positive area, on a decimated copy of the
    spectrum; a coarse grid over (p0, p1) is refined with Nelder-Mead.
    Note: with one dominant solvent peak the first-order phase is poorly
    defined by any automatic criterion -- use the sliders to finish.
    """
    from scipy.optimize import minimize
    n = spec.size
    step = max(1, n // 8192)
    zz = spec[::step]
    kk = (np.arange(n) / n)[::step]

    def obj(p):
        y = (zz * np.exp(-1j * np.deg2rad(p[0] + p[1] * kk))).real
        neg = y[y < 0]
        return (float(np.sum(neg ** 2)) / float(np.sum(y ** 2) + 1e-30)
                - 1e-3 * float(np.sum(y)) / float(np.sum(np.abs(y)) + 1e-30))

    best = min((obj([a, b]), a, b) for a in range(0, 360, 15) for b in range(-180, 181, 30))
    r = minimize(obj, [best[1], best[2]], method="Nelder-Mead",
                 options={"xatol": 0.05, "fatol": 1e-10, "maxiter": 400})
    return float(r.x[0]) % 360.0, float(r.x[1])


def baseline_correct(y: np.ndarray, order: int = 3, n_iter: int = 30) -> np.ndarray:
    """Iterative polynomial baseline (points above the fit are clipped to it
    each round, so peaks drop out and the polynomial follows the floor).
    The fit runs on a <= 4096-point subsample; the polynomial is evaluated
    on the full axis, so it costs milliseconds even for 64k points."""
    n = y.size
    if n < 8:
        return y
    stride = max(1, n // 4096)
    x = np.linspace(-1.0, 1.0, n)
    xs, w = x[::stride], y[::stride].copy()
    base_s = np.zeros_like(w)
    for _ in range(n_iter):
        coef = np.polyfit(xs, w, order)
        base_s = np.polyval(coef, xs)
        w = np.minimum(w, base_s)
    return y - np.polyval(coef, x)


def edge_zones(y: np.ndarray, min_frac: float = 0.02) -> tuple:
    """(n_left, n_right): number of points at each end of the spectrum that
    belong to the roll-off of the digital filter, where the smoothed floor
    falls away below the interior floor.  At least ``min_frac`` of the
    points on each side."""
    from scipy.ndimage import median_filter
    n = y.size
    stride = max(1, n // 4096)
    ys = np.asarray(y[::stride], float)
    m = ys.size
    if m < 50:
        return 0, 0
    noise = float(np.std(np.diff(ys)) / np.sqrt(2.0)) or 1e-30
    sm = median_filter(ys, size=max(3, m // 200))
    floor_lvl = float(np.percentile(sm[int(0.2 * m):int(0.8 * m)], 20))
    out = []
    for side in (0, 1):
        idx = np.arange(m) if side == 0 else np.arange(m - 1, -1, -1)
        k, run = 0, 0
        for t, i in enumerate(idx):
            if sm[i] > floor_lvl - 4.0 * noise:
                run += 1
                if run >= 20:
                    k = t - 19
                    break
            else:
                run = 0
        k = max(k, int(min_frac * m)) + max(2, int(0.01 * m))
        out.append(min(n, k * stride))
    return int(out[0]), int(out[1])


def baseline_als(y: np.ndarray, lam: float = 1e4, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens) with peak masking.

    A smooth curve that hugs the floor of the spectrum.  Peaks and their feet
    (points well above the floor, widened by a margin that grows with the
    peak's height, so the foot of a huge solvent signal is not mistaken for
    baseline) get no weight at all, so the curve is interpolated across them
    instead of bending towards them.  Elsewhere the usual asymmetry applies:
    points above the curve weigh ``p``, below it 1 - p.  Where an unweighted
    zone touches an end of the spectrum the curve is held flat at the last
    weighted value instead of being extrapolated, so nothing can tilt near
    the ends.  Expects the roll-off of the digital filter to be cut off
    already (see ``edge_zones``).  Computed on a <= 4096-point subsample and
    interpolated."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import median_filter, binary_dilation
    n = y.size
    stride = max(1, n // 4096)
    ys = np.asarray(y[::stride], float)
    m = ys.size
    if m < 10:
        return y - float(np.median(y))
    # noise and a robust floor estimate
    noise = float(np.std(np.diff(ys)) / np.sqrt(2.0)) or 1e-30
    sm = median_filter(ys, size=max(3, m // 200))
    core_lo, core_hi = int(0.2 * m), int(0.8 * m)
    floor_lvl = float(np.percentile(sm[core_lo:core_hi], 20))
    keep = np.ones(m)
    # peaks and their feet: above floor + 6 sigma, widened by a margin growing with height
    above = sm > floor_lvl + 6.0 * noise
    mask = above.copy()
    tall = sm > floor_lvl + 200.0 * noise
    mask |= binary_dilation(above, iterations=max(2, int(0.004 * m)))          # ~0.1 % of the width
    mask |= binary_dilation(tall, iterations=max(4, int(0.03 * m)))            # a huge peak: wide feet
    keep[mask] = 0.0
    if keep.sum() < 0.1 * m:                                                  # almost everything masked: fall back
        keep = np.ones(m)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(m, m - 2))
    DD = (lam * D.dot(D.T)).tocsc()
    w = keep.copy()
    z = ys
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, m, m)
        z = spsolve((W + DD).tocsc(), w * ys)
        w = keep * (p * (ys > z) + (1.0 - p) * (ys < z))
    # no extrapolation into an unweighted zone at either end: hold the curve flat there
    kept = np.flatnonzero(keep > 0)
    if kept.size:
        z = np.asarray(z, float).copy()
        z[:kept[0]] = z[kept[0]]
        z[kept[-1] + 1:] = z[kept[-1]]
        # The asymmetric weights make the curve hug the BOTTOM of the noise, about two
        # sigma under its centre.  An integral is a sum of points: that offset times the
        # width of the window would be added to every one.  So the curve is raised to the
        # centre of the noise, by ONE amount for the whole spectrum: the median of what is
        # left on the floor points.  (Following it locally was tried and is worse: between
        # crowded peaks the "floor" points are the peaks' feet and lift the curve.)
        # The feet of peaks hide among the floor points and would pull that median up,
        # so points more than two sigma ABOVE the centre are clipped away and the centre
        # taken again, a few times (sigma from the points below the centre, which the
        # peaks do not touch).
        r_floor = (ys - z)[keep > 0]
        if r_floor.size >= 10:
            c = float(np.median(r_floor))
            for _ in range(8):
                low = r_floor[r_floor <= c]
                sg = float(1.4826 * np.median(c - low)) if low.size else 0.0
                if sg <= 0:
                    break
                sel = r_floor[r_floor <= c + 2.0 * sg]
                c_new = float(np.median(sel)) if sel.size >= 10 else c
                if abs(c_new - c) < 1e-3 * sg:
                    c = c_new
                    break
                c = c_new
            z = z + c
    base = np.interp(np.arange(n), np.arange(0, n, stride)[:m], z)
    return y - base


def apply_baseline(y: np.ndarray, method: str = "als", order: int = 3, lam: float = 1e4) -> np.ndarray:
    """Baseline correction by name: 'als' (default) or 'poly'."""
    if method == "poly":
        return baseline_correct(y, order)
    return baseline_als(y, lam=lam)


def process_experiment(exp: Experiment, settings: Optional[dict] = None) -> Spectrum:
    """FID -> phased, baseline-corrected spectrum on a ppm axis."""
    if ng is None:
        raise ImportError(f"nmrglue is required: pip install nmrglue ({_NG_ERR})")
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    procs = exp.extra.get("procs") or {}

    if s.get("use_1r") and exp.pdata and os.path.isfile(os.path.join(exp.pdata, "1r")):
        _, r = ng.bruker.read_pdata(exp.pdata)
        r = np.asarray(r, dtype=float)
        ppm = _ppm_axis(exp, r.size)
        a, b = edge_zones(r)
        sl = slice(a, r.size - b)
        y = r[sl]
        if s.get("baseline"):
            y = apply_baseline(y, s.get("baseline_method", "als"), int(s.get("baseline_order", 3)), float(s.get("baseline_lam", 1e4)))
        return Spectrum(ppm=ppm[sl], real=y, imag=np.zeros_like(y), exp=exp,
                        phc0=exp.phc0 or 0.0, phc1=exp.phc1 or 0.0, source="1r")

    dic, fid = ng.bruker.read(exp.path)
    fid = np.asarray(fid, dtype=complex)
    # Bruker's digital filter delays the FID by GRPDLY points.  TopSpin does not
    # cut those points off: it transforms the whole FID and takes the delay out
    # as a linear phase of 360 * GRPDLY degrees across the spectrum.  Shifting
    # and truncating the FID instead (nmrglue's default) leaves a broad dome in
    # the baseline, so the TopSpin way is used whenever GRPDLY is known.
    grpdly = _num(exp.extra.get("acqus") or {}, "GRPDLY", -1.0)
    if not (grpdly is not None and grpdly > 0):
        fid = np.asarray(ng.bruker.remove_digital_filter(dic, fid), dtype=complex)
        fid[0] *= 0.5
        grpdly = 0.0
    lb = s.get("lb_hz")
    if lb is None:
        lb = _num(procs, "LB", 0.3)
    sw = exp.sw_h or float(dic["acqus"].get("SW_h", 1.0))
    if lb and sw:
        fid = ng.proc_base.em(fid, lb=float(lb) / sw)
    si = int(max(_num(procs, "SI", 0, int), int(2 ** np.ceil(np.log2(max(1, fid.size)))) * max(1, int(s.get("zero_fill", 2)))))
    fid = ng.proc_base.zf_size(fid, si)
    spec = ng.proc_base.fft(fid)[::-1]
    if grpdly:
        spec = _phase(spec, 0.0, 360.0 * float(grpdly))     # the group delay, TopSpin's way

    src = str(s.get("phase_source", "topspin"))
    if src == "topspin" and exp.phc0 is not None:
        p0, p1 = float(exp.phc0), float(exp.phc1 or 0.0)
    elif src in ("topspin", "auto"):
        p0, p1 = autophase(spec)
    else:
        p0, p1 = 0.0, 0.0
    p0 += float(s.get("phc0", 0.0))
    p1 += float(s.get("phc1", 0.0))
    spec = _phase(spec, p0, p1)
    ppm = _ppm_axis(exp, spec.size)
    # the roll-off of the digital filter at both ends carries no information: cut it
    # off first, so the baseline is fitted on real floor only
    a, b = edge_zones(spec.real)
    sl = slice(a, spec.size - b)
    spec = spec[sl]
    y = spec.real
    if s.get("baseline"):
        y = apply_baseline(y, s.get("baseline_method", "als"), int(s.get("baseline_order", 3)), float(s.get("baseline_lam", 1e4)))
    return Spectrum(ppm=ppm[sl], real=y, imag=spec.imag,
                    exp=exp, phc0=p0, phc1=p1, source="fid", raw=spec.real)


def rephase(spec: Spectrum, d0: float, d1: float, baseline: bool = True,
            order: int = 3, method: str = "als", lam: float = 1e4) -> Spectrum:
    """Apply an extra (d0, d1) to an already processed FID spectrum -- cheap,
    no re-transform.  Not available for spectra read from 1r."""
    if spec.source != "fid":
        return spec
    z = (spec.raw if spec.raw is not None else spec.real) + 1j * spec.imag
    z = _phase(z, d0, d1)
    y = z.real
    if baseline:
        y = apply_baseline(y, method, order, lam)
    return Spectrum(ppm=spec.ppm, real=y, imag=z.imag, exp=spec.exp,
                    phc0=spec.phc0 + d0, phc1=spec.phc1 + d1, source="fid", raw=z.real)


def reference(spec: Spectrum, peak_ppm_now: float, peak_ppm_target: float) -> Spectrum:
    """Shift the ppm axis so the peak at ``peak_ppm_now`` reads ``peak_ppm_target``."""
    return Spectrum(ppm=spec.ppm + (peak_ppm_target - peak_ppm_now), real=spec.real,
                    imag=spec.imag, exp=spec.exp, phc0=spec.phc0, phc1=spec.phc1,
                    source=spec.source, raw=spec.raw)


def nearest_peak(spec: Spectrum, ppm: float, window: float = 0.1) -> float:
    """ppm of the highest point within ±window of ``ppm``."""
    m = (spec.ppm >= ppm - window) & (spec.ppm <= ppm + window)
    if not np.any(m):
        return ppm
    idx = np.flatnonzero(m)
    return float(spec.ppm[idx[np.argmax(spec.real[idx])]])


def integrate(spec: Spectrum, lo: float, hi: float) -> float:
    """Area between two ppm values (order irrelevant), trapezoid rule."""
    lo, hi = min(lo, hi), max(lo, hi)
    m = (spec.ppm >= lo) & (spec.ppm <= hi)
    if m.sum() < 2:
        return 0.0
    x = spec.ppm[m][::-1]
    y = spec.real[m][::-1]
    return float(np.trapz(y, x))


def scale_factor(spec: Spectrum, mode: str) -> float:
    """Multiplier that puts spectra on a common footing.
    'max' -> unit maximum, 'ns_rg' -> per scan and per unit receiver gain,
    'none' -> raw."""
    if mode == "max":
        m = float(np.nanmax(np.abs(spec.real))) if spec.real.size else 1.0
        return 1.0 / m if m > 0 else 1.0
    if mode == "ns_rg":
        return 1.0 / (max(spec.exp.ns, 1) * max(spec.exp.rg, 1e-9))
    return 1.0


# ── Measurements at full resolution (calibration, peak tracking, integrals) ──
def _apex(ppm: np.ndarray, y: np.ndarray, k: int) -> float:
    """Sub-point apex position by parabolic interpolation around index k."""
    if 0 < k < y.size - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = (y0 - 2 * y1 + y2)
        if den != 0:
            d = 0.5 * (y0 - y2) / den
            if abs(d) < 1:
                return float(ppm[k] + d * (ppm[min(k + 1, y.size - 1)] - ppm[k]))
    return float(ppm[k])


def find_peak_near(ppm: np.ndarray, y: np.ndarray, x0: float, window: float,
                   measure: str = "apex") -> Optional[float]:
    """Position of the most prominent peak within ±window of x0, or None.

    Prominence, not height, decides (a shoulder next to a tall line does not
    win).  ``measure`` = 'apex' (parabolic interpolation of the maximum) or
    'centroid' (intensity-weighted centre of the points above 30 % of the
    apex inside the window -- the middle of a multiplet)."""
    m = np.flatnonzero((ppm >= x0 - window) & (ppm <= x0 + window))
    if m.size < 3:
        return None
    seg = y[m]
    try:
        from scipy.signal import find_peaks
        noise = float(np.std(np.diff(seg)) / np.sqrt(2.0)) or 1e-30
        idx, props = find_peaks(seg, prominence=3.0 * noise)
    except Exception:
        idx, props = np.array([], int), {"prominences": np.array([])}
    if idx.size == 0:
        k = int(np.argmax(seg))
        if seg[k] <= 0:
            return None
    else:
        # prefer the most prominent; break near-ties by closeness to x0
        prom = np.asarray(props["prominences"], float)
        score = prom * (1.0 - 0.3 * np.abs(ppm[m][idx] - x0) / max(window, 1e-9))
        k = int(idx[int(np.argmax(score))])
    kk = int(m[k])
    if measure == "centroid":
        thr = 0.3 * seg[k]
        sel = m[seg >= thr]
        # keep the contiguous run around the apex
        runs = np.split(sel, np.flatnonzero(np.diff(sel) > 1) + 1)
        run = next((r for r in runs if kk in r), sel)
        w = np.clip(y[run], 0, None)
        return float(np.sum(ppm[run] * w) / np.sum(w)) if np.sum(w) > 0 else _apex(ppm, y, kk)
    return _apex(ppm, y, kk)


def _pv(x, A, c, g, eta):
    """Pseudo-Voigt line: height A at c, half width at half height g, Lorentzian share eta."""
    d = (x - c) / g
    return A * (eta / (1.0 + d * d) + (1.0 - eta) * np.exp(-np.log(2.0) * d * d))


def pv_area(A, g, eta, lo=None, hi=None, c=0.0):
    """Area of a pseudo-Voigt line: all of it, or the part between lo and hi."""
    if lo is None or hi is None:
        return float(A * g * (eta * np.pi + (1.0 - eta) * _GAUSS_AREA))
    from math import atan, erf, sqrt, log
    a, b = (min(lo, hi) - c) / g, (max(lo, hi) - c) / g
    lor = atan(b) - atan(a)
    k = sqrt(log(2.0))
    gau = 0.5 * sqrt(np.pi) / k * (erf(k * b) - erf(k * a))
    return float(A * g * (eta * lor + (1.0 - eta) * gau))


def deconvolve_region(spec: Spectrum, lo: float, hi: float, max_comp: int = 8,
                      free_baseline: bool = False) -> Optional[dict]:
    """Deconvolution of everything between ``lo`` and ``hi``: a sum of
    pseudo-Voigt lines, fitted together.  The spectrum is baseline-corrected,
    so the baseline is held at zero: a free baseline, even one sigma of it,
    trades against the wings of the lines and inflates their areas.
    ``free_baseline`` frees a straight line, for signals on a solvent tail.

    The lines the finder sees start the fit.  Then, as long as the fit leaves
    a hump in the residual that stands clear of the noise, a line is added
    there and the whole is fitted again: this is how a shoulder, or a broad
    signal hiding under a sharp one, gets a line of its own.  Returns
    {comps: [[A, c, g, eta], ...] sorted by position, c: [c0, c1], x0, lo, hi,
    rmse (relative to the tallest point), n_added} or None."""
    from scipy.optimize import least_squares
    lo, hi = min(lo, hi), max(lo, hi)
    sf = spec.exp.sfo1 or 500.0
    m = (spec.ppm >= lo) & (spec.ppm <= hi)
    if m.sum() < 30:
        return None
    x = spec.ppm[m].astype(float)
    y = spec.real[m].astype(float)
    step = abs(float(x[1] - x[0])) or 1e-4
    x0 = 0.5 * (lo + hi)
    noise, corr = noise_stats(y - np.median(y))
    noise = noise or 1e-30
    top = float(y.max())
    if top <= 6.0 * noise:
        return None
    lines, _ = find_lines(spec.ppm, spec.real, lo, hi, sf)
    comps = []                                   # (centre, FWHM) to start from
    for pos, _h in lines:
        w = min(max(signal_fwhm(spec, pos, sf), 3 * step), 0.5 * (hi - lo))
        comps.append((float(pos), float(w)))
    if not comps:
        k = int(np.argmax(y))
        comps.append((float(x[k]), min(max(signal_fwhm(spec, float(x[k]), sf), 3 * step), 0.5 * (hi - lo))))

    def model(p, n):
        out = p[4 * n] + p[4 * n + 1] * (x - x0)
        for i in range(n):
            out = out + _pv(x, *p[4 * i:4 * i + 4])
        return out

    def fit(comps, prev=None):
        n = len(comps)
        p0, lb, ub = [], [], []
        for i, (c, w) in enumerate(comps):
            if prev is not None and i < len(prev):
                A0, c0, g0, e0 = prev[i]
            else:
                A0, c0, g0, e0 = max(float(np.interp(c, x[::-1], y[::-1])), noise), c, 0.5 * w, 0.5
            p0 += [A0, c0, g0, e0]
            lb += [0.0, c - 1.0 * w, 0.1 * w, 0.0]
            ub += [np.inf, c + 1.0 * w, 2.5 * w, 1.0]
        if free_baseline:
            p0 += [0.0, 0.0]; lb += [-np.inf, -np.inf]; ub += [np.inf, np.inf]
        else:
            _e = 1e-6 * noise                        # held at zero (the solver wants a finite interval)
            p0 += [0.0, 0.0]; lb += [-_e, -_e / (hi - lo)]; ub += [_e, _e / (hi - lo)]
        p0 = np.clip(p0, np.array(lb) + 1e-12, np.array(ub) - 1e-12)
        r = least_squares(lambda p: model(p, n) - y, p0, bounds=(lb, ub), max_nfev=400)
        return r.x

    try:
        p = fit(comps)
        added = 0
        while len(comps) < max_comp:
            n = len(comps)
            res = y - model(p, n)
            w_min = min(2.0 * p[4 * i + 2] for i in range(n))
            k = max(3, int(round(0.5 * w_min / step)) | 1)
            rs = np.convolve(res, np.ones(k) / k, mode="same")
            ns = noise * np.sqrt(min(1.0, corr / k))
            edge = k                                   # the ends of a moving average are not to be trusted
            j = edge + int(np.argmax(rs[edge:-edge])) if rs.size > 2 * edge + 3 else int(np.argmax(rs))
            if rs[j] < max(6.0 * ns, 0.03 * top):
                break
            # width of that residual hump at half its height
            hgt = rs[j]; a = j
            while a > 0 and rs[a] > 0.5 * hgt:
                a -= 1
            b = j
            while b < rs.size - 1 and rs[b] > 0.5 * hgt:
                b += 1
            w_new = min(max((b - a) * step, 3 * step), 0.5 * (hi - lo))
            prev = [list(p[4 * i:4 * i + 4]) for i in range(n)]
            comps = comps + [(float(x[j]), float(w_new))]
            p = fit(comps, prev)
            added += 1
    except Exception:
        return None
    n = len(comps)
    out = [[float(v) for v in p[4 * i:4 * i + 4]] for i in range(n)]
    order = np.argsort([c[1] for c in out])
    rmse = float(np.sqrt(np.mean((y - model(p, n)) ** 2))) / (top or 1.0)
    return {"comps": [out[i] for i in order], "c": [float(p[4 * n]), float(p[4 * n + 1])], "x0": float(x0),
            "lo": float(lo), "hi": float(hi), "rmse": rmse, "n_added": int(added), "noise": float(noise)}


def track_peak(specs: list, start: int, x0: float, window: float,
               measure: str = "apex") -> dict:
    """Follow a peak clicked at ``x0`` on spectrum ``start`` through the list,
    outward in both directions, each spectrum searched around the position
    found in its neighbour.  Returns {label: ppm or None}."""
    out = {}
    order = list(range(start, len(specs))) + list(range(start - 1, -1, -1))
    last_fwd = x0
    last_bwd = x0
    for i in order:
        sp = specs[i]
        ref = last_fwd if i >= start else last_bwd
        pos = find_peak_near(sp.ppm, sp.real, ref, window, measure)
        out[sp.exp.label] = pos
        if pos is not None:
            if i >= start:
                last_fwd = pos
            else:
                last_bwd = pos
    return out


def integrate_region(spec: Spectrum, lo: float, hi: float, baseline: bool = True,
                     edge_pts: int = 5) -> float:
    """Area between lo and hi; with ``baseline`` a straight line through the
    mean of ``edge_pts`` points at each edge is subtracted first."""
    lo, hi = min(lo, hi), max(lo, hi)
    m = np.flatnonzero((spec.ppm >= lo) & (spec.ppm <= hi))
    if m.size < 2:
        return 0.0
    x = spec.ppm[m][::-1]
    y = spec.real[m][::-1].astype(float)
    if baseline:
        e = max(1, min(edge_pts, m.size // 4))
        y0, y1 = float(np.mean(y[:e])), float(np.mean(y[-e:]))
        y = y - np.linspace(y0, y1, y.size)
    return float(np.trapz(y, x))


def shifted(spec: Spectrum, delta_ppm: float) -> Spectrum:
    """Copy of the spectrum with its ppm axis moved by delta_ppm."""
    if not delta_ppm:
        return spec
    return Spectrum(ppm=spec.ppm + float(delta_ppm), real=spec.real, imag=spec.imag,
                    exp=spec.exp, phc0=spec.phc0, phc1=spec.phc1, source=spec.source, raw=spec.raw)


def x_from_title(text: str) -> Optional[float]:
    """A volume-like number in a title: '0.10mL', '25 uL' (-> mL), '150 µL'."""
    if not text:
        return None
    m = re.findall(r"(?<![0-9.])(\d+(?:[.,]\d+)?)\s*(mL|ml|uL|ul|µL|μL)(?![A-Za-z])", text)
    if not m:
        return None
    v, u = m[-1]
    v = float(v.replace(",", "."))
    return v / 1000.0 if u.lower() in ("ul", "µl", "μl") else v


# ── Equilibrist export ──────────────────────────────────────────────────────
def build_equilibrist_table(rows: list, peaks: list, regions: list) -> "pd.DataFrame":
    """rows: [{label, x, spec}] in stack order.  peaks: [{name, pos{label: ppm}}].
    regions: [{name, nH, baseline, limits{label: (lo, hi) or None}}] (or a
    single lo/hi for every spectrum).  Integration columns first, then shift
    columns -- the layout Equilibrist's $nmr mixed mode expects.  A name may
    appear in both groups (Equilibrist tells them apart by position)."""
    import pandas as pd
    names, cols = ["x"], [[np.nan if r["x"] is None else float(r["x"]) for r in rows]]
    for rg in regions:
        col = []
        for r in rows:
            if "values" in rg:                              # already integrated by the caller (fit or area)
                v = rg["values"].get(r["label"])
                col.append(np.nan if v is None else float(v))
                continue
            lim = (rg.get("limits") or {}).get(r["label"]) if "limits" in rg else (rg.get("lo"), rg.get("hi"))
            if not lim or lim[0] is None:
                col.append(np.nan)
            else:
                col.append(integrate_region(r["spec"], float(lim[0]), float(lim[1]), rg.get("baseline", True)))
        names.append(rg["name"]); cols.append(col)
    for pk in peaks:
        names.append(pk["name"])
        cols.append([np.nan if pk["pos"].get(r["label"]) is None else float(pk["pos"][r["label"]]) for r in rows])
    return pd.DataFrame(np.column_stack([np.asarray(c, float) for c in cols]), columns=names)


# ── Multiplets ──────────────────────────────────────────────────────────────
_MULT_NAMES = {1: "s", 2: "d", 3: "t", 4: "q", 5: "quint", 6: "sext", 7: "sept", 8: "oct", 9: "non"}


def noise_stats(yy: np.ndarray, max_lag: int = 16) -> tuple:
    """(sigma, correlation length in points) of the noise in ``yy``.

    Zero-filling and apodisation make neighbouring points nearly equal, so
    the scatter of point-to-point differences reads only a fraction of the
    real noise (about a third at ten points per Hz).  Differences taken
    further apart grow until the points are independent: sigma is that
    plateau.  Each estimate is a median absolute deviation, so the lines in
    the region do not inflate it.  The correlation length, 1 + 2 * sum of
    the autocorrelation, tells how much an average over k points really
    lowers the noise: by sqrt(k / length), not sqrt(k)."""
    yy = np.asarray(yy, float)
    est = []
    for lag in range(1, max_lag + 1):
        if yy.size <= 4 * lag:
            break
        d = yy[lag:] - yy[:-lag]
        est.append(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)))
    if not est or max(est) <= 0:
        return 0.0, 1.0
    sigma = max(est)
    rho = [max(0.0, 1.0 - (e / sigma) ** 2) for e in est]
    return sigma, 1.0 + 2.0 * float(sum(rho))


def find_lines(ppm: np.ndarray, y: np.ndarray, lo: float, hi: float, sf_mhz: float,
               min_snr: float = 4.5, min_prominence_frac: float = 0.08,
               min_rel: float = 0.20, min_rel_width: float = 0.30, min_box_snr: float = 0.0,
               min_frac_height: float = 0.04, min_prom_snr: float = 6.0,
               noise_override: Optional[float] = None) -> tuple:
    """Resolved lines inside [lo, hi] as (ppm, height) pairs, plus the noise.

    Prominence, not height, separates a line from a ripple on a broad hump:
    a line must stand at least ``min_prominence_frac`` of its own height and
    ``min_prom_snr`` sigma above the dip beside it.  Six sigma of the REAL
    noise (see noise_stats), because the peak-to-valley excursions of plain
    noise over a box reach about five: below that, the noise riding on a low
    broad hump reads as a forest of lines.  ``noise_override`` gives the noise when ``y`` has been
    smoothed (its point-to-point differences then underestimate it).
    Positions are refined by a parabola through the three samples around the
    maximum."""
    lo, hi = min(lo, hi), max(lo, hi)
    # look a little beyond the box, so a line cut by its edge keeps its true
    # prominence (measured against the floor, not against the edge); only lines
    # whose apex lies inside the box are returned
    marg = max(0.03, 0.25 * (hi - lo))
    m = np.flatnonzero((ppm >= lo - marg) & (ppm <= hi + marg))
    if m.size < 9:
        return [], 0.0
    x = ppm[m][::-1]                        # ascending ppm
    yy = (y[m][::-1] - float(np.median(y[m]))).astype(float)
    noise, _ = noise_stats(yy)              # the real noise, not the point-to-point scatter
    if noise <= 0:
        noise = float(np.std(np.diff(yy)) / np.sqrt(2.0))
    if noise_override is not None and noise_override > 0:
        noise = float(noise_override)
    if noise <= 0:
        return [], 0.0
    # heights are counted from the floor of the region.  The median is the floor
    # only where most of the region is baseline; in a box that is all signal it
    # sits on the signal.  The 10th percentile, raised by the 1.28 sigma it lies
    # below the centre of plain noise, equals the median on a clean baseline and
    # stays on the floor in a crowded region.
    floor = float(np.percentile(yy, 10.0)) + 1.2816 * noise
    if floor < 0:
        yy = yy - floor
    step_hz = abs(float(np.median(np.diff(x)))) * max(sf_mhz, 1e-9)
    dist = max(int(round(0.8 / max(step_hz, 1e-9))), 1)
    try:
        from scipy.signal import find_peaks
        idx, props = find_peaks(yy, height=min_snr * noise, distance=dist, prominence=min_prom_snr * noise,
                                width=2)          # a real line spans several points; a spike does not
    except Exception:
        return [], noise
    lines = []
    # only candidates whose apex lies inside the box count; the margin serves
    # the prominence measurement alone
    inbox = (x[idx] >= lo) & (x[idx] <= hi)
    idx = idx[inbox]
    props = {k: np.asarray(v)[inbox] for k, v in props.items()}
    if not idx.size:
        return [], noise
    # nothing in the box?  The strongest candidate must stand well above the
    # scatter of the rest of the region (its own neighbourhood excluded, so a
    # box drawn tightly around a big peak still passes).
    if idx.size:
        k_top = int(np.argmax(props["prominences"]))
        i_top, w_pts = int(idx[k_top]), float(props["widths"][k_top])
        keep_pts = np.ones(yy.size, bool)
        keep_pts[max(0, int(i_top - 3 * w_pts)):int(i_top + 3 * w_pts) + 1] = False
        rest = yy[keep_pts]
        if rest.size >= 0.3 * yy.size:
            med = float(np.median(rest))
            # the scatter of the floor, from the points BELOW the median only: a
            # second real signal sharing the box must not count as scatter
            low = rest[rest <= med]
            mad = float(1.4826 * np.median(med - low)) if low.size else 0.0
            if mad > 0 and (yy[i_top] - med) < min_box_snr * mad:
                return [], noise
    # baseline ripple next to a strong line: much narrower than it and far weaker.
    # A real outer line of a multiplet has the width of the strong lines.
    proms, widths = props["prominences"], props["widths"]
    if idx.size:
        k_top = int(np.argmax(proms))
        p_top, w_top = float(proms[k_top]), float(widths[k_top])
    h_top = float(np.max(yy[idx])) if idx.size else 0.0
    for i, prom, wd in zip(idx, proms, widths):
        if prom < min_prominence_frac * yy[i]:
            continue
        # (a candidate standing 12 sigma above its dip is real structure whatever its
        # size next to the main line: the second top of an unresolved doublet, for one)
        if idx.size and prom < min_rel * p_top and wd < min_rel_width * w_top and prom < 12.0 * noise:
            continue
        if yy[i] < min_frac_height * h_top:          # far too weak to belong to the signal (a septet's outer line is 5 %)
            continue
        xp, yp = float(x[i]), float(yy[i])
        if 0 < i < yy.size - 1:
            den = yy[i - 1] - 2.0 * yy[i] + yy[i + 1]
            if den < 0:
                dx = 0.5 * (yy[i - 1] - yy[i + 1]) / den
                if abs(dx) <= 1.0:
                    xp = float(x[i] + dx * (x[i + 1] - x[i - 1]) / 2.0)
                    yp = float(yy[i] - 0.25 * (yy[i - 1] - yy[i + 1]) * dx)
        lines.append((xp, yp))
    return lines, noise


# ── Signals: one click = shift + integration limits on one spectrum ─────────
def measure_signal(spec: Spectrum, lo: float, hi: float, x0: Optional[float] = None,
                   measure: str = "multiplet", frac: float = 0.01, margin: float = 0.0,
                   max_j_hz: float = 20.0, fit_k: float = 6.0, broad_hz: float = 6.0) -> Optional[dict]:
    """The signal inside [lo, hi] on one spectrum: shift, pattern, J, its lines,
    its integration window and its integral.  With ``x0`` (a click) the
    multiplet whose centre is nearest the click wins, otherwise the strongest.

    Sharp lines (narrower than ``broad_hz`` and further apart than their own
    width) form a multiplet: the shift is the centre of the pattern.  A broad
    or unresolved signal (intermediate exchange) is one envelope: the shift is
    its intensity-weighted centre and the pattern is reported as 'br'.
    The integration window is the outer lines +/- ``fit_k`` linewidths; the
    integral is the pseudo-Voigt fit (fit_area) and, always, the numerical
    area over that window with a flank baseline (area)."""
    sf = spec.exp.sfo1 or 500.0
    lines, noise = find_lines(spec.ppm, spec.real, lo, hi, sf)
    if not lines and noise > 0:
        # nothing stands out point by point: look again at the scale of a broad
        # signal.  Averaging over k points lowers the noise by sqrt(k), so a low
        # hump many points wide emerges; first 3 Hz, then 10 Hz for the broadest.
        # (by sqrt(k / correlation length) really: neighbouring points are not independent)
        step_hz = abs(float(spec.ppm[1] - spec.ppm[0])) * sf or 1.0
        marg = max(0.03, 0.25 * abs(hi - lo))
        reg = spec.real[(spec.ppm >= min(lo, hi) - marg) & (spec.ppm <= max(lo, hi) + marg)]
        _, corr = noise_stats(reg - np.median(reg))
        for width_hz in (3.0, 10.0):
            k = max(3, int(round(width_hz / step_hz)) | 1)
            ys = np.convolve(spec.real, np.ones(k) / k, mode="same")
            ns = noise * np.sqrt(min(1.0, corr / k))
            # averaged, even a slow wobble of the baseline stands many sigma above
            # the (now tiny) noise: the signal must also clear 5 sigma of the raw noise
            lines, _ = find_lines(spec.ppm, ys, lo, hi, sf, noise_override=ns,
                                  min_snr=max(4.5, 5.0 * noise / max(ns, 1e-30)))
            if lines:
                break
    if not lines:
        return None
    groups, cur = [], [lines[0]]
    for prev, p in zip(lines, lines[1:]):
        if (p[0] - prev[0]) * sf <= max_j_hz:
            cur.append(p)
        else:
            groups.append(cur); cur = [p]
    groups.append(cur)
    # a group's weight is its rough integral (height x width summed over its
    # lines), so a weak broad hump beats a cluster of narrow ripples
    _wcache = {}
    def fw(p0):
        if p0 not in _wcache:
            _wcache[p0] = signal_fwhm(spec, p0, sf)
        return _wcache[p0]
    def weight(g):
        return sum(p[1] * max(fw(p[0]), 1e-6) for p in g)
    if x0 is not None:
        strongest = max(weight(g) for g in groups)
        cands = [g for g in groups if weight(g) >= 0.1 * strongest] or groups
        best = min(cands, key=lambda g: abs(np.mean([p[0] for p in g]) - x0))
    else:
        best = max(groups, key=weight)
    pos = [p[0] for p in best]
    n = len(pos)
    widths_hz = [fw(p) * sf for p in pos]
    h_top = max(p[1] for p in best)
    strong = [w for w, p in zip(widths_hz, best) if p[1] >= 0.25 * h_top] or widths_hz
    w_med = float(np.median(strong))           # the linewidth of the signal: its strong lines
    gaps_hz = (np.diff(pos) * sf) if n > 1 else np.array([])
    broad = (w_med >= broad_hz) or (n > 1 and float(gaps_hz.min()) <= w_med)
    if broad and n > 1:
        # ripples riding on a broad envelope are not its lines: keep the wide ones
        keep = [(p, w) for p, w in zip(pos, widths_hz) if w >= 0.3 * w_med]
        if keep:
            pos = [p for p, _ in keep]; widths_hz = [w for _, w in keep]; n = len(pos)
            best = [b for b in best if b[0] in pos]
            gaps_hz = (np.diff(pos) * sf) if n > 1 else np.array([])
    # window: outward from the outer lines to the baseline or a valley, at most k linewidths
    w_ppm = w_med / sf
    centre0 = float(np.mean(pos))
    win_lo, win_hi = signal_window(spec, pos, w_ppm, fit_k, frac=frac)
    pattern, J = _MULT_NAMES.get(n, "m"), None
    if broad:
        pattern = "br"
        c_env = envelope_centroid(spec, win_lo, win_hi)
        centre = float(c_env) if c_env is not None else centre0
    else:
        if n > 1:
            if gaps_hz.std() <= 1.2 and gaps_hz.mean() > 0.8 and n <= 9:
                J = float(round(gaps_hz.mean(), 2))
            else:
                pattern = "m"
        if measure == "apex":
            centre = float(max(best, key=lambda p: p[1])[0])
        elif measure == "centroid" or pattern == "m":
            w = np.array([p[1] for p in best]); centre = float(np.sum(np.array(pos) * w) / np.sum(w))
        else:
            centre = centre0
    if measure == "centroid":
        c_env = envelope_centroid(spec, win_lo, win_hi)
        if c_env is not None:
            centre = float(c_env)
    out = {"pos": centre, "pattern": pattern, "J_hz": J, "lines": pos, "fwhm_hz": w_med,
           "lo": float(win_lo), "hi": float(win_hi),
           "area": integrate_window(spec, win_lo, win_hi),
           "fit_area": None, "fit_lo": None, "fit_hi": None, "fit_fwhm_hz": None, "fit_rmse": None,
           "fit_eta": None, "fit_rejected": 0, "fit_nuisance": 0}
    ft = fit_signal_area(spec, pos, centre0, k=fit_k)
    if ft is not None:
        out.update({"fit_area": ft["area"], "fit_lo": ft["lo"], "fit_hi": ft["hi"],
                    "fit_fwhm_hz": ft["fwhm_hz"], "fit_rmse": ft["rmse_rel"], "fit_eta": ft["eta"],
                    "fit_rejected": ft["n_rejected"], "fit_frac": ft["frac_in_window"],
                    "fit_num_expected": ft["num_expected"], "fit_model": ft.get("model")})
        out["fit_nuisance"] = ft.get("n_nuisance", 0)
    return out


# ── Lineshape-fitted integrals ───────────────────────────────────────────────
def signal_fwhm(spec: Spectrum, pos: float, sf_mhz: float, max_ppm: float = 0.1) -> float:
    """Full width at half height (ppm) of the signal at ``pos``, not fooled by
    noise.  On a low broad hump the raw data give the width of the noise spike
    that happens to be its highest point.  So the width is measured again on
    the data averaged over 3 Hz, then 10 Hz (the average's own width taken
    out): a genuine sharp line reads the same at every scale and keeps its
    raw width, a spike on a hump reads several times wider once averaged and
    takes the hump's width."""
    w = line_fwhm(spec, pos, max_ppm)
    step = abs(float(spec.ppm[1] - spec.ppm[0])) or 1e-4
    step_hz = step * max(sf_mhz, 1e-9)
    i0 = int(np.argmin(np.abs(spec.ppm - pos)))
    for width_hz in (3.0, 10.0):
        k = max(3, int(round(width_hz / step_hz)) | 1)
        ys = np.convolve(spec.real, np.ones(k) / k, mode="same")
        r = max(k, int(0.01 / step))
        a, b = max(0, i0 - r), min(ys.size, i0 + r + 1)
        j = a + int(np.argmax(ys[a:b]))
        ws = line_fwhm(spec, float(spec.ppm[j]), max(max_ppm, 0.2), y=ys)
        wc = float(np.sqrt(max(ws * ws - (k * step) ** 2, 0.0)))
        if wc > 3.0 * w:
            w = wc
        else:
            break
    return min(w, max(max_ppm, 0.2))


def line_fwhm(spec: Spectrum, pos: float, max_ppm: float = 0.1, y: Optional[np.ndarray] = None) -> float:
    """Full width at half height (ppm) of the line at ``pos``, measured above
    the local floor (the median within +/- 0.05 ppm) so that a weak line on a
    solvent tail does not read as a broad one; capped at ``max_ppm``.
    ``y`` replaces the spectrum's data (a smoothed copy)."""
    y = spec.real if y is None else y
    k = int(np.argmin(np.abs(spec.ppm - pos)))
    step = abs(float(spec.ppm[1] - spec.ppm[0])) or 1e-4
    n_loc = max(5, int(0.05 / step))
    floor = float(np.median(y[max(0, k - n_loc):k + n_loc + 1]))
    h = floor + 0.5 * (y[k] - floor)
    n_max = max(2, int(max_ppm / step))
    i = k
    while i > 0 and k - i < n_max and y[i] > h:
        i -= 1
    j = k
    while j < y.size - 1 and j - k < n_max and y[j] > h:
        j += 1
    w = float(abs(spec.ppm[i] - spec.ppm[j]))
    return min(max(w, 2 * step), max_ppm)


def _baseline_basis(spec: Spectrum, x0: float, half: float, top: float):
    """Baseline model under a window: a straight line, plus the tail of a
    much taller peak (the solvent) when one stands within 1 ppm."""
    kmax = int(np.argmax(spec.real))
    xs = float(spec.ppm[kmax])
    use_tail = (abs(xs - x0) > half + 0.01) and (abs(xs - x0) < 1.0) and (spec.real[kmax] > 20 * top)
    def basis(x):
        cols = [np.ones_like(x), x - x0]
        if use_tail:
            cols.append(1.0 / ((x - xs) ** 2))
        return np.column_stack(cols)
    return basis, use_tail


def signal_window(spec: Spectrum, lines: list, fwhm_ppm: float, k: float = 6.0,
                  frac: float = 0.01, k_min: float = 1.0) -> tuple:
    """Integration window of a signal, the way one sets it by eye in TopSpin:
    from the outer lines outwards until the (smoothed) signal is back at the
    baseline -- below ``frac`` of its height above the floor, or below three
    times the noise, whichever is larger -- and never further than ``k``
    linewidths.  A side also stops at the bottom of a valley before a
    neighbour.  Neither cut happens closer than ``k_min`` linewidths to the
    outer line (the line's own flank is not a valley).  A height fraction cuts
    every Lorentzian line by the same share of its area, so ratios between
    signals of different widths stay unbiased.  Returns (lo, hi) in ppm."""
    ppm, y = spec.ppm, spec.real
    step = abs(float(ppm[1] - ppm[0])) or 1e-4
    n = y.size
    k_hi = int(np.argmin(np.abs(ppm - max(lines))))      # index of the highest-ppm line (small index)
    k_lo = int(np.argmin(np.abs(ppm - min(lines))))
    a, b = min(k_hi, k_lo), max(k_hi, k_lo)
    sm = max(1, int(0.25 * fwhm_ppm / step))
    ys = np.convolve(y, np.ones(2 * sm + 1) / (2 * sm + 1), mode="same") if sm > 1 else y
    n_nb = max(20, int(0.5 / step))
    reg = y[max(0, a - n_nb):min(n, b + n_nb)]
    floor = float(np.percentile(reg, 20))
    top = float(np.max(y[a:b + 1]))
    n_max = max(3, int(round(k * fwhm_ppm / step)))
    n_min = min(n_max, max(2, int(k_min * fwhm_ppm / step)))

    # a rise counts as a valley only when it clearly exceeds the noise of the smoothed curve
    noise, corr = noise_stats(reg - np.median(reg))
    noise = noise or 1e-30
    rise_min = max(3.0 * noise * np.sqrt(min(1.0, corr / (2 * sm + 1))), 0.02 * (top - floor))

    thr = floor + max(float(frac) * (top - floor), 3.0 * noise)   # "back at the baseline"
    n_run = max(2, sm)                                               # ... for this many points in a row

    def walk(start, direction):
        i, best, ymin, below = start, start, ys[start], 0
        for t in range(n_max):
            j = i + direction
            if j <= 0 or j >= n - 1:
                break
            if ys[j] < ymin:
                ymin, best = ys[j], j
            elif t >= n_min and ys[j] - ymin > rise_min:
                return best                                 # a neighbour: stop at the bottom of the valley
            below = below + 1 if ys[j] <= thr else 0
            if t >= n_min and below >= n_run:
                return j                                    # baseline reached: stop here
            i = j
        return i
    i = walk(a, -1)          # towards higher ppm
    j = walk(b, +1)          # towards lower ppm
    return float(ppm[j]), float(ppm[i])


def window_baseline(spec: Spectrum, lo: float, hi: float, edge_pts: int = 5) -> tuple:
    """(x, y, baseline) inside [lo, hi]: the baseline is the straight line
    through the median of ``edge_pts`` points at each edge of the window --
    the TopSpin bias/slope, anchored where the window was cut (at the
    baseline or in a valley)."""
    lo, hi = min(lo, hi), max(lo, hi)
    m = np.flatnonzero((spec.ppm >= lo) & (spec.ppm <= hi))
    if m.size < 2:
        return spec.ppm[m], spec.real[m].astype(float), np.zeros(m.size)
    x, y = spec.ppm[m], spec.real[m].astype(float)
    e = max(1, min(edge_pts, m.size // 4))
    y0, y1 = float(np.median(y[:e])), float(np.median(y[-e:]))
    return x, y, np.linspace(y0, y1, y.size)


def integrate_window(spec: Spectrum, lo: float, hi: float, baseline: bool = True) -> float:
    """Numerical area inside [lo, hi] after the flank baseline (or raw)."""
    x, y, b = window_baseline(spec, lo, hi)
    if x.size < 2:
        return 0.0
    yy = y - b if baseline else y
    return float(np.trapz(yy[::-1], x[::-1]))


def envelope_centroid(spec: Spectrum, lo: float, hi: float) -> Optional[float]:
    """Intensity-weighted centre of the (baseline-subtracted) signal in the window."""
    x, y, b = window_baseline(spec, lo, hi)
    w = np.clip(y - b, 0, None)
    return float(np.sum(x * w) / w.sum()) if w.sum() > 0 else None


_GAUSS_AREA = float(np.sqrt(np.pi / np.log(2.0)))     # area of exp(-ln2 (x/g)^2) is g * this


def fit_signal_area(spec: Spectrum, lines: list, x0: float, k: float = 6.0,
                    max_rmse: float = 0.05) -> Optional[dict]:
    """Area of a signal from a pseudo-Voigt fitted to each of its lines.

    Each line is A * [eta * Lorentzian + (1 - eta) * Gaussian] with its own
    position, width and shape eta in [0, 1]; the baseline is a straight line
    plus the solvent tail when the solvent is close (see _baseline_basis).
    The window is the outer lines +/- ``k`` linewidths.  Points more than
    4 sigma BELOW the fitted curve (a dispersive glitch) are dropped for a
    second pass; positive misfit is never dropped.  Returns {area, lo, hi,
    fwhm_hz, eta, rmse_rel, n_rejected, centre} or None when the residual
    exceeds ``max_rmse`` of the signal's maximum."""
    from scipy.optimize import least_squares
    if not lines:
        return None
    sf = spec.exp.sfo1 or 500.0
    w = float(np.median([signal_fwhm(spec, p, sf) for p in lines]))     # the signal's width, not a noise spike's
    w_lo, w_hi = signal_window(spec, lines, w, k)
    x0, half = 0.5 * (w_lo + w_hi), 0.5 * (w_hi - w_lo)
    m = (spec.ppm >= w_lo) & (spec.ppm <= w_hi)
    n_t = len(lines)
    # neighbouring lines inside the window are fitted too (nuisance), so that
    # their wings are neither counted as signal nor mistaken for baseline
    others, _ = find_lines(spec.ppm, spec.real, w_lo, w_hi, sf)
    nuis = [p for p, h in others if all(abs(p - t) > 1.5 * w for t in lines)]
    lines = list(lines) + nuis
    n = len(lines)
    if m.sum() < 4 * n + 5:
        return None
    x = spec.ppm[m]
    y = spec.real[m].astype(float)
    top = float(y.max()) or 1.0
    basis, use_tail = _baseline_basis(spec, x0, half, top)
    B = basis(x)
    nb = B.shape[1]
    LN2 = np.log(2.0)

    def model(p):
        out = B @ p[4 * n:4 * n + nb]
        for i in range(n):
            A, c0, g, eta = p[4 * i:4 * i + 4]
            d = (x - c0) / g
            out = out + A * (eta / (1.0 + d * d) + (1.0 - eta) * np.exp(-LN2 * d * d))
        return out

    p0, lb, ub = [], [], []
    for pos in lines:
        p0 += [float(np.interp(pos, x[::-1], y[::-1])), float(pos), 0.5 * w, 0.8]
        lb += [0.0, pos - 2 * w, 0.1 * w, 0.0]
        ub += [np.inf, pos + 2 * w, 8.0 * w, 1.0]
    p0 += [0.0] * nb; lb += [-np.inf] * nb; ub += [np.inf] * nb
    keep = np.ones(y.size, bool)
    try:
        r = least_squares(lambda p: model(p)[keep] - y[keep], p0, bounds=(lb, ub), max_nfev=600)
        p = r.x
        noise = float(np.std(np.diff(y)) / np.sqrt(2.0)) or 1e-30
        res = y - model(p)
        keep = res > -4.0 * noise
        n_rej = int((~keep).sum())
        if n_rej:
            r = least_squares(lambda p: model(p)[keep] - y[keep], p, bounds=(lb, ub), max_nfev=600)
            p = r.x
    except Exception:
        return None
    rmse = float(np.sqrt(np.mean(r.fun ** 2))) / top
    areas = [p[4 * i] * p[4 * i + 2] * (p[4 * i + 3] * np.pi + (1.0 - p[4 * i + 3]) * _GAUSS_AREA) for i in range(n_t)]
    area = float(sum(areas))
    if not np.isfinite(area) or area <= 0 or rmse > max_rmse:
        return None
    centre = float(sum(a * p[4 * i + 1] for i, a in enumerate(areas)) / area)
    # share of the fitted signal that lies inside the window (a numerical area
    # over the same window can only be compared with the fit through it)
    sig_in = np.zeros_like(x)
    for i in range(n_t):
        A, c0, g, eta = p[4 * i:4 * i + 4]
        d = (x - c0) / g
        sig_in += A * (eta / (1.0 + d * d) + (1.0 - eta) * np.exp(-LN2 * d * d))
    frac = float(np.trapz(sig_in[::-1], x[::-1]) / area) if area > 0 else 1.0
    # what a numerical integration of the FITTED curve over this window gives, with the
    # same edge-anchored baseline as the data: the fair reference for the data's area
    mt = model(p)
    e = max(1, min(5, x.size // 4))
    edge = np.linspace(float(np.median(mt[:e])), float(np.median(mt[-e:])), x.size)
    num_expected = float(np.trapz((mt - edge)[::-1], x[::-1]))
    # the fitted model itself, so that it can be drawn: the signal's own lines,
    # the neighbours fitted alongside, and the baseline (line + solvent tail)
    xs = float(spec.ppm[int(np.argmax(spec.real))]) if use_tail else None
    fit_model = {"lines": [[float(v) for v in p[4 * i:4 * i + 4]] for i in range(n_t)],
                 "nuis": [[float(v) for v in p[4 * i:4 * i + 4]] for i in range(n_t, n)],
                 "x0": float(x0), "c": [float(v) for v in p[4 * n:4 * n + nb]], "xs": xs,
                 "lo": float(w_lo), "hi": float(w_hi)}
    return {"area": area, "lo": float(w_lo), "hi": float(w_hi), "frac_in_window": min(max(frac, 1e-3), 1.0),
            "num_expected": num_expected, "model": fit_model,
            "fwhm_hz": float(np.mean([2.0 * p[4 * i + 2] for i in range(n_t)]) * sf),
            "eta": float(np.mean([p[4 * i + 3] for i in range(n_t)])), "rmse_rel": rmse,
            "n_rejected": n_rej, "n_nuisance": len(nuis), "centre": centre}
