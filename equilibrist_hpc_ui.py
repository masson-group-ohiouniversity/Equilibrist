"""
equilibrist_hpc_ui.py — Streamlit panel for submitting Equilibrist's parallel
tools to a SLURM cluster.  All non-UI logic lives in ``equilibrist_hpc``.

Three calls wire one tool ("site") of the app:

    _hpc_go = hpc_ui.controls("bs_eq")                  # under the Parallel box
    if st.button("Run bootstrap") or _hpc_go:
        try:
            ... the existing tool call, unchanged ...
        except hpc_ui.CapturedCall as _cc:              # before `except Exception`
            hpc_ui.handle_capture(_cc, "bs_eq", "Bootstrap", context={...})
        except Exception as e:
            ...
    hpc_ui.jobs_panel("bs_eq", on_result=lambda result, ctx: ...)

Secrets: the password and second-factor fields live only in Streamlit's
in-memory session state under the ``_hpc_`` prefix, which the FAIR session
saver never captures.  They are never written to disk or to job records.

© Eric Masson, Ohio University, 2026
"""
from __future__ import annotations

import traceback

import streamlit as st

import equilibrist_hpc as hpc
from equilibrist_hpc import CapturedCall  # re-exported for the app's except clauses

__all__ = ["controls", "handle_capture", "jobs_panel", "follower", "CapturedCall"]

_SHARED = "_hpc_shared"          # settings + secrets shared by every site
_SEEN = "_hpc_seen"              # last value each site-widget showed
_REMOTE = "_hpc_remote"          # live connection for this browser session
_SECRETS = ("password", "otp")

_FRIENDLY = {
    "NEW": "not submitted", "SUBMITTED": "submitted", "PENDING": "waiting in queue",
    "CONFIGURING": "starting", "RUNNING": "running", "COMPLETING": "finishing",
    "DONE": "finished — ready to fetch", "FETCHED": "fetched", "ERROR": "failed",
    "FAILED": "failed", "CANCELLED": "cancelled", "MISSING": "job folder not found",
    "TIMEOUT": "stopped — wall time exceeded",
}


def _btn(container, label: str, **kw) -> bool:
    """Full-width button that works on old and new Streamlit versions."""
    try:
        return container.button(label, width="stretch", **kw)
    except TypeError:
        return container.button(label, use_container_width=True, **kw)


# ── shared settings ───────────────────────────────────────────────────────
def _shared() -> dict:
    if _SHARED not in st.session_state:
        s = hpc.load_config()
        for k in _SECRETS:
            s[k] = ""
        st.session_state[_SHARED] = s
        st.session_state[_SEEN] = {}
    return st.session_state[_SHARED]


def _field(kind: str, label: str, name: str, site: str, **kw):
    """A widget bound to one shared setting.  The same setting is shown at
    every site; whichever site the user edits wins, the others follow."""
    shared, seen = _shared(), st.session_state[_SEEN]
    key = f"_hpc_{name}_{site}"
    if key in st.session_state and st.session_state[key] != seen.get(key, st.session_state[key]):
        shared[name] = st.session_state[key]              # edited here
    else:
        st.session_state[key] = shared.get(name, kw.pop("default", ""))
    kw.pop("default", None)
    if kind == "text":
        val = st.text_input(label, key=key, **kw)
    elif kind == "password":
        val = st.text_input(label, key=key, type="password", **kw)
    elif kind == "int":
        val = st.number_input(label, key=key, step=1, **kw)
    elif kind == "area":
        val = st.text_area(label, key=key, **kw)
    else:
        raise ValueError(kind)
    shared[name] = val
    seen[key] = val
    return val


def _settings_for_job() -> dict:
    return {k: v for k, v in _shared().items() if k not in _SECRETS}


# ── connection ────────────────────────────────────────────────────────────
def _connect(site: str):
    """Return a live connection, (re)opening it if needed.  Shows the reason
    and returns ``None`` when it cannot connect."""
    s = _shared()
    # Developer test mode: EQUILIBRIST_HPC_TESTDIR=<dir> makes "the cluster" a
    # local folder whose bin/ holds stand-in sbatch / srun / squeue / sacct
    # scripts, so the whole submit → status → fetch flow runs without SSH.
    import os
    _td = os.environ.get("EQUILIBRIST_HPC_TESTDIR")
    if _td:
        return hpc.LocalRemote(_td, env={"PATH": f"{_td}/bin:" + os.environ.get("PATH", ""),
                                         "FAKESLURM_STATE": f"{_td}/state"})
    rem = st.session_state.get(_REMOTE)
    if (rem is not None and getattr(rem, "connected", False)
            and rem.host == s.get("host") and rem.username == s.get("username")):
        return rem
    if not str(s.get("host", "")).strip() or not str(s.get("username", "")).strip():
        st.error("Enter the cluster's host name and your username first.")
        return None
    trust = bool(st.session_state.get(f"_hpc_trust_{site}", False))
    try:
        rem = hpc.SSHRemote().connect(
            s["host"].strip(), s["username"].strip(),
            password=(s.get("password") or None), otp=(s.get("otp") or None),
            port=int(s.get("port", 22) or 22), trust_new_host=trust)
    except hpc.UnknownHostKey as e:
        st.session_state["_hpc_unknown_host"] = {
            "host": e.host, "type": e.key_type, "fp": e.fingerprint}
        st.warning(
            f"This is the first connection to **{e.host}**.  Its {e.key_type} key "
            f"fingerprint is\n\n`SHA256:{e.fingerprint}`\n\nCompare it with the "
            "fingerprint published by your computing centre (OSC lists them on "
            "each cluster's \"SSH key fingerprints\" page).  If it matches, tick "
            "**Trust this host** in the connection settings and try again.")
        return None
    except hpc.HPCError as e:
        st.error(str(e))
        return None
    except Exception as e:                                   # network, paramiko…
        st.error(f"Connection failed: {e}")
        return None
    st.session_state[_REMOTE] = rem
    st.session_state.pop("_hpc_unknown_host", None)
    hpc.save_config(_settings_for_job())
    return rem


# ── 1. controls (under the "Parallel" checkbox) ───────────────────────────
def controls(site: str, *, label: str = "", submits=None):
    """Render the supercomputer controls for one tool.  Returns True on the
    rerun in which the user pressed *Submit*; in that case a capture is armed
    and the tool call that follows will raise ``CapturedCall``.

    ``submits`` -- optional list of ``(key, label)`` pairs: one Submit button
    per entry (e.g. this tab's fit / the global fit); the return value is then
    the key of the button pressed, or None."""
    hpc.disarm_capture()
    if not st.checkbox("🖥️ Run on a supercomputer (SLURM)", value=False,
                       key=f"_hpc_on_{site}",
                       help="Send this calculation to a SLURM cluster instead "
                            "of running it on this computer.  Every refit "
                            "becomes one task, spread over nodes × CPUs."):
        return False

    s = _shared()
    with st.expander("Supercomputer connection & resources", expanded=True):
        presets = list(hpc.PRESETS.keys())
        pkey = f"_hpc_preset_{site}"
        cur = s.get("preset") if s.get("preset") in presets else presets[0]
        chosen = st.selectbox("Cluster preset", presets, index=presets.index(cur), key=pkey,
                              help="Fills in sensible defaults.  Every field "
                                   "below stays editable.")
        if chosen != s.get("preset"):
            s["preset"] = chosen
            s.update(hpc.PRESETS[chosen])
            st.session_state[_SEEN] = {}                     # push new defaults to widgets
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("_hpc_") and k.endswith(f"_{site}") \
                        and k not in (pkey, f"_hpc_on_{site}"):
                    del st.session_state[k]
            _rerun_panel()

        # The app's right-hand column is narrow: at most two fields per row.
        _field("text", "Host", "host", site, placeholder="pitzer.osc.edu")
        c1, c2 = st.columns(2)
        with c1:
            _field("text", "Username", "username", site)
        with c2:
            _field("password", "Password", "password", site,
                   help="Kept in memory for this browser session only — never "
                        "written to disk, to saved sessions or to job records. "
                        "Leave empty if this computer is already remembered "
                        "by the cluster (SSH key).")
        _field("text", "Second factor (only if your cluster asks)", "otp", site,
               help="A passcode, or the option number for a push (often 1).")
        unk = st.session_state.get("_hpc_unknown_host")
        if unk:
            st.checkbox(f"Trust this host ({unk['host']}, SHA256:{unk['fp'][:16]}…)",
                        key=f"_hpc_trust_{site}")

        c1, c2 = st.columns(2)
        with c1:
            _field("int", "Nodes", "nodes", site, min_value=1, max_value=512)
            _field("text", "Wall time", "walltime", site, help="HH:MM:SS")
        with c2:
            _field("int", "CPUs per node", "cpus_per_node", site, min_value=1, max_value=1024)
            _field("text", "Project", "account", site,
                   help="The allocation charged for the job (required at OSC, "
                        "e.g. PAS1234).")
        _wt = hpc.walltime_seconds(s.get("walltime", ""))
        if _wt is None:
            st.warning("Wall time is not in a form SLURM understands (use HH:MM:SS).")
        elif _wt < 1800:
            st.warning(f"Wall time is only {_wt // 60} min.  The scheduler stops the job "
                       "when it expires, however far the calculation has got.  One "
                       "hour or more is a safer request; unused time is not charged.")
        n_tasks = int(s.get("nodes", 1) or 1) * int(s.get("cpus_per_node", 1) or 1)
        st.caption(f"{n_tasks} refits run at the same time.  More tasks than "
                   "refits brings no further gain; one refit is the floor on "
                   "the run time.")

        if st.checkbox("Advanced settings", value=False, key=f"_hpc_adv_{site}"):
            _field("area", "Environment setup (runs before Python in every job)",
                   "setup_cmds", site, height=90)
            _field("area", "Commands that make `conda` available (one-time install)",
                   "conda_cmds", site, height=70)
            _field("area", "Extra #SBATCH lines", "extra_sbatch", site, height=70,
                   help="One option per line, e.g.  --mem=0  or  --constraint=48core")
            _field("text", "Partition (optional)", "partition", site)
            c1, c2 = st.columns(2)
            with c1:
                _field("text", "Remote folder", "remote_base", site)
                _field("text", "Environment name", "env_name", site)
            with c2:
                _field("text", "Python command", "python_cmd", site)

        b1 = b2 = b3 = b4 = st
        if _btn(b1, "Test connection", key=f"_hpc_test_{site}"):
            rem = _connect(site)
            if rem is not None:
                try:
                    who = rem.sh("hostname; command -v sbatch || echo 'sbatch NOT found'").strip()
                    st.success(f"Connected ({rem.auth_method}).\n\n```\n{who}\n```")
                except Exception as e:
                    st.error(str(e))
        if _btn(b2, "Remember this computer", key=f"_hpc_key_{site}",
                     help="Uses your password once to authorise an SSH key "
                          "stored in ~/.equilibrist.  Afterwards the password "
                          "field can stay empty."):
            rem = _connect(site)
            if rem is not None:
                try:
                    kp = hpc.install_ssh_key(rem)
                    st.success(f"Done.  Key stored at `{kp}`; future connections "
                               "from this computer need no password.")
                except Exception as e:
                    st.error(f"Could not install the key: {e}")
        if _btn(b3, "Install environment on the cluster", key=f"_hpc_inst_{site}",
                     help="One-time: creates the Python environment Equilibrist "
                          "needs on the cluster.  Takes several minutes; use "
                          "'Check install' to follow it."):
            rem = _connect(site)
            if rem is not None:
                try:
                    where = hpc.start_env_install(rem, _settings_for_job())
                    st.info(f"Installation started in `{where}`.")
                except Exception as e:
                    st.error(str(e))
        if _btn(b4, "Check the installation", key=f"_hpc_chk_{site}"):
            rem = _connect(site)
            if rem is not None:
                try:
                    r = hpc.check_env_install(rem, _settings_for_job())
                    (st.success if r["state"] == "ready" else
                     st.error if r["state"] == "failed" else st.info)(
                        f"Environment: **{r['state']}**")
                    if r["log"]:
                        st.code(r["log"])
                except Exception as e:
                    st.error(str(e))

        if st.checkbox("Preview the batch script", value=False, key=f"_hpc_prev_{site}"):
            st.code(hpc.render_batch_script(
                _settings_for_job(), job_id="PREVIEW", tool=label or site,
                job_dir_abs="$HOME/<remote folder>/jobs/<job id>",
                code_dir_abs="$HOME/<remote folder>/code_<version>"), language="bash")

    if submits:
        st.caption("A fit is one sequential task: it is submitted on 1 node × 1 CPU whatever "
                   "the fields above say, and runs for the wall time entered above (minus a "
                   "margin for start-up and the final statistics).  The Timeout next to Tol "
                   "applies to fits run on this computer only.")
        pressed = None
        for key, lab in submits:
            if _btn(st, f"🚀 Submit {lab} to the supercomputer",
                    key=f"_hpc_submit_{site}_{key}", type="primary"):
                pressed = key
        if pressed:
            hpc.arm_capture(site)
        return pressed
    go = _btn(st, f"🚀 Submit {label or 'this calculation'} to the supercomputer",
              key=f"_hpc_submit_{site}", type="primary")
    if go:
        hpc.arm_capture(site)
    return bool(go)


def _fit_token(site: str) -> str:
    """Fingerprint of the fit a tool is operating on, so a result fetched
    later can be checked against the fit that is current at that moment."""
    import hashlib, json
    fa = st.session_state.get("_fit_args_kin" if site.endswith("_kin") else "_fit_args_eq") or {}
    start = fa.get("start_logK") or fa.get("start_logk") or {}
    try:
        blob = json.dumps({"k": sorted(map(str, fa.get("fit_keys") or [])),
                           "c": sorted(map(str, fa.get("fit_conc_keys") or [])),
                           "s": {str(k): round(float(v), 6) for k, v in sorted(start.items())},
                           "n": int(fa.get("n_data") or 0)}, sort_keys=True)
    except Exception:
        return ""
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _flash(site: str, kind: str, text: str) -> None:
    st.session_state[f"_hpc_flash_{site}"] = (kind, text)


# ── 2. what to do with a captured call ────────────────────────────────────
def fit_settings() -> dict:
    """Resources a fit gets whatever the panel says: one sequential task."""
    return {"nodes": 1, "cpus_per_node": 1}


def walltime_seconds():
    """The wall time entered in the panel, in seconds (None if unreadable)."""
    return hpc.walltime_seconds(_shared().get("walltime", ""))


def fit_timeout_seconds(fallback: float) -> float:
    """How long a fit may run on the cluster: the panel's wall time minus a
    margin for the environment start-up and the final statistics (15 %, at
    least 3 min); ``fallback`` when no wall time is readable."""
    wt = walltime_seconds()
    if not wt:
        return float(fallback)
    return float(max(60.0, wt - max(180.0, 0.15 * wt)))


def handle_capture(cc: CapturedCall, site: str, tool: str, context: dict | None = None,
                   settings_override: dict | None = None) -> None:
    """Ship a captured tool call to the cluster and record the job.
    ``settings_override`` replaces panel fields for this job only (a fit
    always asks for 1 node × 1 CPU)."""
    hpc.disarm_capture()
    try:
        job = hpc.build_job(cc, tool=tool, site=site, context=context or {})
    except TypeError as e:
        st.error(f"This calculation cannot be sent to a cluster: {e}")
        return
    rem = _connect(site)
    if rem is None:
        return
    try:
        with st.spinner("Uploading and submitting…"):
            _settings = _settings_for_job()
            if settings_override:
                _settings.update(settings_override)
            rec = hpc.submit_job(rem, job, _settings)
    except Exception as e:
        st.error(f"Submission failed: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return
    rec["fit_token"] = _fit_token(site)
    jobs = hpc.load_jobs()
    jobs.append(rec)
    hpc.save_jobs(jobs)
    _flash(site, "success",
           f"Submitted as SLURM job **{rec['slurm_id']}** on {rec['host']} "
           f"({rec['nodes']} × {rec['cpus_per_node']} CPUs).  You can close the "
           "browser; the job is listed here whenever you come back.")
    _rerun_app()                # starts the live follower at the top of the column


# ── 3. job list for one site ──────────────────────────────────────────────
_ACTIVE = ("SUBMITTED", "PENDING", "CONFIGURING", "RUNNING", "COMPLETING",
           "SUSPENDED", "REQUEUED", "RESIZING")
_POLL_SECONDS = 5           # folder look (a cheap file listing on the login node)
_SCHED_SECONDS = 60         # squeue / sacct — kept rare, as cluster admins ask


def _save_records(records) -> None:
    """Merge updated records into the job store (other sites' jobs kept)."""
    by_id = {j["id"]: j for j in hpc.load_jobs()}
    for r in records:
        by_id[r["id"]] = r
    hpc.save_jobs(list(by_id.values()))


def _rerun_panel() -> None:
    """Refresh only the enclosing fragment (no page redraw, no scroll jump);
    fall back to a full rerun where the panel is not inside a fragment."""
    try:
        st.rerun(scope="fragment")
    except Exception as e:
        if type(e).__name__ in ("RerunException", "StopException"):
            raise
        _rerun_app()


def _rerun_app() -> None:
    """Full rerun, asking the next run to put the page back where it was."""
    st.session_state["_hpc_restore_scroll"] = True
    st.rerun()


_SCROLL_JS_PARENT = r"""
(function () {
  if (window.__eqbScrollV2) return; window.__eqbScrollV2 = true;
  var KEY = 'equilibrist_scroll_y', userT = 0, restoring = false;
  function main() {
    return document.querySelector('[data-testid="stMain"]') ||
           document.querySelector('section.main') || document.scrollingElement;
  }
  ['wheel', 'touchmove', 'keydown', 'mousedown'].forEach(function (ev) {
    document.addEventListener(ev, function () { userT = Date.now(); }, {passive: true, capture: true});
  });
  document.addEventListener('mousemove', function (e) {        // dragging the scrollbar
    if (e.buttons) userT = Date.now();
  }, {passive: true, capture: true});
  // Record the position only while the USER is moving the page.  A redraw
  // moves it too (to the top, or clamped while the page is briefly shorter);
  // those positions are not wishes and must not be remembered.
  var first = main();
  if (first && first.scrollTop >= 5) sessionStorage.setItem(KEY, String(first.scrollTop));
  setInterval(function () {
    var el = main(); if (!el || restoring) return;
    if (Date.now() - userT > 1500) return;
    sessionStorage.setItem(KEY, String(el.scrollTop));
  }, 250);
  window.__eqbRestore = function () {
    var y = parseFloat(sessionStorage.getItem(KEY) || '0');
    if (!(y > 0)) return;
    restoring = true;
    [60, 250, 600, 1200, 2200, 3500].forEach(function (t) {
      setTimeout(function () {
        var el = main();
        if (el && Date.now() - userT > 1500 && Math.abs(el.scrollTop - y) > 4) el.scrollTop = y;
      }, t);
    });
    setTimeout(function () { restoring = false; }, 3800);
  };
})();
"""


def _scroll_keeper() -> None:
    """Remember the page's scroll position in the browser and, after a full
    rerun requested by this panel, restore it.  (A full rerun redraws the
    plots, which makes the page jump to the top.)

    The recorder is injected into the MAIN page as a script element.  Code
    that merely runs inside the helper frame would stop working as soon as
    Streamlit re-creates that frame."""
    import json
    restore = bool(st.session_state.pop("_hpc_restore_scroll", False))
    try:
        import streamlit.components.v1 as components
    except Exception:
        return
    components.html(
        "<script>(function(){try{var P=window.parent,D=P.document;"
        "if(!P.__eqbScrollV2){var s=D.createElement('script');"
        "s.textContent=" + json.dumps(_SCROLL_JS_PARENT) + ";D.head.appendChild(s);}"
        + ("if(P.__eqbRestore)P.__eqbRestore();" if restore else "")
        + "}catch(e){}})();</script>", height=0)


def _age_seconds(iso: str) -> float:
    from datetime import datetime
    try:
        return (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except Exception:
        return 1e9


def _fmt_span(sec: float) -> str:
    sec = max(float(sec), 0.0)
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    return f"{sec / 3600:.1f} h"


def _progress_text(rec: dict) -> tuple:
    """(fraction, caption) for the live bar of an active job."""
    status = rec.get("status", "")
    cnt, tot = (rec.get("progress") or [0, 0])[:2]
    if status in ("SUBMITTED", "PENDING", "CONFIGURING"):
        return 0.0, f"Waiting in the queue · {_fmt_span(_age_seconds(rec.get('submitted', '')))}"
    ran = _age_seconds(rec.get("running_since") or rec.get("submitted", ""))
    if not tot:
        prog = (rec.get("progress_text") or "").strip()
        if prog:                                   # a fit: "<fraction> <message>"
            frac_s, _, msg = prog.partition(" ")
            try:
                frac = min(max(float(frac_s), 0.0), 1.0)
            except ValueError:
                frac, msg = 0.0, prog
            return frac, f"{msg.strip()} · running {_fmt_span(ran)}"
        return 0.0, f"Running · starting up · {_fmt_span(ran)}"
    text = f"{rec.get('tool', 'Job')}: {cnt}/{tot} refits · running {_fmt_span(ran)}"
    if 0 < cnt < tot:
        text += f" · about {_fmt_span(ran * (tot - cnt) / cnt)} left"
    return min(cnt / max(tot, 1), 1.0), text


def _current(mine: list) -> list:
    """The jobs worth showing: queued or running ones, a finished job whose
    results have not been loaded yet, and the most recent job if it stopped
    early (so it can be resumed or its error read).  Fetched, cancelled and
    older jobs stay in the local record but are not listed."""
    keep = [j for j in mine if j.get("status") in _ACTIVE or j.get("status") == "DONE"]
    if mine and mine[-1].get("status") in ("TIMEOUT", "FAILED", "ERROR", "MISSING") \
            and mine[-1] not in keep:
        keep.append(mine[-1])
    return keep


def _poll(rem, rec) -> None:
    """One look at a followed job: the folder every time, the scheduler only
    once per ``_SCHED_SECONDS``."""
    from datetime import datetime
    # Ask the scheduler every 20 s until the job runs (queue → running is the
    # transition people wait for), then only once a minute.
    every = _SCHED_SECONDS if rec.get("status") == "RUNNING" else 20
    sched = _age_seconds(rec.get("checked_scheduler") or "") >= every
    hpc.job_status(rem, rec, scheduler=sched)
    if rec.get("status") == "RUNNING" and not rec.get("running_since"):
        rec["running_since"] = datetime.now().isoformat(timespec="seconds")


# ── live follow-up of queued / running jobs ───────────────────────────────
def follower() -> None:
    """Live progress for every queued or running cluster job, from any tool.

    MUST be called from the top level of the app script (not from inside a
    fragment).  The timed refresh is a Streamlit fragment; Streamlit only
    retires the timers of *top-level* fragments when the page is rebuilt.  A
    timed fragment nested inside another fragment keeps an outdated position
    after its parent redraws, and its next tick crashes the page ("Bad delta
    path index").  Hence one top-level follower instead of a timer per tool.

    A timed refresh never interrupts a running script, so a fit started while
    a job is being followed is not disturbed."""
    active = [j for j in hpc.load_jobs()
              if not j.get("hidden") and j.get("status") in _ACTIVE]
    if not active:
        return
    # A keyed container gets the CSS class st-key-<key>; pin it to the
    # bottom-right corner of the window so it is visible at any scroll
    # position of the page or the sidebar.
    st.markdown("""<style>
    .st-key-hpc_follower {
        position: fixed; right: 18px; bottom: 18px; width: min(360px, 60vw);
        z-index: 1000; padding: 10px 14px 6px 14px; border-radius: 10px;
        background: var(--secondary-background-color, #f0f2f6);
        border: 1px solid rgba(128, 128, 128, 0.35);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
    }
    .st-key-hpc_follower p { margin-bottom: 0.25rem; }
    </style>""", unsafe_allow_html=True)
    try:
        box = st.container(key="hpc_follower")
    except TypeError:                       # older Streamlit: no keyed containers
        box = st.container()
    frag = getattr(st, "fragment", None)
    with box:
        if frag is not None:
            frag(run_every=_POLL_SECONDS)(_follower_body)()
        else:
            _follower_body()


def _follower_body() -> None:
    active = [j for j in hpc.load_jobs()
              if not j.get("hidden") and j.get("status") in _ACTIVE]
    if not active:
        return
    due = [r for r in active if _age_seconds(r.get("checked") or "") >= _POLL_SECONDS - 2]
    s = _shared()
    if due and (not str(s.get("host", "")).strip() or not str(s.get("username", "")).strip()):
        st.caption("Cluster jobs are listed but no cluster connection is configured; "
                   "open a tool's supercomputer panel to enter host and username.")
        due = []
    if due:
        rem = _connect("follower")
        if rem is not None:
            for rec in due:
                try:
                    _poll(rem, rec)
                except Exception as e:
                    st.caption(f"Could not reach the cluster just now ({e}); will retry.")
            _save_records(due)
    st.markdown("**🖥️ Supercomputer**")
    for rec in active:
        if rec.get("status") in _ACTIVE:
            frac, text = _progress_text(rec)
            st.progress(frac, text=f"{rec.get('slurm_id', '')} · {text}")
    if any(r.get("status") not in _ACTIVE for r in active):
        # a job has ended: rebuild the page once so that its tool panel loads
        # the results (or shows why it stopped) and this follower retires
        _rerun_app()


# ── 3. job list for one tool ──────────────────────────────────────────────
def jobs_panel(site: str, on_result) -> None:
    """List this tool's current cluster jobs.  ``on_result(result, context)``
    is called when a finished job is loaded; it should store the result where
    the local run would have.

    Plain content — no timer here (see ``follower``).  A finished job is
    loaded automatically the next time this panel is drawn, which the
    follower triggers as soon as the job ends."""
    from datetime import datetime
    flash = st.session_state.pop(f"_hpc_flash_{site}", None)
    if flash:
        getattr(st, flash[0], st.info)(flash[1])
    jobs = hpc.load_jobs()
    mine = _current([j for j in jobs if j.get("site") == site and not j.get("hidden")])
    if not mine:
        return
    _scroll_keeper()
    st.markdown("**Supercomputer job**" if len(mine) == 1 else "**Supercomputer jobs**")
    token_now = _fit_token(site)

    # ── load a finished job straight away ────────────────────────────────
    for rec in mine:
        stale = bool(rec.get("fit_token")) and bool(token_now) and rec["fit_token"] != token_now
        if rec.get("status") == "DONE" and not rec.get("auto_fetched") and not stale:
            rem = _connect(site)
            if rem is None:
                break
            try:
                result, meta = hpc.fetch_result(rem, rec)
                on_result(result, rec.get("context") or {})
                rec["auto_fetched"] = True
                _save_records([rec])
                _flash(site, "success",
                       f"Supercomputer job {rec.get('slurm_id')} finished — results "
                       f"loaded ({meta.get('ranks', '?')} tasks, "
                       f"{_fmt_span(float(meta.get('wall_seconds', 0)))} on the cluster).")
                _rerun_panel()            # draw the result with the tool's own code
            except Exception as e:
                if type(e).__name__ in ("RerunException", "StopException"):
                    raise
                rec["auto_fetched"] = True          # do not retry in a loop
                _save_records([rec])
                st.error(f"Could not fetch the results: {e}")

    changed = False
    restart_follower = False
    for rec in reversed(mine[-8:]):
        jid = rec["id"]
        status = rec.get("status", "")
        prog = rec.get("progress") or [0, 0]
        line = (f"`{rec.get('slurm_id', '—')}` · {rec.get('tool', '')} · "
                f"{rec.get('host', '')} · {rec.get('nodes', '?')} × {rec.get('cpus_per_node', '?')} CPUs · "
                f"submitted {rec.get('submitted', '')[:16].replace('T', ' ')} "
                f"· **{_FRIENDLY.get(status, status.lower() or 'unknown')}**")
        if status in ("ERROR", "FAILED", "TIMEOUT") and rec.get("scheduler_state"):
            line += f" (scheduler: {rec['scheduler_state']})"
        st.markdown(line)
        if status in _ACTIVE:
            st.caption("Live progress is shown in the panel at the bottom-right "
                       "of the window; the results load here as soon as the job ends.")
        elif prog[1]:
            st.caption(f"{prog[0]}/{prog[1]} refits completed")
        c1, c2 = st.columns(2)
        c3, c4 = c1, c2
        if _btn(c1, "Check status", key=f"_hpc_st_{jid}"):
            rem = _connect(site)
            if rem is not None:
                try:
                    hpc.job_status(rem, rec); changed = True
                    if rec.get("status") == "RUNNING" and not rec.get("running_since"):
                        rec["running_since"] = datetime.now().isoformat(timespec="seconds")
                except Exception as e:
                    st.error(str(e))
        stale = bool(rec.get("fit_token")) and bool(token_now) and rec["fit_token"] != token_now
        if stale and status in ("DONE", "FETCHED"):
            st.warning("This job was submitted from a different fit than the one "
                       "currently shown.  Its results describe that earlier fit.")
            allow = st.checkbox("Load them anyway", key=f"_hpc_stale_{jid}")
        else:
            allow = True
        if _btn(c2, "Fetch results", key=f"_hpc_fx_{jid}",
                     disabled=(status not in ("DONE", "FETCHED")) or not allow):
            rem = _connect(site)
            if rem is not None:
                try:
                    with st.spinner("Downloading…"):
                        result, meta = hpc.fetch_result(rem, rec)
                    on_result(result, rec.get("context") or {})
                    rec["auto_fetched"] = True
                    changed = True
                    _flash(site, "success",
                           f"Results loaded ({meta.get('ranks', '?')} tasks, "
                           f"{_fmt_span(float(meta.get('wall_seconds', 0)))} on the cluster).")
                except Exception as e:
                    st.error(f"Could not fetch the results: {e}")
        if _btn(c3, "Cancel job", key=f"_hpc_cx_{jid}",
                     disabled=status in ("DONE", "FETCHED", "CANCELLED", "ERROR", "FAILED", "TIMEOUT")):
            rem = _connect(site)
            if rem is not None:
                try:
                    hpc.cancel_job(rem, rec); changed = True
                except Exception as e:
                    st.error(str(e))
        if _btn(c4, "Remove", key=f"_hpc_rm_{jid}"):
            rec["hidden"] = True; changed = True
        if status in ("TIMEOUT", "CANCELLED", "FAILED", "ERROR") and rec.get("code_digest"):
            s_now = _settings_for_job()
            if _btn(st, f"Resume with {int(s_now.get('nodes', 1))} × "
                        f"{int(s_now.get('cpus_per_node', 1))} CPUs for {s_now.get('walltime', '')}",
                    key=f"_hpc_rs_{jid}",
                    help="Submits the job again in the same folder on the cluster.  "
                         "Refits that were already finished are kept; only the "
                         "missing ones are computed.  Nodes, CPUs and wall time are "
                         "taken from the settings above."):
                rem = _connect(site)
                if rem is not None:
                    try:
                        hpc.resume_job(rem, rec, s_now); changed = True; restart_follower = True
                        _flash(site, "success", f"Resumed as SLURM job {rec['slurm_id']}; "
                                                f"{prog[0]} finished refits are kept.")
                    except Exception as e:
                        st.error(f"Could not resume: {e}")
        if rec.get("error") and status in ("ERROR", "FAILED", "TIMEOUT", "CANCELLED"):
            with st.expander("What went wrong" if status != "CANCELLED" else "Scheduler record"):
                st.code(rec["error"])
    if changed:
        _save_records(mine)
        if restart_follower:
            _rerun_app()        # a resumed job is active again: bring the follower back
        else:
            _rerun_panel()      # status / fetch / cancel / remove: this panel only
