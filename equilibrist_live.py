"""Live view of a running fit.

Every fit objective reports each evaluation here (``notify``); the app installs
a hook for the duration of a fit (``live_hook``) that, every "Refresh" seconds,
re-evaluates the model at the best point so far and redraws the data-vs-model
plot.  The hook is disabled while it runs (it evaluates the model itself, which
would otherwise notify again), and the global fit suppresses the per-tab
notifications (``suppressed``) and reports its own point instead
(``notify_global``).
"""
from __future__ import annotations
import functools

_HOOK = None          # callable(values: dict, objective: float) or None
_DEPTH = 0            # > 0: per-tab notifications are ignored (inside a global fit)


def notify(values: dict, objective: float) -> None:
    """Called by a fit function's objective on every evaluation."""
    if _HOOK is None or _DEPTH > 0:
        return
    _call(values, objective)


def notify_global(values: dict, objective: float, detail: str | None = None) -> None:
    """Called by the global fit after every evaluation with its CURRENT BEST point
    and objective (not suppressed by ``suppressed``).  The report is
    authoritative: the global objective changes scale when the tab weights are
    re-estimated, so a hook must not compare it with earlier values.  ``detail``
    is a short text the hook may show instead of the number (the rms per tab)."""
    if _HOOK is None:
        return
    _call(values, objective, True, detail)


def _call(values, objective, authoritative=False, detail=None):
    global _HOOK
    h, _HOOK = _HOOK, None          # the hook may evaluate the model: no re-entry
    try:
        h(dict(values), float(objective), authoritative, detail)
    except Exception:
        pass
    finally:
        _HOOK = h


class live_hook:
    """``with live_hook(fn): run the fit`` -- installs ``fn`` for the duration."""
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        global _HOOK
        self._prev, _HOOK = _HOOK, self.fn
        return self

    def __exit__(self, *exc):
        global _HOOK
        _HOOK = self._prev
        return False


def suppressed(func):
    """Decorator: per-tab notifications are ignored while ``func`` runs."""
    @functools.wraps(func)
    def wrapper(*a, **k):
        global _DEPTH
        _DEPTH += 1
        try:
            return func(*a, **k)
        finally:
            _DEPTH -= 1
    return wrapper
