"""Operational knobs of the served self-play runner (run_sp_served.py), with NO import-time side effects.

R3-TOOLS-MISC (R2-SL-28 FU3): the runner validates these knobs before it reserves its output, but the loop
(run_td_v33loop.py) launched it, and published its pre-ready marker, without checking them, because importing
run_sp_served into the loop would os.environ.setdefault WOTR_ACT_GEN=v41 and other WOTR_* values in the loop process.
This module holds the single implementation both use: the runner on its own os.environ, the loop on the env dict it is
about to hand the runner, before launching().  Every function reads the given mapping (default: os.environ, read at
call time) and never writes it.
"""
import math
import os

MAX_GAME_FAILURES_ENV = 'WOTR_SP_MAX_GAME_FAILURES'
SEARCH_MAX_SECONDS_ENV = 'WOTR_SEARCH_MAX_SECONDS'            # wall seconds per search, 0 = off
SEARCH_MAX_PRIVATE_GB_ENV = 'WOTR_SEARCH_MAX_PRIVATE_GB'      # private-bytes GROWTH per search (GB), 0 = off
SEARCH_MAX_ADVANCES_ENV = 'WOTR_SEARCH_MAX_ADVANCES'          # native: advances per simulation (factor), 0 = off
SEARCH_WATCHDOG_INTERRUPT_ENV = 'WOTR_SEARCH_WATCHDOG_INTERRUPT'  # 1 (default) = async-raise into the search thread
SEARCH_BUDGET_DEFAULTS = {'max_seconds': 300.0, 'max_private_gb': 4.0, 'max_advances_per_sim': 20.0,
                          'watchdog_interrupt': True}
SEARCH_WATCHDOG_POLL_SECONDS = 1.0
STALL_SECONDS_ENV = 'WOTR_SERVED_STALL_SEC'
STALL_SECONDS_MIN = 1800.0


def _environ(env):
    return os.environ if env is None else env


def max_game_failures(games, env=None):
    """Per-batch game-failure budget: WOTR_SP_MAX_GAME_FAILURES, default = the batch size."""
    raw = _environ(env).get(MAX_GAME_FAILURES_ENV)
    if raw is None or not raw.strip():
        return int(games)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError('%s must be a nonnegative integer; got %r' % (MAX_GAME_FAILURES_ENV, raw)) from None
    if value < 0:
        raise ValueError('%s must be a nonnegative integer' % MAX_GAME_FAILURES_ENV)
    return value


def env_nonnegative(name, default, kind, env=None):
    raw = _environ(env).get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    return value


def search_budget_settings(env=None):
    """The per-search budget in force (env, defaults above; 0 = that check disabled).  The same env names drive the
    orchestrator's guard (native route) and the runner's watchdog thread."""
    environ = _environ(env)
    raw = (environ.get(SEARCH_WATCHDOG_INTERRUPT_ENV) or '1').strip()
    if raw not in ('0', '1'):
        raise ValueError('%s must be 0 or 1; got %r' % (SEARCH_WATCHDOG_INTERRUPT_ENV, raw))
    return {'max_seconds': env_nonnegative(SEARCH_MAX_SECONDS_ENV, SEARCH_BUDGET_DEFAULTS['max_seconds'], float, environ),
            'max_private_gb': env_nonnegative(SEARCH_MAX_PRIVATE_GB_ENV, SEARCH_BUDGET_DEFAULTS['max_private_gb'], float,
                                              environ),
            'max_advances_per_sim': env_nonnegative(SEARCH_MAX_ADVANCES_ENV, SEARCH_BUDGET_DEFAULTS['max_advances_per_sim'],
                                                    float, environ),
            'watchdog_interrupt': raw == '1',
            'watchdog_poll_seconds': SEARCH_WATCHDOG_POLL_SECONDS}


def stall_seconds(env=None):
    """The served progress-stall window: WOTR_SERVED_STALL_SEC, default and minimum 1800 seconds (an empty value refuses)."""
    raw = _environ(env).get(STALL_SECONDS_ENV, '1800')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value < STALL_SECONDS_MIN:
        raise ValueError('%s must be finite and at least 1800 seconds; got %r' % (STALL_SECONDS_ENV, raw))
    return value


def validate_runner_knobs(games, env=None):
    """Every operational knob run_sp_served validates before it reserves its output, checked on `env`.  Raises a
    ValueError naming the knob; returns the resolved values."""
    return {'search_budget': search_budget_settings(env), 'max_game_failures': max_game_failures(games, env),
            'stall_seconds': stall_seconds(env)}
