# -*- coding: utf-8 -*-
"""Interactive training pipeline and TD self-play dashboard.

Loads the active pipeline's committed per-game JSONL feed, lists the
game batches, the games inside each batch, per-game statistics on
click, and a button that launches the game's trace in the replay GUI.
Refresh re-reads the feed, so it can stay open during a run.

Usage: python td_dashboard.py [games_jsonl]
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
import tkinter as tk
from collections import Counter, OrderedDict
from tkinter import messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_resolve  # noqa: E402  (stale-path resolver)


def _active_pipeline():
    """Read the optional workspace-local default without holding its file open."""
    pointer = os.path.join(ROOT, 'data', 'active_training_pipeline.json')
    try:
        with open(pointer, 'rb') as source:
            content = source.read()
        value = json.loads(content.decode('utf-8-sig'))
        result = {}
        workspace = os.path.normcase(os.path.realpath(ROOT))
        for key in ('pipeline_status_path', 'selfplay_feed'):
            path = value[key]
            if not isinstance(path, str) or not os.path.isabs(path):
                raise ValueError('%s must be absolute' % key)
            path = os.path.realpath(path)
            if os.path.commonpath([workspace, os.path.normcase(path)]) != workspace:
                raise ValueError('%s must stay inside the workspace' % key)
            result[key] = path
        if os.path.basename(result['selfplay_feed']).lower() != 'all_games.jsonl':
            raise ValueError('selfplay_feed must name all_games.jsonl')
        return result
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.warn('Ignoring invalid active training pointer: %s' % exc,
                      RuntimeWarning)
        return {}


def _newest_games_log():
    """Environment, then active pipeline, then legacy feeds (even before game 1)."""
    configured = os.environ.get('WOTR_LOOP_DIR')
    if configured:
        return os.path.join(os.path.abspath(configured), 'all_games.jsonl')
    active = _active_pipeline()
    if active:
        return active['selfplay_feed']
    # v33 closed loop: the CUMULATIVE feed (baseline + every archived
    # batch, valid trace paths) outranks the runner's transient
    # per-batch file
    loop_feed = os.path.join(ROOT, 'data', 'v33loop',
                             'all_games.jsonl')
    if os.path.exists(loop_feed):
        return loop_feed
    import glob as _glob
    hits = _glob.glob(os.path.join(ROOT, 'data', 'td_*_games.jsonl'))
    if hits:
        return max(hits, key=os.path.getmtime)
    return os.path.join(ROOT, 'data', 'td_v33_games.jsonl')


DEFAULT_LOG = _newest_games_log()
GUI_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'gui.py')
PY = sys.executable
CARD_INDEX_PATH = os.path.join(ROOT, 'data', 'card_usage_index.json')
CARD_INDEXER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'card_usage_index.py')
FINDER_DECKS = ('FP Character', 'FP Strategy',
                'SH Character', 'SH Strategy')

# ---- loop-control wiring ----
LEGACY_LOOP_DIR = os.path.join(ROOT, 'data', 'v33loop')
LOOP_DIR = LEGACY_LOOP_DIR
LEGACY_CONTROLS_ENABLED = True
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')
RESTART_BAT = os.path.join(SCRIPTS_DIR, 'restart_loop.bat')
SCRIPT_LOG = os.path.join(LOOP_DIR, 'restart_script_last.log')
CREATE_NO_WINDOW = 0x08000000
STATUS_REFRESH_MS = 20000        # status panel auto-refresh


def _sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def _card_short(cid):
    for pre in ('FREE_PEOPLES_Character_', 'FREE_PEOPLES_Strategy_',
                'SHADOW_Character_', 'SHADOW_Strategy_'):
        if cid.startswith(pre):
            return cid[len(pre):].replace('_', ' ')
    return cid.replace('_', ' ')


def load_games(path):
    games = []
    committed = False
    contract_path = os.path.join(os.path.dirname(path), 'dashboard_feed_contract.json')
    if os.path.basename(path) == 'all_games.jsonl' and os.path.exists(contract_path):
        with open(contract_path, encoding='utf-8') as source:
            contract = json.load(source)
        if contract != {'version': 'committed_served_games_v1', 'publication': 'committed_only'}:
            raise ValueError('Unsupported dashboard feed contract')
        committed = True
    if not os.path.exists(path):
        return games
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                games.append(json.loads(line))
            except Exception:
                continue
    # LIVE MERGE (USER 2026-07-30 'gui is not updated'): the loop's
    # feed only gains rows when a batch ARCHIVES (~3h) — merge the
    # runner's transient per-batch file so in-flight games show up
    # immediately, labeled with the batch they will join.  On archive
    # the transient file moves into the batch dir and the feed gains
    # the same rows (batch-stamped), so nothing is ever double-listed.
    legacy_feed = (os.path.normcase(os.path.dirname(os.path.abspath(path))) ==
                   os.path.normcase(os.path.abspath(LEGACY_LOOP_DIR)))
    if os.path.basename(path) == 'all_games.jsonl' and not committed and legacy_feed:
        live = os.path.join(ROOT, 'data', 'td_v33_games.jsonl')
        if os.path.exists(live):
            seen = {str(g.get('trace_path', '')) for g in games}
            try:
                with open(live, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if str(r.get('trace_path', '')) in seen:
                            continue
                        _b = str(r.get('batch', ''))
                        if not (_b.startswith('batch_')
                                or _b == 'baseline'):
                            r['batch'] = 'in flight'
                        games.append(r)
            except OSError:
                pass
    return games


# ---- status collection (module-level: usable without a Tk root) ----
def _iter_jsonl(path):
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _pid_alive(pid):
    """tasklist-based liveness; requires the pid to be a python
    process (pid-reuse guard)."""
    try:
        out = subprocess.run(
            ['tasklist', '/FI', 'PID eq %d' % pid, '/FO', 'CSV',
             '/NH'], capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW).stdout or ''
    except Exception:
        return False
    return ('"%d"' % pid) in out and 'python' in out.lower()


def provisional_served_view(committed=()):
    """Separate replay previews from the journal's current, uncommitted phase."""
    empty = {'games': [], 'summary': '', 'errors': 0}
    if LEGACY_CONTROLS_ENABLED:
        return empty
    def read(path):
        with _open_status_file(path) as stream:
            return json.load(stream)
    # Direct-policy runs publish completed replay previews independently of
    # their paired PPO checkpoint. Do not infer an old served-search journal.
    actor_progress = os.path.join(LOOP_DIR, 'actor_critic_progress.json')
    if os.path.exists(actor_progress):
        try:
            value = read(actor_progress)
            if value.get('schema') != 'actor_critic_progress_v1':
                raise ValueError('Unsupported actor-critic progress schema')
            admitted = {row.get('episode') for row in committed}
            games = [row for row in value['games'] if row.get('episode') not in admitted]
            return dict(empty, games=games, summary=str(value.get('summary', '')),
                        errors=int(value.get('errors', 0)), completed=len(games))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(empty, summary='Actor-critic progress unreadable: %s' % exc)
    try:
        marker = read(os.path.join(LOOP_DIR, '.batch_journal', 'pre_ready.json'))
    except FileNotFoundError:
        return empty
    except (OSError, ValueError) as exc:
        return dict(empty, summary='Provisional phase unreadable: %s' % exc)
    try:
        index = marker['index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid active batch index')
        expected = 'incoming_served_b%03d' % index
        if marker.get('output') != expected:
            return empty  # prepared, but the served phase has not launched
        directory = os.path.realpath(os.path.join(LOOP_DIR, expected))
        loop = os.path.normcase(os.path.realpath(LOOP_DIR))
        if os.path.commonpath([loop, os.path.normcase(directory)]) != loop:
            raise ValueError('Active served path escapes loop')
        phase = read(os.path.join(directory, 'phase.json'))
        identity = phase['phase_id']
        try:
            with _open_status_file(os.path.join(directory, 'attempts.jsonl')) as stream:
                content = stream.read()
        except FileNotFoundError:
            # Initialization can fail before the first episode creates this
            # file. The phase error still needs to be visible.
            content = ''
        attempts = {}
        partial = False
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                partial = True
                continue
            if row.get('phase_id') != identity:
                continue
            key = (row.get('worker_id'), row.get('episode'))
            if key in attempts:
                raise ValueError('Duplicate active episode outcome')
            attempts[key] = row
        complete = [row for row in attempts.values() if row.get('status') == 'completed']
        errors = sum(row.get('status') == 'error' for row in attempts.values())
        runtime_failures = []
        seen_failures = set()
        for failure in [phase.get('parent_error')] + list(phase.get('worker_failures') or []):
            if not isinstance(failure, dict) or not failure or failure.get('stage') == 'reporting_gap':
                continue
            key = (failure.get('stage'), failure.get('worker_id'), failure.get('error'), failure.get('error_type'))
            if key in seen_failures:
                continue
            seen_failures.add(key)
            detail = failure.get('error') or failure.get('error_type') or failure.get('stage') or 'worker failed'
            runtime_failures.append(' '.join(str(detail).split())[:240])
        planned = int(phase.get('planned', marker.get('planned_games', 0)))
        unreported = max(0, planned - len(attempts))
        admitted = {(row.get('phase_id'), row.get('worker_id'), row.get('episode')) for row in committed}
        games = []
        for row in complete:
            if (identity, row.get('worker_id'), row.get('episode')) in admitted:
                continue
            path = os.path.realpath(str(row.get('trace_path') or ''))
            traces = os.path.normcase(os.path.join(directory, 'traces'))
            if os.path.commonpath([traces, os.path.normcase(path)]) != traces:
                raise ValueError('Provisional trace escapes its served phase')
            game = dict(row, _provisional=True,
                        batch='PROVISIONAL — completed games, batch not yet trained (batch_%03d)' % index)
            games.append(game)
        summary = ('Active batch_%03d: %d completed, %d errors, %d unreported / %d planned'
                   % (index, len(complete), errors, unreported, planned))
        if runtime_failures:
            summary += ' — run failed: ' + runtime_failures[0]
            if len(runtime_failures) > 1:
                summary += ' (%d runtime failures)' % len(runtime_failures)
        elif phase.get('finished') and unreported and not errors:
            summary += ' — phase ended incomplete; unreported games remain'
        elif phase.get('finished'):
            summary += (' — phase ended with errors' if errors else
                        ' — phase finished; awaiting validation')
        elif not errors:
            summary += ' — %d more games before batch validation and training' % unreported
        if partial:
            summary += ' — incomplete report line; awaiting publication'
        return {'games': games, 'summary': summary, 'errors': errors,
                'runtime_errors': len(runtime_failures), 'runtime_details': runtime_failures,
                'completed': len(complete), 'unreported': unreported,
                'phase_id': identity}
    except FileNotFoundError:
        return empty
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(empty, summary='Provisional phase unreadable: %s' % exc)


def _read_tail(path, n=10):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return [ln.rstrip('\n') for ln in f.readlines()[-n:]]
    except OSError:
        return []


def _batch_rows():
    """Per-batch table: dir, games, fp/sh wins, Stage-A census
    verdict, quarantine flag.  Reads the batch dirs THEMSELVES
    (incl. quarantine renames), so it is immune to stale feed rows."""
    rows = []
    dirs = []
    bg = os.path.join(LOOP_DIR, 'baseline_games')
    if os.path.isdir(bg):
        dirs.append(bg)
    dirs += sorted(d for d in glob.glob(
        os.path.join(LOOP_DIR, 'batch_*')) if os.path.isdir(d))
    dirs += sorted(d for d in glob.glob(
        os.path.join(LOOP_DIR, 'stale_quarantine_*'))
        if os.path.isdir(d))
    for d in dirs:
        gj = os.path.join(d, 'games.jsonl')
        if not os.path.exists(gj):
            gj = os.path.join(d, 'td_v33_games.jsonl')
        n = fpw = shw = 0
        for r in _iter_jsonl(gj):
            n += 1
            w = r.get('winner')
            fpw += 1 if w == 'fp' else 0
            shw += 1 if w == 'sh' else 0
        census = '-'
        cj = os.path.join(d, 'stage_a_census.json')
        if os.path.exists(cj):
            try:
                with open(cj, encoding='utf-8') as f:
                    c = json.load(f)
                if c.get('ok'):
                    census = 'PASS'
                else:
                    census = 'ABORT: ' + (', '.join(
                        '%s=%s' % (k, v)
                        for k, v in (c.get('fails') or []))
                        or '(no fails listed)')
            except Exception as e:
                census = 'census unreadable: %r' % (e,)
        base = os.path.basename(d)
        flag = ('QUARANTINED' if '_quarantined' in base
                else 'stale-swept' if base.startswith(
                    'stale_quarantine') else '')
        rows.append((base, n, fpw, shw, census, flag))
    live = (os.path.join(ROOT, 'data', 'td_v33_games.jsonl') if LEGACY_CONTROLS_ENABLED
            else os.path.join(LOOP_DIR, 'incoming_games.jsonl'))
    if os.path.exists(live):
        n = fpw = shw = 0
        for r in _iter_jsonl(live):
            n += 1
            w = r.get('winner')
            fpw += 1 if w == 'fp' else 0
            shw += 1 if w == 'sh' else 0
        if n:
            label = ('(in flight: td_v33_games.jsonl)' if LEGACY_CONTROLS_ENABLED else
                     '(pending validation: %s)' % os.path.basename(live))
            rows.append((label, n, fpw,
                         shw, '-', ''))
    return rows


def _net_rows(ck_cache):
    """Checkpoint identity: sha12 + head_act + conf-stamp presence
    for baseline/live/last_good pairs.  torch.load is cached on
    (path, mtime, size) — refreshes stay cheap while the loop is
    idle."""
    rows = []
    pts = []
    for tag in ('baseline', 'live', 'last_good'):
        pts += sorted(glob.glob(os.path.join(LOOP_DIR,
                                             tag + '_*.pt')))
    for p in pts:
        try:
            stt = os.stat(p)
        except OSError:
            continue
        key = (p, stt.st_mtime, stt.st_size)
        info = ck_cache.get(key)
        if info is None:
            info = {'sha12': _sha12(p)}
            try:
                import torch
                try:
                    ck = torch.load(p, map_location='cpu',
                                    weights_only=False)
                except TypeError:      # older torch: no kwarg
                    ck = torch.load(p, map_location='cpu')
                info['head_act'] = str(ck.get('head_act',
                                              '(absent->tanh)'))
                info['conf'] = ('stamped' if any(
                    k in ck for k in ('conf_mech', 'conf_state',
                                      'conf_calib', 'conf_meta'))
                    else 'NO STAMP')
                del ck
            except Exception as e:
                info['head_act'] = 'load-error'
                info['conf'] = repr(e)[:60]
            if len(ck_cache) > 64:
                ck_cache.clear()
            ck_cache[key] = info
        rows.append((os.path.basename(p), info['sha12'],
                     info.get('head_act', ''), info.get('conf', ''),
                     time.strftime('%m-%d %H:%M',
                                   time.localtime(stt.st_mtime))))
    return rows


def _loop_state_summary():
    p = os.path.join(LOOP_DIR, 'loop_state.json')
    if not os.path.exists(p):
        if not LEGACY_CONTROLS_ENABLED:
            return '  (self-play has no committed training batch yet)'
        return '  (no loop_state.json — loop not yet past batch 0, '\
               'or artifacts cleared)'
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        return '  loop_state.json unreadable: %r' % (e,)
    lines = []
    for net in sorted(d):
        s = d[net] or {}
        hist = s.get('history') or []
        last = ' | '.join(str(x) for x in hist[-1]) if hist else '-'
        lines.append('  %s: lr x%s | A-fails %s | B-fails %s | '
                     'last: %s' % (net, s.get('lr_mult', '?'),
                                   s.get('a_fails', '?'),
                                   s.get('b_fails', '?'), last))
    return '\n'.join(lines) or '  (empty)'


def _bat_config():
    """The canonical intended env = the ENV block of
    scripts\\restart_loop.bat (reading a live process's env is not
    feasible on Windows without native calls)."""
    if not LEGACY_CONTROLS_ENABLED:
        path = os.path.join(LOOP_DIR, 'launch_environment.json')
        try:
            with open(path, encoding='utf-8') as stream:
                launch = json.load(stream)
            return ['Run: ' + LOOP_DIR] + ['%s=%s' % item for item in
                sorted((launch.get('environment') or {}).items())]
        except (OSError, ValueError, TypeError):
            return ['Run: ' + LOOP_DIR,
                    'Managed by its dedicated controller; launch environment not yet recorded.']
    if not os.path.exists(RESTART_BAT):
        return ['(scripts\\restart_loop.bat MISSING — control '
                'buttons will refuse)']
    out, inside = [], False
    try:
        with open(RESTART_BAT, encoding='utf-8',
                  errors='replace') as f:
            for ln in f:
                ls = ln.strip()
                if ls.startswith(':: ENV-BEGIN'):
                    inside = True
                    continue
                if ls.startswith(':: ENV-END'):
                    break
                if inside and ls.lower().startswith('set "'):
                    kv = ls[5:].rstrip('"')
                    if kv.endswith('='):
                        out.append(kv + '(cleared at launch)')
                    else:
                        out.append(kv)
    except OSError as e:
        return ['(restart_loop.bat unreadable: %r)' % (e,)]
    return out or ['(no ENV block found in restart_loop.bat)']


# ---- loop settings (USER 2026-08-11: arena checkbox + tunables) ----
# The panel writes LOOP\loop_config.json; run_td_v33loop reads it via
# _apply_loop_config() at loop START (settings apply on restart).
# Keys MUST match run_td_v33loop._CFG_KEYS exactly — unknown keys make
# the loop refuse loudly, so this table is the single source of truth
# on the GUI side and the round-trip test asserts it stays a subset.
LOOP_CONFIG_PATH = os.path.join(LOOP_DIR, 'loop_config.json')


def bind_loop_paths(log_path):
    """Bind status/settings to the selected committed feed, before it exists.

    The legacy batch launcher cannot control an isolated run: it clears the
    loop override and selects an old generation. Custom runs remain visible
    and configurable here, but their owning controller manages processes.
    """
    global LOOP_DIR, LOOP_CONFIG_PATH, SCRIPT_LOG, CARD_INDEX_PATH
    global LEGACY_CONTROLS_ENABLED
    path = os.path.abspath(os.fspath(log_path))
    configured = os.environ.get('WOTR_LOOP_DIR')
    if configured:
        selected = os.path.abspath(configured)
        expected = os.path.join(selected, 'all_games.jsonl')
        if os.path.normcase(path) != os.path.normcase(expected):
            raise ValueError('Dashboard feed differs from WOTR_LOOP_DIR')
    elif os.path.basename(path).lower() == 'all_games.jsonl':
        selected = os.path.dirname(path)
    else:
        selected = LEGACY_LOOP_DIR
    LOOP_DIR = selected
    LOOP_CONFIG_PATH = os.path.join(selected, 'loop_config.json')
    SCRIPT_LOG = os.path.join(selected, 'restart_script_last.log')
    LEGACY_CONTROLS_ENABLED = (os.path.normcase(os.path.abspath(selected)) ==
                               os.path.normcase(os.path.abspath(LEGACY_LOOP_DIR)))
    CARD_INDEX_PATH = os.path.join(ROOT, 'data', 'card_usage_index.json') if LEGACY_CONTROLS_ENABLED else os.path.join(selected, 'card_usage_index.json')
    return path
# (key, default, kind, min, one-line description)
LOOP_SETTINGS = (
    ('arena_enabled', True, 'bool', None,
     'arena promotion gate on/off'),
    ('arena_every_batches', 4, 'int', 1,
     'run the arena every N batches'),
    ('arena_workers', 12, 'int', 1, 'parallel arena workers'),
    ('arena_games_per_worker', 8, 'int', 1,
     'arena games = workers x per-worker (default 96)'),
    ('arena_sims', 60, 'int', 10, 'MCTS sims per move in arena games'),
    ('batch_games', 50, 'int', 10, 'games per training batch'),
    ('lr', 1e-4, 'float', None, 'value+delta head learning rate (>0)'),
    ('epochs', 2, 'int', 1, 'training epochs per batch'),
    ('lam', 0.8, 'float', None, 'TD(lambda) credit horizon (0..1]'),
    ('delta_wd', 1e-4, 'float', None,
     'Adam weight decay on delta params (>=0)'),
    ('buffer_k', 6, 'int', 1, 'batches of replay kept'),
    ('workers', 12, 'int', 1, 'self-play workers'),
    ('sims', None, 'intblank', 50,
     'MCTS simulations per move (blank = default 1000)'),
    ('total_games', 2000, 'int', 1,
     'loop target: stop after this many games'),
)
LOOP_DEFAULTS = {k: d for k, d, _kind, _mn, _ds in LOOP_SETTINGS}

# Loop-only config keys (run_td_v33loop._CFG_KEYS) with NO dashboard
# widget: the frozen-opponent anchor trio (amendment 1.259).  Save
# PRESERVES them from the existing file and the unknown-key warning
# knows them - a GUI save must never strip a hand-configured anchor.
LOOP_PASSTHROUGH_KEYS = ('frozen_opponent_p', 'frozen_ckpt_fp',
                         'frozen_ckpt_sh')


def build_loop_config(values):
    """values: {key: raw} — bool for arena_enabled, strings for the
    rest (as typed).  Returns (cfg, errors).  cfg ALWAYS carries
    arena_enabled (the checkbox is an explicit choice, never left to
    a default that might flip); tunables are included only when they
    differ from the defaults, so an untouched field keeps tracking
    the loop's own default.  errors non-empty => cfg is None."""
    cfg, errors = {}, []
    for key, dflt, kind, mn, _desc in LOOP_SETTINGS:
        raw = values.get(key)
        if kind == 'bool':
            cfg[key] = bool(raw)
            continue
        txt = str(raw if raw is not None else '').strip()
        if kind == 'intblank' and txt == '':
            val = None
        elif kind == 'float':
            try:
                val = float(txt)
            except ValueError:
                errors.append('%s: not a number (%r)' % (key, txt))
                continue
        else:
            try:
                val = int(txt)
            except ValueError:
                errors.append('%s: not an integer (%r)' % (key, txt))
                continue
        # range checks (client-side: never write a json the loop
        # would choke on)
        if key == 'lr' and not val > 0:
            errors.append('lr: must be > 0')
            continue
        if key == 'lam' and not (0 < val <= 1):
            errors.append('lam: must be in (0, 1]')
            continue
        if key == 'delta_wd' and val < 0:
            errors.append('delta_wd: must be >= 0')
            continue
        if mn is not None and val is not None and val < mn:
            errors.append('%s: must be >= %s' % (key, mn))
            continue
        if val != dflt:
            cfg[key] = val
    if errors:
        return None, errors
    return cfg, []


def save_loop_config(values, path=None):
    """Validate + write loop_config.json atomically.  Returns
    (cfg, errors); cfg is what was written."""
    path = path or LOOP_CONFIG_PATH
    cfg, errors = build_loop_config(values)
    if errors:
        return None, errors
    # 1.259: carry loop-only keys forward from the existing file
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                prev = json.load(f)
            for k in LOOP_PASSTHROUGH_KEYS:
                if k in prev:
                    cfg[k] = prev[k]
        except Exception as e:
            return None, ['existing loop_config.json unreadable (%r) - '
                          'fix or delete it before saving' % (e,)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return cfg, []


def load_loop_config_effective(path=None):
    """Returns (effective dict, override-keys set, warning or None):
    file values where present, defaults elsewhere.  Unknown keys in a
    hand-edited file are surfaced as a warning (the loop will refuse
    them)."""
    path = path or LOOP_CONFIG_PATH
    eff = dict(LOOP_DEFAULTS)
    if not os.path.exists(path):
        return eff, set(), None
    try:
        with open(path, encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        return eff, set(), 'loop_config.json unreadable: %r' % (e,)
    unknown = [k for k in cfg if k not in LOOP_DEFAULTS
               and k not in LOOP_PASSTHROUGH_KEYS]
    over = set()
    for k, v in cfg.items():
        if k in LOOP_DEFAULTS:
            eff[k] = v
            over.add(k)
    warn = ('loop_config.json has UNKNOWN key(s) %s - the loop will '
            'REFUSE to start until they are removed (use Save or '
            'Reset here)' % unknown) if unknown else None
    return eff, over, warn


@contextmanager
def _open_status_file(path):
    """Release the OS handle before consumers decode/parse the snapshot.

    Writers still need bounded sharing-violation retries during the brief
    byte read. Windows replacement can fail even with FILE_SHARE_DELETE.
    """
    with open(path, 'rb') as source:
        content = source.read()
    with StringIO(content.decode('utf-8-sig')) as stream:
        yield stream


def _status_timestamp(value):
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _process_status_issue(pid, stamp=None, created=None):
    """Observe PID identity without signalling any process."""
    if type(pid) is not int or pid <= 0:
        return 'process identity missing'
    import psutil
    try:
        process = psutil.Process(pid)
        born = process.create_time()
        if (created is not None and abs(born - float(created)) > 0.1) or (stamp is not None and born > stamp + 1):
            return 'process PID was reused'
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return 'process is no longer running'
        return None
    except psutil.NoSuchProcess:
        return 'process is no longer running'
    except (psutil.AccessDenied, ValueError, TypeError):
        return 'process identity could not be verified'


def _pipeline_view():
    """Read the dedicated controller's stage and linked live progress."""
    if LEGACY_CONTROLS_ENABLED:
        return None
    root = os.path.dirname(LOOP_DIR)
    def read(path):
        try:
            with _open_status_file(path) as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError) as exc:
            return {'_read_error': str(exc)} if path else {}
    active = _active_pipeline()
    selected_feed = os.path.join(LOOP_DIR, 'all_games.jsonl')
    matches_active = (active and os.path.normcase(active['selfplay_feed']) ==
                      os.path.normcase(os.path.realpath(selected_feed)))
    pipeline_path = (active['pipeline_status_path'] if matches_active else
                     os.path.join(root, 'pipeline_status.json'))
    pipeline = read(pipeline_path)
    supervisor_path = (pipeline.get('supervisor_status_path') or
                       os.path.join(root, 'supervisor', 'status.json'))
    supervisor = read(supervisor_path)
    active = supervisor if supervisor.get('status') in (
        'running', 'repairing', 'validating_resume', 'completed', 'failed', 'stopped') else pipeline
    if supervisor.get('_read_error') or active.get('_read_error'):
        return {'summary': 'PIPELINE STATUS UNREADABLE', 'stage': None,
                'detail': supervisor.get('_read_error') or active['_read_error'], 'status': 'unavailable'}
    if not active:
        return {'summary': 'WAITING FOR TRAINING PIPELINE',
                'detail': 'Self-play has not started.', 'stage': None}
    stage = active.get('stage', 'waiting')
    label = {'extraction': 'EXTRACTING CORPUS', 'consolidate': 'VALIDATING DATASET',
             'bc_training': 'TRAINING MODELS', 'selfplay': 'SELF-PLAY',
             'bc': 'TRAINING MODELS'}.get(stage, str(stage).upper())
    status = active.get('status') or active.get('state') or 'waiting'
    issue = None
    stamp = _status_timestamp(active.get('updated_at'))
    if active is supervisor and status in ('running', 'repairing', 'validating_resume'):
        if stamp is None or time.time() - stamp > 90:
            issue = 'supervisor heartbeat is missing or older than 90 seconds'
        owner = read(os.path.join(os.path.dirname(supervisor_path), 'supervisor_result.json'))
        if not issue and owner.get('owner_pid') is not None:
            failure = _process_status_issue(owner['owner_pid'], stamp)
            if failure:
                issue = 'supervisor ' + failure
        if not issue and active.get('pid') is not None:
            issue = _process_status_issue(active['pid'], stamp)
    elif status == 'running' and stage == 'extraction' and pipeline.get('extraction_launcher'):
        owner = pipeline['extraction_launcher']
        issue = _process_status_issue(owner.get('pid'), created=owner.get('create_time'))
    if issue:
        status = 'status stale / process unavailable'
    summary = '%s — %s' % (label, status)
    detail = active.get('error') or active.get('detail') or active.get('message') or ''
    if issue:
        detail = issue + ('. Last report: ' + str(detail) if detail else '')
    if stage == 'extraction':
        progress = read(pipeline.get('extraction_progress_path') or '')
        completed, total = progress.get('finished'), progress.get('total')
        if isinstance(completed, int) and isinstance(total, int):
            summary += ' | %d / %d games' % (completed, total)
            if isinstance(progress.get('rows'), int):
                summary += ' | %s extracted rows' % format(progress['rows'], ',')
        if not detail:
            detail = 'Model training and 500-game self-play follow after validation.'
    if stage == 'selfplay':
        from response_training_status import observe as response_training_observe
        response_training = response_training_observe(LOOP_DIR)
        if response_training and status == 'running' and not issue:
            return response_training
        provisional = provisional_served_view()
        if provisional['summary']:
            detail = provisional['summary'] + (' | ' + detail if detail else '')
        if provisional['errors']:
            summary += ' | %d failed game attempts' % provisional['errors']
        if provisional.get('runtime_errors'):
            summary += ' | %d runtime failures' % provisional['runtime_errors']
    return {'summary': summary, 'detail': str(detail), 'stage': stage,
            'status': status, 'updated_at': active.get('updated_at')}


def _actor_critic_run():
    path = os.path.join(os.path.dirname(LOOP_DIR), 'config.json')
    try:
        with _open_status_file(path) as stream:
            value = json.load(stream)
        return value if value.get('schema') in ('actor_critic_run_v1', 'search_run_v1') else None
    except (OSError, ValueError, TypeError):
        return None


def _actor_critic_config():
    run = _actor_critic_run()
    return None if run is None else run.get('config')


def _actor_critic_status(config, schema='actor_critic_run_v1'):
    root = os.path.dirname(LOOP_DIR)
    learning = 'Search' if schema == 'search_run_v1' else 'PPO'
    module = 'search_runner' if schema == 'search_run_v1' else 'runner'
    pipeline = _pipeline_view()
    games = load_games(os.path.join(LOOP_DIR, 'all_games.jsonl'))
    grouped = {}
    for game in games:
        grouped.setdefault(game['batch'], []).append(game)
    batches = [(key, len(rows), sum(r.get('winner') == 'fp' for r in rows),
                sum(r.get('winner') == 'sh' for r in rows), learning + ' committed', '') for key, rows in grouped.items()]
    nets, completed = [], 0
    try:
        with _open_status_file(os.path.join(root, 'latest.json')) as stream:
            checkpoint = os.path.realpath(os.path.join(root, json.load(stream)['path']))
        if os.path.commonpath([os.path.realpath(root), checkpoint]) != os.path.realpath(root):
            raise ValueError('Checkpoint path escapes run')
        with _open_status_file(os.path.join(checkpoint, 'bundle.json')) as stream:
            completed = json.load(stream)['metadata']['next_batch']
        for side in ('fp', 'sh'):
            receipt_path = os.path.join(checkpoint, side, 'complete.json')
            with _open_status_file(receipt_path) as stream:
                digest = json.load(stream)['sha256']
            nets.append((side.upper(), digest[:12], 'tanh', learning + ' resume',
                         time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(receipt_path)))))
    except FileNotFoundError:
        pass  # initializing, before the first paired checkpoint
    summary = pipeline or {'summary': 'ACTOR-CRITIC', 'detail': '', 'status': 'waiting'}
    return dict(state=summary['summary'], alive=summary['status'] == 'running', pid=None,
        pipeline=pipeline, tail=[summary['detail']], disk_free_gb=shutil.disk_usage(root).free / 2.0 ** 30,
        batches=batches, nets=nets, loop_state='%d paired %s batches committed; %d training games' % (completed, learning, len(games)),
        config=['python -m selfplay_actor_critic.%s --resume --output "%s"' % (module, root)],
        settings=['%s = %s' % (k, json.dumps(v)) for k, v in config.items()])


def _collect_status(ck_cache):
    actor_run = _actor_critic_run()
    if actor_run is not None:
        return _actor_critic_status(actor_run['config'], actor_run['schema'])
    pid = None
    try:
        with open(os.path.join(LOOP_DIR, 'loop.pid')) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pid = None
    alive = _pid_alive(pid) if pid else False
    tail = _read_tail(os.path.join(LOOP_DIR, 'loop_log.md'), 10)
    recent = ' '.join(tail[-3:]).lower()
    if alive:
        state = 'RUNNING (pid %d)' % pid
    elif 'halt' in recent:
        state = 'HALTED (process gone; loop_log tail shows HALT)'
    else:
        state = 'DOWN (no live loop process)' + (
            ' [stale loop.pid %s]' % pid if pid else '')
    try:
        du = shutil.disk_usage(LOOP_DIR if os.path.isdir(LOOP_DIR)
                               else ROOT)
        free_gb = du.free / 2.0 ** 30
    except OSError:
        free_gb = float('nan')
    eff, over, warn = load_loop_config_effective()
    settings = []
    if warn:
        settings.append('!! %s' % warn)
    for key, _d, _k, _m, _ds in LOOP_SETTINGS:
        settings.append('%s = %s%s' % (
            key, json.dumps(eff.get(key)),
            '   * from loop_config.json' if key in over else ''))
    return {'state': state, 'alive': alive, 'pid': pid, 'pipeline': _pipeline_view(),
            'tail': tail, 'disk_free_gb': free_gb,
            'batches': _batch_rows(), 'nets': _net_rows(ck_cache),
            'loop_state': _loop_state_summary(),
            'config': _bat_config(), 'settings': settings}


class Dashboard:
    def __init__(self, log_path):
        self.log_path = bind_loop_paths(log_path)
        self.actor_critic = _actor_critic_config() is not None
        self.root = tk.Tk()
        self.root.title(('Actor-critic training - ' if self.actor_critic else 'TD self-play dashboard - ') + log_path)
        self.root.geometry('1240x720')
        self.games_by_batch = OrderedDict()
        self.current_games = []

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(top, text='Refresh', command=self.refresh)\
            .pack(side=tk.LEFT)
        self.status = tk.Label(top, text='', anchor='w')
        self.status.pack(side=tk.LEFT, padx=10)
        self.pipeline_lbl = tk.Label(self.root, text='', anchor='w', justify=tk.LEFT,
                                     wraplength=1200, fg='#004080')
        self.pipeline_lbl.pack(fill=tk.X, padx=6)

        # --- two tabs: Games (the original
        # dashboard) + Loop status (status-of-everything panel with
        # one-click stop / restart / clear+restart) ---
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        games_tab = tk.Frame(nb)
        status_tab = tk.Frame(nb)
        settings_tab = tk.Frame(nb)
        nb.add(games_tab, text=' Games ')
        nb.add(status_tab, text=' Loop status ')
        nb.add(settings_tab, text=' Loop settings ')
        if not LEGACY_CONTROLS_ENABLED:
            nb.select(status_tab)
        self._nb = nb

        # --- card-usage finder (USER spec 2026-07-15): 4 deck
        # dropdowns + combat checkbox -> game picker -> GUI at the
        # exact step where the card was played ---
        finder = tk.Frame(games_tab)
        finder.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(finder, text='FIND:').pack(side=tk.LEFT)
        self._finder_vars = {}
        self._finder_boxes = {}
        self._finder_maps = {}
        for deck in FINDER_DECKS:
            v = tk.StringVar()
            cb = ttk.Combobox(finder, textvariable=v, width=26,
                              state='readonly')
            cb.pack(side=tk.LEFT, padx=2)
            cb.set(deck)
            cb.bind('<<ComboboxSelected>>',
                    lambda e, d=deck: self._finder_pick(d))
            self._finder_vars[deck] = v
            self._finder_boxes[deck] = cb
        self._finder_combat = tk.BooleanVar(value=False)
        tk.Checkbutton(finder, text='combat card?',
                       variable=self._finder_combat)\
            .pack(side=tk.LEFT, padx=4)
        tk.Button(finder, text='Find', command=self._finder_find)\
            .pack(side=tk.LEFT, padx=2)
        tk.Button(finder, text='Rebuild index',
                  command=self._finder_rebuild)\
            .pack(side=tk.LEFT, padx=2)
        self._finder_deck = None
        self._card_index = None
        self._finder_load_index()

        body = tk.PanedWindow(games_tab, orient=tk.HORIZONTAL,
                              sashwidth=5)
        body.pack(fill=tk.BOTH, expand=True)

        # left: batches, split by profile (USER spec)
        left = tk.Frame(body, width=240)
        tk.Label(left, text='Training batches' if self.actor_critic else 'Batches / profiles').pack(anchor='w')
        self.batch_tree = ttk.Treeview(left, show='tree',
                                       selectmode='browse')
        lsb = ttk.Scrollbar(left, orient='vertical',
                            command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=lsb.set)
        self.batch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb.pack(side=tk.LEFT, fill=tk.Y)
        self.batch_tree.bind('<<TreeviewSelect>>', self.on_batch)
        body.add(left)

        # middle: games table
        mid = tk.Frame(body)
        self.summary = tk.Label(mid, text='', anchor='w',
                                justify=tk.LEFT)
        self.summary.pack(fill=tk.X)
        cols = ('ep', 'fp_prof', 'sh_prof', 'winner', 'vtype', 'turns',
                'sh_vp', 'fp_vp', 'corr', 'crowned')
        self.tree = ttk.Treeview(mid, columns=cols, show='headings',
                                 selectmode='browse')
        widths = (50, 76, 90, 62, 110, 52, 52, 52, 48, 62)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='center')
        if self.actor_critic:
            self.tree.configure(displaycolumns=tuple(c for c in cols if c not in ('fp_prof', 'sh_prof')))
        vsb = ttk.Scrollbar(mid, orient='vertical',
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind('<<TreeviewSelect>>', self.on_game)
        body.add(mid)

        # right: detail + launch
        right = tk.Frame(body, width=330)
        tk.Label(right, text='Game detail').pack(anchor='w')
        self.detail = tk.Text(right, width=44, wrap='word',
                              state=tk.DISABLED)
        self.detail.pack(fill=tk.BOTH, expand=True)
        self.launch_btn = tk.Button(right, text='Open game trace',
                                    state=tk.DISABLED,
                                    command=self.launch_trace)
        self.launch_btn.pack(fill=tk.X, pady=3)
        self.attr_btn = tk.Button(right, text='Attribution view '
                                  '(credit / blame)',
                                  state=tk.DISABLED,
                                  command=self.launch_attribution)
        self.attr_btn.pack(fill=tk.X, pady=1)
        body.add(right)

        self.selected_game = None
        self._resolved_trace = None
        self._trace_index = None
        self._trace_index_t = 0.0
        self._ck_cache = {}
        self._status_busy = False
        self._script_proc = None
        self._build_status_tab(status_tab)
        self._build_settings_tab(settings_tab)
        self.refresh()
        self.refresh_status()
        self.root.after(STATUS_REFRESH_MS, self._status_tick)

    # ------------------------------------------------------------------
    PROFILE_NODES = (('fp', 'ring', 'FP Ring'),
                     ('fp', 'military', 'FP Military'),
                     ('fp', 'mixed', 'FP Mixed'),
                     ('sh', 'corruption', 'SH Corruption'),
                     ('sh', 'military', 'SH Military'),
                     ('sh', 'mixed', 'SH Mixed'))

    def refresh(self):
        games = load_games(self.log_path)
        provisional = provisional_served_view(games)
        evaluation = []
        incomplete = []
        if self.actor_critic:
            evaluation = [dict(row, batch='EVALUATION - ' + str(row.get('batch', '')))
                          for row in _iter_jsonl(os.path.join(LOOP_DIR, 'evaluation_games.jsonl'))]
            incomplete = [dict(row, batch='INCOMPLETE - ' + str(row.get('batch', '')))
                          for row in _iter_jsonl(os.path.join(LOOP_DIR, 'incomplete_games.jsonl'))]
        selected_path = (self.selected_game or {}).get('trace_path')
        signature = (games, provisional, evaluation, incomplete)
        if getattr(self, '_game_view_signature', None) == signature:
            return
        self._game_view_signature = signature
        self.games_by_batch = OrderedDict()
        for g in games + provisional['games'] + evaluation + incomplete:
            self.games_by_batch.setdefault(
                str(g.get('batch', '?')), []).append(g)
        keep = None
        sel = self.batch_tree.selection()
        if sel:
            keep = sel[0]
        self.batch_tree.delete(*self.batch_tree.get_children())
        for bi, (b, gs) in enumerate(self.games_by_batch.items()):
            bid = 'b%d' % bi
            self.batch_tree.insert(
                '', tk.END, iid=bid, text='%s  (%d)' % (b, len(gs)),
                open=True, values=())
            for side_key, prof, label in (() if self.actor_critic else self.PROFILE_NODES):
                n = sum(1 for g in gs
                        if (g.get('profiles') or {}).get(side_key)
                        == prof)
                self.batch_tree.insert(
                    bid, tk.END, iid='%s|%s|%s' % (bid, side_key, prof),
                    text='%s  (%d)' % (label, n))
        self.status.config(text=('%d games committed to training | %d completed in current batch | %s'
            % (len(games), len(provisional['games']), provisional['summary'] or self.log_path))
            if not LEGACY_CONTROLS_ENABLED else '%d games in %d batches | %s' %
            (len(games), len(self.games_by_batch), self.log_path))
        if self.actor_critic:
            self.status.config(text='%d training games | %d pending | %d evaluation games | %d incomplete' %
                               (len(games), len(provisional['games']), len(evaluation), len(incomplete)))
        if keep and self.batch_tree.exists(keep):
            self.batch_tree.selection_set(keep)
        elif self.games_by_batch:
            self.batch_tree.selection_set('b0')
        self.on_batch()
        if selected_path:
            for index, game in enumerate(self.current_games):
                if game.get('trace_path') == selected_path:
                    self.tree.selection_set(str(index))
                    self.on_game()
                    break

    def _selection(self):
        """Returns (batch_key, side_key or None, profile or None)."""
        sel = self.batch_tree.selection()
        if not sel:
            return None, None, None
        iid = sel[0]
        parts = iid.split('|')
        batches = list(self.games_by_batch.keys())
        try:
            bi = int(parts[0][1:])
            bkey = batches[bi]
        except (ValueError, IndexError):
            return None, None, None
        if len(parts) == 3:
            return bkey, parts[1], parts[2]
        return bkey, None, None

    def on_batch(self, _evt=None):
        key, side_key, prof = self._selection()
        self.tree.delete(*self.tree.get_children())
        games = list(self.games_by_batch.get(key, []))
        if side_key and prof:
            games = [g for g in games
                     if (g.get('profiles') or {}).get(side_key) == prof]
            key = '%s / %s %s' % (key, side_key.upper(), prof)
        self.current_games = games
        for i, g in enumerate(self.current_games):
            pp = g.get('profiles') or {}
            self.tree.insert('', tk.END, iid=str(i), values=(
                g.get('episode', ''), pp.get('fp'), pp.get('sh'),
                g.get('winner'), g.get('victory_type'),
                g.get('turns'), g.get('sh_vp'), g.get('fp_vp'),
                g.get('corruption'),
                'yes' if g.get('aragorn_crowned') else ''))
        gs = self.current_games
        if gs:
            n = len(gs)
            fpw = sum(1 for g in gs if g.get('winner') == 'fp')
            shw = sum(1 for g in gs if g.get('winner') == 'sh')
            self.summary.config(text=(
                '%s: n=%d | fp %d / sh %d / to %d | avg turns %.1f | '
                'avg SH VP %.2f | avg FP VP %.2f | avg corr %.2f | '
                'crowned %.0f%% | avg eyes/turn %.2f' % (
                    key, n, fpw, shw, n - fpw - shw,
                    sum(g.get('turns', 0) for g in gs) / n,
                    sum(g.get('sh_vp', 0) for g in gs) / n,
                    sum(g.get('fp_vp', 0) for g in gs) / n,
                    sum(g.get('corruption', 0) for g in gs) / n,
                    100.0 * sum(1 for g in gs
                                if g.get('aragorn_crowned')) / n,
                    sum((sum(g.get('eye_allocs') or [0])
                         / max(1, len(g.get('eye_allocs') or [0])))
                        for g in gs) / n)))
        else:
            self.summary.config(text='(no games)')
        self.selected_game = None
        self.launch_btn.config(state=tk.DISABLED)
        self._set_detail('')

    def on_game(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        g = self.current_games[int(sel[0])]
        self.selected_game = g
        pp = g.get('profiles') or {}
        lines = [
            'Batch: %s   Episode: %s' % (g.get('batch'),
                                         g.get('episode', '-')),
            'Profiles: FP=%s  SH=%s' % (pp.get('fp'), pp.get('sh')),
            'Winner: %s (%s) in %s turns' % (
                g.get('winner'), g.get('victory_type'),
                g.get('turns')),
            '',
            'Shadow VP: %s    FP VP: %s' % (g.get('sh_vp'),
                                            g.get('fp_vp')),
            'Corruption: %s' % g.get('corruption'),
            'Mordor: %s (dist %s)' % (
                'entered' if g.get('mordor_entered') else 'no',
                g.get('mordor_dist')),
            'Aragorn crowned: %s' % (
                'YES' if g.get('aragorn_crowned') else 'no'),
            '',
            # USER 2026-07-27: this is the PRE-ROLL allocation DECISION
            # only — the in-game hunt box additionally receives rolled
            # Eye results and FP dice spent on Fellowship actions, so
            # it is normally larger than these numbers.
            'Eyes ALLOCATED pre-roll by turn: %s' % (g.get('eye_allocs')),
            '  (hunt box also gains rolled Eyes + FP fellowship dice)',
            'Avg allocated/turn: %.2f' % (
                sum(g.get('eye_allocs') or [0])
                / max(1, len(g.get('eye_allocs') or [0]))),
            '',
            'SH-captured: %s' % (', '.join(g.get('sh_captured') or [])
                                 or 'none'),
            'FP-captured: %s' % (', '.join(g.get('fp_captured') or [])
                                 or 'none'),
            'Separations: %s' % (
                ', '.join('%s x%d' % kv for kv in Counter(
                    g.get('separations') or []).most_common())
                or 'none'),
            '',
            'Trace: %s' % (g.get('trace_path') or '(not saved)'),
        ]
        if self.actor_critic:
            lines[1] = 'Policy learning: all legal wins have equal terminal reward'
            if g.get('purpose') == 'evaluation':
                lines.insert(0, 'EVALUATION ONLY: excluded from training and matchmaking statistics.')
        if g.get('_provisional') and (not self.actor_critic or g.get('status') == 'terminated'):
            if self.actor_critic and g.get('purpose') == 'evaluation':
                lines[:0] = ['COMPLETED EVALUATION GAME: awaiting batch commit.', '']
            else:
                lines[:0] = ['COMPLETED GAME: not yet committed to a training update.',
                             'Validation and training follow completion of the entire batch.', '']
        if self.actor_critic and g.get('status') != 'terminated':
            lines[:0] = ['UNFINISHED GAME: excluded from training; no winner was assigned.', '']
        # stale-path resolution (USER bug 2026-08-11: quarantine
        # renamed batch_000 -> batch_000_quarantined_stageA, feed rows
        # kept the dead path and the Open button greyed out)
        resolved, roots = self._resolve_trace(g.get('trace_path'), g.get('trace_sha256'))
        self._resolved_trace = resolved
        stored = str(g.get('trace_path') or '')
        if resolved and resolved != stored:
            lines.append('  -> resolved (dir was renamed):')
            lines.append('  %s' % resolved)
        elif not resolved and stored:
            lines.append('  -> MISSING: not found under:')
            lines.extend('     %s' % r for r in roots)
        self._set_detail('\n'.join(lines))
        ok = bool(resolved)
        self.launch_btn.config(state=tk.NORMAL if ok else tk.DISABLED)
        self.attr_btn.config(state=tk.NORMAL if ok and not g.get('_provisional') and not self.actor_critic else tk.DISABLED)

    def _resolve_trace(self, path, expected_sha256=None):
        """resolve_trace with a ~30s-cached basename index (batch
        listings and the card finder resolve in bursts)."""
        now = time.time()
        roots = [os.path.dirname(os.path.abspath(self.log_path)), *trace_resolve.default_roots()]
        if self._trace_index is None or now - self._trace_index_t > 30:
            self._trace_index = trace_resolve.build_index(roots=roots)
            self._trace_index_t = now
        return trace_resolve.resolve_trace(path,
                                           roots=roots, index=self._trace_index,
                                           expected_sha256=expected_sha256)

    def _set_detail(self, txt):
        self.detail.config(state=tk.NORMAL)
        self.detail.delete('1.0', tk.END)
        self.detail.insert('1.0', txt)
        self.detail.config(state=tk.DISABLED)

    # ---- card-usage finder -------------------------------------
    def _finder_load_index(self):
        try:
            with open(CARD_INDEX_PATH, encoding='utf-8') as f:
                self._card_index = json.load(f)
        except Exception:
            self._card_index = {'cards': {d: [] for d in FINDER_DECKS},
                                'usage': {}}
        usage = self._card_index.get('usage', {})
        for deck in FINDER_DECKS:
            ids = self._card_index.get('cards', {}).get(deck, [])
            vals, m = [], {}
            for cid in ids:
                hits = usage.get(cid, [])
                ne = sum(1 for h in hits if not h.get('combat'))
                nc = sum(1 for h in hits if h.get('combat'))
                disp = '%s  [e:%d c:%d]' % (_card_short(cid), ne, nc)
                vals.append(disp)
                m[disp] = cid
            self._finder_boxes[deck]['values'] = vals
            self._finder_maps[deck] = m

    def _finder_pick(self, deck):
        self._finder_deck = deck
        for d in FINDER_DECKS:
            if d != deck:
                self._finder_vars[d].set(d)

    def _finder_rebuild(self):
        self.status.config(text='rebuilding card index...')

        def _run():
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
            if not LEGACY_CONTROLS_ENABLED:
                env['WOTR_LOOP_DIR'] = LOOP_DIR
            subprocess.call([PY, CARD_INDEXER], env=env,
                            cwd=os.path.dirname(CARD_INDEXER))
            self.root.after(0, self._finder_rebuilt)

        threading.Thread(target=_run, daemon=True).start()

    def _finder_rebuilt(self):
        self._finder_load_index()
        self.status.config(text='card index rebuilt')

    def _finder_find(self):
        deck = self._finder_deck
        if not deck:
            self.status.config(text='pick a card first')
            return
        disp = self._finder_vars[deck].get()
        cid = self._finder_maps.get(deck, {}).get(disp)
        if not cid:
            self.status.config(text='pick a card first')
            return
        want_combat = bool(self._finder_combat.get())
        all_hits = [h for h in
                    self._card_index.get('usage', {}).get(cid, [])
                    if bool(h.get('combat')) == want_combat]
        # stale index entries (deleted or quarantine-RENAMED batch
        # dirs): resolve by basename across the trace roots first, show
        # only hits whose trace still exists, and say so when stale rows
        # were dropped (USER 2026-07-28 / 2026-08-11).
        hits = []
        for h in all_hits:
            tp, _roots = self._resolve_trace(h.get('trace', ''))
            if tp:
                h = dict(h, trace=tp)
                hits.append(h)
        n_stale = len(all_hits) - len(hits)
        if not hits:
            self.status.config(
                text='no %splays of %s found%s' % (
                    'combat-' if want_combat else '',
                    _card_short(cid),
                    ' (%d stale entries — Rebuild index)' % n_stale
                    if n_stale else ''))
            return
        if n_stale:
            self.status.config(
                text='%d stale index entries hidden — Rebuild index'
                % n_stale)
        win = tk.Toplevel(self.root)
        win.title('%s - %d %splay(s)' % (
            _card_short(cid), len(hits),
            'combat ' if want_combat else ''))
        win.geometry('640x420')
        tk.Label(win, text='double-click a game to open the replay '
                 'GUI at the exact action').pack(anchor='w')
        lb = tk.Listbox(win)
        sb = ttk.Scrollbar(win, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        for h in hits:
            lb.insert(tk.END, '%s   (step %d)' % (h['label'],
                                                  h['step']))

        def _open(_evt=None):
            sel = lb.curselection()
            if not sel:
                return
            h = hits[sel[0]]
            if not os.path.exists(str(h.get('trace', ''))):
                tp, roots = self._resolve_trace(h.get('trace', ''))
                if not tp:
                    messagebox.showerror(
                        'Trace not found',
                        trace_resolve.missing_message(
                            h.get('trace', ''), roots),
                        parent=win)
                    return
                h = dict(h, trace=tp)
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
            subprocess.Popen(
                [PY, GUI_PY, str(h['trace']),
                 '--step', str(h['step'])],
                env=env, cwd=os.path.dirname(GUI_PY))

        lb.bind('<Double-Button-1>', _open)
        tk.Button(win, text='Open at action', command=_open)\
            .pack(side=tk.BOTTOM, fill=tk.X)

    def _resolved_or_error(self):
        """Re-resolve the selected game's trace at click time (the
        tree may have changed since selection); on failure show the
        in-GUI error naming the searched roots — never launch a child
        that would die on a dead path."""
        g = self.selected_game
        if not g or not g.get('trace_path'):
            return None
        # force a fresh index if the cached resolution went stale
        resolved, roots = self._resolve_trace(g.get('trace_path'), g.get('trace_sha256'))
        if resolved and not os.path.exists(resolved):
            self._trace_index = None
            resolved, roots = self._resolve_trace(g.get('trace_path'), g.get('trace_sha256'))
        if not resolved:
            messagebox.showerror(
                'Trace not found',
                trace_resolve.missing_message(g.get('trace_path'),
                                              roots, g.get('trace_sha256')),
                parent=self.root)
            return None
        self._resolved_trace = resolved
        return resolved

    def launch_attribution(self):
        tp = self._resolved_or_error()
        if not tp:
            return
        pp = (self.selected_game.get('profiles') or {})
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
        attr_py = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'td_attribution.py')
        self._last_attr_proc = subprocess.Popen(
            [PY, attr_py, tp,
             str(pp.get('fp') or 'ring'),
             str(pp.get('sh') or 'corruption')],
            env=env, cwd=os.path.dirname(attr_py))

    def launch_trace(self):
        tp = self._resolved_or_error()
        if not tp:
            return
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
        self._last_trace_proc = subprocess.Popen(
            [PY, GUI_PY, tp], env=env, cwd=os.path.dirname(GUI_PY))

    # ---- Loop status panel + one-click controls (2026-08-11) -----
    def _build_status_tab(self, tab):
        btns = tk.Frame(tab)
        btns.pack(fill=tk.X, padx=4, pady=3)
        tk.Button(btns, text='Refresh status',
                  command=self.refresh_status).pack(side=tk.LEFT)
        self._ctl_btns = []
        for label, mode in (('Stop Loop', 'stop'),
                            ('Restart Loop', 'restart'),
                            ('Clear Artifacts + Restart', 'clear')):
            b = tk.Button(btns, text=label,
                          command=lambda m=mode: self._confirm_and_run(m),
                          state=tk.NORMAL if LEGACY_CONTROLS_ENABLED else tk.DISABLED)
            b.pack(side=tk.LEFT, padx=6)
            self._ctl_btns.append(b)
        self.script_lbl = tk.Label(btns, text='' if LEGACY_CONTROLS_ENABLED else
                                   'Processes managed by this run\'s controller', anchor='w',
                                   fg='#803000')
        self.script_lbl.pack(side=tk.LEFT, padx=10)

        self.state_lbl = tk.Label(tab, text='(collecting status...)',
                                  anchor='w', justify=tk.LEFT,
                                  font=('TkDefaultFont', 10, 'bold'))
        self.state_lbl.pack(fill=tk.X, padx=6, pady=2)

        mid = tk.PanedWindow(tab, orient=tk.HORIZONTAL, sashwidth=5)
        mid.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(mid)
        tk.Label(left, text='Committed batches (games / wins / update)' if self.actor_critic else 'Batches (games / wins / Stage-A census)')\
            .pack(anchor='w')
        cols = ('dir', 'games', 'fp', 'sh', 'census', 'flag')
        self.batch_status = ttk.Treeview(left, columns=cols,
                                         show='headings', height=9)
        for c, w in zip(cols, (200, 52, 40, 40, 230, 88)):
            self.batch_status.heading(c, text=c)
            self.batch_status.column(c, width=w, anchor='w')
        self.batch_status.pack(fill=tk.BOTH, expand=True)
        tk.Label(left, text='Net checkpoints (sha12 / head_act / '
                            'conf-stamp)').pack(anchor='w')
        ncols = ('file', 'sha12', 'head_act', 'conf', 'mtime')
        self.net_status = ttk.Treeview(left, columns=ncols,
                                       show='headings', height=7)
        for c, w in zip(ncols, (190, 100, 84, 90, 100)):
            self.net_status.heading(c, text=c)
            self.net_status.column(c, width=w, anchor='w')
        self.net_status.pack(fill=tk.BOTH, expand=True)
        mid.add(left)

        right = tk.Frame(mid)
        tk.Label(right, text='Current stage' if self.actor_critic else 'loop_log.md tail').pack(anchor='w')
        self.log_tail = tk.Text(right, height=12, wrap='none',
                                state=tk.DISABLED)
        self.log_tail.pack(fill=tk.BOTH, expand=True)
        tk.Label(right, text=('Training batches and run configuration' if not LEGACY_CONTROLS_ENABLED else
                 'Arena / promotion state + canonical launch config (scripts\\restart_loop.bat)'))\
            .pack(anchor='w')
        self.state_txt = tk.Text(right, height=13, wrap='word',
                                 state=tk.DISABLED)
        self.state_txt.pack(fill=tk.BOTH, expand=True)
        mid.add(right)

    # ---- Loop settings tab (USER 2026-08-11: arena checkbox +
    # tunables -> data\v33loop\loop_config.json) -------------------
    def _build_settings_tab(self, tab):
        if self.actor_critic:
            tk.Label(tab, text='This run uses its saved actor-critic configuration. Resume preserves these settings.',
                     anchor='w', justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=6)
            text = tk.Text(tab, wrap='word')
            text.pack(fill=tk.BOTH, expand=True)
            text.insert('1.0', json.dumps(_actor_critic_config(), indent=2))
            text.configure(state=tk.DISABLED)
            return
        tk.Label(tab, text='Settings are written to data\\v33loop\\'
                 'loop_config.json and APPLY AT THE NEXT LOOP '
                 '(RE)START - press [Restart Loop] when ready. '
                 'Fields left at their default are NOT pinned (they '
                 'keep tracking the loop default); the arena '
                 'checkbox is always written explicitly.',
                 bg='#fff3c4', anchor='w', justify=tk.LEFT,
                 wraplength=1180).pack(fill=tk.X, padx=6, pady=4)
        self._set_warn = tk.Label(tab, text='', fg='#a00000',
                                  anchor='w', justify=tk.LEFT,
                                  wraplength=1180)
        self._set_warn.pack(fill=tk.X, padx=6)

        self._arena_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tab, text='Arena promotion gate ENABLED '
                       '(arena_enabled) - OFF trains without the '
                       'promote/revert arena',
                       variable=self._arena_var,
                       font=('TkDefaultFont', 11, 'bold'))\
            .pack(anchor='w', padx=10, pady=6)

        grid = tk.Frame(tab)
        grid.pack(anchor='w', padx=12, pady=2)
        self._set_entries = {}
        r = 0
        for key, dflt, kind, _mn, desc in LOOP_SETTINGS:
            if kind == 'bool':
                continue
            tk.Label(grid, text=key, anchor='w', width=22)\
                .grid(row=r, column=0, sticky='w', pady=1)
            e = tk.Entry(grid, width=12)
            e.grid(row=r, column=1, sticky='w', padx=4)
            self._set_entries[key] = e
            tk.Label(grid, text='default %s' % (
                json.dumps(dflt) if dflt is not None else '(blank)'),
                fg='#666666').grid(row=r, column=2, sticky='w',
                                   padx=8)
            tk.Label(grid, text=desc, fg='#444444', anchor='w')\
                .grid(row=r, column=3, sticky='w', padx=8)
            r += 1

        btns = tk.Frame(tab)
        btns.pack(anchor='w', padx=10, pady=8)
        tk.Button(btns, text='Save settings',
                  command=self._settings_save).pack(side=tk.LEFT)
        tk.Button(btns, text='Reset to defaults (delete '
                  'loop_config.json)',
                  command=self._settings_reset)\
            .pack(side=tk.LEFT, padx=8)
        tk.Button(btns, text='Restart Loop (applies settings)',
                  command=lambda: self._confirm_and_run('restart'),
                  state=tk.NORMAL if LEGACY_CONTROLS_ENABLED else tk.DISABLED)\
            .pack(side=tk.LEFT, padx=8)
        self._set_status = tk.Label(tab, text='', anchor='w',
                                    fg='#004080', justify=tk.LEFT,
                                    wraplength=1180)
        self._set_status.pack(fill=tk.X, padx=10)
        self._settings_populate()
        eff, over, _w = load_loop_config_effective()
        self._set_status.config(text=(
            'loaded: %d override(s) active from loop_config.json'
            % len(over) if over else
            'no loop_config.json - the loop runs built-in defaults'))

    def _settings_populate(self):
        """Fill widgets from the file-or-defaults effective view."""
        eff, _over, warn = load_loop_config_effective()
        self._set_warn.config(text=warn or '')
        self._arena_var.set(bool(eff.get('arena_enabled', True)))
        for key, _d, kind, _mn, _ds in LOOP_SETTINGS:
            if kind == 'bool':
                continue
            e = self._set_entries[key]
            e.delete(0, tk.END)
            v = eff.get(key)
            if v is not None:
                e.insert(0, json.dumps(v))

    def _settings_save(self):
        values = {'arena_enabled': self._arena_var.get()}
        for key, e in self._set_entries.items():
            values[key] = e.get()
        cfg, errs = save_loop_config(values)
        if errs:
            messagebox.showerror(
                'Invalid settings - NOT saved',
                'Fix these and Save again:\n\n'
                + '\n'.join('- %s' % x for x in errs),
                parent=self.root)
            return
        self._settings_populate()      # re-read from disk = proof
        n_over = len([k for k in cfg if k != 'arena_enabled'])
        self._set_status.config(text=(
            'SAVED %s at %s: arena_enabled=%s + %d tunable '
            'override(s) - applies at next loop (re)start'
            % (LOOP_CONFIG_PATH, time.strftime('%H:%M:%S'),
               cfg.get('arena_enabled'), n_over)))
        self.refresh_status()

    def _settings_reset(self):
        if not messagebox.askokcancel(
                'Reset to defaults',
                'Delete data\\v33loop\\loop_config.json and reset '
                'all fields to the loop defaults?\n\n(A running loop '
                'is unaffected until the next (re)start.)',
                parent=self.root):
            return
        try:
            if os.path.exists(LOOP_CONFIG_PATH):
                os.remove(LOOP_CONFIG_PATH)
        except OSError as e:
            messagebox.showerror('Reset failed', repr(e),
                                 parent=self.root)
            return
        self._settings_populate()
        self._set_status.config(
            text='loop_config.json deleted - the loop uses built-in '
                 'defaults at the next (re)start')
        self.refresh_status()

    def _status_tick(self):
        self.refresh()
        self.refresh_status()
        self.root.after(STATUS_REFRESH_MS, self._status_tick)

    def refresh_status(self):
        if self._status_busy:
            return
        self._status_busy = True

        def _worker():
            try:
                st = _collect_status(self._ck_cache)
            except Exception as e:
                st = {'error': repr(e)}

            def _done():
                self._status_busy = False
                self._apply_status(st)
            try:
                self.root.after(0, _done)
            except tk.TclError:
                pass                      # window closed mid-collect

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_status(self, st):
        if st.get('error'):
            self.state_lbl.config(
                text='status collection FAILED: %s' % st['error'])
            return
        pipeline = st.get('pipeline')
        self.pipeline_lbl.config(text=(pipeline['summary'] + '\n' + pipeline['detail']) if pipeline else '')
        state_label = ('PIPELINE: ' + pipeline['summary']) if pipeline else ('LOOP: ' + st['state'])
        if pipeline and pipeline.get('stage') == 'selfplay':
            state_label += ' | ' + st['state']
        self.state_lbl.config(text='%s | disk free: %.1f GB | checked %s' %
                              (state_label, st['disk_free_gb'], time.strftime('%H:%M:%S')))
        self.log_tail.config(state=tk.NORMAL)
        self.log_tail.delete('1.0', tk.END)
        self.log_tail.insert('1.0', '\n'.join(st['tail'])
                             or '(loop_log.md empty or missing)')
        self.log_tail.see(tk.END)
        self.log_tail.config(state=tk.DISABLED)
        self.batch_status.delete(*self.batch_status.get_children())
        for i, row in enumerate(st['batches']):
            self.batch_status.insert('', tk.END, iid=str(i),
                                     values=row)
        self.net_status.delete(*self.net_status.get_children())
        for i, row in enumerate(st['nets']):
            self.net_status.insert('', tk.END, iid='n%d' % i,
                                   values=row)
        txt = [('BATCH TRAINING STATE:' if not LEGACY_CONTROLS_ENABLED else 'ARENA / PROMOTION (loop_state.json):'),
               st['loop_state'], '',
               'SAVED RUN SETTINGS:' if self.actor_critic else 'EFFECTIVE LOOP SETTINGS (loop_config.json overrides '
               'marked *, else defaults; edit on the Loop settings '
               'tab, applies at next (re)start):']
        txt += ['  ' + s for s in st.get('settings', [])]
        txt += ['',
                ('RUN LAUNCH CONFIG:' if not LEGACY_CONTROLS_ENABLED else
                 'CANONICAL LAUNCH CONFIG (scripts\\restart_loop.bat '
                 '= intended env of the running loop):')]
        txt += ['  ' + c for c in st['config']]
        self.state_txt.config(state=tk.NORMAL)
        self.state_txt.delete('1.0', tk.END)
        self.state_txt.insert('1.0', '\n'.join(txt))
        self.state_txt.config(state=tk.DISABLED)

    # ---- control buttons -----------------------------------------
    _KILL_TXT = ('- taskkill /T /F the pid in data\\v33loop\\loop.pid'
                 ' (if alive)\n'
                 '- kill orphaned run_td_v33loop / run_td_v33sp / '
                 'launch_smoke_gate\n  processes belonging to THIS '
                 'tree (the dashboard and replay\n  windows are NOT '
                 'touched)\n')
    _CLEAR_TXT = (
        '- scripts\\clear_artifacts.bat then DELETES:\n'
        '    batch_* dirs (incl. quarantined), stale_quarantine_*,\n'
        '    buffer contents, all_games.jsonl, probe_ref_seed.json,\n'
        '    loop_state.json, smoke_gate_last.json, '
        'smoke_gate_traces,\n'
        '    runner_*.log, loop.pid, data\\td_v33_traces\\*,\n'
        '    data\\td_v33_games.jsonl, data\\td_v33_smoke_traces\\*\n'
        '  and KEEPS: baseline_/live_/last_good_*.pt, '
        'baseline_manifest.json,\n'
        '    probeset_v1.pkl.gz + manifest, seeds_v5_preserved\\, '
        'baseline_games\\,\n'
        '    SEED_*.pt; loop_log.md is ROTATED to loop_log_prev.md\n')

    def _script_message(self, mode):
        if mode == 'stop':
            return ('STOP LOOP\n\nscripts\\restart_loop.bat stop '
                    'will:\n' + self._KILL_TXT +
                    '\nNothing is deleted.\n\nProceed?')
        if mode == 'restart':
            return ('RESTART LOOP (no clearing)\n\n'
                    'scripts\\restart_loop.bat restart will:\n'
                    + self._KILL_TXT +
                    '- relaunch src\\run_td_v33loop.py DETACHED with '
                    'the canonical env\n  (see config box), stdout -> '
                    'data\\v33loop\\runner_stdout.log\n'
                    '\nNo artifacts are deleted: the loop resumes '
                    'from the existing\nbatch dirs and live nets '
                    '(pre-flight smoke gate runs first).\n\nProceed?')
        return ('CLEAR ARTIFACTS + RESTART (full reset to '
                'baseline)\n\nscripts\\restart_loop.bat clear '
                'will:\n' + self._KILL_TXT + self._CLEAR_TXT +
                '- relaunch the loop DETACHED, fresh from batch 0 '
                '(smoke gate first)\n\nTHIS DELETES ALL RUN '
                'ARTIFACTS OF THE CURRENT LOOP.\n\nProceed?')

    def _confirm_and_run(self, mode):
        if not LEGACY_CONTROLS_ENABLED:
            messagebox.showinfo('Dedicated run controller',
                'This dashboard is viewing %s. Use its dedicated controller '
                'to stop or restart the run.' % LOOP_DIR, parent=self.root)
            return
        if self._script_proc is not None:
            messagebox.showinfo(
                'Busy', 'A control script is already running - wait '
                'for its exit status.', parent=self.root)
            return
        if not os.path.exists(RESTART_BAT):
            messagebox.showerror(
                'Missing script',
                'Not found: %s\n(the canonical launcher was not '
                'installed)' % RESTART_BAT, parent=self.root)
            return
        if not messagebox.askokcancel(
                'Confirm: %s' % mode, self._script_message(mode),
                icon='warning', parent=self.root):
            return
        self._script_proc = mode          # busy marker pre-spawn
        for b in self._ctl_btns:
            b.config(state=tk.DISABLED)
        self.script_lbl.config(
            text='running restart_loop.bat %s ...' % mode)

        def _worker():
            rc, err = None, None
            try:
                with open(SCRIPT_LOG, 'w', encoding='utf-8',
                          errors='replace') as logf:
                    p = subprocess.Popen(
                        ['cmd.exe', '/c', RESTART_BAT, mode],
                        stdout=logf, stderr=subprocess.STDOUT,
                        cwd=ROOT, creationflags=CREATE_NO_WINDOW)
                    rc = p.wait()
            except Exception as e:
                err = repr(e)

            def _done():
                self._script_proc = None
                for b in self._ctl_btns:
                    b.config(state=tk.NORMAL)
                if err is not None:
                    self.script_lbl.config(
                        text='restart_loop.bat %s FAILED to run: %s'
                        % (mode, err))
                elif rc == 0:
                    self.script_lbl.config(
                        text='restart_loop.bat %s: exit 0 (OK)'
                        % mode)
                else:
                    self.script_lbl.config(
                        text='restart_loop.bat %s: exit %s (FAILED) '
                        '- see %s' % (mode, rc, SCRIPT_LOG))
                self.refresh_status()
            try:
                self.root.after(0, _done)
            except tk.TclError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    Dashboard(path).run()
