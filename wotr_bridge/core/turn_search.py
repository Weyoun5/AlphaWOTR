"""TURN SEARCH: the live server's side of the rest-of-turn search.

After the server's ordinary search has picked a move, this module asks a second process (the kit named in
wotr_bridge/turn_search.json) whether another move is better when the REST OF THE TURN and the whole next turn are played
out: the candidates are played on a fast copy of the engine with common dice, judged two turns on, and the best
challenger is played once more on fresh dice against the server's own move.  Only a challenger that wins that last
comparison by a clear margin replaces the server's move.

It is OFF unless wotr_bridge/turn_search.json exists with "enabled": true.  The file's paths (kit, policy, values,
log_dir, python) are places in the game's folder unless they are absolute: a game that carries its kit can be moved.
Whatever goes wrong - no kit, the service does
not start, it refuses the position, it takes too long, its move is not in the menu - improve() returns the server's own
move unchanged, with one line on the console saying why.  Three failures in a row switch it off for the session.
It only looks at decisions of the action phase (the fast engine starts from a rolled turn).

WHAT THE PC GIVES is the service's own business (the kit's ts_device.py: its NVIDIA GPU if that can really be used,
else the processor; as many workers as cores and free memory hold).  Here only what follows from it:
  the Python - the file's "python" if it names a file that is there, else the Python this server runs on;
  a GPU that fails while the service runs (it ends with GPU_FAILED, or its answer says so): the service is started
    again with --device cpu, for the rest of the session;
  a PC that cannot run it now (CANNOT_RUN: memory): off for the session, said once;
  a setting this PC is too slow for (three decisions in a row given up as "out of time"): that setting is not asked
    again this session - the cheaper ones still are;
  the strength of a slider nobody has moved (strength.default): gpu() - the service writes what it decided into TOLD
    in the client's folder, where the client's slider reads it too.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FLAG = Path(__file__).resolve().parents[1] / 'turn_search.json'
GAME = FLAG.parents[1]          # the game's folder: a path of the file that is not absolute is a place in the game
GPU_FAILED, CANNOT_RUN = 4, 5   # the service's exit codes (the kit's ts_device.py)
SAME_BUT_BETTER_ROWS = 400      # improve(): the longest menu a switch is weighed against in full
TOLD = 'ai_turn_search.json'    # in the client's folder: what the PC gives, in the service's own words (its --tell)
_S = {'proc': None, 'conn': None, 'cfg': None, 'fails': 0, 'off': False, 'cannot_run': False, 'log': None, 'told': None,
      'asked': 0, 'switched': 0, 'seconds': 0.0, 'nets': {}, 'unknown': set(), 'device': None, 'slow': {}}


def config():
    if _S['off']:
        return None
    if _S['cfg'] is None:
        try:
            cfg = json.loads(FLAG.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            _S['off'] = True
            return None
        if not cfg.get('enabled'):
            _S['off'] = True
            return None
        _S['cfg'] = cfg
    return _S['cfg']


def _say(text):
    print('  [turn search] ' + text, flush=True)


def channel(name, windows=os.name == 'nt'):
    """The address of the channel to the service: a named pipe on Windows, a socket file in the temporary folder
    elsewhere (the connection tells the two apart by the address)."""
    return '\\\\.\\pipe\\' + name if windows else os.path.join(tempfile.gettempdir(), name + '.sock')


def gpu():
    """Do the nets of THIS session's service run on the GPU?  Its own word (TOLD, written seconds after it starts);
    no word yet, another session's word, a GPU that failed since, the turn search off: no."""
    if config() is None or _S['device'] == 'cpu' or not _S['told']:
        return False
    try:
        with open(_S['told'], encoding='utf-8') as f:
            told = json.load(f)
        return told.get('parent') == os.getpid() and told.get('device') == 'cuda'
    except (OSError, ValueError, AttributeError):
        return False


def start(net_tag=None, client_dir=None):
    """Start the service process (it loads its nets while the game is being set up).  Returns at once.
    client_dir: the client's folder, where the service writes what the PC gives (TOLD).
    net_tag: the pair the server plays (..._fp<key>_sh<key>: win, mil, cor).  Every seat is searched the same way, each
    with ITS net: the key goes with every request, and the service plays the seat's moves with that net and judges them
    with that net's own value (the kit's search_settings.json, "nets").  A net the kit does not hold is said once, and
    that seat keeps the ordinary search alone."""
    if net_tag:
        import re
        m = re.search(r'_fp([a-z]+)_sh([a-z]+)$', str(net_tag))
        if m:
            _S['nets'] = {'fp': m.group(1), 'sh': m.group(2)}
    if client_dir:
        _S['told'] = os.path.join(client_dir, TOLD)
    cfg = config()
    if cfg is None:
        return False
    if _S['proc'] is not None and _S['proc'].poll() is None:
        return True
    try:
        kit = GAME / cfg['kit']
        log = open(GAME / (cfg.get('log_dir') or kit) / 'turn_search_service.log', 'a', encoding='utf-8')
        _S['log'] = log
        env = dict(os.environ)
        for k in ('WOTR_REWARD_CONTRACT', 'WOTR_NATIVE', 'WOTR_LIVE_SEARCH_BACKEND', 'WOTR_FREE_CANDS', 'WOTR_FREE_FORCE',
                  'WOTR_LIVE_INFERENCE_THREADS'):
            env.pop(k, None)                      # the service runs the kit's own engine settings, not the live server's
        python = sys.executable
        if cfg.get('python'):
            python = str(GAME / cfg['python'])
            if not os.path.isfile(python):
                _say('the Python the file names is not there (%s): this server\'s own Python runs the service' % python)
                python = sys.executable
        workers = str(cfg.get('workers', 'auto'))
        _S['pipe'] = channel('wotr_turn_search_%d_%d' % (os.getpid(), int(time.time())))
        _S['proc'] = subprocess.Popen(
            [python, '-B', str(kit / 'ts_service.py'), '--pipe', _S['pipe'], '--workers', workers,
             '--policy', str(GAME / cfg['policy']), '--values', str(GAME / cfg['values']), '--parent', str(os.getpid())]
            + (['--device', _S['device']] if _S['device'] else []) + (['--tell', _S['told']] if _S['told'] else []),
            cwd=str(kit), env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        _S['conn'] = None
        _say('starting the service (workers: %s%s)' % (workers, ', on the processor' if _S['device'] == 'cpu' else ''))
        return True
    except Exception as error:
        _fail('could not start: %s' % (str(error)[:200],))
        return False


def _connect(cfg):
    from multiprocessing.connection import Client
    if _S['conn'] is not None and _S['proc'] is not None and _S['proc'].poll() is None:
        return _S['conn']
    _S['conn'] = None
    if not start():
        raise RuntimeError('the service is not running')
    deadline = time.time() + float(cfg.get('start_seconds', 180))
    last = None
    while time.time() < deadline:
        if _S['proc'].poll() is not None:
            code = _S['proc'].returncode
            if code == GPU_FAILED and _S['device'] != 'cpu':
                _gpu_failed()
                if start():
                    deadline = time.time() + float(cfg.get('start_seconds', 180))
                    continue
            if code == CANNOT_RUN:
                _S['off'] = _S['cannot_run'] = True
                _say('this PC cannot run the turn search now (too little free memory: turn_search_service.log): '
                     'off for this session')
            raise RuntimeError('the service ended with code %s (see turn_search_service.log)' % code)
        try:
            _S['conn'] = Client(_S['pipe'], authkey=b'wotr-turn-search')
            return _S['conn']
        except (OSError, EOFError) as error:
            last = error
            time.sleep(0.5)
    raise RuntimeError('the service did not come up: %r' % (last,))


def _gpu_failed():
    """The GPU failed under the service: from now on it is started on the processor."""
    if _S['device'] != 'cpu':
        _S['device'] = 'cpu'
        _say('the GPU failed: the service runs on the processor for the rest of this session (slower)')


def _fail(why):
    _S['fails'] += 1
    _say('%s -> the ordinary search\'s move is played' % why)
    try:
        if _S['conn'] is not None:
            _S['conn'].close()
    except Exception:
        pass
    _S['conn'] = None
    if _S['fails'] >= 3:
        _S['off'] = True
        _say('three failures in a row: switched off for this session')


def _same(a, b):
    return a.action_type == b.action_type and a.die_idx == b.die_idx and a.params == b.params


def _same_but_better(ai, state, side, menu, hit, own):
    """THE SAME BUT BETTER (ai._strictly_better): the move the turn search switches to is a decided move like any
    other.  It is weighed against the whole menu up to SAME_BUT_BETTER_ROWS options (2 ms an option), in a longer
    menu against the ordinary search's own pick (which that search has cleared against its root options).
    -> the move to play: `hit`, or the option that is the same but better."""
    rows = list(menu) if len(menu) <= SAME_BUT_BETTER_ROWS else [hit, own]
    at = next(i for i, a in enumerate(rows) if a is hit)
    verdicts = ai._strictly_better(state, side, rows)
    if verdicts is not None and verdicts[at] == ai.BANNED:
        _say('its move %s walks away from a siege the Shadow may not lift (the siege ban): keeps %s' % (
            hit.action_type, own.action_type))
        return own
    moved = rows[ai._move_to_better(at, verdicts)]
    if moved is not hit:
        _say('its move %s is the same as %s but worse: plays that' % (hit.action_type, moved.action_type))
    return moved


def improve(ai, state, side, owner, seed, menu, own, setting):
    """own: the move the server's ordinary search picked (an entry of menu).  -> the move to play: own, or the entry of
    menu the turn search confirmed as better.  state: what the search may see (hidden cards already dealt at random).
    setting: how hard the service searches at this decision - a name of the kit's search_settings.json (strength.py
    chooses it); None = this decision is not searched."""
    cfg = config()
    if cfg is None or setting is None or len(menu) < 2 or not state.dice_rolled or _S['slow'].get(setting, 0) >= 3:
        return own
    seat = 'fp' if side is ai.Side.FREE_PEOPLES else 'sh'
    if seat in _S['unknown']:
        return own
    started = time.perf_counter()
    try:
        conn = _connect(cfg)
        conn.send({'op': 'choose', 'state': pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), 'side': seat,
                   'seed': int(seed) & 0x7FFFFFFF, 'rows': len(menu), 'ref': (own.die_idx, own.action_type, own.params),
                   'setting': setting, 'net': _S['nets'].get(seat, 'win'),
                   'owner': None if owner is None else ('fp' if owner is ai.Side.FREE_PEOPLES else 'sh')})
        if not conn.poll(float(cfg.get('decision_seconds', 60))):
            # the answer may still arrive later and would be read as the next decision's: start over with a new service
            try:
                _S['proc'].kill()
            except Exception:
                pass
            raise RuntimeError('no answer in %s s' % cfg.get('decision_seconds', 60))
        ans = conn.recv()
    except Exception as error:
        _fail('the service failed: %s' % (str(error)[:200],))
        return own
    if not ans.get('ok'):
        if ans.get('no_such_net'):
            _S['unknown'].add(seat)
            _say('%s -> this seat keeps the ordinary search alone' % str(ans.get('why'))[:200])
            return own
        if ans.get('gpu_failed'):
            _gpu_failed()
        _fail('refused: %s' % str(ans.get('why'))[:300])
        return own
    _S['fails'] = 0
    _S['asked'] += 1
    _S['seconds'] += time.perf_counter() - started
    how = '%s: %.1f s, %d playouts%s' % (setting, time.perf_counter() - started, ans.get('playouts') or 0,
                                         '; fewer than the setting\'s, to fit the time' if ans.get('cut') else '')
    if ans.get('out_of_time'):
        _S['slow'][setting] = _S['slow'].get(setting, 0) + 1
        _say('keeps %s - out of time before anything was decided (%s)' % (own.action_type, how))
        if _S['slow'][setting] >= 3:
            _say('this PC is too slow for the setting %s: it is not asked again this session '
                 '(a lower position of the AI panel\'s strength slider needs less)' % setting)
        return own
    _S['slow'][setting] = 0
    if not ans.get('switched'):
        _say('keeps %s (%s)' % (own.action_type, how))
        return own
    act = ai.Action(die_idx=int(ans['die']), action_type=str(ans['kind']), params=dict(ans['params'] or {}))
    hit = next((a for a in menu if _same(a, act)), None)
    if hit is None:
        try:
            from ai_policies import action_policy_key
            key = action_policy_key(act)
            hit = next((a for a in menu if action_policy_key(a) == key), None)
        except Exception:
            hit = None
    if hit is None:
        _fail('its move (%s) is not in the menu' % act.action_type)
        return own
    hit = _same_but_better(ai, state, side, menu, hit, own)
    if hit is own:
        return own
    _S['switched'] += 1
    _say('PLAYS %s %s instead of %s %s: +%.1f +- %.1f points two turns on (%s)' % (
        hit.action_type, json.dumps(hit.params, default=str, sort_keys=True)[:110], own.action_type,
        json.dumps(own.params, default=str, sort_keys=True)[:110], 100 * (ans.get('gain') or 0), 100 * (ans.get('se') or 0),
        how))
    return hit


def close():
    try:
        if _S['conn'] is not None:
            _S['conn'].send({'op': 'quit'})
            _S['conn'].close()
    except Exception:
        pass
    _S['conn'] = None
