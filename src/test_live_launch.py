"""Launch-shell pins: ai_panel.py / live_view.py / pid_owner.py on the
LIVE path (wotr_bridge\\live status grammar).

Unit instruments of the human-facing shell; front-end evidence comes
only from the real-client legs (I4).  Each test takes a `fails` list
and appends (name, detail) tuples; main() runs them all.
"""
import importlib.util
import json
import os
import sys
import tempfile
import time

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path, argv1):
    """Import a script module with sys.argv[1] = client dir (both
    shell scripts read CLIENT_DIR from argv at import)."""
    old = sys.argv
    sys.argv = [os.path.basename(path), argv1]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    finally:
        sys.argv = old


# ---------------------------------------------------------------- pid_owner
def test_pid_owner_failed_probe_not_cached(fails):
    """MEASURES: a probe that answers None ('cannot prove') is NOT
    remembered for the pid's lifetime - the next owner_check probes
    again and gets the real answer."""
    import pid_owner as po
    calls = []
    answers = [None, 'python.exe ai_panel.py']
    orig_probe, orig_ct = po._probe, po._create_time
    po._CACHE.clear()
    try:
        po._probe = lambda key: (calls.append(key), answers.pop(0))[1]
        po._create_time = lambda key: 123.0
        a = po.pid_command_line(4242)
        b = po.pid_command_line(4242)
        c = po.pid_command_line(4242)
    finally:
        po._probe, po._create_time = orig_probe, orig_ct
        po._CACHE.clear()
    if a is not None:
        fails.append(('first_probe_answer', repr(a)))
    if b != 'python.exe ai_panel.py':
        fails.append(('failed_probe_was_cached', repr(b)))
    if c != 'python.exe ai_panel.py' or len(calls) != 2:
        fails.append(('real_answer_not_cached', (c, len(calls))))


# ---------------------------------------------------------------- ai_panel
def test_panel_turn_line_live_grammar(fails):
    """MEASURES: the 'Turn N - X to act' text from the LIVE status
    (turn/to_act/alternation nested under human_menu); the monolith's
    top-level keys are accepted too."""
    td = tempfile.mkdtemp()
    m = _load('ai_panel_t2', os.path.join(_BRIDGE, 'ai_panel.py'), td)
    fn = getattr(m, 'turn_text', None)
    if fn is None:
        fails.append(('turn_text_missing', ''))
        return
    live = {'human_menu': {'turn': 3, 'to_act': 'SHADOW',
                           'alternation': 'FREE_PEOPLES'}}
    t = fn(live)
    if 'Turn 3' not in t or 'SHADOW' not in t:
        fails.append(('live_turn_line', repr(t)))
    live2 = {'human_menu': {'turn': 4, 'to_act': None,
                            'alternation': 'FREE_PEOPLES'}}
    t2 = fn(live2)
    if 'Turn 4' not in t2 or 'FREE_PEOPLES' not in t2:
        fails.append(('live_turn_line_alternation', repr(t2)))
    if fn({}) is not None:
        fails.append(('empty_status_turn', repr(fn({}))))
    if any(ord(ch) > 127 for ch in (t + t2)):
        fails.append(('non_ascii_turn_text', repr(t)))


def test_panel_death_line_reads_live_console_log(fails):
    """MEASURES: the dead-server diagnosis reads ai_server_console.log
    (the launcher's redirect) scoped to the LIVE server's own banner,
    so a previous run's error is never quoted."""
    td = tempfile.mkdtemp()
    m = _load('ai_panel_t3', os.path.join(_BRIDGE, 'ai_panel.py'), td)
    fn = getattr(m, 'server_death_line', None)
    if fn is None:
        fails.append(('server_death_line_missing', ''))
        return
    p = os.path.join(td, 'ai_server_console.log')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('nets loaded: wotr_{fp,sh}_corpus_live.pt\n'
                'watching C:/x/ai_bridge_request.log  (poll 0.10s)\n'
                'Traceback (most recent call last):\n'
                'RuntimeError: OLD RUN DIED\n'
                'nets loaded: wotr_{fp,sh}_corpus_live.pt\n'
                'watching C:/x/ai_bridge_request.log  (poll 0.10s)\n'
                '[LIVE_SHADOW]   12 log lines -> ok (0.20s) | 1 line(s)\n')
    why = fn(td)
    if why is not None:
        fails.append(('previous_run_error_quoted', repr(why)))
    with open(p, 'a', encoding='utf-8') as f:
        f.write('! server cycle error: DriverHalt(\'EMISSION HOLD\')\n')
    why = fn(td)
    if not why or 'EMISSION HOLD' not in why:
        fails.append(('current_run_error_missed', repr(why)))
    # the monolith's log name is only a fallback
    os.remove(p)
    with open(os.path.join(td, 'ai_server.log'), 'w',
              encoding='utf-8') as f:
        f.write('RNG seed: 5\nREFUSED: lock held\n')
    why = fn(td)
    if not why or 'REFUSED' not in why:
        fails.append(('fallback_log_not_read', repr(why)))


def test_panel_note_keeps_local_feedback(fails):
    """MEASURES: a note the panel set itself (button feedback) is not
    overwritten by an UNCHANGED status note; a CHANGED status note
    wins; server notes (status 'notes' tail) are rendered."""
    td = tempfile.mkdtemp()
    m = _load('ai_panel_t4', os.path.join(_BRIDGE, 'ai_panel.py'), td)
    fn = getattr(m, 'note_text', None)
    if fn is None:
        fails.append(('note_text_missing', ''))
        return
    # (status_note, last_status_note, local_note) -> text to show
    t = fn({'note': 'waiting for your combat card'},
           'waiting for your combat card', 'RESET + RESYNC sent')
    if t != 'RESET + RESYNC sent':
        fails.append(('local_note_clobbered', repr(t)))
    t = fn({'note': 'illegal: no such move'},
           'waiting for your combat card', 'RESET + RESYNC sent')
    if t != 'illegal: no such move':
        fails.append(('changed_status_note_ignored', repr(t)))
    t = fn({'note': '', 'notes': ['panel: resync requested',
                                  'panel: turn override IGNORED - '
                                  'owes a decision']},
           '', '')
    if 'turn override IGNORED' not in t:
        fails.append(('server_notes_not_rendered', repr(t)))


def test_panel_one_shot_keys_expire(fails):
    """MEASURES: a one-shot lever (skip_pending / reset_resync /
    whose_turn_epoch / clear_error) is removed from the control file
    by the panel's own expiry, so a server restart cannot replay it."""
    td = tempfile.mkdtemp()
    m = _load('ai_panel_t5', os.path.join(_BRIDGE, 'ai_panel.py'), td)
    fn = getattr(m, 'clear_control_keys', None)
    if fn is None:
        fails.append(('clear_control_keys_missing', ''))
        return
    m._write_control(sims=200, skip_pending=time.time(),
                     reset_resync=time.time())
    fn('skip_pending', 'reset_resync')
    cur = m.read_control()
    if 'skip_pending' in cur or 'reset_resync' in cur:
        fails.append(('one_shot_key_persisted', repr(cur)))
    if cur.get('sims') != 200:
        fails.append(('other_key_clobbered', repr(cur)))


# ---------------------------------------------------------------- live_view
def test_view_mask_fails_closed(fails):
    """MEASURES: with NO readable status (pkl-before-json window, open
    failure) BOTH hands are masked; a sided status masks the AI's
    side; the last known sided status is reused."""
    import live_view as lv
    fn = getattr(lv, 'mask_sides', None)
    if fn is None:
        fails.append(('mask_sides_missing', ''))
        return
    both = fn(None, None)
    if set(both) != {'FREE_PEOPLES', 'SHADOW'}:
        fails.append(('no_status_not_fail_closed', repr(both)))
    one = fn({'mode': 'sided', 'ai_side': 'SHADOW'}, None)
    if set(one) != {'SHADOW'}:
        fails.append(('sided_mask', repr(one)))
    last = fn(None, {'mode': 'sided', 'ai_side': 'FREE_PEOPLES'})
    if set(last) != {'FREE_PEOPLES'}:
        fails.append(('last_known_not_reused', repr(last)))
    # the report honours a SET of sides
    import game_state as gs

    class _P(object):
        def __init__(self, hand):
            self.hand = hand

    class _S(object):
        hands = {gs.Side.FREE_PEOPLES: _P(['fp_1']),
                 gs.Side.SHADOW: _P(['sh_1'])}
    rep = lv.build_card_report(_S(), hide_side=both)
    if 'fp_1' in rep or 'sh_1' in rep:
        fails.append(('report_names_masked_hand', rep[:200]))
    rep1 = lv.build_card_report(_S(), hide_side='SHADOW')
    if 'fp_1' not in rep1 or 'sh_1' in rep1:
        fails.append(('report_single_side', rep1[:200]))


def test_view_parser_na_is_not_diff(fails):
    import live_view as lv
    st = {'turn': 2, 'phase': 'live', 'active': None, 'last': 'x',
          'client_divergences': 0,
          'engine_vs_parser': 'n/a (sided: parser IS the view)',
          'engine_vs_client': 'OK'}
    t = lv._make_title(st)
    s = lv._status_text(st)
    if 'DIFF' in t or 'DIFF' in s:
        fails.append(('na_rendered_as_DIFF', (t, s)))
    if 'n/a' not in t or 'n/a' not in s:
        fails.append(('na_not_rendered', (t, s)))
    st['engine_vs_parser'] = '3 diffs'
    if 'DIFF' not in lv._make_title(st):
        fails.append(('real_diff_lost', lv._make_title(st)))


def test_view_poll_keeps_mtime_until_rendered(fails):
    """MEASURES: a snapshot whose pickle load fails is retried on the
    next poll (last_mtime advances only after a successful render);
    the pickle is read as bytes with a short open."""
    import pickle
    import live_view as lv
    td = tempfile.mkdtemp()
    pkl = os.path.join(td, 'ai_live_state.pkl')
    with open(pkl, 'wb') as f:
        f.write(b'not a pickle')

    class _Root(object):
        def title(self, *_a):
            pass

    class _Gui(object):
        root = _Root()
        mirror_marks = None
        state = None

        def refresh(self):
            pass

    class _V(object):
        pass
    v = _V()
    v.pkl = pkl
    v.stj = os.path.join(td, 'ai_live_status.json')
    v.last_mtime = 0
    v.gui = _Gui()
    v._last_status = None
    v.update_card_panel = lambda *a: None
    try:
        lv.LiveView.poll_once(v)
    except Exception:
        pass
    if v.last_mtime != 0:
        fails.append(('mtime_advanced_on_failed_load', v.last_mtime))
    with open(pkl, 'wb') as f:
        pickle.dump({'ok': 1}, f)
    ok = lv.LiveView.poll_once(v)
    if not ok or v.last_mtime == 0 or v.gui.state != {'ok': 1}:
        fails.append(('good_snapshot_not_rendered',
                      (ok, v.last_mtime, v.gui.state)))


def test_view_marks_from_structured_regions(fails):
    """MEASURES: divergence rings come from the status' structured
    'mismatch_regions' list (the live writer's grammar)."""
    import pickle
    import live_view as lv
    td = tempfile.mkdtemp()
    pkl = os.path.join(td, 'ai_live_state.pkl')
    with open(pkl, 'wb') as f:
        pickle.dump({'ok': 1}, f)
    stj = os.path.join(td, 'ai_live_status.json')
    with open(stj, 'w', encoding='utf-8') as f:
        json.dump({'mode': 'sided', 'ai_side': 'SHADOW',
                   'mismatches': ['sidecar_gap: seq 3 has no meta'],
                   'mismatch_regions': ['osgiliath', 'Minas Tirith']},
                  f)

    class _Root(object):
        def title(self, *_a):
            pass

    class _Gui(object):
        root = _Root()
        mirror_marks = None
        state = None

        def refresh(self):
            pass

    class _V(object):
        pass
    v = _V()
    v.pkl, v.stj, v.last_mtime, v.gui = pkl, stj, 0, _Gui()
    v._last_status = None
    v.update_card_panel = lambda *a: None
    lv.LiveView.poll_once(v)
    mk = v.gui.mirror_marks
    if mk != ['osgiliath', 'minas tirith']:
        fails.append(('structured_marks', repr(mk)))


def main():
    tests = [test_pid_owner_failed_probe_not_cached,
             test_panel_turn_line_live_grammar,
             test_panel_death_line_reads_live_console_log,
             test_panel_note_keeps_local_feedback,
             test_panel_one_shot_keys_expire,
             test_view_mask_fails_closed,
             test_view_parser_na_is_not_diff,
             test_view_poll_keeps_mtime_until_rendered,
             test_view_marks_from_structured_regions]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            fails.append((t.__name__ + ':EXC',
                          traceback.format_exc()[-600:]))
        print('%-48s %s' % (t.__name__,
                            'ok' if len(fails) == n0 else 'FAIL'))
    for name, detail in fails:
        print('  FAIL %s: %s' % (name, detail))
    print('%d test(s), %d failure(s)' % (len(tests), len(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
