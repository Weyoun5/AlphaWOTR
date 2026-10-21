#!/usr/bin/env python3
# -*- coding: ascii -*-
"""FLH regression pins (2026-08-30) - fleet seat/harness gaps H1-H3.

H1  (par5_g3 T6, FLEET_FAMILIES D5): the AI demanded "roll ONLY die 1
    (select 101, then roll)" and the seat's `roll` verb could only
    send whole-zone calls - human_gestures.human_roll supports subset
    pids and a hand-built `!CALL roll_action_dice Game.2043 101`
    worked first try.  The verb takes optional pid args
    (`do <name> roll 101 102`) that pass straight through to the
    subset path; a NON-pid arg refuses loudly instead of silently
    degrading to a whole-zone roll (the loud-failure ground rule).

H2  (par5_g4 T4, FLEET_FAMILIES D2): waiting_on `client card_pick
    (WWSAT hand discard)` had NO seat route - the card_pick funnel is
    fed only by the client's own random-discard line (live_server.
    _feed_random_discards), reachable via the `!CALL wwsat` directive,
    and the seat's `raw` verb refused every `!CALL` line.  `raw` now
    accepts the client directive lane (`!CALL` / `!REPLAY`,
    PROTOCOL.md) and `state` surfaces the route as a hint next to the
    wedging wait states.

H3  (par5_g4 D1): the sticky `LIVE HALT ... [sticky - press RESET +
    RESYNC to clear]` was GUI-only.  The seat verb `reset_resync`
    writes the SAME one-shot control keys the panel button writes
    (ai_panel.full_reset: reset_resync + clear_error, epoch-stamped,
    merge-write into <client>/ai_panel_control.json) and expires them
    like the panel's TTL.

Run:  python test_audit0829_FLH.py     (exit 0 = all pass)
"""

import inspect
import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LAB = r'C:\Users\sarve\Downloads\WOTR_AI_training_lab'
for _p in (os.path.join(_LAB, 'agent_seat'),
           os.path.join(_LAB, 'scenarios'),
           os.path.join(_LAB, 'audit'),
           r'C:\Users\sarve\Downloads\WOTR_AI_training\wotr_bridge'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_seat                            # noqa: E402
import human_gestures as hg                  # noqa: E402
import run_scenario as rs                    # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# stubs: a Seat without a booted chain (the verbs under pin only touch
# status/feed/inject plumbing, all faked here; human_gestures/emitter
# stay REAL so the constructed client calls are the real ones)
# ---------------------------------------------------------------------------

class _Ctx(object):
    human_side = 'SHADOW'

    def __init__(self):
        self.injected = []

    def inject(self, label, lines):
        self.injected.append((label, [str(l) for l in lines]))
        return label

    def local_nick(self):
        return 'Wey'


class _FakeRun(object):
    def __init__(self, tmpdir, status=None):
        self.client = tmpdir
        self.ctx = _Ctx()
        self._status = dict(status or {})
        self.feed = os.path.join(tmpdir, 'ai_feed.txt')

        class _Sc(object):
            human_side_name = 'SHADOW'
        self.sc = _Sc()

    def status(self):
        return self._status

    def inject(self, label, lines):
        # the fake client "executes": every injected line grows the feed
        with open(self.feed, 'a', encoding='utf-8') as f:
            for l in lines:
                f.write(str(l) + '\n')
        self.ctx.injected.append((label, [str(l) for l in lines]))
        return lines

    def server_halted(self):
        return None

    def cycle_errors(self):
        return 0


def make_seat(status=None):
    tmpdir = tempfile.mkdtemp(prefix='flh_pin_')
    seat = object.__new__(agent_seat.Seat)
    seat.hg = hg
    seat.RS = rs
    seat.run = _FakeRun(tmpdir, status)
    seat.findings = os.path.join(tmpdir, 'findings.jsonl')
    seat._last_alarms = 0
    return seat, tmpdir


_NO_EYES = {'human_menu': {'eye_minimum': 0, 'eyes_allocated': 0}}


# ---------------------------------------------------------------------------
# H1: subset roll pids pass through; junk args refuse loudly
# ---------------------------------------------------------------------------

@test
def test_h1_roll_subset_args_construct_the_select_call():
    """par5_g3 T6: `roll 101 102` must reach human_roll's subset path
    and construct the exact select-form client call that unwedged the
    session (`!CALL roll_action_dice Game.2043 101 102`)."""
    seat, _ = make_seat(_NO_EYES)
    out = seat.t_roll(['101', '102'])
    assert out.get('rolled_pids') == ['101', '102'], \
        'H1 RED: explicit pids did not pass through: %r' % (out,)
    label, lines = seat.run.ctx.injected[-1]
    assert label == 'roll_pool', 'wrong inject lane: %r' % (label,)
    assert lines == ['!CALL roll_action_dice Game.2043 101 102'], \
        'H1 RED: constructed call is not the select form: %r' % (lines,)
    assert seat._rolled_pids == {'101', '102'}, \
        'the per-turn roll ledger must record the subset: %r' \
        % (seat._rolled_pids,)


@test
def test_h1_roll_bad_arg_refuses_never_whole_zone():
    """Loud-failure rule: `roll die=101` (any non-pid arg) must REFUSE
    with the usage - the old parser silently dropped junk args and
    degraded to a WHOLE-ZONE roll (the exact re-roll the driver then
    refuses, BL-39-5 oversize + phantom-eye restore)."""
    seat, _ = make_seat(_NO_EYES)
    try:
        seat.t_roll(['die=101'])
    except hg.GestureRefused as e:
        assert '101' in str(e) or 'pid' in str(e).lower(), \
            'the refusal must name the bad arg / usage: %r' % (e,)
    else:
        raise AssertionError(
            'H1 RED: non-pid roll arg was swallowed silently '
            '(injected: %r) - expected GestureRefused'
            % (seat.run.ctx.injected,))


# ---------------------------------------------------------------------------
# H2: raw accepts the client directive lane; state hints the routes
# ---------------------------------------------------------------------------

@test
def test_h2_raw_accepts_client_directive_lines():
    """par5_g3 T6 + par5_g4 T4: `!CALL ...` lines ARE executable
    client-side (AiOpp execDirective, PROTOCOL.md directive lane) -
    the raw verb must pass them through, not refuse them as pretty
    text (both wedges were closed by hand-written inject files the
    verb refused to send)."""
    seat, _ = make_seat({})
    try:
        res = seat.t_raw(['!CALL', 'wwsat'])
    except hg.GestureRefused as e:
        raise AssertionError(
            'H2 RED: raw refused the client directive line: %r' % (e,))
    assert res.get('injected') == ['!CALL wwsat'], \
        'directive line must inject verbatim: %r' % (res,)
    assert res.get('feed_grew') is True, \
        'the executed directive grew the (fake) feed: %r' % (res,)


@test
def test_h2_raw_still_refuses_pretty_board_text():
    """Negative control (BL-139(a) stands): untagged pretty board text
    executes NOTHING client-side and stays refused."""
    seat, _ = make_seat({})
    try:
        seat.t_raw(['$Wey kills Saruman in Orthanc.'])
    except hg.GestureRefused:
        pass
    else:
        raise AssertionError('pretty board text must stay refused')


@test
def test_h2_state_hints_the_card_pick_route():
    """par5_g4 T4: waiting_on `client card_pick` with a null menu gave
    the tester NO route.  `state` must carry a hint naming the wwsat
    directive lane so the route is on the surface, not in lore."""
    seat, _ = make_seat({
        'human_menu': {},
        'waiting_on': "client card_pick ({'ctx': 'WWSAT hand discard'})",
        'notes': [],
    })
    out = seat.t_state([])
    assert 'hint' in out, \
        'H2 RED: no route hint published for the card_pick wait state'
    assert 'wwsat' in str(out['hint']), \
        'the hint must name the wwsat directive: %r' % (out['hint'],)


@test
def test_h1_state_hints_the_subset_roll():
    """par5_g3 T6: the subset-roll demand names the pids - the state
    hint must translate it into the seat verb form (`roll 101`)."""
    seat, _ = make_seat({
        'human_menu': {},
        'waiting_on': 'waiting on: YOU to roll ONLY die 1 '
                      '(select 101, then roll)',
        'notes': [],
    })
    out = seat.t_state([])
    assert 'hint' in out, \
        'H1 RED: no route hint published for the subset-roll demand'
    assert 'roll 101' in str(out['hint']), \
        'the hint must spell the verb form: %r' % (out['hint'],)


# ---------------------------------------------------------------------------
# H3: reset_resync writes the panel's control keys (file IPC)
# ---------------------------------------------------------------------------

@test
def test_h3_control_write_merges_and_clear_expires():
    """The control-file write itself: merge-write (never clobber the
    other keys - ai_panel._write_control contract) and one-shot
    expiry (ai_panel.clear_control_keys contract)."""
    w = getattr(agent_seat, '_panel_control_write', None)
    c = getattr(agent_seat, '_panel_control_clear', None)
    assert w is not None and c is not None, \
        'H3 RED: agent_seat has no panel control-file helpers'
    tmpdir = tempfile.mkdtemp(prefix='flh_pin_')
    path = os.path.join(tmpdir, 'ai_panel_control.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'enforce_human': True, 'sims_override': 300}, f)
    w(path, reset_resync=1.5, clear_error=1.5)
    cur = json.load(open(path, encoding='utf-8'))
    assert cur == {'enforce_human': True, 'sims_override': 300,
                   'reset_resync': 1.5, 'clear_error': 1.5}, \
        'merge-write broke the other keys: %r' % (cur,)
    gone = c(path, 'reset_resync', 'clear_error')
    assert sorted(gone) == ['clear_error', 'reset_resync'], \
        'clear must report what it expired: %r' % (gone,)
    cur = json.load(open(path, encoding='utf-8'))
    assert cur == {'enforce_human': True, 'sims_override': 300}, \
        'one-shot expiry clobbered the persistent keys: %r' % (cur,)


@test
def test_h3_reset_resync_verb_writes_panel_keys():
    """par5_g4 D1: the seat verb must write EXACTLY the keys the
    panel's RESET+RESYNC button writes (ai_panel.full_reset:
    reset_resync=now, clear_error=now), see the server consume it,
    then expire the one-shots and keep every other key."""
    assert hasattr(agent_seat.Seat, 't_reset_resync'), \
        'H3 RED: the seat has no reset_resync verb (GUI-only reset)'
    seat, tmpdir = make_seat({
        'notes': ['panel: RESET + RESYNC requested'],
    })
    path = os.path.join(tmpdir, 'ai_panel_control.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'enforce_human': True}, f)
    res = seat.t_reset_resync(['3'])
    assert sorted(res.get('wrote') or {}) \
        == ['clear_error', 'reset_resync'], \
        'the verb must write the panel button\'s keys: %r' % (res,)
    w = res['wrote']
    assert isinstance(w['reset_resync'], float) \
        and w['reset_resync'] == w['clear_error'], \
        'both one-shots carry the SAME epoch stamp: %r' % (w,)
    assert res.get('control_path') == path, \
        'the write must target the SESSION client dir: %r' % (res,)
    assert 'RESET + RESYNC' in str(res.get('consumed_note')), \
        'the server consumption note must be reported: %r' % (res,)
    cur = json.load(open(path, encoding='utf-8'))
    assert cur == {'enforce_human': True}, \
        'after expiry only the persistent keys remain: %r' % (cur,)
    # wiring: the verb is reachable through the do-lane verb table
    assert "'reset_resync'" in inspect.getsource(agent_seat.Seat.serve), \
        'H3 RED: reset_resync is not wired into the serve verb table'


# ---------------------------------------------------------------------------
# tool doc: the routes are on the surface for future testers
# ---------------------------------------------------------------------------

@test
def test_tool_doc_names_the_routes():
    """The header docstring IS the tester's tool manual - it must name
    the subset roll form, the reset_resync verb and the wwsat/raw
    directive route (both par5 wedges were routes that EXISTED but
    were absent from the doc)."""
    doc = agent_seat.__doc__ or ''
    assert 'reset_resync' in doc, \
        'DOC RED: reset_resync route not documented'
    assert 'wwsat' in doc, \
        'DOC RED: the WWSAT card_pick route not documented'
    assert 'roll [pid' in doc or 'roll 101' in doc, \
        'DOC RED: the subset roll form not documented'


def main():
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print('[PASS] %s' % fn.__name__)
        except Exception:
            fails += 1
            print('[FAIL] %s' % fn.__name__)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
