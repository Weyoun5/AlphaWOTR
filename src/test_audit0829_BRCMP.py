"""Audit 2026-08-29 - BRCMP fix-wave pins (comparator + adoption + the
census / reconcile / witness lane).

Every pin builds the OFFLINE rigs (test_live_driver.Rig, the offline
LiveServer) over the REAL 08-21 client dump fixture (made consistent
with the engine the way test_live_adopt does) or a signed real
session capture, and went RED against the unfixed tree:

  BRCMP-1   QUIESCENCE GATE: a half-done human gesture (die staged,
            pieces dragged, die not killed) is never adopted; the
            census publishes WHY it is held; the kill then commits
  BRCMP-2   a STAGED die (client Current Action = Game.2035) is not a
            spent die: no dice row, no 'N -> N' adoption, no flap
            alarm, client_divergences 0
  BRCMP-3/9 the witness skips states the driver KNOWS are owed (echo
            ledger) and its escalation alarm RETRACTS on the heal;
            census_aging is not divergence-class
  BRCMP-4/7/8 adoption = ONE transaction with coherence: an unseated
            garrison never drops the siege on the auto path, the
            casualty ledger is never lowered on the dead pile alone
            (fpgame15's dead Elf stays dead), tracks are censused and
            report-only unless the human asked, every leg rolls back
            together; the bonus-die shield + PIECE CONSERVATION run
            on the live path
  BRCMP-5   a dead companion parked in the client's Card-Play zone is
            not a divergence (shgame19); an in-Fellowship one is
  BRCMP-6/10 REPORT-ONLY BY DEFAULT: rows print with reasons; a
            divergence identical across quiescent reads for >= K s
            LOG -> ASKs; 'yes' adopts transactionally, 'no' keeps the
            engine (no re-ask on the same rows); a transient never
            asks; client_divergences counts quiescent rows only
"""
import os
import sys
import json
import pickle
import tempfile
import time

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
FIX = os.path.join(_LAB, 'fixtures', 'sessions')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import ai                                                  # noqa: E402
import game_state as gs                                    # noqa: E402
import game_logic as gl                                    # noqa: E402
import log_replay as lr                                    # noqa: E402
import emitter                                             # noqa: E402
from game_state import Side                                # noqa: E402
from board import sanitize                                 # noqa: E402
from wotr_bridge.live import client_board as cb            # noqa: E402
from wotr_bridge.live import adopt                         # noqa: E402
from wotr_bridge.live import taxonomy                      # noqa: E402
from test_live_driver import (Rig, adopted_state, fresh_state,  # noqa: E402
                              _army_window_lines)
from test_live_witnesses import build_dump_from_state      # noqa: E402

FIXTURE = os.path.join(_ROOT, 'data', 'live_fixtures',
                       'client_board_20260821_strider_guide.txt')


# ------------------------------------------------------------ dump utils
def load_rows(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            rows.append([parts[0], parts[1], parts[2].split(),
                         parts[3].split() if len(parts) > 3 else []])
    return rows


def write_rows(rows, path, header=None):
    """`header`: an optional '#SEQ ...' line 1 (the candidate jar's
    dump header, client_protocol)."""
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header.rstrip('\n') + '\n')
        for idx, code, pids, flags in rows:
            fl = list(flags) + ['-'] * (len(pids) - len(flags))
            f.write('%s\t%s\t%s\t%s\n' % (idx, code, ' '.join(pids),
                                          ' '.join(fl[:len(pids)])))


def code_for(name):
    """Zone code for a decoded client zone name (the dump reader's own
    tables, so a pin never hand-types a code)."""
    for tbl in (lr.LOCATION_CODES, lr._AUTH_CODES):
        for k, v in tbl.items():
            if str(v) == name:
                return k
    raise LookupError(name)


def move_pid(rows, pid, to_code):
    pid = str(pid)
    for r in rows:
        if pid in r[2]:
            i = r[2].index(pid)
            r[2].pop(i)
            if i < len(r[3]):
                r[3].pop(i)
    want = '<~%s~>' % to_code
    for r in rows:
        if r[1] == want:
            r[2].append(pid)
            r[3].append('-')
            return
    rows.append(['999', want, [pid], ['-']])


def drop_pid(rows, pid):
    pid = str(pid)
    for r in rows:
        if pid in r[2]:
            i = r[2].index(pid)
            r[2].pop(i)
            if i < len(r[3]):
                r[3].pop(i)


def census(st, dump):
    return cb.diff_engine_vs_dump(st, dump, ai_module=ai) or []


def zero_baseline():
    """Engine state made consistent with the real 08-21 fixture (dice
    all in their pools, decks aligned, tracks/politics aligned) so the
    pair censuses to ZERO rows.  Returns (state, rows, dump_path)."""
    st = adopted_state()
    rows = load_rows(FIXTURE)
    td = tempfile.mkdtemp(prefix='brcmp_')
    path = os.path.join(td, 'ai_client_board.txt')
    for p in ('099', '100', '101', '102', '103', '104', '105'):
        move_pid(rows, p, 'Game.2043')
    for p in ('210', '211', '212', '213'):
        move_pid(rows, p, 'Game.2042')
    write_rows(rows, path)
    dump = cb.read_board(path)
    adopt.adopt_client_board(st, dump, ai_module=ai)
    # tracks / politics: engine := client (the fixture is the truth here)
    adopt.adopt_tracks(st, dump)
    adopt.adopt_political(st, dump, cb.read_board_flags(path))
    for r in census(st, dump):
        if r.startswith('deck '):
            name = r.split(':')[0][5:]
            attr = {'fp char deck': 'fp_char_deck',
                    'fp strat deck': 'fp_strat_deck',
                    'sa char deck': 'sh_char_deck',
                    'sa strat deck': 'sh_strat_deck'}[name]
            want = int(r.split('client=')[1].split()[0])
            lst = getattr(st, attr)
            while len(lst) > want:
                lst.pop()
            while len(lst) < want:
                lst.append('HU_PAD_%d' % len(lst))
    res = census(st, dump)
    assert res == [], 'zero baseline not zero: %r' % (res,)
    return st, rows, path


def _army_counts(es, *regions):
    out = []
    for reg in regions:
        a = next((v for k, v in es.armies.items()
                  if sanitize(str(k)) == sanitize(reg)), None)
        out.append(len(a.units) if a is not None else 0)
    return tuple(out)


def _move_pieces_in_dump(rows, st, action):
    """The action's units, moved in the DUMP only: returns
    (frm, to, kind, moved_pids)."""
    p = action.params
    frm, to = sanitize(str(p['from'])), sanitize(str(p['to']))
    n = int(p.get('n_regular') or 0) + int(p.get('n_elite') or 0)
    kind = 'elite' if p.get('n_elite') else 'regular'
    frm_code = emitter.zone_code_for(frm)
    to_code = emitter.zone_code_for(to)
    moved = []
    for r in rows:
        if r[1] == '<~%s~>' % frm_code:
            for pid in list(r[2]):
                try:
                    _nat, ut = cb.unit_type(pid)
                except Exception:
                    continue
                if ut == kind and len(moved) < n:
                    moved.append(pid)
    for pid in moved:
        move_pid(rows, pid, to_code)
    return frm, to, kind, moved, frm_code, to_code


def _notes(items):
    return [str(i[1]) for i in items if i and i[0] == 'note']


def _kinds(drv):
    return [k for k, _ in drv.findings]


def _server_rig(seed=99):
    """Offline LiveServer (AI = SHADOW, human = FP) over the zero
    baseline; returns (srv, drv, rows, dump_path)."""
    from wotr_bridge.live.live_server import LiveServer
    st, rows, path = zero_baseline()
    td = os.path.dirname(path)
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv = LiveServer(td, Side.SHADOW, load_nets=False, seed=seed)
    drv = srv.driver
    drv.es = st
    for hm in (getattr(drv, 'hand_models', {}) or {}).values():
        hm.es = st
    drv.hands.es = st
    drv.view.dump_path = path
    assert drv.view.refresh_dump()
    return srv, drv, rows, path


def _status(srv):
    srv._mirror_last = (None, 0.0)
    srv._write_mirror_snapshot()
    with open(os.path.join(srv.client_dir, 'ai_live_status.json'),
              encoding='utf-8') as f:
        return json.load(f)


def _cadence(srv):
    """One witness cadence, its items dispatched the way the tick does
    (held notes / reasoned rows / an ASK ride the driver items)."""
    out = []
    for it in srv._witness_cadence() or []:
        out.extend(srv._dispatch(it))
    return out


# ----------------------------------------------------------- BRCMP-1
def test_brcmp1_open_window_is_never_adopted():
    st, rows, path = zero_baseline()
    a, lines = _army_window_lines(st)
    assert a is not None, 'precondition: no simple army_move'
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')     # human = SH
    drv = rig.drv
    drv.view.dump_path = path
    assert drv.view.refresh_dump()
    rig.feed(lines[0], lines[1])                          # stage + drag
    assert drv.seg.window is not None and not drv.commits
    move_pid(rows, '099', 'Game.2035')                    # die staged
    frm, to, kind, moved, frm_code, to_code = _move_pieces_in_dump(
        rows, st, a)
    assert moved, 'precondition: nothing moved in the dump'
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    before = _army_counts(drv.es, frm, to)
    drv._last_commit_ts = time.time() - 10
    out = []
    for _ in range(3):
        drv._reconcile_due = True
        out.extend(drv.reconcile_board_quiet())
    after = _army_counts(drv.es, frm, to)
    assert after == before, \
        'BRCMP-1 RED: auto-reconcile adopted a half-done gesture: ' \
        'engine units at (%s, %s) %s -> %s with %d commit(s); notes=%r' \
        % (frm, to, before, after, len(drv.commits), _notes(out))
    assert 'board_adopt' not in _kinds(drv), _kinds(drv)
    held = [n for n in _notes(out) if 'census held' in n]
    assert held and 'open window' in held[0], _notes(out)
    assert drv.quiescent() is False
    assert any('open window' in h for h in drv.quiescence_holds())
    # the human finishes the gesture: the kill commits it normally
    rig.feed(lines[2])
    assert drv.commits and drv.commits[-1][2] == 'window', \
        [c[2] for c in drv.commits]
    assert not drv.alarms, drv.alarms
    assert _army_counts(drv.es, frm, to) != before


# ----------------------------------------------------------- BRCMP-2
def test_brcmp2_staged_die_is_not_spent():
    st, rows, path = zero_baseline()
    # (a) the dump reader keeps the client's own name for Game.2035
    move_pid(rows, '099', 'Game.2035')
    write_rows(rows, path)
    dump = cb.read_board(path)
    assert dump['099'] == 'Current Action', \
        'BRCMP-2 RED: Game.2035 decodes to %r (a staged die reads as ' \
        'spent)' % (dump['099'],)
    dr = [r for r in census(st, dump) if r.startswith('dice ')]
    assert not dr, 'BRCMP-2 RED: a merely STAGED die produced dice ' \
                   'rows: %r' % (dr,)
    # (a2) the AI's spent-but-not-yet-killed die (engine used, client
    # staged) is the same in-flight picture: no row
    st.dice[Side.SHADOW][0].used = True
    dr = [r for r in census(st, dump) if r.startswith('dice ')]
    assert not dr, dr
    st.dice[Side.SHADOW][0].used = False
    # (b) SENSITIVITY: a die the client KILLED (SA Dead) while the
    # engine holds it unused is a row, and the row names pid + zone
    move_pid(rows, '099', code_for('SA Dead'))
    write_rows(rows, path)
    dump = cb.read_board(path)
    dr = [r for r in census(st, dump) if r.startswith('dice shadow')]
    assert dr, 'sensitivity: a killed die vs an unused engine die ' \
               'must row'
    assert any('099' in r and 'sa dead' in r.lower() for r in dr), dr
    # (c) driver: six quiet reads with the die staged inside an open
    # window -> no adoption, no 'N -> N' flap alarm
    st2, rows2, path2 = zero_baseline()
    a, lines = _army_window_lines(st2)
    rig = Rig(st2, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv.view.dump_path = path2
    assert drv.view.refresh_dump()
    rig.feed(lines[0])
    move_pid(rows2, '099', 'Game.2035')
    write_rows(rows2, path2)
    assert drv.view.refresh_dump()
    drv._last_commit_ts = time.time() - 10
    for _ in range(6):
        drv._reconcile_due = True
        drv.reconcile_board_quiet()
    assert 'board_adopt' not in _kinds(drv), _kinds(drv)
    assert taxonomy.partition(list(drv.alarms))[0] == [], \
        'BRCMP-2 RED: a staged die became a divergence alarm: %r' \
        % (drv.alarms,)
    # (d) server: the human FP's staged die (no window - a manual
    # stage) is not a census row and client_divergences stays 0
    srv, drv3, rows3, path3 = _server_rig()
    move_pid(rows3, '212', 'Game.2035')
    write_rows(rows3, path3)
    _cadence(srv)
    stt = _status(srv)
    assert stt['census_rows'] == 0, \
        'BRCMP-2 RED: staged FP die counted: census_rows=%r first=%r' \
        % (stt['census_rows'], stt.get('census_first'))
    assert stt['client_divergences'] == 0, stt['client_divergences']


# --------------------------------------------------------- BRCMP-3/9
def _recruit_north_at_carrock(drv):
    nat = gs.Nation.THE_NORTH
    gl.get_or_create_army(drv.es, 'carrock').units.append(
        gs.Unit(nation=nat, kind=gs.UnitKind.REGULAR))


def _heal_carrock(drv, rows, path):
    free = [p for p, z in drv.view.piece_region.items()
            if str(z).lower() == 'fp reinf'
            and cb.unit_type(p)[1] == 'regular'
            and str(getattr(cb.unit_type(p)[0], 'value', '')).lower()
            == 'the north']
    move_pid(rows, free[0], emitter.zone_code_for('carrock'))
    write_rows(rows, path)


def test_brcmp3_witness_skips_owed_and_retracts():
    srv, drv, rows, path = _server_rig()
    _recruit_north_at_carrock(drv)
    drv.echo_ledger.append({'kind': 'unit_in', 'side': 'fp',
                            'region': 'carrock',
                            'tkey': 'the north:regular', 'n': 1})
    for _ in range(3):
        _cadence(srv)
    assert not any('WITNESS ESCALATION' in a for a in drv.alarms), \
        'BRCMP-3 RED: the witness escalated an OWED render: %r' \
        % (drv.alarms,)
    held = [n for n in srv.notes if 'census held' in n]
    assert held and 'owed' in held[0], srv.notes[-4:]
    # the human drags the piece: heal
    _heal_carrock(drv, rows, path)
    drv.echo_ledger = []
    _cadence(srv)
    stt = _status(srv)
    assert stt['client_divergences'] == 0, \
        'BRCMP-3 RED: client_divergences=%r after the heal (%r)' \
        % (stt['client_divergences'], stt.get('engine_vs_client'))
    assert stt['census_rows'] == 0 and not stt.get('census_held'), stt
    # SENSITIVITY: the same missing piece with NO owed entry is a real
    # divergence - it escalates; the heal RETRACTS the alarm
    srv2, drv2, rows2, path2 = _server_rig(seed=7)
    _recruit_north_at_carrock(drv2)
    for _ in range(3):
        _cadence(srv2)
    assert any('WITNESS ESCALATION' in a for a in drv2.alarms), \
        'sensitivity: a genuine missing piece must escalate: %r' \
        % (drv2.alarms,)
    assert _status(srv2)['client_divergences'] >= 1
    _heal_carrock(drv2, rows2, path2)
    _cadence(srv2)
    assert not any('WITNESS ESCALATION' in a for a in drv2.alarms), \
        'BRCMP-9 RED: the escalation alarm did not retract: %r' \
        % (drv2.alarms,)
    assert _status(srv2)['client_divergences'] == 0
    assert taxonomy.classify('census_aging: gated x10') == 'error', \
        'BRCMP-9 RED: census_aging is divergence-class'


# ------------------------------------------------------- BRCMP-4/7/8
def test_brcmp4_unseated_garrison_keeps_siege_and_ledger():
    d = os.path.join(FIX, 'fpgame15')
    pk = os.path.join(d, 'ai_live_state.pkl')
    dp = os.path.join(d, 'ai_client_board.txt')
    assert os.path.exists(pk) and os.path.exists(dp), 'fpgame15 missing'
    with open(pk, 'rb') as f:
        st = pickle.load(f)
    dump = cb.read_board(dp)
    assert 'lorien' in st.besieged_strongholds
    cas0 = int(st.fp_casualties[gs.Nation.ELVES]['regular'])
    assert cas0 == 2, cas0
    rep, nb, na = adopt.adopt_client_board(st, dump, ai_module=ai)
    assert 'lorien' in st.besieged_strongholds, \
        'BRCMP-4 RED: the auto adoption DROPPED the Lorien siege ' \
        '(garrison merely unseated): %r' % (rep,)
    assert int(st.fp_casualties[gs.Nation.ELVES]['regular']) == cas0, \
        'BRCMP-4 RED: the casualty ledger was LOWERED on the dead ' \
        'pile alone (%d -> %d): %r' \
        % (cas0, st.fp_casualties[gs.Nation.ELVES]['regular'], rep)
    assert any('seat' in r.lower() and 'lorien' in r.lower()
               for r in rep), rep
    # the MANUAL lever (BL-147: client AS IS) still drops it, loudly
    with open(pk, 'rb') as f:
        st2 = pickle.load(f)
    rep2, _, _ = adopt.adopt_client_board(
        st2, dump, ai_module=ai, manual=True,
        human_side=Side.FREE_PEOPLES)
    assert 'lorien' not in st2.besieged_strongholds, rep2
    assert any('siege dropped' in r for r in rep2), rep2


def test_brcmp7_tracks_censused_and_transactional():
    st, rows, path = zero_baseline()
    move_pid(rows, '101', code_for('SA Dead'))       # a staging row
    write_rows(rows, path)
    dump = cb.read_board(path)
    p0, c0 = st.fellowship.progress, st.fellowship.corruption
    st.fellowship.progress = p0 + 2
    st.fellowship.corruption = c0 + 3
    rows_c = census(st, dump)
    assert any(r.startswith('track') and 'corruption' in r
               for r in rows_c), \
        'BRCMP-7 RED: corruption disagreement invisible to the ' \
        'census: %r' % (rows_c,)
    assert any(r.startswith('track') and 'progress' in r
               for r in rows_c), rows_c
    # the plain adoption (nobody asked for tracks) leaves them alone
    rep, nb, na = adopt.adopt_client_board(st, dump, ai_module=ai)
    assert (st.fellowship.progress, st.fellowship.corruption) \
        == (p0 + 2, c0 + 3), \
        'BRCMP-7 RED: adoption reset the tracks (%r): %r' \
        % ((st.fellowship.progress, st.fellowship.corruption), rep)
    # the MANUAL lever adopts them (client as-is)
    st_m, _rows_m, _path_m = zero_baseline()
    st_m.fellowship.progress = p0 + 2
    st_m.fellowship.corruption = c0 + 3
    adopt.adopt_client_board(st_m, dump, ai_module=ai, manual=True,
                             human_side=Side.FREE_PEOPLES)
    assert (st_m.fellowship.progress, st_m.fellowship.corruption) \
        == (p0, c0), (st_m.fellowship.progress, st_m.fellowship.corruption)
    # ONE TRANSACTION: a leg failing mid-way rolls EVERY leg back
    st_t, _rows_t, _path_t = zero_baseline()
    st_t.fellowship.progress = p0 + 2
    st_t.fellowship.corruption = c0 + 3
    orig = adopt.reconcile_fp_casualties

    def _boom(*a, **k):
        raise RuntimeError('leg exploded')
    adopt.reconcile_fp_casualties = _boom
    try:
        rep_t, _, _ = adopt.adopt_client_board(
            st_t, dump, ai_module=ai, manual=True,
            human_side=Side.FREE_PEOPLES)
    finally:
        adopt.reconcile_fp_casualties = orig
    assert (st_t.fellowship.progress, st_t.fellowship.corruption) \
        == (p0 + 2, c0 + 3), \
        'BRCMP-7 RED: a leg failed but the tracks adopted earlier ' \
        'stayed written (%r): %r' \
        % ((st_t.fellowship.progress, st_t.fellowship.corruption), rep_t)
    assert any('rolled back' in r for r in rep_t), rep_t


def test_brcmp8_live_shields_run():
    st, rows, path = zero_baseline()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv.view.dump_path = path
    assert drv.view.refresh_dump()
    n_sh = len(drv.es.dice[Side.SHADOW])
    assert n_sh == 7, n_sh
    # a NEVER-GRANTED bonus die parked in SA Dead is not a spent die
    move_pid(rows, '360', code_for('SA Dead'))
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    board, staging = drv.census_rows()
    assert not any(r.startswith('dice shadow') for r in board + staging), \
        'BRCMP-8 RED: a never-granted bonus die counted as used: %r' \
        % (board + staging,)
    # PIECE CONSERVATION: a unit pid that VANISHES from the dump after
    # the boot capture is a row naming it
    victim = next(p for p, z in sorted(drv.view.piece_region.items())
                  if sanitize(str(z)) == 'orthanc'
                  and cb.unit_type(p)[1] in ('regular', 'elite'))
    drop_pid(rows, victim)
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    board, staging = drv.census_rows()
    assert any('CONSERVATION' in r and victim in r for r in board), \
        'BRCMP-8 RED: no CONSERVATION row for vanished pid %s: %r' \
        % (victim, board)


# ----------------------------------------------------------- BRCMP-5
def test_brcmp5_dead_companion_in_card_play_is_not_a_divergence():
    d = os.path.join(FIX, 'shgame19')
    pk = os.path.join(d, 'ai_live_state.pkl')
    dp = os.path.join(d, 'ai_client_board.txt')
    assert os.path.exists(pk) and os.path.exists(dp), 'shgame19 missing'
    with open(pk, 'rb') as f:
        st = pickle.load(f)
    dump = cb.read_board(dp)
    assert dump.get('244') == 'SA Card Play', dump.get('244')
    assert 'Boromir' not in [c.name for c in st.fellowship.companions]
    rows = [r for r in census(st, dump) if r.startswith('companion')]
    assert not rows, \
        'BRCMP-5 RED: a dead companion parked in Card Play rows: %r' \
        % (rows,)
    # SPECIFICITY: a companion the engine holds IN the Fellowship,
    # parked by the client in Card Play, still rows
    st2 = fresh_state()
    dump2 = build_dump_from_state(st2)
    dump2['244'] = 'SA Card Play'
    rows2 = [r for r in census(st2, dump2) if r.startswith('companion')]
    assert rows2 and 'Boromir' in rows2[0] and 'in fellowship' in rows2[0], \
        rows2


# -------------------------------------------------------- BRCMP-6/10
def _diverged_rig():
    """Zero baseline + a BOARD divergence with nothing owed and no
    window: the client shows the elite(s) of a legal move already at
    the destination (a die-less manual drag)."""
    st, rows, path = zero_baseline()
    a, lines = _army_window_lines(st)
    assert a is not None
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv.view.dump_path = path
    assert drv.view.refresh_dump()
    frm, to, kind, moved, frm_code, to_code = _move_pieces_in_dump(
        rows, st, a)
    assert moved
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    drv._last_commit_ts = time.time() - 10
    info = {'frm': frm, 'to': to, 'kind': kind, 'moved': moved,
            'frm_code': frm_code, 'to_code': to_code}
    return rig, drv, rows, path, info


def _read(drv):
    drv._reconcile_due = True
    return drv.reconcile_board_quiet()


def _age_streak(drv, seconds=30):
    drv._recon_sig_since = float(drv._recon_sig_since) - seconds


def test_brcmp6_confirmed_divergence_asks_then_yes_adopts():
    rig, drv, rows, path, info = _diverged_rig()
    frm, to = info['frm'], info['to']
    before = _army_counts(drv.es, frm, to)
    out1 = _read(drv)
    assert _army_counts(drv.es, frm, to) == before
    assert drv._repair is None and 'board_adopt' not in _kinds(drv)
    row_notes = [n for n in _notes(out1)
                 if n.startswith('units') and 'engine=' in n
                 and 'client=' in n and 'pids=' in n
                 and '| owed:' in n and '| since' in n]
    assert row_notes, \
        'BRCMP-10 RED: no row printed with its reason: %r' \
        % (_notes(out1),)
    out2 = _read(drv)                      # within K: still nothing
    assert _army_counts(drv.es, frm, to) == before, \
        'BRCMP-6 RED: the second quiet read adopted the board: %r' \
        % (_notes(out2),)
    assert drv._repair is None, drv._repair
    _age_streak(drv)
    out3 = _read(drv)
    assert _army_counts(drv.es, frm, to) == before
    r = drv._repair
    assert r is not None and r.get('kind') == 'census' \
        and r.get('phase') == 'offered', \
        'BRCMP-6 RED: no ASK after the confirmation window: %r' % (r,)
    assert 'census_adopt_offered' in _kinds(drv), _kinds(drv)
    assert 'census_divergence_confirmed' in _kinds(drv), _kinds(drv)
    assert 'stall_log_written' in _kinds(drv), _kinds(drv)
    assert any(a.startswith('BOARD DIVERGENCE') for a in drv.alarms), \
        drv.alarms
    # 'yes' -> the transactional adoption, nothing else
    rig.feed('<Wey> yes')
    assert drv._repair is None
    assert 'census_adopt_consented' in _kinds(drv), _kinds(drv)
    assert 'board_adopt' in _kinds(drv), _kinds(drv)
    after = _army_counts(drv.es, frm, to)
    assert after != before and after[1] == before[1] + len(info['moved']), \
        (before, after)
    # the next quiet read is clean: streak + alarm retracted
    out4 = _read(drv)
    assert drv._reconcile_due is False, _notes(out4)
    assert not any(a.startswith('BOARD DIVERGENCE') for a in drv.alarms), \
        drv.alarms


def test_brcmp6_no_keeps_engine_and_never_reasks():
    rig, drv, rows, path, info = _diverged_rig()
    frm, to = info['frm'], info['to']
    before = _army_counts(drv.es, frm, to)
    _read(drv)
    _age_streak(drv)
    _read(drv)
    assert drv._repair is not None and drv._repair.get('kind') == 'census'
    rig.feed('<Wey> no')
    assert drv._repair is None
    out = _read(drv)
    assert _army_counts(drv.es, frm, to) == before
    assert 'census_adopt_declined' in _kinds(drv), _kinds(drv)
    assert any('engine kept' in n for n in _notes(out)), _notes(out)
    # the same rows, aged again: no second ask
    _age_streak(drv)
    _read(drv)
    assert drv._repair is None
    assert _kinds(drv).count('census_adopt_offered') == 1, _kinds(drv)
    assert _army_counts(drv.es, frm, to) == before
    # the rows CHANGE (the human moves a second piece): a new ask may
    # be made once confirmed
    extra = next(p for p, z in sorted(drv.view.piece_region.items())
                 if sanitize(str(z)) == frm
                 and cb.unit_type(p)[1] == 'regular')
    move_pid(rows, extra, info['to_code'])
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    _read(drv)
    _age_streak(drv)
    _read(drv)
    assert _kinds(drv).count('census_adopt_offered') == 2, _kinds(drv)


def test_brcmp6_transient_divergence_never_asks():
    rig, drv, rows, path, info = _diverged_rig()
    frm, to = info['frm'], info['to']
    before = _army_counts(drv.es, frm, to)
    _read(drv)
    # the client heals before K
    for pid in info['moved']:
        move_pid(rows, pid, info['frm_code'])
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    _age_streak(drv)
    out = _read(drv)
    assert drv._repair is None, drv._repair
    assert 'census_divergence_confirmed' not in _kinds(drv), _kinds(drv)
    assert 'census_adopt_offered' not in _kinds(drv), _kinds(drv)
    assert drv._reconcile_due is False, _notes(out)
    assert _army_counts(drv.es, frm, to) == before
    assert not drv.alarms, drv.alarms


def test_brcmp10_counter_is_the_quiescent_census():
    srv, drv, rows, path = _server_rig()
    # an alarm alone (an event) is a stop signal, not a census row
    drv.alarms.append('WITNESS ESCALATION: 3 row(s) settled over 3 '
                      'checks (first: x)')
    stt = _status(srv)
    assert stt['client_divergences'] == 0, \
        'BRCMP-10 RED: client_divergences counts alarms: %r' \
        % (stt['client_divergences'],)
    assert stt.get('stop_signals') == 1, stt.get('stop_signals')
    drv.alarms[:] = []
    # a quiescent read with BOARD rows (the missing recruit and its
    # honest PHANTOM-pool twin: the client's reinforcement box holds
    # one North regular more than the engine allows) -> the count IS
    # the row count, the first row names the region
    _recruit_north_at_carrock(drv)
    _cadence(srv)
    stt = _status(srv)
    lq = drv._last_quiescent_census
    assert lq and lq['board'], lq
    assert stt['client_divergences'] == len(lq['board']) >= 1, stt
    assert stt['census_rows'] == stt['client_divergences'] \
        and 'carrock' in str(stt['census_first']), stt
    assert stt['stop_signals'] == 0 and stt['census_held'] is None, stt


# ------------------------------------------------- BRJAVA-2 / BRJAVA-7
# fixture text from test_audit0829_BRJAVA.py (the candidate jar's
# '#SEQ' dump header shape); feed_paragraphs is set per pin so the
# dump sits behind / ahead of / at the feed lines the driver consumed
_SEQ_HDR = ('#SEQ dump=12 feed_paragraphs=%d raw_lines=109 '
            'ts=1756500000123 session=%d')
_SESSION = 1756499000000
_OTHER_SESSION = 1756490000000


def _seq_rig():
    st, rows, path = zero_baseline()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv.view.dump_path = path
    rig.feed('<Wey> hello')                 # one consumed feed line
    n = len(drv.feed.lines)
    assert n >= 1, n
    return st, rows, path, rig, drv, n


def test_brjava2_dump_not_at_feed_position_is_held_not_judged():
    from wotr_bridge.live import client_protocol as cp
    st, rows, path, rig, drv, n = _seq_rig()
    at = n - cp.FEED_HEADER_LINES                 # delta == 0
    # BEHIND the consumed feed: held, never a row, quiescent() False
    write_rows(rows, path, header=_SEQ_HDR % (at - 1, _SESSION))
    assert drv.view.refresh_dump()
    holds = drv.quiescence_holds()
    assert any('dump not at feed position' in h and 'behind' in h
               for h in holds), holds
    assert drv.quiescent() is False
    drv._reconcile_due = True
    out = drv.reconcile_board_quiet()
    assert any('census held' in x and 'dump not at feed position' in x
               for x in _notes(out)), _notes(out)
    assert not any(x.startswith(('units', 'census:')) for x in _notes(out))
    assert drv._reconcile_due is True
    # AHEAD of the consumed feed: held too
    write_rows(rows, path, header=_SEQ_HDR % (at + 2, _SESSION))
    assert drv.view.refresh_dump()
    holds = drv.quiescence_holds()
    assert any('dump not at feed position' in h and 'ahead' in h
               for h in holds), holds
    # AT position: judged (a clean quiescent read clears the flag)
    write_rows(rows, path, header=_SEQ_HDR % (at, _SESSION))
    assert drv.view.refresh_dump()
    assert not any('feed position' in h for h in drv.quiescence_holds())
    drv._reconcile_due = True
    drv.reconcile_board_quiet()
    assert drv._reconcile_due is False
    # header-less dump (the shipped jar): today's behaviour, judged
    write_rows(rows, path)
    assert drv.view.refresh_dump()
    assert not any('feed position' in h for h in drv.quiescence_holds())


def test_brjava7_cross_session_dump_halts_without_adoption():
    from wotr_bridge.live import client_protocol as cp
    from wotr_bridge.live.driver import DriverHalt
    st, rows, path, rig, drv, n = _seq_rig()
    at = n - cp.FEED_HEADER_LINES
    a, lines = _army_window_lines(st)
    # session 0 (a boot dump before any live loop): accepted, no nonce
    write_rows(rows, path, header=_SEQ_HDR % (at, 0))
    assert drv.view.refresh_dump()
    drv._reconcile_due = True
    drv.reconcile_board_quiet()
    assert getattr(drv, 'session_nonce', None) is None
    # the first LIVE header fixes the nonce
    write_rows(rows, path, header=_SEQ_HDR % (at, _SESSION))
    assert drv.view.refresh_dump()
    drv._reconcile_due = True
    drv.reconcile_board_quiet()
    assert drv.session_nonce == _SESSION, drv.session_nonce
    # a diverging dump from ANOTHER session: loud halt, no adoption
    frm, to, kind, moved, frm_code, to_code = _move_pieces_in_dump(
        rows, st, a)
    write_rows(rows, path, header=_SEQ_HDR % (at, _OTHER_SESSION))
    assert drv.view.refresh_dump()
    before = _army_counts(drv.es, frm, to)
    drv._reconcile_due = True
    halted = None
    try:
        drv.reconcile_board_quiet()
    except DriverHalt as e:
        halted = str(e)
    assert halted and str(_OTHER_SESSION) in halted \
        and str(_SESSION) in halted, \
        'BRJAVA-7 RED: a cross-session dump was judged: %r' % (halted,)
    assert _army_counts(drv.es, frm, to) == before
    assert 'client_session_mismatch' in _kinds(drv), _kinds(drv)
    assert 'board_adopt' not in _kinds(drv)
    assert drv._repair is None
    assert taxonomy.classify('client_session_mismatch: x') == 'divergence'


def main():
    tests = [test_brcmp1_open_window_is_never_adopted,
             test_brcmp2_staged_die_is_not_spent,
             test_brcmp3_witness_skips_owed_and_retracts,
             test_brcmp4_unseated_garrison_keeps_siege_and_ledger,
             test_brcmp7_tracks_censused_and_transactional,
             test_brcmp8_live_shields_run,
             test_brcmp5_dead_companion_in_card_play_is_not_a_divergence,
             test_brcmp6_confirmed_divergence_asks_then_yes_adopts,
             test_brcmp6_no_keeps_engine_and_never_reasks,
             test_brcmp6_transient_divergence_never_asks,
             test_brcmp10_counter_is_the_quiescent_census,
             test_brjava2_dump_not_at_feed_position_is_held_not_judged,
             test_brjava7_cross_session_dump_halts_without_adoption]
    n_fail = 0
    for t in tests:
        try:
            t()
            print('  [PASS] %s' % t.__name__)
        except Exception as e:
            n_fail += 1
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            print('  [FAIL] %s: %s: %s' % (t.__name__, type(e).__name__,
                                           str(e)[:400]))
            print('         at %s' % tb[-3].strip()[:200])
    print('=' * 60)
    print('TOTAL=%d FAIL=%d' % (len(tests), n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
