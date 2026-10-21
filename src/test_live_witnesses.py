"""LIFT-B instruments: wotr_bridge/live/client_board.py (board-dump
reader + client-vs-engine witnesses) and wotr_bridge/live/undo_model.py
(client-native undo stack model + refusal press computation).

Each test states its MEASURED QUANTITY first (instrument protocol,
USER directive 2026-08-19).  Every zero ships with a proven non-zero;
sensitivity AND specificity controls on every witness.

Undo-model test material is REAL rendered-feed specimens, embedded
verbatim with provenance (WOTR_AI_training_lab/audit/blind_run/
feed_blinded.txt and .../sided_run/feed_G2_refuse.txt); the one
synthesized line (an 'undoes last action' press) follows the
rendered_replay._UNDO_RE grammar because the captured feeds contain no
undo press.  The board-dump acceptance leg reads the REAL 428-piece
dump specimen (client fact #14) when the lab is present.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'wotr_bridge'),
           os.path.join(_ROOT, 'wotr_bridge', 'live')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as board_mod
import game_logic
import game_state as gs
from game_state import Side

import client_board as cb
import undo_model as um

_LAB_DUMP = os.path.join(
    os.path.dirname(_ROOT), 'WOTR_AI_training_lab', 'audit',
    'blind_run', 'ai_client_board.txt')


def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def _nv(x):
    return str(getattr(x, 'value', x)).lower()


# REAL consecutive slice, feed_blinded.txt lines 104-114 (blind run,
# 2026-08-18): one die-stage move, chat, a card choose, its <auto>
# card-play trailer, chat, three system d6 rolls, two chats, one kill.
FEED_WINDOW = [
    '$WOTR_AI moves Shadow Action Die (C) from SA Dice to '
    'Current Action.',
    '<WOTR_AI> old forest road from northern rhovanion ',
    '$WOTR_AI chooses a card.[Shadow Strategy Card]',
    '<auto> moves Return of the Witch-King/Swarm of Bats from '
    'Shadow Hand (chosen) to Current SA Card.',
    '<WOTR_AI> nc ',
    '<game> Wey rolls  4 4 5 4 1 ',
    '<game> Wey rolls  6 ',
    '<game> Wey rolls  5 5 ',
    '<WOTR_AI_FP> 3 ',
    '<WOTR_AI_FP> 1 ',
    '$WOTR_AI kills North Regular in Old Forest Road.',
]

# REAL lines: feed_G2_refuse.txt:20 (human fellowship move),
# feed_G2_refuse.txt:17 (AI die-to-hunt emission),
# feed_G2_refuse.txt:9 (human deck draw),
# feed_blinded.txt:126 (human draw), feed_blinded.txt:18 (roll).
HUMAN_MOVE = ('$Franatleti moves The Fellowship from Rivendell to '
              'Fords of Bruinen.')
AI_EMISSION = ('$WOTR_AI_Shadow moves Shadow Action Die (E) from '
               'SA Dice to The Hunt For The Ring.')
HUMAN_DRAW = '$Wey draws Free Strategy Card.'
HUMAN_ROLL = '$Wey rolls P H M A H M .'
AUTO_CARD = ('<auto> moves Kindred of Glorfindel/Advantageous '
             'Position from Free Peoples Hand (chosen) to '
             'Current FP Card.')
# grammar-conformant synthesized press (rendered_replay._UNDO_RE);
# the captured feeds hold no real undo press
UNDO_PRESS = '$Wey undoes last action : kills North Regular in ' \
             'Old Forest Road.'


def test_verb_coverage(fails):
    """MEASURES: every op live_parser can emit carries a stack
    classification - grammar growth must break THIS line, never
    silently drop a verb (defect-9 discipline)."""
    missing = um.unclassified_ops()
    if missing:
        fails.append(('verb_coverage', 'unclassified ops %r' % missing))


def test_stack_after_real_fold(fails):
    """MEASURES: on the REAL feed window, stack_after models exactly
    the client's snapshot pushes (ay_0.k()/h()): the die-stage move
    and the kill push; the choose/rolls/chat push NOTHING; the <auto>
    card-play reveal is a snapshot of its own that FOLDS into the
    press of the entry pushed ABOVE it (client fact #1: ay_0.o() pops
    the newest snapshot, then every next-older '<auto>' line; D5
    measured on the real client: one press after choose + reveal
    retracted the reveal only, the older staged die stayed) - here
    the kill at 10 carries it."""
    entries = um.stack_after(FEED_WINDOW, 0)
    if len(entries) != 2:
        fails.append(('fold_entry_count', 'entries=%r' % (entries,)))
        return
    e0, e1 = entries
    if (e0.index, e0.verb, e0.folds, e0.modelled)             != (0, 'moves', (), True):
        fails.append(('fold_first_entry', repr(e0)))
    if (e1.index, e1.verb, e1.folds) != (10, 'kills', (3,)):
        fails.append(('fold_second_entry',
                      '<auto> at 3 not folded into the kill: %r'
                      % (e1,)))
    # the reveal still on TOP is its own entry (costs its own press)
    top = um.stack_after(FEED_WINDOW[:4], 0)
    if [(e.index, e.verb, e.modelled) for e in top]             != [(0, 'moves', True), (3, '<auto>', True)]:
        fails.append(('reveal_on_top_own_entry', repr(top)))


def test_stack_after_lifo_pop(fails):
    """MEASURES: an 'undoes' press pops the NEWEST entry first (LIFO,
    driven 7/7 on the shipped jar by streak/client_undo_probe.py),
    folded <auto> lines ride their entry, and popping past the
    modelled bottom is an unmodelled entry (uncountable stack)."""
    lines = FEED_WINDOW + [UNDO_PRESS]
    entries = um.stack_after(lines, 0)
    if [e.verb for e in entries] != ['moves']:
        fails.append(('lifo_pop_newest', 'entries=%r' % (entries,)))
    elif entries[0].folds != ():
        fails.append(('lifo_pop_took_folds', repr(entries[0])))
    entries = um.stack_after(lines + [UNDO_PRESS], 0)
    if entries:
        fails.append(('lifo_pop_both', 'entries=%r' % (entries,)))
    entries = um.stack_after(lines + [UNDO_PRESS, UNDO_PRESS], 0)
    if not (len(entries) == 1 and not entries[0].modelled):
        fails.append(('lifo_underflow_unmodelled',
                      'entries=%r' % (entries,)))


def test_presses_own_emission_eats_press(fails):
    """MEASURES (client fact #2): with the AI's OWN emission pushed
    after the human's gesture, the press count charges one extra press
    (the own line pops first); the bare window costs exactly one."""
    lines = [HUMAN_MOVE, AI_EMISSION]
    n = um.presses_to_revert(lines, (0, 1), own_nick='WOTR_AI_Shadow')
    if n != 2:
        fails.append(('own_emission_extra_press', 'n=%r want 2' % (n,)))
    n = um.presses_to_revert([HUMAN_MOVE], (0, 1),
                             own_nick='WOTR_AI_Shadow')
    if n != 1:
        fails.append(('bare_window_one_press', 'n=%r want 1' % (n,)))


def test_presses_ineligible_never_guessed(fails):
    """MEASURES: every window a press cannot honestly reach returns
    None - a HUMAN line pushed after the gesture (his later gestures
    may never be re-forged), a roll inside the window (client
    randomness), a reveal-less 'chooses' (the hidden combat commit:
    client hidden state no press may re-drive)."""
    n = um.presses_to_revert([HUMAN_MOVE, HUMAN_DRAW], (0, 1),
                             own_nick='WOTR_AI_Shadow')
    if n is not None:
        fails.append(('human_above_none', 'n=%r' % (n,)))
    n = um.presses_to_revert([HUMAN_ROLL], (0, 1))
    if n is not None:
        fails.append(('roll_window_none', 'n=%r' % (n,)))
    n = um.presses_to_revert([HUMAN_MOVE, FEED_WINDOW[2]], (0, 2))
    if n is not None:
        fails.append(('hidden_choose_none', 'n=%r' % (n,)))


def test_presses_client_true_details(fails):
    """MEASURES the client-true refinements: (1) a DRAW window is
    undo-ELIGIBLE at one press (fact #1: draws snapshot; the donor's
    _snap9 refused this window - KNOWN WRONG there, kept only for old
    pins); (2) an <auto> reveal above the window is the TOP snapshot:
    one press pops it, the window's own press follows (client fact
    #1, the D5 measurement); inside the window it rides the press of
    the human line pushed above it; (3) a human push above the window
    that the human himself undid nets to a reachable stack."""
    n = um.presses_to_revert([HUMAN_DRAW], (0, 1))
    if n != 1:
        fails.append(('draw_window_eligible', 'n=%r want 1' % (n,)))
    n = um.presses_to_revert([HUMAN_MOVE, AUTO_CARD], (0, 1))
    if n != 2:
        fails.append(('auto_on_top_own_press', 'n=%r want 2' % (n,)))
    n = um.presses_to_revert([AUTO_CARD, HUMAN_DRAW], (0, 2))
    if n != 1:
        fails.append(('auto_beneath_folds', 'n=%r want 1' % (n,)))
    n = um.presses_to_revert(
        [HUMAN_MOVE, HUMAN_DRAW, UNDO_PRESS], (0, 1))
    if n != 1:
        fails.append(('self_undone_push_nets_out', 'n=%r want 1' % (n,)))


def test_restore_watch_alarm_fires(fails):
    """MEASURES (finding C4/NEXT-1): the restore alarm CAN fire.
    Sensitivity: a watch fed a contradicting dump on every tick sets
    the sticky alarm at exhaustion, reporting both the expected-back
    miss and the STILL-AT hit.  The donor's inline watch never fired
    outside live play; this leg is the proof this one can."""
    rw = um.RestoreWatch(expect={'213': 'FP Dice'},
                         notat={'103': 'Used'}, ticks=2)
    bad = rw.verify({'213': 'Current Action', '103': 'Used'})
    if not (bad and len(bad) == 2 and rw.alarm is None and rw.armed):
        fails.append(('alarm_tick1', 'bad=%r alarm=%r armed=%r'
                      % (bad, rw.alarm, rw.armed)))
        return
    if 'STILL@' not in bad[1]:
        fails.append(('alarm_notat_row', 'bad=%r' % (bad,)))
    bad = rw.verify({'213': 'Current Action', '103': 'Used'})
    if not (bad and rw.alarm and '213' in rw.alarm and not rw.armed):
        fails.append(('alarm_fires', 'bad=%r alarm=%r' % (bad, rw.alarm)))
    rw.verify({'213': 'FP Dice'})
    if not rw.alarm:
        fails.append(('alarm_sticky', 'alarm cleared by a later dump'))


def test_restore_watch_specificity(fails):
    """MEASURES: a matching dump verifies (returns []), disarms the
    watch, raises NO alarm; stronghold sub-box names normalize to the
    region (donor _norm9w)."""
    rw = um.RestoreWatch(expect={'213': 'FP Dice',
                                 '005': 'Minas Tirith'},
                         notat={'103': 'Used'}, ticks=2)
    bad = rw.verify({'213': 'FP Dice',
                     '005': 'Minas Tirith Stronghold',
                     '103': 'SA Dice'})
    if bad != [] or rw.alarm is not None or rw.armed:
        fails.append(('watch_specificity', 'bad=%r alarm=%r armed=%r'
                      % (bad, rw.alarm, rw.armed)))


def test_restore_watch_track_wording(fails):
    """MEASURES (BL-132 wording miss, fpgame18 wave 26): the dump
    names the FSP zone 'track 1' where the rendered line said
    'fellowship track 1' - one zone, two spellings.  A restore that
    LANDED must verify (the live watch fired 'restore did not land
    (248@track 1 (expected fellowship track 1))' on a board that was
    fully restored).  SPECIFICITY: a genuinely wrong track number
    still fails."""
    rw = um.RestoreWatch(expect={'248': 'Fellowship Track 1'},
                         ticks=2)
    bad = rw.verify({'248': 'Track 1'})
    if bad != [] or rw.alarm is not None or rw.armed:
        fails.append(('track_wording_verifies',
                      'bad=%r alarm=%r armed=%r'
                      % (bad, rw.alarm, rw.armed)))
    rw2 = um.RestoreWatch(expect={'248': 'Fellowship Track 1'},
                          ticks=2)
    bad2 = rw2.verify({'248': 'Track 2'})
    if not bad2:
        fails.append(('track_wording_specificity',
                      'a wrong track number verified'))


def test_expectations_from_window(fails):
    """MEASURES: a rendered window builds pid-keyed expectations for
    NAMED figures (A17 aliases: Strider and Aragorn are both 241) and
    REPORTS fungible names as unverifiable instead of dropping them
    (the silent version of this gap is finding C4)."""
    lines = [
        # feed_blinded.txt:22 (real): named-figure move
        '$WOTR_AI moves Strider from Fellowship Of The Ring to '
        'Guide Of The Fellowship.',
        # feed_blinded.txt:26 (real): named-figure kill
        '$WOTR_AI kills Gandalf the Grey in Current Action.',
        # feed_blinded.txt:114 (real): fungible kill
        '$WOTR_AI kills North Regular in Old Forest Road.',
    ]
    expect, notat, unver = um.expectations_from_window(lines)
    if expect.get('241') != 'Fellowship Of The Ring':
        fails.append(('window_expect_named', 'expect=%r' % (expect,)))
    if notat.get('240') != 'Current Action':
        fails.append(('window_notat_named', 'notat=%r' % (notat,)))
    if not any('North Regular' in str(u[0]) for u in unver):
        fails.append(('window_unverifiable_reported',
                      'unver=%r' % (unver,)))
    if um.resolve_named('Aragorn') != '241' \
            or um.resolve_named('Strider') != '241':
        fails.append(('alias_resolution', 'Aragorn/Strider != 241'))


def _write_dump_file(path, dump):
    with open(path, 'w', encoding='utf-8') as f:
        for i, (zone, pids) in enumerate(sorted(
                _group_by_zone(dump).items())):
            f.write('%d\t%s\t%s\t%s\n'
                    % (i, zone, ' '.join(pids),
                       ' '.join('-' for _ in pids)))


def _group_by_zone(dump):
    out = {}
    for pid, z in dump.items():
        out.setdefault(str(z), []).append(str(pid))
    return out


def test_read_board_torn_tolerance(fails):
    """MEASURES (client fact #14): a dump under 400 pieces is torn -
    read_board re-reads and returns None; a stable full dump is
    accepted with decoded zones; two full reads that DISAGREE are torn
    until a read repeats.  Acceptance is proven on the REAL 428-piece
    specimen when the lab is present."""
    tmp = tempfile.mkdtemp(prefix='wotr_dump_')
    torn = os.path.join(tmp, 'torn.txt')
    _write_dump_file(torn, {'%03d' % i: 'FP Reinf' for i in range(20)})
    if cb.read_board(torn, delay=0) is not None:
        fails.append(('torn_rejected', 'sub-400 dump accepted'))
    full = os.path.join(tmp, 'full.txt')
    big = {'%03d' % i: 'FP Reinf' for i in range(420)}
    big['213'] = 'FP Dice'
    _write_dump_file(full, big)
    got = cb.read_board(full, delay=0)
    if not (got and len(got) == 420 and got['213'] == 'FP Dice'):
        fails.append(('full_accepted', 'got=%r'
                      % (got and len(got),)))
    # instability: two differing full reads, then a repeat
    a = dict(big)
    b = dict(big)
    b['213'] = 'Current Action'
    seq = iter([a, b, b])
    got = cb.read_board('ignored', delay=0, retries=3,
                        _parse=lambda: next(seq))
    if not (got and got['213'] == 'Current Action'):
        fails.append(('unstable_settles', 'got=%r' % (got,)))
    seq2 = iter([a, b, a])
    got = cb.read_board('ignored', delay=0, retries=3,
                        _parse=lambda: next(seq2))
    if got is not None:
        fails.append(('unstable_rejected', 'flapping dump accepted'))
    if os.path.exists(_LAB_DUMP):
        real = cb.read_board(_LAB_DUMP, delay=0)
        if not (real and len(real) == 428
                and real.get('249') == 'Vilya Slot'):
            fails.append(('real_specimen', 'n=%r 249=%r'
                          % (real and len(real),
                             real and real.get('249'))))
    else:
        print('  [read_board] real-specimen leg NOT-MEASURED '
              '(lab dump absent)')


def _pid_buckets():
    """(nation, kind) -> unclaimed pids, from log_replay's own tables
    (the same answerer the witness types with)."""
    import log_replay as lr
    import emitter as em
    skip = (set(em._FP_DIE_IDS) | set(em._SH_DIE_IDS)
            | set(lr.COMPANIONS) | set(lr.MINION_PIECES)
            | set(lr.HUNT_TILE_PIECES) | lr.FELLOWSHIP_MARKERS
            | {'%03d' % i for i in range(405, 432)}
            | {'422', '423'})
    buckets = {}
    for i in range(0, 470):
        pid = '%03d' % i
        if pid in skip:
            continue
        nat, ut = lr.get_unit_type_from_piece_id(pid)
        if ut is None:
            continue
        buckets.setdefault((_nv(nat), _nv(ut)), []).append(pid)
    return buckets


def build_dump_from_state(st):
    """A client dump {pid: zone} that IS the given engine state,
    assembled from log_replay's own pid tables (units/leaders/nazgul),
    the emitter's die pid constants, and the decoded zone names
    read_board produces.  Spare physical pieces go to 'Spare' (a real
    dump zone; census-invisible)."""
    from game_state import Nazgul
    import log_replay as lr
    buckets = _pid_buckets()
    used = set()

    def take(key):
        for pid in buckets.get(key, []):
            if pid not in used:
                used.add(pid)
                return pid
        raise LookupError('no free pid for %r' % (key,))
    dump = {}
    for k, a in st.armies.items():
        zone = str(k)
        for u in a.units:
            dump[take((_nv(u.nation), _nv(u.kind)))] = zone
        for ld in a.leaders:
            if isinstance(ld, Nazgul):
                dump[take(('none', 'nazgul'))] = zone
            else:
                dump[take((_nv(ld.nation), 'leader'))] = zone
    for pid in ('210', '211', '212', '213'):
        dump[pid] = 'FP Dice'
    for pid in ('099', '100', '101', '102', '103', '104', '105'):
        dump[pid] = 'SA Dice'
    for pid in ('358', '359'):
        dump[pid] = 'FP Reinf'      # never-granted bonus dice
    for pid in ('360', '361', '362'):
        dump[pid] = 'SA Reinf'
    for zone, deck in (('FP Char Deck', st.fp_char_deck),
                       ('FP Strat Deck', st.fp_strat_deck),
                       ('SA Char Deck', st.sh_char_deck),
                       ('SA Strat Deck', st.sh_strat_deck)):
        base = 500 + 30 * ('FP Char Deck', 'FP Strat Deck',
                           'SA Char Deck',
                           'SA Strat Deck').index(zone)
        for j in range(len(deck or [])):
            dump[str(base + j)] = zone
    dump['240'] = 'Guide Box'
    for pid in ('241', '242', '243', '244', '245', '246'):
        dump[pid] = 'Fellowship Box'
    dump['364'] = 'FP Reinf'                      # Gollum waits
    for pid in ('253', '254', '255'):
        dump[pid] = 'SA Reinf'                    # minions out of play
    dump['249'], dump['250'], dump['251'] = ('Vilya Slot',
                                             'Nenya Slot',
                                             'Narya Slot')
    for i in range(405, 411):
        dump['%03d' % i] = 'FP Reinf'
    for i in range(411, 421):
        dump['%03d' % i] = 'SA Reinf'
    for pid in lr.HUNT_TILE_PIECES:
        dump[pid] = 'Hunt Pool'
    for key, pids in buckets.items():
        for pid in pids:
            if pid not in used and pid not in dump:
                dump[pid] = 'Spare'
    return dump


def test_diff_specificity_zero_rows(fails):
    """MEASURES: a fresh engine state vs a dump constructed FROM that
    state produces ZERO discrepancy rows across every witness section
    (units, leaders, nazgul, dice, decks, pools, companions, control
    markers, rings, conservation).  The sensitivity twins below prove
    this zero is readable."""
    import ai
    st = fresh_state()
    dump = build_dump_from_state(st)
    rows = cb.diff_engine_vs_dump(st, dump, ai_module=ai,
                                  initial_dump=dump)
    if rows != []:
        fails.append(('diff_specificity', 'rows=%r' % (rows[:6],)))
    census = cb.card_census(st, dump)
    if census != []:
        fails.append(('census_specificity', 'rows=%r' % (census,)))


def test_diff_sensitivity_one_piece(fails):
    """MEASURES: perturbing ONE piece in the dump produces EXACTLY ONE
    row naming its region and type (a unit slipped off-board to
    'Spare' - the deleted-piece render class)."""
    import ai
    st = fresh_state()
    dump = build_dump_from_state(st)
    victim = next(p for p, z in sorted(dump.items())
                  if _nv(z) == 'gorgoroth'
                  and cb.unit_type(p)[1] == 'regular')
    dump2 = dict(dump)
    dump2[victim] = 'Spare'
    rows = cb.diff_engine_vs_dump(st, dump2, ai_module=ai)
    if len(rows) != 1:
        fails.append(('sensitivity_row_count', 'rows=%r' % (rows,)))
        return
    if 'units gorgoroth sauron/regular' not in rows[0] \
            or 'engine=3 client=2' not in rows[0]:
        fails.append(('sensitivity_row_names_it', rows[0]))


def test_diff_phantom_pool_dynamic_types(fails):
    """MEASURES (physical-limits ruling + A17): a pool holding MORE
    pieces than the engine allows is a PHANTOM row, and a pid typed
    only by dynamic_types (runtime swap assignment) is typed through
    it.  Non-zero control built in: 3 real spare dwarf elites in FP
    Reinf equal the engine's avail=3 (no row); the 4th, a
    dynamic-typed phantom, tips it."""
    import ai
    import log_replay as lr
    from game_state import Nation, UnitKind
    st = fresh_state()
    dump = build_dump_from_state(st)
    spares = [p for p, z in sorted(dump.items())
              if z == 'Spare' and cb.unit_type(p)
              == (Nation.DWARVES, 'elite')]
    if len(spares) != 3:
        fails.append(('phantom_precondition',
                      'dwarf elite spares=%r' % (spares,)))
        return
    dump3 = dict(dump)
    for p in spares:
        dump3[p] = 'FP Reinf'
    rows = cb.diff_engine_vs_dump(st, dump3, ai_module=ai)
    if rows != []:
        fails.append(('phantom_control', 'at-limit pool rows=%r'
                      % (rows,)))
        return
    dyn = {'901': (Nation.DWARVES, UnitKind.ELITE)}
    dump3['901'] = 'FP Reinf'
    rows = cb.diff_engine_vs_dump(st, dump3, dynamic_types=dyn,
                                  ai_module=ai)
    if len(rows) != 1 or 'PHANTOM' not in rows[0] \
            or 'dwarves/elite' not in rows[0]:
        fails.append(('phantom_fires', 'rows=%r' % (rows,)))


def test_diff_conservation_and_companion(fails):
    """MEASURES: a pid VANISHED from the dump raises the CONSERVATION
    row naming it; a companion out of its engine zone class raises the
    companion row (both sensitivity twins of the zero above)."""
    import ai
    st = fresh_state()
    dump = build_dump_from_state(st)
    victim = next(p for p, z in sorted(dump.items())
                  if _nv(z) == 'gorgoroth'
                  and cb.unit_type(p)[1] == 'regular')
    dump4 = dict(dump)
    del dump4[victim]
    rows = cb.diff_engine_vs_dump(st, dump4, ai_module=ai,
                                  initial_dump=dump)
    if not any('CONSERVATION' in r and victim in r for r in rows):
        fails.append(('conservation_fires', 'rows=%r' % (rows,)))
    dump5 = dict(dump)
    dump5['243'] = 'Lorien'          # Gimli is in the fellowship
    rows = cb.diff_engine_vs_dump(st, dump5, ai_module=ai)
    if not any(r.startswith('companion Gimli:') for r in rows):
        fails.append(('companion_fires', 'rows=%r' % (rows,)))


def test_card_census_size_and_identity(fails):
    """MEASURES (client fact #10): hand SIZE comes from the dump's
    hand-zone membership; IDENTITY rows appear only against caller-
    supplied commitment-decoded cids, never from the dump or a pid
    map.  Sensitivity and specificity on both axes."""
    st = fresh_state()
    for _ in range(2):
        game_logic.draw_event_card(st, Side.FREE_PEOPLES, 'character')
    dump = build_dump_from_state(st)    # decks already 2 short
    dump['531'], dump['532'] = 'FP Hand', 'FP Hand'
    rows = cb.card_census(st, dump)
    if rows != []:
        fails.append(('census_size_specificity', 'rows=%r' % (rows,)))
    short = dict(dump)
    del short['532']
    rows = cb.card_census(st, short)
    if not (len(rows) == 1 and 'hand size FP: engine=2 client=1'
            in rows[0]):
        fails.append(('census_size_sensitivity', 'rows=%r' % (rows,)))
    eng = sorted(c for c in st.hands[Side.FREE_PEOPLES].hand
                 if isinstance(c, str))
    good = {Side.FREE_PEOPLES: list(eng), Side.SHADOW: []}
    rows = cb.card_census(st, dump, hand_cids=good)
    if rows != []:
        fails.append(('census_id_specificity', 'rows=%r' % (rows,)))
    wrong = {Side.FREE_PEOPLES: eng[:-1] + ['bogus_card'],
             Side.SHADOW: []}
    rows = cb.card_census(st, dump, hand_cids=wrong)
    if not (len(rows) == 1 and 'engine-only' in rows[0]
            and 'bogus_card' in rows[0]):
        fails.append(('census_id_sensitivity', 'rows=%r' % (rows,)))


def test_proj_and_delta(fails):
    """MEASURES: board_delta is empty across two independently built
    fresh states (deck shuffles excluded by design) and reports an
    army unit removal and a ring spend as increments."""
    st1, st2 = fresh_state(), fresh_state()
    d = cb.board_delta(st1, st2)
    if d != {}:
        fails.append(('delta_specificity', 'd=%r' % (d,)))
    key = next(k for k in st2.armies if 'orgoroth' in str(k))
    st2.armies[key].units.pop()
    st2.fp_elven_rings = 2
    d = cb.board_delta(st1, st2)
    if 'gorgoroth' not in d or d.get('__rings__') != (-1, 0):
        fails.append(('delta_sensitivity', 'd=%r' % (d,)))
        return
    ud, ld = d['gorgoroth']
    if ud != (('sauron:regular', -1),) or ld != 0:
        fails.append(('delta_increment_shape', 'row=%r' % (d['gorgoroth'],)))


def main():
    tests = [
        test_verb_coverage,
        test_stack_after_real_fold,
        test_stack_after_lifo_pop,
        test_presses_own_emission_eats_press,
        test_presses_ineligible_never_guessed,
        test_presses_client_true_details,
        test_restore_watch_alarm_fires,
        test_restore_watch_specificity,
        test_restore_watch_track_wording,
        test_expectations_from_window,
        test_read_board_torn_tolerance,
        test_diff_specificity_zero_rows,
        test_diff_sensitivity_one_piece,
        test_diff_phantom_pool_dynamic_types,
        test_diff_conservation_and_companion,
        test_card_census_size_and_identity,
        test_proj_and_delta,
    ]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
