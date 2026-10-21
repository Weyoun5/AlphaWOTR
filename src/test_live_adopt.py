"""CLIENT -> ENGINE adoption + the unprompted-random-draw guard.

USER 2026-08-21:
  * "the resync is supposed to take the CLIENT state and give it to
     the engine, not the other way around"
  * "it also drew a random companion without me saying so ... i moved
     the fellowship to get revealed and then was going to kill the
     guide but the ai decided that meant random for some reason"

The adoption legs run against the USER's REAL board dump from the
reported game, not a fixture I invented.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ai                                       # noqa: E402
from game_state import Side                     # noqa: E402

from wotr_bridge.live import adopt              # noqa: E402
from wotr_bridge.live import client_board as cb  # noqa: E402
from test_live_driver import Rig, fresh_state   # noqa: E402

FIXTURE = os.path.join(_ROOT, 'data', 'live_fixtures',
                       'client_board_20260821_strider_guide.txt')


def real_dump():
    d = cb.read_board(FIXTURE)
    assert d, 'real dump must survive read_board'
    return d


def census(st, dump):
    return cb.diff_engine_vs_dump(st, dump, ai_module=ai) or []


# ------------------------------------------------------------------ adopt
def test_adoption_reduces_divergence_on_the_real_board():
    dump, st = real_dump(), fresh_state()
    before = census(st, dump)
    rep, nb, na = adopt.adopt_client_board(st, dump, ai_module=ai)
    assert nb == len(before)
    assert na < nb, 'adoption must reduce divergence (%d -> %d)' % (nb, na)
    after = census(st, dump)
    assert len(after) == na
    # every adopted CLASS must be fully gone, not merely reduced
    for cls in ('units ', 'leaders ', 'nazgul ', 'companion '):
        assert not [r for r in after if r.startswith(cls)], \
            [r for r in after if r.startswith(cls)]


def test_guide_becomes_strider_BY_THE_RULES():
    """The engine defaults the guide to Gandalf.  The client shows
    Gandalf DEAD, so once he is adopted out of the Fellowship, Strider
    is the sole level-3 companion and change_guide picks him with no
    tie to break - the right answer arrives through the rules, not an
    override."""
    dump, st = real_dump(), fresh_state()
    assert st.fellowship.guide.name == 'Gandalf the Grey'
    adopt.adopt_client_board(st, dump, ai_module=ai)
    assert st.fellowship.guide.name == 'Strider', \
        st.fellowship.guide.name
    assert 'Gandalf the Grey' not in [
        c.name for c in st.fellowship.companions]


def test_adoption_is_idempotent():
    dump, st = real_dump(), fresh_state()
    _, _, na1 = adopt.adopt_client_board(st, dump, ai_module=ai)
    rep2, nb2, na2 = adopt.adopt_client_board(st, dump, ai_module=ai)
    assert nb2 == na1 and na2 == na1, (nb2, na1, na2)
    assert not [r for r in rep2
                if r.startswith(('units ', 'leaders ', 'companion '))], \
        rep2


def test_gate_declines_and_rolls_back_when_adoption_would_worsen():
    """SAFETY CONTROL: the census is the oracle.  If an adopter step
    made things worse the write must be rolled back entirely - this is
    what makes a partial adopter safe to ship."""
    dump, st = real_dump(), fresh_state()
    baseline = census(st, dump)
    orig = adopt.adopt_units

    def wrecker(state, dmp, dynamic_types=None):
        for a in state.armies.values():   # corrupt every roster
            a.units = list(a.units) * 3
        return ['(deliberate corruption)']

    adopt.adopt_units = wrecker
    try:
        rep, nb, na = adopt.adopt_client_board(st, dump, ai_module=ai)
    finally:
        adopt.adopt_units = orig
    assert any('DECLINED' in r for r in rep), rep
    assert na == nb
    assert len(census(st, dump)) == len(baseline), \
        'rollback failed - engine was left corrupted'


def test_mutation_no_unit_adoption_leaves_unit_divergence():
    """MUTATION: with unit adoption disabled the unit rows must come
    BACK, or the leg above proves nothing."""
    dump, st = real_dump(), fresh_state()
    orig = adopt.adopt_units
    adopt.adopt_units = lambda *a, **k: []
    try:
        adopt.adopt_client_board(st, dump, ai_module=ai)
    finally:
        adopt.adopt_units = orig
    assert [r for r in census(st, dump) if r.startswith('units ')], \
        'MUTATION NOT DETECTED: units clean with adoption disabled'


def test_resync_lever_adopts_into_the_engine():
    dump, st = real_dump(), fresh_state()
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.piece_region = dump
    import shutil
    from bridge_evidence_fixture import snapshot_header
    path = os.path.join(rig.td, 'board.txt')
    shutil.copy2(os.path.join(_ROOT, 'data', 'live_fixtures',
                              'client_board_20260821_strider_guide.txt'), path)
    rig.drv.view.dump_path = path
    snapshot_header(rig.drv)
    items = rig.drv.resync_recover(drain_feed=False)
    notes = ' | '.join(str(v) for k, v in items if k == 'note')
    assert 'resync-adopt' in notes, notes
    assert st.fellowship.guide.name == 'Strider'
    assert not [r for r in census(st, dump) if r.startswith('units ')]


# ------------------------------------- unprompted random companion draw
def hunt_pending(st):
    st.fellowship.hidden = False
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    return ai.describe_pending(st)


def test_random_casualty_is_not_gesture_matchable():
    """The victim of hunt_sacrifice_random is drawn by the CLIENT when
    the action is applied, so no already-rendered record can evidence
    it - and 'matching' it EXECUTES the draw.  It must be held."""
    st = fresh_state()
    dp = hunt_pending(st)
    raw = {a.action_type for a in dp['actions']}
    assert 'hunt_sacrifice_random' in raw, raw
    rig = Rig(st, Side.SHADOW)
    rig.drv.pending_buf = None                 # nothing evidenced
    kept = {a.action_type for a in rig.drv._pending_menu(dp)}
    assert 'hunt_sacrifice_random' not in kept, kept
    assert 'hunt_sacrifice_guide' in kept, \
        'held the random draw but also lost the guide sacrifice'
    assert 'hunt_take_corruption' in kept, kept
    # BL-144 verdict: PROTOCOL-INFO wording - the random option is
    # WITHHELD from gesture matching, never a hold FROM the human
    assert any(k == 'random_casualty_withheld'
               for k, _ in rig.drv.findings)


def test_manual_resync_adopts_client_as_is():
    """USER 2026-08-25 (verbatim): "if I run a manual resync, the
    engine should take the client state AS IS."  The MANUAL lever
    adopts the families the auto-reconcile leaves engine-side:
    political boxes + activation (the dump's flag column), both hands
    (pid -> cid), deck counts.  On the REAL fixture dump overlaid with
    a turn-1 shape: North moved to 'Political 1' and flipped active;
    the FP hand holding two decoded pids (one the engine never saw
    drawn); a card the engine held but the client does not ->
    face-up discard.  SPECIFICITY: the auto path (manual=False) leaves
    politics and hands untouched."""
    from game_state import Nation
    import game_logic as gl
    dump, st = real_dump(), fresh_state()
    # move every real FP-hand piece out of the hand zone, then place
    # two controlled pids there
    for p, z in list(dump.items()):
        if str(z).strip().lower() == 'fp hand':
            dump[p] = 'FP Dead'
    dump['145'] = 'FP Hand'
    dump['148'] = 'FP Hand'
    dump['258'] = 'Political 1'                      # engine box 2
    flags = {'258': '-'}                             # active side
    hand = st.hands[Side.FREE_PEOPLES].hand
    while len(hand) < 2:                    # setup deals no cards yet
        hand.append(st.fp_char_deck.pop())
        hand.append(st.fp_strat_deck.pop())
    cid_a = hand[0]
    cid_x = hand[1]
    cid_b = st.fp_strat_deck[-1]                     # an undrawn card
    pid_to_cid = {'145': cid_a, '148': cid_b}
    n_strat_zone = sum(1 for z in dump.values()
                       if str(z).strip().lower() == 'fp strat deck')
    mint = lambda deck: 'HU_FR_%s_9' % ('CHAR' if deck == 'character'
                                        else 'STRAT')
    # specificity first: the AUTO path leaves these families alone
    st_auto = fresh_state()
    adopt.adopt_client_board(st_auto, dump, ai_module=ai)
    assert st_auto.political.boxes[Nation.THE_NORTH] != 2, \
        'auto-reconcile adopted a political box'
    rep, nb, na = adopt.adopt_client_board(
        st, dump, ai_module=ai, manual=True, pid_to_cid=pid_to_cid,
        mint=mint, flags=flags, human_side=Side.FREE_PEOPLES)
    assert st.political.boxes[Nation.THE_NORTH] == 2, \
        (st.political.boxes[Nation.THE_NORTH], rep)
    assert st.political.active[Nation.THE_NORTH] is True, rep
    got = sorted(st.hands[Side.FREE_PEOPLES].hand)
    assert got == sorted([cid_a, cid_b]), (got, rep)
    deck_x = gl.card_deck_of(cid_x) or 'strategy'
    assert cid_x in st.discard_face_up['fp_%s' % deck_x], rep
    assert cid_b not in st.fp_strat_deck, 'adopted cid lingered in the deck'
    if n_strat_zone:
        assert len(st.fp_strat_deck) == n_strat_zone, \
            (len(st.fp_strat_deck), n_strat_zone, rep)
    assert not st.hands[Side.FREE_PEOPLES].needs_discard


def test_control_non_hunt_pendings_are_untouched():
    """SENSITIVITY CONTROL: the guard must be scoped to hunt pendings,
    or it would silently prune unrelated menus."""
    st = fresh_state()
    dp = {'kind': 'discard', 'actions': list(hunt_pending(st)['actions'])}
    rig = Rig(st, Side.SHADOW)
    rig.drv.pending_buf = None
    kept = rig.drv._pending_menu(dp)
    assert len(kept) == len(dp['actions'])


def test_mutation_without_the_guard_random_stays_matchable():
    st = fresh_state()
    dp = hunt_pending(st)
    rig = Rig(st, Side.SHADOW)
    rig.drv.pending_buf = None
    rig.drv._drop_unevidenced_random = lambda d, m: m      # mutation
    kept = {a.action_type for a in rig.drv._pending_menu(dp)}
    assert 'hunt_sacrifice_random' in kept, \
        'MUTATION NOT DETECTED: random already absent without the guard'


def test_adopt_tracks_political_and_fellowship():
    """MEASURES (USER trust ruling 2026-08-24 'trust engine for
    political track, client otherwise'): adopt_tracks adopts FSP
    progress (247/248 on 'Track N') and corruption (252) from the
    client but NEVER the political boxes - the engine is the
    political authority and the driver's _political_recon walks the
    CLIENT markers instead.  Sensitivity: the client markers shown at
    war do NOT move the engine boxes."""
    from game_state import Nation
    st = fresh_state()
    rohan0 = st.political.boxes[Nation.ROHAN]
    isen0 = st.political.boxes[Nation.ISENGARD]
    act0 = st.political.active[Nation.ROHAN]
    dump = {'259': 'Political 0',      # Rohan at war (client 0)
            '263': 'Political 3',      # Isengard passive (client 3)
            '248': 'Track 4', '252': 'Track 2',
            '256': 'Political %d' % (3 - st.political.boxes[Nation.ELVES])}
    rep = adopt.adopt_tracks(st, dump)
    assert st.political.boxes[Nation.ROHAN] == rohan0, \
        'political adopted from the client (engine is the authority)'
    assert st.political.boxes[Nation.ISENGARD] == isen0, \
        'political adopted from the client (engine is the authority)'
    assert st.political.active[Nation.ROHAN] == act0, 'active flag moved'
    assert st.fellowship.progress == 4, st.fellowship.progress
    assert st.fellowship.corruption == 2, st.fellowship.corruption
    assert not any('political' in r.lower() for r in rep), rep
    assert rohan0 != 3


def test_adopt_fellowship_pawn_region():
    """MEASURES (USER bug 2026-08-24 'the fellowship pawn was not
    tracked during resyncs'): the pawn (pid 247) on a MAP REGION in
    the dump moves the engine fellowship there (progress reset); on a
    'Mordor N' box the engine adopts the Mordor step.  Sensitivity: a
    pawn already in place changes nothing."""
    import game_logic as gl
    st = fresh_state()
    cur = str(gl.fellowship_region(st))
    rep = adopt.adopt_fellowship_position(st, {'247': 'Trollshaws'})
    assert adopt._norm(str(gl.fellowship_region(st))) == 'trollshaws', \
        gl.fellowship_region(st)
    assert st.fellowship.progress == 0
    assert any('pawn' in r for r in rep), rep
    assert adopt._norm(cur) != 'trollshaws'
    # already in place: silent no-op
    rep2 = adopt.adopt_fellowship_position(st, {'247': 'Trollshaws'})
    assert rep2 == [], rep2
    # Mordor box: step adopted
    st2 = fresh_state()
    rep3 = adopt.adopt_fellowship_position(
        st2, {'247': 'Mordor 2 - somewhere'})
    assert st2.fellowship.mordor_entered is True
    assert st2.fellowship.mordor_step == 2, st2.fellowship.mordor_step
    assert any('Mordor' in r for r in rep3), rep3


def test_adopt_client_board_carries_tracks():
    """MEASURES: the ONE adopter (adopt_client_board) reports the
    track adoption in its report lines - the lenient path and the
    reset lever call it, never adopt_tracks directly."""
    import inspect
    src = inspect.getsource(adopt.adopt_client_board)
    assert 'adopt_tracks(' in src, 'adopt_client_board does not call adopt_tracks'


# ---------------------------------------------------------- adopt_sieges
def _siege_dump(inside_side='fp', messy=False):
    """The REAL dump, reshaped into a siege at Helm's Deep: two
    `inside_side` units inside the walls, two opposing units in the
    field (messy=True additionally leaves one inside-side unit in the
    field)."""
    from combat import side_of_nation
    dump = dict(real_dump())
    # the real board holds an FP army at Helm's Deep - vacate the
    # region first so the constructed shape is the ONLY occupancy
    for pid, z in list(dump.items()):
        if str(z).lower().startswith("helm's deep"):
            dump[pid] = 'Westemnet'
    fp_pids, sh_pids = [], []
    for pid in dump:
        try:
            nat, ut = cb.unit_type(pid)
        except Exception:
            continue
        if ut not in ('regular', 'elite'):
            continue
        (fp_pids if side_of_nation(nat) is Side.FREE_PEOPLES
         else sh_pids).append(pid)
    assert len(fp_pids) >= 3 and len(sh_pids) >= 2, 'fixture too thin'
    ins = fp_pids if inside_side == 'fp' else sh_pids
    fld = sh_pids if inside_side == 'fp' else fp_pids
    dump[ins[0]] = "Helm's Deep Stronghold"
    dump[ins[1]] = "Helm's Deep Stronghold"
    dump[fld[0]] = "Helm's Deep"
    dump[fld[1]] = "Helm's Deep"
    if messy:
        dump[ins[2]] = "Helm's Deep"
    return dump


def test_adopt_sieges_flags_client_siege():
    """MEASURES (tester 2026-08-24 'can't play The Fighting Uruk-hai'):
    a siege the client shows (garrison inside the walls, besieger in
    the field) is adopted into state.besieged_strongholds - before
    this, the flag was engine-combat-only and a leniently-accepted
    siege left every siege-gated rule misjudging, unrepairable by
    RESYNC."""
    st = fresh_state()
    key = adopt._norm("Helm's Deep")
    assert key not in getattr(st, 'besieged_strongholds', set())
    adopt.adopt_client_board(st, _siege_dump(), ai_module=ai)
    assert key in st.besieged_strongholds, \
        sorted(st.besieged_strongholds)


def test_adopt_sieges_stale_flag_kept_unless_asked():
    """REWRITTEN 2026-08-29 (BRCMP-4, fpgame15 'siege dropped x60'):
    a flagged stronghold the dump shows nobody inside is the BL-76
    seat demand (the garrison not yet dragged into the Stronghold
    box), not a lifted siege - the plain adoption KEEPS the rules flag
    and reports the demand; only a consent form (the MANUAL resync,
    BL-147 client AS IS / the human's 'yes') drops it, loudly."""
    st = fresh_state()
    key = adopt._norm('Minas Tirith')
    st.besieged_strongholds.add(key)
    rep, _, _ = adopt.adopt_client_board(st, dict(real_dump()),
                                         ai_module=ai)
    assert key in st.besieged_strongholds, \
        'plain adoption dropped a rules flag: %r' % (rep,)
    assert any('siege kept' in r and 'seat' in r for r in rep), rep
    st2 = fresh_state()
    st2.besieged_strongholds.add(key)
    rep2, _, _ = adopt.adopt_client_board(
        st2, dict(real_dump()), ai_module=ai, manual=True,
        human_side=Side.SHADOW)
    assert key not in st2.besieged_strongholds, \
        sorted(st2.besieged_strongholds)
    assert any('siege dropped' in r for r in rep2), rep2


def test_adopt_sieges_messy_shape_left_alone():
    """SPECIFICITY: one side split inside AND outside is not a clean
    siege - the flag is neither added nor invented; the report says
    so."""
    st = fresh_state()
    key = adopt._norm("Helm's Deep")
    rep, _, _ = adopt.adopt_client_board(st, _siege_dump(messy=True),
                                         ai_module=ai)
    assert key not in st.besieged_strongholds, \
        sorted(st.besieged_strongholds)
    assert any('messy shape' in r for r in rep), rep


def test_adopt_minions_relocates_witch_king():
    """MEASURES (tester 2026-08-24, WK stranded in Eastemnet): the
    adopter relocates the engine's EXISTING Minion object to the dump
    region (pid 253); it never fabricates one the engine lacks."""
    import game_logic as gl
    from game_state import Nation
    st = fresh_state()
    m = gl.Minion(nation=Nation.SAURON, name='Witch King')
    gl.get_or_create_army(st, 'eastemnet').leaders.append(m)
    rep = adopt.adopt_minions(st, {'253': 'Westemnet'})
    west = next((a for k, a in st.armies.items()
                 if adopt._norm(k) == 'westemnet'), None)
    assert west is not None and m in west.leaders, rep
    east = next((a for k, a in st.armies.items()
                 if adopt._norm(k) == 'eastemnet'), None)
    assert east is None or m not in east.leaders
    assert any('Witch' in r for r in rep), rep
    # never fabricates: no object -> no write
    st2 = fresh_state()
    rep2 = adopt.adopt_minions(st2, {'253': 'Westemnet'})
    assert rep2 == [] and not any(
        type(l).__name__ == 'Minion'
        for k, a in st2.armies.items() for l in a.leaders), rep2


def test_army_move_take_leaders_carries_minion():
    """MEASURES (tester 2026-08-24 half 1, the 1.150 pattern): an
    army_move with take_leaders=True and an explicit n_nazgul must
    carry the non-Nazgul leader figures (the Witch-King Minion) - the
    old move_leaders=None fell into take()'s None->0 stranding."""
    import ai
    import game_logic as gl
    import game_state as gsm
    from game_state import Nation, UnitKind
    st = fresh_state()
    # This fixture is a Sauron army operating inside Rohan; its nation
    # must be At War before a voluntary move into another Rohan region.
    st.political.boxes[Nation.SAURON] = 3
    a = gl.get_or_create_army(st, 'eastemnet')
    a.units = [gsm.Unit(Nation.SAURON, UnitKind.REGULAR)
               for _ in range(3)]
    a.leaders = [gl.Minion(nation=Nation.SAURON, name='Witch King'),
                 gsm.Nazgul(nation=Nation.SAURON)]
    act = ai.Action(die_idx=-1, action_type='army_move',
                    params={'from': 'eastemnet', 'to': 'westemnet',
                            'n_regular': 3, 'n_elite': 0,
                            'take_leaders': True, 'n_nazgul': 0,
                            'side': 'sh'})
    post = ai.apply_action(st, act)
    west = next((ar for k, ar in post.armies.items()
                 if adopt._norm(k) == 'westemnet'), None)
    assert west is not None
    names = [str(getattr(l, 'name', type(l).__name__))
             for l in west.leaders]
    assert any('Witch' in n for n in names), \
        'WK stranded: westemnet leaders=%r' % (names,)
    assert not any(isinstance(l, gsm.Nazgul) for l in west.leaders), \
        'n_nazgul=0 must leave the Nazgul behind'


def test_adopt_tracks_owed_hold():
    """MEASURES (BL-2, 2026-08-25 - fpgame1: the tile-1r corruption
    the engine correctly applied was silently ERASED by adopt_tracks
    reading the untouched client marker): while an engine-side track
    change is OWED its rendered marker move, that track is HELD, not
    adopted.  Release: the dump showing the owed target flows again.
    SENSITIVITY: without the owed dict the same stale dump still
    erases (the lenient default the guard exists to stop)."""
    st = fresh_state()
    st.fellowship.corruption = 1          # engine applied hunt damage
    rep = adopt.adopt_tracks(st, {'252': 'Track 0'},
                             owed={'252': 1})
    assert st.fellowship.corruption == 1, st.fellowship.corruption
    assert any('HELD' in r for r in rep), rep
    # the client marker catches up: no hold, nothing to change
    rep2 = adopt.adopt_tracks(st, {'252': 'Track 1'},
                              owed={'252': 1})
    assert st.fellowship.corruption == 1
    assert not any('HELD' in r for r in rep2), rep2
    # sensitivity: WITHOUT the hold the stale marker erases the damage
    st2 = fresh_state()
    st2.fellowship.corruption = 1
    adopt.adopt_tracks(st2, {'252': 'Track 0'})
    assert st2.fellowship.corruption == 0, \
        'MUTATION NOT DETECTED: the unguarded path no longer adopts'


def test_capture_coherence_reverts_chimera():
    """MEASURES (BL-21, the Woodland Realm chimera): a settlement
    marked CAPTURED while the original owner's units stand in it is
    rules-impossible - capture_coherence reverts the capture loudly,
    and rules_invariant_rows sees the state as a census row BEFORE
    the revert (so DIVERGENCE=STOP can fire on agreed-wrong
    boards)."""
    import game_state as gs
    from wotr_bridge.live import client_board as cbm
    st = fresh_state()
    key = next(k for k in st.armies
               if adopt._norm(k) == adopt._norm('Woodland Realm'))
    assert any(u for u in st.armies[key].units), 'precondition'
    st.enemy_controlled.setdefault(gs.Side.SHADOW, set()).add(key)
    rows = cbm.rules_invariant_rows(st)
    assert any('captured-with-garrison' in r for r in rows), rows
    rep = adopt.capture_coherence(st)
    assert any('REVERTED' in r for r in rep), rep
    assert key not in st.enemy_controlled.get(gs.Side.SHADOW, set())
    assert not cbm.rules_invariant_rows(st)


def test_mutation_without_adopt_sieges_flag_stays_absent():
    """MUTATION: with adopt_sieges disabled the flag must NOT appear,
    or the positive leg proves nothing."""
    st = fresh_state()
    key = adopt._norm("Helm's Deep")
    orig = adopt.adopt_sieges
    adopt.adopt_sieges = lambda *a, **k: []
    try:
        adopt.adopt_client_board(st, _siege_dump(), ai_module=ai)
    finally:
        adopt.adopt_sieges = orig
    assert key not in getattr(st, 'besieged_strongholds', set()), \
        'MUTATION NOT DETECTED: flag adopted with adopt_sieges disabled'


_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
_FPGAME2 = os.path.join(_LAB, 'fixtures', 'sessions', 'fpgame2')


def test_bl21_5_reconcile_clears_resurrected_casualties():
    """BL-21(5), proved on the real fpgame2 capture: the resync
    resurrected the battle-dead Woodland Realm Elf garrison to client
    truth but left fp_casualties={reg,elite,leader:1}, so conservation
    broke (Elves leader 4+1>4).  reconcile_fp_casualties on the
    client's (empty) FP dead pile must clear the ghosts to 0 and leave
    conservation clean.  SENSITIVITY: the frozen pickle must violate
    BEFORE the reconcile - a pin that cannot fail is not a pin."""
    import pickle
    from test_conservation import conservation_violations
    pk = os.path.join(_FPGAME2, 'ai_live_state.pkl')
    dp = os.path.join(_FPGAME2, 'ai_client_board.txt')
    if not (os.path.exists(pk) and os.path.exists(dp)):
        print('  SKIP bl21_5: fixture missing')
        return
    with open(pk, 'rb') as f:
        st = pickle.load(f)
    before = conservation_violations(st)
    assert any(v[0] == 'Elves' and v[1] == 'leader' for v in before), \
        'SENSITIVITY: fpgame2 must violate conservation before reconcile'
    dump = cb.read_board(dp)
    notes = adopt.reconcile_fp_casualties(st, dump, manual=False)
    after = conservation_violations(st)
    assert after == [], \
        'reconcile left a conservation violation: %r (notes %r)' % (after, notes)
    assert any('fp_casualties' in n and 'ELVES' in n for n in notes), \
        'reconcile emitted no loud note: %r' % (notes,)


def test_bl21_5_auto_never_moves_the_ledger_on_the_dead_pile_alone():
    """REWRITTEN 2026-08-29 (BRCMP-4, fpgame15: 'fp_casualties ELVES
    regular: 2 -> 1 (auto downward)' put a dead Elf back into the
    muster pool because the client's dead pile LAGGED).  AUTO never
    lowers the ledger on the client's dead-pile count alone - neither
    down (a lagging / mis-parked dead piece) nor up (no fabricated
    death).  It lowers only by what the SAME adoption resurrected
    (`resurrected`), never below the client's count, and to restore
    the PHYSICAL COMPONENT LIMIT (USER 2026-08-18).  The MANUAL lever
    takes the client's count exactly (BL-147)."""
    st = fresh_state()
    import game_state as gs
    st.fp_casualties = {gs.Nation.GONDOR: {'regular': 2}}
    # empty dump (no FP dead pieces): the ledger is KEPT
    auto = adopt.reconcile_fp_casualties(st, {}, manual=False)
    assert st.fp_casualties[gs.Nation.GONDOR]['regular'] == 2 and not auto, \
        'auto lowered the ledger on the dead pile alone: %r' % (auto,)
    # this adoption resurrected one Gondor regular: lower by ONE
    res = adopt.reconcile_fp_casualties(
        st, {}, manual=False,
        resurrected={(gs.Nation.GONDOR, 'regular'): 1})
    assert st.fp_casualties[gs.Nation.GONDOR]['regular'] == 1, res
    assert res and 'resurrected' in res[0], res
    # the physical limit: deployed + casualties > pieces is repaired
    mx = ai.NATION_MAX_FIGURES[gs.Nation.GONDOR]['regular']
    dep = ai._count_deployed_all(st)[gs.Nation.GONDOR]['regular']
    st.fp_casualties = {gs.Nation.GONDOR: {'regular': mx}}
    cons = adopt.reconcile_fp_casualties(st, {}, manual=False)
    assert st.fp_casualties[gs.Nation.GONDOR]['regular'] == mx - dep, \
        (st.fp_casualties, cons)
    assert cons and 'conservation' in cons[0], cons
    # manual: client as-is (empty dead pile -> 0)
    st.fp_casualties = {gs.Nation.GONDOR: {'regular': 2}}
    man = adopt.reconcile_fp_casualties(st, {}, manual=True)
    assert st.fp_casualties[gs.Nation.GONDOR]['regular'] == 0, man


def test_bl21_5_runtime_witness_flags_conservation():
    """The runtime witness (rules_invariant_rows) must surface a
    conservation break so the tester's DIVERGENCE=STOP fires live -
    the fpgame2 ghost is the proof."""
    import pickle
    pk = os.path.join(_FPGAME2, 'ai_live_state.pkl')
    if not os.path.exists(pk):
        print('  SKIP bl21_5 witness: fixture missing')
        return
    with open(pk, 'rb') as f:
        st = pickle.load(f)
    rows = cb.rules_invariant_rows(st) or []
    assert any('conservation' in r and 'ELVES' in r for r in rows), \
        'witness did not flag the conservation break: %r' % (rows[:4],)


if __name__ == '__main__':
    fails = 0
    for nm, fn in sorted(globals().items()):
        if not nm.startswith('test_'):
            continue
        try:
            fn()
            print('PASS %s' % nm)
        except Exception as e:
            fails += 1
            print('FAIL %s: %r' % (nm, e))
    print('\n%s' % ('ALL PASS' if not fails else '%d FAILED' % fails))
    raise SystemExit(1 if fails else 0)
