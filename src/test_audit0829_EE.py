#!/usr/bin/env python3
"""Regression pins for the 2026-08-29 audit fix wave, prefix EE
(src/event_effects.py + src/card_options.py).

Every test builds its state the way test_rules.py does (board.load_board()
-> game_state.setup_initial_game -> mutate -> call the engine).  Each pin
went RED against the unfixed tree before the fix landed (see
EE_FIX_REPORT.md for the RED/GREEN lines).

Run:  python test_audit0829_EE.py      (prints TOTAL=<n> FAIL=<m>; exit 1
                                        on any failure)
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import board as board_mod                     # noqa: E402
from board import sanitize                    # noqa: E402
import game_state as gs                       # noqa: E402
from game_state import Side, Nation, UnitKind, Unit   # noqa: E402
import game_logic as gl                       # noqa: E402
import combat as cb                           # noqa: E402
import cards                                  # noqa: E402
import event_effects as ee                    # noqa: E402
import card_options as co                     # noqa: E402
import ai                                     # noqa: E402

FP, SH = Side.FREE_PEOPLES, Side.SHADOW
F = gs.ActionDieFace


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

def setup_game():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    gl.initialize_decks(st)
    cb.FINAL_HITS_HOOK = None
    return st


def clear(st, r):
    for k in list(st.armies):
        if sanitize(k) == sanitize(r):
            st.armies.pop(k)


def set_army(st, r, units, leaders=()):
    clear(st, r)
    a = gl.get_or_create_army(st, sanitize(r))
    a.units = list(units)
    a.leaders = list(leaders)
    return a


def army_at(st, r):
    return st.armies.get(ee._find_army_key(st, sanitize(r)))


def units_of(st, r, nation):
    a = army_at(st, r)
    return [u for u in (a.units if a else []) if u.nation == nation]


def R(nation, n):
    return [Unit(nation, UnitKind.REGULAR) for _ in range(n)]


def E(nation, n):
    return [Unit(nation, UnitKind.ELITE) for _ in range(n)]


def take_companion(st, name):
    for c in list(st.fellowship.companions):
        if c.name == name:
            st.fellowship.companions.remove(c)
            if st.fellowship.guide is c:
                st.fellowship.guide = st.fellowship.companions[0]
            return c
    raise KeyError(name)


def card_by_name(name):
    for cid, c in cards.get_cards().items():
        if c.card_name == name:
            return cid, c
    raise KeyError(name)


def table_card(st, name):
    from persistent_effects import create_persistent_card
    cid, c = card_by_name(name)
    pc = create_persistent_card(cid, c.card_name, c.side)
    st.persistent_cards.add_card(pc)
    return pc


def board_names(st):
    return {getattr(l, 'name', '') for a in st.armies.values()
            for l in a.leaders}


def opt_action(pco, idx):
    return ai.Action(die_idx=-1, action_type='card_option',
                     params={'idx': idx, 'card_id': pco['card_id'],
                             'card_name': pco['card_name'],
                             'side': pco['side'],
                             'choices': pco['options'][idx]})


def lorien_siege():
    """Lorien: 2 Elves inside, 3 Sauron besieging; 2 Sauron at Dimrill
    Dale (the EE2 fixture)."""
    st = setup_game()
    set_army(st, 'lorien', R(Nation.ELVES, 2) + R(Nation.SAURON, 3))
    st.besieged_strongholds = {'lorien'}
    set_army(st, 'dimrill dale', R(Nation.SAURON, 2))
    assert cb.garrison_side_for(st, 'lorien') == FP
    return st


# ---------------------------------------------------------------------------
# EE1-01  Through a Day and a Night: a besieged FP garrison cannot walk out
# ---------------------------------------------------------------------------

def test_ee1_01_tdaan_besieged_garrison_cannot_march_out():
    st = setup_game()
    bor = take_companion(st, 'Boromir')
    set_army(st, 'minas tirith', R(Nation.GONDOR, 3) + R(Nation.SAURON, 6),
             [bor])
    st.besieged_strongholds = {'minas tirith'}
    for r in st.board.adjacent('minas tirith'):
        clear(st, r)
    assert ee._besieged_garrison_source(st, 'minas tirith', FP)
    opts = ee._tdaan_options(st)
    assert not any(o['from_region'] == 'minas tirith' for o in opts), \
        "TDaaN offers a move OUT of a besieged garrison: %r" % (
            [o for o in opts if o['from_region'] == 'minas tirith'][:3],)
    spec = co.get_card_options(st, FP, 'Through a Day and a Night')
    assert not any(o.get('from_region') == 'minas tirith' for o in spec), \
        "card_options offers the besieged garrison as a TDaaN source"
    to = sanitize(next(iter(st.board.adjacent('minas tirith'))))
    _, card = card_by_name('Through a Day and a Night')
    res = ee._effect_through_day_and_night(
        st, card, {'from_region': 'minas tirith', 'to_region': to,
                   '_from_pending': True})
    assert 'minas tirith' in st.besieged_strongholds, \
        "the siege was lifted by the FP card play: %r" % (res,)
    assert len(units_of(st, 'minas tirith', Nation.GONDOR)) == 3, \
        "the garrison marched out: %r" % (res,)
    assert 'minas tirith' not in st.enemy_controlled.get(SH, set()), \
        "Minas Tirith captured by the FP play"


# ---------------------------------------------------------------------------
# 1.256  Paths of the Woses / Dead Men of Dunharrow DO leave a besieged
#        Helm's Deep (USER ruling, card-specific)
# ---------------------------------------------------------------------------

def test_1_256_paths_and_dead_men_leave_a_besieged_helms_deep():
    hd = sanitize("Helm's Deep")
    # Paths of the Woses: the besieged Helm's Deep army is a legal source
    st = setup_game()
    st.political.boxes[Nation.ROHAN] = 3
    st.political.active[Nation.ROHAN] = True
    set_army(st, hd, R(Nation.ROHAN, 3) + R(Nation.ISENGARD, 6))
    st.besieged_strongholds = {hd}
    opts = ee._paths_of_woses_options(st)
    assert {'from_region': hd, 'to_region': 'minas tirith'} in opts, \
        "1.256: Paths must offer the besieged Helm's Deep army: %r" % (opts,)
    # Dead Men of Dunharrow: the companion group leaves with Strider
    st = setup_game()
    st.political.boxes[Nation.ROHAN] = 3
    st.political.active[Nation.ROHAN] = True
    strd = take_companion(st, 'Strider')
    bor = take_companion(st, 'Boromir')
    set_army(st, hd, R(Nation.ROHAN, 3) + R(Nation.ISENGARD, 6), [strd, bor])
    st.besieged_strongholds = {hd}
    clear(st, 'erech')
    cid, card = card_by_name('Dead Men of Dunharrow')
    ok, why = ee.check_event_requirement(st, cid)
    assert ok, "Dead Men playable with Strider in a besieged Helm's Deep: %s" % why
    res = ee._effect_dead_men(st, card, {'target_region': 'erech',
                                         '_from_pending': True})
    names = {l.name for l in (army_at(st, 'erech').leaders
                              if army_at(st, 'erech') else [])}
    assert {'Strider', 'Boromir'} <= names, \
        "1.256: the companion group leaves Helm's Deep with Strider: %r %r" % (
            res, names)
    assert hd in st.besieged_strongholds, "the siege itself is untouched"
    assert len(units_of(st, hd, Nation.ROHAN)) == 3, "the garrison stays"


# ---------------------------------------------------------------------------
# EE1-02  separating the Guide on a Level tie opens pending_guide_change
# ---------------------------------------------------------------------------

def test_ee1_02_separating_the_guide_on_a_level_tie_opens_the_pending():
    cases = (
        ('I Will Go Alone', ee._effect_i_will_go_alone,
         lambda g, r: {'companions': [g], 'destination': r}),
        ('There and Back Again', ee._effect_there_and_back_again,
         lambda g, r: {'companions': [g], 'destination': r}),
        ('Gwaihir the Windlord', ee._effect_gwaihir,
         lambda g, r: {'companion_action': 'sep_' + g, 'destination': r}),
        ('We Prove Swifter', ee._effect_we_prove_swifter,
         lambda g, r: {'companion_action': 'sep_' + g, 'destination': r}),
    )
    for name, fn, mk in cases:
        st = setup_game()
        take_companion(st, 'Strider')          # leaves Gandalf L3 alone on top
        g = st.fellowship.guide
        assert g.name == 'Gandalf the Grey', g.name
        st.pending_guide_change = None
        _, card = card_by_name(name)
        res = fn(st, card, dict(mk(g.name, gl.fellowship_region(st)),
                                _from_pending=True))
        assert not res.get('refund'), (name, res)
        assert g not in st.fellowship.companions, (name, res)
        assert st.fellowship.guide in st.fellowship.companions, \
            "%s: guide is STALE (%s not in the Fellowship)" % (
                name, getattr(st.fellowship.guide, 'name', None))
        assert st.pending_guide_change is not None, \
            "%s: the Level-2 tie must open pending_guide_change" % name
        assert set(st.pending_guide_change['candidates']) == \
            {'Boromir', 'Legolas', 'Gimli'}, st.pending_guide_change


# ---------------------------------------------------------------------------
# EE2-1  Corsairs of Umbar: an empty-settlement landing advances Gondor ONCE
# ---------------------------------------------------------------------------

def test_ee2_1_corsairs_empty_pelargir_advances_gondor_once():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 0
    st.political.active[Nation.GONDOR] = False
    set_army(st, 'umbar', R(Nation.SOUTHRON_EASTERLINGS, 6))
    clear(st, 'pelargir')
    _, card = card_by_name('Corsairs of Umbar')
    res = ee._effect_corsairs_of_umbar(st, card, {'target_region': 'pelargir'})
    assert st.political.active[Nation.GONDOR] is True, res
    assert st.political.boxes[Nation.GONDOR] == 1, \
        "capture-advance must fire exactly once (box %d): %r" % (
            st.political.boxes[Nation.GONDOR], res)
    assert 'pelargir' in st.enemy_controlled[SH]


# ---------------------------------------------------------------------------
# EE2-2 / EE2-9  siege-aware join surface for TSL / TSiM (+ CL-253 parity)
# ---------------------------------------------------------------------------

def test_ee2_2_shadow_lengthens_chain_reinforces_and_lifts_a_siege():
    st = lorien_siege()
    # Joining a siege in Elven territory still requires Sauron at war.
    st.political.boxes[Nation.SAURON] = 3
    assert 'lorien' in ee._tsl_dests0(st), \
        "a besieger stack is a legal join destination"
    nxt = ee._tsl_next_options(st, set())
    assert any(o['from_region'] == 'dimrill dale' and o['to_region'] == 'lorien'
               for o in nxt), "dimrill dale -> lorien (join the besiegers) missing"
    assert any(o['from_region'] == 'lorien' for o in nxt), \
        "the besieger stack is a legal SOURCE (lifting the siege)"
    res = {'effects': []}
    ee._tsl_execute_staged(st, [['dimrill dale', 'lorien']], res)
    assert len(units_of(st, 'lorien', Nation.SAURON)) == 5, res
    assert not units_of(st, 'dimrill dale', Nation.SAURON), res
    assert 'lorien' in st.besieged_strongholds
    # CL-253 parity: a besieged SHADOW garrison is neither a source nor a
    # destination
    st2 = setup_game()
    set_army(st2, 'morannon', R(Nation.SAURON, 3))
    st2.besieged_strongholds = {'morannon'}
    assert ee._besieged_garrison_source(st2, 'morannon', SH)
    assert 'morannon' not in ee._tsl_dests0(st2)
    assert not any(o['from_region'] == 'morannon'
                   for o in ee._tsl_next_options(st2, set()))


def test_ee2_9_shadow_is_moving_besieger_source_and_destination():
    st = lorien_siege()
    _, card = card_by_name('The Shadow is Moving')
    spec = co.get_card_options(st, SH, 'The Shadow is Moving')
    assert any(o['from_region'] == 'lorien' for o in spec), \
        "spec: the besieger stack is a legal mover"
    assert any(o['from_region'] == 'dimrill dale' and o['to_region'] == 'lorien'
               for o in spec), "spec: joining the besiegers is legal"
    nx = ee._tsim_next_options(st, set(), 0)
    assert any(o['from_region'] == 'lorien' for o in nx)
    assert any(o['to_region'] == 'lorien' for o in nx)
    res = ee._effect_shadow_is_moving(
        st, card, {'from_region': 'dimrill dale', 'to_region': 'lorien'})
    assert len(units_of(st, 'lorien', Nation.SAURON)) == 5, res
    assert 'lorien' in st.besieged_strongholds
    # CL-253: a besieged SHADOW garrison is still refused as a source and
    # never offered as a destination
    st2 = setup_game()
    set_army(st2, 'morannon', R(Nation.SAURON, 3))
    st2.besieged_strongholds = {'morannon'}
    nx2 = ee._tsim_next_options(st2, set(), 0)
    assert not any(o['from_region'] == 'morannon' for o in nx2)
    assert not any(o['to_region'] == 'morannon' for o in nx2)
    to0 = sanitize(next(iter(st2.board.adjacent('morannon'))))
    r2 = ee._effect_shadow_is_moving(
        st2, card, {'from_region': 'morannon', 'to_region': to0})
    assert any('besieged garrison' in e for e in r2['effects']), r2


# ---------------------------------------------------------------------------
# EE2-3  Lure of the Ring: the draw precedes the FP decision (two stages)
# ---------------------------------------------------------------------------

def test_ee2_3_lure_of_the_ring_draws_the_victim_before_the_fp_decides():
    st = setup_game()
    st.fellowship.hidden = False
    st.fellowship.region = st.board.region('hollin')
    cid, card = card_by_name('Lure of the Ring')
    st.hands[SH].hand.append(cid)
    for d in st.dice[SH]:
        d.used = False
    st.dice[SH][0].current = F.EVENT
    comps0 = [c.name for c in st.fellowship.companions]
    s2 = ai.apply_action(st, ai.Action(
        die_idx=0, action_type='play_event',
        params={'card_id': cid, 'side': 'sh'}))
    pco = getattr(s2, 'pending_card_option', None)
    assert pco is not None and pco['card_name'] == 'Lure of the Ring', pco
    assert pco.get('decider') == 'fp', pco
    victims = {o.get('victim') for o in pco['options']}
    assert len(victims) == 1 and None not in victims, \
        "the FP decides BLIND - options carry no drawn victim: %r" % (
            pco['options'],)
    victim = next(iter(victims))
    assert victim in comps0
    assert all(o.get('level') for o in pco['options']), pco['options']
    assert [c.name for c in s2.fellowship.companions] == comps0, \
        "nothing is eliminated before the FP decides"
    assert {o.get('action') for o in pco['options']} == \
        {'corruption', 'eliminate'}
    idx = next(i for i, o in enumerate(pco['options'])
               if o.get('action') == 'eliminate')
    s3 = ai.apply_action(s2, opt_action(pco, idx))
    assert victim not in [c.name for c in s3.fellowship.companions], \
        "the FP's eliminate pick removes the DRAWN victim"
    assert s3.fellowship.guide in s3.fellowship.companions \
        or s3.fellowship.guide.name == 'Gollum'


# ---------------------------------------------------------------------------
# EE2-4 + 1.257  companion-linked table cards leave on BotF / Lure; Lure
#                books the elimination but never fires WWSAT
# ---------------------------------------------------------------------------

def test_ee2_4_breaking_of_the_fellowship_discards_horn_of_gondor():
    st = setup_game()
    horn = table_card(st, 'Horn of Gondor')
    cur = sanitize(gl.fellowship_region(st))
    pco = {'card_id': 'x', 'card_name': 'The Breaking of the Fellowship',
           'side': 'sh', 'decider': 'fp', 'bof_separation': True,
           'bof_n_left': 1, 'bof_region': cur, 'bof_mordor': False,
           'options': []}
    ee._bof_apply_separation(st, pco, 'Boromir')
    assert not any(c.name == 'Boromir' for c in st.fellowship.companions)
    assert st.persistent_cards.get_by_type(gs.PersistentCardType.HORN_OF_GONDOR) \
        is None, "Horn of Gondor must leave the table when Boromir leaves"
    assert horn.card_id in st.discard_face_up.get('fp_character', []), \
        "the table card is DISCARDED face-up (conservation)"
    # direct (choices) path of the handler
    st = setup_game()
    table_card(st, 'Horn of Gondor')
    st.hunt_pool = [gs.HuntTile(damage=1)]
    _, card = card_by_name('The Breaking of the Fellowship')
    ee._effect_breaking_of_fellowship(st, card, {'separate': ['Boromir']})
    assert not any(c.name == 'Boromir' for c in st.fellowship.companions)
    assert st.persistent_cards.get_by_type(gs.PersistentCardType.HORN_OF_GONDOR) \
        is None, "direct BotF path: Horn of Gondor must leave the table"


def test_1_257_lure_elimination_discards_horn_but_never_fires_wwsat():
    st = setup_game()
    horn = table_card(st, 'Horn of Gondor')
    table_card(st, 'Worn with Sorrow and Toil')
    st.hands[FP].hand = ['FREE_PEOPLES_Character_Athelas',
                         "FREE_PEOPLES_Character_Bilbo's_Song"]
    st._wwsat_owed = []
    _, card = card_by_name('Lure of the Ring')
    res = ee._effect_lure_of_ring(
        st, card, {'victim': 'Boromir', 'action': 'eliminate',
                   '_lure_drawn': True, '_from_pending': True})
    assert not any(c.name == 'Boromir' for c in st.fellowship.companions), res
    assert st.persistent_cards.get_by_type(gs.PersistentCardType.HORN_OF_GONDOR) \
        is None, "Lure elimination: Horn of Gondor must leave the table"
    assert horn.card_id in st.discard_face_up.get('fp_character', [])
    assert len(st.hands[FP].hand) == 2, \
        "1.257: WWSAT must NOT fire on a Lure elimination (hand touched)"
    assert not getattr(st, '_wwsat_owed', None), \
        "1.257: no WWSAT discard may be owed after Lure"
    assert st.fellowship.guide in st.fellowship.companions


# ---------------------------------------------------------------------------
# EE1-03  empty-pool rebuilds go through _recycle_hunt_pool (1.136)
# ---------------------------------------------------------------------------

def _stage_shelob_pre_mordor(st):
    st.fellowship.mordor_entered = False
    ee.stage_special_hunt_tile(st, "shelob's lair")
    assert any(getattr(t, 'is_special', False) for t in st.hunt_discard)


def test_ee1_03_challenge_of_the_king_rebuild_keeps_specials_out():
    st = setup_game()
    _stage_shelob_pre_mordor(st)
    std = list(st.hunt_pool)
    st.hunt_discard.extend(std[:-1])
    st.hunt_pool = std[-1:]
    strd = take_companion(st, 'Strider')
    set_army(st, 'minas tirith', R(Nation.GONDOR, 1), [strd])
    _, card = card_by_name('Challenge of the King')
    res = ee._effect_challenge_of_king(st, card, {})
    assert not any(getattr(t, 'is_special', False) for t in st.hunt_pool), \
        "pre-Mordor special tile pulled into the live pool: %r" % (res,)
    assert any(getattr(t, 'is_special', False) for t in st.hunt_discard), \
        "Shelob's Lair must stay parked in the used pile until Mordor entry"
    assert any('3 tiles' in e for e in res['effects']) \
        or '3 tiles' in res.get('message', ''), res


def test_ee1_03_mithril_redraw_rebuild_keeps_specials_out():
    st = setup_game()
    _stage_shelob_pre_mordor(st)
    st.hunt_discard.extend(st.hunt_pool)
    st.hunt_pool = []
    table_card(st, 'Mithril Coat and Sting')
    st._mithril_stash = {'tile': gs.HuntTile(damage=1), 'successes': 1,
                         'mordor': False}
    _, card = card_by_name('Mithril Coat and Sting')
    res = ee._effect_mithril_coat(st, card, {'mithril_redraw': True})
    assert not any(getattr(t, 'is_special', False) for t in st.hunt_pool), \
        "Mithril redraw rebuild pulled a pre-Mordor special into the pool: %r" \
        % (res,)
    assert any(getattr(t, 'is_special', False) for t in st.hunt_discard)


# ---------------------------------------------------------------------------
# EE2-5 / EE1-04  Foul Thing recycles an empty pool; Nazgul Strike hunt draws
# ---------------------------------------------------------------------------

def test_ee2_5_foul_thing_empty_pool_recycles_and_draws():
    st = setup_game()
    st.fellowship.region = st.board.region('hollin')
    n_all = len(st.hunt_pool)
    st.hunt_discard.extend(st.hunt_pool)
    st.hunt_pool = []
    _, card = card_by_name('Foul Thing from the Deep')
    res = ee._effect_foul_thing(st, card, {'_from_pending': True,
                                           '_skip_wizard': True})
    assert len(st.hunt_pool) == n_all - 1, \
        "1.136: Foul Thing must recycle the used pile and draw: %r pool=%d" % (
            res, len(st.hunt_pool))
    assert res['effects'], "a tile was drawn and reported"


def test_ee2_5_nazgul_strike_hunt_recycles_an_empty_pool():
    st = setup_game()
    st.fellowship.region = st.board.region('hollin')
    st.hunt_discard.extend(st.hunt_pool)
    st.hunt_pool = []
    st.hunt_box = [st.dice[SH][0]]                  # Hunt Level 1
    gl.get_or_create_army(st, 'hollin').leaders.append(gs.Nazgul(Nation.SAURON))
    _, card = card_by_name('Nazgûl Strike!')
    saved = gl._hunt_roll
    gl._hunt_roll = lambda state, num_dice, thresh, fp_modifier=0: 1
    try:
        res = ee._effect_nazgul_strike(
            st, card, {'action': 'hunt', '_from_pending': True})
    finally:
        gl._hunt_roll = saved
    assert any('tile drawn' in e for e in res['effects']), \
        "a successful Strike hunt must draw a tile from the recycled pool: %r" \
        % (res,)
    assert len(st.hunt_pool) + len(st.hunt_discard) >= 16
    assert len(st.hunt_pool) > 0, "the pool was recycled (1.136)"


# ---------------------------------------------------------------------------
# EE2-6  Foul Thing rolls the Shelob's Lair d6 (one body with Isildur's Bane)
# ---------------------------------------------------------------------------

def test_ee2_6_foul_thing_rolls_the_shelob_die():
    st = setup_game()
    st.fellowship.mordor_entered = True
    st.fellowship.region = st.board.region('minas morgul')
    st.fellowship.hidden = True
    st.hunt_discard.extend(st.hunt_pool)
    st.hunt_pool = []
    assert ee.stage_special_hunt_tile(st, "Shelob's Lair")
    assert st.hunt_pool and getattr(st.hunt_pool[0], 'is_dice_roll', False)
    c0 = st.fellowship.corruption
    n0 = len(st.fellowship.companions)
    saved = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [6] * n
    try:
        _, card = card_by_name('Foul Thing from the Deep')
        res = ee._effect_foul_thing(st, card, {'_from_pending': True,
                                               '_skip_wizard': True})
    finally:
        ee._d6_batch = saved
    assert not any('no effect' in e for e in res['effects']), \
        "Shelob's Lair read as a 0-damage tile: %r" % (res,)
    moved = (st.fellowship.corruption != c0
             or len(st.fellowship.companions) != n0
             or getattr(st, 'pending_card_option', None) is not None
             or getattr(st, 'pending_hunt', None) is not None)
    assert moved, "the d6 damage must reach the casualty/corruption path: %r" \
        % (res,)
    assert hasattr(ee, '_card_tile_damage'), \
        "the tile-damage evaluation must be ONE shared body"


# ---------------------------------------------------------------------------
# EE2-7  Nazgul Strike hunt rolls hunt_level dice (0 -> no roll)
# ---------------------------------------------------------------------------

def test_ee2_7_nazgul_strike_hunt_with_zero_hunt_level_rolls_nothing():
    st = setup_game()
    st.fellowship.region = st.board.region('hollin')
    st.hunt_box = [d for d in st.hunt_box if d.side is not SH]
    gl.get_or_create_army(st, 'hollin').leaders.append(gs.Nazgul(Nation.SAURON))
    calls = []
    saved = gl._hunt_roll

    def spy(state, num_dice, thresh, fp_modifier=0):
        calls.append(num_dice)
        return 0
    gl._hunt_roll = spy
    try:
        _, card = card_by_name('Nazgûl Strike!')
        res = ee._effect_nazgul_strike(
            st, card, {'action': 'hunt', '_from_pending': True})
    finally:
        gl._hunt_roll = saved
    assert not [n for n in calls if n > 0], \
        "0 Shadow dice in the Hunt Box must roll 0 dice, got %r" % (calls,)
    assert any('Hunt Level 0' in e for e in res['effects']), res


# ---------------------------------------------------------------------------
# EE2-8  walk-through regions activate the FP nation entered
# ---------------------------------------------------------------------------

def _rohan_walkthrough():
    st = setup_game()
    # This fixture exercises entry activation after a legal border crossing.
    st.political.boxes[Nation.ISENGARD] = 3
    for r in ('gap of rohan', 'fords of isen', 'fangorn', 'orthanc'):
        clear(st, r)
    set_army(st, 'gap of rohan', R(Nation.ISENGARD, 3))
    set_army(st, 'fangorn', R(Nation.ISENGARD, 1))
    st.political.boxes[Nation.ROHAN] = 0
    st.political.active[Nation.ROHAN] = False
    assert ee._free_path_mids(st, 'gap of rohan', 'fangorn', 3) == \
        ['fords of isen']
    return st


def test_ee2_8_walk_through_entry_activates_rohan():
    st = _rohan_walkthrough()
    _, sg = card_by_name('Shadows Gather')
    res = ee._effect_shadows_gather(
        st, sg, {'from_region': 'gap of rohan', 'to_region': 'fangorn'})
    assert st.political.active[Nation.ROHAN] is True, \
        "Shadows Gather through Fords of Isen must activate Rohan: %r" % (res,)
    st = _rohan_walkthrough()
    res = {'effects': []}
    ee._tsl_execute_staged(st, [['gap of rohan', 'fangorn']], res)
    assert st.political.active[Nation.ROHAN] is True, \
        "TSL staged walk-through must activate Rohan: %r" % (res,)
    st = _rohan_walkthrough()
    set_army(st, 'westemnet', R(Nation.ISENGARD, 2))
    clear(st, 'eastemnet')
    _, tsl = card_by_name('The Shadow Lengthens')
    res = ee._effect_shadow_lengthens(
        st, tsl, {'from_region': 'gap of rohan', 'to_region': 'fangorn',
                  'from_region_2': 'fangorn', 'to_region_2': 'gap of rohan'})
    assert st.political.active[Nation.ROHAN] is True, \
        "TSL chained-pair walk-through must activate Rohan: %r" % (res,)


# ---------------------------------------------------------------------------
# EE1-05  Ents Awake free chain offers only PLAYABLE Character cards
# ---------------------------------------------------------------------------

def test_ee1_05_ents_chain_offers_only_playable_cards():
    st = setup_game()
    gtw = gs.Companion("Gandalf the White", 3, 1)
    take_companion(st, 'Gandalf the Grey')
    set_army(st, 'fangorn', [], [gtw])
    hand = st.hands[FP].hand
    hand[:] = ["FREE_PEOPLES_Character_The_Ents_Awake:_Treebeard",
               "FREE_PEOPLES_Character_Challenge_of_the_King",
               "FREE_PEOPLES_Character_Dead_Men_of_Dunharrow",
               "FREE_PEOPLES_Character_Athelas"]
    assert not ee.check_event_requirement(
        st, "FREE_PEOPLES_Character_Challenge_of_the_King")[0]
    assert ee.check_event_requirement(st, "FREE_PEOPLES_Character_Athelas")[0]
    _, card = card_by_name("The Ents Awake: Treebeard")
    saved = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [1] * n     # no hits
    try:
        res = ee._effect_ents_awake_common(st, card, {'_from_pending': True})
    finally:
        ee._d6_batch = saved
    pco = st.pending_card_option
    assert pco is not None, res
    offered = {o.get('ents_chain_play') for o in pco['options']
               if o.get('ents_chain_play')}
    assert "FREE_PEOPLES_Character_Athelas" in offered, offered
    assert "FREE_PEOPLES_Character_Challenge_of_the_King" not in offered, \
        "an unplayable card (requirement False) is on the chain menu"
    assert "FREE_PEOPLES_Character_Dead_Men_of_Dunharrow" not in offered, \
        offered


# ---------------------------------------------------------------------------
# EE1-06  I Will Go Alone / There and Back Again offer GROUP separations
# ---------------------------------------------------------------------------

def test_ee1_06_iwga_and_taba_offer_group_separations():
    for name, fn in (('I Will Go Alone', ee._effect_i_will_go_alone),
                     ('There and Back Again',
                      ee._effect_there_and_back_again)):
        st = setup_game()
        opts = co.get_card_options(st, FP, name)
        groups = [o for o in opts if len(o.get('companions') or []) >= 2]
        assert groups, "%s offers no group separation (1.58e)" % name
        pick = next(o for o in groups
                    if set(o['companions']) == {'Meriadoc', 'Peregrin'})
        _, card = card_by_name(name)
        res = fn(st, card, dict(pick, _from_pending=True))
        assert not res.get('refund'), (name, res)
        assert {'Meriadoc', 'Peregrin'} <= board_names(st), (name, res)


# ---------------------------------------------------------------------------
# EE1-07 / EE1-14  per-unit non-belligerent border rule (1.222b) on TDaaN
#                  and Paths of the Woses
# ---------------------------------------------------------------------------

def test_ee1_07_tdaan_passive_units_may_not_enter_another_nation():
    st = setup_game()
    st.political.boxes[Nation.THE_NORTH] = 0
    st.political.active[Nation.THE_NORTH] = False
    bor = take_companion(st, 'Boromir')
    set_army(st, 'dale', R(Nation.THE_NORTH, 2), [bor])
    dests = {o['to_region'] for o in ee._tdaan_options(st)
             if o['from_region'] == 'dale'}
    assert 'erebor' not in dests and 'iron hills' not in dests, \
        "passive North units offered a Dwarves-border crossing: %r" % (
            sorted(dests),)
    # mixed stack: the At-War Elf may cross alone, the passive North
    # regulars stay behind (per-unit rule)
    st = setup_game()
    st.political.boxes[Nation.THE_NORTH] = 0
    st.political.active[Nation.THE_NORTH] = False
    ee.set_at_war(st, Nation.ELVES)
    assert st.political.is_at_war(Nation.ELVES)
    bor = take_companion(st, 'Boromir')
    set_army(st, 'dale', R(Nation.THE_NORTH, 2) + R(Nation.ELVES, 1), [bor])
    opts = ee._tdaan_options(st)
    assert {'from_region': 'dale', 'to_region': 'erebor'} in opts, \
        "the at-war Elf detachment may enter Erebor"
    assert not any(o['from_region'] == 'dale' and o['to_region'] == 'erebor'
                   and o.get('n_regular', 0) > 1 for o in opts), \
        "the split grid must be built from the MOVABLE pool only"
    _, card = card_by_name('Through a Day and a Night')
    res = ee._effect_through_day_and_night(
        st, card, {'from_region': 'dale', 'to_region': 'erebor',
                   '_from_pending': True})
    assert len(units_of(st, 'erebor', Nation.ELVES)) == 1, res
    assert not units_of(st, 'erebor', Nation.THE_NORTH), \
        "passive North units carried across the Dwarves border: %r" % (res,)
    assert len(units_of(st, 'dale', Nation.THE_NORTH)) == 2


def test_ee1_14_paths_lets_passive_gondor_units_go_home():
    st = setup_game()
    st.political.boxes[Nation.ROHAN] = 3
    st.political.active[Nation.ROHAN] = True
    st.political.boxes[Nation.GONDOR] = 1
    st.political.active[Nation.GONDOR] = False
    set_army(st, 'westemnet', R(Nation.ROHAN, 2) + R(Nation.GONDOR, 1))
    opts = ee._paths_of_woses_options(st)
    assert {'from_region': 'westemnet', 'to_region': 'minas tirith'} in opts
    _, card = card_by_name('Paths of the Woses')
    n_mt = len(units_of(st, 'minas tirith', Nation.GONDOR))
    res = ee._effect_paths_of_woses(
        st, card, {'from_region': 'westemnet', 'to_region': 'minas tirith'})
    assert not units_of(st, 'westemnet', Nation.GONDOR), \
        "1.222(b): a passive Gondor unit may enter its OWN nation: %r" % (res,)
    assert len(units_of(st, 'minas tirith', Nation.GONDOR)) == n_mt + 1
    assert not units_of(st, 'westemnet', Nation.ROHAN)


# ---------------------------------------------------------------------------
# 1.254  ALL card casualties are the OWNER's learned allocation
# ---------------------------------------------------------------------------

def _resolve_casualty(st, operation, nation, *, allow_roll=False):
    """One literal operation; only multi-target cards may roll their next target."""
    from unittest.mock import patch
    pco = st.pending_card_option
    assert pco is not None and pco.get('card_casualty'), pco
    idx = next(i for i, o in enumerate(pco['options'])
               if o.get('casualty_operation') == operation and o.get('nation') == nation)
    if allow_roll:
        return ai.apply_action(st, opt_action(pco, idx))
    with patch.object(ee, '_d6_batch', side_effect=AssertionError('card roll repeated on casualty resume')):
        return ai.apply_action(st, opt_action(pco, idx))


def test_1_254_dead_men_casualties_are_owner_allocated_then_the_card_resumes():
    st = setup_game()
    strd = take_companion(st, 'Strider')
    set_army(st, 'westemnet', R(Nation.ROHAN, 1), [strd])
    set_army(st, 'erech', R(Nation.SAURON, 1) + E(Nation.SAURON, 2))
    _, card = card_by_name('Dead Men of Dunharrow')
    saved = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [2] * n      # 2 hits
    try:
        res = ee._effect_dead_men(st, card, {'target_region': 'erech',
                                             '_from_pending': True})
    finally:
        ee._d6_batch = saved
    pco = st.pending_card_option
    assert pco is not None and pco.get('card_casualty'), \
        "1.254: Dead Men must open the Shadow owner's allocation: %r" % (res,)
    assert pco['decider'] == 'sh' and pco['cas_region'] == 'erech', pco
    assert {o['casualty_operation'] for o in pco['options']} == {'remove_regular', 'reduce_elite', 'remove_elite'}
    assert 'Strider' in {l.name for l in (army_at(st, 'erech').leaders)}, \
        "printed order: Strider arrives before Shadow allocates casualties"
    s2 = _resolve_casualty(st, 'reduce_elite', Nation.SAURON)
    assert s2.pending_card_option['cas_hits'] == 1
    assert any(getattr(l, 'name', None) == 'Strider' for l in army_at(s2, 'erech').leaders)
    s2 = _resolve_casualty(s2, 'reduce_elite', Nation.SAURON)
    assert not units_of(s2, 'erech', Nation.SAURON) \
        or len([u for u in units_of(s2, 'erech', Nation.SAURON)
                if u.kind == UnitKind.ELITE]) == 0, \
        "two downgrades: no Sauron Elite survives at Erech"
    # the card's remaining legs (forced retreat -> recruit)
    # resume after the allocation; the Shadow-decided retreat may itself be
    # a pending
    pco2 = getattr(s2, 'pending_card_option', None)
    if (pco2 is not None and pco2.get('card_name') == 'Dead Men of Dunharrow'
            and pco2.get('decider') == 'sh'):
        assert all('destination' in o for o in pco2['options']), pco2
        s2 = ai.apply_action(s2, opt_action(pco2, 0))
    er = army_at(s2, 'erech')
    assert er is not None and any(l.name == 'Strider' for l in er.leaders), \
        "Strider remains at Erech through Shadow allocation and retreat"
    assert not units_of(s2, 'erech', Nation.SAURON), \
        "the surviving Shadow army retreated out of Erech"
    recruits = s2.pending_card_option
    assert recruits and recruits['decider'] == 'fp', recruits
    assert {o.get('recruit_count') for o in recruits['options']} == {0, 1, 2, 3}
    assert not units_of(s2, 'erech', Nation.GONDOR), \
        "optional recruits wait until the Free Peoples chooses their count"
    idx = next(i for i, o in enumerate(recruits['options'])
               if o.get('recruit_count') == 3)
    s2 = ai.apply_action(s2, opt_action(recruits, idx))
    assert units_of(s2, 'erech', Nation.GONDOR), \
        "the recruit leg resumed (Gondor regulars at Erech)"


def test_1_254_return_to_valinor_chains_owner_allocations_per_stronghold():
    st = setup_game()
    for r in ('rivendell', 'lorien', 'grey havens', 'woodland realm'):
        set_army(st, r, R(Nation.ELVES, 1) + E(Nation.ELVES, 2))
    st.besieged_strongholds = set()
    _, card = card_by_name('Return to Valinor')
    saved = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [6] * n
    try:
        res = ee._effect_return_to_valinor(st, card, {'_from_pending': True})
        pco = st.pending_card_option
        assert pco is not None and pco.get('card_casualty'), \
            "1.254: RtV must open the FP owner's allocation: %r" % (res,)
        assert pco['decider'] == 'fp', pco
        regions = [pco['cas_region']]
        s = st
        for _ in range(6):
            s = _resolve_casualty(s, 'remove_regular', Nation.ELVES, allow_roll=True)
            assert s.pending_card_option['cas_hits'] == 2
            s = _resolve_casualty(s, 'remove_elite', Nation.ELVES, allow_roll=True)
            p = getattr(s, 'pending_card_option', None)
            if p is None or not p.get('card_casualty'):
                break
            regions.append(p['cas_region'])
    finally:
        ee._d6_batch = saved
    assert sorted(regions) == sorted(
        ['rivendell', 'lorien', 'grey havens', 'woodland realm']), \
        "every stronghold's hits resolve through its own allocation: %r" % (
            regions,)
    for r in ('rivendell', 'lorien', 'grey havens', 'woodland realm'):
        assert len(units_of(s, r, Nation.ELVES)) == 1, \
            "3 hits: explicit Regular removal followed by whole-Elite removal at %s" % r


def test_1_254_faramirs_rangers_and_ents_awake_route_casualties():
    # Faramir's Rangers: allocation pending, then the recruit leg resumes
    st = setup_game()
    set_army(st, 'south ithilien', R(Nation.SAURON, 1) + E(Nation.SAURON, 2))
    set_army(st, 'osgiliath', R(Nation.GONDOR, 1))
    _, card = card_by_name("Faramir's Rangers")
    saved = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [6, 6, 1]     # 2 hits
    try:
        res = ee._effect_faramirs_rangers(
            st, card, {'target_region': 'south ithilien',
                       '_from_pending': True})
    finally:
        ee._d6_batch = saved
    pco = st.pending_card_option
    assert pco is not None and pco.get('card_casualty') \
        and pco['decider'] == 'sh', \
        "1.254: Faramir's Rangers must open the Shadow allocation: %r" % (res,)
    assert len(units_of(st, 'osgiliath', Nation.GONDOR)) == 1, \
        "the recruit leg waits for the allocation"
    s2 = _resolve_casualty(st, 'reduce_elite', Nation.SAURON)
    assert s2.pending_card_option['cas_hits'] == 1
    s2 = _resolve_casualty(s2, 'remove_regular', Nation.SAURON)
    recruits = s2.pending_card_option
    assert recruits and recruits.get('faramir_recruit') and recruits['decider'] == 'fp', \
        "after Shadow casualties, the human chooses the Gondor unit kind"
    assert len(units_of(s2, 'osgiliath', Nation.GONDOR)) == 1, \
        "no unit is placed before the human recruitment choice"
    idx = next(i for i, o in enumerate(recruits['options']) if o.get('unit_type') == 'regular')
    s2 = ai.apply_action(s2, opt_action(recruits, idx))
    assert len(units_of(s2, 'osgiliath', Nation.GONDOR)) == 2, \
        "Faramir's recruit leg must resume after the allocation"
    assert all(u.kind == UnitKind.REGULAR for u in units_of(s2, 'osgiliath', Nation.GONDOR)), \
        "the chosen Regular cannot silently become an Elite"
    # The Ents Awake: allocation pending, chain/Saruman leg resumes
    st = setup_game()
    set_army(st, 'orthanc', R(Nation.ISENGARD, 1) + E(Nation.ISENGARD, 2))
    _, card = card_by_name("The Ents Awake: Treebeard")
    ee._d6_batch = lambda n, ctx='card effect': [4, 4, 1]     # 2 hits
    try:
        res = ee._effect_ents_awake_common(st, card, {'_from_pending': True})
    finally:
        ee._d6_batch = saved
    pco = st.pending_card_option
    assert pco is not None and pco.get('card_casualty') \
        and pco['cas_region'] == 'orthanc', \
        "1.254: Ents Awake must open the Shadow allocation: %r" % (res,)
    s2 = _resolve_casualty(st, 'remove_regular', Nation.ISENGARD)
    isen = units_of(s2, 'orthanc', Nation.ISENGARD)
    # The selected Regular death leaves one hit and only one unit kind /
    # nation available, so the remaining Elite reduction resolves automatically.
    assert sorted(u.kind.value for u in isen) == sorted(
        [UnitKind.ELITE.value, UnitKind.REGULAR.value]), \
        "2 hits absorbed as 1 kill + 1 downgrade: %r" % (
            [(u.kind.value) for u in isen],)


# ---------------------------------------------------------------------------
# EE1-09  a raising handler is LOUD (never a swallowed half-applied play)
# ---------------------------------------------------------------------------

def test_ee1_09_handler_exception_propagates_out_of_play_event_card():
    st = setup_game()
    cid = "FREE_PEOPLES_Character_Bilbo's_Song"
    if cid not in st.hands[FP].hand:
        st.hands[FP].hand.append(cid)
    assert ee.check_event_requirement(st, cid)[0]
    key = cards.get_card(cid).card_name.lower()
    saved = ee.CARD_EFFECT_HANDLERS.get(key)

    def boom(state, card, choices):
        state.fellowship.corruption += 5
        raise RuntimeError('synthetic handler failure')
    ee.CARD_EFFECT_HANDLERS[key] = boom
    try:
        try:
            res = ee.play_event_card(st, FP, cid, choices={})
        except RuntimeError as exc:
            assert 'synthetic' in str(exc)
        else:
            raise AssertionError(
                "handler exception swallowed: %r (card in hand: %s, "
                "corruption %d)" % (res, cid in st.hands[FP].hand,
                                    st.fellowship.corruption))
    finally:
        if saved is not None:
            ee.CARD_EFFECT_HANDLERS[key] = saved
        else:
            ee.CARD_EFFECT_HANDLERS.pop(key, None)


# ---------------------------------------------------------------------------
# LOW  EE1-10 / EE1-11 / EE1-12 / EE1-13 / EE2-11 / EE2-12
# ---------------------------------------------------------------------------

def test_ee1_10_gwaihir_moves_a_board_group_and_leaves_no_empty_entry():
    st = setup_game()
    gg = take_companion(st, 'Gandalf the Grey')
    lg = take_companion(st, 'Legolas')
    set_army(st, 'fangorn', [], [gg, lg])
    opts = co.get_card_options(st, FP, 'Gwaihir the Windlord')
    grp = [o for o in opts if o.get('companion_action') == 'move_group'
           and set(o.get('companions') or []) == {'Gandalf the Grey',
                                                  'Legolas'}]
    assert grp, "Gwaihir offers no board-group move (card: 'one group')"
    _, card = card_by_name('Gwaihir the Windlord')
    res = ee._effect_gwaihir(st, card, dict(grp[0], _from_pending=True))
    dest = grp[0]['destination']
    names = {l.name for l in army_at(st, dest).leaders}
    assert {'Gandalf the Grey', 'Legolas'} <= names, (res, names)
    assert army_at(st, 'fangorn') is None, \
        "an empty Army entry was left behind at Fangorn"
    # single-companion board move: no empty entry either
    st = setup_game()
    lg = take_companion(st, 'Legolas')
    set_army(st, 'fangorn', [], [lg])
    single = [o for o in co.get_card_options(st, FP, 'We Prove Swifter')
              if o.get('companion_action') == 'move_Legolas']
    assert single
    _, wps = card_by_name('We Prove Swifter')
    ee._effect_we_prove_swifter(st, wps, dict(single[0], _from_pending=True))
    assert army_at(st, 'fangorn') is None, \
        "WPS move_ arm left an empty Army entry at Fangorn"


def test_ee1_11_nazgul_strike_negative_verdict_is_stated():
    st = setup_game()
    _, card = card_by_name('Nazgûl Strike!')
    res = {'effects': []}
    ee._nazgul_card_post(st, card, 'strike', res)
    assert getattr(st, 'pending_card_option', None) is None
    assert any('no Nazgul' in e for e in res['effects']), \
        "the skipped either/or must be stated: %r" % (res,)


def test_ee1_12_besieged_destination_bfs_stops_at_a_shadow_stronghold():
    st = lorien_siege()
    assert ai._is_shadow_controlled_stronghold(st, 'moria')
    assert not ee._dest_ok_from(st, 'hollin', 'lorien', 3,
                                allow_besieged=True), \
        "Hollin -> Moria -> Dimrill Dale -> Lorien passes THROUGH Moria"
    assert ee._dest_ok_from(st, 'dimrill dale', 'lorien', 1,
                            allow_besieged=True), \
        "the adjacent besieged stronghold is still a legal END region"
    assert ee._dest_ok_from(st, 'hollin', 'moria', 1, allow_besieged=True), \
        "a Shadow stronghold may be ENTERED as the end region"


def test_ee1_13_challenge_all_eyes_books_the_elimination():
    st = setup_game()
    strd = take_companion(st, 'Strider')
    set_army(st, 'minas tirith', R(Nation.GONDOR, 1), [strd])
    st.hunt_pool = [gs.HuntTile(damage=0, is_eye=True) for _ in range(3)]
    st.hunt_discard = []
    st.hands[FP].hand = ['FREE_PEOPLES_Character_Athelas']
    table_card(st, 'Worn with Sorrow and Toil')
    _, card = card_by_name('Challenge of the King')
    res = ee._effect_challenge_of_king(st, card, {})
    assert 'Strider' not in board_names(st), res
    assert st.hands[FP].hand == [], \
        "the companion-casualty hooks (WWSAT) did not run: %r" % (res,)


def test_ee2_11_dreadful_spells_needs_a_shadow_army_not_a_lone_nazgul():
    st = setup_game()
    for r in ('minas morgul', 'osgiliath', 'north ithilien',
              'south ithilien', 'minas tirith', 'dagorlad', 'dead marshes'):
        clear(st, r)
    mm = set_army(st, 'minas morgul', R(Nation.SAURON, 1))
    mm.leaders.append(gs.Nazgul(Nation.SAURON))
    set_army(st, 'north ithilien', R(Nation.GONDOR, 1))
    dg = set_army(st, 'dagorlad', [])
    dg.leaders = [gs.Nazgul(Nation.SAURON) for _ in range(3)]
    set_army(st, 'dead marshes', R(Nation.GONDOR, 1))
    cid, card = card_by_name('Dreadful Spells')
    assert ee.check_event_requirement(st, cid)[0]
    opts = co.get_card_options(st, SH, 'Dreadful Spells')
    assert not any(o.get('target_region') == 'dagorlad' for o in opts), \
        "a unit-less Nazgul entry offered as 'that Army': %r" % (opts,)
    assert any(o.get('target_region') == 'minas morgul' for o in opts)
    res = ee._effect_dreadful_spells(
        st, card, {'target_region': 'dagorlad',
                   'target_fp_army': 'dead marshes'})
    assert not any('Rolled' in e for e in res['effects']), \
        "the handler rolled from a lone Nazgul entry: %r" % (res,)


def test_ee2_12_nazgul_teleport_respects_the_fp_stronghold_rule():
    st = setup_game()
    st.fellowship.region = st.board.region('rivendell')
    st.fellowship.progress = 1
    dg = set_army(st, 'dagorlad', [])
    dg.leaders = [gs.Nazgul(Nation.SAURON), gs.Nazgul(Nation.SAURON)]
    res = {'effects': []}
    n = ee._move_all_nazgul_to_fellowship(st, res)
    riv = army_at(st, 'rivendell')
    assert n == 0 and not any(isinstance(l, gs.Nazgul)
                              for l in (riv.leaders if riv else [])), \
        "Nazgul teleported into an unbesieged FP stronghold: %r" % (res,)
    # control: an open region is reachable (the setup Nazgul come too)
    st = setup_game()
    st.fellowship.region = st.board.region('hollin')
    dg = set_army(st, 'dagorlad', [])
    dg.leaders = [gs.Nazgul(Nation.SAURON), gs.Nazgul(Nation.SAURON)]
    res = {'effects': []}
    assert ee._move_all_nazgul_to_fellowship(st, res) >= 2, res
    assert not any(isinstance(l, gs.Nazgul)
                   for l in (army_at(st, 'dagorlad').leaders
                             if army_at(st, 'dagorlad') else []))


# ---------------------------------------------------------------------------
# GL handoff item 3 (GL-4 class): the Foul Thing post-pick remainder runs
# game_logic's hunt chain (new guide's ability live), not inline corruption
# ---------------------------------------------------------------------------

def test_gl3_foul_thing_remainder_continues_with_the_new_guide():
    st = setup_game()
    st.fellowship.hidden = True
    st.fellowship.region = st.board.region('hollin')
    merry = next(c for c in st.fellowship.companions if c.name == 'Meriadoc')
    st.fellowship.companions = [merry]          # Merry guides; Gollum next
    st.fellowship.guide = merry
    st.pending_hunt = None
    st._pending_hunt_damage = 0
    c0 = st.fellowship.corruption
    res = ee._foul_apply_after_reduction(st, {'victim': 'Meriadoc'}, 3)
    assert not any(c.name == 'Meriadoc' for c in st.fellowship.companions)
    g = st.fellowship.guide
    assert g is not None and g.name == 'Gollum' \
        and getattr(g, 'guide_ability', None) == 'no_reveal_on_damage', g
    assert [c.name for c in (getattr(st, '_hobbit_place_queue', None)
                             or [])] == ['Meriadoc'], \
        "the drawn hobbit is QUEUED for placement after the chain: %r" % (res,)
    ph = getattr(st, 'pending_hunt', None)
    assert ph is not None and ph.get('phase') == 'hunt_gollum' \
        and int(ph.get('damage', 0)) == 2, \
        "the 2-point remainder must offer the NEW guide's (Gollum) reveal: " \
        "pending_hunt=%r effects=%r" % (ph, res)
    assert st.fellowship.corruption == c0, \
        "the remainder was applied INLINE as corruption (+%d)" % (
            st.fellowship.corruption - c0)
    # decline the reveal: the remainder lands, then the placement opens
    gl.resolve_hunt_decision(st, 'hunt_gollum_no_reveal')
    assert st.fellowship.corruption == c0 + 2
    ph2 = st.pending_hunt
    assert ph2 is not None and ph2.get('phase') == 'hobbit_place' \
        and ph2.get('companion') == 'Meriadoc', ph2
    # control: a non-hobbit victim with no reducing next guide - the
    # remainder is confronted (corruption) and the chain closes
    st2 = setup_game()
    st2.fellowship.hidden = True
    bor = next(c for c in st2.fellowship.companions if c.name == 'Boromir')
    leg = next(c for c in st2.fellowship.companions if c.name == 'Legolas')
    st2.fellowship.companions = [bor, leg]
    st2.fellowship.guide = bor
    st2.pending_hunt = None
    st2._pending_hunt_damage = 0
    c0 = st2.fellowship.corruption
    ee._foul_apply_after_reduction(st2, {'victim': 'Boromir'}, 3)
    assert not any(c.name == 'Boromir' for c in st2.fellowship.companions)
    assert st2.fellowship.guide is leg
    assert st2.fellowship.corruption == c0 + 1
    assert getattr(st2, 'pending_hunt', None) is None


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

TESTS = [v for k, v in sorted(globals().items()) if k.startswith('test_')]


def main():
    nfail = 0
    for fn in TESTS:
        try:
            fn()
            print("  [PASS] %s" % fn.__name__)
        except Exception:
            nfail += 1
            print("  [FAIL] %s" % fn.__name__)
            for line in traceback.format_exc().splitlines()[-4:]:
                print("         " + line)
    print("TOTAL=%d FAIL=%d" % (len(TESTS), nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
