# -*- coding: ascii -*-
"""Regression pins for the 2026-08-29 audit fix wave, owner AI
(src/ai.py + src/engine_play.py).

One plain test_* function per finding; every pin was run RED against
the unfixed tree before the fix landed (the FIX_REPORT quotes the RED
line).  Findings: LA-1/2/10 (Elven-Ring waiver face), LA-3 (companion
siege bar = friendly stronghold only), LA-4 -> ruling 1.251, LA-5 ->
ruling 1.252, LA-6 (companion ability on any die), LA-7 -> ruling
1.253, LA-8 (partial leader splits), LA-9 (loud swallows), EX-1/EX-4
(handler raise = rejected action), EX-2/EX-5 (turn crossing: FP first,
battle marker reset), EX-6 (dropped pendings are loud), EX-7
(transactional recruit pool check), MC2/MC3 (blind only while the
commit is unrevealed; the encoding hides it), MC5 (proofs ignore
hidden-hand card edges), MC6 (empty menu passes to the opponent).
"""
import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import board
import game_state as gs
import game_logic
import ai
from board import sanitize
from game_state import (Side, Nation, UnitKind, Unit, Leader, Nazgul,
                        ActionDieFace as Face)

FP = Side.FREE_PEOPLES
SH = Side.SHADOW


# ---------------------------------------------------------------- helpers
def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for attr in ('_declare_phase', '_eye_allocation_phase'):
        setattr(st, attr, False)
    st.pending_discard = None
    return st


def set_dice(st, side, faces):
    st.dice[side] = []
    for f in faces:
        d = gs.ActionDie(faces=[f], side=side)
        d.current = f
        st.dice[side].append(d)


def place(st, rk, nation, regulars=0, elites=0, leaders=0, nazgul=0,
          clear=False):
    a = game_logic.get_or_create_army(st, rk)
    if clear:
        a.units.clear()
        a.leaders.clear()
    for _ in range(regulars):
        a.units.append(Unit(nation=nation, kind=UnitKind.REGULAR))
    for _ in range(elites):
        a.units.append(Unit(nation=nation, kind=UnitKind.ELITE))
    for _ in range(leaders):
        a.leaders.append(Leader(nation=nation))
    for _ in range(nazgul):
        a.leaders.append(Nazgul(nation=Nation.SAURON))
    return a


def take_companion(st, name):
    for c in list(st.fellowship.companions):
        if c.name == name:
            st.fellowship.companions.remove(c)
            return c
    raise KeyError(name)


def acts(st, side, atype=None, **flt):
    out = []
    for a in ai.get_legal_actions(st, side):
        if atype and a.action_type != atype:
            continue
        ok = True
        for k, v in flt.items():
            if sanitize(str(a.params.get(k, ''))) != sanitize(str(v)):
                ok = False
        if ok:
            out.append(a)
    return out


def army_at(st, rk):
    return next((v for k, v in st.armies.items() if sanitize(k) == rk),
                None)


def ring(st, side, die_idx, face):
    """Pay an Elven Ring THROUGH THE EXECUTOR (the ai.py site that
    records the requested face)."""
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='elven_ring',
        params={'target_die_idx': die_idx, 'new_face': face.value,
                'side': 'fp' if side == FP else 'sh'}))


def gondor_at_war(st):
    st.political.boxes[Nation.GONDOR] = 3
    st.political.active[Nation.GONDOR] = True


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the pin wants ANY raise
        return e
    return None


# ------------------------------------------------------------ LA-1 / LA-10
def test_la1_ring_char_die_no_leaderless_move():
    """Ring paid 'to Event' + a CHARACTER die: the die is NOT an Army
    die, so a leaderless army move/attack is not offered on it (the
    old menu widened every die to all four categories)."""
    st = setup_game()
    place(st, 'pelargir', Nation.GONDOR, regulars=3, clear=True)
    set_dice(st, FP, [Face.CHARACTER, Face.MUSTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    ring(st, FP, 0, Face.EVENT)
    assert game_logic.ring_waiver_active(st, FP), 'waiver must be up'
    mv = [a for a in acts(st, FP, 'army_move')
          if sanitize(str(a.params['from'])) == 'pelargir'
          and a.die_idx == 0]
    assert not mv, ('LA-1: ring-to-Event + CHARACTER die offered %d '
                    'leaderless army moves' % len(mv))
    # the EVENT category IS bought: an event action rides die 0
    ev = [a for a in acts(st, FP) if a.die_idx == 0
          and a.action_type in ('draw_card', 'play_event')]
    assert ev, 'the waived face (Event) must be fundable on die 0'


def test_la10_executor_refuses_leaderless_character_die():
    """Executor defence in depth: a hand-built leaderless army_move /
    army_attack on a CHARACTER die (no waiver) is refused, state
    untouched (rulebook 2427-2428)."""
    st = setup_game()
    place(st, 'pelargir', Nation.GONDOR, regulars=3, clear=True)
    place(st, 'lamedon', Nation.SOUTHRON_EASTERLINGS, regulars=2,
          clear=True)
    gondor_at_war(st)
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    e1 = raises(lambda: ai.apply_action(st, ai.Action(
        0, 'army_move', {'from': 'pelargir', 'to': 'lossarnach',
                         'side': 'fp'})))
    assert e1 is not None, \
        'LA-10: leaderless CHARACTER-die army_move executed'
    e2 = raises(lambda: ai.apply_action(st, ai.Action(
        0, 'army_attack', {'from': 'pelargir', 'to': 'lamedon',
                           'side': 'fp'})))
    assert e2 is not None, \
        'LA-10: leaderless CHARACTER-die army_attack executed'
    assert not st.dice[FP][0].used and army_at(st, 'lossarnach') is None


def test_la1_ring_char_die_leaderless_attack_spends():
    """Ring 'to Army' + CHARACTER die: the leaderless attack is legal
    (the die is treated AS an Army die) and the waiver is SPENT when
    the die is consumed - never a second widened action."""
    st = setup_game()
    place(st, 'pelargir', Nation.GONDOR, regulars=3, clear=True)
    place(st, 'lamedon', Nation.SOUTHRON_EASTERLINGS, regulars=2,
          clear=True)
    gondor_at_war(st)
    set_dice(st, FP, [Face.CHARACTER, Face.MUSTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    ring(st, FP, 0, Face.ARMY)
    atk = [a for a in acts(st, FP, 'army_attack')
           if sanitize(str(a.params['from'])) == 'pelargir'
           and a.die_idx == 0]
    assert atk, 'ring-to-Army must let the CHARACTER die attack leaderless'
    s2 = ai.apply_action(st, atk[0])
    assert s2.dice[FP][0].used, 'die 0 pays'
    assert not game_logic.ring_waiver_active(s2, FP), \
        'LA-1: waiver survived a waived attack (one ring bought two actions)'
    assert (s2.pending_combat or {}).get('phase'), 'battle opened'


# ------------------------------------------------------------------- LA-2
def test_la2_ring_to_army_on_event_die_double_move():
    """Ring 'to Army' on an EVENT die buys the Army die's TWO moves:
    the first move opens the second-move pending; the waiver is spent
    exactly once, when the die is finally consumed."""
    st = setup_game()
    place(st, 'pelargir', Nation.GONDOR, regulars=3, clear=True)
    set_dice(st, FP, [Face.EVENT, Face.MUSTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    ring(st, FP, 0, Face.ARMY)
    mv = [a for a in acts(st, FP, 'army_move')
          if sanitize(str(a.params['from'])) == 'pelargir'
          and a.die_idx == 0]
    assert mv, 'ring-to-Army must let the EVENT die move a leaderless army'
    s2 = ai.apply_action(st, mv[0])
    assert s2.pending_army_move_die == 0, \
        ('LA-2: ring-to-Army on an EVENT die gave ONE move '
         '(pending_army_move_die=%r)' % (s2.pending_army_move_die,))
    assert not s2.dice[FP][0].used, 'the Army die pays after move 2'
    types2 = {a.action_type for a in ai.get_legal_actions(s2, FP)}
    assert types2 <= {'army_move', 'pass'}, types2
    p = [a for a in ai.get_legal_actions(s2, FP) if a.action_type == 'pass']
    s3 = ai.apply_action(s2, p[0])
    assert s3.dice[FP][0].used, 'closing the pair consumes die 0'
    assert not game_logic.ring_waiver_active(s3, FP), \
        'LA-2: waiver survived the waived Army-die pair'
    # no second widened action rides the MUSTER die afterwards
    fm = [a for a in acts(s3, FP, 'fellowship_move') if a.die_idx == 1]
    assert not fm, 'a MUSTER die must not fund fellowship_move once spent'


def test_la2_ring_to_muster_double_recruit_still_works():
    """Pin (c): ring 'to Muster' on an ARMY die still buys the Muster
    die's double recruit (regression guard, never RED)."""
    st = setup_game()
    gondor_at_war(st)
    set_dice(st, FP, [Face.ARMY])
    set_dice(st, SH, [Face.ARMY] * 3)
    ring(st, FP, 0, Face.MUSTER)
    rc = [a for a in acts(st, FP, 'muster_recruit') if a.die_idx == 0]
    assert rc, 'ring-to-Muster must let the ARMY die recruit'
    s2 = ai.apply_action(st, rc[0])
    assert s2.pending_muster_die == 0, 'double recruit pending opens'
    p = [a for a in ai.get_legal_actions(s2, FP) if a.action_type == 'pass']
    s3 = ai.apply_action(s2, p[0])
    assert s3.dice[FP][0].used and not game_logic.ring_waiver_active(s3, FP)


def test_la1_waiver_face_survives_clone():
    st = setup_game()
    set_dice(st, FP, [Face.EVENT])
    ring(st, FP, 0, Face.ARMY)
    c = ai.clone_state(st)
    assert ai._ring_waiver_face(c, FP) == Face.ARMY, \
        'the recorded waiver face must survive clone_state'


# ------------------------------------------------------------------- LA-4
def test_la4_nazgul_ends_on_fp_field_army():
    """Ruling 1.251: a Nazgul may END its flight in a region holding an
    FP FIELD army; the only bar is an unbesieged FP-controlled
    stronghold."""
    st = setup_game()
    place(st, 'eastemnet', Nation.ROHAN, regulars=3, clear=True)
    place(st, 'dol guldur', Nation.SAURON, regulars=1, nazgul=1, clear=True)
    set_dice(st, SH, [Face.CHARACTER])
    set_dice(st, FP, [Face.CHARACTER])
    nz = acts(st, SH, 'nazgul_move', to='eastemnet')
    assert nz, 'LA-4 / 1.251: nazgul_move onto the Rohan field army not offered'
    assert not acts(st, SH, 'nazgul_move', to='minas tirith'), \
        'unbesieged FP stronghold stays barred'


# ------------------------------------------------------------------- LA-5
def test_la5_companion_ends_on_shadow_field_army():
    """Ruling 1.252: Shadow FIELD armies never inhibit companion
    movement - a companion may end on one (move, separation, hunt
    placement)."""
    st = setup_game()
    place(st, 'eastemnet', Nation.SAURON, regulars=3, clear=True)
    g = take_companion(st, 'Gandalf the Grey')
    game_logic.get_or_create_army(st, 'fangorn').leaders.append(g)
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    cm = acts(st, FP, 'companion_move', **{'from': 'fangorn', 'to': 'eastemnet'})
    assert cm, 'LA-5 / 1.252: companion_move onto a Shadow field army not offered'
    ok, why = ai.is_legal_companion_action(st, ai.Action(
        0, 'companion_move', {'from': 'fangorn', 'to': 'eastemnet'}))
    assert ok, why
    # separation onto a Shadow-army region adjacent to the Fellowship
    st = setup_game()
    cur = sanitize(game_logic.fellowship_region(st))
    tgt = next(sanitize(a) for a in st.board.adjacent(cur)
               if not board.regions.get(sanitize(a), {}).get('is_stronghold'))
    place(st, tgt, Nation.SAURON, regulars=2, clear=True)
    st.fellowship.progress = 1
    set_dice(st, FP, [Face.CHARACTER])
    a9, why9 = ai.companion_separate_check(st, 0, ('Strider',), tgt)
    assert a9 is not None, 'separation onto %s refused: %s' % (tgt, why9)
    # hunt-placed hobbit (the 7th site) too
    hp = ai._synthetic_decision_actions(
        st, FP, 'companion_move',
        {'hunt_placement': True, 'companion': 'Meriadoc'}, {})
    dests = {sanitize(str(a.params.get('to_region', ''))) for a in hp}
    assert tgt in dests, 'hunt placement onto a Shadow army not offered'


# ------------------------------------------------------------------- LA-3
def test_la3_siege_bar_is_friendly_stronghold_only():
    """The companion siege bar binds only a FRIENDLY (FP-controlled)
    stronghold besieged by the enemy (rulebook 2117-2119 / 3513-3516).
    An FP-besieged SHADOW stronghold is a normal stop-on-entry
    terminus: companions may leave it, join the besiegers there, and
    the Fellowship separating there may pick outward destinations."""
    adj = sorted(sanitize(x) for x in board.load_board().adjacent('dol guldur'))

    def dg_siege():
        st = setup_game()
        place(st, 'dol guldur', Nation.SAURON, regulars=2, clear=True)
        place(st, 'dol guldur', Nation.ELVES, regulars=4)
        st.besieged_strongholds = {'dol guldur'}
        set_dice(st, FP, [Face.CHARACTER])
        set_dice(st, SH, [Face.ARMY] * 3)
        return st

    st = dg_siege()
    g = take_companion(st, 'Gandalf the Grey')
    army_at(st, 'dol guldur').leaders.append(g)
    out = acts(st, FP, 'companion_move', **{'from': 'dol guldur'})
    assert out, 'LA-3: companion cannot LEAVE the Shadow stronghold it besieges'
    st = dg_siege()
    g = take_companion(st, 'Gandalf the Grey')
    game_logic.get_or_create_army(st, adj[0]).leaders.append(g)
    into = acts(st, FP, 'companion_move', **{'from': adj[0], 'to': 'dol guldur'})
    assert into, 'LA-3: companion cannot JOIN the FP besiegers at Dol Guldur'
    st = dg_siege()
    st.fellowship.region = st.board.region('dol guldur')
    st.fellowship.progress = 2
    seps = acts(st, FP, 'companion_separate')
    dests = {sanitize(str(a.params.get('to_region'))) for a in seps}
    assert dests - {'dol guldur'}, \
        'LA-3: separation from an FP-besieged Shadow stronghold offers only in-place'
    # CONTROL: the rule itself stands for a FRIENDLY besieged stronghold
    st = setup_game()
    place(st, 'minas tirith', Nation.GONDOR, regulars=2, clear=True)
    place(st, 'minas tirith', Nation.SAURON, regulars=5)
    st.besieged_strongholds = {'minas tirith'}
    g = take_companion(st, 'Gandalf the Grey')
    army_at(st, 'minas tirith').leaders.append(g)
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY] * 3)
    assert not acts(st, FP, 'companion_move', **{'from': 'minas tirith'}), \
        'a companion inside a besieged FRIENDLY stronghold can never leave'
    ok, why = ai.is_legal_companion_action(st, ai.Action(
        0, 'companion_move', {'from': 'minas tirith', 'to': 'osgiliath'}))
    assert not ok and 'besieged' in str(why).lower(), \
        'the explainer must name the siege: %r' % (why,)
    g2 = take_companion(st, 'Legolas')
    game_logic.get_or_create_army(st, 'osgiliath').leaders.append(g2)
    assert not acts(st, FP, 'companion_move', **{'from': 'osgiliath', 'to': 'minas tirith'}), \
        'a companion can never ENTER a besieged friendly stronghold'


# ------------------------------------------------------------------- LA-7
def test_la7_retreat_into_region_you_besiege():
    """Ruling 1.253: a besieged adjacent region is a legal retreat
    destination for the BESIEGER's side (Shadow in Osgiliath -> the
    Minas Tirith it besieges), never for the garrison's side (FP in
    Osgiliath cannot retreat through the siege lines)."""
    st = setup_game()
    place(st, 'minas tirith', Nation.GONDOR, regulars=2, clear=True)
    place(st, 'minas tirith', Nation.SAURON, regulars=5)
    st.besieged_strongholds = {'minas tirith'}
    place(st, 'osgiliath', Nation.SAURON, regulars=2, clear=True)
    place(st, 'pelargir', Nation.GONDOR, regulars=4, clear=True)
    pc = {'defender_region': 'osgiliath', 'defender_side': SH,
          'attacker_region': 'pelargir', 'attacker_side': FP}
    dests = {sanitize(a.params['to_region'])
             for a in ai._get_retreat_actions(st, pc, defending=True)}
    assert 'minas tirith' in dests, \
        'LA-7 / 1.253: the besieger cannot retreat into the region it besieges: %s' % sorted(dests)
    # the garrison's side: excluded
    st = setup_game()
    place(st, 'minas tirith', Nation.GONDOR, regulars=2, clear=True)
    place(st, 'minas tirith', Nation.SAURON, regulars=5)
    st.besieged_strongholds = {'minas tirith'}
    place(st, 'osgiliath', Nation.GONDOR, regulars=2, clear=True)
    place(st, 'south ithilien', Nation.SAURON, regulars=4, clear=True)
    pc = {'defender_region': 'osgiliath', 'defender_side': FP,
          'attacker_region': 'south ithilien', 'attacker_side': SH}
    dests = {sanitize(a.params['to_region'])
             for a in ai._get_retreat_actions(st, pc, defending=True)}
    assert 'minas tirith' not in dests, \
        '1.253: the garrison side must not bypass the siege by retreating in'


# ------------------------------------------------------------------- LA-6
def test_la6_companion_ability_advance_on_any_face():
    """Amendment 1.5: Boromir/Legolas/Gimli advance their nation with
    ANY die - Muster-family and Will faces included - and the ability
    survives Threats and Promises."""
    from game_state import PersistentCard, PersistentCardType
    st = setup_game()
    b = take_companion(st, 'Boromir')
    game_logic.get_or_create_army(st, 'minas tirith').leaders.append(b)
    st.political.boxes[Nation.GONDOR] = 1
    st.political.active[Nation.GONDOR] = False
    st.persistent_cards.add_card(PersistentCard(
        card_id='SH_TP_pin', card_name='Threats and Promises', side=SH,
        card_type=PersistentCardType.THREATS_AND_PROMISES))
    for face in (Face.MUSTER, Face.MUSTER_ARMY, Face.WILL_OF_WEST,
                 Face.CHARACTER):
        set_dice(st, FP, [face])
        set_dice(st, SH, [Face.ARMY] * 3)
        adv = [a for a in acts(st, FP, 'advance_political')
               if a.params.get('nation') == Nation.GONDOR
               and a.params.get('companion_ability') == 'Boromir']
        assert adv, ('LA-6: companion-ability advance missing on face %s '
                     'under Threats and Promises' % face.name)


# ------------------------------------------------------------------- LA-8
def test_la8_partial_leader_split():
    """Rulebook 2437-2441: when an army splits each Leader may move or
    stay; a Character-die split needs >= 1 leader figure with the
    moving part.  The menu carries per-count leader variants."""
    st = setup_game()
    place(st, 'pelargir', Nation.GONDOR, regulars=4, leaders=2, clear=True)
    set_dice(st, FP, [Face.ARMY])
    set_dice(st, SH, [Face.ARMY] * 3)
    mv = acts(st, FP, 'army_move', **{'from': 'pelargir', 'to': 'lamedon'})
    part = [a for a in mv if a.params.get('n_leaders') == 1]
    assert part, 'LA-8: no n_leaders=1 split variant on the Army die'
    set_dice(st, FP, [Face.CHARACTER])
    mv = acts(st, FP, 'army_move', **{'from': 'pelargir', 'to': 'lamedon'})
    part = [a for a in mv if a.params.get('n_leaders') == 1
            and a.params.get('n_regular') == 2]
    assert part, 'LA-8: no n_leaders=1 split variant on the Character die'
    assert all(a.params.get('take_leaders', True)
               and a.params.get('n_leaders', 1) >= 1 for a in mv), \
        'a Character-die move must keep >= 1 leader with the movers'
    s2 = ai.apply_action(st, part[0])
    assert len(army_at(s2, 'lamedon').leaders) == 1
    assert len(army_at(s2, 'pelargir').leaders) == 1
    assert len(army_at(s2, 'lamedon').units) == 2
    # the variants are distinguishable to the net
    full = [a for a in mv if a.params.get('n_leaders') is None
            and a.params.get('n_regular') == 2 and a.params.get('n_elite') == 0]
    assert full
    e1 = ai.encode_action(part[0], st, FP)
    e2 = ai.encode_action(full[0], st, FP)
    assert not np.array_equal(e1, e2), 'n_leaders variants encode identically'


# ------------------------------------------------------------------- LA-9
def test_la9_folly_option_error_is_loud():
    from game_state import PersistentCard, PersistentCardType
    import persistent_effects as pe
    st = setup_game()
    st.persistent_cards.add_card(PersistentCard(
        card_id="SHADOW_Strategy_Denethor's_Folly", card_name="Denethor's Folly",
        card_type=PersistentCardType.DENETHORS_FOLLY, side=SH))
    real = pe.get_denethors_folly_discard_options

    def boom(*a, **k):
        raise RuntimeError('simulated option-body failure')
    pe.get_denethors_folly_discard_options = boom
    try:
        err = raises(lambda: ai._get_table_kill_actions(st, FP, 0, Face.CHARACTER))
    finally:
        pe.get_denethors_folly_discard_options = real
    assert err is not None, 'LA-9: the Folly option error was swallowed'


def test_la9_reveal_stay_fails_closed():
    """A failure inside the 'stay in place' bar must never OFFER staying
    inside an unconquered FP settlement (rulebook 3403-3407)."""
    st = setup_game()
    st.fellowship.region = st.board.region('rivendell')
    st.fellowship.progress = 2
    ai._get_reveal_placement_actions(st)   # warm the reveal memo
    real = game_logic.is_shadow_region

    def boom(*a, **k):
        raise RuntimeError('simulated settlement-check failure')
    game_logic.is_shadow_region = boom
    try:
        try:
            got = ai._get_reveal_placement_actions(st)
        except Exception:
            got = None   # loud = closed
    finally:
        game_logic.is_shadow_region = real
    if got is not None:
        dests = {sanitize(str(a.params['region'])) for a in got}
        assert 'rivendell' not in dests, \
            'LA-9: reveal placement failed OPEN (stay inside Rivendell offered)'


# ------------------------------------------------------------------- EX-1
def test_ex1_card_option_resume_raise_rejects_action():
    import event_effects
    from cards import get_card
    st = setup_game()
    game_logic.draw_event_card(st, FP, 'character')
    cid = st.hands[FP].hand[0]
    st.pending_card_option = {
        'card_id': cid, 'card_name': get_card(cid).card_name,
        'side': 'fp', 'decider': 'fp', 'options': [{'x': 1}, {'x': 2}],
        'deck': 'character', 'die_face': 'EVENT'}
    real = event_effects.play_event_card

    def boom(*a, **k):
        raise RuntimeError('simulated handler failure')
    event_effects.play_event_card = boom
    try:
        err = raises(lambda: ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='card_option', params={'choices': {'x': 1}})))
    finally:
        event_effects.play_event_card = real
    assert err is not None, 'EX-1: the handler raise was swallowed'
    assert cid in st.hands[FP].hand, 'EX-1: the card was annihilated'


# ------------------------------------------------------------------- EX-4
def test_ex4_play_event_handler_raise_propagates():
    import event_effects
    st = setup_game()
    game_logic.draw_event_card(st, FP, 'character')
    cid = st.hands[FP].hand[0]
    set_dice(st, FP, [Face.EVENT])
    real = event_effects.play_event_card

    def half(state, side, card_id, choices=None):
        state.fellowship.corruption += 3
        raise RuntimeError('simulated mid-effect failure')
    event_effects.play_event_card = half
    try:
        err = raises(lambda: ai._execute_action(st, ai.Action(
            die_idx=0, action_type='play_event',
            params={'card_id': cid, 'side': 'fp', 'choices': {}})))
    finally:
        event_effects.play_event_card = real
    assert err is not None, \
        'EX-4: a half-applied event play was committed silently'
    # apply_action (clone) rejects the action and leaves the input alone
    st2 = setup_game()
    game_logic.draw_event_card(st2, FP, 'character')
    cid2 = st2.hands[FP].hand[0]
    set_dice(st2, FP, [Face.EVENT])
    event_effects.play_event_card = half
    try:
        err = raises(lambda: ai.apply_action(st2, ai.Action(
            die_idx=0, action_type='play_event',
            params={'card_id': cid2, 'side': 'fp', 'choices': {}})))
    finally:
        event_effects.play_event_card = real
    assert err is not None and st2.fellowship.corruption == 0 \
        and cid2 in st2.hands[FP].hand


# ------------------------------------------------------------- EX-2 / EX-5
def test_ex5_battle_marker_reset_at_turn_boundary():
    st = setup_game()
    st._battle_resolved_attacker = SH
    ai._safe_next_turn(st)
    assert getattr(st, '_battle_resolved_attacker', None) is None, \
        'EX-5: _battle_resolved_attacker survived _safe_next_turn'


def test_ex2_end_turn_crossing_starts_with_free_peoples():
    """After an in-tree end_turn edge the new turn's first actor is the
    Free Peoples (rulebook 1406-1407) - never alternation from the old
    turn's last actor or a stale battle marker."""
    st = setup_game()
    ai._safe_next_turn(st)
    st.pending_discard = None
    game_logic.roll_action_dice(st, pool=FP)
    game_logic.roll_action_dice(st, pool=SH)
    st._battle_resolved_attacker = FP      # stale marker from the old turn
    node = ai.MCTSNode(state=st, side=FP, parent=None,
                       action=ai.Action(die_idx=-1, action_type='end_turn',
                                        params={}))
    real = ai._expand_node
    ai._expand_node = lambda *a, **k: 0.0     # nets are not the subject
    try:
        ai._settle_new_node(node, FP, None, None, FP)
    finally:
        ai._expand_node = real
    assert node.side == FP, \
        'EX-2: the turn crossing handed the first action to %s' % node.side


# ------------------------------------------------------------------- EX-6
def test_ex6_turn_end_dropped_pending_is_loud():
    st = setup_game()
    st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
    st._pending_hunt_damage = 3
    turn = st.turn
    err = raises(lambda: ai._safe_next_turn(st))
    assert isinstance(err, RuntimeError) and 'unfinished_turn' in str(err)
    assert st.turn == turn and st._pending_hunt_damage == 3
    assert st.pending_hunt == {'damage': 3, 'phase': 'hunt_sacrifice'}


# ------------------------------------------------------------------- EX-7
def test_ex7_muster_recruit_pool_check_transactional():
    import state_invariants as si
    st = setup_game()
    gondor_at_war(st)
    set_dice(st, FP, [Face.MUSTER])
    real = ai._can_recruit
    ai._can_recruit = lambda *a, **k: False
    n0 = len(si.VIOLATIONS)
    try:
        n_before = len(army_at(st, 'minas tirith').units)
        ai._execute_action(st, ai.Action(
            die_idx=0, action_type='muster_recruit',
            params={'nation': Nation.GONDOR, 'region': 'Minas Tirith',
                    'type': 'regular'}))
    finally:
        ai._can_recruit = real
    assert st.pending_muster_die is None, \
        'EX-7: the double-recruit pending opened with nothing placed'
    assert not st.dice[FP][0].used
    assert len(army_at(st, 'minas tirith').units) == n_before
    assert any(v.get('code') == 'muster_pool_empty'
               for v in si.VIOLATIONS[n0:]), 'refusal must be loud'


# ------------------------------------------------------------- MC2 / MC3
def _commit_fixture():
    st = setup_game()
    x = st.sh_char_deck[0]
    st.sh_char_deck.remove(x)
    st.pending_combat = {
        'attacker_region': 'minas morgul', 'defender_region': 'osgiliath',
        'attacker_side': SH, 'defender_side': FP,
        'phase': 'defender_combat_card', 'deciding_side': FP,
        'is_siege': False, 'rounds_fought': 0, 'atk_combat_card': x}
    st.discard_face_up.setdefault('sh_character', []).append(x)
    twin = ai.clone_state(st)
    twin.discard_face_up['sh_character'].remove(x)
    return st, twin, x


def test_mc3_encoding_hides_unrevealed_commit():
    st, twin, x = _commit_fixture()
    assert np.array_equal(ai.encode_state(st, FP), ai.encode_state(twin, FP)), \
        'MC3: the defender training vector carries the unrevealed committed card'
    assert not np.array_equal(ai.encode_state(st, SH), ai.encode_state(twin, SH)), \
        'the attacker sees its own commit'
    # once revealed (defender committed too), the card is public
    st.pending_combat['def_combat_card'] = None
    st.pending_combat['phase'] = 'combat_effect_choice'
    twin.pending_combat['def_combat_card'] = None
    twin.pending_combat['phase'] = 'combat_effect_choice'
    assert not np.array_equal(ai.encode_state(st, FP), ai.encode_state(twin, FP)), \
        'a revealed card must show in the card-status plane'


def test_mc2_blinding_only_while_unrevealed():
    st, twin, x = _commit_fixture()
    assert ai._hidden_commit_card(st, FP) == x
    assert ai._hidden_commit_card(st, SH) is None
    ov = ai._SimOverlay(st, FP)
    assert ov.commit_cid is not None, 'unrevealed commit is sampled'
    st.pending_combat['def_combat_card'] = None
    st.pending_combat['phase'] = 'combat_effect_choice'
    assert ai._hidden_commit_card(st, FP) is None
    ov2 = ai._SimOverlay(st, FP)
    assert ov2.commit_cid is None, \
        'MC2: the overlay blinds a REVEALED attacker card (sampled %r)' % (ov2.commit_cid,)
    # K1 hidden (human) commit stays blinded past the defender's commit
    st.pending_combat['atk_hidden_cc'] = {'deck': 'character'}
    assert ai._hidden_commit_card(st, FP) == x


# ------------------------------------------------------------------- MC5
def test_mc5_proofs_ignore_hidden_hand_edges():
    root = ai.MCTSNode(state=None, side=FP)
    opp = ai.MCTSNode(state=None, side=SH, parent=root,
                      action=ai.Action(0, 'pass', {}))
    root.children.append(opp)
    c_card = ai.MCTSNode(state=None, side=FP, parent=opp,
                         action=ai.Action(0, 'play_event',
                                          {'card_id': 'SHADOW_Character_X'}))
    c_card.proven, c_card.proven_value = 'L', -1.0
    c_plain = ai.MCTSNode(state=None, side=FP, parent=opp,
                          action=ai.Action(0, 'army_move', {}))
    opp.children.extend([c_card, c_plain])
    ai._propagate_proofs([root, opp], FP)
    assert opp.proven is None, \
        'MC5: an opponent node was proven LOST on a card the opponent may not hold'
    # a non-card proven child still proves
    c_plain.proven, c_plain.proven_value = 'L', -1.0
    ai._propagate_proofs([root, opp], FP)
    assert opp.proven == 'L'


# ------------------------------------------------------------------- MC6
def test_mc6_empty_menu_passes_to_opponent():
    """Rulebook 1681-1683: when one side has no legal action left the
    opponent executes its remaining actions; the action phase does not
    end for both."""
    import engine_play as ep
    real = ai.get_legal_actions

    def starved(state, side):
        if side is FP and not getattr(state, '_declare_phase', False) \
                and getattr(state, 'pending_discard', None) is None \
                and state.pending_combat is None \
                and state.pending_hunt is None \
                and getattr(state, 'pending_card_option', None) is None:
            return []
        return real(state, side)
    ai.get_legal_actions = starved
    try:
        s = ep.EngineSession(seed=5, mcts_sims=1, max_turns=2)
        dp = s.current()
        guard = 0
        while dp is not None and dp.kind != 'action' and guard < 20:
            guard += 1
            s.submit(0, dp.side)
            dp = s.current()
        assert dp is not None and dp.kind == 'action', \
            'MC6: no action decision reached (turn %s)' % s.state.turn
        assert dp.side is SH and s.state.turn == 1, \
            'MC6: first action went to %s on turn %s' % (dp.side, s.state.turn)
    finally:
        ai.get_legal_actions = real


# -------------------------------------------------- CB handoff A (1.255)
def units_of(st, rk, side):
    return ai._alloc_side_units(st, rk, side)


def test_1255_field_wipe_captures_only_on_entry():
    """USER ruling 1.255: a field wipe from an ADJACENT region captures
    nothing; the settlement is captured when the victor ENTERS it
    (victor_advance).  RED evidence: CB_suite_live_driver.log
    'cap_1255_wipe' (the combat owner's rewritten leg) before the ai.py
    gates landed."""
    from test_live_engine_fixwave import (field_battle_state,
                                          decline_cards, seed_search)
    pre = field_battle_state(atk_reg=4, def_reg=1)

    def make():
        return decline_cards(pre)

    def wiped(p):
        return (units_of(p, 'west harondor', SH)
                and not units_of(p, 'pelargir', FP))
    seed, post = seed_search(make, wiped)
    assert post is not None, 'fixture: no seed wiped the garrison'
    assert 'pelargir' not in post.enemy_controlled[SH], \
        '1.255: captured at the wipe without entering'
    pc = post.pending_combat
    assert pc is not None and pc.get('phase') == 'victor_advance', pc
    advs = [a for a in ai.get_legal_actions(post, SH)
            if a.action_type == 'victor_advance'
            and int(a.params.get('n_regular', 0)) > 0]
    entered = ai.apply_action(post, max(advs, key=lambda a: int(
        a.params.get('n_regular', 0))))
    assert units_of(entered, 'pelargir', SH), 'the advance moved in'
    assert 'pelargir' in entered.enemy_controlled[SH], \
        '1.255: the capture must fire on ENTRY'


# -------------------------------------------------- CB handoff B (1.254)
def _deferred_round_fixture(atk_elites, def_units=(2, 2)):
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'west harondor', Nation.SAURON, regulars=3,
          elites=atk_elites, clear=True)
    place(st, 'pelargir', Nation.GONDOR, regulars=def_units[0],
          elites=def_units[1], clear=True)
    st.hands[SH].hand = []
    st.hands[FP].hand = []
    return st


def test_1254_wctk_post_hits_open_enemy_allocation():
    """USER ruling 1.254: We Come to Kill's Phase-9 hits against the
    defender are the DEFENDER's learned allocation (casualty_allocate),
    never a hardcoded preserve_elites apply."""
    import random
    st = _deferred_round_fixture(atk_elites=2)
    st.pending_combat = {
        'attacker_region': 'west harondor', 'defender_region': 'pelargir',
        'attacker_side': SH, 'defender_side': FP,
        'phase': 'casualty_allocation', 'deciding_side': FP,
        'is_siege': False, 'rounds_fought': 1, 'alloc_stage': None,
        '_p9': {'atk_card': 'SHADOW_Strategy_Half-orcs_and_Goblin-men',
                'def_card': None, 'atk_cancelled': False,
                'def_cancelled': False}}
    hit = None
    for seed in range(400):
        random.seed(seed)
        s = ai.clone_state(st)
        ai._complete_allocated_round(s, s.pending_combat)
        pc = s.pending_combat or {}
        if len(units_of(s, 'pelargir', FP)) < 4 \
                or pc.get('phase') == 'casualty_allocation':
            hit = s
            break
    assert hit is not None, 'fixture: no seed scored a WCtK hit'
    pc = hit.pending_combat or {}
    assert pc.get('phase') == 'casualty_allocation' \
        and pc.get('alloc_stage') == 'defender' \
        and int(pc.get('def_hits', 0)) >= 1 \
        and pc.get('deciding_side') == FP, \
        ('1.254: WCtK hits applied without the defender\'s allocation '
         '(pending=%r, defender units=%d)'
         % ({k: pc.get(k) for k in ('phase', 'alloc_stage', 'def_hits')},
            len(units_of(hit, 'pelargir', FP))))
    assert len(units_of(hit, 'pelargir', FP)) == 4, 'hits not yet applied'
    # the owner's pick resolves and the round finishes normally
    menu = [a for a in ai.get_legal_actions(hit, FP)
            if a.action_type == 'casualty_allocate']
    assert menu, 'casualty_allocate must be on the defender menu'
    done = ai.apply_action(hit, menu[0])
    assert len(units_of(done, 'pelargir', FP)) < 4, 'allocation applied'
    assert (done.pending_combat or {}).get('phase') != 'casualty_allocation'


def test_1254_onslaught_hits_open_enemy_allocation():
    """USER ruling 1.254: the Onslaught's enemy hits are the ENEMY
    owner's learned allocation."""
    import random
    st = _deferred_round_fixture(atk_elites=0)
    st.pending_combat = {
        'attacker_region': 'west harondor', 'defender_region': 'pelargir',
        'attacker_side': SH, 'defender_side': FP,
        'phase': 'combat_effect_choice', 'deciding_side': SH,
        'is_siege': False, 'rounds_fought': 1,
        'effect_choice': {'stage': 'post', 'owner': 'atk',
                          'effect': 'onslaught',
                          'card_id': 'SHADOW_Strategy_The_Fighting_Uruk-hai',
                          'max_count': 3}}
    hit = None
    for seed in range(400):
        random.seed(seed)
        s = ai.clone_state(st)
        ai._execute_action(s, ai.Action(
            die_idx=-1, action_type='combat_effect_choice',
            params={'count': 2}))
        pc = s.pending_combat or {}
        if len(units_of(s, 'pelargir', FP)) < 4 \
                or pc.get('phase') == 'casualty_allocation':
            hit = s
            break
    assert hit is not None, 'fixture: no seed scored an Onslaught hit'
    pc = hit.pending_combat or {}
    assert pc.get('phase') == 'casualty_allocation' \
        and pc.get('alloc_stage') == 'defender' \
        and int(pc.get('def_hits', 0)) >= 1, \
        ('1.254: Onslaught hits applied without the defender\'s '
         'allocation (pending=%r, defender units=%d)'
         % ({k: pc.get(k) for k in ('phase', 'alloc_stage', 'def_hits')},
            len(units_of(hit, 'pelargir', FP))))
    assert len(units_of(hit, 'west harondor', SH)) == 1, 'own sacrifice applied'
    assert len(units_of(hit, 'pelargir', FP)) == 4, 'hits not yet applied'
    menu = [a for a in ai.get_legal_actions(hit, FP)
            if a.action_type == 'casualty_allocate']
    done = ai.apply_action(hit, menu[0])
    assert len(units_of(done, 'pelargir', FP)) < 4
    assert (done.pending_combat or {}).get('phase') != 'combat_effect_choice'


# --------------------------------------------------- EE handoff A (EE1-09)
def test_ee_handler_raise_rejects_play_end_to_end():
    """event_effects.play_event_card now RE-RAISES a handler exception;
    the play_event executor must turn it into a REJECTED action: the
    input state untouched (card in hand, die unused, no partial
    mutation) - through the real play_event_card, no patch of it."""
    import event_effects as ee
    from cards import get_card
    st = setup_game()
    for _ in range(4):
        game_logic.draw_event_card(st, FP, 'character')
    set_dice(st, FP, [Face.EVENT])
    set_dice(st, SH, [Face.ARMY] * 3)
    plays = acts(st, FP, 'play_event')      # enumerated BEFORE the patch
    assert plays, 'fixture: no playable card'
    act = plays[0]
    name = get_card(act.params['card_id']).card_name.lower()
    orig = ee.CARD_EFFECT_HANDLERS.get(name)

    def half(state, card, choices):
        state.fellowship.corruption += 3
        raise RuntimeError('simulated handler failure')
    ee.CARD_EFFECT_HANDLERS[name] = half
    try:
        err = raises(lambda: ai.apply_action(st, act))
    finally:
        if orig is None:
            ee.CARD_EFFECT_HANDLERS.pop(name, None)
        else:
            ee.CARD_EFFECT_HANDLERS[name] = orig
    assert err is not None, 'the handler raise was swallowed by the executor'
    assert st.fellowship.corruption == 0, 'partial mutation leaked'
    assert act.params['card_id'] in st.hands[FP].hand, 'card annihilated'
    assert not st.dice[FP][0].used, 'die spent on a rejected play'


# --------------------------------------------- kernel 'error' fallback
def test_kernel_error_fallback_is_loud_and_strict():
    """The combat kernel's blanket 'error' fallback hid a dead kernel for
    11 days: it must stay counted, capture the FIRST error text for the
    meters, and re-raise under WOTR_KERNEL_STRICT=1."""
    import combat_dist as cd
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    mm = next(k for k in st.armies if sanitize(k) == 'minas morgul')
    osg = next(k for k in st.armies if sanitize(k) == 'osgiliath')
    set_dice(st, FP, [Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY])
    st.pending_combat = {'attacker_region': mm, 'defender_region': osg,
                         'attacker_side': SH, 'defender_side': FP,
                         'phase': 'defender_combat_card',
                         'deciding_side': FP, 'is_siege': False,
                         'rounds_fought': 0, 'atk_combat_card': None}
    a = ai.Action(die_idx=-1, action_type='combat_card_none', params={})
    s4, _rng = ai._apply_action_detect_rng(st, a)
    real = cd.run_with_plan

    def boom(*args, **kw):
        raise RuntimeError('simulated kernel failure')
    cd.run_with_plan = boom
    old_env = os.environ.pop('WOTR_KERNEL_STRICT', None)
    try:
        ai.KERNEL_ERROR_FIRST = None
        n0 = ai.KERNEL_STATS['fallback'].get('error', 0)
        root = ai.MCTSNode(state=ai.clone_state(st), side=FP)
        node = ai.MCTSNode(state=None, side=FP, parent=root, action=a,
                           prior=1.0)
        ok = ai._kernel_expand_combat(node, s4)
        assert ok is False, 'the fallback still returns False'
        assert ai.KERNEL_STATS['fallback'].get('error', 0) == n0 + 1, \
            'the error class must stay counted'
        assert ai.KERNEL_ERROR_FIRST and \
            'simulated kernel failure' in str(ai.KERNEL_ERROR_FIRST), \
            'first error text not captured: %r' % (ai.KERNEL_ERROR_FIRST,)
        assert 'simulated kernel failure' in str(
            ai.kernel_stats_summary().get('error_first')), \
            'kernel_stats_summary must expose the first error'
        os.environ['WOTR_KERNEL_STRICT'] = '1'
        node2 = ai.MCTSNode(state=None, side=FP, parent=root, action=a,
                            prior=1.0)
        err = raises(lambda: ai._kernel_expand_combat(node2, s4))
        assert isinstance(err, RuntimeError) \
            and 'simulated kernel failure' in str(err), \
            'WOTR_KERNEL_STRICT=1 must re-raise, got %r' % (err,)
    finally:
        cd.run_with_plan = real
        os.environ.pop('WOTR_KERNEL_STRICT', None)
        if old_env is not None:
            os.environ['WOTR_KERNEL_STRICT'] = old_env


TESTS = [
    test_1255_field_wipe_captures_only_on_entry,
    test_1254_wctk_post_hits_open_enemy_allocation,
    test_1254_onslaught_hits_open_enemy_allocation,
    test_ee_handler_raise_rejects_play_end_to_end,
    test_kernel_error_fallback_is_loud_and_strict,
    test_la1_ring_char_die_no_leaderless_move,
    test_la10_executor_refuses_leaderless_character_die,
    test_la1_ring_char_die_leaderless_attack_spends,
    test_la2_ring_to_army_on_event_die_double_move,
    test_la2_ring_to_muster_double_recruit_still_works,
    test_la1_waiver_face_survives_clone,
    test_la4_nazgul_ends_on_fp_field_army,
    test_la5_companion_ends_on_shadow_field_army,
    test_la3_siege_bar_is_friendly_stronghold_only,
    test_la7_retreat_into_region_you_besiege,
    test_la6_companion_ability_advance_on_any_face,
    test_la8_partial_leader_split,
    test_la9_folly_option_error_is_loud,
    test_la9_reveal_stay_fails_closed,
    test_ex1_card_option_resume_raise_rejects_action,
    test_ex4_play_event_handler_raise_propagates,
    test_ex5_battle_marker_reset_at_turn_boundary,
    test_ex2_end_turn_crossing_starts_with_free_peoples,
    test_ex6_turn_end_dropped_pending_is_loud,
    test_ex7_muster_recruit_pool_check_transactional,
    test_mc3_encoding_hides_unrevealed_commit,
    test_mc2_blinding_only_while_unrevealed,
    test_mc5_proofs_ignore_hidden_hand_edges,
    test_mc6_empty_menu_passes_to_opponent,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            msg = str(e).encode('ascii', 'replace').decode()
            print('FAIL %s: %s: %s' % (t.__name__, type(e).__name__, msg))
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
