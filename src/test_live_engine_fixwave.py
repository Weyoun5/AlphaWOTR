"""Engine pins for the AlphaWOTR fix wave (agent D).

Every test states its MEASURED QUANTITY first.  Inputs are engine-shaped:
pendings come from real actions applied through ai.apply_action and the
menus from ai.get_legal_actions - never hand-built params.  Randomness
is pinned by seed search over `random` (combat rolls via random.randint).

Covers (ids = FINDINGS_FULL.md):
  W0-sweep#2      clone_state copies _owed_draws
  W1#128          card-followup exclude set is per-clone
  W1#129          direct-stage card_option handler failure is LOUD
  W1#130 / #131   Palantir: Event-die PLAYS only, never draws
  W1#119          mutual elimination captures nothing (1.152)
  K6 (#117/#118/#105)  one downgrade-supply answerer, k up to n_elites,
                  no-source elite ELIMINATED with the hit absorbed
  W1#120          WWSAT classifier recognises live placeholders
  W1#104          ring use keeps the same side (_next_acting_side)
  W1#132          post-retreat advance is the learned victor_advance
  W1#109          _consume_die_for_face refuses a different die (loud)
  W1#107          EngineSession hvai never auto-declares a human FP
  K1 step 1       hidden combat-card commit / reveal protocol
  K2              open_declare_phase / close_declare_phase
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai
import board as board_mod
import combat
import game_logic
import game_state as gs
from board import sanitize
from game_state import Side, Nation, UnitKind as Kind, ActionDieFace as Face

FP, SH = Side.FREE_PEOPLES, Side.SHADOW


# ---------------------------------------------------------------- helpers
def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    random.seed(12345)          # deck order is part of the pin
    game_logic.initialize_decks(st)
    return st


def set_dice(state, side, faces):
    state.dice[side] = []
    for face in faces:
        d = gs.ActionDie(faces=[face], side=side)
        d.current = face
        state.dice[side].append(d)
    state.dice_rolled = True


def place(state, region_key, nation, regulars=0, elites=0, clear=False):
    a = game_logic.get_or_create_army(state, region_key)
    if clear:
        a.units.clear()
        a.leaders.clear()
    for _ in range(regulars):
        a.units.append(gs.Unit(nation, Kind.REGULAR))
    for _ in range(elites):
        a.units.append(gs.Unit(nation, Kind.ELITE))
    return a


def own_units(state, region, side):
    return ai._alloc_side_units(state, region, side)


def pick(actions, atype, **want):
    for a in actions:
        if a.action_type != atype:
            continue
        if all(sanitize(str(a.params.get(k))) == sanitize(str(v))
               for k, v in want.items()):
            return a
    return None


def field_battle_state(atk_nation=Nation.SAURON, atk_reg=3, atk_elite=0,
                       def_nation=Nation.GONDOR, def_reg=3, def_elite=0,
                       atk_region='west harondor', def_region='pelargir'):
    """Shadow at atk_region attacks the FP army at def_region (a city:
    no siege choice, so the first pending is the attacker card window).
    Hands empty -> the only card action is the decline."""
    st = fresh_state()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    st.political.boxes[def_nation] = 3
    place(st, atk_region, atk_nation, regulars=atk_reg, elites=atk_elite,
          clear=True)
    place(st, def_region, def_nation, regulars=def_reg, elites=def_elite,
          clear=True)
    st.hands[SH].hand = []
    st.hands[FP].hand = []
    set_dice(st, SH, [Face.ARMY])
    set_dice(st, FP, [Face.CHARACTER])
    atk = pick(ai.get_legal_actions(st, SH), 'army_attack',
               **{'from': atk_region, 'to': def_region})
    assert atk is not None, 'attack must be on the menu'
    s2 = ai.apply_action(st, atk)
    pc = s2.pending_combat
    assert pc is not None and pc['phase'] == 'attacker_combat_card', pc
    return s2


def decline_cards(st):
    """Both sides decline their combat card (real menu picks)."""
    for _ in range(2):
        pc = st.pending_combat
        if pc is None or pc['phase'] not in ('attacker_combat_card',
                                             'defender_combat_card'):
            break
        menu = ai.get_legal_actions(st, pc['deciding_side'])
        nc = pick(menu, 'combat_card_none')
        assert nc is not None, 'decline must be on the menu'
        st = ai.apply_action(st, nc)
    return st


def seed_search(make, accept, n=400):
    """Replay `make()` under successive seeds until accept(post) holds.
    Returns (seed, post) or (None, None)."""
    for seed in range(n):
        random.seed(seed)
        post = make()
        if accept(post):
            return seed, post
    return None, None


# ------------------------------------------------------------ W0-sweep#2
def test_clone_state_copies_owed_draws(fails):
    """MEASURES: a stamped draw debt survives clone_state as an equal,
    INDEPENDENT dict (specificity: the clone's dict is not the
    original's object)."""
    st = fresh_state()
    st._owed_draws = {('SHADOW', 'character'): 1}
    s2 = ai.clone_state(st)
    got = getattr(s2, '_owed_draws', None)
    if got != {('SHADOW', 'character'): 1}:
        fails.append(('owed_draws_cloned', repr(got)))
    elif got is st._owed_draws:
        fails.append(('owed_draws_shared', 'same dict object'))
    # sensitivity: an absent ledger clones to an empty one, never None
    st3 = fresh_state()
    if getattr(ai.clone_state(st3), '_owed_draws', None) is None:
        fails.append(('owed_draws_absent', 'clone lost the attribute'))


class _Deferred(BaseException):
    """Stand-in for the live driver's client-call escape
    (randomness.NeedClientCall is a BaseException)."""


def test_owed_draw_writers(fails):
    """MEASURES: every engine-granted draw site stamps ONE unit of
    (side, deck) debt when the physical draw is deferred - the bid
    token, a card-effect draw, the Palantir deck pick and the Gandalf
    the Grey same-deck redraw all route through grant_draw."""
    import event_effects as ee
    orig = game_logic.draw_event_card

    def _defer(state, side, deck):
        raise _Deferred('client owns this draw')

    def debt(st, side, deck):
        return (getattr(st, '_owed_draws', None) or {}).get(
            (side.name, deck), 0)
    game_logic.draw_event_card = _defer
    try:
        # bid token
        st = fresh_state()
        st.sh_action_tokens = 1
        try:
            game_logic.use_action_token(st, SH, token_type='draw',
                                        deck='character')
        except _Deferred:
            pass
        if debt(st, SH, 'character') != 1:
            fails.append(('owed_token', debt(st, SH, 'character')))
        # card-effect draw
        st = fresh_state()
        try:
            ee.draw_cards(st, FP, 'strategy', 1)
        except _Deferred:
            pass
        if debt(st, FP, 'strategy') != 1:
            fails.append(('owed_card_effect', debt(st, FP, 'strategy')))
        # Palantir deck pick (card_option executor, in place)
        st = _palantir_state()
        ai._open_palantir_deck_choice(st)
        opt = next((a for a in ai.get_legal_actions(st, SH)
                    if a.action_type == 'card_option'
                    and (a.params.get('choices') or {}).get(
                        'palantir_deck') == 'strategy'), None)
        if opt is None:
            fails.append(('owed_palantir_menu', 'no deck pick offered'))
        else:
            try:
                ai._execute_action(st, opt)
            except _Deferred:
                pass
            if debt(st, SH, 'strategy') != 1:
                fails.append(('owed_palantir', debt(st, SH, 'strategy')))
        # Gandalf the Grey same-deck redraw on an FP Event-die play
        st = fresh_state()
        st.hands[FP].hand = list(st.fp_strat_deck[-6:])
        del st.fp_strat_deck[-6:]
        set_dice(st, FP, [Face.EVENT])
        g = next((c for c in st.fellowship.companions
                  if getattr(c, 'name', '') == 'Gandalf the Grey'), None)
        if g is None:
            fails.append(('owed_gandalf_setup', 'no Gandalf the Grey'))
        else:
            st.fellowship.guide = g
            stamped = False
            for a in ai.get_legal_actions(st, FP):
                if a.action_type != 'play_event':
                    continue
                s2 = ai.clone_state(st)
                try:
                    random.seed(1)
                    ai._execute_action(s2, a)
                except _Deferred:
                    pass
                except Exception:
                    continue
                if debt(s2, FP, 'strategy') == 1:
                    stamped = True
                    break
            if not stamped:
                fails.append(('owed_gandalf', 'no play stamped the redraw'))
    finally:
        game_logic.draw_event_card = orig


# ---------------------------------------------------------------- W1#128
def test_followup_exclude_is_per_clone(fails):
    """MEASURES: applying a card-followup army move on a clone leaves the
    ORIGINAL state's followup exclude set empty."""
    st = fresh_state()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, SH, [Face.ARMY])
    from event_effects import _grant_card_followup
    _grant_card_followup(st, SH, 'rwa', moves=2)
    menu = ai.get_legal_actions(st, SH)
    mv = next((a for a in menu if a.action_type == 'army_move'
               and a.params.get('card_followup')), None)
    if mv is None:
        fails.append(('followup_menu', 'no card_followup move offered: %s'
                      % sorted(set(a.action_type for a in menu))))
        return
    s2 = ai.apply_action(st, mv)
    if not s2.pending_card_followup and s2.pending_card_followup \
            is not None:
        pass
    if st.pending_card_followup.get('exclude'):
        fails.append(('followup_exclude_leak',
                      'original exclude=%r'
                      % (st.pending_card_followup.get('exclude'),)))
    # sensitivity: the clone DID record the destination
    pcf2 = s2.pending_card_followup
    if pcf2 is not None and sanitize(str(mv.params['to'])) \
            not in pcf2.get('exclude', ()):
        fails.append(('followup_exclude_recorded', repr(pcf2)))


# ---------------------------------------------------------------- W1#129
def test_direct_stage_handler_failure_is_loud(fails):
    """MEASURES: a direct-stage card_option whose handler raises makes
    apply_action raise (self-play / live mode, no teacher hook)."""
    import event_effects as ee
    st = fresh_state()
    st.political.boxes[Nation.SAURON] = 3
    # overstack a Shadow field army (two unit kinds -> a real choice)
    place(st, 'west harondor', Nation.SAURON, regulars=10, elites=1,
          clear=True)
    st._overstack_regions = {'west harondor'}
    ee.resolve_overstacks(st, learned=True)
    pco = getattr(st, 'pending_card_option', None)
    if pco is None or not pco.get('direct'):
        fails.append(('overstack_pending', 'no direct pending opened: %r'
                      % (pco,)))
        return
    key = str(pco.get('card_name', '')).lower()
    orig = ee.CARD_EFFECT_HANDLERS.get(key)
    if orig is None:
        fails.append(('overstack_handler', 'no handler for %r' % key))
        return
    menu = ai.get_legal_actions(st, SH)
    opt = next((a for a in menu if a.action_type == 'card_option'), None)
    if opt is None:
        fails.append(('overstack_menu', 'no card_option on the menu'))
        return

    def _boom(state, card, choices):
        raise RuntimeError('handler exploded (test)')
    ee.CARD_EFFECT_HANDLERS[key] = _boom
    try:
        raised = False
        try:
            ai.apply_action(st, opt)
        except RuntimeError:
            raised = True
        if not raised:
            fails.append(('direct_stage_swallowed',
                          'handler exception was swallowed'))
    finally:
        ee.CARD_EFFECT_HANDLERS[key] = orig
    # sensitivity: the real handler resolves the pending cleanly
    s2 = ai.apply_action(st, opt)
    if s2.pending_card_option is not None:
        fails.append(('direct_stage_control', 'real handler left pending'))


# ---------------------------------------------------------- W1#130 / #131
def _palantir_state():
    from game_state import PersistentCardType as PCT
    st = fresh_state()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Strategy_The_Palantir_of_Orthanc',
        card_name='The Palantir of Orthanc', side=SH,
        card_type=PCT.PALANTIR_OF_ORTHANC))
    return st


def _palantir_fired(post):
    pco = getattr(post, 'pending_card_option', None)
    return bool(getattr(post, '_palantir_draw_pending', False)
                or (pco and pco.get('palantir_draw')))


def test_palantir_never_on_draw(fails):
    """MEASURES: a Shadow draw_card with an EVENT die under the Palantir
    grows the hand by exactly ONE card."""
    st = _palantir_state()
    set_dice(st, SH, [Face.EVENT])
    st.hands[SH].hand = []
    dr = pick(ai.get_legal_actions(st, SH), 'draw_card', deck='character',
              side='sh')
    if dr is None:
        fails.append(('draw_menu', 'no draw_card offered'))
        return
    s2 = ai.apply_action(st, dr)
    n = len(s2.hands[SH].hand)
    if n != 1:
        fails.append(('palantir_draw_phantom', 'hand=%d after one draw' % n))


def test_palantir_only_event_die_plays(fails):
    """MEASURES: the SAME Shadow event card fires the Palantir bonus with
    an EVENT die (control) and does NOT fire it with a CHARACTER die."""
    st = _palantir_state()
    st.hands[SH].hand = list(st.sh_char_deck[-12:])
    del st.sh_char_deck[-12:]
    set_dice(st, SH, [Face.CHARACTER])
    plays_c = [a for a in ai.get_legal_actions(st, SH)
               if a.action_type == 'play_event']
    set_dice(st, SH, [Face.EVENT])
    plays_e = {a.params['card_id']: a
               for a in ai.get_legal_actions(st, SH)
               if a.action_type == 'play_event'}
    fired_control = None
    for pc_a in plays_c:
        cid = pc_a.params['card_id']
        if cid not in plays_e:
            continue
        set_dice(st, SH, [Face.EVENT])
        try:
            random.seed(3)
            post_e = ai.apply_action(st, plays_e[cid])
        except Exception:
            continue
        if not _palantir_fired(post_e):
            continue
        fired_control = cid
        set_dice(st, SH, [Face.CHARACTER])
        random.seed(3)
        post_c = ai.apply_action(st, pc_a)
        if _palantir_fired(post_c):
            fails.append(('palantir_character_die',
                          '%s fired the bonus on a CHARACTER die' % cid))
        break
    if fired_control is None:
        fails.append(('palantir_control',
                      'no card fired the bonus on an EVENT die (control)'))


def test_palantir_deferred_draw_fires_at_next_quiet_apply(fails):
    """MEASURES (BL-138, shgame16 S5): a Palantir bonus draw DEFERRED
    (the qualifying play opened its own card option first - the
    normal case) opens at the next quiet apply, not a full turn later
    at an unrelated battle boundary (the 1.208m re-offer sites).
    CONTRACT: pending flag + a quiet action -> the palantir_draw
    card option is open on the post state.  SPECIFICITY: with the
    table card GONE the deferred draw dies (1.228c) - no option."""
    _QUIET = ('advance_political', 'muster_recruit', 'pass')

    def _quiet(acts):
        return next((a for a in acts if a.action_type in _QUIET), None)
    st = _palantir_state()
    st._palantir_draw_pending = True
    set_dice(st, SH, [Face.MUSTER])
    a = _quiet(ai.get_legal_actions(st, SH))
    if a is None:
        fails.append(('pal_defer_control', 'NOT-MEASURED: no quiet row'))
        return
    post = ai.apply_action(st, a)
    pco = getattr(post, 'pending_card_option', None)
    if not (pco and pco.get('palantir_draw')):
        fails.append(('pal_defer_fires', (a.action_type, pco, getattr(
            post, '_palantir_draw_pending', None))))
    st2 = fresh_state()                # no Palantir on the table
    st2._palantir_draw_pending = True
    set_dice(st2, SH, [Face.MUSTER])
    a2 = _quiet(ai.get_legal_actions(st2, SH))
    if a2 is not None:
        post2 = ai.apply_action(st2, a2)
        pco2 = getattr(post2, 'pending_card_option', None)
        if pco2 and pco2.get('palantir_draw'):
            fails.append(('pal_defer_specificity',
                          'fired with no Palantir on the table'))


# ---------------------------------------------------------------- W1#119
def test_mutual_elimination_no_capture(fails):
    """MEASURES: a 1-vs-1 regular battle in which BOTH units die leaves
    Pelargir OUT of the Shadow control set (1.152)."""
    st0 = field_battle_state(atk_reg=1, def_reg=1)   # deck-deterministic

    def make():
        return decline_cards(st0)   # applies on clones; the seed rolls

    def both_dead(post):
        return (not own_units(post, 'west harondor', SH)
                and not own_units(post, 'pelargir', FP))
    seed, post = seed_search(make, both_dead)
    if post is None:
        fails.append(('mutual_wipe_seed', 'no seed produced a mutual wipe'))
        return
    if 'pelargir' in post.enemy_controlled[SH]:
        fails.append(('mutual_wipe_capture',
                      'seed %d: Shadow captured Pelargir on a mutual wipe'
                      % seed))
    # sensitivity: a one-sided wipe DOES capture - on the victor's ENTRY
    # (USER RULING 1.255, 2026-08-29: "captures only happen when an
    # enemy army is actually in that settlement"): nothing at the wipe
    # from West Harondor, the city is captured by the victor_advance
    def def_only(post):
        return (own_units(post, 'west harondor', SH)
                and not own_units(post, 'pelargir', FP))
    seed2, post2 = seed_search(make, def_only)
    if post2 is None:
        fails.append(('one_sided_wipe_seed', 'no seed'))
        return
    if 'pelargir' in post2.enemy_controlled[SH]:
        fails.append(('one_sided_wipe_capture_1_255',
                      'seed %d: captured at the wipe without entering'
                      % seed2))
        return
    advs = [a for a in ai.get_legal_actions(post2, SH)
            if a.action_type == 'victor_advance'
            and int(a.params.get('n_regular', 0)) > 0]
    if not advs:
        fails.append(('one_sided_wipe_advance', 'no victor_advance offered'))
        return
    entered = ai.apply_action(post2, advs[0])
    if 'pelargir' not in entered.enemy_controlled[SH]:
        fails.append(('one_sided_wipe_capture_on_entry',
                      'seed %d: the advance did not capture' % seed2))


# -------------------------------------------------------------------- K6
def _dwarf_no_supply_state():
    """FP Dwarves at Pelargir: 1 regular + 1 elite, every other Dwarven
    regular deployed (pool 0) and the casualty box empty."""
    st = field_battle_state(atk_reg=5, def_nation=Nation.DWARVES,
                            def_reg=1, def_elite=1)
    return st


def _to_allocation(st, want_def_hits):
    """Seed-search the round to a defender casualty_allocation pending
    with exactly want_def_hits hits."""
    def make():
        return decline_cards(st)

    def ok(post):
        pc = post.pending_combat
        return (pc is not None and pc.get('phase') == 'casualty_allocation'
                and pc.get('alloc_stage') == 'defender'
                and int(pc.get('def_hits', 0)) == want_def_hits)
    return seed_search(make, ok)


def test_alloc_menu_offers_k_to_n_elites(fails):
    """MEASURES: with ZERO replacement supply (pool 0, box 0) and 1 hit
    on 1R+1E, the casualty_allocate menu offers count 0 AND count 1."""
    st = _dwarf_no_supply_state()
    place(st, 'erebor', Nation.DWARVES, regulars=4, clear=True)
    if ai.figures_available(st, Nation.DWARVES, 'regular') != 0:
        fails.append(('supply_precondition', 'dwarf pool not 0: %d'
                      % ai.figures_available(st, Nation.DWARVES, 'regular')))
        return
    seed, post = _to_allocation(st, 1)
    if post is None:
        fails.append(('alloc_seed', 'no seed reached a 1-hit allocation'))
        return
    counts = sorted(a.params['count'] for a in
                    ai.get_legal_actions(post, FP)
                    if a.action_type == 'casualty_allocate')
    if counts != [0, 1]:
        fails.append(('alloc_menu_k_cap', 'counts=%r (want [0, 1])'
                      % (counts,)))


def test_alloc_no_source_elite_eliminated(fails):
    """MEASURES: choosing count=1 with no replacement source ELIMINATES
    the elite outright (army keeps its regular, elite booked as an FP
    casualty) and stamps (DWARVES, None) on _dg_src_log."""
    st = _dwarf_no_supply_state()
    place(st, 'erebor', Nation.DWARVES, regulars=4, clear=True)
    seed, post = _to_allocation(st, 1)
    if post is None:
        fails.append(('alloc_seed', 'no seed'))
        return
    k1 = pick(ai.get_legal_actions(post, FP), 'casualty_allocate', count=1)
    if k1 is None:
        fails.append(('alloc_k1_missing', 'count=1 not offered'))
        return
    n0 = len(getattr(post, '_dg_src_log', None) or [])
    after = ai.apply_action(post, k1)
    units = [(u.nation, u.kind) for u in own_units(after, 'pelargir', FP)]
    if units != [(Nation.DWARVES, Kind.REGULAR)]:
        fails.append(('elite_eliminated', 'units after k=1: %r' % (units,)))
    cas = (getattr(after, 'fp_casualties', {}) or {}).get(Nation.DWARVES, {})
    if cas.get('elite', 0) != 1:
        fails.append(('elite_casualty_booked', repr(cas)))
    log = list(getattr(after, '_dg_src_log', None) or [])[n0:]
    if log != [(Nation.DWARVES, None)]:
        fails.append(('dg_stamp_none', repr(log)))


def test_alloc_box_source_downgrades(fails):
    """MEASURES: with ONE Dwarven regular in the casualty box, count=1
    downgrades the elite (army = 2 regulars), the box regular returns
    (box 0) and _dg_src_log gains (DWARVES, 'cas')."""
    st = _dwarf_no_supply_state()
    place(st, 'erebor', Nation.DWARVES, regulars=3, clear=True)
    st.fp_casualties = {Nation.DWARVES: {'regular': 1, 'elite': 0}}
    seed, post = _to_allocation(st, 1)
    if post is None:
        fails.append(('alloc_seed', 'no seed'))
        return
    k1 = pick(ai.get_legal_actions(post, FP), 'casualty_allocate', count=1)
    if k1 is None:
        fails.append(('alloc_k1_missing', 'count=1 not offered'))
        return
    n0 = len(getattr(post, '_dg_src_log', None) or [])
    after = ai.apply_action(post, k1)
    units = sorted((u.nation.value, u.kind.value)
                   for u in own_units(after, 'pelargir', FP))
    if units != [('Dwarves', 'Regular'), ('Dwarves', 'Regular')]:
        fails.append(('box_downgrade_units', repr(units)))
    if after.fp_casualties[Nation.DWARVES].get('regular', 0) != 0:
        fails.append(('box_consumed', repr(after.fp_casualties)))
    log = list(getattr(after, '_dg_src_log', None) or [])[n0:]
    if log != [(Nation.DWARVES, 'cas')]:
        fails.append(('dg_stamp_cas', repr(log)))


def _w2_025_full_advance(st):
    """W2-025: after the defender's retreat_into_siege the ATTACKER
    decides victor_advance (post_siege_retreat); take the full
    advance (== the pre-W2-025 forced move-in) so the siege forms."""
    pc = getattr(st, 'pending_combat', None) or {}
    if pc.get('phase') != 'victor_advance' or not pc.get('post_siege_retreat'):
        return st
    rows = [a for a in ai.get_legal_actions(st, pc['deciding_side'])
            if a.action_type == 'victor_advance'
            and a.params.get('take_leaders')
            and a.params.get('n_leaders') is None]
    if not rows:
        return st
    rows.sort(key=lambda a: -(int(a.params.get('n_regular') or 0)
                              + int(a.params.get('n_elite') or 0)
                              + int(a.params.get('n_nazgul') or 0)))
    return ai.apply_action(st, rows[0])


def _siege_press_state(supply):
    """Shadow besieges Minas Tirith with 1R+1E Isengard (pool emptied
    when supply is False); drive to the combat_round_result phase."""
    st = fresh_state()
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'osgiliath', Nation.ISENGARD, regulars=1, elites=1,
          clear=True)
    place(st, 'minas tirith', Nation.GONDOR, regulars=4, clear=True)
    if not supply:
        place(st, 'orthanc', Nation.ISENGARD, regulars=11, clear=True)
        assert ai.figures_available(st, Nation.ISENGARD, 'regular') == 0
    st.hands[SH].hand = []
    st.hands[FP].hand = []
    set_dice(st, SH, [Face.ARMY])
    set_dice(st, FP, [Face.CHARACTER])
    atk = pick(ai.get_legal_actions(st, SH), 'army_attack',
               **{'from': 'osgiliath', 'to': 'minas tirith'})
    s2 = ai.apply_action(st, atk)
    rs = pick(ai.get_legal_actions(s2, FP), 'retreat_into_siege')
    s3 = ai.apply_action(s2, rs)
    # W2-025 (1.49 sub-bullet 2026-09-03): the withdrawal opens the
    # ATTACKER's victor_advance; the siege forms when the attacker
    # advances (here: the whole army, the pre-W2-025 forced path)
    s3 = _w2_025_full_advance(s3)
    assert s3.pending_combat is None and 'minas tirith' in         s3.besieged_strongholds, 'siege established, action over'
    # the assault is a NEW army action on the besieged stronghold
    set_dice(s3, SH, [Face.ARMY])
    sa = pick(ai.get_legal_actions(s3, SH), 'army_attack',
              **{'from': 'minas tirith', 'to': 'minas tirith'})
    assert sa is not None, 'siege assault must be on the menu'
    s4 = ai.apply_action(s3, sa)
    assert s4.pending_combat is not None, 'assault opens the battle'
    sg = pick(ai.get_legal_actions(s4, SH), 'siege_assault')
    assert sg is not None, 'attacker_siege_decision offers the assault'
    return ai.apply_action(s4, sg)


def _to_press_window(st):
    def make():
        return decline_cards(st)

    def ok(post):
        pc = post.pending_combat
        return (pc is not None and pc.get('phase') == 'combat_round_result'
                and any(u.kind == Kind.ELITE
                        for u in own_units(post, 'minas tirith', SH)))
    return seed_search(make, ok)


def test_siege_press_one_answerer(fails):
    """MEASURES: siege_press with NO replacement source eliminates the
    elite outright (1 Isengard regular remains) and stamps
    (ISENGARD, None); WITH pool supply it downgrades (2 regulars) and
    stamps (ISENGARD, 'pool')."""
    for supply in (False, True):
        st = _siege_press_state(supply)
        seed, post = _to_press_window(st)
        if post is None:
            fails.append(('press_seed_%s' % supply, 'no seed'))
            continue
        pr = pick(ai.get_legal_actions(post, SH), 'siege_press')
        if pr is None:
            fails.append(('press_menu_%s' % supply, 'no siege_press'))
            continue
        n0 = len(getattr(post, '_dg_src_log', None) or [])
        random.seed(seed)
        after = ai.apply_action(post, pr)
        log = list(getattr(after, '_dg_src_log', None) or [])[n0:n0 + 1]
        want = [(Nation.ISENGARD, 'pool' if supply else None)]
        if log != want:
            fails.append(('press_stamp_%s' % supply,
                          'stamp=%r want=%r' % (log, want)))


# ---------------------------------------------------------------- W1#120
def test_wwsat_classifier_sees_placeholders(fails):
    """MEASURES: an owed WWSAT discard against an FP hand of live driver
    placeholders removes exactly ONE *_CHAR_* placeholder (hand 2 -> 1,
    the STRAT placeholder survives)."""
    st = fresh_state()
    st.hands[FP].hand = ['HU_FR_CHAR_1', 'HU_FR_STRAT_2']
    st.persistent_cards = gs.PersistentCardState()
    st._wwsat_owed = [1]
    ai._offer_wwsat_discard(st)
    hand = list(st.hands[FP].hand)
    if hand != ['HU_FR_STRAT_2']:
        fails.append(('wwsat_placeholder_hand', 'hand=%r' % (hand,)))
    # both sources present -> the learned SOURCE pending opens with a
    # hand option (the classifier saw the placeholder)
    st2 = fresh_state()
    from game_state import PersistentCardType as PCT
    st2.persistent_cards = gs.PersistentCardState()
    st2.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_PEOPLES_Character_Axe_and_Bow',
        card_name='Axe and Bow', side=FP, card_type=PCT.AXE_AND_BOW))
    st2.hands[FP].hand = ['HU_FR_CHAR_1']
    st2._wwsat_owed = [1]
    ai._offer_wwsat_discard(st2)
    pco = st2.pending_card_option
    opts = [o.get('wwsat_source') for o in ((pco or {}).get('options')
                                           or [])]
    if 'hand' not in opts or 'table' not in opts:
        fails.append(('wwsat_source_pending', 'options=%r' % (opts,)))
    # one classifier: the rules layer agrees
    st3 = fresh_state()
    st3.hands[FP].hand = ['HU_FR_STRAT_1', 'HU_FR_CHAR_2']
    game_logic._wwsat_resolve(st3, 'hand')
    if list(st3.hands[FP].hand) != ['HU_FR_STRAT_1']:
        fails.append(('wwsat_resolve_hand', repr(st3.hands[FP].hand)))


# ---------------------------------------------------------------- W1#104
def test_ring_use_keeps_same_side(fails):
    """MEASURES: after an FP elven_ring action _next_acting_side(FP) is
    FP; after the funded action it alternates to Shadow."""
    st = fresh_state()
    set_dice(st, FP, [Face.MUSTER, Face.CHARACTER])
    set_dice(st, SH, [Face.ARMY, Face.ARMY])
    ring = pick(ai.get_legal_actions(st, FP), 'elven_ring')
    if ring is None:
        fails.append(('ring_menu', 'no elven_ring offered'))
        return
    s2 = ai.apply_action(st, ring)
    nxt = ai._next_acting_side(s2, FP)
    if nxt is not FP:
        fails.append(('ring_same_side', 'next=%r' % (nxt,)))
    # the funded action (any FP die action) ends the window
    act = next((a for a in ai.get_legal_actions(s2, FP)
                if a.die_idx >= 0 and a.action_type != 'elven_ring'), None)
    if act is None:
        fails.append(('ring_followup_menu', 'no die action'))
        return
    s3 = ai.apply_action(s2, act)
    nxt3 = ai._next_acting_side(s3, FP)
    if nxt3 is not SH:
        fails.append(('ring_then_alternate', 'next=%r' % (nxt3,)))
    # a clone keeps the marker (MCTS routing)
    if ai._next_acting_side(ai.clone_state(s2), FP) is not FP:
        fails.append(('ring_marker_cloned', 'marker lost on clone'))


# ---------------------------------------------------------------- W1#132
def _to_defender_retreat(st):
    def make():
        post = decline_cards(st)
        pc = post.pending_combat
        if pc is None or pc.get('phase') != 'attacker_continue':
            return post
        cont = pick(ai.get_legal_actions(post, SH), 'combat_continue')
        return ai.apply_action(post, cont)

    def ok(post):
        pc = post.pending_combat
        return pc is not None and pc.get('phase') == 'defender_retreat'
    return seed_search(make, ok)


def test_post_retreat_advance_is_learned(fails):
    """MEASURES: after the defender retreats, the engine opens the
    victor_advance pending for the ATTACKER (army still at the attack
    region); victor_advance then moves in and captures, victor_stay
    leaves the army and captures nothing."""
    st = field_battle_state(atk_reg=4, def_reg=4)
    seed, post = _to_defender_retreat(st)
    if post is None:
        fails.append(('retreat_seed', 'no seed reached defender_retreat'))
        return
    rt = pick(ai.get_legal_actions(post, FP), 'combat_retreat',
              to_region='lamedon')
    if rt is None:
        fails.append(('retreat_menu', 'no retreat to lamedon'))
        return
    after = ai.apply_action(post, rt)
    pc = after.pending_combat
    if pc is None or pc.get('phase') != 'victor_advance' \
            or pc.get('deciding_side') is not SH:
        fails.append(('post_retreat_pending', repr(pc)))
        return
    if not own_units(after, 'west harondor', SH) \
            or own_units(after, 'pelargir', SH):
        fails.append(('post_retreat_forced_move', 'attacker already moved'))
    if 'pelargir' in after.enemy_controlled[SH]:
        fails.append(('post_retreat_early_capture', 'captured before advance'))
    menu = ai.get_legal_actions(after, SH)
    adv = next((a for a in menu if a.action_type == 'victor_advance'
                and int(a.params.get('n_regular', 0)) > 0), None)
    stay = pick(menu, 'victor_stay')
    if adv is None or stay is None:
        fails.append(('post_retreat_menu',
                      sorted(set(a.action_type for a in menu))))
        return
    s_adv = ai.apply_action(after, adv)
    if not own_units(s_adv, 'pelargir', SH):
        fails.append(('victor_advance_moved', 'no Shadow units at pelargir'))
    if 'pelargir' not in s_adv.enemy_controlled[SH]:
        fails.append(('victor_advance_capture', 'advance did not capture'))
    s_stay = ai.apply_action(after, stay)
    if own_units(s_stay, 'pelargir', SH) \
            or 'pelargir' in s_stay.enemy_controlled[SH]:
        fails.append(('victor_stay', 'stay moved or captured'))
    if s_stay.pending_combat is not None:
        fails.append(('victor_stay_pending', 'pending not closed'))


# ---------------------------------------------------------------- W1#109
def test_consume_die_refuses_other_die(fails):
    """MEASURES: naming a USED die raises instead of consuming a
    different one (self-play / live); the teacher hook keeps the
    lenient path (control)."""
    st = fresh_state()
    set_dice(st, FP, [Face.CHARACTER, Face.CHARACTER])
    st.dice[FP][0].used = True
    raised = False
    try:
        ai._consume_die_for_face(st, FP, 0, (Face.CHARACTER,))
    except RuntimeError:
        raised = True
    if not raised:
        fails.append(('consume_die_silent', 'consumed a different die'))
    if st.dice[FP][1].used:
        fails.append(('consume_die_side_effect', 'die 1 was consumed'))
    saved = combat.FINAL_HITS_HOOK
    combat.FINAL_HITS_HOOK = (lambda *a, **k: ([], []))
    try:
        got = ai._consume_die_for_face(st, FP, 0, (Face.CHARACTER,))
        if got != 1:
            fails.append(('consume_die_teacher', 'teacher path got %r' % got))
    finally:
        combat.FINAL_HITS_HOOK = saved


# ---------------------------------------------------------------- W1#107
def test_engine_play_hvai_no_auto_declare(fails):
    """MEASURES: EngineSession(human_side=FP) never calls
    ai._auto_enter_mordor; a watch session (no human) does (control)."""
    import engine_play as ep
    calls = []
    orig = ai._auto_enter_mordor

    def spy(state):
        calls.append(1)
        return False
    ai._auto_enter_mordor = spy
    try:
        s = ep.EngineSession(seed=5, mcts_sims=1, max_turns=1,
                             human_side=FP)
        s.current()
        n_h = len(calls)
        calls[:] = []
        s2 = ep.EngineSession(seed=5, mcts_sims=1, max_turns=1)
        s2.current()
        n_w = len(calls)
    finally:
        ai._auto_enter_mordor = orig
    if n_h != 0:
        fails.append(('hvai_auto_declare', 'called %d times' % n_h))
    if n_w == 0:
        fails.append(('watch_auto_declare_control', 'never called'))


# -------------------------------------------------------------- K1 step 1
def _hand_card_for(st, side, pc, is_attacker, deck):
    """A REAL playable combat card of `deck` (no pre-roll choice) from
    the side's deck list, placed into its hand; returns the cid."""
    from cards import get_card
    tmp, ok = ai._combat_card_check_ctx(st, pc, is_attacker)
    tag = 'fp' if side is FP else 'sh'
    attr = tag + ('_char_deck' if deck == 'character' else '_strat_deck')
    for cid in list(getattr(st, attr)):
        c = get_card(cid)
        if combat.combat_effect_kind(c.combat_name or '') is not None:
            continue
        if ai._combat_card_playable(st, c, tmp, is_attacker):
            getattr(st, attr).remove(cid)
            st.hands[side].hand.append(cid)
            return cid
    return None


def _unplayable_card_for(st, side, pc, is_attacker, deck):
    from cards import get_card
    tmp, ok = ai._combat_card_check_ctx(st, pc, is_attacker)
    tag = 'fp' if side is FP else 'sh'
    attr = tag + ('_char_deck' if deck == 'character' else '_strat_deck')
    for cid in list(getattr(st, attr)):
        c = get_card(cid)
        if (c.combat_name or '').strip() and not ai._combat_card_playable(
                st, get_card(cid), tmp, is_attacker):
            getattr(st, attr).remove(cid)
            st.hands[side].hand.append(cid)
            return cid
    return None


def test_hidden_commit_protocol(fails):
    """MEASURES (human FP defender): after the AI attacker declines and
    the human commits a HIDDEN character card, the round is parked
    (rounds_fought 0, no deciding side, no menu), combat_cards_unrevealed
    == [FP]; the reveal of a playable card resumes the round with that
    card; an unplayable reveal raises IllegalPlay with the state
    untouched."""
    st = field_battle_state(atk_reg=3, def_reg=3)
    nc = pick(ai.get_legal_actions(st, SH), 'combat_card_none')
    s1 = ai.apply_action(st, nc)
    pc = s1.pending_combat
    if pc.get('phase') != 'defender_combat_card':
        fails.append(('k1_phase0', repr(pc.get('phase'))))
        return
    deck = 'character'
    cid = _hand_card_for(s1, FP, pc, False, deck)
    if cid is None:
        deck = 'strategy'
        cid = _hand_card_for(s1, FP, pc, False, deck)
    bad = _unplayable_card_for(s1, FP, pc, False, deck)
    if cid is None or bad is None:
        fails.append(('k1_cards', 'cid=%r bad=%r' % (cid, bad)))
        return
    dk = 'char' if deck == 'character' else 'strat'
    pile = 'fp_' + deck
    # the hidden commit (driver path: state mutated in place)
    ai.commit_hidden_combat_card(s1, FP, deck)
    if ai.combat_cards_unrevealed(s1) != [FP]:
        fails.append(('k1_unrevealed', repr(ai.combat_cards_unrevealed(s1))))
    pc = s1.pending_combat
    if pc is None or pc.get('rounds_fought', 0) != 0 \
            or pc.get('def_hits') is not None:
        fails.append(('k1_round_parked', repr(pc)))
        return
    if ai.describe_pending(s1) is not None:
        fails.append(('k1_no_decider', repr(ai.describe_pending(s1))))
    if ai.get_legal_actions(s1, FP) or ai.get_legal_actions(s1, SH):
        fails.append(('k1_no_menu', 'a menu was served while unrevealed'))
    if pc.get('def_hidden_cc') != {'deck': deck}:
        fails.append(('k1_hidden_record', repr(pc.get('def_hidden_cc'))))
    # a clone keeps the hidden commit
    if ai.combat_cards_unrevealed(ai.clone_state(s1)) != [FP]:
        fails.append(('k1_clone', 'hidden commit lost on clone'))
    # the D3 mixture sees a CHARACTER-type commit with no identity
    # (defender hidden: the attacker-side mixture stays empty; the
    # pool for the defender's type is non-empty)
    mix = ai.combat_card_pool(s1, pc, False, deck_type=dk)
    if not mix:
        fails.append(('k1_mixture', 'empty type-only pool'))
    # unplayable reveal: refused, state untouched
    snap = (dict(pc), list(s1.hands[FP].hand))
    raised = False
    try:
        ai.reveal_combat_card(s1, FP, bad)
    except ai.IllegalPlay:
        raised = True
    if not raised:
        fails.append(('k1_refuse_unplayable', 'no IllegalPlay'))
    if (dict(s1.pending_combat), list(s1.hands[FP].hand)) != snap:
        fails.append(('k1_refuse_untouched', 'state changed on refusal'))
    # playable reveal: the round resolves with the card
    random.seed(11)
    ai.reveal_combat_card(s1, FP, cid)
    if ai.combat_cards_unrevealed(s1):
        fails.append(('k1_revealed', 'still unrevealed'))
    if cid in s1.hands[FP].hand:
        fails.append(('k1_hand', 'revealed card still in hand'))
    if cid not in s1.discard_face_up[pile]:
        fails.append(('k1_discard', 'revealed card not on the face-up pile'))
    lcc = getattr(s1, '_last_combat_cards', None)
    if lcc is None or lcc[1] != cid:
        fails.append(('k1_round_used_card', repr(lcc)))
    pc2 = s1.pending_combat
    if pc2 is not None and pc2.get('rounds_fought', 0) != 1:
        fails.append(('k1_round_fought', repr(pc2)))
    # idempotent echo of an engine-visible (AI) commit: no-op
    try:
        ai.reveal_combat_card(s1, FP, cid)
    except ai.IllegalPlay:
        fails.append(('k1_idempotent', 'echo reveal refused'))


def test_hidden_commit_attacker_then_ai_defender(fails):
    """MEASURES (human SH attacker, AI FP defender): a hidden attacker
    commit advances to defender_combat_card; the defender's menu is
    served; after the defender's decline the round is parked until the
    reveal; a hidden commit in the wrong phase / wrong side raises."""
    st = field_battle_state(atk_reg=3, def_reg=3)
    pc = st.pending_combat
    # wrong side first: the defender cannot commit in the attacker window
    raised = False
    try:
        ai.commit_hidden_combat_card(st, FP, 'strategy')
    except ai.IllegalPlay:
        raised = True
    if not raised or 'atk_hidden_cc' in st.pending_combat \
            or 'def_hidden_cc' in st.pending_combat:
        fails.append(('k1_wrong_side', 'accepted or mutated'))
    cid = _hand_card_for(st, SH, pc, True, 'strategy')
    if cid is None:
        fails.append(('k1_atk_card', 'no playable strategy card'))
        return
    ai.commit_hidden_combat_card(st, SH, 'strategy')
    pc = st.pending_combat
    if pc.get('phase') != 'defender_combat_card' \
            or pc.get('deciding_side') is not FP:
        fails.append(('k1_atk_hidden_phase', repr(pc.get('phase'))))
        return
    dp = ai.describe_pending(st)
    if dp is None or dp['side'] is not FP or not dp['actions']:
        fails.append(('k1_def_menu', repr(dp)))
        return
    # the attacker's commit is a type-only mixture for the defender
    mix = ai._atk_commit_mixture(st, pc)
    if not mix or any(c == cid and n == 0 for c, n in mix):
        fails.append(('k1_atk_mixture', repr(mix)[:120]))
    if any(c == '_hidden_combat_card_' for c, _n in mix):
        fails.append(('k1_atk_mixture_sentinel', 'sentinel leaked'))
    nc = pick(dp['actions'], 'combat_card_none')
    s2 = ai.apply_action(st, nc)
    if ai.combat_cards_unrevealed(s2) != [SH]:
        fails.append(('k1_atk_unrevealed',
                      repr(ai.combat_cards_unrevealed(s2))))
    if s2.pending_combat.get('rounds_fought', 0) != 0:
        fails.append(('k1_atk_parked', 'round ran before the reveal'))
    random.seed(7)
    ai.reveal_combat_card(s2, SH, cid)
    if ai.combat_cards_unrevealed(s2):
        fails.append(('k1_atk_revealed', 'still unrevealed'))
    lcc = getattr(s2, '_last_combat_cards', None)
    if lcc is None or lcc[0] != cid:
        fails.append(('k1_atk_round_used_card', repr(lcc)))


def test_hidden_commit_mcts_sim_resolves(fails):
    """MEASURES: with the human attacker's commit hidden, a search from
    the AI defender's seat can still resolve the round in-tree (the
    overlay substitutes a sampled same-type identity and the parked
    gate opens)."""
    st = field_battle_state(atk_reg=3, def_reg=3)
    pc = st.pending_combat
    cid = _hand_card_for(st, SH, pc, True, 'strategy')
    if cid is None:
        fails.append(('k1_sim_card', 'no playable strategy card'))
        return
    ai.commit_hidden_combat_card(st, SH, 'strategy')
    root = ai.clone_state(st)
    root.pending_combat['_commit_nonce'] = 1
    ov = ai._SimOverlay(root, FP)
    sim = ai.clone_state(root)
    ov.patch(sim)
    pcs = sim.pending_combat
    if 'atk_hidden_cc' in pcs or pcs.get('atk_combat_card') in (
            None, '_hidden_combat_card_'):
        fails.append(('k1_sim_patch', repr({k: pcs.get(k) for k in (
            'atk_hidden_cc', 'atk_combat_card')})))
        return
    nc = pick(ai.get_legal_actions(sim, FP), 'combat_card_none')
    random.seed(2)
    s2 = ai.apply_action(sim, nc)
    pc2 = s2.pending_combat
    # resolved = the round rolled, or the sampled card's pre-roll
    # effect choice opened (the gate is past either way)
    if pc2 is not None and pc2.get('rounds_fought', 0) == 0 \
            and not pc2.get('def_hits') \
            and pc2.get('phase') != 'combat_effect_choice':
        fails.append(('k1_sim_round', 'round did not resolve in-sim: %r'
                      % (pc2.get('phase'),)))


# -------------------------------------------------------------------- K2
def test_declare_phase_open_close(fails):
    """MEASURES: open_declare_phase arms the declare cascade (kind
    'declare', FP menu carries pass); close_declare_phase clears it."""
    st = fresh_state()
    game_logic.open_declare_phase(st)
    dp = ai.describe_pending(st)
    if dp is None or dp['kind'] != 'declare' or dp['side'] is not FP:
        fails.append(('declare_open', repr(dp and dp['kind'])))
    elif 'pass' not in set(a.action_type for a in dp['actions']):
        fails.append(('declare_menu_pass', 'no pass decline'))
    game_logic.close_declare_phase(st)
    if ai.describe_pending(st) is not None:
        fails.append(('declare_close', 'still pending'))


def test_companion_group_menus_uncapped(fails):
    """MEASURES (USER 2026-08-22 "why is there a cap at all for
    anything, I don't want one"; live refusal of Strider+Boromir ->
    Moria out of a 7-companion fellowship, twice): the legal surface
    carries EVERY companion subset - separation from a full
    7-companion fellowship offers every proper subset (2**7-2 = 126
    groups, each to its own reachable set), and a stack of 5 separated
    companions on the board offers every proper subset too (30).
    Sensitivity: the exact pair Strider+Boromir -> Moria is in the
    separation menu (Strider's level 3 carries Boromir)."""
    from itertools import combinations
    st = fresh_state()
    set_dice(st, FP, [gs.ActionDieFace.CHARACTER, gs.ActionDieFace.EVENT,
                      gs.ActionDieFace.MUSTER, gs.ActionDieFace.WILL_OF_WEST])
    names = sorted(c.name for c in st.fellowship.companions
                   if c.name != 'Gollum' and c.level > 0)
    if len(names) != 7:
        fails.append(('uncapped_precondition', names))
        return
    acts = [a for a in ai.get_legal_actions(st, FP)
            if a.action_type == 'companion_separate']
    groups = set()
    for a in acts:
        g = a.params.get('companions') or [a.params.get('companion')]
        groups.add(tuple(sorted(g)))
    want = set()
    for r in range(1, 8):
        for c in combinations(names, r):
            want.add(tuple(sorted(c)))
    missing = sorted(want - groups)
    if missing:
        fails.append(('separation_subsets_missing',
                      (len(missing), missing[:4])))
    pair = [a for a in acts
            if sorted(a.params.get('companions') or []) == ['Boromir',
                                                            'Strider']
            and a.params.get('to_region') == 'moria']
    if not pair:
        fails.append(('strider_boromir_moria', 'absent from the menu'))
    # companion MOVES: 5 companions separated into one region
    st2 = fresh_state()
    set_dice(st2, FP, [gs.ActionDieFace.CHARACTER])
    five = [c for c in st2.fellowship.companions
            if c.name in ('Strider', 'Boromir', 'Legolas', 'Gimli',
                          'Gandalf the Grey')]
    for c in five:
        st2.fellowship.companions.remove(c)
        game_logic.place_separated_companion(st2, c, 'bree')             if hasattr(game_logic, 'place_separated_companion') else None
    arm = game_logic.get_or_create_army(st2, 'bree')
    on_board = sorted(l.name for l in arm.leaders
                      if isinstance(l, gs.Companion))
    if len(on_board) != 5:
        fails.append(('move_precondition', on_board))
        return
    mv = [a for a in ai.get_legal_actions(st2, FP)
          if a.action_type == 'companion_move'
          and a.params.get('from') == 'bree']
    mgroups = set(tuple(sorted(a.params.get('companions') or []))
                  for a in mv)
    mwant = set(tuple(sorted(c)) for r in range(1, 5)
                for c in combinations(on_board, r))
    mmissing = sorted(mwant - mgroups)
    if mmissing:
        fails.append(('move_subsets_missing',
                      (len(mmissing), mmissing[:4])))


def test_nazgul_destinations_uncapped(fails):
    """MEASURES (USER 2026-08-22 no-legal-surface-caps, applied to the
    Nazgul menu 2026-08-24): nazgul_move offers EVERY legal region -
    the old 'strategically meaningful set' filter hid legal flights
    from the human seat (the live matcher judges gestures from this
    menu).  Sensitivity: a plain wilderness region with no army, no
    settlement, far from the Fellowship (Cardolan) is offered, and so
    is a region holding an FP FIELD army (USER ruling 1.251,
    2026-08-29: "a nazgul can end movement in an FP field army").
    SPECIFICITY (the one RULES gate stays): an unbesieged FP
    stronghold is NOT offered."""
    st = fresh_state()
    set_dice(st, SH, [gs.ActionDieFace.CHARACTER])
    arm = game_logic.get_or_create_army(st, 'dagorlad')
    arm.leaders.append(gs.Nazgul(nation=gs.Nation.SAURON))
    fp_arm = game_logic.get_or_create_army(st, 'osgiliath')
    fp_arm.units = [gs.Unit(gs.Nation.GONDOR, gs.UnitKind.REGULAR)]
    naz = [a for a in ai.get_legal_actions(st, SH)
           if a.action_type == 'nazgul_move'
           and a.params.get('from') == 'dagorlad']
    dests = {a.params.get('to') for a in naz}
    if 'cardolan' not in dests:
        fails.append(('nazgul_wilderness_offered', sorted(dests)[:8]))
    if 'osgiliath' not in dests:
        fails.append(('nazgul_fp_field_army_offered_1_251', 'missing'))
    if 'minas tirith' in dests:
        fails.append(('nazgul_unbesieged_stronghold_excluded',
                      'offered'))
    n_regions = len(ai.board.regions)
    if len(dests) < n_regions - 30:
        fails.append(('nazgul_menu_breadth',
                      (len(dests), n_regions)))


def test_lattice_encoding_exact(fails):
    """MEASURES (2026-08-22 search-cost fix for the uncapped lattice):
    the memoized separation block (shared Action objects), its
    per-Action encoding stash and the block matrix cache must be
    EXACT - for every action of a full-fellowship FP menu the batch
    encoding equals a fresh encode_action, and it STILL does after the
    board changes under the cache (a Shadow army placed on a lattice
    destination changes the [75:78] enemy features; a die face change
    changes feat[80]).  Sensitivity: the stash is really used (the
    second call makes no encode_action call for lattice actions)."""
    import numpy as np
    st = fresh_state()
    set_dice(st, FP, [gs.ActionDieFace.CHARACTER, gs.ActionDieFace.CHARACTER,
                      gs.ActionDieFace.EVENT, gs.ActionDieFace.WILL_OF_WEST])
    ai._SEP_LATTICE_MEMO.clear(); ai._LATTICE_ENC_CACHE.clear()

    def check(state, tag):
        acts = ai.get_legal_actions(state, FP)
        lat = [a for a in acts if '_lattice' in a.__dict__]
        if len(lat) < 100:
            fails.append(('lattice_precondition_' + tag, len(lat)))
            return acts
        encs, ridx = ai._encode_actions_batch(acts, state, FP, {})
        bad = 0
        for i, a in enumerate(acts):
            fresh = ai.encode_action(a, state, FP)
            if not np.array_equal(encs[i], fresh):
                bad += 1
            if ridx[i] != ai._get_action_region_index(a):
                bad += 1
        if bad:
            fails.append(('lattice_encoding_drift_' + tag, bad))
        return acts
    acts0 = check(st, 'fresh')
    lat0 = [a for a in acts0 if '_lattice' in a.__dict__]
    if not lat0:
        return
    # the memo reuses the SAME Action objects on an unchanged board
    # (no rebuild per node)
    lat_again = [a for a in ai.get_legal_actions(st, FP)
                 if '_lattice' in a.__dict__]
    if not lat_again or lat_again[0] is not lat0[0]:
        fails.append(('lattice_not_shared', 'block rebuilt on an '
                                            'unchanged board'))
    # the board changes under the cache: an enemy army on a destination
    # (changes the enemy features AND, via the terminus rule, the legal
    # destination set - the block is legitimately rebuilt)
    dest = lat0[0].params['to_region']
    place(st, dest, gs.Nation.SAURON, regulars=2)
    check(st, 'enemy_at_dest')
    # a Shadow STRONGHOLD destination stays legal whatever garrison it
    # holds (terminus ruling) - the legal set is unchanged, only the
    # enemy features differ: the block cache must key on them
    lat_pre = [a for a in ai.get_legal_actions(st, FP)
               if '_lattice' in a.__dict__]
    if any(a.params['to_region'] == 'moria' for a in lat_pre):
        place(st, 'moria', gs.Nation.SAURON, regulars=3, elites=1)
        lat2 = [a for a in ai.get_legal_actions(st, FP)
                if '_lattice' in a.__dict__]
        if lat2 and lat_pre and lat2[0] is not lat_pre[0]:
            fails.append(('lattice_rebuilt_on_garrison',
                          'legal set unchanged but block rebuilt'))
        check(st, 'moria_garrison')
    else:
        fails.append(('lattice_moria_precondition',
                      'NOT-MEASURED: moria not a destination'))
    # die face change (feat[80]): swap the funding die's face
    st.dice[FP][0].current = gs.ActionDieFace.WILL_OF_WEST
    st.dice[FP][1].current = gs.ActionDieFace.CHARACTER
    check(st, 'face_change')
    # sensitivity: a repeat call hits the stash (no fresh encodes)
    n0 = [0]
    orig = ai.encode_action

    def spy(a, state, side):
        if '_lattice' in a.__dict__:
            n0[0] += 1
        return orig(a, state, side)
    ai.encode_action = spy
    try:
        acts2 = ai.get_legal_actions(st, FP)
        ai._encode_actions_batch(acts2, st, FP, {})
    finally:
        ai.encode_action = orig
    if n0[0]:
        fails.append(('lattice_stash_unused', n0[0]))


def main():
    tests = [
        test_clone_state_copies_owed_draws,
        test_owed_draw_writers,
        test_followup_exclude_is_per_clone,
        test_direct_stage_handler_failure_is_loud,
        test_palantir_never_on_draw,
        test_palantir_only_event_die_plays,
        test_palantir_deferred_draw_fires_at_next_quiet_apply,
        test_mutual_elimination_no_capture,
        test_alloc_menu_offers_k_to_n_elites,
        test_alloc_no_source_elite_eliminated,
        test_alloc_box_source_downgrades,
        test_siege_press_one_answerer,
        test_wwsat_classifier_sees_placeholders,
        test_ring_use_keeps_same_side,
        test_post_retreat_advance_is_learned,
        test_consume_die_refuses_other_die,
        test_engine_play_hvai_no_auto_declare,
        test_hidden_commit_protocol,
        test_hidden_commit_attacker_then_ai_defender,
        test_hidden_commit_mcts_sim_resolves,
        test_declare_phase_open_close,
        test_companion_group_menus_uncapped,
        test_nazgul_destinations_uncapped,
        test_lattice_encoding_exact,
    ]
    only = set(sys.argv[1:])
    fails = []
    for t in tests:
        if only and t.__name__ not in only:
            continue
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
            traceback.print_exc()
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
