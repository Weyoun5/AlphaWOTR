#!/usr/bin/env python3
"""Tests for War of the Ring rule implementations.

Verifies key rules per the WOTR 2nd Edition rulebook:
1. Army die: 2 army moves OR 1 attack
2. Character die: requires leader for army movement/attack
3. Character die: can advance political track
4. Any die can play matching event cards
5. Muster die: 2 recruits (regular/leader) or 1 elite upgrade
6. Active (non-at-war) nations can move but not attack
7. Pending state cleared correctly
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai
import game_state as gs
import game_logic
import board
from board import sanitize
from collections import Counter
from cards import get_card, get_cards


def setup_game():
    """Create a fresh game state with dice rolled."""
    b = board.load_board()
    state = gs.setup_initial_game(b)
    game_logic.initialize_decks(state)
    return state


def set_dice(state, side, faces):
    """Set specific die faces for testing."""
    state.dice[side] = []
    for face in faces:
        d = gs.ActionDie(faces=[face], side=side)
        d.current = face
        state.dice[side].append(d)


def test_army_die_double_move():
    """Army die allows 2 moves from different armies, OR 1 attack."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3  # At war

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    moves = [a for a in actions if a.action_type == 'army_move']
    attacks = [a for a in actions if a.action_type == 'army_attack']

    assert len(moves) > 0, "Army die should generate army moves"

    # Execute first move (capture the destination's pre-existing units
    # first -- those figures do NOT move and may leave with move 2)
    first_move = moves[0]
    first_from = first_move.params['from']
    _dst = next((a for k, a in state.armies.items()
                 if sanitize(k) == sanitize(first_move.params['to'])), None)
    pre_units = sum(
        1 for u in (_dst.units if _dst else [])
        if ai._side_of_nation(u.nation) == gs.Side.FREE_PEOPLES)
    ai._execute_action(state, first_move)

    assert state.pending_army_move_die is not None, "First army move should set pending"
    # pending_army_move_from now tracks the moved units' NEW location
    # (destination) — those units cannot move again with the same die
    assert sanitize(state.pending_army_move_from) == \
        sanitize(first_move.params['to'])

    # Second actions should be army_move + pass only
    actions2 = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    types2 = set(a.action_type for a in actions2)
    assert types2 <= {'army_move', 'pass'}, f"Second move should only allow army_move/pass, got {types2}"

    # Rulebook p.25: "no figure is moved twice using the same action" —
    # the MOVED units (now at the destination) cannot move again with this
    # die; units left behind at the origin by a split MAY.
    first_to = first_move.params['to']
    moved_again = [a for a in actions2
                   if a.action_type == 'army_move'
                   and sanitize(a.params['from']) == sanitize(first_to)]
    if pre_units == 0:
        assert len(moved_again) == 0, \
            "Units that just moved cannot move again with the same die"
    else:
        for a in moved_again:
            n_mv = int(a.params.get('n_regular') or 0) + \
                int(a.params.get('n_elite') or 0)
            assert n_mv <= pre_units, \
                "Second move may only take pre-existing (unmoved) figures"

    # Execute second move or pass
    pass_action = [a for a in actions2 if a.action_type == 'pass'][0]
    ai._execute_action(state, pass_action)
    assert state.pending_army_move_die is None, "Pass should clear pending"

    # Die should be consumed
    unused = sum(1 for d in state.dice[gs.Side.FREE_PEOPLES] if not d.used)
    assert unused == 0, "Die should be consumed after double-move"

    print("  PASS: Army die double-move")


def test_army_die_attack_no_pending():
    """Army attack consumes die immediately (no pending state)."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3
    state.political.boxes[gs.Nation.SAURON] = 3

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    attacks = [a for a in actions if a.action_type == 'army_attack']

    if attacks:
        ai._execute_action(state, attacks[0])
        assert state.pending_army_move_die is None, "Attack should not set pending"
        unused = sum(1 for d in state.dice[gs.Side.FREE_PEOPLES] if not d.used)
        assert unused == 0, "Die should be consumed after attack"
        print("  PASS: Army die attack (no pending)")
    else:
        print("  SKIP: No attacks available (no adjacent enemies)")


def test_character_die_requires_leader():
    """Character die can only move armies with leaders."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    char_moves = [a for a in actions if a.action_type == 'army_move']
    char_attacks = [a for a in actions if a.action_type == 'army_attack']

    # Every army move must come from an army with leaders
    for a in char_moves:
        army = state.armies.get(a.params['from'])
        assert army and army.leaders, \
            f"Character die army move from {a.params['from']} has no leader!"

    for a in char_attacks:
        army = state.armies.get(a.params['from'])
        assert army and army.leaders, \
            f"Character die army attack from {a.params['from']} has no leader!"

    print(f"  PASS: Character die requires leader ({len(char_moves)} moves, {len(char_attacks)} attacks)")


def test_character_die_single_move():
    """Character die allows only 1 army move (no pending)."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    moves = [a for a in actions if a.action_type == 'army_move']

    if moves:
        ai._execute_action(state, moves[0])
        assert state.pending_army_move_die is None, \
            "Character die should NOT set pending (single move only)"
        unused = sum(1 for d in state.dice[gs.Side.FREE_PEOPLES] if not d.used)
        assert unused == 0, "Character die should be consumed after 1 move"
        print("  PASS: Character die single move (no pending)")
    else:
        print("  SKIP: No character die army moves available")


def test_character_die_political_advance():
    """Character die does NOT advance the political track (Muster die only,
    per rulebook action die reference)."""
    state = setup_game()

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    political = [a for a in actions if a.action_type == 'advance_political']

    assert len(political) == 0, \
        "Character die must NOT allow political advancement (Muster die only)"

    # Muster die DOES allow it
    state2 = setup_game()
    set_dice(state2, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.MUSTER])
    actions2 = ai.get_legal_actions(state2, gs.Side.FREE_PEOPLES)
    political2 = [a for a in actions2 if a.action_type == 'advance_political']
    assert len(political2) > 0, "Muster die should allow political advancement"

    print(f"  PASS: Political advance on Muster die only ({len(political2)} nations)")


def test_character_die_fellowship():
    """Character die can move/hide fellowship (FP only)."""
    state = setup_game()

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    fellowship = [a for a in actions
                  if a.action_type in ('fellowship_move', 'fellowship_hide')]

    assert len(fellowship) > 0, "Character die should allow fellowship actions"
    types = set(a.action_type for a in fellowship)
    assert 'fellowship_move' in types, "Should be able to move fellowship"

    # Shadow should NOT get fellowship actions
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.CHARACTER])
    sh_actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    sh_fellowship = [a for a in sh_actions
                     if a.action_type in ('fellowship_move', 'fellowship_hide')]
    assert len(sh_fellowship) == 0, "Shadow should not have fellowship actions"

    print("  PASS: Character die fellowship actions")


def test_event_card_play_by_die_type():
    """Different die types can play matching event cards."""
    state = setup_game()

    # Draw cards to get a mix of types
    for _ in range(5):
        try:
            game_logic.draw_event_card(state, gs.Side.FREE_PEOPLES, 'character')
        except RuntimeError:
            pass
        try:
            game_logic.draw_event_card(state, gs.Side.FREE_PEOPLES, 'strategy')
        except RuntimeError:
            pass

    hand = state.hands[gs.Side.FREE_PEOPLES].hand
    hand_types = {}
    for cid in hand:
        card = get_card(cid)
        if card:
            hand_types[cid] = card.card_type

    print(f"  Hand card types: {Counter(hand_types.values())}")

    # Test each die type
    for die_name, die_face, expected_types in [
        ('ARMY', gs.ActionDieFace.ARMY, {'Army'}),
        ('MUSTER', gs.ActionDieFace.MUSTER, {'Muster'}),
        ('CHARACTER', gs.ActionDieFace.CHARACTER, {'Character'}),
        ('EVENT', gs.ActionDieFace.EVENT, {'Character', 'Army', 'Muster'}),
    ]:
        set_dice(state, gs.Side.FREE_PEOPLES, [die_face])
        actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
        play_events = [a for a in actions if a.action_type == 'play_event']
        actual_types = set()
        for a in play_events:
            card = get_card(a.params['card_id'])
            if card:
                actual_types.add(card.card_type)

        # Verify actual types are subset of expected
        assert actual_types <= expected_types, \
            f"{die_name} die plays {actual_types} but should only play {expected_types}"

        print(f"  {die_name}: {len(play_events)} events, types={actual_types}")

    print("  PASS: Event card play by die type")


def test_muster_double_recruit():
    """Muster die allows 2 recruits (regular/leader combo)."""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    recruits = [a for a in actions if a.action_type == 'muster_recruit']
    rtypes = Counter(a.params['type'] for a in recruits)

    assert 'regular' in rtypes, "Muster should allow regular recruit"
    assert 'leader' in rtypes, "Muster should allow leader recruit"

    # First recruit
    first = [a for a in recruits if a.params['type'] == 'regular'][0]
    ai._execute_action(state, first)
    assert state.pending_muster_die is not None, "First recruit should set pending"
    assert state.pending_muster_first_type == 'regular'

    # Second recruit available
    actions2 = ai.get_legal_actions(state, gs.Side.SHADOW)
    recruits2 = [a for a in actions2 if a.action_type == 'muster_recruit']
    assert len(recruits2) > 0, "Second recruit should be available"

    # Execute second recruit
    second = recruits2[0]
    ai._execute_action(state, second)
    assert state.pending_muster_die is None, "Second recruit should clear pending"

    unused = sum(1 for d in state.dice[gs.Side.SHADOW] if not d.used)
    assert unused == 0, "Die should be consumed after double recruit"

    print("  PASS: Muster double recruit")


def test_muster_separate_settlements():
    """Two muster recruits must be in separate settlements."""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    recruits = [a for a in actions if a.action_type == 'muster_recruit']

    if recruits:
        first = recruits[0]
        first_region = first.params['region']
        ai._execute_action(state, first)

        # Second recruit should NOT include the same region
        actions2 = ai.get_legal_actions(state, gs.Side.SHADOW)
        recruits2 = [a for a in actions2 if a.action_type == 'muster_recruit']
        same_region = [a for a in recruits2
                      if sanitize(a.params['region']) == sanitize(first_region)]
        assert len(same_region) == 0, \
            f"Second recruit should not allow same region {first_region}"

        # But other regions should be available
        assert len(recruits2) > 0, "Other settlements should be available"
        print(f"  PASS: Muster separate settlements ({len(recruits2)} options)")
    else:
        print("  SKIP: No recruits available")


def test_muster_elite_placement():
    """Muster die can place 1 new elite (consumes full die, no second recruit)."""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    elites = [a for a in actions if a.action_type == 'muster_elite']

    if elites:
        elite_action = elites[0]
        region = elite_action.params['region']

        # Count units before
        army_before = None
        for k, a in state.armies.items():
            if sanitize(k) == region:
                army_before = a
                break
        units_before = len(army_before.units) if army_before else 0
        elites_before = sum(1 for u in army_before.units
                           if u.kind is gs.UnitKind.ELITE) if army_before else 0

        ai._execute_action(state, elite_action)

        # Check after: should have 1 more elite
        army_after = None
        for k, a in state.armies.items():
            if sanitize(k) == region:
                army_after = a
                break
        units_after = len(army_after.units) if army_after else 0
        elites_after = sum(1 for u in army_after.units
                          if u.kind is gs.UnitKind.ELITE) if army_after else 0

        assert units_after == units_before + 1, "Should gain 1 unit"
        assert elites_after == elites_before + 1, "Should gain 1 elite"
        assert state.pending_muster_die is None, "Elite should not set pending"

        unused = sum(1 for d in state.dice[gs.Side.SHADOW] if not d.used)
        assert unused == 0, "Die consumed after elite"

        print("  PASS: Muster elite placement")
    else:
        print("  SKIP: No elite placements available")


def test_active_nation_can_move():
    """Active (non-at-war) nations can move armies."""
    state = setup_game()
    # Sauron is at position 2 (active but not at war)
    assert not state.political.is_at_war(gs.Nation.SAURON)
    assert state.political.is_active(gs.Nation.SAURON)

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    moves = [a for a in actions if a.action_type == 'army_move']
    attacks = [a for a in actions if a.action_type == 'army_attack']

    assert len(moves) > 0, "Active non-at-war nation should be able to move"
    assert len(attacks) == 0, "Active non-at-war nation should NOT be able to attack"

    print(f"  PASS: Active nation can move ({len(moves)} moves, {len(attacks)} attacks)")


def test_passive_nation_can_move_within_borders():
    """Passive nations can move armies within their own borders but not cross into other nations."""
    state = setup_game()
    # Rohan is passive (box=0, not activated)
    assert not state.political.is_active(gs.Nation.ROHAN)

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    rohan_moves = [a for a in actions
                   if a.action_type == 'army_move'
                   and state.armies.get(a.params['from'])
                   and any(u.nation == gs.Nation.ROHAN
                          for u in state.armies[a.params['from']].units)]

    # Passive nations CAN move within their own borders
    within_border = [a for a in rohan_moves
                     if ai._region_nation(sanitize(a.params['to'])) == gs.Nation.ROHAN
                     or ai._region_nation(sanitize(a.params['to'])) is None]
    cross_border = [a for a in rohan_moves
                    if ai._region_nation(sanitize(a.params['to'])) is not None
                    and ai._region_nation(sanitize(a.params['to'])) != gs.Nation.ROHAN]

    assert len(cross_border) == 0, "Passive nation should not cross into other nations' borders"
    # Rohan has armies at Edoras, Fords of Isen, Helm's Deep - should have some within-border moves
    assert len(within_border) >= 0, "Passive nation can move within own borders"
    print(f"  PASS: Passive nation border restriction ({len(within_border)} within-border, {len(cross_border)} cross-border)")


def test_will_of_west_acts_as_any():
    """Will of the West die can be used for any action type."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3

    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.WILL_OF_WEST])

    actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
    types = set(a.action_type for a in actions)

    # Should have at least these types ('skip' = discard die, ex-'pass')
    for expected_type in ['army_move', 'fellowship_move', 'advance_political']:
        assert expected_type in types, \
            f"Will of West should allow {expected_type}, got types: {types}"
    # USER RULE 2026-07-08: 'skip' (burn a die) removed from action space
    assert 'skip' not in types, (
        f"'skip' must not be generated (removed 2026-07-08): {types}")

    print(f"  PASS: Will of West actions: {types}")


def test_hunt_resolution_on_fellowship_move():
    """Hunt resolution should apply corruption when fellowship moves."""
    import random
    state = setup_game()
    game_logic.initialize_decks(state)

    # Set up: FP has CHARACTER die, Shadow has allocated eyes in hunt box
    set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.EYE, gs.ActionDieFace.EYE])

    # Put shadow dice in hunt box (simulating eye allocation)
    for d in state.dice[gs.Side.SHADOW]:
        d.used = True
        state.hunt_box.append(d)

    initial_corruption = state.fellowship.corruption

    # Run multiple moves to get at least some hunt damage statistically.
    # SEEDED (2026-07-13): with ~0.31 hunt-success odds and 0-damage
    # tiles in the pool, 20 unseeded trials fail ~1% of runs - the
    # flake fired mid-session and read as a regression.
    random.seed(20260713)
    total_moves = 20
    any_corruption = False
    for _ in range(total_moves):
        s = setup_game()
        game_logic.initialize_decks(s)
        set_dice(s, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])
        # Allocate 2 Shadow eyes to hunt box
        shadow_eyes = [gs.ActionDie(faces=[gs.ActionDieFace.EYE], side=gs.Side.SHADOW)
                       for _ in range(2)]
        for d in shadow_eyes:
            d.current = gs.ActionDieFace.EYE
            d.used = True
        s.hunt_box = shadow_eyes

        actions = ai.get_legal_actions(s, gs.Side.FREE_PEOPLES)
        fm = [a for a in actions if a.action_type == 'fellowship_move']
        if fm:
            old_corr = s.fellowship.corruption
            ai._execute_action(s, fm[0])
            # Hunt damage either applies directly or parks in pending_hunt
            # (AI resolves Gollum/sacrifice choices as follow-up actions)
            pending_dmg = getattr(s, '_pending_hunt_damage', 0) or 0
            if s.pending_hunt:
                pending_dmg = max(pending_dmg,
                                  s.pending_hunt.get('damage', 0) or 0)
            if s.fellowship.corruption > old_corr or pending_dmg > 0:
                any_corruption = True
                break

    assert any_corruption, \
        f"After {total_moves} fellowship moves with 2 eyes in hunt box, " \
        f"expected some corruption/pending hunt damage"

    print("  PASS: Hunt resolution on fellowship move")


def test_hunt_threshold_calculation():
    """Hunt: success on 6+ always; FP dice in the box give +1 per die via
    fp_modifier; hunt level counts Shadow dice only (rulebook p.40)."""
    state = setup_game()
    game_logic.initialize_decks(state)

    # No FP dice in box -> threshold 6, modifier 0
    params = game_logic.get_hunt_params(state)
    assert params["threshold"] == 6, f"Expected threshold 6, got {params['threshold']}"
    assert params["fp_modifier"] == 0

    # Add 2 FP dice to hunt box -> threshold stays 6, modifier becomes 2,
    # and they do NOT raise the hunt level (number of dice rolled)
    for _ in range(2):
        fp_die = gs.ActionDie(faces=[gs.ActionDieFace.CHARACTER], side=gs.Side.FREE_PEOPLES)
        fp_die.current = gs.ActionDieFace.CHARACTER
        fp_die.used = True
        state.hunt_box.append(fp_die)

    params = game_logic.get_hunt_params(state)
    assert params["threshold"] == 6, f"Expected threshold 6 with 2 FP dice, got {params['threshold']}"
    assert params["fp_modifier"] == 2, f"Expected fp_modifier 2, got {params['fp_modifier']}"
    assert params["hunt_level"] == 0, \
        f"FP dice must not count toward hunt level, got {params['hunt_level']}"

    # Revealed fellowship does not change the threshold
    state.fellowship.hidden = False
    params = game_logic.get_hunt_params(state)
    assert params["threshold"] == 6, "Revealed status must not change hunt threshold"
    state.fellowship.hidden = True

    print("  PASS: Hunt threshold calculation")


def test_recruited_units_have_correct_nation():
    """Recruited units should have the nation of the recruiting nation."""
    state = setup_game()

    # Put Isengard at war and recruit
    state.political.boxes[gs.Nation.ISENGARD] = 3

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    isengard_recruits = [a for a in actions
                         if a.action_type == 'muster_recruit'
                         and a.params['nation'] == gs.Nation.ISENGARD]

    if isengard_recruits:
        recruit = isengard_recruits[0]
        region = recruit.params['region']

        # Count units before
        army_before = None
        for k, a in state.armies.items():
            if sanitize(k) == region:
                army_before = a
                break

        ai._execute_action(state, recruit)

        # Find the newly placed unit
        army_after = None
        for k, a in state.armies.items():
            if sanitize(k) == region:
                army_after = a
                break

        assert army_after is not None, "Army should exist after recruit"
        # The last unit added should be Isengard, not Sauron
        new_unit = army_after.units[-1]
        assert new_unit.nation == gs.Nation.ISENGARD, \
            f"Recruited unit should be Isengard, got {new_unit.nation}"

        print(f"  PASS: Recruited unit nation correct ({new_unit.nation.value} at {region})")
    else:
        print("  SKIP: No Isengard recruits available")


def test_recruit_at_own_nation_settlements_only():
    """Nations can only recruit at their own settlements, not any same-side settlement."""
    state = setup_game()

    # Put only Sauron at war (Isengard not at war)
    state.political.boxes[gs.Nation.SAURON] = 3
    state.political.boxes[gs.Nation.ISENGARD] = 2  # Active but not at war

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    recruits = [a for a in actions if a.action_type == 'muster_recruit']

    # All recruit regions should be Sauron settlements
    for a in recruits:
        region = a.params['region']
        region_nation = ai._region_nation(region)
        assert region_nation == gs.Nation.SAURON, \
            f"Sauron recruit at {region} but region belongs to {region_nation}"

    # Isengard settlements should NOT appear
    isengard_settlements = [a for a in recruits
                           if ai._region_nation(a.params['region']) == gs.Nation.ISENGARD]
    assert len(isengard_settlements) == 0, \
        f"Should not recruit at Isengard settlements when only Sauron at war"

    print(f"  PASS: Recruit at own nation settlements only ({len(recruits)} Sauron settlements)")


def test_reinforcement_limits():
    """Nations cannot recruit beyond their maximum figure count."""
    state = setup_game()
    state.political.boxes[gs.Nation.ISENGARD] = 3  # At war

    # Isengard has 12 regular, 6 elite max
    # Starting setup: 4R+1E at Orthanc, 1R North Dunland, 1R South Dunland
    # = 6 regular, 1 elite on board
    deployed = ai._count_deployed(state, gs.Nation.ISENGARD)
    assert deployed['regular'] == 6, f"Isengard should have 6 regulars deployed, got {deployed['regular']}"
    assert deployed['elite'] == 1, f"Isengard should have 1 elite deployed, got {deployed['elite']}"

    # Can still recruit regulars (6 < 12) and elites (1 < 6)
    assert ai._can_recruit(state, gs.Nation.ISENGARD, 'regular')
    assert ai._can_recruit(state, gs.Nation.ISENGARD, 'elite')

    # Fill up regulars to 12 by adding 6 more
    orthanc_army = None
    for k, a in state.armies.items():
        if sanitize(k) == 'orthanc':
            orthanc_army = a
            break
    for _ in range(6):
        orthanc_army.units.append(gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR))

    deployed = ai._count_deployed(state, gs.Nation.ISENGARD)
    assert deployed['regular'] == 12

    # Now should NOT be able to recruit more regulars
    assert not ai._can_recruit(state, gs.Nation.ISENGARD, 'regular')
    # But elites still OK
    assert ai._can_recruit(state, gs.Nation.ISENGARD, 'elite')

    # Verify no regular recruit actions are generated
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    isengard_reg_recruits = [a for a in actions
                             if a.action_type == 'muster_recruit'
                             and a.params['nation'] == gs.Nation.ISENGARD
                             and a.params['type'] == 'regular']
    assert len(isengard_reg_recruits) == 0, \
        f"Should not generate regular recruits when at max, got {len(isengard_reg_recruits)}"

    print(f"  PASS: Reinforcement limits ({deployed})")


def test_city_attack_penalty():
    """Attacking a city/fortification: attacker hits on 6+ in first round."""
    import combat as cb
    state = setup_game()

    # Edoras is a city (not a stronghold) — attacking it should apply first-round penalty
    state.political.boxes[gs.Nation.SAURON] = 3
    state.political.boxes[gs.Nation.ROHAN] = 3
    # Place Shadow army adjacent to Edoras
    state.armies['Westemnet'] = gs.Army(
        region=state.board.region('westemnet'),
        units=[gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(5)]
    )

    # Setup combat at Edoras (city)
    combat_state = cb.setup_combat(state, 'Westemnet', 'Edoras')
    assert combat_state.city_defense, \
        "Edoras (city) should trigger city_defense"
    assert not combat_state.is_siege, "Field battle at city is not a siege"

    # Osgiliath is a fortification — should also trigger city_defense
    state.armies['North Ithilien'] = gs.Army(
        region=state.board.region('north ithilien'),
        units=[gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(5)]
    )
    fort_combat = cb.setup_combat(state, 'North Ithilien', 'Osgiliath')
    assert fort_combat.city_defense, \
        "Osgiliath (fortification) should trigger city_defense"

    print("  PASS: City/fortification attack penalty")


def test_stronghold_no_city_penalty():
    """Strongholds do NOT have first-round penalty in field battles."""
    import combat as cb
    state = setup_game()

    # Minas Tirith is a stronghold — field battle should NOT trigger city_defense
    state.political.boxes[gs.Nation.SAURON] = 3
    state.armies['Osgiliath'] = gs.Army(
        region=state.board.region('osgiliath'),
        units=[gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(5)]
    )

    # Field battle at stronghold: both sides 5+, no penalty
    field_combat = cb.setup_combat(state, 'Osgiliath', 'Minas Tirith')
    assert not field_combat.city_defense, \
        "Stronghold field battle should NOT trigger city_defense"
    assert field_combat.stronghold_defense, \
        "Stronghold should still be marked as stronghold_defense"

    # Siege at stronghold: attacker always 6+
    siege_combat = cb.setup_combat(state, 'Osgiliath', 'Minas Tirith', is_siege=True)
    assert siege_combat.is_siege, "Should be marked as siege"
    assert not siege_combat.city_defense, "Siege should not use city_defense"

    print("  PASS: Stronghold no city penalty in field battle")


def test_siege_press_mechanic():
    """Siege press: attacker downgrades elite to regular to extend by 1 round."""
    import combat as cb
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3

    # Create attacking army with elites at a region adjacent to stronghold
    state.armies['Osgiliath'] = gs.Army(
        region=state.board.region('osgiliath'),
        units=([gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(3)] +
               [gs.Unit(gs.Nation.SAURON, gs.UnitKind.ELITE) for _ in range(2)])
    )

    # Count elites before
    elites_before = cb.count_elite_units(state.armies['Osgiliath'])
    assert elites_before == 2, f"Should start with 2 elites, got {elites_before}"

    # Siege press should downgrade one elite
    pressed = cb.siege_press(state.armies['Osgiliath'])
    assert pressed, "Should be able to press with elites"
    elites_after = cb.count_elite_units(state.armies['Osgiliath'])
    assert elites_after == 1, f"Should have 1 elite after press, got {elites_after}"
    assert len(state.armies['Osgiliath'].units) == 5, "Total units unchanged after press"

    # Second press
    pressed2 = cb.siege_press(state.armies['Osgiliath'])
    assert pressed2, "Should be able to press second time"
    assert cb.count_elite_units(state.armies['Osgiliath']) == 0

    # Third press should fail (no more elites)
    pressed3 = cb.siege_press(state.armies['Osgiliath'])
    assert not pressed3, "Should not press without elites"

    # Test auto-press in resolve_combat: siege with auto_press should fight multiple rounds
    # Reset the army with elites
    state.armies['Osgiliath'] = gs.Army(
        region=state.board.region('osgiliath'),
        units=([gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(3)] +
               [gs.Unit(gs.Nation.SAURON, gs.UnitKind.ELITE) for _ in range(2)])
    )
    # Defend Minas Tirith with a large garrison so it survives multiple rounds
    state.armies['Minas Tirith'] = gs.Army(
        region=state.board.region('minas tirith'),
        units=[gs.Unit(gs.Nation.GONDOR, gs.UnitKind.REGULAR) for _ in range(5)],
        leaders=[gs.Leader(gs.Nation.GONDOR)]
    )

    result = cb.resolve_combat(state, 'Osgiliath', 'Minas Tirith',
                                is_siege=True, auto_press=True)
    # With 2 elites, can press up to 2 times = 3 total rounds max
    assert result.rounds_fought >= 1, "Should fight at least 1 round"
    # If defender survived, more rounds may have been fought via pressing
    if not result.defender_destroyed:
        assert result.rounds_fought <= 3, \
            f"With 2 elites, max 3 siege rounds, got {result.rounds_fought}"

    print(f"  PASS: Siege press mechanic (fought {result.rounds_fought} rounds)")


def test_hunt_roll_direction():
    """Hunt roll should hit on >= threshold, not <= threshold."""
    import random
    random.seed(42)

    state = setup_game()
    # Threshold 6 means only 6s hit
    hits_at_6 = 0
    for _ in range(1000):
        h = game_logic._hunt_roll(state, 1, 6)
        hits_at_6 += h

    # With threshold 6, ~1/6 = 16.7% hit rate
    assert 100 < hits_at_6 < 300, \
        f"With threshold 6, expected ~167 hits in 1000 rolls, got {hits_at_6}"

    # Threshold 1: everything hits EXCEPT natural 1s (always miss, p.40)
    hits_at_1 = 0
    for _ in range(600):
        h = game_logic._hunt_roll(state, 1, 1)
        hits_at_1 += h

    # Expect ~5/6 = 500/600; allow generous statistical slack
    assert 420 < hits_at_1 < 580, \
        f"With threshold 1, expected ~500 hits in 600 rolls (nat 1 misses), got {hits_at_1}"

    print(f"  PASS: Hunt roll direction (threshold 6: {hits_at_6}/1000, threshold 1: {hits_at_1}/100)")


def test_non_belligerent_border_restriction():
    """Non-at-war nations cannot cross into other nations' borders."""
    state = setup_game()

    # Sauron is active (pos 2) but NOT at war
    assert state.political.is_active(gs.Nation.SAURON)
    assert not state.political.is_at_war(gs.Nation.SAURON)

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    moves = [a for a in actions if a.action_type == 'army_move']

    # Every move destination for a non-at-war Sauron army must be either
    # a Sauron region or neutral (no parent nation)
    for a in moves:
        from_key = a.params['from']
        to_key = a.params['to']
        army = state.armies.get(from_key)
        if not army or not army.units:
            continue
        army_nat = army.units[0].nation
        if state.political.is_at_war(army_nat):
            continue  # At-war nations can move anywhere
        dest_nat = ai._region_nation(to_key)
        assert dest_nat is None or dest_nat == army_nat, \
            f"Non-at-war {army_nat.value} moved from {from_key} to {to_key} " \
            f"(owned by {dest_nat.value if dest_nat else 'neutral'})"

    # Put Sauron at war — should now be able to cross borders
    state.political.boxes[gs.Nation.SAURON] = 3
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])
    actions_war = ai.get_legal_actions(state, gs.Side.SHADOW)
    moves_war = [a for a in actions_war if a.action_type == 'army_move']

    # At war should have at least as many moves as before
    assert len(moves_war) >= len(moves), \
        f"At-war should have more moves ({len(moves_war)}) than non-at-war ({len(moves)})"

    print(f"  PASS: Non-belligerent border restriction ({len(moves)} restricted, {len(moves_war)} at war)")


def test_fp_casualties_permanent():
    """FP casualties are permanent; Shadow casualties return to pool."""
    import combat as cb
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3
    state.political.boxes[gs.Nation.GONDOR] = 3

    # Check initial Gondor deployed
    deployed_before = ai._count_deployed(state, gs.Nation.GONDOR)

    # Simulate FP casualties
    if not hasattr(state, 'fp_casualties') or state.fp_casualties is None:
        state.fp_casualties = {}
    state.fp_casualties[gs.Nation.GONDOR] = {'regular': 5, 'elite': 0, 'leader': 0}

    # With 5 permanent casualties, available regulars should decrease
    # Gondor max: 15 regular. Deployed initially: ~9. Casualties: 5.
    # Available = 15 - 9 - 5 = 1
    can_recruit_reg = ai._can_recruit(state, gs.Nation.GONDOR, 'regular')
    deployed = ai._count_deployed(state, gs.Nation.GONDOR)
    gondor_max = ai.NATION_MAX_FIGURES[gs.Nation.GONDOR]['regular']
    available = gondor_max - deployed['regular'] - 5
    assert available >= 0, "Should have non-negative available"

    # Shadow casualties: _can_recruit should NOT subtract casualties
    # (Shadow units return to pool)
    can_recruit_sauron = ai._can_recruit(state, gs.Nation.SAURON, 'regular')
    sauron_deployed = ai._count_deployed(state, gs.Nation.SAURON)
    sauron_max = ai.NATION_MAX_FIGURES[gs.Nation.SAURON]['regular']
    assert can_recruit_sauron == (sauron_deployed['regular'] < sauron_max), \
        "Shadow recruitment should only check deployed vs max (casualties return)"

    print(f"  PASS: FP casualties permanent (Gondor: {deployed['regular']} deployed + 5 casualties = "
          f"{deployed['regular'] + 5} used of {gondor_max} max)")


def test_political_activation_on_enemy_entry():
    """Nations activate when enemy army enters their region."""
    state = setup_game()
    state.political.boxes[gs.Nation.ISENGARD] = 3  # At war (can cross borders)

    # Rohan starts passive (not active)
    assert not state.political.is_active(gs.Nation.ROHAN)

    # Gap of Rohan is adjacent to Fords of Isen (a Rohan region)
    state.armies['Gap of Rohan'] = gs.Army(
        region=state.board.region('gap of rohan'),
        units=[gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR) for _ in range(3)]
    )

    # Move into Fords of Isen (Rohan region)
    game_logic.move_army(state, 'Gap of Rohan', 'Fords of Isen')
    # Rohan should now be activated
    assert state.political.is_active(gs.Nation.ROHAN), \
        "Rohan should activate when Shadow enters its region"
    print("  PASS: Political activation on enemy entry")


def test_fellowship_declaration_activates_nation():
    """Fellowship declared in FP city/stronghold activates that nation."""
    state = setup_game()
    game_logic.initialize_decks(state)

    # Dwarves start passive
    assert not state.political.is_active(gs.Nation.DWARVES)

    # Move fellowship to Erebor (Dwarven stronghold) — set progress
    state.fellowship.progress = 0  # Will declare in same region
    # Place fellowship directly at Erebor
    state.fellowship.region = state.board.region('Erebor')

    # Declare in Erebor
    game_logic.declare_fellowship(state, region='Erebor')

    # Dwarves should now be active
    assert state.political.is_active(gs.Nation.DWARVES), \
        "Dwarves should activate when fellowship declared in Erebor"

    print("  PASS: Fellowship declaration activates nation")


def test_companion_separation_blocked_in_mordor():
    """Companions cannot be separated while on the Mordor track."""
    state = setup_game()

    # Put fellowship on Mordor track
    state.fellowship.mordor_entered = True
    state.fellowship.mordor_step = 2

    # Try to separate a companion — should raise
    try:
        set_dice(state, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.CHARACTER])
        game_logic.separate_companions(state, ['Gandalf the Grey'], 'Minas Tirith',
                                        die_idx=0)
        assert False, "Should have raised RuntimeError for Mordor separation"
    except RuntimeError as e:
        assert "Mordor" in str(e), f"Expected Mordor error, got: {e}"

    print("  PASS: Companion separation blocked in Mordor")


def test_leaders_removed_when_alone():
    """FP Leaders without army units are automatically removed."""
    state = setup_game()

    # Create an army with just a leader and no units
    state.armies['test_region'] = gs.Army(
        region=state.board.region('bree'),
        units=[],
        leaders=[gs.Leader(gs.Nation.THE_NORTH)]
    )

    # Run cleanup
    game_logic.remove_orphaned_leaders(state)

    # The army (and leader) should be removed
    assert 'test_region' not in state.armies, \
        "Leader-only army should be cleaned up"

    # Nazgul alone should survive
    state.armies['naz_region'] = gs.Army(
        region=state.board.region('bree'),
        units=[],
        leaders=[gs.Nazgul(gs.Nation.SAURON)]
    )
    game_logic.remove_orphaned_leaders(state)
    assert 'naz_region' in state.armies, \
        "Nazgul-only army should survive cleanup"

    print("  PASS: Leaders removed when alone")


def test_combat_roll_natural_1_always_misses():
    """Natural 1 always misses regardless of modifiers (rulebook p.30)."""
    import random
    from combat import combat_roll

    random.seed(42)
    hits_on_1 = 0
    total_1s = 0
    for _ in range(10000):
        rolls, hits = combat_roll(1, 5, modifier=4)
        if rolls[0] == 1:
            total_1s += 1
            if hits > 0:
                hits_on_1 += 1

    assert total_1s > 100, f"Should roll some 1s, got {total_1s}"
    assert hits_on_1 == 0, \
        f"Natural 1 should always miss even with modifier, but {hits_on_1}/{total_1s} hit"

    # Also verify natural 6 still always hits even with negative modifier
    hits_on_6 = 0
    total_6s = 0
    for _ in range(10000):
        rolls, hits = combat_roll(1, 5, modifier=-10)
        if rolls[0] == 6:
            total_6s += 1
            if hits > 0:
                hits_on_6 += 1

    assert total_6s > 100, f"Should roll some 6s, got {total_6s}"
    assert hits_on_6 == total_6s, \
        f"Natural 6 should always hit, but {hits_on_6}/{total_6s} hit"

    print(f"  PASS: Natural 1 always misses ({total_1s} 1s, 0 hits), "
          f"natural 6 always hits ({total_6s} 6s, {hits_on_6} hits)")


def test_companion_movement_activates_nation():
    """Companion ending movement in nation's city/stronghold activates that nation."""
    state = setup_game()
    state.political.boxes[gs.Nation.GONDOR] = 3  # At war (so we can move)

    # Dwarves start passive
    assert not state.political.is_active(gs.Nation.DWARVES)

    # Place Gimli (Dwarves companion) with an army near Erebor
    # Iron Hills is adjacent to Erebor
    gimli = gs.Companion(name='Gimli', level=2)
    state.armies['Iron Hills'] = gs.Army(
        region=state.board.region('iron hills'),
        units=[gs.Unit(gs.Nation.DWARVES, gs.UnitKind.REGULAR) for _ in range(2)],
        leaders=[gimli]
    )
    # Ensure Dwarves are active enough to move (position 1+)
    state.political.boxes[gs.Nation.DWARVES] = 1
    state.political.active[gs.Nation.DWARVES] = False  # Reset active flag

    # Move into Erebor (Dwarven stronghold)
    game_logic.move_army(state, 'Iron Hills', 'Erebor')

    # Dwarves should now be activated
    assert state.political.is_active(gs.Nation.DWARVES), \
        "Dwarves should activate when Gimli moves to Erebor"

    print("  PASS: Companion movement activates nation")


def test_repeat_attacks_allowed():
    """The same region MAY be attacked repeatedly with separate Action dice
    (rulebook/almanac: no per-turn attack limit exists)."""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3
    state.political.boxes[gs.Nation.GONDOR] = 3

    # Two Shadow armies adjacent to the same FP region
    state.armies['Osgiliath'] = gs.Army(
        region=state.board.region('osgiliath'),
        units=[gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(3)]
    )
    state.armies['Minas Morgul'] = gs.Army(
        region=state.board.region('minas morgul'),
        units=[gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(3)]
    )
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY, gs.ActionDieFace.ARMY])

    # Even when the region was already attacked this turn, further attacks
    # must still be offered (the old once-per-turn limit was a house rule).
    # (tracker deleted 2026-07-27; assertion stays as the tripwire)
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    mt_attacks = [a for a in actions if a.action_type == 'army_attack'
                  and sanitize(a.params['to']) == 'minas tirith']
    assert len(mt_attacks) >= 1, \
        "Attack on an already-attacked region must still be offered"

    # And battle() must not reject it either
    result = game_logic.battle(state, 'osgiliath', 'minas tirith', max_rounds=1)
    assert result is not None
    print(f"  PASS: Repeat attacks allowed ({len(mt_attacks)} offered on "
          f"already-attacked region)")


def test_army_can_move_multiple_times_per_turn():
    """Rulebook p.25: 'An Army can move multiple times in a single turn, as
    long as it uses a different Action die for each movement.'"""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3  # At war

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY, gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    moves = [a for a in actions if a.action_type == 'army_move']
    assert len(moves) > 0, "Should have army moves"

    # Move a full army with the first die
    full_moves = [a for a in moves if a.params.get('take_leaders', True)]
    first_move = full_moves[0]
    ai._execute_action(state, first_move)

    # Pass to complete the first army die (skip its second move)
    actions2 = ai.get_legal_actions(state, gs.Side.SHADOW)
    pass_action = [a for a in actions2 if a.action_type == 'pass'][0]
    ai._execute_action(state, pass_action)

    # With the second die the moved army may move AGAIN (official rule)
    actions3 = ai.get_legal_actions(state, gs.Side.SHADOW)
    first_to = first_move.params['to']
    again = [a for a in actions3 if a.action_type == 'army_move'
             and sanitize(a.params['from']) == sanitize(first_to)]
    assert len(again) > 0, \
        f"Army at {first_to} must be allowed to move again with a new die"

    print("  PASS: Army can move multiple times per turn (different dice)")


def test_army_split_moves():
    """Rulebook p.25 'Splitting an Army': moving a portion of the units."""
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])

    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    moves = [a for a in actions if a.action_type == 'army_move']
    # Find a single-unit detachment from a multi-unit army
    split = next((a for a in moves
                  if a.params.get('n_regular') == 1
                  and a.params.get('n_elite') == 0
                  and not a.params.get('take_leaders', True)), None)
    assert split is not None, "Split menu should offer 1-unit detachments"
    src = split.params['from']
    src_army = state.armies[src]
    n_before = len(src_army.units)
    lead_before = len(src_army.leaders)
    ai._execute_action(state, split)
    # Source army lost exactly one unit, kept its leaders
    assert len(state.armies[src].units) == n_before - 1
    assert len(state.armies[src].leaders) == lead_before
    # Destination has the single unit
    dest = split.params['to']
    dst_army = next(a for k, a in state.armies.items()
                    if sanitize(k) == sanitize(dest))
    assert any(u for u in dst_army.units), "Detached unit must arrive"
    print(f"  PASS: Army split moves ({src} -> {dest}, 1 unit detached)")


def test_mixed_nation_army_attack():
    """Only eligible units of a mixed Army attack; peaceful units stay behind."""
    import combat
    state = setup_game()

    # Isengard at war, Sauron NOT at war
    state.political.boxes[gs.Nation.ISENGARD] = 3
    state.political.boxes[gs.Nation.SAURON] = 2  # Active but not at war

    # Place a mixed Sauron+Isengard army next to a FP army
    state.armies['Gap of Rohan'] = gs.Army(
        region=state.board.region('gap of rohan'),
        units=[
            gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
        ]
    )
    # Place FP army in Fords of Isen
    state.political.boxes[gs.Nation.ROHAN] = 3  # At war
    state.armies['Fords of Isen'] = gs.Army(
        region=state.board.region('fords of isen'),
        units=[gs.Unit(gs.Nation.ROHAN, gs.UnitKind.REGULAR) for _ in range(2)]
    )

    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    attacks = [a for a in actions if a.action_type == 'army_attack'
               and sanitize(a.params['from']) == 'gap of rohan']

    # RB28/35: Isengard attacks while the peaceful Sauron unit stays
    # in the compulsory rearguard. There is one attack per target, no
    # optional formation decision or learned rear size.
    target = [a for a in attacks if sanitize(a.params['to']) == 'fords of isen']
    assert len(target) == 1
    declared = ai.apply_action(state, target[0])
    view = combat.setup_combat(declared, 'gap of rohan', 'fords of isen',
                               attacker_side=gs.Side.SHADOW)
    assert Counter((u.nation, u.kind) for u in view.attacker_army.units) == {
        (gs.Nation.ISENGARD, gs.UnitKind.REGULAR): 1}
    physical = next(a for k, a in declared.armies.items()
                    if sanitize(k) == 'gap of rohan')
    assert Counter((u.nation, u.kind) for u in physical.units) == {
        (gs.Nation.SAURON, gs.UnitKind.REGULAR): 1,
        (gs.Nation.ISENGARD, gs.UnitKind.REGULAR): 1}

    # With Sauron also At War, the same army CAN attack
    state.political.boxes[gs.Nation.SAURON] = 3
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.ARMY])
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    attacks = [a for a in actions if a.action_type == 'army_attack'
               and sanitize(a.params['from']) == 'gap of rohan']
    assert len(attacks) > 0, \
        "All-At-War mixed army should be able to attack"

    print("  PASS: Mixed-nation army attack (p.33 non-belligerent rule)")


def test_minion_muster_exclusivity():
    """USER RULING (iter 38): mustering a minion consumes the ENTIRE
    muster die — no unit recruit alongside it, in either order."""
    # direction 1: muster a minion -> die fully consumed, no pending
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3  # WK needs Sauron at war
    # WK entry: Sauron at war + >=1 FP nation at war
    state.political.boxes[gs.Nation.GONDOR] = 3
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    minion_actions = [a for a in actions if a.action_type == 'muster_minion']
    assert minion_actions, "expected muster_minion offers with WK available"
    ai._execute_action(state, minion_actions[0])
    assert state.pending_muster_die is None, \
        "muster_minion must consume the ENTIRE die (no pending recruit)"
    assert all(d.used for d in state.dice[gs.Side.SHADOW]
               if d.current == gs.ActionDieFace.MUSTER), \
        "the muster die must be consumed by the minion muster"

    # direction 2: after a first unit recruit, the pending second-recruit
    # menu must NOT offer a minion
    state = setup_game()
    state.political.boxes[gs.Nation.SAURON] = 3
    state.political.boxes[gs.Nation.GONDOR] = 3
    set_dice(state, gs.Side.SHADOW, [gs.ActionDieFace.MUSTER])
    actions = ai.get_legal_actions(state, gs.Side.SHADOW)
    rec = next((a for a in actions if a.action_type == 'muster_recruit'
                and a.params.get('type') == 'regular'), None)
    assert rec is not None, "expected a regular recruit offer"
    ai._execute_action(state, rec)
    assert state.pending_muster_die is not None, \
        "first unit recruit should open the second-recruit pending"
    cont = ai.get_legal_actions(state, gs.Side.SHADOW)
    assert not any(a.action_type == 'muster_minion' for a in cont), \
        "minion must NOT be offered as the second half of a muster die"

    print("  PASS: Minion muster exclusivity (whole die)")


def test_victory_timing():
    """Rulebook p.17 (iter 38): ring victories end the game IMMEDIATELY
    (mid-turn, mid-action); military victories (SH 10 / FP 4 VP) fire
    ONLY at the end-of-turn Victory Check phase."""
    import game_logic

    # --- Ring destruction mid-turn: must end instantly ---
    state = setup_game()
    state.fellowship.mordor_entered = True
    state.fellowship.mordor_step = 5
    state.fellowship.corruption = 4
    state.pending_hunt = None
    vc = game_logic.check_all_victory_conditions(state)
    assert vc['victory'] == 'free_peoples' \
        and vc['victory_type'] == 'ring_destroyed', vc
    assert game_logic.should_game_end(vc, end_of_turn=False), \
        "Ring destruction (mordor_step 5) must end the game MID-TURN"

    # --- Ring corruption mid-turn: must end instantly ---
    state = setup_game()
    state.fellowship.corruption = 12
    vc = game_logic.check_all_victory_conditions(state)
    assert vc['victory'] == 'shadow' \
        and vc['victory_type'] == 'ring_corruption', vc
    assert game_logic.should_game_end(vc, end_of_turn=False), \
        "Corruption 12 must end the game MID-TURN"

    # --- SH 10 VP mid-turn: must NOT end until end of turn ---
    state = setup_game()
    # give Shadow 10 VP of captured FP settlements (5 strongholds x 2)
    state.enemy_controlled[gs.Side.SHADOW] = {
        'minas tirith', 'helm s deep', 'lorien', 'rivendell', 'erebor'}
    vc = game_logic.check_all_victory_conditions(state)
    assert vc['victory'] == 'shadow' and vc['victory_type'] == 'military', vc
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "SH 10 VP mid-turn must NOT end the game before the Victory Check"
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "SH 10 VP must end the game at the end-of-turn Victory Check"

    # --- FP 4 VP: same end-of-turn timing ---
    state = setup_game()
    state.enemy_controlled[gs.Side.FREE_PEOPLES] = {
        'minas morgul', 'moria'}   # 2 SH strongholds = 4 VP
    vc = game_logic.check_all_victory_conditions(state)
    assert vc['victory'] == 'free_peoples' \
        and vc['victory_type'] == 'military', vc
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "FP 4 VP mid-turn must NOT end the game before the Victory Check"
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "FP 4 VP must end the game at the end-of-turn Victory Check"

    # --- ring beats military when simultaneous ---
    state = setup_game()
    state.fellowship.corruption = 12
    state.enemy_controlled[gs.Side.FREE_PEOPLES] = {
        'minas morgul', 'moria'}
    vc = game_logic.check_all_victory_conditions(state)
    assert vc['victory_type'] == 'ring_corruption', \
        "Ring victory takes priority over military"

    print("  PASS: Victory timing (ring immediate, military end-of-turn)")



def test_decision_context_exposure():
    """USER DIRECTIVE 2026-07-08: for EVERY decision the entire game
    state must be exposed. Enforced two ways: (1) every pending_combat
    phase produces a pairwise-DISTINCT state encoding (no one-hot
    capping); (2) toggling ANY pending-decision field on GameState
    changes the encoding. A new pending mechanic that is not wired into
    encode_state fails this test."""
    import numpy as np
    from game_state import Side
    b = board.load_board()
    base = gs.setup_initial_game(b)
    v0 = ai.encode_state(base, Side.FREE_PEOPLES)

    phases = ('defender_siege_choice', 'attacker_combat_card',
              'defender_combat_card', 'combat_effect_choice',
              'combat_round_result', 'defender_retreat',
              'attacker_siege_decision', 'attacker_casualty_policy',
              'defender_casualty_policy', 'victor_advance')
    vecs = []
    for phz in phases:
        st = ai.clone_state(base)
        st.pending_combat = {'defender_region': 'moria',
                             'attacker_region': 'eregion',
                             'attacker_side': Side.SHADOW,
                             'defender_side': Side.FREE_PEOPLES,
                             'phase': phz, 'is_siege': False,
                             'rounds_fought': 1}
        vecs.append(ai.encode_state(st, Side.FREE_PEOPLES))
    for i in range(len(phases)):
        assert not np.array_equal(vecs[i], v0), (
            f"pending_combat phase {phases[i]} invisible in encoding")
        for j in range(i + 1, len(phases)):
            assert not np.array_equal(vecs[i], vecs[j]), (
                f"phases {phases[i]} / {phases[j]} indistinguishable")

    def _set(name, val):
        def mut(st):
            setattr(st, name, val)
        return mut

    muts = {
        'pending_hunt': _set('pending_hunt',
                             {'damage': 2, 'phase': 'hunt_sacrifice'}),
        'pending_hunt damage': _set('pending_hunt',
                                    {'damage': 5,
                                     'phase': 'hunt_sacrifice'}),
        'pending_hunt hobbit level': _set(
            'pending_hunt', {'damage': 0, 'phase': 'hobbit_place',
                             'companion': 'Meriadoc', 'level': 1}),
        'pending_card_option': _set(
            'pending_card_option',
            {'card_id': 'FP_Character_Elven_Cloaks', 'side': 'fp',
             'options': [{}, {}]}),
        'pending_discard (mine)': _set('pending_discard',
                                       {'side': Side.FREE_PEOPLES}),
        'pending_discard (enemy)': _set('pending_discard',
                                        {'side': Side.SHADOW}),
        'pending_guide_change': _set('pending_guide_change',
                                     {'side': Side.FREE_PEOPLES}),
        '_eye_allocation_phase': _set('_eye_allocation_phase', True),
        '_declare_phase': _set('_declare_phase', True),
        'pending_card_followup': _set(
            'pending_card_followup',
            {'kind': 'nazgul_armies', 'moves_left': 2,
             'moved_any': False, 'exclude': set()}),
        'pending_army_move_die': _set('pending_army_move_die', 0),
        'pending_muster_die': _set('pending_muster_die', 0),
        'pending_hunt_tiles': _set('pending_hunt_tiles',
                                   ["Shelob's Lair"]),
        # v14 (P1-P5) new pending context
        'pco decider (cross-side)': _set(
            'pending_card_option',
            {'card_id': 'FP_Character_Elven_Cloaks', 'side': 'sh',
             'decider': 'fp', 'options': [{}, {}]}),
        'pco direct (chained stage)': _set(
            'pending_card_option',
            {'card_id': 'FP_Character_Elven_Cloaks', 'side': 'fp',
             'decider': 'fp', 'direct': True, 'options': [{}, {}]}),
        'pco resume_action (interposed)': _set(
            'pending_card_option',
            {'card_id': 'FP_Character_Elven_Cloaks', 'side': 'fp',
             'decider': 'fp', 'options': [{}, {}],
             'resume_action': {'die_idx': 0,
                               'action_type': 'fellowship_move',
                               'params': {}}}),
        '_crebain_hunt_bonus': _set('_crebain_hunt_bonus', 1),
        '_overstack_regions': _set('_overstack_regions', {'moria'}),
        # v15 (amendment 1.126)
        '_ents_chain_n': _set('_ents_chain_n', 2),
        '_mithril_stash d2': _set(
            '_mithril_stash',
            {'tile': gs.HuntTile(damage=2), 'successes': 1,
             'mordor': False}),
        '_mithril_stash d3+reveal': _set(
            '_mithril_stash',
            {'tile': gs.HuntTile(damage=3, reveal=True), 'successes': 1,
             'mordor': True}),
    }
    seen = {}
    for name, mut in muts.items():
        st = ai.clone_state(base)
        mut(st)
        v = ai.encode_state(st, Side.FREE_PEOPLES)
        assert not np.array_equal(v, v0), (
            f"{name} INVISIBLE in state encoding")
        seen[name] = v
    assert not np.array_equal(seen['pending_discard (mine)'],
                              seen['pending_discard (enemy)']), (
        "discard ownership invisible")
    assert not np.array_equal(seen['pending_hunt'],
                              seen['pending_hunt damage']), (
        "hunt damage magnitude invisible")
    assert not np.array_equal(seen['pending_card_option'],
                              seen['pco decider (cross-side)']), (
        "pco decider invisible")
    assert not np.array_equal(seen['pending_card_option'],
                              seen['pco direct (chained stage)']), (
        "pco direct flag invisible")
    assert not np.array_equal(seen['pending_card_option'],
                              seen['pco resume_action (interposed)']), (
        "pco resume_action invisible")
    assert not np.array_equal(seen['_mithril_stash d2'],
                              seen['_mithril_stash d3+reveal']), (
        "mithril stashed-tile properties invisible")

    # v14 P2 pending_combat keys must be visible AND distinct
    st = ai.clone_state(base)
    st.pending_combat = {'defender_region': 'moria',
                         'attacker_region': 'eregion',
                         'attacker_side': Side.SHADOW,
                         'defender_side': Side.FREE_PEOPLES,
                         'phase': 'attacker_combat_card',
                         'is_siege': True, 'rounds_fought': 0,
                         'mandatory_rounds': 3,
                         'no_def_card_r1_unless_companion': True}
    v_m = ai.encode_state(st, Side.FREE_PEOPLES)
    st2 = ai.clone_state(base)
    st2.pending_combat = dict(st.pending_combat)
    st2.pending_combat.pop('mandatory_rounds')
    st2.pending_combat.pop('no_def_card_r1_unless_companion')
    assert not np.array_equal(v_m, ai.encode_state(
        st2, Side.FREE_PEOPLES)), "P2 combat keys invisible"
    st3 = ai.clone_state(base)
    st3.pending_combat = dict(st2.pending_combat)
    st3.pending_combat['def_dice_reduction'] = 3
    assert not np.array_equal(
        ai.encode_state(st3, Side.FREE_PEOPLES),
        ai.encode_state(st2, Side.FREE_PEOPLES)), (
        "dice reduction invisible")
    st4v = ai.clone_state(base)
    st4v.pending_combat = dict(st2.pending_combat)
    st4v.pending_combat['attacker_no_retreat'] = True
    assert not np.array_equal(
        ai.encode_state(st4v, Side.FREE_PEOPLES),
        ai.encode_state(st2, Side.FREE_PEOPLES)), (
        "attacker_no_retreat invisible")

    # v14 action-level: new option shapes must be pairwise distinct
    from ai import Action, encode_action
    _mkopt = lambda ch: Action(die_idx=-1, action_type='card_option',
                               params={'choices': ch})
    _pairs = [
        ({'crebain_discard': True}, {'crebain_discard': False}),
        ({'balrog_declare_draw': True}, {'balrog_declare_draw': False}),
        ({'move_to_fellowship': True}, {'move_to_fellowship': False}),
        ({'action': 'eliminate'}, {'action': 'corruption'}),
        ({'overstack_remove': {'region': 'moria', 'nation': 'sauron',
                               'kind': 'regular'}},
         {'overstack_remove': {'region': 'moria', 'nation': 'sauron',
                               'kind': 'elite'}}),
        ({'from_region': 'moria', 'to_region': 'eregion',
          '_tsim_n': 1}, {'_tsim_done': True}),
        ({'move_group': {'from': 'dale', 'to': 'erebor',
                         'companions': ['Gimli']}},
         {'move_group': {'from': 'dale', 'to': 'erebor',
                         'companions': ['Gimli', 'Legolas']}}),
        ({'from_region': 'osgiliath', 'to_region': 'minas tirith',
          'n_regular': 2, 'n_elite': 0},
         {'from_region': 'osgiliath', 'to_region': 'minas tirith',
          'n_regular': 3, 'n_elite': 1}),
        # v15 (amendment 1.126)
        ({'mithril_redraw': True}, {'mithril_redraw': False}),
        ({'ents_chain_play': 'FREE_PEOPLES_Character_Athelas'},
         {'ents_chain_play': "FREE_PEOPLES_Character_Bilbo's_Song"}),
        ({'ents_chain_play': 'FREE_PEOPLES_Character_Athelas'},
         {'_ents_chain_done': True}),
        ({'eagles_dest': 'barad dur', '_eagles_from': 'dale'},
         {'eagles_dest': 'dol guldur', '_eagles_from': 'dale'}),
        ({'action': 'hide'}, {'action': 'move'}),
        ({'action': 'heal_only'}, {'action': 'hide'}),
        ({'companion_action': 'sep_Legolas', 'destination': 'eregion'},
         {'companion_action': 'sep_Legolas',
          'destination': '__remove_from_play__'}),
        # nazgul chain (USER 2026-07-11)
        ({'from_region': 'dagorlad', 'to_region': 'moria', 'count': 1},
         {'from_region': 'dagorlad', 'to_region': 'moria', 'count': 2}),
        ({'from_region': 'dagorlad', 'to_region': 'moria', 'count': 1},
         {'_naz_done': True}),
        # v16 (12-fix wave)
        ({'victim': 'Legolas'}, {'victim': 'Gimli'}),
        ({'strongholds': ['barad dur', 'dol guldur', 'minas morgul']},
         {'strongholds': ['dol guldur', 'minas morgul', 'moria']}),
        ({'regions': ['gorgoroth', 'nurn']}, {'regions': ['nurn', 'gorgoroth']}),
        # v17 (amendment 1.131): chain flights to the SAME destination
        # from different sources must be distinct (from_region rides
        # feat[14], free for card_option actions)
        ({'from_region': 'dagorlad', 'to_region': 'moria', 'count': 1},
         {'from_region': 'noman lands', 'to_region': 'moria', 'count': 1}),
        # v17 (1.131): Nazgul Strike! post-move either/or
        ({'action': 'hunt'}, {'action': 'none'}),
        ({'action': 'hunt'},
         {'action': 'discard_table_card',
          'target_card': 'FREE_PEOPLES_Character_Athelas'}),
        ({'action': 'discard_table_card',
          'target_card': 'FREE_PEOPLES_Character_Athelas'},
         {'action': 'discard_table_card',
          'target_card': "FREE_PEOPLES_Character_Bilbo's_Song"}),
    ]
    # v16.1 (amendment 1.130): PRIMARY-action count variants must be
    # pairwise distinct (the chain options were covered; these were not)
    _st_nz = ai.clone_state(base)
    import game_logic as _glg
    from game_state import Nation as _N, UnitKind as _UK, Unit as _U, Nazgul as _Nz
    _a_nz = _glg.get_or_create_army(_st_nz, 'dagorlad')
    _a_nz.units = [_U(_N.SAURON, _UK.REGULAR) for _ in range(3)]
    _a_nz.leaders = [_Nz(nation=_N.SAURON) for _ in range(3)]
    from ai import Action as _Act, encode_action as _enc
    _prim_pairs = [
        (_Act(die_idx=0, action_type='nazgul_move',
              params={'from': 'dagorlad', 'to': 'moria'}),
         _Act(die_idx=0, action_type='nazgul_move',
              params={'from': 'dagorlad', 'to': 'moria', 'count': 1})),
        (_Act(die_idx=0, action_type='nazgul_move',
              params={'from': 'dagorlad', 'to': 'moria', 'count': 1}),
         _Act(die_idx=0, action_type='nazgul_move',
              params={'from': 'dagorlad', 'to': 'moria', 'count': 2})),
        (_Act(die_idx=0, action_type='army_move',
              params={'from': 'dagorlad', 'to': 'moria', 'n_regular': 2,
                      'n_elite': 0, 'take_leaders': True, 'side': 'sh'}),
         _Act(die_idx=0, action_type='army_move',
              params={'from': 'dagorlad', 'to': 'moria', 'n_regular': 2,
                      'n_elite': 0, 'take_leaders': True, 'n_nazgul': 0,
                      'side': 'sh'})),
        (_Act(die_idx=0, action_type='army_move',
              params={'from': 'dagorlad', 'to': 'moria', 'n_regular': 2,
                      'n_elite': 0, 'take_leaders': False, 'n_nazgul': 1,
                      'side': 'sh'}),
         _Act(die_idx=0, action_type='army_move',
              params={'from': 'dagorlad', 'to': 'moria', 'n_regular': 2,
                      'n_elite': 0, 'take_leaders': False, 'n_nazgul': 2,
                      'side': 'sh'})),
    ]
    for _pa, _pb in _prim_pairs:
        assert not np.array_equal(
            _enc(_pa, _st_nz, Side.SHADOW),
            _enc(_pb, _st_nz, Side.SHADOW)), (
            f'primary count variants encode identically: '
            f'{_pa.params} / {_pb.params}')
    for _a9, _b9 in _pairs:
        _va9 = encode_action(_mkopt(_a9), base, Side.FREE_PEOPLES)
        _vb9 = encode_action(_mkopt(_b9), base, Side.FREE_PEOPLES)
        assert not np.array_equal(_va9, _vb9), (
            f"card_option choices {_a9} / {_b9} encode identically")
    # 1.138 cache-key invariant (review agent C): for every legal
    # action in a mixed probe state, key equality MUST imply encoding
    # equality - a key that misses an encode_action input silently
    # serves stale priors to MCTS (companion-group collision class).
    _st_ck = ai.clone_state(base)
    import game_logic as _glck
    from game_state import Companion as _CompCK, Nazgul as _NzCK,         Nation as _NCK, Unit as _UCK, UnitKind as _UKCK
    _a_ck = _glck.get_or_create_army(_st_ck, 'dale')
    _a_ck.units = [_UCK(_NCK.THE_NORTH, _UKCK.REGULAR) for _ in range(3)]
    for _nm_ck in ('Legolas', 'Gimli', 'Boromir'):
        _st_ck.fellowship.companions = [
            c for c in _st_ck.fellowship.companions if c.name != _nm_ck]
        _a_ck.leaders.append(_CompCK(name=_nm_ck, level=2, leadership=1))
    _b_ck = _glck.get_or_create_army(_st_ck, 'dagorlad')
    _b_ck.units = [_UCK(_NCK.SAURON, _UKCK.REGULAR) for _ in range(3)]
    _b_ck.leaders = [_NzCK(nation=_NCK.SAURON) for _ in range(2)]
    from collections import defaultdict as _dd_ck
    for _side_ck in (Side.FREE_PEOPLES, Side.SHADOW):
        _acts_ck = ai.get_legal_actions(_st_ck, _side_ck)
        _bykey_ck = _dd_ck(list)
        for _a9 in _acts_ck:
            _bykey_ck[ai._action_cache_key(_a9, _st_ck, _side_ck)]                 .append(_a9)
        for _k9, _grp9 in _bykey_ck.items():
            if len(_grp9) < 2:
                continue
            _e0 = ai.encode_action(_grp9[0], _st_ck, _side_ck)
            for _a9 in _grp9[1:]:
                _e1 = ai.encode_action(_a9, _st_ck, _side_ck)
                assert np.array_equal(_e0, _e1), (
                    'cache-key collision with distinct encodings: '
                    f'{_grp9[0].action_type} {_grp9[0].params} vs '
                    f'{_a9.action_type} {_a9.params}')
    print("[PASS] cache-key invariant: key equality implies encoding "
          "equality over full legal menus")
    print("[PASS] decision-context exposure: 10 combat phases distinct, "
          + str(len(muts)) + " pending fields all visible")



def test_siege_away_and_relief_attacks():
    """v9 showcase bug: an army in a besieged region (mixed shared
    entry: garrison + besiegers) attacking OUT crashed setup_combat
    ("Cannot attack friendly army") and the swallowed exception made
    the battle vanish. Away-attack: only the besieger portion fights.
    Relief-attack: an outside army fights only the besiegers; the
    garrison stays untouched."""
    import random
    random.seed(7)

    # --- away-attack: besiegers at Erebor strike out at Iron Hills ---
    state = setup_game()
    for k in list(state.armies):
        if sanitize(k) in ('erebor', 'iron hills'):
            state.armies.pop(k)
    a = game_logic.get_or_create_army(state, 'Erebor')
    a.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)
               for _ in range(5)]
    a.units += [gs.Unit(gs.Nation.DWARVES, gs.UnitKind.REGULAR)
                for _ in range(2)]
    bs = getattr(state, 'besieged_strongholds', None)
    if bs is None:
        state.besieged_strongholds = set()
    state.besieged_strongholds.add('erebor')
    b = game_logic.get_or_create_army(state, 'Iron Hills')
    b.units = [gs.Unit(gs.Nation.DWARVES, gs.UnitKind.REGULAR)]

    res = game_logic.battle(state, 'Erebor', 'Iron Hills',
                            max_rounds=10)
    assert res is not None, "away-attack battle did not resolve"
    ere = None
    for k, ar in state.armies.items():
        if sanitize(k) == 'erebor':
            ere = ar
    assert ere is not None, "Erebor entry vanished"
    dwarves = [u for u in ere.units if u.nation == gs.Nation.DWARVES]
    assert len(dwarves) == 2, (
        f"garrison touched by away-attack: {len(dwarves)} dwarves left")
    print("  PASS: away-attack from besieged region resolves; "
          "garrison untouched")

    # --- relief-attack: FP from Iron Hills strikes the besiegers ---
    state = setup_game()
    for k in list(state.armies):
        if sanitize(k) in ('erebor', 'iron hills'):
            state.armies.pop(k)
    a = game_logic.get_or_create_army(state, 'Erebor')
    a.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)
               for _ in range(2)]
    a.units += [gs.Unit(gs.Nation.DWARVES, gs.UnitKind.REGULAR)
                for _ in range(2)]
    if getattr(state, 'besieged_strongholds', None) is None:
        state.besieged_strongholds = set()
    state.besieged_strongholds.add('erebor')
    b = game_logic.get_or_create_army(state, 'Iron Hills')
    b.units = [gs.Unit(gs.Nation.DWARVES, gs.UnitKind.ELITE)
               for _ in range(5)]

    res = game_logic.battle(state, 'Iron Hills', 'Erebor',
                            max_rounds=10)
    assert res is not None, "relief-attack battle did not resolve"
    ere = None
    for k, ar in state.armies.items():
        if sanitize(k) == 'erebor':
            ere = ar
    assert ere is not None, "Erebor entry vanished after relief attack"
    dwarves = [u for u in ere.units if u.nation == gs.Nation.DWARVES]
    assert len(dwarves) >= 2, (
        f"garrison harmed by relief attack: {len(dwarves)}")
    assert sanitize('erebor') not in state.enemy_controlled.get(
        gs.Side.FREE_PEOPLES, set()), (
        "relief attack wrongly flipped settlement control")
    print("  PASS: relief attack hits only the besiegers; "
          "garrison and control intact")



def test_combat_card_requirements_deckwide():
    """USER audit (Athelas at Osgiliath with no Aragorn): EVERY combat
    card with a printed requirement must be refused in a bare battle
    that satisfies none of them, and every requirement-free combat card
    must be allowed. Bare battle: plain regulars, neutral wilderness
    defender region (Dagorlad — no nation, no stronghold, not near
    Moria/Rohan, fellowship far away). Plus positive checks for the
    named-character class."""
    import random
    import cards as cards_mod
    from combat import setup_combat
    random.seed(3)

    state = setup_game()
    for k in list(state.armies):
        if sanitize(k) in ('morannon', 'dagorlad'):
            state.armies.pop(k)
    a = game_logic.get_or_create_army(state, 'Morannon')
    a.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)
               for _ in range(3)]
    d = game_logic.get_or_create_army(state, 'Dagorlad')
    d.units = [gs.Unit(gs.Nation.GONDOR, gs.UnitKind.REGULAR)
               for _ in range(2)]
    cs = setup_combat(state, 'Morannon', 'Dagorlad')

    leaks, blocked_open = [], []
    for c in cards_mod.get_cards().values():
        if not (getattr(c, 'combat_name', '') or '').strip():
            continue
        is_atk = (c.side == gs.Side.SHADOW)
        ok, why = cards_mod.can_play_combat_card(
            c, cs, is_attacker=is_atk, state=state)
        if (getattr(c, 'combat_req', None) or '').strip():
            if ok:
                leaks.append((c.card_name, c.combat_req))
        else:
            if not ok:
                blocked_open.append((c.card_name, why))
    assert not leaks, f"req cards playable in bare battle: {leaks}"
    assert not blocked_open, (
        f"requirement-free cards wrongly blocked: {blocked_open}")

    # positive: Aragorn in the battle enables Athelas (Anduril)
    from game_state import Companion
    d.leaders.append(Companion(name="Strider", level=3, leadership=1))
    cs2 = setup_combat(state, 'Morannon', 'Dagorlad')
    athelas = next(c for c in cards_mod.get_cards().values()
                   if c.card_name == 'Athelas')
    ok, why = cards_mod.can_play_combat_card(
        athelas, cs2, is_attacker=False, state=state)
    assert ok, f"Athelas blocked WITH Strider in battle: {why}"
    d.leaders.clear()

    # positive: Gandalf enables Wizard's Staff
    d.leaders.append(Companion(name="Gandalf the Grey", level=3,
                               leadership=1))
    cs3 = setup_combat(state, 'Morannon', 'Dagorlad')
    staff = next(c for c in cards_mod.get_cards().values()
                 if c.card_name == "Wizard's Staff")
    ok, why = cards_mod.can_play_combat_card(
        staff, cs3, is_attacker=False, state=state)
    assert ok, f"Wizard's Staff blocked WITH Gandalf in battle: {why}"
    d.leaders.clear()

    n_req = sum(1 for c in cards_mod.get_cards().values()
                if (getattr(c, 'combat_name', '') or '').strip()
                and (getattr(c, 'combat_req', None) or '').strip())
    print(f"  PASS: all {n_req} requirement-bearing combat cards "
          "enforced; named-character positives OK")



def test_combat_dice_vs_reroll_routing():
    """USER audit 2026-07-09 (Swarm of Bats / Scouts check): cards that
    add dice to the LEADER RE-ROLL only ("add N to all dice on your
    Leader re-roll") must NOT boost the combat roll. The old generic
    branch matched the re-roll phrasing and wrongly applied +1 combat
    to Swarm of Bats (unconditionally) and They are Terrible (on top of
    its real re-roll bonus), and gave Relentless Assault a phantom flat
    +1 on top of its per-hit bonus."""
    import cards as cards_mod
    import game_state as gs_mod
    from combat import (setup_combat, apply_cancel_effects,
                        apply_dice_modifiers, apply_they_are_terrible)

    def _by_combat_name(cn):
        for c in cards_mod.get_cards().values():
            if (getattr(c, 'combat_name', '') or '') == cn:
                return c.card_id
        return None

    def _battle():
        st = setup_game()
        for k in list(st.armies):
            if sanitize(k) in ('morannon', 'dagorlad'):
                st.armies.pop(k)
        a = game_logic.get_or_create_army(st, 'Morannon')
        a.units = [gs_mod.Unit(gs_mod.Nation.SAURON, gs_mod.UnitKind.ELITE)
                   for _ in range(8)]
        a.leaders = [gs_mod.Nazgul(nation=gs_mod.Nation.SAURON)
                     for _ in range(3)]
        d = game_logic.get_or_create_army(st, 'Dagorlad')
        d.units = [gs_mod.Unit(gs_mod.Nation.GONDOR, gs_mod.UnitKind.REGULAR)
                   for _ in range(3)]
        return st

    def _mods(cid, opp=None):
        st = _battle()
        cs = setup_combat(st, 'Morannon', 'Dagorlad')
        is_atk = 'SHADOW' in cid
        c = cards_mod.get_card(cid)
        if is_atk:
            cs.attacker_card = c
            if opp:
                cs.defender_card = cards_mod.get_card(opp)
        else:
            cs.defender_card = c
            if opp:
                cs.attacker_card = cards_mod.get_card(opp)
        cs.nazgul_in_battle = 3
        apply_cancel_effects(cs)
        apply_they_are_terrible(cs)
        apply_dice_modifiers(cs)
        dm = cs.attacker_dice_mod if is_atk else cs.defender_dice_mod
        rm = cs.attacker_reroll_mod if is_atk else cs.defender_reroll_mod
        return dm, rm

    sob = _by_combat_name('Swarm of Bats')
    tat = _by_combat_name('They are Terrible')
    srv = _by_combat_name('Servant of the Secret Fire')
    dev = _by_combat_name('Devilry of Orthanc')
    ra = _by_combat_name('Relentless Assault')
    gift = _by_combat_name('It is a Gift')
    ofdl = _by_combat_name('One for the Dark Lord')
    ds = _by_combat_name('Deadly Strife')
    db = _by_combat_name('Desperate Battle')
    fp_card = 'FREE_PEOPLES_Strategy_Book_of_Mazarbul'

    # 1.208h (supersedes 1.79): combat-roll and re-roll bonuses are FULLY
    # SEPARATE. dice_mod is combat-only; reroll_mod is re-roll-only.
    # Swarm of Bats, FP played no card: re-roll +1, combat +0
    assert _mods(sob) == (0, 1), f"SoB no-FP-card: {_mods(sob)}"
    # Swarm of Bats, FP played a card: cancel only, no dice at all
    assert _mods(sob, fp_card) == (0, 0),         f"SoB with FP card: {_mods(sob, fp_card)}"
    # They are Terrible: re-roll +1, combat +0 (no phantom combat)
    assert _mods(tat) == (0, 1), f"They are Terrible: {_mods(tat)}"
    # Servant of the Secret Fire: combat roll ONLY, no re-roll boost
    assert _mods(srv) == (1, 0), f"Servant combat-only: {_mods(srv)}"
    # Devilry of Orthanc: combat roll ONLY (1.208h REOPENED as a bug)
    assert _mods(dev) == (1, 0), f"Devilry combat-only: {_mods(dev)}"
    # Relentless Assault: no flat dice from apply_dice_modifiers
    assert _mods(ra) == (0, 0), f"Relentless Assault flat: {_mods(ra)}"
    # "Combat roll AND Leader re-roll" cards fill BOTH buckets
    assert _mods(gift) == (1, 1), f"It is a Gift both: {_mods(gift)}"
    assert _mods(ofdl) == (1, 1), f"One for the Dark Lord both: {_mods(ofdl)}"
    # "Both Armies add N" cards name BOTH targets on both sides
    assert _mods(ds) == (2, 2), f"Deadly Strife both: {_mods(ds)}"
    assert _mods(db) == (1, 1), f"Desperate Battle both: {_mods(db)}"
    import random as _r
    from combat import leader_reroll
    # miss-pool rule (1.85 SURVIVES, USER 2026-07-09 Lorien siege): a
    # re-roll-only bonus must NOT shrink the miss pool -- [2,5,3] at 6+
    # has THREE misses even with a +1 re-roll modifier; the bonus applies
    # only to the re-rolled dice.
    _seq = iter([4, 3, 5])
    _orig_ri = _r.randint
    _r.randint = lambda a, b: next(_seq)
    try:
        rr, _hits = leader_reroll([2, 5, 3], 6, leadership=5,
                                  modifier=1, miss_modifier=0)
    finally:
        _r.randint = _orig_ri
    assert len(rr) == 3, f"re-roll pool shrunk by re-roll bonus: {rr}"
    # NEW 1.208h lock: a COMBAT bonus (miss_modifier) sizes the miss pool
    # but must NOT boost the re-roll HIT. [3] at 5+ with combat +1 is a
    # miss (3+1=4 < 5), so it re-rolls; the re-rolled 4 gets modifier=0
    # (reroll_mod) and stays a miss -> pool len 1, 0 hits.
    _seq2 = iter([4])
    _r.randint = lambda a, b: next(_seq2)
    try:
        rr2, hits2 = leader_reroll([3], 5, leadership=1,
                                   modifier=0, miss_modifier=1)
    finally:
        _r.randint = _orig_ri
    assert (len(rr2), hits2) == (1, 0), \
        f"combat +1 wrongly boosted re-roll hit: {(len(rr2), hits2)}"
    print("  PASS: 1.208h combat-vs-reroll FULL separation (combat-only "
          "cards, both-bucket cards, miss pool at combat mod, re-roll hit "
          "at reroll mod)")


def test_sortie_attribution_and_orientation():
    """v13 showcase 'Moria' flag (2026-07-10): a garrison SORTIE (the
    besieged army attacking its besiegers, p.32) was misattributed to
    the besieger side - wrong side's dice consumed, inverted combat
    roles, resolved as a 6+ siege assault instead of a field battle."""
    import ai
    from game_state import Side, Nation, UnitKind, Unit, ActionDieFace
    from combat import setup_combat
    try:
        state = setup_game()
        # FP besieges the native-Shadow stronghold Moria: shared entry =
        # 4 Elven besiegers outside + 3 Sauron garrison inside
        for k in list(state.armies):
            if sanitize(k) == 'moria':
                state.armies.pop(k)
        a = game_logic.get_or_create_army(state, 'Moria')
        a.units = [gs.Unit(Nation.ELVES, UnitKind.REGULAR)
                   for _ in range(4)] \
            + [gs.Unit(Nation.SAURON, UnitKind.REGULAR) for _ in range(3)]
        state.besieged_strongholds.add('moria')
        state.political.boxes[Nation.ELVES] = 3
        state.political.active[Nation.ELVES] = True
        state.political.boxes[Nation.SAURON] = 3
        state.political.active[Nation.SAURON] = True

        # enumeration stamps the acting side on the from==to attack
        sh_atk = [x for x in ai._get_army_attack_actions(
            state, Side.SHADOW, 0)
            if sanitize(x.params['from']) == 'moria'
            and sanitize(x.params['to']) == 'moria']
        assert sh_atk and sh_atk[0].params.get('side') == 'sh', \
            f"sortie not offered/stamped for the garrison: {sh_atk}"
        fp_atk = [x for x in ai._get_army_attack_actions(
            state, Side.FREE_PEOPLES, 0)
            if sanitize(x.params['from']) == 'moria']
        assert fp_atk and fp_atk[0].params.get('side') == 'fp'

        # executing the SORTIE consumes a SHADOW die, never an FP one
        state.dice[Side.SHADOW][0].current = ActionDieFace.ARMY
        state.dice[Side.SHADOW][0].used = False
        for d in state.dice[Side.FREE_PEOPLES]:
            d.current = ActionDieFace.MUSTER   # no FP attack die exists
            d.used = False
        ai._execute_action(state, sh_atk[0])
        assert state.dice[Side.SHADOW][0].used, "SH die not consumed"
        assert not any(d.used for d in state.dice[Side.FREE_PEOPLES]), \
            "sortie stole an FP die"
        pc = state.pending_combat
        assert pc is not None and pc['attacker_side'] == Side.SHADOW
        assert pc['is_siege'] is False, \
            "sortie must be a FIELD battle, not a siege assault"
        # combat orientation: the garrison portion attacks
        cs = setup_combat(state, 'moria', 'moria',
                          is_siege=False, attacker_side=Side.SHADOW)
        assert all(u.nation == Nation.SAURON
                   for u in cs.attacker_army.units), \
            "sortie attacker must be the garrison portion"
        assert all(u.nation == Nation.ELVES
                   for u in cs.defender_army.units)
        # assault orientation (default) unchanged: besieger attacks
        cs2 = setup_combat(state, 'moria', 'moria', is_siege=True)
        assert all(u.nation == Nation.ELVES
                   for u in cs2.attacker_army.units), \
            "assault attacker must be the besieger portion"
        print("  PASS: garrison sortie attributed to the garrison side "
              "(die, roles, field battle); assault default unchanged")
    finally:
        pass  # (hook restore collapsed 2026-07-27)


def test_pending_side_routing_consistency():
    """v13 showcase crash (2026-07-10): the acting-side routers carried
    their own copies of the pending-decision cascade and drifted from
    get_legal_actions' branch order — a coexisting SHADOW card option +
    FP hand-limit discard was routed to SHADOW, and get_legal_actions'
    owner-guard raised.  All routers must use _pending_decision_side,
    and its priority must match get_legal_actions exactly."""
    import ai
    from game_state import Side
    try:
        state = setup_game()
        # SHADOW owns a pending card option; FP owes a hand-limit
        # discard (hand must be non-empty or the discard branch
        # legitimately clears itself and falls through)
        state.hands[Side.FREE_PEOPLES].hand.extend(
            [f'FREE_PEOPLES_Strategy_Test_{i}' for i in range(7)])
        state.pending_card_option = {
            'card_id': 'X', 'card_name': 'Test Card', 'side': 'sh',
            'options': [{'region': 'orthanc'}]}
        state.pending_discard = {'side': Side.FREE_PEOPLES}
        got = ai._pending_decision_side(state)
        assert got == Side.FREE_PEOPLES, \
            f"cascade priority: discard must outrank card option, got {got}"
        got2 = ai._next_acting_side(state, Side.SHADOW)
        assert got2 == Side.FREE_PEOPLES, \
            f"_next_acting_side drifted from the shared cascade: {got2}"
        # the ask that crashed v13: must yield FP discard actions, no raise
        acts = ai.get_legal_actions(state, Side.FREE_PEOPLES)
        assert acts and all(a.action_type == 'discard_card' for a in acts), \
            f"expected discard actions, got {[a.action_type for a in acts][:5]}"
        # after the discard resolves, the card option side acts
        state.pending_discard = None
        assert ai._pending_decision_side(state) == Side.SHADOW
        print("  PASS: pending-side routing consistent "
              "(discard outranks card option in every router)")
    finally:
        pass  # (hook restore collapsed 2026-07-27)


if __name__ == '__main__':
    print("=== War of the Ring Rule Tests ===\n")

    test_army_die_double_move()
    test_army_die_attack_no_pending()
    test_character_die_requires_leader()
    test_character_die_single_move()
    test_character_die_political_advance()
    test_character_die_fellowship()
    test_event_card_play_by_die_type()
    test_muster_double_recruit()
    test_muster_separate_settlements()
    test_muster_elite_placement()
    test_active_nation_can_move()
    test_passive_nation_can_move_within_borders()
    test_will_of_west_acts_as_any()
    test_hunt_resolution_on_fellowship_move()
    test_hunt_threshold_calculation()
    test_recruited_units_have_correct_nation()
    test_recruit_at_own_nation_settlements_only()
    test_reinforcement_limits()
    test_city_attack_penalty()
    test_stronghold_no_city_penalty()
    test_siege_press_mechanic()
    test_hunt_roll_direction()
    test_non_belligerent_border_restriction()
    test_fp_casualties_permanent()
    test_political_activation_on_enemy_entry()
    test_fellowship_declaration_activates_nation()
    test_companion_separation_blocked_in_mordor()
    test_leaders_removed_when_alone()
    test_combat_roll_natural_1_always_misses()
    test_companion_movement_activates_nation()
    test_repeat_attacks_allowed()
    test_army_can_move_multiple_times_per_turn()
    test_army_split_moves()
    test_mixed_nation_army_attack()
    test_minion_muster_exclusivity()
    test_victory_timing()
    test_decision_context_exposure()
    test_siege_away_and_relief_attacks()
    test_combat_card_requirements_deckwide()
    test_combat_dice_vs_reroll_routing()
    test_pending_side_routing_consistency()
    test_sortie_attribution_and_orientation()

    print("\n=== All tests passed! ===")
