#!/usr/bin/env python3
# main.py - Entry point for War of the Ring
"""War of the Ring board game implementation with full combat, cards, and GUI."""

import argparse

# Set matplotlib backend before any imports to prevent Tkinter conflicts on macOS
import matplotlib
matplotlib.use('Agg')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="War of the Ring board game implementation"
    )
    parser.add_argument(
        "--gui", "-g", action="store_true",
        help="Launch interactive GUI"
    )
    parser.add_argument(
        "--test", "-t", action="store_true",
        help="Run basic tests"
    )
    parser.add_argument(
        "--combat-demo", "-c", action="store_true",
        help="Run combat system demo"
    )
    parser.add_argument(
        "--ai", choices=["shadow", "fp"], default=None,
        help="Enable AI for specified side (use with --gui)"
    )
    parser.add_argument(
        "--watch", "-w", action="store_true",
        help="Watch AI vs AI self-play game in GUI"
    )
    parser.add_argument(
        "--replay-trace", metavar="PATH", default=None,
        help="Step through a recorded self-play trace "
             "(e.g. selfplay_traces/ep00042_sh_military.trace.gz)"
    )

    args = parser.parse_args()

    # Load board and state
    print("Loading War of the Ring...")
    import board
    from game_state import setup_initial_game

    b = board.load_board()
    state = setup_initial_game(b)
    print(f"Board loaded with {len(b._regions)} regions")

    # Load cards
    from cards import get_cards
    cards = get_cards()
    print(f"Loaded {len(cards)} cards")

    if args.test:
        run_tests(state, b)
    elif args.combat_demo:
        run_combat_demo(state)
    elif args.replay_trace:
        from gui import launch_gui_replay
        launch_gui_replay(args.replay_trace, b)
    elif args.watch:
        from gui import launch_gui_selfplay
        launch_gui_selfplay(state, b)
    elif args.gui:
        from gui import launch_gui
        launch_gui(state, b, ai_side=args.ai)
    else:
        # Default: print game state summary
        print_state_summary(state)


def print_state_summary(state):
    """Print a summary of the game state."""
    print("\n" + "=" * 60)
    print("WAR OF THE RING - Game State Summary")
    print("=" * 60)

    print(f"\nTurn: {state.turn}")
    print(f"VP: Free Peoples {state.vp.free_peoples}, Shadow {state.vp.shadow}")

    from game_logic import fellowship_region
    fsp_region = fellowship_region(state)
    hidden = "Hidden" if state.fellowship.hidden else "Revealed"
    print(f"\nFellowship: {fsp_region} ({hidden})")
    print(f"Corruption: {state.fellowship.corruption}/12")
    print(f"Progress: {state.fellowship.progress}")
    print(f"Companions: {len(state.fellowship.companions)}")
    for c in state.fellowship.companions:
        guide_marker = " (Guide)" if c == state.fellowship.guide else ""
        print(f"  - {c.name} (Level {c.level}){guide_marker}")

    print("\nPolitical Track:")
    from game_state import Nation
    for nation in Nation:
        pos = state.political.boxes[nation]
        active = "Active" if state.political.active[nation] else "Passive"
        at_war = " [AT WAR]" if pos >= 3 else ""
        print(f"  {nation.value}: {pos} ({active}){at_war}")

    print(f"\nArmies: {len(state.armies)} regions occupied")

    # Count by side
    from game_state import Side
    fp_armies = 0
    sh_armies = 0
    fp_units = 0
    sh_units = 0

    shadow_nations = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}

    for army in state.armies.values():
        if army.units:
            if army.units[0].nation in shadow_nations:
                sh_armies += 1
                sh_units += len(army.units)
            else:
                fp_armies += 1
                fp_units += len(army.units)

    print(f"  Free Peoples: {fp_armies} armies, {fp_units} units")
    print(f"  Shadow: {sh_armies} armies, {sh_units} units")

    print("\nAction Dice:")
    print(f"  Free Peoples: {len(state.dice[Side.FREE_PEOPLES])} dice")
    print(f"  Shadow: {len(state.dice[Side.SHADOW])} dice")
    print(f"  Hunt Box: {len(state.hunt_box)} dice")

    print(f"\nHunt Pool: {len(state.hunt_pool)} tiles")
    print(f"Elven Rings: FP {state.fp_elven_rings}, Shadow {state.sh_elven_rings}")

    print("\n" + "=" * 60)


def run_tests(state, b):
    """Run basic tests."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    # Test 1: Combat system
    print("\nTest 1: Combat System")
    from combat import calc_combat_strength, calc_leadership, combat_roll

    # Get an army to test
    osgiliath_army = state.armies.get("Osgiliath")
    if osgiliath_army:
        strength = calc_combat_strength(osgiliath_army)
        leadership = calc_leadership(osgiliath_army)
        print(f"  Osgiliath army: strength={strength}, leadership={leadership}")

        rolls, hits = combat_roll(strength, 5, 0)
        print(f"  Combat roll: {rolls} -> {hits} hits")

    # Test 2: Card loading
    print("\nTest 2: Card Loading")
    from cards import get_cards, get_card, categorize_effect

    cards = get_cards()
    print(f"  Total cards: {len(cards)}")

    # Check some specific cards
    test_cards = [
        "FREE_PEOPLES_Character_Elven_Cloaks",
        "SHADOW_Strategy_Deadly_Strife",
    ]
    for card_id in test_cards:
        card = get_card(card_id)
        if card:
            effects = categorize_effect(card.combat_effect)
            print(f"  {card.card_name}: combat_value={card.combat_value}, effects={effects}")

    # Test 3: Combat with cards
    print("\nTest 3: Combat Resolution")
    from combat import resolve_combat

    # Setup a test battle
    test_from = "Minas Morgul"
    test_to = "Osgiliath"

    if test_from in state.armies and test_to in state.armies:
        try:
            result = resolve_combat(state, test_from, test_to)
            print(f"  Battle: {test_from} -> {test_to}")
            print(f"  Attacker hits: {result.attacker_total_hits}")
            print(f"  Defender hits: {result.defender_total_hits}")
            print(f"  Attacker casualties: {len(result.attacker_casualties)}")
            print(f"  Defender casualties: {len(result.defender_casualties)}")
        except Exception as e:
            print(f"  Combat test error: {e}")

    # Test 4: Event effects
    print("\nTest 4: Event Requirements")
    from event_effects import check_event_requirement

    test_events = [
        "FREE_PEOPLES_Character_Athelas",
        "SHADOW_Character_Nazgûl_Search",  # Note: underscore and special char
    ]
    for card_id in test_events:
        can_play, msg = check_event_requirement(state, card_id)
        print(f"  {card_id}: {can_play} ({msg})")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


def run_combat_demo(state):
    """Run a combat demonstration."""
    print("\n" + "=" * 60)
    print("COMBAT SYSTEM DEMONSTRATION")
    print("=" * 60)

    from combat import (
        resolve_combat, calc_combat_strength, calc_leadership
    )

    # Find two adjacent opposing armies for a realistic battle
    print("\nFinding adjacent opposing armies...")

    from game_state import Nation
    shadow_nations = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}

    # Use Minas Morgul (Shadow) attacking Osgiliath (FP) - classic matchup
    sh_army_region = "Minas Morgul"
    fp_army_region = "Osgiliath"

    sh_army = state.armies.get(sh_army_region)
    fp_army = state.armies.get(fp_army_region)

    if not sh_army or not fp_army:
        print(f"Could not find armies in {sh_army_region} or {fp_army_region}")
        # Fallback to finding any opposing armies
        for region, army in state.armies.items():
            if not army.units:
                continue
            if army.units[0].nation in shadow_nations:
                if sh_army is None:
                    sh_army_region = region
                    sh_army = army
            else:
                if fp_army is None:
                    fp_army_region = region
                    fp_army = army
            if sh_army and fp_army:
                break

    if not sh_army or not fp_army:
        print("Could not find opposing armies for demo")
        return

    print(f"\nAttacker: {sh_army_region}")
    print(f"  Units: {len(sh_army.units)}")
    print(f"  Combat Strength: {calc_combat_strength(sh_army)}")
    print(f"  Leadership: {calc_leadership(sh_army)}")

    print(f"\nDefender: {fp_army_region}")
    print(f"  Units: {len(fp_army.units)}")
    print(f"  Combat Strength: {calc_combat_strength(fp_army)}")
    print(f"  Leadership: {calc_leadership(fp_army)}")

    # Run battle without cards
    print("\n--- Battle without cards ---")
    result = resolve_combat(state, sh_army_region, fp_army_region)
    print(result)

    # Reload state for card demo
    from game_state import setup_initial_game
    import board
    b = board.load_board()
    state = setup_initial_game(b)

    # Run battle with cards
    print("\n--- Battle with Deadly Strife (Shadow) ---")
    result = resolve_combat(
        state, sh_army_region, fp_army_region,
        attacker_card_id="SHADOW_Strategy_Deadly_Strife"
    )
    print(result)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
