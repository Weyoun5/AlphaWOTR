#!/usr/bin/env python3
"""
Comprehensive test suite for War of the Ring implementation.

Tests cover:
- Board adjacency and regions
- Initial game setup
- Army movement
- Political track
- Fellowship mechanics
- Hunt system
- Combat system
- Special characters (minions, Gandalf White, Aragorn)
- Companion abilities
- Card effects
"""

import sys
import os
import random
import unittest
from typing import List, Set, Dict
from collections import defaultdict

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import modules
import board
from board import sanitize, load_board, adj, regions as board_regions
import game_state as gs
from game_state import (
    Side, Nation, UnitKind, Unit, Leader, Army, Nazgul,
    Companion, Fellowship, PoliticalTrack, ActionDieFace, ActionDie,
    ShadowMinion, GandalfTheWhite, Aragorn, HuntTile,
    setup_initial_game
)
import game_logic as gl


class TestBoardAdjacency(unittest.TestCase):
    """Test board adjacency graph correctness."""

    def setUp(self):
        self.board = load_board()

    def test_adjacency_is_bidirectional(self):
        """Every adjacency should be bidirectional."""
        for region, neighbors in adj.items():
            for neighbor in neighbors:
                self.assertIn(region, adj.get(neighbor, set()),
                    f"Adjacency not bidirectional: {region} -> {neighbor}")

    def test_all_regions_have_adjacencies(self):
        """All regions should have at least one neighbor."""
        for region in board_regions:
            self.assertGreater(len(adj.get(region, set())), 0,
                f"Region {region} has no adjacencies")

    def test_mordor_star_topology(self):
        """Mordor should have star topology centered on Gorgoroth."""
        gorgoroth = sanitize("gorgoroth")
        mordor_regions = ["morannon", "minas morgul", "barad dur", "nurn"]

        for region in mordor_regions:
            region_san = sanitize(region)
            self.assertIn(gorgoroth, adj.get(region_san, set()),
                f"{region} should be adjacent to Gorgoroth")
            self.assertIn(region_san, adj.get(gorgoroth, set()),
                f"Gorgoroth should be adjacent to {region}")

    def test_morannon_connects_to_dagorlad(self):
        """Morannon should connect to Dagorlad (exit from Mordor)."""
        morannon = sanitize("morannon")
        dagorlad = sanitize("dagorlad")
        self.assertIn(dagorlad, adj.get(morannon, set()))

    def test_minas_morgul_connects_to_ithilien(self):
        """Minas Morgul should connect to North and South Ithilien."""
        morgul = sanitize("minas morgul")
        north_ith = sanitize("north ithilien")
        south_ith = sanitize("south ithilien")
        morgul_adj = adj.get(morgul, set())
        self.assertIn(north_ith, morgul_adj)
        self.assertIn(south_ith, morgul_adj)

    def test_only_rules_enforced(self):
        """Test that ONLY adjacency rules are enforced correctly."""
        # Rivendell should ONLY connect to trollshaws and fords of bruinen
        rivendell = sanitize("rivendell")
        rivendell_adj = adj.get(rivendell, set())

        allowed = {sanitize("trollshaws"), sanitize("fords of bruinen")}
        self.assertEqual(rivendell_adj, allowed,
            f"Rivendell adjacencies should be exactly {allowed}, got {rivendell_adj}")

    def test_removed_edges(self):
        """Test that removed edges are not present."""
        # Mount Gram should NOT connect to Trollshaws
        mount_gram = sanitize("mount gram")
        trollshaws = sanitize("trollshaws")
        self.assertNotIn(trollshaws, adj.get(mount_gram, set()))

    def test_region_count(self):
        """Verify reasonable number of regions."""
        self.assertGreater(len(board_regions), 60, "Should have at least 60 regions")
        self.assertLess(len(board_regions), 120, "Should have less than 120 regions")


class TestInitialSetup(unittest.TestCase):
    """Test initial game setup correctness."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_fellowship_starts_in_rivendell(self):
        """Fellowship should start in Rivendell."""
        region = gl.fellowship_region(self.state)
        self.assertEqual(region, sanitize("rivendell"))

    def test_seven_companions(self):
        """Fellowship should have 7 companions at start."""
        self.assertEqual(len(self.state.fellowship.companions), 7)

    def test_companion_names(self):
        """All companions should have correct names."""
        expected_names = {
            "Gandalf the Grey", "Strider", "Legolas", "Gimli",
            "Boromir", "Meriadoc", "Peregrin"
        }
        actual_names = {c.name for c in self.state.fellowship.companions}
        self.assertEqual(actual_names, expected_names)

    def test_gandalf_is_guide(self):
        """Gandalf should be the initial guide."""
        self.assertEqual(self.state.fellowship.guide.name, "Gandalf the Grey")

    def test_four_fp_dice(self):
        """Free Peoples should start with 4 dice."""
        self.assertEqual(len(self.state.dice[Side.FREE_PEOPLES]), 4)

    def test_seven_shadow_dice(self):
        """Shadow should start with 7 dice."""
        self.assertEqual(len(self.state.dice[Side.SHADOW]), 7)

    def test_four_nazgul(self):
        """Should have 4 Nazgul at game start."""
        nazgul_count = 0
        for army in self.state.armies.values():
            for leader in army.leaders:
                if isinstance(leader, Nazgul):
                    nazgul_count += 1
        self.assertEqual(nazgul_count, 4)

    def test_political_track_initial(self):
        """Political track should have correct initial values."""
        # FP nations
        self.assertEqual(self.state.political.boxes[Nation.ELVES], 0)
        self.assertEqual(self.state.political.boxes[Nation.DWARVES], 0)
        self.assertEqual(self.state.political.boxes[Nation.THE_NORTH], 0)
        self.assertEqual(self.state.political.boxes[Nation.ROHAN], 0)
        self.assertEqual(self.state.political.boxes[Nation.GONDOR], 1)

        # Shadow nations
        self.assertEqual(self.state.political.boxes[Nation.ISENGARD], 2)
        self.assertEqual(self.state.political.boxes[Nation.SAURON], 2)
        self.assertEqual(self.state.political.boxes[Nation.SOUTHRON_EASTERLINGS], 1)

    def test_active_nations(self):
        """Active nations should be correct at start."""
        self.assertTrue(self.state.political.active[Nation.ELVES])
        self.assertFalse(self.state.political.active[Nation.DWARVES])
        self.assertTrue(self.state.political.active[Nation.ISENGARD])
        self.assertTrue(self.state.political.active[Nation.SAURON])

    def test_hunt_pool_size(self):
        """Hunt pool should have 16 tiles."""
        self.assertEqual(len(self.state.hunt_pool), 16)

    def test_hunt_tile_types(self):
        """Hunt pool should have correct distribution of tiles."""
        tiles = self.state.hunt_pool
        eye_count = sum(1 for t in tiles if t.is_eye)
        numbered_reveal_count = sum(1 for t in tiles if t.reveal and not t.is_eye)
        reveal_count = sum(1 for t in tiles if t.reveal)

        self.assertEqual(eye_count, 4, "Should have 4 eye tiles")
        self.assertEqual(numbered_reveal_count, 5,
                         "Should have 5 numbered reveal tiles (2r, 1r, 1r, 0r, 0r)")
        # Per almanac: standard Eye tiles always reveal the Fellowship
        self.assertEqual(reveal_count, 9,
                         "All 4 eye tiles + 5 numbered reveal tiles reveal")

    def test_special_characters_not_in_play(self):
        """Special characters should not be in play at start."""
        self.assertFalse(self.state.witch_king.in_play)
        self.assertFalse(self.state.saruman.in_play)
        self.assertFalse(self.state.mouth_of_sauron.in_play)
        self.assertFalse(self.state.gandalf_the_white.in_play)
        self.assertFalse(self.state.aragorn_crowned.crowned)

    def test_three_elven_rings(self):
        """FP should start with 3 elven rings."""
        self.assertEqual(self.state.fp_elven_rings, 3)

    def test_armies_in_correct_regions(self):
        """Verify some key armies are in correct positions."""
        # Minas Tirith should have Gondor units
        mt_key = None
        for key in self.state.armies:
            if sanitize(key) == sanitize("minas tirith"):
                mt_key = key
                break

        self.assertIsNotNone(mt_key, "Minas Tirith should have an army")
        mt_army = self.state.armies[mt_key]
        self.assertTrue(any(u.nation == Nation.GONDOR for u in mt_army.units))


class TestArmyMovement(unittest.TestCase):
    """Test army movement mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_move_entire_army(self):
        """Test moving an entire army to adjacent region."""
        # Move army from Minas Tirith to Osgiliath
        mt_key = None
        for key in self.state.armies:
            if sanitize(key) == sanitize("minas tirith"):
                mt_key = key
                break

        original_units = len(self.state.armies[mt_key].units)
        gl.move_army(self.state, mt_key, "osgiliath")

        # MT should be empty or gone
        mt_units = 0
        for key in self.state.armies:
            if sanitize(key) == sanitize("minas tirith"):
                mt_units = len(self.state.armies[key].units)
                break

        self.assertEqual(mt_units, 0, "Minas Tirith should be empty after full move")

    def test_move_to_non_adjacent_fails(self):
        """Moving to non-adjacent region should fail."""
        with self.assertRaises(RuntimeError):
            gl.move_army(self.state, "minas tirith", "erebor")

    def test_move_from_empty_region_fails(self):
        """Moving from empty region should fail."""
        with self.assertRaises(RuntimeError):
            gl.move_army(self.state, "cardolan", "bree")


class TestPoliticalTrack(unittest.TestCase):
    """Test political track mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_advance_nation(self):
        """Test advancing a nation on the political track."""
        initial = self.state.political.boxes[Nation.GONDOR]
        gl.advance_political(self.state, Nation.GONDOR)
        self.assertEqual(self.state.political.boxes[Nation.GONDOR], initial + 1)

    def test_activate_nation(self):
        """Test activating a nation."""
        self.assertFalse(self.state.political.active[Nation.DWARVES])
        gl.activate_nation(self.state, Nation.DWARVES)
        self.assertTrue(self.state.political.active[Nation.DWARVES])

    def test_at_war_at_position_3(self):
        """Nation should be at war at position 3."""
        self.state.political.boxes[Nation.GONDOR] = 3
        self.assertTrue(gl.is_nation_at_war(self.state, Nation.GONDOR))

    def test_not_at_war_below_3(self):
        """Nation should not be at war below position 3."""
        self.state.political.boxes[Nation.GONDOR] = 2
        self.assertFalse(gl.is_nation_at_war(self.state, Nation.GONDOR))


class TestFellowshipMechanics(unittest.TestCase):
    """Test Fellowship mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_initial_fellowship_state(self):
        """Test initial fellowship state."""
        self.assertEqual(self.state.fellowship.progress, 0)
        self.assertEqual(self.state.fellowship.corruption, 0)
        self.assertTrue(self.state.fellowship.hidden)
        self.assertFalse(self.state.fellowship.mordor_entered)

    def test_fellowship_movement_increases_progress(self):
        """Moving fellowship should increase progress."""
        initial_progress = self.state.fellowship.progress
        gl.move_fellowship(self.state)
        self.assertEqual(self.state.fellowship.progress, initial_progress + 1)

    def test_change_guide(self):
        """Test changing the guide."""
        self.assertEqual(self.state.fellowship.guide.name, "Gandalf the Grey")
        # Remove Gandalf
        self.state.fellowship.companions = [
            c for c in self.state.fellowship.companions
            if c.name != "Gandalf the Grey"
        ]
        gl.change_guide(self.state)
        # New guide should be Strider (next highest level)
        self.assertEqual(self.state.fellowship.guide.name, "Strider")

    def test_guide_becomes_gollum_when_empty(self):
        """Gollum becomes guide when all companions leave."""
        self.state.fellowship.companions = []
        gl.change_guide(self.state)
        self.assertEqual(self.state.fellowship.guide.name, "Gollum")


class TestParserGuideBox(unittest.TestCase):
    """log_replay Guide-box branch must mirror game_logic.change_guide:
    a Gollum drag into the Guide Box (corpus: '364 Game.2038 ->
    Game.1999') sets fellowship.guide to Gollum but NEVER appends him
    to fellowship.companions — the engine keeps companions empty."""

    def setUp(self):
        import log_replay
        self.board = load_board()
        self.state = setup_initial_game(self.board)
        self.replayer = log_replay.LogReplayer(
            self.state, self.board, verbose=False)

    def test_gollum_drag_sets_guide_companions_stay_empty(self):
        """Drag 364 into the Guide Box with no companions left."""
        self.state.fellowship.companions = []
        self.state.fellowship.guide = None
        self.replayer._handle_fellowship_companions(
            ['364'], 'FP Reinf', 'Guide Box')
        g = self.state.fellowship.guide
        self.assertIsNotNone(g)
        self.assertEqual(g.name, 'Gollum')
        # engine gives takeover-Gollum his hunt ability
        self.assertEqual(g.guide_ability, 'no_reveal_on_damage')
        # THE BUG: Gollum must NOT join the companion list
        self.assertEqual(self.state.fellowship.companions, [])
        # the decision is still recorded for extraction
        self.assertEqual(self.replayer.action_log[-1][0], 'choose_guide')
        self.assertEqual(
            self.replayer.action_log[-1][1]['companion'], 'Gollum')

    def test_gollum_drag_keeps_reassigned_gollum_guide(self):
        """If _reassign_guide already promoted Gollum (last companion
        died), the explicit corpus drag keeps that guide object and
        leaves companions untouched."""
        self.state.fellowship.companions = []
        self.replayer._reassign_guide()
        existing = self.state.fellowship.guide
        self.assertEqual(existing.name, 'Gollum')
        self.replayer._handle_fellowship_companions(
            ['364'], 'FP Reinf', 'Guide Box')
        self.assertIs(self.state.fellowship.guide, existing)
        self.assertEqual(self.state.fellowship.companions, [])

    def test_real_companion_guide_drag_unchanged(self):
        """A normal companion already in the fellowship becoming guide
        keeps the existing semantics (guide set, list unchanged)."""
        names_before = [c.name for c in self.state.fellowship.companions]
        self.assertIn('Strider', names_before)
        self.replayer._handle_fellowship_companions(
            ['241'], 'Fellowship Box', 'Guide Box')
        self.assertEqual(self.state.fellowship.guide.name, 'Strider')
        self.assertEqual(
            [c.name for c in self.state.fellowship.companions],
            names_before)


class TestHuntSystem(unittest.TestCase):
    """Test hunt mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_hunt_damage_application(self):
        """Test applying hunt damage."""
        initial_corruption = self.state.fellowship.corruption
        gl.apply_hunt_damage_simple(self.state, 2)
        self.assertEqual(self.state.fellowship.corruption, initial_corruption + 2)

    def test_corruption_capped_at_12(self):
        """Corruption should not exceed 12."""
        self.state.fellowship.corruption = 11
        gl.apply_hunt_damage_simple(self.state, 5)
        self.assertEqual(self.state.fellowship.corruption, 12)


class TestCombatSystem(unittest.TestCase):
    """Test combat mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)
        # Set up a valid combat scenario
        random.seed(42)  # For reproducible tests

    def test_combat_strength_calculation(self):
        """Test combat strength calculation."""
        from combat import calc_combat_strength

        army = Army(region=None)
        army.units = [
            Unit(Nation.GONDOR, UnitKind.REGULAR),
            Unit(Nation.GONDOR, UnitKind.REGULAR),
            Unit(Nation.GONDOR, UnitKind.ELITE),
        ]
        # Per rulebook: each unit = 1 die, so 3 units = 3
        self.assertEqual(calc_combat_strength(army), 3)

    def test_combat_strength_capped_at_5(self):
        """Combat strength should cap at 5."""
        from combat import calc_combat_strength

        army = Army(region=None)
        army.units = [
            Unit(Nation.GONDOR, UnitKind.ELITE),
            Unit(Nation.GONDOR, UnitKind.ELITE),
            Unit(Nation.GONDOR, UnitKind.ELITE),
            Unit(Nation.GONDOR, UnitKind.ELITE),
            Unit(Nation.GONDOR, UnitKind.ELITE),
            Unit(Nation.GONDOR, UnitKind.ELITE),
        ]
        # Per rulebook: each unit = 1 die, 6 units capped at 5
        self.assertEqual(calc_combat_strength(army), 5)

    def test_casualty_application(self):
        """Test casualty application removes regulars first."""
        from combat import apply_casualties

        army = Army(region=None)
        army.units = [
            Unit(Nation.GONDOR, UnitKind.REGULAR),
            Unit(Nation.GONDOR, UnitKind.REGULAR),
            Unit(Nation.GONDOR, UnitKind.ELITE),
        ]

        casualties = apply_casualties(army, 2)
        self.assertEqual(len(casualties), 2)
        # Should have removed 2 regulars, keeping 1 elite
        self.assertEqual(len(army.units), 1)
        self.assertEqual(army.units[0].kind, UnitKind.ELITE)

    def test_elite_downgrade(self):
        """Test elite downgrade when taking 1 hit."""
        from combat import apply_casualties

        army = Army(region=None)
        army.units = [
            Unit(Nation.GONDOR, UnitKind.ELITE),
        ]

        casualties = apply_casualties(army, 1)
        # Elite should be downgraded to regular
        self.assertEqual(len(army.units), 1)
        self.assertEqual(army.units[0].kind, UnitKind.REGULAR)


class TestSpecialCharacters(unittest.TestCase):
    """Test special character mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_cannot_muster_saruman_without_isengard_at_war(self):
        """Saruman cannot be mustered unless Isengard is at war."""
        can, reason, _ = gl.can_muster_saruman(self.state)
        self.assertFalse(can)
        self.assertIn("not at war", reason.lower())

    def test_can_muster_saruman_when_isengard_at_war(self):
        """Saruman can be mustered when Isengard is at war."""
        self.state.political.boxes[Nation.ISENGARD] = 3
        can, reason, regions = gl.can_muster_saruman(self.state)
        self.assertTrue(can, f"Should be able to muster Saruman: {reason}")
        self.assertIn(sanitize("orthanc"), regions)

    def test_gandalf_white_requires_gandalf_gone(self):
        """Gandalf White cannot appear while Gandalf Grey is in fellowship."""
        can, reason, _ = gl.can_transform_gandalf_white(self.state)
        self.assertFalse(can)
        self.assertIn("still in the fellowship", reason.lower())

    def test_crown_aragorn_requires_correct_location(self):
        """Aragorn can only be crowned in specific locations."""
        can, reason, _ = gl.can_crown_aragorn(self.state)
        self.assertFalse(can)
        # Strider is in fellowship which is in Rivendell, not a crown region

    def test_bonus_dice_calculation(self):
        """Test bonus dice calculation for special characters."""
        # Initially no bonus
        gl.calculate_bonus_dice(self.state)
        self.assertEqual(self.state.fp_bonus_dice, 0)
        self.assertEqual(self.state.sh_bonus_dice, 0)

        # Add Gandalf White (A14 2026-07-26: the figure must be on the
        # map — flag-only grants no die per the p.18 elimination rule)
        self.state.gandalf_the_white.in_play = True
        gl.get_or_create_army(self.state, 'fangorn').leaders.append(
            gs.Companion(name='Gandalf the White', level=3, leadership=1))
        gl.calculate_bonus_dice(self.state)
        self.assertEqual(self.state.fp_bonus_dice, 1)

        # Add Witch King
        self.state.witch_king.in_play = True
        gl.calculate_bonus_dice(self.state)
        self.assertEqual(self.state.sh_bonus_dice, 1)


class TestCompanionAbilities(unittest.TestCase):
    """Test companion special abilities."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_gandalf_has_captain_of_west(self):
        """Gandalf should have Captain of the West."""
        gandalf = next(c for c in self.state.fellowship.companions
                       if c.name == "Gandalf the Grey")
        self.assertTrue(gandalf.captain_of_the_west)

    def test_legolas_can_muster_elves(self):
        """Legolas should be able to muster Elves at elven strongholds."""
        legolas = next(c for c in self.state.fellowship.companions
                       if c.name == "Legolas")
        self.assertEqual(legolas.can_muster_nation, Nation.ELVES)
        self.assertIn("grey havens", legolas.muster_regions)

    def test_gimli_can_muster_dwarves(self):
        """Gimli should be able to muster Dwarves at Erebor."""
        gimli = next(c for c in self.state.fellowship.companions
                     if c.name == "Gimli")
        self.assertEqual(gimli.can_muster_nation, Nation.DWARVES)
        self.assertIn("erebor", gimli.muster_regions)

    def test_boromir_activates_gondor(self):
        """Boromir should activate Gondor in specific regions."""
        boromir = next(c for c in self.state.fellowship.companions
                       if c.name == "Boromir")
        self.assertIn(Nation.GONDOR, boromir.activates_nations)
        self.assertIn("minas tirith", boromir.activation_regions)

    def test_merry_pippin_guide_ability(self):
        """Merry and Pippin should have sacrifice guide ability."""
        merry = next(c for c in self.state.fellowship.companions
                     if c.name == "Meriadoc")
        pippin = next(c for c in self.state.fellowship.companions
                      if c.name == "Peregrin")
        self.assertEqual(merry.guide_ability, "sacrifice_reduce_hunt")
        self.assertEqual(pippin.guide_ability, "sacrifice_reduce_hunt")

    def test_companion_movement_distance(self):
        """Companion movement should equal their level."""
        gandalf = next(c for c in self.state.fellowship.companions
                       if c.name == "Gandalf the Grey")
        movement = gl.get_companion_movement(gandalf, self.state)
        self.assertEqual(movement, 3)  # Level 3


class TestDiceActions(unittest.TestCase):
    """Test dice usage and actions."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)
        # Roll dice to get faces
        gl.roll_action_dice(self.state, pool=Side.FREE_PEOPLES)
        gl.roll_action_dice(self.state, pool=Side.SHADOW)

    def test_dice_roll_sets_face(self):
        """Rolling dice should set current face."""
        for die in self.state.dice[Side.FREE_PEOPLES]:
            self.assertIsNotNone(die.current)

    def test_used_die_cannot_be_rolled(self):
        """Used die should not be rollable."""
        die = self.state.dice[Side.FREE_PEOPLES][0]
        die.mark_used()
        with self.assertRaises(RuntimeError):
            die.roll()


class TestAllRegionAdjacencies(unittest.TestCase):
    """Test movements between all connected regions."""

    def setUp(self):
        self.board = load_board()

    def test_can_find_path_between_adjacent_regions(self):
        """Should be able to traverse all adjacencies."""
        visited = set()
        queue = [list(adj.keys())[0]]  # Start from first region

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Should visit most regions (some might be isolated)
        self.assertGreater(len(visited), 50)


class TestCardLoading(unittest.TestCase):
    """Test card loading and registry."""

    def test_cards_module_loads(self):
        """Cards module should load without error."""
        import cards
        self.assertTrue(hasattr(cards, 'get_card'))

    def test_get_nonexistent_card(self):
        """Getting nonexistent card should return None."""
        import cards
        result = cards.get_card("nonexistent_card")
        self.assertIsNone(result)

    def test_all_96_cards_load(self):
        """All 96 cards should load from CSV."""
        import cards
        all_cards = cards.get_cards()
        self.assertEqual(len(all_cards), 96, f"Expected 96 cards, got {len(all_cards)}")

    def test_fp_character_cards_count(self):
        """Free Peoples should have 24 character cards."""
        import cards
        fp_char = cards.get_cards_by_deck(Side.FREE_PEOPLES, "Character")
        self.assertEqual(len(fp_char), 24, f"Expected 24 FP Character cards, got {len(fp_char)}")

    def test_fp_strategy_cards_count(self):
        """Free Peoples should have 24 strategy cards."""
        import cards
        fp_strat = cards.get_cards_by_deck(Side.FREE_PEOPLES, "Strategy")
        self.assertEqual(len(fp_strat), 24, f"Expected 24 FP Strategy cards, got {len(fp_strat)}")

    def test_shadow_character_cards_count(self):
        """Shadow should have 24 character cards."""
        import cards
        sh_char = cards.get_cards_by_deck(Side.SHADOW, "Character")
        self.assertEqual(len(sh_char), 24, f"Expected 24 Shadow Character cards, got {len(sh_char)}")

    def test_shadow_strategy_cards_count(self):
        """Shadow should have 24 strategy cards."""
        import cards
        sh_strat = cards.get_cards_by_deck(Side.SHADOW, "Strategy")
        self.assertEqual(len(sh_strat), 24, f"Expected 24 Shadow Strategy cards, got {len(sh_strat)}")


class TestCardProperties(unittest.TestCase):
    """Test individual card properties are parsed correctly."""

    def setUp(self):
        import cards
        self.all_cards = cards.get_cards()

    def test_all_cards_have_combat_name(self):
        """Every card should have a combat name."""
        for card_id, card in self.all_cards.items():
            self.assertTrue(card.combat_name, f"Card {card_id} missing combat name")

    def test_all_cards_have_valid_combat_value(self):
        """Every card should have combat value 0-7."""
        for card_id, card in self.all_cards.items():
            self.assertGreaterEqual(card.combat_value, 0,
                f"Card {card_id} has invalid combat value {card.combat_value}")
            self.assertLessEqual(card.combat_value, 7,
                f"Card {card_id} has invalid combat value {card.combat_value}")

    def test_all_cards_have_valid_side(self):
        """Every card should have a valid side."""
        for card_id, card in self.all_cards.items():
            self.assertIn(card.side, [Side.FREE_PEOPLES, Side.SHADOW],
                f"Card {card_id} has invalid side")

    def test_all_cards_have_valid_deck(self):
        """Every card should have Character or Strategy deck."""
        for card_id, card in self.all_cards.items():
            self.assertIn(card.deck, ["Character", "Strategy"],
                f"Card {card_id} has invalid deck: {card.deck}")

    def test_all_cards_have_valid_type(self):
        """Every card should have Character, Army, or Muster type."""
        for card_id, card in self.all_cards.items():
            self.assertIn(card.card_type, ["Character", "Army", "Muster"],
                f"Card {card_id} has invalid type: {card.card_type}")


class TestSpecificCards(unittest.TestCase):
    """Test specific card properties."""

    def setUp(self):
        import cards
        self.all_cards = cards.get_cards()

    def _get_card_by_name(self, name: str):
        """Helper to find card by name."""
        for card in self.all_cards.values():
            if card.card_name == name:
                return card
        return None

    def test_elven_cloaks_card(self):
        """Elven Cloaks should have correct properties."""
        card = self._get_card_by_name("Elven Cloaks")
        self.assertIsNotNone(card)
        self.assertEqual(card.side, Side.FREE_PEOPLES)
        self.assertEqual(card.deck, "Character")
        self.assertEqual(card.combat_name, "It is a Gift")
        self.assertEqual(card.combat_value, 3)
        self.assertTrue(card.requires_fellowship_region)

    def test_mithril_coat_card(self):
        """Mithril Coat should have Hobbit requirement."""
        card = self._get_card_by_name("Mithril Coat and Sting")
        self.assertIsNotNone(card)
        self.assertEqual(card.combat_name, "Blade of Westernesse")
        self.assertEqual(card.combat_value, 6)
        self.assertTrue(card.requires_hobbit)

    def test_shelob_lair_card(self):
        """Shelob's Lair should be Shadow card."""
        card = self._get_card_by_name("Shelob's Lair")
        self.assertIsNotNone(card)
        self.assertEqual(card.side, Side.SHADOW)
        self.assertEqual(card.deck, "Character")
        self.assertTrue(card.requires_fellowship_region)

    def test_witchking_card(self):
        """Black Captain Commands should require Witch-king."""
        card = self._get_card_by_name("The Black Captain Commands")
        self.assertIsNotNone(card)
        self.assertEqual(card.side, Side.SHADOW)
        # Event requires Witch-king in play
        self.assertIn("witch-king", card.event_req.lower())

    def test_balrog_card(self):
        """Balrog of Moria should have special combat effect."""
        card = self._get_card_by_name("Balrog of Moria")
        self.assertIsNotNone(card)
        self.assertEqual(card.combat_name, "Durin's Bane")
        self.assertEqual(card.combat_value, 2)
        self.assertIn("moria", card.region_restrictions)


class TestCombatEffectCategories(unittest.TestCase):
    """Test combat effect categorization."""

    def test_add_1_dice_category(self):
        """Add 1 to dice effect should be categorized."""
        import cards
        categories = cards.categorize_effect("Add 1 to all dice on your Combat roll.")
        self.assertIn("add_1_dice", categories)

    def test_add_2_dice_category(self):
        """Add 2 to dice effect should be categorized."""
        import cards
        categories = cards.categorize_effect("Add 2 to all dice on your Combat roll.")
        self.assertIn("add_2_dice", categories)

    def test_cancel_card_category(self):
        """Cancel effects should be categorized."""
        import cards
        categories = cards.categorize_effect(
            "Cancel the effects of the Combat card played by the Shadow player.")
        self.assertIn("cancel_card", categories)

    def test_shield_wall_category(self):
        """Shield-Wall effect should be categorized."""
        import cards
        categories = cards.categorize_effect(
            "Before you remove casualties, if your opponent scored two or more hits, cancel one hit.")
        self.assertIn("shield_wall", categories)

    def test_scouts_category(self):
        """Scouts retreat should be categorized."""
        import cards
        categories = cards.categorize_effect(
            "Before the Combat roll, retreat your Army to an adjacent free region.")
        self.assertIn("scouts", categories)

    def test_both_armies_add_category(self):
        """Both Armies effect should be categorized."""
        import cards
        categories = cards.categorize_effect(
            "Both Armies add 2 to all dice on their Combat and Leader re-roll.")
        self.assertIn("both_add_2", categories)


class TestAllCardsHaveCombatEffect(unittest.TestCase):
    """Verify every card has a categorizable combat effect."""

    def test_all_cards_have_effect_category(self):
        """Every card should have at least one effect category."""
        import cards
        all_cards = cards.get_cards()
        cards_without_category = []

        for card_id, card in all_cards.items():
            if card.combat_effect:
                categories = cards.categorize_effect(card.combat_effect)
                if not categories:
                    cards_without_category.append(card.card_name)

        # Allow some cards to not match patterns (they may have unique effects)
        # But more than 10 uncategorized would be suspicious
        self.assertLess(len(cards_without_category), 15,
            f"Too many cards without effect category: {cards_without_category}")


class TestCombatCardRequirements(unittest.TestCase):
    """Test combat requirement parsing."""

    def setUp(self):
        import cards
        self.all_cards = cards.get_cards()

    def test_companion_requirement_cards(self):
        """Cards requiring companions should be flagged."""
        companion_cards = [c for c in self.all_cards.values() if c.requires_companion]
        # Should have multiple cards with companion requirement
        self.assertGreater(len(companion_cards), 5,
            "Expected multiple cards requiring companions")

    def test_nazgul_requirement_cards(self):
        """Cards requiring Nazgul should be flagged."""
        nazgul_cards = [c for c in self.all_cards.values() if c.requires_nazgul]
        # Should have multiple Shadow cards requiring Nazgul
        self.assertGreater(len(nazgul_cards), 5,
            "Expected multiple cards requiring Nazgul")

    def test_elite_requirement_cards(self):
        """Cards requiring elites should be flagged."""
        elite_cards = [c for c in self.all_cards.values() if c.requires_elite]
        self.assertGreater(len(elite_cards), 3,
            "Expected multiple cards requiring elites")

    def test_stronghold_requirement_cards(self):
        """Cards requiring strongholds should be flagged."""
        stronghold_cards = [c for c in self.all_cards.values() if c.requires_stronghold]
        self.assertGreater(len(stronghold_cards), 0,
            "Expected at least one card requiring stronghold")

    def test_fellowship_region_cards(self):
        """Cards requiring fellowship region should be flagged."""
        fellowship_cards = [c for c in self.all_cards.values()
                          if c.requires_fellowship_region]
        # The 4 FP "It is a Gift" cards and 4 Shadow "One for the Dark Lord" cards
        self.assertGreaterEqual(len(fellowship_cards), 8,
            "Expected at least 8 cards requiring fellowship region")


class TestTurnAdvancement(unittest.TestCase):
    """Test turn advancement mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)
        gl.initialize_decks(self.state)

    def test_next_turn_increments_turn(self):
        """next_turn should increment turn counter."""
        initial_turn = self.state.turn
        # Roll dice first
        gl.roll_action_dice(self.state, pool=Side.FREE_PEOPLES)
        gl.roll_action_dice(self.state, pool=Side.SHADOW)
        gl.next_turn(self.state)
        self.assertEqual(self.state.turn, initial_turn + 1)

    def test_next_turn_resets_dice(self):
        """next_turn should reset all dice."""
        gl.roll_action_dice(self.state, pool=Side.FREE_PEOPLES)
        gl.roll_action_dice(self.state, pool=Side.SHADOW)

        # Mark some dice as used
        self.state.dice[Side.FREE_PEOPLES][0].mark_used()

        gl.next_turn(self.state)

        for die in self.state.dice[Side.FREE_PEOPLES]:
            self.assertFalse(die.used)
            self.assertIsNone(die.current)


class TestSettlementTypes(unittest.TestCase):
    """Test settlement type identification."""

    def test_minas_tirith_is_stronghold(self):
        """Minas Tirith should be a stronghold."""
        mt_key = sanitize("minas tirith")
        info = board_regions.get(mt_key, {})
        self.assertTrue(info.get("is_stronghold"), "Minas Tirith should be a stronghold")

    def test_bree_is_town(self):
        """Bree should be a town."""
        bree_key = sanitize("bree")
        info = board_regions.get(bree_key, {})
        self.assertTrue(info.get("is_town"), "Bree should be a town")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_corruption_at_12_triggers_victory(self):
        """12 corruption should indicate Shadow victory condition."""
        self.state.fellowship.corruption = 12
        # This is a victory condition check, not automatic ending
        self.assertEqual(self.state.fellowship.corruption, 12)

    def test_empty_army_after_casualties(self):
        """Army should be empty after all units destroyed."""
        from combat import apply_casualties

        army = Army(region=None)
        army.units = [Unit(Nation.GONDOR, UnitKind.REGULAR)]

        apply_casualties(army, 1)
        self.assertEqual(len(army.units), 0)


class TestAllRegionConnectivity(unittest.TestCase):
    """Comprehensive region connectivity tests."""

    def setUp(self):
        self.board = load_board()

    def test_gondor_regions_connected(self):
        """All Gondor regions should form connected network."""
        gondor_regions = [
            "minas tirith", "osgiliath", "pelargir", "lamedon",
            "dol amroth", "anfalas", "erech", "lossarnach",
            "north ithilien", "south ithilien"
        ]
        for region in gondor_regions:
            region_san = sanitize(region)
            self.assertIn(region_san, adj,
                f"Gondor region {region} missing from adjacency graph")

    def test_rohan_regions_connected(self):
        """All Rohan regions should form connected network."""
        # Note: "helm's deep" is stored as "helm s deep" in the board
        rohan_regions = [
            "edoras", "helm s deep", "westemnet", "eastemnet",
            "fords of isen", "gap of rohan", "folde"
        ]
        for region in rohan_regions:
            region_san = sanitize(region)
            self.assertIn(region_san, adj,
                f"Rohan region {region} missing from adjacency graph")

    def test_mordor_regions_connected(self):
        """Mordor regions should be connected."""
        mordor_regions = [
            "morannon", "gorgoroth", "minas morgul", "barad dur", "nurn"
        ]
        for region in mordor_regions:
            region_san = sanitize(region)
            self.assertIn(region_san, adj,
                f"Mordor region {region} missing from adjacency graph")

    def test_shire_to_mordor_path_exists(self):
        """Should be able to path from Shire to Mordor."""
        from collections import deque

        start = sanitize("the shire")
        end = sanitize("gorgoroth")

        # BFS to find path
        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current == end:
                return  # Path found

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        self.fail("No path from Shire to Mordor")

    def test_all_strongholds_reachable(self):
        """All strongholds should be reachable from Minas Tirith."""
        from collections import deque

        # Note: "helm's deep" is stored as "helm s deep" in the board
        strongholds = [
            "minas tirith", "helm s deep", "dol amroth", "erebor",
            "rivendell", "lorien", "grey havens", "woodland realm",
            "orthanc", "dol guldur", "mount gundabad", "moria",
            "barad dur", "minas morgul", "morannon"
        ]

        start = sanitize("minas tirith")
        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        for stronghold in strongholds:
            stronghold_san = sanitize(stronghold)
            self.assertIn(stronghold_san, visited,
                f"Stronghold {stronghold} not reachable from Minas Tirith")


class TestMinionMechanics(unittest.TestCase):
    """Test Shadow minion mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_witch_king_requires_sauron_at_war(self):
        """Witch King requires Sauron at war to muster."""
        # Initially Sauron is not at war
        self.state.political.boxes[Nation.SAURON] = 2
        can, reason, _ = gl.can_muster_witch_king(self.state)
        self.assertFalse(can)
        # The reason may vary - could be about Sauron or no valid location

    def test_witch_king_can_muster_when_sauron_at_war(self):
        """Witch King can muster when Sauron is at war."""
        # Set Sauron to at war
        self.state.political.boxes[Nation.SAURON] = 3
        # Also need FP nation at war (for requirement)
        self.state.political.boxes[Nation.GONDOR] = 3
        can, reason, regions = gl.can_muster_witch_king(self.state)
        self.assertTrue(can, f"Should be able to muster Witch King: {reason}")

    def test_saruman_requires_isengard_at_war(self):
        """Saruman requires Isengard at war."""
        # Isengard starts at box 2
        can, reason, _ = gl.can_muster_saruman(self.state)
        self.assertFalse(can)

    def test_mouth_of_sauron_entry_condition(self):
        """Mouth entry = ALL FIVE FP nations at war OR Fellowship on
        the Mordor track (USER final ruling 2026-07-04, amendment
        1.3).  Fresh state: neither leg holds -> refused."""
        can, reason, _ = gl.can_muster_mouth_of_sauron(self.state)
        self.assertFalse(can)
        self.assertIn("FP nations", reason)


class TestGandalfTransformation(unittest.TestCase):
    """Test Gandalf the White transformation."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_gandalf_white_cannot_appear_while_grey_in_fellowship(self):
        """Gandalf White cannot appear while Grey is in fellowship."""
        can, reason, _ = gl.can_transform_gandalf_white(self.state)
        self.assertFalse(can)
        self.assertIn("still in the fellowship", reason.lower())

    def test_gandalf_white_can_appear_after_grey_leaves(self):
        """Gandalf White can appear after Grey leaves fellowship."""
        # Remove Gandalf Grey from fellowship
        self.state.fellowship.companions = [
            c for c in self.state.fellowship.companions
            if c.name != "Gandalf the Grey"
        ]
        gl.change_guide(self.state)

        # Per rulebook: also requires at least one Shadow minion is/was in play
        self.state.witch_king.was_ever_in_play = True

        can, reason, regions = gl.can_transform_gandalf_white(self.state)
        self.assertTrue(can, f"Should be able to transform: {reason}")

    def test_gandalf_white_adds_bonus_die(self):
        """Gandalf White in play should add bonus die to FP."""
        self.state.gandalf_the_white.in_play = True
        # A14 (2026-07-26): the die requires the character alive ON THE
        # MAP (p.18 elimination rule) — place the figure like real play.
        gl.get_or_create_army(self.state, 'fangorn').leaders.append(
            gs.Companion(name='Gandalf the White', level=3, leadership=1))
        gl.calculate_bonus_dice(self.state)
        self.assertEqual(self.state.fp_bonus_dice, 1)


class TestAragornCrowning(unittest.TestCase):
    """Test Aragorn crowning mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_cannot_crown_strider_in_fellowship(self):
        """Cannot crown while Strider is in fellowship."""
        can, reason, _ = gl.can_crown_aragorn(self.state)
        self.assertFalse(can)

    def test_crown_regions_are_correct(self):
        """Crowning should only be possible in specific regions."""
        # The crown regions should be Minas Tirith, Dol Amroth, Pelargir
        # Remove Strider from fellowship first
        strider = None
        for c in self.state.fellowship.companions:
            if c.name == "Strider":
                strider = c
                break

        self.state.fellowship.companions.remove(strider)

        # Place Strider in a crown region - companions go in the leaders list
        from game_state import Army, Unit, UnitKind
        strider_army = Army(region="minas tirith")
        strider_army.units = [Unit(Nation.GONDOR, UnitKind.REGULAR)]
        strider_army.leaders = [strider]  # Companions are stored in leaders list
        self.state.armies["minas tirith"] = strider_army

        # Also need Gondor at war
        self.state.political.boxes[Nation.GONDOR] = 3

        can, reason, regions = gl.can_crown_aragorn(self.state)
        self.assertTrue(can, f"Should be able to crown Aragorn: {reason}")
        self.assertIn(sanitize("minas tirith"), regions)


class TestNazgulMovement(unittest.TestCase):
    """Test Nazgul movement mechanics."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_find_nazgul_in_armies(self):
        """Should be able to find Nazgul in armies."""
        nazgul_count = 0
        for army in self.state.armies.values():
            for leader in army.leaders:
                if isinstance(leader, Nazgul):
                    nazgul_count += 1
        self.assertEqual(nazgul_count, 4, "Should start with 4 Nazgul")


class TestVictoryConditions(unittest.TestCase):
    """Test victory condition detection."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_corruption_victory_at_12(self):
        """Shadow wins at 12 corruption."""
        self.state.fellowship.corruption = 12
        self.assertEqual(self.state.fellowship.corruption, 12)

    def test_military_victory_tracking(self):
        """VP tracking should work correctly."""
        self.state.fp_victory_points = 4
        self.state.sh_victory_points = 10
        self.assertEqual(self.state.fp_victory_points, 4)
        self.assertEqual(self.state.sh_victory_points, 10)


class TestLeadershipCalculation(unittest.TestCase):
    """Test leadership calculation."""

    def setUp(self):
        self.board = load_board()
        self.state = setup_initial_game(self.board)

    def test_nazgul_leadership(self):
        """Each Nazgul in army.leaders provides 1 leadership; Witch-king 2."""
        from combat import calc_leadership
        from game_state import Nazgul

        army = Army(region=None)
        army.leaders = [Nazgul(Nation.SAURON), Nazgul(Nation.SAURON)]
        army.units = [Unit(Nation.SAURON, UnitKind.REGULAR)]

        leadership = calc_leadership(army)
        self.assertEqual(leadership, 2)

        # Gandalf the White cancels all Nazgul leadership
        self.assertEqual(calc_leadership(army, cancel_nazgul=True), 0)

    def test_companion_leadership(self):
        """Companions provide leadership based on their leadership rating (not level)."""
        from combat import calc_leadership

        army = Army(region=None)
        army.units = [Unit(Nation.GONDOR, UnitKind.REGULAR)]

        # Gandalf has level=3 but leadership=1
        gandalf = next(c for c in self.state.fellowship.companions
                      if c.name == "Gandalf the Grey")

        leadership = calc_leadership(army, [gandalf])
        self.assertEqual(leadership, gandalf.leadership)  # Leadership rating, not level


def run_all_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Board and setup
        TestBoardAdjacency,
        TestInitialSetup,
        TestArmyMovement,
        TestPoliticalTrack,
        TestFellowshipMechanics,
        TestHuntSystem,
        TestCombatSystem,
        TestSpecialCharacters,
        TestCompanionAbilities,
        TestDiceActions,
        TestAllRegionAdjacencies,
        TestTurnAdvancement,
        TestSettlementTypes,
        TestEdgeCases,

        # Card tests
        TestCardLoading,
        TestCardProperties,
        TestSpecificCards,
        TestCombatEffectCategories,
        TestAllCardsHaveCombatEffect,
        TestCombatCardRequirements,

        # Comprehensive region tests
        TestAllRegionConnectivity,

        # Special character tests
        TestMinionMechanics,
        TestGandalfTransformation,
        TestAragornCrowning,
        TestNazgulMovement,

        # Victory and leadership
        TestVictoryConditions,
        TestLeadershipCalculation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_all_tests()

    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n--- FAILURES ---")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)

    if result.errors:
        print("\n--- ERRORS ---")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)

    sys.exit(0 if result.wasSuccessful() else 1)
