"""Dead Men C22: optional recruitment through engine, encoding and log labels.

Oracle: card_list.csv says "may then recruit up to three"; Almanac pp15-16
places recruitment after the attack/forced retreat and limits it to supply.
Expected counts, phases and resource effects below are independently authored.
"""
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import ai
import board
import event_effects as ee
import game_logic as gl
import game_state as gs
import rendered_extract as rx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CID = 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow'
TITLE = 'Dead Men of Dunharrow/Sudden Strike'


def before_play(available=15):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.armies.clear()
    strider = next(c for c in state.fellowship.companions if c.name == 'Strider')
    state.fellowship.companions.remove(strider)
    gl.get_or_create_army(state, 'westemnet').leaders.append(strider)
    state.hands[FP].hand = [CID]
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        deck = getattr(state, name)
        if CID in deck:
            deck.remove(CID)
    state.fp_casualties = {gs.Nation.GONDOR: {'regular': 15 - available}}
    die = gs.ActionDie([gs.ActionDieFace.CHARACTER], FP)
    die.current = gs.ActionDieFace.CHARACTER
    state.dice = {FP: [die], SH: []}
    state.enemy_controlled = {FP: set(), SH: {'pelargir'}}
    return state


def choose(state, side, **wanted):
    menu = ai.get_legal_actions(state, side)
    selected = [a for a in menu if a.action_type == 'card_option'
                and all((a.params.get('choices') or {}).get(k) == v
                        for k, v in wanted.items())]
    if len(selected) != 1:
        raise AssertionError('Expected exactly one fixed player choice: %r' % wanted)
    return ai.apply_action(state, selected[0])


def target_menu(state):
    return ai.apply_action(state, ai.Action(0, 'play_event', {
        'card_id': CID, 'side': 'fp'}))


def gondor_at(state, region='pelargir'):
    army = next((a for k, a in state.armies.items() if board.sanitize(k) == region), None)
    return sum(u.nation == gs.Nation.GONDOR and u.kind == gs.UnitKind.REGULAR
               for u in (army.units if army else []))


def record(line, obj, frm, to):
    return {'op': 'move', 'line': line, 'obj': obj, 'from': frm, 'to': to}


def recorded_batch(count, complete=True, wrong_pool=False):
    """Reviewed synthetic transcript, not snapshots produced by the engine."""
    records = [record(10, TITLE, 'Free Peoples Hand', 'Current FP Card'),
               record(11, 'Strider', 'Westemnet', 'Pelargir')]
    if count:
        records.append(record(12, '[%d]Gondor Regular' % count,
                              'casualties' if wrong_pool else 'FP Reinforcements',
                              'Pelargir'))
    if complete:
        records.append({'op': 'kill', 'line': 13, 'zone': 'Current Action',
                        'pieces': ['Free Action Die (C)', TITLE]})
    fellowship = {'region': 'rivendell', 'comps': [], 'corruption': 0}
    pre = {'armies': {'westemnet': ([], [('NAMED', 'Strider')])},
           'pol': {}, 'fel': fellowship}
    post = {'armies': {'pelargir': ([('GONDOR', 'REGULAR')] * count,
                                  [('NAMED', 'Strider')])},
            'pol': {}, 'fel': fellowship}
    return {'kind': 'bookkeeping', 'die_side': 'FREE_PEOPLES', 'die_face': 'C',
            'start_line': 9, 'end_line': 14, 'turn': 1, 'records': records,
            'rng': [], 'truth_before': pre, 'truth': post}


class DeadMenDecision(unittest.TestCase):
    def test_destination_and_count_are_separate_player_decisions(self):
        state = target_menu(before_play())
        self.assertEqual({o['target_region'] for o in state.pending_card_option['options']},
                         {'erech', 'lamedon', 'pelargir'})
        self.assertFalse(any('recruit_count' in o for o in state.pending_card_option['options']))
        state = choose(state, FP, target_region='pelargir')
        self.assertEqual(state.pending_card_option['decider'], 'fp')
        self.assertEqual({o['recruit_count'] for o in state.pending_card_option['options']},
                         {0, 1, 2, 3})
        self.assertEqual(gondor_at(state), 0)
        self.assertNotIn(CID, state.hands[FP].hand)
        self.assertTrue(state.dice[FP][0].used)

    def test_each_count_changes_only_recruitment_and_control(self):
        prepared = choose(target_menu(before_play()), FP, target_region='pelargir')
        for count in range(4):
            with self.subTest(count=count):
                with patch.object(ee, '_d6_batch', side_effect=AssertionError('repeated attack')):
                    result = choose(prepared, FP, recruit_count=count)
                self.assertEqual(gondor_at(result), count)
                self.assertEqual('pelargir' in result.enemy_controlled[SH], count == 0)
                self.assertIsNone(result.pending_card_option)
                self.assertTrue(result.dice[FP][0].used)
                self.assertEqual(result.discard_face_up['fp_character'].count(CID), 1)

    def test_limited_pool_does_not_offer_equivalent_unavailable_counts(self):
        state = choose(target_menu(before_play(available=1)), FP, target_region='pelargir')
        self.assertEqual({o['recruit_count'] for o in state.pending_card_option['options']}, {0, 1})
        result = choose(state, FP, recruit_count=1)
        self.assertEqual(gondor_at(result), 1)

    def test_empty_pool_has_no_phantom_choice_or_capture(self):
        state = choose(target_menu(before_play(available=0)), FP, target_region='pelargir')
        self.assertIsNone(state.pending_card_option)
        self.assertEqual(gondor_at(state), 0)
        self.assertIn('pelargir', state.enemy_controlled[SH])

    def test_recruit_choice_is_after_shadow_retreat(self):
        state = before_play()
        army = gl.get_or_create_army(state, 'pelargir')
        army.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(2)]
        with patch.object(ee, '_d6_batch', return_value=[1]):
            state = choose(target_menu(state), FP, target_region='pelargir')
        self.assertEqual(state.pending_card_option['decider'], 'sh')
        self.assertFalse(any('recruit_count' in o for o in state.pending_card_option['options']))
        state = choose(state, SH, destination='west harondor')
        self.assertEqual(state.pending_card_option['decider'], 'fp')
        self.assertEqual(gondor_at(state), 0)
        state = choose(state, FP, recruit_count=2)
        self.assertEqual(gondor_at(state), 2)

    def test_four_counts_have_four_distinct_action_encodings(self):
        state = choose(target_menu(before_play()), FP, target_region='pelargir')
        menu = ai.get_legal_actions(state, FP)
        encodings = [ai.encode_action(action, state, FP) for action in menu]
        self.assertEqual(len({encoding.tobytes() for encoding in encodings}), 4)
        for action, encoding in zip(menu, encodings):
            self.assertAlmostEqual(float(encoding[23]),
                                   action.params['choices']['recruit_count'] / 3.0)

    def test_invalid_count_rejects_without_changing_the_pending_state(self):
        state = choose(target_menu(before_play(available=1)), FP, target_region='pelargir')
        pending = state.pending_card_option
        with self.assertRaisesRegex(ValueError, 'count is not available'):
            ai.apply_action(state, ai.Action(-1, 'card_option', {
                'card_id': CID, 'choices': {'target_region': 'pelargir',
                '_dmd_recruit_only': True, 'recruit_count': 2}}))
        self.assertIs(state.pending_card_option, pending)
        self.assertEqual(gondor_at(state), 0)

    def test_bridge_advertised_pick_can_decline_without_board_operations(self):
        from wotr_bridge.live.chat import match_choice
        from wotr_bridge.live.emission_card import render_card_effect
        state = choose(target_menu(before_play()), FP, target_region='pelargir')
        menu = ai.get_legal_actions(state, FP)
        zero_index = next(i for i, a in enumerate(menu) if a.params['choices']['recruit_count'] == 0)
        action = match_choice('pick %d' % (zero_index + 1), menu)
        self.assertIs(action, menu[zero_index])
        post = ai.apply_action(state, action)
        emitted = render_card_effect(SimpleNamespace(state=state), 'AI', FP,
                                     'card_option', action, state, post)
        self.assertFalse(emitted.hold)
        self.assertFalse(emitted.lines)


class DeadMenExpertLabels(unittest.TestCase):
    def test_recorded_game15_play_reaches_real_parser_engine_and_encoding(self):
        """Existing expert log lines 788-794 explicitly recruits three.

        Hand inventory is seeded only with the publicly played card; all
        board reconstruction, matching, rule application and encoding are real.
        The sample remains in memory and is never written to a corpus.
        """
        import rendered_replay as rr
        path = ROOT / 'data' / 'rendered_logs' / (
            'GAME15_2025-02-05_204637_FP_Vayze_SP_ProniX.rendered.txt')
        game = rr.parse_game(str(path))
        rx._annotate_hero_flags(game['batches'])
        batch = next(b for b in game['batches'] if b['start_line'] == 788)
        def recorded_card(state, _batch):
            state.hands[FP].hand = [CID]
        stats = defaultdict(Counter)
        with patch.object(rx, '_install_ledger_hands', side_effect=recorded_card):
            samples = rx._emit_card_interiors(batch, stats)
        self.assertEqual(stats['kinds_emit']['cardopt_dmd_recruit'], 1)
        self.assertEqual(len(samples), 2)
        sample = samples[-1]
        self.assertEqual(sample['side'], 'fp')
        self.assertEqual(len(sample['encs']), 4)
        self.assertEqual(float(sample['encs'][sample['human']][23]), 1.0)

    def test_complete_zero_one_two_three_counts_match_unique_training_rows(self):
        for count in range(4):
            with self.subTest(count=count):
                batch = recorded_batch(count)
                stats = defaultdict(Counter)
                with patch.object(rx, '_mk_state_with_die', return_value=(before_play(), FP)):
                    samples = rx._emit_card_interiors(batch, stats)
                self.assertEqual(stats['kinds_emit']['cardopt_dmd_recruit'], 1, dict(stats))
                self.assertEqual(len(samples), 2)
                sample = samples[-1]
                self.assertEqual(len(sample['encs']), 4)
                self.assertAlmostEqual(float(sample['encs'][sample['human']][23]), count / 3.0)
                self.assertEqual(sample['side'], 'fp')

    def test_partial_log_never_means_the_player_chose_zero(self):
        batch = recorded_batch(0, complete=False)
        stats = defaultdict(Counter)
        with patch.object(rx, '_mk_state_with_die', return_value=(before_play(), FP)):
            samples = rx._emit_card_interiors(batch, stats)
        self.assertEqual(len(samples), 1)  # destination only
        self.assertEqual(stats['kinds_emit']['cardopt_dmd_recruit'], 0)
        self.assertEqual(stats['drop_reasons']['cardint_dmd_recruit_incomplete'], 1)

    def test_casualty_box_is_not_evidence_of_a_legal_fp_recruit(self):
        batch = recorded_batch(2, wrong_pool=True)
        stats = defaultdict(Counter)
        with patch.object(rx, '_mk_state_with_die', return_value=(before_play(), FP)):
            samples = rx._emit_card_interiors(batch, stats)
        self.assertEqual(len(samples), 1)
        self.assertEqual(stats['drop_reasons']['cardint_dmd_recruit_not_reinforcement'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
