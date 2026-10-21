"""AI conventions: literal choices, real effects, and search boundaries.

These are user policies, not restrictions on printed human legality.
No policy targets, models, corpora or released bundles are modified.
"""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ai
import ai_policies as policies
import game_state as gs
import game_trace
from test_audit0904_recruit_choices import (
    before_play, exhaust_gondor_pool, gondor_figures, FP, SH, CID, GONDOR)
from test_audit0904_gandalf_draw import opening_for, finish_house, discard
from test_audit0904_stewards_choices import S1, FILLERS
from test_audit0904_stewards_choices import before_play as house_before, CID as HOUSE


def guards_pending():
    return ai.apply_action(before_play(), ai.Action(0, 'play_event', {
        'card_id': CID, 'side': 'fp'}))


def projected(state, side=FP):
    return policies.project_ai_actions(state, side, ai.get_legal_actions(state, side))


def search_house_chain(*, batch=1, refuse_fixed=False):
    """One simulation with a literal learned House play and real continuation."""
    state = house_before(face=gs.ActionDieFace.EVENT)
    root_play = ai.Action(0, 'play_event', {'card_id': HOUSE, 'side': 'fp'})
    original = ai.get_ai_policy
    observed, applied = [], []
    chance_before = ai.OVERLAY_STATS['det_draw_chance']
    original_apply = ai._apply_action_detect_rng
    def apply(st, action):
        applied.append((action.action_type, dict(action.params.get('choices') or {})))
        if refuse_fixed and action.action_type == 'card_option':
            raise ValueError('literal fixed transition refusal')
        return original_apply(st, action)
    def menu(st, side):
        if getattr(st, 'pending_card_option', None) is None and HOUSE in st.hands[FP].hand:
            return policies.PolicyProjection((root_play,))
        return original(st, side)
    def evaluate(st, side, net, actions, *args, **kwargs):
        if original(st, side).forced_reason:
            raise AssertionError('No network at fixed transition')
        observed.append((len(st.hands[FP].hand), getattr(st, 'pending_card_option', None)))
        return [1 / len(actions)] * len(actions), .25
    with patch.object(ai, 'get_ai_policy', side_effect=menu), \
            patch.object(ai, '_get_priors_and_value', side_effect=evaluate), \
            patch.object(ai, '_mcts_eval_state', return_value=.25), \
            patch.object(ai, '_apply_action_detect_rng', side_effect=apply), \
            patch.object(ai, '_sims_budget', return_value=1), \
            patch.object(ai, '_LEAF_BATCH', batch), patch.object(ai, '_TREE_REUSE', False):
        try:
            chosen, visits = ai.mcts_search(state, FP, None, None,
                num_simulations=1, add_noise=False, determinize=True)
        finally:
            ai._SIM_OVERLAY = None
    return chosen, visits, applied, observed, ai.OVERLAY_STATS['det_draw_chance'] - chance_before


class FixedAiPolicies(unittest.TestCase):
    def test_guards_ai_elite_human_regular_and_original_objects_remain(self):
        state = guards_pending()
        legal = ai.get_legal_actions(state, FP)
        before = copy.deepcopy(state.pending_card_option)
        result = policies.project_ai_actions(state, FP, legal)
        self.assertEqual([a.params['choices']['unit_type'] for a in legal], ['regular', 'elite'])
        self.assertEqual(len(result.actions), 1)
        self.assertIs(result.actions[0], legal[1])
        self.assertEqual(result.forced_reason, policies.RECRUIT_ELITE)
        self.assertEqual(state.pending_card_option, before)
        self.assertEqual(gondor_figures(ai.apply_action(state, legal[0])), (4, 1, 2))
        self.assertEqual(gondor_figures(ai.apply_action(state, result.actions[0])), (3, 2, 2))

    def test_gandalf_human_decline_remains_ai_takes_once(self):
        state = opening_for(CID, 'strategy')
        before = len(state.hands[FP].hand)
        legal = ai.get_legal_actions(state, FP)
        result = projected(state)
        self.assertEqual([a.params['choices'] for a in result.actions], [{'gandalf_draw': True}])
        self.assertEqual(result.forced_reason, policies.BONUS_DRAW)
        declined = ai.apply_action(state, next(a for a in legal if not a.params['choices']['gandalf_draw']))
        self.assertEqual(len(declined.hands[FP].hand), before)
        taken = ai.apply_action(state, result.actions[0])
        self.assertEqual(len(taken.hands[FP].hand), before + 1)
        with self.assertRaises(ValueError):
            ai.apply_action(taken, result.actions[0])

    def test_first_discard_still_precedes_forced_bonus_and_second_discard(self):
        state = finish_house(full_hand=True)
        self.assertEqual(len(state.hands[FP].hand), 7)
        first = projected(state)
        self.assertIsNone(first.forced_reason)
        self.assertTrue(all(a.action_type == 'discard_card' for a in first.actions))
        state = discard(state, S1)
        self.assertEqual(len(state.hands[FP].hand), 6)
        bonus = projected(state)
        self.assertEqual(bonus.forced_reason, policies.BONUS_DRAW)
        state = ai.apply_action(state, bonus.actions[0])
        self.assertEqual(len(state.hands[FP].hand), 7)
        self.assertIsNone(projected(state).forced_reason)
        self.assertTrue(all(a.action_type == 'discard_card' for a in projected(state).actions))

    def test_ordinary_muster_elite_preference_retains_leader_region_and_nation(self):
        state = before_play()
        for nation in (GONDOR, gs.Nation.ROHAN):
            state.political.boxes[nation] = 3
            state.political.active[nation] = True
        result = projected(state)
        self.assertIsNone(result.forced_reason)
        self.assertFalse(any(a.action_type == 'muster_recruit' and a.params.get('type') == 'regular'
                             for a in result.actions))
        pairs = {(a.params['nation'], a.params['region']) for a in result.actions
                 if a.action_type == 'muster_elite'}
        self.assertIn((GONDOR, 'minas tirith'), pairs)
        self.assertIn((GONDOR, 'pelargir'), pairs)
        self.assertIn((gs.Nation.ROHAN, 'edoras'), pairs)
        self.assertTrue(any(a.action_type == 'muster_recruit' and a.params['type'] == 'leader'
                            for a in result.actions))

    def test_exhausted_elite_pool_keeps_regular_and_one_point_followup(self):
        state = before_play()
        state.political.boxes[GONDOR] = 3
        state.political.active[GONDOR] = True
        exhaust_gondor_pool(state, 'elite')
        first = ai.Action(0, 'muster_recruit', {
            'nation': GONDOR, 'region': 'minas tirith', 'type': 'regular'})
        self.assertIn(policies.action_policy_key(first),
                      [policies.action_policy_key(a) for a in projected(state).actions])
        state = ai.apply_action(state, first)
        self.assertEqual(state.pending_muster_die, 0)
        self.assertTrue(any(a.action_type == 'muster_recruit' and a.params['type'] == 'regular'
                            for a in projected(state).actions))

    def test_one_point_followup_keeps_regular_even_with_elites_in_reserve(self):
        state = before_play()
        state.political.boxes[GONDOR] = 3
        state.political.active[GONDOR] = True
        # A human may have chosen the first Regular; AI completion has one point.
        state = ai.apply_action(state, ai.Action(0, 'muster_recruit', {
            'nation': GONDOR, 'region': 'minas tirith', 'type': 'regular'}))
        acts = projected(state).actions
        self.assertTrue(any(a.action_type == 'muster_recruit' and a.params['type'] == 'regular'
                            for a in acts))
        self.assertFalse(any(a.action_type == 'muster_elite' for a in acts))

    def test_composite_scarce_elite_placement_and_unrelated_modes_are_not_fixed(self):
        state = SimpleNamespace(pending_card_option={})
        def action(a, b, region='minas tirith', mode='recruit'):
            return ai.Action(-1, 'card_option', {'card_id': 'fixture', 'choices': {
                'north_unit_type': a, 'dwarf_unit_type': b, 'region': region, 'mode': mode}})
        # Fixture legal menu has one scarce Elite shared by two placement choices.
        rows = [action('regular', 'regular'), action('elite', 'regular'),
                action('regular', 'elite'), action('regular', 'regular', 'pelargir'),
                action('regular', 'regular', mode='different-resource')]
        result = policies.project_ai_actions(state, FP, rows)
        self.assertEqual(result.actions, tuple(rows[1:]))
        self.assertIsNone(result.forced_reason)

    def test_two_recruit_combo_takes_maximum_available_without_changing_other_choices(self):
        state = SimpleNamespace(pending_card_option={})
        rows = [ai.Action(-1, 'card_option', {'card_id': 'fixture', 'choices': {
            'region': 'orthanc', 'unit_combo': c}}) for c in ('2regular', 'mixed')]
        result = policies.project_ai_actions(state, SH, rows)
        self.assertEqual(result.actions, (rows[1],))
        self.assertEqual(result.forced_reason, policies.RECRUIT_ELITE)

    def test_saruman_new_units_and_replacements_are_distinct_modes(self):
        state = SimpleNamespace(pending_card_option={})
        rows = [ai.Action(0, 'muster_recruit', {'nation': gs.Nation.ISENGARD,
                'region': 'orthanc', 'type': 'regular', 'saruman': 'settlements'}),
                ai.Action(0, 'muster_elite', {'nation': gs.Nation.ISENGARD,
                'region': 'orthanc', 'saruman': 'swap2'})]
        self.assertEqual(policies.project_ai_actions(state, SH, rows).actions, tuple(rows))

    def test_palantir_deck_choice_remains_learned_and_single_deck_is_forced(self):
        state = SimpleNamespace(pending_card_option={'palantir_draw': True})
        rows = [ai.Action(-1, 'card_option', {'card_id': '__palantir_draw__',
                'choices': {'palantir_deck': d}}) for d in ('character', 'strategy')]
        result = policies.project_ai_actions(state, SH, rows)
        self.assertEqual(result.actions, tuple(rows))
        self.assertIsNone(result.forced_reason)
        self.assertEqual(policies.project_ai_actions(state, SH, rows[:1]).forced_reason, policies.BONUS_DRAW)

    def test_hunt_redraw_and_ring_decisions_are_not_bonus_event_draws(self):
        state = SimpleNamespace(pending_card_option={})
        rows = [ai.Action(-1, 'card_option', {'choices': {key: take}})
                for key in ('mithril_redraw', 'balrog_declare_draw') for take in (True, False)]
        rows.append(ai.Action(-1, 'elven_ring', {'face': 'MUSTER'}))
        self.assertEqual(policies.project_ai_actions(state, FP, rows).actions, tuple(rows))
        self.assertIsNone(policies.project_ai_actions(state, FP, rows).forced_reason)

    def test_ai_entry_point_projects_full_menu(self):
        state = guards_pending()
        self.assertEqual([a.params['choices']['unit_type'] for a in ai.get_ai_actions(state, FP)], ['elite'])
        self.assertEqual(len(ai.get_legal_actions(state, FP)), 2)

    def test_forced_root_bypasses_policy_and_returns_no_training_distribution(self):
        state = guards_pending()
        with patch.object(ai, '_get_priors_and_value', side_effect=AssertionError('forced policy inference')):
            action, distribution = ai.mcts_search(state, FP, None, None, num_simulations=1)
        self.assertEqual(action.params['choices']['unit_type'], 'elite')
        self.assertIsNone(distribution)

    def test_forced_interior_expansion_has_one_deterministic_edge_without_policy_call(self):
        state = opening_for(CID, 'strategy')
        node = ai.MCTSNode(state=state, side=FP)
        with patch.object(ai, '_get_priors_and_value', side_effect=AssertionError('forced policy inference')):
            ai._expand_node(node, None, None, FP)
        self.assertEqual(len(node.children), 1)
        self.assertEqual(node.children[0].action.params['choices'], {'gandalf_draw': True})
        self.assertEqual(node.children[0].prior, 1.)
        self.assertFalse(node.is_terminal)
        self.assertIsNone(node.proven)

    def test_supplied_recruit_rows_without_display_fields_cannot_bypass_full_menu(self):
        state = guards_pending()
        legal = ai.get_legal_actions(state, FP)
        row = ai.Action(-1, 'card_option', {'card_id': CID,
            'choices': {'region': 'minas tirith', 'unit_type': 'regular'}})
        result = policies.project_ai_actions(state, FP, legal + [row])
        self.assertFalse(any(a is row for a in result.actions))

    def test_greedy_search_phase_obeys_fixed_projection_without_inference(self):
        state = guards_pending()
        with patch.object(ai, '_get_priors_and_value', side_effect=AssertionError('must not infer')):
            choice = ai._greedy_prior_action(state, FP, None, ai.get_legal_actions(state, FP))
        self.assertEqual(choice.params['choices']['unit_type'], 'elite')

    def test_real_episode_records_forced_steps_without_training_rows_or_reward_costs(self):
        state = house_before(face=gs.ActionDieFace.EVENT)
        # This fixture isolates fixed card policies; Gandalf is the sole guide candidate.
        state.fellowship.companions = [c for c in state.fellowship.companions
                                       if c.name == 'Gandalf the Grey']
        state.dice[FP] = state.dice[FP][:1]
        state.dice[SH] = []
        calls = []
        def choose(st, side, *args, **kwargs):
            self.assertIsNone(getattr(st, 'pending_card_option', None), 'Forced choice must bypass MCTS')
            self.assertEqual(side, FP)
            calls.append('play')
            return ai.Action(0, 'play_event', {'card_id': HOUSE, 'side': 'fp'}), None
        intervals = []
        original_draw = ai.game_logic.draw_event_card
        draws = []
        def draw(st, side, deck, *args, **kwargs):
            draws.append((side, deck))
            if len(draws) <= 4:
                return None  # fixture already supplies the literal opening hand
            return original_draw(st, side, deck, *args, **kwargs)
        def reward(pre, st, side, *, include_decision_cost=True):
            intervals.append((pre['hand'], len(st.hands[side].hand), include_decision_cost))
            return float(len(st.hands[side].hand) - pre['hand']), 0.
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': 'predecision_winloss_v1'}), \
                patch.object(ai.gs, 'setup_initial_game', return_value=state), \
                patch.object(ai.game_logic, 'initialize_decks'), \
                patch.object(ai.game_logic, 'draw_event_card', side_effect=draw), \
                patch.object(ai.game_logic, 'roll_action_dice'), \
                patch.object(ai, '_get_declare_actions', return_value=[]), \
                patch.object(ai, '_auto_enter_mordor', return_value=False), \
                patch.object(ai, '_balrog_stationary_turn_start', return_value=False), \
                patch.object(ai, '_safe_next_turn', return_value=True), \
                patch.object(ai, 'mcts_action_with_policy', side_effect=choose), \
                patch.object(ai, '_reward_stats', side_effect=lambda st: {'hand': len(st.hands[FP].hand)}), \
                patch.object(ai, '_compute_step_reward_split', side_effect=reward):
            result = ai._run_single_episode(None, None, {
                'max_turns': 1, 'mcts_sims': 1, 'trace_path': str(Path(tmp) / 'episode.trace.gz')})
            trace = game_trace.load_trace(str(next(Path(tmp).glob('*.trace.gz'))))
        rows = [r for r in trace['steps'] if r.get('kind') == 'action']
        forced = [r for r in rows if r.get('policy_forced')]
        self.assertEqual(calls, ['play'])
        self.assertEqual(len(result['states']), 1)
        self.assertEqual(result['rewards'].tolist(), [2.])
        self.assertEqual(intervals, [(1, 1, True), (1, 3, False)])
        self.assertEqual([r['forced_policy_reason'] for r in forced],
                         [policies.RECRUIT_ELITE, policies.BONUS_DRAW])
        self.assertEqual(trace['meta']['ai_policy_contract'], policies.POLICY_VERSION)
        self.assertEqual(result['forced_transitions'], {policies.RECRUIT_ELITE: 1, policies.BONUS_DRAW: 1})
        self.assertEqual(result['game_stats']['forced_transitions'], result['forced_transitions'])
        self.assertEqual(result['game_stats']['ai_policy_contract'], policies.POLICY_VERSION)
        import trace_admission
        admission = trace_admission.validate_trace(trace, require_decided=False, audit_rules=False)
        self.assertFalse(any(e['code'] == 'forced_policy_semantics' for e in admission['errors']))
        for row in forced:
            self.assertNotIn('search_stats', row)
            self.assertNotIn('regret', row)
            self.assertNotIn('value', row['decision_context'])
            self.assertIn('state', row['decision_context'])

    def test_search_executes_forced_recruit_and_draw_before_evaluating_next_choice(self):
        chosen, visits, applied, observed, chance_count = search_house_chain()
        self.assertEqual(chosen.params['card_id'], HOUSE)
        self.assertIn(('card_option', {'boromir_region': 'minas tirith', 'unit_type': 'elite'}), applied)
        self.assertIn(('card_option', {'gandalf_draw': True}), applied)
        self.assertTrue(any(hand == 3 and pending is None for hand, pending in observed))
        self.assertEqual(len(visits['action_encs']), 1)
        self.assertGreaterEqual(chance_count, 2)

    def test_batched_search_crosses_fixed_chance_chain_before_leaf_evaluation(self):
        chosen, visits, applied, observed, chance_count = search_house_chain(batch=4)
        self.assertEqual(chosen.params['card_id'], HOUSE)
        self.assertEqual([choice for typ, choice in applied if typ == 'card_option'], [
            {'boromir_region': 'minas tirith', 'unit_type': 'elite'}, {'gandalf_draw': True}])
        self.assertTrue(any(hand == 3 and pending is None for hand, pending in observed))
        self.assertGreaterEqual(chance_count, 2)

    def test_failed_fixed_search_transition_raises_instead_of_proving_a_win(self):
        with patch.object(ai, '_search_apply_fail', return_value=False), \
                self.assertRaisesRegex(ValueError, 'literal fixed transition refusal'):
            search_house_chain(refuse_fixed=True)

    def test_failed_fixed_episode_restores_threshold_and_saves_error_trace(self):
        state = guards_pending()
        pending = copy.deepcopy(state.pending_card_option)
        state.pending_card_option = None
        def reach_action_phase(st, *, pool):
            # This failure belongs to an action-phase fixed choice. Seeding it
            # before declaration instead exercises the earlier Fellowship tail.
            if pool == FP:
                st.pending_card_option = copy.deepcopy(pending)
        # This fixture isolates fixed card policies; Gandalf is the sole guide candidate.
        state.fellowship.companions = [c for c in state.fellowship.companions
                                       if c.name == 'Gandalf the Grey']
        state.dice[FP] = state.dice[FP][:1]
        state.dice[SH] = []
        original_threshold = ai.game_logic.SHADOW_VP_VICTORY
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': 'predecision_winloss_v1'}), \
                patch.object(ai.gs, 'setup_initial_game', return_value=state), \
                patch.object(ai.game_logic, 'initialize_decks'), \
                patch.object(ai.game_logic, 'draw_event_card'), \
                patch.object(ai.game_logic, 'roll_action_dice', side_effect=reach_action_phase), \
                patch.object(ai, '_get_declare_actions', return_value=[]), \
                patch.object(ai, '_auto_enter_mordor', return_value=False), \
                patch.object(ai, '_balrog_stationary_turn_start', return_value=False), \
                patch.object(ai, '_traced_execute', side_effect=ValueError('fixed apply refused')):
            path = str(Path(tmp) / 'failure.trace.gz')
            with self.assertRaisesRegex(ValueError, 'fixed apply refused'):
                ai._run_single_episode(None, None, {
                    'max_turns': 1, 'shadow_vp': 7, 'trace_path': path})
            trace = game_trace.load_trace(path)
        self.assertEqual(ai.game_logic.SHADOW_VP_VICTORY, original_threshold)
        self.assertIsNone(game_trace.ACTIVE_TRACER)
        self.assertEqual(trace['meta']['status'], 'error')
        self.assertTrue(any(e['code'] == 'forced_action_failed' for e in trace['meta']['source_errors']))


if __name__ == '__main__':
    unittest.main()
