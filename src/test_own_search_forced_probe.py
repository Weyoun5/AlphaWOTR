"""Reply leaf closure must not serialize unused ordinary action menus."""
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
import random
import unittest

import ai
import ai_policies
import board
import game_state as gs
import own_search
from reply_evaluation import menu, transition_rng


def reference_value(context, state, previous_actor):
    """The pre-optimization closure, retained as an independent result oracle."""
    for _ in range(128):
        result = context.ai.game_logic.check_all_victory_conditions(state)
        if (getattr(state, '_search_endturn_victory', None) is not None
                or context.ai.game_logic.should_game_end(result, end_of_turn=False)):
            break
        actor = context.ai._next_acting_side(state, previous_actor)
        if actor is None:
            break
        actions, forced = menu(context.ai, state, actor)
        if not forced:
            break
        if len(actions) != 1:
            raise RuntimeError('Fixed policy requires exactly one action')
        action = next(iter(actions.values()))
        successor = context.ai.apply_action(state, action)
        context.ai._resume_preaction_for_search(state, successor, action)
        state = successor
        previous_actor = actor
        context.stats['forced_reply_transitions'] += 1
    else:
        raise RuntimeError('Reply fixed-policy closure exceeded 128 transitions')
    context.stats['reply_value_evaluations'] += 1
    return own_search.root_value(context.ai, state, context.side, context.net)


def context(engine=ai):
    value = object.__new__(own_search.SearchContext)
    value.ai = engine
    value.side = gs.Side.FREE_PEOPLES
    value.net = None
    value.stats = Counter()
    return value


def army_state():
    state = gs.setup_initial_game(board.load_board())
    ai.game_logic.initialize_decks(state)
    for nation in state.political.boxes:
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    faces = [gs.ActionDieFace.CHARACTER, gs.ActionDieFace.MUSTER_ARMY,
             gs.ActionDieFace.EVENT, gs.ActionDieFace.WILL_OF_WEST]
    ai.game_logic.adopt_rolled_faces(state, [(gs.Side.FREE_PEOPLES, i, face)
                                           for i, face in enumerate(faces)])
    return state


class ForcedReplyProbeTests(unittest.TestCase):
    def stub(self, policies, actor=True, terminal=False):
        selected = []
        engine = SimpleNamespace(
            game_logic=SimpleNamespace(check_all_victory_conditions=lambda s: terminal,
                                       should_game_end=lambda result, **kw: result),
            _next_acting_side=lambda s, previous: 'actor' if actor else None,
            get_ai_policy=lambda s, side: policies[s.step],
            _resume_preaction_for_search=lambda *args: None)
        def apply(state, action):
            selected.append(action)
            return SimpleNamespace(step=state.step + 1)
        engine.apply_action = apply
        return engine, selected

    def test_ordinary_terminal_and_no_actor_preserve_values_without_keys(self):
        normal = ai.Action(0, 'army_move', {'nested': [[1, 2], {'path': ['a', 'b']}]})
        for actions in ((), (normal,), (normal,) * 500):
            for terminal, actor in ((False, True), (True, True), (False, False)):
                with self.subTest(size=len(actions), terminal=terminal, actor=actor):
                    engine, selected = self.stub([ai_policies.PolicyProjection(actions)], actor, terminal)
                    expected = context(engine)
                    actual = context(engine)
                    with patch.object(own_search, 'root_value', return_value=0.37):
                        before = reference_value(expected, SimpleNamespace(step=0), None)
                        with patch.object(ai_policies, 'action_policy_key', side_effect=AssertionError('unused key')):
                            after = actual.value(SimpleNamespace(step=0), None)
                    self.assertEqual(before, after)
                    self.assertEqual(expected.stats, actual.stats)
                    self.assertEqual(selected, [])

    def test_forced_duplicates_keep_first_action_and_reject_distinct_choices(self):
        first = ai.Action(-1, 'card_option', {'card_id': 'x', 'choices': {'take': True}})
        same = ai.Action(-1, 'card_option', {'card_id': 'x', 'choices': {'take': True}, 'card_name': 'label'})
        other = ai.Action(-1, 'card_option', {'card_id': 'x', 'choices': {'take': False}})
        end = ai_policies.PolicyProjection(())
        for actions, invalid in (((first,), False), ((first, same), False), ((first, other), True), ((), True)):
            with self.subTest(size=len(actions), invalid=invalid):
                engine, selected = self.stub([ai_policies.PolicyProjection(actions, 'forced'), end])
                with patch.object(own_search, 'root_value', return_value=0.11):
                    for evaluate in (reference_value, own_search.SearchContext.value):
                        if invalid:
                            with self.assertRaisesRegex(RuntimeError, 'exactly one action'):
                                evaluate(context(engine), SimpleNamespace(step=0), None)
                        else:
                            self.assertEqual(evaluate(context(engine), SimpleNamespace(step=0), None), 0.11)
                if not invalid:
                    self.assertEqual(len(selected), 2)
                    self.assertTrue(all(action is first for action in selected))

    def test_real_army_menu_and_seeded_gandalf_closure_match_reference(self):
        ordinary = army_state()
        projected = ai.get_ai_policy(ordinary, gs.Side.FREE_PEOPLES)
        self.assertFalse(projected.forced_reason)
        self.assertTrue(any(a.action_type == 'army_move' for a in projected.actions))
        forced = ai.clone_state(ordinary)
        forced.pending_card_option = {
            'card_id': '__gandalf_draw__', 'card_name': 'Gandalf guide draw',
            'side': 'fp', 'decider': 'fp', 'direct': True, 'gandalf_draw': True,
            'source_card_id': 'FREE_PEOPLES_Character_House_of_the_Stewards',
            'deck': 'character', 'die_face': 'EVENT',
            'options': [{'gandalf_draw': True}, {'gandalf_draw': False}]}
        def score(engine, state, side, net):
            return (tuple(state.hands[side].hand), state.pending_card_option,
                    random.random(), float(ai.np.random.random()))
        original_apply = ai.apply_action
        for state in (ordinary, forced):
            for seed in (17, 941):
                results = []
                for evaluate in (reference_value, own_search.SearchContext.value):
                    selected = []
                    def apply(s, action):
                        selected.append(ai_policies.action_policy_key(action))
                        return original_apply(s, action)
                    ctx = context()
                    with transition_rng(ai, seed), patch.object(own_search, 'root_value', score), patch.object(ai, 'apply_action', apply):
                        result = evaluate(ctx, ai.clone_state(state), gs.Side.SHADOW)
                        results.append((result, selected, dict(ctx.stats)))
                self.assertEqual(*results)
                self.assertEqual(len(results[0][1]), int(state is forced))


if __name__ == '__main__':
    unittest.main(verbosity=2)
