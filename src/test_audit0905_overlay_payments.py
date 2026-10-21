"""Hidden-hand payment controls for removing A Power too Great.

Printed payment is one Army-type card plus one Character card. A Strategy
Muster card is not an Army card. Use real posterior partition reconciliation.
"""
from collections import Counter
import os
import unittest
from unittest.mock import patch

import ai
import board
import cards
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
ARMY = 'SHADOW_Strategy_Return_to_Valinor'
ARMY2 = 'SHADOW_Strategy_The_Fighting_Uruk-hai'
MUSTER = 'SHADOW_Strategy_Stormcrow'
CHAR = 'SHADOW_Character_Balrog_of_Moria'
CHAR2 = 'SHADOW_Character_Dreadful_Spells'
TABLE = 'FREE_PEOPLES_Strategy_A_Power_too_Great'


def position(hand):
    for cid in hand:
        assert cards.get_card(cid) is not None, cid
    state = gs.setup_initial_game(board.load_board())
    state.hands[SH].hand = list(hand)
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        setattr(state, name, [])
    die = gs.ActionDie([gs.ActionDieFace.ARMY], SH)
    die.current = gs.ActionDieFace.ARMY
    state.dice = {SH: [die], FP: []}
    state.persistent_cards.add_card(gs.PersistentCard(card_id=TABLE,
        card_name='A Power too Great', side=FP, card_type=gs.PersistentCardType.POWER_TOO_GREAT))
    return state


def sampled_node(state, hand):
    for cid in hand:
        assert cards.get_card(cid) is not None, cid
    old_hand = list(state.hands[SH].hand)
    for deck in ('char', 'strat'):
        old = [c for c in old_hand if ai._ov_deck_of(c) == deck]
        new = [c for c in hand if ai._ov_deck_of(c) == deck]
        assert len(old) == len(new)
        setattr(state, ai._ov_deck_attr('sh', deck), list((Counter(new)-Counter(old)).elements()))
    node = ai.MCTSNode(state, SH)
    node.children = [ai.MCTSNode(None, SH, parent=node, action=a, prior=.1)
                     for a in ai.get_ai_actions(state, SH)]
    node.visit_count = 32
    ov = ai._SimOverlay(state, FP)
    for deck in ('char', 'strat'):
        old = [c for c in old_hand if ai._ov_deck_of(c) == deck]
        new = [c for c in hand if ai._ov_deck_of(c) == deck]
        ov.hand[deck] = new
        ov.deck[deck] = list((Counter(old)-Counter(new)).elements())
    kind = ai._overlay_lazy_children(node, ov)
    return node, ov, kind


class OverlayPayments(unittest.TestCase):
    def test_muster_sample_masks_previously_affordable_table_removal(self):
        node, ov, _ = sampled_node(position([ARMY, CHAR]), [MUSTER, CHAR])
        edge = next(c for c in node.children if c.action.action_type == 'discard_table_card')
        self.assertFalse(ai._ov_child_ok(edge, node, ov))
        self.assertTrue(ai._is_hidden_hand_card_edge(edge.action))

    def test_army_sample_adds_newly_affordable_table_removal(self):
        node, ov, _ = sampled_node(position([MUSTER, CHAR]), [ARMY, CHAR])
        edges = [c for c in node.children if c.action.action_type == 'discard_table_card']
        self.assertEqual(len(edges), 1)
        self.assertTrue(ai._ov_child_ok(edges[0], node, ov))
        result = ai.apply_action(node.state, edges[0].action)
        self.assertEqual(result.pending_card_option['options'], [{'army_card': ARMY, 'character_card': CHAR}])

    def test_pending_payment_tracks_sampled_cards_not_old_pair(self):
        state = position([ARMY, CHAR])
        state = ai.apply_action(state, ai.Action(0, 'discard_table_card', {
            'card_id': TABLE, 'card_name': 'A Power too Great'}))
        node, ov, kind = sampled_node(state, [ARMY2, CHAR2])
        self.assertNotEqual(kind, 'none')
        compatible = [c for c in node.children if ai._ov_child_ok(c, node, ov)]
        self.assertEqual([c.action.params['choices'] for c in compatible],
                         [{'army_card': ARMY2, 'character_card': CHAR2}])
        self.assertTrue(ai._is_hidden_hand_card_edge(compatible[0].action))
        after = ai.apply_action(node.state, compatible[0].action)
        self.assertEqual(after.hands[SH].hand, [])
        self.assertEqual(set(after.hands[SH].discarded_face_down), {ARMY2, CHAR2})
        self.assertIsNone(after.pending_card_option)

    def test_unaffordable_table_removal_with_empty_ordinary_menu(self):
        state = position([ARMY, CHAR])
        state.armies = {}  # No army move can use the remaining Army die.
        node, ov, _ = sampled_node(state, [MUSTER, CHAR])
        self.assertEqual(ai.get_ai_actions(node.state, SH), [])
        before = list(node.children)
        with patch.object(ai, '_SIM_OVERLAY', ov):
            self.assertIsNone(ai._select_child(node, 1.0))
        self.assertEqual(node.children, before)
        self.assertEqual(node.side, SH)
        self.assertIsNone(node.proven)

    def test_impossible_payment_sample_cannot_fall_back_to_absent_cards(self):
        state = position([ARMY, CHAR])
        state = ai.apply_action(state, ai.Action(0, 'discard_table_card', {
            'card_id': TABLE, 'card_name': 'A Power too Great'}))
        node, ov, _ = sampled_node(state, [MUSTER, CHAR])
        with patch.object(ai, '_SIM_OVERLAY', ov):
            with self.assertRaisesRegex(RuntimeError, 'No compatible action'):
                ai._select_child(node, 1.0)

    def test_sample_dependent_payment_cannot_prove_opponent_win(self):
        node, ov, _ = sampled_node(position([ARMY, CHAR]), [ARMY, CHAR])
        edge = next(c for c in node.children if c.action.action_type == 'discard_table_card')
        node.children = [edge]
        edge.proven, edge.proven_value = 'L', -1.
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_winloss_v1'):
            ai._propagate_proofs([node], FP)
        self.assertIsNone(node.proven)

    def test_failed_hand_reconciliation_cannot_disable_legality_mask(self):
        node, ov, _ = sampled_node(position([ARMY, CHAR]), [ARMY, CHAR])
        with patch.object(ai, '_SIM_OVERLAY', ov), \
                patch.object(ai, 'OVERLAY_ASSERT', False), \
                patch.object(ai, '_overlay_lazy_children', side_effect=ValueError('sample reconciliation refused')):
            with self.assertRaisesRegex(ValueError, 'sample reconciliation refused'):
                ai._select_child(node, 1.0)


if __name__ == '__main__':
    unittest.main()
