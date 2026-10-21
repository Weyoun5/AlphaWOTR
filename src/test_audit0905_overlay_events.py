"""Opponent resampling must obey printed card icons and requirements.

Literal RB22 payment controls and CSV card requirements. The real overlay,
menu and action executor are used; only the sampled hidden hand is fixed.
"""
import unittest
from collections import Counter

import ai
import board
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
F = gs.ActionDieFace
GUARDS = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
DAY = 'FREE_PEOPLES_Strategy_Through_a_Day_and_a_Night'
FARAMIR = "FREE_PEOPLES_Strategy_Faramir's_Rangers"
SONG = "FREE_PEOPLES_Character_Bilbo's_Song"
CHALLENGE = 'FREE_PEOPLES_Character_Challenge_of_the_King'


def position(cards, faces):
    s = gs.setup_initial_game(board.load_board())
    boromir = next(c for c in s.fellowship.companions if 'Boromir' in c.name)
    s.fellowship.companions.remove(boromir)
    next(a for key, a in s.armies.items() if board.sanitize(key) == 'minas tirith').leaders.append(boromir)
    s.hands[FP].hand = list(cards)
    s.dice = {FP: [], SH: []}
    for face in faces:
        die = gs.ActionDie([face], FP)
        die.current = face
        s.dice[FP].append(die)
    # A fixed posterior is enough to exercise the real resampling path.
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        setattr(s, name, [])
    return s


def resample(state, cards):
    old_cards = list(state.hands[FP].hand)
    for deck in ('char', 'strat'):
        old = [c for c in old_cards if ai._ov_deck_of(c) == deck]
        new = [c for c in cards if ai._ov_deck_of(c) == deck]
        assert len(old) == len(new), 'Public per-deck hand counts must agree'
        setattr(state, ai._ov_deck_attr('fp', deck),
                list((Counter(new) - Counter(old)).elements()))
    node = ai.MCTSNode(state, FP)
    node.children = [ai.MCTSNode(None, FP, parent=node, action=a, prior=.1)
                     for a in ai.get_ai_actions(state, FP)]
    node.visit_count = 32
    overlay = ai._SimOverlay(state, SH)
    # Pin one valid posterior partition; patch/reconciliation remains real.
    for deck in ('char', 'strat'):
        old = [c for c in old_cards if ai._ov_deck_of(c) == deck]
        new = [c for c in cards if ai._ov_deck_of(c) == deck]
        overlay.hand[deck] = new
        overlay.deck[deck] = list((Counter(old) - Counter(new)).elements())
    ai._overlay_lazy_children(node, overlay)
    return node, overlay


def event_edges(node, cid):
    return [c.action for c in node.children if c.action.action_type == 'play_event'
            and c.action.params['card_id'] == cid]


class OverlayEventLegality(unittest.TestCase):
    def test_one_visit_sample_gets_held_event_without_old_card(self):
        # Literal stored card-only node: its previous hand offered Guards.
        # Changing the hand must add Day before the 16-visit threshold.
        node, overlay = resample(position([GUARDS], [F.EVENT]), [DAY])
        old = next(c for c in node.children if c.action.action_type == 'play_event'
                   and c.action.params['card_id'] == GUARDS)
        node.children = [old]
        node.visit_count = 1
        node.ov_seen = set()
        ai._overlay_lazy_children(node, overlay)
        self.assertEqual([a.die_idx for a in event_edges(node, DAY)], [0])
        self.assertFalse(ai._ov_child_ok(old, node, overlay))
        self.assertNotIn(GUARDS, node.state.hands[FP].hand)

    def test_army_card_cannot_borrow_muster_template(self):
        node, _ = resample(position([GUARDS], [F.MUSTER]), [DAY])
        self.assertEqual(event_edges(node, DAY), [])

    def test_sampled_card_uses_its_own_matching_die(self):
        node, _ = resample(position([GUARDS], [F.MUSTER, F.ARMY]), [DAY])
        actions = event_edges(node, DAY)
        self.assertEqual([a.die_idx for a in actions], [1])
        result = ai.apply_action(node.state, actions[0])
        self.assertFalse(result.dice[FP][0].used)
        self.assertTrue(result.dice[FP][1].used)
        self.assertEqual(result.pending_card_option['card_id'], DAY)

    def test_all_distinct_legal_funding_faces_remain_available(self):
        node, _ = resample(position([GUARDS], [F.MUSTER, F.ARMY, F.EVENT]), [DAY])
        self.assertEqual({a.die_idx for a in event_edges(node, DAY)}, {1, 2})

    def test_faramir_without_target_is_not_a_zero_effect_play(self):
        state = position([GUARDS], [F.EVENT])
        state.armies = {key: army for key, army in state.armies.items()
                        if board.sanitize(key) not in ('osgiliath', 'south ithilien', 'north ithilien')}
        node, _ = resample(state, [FARAMIR])
        self.assertEqual(event_edges(node, FARAMIR), [])

    def test_new_deck_needs_no_same_deck_template(self):
        node, _ = resample(position([GUARDS, CHALLENGE], [F.EVENT]), [GUARDS, SONG])
        actions = event_edges(node, SONG)
        self.assertEqual([a.die_idx for a in actions], [0])
        result = ai.apply_action(node.state, actions[0])
        self.assertEqual(result.fellowship.corruption, 0)
        self.assertTrue(result.dice[FP][0].used)
        self.assertNotIn(SONG, result.hands[FP].hand)

    def test_initial_hand_without_playable_event_does_not_close_window(self):
        node, _ = resample(position([DAY], [F.MUSTER]), [GUARDS])
        self.assertEqual([a.die_idx for a in event_edges(node, GUARDS)], [0])

    def test_repeated_sample_does_not_duplicate_children(self):
        node, overlay = resample(position([GUARDS], [F.EVENT]), [DAY])
        self.assertEqual(len(event_edges(node, DAY)), 1)
        ai._overlay_lazy_children(node, overlay)
        self.assertEqual(len(event_edges(node, DAY)), 1)

    def test_absent_old_card_remains_masked(self):
        node, overlay = resample(position([GUARDS], [F.EVENT]), [DAY])
        old = next(c for c in node.children if c.action.action_type == 'play_event'
                   and c.action.params['card_id'] == GUARDS)
        self.assertFalse(ai._ov_child_ok(old, node, overlay))

    def test_used_matching_die_is_not_reused(self):
        state = position([GUARDS], [F.MUSTER, F.ARMY])
        state.dice[FP][1].used = True
        node, _ = resample(state, [DAY])
        self.assertEqual(event_edges(node, DAY), [])

    def test_ring_payment_keeps_exact_physical_die(self):
        state = position([GUARDS], [F.MUSTER, F.CHARACTER])
        state = ai.apply_action(state, ai.Action(-1, 'elven_ring', {
            'target_die_idx': 1, 'new_face': F.MUSTER_ARMY.value, 'side': 'fp'}))
        self.assertEqual(state.fp_elven_rings, 2)
        node, _ = resample(state, [DAY])
        self.assertEqual([a.die_idx for a in event_edges(node, DAY)], [1])

    def test_compulsory_guide_pending_cannot_open_event_window(self):
        state = position([DAY], [F.EVENT])
        state.pending_guide_change = {'candidates': ['Gandalf the Grey', 'Strider']}
        node, _ = resample(state, [GUARDS])
        self.assertEqual(event_edges(node, GUARDS), [])


if __name__ == '__main__':
    unittest.main()
