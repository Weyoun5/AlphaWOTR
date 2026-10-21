"""Case 681: two paid Regulars reduce six units to four combat dice.

Literal card oracle: Relentless Assault applies its losses before rolling;
each paid hit changes faces by +1, not the number of dice. The five recorded
faces are preserved as an inconsistent tape, never truncated or relabelled.
"""
import copy
import unittest
from collections import Counter
from unittest.mock import patch
from test_combat_continuation_literal import live_battle, minion, card_id, cursor, gs, cb
import rendered_extract as rx


def payment_position():
    st, a, d = live_battle(fp_attacks=False, fp_regulars=1, sh_regulars=6,
        shadow_nation=gs.Nation.SAURON,
        sh_leaders=[minion('Witch King')] + [gs.Nazgul(gs.Nation.SAURON) for _ in range(3)])
    with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Premature dice')):
        r = cb.resolve_combat(st, a, d, is_siege=True,
            attacker_card_id=card_id('Relentless Assault'),
            choices={'defer_round_choices': True, 'defer_casualties': True})
        assert r.round_pending['effect'] == 'relentless_assault'
        r = cb.resume_combat(st, r.round_resume, {'count': 2})
        r = cb.resume_combat(st, r.round_resume,
            {'nation': 'SAURON', 'kind': 'regular', 'operation': 'remove'})
    return st, r


class RecordedRelentlessDice(unittest.TestCase):
    def test_two_regular_payments_leave_four_dice_and_plus_two_faces(self):
        st, r = payment_position()
        self.assertEqual(len(cursor(r).combat.attacker_army.units), 5)
        self.assertEqual(r.round_pending['options'],
            [{'nation': 'SAURON', 'kind': 'regular', 'operation': 'remove'}])
        calls = []
        def literal_dice(n, ctx='', side=None):
            calls.append((n, ctx, side))
            if ctx == 'combat roll' and side == gs.Side.SHADOW:
                self.assertEqual(n, 4)
                return [4, 4, 4, 4]
            if ctx == 'combat roll' and side == gs.Side.FREE_PEOPLES:
                self.assertEqual(n, 1)
                return [1]
            self.fail('Unexpected reroll or extra dice')
        with patch.object(cb, 'roll_d6_batch', side_effect=literal_dice):
            done = cb.resume_combat(st, r.round_resume,
                {'nation': 'SAURON', 'kind': 'regular', 'operation': 'remove'})
        self.assertEqual([n for n, _, _ in calls], [4, 1])
        self.assertEqual(done.deferred_hits_on_defender, 4)
        self.assertEqual(done.deferred_hits_on_attacker, 0)

    def test_actual_five_face_record_refuses_without_consuming_or_truncating(self):
        st, r = payment_position()
        rounds = [[({'ints': [6, 1, 5, 2, 6], 'line': 615}, 'A', 'main'),
                   ({'ints': [5], 'line': 616}, 'D', 'main')]]
        original = copy.deepcopy(rounds)
        tape = rx._RecordedCombatDice(rounds, gs.Side.SHADOW, {'detected': Counter()}, allow_special=True)
        with patch.object(cb, 'roll_d6_batch', side_effect=tape.draw):
            with self.assertRaisesRegex(rx._RecordedCombatDiceError, '^main_count$'):
                cb.resume_combat(st, r.round_resume,
                    {'nation': 'SAURON', 'kind': 'regular', 'operation': 'remove'})
        self.assertEqual(rounds, original)
        self.assertEqual(tape.seen, set())
        self.assertEqual(tape.index, 0)


if __name__ == '__main__':
    unittest.main()
