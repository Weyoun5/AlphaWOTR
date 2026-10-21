"""Literal combat-card rules exercised through the live battle continuation.

Oracles: supplied Almanac pp37,39,40,44 and card_list.csv. Prescribed physical
d6 batches determine hits; neither the analytic kernel nor an earlier engine
version supplies expected results. Run with python -B this_file.py.
WOTR_TEST_COMBAT_SOURCE may point at an isolated candidate for preintegration.
"""
from pathlib import Path
import copy
import os
import pickle
import random
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('WOTR_TEST_COMBAT_SOURCE', str(HERE))).resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(SOURCE), str(HERE)]
os.environ.update(WOTR_NATIVE='0', WOTR_COMBAT_TABLE='', CUDA_VISIBLE_DEVICES='-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v51')
import board
import combat as cb
import game_logic as gl
import game_state as gs
from cards import get_cards


def card_id(name):
    return next(k for k, v in get_cards().items() if v.combat_name == name)


def companion(name='Meriadoc Brandybuck', level=1, leadership=1):
    return gs.Companion(name, level, leadership)


def minion(name):
    return gl.Minion(gs.Nation.SAURON, name)


def live_battle(fp_attacks=True, fp_regulars=3, fp_elites=0,
                sh_regulars=3, sh_elites=0, fp_leaders=(), sh_leaders=(),
                shadow_nation=gs.Nation.ISENGARD):
    state = gs.setup_initial_game(board.load_board())
    for nation in state.political.boxes:
        state.political.boxes[nation] = 3
    fp = gs.Army('fords of isen',
        [gs.Unit(gs.Nation.ROHAN, gs.UnitKind.REGULAR) for _ in range(fp_regulars)] +
        [gs.Unit(gs.Nation.ROHAN, gs.UnitKind.ELITE) for _ in range(fp_elites)],
        list(fp_leaders))
    sh = gs.Army('orthanc',
        [gs.Unit(shadow_nation, gs.UnitKind.REGULAR) for _ in range(sh_regulars)] +
        [gs.Unit(shadow_nation, gs.UnitKind.ELITE) for _ in range(sh_elites)],
        list(sh_leaders))
    state.armies = {fp.region: fp, sh.region: sh}
    state.witch_king.in_play = any(getattr(l, 'name', '') == 'Witch King' for l in sh.leaders)
    state.mouth_of_sauron.in_play = any(getattr(l, 'name', '') == 'Mouth of Sauron' for l in sh.leaders)
    a, d = (fp.region, sh.region) if fp_attacks else (sh.region, fp.region)
    return state, a, d


class Dice:
    """An exhaustive script of physical rolls: wrong count or extra draw fails."""
    def __init__(self, *batches):
        self.batches = [list(x) for x in batches]
        self.calls = []

    def __call__(self, n, ctx='', side=None):
        if not self.batches:
            raise AssertionError('Unexpected additional random batch')
        batch = self.batches.pop(0)
        if len(batch) != n:
            raise AssertionError(f'Expected {len(batch)} physical dice, engine requested {n}')
        self.calls.append((n, str(side), tuple(batch)))
        return batch


def begin(state, a, d, fp_card=None, sh_card=None, fp_attacks=True):
    return cb.resolve_combat(state, a, d,
        attacker_card_id=card_id(fp_card if fp_attacks else sh_card) if (fp_card if fp_attacks else sh_card) else None,
        defender_card_id=card_id(sh_card if fp_attacks else fp_card) if (sh_card if fp_attacks else fp_card) else None,
        choices={'defer_round_choices': True, 'defer_casualties': True})


def cursor(result):
    return result.round_resume['cursor']


def names(army):
    return [getattr(x, 'name', 'nazgul' if isinstance(x, gs.Nazgul) else 'leader')
            for x in army.leaders]


class HitConsumption(unittest.TestCase):
    def test_blade_uses_one_hit_for_minion_but_fateful_is_additional(self):
        # These helper-level controls also run against the preserved old source.
        for name, expected in [('Blade of Westernesse', 0), ('Fateful Strike', 1)]:
            with self.subTest(card=name):
                sh_leaders = [minion('Mouth of Sauron')] if name.startswith('Blade') else [gs.Nazgul(gs.Nation.SAURON)]
                st, a, d = live_battle(fp_leaders=[companion()], sh_leaders=sh_leaders)
                c = cb.setup_combat(st, a, d)
                c.attacker_card = get_cards()[card_id(name)].to_combat_card()
                c.attacker_hits = 1
                if name.startswith('Blade'):
                    cb.apply_blade_of_westernesse(c, 1, True, {'atk_effect_target': 'Mouth of Sauron'})
                else:
                    cb.apply_fateful_strike(c, 1, True)
                self.assertEqual(c.defender_army.leaders, [])
                self.assertEqual(c.attacker_hits, expected)
                self.assertEqual(len(c.defender_army.units), 3)

    def test_blade_without_participating_minion_keeps_army_hit(self):
        st, a, d = live_battle(fp_leaders=[companion()], sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
        c = cb.setup_combat(st, a, d)
        c.attacker_card = get_cards()[card_id('Blade of Westernesse')].to_combat_card()
        c.attacker_hits = 1
        cb.apply_blade_of_westernesse(c, 1, True)
        self.assertEqual(c.attacker_hits, 1)
        self.assertEqual(len(c.defender_army.leaders), 1)


class LiveContinuation(unittest.TestCase):
    def assert_pending(self, result, effect, owner, options):
        self.assertTrue(result.round_paused)
        self.assertEqual(result.round_pending['effect'], effect)
        self.assertEqual(result.round_pending['owner'], owner)
        self.assertEqual(result.round_pending['options'], options)
        self.assertEqual(result.rounds_fought, 0)

    def test_blade_target_compulsory_and_consumes_hit_only_after_mumakil(self):
        # Both sides score one ordinary+reroll hit. Mumakil is NOT ahead at
        # initiative5. Blade redirects FP's hit later; it must not manufacture
        # a retrospective Mumakil bonus.
        st, a, d = live_battle(fp_regulars=1, sh_regulars=0, sh_elites=1,
            fp_leaders=[companion()], sh_leaders=[minion('Witch King'), minion('Mouth of Sauron')],
            shadow_nation=gs.Nation.SOUTHRON_EASTERLINGS)
        rolls = Dice([1], [6], [6])
        with patch.object(cb, 'roll_d6_batch', rolls):
            r = begin(st, a, d, 'Blade of Westernesse', 'Mûmakil')
        self.assertEqual(rolls.batches, [])
        self.assert_pending(r, 'blade_of_westernesse', 'atk',
            [{'target': 'Witch King'}, {'target': 'Mouth of Sauron'}])
        self.assertEqual(cursor(r).combat.defender_bonus_hits, 0)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Earlier rolls repeated')):
            done = cb.resume_combat(st, r.round_resume, {'target': 'Mouth of Sauron'})
        self.assertFalse(done.round_paused)
        self.assertEqual((done.deferred_hits_on_defender, done.deferred_hits_on_attacker), (0, 1))
        self.assertEqual(names(st.armies['orthanc']), ['Witch King'])
        self.assertFalse(st.mouth_of_sauron.in_play)
        self.assertTrue(st.witch_king.in_play)
        self.assertEqual(done.rounds_fought, 1)

    def test_fateful_actual_two_hits_keep_army_damage_and_choose_victim(self):
        st, a, d = live_battle(fp_leaders=[gs.Leader(gs.Nation.ROHAN) for _ in range(3)],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON), minion('Witch King'), minion('Mouth of Sauron')])
        # Attacker normal, attacker re-roll, then defender normal.
        rolls = Dice([1, 1, 1], [6, 6, 1], [6, 6, 6])
        with patch.object(cb, 'roll_d6_batch', rolls):
            r = begin(st, a, d, 'Fateful Strike')
        self.assert_pending(r, 'fateful_strike', 'atk',
            [{'target': 'nazgul'}, {'target': 'Witch King'}, {'target': 'Mouth of Sauron'}])
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Rerolled a completed phase')):
            done = cb.resume_combat(st, r.round_resume, {'target': 'Mouth of Sauron'})
        self.assertEqual(done.deferred_hits_on_defender, 2)
        self.assertEqual(names(st.armies['orthanc']), ['nazgul', 'Witch King'])

    def test_fateful_one_hit_excludes_minions_and_bad_choice_is_atomic(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=1,
            fp_leaders=[gs.Leader(gs.Nation.ROHAN)],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON), minion('Witch King')])
        with patch.object(cb, 'roll_d6_batch', Dice([1], [6], [6])):
            r = begin(st, a, d, 'Fateful Strike')
        self.assert_pending(r, 'fateful_strike', 'atk', [{'target': 'nazgul'}])
        original = pickle.dumps((st, r.round_resume))
        rng = random.getstate()
        with self.assertRaises(ValueError):
            cb.resume_combat(st, r.round_resume, {'target': 'Witch King'})
        self.assertEqual(pickle.dumps((st, r.round_resume)), original)
        self.assertEqual(random.getstate(), rng)

    def test_mighty_pauses_after_combat_and_clone_resume_does_not_repeat_rolls(self):
        st, a, d = live_battle(fp_regulars=4,
            fp_leaders=[companion('Aragorn', 3, 2), companion(), gs.Leader(gs.Nation.ROHAN)])
        first = Dice([6, 1, 1, 1, 1])
        with patch.object(cb, 'roll_d6_batch', first):
            r = begin(st, a, d, 'Mighty Attack')
        self.assert_pending(r, 'mighty_attack', 'atk',
            [{'target': 'Aragorn'}, {'target': 'Meriadoc Brandybuck'}])
        self.assertEqual(cursor(r).combat.attacker_rerolls, [])
        self.assertEqual(cursor(r).combat.defender_combat_rolls, [])
        self.assertNotIn(('combat','defender'),cursor(r).observed_roll_sides)
        self.assertEqual(first.batches, [])
        original = pickle.dumps((st, r.round_resume))
        for copier in (copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))):
            for payer, n in [('Aragorn', 2), ('Meriadoc Brandybuck', 3)]:
                with self.subTest(copy=copier.__name__, payer=payer):
                    clone, payload = copier((st, r.round_resume))
                    tail = Dice([6] * n, [6, 6, 6])
                    with patch.object(cb, 'roll_d6_batch', tail):
                        done = cb.resume_combat(clone, payload, {'target': payer})
                    self.assertEqual(tail.batches, [])
                    self.assertEqual(done.deferred_hits_on_defender, 2 + n)
                    self.assertEqual(done.rounds_fought, 1)
                    self.assertEqual(names(clone.armies['fords of isen']),
                        ['Aragorn', 'Meriadoc Brandybuck', 'leader'])
                    self.assertEqual(pickle.dumps((st, r.round_resume)), original)

    def test_relentless_capacity_follows_charge_and_fullwipe_has_no_combat_roll(self):
        st, a, d = live_battle(fp_regulars=0, fp_elites=1, sh_regulars=2,
            sh_leaders=[minion('Witch King'), gs.Nazgul(gs.Nation.SAURON)])
        with patch.object(cb, 'roll_d6_batch', Dice([6])):
            r = begin(st, a, d, 'Charge', 'Relentless Assault')
        self.assert_pending(r, 'pre_combat_casualty', 'def',
            [{'nation': 'ISENGARD', 'kind': 'regular', 'operation': 'remove'}])
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll before initiative3')):
            r = cb.resume_combat(st, r.round_resume,
                {'nation': 'ISENGARD', 'kind': 'regular', 'operation': 'remove'})
        self.assert_pending(r, 'relentless_assault', 'def', [{'count': 0}, {'count': 1}])
        self.assertEqual(len(cursor(r).combat.defender_army.units), 1)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Combat roll after full wipe')):
            payment = cb.resume_combat(st, r.round_resume, {'count': 1})
            self.assert_pending(payment, 'relentless_assault_payment', 'def',
                [{'nation': 'ISENGARD', 'kind': 'regular', 'operation': 'remove'}])
            done = cb.resume_combat(st, payment.round_resume,
                {'nation': 'ISENGARD', 'kind': 'regular', 'operation': 'remove'})
        self.assertTrue(done.defender_destroyed)
        self.assertFalse(st.witch_king.in_play)
        self.assertNotIn('orthanc', st.armies)
        self.assertEqual(len(st.armies['fords of isen'].units), 1)

    def test_relentless_decline_keeps_lone_elite_and_then_rolls(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=0, sh_elites=1)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll before initiative3 choice')):
            r = begin(st, a, d, sh_card='Relentless Assault')
        self.assert_pending(r, 'relentless_assault', 'def', [{'count': 0}, {'count': 1}, {'count': 2}])
        with patch.object(cb, 'roll_d6_batch', Dice([1], [1])):
            done = cb.resume_combat(st, r.round_resume, {'count': 0})
        self.assertEqual([u.kind for u in st.armies['orthanc'].units], [gs.UnitKind.ELITE])
        self.assertEqual((done.deferred_hits_on_defender, done.deferred_hits_on_attacker), (0, 0))

    def test_heroic_named_hobbit_and_decline_are_distinct_after_actual_hits(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=1,
            fp_leaders=[gs.Leader(gs.Nation.ROHAN), companion()])
        with patch.object(cb, 'roll_d6_batch', Dice([6], [6])):
            r = begin(st, a, d, 'Heroic Death')
        self.assert_pending(r, 'heroic_death', 'atk',
            [{'count': 0}, {'count': 1, 'target': 'leader'}, {'count': 1, 'target': 'Meriadoc Brandybuck'}])
        for choice, remaining, hits in [({'count': 0}, ['leader', 'Meriadoc Brandybuck'], 1),
                ({'count': 1, 'target': 'Meriadoc Brandybuck'}, ['leader'], 0)]:
            clone, payload = pickle.loads(pickle.dumps((st, r.round_resume)))
            with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll during sacrifice')):
                done = cb.resume_combat(clone, payload, choice)
            self.assertEqual(names(clone.armies['fords of isen']), remaining)
            self.assertEqual(done.deferred_hits_on_attacker, hits)

    def test_black_breath_optional_decline_then_heroic_uses_surviving_figures(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=2,
            fp_leaders=[companion(), gs.Leader(gs.Nation.ROHAN)],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
        with patch.object(cb, 'roll_d6_batch', Dice([6], [1, 1], [6])):
            r = begin(st, a, d, 'Heroic Death', 'Black Breath')
        self.assert_pending(r, 'black_breath', 'def',
            [{'count': 0}, {'target': 'Meriadoc Brandybuck'}, {'target': 'leader'}])
        original = pickle.dumps((st, r.round_resume))
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Repeated die roll')):
            r2 = cb.resume_combat(st, r.round_resume, {'target': 'leader'})
            self.assert_pending(r2, 'heroic_death', 'atk',
                [{'count': 0}, {'count': 1, 'target': 'Meriadoc Brandybuck'}])
            done = cb.resume_combat(st, r2.round_resume, {'count': 1, 'target': 'Meriadoc Brandybuck'})
        self.assertEqual(st.armies['fords of isen'].leaders, [])
        self.assertEqual(done.deferred_hits_on_attacker, 0)
        clone, payload = pickle.loads(original)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Repeated roll on decline')):
            declined = cb.resume_combat(clone, payload, {'count': 0})
        self.assertEqual(declined.round_pending['effect'], 'heroic_death')
        self.assertEqual(names(clone.armies['fords of isen']), ['Meriadoc Brandybuck', 'leader'])

    def test_public_action_menu_clone_and_serialized_resume(self):
        import ai
        st, a, d = live_battle(fp_regulars=4,
            fp_leaders=[companion('Aragorn', 3, 2), companion(), gs.Leader(gs.Nation.ROHAN)])
        cid = card_id('Mighty Attack')
        st.hands[gs.Side.FREE_PEOPLES].hand = [cid]
        st.hands[gs.Side.SHADOW].hand = []
        st.pending_combat = {'phase': 'attacker_combat_card',
            'attacker_region': a, 'defender_region': d,
            'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
            'deciding_side': gs.Side.FREE_PEOPLES, 'rounds_fought': 0, 'is_siege': False}
        play = next(x for x in ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)
                    if x.action_type == 'combat_card_play' and x.params.get('card_id') == cid)
        st = ai.apply_action(st, play)
        decline = next(x for x in ai.get_legal_actions(st, gs.Side.SHADOW)
                       if x.action_type == 'combat_card_none')
        with patch.object(cb, 'roll_d6_batch', Dice([6, 1, 1, 1, 1])):
            st = ai.apply_action(st, decline)
        self.assertEqual(st.pending_combat['effect_choice']['effect'], 'mighty_attack')
        menu = ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)
        self.assertEqual([x.params.get('target') for x in menu], ['Aragorn', 'Meriadoc Brandybuck'])
        original = pickle.dumps(st)
        for clone in (ai.clone_state(st), pickle.loads(pickle.dumps(st))):
            c = clone.pending_combat['_round_resume']['cursor'].combat
            self.assertIs(c._live_state, clone)
            self.assertIs(c.attacker_army, clone.armies['fords of isen'])
            self.assertIs(c.companions_in_battle[0], clone.armies['fords of isen'].leaders[0])
            action = next(x for x in ai.get_legal_actions(clone, gs.Side.FREE_PEOPLES)
                          if x.params.get('target') == 'Aragorn')
            with patch.object(cb, 'roll_d6_batch', Dice([6, 6], [6, 6, 6])):
                done = ai.apply_action(clone, action)
            self.assertEqual(done.pending_combat['rounds_fought'], 1)
            self.assertEqual(done._last_battle_result.deferred_hits_on_defender, 4)
            self.assertNotIn('_round_resume', done.pending_combat)
            self.assertEqual(pickle.dumps(st), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
