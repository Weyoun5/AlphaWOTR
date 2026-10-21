"""Mixed garrisons and physical cancellation of early siege capacity returns."""
from collections import Counter
import pickle
import shutil
import unittest

import ai
import game_logic as gl
import game_state as gs
import test_live_driver as td
import test_bridge_oct11_siege_withdrawal_prefix as original
from test_bridge_oct11_siege_withdrawal_prefix import RETURN, SEAT
from wotr_bridge.live import live_parser, siege_withdrawal_gestures as siege


class SiegeWithdrawalAdversarial(unittest.TestCase):
    def mixed(self, human, returned_elites):
        state = td.fresh_state()
        defender = gs.Nation.GONDOR if human is gs.Side.FREE_PEOPLES else gs.Nation.SAURON
        enemy = gs.Nation.SAURON if human is gs.Side.FREE_PEOPLES else gs.Nation.ELVES
        ai_side = gs.Side.SHADOW if human is gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES
        region, source = ('morannon', 'dagorlad') if human is gs.Side.FREE_PEOPLES else ('lorien', 'dimrill dale')
        for nation in (defender, enemy):
            state.political.boxes[nation] = 3
            state.political.active[nation] = True
        state.enemy_controlled[human].add(region)
        leaders = [gs.Leader(defender)] if human is gs.Side.FREE_PEOPLES else [gs.Nazgul(gs.Nation.SAURON)]
        if human is gs.Side.FREE_PEOPLES:
            companion = next(c for c in state.fellowship.companions if c.name == 'Boromir')
            state.fellowship.companions.remove(companion)
            leaders.append(companion)
        state.armies = {
            region: gs.Army(state.board.region(region),
                [gs.Unit(defender, gs.UnitKind.REGULAR) for _ in range(4)]
                + [gs.Unit(defender, gs.UnitKind.ELITE) for _ in range(3)], leaders),
            source: gs.Army(state.board.region(source), [gs.Unit(enemy, gs.UnitKind.REGULAR) for _ in range(3)], [])}
        for hand in state.hands.values():
            hand.hand[:] = []
        gl.adopt_rolled_faces(state, [(ai_side, 0, gs.ActionDieFace.ARMY)])
        attack = next(a for a in ai.get_legal_actions(state, ai_side)
                      if a.action_type == 'army_attack' and a.die_idx == 0
                      and a.params['from'] == source and a.params['to'] == region)
        state = ai.apply_action(state, attack)
        rig = td.Rig(state, ai_side, phase='actions')
        self.addCleanup(shutil.rmtree, rig.td)
        rig.drv.last_actor = ai_side
        # Independent client snapshot after the two human gestures below.
        # The engine still precedes those gestures. Include the attacking
        # field army: its subsequent legal advance must render real pieces.
        # Literal IDs follow the native construction roster, not engine units.
        if human is gs.Side.FREE_PEOPLES:
            rig.drv.view.piece_region = {
                '075': 'Morannon Stronghold', '076': 'Morannon Stronghold',
                '077': 'Morannon Stronghold', '079': 'Morannon Stronghold',
                '084': 'Morannon Stronghold', '273': 'Morannon Stronghold',
                '274': 'Morannon Stronghold', '085': 'Morannon Stronghold',
                '244': 'Morannon Stronghold',  # Gondor Leader and Boromir
                '337': 'Dagorlad', '338': 'Dagorlad', '339': 'Dagorlad',
            }
            returned_pids = ('084', '273') if returned_elites else ('075', '076')
            returned_zone = 'FP Reinforcements'
        else:
            rig.drv.view.piece_region = {
                '337': 'Lorien Stronghold', '338': 'Lorien Stronghold',
                '339': 'Lorien Stronghold', '340': 'Lorien Stronghold',
                '035': 'Lorien Stronghold', '036': 'Lorien Stronghold',
                '333': 'Lorien Stronghold', '037': 'Lorien Stronghold',
                '086': 'Dimrill Dale', '087': 'Dimrill Dale', '088': 'Dimrill Dale',
            }
            returned_pids = ('035', '036') if returned_elites else ('337', '338')
            returned_zone = 'SA Reinforcements'
        for pid in returned_pids:
            rig.drv.view.piece_region[pid] = returned_zone
        nation = 'Gondor' if human is gs.Side.FREE_PEOPLES else 'Sauron'
        reserve = 'FP Reinforcements' if human is gs.Side.FREE_PEOPLES else 'SA Reinforcements'
        returning = '%s Elite' % nation if returned_elites else '%s Regular' % nation
        rest = '[4]%s Regular, %s Elite' % (nation, nation) if returned_elites else '[2]%s Regular, [3]%s Elite' % (nation, nation)
        rest += ', Gondor Leader, Boromir' if human is gs.Side.FREE_PEOPLES else ', Nazgul'
        lines = ['$Wey moves group ([2]%s) from %s to %s.' % (returning, region.title(), reserve),
                 '$Wey moves group (%s) from %s to %s Stronghold.' % (rest, region.title(), region.title())]
        return rig, lines, region, defender

    def test_both_owners_group_returns_and_named_garrison(self):
        for owner in gs.Side:
            for elites in (False, True):
                with self.subTest(owner=owner, returned_elites=elites):
                    rig, lines, region, nation = self.mixed(owner, elites)
                    before_losses = pickle.dumps(rig.drv.es.fp_casualties)
                    rig.feed(*lines)
                    pending = ai.describe_pending(rig.drv.es)
                    self.assertEqual(pending['kind'], 'combat:victor_advance', rig.drv.findings[-4:])
                    self.assertIs(pending['side'], rig.drv.ai_side)
                    self.assertEqual(len(rig.drv.es.armies[region].units), 7)
                    advance = next(a for a in pending['actions'] if a.action_type == 'victor_advance')
                    self.assertTrue(rig.drv._commit_chain([advance], [], 'ai:combat:victor_advance'))
                    rig.drv.tick()
                    ours = [u for u in rig.drv.es.armies[region].units if u.nation is nation]
                    self.assertEqual(Counter(u.kind for u in ours),
                        {gs.UnitKind.REGULAR: 4 if elites else 2,
                         gs.UnitKind.ELITE: 1 if elites else 3})
                    self.assertEqual(pickle.dumps(rig.drv.es.fp_casualties), before_losses)
                    self.assertIsNone(rig.drv.pending_buf)

    def stay_rig(self):
        rig = original.SiegeWithdrawalPrefix.rig(self)
        rig.feed(RETURN, SEAT)
        dp = ai.describe_pending(rig.drv.es)
        action = next(a for a in dp['actions'] if a.action_type == 'victor_stay')
        pre = rig.drv.es
        post = ai.apply_action(pre, action)
        # The real server records its chosen action so native Undo can rebase.
        rig.drv._snapshot([], 'ai:combat:victor_advance', post=post, action=action)
        rig.drv.es = post
        rig.drv.hands.es = post
        rig.drv.post_commit_sweep(pre, post, actions=[action], label='ai:combat:victor_advance')
        rig.drv.tick()
        return rig

    def test_stay_can_be_corrected_by_native_undo_without_losing_ai_choice(self):
        rig = self.stay_rig()
        rig.feed('$Wey undoes last action : '+SEAT)
        rig.feed('$Wey undoes last action : '+RETURN)
        self.assertIsNone(rig.drv.pending_buf)
        self.assertIsNone(rig.drv.es.pending_combat)
        self.assertEqual(len(rig.drv.es.armies['morannon'].units), 6)
        self.assertEqual(len(rig.drv.es.armies['dagorlad'].units), 10)
        self.assertFalse(any('RE-BASE FAILED' in text for text in rig.drv.alarms))

    def test_stay_can_be_corrected_by_actual_return_movements(self):
        rig = self.stay_rig()
        before = pickle.dumps(rig.drv.es)
        out = rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Morannon.',
                       '$Wey moves group ([5]Gondor Regular) from Morannon Stronghold to Morannon.')
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertIsNone(rig.drv.pending_buf, (rig.drv.alarms, out))

    def test_partial_correction_and_undo_preserve_engine_and_rng(self):
        rig = self.stay_rig()
        back = '$Wey moves Gondor Regular from FP Reinforcements to Morannon.'
        unseat = '$Wey moves group ([5]Gondor Regular) from Morannon Stronghold to Morannon.'
        rig.rand.feed('d6', [2, 4])
        rig.rand._consume_d6(1)
        before = pickle.dumps((rig.drv.es, rig.rand._feed, rig.rand._feed_pos))
        commits = rig.drv.n_commits_total
        rig.feed(back)
        self.assertIsNotNone(rig.drv.pending_buf)
        rig.feed('$Wey undoes last action : '+back)
        rig.feed(unseat, back)
        self.assertIsNone(rig.drv.pending_buf)
        self.assertEqual(rig.drv.n_commits_total, commits)
        self.assertEqual(pickle.dumps((rig.drv.es, rig.rand._feed, rig.rand._feed_pos)), before)
        # Undo of the final correction reopens only the physical remainder.
        rig.feed('$Wey undoes last action : '+back)
        self.assertIsNotNone(rig.drv.pending_buf)
        rig.feed(back)
        self.assertIsNone(rig.drv.pending_buf)
        self.assertEqual(pickle.dumps((rig.drv.es, rig.rand._feed, rig.rand._feed_pos)), before)

    def test_correction_rejects_extra_wrong_units_and_mismatched_native_pids(self):
        rig = self.stay_rig()
        buf = rig.drv.pending_buf
        buf.records[0]['meta'] = {'pids': ['081']}
        buf._siege_awaiting_advance['seats'][0]['meta'] = {'pids': ['075','076','077','082','083']}
        before = pickle.dumps(rig.drv.es)
        bad_lines = (
            '$Wey moves group ([2]Gondor Regular) from FP Reinforcements to Morannon.',
            '$Wey moves Rohan Regular from FP Reinforcements to Morannon.',
            '$Wey moves Gondor Elite from FP Reinforcements to Morannon.',
            '$Wey moves Gondor Regular from FP Casualties to Morannon.',
            '$Wey moves Gondor Regular from FP Reinforcements to Dagorlad.')
        for line in bad_lines:
            self.assertIsNone(siege.consume_correction(rig.drv, live_parser.parse(line)), line)
        back = live_parser.parse('$Wey moves Gondor Regular from FP Reinforcements to Morannon.')
        back['meta'] = {'pids': ['075']}  # This figure was seated, never returned.
        self.assertIsNone(siege.consume_correction(rig.drv, back))
        back['meta'] = {'pids': ['081']}
        self.assertEqual(siege.consume_correction(rig.drv, back), [])
        self.assertIsNotNone(rig.drv.pending_buf)  # Five seated figures still owed.
        unseat = live_parser.parse('$Wey moves group ([5]Gondor Regular) from Morannon Stronghold to Morannon.')
        unseat['meta'] = {'pids': ['075','076','077','082','083']}
        self.assertEqual(siege.consume_correction(rig.drv, unseat), [])
        self.assertIsNone(rig.drv.pending_buf)
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertIsNone(siege.consume_correction(rig.drv, unseat))
        self.assertEqual(pickle.dumps(rig.drv.es), before)

    def test_correction_cannot_replace_an_unmade_attacker_choice(self):
        rig = original.SiegeWithdrawalPrefix.rig(self)
        rig.feed(RETURN, SEAT)
        record = live_parser.parse('$Wey moves Gondor Regular from FP Reinforcements to Morannon.')
        before = pickle.dumps(rig.drv.es)
        self.assertIsNone(siege.consume_correction(rig.drv, record))
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:victor_advance')


if __name__ == '__main__':
    unittest.main(verbosity=2)
