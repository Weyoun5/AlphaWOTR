"""Literal native Dead Men prefixes through the real LiveDriver; no desktop."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import board
import event_effects as ee
import game_logic as gl
import game_state as gs
from test_live_driver import fresh_state
from live.driver import LiveDriver
from live.randomness import ClientRandomness
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CID = 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow'

class Rig:
    """One canonical live.* package, matching the production server imports.

    Installing both live.randomness and wotr_bridge.live.randomness creates
    different process-global active funnels and corrupts later test cases.
    """
    def __init__(self, state, ai_side, phase=None, seed=7):
        self.td = tempfile.mkdtemp()
        self.path = str(Path(self.td) / 'feed.txt')
        Path(self.path).write_text('', encoding='utf8')
        self.rand = ClientRandomness(seed=seed)
        self.rand.install()
        self.drv = LiveDriver(state, ai_side, self.path, randomness=self.rand)
        if phase:
            self.drv.phase = phase

    def feed(self, *lines):
        with open(self.path, 'a', encoding='utf8') as stream:
            for line in lines:
                stream.write(line+'\n')
        return self.drv.tick()

class DeadMenPrefixes(unittest.TestCase):
    def position(self, recruit=True):
        state = fresh_state()
        state.armies = {}
        state._declare_phase = False
        state.dice_rolled = True
        for pool in state.dice.values():
            for die in pool:
                die.used = True
        state.dice[FP][0].current = gs.ActionDieFace.CHARACTER
        state.hands[FP].hand = []
        if recruit:
            ee._dead_men_recruit(state, 'erech', {'_from_pending': True}, {'effects': []})
            self.assertEqual([o['recruit_count'] for o in state.pending_card_option['options']], [0, 1, 2, 3])
        else:
            army = gl.get_or_create_army(state, 'westemnet')
            for name in ('Strider', 'Gimli'):
                comp = next(c for c in state.fellowship.companions if c.name == name)
                state.fellowship.companions.remove(comp)
                army.leaders.append(comp)
            from card_options import dead_men_options
            state.pending_card_option = dict(card_id=CID, card_name='Dead Men of Dunharrow',
                side='fp', decider='fp', direct=True, options=dead_men_options(state, FP))
        rig = Rig(state, SH, phase='actions')
        self.addCleanup(shutil.rmtree, rig.td)
        rig.drv.enforce_human = True
        rig.drv.last_actor = FP
        return rig

    def count(self, rig):
        return sum(u.nation is gs.Nation.GONDOR and u.kind is gs.UnitKind.REGULAR
            for region, army in rig.drv.es.armies.items() if board.sanitize(region) == 'erech'
            for u in army.units)

    def test_recruit_one_prefix_of_two_remains_uncommitted(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        self.assertEqual(rig.drv.n_commits_total, 0, (rig.drv.findings[-5:], rig.drv.alarms))
        self.assertEqual(self.count(rig), 0)
        self.assertIsNotNone(ai.describe_pending(rig.drv.es))
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('<Wey> done')
        self.assertEqual(self.count(rig), 2)
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertFalse(rig.drv.alarms)

    def test_recruit_one_prefix_of_three_remains_uncommitted(self):
        rig = self.position()
        for _ in range(3):
            rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
            self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('<Wey> done')
        self.assertEqual(self.count(rig), 3)
        self.assertEqual(rig.drv.n_commits_total, 1)

    def test_recruit_group_two_is_also_incomplete_without_close(self):
        rig = self.position()
        rig.feed('$Wey moves group ([2]Gondor Regular) from FP Reinforcements to Erech.')
        self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('<Wey> done')
        self.assertEqual(self.count(rig), 2)

    def test_optional_companion_prefix_does_not_choose_smaller_group(self):
        rig = self.position(recruit=False)
        rig.feed('$Wey moves Strider from Westemnet to Erech.')
        self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('$Wey moves Gimli from Westemnet to Erech.')
        rig.feed('<Wey> done')
        names = {l.name for k,a in rig.drv.es.armies.items() if board.sanitize(k)=='erech' for l in a.leaders}
        self.assertEqual(names, {'Strider', 'Gimli'})

    def test_completed_single_recruit_is_valid_control(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.', '<Wey> done')
        self.assertEqual(self.count(rig), 1)
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertFalse(rig.drv.alarms)

    def test_full_three_group_is_valid_control(self):
        rig = self.position()
        rig.feed('$Wey moves group ([3]Gondor Regular) from FP Reinforcements to Erech.', '<Wey> done')
        self.assertEqual(self.count(rig), 3)
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertFalse(rig.drv.alarms)

    def test_complete_companion_group_is_valid_control(self):
        rig = self.position(recruit=False)
        rig.feed('$Wey moves group (Strider, Gimli) from Westemnet to Erech.', '<Wey> done')
        names = {l.name for k,a in rig.drv.es.armies.items() if board.sanitize(k)=='erech' for l in a.leaders}
        self.assertEqual(names, {'Strider', 'Gimli'})
        self.assertFalse(rig.drv.alarms)

    def test_zero_recruits_require_explicit_done(self):
        rig = self.position()
        rig.drv.tick()
        self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('<Wey> done')
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertEqual(self.count(rig), 0)
        self.assertIsNone(ai.describe_pending(rig.drv.es))

    def test_card_disposal_closes_final_recruit_choice(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        self.assertEqual(rig.drv.n_commits_total, 0)
        rig.feed('$Wey kills Dead Men of Dunharrow in Current FP Card.')
        self.assertEqual(self.count(rig), 1)
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertFalse(rig.drv.alarms)

    def test_same_card_in_other_display_slot_closes_human_choice(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        rig.feed('$Wey kills Dead Men of Dunharrow in Current SA Card.')
        self.assertEqual(rig.drv.n_commits_total, 1)
        self.assertEqual(self.count(rig), 1)
        self.assertIsNone(ai.describe_pending(rig.drv.es))

    def test_different_card_disposal_does_not_close_choice(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        rig.feed('$Wey kills Gwaihir the Windlord in Current FP Card.')
        self.assertEqual(rig.drv.n_commits_total, 0)
        self.assertEqual(self.count(rig), 0)
        self.assertIsNotNone(ai.describe_pending(rig.drv.es))

    def test_known_ai_actor_disposal_does_not_close_choice(self):
        rig = self.position()
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        rig.feed('$%s kills Dead Men of Dunharrow in Current FP Card.' % rig.drv.nick)
        self.assertEqual(rig.drv.n_commits_total, 0)
        self.assertEqual(self.count(rig), 0)
        self.assertIsNotNone(ai.describe_pending(rig.drv.es))

    def test_initial_group_done_requests_attack_rng_without_clearing_choice(self):
        rig = self.position(recruit=False)
        enemy = gl.get_or_create_army(rig.drv.es, 'erech')
        enemy.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)]
        output = rig.feed('$Wey moves Strider from Westemnet to Erech.')
        self.assertFalse(any(item[0] == 'rng_ask' for item in output))
        self.assertEqual(rig.drv.n_commits_total, 0)
        output = rig.feed('<Wey> done')
        self.assertEqual(sum(item[0] == 'rng_ask' for item in output), 1)
        self.assertEqual(rig.drv.n_commits_total, 0)
        self.assertIsNotNone(ai.describe_pending(rig.drv.es))
        self.assertFalse(rig.drv.alarms)

    def test_table_staging_is_not_a_completion_signal(self):
        rig = self.position()
        from live.card_completion import closes_card
        from live.live_parser import parse
        record = parse('$Wey moves Dead Men of Dunharrow from Current FP Card to Board 4.')
        self.assertFalse(closes_card(rig.drv.es, record))

    def test_regular_elite_attack_roll_opens_owner_casualty_choice(self):
        rig = self.position(recruit=False)
        enemy = gl.get_or_create_army(rig.drv.es, 'erech')
        enemy.units = [gs.Unit(gs.Nation.SAURON, kind) for kind in (gs.UnitKind.REGULAR, gs.UnitKind.ELITE)]
        rig.drv.es.political.boxes[gs.Nation.SAURON] = 3
        rig.feed('$Wey moves Strider from Westemnet to Erech.')
        rig.feed('$Wey moves Gimli from Westemnet to Erech.')
        rig.feed('<Wey> done')
        rig.feed('<game> Wey rolls 1')
        self.assertEqual(rig.drv.n_commits_total, 1)
        pending = ai.describe_pending(rig.drv.es)
        self.assertIs(pending['side'], SH)
        self.assertTrue(rig.drv.es.pending_card_option['card_casualty'])
        self.assertEqual(rig.drv.es.pending_card_option['cas_hits'], 1)
        self.assertFalse(rig.drv.alarms)

    def test_live_server_emits_unique_shadow_consequences_after_human_roll(self):
        # Human FP must not perform the AI-owned unique casualty/retreat.
        rig = self.position(recruit=False)
        enemy = gl.get_or_create_army(rig.drv.es, 'erech')
        enemy.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(3)]
        rig.drv.es.political.boxes[gs.Nation.SAURON] = 3
        from live.live_server import LiveServer
        server = LiveServer(rig.td, SH, load_nets=False, seed=7)
        server.driver.es = rig.drv.es
        for model in server.driver.hand_models.values():
            model.es = server.driver.es
        server.driver.phase = 'actions'
        server.driver.last_actor = FP
        server.driver.enforce_human = True
        server.driver.view.piece_region.update({'241': 'Westemnet', '243': 'Westemnet',
                                                '011': 'Erech', '012': 'Erech', '013': 'Erech'})
        feed = Path(rig.td) / 'ai_feed.txt'
        def physical(line):
            with feed.open('a', encoding='utf8') as stream:
                stream.write(line+'\n')
            return server.handle({'lines': ()})
        physical('$Wey moves Strider from Westemnet to Erech.')
        physical('$Wey moves Gimli from Westemnet to Erech.')
        request = physical('<Wey> done')
        self.assertTrue(any('roll 1 die' in line for line in request))
        response = physical('<game> Wey rolls 1')
        commands = [line for line in response if line.startswith('$')]
        # Independent literal public board oracle: return all three original
        # Regulars, then place exactly two in the sole legal retreat region.
        # Game.1913=Erech, Game.2040=SA Reinforcements, Game.1909=Anfalas.
        self.assertEqual(commands, [
            '$WOTR_AI_Shadow<~Controls.296~>011 012 013 <~Controls.297~><~Game.1913~>',
            '$WOTR_AI_Shadow<~Scribe.141~>011 <~Scribe.142~><~Game.2040~><~Scribe.143~><~Game.1909~><~Scribe.144~>',
            '$WOTR_AI_Shadow<~Scribe.141~>012 <~Scribe.142~><~Game.2040~><~Scribe.143~><~Game.1909~><~Scribe.144~>',
        ])
        self.assertFalse(any('roll 1 die' in line for line in response))
        for _ in range(3):
            self.assertEqual(server.handle({'lines': ()}), [])
        self.assertEqual(server.driver.n_commits_total, 1)
        post = server.driver.es
        self.assertEqual({l.name for l in post.armies['erech'].leaders}, {'Strider', 'Gimli'})
        self.assertEqual(post.armies['erech'].units, [])
        self.assertEqual(len(post.armies['anfalas'].units), 2)
        self.assertTrue(all(u.nation is gs.Nation.SAURON and u.kind is gs.UnitKind.REGULAR
                            for u in post.armies['anfalas'].units))
        self.assertEqual([o['recruit_count'] for o in post.pending_card_option['options']], [0, 1, 2, 3])
        self.assertIs(ai.describe_pending(post)['side'], FP)
        self.assertFalse(server.driver.alarms)
        # Human recruitment remains a fresh, still-open optional choice.
        physical('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertEqual(sum(u.nation is gs.Nation.GONDOR for u in server.driver.es.armies['erech'].units), 0)
        physical('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        physical('<Wey> done')
        self.assertEqual(server.driver.n_commits_total, 2)
        self.assertEqual(sum(u.nation is gs.Nation.GONDOR for u in server.driver.es.armies['erech'].units), 2)
        self.assertFalse(server.driver.alarms)

    def test_casualty_phase_is_not_misclassified_as_group_staging(self):
        rig = self.position()
        rig.drv.es.pending_card_option = dict(card_id=CID, card_name='Dead Men of Dunharrow',
            side='fp', decider='sh', direct=True, card_casualty=True,
            options=[{'casualties': {'regular': 1}}])
        from live.card_completion import needs_boundary, choice_needs_boundary
        self.assertFalse(needs_boundary(rig.drv.es))
        self.assertFalse(choice_needs_boundary(rig.drv.es, {'casualties': {'regular': 1}}))

if __name__ == '__main__':
    unittest.main()

