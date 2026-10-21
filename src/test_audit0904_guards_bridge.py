"""Literal Guards player gestures through the real live feed and matcher.

The card/rulebook oracle starts Minas Tirith at 3R/1E/1L and adds one
chosen Gondor Regular OR Elite and one Leader. Inputs below are authored
client grammar, never generated from a legal menu or an engine emitter.
These are Python bridge acceptance controls, not archived or Java games.
"""
import copy
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import ai
import emitter
import log_replay
import test_audit0904_recruit_choices as guards

# Sensitivity run against preserved bridge modules, with the corrected
# engine/menu still loaded. No source files are overwritten by this run.
if os.environ.get('WOTR_AUDIT_GUARDS_BRIDGE_BASELINE'):
    import live
    baseline = Path(os.environ['WOTR_AUDIT_GUARDS_BRIDGE_BASELINE'])
    for name in ('gesture_match', 'driver'):
        spec = importlib.util.spec_from_file_location('live.' + name,
                                                       baseline / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        setattr(live, name, module)
        spec.loader.exec_module(module)
from live.driver import LiveDriver
from live.randomness import ClientRandomness

STAGE = '$Wey moves Free Action Die (M) from FP Dice to Current Action.'
PLAY = ('$Wey moves Guards of the Citadel/Charge from Free Peoples Hand '
        'to Current FP Card.')
REGULAR = '$Wey moves Gondor Regular from FP Reinforcements to Minas Tirith.'
ELITE = '$Wey moves Gondor Elite from FP Reinforcements to Minas Tirith.'
LEADER = '$Wey moves Gondor Leader from FP Reinforcements to Minas Tirith.'
CLEAN = ('$Wey kills group (Free Action Die (M), Guards of the Citadel/Charge) '
         'in Current Action.')


class GuardsBridgeAcceptance(unittest.TestCase):
    diagnostics = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get('WOTR_AUDIT_GUARDS_BRIDGE_LOG')
        if path:
            Path(path).write_text(json.dumps(cls.diagnostics, indent=2) + '\n',
                                  encoding='utf-8')

    def rig(self, state=None):
        temp = tempfile.TemporaryDirectory(prefix='wotr-guards-bridge-')
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        rand = ClientRandomness(seed=17)
        rand.install()
        driver = LiveDriver(guards.before_play() if state is None else state,
                            guards.SH, str(path),
                            randomness=rand, nick='WOTR_AI_Shadow')
        driver.phase = 'actions'
        return driver, path

    def feed(self, driver, path, *lines):
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            if lines:
                stream.write('\n'.join(lines) + '\n')
        out = driver.tick()
        self.diagnostics.append(copy.deepcopy(dict(test=self.id(), lines=lines,
            output=list(map(repr, out)), figures=guards.gondor_figures(driver.es),
            paid=[d.used for d in driver.es.dice[guards.FP]],
            pending=getattr(driver.es, 'pending_card_option', None),
            commits=driver.n_commits_total, alarms=driver.alarms,
            findings=list(map(repr, driver.findings)),
            pending_records=None if driver.pending_buf is None
                else list(map(repr, driver.pending_buf.records)))))
        return out

    def settled(self, driver, kind):
        self.assertEqual(guards.gondor_figures(driver.es),
                         (4, 1, 2) if kind == 'regular' else (3, 2, 2))
        self.assertEqual(driver.es.hands[guards.FP].hand, [])
        self.assertEqual(driver.es.discard_face_up['fp_strategy'].count(guards.CID), 1)
        self.assertEqual([d.used for d in driver.es.dice[guards.FP]], [True, False])
        self.assertEqual([d.used for d in driver.es.dice[guards.SH]], [False])
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.pending_buf)
        self.assertIsNone(driver.deferred)
        self.assertIsNone(driver.seg.window)
        self.assertFalse(driver.holds)
        self.assertEqual(driver.alarms, [])
        self.assertFalse(any(driver.rand.unconsumed().values()))

    def test_full_literal_window_accepts_either_printed_kind(self):
        for kind, unit in [('regular', REGULAR), ('elite', ELITE)]:
            with self.subTest(kind=kind):
                driver, path = self.rig()
                self.feed(driver, path, STAGE, PLAY, unit, LEADER, CLEAN)
                self.settled(driver, kind)

    def test_each_feed_line_may_arrive_separately(self):
        for kind, unit in [('regular', REGULAR), ('elite', ELITE)]:
            with self.subTest(kind=kind):
                driver, path = self.rig()
                for line in (STAGE, PLAY, unit, LEADER, CLEAN):
                    self.feed(driver, path, line)
                self.settled(driver, kind)

    def test_leader_first_does_not_choose_unit_kind(self):
        for kind, unit in [('regular', REGULAR), ('elite', ELITE)]:
            with self.subTest(kind=kind):
                driver, path = self.rig()
                self.feed(driver, path, STAGE, PLAY, LEADER)
                self.assertEqual(guards.gondor_figures(driver.es)[:2], (3, 1))
                self.feed(driver, path, unit)
                self.feed(driver, path, CLEAN)
                self.settled(driver, kind)

    def test_one_group_drag_carries_unit_and_leader(self):
        for kind in ('regular', 'elite'):
            with self.subTest(kind=kind):
                driver, path = self.rig()
                group = ('$Wey moves group (Gondor Leader, Gondor %s) '
                         'from FP Reinforcements to Minas Tirith.' % kind.title())
                self.feed(driver, path, STAGE, PLAY)
                self.feed(driver, path, group)
                self.feed(driver, path, CLEAN)
                self.settled(driver, kind)

    def test_cleanup_can_be_card_then_die_in_separate_deliveries(self):
        driver, path = self.rig()
        self.feed(driver, path, STAGE, PLAY, REGULAR, LEADER)
        self.feed(driver, path, '$Wey kills Guards of the Citadel/Charge in Current FP Card.')
        self.feed(driver, path, '$Wey kills Free Action Die (M) in Current Action.')
        self.settled(driver, 'regular')
        commits = driver.n_commits_total
        for _ in range(3):
            self.feed(driver, path)
        self.assertEqual(driver.n_commits_total, commits)
        self.settled(driver, 'regular')

    def test_units_from_fp_dead_pool_cannot_be_accepted_as_reinforcements(self):
        for source in ('FP Casualties', 'SA Reinforcements', 'Reinforcements'):
            with self.subTest(source=source):
                driver, path = self.rig()
                out = self.feed(driver, path, STAGE, PLAY,
                               REGULAR.replace('FP Reinforcements', source), LEADER, CLEAN)
                self.assertNotEqual(guards.gondor_figures(driver.es), (4, 1, 2))
                self.assertFalse(driver.es.discard_face_up['fp_strategy'].count(guards.CID))
                self.assertTrue(any('FP Reinforcements' in str(item) for item in out), out)

    def test_wrong_leader_cannot_satisfy_the_gondor_leader(self):
        for figure in ('Rohan Leader', 'Nazgul', 'The Witch-King', 'Gimli'):
            with self.subTest(figure=figure):
                driver, path = self.rig()
                self.feed(driver, path, STAGE, PLAY, REGULAR,
                          LEADER.replace('Gondor Leader', figure), CLEAN)
                self.assertNotEqual(guards.gondor_figures(driver.es), (4, 1, 2))
                self.assertFalse(driver.es.discard_face_up['fp_strategy'].count(guards.CID))

    def test_pending_after_early_die_close_accepts_leader_first(self):
        for kind, unit in [('regular', REGULAR), ('elite', ELITE)]:
            with self.subTest(kind=kind):
                driver, path = self.rig()
                self.feed(driver, path, STAGE, PLAY,
                          '$Wey kills Free Action Die (M) in Current Action.')
                self.assertIsNotNone(driver.es.pending_card_option)
                self.feed(driver, path, LEADER)
                self.assertEqual(guards.gondor_figures(driver.es), (3, 1, 1))
                self.feed(driver, path, unit)
                self.feed(driver, path,
                          '$Wey kills Guards of the Citadel/Charge in Current FP Card.')
                self.settled(driver, kind)

    def test_pending_cannot_recruit_a_dead_unit(self):
        driver, path = self.rig()
        self.feed(driver, path, STAGE, PLAY,
                  '$Wey kills Free Action Die (M) in Current Action.')
        out = self.feed(driver, path, LEADER,
                        REGULAR.replace('FP Reinforcements', 'FP Casualties'))
        self.assertEqual(guards.gondor_figures(driver.es), (3, 1, 1))
        self.assertIsNotNone(driver.es.pending_card_option)
        self.assertFalse(driver.es.discard_face_up['fp_strategy'].count(guards.CID))
        self.assertTrue(any('FP Reinforcements' in str(item) for item in out), out)

    def test_pending_wrong_pool_can_be_undone_then_completed_once(self):
        driver, path = self.rig()
        self.feed(driver, path, STAGE, PLAY,
                  '$Wey kills Free Action Die (M) in Current Action.')
        bad = REGULAR.replace('FP Reinforcements', 'FP Casualties')
        self.feed(driver, path, LEADER, bad)
        self.assertEqual(guards.gondor_figures(driver.es), (3, 1, 1))
        self.feed(driver, path, '$Wey undoes last action : ' + bad.lstrip('$'))
        self.feed(driver, path, REGULAR)
        self.feed(driver, path,
                  '$Wey kills Guards of the Citadel/Charge in Current FP Card.')
        self.settled(driver, 'regular')
        commits = driver.n_commits_total
        for _ in range(3):
            self.feed(driver, path)
        self.assertEqual(driver.n_commits_total, commits)
        self.settled(driver, 'regular')

    def test_besieged_recruits_may_be_placed_via_region_then_box(self):
        for kind in ('regular', 'elite'):
            for via_region in (False, True):
                with self.subTest(kind=kind, via_region=via_region):
                    state = guards.before_play()
                    guards.later_war(state)
                    guards.army(state).units.extend([
                        guards.gs.Unit(guards.SAURON, guards.REGULAR) for _ in range(2)])
                    state.besieged_strongholds.add(guards.MT)
                    driver, path = self.rig(state)
                    group = 'group (Gondor Leader, Gondor %s)' % kind.title()
                    self.feed(driver, path, STAGE, PLAY,
                              '$Wey moves %s from FP Reinforcements to %s.' %
                              (group, 'Minas Tirith' if via_region else 'Minas Tirith Stronghold'))
                    if via_region:
                        self.feed(driver, path, '$Wey moves %s from Minas Tirith '
                                  'to Minas Tirith Stronghold.' % group)
                    self.feed(driver, path, CLEAN)
                    self.settled(driver, kind)
                    self.assertIn(guards.MT, driver.es.besieged_strongholds)
                    self.assertNotIn(guards.MT, driver.es.enemy_controlled[guards.SH])
                    self.assertEqual(sum(u.nation == guards.SAURON
                                         for u in guards.army(driver.es).units), 2)

    def test_fp_besieger_cannot_place_recruits_in_shadow_stronghold_box(self):
        for kind in ('regular', 'elite'):
            for via_region in (False, True):
                with self.subTest(kind=kind, via_region=via_region):
                    state = guards.before_play()
                    guards.later_war(state)
                    guards.army(state).units.extend([
                        guards.gs.Unit(guards.SAURON, guards.REGULAR) for _ in range(2)])
                    state.besieged_strongholds.add(guards.MT)
                    state.enemy_controlled[guards.SH].add(guards.MT)
                    driver, path = self.rig(state)
                    group = 'group (Gondor Leader, Gondor %s)' % kind.title()
                    lines = [STAGE, PLAY, '$Wey moves %s from FP Reinforcements to %s.' %
                             (group, 'Minas Tirith' if via_region else 'Minas Tirith Stronghold')]
                    if via_region:
                        lines.append('$Wey moves %s from Minas Tirith '
                                     'to Minas Tirith Stronghold.' % group)
                    out = self.feed(driver, path, *lines, CLEAN)
                    self.assertEqual(guards.gondor_figures(driver.es), (3, 1, 1))
                    self.assertFalse(driver.es.discard_face_up['fp_strategy'].count(guards.CID))
                    self.assertTrue(any('Stronghold box' in str(item)
                                        and 'Minas Tirith region' in str(item) for item in out), out)

    def test_fp_besieger_recruits_in_the_field_beside_shadow_garrison(self):
        for kind, unit in [('regular', REGULAR), ('elite', ELITE)]:
            with self.subTest(kind=kind):
                state = guards.before_play()
                guards.later_war(state)
                guards.army(state).units.extend([
                    guards.gs.Unit(guards.SAURON, guards.REGULAR) for _ in range(2)])
                state.besieged_strongholds.add(guards.MT)
                state.enemy_controlled[guards.SH].add(guards.MT)
                driver, path = self.rig(state)
                self.feed(driver, path, STAGE, PLAY, unit, LEADER, CLEAN)
                self.settled(driver, kind)
                self.assertIn(guards.MT, driver.es.enemy_controlled[guards.SH])
                self.assertIn(guards.MT, driver.es.besieged_strongholds)
                self.assertEqual(sum(u.nation == guards.SAURON
                                     for u in guards.army(driver.es).units), 2)

    def test_ai_selected_kind_uses_exact_pool_pieces_and_one_cleanup(self):
        from live.live_server import LiveServer
        temp = tempfile.TemporaryDirectory(prefix='wotr-guards-ai-')
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        server = LiveServer(temp.name, guards.FP, load_nets=False, seed=17)
        driver = server.driver
        driver.es = guards.before_play()
        driver.hands.es = driver.es
        driver.phase = 'actions'
        # Literal client inventory: 277 is a Gondor Regular,
        # 273 an Elite, 270 a Leader, all in FP Reinforcements.
        # Card pid 130 represents this game's commitment decode.
        positions = json.loads((ROOT / 'wotr_bridge' / 'initial_positions.json')
                               .read_text(encoding='utf-8'))
        driver.view.piece_region = {pid: log_replay.decode_location(zone)
                                   for pid, zone in positions.items()}
        driver.view.card_pid_map[guards.CID] = '130'
        driver.view.piece_region['130'] = 'Free Peoples Hand'
        driver.view.piece_region['210'] = 'FP Dice'
        play = ai.Action(0, 'play_event', {'card_id': guards.CID, 'side': 'fp'})
        output = server._ai_decide({'side': guards.FP, 'kind': 'action',
                                    'actions': [play]})
        self.assertEqual([s for s in output if s.startswith('!CALL')],
                         ['!CALL play_card 130'])
        self.assertEqual(guards.gondor_figures(driver.es), (3, 1, 1))
        self.assertEqual(server._flush_owed_die_kills(), [])
        # Literal rendered echoes from the AI's die/card prologue.
        self.feed(driver, path,
                  STAGE.replace('$Wey', '$' + server.nick),
                  PLAY.replace('$Wey', '$' + server.nick))
        menu = ai.get_legal_actions(driver.es, guards.FP)
        effects = server._ai_decide({'side': guards.FP, 'kind': 'card_option',
                                     'actions': menu})
        moves = [emitter.classify_line(emitter.strip_tail(s)) for s in effects]
        moves = [(op, fields) for op, fields in moves if op == 'move']
        self.assertEqual(len(moves), 2, effects)
        self.assertEqual({tuple(fields['pieces']) for _, fields in moves},
                         {('273',), ('270',)})
        for _, fields in moves:
            self.assertEqual(fields['frm'], 'Game.2038')
            self.assertEqual(fields['to'], 'Game.1944')
        cleanup = server._flush_owed_die_kills()
        self.assertEqual(len(cleanup), 1)
        op, fields = emitter.classify_line(emitter.strip_tail(cleanup[0]))
        self.assertEqual(op, 'kill')
        self.assertEqual(fields['pieces'], ['210', '130'])
        self.assertEqual(server._flush_owed_die_kills(), [])
        commits = driver.n_commits_total
        self.feed(driver, path,
                  ELITE
                      .replace('$Wey', '$' + server.nick),
                  LEADER.replace('$Wey', '$' + server.nick),
                  CLEAN.replace('$Wey', '$' + server.nick))
        self.assertEqual(driver.n_commits_total, commits)
        self.settled(driver, 'elite')


if __name__ == '__main__':
    unittest.main(verbosity=2)
