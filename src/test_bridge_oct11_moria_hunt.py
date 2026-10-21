"""Moria Hunt rerolls through the real Crebain/live randomness boundary."""
from pathlib import Path
from types import SimpleNamespace
import copy
import itertools
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import ai
import board
import cards
import event_effects
import game_logic as gl
import game_state as gs
from live import live_parser
from live.live_server import LiveServer

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
MOVE = [
    '$Human moves Free Action Die (C) from FP Dice to Current Action.',
    '$Human moves FSP from Fellowship Track 0 to Fellowship Track 1.',
    '$Human moves Free Action Die (C) from Current Action to The Hunt For The Ring.',
]


class MoriaHuntTests(unittest.TestCase):
    def rig(self, nazgul, discard, ai_side=SH):
        folder = tempfile.TemporaryDirectory(prefix='moria_hunt_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        feed = Path(folder.name, 'ai_feed.txt')
        feed.write_text('', encoding='utf8')
        server = LiveServer(folder.name, ai_side, load_nets=False, seed=43)
        state = gs.setup_initial_game(board.load_board())
        gl.initialize_decks(state)
        gl.adopt_rolled_faces(state, [(SH, 0, gs.ActionDieFace.ARMY),
                                     (SH, 1, gs.ActionDieFace.CHARACTER),
                                     (FP, 0, gs.ActionDieFace.CHARACTER)])
        gl.allocate_eyes(state, 3, indices=[2, 3, 4])
        state.fellowship.region = state.board.region('moria')
        army = next(a for key, a in state.armies.items() if board.sanitize(key) == 'moria')
        self.assertTrue(army.units)
        if nazgul:
            army.leaders.append(gs.Nazgul(gs.Nation.SAURON))
        state.hands[FP].hand[:] = []
        card = next(c for c in cards.get_cards().values()
                    if c.card_name == 'Flocks of Crebain')
        event_effects._effect_flocks_of_crebain(state, card, {})
        server.driver.es = state
        server.driver.hands.es = state
        server.driver.phase = 'actions'
        server._table_book = {'196': ('Game.2034', card.card_id)}
        server.script = SimpleNamespace(rank=lambda srv, dp, actions:
            [a for a in actions if a.params['choices']['crebain_discard'] is discard])
        expected = ['Shadow army present', 'Shadow stronghold']
        if nazgul:
            expected.insert(0, 'Nazgul present')
        self.assertEqual(gl.get_hunt_params(state)['reroll_sources'], expected)
        if ai_side is FP:
            server.driver.nick_side['Human'] = SH.name
            server.driver.view.piece_region.update({
                '247': 'Moria', '248': 'Fellowship Track 0', '210': 'FP Dice'})
            server.driver.last_actor = SH
            server.script = SimpleNamespace(rank=lambda srv, dp, actions:
                [a for a in actions if a.action_type == 'fellowship_move'])
            return server, {'kind': 'action', 'side': FP,
                            'actions': ai.get_legal_actions(state, FP)}
        feed.write_text('\n'.join(MOVE) + '\n', encoding='utf8')
        output = server.driver.tick()
        pending = next(item[1] for item in output if item[0] == 'ai_decide')
        self.assertEqual(pending['kind'], 'card_option')
        self.assertEqual(server.driver.es.fellowship.progress, 0)
        return server, pending

    def roll(self, server, values):
        # Same rendered grammar as the real native Java roll receipt. Route
        # through the driver's ordinary phase consumer, including side binding.
        roller = server.nick if server.ai_side is SH else 'Human'
        record = live_parser.parse('<game> %s rolls %s ' %
                                   (roller, ' '.join(map(str, values))))
        self.assertEqual(record['op'], 'roll_num')
        self.assertEqual(server.driver._d6_roll_side(record), SH.name)
        server.driver._on_phase(record)

    def feed_native(self, server, line):
        with Path(server.driver.feed.path).open('a', encoding='utf8') as stream:
            stream.write(line + '\n')
        return server.driver.tick()

    def expect_roll(self, server, dp, count, context, cursor):
        output = server._ai_decide(dp)
        pending = server.rand.pending()
        self.assertIsNotNone(pending, output)
        self.assertEqual((pending.kind, pending.kw['ctx'], pending.kw['need'],
                          pending.kw['side'], pending._request_cursor),
                         ('d6', context, count, SH, cursor))
        self.assertEqual([line for line in output if line.startswith('!CALL')],
                         ['!CALL combat_roll %d' % count])
        self.assertIn('Shadow (AI)', server._waiting_on())
        self.assertEqual(server.driver.es.fellowship.progress, 0)
        self.assertFalse(server.driver.es.dice[FP][0].used)
        self.assertFalse(server._ai_decide(dp))  # Idle retry cannot roll again.

    def write_native_dump(self, server):
        from collections import defaultdict
        import emitter
        from live import client_board
        from test_live_witnesses import build_dump_from_state
        dump = client_board.read_board(str(ROOT / 'data/live_fixtures/client_board_20260821_strider_guide.txt'))
        self.assertEqual(len(dump), 428)
        for pid in dump:
            nation, kind = client_board.unit_type(pid)
            if kind in ('regular', 'elite', 'leader', 'nazgul'):
                side = SH if kind == 'nazgul' else ai._side_of_nation(nation)
                dump[pid] = 'SA Reinforcements' if side is SH else 'FP Reinforcements'
        for pid, zone in build_dump_from_state(server.driver.es).items():
            if pid in dump and zone != 'Spare':
                dump[pid] = zone
        dump.update({'248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
        groups = defaultdict(list)
        for pid, zone in dump.items():
            groups[emitter.zone_code_for(board.sanitize(zone))].append(pid)
        feed_position = len(Path(server.driver.feed.path).read_text(encoding='utf8').splitlines()) - 1
        lines = ['#SEQ dump=1 feed_paragraphs=%d raw_lines=0 session=9943' % feed_position]
        for index, (code, pids) in enumerate(groups.items()):
            lines.append('%d\t<~%s~>\t%s\t%s\t%s' % (index, code, ' '.join(pids),
                         ' '.join('-' for _ in pids), ' '.join('-' for _ in pids)))
        Path(server.driver.view.dump_path).write_text('\n'.join(lines) + '\n', encoding='utf8')

    def test_moria_keep_and_discard_observe_all_real_rerolls_once(self):
        cases = [
            (False, False, [1, 1, 1], [1, 6], 1),
            (True, False, [5, 1, 1], [1, 1, 6], 1),
            (True, True, [5, 1, 1], [1, 5], 2),
            (True, True, [1, 1, 1], [1, 1, 1], 0),
        ]
        for nazgul, discard, initial, rerolls, hits in cases:
            with self.subTest(nazgul=nazgul, discard=discard, hits=hits):
                server, dp = self.rig(nazgul, discard)
                self.expect_roll(server, dp, 3, 'hunt roll', ())
                self.roll(server, initial)
                for index, value in enumerate(rerolls):
                    self.expect_roll(server, dp, 1, 'hunt reroll',
                                     tuple(range(3 + index)))
                    self.roll(server, [value])
                output = server._ai_decide(dp)
                if hits:
                    self.assertEqual(server.rand.pending().kind, 'tile')
                    self.assertEqual([line for line in output if line.startswith('!CALL')],
                                     ['!CALL hunt_tile'])
                    tile = next(t for t in server.driver.es.hunt_pool if t.is_eye)
                    server.rand.feed('tile', tile)
                    output = server._ai_decide(dp)
                    self.assertEqual(server.driver.es.pending_hunt['damage'], hits)
                else:
                    self.assertIsNone(server.driver.es.pending_hunt)
                    self.assertIsNone(server.rand.pending())
                post = server.driver.es
                self.assertEqual(post.fellowship.progress, 1)
                self.assertTrue(post.dice[FP][0].used)
                self.assertEqual(bool(post.persistent_cards.get_by_type(
                    gs.PersistentCardType.FLOCKS_OF_CREBAIN)), not discard)
                self.assertFalse(getattr(post, '_crebain_hunt_bonus', 0))
                self.assertFalse(any(kind in ('emission_hold_degraded', 'progress_sweep_emitted')
                                     for kind, _ in server.driver.findings), server.driver.findings[-5:])
                self.assertIsNone(server.driver._crebain_human_transport)
                self.assertFalse(server.rand.shim_uses)
                self.assertEqual(server.rand.unconsumed()['d6'], 0)

    def test_ai_fellowship_human_crebain_keeps_one_physical_move(self):
        import emitter
        for discard, resync in itertools.product((False, True), repeat=2):
            with self.subTest(discard=discard, resync=resync):
                server, dp = self.rig(True, discard, ai_side=FP)
                output = server._ai_decide(dp)
                pending = ai.describe_pending(server.driver.es)
                self.assertEqual(pending['side'], SH)
                self.assertEqual(server.driver.es.fellowship.progress, 0)
                self.assertFalse(server.driver.es.dice[FP][0].used)
                self.assertIsNone(server.rand.pending())
                self.assertFalse(any(line.startswith('!CALL') for line in output))
                physical = [emitter.classify_line(line) for line in output]
                self.assertEqual([row[0] for row in physical if row and row[0] != 'chat'],
                                 ['die_spend_fp', 'track_marker', 'die_to_hunt'])
                # The AI declares its move physically before the human's
                # interposed choice. Native echoes establish this visible state.
                server.driver.view.piece_region.update({
                    '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
                if resync:
                    self.write_native_dump(server)
                    before = server.driver.es
                    for _ in range(2):
                        items, report = server.driver.reset_and_resync()
                        self.assertIs(server.driver.es, before, report)
                        self.assertFalse(server.driver._last_resync_adoption['complete'])
                        self.assertFalse(any(item[0] == 'line' for item in items), items)
                    self.assertEqual(server.driver.view.piece_region['248'], 'Track 1')
                action = next(a for a in pending['actions']
                              if a.params['choices']['crebain_discard'] is discard)
                command = 'discard flocks' if discard else 'keep flocks'
                if resync:
                    # The panel publishes an identity-bound pick command.
                    menu = server.status_menu()['human_menu']['actions']
                    entry = next(row for row in menu
                                 if row['params']['choices']['crebain_discard'] is discard)
                    command = 'pick %d' % entry['params']['say_pick']
                server.driver._on_chat(live_parser.parse('<Human> ' + command))
                call = server.rand.pending()
                self.assertEqual((call.kw['ctx'], call.kw['side'], call.kw['need']),
                                 ('hunt roll', SH, 3))
                asks = server._window_ask_lines(call)
                self.assertFalse(any(line.startswith('!CALL') for line in asks), asks)
                self.assertTrue(any('Shadow (you)' in line for line in asks), asks)
                self.assertTrue(any('roll 3 dice #' in line for line in asks), asks)
                self.assertEqual(server._window_ask_lines(call), [])
                self.roll(server, [5, 1, 1])
                requests = [line for line in asks if 'roll 3 dice #' in line]
                rerolls = [1, 5] if discard else [1, 1, 6]
                for index, value in enumerate(rerolls):
                    server.driver._attempt_chat_action(action, via='chat')
                    call = server.rand.pending()
                    self.assertEqual((call.kw['ctx'], call.kw['side'], call.kw['need'],
                                      call._request_cursor),
                                     ('hunt reroll', SH, 1, tuple(range(3 + index))))
                    asks = server._window_ask_lines(call)
                    self.assertFalse(any(line.startswith('!CALL') for line in asks), asks)
                    self.assertTrue(any('Shadow (you)' in line for line in asks), asks)
                    requests.extend(line for line in asks if 'roll 1 die #' in line)
                    self.assertEqual(len(requests), 2 + index)
                    self.assertEqual(len(set(requests)), len(requests))
                    self.assertEqual(server._window_ask_lines(call), [])
                    self.roll(server, [value])
                server.driver._attempt_chat_action(action, via='chat')
                self.assertEqual(server.rand.pending().kind, 'tile')
                tile = next(t for t in server.driver.es.hunt_pool if t.is_eye)
                server.rand.feed('tile', tile)
                result = server.driver._attempt_chat_action(action, via='chat')
                self.assertEqual(server.driver.es.fellowship.progress, 1)
                self.assertTrue(server.driver.es.dice[FP][0].used)
                self.assertEqual(server.driver.es.pending_hunt['damage'], 2 if discard else 1)
                self.assertFalse(any(kind == 'progress_sweep_emitted'
                                     for kind, _ in server.driver.findings),
                                 (result, server.driver.findings[-6:]))

    def test_observed_ai_progress_requires_exact_pending_and_physical_payment(self):
        from live.crebain_gestures import observed_ai_progress
        server, dp = self.rig(True, False, ai_side=FP)
        server._ai_decide(dp)
        pre = server.driver.es
        action = next(a for a in ai.describe_pending(pre)['actions']
                      if a.params['choices']['crebain_discard'] is False)
        post = ai.clone_state(pre)
        post.fellowship.progress += 1
        post.dice[FP][0].used = True
        server.driver.view.piece_region.update({
            '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
        self.assertIsNotNone(observed_ai_progress(server.driver, pre, action, post))
        for pid, wrong in [('248', 'Fellowship Track 0'), ('248', 'Fellowship Track 2'),
                           ('210', 'FP Dice')]:
            with self.subTest(pid=pid, zone=wrong):
                original = server.driver.view.piece_region[pid]
                server.driver.view.piece_region[pid] = wrong
                self.assertIsNone(observed_ai_progress(server.driver, pre, action, post))
                server.driver.view.piece_region[pid] = original
        wrong = copy.deepcopy(action)
        wrong.params['card_id'] = 'unrelated-card'
        self.assertIsNone(observed_ai_progress(server.driver, pre, wrong, post))
        post.fellowship.progress = 2
        self.assertIsNone(observed_ai_progress(server.driver, pre, action, post))

    def test_native_combined_title_discard_matches_without_replaying_ai_move(self):
        variants = {
            'current': ([], 'Current Action'),
            'board4': ([], 'Board4'),
            'lift_then_kill': ([('Board4', 'Current SA Card')], 'Current Action'),
            'other_current': ([('Board4', 'Current FP Card')], 'Current Action'),
            'several_public_slots': ([('Board4', 'Board5'), ('Board5', 'Current FP Card'),
                                      ('Current FP Card', 'Board6'), ('Board6', 'Current SA Card')],
                                     'Current Action'),
        }
        for resync, variant in itertools.product((False, True), variants):
            with self.subTest(resync=resync, variant=variant):
                server, dp = self.rig(True, True, ai_side=FP)
                server._ai_decide(dp)
                server.driver.view.piece_region.update({
                    '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
                if resync:
                    self.write_native_dump(server)
                    server.driver.reset_and_resync()
                transits, zone = variants[variant]
                for origin, destination in transits:
                    self.feed_native(server,
                        '$Human moves Flocks of Crebain/They are Terrible from %s to %s.' % (origin, destination))
                    self.assertIsNone(server.rand.pending())
                    self.assertEqual(server.driver.es.fellowship.progress, 0)
                prefix = copy.deepcopy(server.driver.pending_buf.records) if server.driver.pending_buf else []
                record = '$Human kills Flocks of Crebain/They are Terrible in %s.' % zone
                self.feed_native(server, record)
                original = copy.deepcopy(server.driver.pending_buf.records)
                self.assertEqual(original[:-1], prefix)
                self.assertEqual(original[-1]['_table_discards'],
                                 ['SHADOW_Character_Flocks_of_Crebain'])
                call = server.rand.pending()
                self.assertEqual((call.kind, call.kw['ctx'], call.kw['side'], call.kw['need']),
                                 ('d6', 'hunt roll', SH, 3))
                first = server._window_ask_lines(call)
                self.assertTrue(any('roll 3 dice #' in line for line in first))
                self.assertEqual(server._window_ask_lines(call), [])
                self.feed_native(server, '<game> Human rolls 5 1 1 ')
                for value in (1, 5):
                    call = server.rand.pending()
                    self.assertEqual((call.kind, call.kw['ctx'], call.kw['need']),
                                     ('d6', 'hunt reroll', 1))
                    asks = server._window_ask_lines(call)
                    self.assertTrue(any('roll 1 die #' in line for line in asks), asks)
                    self.assertEqual(server._window_ask_lines(call), [])
                    self.feed_native(server, '<game> Human rolls %d ' % value)
                self.assertEqual(server.rand.pending().kind, 'tile')
                self.assertEqual(server.driver.pending_buf.records, original)
                server.rand.feed('tile', next(t for t in server.driver.es.hunt_pool if t.is_eye))
                self.assertTrue(server.driver._try_pending(), server.driver._last_pending_match)
                post = server.driver.es
                self.assertEqual(post.fellowship.progress, 1)
                self.assertTrue(post.dice[FP][0].used)
                self.assertEqual(post.pending_hunt['damage'], 2)
                self.assertIsNone(post.persistent_cards.get_by_type(gs.PersistentCardType.FLOCKS_OF_CREBAIN))
                self.assertFalse(any(kind == 'progress_sweep_emitted'
                                     for kind, _ in server.driver.findings), server.driver.findings[-5:])
                self.assertTrue(server.driver.commits[-1][2].startswith('pending:card_option'))
                self.assertFalse(any('observed AI' in str(row) for row in server.driver.commits[-1][1]))
                self.assertEqual(server.rand.unconsumed()['d6'], 0)

    def test_repositioning_or_lifting_flocks_alone_does_not_choose_discard(self):
        for destination in ('Board5', 'Current SA Card', 'Current FP Card'):
            with self.subTest(destination=destination):
                server, dp = self.rig(True, True, ai_side=FP)
                server._ai_decide(dp)
                server.driver.view.piece_region.update({
                    '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
                before = server.driver.es
                line = ('$Human moves Flocks of Crebain/They are Terrible '
                        'from Board4 to %s.' % destination)
                self.feed_native(server, line)
                for _ in range(2):
                    server.driver._try_pending()
                self.assertIs(server.driver.es, before)
                self.assertIsNone(server.rand.pending())
                self.assertIsNotNone(before.persistent_cards.get_by_type(gs.PersistentCardType.FLOCKS_OF_CREBAIN))

    def test_unrelated_card_cannot_piggyback_on_flocks_transit(self):
        server, dp = self.rig(True, True, ai_side=FP)
        server._ai_decide(dp)
        server.driver.view.piece_region.update({
            '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
        before = server.driver.es
        for line in (
                '$Human moves The Palantir of Orthanc/Words of Power from Board4 to Current SA Card.',
                '$Human kills Flocks of Crebain/They are Terrible in Current Action.'):
            self.feed_native(server, line)
        self.assertIs(server.driver.es, before)
        self.assertIsNone(server.rand.pending())
        self.assertEqual(len(server.driver.pending_buf.records), 2)

    def test_unproved_physical_discard_holds_before_requesting_hunt_dice(self):
        cases = [('248', None, 'Flocks of Crebain/They are Terrible'),
                 ('248', 'Fellowship Track 2', 'Flocks of Crebain/They are Terrible'),
                 ('210', 'FP Dice', 'Flocks of Crebain/They are Terrible'),
                 (None, None, 'The Palantir of Orthanc/Words of Power')]
        for pid, zone, title in cases:
            with self.subTest(pid=pid, zone=zone, title=title):
                server, dp = self.rig(True, True, ai_side=FP)
                server._ai_decide(dp)
                server.driver.view.piece_region.update({
                    '248': 'Fellowship Track 1', '210': 'The Hunt For The Ring'})
                if pid:
                    if zone is None:
                        server.driver.view.piece_region.pop(pid)
                    else:
                        server.driver.view.piece_region[pid] = zone
                before = server.driver.es
                server.driver._ingest_line('$Human kills %s in Current Action.' % title, None)
                self.assertIs(server.driver.es, before)
                self.assertIsNone(server.rand.pending())
                if pid:
                    self.assertIsNotNone(server.driver.pending_buf)
                    self.assertEqual(len(server.driver.pending_buf.records), 1)

    def test_human_fellowship_transport_witness_survives_retained_resync(self):
        server, dp = self.rig(True, False)
        witness = server.driver._crebain_human_transport
        before = server.driver.es
        self.write_native_dump(server)
        for _ in range(2):
            server.driver.reset_and_resync()
            self.assertIs(server.driver.es, before)
            self.assertIs(server.driver._crebain_human_transport, witness)
        self.expect_roll(server, dp, 3, 'hunt roll', ())
        self.roll(server, [6, 6, 6])
        server._ai_decide(dp)
        server.rand.feed('tile', next(t for t in server.driver.es.hunt_pool if t.is_eye))
        server._ai_decide(dp)
        self.assertEqual(server.driver.es.fellowship.progress, 1)
        self.assertIsNone(server.driver._crebain_human_transport)
        self.assertFalse(any(kind == 'progress_sweep_emitted'
                             for kind, _ in server.driver.findings), server.driver.findings[-5:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
