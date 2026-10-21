"""Real native chosen-card and Ring feed regressions, not emitter round trips.

Captured by the private-desktop packaged Game/hook on 2026-09-05. The House
fixture starts with Boromir in Minas Tirith and the named card in hand.
Expected recruitment/draw order follows Almanac16/46; ring timing follows
RB21/Almanac31-32. The original failing Java captures remain immutable.
"""
import ast
import tokenize
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import ai
import test_audit0904_stewards_bridge as stewards
import test_audit0904_elven_ring_bridge as rings
from live import chat
from live.sidecar import SidecarReader

FIXTURE = ROOT / 'wotr_bridge/tests/fixtures/house_ring_native_20260905'


class NativeHouseRingAcceptance(unittest.TestCase):
    diagnostics = []

    def house_rig(self):
        state = stewards.house.before_play(face=stewards.house.gs.ActionDieFace.EVENT,
                                          fillers=[stewards.house.S3])
        driver, path = stewards.StewardsBridgeAcceptance.rig(self, state=state)
        driver.sidecar = SidecarReader(str(path) + '.meta.jsonl')
        return driver, path

    def native_house_prefix(self, driver, path, split, wrong_unit=False):
        lines = (FIXTURE / 'house_ai_feed.txt').read_text(encoding='utf-8').splitlines()
        start = next(i for i, line in enumerate(lines) if 'moves Free Action Die (P)' in line)
        meta = {r['seq']: r for r in map(json.loads, (FIXTURE / 'house_ai_feed.txt.meta.jsonl').read_text().splitlines())}
        selected = lines[start:]
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[2], '<auto> moves House of the Stewards/Brave Stand from Free Peoples Hand (chosen) to Current FP Card.')
        if wrong_unit:
            selected[3] = selected[3].replace('Gondor Regular', 'Gondor Leader')
        output = []
        for offset, line in enumerate(selected):
            row = dict(meta[start + offset + 1]) # #BLIND attestation counts as seq1
            if offset == 0:
                self.assertEqual(row['pids'], ['210'])
            elif offset == 2:
                self.assertIn('Controls.861', row['tags'])
                self.assertEqual(row['pids'], 'redacted')
            if offset == 3 and wrong_unit:
                row['pids'] = ['270']
                row['segs'][1] = ['270']
            row['seq'] = offset + 1
            with Path(str(path) + '.meta.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            with path.open('a', encoding='utf-8') as stream:
                stream.write(line + '\n')
            if split:
                driver.sidecar.poll()
                output += driver.tick()
        driver.sidecar.poll()
        output += driver.tick()
        return output

    def test_native_chosen_house_completes_before_die_cleanup(self):
        for split in (False, True):
            with self.subTest(split=split):
                driver, path = self.house_rig()
                char_before = list(driver.es.fp_char_deck)
                self.native_house_prefix(driver, path, split)
                self.assertEqual(stewards.house.figures(driver.es), (4, 1, 1))
                hand = driver.es.hands[stewards.house.FP].hand
                self.assertEqual(len(hand), 3)
                self.assertIn(stewards.house.S3, hand)
                # Native human draw metadata is intentionally blinded. The
                # two receipts are distinct Strategy placeholders until a
                # commitment/reveal supplies their identities.
                hidden = [c for c in hand if c != stewards.house.S3]
                self.assertEqual(len(set(hidden)), 2)
                self.assertTrue(all(c.startswith('HU_FR_STRAT_') for c in hidden))
                self.assertEqual(driver.es.pending_card_option['card_id'], '__gandalf_draw__')
                self.assertEqual(driver.es.fp_char_deck, char_before)
                self.assertIsNotNone(driver.seg.window)
                stewards.StewardsBridgeAcceptance.feed(self, driver, path, '<AuditPlayer> decline')
                commits = driver.n_commits_total
                stewards.StewardsBridgeAcceptance.feed(self, driver, path,
                    '$AuditPlayer kills group (Free Action Die (P), House of the Stewards/Brave Stand) in Current Action.')
                self.assertEqual(driver.n_commits_total, commits)
                self.assertIsNone(driver.seg.window)
                self.assertIsNone(ai.describe_pending(driver.es))
                self.assertEqual(driver.es.fp_char_deck, char_before)
                stewards.StewardsBridgeAcceptance.assert_paid(self, driver)
                self.assertFalse(any(driver.rand.unconsumed().values()))

    def test_native_chosen_house_wrong_figure_does_not_commit_by_title(self):
        driver, path = self.house_rig()
        before = list(driver.es.hands[stewards.house.FP].hand)
        self.native_house_prefix(driver, path, split=True, wrong_unit=True)
        self.assertEqual(stewards.house.figures(driver.es), (3, 1, 1))
        self.assertEqual(driver.es.hands[stewards.house.FP].hand, before)
        self.assertFalse(driver.es.dice[stewards.house.FP][0].used)
        self.assertIsNone(getattr(driver.es, 'pending_card_option', None))

    def test_native_house_gandalf_prompt_passes_actual_chat_policy(self):
        driver, path = self.house_rig()
        output = self.native_house_prefix(driver, path, split=True)
        prompts = [item for item in output if item[:2] == ('chat', 'prompt')
                   and 'Gandalf' in item[2]]
        self.assertEqual(len(prompts), 1)
        for _, kind, text in prompts:
            chat.ChatOut().say(kind, text)
        self.assertIn('Character', prompts[0][2])
        self.assertIn('decline', prompts[0][2])

    def test_ring_counter_prompt_passes_actual_chat_policy(self):
        driver, path = rings.ElvenRingAcceptance.rig(self)
        output = rings.ElvenRingAcceptance.feed(self, driver, path,
            '$AuditPlayer moves Vilya from Vilya Slot to Used Ring 1.')
        prompts = [item for item in output if item[:2] == ('chat', 'prompt')]
        self.assertEqual(len(prompts), 1)
        sink = chat.ChatOut()
        for _, kind, text in prompts:
            sink.say(kind, text)
        self.assertIn('selected unused die', sink.sent[0][1])
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual(driver.es.fp_elven_rings, 3)

    def test_ring_face_first_prompt_passes_actual_chat_policy(self):
        driver, path = rings.ElvenRingAcceptance.rig(self)
        output = rings.ElvenRingAcceptance.feed(self, driver, path,
            '$AuditPlayer changes Free Action Die (C) to H')
        prompts = [item for item in output if item[:2] == ('chat', 'prompt')]
        self.assertEqual(len(prompts), 1)
        for _, kind, text in prompts:
            chat.ChatOut().say(kind, text)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertFalse(driver.es.fp_ring_used_this_turn)

    def test_counter_then_pass_is_held_with_a_sanctioned_prompt(self):
        driver, path = rings.ElvenRingAcceptance.rig(self)
        rings.ElvenRingAcceptance.feed(self, driver, path, rings.RING)
        output = rings.ElvenRingAcceptance.feed(self, driver, path, '<Wey> pass')
        prompts = [item for item in output if item[:2] == ('chat', 'prompt')]
        self.assertEqual(len(prompts), 1)
        for _, kind, text in prompts:
            chat.ChatOut().say(kind, text)
        self.assertIn('undo', prompts[0][2])
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual(driver.es.fp_elven_rings, 3)
        self.assertFalse(driver.es.fp_ring_used_this_turn)

    def test_driver_and_server_do_not_emit_literal_unregistered_prompts(self):
        violations = []
        for name in ('driver', 'live_server'):
            with tokenize.open(ROOT / 'wotr_bridge/live' / (name + '.py')) as source:
                tree = ast.parse(source.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Tuple) and len(node.elts) >= 3:
                    first = node.elts[:2]
                    if all(isinstance(e, ast.Constant) for e in first) and [e.value for e in first] == ['chat', 'prompt']:
                        if isinstance(node.elts[2], (ast.Constant, ast.JoinedStr)):
                            violations.append((name, node.lineno))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_say' and len(node.args) >= 2:
                    if isinstance(node.args[0], ast.Constant) and node.args[0].value == 'prompt' and isinstance(node.args[1], (ast.Constant, ast.JoinedStr)):
                        violations.append((name, node.lineno))
        self.assertEqual(violations, [], 'Prompts must use reviewed ChatOut builders')

    def test_unregistered_prompt_still_rejected(self):
        with self.assertRaises(chat.ChatPolicyError):
            chat.ChatOut().say('prompt', 'arbitrary unregistered generated instruction')


if __name__ == '__main__':
    unittest.main(verbosity=2)
