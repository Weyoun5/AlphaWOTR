"""Real feed regressions for declaration completion with one eligible Guide.

The optional baseline-driver argument loads a frozen complete driver module for
failure evidence. It does not replace the engine's menus, actions or matcher.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
START_DIRECTORY = Path.cwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['WOTR_NATIVE'] = '0'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import ai
import game_logic as gl
from board import sanitize
from game_state import Side
import test_live_driver as fixture

FP, SH = Side.FREE_PEOPLES, Side.SHADOW
DRIVER_SOURCE = ROOT / 'wotr_bridge/live/driver.py'


def status(driver):
    pending = ai.describe_pending(driver.es)
    return dict(phase=driver.phase, region=sanitize(str(gl.fellowship_region(driver.es))),
                progress=driver.es.fellowship.progress, hidden=driver.es.fellowship.hidden,
                pending=pending['kind'] if pending else None,
                guide=driver.es.fellowship.guide.name,
                guide_choice=copy.deepcopy(driver.es.pending_guide_change),
                eye_phase=bool(getattr(driver.es, '_eye_allocation_phase', False)),
                commits=driver.n_commits_total,
                labels=[commit[2] for commit in driver.commits],
                buffered=len(driver.pending_buf.records) if driver.pending_buf else 0)


class DeclarationCompletion(unittest.TestCase):
    records = []

    def rig(self, roster='strider', progress=4, phase='declare'):
        state = fixture.fresh_state()
        state.turn = 3
        state.fellowship.progress = progress
        state.fellowship.hidden = True
        if roster == 'strider':
            state.fellowship.companions = [c for c in state.fellowship.companions
                                          if c.name != 'Gandalf the Grey']
            gl.change_guide(state, chosen='Strider')
            self.assertEqual({c.name for c in state.fellowship.companions},
                             {'Strider', 'Legolas', 'Gimli', 'Boromir', 'Meriadoc', 'Peregrin'})
        elif roster == 'gandalf':
            state.fellowship.companions = [c for c in state.fellowship.companions if c.name != 'Strider']
            gl.change_guide(state, chosen='Gandalf the Grey')
        else:
            self.assertEqual(roster, 'two')
        rig = fixture.Rig(state, SH, phase=phase, seed=1910)
        self.addCleanup(shutil.rmtree, rig.td, ignore_errors=True)
        rig.drv._ensure_declare_armed()
        return rig

    def send(self, rig, lines, batched):
        if batched:
            return rig.feed(*lines)
        output = []
        for line in lines:
            output.extend(rig.feed(line))
        return output

    def assert_closed_once(self, rig, lines, batched, label, destination='dimrill dale'):
        driver = rig.drv
        initial = status(driver)
        if batched:
            output = rig.feed(*(lines + ['<Wey> fp ok']))
            before_ok = None  # No poll boundary exists inside this transcript.
            close = []
        else:
            output = self.send(rig, lines, False)
            before_ok = status(driver)
            close = rig.feed('<Wey> fp ok')
        after_ok = status(driver)
        repeat = rig.feed('<Wey> fp ok')
        after_repeat = status(driver)
        record = dict(label=label, batched=batched, transcript=lines,
                      initial=initial, before_ok=before_ok, after_ok=after_ok,
                      after_repeat=after_repeat,
                      refusals=[item[2] for item in output + close + repeat
                                if item[0] == 'chat' and item[1] == 'refusal'],
                      ai_eyes=sum(item[0] == 'ai_eyes' for item in output + close + repeat))
        self.records.append(record)
        self.assertEqual(after_ok['region'], destination, record)
        self.assertEqual(after_ok['progress'], 0, record)
        self.assertTrue(after_ok['hidden'], record)
        self.assertEqual(after_ok['phase'], 'eyes', record)
        self.assertEqual(after_ok['pending'], 'eye', record)
        self.assertIsNone(after_ok['guide_choice'], record)
        self.assertTrue(after_ok['eye_phase'], record)
        self.assertEqual(after_ok['guide'], initial['guide'], record)
        self.assertEqual(after_repeat, after_ok, record)
        # Rig has no server consuming the scheduled AI step. Later ticks may
        # repeat its ready notification; phase/state/commit idempotence above
        # is the completion contract, not the number of scheduler reminders.
        self.assertGreaterEqual(record['ai_eyes'], 1, record)
        self.assertFalse(record['refusals'], record)
        self.assertEqual(after_ok['labels'].count('pending:declare'), 1, record)
        self.assertFalse(getattr(driver.es, '_declare_phase', False))
        self.assertFalse(driver.es.dice_rolled)

    def test_declared_single_guide_and_two_guide_control_complete_from_real_feed(self):
        for roster in ('strider', 'gandalf', 'two'):
            for gesture in ('flip_pair', 'progress_reset'):
                for batched in (False, True):
                    with self.subTest(roster=roster, gesture=gesture, batched=batched):
                        rig = self.rig(roster)
                        legal = ai.get_legal_actions(rig.drv.es, FP)
                        self.assertTrue(any(a.action_type == 'fellowship_declare'
                                            and a.params.get('region') == 'dimrill dale'
                                            for a in legal))
                        lines = ['$Wey moves The Fellowship from Rivendell to Dimrill Dale.']
                        if gesture == 'flip_pair':
                            lines += ['$Wey flips The Fellowship in Dimrill Dale. Previously on Fellowship Track 4.',
                                      '$Wey flips The Fellowship in Dimrill Dale.']
                        else:
                            lines += ['$Wey moves FSP from Fellowship Track 4 to Fellowship Track 0.']
                        self.assert_closed_once(rig, lines, batched, roster + ':' + gesture)

    def test_redundant_declaration_tail_is_idempotent_after_commit(self):
        move = '$Wey moves The Fellowship from Rivendell to Dimrill Dale.'
        flip = '$Wey flips The Fellowship in Dimrill Dale. Previously on Fellowship Track 4.'
        unflip = '$Wey flips The Fellowship in Dimrill Dale.'
        reset = '$Wey moves FSP from Fellowship Track 4 to Fellowship Track 0.'
        sequences = {
            'reset_after_flips': [move, flip, unflip, reset],
            'flips_after_reset': [move, reset, flip, unflip],
            'duplicate_reset': [move, reset, reset],
        }
        for roster in ('strider', 'two'):
            for name, lines in sequences.items():
                with self.subTest(roster=roster, sequence=name):
                    rig = self.rig(roster)
                    # Each feed call finishes its poll, so the last reset or
                    # flip arrives after declaration has already committed.
                    self.assert_closed_once(rig, lines, False, roster + ':' + name)
                    self.assertIsNone(rig.drv.pending_buf)

    def test_nonzero_progress_is_not_a_redundant_declaration_tail(self):
        for roster in ('strider', 'two'):
            with self.subTest(roster=roster):
                rig = self.rig(roster)
                rig.feed('$Wey moves The Fellowship from Rivendell to Dimrill Dale.')
                rig.feed('$Wey moves FSP from Fellowship Track 4 to Fellowship Track 0.')
                before = status(rig.drv)
                output = rig.feed('$Wey moves FSP from Fellowship Track 0 to Fellowship Track 1.')
                output += rig.feed('<Wey> fp ok')
                after = status(rig.drv)
                refusals = [item[2] for item in output if item[0:2] == ('chat', 'refusal')]
                record = dict(label='nonzero_tail:' + roster, before=before, after=after,
                              refusals=refusals)
                self.records.append(record)
                self.assertEqual(after['region'], 'dimrill dale', record)
                self.assertEqual(after['progress'], 0, record)
                self.assertEqual(after['labels'].count('pending:declare'), 1, record)
                self.assertEqual(after['commits'], before['commits'], record)
                self.assertTrue(refusals, record)
                if roster == 'two':
                    self.assertEqual(after['phase'], 'guide', record)
                    self.assertEqual(after['guide_choice'], before['guide_choice'], record)
                    self.assertEqual(after['buffered'], 1, record)

    def test_unchanged_board_fp_ok_still_declines_with_one_or_two_guides(self):
        for roster in ('strider', 'gandalf', 'two'):
            with self.subTest(roster=roster):
                rig = self.rig(roster);before = status(rig.drv)
                out = rig.feed('<Wey> fp ok');after = status(rig.drv)
                second = rig.feed('<Wey> fp ok')
                self.records.append(dict(label='decline:' + roster, before=before, after=after))
                self.assertEqual(after['region'], 'rivendell')
                self.assertEqual(after['progress'], 4)
                self.assertEqual(after['phase'], 'eyes')
                self.assertEqual(after['pending'], 'eye')
                self.assertEqual(after['labels'].count('declare-decline'), 1)
                self.assertEqual(status(rig.drv), after)
                self.assertGreaterEqual(sum(item[0] == 'ai_eyes' for item in out + second), 1)

    def test_skip_ready_phase_closes_fp_without_allocating_shadow_eyes(self):
        for forced_eye in (False, True):
            with self.subTest(forced_eye=forced_eye):
                rig = self.rig();driver = rig.drv
                if forced_eye:
                    driver.es.fellowship.companions = [c for c in driver.es.fellowship.companions
                                                      if c.name == 'Strider']
                    driver.es._fp_used_hunt_box_last_turn = True
                rig.feed('$Wey moves The Fellowship from Rivendell to Dimrill Dale.',
                         '$Wey moves FSP from Fellowship Track 4 to Fellowship Track 0.')
                self.assertEqual(driver.phase, 'guide')
                self.assertEqual(ai.describe_pending(driver.es)['kind'], 'eye')
                self.assertIsNone(driver.es.pending_guide_change)
                choices = ai.get_legal_actions(driver.es, SH)
                if forced_eye:
                    self.assertEqual([(a.action_type, a.params.get('count')) for a in choices],
                                     [('eye_allocate', 1)])
                before = status(driver)
                dice_before = [(die.current, die.used) for die in driver.es.dice[SH]]
                eyes_before = driver.es.eyes_allocated
                items, message = driver.skip_open_decision()
                after = status(driver)
                _, repeated_message = driver.skip_open_decision()
                repeated = status(driver)
                record = dict(label='skip_ready_eye', forced_eye=forced_eye, before=before,
                              after=after, repeated=repeated, message=message,
                              repeated_message=repeated_message)
                self.records.append(record)
                self.assertEqual(after['phase'], 'eyes', record)
                self.assertEqual(after['pending'], 'eye', record)
                self.assertTrue(after['eye_phase'], record)
                self.assertEqual(after['commits'], before['commits'], record)
                self.assertEqual(driver.es.eyes_allocated, eyes_before, record)
                self.assertEqual([(die.current, die.used) for die in driver.es.dice[SH]], dice_before, record)
                self.assertEqual(repeated, after, record)
                self.assertTrue(any(item[0] == 'ai_eyes' for item in items), record)

    def test_invalid_destination_or_unfinished_progress_cannot_close(self):
        for label, destination, progress_to in (
                ('unreachable', 'Far Harad', 0), ('unfinished_reset', 'Dimrill Dale', 1)):
            for batched in (False, True):
                with self.subTest(label=label, batched=batched):
                    rig = self.rig();driver = rig.drv;before = status(driver)
                    if label == 'unreachable':
                        self.assertFalse(any(a.action_type == 'fellowship_declare'
                                             and a.params.get('region') == 'far harad'
                                             for a in ai.get_legal_actions(driver.es, FP)))
                    lines = ['$Wey moves The Fellowship from Rivendell to %s.' % destination,
                             '$Wey moves FSP from Fellowship Track 4 to Fellowship Track %d.' % progress_to]
                    output = self.send(rig, lines, batched) + rig.feed('<Wey> fp ok')
                    after = status(driver)
                    self.records.append(dict(label=label, batched=batched, before=before, after=after))
                    self.assertEqual(after['commits'], before['commits'])
                    self.assertEqual(after['region'], 'rivendell')
                    self.assertEqual(after['progress'], 4)
                    self.assertEqual(after['pending'], 'declare')
                    self.assertNotEqual(after['phase'], 'eyes')
                    self.assertTrue(any(item[0] == 'chat' and item[1] == 'refusal' for item in output))
                    self.assertFalse(any(item[0] == 'ai_eyes' for item in output))

    def test_compulsory_guide_replacement_is_not_optional_phase_keep(self):
        rig = self.rig('two');state = rig.drv.es
        gl.close_declare_phase(state)
        # A tie opened by the engine before the optional Fellowship-phase
        # choice is compulsory, even though the driver's phase says guide.
        gl.change_guide(state, allow_pending=True)
        rig.drv.phase = 'guide'
        self.assertFalse(getattr(state, '_eye_allocation_phase', False))
        self.assertEqual(ai.describe_pending(state)['kind'], 'guide_change')
        before = status(rig.drv)
        out = rig.feed('<Wey> fp ok')
        after = status(rig.drv)
        self.records.append(dict(label='compulsory_guide', before=before, after=after))
        self.assertEqual(after, before)
        self.assertTrue(any(item[0] == 'chat' and item[1] == 'refusal' for item in out))
        self.assertFalse(any(item[0] == 'ai_eyes' for item in out))

    def test_hunt_decision_stays_blocked_even_if_eye_phase_flag_is_set(self):
        for eye_ready in (False, True):
            with self.subTest(eye_ready=eye_ready):
                rig = self.rig();state = rig.drv.es
                gl.close_declare_phase(state)
                gl.continue_hunt_chain(state, 2, casualty_taken=False)
                state._eye_allocation_phase = eye_ready
                rig.drv.phase = 'guide'
                self.assertEqual(ai.describe_pending(state)['kind'], 'hunt')
                before = status(rig.drv);hunt = copy.deepcopy(state.pending_hunt)
                corruption = state.fellowship.corruption
                out = rig.feed('<Wey> fp ok');after = status(rig.drv)
                self.records.append(dict(label='hunt', eye_ready=eye_ready, before=before, after=after))
                self.assertEqual(after, before)
                self.assertEqual(rig.drv.es.pending_hunt, hunt)
                self.assertEqual(rig.drv.es.fellowship.corruption, corruption)
                self.assertTrue(any(item[0] == 'chat' and item[1] == 'refusal' for item in out))
                self.assertFalse(any(item[0] == 'ai_eyes' for item in out))


def main():
    global DRIVER_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'audit/bridge_declaration_oct19/regression.json')
    parser.add_argument('--baseline-driver', type=Path)
    args = parser.parse_args();args.out = (START_DIRECTORY / args.out).resolve()
    if args.baseline_driver:
        DRIVER_SOURCE = (START_DIRECTORY / args.baseline_driver).resolve()
        name = 'wotr_bridge.live._oct19_declaration_baseline'
        spec = importlib.util.spec_from_file_location(name, DRIVER_SOURCE)
        module = importlib.util.module_from_spec(spec);sys.modules[name] = module
        spec.loader.exec_module(module)
        fixture.LiveDriver = module.LiveDriver
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DeclarationCompletion))
    report = dict(passed=result.wasSuccessful(), methods=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  driver_source=str(DRIVER_SOURCE),
                  driver_sha256=hashlib.sha256(DRIVER_SOURCE.read_bytes()).hexdigest(),
                  records=DeclarationCompletion.records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    args.out.with_suffix('.log').write_text(stream.getvalue())
    print(json.dumps({key: len(value) if key == 'records' else value for key, value in report.items()}))
    if not result.wasSuccessful():
        print('Failure details: ' + str(args.out.with_suffix('.log')))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
