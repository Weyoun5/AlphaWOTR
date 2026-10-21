"""Rules-based literal Hunt gestures: choosing a guide is not a random draw.

These are explicitly constructed interaction controls, not archived play.
Optional JSON diagnostics retain the actual ClientRandomness feed/cursors.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import game_state as gs
from live.driver import LiveDriver
from live.randomness import ClientRandomness
from test_audit0904_bridge_followup import initial_action_phase

if os.environ.get('WOTR_AUDIT_MIXED_HUNT_BASELINE'):
    spec = importlib.util.spec_from_file_location(
        'live._audit_before_mixed_hunt', os.environ['WOTR_AUDIT_MIXED_HUNT_BASELINE'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LiveDriver = module.LiveDriver

CORRUPTION = '$Wey moves Corruption from Fellowship Track 0 to Fellowship Track 1.'
GUIDE_KILL = '$Wey moves Boromir from Guide Of The Fellowship to FP Casualties.'
MERRY_GUIDE = '$Wey moves Meriadoc from Fellowship Box to Guide Of The Fellowship.'


class HuntChoiceEvidenceTests(unittest.TestCase):
    diagnostics = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get('WOTR_AUDIT_MIXED_HUNT_LOG')
        if path:
            Path(path).write_text(json.dumps(cls.diagnostics, indent=2) + '\n', encoding='utf-8')

    def rig(self, names=('Boromir', 'Meriadoc', 'Peregrin')):
        state = initial_action_phase()
        state.fellowship.companions[:] = [c for c in state.fellowship.companions if c.name in names]
        state.fellowship.guide = next(c for c in state.fellowship.companions if c.name == 'Boromir')
        state.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
        temp = tempfile.TemporaryDirectory(prefix='wotr-hunt-choice-')
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        rand = ClientRandomness(seed=13)
        rand.install()
        driver = LiveDriver(state, gs.Side.SHADOW, str(path), randomness=rand,
                            nick='WOTR_AI_Shadow')
        driver.phase = 'actions'
        driver.audit_committed_chains = []
        commit = driver._commit_chain

        def record_commit(chain, *args, **kwargs):
            result = commit(chain, *args, **kwargs)
            if result:
                driver.audit_committed_chains.append([a.action_type for a in chain])
            return result

        driver._commit_chain = record_commit
        return driver, path, rand

    def feed(self, driver, path, rand, lines):
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')
        out = driver.tick()
        pending = rand.pending()
        self.diagnostics.append({
            'test': self.id(), 'lines': lines,
            'rng_requests': [item[1].kind for item in out if item[0] == 'rng_ask'],
            'rng_pending': None if pending is None else {'kind': pending.kind, 'kw': pending.kw},
            'companion_feed': list(rand._feed['companion']),
            'companion_cursor': rand._feed_pos['companion'],
            'unconsumed': rand.unconsumed(), 'journal': list(map(repr, rand._journal)),
            'pending_hunt': driver.es.pending_hunt,
            'corruption': driver.es.fellowship.corruption,
            'companions': [c.name for c in driver.es.fellowship.companions],
            'guide': driver.es.fellowship.guide.name,
            'alarms': list(driver.alarms),
            'committed_chains': list(driver.audit_committed_chains),
        })
        return out

    def assert_no_random_request(self, out, rand):
        self.assertFalse(any(i[0] == 'rng_ask' and i[1].kind == 'randomcomp' for i in out))
        self.assertFalse(rand.pending() is not None and rand.pending().kind == 'randomcomp')

    def test_explicit_guide_casualty_waits_for_named_guide_without_rng(self):
        driver, path, rand = self.rig()
        self.feed(driver, path, rand, [CORRUPTION])
        out = self.feed(driver, path, rand, [GUIDE_KILL])
        self.assert_no_random_request(out, rand)
        self.assertIsNotNone(driver.es.pending_hunt)
        out = self.feed(driver, path, rand, [MERRY_GUIDE])
        self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.fellowship.corruption, 1)
        self.assertEqual([c.name for c in driver.es.fellowship.companions], ['Meriadoc', 'Peregrin'])
        self.assertEqual(driver.es.fellowship.guide.name, 'Meriadoc')
        self.assertIsNone(driver.es.pending_hunt)
        self.assertFalse(any(rand.unconsumed().values()))
        self.assertEqual(driver.alarms, [])

    def test_guide_first_order_and_full_batch_need_no_random_draw(self):
        for groups in (([GUIDE_KILL], [CORRUPTION], [MERRY_GUIDE]),
                       ([CORRUPTION, GUIDE_KILL, MERRY_GUIDE],)):
            with self.subTest(groups=groups):
                driver, path, rand = self.rig()
                for lines in groups:
                    out = self.feed(driver, path, rand, lines)
                    self.assert_no_random_request(out, rand)
                self.assertEqual(driver.es.fellowship.corruption, 1)
                self.assertIsNone(driver.es.pending_hunt)
                self.assertFalse(any(rand.unconsumed().values()))
                self.assertEqual(driver.alarms, [])

    def test_explicit_random_chat_still_requests_and_uses_client_draw(self):
        driver, path, rand = self.rig(('Boromir', 'Gimli', 'Meriadoc', 'Peregrin'))
        out = self.feed(driver, path, rand, ['<Wey> random'])
        self.assertEqual([i[1].kind for i in out if i[0] == 'rng_ask'], ['randomcomp'])
        self.assertEqual(driver.es.fellowship.corruption, 0)
        self.assertIn('Gimli', [c.name for c in driver.es.fellowship.companions])
        out = self.feed(driver, path, rand, ['$Wey draws Gimli.'])
        self.assert_no_random_request(out, rand)
        self.assertNotIn('Gimli', [c.name for c in driver.es.fellowship.companions])
        self.feed(driver, path, rand, [
            '$Wey moves Gimli from Fellowship Box to FP Casualties.', CORRUPTION])
        self.assertEqual(driver.es.fellowship.corruption, 1)
        self.assertIsNone(driver.es.pending_hunt)
        self.assertFalse(any(rand.unconsumed().values()))
        self.assertEqual(driver.alarms, [])

    def test_client_draw_before_named_casualty_is_consumed_without_another_draw(self):
        driver, path, rand = self.rig(('Boromir', 'Gimli', 'Meriadoc', 'Peregrin'))
        out = self.feed(driver, path, rand, ['$Wey draws Gimli.'])
        self.assert_no_random_request(out, rand)
        self.assertEqual(rand.unconsumed()['companion'], 1)
        out = self.feed(driver, path, rand, [
            '$Wey moves Gimli from Fellowship Box to FP Casualties.', CORRUPTION])
        self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.fellowship.corruption, 1)
        self.assertNotIn('Gimli', [c.name for c in driver.es.fellowship.companions])
        self.assertIn('Boromir', [c.name for c in driver.es.fellowship.companions])
        self.assertIsNone(driver.es.pending_hunt)
        self.assertFalse(any(rand.unconsumed().values()))
        self.assertEqual(driver.alarms, [])

    def test_random_draw_of_current_guide_is_not_recast_as_deterministic_choice(self):
        driver, path, rand = self.rig()
        for line in ('$Wey draws Boromir.', CORRUPTION, GUIDE_KILL):
            out = self.feed(driver, path, rand, [line])
            self.assert_no_random_request(out, rand)
        self.assertEqual(rand._feed['companion'], [('Boromir', 'Boromir')])
        self.assertEqual(rand._feed_pos['companion'], 1)
        self.assertEqual(driver.n_commits_total, 0)
        for _ in range(3):
            # Exercise the driver's real pending retry entrypoint while
            # the engine's guide tie still has no expressed selection.
            driver._poke_pending_buffer()
            out = self.feed(driver, path, rand, [])
            self.assert_no_random_request(out, rand)
            self.assertEqual(rand._feed['companion'], [('Boromir', 'Boromir')])
            self.assertEqual(driver.n_commits_total, 0)
        out = self.feed(driver, path, rand, [MERRY_GUIDE])
        self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.fellowship.corruption, 1)
        self.assertIsNone(driver.es.pending_hunt)
        self.assertEqual(driver.es.fellowship.guide.name, 'Meriadoc')
        self.assertFalse(any(rand.unconsumed().values()))
        self.assertFalse(any(k == 'unconsumed_rng' for k, _ in driver.findings))
        self.assertTrue(any(chain[0] == 'hunt_sacrifice_random'
                            for chain in driver.audit_committed_chains))
        self.assertFalse(any('hunt_sacrifice_guide' in chain
                             for chain in driver.audit_committed_chains))
        self.assertEqual(driver.alarms, [])

    def test_explicit_random_chat_may_select_the_current_guide(self):
        driver, path, rand = self.rig()
        out = self.feed(driver, path, rand, ['<Wey> random'])
        self.assertEqual([i[1].kind for i in out if i[0] == 'rng_ask'], ['randomcomp'])
        out = self.feed(driver, path, rand, ['$Wey draws Boromir.'])
        self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.pending_hunt['damage'], 1)
        self.assertTrue(driver.es.pending_hunt['casualty_taken'])
        self.assertNotIn('Boromir', [c.name for c in driver.es.fellowship.companions])
        self.assertFalse(any(rand.unconsumed().values()))
        for line in (GUIDE_KILL, CORRUPTION, MERRY_GUIDE):
            out = self.feed(driver, path, rand, [line])
            self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.fellowship.corruption, 1)
        self.assertEqual(driver.es.fellowship.guide.name, 'Meriadoc')
        self.assertIsNone(driver.es.pending_hunt)
        self.assertFalse(any(k == 'unconsumed_rng' for k, _ in driver.findings))
        self.assertEqual(driver.alarms, [])

    def test_non_guide_removal_without_random_draw_cannot_request_or_invent_one(self):
        driver, path, rand = self.rig(('Boromir', 'Gimli', 'Meriadoc', 'Peregrin'))
        out = self.feed(driver, path, rand, [
            '$Wey moves Gimli from Fellowship Box to FP Casualties.'])
        self.assert_no_random_request(out, rand)
        self.assertEqual(driver.es.fellowship.corruption, 0)
        self.assertIn('Gimli', [c.name for c in driver.es.fellowship.companions])
        self.assertIsNotNone(driver.es.pending_hunt)
        self.assertEqual(driver.n_commits_total, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
