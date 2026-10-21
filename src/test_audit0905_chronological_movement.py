"""Literal expert drags and ambiguity controls against the shared extractor."""
from pathlib import Path
import importlib.util
import os
import sys
import unittest
SOURCE = Path(__file__).resolve().parent
FIXTURES = SOURCE.parent / 'audit/2026-09-04/decision_pass/encoded_menu_review/deferred_movement_review'
os.environ.update(WOTR_STAGED_SOURCE=str(SOURCE), WOTR_GLOBAL_SOURCE=str(SOURCE), WOTR_CHRONOLOGY_SOURCE=str(SOURCE))
sys.path.insert(0, str(FIXTURES / 'chronological_rx_recovery'))
import test_chronology as prior
assert Path(prior.ai.__file__).resolve().parent == SOURCE
assert Path(prior.movement.__file__).resolve().parent == SOURCE


class CurrentChronologyControls(prior.ChronologicalControls):
    def test_identical_original_and_incoming_group_stays_ambiguous(self):
        bundle = prior.captured(62, 131)
        bundle['batch']['records'] = [r for r in bundle['batch']['records'] if r.get('line') not in (133, 134)]
        with self.assertRaisesRegex(prior.movement.Unproven, 'tsl_origin_or_through_ambiguous'):
            prior.plan(bundle)


spec = importlib.util.spec_from_file_location('literal_global_controls', FIXTURES / 'global_tsl_rx_recovery/test_global_bound.py')
global_controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_controls)
GlobalBoundControls = global_controls.GlobalBoundControls

if __name__ == '__main__':
    unittest.main(verbosity=2)
