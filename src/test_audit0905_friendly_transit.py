"""Printed movement rules, saved expert positions, and original-cargo controls."""
from pathlib import Path
import importlib.util
import os
import sys
import unittest
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'src'
REVIEW = ROOT / 'audit/2026-09-04/decision_pass/encoded_menu_review'
os.environ.update(WOTR_ACT_GEN='v45', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1',
                  WOTR_TRANSIT_SOURCE=str(SOURCE), WOTR_STAGED_SOURCE=str(SOURCE),
                  WOTR_CHRONOLOGY_SOURCE=str(SOURCE))
sys.path.insert(0, str(SOURCE))
cases = []
for index, relative in enumerate((
        'deferred_movement_review/global_tsl_rx_recovery/test_friendly_transit.py',
        'friendly_transit_r1/test_paths.py', 'friendly_transit_r2/test_join_capacity.py')):
    spec = importlib.util.spec_from_file_location('literal_transit_' + str(index), REVIEW / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases.append(unittest.defaultTestLoader.loadTestsFromModule(module))
import ai
import event_effects
assert Path(ai.__file__).resolve().parent == SOURCE
assert Path(event_effects.__file__).resolve().parent == SOURCE

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))
    raise SystemExit(not result.wasSuccessful())
