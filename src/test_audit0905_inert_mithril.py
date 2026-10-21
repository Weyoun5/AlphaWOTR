"""Printed inert card placement and real use boundaries, shared implementation."""
from pathlib import Path
import os
import sys
import unittest
SOURCE = Path(__file__).resolve().parent
FIXTURES = SOURCE.parent / 'audit/2026-09-04/decision_pass/hunt_pool_observation_review'
os.environ['WOTR_MITHRIL_SOURCE'] = str(SOURCE)
sys.path.insert(0, str(FIXTURES))
from test_inert_mithril import InertMithrilControls

if __name__ == '__main__':
    unittest.main(verbosity=2)
