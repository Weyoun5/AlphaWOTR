#!/usr/bin/env python3
"""Audit 2026-08-29 regression pins - CLC (bridge cleanup wave).

CLC-1  live_server._handle_inner's '#BLIND' attestation check must be
       reachable: a request whose raw log carries the AiFeed attestation
       line ('#BLIND aiside=<side> hands=<sides>') is judged - the human
       side named in `hands` passes, any other attestation is the design
       rule 8 FAIL-CLOSED DriverHalt.  The check referenced `gs` with no
       game_state import in scope, so EVERY attested request raised
       NameError (surfacing as a server cycle error) instead of a verdict.

Offline: LiveServer with load_nets=False, the dict request shape the
suites use.  main() runs them all and prints TOTAL=<n> FAIL=<m>; exit 1 on
any failure.  cp1252 console: ASCII only.
"""

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

from game_state import Side  # noqa: E402
from wotr_bridge.live.live_server import LiveServer  # noqa: E402
from wotr_bridge.live.driver import DriverHalt  # noqa: E402


def _server(seed):
    td = tempfile.mkdtemp(prefix='clc_pin_')
    return LiveServer(td, Side.SHADOW, load_nets=False, seed=seed)


def test_blind_attestation_matching_hands_is_no_error():
    """CLC-1 (RED before the fix: NameError 'gs' out of handle())."""
    srv = _server(101)
    out = srv.handle({'side': 'LIVE_SHADOW',
                      'lines': ['#BLIND aiside=SA hands=FP']})
    assert isinstance(out, list), 'handle() must return a RESP list'


def test_blind_attestation_wrong_hands_is_fail_closed():
    """Design rule 8: the human (FP) hand not in the attested hands ->
    DriverHalt('BLINDING VIOLATION ...'), never a NameError, never play."""
    srv = _server(102)
    for att in ('#BLIND aiside=SA hands=SA', '#BLIND aiside=none hands=none'):
        try:
            srv.handle({'side': 'LIVE_SHADOW', 'lines': [att]})
        except DriverHalt as e:
            assert 'BLINDING VIOLATION' in str(e), (att, str(e))
        else:
            raise AssertionError('no DriverHalt for %r' % att)


def test_blind_attestation_ai_fp_seat():
    """The other seat: AI FP, human SA - 'hands=SA' passes, 'hands=FP'
    halts (the human side is derived from ai_side, never assumed)."""
    td = tempfile.mkdtemp(prefix='clc_pin_')
    srv = LiveServer(td, Side.FREE_PEOPLES, load_nets=False, seed=103)
    out = srv.handle({'side': 'LIVE_FREE_PEOPLES',
                      'lines': ['#BLIND aiside=FP hands=SA']})
    assert isinstance(out, list)
    try:
        srv.handle({'side': 'LIVE_FREE_PEOPLES',
                    'lines': ['#BLIND aiside=FP hands=FP']})
    except DriverHalt as e:
        assert 'BLINDING VIOLATION' in str(e), str(e)
    else:
        raise AssertionError('no DriverHalt for hands=FP with a human SA')


TESTS = [test_blind_attestation_matching_hands_is_no_error,
         test_blind_attestation_wrong_hands_is_fail_closed,
         test_blind_attestation_ai_fp_seat]


def main():
    fail = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception as e:  # noqa: BLE001 - the pin reports, never hides
            fail += 1
            print('FAIL %s: %s: %s' % (t.__name__, type(e).__name__,
                                       str(e)[:200]))
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fail))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
