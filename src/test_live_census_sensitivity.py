"""SENSITIVITY CONTROLS for the L3 census oracle (BL-148 triage,
2026-08-25).  The census comparator diff_engine_vs_dump is only
trustworthy if it FIRES on a real engine-vs-client divergence, not
just returns [] on agreement.  These prove it fires, and pin the
row classifier, so a future change to the comparator or the runner
cannot silently blind the L3 layer.
"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
FIXTURES = os.path.join(_LAB, 'fixtures', 'sessions')
for _p in (_SRC, _ROOT, os.path.join(_ROOT, 'wotr_bridge')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import test_live_session_census as C                     # noqa: E402


def test_comparator_fires_on_real_divergence(fails):
    """manual_release_20260824 froze a REAL divergence (BL-148: the
    Witch-king stranded at Eastemnet while its army marched to the
    Helm's Deep siege).  The comparator MUST report a minion row - a
    census that returned clean here would be blind to the exact class
    of bug the L3 layer exists to freeze."""
    d = os.path.join(FIXTURES, 'manual_release_20260824')
    if not os.path.exists(os.path.join(d, 'ai_live_state.pkl')):
        fails.append(('fixture_missing', 'manual_release_20260824'))
        return
    r = C.census_fixture('manual_release_20260824')
    if r.get('error'):
        fails.append(('census_error', r['error']))
        return
    board = r['board']
    if not any(b.startswith('minion') for b in board):
        fails.append(('no_minion_row', board[:4]))
    if not any(b.startswith('leaders') for b in board):
        fails.append(('no_leaders_row', board[:4]))


def test_row_classifier(fails):
    """dice/deck -> staging (async writers); every game-piece field
    -> board (the signal)."""
    staging = ['dice fp pool: engine=2 client=1',
               'dice shadow hunt: engine=1 client=0',
               'deck fp strat deck: engine=23 client=22']
    board = ['leaders edoras (fp): engine=1 client=0',
             'units dale the north/regular: engine=0 client=1',
             'companion Gandalf the Grey: engine=dead/out, client zone=guide box',
             'minion witch-king (253): engine@eastemnet client@helm s deep',
             'nazgul moria: engine=1 client=0',
             'siege lorien: garrison-side piece 097 OUTSIDE the stronghold',
             'control sh: client marker at dale, engine does not']
    for row in staging:
        if C.classify_row(row) != 'staging':
            fails.append(('staging_misclassified', row))
    for row in board:
        if C.classify_row(row) != 'board':
            fails.append(('board_misclassified', row))


def test_signed_known_fixtures_still_diverge(fails):
    """A fixture signed {"census":"known"} froze a real divergence;
    it must STILL show its recorded board-row count (a frozen pickle
    is deterministic).  If it went clean, the recording changed under
    us - a ratchet to investigate, surfaced as a fail here."""
    for name in ('shgame4', 'fpgame15', 'manual_release_20260824'):
        exp = C.expected_for(name)
        if not exp or exp.get('census') != 'known':
            continue
        r = C.census_fixture(name)
        nb = len(r.get('board', [])) + len(r.get('invariants', []))
        if nb == 0:
            fails.append(('known_went_clean', (name, exp.get('bug'))))
        elif exp.get('rows') is not None and nb != exp['rows']:
            fails.append(('known_count_drift',
                          (name, nb, exp['rows'])))


def test_dynamic_types_absence_is_tracked(fails):
    """H1: the runner records whether the piece-swap map was present.
    Existing fixtures predate the freeze-side persistence, so absence
    is expected and must be VISIBLE (never silently assumed benign)."""
    r = C.census_fixture('shgame4')
    if 'dyn_absent' not in r:
        fails.append(('dyn_flag_missing', list(r)))


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    n_fail = 0
    for t in tests:
        fails = []
        try:
            t(fails)
        except Exception as e:
            import traceback
            fails.append(('exception', repr(e), traceback.format_exc()[-300:]))
        print('  [%s] %s' % ('PASS' if not fails else 'FAIL', t.__name__))
        for f in fails:
            print('    FAIL: %r' % (f,))
        n_fail += bool(fails)
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), n_fail))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
