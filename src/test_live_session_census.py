"""L3 REAL-CAPTURE FIXTURES - the STATIC census oracle (TEST_STRATEGY
_SPEC.md section 3 L3).

Every seat session and manual game leaves TWO independent artifacts:

  ai_live_state.pkl    the ENGINE's GameState, pickled by live_server
  ai_client_board.txt  the CLIENT's own board export (written by the
                       Java client, a different process)

The honest oracle is a DIRECT diff of the two -
`client_board.diff_engine_vs_dump(pickle, dump)` - with NO replay and
NO randomness.  (A first cut replayed each feed from the initial
position under a fixed seed; that STARVES the lane of the real game's
client randomness - dice, hunt tiles, draws - so the board drifts and
census lights up on ~50 phantom rows.  That is the instrument lying,
the exact failure this spec exists to end.  The reconstruct()
diagnostic below is kept but is NOT an oracle.)

Row classes (a census row is one disagreement):
  BOARD  armies / leaders / companions / minions / control markers /
         political / rings - the game state itself; these are the
         signal a fixture's verdict is built on.
  STAGING the two ASYNC-written zones - action-dice (pool / used /
         hunt-box) and deck COUNTS.  The client export and the engine
         pickle are written by different processes at different
         instants, so a die or draw in flight shows here.  SEPARATED,
         never silently dropped (a drift over 2 is printed as
         staging(>2) for a human's eye). An unmatched staging row makes
         the snapshot UNVERIFIED: asynchronous sampling does not prove
         that a missing die or card is harmless. Everything else is BOARD.

EXPECTED.json in the fixture dir records the signed verdict:
  {"census": "clean"}                    BOARD rows must be 0
  {"census": "known", "rows": N, "bug": "BL-...", "note": "..."}
A fixture WITHOUT EXPECTED.json is UNVERIFIED, never PASS. A 'clean'
fixture that grows a BOARD row is a regression. A 'known' fixture with
its recorded discrepancy is KNOWN_FAIL; reaching zero requires review
of the expected record and is UNVERIFIED until then.

Run:  python test_live_session_census.py
      python test_live_session_census.py shgame21
"""
import argparse
import json
import os
import pickle
import re
import sys

_START_CWD = os.getcwd()
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
FIXTURES = os.path.join(_LAB, 'fixtures', 'sessions')
for _p in (_SRC, _ROOT, os.path.join(_ROOT, 'wotr_bridge')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from wotr_bridge.live import client_board as cb          # noqa: E402
import live_oracles as lo                                # noqa: E402

_STAGING_RE = re.compile(r'^(dice |deck )')
_MAG_RE = re.compile(r'engine=(\d+) client=(\d+)')


def classify_row(row):
    """'staging' for the two ASYNC-written zones - the action-dice
    pool/used/hunt-box staging and the deck COUNTS.  The client
    exports its board and the engine pickles as separate processes at
    separate instants, so a die or draw in flight shows here; these
    are reported with magnitude and make the capture unverified until
    independently synchronized. 'board' = everything else:
    armies, leaders, companions (guide included), minions, nazgul,
    control markers, sieges, political, rings - the game state itself,
    the signal a verdict is built on."""
    return 'staging' if _STAGING_RE.match(str(row)) else 'board'


def _staging_magnitude(row):
    m = _MAG_RE.search(str(row))
    return abs(int(m.group(1)) - int(m.group(2))) if m else None


def census_fixture(name, fixture_root=None):
    d = os.path.join(fixture_root or FIXTURES, name)
    pk = os.path.join(d, 'ai_live_state.pkl')
    dp = os.path.join(d, 'ai_client_board.txt')
    if not (os.path.exists(pk) and os.path.exists(dp)):
        return {'name': name, 'error': 'missing pkl or dump'}
    try:
        with open(pk, 'rb') as f:
            st = pickle.load(f)
    except Exception as e:
        return {'name': name, 'error': 'pickle: %r' % (e,)}
    # H1 (BL-148): decode the client dump under the captured piece-type
    # SWAP map when the freeze persisted one; without it a downgraded
    # elite reads as its static type.  Absent = noted, never silently
    # assumed benign.
    dt_path = os.path.join(d, 'ai_dynamic_types.json')
    dyn = None
    dyn_absent = True
    if os.path.exists(dt_path):
        try:
            with open(dt_path, encoding='utf-8') as f:
                dyn = json.load(f)
            dyn_absent = False
        except Exception:
            dyn = None
    try:
        rows = [str(r) for r in cb.diff_engine_vs_dump(
            st, cb.read_board(dp), dynamic_types=dyn)]
    except Exception as e:
        return {'name': name, 'error': 'census: %r' % (e,)}
    board = [r for r in rows if classify_row(r) == 'board']
    staging = [r for r in rows if classify_row(r) == 'staging']
    inv = lo.state_invariant_violations(st)
    big = [r for r in staging if (_staging_magnitude(r) or 0) > 2]
    return {'name': name, 'turn': getattr(st, 'turn', None),
            'board': board, 'staging': staging, 'staging_big': big,
            'invariants': inv, 'dyn_absent': dyn_absent}


def expected_for(name, fixture_root=None):
    p = os.path.join(fixture_root or FIXTURES, name, 'EXPECTED.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def verdict(r, expected):
    if r.get('error'):
        return 'ERROR', r['error']
    nb = len(r['board']) + len(r['invariants'])
    if expected is None:
        return 'UNVERIFIED', 'no independently reviewed EXPECTED.json (%d board/invariant rows)' % nb
    mode = expected.get('census')
    if mode == 'clean':
        if nb:
            return 'FAIL', 'clean fixture grew %d board/invariant row(s)' % nb
        if r.get('staging'):
            return 'UNVERIFIED', '%d unsynchronized dice/deck row(s)' % len(r['staging'])
        return 'PASS', None
    if mode == 'known':
        want = int(expected.get('rows', -1))
        if nb == 0:
            return 'UNVERIFIED', 'known-%s is now clean; review the expected record' % expected.get('bug')
        if want >= 0 and nb != want:
            return 'FAIL', 'known count %d != recorded %d' % (nb, want)
        return 'KNOWN_FAIL', '%s: %d recorded board/invariant row(s) remain' % (expected.get('bug', 'known discrepancy'), nb)
    return 'FAIL', 'unknown EXPECTED.census %r' % mode


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*')
    parser.add_argument('--fixtures', default=FIXTURES)
    parser.add_argument('--output', help='Optional JSON output; captures are never modified')
    args = parser.parse_args(argv)
    args.fixtures = os.path.abspath(os.path.join(_START_CWD, args.fixtures))
    if args.output:
        args.output = os.path.abspath(os.path.join(_START_CWD, args.output))
    if not os.path.isdir(args.fixtures):
        print('ERROR: fixture directory missing: %s' % args.fixtures)
        print('TOTAL 0 PASS 0 FAIL 1 SKIP 0')
        return 1
    names = sorted(n for n in os.listdir(args.fixtures)
                   if os.path.isdir(os.path.join(args.fixtures, n)))
    if args.names:
        missing = sorted(set(args.names) - set(names))
        if missing:
            print('ERROR: requested fixtures missing: %s' % ', '.join(missing))
            print('TOTAL 0 PASS 0 FAIL 1 SKIP 0')
            return 1
        names = [n for n in names if n in args.names]
    if not names:
        print('ERROR: no session fixtures found')
        print('TOTAL 0 PASS 0 FAIL 1 SKIP 0')
        return 1
    n_fail = 0
    n_clean = 0
    n_unverified = 0
    n_known = 0
    summary = []
    for name in names:
        r = census_fixture(name, args.fixtures)
        v, why = verdict(r, expected_for(name, args.fixtures))
        n_fail += v in ('FAIL', 'ERROR', 'KNOWN_FAIL')
        n_clean += v == 'PASS'
        n_unverified += v == 'UNVERIFIED'
        n_known += v == 'KNOWN_FAIL'
        if r.get('error'):
            print('  [%s] %-24s %s' % (v, name, r['error']))
        else:
            print('  [%s] %-24s turn=%s board=%d staging=%d inv=%d%s'
                  % (v, name, r['turn'], len(r['board']), len(r['staging']),
                     len(r['invariants']), ('  <- ' + why) if why else ''))
            for sb in r.get('staging_big', [])[:2]:
                print('        staging(>2): %s' % sb[:120])
            for b in r['board'][:4]:
                print('        board: %s' % b[:130])
            for k, dd in r['invariants'][:3]:
                print('        invariant: %s %s' % (k, str(dd)[:110]))
        summary.append({'name': name, 'verdict': v, 'why': why,
                        'turn': r.get('turn'), 'board': r.get('board', []),
                        'staging': len(r.get('staging', [])), 'staging_big': r.get('staging_big', []),
                        'invariants': [[k, str(dd)[:160]]
                                       for k, dd in r.get('invariants', [])],
                        'error': r.get('error')})
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1)
    print('=' * 64)
    print('TOTAL %d PASS %d FAIL %d KNOWN_FAIL %d SKIP %d'
          % (len(names), n_clean, n_fail, n_known, n_unverified))
    if n_unverified:
        print('SKIP counts UNVERIFIED captures; it does not mean those games passed.')
    return n_fail + n_unverified


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1:]) else 0)
