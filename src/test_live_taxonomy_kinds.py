"""L6 SURFACE PARITY - every finding kind is classified (TEST_STRATEGY
_SPEC.md section 3 L6; BL-145: the panel printed an informational
finding under '!!! DIVERGENCES !!!' because the classifier was a word
list nobody had checked against the kinds the driver actually files).

MEASURES: an AST walk over wotr_bridge/live/*.py (+ live_view.py)
collects every `<x>.findings.append(('<kind>', ...))` literal; each
kind MUST appear in taxonomy.FINDING_KINDS with an explicit class.  A
new kind without a class fails the build - the author decides whether
it is a STOP signal, an error to file, or bookkeeping, at the moment
the kind is born.  Finding appends whose kind is not a literal are
counted against a small baseline (ratchet).
"""
import ast
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LIVE = os.path.join(_BRIDGE, 'live')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wotr_bridge.live import taxonomy as tx              # noqa: E402

DYNAMIC_BASELINE = 4       # 2026-08-25: driver 2, live_server 1, spectator 1


def literal_kinds(node):
    """Include both literal arms of conditional finding names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        left, right = literal_kinds(node.body), literal_kinds(node.orelse)
        if left and right:
            return left | right
    return set()


def enumerate_kinds():
    """({kind: {module,...}}, {module: n_dynamic})"""
    kinds, dyn = {}, {}
    files = [os.path.join(_LIVE, f) for f in sorted(os.listdir(_LIVE))
             if f.endswith('.py')]
    files.append(os.path.join(_BRIDGE, 'live_view.py'))
    for p in files:
        with open(p, encoding='utf-8') as f:
            tree = ast.parse(f.read())
        mod = os.path.basename(p)
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'append' and n.args):
                continue
            tgt = n.func.value
            name = getattr(tgt, 'attr', None) or getattr(tgt, 'id', None)
            if name != 'findings':
                continue
            a = n.args[0]
            names = literal_kinds(a.elts[0]) if isinstance(a, ast.Tuple) and a.elts else set()
            if names:
                for name in names:
                    kinds.setdefault(name, set()).add(mod)
            else:
                dyn[mod] = dyn.get(mod, 0) + 1
    return kinds, dyn


def test_every_kind_classified(fails):
    kinds, dyn = enumerate_kinds()
    missing = sorted(k for k in kinds if k not in tx.FINDING_KINDS)
    if missing:
        fails.append(('unclassified_kinds', missing))
    bad = sorted(k for k, v in tx.FINDING_KINDS.items()
                 if v not in ('divergence', 'error', 'info'))
    if bad:
        fails.append(('bad_class', bad))
    stale = sorted(k for k in tx.FINDING_KINDS if k not in kinds)
    n_dyn = sum(dyn.values())
    print('  %d kinds filed, %d classified, %d stale table rows, '
          '%d dynamic appends (baseline %d)'
          % (len(kinds), len(tx.FINDING_KINDS), len(stale), n_dyn,
              DYNAMIC_BASELINE))
    if stale:
        print('  stale (in table, no longer filed): %s'
              % ', '.join(stale[:12]))
    if n_dyn > DYNAMIC_BASELINE:
        fails.append(('dynamic_kind_appends_over_baseline', dyn))


def test_classify_uses_kind_before_words(fails):
    """A known kind's class wins over any word in its detail; an
    unknown kind falls back to the word lists; alarms (no kind) still
    classify by words."""
    if tx.classify('die_rebound_to_staged: idx 0 -> 1 (MUSTER)') != 'info':
        fails.append(('info_kind', True))
    if tx.classify('refusal_loop: 6 identical refusals') != 'divergence':
        fails.append(('divergence_kind', True))
    if tx.classify('unconsumed_rng: d6 x2') != 'error':
        fails.append(('error_kind', True))
    # an info kind whose detail quotes a divergence word stays info
    if tx.classify('mirror_apply: chat said DIVERGENCE') != 'info':
        fails.append(('kind_beats_words', True))
    if tx.classify('WITNESS ESCALATION: 3 row(s)') != 'divergence':
        fails.append(('alarm_words', True))
    if tx.classify('brand_new_kind: FAILED to x') != 'error':
        fails.append(('unknown_kind_words', True))


def test_partition_counts_match_classes(fails):
    # a first WITNESS observation is info by design; only its
    # ESCALATION is the stop signal.  REWRITTEN 2026-08-29 (BRCMP-9):
    # `census_aging` is instrument health (the witness gated too long)
    # - ERROR-class, never a divergence: it moved the seat's STOP rule
    # with no board disagreement behind it.
    rows = ['refusal_loop: x', 'mirror_apply: y', 'feed_error: z',
            'WITNESS: 2 discrepancies', 'census_aging: 3 checks']
    d, e, i = tx.partition(rows)
    if (len(d), len(e), len(i)) != (1, 2, 2):
        fails.append(('partition', (d, e, i)))
    if tx.classify('census_aging: gated x10') != 'error':
        fails.append(('census_aging_class',
                      tx.classify('census_aging: gated x10')))


def main():
    tests = [test_every_kind_classified, test_classify_uses_kind_before_words,
             test_partition_counts_match_classes]
    n_fail = 0
    for t in tests:
        fails = []
        try:
            t(fails)
        except Exception as e:
            fails.append(('exception', repr(e)))
        print('  [%s] %s' % ('PASS' if not fails else 'FAIL', t.__name__))
        for f in fails:
            print('    FAIL: %r' % (f,))
        n_fail += bool(fails)
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), n_fail))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
