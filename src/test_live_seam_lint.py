"""L5 SEAM LINT (TEST_STRATEGY_SPEC.md section 3 L5; BL-143: an
IPC read under a broad `except` swallowed a NameError, consumed the
request, published null, and every physical answer timed out).

MEASURES: the number of SILENT broad exception handlers per live
module - `except:` / `except Exception:` / `except BaseException:`
whose whole body is `pass`, `continue`, or a bare `return` - must
never exceed the recorded baseline (a RATCHET: the numbers only go
down; lowering one here is the fix's proof).  The baseline was taken
2026-08-25 at 61 handlers; each is a place where a defect can hide
without a voice.
"""
import ast
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_LIVE = os.path.join(os.path.dirname(_SRC), 'wotr_bridge', 'live')

# module -> silent broad handlers allowed (2026-08-25 baseline)
BASELINE = {
    'driver.py': 18, 'live_server.py': 14, 'transport.py': 8,
    'client_board.py': 4, 'gesture_match.py': 6, 'adopt.py': 2,
    'emission.py': 1, 'chat.py': 2, 'randomness.py': 1,
    'spectator.py': 2, 'scenario.py': 1,               # 59 total (adopt
    #   4->2: BL-21(5) consolidated the triplicated cb.unit_type decode
    #   into _decode_piece and dropped the side-check try)
}


def _is_broad(handler):
    t = handler.type
    if t is None:
        return True
    if isinstance(t, ast.Name):
        return t.id in ('Exception', 'BaseException')
    if isinstance(t, ast.Tuple):
        return any(isinstance(e, ast.Name)
                   and e.id in ('Exception', 'BaseException') for e in t.elts)
    return False


def _is_silent(handler):
    body = handler.body
    if len(body) != 1:
        return False
    st = body[0]
    if isinstance(st, (ast.Pass, ast.Continue)):
        return True
    return isinstance(st, ast.Return) and st.value is None


def silent_broad_handlers(path):
    """[(lineno, enclosing function)] of silent broad handlers."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = []
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _is_broad(node) \
                and _is_silent(node):
            fn = node
            while fn is not None and not isinstance(
                    fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents.get(fn)
            out.append((node.lineno, fn.name if fn is not None else '<module>'))
    return out


def test_silent_handler_ratchet(fails):
    counts = {}
    for fn in sorted(os.listdir(_LIVE)):
        if not fn.endswith('.py'):
            continue
        hits = silent_broad_handlers(os.path.join(_LIVE, fn))
        counts[fn] = len(hits)
        allowed = BASELINE.get(fn, 0)
        if len(hits) > allowed:
            fails.append(('silent_handlers_over_baseline',
                          (fn, len(hits), allowed,
                           [h for h in hits][:6])))
    total = sum(counts.values())
    print('  silent broad handlers: %d total :: %s'
          % (total, ', '.join('%s=%d' % (k, v) for k, v in counts.items()
                              if v)))
    if fails:
        return
    # the ratchet's other half: a lowered count is recorded (loud, not
    # failing - the author lowers BASELINE in the same change)
    for fn, n in counts.items():
        if n < BASELINE.get(fn, 0):
            print('  RATCHET: %s now %d < baseline %d - lower BASELINE'
                  % (fn, n, BASELINE[fn]))


def test_lint_is_sensitive(fails):
    """Sensitivity control: the detector fires on the three silent
    shapes and stays quiet on a voiced handler."""
    src = (
        "def f():\n"
        "    try:\n        x()\n    except Exception:\n        pass\n"
        "    try:\n        x()\n    except:\n        return\n"
        "    for i in y:\n        try:\n            x()\n"
        "        except BaseException:\n            continue\n"
        "    try:\n        x()\n    except Exception as e:\n"
        "        log(e)\n        return\n"
        "    try:\n        x()\n    except ValueError:\n        pass\n"
    )
    import tempfile
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'probe.py')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(src)
    hits = silent_broad_handlers(p)
    if len(hits) != 3:
        fails.append(('lint_sensitivity', hits))


def main():
    tests = [test_silent_handler_ratchet, test_lint_is_sensitive]
    n_fail = 0
    for t in tests:
        fails = []
        try:
            t(fails)
        except Exception as e:
            fails.append(('exception', repr(e)))
        status = 'PASS' if not fails else 'FAIL'
        print('  [%s] %s' % (status, t.__name__))
        for f in fails:
            print('    FAIL: %r' % (f,))
        n_fail += bool(fails)
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), n_fail))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
