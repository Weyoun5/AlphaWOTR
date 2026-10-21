"""PRODUCTION-IMPORT GUARD.

Every other live suite imports the stack as `wotr_bridge.live.X` from
the PROJECT ROOT.  Production does not: `Play_vs_AI.bat` launches

    cd wotr_bridge && python -m live.live_server ...

so at runtime there is NO `wotr_bridge` package on sys.path at all --
the package is `live`.  An absolute `from wotr_bridge.live import X`
therefore works in every test and explodes in the real game.

USER 2026-08-20, live: "it says no module named wotr bridge, and kinda
bugged out."  Cause: `live/segmenter.py:51` lazily imported
`from wotr_bridge.live import live_parser` inside _norm_text, so the
first record that needed normalising raised ModuleNotFoundError and
took the serve cycle down.  158 green legs never saw it, because they
all run in the friendly environment.

This suite runs a child interpreter in the PRODUCTION layout and
imports every live module, so the whole class is pinned -- not just
the one line that bit.
"""
import os
import subprocess
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LIVE = os.path.join(_BRIDGE, 'live')

# the child must see EXACTLY what production sees: cwd=wotr_bridge,
# src on the path (engine modules), and NOT the project root.
_CHILD = r'''
import os, sys
sys.path.insert(0, r"{src}")
# prove the friendly name is genuinely unavailable, so this guard
# cannot pass by accident
import importlib.util as _u
if _u.find_spec("wotr_bridge") is not None:
    print("SETUP-BROKEN: wotr_bridge is importable here")
    raise SystemExit(2)
import importlib, pkgutil
bad = []
import live
for m in sorted(x.name for x in pkgutil.iter_modules(live.__path__)):
    try:
        importlib.import_module("live." + m)
    except Exception as e:
        bad.append((m, repr(e)))
for m, e in bad:
    print("IMPORT-FAIL", m, e)
print("MODULES", len(list(pkgutil.iter_modules(live.__path__))))
raise SystemExit(1 if bad else 0)
'''.format(src=_SRC)


def test_live_package_imports_in_production_layout(fails):
    """MEASURES: every module in wotr_bridge/live imports cleanly with
    cwd=wotr_bridge and NO project root on sys.path."""
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE',
               PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='-1')
    env.pop('PYTHONPATH', None)
    p = subprocess.run([sys.executable, '-c', _CHILD], cwd=_BRIDGE,
                       env=env, capture_output=True, text=True)
    out = (p.stdout or '') + (p.stderr or '')
    if p.returncode == 2:
        fails.append(('prodimport_setup', out.strip()[:300]))
        return
    if p.returncode != 0:
        fails.append(('prodimport_fail',
                      [l for l in out.splitlines()
                       if 'IMPORT-FAIL' in l][:6] or out[-300:]))
    # NOT VACUOUS: the child must actually have found the modules
    n = 0
    for line in out.splitlines():
        if line.startswith('MODULES'):
            n = int(line.split()[1])
    if n < 10:
        fails.append(('prodimport_vacuous',
                      'only %d live modules discovered' % n))


def test_norm_text_runs_in_production_layout(fails):
    """MEASURES the exact call that bit: segmenter._norm_text (a LAZY
    import, so importing the module alone does not exercise it)."""
    child = _CHILD.split('import importlib, pkgutil')[0] + r'''
from live import segmenter
v = segmenter._norm_text("$Wey moves Horn of Gondor/Mighty Attack "
                         "from Current FP Card to Board3.")
print("NORM", repr(v)[:60])
raise SystemExit(0 if v else 1)
'''
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE',
               PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='-1')
    env.pop('PYTHONPATH', None)
    p = subprocess.run([sys.executable, '-c', child], cwd=_BRIDGE,
                       env=env, capture_output=True, text=True)
    out = (p.stdout or '') + (p.stderr or '')
    if p.returncode != 0 or 'NORM' not in out:
        fails.append(('norm_text_prod', out.strip()[-300:]))


def main():
    tests = [test_live_package_imports_in_production_layout,
             test_norm_text_runs_in_production_layout]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
