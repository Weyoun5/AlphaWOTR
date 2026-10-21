# -*- coding: utf-8 -*-
"""Standing trace rule-auditor (session 35, USER escalation 2026-07-20).

Every rules violation the user found by MANUALLY inspecting
w5_g2_sh_military.trace.gz becomes a machine assertion here, run over
EVERY self-play trace batch — traces must audit clean before a human
ever looks at them. Add a check for every future finding.

Checks (per step, on the pickled GameState snapshots):
  A1 retired: map travel in Mordor's regions is legal; the Mordor Track
     is separate from those regions (rulebook pp. 38, 43).
  A2 a card-forced hunt draw of an EYE tile never coincides with a
     reveal (hidden True->False at that step)
  A3 every play_event action's card was in the acting side's REAL hand
     on the previous snapshot (phantom-play detector)
  A4 piece conservation: deployed + fp_casualties <= NATION_MAX_FIGURES
     for every nation at every step
  A5 FP casualty boxes are non-decreasing (FP losses are permanent)

Usage:
  python trace_audit.py <trace.gz | dir> [more ...]
Exit 0 = all clean; 1 = violations (printed, with step indices).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# board chdirs to src/ at import — capture the invoker's cwd FIRST so
# relative CLI paths resolve against it, not src/
_INVOKE_CWD = os.getcwd()

from board import sanitize  # noqa: E402
from game_trace import load_trace  # noqa: E402

_CARD_DRAW_CTX = ('foul thing', "isildur's bane", 'isildur s bane',
                  'orc patrol', 'breaking of the fellowship',
                  'revealed in', 'entering', 'leaving')
# annotation row types (state on these rows is the SAME post-action
# snapshot as the causing action's row — never a pre-event state)
_ANN_TYPES = ('hunt_tile_draw', 'dice_roll', 'random_draw',
              'stronghold_tile_draw', 'card_drawn')


def _fs_region(state):
    import game_logic
    return game_logic.fellowship_region(state)


def _deployed(state):
    import ai
    return ai._count_deployed_all(state)


def audit_trace(path):
    return audit_loaded_trace(load_trace(path), os.path.basename(path))


def audit_loaded_trace(t, label='trace'):
    """Run rule checks on a loaded trace without writing or changing it."""
    import ai
    from game_state import Side
    steps = t['steps']
    v = []

    def flag(i, code, msg):
        v.append('%s step %d [%s] %s' % (label, i,
                                         code, msg))

    prev_state = None
    pre_exec_state = None   # last state from a NON-annotation row
    prev_cas = {}
    for i, s in enumerate(steps):
        st = s.get('state')
        if st is None:
            continue
        if str(s.get('action_type')) not in _ANN_TYPES:
            # Annotations follow their causing action's POST-state.
            # Preserve that action's PRE-state for the Eye/reveal check.
            pre_exec_state = prev_state
        # A2 - card-forced Eye draw must not reveal.  The tracer emits
        # hunt tiles with kind='action'; accept both kinds.
        try:
            if (s.get('kind') in ('annotation', 'action')
                    and s.get('action_type') == 'hunt_tile_draw'):
                p = s.get('params') or {}
                ctx = str(p.get('ctx', '')).lower()
                is_eye = 'eye' in str(p.get('tile_name', '')).lower()
                if is_eye and any(c in ctx for c in _CARD_DRAW_CTX):
                    # compare against the last PRE-execution state:
                    # annotation rows flush AFTER the causing action,
                    # so prev_state at an annotation row is already
                    # the post-action snapshot
                    if (pre_exec_state is not None
                            and pre_exec_state.fellowship.hidden
                            and not st.fellowship.hidden):
                        flag(i, 'A2', 'card-drawn Eye revealed the '
                                      'fellowship (ctx=%s)' % ctx)
        except Exception as e:
            flag(i, 'A2?', 'check error %r' % e)
        # A3 — phantom card play
        try:
            if (s.get('kind') == 'action'
                    and s.get('action_type') == 'play_event'
                    and prev_state is not None):
                p = s.get('params') or {}
                cid = p.get('card_id')
                tag = str(s.get('side') or p.get('side') or '').lower()
                side = (Side.FREE_PEOPLES if tag == 'fp'
                        else Side.SHADOW)
                if cid and cid not in prev_state.hands[side].hand:
                    flag(i, 'A3', 'phantom play_event %s (not in %s '
                                  'hand)' % (cid, side.name))
        except Exception as e:
            flag(i, 'A3?', 'check error %r' % e)
        # A4 — conservation
        try:
            dep = _deployed(st)
            fcas = getattr(st, 'fp_casualties', None) or {}
            for nat, mx in ai.NATION_MAX_FIGURES.items():
                for kind in ('regular', 'elite'):
                    used = dep.get(nat, {}).get(kind, 0) \
                        + fcas.get(nat, {}).get(kind, 0)
                    if used > mx.get(kind, 0):
                        flag(i, 'A4', '%s %s %d > max %d'
                             % (nat, kind, used, mx.get(kind, 0)))
        except Exception as e:
            flag(i, 'A4?', 'check error %r' % e)
        # A5 — FP casualty boxes: elites/leaders never shrink; a
        # REGULAR decrease is legal ONLY as a downgrade's box-pull
        # (USER RULING 2026-07-20d: the one return path, bar Gandalf
        # the White), which books >= as many elite casualties in the
        # SAME step.
        try:
            fcas = getattr(st, 'fp_casualties', None) or {}
            # union of old+new keys: wholesale DELETION of a nation's
            # entry is the most extreme illegal shrinkage
            for nat in set(prev_cas) | set(fcas):
                box = fcas.get(nat, {})
                pb = prev_cas.get(nat, {})
                d_reg = box.get('regular', 0) - pb.get('regular', 0)
                d_eli = box.get('elite', 0) - pb.get('elite', 0)
                d_led = box.get('leader', 0) - pb.get('leader', 0)
                if d_eli < 0:
                    flag(i, 'A5', '%s elite casualties shrank by %d'
                         % (nat, -d_eli))
                if d_led < 0:
                    flag(i, 'A5', '%s leader casualties shrank by %d'
                         % (nat, -d_led))
                if d_reg < 0 and d_eli < -d_reg:
                    flag(i, 'A5', '%s regular box-pull %d without '
                                  'matching downgrades (elite +%d)'
                         % (nat, -d_reg, d_eli))
            prev_cas = {n: dict(b) for n, b in fcas.items()}
        except Exception as e:
            flag(i, 'A5?', 'check error %r' % e)
        prev_state = st
    return v


def main(paths):
    targets = []
    for p in paths:
        # resolve against the INVOKER's cwd (board chdir'd to src/)
        if not os.path.isabs(p):
            p = os.path.join(_INVOKE_CWD, p)
        if os.path.isdir(p):
            targets.extend(os.path.join(p, f) for f in sorted(os.listdir(p))
                           if f.endswith('.trace.gz'))
        else:
            targets.append(p)
    if not targets:
        # an empty/mistargeted batch dir must never read as a pass
        print('AUDIT ERROR: no .trace.gz targets found in %r' % (paths,))
        return 1
    total_v = []
    for tp in targets:
        try:
            vs = audit_trace(tp)
        except Exception as e:
            vs = ['%s LOAD-ERROR %r' % (os.path.basename(tp), e)]
        total_v.extend(vs)
        print('%-45s %s' % (os.path.basename(tp),
                            'CLEAN' if not vs else
                            '%d VIOLATIONS' % len(vs)), flush=True)
    for line in total_v:
        print('  ' + line)
    print('=' * 60)
    print('AUDIT %s: %d traces, %d violations'
          % ('PASS' if not total_v else 'FAIL', len(targets),
             len(total_v)))
    return 0 if not total_v else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
