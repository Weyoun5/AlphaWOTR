"""X-audit 2026-08-31 pins (USER: "keep auditing"): state-encoding data
gaps found by the corpus-wide dim-variance census on the decision-ledger
game set.

B1  eyes_allocated (X3192) was NEVER reconstructed - 0 across every row
    of every audited game; staged hunt-box dice all read as "rolled"
    (X3193).  Fix: extract_one publishes per-turn allocation/rolled-Eye
    counts; _emit_sample stamps them encoding-only (menus already
    enumerated), with the eye_allocate decision's own pre-roll state
    exempt (stamping it would leak the label).
B4  Gollum guiding the Fellowship lit only the guide one-hot - his
    companion-details dims (X3370-73) stayed 0 because he lives in
    state.gollum / the guide slot, never fellowship.companions.
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

X_EYE_ALLOC = 3192
X_EYE_ROLLED = 3193
X_GOLLUM_ALIVE = 3370
X_GOLLUM_INFEL = 3371


def _fresh_state():
    import board
    import game_state as gs
    return gs.setup_initial_game(board.load_board())


def test_b4_gollum_guide_lights_companion_block():
    import ai
    import game_state as gs
    from game_state import Side
    st = _fresh_state()
    x0 = np.asarray(ai.encode_state(st, Side.FREE_PEOPLES))
    assert x0[X_GOLLUM_ALIVE] == 0.0 and x0[X_GOLLUM_INFEL] == 0.0, \
        'fresh state must not report Gollum (got %r/%r)' % (
            x0[X_GOLLUM_ALIVE], x0[X_GOLLUM_INFEL])
    gollum = gs.Companion(name='Gollum', level=0, leadership=0)
    st.gollum = gollum
    st.fellowship.guide = gollum
    x1 = np.asarray(ai.encode_state(st, Side.FREE_PEOPLES))
    assert x1[X_GOLLUM_ALIVE] == 1.0, 'Gollum guide: alive dim dark'
    assert x1[X_GOLLUM_INFEL] == 1.0, 'Gollum guide: in-fellowship dim dark'


def test_b1_eyes_stamp_reaches_encoding():
    import ai
    import rendered_extract as rx
    from game_state import Side
    st = _fresh_state()
    st.turn = 2
    legal = [ai.Action(die_idx=-1, action_type='pass', params={})]
    old_a, old_r = rx._RX_EYEALLOC, rx._RX_EYEROLL
    rx._RX_EYEALLOC = {2: 1}
    rx._RX_EYEROLL = {2: 2}
    try:
        s = rx._emit_sample(st, Side.SHADOW, legal, 0, 'pass', 2)
        x = np.asarray(s['x'])
        assert abs(x[X_EYE_ALLOC] - 1 / 5.0) < 1e-6, \
            'allocated-eyes dim %r (want 0.2)' % (x[X_EYE_ALLOC],)
        assert abs(x[X_EYE_ROLLED] - 2 / 4.0) < 1e-6, \
            'rolled-eyes dim %r (want 0.5)' % (x[X_EYE_ROLLED],)
    finally:
        rx._RX_EYEALLOC, rx._RX_EYEROLL = old_a, old_r


def test_b1_eye_decision_state_stays_preroll():
    """The eye_allocate decision itself must NOT carry its own answer."""
    import ai
    import rendered_extract as rx
    from game_state import Side
    st = _fresh_state()
    st.turn = 2
    legal = [ai.Action(die_idx=-1, action_type='eye_allocate',
                       params={'count': 1})]
    old_a, old_r = rx._RX_EYEALLOC, rx._RX_EYEROLL
    rx._RX_EYEALLOC = {2: 1}
    rx._RX_EYEROLL = {2: 2}
    try:
        s = rx._emit_sample(st, Side.SHADOW, legal, 0, 'eye_allocate', 2)
        x = np.asarray(s['x'])
        assert x[X_EYE_ALLOC] == 0.0 and x[X_EYE_ROLLED] == 0.0, \
            'eye decision state leaked its label: %r/%r' % (
                x[X_EYE_ALLOC], x[X_EYE_ROLLED])
    finally:
        rx._RX_EYEALLOC, rx._RX_EYEROLL = old_a, old_r


def test_b1_prestaged_box_kept():
    """A hunt-drive state that staged its own box keeps it verbatim."""
    import ai
    import game_state as gs
    import rendered_extract as rx
    from game_state import Side
    st = _fresh_state()
    st.turn = 3
    d = gs.ActionDie(gs.SH_FACES, Side.SHADOW)
    d.current = gs.ActionDieFace.EYE
    st.hunt_box = [d]
    legal = [ai.Action(die_idx=-1, action_type='pass', params={})]
    old_a, old_r = rx._RX_EYEALLOC, rx._RX_EYEROLL
    rx._RX_EYEALLOC = {3: 2}
    rx._RX_EYEROLL = {3: 1}
    try:
        s = rx._emit_sample(st, Side.SHADOW, legal, 0, 'pass', 3)
        x = np.asarray(s['x'])
        assert len(st.hunt_box) == 1, 'pre-staged box was overwritten'
        assert abs(x[X_EYE_ALLOC] - 2 / 5.0) < 1e-6, \
            'allocation stamp missing on pre-staged-box state'
    finally:
        rx._RX_EYEALLOC, rx._RX_EYEROLL = old_a, old_r


def main():
    tests = [
        test_b4_gollum_guide_lights_companion_block,
        test_b1_eyes_stamp_reaches_encoding,
        test_b1_eye_decision_state_stays_preroll,
        test_b1_prestaged_box_kept,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print('PASS %s' % t.__name__)
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (t.__name__, e))
        except Exception as e:
            fails += 1
            print('FAIL %s: %s: %s' % (t.__name__, type(e).__name__, e))
    print('TOTAL=%d FAIL=%d' % (len(tests), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
