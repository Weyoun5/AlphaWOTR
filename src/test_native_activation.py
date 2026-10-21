"""Native-path ACTIVATION gate (PERF F10, 2026-08-11).

Proves the WOTR_NATIVE compiled path is ACTUALLY TAKEN, not silently
absent.  The F10 failure mode this gate exists for: the board.py hunk
+ pyx + .pyd were in the tree but the ai.py wiring block had been
clobbered, so WOTR_NATIVE=1 was a near-no-op and OFF==ON parity
smokes passed trivially.  This gate FAILS (SystemExit(4)) in exactly
that state.

Legs (flag ON):
  A  wotr_native._INITED is True — init_ai only runs from the ai.py
     wiring block; True is direct runtime evidence the block executed.
  B  IDENTITY: every swapped ai symbol IS the wotr_native function
     (call sites resolve these names through module globals, so
     identity == the native path is what executes), and
     board.sanitize / ai.sanitize / game_logic.sanitize are native.
  C  LIVE CALL on a real setup state: ai.encode_state / menu encode /
     clone+sig produce byte-identical results to calling wotr_native
     directly (belt over legs A/B: the value actually flows).

Flag unset/0: asserts the INVERSE (every symbol pure-python) so the
same gate certifies default-OFF integrity.

Usage:
  python src\\test_native_activation.py            (flag from env)
  WOTR_SRC=<dir> overrides the src dir under test (lab tooling).
Exit 0 = PASS; SystemExit(4) = ACTIVATION FAILURE (details printed).
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

_SRC = os.environ.get('WOTR_SRC') or os.path.dirname(
    os.path.abspath(__file__))
sys.path.insert(0, _SRC)

FLAG_ON = os.environ.get('WOTR_NATIVE') == '1'
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print('  FAIL: %s' % msg)
    else:
        print('  ok:   %s' % msg)


def main():
    import board
    import ai
    import game_logic
    import game_state as gs
    from game_state import Side

    # The swapped surface: (ai attribute, wotr_native attribute)
    swaps = [('_side_of_nation', 'side_of_nation'),
             ('_armies_lookup', 'armies_lookup'),
             ('encode_state', 'encode_state'),
             ('clone_state', 'clone_state'),
             ('_state_sig', 'state_sig')]
    torch_swaps = [('encode_action', 'encode_action'),
                   ('_action_cache_key', 'action_cache_key'),
                   ('_encode_action_cached', 'encode_action_cached'),
                   ('_region_index_cached', 'region_index_cached'),
                   ('_get_action_region_index',
                    'get_action_region_index')]

    if FLAG_ON:
        print('[activation] WOTR_NATIVE=1: asserting NATIVE path')
        import wotr_native
        # A: the ai.py wiring block ran (sole caller of init_ai)
        check(getattr(wotr_native, '_INITED', False) is True,
              'wotr_native._INITED (ai.py wiring block executed)')
        # B: identity of every swapped symbol
        all_swaps = swaps + (torch_swaps if ai.HAS_TORCH else [])
        check(ai.HAS_TORCH, 'ai.HAS_TORCH (action-complex swaps live)')
        for ai_name, nat_name in all_swaps:
            check(getattr(ai, ai_name, None)
                  is getattr(wotr_native, nat_name, None),
                  'ai.%s IS wotr_native.%s' % (ai_name, nat_name))
        check(board.sanitize is wotr_native.sanitize,
              'board.sanitize IS wotr_native.sanitize')
        check(ai.sanitize is wotr_native.sanitize,
              'ai.sanitize IS wotr_native.sanitize')
        check(getattr(game_logic, 'sanitize', None)
              is wotr_native.sanitize
              or not hasattr(game_logic, 'sanitize'),
              'game_logic.sanitize native (or not imported there)')
        # C: live-call byte equality through both entries
        b = board.load_board()
        st = gs.setup_initial_game(b)
        game_logic.initialize_decks(st)
        # prime the turn: an unrolled setup state has NO legal menu
        # (the certified lab run rolled dice before the menu check;
        # this priming was lost in the patch cut)
        game_logic.roll_action_dice(st, pool=Side.FREE_PEOPLES)
        game_logic.roll_action_dice(st, pool=Side.SHADOW)
        x_ai = ai.encode_state(st, Side.FREE_PEOPLES)
        x_nat = wotr_native.encode_state(st, Side.FREE_PEOPLES)
        check(x_ai.tobytes() == x_nat.tobytes(),
              'encode_state via ai == via wotr_native (bytes)')
        acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        check(len(acts) > 0, 'setup state has a legal menu')
        cm = {}
        ok_menu = True
        for a in acts:
            e1 = ai._encode_action_cached(a, st, Side.FREE_PEOPLES, cm)
            e2 = wotr_native.encode_action(a, st, Side.FREE_PEOPLES)
            if e1.tobytes() != e2.tobytes():
                ok_menu = False
                break
        check(ok_menu, 'menu encodings via ai cached == native direct '
                       '(%d actions)' % len(acts))
        cl = ai.clone_state(st)
        check(ai._state_sig(cl) == wotr_native.state_sig(st),
              'clone sig (via ai) == state sig (via native)')
    else:
        print('[activation] flag unset: asserting PURE-PYTHON path')
        for ai_name, _ in swaps + torch_swaps:
            fn = getattr(ai, ai_name, None)
            check(getattr(fn, '__module__', '?') == 'ai',
                  'ai.%s is pure python (module=%r)'
                  % (ai_name, getattr(fn, '__module__', '?')))
        check(getattr(board.sanitize, '__module__', '?') == 'board',
              'board.sanitize is pure python')

    if fails:
        print('[activation] FAILURES: %d — the WOTR_NATIVE state of '
              'this tree does NOT match the flag' % len(fails))
        raise SystemExit(4)
    print('[activation] PASS (%s)' % ('native' if FLAG_ON else 'pure'))


if __name__ == '__main__':
    main()
