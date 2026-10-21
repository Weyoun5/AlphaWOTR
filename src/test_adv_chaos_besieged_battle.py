"""RED-TEAM finding CHAOS-1: a co-located BESIEGED battle can reach a
state with NO attacker units -- combat.py setup_combat raises
'Co-located battle at <region>: missing attacker units', which the
engine's combat_card_menu build catches as a [STATE-INVARIANT]
(combat_card_menu_failed) and degrades to combat_card_none.  The game
does not crash, but a stronghold sitting in besieged_strongholds with
no besieging army present is an illegal/degenerate state the engine
should never reach (a siege-cleanup gap: the region stays 'besieged'
after the besieger vacates or dies, leaving only the garrison).

Mechanism walkthrough (bug-explanation directive):
  RULE: a siege exists only while an enemy army co-occupies a
  defender's stronghold region.  When the besieger leaves or is
  eliminated, the siege is lifted (the region leaves the besieged
  set).
  ENGINE: combat.py setup_combat, co-located branch, splits the
  region army into garrison (v_gar) and besieger (v_bes) portions by
  nation; v_atk = the besieger portion (assault default).  If the
  region is flagged besieged but no besieger units are actually
  present, v_atk.units is empty and it raises
  'missing attacker units' (combat.py:257-260).
  NUMBERS: net-free chaos, DETERMINISTIC -- seed 95376 flavor
  'attacks' fires the invariant at TURN 21 (minas tirith), side=sh
  then side=fp, then action_exec_failed on combat_card_none.  Found
  in red_team chaos wave 4 (seeds 95000-95499); the harness marks the
  game clean (error=False) because the engine catches it internally
  and the game runs to the turn cap.
  CONSEQUENCE: the region is treated as besieged when no siege is in
  progress; every combat_card menu query and every attack resolution
  there fails and silently degrades.  On the SIDED/live path a
  combat_card_menu_failed at a real stronghold could halt or misplay
  a battle the human is fighting.  The underlying stale
  besieged_strongholds entry may also skew siege-only rules
  (5-unit cap, sortie direction, victory checks).

EXPECTED: FAILS against the current tree (the invariant fires).
This is an ENGINE finding (net-free); no nets are loaded.
"""
import io
import contextlib
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')


def _block_torch():
    if 'ai' in sys.modules:
        return
    if 'torch' not in sys.modules:
        sys.modules['torch'] = types.ModuleType('torch')


def _run_seed(seed, flavor, max_turns=40):
    """One net-free random-legal game; returns captured stdout."""
    _block_torch()
    import engine_play as ep
    import ai
    assert getattr(ai, 'HAS_TORCH', None) is False, \
        'net-free mode failed (torch loaded)'
    import random
    rng = random.Random((seed * 2654435761) & 0xffffffff)

    def _pick(dp):
        acts = dp.actions

        def _of(t):
            return [i for i, a in enumerate(acts)
                    if getattr(a, 'action_type', '') == t]
        cc = _of('combat_card_play')
        if cc and rng.random() < 0.5:
            return rng.choice(cc)
        aa = _of('army_attack') + _of('siege_assault')
        if aa and rng.random() < 0.4:
            return rng.choice(aa)
        return rng.randrange(len(acts))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s = ep.EngineSession(seed=seed, mcts_sims=1,
                             max_turns=max_turns)
        n = 0
        while True:
            dp = s.current()
            if dp is None:
                break
            s.submit(_pick(dp), dp.side)
            n += 1
            if n > 40000:
                break
    return buf.getvalue()


def test_no_co_located_battle_missing_attacker_units():
    out = _run_seed(95376, 'attacks', max_turns=40)
    hits = [ln for ln in out.splitlines()
            if 'missing attacker units' in ln
            or 'missing defender units' in ln]
    assert not hits, (
        'CO-LOCATED BATTLE MISSING ATTACKER UNITS: net-free seed 95376 '
        '(flavor attacks) reaches a besieged co-located battle at '
        'minas tirith with no attacker units at turn 21 -- combat.py '
        'setup_combat raises and the engine degrades to a '
        '[STATE-INVARIANT] combat_card_menu_failed.  A stronghold in '
        'besieged_strongholds with no besieging army is a siege-'
        'cleanup gap.  %d invariant line(s): %s'
        % (len(hits), [h.strip()[:120] for h in hits[:2]]))


if __name__ == '__main__':
    try:
        test_no_co_located_battle_missing_attacker_units()
        print('PASS test_no_co_located_battle_missing_attacker_units')
    except AssertionError as e:
        print('FAIL test_no_co_located_battle_missing_attacker_units: '
              '%s' % str(e).encode('ascii', 'replace').decode())
        sys.exit(1)
