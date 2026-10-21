"""Client-true undo model for the chooser card play (SCENARIO_DEFECTS
D-T2-5) and the <auto> fold direction.

CLIENT FACTS (decompiled ay_0.java + the real-client measurement in
D5 / probe_cc_refusal_u_green3):
  * '$Wey chooses a card.[..]' (l()) pushes NO undo snapshot;
  * the '<auto> moves <title> from <side> Hand (chosen) to Current
    <FP|SA> Card.' reveal goes through the interpreter (':411 accepts
    <auto>') and k() runs for 'moves' -> ITS OWN snapshot;
  * one press (o()) pops the newest snapshot and then keeps popping
    while the next-OLDER logged line starts with '<auto>' - autos
    fold into the press of the entry ABOVE them; an <auto> on top
    costs its own press.  Measured: one press after choose + reveal
    retracted the reveal only, the older staged die stayed in
    Current Action (pid 101 in Game.2035 in the dump).

A refused chooser event play (the muster card played without its
placements, card_play_event_chooser r2c) is therefore undo-ELIGIBLE:
[stage P, chooses, <auto> reveal, kills group(die, card), draws] =
3 presses (draw; kill group + the folded reveal; stage).  Before the
fix presses_to_revert returned None for ANY 'chooses' in the span and
the window was HELD with a 'corrective restore owed' alarm while the
boards split.

Netless, clientless - unit pins; the flow evidence is the real-client
scenario.
"""
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side, ActionDieFace
import game_logic
import board as board_mod
import game_state as gs

from wotr_bridge.live import undo_model as um
from wotr_bridge.live.driver import LiveDriver

STAGE = '$Wey moves Free Action Die (P) from FP Dice to Current Action.'
CHOOSE = '$Wey chooses a card.[Free Strategy Card]'
REVEAL = ('<auto> moves Grimbeorn the Old, Son of Beorn/Scouts from '
          'Free Peoples Hand (chosen) to Current FP Card.')
KILL = ('$Wey kills group (Free Action Die (P), Grimbeorn the Old, '
        'Son of Beorn/Scouts) in Current Action.')
DRAW = '$Wey draws Free Strategy Card.'
VIEWS = '$Wey views the Free Peoples event hand.'
REVEAL2 = ('<auto> moves Kindred of Glorfindel/Advantageous Position '
           'from Free Peoples Hand (chosen) to Current FP Card.')
UNDO = '$WOTR_AI_Shadow undoes last action : %s'


def test_presses_chooser_pair(fails):
    """The chooser window (r2c shape) costs 3 presses; the hidden
    commit (choose, no reveal) stays ineligible; a reveal on top of
    the stack costs its own press; two stacked reveals = one press."""
    lines = [VIEWS, STAGE, CHOOSE, REVEAL, KILL, DRAW]
    n = um.presses_to_revert(lines, (0, len(lines)),
                             own_nick='WOTR_AI_Shadow')
    if n != 3:
        fails.append(('chooser_window_presses', 'n=%r want 3' % (n,)))
    n = um.presses_to_revert([STAGE, CHOOSE], (0, 2))
    if n is not None:
        fails.append(('hidden_commit_eligible', n))
    n = um.presses_to_revert([STAGE, CHOOSE, REVEAL], (0, 3))
    if n != 2:
        fails.append(('reveal_on_top', 'n=%r want 2' % (n,)))
    n = um.presses_to_revert([REVEAL, REVEAL2], (0, 2))
    if n != 1:
        fails.append(('stacked_reveals', 'n=%r want 1' % (n,)))
    # a chooses whose next stateful line is NOT its reveal
    n = um.presses_to_revert([STAGE, CHOOSE, KILL], (0, 3))
    if n is not None:
        fails.append(('choose_without_reveal', n))


def test_stack_fold_direction(fails):
    """stack_after: the reveal is an entry until the kill is pushed on
    top of it (then it rides the kill's press); a press on [stage,
    choose, reveal] pops the reveal ONLY (the D5 measurement)."""
    ents = um.stack_after([STAGE, CHOOSE, REVEAL], 0)
    if [(e.index, e.verb, e.folds, e.modelled) for e in ents] \
            != [(0, 'moves', (), True), (2, '<auto>', (), True)]:
        fails.append(('reveal_own_entry', ents))
    ents = um.stack_after([STAGE, CHOOSE, REVEAL, KILL], 0)
    if [(e.index, e.verb, e.folds) for e in ents] \
            != [(0, 'moves', ()), (3, 'kills', (2,))]:
        fails.append(('reveal_folds_into_kill', ents))
    ents = um.stack_after([STAGE, CHOOSE, REVEAL, UNDO % REVEAL[7:]], 0)
    if [(e.index, e.verb) for e in ents] != [(0, 'moves')]:
        fails.append(('press_pops_reveal_only', ents))
    ents = um.stack_after([STAGE, CHOOSE, REVEAL, KILL,
                           UNDO % KILL[5:]], 0)
    if [(e.index, e.verb) for e in ents] != [(0, 'moves')]:
        fails.append(('press_pops_kill_and_reveal', ents))


def _fresh():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.EVENT),
    ])
    return st


class _Rig(object):
    def __init__(self, state, ai_side, phase='actions', seed=7):
        from wotr_bridge.live.randomness import ClientRandomness
        self.td = tempfile.mkdtemp()
        self.path = os.path.join(self.td, 'feed.txt')
        open(self.path, 'w').close()
        self.rand = ClientRandomness(seed=seed)
        self.rand.install()
        self.drv = LiveDriver(state, ai_side, self.path,
                              randomness=self.rand)
        self.drv.phase = phase

    def feed(self, *lines):
        with open(self.path, 'a', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln + '\n')
        return self.drv.tick()


def test_driver_refuses_chooser_play_with_presses(fails):
    """INVERTED by BL-94(a) (wave 16/18, fpgame10 G1 + fpgame11):
    Grimbeorn (recruit a North unit + Leader in Carrock) played
    through the chooser with NO recruits dragged now COMMITS on the
    live lane - the card effect's board legs are ENGINE follow-
    through the gesture can never express (no menu row carries unit
    params; the old refusal looped forever), BANKED as the human's
    arrival demands ('unit_in' rows at carrock on the echo ledger).
    The die spends, the card leaves the hand, no undo presses, no
    refusal chat."""
    st = _fresh()
    rig = _Rig(st, Side.SHADOW)                  # human = FP
    drv = rig.drv
    drv.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                      (Side.FREE_PEOPLES, 'strategy'): 1}
    hm = drv.hand_models[Side.FREE_PEOPLES]
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    hand.append(hm._mint('strategy'))
    cid = 'FREE_PEOPLES_Strategy_Grimbeorn_the_Old'
    cands = [c for c in drv.es.fp_strat_deck if 'Grimbeorn' in c]
    if cid not in drv.es.fp_strat_deck and not cands:
        drv.es.fp_strat_deck.append(cid)
    out = rig.feed(VIEWS, STAGE, CHOOSE, REVEAL, KILL)
    if drv.deferred is not None:
        # the guide's same-deck draw starves the window (r2c): the
        # draw line then arrives after the terminator
        out += rig.feed(DRAW)
    if not drv.window_stats.get('matched', 0):
        fails.append(('chooser_play_not_committed',
                      (dict(drv.window_stats), drv.findings[-3:],
                       [str(i)[:90] for i in out
                        if i[0] == 'chat'])))
        return
    presses = [i for i in out if i[0] == 'line'
               and 'Controls.819' in str(i[1])]
    if presses:
        fails.append(('chooser_commit_pressed_undo', len(presses)))
    if not any(d.used for d in drv.es.dice[Side.FREE_PEOPLES]):
        fails.append(('chooser_commit_die_unspent', True))
    _hand_now = drv.es.hands[Side.FREE_PEOPLES].hand
    if any('Grimbeorn' in str(c) for c in _hand_now):
        fails.append(('chooser_commit_card_still_in_hand',
                      list(_hand_now)))
    rows = [e for e in drv.echo_ledger
            if e.get('side') == 'fp'
            and str(e.get('kind', '')).endswith('_in')
            and str(e.get('region', '')) == 'carrock']
    if not rows:
        fails.append(('chooser_commit_no_arrival_demand',
                      list(drv.echo_ledger)))
    if any('not legal' in str(i[2]) for i in out if i[0] == 'chat'):
        fails.append(('chooser_commit_refusal_chat',
                      [str(i[2])[:80] for i in out
                       if i[0] == 'chat']))


def test_restore_watch_uses_sidecar_pids(fails):
    """expectations_from_window: the sidecar's exact pids are the first
    answerer for fungible pieces (the die, the card), so a refused
    chooser window is VERIFIABLE on the client dump (r2c: 5 pieces
    unverifiable by name, the watch blind); a redacted record (None)
    still reports its pieces as unverifiable."""
    kill_plain = ('$Wey kills group (Free Action Die (P), Phial of '
                  'Galadriel/It is a Gift) in Current Action.')
    lines = [STAGE, CHOOSE, REVEAL, kill_plain]
    pids = [['102'], None, None, ['102', '121']]
    expect, notat, unver = um.expectations_from_window(lines, pids=pids)
    if expect.get('102') != 'FP Dice':
        fails.append(('die_expect_by_pid', expect))
    if notat.get('102') != 'Current Action'             or notat.get('121') != 'Current Action':
        fails.append(('kill_notat_by_pid', notat))
    if not any('Grimbeorn' in str(u[0]) for u in unver):
        fails.append(('redacted_reveal_reported', unver))
    # a pid list that does not fit the line's pieces is never zipped
    # (the comma-titled card renders as two group pieces: reported)
    _e, _n, unver2 = um.expectations_from_window([KILL],
                                                 pids=[['102', '121']])
    if _n or len(unver2) != 3:
        fails.append(('pid_count_mismatch_guessed', (_n, unver2)))


def main():
    tests = [test_presses_chooser_pair, test_stack_fold_direction,
             test_driver_refuses_chooser_play_with_presses,
             test_restore_watch_uses_sidecar_pids]
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
