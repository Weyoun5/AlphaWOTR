"""Per-round hits announcement = the POST-prevention count
(SCENARIO_DEFECTS D-T1-R2-3).

Real client (ai_attacks_human_defends r2c): the human defender's
Shield-wall cancelled one of the AI attacker's three hits; the engine's
allocation pending asked for TWO casualties (correct) but the chat
announced '3 hits for me'.  USER doctrine: chat only for what pieces
cannot show - the hits line exists to tell the human how many
casualties to take, so it must equal the number the allocation
pending will make them allocate.

ONE quantity: `CombatResult.attacker_total_hits / defender_total_hits`
accumulate the round's FINAL hits (the same value the deferred
allocation carries in `deferred_hits_on_defender / _on_attacker`).
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board
import game_logic
import game_state as gs
import combat as cb
from game_state import Side, Nation, UnitKind


def _setup():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def test_shield_wall_round_totals(fails):
    """Attacker rolls [5,5,5,1,1] (3 combat hits at 5+ in the field - North Ithilien is a plain region; Osgiliath would be a 6+ fortification),
    the defender holds Shield-wall (Book of Mazarbul: cancel one hit
    when the opponent scored 2+) -> the deferred allocation carries 2
    and the result's attacker_total_hits says 2 too; the chat line
    rendered from the result reads '2 hits'."""
    st = _setup()
    atk = game_logic.get_or_create_army(st, 'dagorlad')
    atk.units = [gs.Unit(Nation.SAURON, UnitKind.REGULAR)
                 for _ in range(5)]
    atk.leaders = []
    dfn = game_logic.get_or_create_army(st, 'north ithilien')
    dfn.units = [gs.Unit(Nation.GONDOR, UnitKind.REGULAR)
                 for _ in range(3)]
    dfn.leaders = []
    # A1 (2026-08-24): Book of Mazarbul's combat side is Advantageous
    # Position (client-confirmed) - the Shield-wall specimen is King
    # Brand's Men
    sw = "FREE_PEOPLES_Strategy_King_Brand's_Men"
    orig = cb.roll_d6_batch
    rolls = {'n': 0}

    def forced(n, ctx='', **kw):
        rolls['n'] += 1
        if rolls['n'] == 1:
            return ([5, 5, 5] + [1] * n)[:n]       # attacker: 3 hits
        return [1] * n                             # everything else
    cb.roll_d6_batch = forced
    try:
        res = cb.resolve_combat(st, 'dagorlad', 'north ithilien',
                                attacker_card_id=None,
                                defender_card_id=sw,
                                choices={'defer_casualties': True})
    finally:
        cb.roll_d6_batch = orig
    if int(res.deferred_hits_on_defender) != 2:
        fails.append(('precondition_deferred_hits',
                      'NOT-MEASURED: deferred=%r (want 2: 3 hits - '
                      'Shield-wall)' % (res.deferred_hits_on_defender,)))
        return
    if int(res.attacker_total_hits) != 2:
        fails.append(('total_hits_pre_cancellation',
                      'attacker_total_hits=%r, deferred=%r'
                      % (res.attacker_total_hits,
                         res.deferred_hits_on_defender)))
    if int(res.defender_total_hits) != 0:
        fails.append(('defender_total', res.defender_total_hits))
    # the chat rendered from the result
    from wotr_bridge.live import emission
    em = emission.render_hits(int(res.attacker_total_hits),
                              int(res.defender_total_hits),
                              attacker_is_human=False)
    txt = ' '.join(str(t) for _k, t in em.chat)
    # 'for you' = the hits the HUMAN scored; the AI attacker scored the
    # 2 that survive Shield-wall, so they are 'for me'
    if '0 hits for you, 2 hits for me' != txt.strip():
        fails.append(('hits_chat_text', txt))


def test_no_card_round_totals_unchanged(fails):
    """Control: with no prevention card the totals equal the raw
    combat hits (3)."""
    st = _setup()
    atk = game_logic.get_or_create_army(st, 'dagorlad')
    atk.units = [gs.Unit(Nation.SAURON, UnitKind.REGULAR)
                 for _ in range(5)]
    atk.leaders = []
    dfn = game_logic.get_or_create_army(st, 'north ithilien')
    dfn.units = [gs.Unit(Nation.GONDOR, UnitKind.REGULAR)
                 for _ in range(3)]
    dfn.leaders = []
    orig = cb.roll_d6_batch
    rolls = {'n': 0}

    def forced(n, ctx='', **kw):
        rolls['n'] += 1
        if rolls['n'] == 1:
            return ([5, 5, 5] + [1] * n)[:n]
        return [1] * n
    cb.roll_d6_batch = forced
    try:
        res = cb.resolve_combat(st, 'dagorlad', 'north ithilien',
                                choices={'defer_casualties': True})
    finally:
        cb.roll_d6_batch = orig
    if int(res.attacker_total_hits) != 3 \
            or int(res.deferred_hits_on_defender) != 3:
        fails.append(('control_totals',
                      (res.attacker_total_hits,
                       res.deferred_hits_on_defender)))


def main():
    tests = [test_shield_wall_round_totals,
             test_no_card_round_totals_unchanged]
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
