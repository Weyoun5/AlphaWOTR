"""SCENARIO_DEFECTS D-T2-3 pin (real-client reproduction
`hand_limit_discard_and_wwsat_r1a`: LIVE HALT 'client drew
SHADOW_Strategy_Rage_of_the_Dunlendings (pid 176) but the engine
sh_strat_deck does not hold it - deck desync').

The prefix lane draws the AI's cards as placeholders, each pop taking
a REAL cid out of the fictional deck; the flip adopted only the cids
the client's hand names and never restored the popped ones, so after
any flip the AI's decks were short one real cid per prefix draw and
the client's fresh tables could draw one of them.  Invariant pinned:
after the flip the AI side's decks hold EVERY card-table cid of that
side/deck that is not in the adopted hand (nor face-up / on the
table), and no placeholder; a live draw of any such cid passes the
funnel's deck-membership check.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side


def test_flip_restores_ai_decks(fails):
    """MEASURES: fixture prefix -> flip; the AI's two decks equal the
    card table minus the adopted hand, with zero placeholders, and
    the funnel accepts a live draw of a cid the prefix lane had
    popped."""
    import test_live_scenario as TS
    from cards import get_cards_by_deck
    rig = TS._Rig()
    try:
        srv, sc = rig.srv, rig.sc
        n = sc.cut
        rig.append_feed(n)
        rig.cycle()
        pids = TS._ai_hand_pids(rig)
        TS._dump_with_ai_hand(os.path.join(TS.SESSION,
                                           'ai_client_board.txt'),
                              rig.dump, pids)
        rig.cycle()
        if not srv.driver.live:
            fails.append(('flip_control', 'NOT-MEASURED: no flip: %r'
                          % (srv.notes[-3:],)))
            return
        es = srv.driver.es
        ai_side = srv.ai_side
        tag = 'fp' if ai_side is Side.FREE_PEOPLES else 'sh'
        hand = set(es.hands[ai_side].hand)
        popped = []
        for deck, attr in (('Character', tag + '_char_deck'),
                           ('Strategy', tag + '_strat_deck')):
            table = set(c.card_id for c in get_cards_by_deck(ai_side,
                                                              deck))
            dl = list(getattr(es, attr, None) or ())
            ph = [c for c in dl if str(c).startswith('HU_')]
            if ph:
                fails.append(('ai_deck_placeholders', (attr, ph)))
            want = table - hand
            got = set(c for c in dl if not str(c).startswith('HU_'))
            if got != want:
                miss = sorted(want - got)
                extra = sorted(got - want)
                fails.append(('ai_deck_short', (attr, 'missing', miss,
                                                'extra', extra)))
                popped.extend((deck.lower(), c) for c in miss)
        # a live draw of a popped cid must pass the funnel (deck
        # membership is the desync witness)
        if popped:
            return                  # already failed above
        deck, attr = 'Character', tag + '_char_deck'
        dl = list(getattr(es, attr))
        cid = dl[0]
        try:
            srv.rand.feed('card', (ai_side.name, 'character', cid, '999'))
            import game_logic as gl
            with srv.rand.advance():
                gl.draw_event_card(es, ai_side, 'character')
        except Exception as e:
            fails.append(('live_draw_after_flip', repr(e)[:160]))
            return
        if cid not in es.hands[ai_side].hand:
            fails.append(('live_draw_landed', (cid, es.hands[ai_side].hand)))
    finally:
        rig.close()


def main():
    tests = [test_flip_restores_ai_decks]
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
