"""gesture_match instruments: window facts + menu-sized delta matching.

MEASURED QUANTITIES per test below.  Windows are built through the
REAL pipeline (live_parser + Segmenter) from lines in the client's own
grammar — specimen shapes from LIVE_0818_BUGLIST where one exists; the
synthetic army-move lines state their piece tokens through a
parse_piece ROUND-TRIP CONTROL so a token the grammar would not
produce fails the test rather than silently testing nothing.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as board_mod
import game_logic
import game_state as gs
from game_state import Side, ActionDieFace

import rendered_replay as rr
from wotr_bridge.live import live_parser
from wotr_bridge.live import gesture_match as gm
from wotr_bridge.live.segmenter import Segmenter


def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def adopted_state():
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    return st


def window_from(lines):
    seg = Segmenter()
    w = None
    for raw in lines:
        for ev, p in seg.push(live_parser.parse(raw)):
            if ev == 'closed':
                w = p
    return w


_DISPLAY = ('Sauron', 'Isengard', 'S&E', 'Gondor', 'Rohan', 'Elven',
            'Dwarven', 'North')


def token_for(nation, kind_word, count=1):
    """Client piece token for (nation enum, 'Regular'/'Elite'), proven
    by a parse_piece ROUND-TRIP: the token must parse back to the same
    nation and kind, or the test refuses to use it."""
    for disp in _DISPLAY:
        tok = '%s %s' % (disp, kind_word)
        pp = rr.parse_piece(tok)
        if pp and pp[0] == 'unit' \
                and gm._nation_enum(pp[1]) is nation \
                and pp[2] == kind_word.upper():
            if count > 1:
                return '[%d]%s' % (count, tok)
            return tok
    raise AssertionError('no client token round-trips for %r %s'
                         % (nation, kind_word))


def test_zone_key(fails):
    """MEASURES: real client zone strings classify to the documented
    key classes."""
    cases = [
        ('Osgiliath', ('region', 'osgiliath')),
        ('Free Peoples Hand', ('hand', 'FP')),
        ('Current SA Card', ('play', 'SH')),
        ('FP Reinforcements', ('reinf', 'FP')),
        ('Fellowship Track 3', ('fs_track', 3)),
        ('Political Track 2', ('pol_track', 2)),
    ]
    for z, want in cases:
        got = gm.zone_key(z)
        if got != want:
            fails.append(('zone_key', '%r -> %r want %r'
                          % (z, got, want)))
    st = gm.zone_key('Woodland Realm Stronghold')
    if st[0] != 'stronghold':
        fails.append(('zone_key_stronghold', repr(st)))


def test_window_facts_specimens(fails):
    """MEASURES: WindowFacts extracted from two REAL buglist specimen
    windows (C2 fellowship move; E1 card play with reinforcement
    muster) express exactly the human's action content."""
    w = window_from([
        '$Wey moves Free Action Die (M) from FP Dice to Current '
        'Action.',
        '$Wey moves FSP from Fellowship Track 3 to Fellowship '
        'Track 4.',
        '$Wey moves Free Action Die (M) from Current Action to The '
        'Hunt For The Ring.',
    ])
    if w is None or w.closed != 'die_to_hunt':
        fails.append(('c2_window', repr(w and w.closed)))
        return
    f = gm.window_facts(w)
    if f.fs_progress != 1 or f.armies or f.cards_played:
        fails.append(('c2_facts', 'progress=%d armies=%r'
                      % (f.fs_progress, dict(f.armies))))
    w = window_from([
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey moves Grimbeorn the Old, Son of Beorn/Scouts from Free '
        'Peoples Hand to Current FP Card.',
        '$Wey moves group (North Leader, North Elite) from FP '
        'Reinforcements to Carrock.',
        '$Wey draws Free Strategy Card.',
        '$Wey kills group (Free Action Die (P), Grimbeorn the Old, '
        'Son of Beorn/Scouts) in Current Action.',
    ])
    if w is None or w.closed != 'die_killed':
        fails.append(('e1_window', repr(w and w.closed)))
        return
    f = gm.window_facts(w)
    if not any('Grimbeorn' in c for c in f.cards_played):
        fails.append(('e1_card', repr(f.cards_played)))
    if f.leaders.get('carrock') != 1:
        fails.append(('e1_leader', repr(dict(f.leaders))))
    elite_rows = [(k, v) for k, v in f.armies.items()
                  if k[0] == 'carrock' and v == 1]
    if not elite_rows:
        fails.append(('e1_elite', repr(dict(f.armies))))


def _first_simple_move(st, menu):
    """First army_move from a region with no leaders and no named
    figures (keeps the synthetic window to plain unit tokens)."""
    for a in menu:
        if a.action_type != 'army_move':
            continue
        p = a.params
        key = None
        for k in st.armies:
            from board import sanitize
            if sanitize(k) == sanitize(p['from']):
                key = k
                break
        if key is None:
            continue
        army = st.armies[key]
        if getattr(army, 'leaders', None):
            continue
        if p.get('n_regular', 0) + p.get('n_elite', 0) >= 1:
            return a
    return None


def test_match_real_army_move(fails):
    """MEASURES: a window expressing a MENU action's move in client
    grammar matches back to exactly that action (or its physical
    equivalent), with the die-face filter active."""
    import ai
    st = adopted_state()
    menu = ai.get_legal_actions(st, Side.SHADOW)
    a = _first_simple_move(st, menu)
    if a is None:
        fails.append(('army_precondition', 'no simple army_move'))
        return
    p = a.params
    nat_reg = nat_eli = None
    for k in st.armies:
        from board import sanitize
        if sanitize(k) == sanitize(p['from']):
            for u in st.armies[k].units:
                if str(u.kind.name) == 'REGULAR':
                    nat_reg = u.nation
                else:
                    nat_eli = u.nation
    toks = []
    if p.get('n_regular'):
        toks.append(token_for(nat_reg, 'Regular', p['n_regular']))
    if p.get('n_elite'):
        toks.append(token_for(nat_eli, 'Elite', p['n_elite']))
    obj = toks[0] if len(toks) == 1 else 'group (%s)' % ', '.join(toks)
    lines = [
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves %s from %s to %s.' % (obj, str(p['from']).title(),
                                          str(p['to']).title()),
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ]
    w = window_from(lines)
    if w is None or w.closed != 'die_killed':
        fails.append(('army_window', repr(w and w.closed)))
        return
    res = gm.match_window(st, Side.SHADOW, w, menu=menu)
    if res.status != 'match':
        fails.append(('army_match', repr(res)))
        return
    got = res.chain[0]
    from board import sanitize
    if got.action_type != 'army_move' \
            or sanitize(got.params['from']) != sanitize(p['from']) \
            or sanitize(got.params['to']) != sanitize(p['to']) \
            or got.params.get('n_regular') != p.get('n_regular') \
            or got.params.get('n_elite') != p.get('n_elite'):
        fails.append(('army_params', '%r vs %r'
                      % (got.params, p)))
    if res.post is st:
        fails.append(('army_post_clone', 'post is the input state'))
    print('  [army] menu=%d applies=%d' % (len(menu), res.applies))


def test_match_none_illegal(fails):
    """MEASURES (sensitivity): a window expressing a move NO menu
    action produces returns 'none', never a forced match."""
    import ai
    st = adopted_state()
    menu = ai.get_legal_actions(st, Side.SHADOW)
    lines = [
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves [3]Sauron Regular from The Shire to Rivendell.',
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ]
    w = window_from(lines)
    res = gm.match_window(st, Side.SHADOW, w, menu=menu)
    if res.status != 'none':
        fails.append(('illegal_none', repr(res)))


def test_budget_hold(fails):
    """MEASURES: budget exhaustion HOLDS (B13-1 doctrine), never
    refuses."""
    import ai
    st = adopted_state()
    menu = ai.get_legal_actions(st, Side.SHADOW)
    a = _first_simple_move(st, menu)
    if a is None:
        fails.append(('hold_precondition', 'no simple army_move'))
        return
    lines = [
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves [9]Sauron Regular from Barad-Dur to Gorgoroth.',
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ]
    w = window_from(lines)
    res = gm.match_window(st, Side.SHADOW, w, menu=menu, budget=1)
    if res.status != 'hold':
        fails.append(('budget_hold', repr(res)))


def test_unmapped_hold(fails):
    """MEASURES: a window carrying a piece token the grammar cannot
    express HOLDS loudly (rule 12), never silently partial-matches."""
    st = adopted_state()
    lines = [
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves Completely Unknown Object from Osgiliath to '
        'Minas Tirith.',
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ]
    w = window_from(lines)
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status != 'hold' or 'unmapped' not in res.note:
        fails.append(('unmapped_hold', repr(res)))


def test_fs_move_match(fails):
    """MEASURES: the C2-shaped fellowship-move window matches the FP
    fellowship_move menu action under the CHARACTER die filter."""
    import ai
    st = adopted_state()
    menu = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    if not any(a.action_type == 'fellowship_move' for a in menu):
        fails.append(('fs_precondition', 'no fellowship_move in menu'))
        return
    lines = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves FSP from Fellowship Track 0 to Fellowship '
        'Track 1.',
        '$Wey moves Free Action Die (C) from Current Action to The '
        'Hunt For The Ring.',
    ]
    w = window_from(lines)
    if w is None or w.closed != 'die_to_hunt':
        fails.append(('fs_window', repr(w and w.closed)))
        return
    res = gm.match_window(st, Side.FREE_PEOPLES, w, menu=menu)
    if res.status != 'match' \
            or res.chain[0].action_type != 'fellowship_move':
        fails.append(('fs_match', repr(res)))


def test_draw_window(fails):
    """MEASURES (lane L1, specimen game 2023-01-02 L39-41): an
    event-die window whose content is ONE deck-draw line matches the
    draw_card menu action for exactly that deck (the other deck's
    draw_card is refused by the draws-coverage row).  The engine deck
    list must shrink by one on the committed post state."""
    import ai
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.EVENT),
        (Side.SHADOW, 1, ActionDieFace.ARMY),
    ])
    n0 = len(st.sh_strat_deck)
    w = window_from([
        '%Sirrus233 moves Shadow Action Die (P) from SA Dice to '
        'Current Action.',
        '%Sirrus233 draws Shadow Strategy Card.',
        '%Sirrus233 kills Shadow Action Die (P) in Current Action.',
    ])
    if w is None or w.closed != 'die_killed':
        fails.append(('draw_window', repr(w and w.closed)))
        return
    f = gm.window_facts(w)
    if dict(f.draws) != {('SHADOW', 'strategy'): 1}:
        fails.append(('draw_facts', repr(dict(f.draws))))
        return
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status != 'match' \
            or res.chain[0].action_type != 'draw_card' \
            or res.chain[0].params.get('deck') != 'strategy':
        fails.append(('draw_match', repr(res)))
        return
    if len(getattr(res.post, 'sh_strat_deck', [])) != n0 - 1:
        fails.append(('draw_deck_len',
                      len(getattr(res.post, 'sh_strat_deck', []))))


def test_political_window(fails):
    """MEASURES (lane L2, specimen game 2023-01-02 L43-45): a
    muster-die window moving a nation marker DOWN the client political
    track (client steps DESCEND toward war) matches advance_political
    for exactly that nation, and the committed engine box INCREMENTS."""
    import ai
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.MUSTER),
        (Side.SHADOW, 1, ActionDieFace.ARMY),
    ])
    from game_state import Nation
    b0 = st.political.boxes[Nation.ISENGARD]
    if b0 != 2:
        fails.append(('political_precondition',
                      'Isengard box %r (want 2 = client track 1)' % b0))
        return
    w = window_from([
        '%Sirrus233 moves Shadow Action Die (M) from SA Dice to '
        'Current Action.',
        '%Sirrus233 moves Isengard from Political Track 1 to '
        'Political Track 0.',
        '%Sirrus233 kills Shadow Action Die (M) in Current Action.',
    ])
    f = gm.window_facts(w)
    if dict(f.political) != {'ISENGARD': 1}:
        fails.append(('political_facts', repr(dict(f.political))))
        return
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status != 'match' \
            or res.chain[0].action_type != 'advance_political' \
            or str(getattr(res.chain[0].params.get('nation'), 'name',
                           res.chain[0].params.get('nation'))) \
            != 'ISENGARD':
        fails.append(('political_match', repr(res)))
        return
    if res.post.political.boxes[Nation.ISENGARD] != b0 + 1:
        fails.append(('political_box',
                      res.post.political.boxes[Nation.ISENGARD]))


def test_flip_hide_window(fails):
    """MEASURES (lane L3, specimen game 2023-01-02 L76-78 class): a
    windowed plain fellowship flip (no 'Previously' clause) matches
    the fellowship_hide menu action when the engine fellowship is
    revealed; the committed state is hidden again."""
    import ai
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 1, ActionDieFace.MUSTER),
    ])
    st.fellowship.hidden = False
    w = window_from([
        '%GenericPie moves Free Action Die (C) from FP Dice to '
        'Current Action.',
        '%GenericPie flips The Fellowship in Rivendell.',
        '%GenericPie kills Free Action Die (C) in Current Action.',
    ])
    f = gm.window_facts(w)
    if f.fel_flips != [('rivendell', 'flip')]:
        fails.append(('flip_facts', repr(f.fel_flips)))
        return
    res = gm.match_window(st, Side.FREE_PEOPLES, w)
    if res.status != 'match' \
            or res.chain[0].action_type != 'fellowship_hide':
        fails.append(('flip_match', repr(res)))
        return
    if not res.post.fellowship.hidden:
        fails.append(('flip_post_hidden', False))
    # SPECIFICITY: a reveal-style flip is the ENGINE's consequence —
    # classified 'reveal' (soft hidden constraint) and it expresses
    # the track reset the client renders no FSP move for; a hide
    # candidate (fs delta 0) must NOT match it
    w2 = window_from([
        '%GenericPie moves Free Action Die (C) from FP Dice to '
        'Current Action.',
        '%GenericPie flips The Fellowship in Rivendell. Previously '
        'on Fellowship Track 2.',
        '%GenericPie kills Free Action Die (C) in Current Action.',
    ])
    f2 = gm.window_facts(w2)
    if f2.fel_flips != [('rivendell', 'reveal')] \
            or f2.fs_progress != -2:
        fails.append(('flip_reveal_class',
                      '%r fsp=%d' % (f2.fel_flips, f2.fs_progress)))
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    st2.fellowship.hidden = False
    res2 = gm.match_window(st2, Side.FREE_PEOPLES, w2)
    if res2.status == 'match':
        fails.append(('flip_reveal_not_hide', repr(res2)))


def test_card_play_window(fails):
    """MEASURES (lane L4, specimen game 2023-01-02 L70-75 class): a
    window naming a played card title matches ONLY the play_event
    action for that cid (vocabulary lane), with the effect's board
    legs covered by the same window's army facts.  The cid is planted
    in the FP hand first — the reveal wiring the drivers perform."""
    import ai
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.EVENT),
        (Side.FREE_PEOPLES, 1, ActionDieFace.CHARACTER),
    ])
    title = 'Grimbeorn the Old, Son of Beorn/Scouts'
    cid = gm.title_to_cid(title)
    if cid is None:
        fails.append(('card_resolve', 'title_to_cid None for %r'
                      % title))
        return
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    # the plant must LEAVE the deck too (the reveal-swap's own
    # bookkeeping): otherwise the play's replacement draw can re-draw
    # the same cid and the hand assertion becomes RNG-dependent
    if cid in st.fp_strat_deck:
        st.fp_strat_deck.remove(cid)
    menu = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    if not any(a.action_type == 'play_event'
               and a.params.get('card_id') == cid for a in menu):
        fails.append(('card_precondition',
                      'play_event %s not in menu' % cid))
        return
    # the REAL E1 window shape: effect legs (leader + elite) AND the
    # replacement draw all render inside the play window
    w = window_from([
        '%Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '%Wey moves Grimbeorn the Old, Son of Beorn/Scouts from Free '
        'Peoples Hand to Current FP Card.',
        '%Wey moves group (North Leader, North Elite) from FP '
        'Reinforcements to Carrock.',
        '%Wey draws Free Strategy Card.',
        '%Wey kills group (Free Action Die (P), Grimbeorn the Old, '
        'Son of Beorn/Scouts) in Current Action.',
    ])
    res = gm.match_window(st, Side.FREE_PEOPLES, w, menu=menu)
    if res.status != 'match':
        fails.append(('card_match', repr(res)))
        return
    if res.chain[0].action_type != 'play_event' \
            or res.chain[0].params.get('card_id') != cid:
        fails.append(('card_chain', repr(res.chain)))
    if cid in res.post.hands[Side.FREE_PEOPLES].hand:
        fails.append(('card_left_hand', cid))


def test_two_army_window(fails):
    """MEASURES (specimen game 2023-01-02 L87-90): one Army-die window
    moving TWO armies chains [army_move, army_move] through the
    engine's pending second move, judged on the CUMULATIVE root delta
    (the per-level delta bug matched these windows 'none')."""
    import ai
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
    ])
    w = window_from([
        '%Sirrus233 moves Shadow Action Die (A) from SA Dice to '
        'Current Action.',
        '%Sirrus233 moves group (Nazgul, Sauron Elite, [4]Sauron '
        'Regular) from Barad-Dur to Gorgoroth.',
        '%Sirrus233 moves group (Nazgul, Sauron Elite, [5]Sauron '
        'Regular) from Dol Guldur to South Anduin Vale.',
        '%Sirrus233 kills Shadow Action Die (A) in Current Action.',
    ])
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status != 'match':
        fails.append(('two_army_match', repr(res)))
        return
    moves = [a for a in res.chain if a.action_type == 'army_move']
    if len(moves) != 2:
        fails.append(('two_army_chain',
                      [a.action_type for a in res.chain]))
        return
    from board import sanitize
    srcs = sorted(sanitize(str(a.params['from'])) for a in moves)
    if srcs != ['barad dur', 'dol guldur']:
        fails.append(('two_army_sources', srcs))
    # the die is spent: no phantom open half-action on the committed
    # state (pass-closure ran or the second move consumed it)
    if getattr(res.post, 'pending_army_move_die', None) is not None:
        fails.append(('two_army_pending_open', True))


def test_ring_zone_not_fellowship(fails):
    """MEASURES (game 2023-01-28 T1 specimen): a companion move OUT of
    'Fellowship Of The Ring' is a SEPARATION (named row), never a ring
    walk — the old ring test matched the substring 'ring' in the
    source zone and swallowed the whole crown route."""
    w = window_from([
        '%savedbyhim moves Free Action Die (C) from FP Dice to '
        'Current Action.',
        '%savedbyhim moves Strider from Fellowship Of The Ring to '
        'Moria.',
        '%savedbyhim kills Free Action Die (C) in Current Action.',
    ])
    f = gm.window_facts(w)
    if f.ring_moves or f.rings:
        fails.append(('ring_misclass', repr(f.ring_moves)))
    if f.named.get(('Strider', 'moria'), 0) != 1:
        fails.append(('sep_named', dict(f.named)))
    # a REAL ring walk still classifies
    w = window_from([
        '%X moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '%X moves Narya from Elven Ring 3 to Dark Ring 1.',
        '%X kills Free Action Die (P) in Current Action.',
    ])
    f = gm.window_facts(w)
    if not f.ring_moves or f.rings.get('FP', 0) != -1 \
            or f.rings.get('SH', 0) != 1:
        fails.append(('ring_real', (f.ring_moves, dict(f.rings))))


def test_siege_fact_row(fails):
    """MEASURES (Corsairs/Dol-Amroth specimen): the garrison's
    region->own-stronghold drag expresses sieges[region]=+1 (net-zero
    in army rows), and _covers refuses an engine delta whose siege
    flag did not flip — the row is what forces the chain through
    defender_siege_choice."""
    w = window_from([
        '%SH moves Shadow Action Die (H) from SA Dice to Current '
        'Action.',
        '%FP moves group ([3]Gondor Regular) from Dol Amroth to '
        'Dol Amroth Stronghold.',
        '%SH kills Shadow Action Die (H) in Current Action.',
    ])
    f = gm.window_facts(w)
    if f.sieges.get('dol amroth', 0) <= 0:
        fails.append(('siege_row', dict(f.sieges)))
    if any(v for v in f.armies.values()):
        fails.append(('siege_army_net', dict(f.armies)))
    eng = gm.EngFacts()
    eng.dice[('SHADOW', 'H')] = -1
    f2 = gm.WindowFacts()
    f2.sieges['dol amroth'] = 1
    f2.dice_faces[('SHADOW', 'H')] = -1
    if gm._covers(f2, eng):
        fails.append(('siege_covers_without_flag', True))
    eng.sieges['dol amroth'] = 1
    if not gm._covers(f2, eng):
        fails.append(('siege_covers_with_flag', False))


def test_relax_draws_bounds(fails):
    """MEASURES: the relaxed-draws coverage (client misplay lane) is a
    BOUNDED subset — engine must draw >=1 and <=window per deck the
    window drew, nothing it did not, and a draw-free candidate can
    never cover a draw window even relaxed (the 'pass covers
    everything' hazard)."""
    win = gm.WindowFacts()
    win.draws[('FREE_PEOPLES', 'strategy')] = 2

    def eng_with(n, key=('FREE_PEOPLES', 'strategy')):
        e = gm.EngFacts()
        if n:
            e.draws[key] = n
        return e
    if gm._covers(win, eng_with(1)):
        fails.append(('strict_should_fail', True))
    if not gm._covers(win, eng_with(1), relax_draws=True):
        fails.append(('relax_1_of_2', False))
    if gm._covers(win, eng_with(0), relax_draws=True):
        fails.append(('relax_zero_draw', True))
    if gm._covers(win, eng_with(3), relax_draws=True):
        fails.append(('relax_over', True))
    if gm._covers(win, eng_with(1, ('FREE_PEOPLES', 'character')),
                  relax_draws=True):
        fails.append(('relax_wrong_deck', True))


def test_crown_flip_transform(fails):
    """MEASURES (game 2023-01-28 T1): a STANDALONE dual-face flip
    ('flips Aragorn in Minas Tirith', Strider already there) expresses
    the in-place identity transform; with an in-window arrival the
    rows still net to arrival-as-new-face (GtW resurrection)."""
    w = window_from([
        '%FP moves Free Action Die (W) from FP Dice to Current '
        'Action.',
        '%FP flips Aragorn in Minas Tirith.',
        '%FP kills Free Action Die (W) in Current Action.',
    ])
    f = gm.window_facts(w)
    if f.named.get(('Strider', 'minas tirith'), 0) != -1 \
            or f.named.get(('Aragorn', 'minas tirith'), 0) != 1:
        fails.append(('crown_rows', dict(f.named)))
    w = window_from([
        '%FP moves Free Action Die (W) from FP Dice to Current '
        'Action.',
        '%FP moves Gandalf the Grey from FP Casualties to The Grey '
        'Havens.',
        '%FP flips Gandalf the White in The Grey Havens.',
        '%FP kills Free Action Die (W) in Current Action.',
    ])
    f = gm.window_facts(w)
    if f.named.get(('Gandalf the White', 'grey havens'), 0) != 1 \
            or f.named.get(('Gandalf the Grey', 'grey havens'), 0) \
            != 0:
        fails.append(('gtw_rows', dict(f.named)))


def test_covers_card_leg_tolerance(fails):
    """BL-94(a) + the fpgame11 lesson (the tolerance shipped INERT
    because no pin existed): on a cards_played window with the LIVE
    flag, the card effect's ENGINE-EXTRA board legs are tolerated
    (Imrahil's recruits, a heal's corruption); without the flag
    (spectator) parity stays strict; a WRONG voiced drag refuses
    either way."""
    win = gm.WindowFacts()
    win.cards_played.append('Imrahil of Dol Amroth/Shield-wall')
    eng = gm.EngFacts()
    eng.armies[('dol amroth', 'gondor:regular')] = 2
    if not gm._covers(win, eng, card_leg_tolerance=True):
        fails.append(('card_leg_tolerated', False))
    if gm._covers(win, eng):
        fails.append(('spectator_strict_leak', True))
    eng2 = gm.EngFacts()
    eng2.corruption = -1
    if not gm._covers(win, eng2, card_leg_tolerance=True):
        fails.append(('corruption_leg_tolerated', False))
    win2 = gm.WindowFacts()
    win2.cards_played.append('Imrahil of Dol Amroth/Shield-wall')
    win2.armies[('osgiliath', 'gondor:regular')] = 1
    eng3 = gm.EngFacts()
    eng3.armies[('minas tirith', 'gondor:regular')] = 1
    if gm._covers(win2, eng3, card_leg_tolerance=True):
        fails.append(('wrong_drag_accepted', True))


def test_follow_through_table(fails):
    """MEASURES (tracker hygiene after BL-94(a)/BL-142/BL-144): the
    three live tolerances are ONE rule - engine follow-through the
    human's choreography renders as a marker/flip/nothing is tolerated
    on the candidate whose apply carried it.  (1) follow_through_for
    unions the chain's rows and is None for the spectator; (2) a
    declare candidate tolerates ONLY an un-voiced corruption delta;
    (3) a reveal-placement candidate tolerates ONLY the marker flip
    on a no-hidden-change delta; (4) an army_move chain tolerates
    nothing (empty set) - strict parity on every family."""
    import types
    A = lambda t: types.SimpleNamespace(action_type=t)
    if gm.follow_through_for([A('army_move')], False) is not None:
        fails.append(('ft_spectator_none', 'not None'))
    if gm.follow_through_for([A('army_move')], True) != frozenset():
        fails.append(('ft_army_move_empty',
                      gm.follow_through_for([A('army_move')], True)))
    ft_pe = gm.follow_through_for([A('elven_ring'), A('play_event')], True)
    if not {'armies', 'corruption', 'draws'} <= ft_pe:
        fails.append(('ft_play_event', ft_pe))
    # (2) declare: corruption only
    win = gm.WindowFacts()
    win.fel_flips += [('rivendell', 'flip'), ('rivendell', 'flip')]
    win.fs_progress = -2
    eng = gm.EngFacts()
    eng.fs_progress = -2
    eng.corruption = -1
    eng.fel_region = 'rivendell'
    ft_d = gm.follow_through_for([A('fellowship_declare')], True)
    if not gm._covers(win, eng, follow_through=ft_d):
        fails.append(('ft_declare_heal', False))
    if gm._covers(win, eng, follow_through=frozenset()):
        fails.append(('ft_declare_strict_leak', True))
    eng_b = gm.EngFacts()
    eng_b.fs_progress = -2
    eng_b.fel_region = 'rivendell'
    eng_b.armies[('hollin', 'elves:regular')] = 1     # not a heal
    if gm._covers(win, eng_b, follow_through=ft_d):
        fails.append(('ft_declare_over_tolerant', 'army row tolerated'))
    # (3) reveal placement: the marker flip on a no-change delta
    win2 = gm.WindowFacts()
    win2.fel_flips.append(('fords of bruinen', 'flip'))
    win2.fs_progress = -3
    eng2 = gm.EngFacts()
    eng2.fs_progress = -3
    eng2.fel_region = 'fords of bruinen'
    eng2.hidden_change = False
    ft_r = gm.follow_through_for([A('fellowship_reveal_place')], True)
    if not gm._covers(win2, eng2, follow_through=ft_r):
        fails.append(('ft_reveal_marker', False))
    if gm._covers(win2, eng2, follow_through=frozenset()):
        fails.append(('ft_reveal_strict_leak', True))
    eng2.hidden_change = True        # a real toggle still must match
    if not gm._covers(win2, eng2, follow_through=ft_r):
        fails.append(('ft_reveal_real_toggle', False))
    eng2.hidden_change = False
    eng2.fel_region = 'hollin'         # region parity stays strict
    if gm._covers(win2, eng2, follow_through=ft_r):
        fails.append(('ft_reveal_region_leak', True))


def test_play_event_refusal_names_the_reason(fails):
    """USER 2026-08-28 (explicit, never generic): a refused card play's
    note says WHICH engine fact refuses it.  (1) the card is not in the
    engine's copy of the hand -> 'not in my copy of your hand ... say
    resync' (the BL-150 drift); (2) its own play condition is unmet ->
    the card's requirement text (Challenge of the King without
    Strider/Aragorn in a Gondor/Rohan army); the generic 'not in the
    engine menu' text is gone from both."""
    from game_state import ActionDieFace
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
        (Side.SHADOW, 0, ActionDieFace.ARMY)])
    lines = ['$Wey moves Free Action Die (C) from FP Dice to Current '
             'Action.',
             '$Wey moves Challenge of the King/Sudden Strike from Free '
             'Peoples Hand to Current FP Card.',
             '$Wey kills group (Free Action Die (C), Challenge of the '
             'King/Sudden Strike) in Current Action.']
    cid = 'FREE_PEOPLES_Character_Challenge_of_the_King'
    for lst in (st.fp_char_deck, st.fp_strat_deck):
        if cid in lst:
            lst.remove(cid)
    # (1) not in the engine hand
    st.hands[Side.FREE_PEOPLES].hand[:] = []
    res = gm.match_window(st, Side.FREE_PEOPLES, window_from(lines))
    if res.status != 'none' or 'not in my copy of your hand' not in \
            str(res.note) or 'engine menu' in str(res.note):
        fails.append(('reason_hand_drift', (res.status, res.note)))
    # (2) in hand, requirement unmet (Strider still in the Fellowship)
    st.hands[Side.FREE_PEOPLES].hand[:] = [cid]
    res = gm.match_window(st, Side.FREE_PEOPLES, window_from(lines))
    if res.status != 'none' or 'Strider/Aragorn' not in str(res.note) \
            or 'engine menu' in str(res.note):
        fails.append(('reason_requirement', (res.status, res.note)))


def test_bl150_chain_stops_before_a_discard_pending(fails):
    """BL-150 D1 (USER manual game 2026-08-28, the verbatim Eomer
    window feed 159-163): an engine ordering bug (E1, fixed) granted
    Gandalf's same-deck redraw while the optioned Eomer still sat in a
    6-card hand, opening a hand-limit discard pending MID-CHAIN; the
    matcher then continued through that pending and committed
    discard_card(hand[0]) - Challenge of the King - inside the Eomer
    play.  CONTRACT (defense in depth, with the E1 bug SIMULATED around
    the real matcher): a chain NEVER traverses a discard pending - the
    matched chain has no discard_card leg, hand[0] survives, and the
    pending stays open for the human's own gesture."""
    import ai
    from game_state import ActionDieFace
    st = fresh_state()
    if getattr(st.fellowship.guide, 'name', '') != 'Gandalf the Grey':
        fails.append(('bl150_d1_control', 'NOT-MEASURED: guide'))
        return
    eomer = next((c for c in st.fp_strat_deck if 'omer' in c), None)
    first = 'FREE_PEOPLES_Character_Challenge_of_the_King'
    if eomer is None:
        fails.append(('bl150_d1_control', 'NOT-MEASURED: no Eomer cid'))
        return
    for lst in (st.fp_char_deck, st.fp_strat_deck):
        for c in (first, eomer):
            if c in lst:
                lst.remove(c)
    fill = [c for c in st.fp_char_deck[:4]]
    for c in fill:
        st.fp_char_deck.remove(c)
    st.hands[Side.FREE_PEOPLES].hand[:] = [first, eomer] + fill      # 6
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.WILL_OF_WEST),
        (Side.SHADOW, 0, ActionDieFace.ARMY)])
    lines = ['$Wey moves Free Action Die (W) from FP Dice to Current '
             'Action.',
             '$Wey moves Eomer, Son of Eomund/Valour from Free Peoples '
             'Hand to Current FP Card.',
             "$Wey moves group (Rohan Leader, Rohan Elite) from FP "
             "Reinforcements to Helm's Deep.",
             '$Wey draws Free Strategy Card.',
             '$Wey kills group (Free Action Die (W), Eomer, Son of '
             'Eomund/Valour) in Current Action.']
    # SIMULATE the (fixed) E1 bug around the real matcher: the redraw
    # lands at deferral, the 6-hand goes to 7, a discard pending opens
    # mid-chain
    orig = ai.apply_action

    def buggy(state, action):
        post = orig(state, action)
        if getattr(action, 'action_type', '') == 'play_event' \
                and getattr(post, 'pending_card_option', None) \
                and (action.params or {}).get('card_id') == eomer:
            # the OLD timing exactly: the redraw at deferral (and not
            # again at resolution), the hand-limit check right after
            game_logic.grant_draw(post, Side.FREE_PEOPLES, 'strategy')
            post.pending_card_option['die_face'] = None
            game_logic.open_hand_limit_discard(post)
        return post
    ai.apply_action = buggy
    try:
        res = gm.match_window(st, Side.FREE_PEOPLES, window_from(lines))
    finally:
        ai.apply_action = orig
    if res.status != 'match':
        fails.append(('bl150_d1_control',
                      'NOT-MEASURED: Eomer window did not match (%s: %s)'
                      % (res.status, str(res.note)[:100])))
        return
    legs = [a.action_type for a in res.chain]
    if 'discard_card' in legs:
        fails.append(('bl150_d1_chain_discarded', legs))
    hand = res.post.hands[Side.FREE_PEOPLES].hand
    if first not in hand:
        fails.append(('bl150_d1_hand0_lost', list(hand)))
    # E2: once Eomer has left, the hand is back to 6 - the phantom
    # pending is re-derived away, nothing is owed and nothing was guessed
    if len(hand) != 6 or getattr(res.post, 'pending_discard', None) \
            is not None:
        fails.append(('bl150_d1_phantom_pending_persisted',
                      (len(hand), res.post.pending_discard)))


def test_exact_parity_beats_tolerance_fit(fails):
    """MEASURES (USER manual test 2026-08-25, Fear! Fire! Foes!): two
    chains both COVER a window only because the follow-through table
    tolerates engine-extra rows; the chain with ZERO un-voiced rows is
    the gesture, a tolerance-fitted rival is not.  _unvoiced_rows: 0 on
    exact parity, +1 per differing family row (an extra engine draw, an
    extra corruption walk, an un-voiced hidden toggle).  Only a UNIQUE
    exact chain may win - two exact chains stay ambiguous (that rule is
    the matcher's, pinned by the ambiguity tests)."""
    win = gm.WindowFacts()
    win.armies[('the shire', 'north:regular')] = 1
    eng = gm.EngFacts()
    eng.armies[('the shire', 'north:regular')] = 1
    if gm._unvoiced_rows(win, eng) != 0:
        fails.append(('parity_zero', gm._unvoiced_rows(win, eng)))
    eng2 = gm.EngFacts()
    eng2.armies[('the shire', 'north:regular')] = 1
    eng2.draws[('FREE_PEOPLES', 'strategy')] = 1
    if gm._unvoiced_rows(win, eng2) != 1:
        fails.append(('parity_extra_draw', gm._unvoiced_rows(win, eng2)))
    eng3 = gm.EngFacts()
    eng3.armies[('the shire', 'north:regular')] = 1
    eng3.corruption = -1
    eng3.hidden_change = True
    if gm._unvoiced_rows(win, eng3) != 2:
        fails.append(('parity_extra_two', gm._unvoiced_rows(win, eng3)))
    win2 = gm.WindowFacts()
    win2.fel_flips.append(('hollin', 'flip'))
    eng4 = gm.EngFacts()
    eng4.hidden_change = True
    if gm._unvoiced_rows(win2, eng4) != 0:
        fails.append(('parity_voiced_flip', gm._unvoiced_rows(win2, eng4)))


def test_subset_flux(fails):
    """MEASURES: army-row waypoint semantics open ONLY inside combat
    chains — a casualty-then-downgrade flux row (Grond specimen:
    regular +3 mid-chain vs net +1) may continue when the chain holds
    a combat leg AND the region is one the window spoke of; the same
    row is refused for movement-only chains and for regions the
    window never mentioned."""
    win = gm.WindowFacts()
    win.armies[('minas tirith', 'gondor:regular')] = 1
    eng = gm.EngFacts()
    eng.armies[('minas tirith', 'gondor:regular')] = 3
    if gm._subset(eng, win):
        fails.append(('flux_strict_leak', True))
    if not gm._subset(eng, win, combat_flux=True):
        fails.append(('flux_combat_blocked', False))
    eng2 = gm.EngFacts()
    eng2.armies[('lorien', 'elves:elite')] = 1
    if gm._subset(eng2, win, combat_flux=True):
        fails.append(('flux_foreign_region', True))


def test_split_twin_disambiguation(fails):
    """MEASURES (specimen game 2023-01-07 T2 w019, the g2 root): ONE
    rendered group-move record must resolve to ONE engine move leg —
    the degenerate split twin ([move-half, move-half] to the same
    destination) implies a second drag the client never made, and the
    physical-equivalence collapse cannot merge it (different
    action-type tuples).  The pair-count filter must settle this as a
    MATCH with a single army_move, never 'ambiguous'."""
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
    ])
    w = window_from([
        '%GJB moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '%GJB moves group (Nazgul, Sauron Elite, [4]Sauron Regular) '
        'from Barad-Dur to Gorgoroth.',
        '%GJB kills Shadow Action Die (A) in Current Action.',
    ])
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status != 'match':
        fails.append(('split_twin_status', repr(res)))
        return
    moves = [a for a in res.chain if a.action_type == 'army_move']
    if len(moves) != 1:
        fails.append(('split_twin_legs',
                      [a.action_type for a in res.chain]))


def test_flux_political_tolerance(fails):
    """MEASURES (specimen game 2023-01-31 T2 w017, the g8 siege root):
    the attacked nation's p.34 political advance is ENGINE bookkeeping
    a client may never render — inside a combat chain an engine-extra
    political row must not refuse the cover, while a VOICED political
    row still demands exact parity; outside combat chains strict
    parity holds.  Same tolerance in _subset (the root attack's delta
    is pure political+die, and the strict bound pruned expansion)."""
    win = gm.WindowFacts()
    win.armies[('woodland realm', 'elves:elite')] = 0   # silent
    eng = gm.EngFacts()
    eng.political['ELVES'] = 1
    if gm._covers(win, eng):
        fails.append(('flux_pol_strict_leak', True))
    if not gm._covers(win, eng, combat_flux=True):
        fails.append(('flux_pol_blocked', False))
    # voiced rows keep exact parity even under flux
    win2 = gm.WindowFacts()
    win2.political['ELVES'] = 2
    if gm._covers(win2, eng, combat_flux=True):
        fails.append(('flux_pol_voiced_leak', True))
    # _subset: engine political the window never voiced may continue
    # ONLY inside a combat chain
    if gm._subset(eng, win):
        fails.append(('flux_pol_subset_strict_leak', True))
    if not gm._subset(eng, win, combat_flux=True):
        fails.append(('flux_pol_subset_blocked', False))


def test_memo_combat_context_tag(fails):
    """MEASURES: the exploration-memo context tag separates battles
    the continuation menus cannot (game 2023-01-25 T3: the Dale probe
    pruned the Old Forest Road attack's subtree — identical partial
    delta + identical [combat_card_none] menu).  Different battle
    regions/phases/rounds must produce different tags; no-combat
    states with equal pending kinds must produce equal tags."""
    class _P(object):
        def __init__(self, pc=None):
            if pc is not None:
                self.pending_combat = pc
    base = {'attacker_region': 'northern rhovanion',
            'defender_region': 'old forest road',
            'phase': 'attacker_combat_card', 'rounds_fought': 0}
    other = dict(base, defender_region='dale')
    t1 = gm._pending_ctx_tag(_P(base), 'combat')
    t2 = gm._pending_ctx_tag(_P(other), 'combat')
    if t1 == t2:
        fails.append(('ctx_tag_region_blind', t1))
    t3 = gm._pending_ctx_tag(_P(dict(base, rounds_fought=1)),
                             'combat')
    if t1 == t3:
        fails.append(('ctx_tag_round_blind', t1))
    if gm._pending_ctx_tag(_P(), 'x') != gm._pending_ctx_tag(_P(),
                                                            'x'):
        fails.append(('ctx_tag_unstable', True))
    if gm._pending_ctx_tag(_P(), 'x') == gm._pending_ctx_tag(_P(),
                                                            'y'):
        fails.append(('ctx_tag_kind_blind', True))


def test_relax_nations_lane(fails):
    """MEASURES (specimen game 2023-01-18 T2 w018, the g4 root):
    a mixed-nation drag (1 Isengard + 1 Sauron regular) is UNREACHABLE
    from the engine menu (move_army's split take() has no composition
    parameter) — strict matching refuses it; the relax_nations lane
    matches on nation-blind (region, kind) totals so the spectator can
    commit the closest legal composition under a NAMED engine-gap row.
    Integration: a real mixed army, real window, both modes."""
    import game_logic as gl
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
    ])
    army = gl.get_or_create_army(st, 'Moria')
    # Moria starts Sauron-held; add one Isengard regular to mix it
    army.units.append(gs.Unit(nation=gs.Nation.ISENGARD,
                              kind=gs.UnitKind.REGULAR))
    w = window_from([
        '%M moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '%M moves group (Isengard Regular, Sauron Regular) from '
        'Moria to Dimrill Dale.',
        '%M kills Shadow Action Die (A) in Current Action.',
    ])
    res = gm.match_window(st, Side.SHADOW, w)
    if res.status == 'match':
        fails.append(('relax_nations_strict_matched',
                      'engine grew a composition parameter? retire '
                      'the lane' ))
        return
    res2 = gm.match_window(st, Side.SHADOW, w, relax_nations=True)
    if res2.status != 'match':
        fails.append(('relax_nations_no_match', repr(res2)))
        return
    eng = gm.delta_facts(st, res2.post)
    nb_win = gm._nation_blind(gm.window_facts(w).armies)
    nb_eng = gm._nation_blind(eng.armies)
    if nb_win != nb_eng:
        fails.append(('relax_nations_totals', (nb_win, nb_eng)))


def test_cross_side_hand_play(fails):
    """MEASURES (specimen game 2023-01-28 T2 w015): a hand->play move
    whose HAND side differs from the staged die's side is the OTHER
    seat's COMBAT card (zone-derived attribution), never a play_event
    vocabulary constraint; same-side plays stay cards_played."""
    w = window_from([
        '%feebleblobber moves Shadow Action Die (A) from SA Dice to '
        'Current Action.',
        '%savedbyhim moves The Spirit of Mordor/Scouts from Free '
        'Peoples Hand to Current FP Card.',
        '%feebleblobber kills Shadow Action Die (A) in Current '
        'Action.',
    ])
    f = gm.window_facts(w)
    if f.cards_played:
        fails.append(('cross_side_as_play', list(f.cards_played)))
    if f.combat_cards != [('FP', 'The Spirit of Mordor/Scouts')]:
        fails.append(('cross_side_combat', list(f.combat_cards)))
    w2 = window_from([
        '%X moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '%X moves Elven Rope/It is a Gift from Free Peoples Hand to '
        'Current FP Card.',
        '%X kills Free Action Die (P) in Current Action.',
    ])
    f2 = gm.window_facts(w2)
    if f2.combat_cards or not f2.cards_played:
        fails.append(('same_side_as_combat',
                      (list(f2.combat_cards), list(f2.cards_played))))


def test_zero_hit_battle_gate(fails):
    """MEASURES (specimen game 2023-01-31: 'none SHADOW die=H ::
    <game> rolls'): a fact-empty window WITH roll records must not be
    refused as 'expressed nothing' — the rolls are the battle's only
    observable; a fact-empty window WITHOUT rolls keeps the refusal."""
    st = adopted_state()
    w = window_from([
        '%X moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '%X kills Shadow Action Die (A) in Current Action.',
    ])
    res = gm.match_window(st, Side.SHADOW, w)
    if 'expressed nothing' not in (res.note or ''):
        fails.append(('empty_no_rolls', repr(res)))
    w2 = window_from([
        '%X moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '<game> X rolls  1 2 1',
        '%X kills Shadow Action Die (A) in Current Action.',
    ])
    res2 = gm.match_window(st, Side.SHADOW, w2)
    if 'expressed nothing' in (res2.note or ''):
        fails.append(('rolls_only_refused', repr(res2)))


# ---------------------------------------------------------------------
# FIX WAVE 1 (agent B) pins: ids = audit FINDINGS_FULL.md W1-playpath#n
# ---------------------------------------------------------------------
def test_pending_buffer_hand_play_is_combat_card(fails):
    """MEASURES (W1#30): a hand -> play-area DRAG buffered while an
    engine pending is open (a PhaseWindow: no staged die) is the
    hand's side's COMBAT card, never an event play - no die was
    staged, so no event play is possible.  SPECIFICITY: the die-window
    same-side drag stays cards_played (the event play)."""
    from wotr_bridge.live.driver import PhaseWindow
    pw = PhaseWindow()
    pw.records.append(live_parser.parse(
        '$Wey moves The Red Arrow/Scouts from Free Peoples Hand to '
        'Current FP Card.'))
    f = gm.window_facts(pw)
    if f.combat_cards != [('FP', 'The Red Arrow/Scouts')]:
        fails.append(('buffer_drag_combat', list(f.combat_cards)))
    if f.cards_played:
        fails.append(('buffer_drag_as_play', list(f.cards_played)))
    w = window_from([
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey moves The Red Arrow/Scouts from Free Peoples Hand to '
        'Current FP Card.',
        '$Wey kills Free Action Die (P) in Current Action.',
    ])
    f2 = gm.window_facts(w)
    if f2.combat_cards or f2.cards_played != ['The Red Arrow/Scouts']:
        fails.append(('die_window_drag_play',
                      (list(f2.combat_cards), list(f2.cards_played))))


def test_chooser_event_play_in_die_window(fails):
    """MEASURES (W1#31 / K1 step 3): the client chooser's '(chosen)'
    reveal inside the player's OWN die window is an EVENT play when
    no combat-card phase is open (CardBuffet.9; wkf1 specimen
    L67-68), a COMBAT card when one is (the corpus battle grammar),
    and - with no context stated (the spectator) - keeps the combat
    reading."""
    lines = [
        '$WOTR_AI_FP moves Free Action Die (P) from FP Dice to '
        'Current Action.',
        '$Wey chooses a card.[Free Character Card]',
        '<auto> moves Mirror of Galadriel/Heroic Death from Free '
        'Peoples Hand (chosen) to Current FP Card.',
        '$WOTR_AI_FP kills Free Action Die (P) in Current Action.',
    ]
    w = window_from(lines)
    if w is None:
        fails.append(('chooser_window', 'no window'))
        return
    w.combat_cards_open = False
    f = gm.window_facts(w)
    if f.cards_played != ['Mirror of Galadriel/Heroic Death'] \
            or f.combat_cards:
        fails.append(('chooser_event_play',
                      (list(f.cards_played), list(f.combat_cards))))
    w.combat_cards_open = True
    f2 = gm.window_facts(w)
    if f2.combat_cards != [('FP', 'Mirror of Galadriel/Heroic Death')] \
            or f2.cards_played:
        fails.append(('chooser_combat_when_open',
                      (list(f2.cards_played), list(f2.combat_cards))))
    w3 = window_from(lines)
    f3 = gm.window_facts(w3)
    if not f3.combat_cards:
        fails.append(('chooser_spectator_default',
                      (list(f3.cards_played), list(f3.combat_cards))))
    # the OTHER seat's chooser reveal inside this window is always a
    # combat card (the defender picks inside the attacker's window)
    w4 = window_from([
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '<auto> moves The Red Arrow/Scouts from Free Peoples Hand '
        '(chosen) to Current FP Card.',
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ])
    w4.combat_cards_open = False
    f4 = gm.window_facts(w4)
    if f4.combat_cards != [('FP', 'The Red Arrow/Scouts')]:
        fails.append(('chooser_other_seat', (list(f4.cards_played),
                                             list(f4.combat_cards))))


def test_refusal_diff_compares_like_with_like(fails):
    """MEASURES (W1#32): the refusal explainer names ONLY the fact
    categories that really differ.  A window whose engine candidate
    matches on every row yields [] (no phantom 'the card played' /
    'the action die'); a draw-count mismatch yields exactly 'card
    draws'.  Built from a REAL menu draw_card action applied on a
    clone (engine-shaped delta)."""
    import ai
    st = adopted_state()
    for d in st.dice[Side.FREE_PEOPLES]:
        d.current = ActionDieFace.EVENT
    menu = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    dc = next((a for a in menu if a.action_type == 'draw_card'
               and str(a.params.get('deck', '')).startswith('char')),
              None)
    if dc is None:
        fails.append(('refusal_diff_control',
                      'NOT-MEASURED: no character draw_card on an '
                      'Event die'))
        return
    post = ai.apply_action(st, dc)
    got = gm.delta_facts(st, post)
    w = window_from([
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey draws Free Character Card.',
        '$Wey kills Free Action Die (P) in Current Action.',
    ])
    want = gm.window_facts(w)
    diff = gm._facts_differ(want, got)
    if diff:
        fails.append(('refusal_diff_phantom', diff))
    w2 = window_from([
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey draws Free Character Card.',
        '$Wey draws Free Character Card.',
        '$Wey kills Free Action Die (P) in Current Action.',
    ])
    want2 = gm.window_facts(w2)
    diff2 = gm._facts_differ(want2, got)
    if diff2 != ['card draws']:
        fails.append(('refusal_diff_specific', diff2))


def main():
    tests = [test_zone_key, test_window_facts_specimens,
             test_match_real_army_move, test_match_none_illegal,
             test_budget_hold, test_unmapped_hold, test_fs_move_match,
             test_draw_window, test_political_window,
             test_flip_hide_window, test_card_play_window,
             test_two_army_window, test_ring_zone_not_fellowship,
             test_siege_fact_row, test_relax_draws_bounds,
             test_crown_flip_transform, test_follow_through_table,
             test_exact_parity_beats_tolerance_fit, test_subset_flux,
             test_play_event_refusal_names_the_reason,
             test_bl150_chain_stops_before_a_discard_pending,
             test_split_twin_disambiguation,
             test_flux_political_tolerance,
             test_memo_combat_context_tag,
             test_relax_nations_lane,
             test_cross_side_hand_play,
             test_zero_hit_battle_gate,
             test_pending_buffer_hand_play_is_combat_card,
             test_chooser_event_play_in_die_window,
             test_refusal_diff_compares_like_with_like]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', default=None)
    args = ap.parse_args()
    if args.k:
        tests = [t for t in tests if args.k in t.__name__]
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
