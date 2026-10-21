"""SpectatorDriver instruments (the corpus-replay census loop).

MEASURED QUANTITIES per test.  Feeds are synthetic but written in the
REAL replay grammar ('%' channel) and ingested through the REAL
pipeline (feed file -> live_parser -> Segmenter -> SpectatorDriver),
so a grammar shape the parser would not classify fails the test
instead of testing nothing.
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

from wotr_bridge.live import spectator  # noqa: E402


HEADER = [
    '<game> Wey has started a new War of the Ring game.'
    '[WOTR-Gandalf2 wotr base2]',
    '%<replay> Turn 1 ',
    '%FPP draws Free Character Card.',
    '%FPP draws Free Strategy Card.',
    '%SHP draws Shadow Character Card.',
    '%SHP draws Shadow Strategy Card.',
    # CORPUS eye-allocation form (client fact 12): the plain drag
    '%SHP moves Shadow Action Die (E) from SA Dice to The Hunt For '
    'The Ring.',
    '%SHP rolls H P M H P H .',
    '%SHP adds 0 eyes to the hunt.',
    '%FPP rolls P P M H .',
]


def ingest_lines(lines):
    """Run the lines through the real ingest; returns (census, driver).
    The driver is captured via a hook so engine state is inspectable."""
    got = {}
    orig = spectator.SpectatorDriver.census

    def hooked(self):
        got['drv'] = self
        return orig(self)
    spectator.SpectatorDriver.census = hooked
    try:
        fd, path = tempfile.mkstemp(suffix='.feed.txt')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        try:
            c = spectator.ingest_feed_file(path)
        finally:
            os.unlink(path)
    finally:
        spectator.SpectatorDriver.census = orig
    return c, got['drv']


def test_eye_drag_allocation(fails):
    """MEASURES (games 2023-01-13/-18/-25/-28 root): the corpus
    eye-allocation DRAG puts one eye in the hunt box BEFORE the roll,
    the 6-face roll then resolves its side by count and adopts, and a
    turn-1 political window MATCHES (the menu is not empty)."""
    import game_state as gs
    lines = HEADER + [
        '%SHP moves Shadow Action Die (M) from SA Dice to Current '
        'Action.',
        '%SHP moves Isengard from Political Track 1 to Political '
        'Track 0.',
        '%SHP kills Shadow Action Die (M) in Current Action.',
    ]
    c, drv = ingest_lines(lines)
    if len(getattr(drv.es, 'hunt_box', [])) != 1:
        fails.append(('eye_in_box', len(getattr(drv.es, 'hunt_box',
                                                []))))
    faces = [getattr(d.current, 'name', None)
             for d in drv.es.dice[gs.Side.SHADOW]]
    if faces.count(None) != 0:
        fails.append(('roll_adopted', faces))
    if c['windows'].get('matched', 0) < 1:
        fails.append(('political_window', c['windows']))
    if any(k == 'eye_alloc_failed' for k, _ in c['findings']):
        fails.append(('eye_alloc_failed_rows', True))


def test_relaxed_draw_named_row(fails):
    """MEASURES (game 2023-01-13 misplay, raw-log-proven): a card play
    whose window renders one draw MORE than the effect grants commits
    the legal subset and the surplus gets the NAMED row
    client_unowed_draw — never a silent 'none'.  In the fresh state
    Gandalf the Grey guides, so an Event-die card play OWES two draws
    (the card's own + the guide's same-deck draw); the THIRD draw is
    the surplus this instrument measures."""
    lines = HEADER + [
        '%FPP moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '%FPP moves Kindred of Glorfindel/Advantageous Position from '
        'Free Peoples Hand to Current FP Card.',
        '%FPP moves Elven Elite from FP Reinforcements to Rivendell.',
        '%FPP draws Free Strategy Card.',
        '%FPP draws Free Strategy Card.',
        '%FPP draws Free Strategy Card.',
        '%FPP kills group (Free Action Die (P), Kindred of '
        'Glorfindel/Advantageous Position) in Current Action.',
    ]
    c, _drv = ingest_lines(lines)
    if c['windows'].get('matched', 0) < 1:
        fails.append(('kindred_match', c['windows']))
    if not any(k == 'client_unowed_draw' for k, _ in c['findings']):
        fails.append(('named_row', [k for k, _ in c['findings']]))


def test_unreveal_on_failed_window(fails):
    """MEASURES (game 2023-01-13 cascade root): a card play whose
    window cannot match leaves NO real cid stranded in the engine
    hand — the reveal-swap is rolled back, so later menu builds carry
    no phantom playable card (the dry-run starvation class)."""
    import game_state as gs
    lines = HEADER + [
        '%FPP moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '%FPP moves Kindred of Glorfindel/Advantageous Position from '
        'Free Peoples Hand to Current FP Card.',
        # impossible board effect: Kindred recruits in Rivendell only
        '%FPP moves Elven Elite from FP Reinforcements to Umbar.',
        '%FPP kills group (Free Action Die (P), Kindred of '
        'Glorfindel/Advantageous Position) in Current Action.',
    ]
    c, drv = ingest_lines(lines)
    if c['windows'].get('matched', 0) != 0:
        fails.append(('bogus_matched', c['windows']))
    hand = drv.es.hands[gs.Side.FREE_PEOPLES].hand
    real = [x for x in hand
            if not str(x).startswith('HU_')]
    if real:
        fails.append(('cid_stranded', real))


def test_rand_ctl_cursor_restore(fails):
    """MEASURES: the matcher's rng controller restores feed cursors on
    backtrack — after save/consume/restore, a fresh consumer reads the
    SAME d6 values (sensitivity control: without the restore the
    second read sees the tail)."""
    from wotr_bridge.live import randomness as rnd_mod
    rand = rnd_mod.ClientRandomness(seed=7)
    ctl = spectator._RandCtl(rand)
    rand.feed('d6', [1, 2, 3])
    rand.feed('d6', [4, 5])
    ctl.root_attempt()
    snap = ctl.save()
    with ctl.apply_cm():
        first = rand._consume_d6(3, ctx='t')
    ctl.restore(snap)
    with ctl.apply_cm():
        second = rand._consume_d6(3, ctx='t')
    if first != [1, 2, 3] or second != [1, 2, 3]:
        fails.append(('cursor_restore', (first, second)))
    # specificity: WITHOUT restore the cursor must advance
    with ctl.apply_cm():
        third = rand._consume_d6(2, ctx='t')
    if third != [4, 5]:
        fails.append(('cursor_advances', third))


def test_silent_declare_decline(fails):
    """MEASURES: with no declare gesture before the action roll, the
    engine's declare pending settles as the decline and the turn
    reaches the actions phase (absence-is-the-choice)."""
    c, drv = ingest_lines(HEADER)
    if drv.phase != 'actions':
        fails.append(('phase', drv.phase))
    import ai
    if ai.describe_pending(drv.es) is not None:
        fails.append(('pending_open',
                      ai.describe_pending(drv.es)['kind']))


def test_illegal_muster_named(fails):
    """MEASURES (game 2023-01-31 T2, Elves one step short of At War):
    a DICE-window recruit for a not-At-War nation commits nothing and
    gets the NAMED row client_illegal_muster (p.26/CL-084); the row
    must not fire on card windows (card effects place freely)."""
    lines = HEADER + [
        '%FPP moves Free Action Die (M) from FP Dice to Current '
        'Action.',
        '%FPP moves Gondor Elite from FP Reinforcements to Minas '
        'Tirith.',
        '%FPP kills Free Action Die (M) in Current Action.',
    ]
    c, _drv = ingest_lines(lines)
    rows = [d for k, d in c['findings']
            if k == 'client_illegal_muster']
    if not rows:
        fails.append(('no_named_row',
                      [k for k, _ in c['findings']]))
    elif 'Gondor' not in rows[0]:
        fails.append(('wrong_nation', rows[0][:80]))
    if c['windows'].get('matched', 0) != 0:
        fails.append(('illegal_committed', c['windows']))


def test_cosmetic_marker_class(fails):
    """MEASURES (game 2023-01-22 residual; USER broken-vs-messy
    directive): an FSP-MARKER-only mismatch at a turn boundary is
    reported as 'marker_cosmetic', counted in census cosmetic_rows,
    and does NOT set first_divergent_turn — fdt reflects BREAKAGE."""
    lines = HEADER + [
        # a stray FSP marker drag with NO fellowship action behind it:
        # the mirror follows the marker, the engine does not move
        '%FPP moves FSP from Fellowship Track 0 to Fellowship '
        'Track 1.',
        '%<replay> Turn 2 ',
    ]
    c, _drv = ingest_lines(lines)
    if c.get('cosmetic_rows', 0) != 1:
        fails.append(('cosmetic_count', c.get('cosmetic_rows')))
    if c['first_divergent_turn'] is not None:
        fails.append(('fdt_polluted', c['first_divergent_turn']))
    # the row is still REPORTED (never hidden)
    t2 = [rows for t, _n, rows in c['turn_reports'] if t == 2]
    if not t2 or not any(k == 'marker_cosmetic'
                         for k, _ in (t2[0] or [])):
        fails.append(('row_missing', c['turn_reports']))


def main():
    tests = [test_eye_drag_allocation, test_relaxed_draw_named_row,
             test_unreveal_on_failed_window,
             test_rand_ctl_cursor_restore,
             test_silent_declare_decline,
             test_illegal_muster_named,
             test_cosmetic_marker_class]
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
