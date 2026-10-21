"""wotr_bridge.live core instruments: feed_reader + live_parser.

Instrument protocol: measured quantity stated per test; every zero
ships with its non-zero control; real client material, never synthetic
grammar, wherever a real specimen exists (the AI's grammar is not the
human's grammar).
"""
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)

from wotr_bridge.live import live_parser
from wotr_bridge.live.feed_reader import FeedReader, FeedRestart

_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                    'audit')
_REAL_FEEDS = [
    os.path.join(_LAB, 'blind_run', 'feed_blinded.txt'),
    os.path.join(_LAB, 'noflag_run', 'feed_noflag.txt'),
    os.path.join(_LAB, 'sided_run', 'feed_F1_full.txt'),
    os.path.join(_LAB, 'sided_run', 'feed_G1_nazgul_postadopt.txt'),
    os.path.join(_LAB, 'client_run', 'feed_live.txt'),
]


def test_parser_on_real_feeds(fails):
    """MEASURES: the unknown-shape count of live_parser.parse over REAL
    captured AiFeed output (live channel).  Non-zero control: the run
    must classify a stated minimum of board records (move+kill+draw all
    present) or it reports NOT-MEASURED instead of a hollow zero."""
    counts = {}
    unknown = []
    seen_files = 0
    for path in _REAL_FEEDS:
        if not os.path.exists(path):
            continue
        seen_files += 1
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for raw in f.read().splitlines():
                rec = live_parser.parse(raw)
                counts[rec['op']] = counts.get(rec['op'], 0) + 1
                if rec.get('unknown'):
                    unknown.append(rec['raw'])
    if seen_files == 0:
        fails.append(('real_feeds', 'NOT-MEASURED: no specimens found'))
        return
    total_board = sum(v for k, v in counts.items()
                      if k not in ('chat', 'blank', 'system', 'noop'))
    print('  [feeds] files=%d ops=%s' % (seen_files, sorted(
        (k, v) for k, v in counts.items())))
    for k in ('move', 'kill', 'draw'):
        if counts.get(k, 0) == 0:
            fails.append(('real_feeds_control',
                          'NOT-MEASURED: zero %r records — the zero '
                          'unknowns below would be unreadable' % k))
            return
    if total_board < 50:
        fails.append(('real_feeds_control',
                      'NOT-MEASURED: only %d board records' % total_board))
        return
    if unknown:
        uniq = sorted(set(unknown))
        fails.append(('real_feeds_unknowns',
                      '%d unknown lines, %d distinct; first 5: %r'
                      % (len(unknown), len(uniq), uniq[:5])))


def test_parser_prefix_forms(fails):
    """MEASURES: each live prefix form maps to its documented record
    (A15 table): '$Nick move', '<auto> move', '<game> roll', chat,
    replay '%' tolerance, system tags, gate-2 classified others."""
    cases = [
        ('$Wey moves Free Action Die (P) from FP Dice to Current '
         'Action.', 'move'),
        ('<auto> moves North Regular from Board2 to Spare.', 'move'),
        ('<game> Wey rolls 3 1 4', 'roll_num'),
        ('<Wey> fp ok', 'chat'),
        ('%Wey draws Free Character Card.', 'draw'),
        ('$Wey draws Shadow Strategy Card.', 'draw'),
        ('<replay> Turn 3', 'turn'),
        ('<address> NETWORK: PC/1.2.3.4', 'system'),
        ('$Wey views the Shadow event hand.', 'noop'),
        ('$Wey changes Shadow Action Die (E) in SA Dice.',
         'die_change'),
        ('$Wey randomly discards a Free Strategy Card.',
         'random_discard'),
        ('$Wey performs an act with no grammar whatsoever', 'other'),
        # replay-channel forms: '%' prefixes every line kind (A15)
        ('%<GenericPie> pass', 'chat'),
        ('%<game> Sirrus233 rolls  3 1 4', 'roll_num'),
        ('%auto> moves North Regular from Board2 to Spare.', 'move'),
    ]
    for raw, want in cases:
        rec = live_parser.parse(raw)
        if rec['op'] != want:
            fails.append(('prefix_form', '%r -> %r, want %r'
                          % (raw, rec['op'], want)))
    rec = live_parser.parse('$Wey performs an act with no grammar '
                            'whatsoever')
    if not rec.get('unknown'):
        fails.append(('prefix_unknown_flag',
                      'unclassifiable line not flagged unknown'))
    rec = live_parser.parse('<Wey> fp ok')
    if rec.get('nick') != 'Wey' or rec.get('body') != 'fp ok':
        fails.append(('chat_fields', repr(rec)))


def test_feed_reader_tail(fails):
    """MEASURES: poll() yields exactly the appended complete lines, in
    order, once each; an unterminated tail is held back until its
    newline arrives."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'feed.txt')
        r = FeedReader(p)
        if r.poll() != []:
            fails.append(('tail_absent', 'poll on missing file'))
        with open(p, 'wb') as f:
            f.write(b'$A moves X from Y to Z.\n$B kills W in V.\npart')
        got = r.poll()
        if got != ['$A moves X from Y to Z.', '$B kills W in V.']:
            fails.append(('tail_first', repr(got)))
        with open(p, 'ab') as f:
            f.write(b'ial line\n')
        got = r.poll()
        if got != ['partial line']:
            fails.append(('tail_partial', repr(got)))
        if r.poll() != []:
            fails.append(('tail_idle', 'idle poll not empty'))
        if r.lines != ['$A moves X from Y to Z.', '$B kills W in V.',
                       'partial line']:
            fails.append(('tail_history', repr(r.lines)))


def test_feed_reader_truncate_reemit(fails):
    """MEASURES client fact 21: a truncation followed by an identical
    prefix re-emit resumes with no duplicate and no loss; a divergent
    or shortened re-emit raises FeedRestart (sensitivity control)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'feed.txt')
        r = FeedReader(p)
        with open(p, 'wb') as f:
            f.write(b'line one\nline two\n')
        r.poll()
        # re-attach: file truncated, history re-emitted, new line after
        with open(p, 'wb') as f:
            f.write(b'line one\nline two\nline three\n')
        got = r.poll()
        if got != ['line three']:
            fails.append(('reemit_resume', repr(got)))
        # divergent re-emit -> FeedRestart
        with open(p, 'wb') as f:
            f.write(b'DIFFERENT\n')
        try:
            r.poll()
            fails.append(('reemit_divergent', 'no FeedRestart raised'))
        except FeedRestart:
            pass
        # sensitivity 2: shortened re-emit on a fresh reader
        r2 = FeedReader(p)
        with open(p, 'wb') as f:
            f.write(b'a\nb\nc\n')
        r2.poll()
        with open(p, 'wb') as f:
            f.write(b'a\n')
        try:
            r2.poll()
            fails.append(('reemit_shortened', 'no FeedRestart raised'))
        except FeedRestart:
            pass


def test_feed_reader_append_reattach(fails):
    """MEASURES (W1#19): the two re-emission shapes the shrink check
    cannot see, each surfaced as FeedRestart (rule 12), never read as
    a continuation:
    (1) APPEND re-attach: the agent re-emits a NEW chat document from
        paragraph 0 onto the SAME feed stream (the file only grows) -
        the client's own new-game announcement ('<game> X has started
        a new War of the Ring game.') arriving again in-stream is the
        marker;
    (2) truncate-then-longer: a premain truncation followed by a
        longer re-emission (size >= the old offset) - the file HEAD
        no longer matches what was consumed.
    Specificity: a genuine continuation (new lines appended, head
    intact) keeps streaming; the very first announcement is not a
    restart."""
    import tempfile
    start = ('<game> Wey has started a new War of the Ring game.'
             '[WOTR-Gandalf2 wotr base2]\n')
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'feed.txt')
        r = FeedReader(p)
        with open(p, 'wb') as f:
            f.write((start + '<replay> Turn 1\n'
                     '$Wey draws Free Character Card.\n')
                    .encode('utf-8'))
        got = r.poll()
        if len(got) != 3:
            fails.append(('append_first_read', repr(got)))
        # specificity: a plain continuation streams
        with open(p, 'ab') as f:
            f.write(b'$Wey draws Free Strategy Card.\n')
        if r.poll() != ['$Wey draws Free Strategy Card.']:
            fails.append(('append_continuation', 'continuation lost'))
        # (1) a new game's document re-emitted by APPEND
        with open(p, 'ab') as f:
            f.write((start + '<replay> Turn 1\n').encode('utf-8'))
        try:
            got = r.poll()
            fails.append(('append_reattach_undetected',
                          'new-game re-emission read as continuation: '
                          '%r' % (got,)))
        except FeedRestart:
            pass
        # (2) truncate, then a LONGER re-emission of a different game
        r2 = FeedReader(p)
        with open(p, 'wb') as f:
            f.write((start + '$Wey draws Free Character Card.\n')
                    .encode('utf-8'))
        r2.poll()
        with open(p, 'wb') as f:
            f.write(('<game> Bob has started a new War of the Ring '
                     'game.[WOTR-Gandalf2 wotr base2]\n'
                     '<replay> Turn 1\n$Bob draws Free Character '
                     'Card.\n$Bob draws Free Strategy Card.\n')
                    .encode('utf-8'))
        try:
            got = r2.poll()
            fails.append(('truncate_longer_undetected',
                          'read resumed mid-stream: %r' % (got,)))
        except FeedRestart:
            pass


def main():
    tests = [test_parser_on_real_feeds, test_parser_prefix_forms,
             test_feed_reader_tail, test_feed_reader_truncate_reemit,
             test_feed_reader_append_reattach]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
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
