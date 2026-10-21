#!/usr/bin/env python3
"""Audit 2026-08-29 regression pins - BRJAVA family (the Java client mod's
sequencing / ack / error channel, consumed by wotr_bridge/live/client_protocol.py).

Pure-function pins against fixture text: the exact header / ack / feed
shapes the CANDIDATE jar (WOTR-Gandalf-AI.jar.CANDIDATE_seqack_20260829)
writes.  No client is launched (I4 forbids headless; the jar leg is
separate).  main() runs them all and prints TOTAL=<n> FAIL=<m>; exit 1 on
any failure.

Findings pinned (ids from BRJAVA_report.md):
  BRJAVA-1  failed client actions are recognised and yield a HALT verdict;
            elapsed-wait notices do not discard a still-valid late response
  BRJAVA-2  the #SEQ header ties a dump to a feed position; a dump behind
            or ahead of the consumed feed is refused
  BRJAVA-3  per-line ack records: unacked lines, refused (zero-paragraph /
            exception / from-zone mismatch) records
  BRJAVA-4  REQ/RESP sequence echo helpers (req in the REQ header, the
            RESP echo line, stale-drop report)
  BRJAVA-7  session nonce: cross-session artefacts are refused
  BRJAVA-9  the dump's 5th column (marker state via H.k()) reads back

cp1252 console: ASCII only.
"""

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIVE = os.path.abspath(os.path.join(_HERE, '..', 'wotr_bridge', 'live'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..', 'wotr_bridge'))
for _p in (_HERE, _BRIDGE, _LIVE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import client_protocol as cp  # noqa: E402

_scratch_owner = tempfile.TemporaryDirectory(prefix='wotr-brjava-')
SCRATCH = os.environ.get('WOTR_SCRATCH', _scratch_owner.name)
os.makedirs(SCRATCH, exist_ok=True)


# ---------------------------------------------------------------------------
# fixtures: the exact text shapes the candidate jar writes
# ---------------------------------------------------------------------------

DUMP_HDR = ('#SEQ dump=12 feed_paragraphs=83 raw_lines=109 '
            'ts=1756500000123 session=1756499000000')
REQ_HDR = ('#SEQ req=7 dump=12 feed_paragraphs=83 raw_lines=109 '
           'ts=1756500000123 session=1756499000000 dropped_resp=5')
DUMP_BODY = [
    '121\t<~Game.2004~>\t155 152\tU U\t- -',
    '178\t<~Controls.45~>\t100 101 102 103\t- - - -\t- - - -',
    '85\t<~Game.1965~>\t312\t-\t2',           # a political marker at step 2
]

ACKS = [
    '{"session":1756499000000,"req":7,"i":0,"kind":"line",'
    '"raw":"$Wey<~Scribe.141~>312 <~Scribe.142~><~Game.2038~>'
    '<~Scribe.143~><~Game.1965~><~Scribe.144~>",'
    '"paragraphs_before":83,"paragraphs_after":84,"ok":true}',
    '{"session":1756499000000,"req":7,"i":1,"kind":"line",'
    '"raw":"$Wey bogusverb 1 2 3","paragraphs_before":84,'
    '"paragraphs_after":84,"ok":true}',
    '{"session":1756499000000,"req":7,"i":2,"kind":"call",'
    '"raw":"!CALL roll_action_dice Game.9999","paragraphs_before":84,'
    '"paragraphs_after":85,"ok":false,'
    '"error":"<game> AI ERROR: no zone Game.9999"}',
    '{"session":1756499000000,"req":7,"i":3,"kind":"line",'
    '"raw":"$Wey<~Scribe.141~>400 <~Scribe.142~><~Game.2040~>'
    '<~Scribe.143~><~Game.2041~><~Scribe.144~>",'
    '"paragraphs_before":85,"paragraphs_after":86,"ok":true,'
    '"claimed_from":"<~Game.2040~>","real_from":"400=<~Game.2039~>"}',
    '{"session":1756490000000,"req":3,"i":0,"kind":"line",'
    '"raw":"$Wey old session line","paragraphs_before":10,'
    '"paragraphs_after":11,"ok":true}',
]

RESP_LINES = [
    '$Wey<~Scribe.141~>312 <~Scribe.142~><~Game.2038~><~Scribe.143~>'
    '<~Game.1965~><~Scribe.144~>',
    '$Wey bogusverb 1 2 3',
    '!CALL roll_action_dice Game.9999',
    '$Wey<~Scribe.141~>400 <~Scribe.142~><~Game.2040~><~Scribe.143~>'
    '<~Game.2041~><~Scribe.144~>',
    '$Wey<~Scribe.141~>401 <~Scribe.142~><~Game.2040~><~Scribe.143~>'
    '<~Game.2041~><~Scribe.144~>',      # never acked (5th line)
    'PASS',
]

FEED_CLEAN = [
    '#BLIND aiside=SA hands=FP',
    '<game> Wey has started a new War of the Ring game',
    '$Wey moves Gondor Regular from Minas Tirith to Osgiliath.',
    '<game> Playing vs AI (Shadow). AI -> Stop AI to halt.',
    '<game> AI menu ready - use AI -> Play vs AI to start.',
    '<game> AI server already running - reusing it.',
    '<WOTR_AI_Shadow> I attack Osgiliath.',
]

FEED_ERRORS = {
    'no zone': '<game> AI ERROR: no zone Game.9999',
    'directive failed': "<game> AI ERROR: directive 'play_card x' failed: "
                        'java.lang.NumberFormatException',
    'injection failed': '<game> AI: injection failed: '
                        'java.lang.NullPointerException',
    'loop error': '<game> AI loop error: java.lang.InterruptedException',
    'rendered nothing': '<game> AI: injection rendered nothing: '
                        '$Wey bogusverb 1 2 3',
    'game replaced': '<game> AI: game window replaced (File > New / Load) '
                     '- live loop stopped; click AI -> Play vs AI again.',
    'modal': "<game> AI: modal dialog open: title='Error' text='Error in "
             "the card signiture' (injecting=true)",
}


def _fail(msg):
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# BRJAVA-2 / BRJAVA-7: the #SEQ header
# ---------------------------------------------------------------------------

def test_parse_seq_header_dump():
    h = cp.parse_seq_header(DUMP_HDR)
    if h is None:
        _fail('dump header not parsed')
    for k, v in (('dump', 12), ('feed_paragraphs', 83), ('raw_lines', 109),
                 ('ts', 1756500000123), ('session', 1756499000000)):
        if h.get(k) != v:
            _fail('%s: %r != %r' % (k, h.get(k), v))
    if 'req' in h:
        _fail('a dump header carries no req')
    if cp.parse_seq_header('121\t<~Game.2004~>\t155 152\tU U') is not None:
        _fail('a body row parsed as a header')
    if cp.parse_seq_header('#SEQ dump=x') is not None:
        _fail('a malformed header parsed')


def test_parse_seq_header_req():
    h = cp.parse_seq_header(REQ_HDR)
    if h is None or h.get('req') != 7 or h.get('dropped_resp') != 5:
        _fail('req header: %r' % (h,))


def test_split_seq_header_strips_line_two():
    lines = [REQ_HDR, '<auto> silent null', '$Wey moves ...']
    h, rest = cp.split_seq_header(lines)
    if h is None or h['req'] != 7:
        _fail('header not split: %r' % (h,))
    if rest != lines[1:]:
        _fail('rest wrong: %r' % (rest,))
    h2, rest2 = cp.split_seq_header(lines[1:])
    if h2 is not None or rest2 != lines[1:]:
        _fail('header-less request altered: %r %r' % (h2, rest2))


def test_dump_at_feed_position():
    h = cp.parse_seq_header(DUMP_HDR)
    # 83 non-empty paragraphs + the #BLIND line = 84 consumed feed lines
    if not cp.dump_at_feed_position(h, 84):
        _fail('dump AT the feed position refused')
    if cp.feed_position_delta(h, 84) != 0:
        _fail('delta at position != 0')
    if cp.dump_at_feed_position(h, 85):
        _fail('dump BEHIND the feed (one paragraph short) accepted')
    if cp.feed_position_delta(h, 85) != -1:
        _fail('behind delta != -1')
    if cp.dump_at_feed_position(h, 80):
        _fail('dump AHEAD of the consumed feed accepted')
    if cp.feed_position_delta(h, 80) != 4:
        _fail('ahead delta != 4')
    if cp.dump_at_feed_position(None, 84):
        _fail('a header-less dump must never count as at-position')


def test_read_dump_header_from_file():
    path = os.path.join(SCRATCH, 'brjava_pin_dump.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(DUMP_HDR + '\n' + '\n'.join(DUMP_BODY) + '\n')
    h = cp.read_dump_header(path)
    if h is None or h['dump'] != 12:
        _fail('dump header not read back: %r' % (h,))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(DUMP_BODY) + '\n')
    if cp.read_dump_header(path) is not None:
        _fail('a legacy dump (no header) must read as None')


def test_session_matches():
    h = cp.parse_seq_header(DUMP_HDR)
    if not cp.session_matches(h, 1756499000000):
        _fail('same session refused')
    if cp.session_matches(h, 1756490000000):
        _fail('cross-session artefact accepted')
    if cp.session_matches(h, None) or cp.session_matches(None, 1):
        _fail('an unknown session must never match')


# ---------------------------------------------------------------------------
# BRJAVA-3: ack records
# ---------------------------------------------------------------------------

def _write_acks(name, torn_tail=False):
    path = os.path.join(SCRATCH, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ACKS) + '\n')
        if torn_tail:
            f.write('{"session":1756499000000,"req":8,"i":0,"ki')
    return path


def test_read_acks_filters_and_tolerates_torn_tail():
    path = _write_acks('brjava_pin_acks.jsonl', torn_tail=True)
    allr = cp.read_acks(path)
    if len(allr) != 5:
        _fail('expected 5 whole records, got %d' % len(allr))
    mine = cp.read_acks(path, session=1756499000000, req=7)
    if len(mine) != 4:
        _fail('session/req filter: %d != 4' % len(mine))
    if cp.read_acks(os.path.join(SCRATCH, 'brjava_pin_missing.jsonl')) != []:
        _fail('a missing ack file must read as []')


def test_unacked_lines():
    path = _write_acks('brjava_pin_acks2.jsonl')
    acks = cp.read_acks(path, session=1756499000000, req=7)
    missing = cp.unacked(RESP_LINES, acks)
    if missing != [RESP_LINES[4]]:
        _fail('unacked: %r' % (missing,))
    if cp.unacked(['PASS', '', '#SEQ req=7'], acks) != []:
        _fail('PASS / blank / echo lines are never awaited')


def test_refused_records():
    path = _write_acks('brjava_pin_acks3.jsonl')
    acks = cp.read_acks(path, session=1756499000000, req=7)
    ref = cp.refused(acks)
    idx = sorted(r['i'] for r in ref)
    if idx != [1, 2, 3]:
        _fail('refused indices %r != [1, 2, 3]' % (idx,))
    why = {r['i']: r['why'] for r in ref}
    if 'rendered nothing' not in why[1]:
        _fail('zero-paragraph record not named: %r' % why[1])
    if 'no zone' not in why[2]:
        _fail('directive error not carried: %r' % why[2])
    if 'from-zone' not in why[3] or 'Game.2039' not in why[3]:
        _fail('from-zone mismatch not named: %r' % why[3])


# ---------------------------------------------------------------------------
# BRJAVA-1: the error channel in the feed
# ---------------------------------------------------------------------------

def test_clean_feed_has_no_halt():
    halt, why = cp.halt_verdict(FEED_CLEAN)
    if halt:
        _fail('clean feed halted: %r' % why)
    if cp.client_error_lines(FEED_CLEAN):
        _fail('clean feed flagged: %r' % cp.client_error_lines(FEED_CLEAN))


def test_every_error_shape_halts():
    for name, line in sorted(FEED_ERRORS.items()):
        halt, why = cp.halt_verdict(FEED_CLEAN + [line])
        if not halt:
            _fail('%s: not a halt' % name)
        if line not in why:
            _fail('%s: verdict does not name the line: %r' % (name, why))
        if cp.client_error_lines([line]) != [line]:
            _fail('%s: not recognised' % name)


def test_ai_chat_lines_are_not_errors():
    benign = ['<WOTR_AI_Shadow> AI ERROR is what I would say if I could',
              '$Wey moves AI ERROR from a to b',
              '<game> AI vs AI running (1600ms per move).',
              '<game> AI: no reply from the AI server for 30s - it may '
              'have halted; check the AI panel.',
              '<game> AI: no reply from the AI server for 30s - it may '
              'have halted; check the AI panel. null']
    if cp.client_error_lines(benign):
        _fail('benign lines flagged: %r' % cp.client_error_lines(benign))


# ---------------------------------------------------------------------------
# BRJAVA-4: REQ/RESP echo helpers
# ---------------------------------------------------------------------------

def test_resp_echo_round_trip():
    ln = cp.resp_echo_line(7)
    if cp.parse_resp_echo(ln) != 7:
        _fail('echo %r did not round-trip' % ln)
    if cp.parse_resp_echo('$Wey moves') is not None:
        _fail('a move line read as an echo')
    if cp.parse_seq_header(ln) is None or cp.parse_seq_header(ln)['req'] != 7:
        _fail('the echo is a #SEQ header too')


# ---------------------------------------------------------------------------
# BRJAVA-9: the marker-state column
# ---------------------------------------------------------------------------

def test_marker_state_column():
    path = os.path.join(SCRATCH, 'brjava_pin_dump2.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(DUMP_HDR + '\n' + '\n'.join(DUMP_BODY) + '\n')
    st = cp.read_marker_states(path)
    if st.get('312') != 2:
        _fail('marker 312 state %r != 2' % st.get('312'))
    if '155' in st or '100' in st:
        _fail('non-marker pieces carry no state: %r' % (st,))
    legacy = os.path.join(SCRATCH, 'brjava_pin_dump3.txt')
    with open(legacy, 'w', encoding='utf-8') as f:
        f.write('85\t<~Game.1965~>\t312\t-\n')
    if cp.read_marker_states(legacy) != {}:
        _fail('a legacy 4-column dump must read as {} (no guess)')


TESTS = [
    test_parse_seq_header_dump,
    test_parse_seq_header_req,
    test_split_seq_header_strips_line_two,
    test_dump_at_feed_position,
    test_read_dump_header_from_file,
    test_session_matches,
    test_read_acks_filters_and_tolerates_torn_tail,
    test_unacked_lines,
    test_refused_records,
    test_clean_feed_has_no_halt,
    test_every_error_shape_halts,
    test_ai_chat_lines_are_not_errors,
    test_resp_echo_round_trip,
    test_marker_state_column,
]


def main():
    fail = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception as e:      # noqa: BLE001 - a pin reports, never hides
            fail += 1
            print('FAIL %s: %s: %s' % (t.__name__, type(e).__name__, e))
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fail))
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
