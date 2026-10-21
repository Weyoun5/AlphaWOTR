# -*- coding: ascii -*-
"""client_protocol.py - the Java client mod's sequencing / ack / error
channel, as PURE functions (BRJAVA fix wave 2026-08-29).

The candidate jar (WOTR-Gandalf-AI.jar.CANDIDATE_seqack_20260829, sources
wotr_bridge/client_build/hook_out/{AiBoot,AiOpp}.java) adds four things
to the file protocol; this module is their one reader.  Nothing here
touches the engine, the driver, the transport or any file except the two
it is asked to read (a dump header, an ack file).

  1. #SEQ HEADER (BRJAVA-2/4/7).  Line 1 of ai_client_board.txt and line
     2 of ai_bridge_request.log:
        #SEQ [req=<n>] dump=<n> feed_paragraphs=<n> raw_lines=<n>
             ts=<ms> session=<nonce> [dropped_resp=<n>]
     All four counts are read on the EDT in the SAME runnable that writes
     the dump, so they are exact for that dump:
       dump            monotonic per JVM (every dumpBoard call)
       feed_paragraphs the chat document's paragraph count as AiFeed's
                       drain counts it (the trailing empty paragraph is
                       not a feed line) - the feed file holds exactly
                       FEED_HEADER_LINES (the '#BLIND' attestation) +
                       feed_paragraphs lines at that instant
       raw_lines       the raw running log's line count (== the request
                       body's line count when the header is on a REQ)
       ts              System.currentTimeMillis()
       session         the client's live-loop boot nonce (ms); 0 when no
                       live/auto loop has started
       req             the request's sequence number (REQ header only)
       dropped_resp    a RESP the client DROPPED since its last REQ
                       because its echo named another req (REQ only)
  2. RESP ECHO (BRJAVA-4).  The server may write `#SEQ req=<n>` as RESP
     line 1 (resp_echo_line); the client injects a RESP without an echo
     as before, and DROPS one whose echo is not the outstanding request.
  3. ACK RECORDS (BRJAVA-3).  ai_inject_ack.jsonl, append-only, one JSON
     object per injected RESP line, written on the EDT right after the
     line was applied:
        {"session","req","i","kind":"line|call|replay","raw",
         "paragraphs_before","paragraphs_after","ok",
         ["error"],["claimed_from","real_from"]}
     paragraphs_* are feed paragraph counts around the injection; equal
     counts = the client rendered NOTHING for that line (an unknown verb,
     a rejected line).  claimed_from/real_from appear only when a piece
     named by a move line was not in the from-zone the line claimed
     (real_from = 'pid=<~zone~>;pid=<~zone~>').
  4. LOUD LINES (BRJAVA-1/6/7).  Every client-side complaint now lands in
     the chat document (hence the feed) as a '<game> AI ...' line - the
     shapes in CLIENT_ERROR_RE.  halt_verdict() names the first one.

Consumption contract (the design section of BRJAVA_report.md): a
comparator may run only on a dump whose header is AT the consumed feed
position (dump_at_feed_position); a refused emission is a loud halt
naming the line (refused / unacked), never a "restore not landed" guess;
a header from another session is refused (session_matches).
"""
import json
import os
import re

SEQ_PREFIX = '#SEQ '

# the feed file's own header lines that are not chat paragraphs: line 1
# is AiFeed's '#BLIND aiside=.. hands=..' attestation (written at
# premain, seq 1); every later feed line is one chat paragraph
FEED_HEADER_LINES = 1

_INT_KEYS = ('req', 'dump', 'feed_paragraphs', 'raw_lines', 'ts',
             'session', 'dropped_resp')
_KV_RE = re.compile(r'^([a-z_]+)=(-?\d+)$')

# the client mod's complaint shapes (AiOpp.loud / AiOpp.err).  Anchored
# at the line start on the '<game> AI' system tag so a human chat line
# or a rendered move that merely CONTAINS the words never matches.
CLIENT_ERROR_RE = re.compile(
    r'^<game> AI(?:'
    r' ERROR[: ]'                       # every execDirective / RNG failure
    r'|: injection failed'              # game.z.a threw
    r'|: replay-inject failed'          # !REPLAY path threw
    r'|: injection rendered nothing'    # zero-paragraph injection
    r'| loop error'                     # the live/auto loop threw
    r'|: game window replaced'          # File > New under a live loop
    r'|: modal dialog open'             # the live-mode dialog watchdog
    r'|: EDT wedged'                    # the invokeAndWait watchdog
    r'|: stale reply dropped'           # RESP echo named another req
    r'|: panel launch failed'
    r'|: AI server not running'
    r'|: yes-dialog arm failed'
    r')')


# ---------------------------------------------------------------------
# 1. the #SEQ header
# ---------------------------------------------------------------------
def parse_seq_header(line):
    """'#SEQ k=v k=v ...' -> {k: int} or None (not a header / malformed).
    Every value is an integer; an unknown key is kept (as int) so a
    later jar can add fields without breaking this reader; a value that
    is not an integer makes the whole line NOT a header (no guessing)."""
    if not isinstance(line, str):
        return None
    s = line.rstrip('\r\n')
    if not s.startswith(SEQ_PREFIX):
        return None
    out = {}
    for tok in s[len(SEQ_PREFIX):].split():
        m = _KV_RE.match(tok)
        if not m:
            return None
        out[m.group(1)] = int(m.group(2))
    if 'dump' not in out and 'req' not in out:
        return None
    return out


def split_seq_header(lines):
    """(header or None, the lines without it).  For a REQ body
    (transport.parse_request's lines-after-the-side-header): the client
    writes the #SEQ line as REQ line 2, i.e. the FIRST log line."""
    lines = list(lines or ())
    if lines:
        h = parse_seq_header(lines[0])
        if h is not None:
            return h, lines[1:]
    return None, lines


def read_dump_header(path):
    """The #SEQ header of a board dump file, or None (legacy dump with no
    header / unreadable).  Reads the first line only."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            first = f.readline()
    except (OSError, IOError):
        return None
    return parse_seq_header(first)


def feed_position_delta(header, consumed_feed_lines,
                        feed_header_lines=FEED_HEADER_LINES):
    """(feed lines the dump is AT) - (feed lines the consumer has
    consumed).  0 = the dump is exactly at the consumed position;
    negative = the dump is BEHIND the feed (a gesture rendered after the
    dump - the normal-cycle stale read); positive = the dump is AHEAD
    (a slow cycle: the client dumped again before the consumer caught
    up).  None when the header carries no feed position."""
    if not header or 'feed_paragraphs' not in header:
        return None
    return (int(header['feed_paragraphs']) + int(feed_header_lines)
            - int(consumed_feed_lines))


def dump_at_feed_position(header, consumed_feed_lines,
                          feed_header_lines=FEED_HEADER_LINES):
    """True only when the dump was taken AT the feed position the
    consumer has consumed (BRJAVA-2).  A header-less dump is never at
    position: no witness row, no reconcile, no restore-watch verdict on
    it."""
    d = feed_position_delta(header, consumed_feed_lines, feed_header_lines)
    return d == 0


def session_matches(header, session):
    """True when the header was written by the live-loop session
    `session` (the nonce the server adopted at its first request).  An
    unknown session on either side never matches (BRJAVA-7)."""
    if not header or session is None:
        return False
    try:
        return int(header.get('session', -1)) == int(session) \
            and int(session) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------
# 2. the RESP echo
# ---------------------------------------------------------------------
def resp_echo_line(req):
    """RESP line 1 the server writes so the client can drop a stale
    reply: '#SEQ req=<n>'."""
    return '%sreq=%d' % (SEQ_PREFIX, int(req))


def parse_resp_echo(line):
    """The req a RESP echo line names, or None."""
    h = parse_seq_header(line)
    if not h or 'req' not in h:
        return None
    return h['req']


# ---------------------------------------------------------------------
# 3. ack records
# ---------------------------------------------------------------------
def read_acks(path, session=None, req=None):
    """Every whole JSON record of the ack file, in order, optionally
    filtered to one session / one request.  A torn last line (the
    client mid-append) is skipped, never raised: the next read sees it
    whole.  A missing file is []."""
    out = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except (OSError, IOError):
        return out
    for ln in raw.split('\n'):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue                       # torn tail / partial append
        if not isinstance(rec, dict):
            continue
        if session is not None and rec.get('session') != session:
            continue
        if req is not None and rec.get('req') != req:
            continue
        out.append(rec)
    return out


def _awaited(lines):
    """The RESP lines the client actually injects (its own skip rules:
    blank and PASS are skipped; the #SEQ echo is stripped)."""
    out = []
    for ln in lines or ():
        if not isinstance(ln, str):
            continue
        s = ln.strip()
        if not s or s == 'PASS' or parse_seq_header(s) is not None:
            continue
        out.append(s)
    return out


def unacked(lines, acks):
    """RESP lines (as written by the server) that have NO ack record in
    `acks` - the client never reached them (a loop error mid-response,
    a modal wedge, a dropped reply).  Matching is by raw text in order,
    so a repeated identical line needs one ack per occurrence."""
    remaining = [str(a.get('raw', '')).strip() for a in acks or ()]
    out = []
    for s in _awaited(lines):
        if s in remaining:
            remaining.remove(s)
        else:
            out.append(s)
    return out


def refused(acks):
    """The ack records that say the client did NOT do what the line
    asked: a directive / injection error, a line that rendered no
    paragraph, or a move whose pieces were not in the claimed from-zone.
    Each returned record carries a 'why' string naming the cause and the
    raw line - the loud halt text the driver owner raises."""
    out = []
    for rec in acks or ():
        raw = str(rec.get('raw', ''))
        why = None
        if rec.get('ok') is False or rec.get('error'):
            why = 'client refused %r: %s' % (
                raw, rec.get('error') or 'ok=false')
        elif 'claimed_from' in rec or 'real_from' in rec:
            why = ('from-zone mismatch on %r: the line claimed %s but the '
                   'client moved from %s' % (raw, rec.get('claimed_from'),
                                             rec.get('real_from')))
        elif (rec.get('paragraphs_before') is not None
              and rec.get('paragraphs_after') is not None
              and int(rec['paragraphs_after'])
              <= int(rec['paragraphs_before'])):
            why = ('client rendered nothing for %r (paragraphs %s -> %s): '
                   'unknown verb or rejected line' % (
                       raw, rec['paragraphs_before'],
                       rec['paragraphs_after']))
        if why is not None:
            r2 = dict(rec)
            r2['why'] = why
            out.append(r2)
    return out


# ---------------------------------------------------------------------
# 4. the loud lines
# ---------------------------------------------------------------------
_HUMAN_HAND_DIALOG_RE = re.compile(
    r"^<game> AI: modal dialog open: (?:title='Confirm View Hand' text='.*'"
    r"|title='Input' text='Enter password for encryption(?: Enter password for encryption)?') "
    r"\(injecting=false\)(?: null)?$")


_HUMAN_UNCHANGED_DESTINATION_RE = re.compile(
    r"^<game> AI: modal dialog open: title='Message' text='"
    r"You have selected the same place as the destination"
    r"(?: You have selected the same place as the destination)?' "
    r"\(injecting=false\)(?: null)?$")


_HUMAN_SAVE_DIALOG_RE = re.compile(
    r"^<game> AI: modal dialog open: title='Save' "
    r"text='Save In: File Name: Files of Type:' "
    r"\(injecting=false\)(?: null)?$")


def client_error_lines(lines):
    """Client failures, excluding known human dialogs with no board effect.

    The watchdog also reports dialogs that it deliberately leaves to the
    human. Releasing a piece on its original region also changes nothing.
    Opening this native UI is not a failed game action. AI injection
    dialogs, unknown dialogs and actual error lines still halt. A Java
    "no reply" notice reports elapsed waiting, not a failed board action:
    the request may still finish, and the client accepts a matching late
    reply. Turning that notice into DriverHalt would kill a healthy reply.
    """
    return [ln for ln in lines or ()
            if isinstance(ln, str) and CLIENT_ERROR_RE.match(ln.strip())
            and not _HUMAN_HAND_DIALOG_RE.fullmatch(ln.strip())
            and not _HUMAN_UNCHANGED_DESTINATION_RE.fullmatch(ln.strip())
            and not _HUMAN_SAVE_DIALOG_RE.fullmatch(ln.strip())]


def halt_verdict(lines):
    """(halt, reason).  halt is True when any line is a client complaint;
    the reason names the FIRST such line verbatim (the driver owner
    raises DriverHalt(reason))."""
    bad = client_error_lines(lines)
    if not bad:
        return False, ''
    return True, ('client reported a failure: %s' % bad[0]
                  + ('' if len(bad) == 1
                     else '  (+%d more)' % (len(bad) - 1)))


# ---------------------------------------------------------------------
# BRJAVA-9: the marker-state column
# ---------------------------------------------------------------------
def read_marker_states(path):
    """{pid: int state} from the dump's 5th column - the H.k() value of
    every piece that implements the client's marker interface H (the
    political / fellowship / corruption / hunt markers), '-' for every
    other piece.  A legacy 4-column dump gives {} (no guess from the
    reused M/U column)."""
    out = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) < 5:
                    continue
                pids = parts[2].split()
                states = parts[4].split()
                for i, pid in enumerate(pids):
                    if i >= len(states) or states[i] == '-':
                        continue
                    try:
                        out[pid] = int(states[i])
                    except ValueError:
                        continue
    except (OSError, IOError):
        return {}
    return out


def read_action_die_faces(path):
    """Public native die faces in column six; legacy exports have no evidence."""
    out = {}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) < 6:
                continue
            pids, faces = parts[2].split(), parts[5].split()
            if len(pids) != len(faces) or any(face not in ('-', 'C', 'P', 'H', 'M', 'W', 'A', 'E') for face in faces):
                raise ValueError('Malformed public action-die face column')
            out.update((pid, face) for pid, face in zip(pids, faces) if face != '-')
    return out
