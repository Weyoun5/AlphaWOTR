# -*- coding: ascii -*-
"""client_protocol.py - the Java client mod's sequencing / ack / error
channel, as PURE functions (BRJAVA fix wave 2026-08-29).

The candidate jar (WOTR-Gandalf-AI.jar.CANDIDATE_seqack_20260829, sources
wotr_bridge/client_build/hook_out/{AiBoot,AiOpp}.java) added four things
to the file protocol; this module is their one reader.  That jar was
archived to WOTR_AI_training_lab/retired_2026-10-24 (R6-DEAD-CODE-4, one
MANIFEST.tsv row); the hook sources still write all four, and the head
runs wotr_bridge/play_client/WOTR-Gandalf-AI.jar.  Nothing here
touches the engine, the driver, the transport or any file (every file of
the exchange is opened through transport.py).

  1. #SEQ HEADER (BRJAVA-2/4/7).  Line 1 of ai_client_board.txt and line
     2 of ai_bridge_request.log:
        #SEQ [req=<n>] dump=<n> feed_paragraphs=<n> raw_lines=<n>
             ts=<ms> session=<nonce> [dropped_resp=<n>, pre-X1-08 jars]
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
       dropped_resp    RETIRED (X1-08): jars before that fix appended it
                       after dropping a stale RESP; nothing here reads it
                       (parse_seq_header keeps it as a plain int).  A drop
                       reaches the server as the loud '<game> AI: stale
                       reply dropped' line (CLIENT_ERROR_RE, a halt).
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
     (real_from = 'pid=<~zone~>;pid=<~zone~>'): the person had moved it
     since the board the line was made for.  link.Receipts.read notes it.
     A record states that a line or call was CARRIED OUT
     and which paragraphs it wrote - never that it had its effect (a
     dialog of the client may have swallowed it): whether the game's step
     is on the board stays the comparison's question.
  4. LOUD LINES (BRJAVA-1/6/7).  Every client-side complaint now lands in
     the chat document (hence the feed) as a '<game> AI ...' line - the
     shapes in CLIENT_ERROR_RE (client_error_lines).

Consumption contract (the design section of BRJAVA_report.md): a
comparator may run only on a dump whose header is AT the consumed feed
position (link.ClientLink.read_board); a line the client held or refused
is noted with its reason (link.Receipts.read), never guessed at;
a header from another session is refused (link.ClientLink.read).
"""
import re

SEQ_PREFIX = '#SEQ '

# the feed file's own header lines that are not chat paragraphs: line 1
# is AiFeed's '#BLIND aiside=.. hands=..' attestation (written at
# premain, seq 1); every later feed line is one chat paragraph
FEED_HEADER_LINES = 1

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


# ---------------------------------------------------------------------
# 2. the RESP echo
# ---------------------------------------------------------------------
def resp_echo_line(req):
    """RESP line 1 the server writes so the client can drop a stale
    reply: '#SEQ req=<n>'."""
    return '%sreq=%d' % (SEQ_PREFIX, int(req))


# ---------------------------------------------------------------------
# 4. the loud lines
# ---------------------------------------------------------------------
_HUMAN_DIALOG_RE = re.compile(
    r"^<game> AI: modal dialog open: title='.*' text='.*' "
    r"\(injecting=false\)(?: null)?$")


def client_error_lines(lines):
    """Client failures, excluding UI dialogs opened outside AI injection.

    Human dialogs are left open for the player and are not evidence that
    an AI emission failed. Their board effects still pass through normal
    reconciliation; failed AI acknowledgements remain errors. This also
    covers ordinary notices such as flipping an empty selection, without
    maintaining a list of every dialog's wording. AI injection dialogs,
    malformed dialog telemetry and actual error lines still halt. A Java
    "no reply" notice reports elapsed waiting, not a failed board action:
    the request may still finish, and the client accepts a matching late
    reply. Turning that notice into DriverHalt would kill a healthy reply.
    """
    return [ln for ln in lines or ()
            if isinstance(ln, str) and CLIENT_ERROR_RE.match(ln.strip())
            and not _HUMAN_DIALOG_RE.fullmatch(ln.strip())]

