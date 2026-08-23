"""Gesture-window segmentation over the live record stream.

The batching semantics are LIFTED from the corpus-proven
rendered_replay.parse_game (src/rendered_replay.py:1149-1286): a window
opens on an action-die staging (parse_line stamps 'die_stage' on the
record) and collects the records of that one action.  parse_game closes
a batch only when the NEXT staging arrives — a batch parser can wait;
a live driver cannot.  Live closure uses the client's OWN action
terminators (design rule 4: the window closes on the client's
statement, never on a timeout):

  close reasons
    'die_killed'   kill of the staged die in Current Action (with or
                   without a card in the group) — the standard end of a
                   die action (live specimens: LIVE_0818 B2/E1)
    'die_to_hunt'  the staged FREE die moves to The Hunt For The Ring —
                   the fellowship-move completion (rulebook p.37; the
                   die enters the box AFTER the move; client fact 20)
    'restaged'     a new die staged before the open window closed —
                   tolerated, closes the old window first
    'turn'         a Turn marker arrived
  discard reason
    'self_undo'    the staged die returned to its pool before the
                   action completed — LEGAL and SILENT (USER item L1);
                   the window's records are withdrawn, nothing happened

Render-dup dedup is lifted verbatim in criterion from parse_game
(:1177-1196): the chat pane occasionally renders the SAME move gesture
twice on consecutive lines; dedup applies to MOVE records only, when
the source is a board region and the move names >1 piece or a
named/nazgul/leader figure.

Undo/redo records are NOT resolved here (parse_game collapses them
before segmentation — a batch construct).  They are surfaced as
('undo'|'redo', rec) events; the driver owns the undo doctrine.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import rendered_replay as rr


def _norm_text(line):
    """Prefix-normalized board text of a line (the same 'norm' the
    parser stamps on records) — lazy import avoids a package cycle.

    RELATIVE, not absolute (USER 2026-08-20 live: "no module named
    wotr_bridge").  Production launches the server as
    `python -m live.live_server` with cwd=wotr_bridge, so there is no
    `wotr_bridge` package on sys.path at all - this line raised
    ModuleNotFoundError the first time a record needed normalising and
    took the whole serve cycle down.  The offline battery never saw it
    because the suites run from the PROJECT ROOT, where `wotr_bridge`
    IS importable: an environment-dependent import is invisible to a
    test that only ever runs in the friendly environment."""
    from . import live_parser
    return live_parser.normalize(line)[1]


_POOL_ZONES = ('FP Dice', 'SA Dice')
_HUNT_ZONE = 'the hunt for the ring'
_CUR_ACTION = 'current action'
_DIE_TOKEN = 'action die'


def _is_die_name(obj):
    return _DIE_TOKEN in str(obj).lower()


class Window(object):
    def __init__(self, die_side, die_face, rec):
        self.die_side = die_side          # 'SHADOW' | 'FREE_PEOPLES'
        self.die_face = die_face          # letter (A C E H M P W)
        self.records = [rec]              # staging record included
        self.closed = None                # close reason once closed

    def body_records(self):
        """The window's records without the staging/terminator die
        bookkeeping — the gesture content a matcher consumes."""
        out = []
        for r in self.records:
            if r.get('die_stage'):
                continue
            if r['op'] == 'move' and _is_die_name(r.get('obj', '')):
                continue
            if r['op'] == 'kill' and all(
                    _is_die_name(p) or '/' in str(p) or
                    str(p).lower().endswith('card')
                    for p in r.get('pieces', [])) \
                    and str(r.get('zone', '')).lower() == _CUR_ACTION:
                continue
            out.append(r)
        return out


class Segmenter(object):
    """push(rec) -> list of (event, payload) tuples, in order.

    Takes EVERY parsed record (chat/system/noop included — they pass
    through as 'passthrough' and break move-line adjacency for the
    dedup, exactly as intervening lines do in parse_game's
    line-number-adjacency criterion).

    events: 'phase'  (board record outside any window: draws, rolls,
                      eyes, die pool moves at turn start...)
            'open' / 'update'      (window,)
            'closed'               (window,)  window.closed = reason
            'discarded'            (window,)  self-undo
            'undo' / 'redo'        (rec,)
            'passthrough'          (rec,)  chat/system/noop/blank
            'dropped_dup'          (rec,)  render-dup twin dropped
    """

    _NON_BOARD = frozenset(('chat', 'system', 'noop', 'blank'))

    def __init__(self, nick=None):
        # WAVE-I (WAVE-H blocker 5): the segmenter is NICK-AWARE for
        # exactly ONE purpose — the AI SERVER'S OWN emission echoes.
        # The client echoes every injected line back on the feed under
        # the server's nick; a nick-blind segmenter (a) opened a
        # window on the AI's die-spend echo and swallowed the human's
        # next records into it, (b) let the AI's owed die-kill echo
        # FALSELY TERMINATE an open HUMAN window, and (c) polluted an
        # open human window's body with AI emission records (chain
        # battles: the AI's casualty lines land while the human's die
        # window is still open).  The WAVE-H driver bypass only fixed
        # (a).  Self records never open, close, or join a window; they
        # pass through as 'phase' events (they still break move-line
        # adjacency for the render-dup criterion, as any intervening
        # line does).  Attribution stays ZONE-derived everywhere else
        # — this uses the nick ONLY to recognize the server's own
        # output, which the attribution ruling explicitly permits
        # (self-echo detection, driver.py `_route`).  nick=None (the
        # spectator) keeps the segmenter fully nick-blind.
        self.nick = str(nick) if nick else None
        self.window = None
        self._last_move = None            # adjacent-move key for dedup

    def _is_self(self, rec):
        return (self.nick is not None
                and str(rec.get('player', '')).startswith(self.nick))

    # -- render-dup dedup (parse_game:1177-1196 criterion: identical
    # move on the IMMEDIATELY ADJACENT line, >1 piece or named figure,
    # source a board region).  _last_move is reset by EVERY other
    # record, so only true line-adjacent twins qualify. ----------------
    def _is_render_dup(self, rec):
        key = (rec.get('obj'), rec.get('from'), rec.get('to'))
        dup = (self._last_move == key)
        self._last_move = key
        if not dup:
            return False
        # ONE answerer where the sidecar speaks: the client's own
        # raw_new flag says whether this paragraph is a re-render of
        # the previous raw line (False) or a new statement (True - a
        # real second identical drag).  The text heuristic below
        # serves only channel-absent feeds (corpus, old jars).
        meta = rec.get('meta')
        if isinstance(meta, dict) and isinstance(meta.get('raw_new'),
                                                 bool):
            return not meta['raw_new']
        obj = str(rec.get('obj', '') or '')
        pp = rr.parse_piece(obj)
        multi = (obj.startswith('group ')
                 or (pp is not None
                     and pp[0] in ('named', 'nazgul', 'leader')))
        try:
            import board
            in_region = rr.sanitize(
                str(rec.get('from', '') or '')) in board.regions
        except Exception:
            in_region = False
        return bool(multi and in_region)

    def push(self, rec):
        op = rec.get('op')
        out = []
        if op in self._NON_BOARD:
            self._last_move = None
            return [('passthrough', rec)]
        if op not in ('undo', 'redo') and self._is_self(rec):
            # the AI server's own emission echo: NEVER window traffic
            # (see __init__).  It breaks move adjacency like any
            # intervening line and flows as an ordinary phase record.
            self._last_move = None
            return [('phase', rec)]
        if op == 'move':
            if self._is_render_dup(rec):
                # a re-rendered twin: dropped, but SURFACED (the
                # driver books the row) - never a silent []
                return [('dropped_dup', rec)]
        else:
            self._last_move = None
        if op in ('undo', 'redo'):
            if op == 'undo' and self.window is not None:
                # client undo is LIFO per LINE (fact 1): an undo whose
                # inner names the OPEN window's newest record withdraws
                # that record — window-internal bookkeeping that never
                # reached the driver's commit ledger.  Undoing the
                # staging itself un-opens the window (real specimen:
                # game 2023-01-02 L50-51, staged A die undone, the
                # 1-record ghost window matched 'none').
                inner = str(rec.get('inner', '')).strip()
                norm = _norm_text(inner)
                w = self.window
                if w.records and str(
                        w.records[-1].get('norm', '')) == norm:
                    popped = w.records.pop()
                    if not w.records:
                        w.closed = 'undone'
                        self.window = None
                        return [('withdrawn', popped),
                                ('discarded', w)]
                    return [('withdrawn', popped)]
            return [(op, rec)]
        if op == 'turn':
            if self.window is not None:
                self.window.closed = 'turn'
                out.append(('closed', self.window))
                self.window = None
            out.append(('phase', rec))
            return out
        if op == 'move' and rec.get('die_stage'):
            if self.window is not None:
                self.window.closed = 'restaged'
                out.append(('closed', self.window))
            self.window = Window(rec['die_stage'][0],
                                 rec['die_stage'][1], rec)
            out.append(('open', self.window))
            return out
        if self.window is None:
            return [('phase', rec)]
        # inside an open window ---------------------------------------
        w = self.window
        if op == 'move' and _is_die_name(rec.get('obj', '')):
            frm = str(rec.get('from', '')).strip().lower()
            to = str(rec.get('to', '')).strip()
            if frm == _CUR_ACTION and to in _POOL_ZONES:
                # self-undo: die back to pool before completion (L1)
                w.records.append(rec)
                w.closed = 'self_undo'
                self.window = None
                return [('discarded', w)]
            if to.strip().lower() == _HUNT_ZONE:
                w.records.append(rec)
                w.closed = 'die_to_hunt'
                self.window = None
                return [('closed', w)]
        if op == 'kill' and str(rec.get('zone', '')).lower() \
                == _CUR_ACTION and any(_is_die_name(p)
                                       for p in rec.get('pieces', [])):
            w.records.append(rec)
            w.closed = 'die_killed'
            self.window = None
            return [('closed', w)]
        w.records.append(rec)
        return [('update', w)]
