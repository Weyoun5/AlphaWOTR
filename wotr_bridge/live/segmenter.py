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

Render-dup dedup is lifted from parse_game (:1177-1196) for moves:
the chat pane occasionally renders the same gesture on consecutive
lines. Public die kills also recognize an identical adjacent paragraph
when the sidecar explicitly marks it as the same raw command. No
text-only kill dedup is inferred.

Undo/redo outside a proven unpaid human window are surfaced to the
driver. Inside that window, exact saved stage/movement records can be
withdrawn and restored without replaying arbitrary historical text.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import rendered_replay as rr
from . import action_die_zones as die_zones


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


_HUNT_ZONE = die_zones.HUNT_ZONE
_DIE_TOKEN = 'action die'


def _is_die_name(obj):
    return _DIE_TOKEN in str(obj).lower()


def _matches_staged_die(window, rec):
    """An action terminator must name the die which actually opened it.

    Native piece IDs win when both records carry them. Rendered-only feeds
    retain the physical side/face, including explicit changes of that die;
    an Elven Ring's expanded legal action menu does not change its printed face.
    """
    from .sidecar import meta_pids
    def public_die_pids(record):
        return meta_pids(record.get('meta')) or record.get('_native_die_pids')
    selected = getattr(window, '_staged_die_pid', None)
    if selected is not None:
        ending = public_die_pids(rec)
        return ending is not None and selected in {str(pid) for pid in ending}
    staged = next((r for r in window.records if r.get('die_stage')), None)
    if staged is None:
        return False
    normalize = lambda values: {str(int(v)) if str(v).isdigit() else str(v)
                               for v in values}
    original = public_die_pids(staged)
    ending = public_die_pids(rec)
    if original is not None and ending is not None:
        return bool(normalize(original) & normalize(ending))
    side, face = staged['die_stage']
    for changed in window.records:
        if changed.get('op') != 'die_change' or changed.get('side') != side \
                or changed.get('frm') != face:
            continue
        ids = public_die_pids(changed)
        if original is not None and ids is not None \
                and not (normalize(original) & normalize(ids)):
            continue
        face = changed.get('to', face)
    names = rec.get('pieces', []) if rec.get('op') == 'kill' else [rec.get('obj', '')]
    for name in names:
        matched = re.fullmatch(r'(?:\[[1-9]\d*\]\s*)?(Free|Shadow) Action Die \(([ACEHMPW])\)',
                               str(name).strip(), re.IGNORECASE)
        if matched and ('FREE_PEOPLES' if matched.group(1).lower() == 'free'
                        else 'SHADOW') == side and matched.group(2).upper() == face:
            return True
    return False


def fellowship_track_step(rec):
    """Return the explicit progress endpoints of an FSP track drag."""
    if rec.get('op') != 'move' \
            or str(rec.get('obj', '')).strip().lower() != 'fsp':
        return None
    values = []
    for key in ('from', 'to'):
        zone = str(rec.get(key, '')).strip().lower()
        prefix = 'fellowship track '
        if not zone.startswith(prefix) or not zone[len(prefix):].isdigit():
            return None
        values.append(int(zone[len(prefix):]))
    return tuple(values)


class Window(object):
    def __init__(self, die_side, die_face, rec):
        self.die_side = die_side          # 'SHADOW' | 'FREE_PEOPLES'
        self.die_face = die_face          # letter (A C E H M P W)
        self.records = [rec]              # staging record included
        self.closed = None                # close reason once closed
        self.awaiting_fellowship_die = False
        # A direct FP pool -> Hunt Box drag pays for a Fellowship move
        # without visiting Current Action.  Its following track drag is
        # the terminator; waiting for another die would deadlock live play.
        self.fellowship_shortcut = (
            die_side == 'FREE_PEOPLES'
            and str(rec.get('from', '')).strip().lower() == 'fp dice'
            and str(rec.get('to', '')).strip().lower() == _HUNT_ZONE)

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
                    and die_zones.temporary(r.get('zone', '')) \
                    and not r.get('unmatched_action_die'):
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
        # the segmenter is NICK-AWARE for exactly ONE purpose: the AI
        # SERVER'S OWN emission echoes (the client echoes every injected
        # line back on the feed under the server's nick).  Self records
        # never open, close, or join a window; they pass through as
        # 'phase' events (still breaking move-line adjacency for the
        # render-dup criterion, as any intervening line does).
        # Attribution stays ZONE-derived everywhere else - this uses
        # the nick ONLY to recognize the server's own output, which the
        # attribution ruling explicitly permits (self-echo detection,
        # driver.py `_route`).  nick=None (the spectator) keeps the
        # segmenter fully nick-blind.
        self.nick = str(nick) if nick else None
        self.window = None
        self._last_move = None            # adjacent-move key for dedup
        self._last_die_kill = None
        self._last_history_command = None
        self._redo_window = None
        self._redo_records = []
        self._redo_prefix = ()
        self._redo_context = None

    def _clear_open_redo(self):
        self._redo_window = None
        self._redo_records = []
        self._redo_prefix = ()
        self._redo_context = None

    @staticmethod
    def _open_redo_record(rec):
        if rec.get('die_stage'):
            return True
        if rec.get('op') != 'move' or _is_die_name(rec.get('obj', '')):
            return False
        # Ordinary piece placement only: no cards, RNG, ring payment,
        # fellowship track or engine-applied pending decision.
        from .gesture_match import zone_key
        return all(zone_key(rec.get(key))[0] == 'region' for key in ('from', 'to'))

    def _is_self(self, rec):
        return (self.nick is not None
                and str(rec.get('player', '')) == self.nick)

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

    def classify_render_duplicate(self, rec):
        """Advance feed adjacency once, before any live identity consumer."""
        previous_kill = self._last_die_kill
        self._last_die_kill = None
        previous_history = self._last_history_command
        self._last_history_command = None
        if rec.get('op') in ('undo', 'redo'):
            self._last_move = None
            meta = rec.get('meta') or {}
            key = (rec['op'], _norm_text(str(rec.get('inner', ''))),
                   meta.get('gen'), meta.get('raw_seq'))
            self._last_history_command = key
            return (previous_history == key and meta.get('raw_new') is False
                    and meta.get('raw_seq') is not None)
        if rec.get('op') == 'move' and not self._is_self(rec):
            return self._is_render_dup(rec)
        self._last_move = None
        if rec.get('op') == 'kill' and not self._is_self(rec):
            from .native_die_identity import kill_faces
            if kill_faces(rec):
                key = (rec.get('player'), tuple(rec.get('pieces', ())), rec.get('zone'))
                self._last_die_kill = key
                meta = rec.get('meta') or {}
                return previous_kill == key and meta.get('raw_new') is False
        return False

    def push(self, rec, allow_fellowship_prefix=False, render_duplicate=None,
             open_redo_context=None):
        op = rec.get('op')
        out = []
        if (open_redo_context is None or (self._redo_records
                and open_redo_context != self._redo_context)):
            self._clear_open_redo()
        if render_duplicate is None:
            render_duplicate = self.classify_render_duplicate(rec)
        if op in self._NON_BOARD:
            self._last_move = None
            return [('passthrough', rec)]
        if op not in ('undo', 'redo') and self._is_self(rec):
            # the AI server's own emission echo: NEVER window traffic
            # (see __init__).  It breaks move adjacency like any
            # intervening line and flows as an ordinary phase record.
            self._last_move = None
            self._clear_open_redo()
            return [('phase', rec)]
        if render_duplicate:
            # a re-rendered twin: dropped, but SURFACED (the
            # driver books the row) - never a silent []
            return [('dropped_dup', rec)]
        if op in ('undo', 'redo'):
            if op == 'redo' and self._redo_records:
                w = self._redo_window
                saved = self._redo_records[-1]
                prefix_matches = (len(w.records) == len(self._redo_prefix)
                    and all(a is b for a, b in zip(w.records, self._redo_prefix)))
                if (self.window in (None, w) and prefix_matches
                        and _norm_text(str(rec.get('inner', ''))) == saved.get('norm')):
                    # Native Redo restores a board snapshot. Reuse the exact
                    # record already observed, including its PID provenance;
                    # it is not a new raw move or another rules action.
                    reopening = self.window is None
                    w.records.append(self._redo_records.pop())
                    w.closed = None
                    self.window = w
                    self._redo_prefix = tuple(w.records)
                    if saved.get('op') == 'move':
                        self._last_move = tuple(saved.get(k) for k in ('obj', 'from', 'to'))
                    if not self._redo_records:
                        self._clear_open_redo()
                    return [('open' if reopening else 'update', w)]
                self._clear_open_redo()
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
                    if open_redo_context is not None and self._open_redo_record(popped):
                        if self._redo_window is not w:
                            self._clear_open_redo()
                        self._redo_window = w
                        self._redo_records.append(popped)
                        self._redo_prefix = tuple(w.records)
                        self._redo_context = open_redo_context
                    else:
                        self._clear_open_redo()
                    if not w.records:
                        w.closed = 'undone'
                        self.window = None
                        return [('withdrawn', popped),
                                ('discarded', w)]
                    return [('withdrawn', popped)]
            self._clear_open_redo()
            return [(op, rec)]
        # A new native board command abandons the native redo branch.
        # Passthrough chatter and proven duplicate paragraphs returned above.
        self._clear_open_redo()
        if op == 'turn':
            if self.window is not None:
                self.window.closed = 'turn'
                out.append(('closed', self.window))
                self.window = None
            out.append(('phase', rec))
            return out
        if op == 'move' and rec.get('die_stage'):
            if self.window is not None \
                    and self.window.awaiting_fellowship_die \
                    and rec['die_stage'][0] == 'FREE_PEOPLES':
                w = self.window
                w.die_face = rec['die_stage'][1]
                w.awaiting_fellowship_die = False
                w.records.append(rec)
                if str(rec.get('to', '')).strip().lower() == _HUNT_ZONE:
                    w.closed = 'die_to_hunt'
                    self.window = None
                    return [('closed', w)]
                return [('update', w)]
            if self.window is not None:
                self.window.closed = 'restaged'
                out.append(('closed', self.window))
            self.window = Window(rec['die_stage'][0],
                                 rec['die_stage'][1], rec)
            out.append(('open', self.window))
            return out
        if self.window is None:
            if allow_fellowship_prefix and fellowship_track_step(rec):
                self.window = Window('FREE_PEOPLES', None, rec)
                self.window.awaiting_fellowship_die = True
                return [('open', self.window)]
            return [('phase', rec)]
        # inside an open window ---------------------------------------
        w = self.window
        if w.fellowship_shortcut and fellowship_track_step(rec):
            w.records.append(rec)
            w.closed = 'die_to_hunt'
            self.window = None
            return [('closed', w)]
        if op == 'move' and _is_die_name(rec.get('obj', '')):
            frm = die_zones.zone(rec.get('from', ''))
            to = die_zones.zone(rec.get('to', ''))
            matches = _matches_staged_die(w, rec)
            if matches and die_zones.temporary(frm) and to == die_zones.pool(w.die_side):
                # self-undo: die back to pool before completion (L1)
                w.records.append(rec)
                w.closed = 'self_undo'
                self.window = None
                return [('discarded', w)]
            if (matches and die_zones.temporary(frm)
                    and to == die_zones.casualties(w.die_side)):
                # Dragging this same die to its casualties spends it exactly
                # as the native Kill menu does, from any temporary seat.
                w.records.append(rec)
                w.closed = 'die_killed'
                self.window = None
                return [('closed', w)]
            if to == _HUNT_ZONE and matches and w.die_side == 'FREE_PEOPLES':
                w.records.append(rec)
                w.closed = 'die_to_hunt'
                self.window = None
                return [('closed', w)]
        if op == 'kill' and die_zones.temporary(rec.get('zone', '')) \
                and any(_is_die_name(p)
                                       for p in rec.get('pieces', [])):
            if _matches_staged_die(w, rec):
                w.records.append(rec)
                w.closed = 'die_killed'
                self.window = None
                return [('closed', w)]
            rec['unmatched_action_die'] = True
        w.records.append(rec)
        return [('update', w)]
