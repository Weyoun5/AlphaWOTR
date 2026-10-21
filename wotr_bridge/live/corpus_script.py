# -*- coding: ascii -*-
"""Corpus replay: the AI seat pinned to a RECORDED player's lines.

USER 2026-08-28 (lab/corpus_replay/DESIGN.md): replay a recorded game
through the real client - the human seat's raw lines injected one by
one, the AI seat playing the recorded opponent's decisions with the
full live driver/emitter so the human lane runs exactly as in a real
game.

At every AI decision the server hands the engine menu to
ScriptChooser.rank: each candidate is applied to a CLONE of the engine
state under the driver's probe pin (no live randomness consumed) and
rendered through the server's own emitter into the client's RAW line
grammar; the candidate whose lines mean what the recorded player's
next raw lines mean wins.

"Mean": the recorded human DRAGGED (Scribe.141 moves) where the emitter
presses the client's buttons (Controls.276 die spend, HighlightLabel.101
political step, ...) - different raw grammars the client renders to the
same text.  So both sides are reduced to per-piece ATOMS
    (verb, piece, from-zone, to-zone, extras)
with verb = move|kill|..., piece = the literal pid for unique pieces
(dice, cards, characters, markers) and the nation/kind label for
fungible army units (the emitter picks its own regulars), and the
multisets are compared - grouping- and grammar-insensitive, exact in
substance.  No engine action is ever substituted or invented: the pick
is one of the menu's own actions.  No candidate matching is a
DriverHalt (the replay runner reports DIVERGED) - never a silent
fallback to the net.
"""
import os
import re

try:
    from .driver import DriverHalt
except ImportError:                      # flat-path imports (cwd=live)
    from driver import DriverHalt

_HASH_TAIL = re.compile(r'\s+(?:[0-9A-Fa-f]{6}|null)\s*$')

# raw ops the CLIENT produces as a consequence of an action or that are
# bookkeeping/protocol - never the recorded player's decision
CONSEQUENCE_OPS = frozenset((
    'deck_commit', 'send_tokens', 'card_draw', 'dice_roll', 'dice_pickup',
    'eye_alloc', 'combat_roll', 'hunt_tile', 'names', 'deselects',
    'dice_book_726', 'dice_book_731', 'dice_book_757', 'dice_book_762',
    'turn_marker', 'directive', 'chat'))
MOVE_OPS = frozenset((
    'move', 'reveal_move', 'die_spend_sh', 'die_spend_fp', 'die_spend_fp81',
    'die_to_hunt', 'track_marker', 'card_move', 'political_91',
    'political_96', 'political_101', 'move_662', 'move_637', 'marker_366',
    'token_271', 'buffet_move14', 'die_change', 'swap_place',
    'die_refund_736', 'die_refund_767', 'die_refund_784'))
KILL_OPS = frozenset(('kill', 'hand_exit', 'hand_exit_b',
                      'reveal_hand_exit', 'swap_remove'))
PID_FIELDS = ('pieces', 'piece', 'die', 'dice', 'card', 'cards')
HUNT_ZONE = 'Game.1997'
SA_DICE_ZONE = 'Game.2043'


def _emitter():
    import emitter
    return emitter


_LABELS = None


def piece_label(pid, faces=None):
    """Fungible army units -> 'unit:<nation>:<kind>'; action dice ->
    'die:<face code>' (same-face dice are interchangeable - the emitter
    spends pid 105 where the recorded player dragged 103, both 'A');
    everything else (cards, characters, markers) -> the literal pid."""
    global _LABELS
    if _LABELS is None:
        import piece_table_authoritative as pt
        _LABELS = {}
        for k, v in pt.AUTH_ARMY_PIECES.items():
            _LABELS[str(k)] = 'unit:%s' % (
                ':'.join(str(x) for x in v) if isinstance(v, (tuple, list))
                else v)
        for k, v in pt.AUTH_PIECE_LABELS.items():
            if str(v).endswith('action die'):
                _LABELS.setdefault(str(k), 'die:?')
    p = str(pid)
    if faces and p in faces:
        return 'die:%s' % faces[p]
    return _LABELS.get(p, p)


def verb_of(op):
    if op in MOVE_OPS:
        return 'move'
    if op in KILL_OPS:
        return 'kill'
    return str(op)


def atoms_of(line, faces=None):
    """Per-piece atoms of one raw client line; None for a line that is
    not a game line (chat, directive) or not classifiable."""
    em = _emitter()
    op, f = em.classify_line(line)
    if op is None or op in ('chat', 'directive'):
        return None
    pids = []
    extra = []
    for k, v in f.items():
        if k in ('nick', '_trail', '_sp'):
            continue
        if k in PID_FIELDS:
            pids.extend(v if isinstance(v, list) else [v])
        elif k not in ('frm', 'to'):
            extra.append((k, ' '.join(v) if isinstance(v, list) else str(v)))
    frm = str(f.get('frm', '') or '')
    to = str(f.get('to', '') or '')
    verb = verb_of(op)
    if verb == 'kill':
        frm = ''                          # the kill zone token is a constant
    ex = tuple(sorted(extra))
    if not pids:
        return [(verb, '', frm, to, ex)]
    return [(verb, piece_label(p, faces), frm, to, ex) for p in pids]


def _is_die_kill(atoms):
    return bool(atoms) and all(a[0] == 'kill' and a[1].startswith('die:')
                               for a in atoms)


def army_box_rewrites(raw_lines):
    """The corpus ARMY-BOX convention: oversized stacks parked off-map
    in the Mordor-track boxes (Game.524..533) with a counter on the
    true region.  The engine/emitter have no such notion, so a recorded
    box line is read as the move the parser infers for it -
    log_replay.LogReplayer._preprocess_army_boxes run VERBATIM on a stub
    (the same rule the corpus was extracted with; ruling 1.156):
      {raw line number: (kind, region_code, region_name)}
    kind 'entry' = the line moved the army INTO region; 'exit' = the
    army left FROM region; 'noop' = state noise (no move at all)."""
    import types
    import board as board_mod
    import log_replay as lr
    LR = lr.LogReplayer
    stub = types.SimpleNamespace(
        board=board_mod.load_board(), piece_types=None, _box_rewrites={},
        _SCRIBE_MOVE_RE=LR._SCRIBE_MOVE_RE, _ARMY_BOX_CODES=LR._ARMY_BOX_CODES,
        _TRACK_MARKER_PIECES=LR._TRACK_MARKER_PIECES)
    LR._preprocess_army_boxes(stub, list(raw_lines))
    by_san = {}
    for code, name in lr.LOCATION_CODES.items():
        by_san.setdefault(board_mod.sanitize(name), (code, name))
    out = {}
    for n, (kind, reg) in stub._box_rewrites.items():
        if kind == 'noop' or reg is None:
            out[n] = ('noop', None, None)
        elif reg in by_san:
            out[n] = (kind, by_san[reg][0], by_san[reg][1])
    return out


class ScriptChooser(object):
    def __init__(self, script_path, ai_nick):
        raw_path = script_path
        if not script_path.lower().endswith('.log'):
            stem = os.path.basename(script_path).split('.rendered')[0]
            raw_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(script_path))), 'game_logs', stem + '.log')
        with open(raw_path, 'rb') as f:
            txt = f.read().decode('utf-8', 'replace')
        lines = [l.rstrip('\r') for l in txt.split('\n')]
        self.ai_nick = str(ai_nick)
        pre = '$' + self.ai_nick + '<~'
        em = _emitter()
        self.box_rewrites = army_box_rewrites(lines)
        if self.box_rewrites:
            print('  [script] army-box lines resolved (ruling 1.156): %s'
                  % ', '.join('raw %d %s %s' % (n, k, r or '')
                              for n, (k, _c, r) in
                              sorted(self.box_rewrites.items())))
        # [(1-based raw index, line, atoms, faces-at-that-point)]
        self.ai_lines = []
        faces = {}                       # die pid -> face code, per roll
        for i, l in enumerate(lines):
            if not l.startswith(pre):
                continue
            op, f = em.classify_line(l)
            if op == 'dice_roll':
                for p, v in zip(f.get('dice') or [], f.get('values') or []):
                    faces[str(p)] = str(v)
                continue
            if op is None or op in CONSEQUENCE_OPS:
                continue
            at = atoms_of(l, faces)
            if not at:
                continue
            rw = self.box_rewrites.get(i + 1)
            if rw is not None:
                kind, code, _name = rw
                if kind == 'noop':
                    continue                  # state noise, no move
                at = [(v, p, code if kind == 'exit' else frm,
                       code if kind == 'entry' else to, ex)
                      for (v, p, frm, to, ex) in at]
            self.ai_lines.append((i + 1, l, at, dict(faces)))
        self.k = 0                       # cursor into ai_lines
        self.decisions = 0
        self._errs = []

    # ------------------------------------------------------------------
    def _next(self):
        return self.ai_lines[self.k] if self.k < len(self.ai_lines) \
            else (None, '<script exhausted>', [], {})

    def faces_now(self):
        return self._next()[3]

    def _skip_die_kills(self):
        """The die kill is emitted at chain end, outside any decision:
        step the cursor over recorded die-kill lines (the runner's
        rendered parity still verifies the AI emitted them)."""
        while self.k < len(self.ai_lines) \
                and _is_die_kill(self.ai_lines[self.k][2]):
            print('  [script] recorded die kill at raw %d consumed (chain-end '
                  'emission)' % self.ai_lines[self.k][0])
            self.k += 1

    def _window_has_substance(self):
        """Do the recorded lines from the cursor up to the next die
        kill contain any non-die atom (pieces moving)?"""
        j = self.k
        while j < len(self.ai_lines):
            at = self.ai_lines[j][2]
            if _is_die_kill(at):
                return False
            if any(not a[1].startswith('die:') for a in at):
                return True
            j += 1
        return False

    def _match_atoms(self, atoms):
        """Consume recorded lines until their atom multiset covers the
        candidate's; equal -> the number of lines consumed, else -1."""
        want = sorted(atoms)
        have, n = [], 0
        while len(have) < len(want) and self.k + n < len(self.ai_lines):
            have.extend(self.ai_lines[self.k + n][2])
            n += 1
        return n if sorted(have) == want else -1

    def _dry_lines(self, server, dp, act):
        """The RAW lines the emitter would produce for `act` (clone-
        applied post under the probe pin, else the pre-apply render);
        None if neither render is possible."""
        import ai
        import copy
        post = None
        try:
            clone = ai.clone_state(server.driver.es)
            rctl = getattr(server.driver, 'rctl', None)
            cm = rctl.probe_cm if rctl is not None else server.rand.pin
            with cm():
                post = ai.apply_action(clone, act)
        except Exception as e:
            self._errs.append('apply %s: %r' % (act.action_type, e))
            post = None
        # the render BOOKS the pieces it moves on the client view (the
        # live emission relies on that); a dry run must leave the view
        # exactly as it found it or every later candidate that needs
        # the same pieces holds ('found 0') - base12 attacked because
        # of that.  Snapshot / restore the view around each render.
        view = server.driver.view
        snap = copy.deepcopy(view.__dict__)
        try:
            try:
                em = server._render(dp, act, post=post)
            except Exception as e:
                self._errs.append('render %s: %r' % (act.action_type, e))
                if post is None:
                    return None
                try:
                    em = server._render(dp, act, post=None)
                except Exception as e2:
                    self._errs.append('pre-render %s: %r'
                                      % (act.action_type, e2))
                    return None
        finally:
            view.__dict__.clear()
            view.__dict__.update(snap)
        if getattr(em, 'hold', None):
            # the emitter cannot express this candidate from the client
            # board (the live AI could not play it either): unrenderable
            self._errs.append('hold %s: %s' % (act.action_type,
                                               str(em.hold)[:90]))
            return None
        return list(getattr(em, 'lines', []) or [])

    def _rank_eyes(self, server, dp, cands):
        """Eye allocation renders through the planners at commit (not
        _render): the count is read straight from the recorded die
        moves into the hunt box (N consecutive SA-die moves to
        Game.1997 = N eyes)."""
        n = 0
        while self.k + n < len(self.ai_lines):
            at = self.ai_lines[self.k + n][2]
            if at and all(a[0] == 'move' and a[3] == HUNT_ZONE
                          and a[2] == SA_DICE_ZONE for a in at):
                n += 1
            else:
                break
        win = next((a for a in cands
                    if int((a.params or {}).get('count', -1)) == n), None)
        nxt = self._next()
        if win is None:
            raise DriverHalt(
                'SCRIPT: the recorded player allocated %d eye(s) at raw '
                'line %s but the engine menu offers %s - decision %d'
                % (n, nxt[0],
                   sorted(int((a.params or {}).get('count', -1))
                          for a in cands), self.decisions))
        self.k += n
        print('  [script] decision %d (%s eye): %d eye(s) <- raw %s'
              % (self.decisions, dp['side'].name, n, nxt[0]))
        ordered = list(cands)
        ordered.remove(win)
        ordered.insert(0, win)
        return ordered

    def rank(self, server, dp, actions):
        cands = list(actions)
        self.decisions += 1
        if str(dp.get('kind')) == 'eye' or all(
                a.action_type == 'eye_allocate' for a in cands):
            return self._rank_eyes(server, dp, cands)
        self._skip_die_kills()
        faces = self.faces_now()
        self._errs = []
        scored = []                      # (n_lines, action, atoms)
        for a in cands:
            lines = self._dry_lines(server, dp, a)
            if lines is None:
                scored.append((-2, a, []))
                continue
            atoms = []
            for l in lines:
                at = atoms_of(l, faces)
                if at:
                    atoms.extend(at)
            if not atoms:
                scored.append((0, a, []))
                continue
            if all(x[1].startswith('die:') for x in atoms) \
                    and self._window_has_substance():
                # only a die stage on the board (an attack declares by
                # chat): it cannot be a recorded window that shows
                # pieces moving before the die is killed
                scored.append((-1, a, atoms))
                continue
            n = self._match_atoms(atoms)
            scored.append((n if n > 0 else -1, a, atoms))
        best = max(s for s, _, _ in scored) if scored else -2
        nxt = self._next()
        if best > 0:
            winners = [t for t in scored if t[0] == best]
            if len(winners) > 1:
                msg = ('script: %d candidates mean the same at raw line '
                       '%s (%s %s); taking the first'
                       % (len(winners), nxt[0], dp['side'].name,
                          dp['kind']))
                server.notes.append(msg)
                print('  [script] %s' % msg)
            win = winners[0]
        elif best == 0:
            silent = [t for t in scored if t[0] == 0]
            if len(silent) != 1:
                loud = [t for t in scored if t[0] == -1][:3]
                raise DriverHalt(
                    'SCRIPT: no candidate means raw line %s %r (%s %s, '
                    '%d candidates, %d silent; e.g. %s%s) - decision %d'
                    % (nxt[0], nxt[1][:90], dp['side'].name, dp['kind'],
                       len(cands), len(silent),
                       '; '.join('%s->%r' % (t[1].action_type, t[2][:2])
                                 for t in loud) or '-',
                       ('; dry-run errors: ' + '; '.join(self._errs[:3]))
                       if self._errs else '', self.decisions))
            win = silent[0]
        else:
            sample = '; '.join('%s->%r' % (t[1].action_type, t[2][:2])
                               for t in scored[:4])
            raise DriverHalt(
                'SCRIPT: no candidate means raw line %s %r (%s %s, %d '
                'candidates; recorded atoms %r; e.g. %s%s) - decision %d'
                % (nxt[0], nxt[1][:90], dp['side'].name, dp['kind'],
                   len(cands), nxt[2][:3], sample,
                   ('; dry-run errors: ' + '; '.join(self._errs[:3]))
                   if self._errs else '', self.decisions))
        self.k += win[0]
        print('  [script] decision %d (%s %s): %s die=%r <- raw %s '
              '(+%d line(s))' % (self.decisions, dp['side'].name,
                                  dp['kind'], win[1].action_type,
                                  win[1].die_idx, nxt[0], win[0]))
        ordered = cands
        ordered.remove(win[1])
        ordered.insert(0, win[1])
        return ordered
