# -*- coding: utf-8 -*-
"""menu - the match.Menu of an open decision of the person (bridge2/m3/INTERFACE.md).

    opening = tables(state, board, cid_pid, bound)          once, when the decision opens
    m       = build(state, side, kind, options, opening, ctx)
    v       = match.judge(opening.base, table(board), m, opened=opening.opened)   per changed board
    steps   = realise(state, v.chain, v)                    at the commit: [(action, fix)]

Every callback of the Menu works on demand and remembers its result per option:
  plan(o)     the engine probe (ctx.apply), then the core's own renderer on a COPY of the view
              that shows BASE; the rendered lines and calls are read back as piece changes the
              way the client executes them.  Without a state after o (a roll or draw comes
              first) it is the move as the person performs it before the result is known
  conseq(o)   what reconcile still wants on BASE + plan(o) for the state after o
  expand(o)   the same person's next decision (ctx.following), asked on BASE + plan(o)
  sig / key / name / outcome   from the action's own parameters / the state after it
Where the rules leave the person a choice the renderer cannot name, the plan names a CLASS of
pieces (boardmodel.transfer) and realise() makes the engine hold what he did.  Nothing here
sends a line, touches the live view or consumes randomness, or imports the old bridge.
"""
import collections
import copy
import hashlib
import json
import re

from . import boardmodel as bm
from . import emission
from . import emitter as em
from . import layout
from . import match
from . import reconcile as rc

# a decision's opening: OPENED and BASE as tables, BASE as a board, what the comparison still
# wanted on the opening board (its ops, the rows nothing can show), and `pending` - every piece
# it will move once the game is settled: what it moves beyond them after his answer, the answer
# brought
Opening = collections.namedtuple('Opening', 'opened base board ops unshown pending')
Ctx = collections.namedtuple('Ctx', 'view nick apply following owned')
Ctx.__new__.__defaults__ = (None,)      # owned(state, board): see _wanted
# one option probed: the state after it (None: a roll / draw is needed first), the board after
# its own lines, its unnamed transfers, why it cannot be rendered, what the person may also do
_Run = collections.namedtuple('_Run', 'post after anon hold may')

PLAY = {'fp': 'Game.2037', 'sh': 'Game.2036'}             # the two Current Card slots
HAND = {'fp': rc.FP_HAND, 'sa': rc.SA_HAND}
DECK = {('fp', 'char'): 'Game.2004', ('fp', 'strat'): 'Game.2005',
        ('sa', 'char'): 'Game.2006', ('sa', 'strat'): 'Game.2007'}
# political, control and victory markers: only the political action moves one as its own move;
# everywhere else the person MAY move them (the server does it otherwise)
BOOKKEEPING = frozenset([str(i) for i in range(256, 264)] + [str(i) for i in range(405, 421)]
                        + ['422', '423'])
_UNIT =re.compile(r'^([A-Z_]+) (REGULAR|ELITE)$')
_FIGURE = re.compile(r'^([A-Z_]+) (REGULAR|ELITE|LEADER)$')
_SILENT = frozenset((None, 'chat', 'deck_commit', 'send_tokens', 'names', 'deselects', 'combat_roll',
                     'ctrl_179', 'ctrl_16', 'counter_adjust', 'turn_marker', 'combat_card_pick', 'combat_card_unpick',
                     'undo', 'dice_roll', 'dice_pickup'))    # lines that change no piece
_KILLS = frozenset(('kill', 'hand_exit_b', 'swap_place', 'move_662'))
_TO_PLAY = frozenset(('reveal_buffet', 'reveal_buffet9', 'buffet_move14'))
FACE_WORDS = {'A': 'Army', 'C': 'Character', 'M': 'Muster', 'H': 'Muster Army', 'P': 'Event', 'E': 'Eye',
              'W': 'Will of the West'}

# ---- the signature table: which parameters name the map regions an action's own moves touch
# (exactly), which actions touch none, and the zone families its moves may touch at all.  It is
# checked against every rendered plan of the corpus (test_core_match_corpus: prefilter soundness).
SIG_REGIONS = {'army_move': ('from', 'to'), 'companion_move': ('from', 'to'),
               'muster_recruit': ('region',), 'muster_elite': ('region',), 'muster_minion': ('region',),
               'companion_separate': ('to_region',), 'nazgul_move': ('from', 'to')}
SIG_NO_REGION = frozenset(('fellowship_move', 'advance_political', 'draw_card', 'play_event',
                           'elven_ring', 'skip', 'pass', 'army_attack', 'discard_table_card'))
SIG_SAME = frozenset(('army_attack', 'skip'))
SIG_MAP_ZONES = frozenset(('fp reinf', 'sa reinf', 'fp dead', 'sa dead', 'political', 'vp'))
SIG_MAP_TYPES = frozenset(('army_move', 'companion_move', 'nazgul_move', 'army_attack', 'pass', 'skip'))
SIG_ZONES = {'muster_recruit': SIG_MAP_ZONES | frozenset(('fp reinf:unit', 'sa reinf:unit',
                                                          'fp dead:unit', 'sa dead:unit')),
             'companion_separate': SIG_MAP_ZONES | frozenset(('fellowship box', 'guide box'))}
SIG_ZONES['muster_elite'] = SIG_ZONES['muster_recruit']


def signature(t, p, die):
    """match.Sig of an action (type, parameters, its die's kind) WITHOUT rendering it."""
    zones = SIG_MAP_ZONES if t in SIG_MAP_TYPES else SIG_ZONES.get(t)
    if t == 'card_option':
        # "this figure / group goes from A to B" rows of a card: the regions are parameters
        ch = p.get('choices') or {}
        grp = ch.get('move_group') if isinstance(ch.get('move_group'), dict) else {}
        fr, to = ch.get('from_region') or grp.get('from'), ch.get('to_region') or grp.get('to')
        if fr and to:
            return match.Sig(frozenset((bm.region_token(fr), bm.region_token(to))), None, None, None)
        return None
    if t in SIG_REGIONS:
        units = [p[k] for k in ('n_regular', 'n_elite') if isinstance(p.get(k), int)]
        return match.Sig(frozenset(bm.region_token(p[k]) for k in SIG_REGIONS[t] if p.get(k)),
                         die, None, zones, sum(units) if units else None)
    if t in SIG_NO_REGION:
        return match.Sig(frozenset(), die, (t, die) if t in SIG_SAME else None, zones)
    return None


def plain(x):
    """Action parameters as plain JSON values (enums by name, sets sorted)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [plain(v) for v in (sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x)]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(getattr(x, 'name', None) or getattr(x, 'value', None) or repr(x))[:80]


def choice_key(t, p, die):
    """Options with the same key are the same choice: the same action through another die of
    the same face; an Elven Ring used on another die that shows the same face."""
    if t == 'elven_ring' and isinstance(p, dict):
        p = {k: v for k, v in p.items() if k != 'target_die_idx'}
    return (t, json.dumps(plain(p), sort_keys=True), die)


# ---------------------------------------------------------------- boards (zone codes)
def table(board):
    """reconcile.Board -> boardmodel.Table (the `now` of match.judge)."""
    return bm.from_board(board.zone, board.face, board.mark)


def board_key(board):
    """Equal for two polls exactly when no piece changed its zone, die face or marker side."""
    return (frozenset(board.zone.items()), frozenset(board.face.items()),
            frozenset(board.mark.items()))


def _wanted(state, board, cid_pid=None, bound=None, settled=None, keep=None, dice=None, his=()):
    """What reconcile would still move on `board` for `state` -> (rows [(pid, zone, face,
    side)], its ops, the rows nothing can show).  `settled`: as if no decision were open.
    What an open or owed decision is about (keep(state, board); the owed rows of PICKS when
    the caller hands in no other answer) is wanted where it lies: the engine's provisional
    pick is never on the board a decision is read from.  dice: the binding of the die pieces,
    when the board is the client's own (reconcile.bind_dice; else bound here).  his: {piece:
    where he took it from} - of several of a kind THAT piece goes, and back there."""
    ops, unshown = rc.moves(rc.difference(state, board, cid_pid, settled or rc.settled(state), bound, dice,
                                          keep=(keep or owned)(state, board)), his)
    rows = []
    for op in ops:
        pid = str(op[2])
        if op[0] == 'move' and op[3] != op[4]:
            rows.append((pid, op[4], None, None))
        elif op[0] == 'face':
            rows.append((pid, None, op[3], None))
        elif op[0] == 'flip' and board.mark.get(pid) in (0, 1):
            rows.append((pid, None, None, 1 - board.mark[pid]))
    return rows, ops, unshown


def _put(board, rows):
    """A copy of the board with the rows applied."""
    out = rc.Board(dict(board.zone), dict(board.face), dict(board.mark))
    for row in rows:
        for part, value in zip(out, row[1:]):
            if value is not None:
                part[row[0]] = value
    return out


def _diff(a, b):
    """Piece changes from board a to board b as rows."""
    out = []
    for pid, z in b.zone.items():
        fa, fb, ma, mb = a.face.get(pid), b.face.get(pid), a.mark.get(pid), b.mark.get(pid)
        if a.zone.get(pid) != z or fa != fb or ma != mb:
            out.append((pid, z if a.zone.get(pid) != z else None, fb if fa != fb else None,
                        mb if ma != mb else None))
    return out


def _change(row, optional=False, src=None):
    return bm.Change(row[0], row[1], row[2], row[3], None, src, 1, optional)


def tables(state, board, cid_pid=None, bound=None, keep=None, his=(), dice=None, turned=()):
    """OPENED (the client board as it is) and BASE (the board as the engine state has it).
    keep(state, board): the pieces of a decision of his that is open or owed (see _wanted).
    his: the pieces he moved before this decision was asked and that are its to read
    (boardanswer: {piece: where he took it from}).  They are read as moved - OPENED has them in
    the PLACE BASE has them (a side or a face turned meanwhile is read both ways, like all that
    was not yet BASE - but of those HE turned, `turned`: OPENED shows them as BASE does) -; a
    piece of the same kind in the same place stands for one of them.
    dice: the loop's binding of the die pieces of this board (one binding for all who read it)."""
    his = dict(his)
    rows, ops, unshown = _wanted(state, board, cid_pid, bound, None, keep, dice, his)
    # a card the engine holds in a hand that already lies in a Current Card slot was put there
    # early (the next card of a chain, ahead of a roll).  While a decision is open the comparison
    # leaves the play area alone; BASE has such a card where the comparison will want it - in
    # the hand - so that playing it stays the move of the option that plays it
    later = _wanted(state, board, cid_pid, bound, True, keep, dice, his)[0]
    rows += [r for r in later if r[1] in HAND.values() and board.zone[r[0]] in PLAY.values()]
    # ... and a die or card HE put away before this decision was asked (out of the play area, or
    # out of his pool or hand by way of it: not one he only moved from box to box) lies in BASE in
    # the play area, where the action under way has it: tidied before or after, the same answer
    for pid, frm in sorted(his.items() if not rc.settled(state) else ()):
        row, here = str(layout.family(pid)), board.zone.get(pid)
        if not row.endswith((' die', ' card')) or any(r[0] == pid for r in rows):
            continue
        if here in layout.boxes(pid) and frm not in layout.boxes(pid):
            slot = rc.STAGED if row.endswith(' die') else PLAY['fp' if row.startswith('fp') else 'sh']
            rows.append((pid, frm if frm in rc.DISPLAY else slot, None, None))
        elif here not in rc.DISPLAY + tuple(layout.boxes(pid)) and frm and any(r[0] == pid for r in later):
            # ... and one he moved elsewhere, where the game does not have it (an Eye into the Hunt Box
            # before he is asked): BASE has it where he took it from - read as moved
            rows.append((pid, frm, None, None))
    base = _put(board, rows)
    like = {(bm.kind_of(p), board.zone.get(p)) for p in his}
    opened = _put(board, [r for r in rows if (r[0] in his and r[0] in turned) or (r[1] is not None and (
        r[0] in his or (bm.kind_of(r[0]), board.zone[r[0]]) in like))])
    # (a spent die still to be put away and a card in the play area are the staging pieces of an
    # action under way: tidying them is what the answer that ends the action brings, never older work)
    older = frozenset(r[0] for r in rows + later if not (
        (bm.kind_of(r[0]).startswith('die:') and r[1] not in (rc.FP_DICE, rc.SA_DICE))
        or (bm.kind_of(r[0]).startswith(('die:', 'card')) and board.zone[r[0]] in rc.DISPLAY)))
    return Opening(table(opened), table(base), base, ops, unshown, older)


def _named(f):
    """The pieces a rendered client line names (f: its fields)."""
    return [str(x) for k in ('pieces', 'piece', 'die', 'dice')
            for x in (f.get(k) if isinstance(f.get(k), list) else str(f.get(k) or '').split()) if str(x).isdigit()]


def execute(board, plan, tag, types=None):
    """The client's own execution of rendered lines and calls ([(op, fields)] of
    emission.plan_of) on `board` (changed in place) -> the unnamed transfers they cause."""
    anon = []
    for op, f in plan:
        f = f or {}
        if op == 'directive':
            parts = str(f.get('line', '')).split()
            name = parts[1] if len(parts) > 1 else ''
            draw = re.match(r'^draw_(fp|sa)_(char|strat)', name)
            if name == 'play_card' and len(parts) > 2:      # the card goes to his Current Card slot
                board.zone[parts[2]] = PLAY[tag]
            elif draw:
                anon.append(('card', DECK[draw.groups()], HAND[draw.group(1)], 1))
            elif name == 'wwsat':       # the client's own random discard from the Free Peoples hand
                anon.append(('card', rc.FP_HAND, rc.FP_DEAD, 1))
            continue
        if op in _SILENT:
            continue
        for pid in _named(f):
            if op == 'eye_alloc':
                board.zone[pid], board.face[pid] = em.HUNT_BOX, 'E'
            elif op == 'die_change':
                board.face[pid] = str(f.get('face') or '-').upper()[:1]
            elif op == 'flip':
                if board.mark.get(pid) in (0, 1):
                    board.mark[pid] = 1 - board.mark[pid]
                if pid == bm.FELLOWSHIP and board.mark.get(pid) == 1 and '248' in board.zone:
                    board.zone['248'] = 'Game.2014'         # the client resets the progress itself
            elif op in _KILLS or (op == 'hand_exit' and not f.get('to')):
                board.zone[pid] = layout.kill_box(pid, types)
            elif op in _TO_PLAY:
                board.zone[pid] = PLAY['fp' if f.get('frm') == rc.FP_HAND else 'sh']
            elif f.get('to'):
                board.zone[pid] = f['to']
    return anon


def idle(board, line, types=None):
    """Would the client change NOTHING by this line of a reply, although it is a line that moves pieces: every
    piece it names already lies where the client puts it (a move's place, the box of a kill: as execute has it) -
    or it plays a card that lies in no hand any more?  Such a line is never sent (link.replied)."""
    if line.startswith('!CALL play_card '):
        return board.zone.get(line.split()[-1]) not in (None,) + tuple(HAND.values())
    op, f = em.classify_line(line)
    to = {p: layout.kill_box(p, types) if op in _KILLS else f.get('to') for p in _named(f or {})}
    return bool(to) and all(z is not None and board.zone.get(p) == z for p, z in to.items())


def _view(live, state, board):
    """A copy of the live view that shows `board` under `state` (what Server._read_board sets)."""
    v = copy.copy(live)
    v._state_fn = lambda: state
    v.piece_region, bound = layout.bind_shared_boxes(state, board.zone)
    v.shadow_stronghold_regions = dict(bound)
    v.die_id_map = rc.bind_dice(state, board)
    v.piece_mark = dict(board.mark)
    v.fellowship_location = v.piece_region.get(bm.FELLOWSHIP)
    return v


def _render(view, nick, side, kind, act, pre, post):
    """The core's renderer for one option, on a view of its own, as a PERSON performs it.  The
    seat fact of the reveal flip: he is one seat, so the Free Peoples seat is the AI's exactly
    when he plays the Shadow (the AI's placement then turns the figure)."""
    v = copy.copy(view)
    for k in ('piece_region', 'shadow_stronghold_regions', 'die_id_map', 'piece_mark'):
        setattr(v, k, dict(getattr(view, k)))
    if kind == 'declare' and act.action_type == 'pass':
        return emission.Emission()
    return emission.render_step(v, nick, side, kind, act, post, pre, em.canon_side(side) == 'SHADOW', True)


def state_key(post):
    """A key equal for two engine states that are the same position: every field by value,
    armies as unordered rosters under their region.  Raises when the state cannot be read so."""
    from enum import Enum
    active = set()
    # the dice whose face no rule reads any more: a spent die, and the die of the two-step action under way
    # (a die in the Hunt Box keeps its face: it lies there AS that face)
    done = {id(d) for dice in post.dice.values() for d in dice if d.used}
    for step in ('pending_army_move', 'pending_muster'):
        i, dice = getattr(post, step + '_die', None), post.dice.get(getattr(post, step + '_side', None), ())
        if isinstance(i, int) and 0 <= i < len(dice):
            done.add(id(dice[i]))
    done -= {id(d) for d in post.hunt_box}

    def canon(x):
        if isinstance(x, Enum):
            return x.name
        if x is None or isinstance(x, (str, bytes, int, float, bool)):
            return x
        if id(x) in active:
            raise ValueError('cyclic state')
        active.add(id(x))
        try:
            if isinstance(x, dict):
                return tuple(sorted(((repr(canon(k)), canon(v)) for k, v in x.items()), key=repr))
            if isinstance(x, (set, frozenset)):
                return tuple(sorted(repr(canon(v)) for v in x))
            if isinstance(x, (list, tuple, collections.deque)):
                return tuple(canon(v) for v in x)
            if callable(x):
                return getattr(x, '__qualname__', type(x).__name__)
            fields = dict(getattr(x, '__dict__', None) or {})
            fields.update((n, getattr(x, n)) for n in getattr(type(x), '__slots__', ()) if hasattr(x, n))
            if id(x) in done:                               # the face such a die shows is no part of the position
                fields['current'] = None
            if type(x).__name__ == 'Army':                  # a roster has no order
                for name in ('units', 'leaders'):
                    fields[name] = sorted((canon(f) for f in fields.get(name) or ()), key=repr)
            return (type(x).__name__,) + tuple((k, canon(v)) for k, v in sorted(fields.items()))
        finally:
            active.discard(id(x))
    fields = dict(vars(post))
    for log in ('_casualty_operations', '_dg_src_log'):         # how it got there, not where it is
        fields.pop(log, None)
    armies = {}
    for region, army in (fields.pop('armies', None) or {}).items():
        armies.setdefault(bm.zone_id(region), []).append(canon(army))
    where = {id(d): (s.name, i) for s, dice in post.dice.items() for i, d in enumerate(dice)}
    body = (canon(fields), tuple(sorted(armies.items())),
            tuple(where.get(id(d)) for d in post.hunt_box))
    return hashlib.md5(repr(body).encode('utf-8', 'replace')).hexdigest()[:16]


def _unit_choice(act):
    """The card recruits "one unit (Regular or Elite)" and the option does not say which."""
    p = getattr(act, 'params', None) or {}
    if act.action_type not in ('play_event', 'card_option') or not p.get('card_id'):
        return False

    def names_type(d, depth=0):
        return isinstance(d, dict) and depth < 4 and (
            d.get('unit_type') is not None or d.get('unit_kind') is not None
            or any(names_type(v, depth + 1) for v in d.values()))
    try:
        import cards
        text = str(getattr(cards.get_card(str(p['card_id'])), 'event_effect', '') or '')
    except Exception:
        return False
    return bool(re.search(r'\(\s*regular\s+or\s+elite\s*\)', text, re.I)) and not names_type(p)


class _Decision(object):
    """One decision of the person on one board: each option's probe, plan and what follows."""

    def __init__(self, root, prefix, state, side, kind, options, board, ctx, base0, depth, faults):
        self.faults = faults                                # match.Menu.faults of the whole gesture
        self.root, self.prefix = root, list(prefix)         # ctx.apply starts at the gesture's opening
        self.state, self.side, self.kind = state, side, str(kind)
        self.options, self.board, self.ctx = list(options), board, ctx
        self.base0, self.depth = base0, depth               # BASE of the gesture: where pieces stood
        self.tag = 'fp' if getattr(side, 'name', side) == 'FREE_PEOPLES' else 'sh'
        self.cid_pid = dict(getattr(ctx.view, 'card_pid_map', None) or {})
        self.types = getattr(ctx.view, 'dynamic_types', None) or None
        pc = getattr(state, 'pending_combat', None)
        self.battle = sorted(set(str(pc[k]) for k in ('defender_region', 'attacker_region', 'region',
                                                      'trim_region') if pc.get(k))) \
            if isinstance(pc, dict) else []
        # the pair of shared Shadow stronghold boxes is one place while it holds one garrison
        held = self.battle + list(getattr(state, 'besieged_strongholds', None) or ())
        self.either = tuple(layout.SHARED_BOXES.values()) if layout.one_garrison(
            bm.region_token(r) for r in held) else ()
        self._v = self._runs = self._rest = None
        self._next = {}                                     # the Menu of the decision that follows each option

    def menu(self):
        return match.Menu(self.options, self.plan, self.sig, self.key, self.name, self.expand,
                          self.conseq, self.outcome, self.tag, self.net, self.faults)

    def net(self, o):
        """The option as the renderer performs it and all the engine does alone after it, as ONE
        change of this decision's board (named pieces: the replacement Regular out of the box
        the engine takes it from); None when that needs a roll, or a piece cannot be named."""
        post, after, anon, hold, _may = self.run(o)
        if post is None or hold is not None or anon:
            return None
        try:
            rows = _wanted(post, after, self.cid_pid, keep=self.ctx.owned)[0]
            if self._rest is None:
                self._rest = _wanted(self.state, self.board, self.cid_pid, keep=self.ctx.owned)[0]
        except Exception as e:
            return self._failed(o, 'the comparison', e)
        return [_change(r) for r in _diff(self.board, _put(after, [r for r in rows if r not in self._rest]))]

    def _failed(self, o, what, error):
        """A FAULT (not a hold) took an option, its consequences or its outcome away from the
        matcher: recorded once on the Menu (match.Menu.faults), never swallowed.  -> None"""
        if not any(f[0] is o and f[1] == what for f in self.faults):
            self.faults.append((o, what, error))

    def run(self, o):
        """The probe of one option (_Run), once per option."""
        if self._runs is None:
            self._runs, self._v = {}, _view(self.ctx.view, self.state, self.board)
        got = self._runs.get(id(o))
        if got is None:
            post = self.ctx.apply(self.root, self.prefix + [o])
            try:
                out = _render(self._v, self.ctx.nick, self.side, self.kind, o, self.state, post)
                hold = str(out.hold)[:160] if out.hold else None
            except Exception as e:
                hold = 'render error %s: %s' % (type(e).__name__, str(e)[:100])
                self._failed(o, 'the renderer', e)
            after, anon, may = None, [], []
            if hold is None:
                after = _put(self.board, ())
                anon = execute(after, emission.plan_of(out), self.tag, self.types)
                if out.may:
                    other = _put(after, ())
                    execute(other, emission.may_of(out), self.tag, self.types)
                    may = _diff(after, other)
            got = self._runs[id(o)] = _Run(post, after, anon, hold, may)
        return got

    def plan(self, o):
        post, after, anon, hold, may = self.run(o)
        t = o.action_type
        if hold is not None or t in HIDDEN:
            return None
        # every piece change of the rendered lines is the option's OWN move (nothing that all
        # options share is ever factored out: with fungible pieces - dice, units - that turned
        # "k dice" into "k or k + 1 dice"); only the bookkeeping markers, and what the renderer
        # itself says a person may leave to the server (Emission.may), are optional
        rows = _diff(self.board, after)
        dice = bm.FP_DICE if self.tag == 'fp' else bm.SH_DICE
        book = [r for r in rows if r[0] in BOOKKEEPING]
        if not anon and all(r in book or r[0] in dice for r in rows):
            book = []           # a marker step that is ALL the option shows beside its die is its own
            #                     move (the political action, a card's choice of a nation)
        own, classes = self._free(o, [r for r in rows if r not in book])
        # where the person may put a piece INSTEAD of the place the renderer names (the declared
        # Fellowship straight onto the Mordor Track): that piece, in either place
        other = {r[0]: r[1] for r in may if r[1] is not None and r[2] is None and r[3] is None}
        for r in [r for r in own if r[0] in other and r[1] is not None and r[2] is None and r[3] is None]:
            own = [x for x in own if x is not r]
            may = [x for x in may if x[0] != r[0]]
            kind = bm.kind_of(r[0], after.face.get(r[0]), bm.side(r[0], after.mark.get(r[0])))
            classes = classes + [(kind, self.board.zone[r[0]], (r[1], other[r[0]]), 1)]
        # the window closes: his die and card are killed.  A card the option took out of a Board
        # slot is on its way to the discards whatever follows: it may be killed at once
        closed = rc.settled(post) or (post is None and t == 'draw_card')    # (a draw always ends its action)
        sweep = [(pid, layout.kill_box(pid), None, None) for pid, z in after.zone.items()
                 if (closed and z == em.DICE_USED and pid in dice)
                 or (z == PLAY[self.tag] and bm.kind_of(pid).startswith('card#')
                     and (closed or self.board.zone.get(pid) in em._TABLE_SLOTS))]
        # ... also when the rendered lines make that kill themselves: the die of a two-step
        # action lay in the play area already, and its second move closes the action; or the
        # die comes out of the pool and the action closes at once - then putting it into the
        # play area is his move and the kill is "I am done", never required (an option that is
        # nothing but the kill - "no second move", "discard the die" - keeps it as its own move)
        pool = rc.FP_DICE if self.tag == 'fp' else rc.SA_DICE
        killed = [r for r in own if r[0] in dice and r[1] == layout.kill_box(r[0]) and (
            self.board.zone.get(r[0]) == pool or (closed and self.board.zone.get(r[0]) == em.DICE_USED))]
        if len(killed) < len(own) + len(classes):
            fresh = [r for r in killed if self.board.zone[r[0]] == pool]
            own = [r for r in own if r not in killed] + [(r[0], em.DICE_USED) + r[2:] for r in fresh]
            sweep += [r[:2] + (None, None) if r in fresh else r for r in killed]
        return ([_change(r, src=self.board.zone.get(r[0])) for r in own] + [bm.transfer(*c) for c in classes + anon]
                + [_change(r, True) for r in book + sweep + may])

    def _free(self, o, rows, part='own'):
        """-> (rows that stay named, class transfers): the renderer names ONE way to perform an
        option; where a piece it names is a pick of the person (a row of PICKS for this part of
        the option) the plan names the row's class and places instead."""
        picks = [k for k in PICKS if k.part == part and k.on(o)]
        keep, groups = [], collections.Counter()
        for row in rows:
            kind = bm.kind_of(row[0])
            m = _UNIT.match(kind)
            mv = _Move(row, kind, m and m.groups(), 'fp' if m and m.group(1) in bm.FP_NATIONS else 'sa',
                       self.base0.place(row[0])[0] if row[0] in self.base0.p else None,
                       bm.zone_id(row[1]) if row[1] is not None else None)
            hit = next((t for t in (k.take(self, o, mv) for k in picks if k.take) if t), None)
            if row[1] in self.either and mv.was is not None:    # into either box of the pair
                hit = (hit[0] if hit else kind, hit[1] if hit else mv.was, self.either)
            if hit:
                groups[hit] += 1
            else:
                keep.append(row)
        return keep, [(c, src, dst, n) for (c, src, dst), n in groups.items()] + [
            c for k in picks if k.adds and not rows for c in k.adds(self, o)]

    def conseq(self, o):
        post, after, anon, hold, _may = self.run(o)
        if post is None or hold is not None:
            return None
        try:
            rows = _wanted(post, after, self.cid_pid, keep=self.ctx.owned)[0]
            if self._rest is None:      # what reconcile wants on this decision's board anyway
                self._rest = _wanted(self.state, self.board, self.cid_pid, keep=self.ctx.owned)[0]
        except Exception as e:
            return self._failed(o, 'the comparison', e)
        rows = [r for r in rows if r not in self._rest]
        more = []
        for (tag, dk), deck in DECK.items():    # cards the engine drew, beyond the named ones
            attr = ('fp_' if tag == 'fp' else 'sh_') + ('char_deck' if dk == 'char' else 'strat_deck')
            n = len(getattr(self.state, attr, None) or ()) - len(getattr(post, attr, None) or ())
            n -= sum(1 for r in rows if r[1] == HAND[tag] and after.zone.get(r[0]) == deck)
            n -= sum(a[3] for a in anon if a[1] == deck)
            if n > 0:
                more.append(('card', deck, HAND[tag], n))
        rows, free = self._free(o, rows, 'after')
        return ([_change(r) for r in rows] + [bm.transfer(*a) for a in more + free]) or None

    def expand(self, o):
        post, after, _anon, hold, _may = self.run(o)
        if hold is not None:
            return None
        if post is None:
            return match.ROLL           # what follows needs a roll / draw / the other seat
        if id(o) not in self._next:     # (once per option: every later judge of the gesture reads the SAME decision,
            nxt = self.ctx.following(post, self.side) if self.depth > 0 else None   # with what was rendered for it)
            self._next[id(o)] = _Decision(self.root, self.prefix + [o], post, self.side, nxt[0], nxt[1], after, self.ctx,
                                          self.base0, self.depth - 1, self.faults).menu() if nxt and nxt[1] else None
        return self._next[id(o)]

    def outcome(self, o):
        post = self.run(o).post
        try:
            return state_key(post) if post is not None else None
        except Exception as e:
            return self._failed(o, 'the outcome', e)

    def _die(self, o):
        letter = die_letter(self.state, self.side, o)
        return 'die:%s:%s' % (self.tag, letter) if letter else None

    def sig(self, o):
        # the die of a two-step action under way left the pool with the FIRST step: the second
        # move, the second recruit take no die from the pool (Sig.die None)
        i, st = getattr(o, 'die_idx', None), self.state
        under_way = any(getattr(st, a + '_die', None) == i and getattr(st, a + '_side', None) is self.side
                        for a in ('pending_army_move', 'pending_muster')) and i is not None
        return signature(o.action_type, o.params or {}, None if under_way else self._die(o))

    def key(self, o):
        return choice_key(o.action_type, o.params, self._die(o))

    def name(self, o):
        try:
            from . import words
            text = words.option_label(self.state, self.side, o, self.side)
        except Exception:
            text = str(getattr(o, 'action_type', o))
        # ONE CHOICE, ONE NAME (ruling 9): the same action through a die of ANOTHER face is
        # another choice (another die is spent) - then the name tells the die
        die = self._die(o)
        if die and any(x.action_type == o.action_type and x.params == o.params
                       and self._die(x) not in (None, die) for x in self.options):
            text += ' (%s die)' % FACE_WORDS.get(die[-1], die[-1])
        return text


def die_letter(state, side, o):
    """The face of the unspent action die an option spends ('A', 'C', ...), or None."""
    i = getattr(o, 'die_idx', None)
    try:
        die = state.dice[side][i] if i is not None and i >= 0 else None
    except Exception:
        die = None
    return rc.FACE_LETTER.get(getattr(getattr(die, 'current', None), 'name', None)) \
        if die is not None and not die.used else None


# the options whose move is never read from the board: the hidden commit of a combat card (it is
# chosen with the client's chooser, a click or a word)
HIDDEN = frozenset(('combat_card_play',))


def uses(state, side, options, board, pid, cid_pid, following=None):
    """Does a decision (its state, side and options) use a die or a card that lies in the play
    area: one of its options that the board can show names that card, or spends a die that
    shows the face this one shows?  Or - following(a) -> (state, options) of his decision that
    follows the option a AT ONCE, or None - does the decision that follows an option which
    spends no die itself (it may turn one: the chain of the two is one answer of his)?"""
    cid = next((json.dumps(str(c)) for c, p in cid_pid.items() if str(p) == pid), None)
    face, shown = board.face.get(pid), [a for a in options if a.action_type not in HIDDEN]
    if any((cid is not None and cid in json.dumps(plain(a.params or {})))
           or (face is not None and die_letter(state, side, a) == face) for a in shown):
        return True
    later = [a for a in shown if die_letter(state, side, a) is None] if following is not None and (cid or face) else ()
    return any(uses(nxt[0], side, nxt[1], board, pid, cid_pid) for nxt in map(following, later) if nxt)


def build(state, side, kind, options, opening, ctx):
    """The match.Menu of the decision the person has open (INTERFACE.md section 1)."""
    return _Decision(state, [], state, side, kind, options, opening.board, ctx, opening.base,
                     match.MAX_DEPTH, []).menu()


# ---------------------------------------------------------------- the pick table
# One row per thing a person picks among equivalent pieces while he performs an option, or owns
# while a decision of his is open or owed.  Three readers, no fourth copy of any rule:
#   the plan (_Decision._free)   where a rendered line names a piece of a row, the plan names the
#                                row's class and places (take / adds), so any such piece fits
#   realise                      the engine is told by `tell`: (a parameter of the action, its
#                                value from the pieces he used), or HOLD, or None = nothing to tell
#   the comparison (owned)       `owed(state, board, phase, person)`: the pieces it leaves where
#                                they lie; `about`: the kind of the decision whose answer they
#                                are - any other open decision of his does not read them
# The places that are one place for a family of pieces are layout's rows; a row here names a
# CHOICE of place only where the rules give one.
Pick = collections.namedtuple('Pick', 'what on part take adds tell owed about')
Pick.__new__.__defaults__ = (None,)
# one rendered piece change: the (pid, zone code, face, side) row, its kind, (NATION, TYPE) of a
# unit, its side, the place it had when the gesture began, the zone id it goes to
_Move = collections.namedtuple('_Move', 'row kind unit side was dst')
# the ONE generic hold: the fields of the state after a step that _hold may set, and no others
HOLD = ('hold', ('the nation of a Leader in a region the step touched, and the casualty record of '
                 'that Leader', 'Regular for Elite of a unit the step placed'))


def _types(*names):
    return lambda o: o.action_type in names


def _by_counts(*names):
    return lambda o: o.action_type in names and not (o.params or {}).get('unit_set')


def _eyes(d, o, mv):
    """Any dice of his pool, whatever they show (before the roll the faces mean nothing)."""
    return ('die:sh', em.SH_DICE_POOL, em.HUNT_BOX) if mv.row[1] == em.HUNT_BOX else None


def _leader(d, o):
    """A Leader of any nation leaves the battle region (or its stronghold box) for the card or
    the casualties: the card names no nation, and without a count it takes one Leader."""
    zones = tuple(z for r in d.battle for z in (bm.zone_id(r), bm.zone_id(r) + ' stronghold'))
    n = (o.params or {}).get('count')
    n = 1 if n is None else int(n)
    return [('leader:fp', zones, ('card play', 'fp dead'), n)] if zones and n > 0 else []


def _units(d, o, mv):
    """A Regular / an Elite of any nation of the side, out of the region the option names (its
    stronghold box when it stood in the box when the gesture began - never the place an EARLIER
    step moved it to)."""
    frm = bm.zone_id(o.params['from']) if (o.params or {}).get('from') else None
    src = None
    if mv.was is not None and bm.is_map_region(mv.was):
        src = mv.was if (frm is None or bm.region_of(mv.was) == frm) else frm
    if mv.unit and mv.dst is not None and bm.is_map_region(mv.dst) and src not in (None, mv.dst):
        return ('%s:%s' % (mv.unit[1].lower(), mv.side), src, mv.row[1])


def _replacement(d, o, mv, klass=None):
    """The Regular that takes a reduced Elite's seat comes from either box of its side."""
    if mv.unit and mv.unit[1] == 'REGULAR' and mv.dst is not None and bm.is_map_region(mv.dst):
        return (klass or mv.kind, (mv.side + ' dead', mv.side + ' reinf'), mv.row[1])


def _payer(d, o, mv):
    """Any Elite of the side that stands in the battle pays; the Regular that takes its seat
    follows its nation."""
    if mv.unit and mv.unit[1] == 'ELITE' and mv.was is not None and bm.is_map_region(mv.was) \
            and mv.dst is not None and not bm.is_map_region(mv.dst):
        return ('elite:' + mv.side, mv.was, mv.row[1])
    return _replacement(d, o, mv, 'regular:' + mv.side)


def _either_type(d, o, mv):
    """The unit a card recruits, of that nation, Regular or Elite."""
    if mv.unit and mv.was is not None and not bm.is_map_region(mv.was) and mv.dst is not None \
            and bm.is_map_region(mv.dst):
        return ('unit:' + mv.unit[0], mv.was, mv.row[1])


def _reduces(o):
    return (em.casualty_action_params(o) or {}).get('operation') == 'reduce_elite'


def _army(state, region):
    """The army of the state in a map region (boardmodel.region_token), or None."""
    return next((a for k, a in state.armies.items() if bm.region_token(k) == region), None)


def _unit_set(o, put, state):
    """The units he moved into the region the move names.  Where they are fewer than it moves
    (he named the move by a click or a word), the rest are the first units of their kind and
    side that still stand in the region it leaves: his units stay, the others are moved for
    him.  state: the game before the move, or None where it is not known (then: whole sets)."""
    from army_card_cargo import freeze
    p = o.params or {}
    units = collections.Counter({('unit', n, k): v for (r, n, k), v in put.items()
                                 if p.get('to') and r == bm.region_token(p['to']) and v > 0 and k != 'LEADER'})
    lack = collections.Counter(REGULAR=int(p.get('n_regular') or 0), ELITE=int(p.get('n_elite') or 0))
    for key, v in units.items():
        lack[key[2]] -= v
    his = collections.Counter(units)
    fp = any(k[1] in layout.FP_NATIONS for k in units)
    for u in getattr(_army(state, bm.region_token(p['from'])), 'units', ()) if state is not None and units else ():
        key = ('unit', u.nation.name, u.kind.name)
        if his[key] > 0:
            his[key] -= 1                       # (one of those he moved)
        elif lack[key[2]] > 0 and (key[1] in layout.FP_NATIONS) == fp:
            lack[key[2]], units[key] = lack[key[2]] - 1, units[key] + 1
    if units and not any(lack.values()):
        return freeze(units)


def _nation(o, put, state):
    """The nation of the one Elite he reduced."""
    from game_state import Nation
    gone = [n for (_r, n, k), v in put.items() if k == 'ELITE' and v < 0]
    return Nation[gone[0]] if len(gone) == 1 else None


def _companions(board, names):
    """The pieces on the board of the Companions with these names."""
    import log_replay as lr
    return [pid for pid in lr.COMPANIONS if pid in board.zone
            and names & set(lr.names_for_pid(pid) or (lr.COMPANIONS[pid],))]


# the phases of the turn (the loop's cursor) in which the Guide of a Free Peoples person is still
# to be chosen: from the turn's start until the Guide question is answered
GUIDE_OWED = ('draws', 'open', 'declare', 'tail0', 'tail', 'guide')


def _guide(state, board, phase, person):
    """While the Guide is to be chosen the Guide box and the Fellowship box are his: the
    Companion the engine seats meanwhile is not put there for him.  A person who plays the
    Free Peoples may seat another Guide during the whole Fellowship phase - while its
    declaration is open or owed, until the Guide question is answered."""
    fs = state.fellowship
    if state.pending_guide_change is None and not (
            phase in GUIDE_OWED and person is not None and person.name == 'FREE_PEOPLES'):
        return ()
    names = {str(c.name) for c in list(fs.companions) + ([fs.guide] if fs.guide else [])}
    return [pid for pid in _companions(board, names)
            if board.zone[pid] in (em.GUIDE_BOX, em.FELLOWSHIP_BOX)]


def _hobbit(state, board, *_turn):
    """A hobbit the Hunt took out of the Fellowship stands in no place of the engine until
    his placement is answered: he stays where he lies."""
    hunt = state.pending_hunt or {}
    names = {str(c.name) for c in getattr(state, '_hobbit_place_queue', None) or ()}
    return _companions(board, names | ({str(hunt['companion'])} if hunt.get('phase') == 'hobbit_place' else set()))


# the track markers a put-off action of each type has moved when a person made it on the board
PUT_OFF = {'fellowship_move': ('248',)}


def _put_off(state, board, *_turn):
    """An action the engine has put off behind the other side's card choice and takes up when
    that is answered (the Fellowship's move behind Flocks of Crebain) is on his board whole
    already: its marker, and its die in the Hunt Box, stay where they lie until then."""
    act = (getattr(state, 'pending_card_option', None) or {}).get('resume_action') or {}
    track = PUT_OFF.get(act.get('action_type'), ())
    return list(track) + [pid for pid in (bm.FP_DICE if track else ()) if board.zone.get(pid) == em.HUNT_BOX]


PICKS = (
    Pick('the dice that become Eyes', _types('eye_allocate'), 'own', _eyes, None, None, None),
    Pick('the Leader a combat card takes', lambda o: o.action_type == 'combat_effect_choice'
         and (o.params or {}).get('target') == 'leader', 'own', None, _leader, HOLD, None),
    Pick('the units of an army move', _by_counts('army_move'), 'own', _units, None,
         ('unit_set', _unit_set), None),
    Pick('the units of an army that retreats or advances whole', _by_counts('victor_advance', 'combat_retreat'),
         'own', _units, None, None, None),
    Pick('the Elite that pays for pressing a siege', _types('siege_press'), 'own', _payer, None,
         ('nation', _nation), None),
    Pick('the box the Regular of a reduced Elite comes from', _reduces, 'own', _replacement, None, None, None),
    Pick('Regular or Elite, where a card recruits either', _unit_choice, 'after', _either_type, None,
         HOLD, None),
    Pick('the Guide, while he is to be chosen', None, None, None, None, None, _guide, 'guide_change'),
    Pick('a hobbit taken out of the Fellowship, until he is placed', None, None, None, None, None, _hobbit),
    Pick('the pieces of a move the game has put off, until it is taken up', None, None, None, None,
         None, _put_off),
)


def owned(state, board, phase=None, person=None, about=None):
    """The pieces an open or owed decision of a person is about (the `owed` rows of PICKS): the
    comparison leaves each where it lies, and BASE of a decision is built with them where they
    lie.  phase, person: the turn's phase and the person's side, when the caller knows them.
    about: only the rows that are another decision's than one of this kind (elsewhere).  A die
    or a card in the play area is in no row: the board does not say who laid it there (the
    loop does: boardanswer)."""
    return frozenset(str(pid) for k in PICKS if k.owed and (about is None or k.about not in (None, about))
                     for pid in k.owed(state, board, phase, person))


def reading(board, opening, elsewhere):
    """The board as an open decision reads it: the pieces of ANOTHER decision of his that is
    owed (owned(..., about=its kind)) lie where the BASE of its opening has them - what he does
    with them is that other decision's answer, never a move of this one."""
    if not elsewhere:
        return table(board)
    base, out = opening.board, rc.Board(dict(board.zone), dict(board.face), dict(board.mark))
    for pid in elsewhere:
        for now, was in zip(out, base):
            if pid in was:
                now[pid] = was[pid]
    return table(out)


def realise(state, chain, verdict):
    """-> [(action, fix)] per step of a committed chain: the action with the parameter that
    makes the engine take the person's own pieces (the `tell` of its rows of PICKS), and -
    where the engine has no parameter - fix(post), which makes the state just applied hold what
    he showed (see _hold).  The core applies the pairs in order and calls every fix; a fix
    changes figures only inside regions its step touched, and no count."""
    import ai
    done = list(getattr(verdict, 'performed', None) or ())
    cell, steps = [state], []           # cell: the state before the step being fixed
    for i, o in enumerate(chain):
        put = collections.Counter()     # (region, NATION, KIND): +n / -n the person showed
        for z, k, v in (done[i] if i < len(done) else ()):
            m = _FIGURE.match(k)
            if m and bm.is_map_region(z):
                put[(bm.region_of(z), m.group(1), m.group(2))] += v
        for name, value in [k.tell for k in PICKS if k.on and k.tell not in (None, HOLD) and k.on(o)]:
            told = value(o, put, None if i else state) if (o.params or {}).get(name) is None else None
            if told is not None:
                o = ai.Action(o.die_idx, o.action_type, dict(o.params or {}, **{name: told}))
        steps.append((o, put))
    if not any(put for _a, put in steps):
        return [(a, None) for a, _put in steps]

    def fixer(put):
        def fix(post):
            pre, cell[0] = cell[0], post
            if put:
                _hold(pre, post, put)
        return fix
    return [(a, fixer(put)) for a, put in steps]


def _hold(pre, post, put):
    """Make `post` hold the person's figures where the engine chose others of the same count:
    a remaining Leader takes the nation of the one the engine removed instead (the casualty
    record follows); an Elite the engine placed becomes the Regular he placed (while the
    reinforcements hold one)."""
    import ai
    from game_state import Unit, UnitKind

    def figures(state, region):
        army = _army(state, region)
        return army, collections.Counter(
            [(f.nation.name, 'LEADER') for f in getattr(army, 'leaders', ()) if type(f).__name__ == 'Leader']
            + [(u.nation.name, u.kind.name) for u in getattr(army, 'units', ())])
    for region in sorted(set(r for r, _n, _k in put)):
        army, now = figures(post, region)
        was = figures(pre, region)[1]
        if army is None:
            continue
        diff = {k: now[k] - was[k] - put.get((region,) + k, 0) for k in set(now) | set(was)
                | set((n, k) for r, n, k in put if r == region)}
        over = sorted(k for k, v in diff.items() for _ in range(max(0, v)))     # the engine has, he has not
        under = sorted(k for k, v in diff.items() for _ in range(max(0, -v)))   # he has, the engine has not
        for have, want in zip([k for k in over if k[1] == 'LEADER'], [k for k in under if k[1] == 'LEADER']):
            fig = next((f for f in army.leaders if type(f).__name__ == 'Leader'
                        and f.nation.name == have[0]), None)
            target = next((n for n in type(fig.nation) if n.name == want[0]), None) if fig else None
            if target is None:
                continue
            dead, old = getattr(post, 'fp_casualties', None), getattr(pre, 'fp_casualties', None) or {}
            if isinstance(dead, dict) and (dead.get(target) or {}).get('leader', 0) \
                    > (old.get(target) or {}).get('leader', 0):
                dead[target]['leader'] -= 1
                box = dead.setdefault(fig.nation, {'regular': 0, 'elite': 0, 'leader': 0})
                box['leader'] = box.get('leader', 0) + 1
            fig.nation = target
        for nation, kind in over:
            i = next((i for i in range(len(army.units) - 1, -1, -1)
                      if army.units[i].nation.name == nation and army.units[i].kind is UnitKind.ELITE), None)
            if kind != 'ELITE' or (nation, 'REGULAR') not in under or i is None:
                continue
            under.remove((nation, 'REGULAR'))
            unit = army.units.pop(i)
            ok = ai._can_recruit(post, unit.nation, 'regular')
            army.units.insert(i, Unit(nation=unit.nation, kind=UnitKind.REGULAR) if ok else unit)
