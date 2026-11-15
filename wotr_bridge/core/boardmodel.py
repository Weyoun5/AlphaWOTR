# -*- coding: utf-8 -*-
"""boardmodel - the client board as a piece table (bridge2/DESIGN.md, principle 4).

A Table is {piece id: (zone, face, state)} read from the client's board dump
(ai_client_board.txt: 'idx <TAB> <~zone code~> <TAB> pids <TAB> flags <TAB>
marker states <TAB> die faces').  Two tables are compared piece by piece, but
interchangeable pieces count as a multiset: every piece has a KIND
('GONDOR REGULAR', 'die:fp:C', 'card#141', ...) and a board is the count of
each kind in each zone.  The matcher works on the DIFFERENCE between two such
counts (a small dict {(zone, kind): +n / -n}).

  Table.apply(plan)                    -> the table after a move plan
  delta(a, b)                          -> {(zone, kind): b - a}
  plan_delta(base, plan)               -> (delta of the plan, coarse rules, optional delta)
  fits / inside / between              -> the three relations the matcher uses
  residue / part_of / kills / closes_only -> what is left over, strict parts, the closing kill
  performed(d, plan)                   -> which pieces realised a plan's free choices
  bind(d, coarse, now)                 -> unnamed card / tile transfers named by the board
  ops_plan(table, ops)                 -> reconcile.moves' ops as a plan (engine-projected BASE)
  arrived(now, opened, base)           -> NOW with what was still in transit taken as arrived
  distance(d1, d2)                     -> how many piece placements differ
  sentence(now_delta, target_delta)    -> the difference in plain words
  observed(d) / touched(d)             -> what a delta touches, for the prefilter

A move plan is a list of Change:
  move(pid, zone)              the piece goes to a zone
  face(pid, 'C')               an action die shows another face
  flip(pid, state)             a marker shows its other side (0 / 1)
  transfer('card', src, dst)   one unnamed piece of a class goes from src to
                               dst (a card drawn from a deck: which card is
                               not known when the plan is made).  The class
                               may be a FREE CHOICE of the player: 'leader:fp'
                               (a Leader of any nation), 'unit:ROHAN' (Regular
                               or Elite), 'regular:fp' (a Regular of any Free
                               Peoples nation); src / dst may be a tuple of zones
                               (the Regular comes from either box).  Every
                               realisation fits; performed() names the one
                               on the board.
  optional(change)             a change the player MAY also make (closing the
                               die / card window, placing what the server
                               would otherwise place): the board fits the
                               plan with or without it
Zones are client zone codes ('Game.1965') or their names ('erebor'); both are
normalised by zone_id().

Equivalences built into the model (each is a statement about the game, not
about one gesture):
  * action dice of one side showing the same face are one kind;
  * army pieces of one nation and type are one kind; so are the control
    markers of a side, the three Elven rings, hunt tiles with the same face;
  * the slots of layout.SLOTS are one zone each: the numbered table slots
    ('Board1'..'Board14'), the Elven ring slots, the used-ring slots, the two
    Current Card slots (a card is one of a kind: the slot it lies in does
    not change which card it is);
  * a hunt tile lying in the play area (Current Action / a Current Card slot)
    counts as a used tile;
  * the boxes off the map that are one place for a family of pieces
    (layout.BOXES) are one zone for a piece of that family: a spent action
    die in any of the four boxes, a Shadow unit in either Shadow box, a dead
    Companion wherever it was put - never a dead Free Peoples Regular or
    Elite, which the rules count;
  * an action die has a face only in its dice pool: staged in the play area
    or used, it is 'a die' (a player who uses an Elven Ring often does not
    turn the die before he stages it).  What the dice outside the pools SHOW
    is still read by the matcher (moved_faces) to reject a reading the shown
    face contradicts;
  * the selection flags of the dump (4th column) are ignored.
Nothing here imports the old bridge (wotr_bridge/live).
"""
import collections
import re
import unicodedata

from . import layout
from .layout import ELVEN_RINGS, FP_DICE, FP_NATIONS, SH_DICE

Change = collections.namedtuple('Change', 'pid zone face state klass src n opt')


def move(pid, zone):
    return Change(str(pid), zone, None, None, None, None, 1, False)


def face(pid, letter):
    return Change(str(pid), None, str(letter), None, None, None, 1, False)


def flip(pid, state):
    return Change(str(pid), None, None, int(state), None, None, 1, False)


def transfer(klass, src, dst, n=1):
    return Change(None, dst, None, None, str(klass), src, int(n), False)


def optional(change):
    return change._replace(opt=True)


# ---------------------------------------------------------------- piece kinds
FELLOWSHIP = '247'
# pieces with two sides that matter: Gandalf (Grey / White), Strider (/ Aragorn), the Fellowship
# (hidden / revealed), the political markers (passive / active); side 0 / 1 as the dump states it
STATEFUL = frozenset(('240', '241', '247', '256', '257', '258', '259', '260', '261', '262', '263'))
_SIDES = {'Gandalf': ('Gandalf the Grey', 'Gandalf the White'), 'Strider': ('Strider', 'Aragorn'),
          'Fellowship': ('the Fellowship (hidden side up)', 'the Fellowship (revealed side up)')}


def side(pid, state):
    """The side (0 / 1 / None) a dump's marker state means for a piece (Gandalf has two
    White states)."""
    return 1 if (state == 2 and str(pid) == '240') else state if state in (0, 1) else None
_LABELS = None
_TILES = None


def _tables():
    """(piece labels, hunt tile faces) from the engine's own tables (src/)."""
    global _LABELS, _TILES
    if _LABELS is None:
        try:
            import piece_table_authoritative as pta
            _LABELS = dict(pta.AUTH_PIECE_LABELS)
        except Exception:
            _LABELS = {}
        try:
            import log_replay as lr
            _TILES = {str(p): str(v) for p, v in lr.HUNT_TILE_PIECES.items()}
        except Exception:
            _TILES = {}
    return _LABELS, _TILES


def kind_of(pid, face_letter=None, state=None, types=None):
    """The kind of one piece.  `types` = optional {pid: label} overrides (a
    piece the client re-typed at run time)."""
    pid = str(pid)
    labels, tiles = _tables()
    if pid in FP_DICE:
        return 'die:fp:%s' % (face_letter if face_letter not in (None, '', '-') else '?')
    if pid in SH_DICE:
        return 'die:sh:%s' % (face_letter if face_letter not in (None, '', '-') else '?')
    if types and pid in types:
        return str(types[pid])
    if pid in tiles:
        return 'tile:%s' % tiles[pid]
    if pid in ELVEN_RINGS:
        return 'elven ring'
    label = labels.get(pid)
    if label is None:
        return 'piece#%s' % pid
    if '/' in label:                       # an event card: one of a kind
        return 'card#%s' % pid
    if label == 'hunt tile':
        return 'tile#%s' % pid
    if pid in STATEFUL and state in (0, 1):
        return '%s:%d' % (label, state)
    return label


_UNIT_RE = re.compile(r'^([A-Z_]+) (REGULAR|ELITE|LEADER)$')
_STATE_RE = re.compile(r'^(.*):[01]$')


def classes_of(kind):
    """The classes an unnamed transfer can name a piece by, narrowest first:
    'card', 'tile'; for army pieces 'unit:<NATION>' (Regular or Elite of one
    nation), 'regular:fp' / 'elite:sa' (that type of any nation of a side),
    'leader:fp' (a Leader of any nation); for an action die itself, then
    'die:fp' / 'die:sh' (a die of that side whatever it shows); else the kind
    itself."""
    if kind.startswith('card#'):
        return ('card',)
    if kind.startswith('die:') and kind.count(':') == 2:
        return (kind, kind[:6])
    if kind.startswith('tile'):
        return ('tile',)
    m = _UNIT_RE.match(kind)
    if m:
        side = 'fp' if m.group(1) in FP_NATIONS else 'sa'
        if m.group(2) == 'LEADER':
            return ('leader:' + side,)
        return ('unit:' + m.group(1), '%s:%s' % (m.group(2).lower(), side))
    return (kind,)


_OWNERS = None


def owner(kind):
    """'fp' / 'sh' for a kind of piece that belongs to one side - action dice, army units and
    Leaders, Nazgul, Companions, Minions, the Fellowship, control markers; None for the rest
    (cards, hunt tiles, track / political / victory markers, Elven Rings)."""
    global _OWNERS
    if kind.startswith('die:'):
        return kind[4:6]
    if kind.split(':')[0] in ('unit', 'regular', 'elite', 'leader'):
        tail = kind.split(':', 1)[1]
        return ('fp' if tail in FP_NATIONS else 'sh') if kind.startswith('unit:') else \
            ('fp' if tail == 'fp' else 'sh')
    m = _UNIT_RE.match(kind)
    if m:
        return 'fp' if m.group(1) in FP_NATIONS else 'sh'
    if _OWNERS is None:
        labels, own = _tables()[0], {'NAZGUL': 'sh', 'Fellowship': 'fp'}
        try:
            import log_replay as lr
            for side, pids in (('fp', list(lr.COMPANIONS) + list(lr.FP_CONTROL_MARKERS)),
                               ('sh', list(lr.MINION_PIECES) + list(lr.SA_CONTROL_MARKERS))):
                own.update((labels[p], side) for p in pids if p in labels)
        except Exception:
            pass
        _OWNERS = own
    m = _STATE_RE.match(kind)
    return _OWNERS.get(m.group(1) if m else kind)


def klass_of(kind):
    """The narrowest class of a kind (see classes_of)."""
    return classes_of(kind)[0]


def kind_words(kind, n=1):
    """A kind in plain words ('2 Gondor Regulars', 'an action die showing C')."""
    if kind.startswith('die:'):
        side = 'Free Peoples' if kind[4:6] == 'fp' else 'Shadow'
        f = kind[7:] or '?'
        w = '%s action %s' % (side, 'die' if n == 1 else 'dice')
        return ('%d %s' % (n, w) if n != 1 else 'a ' + w) + ('' if f == '?' else ' showing %s' % f)
    if kind.startswith('card#') or kind == 'card':
        return 'a card' if n == 1 else '%d cards' % n
    if kind == 'leader:fp':
        return '1 Leader' if n == 1 else '%d Leaders' % n
    if kind in ('regular:fp', 'regular:sa', 'elite:fp', 'elite:sa'):
        w = '%s %s' % ('Free Peoples' if kind.endswith('fp') else 'Shadow', kind[:-3].title())
        return '1 %s' % w if n == 1 else '%d %ss' % (n, w)
    if kind.startswith('unit:'):
        return '%d %s unit%s' % (n, kind[5:].replace('_', ' ').title(), '' if n == 1 else 's')
    if kind.startswith('tile'):
        return 'a hunt tile' if n == 1 else '%d hunt tiles' % n
    if kind.startswith('piece#'):
        return 'piece %s' % kind[6:]
    m = re.match(r'^(.*):([01])$', kind)
    if m:
        if m.group(1) in _SIDES:
            return _SIDES[m.group(1)][int(m.group(2))]
        return 'the %s political marker (%s side up)' % (m.group(1), ('passive', 'active')[int(m.group(2))])
    m = re.match(r'^([A-Z_]+) (REGULAR|ELITE|LEADER)$', kind)
    if m:
        nation = {'THE_NORTH': 'North', 'SOUTHRON_EASTERLINGS': 'Southron & Easterling'}.get(
            m.group(1), m.group(1).title())
        word = '%s %s' % (nation, m.group(2).title())
        return ('1 %s' % word) if n == 1 else '%d %ss' % (n, word)
    if kind == 'NAZGUL':
        return '1 Nazgul' if n == 1 else '%d Nazgul' % n
    name = re.sub(r'\s*\(cls:[^)]*\)', '', kind)
    name = 'a game marker' if name.startswith('cls:') else name
    return name if n == 1 else '%d x %s' % (n, name)


# ---------------------------------------------------------------- zones
_ZONE_CACHE = {}
_ZONE_WORDS = {}
_ZONE_TABLE = []                # the zone ids of the client's table, once every code was read
_ROWS = {}                      # piece id -> the zone ids of the boxes that are one place for it
PLAY_AREA = frozenset(('current action', 'card play'))
DICE_POOLS = frozenset(('fp dice', 'sa dice'))
HANDS = frozenset(('fp hand', 'sa hand'))
_CODE_RE = re.compile(r'^(?:<~)?((?:Game|Controls)\.\d+)(?:~>)?$')


def zone_id(zone):
    """One id per board zone, from a client code or a name."""
    z = _ZONE_CACHE.get(zone)
    if z is not None:
        return z
    raw = str(zone).strip()
    m = _CODE_RE.match(raw)
    name = words = None
    if m:
        code = m.group(1)
        if code == 'Game.2035':
            name = words = 'Current Action'
        else:
            try:
                import log_replay as lr
                name = lr.decode_location(code)
            except Exception:
                name = None
            try:
                import location_codes_authoritative as lca
                words = lca.AUTH_LOCATION_CODES.get(code)
            except Exception:
                words = None
        name = str(name or code)
    else:
        name = raw
    name = ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))
    z = re.sub(r'[^a-z0-9&]+', ' ', name.lower()).strip()
    if z == 'used':
        z = 'current action'
    z, words = layout.slot(z) or (z, words)     # the slots that are one place (layout.SLOTS)
    _ZONE_CACHE[zone] = z
    if words:                   # the client's table names the zone; an engine key never does
        _ZONE_WORDS[z] = str(words)
    elif m:
        _ZONE_WORDS.setdefault(z, name)
    return z


def zone_words(zid):
    """A zone in the words of the client's own table of location codes - whoever named the
    zone first (an action's parameter names a region by the engine's key)."""
    if not _ZONE_TABLE:
        import location_codes_authoritative as lca
        _ZONE_TABLE.extend(zone_id(code) for code in sorted(lca.AUTH_LOCATION_CODES))
    return _ZONE_WORDS.get(zid, zid)


def _row(pid):
    """The zone ids of the boxes off the map that are ONE place for a piece (layout.BOXES)."""
    if pid not in _ROWS:
        _ROWS[pid] = tuple(zone_id(code) for code in layout.boxes(pid))
    return _ROWS[pid]


# the boxes a spent action die lies in, and where a killed die or card arrives
USED_DICE = frozenset(zone_id(code) for code in layout.BOXES['fp die'])
CLOSED_BOXES = frozenset(zone_id(layout.BOXES[f][0]) for f in ('fp die', 'sa die', 'fp card', 'sa card'))
_GENERIC = re.compile(
    r'^(fp|sa) (reinf|dead|dice|hand|char deck|strat deck)$|^(current action|card play|table|'
    r'elven rings|used rings|hunt box|hunt pool|used hunt tiles|removed hunt tiles|spare|'
    r'guide box|fellowship box)$|^(track|political|vp|mordor) \d+$')


def is_map_region(zid):
    """A zone that is a region of the map (or its stronghold box) - not a
    track, box, pool, hand, deck or play area."""
    return not _GENERIC.match(zid)


def region_of(zid):
    """The map region of a zone id: the stronghold box belongs to its region."""
    return zid[:-11] if zid.endswith(' stronghold') else zid


def region_token(name):
    """The id a signature function must use for a map region name."""
    return region_of(zone_id(name))


# ---------------------------------------------------------------- the table
class Table(object):
    """{pid: (zone id, face, state)}; immutable in use (apply returns a new one)."""
    __slots__ = ('p', 'types', '_counts')

    def __init__(self, pieces=None, types=None):
        self.p = dict(pieces or {})
        self.types = types
        self._counts = None

    def __len__(self):
        return len(self.p)

    def __eq__(self, other):
        return isinstance(other, Table) and not delta(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def kind(self, pid):
        z, f, st = self.p[pid]
        return kind_of(pid, f, st, self.types)

    def place(self, pid):
        """(zone id used for comparison, kind) of one piece."""
        z, f, st = self.p[pid]
        k = kind_of(pid, f, st, self.types)
        row = _row(pid)
        if z in row:                    # any box of its row is the same place (layout.BOXES)
            z = row[0]
        if z in PLAY_AREA and k.startswith('tile'):
            z = 'used hunt tiles'
        elif k.startswith('die:') and (z == row[0] or z in PLAY_AREA):
            k = k[:7] + '?'
        return z, k

    def counts(self):
        if self._counts is None:
            c = collections.Counter()
            for pid in self.p:
                c[self.place(pid)] += 1
            self._counts = c
        return self._counts

    def apply(self, plan, with_optional=False):
        """The table after the named changes of a plan (unnamed transfers are
        not applied to pieces: see plan_delta; optional changes only on request)."""
        out = Table(self.p, self.types)
        for ch in plan or ():
            if ch.pid is None or (ch.opt and not with_optional):
                continue
            cur = out.p.get(ch.pid)
            if cur is None:
                cur = ('spare', None, None)
            out.p[ch.pid] = (zone_id(ch.zone) if ch.zone is not None else cur[0],
                             ch.face if ch.face is not None else cur[1],
                             ch.state if ch.state is not None else cur[2])
        return out


# ---------------------------------------------------------------- differences
def delta(a, b):
    """{(zone, kind): count in b minus count in a}, zero entries left out."""
    ca, cb = a.counts(), b.counts()
    out = {}
    for k in set(ca) | set(cb):
        d = cb.get(k, 0) - ca.get(k, 0)
        if d:
            out[k] = d
    return out


def _zones(zone):
    """The zone ids an unnamed transfer names as its source or destination (one, or a choice)."""
    many = isinstance(zone, (tuple, list, set, frozenset))
    return sorted(set(zone_id(z) for z in (zone if many else (zone,))))


def plan_delta(base, plan):
    """The difference a plan makes to `base` -> (delta, coarse, optional delta).

    Named changes move one known piece - from where the plan's own board has it
    (Change.src, when the plan states it: a step before it may have brought the
    piece there as one of a class, which `base` does not show), else from where
    `base` has it.  An unnamed transfer moves n pieces of
    a class from src to dst: it is booked on the class, and `coarse` lists the
    (zone, class) pairs that must then be compared by class (see project).
    The optional delta is what the plan's optional changes add on top."""
    out = {}
    coarse = set()
    pos, was = {}, {}
    for ch in plan or ():
        if ch.opt:
            continue
        if ch.pid is None:
            for zone, sign in ((ch.src, -1), (ch.zone, 1)):
                zs = _zones(zone)
                z = '|'.join(zs)
                if len(zs) > 1:
                    coarse.update((m, ch.klass, z) for m in zs)
                else:
                    coarse.add((z, ch.klass))
                out[(z, ch.klass)] = out.get((z, ch.klass), 0) + sign * ch.n
            continue
        cur = pos.get(ch.pid)
        if cur is None:
            cur = base.p.get(ch.pid) or ('spare', None, None)
            cur = was[ch.pid] = cur if ch.src is None else (zone_id(ch.src),) + cur[1:]
        pos[ch.pid] = (zone_id(ch.zone) if ch.zone is not None else cur[0],
                       ch.face if ch.face is not None else cur[1],
                       ch.state if ch.state is not None else cur[2])
    probe = Table(types=base.types)

    def book(d, pid, old, new):
        if old == new:
            return
        probe.p[pid] = old
        k0 = probe.place(pid)
        probe.p[pid] = new
        k1 = probe.place(pid)
        if k0 != k1:
            d[k0] = d.get(k0, 0) - 1
            d[k1] = d.get(k1, 0) + 1
    for pid, new in pos.items():
        book(out, pid, was[pid], new)
    opt = {}
    pos2 = {}
    for ch in plan or ():
        if not ch.opt:
            continue
        if ch.pid is None:              # an optional unnamed transfer (a consequence: a card drawn)
            for zone, sign in ((ch.src, -1), (ch.zone, 1)):
                z = zone_id(zone)
                coarse.add((z, ch.klass))
                opt[(z, ch.klass)] = opt.get((z, ch.klass), 0) + sign * ch.n
            continue
        cur = pos2.get(ch.pid) or pos.get(ch.pid) or base.p.get(ch.pid) or ('spare', None, None)
        pos2[ch.pid] = (zone_id(ch.zone) if ch.zone is not None else cur[0],
                        ch.face if ch.face is not None else cur[1],
                        ch.state if ch.state is not None else cur[2])
    for pid, new in pos2.items():
        book(opt, pid, pos.get(pid) or base.p.get(pid) or ('spare', None, None), new)
    return ({k: v for k, v in out.items() if v}, frozenset(coarse),
            {k: v for k, v in opt.items() if v})


_ZONE_MERGE = {}


def _coarse_key(z, k, coarse):
    zm = _ZONE_MERGE.get(coarse)
    if zm is None:
        if len(_ZONE_MERGE) > 4096:
            _ZONE_MERGE.clear()
        zm = _ZONE_MERGE[coarse] = {(e[0], e[1]): e[2] for e in coarse if len(e) == 3}
    for c in classes_of(k):
        if (z, c) in coarse or (z, c) in zm:
            k = c
            break
    return zm.get((z, k), z), k


def project(d, coarse):
    """A delta read the way a plan with free choices reads the board: the
    kinds of the `coarse` (zone, class) pairs merged into their class, the
    zones of a (zone, class, merged zone) rule merged into one."""
    if not coarse:
        return d
    out = {}
    for (z, k), v in d.items():
        key = _coarse_key(z, k, coarse)
        out[key] = out.get(key, 0) + v
    return {k: v for k, v in out.items() if v}


def bind(d, coarse, now):
    """A plan delta whose unnamed card / tile transfers are replaced by the
    named pieces the board shows making exactly that trip - so that a card
    drawn and another card discarded in one gesture do not cancel each other
    in the hand.  -> (delta, coarse rules); unchanged where the board does not
    name the pieces."""
    anon = [e for e in coarse if len(e) == 2 and e[1] in ('card', 'tile') and d.get(e)]
    if not anon:
        return d, coarse
    out, drop = dict(d), set()
    for zs, c in [e for e in anon if d[e] < 0]:
        n = -d[(zs, c)]
        for zd, c2 in [e for e in anon if d[e] > 0]:
            if c2 != c or d[(zd, c)] != n or (zd, c) in drop:
                continue
            movers = [k for (z, k), v in now.items()
                      if z == zs and v < 0 and k != c and klass_of(k) == c and now.get((zd, k), 0) > 0]
            if len(movers) == n:
                del out[(zs, c)], out[(zd, c)]
                for k in movers:
                    out[(zs, k)] = out.get((zs, k), 0) - 1
                    out[(zd, k)] = out.get((zd, k), 0) + 1
                drop.update(((zs, c), (zd, c)))
            break
    return {k: v for k, v in out.items() if v}, frozenset(e for e in coarse if e not in drop)


def performed(d, plan):
    """Which pieces realised the free choices of a plan: [(zone, kind, +n / -n)] for every
    transfer that only named a class or a set of zones (the Leader's nation, Regular or
    Elite, the box the Regular left) - the pieces that made exactly that trip in `d`, the board
    difference this step may still claim (what the steps before it took is no longer in it)."""
    out = collections.Counter()
    for ch in sorted(plan or (), key=lambda c: c.opt):
        if ch.pid is not None or ch.klass in ('card', 'tile'):
            continue
        left, src, dst = ch.n, _zones(ch.src), _zones(ch.zone)
        ends = [(z, k, v) for (z, k), v in sorted(d.items()) if (z in src if v < 0 else z in dst)
                and (k == ch.klass or ch.klass in classes_of(k))]
        for z, k, v in [e for e in ends if e[2] < 0]:
            for t in dst:
                n = min(left, out[(z, k)] - v, d.get((t, k), 0) - out[(t, k)])
                if n > 0:
                    left, out[(z, k)], out[(t, k)] = left - n, out[(z, k)] - n, out[(t, k)] + n
        # a trip of which the board shows ONE end (another step of the same answer filled the
        # place it left again, or took the pieces on from where it brought them): it was made
        # by the pieces that show at that end, as many as the other end lacks of the class
        for sign, far in ((-1, dst[0]), (1, src[0])) if not ch.opt else ():
            lack = ch.n - sum(-sign * v for _z, _k, v in ends if v * sign < 0)
            for z, k, v in [e for e in ends if e[2] * sign > 0]:
                n = min(left, lack, sign * (v - out[(z, k)]))
                if n > 0:
                    left, lack = left - n, lack - n
                    out[(z, k)], out[(far, k)] = out[(z, k)] + sign * n, out[(far, k)] - sign * n
    return sorted((z, k, v) for (z, k), v in out.items() if v)


def picks(base, plan):
    """Does a plan leave the player a pick of PIECES?  A required transfer by class does when
    its source holds pieces of several KINDS of the class, more of them than it moves (which
    Leader, which dice); one that takes every such piece, pieces of one kind wherever they lie,
    or one piece that only has two places to go, does not."""
    have = base.counts()
    for ch in (c for c in plan or () if c.pid is None and not c.opt):
        kinds = [(k, v) for (z, k), v in have.items()
                 if z in _zones(ch.src) and (k == ch.klass or ch.klass in classes_of(k))]
        if len(set(k for k, _v in kinds)) > 1 and ch.n < sum(v for _k, v in kinds):
            return True
    return False


def add_delta(d1, d2):
    out = dict(d1)
    for k, v in d2.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def sub_delta(d1, d2):
    out = dict(d1)
    for k, v in d2.items():
        out[k] = out.get(k, 0) - v
    return {k: v for k, v in out.items() if v}


def residue(now, target, coarse=frozenset(), opt=None):
    """What the board shows beyond the target and (any part of) the optional
    changes: {} when it fits."""
    n, t = project(now, coarse), project(target, coarse)
    if n == t:
        return {}
    opt = project(opt, coarse) if opt else {}
    out = {}
    for k, v in sub_delta(n, t).items():
        x = opt.get(k, 0)
        if (v > 0 and not v <= x) or (v < 0 and not x <= v):
            out[k] = v
    return out


def fits(now, target, coarse=frozenset(), opt=None):
    """The board shows the target, with or without (any part of) the plan's
    optional changes."""
    return not residue(now, target, coarse, opt)


def _closable(kind):
    return kind.startswith(('die:', 'card'))


def kills(d):
    """An action die or a card left the play area, or went to a used /
    casualty box: the human closed a die or card window."""
    return any(_closable(k) and ((v < 0 and z in PLAY_AREA) or (v > 0 and z in CLOSED_BOXES))
               for (z, k), v in d.items())


def turned(now, opened, base):
    """The markers whose SIDE was not yet BASE's when the decision opened and
    that the human has turned to it since (the Fellowship turned to its
    revealed side at the end of its placement)."""
    return frozenset(pid for pid, b in base.p.items()
                     if b[2] is not None and pid in opened.p and pid in now.p
                     and opened.p[pid][2] != b[2] and now.p[pid][2] == b[2])


def closes_only(d, out_of=PLAY_AREA):
    """Nothing but action dice and cards going from the play area to a used /
    casualty box: the closing of a die or card window."""
    if not d:
        return False
    return all(_closable(k) and ((v < 0 and z in out_of) or (v > 0 and z in CLOSED_BOXES))
               for (z, k), v in d.items())


def kills_only(d, hands=False):
    """Nothing but action dice and cards killed - out of the play area, or a die
    straight out of its pool (discarded before, or without, its action): a kill
    with no move beside it.  hands: also a card straight out of a hand (played
    and put away in one go) - no piece of the game was moved at all."""
    return closes_only(d, PLAY_AREA | DICE_POOLS | (HANDS if hands else frozenset()))


def inside(part, whole, coarse=frozenset()):
    """Every placement of `part` is present in `whole` (same direction, at
    most as many): `part` can be the first step of a longer gesture."""
    p, w = project(part, coarse), project(whole, coarse)
    if not p:
        return False
    for k, v in p.items():
        if k[0] in PLAY_AREA or (v > 0 and k[0] in DICE_POOLS):
            continue        # what a first step leaves in the play area (or a die it turned
            #                 in the pool) may have moved on with the next step
        x = w.get(k, 0)
        if (v > 0 and not 0 < v <= x) or (v < 0 and not x <= v < 0):
            return False
    return True


def lent(part, whole, coarse=frozenset()):
    """`part` shows on the map in `whole` and is all there but for pieces leaving or entering ONE
    map region: another step of the same answer brought as many back there, or took them on
    (two armies moved one behind the other).  Everything off the map - its die - must be there,
    but the other end of a trip into or out of that region which the other step made back WHOLE
    (the Regular a reduce brought, removed as the next hit: neither end shows)."""
    p, w = project(part, coarse), project(whole, coarse)
    shown, gap, off, miss = False, set(), set(), collections.Counter()
    for k, v in p.items():
        if k[0] in PLAY_AREA or (v > 0 and k[0] in DICE_POOLS):
            continue
        on_map = all(is_map_region(z) for z in k[0].split('|'))    # (a choice of zones: every one of them)
        if (0 < v <= w.get(k, 0)) or (w.get(k, 0) <= v < 0):
            shown = shown or on_map
            continue
        miss[k[1]] += v
        (gap if on_map else off).add(region_of(k[0]) if on_map else k[1])
    return shown and len(gap) == 1 and not any(miss[kind] for kind in off)


def _loose(d):
    """A delta without the side a marker shows (the Fellowship moved but not
    yet flipped is the Fellowship on its way)."""
    out = {}
    for (z, k), v in d.items():
        m = _STATE_RE.match(k)
        key = (z, m.group(1)) if m else (z, k)
        out[key] = out.get(key, 0) + v
    return {k: v for k, v in out.items() if v}


def part_of(part, whole):
    """Every placement of `part` is in `whole` (same direction, at most as
    many), play area included, and `whole` has more: `part` is the smaller
    reading of the same pieces."""
    if not part or part == whole:
        return False
    for k, v in part.items():
        x = whole.get(k, 0)
        if (v > 0 and not v <= x) or (v < 0 and not x <= v):
            return False
    return True


def between(now, target, coarse=frozenset()):
    """`now` lies on the way from the base board to the target: outside the
    play area no zone holds more or fewer pieces of a kind than base and
    target allow, and whatever waits in the play area is a kind the target
    moves.  A marker that is where the target wants it but still shows its
    old side (or was flipped before it is moved) is on the way too.  False
    for the untouched board and for the target itself."""
    n, t = project(now, coarse), project(target, coarse)
    if not n or n == t:
        return False
    if _on_way(n, t) or _walks_on(n, t):
        return True
    # a marker the board moved or turned: on the way only to a target that moves or turns
    # that same marker
    marks = set(_STATE_RE.match(k).group(1) for (z, k) in n if _STATE_RE.match(k))
    if not marks or not marks <= set(_STATE_RE.match(k).group(1) for (z, k) in t if _STATE_RE.match(k)):
        return False
    ln, lt = _loose(n), _loose(t)
    return not ln or ln == lt or _on_way(ln, lt) or _walks_on(ln, lt)


def _walker(kind):
    """A piece that may be walked on within one decision: the Fellowship, a
    Companion, a Minion, a Nazgul - not an army unit, a die, a card or a tile
    (an army moves one region and stops)."""
    return not (_UNIT_RE.match(kind) or _closable(kind) or kind.startswith(('tile', 'unit:', 'regular:',
                                                                            'elite:', 'leader:')))


def _walks_on(n, t):
    """The board has a walker where the target does not want it, the target
    wants one of that kind where the board has none, and with the walker taken
    on to that place the board is the target or on its way: the first region
    of a walked placement."""
    have, need = {}, {}
    for (z, k), v in n.items():
        if v > 0 and _walker(k) and v > max(t.get((z, k), 0), 0):
            have.setdefault(k, []).append([z, v - max(t.get((z, k), 0), 0)])
    for (z, k), v in t.items():
        if v > 0 and k in have and v > max(n.get((z, k), 0), 0):
            need.setdefault(k, []).append([z, v - max(n.get((z, k), 0), 0)])
    if not need:
        return False
    w = dict(n)
    for k, wants in need.items():
        for z2, c2 in wants:
            while c2 and have[k]:
                z1, c1 = have[k][0]
                m = min(c1, c2)
                w[(z1, k)] -= m
                w[(z2, k)] = w.get((z2, k), 0) + m
                c2 -= m
                have[k][0][1] -= m
                if not have[k][0][1]:
                    have[k].pop(0)
    w = {k: v for k, v in w.items() if v}
    return w == t or _on_way(w, t)


def _on_way(n, t):
    leaving = set(k for (z, k), v in t.items() if v < 0)
    leaving |= set(k[:7] + '?' for k in leaving if k.startswith('die:'))    # a staged die has no face
    for (z, k), v in n.items():
        x = t.get((z, k), 0)
        if z in PLAY_AREA:
            if v > x and k not in leaving:
                return False
            continue
        if (v > 0 and v > x) or (v < 0 and v < x):
            return False
    return True


def distance(d1, d2):
    """How many piece placements separate two deltas."""
    return sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in set(d1) | set(d2))


def stages_only(d, cards=False):
    """The delta only takes action dice from a dice pool into the play area
    (a die staged for an action that moves nothing else): not a move of its
    own.  cards: or cards from a hand - the staging pieces of an action."""
    kinds = ('die:', 'card') if cards else ('die:',)
    return bool(d) and all(k.startswith(kinds) and ((v < 0 and z in DICE_POOLS | HANDS) or (v > 0 and z in PLAY_AREA))
                           for (z, k), v in d.items())


def arrived(now, opened, base):
    """NOW with every piece attribute (zone, die face, marker side) that was
    not yet BASE when the decision opened and HAS NOT CHANGED SINCE taken as
    arrived: the previous step's pieces still in transit, a marker the engine
    has already turned.  An attribute the human changed since is kept.
    -> Table (NOW itself when nothing lags)."""
    out = None
    for pid, b in base.p.items():
        o, n = opened.p.get(pid), now.p.get(pid)
        if o is None or n is None or o == b or n == b:
            continue
        new = tuple(b[i] if (n[i] == o[i] and o[i] != b[i]) else n[i] for i in range(3))
        if new != n:
            if out is None:
                out = Table(now.p, now.types)
            out.p[pid] = new
    return out if out is not None else now


def from_board(zone, face=None, mark=None, types=None):
    """Table from reconcile.parse_board's maps (pid -> zone code / die face /
    marker state)."""
    face, mark = face or {}, mark or {}
    return Table({pid: (zone_id(z), face.get(pid), side(pid, mark.get(pid)))
                  for pid, z in zone.items()}, types)


def ops_plan(table, ops):
    """reconcile.moves' ops (('move', side, pid, frm, to) | ('face', side,
    pid, letter) | ('flip', side, pid)) as a plan.  BASE of a decision =
    table.apply(ops_plan(table, moves(difference(state, board))[0])): the
    board as the engine state has it, whatever is still in transit."""
    out = []
    for op in ops or ():
        if op[0] == 'move':
            out.append(move(op[2], op[4]))
        elif op[0] == 'face':
            out.append(face(op[2], op[3]))
        elif op[0] == 'flip':
            cur = table.p.get(str(op[2]))
            if cur is not None and cur[2] in (0, 1):
                out.append(flip(op[2], 1 - cur[2]))
    return out


def touched(d):
    """More of what a delta touches, for the prefilter: (the families of the
    zones that are NOT map regions, dice zones or the play area - 'fp hand',
    'political', 'track', 'sa reinf' (a marker there), 'sa reinf:unit' (an
    army piece there), ... -, the kinds of action dice that left a dice pool)."""
    zones, out = set(), set()
    for (z, k), v in d.items():
        if k.startswith('die:'):
            if v < 0 and z in DICE_POOLS:
                out.add(k)
            continue
        if z in PLAY_AREA or is_map_region(z):
            continue
        unit = _UNIT_RE.match(k) or k == 'NAZGUL' or k.split(':')[0] in ('unit', 'regular', 'elite', 'leader')
        zones.add(re.sub(r' \d+$', '', z) + (':unit' if unit else ''))
    return frozenset(zones), frozenset(out)


def moved_faces(base, table):
    """(the faces shown in `table` by the action dice that were in a dice pool
    in `base` and are outside the pools in `table`, the faces those same dice
    showed in `base`) - two multisets of (is Free Peoples, face).  The board
    model ignores a face outside the pools; the matcher reads these to tell
    readings apart that differ in nothing else (which face an Elven Ring gave
    the die), and to reject a reading the shown faces contradict."""
    shown, was = collections.Counter(), collections.Counter()
    for pid, t in table.p.items():
        if (pid in FP_DICE or pid in SH_DICE) and t[0] not in DICE_POOLS:
            b = base.p.get(pid)
            if b is not None and b[0] in DICE_POOLS:
                shown[(pid in FP_DICE, t[1])] += 1
                was[(pid in FP_DICE, b[1])] += 1
    return shown, was


def observed(d):
    """What a delta touches, for the prefilter: (map regions whose pieces
    changed, kinds of action dice that left or entered a zone)."""
    regions = set()
    dice = set()
    for (z, k) in d:
        if k.startswith('die:'):
            dice.add(k)
        if is_map_region(z) and z not in SHARED_BOXES:
            regions.add(region_of(z))
    return frozenset(regions), frozenset(dice)


# A piece in a stronghold's box stands in that stronghold's region.  A Free Peoples stronghold
# has a box of its own ('<region> stronghold': region_of); the Shadow strongholds share ONE pair
# of boxes, and which stronghold a shared box holds is the engine's knowledge, not the board's.
SHARED_BOXES = frozenset(zone_id(code) for code in layout.SHARED_BOXES.values())


def boxed(d):
    """How many shared stronghold boxes a delta touches: each stands for one region of
    layout.SHADOW_STRONGHOLDS that the board does not name (for the prefilter)."""
    return len({z for (z, _k) in d if z in SHARED_BOXES})


# ---------------------------------------------------------------- plain words
def _moves_in_words(d, limit=6):
    """A delta as 'X from A to B' phrases (greedy pairing of the same kind)."""
    minus = collections.defaultdict(list)
    plus = collections.defaultdict(list)
    for (z, k), v in sorted(d.items()):
        (plus if v > 0 else minus)[k].append([z, abs(v)])
    out = []
    for k in sorted(set(minus) | set(plus)):
        m, p = minus.get(k, []), plus.get(k, [])
        while m and p:
            n = min(m[0][1], p[0][1])
            out.append('%s from %s to %s' % (kind_words(k, n), zone_words(m[0][0]),
                                             zone_words(p[0][0])))
            m[0][1] -= n
            p[0][1] -= n
            if not m[0][1]:
                m.pop(0)
            if not p[0][1]:
                p.pop(0)
        for z, n in m:
            out.append('%s out of %s' % (kind_words(k, n), zone_words(z)))
        for z, n in p:
            out.append('%s into %s' % (kind_words(k, n), zone_words(z)))
    more = len(out) - limit
    return out[:limit] + (['%d more' % more] if more > 0 else [])


def moved_words(d):
    """A delta in plain words: 'X from A to B; Y out of C'."""
    return '; '.join(_moves_in_words(d))


def ops_words(now, ops):
    """The ops of the comparison (reconcile.moves) on the table `now` in plain words: each piece from where the
    op takes it to where it puts it - or turned where it lies (a side, a die's face)."""
    d = collections.Counter()
    for op in [op for op in ops if op[0] == 'move']:
        d[(zone_id(op[3]), kind_of(op[2]))] -= 1
        d[(zone_id(op[4]), kind_of(op[2]))] += 1
    d.update(delta(now, now.apply(ops_plan(now, [op for op in ops if op[0] != 'move']))))
    return moved_words(d)


def sentence(now, target, name=None, coarse=frozenset()):
    """What separates the board from a target, in plain words: what is still
    to move for it, and what was moved that it does not move."""
    n, t = project(now, coarse), project(target, coarse)
    todo = sub_delta(t, n)
    need, extra = {}, {}
    for k, v in todo.items():
        tv = t.get(k, 0)
        # a placement the target wants more of (or less of) = still to do;
        # a placement only the board has = extra
        if (v > 0 and tv > 0) or (v < 0 and tv < 0):
            need[k] = v
        else:
            extra[k] = -v
    parts = []
    what = ('"%s"' % name) if name else 'the nearest legal move'
    if need:
        parts.append('for %s the board still needs: %s' % (what, '; '.join(_moves_in_words(need))))
    if extra:
        parts.append('%s does not move: %s' % (what if not need else 'it',
                                              '; '.join(_moves_in_words(extra))))
    if not parts:
        return 'The board shows %s.' % what
    s = '. '.join(parts)
    return s[0].upper() + s[1:] + '.'
