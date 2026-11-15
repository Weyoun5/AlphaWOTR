# -*- coding: utf-8 -*-
"""layout - the client's table as data: which pieces are one family, and which places count
as ONE place for a family.

A person may put a piece into any place of its row and the board is the same board.  Three
readers, each for its own job, and no rule about a box anywhere else:
  boardmodel   two places of one row are one zone (Table.place, zone_id): the matcher never
               sees which of them a piece lies in
  menu         kill_box: where the client's own kill command sends a piece
  reconcile    the comparison never moves a piece between two places of one row
Layer 0: it knows the client's codes and the engine's piece tables, nothing above them.
"""
import json
import os
import re

FP_REINF, FP_DEAD, SA_REINF, SA_DEAD = 'Game.2038', 'Game.2039', 'Game.2040', 'Game.2041'
TILE_USED = 'Game.2045'
FP_DICE = frozenset(('210', '211', '212', '213', '358', '359'))
SH_DICE = frozenset(('099', '100', '101', '102', '103', '104', '105', '360', '361', '362'))
ELVEN_RINGS = frozenset(('249', '250', '251'))
FP_NATIONS = frozenset(('GONDOR', 'ROHAN', 'ELVES', 'DWARVES', 'THE_NORTH'))

# THE BOXES OFF THE MAP, one row per family of pieces: the boxes that are ONE place for it.
# The first is where the client's own kill sends such a piece, and where a plan shows it.
BOXES = {
    'fp unit': (FP_DEAD,),          # a dead Regular / Elite is counted by the rules: the stock
    #                                 (FP_REINF) is another place, and the only one a recruit leaves
    'fp leader': (FP_DEAD, FP_REINF),
    'companion': (FP_DEAD, FP_REINF, SA_DEAD),
    'sa unit': (SA_REINF, SA_DEAD),     # Shadow Regulars, Elites and Nazgul come back into play
    'minion': (SA_DEAD, SA_REINF),
    'fp marker': (FP_REINF,), 'sa marker': (SA_REINF,),
    'fp die': (FP_DEAD, SA_DEAD, FP_REINF, SA_REINF),   # a spent die is out until the next roll
    'sa die': (SA_DEAD, FP_DEAD, FP_REINF, SA_REINF),
    'fp card': (FP_DEAD,), 'sa card': (SA_DEAD,), 'ring': (SA_DEAD,), 'tile': (TILE_USED,),
}
# the families a player himself places, moves and removes on the map - his figures; all else on the
# table is a marker, a die, a card or a tile
FIGURES = frozenset(('fp unit', 'fp leader', 'companion', 'sa unit', 'minion'))
# THE BOXES OF THE STRONGHOLDS: a Free Peoples stronghold has a box of its own; the Shadow
# strongholds share one pair, and a pair that holds ONE garrison is one place.  Which stronghold
# a shared box holds is the engine's knowledge, so this row is applied where the state is known:
# by the plans (menu: a piece that goes into the pair may go into either box) and by the
# comparison (reconcile: either box is right) - not by the board model
SHARED_BOXES = {'shadow stronghold': 'Game.1996', 'shadow stronghold 2': 'Game.2028'}
SHADOW_STRONGHOLDS = frozenset(('orthanc', 'moria', 'dol guldur', 'mount gundabad', 'morannon',
                                'minas morgul', 'barad dur', 'umbar'))
# THE SLOTS: zones (by the client's own name, lower case) that are one place for every piece -
# a card or a ring is one of a kind, the slot it lies in does not change which one it is.
# (pattern, the one zone, its words)
SLOTS = ((r'(fp|sa) card play', 'card play', 'the Current Card slots'),
         (r'board ?\d+', 'table', 'the table card slots'),
         (r'(vilya|nenya|narya) slot|elven ring \d', 'elven rings', 'the Elven Ring slots'),
         (r'(used|dark) ring \d', 'used rings', 'the used-ring slots'))

# A board dump's own names: the client's Current Action slot is where a die WAITS (staged), not
# the corpus parser's 'Used' (there a kill in Current Action means spent)
LOCATION_CODES = {'Game.2035': 'Current Action'}
# the name each side of a two-sided Companion piece shows
COMPANION_FACES = {'240': {0: 'Gandalf the Grey', 1: 'Gandalf the White', 2: 'Gandalf the White'},
                   '241': {0: 'Strider', 1: 'Aragorn'}}

_HOME, _INITIAL, _FAMILY = {}, {}, {}


def side_tag(side):
    """'fp' / 'sa': how the client's table names a side (a Side, or its name)."""
    return 'fp' if getattr(side, 'name', side) == 'FREE_PEOPLES' else 'sa'


def slot(name):
    """(the one zone, its words) of a zone that is a slot of a row of SLOTS, else None."""
    return next((row[1:] for row in SLOTS if re.fullmatch(row[0], name)), None)


def home(pid):
    """The code of the zone a piece starts the game in (a card: its deck) - the client's fixed
    set-up, which no log line and no dump of a later board states."""
    if not _HOME:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, 'initial_positions.json')) as f:
            _HOME.update(json.load(f))
    return _HOME.get(str(pid))


def initial():
    """{piece id: the decoded name of the zone it starts in}."""
    if not _INITIAL:
        import log_replay as lr
        home(None)
        _INITIAL.update((p, lr.decode_location(c)) for p, c in _HOME.items())
    return _INITIAL


def family(pid, types=None):
    """The family of a piece (a key of BOXES); None for a piece no row knows.  types: the
    client's run-time re-typing of unit pieces, when the caller has it."""
    pid = str(pid)
    if types is None and pid in _FAMILY:
        return _FAMILY[pid]
    import log_replay as lr
    deck = str(initial().get(pid) or '')
    if pid in FP_DICE or pid in SH_DICE:
        out = 'fp die' if pid in FP_DICE else 'sa die'
    elif pid in lr.COMPANIONS or pid in lr.MINION_PIECES:
        out = 'companion' if pid in lr.COMPANIONS else 'minion'
    elif pid in lr.HUNT_TILE_PIECES or pid in ELVEN_RINGS:
        out = 'tile' if pid in lr.HUNT_TILE_PIECES else 'ring'
    elif pid in lr.FP_CONTROL_MARKERS or pid in lr.SA_CONTROL_MARKERS:
        out = 'fp marker' if pid in lr.FP_CONTROL_MARKERS else 'sa marker'
    elif deck.endswith('Deck'):
        out = 'fp card' if deck.startswith('FP') else 'sa card'
    else:
        nation, kind = lr.get_unit_type_from_piece_id(pid, types)
        kind = str(kind).rsplit('.', 1)[-1].lower()         # (a UnitKind, or the word 'NAZGUL')
        fp = nation is not None and nation.name in FP_NATIONS
        out = ('sa unit' if kind == 'nazgul' else None if kind not in ('regular', 'elite', 'leader')
               or nation is None else 'fp leader' if fp and kind == 'leader' else
               'fp unit' if fp else 'sa unit')
    if types is None:
        _FAMILY[pid] = out
    return out


def side_of(pid):
    """'fp' / 'sa': the side this piece belongs to on the table (the Hunt tiles are the Shadow's,
    the Elven Rings the Free Peoples'); None for a marker of a track, of politics or the score."""
    row = family(pid)
    return None if row is None else 'fp' if row.startswith('fp') or row in ('companion', 'ring') else 'sa'


def boxes(pid, types=None):
    """The boxes off the map that are one place for this piece (the kill box first)."""
    return BOXES.get(family(pid, types), (FP_DEAD,))


def kill_box(pid, types=None):
    """Where the client's own kill command sends a piece."""
    return boxes(pid, types)[0]


def one_garrison(regions):
    """Do the shared boxes hold at most one garrison?  regions: the strongholds under siege and
    the regions of an open battle, by name."""
    return len(SHADOW_STRONGHOLDS.intersection(regions)) <= 1


def shared_boxes(region, bound=None):
    """The shared boxes that may hold the garrison of a Shadow stronghold (the first: where a
    plan shows it): the one that `bound` ({piece: (region, box)} of the pieces lying in the
    shared boxes) ties to it, then every shared box tied to no stronghold."""
    used = {code: reg for reg, code in (bound or {}).values()}
    codes = tuple(SHARED_BOXES.values())
    return tuple(c for c in codes if used.get(c) == region) + tuple(c for c in codes if c not in used)


def unit_type(pid, dynamic_types=None, side_hint=None):
    """(nation, kind) of a client piece, the kind in lower case: 'regular', 'elite', 'leader',
    'nazgul', or 'companion' / 'minion' with the figure's name for the nation.  dynamic_types:
    the client's run-time re-typing of unit pieces, when the caller has it."""
    import log_replay as lr
    pid = str(pid)
    if pid in lr.COMPANIONS:
        return lr.COMPANIONS[pid], 'companion'
    if pid in lr.MINION_PIECES:
        return lr.MINION_PIECES[pid], 'minion'
    nation, kind = lr.get_unit_type_from_piece_id(pid, dynamic_types)
    if kind is None and side_hint:
        nation, kind = lr.get_unit_type_from_piece_id(pid, dynamic_types, side_hint=side_hint)
    return nation, (str(getattr(kind, 'value', kind)).lower() if kind is not None else None)


def bind_shared_boxes(state, zone, dynamic_types=None):
    """A board's {piece: zone code} -> ({piece: the name of its zone}, {piece: (region, shared
    box code)}).  A piece in a shared Shadow stronghold box is named '<region> stronghold': the
    box belongs to the besieged Shadow stronghold whose engine garrison its unit pieces match
    best; with no besieged Shadow stronghold the pieces keep the box's own name."""
    import log_replay as lr
    from board import sanitize
    names = {pid: (LOCATION_CODES.get(code) or lr.decode_location(code) or code) for pid, code in zone.items()}
    slots = {}
    for pid, code in zone.items():
        if code in SHARED_BOXES.values():
            slots.setdefault(code, []).append(pid)
    if not slots or state is None:
        return names, {}
    sieged = sorted(sanitize(str(b)) for b in (getattr(state, 'besieged_strongholds', None) or ())
                    if sanitize(str(b)) in SHADOW_STRONGHOLDS)
    armies = {sanitize(str(k)): a for k, a in state.armies.items()}
    low = lambda x: str(getattr(x, 'value', x)).lower()

    def score(region, pids):
        have = {}
        for u in getattr(armies.get(region), 'units', ()) or ():
            key = (low(u.nation), low(u.kind))
            have[key] = have.get(key, 0) + 1
        hit = 0
        for pid in pids:
            nation, kind = unit_type(pid, dynamic_types)
            if have.get((low(nation), kind), 0) > 0:
                have[(low(nation), kind)] -= 1
                hit += 1
        return hit

    bound, taken = {}, set()
    for code in sorted(slots, key=lambda c: -len(slots[c])):
        free = [r for r in sieged if r not in taken]
        if not free:
            continue
        region = max(free, key=lambda r: score(r, slots[code]))
        taken.add(region)
        for pid in slots[code]:
            names[pid] = region + ' stronghold'
            bound[pid] = (region, code)
    return names, bound


def political_marker_active(pid, marks):
    """Does a political marker show its active side?  None when the dump does not state a
    side for it (the four nations that start at war have no passive side)."""
    if str(pid) in ('256', '261', '262', '263'):
        return True
    face = (marks or {}).get(str(pid))
    return bool(face) if face in (0, 1) else None
