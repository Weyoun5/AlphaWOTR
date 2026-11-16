"""RECONCILE - make the client board equal the engine (engine -> client).

    board  = parse_board(the dump's text)     what the client shows
    rows   = difference(state, board, ...)    where it is not the engine
    ops    = moves(rows)                      THE generic step: rows -> piece moves
    lines  = render(ops, nick_of)             client lines

Invariants:
  * pure and idempotent: everything is computed from (engine state, board);
    a board that equals the engine gives no rows and no lines;
  * a row is one class of interchangeable pieces: `want` = how many the
    engine puts in each zone, `have` = where the client pieces of that class
    are, `pool` = the off-board boxes that hold the rest (first = where a
    piece leaving the board goes, and where a new one is taken from);
  * a difference with no piece to show it stays a row: the caller counts
    tries and says one sentence (words.could_not_show), never loops.

`settled` (no decision open, no two-step die under way): only then dice, played
cards and drawn tiles are tidied out of the display slots; until then they stay.
"""
from collections import Counter, namedtuple

from . import emitter as em
from . import layout
from .layout import FP_DEAD, FP_REINF, SA_DEAD, SA_REINF, TILE_USED

MIN_DUMP_PIECES = 400       # a whole dump holds about 428 pieces: fewer is one torn in the writing
Board = namedtuple('Board', 'zone face mark')       # pid -> code / die letter / marker state
# want: Counter{code: n}; have: {pid: code}; pool: [code]; flip / face: {pid: wanted}
Row = namedtuple('Row', 'text side want have pool flip face')

FP_DICE, SA_DICE = em.FP_DICE_POOL, em.SH_DICE_POOL
HUNT_BOX, STAGED = em.HUNT_BOX, em.DICE_USED
TILE_POOL, TILE_REMOVED = 'Game.2044', 'Game.2046'
FP_HAND, SA_HAND = 'Controls.45', 'Controls.46'
TABLE = tuple('Game.%d' % n for n in range(2029, 2035))
DISPLAY = (STAGED, 'Game.2036', 'Game.2037')         # Current Action / SA card / FP card
DECK = {'FREE_PEOPLES': {'character': 'Game.2004', 'strategy': 'Game.2005'},
        'SHADOW': {'character': 'Game.2006', 'strategy': 'Game.2007'}}
FACE_LETTER = em._FACE_LETTER
RINGS = ('249', '250', '251')
# a Minion with a "used" side shows it (side 1) while this fact of the engine state holds
MINION_USED = {'255': 'sh_mouth_muster_used_this_turn'}


def settled(state):
    """Is no decision open in this state and no two-step die under way?  (THE test of when the
    comparison looks at dice, played cards and drawn tiles.)"""
    import ai
    return (state is not None and ai._pending_decision_kind_side(state)[0] is None
            and getattr(state, 'pending_army_move_die', None) is None
            and getattr(state, 'pending_muster_die', None) is None)


def looks(pid, settled):
    """Does a pass of difference() look at this piece?  At an action die, an event card and a
    Hunt tile only when the game is settled."""
    return settled or not str(layout.family(pid)).endswith((' die', ' card', 'tile'))


def parse_board(text):
    """The text of a dump: Board(pid -> zone code, die face letter, marker
    state), or None while it is missing or torn."""
    zone, face, mark = {}, {}, {}
    for line in (text or '').split('\n'):
        if line.startswith('#'):
            continue
        parts = line.rstrip('\r').split('\t')
        if len(parts) < 3:
            continue
        code = parts[1].strip().strip('<~>')
        pids = parts[2].split()
        marks = parts[4].split() if len(parts) > 4 else []
        faces = parts[5].split() if len(parts) > 5 else []
        for i, pid in enumerate(pids):
            zone[pid] = code
            if i < len(marks) and marks[i].lstrip('-').isdigit():
                mark[pid] = int(marks[i])
            if i < len(faces) and faces[i] != '-':
                face[pid] = faces[i]
    return Board(zone, face, mark) if len(zone) >= MIN_DUMP_PIECES else None


def pool_moves(pool, used, board):
    """The piece moves that make the client's Hunt Pool hold what the
    engine's pool holds at a draw (`pool`, `used`: tile classes), touching
    nothing but the pool: a missing tile enters it from the used box, else
    from the removed box; a tile too many leaves it to the used box when the
    engine has one there, else to the removed box.  A tile on display stays."""
    import log_replay as lr
    from .randomness import tile_class
    want, spare = Counter(pool), Counter(used)
    pieces = {}
    for pid, name in lr.HUNT_TILE_PIECES.items():
        if pid in board.zone:
            pieces.setdefault(tile_class(name), []).append(pid)
    ops = []
    for cls in sorted(set(want) | set(pieces)):
        pids = sorted(pieces.get(cls, ()))
        inside = [p for p in pids if board.zone[p] == TILE_POOL]
        for pid in inside[want[cls]:]:
            spare[cls] -= 1
            ops.append(('move', 'sa', pid, TILE_POOL, TILE_USED if spare[cls] >= 0 else TILE_REMOVED))
        outside = [p for p in pids if board.zone[p] == TILE_USED] \
            + [p for p in pids if board.zone[p] == TILE_REMOVED]
        for pid in outside[:max(0, want[cls] - len(inside))]:
            ops.append(('move', 'sa', pid, board.zone[pid], TILE_POOL))
    return ops


def _key(x):
    from board import sanitize
    return sanitize(str(x))


def _code(name):
    return em.zone_code_for(_key(name))


def _nv(x):
    return str(getattr(x, 'value', x)).lower()


def _side(nation):
    return 'fp' if em._side_of_nation(nation) == 'fp' else 'sa'


def bind_dice(state, board, last=()):
    """{pid: (Side, engine die index)}: every engine die gets the client die
    that shows it best (same box and face first), so the binding is a function of the two
    states, never of the roll history.  last: the dice that are a person's - {piece: where he
    took it from}, or the pieces.  The game has not followed his move: such a die stands for
    the engine die that is where he took it FROM, and for any other only where no other piece does."""
    from game_state import Side
    out = {}
    hunt = [id(d) for d in state.hunt_box]
    for side, family, pool, dead, reinf in (
            (Side.FREE_PEOPLES, em._FP_DIE_IDS, FP_DICE, FP_DEAD, FP_REINF),
            (Side.SHADOW, em._SH_DIE_IDS, SA_DICE, SA_DEAD, SA_REINF)):
        where = {}

        def at(z, pool=pool, dead=dead):
            return ('pool' if z == pool else 'hunt' if z == HUNT_BOX
                    else 'used' if z == dead else 'staged' if z in DISPLAY else 'out')
        for pid in family:
            # (a piece he brought in from outside - the spare a minion earns - stands where it LIES)
            came = at(last.get(pid)) if isinstance(last, dict) and last.get(pid) else 'out'
            where[pid] = came if came != 'out' else at(board.zone.get(pid))
        free = sorted(family, key=lambda p: p in last)
        engine = []
        for i, d in enumerate(state.dice.get(side, [])):
            cat = 'hunt' if id(d) in hunt else 'used' if d.used else 'pool'
            engine.append((i, cat, FACE_LETTER.get(getattr(d.current, 'name', None))))
        todo = list(engine)
        # the die of a two-step action under way is the one staged for it
        for attr in ('pending_army_move', 'pending_muster'):
            i = getattr(state, attr + '_die', None)
            if getattr(state, attr + '_side', None) is side and isinstance(i, int)                     and 0 <= i < len(engine):
                pid = next((p for p in free if where[p] == 'staged'
                            and engine[i][2] in (None, board.face.get(p))), None)
                if pid is not None:
                    free.remove(pid)
                    out[pid] = (side, i)
                    todo.remove(engine[i])
        tests = (
            # a spent die first takes a spent piece that shows ITS face: taken by box alone, an
            # older spent die of another face would leave the die he has just staged unspent,
            # and the comparison would finish a die of his pool in its place
            lambda p, cat, f: cat != 'pool' and where[p] == cat and f is not None and f == board.face.get(p),
            lambda p, cat, f: cat != 'pool' and where[p] == cat,
            # a die in the play area is the die just spent, else the NEXT die of its face to be
            # spent: the first one of that face in the engine's pool (the die an option names)
            lambda p, cat, f: cat == 'used' and where[p] == 'staged' and f in (None, board.face.get(p)),
            lambda p, cat, f: cat == 'pool' and where[p] == 'staged' and f and f == board.face.get(p),
            lambda p, cat, f: where[p] == cat and f in (None, board.face.get(p)),
            lambda p, cat, f: where[p] == cat,
            lambda p, cat, f: where[p] == 'staged' and f in (None, board.face.get(p)),
            lambda p, cat, f: where[p] == 'staged',
            lambda p, cat, f: where[p] != 'out',
            lambda p, cat, f: True)
        for test in tests:
            rest = []
            for i, cat, f in todo:
                pid = next((p for p in free if test(p, cat, f)), None)
                if pid is None:
                    rest.append((i, cat, f))
                else:
                    free.remove(pid)
                    out[pid] = (side, i)
            todo = rest
    return out


def rolled(board, tag, n):
    """The faces an action roll of `n` dice of a side shows on the board, sorted: its dice in
    the pool and, for what is missing of n, the Eyes a Shadow roll sends on to the Hunt Box.
    None while no board is read."""
    if board is None:
        return None
    pool, family = (FP_DICE, layout.FP_DICE) if tag == 'fp' else (SA_DICE, layout.SH_DICE)
    faces = [str(board.face.get(p)) for p in family if board.zone.get(p) == pool]
    return sorted(faces + ['E'] * (n - len(faces) if tag == 'sa' else 0))


def _off_board(tag, have, dead):
    """The boxes that are one place for Leaders (fp) and for Shadow army pieces, the one a piece
    that leaves the board goes to first: a Free Peoples Leader to the casualties while the
    game counts more dead Leaders of its nation than lie there, else to the reinforcements."""
    pool = list(layout.BOXES['fp leader' if tag == 'fp' else 'sa unit'])
    return pool[::-1] if tag == 'fp' and sum(z == FP_DEAD for z in have.values()) >= dead else pool


def _army_rows(state, board, bound, rows):
    """Units, Leaders, Nazgul, Minions, separated Companions: region by
    region, inside the stronghold box for the garrison of a siege."""
    from combat import garrison_side_for
    from army_card_cargo import key as category
    import log_replay as lr
    want = {}
    named = {}
    sieged = {_key(b) for b in (getattr(state, 'besieged_strongholds', None) or ())}
    for key, army in state.armies.items():
        region = _key(key)
        try:
            _code(region)
        except LookupError:
            continue
        garrison = garrison_side_for(state, region) if region in sieged else None
        # (a relief battle outside the walls: of the garrison's side only the garrison is inside)
        stays = em.relief_garrison(state, region, garrison) if garrison is not None else None

        def zone(side, figure, unit=False):     # the field, or the stronghold's box for the garrison's side
            side = 'sh' if side == 'sa' else side
            if stays is not None and em.canon_side(side) == em.canon_side(garrison):
                stays[category(figure, unit)] -= 1
                if stays[category(figure, unit)] < 0:
                    return em.arrival_zone(region, side, None, bound)
            return em.arrival_zone(region, side, garrison, bound)
        for u in army.units:
            side = _side(u.nation)
            cls = ('unit', _nv(u.nation), _nv(u.kind), side)
            want.setdefault(cls, Counter())[zone(side, u, True)] += 1
        for ld in army.leaders:
            kind = type(ld).__name__
            if kind == 'Nazgul':
                cls = ('nazgul', '', 'nazgul', 'sa')
            elif kind == 'Leader':
                cls = ('leader', _nv(ld.nation), 'leader', _side(ld.nation))
            else:
                named[str(ld.name)] = zone('fp' if kind == 'Companion' else 'sa', ld)
                continue
            want.setdefault(cls, Counter())[zone(cls[3], ld)] += 1
    # a stronghold an open battle involves: the withdrawal into its box is
    # shown before the engine forms the siege, so both zones count there
    battle = getattr(state, 'pending_combat', None) or {}
    flux = {}
    fought = {_key(battle[k]) for k in ('defender_region', 'attacker_region', 'region', 'trim_region')
              if isinstance(battle, dict) and battle.get(k)}
    shadow = (sieged | fought) & layout.SHADOW_STRONGHOLDS
    for region in sorted(fought - sieged):
        try:                            # (a battle region without a stronghold has no box)
            boxes = layout.shared_boxes(region, bound) if region in shadow else (
                em.arrival_zone(region, 'fp', 'fp', bound),)
            flux.update((box, _code(region)) for box in boxes)
        except LookupError:
            pass
    for region in sorted(shadow & sieged if layout.one_garrison(sieged | fought) else ()):
        # the two shared boxes are one place while they hold ONE garrison (layout.SHARED_BOXES)
        boxes = layout.shared_boxes(region, bound)
        flux.update((box, boxes[0]) for box in boxes[1:])
    have = {}
    for pid, code in board.zone.items():
        if pid in lr.COMPANIONS or pid in lr.MINION_PIECES:
            continue
        try:
            nat, kind = layout.unit_type(pid)
        except Exception:
            continue
        if kind in ('regular', 'elite'):
            cls = ('unit', _nv(nat), kind, _side(nat))
        elif kind == 'leader':
            cls = ('leader', _nv(nat), 'leader', _side(nat))
        elif kind == 'nazgul':
            cls = ('nazgul', '', 'nazgul', 'sa')
        else:
            continue
        have.setdefault(cls, {})[pid] = code
    dead = Counter()
    for nation, kinds in (state.fp_casualties or {}).items():
        for kind, n in kinds.items():
            if kind in ('regular', 'elite', 'leader') and n:
                dead[('leader' if kind == 'leader' else 'unit', _nv(nation), kind, 'fp')] += n
    for cls in sorted(set(want) | set(have)):
        w = Counter(want.get(cls, ()))
        for inside, outside in flux.items():
            n = min(w[outside], sum(1 for z in have.get(cls, {}).values() if z == inside))
            if n:
                w[outside] -= n
                w[inside] += n
        if cls[0] == 'unit' and cls[3] == 'fp':
            w[FP_DEAD] += dead[cls]     # (the rules count them: only the stock is a pool)
            pool = [FP_REINF]
        else:                           # the boxes that are one place for the class (layout.BOXES)
            pool = _off_board(cls[3], have.get(cls, {}), dead[cls])
        rows.append(Row('%s %s' % (cls[1], cls[2]), cls[3], w, have.get(cls, {}), pool, {}, {}))
    # named figures: one piece each
    fs = state.fellowship
    guide = getattr(getattr(fs, 'guide', None), 'name', None)
    inside_fs = {str(c.name) for c in fs.companions}
    if guide:
        inside_fs.add(str(guide))
    for pid, name in sorted(lr.COMPANIONS.items()):
        if pid not in board.zone:
            continue
        names = set(lr.names_for_pid(pid) or (name,))
        w, pool = Counter(), list(layout.BOXES['companion'])
        if names & inside_fs:
            w[em.GUIDE_BOX if guide in names else em.FELLOWSHIP_BOX] = 1
        elif names & set(named):
            at = next(named[n] for n in names if n in named)
            w[board.zone[pid] if flux.get(board.zone[pid]) == at else at] = 1
        elif board.zone[pid] in pool[1:] or (
                board.zone[pid] in DISPLAY and state.pending_hunt is not None):
            continue                    # not in play, or the random draw is still shown
        flip = {}
        face = layout.COMPANION_FACES.get(pid, {}).get(board.mark.get(pid))
        held = (names & inside_fs) | (names & set(named))
        if face is not None and held and face not in held:
            flip[pid] = next(iter(held))
        rows.append(Row('companion %s' % name, 'fp', w, {pid: board.zone[pid]}, pool, flip, {}))
    for pid, name in sorted(lr.MINION_PIECES.items()):
        if pid not in board.zone:
            continue
        word = name.lower().split()[0]
        at = next((code for n, code in named.items() if word in n.lower()), None)
        if at and flux.get(board.zone[pid]) == at:
            at = board.zone[pid]
        w = Counter({at: 1}) if at else Counter()
        flip = {}
        used = MINION_USED.get(pid)
        if used and board.mark.get(pid) in (0, 1) and (board.mark[pid] == 1) != bool(getattr(state, used)):
            flip[pid] = 'used' if getattr(state, used) else 'ready'
        rows.append(Row('minion %s' % name, 'sa', w, {pid: board.zone[pid]},
                        list(layout.BOXES['minion']), flip, {}))


def _marker_rows(state, board, rows):
    """Fellowship pawn and tracks, political markers, control and victory
    markers, Elven Rings."""
    import board as bd
    import game_logic as gl
    import log_replay as lr
    from game_state import Side
    fs = state.fellowship
    mordor = bool(getattr(fs, 'mordor_entered', False))
    if '247' in board.zone:
        if mordor:
            want = em.MORDOR_TRACK_ZONES[max(0, min(5, int(fs.mordor_step or 0)))]
        else:
            want = _code(gl.fellowship_region(state))
            gate = str(gl.fellowship_region(state)).lower().replace(' ', '_')
            if gate in ('morannon', 'minas_morgul') and board.zone['247'] == em.MORDOR_TRACK_ZONES[0]:
                want = board.zone['247']    # declared straight onto the Mordor Track: it may stay
        flip = {}
        if board.mark.get('247') in (0, 1) and (board.mark['247'] == 0) != bool(fs.hidden):
            flip['247'] = 'hidden' if fs.hidden else 'revealed'
        rows.append(Row('the Fellowship', 'fp', Counter({want: 1}),
                        {'247': board.zone['247']}, [], flip, {}))
    # (the client itself sets the progress to 0 when the figure is turned up, the game when the
    # revealed Fellowship is placed: until then the marker on step 0 is right as it lies)
    reset = mordor or (not fs.hidden and board.zone.get('248') == 'Game.2014')
    for pid, n, text in (('248', 0 if reset else int(fs.progress or 0), 'Fellowship progress'),
                         ('252', int(fs.corruption or 0), 'Corruption')):
        if pid in board.zone:
            rows.append(Row(text, 'fp', Counter({'Game.%d' % (2014 + max(0, min(12, n))): 1}),
                            {pid: board.zone[pid]}, [], {}, {}))
    pol = state.political
    for pid, nation in lr.POLITICAL_MARKERS.items():
        if pid not in board.zone or pol.boxes.get(nation) is None:
            continue
        flip = {}
        shown = layout.political_marker_active(pid, board.mark)
        if shown is not None and shown != bool(pol.active.get(nation, False)):
            flip[pid] = 'active' if pol.active.get(nation) else 'passive'
        rows.append(Row('politics of %s' % _nv(nation), _side(nation),
                        Counter({_code('political %d' % (3 - int(pol.boxes[nation]))): 1}),
                        {pid: board.zone[pid]}, [], flip, {}))
    inner = {}
    for region, info in bd.regions.items():
        if info.get('is_stronghold'):
            try:
                inner[_code(region + ' stronghold')] = _code(region)
            except LookupError:
                pass
    for side, tag, pids in ((Side.SHADOW, 'sa', range(411, 421)), (Side.FREE_PEOPLES, 'fp', range(405, 411))):
        pids = ['%03d' % p for p in pids if '%03d' % p in board.zone]
        have = {p: inner.get(board.zone[p], board.zone[p]) for p in pids}
        # more regions held than markers: the regions that have one keep it, then by name
        held = sorted((_code(r) for r in sorted(_key(k) for k in state.enemy_controlled.get(side, ()))),
                      key=lambda code: code not in have.values())
        rows.append(Row('%s control markers' % ('Shadow' if tag == 'sa' else 'Free Peoples'), tag,
                        Counter(held[:len(pids)]), have, list(layout.BOXES[tag + ' marker']), {}, {}))
    vc = gl.check_all_victory_conditions(state)
    for pid, key, side in (('422', 'free_peoples_vp', 'fp'), ('423', 'shadow_vp', 'sa')):
        if pid not in board.zone or vc.get(key) is None:
            continue
        try:
            want = _code('vp %d' % int(vc[key]))
        except LookupError:
            continue
        rows.append(Row('victory points', side, Counter({want: 1}),
                        {pid: board.zone[pid]}, [], {}, {}))
    # Elven Rings: which ring is irrelevant, only how many are where - available, the Shadow's,
    # or out of the game in the ring's box (a ring that lies anywhere else is in none of the three)
    rings = [p for p in RINGS if p in board.zone]
    need = (['fp'] * int(getattr(state, 'fp_elven_rings', 3))
            + ['sa'] * int(getattr(state, 'sh_elven_rings', 0)))
    need += ['out'] * (len(rings) - len(need))
    gone = layout.BOXES['ring']

    def where(pid):
        z = board.zone[pid]
        return 'fp' if z in em.RING_ACTIVE_SLOTS else 'sa' if z in em.RING_USED_SLOTS else \
            'out' if z in gone else None
    left = list(rings)
    for cat in list(need):
        pid = next((p for p in left if where(p) == cat), None)
        if pid is not None:
            left.remove(pid)
            need.remove(cat)
    taken = {board.zone[p] for p in rings if p not in left}
    for pid, cat in zip(left, need):
        slots = em.RING_ACTIVE_SLOTS if cat == 'fp' else em.RING_USED_SLOTS if cat == 'sa' else gone
        to = next((s for s in slots if s not in taken or s in gone), slots[0])
        taken.add(to)
        rows.append(Row('Elven Rings', 'fp', Counter({to: 1}), {pid: board.zone[pid]}, [], {}, {}))


def _dice_rows(state, board, dice, rows):
    """Action dice: pool (with its face), Hunt Box, used; a die the engine
    does not have goes back to its reserve."""
    from game_state import Side
    hunt = [id(d) for d in state.hunt_box]
    for side, tag, family, pool, dead, reinf in (
            (Side.FREE_PEOPLES, 'fp', em._FP_DIE_IDS, FP_DICE, FP_DEAD, FP_REINF),
            (Side.SHADOW, 'sa', em._SH_DIE_IDS, SA_DICE, SA_DEAD, SA_REINF)):
        bound = {pid: idx for pid, (s, idx) in dice.items() if s is side}
        text = '%s action dice' % ('Free Peoples' if tag == 'fp' else 'Shadow')
        on_board = {idx for pid, idx in bound.items() if pid in board.zone}
        for idx in range(len(state.dice.get(side, []))):
            if idx not in on_board:     # a die of the game that no die piece stands for: a row
                rows.append(Row(text, tag, Counter({pool: 1}), {}, [], {}, {}))     # nothing can show
        for pid in family:
            if pid not in board.zone:
                continue
            face = {}
            if pid in bound:
                d = state.dice[side][bound[pid]]
                # (a spent die lies in any box of its row: layout.BOXES)
                out = board.zone[pid] if board.zone[pid] in layout.boxes(pid) else dead
                want = HUNT_BOX if id(d) in hunt else out if d.used else pool
                letter = FACE_LETTER.get(getattr(d.current, 'name', None))
                # (a pool die that shows no face - a game started from a saved position
                # never rolled - is turned to the engine's face like one that shows another)
                if want == pool and letter and board.face.get(pid) != letter:
                    face[pid] = letter
            elif board.zone[pid] in (pool, dead, HUNT_BOX) + DISPLAY:
                want = reinf
            else:
                continue
            rows.append(Row(text, tag, Counter({want: 1}), {pid: board.zone[pid]}, [], {}, face))


def dice_moves(state, board):
    """The piece moves that make the action dice on the table the game's at a roll, touching nothing but
    dice: a roll takes EVERY die that lies in a pool, so each die of the game is put in its place and a die
    piece that stands for none goes back to its reserve.  Each piece counts where it lies (no move of a
    person is waited for at a roll)."""
    rows = []
    _dice_rows(state, board, bind_dice(state, board), rows)
    return moves([r for r in rows if r.flip or r.face or _unbalanced(r)])[0]


def table_cards(state):
    """[card id] of every card in play on the table (both sides)."""
    table = getattr(state, 'persistent_cards', None)
    return [str(c.card_id) for c in list(getattr(table, 'fp_cards', ())) + list(getattr(table, 'sh_cards', ()))]


def _card_rows(state, board, cid_pid, rows):
    """Event cards: one whose identity is known lies in its hand, on the table, in its deck or
    discarded; the cards nobody has seen lie in their decks."""
    from game_state import Side
    table = set(table_cards(state))
    used = {board.zone[p] for p in cid_pid.values() if board.zone.get(p) in TABLE}
    for cid, pid in sorted(cid_pid.items()):
        z = board.zone.get(pid)
        if z is None:
            continue
        side = Side.FREE_PEOPLES if str(cid).startswith('FREE_PEOPLES') else Side.SHADOW
        tag = 'fp' if side is Side.FREE_PEOPLES else 'sa'
        deck = 'character' if '_Character_' in str(cid) else 'strategy'
        if cid in state.hands[side].hand:
            want = FP_HAND if tag == 'fp' else SA_HAND
        elif cid in table:
            if z in TABLE:
                continue
            want = next((s for s in TABLE if s not in used), None)
            if want is None:
                continue
            used.add(want)
        elif cid in (getattr(state, ('fp_' if tag == 'fp' else 'sh_')
                             + ('char_deck' if deck == 'character' else 'strat_deck')) or ()):
            want = DECK[side.name][deck]
        else:
            want = FP_DEAD if tag == 'fp' else SA_DEAD
        if z != want:
            rows.append(Row('card of the %s' % ('Free Peoples' if tag == 'fp' else 'Shadow'),
                            tag, Counter({want: 1}), {pid: z}, [], {}, {}))
    # the cards nobody has seen: a card piece whose identity is not known belongs in its deck,
    # and the deck holds as many of them as the game has undealt cards there.  One that lies
    # elsewhere goes back (unseen); one too many is a card the game holds outside the deck
    # that no piece stands for - a row nothing can show
    seen = {str(p) for p in cid_pid.values()}
    for side, tag in ((Side.FREE_PEOPLES, 'fp'), (Side.SHADOW, 'sa')):
        for deck, code in sorted(DECK[side.name].items()):
            pile = getattr(state, ('fp_' if tag == 'fp' else 'sh_')
                           + ('char_deck' if deck == 'character' else 'strat_deck')) or ()
            rows.append(Row('unseen cards of the %s %s deck' % ('Free Peoples' if tag == 'fp' else 'Shadow', deck),
                            tag, Counter({code: sum(1 for cid in pile if cid not in cid_pid)}),
                            {p: z for p, z in board.zone.items() if p not in seen and layout.home(p) == code},
                            [], {}, {}))


def _tile_rows(state, board, rows):
    """Hunt tiles by the face a player sees: pool, used; the rest is removed."""
    import log_replay as lr
    from .randomness import tile_class
    want, have = {}, {}
    for code, pile in ((TILE_POOL, state.hunt_pool), (TILE_USED, state.hunt_discard)):
        for tile in pile or ():
            want.setdefault(tile_class(tile), Counter())[code] += 1
    for pid, name in lr.HUNT_TILE_PIECES.items():
        if pid in board.zone:
            have.setdefault(tile_class(name), {})[pid] = board.zone[pid]
    for cls in sorted(set(want) | set(have)):
        rows.append(Row('Hunt tile %s' % cls, 'sa', want.get(cls, Counter()),
                        have.get(cls, {}), [TILE_REMOVED], {}, {}))


def difference(state, board, cid_pid=None, settled=True, bound=None, dice=None, keep=(), fate=None):
    """Every row where the board is not the engine ([] = equal).  A plain
    function of (engine state, board); the rest is optional knowledge:
    cid_pid  {card id: piece} of the cards whose identity is known
    settled  False while a decision is open (display slots are left alone)
    bound    layout.bind_shared_boxes' pid -> (region, shared box)
    dice     bind_dice's result, when the caller already has it
    keep     the pieces that are a person's (boardanswer.BoardAnswers.owned; menu.owned): each
             is wanted where it lies (_kept) - {piece: the place he took it from, or None}, or
             the pieces alone; no other piece is moved in the stead of one (dice: bind_dice)
    fate     a dict: this pass ASKS whether what is a person's may stay where it lies, and is
             told what _kept found for each of his pieces it looks at (see there) - and True
             for one it looks at that no row holds (a card in its place): the game has it there."""
    dice = bind_dice(state, board, keep) if dice is None else dice
    rows = [_kept(r, keep, fate, settled) for r in _rows(state, board, cid_pid, settled, bound, dice)]
    if fate is not None and isinstance(keep, dict):
        fate.update((p, True) for p, frm in keep.items() if frm is not None and p not in fate and looks(p, settled))
    return [r for r in rows if r.flip or r.face or _unbalanced(r)]


def _rows(state, board, cid_pid=None, settled=True, bound=None, dice=None):
    """Every row of the comparison - where each class of pieces lies and where the engine
    has it -, balanced or not (see difference)."""
    if bound is None:
        bound = layout.bind_shared_boxes(state, board.zone)[1]
    rows = []
    _army_rows(state, board, bound or {}, rows)
    _marker_rows(state, board, rows)
    dice = bind_dice(state, board) if dice is None else dice
    if settled:
        _card_rows(state, board, cid_pid or {}, rows)
        _dice_rows(state, board, dice, rows)
        _tile_rows(state, board, rows)
    for attr in ('pending_army_move', 'pending_muster'):
        # the die of a two-step action under way lies in the play area: one that was killed
        # before the action ended (a take-back of the step that closed it) comes back
        side = getattr(state, attr + '_side', None)
        if isinstance(getattr(state, attr + '_die', None), int) and side is not None:
            tag, dead = ('fp', FP_DEAD) if side.name == 'FREE_PEOPLES' else ('sa', SA_DEAD)
            mine = [p for p, (s, i) in sorted(dice.items()) if s is side]
            early = [p for p in mine if board.zone.get(p) == dead and not state.dice[side][dice[p][1]].used]
            if early and not any(board.zone.get(p) in DISPLAY for p in mine):
                rows.append(Row('%s action dice' % ('Free Peoples' if tag == 'fp' else 'Shadow'), tag,
                                Counter({STAGED: 1}), {early[0]: dead}, [], {}, {}))
    return rows


def _kept(row, keep, fate=None, settled=True):
    """The row with the pieces of `keep` wanted where they lie - and THE arithmetic of "may this
    piece stay where it lies under the game as it is now".  A kept piece fills a place the game
    wants filled there, or it leaves a place short instead: the one it was taken from when
    that is short, else the first.  A pass that ASKS (fate; of a person's piece - its place of
    origin is known - that the pass looks at: `looks`) is told: True - it lies where the game
    has one of its class; False - the game still has a piece for it where he took it from (it
    has one of the class there that no other kept piece stands for - whichever piece lies
    there now: that one is then the spare the row moves - or that place, or the one it lies
    in, is a box of the row's pool, which holds what is not on the map); None - neither: the
    game has it elsewhere, or not at all.  Such a piece is NOT kept, and the row moves it like
    any piece."""
    mine = [p for p in sorted(row.have) if p in keep]
    if not mine:
        return row
    want, have, kept = Counter(row.want), dict(row.have), []
    for pid in mine:
        here = have.pop(pid)
        there = Counter(have.values())
        short = [z for z in sorted(want) if want[z] > there[z]]
        frm = keep.get(pid) if isinstance(keep, dict) else None
        ask = fate is not None and frm is not None and looks(pid, settled)
        hole = here if here in short else frm if frm in short or (ask and want[frm] > 0) else \
            short[0] if short and not ask else None
        if hole is not None:
            want[hole] -= 1
        if ask and hole is None and here not in row.pool and frm not in row.pool:
            have[pid] = here            # (not kept: it counts as a piece of the board like any other)
        else:
            kept.append(pid)
        if ask:     # (one that lies where he took it from and shows another side or face than the game's is one
            # he TURNED there: it is not "where the game has it", and stays his like any piece he moved)
            fate[pid] = (hole == here and not (here == frm and (pid in row.flip or pid in row.face))) if pid in kept else None
    return row._replace(want=+want, have=have, flip={p: v for p, v in row.flip.items() if p not in kept},
                        face={p: v for p, v in row.face.items() if p not in kept})


def _unbalanced(row):
    have = Counter(row.have.values())
    return any(have[z] != n for z, n in row.want.items()) or any(
        z not in row.want and z not in row.pool for z in have)


def moves(rows, first=()):
    """THE generic step: difference rows -> (ops, rows nothing can show).
    ops: ('flip', side, pid) | ('face', side, pid, letter) | ('move', side, pid, frm, to).
    first: the pieces to move before others of their class where one has to go (what has ceased
    to be a person's: the piece that goes back is the one he moved, never another in its stead) -
    as {piece: where it came from}, each goes back THERE when that is a place the row wants filled
    or a box of its pool."""
    ops, unshown, home = [], [], first if isinstance(first, dict) else {}
    for row in rows:
        for pid in sorted(row.flip):
            ops.append(('flip', row.side, pid))
        for pid, letter in sorted(row.face.items()):
            ops.append(('face', row.side, pid, letter))
        have = Counter(row.have.values())
        spare, free = [], []            # pieces to move off / pieces the pool offers
        for pid in sorted(row.have, key=lambda p: (p not in first, p)):
            z = row.have[pid]
            if z in row.want:
                if have[z] > row.want[z]:
                    have[z] -= 1
                    spare.append(pid)
            elif z in row.pool:
                free.append(pid)
            else:
                spare.append(pid)
        # a new piece comes from the reserve first, then from the pool's other boxes
        free.sort(key=lambda p: (row.have[p] not in (FP_REINF, SA_REINF), row.pool.index(row.have[p])))
        short = False
        for z in sorted(row.want):
            for _ in range(row.want[z] - have[z]):
                pid = next((p for p in spare + free if home.get(p) == z), spare[0] if spare else free[0] if free else None)
                if pid is None:
                    short = True
                    break
                (spare if pid in spare else free).remove(pid)
                ops.append(('move', row.side, pid, row.have[pid], z))
        for pid in spare:
            if not row.pool:
                short = True
                break
            ops.append(('move', row.side, pid, row.have[pid], home[pid] if home.get(pid) in row.pool else row.pool[0]))
        if short:
            unshown.append(row)
    return ops, unshown


def tidies(op):
    """Does this op of moves() only END a gesture - the die or card of an action, or a drawn
    tile, leaves the play area, or a die goes spent into a box of its row - and show nothing
    the game did beyond it?"""
    return op[0] == 'move' and not looks(op[2], False) and (op[3] in DISPLAY or (
        op[2] in layout.FP_DICE | layout.SH_DICE and op[4] in layout.boxes(op[2])))


def undoes(op):
    """Does this op of moves() take back what a person did with the piece: anything but tidying
    at a gesture's end - but also a die going back to its pool, a card back to its hand?"""
    return not tidies(op) or op[4] in (FP_HAND, SA_HAND, FP_DICE, SA_DICE)


def render(ops, nick_of):
    """ops -> client lines; flips first (a reveal flip moves the progress
    marker itself), then faces, then moves grouped by (side, from, to)."""
    lines = []
    for op in ops:
        if op[0] == 'flip':
            lines.append(em.render('flip', {'nick': nick_of(op[1]), 'pieces': [op[2]]}, local=False))
    for op in ops:
        if op[0] == 'face':
            lines.append(em.render('die_change', {'nick': nick_of(op[1]), 'die': op[2],
                                                  'face': op[3]}, local=False))
    groups = {}
    for op in ops:
        if op[0] == 'move' and op[3] != op[4]:
            groups.setdefault((op[1], op[3], op[4]), []).append(op[2])
    for (side, frm, to), pids in groups.items():
        lines.append(em.render('move', {'nick': nick_of(side), 'pieces': pids,
                                        'frm': frm, 'to': to}, local=False))
    return lines


def row_text(row):
    """One row in plain words for the log and the give-up sentence."""
    have = Counter(row.have.values())
    return '%s: board %s, engine %s' % (
        row.text, dict(have) or 'none', dict(row.want) or 'off the board')


NICK = {'fp': 'WOTR_AI_FP', 'sa': 'WOTR_AI_Shadow'}


def row_words(row):
    """One row in plain words (the sentence a person reads)."""
    import log_replay as lr
    have = Counter(row.have.values())
    less = [lr.decode_location(z) or z for z in sorted(row.want) if row.want[z] > have[z]]
    more = [lr.decode_location(z) or z for z in sorted(have)
            if z not in row.pool and have[z] > row.want.get(z, 0)]
    parts = (['missing in %s' % ', '.join(less)] if less else []) + (
        ['too many in %s' % ', '.join(more)] if more else [])
    return '%s (%s)' % (row.text, '; '.join(parts) or 'shown differently')
