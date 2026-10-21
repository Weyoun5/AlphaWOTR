"""log_census.py -- independent raw-log census harness (metrology gate).

Hand-written regex/text-heuristic counters over the RAW VASSAL logs act
as ground truth and are compared PER GAME against the parser's output
(log_replay.LogReplayer action_log + final state).  The census is
implemented INDEPENDENTLY of the parser's handler logic: it consumes
only the authoritative DATA tables (move-template key triples from
messages.properties, the piece-id table, the location-code table), plus
its own implementations of the client-level log normalizations that are
properties of the LOG TRANSPORT, not parser opinion:

  * VASSAL undo/redo netting (Controls.819 undo cancels the nearest
    preceding snapshotted $-command; die-face-change lines are NOT
    snapshotted so the walk continues past them; Controls.16 redo
    restores LIFO; a fresh $-command clears the redo stack),
  * adjacent transport-echo dedupe (byte-identical adjacent action
    lines, checksum-stripped, for move/kill/tile grammars),
  * aborted-drag netting (same pieces A->B then straight back B->A),
  * split-drag merge (disjoint handful, same route, within 3 lines).

Usage:
    python log_census.py --sample 500
        [--types elven_ring_fp_use,pass,...]   (REPORT_ROWS names;
        the ring deep-dive always runs regardless of --types)
        [--workers 4] [--no-parser] [--out data/log_census_report.json]

Deterministic: files are sorted by name and sampled by even stride.
Parser results are cached in data/log_census_parser_cache.jsonl so
re-runs only parse new games.  Console output is ASCII-only (cp1252).
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
LOG_DIR = os.path.join(ROOT, 'data', 'game_logs')
DATA_DIR = os.path.join(ROOT, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'log_census_parser_cache.jsonl')
DEFAULT_OUT = os.path.join(DATA_DIR, 'log_census_report.json')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Authoritative DATA tables (log-format spec, not parser logic)
# ---------------------------------------------------------------------------
from move_templates_authoritative import MOVE_TRIPLES
from piece_table_authoritative import AUTH_ARMY_PIECES

_K1 = '|'.join(re.escape(t[0]) for t in MOVE_TRIPLES)
_K2 = '|'.join(re.escape(t[1]) for t in MOVE_TRIPLES)
_K3 = '|'.join(re.escape(t[2]) for t in MOVE_TRIPLES)
# Piece-move line in ANY client encoding: <~k1~>pieces <~k2~><~from~><~k3~><~to~>
MOVE_RE = re.compile(
    r'<~(?:%s)~>([\d\s]+?)<~(?:%s)~><~([\w.]+)~><~(?:%s)~><~([\w.]+)~>'
    % (_K1, _K2, _K3))

# Die-face change chat command: '$Name changes 211  to P'
CHANGE_RE = re.compile(r'^\$[\w.]+\s+changes\s+(\d+)\s+to\s+([A-Z])\b')
# Dice roll line: <~Controls.345~>ids<~Controls.346~>values
ROLL_RE = re.compile(r'<~Controls\.345~>([\d\s]+?)<~Controls\.346~>([\d\s]+)')
# Hunt tile draw button
TILE_DRAW_RE = re.compile(r'<~Controls\.624~>')
# Eye-allocation button
EYE_BTN_RE = re.compile(r'<~Controls\.372~>(\d+)')
# Card draw buttons (deck -> hand); zone = FP Char/FP Strat/SA Char/SA Strat
CARD_DRAW_RE = re.compile(
    r'<~Controls\.(?:509|491|0|549|557)~>(\d+)<~Controls\.\d+~><~(Game\.200[4-7])~>')
# Piece flip (marker toggles / activations / coronation)
FLIP_RE = re.compile(r'<~Controls\.323~>(\d+)')
# Reveal-MOVE line (the payload is not purely numeric, so MOVE_RE cannot
# match it): <~k1~>reveal pid hash num <~k2~><~from~><~k3~><~to~>
REVEAL_MOVE_RE = re.compile(
    r'<~(?:%s)~>reveal\s+(\d+)\s+[A-F0-9]{8,}\s+(\d+)\s*'
    r'<~(?:%s)~><~([\w.]+)~><~(?:%s)~><~([\w.]+)~>' % (_K1, _K2, _K3))
# Swap grammars: 304/305 = removal at zone, 313/307-309/314 = placement
SWAP_REMOVE_RE = re.compile(
    r'<~Controls\.304~>([\d\s]+?)<~Controls\.305~><~([\w.]+)~>')
SWAP_PLACE_RE = re.compile(
    r'<~Controls\.313~>([\d\s]+?)<~Controls\.(307|308|309)~>'
    r'[^<]*<~Controls\.314~><~([\w.]+)~>')
# Die-consumption signatures (for die-less flip-pair detection)
DIE_SPEND_RE = re.compile(
    r'<~Controls\.276~>\d+<~Controls\.277~>|'
    r'<~HighlightLabel\.86~>\d+<~HighlightLabel\.87~>|'
    r'<~Controls\.345~>')
# Kill / cleanup command
KILL_RE = re.compile(r'<~Controls\.(?:296|644)~>([\d\s]+?)<~Controls\.(?:297|645)~>')
# Chat line
CHAT_RE = re.compile(r'^<([\w.]+)>\s*(.*)$')

# ---- piece vocabulary (authoritative piece table) -------------------------
UNIT_PIDS = frozenset(k for k, v in AUTH_ARMY_PIECES.items()
                      if v[1] in ('REGULAR', 'ELITE'))
LEADER_PIDS = frozenset(k for k, v in AUTH_ARMY_PIECES.items()
                        if v[1] == 'LEADER')
NAZGUL_PIDS = frozenset(k for k, v in AUTH_ARMY_PIECES.items()
                        if v[1] == 'NAZGUL')
COMPANION_PIDS = frozenset({'240', '241', '242', '243', '244', '245', '246',
                            '364'})
MINION_PIDS = frozenset({'253', '254', '255'})
RING_PIECES = {'249': 'Vilya', '250': 'Nenya', '251': 'Narya'}
DWARVEN_RING_PIDS = frozenset({'428', '429', '430', '431'})
TOKEN_PIDS = frozenset({'426', '427'})       # Hunt-for-the-Ring bid tokens
FSP_MARKER_PIDS = frozenset({'247', '248'})  # fellowship progress pawns
CORRUPTION_PID = '252'
POLITICAL_PIDS = frozenset('%03d' % i for i in range(256, 264))
CARD_PIDS = frozenset('%03d' % i for i in range(106, 210))
BASE_SH_DIE_PIDS = frozenset({'099', '100', '101', '102', '103', '104', '105'})
BASE_FP_DIE_PIDS = frozenset({'210', '211', '212', '213'})
KNOWN_BONUS_FP = frozenset({'358', '359'})
KNOWN_BONUS_SH = frozenset({'360', '361', '362'})

# ---- zone vocabulary (authoritative location table) -----------------------
RING_ACTIVE_LOCS = frozenset({'Game.2008', 'Game.2009', 'Game.2010'})
RING_USED_LOCS = frozenset({'Game.2011', 'Game.2012', 'Game.2013'})
FP_DICE_ZONE, SA_DICE_ZONE = 'Game.2042', 'Game.2043'
USED_ZONE = 'Game.2035'                      # Current Action (used dice)
HUNT_BOX_ZONE = 'Game.1997'
FP_REINF_ZONE, SA_REINF_ZONE = 'Game.2038', 'Game.2040'
FP_DEAD_ZONE, SA_DEAD_ZONE = 'Game.2039', 'Game.2041'
DECK_ZONES = {'Game.2004': 'fp_char', 'Game.2005': 'fp_strat',
              'Game.2006': 'sa_char', 'Game.2007': 'sa_strat'}
HAND_ZONES = frozenset({'Controls.45', 'Controls.46'})
PLAY_ZONES = frozenset({'Game.2029', 'Game.2030', 'Game.2031', 'Game.2032',
                        'Game.2033', 'Game.2034', 'Game.2035', 'Game.2036',
                        'Game.2037'})
FSP_TRACK_ZONES = {('Game.%d' % c): (c - 2014) for c in range(2014, 2027)}
MORDOR_TRACK_ZONES = {'Game.524': 0, 'Game.525': 1, 'Game.528': 2,
                      'Game.529': 3, 'Game.532': 4, 'Game.533': 5}
FELLOWSHIP_BOX, GUIDE_BOX = 'Game.1998', 'Game.1999'
# Political track: nation home boxes (printed start slots) + shared boxes.
# Advance = toward 'Political Track 0' (Game.2003 = At War).
POLITICAL_BOX_RANK = {'Game.2000': 0, 'Game.2001': 1, 'Game.2002': 2,
                      'Game.2003': 3}
POLITICAL_HOME_BOXES = frozenset(
    {'Game.1863', 'Game.1865', 'Game.1868', 'Game.1871', 'Game.1874',
     'Game.1877', 'Game.1879', 'Game.1881'})


def _map_region_codes():
    """Map-region Game codes: the contiguous authoritative region block
    1883..1987 (Minhiriath .. North Rhun)."""
    return frozenset('Game.%d' % c for c in range(1883, 1988))


MAP_REGION_CODES = _map_region_codes()
SIEGE_BOX_CODES = frozenset({'Game.%d' % c for c in range(1988, 1997)}
                            | {'Game.2028', 'Game.19'})

# ---------------------------------------------------------------------------
# Raw-line loading and client-level normalization (independent impl)
# ---------------------------------------------------------------------------
DIE_CHANGE_LINE_RE = re.compile(r'^\$[\w.]+\s+changes\s+\d+\s+to\s+[A-Z]\b')


def read_lines(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [ln.strip() for ln in f if ln.strip()]


def strip_core(line):
    line = re.sub(r'\s+[A-F0-9]{6}$', '', line)
    return re.sub(r'\s+null$', '', line)


def undo_redo_net(lines):
    """Native snapshot stack; rolls and card choices do not clear redo."""
    from client_snapshot_net import net_raw
    return net_raw(lines)


def dedupe_echoes(lines):
    """Blank adjacent byte-identical (checksum-stripped) action lines for
    move / kill / tile-draw grammars (transport echoes)."""
    out = list(lines)
    last_core = None
    for i, line in enumerate(out):
        ls = line.strip()
        if not ls or not ls.startswith('$') or ' sendTokens ' in ls or not (
                MOVE_RE.search(ls) or '<~Controls.296~>' in ls
                or '<~Controls.644~>' in ls or '<~Controls.624~>' in ls):
            last_core = None
            continue
        core = strip_core(ls)
        if core == last_core:
            out[i] = ''
            continue
        last_core = core
    return out


def _is_skippable(line):
    return (line.startswith('<auto>') or line.startswith('$base')
            or line.startswith('$TEST'))


class Gesture(object):
    __slots__ = ('line', 'pieces', 'frm', 'to')

    def __init__(self, line, pieces, frm, to):
        self.line = line
        self.pieces = pieces
        self.frm = frm
        self.to = to


def collect_die_ids(lines):
    """All die piece ids seen, and the Shadow-side subset (a roll line
    that contains a base-Shadow id classifies every id it lists)."""
    ids = set(BASE_SH_DIE_PIDS) | set(BASE_FP_DIE_PIDS) \
        | set(KNOWN_BONUS_FP) | set(KNOWN_BONUS_SH)
    sh = set(BASE_SH_DIE_PIDS) | set(KNOWN_BONUS_SH)
    fp = set(BASE_FP_DIE_PIDS) | set(KNOWN_BONUS_FP)
    for ln in lines:
        m = ROLL_RE.search(ln)
        if m:
            roll_ids = m.group(1).split()
            ids.update(roll_ids)
            if any(r in BASE_SH_DIE_PIDS for r in roll_ids):
                sh.update(r for r in roll_ids if r not in fp)
            elif any(r in BASE_FP_DIE_PIDS for r in roll_ids):
                fp.update(r for r in roll_ids if r not in sh)
    return frozenset(ids), frozenset(sh)


def extract_gestures(lines, die_ids):
    """All piece-move gestures with aborted-drag netting and split-drag
    merging applied (map-route moves only, mirroring client behavior).
    Returns (gestures, skipped_line_set)."""
    raw = []
    for n, ln in enumerate(lines, 1):
        if not ln or _is_skippable(ln):
            continue
        core = strip_core(ln)
        m = MOVE_RE.search(core)
        if m is None:
            continue
        pieces = tuple(sorted(m.group(1).split()))
        if not pieces:
            continue
        raw.append(Gesture(n, pieces, m.group(2), m.group(3)))

    def _map(tok):
        return tok in MAP_REGION_CODES

    skipped = set()
    last = None
    for g in raw:
        pieces = g.pieces
        if any(p in FSP_MARKER_PIDS or p == CORRUPTION_PID
               for p in pieces):
            last = None
            continue
        if all(p in die_ids for p in pieces):
            last = None          # dice bookkeeping never pairs
            continue
        if last is not None:
            lg = last
            if pieces == lg.pieces and g.frm == lg.to and g.to == lg.frm \
                    and g.frm != g.to and g.line - lg.line <= 6:
                skipped.add(lg.line)
                skipped.add(g.line)
                last = None
                continue
            if (set(pieces).isdisjoint(lg.pieces) and g.frm == lg.frm
                    and g.to == lg.to and g.frm != g.to
                    and g.line - lg.line <= 3 and _map(g.frm) and _map(g.to)):
                skipped.add(g.line)
                lg.pieces = tuple(sorted(set(lg.pieces) | set(pieces)))
                continue
            if (set(pieces) < set(lg.pieces) and g.frm == lg.to
                    and g.to == lg.frm and g.frm != g.to
                    and _map(g.frm) and _map(g.to)):
                skipped.add(g.line)
                lg.pieces = tuple(sorted(set(lg.pieces) - set(pieces)))
                continue
        last = g
    return [g for g in raw if g.line not in skipped], skipped


def prepare_game(path):
    """Load + normalize a game log. Returns a context dict."""
    raw_lines = read_lines(path)
    net_lines = undo_redo_net(dedupe_echoes(list(raw_lines)))
    die_ids, sh_die_ids = collect_die_ids(net_lines)
    gestures, skipped = extract_gestures(net_lines, die_ids)
    return {
        'raw': raw_lines,
        'net': net_lines,
        'die_ids': die_ids,
        'sh_die_ids': sh_die_ids,
        'gestures': gestures,
        'skipped': skipped,
    }


# ---------------------------------------------------------------------------
# PART 1 -- Elven-ring census (per-piece state machine)
# ---------------------------------------------------------------------------
def _sh_die_change_lines(lines, die_ids, sh_die_ids):
    """Line numbers of Shadow die-face changes and FP die-face changes.
    Classifies by the static vocabulary PLUS the per-game discovered
    ids (collect_die_ids) so bonus dice count."""
    sh, fp = [], []
    sh_ids = set(BASE_SH_DIE_PIDS) | set(KNOWN_BONUS_SH) | set(sh_die_ids)
    fp_ids = (set(BASE_FP_DIE_PIDS) | set(KNOWN_BONUS_FP)
              | (set(die_ids) - sh_ids))
    for n, ln in enumerate(lines, 1):
        if not ln:
            continue
        m = CHANGE_RE.match(strip_core(ln))
        if not m:
            continue
        pid = m.group(1)
        if pid in sh_ids:
            sh.append((n, pid, m.group(2)))
        elif pid in fp_ids:
            fp.append((n, pid, m.group(2)))
    return sh, fp


def _zone_class(z):
    if z in RING_ACTIVE_LOCS:
        return 'A'
    if z in RING_USED_LOCS:
        return 'U'
    return 'O'


def census_ring(ctx):
    """Elven-ring census: FP uses, Shadow collects, proven Shadow uses.

    Zone-transition machine per ring piece (249/250/251).  The FROM zone
    of each drag is trusted over remembered state (multi-ring drags and
    module quirks can move pieces invisibly); memory only distinguishes
    'in flight from active' vs 'collected' for the O (other) zone class.
      A -> U : FP use             A -> O : FP use (hop 1, pending)
      A -> A : slot tidy (no-op)  U -> A : FP-use retraction
      U -> O : Shadow collect     U -> U : tidy (no-op)
      O -> U : hop-2 landing (no count) or un-collect
      O -> A : abort hop (fp-1) or full revert (fp-1, collect-1)
    A Shadow COLLECT with a Shadow die-face change within +-6 lines is a
    PROVEN Shadow use (rings are spent to change a die result)."""
    lines = ctx['net']
    sh_changes, fp_changes = _sh_die_change_lines(
        lines, ctx['die_ids'], ctx['sh_die_ids'])
    sh_change_lines = [n for (n, _p, _f) in sh_changes]

    RETRACT_WINDOW = 30  # lines: a return to the active row only NETS a
    # use when it comes soon after (undo semantics); later returns are
    # RE-GRANTS (players regenerating a ring -- a misplay, but the uses
    # before and after both really happened and both count)
    mem = {p: None for p in RING_PIECES}   # None|'pending_a'|'collected'
    last_use_line = {p: None for p in RING_PIECES}
    fp_use, sh_collect = 0, 0
    events = []          # (line, kind, ring)
    collect_events = []  # [line, ring] surviving collects
    fp_events = []       # [line, ring] surviving fp uses
    regrants = 0

    def _drop_last(evlist, pid):
        name = RING_PIECES[pid]
        for i in range(len(evlist) - 1, -1, -1):
            if evlist[i][1] == name:
                evlist.pop(i)
                return
        if evlist:
            evlist.pop()

    for n, ln in enumerate(lines, 1):
        if not ln or _is_skippable(ln):
            continue
        m = MOVE_RE.search(strip_core(ln))
        if m is None:
            continue
        all_pieces = m.group(1).split()
        rings = [p for p in all_pieces if p in RING_PIECES]
        if not rings:
            continue
        frm, to = m.group(2), m.group(3)
        fc, tc = _zone_class(frm), _zone_class(to)
        for pid in rings:
            name = RING_PIECES[pid]
            if fc == 'A':
                if tc == 'U':
                    fp_use += 1
                    mem[pid] = None
                    last_use_line[pid] = n
                    events.append((n, 'fp_use', name))
                    fp_events.append([n, name])
                elif tc == 'O':
                    fp_use += 1
                    mem[pid] = ('pending_a', n)
                    last_use_line[pid] = n
                    events.append((n, 'fp_use_hop1', name))
                    fp_events.append([n, name])
                # A -> A: slot tidy, no-op
            elif fc == 'U':
                if tc == 'A':
                    lu = last_use_line[pid]
                    if lu is not None and n - lu <= RETRACT_WINDOW \
                            and fp_use > 0:
                        fp_use -= 1
                        events.append((n, 'fp_retract', name))
                        _drop_last(fp_events, pid)
                    else:
                        regrants += 1
                        events.append((n, 'fp_regrant', name))
                    mem[pid] = None
                elif tc == 'O':
                    sh_collect += 1
                    mem[pid] = ('collected', n)
                    events.append((n, 'sh_collect', name))
                    collect_events.append([n, name])
                # U -> U: tidy, no-op
            else:  # from O
                st = mem[pid]
                if tc == 'U':
                    if st and st[0] == 'pending_a' and n - st[1] <= 12:
                        events.append((n, 'fp_use_hop2', name))
                    elif st and st[0] == 'collected' and n - st[1] <= 12:
                        # quick walk-back of the pickup; a LATER return
                        # into a dark slot is the Shadow player parking
                        # the held ring there (no un-collect)
                        if sh_collect > 0:
                            sh_collect -= 1
                            events.append((n, 'sh_uncollect', name))
                            _drop_last(collect_events, pid)
                    mem[pid] = None
                elif tc == 'A':
                    if st and st[0] == 'pending_a':
                        if fp_use > 0:
                            fp_use -= 1
                            events.append((n, 'fp_abort_hop', name))
                            _drop_last(fp_events, pid)
                    elif st and st[0] == 'collected':
                        cl = st[1]
                        if n - cl <= RETRACT_WINDOW:
                            if sh_collect > 0:
                                sh_collect -= 1
                                _drop_last(collect_events, pid)
                            lu = last_use_line[pid]
                            if lu is not None \
                                    and n - lu <= 2 * RETRACT_WINDOW \
                                    and fp_use > 0:
                                fp_use -= 1
                                _drop_last(fp_events, pid)
                            events.append((n, 'full_revert', name))
                        else:
                            regrants += 1
                            events.append((n, 'fp_regrant', name))
                    mem[pid] = None

    sh_use_proven = 0
    for (n, _name) in collect_events:
        if any(abs(c - n) <= 6 for c in sh_change_lines):
            sh_use_proven += 1
    fp_use = max(0, fp_use)
    sh_collect = max(0, sh_collect)
    return {
        'fp_uses': fp_use,
        'sh_collects': sh_collect,
        'sh_uses_proven': sh_use_proven,
        'total_strict': fp_use + sh_use_proven,
        'total_loose': fp_use + sh_collect,
        'fp_lines': [n for (n, _r) in fp_events],
        'collect_lines': [n for (n, _r) in collect_events],
        'events': events,
        'regrants': regrants,
        'sh_changes': len(sh_change_lines),
        'fp_changes': len(fp_changes),
    }


def census_ring_naive(ctx):
    """Naive drag counting on the RAW (un-netted) line list: every
    active->used / active->out drag = +1 FP, every used->out drag = +1
    SH.  No undo netting, no state machine.  This is the count a naive
    regex would produce; the delta vs census_ring is the undo noise."""
    fp = sh = 0
    for ln in ctx['raw']:
        if not ln or _is_skippable(ln):
            continue
        m = MOVE_RE.search(strip_core(ln))
        if m is None:
            continue
        pieces = m.group(1).split()
        if len(pieces) != 1 or pieces[0] not in RING_PIECES:
            continue
        frm, to = m.group(2), m.group(3)
        if frm in RING_ACTIVE_LOCS and to not in RING_ACTIVE_LOCS:
            fp += 1
        elif frm in RING_USED_LOCS and to not in RING_USED_LOCS \
                and to not in RING_ACTIVE_LOCS:
            sh += 1
    return {'fp_uses': fp, 'sh_collects': sh, 'total': fp + sh}


# ---------------------------------------------------------------------------
# PART 2 -- generic per-type census counters
# ---------------------------------------------------------------------------
def census_pass(ctx):
    n = 0
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln:
            continue
        m = CHAT_RE.match(strip_core(ln))
        if m and m.group(1) not in ('auto', 'game', 'replay') \
                and m.group(2).strip().lower() == 'pass':
            n += 1
            lines_at.append(i)
    return {'count': n, 'lines': lines_at}


def census_chat_token(ctx, token):
    """Chat lines whose LAST whitespace token is *token* (nc) — players
    type card counts before it — or whose payload matches ffN."""
    n = 0
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln:
            continue
        m = CHAT_RE.match(strip_core(ln))
        if not m or m.group(1) in ('auto', 'game', 'replay'):
            continue
        msg = m.group(2).strip().lower()
        toks = msg.split()
        if token == 'nc':
            if toks and toks[-1] == 'nc':
                n += 1
                lines_at.append(i)
        elif token == 'ff':
            if re.fullmatch(r'ff\s*\d', msg):
                n += 1
                lines_at.append(i)
    return {'count': n, 'lines': lines_at}


def census_dice_roll(ctx):
    multi = single = 0
    last_roll = 0
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln):
            continue
        m = ROLL_RE.search(ln)
        if m:
            ids = m.group(1).split()
            if len(ids) >= 2:
                multi += 1
            else:
                single += 1
            last_roll = i
    return {'multi': multi, 'single': single, 'last_roll_line': last_roll}


def census_hunt_tile_draw(ctx):
    total = 0
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        if TILE_DRAW_RE.search(ln):
            total += 1
            lines_at.append(i)
    last_roll = census_dice_roll(ctx)['last_roll_line']
    tail = sum(1 for n in lines_at if n > last_roll)
    return {'count': total, 'tail_after_last_roll': tail, 'lines': lines_at}


def census_draw_card(ctx):
    """Deck exits.  Primary grammar: the draw button (pid, deck zone).
    Fallback: a card piece dragged out of a deck zone by a move triple
    on a line with no draw button (some client builds)."""
    per_deck = collections.Counter()
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        core = strip_core(ln)
        m = CARD_DRAW_RE.search(core)
        if m:
            per_deck[DECK_ZONES[m.group(2)]] += 1
            lines_at.append(i)
            continue
        mm = MOVE_RE.search(core)
        if mm and mm.group(2) in DECK_ZONES:
            pieces = mm.group(1).split()
            if pieces and all(p in CARD_PIDS for p in pieces):
                per_deck[DECK_ZONES[mm.group(2)]] += len(pieces)
                lines_at.append(i)
    return {'count': sum(per_deck.values()),
            'per_deck': dict(per_deck), 'lines': lines_at}


SH_EYE_VALUES = {'5', '15'}


def census_eye_allocate(ctx):
    """Eye allocations: SHADOW dice into the Hunt Box outside roll lines
    (dice that just ROLLED an Eye are excluded -- their drag into the
    box is bookkeeping of the roll) + Controls.372 button presses, net
    of pre-roll retractions.  A button press followed within 3 lines by
    a drag of the same die into the box counts once."""
    lines = ctx['net']
    sh_ids = ctx['sh_die_ids']
    events = []          # [line, die, kind]
    pending = {}
    btn_recent = {}
    rolled_eyes = set()
    for i, ln in enumerate(lines, 1):
        if not ln or _is_skippable(ln):
            continue
        core = strip_core(ln)
        if '<~Game.7~>' in core:
            rolled_eyes.clear()
            pending.clear()
            continue
        rm = ROLL_RE.search(core)
        if rm:
            pending.clear()
            ids = rm.group(1).split()
            vals = rm.group(2).split()
            if any(r in BASE_SH_DIE_PIDS for r in ids):
                for r, v in zip(ids, vals):
                    if v in SH_EYE_VALUES:
                        rolled_eyes.add(r)
            continue
        if TILE_DRAW_RE.search(core):
            pending.clear()
            continue
        bm = EYE_BTN_RE.search(core)
        if bm and core.startswith('$'):
            die = bm.group(1)
            events.append([i, die, 'btn'])
            pending[die] = len(events) - 1
            btn_recent[die] = i
            continue
        m = MOVE_RE.search(core)
        if m is None:
            continue
        pieces = m.group(1).split()
        frm, to = m.group(2), m.group(3)
        if to == HUNT_BOX_ZONE:
            for p in pieces:
                if p not in sh_ids:
                    continue
                if p in rolled_eyes:
                    rolled_eyes.discard(p)   # roll bookkeeping drag
                    continue
                if p in btn_recent and i - btn_recent[p] <= 3:
                    continue                 # same allocation, two dialects
                events.append([i, p, 'drag'])
                pending[p] = len(events) - 1
        elif frm == HUNT_BOX_ZONE and to in (FP_DICE_ZONE, SA_DICE_ZONE,
                                             USED_ZONE):
            for p in pieces:
                idx = pending.pop(p, None)
                if idx is not None:
                    events[idx][2] = 'retracted'
    kept = [e for e in events if e[2] in ('btn', 'drag')]
    return {'count': len(kept),
            'retracted': sum(1 for e in events if e[2] == 'retracted'),
            'lines': [e[0] for e in kept]}


def census_fellowship_track(ctx):
    """Fellowship progress pawn (247/248) walks on the progress track and
    the Mordor track; corruption marker (252) excluded."""
    p1 = resets = mordor = 0
    p1_lines = []
    mordor_lines = []
    for g in ctx['gestures']:
        if len(g.pieces) != 1:
            continue
        p = g.pieces[0]
        if p not in FSP_MARKER_PIDS:
            continue
        f_t = FSP_TRACK_ZONES.get(g.frm)
        t_t = FSP_TRACK_ZONES.get(g.to)
        if f_t is not None and t_t is not None:
            if t_t == f_t + 1:
                p1 += 1
                p1_lines.append(g.line)
            elif t_t < f_t:
                resets += 1
            continue
        f_m = MORDOR_TRACK_ZONES.get(g.frm)
        t_m = MORDOR_TRACK_ZONES.get(g.to)
        if t_m is not None and (f_m is not None or g.frm not in
                                MORDOR_TRACK_ZONES):
            if f_m is not None and t_m == f_m + 1:
                mordor += 1
                mordor_lines.append(g.line)
    return {'advance_p1': p1, 'resets': resets, 'mordor_steps': mordor,
            'p1_lines': p1_lines, 'mordor_lines': mordor_lines}


def census_fsp_flip(ctx):
    """Fellowship marker flips (hide/reveal toggles).  Quick DIE-LESS
    flip pairs (no die spend / roll between) are netted as tracker-reset
    bookkeeping, mirroring the client convention."""
    lines = ctx['net']
    flips = []
    for i, ln in enumerate(lines, 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        m = FLIP_RE.search(strip_core(ln))
        if m and m.group(1) in FSP_MARKER_PIDS:
            flips.append(i)

    def _die_between(a, b):
        for j in range(a, b - 1):
            if lines[j] and DIE_SPEND_RE.search(lines[j]):
                return True
        return False

    netted = []
    i = 0
    while i < len(flips):
        if i + 1 < len(flips) and flips[i + 1] - flips[i] <= 6 \
                and not _die_between(flips[i], flips[i + 1]):
            i += 2               # die-less quick pair: net both
            continue
        netted.append(flips[i])
        i += 1
    return {'count_raw': len(flips), 'count_net': len(netted),
            'lines': netted}


def census_fsp_pawn_place(ctx):
    """Fellowship pawn 247 placed onto a MAP region = declaration or
    reveal placement.  Multi-hop drags (next FROM == previous TO within
    a short window) collapse into ONE placement, mirroring the drag
    convention (Hollin -> Moria -> Dimrill Dale is one declare)."""
    events = []
    last = None          # (line, to)
    for g in ctx['gestures']:
        if len(g.pieces) != 1 or g.pieces[0] != '247':
            continue
        if g.to not in MAP_REGION_CODES:
            last = None
            continue
        if last is not None and g.frm == last[1] and g.line - last[0] <= 6:
            last = (g.line, g.to)          # hop continuation
            events[-1] = g.line
            continue
        events.append(g.line)
        last = (g.line, g.to)
    return {'count': len(events), 'lines': events}


def census_political(ctx):
    """Political marker walks between track boxes + activation flips."""
    walks = advances = 0
    walk_lines = []
    flips = 0
    flip_pids = set()
    for g in ctx['gestures']:
        if len(g.pieces) != 1 or g.pieces[0] not in POLITICAL_PIDS:
            continue
        f_r = POLITICAL_BOX_RANK.get(g.frm)
        t_r = POLITICAL_BOX_RANK.get(g.to)
        f_home = g.frm in POLITICAL_HOME_BOXES
        t_home = g.to in POLITICAL_HOME_BOXES
        if (f_r is not None or f_home) and (t_r is not None or t_home):
            walks += 1
            walk_lines.append(g.line)
            if t_r is not None and (f_home or (f_r is not None
                                               and t_r > f_r)):
                advances += 1
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        m = FLIP_RE.search(strip_core(ln))
        if m and m.group(1) in POLITICAL_PIDS:
            flips += 1
            flip_pids.add(m.group(1))
    return {'walks': walks, 'advances': advances, 'walk_lines': walk_lines,
            'flips': flips, 'flip_distinct': len(flip_pids)}


def census_dwarven_ring(ctx):
    """Dwarven-ring piece (428-431) uses (drag out of a dice zone) and
    grants ('adds N Dwarven Ring(s)' announce or drag INTO a dice zone)."""
    uses = grants = 0
    use_lines = []
    for g in ctx['gestures']:
        if not all(p in DWARVEN_RING_PIDS for p in g.pieces):
            continue
        if g.frm in (FP_DICE_ZONE, SA_DICE_ZONE) \
                and g.to not in (FP_DICE_ZONE, SA_DICE_ZONE):
            uses += len(g.pieces)
            use_lines.append(g.line)
        elif g.to in (FP_DICE_ZONE, SA_DICE_ZONE) \
                and g.frm not in (FP_DICE_ZONE, SA_DICE_ZONE):
            grants += len(g.pieces)
    # NOTE: 'adds N Dwarven Ring(s)' announce lines accompany the grant
    # drag in the same client gesture -- counting both would double.
    return {'uses': uses, 'grants': grants, 'use_lines': use_lines}


def census_use_token(ctx):
    uses = 0
    lines_at = []
    for g in ctx['gestures']:
        if not all(p in TOKEN_PIDS for p in g.pieces):
            continue
        if g.frm in (FP_DICE_ZONE, SA_DICE_ZONE) \
                and g.to not in (FP_DICE_ZONE, SA_DICE_ZONE) \
                and g.frm != g.to:
            uses += len(g.pieces)
            lines_at.append(g.line)
    return {'count': uses, 'lines': lines_at}


CARDBUFFET_REVEAL_RE = re.compile(
    r'<~CardBuffet\.(9|14)~>reveal\s+(\d+)\s+[A-F0-9]{8,}\s+(\d+)'
    r'<~CardBuffet\.(?:10|15)~><~(Controls\.4[56])~><~CardBuffet\.(?:11|16)~>')
COMMIT_REVEAL_RE = re.compile(
    r'<~Controls\.(3|861|866)~>reveal\s+(\d+)\s+[A-F0-9]{8,}\s+(\d+)'
    r'<~Controls\.\d+~><~([\w.]+)~><~Controls\.\d+~>(?:<~([\w.]+)~>)?')


def census_play_event(ctx):
    """Face-up card reveals leaving a hand.  Three dialects:
      A: move-triple reveal from hand to a play zone,
      B: CardBuffet.9/14 suffixed reveal (destination baked in),
      C: Controls.3/861/866 commit reveal (playchosen / combat-card
         commit; combat cards share this grammar with event plays)."""
    a = b = c = 0
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln:
            continue
        core = strip_core(ln)
        if core.startswith('<auto>'):
            core = core[len('<auto>'):]
        elif not core.startswith('$'):
            continue
        m = REVEAL_MOVE_RE.search(core)
        if m and m.group(3) in HAND_ZONES and m.group(4) in PLAY_ZONES:
            a += 1
            lines_at.append(i)
            continue
        m = CARDBUFFET_REVEAL_RE.search(core)
        if m:
            b += 1
            lines_at.append(i)
            continue
        m = COMMIT_REVEAL_RE.search(core)
        if m:
            c += 1
    return {'count': a + b, 'hand_to_zone': a, 'cardbuffet': b,
            'commit_reveals': c, 'lines': lines_at}


def census_flip_piece(ctx, pid):
    n = 0
    lines_at = []
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        m = FLIP_RE.search(strip_core(ln))
        if m and m.group(1) == pid:
            n += 1
            lines_at.append(i)
    return {'count': n, 'lines': lines_at}


def census_companion_separate(ctx):
    """Companion pieces leaving the Fellowship/Guide box for the map or a
    staging Board box."""
    n = 0
    lines_at = []
    board_stage = frozenset({'Game.2030', 'Game.2031', 'Game.2032',
                             'Game.2033', 'Game.2034'})
    for g in ctx['gestures']:
        if not g.pieces or not all(p in COMPANION_PIDS for p in g.pieces):
            continue
        if g.frm in (FELLOWSHIP_BOX, GUIDE_BOX) and (
                g.to in MAP_REGION_CODES or g.to in SIEGE_BOX_CODES
                or g.to in board_stage):
            n += len(g.pieces)
            lines_at.append(g.line)
    return {'count': n, 'lines': lines_at}


def census_choose_guide(ctx):
    """Net companion entries into the Guide Box (walk-backs subtract)."""
    n = 0
    lines_at = []
    for g in ctx['gestures']:
        if not g.pieces or not all(p in COMPANION_PIDS for p in g.pieces):
            continue
        if g.to == GUIDE_BOX:
            n += 1
            lines_at.append(g.line)
        elif g.frm == GUIDE_BOX and g.to == FELLOWSHIP_BOX:
            n -= 1
            if lines_at:
                lines_at.pop()
    return {'count': max(0, n), 'lines': lines_at}


CASUALTY_ZONES = frozenset({'Game.161', 'Game.163', 'Game.304', 'Game.306',
                            FP_DEAD_ZONE, SA_DEAD_ZONE})


def census_companion_death(ctx):
    """Companion deaths: drags into a casualty zone / Current Action,
    plus kill-command (Controls.296/644) lines listing companion pids."""
    n = 0
    lines_at = []
    death_zones = CASUALTY_ZONES | {USED_ZONE}
    for g in ctx['gestures']:
        comp = [p for p in g.pieces if p in COMPANION_PIDS]
        if comp and g.to in death_zones:
            n += len(comp)
            lines_at.append(g.line)
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        m = KILL_RE.search(strip_core(ln))
        if not m:
            continue
        comp = [p for p in m.group(1).split() if p in COMPANION_PIDS]
        if comp:
            n += len(comp)
            lines_at.append(i)
    return {'count': n, 'lines': lines_at}


def census_muster(ctx):
    """Muster gestures: army pieces from a reinforcement OR casualty
    zone (the one return path: replacement regulars come back from the
    casualty box) to a map region or siege box.  Control/VP markers and
    non-army pieces are excluded."""
    n = units = from_cas = 0
    lines_at = []
    for g in ctx['gestures']:
        src_reinf = g.frm in (FP_REINF_ZONE, SA_REINF_ZONE)
        src_cas = g.frm in CASUALTY_ZONES
        if not (src_reinf or src_cas):
            continue
        if not (g.to in MAP_REGION_CODES or g.to in SIEGE_BOX_CODES):
            continue
        army = [p for p in g.pieces
                if p in UNIT_PIDS or p in LEADER_PIDS or p in NAZGUL_PIDS
                or p in MINION_PIDS]
        if not army:
            continue
        n += 1
        units += len(army)
        if src_cas:
            from_cas += 1
        lines_at.append(g.line)
    return {'gestures': n, 'units': units, 'from_casualties': from_cas,
            'lines': lines_at}


def census_army_moves(ctx):
    """Map->map drag gestures partitioned by piece class."""
    army = nazgul_only = comp_only = 0
    army_lines = []
    naz_lines = []
    siege_in = 0
    for g in ctx['gestures']:
        interesting = [p for p in g.pieces
                       if p in UNIT_PIDS or p in LEADER_PIDS
                       or p in NAZGUL_PIDS or p in COMPANION_PIDS
                       or p in MINION_PIDS]
        if not interesting or len(interesting) != len(g.pieces):
            continue
        frm_map = g.frm in MAP_REGION_CODES
        to_map = g.to in MAP_REGION_CODES
        if frm_map and g.to in SIEGE_BOX_CODES:
            siege_in += 1
            continue
        if not (frm_map and to_map):
            continue
        if all(p in NAZGUL_PIDS for p in g.pieces):
            nazgul_only += 1
            naz_lines.append(g.line)
        elif all(p in COMPANION_PIDS for p in g.pieces):
            comp_only += 1
        else:
            army += 1
            army_lines.append(g.line)
    return {'army_gestures': army, 'nazgul_only': nazgul_only,
            'companion_only': comp_only, 'siege_entries': siege_in,
            'army_lines': army_lines, 'nazgul_lines': naz_lines}


def census_casualties(ctx):
    """Army pieces removed: drags into casualty zones, kill commands
    (Controls.296/644, killed in place at the battle region), and
    swap-removals (Controls.304, elite downgrade / press payments).
    Piece count, all army kinds."""
    drag_units = kill_units = swap_units = 0
    lines_at = []

    def _army(pids):
        return [p for p in pids if p in UNIT_PIDS or p in LEADER_PIDS
                or p in NAZGUL_PIDS]

    for g in ctx['gestures']:
        if g.to in CASUALTY_ZONES:
            u = _army(g.pieces)
            if u:
                drag_units += len(u)
                lines_at.append(g.line)
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        core = strip_core(ln)
        km = KILL_RE.search(core)
        if km:
            u = _army(km.group(1).split())
            if u:
                kill_units += len(u)
                lines_at.append(i)
            continue
        sm = SWAP_REMOVE_RE.search(core)
        if sm:
            u = _army(sm.group(1).split())
            if u:
                swap_units += len(u)
                lines_at.append(i)
    return {'units': drag_units + kill_units + swap_units,
            'drag_units': drag_units, 'kill_units': kill_units,
            'swap_units': swap_units, 'lines': lines_at}


def census_swap_place(ctx):
    """Controls.313 placement lines listing army pieces, ANY
    destination zone (casualty-replacement / downgrade placements);
    the captured zone is deliberately not filtered."""
    n = units = 0
    for i, ln in enumerate(ctx['net'], 1):
        if not ln or _is_skippable(ln) or not ln.startswith('$'):
            continue
        m = SWAP_PLACE_RE.search(strip_core(ln))
        if not m:
            continue
        u = [p for p in m.group(1).split() if p in UNIT_PIDS
             or p in LEADER_PIDS or p in NAZGUL_PIDS]
        if u:
            n += 1
            units += len(u)
    return {'gestures': n, 'units': units}


# ---------------------------------------------------------------------------
# Parser side
# ---------------------------------------------------------------------------
def parser_summarize(path):
    """Run log_replay.LogReplayer on one game; extract comparable counts.
    Heavy import happens here so census-only runs stay light."""
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    import board as board_mod
    import game_state
    from log_replay import LogReplayer
    b = board_mod.load_board()
    st = game_state.setup_initial_game(b)
    rep = LogReplayer(st, b, verbose=False)
    rep.parse_and_replay(path)
    counts = collections.Counter(a for a, _p, _s in rep.action_log)
    sub = collections.Counter()
    ring_lines = []
    collect_lines = []
    dwarven_missing_die = 0
    casual_units = 0
    for a, p, _s in rep.action_log:
        p = p or {}
        if a == 'elven_ring':
            if p.get('collect'):
                sub['elven_collect'] += 1
                collect_lines.append(p.get('_rec_line'))
            else:
                sub['elven_fp'] += 1
                ring_lines.append(p.get('_rec_line'))
        elif a == 'discard_table_card' and p.get('payment') == 'ring':
            sub['discard_ring'] += 1
            ring_lines.append(p.get('_rec_line'))
        elif a == 'political':
            if p.get('activate'):
                sub['political_activate'] += 1
            else:
                sub['political_walk'] += 1
        elif a == 'dwarven_ring':
            if not p.get('die_used'):
                dwarven_missing_die += 1
            if p.get('placeholder_reroll'):
                sub['dwarven_placeholder'] += 1
        elif a == 'combat_casualties':
            rm = p.get('removed') or {}
            casual_units += sum(int(rm.get(k, 0) or 0)
                                for k in ('regular', 'elite', 'leader',
                                          'nazgul'))
        elif a == 'draw_card' and p.get('auto'):
            sub['draw_auto'] += 1
        elif a == 'muster' and p.get('auto'):
            sub['muster_auto'] += 1
    marks = getattr(rep, 'casualty_marks', None) or []
    mark_units = 0
    for mk in marks:
        spec = mk.get('spec') or {}
        mu = sum(int(spec.get(k, 0) or 0)
                 for k in ('kill_regular', 'kill_elite', 'downgrade'))
        mark_units += mu
    return {
        'counts': dict(counts),
        'sub': dict(sub),
        'ring_lines': [x for x in ring_lines if x],
        'collect_lines': [x for x in collect_lines if x],
        'dwarven_missing_die': dwarven_missing_die,
        'casualty_units': casual_units,
        'casualty_marks': len(marks),
        'casualty_mark_units': mark_units,
        'fp_rings_final': rep.state.fp_elven_rings,
        'sh_rings_final': rep.state.sh_elven_rings,
        'turn': rep.state.turn,
        'errors': len(rep.errors),
    }


def _parser_worker(path):
    try:
        return os.path.basename(path), parser_summarize(path)
    except Exception as e:
        return os.path.basename(path), {'error': '%s: %s'
                                        % (type(e).__name__, e)}


def load_cache():
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    cache[rec['file']] = rec['summary']
                except (ValueError, KeyError):
                    continue
    return cache


def append_cache(records):
    with open(CACHE_PATH, 'a', encoding='utf-8') as f:
        for fn, summary in records:
            f.write(json.dumps({'file': fn, 'summary': summary}) + '\n')


def run_parser(files, workers=4, refresh=False):
    if refresh and os.path.exists(CACHE_PATH):
        # append_cache opens in 'a' mode: without truncation every
        # refresh duplicates all records (last-write-wins masks it)
        open(CACHE_PATH, 'w').close()
    cache = {} if refresh else load_cache()
    todo = [p for p in files if os.path.basename(p) not in cache]
    if todo:
        print('parser: %d cached, %d to parse (workers=%d)'
              % (len(files) - len(todo), len(todo), workers))
        new = []
        if workers <= 1:
            for i, p in enumerate(todo):
                new.append(_parser_worker(p))
                if (i + 1) % 25 == 0:
                    print('  parsed %d/%d' % (i + 1, len(todo)))
                    append_cache(new)
                    cache.update(dict(new))
                    new = []
        else:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                done = 0
                for fn, summary in pool.imap_unordered(_parser_worker, todo,
                                                       chunksize=4):
                    new.append((fn, summary))
                    done += 1
                    if done % 25 == 0:
                        print('  parsed %d/%d' % (done, len(todo)))
                        append_cache(new)
                        cache.update(dict(new))
                        new = []
        if new:
            append_cache(new)
            cache.update(dict(new))
    return {os.path.basename(p): cache.get(os.path.basename(p))
            for p in files}


# ---------------------------------------------------------------------------
# Census driver
# ---------------------------------------------------------------------------
def census_game(path):
    ctx = prepare_game(path)
    out = {}
    out['ring'] = census_ring(ctx)
    out['ring_naive'] = census_ring_naive(ctx)
    out['pass'] = census_pass(ctx)
    out['nc'] = census_chat_token(ctx, 'nc')
    out['ff'] = census_chat_token(ctx, 'ff')
    out['dice_roll'] = census_dice_roll(ctx)
    out['hunt_tile_draw'] = census_hunt_tile_draw(ctx)
    out['draw_card'] = census_draw_card(ctx)
    out['eye_allocate'] = census_eye_allocate(ctx)
    out['fsp_track'] = census_fellowship_track(ctx)
    out['fsp_flip'] = census_fsp_flip(ctx)
    out['fsp_pawn_place'] = census_fsp_pawn_place(ctx)
    out['political'] = census_political(ctx)
    out['dwarven_ring'] = census_dwarven_ring(ctx)
    out['use_token'] = census_use_token(ctx)
    out['play_event'] = census_play_event(ctx)
    out['crown_flip'] = census_flip_piece(ctx, '241')
    out['companion_separate'] = census_companion_separate(ctx)
    out['choose_guide'] = census_choose_guide(ctx)
    out['companion_death'] = census_companion_death(ctx)
    out['muster'] = census_muster(ctx)
    out['army'] = census_army_moves(ctx)
    out['casualties'] = census_casualties(ctx)
    out['swap_place'] = census_swap_place(ctx)
    return out


# ---------------------------------------------------------------------------
# Comparison spec: census metric vs parser metric, per report row
# ---------------------------------------------------------------------------
def _pc(ps, *keys):
    """Sum of parser action-log counts."""
    c = ps.get('counts', {})
    return sum(int(c.get(k, 0) or 0) for k in keys)


def _sub(ps, key):
    return int(ps.get('sub', {}).get(key, 0) or 0)


REPORT_ROWS = [
    # name, census getter, parser getter, note
    ('elven_ring_fp_use',
     lambda cs: cs['ring']['fp_uses'],
     lambda ps: _sub(ps, 'elven_fp') + _sub(ps, 'discard_ring'),
     'census state-machine FP uses vs elven_ring(non-collect)+ring-paid '
     'Palantir kills'),
    ('elven_ring_fp_state',
     lambda cs: cs['ring']['fp_uses'],
     lambda ps: max(0, 3 - int(ps.get('fp_rings_final', 3))),
     'census FP uses vs 3 - final state fp_elven_rings'),
    ('elven_ring_collect',
     lambda cs: cs['ring']['sh_collects'],
     lambda ps: _sub(ps, 'elven_collect'),
     'census Shadow pickups vs elven_ring collect records'),
    ('pass',
     lambda cs: cs['pass']['count'],
     lambda ps: _pc(ps, 'pass'), 'chat-pass lines'),
    ('dice_roll',
     lambda cs: cs['dice_roll']['multi'] + cs['dice_roll']['single'],
     lambda ps: _pc(ps, 'dice_roll') + _sub(ps, 'dwarven_placeholder'),
     'all Controls.345 lines vs dice_roll records + the single-die '
     'rerolls the parser folds into dwarven_ring'),
    ('hunt_tile_draw',
     lambda cs: cs['hunt_tile_draw']['count'],
     lambda ps: _pc(ps, 'hunt_tile_draw'),
     'Controls.624 presses; parser skips post-game curiosity draws '
     '(ruling 1.157) so census-tail extras are expected'),
    ('draw_card',
     lambda cs: cs['draw_card']['count'],
     lambda ps: _pc(ps, 'draw_card'), 'deck-exit events'),
    ('eye_allocate',
     lambda cs: cs['eye_allocate']['count'],
     lambda ps: _pc(ps, 'eye_allocate'),
     'net pre-roll hunt-box entries + 372 presses'),
    ('fellowship_move',
     lambda cs: cs['fsp_track']['advance_p1'],
     lambda ps: _pc(ps, 'fellowship_move'), '+1 progress-track walks'),
    ('mordor_move',
     lambda cs: cs['fsp_track']['mordor_steps'],
     lambda ps: _pc(ps, 'mordor_move'), '+1 mordor-track walks'),
    ('fellowship_hide_reveal',
     lambda cs: cs['fsp_flip']['count_net'],
     lambda ps: _pc(ps, 'fellowship_hide', 'fellowship_reveal'),
     'APPROX: die-less-pair-netted flips vs hide+reveal records; the '
     'parser legitimately folds flips into declares / reveal '
     'placements / hunt auto-reveals, so census >= parser is expected'),
    ('fsp_pawn_place',
     lambda cs: cs['fsp_pawn_place']['count'],
     lambda ps: _pc(ps, 'fellowship_declare', 'fellowship_reveal_place'),
     'pawn onto map vs declare+reveal_place (aggregate; the split is '
     'context-only)'),
    ('political_walk',
     lambda cs: cs['political']['walks'],
     lambda ps: _sub(ps, 'political_walk'),
     'marker walks between track boxes'),
    ('political_activate',
     lambda cs: cs['political']['flips'],
     lambda ps: _sub(ps, 'political_activate'),
     'activation flips; parser dedupes already-active flips so census '
     'may run higher'),
    ('dwarven_ring',
     lambda cs: cs['dwarven_ring']['uses'],
     lambda ps: _pc(ps, 'dwarven_ring') - _sub(ps, 'dwarven_placeholder'),
     'piece-dialect uses only (placeholder rerolls have no drag)'),
    ('use_token',
     lambda cs: cs['use_token']['count'],
     lambda ps: _pc(ps, 'use_token'), 'bid-token uses'),
    ('play_event',
     lambda cs: cs['play_event']['count'],
     lambda ps: _pc(ps, 'play_event'),
     'face-up hand->play-zone + CardBuffet reveals (commit-grammar '
     'reveals excluded: shared with combat cards)'),
    ('card_reveals_all',
     lambda cs: cs['play_event']['count'] + cs['play_event']['commit_reveals'],
     lambda ps: _pc(ps, 'play_event', 'combat_card'),
     'every reveal dialect vs play_event + combat_card records '
     '(aggregate check)'),
    ('crown_aragorn',
     lambda cs: cs['crown_flip']['count'],
     lambda ps: _pc(ps, 'crown_aragorn'), 'piece-241 flips'),
    ('companion_separate',
     lambda cs: cs['companion_separate']['count'],
     lambda ps: _pc(ps, 'companion_separate'),
     'companions out of Fellowship/Guide box (piece count)'),
    ('choose_guide',
     lambda cs: cs['choose_guide']['count'],
     lambda ps: _pc(ps, 'choose_guide'), 'net Guide-Box entries'),
    ('companion_death_family',
     lambda cs: cs['companion_death']['count'],
     lambda ps: _pc(ps, 'companion_death', 'hunt_sacrifice_guide',
                    'hunt_sacrifice_random'),
     'companions into dead piles vs death+sacrifice records'),
    ('muster_gesture',
     lambda cs: cs['muster']['gestures'],
     lambda ps: _pc(ps, 'muster'),
     'reinforcement->map drag gestures vs muster records (records also '
     'cover swap-place and card placements: aggregate check only)'),
    ('muster_plus_swap',
     lambda cs: cs['muster']['gestures'] + cs['swap_place']['gestures'],
     lambda ps: _pc(ps, 'muster'),
     'reinforcement drags + Controls.313 placements vs muster records'),
    ('army_gesture',
     lambda cs: cs['army']['army_gestures'],
     lambda ps: _pc(ps, 'army_move', 'battle_advance', 'combat_retreat'),
     'map->map army drags vs parser move/advance/retreat records '
     '(army_attack excluded: adjacent attacks need no drag; '
     'classification is context-only; aggregate check)'),
    ('nazgul_move',
     lambda cs: cs['army']['nazgul_only'],
     lambda ps: _pc(ps, 'nazgul_move'), 'nazgul-only map->map drags'),
    ('retreat_into_siege',
     lambda cs: cs['army']['siege_entries'],
     lambda ps: _pc(ps, 'retreat_into_siege'),
     'map->stronghold-box drags'),
    ('combat_casualty_units',
     lambda cs: cs['casualties']['units'],
     lambda ps: ps.get('casualty_units', 0),
     'army pieces removed (drags+kills+swaps) vs combat_casualties '
     'record sums -- drag-based removals never emit this record '
     '(marks are the modern currency, see casualty_units_vs_marks)'),
    ('casualty_units_vs_marks',
     lambda cs: cs['casualties']['units'],
     lambda ps: max(ps.get('casualty_mark_units', 0),
                    ps.get('casualty_units', 0)),
     'army pieces removed vs max(mark spec units, record sums) -- the '
     'kill path emits BOTH a record and a mark, so the max avoids the '
     'double count (marks only exist inside battle windows)'),
    ('combat_card_none',
     lambda cs: cs['nc']['count'],
     lambda ps: _pc(ps, 'combat_card_none'),
     'nc chats; parser gates by battle-window proximity'),
    ('combat_effect_choice_ff',
     lambda cs: cs['ff']['count'],
     lambda ps: _pc(ps, 'combat_effect_choice'),
     'ffN chats; parser has additional non-ff effect-choice sources'),
]

# Per-row verdicts from the 2026-08 500-game triage (kept with the row
# results in the JSON report; update after re-triage when the parser or
# census changes).
VERDICTS = {
    'elven_ring_fp_use': ('PARSER-BUG-SUSPECTED',
                          'phantom uses on active->active tidy drags '
                          '(log_replay.py:3454 lacks a to-in-active '
                          'guard) and no netting of quick returns'),
    'elven_ring_collect': ('PARSER-BUG-SUSPECTED',
                           'used->active returns and used->used tidies '
                           'recorded as Shadow collects '
                           '(log_replay.py:3473, 4019); collects can '
                           'exceed FP uses in parser output'),
    'elven_ring_fp_state': ('PARSER-BUG-SUSPECTED',
                            'fp_elven_rings never restored on ring '
                            'returns; clipped at 0 by max()'),
    'pass': ('MATCH', ''),
    'crown_aragorn': ('MATCH', ''),
    'dice_roll': ('MATCH', ''),
    'fellowship_move': ('PARSER-BUG-SUSPECTED',
                        'two handlers for progress-marker drags; the '
                        'sync-only path (log_replay.py:4693) records NO '
                        'fellowship_move for real +1 advances (2025-07-03 '
                        'Eazy_kapcerz: 10 advances, 0 records)'),
    'companion_separate': ('MATCH', ''),
    'nazgul_move': ('REGEX-LIMITATION',
                    'census excludes WK/Saruman/Mouth flights and '
                    'stronghold-box codes Game.20/21; no parser phantoms'),
    'army_gesture': ('REGEX-LIMITATION',
                     'census piece table stops at pid 357; expansion '
                     'block 460-502/365/366/442 dropped; attack/advance '
                     'split is context-only'),
    'retreat_into_siege': ('REGEX-LIMITATION',
                           'census SIEGE_BOX_CODES misses Game.20/21/'
                           '22/23/26'),
    'mordor_move': ('PARSER-SUSPECTED-MILD',
                    'parser records backward Mordor-track oscillation '
                    'drags (from_step > to_step, game-illegal) as '
                    'mordor_move; census counts forward steps only'),
    'fsp_pawn_place': ('REGEX-LIMITATION',
                       'declare-in-place via corruption-marker walks, '
                       'Mordor-entry declares, and pawn+army group drags '
                       'are invisible to the pawn-only census'),
    'draw_card': ('EOG-NOISE', 'census extras are post-game curiosity '
                  'draws the parser skips (ruling 1.157)'),
    'hunt_tile_draw': ('EOG-NOISE', 'same as draw_card'),
    'combat_card_none': ('REGEX-LIMITATION', 'nc chats outside a battle '
                         'window are not decline decisions; the census '
                         'cannot see the window'),
    'combat_effect_choice_ff': ('REGEX-LIMITATION', 'ff chats are one of '
                                'several effect-choice sources'),
    'fellowship_hide_reveal': ('REGEX-LIMITATION', 'parser folds flips '
                               'into declares/reveal placements/hunt '
                               'auto-reveals'),
    'combat_casualty_units': ('PARSER-REPRESENTATION', 'drag-based '
                              'removals emit casualty MARKS, not '
                              'combat_casualties records; whole games '
                              'have zero records despite battles'),
    'use_token': ('PARSER-BUG-SUSPECTED',
                  'FP die-consumption interceptions (log_replay.py:3227 '
                  'Controls.271 grammar) swallow token pids 426/427 '
                  'before _handle_token_move: Controls.271-dialect token '
                  'uses are never recorded (~66 uses in 500 games)'),
    'dwarven_ring': ('GRAMMAR-ROUTING-ASYMMETRY',
                     'ring pids 428-431 riding the die-consumption '
                     'grammars (log_replay.py:3201/3227) are swallowed '
                     '(252/257 of the delta); the ring doubles as a '
                     'rolled die in placeholder clients so routing is '
                     'defensible but lossy; die_used missing on 92.2% '
                     'of dwarven_ring records (741/804) - structural'),
    'play_event': ('REGEX-LIMITATION',
                   'combat-card commits share the reveal grammar (~84% '
                   'of extras); census cannot see the battle window'),
    'political_walk': ('REGEX-LIMITATION',
                       'census drops multi-nation group walks'),
    'political_activate': ('REGEX-LIMITATION',
                           'census counts every flip; parser nets '
                           'flip-undo pairs and already-active flips'),
    'eye_allocate': ('MATCH', 'bidirectional boundary cases (bonus dice, '
                     'ring-as-die, undo churn), near-zero net'),
    'muster_gesture': ('REGEX-LIMITATION',
                       'census piece table stops at pid 357 (~50-84% of '
                       'gap = 452+ reinforcement pool ids); staged '
                       'reinf->card-box->map placements; per-unit 313 '
                       'records; minor parser imperfection: staged '
                       'map->2037->map relocations booked as muster'),
    'muster_plus_swap': ('REGEX-LIMITATION',
                         'only ~11.5% of Controls.313 placements are '
                         'musters; 88.5% are battle/card downgrade '
                         'replacements the parser correctly keeps '
                         'state-only'),
    'casualty_units_vs_marks': ('MATCH',
                                '83% of census casualties sit within 4 '
                                'lines of a parser mark; residual ~14% = '
                                'real non-battle removals that have no '
                                'mark by design (state-only)'),
    'companion_death_family': ('PARSER-BUG-SUSPECTED',
                               'consecutive Mordor-track sacrifices '
                               'dropped (hunt-resolution latch not '
                               're-armed, log_replay.py:7303-7315); '
                               'census side also overcounts re-drags/'
                               'Gollum/leader-cas kills'),
    'choose_guide': ('REGEX-LIMITATION',
                     'census nets guide SWAPS to zero (new guide in + '
                     'old guide out); parser correctly records each'),
    'card_reveals_all': ('MATCH', 'aggregate reconciliation of all '
                         'reveal dialects vs play_event+combat_card'),
}

# Parser defects the census believes are REAL (2026-08 triage; report,
# do not fix here).  Each entry: locus, mechanism, example games with
# line numbers (stripped non-empty numbering, = parser _rec_line).
PARSER_DEFECTS = {
    'P1_use_token_swallowed': {
        'locus': 'log_replay.py:3227 (Controls.271 grammar) and :3201 '
                 '(HighlightLabel.86) FP die-consumption interceptions',
        'mechanism': 'the interception captures ANY pid, no-ops '
                     '_track_die_consumed for non-die pids (426/427 not '
                     'in die_id_map) and returns, so Controls.271-'
                     'dialect token uses never reach _handle_token_move '
                     '(~66 unrecorded uses / 500 games; dwarven-ring '
                     'pids 428-431 share the mechanism, 252 drags)',
        'examples': [
            '2023-01-10_141039_FP_DerGuy_SP_Sonnyl.log L206 (pid 426), '
            'L303 (pid 427): Controls.271 Game.2042->Game.2035, parser '
            'use_token count 0',
            '2025-12-12_222536_FP_joe_tschu_SP_SlickRick441.log L372, '
            'L611',
            '2024-06-30_180803_FP_GJB_SP_gus833.log L362, L446'],
    },
    'P2_fellowship_move_dropped': {
        'locus': 'log_replay.py:4693-4703 (sync-only marker-drag path) '
                 'vs :7574-7578 (recording path)',
        'mechanism': 'inconsistent dispatch: a subset of real +1 '
                     'progress advances hit the sync-only path and emit '
                     'NO fellowship_move record',
        'examples': [
            '2025-07-03_120717_FP_Eazy_SP_kapcerz.log: 10 real advances '
            '(L144,179,188,246,295,304,312,367,396,476), 0 records',
            '2023-11-09 Thistlefoot game: L379, L941, L1031 dropped '
            '(Track0->Track1 legs)'],
    },
    'P3_elven_ring_branch': {
        'locus': 'log_replay.py:3427 (len==1 guard), :3454 (active-> '
                 'any = use), :3473 and :4019 (used->any = collect)',
        'mechanism': 'R1: active->active slot tidies recorded as FP '
                     'uses; R2: used->active returns and used->used '
                     'tidies recorded as Shadow collects (collects can '
                     'exceed uses; fp_elven_rings never restored, '
                     'clipped by max(0,...)); R3: multi-ring drags '
                     'bypass the ring branch entirely. 32/500 games '
                     'diverge; census invariant <=6 holds in 100%',
        'examples': [
            "R1: 2024-04-04_191632_FP_Mistakentuna_SP_Jaymanen.log "
            "L1312 (251 Game.2008->Game.2010) -> phantom elven_ring use",
            "R2: 2025-07-06_191415_FP_Alexey'a_SP_Rexxarchik.log L804/"
            "L806 returns -> collect records; parser collects 4 > uses "
            "3 (impossible); final board has 2 active rings, parser "
            "state fp=0",
            'R2: 2025-02-26_130841_FP_Spartako_SP_savedbyhim01.log '
            'L753 (2013->2010 return booked as collect)',
            'R2c: 2025-09-07_023751_FP_davidmand_SP_SilmarillWill.log '
            'L894 (2012->2011 tidy booked as collect)',
            'R3: Mistakentuna L1311 (249 251 dragged together, ring '
            'branch skipped)'],
    },
    'P4_dead_marshes_companion_death': {
        'locus': "log_replay.py:7272 death_areas = ['Card Play', "
                 "'Dead', 'Casualties', 'Current Action'] + :7278 "
                 'substring match',
        'mechanism': "'Dead' in 'Dead Marshes' -> a companion "
                     'SEPARATING to the Dead Marshes region is recorded '
                     'as companion_death and removed. 34 affected games '
                     'found in the first 4000 corpus logs scanned '
                     '(Dead Marshes is on the natural path to Mordor)',
        'examples': [
            '2023-01-18_140317_FP_Franatleti_SP_Lancer.log L632: '
            'Boromir (244) Game.1998->Game.1959, parser records '
            'companion_death (PROVEN by replay)',
            '2023-01-27_015435_FP_GrayPockets_SP_Matzerotti.log L1011: '
            'Peregrin (246) Guide Box->Dead Marshes',
            '2023-02-21_200620_FP_Amroth_SP_Sonnyl.log L929: 245+246 '
            'group separation Guide Box->Dead Marshes'],
    },
    'P5_mordor_sacrifices_dropped': {
        'locus': 'log_replay.py:7303-7315 hunt-resolution latch',
        'mechanism': 'consecutive/late Mordor-track companion '
                     'sacrifices with the identical 644 <pid> <tile> -> '
                     'Used grammar are only partially recorded (latch '
                     'not re-armed); medium confidence',
        'examples': [
            '2023-09-27_003626_FP_PeterM_SP_Terminatorhenk.log: L778 '
            'recorded (Strider) but L792 (Legolas + tile 238) and '
            'L1058-1060 (Peregrin) missed; 5/7 recorded',
            '2023-09-29_044452_FP_lanxing_SP_Thistlefoot.log: L937 '
            '(Legolas), L915 (Peregrin) missed; 3/5 recorded'],
    },
    'P6_mordor_move_backward': {
        'locus': 'log_replay.py:~7668 Mordor-track handler',
        'mechanism': 'no direction check: backward oscillation drags '
                     '(from_step > to_step, game-illegal) are recorded '
                     'as mordor_move; mild (human re-drag noise)',
        'examples': [
            '2025-02-23 Gotti16850_TheLastNumenorean game: L934, '
            'L1042, L1113 each Mordor4->Mordor3',
            '2024-01-21 Disgruntled_Feanor game: L904, L1119, L1243'],
    },
    'P7_staged_relocation_as_muster': {
        'locus': 'muster path (card-box staging)',
        'mechanism': 'map->Game.2037->map staged relocations: pickup '
                     'leg silently dropped, drop leg booked as muster '
                     '(a move mislabeled; board state stays correct); '
                     'minor, rare',
        'examples': [
            '2023-02-16 jallendaws_ButterflySlayer game L89-90 '
            '(032 031 030 Gorgoroth->2037->Minas Morgul)',
            '2024-12-21 Feanor_Upward game L1832-1833'],
    },
    'P8_casualty_representation': {
        'locus': '_remove_pieces_from_armies (drag path) emits marks '
                 'only; non-battle removals emit neither',
        'mechanism': 'REPRESENTATION note, not a bug: drag-based '
                     'casualties produce casualty MARKS (no '
                     'combat_casualties record; whole games have 0 '
                     'records despite 18 attacks but marks cover them); '
                     '~14% of removals are non-battle (card effects, '
                     'downgrades outside windows) and exist only as '
                     'state mutation - consumers must read marks + '
                     'state, never records alone',
        'examples': [
            '2024-09-28_065122_FP_Pallando_SP_Ira.log: 18 attacks, 0 '
            'combat_casualties records, 39 marks / 53 mark-units'],
    },
}

# Action types with NO reliable independent raw signature (documented,
# not forced into a weak regex):
NO_SIGNATURE_TYPES = {
    'combat_start': 'synthesized from chat hints + roll windows',
    'army_attack': 'attack-vs-move split is battle-context only',
    'battle_advance': 'advance-vs-move split is battle-context only',
    'combat_card': 'reveal grammar shared with event plays; battle '
                   'window context required',
    'combat_effect_choice': 'ff chats censused; other magnitude choices '
                            'are unit-drag context',
    'fellowship_declare': 'declare-vs-reveal-place split needs hunt '
                          'context (censused in aggregate)',
    'fellowship_reveal_place': 'see fellowship_declare',
    'hunt_take_corruption': 'corruption walks are effects; the decision '
                            'subset is context-only',
    'siege_press': 'elite-downgrade press uses the swap grammar shared '
                   'with ordinary casualties',
    'companion_move': 'die-vs-continuation split is schedule context',
    'transform_gandalf': 'map return of dead piece 240; needs death '
                         'state (flip census is only a weak proxy)',
    'hunt_sacrifice_guide': 'guide-vs-random needs guide state '
                            '(censused in death aggregate)',
    'mordor_track_entry': 'implied declares synthesized by parser',
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def select_files(sample):
    files = sorted(f for f in os.listdir(LOG_DIR)
                   if f.endswith('.log') and 'replaynav' not in f)
    if sample and sample < len(files):
        step = len(files) / float(sample)
        idx = sorted({int(i * step) for i in range(sample)})
        files = [files[i] for i in idx]
    return [os.path.join(LOG_DIR, f) for f in files]


def compare(census_by_game, parser_by_game, types=None):
    rows = {}
    for name, cget, pget, note in REPORT_ROWS:
        if types and name not in types:
            continue
        per_game = []
        getter_errors = 0
        for fn, cs in census_by_game.items():
            ps = parser_by_game.get(fn)
            if not ps or 'error' in ps:
                continue
            try:
                c = int(cget(cs))
                p = int(pget(ps))
            except Exception:
                getter_errors += 1
                continue
            per_game.append((fn, c, p))
        if not per_game:
            continue
        diffs = [abs(c - p) for (_f, c, p) in per_game]
        div_games = [(f, c, p) for (f, c, p) in per_game if c != p]
        div_games.sort(key=lambda t: -abs(t[1] - t[2]))
        v = VERDICTS.get(name)
        rows[name] = {
            'note': note,
            'verdict': v[0] if v else 'UNTRIAGED',
            'verdict_note': v[1] if v else '',
            'games': len(per_game),
            'getter_errors': getter_errors,
            'census_total': sum(c for (_f, c, _p) in per_game),
            'parser_total': sum(p for (_f, _c, p) in per_game),
            'games_diverging': len(div_games),
            'divergence_rate': round(len(div_games) / len(per_game), 4),
            'median_abs_divergence': statistics.median(diffs) if diffs else 0,
            'top_offenders': [
                {'file': f, 'census': c, 'parser': p}
                for (f, c, p) in div_games[:12]],
        }
    return rows


def ring_deep_report(census_by_game, parser_by_game):
    dist_strict = collections.Counter()
    dist_loose = collections.Counter()
    dist_naive = collections.Counter()
    naive_inflated = 0
    tot_fp = tot_col = tot_proven = tot_regrant = 0
    div = []
    for fn, cs in census_by_game.items():
        r = cs['ring']
        rn = cs['ring_naive']
        dist_strict[min(r['total_strict'], 9)] += 1
        dist_loose[min(r['total_loose'], 9)] += 1
        dist_naive[min(rn['total'], 12)] += 1
        tot_fp += r['fp_uses']
        tot_col += r['sh_collects']
        tot_proven += r['sh_uses_proven']
        tot_regrant += r.get('regrants', 0)
        if rn['total'] > r['total_loose']:
            naive_inflated += 1
        ps = parser_by_game.get(fn)
        if not ps or 'error' in ps:
            continue
        p_fp = _sub(ps, 'elven_fp') + _sub(ps, 'discard_ring')
        p_col = _sub(ps, 'elven_collect')
        d = abs(r['fp_uses'] - p_fp) + abs(r['sh_collects'] - p_col)
        if d:
            div.append({
                'file': fn, 'abs_div': d,
                'census_fp': r['fp_uses'], 'parser_fp': p_fp,
                'census_collect': r['sh_collects'], 'parser_collect': p_col,
                'census_fp_lines': r['fp_lines'],
                'census_collect_lines': r['collect_lines'],
                'parser_ring_lines': ps.get('ring_lines', []),
                'parser_collect_lines': ps.get('collect_lines', []),
                'fp_state_final': ps.get('fp_rings_final'),
            })
    div.sort(key=lambda d: -d['abs_div'])
    n = len(census_by_game)
    within6_strict = sum(v for k, v in dist_strict.items() if k <= 6)
    within6_naive = sum(v for k, v in dist_naive.items() if k <= 6)
    return {
        'games': n,
        'census_fp_uses': tot_fp,
        'census_sh_collects': tot_col,
        'census_sh_uses_proven': tot_proven,
        'census_ring_regrants': tot_regrant,
        'total_strict_distribution': {str(k): dist_strict[k]
                                      for k in sorted(dist_strict)},
        'total_loose_distribution': {str(k): dist_loose[k]
                                     for k in sorted(dist_loose)},
        'naive_total_distribution': {str(k): dist_naive[k]
                                     for k in sorted(dist_naive)},
        'pct_within_6_strict': round(100.0 * within6_strict / n, 2) if n else 0,
        'pct_within_6_naive': round(100.0 * within6_naive / n, 2) if n else 0,
        'games_naive_inflated': naive_inflated,
        'divergent_games': len(div),
        'worst_20': div[:20],
    }


def print_table(rows):
    hdr = '%-26s %6s %9s %9s %7s %7s %7s  %s' % (
        'type', 'games', 'census', 'parser', 'ndiv', 'rate', 'medad',
        'verdict')
    print(hdr)
    print('-' * len(hdr))
    for name in sorted(rows):
        r = rows[name]
        print('%-26s %6d %9d %9d %7d %6.1f%% %7.1f  %s' % (
            name, r['games'], r['census_total'], r['parser_total'],
            r['games_diverging'], 100 * r['divergence_rate'],
            r['median_abs_divergence'], r.get('verdict', '')))


def main(argv=None):
    ap = argparse.ArgumentParser(description='raw-log census harness')
    ap.add_argument('--sample', type=int, default=500)
    ap.add_argument('--types', type=str, default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-parser', action='store_true',
                    help='census only (skip/ignore parser comparison)')
    ap.add_argument('--refresh-cache', action='store_true')
    ap.add_argument('--out', type=str, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    types = set(t.strip() for t in args.types.split(',') if t.strip()) or None
    files = select_files(args.sample)
    print('census over %d games (deterministic stride sample)' % len(files))

    census_by_game = {}
    for i, p in enumerate(files):
        try:
            census_by_game[os.path.basename(p)] = census_game(p)
        except Exception as e:
            print('  census FAILED %s: %s' % (os.path.basename(p), e))
        if (i + 1) % 100 == 0:
            print('  censused %d/%d' % (i + 1, len(files)))

    parser_by_game = {}
    if not args.no_parser:
        parser_by_game = run_parser(files, workers=args.workers,
                                    refresh=args.refresh_cache)
        parser_by_game = {k: v for k, v in parser_by_game.items()
                          if v is not None}

    rows = compare(census_by_game, parser_by_game, types)
    ring = ring_deep_report(census_by_game, parser_by_game)

    report = {
        'meta': {
            'games': len(census_by_game),
            'sample_arg': args.sample,
            'parser_games': len(parser_by_game),
            'no_signature_types': NO_SIGNATURE_TYPES,
        },
        'ring_deep_dive': ring,
        'types': rows,
        'parser_defects': PARSER_DEFECTS,
    }
    _outd = os.path.dirname(args.out)
    if _outd:
        os.makedirs(_outd, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print('\n=== RING DEEP DIVE ===')
    print('games %d | within-6 strict %.1f%% | naive within-6 %.1f%% | '
          'naive-inflated games %d | divergent-vs-parser %d'
          % (ring['games'], ring['pct_within_6_strict'],
             ring['pct_within_6_naive'], ring['games_naive_inflated'],
             ring['divergent_games']))
    print('strict total distribution: %s'
          % ring['total_strict_distribution'])
    print('naive  total distribution: %s'
          % ring['naive_total_distribution'])
    print('\n=== PER-TYPE DIVERGENCE TABLE ===')
    print_table(rows)
    print('\nreport written: %s' % args.out)
    return report


if __name__ == '__main__':
    main()
