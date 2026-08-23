"""Human gesture window -> engine action (design rule 4).

THE USER RULING THAT SHAPES THIS MODULE (quoted in the monolith at
:5433-5440): gesture legality is judged by simulate-and-compare over
legal moves — the engine is the sole judge.  What this module fixes is
the COST and the SCOPE, not the ruling: the old matcher applied EVERY
legal action with a depth-3 pending expansion over the full action
space (measured 68,731 applies / 48.1s for ONE drag, 60,000-apply
budget as band-aid).  Here:

  * candidates come from the CURRENT engine menu only
    (ai.get_legal_actions / ai.describe_pending) — tens of actions;
  * comparison is DELTA-based on the lifted projection
    (client_board.board_delta, increment semantics) so a one-tick view
    offset cannot refuse a legal action;
  * pending-chain expansion happens ONLY while the window still has
    unexplained content, depth-capped, apply-budgeted; exhaustion
    HOLDS, never refuses (B13-1 doctrine);
  * identity-carrying gestures (a card play names its title, a ring
    walk names its marker) are matched by VOCABULARY first — no
    simulation needed when the client already said which action it is.

Every apply runs on a CLONE (ai.apply_action semantics: original
untouched) — the transactional tick (design rule 2).  A match's
post-state is returned so the driver COMMITS by adopting it, never by
re-applying.
"""
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rendered_replay as rr
from board import sanitize

from . import client_board


# ---------------------------------------------------------------------------
# zone classification (client vocabulary -> engine key space)
# ---------------------------------------------------------------------------
_HAND_ZONES = {'free peoples hand': 'FP', 'shadow hand': 'SH'}
# keys are SANITIZED forms (board.sanitize folds the parentheses)
_HAND_CHOSEN_ZONES = {'free peoples hand chosen': 'FP',
                      'shadow hand chosen': 'SH'}
_PLAY_AREAS = {'current fp card': 'FP', 'current sa card': 'SH'}
_REINF = {'fp reinforcements': 'FP', 'sa reinforcements': 'SH',
          'reinforcements': None}
_DEADISH = {'dead', 'casualties', 'fp casualties', 'sa casualties',
            'fp dead', 'sa dead'}
_FS_TRACK_RE = re.compile(r'^fellowship track (\d+)$')
_POL_TRACK_RE = re.compile(r'^political track (\d+)$')
_STRONGHOLD_RE = re.compile(r'^(.*) stronghold$')


def zone_key(zone):
    """('region', key) / ('stronghold', region_key) / ('hand', side) /
    ('play', side) / ('reinf', side|None) / ('dead',) /
    ('fs_track', n) / ('pol_track', n) / ('other', sanitized).
    The region test is board.regions membership after sanitize — the
    same equivalence the corpus parser uses."""
    z = sanitize(str(zone or '')).strip()
    zl = z.lower()
    if zl in _HAND_ZONES:
        return ('hand', _HAND_ZONES[zl])
    if zl in _HAND_CHOSEN_ZONES:
        # the client's card-chooser reveal source (combat cards):
        # 'auto> moves TITLE from Free Peoples Hand (chosen) to ...'
        return ('hand_chosen', _HAND_CHOSEN_ZONES[zl])
    if zl.endswith(' chosen'):
        # the chooser let the human 'choose' a card that was NOT in
        # a hand (the casualties / discard pile: 'FP Casualties
        # (chosen)') - a reveal of a card the side does not hold
        return ('chosen_other',
                'FP' if zl.startswith(('fp ', 'free ')) else
                'SH' if zl.startswith(('sa ', 'shadow ')) else None)
    if zl in _PLAY_AREAS:
        return ('play', _PLAY_AREAS[zl])
    if zl in _REINF:
        return ('reinf', _REINF[zl])
    if zl in _DEADISH:
        return ('dead',)
    m = _FS_TRACK_RE.match(zl)
    if m:
        return ('fs_track', int(m.group(1)))
    m = _POL_TRACK_RE.match(zl)
    if m:
        return ('pol_track', int(m.group(1)))
    if zl.startswith('mordor'):
        m = re.search(r'mordor\s+(\d+)', zl)
        return ('mordor', int(m.group(1)) if m else 0)
    try:
        rk = region_lookup(zl)
        if rk is not None:
            return ('region', rk)
        m = _STRONGHOLD_RE.match(zl)
        if m:
            rk = region_lookup(m.group(1).strip())
            if rk is not None:
                return ('stronghold', rk)
    except Exception:
        pass
    return ('other', zl)


def region_lookup(zl):
    """Sanitized client zone text -> canonical engine region key, or
    None.  The client labels some regions with a definite article the
    canonical key lacks ('The Grey Havens' vs 'grey havens' — found on
    the Gandalf-the-White window, game 2023-01-13 T2) while 'the
    shire' keeps its article; membership is tried both ways."""
    import board
    if zl in board.regions:
        return zl
    if zl.startswith('the ') and zl[4:] in board.regions:
        return zl[4:]
    return None


def _unit_type_key(nation, kind):
    """The projection's army type key, built by parity_test's OWN
    helper so the two sides cannot drift (single answerer)."""
    import parity_test as pt
    return '%s:%s' % (str(pt._u(nation)).lower(),
                      str(pt._u(kind)).lower())


_NATION_ENUM = None


def _nation_enum(name):
    global _NATION_ENUM
    if _NATION_ENUM is None:
        import game_state as gs
        _NATION_ENUM = {}
        for n in gs.Nation:
            _NATION_ENUM[str(n.name).upper()] = n
            _NATION_ENUM[str(n.value).upper()] = n
    return _NATION_ENUM.get(str(name).upper())


# ---------------------------------------------------------------------------
# window -> expressed deltas
# ---------------------------------------------------------------------------
class WindowFacts(object):
    """What the window's records EXPRESS, in engine key space.

    armies: Counter[(region_key, type_key)] -> signed unit count
    leaders: Counter[region_key] -> signed generic-leader count
    named: Counter[(name, region_key)] -> signed (arrivals +1 / -1)
    fs_progress: net Fellowship Track movement (FSP marker moves)
    corruption: net Corruption marker movement (same client track)
    draws: Counter[(side_name, deck)] -> deck draws the window
           expressed ('draws Shadow Strategy Card.' — the event-die
           draw action and card-effect draw grants; hunt-tile and
           random-companion 'draws' are RANDOMNESS records, not intent)
    political: Counter[nation_enum_NAME] -> engine-direction political
           increments.  CLIENT track numbers DESCEND toward war
           ('Political Track 0' = At War box; engine boxes ASCEND,
           3 = At War), so engine_increment = client_from - client_to.
    fel_flips: [(region_key, 'flip'|'reveal')] fellowship flips in
           record order.  A plain flip is the human's hide/declare
           gesture; a reveal-style flip ('... Previously on Fellowship
           Track N') is the hunt/engine consequence AND expresses the
           track reset (fs_progress -= prev_track) — the client renders
           no separate FSP move for it.
    fel_moves: [(from_key, to_key)] 'The Fellowship' pawn region moves
           (the reveal/declare placement path; final entry names the
           destination the engine must land on)
    guide_set: canonical companion name moved onto 'Guide Of The
           Fellowship' (the guide-change gesture), or None
    comps_gone: Counter[name] companions killed OUTSIDE regions
           (Current Action / casualties zones) — hunt/effect casualties
    hand_discards: Counter[(side_name, deck)] identity-less card kills
           in a hand zone ('kills Shadow Character Card in Shadow
           Hand.' — the hand-limit discard gesture)
    cards_played: [card title strings] (hand -> play area, identity)
    combat_cards: [card title strings] chosen via the client's card
           chooser ('... Hand (chosen)' source) — played as COMBAT
           cards inside a battle chain, never as play_event actions
    ring_moves: [(marker, from, to)]
    sieges: Counter[region_key] -> +1 the garrison withdrew INTO the
           stronghold box (siege formed), -1 it came back out (siege
           lifted/sortied).  The client's 'moves ... from X to X
           Stronghold' drag is the defender's siege-choice GESTURE; in
           plain army rows it cancels to zero, which let a committed
           chain stop BEFORE resolving defender_siege_choice and the
           open pending then ate the next windows (game 2023-01-13:
           the Corsairs commit left the Dol Amroth choice open and the
           Witch-King muster window died 'window_while_pending').
    unmapped: records this converter could not express (LOUD — the
              matcher treats a window with unmapped records as
              hold-for-review, never silently partial)
    """

    def __init__(self):
        self.armies = Counter()
        self.leaders = Counter()
        self.named = Counter()
        self.fs_progress = 0
        self.corruption = 0
        self.draws = Counter()
        self.political = Counter()
        self.fel_flips = []
        self.fel_moves = []
        self.guide_set = None
        self.comps_gone = Counter()
        self.hand_discards = Counter()
        self.cards_played = []
        self.combat_cards = []
        self.ring_moves = []
        self.sieges = Counter()     # region -> siege formed/lifted
        self.rings = Counter()      # 'FP'/'SH' ring-count deltas
        self.dice_faces = Counter()  # (side_name, letter) -> UNUSED-
        #   face multiset delta: die_change records (-frm/+to) plus
        #   the staged die's consumption at its FINAL letter (ring/
        #   effect flips chase the staged die through the window)
        self.unmapped = []

    def is_empty(self):
        return not (self.armies or self.leaders or self.named
                    or self.fs_progress or self.corruption
                    or self.draws or self.political or self.fel_flips
                    or self.fel_moves or self.guide_set
                    or self.comps_gone or self.hand_discards
                    or self.cards_played or self.combat_cards
                    or self.ring_moves or self.sieges)


def _apply_piece(facts, piece_tok, zk, sign):
    """One piece token entering (+1) or leaving (-1) a board zone."""
    pp = rr.parse_piece(piece_tok)
    if pp is None:
        facts.unmapped.append(piece_tok)
        return
    kind, a, b, count = pp
    if zk[0] == 'stronghold':
        # the stronghold sub-box is the same engine region (siege state
        # is engine-derived); count pieces against the region
        zk = ('region', zk[1])
    if zk[0] == 'mordor':
        # the Mordor-track boxes physically overlay GORGOROTH — army
        # pieces parked on them alias to that region (the corpus
        # Mirror's own task-documented alias, rendered_replay:392)
        zk = ('region', 'gorgoroth')
    if zk[0] != 'region':
        # entering/leaving hands, reinforcements, dead piles etc. is
        # expressed by the engine action itself (musters, casualties);
        # the region-side rows carry the signal the matcher compares
        return
    reg = zk[1]
    if kind == 'unit':
        nat = _nation_enum(a)
        import game_state as gs
        uk = (gs.UnitKind.REGULAR if b == 'REGULAR'
              else gs.UnitKind.ELITE)
        if nat is None:
            facts.unmapped.append(piece_tok)
            return
        facts.armies[(reg, _unit_type_key(nat, uk))] += sign * count
    elif kind == 'leader':
        facts.leaders[reg] += sign * count
    elif kind == 'nazgul':
        facts.leaders[reg] += sign * count
    elif kind == 'named':
        facts.named[(a, reg)] += sign * count
    elif kind in ('ctrlmarker', 'vp', 'tile', 'die', 'nation'):
        pass          # client bookkeeping the engine derives itself
    else:
        facts.unmapped.append(piece_tok)


_DECK_DRAW_RE = re.compile(
    r'^(free|shadow)\s+(character|strategy)\s+card$')
# the packaged build's FP active ring slots (see the ring-count rules)
_FP_RING_SLOTS = frozenset(('vilya slot', 'nenya slot', 'narya slot'))
_FS_MARKERS = {'fsp': 'fs_progress', 'corruption': 'corruption'}
_GUIDE_ZONE = 'guide of the fellowship'
_COMPANION_NAMES = frozenset(rr.COMP_ALIAS.values())
# flipped-to identity (lower) -> the OTHER face's canonical name (the
# piece the client moved before flipping it)
_FLIP_RENAME = {'gandalf the white': 'Gandalf the Grey',
                'gandalf the grey': 'Gandalf the White',
                'aragorn': 'Strider',
                'strider': 'Aragorn'}


def window_facts(window):
    """Convert a closed Window's body records to WindowFacts."""
    f = WindowFacts()
    staged = [getattr(window, 'die_face', None)]   # chased through
    #   die_change records: the client renames the staged die in place
    w_side = getattr(window, 'die_side', None)
    for rec in window.body_records():
        op = rec.get('op')
        if op == 'move':
            obj = str(rec.get('obj', '') or '')
            src = zone_key(rec.get('from'))
            dst = zone_key(rec.get('to'))
            # siege-choice gesture: the garrison steps between the
            # region and its OWN stronghold sub-box (net-zero in army
            # rows, so it gets its own fact row)
            if src[0] == 'region' and dst == ('stronghold', src[1]):
                f.sieges[src[1]] += 1
            elif dst[0] == 'region' and src == ('stronghold', dst[1]):
                f.sieges[dst[1]] -= 1
            if src[0] == 'fs_track' and dst[0] == 'fs_track':
                # the track carries TWO markers (FSP = progress,
                # Corruption) which the client also moves as a group
                delta = dst[1] - src[1]
                toks = (rr.split_group(obj[len('group ('):-1])
                        if obj.startswith('group (')
                        and obj.endswith(')') else [obj])
                for t in toks:
                    slot = _FS_MARKERS.get(t.strip().lower())
                    if slot == 'fs_progress':
                        f.fs_progress += delta
                    elif slot == 'corruption':
                        f.corruption += delta
                    else:
                        f.unmapped.append(t)
                continue
            if src[0] == 'pol_track' and dst[0] == 'pol_track':
                # client steps DESCEND toward war; engine boxes ASCEND
                pp = rr.parse_piece(obj)
                if pp is not None and pp[0] == 'nation':
                    f.political[pp[1]] += src[1] - dst[1]
                else:
                    f.unmapped.append(obj)
                continue
            if src[0] == 'hand' and (
                    dst[0] == 'play'
                    or (dst[0] == 'other'
                        and re.match(r'^board\d*$', dst[1]))):
                # hand -> play area OR hand -> table slot (BoardN):
                # persistent/table cards render straight onto the
                # table (Denethor's Folly specimen, game 2023-01-13
                # T3) — both are the card-play gesture.  Later
                # BoardN-to-BoardN walks are counters, not plays.
                # An EVENT play is always die-funded, so it exists
                # only inside a die window and only for the staged
                # die's own side.  Every other hand play is a COMBAT
                # card for the hand's side: the OTHER seat reacting
                # inside this window (game 2023-01-28 T2: the defender
                # played Scouts from the FP hand inside the Shadow
                # attack window), or ANY hand play with no die staged
                # (a PhaseWindow: the human's combat card dragged
                # while the engine's combat-card pending is open).
                # Zone-derived attribution (the hand names its
                # owner), never nick.
                if w_side is None or ((src[1] == 'FP')
                                      != (w_side == 'FREE_PEOPLES')):
                    f.combat_cards.append((src[1], obj))
                else:
                    f.cards_played.append(obj)
                continue
            if src[0] == 'hand_chosen':
                # the client chooser serves BOTH the hidden combat-
                # card reveal and the CardBuffet.9 event play, with
                # one rendering.  The window's stated context decides
                # (window.combat_cards_open, set by the live driver
                # from the engine: a combat-card phase open -> combat;
                # closed -> the staged side's own chooser reveal is
                # its event play).  Unstated (the corpus spectator:
                # both seats external, the battle lives inside the
                # attacker's window) keeps the combat reading.  The
                # OTHER seat's chooser reveal is always a combat card.
                cco = getattr(window, 'combat_cards_open', None)
                if cco is False and w_side and (
                        (src[1] == 'FP') == (w_side == 'FREE_PEOPLES')):
                    f.cards_played.append(obj)
                else:
                    f.combat_cards.append((src[1], obj))
                continue
            ol = obj.strip().lower()
            if ol == 'the fellowship' and src[0] == 'region' \
                    and dst[0] == 'region':
                # the pawn's placement path (reveal/declare)
                f.fel_moves.append((src[1], dst[1]))
                continue
            if str(rec.get('to', '')).strip().lower() == _GUIDE_ZONE:
                # guide-change gesture: a companion becomes the guide
                f.guide_set = rr.COMP_ALIAS.get(ol, obj.strip())
                continue
            frl = str(rec.get('from', '')).lower()
            tol = str(rec.get('to', '')).lower()
            if is_ring_track_move(obj, rec.get('from'), rec.get('to')):
                # RING-TRACK zones only.  The old test was 'ring' in
                # the from-zone, which swallowed every move out of
                # 'Fellowship Of The RING' — companion separations
                # classified as ring walks and expressed nothing
                # (game 2023-01-28 T1: the Strider crown route died
                # here, all four legs)
                f.ring_moves.append((obj, rec.get('from'),
                                     rec.get('to')))
                # ring COUNT deltas: an FP use walks the marker from
                # an Elven Ring slot to a Dark Ring slot; a Shadow use
                # spends it out of the Dark Ring slots.  ZONE NAMES
                # VARY BY MODULE BUILD (WAVE-I ring rehearsal): the
                # 2023 corpus renders 'Elven Ring N'/'Dark Ring N',
                # the packaged WOTR-Gandalf2 build renders the named
                # active slots ('Vilya Slot'...) and 'Used Ring N'
                # (Shadow-collected) — both vocabularies count.
                _fp_active = ('elven ring' in frl
                              or frl.strip() in _FP_RING_SLOTS,
                              'elven ring' in tol
                              or tol.strip() in _FP_RING_SLOTS)
                _sh_held = ('dark ring' in frl
                            or frl.strip().startswith('used ring'),
                            'dark ring' in tol
                            or tol.strip().startswith('used ring'))
                if _fp_active[0]:
                    f.rings['FP'] -= 1
                if _sh_held[0]:
                    f.rings['SH'] -= 1
                if _fp_active[1]:
                    f.rings['FP'] += 1
                if _sh_held[1]:
                    f.rings['SH'] += 1
                continue
            if obj.startswith('group (') and obj.endswith(')'):
                # split_group takes the INNER text (parse_line's own
                # call passes the regex capture, not the wrapper)
                pieces = rr.split_group(obj[len('group ('):-1])
            else:
                pieces = [obj]
            for p in pieces:
                _apply_piece(f, p, src, -1)
                _apply_piece(f, p, dst, +1)
        elif op == 'kill':
            zk = zone_key(rec.get('zone'))
            if zk[0] in ('region', 'stronghold'):
                for p in rec.get('pieces', []):
                    _apply_piece(f, p, zk, -1)
                continue
            for p in rec.get('pieces', []):
                pp = rr.parse_piece(p)
                if pp is not None and pp[0] == 'named' \
                        and pp[1] in _COMPANION_NAMES:
                    # companion killed outside a region = taken as a
                    # hunt/effect casualty
                    f.comps_gone[pp[1]] += 1
                    continue
                if zk[0] == 'hand':
                    m = _DECK_DRAW_RE.match(str(p).strip().lower())
                    if m:
                        # identity-less deck-label kill in a hand zone:
                        # the hand-limit discard gesture
                        f.hand_discards[
                            ('FREE_PEOPLES' if m.group(1) == 'free'
                             else 'SHADOW', m.group(2))] += 1
                        continue
                    # an identified card discarded from hand: no
                    # specimen yet — LOUD hold, never silent
                    f.unmapped.append(p)
                # dice / tiles / cards killed in off-board zones are
                # client bookkeeping the engine derives itself
        elif op == 'flip':
            fobj = str(rec.get('obj', '') or '').strip().lower()
            fzone = str(rec.get('zone', '') or '').lower()
            if (fobj in ('the fellowship', 'fsp')
                    and not fzone.startswith(('political', 'mordor'))):
                # BOTH pawn and FSP-MARKER forms: some clients keep
                # the fellowship on the track and render hide/reveal
                # as 'flips FSP in Fellowship Track N' (game
                # 2023-01-22) — with 'Previously on' it is the
                # reveal + track reset, without it the HIDE gesture.
                # A track zone is never a region constraint.
                reg = sanitize(str(rec.get('region', '') or '')
                               ).strip().lower()
                if fobj == 'fsp' or reg.startswith('fellowship track'):
                    reg = ''
                if rec.get('revealed'):
                    # hunt/engine reveal: ALSO expresses the track
                    # reset (no separate FSP move is rendered)
                    f.fel_flips.append((reg, 'reveal'))
                    f.fs_progress += -int(rec.get('prev_track', 0))
                elif reg or fobj == 'fsp':
                    f.fel_flips.append((reg, 'flip'))
                continue
            if fobj in _FLIP_RENAME and rec.get('region'):
                # dual-faced figure flipped to its OTHER identity
                # (Gandalf the White transform: the client moves the
                # Grey piece then flips it) — rewrite the arrival.
                # The region must be CANONICALIZED the same way the
                # arrival row was, or the rename misses ('The Grey
                # Havens' vs 'grey havens', game 2023-01-13 T2)
                old = _FLIP_RENAME[fobj]
                new_reg = sanitize(str(rec['region'])).strip().lower()
                new_reg = region_lookup(new_reg) or new_reg
                # the OLD face leaves, the NEW face appears.  With an
                # in-window arrival this rewrites the arrival row
                # (+1-1=0, new +1: the GtW resurrection); STANDALONE
                # it expresses an in-place transform (the Aragorn
                # crown: 'flips Aragorn in Minas Tirith' with Strider
                # already standing there)
                f.named[(old, new_reg)] -= 1
                f.named[(rr.COMP_ALIAS.get(fobj, fobj), new_reg)] += 1
            # political activation flips stay engine-derived
            continue
        elif op == 'draw':
            m = _DECK_DRAW_RE.match(
                str(rec.get('what', '') or '').strip().lower())
            if m:
                side_name = ('FREE_PEOPLES' if m.group(1) == 'free'
                             else 'SHADOW')
                f.draws[(side_name, m.group(2))] += 1
            # non-deck 'draws' (hunt tiles, random companions) are
            # randomness records: fed to the funnels, never intent
            continue
        elif op == 'die_change':
            ds, frm, to = (rec.get('side'), rec.get('frm'),
                           rec.get('to'))
            if not (ds and frm and to):
                f.unmapped.append(repr(rec)[:80])
                continue
            f.dice_faces[(ds, frm)] -= 1
            f.dice_faces[(ds, to)] += 1
            if ds == w_side and frm == staged[0]:
                staged[0] = to      # the staged die itself flipped
            continue
        elif op in ('add_eye', 'alloc_eye', 'roll_num',
                    'roll_dice', 'choose', 'random_discard'):
            continue      # randomness/phase records: router territory
        else:
            f.unmapped.append(repr(rec)[:80])
    # the staged die is SPENT when the window closed on the client's
    # own consumption terminators, at its FINAL letter
    if w_side and staged[0] and getattr(window, 'closed', None) \
            in ('die_killed', 'die_to_hunt'):
        f.dice_faces[(w_side, staged[0])] -= 1
    return f


# ---------------------------------------------------------------------------
# engine delta -> the same fact space
# ---------------------------------------------------------------------------
class EngFacts(object):
    """Engine apply-on-clone delta, in window_facts' key space."""
    __slots__ = ('armies', 'leaders', 'named', 'fs_progress',
                 'corruption', 'draws', 'political', 'hidden_change',
                 'fel_region', 'guide', 'comps_left', 'hand_delta',
                 'rings', 'dice', 'sieges')

    def __init__(self):
        self.armies = Counter()
        self.leaders = Counter()
        self.named = Counter()
        self.fs_progress = 0
        self.corruption = 0
        self.draws = Counter()      # (side_name, deck) -> cards drawn
        self.political = Counter()  # nation NAME -> box increment
        self.hidden_change = False  # fellowship hidden toggled?
        self.fel_region = None      # post-state fellowship region key
        self.guide = None           # new guide name (only if changed)
        self.comps_left = Counter()  # companions who left the ring
        self.hand_delta = Counter()  # side_name -> hand size change
        self.rings = Counter()      # 'FP'/'SH' ring-count deltas
        self.dice = Counter()       # (side_name, letter) unused-face
        #                             multiset delta
        self.sieges = Counter()     # region -> +1 siege formed /
        #                             -1 lifted (besieged_strongholds)


_DECK_ATTRS = (('FREE_PEOPLES', 'character', 'fp_char_deck'),
               ('FREE_PEOPLES', 'strategy', 'fp_strat_deck'),
               ('SHADOW', 'character', 'sh_char_deck'),
               ('SHADOW', 'strategy', 'sh_strat_deck'))


def delta_facts(pre, post):
    """board_delta rows (+ direct deck/fellowship reads) -> EngFacts."""
    d = client_board.board_delta(pre, post)
    ef = EngFacts()
    for k, v in d.items():
        if k == '__rings__':
            if v[0]:
                ef.rings['FP'] = int(v[0])
            if v[1]:
                ef.rings['SH'] = int(v[1])
            continue
        if k == '__dice__':
            # unused-face multiset deltas (used-count changes ride the
            # face leaving the multiset; face names -> roll letters)
            for side_name, before, after in v:
                b = Counter(before[0] if before else ())
                a2 = Counter(after[0] if after else ())
                for face in set(b) | set(a2):
                    if face == 'None':
                        # an engine-added bonus die is UNSET until the
                        # next roll — not an observable board face (the
                        # client renders the new die with an arbitrary
                        # letter and kills it; Aragorn-crown specimen)
                        continue
                    n = a2.get(face, 0) - b.get(face, 0)
                    if n:
                        ef.dice[(side_name,
                                 FACE_LETTER.get(face, face))] += n
            continue
        if k == '__hands__' or k == '__table__':
            continue
        if k == '__political__':
            # summarize keys are nation VALUE strings; canonicalize to
            # the enum NAME (the same key space window_facts uses)
            for nm, inc in v:
                en = _nation_enum(nm)
                ef.political[en.name if en is not None
                             else str(nm).upper()] += int(inc)
            continue
        if k == '__fellowship__':
            for fk, fv in v:
                if fk == 'progress':
                    ef.fs_progress = int(fv)
                elif fk == 'corruption':
                    ef.corruption = int(fv)
                elif fk == 'guide':
                    ef.guide = str(fv)
                elif fk == 'left':
                    for nm in fv:
                        ef.comps_left[str(nm)] += 1
            continue
        if k == '__named__':
            for reg, before, after in v:
                for nm in set(before) - set(after):
                    ef.named[(nm, reg)] -= 1
                for nm in set(after) - set(before):
                    ef.named[(nm, reg)] += 1
            continue
        ud, ld = v
        for t, n in ud:
            ef.armies[(k, t)] += n
        if ld:
            ef.leaders[k] += ld
    # the engine's army.leaders holds named figures (companions/
    # minions) TOO, so the summarize leader count double-books what
    # __named__ already carries — fold it out: ef.leaders is the
    # GENERIC leader + nazgul delta, the same thing the client's
    # '<Nation> Leader'/'Nazgul' tokens express (Saruman muster
    # specimen: window named +1 leaders 0, engine counted both)
    for (nm, reg), v in ef.named.items():
        if v:
            ef.leaders[reg] -= v
    # draws: the deck lists are the one quantity ONLY a draw changes
    # (hand sizes net out when a chain draws and plays in one gesture)
    for side_name, deck, attr in _DECK_ATTRS:
        n = (len(getattr(pre, attr, ()) or ())
             - len(getattr(post, attr, ()) or ()))
        if n:
            ef.draws[(side_name, deck)] = n
    try:
        b0 = set(getattr(pre, 'besieged_strongholds', set()) or set())
        b1 = set(getattr(post, 'besieged_strongholds', set()) or set())
        for r in b1 - b0:
            ef.sieges[sanitize(str(r)).strip().lower()] += 1
        for r in b0 - b1:
            ef.sieges[sanitize(str(r)).strip().lower()] -= 1
    except Exception:
        pass
    try:
        ef.hidden_change = (bool(pre.fellowship.hidden)
                            != bool(post.fellowship.hidden))
        reg = post.fellowship.region
        ef.fel_region = sanitize(
            str(getattr(reg, 'key', reg))).strip().lower()
    except Exception:
        pass
    try:
        from game_state import Side as _S
        for sd in (_S.FREE_PEOPLES, _S.SHADOW):
            n = (len(post.hands[sd].hand) - len(pre.hands[sd].hand))
            if n:
                ef.hand_delta[sd.name] = n
    except Exception:
        pass
    return ef


def _nation_blind(rows):
    """Army rows aggregated to (region, kind) — the nation half of the
    type key folded out (the relax_nations lane's key space)."""
    out = Counter()
    for (reg, t), v in rows.items():
        out[(reg, t.split(':', 1)[1] if ':' in t else t)] += v
    return {k: v for k, v in out.items() if v}


def _shadow_tkey(tkey):
    n = str(tkey).split(':', 1)[0]
    return any(s in n for s in ('sauron', 'isengard', 'southron',
                                'easterling'))


def _named_side_tag(name):
    want = str(name).strip().lower()
    for nm in rr.NAMED_MINIONS.values():
        if str(nm).strip().lower() == want:
            return 'shadow'
    return 'fp'


def _advance_extra_ok(win_rows, eng_rows, adv_tag, side_of,
                      group=lambda k: k):
    """WAVE-I (the side=-fixed post-retreat AUTO-ADVANCE): engine-extra
    board rows are tolerated ONLY when every differing row belongs to
    the ADVANCING side (`adv_tag`, the retreat decider's OPPONENT) and
    the differences are count-CONSERVING (a pure relocation: per type
    the +/- regions cancel).  The client never renders the opponent's
    engine-auto advance inside the defender's own retreat gesture; the
    post-commit sweep emits it (driver.post_commit_sweep)."""
    diffs = {}
    for k in set(win_rows) | set(eng_rows):
        d = eng_rows.get(k, 0) - win_rows.get(k, 0)
        if not d:
            continue
        if side_of(k) != adv_tag:
            return False
        diffs[group(k)] = diffs.get(group(k), 0) + d
    return all(v == 0 for v in diffs.values())



def _only_extra_removals(win_rows, eng_rows):
    """True when the engine rows equal the window rows plus EXTRA
    removals only (negative counts on keys the window never voiced):
    the wiped army's leaders / figures the round aftermath removed."""
    for k, v in win_rows.items():
        if eng_rows.get(k) != v:
            return False
    for k, v in eng_rows.items():
        if k not in win_rows and v >= 0:
            return False
    return True

def _covers(win, eng, relax_draws=False, fs_reset_relax=False,
            combat_flux=False, relax_nations=False,
            auto_advance_side=None):
    """Does the engine delta EXPRESS everything the window expressed?
    Exact equality on army/draw/political rows; named/leader rows must
    agree where the window spoke.  Fellowship hidden-state is checked
    ONLY when the window carried a flip gesture: reveal-style flips
    are engine-derived and deliberately absent from window facts, so
    a card/hunt chain toggling hidden without a windowed flip is not
    a refusal.

    relax_draws (spectator fallback lane ONLY, always paired with a
    NAMED census row): the draw rows become a bounded-subset test —
    the engine must draw AT LEAST one and AT MOST the window's count
    of every deck the window drew, and nothing the window did not.
    Exists for the measured client-permitted misplay class (game
    2023-01-13: two draws rendered on 'draw one Strategy card' plays,
    raw-log pids 122/120 + 146/134 prove two REAL draws): the engine
    commits the rules-legal subset, the extra is the finding.

    combat_flux (the chain contains a combat leg): POLITICAL rows the
    window voiced must match exactly, but ENGINE-EXTRA political rows
    are tolerated — being attacked advances the defender's track
    (p.34) and that is engine bookkeeping some clients simply never
    render (game 2023-01-31 T2: the Woodland Realm siege window
    carries no Elves track move; strict parity refused every battle
    chain at the root).

    relax_nations (spectator fallback lane ONLY, always paired with a
    NAMED engine-gap census row): army rows compare on nation-blind
    (region, kind) totals.  Exists because game_logic.move_army's
    split take() picks units in list order and carries NO per-nation
    composition parameter, so a mixed-nation drag (1 Isengard + 1
    Sauron regular, game 2023-01-18 T2 Moria) is UNREACHABLE from the
    menu — the engine commits the closest composition, the census row
    keeps the engine debt (fix owed: composition-aware move_army)."""
    w_arm = {k: v for k, v in win.armies.items() if v}
    e_arm = {k: v for k, v in eng.armies.items() if v}
    if relax_nations:
        if _nation_blind(win.armies) != _nation_blind(eng.armies):
            return False
    elif w_arm != e_arm:
        if auto_advance_side is None or not _advance_extra_ok(
                w_arm, e_arm, auto_advance_side,
                lambda k: ('shadow' if _shadow_tkey(k[1]) else 'fp'),
                group=lambda k: k[1]):
            return False
    w_led = {k: v for k, v in win.leaders.items() if v}
    e_led = {k: v for k, v in eng.leaders.items() if v}
    if w_led != e_led:
        # generic leader tokens carry no side; the advance tolerance
        # accepts only count-conserving relocations (side attribution
        # rides the army rows above).  Under combat flux an army WIPE
        # removes its leaders/figures inside the engine's round
        # aftermath - a forced row the gesturing side never drags
        # (the opponent's pieces; the sweep emits them) - so engine
        # EXTRA removals are tolerated, never extra placements.
        if combat_flux and _only_extra_removals(w_led, e_led):
            pass
        elif auto_advance_side is None or not _advance_extra_ok(
                w_led, e_led, auto_advance_side,
                lambda k: auto_advance_side,
                group=lambda k: '_'):
            return False
    w_nam = {k: v for k, v in win.named.items() if v}
    e_nam = {k: v for k, v in eng.named.items() if v}
    if w_nam != e_nam:
        if combat_flux and _only_extra_removals(w_nam, e_nam):
            pass
        elif auto_advance_side is None or not _advance_extra_ok(
                w_nam, e_nam, auto_advance_side,
                lambda k: _named_side_tag(k[0]),
                group=lambda k: k[0]):
            return False
    if win.fs_progress != eng.fs_progress:
        # the reveal's TRACK RESET is engine bookkeeping: the client
        # renders it as an FSP marker flip ('flips FSP in Fellowship
        # Track 0', game 2023-01-22), a separate drag, or not at all.
        # ONLY at expiry-time PREFIX matching (fs_reset_relax: the
        # buffer is complete but split mid-cascade) may a pawn-
        # placement prefix tolerate the engine reset — a GROWING
        # buffer must stay strict or a one-record prefix covers before
        # the flip arrives (game 2023-01-02 regression: the pawn
        # stranded at Fords of Bruinen).
        if not (fs_reset_relax and win.fel_moves
                and eng.fs_progress <= win.fs_progress <= 0):
            return False
    if win.corruption != eng.corruption:
        return False
    w_drw = {k: v for k, v in win.draws.items() if v}
    e_drw = {k: v for k, v in eng.draws.items() if v}
    if relax_draws:
        for k in set(w_drw) | set(e_drw):
            e, wv = e_drw.get(k, 0), w_drw.get(k, 0)
            if e > wv or (wv > 0 and e < 1) or (wv == 0 and e):
                return False
    elif w_drw != e_drw:
        return False
    w_pol = {k: v for k, v in win.political.items() if v}
    e_pol = {k: v for k, v in eng.political.items() if v}
    if combat_flux:
        # voiced rows exact; engine-extra rows = the attacked nation's
        # p.34 advance the client did not render (see docstring)
        for k, v in w_pol.items():
            if e_pol.get(k, 0) != v:
                return False
    elif w_pol != e_pol:
        return False
    w_rng = {k: v for k, v in win.rings.items() if v}
    e_rng = {k: v for k, v in eng.rings.items() if v}
    if w_rng != e_rng:
        return False
    w_dice = {k: v for k, v in win.dice_faces.items() if v}
    e_dice = {k: v for k, v in eng.dice.items() if v}
    if w_dice != e_dice:
        return False
    # siege rows: SIGN parity where the window spoke (the garrison may
    # walk in as several drag records; the engine flag flips once)
    def _sgn(x):
        return (x > 0) - (x < 0)
    for k, v in win.sieges.items():
        if _sgn(v) != _sgn(eng.sieges.get(k, 0)):
            return False
    if win.fel_flips:
        # plain flips are the human's own toggle gesture — strict
        # parity.  A window carrying a REVEAL flip keeps only the
        # region + track-reset constraints: the engine's hidden toggle
        # may have landed in an earlier step (the hunt reveal fires
        # inside the fellowship-move apply; the client renders the
        # flip with the consequence records that follow).
        if all(k == 'flip' for _r, k in win.fel_flips):
            want_toggle = (len(win.fel_flips) % 2) == 1
            if eng.hidden_change != want_toggle:
                return False
        if win.fel_flips[-1][0] \
                and eng.fel_region != win.fel_flips[-1][0]:
            return False
    if win.fel_moves and eng.fel_region != win.fel_moves[-1][1]:
        return False
    if win.guide_set is not None and eng.guide != win.guide_set:
        return False
    for nm, v in win.comps_gone.items():
        # one-way: engine departures the window did not voice are
        # engine-derived bookkeeping (separations render as named
        # region arrivals instead)
        if v and eng.comps_left.get(nm, 0) != v:
            return False
    if win.hand_discards:
        want = Counter()
        # under the relaxed-draws lane the hand grows by what the
        # ENGINE actually drew (the window's extra never entered it)
        for (sd, _dk), v in (eng.draws if relax_draws
                             else win.draws).items():
            want[sd] += v
        for (sd, _dk), v in win.hand_discards.items():
            want[sd] -= v
        for sd in set(want) | set(eng.hand_delta):
            if want.get(sd, 0) != eng.hand_delta.get(sd, 0):
                return False
    return True


class MatchResult(object):
    def __init__(self, status, chain=None, post=None, applies=0,
                 note='', starved=False):
        self.status = status      # 'match'|'ambiguous'|'none'|'hold'
        self.chain = chain or []  # [Action, ...] (action + pendings)
        self.post = post          # committed-state candidate
        self.applies = applies
        self.note = note
        # True when at least one candidate STARVED the client-
        # randomness feed (NeedClientCall) and nothing matched: the
        # gesture's randomness has not arrived yet — the caller may
        # retry after feeding more (spectator deferral), never refuse
        self.starved = starved

    def __repr__(self):
        return '<MatchResult %s chain=%d applies=%d %s%s>' % (
            self.status, len(self.chain), self.applies, self.note,
            ' STARVED' if self.starved else '')


# 30000: the Ringwraiths-are-Abroad window (2 nazgul walks + a 2-army
# followup) needed 22k applies even with the pair prefilter + state
# memo; the budget is a HOLD guard, never a refusal (B13-1)
DEFAULT_BUDGET = 30000
# depth 12: a card play can chain play_event + several card_option/
# followup legs + the hunt/casualty pendings its effect opens (The
# Nazgul Strike!: play, two nazgul-move options, strike, random
# casualty, hobbit placement) — depth 3 and then 5 stranded real
# corpus windows mid-chain.  Multi-round sieges (Grond specimen:
# 3 assault rounds, ~7 CHOICE legs once forced one-action pendings
# stopped costing depth) needed the raise from 8; depth bounds
# branching, the budget bounds work.
DEFAULT_DEPTH = 12

_CARD_LOOKUP = None


def is_ring_track_move(obj, frm, to):
    """THE ONE test for 'this move record is an Elven-Ring marker
    walk': a ring-track zone on either end (both module vocabularies:
    'Elven Ring N' / 'Dark Ring N' and the named slots), or the bare
    ring name.  Shared by window_facts (in-window ring) and the
    driver's standalone-ring lane."""
    obj = str(obj or '')
    frl = str(frm or '').lower()
    tol = str(to or '').lower()
    return ('elven ring' in frl or 'dark ring' in frl
            or 'elven ring' in tol or 'dark ring' in tol
            or frl.strip() in _FP_RING_SLOTS
            or tol.strip() in _FP_RING_SLOTS
            or ('/' not in obj and obj.strip()
                in ('Narya', 'Nenya', 'Vilya')))


def title_to_cid(title):
    """Rendered card TITLE ('Event/Combat' dual title) -> engine
    card_id via ai's own accent/alias-tolerant lookup (the single
    answerer the corpus extractor uses too).  None on miss."""
    global _CARD_LOOKUP
    import ai
    if _CARD_LOOKUP is None:
        _CARD_LOOKUP = ai._build_card_name_lookup()
    if not title:
        return None
    cid = ai._lookup_card_id(_CARD_LOOKUP, str(title))
    if cid:
        return cid
    head = str(title).split('/')[0].strip()
    return ai._lookup_card_id(_CARD_LOOKUP, head) if head else None

# rendered roll letters <-> engine face names (roll grammar [ACEHMPW])
FACE_LETTER = {'CHARACTER': 'C', 'ARMY': 'A', 'MUSTER': 'M',
               'EVENT': 'P', 'MUSTER_ARMY': 'H', 'EYE': 'E',
               'WILL_OF_WEST': 'W'}


def _die_filter(state, side, die_face):
    """Menu filter for the staged die's face: the window SAID which die
    funds the action (the staging record carries side+face), so menu
    actions funded by a different face are not candidates.  A live
    ring waiver widens funding (A7 legality expansion), so with the
    waiver up the filter admits every face."""
    import game_logic as gl
    try:
        if gl.ring_waiver_active(state, side):
            return lambda a: True
    except Exception:
        pass

    def ok(a):
        di = getattr(a, 'die_idx', None)
        if di is None or di < 0:
            return True
        dice = state.dice.get(side, [])
        if not 0 <= di < len(dice):
            return False
        cur = getattr(dice[di].current, 'name', None)
        return FACE_LETTER.get(str(cur), '?') == die_face
    return ok


def _facts_fingerprint(eng):
    """Canonical fingerprint of an engine delta in fact space."""
    return (tuple(sorted((k, v) for k, v in eng.armies.items() if v)),
            tuple(sorted((k, v) for k, v in eng.leaders.items() if v)),
            tuple(sorted((k, v) for k, v in eng.named.items() if v)),
            eng.fs_progress, eng.corruption,
            tuple(sorted((k, v) for k, v in eng.draws.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.political.items() if v)),
            eng.hidden_change, eng.fel_region, eng.guide,
            tuple(sorted((k, v)
                         for k, v in eng.comps_left.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.hand_delta.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.rings.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.dice.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.sieges.items() if v)))


def _chain_key(chain, fp):
    """Physical-equivalence key: chains that produce the IDENTICAL
    board delta through the same action types are the SAME human
    gesture — a different same-faced die, or take_leaders over a
    leaderless army, is not an ambiguity a human could express."""
    return (tuple(getattr(a, 'action_type', '?') for a in chain), fp)


def _pending_ctx_tag(post, nxt_kind):
    """The pending CONTEXT half of the exploration-memo key.  The
    combat menus do not name their battle (combat_card_none carries
    only a side), so two different battles look memo-identical without
    this: attacking Dale and attacking Old Forest Road both delta to
    {political +1, die -1} with continuation [combat_card_none] — the
    first probe's memo entry pruned the true attack's whole subtree
    (game 2023-01-25 T3, the g6 root)."""
    pc = getattr(post, 'pending_combat', None)
    return (nxt_kind,
            ((str(pc.get('attacker_region')),
              str(pc.get('defender_region')),
              str(pc.get('phase')),
              pc.get('rounds_fought'))
             if isinstance(pc, dict) else None))


class _PrefixShim(object):
    """Window-shaped record-prefix container for window_facts."""
    def __init__(self, records, die_side, die_face):
        self.records = records
        self.die_side = die_side
        self.die_face = die_face
        self.closed = None

    def body_records(self):
        return self.records


def _board_rows_key_eng(ef):
    return (tuple(sorted((k, v) for k, v in ef.armies.items() if v)),
            tuple(sorted((k, v) for k, v in ef.leaders.items() if v)),
            tuple(sorted((k, v) for k, v in ef.named.items() if v)))


def _board_rows_key_win(wf):
    return (tuple(sorted((k, v) for k, v in wf.armies.items() if v)),
            tuple(sorted((k, v) for k, v in wf.leaders.items() if v)),
            tuple(sorted((k, v) for k, v in wf.named.items() if v)))


def _record_order_filter(state, side, window, cands, rng):
    """PER-RECORD leg matching over net-tied chains (the multi-round
    siege refinement): the client rendered the battle's INTERMEDIATE
    states (round-1 casualties before round-2's), so among chains with
    the identical NET delta only those whose cumulative leg states
    pass through every client-record state IN ORDER are the real
    gesture (Grond specimen: 5 net-identical allocations, one true
    round-by-round history).  Board rows only (armies/leaders/named).
    Returns the surviving (chain, post) list; on replay trouble the
    candidate is kept (never disambiguate by accident)."""
    import ai
    # checkpoints at ROUND boundaries: the client may render one
    # engine leg as several records (the elite downgrade: kill line +
    # replacement line), so per-record prefixes produce states no
    # engine leg ever visits.  The roll groups partition the battle
    # into rounds; the state at each round START is one the chain must
    # pass through exactly.
    recs = list(window.records)
    marks = []
    for i in range(1, len(recs)):
        if recs[i].get('op') == 'roll_num' \
                and recs[i - 1].get('op') != 'roll_num':
            marks.append(i)
    body = window.body_records()
    marks.append(None)          # the full window as the last mark
    want = []
    body_ids = list(map(id, body))
    for m in marks:
        sub = recs[:m] if m is not None else recs
        subset = [r for r in sub if id(r) in set(body_ids)]
        wf = window_facts(_PrefixShim(subset, window.die_side,
                                      window.die_face))
        k = _board_rows_key_win(wf)
        if not want or want[-1] != k:
            want.append(k)
    want = [k for k in want if any(k)]
    if len(want) < 2:
        return cands            # no intermediate history to key on
    kept = []
    for chain, post in cands:
        rng.root_attempt()
        cur = state
        keys = []
        ok = True
        for leg in chain:
            try:
                with rng.apply_cm():
                    cur = ai.apply_action(cur, leg)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                ok = None       # replay trouble: keep, do not judge
                break
            keys.append(_board_rows_key_eng(delta_facts(state, cur)))
        if ok is None:
            kept.append((chain, post))
            continue
        wi = 0
        for k in keys:
            if wi < len(want) and k == want[wi]:
                wi += 1
        if wi == len(want):
            kept.append((chain, post))
    return kept or cands


# pendings whose RESOLUTION renders as records AFTER a FELLOWSHIP-MOVE
# gesture (hunt consequences, reveal placement, guide changes): the
# window chain must stop at them — the pending-buffer phase owns their
# content.  Applies ONLY to die_to_hunt windows: a CARD window (The
# Nazgul Strike!) renders its whole hunt cascade IN-window and must
# traverse.  Depth-12 expansion sailing through reveal_place guessed
# an unrendered pawn placement and stranded the fellowship at the
# wrong region (game 2023-01-02 regression).
POST_WINDOW_KINDS = frozenset(('hunt', 'reveal_place', 'guide_change',
                               'discard', 'declare', 'eye'))


def match_window(state, side, window, menu=None, budget=DEFAULT_BUDGET,
                 depth=DEFAULT_DEPTH, pin=None, rng=None,
                 relax_draws=False, stop_kinds=POST_WINDOW_KINDS,
                 fs_reset_relax=False, relax_nations=False,
                 advance_tolerance=False, fs_prefilter=False):
    """Match a closed gesture window against the engine menu.

    Returns MatchResult.  'hold' when the budget exhausts or the window
    holds unmapped records (never a refusal on matcher limitations —
    B13-1 doctrine: exhaustion HOLDS).  `pin` is an optional context
    manager factory (ClientRandomness.pin) so legality probes consume
    no live randomness.

    `rng` (spectator lane): a cursor controller separating the two
    randomness roles the single `pin` conflated —
      probe_cm()     enumeration context (get_legal_actions /
                     describe_pending): funnels OFF, RNG pinned.  The
                     conflation put menu builds inside advance(), so
                     _event_card_has_effect dry-runs reached the LIVE
                     card funnel and starved (game 2023-01-13's ten
                     route_error rows: NeedClientCall('draws') escaping
                     get_legal_actions);
      root_attempt() fresh attempt before each ROOT candidate;
      apply_cm()     advance-only apply context (NO per-leg attempt
                     reset: a chain's legs consume the fed randomness
                     CONTIGUOUSLY, exactly as _commit_chain replays
                     them — per-leg begin_attempt made every leg re-read
                     the feed from index 0, so any chain rolling dice in
                     two legs probed one thing and committed another);
      save()/restore(snap)  cursor bracketing around depth>0 candidate
                     bodies so sibling legs backtrack cleanly.
    Without `rng`, `pin` serves every role (LiveDriver's current
    contract, unchanged)."""
    import ai
    import contextlib
    pin = pin or (lambda: contextlib.nullcontext())
    if rng is not None:
        probe_ctx, apply_ctx = rng.probe_cm, rng.apply_cm
    else:
        probe_ctx = apply_ctx = pin
    if getattr(window, 'closed', None) != 'die_to_hunt':
        stop_kinds = ()
    facts = window_facts(window)
    if facts.unmapped:
        return MatchResult('hold', note='unmapped records: %r'
                           % facts.unmapped[:3])
    # ZERO-HIT BATTLE lane: a battle whose every roll missed moves no
    # piece — the window's ONLY observables are its roll records (and
    # the spent die), which express nothing in fact space.  Such a
    # window must still match the battle chain that consumes exactly
    # those rolls (empty board delta covers the empty facts; the
    # commit's unconsumed-randomness honesty gate polices the rolls).
    rolls_only = (facts.is_empty()
                  and getattr(window, 'die_face', None) is not None
                  and any(r.get('op') == 'roll_num'
                          for r in window.records))
    if facts.is_empty() and not rolls_only:
        return MatchResult('none', note='window expressed nothing')
    if menu is None:
        with probe_ctx():
            menu = ai.get_legal_actions(state, side)
    if getattr(window, 'die_face', None):
        menu = [a for a in menu
                if _die_filter(state, side, window.die_face)(a)]
    cids = []
    if facts.cards_played:
        for t in facts.cards_played:
            cid = title_to_cid(t)
            if cid is None:
                return MatchResult('hold',
                                   note='unresolved card title %r' % t)
            cids.append(cid)
    cc_want = {}
    for tag, t in facts.combat_cards:
        cid = title_to_cid(t)
        if cid is None:
            return MatchResult('hold',
                               note='unresolved combat card %r' % t)
        cc_want.setdefault('fp' if tag == 'FP' else 'sh',
                           []).append(cid)

    def _combat_cards_ok(chain):
        """The client's chooser NAMED each combat card (and its side);
        a chain playing different combat cards — or any at all when
        the window shows none — is a different battle, even when the
        rolls made the outcome identical (Osgiliath specimen: four
        net-identical card combinations)."""
        got = {}
        for x in chain:
            if x.action_type == 'combat_card_play':
                got.setdefault(str((x.params or {}).get('side')),
                               []).append(
                                   (x.params or {}).get('card_id'))
        return ({k: sorted(v) for k, v in got.items()}
                == {k: sorted(v) for k, v in cc_want.items()})

    def _vocab_prefilter(acts):
        """VOCABULARY lane: the client NAMED the played card — the
        only ACTION candidates are the play_event actions for exactly
        those cids, plus elven_ring when the window walked a ring
        marker (the ring flips the funding die BEFORE the play).
        Independently: a deck draw ALWAYS renders its 'draws ... Card.'
        line, so draw_card candidates in a window with no draw fact
        are never the gesture — probing them starves the card funnel
        and mis-classifies plain non-matches as rng-deferrals."""
        if not facts.draws:
            acts = [a for a in acts if a.action_type != 'draw_card']
        if not cids:
            return acts
        out = []
        for a in acts:
            if a.action_type == 'play_event' \
                    and a.params.get('card_id') in cids:
                out.append(a)
            elif facts.ring_moves and a.action_type == 'elven_ring':
                out.append(a)
        return out

    menu = _vocab_prefilter(menu)
    if cids and not menu:
        return MatchResult(
            'none', note='play_event %r not in the engine menu'
                         % cids)
    counter = {'n': 0, 'starved': 0}
    # SOUND move-leg prefilter: the client renders EVERY board move
    # leg as its own 'moves ... from X to Y' record, so an engine
    # move-shaped leg (army_move, a card_option nazgul relocation)
    # whose (from, to) pair the window never rendered cannot be part
    # of this gesture — applying it only burns budget, and net-
    # equivalent multi-hop paths through un-rendered regions read as
    # false ambiguity (the Nazgul-Strike 3-hop twin).  Placement legs
    # (companion_separate) carry no client 'from': region-level check.
    win_regs = set(r for (r, _t) in facts.armies)
    win_regs.update(facts.leaders)
    win_regs.update(r for (_n, r) in facts.named)
    win_pair_counts = Counter()
    for _rec in window.body_records():
        if _rec.get('op') != 'move':
            continue
        _s = zone_key(_rec.get('from'))
        _d = zone_key(_rec.get('to'))
        if _s[0] in ('stronghold', 'mordor'):
            _s = ('region', 'gorgoroth' if _s[0] == 'mordor'
                  else _s[1])
        if _d[0] in ('stronghold', 'mordor'):
            _d = ('region', 'gorgoroth' if _d[0] == 'mordor'
                  else _d[1])
        if _s[0] == 'region' and _d[0] == 'region':
            win_pair_counts[(_s[1], _d[1])] += 1
    win_pairs = set(win_pair_counts)

    # WAVE-K (wkg2 LIVE, FAILED-ALARM): a FELLOWSHIP gesture always
    # renders fellowship vocabulary - the FSP track walk, the pawn, a
    # flip/reveal, a guide change, a corruption walk.  A window with
    # NONE of it can never be one; probing a fellowship_move candidate
    # anyway starved the HUNT funnel, deferred an ILLEGAL army-move
    # window, ordered a REAL hunt roll + hunt tile for a non-gesture
    # (the client hunt pool lost a tile the engine never drew) and the
    # client lines pushed after the window made the refusal's undo
    # INELIGIBLE (fact 2).  Structural, rendering-contract prefilter -
    # never a rules guess.
    _fs_vocab = bool(facts.fs_progress or facts.fel_flips
                     or facts.fel_moves or facts.corruption
                     or facts.guide_set or facts.comps_gone)

    def _region_prefilter(acts):
        out = []
        for a in acts:
            p = a.params or {}
            fr = to = None
            pair_leg = False
            if fs_prefilter and not _fs_vocab \
                    and a.action_type in ('fellowship_move',
                                          'fellowship_hide',
                                          'fellowship_declare',
                                          'fellowship_reveal_place'):
                # (spectator-gated like advance_tolerance: the corpus
                # census stays byte-pinned; LIVE drivers pass True)
                continue
            if fs_prefilter and facts.sieges \
                    and a.action_type == 'combat_no_retreat':
                # WAVE-K (wkg3): a window that DRAGGED into the
                # stronghold is never the field-battle decision -
                # probing it starves the combat-roll funnel (the K2
                # class: rng ordered for a non-gesture)
                continue
            if a.action_type == 'army_attack':
                # a battle ALWAYS marks its regions (casualties,
                # advances, siege retreats); an attack in regions the
                # window never touched is not this gesture — probing
                # it starves the combat-roll funnel and turns plain
                # non-matches into false rng-deferrals (measured on
                # every T5 fellowship-move window of game 2023-01-13).
                # EXCEPT the zero-hit lane: an all-miss battle marks
                # NO region — the rolls themselves pick the battle.
                if rolls_only:
                    out.append(a)
                    continue
                frk = sanitize(str(p.get('from', ''))).strip().lower()
                tok = sanitize(str(p.get('to', ''))).strip().lower()
                if frk not in win_regs and tok not in win_regs \
                        and (frk, tok) not in win_pairs \
                        and frk not in facts.sieges \
                        and tok not in facts.sieges:
                    continue
                out.append(a)
                continue
            if a.action_type == 'army_move':
                fr, to = p.get('from'), p.get('to')
                pair_leg = True
            elif a.action_type == 'companion_separate':
                to = p.get('to_region')
            elif a.action_type == 'card_option':
                ch = p.get('choices')
                if isinstance(ch, dict) and 'from_region' in ch \
                        and 'to_region' in ch:
                    fr, to = ch.get('from_region'), ch.get('to_region')
                    pair_leg = True
            if pair_leg and fr is not None and to is not None:
                frk = sanitize(str(fr)).strip().lower()
                tok = sanitize(str(to)).strip().lower()
                if (frk, tok) not in win_pairs:
                    continue
            elif to is not None and sanitize(str(to)).strip().lower() \
                    not in win_regs:
                continue
            out.append(a)
        return out

    def _pair_count_filter(cands):
        """AMBIGUITY-ONLY disambiguation by rendered-record counts:
        the client renders every board move leg as its own record, so
        a chain using MORE move-shaped legs on a (from, to) pair than
        the window rendered records for that pair implies drags the
        client never made.  Kills the degenerate full-army split twin
        (game 2023-01-07 T2: one 'moves group ...' record matched BOTH
        [move-half, move-half] and [move-whole, pass] — different
        action-type tuples, so the physical-equivalence collapse could
        not merge them).  <= bound, never ==: one record may cover a
        multi-effect engine leg, and the filter only ever REMOVES
        chains when at least one candidate survives."""
        kept = []
        for chain, post in cands:
            use = Counter()
            for x in chain:
                p = x.params or {}
                fr = to = None
                if x.action_type in ('army_move', 'victor_advance',
                                     'nazgul_move'):
                    # nazgul_move counts too: WK+escort rendered as
                    # ONE group drag matched both [move-all-inc-WK]
                    # and [move-escort, option-move-WK] — two legs on
                    # one record (game 2023-01-02 T4 w066)
                    fr, to = p.get('from'), p.get('to')
                elif x.action_type == 'card_option':
                    ch = p.get('choices')
                    if isinstance(ch, dict) \
                            and 'from_region' in ch \
                            and 'to_region' in ch:
                        fr, to = (ch.get('from_region'),
                                  ch.get('to_region'))
                if fr is None or to is None:
                    continue
                use[(sanitize(str(fr)).strip().lower(),
                     sanitize(str(to)).strip().lower())] += 1
            if all(v <= win_pair_counts.get(k, 0)
                   for k, v in use.items()):
                kept.append((chain, post))
        return kept

    def _close_multistep(post, chain):
        """A window closed = the die action is COMPLETE (the client
        killed the die).  Close any engine multi-step half (second
        army move / second recruit) with its explicit 'pass' so the
        committed state carries no phantom open half-action."""
        for _ in range(4):
            if getattr(post, 'pending_army_move_die', None) is None \
                    and getattr(post, 'pending_muster_die',
                                None) is None:
                break
            owner = (getattr(post, 'pending_army_move_side', None)
                     if getattr(post, 'pending_army_move_die', None)
                     is not None
                     else getattr(post, 'pending_muster_side', None))
            if owner not in (None, side):
                break
            try:
                with probe_ctx():
                    acts = ai.get_legal_actions(post, side)
                pa = next((x for x in acts
                           if x.action_type == 'pass'), None)
                if pa is None:
                    break
                with apply_ctx():
                    post = ai.apply_action(post, pa)
                chain = chain + [pa]
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                break
        return post, chain

    seen = {}     # cumulative-delta fingerprint -> best remaining
    #   depth explored: different leg ORDERS reach identical partial
    #   states (the Ringwraiths 4-leg permutations exhausted 8000
    #   applies) — one exploration per state suffices, the collapse
    #   would merge the type-identical chains anyway

    def _try(st, acts, chain, d):
        hits = []
        at_root = not chain
        for a in acts:
            if counter['n'] >= budget:
                return 'BUDGET'
            counter['n'] += 1
            # cursor discipline (spectator rng only): a ROOT candidate
            # opens a fresh attempt; a deeper leg continues the chain's
            # own consumption and is BRACKETED so siblings backtrack
            snap = None
            if rng is not None:
                if at_root:
                    rng.root_attempt()
                else:
                    snap = rng.save()
            try:
                try:
                    with apply_ctx():
                        post = ai.apply_action(st, a)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    # NeedClientCall (a BaseException): the candidate
                    # starved the randomness feed beyond what has
                    # arrived — counted so the caller can distinguish
                    # 'randomness not here yet' (retry later) from a
                    # genuine non-match
                    if type(e).__name__ == 'NeedClientCall':
                        counter['starved'] += 1
                        # keep the starve KINDS visible even though
                        # backtracking restores rand._pending (the
                        # deferral-expiry row was logging pending=None)
                        counter.setdefault('starve_kinds', set()).add(
                            '%s:%s' % (getattr(e, 'kind', '?'),
                                       (getattr(e, 'kw', {}) or {})
                                       .get('ctx', '')))
                        # WAVE-J (wjg12): the starving CALL OBJECT
                        # survives too - backtracking restores
                        # rand._pending to None, so the deferral's
                        # rng ask had NOTHING to order (three live
                        # games expired on exactly this)
                        counter['starve_call'] = e
                    continue
                # CUMULATIVE delta from the WINDOW's root state — a
                # chain's coverage is everything it did, not its last
                # leg (the per-level delta could never cover a two-leg
                # army move: leg 2's own delta misses leg 1's rows;
                # found on the split-move specimen, game 2023-01-02
                # T1).  Coverage is judged on the PASS-CLOSED
                # candidate: an open half-action has not consumed its
                # die yet, and the window's die-kill says the die IS
                # spent (dice-row lane).
                eng = delta_facts(state, post)
                post2, chain2 = _close_multistep(post, chain + [a])
                eng2 = (eng if post2 is post
                        else delta_facts(state, post2))
                flux = any(getattr(x, 'action_type', '')
                           in _FLUX_TYPES for x in chain + [a])
                # WAVE-I: a chain with the defender's combat_retreat
                # carries the engine's side=-fixed AUTO-ADVANCE of the
                # OPPONENT's army — rows this gesture can never voice
                # IN LIVE SIDED MODE (the AI renders no gesture; the
                # post-commit sweep emits them).  LIVE-ONLY, bounded
                # relocation tolerance (_advance_extra_ok): in
                # spectator mode BOTH seats' records render, coverage
                # stays exact (a broad tolerance measurably stole g1
                # matches by letting retreat candidates cover foreign
                # windows).
                _adv = None
                # WAVE-K (wkg3 LIVE, siege formation): the defender's
                # retreat_into_siege carries the SAME engine-forced
                # opponent relocation (the besieger moves INTO the
                # region; the garrison is in the stronghold sub-box) -
                # the post-commit sweep emits it (wkg3 feed: the AI's
                # three move lines right after the withdrawal)
                if advance_tolerance and any(
                        x.action_type in ('combat_retreat',
                                          'retreat_into_siege')
                        for x in chain2):
                    _adv = ('shadow'
                            if side.name == 'FREE_PEOPLES' else 'fp')
                if _covers(facts, eng2, relax_draws=relax_draws,
                           fs_reset_relax=fs_reset_relax,
                           combat_flux=flux,
                           relax_nations=relax_nations,
                           auto_advance_side=_adv):
                    if _combat_cards_ok(chain2):
                        hits.append((chain2, post2,
                                     _facts_fingerprint(eng2)))
                    continue
                if d > 0 and _subset(eng, facts, combat_flux=flux,
                                     relax_nations=relax_nations):
                    with probe_ctx():
                        nxt = ai.describe_pending(post)
                    if nxt is not None and nxt['kind'] in stop_kinds:
                        # post-window pending: its records have not
                        # rendered yet — the chain stops here, the
                        # pending buffer resolves it
                        acts2 = None
                    elif nxt is not None:
                        acts2 = nxt['actions']
                    elif (a.action_type == 'elven_ring'
                            or getattr(post, 'pending_army_move_die',
                                       None) is not None
                            or getattr(post, 'pending_muster_die',
                                       None) is not None):
                        # engine multi-step half-action (second army
                        # move / second recruit), or a FREE elven-ring
                        # flip whose die action follows in the same
                        # window: expand through the side's own fresh
                        # menu ('pass' excluded: an early close is the
                        # _close_multistep hit path, never a chain
                        # branch)
                        with probe_ctx():
                            acts2 = _vocab_prefilter(
                                [x for x in ai.get_legal_actions(post,
                                                                 side)
                                 if x.action_type != 'pass'])
                    else:
                        acts2 = None
                    if acts2:
                        acts2 = _region_prefilter(acts2)
                        # one exploration per (partial delta, pending
                        # CONTEXT, available continuations) — leg-order
                        # permutations reach identical states (the
                        # Ringwraiths 4-leg blowup); the continuation
                        # menu is part of the key so two states that
                        # merely LOOK alike on the board but offer
                        # different futures stay distinct (the
                        # delta-only key wrongly merged Nazgul-Strike
                        # option states and pruned the true chain).
                        # The pending-combat CONTEXT is part of the key
                        # (see _pending_ctx_tag).
                        _ctx = _pending_ctx_tag(
                            post, nxt['kind'] if nxt is not None
                            else None)
                        key = (_facts_fingerprint(eng), _ctx,
                               tuple(sorted(
                            str((x.action_type, x.die_idx,
                                 tuple(sorted((str(k1), str(v1))
                                              for k1, v1
                                              in (x.params or {}
                                                  ).items()))
                                 )) for x in acts2)))
                        if seen.get(key, -1) >= d:
                            continue
                        seen[key] = d
                        # FORCED legs (a one-action pending: the
                        # combat-card 'none' beats, auto steps) spend
                        # no depth — depth bounds BRANCHING, and a
                        # 3-round siege is long but narrow (Grond
                        # specimen: 16 legs, ~7 of them choices)
                        d2 = d if len(acts2) == 1 else d - 1
                        sub = _try(post, acts2, chain + [a], d2)
                        if sub == 'BUDGET':
                            return 'BUDGET'
                        hits.extend(sub)
            finally:
                if snap is not None:
                    rng.restore(snap)
        return hits

    res = _try(state, _region_prefilter(menu), [], depth)
    if res == 'BUDGET':
        return MatchResult('hold', applies=counter['n'],
                           note='apply budget exhausted')
    if not res:
        if counter['starved']:
            _mr = MatchResult(
                'hold', applies=counter['n'], starved=True,
                note='rng-starved (%d candidates await client '
                     'randomness: %s)'
                     % (counter['starved'],
                        ','.join(sorted(counter.get('starve_kinds',
                                                    ())))))
            _mr.starve_call = counter.get('starve_call')
            return _mr
        return MatchResult('none', applies=counter['n'])
    # collapse physically-equivalent chains (identical delta through
    # the same action types): ONE human gesture, not an ambiguity
    uniq = {}
    for chain, post, fp in res:
        uniq.setdefault(_chain_key(chain, fp), (chain, post))
    if len(uniq) > 1:
        kept = _pair_count_filter(list(uniq.values()))
        if kept and len(kept) < len(uniq):
            uniq = {i: cp for i, cp in enumerate(kept)}
    if len(uniq) > 1 and rng is not None:
        kept = _record_order_filter(state, side, window,
                                    list(uniq.values()), rng)
        if len(kept) < len(uniq):
            uniq = {i: cp for i, cp in enumerate(kept)}
    if len(uniq) > 1:
        return MatchResult('ambiguous', applies=counter['n'],
                           chain=[c for c, _ in uniq.values()],
                           note='%d distinct actions produce the '
                                'window delta' % len(uniq))
    chain, post = next(iter(uniq.values()))
    return MatchResult('match', chain=chain, post=post,
                       applies=counter['n'])


_FLUX_TYPES = frozenset((
    'army_attack', 'siege_assault', 'casualty_allocate',
    'combat_card_none', 'combat_card_play', 'combat_no_retreat',
    'retreat_into_siege', 'combat_retreat', 'victor_advance',
    'victor_stay', 'siege_trim'))


# WindowFacts row -> EngFacts row it is compared with, and the words a
# player uses for it.  A refusal names the CATEGORY that differed,
# which is the thing a human can act on ("card draws" tells you your
# draw count was wrong; a raw board-delta key does not).
_FACT_ROWS = (
    ('armies', 'armies', 'army units'),
    ('leaders', 'leaders', 'leaders'),
    ('named', 'named', 'characters'),
    ('draws', 'draws', 'card draws'),
    ('political', 'political', 'the political track'),
    ('rings', 'rings', 'the ring track'),
    ('dice_faces', 'dice', 'the action die'),
)


def _facts_differ(want, got, action=None):
    """Player-word names of the fact categories where a window
    (WindowFacts `want`) and an engine delta (EngFacts `got`)
    disagree - each window row against the engine row it is judged
    on in _covers (like with like; one-way rows only where the window
    spoke).  The card identities live on the candidate ACTION, not
    in the engine delta: `action` (the applied root candidate) is
    what the window's named cards are compared with."""
    out = []

    def _cnt(x):
        return {k: v for k, v in (x or {}).items() if v}

    for w_attr, e_attr, word in _FACT_ROWS:
        if _cnt(getattr(want, w_attr, None)) \
                != _cnt(getattr(got, e_attr, None)):
            out.append(word)
    if int(want.fs_progress) != int(got.fs_progress):
        out.append('Fellowship progress')
    if int(want.corruption) != int(got.corruption):
        out.append('corruption')

    def _sgn(x):
        return (x > 0) - (x < 0)
    for k, v in want.sieges.items():
        if v and _sgn(v) != _sgn(got.sieges.get(k, 0)):
            out.append('sieges')
            break
    for nm, v in want.comps_gone.items():
        if v and got.comps_left.get(nm, 0) != v:
            out.append('companions leaving the Fellowship')
            break
    if want.hand_discards:
        need = Counter()
        for (sd, _dk), v in want.draws.items():
            need[sd] += v
        for (sd, _dk), v in want.hand_discards.items():
            need[sd] -= v
        if any(need.get(sd, 0) != got.hand_delta.get(sd, 0)
               for sd in set(need) | set(got.hand_delta)):
            out.append('cards discarded from hand')
    if want.fel_flips:
        bad = False
        if all(k == 'flip' for _r, k in want.fel_flips):
            bad = got.hidden_change != ((len(want.fel_flips) % 2) == 1)
        if want.fel_flips[-1][0] \
                and got.fel_region != want.fel_flips[-1][0]:
            bad = True
        if bad:
            out.append('the Fellowship marker')
    if want.fel_moves and got.fel_region != want.fel_moves[-1][1]:
        out.append('the Fellowship position')
    if want.guide_set is not None and got.guide != want.guide_set:
        out.append('the guide')
    if action is not None:
        p = getattr(action, 'params', None) or {}
        at = getattr(action, 'action_type', '')
        played = {p.get('card_id')} if at == 'play_event' else set()
        want_pl = set(title_to_cid(t) or t for t in want.cards_played)
        if want_pl != played:
            out.append('the card played')
        cc = {p.get('card_id')} if at == 'combat_card_play' else set()
        want_cc = set(title_to_cid(t) or t
                      for _s, t in want.combat_cards)
        if want_cc != cc:
            out.append('combat cards')
    return out


def explain_refusal(state, side, window, menu, rng=None):
    """Why a gesture found no legal action, in words the player can act
    on: the CLOSEST legal action plus the categories that differed.

    USER 2026-08-20: "if it says a move isn't legal it should specify
    what rule it thinks was broken."  match_window's 'none' branch
    returned no note at all, so driver._refuse fell back to the literal
    word 'gesture' and the player got 'that move is not legal -
    gesture'.  The chat vocabulary for the good message
    (chat.refusal_closest) was LIFTED from the monolith but never given
    a caller here -- the explainer itself
    (ai_move_server.py:5760-5862) stayed behind.

    Refusal-only cost: this re-applies every menu candidate, which is
    far too expensive per gesture but is exactly what the donor did,
    and a refusal is rare.  Returns (description, [differing words]) or
    (None, []) when nothing could be applied at all."""
    import engine_play
    want = window_facts(window)
    best, best_score, best_got = None, None, None
    for a in (menu or ()):
        # save()/restore() is the matcher's OWN bracket (see _try at
        # :1418) - an invented snapshot()/restore() pair made this
        # explainer raise AttributeError and the player got the bare
        # refusal anyway (caught live on the USER's replayed feed).
        snap = rng.save() if rng is not None else None
        try:
            post = ai_apply_for_probe(state, a)
        except BaseException:
            # BaseException on purpose: NeedClientCall is one, so a
            # bare `except Exception` would let it escape and turn a
            # refusal EXPLANATION into a crash.
            post = None
        finally:
            if snap is not None:
                try:
                    rng.restore(snap)
                except Exception:
                    pass
        if post is None:
            continue
        got = delta_facts(state, post)
        diff = _facts_differ(want, got, action=a)
        # the card the window PLAYED is the player's intent: a
        # candidate playing that card ranks above every other however
        # many board facts it misses, so the notice names what the
        # play lacked (its placements), not 'the card played'
        score = (1 if ('the card played' in diff and want.cards_played)
                 else 0, len(diff))
        if best_score is None or score < best_score:
            best, best_score, best_got = a, score, got
    if best is None:
        return None, []
    try:
        desc = engine_play.describe_action(state, side, best)
    except Exception:
        desc = str(getattr(best, 'action_type', '?'))
    return desc, _facts_differ(want, best_got, action=best)


def ai_apply_for_probe(state, action):
    """Apply one action to a CLONE for diagnosis only.  Never touches
    the live state, and a candidate that needs client randomness is
    simply not a candidate for the explanation."""
    import ai
    try:
        return ai.apply_action(state, action)
    except BaseException as e:
        # matched BY NAME, the way _try does at :1431 - NeedClientCall
        # is a BaseException and importing it here would add a cycle
        if type(e).__name__ == 'NeedClientCall':
            return None
        raise


def _subset(eng, facts, combat_flux=False, relax_nations=False):
    """May this chain CONTINUE — is the engine delta so far a
    non-contradicting PART of the window?

    BOARD rows (armies/leaders/named) use the WAYPOINT principle: a
    multi-leg gesture passes through intermediate states whose row
    values overshoot the window's NET (Ringwraiths specimen: a nazgul
    stops in Moria then rides on — mid-chain Moria +1, net 0), so a
    row may be in ANY flux while its region is one the window spoke
    of; a nonzero row in a region the window NEVER mentioned is a
    contradiction.  Dice rows likewise flux through faces the window
    touched (ring flip +P then play -P).  Scalar/one-way quantities
    (draws, political, rings, corruption) keep directional bounds.
    An EMPTY delta is allowed: an action that only OPENS a pending
    (play_event -> card_option) expresses nothing yet; expansion is
    bounded to open pendings/multi-step halves plus depth+budget.

    combat_flux (chain contains a combat leg): ARMY rows switch from
    the directional movement bound to the waypoint rule too — a
    casualty-then-downgrade sequence walks a row down and up (Grond
    3-round siege, game 2023-01-13 T3: gondor elite -3 / regular +3
    from the pool, then -2 killed; the directional bound pruned the
    TRUE chain at its first casualty leg).  Movement-only chains keep
    the tight bound (units move once per leg)."""
    win_regs = set(r for (r, _t) in facts.armies)
    win_regs.update(facts.leaders)
    win_regs.update(r for (_n, r) in facts.named)
    eng_army_rows = (
        {(k, t): v for (k, t), v in _nation_blind(eng.armies).items()}
        if relax_nations else eng.armies)
    win_army_get = ((lambda k: _nation_blind(facts.armies).get(k, 0))
                    if relax_nations
                    else (lambda k: facts.armies.get(k, 0)))
    for k, v in eng_army_rows.items():
        if not v:
            continue
        if combat_flux:
            if k[0] not in win_regs:
                return False
            continue
        # units move ONCE per gesture leg: directional bound (the
        # waypoint riders are the leader-ish pieces, never the ranks)
        w = win_army_get(k)
        if (v > 0 and w < v) or (v < 0 and w > v):
            return False
    for reg, v in eng.leaders.items():
        if v and reg not in win_regs:
            return False
    for (_nm, reg), v in eng.named.items():
        if v and reg not in win_regs:
            return False
    for k, v in eng.draws.items():
        if not v:
            continue
        w = facts.draws.get(k, 0)
        if (v > 0 and w < v) or (v < 0 and w > v):
            return False
    for k, v in eng.political.items():
        if not v:
            continue
        if combat_flux:
            # the attacked nation's p.34 advance is engine bookkeeping
            # the client may never render (see _covers docstring)
            continue
        w = facts.political.get(k, 0)
        if (v > 0 and w < v) or (v < 0 and w > v):
            return False
    for k, v in eng.rings.items():
        if not v:
            continue
        w = facts.rings.get(k, 0)
        if (v > 0 and w < v) or (v < 0 and w > v):
            return False
    for k, v in eng.dice.items():
        if v and k not in facts.dice_faces:
            return False
    for k, v in eng.sieges.items():
        if v and k not in facts.sieges and k not in win_regs:
            return False
    if eng.corruption:
        w, v = facts.corruption, eng.corruption
        if (v > 0 and w < v) or (v < 0 and w > v):
            return False
    for nm, v in eng.comps_left.items():
        if v and facts.comps_gone.get(nm, 0) not in (0, v):
            return False
    if eng.guide is not None and facts.guide_set is not None \
            and facts.guide_set != eng.guide:
        return False
    return True          # non-contradiction is the whole test
