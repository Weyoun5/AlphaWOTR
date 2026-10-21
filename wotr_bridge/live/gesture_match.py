"""Human gesture window -> engine action (design rule 4).

THE USER RULING THAT SHAPES THIS MODULE (quoted in the monolith at
:5433-5440): gesture legality is judged by simulate-and-compare over
legal moves â€” the engine is the sole judge.  What this module fixes is
the COST and the SCOPE, not the ruling: the old matcher applied EVERY
legal action with a depth-3 pending expansion over the full action
space (measured 68,731 applies / 48.1s for ONE drag, 60,000-apply
budget as band-aid).  Here:

  * candidates come from the CURRENT engine menu only
    (ai.get_legal_actions / ai.describe_pending) â€” tens of actions;
  * comparison is DELTA-based on the lifted projection
    (client_board.board_delta, increment semantics) so a one-tick view
    offset cannot refuse a legal action;
  * pending-chain expansion happens ONLY while the window still has
    unexplained content, depth-capped, apply-budgeted; exhaustion
    HOLDS, never refuses (B13-1 doctrine);
  * identity-carrying gestures (a card play names its title, a ring
    walk names its marker) are matched by VOCABULARY first â€” no
    simulation needed when the client already said which action it is.

Every apply runs on a CLONE (ai.apply_action semantics: original
untouched) â€” the transactional tick (design rule 2).  A match's
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
    The region test is board.regions membership after sanitize â€” the
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


_REGION_SPELLINGS = None


def region_lookup(zl):
    """Sanitized client zone text -> canonical engine region key, or
    None.  The client labels some regions with a definite article the
    canonical key lacks ('The Grey Havens' vs 'grey havens' â€” found on
    the Gandalf-the-White window, game 2023-01-13 T2) while 'the
    shire' keeps its article; membership is tried both ways. Possessive
    punctuation follows the chat resolver: straight/curly apostrophes and
    the apostrophe-free spelling identify the same complete region name.
    Partial names never gain chat's prefix matching in a board gesture."""
    global _REGION_SPELLINGS
    import board
    zl = sanitize(str(zl or ''))
    if zl in board.regions:
        return zl
    if zl.startswith('the ') and zl[4:] in board.regions:
        return zl[4:]
    if _REGION_SPELLINGS is None:
        from .chat import _fold_region
        _REGION_SPELLINGS = {}
        for key in board.regions:
            _REGION_SPELLINGS.setdefault(_fold_region(key), set()).add(key)
    from .chat import _fold_region
    folded = _fold_region(zl)
    matches = _REGION_SPELLINGS.get(folded)
    if not matches and folded.startswith('the '):
        matches = _REGION_SPELLINGS.get(folded[4:])
    if matches and len(matches) == 1:
        return next(iter(matches))
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
           expressed ('draws Shadow Strategy Card.' â€” the event-die
           draw action and card-effect draw grants; hunt-tile and
           random-companion 'draws' are RANDOMNESS records, not intent)
    political: Counter[nation_enum_NAME] -> engine-direction political
           increments.  CLIENT track numbers DESCEND toward war
           ('Political Track 0' = At War box; engine boxes ASCEND,
           3 = At War), so engine_increment = client_from - client_to.
    activations: nation enum NAME -> observed flip parity. False entries
           retain an expressed flip/unflip and constrain the net engine delta.
    fel_flips: [(region_key, 'flip'|'reveal')] fellowship flips in
           record order.  A plain flip is the human's hide/declare
           gesture; a reveal-style flip ('... Previously on Fellowship
           Track N') is the hunt/engine consequence AND expresses the
           track reset (fs_progress -= prev_track) â€” the client renders
           no separate FSP move for it.
    fel_moves: [(from_key, to_key)] 'The Fellowship' pawn region moves
           (the reveal/declare placement path; final entry names the
           destination the engine must land on)
    fel_endpoint: last observed pawn region across both moves and flips;
           correcting a placement after flipping supersedes the flip's region
    guide_set: canonical companion name moved onto 'Guide Of The
           Fellowship' (the guide-change gesture), or None
    comps_gone: Counter[name] companions killed OUTSIDE regions
           (Current Action / casualties zones) â€” hunt/effect casualties
    hand_discards: Counter[(side_name, deck)] identity-less card kills
           in a hand zone ('kills Shadow Character Card in Shadow
           Hand.' â€” the hand-limit discard gesture)
    cards_played: [card title strings] (hand -> play area, identity)
    combat_cards: [card title strings] chosen via the client's card
           chooser ('... Hand (chosen)' source) â€” played as COMBAT
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
    unmapped: records this converter could not express (LOUD â€” the
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
        self.activations = {}
        self.fel_flips = []
        self.fel_moves = []
        self.fel_endpoint = None
        self.fel_hidden = None  # known final face after an explicit reveal
        self.guide_set = None
        self.comps_gone = Counter()
        self.hand_discards = Counter()
        self.cards_played = []
        self.hunt_reduction_cards = []
        self.table_activations = []
        self.table_discards = []
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
                    or self.draws or self.political or any(self.activations.values())
                    or self.fel_flips
                    or self.fel_moves or self.guide_set
                    or self.comps_gone or self.hand_discards
                    or self.cards_played or self.combat_cards or self.hunt_reduction_cards
                    or self.table_activations or self.table_discards
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
        # the Mordor-track boxes physically overlay GORGOROTH â€” army
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
        # Physical grouping does not choose an engine action order. Compare
        # the complete named-piece delta against legal decision endpoints.
        facts.named[(a, reg)] += sign * count
    elif kind in ('ctrlmarker', 'vp', 'tile', 'die', 'nation'):
        pass          # client bookkeeping the engine derives itself
    elif kind == 'card' and title_to_cid(a) is not None:
        # A known card carried with figures is not an unknown army piece.
        # Hand plays and table use/disposal retain their identities in the
        # dedicated window_facts branches.
        pass
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


def _draw_deck_matches(action, draws):
    if action.action_type != 'draw_card':
        return True
    owner = 'FREE_PEOPLES' if action.params.get('side', 'fp') == 'fp' else 'SHADOW'
    return draws.get((owner, action.params.get('deck')), 0) > 0


def window_facts(window):
    """Convert a closed Window's body records to WindowFacts."""
    f = WindowFacts()
    staged = [getattr(window, 'die_face', None)]   # chased through
    #   die_change records: the client renames the staged die in place
    w_side = getattr(window, 'die_side', None)
    from .staged_event_reveal import normalized_records
    for rec in normalized_records(window):
        # Native kill paragraphs always print Current Action, even when the
        # piece was on BoardN. The driver binds that public table identity
        # before its seat/PID bookkeeping forgets the killed piece.
        f.table_discards.extend(rec.get('_table_discards') or ())
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
            if (src[0] == 'other' and re.fullmatch(r'board\d*', src[1])
                    and (dst[0] in ('play', 'dead')
                         or 'discard' in str(dst[-1]))):
                cid = title_to_cid(obj)
                if cid is not None:
                    # Card identity is evidence regardless of its particular
                    # effect. Legal pending actions decide what it can do.
                    if dst[0] == 'play':
                        f.table_activations.append(cid)
                        if cid in ('FREE_PEOPLES_Character_Axe_and_Bow',
                                   'FREE_PEOPLES_Character_Horn_of_Gondor'):
                            f.hunt_reduction_cards.append(cid)
                    else:
                        f.table_discards.append(cid)
                    continue
            if src[0] == 'hand' and (
                    dst[0] == 'play'
                    or (dst[0] == 'other'
                        and re.match(r'^board\d*$', dst[1]))):
                # hand -> play area OR hand -> table slot (BoardN):
                # persistent/table cards render straight onto the
                # table (Denethor's Folly specimen, game 2023-01-13
                # T3) â€” both are the card-play gesture.  Later
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
                f.fel_endpoint = dst[1]
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
                # 'Fellowship Of The RING' â€” companion separations
                # classified as ring walks and expressed nothing
                # (game 2023-01-28 T1: the Strider crown route died
                # here, all four legs)
                f.ring_moves.append((obj, rec.get('from'),
                                     rec.get('to')))
                # ring COUNT deltas: an FP use walks the marker from
                # an Elven Ring slot to a Dark Ring slot; a Shadow use
                # spends it out of the Dark Ring slots.  ZONE NAMES
                # VARY BY MODULE BUILD: the 2023 corpus renders 'Elven
                # Ring N'/'Dark Ring N', the packaged WOTR-Gandalf2
                # build renders the named active slots ('Vilya
                # Slot'...) and 'Used Ring N' (Shadow-collected) - both
                # vocabularies count.
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
                if title_to_cid(p) in (rec.get('_table_discards') or ()):
                    continue
                # FLA-13 (par5 g2#6 + g1#3): a companion dragged INTO
                # a casualties zone is the same casualty gesture as
                # the kill form.  The move produced EMPTY facts before
                # (_apply_piece ignores non-region zones), so
                # _rec_inert dropped the hunt guide-sacrifice
                # performance and the pending never closed.
                if dst[0] == 'dead':
                    _ppd = rr.parse_piece(p)
                    if _ppd is not None and _ppd[0] == 'named' \
                            and _ppd[1] in _COMPANION_NAMES:
                        f.comps_gone[_ppd[1]] += 1
                        continue
                _apply_piece(f, p, src, -1)
                _apply_piece(f, p, dst, +1)
        elif op == 'kill':
            ring_pieces = rec.get('pieces') or []
            ring_zone = str(rec.get('zone', '')).strip().lower()
            if len(ring_pieces) == 1 and str(ring_pieces[0]).lower() in ('vilya', 'nenya', 'narya') \
                    and ring_zone.startswith(('dark ring', 'used ring')):
                f.ring_moves.append((ring_pieces[0], rec.get('zone'), 'SA Casualties'))
                f.rings['SH'] -= 1
                continue
            zk = zone_key(rec.get('zone'))
            if zk[0] in ('region', 'stronghold'):
                for p in rec.get('pieces', []):
                    _apply_piece(f, p, zk, -1)
                continue
            for p in rec.get('pieces', []):
                if zk[0] == 'other' and re.fullmatch(r'board\d*', zk[1]):
                    cid = title_to_cid(p)
                    if cid is not None:
                        f.table_discards.append(cid)
                        continue
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
                    # FLD-2b (par5b g4#6, '$Wey kills On, Shadow card
                    # in Shadow Hand.'): a comma-carrying title
                    # renders truncated, and ANY card kill in a hand
                    # zone IS the discard gesture - identity rides the
                    # sidecar pid (driver._discard_identity_filter),
                    # never the rendered name.  Unmapped here wedged
                    # the pending buffer ('matches none of its 7
                    # legal options').
                    f.hand_discards[
                        ('FREE_PEOPLES' if zk[1] == 'FP'
                         else 'SHADOW', None)] += 1
                    continue
                # dice / tiles / cards killed in off-board zones are
                # client bookkeeping the engine derives itself
        elif op == 'flip':
            fobj = str(rec.get('obj', '') or '').strip().lower()
            fzone = str(rec.get('zone', '') or '').lower()
            marker = rr.parse_piece(fobj)
            if (marker and marker[0] == 'nation'
                    and zone_key(fzone)[0] == 'pol_track'):
                nation = marker[1]
                f.activations[nation] = not f.activations.get(nation, False)
                continue
            if (fobj in ('the fellowship', 'fsp')
                    and not fzone.startswith(('political', 'mordor'))):
                # BOTH pawn and FSP-MARKER forms: some clients keep
                # the fellowship on the track and render hide/reveal
                # as 'flips FSP in Fellowship Track N' (game
                # 2023-01-22) â€” with 'Previously on' it is the
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
                    f.fel_hidden = False
                    f.fs_progress += -int(rec.get('prev_track', 0))
                elif reg or fobj == 'fsp':
                    f.fel_flips.append((reg, 'flip'))
                    if f.fel_hidden is not None:
                        f.fel_hidden = not f.fel_hidden
                if reg:
                    f.fel_endpoint = reg
                continue
            if fobj in _FLIP_RENAME and rec.get('region'):
                # dual-faced figure flipped to its OTHER identity
                # (Gandalf the White transform: the client moves the
                # Grey piece then flips it) â€” rewrite the arrival.
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
    # A completed House printed effect may still have its paid die on
    # Current Action while the player resolves hand-limit/guide choices.
    if w_side and staged[0] and (getattr(window, 'closed', None)
            in ('die_killed', 'die_to_hunt')
            or getattr(window, 'card_payment_staged', False)):
        f.dice_faces[(w_side, staged[0])] -= 1
    return f


# ---------------------------------------------------------------------------
# engine delta -> the same fact space
# ---------------------------------------------------------------------------
class EngFacts(object):
    """Engine apply-on-clone delta, in window_facts' key space."""
    __slots__ = ('armies', 'leaders', 'named', 'fs_progress',
                 'corruption', 'draws', 'political', 'activations', 'hidden_change',
                 'fel_region', 'fel_moved', 'fel_hidden', 'guide', 'comps_left',
                 'hand_delta', 'rings', 'dice', 'sieges')

    def __init__(self):
        self.armies = Counter()
        self.leaders = Counter()
        self.named = Counter()
        self.fs_progress = 0
        self.corruption = 0
        self.draws = Counter()      # (side_name, deck) -> cards drawn
        self.political = Counter()  # nation NAME -> box increment
        self.activations = {}       # nation NAME -> net active-state toggle
        self.hidden_change = False  # fellowship hidden toggled?
        self.fel_region = None      # post-state fellowship region key
        self.fel_moved = False      # the pawn's region changed pre->post
        self.fel_hidden = None
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
    ef.fel_hidden = bool(post.fellowship.hidden)
    before_active = getattr(getattr(pre, 'political', None), 'active', {}) or {}
    after_active = getattr(getattr(post, 'political', None), 'active', {}) or {}
    for nation in set(before_active) | set(after_active):
        if bool(before_active.get(nation, False)) != bool(after_active.get(nation, False)):
            ef.activations[getattr(nation, 'name', str(nation))] = True
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
                        # next roll â€” not an observable board face (the
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
    # __named__ already carries â€” fold it out: ef.leaders is the
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
        reg0 = pre.fellowship.region
        ef.fel_moved = (sanitize(str(getattr(reg0, 'key', reg0)))
                        .strip().lower() != ef.fel_region)
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
    """Army rows aggregated to (region, kind) â€” the nation half of the
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
        # B-W2-05 (ii) (COORD #2 2026-09-03, USER bridge go): the
        # tolerance covers a relocation the ENGINE performed and the
        # window never voiced; a relocation the WINDOW voiced (a human
        # drag of the advancing side's stack) is never the engine's -
        # compare against the engine delta, not mere count conservation.
        if win_rows.get(k, 0):
            return False
        diffs[group(k)] = diffs.get(group(k), 0) + d
    return all(v == 0 for v in diffs.values())



def _forced_defender_allocation_rows(state, side):
    """Exact army follow-through after the human attacker's allocation.

    Only a defender with no allocation choice advances automatically. Use
    the same isolated allocation on a clone to witness its delta, including
    a forced Elite downgrade; do not infer permission from removal signs.
    """
    import ai
    pc = getattr(state, 'pending_combat', None) or {}
    if (pc.get('phase') != 'casualty_allocation'
            or pc.get('alloc_stage') != 'attacker'
            or pc.get('attacker_side') is not side
            or pc.get('deciding_side') is not side):
        return {}
    defender = pc.get('defender_side')
    region = pc.get('defender_region')
    hits = int(pc.get('def_hits', 0) or 0)
    if defender is None or defender is side or not region or hits <= 0:
        return {}
    forced = ai.clone_state(state)
    ai._forced_casualty_allocation(forced, region, defender, hits)
    return {k: v for k, v in delta_facts(state, forced).armies.items() if v}


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

# LIVE-ONLY FOLLOW-THROUGH TABLE (tracker hygiene after BL-94(a),
# BL-142, BL-144 - three special cases that were one rule): engine
# follow-through that the human's choreography renders as a marker, a
# flip, or nothing at all is tolerated on the candidate whose apply
# already carried it; the driver's post-commit machinery renders it
# for AI pieces and DEMANDS it for the human's (echo ledger, owed-track
# markers, draw asks).  Keyed by the CHAIN's action types; each row
# names the fact families a window may leave to the engine.  The
# spectator's byte-pinned census never reads this table (strict
# parity).  The fourth instance is a ROW here, never a special case.
FOLLOW_THROUGH = {
    # BL-94(a): a card play's EFFECT legs - Imrahil's
    # recruits, a heal's corruption walk, an effect draw, the card's
    # own fellowship-track leg.  No menu row carries unit/leader
    # params, so the gesture can never express them.  'draws' here is
    # NOT license to route fed draws back into window facts: a draw
    # supplied against a funnel demand credits the funnel BEFORE
    # window membership (BL-132 L2) and rarely reaches _covers at
    # all; the row covers the engine's own effect draws only.
    'play_event': frozenset(('armies', 'leaders', 'named', 'fs_progress',
                             'corruption', 'draws')),
    # BL-142: declaring in a Free Peoples city/stronghold
    # heals 1 corruption; the declare choreography (pawn walk,
    # progress reset, flip-unflip pair) voices no marker.
    'fellowship_declare': frozenset(('corruption',)),
    # BL-144: the hunt already revealed the Fellowship; the
    # placement choreography's marker flip is the client catching up,
    # not a toggle demand.
    'fellowship_reveal_place': frozenset(('hidden_marker',)),
    # Ending a battle can grant the Witch-king's Sorcerer draw. The engine
    # owns that draw; removing casualties need not also voice the AI's card.
    'casualty_allocate': frozenset(('draws',)),
    # FLD-4b (par5b_g2 siege_trim stall): the garrison's
    # seat-into-the-box drag is CLIENT choreography for a siege the
    # engine ALREADY holds - the trim option itself moves no siege
    # flag, so strict siege parity refused the row's own gesture.
    'siege_trim': frozenset(('sieges',)),
    # B-W2-04 (W2-025, amendment 1.49 sub-bullet 2026-09-03): the
    # garrison's seat-into-the-box drag IS the human's
    # retreat_into_siege, but the siege FLAG is now set by the
    # ATTACKER's own victor_advance (the opponent's open engine
    # pending, decided by the AI), never inside the retreat - strict
    # siege parity refused the human's only gesture and the chain
    # search wandered into the AI's decision (two composition chains
    # -> 'ambiguous' refusal, test_siege_withdraw_window).
    'retreat_into_siege': frozenset(('sieges',)),
}


def follow_through_for(chain, live):
    """The fact families a LIVE window may leave to the engine for
    this candidate chain (union over the chain's action types); None
    for the spectator (strict parity, and the wiring pin can tell the
    two apart)."""
    if not live:
        return None
    out = set()
    for a in chain or ():
        out |= FOLLOW_THROUGH.get(str(getattr(a, 'action_type', '')),
                                  frozenset())
        params = getattr(a, 'params', None) or {}
        choices = params.get('choices') or {}
        if (getattr(a, 'action_type', '') == 'card_option'
                and params.get('card_id') in (
                    'FREE_PEOPLES_Strategy_Book_of_Mazarbul',
                    'FREE_PEOPLES_Strategy_Fear!_Fire!_Foes!',
                    'FREE_PEOPLES_Character_There_and_Back_Again')):
            # A staged play has already left the human's effect window.
            # Its legal companion choice (or explicit movement finish) still
            # owns the printed "Then" political consequence. The post-commit
            # sweep supplies unvoiced steps; any voiced step must agree.
            out.add('political')
        if getattr(a, 'action_type', '') == 'card_option' \
                and params.get('card_id') == 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow' \
                and choices.get('target_region') \
                and choices.get('companions'):
            # The already-played card's chosen group is the human gesture.
            # Its attack advances the defenders automatically; the live
            # post-commit sweep renders those markers for their owner.
            out.add('political')
        if (getattr(a, 'action_type', '') == 'card_option'
                and params.get('card_id') == 'SHADOW_Strategy_Corsairs_of_Umbar'
                and choices.get('target_region') and '_cargo_index' in choices):
            # Sailing figures identify this already-played card's landing.
            # The legal landing itself supplies the attacked/captured nation's
            # political consequence; a voiced marker move must still agree.
            out.add('political')
    return frozenset(out)


def _covers(win, eng, relax_draws=False, fs_reset_relax=False,
            combat_flux=False, relax_nations=False,
            auto_advance_side=None, card_leg_tolerance=False,
            reveal_place=False, follow_through=None, forced_army_rows=None):
    """Does the engine delta EXPRESS everything the window expressed?
    Exact equality on army/draw/political rows; named/leader rows must
    agree where the window spoke.  Fellowship hidden-state is checked
    ONLY when the window carried a flip gesture: reveal-style flips
    are engine-derived and deliberately absent from window facts, so
    a card/hunt chain toggling hidden without a windowed flip is not
    a refusal.

    relax_draws (spectator fallback lane ONLY, always paired with a
    NAMED census row): the draw rows become a bounded-subset test â€”
    the engine must draw AT LEAST one and AT MOST the window's count
    of every deck the window drew, and nothing the window did not.
    Exists for the measured client-permitted misplay class (game
    2023-01-13: two draws rendered on 'draw one Strategy card' plays,
    raw-log pids 122/120 + 146/134 prove two REAL draws): the engine
    commits the rules-legal subset, the extra is the finding.

    combat_flux (the chain contains a combat leg): POLITICAL rows the
    window voiced must match exactly, but ENGINE-EXTRA political rows
    are tolerated â€” being attacked advances the defender's track
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
    menu â€” the engine commits the closest composition, the census row
    keeps the engine debt (fix owed: composition-aware move_army).

    card_leg_tolerance (LIVE lane ONLY - rng is None at the call
    site; the spectator's byte-pinned census keeps strict parity):
    BL-94(a), fpgame10 G1.  On a window whose facts carry a CARD
    PLAY, the card EFFECT's board legs (Imrahil's recruits, a heal's
    corruption walk, an effect draw) are ENGINE follow-through the
    gesture can never express - no menu row carries unit/leader
    params, so strict parity refuses the play and re-selecting loops
    forever.  Rule (the political-rider precedent below,
    generalized): every row the window VOICED must match exactly;
    ENGINE-EXTRA rows are tolerated and land in the post-commit
    sweep's banking - rendered for AI pieces, DEMANDED for the
    human's (BL-77 echo machinery / owed-track markers / draw asks).
    A wrong voiced drag still refuses."""

    def _voiced_subset94(w, e):
        return all(e.get(k, 0) == v for k, v in w.items())
    # the follow-through families this candidate may leave to the
    # engine (FOLLOW_THROUGH table); the two legacy flags are aliases
    # for direct callers (pins) - the matcher passes the table lookup
    ft = set(follow_through or ())
    if card_leg_tolerance and win.cards_played:
        ft |= FOLLOW_THROUGH['play_event']
    if reveal_place:
        ft |= FOLLOW_THROUGH['fellowship_reveal_place']
    w_arm = {k: v for k, v in win.armies.items() if v}
    e_arm = {k: v for k, v in eng.armies.items() if v}
    if relax_nations:
        if _nation_blind(win.armies) != _nation_blind(eng.armies):
            return False
    elif w_arm != e_arm:
        if (forced_army_rows
                and not any(k in w_arm for k in forced_army_rows)
                and e_arm == {**w_arm, **forced_army_rows}):
            pass       # exact forced opponent allocation, emitted by sweep
        elif 'armies' in ft and _voiced_subset94(w_arm, e_arm):
            pass       # follow-through: card-effect board legs, banked
        elif auto_advance_side is None or not _advance_extra_ok(
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
        if 'leaders' in ft and _voiced_subset94(w_led, e_led):
            pass       # follow-through: card-effect leader legs, banked
        elif combat_flux and _only_extra_removals(w_led, e_led):
            pass
        elif auto_advance_side is None or not _advance_extra_ok(
                w_led, e_led, auto_advance_side,
                lambda k: auto_advance_side,
                group=lambda k: '_'):
            return False
    w_nam = {k: v for k, v in win.named.items() if v}
    e_nam = {k: v for k, v in eng.named.items() if v}
    if w_nam != e_nam:
        if 'named' in ft and _voiced_subset94(w_nam, e_nam):
            pass       # follow-through: card-effect named legs, banked
        elif combat_flux and _only_extra_removals(w_nam, e_nam):
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
        # placement prefix tolerate the engine reset â€” a GROWING
        # buffer must stay strict or a one-record prefix covers before
        # the flip arrives (game 2023-01-02 regression: the pawn
        # stranded at Fords of Bruinen).
        if 'fs_progress' in ft and not win.fel_moves:
            pass       # follow-through: a card's own fellowship-track
            #            leg (There is Another Way etc.) - the owed-
            #            pawn/track machinery demands the client walk
        elif not (fs_reset_relax and win.fel_moves
                  and eng.fs_progress <= win.fs_progress <= 0):
            return False
    if win.corruption != eng.corruption:
        # follow-through: a heal/corrupt track leg the window never
        # voiced (Athelas from the menu drags no marker - BL-94(a);
        # the sanctuary heal on a declaration - BL-142); the owed
        # corruption marker (pid 252 sweep) renders it.  A voiced
        # marker walk still has to match exactly.
        if not ('corruption' in ft and win.corruption == 0):
            return False
    w_drw = {k: v for k, v in win.draws.items() if v}
    e_drw = {k: v for k, v in eng.draws.items() if v}
    if relax_draws:
        for k in set(w_drw) | set(e_drw):
            e, wv = e_drw.get(k, 0), w_drw.get(k, 0)
            if e > wv or (wv > 0 and e < 1) or (wv == 0 and e):
                return False
    elif w_drw != e_drw:
        # BL-94(a): a card's effect draw the window did not render -
        # the grant_draw/draw_ask machinery demands the client draw.
        # Every RENDERED draw must be engine-accounted (a human
        # over-draw still refuses).
        if not ('draws' in ft and all(e_drw.get(k, 0) >= v
                                      for k, v in w_drw.items())):
            return False
    w_pol = {k: v for k, v in win.political.items() if v}
    e_pol = {k: v for k, v in eng.political.items() if v}
    if combat_flux or win.cards_played or 'political' in ft:
        # voiced rows exact; engine-extra rows = AUTOMATIC advances
        # the client does not render for the human: the attacked
        # nation's p.34 advance in battle chains (see docstring), and
        # a CARD EFFECT'S political rider on a play window (tester
        # 2026-08-24: the human's 'A Power too Great' was refused
        # 'differs on the political track' + 5 undo presses because
        # the Elves advance was not dragged in the same window; the
        # play could never stay on the board).  The post-commit sweep
        # walks the client marker for the un-voiced advance, so the
        # board still ends right.
        for k, v in w_pol.items():
            if e_pol.get(k, 0) != v:
                return False
    elif w_pol != e_pol:
        return False
    # A flip is a toggle, not a declaration that a nation is active.
    # Unvoiced automatic activations retain their existing sweep ownership.
    if any(eng.activations.get(nation, False) != parity
           for nation, parity in win.activations.items()):
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
            # FLD-4b: a chain granted 'sieges' follow-through
            # (siege_trim) tolerates the window's siege-FORMING drag
            # for an already-besieged engine stronghold; a siege
            # LIFT stays strict
            if 'sieges' in ft and v > 0:
                continue
            return False
    if win.fel_flips:
        if win.fel_hidden is not None and eng.fel_hidden != win.fel_hidden:
            return False
        # plain flips are the human's own toggle gesture â€” strict
        # parity.  A window carrying a REVEAL flip keeps only the
        # region + track-reset constraints: the engine's hidden toggle
        # may have landed in an earlier step (the hunt reveal fires
        # inside the fellowship-move apply; the client renders the
        # flip with the consequence records that follow).
        if all(k == 'flip' for _r, k in win.fel_flips):
            want_toggle = (len(win.fel_flips) % 2) == 1
            # follow-through 'hidden_marker' (BL-144): the engine
            # already toggled hidden inside the hunt; the placement
            # choreography's marker flip is the client catching up,
            # not a toggle demand (region parity below stays strict)
            if eng.hidden_change != want_toggle \
                    and not ('hidden_marker' in ft and want_toggle
                             and not eng.hidden_change):
                return False
    if win.fel_endpoint and eng.fel_region != win.fel_endpoint:
        return False
    if not win.fel_moves and not win.cards_played \
            and (win.fs_progress or win.fel_flips) \
            and not any(k == 'reveal' for _r, k in win.fel_flips) \
            and getattr(eng, 'fel_moved', False):
        # BRLANE-11: a track
        # reset / plain flip with NO pawn walk means the pawn STAYED -
        # a candidate that moves the pawn (a declaration elsewhere) is
        # not this gesture; declaring IN PLACE is then unique.  (A
        # hunt REVEAL flip keeps its engine-derived latitude; a card
        # play's pawn leg is follow-through the pawn sweep renders.)
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
                 note='', starved=False, candidates=None):
        self.status = status      # 'match'|'ambiguous'|'none'|'hold'
        self.chain = chain or []  # [Action, ...] (action + pendings)
        self.post = post          # committed-state candidate
        self.applies = applies
        self.note = note
        # True when at least one candidate STARVED the client-
        # randomness feed (NeedClientCall) and nothing matched: the
        # gesture's randomness has not arrived yet â€” the caller may
        # retry after feeding more (spectator deferral), never refuse
        self.starved = starved
        # Completed legal endpoints before presentation/route-count ranking.
        # The live owner can compare these with its verified physical board;
        # execution keeps RNG consumption distinct even at equal boards.
        self.candidates = candidates or []

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
# casualty, hobbit placement) â€” depth 3 and then 5 stranded real
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
        cl = FACE_LETTER.get(str(cur), '?')
        # BL-32: WILL is the wildcard face - a staged W matches every
        # candidate; a W-funded candidate matches any staged face (one
        # answerer with the menu side)
        if die_face == 'W' or cl == 'W':
            return True
        return cl == die_face
    return ok


def _inferred_ring_target(state, side, window, facts):
    """A ring funds the staged die without requiring a physical face flip.

    The driver supplies the sidecar-resolved engine index. Without identity
    metadata only a unique unused die showing the exact staged face is safe.
    Explicit face-change gestures retain their existing conversion semantics.
    """
    if not facts.ring_moves or not getattr(window, 'die_face', None) \
            or any(r.get('op') == 'die_change' for r in window.records):
        return None, None
    if getattr(window, 'die_side', None) != side.name:
        return None, 'ring payment does not belong to the staged side'
    dice = state.dice.get(side, [])
    matches = [i for i, die in enumerate(dice) if not die.used
               and FACE_LETTER.get(getattr(die.current, 'name', '')) == window.die_face]
    target = getattr(window, '_inferred_ring_target', None)
    if target is None:
        if len(matches) != 1:
            return None, 'ring payment needs the staged die identity'
        target = matches[0]
    if type(target) is not int or target not in matches:
        return None, 'ring payment target does not match the staged unused die'
    return target, None


def _typed_value_key(root, *, ordered_mappings=False, _known_values=None):
    """Immutable typed values without object-allocation or dict-order artifacts."""
    from enum import Enum
    active = set()
    memo = {}
    tags = {}

    def value_key(value):
        cls = type(value)
        tag = tags.get(cls)
        if tag is None:
            tag = tags[cls] = (cls.__module__, cls.__qualname__)
        if isinstance(value, Enum):
            return ('enum', tag, value.name)
        if value is None or cls in (str, bytes, int, bool):
            return (tag, value)
        if cls is float:
            return (tag, value.hex())
        ident = id(value)
        if _known_values is not None and ident in _known_values:
            return _known_values[ident]
        if ident in active:
            raise ValueError('cyclic candidate state')
        if ident in memo:
            return memo[ident]
        active.add(ident)
        try:
            if cls is dict:
                rows = [(value_key(k), value_key(v)) for k, v in value.items()]
                result = (tag, tuple(rows if ordered_mappings else sorted(rows, key=lambda row: repr(row[0]))))
            elif cls in (set, frozenset):
                result = (tag, tuple(sorted((value_key(v) for v in value), key=repr)))
            elif cls in (list, tuple):
                result = (tag, tuple(value_key(v) for v in value))
            elif isinstance(value, (dict, set, frozenset, list, tuple)):
                # Container subclasses can carry extra semantics (for example
                # defaultdict.default_factory). Do not silently drop them.
                raise TypeError('unsupported candidate container: %s' % (tag,))
            else:
                fields = dict(vars(value)) if hasattr(value, '__dict__') else {}
                for base in cls.__mro__:
                    slots = getattr(base, '__slots__', ())
                    if isinstance(slots, str):
                        slots = (slots,)
                    for field in slots:
                        if field not in ('__dict__', '__weakref__') and hasattr(value, field):
                            fields[field] = getattr(value, field)
                if not fields:
                    raise TypeError('unsupported candidate state value: %s' % (tag,))
                result = (tag, tuple((name, value_key(v)) for name, v in sorted(fields.items())))
            memo[ident] = result
            return result
        finally:
            active.remove(ident)

    return value_key(root)


def _semantic_state_key(post, paid_ring=None):
    """Full-state equality with canonical army regions and unordered rosters.

    A spent A versus H conversion is not a second player decision. All other
    fields, including every unused die, pending choice and hidden card, remain
    in the key. Army units/leaders are unordered collections: moving a group
    together or its members separately can insert the same pieces in a different
    order. Sort those rosters by each piece's complete typed attributes,
    and resolve army dictionary keys through the same board-region identity
    used by the rules and native projection: display and sanitized spellings
    name the same region. Colliding aliases refuse equality, never drop armies.
    Fellowship, hands, decks, pending options and other lists keep their order.
    Python allocation/sharing and dictionary insertion order are not game rules.
    A typed value traversal avoids pickle memo-layout false ambiguities. Unknown
    values or cycles refuse equivalence instead of falling back to a weak key.
    """
    import copy
    cloned = copy.copy(post)
    if paid_ring is not None:
        side, target = paid_ring
        if not post.dice[side][target].used:
            return None
        cloned.dice = dict(post.dice)
        cloned.dice[side] = list(post.dice[side])
        old = post.dice[side][target]
        paid = copy.copy(old)
        paid.current = None
        cloned.dice[side][target] = paid
        cloned.hunt_box = [paid if die is old else die for die in post.hunt_box]
    try:
        cloned.armies = {}
        for region, army in post.armies.items():
            canonical = region_lookup(sanitize(region)) if isinstance(region, str) else None
            key = canonical if canonical is not None else region
            if key in cloned.armies:
                return None
            normalized = copy.copy(army)
            normalized.units = sorted(army.units, key=lambda piece: repr(_typed_value_key(piece)))
            normalized.leaders = sorted(army.leaders, key=lambda piece: repr(_typed_value_key(piece)))
            cloned.armies[key] = normalized
        # Preserve the engine's owned-die/Hunt alias contract explicitly; value
        # equality alone cannot identify which of two equal dice is in Hunt.
        die_ids = {id(die): (owner.name, i) for owner, dice in post.dice.items()
                   for i, die in enumerate(dice)}
        hunt_ids = tuple(die_ids[id(die)] for die in post.hunt_box)
        return (_typed_value_key(cloned), hunt_ids)
    except Exception:
        return None


def _continuation_execution(rng, *, include_probe_index=True):
    import random
    if rng is None:
        return ('global rng', random.getstate())
    from .randomness import ClientRandomness
    r = rng.r
    if type(r) is not ClientRandomness:
        raise TypeError('unknown continuation RNG adapter')
    pending = r._pending
    return (
        r._feed, r._feed_pos,
        None if pending is None else (pending.kind, pending.kw),
        r._pending_got, r.shim_uses, r.reroll_decline_shim,
        r._step_open, r._step_seed, r._probe_ix if include_probe_index else None, r._step_rng_cur,
        r._journal[r._journal_ix:], r.sticky_halt,
        None if r._step_open else random.getstate())


def _continuation_key(post, actions, rng, evidence=(), *, execution=None, known_values=None):
    """Full continuation, including hidden state and future client/RNG context.

    Unlike completed outcomes, intermediate roster and mapping order are kept:
    future indexed choices can depend on them. Unknown values skip pruning.
    """
    import hashlib
    import io
    import pickle
    try:
        # Paused combat carries identity-sensitive links to live armies and
        # supply ledgers. Value equality cannot prove those alias graphs equal.
        if (getattr(post, 'pending_combat', None) or {}).get('_round_resume'):
            return None
        if execution is None:
            execution = _continuation_execution(rng)
        die_ids = {id(die): (owner.name, i) for owner, dice in post.dice.items()
                   for i, die in enumerate(dice)}
        hunt_ids = tuple(die_ids[id(die)] for die in post.hunt_box)
        value = _typed_value_key((post, hunt_ids, actions, execution, evidence),
                                 ordered_mappings=True, _known_values=known_values)
        stream = io.BytesIO()
        writer = pickle.Pickler(stream, protocol=4)
        # The input is already a typed acyclic value tree. Disabling pickle's
        # memo removes allocation/sharing artifacts from its serialization.
        writer.fast = True
        writer.dump(value)
        return hashlib.sha256(stream.getbuffer()).digest()
    except Exception:
        return None


class _ContinuationMemo:
    """Cheap buckets only index candidates; full keys alone justify pruning.

    Delay expensive full-state traversal until a bucket actually collides.
    Retain at most 64 immutable ai.apply_action results while doing so; beyond
    that bound compute compact keys eagerly. Execution context is snapshotted
    immediately because the RNG adapter is mutable during sibling exploration.
    """
    def __init__(self, deferred_limit=64):
        self.buckets = {}
        self.deferred = 0
        self.deferred_limit = deferred_limit
        self.board_cache = None

    def _key(self, post, actions, evidence, execution):
        """Reuse typed map values only after checking every serialized field."""
        import pickle
        import board
        known = None
        if type(post.board) is board.Board:
            try:
                # Board has no custom pickle state/reduction. A C-level value
                # snapshot is much cheaper than repeatedly walking its map.
                # Compare bytes, not identity: even an accidental map mutation
                # invalidates this cache immediately.
                snapshot = pickle.dumps(post.board, protocol=4)
                if self.board_cache is None or self.board_cache[0] != snapshot:
                    self.board_cache = (snapshot, _typed_value_key(
                        post.board, ordered_mappings=True))
                known = {id(post.board): self.board_cache[1]}
            except Exception:
                pass
        return _continuation_key(post, actions, None, evidence,
                                 execution=execution, known_values=known)

    def dominated(self, bucket, post, actions, rng, evidence, depth):
        import copy
        try:
            execution = copy.deepcopy(_continuation_execution(rng))
        except Exception:
            return False
        rows = self.buckets.setdefault(bucket, [])
        key = None
        if rows or self.deferred >= self.deferred_limit:
            key = self._key(post, actions, evidence, execution)
            if key is None:
                return False
        for row in rows:
            if row[2] is not None:
                old_post, old_actions, old_execution, old_evidence = row[2]
                row[1] = self._key(old_post, old_actions, old_evidence, old_execution)
                row[2] = None
                self.deferred -= 1
            if row[1] is not None and row[1] == key:
                if row[0] >= depth:
                    return True
                row[0] = depth
                return False
        pending = None
        if key is None:
            pending = (post, copy.deepcopy(actions), execution, copy.deepcopy(evidence))
            self.deferred += 1
        rows.append([depth, key, pending])
        return False


def _continuation_history(chain, ordered):
    """Keep evidence used by later filters, not fungible piece-index choices."""
    if ordered:
        return tuple((x.action_type, x.die_idx, x.params) for x in chain)
    pairs = Counter()
    cards = []
    first_guide = None
    for action in chain:
        params = action.params or {}
        fr = to = None
        if action.action_type in ('army_move', 'victor_advance', 'nazgul_move'):
            fr, to = params.get('from'), params.get('to')
        elif action.action_type == 'card_option':
            choice = params.get('choices')
            if isinstance(choice, dict):
                fr, to = choice.get('from_region'), choice.get('to_region')
        if fr is not None and to is not None:
            frk, tok = sanitize(str(fr)).strip().lower(), sanitize(str(to)).strip().lower()
            if frk != tok:
                pairs[(frk, tok)] += 1
        if action.action_type == 'combat_card_play':
            cards.append((str(params.get('side')), params.get('card_id')))
        if first_guide is None and action.action_type == 'choose_guide':
            first_guide = str(params.get('companion') or '') or None
    return (frozenset(x.action_type for x in chain),
            tuple(sorted(pairs.items())), tuple(sorted(cards)), first_guide,
            follow_through_for(chain, True),
            sum(_leg_is_move(x) for x in chain), len(chain))


def _facts_fingerprint(eng):
    """Canonical fingerprint of an engine delta in fact space."""
    return (tuple(sorted((k, v) for k, v in eng.armies.items() if v)),
            tuple(sorted((k, v) for k, v in eng.leaders.items() if v)),
            tuple(sorted((k, v) for k, v in eng.named.items() if v)),
            eng.fs_progress, eng.corruption,
            tuple(sorted((k, v) for k, v in eng.draws.items() if v)),
            tuple(sorted((k, v)
                         for k, v in eng.political.items() if v)),
            tuple(sorted(eng.activations.items())),
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


def _unvoiced_rows(win, eng):
    """How many engine-delta rows the window did NOT voice (the
    tolerance the follow-through table had to lend a candidate to
    cover the gesture).  0 = exact parity.  Counts per family: armies,
    leaders, named, political, draws, corruption, fs_progress, hidden."""
    n = 0
    for fam in ('armies', 'leaders', 'named', 'political', 'draws'):
        w = {k: v for k, v in getattr(win, fam, {}).items() if v}
        e = {k: v for k, v in getattr(eng, fam, {}).items() if v}
        for k in set(w) | set(e):
            if w.get(k, 0) != e.get(k, 0):
                n += 1
    if int(win.corruption or 0) != int(eng.corruption or 0):
        n += 1
    if int(win.fs_progress or 0) != int(eng.fs_progress or 0):
        n += 1
    if win.fel_flips:
        want_toggle = (all(k == 'flip' for _r, k in win.fel_flips)
                       and (len(win.fel_flips) % 2) == 1)
        if bool(eng.hidden_change) != want_toggle:
            n += 1
    elif eng.hidden_change:
        n += 1
    return n


def _inferred_ring_state_key(post, side, target):
    """Same outcome comparison, with the observed spent ring die normalized."""
    return _semantic_state_key(post, paid_ring=(side, target))


def _state_fingerprint(post):
    """Complete rule state; visible equality alone cannot prove equivalence."""
    return _semantic_state_key(post)


def _leg_is_move(a):
    """A chain leg the client renders as ONE region->region drag."""
    at = str(getattr(a, 'action_type', ''))
    params = getattr(a, 'params', None) or {}
    if at in ('army_move', 'companion_move', 'nazgul_move',
              'victor_advance', 'mouth_move'):
        return not (params.get('from') is not None and params.get('to') is not None
                    and sanitize(str(params['from'])) == sanitize(str(params['to'])))
    if at == 'card_option':
        ch = params.get('choices')
        group = ch.get('move_group') if isinstance(ch, dict) else None
        return (isinstance(ch, dict) and 'move_group' in ch
                and not (isinstance(group, dict) and group.get('from') is not None
                         and group.get('to') is not None
                         and sanitize(str(group['from'])) == sanitize(str(group['to']))))
    return False


def _collapse_same_state(uniq, n_move_records=None, paid_ring=None, execution_keys=None):
    """Group legal paths by their complete resulting game state.

    Internal action names and intermediate groupings are not player choices.  Among them the survivor is the chain
    whose MOVE-leg count equals the number of region->region drags
    the human rendered (one group drag -> one move leg: the split
    twin; two drags -> the two-group chain: Fear! Fire! Foes!), then
    the shortest.  Entries whose fingerprint cannot be taken are kept
    apart."""
    out, seen = {}, {}

    def _rank(chain):
        n_mv = sum(1 for a in chain if _leg_is_move(a))
        gap = (abs(n_mv - int(n_move_records))
               if n_move_records is not None else 0)
        return (gap, len(chain))
    for key, (chain, post) in uniq.items():
        from .casualty_gestures import equivalence_state
        fp = _semantic_state_key(equivalence_state(post, chain), paid_ring=paid_ring)
        if fp is None:
            out[key] = (chain, post)
            continue
        if execution_keys is not None:
            fp = (fp, execution_keys[key])
        prev = seen.get(fp)
        if prev is None:
            seen[fp] = key
            out[key] = (chain, post)
        elif _rank(chain) < _rank(out[prev][0]):
            out[prev] = (chain, post)
    return out



def _casualty_prefix_matches(events, records):
    from .casualty_gestures import prefix_matches
    return prefix_matches(events, records)


def _is_casualty_action(action):
    import emitter
    return emitter.casualty_action_params(action) is not None


def _casualty_budget_decreased(before, after, action, side):
    """Same-owner casualty microsteps have a strict, rules-sized bound."""
    import emitter
    if not _is_casualty_action(action):
        return False
    old, new = emitter.casualty_context(before), emitter.casualty_context(after)
    if old is None or new is None or old['side'] is not side or new['side'] is not side:
        return False
    if any(old[key] != new[key] for key in ('kind', 'stage', 'region', 'card_id')):
        return False
    return 0 < new['hits'] < old['hits']


def _pending_ctx_tag(post, nxt_kind):
    """The pending CONTEXT half of the exploration-memo key.  The
    combat menus do not name their battle (combat_card_none carries
    only a side), so two different battles look memo-identical without
    this: attacking Dale and attacking Old Forest Road both delta to
    {political +1, die -1} with continuation [combat_card_none] â€” the
    first probe's memo entry pruned the true attack's whole subtree
    (game 2023-01-25 T3, the g6 root)."""
    pc = getattr(post, 'pending_combat', None)
    import emitter
    return (nxt_kind, repr(emitter.casualty_context(post)),
            ((str(pc.get('attacker_region')),
              str(pc.get('defender_region')),
              str(pc.get('phase')),
              pc.get('rounds_fought'),
              str(pc.get('alloc_stage')),
              pc.get('atk_hits'), pc.get('def_hits'))
             if isinstance(pc, dict) else None),
            repr(getattr(post, 'fp_casualties', None)))


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
# window chain must stop at them â€” the pending-buffer phase owns their
# content.  Applies ONLY to die_to_hunt windows: a CARD window (The
# Nazgul Strike!) renders its whole hunt cascade IN-window and must
# traverse.  Depth-12 expansion sailing through reveal_place guessed
# an unrendered pawn placement and stranded the fellowship at the
# wrong region (game 2023-01-02 regression).
POST_WINDOW_KINDS = frozenset(('hunt', 'reveal_place', 'guide_change',
                               'discard', 'declare', 'eye'))


def _play_event_absence_reason(state, side, cids, die_face):
    """One explicit sentence per refused card play (USER 2026-08-25):
    which of the three engine facts refuses it."""
    parts = []
    try:
        hand = list(state.hands[side].hand)
    except Exception:
        hand = []
    for cid in list(cids)[:2]:
        name = str(cid).split('_', 2)[-1].replace('_', ' ')
        if cid not in hand:
            parts.append("%s is not in my copy of your hand (my hand "
                         "record has drifted - say resync)" % name)
            continue
        try:
            from event_effects import check_event_requirement
            ok, why = check_event_requirement(state, cid)
        except Exception:
            ok, why = True, ''
        if not ok:
            parts.append('%s cannot be played now: %s' % (name, why))
            continue
        if die_face:
            try:
                dice = state.dice.get(side, [])
                have = [i for i, d in enumerate(dice)
                        if not d.used and FACE_LETTER.get(
                            str(getattr(d.current, 'name', None)), '?')
                        in (die_face, 'W')]
            except Exception:
                have = [None]
            if not have:
                parts.append('%s: no unused %s die in my dice - your '
                             'roll may not have been adopted (say '
                             'resync)' % (name, die_face))
                continue
        parts.append('%s is not on my menu for this die' % name)
    return 'play_event refused - ' + '; '.join(parts) if parts else \
        'play_event %r not in the engine menu' % cids


def _guards_recruitment_facts(state, window, facts):
    """Validate Guards' pool/nation/box and normalize existing garrison seating.

    Board deltas alone make a permanently dead FP unit look like a
    reinforcement and collapse all ordinary Leaders to the same row.
    This card's printed placements require Gondor figures from the FP
    reinforcement pool; the Java client permits arbitrary free drags.
    Moving those figures through the region into an already besieged FP
    garrison is a physical placement step, not creation of a second siege.
    """
    cid = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
    pending = getattr(state, 'pending_card_option', None) or {}
    if pending.get('card_id') != cid and not any(
            title_to_cid(title) == cid for title in facts.cards_played):
        return None
    import combat
    import game_state as gs
    fp_garrison = ('minas tirith' in {
        sanitize(str(region)) for region in state.besieged_strongholds}
        and combat.garrison_side_for(state, 'minas tirith')
        is gs.Side.FREE_PEOPLES)
    seated = 0
    for rec in window.body_records():
        if rec.get('op') != 'move':
            continue
        target = zone_key(rec.get('to'))
        if target not in (('region', 'minas tirith'),
                          ('stronghold', 'minas tirith')):
            continue
        obj = str(rec.get('obj', ''))
        pieces = (rr.split_group(obj[len('group ('):-1])
                  if obj.startswith('group (') and obj.endswith(')')
                  else [obj])
        parsed = [rr.parse_piece(token) for token in pieces]
        source = zone_key(rec.get('from'))
        internal_seating = (fp_garrison and source == ('region', 'minas tirith')
                            and target == ('stronghold', 'minas tirith'))
        for piece in parsed:
            if piece is None or piece[0] not in ('unit', 'leader', 'nazgul'):
                continue
            if target == ('stronghold', 'minas tirith') and not fp_garrison:
                return ('This Guards recruitment belongs in the Minas Tirith region. '
                        'The Minas Tirith Stronghold box is reserved for an '
                        'already besieged Free Peoples garrison; it cannot '
                        'receive the Free Peoples besieging army. Undo that placement.')
            if str(piece[1]).upper() != 'GONDOR' \
                    or (source != ('reinf', 'FP') and not internal_seating):
                return ('Guards of the Citadel requires Gondor figures '
                        'from FP Reinforcements. Undo the incorrect '
                        'placement, then place the chosen Gondor unit '
                        'and Gondor Leader from that pool.')
        # Remove only the existing FP garrison's Gondor seating record.
        # A named enemy or another figure in the same group must retain
        # its distinct evidence, even if ordinary army deltas net to zero.
        if internal_seating and parsed and all(
                piece is not None and piece[0] in ('unit', 'leader')
                and str(piece[1]).upper() == 'GONDOR' for piece in parsed):
            seated += 1
    if seated:
        facts.sieges['minas tirith'] -= seated
        if not facts.sieges['minas tirith']:
            del facts.sieges['minas tirith']
    return None


def table_card_ids(state):
    """Public table identities, excluding hand identities and discard history."""
    pc = getattr(state, 'persistent_cards', None)
    return {c.card_id for attr in ('fp_cards', 'sh_cards')
            for c in (getattr(pc, attr, None) or ())}


def table_card_evidence_matches(before, after, chain, facts):
    """Match table departures to legal effects, not a hardcoded card list.

    A move to the action area expresses activation. A discard expresses
    removal, which may also be automatic. Declining a protection card and
    subsequently losing its required companion must not count as activation.
    """
    activated = set(facts.table_activations)
    discarded = set(facts.table_discards)
    wanted = activated | discarded
    if not wanted:
        return True
    removed = table_card_ids(before) - table_card_ids(after)
    if not wanted <= removed:
        return False
    direct = set()
    for action in chain:
        params = action.params or {}
        choices = params.get('choices') or {}
        if action.action_type == 'discard_table_card':
            direct.add(params.get('card_id'))
        elif action.action_type == 'card_option':
            # Identity fields do not make a declined option affirmative.
            intent = {k: v for k, v in choices.items()
                      if k not in ('card_id', 'card_name', 'side', 'decider')
                      and not k.startswith('_')}
            if intent and any(bool(v) for v in intent.values()) \
                    and intent.get('action') not in ('none', 'decline'):
                direct.add(choices.get('card_id') or params.get('card_id'))
    return activated <= direct


def _native_move_endpoint_pairs(window):
    """Typed net relocation endpoints of one native gesture transaction.

    A player may reposition pieces before finishing the decision. A waypoint
    drag is not a second rules action. Keep named identities and unit/leader
    types separate; never invent a route through one piece's unrelated move.
    This only widens the movement-option prefilter. Legal menus, cumulative
    coverage and recorded randomness still decide acceptance. Siege-box and
    non-board movements retain their own exact choreography.
    """
    flow = {}
    for record in window.body_records():
        if record.get('op') != 'move':
            continue
        source, target = zone_key(record.get('from')), zone_key(record.get('to'))
        if source[0] != 'region' or target[0] != 'region':
            continue
        obj = str(record.get('obj', '') or '')
        pieces = (rr.split_group(obj[len('group ('):-1])
                  if obj.startswith('group (') and obj.endswith(')') else [obj])
        for piece in pieces:
            parsed = rr.parse_piece(piece)
            if parsed is None or parsed[0] not in ('unit', 'leader', 'nazgul', 'named'):
                continue
            kind, first, second, count = parsed
            counts = flow.setdefault((kind, first, second), Counter())
            counts[source[1]] -= count
            counts[target[1]] += count
    return {(source, target) for counts in flow.values()
            for source, lost in counts.items() if lost < 0
            for target, gained in counts.items() if gained > 0 and source != target}


def _net_endpoint_movement(action):
    """A legal movement choice whose input is its source and destination.

    This includes card-driven flight and Army relocation phases. The engine
    still validates its real path and applies every capture/political effect;
    a temporary native drag is not an additional rules move. Battle actions
    and their recorded intermediate outcomes keep their existing history test.
    """
    if action.action_type == 'army_move':
        return True
    if action.action_type != 'card_option':
        return False
    choices = (action.params or {}).get('choices') or {}
    return bool(choices.get('from_region') and choices.get('to_region'))


def match_window(state, side, window, menu=None, budget=DEFAULT_BUDGET,
                 depth=DEFAULT_DEPTH, pin=None, rng=None,
                 relax_draws=False, stop_kinds=POST_WINDOW_KINDS,
                 fs_reset_relax=False, relax_nations=False,
                 advance_tolerance=False, fs_prefilter=False):
    """Match a closed gesture window against the engine menu.

    Returns MatchResult.  'hold' when the budget exhausts or the window
    holds unmapped records (never a refusal on matcher limitations â€”
    B13-1 doctrine: exhaustion HOLDS).  `pin` is an optional context
    manager factory (ClientRandomness.pin) so legality probes consume
    no live randomness.

    `rng` (spectator lane): a cursor controller separating the two
    randomness roles the single `pin` conflated â€”
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
                     them â€” per-leg begin_attempt made every leg re-read
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
        # BL-150 D1 (2026-08-28): a chain may traverse the in-window
        # pendings (card options, combat) but NEVER a hand-limit
        # discard - which card leaves the hand is the human's own
        # gesture.  Continuing through it silently discarded hand[0]
        # (Challenge of the King) inside an Eomer play.
        stop_kinds = ('discard',)
    facts = window_facts(window)
    # Preserve gross piece flow for a casualty-only window. Net board deltas
    # alone invent invisible reduce-then-kill histories for a whole-Elite
    # removal, sometimes consuming a different FP reinforcement pool.
    _casualty_flow = None
    _payment_flow = None
    _casualty_required_reductions = Counter()
    import emitter
    _ordinary_casualty = emitter.casualty_context(state) is not None
    from .casualty_gestures import live_context
    # Native downgrade intent also binds explicit recovery windows, whose
    # complete stable board may waive ambiguity but never an actual reduction.
    payment_effect = ((getattr(state, 'pending_combat', None) or {}).get('effect_choice') or {}).get('effect')
    _live_payment = (not _ordinary_casualty
                     and payment_effect in ('onslaught_payment', 'relentless_assault_payment',
                                            'pre_combat_casualty')
                     and live_context(state) is not None)
    if _ordinary_casualty or _live_payment:
        _casualty_flow = Counter()
        from .casualty_gestures import observed_unit_flow
        _casualty_flow_records, _casualty_required_reductions = observed_unit_flow(
            window.body_records(), side, undo_drafts=fs_prefilter or advance_tolerance)
        for _record_flow in _casualty_flow_records:
            _casualty_flow.update(_record_flow)
        if _live_payment:
            _payment_flow, _casualty_flow = _casualty_flow, None
    if facts.unmapped:
        return MatchResult('hold', note='unmapped records: %r'
                           % facts.unmapped[:3])
    guards_error = _guards_recruitment_facts(state, window, facts)
    if guards_error:
        result = MatchResult('none', note=guards_error)
        result.invalid_guards_recruit = True
        return result
    # ZERO-HIT BATTLE lane: a battle whose every roll missed moves no
    # piece â€” the window's ONLY observables are its roll records (and
    # the spent die), which express nothing in fact space.  Such a
    # window must still match the battle chain that consumes exactly
    # those rolls (empty board delta covers the empty facts; the
    # commit's unconsumed-randomness honesty gate polices the rolls).
    rolls_only = (facts.is_empty()
                  and (getattr(window, 'die_face', None) is not None
                       or (getattr(window, '_relentless_roll_closed', False)
                           and ((getattr(state, 'pending_combat', None) or {}).get(
                               'effect_choice') or {}).get('effect') == 'relentless_assault'))
                  and any(r.get('op') == 'roll_num'
                          for r in window.records))
    from .card_completion import empty_close
    if facts.is_empty() and not rolls_only and not empty_close(state, window):
        return MatchResult('none', note='window expressed nothing')
    if menu is None:
        with probe_ctx():
            menu = ai.get_legal_actions(state, side)
    inferred_target, ring_error = _inferred_ring_target(state, side, window, facts)
    if ring_error:
        return MatchResult('hold', note=ring_error)
    explicit_targets = getattr(window, '_explicit_ring_targets', None)
    if explicit_targets is not None:
        menu = [a for a in menu if a.action_type != 'elven_ring'
                or a.params.get('target_die_idx') in explicit_targets]
    if inferred_target is not None:
        # Only actual legal conversion faces may appear as an intermediate
        # unused-face delta. Zero keys widen _subset, never final _covers.
        menu = [a for a in menu if a.action_type == 'elven_ring'
                and a.params.get('target_die_idx') == inferred_target]
        from game_state import ActionDieFace
        for action in menu:
            face = ActionDieFace(action.params['new_face'])
            facts.dice_faces.setdefault((side.name, FACE_LETTER[face.name]), 0)
        window._inferred_ring_target = inferred_target
    if getattr(window, 'die_face', None):
        menu = [a for a in menu
                if _die_filter(state, side, window.die_face)(a)]
    def _rtx30(r):
        return str((r or {}).get('norm', '') if isinstance(r, dict)
                   else r).lower()
    _fs_legs30 = any(
        'fellowship track' in _rtx30(r)
        or 'hunt for the ring' in _rtx30(r)
        for r in (getattr(window, 'records', None) or ()))
    if facts.cards_played or facts.combat_cards or _fs_legs30 \
            or getattr(window, 'closed', None) is None:
        # BL-30c / BL-66 / BL-72: a window carrying a card play
        # (drag-shape in cards_played OR chooser-shape in combat_cards)
        # OR fellowship legs (the FSP track walk / die into the Hunt
        # Box) can never be a draw - probing draw_card would starve
        # the funnel (die unconsumed, free replays, phantom draw asks);
        # an OPEN (still-growing) window never probes draws either
        # (the chooser line lands a beat after the die stage).  NOT
        # keyed on facts.is_empty(): a draw window's own draw line is
        # a fact.
        menu = [a for a in menu
                if str(getattr(a, 'action_type', '')) != 'draw_card']
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

    def _combat_cards_ok(chain, post):
        """The client's chooser NAMED each combat card (and its side);
        a chain playing different combat cards â€” or any at all when
        the window shows none â€” is a different battle, even when the
        rolls made the outcome identical (Osgiliath specimen: four
        net-identical card combinations)."""
        if not table_card_evidence_matches(state, post, chain, facts):
            return False
        got = {}
        for x in chain:
            if x.action_type == 'combat_card_play':
                got.setdefault(str((x.params or {}).get('side')),
                               []).append(
                                   (x.params or {}).get('card_id'))
        return ({k: sorted(v) for k, v in got.items()}
                == {k: sorted(v) for k, v in cc_want.items()})

    def _vocab_prefilter(acts):
        """VOCABULARY lane: the client NAMED the played card â€” the
        only ACTION candidates are the play_event actions for exactly
        those cids, plus elven_ring when the window walked a ring
        marker (the ring flips the funding die BEFORE the play).
        Independently: a deck draw ALWAYS renders its 'draws ... Card.'
        line, so draw_card candidates in a window with no draw fact
        are never the gesture â€” probing them starves the card funnel
        and mis-classifies plain non-matches as rng-deferrals."""
        # The public draw paragraph names its owner and deck even though
        # the card identity is hidden. A different deck cannot explain this
        # gesture; probing it would ask the client to draw an unrelated card
        # when the observed candidate fails (for example two cards on one die).
        acts = [a for a in acts if _draw_deck_matches(a, facts.draws)]
        if not cids:
            if not facts.combat_cards:
                # BL-134 (the converse of the vocabulary lane): a card
                # play ALWAYS renders the card leaving the hand
                # (window_facts files every hand exit in
                # cards_played/combat_cards), so a window with NO hand
                # exit can never be a play_event - probing one starves
                # the DRAWS funnel on the same-deck redraw and a legal
                # fellowship_move would wait on a draw nobody owed
                acts = [a for a in acts
                        if a.action_type != 'play_event']
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
        # USER 2026-08-28 (explicit, never generic): say WHY the play
        # is not on the engine's menu - the card is not in the engine's
        # copy of the hand (identity drift -> resync), its own play
        # condition is unmet (the card text), or no unused die of the
        # staged face exists in the engine
        return MatchResult(
            'none', note=_play_event_absence_reason(
                state, side, cids, getattr(window, 'die_face', None)))
    counter = {'n': 0, 'starved': 0}
    # SOUND move-leg prefilter: the client renders EVERY board move
    # leg as its own 'moves ... from X to Y' record, so an engine
    # move-shaped leg (army_move, a card_option nazgul relocation)
    # whose (from, to) pair the window never rendered cannot be part
    # of this gesture â€” applying it only burns budget, and net-
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
    # LIVE decisions are one before/after board transaction, even if the human
    # dragged through a waypoint or reversed a draft placement. The spectator
    # and combat histories keep their existing literal evidence contract.
    endpoint_pairs = (_native_move_endpoint_pairs(window)
                      if fs_prefilter or advance_tolerance else set())
    # BL-99: regions the window rendered movement both INTO and OUT
    # OF - only these get the ranks' waypoint waiver in _subset
    _bidir99 = ({f for f, _t in win_pairs}
                & {t for _f, t in win_pairs})

    # a FELLOWSHIP gesture always renders fellowship vocabulary (the
    # FSP track walk, the pawn, a flip/reveal, a guide change, a
    # corruption walk); a window with NONE of it can never be one -
    # probing a fellowship_move candidate anyway starves the HUNT
    # funnel, defers an unrelated window and orders REAL randomness
    # for a non-gesture (and the client lines pushed after the window
    # make the refusal's undo INELIGIBLE, fact 2).  Structural
    # rendering-contract prefilter - never a rules guess.
    _fs_vocab = bool(facts.fs_progress or facts.fel_flips
                     or facts.fel_moves or facts.corruption
                     or facts.guide_set or facts.comps_gone)

    def _region_prefilter(acts):
        out = []
        for a in acts:
            if _casualty_flow is not None and not _is_casualty_action(a):
                continue
            p = a.params or {}
            from .card_completion import choice_needs_boundary
            ch_staged = p.get('choices') or {}
            if (fs_prefilter and choice_needs_boundary(state, ch_staged)
                    and getattr(window, 'closed', None) == 'pending'
                    and not getattr(window, '_staged_card_closed', False)):
                continue
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
            if fs_prefilter and a.action_type in (
                    'fellowship_declare', 'fellowship_reveal_place'):
                # Applying a placement can request a real stronghold/Balrog
                # tile before _covered sees the resulting state. Reject
                # unvoiced destinations and unfinished resets BEFORE probing,
                # or a trial Moria route can order a tile for a Goblin's Gate
                # gesture whose reveal flip has not arrived yet.
                destination = sanitize(str(p.get('region', ''))).strip().lower()
                if a.action_type == 'fellowship_reveal_place' and facts.fel_hidden is True:
                    continue
                if facts.fel_endpoint:
                    if facts.fel_endpoint != destination:
                        continue
                else:
                    import game_logic as _gl_place
                    if destination != _gl_place.fellowship_region(state):
                        continue
                reset = -state.fellowship.progress
                if facts.fs_progress != reset and not (
                        fs_reset_relax and facts.fel_moves
                        and reset <= facts.fs_progress <= 0):
                    continue
            if fs_prefilter and facts.sieges \
                    and a.action_type == 'combat_no_retreat':
                # a window that DRAGGED into the stronghold is never
                # the field-battle decision (probing it starves the
                # combat-roll funnel: rng ordered for a non-gesture)
                continue
            if a.action_type == 'army_attack':
                # a battle ALWAYS marks its regions (casualties,
                # advances, siege retreats); an attack in regions the
                # window never touched is not this gesture â€” probing
                # it starves the combat-roll funnel and turns plain
                # non-matches into false rng-deferrals (measured on
                # every T5 fellowship-move window of game 2023-01-13).
                # EXCEPT the zero-hit lane: an all-miss battle marks
                # NO region â€” the rolls themselves pick the battle.
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
                # Only useful stationary descriptors exist in the legal menu.
                # Their actual activation delta must still cover the voiced
                # marker parity; an empty die window cannot use this path.
                stationary_activation = (frk == tok and any(facts.activations.values()))
                endpoint_leg = (_net_endpoint_movement(a)
                                and (frk, tok) in endpoint_pairs)
                if ((frk, tok) not in win_pairs and not endpoint_leg
                        and not stationary_activation):
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
        [move-half, move-half] and [move-whole, pass] â€” different
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
                    # and [move-escort, option-move-WK] â€” two legs on
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
                frk, tok = sanitize(str(fr)).strip().lower(), sanitize(str(to)).strip().lower()
                if frk != tok:
                    use[(frk, tok)] += 1
            if all(v <= win_pair_counts.get(k, 0)
                   for k, v in use.items()):
                kept.append((chain, post))
        return kept

    def _close_multistep(post, chain):
        """A window closed = the die action is COMPLETE (the client
        killed the die).  Close any engine multi-step half (second
        army move / second recruit) with its explicit 'pass' so the
        committed state carries no phantom open half-action."""
        pending = getattr(post, 'pending_card_option', None) or {}
        # A killed Character die is the player's explicit end of an ordinary
        # Nazgul/Minion movement action. The internal per-figure menu must not
        # require a second chat-only "done". Card-seeded Nazgul moves deliberately
        # lack Mouth eligibility and may still owe a Hunt/reveal decision.
        normal_nazgul = (pending.get('card_id') == '__nazgul_moves__'
                        and any((option.get('_naz_arr') or {}).get('__mo_ok__')
                                for option in pending.get('options', ())
                                if isinstance(option, dict)))
        # Ordinary companion groups use the same paid-die completion signal.
        # Apply the menu's finish action rather than clearing pending metadata;
        # any distinct rule follow-up remains owned by the engine.
        normal_companions = (pending.get('card_id') == '__companion_moves__')
        finish_key = ('_naz_done' if normal_nazgul else
                      '_comp_done' if normal_companions else None)
        if getattr(window, 'closed', None) == 'die_killed' and finish_key:
            with probe_ctx():
                finish = [a for a in ai.get_legal_actions(post, side)
                          if a.action_type == 'card_option'
                          and (a.params.get('choices') or {}).get(finish_key)]
            if len(finish) == 1:
                with apply_ctx():
                    post = ai.apply_action(post, finish[0])
                chain = chain + finish
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

    seen = _ContinuationMemo()
    # Preserve ordered paths when native combat records supply checkpoints.
    ordered_history = rng is not None and any(
        window.records[i].get('op') == 'roll_num'
        and window.records[i - 1].get('op') != 'roll_num'
        for i in range(1, len(window.records)))

    def _try(st, acts, chain, d, casualty_flow=None, casualty_events=(),
             casualty_reductions=None, combat_draw_start=None):
        hits = []
        casualty_flow = casualty_flow or Counter()
        casualty_reductions = casualty_reductions or Counter()
        at_root = not chain
        # The AI has one compact siege-press policy row, but a human may pay
        # with any participating Elite. Enumerate these physical realizations
        # only in the gesture matcher; keep the policy/action encoding intact.
        physical_acts = []
        for action in acts:
            if action.action_type == 'siege_press' and not (action.params or {}).get('nation'):
                import game_state as _gs_press
                pc = getattr(st, 'pending_combat', None) or {}
                nations = dict.fromkeys(u.nation for u in ai._alloc_side_units(
                    st, pc.get('attacker_region', ''), pc.get('attacker_side'),
                    _gs_press.UnitKind.ELITE))
                for nation in nations:
                    params = dict(action.params or {})
                    params['nation'] = nation.value
                    physical_acts.append(ai.Action(action.die_idx, action.action_type, params))
            else:
                physical_acts.append(action)
        for a in physical_acts:
            if (advance_tolerance and a.action_type == 'hunt_gollum_reveal'
                    and not facts.fel_flips):
                # Gollum's reveal is a voluntary damage choice. A companion's
                # earlier separation cannot authorize it merely because hidden
                # marker follow-through is normally engine-owned.
                continue
            if advance_tolerance:
                from .mighty_gestures import permits_choice as permits_mighty_choice
                if not permits_mighty_choice(st, side, window, a):
                    continue
            from .relentless_gestures import permits_payment
            if not permits_payment(state, st, a, facts, window):
                continue
            if inferred_target is not None and chain:
                if a.action_type == 'elven_ring':
                    continue
                if a.die_idx is not None and a.die_idx >= 0 \
                        and a.die_idx != inferred_target:
                    # The engine menu keeps one representative per face. A
                    # late-index M->C payment can therefore expose only an
                    # earlier C die's action row. Bind that legal fungible row
                    # to the converted physical die before applying it; never
                    # widen across faces or consume the representative die.
                    dice = st.dice.get(side, [])
                    if not (0 <= a.die_idx < len(dice)
                            and not dice[a.die_idx].used
                            and not dice[inferred_target].used
                            and dice[a.die_idx].current is dice[inferred_target].current):
                        continue
                    a = ai.Action(inferred_target, a.action_type, dict(a.params or {}))
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
            from .combat_draw_gestures import draw_cursor
            if advance_tolerance and at_root:
                # Coverage compares the full transaction against its root
                # state, including a draw settled by its final casualty leg.
                # Keep the same cursor through every descendant and sibling.
                combat_draw_start = draw_cursor(rng)
            try:
                try:
                    with apply_ctx():
                        from .onslaught_gestures import is_payment_action
                        from .casualty_randomness import is_round_unit_payment
                        if advance_tolerance and (a.action_type == 'casualty_allocate'
                                                  or is_payment_action(a)
                                                  or is_round_unit_payment(st, a)):
                            from .casualty_randomness import apply_observed_casualty
                            post = apply_observed_casualty(st, a)
                        else:
                            post = ai.apply_action(st, a)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    # NeedClientCall (a BaseException): the candidate
                    # starved the randomness feed beyond what has
                    # arrived â€” counted so the caller can distinguish
                    # 'randomness not here yet' (retry later) from a
                    # genuine non-match
                    if type(e).__name__ == 'NeedClientCall':
                        if advance_tolerance:
                            stopped = getattr(e, '_casualty_pre_request_state', None)
                            if (a.action_type == 'casualty_allocate'
                                    and _casualty_flow is not None and stopped is not None):
                                from .casualty_gestures import event_flow, awaiting_replacement
                                stopped_events = list(casualty_events)
                                for event in getattr(stopped, '_casualty_operations', ()) or ():
                                    if event.get('side') is side:
                                        stopped_events.extend(event_flow(event, _unit_type_key, sanitize))
                                if awaiting_replacement(Counter(stopped_events), _casualty_flow,
                                                        stopped_events, _casualty_flow_records,
                                                        _casualty_prefix_matches):
                                    counter['partial_elite_payment'] = True
                            from .casualty_randomness import completed_payment_key
                            paid_key = completed_payment_key(state, side, window, chain + [a], e)
                            from .casualty_gestures import native_reductions_covered, payment_reductions
                            performed = casualty_reductions.copy()
                            if _payment_flow is not None:
                                performed.update(payment_reductions(st, a, side))
                            elif stopped is not None:
                                import game_state as _gs
                                for event in getattr(stopped, '_casualty_operations', ()) or ():
                                    if event.get('side') is side and event.get('operation') == 'reduce_elite':
                                        performed[(sanitize(str(event['region'])).lower(),
                                                   _unit_type_key(event['nation'], _gs.UnitKind.ELITE))] += 1
                            if not native_reductions_covered(_casualty_required_reductions, performed):
                                paid_key = None
                            if paid_key is not None:
                                stopped = getattr(e, '_casualty_pre_request_state', None)
                                from .casualty_gestures import equivalence_state
                                demand_key = _typed_value_key((e.kind, e.kw))
                                # Equivalent, completed casualty paths may
                                # enumerate different numbers of legal menus.
                                # That probe count is not a consumed outcome.
                                # Keep the actual RNG/feed/journal strict and
                                # resume the chosen representative's own path.
                                cursor_key = _continuation_execution(rng, include_probe_index=False)
                                counter.setdefault('proven_starves', []).append(
                                    ((paid_key, demand_key, cursor_key,
                                      _semantic_state_key(equivalence_state(stopped, chain + [a]))
                                      if stopped is not None else None),
                                     e, chain + [a], stopped))
                        counter['starved'] += 1
                        # keep the starve KINDS visible even though
                        # backtracking restores rand._pending (the
                        # deferral-expiry row was logging pending=None)
                        counter.setdefault('starve_kinds', set()).add(
                            '%s:%s' % (getattr(e, 'kind', '?'),
                                       (getattr(e, 'kw', {}) or {})
                                       .get('ctx', '')))
                        # the starving CALL OBJECT survives too -
                        # backtracking restores rand._pending to
                        # None, so the deferral's rng ask would
                        # otherwise have NOTHING to order
                        counter['starve_call'] = e
                        # BL-134: EVERY root starve keyed by its
                        # candidate - the deferral must ask for the
                        # randomness of the candidate that CAN be the
                        # gesture, never whichever starved last
                        if at_root:
                            counter.setdefault('starve_calls',
                                               []).append((a, e))
                    continue
                next_casualty_flow = casualty_flow
                next_casualty_events = list(casualty_events)
                next_casualty_reductions = casualty_reductions.copy()
                if _payment_flow is not None:
                    from .casualty_gestures import payment_reductions
                    next_casualty_reductions.update(payment_reductions(st, a, side))
                withheld_casualty_rows = Counter()
                if (_casualty_flow is not None and _is_casualty_action(a)
                        and all(_is_casualty_action(x) for x in chain)):
                    next_casualty_flow = casualty_flow.copy()
                    import game_state as _gs
                    _events = getattr(post, '_casualty_operations', None) or []
                    if advance_tolerance:
                        from .casualty_gestures import observed_operations
                        _events, withheld_casualty_rows = observed_operations(
                            st, post, a, side, casualty_flow, casualty_events,
                            _casualty_flow, _casualty_flow_records,
                            _unit_type_key, sanitize, _casualty_prefix_matches)
                    for _event in _events:
                        if _event.get('side') is not side:
                            continue
                        _region = sanitize(str(_event['region'])).lower()
                        _nation, _operation = _event['nation'], _event['operation']
                        _kind = (_gs.UnitKind.REGULAR if _operation == 'remove_regular'
                                 else _gs.UnitKind.ELITE)
                        _loss_key = ((_region, _unit_type_key(_nation, _kind)), -1)
                        next_casualty_flow[_loss_key] += 1
                        next_casualty_events.append(_loss_key)
                        if _operation == 'reduce_elite':
                            next_casualty_reductions[_loss_key[0]] += 1
                        if _operation == 'reduce_elite' and _event['replacement_source'] is not None:
                            _key = (_region, _unit_type_key(_nation, _gs.UnitKind.REGULAR))
                            next_casualty_flow[(_key, 1)] += 1
                            next_casualty_events.append((_key, 1))
                    if (any(v > _casualty_flow[k] for k, v in next_casualty_flow.items())
                            or not _casualty_prefix_matches(next_casualty_events,
                                                           _casualty_flow_records)):
                        if advance_tolerance:
                            from .casualty_gestures import awaiting_replacement
                            if awaiting_replacement(next_casualty_flow, _casualty_flow,
                                                    next_casualty_events,
                                                    _casualty_flow_records,
                                                    _casualty_prefix_matches):
                                counter['partial_elite_payment'] = True
                        continue
                # CUMULATIVE delta from the WINDOW's root state â€” a
                # chain's coverage is everything it did, not its last
                # leg (the per-level delta could never cover a two-leg
                # army move: leg 2's own delta misses leg 1's rows;
                # found on the split-move specimen, game 2023-01-02
                # T1).  Coverage is judged on the PASS-CLOSED
                # candidate: an open half-action has not consumed its
                # die yet, and the window's die-kill says the die IS
                # spent (dice-row lane).
                eng = delta_facts(state, post)
                if (_payment_flow is not None and at_root
                        and (a.params or {}).get('operation') == 'downgrade'):
                    # This candidate was already applied by the public engine.
                    # Its missing replacement proves the physical window can
                    # still grow; do not release another candidate's roll yet.
                    from .casualty_gestures import awaiting_replacement
                    payment_events = []
                    for sign in (-1, 1):
                        for key, value in eng.armies.items():
                            if (value * sign > 0
                                    and _shadow_tkey(key[1]) == (side.name == 'SHADOW')):
                                payment_events.extend([(key, sign)] * abs(value))
                    if awaiting_replacement(Counter(payment_events), _payment_flow,
                                            payment_events, _casualty_flow_records,
                                            _casualty_prefix_matches):
                        counter['partial_elite_payment'] = True
                post2, chain2 = _close_multistep(post, chain + [a])
                eng2 = (eng if post2 is post
                        else delta_facts(state, post2))
                if eng2.guide is None:
                    # FLD-4 (par5b_g5#15): the engine seats a
                    # TEMPORARY guide (change_guide allow_pending:
                    # best[0]) while the tie pending is open -
                    # picking exactly that candidate is a NO-OP
                    # delta, so the row's own guide-box drag matched
                    # none of the options.  The choose_guide leg's
                    # identity IS the engine fact.
                    for _xg in chain2:
                        if getattr(_xg, 'action_type', '') \
                                == 'choose_guide':
                            eng2.guide = (str((_xg.params or {})
                                              .get('companion') or '')
                                          or None)
                            break
                flux = any(getattr(x, 'action_type', '')
                           in _FLUX_TYPES for x in chain + [a])
                # a chain with the defender's combat_retreat carries
                # the engine's AUTO-ADVANCE of the OPPONENT's army -
                # rows this gesture can never voice IN LIVE SIDED MODE
                # (the AI renders no gesture; the post-commit sweep
                # emits them).  LIVE-ONLY, bounded relocation
                # tolerance (_advance_extra_ok): in spectator mode
                # BOTH seats' records render and coverage stays exact
                # (a broad tolerance lets retreat candidates cover
                # foreign windows).
                _adv = None
                # the defender's retreat_into_siege carries the SAME
                # engine-forced opponent relocation (the besieger
                # moves INTO the region; the garrison is in the
                # stronghold sub-box) - the post-commit sweep emits it
                if advance_tolerance and any(
                        x.action_type in ('combat_retreat',
                                          'retreat_into_siege')
                        for x in chain2):
                    _adv = ('shadow'
                            if side.name == 'FREE_PEOPLES' else 'fp')
                from .casualty_gestures import native_reductions_covered
                _flow_complete = (
                    native_reductions_covered(_casualty_required_reductions, next_casualty_reductions)
                    and (_casualty_flow is None
                         or any(not _is_casualty_action(x) for x in chain2)
                         or next_casualty_flow == _casualty_flow))
                from .relentless_gestures import payment_coverage
                coverage = (payment_coverage(state, post2, chain2, eng2)
                            if advance_tolerance else eng2)
                if advance_tolerance:
                    from .onslaught_gestures import payment_coverage as onslaught_coverage
                    coverage, onslaught_paid = onslaught_coverage(
                        state, post2, chain2, coverage, side)
                    # A proved combat payment can force figure removal and
                    # capture, just as an ordinary casualty allocation can.
                    flux = flux or onslaught_paid
                if advance_tolerance and len(chain2) == 1:
                    from .mighty_gestures import payment_coverage as mighty_coverage
                    coverage = mighty_coverage(state, side, window, chain2[0], coverage)
                    from .dead_men_gestures import forced_shadow_coverage
                    coverage = forced_shadow_coverage(state, post2, side, chain2, coverage, facts)
                if advance_tolerance:
                    from .combat_draw_gestures import consumed_draw_coverage
                    coverage = consumed_draw_coverage(state, post2, chain2, coverage,
                                                      facts, rng, combat_draw_start)
                if withheld_casualty_rows:
                    from .casualty_gestures import physical_coverage
                    coverage = physical_coverage(coverage, withheld_casualty_rows)
                if _flow_complete and _covers(facts, coverage, relax_draws=relax_draws,
                           fs_reset_relax=fs_reset_relax,
                           combat_flux=flux,
                           relax_nations=relax_nations,
                           auto_advance_side=_adv,
                           follow_through=follow_through_for(
                               chain2, advance_tolerance),
                           forced_army_rows=(
                               _forced_defender_allocation_rows(state, side)
                               if advance_tolerance and chain2
                               and all(_is_casualty_action(x) for x in chain2)
                               else None)):
                    if _combat_cards_ok(chain2, post2) and (inferred_target is None
                            or (len(chain2) > 1 and post2.dice[side][inferred_target].used)):
                        from .casualty_gestures import unfinished_elite_removal
                        if (advance_tolerance
                                and unfinished_elite_removal(state, post2, chain2, side)):
                            counter['partial_elite_payment'] = True
                            continue
                        # Equal boards cannot erase different consumption of
                        # the client's recorded dice, tiles or cards.
                        execution = ('no client rng',)
                        if rng is not None:
                            try:
                                execution = rng.outcome_key()
                            except (AttributeError, TypeError, ValueError):
                                execution = ('unverified client rng', object())
                        hits.append((chain2, post2,
                                     _facts_fingerprint(eng2), execution))
                    continue
                if withheld_casualty_rows:
                    # The observed prefix must completely cover this window.
                    # Do not use withheld physical costs to search later choices.
                    continue
                _casualty_leg = _casualty_budget_decreased(st, post, a, side)
                # A replacement may die in the next explicit card-casualty
                # step, so its intermediate +Regular can exceed the final
                # net delta. Gross flow above already bounds every operation;
                # relax only this casualty region's intermediate unit rows.
                _casualty_flux_regions = (
                    {sanitize(str(emitter.casualty_context(st)['region'])).lower()}
                    if _casualty_leg else set())
                if (d > 0 or _casualty_leg) and _subset(eng, facts, combat_flux=flux,
                                     relax_nations=relax_nations,
                                     flux_regions=set(_bidir99) | _casualty_flux_regions):
                    with probe_ctx():
                        nxt = ai.describe_pending(post)
                    if nxt is not None and nxt['kind'] in stop_kinds:
                        # post-window pending: its records have not
                        # rendered yet â€” the chain stops here, the
                        # pending buffer resolves it
                        acts2 = None
                    elif nxt is not None \
                            and nxt.get('side') is not side:
                        # B-W2-05 (ii) (COORD #2 2026-09-03, USER bridge
                        # go): a pending the OPPONENT decides is the
                        # opponent's decision - the AI takes it through
                        # its own ai_decide item, never from the
                        # human's window (a human drag of the AI
                        # attacker's stack must not compose the AI's
                        # victor_advance onto the garrison's retreat).
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
                        # Identical visible deltas/menus are not proof of an
                        # identical future. Preserve full rule/RNG state and
                        # evidence that later coverage/history filters inspect.
                        prefix = chain + [a]
                        history = _continuation_history(prefix, ordered_history)
                        evidence = (history, tuple(next_casualty_events),
                                    dict(next_casualty_flow), dict(next_casualty_reductions),
                                    combat_draw_start)
                        bucket = (_facts_fingerprint(eng),
                                  _pending_ctx_tag(post, nxt['kind'] if nxt else None),
                                  tuple(sorted(str((x.action_type, x.die_idx, x.params))
                                               for x in acts2)))
                        if seen.dominated(bucket, post, acts2, rng, evidence, d):
                            continue
                        # FORCED legs (a one-action pending: the
                        # combat-card 'none' beats, auto steps) spend
                        # no depth â€” depth bounds BRANCHING, and a
                        # 3-round siege is long but narrow (Grond
                        # specimen: 16 legs, ~7 of them choices)
                        d2 = d if len(acts2) == 1 or _casualty_leg else d - 1
                        sub = _try(post, acts2, chain + [a], d2,
                                   next_casualty_flow, next_casualty_events,
                                   next_casualty_reductions, combat_draw_start)
                        if sub == 'BUDGET':
                            return 'BUDGET'
                        hits.extend(sub)
            finally:
                if snap is not None:
                    rng.restore(snap)
        return hits

    def _pick_starve134(pairs, fallback):
        """BL-134: the deferral's ask is the starving call of the
        candidate CONSISTENT with the window's vocabulary - a
        fellowship-family candidate when the window walked the FSP /
        flipped / took corruption; a non-draw demand when no card
        left the hand (only card plays and deck draws starve on
        draws).  Last-starve-wins was the wedge."""
        if not pairs:
            return fallback
        _FS = ('fellowship_move', 'fellowship_hide',
               'fellowship_declare', 'fellowship_reveal_place')
        if _fs_vocab:
            fs = [e for a, e in pairs
                  if getattr(a, 'action_type', '') in _FS]
            if fs:
                return fs[0]
        if not cids and not facts.combat_cards:
            nd = [e for _a, e in pairs
                  if getattr(e, 'kind', '') not in ('draws', 'card',
                                                    'card_pick')]
            if nd:
                return nd[0]
        return pairs[0][1]

    res = _try(state, _region_prefilter(menu), [], depth)
    if res == 'BUDGET':
        return MatchResult('hold', applies=counter['n'],
                           note='apply budget exhausted')
    if ((res or counter.get('proven_starves')) and counter.get('partial_elite_payment')
            and getattr(window, '_hold_ambiguous_elite_prefix', False)
            and not getattr(window, '_casualty_closed', False)):
        # A complete whole-Elite loss can also be the first native line of
        # a legal reduction. A live feed poll is not a gesture boundary:
        # keep the original state until replacements arrive or the player
        # explicitly completes the physical allocation.
        result = MatchResult('hold', applies=counter['n'],
                             note='finish Elite replacements, or say done to confirm whole-Elite losses')
        result.waiting_casualty_completion = True
        result.ambiguous_elite_prefix = True
        return result
    if not res:
        if counter.get('partial_elite_payment') and not counter.get('proven_starves'):
            result = MatchResult('hold', applies=counter['n'],
                                 note='awaiting Elite replacement or remaining casualty payment')
            result.waiting_casualty_completion = True
            return result
        if counter['starved']:
            _mr = MatchResult(
                'hold', applies=counter['n'], starved=True,
                note='rng-starved (%d candidates await client '
                     'randomness: %s)'
                     % (counter['starved'],
                        ','.join(sorted(counter.get('starve_kinds',
                                                    ())))))
            _mr.starve_call = _pick_starve134(
                counter.get('starve_calls'), counter.get('starve_call'))
            proven = counter.get('proven_starves') or []
            if proven:
                first = proven[0]
                if all(row[0] == first[0] for row in proven[1:]):
                    _mr.proven_starve_key = first[0]
                    _mr.proven_starve_call = first[1]
                    _mr.proven_starve_chain = first[2]
                    _mr.proven_starve_state = first[3]
                else:
                    _mr.proven_starve_conflict = True
                    _mr.note = 'completed casualty gestures imply conflicting continuation randomness; held for review'
            return _mr
        return MatchResult('none', applies=counter['n'])
    # No action-type/visible-delta deduplication may precede this comparison:
    # equal-looking boards can retain different abilities or pending decisions.
    candidates = {i: (chain, post) for i, (chain, post, _fp, _exec) in enumerate(res)}
    if len(candidates) > 1 and rng is not None:
        # The intermediate battle record is evidence about which path happened.
        # Apply that evidence before choosing a representative outcome.
        kept = _record_order_filter(state, side, window, list(candidates.values()), rng)
        keep_ids = {id(chain) for chain, _post in kept}
        candidates = {i: cp for i, cp in candidates.items() if id(cp[0]) in keep_ids}
    endpoints = [{'chain': chain, 'post': post, 'execution': res[i][3]}
                 for i, (chain, post) in candidates.items()]
    uniq = _collapse_same_state(
        candidates, sum(win_pair_counts.values()),
        paid_ring=(side, inferred_target) if inferred_target is not None else None,
        execution_keys={i: row[3] for i, row in enumerate(res)})
    completed_live_endpoint = fs_prefilter or advance_tolerance
    if len(uniq) > 1 and not completed_live_endpoint:
        kept = _pair_count_filter(list(uniq.values()))
        if kept and len(kept) < len(uniq):
            uniq = {i: cp for i, cp in enumerate(kept)}
    if len(uniq) > 1 and not completed_live_endpoint:
        # USER manual test 2026-08-25 (Fear! Fire! Foes!): two chains
        # both COVER the window only because the follow-through table
        # tolerates engine-extra rows; the human voiced everything
        # they did, so the chain whose engine delta equals the gesture
        # EXACTLY (no un-voiced rows) is the gesture - a chain that
        # needs tolerance to fit is not a rival.  Never a guess: only a
        # UNIQUE exact-parity chain wins; otherwise still ambiguous.
        exact = [(c, p) for c, p in uniq.values()
                 if _unvoiced_rows(facts, delta_facts(state, p)) == 0]
        if len(exact) == 1:
            chain, post = exact[0]
            result = MatchResult('match', chain=chain, post=post,
                               applies=counter['n'],
                               candidates=endpoints,
                               note='exact-parity chain preferred over '
                                    '%d tolerance-fitted rival(s)'
                                    % (len(uniq) - 1))
            if inferred_target is not None:
                result._inferred_ring_target = inferred_target
            return result
    if len(uniq) > 1:
        # Completed native decisions are identified by their resulting board,
        # not by an internal decomposition into move legs. Preserve all legal
        # endpoints for the live owner's fresh-board comparison. Historical
        # corpus callers retain their record-count/exact-parity policy above.
        result = MatchResult('ambiguous', applies=counter['n'],
                             chain=[c for c, _ in uniq.values()],
                             candidates=endpoints,
                             note='%d distinct actions produce the '
                                  'window delta' % len(uniq))
        if inferred_target is not None:
            result._inferred_ring_target = inferred_target
        return result
    chain, post = next(iter(uniq.values()))
    result = MatchResult('match', chain=chain, post=post,
                         applies=counter['n'], candidates=endpoints)
    if inferred_target is not None:
        result._inferred_ring_target = inferred_target
    return result


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
    if any(got.activations.get(nation, False) != parity
           for nation, parity in want.activations.items()):
        out.append('political activation')
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
        bad = want.fel_hidden is not None and got.fel_hidden != want.fel_hidden
        if all(k == 'flip' for _r, k in want.fel_flips):
            bad = got.hidden_change != ((len(want.fel_flips) % 2) == 1)
        if bad:
            out.append('the Fellowship marker')
    if want.fel_endpoint and got.fel_region != want.fel_endpoint:
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
        if not _draw_deck_matches(a, want.draws):
            continue
        # save()/restore() is the matcher's OWN bracket (see _try)
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


def _subset(eng, facts, combat_flux=False, relax_nations=False,
            flux_regions=()):
    """May this chain CONTINUE â€” is the engine delta so far a
    non-contradicting PART of the window?

    BOARD rows (armies/leaders/named) use the WAYPOINT principle: a
    multi-leg gesture passes through intermediate states whose row
    values overshoot the window's NET (Ringwraiths specimen: a nazgul
    stops in Moria then rides on â€” mid-chain Moria +1, net 0), so a
    row may be in ANY flux while its region is one the window spoke
    of; a nonzero row in a region the window NEVER mentioned is a
    contradiction.  Dice rows likewise flux through faces the window
    touched (ring flip +P then play -P).  Scalar/one-way quantities
    (draws, political, rings, corruption) keep directional bounds.
    An EMPTY delta is allowed: an action that only OPENS a pending
    (play_event -> card_option) expresses nothing yet; expansion is
    bounded to open pendings/multi-step halves plus depth+budget.

    combat_flux (chain contains a combat leg): ARMY rows switch from
    the directional movement bound to the waypoint rule too â€” a
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
        # BL-99: the WAYPOINT rule extends to the RANKS for regions the
        # window rendered movement BOTH into and out of (flux_regions)
        # - the two-army die's vacated-source shape nets to zero there
        # (p.22 allows two armies per die).  One-way regions keep the
        # tight bound (the Grond overshoot guard).  Sound: this only
        # widens EXPLORATION - final acceptance still demands the
        # exact cumulative cover.
        if k[0] in flux_regions:
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
