"""Card option enumeration — makes optioned event cards LEARNED decisions.

USER FINDING (2026-07-07): optioned cards always took the handler's
hardcoded default in self-play (Shadows on the Misty Mountains mustered
in Mount Gram — the card text's first-listed region — though humans
muster in Moria ~90% of the time).  Root cause: handlers read
`choices.get(key, <default>)` and self-play never filled `choices`.

Fix: each optioned card declares its option space here.  Playing such a
card defers the effect into `state.pending_card_option`; get_legal_actions
returns one `card_option` action per legal choices-dict, the net/MCTS
picks in self-play, and in extraction the human's logged choices become
the labeled pick (record-backed provenance).

Every spec returns a list of COMPLETE choices dicts (possibly empty →
the play falls back to direct handler execution).  All region names are
validated against board.regions before being offered — a name that is
not on this board can never enter an option (Anorien lesson).

Multi-select cards are ALL specced (2026-07-27 audit closing the stale
TODO(card-options-multi)): Hill-trolls / Many Kings / Pits of Mordor
enumerate bounded combos (amendments 1.117/1.131); the nazgul_moves
half of The Black Captain Commands rides the __nazgul_moves__ chain
(1.128).  Orc Patrol / Foul Thing victims are RNG DRAWS by rule
(almanac p.43, adversarial repair 2026-07-11) — deliberately NOT
learned picks.  Card-inflicted CASUALTY allocation routes through the
shared owner decision (event_effects._route_card_casualties,
USER 2026-07-27).
"""
from typing import Callable, Dict, List

import board
from board import sanitize
from game_state import Companion, Nation, Side

CARD_OPTION_SPECS: Dict[str, Callable] = {}

# P1 (USER ruling 1.112.3): cards whose deferred choice belongs to the
# OPPONENT of the card's owner (cross-side decisions)
CARD_OPTION_DECIDER: Dict[str, str] = {
    'lure of the ring': 'opponent',
}


def _spec(name):
    def deco(fn):
        CARD_OPTION_SPECS[name.lower()] = fn
        return fn
    return deco


def _valid_region(name):
    """Region name -> sanitized key iff it exists on THIS board."""
    san = sanitize(str(name))
    return san if san in board.regions else None


def _regions(names):
    out = []
    for n in names:
        san = _valid_region(n)
        if san and san not in out:
            out.append(san)
    return out


def _army_at(state, san):
    for k, a in state.armies.items():
        if sanitize(k) == san:
            return a
    return None


def _is_fp(a):
    # units[0] semantics kept: switching these to membership reshaped
    # log COLLECTION and broke canary Adobim (2026-07-11); the mixed-
    # entry fix lives inline in _free_for_side instead
    from event_effects import SHADOW_NATIONS as _SN
    return bool(a.units) and a.units[0].nation not in _SN


def _is_sh(a):
    from event_effects import SHADOW_NATIONS as _SN
    return bool(a.units) and a.units[0].nation in _SN






def _companion_regions(state):
    out = []
    for k, a in state.armies.items():
        if any(isinstance(l, Companion) for l in a.leaders):
            out.append(sanitize(k))
    return out


def _separable(state):
    return [c for c in state.fellowship.companions
            if c.name != "Gollum" and c.level > 0]


def _move_dests(state, cur, rng, allow_besieged=False):
    """Card-move destinations for a previously separated companion,
    anchored at ITS region (QA 2026-07-11 Gwaihir CASE 8)."""
    from ai import _companion_reachable
    out = set(_companion_reachable(state, cur, max(0, rng)))
    out.discard(cur)
    if allow_besieged:
        from event_effects import _dest_ok_from
        for b in getattr(state, 'besieged_strongholds', set()):
            bs = sanitize(b)
            if bs != cur and bs not in out and _dest_ok_from(
                    state, cur, bs, max(0, rng), allow_besieged=True):
                out.add(bs)
    return sorted(out)


def _sep_destinations(state, movement, allow_besieged=False):
    """Destinations for a card separation with the given movement range.

    QA 2026-07-11 clusters B/C: on the Mordor Track the four separation
    cards may still separate - the companions are REMOVED FROM PLAY
    (marker destination); with allow_besieged (Gwaihir / We Prove
    Swifter) friendly besieged strongholds in range are offered too."""
    import game_logic
    from ai import _companion_reachable
    if state.fellowship.mordor_entered:
        return ['__remove_from_play__']
    cur = game_logic.fellowship_region(state)
    if cur in getattr(state, 'besieged_strongholds', set()):
        return [cur]
    reach = _companion_reachable(state, cur, max(0, movement))
    reach.add(cur)
    if allow_besieged:
        from event_effects import _dest_ok_from
        for b in getattr(state, 'besieged_strongholds', set()):
            bs = sanitize(b)
            if bs not in reach and _dest_ok_from(
                    state, cur, bs, max(0, movement),
                    allow_besieged=True):
                reach.add(bs)
    return sorted(reach)


# Official coastal regions (rulebook map), filtered through THIS board.
# QA 2026-07-09: Lorien is landlocked and was actually offered (the
# "dropped by filter" comment was wrong - _regions() only drops names
# that are not on the board, and Lorien is on the board).
# USER RULING 2026-07-21 (Cirdan's Ships, authoritative coastal list;
# +dol amroth per the same-day correction — supersedes the old
# 10-entry list which wrongly included near harad and missed eight
# regions):
_COASTAL = ("pelargir", "dol amroth", "west harondor", "umbar",
            "lamedon", "anfalas", "andrast", "druwaith iaur",
            "minhiriath", "enedwaith", "south ered luin",
            "grey havens", "forlindon", "harlindon",
            "north ered luin", "evendim", "arnor")


def _free_for_side(state, san, side):
    """Region eligible for a card muster by `side` (USER RULES
    2026-07-08): no enemy field army; enemy-CAPTURED settlements are
    eligible only when my army is there (recruiting while BESIEGING —
    e.g. Thranduil's Archers into the army retaking Woodland Realm);
    my own besieged settlements stay eligible (recruit while
    besieged)."""
    a = _army_at(state, san)
    my_units = enemy_units = False
    if a and a.units:
        # membership tests (QA 2026-07-11 Half-orcs C6): a shared
        # besieged entry holds BOTH sides; units[0] order hid the
        # besieger side and no-opped legal recruits
        from event_effects import SHADOW_NATIONS as _SN
        _has_sh = any(u.nation in _SN for u in a.units)
        _has_fp = any(u.nation not in _SN for u in a.units)
        if side == Side.FREE_PEOPLES:
            my_units, enemy_units = _has_fp, _has_sh
        else:
            my_units, enemy_units = _has_sh, _has_fp
    besieged = san in getattr(state, 'besieged_strongholds', set())
    if enemy_units and not (besieged and my_units):
        return False
    ec = getattr(state, 'enemy_controlled', {}) or {}
    holder = None
    for sd, regs in ec.items():
        if san in regs:
            holder = sd
    if holder is not None and holder != side and not my_units:
        return False
    return True


# --------------------------------------------------------------------------
# Free Peoples
# --------------------------------------------------------------------------

@_spec("Shadows on the Misty Mountains")
def _(state, side):
    return [{"region": r} for r in _regions(["Mount Gram", "Moria"])
            if _free_for_side(state, r, Side.SHADOW)]


@_spec("Wisdom of Elrond")
def _(state, side):
    opts = []
    for nat, nm in ((Nation.ROHAN, 'rohan'), (Nation.GONDOR, 'gondor'),
                    (Nation.THE_NORTH, 'north'), (Nation.DWARVES, 'dwarves'),
                    (Nation.ELVES, 'elves')):
        if state.political.boxes.get(nat, 0) < 3:
            opts.append({"nation": nm})
    return opts


@_spec("Dead Men of Dunharrow")
def _(state, side):
    # USER RULING 2026-07-21: the REGION is the learned decision; the
    # recruit COUNT is AUTO always-3 (pool-capped by recruit_unit) to
    # keep the net's decision space small. The handler still consumes
    # choices['recruit_count'] (default 3) so the TEACHER can force the
    # logged count for board fidelity (phantom-regulars fix, 1.202).
    return [{"target_region": r}
            for r in _regions(["Erech", "Lamedon", "Pelargir"])]


@_spec("Riders of Théoden")
def _(state, side):
    # 1.211c (2026-07-21, MAIN-SESSION CORRECTED INTERPRETATION): card
    # text "or any Rohan region CONTAINING a Companion" is a PER-REGION
    # gate, NOT a Rohan-wide unlock. Legal = Edoras (if free) UNION every
    # Rohan-parent region that contains a BOARD Companion, where Helm's
    # Deep qualifies EVEN UNDER SIEGE (companion inside) but every other
    # region must be free for the Free Peoples. Fellowship companions do
    # not exist on the board (doctrine 1.211l) - _companion_regions reads
    # only board army leaders.
    _hd = sanitize("Helm's Deep")
    _rohan_comp = [
        r for r in _companion_regions(state)
        if board.fid_to_name.get(
            board.regions.get(r, {}).get('parent_id')) == 'rohan'
        and (r == _hd or _free_for_side(state, r, Side.FREE_PEOPLES))]
    seen, opts = set(), []
    for r in _regions(["Edoras"]):
        if _free_for_side(state, r, Side.FREE_PEOPLES):
            seen.add(r)
            opts.append({"region": r})
    for r in _rohan_comp:
        if r not in seen:
            seen.add(r)
            opts.append({"region": r})
    return opts


@_spec("Éomer son of Éomund")
def _(state, side):
    rohan = [k for k, m in board.regions.items()
             if board.fid_to_name.get(m.get('parent_id')) == 'rohan'
             and (m.get('is_town') or m.get('is_city')
                  or m.get('is_stronghold'))]
    bs = getattr(state, 'besieged_strongholds', set())
    return [{"region": r} for r in sorted(rohan)
            if _free_for_side(state, r, Side.FREE_PEOPLES) and r not in bs]


@_spec("House of the Stewards")
def _(state, side):
    opts = []
    for k, a in state.armies.items():
        if any(isinstance(l, Companion) and l.name == "Boromir"
               for l in a.leaders):
            opts.append({"boromir_region": sanitize(k)})
    return opts


@_spec("The Grey Company")
def _(state, side):
    # QA 2026-07-11 cluster I: crowned Aragorn counts; only FP-side
    # regulars; only nations with an Elite physically available
    from game_state import UnitKind
    from combat import side_of_nation as _son
    from ai import _can_recruit
    _names9 = ("Strider", "Aragorn", "Aragorn - Heir to Isildur")
    opts = []
    for k, a in state.armies.items():
        if not any(isinstance(l, Companion) and l.name in _names9
                   for l in a.leaders):
            continue
        for nat in {u.nation for u in a.units
                    if u.kind == UnitKind.REGULAR
                    and _son(u.nation) == Side.FREE_PEOPLES}:
            if _can_recruit(state, nat, 'elite'):
                opts.append({"strider_region": sanitize(k),
                             "nation": nat.value.lower()})
    return opts


@_spec("Cirdan's Ships")
def _(state, side):
    # USER RULING 2026-07-21g: "a coastal region containing a Free
    # Peoples Army" INCLUDES a besieged FP stronghold, whose shared
    # army entry also holds the Shadow besiegers. _fp_army_regions ->
    # _is_fp tests units[0] (kept globally for log-collection parity,
    # canary Adobim) and drops the region when a besieger sorts first.
    # Test FP PRESENCE (any FP unit) directly here.
    from event_effects import SHADOW_NATIONS as _SN
    out = []
    for r in _regions(_COASTAL):
        a = _army_at(state, r)
        if a and any(u.nation not in _SN for u in a.units):
            out.append({"region": r})
    return out


@_spec("The Spirit of Mordor")
def _(state, side):
    # QA 2026-07-09 (almanac A5): only SHADOW Army units count - FP
    # bystanders in a shared siege entry must not disqualify the target
    from event_effects import SHADOW_NATIONS
    opts = []
    for k, a in state.armies.items():
        sh_nations = {u.nation for u in a.units
                      if u.nation in SHADOW_NATIONS}
        if len(sh_nations) >= 2:
            opts.append({"target_region": sanitize(k)})
    return opts


@_spec("Faramir's Rangers")
def _(state, side):
    opts = []
    for r in _regions(["Osgiliath", "South Ithilien", "North Ithilien"]):
        a = _army_at(state, r)
        if a and a.units and _is_sh(a):
            opts.append({"target_region": r})
    return opts


@_spec("Help Unlooked For")
def _(state, side):
    # QA 2026-07-09 (almanac A10): the card attacks a Shadow army
    # BESIEGING a stronghold, with an FP army in an adjacent region
    # that contains at least one At-War unit; the old spec had the
    # roles inverted and ignored both the siege and the At-War gate.
    from combat import garrison_side_for
    from event_effects import SHADOW_NATIONS
    opts = []
    for rsan in sorted(getattr(state, 'besieged_strongholds', set())):
        if garrison_side_for(state, rsan) != Side.FREE_PEOPLES:
            continue
        entry = _army_at(state, rsan)
        if not entry or not any(u.nation in SHADOW_NATIONS
                                for u in entry.units):
            continue
        for adj in state.board.adjacent(rsan):
            adj_san = sanitize(adj)
            aa = _army_at(state, adj_san)
            if aa and any(u.nation not in SHADOW_NATIONS
                          and state.political.is_at_war(u.nation)
                          for u in aa.units):
                opts.append({"stronghold": rsan, "attack_from": adj_san})
    return opts


@_spec("Paths of the Woses")
def _(state, side):
    # QA 2026-07-09: direct Rohan-to-Minas-Tirith move (almanac A11);
    # the legal choices live in event_effects next to the handler
    from event_effects import _paths_of_woses_options
    return _paths_of_woses_options(state)


@_spec("Through a Day and a Night")
def _(state, side):
    # QA 2026-07-09: gate = FP army CONTAINING A COMPANION (any nation,
    # not Rohan), 2 free regions max via unit-free intermediates
    # (almanac A12); the legal choices live next to the handler
    from event_effects import _tdaan_options
    return _tdaan_options(state)


def _separation_card(extra, group):
    """Shared spec for the companion-separation cards."""
    def fn(state, side):
        opts = []
        comps = _separable(state)
        if not comps:
            return opts
        if group:
            # one option per (single companion, dest); the historical
            # group separations ride the closed-loop choices directly
            for c in comps:
                for dest in _sep_destinations(
                        state, state.fellowship.progress + c.level + extra):
                    opts.append({"companions": [c.name],
                                 "destination": dest})
        return opts
    return fn


CARD_OPTION_SPECS["i will go alone"] = _separation_card(1, True)
CARD_OPTION_SPECS["there and back again"] = _separation_card(1, True)


def _ents_spec(state, side):
    # single empty option: the auto-pick carries '_from_pending' so the
    # free-chain (cluster D) opens on the learned path; teacher plays
    # supply logged choices and never reach the auto-pick branch
    return [{}]


CARD_OPTION_SPECS["the ents awake: treebeard"] = _ents_spec
CARD_OPTION_SPECS["the ents awake: huorns"] = _ents_spec
CARD_OPTION_SPECS["the ents awake: entmoot"] = _ents_spec


@_spec("The Eagles are Coming")
def _(state, side):
    # QA 2026-07-11 cluster E: target = a Shadow army holding regular
    # Nazgul that is co-located with or adjacent to an FP army
    # containing a Companion (requirement); the FP player picks WHICH
    from game_state import Nazgul as _NazE
    opts = []

    def _fp_comp(a):
        from event_effects import SHADOW_NATIONS as _SN
        return (any(u.nation not in _SN for u in a.units)
                and any(isinstance(l, Companion) for l in a.leaders))

    for k, a in state.armies.items():
        if not any(isinstance(l, _NazE) for l in a.leaders):
            continue
        rsan = sanitize(k)
        near = [rsan] + [sanitize(x) for x in state.board.adjacent(rsan)]
        ok = False
        for nb in near:
            aa = _army_at(state, nb)
            if aa is not None and _fp_comp(aa):
                ok = True
                break
        if ok:
            opts.append({"target_region": rsan})
    return opts


@_spec("Gwaihir the Windlord")
def _(state, side):
    from itertools import combinations as _comb
    opts = []
    comps = _separable(state)
    for c in comps:
        for dest in _sep_destinations(state,
                                      state.fellowship.progress + 4,
                                      allow_besieged=True):
            opts.append({"companion_action": f"sep_{c.name}",
                         "destination": dest})
    # 1.149c: GROUP separations (subsets size>=2 fly together as Level 4)
    _gdests = _sep_destinations(state, state.fellowship.progress + 4,
                                allow_besieged=True)
    for _r in range(2, len(comps) + 1):
        for _sub in _comb(comps, _r):
            _names = [c.name for c in _sub]
            for dest in _gdests:
                opts.append({"companion_action": "group",
                             "companions": _names, "destination": dest})
    for k, a in state.armies.items():
        for l in a.leaders:
            if isinstance(l, Companion):
                for dest in _move_dests(state, sanitize(k), 4,
                                        allow_besieged=True):
                    opts.append({"companion_action": f"move_{l.name}",
                                 "destination": dest})
    return opts


@_spec("We Prove Swifter")
def _(state, side):
    from itertools import combinations as _comb
    opts = []
    comps = _separable(state)
    for c in comps:
        for dest in _sep_destinations(
                state, state.fellowship.progress + c.level + 2,
                allow_besieged=True):
            opts.append({"companion_action": f"sep_{c.name}",
                         "destination": dest})
    # 1.149c: GROUP separations (subsets size>=2; range = group max
    # level + 2, independent of Fellowship members)
    for _r in range(2, len(comps) + 1):
        for _sub in _comb(comps, _r):
            _names = [c.name for c in _sub]
            _gmax = max(c.level for c in _sub)
            for dest in _sep_destinations(
                    state, state.fellowship.progress + _gmax + 2,
                    allow_besieged=True):
                opts.append({"companion_action": "group",
                             "companions": _names, "destination": dest})
    for k, a in state.armies.items():
        for l in a.leaders:
            if isinstance(l, Companion):
                for dest in _move_dests(state, sanitize(k), l.level + 2,
                                        allow_besieged=True):
                    opts.append({"companion_action": f"move_{l.name}",
                                 "destination": dest})
    return opts


# --------------------------------------------------------------------------
# Shadow
# --------------------------------------------------------------------------

@_spec("Cruel Weather")
def _(state, side):
    import game_logic
    cur = game_logic.fellowship_region(state)
    # Batch-003 audit catch (2026-07-29): the fellowship PAWN never
    # enters Mordor's interior (1.155 track model; ruling 2026-07-20b
    # covered declares/reveals — this card path was the gap: Shadow
    # dragged the pawn from Minas Morgul into Gorgoroth).
    return [{"target_region": sanitize(r)}
            for r in state.board.adjacent(cur)
            if sanitize(r) not in game_logic._MORDOR_INTERIOR]


@_spec("Dreadful Spells")
def _(state, side):
    # QA 2026-07-11 C2: target_region must hold the NAZGUL (dice
    # source); the FP army is in or adjacent to it
    import game_logic as _gl
    opts = []
    for k, a in state.armies.items():
        if not any(_gl.is_nazgul_figure(l) for l in a.leaders):
            continue
        nz = sanitize(k)
        for r in [nz] + [sanitize(x) for x in state.board.adjacent(nz)]:
            fa = _army_at(state, r)
            # FP PRESENCE membership, never units[0]: a besieged FP
            # garrison shares its entry with the Shadow besiegers
            from event_effects import SHADOW_NATIONS as _SN_DS
            if fa and any(u.nation not in _SN_DS for u in fa.units):
                opts.append({"target_region": nz, "target_fp_army": r})
    return opts


# The Ringwraiths Are Abroad: spec RETIRED (amendment 1.131) - the old
# single-destination gather-all option collapsed every human split; the
# handler now seeds the per-figure '__nazgul_moves__' chain directly.


@_spec("The Fighting Uruk-hai")
def _(state, side):
    # card text: the Isengard army must be BESIEGING a stronghold; the
    # attack targets THAT stronghold (co-located siege assault) — the
    # old spec offered adjacent field attacks (USER 2026-07-09)
    from combat import garrison_side_for
    opts = []
    bs = getattr(state, 'besieged_strongholds', set()) or set()
    for k, a in state.armies.items():
        frm = sanitize(k)
        if frm not in bs or not a.units:
            continue
        # the Isengard army must be the BESIEGER (FP garrison inside),
        # never the besieged garrison itself (Grond-spec guard)
        if garrison_side_for(state, frm) != Side.FREE_PEOPLES:
            continue
        if any(u.nation == Nation.ISENGARD for u in a.units):
            opts.append({"attack_from": frm, "attack_target": frm})
    return opts


@_spec("Grond Hammer of the Underworld")
def _(state, side):
    # co-located sieges where the SHADOW side besieges (garrison-first
    # unit order must not hide the besieger - QA 2026-07-11 C9)
    from combat import garrison_side_for
    from event_effects import _army_contains_witch_king
    opts = []
    bs = getattr(state, 'besieged_strongholds', set())
    for k, a in state.armies.items():
        frm = sanitize(k)
        if frm not in bs:
            continue
        if garrison_side_for(state, frm) != Side.FREE_PEOPLES:
            continue
        # FIX (game 16): require the Witch-king among the besiegers, not
        # merely a Sauron unit - the old proxy skipped the real WK siege
        # at Dol Amroth and false-matched Erebor (one stray Sauron
        # regular), fabricating a phantom siege.  Grond is his weapon.
        if not _army_contains_witch_king(a):
            continue
        opts.append({"attack_from": frm, "attack_target": frm})
    return opts


@_spec("Corsairs of Umbar")
def _(state, side):
    # QA 2026-07-10 (almanac A10): targets are GONDOR coastal regions;
    # FP-occupied targets are legal (the move starts the battle); no
    # move while a Shadow army is besieged inside Umbar
    from combat import garrison_side_for
    from event_effects import SHADOW_NATIONS, _shadow_units_at
    umb = 'umbar'
    if umb in getattr(state, 'besieged_strongholds', set()) \
            and garrison_side_for(state, umb) == Side.SHADOW:
        return []
    if not _shadow_units_at(state, umb):
        return []
    _gondor_coastal = ('anfalas', 'lamedon', 'pelargir', 'dol amroth')
    out = []
    for r in _gondor_coastal:
        if r not in board.regions:
            continue
        if r in getattr(state, 'besieged_strongholds', set()):
            # 1.210(b) example (c): a Shadow-garrison coastal stronghold
            # under FP siege is a legal RELIEF landing (the corsairs land
            # into the besieged Shadow stronghold and a siege battle forms
            # where the no-cease rule does NOT apply - the Umbar-relief
            # case).  Other besieged (FP-garrison) coastal targets stay
            # excluded.
            if garrison_side_for(state, r) != Side.SHADOW:
                continue
        out.append({"target_region": r})
    return out


@_spec("Rage of the Dunlendings")
def _(state, side):
    # QA 2026-07-10 (almanac): free regions adjacent to N/S Dunland
    # (Orthanc excluded - impassable border) + Moria + N/S Dunland;
    # unplayable with no Isengard Regular in reinforcements
    from ai import _can_recruit
    from event_effects import _rage_regions
    from game_state import Nation as _N
    if not _can_recruit(state, _N.ISENGARD, 'regular'):
        return []
    return [{"recruit_region": r} for r in _rage_regions(state)]


def _shadow_army_regions(state):
    from event_effects import SHADOW_NATIONS
    return sorted({sanitize(k) for k, a in state.armies.items()
                   if any(u.nation in SHADOW_NATIONS for u in a.units)})


@_spec("Half-orcs and Goblin-men")
def _(state, side):
    # QA 2026-07-10: card text = "a region where a Shadow Army is
    # present"; QA 2026-07-11 C5: never offered with an empty pool
    from ai import _can_recruit
    from game_state import Nation as _N
    if not (_can_recruit(state, _N.ISENGARD, 'regular')
            or _can_recruit(state, _N.ISENGARD, 'elite')):
        return []
    return [{"region": r} for r in _shadow_army_regions(state)
            if _free_for_side(state, r, Side.SHADOW)]


@_spec("Olog-hai")
def _(state, side):
    # QA 2026-07-10: card text = "a region where a Shadow Army is
    # present"; pool guard added with Half-orcs (QA 2026-07-11)
    from ai import _can_recruit
    from game_state import Nation as _N
    if not (_can_recruit(state, _N.SAURON, 'regular')
            or _can_recruit(state, _N.SAURON, 'elite')):
        return []
    return [{"region": r} for r in _shadow_army_regions(state)
            if _free_for_side(state, r, Side.SHADOW)]


@_spec("The Black Captain Commands")
def _(state, side):
    # USER CORRECTION 2026-07-08: BCC does NOT muster the Witch-king
    # (my earlier spec invented that).  Card: "recruit two Nazgul in the
    # region containing the Witch-king, OR move any or all of the
    # Nazgul.  Then, you may move or attack with an Army containing the
    # Witch-king."  Card req = WK in play.  The recruit region is the
    # WK's OWN region (no choice); the modal choice is recruit-vs-move.
    opts = []
    wk_region = None
    for k, a in state.armies.items():
        if any('witchking' in __import__('re').sub(
                r'[^a-z]', '', str(getattr(l, 'name', '')).lower())
               for l in a.leaders):
            wk_region = sanitize(k)
            break
    if wk_region is not None:
        # never offer a guaranteed no-op (QA 2026-07-11 C5): the
        # recruit arm needs Nazgul figures left in the pool
        from ai import _can_recruit
        if _can_recruit(state, Nation.SAURON, 'leader'):
            opts.append({"action": "recruit",
                         "witch_king_region": wk_region})
    opts.append({"action": "nazgul_moves"})
    return opts


def _companion_move_card(state, side):
    """Book of Mazarbul / Fear! Fire! Foes! (P3, USER ruling 1.112.4):
    'Move any or all Companions' - GROUP options (single source of
    truth in event_effects._companion_group_move_options); the chain
    for further groups + the war trigger live in the handlers.  The
    terminator = play with no moves (the 'Then...' trigger still
    fires)."""
    from event_effects import _companion_group_move_options
    opts = _companion_group_move_options(state)
    if opts:
        opts.append({'_ffm_done': True})
    return opts


@_spec("There is Another Way")
def _(state, side):
    # USER RE-RULING 2026-08-24 (supersedes 2026-07-21f): the heal
    # ALWAYS applies; with Gollum as Guide the FP may ALSO move the
    # Fellowship on the same card (no die enters the Hunt Box).  The
    # learned decision is move vs heal-only, and it exists only when
    # a hidden move is possible; no Gollum, or revealed -> [] (the
    # handler heals directly, nothing to decide).
    g = state.fellowship.guide
    if g is None or 'Gollum' not in getattr(g, 'name', ''):
        return []
    if not state.fellowship.hidden:
        return []
    return [{"action": "move"}, {"action": "heal_only"}]


CARD_OPTION_SPECS["book of mazarbul"] = _companion_move_card
CARD_OPTION_SPECS["fear! fire! foes!"] = _companion_move_card


@_spec("Shadows Gather")
def _(state, side):
    from event_effects import _sg_join_options
    return _sg_join_options(state, 3)


@_spec("The Shadow Lengthens")
def _(state, side):
    # P1 (amendment 1.117, TSiM pattern): each of the (up to two) card
    # moves is an INDEPENDENT chained pending on the learned path - the
    # spec offers single joins and the handler chains a direct pending
    # for the second move.  The old n_armies=2 pair enumeration (which
    # surfaced from_region_2/to_region_2 the action encoder cannot see,
    # and re-picked-up units delivered by the first move on a general
    # chain) is removed.
    from event_effects import _sg_join_options
    return _sg_join_options(state, 2)


@_spec("The Shadow is Moving")
def _(state, side):
    # single 1-region moves into free or friendly regions; moves 2-4
    # are INDEPENDENT chained pendings (P1, amendment 1.117) - the
    # old at-most-two-moves limitation note was stale.
    from combat import side_of_nation as _son
    from event_effects import SHADOW_NATIONS
    bs = getattr(state, 'besieged_strongholds', set())
    singles = []
    for k, a in state.armies.items():
        frm = sanitize(k)
        if frm in bs:
            continue
        if not any(u.nation in SHADOW_NATIONS for u in a.units):
            continue
        for adj in state.board.adjacent(frm):
            to = sanitize(adj)
            if to in bs:
                continue
            a_to = _army_at(state, to)
            if a_to and any(_son(u.nation) != Side.SHADOW
                            for u in a_to.units):
                continue
            singles.append({"from_region": frm, "to_region": to})
    # P1 (USER ruling 1.112.7): each of the card's four moves is an
    # INDEPENDENT learned decision - the spec offers single moves and
    # the handler chains a direct pending for moves 2-4
    return singles


@_spec("Many Kings to the Service of Mordor")
def _(state, side):
    from itertools import combinations
    from event_effects import _free_se_settlements
    free = _free_se_settlements(state)
    if not free:
        return []
    k = min(3, len(free))
    return [{"settlements": list(c)} for c in combinations(free, k)]


@_spec("Pits of Mordor")
def _(state, side):
    from itertools import combinations
    from event_effects import _free_sauron_strongholds
    free = _free_sauron_strongholds(state)
    if not free:
        return []
    k = min(3, len(free))
    return [{"strongholds": list(c)} for c in combinations(free, k)]


@_spec("Horde from the East")
def _(state, side):
    # almanac: East Rhun / South Rhun / Khand / Far Harad only
    from event_effects import _region_free_for_shadow
    return [{"region": r}
            for r in ("east rhun", "south rhun", "khand", "far harad")
            if r in board.regions and _region_free_for_shadow(state, r)]


@_spec("Hill-trolls")
def _(state, side):
    # The card REPLACES TWO Sauron Regulars (card text); each option is a
    # COMPLETE choice - single-army [r, r] (two upgrades in one army) or
    # two-army [r1, r2] (one unit in each of two separate armies, almanac
    # A15).  An Event card may not be partially played, so a 1-upgrade
    # option is offered ONLY as the max-extent fallback when two upgrades
    # are physically impossible: fewer than two upgradeable Sauron
    # Regulars on the board, or fewer than two Sauron Elites available in
    # reinforcements.  (QA 2026-07-11: the old spec always emitted partial
    # singles - letting self-play upgrade one when it must upgrade two -
    # and also offered a two-in-one-army pair the elite pool could not
    # support.)  Only Regular SAURON units count; the membership tests
    # keep FP / non-Sauron units in a shared besieged entry off the list.
    from ai import _can_recruit, figures_available as _figs_avail9
    from game_state import Nation as _N, UnitKind as _UK
    if not _can_recruit(state, _N.SAURON, 'elite'):
        return []
    reg_counts = {}
    for k, a in state.armies.items():
        n = sum(1 for u in a.units
                if u.nation == _N.SAURON and u.kind == _UK.REGULAR)
        if n:
            san = sanitize(k)
            reg_counts[san] = reg_counts.get(san, 0) + n
    if not reg_counts:
        return []
    regs = sorted(reg_counts)
    total_regs = sum(reg_counts.values())
    # ONE ANSWERER (USER ruling 2026-08-18, strict component limits) -
    # this was a seventh inline copy of the cap arithmetic.
    avail_elite = _figs_avail9(state, _N.SAURON, 'elite')
    max_up = min(2, total_regs, max(0, avail_elite))
    if max_up <= 0:
        return []
    if max_up == 1:
        # only ONE upgrade is physically achievable (one Regular on the
        # board, or one Elite in the pool) -> single-region options; the
        # pool cannot support any two-upgrade combo, so none are offered
        return [{"regions": [r]} for r in regs]
    # two upgrades achievable -> COMPLETE two-upgrade combos only
    opts = []
    for i, r1 in enumerate(regs):
        if reg_counts[r1] >= 2:
            opts.append({"regions": [r1, r1]})       # two in one army
        for r2 in regs[i + 1:]:
            opts.append({"regions": [r1, r2]})       # one in each of two
    return opts


@_spec("Stormcrow")
def _(state, side):
    # QA 2026-07-10 (almanac M6): only not-At-War FP nations whose
    # borders hold the Ring-bearers or a separated Companion
    from event_effects import _stormcrow_valid_nations
    from game_state import Nation as _N
    _tag = {_N.THE_NORTH: 'north'}
    return [{"nation": _tag.get(n, n.value.lower())}
            for n in sorted(_stormcrow_valid_nations(state),
                            key=lambda x: x.value)]


# Nazgûl Strike!: spec RETIRED (amendment 1.131) - the gather-all +
# bundled either/or collapsed the per-figure movement; the handler now
# seeds the '__nazgul_moves__' chain and the either/or fires at chain
# end as its own learned pending (see _nazgul_card_post).


@_spec("Lure of the Ring")
def _(state, side):
    # the FREE PEOPLES player chooses (CARD_OPTION_DECIDER routes the
    # pending to FP); Gollum-guide plays skip the choice entirely
    g = state.fellowship.guide
    if g is not None and 'Gollum' in getattr(g, 'name', ''):
        return []
    if not state.fellowship.companions:
        return []
    return [{"action": "corruption"}, {"action": "eliminate"}]


# Nazgûl Search: spec RETIRED (amendment 1.131) - the all-or-nothing
# teleport collapsed the per-figure movement; the handler now seeds
# the '__nazgul_moves__' chain with the reveal check deferred to the
# chain's end ('_naz_post': 'search').


@_spec("The Lidless Eye")
def _(state, side):
    from game_state import ActionDieFace as _F
    free = sum(1 for d in state.dice.get(Side.SHADOW, [])
               if not d.used and d.current != _F.EYE)
    return [{"dice_count": n} for n in range(1, min(3, free) + 1)]


def get_card_options(state, side, card_name):
    """All legal choices dicts for the card, [] if unspecced/no options."""
    fn = CARD_OPTION_SPECS.get(str(card_name).lower())
    if fn is None:
        return []
    try:
        opts = fn(state, side)
    except Exception as _spec_err9:
        # LOUD: a crashed spec silently degrading to the handler's
        # hardcoded default is the exact regression class this module
        # exists to prevent
        import traceback
        print('CARD_OPTION SPEC CRASHED for %r: %r'
              % (card_name, _spec_err9))
        traceback.print_exc()
        return []
    # de-duplicate while preserving order
    seen, out = set(), []
    for o in opts:
        key = tuple(sorted((k, str(v)) for k, v in o.items()))
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out


def option_region(choices):
    """Primary region of an option (for spatial action encoding)."""
    for key in ("region", "target_region", "to_region", "destination",
                "recruit_region", "stronghold", "witch_king_region",
                "nazgul_destination", "boromir_region", "strider_region",
                "attack_target", "from_region", "attack_from",
                "eagles_dest"):
        v = choices.get(key)
        if v:
            return str(v)
    # Multi-region recruitment specs carry a LIST of target regions
    # (Pits of Mordor: 'strongholds').  Expose the FIRST element so the
    # trio is spatially VISIBLE to the GNN action scorer instead of
    # collapsing to region index -1 for EVERY trio (the SoMM
    # Mount-Gram-vs-Moria invisibility class documented at ai.py:4113,
    # where uniform priors made the option unlearnable).
    # sentinel: _pits_strongholds_optreg  (v16: also 'regions' for
    # Hill-trolls and 'settlements' for Many Kings - same
    # spatial-invisibility class)
    v = (choices.get("strongholds") or choices.get("regions")
         or choices.get("settlements"))
    if isinstance(v, (list, tuple)) and v:
        return str(v[0])
    return None
