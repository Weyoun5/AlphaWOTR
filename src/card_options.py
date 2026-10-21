"""Card option enumeration — makes optioned event cards LEARNED decisions
(USER 2026-07-07).

Each optioned card declares its option space here.  Playing such a
card defers the effect into `state.pending_card_option`; get_legal_actions
returns one `card_option` action per legal choices-dict, the net/MCTS
picks in self-play, and in extraction the human's logged choices become
the labeled pick (record-backed provenance).

Every spec returns a list of COMPLETE choices dicts (possibly empty →
the play falls back to direct handler execution).  All region names are
validated against board.regions before being offered — a name that is
not on this board can never enter an option (Anorien lesson).

Implemented multi-select families include Hill-trolls / Many Kings /
Pits of Mordor, which enumerate bounded combos (amendments 1.117/1.131);
this registry is not a completeness claim for every printed choice.
Most recruitment-card Regular/Elite choices remain absent;
Dead Men exposes its companion subsets. Guards of the Citadel, House of the
Stewards and Faramir's Rangers expose their unit kinds separately. The nazgul_moves
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

# 1.112.3: cards whose deferred choice belongs to the card owner's OPPONENT
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
    from ai import _companion_reachable, _friendly_besieged_strongholds
    if state.fellowship.mordor_entered:
        return ['__remove_from_play__']
    cur = game_logic.fellowship_region(state)
    if cur in _friendly_besieged_strongholds(state):
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


# Official coastal regions (USER RULING 2026-07-21, authoritative Cirdan's
# Ships list incl. Dol Amroth), filtered through THIS board.
_COASTAL =("pelargir", "dol amroth", "west harondor", "umbar",
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
        # membership tests (QA C6): a shared besieged entry holds BOTH sides
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
def dead_men_options(state, side):
    # Choose the destination first. The printed "may THEN recruit up to
    # three" is a separate FP decision after damage/retreat/movement;
    # event_effects._dead_men_recruit opens that later count menu.
    from itertools import combinations
    required_names = {'Strider', 'Aragorn', 'Aragorn - Heir to Isildur'}
    for army in state.armies.values():
        companions = [leader.name for leader in army.leaders if isinstance(leader, Companion)]
        required = [name for name in companions if name in required_names]
        if not required:
            continue
        if len(required) != 1 or len(set(companions)) != len(companions):
            raise ValueError('Dead Men requires one identifiable Strider/Aragorn group')
        optional = sorted(name for name in companions if name not in required_names)
        groups = [[required[0], *group] for count in range(len(optional) + 1)
                  for group in combinations(optional, count)]
        return [{'target_region': region, 'companions': group}
                for region in _regions(['Erech', 'Lamedon', 'Pelargir']) for group in groups]
    return []


@_spec("Riders of Théoden")
def _(state, side):
    # 1.211c: "or any Rohan region CONTAINING a Companion" is a PER-REGION gate:
    # Edoras (if free) UNION every Rohan region with a BOARD Companion (Helm's
    # Deep even under siege; any other must be free for the FP).  Fellowship
    # companions are not on the board (1.211l).
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
def house_of_the_stewards_options(state, side):
    """Character23 / Almanac16: one supplied kind with board Boromir.

    If recruitment is blocked or both pools are empty, the compulsory
    Strategy draws still resolve. There is no optional decline of an
    available recruit, and recruitment may precede an overstack decision.
    """
    if side != Side.FREE_PEOPLES:
        return []
    from ai import figures_available
    from event_effects import _get_region_nation
    for k, a in state.armies.items():
        if any(isinstance(l, Companion) and l.name == "Boromir"
               for l in a.leaders):
            region = sanitize(k)
            if _get_region_nation(region) != 'gondor':
                return []
            base = {'boromir_region': region}
            if not _free_for_side(state, region, side):
                return [base]
            return [dict(base, unit_type=kind)
                    for kind in ('regular', 'elite')
                    if figures_available(state, Nation.GONDOR, kind) > 0] or [base]
    return []


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
        # W1-020/W3-019: deterministic option order - a set of Nation
        # iterates in hash order (PYTHONHASHSEED), which flipped the
        # option INDEX between runs (2024-05-21 L1047 idx 0 vs 1)
        for nat in sorted({u.nation for u in a.units
                           if u.kind == UnitKind.REGULAR
                           and _son(u.nation) == Side.FREE_PEOPLES},
                          key=lambda n: n.value):
            if _can_recruit(state, nat, 'elite'):
                opts.append({"strider_region": sanitize(k),
                             "nation": nat.value.lower()})
    return opts


@_spec("Cirdan's Ships")
def _(state, side):
    # USER RULING 2026-07-21g: a besieged FP stronghold counts - test FP
    # PRESENCE (any FP unit), never units[0].
    from event_effects import SHADOW_NATIONS as _SN, _cirdan_unit_combos
    combos = _cirdan_unit_combos(state)
    out = []
    for r in _regions(_COASTAL):
        a = _army_at(state, r)
        if a and any(u.nation not in _SN for u in a.units):
            out.extend({"region": r, "unit_combo": combo} for combo in combos)
            if not combos:
                out.append({"region": r})
    return out


@_spec("Guards of the Citadel")
def guards_of_the_citadel_options(state, side):
    """FP Strategy14: choose the supplied Gondor Regular or Elite.

    RB22/27 and Almanac21-22 require maximum feasible placement, including
    the independent Leader. A single empty continuation preserves that
    Leader-only or blocked zero-effect outcome without inventing a decline
    while an Army unit can be recruited. Card recruitment may overstack;
    the owner chooses the return only after the recruitment resolves.
    """
    if side != Side.FREE_PEOPLES:
        return []
    if not _free_for_side(state, 'minas tirith', side):
        return [{}]
    from ai import figures_available
    options = [{'region': 'minas tirith', 'unit_type': kind}
               for kind in ('regular', 'elite')
               if figures_available(state, Nation.GONDOR, kind) > 0]
    return options or [{}]


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
def faramirs_rangers_options(state, side):
    """Almanac18: choose a strike target, or a legal recruitment-only leg.

    Unit kind is selected after the strike/casualty leg, in the handler's
    direct pending. An empty deterministic continuation is a legal effect,
    not an absent target warning or an optional decline of the strike.
    """
    if side != Side.FREE_PEOPLES:
        return []
    from event_effects import _faramir_targets, _faramir_fp_army_present
    targets = _faramir_targets(state)
    if targets:
        return [{'target_region': r} for r in targets]
    return [{}] if _faramir_fp_army_present(state) else []


@_spec("Help Unlooked For")
def _(state, side):
    # almanac A10: the card attacks a Shadow army BESIEGING a stronghold from an
    # adjacent FP army holding at least one At-War unit
    from combat import garrison_side_for
    from game_logic import can_attack_with_army
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
            if aa and can_attack_with_army(state, aa, side=Side.FREE_PEOPLES)[0]:
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
    """Shared spec for the companion-separation cards: card text
    "Separate one Companion or ONE GROUP of Companions" (1.58(e)) -
    singles plus every size>=2 subset, range = progress + the group's
    highest Level + extra (the handlers' own range rule)."""
    from itertools import combinations as _comb

    def fn(state, side):
        opts = []
        comps = _separable(state)
        if not comps:
            return opts
        for c in comps:
            for dest in _sep_destinations(
                    state, state.fellowship.progress + c.level + extra):
                opts.append({"companions": [c.name],
                             "destination": dest})
        if group:
            for _r in range(2, len(comps) + 1):
                for _sub in _comb(comps, _r):
                    _names = [c.name for c in _sub]
                    _gmax = max(c.level for c in _sub)
                    for dest in _sep_destinations(
                            state, state.fellowship.progress + _gmax + extra):
                        opts.append({"companions": _names,
                                     "destination": dest})
        return opts
    return fn


CARD_OPTION_SPECS["i will go alone"] = _separation_card(1, True)
@_spec("There and Back Again")
def there_and_back_again_options(state, side):
    # Almanac14: Gollum needs no separation; the political leg still applies.
    if (getattr(state.fellowship.guide, 'name', None) == 'Gollum'
            and not _separable(state)):
        return [{'_taba_done': True}]
    return _separation_card(1, True)(state, side)


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
        _here = [l for l in a.leaders if isinstance(l, Companion)]
        for l in _here:
            for dest in _move_dests(state, sanitize(k), 4,
                                    allow_besieged=True):
                opts.append({"companion_action": f"move_{l.name}",
                             "destination": dest})
        # card: "move ... one group of Companions" - a co-located
        # board group flies together (as Level 4)
        for _r in range(2, len(_here) + 1):
            for _sub in _comb(_here, _r):
                for dest in _move_dests(state, sanitize(k), 4,
                                        allow_besieged=True):
                    opts.append({"companion_action": "move_group",
                                 "companions": [c.name for c in _sub],
                                 "destination": dest})
    return opts


@_spec("We Prove Swifter")
def _(state, side):
    from itertools import combinations as _comb
    from ai import _companion_group_movement
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
        _here = [l for l in a.leaders if isinstance(l, Companion)]
        for l in _here:
            for dest in _move_dests(state, sanitize(k),
                                    _companion_group_movement([l]) + 2,
                                    allow_besieged=True):
                opts.append({"companion_action": f"move_{l.name}",
                             "destination": dest})
        # card: "move ... one group of Companions" - a co-located board
        # group moves together, range = the group's max Level + 2
        for _r in range(2, len(_here) + 1):
            for _sub in _comb(_here, _r):
                _gmax = _companion_group_movement(_sub)
                for dest in _move_dests(state, sanitize(k), _gmax + 2,
                                        allow_besieged=True):
                    opts.append({"companion_action": "move_group",
                                 "companions": [c.name for c in _sub],
                                 "destination": dest})
    return opts


# --------------------------------------------------------------------------
# Shadow
# --------------------------------------------------------------------------

@_spec("Cruel Weather")
def _(state, side):
    import game_logic
    cur = game_logic.fellowship_region(state)
    return [{"target_region": sanitize(r)}
            for r in state.board.adjacent(cur)]


@_spec("Dreadful Spells")
def _(state, side):
    # QA C2: target_region must hold the NAZGUL as part of a Shadow ARMY (units -
    # a lone Nazgul is not "that Army", USER 2026-07-16 item 5; ONE predicate
    # with gate and handler); the FP army is in or adjacent to it
    from event_effects import _is_shadow_army_with_nazgul
    opts = []
    for k, a in state.armies.items():
        if not _is_shadow_army_with_nazgul(a):
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


# The Ringwraiths Are Abroad: no spec (1.131) - the handler seeds the
# per-figure '__nazgul_moves__' chain


@_spec("The Fighting Uruk-hai")
def _(state, side):
    # card text: the Isengard army must be BESIEGING a stronghold; the attack
    # targets THAT stronghold (co-located siege assault, USER 2026-07-09)
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
    from event_effects import _grond_siege_regions
    return [{"attack_from": region, "attack_target": region}
            for region in _grond_siege_regions(state)]


@_spec("Corsairs of Umbar")
def _(state, side):
    # almanac A10: GONDOR coastal targets; FP-occupied targets are legal (the
    # move starts the battle); no move while a Shadow army is besieged inside
    # Umbar
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
        # Both siege states are legal (Almanac A10): join Shadow besiegers
        # outside an FP garrison, or relieve a besieged Shadow garrison.
        # Their different battle/merging timing belongs to the handler.
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
    # USER 2026-07-08: BCC does NOT muster the Witch-king.  Card: recruit two
    # Nazgul in the WK's region OR move any/all Nazgul; then move/attack with
    # the WK's Army.  The recruit region is the WK's own (no choice); the
    # modal choice is recruit-vs-move.
    opts = []
    wk_region = None
    for k, a in state.armies.items():
        if any('witchking' in __import__('re').sub(
                r'[^a-z]', '', str(getattr(l, 'name', '')).lower())
               for l in a.leaders):
            wk_region = sanitize(k)
            break
    if wk_region is not None:
        # never offer a guaranteed no-op (QA C5): the recruit arm needs Nazgul
        # in the pool
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
    # USER RULING 2026-08-25 (verbatim: "The hide option does exist, yes ...
    # you always get the heal though"), supersedes the misrecorded 1.229
    # basis: heal ALWAYS (handler); Gollum guide: HIDDEN -> optional move,
    # REVEALED -> optional hide; no die enters the Hunt Box. Without Gollum,
    # heal-only completes automatically through the singleton option.
    g = state.fellowship.guide
    if g is None or 'Gollum' not in getattr(g, 'name', ''):
        return [{"action": "heal_only"}]
    if state.fellowship.hidden:
        return [{"action": "move"}, {"action": "heal_only"}]
    return [{"action": "hide"}, {"action": "heal_only"}]


@_spec("Book of Mazarbul")
def _book_of_mazarbul_options(state, side):
    # Almanac18/46: no separated movable Companion is required to play.
    # Complete automatically when there is nothing to move; the existing
    # terminator still applies the independent Erebor/Ered Luin trigger.
    return _companion_move_card(state, side) or [{'_ffm_done': True}]


@_spec("Fear! Fire! Foes!")
def _fear_fire_foes_options(state, side):
    # No separated Companion is required. Still resolve the independent
    # North trigger when no movement choice remains.
    return _companion_move_card(state, side) or [{'_ffm_done': True}]


@_spec("Shadows Gather")
def _(state, side):
    from event_effects import _sg_join_options
    return _sg_join_options(state, 3)


@_spec("The Shadow Lengthens")
def _(state, side):
    # 1.117 (TSiM pattern): each of the (up to two) moves is an INDEPENDENT
    # chained pending - the spec offers single joins, the handler chains the
    # second move.
    from event_effects import _sg_join_options
    return _sg_join_options(state, 2) + [{"_tsl_done": True}]


@_spec("The Shadow is Moving")
def _(state, side):
    # single 1-region moves; moves 2-4 are INDEPENDENT chained pendings
    # (1.117 / 1.112.7).  1.211h / row 43: only a besieged Shadow GARRISON may
    # not move; a besieger stack may (lifting the siege) and may be joined -
    # predicates shared with the handler and the chain surface
    from event_effects import (SHADOW_NATIONS, _besieged_garrison_source,
                               _tsim_dest_ok)
    singles = []
    for k, a in state.armies.items():
        frm = sanitize(k)
        if _besieged_garrison_source(state, frm, Side.SHADOW):
            continue
        if not any(u.nation in SHADOW_NATIONS for u in a.units):
            continue
        for adj in state.board.adjacent(frm):
            to = sanitize(adj)
            if not _tsim_dest_ok(state, to):
                continue
            singles.append({"from_region": frm, "to_region": to})
    return singles + [{"_tsim_done": True}]


@_spec("Many Kings to the Service of Mordor")
def _(state, side):
    from shadow_recruit_allocations import allocation_options
    scarce = allocation_options(state, "Many Kings to the Service of Mordor")
    if scarce is not None:
        return scarce
    from itertools import combinations
    from event_effects import _free_se_settlements
    free = _free_se_settlements(state)
    if not free:
        return []
    k = min(3, len(free))
    return [{"settlements": list(c)} for c in combinations(free, k)]


@_spec("Pits of Mordor")
def _(state, side):
    from shadow_recruit_allocations import allocation_options
    scarce = allocation_options(state, "Pits of Mordor")
    if scarce is not None:
        return scarce
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
    # The card REPLACES TWO Sauron Regulars: each option is a COMPLETE choice -
    # [r, r] (two in one army) or [r1, r2] (one each, almanac A15).  A
    # 1-upgrade option is offered ONLY as the max-extent fallback when two
    # are physically impossible (Regulars on the board or Elites in the
    # pool).  Only Regular SAURON units count (membership tests).
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
    # ONE ANSWERER (USER 2026-08-18): ai.figures_available owns the cap
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


# Nazgul Strike!: no spec (1.131) - the handler seeds the '__nazgul_moves__'
# chain and the either/or fires at chain end (_nazgul_card_post)


@_spec("Lure of the Ring")
def _(state, side):
    # Card text: the Companion is DRAWN first, then the FREE PEOPLES player
    # chooses (audit EE2-3): a single auto-pick option carries '_from_pending'
    # into stage 1 (the draw), which opens the FP's cross-side pending.
    return [{}]


# Nazgul Search: no spec (1.131) - the handler seeds the chain; the reveal
# check runs at chain end ('_naz_post': 'search')


@_spec("The Lidless Eye")
def _(state, side):
    from event_effects import _lidless_options
    return _lidless_options(state)


@_spec("Denethor's Folly")
def _(state, side):
    from event_effects import _denethor_targets
    nations = sorted({leader.nation.name for leader in _denethor_targets(state)})
    return [{"nation": nation.lower()} for nation in nations] or [{}]


def _fixed_shadow_recruit_options(name):
    def options(state, side):
        from shadow_recruit_allocations import allocation_options
        scarce = allocation_options(state, name)
        return scarce if scarce is not None else [{}]
    return options


for _recruit_name in ("Orcs Multiplying Again", "Monsters Roused", "A New Power is Rising"):
    CARD_OPTION_SPECS[_recruit_name.lower()] = _fixed_shadow_recruit_options(_recruit_name)


def get_card_options(state, side, card_name):
    """All legal choices dicts for the card, [] if unspecced/no options."""
    fn = CARD_OPTION_SPECS.get(str(card_name).lower())
    if fn is None:
        return []
    # [] means enumeration succeeded and found no choices. A crashed spec
    # must propagate: the card-play caller can then stop before spending the
    # card/die instead of treating the failure as a zero-effect legal play.
    opts = fn(state, side)
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
    # Multi-region specs carry a LIST (strongholds / regions / settlements):
    # expose the FIRST element so the trio is spatially visible to the action
    # encoder (v16) instead of collapsing to region index -1.
    v = (choices.get("strongholds") or choices.get("regions")
         or choices.get("settlements"))
    if isinstance(v, (list, tuple)) and v:
        return str(v[0])
    return None
