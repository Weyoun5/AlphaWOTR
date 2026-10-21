# event_effects.py - Event card effect handlers for War of the Ring
"""Event card effects for recruitment, movement, hunt tiles, and political tracks."""

from __future__ import annotations
import random
import re

# card recruits may OVERSTACK then resolve immediately (almanac,
# 'Points common to all Shadow recruitment cards'); set only
# around play_event_card handler dispatch
CARD_RECRUIT_OVERSTACK = False
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState

from game_state import Side, Nation, UnitKind, Unit, Leader, Army
from board import sanitize
import board


# ---------------------------------------------------------------------------
# Nation mapping
# ---------------------------------------------------------------------------

SHADOW_NATIONS = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}
FP_NATIONS = {Nation.ELVES, Nation.DWARVES, Nation.GONDOR, Nation.THE_NORTH, Nation.ROHAN}


# RANDOMNESS FUNNELS (session 33): every card-effect d6 batch and hunt
# tile draw routes through game_logic's module-level funnels so the
# LIVE bridge can source them from the Java client's own handlers.
# Lazy imports keep the module graph acyclic; default behavior is
# byte-identical to the inlined random calls these wrappers replaced.
def _d6_batch(n, ctx='card effect'):
    import game_logic as _gl
    return _gl.roll_d6_batch(n, ctx)


def _tile_choice(state):
    import game_logic as _gl
    return _gl.pick_hunt_tile(state)


def _tile_pop(state):
    import game_logic as _gl
    return _gl.pop_hunt_tile(state)

NATION_KEYWORDS = {
    "gondor": Nation.GONDOR,
    "rohan": Nation.ROHAN,
    "elven": Nation.ELVES,
    "elves": Nation.ELVES,
    "dwarven": Nation.DWARVES,
    "dwarves": Nation.DWARVES,
    "north": Nation.THE_NORTH,
    "isengard": Nation.ISENGARD,
    "sauron": Nation.SAURON,
    "southron": Nation.SOUTHRON_EASTERLINGS,
    "easterling": Nation.SOUTHRON_EASTERLINGS,
}


def parse_nation(text: str) -> Optional[Nation]:
    """Extract nation from text."""
    text_lower = text.lower()
    for keyword, nation in NATION_KEYWORDS.items():
        if keyword in text_lower:
            return nation
    return None


# ---------------------------------------------------------------------------
# Recruitment Helpers
# ---------------------------------------------------------------------------

def _find_army_key(state: "GameState", region: str) -> str:
    """Find the actual key for a region in the armies dict, handling case differences."""
    rkey = sanitize(region)
    # Check for exact match first
    if rkey in state.armies:
        return rkey
    # Check for case-insensitive match
    for key in state.armies.keys():
        if sanitize(key) == rkey:
            return key
    # Return sanitized version for new armies
    return rkey


def _besieged_garrison_source(state, frm, side):
    """1.211h + row 43: TRUE when region `frm` is a besieged stronghold
    whose GARRISON belongs to `side` - i.e. moving `side` units OUT of
    `frm` would illegally march the garrison out of the walls.  PRECISE
    (garrison only): a BESIEGER lifting its siege stays legal (the
    besieger holds the region, not the garrison), so this returns False
    for it - never refuse a logged besieger relocation."""
    from combat import garrison_side_for
    frm = sanitize(frm)
    bs = getattr(state, 'besieged_strongholds', set())
    return frm in bs and garrison_side_for(state, frm) == side


def _unit_may_enter_nation(state, unit, to_san):
    """Amendment 1.222(b) (standing ruling 2026-08-25), per UNIT: a
    non-belligerent unit may never enter ANOTHER nation's borders; its
    own nation's regions and free regions are unrestricted, and an
    At-War unit moves anywhere.  ONE predicate for every voluntary
    card move (Through a Day and a Night, Paths of the Woses) - the
    same test the die-driven army move applies inline in
    ai._get_army_move_actions (not a callable helper there)."""
    from ai import _region_nation
    dest_nation = _region_nation(sanitize(to_san))
    if dest_nation is None:
        return True
    return (state.political.is_at_war(unit.nation)
            or unit.nation == dest_nation)


def _card_tile_damage(tile):
    """Damage of a card-drawn hunt tile: a Die icon (Shelob's Lair)
    rolls a d6 through the funnel (rulebook: 'considered to be a
    numbered tile for all Event card effects'); every other tile reads
    its printed number.  ONE body for Isildur's Bane and Foul Thing
    from the Deep."""
    if getattr(tile, 'is_dice_roll', False):
        return _d6_batch(1, 'shelob d6 damage')[0]
    return int(getattr(tile, 'damage', 0) or 0)


def _is_shadow_army_with_nazgul(army) -> bool:
    """'A Shadow Army containing a Nazgul' (USER 2026-07-16 item 5): the
    entry must hold Shadow Army UNITS - a lone Nazgul figure is not an
    Army.  ONE predicate for the Dreadful Spells gate, spec and handler."""
    return (army is not None and _army_contains_nazgul(army)
            and any(u.nation in SHADOW_NATIONS for u in army.units))


def _card_by_name(name):
    """The card object whose card_name == name (None when unknown) - for
    handlers re-entered from a pending that carries only the name."""
    from cards import get_cards
    for c in get_cards().values():
        if getattr(c, 'card_name', None) == name:
            return c
    return None


def _default_unit_type(state, nation, choices, key="unit_type"):
    """USER RULE (2026-07-07, amendment 1.45): mustering event cards that
    offer 'Recruit ... (Regular or Elite)' place the ELITE whenever the
    reinforcement pool has one — "humans do this ~99% of the time".
    Correction (same day): "king brand's men only puts regulars down.
    otherwise yes put elites down if possible." — cards whose printed
    text restricts the unit type (King Brand's Men, Many Kings, The King
    is Revealed, Orcs Multiplying Again, the Dunland half of A New Power
    is Rising) are NOT routed through this helper; the text wins.
    Explicit choices (log-forced placements) always win over the default.

    PERMANENT SANCTIONED HARDCODE (USER 2026-08-02, comment-archaeology
    review): "Muster event elite first is another exception."  NOT a
    learned-pending candidate — do not flag in future deferred-decision
    sweeps.
    """
    ut = choices.get(key)
    if ut:
        return ut
    try:
        from ai import _can_recruit
        if _can_recruit(state, nation, 'elite'):
            return 'elite'
    except Exception:
        pass
    return 'regular'


def _default_unit_combo(state, nation, choices, key="unit_combo"):
    """Two-unit variant of the 1.45 elite default: as many elites as the
    reinforcement pool allows (2elite > mixed > 2regular)."""
    combo = choices.get(key)
    if combo:
        return combo
    try:
        # ONE ANSWERER (USER ruling 2026-08-18, strict component
        # limits) - this was an inline copy of the cap arithmetic.
        from ai import figures_available as _figs_avail9
        n_el = _figs_avail9(state, nation, 'elite')
        if n_el >= 2:
            return '2elite'
        if n_el == 1:
            return 'mixed'
    except Exception:
        pass
    return '2regular'


def recruit_unit(state: "GameState", region: str, nation: Nation,
                 unit_kind: UnitKind,
                 ignore_captured: bool = False) -> Optional[Unit]:
    """Recruit a unit in a region (respects stacking: 10, or 5 if besieged).

    Returns None when the stacking limit blocks the placement (per rules,
    excess units stay in/return to reinforcements).
    """
    # Reinforcement-pool guard (physical piece limits).  Only the circular-
    # import window may fall through; a real failure stays LOUD.
    try:
        from ai import _can_recruit
    except ImportError:
        _can_recruit = None
    if _can_recruit is not None:
        typ = 'elite' if unit_kind == UnitKind.ELITE else 'regular'
        if not _can_recruit(state, nation, typ):
            return None
    # USER RULES (2026-07-08): "you can recruit while besieged, or
    # recruit if you are besieging" (e.g. FP retaking Woodland Realm
    # plays Thranduil's Archers into the besieging army); recruiting
    # NEVER regains control of a captured settlement (only Dead Men of
    # Dunharrow explicitly does); you cannot recruit into a CAPTURED
    # settlement without a besieging army there, nor into a region
    # holding an enemy field army.
    from combat import side_of_nation as _son
    import game_logic as _gl
    my_side = _son(nation)
    rsan = sanitize(region)
    _ec = getattr(state, 'enemy_controlled', {}) or {}
    holder = next((sd for sd, regs in _ec.items() if rsan in regs), None)
    _a0 = state.armies.get(_find_army_key(state, region))
    my_units_there = bool(_a0 and any(
        _son(u.nation) == my_side for u in _a0.units))
    enemy_units_there = bool(_a0 and any(
        _son(u.nation) != my_side for u in _a0.units))
    besieged = getattr(state, 'besieged_strongholds', set())
    is_besieged = rsan in besieged
    if (holder is not None and holder != my_side and not my_units_there
            and not ignore_captured):
        # captured settlement, no besieging army of mine.  Dead Men of
        # Dunharrow (QA 2026-07-11) explicitly recruits INTO the
        # Shadow-held settlement and then flips control - it passes
        # ignore_captured=True (pool/stacking checks still apply).
        return None
    if enemy_units_there and not is_besieged:
        return None  # enemy field army present

    rkey = _find_army_key(state, region)
    if rkey not in state.armies:
        # Create new army with the sanitized key
        rkey = sanitize(region)
        state.armies[rkey] = Army(region=state.board.region(rkey))
    army = state.armies[rkey]
    # Siege cap 5 applies to the GARRISON side inside the stronghold;
    # the besieging side recruits against the normal cap of 10
    my_count = sum(1 for u in army.units if _son(u.nation) == my_side)
    if is_besieged:
        from game_state import Side as _S
        # the garrison is whoever currently holds the stronghold: the
        # capturer if it was captured, else the native side
        if holder is not None:
            garrison_side = holder
        else:
            garrison_side = (_S.FREE_PEOPLES if _gl.is_fp_region(rsan)
                             else _S.SHADOW)
        cap = 5 if my_side == garrison_side else 10
    else:
        cap = 10
    if my_count >= cap:
        if not CARD_RECRUIT_OVERSTACK:
            return None
        # Recruitment CARDS may over-stack (almanac); WHICH excess returns is
        # the owner's decision (1.112.7) - record the region; resolve_overstacks
        # runs after the effect (learned pending on the pending path,
        # regulars-first heuristic for teacher/GUI bookkeeping).
        unit = Unit(nation, unit_kind)
        army.units.append(unit)
        _ovr = getattr(state, '_overstack_regions', None)
        if _ovr is None:
            _ovr = set()
            state._overstack_regions = _ovr
        _ovr.add(rkey)
        return unit
    unit = Unit(nation, unit_kind)
    army.units.append(unit)
    return unit


def _stack_cap(state: "GameState", rsan: str, my_side) -> int:
    """Stacking cap for *my_side* in region: 5 for the garrison of a
    besieged stronghold, 10 otherwise (mirrors recruit_unit)."""
    import game_logic as _gl
    from game_state import Side as _S
    # sanitize: besieged_strongholds holds sanitized keys, callers pass
    # display-cased ones (QA C6)
    rsan = sanitize(rsan)
    if rsan not in getattr(state, 'besieged_strongholds', set()):
        return 10
    _ec = getattr(state, 'enemy_controlled', {}) or {}
    holder = next((sd for sd, regs in _ec.items() if rsan in regs), None)
    if holder is not None:
        garrison_side = holder
    else:
        garrison_side = (_S.FREE_PEOPLES if _gl.is_fp_region(rsan)
                         else _S.SHADOW)
    return 5 if my_side == garrison_side else 10


def _overstack_excess(state: "GameState", army, rsan: str,
                      my_side) -> int:
    from combat import side_of_nation as _son
    mine = sum(1 for u in army.units if _son(u.nation) == my_side)
    return mine - _stack_cap(state, rsan, my_side)


def resolve_overstacks(state: "GameState", learned: bool = True) -> None:
    """Resolve regions recorded in state._overstack_regions after a
    card effect completes (P1, USER ruling 1.112.7: 'resolving the
    overstack is another pending learned decision... not just here,
    but in general').

    learned=True: open a pending_card_option ('__overstack__', direct)
    so the owning side PICKS which excess units return to
    reinforcements (one unit per decision, chained by the handler).
    learned=False: teacher/fx/GUI bookkeeping - regulars-first
    heuristic identical to the pre-P1 inline behavior (the log's own
    piece rows are the ground truth there).

    Removal is a plain army.units.remove: the reinforcement pool is
    COMPUTED (max - deployed - FP tracked casualties), so an untracked
    removal IS a return to reinforcements for both sides.
    """
    regs = getattr(state, '_overstack_regions', None)
    if not regs:
        return
    if learned and getattr(state, 'pending_card_option', None) is not None:
        return  # a chain is in flight; re-scanned after the next stage
    from combat import side_of_nation as _son
    from game_state import Side as _S
    for rsan in sorted(regs):
        army = state.armies.get(_find_army_key(state, rsan))
        if army is None:
            regs.discard(rsan)
            continue
        resolved = True
        for my_side in (_S.FREE_PEOPLES, _S.SHADOW):
            cap = _stack_cap(state, rsan, my_side)
            mine = [u for u in army.units if _son(u.nation) == my_side]
            excess = len(mine) - cap
            if excess <= 0:
                continue
            if not learned:
                removable = ([u for u in mine
                              if u.kind == UnitKind.REGULAR]
                             + [u for u in mine
                                if u.kind == UnitKind.ELITE])
                for u in removable[:excess]:
                    army.units.remove(u)
                continue
            # offer REGULAR removals before ELITE ones (matches the
            # learned=False regulars-first heuristic)
            kinds = sorted({(u.nation.value,
                             'elite' if u.kind == UnitKind.ELITE
                             else 'regular') for u in mine},
                           key=lambda _nk: (_nk[1] == 'elite', _nk[0]))
            opts = [{'overstack_remove': {'region': rsan, 'nation': nv,
                                          'kind': kd}}
                    for nv, kd in kinds]
            if len(opts) == 1:
                # forced removal: no decision to learn - apply now
                # (single-option pendings are auto-picked anyway)
                _overstack_remove_one(state, army, my_side,
                                      opts[0]['overstack_remove'])
                if _overstack_excess(state, army, rsan, my_side) > 0:
                    resolved = False  # rescan (kinds may differ now)
                continue
            _tag = 'fp' if my_side == _S.FREE_PEOPLES else 'sh'
            state.pending_card_option = {
                'card_id': '__overstack__',
                'card_name': '__overstack__',
                'side': _tag, 'decider': _tag, 'direct': True,
                'options': opts}
            return  # one decision at a time; the handler chains
        if resolved:
            regs.discard(rsan)
        else:
            return resolve_overstacks(state, learned=learned)


def _overstack_remove_one(state: "GameState", army, my_side,
                          spec: dict) -> bool:
    """Return ONE (nation, kind) unit of *my_side* to reinforcements."""
    from combat import side_of_nation as _son
    want_kind = (UnitKind.ELITE if spec.get('kind') == 'elite'
                 else UnitKind.REGULAR)
    want_nat = str(spec.get('nation', '')).lower()
    for u in list(army.units):
        if (_son(u.nation) == my_side and u.kind == want_kind
                and u.nation.value.lower() == want_nat):
            army.units.remove(u)
            return True
    return False


def _nazgul_chain_options(state, arrivals: dict) -> list:
    """Remaining single-nazgul flight options for the current Character
    die (USER 2026-07-11): every region whose PLAIN-nazgul count
    exceeds the figures that already flew there this die, paired with
    the standard nazgul destinations.  One flight per option."""
    import ai as _ai9
    from game_state import Nazgul as _NazN
    opts = []
    cand = _ai9._get_nazgul_move_actions(state, -1)
    seen = set()
    _wk_ok9 = bool(arrivals.get('__wk_ok__')) \
        and not arrivals.get('__wk__')
    for a in cand:
        pp = a.params
        if pp.get('count') is not None:
            continue
        if pp.get('include_wk'):
            # the WK is a Nazgul for all rules: offer his flight when the chain
            # permits ('__wk_ok__') and he has not flown this die
            if not _wk_ok9:
                continue
            key = ('WK', pp['from'], pp['to'])
            if key in seen:
                continue
            seen.add(key)
            opts.append({'from_region': sanitize(pp['from']),
                         'to_region': sanitize(pp['to']),
                         'include_wk': True,
                         '_naz_arr': dict(arrivals)})
            continue
        frm, dst = pp['from'], pp['to']
        key = (frm, dst)
        if key in seen:
            continue
        seen.add(key)
        army = state.armies.get(_find_army_key(state, frm))
        if army is None:
            continue
        n = sum(1 for l in army.leaders if isinstance(l, _NazN))
        if n - int(arrivals.get(sanitize(frm), 0)) <= 0:
            continue  # every nazgul here already flew this die
        opts.append({'from_region': sanitize(frm),
                     'to_region': sanitize(dst), 'count': 1,
                     '_naz_arr': dict(arrivals)})
    # W4-027 (USER ruling 1.262, 2026-09-02): on a DIE-seeded chain
    # ('__mo_ok__', set only by the nazgul_move executor - never by the
    # card seeds: The Black Captain Commands / Ringwraiths Are Abroad
    # move Nazgul only) the Mouth of Sauron may fly once, up to 3
    # regions companion-style (ai._mouth_reachable).  The option carries
    # the Mouth's identity for the action encoder's EXISTING companion
    # slot ('companions', slot 12) - no new dims.
    if arrivals.get('__mo_ok__') and not arrivals.get('__mo__'):
        _mo_src = _ai9._mouth_region(state)
        if _mo_src is not None:
            for _mo_dst in sorted(_ai9._mouth_reachable(state, _mo_src)):
                opts.append({'from_region': _mo_src, 'to_region': _mo_dst,
                             'include_mouth': True,
                             'companions': ['Mouth of Sauron'],
                             '_naz_arr': dict(arrivals)})
    return opts


def _seed_card_nazgul_chain(state, card, post_key=None, followup=None):
    """Seed the shared per-figure '__nazgul_moves__' chain for a card's
    'Move any or all of the Nazgul' half (amendment 1.131: Nazgul
    Strike! / Nazgul Search / The Ringwraiths Are Abroad).  The
    pending's card_id keeps the REAL card so the action encoder embeds
    the card identity (dim 33); card_name routes executor dispatch to
    the chain handler.  The card's 'Then, ...' half is deferred to the
    chain's end via '_naz_post' / '_then_followup'.  Returns True when
    a pending was created (no movable nazgul -> False, caller falls
    through to its immediate-consequence path)."""
    opts = _nazgul_chain_options(state, {'__wk_ok__': 1})
    if not opts:
        return False
    term = {'_naz_done': True}
    for _o in opts + [term]:
        if post_key:
            _o['_naz_post'] = post_key
        if followup:
            _o['_then_followup'] = followup
    opts.append(term)
    state.pending_card_option = {
        'card_id': getattr(card, 'card_id', '__nazgul_moves__'),
        'card_name': '__nazgul_moves__',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': opts}
    return True


def _nazgul_search_reveal_check(state, result):
    """Nazgul Search's 'Then, if at least one Nazgul is in the region
    with the Fellowship, the Fellowship is revealed' half.

    A card reveal is a FULL reveal (amendment 1.132, user QA): the
    Ring-bearers figure moves from the last known position per the
    Progress counter and the counter resets to 0 - same procedure as
    a hunt-tile reveal (_auto_reveal_place also handles the extra
    Shadow-stronghold tile draws along the traced path)."""
    from game_logic import fellowship_region
    fsp_region = sanitize(fellowship_region(state))
    _found_n7 = False
    for region_key, army in state.armies.items():
        if sanitize(region_key) == fsp_region and _army_contains_nazgul(army):
            _found_n7 = True
            state.fellowship.hidden = False
            result["effects"].append(
                "Nazgûl in Fellowship region - Fellowship revealed!")
            import combat as _cmbsr
            try:
                if _cmbsr.FINAL_HITS_HOOK is None \
                        and not state.fellowship.mordor_entered:
                    # AUTO-RESOLVER PURGE (2026-08-16): placement is a
                    # LEARNED FP DECISION - defer to
                    # ai.open_reveal_placement (apply_action tail)
                    state._reveal_place_due = True
                    state._sh_tile_drawn_regions = set()
                    result["effects"].append(
                        "Ring-bearers placement decision pending")
            except Exception:
                pass
            break
    if not _found_n7:
        # N7: state the negative verdict (no-silent-ignore doctrine)
        result["effects"].append(
            "Nazgul Search: no Nazgul with the Fellowship - "
            "nothing found")


def _nazgul_card_post(state, card, key, result):
    """Deferred 'Then, ...' half of a nazgul-movement card, fired when
    the flight chain ends (amendment 1.131)."""
    if key == 'search':
        _nazgul_search_reveal_check(state, result)
    elif key == 'strike':
        # 'Then, if at least one Nazgul is in the region with the
        # Ring-bearers, you may either discard one Free Peoples
        # Character Event card from the table or roll for the Hunt' -
        # the Shadow player's choice, a learned either/or pending
        from game_logic import fellowship_region as _fr131
        fsp = sanitize(_fr131(state))
        army = state.armies.get(_find_army_key(state, fsp))
        if army is None or not _army_contains_nazgul(army):
            # N7 no-silent-ignore: the skipped either/or is STATED
            result["effects"].append(
                "Nazgul Strike: no Nazgul with the Ring-bearers - no "
                "Hunt roll / table-card discard option")
            return
        opts = [{'action': 'hunt'}]
        try:
            from cards import get_card as _gc131
            pc = getattr(state, 'persistent_cards', None)
            for c in (getattr(pc, 'fp_cards', None) or []):
                cd = _gc131(getattr(c, 'card_id', ''))
                if cd is not None and str(getattr(cd, 'deck', '')) \
                        .lower() == 'character':
                    opts.append({'action': 'discard_table_card',
                                 'target_card': c.card_id})
        except Exception:
            pass
        opts.append({'action': 'none'})
        state.pending_card_option = {
            'card_id': getattr(card, 'card_id',
                               'SHADOW_Character_Nazgûl_Strike!'),
            'card_name': 'Nazgûl Strike!',
            'side': 'sh', 'decider': 'sh', 'direct': True,
            'options': opts}


def _effect_nazgul_chain(state: "GameState", card, choices: dict) -> dict:
    """One more nazgul flight on the same Character die, then re-offer
    (pseudo-handler '__nazgul_moves__', USER 2026-07-11)."""
    from game_state import Nazgul as _NazN, Army as _ArmyN
    result = {"effects": []}
    # BCC (1.128): a chain started by The Black Captain Commands carries the
    # deferred 'Then, move or attack with the Witch-king's Army' follow-up,
    # granted only when the chain ends (it outranks a coexisting pending).
    _fu = choices.get('_then_followup')
    _post131 = choices.get('_naz_post')
    if choices.get('_naz_done'):
        result["effects"].append("Nazgul movement ends")
        if _fu:
            _grant_card_followup(
                state, Side.SHADOW, str(_fu.get('kind', 'wk_army')),
                moves=int(_fu.get('moves', 1)))
        if _post131:
            # card-driven chain (1.131): the card's 'Then, ...' half
            # fires now that every flight is resolved
            _nazgul_card_post(state, card, str(_post131), result)
        return result
    frm = sanitize(str(choices.get('from_region', '')))
    dst = sanitize(str(choices.get('to_region', '')))
    if not choices.get('include_mouth'):
        from ai import _nazgul_destination_allowed
        if not _nazgul_destination_allowed(state, dst):
            raise RuntimeError("Nazgul cannot fly into an unbesieged FP-controlled Stronghold")
    arr = {str(k): int(v) for k, v in
           (choices.get('_naz_arr') or {}).items()}
    src_key = _find_army_key(state, frm)
    src = state.armies.get(src_key) if src_key else None
    if src is not None and choices.get('include_mouth'):
        # W4-027 (1.262): the Mouth of Sauron's own leg of the die chain
        # (armies only via game_logic.get_or_create_army)
        import game_logic as _gl_mo
        for l in list(src.leaders):
            if type(l).__name__ == 'Minion' and                     'mouth' in str(getattr(l, 'name', '')).lower():
                src.leaders.remove(l)
                if not src.units and not src.leaders:
                    state.armies.pop(src_key, None)
                _gl_mo.get_or_create_army(state, dst).leaders.append(l)
                arr['__mo__'] = 1
                result["effects"].append(
                    f"The Mouth of Sauron moves {frm} -> {dst}")
                break
    elif src is not None and choices.get('include_wk'):
        # WK flight (adversarial repair: BCC MAJOR)
        import re as _re_wk
        for l in list(src.leaders):
            if type(l).__name__ == 'Minion' and 'witchking' in _re_wk.sub(
                    r'[^a-z]', '', str(getattr(l, 'name', '')).lower()):
                src.leaders.remove(l)
                if not src.units and not src.leaders:
                    state.armies.pop(src_key, None)
                _army_entry_for(state, dst).leaders.append(l)
                arr['__wk__'] = 1
                result["effects"].append(
                    f"The Witch-king flies {frm} -> {dst}")
                break
    elif src is not None:
        naz = [l for l in src.leaders if isinstance(l, _NazN)]
        movable = len(naz) - arr.get(frm, 0)
        if movable > 0 and naz:
            nz = naz[0]
            src.leaders.remove(nz)
            if not src.units and not src.leaders:
                state.armies.pop(src_key, None)
            _army_entry_for(state, dst).leaders.append(nz)
            arr[dst] = arr.get(dst, 0) + 1
            result["effects"].append(f"Nazgul flies {frm} -> {dst}")
    opts = _nazgul_chain_options(state, arr)
    if opts:
        _term = {'_naz_done': True}
        for _o in opts + [_term]:
            if _fu:
                _o['_then_followup'] = _fu
            if _post131:
                _o['_naz_post'] = _post131
        opts.append(_term)
        state.pending_card_option = {
            # card-driven chains keep the REAL card id for the action
            # encoder (dim 33); die chains stay on the pseudo id (1.131)
            'card_id': getattr(card, 'card_id', '__nazgul_moves__'),
            'card_name': '__nazgul_moves__',
            'side': 'sh', 'decider': 'sh', 'direct': True,
            'options': opts}
    else:
        # No movable Nazgul remain: the chain is exhausted - grant the
        # deferred Witch-king army follow-up (BCC 'Then...' half) and/or
        # fire the card's deferred consequence (1.131).
        if _fu:
            _grant_card_followup(
                state, Side.SHADOW, str(_fu.get('kind', 'wk_army')),
                moves=int(_fu.get('moves', 1)))
        if _post131:
            _nazgul_card_post(state, card, str(_post131), result)
    return result


def _effect_overstack(state: "GameState", card, choices: dict) -> dict:
    """Pseudo-handler resolving one learned overstack removal, then
    chaining the next decision while any recorded region is still over
    its cap (P1, USER ruling 1.112.7)."""
    result = {"effects": []}
    from game_state import Side as _S
    spec = (choices or {}).get('overstack_remove') or {}
    rsan = sanitize(str(spec.get('region', '')))
    army = state.armies.get(_find_army_key(state, rsan))
    if army is not None and spec:
        my_side = (_S.FREE_PEOPLES
                   if any(str(spec.get('nation', '')).lower()
                          == n.value.lower() and _fp_nation(n)
                          for n in Nation)
                   else _S.SHADOW)
        # USER RULING 2026-07-27 (Cirdan/Dol-Amroth double-trim): the
        # pending may be STALE — some other path (clamp, casualty)
        # already reduced the stack.  Re-verify the excess before
        # removing; a no-longer-over stack closes the chain untouched.
        from combat import side_of_nation as _son_ov
        _n_ov = sum(1 for u in army.units
                    if _son_ov(u.nation) == my_side)
        if _n_ov <= _stack_cap(state, rsan, my_side):
            _ovr_ov = getattr(state, '_overstack_regions', None)
            if _ovr_ov:
                _ovr_ov.discard(rsan)
                _ovr_ov.discard(str(spec.get('region', '')))
            resolve_overstacks(state, learned=True)
            return result
        if _overstack_remove_one(state, army, my_side, spec):
            result["effects"].append(
                f"Overstack: {spec.get('nation')} {spec.get('kind')} "
                f"returned to reinforcements from {rsan}")
    resolve_overstacks(state, learned=True)
    return result


def _fp_nation(n) -> bool:
    from combat import side_of_nation as _son
    from game_state import Side as _S
    return _son(n) == _S.FREE_PEOPLES


def recruit_leader(state: "GameState", region: str,
                   nation: Nation) -> Optional[Leader]:
    """Recruit a leader in a region.

    QA 2026-07-09: mirrors recruit_unit's guards - recruit_leader had
    NONE, so event cards minted 5th leaders past the 4-figure pool and
    placed lone leaders into captured/enemy-occupied settlements.
    Leaders do not count against stacking, so no stacking check; but a
    leader may never stand alone without own combat units (p.8), so
    placement also requires own units in the region.
    """
    from ai import _can_recruit
    if not _can_recruit(state, nation, 'leader'):
        return None  # reinforcement pool exhausted
    from combat import side_of_nation as _son
    my_side = _son(nation)
    rsan = sanitize(region)
    _ec = getattr(state, 'enemy_controlled', {}) or {}
    holder = next((sd for sd, regs in _ec.items() if rsan in regs), None)
    _a0 = state.armies.get(_find_army_key(state, region))
    my_units_there = bool(_a0 and any(
        _son(u.nation) == my_side for u in _a0.units))
    enemy_units_there = bool(_a0 and any(
        _son(u.nation) != my_side for u in _a0.units))
    is_besieged = rsan in getattr(state, 'besieged_strongholds', set())
    if holder is not None and holder != my_side and not my_units_there:
        return None  # captured settlement, no army of mine there
    if enemy_units_there and not is_besieged:
        return None  # enemy field army present
    if not my_units_there:
        return None  # a leader may never stand alone (rulebook p.8)
    rkey = _find_army_key(state, region)
    if rkey not in state.armies:
        rkey = sanitize(region)
        state.armies[rkey] = Army(region=state.board.region(rkey))
    army = state.armies[rkey]
    leader = Leader(nation)
    army.leaders.append(leader)
    return leader


def recruit_nazgul(state: "GameState", region: str) -> Optional["object"]:
    """Recruit a Nazgul in a region.

    QA 2026-07-10: mirrors recruit_unit/recruit_leader guards - no pool
    cap (8 Nazgul figures) and no captured/enemy-occupancy checks let
    cards mint a 9th Nazgul and leak lone Nazgul into FP-held regions.
    Nazgul MAY stand alone, so no own-units guard."""
    from game_state import Nazgul
    from ai import _can_recruit
    if not _can_recruit(state, Nation.SAURON, 'leader'):
        return None  # all 8 Nazgul figures deployed
    from combat import side_of_nation as _son
    rsan = sanitize(region)
    _ec = getattr(state, 'enemy_controlled', {}) or {}
    holder = next((sd for sd, regs in _ec.items() if rsan in regs), None)
    _a0 = state.armies.get(_find_army_key(state, region))
    my_units_there = bool(_a0 and any(
        _son(u.nation) == Side.SHADOW for u in _a0.units))
    enemy_units_there = bool(_a0 and any(
        _son(u.nation) != Side.SHADOW for u in _a0.units))
    is_besieged = rsan in getattr(state, 'besieged_strongholds', set())
    if holder is not None and holder != Side.SHADOW             and not my_units_there:
        return None  # captured settlement, no besieging army of mine
    if enemy_units_there and not is_besieged:
        return None  # enemy field army present
    rkey = _find_army_key(state, region)
    if rkey not in state.armies:
        rkey = sanitize(region)
        state.armies[rkey] = Army(region=state.board.region(rkey))
    army = state.armies[rkey]
    nazgul = Nazgul(Nation.SAURON)
    army.leaders.append(nazgul)
    return nazgul


# ---------------------------------------------------------------------------
# Political Track Helpers
# ---------------------------------------------------------------------------

def advance_nation(state: "GameState", nation: Nation, steps: int = 1) -> int:
    """Advance a nation on the political track (event-card advancement).

    USER RULING (2026-07, revised): advancing the political counter
    does NOT activate the nation.  Activation comes only from the
    legitimate mechanisms (fellowship declaration, companion entry,
    being attacked) or from cards whose printed text activates -
    Wisdom of Elrond, There and Back Again's trigger, and the
    Mazarbul/Fear-Fire-Foes At-War triggers (set_at_war activates).

    A passive nation's counter may never advance INTO "At War"
    (rulebook p.34) — clamp at box 2 while passive.
    """
    current = state.political.boxes[nation]
    cap = 3 if state.political.active.get(nation, False) else 2
    new_pos = min(cap, current + steps)
    if new_pos > current:
        state.political.boxes[nation] = new_pos
    return state.political.boxes[nation]


def retreat_nation(state: "GameState", nation: Nation, steps: int = 1) -> int:
    """Move a nation back on the political track."""
    current = state.political.boxes[nation]
    new_pos = max(0, current - steps)
    state.political.boxes[nation] = new_pos
    return new_pos


def activate_nation(state: "GameState", nation: Nation) -> None:
    """Activate a nation (flip to colored side)."""
    state.political.active[nation] = True


def set_at_war(state: "GameState", nation: Nation) -> None:
    """Move nation directly to At War."""
    state.political.boxes[nation] = 3
    state.political.active[nation] = True


# ---------------------------------------------------------------------------
# Hunt Tile Helpers
# ---------------------------------------------------------------------------

def _ann_rng(state, ctx, faces=None, result=None, side=None):
    """USER 2026-07-09: every RNG call in the engine is visible in the
    trace/GUI log — dice as dice_roll rows, picks as random_draw."""
    try:
        from game_trace import emit_annotation
        if faces is not None:
            emit_annotation(state, 'dice_roll',
                            {'faces': list(faces), 'ctx': str(ctx),
                             'side': side})
        else:
            emit_annotation(state, 'random_draw',
                            {'ctx': str(ctx), 'result': str(result)})
    except Exception:
        pass


def _tile_key_norm(s0) -> str:
    """Accent/punctuation-proof tile-name key (QA 2026-07-11 cluster H:
    'smeagol helps nice master' never matched the accented
    'Sm\u00e9agol Helps Nice Master' table key - the tile was silently
    never staged in either branch)."""
    import unicodedata
    import re as _re
    s1 = unicodedata.normalize('NFKD', str(s0))
    s1 = ''.join(ch for ch in s1 if not unicodedata.combining(ch))
    return _re.sub(r'[^a-z]', '', s1.lower())


def stage_special_hunt_tile(state: "GameState", tile_name: str) -> bool:
    """Bring a card-added special tile into play (USER FIX 2026-07-08:
    Phial of Galadriel etc. must be VISIBLE and actually enter play).

    On the Mordor Track: add straight to the hunt pool.  Before Mordor:
    place the real tile in the USED pile (hunt_discard) so it shows in
    the GUI's Used pool and is returned to the pool on Mordor entry
    (amendment 1.53's eye+special return) — no invisible pending queue.
    Returns True if placed into the pool immediately.
    """
    from game_state import SPECIAL_HUNT_TILES, HuntTile
    template = None
    for cn, t in SPECIAL_HUNT_TILES.items():
        if _tile_key_norm(cn) == _tile_key_norm(tile_name):
            template = t
            break
    if template is None:
        return False
    tile = HuntTile(
        damage=template.damage, is_eye=template.is_eye,
        reveal=template.reveal, is_special=True,
        special_name=template.special_name,
        stops_movement=template.stops_movement,
        is_dice_roll=getattr(template, 'is_dice_roll', False),
        is_fellowship_special=template.is_fellowship_special,
        image_file=template.image_file)
    if state.fellowship.mordor_entered:
        state.hunt_pool.append(tile)
        random.shuffle(state.hunt_pool)
        return True
    state.hunt_discard.append(tile)   # used pile — visible, returns on entry
    return False


# ---------------------------------------------------------------------------
# Card Drawing
# ---------------------------------------------------------------------------

def draw_cards(state: "GameState", side: Side, deck: str, count: int = 1) -> List[str]:
    """Draw cards from a deck.  Routes through grant_draw (A13 source 7:
    card-effect draws are engine-granted — a deferred client-side draw
    must be recognized as owed by the driver)."""
    from game_logic import grant_draw
    drawn = []
    for _ in range(count):
        try:
            grant_draw(state, side, deck)
            drawn.append(state.hands[side].hand[-1])
        except RuntimeError:
            break  # Deck empty
    return drawn


def _handle_persistent_card(state: "GameState", side: Side, card,
                            result: dict) -> None:
    """Handle cards that stay on the table with ongoing effects."""
    from persistent_effects import create_persistent_card
    from game_state import PersistentCardState

    # Ensure persistent_cards exists
    if not hasattr(state, 'persistent_cards') or state.persistent_cards is None:
        state.persistent_cards = PersistentCardState()

    try:
        persistent = create_persistent_card(card.card_id, card.card_name, side)
        state.persistent_cards.add_card(persistent)
        result["effects"].append(f"{card.card_name} placed on table")
    except ValueError:
        # Unknown card type - still track it but without special effects
        from game_state import PersistentCard
        # Create a generic persistent card entry
        state.persistent_cards.add_card(PersistentCard(
            card_id=card.card_id,
            card_name=card.card_name,
            side=side,
            card_type=None  # Will be handled generically
        ))
        result["effects"].append(f"{card.card_name} placed on table (generic)")


# ---------------------------------------------------------------------------
# Card Effect Registry - Maps card names to specific effect handlers
# ---------------------------------------------------------------------------

CARD_EFFECT_HANDLERS: Dict[str, callable] = {}


def register_effect(card_name: str):
    """Decorator to register a card effect handler."""
    def decorator(func):
        CARD_EFFECT_HANDLERS[card_name.lower()] = func
        return func
    return decorator


# ---------------------------------------------------------------------------
# Event Effect Handlers
# ---------------------------------------------------------------------------

CARD_EFFECT_HANDLERS["__overstack__"] = _effect_overstack
CARD_EFFECT_HANDLERS["__nazgul_moves__"] = _effect_nazgul_chain


def _selfplay_refund(result):
    """Mark a card play as a no-op REFUND (no card discard, no die
    spent).  CARD-FLEET LOW-6/LOW-8 (2026-07-21)."""
    result["refund"] = True
    return result


def play_event_card(state: "GameState", side: Side, card_id: str,
                    choices: dict = None) -> dict:
    """
    Play an event card and apply its effect.

    Args:
        state: Game state
        side: Playing side
        card_id: Card ID to play
        choices: Player choices for interactive effects

    Returns:
        Dict describing what happened
    """
    from cards import get_card

    # Whether the caller supplied ANY choices decides the overstack
    # resolution mode below — capture before normalization.
    _had_no_choices = choices is None
    choices = choices or {}
    result = {"success": False, "message": "", "effects": []}

    card = get_card(card_id)
    if not card:
        result["message"] = f"Card not found: {card_id}"
        return result

    if card.side != side:
        result["message"] = "Cannot play opponent's card"
        return result

    # Check if card is in hand
    if card_id not in state.hands[side].hand:
        result["message"] = "Card not in hand"
        return result

    # Enforce play requirements here (not just in AI action generation) so
    # no caller can play a card illegally
    req_ok, req_reason = check_event_requirement(state, card_id)
    if not req_ok:
        result["message"] = f"Requirement not met: {req_reason}"
        return result

    try:
        # Check for specific card handler first
        handler = CARD_EFFECT_HANDLERS.get(card.card_name.lower())
        if handler:
            global CARD_RECRUIT_OVERSTACK
            CARD_RECRUIT_OVERSTACK = True
            # Publish the state around the WHOLE handler so any casualty
            # application inside it is supply-checked (card handlers run
            # outside battle()).
            from combat import set_active_state as _sas_c
            _prev_as_c = _sas_c(state)   # save/restore (nesting-safe)
            try:
                handler_result = handler(state, card, choices)
            finally:
                CARD_RECRUIT_OVERSTACK = False
                _sas_c(_prev_as_c)
            result["effects"].extend(handler_result.get("effects", []))
            if handler_result.get("message"):
                result["message"] = handler_result["message"]
            # LOW-6/LOW-8: a handler that no-ops sets result['refund'] - keep
            # the card in hand, no discard, no overstack resolve.
            if handler_result.get("refund"):
                # USER RULING 1.230 (2026-08-28, verbatim: "zero effect plays
                # are allowed, always. even in self play"): a handler that
                # could apply NOTHING is a completed zero-effect play (card
                # discarded, die spent).  The refund survives ONLY when legal
                # choices EXISTED and the action carried a bad or absent one
                # (an invalid action, not a zero-effect play).
                _bad_choice = False
                try:
                    from card_options import get_card_options as _gco_r
                    _opts_r = _gco_r(state, side, card.card_name) or []
                    if _opts_r:
                        def _nv(v):
                            return sanitize(str(v)) if isinstance(
                                v, str) else v
                        _bad_choice = not any(
                            all(_nv((choices or {}).get(k)) == _nv(v)
                                for k, v in o.items()
                                if not str(k).startswith('_'))
                            for o in _opts_r)
                except Exception:
                    _bad_choice = False
                if _bad_choice:
                    result["refund"] = True
                    result["success"] = False
                    if not result["message"]:
                        result["message"] = (
                            f"{card.card_name}: no legal effect - not "
                            f"played")
                    return result
                result["zero_effect"] = True
                result["effects"].append(
                    f"{card.card_name}: no effect (zero-effect play, "
                    f"ruling 1.230)")
        else:
            # Fall back to generic effect parsing
            _apply_generic_effects(state, card, choices, result)

            # Handle persistent cards (play on table) - only for generic handling
            # Specific handlers should call _handle_persistent_card themselves
            if "play on the table" in card.event_effect.lower():
                _handle_persistent_card(state, side, card, result)

        # Remove from hand and add to discard (unless persistent)
        if card_id in state.hands[side].hand:
            state.hands[side].hand.remove(card_id)
            # Only add to discard if not persistent
            if "play on the table" not in card.event_effect.lower():
                deck_key = f"{'fp' if side == Side.FREE_PEOPLES else 'sh'}_{card.deck.lower()}"
                state.discard_face_up[deck_key].append(card_id)

        # P1 (USER ruling 1.112.7): resolve any recorded overstack.
        # Learned pending on the pending path (spec picks / chained
        # stages / direct self-play plays with no choices); heuristic
        # for teacher-logged and GUI-dialog choices (log = truth).
        _lrn9 = _had_no_choices or bool(choices.get('_from_pending'))
        import combat as _cmb9
        if _cmb9.FINAL_HITS_HOOK is not None:
            _lrn9 = False
        resolve_overstacks(state, learned=_lrn9)

        result["success"] = True
        if not result["message"]:
            result["message"] = f"Played {card.card_name}"

    except Exception as e:
        # LOUD (audit EE1-09): a handler that raises after mutating must never
        # be reported as a quiet failure; the ai.py executor owns the
        # conservation branch for a raising handler.
        print(f"[play_event_card] handler raised for {card_id}: {e!r}")
        raise

    return result


def _apply_generic_effects(state: "GameState", card, choices: dict, result: dict) -> None:
    """Apply effects based on generic text parsing."""
    effect_text = card.event_effect.lower()

    # Hunt tile cards (fallback only — all 8 tile cards have registered
    # handlers; staging keeps the tile VISIBLE in the used pile pre-Mordor)
    if "add the tile to the hunt pool" in effect_text:
        tile_name = card.card_name
        stage_special_hunt_tile(state, tile_name)
        result["effects"].append(f"Added {tile_name} to Hunt Pool")

    # Political advancement
    if "advance" in effect_text and "political" in effect_text:
        nation = parse_nation(effect_text)
        if nation:
            steps = 2 if "two steps" in effect_text else 1
            new_pos = advance_nation(state, nation, steps)
            result["effects"].append(f"Advanced {nation.value} to {new_pos}")

    # Direct to At War
    if "directly to" in effect_text and "at war" in effect_text:
        nation = parse_nation(effect_text)
        if nation:
            set_at_war(state, nation)
            result["effects"].append(f"{nation.value} moved to At War")

    # Activation
    if "activate" in effect_text:
        nation = parse_nation(effect_text)
        if nation:
            activate_nation(state, nation)
            result["effects"].append(f"Activated {nation.value}")

    # Retreat nation
    if "back one step" in effect_text:
        nation = parse_nation(effect_text)
        if nation:
            new_pos = retreat_nation(state, nation, 1)
            result["effects"].append(f"Retreated {nation.value} to {new_pos}")

    # Recruitment
    if "recruit" in effect_text:
        _handle_recruitment(state, card, effect_text, choices, result)

    # Corruption healing
    if "heal" in effect_text and "corruption" in effect_text:
        heal_amount = 2 if "two corruption" in effect_text else 1
        state.fellowship.corruption = max(0, state.fellowship.corruption - heal_amount)
        result["effects"].append(f"Healed {heal_amount} corruption")

    # Card drawing
    if "draw" in effect_text:
        _handle_draw(state, card.side, effect_text, result)


def _handle_recruitment(state: "GameState", card, effect_text: str,
                        choices: dict, result: dict) -> None:
    """Handle recruitment effects."""
    # Parse what to recruit
    regular_count = 0
    elite_count = 0
    leader_count = 0

    # Count units to recruit
    if "two regular" in effect_text or "2 regular" in effect_text:
        regular_count = 2
    elif "three regular" in effect_text or "3 regular" in effect_text:
        regular_count = 3
    elif "five regular" in effect_text or "5 regular" in effect_text:
        regular_count = 5
    elif "one regular" in effect_text or "1 regular" in effect_text:
        regular_count = 1

    if "two elite" in effect_text or "2 elite" in effect_text:
        elite_count = 2
    elif "one elite" in effect_text or "1 elite" in effect_text:
        elite_count = 1

    if "one leader" in effect_text or "1 leader" in effect_text:
        leader_count = 1

    # Parse nation
    nation = parse_nation(effect_text)
    if not nation:
        # Default based on side
        nation = Nation.GONDOR if card.side == Side.FREE_PEOPLES else Nation.SAURON

    # Flexible unit choice (1.45 elite default: card text offers the
    # choice, pool decides; explicit choices win)
    if "(regular or elite)" in effect_text:
        unit_type = _default_unit_type(state, nation, choices)
        if unit_type == "elite":
            elite_count = max(1, elite_count)
            regular_count = 0
        else:
            regular_count = max(1, regular_count)

    # Parse region
    region = choices.get("recruit_region")
    if not region:
        # Try to extract from effect text
        region_match = re.search(r"in (\w[\w\s]+?)(?:\.|,|$)", effect_text)
        if region_match:
            region = region_match.group(1).strip()

    if region:
        for _ in range(regular_count):
            recruit_unit(state, region, nation, UnitKind.REGULAR)
            result["effects"].append(f"Recruited {nation.value} Regular in {region}")

        for _ in range(elite_count):
            recruit_unit(state, region, nation, UnitKind.ELITE)
            result["effects"].append(f"Recruited {nation.value} Elite in {region}")

        for _ in range(leader_count):
            recruit_leader(state, region, nation)
            result["effects"].append(f"Recruited {nation.value} Leader in {region}")


def _handle_draw(state: "GameState", side: Side, effect_text: str,
                 result: dict) -> None:
    """Handle card draw effects."""
    draw_count = 2 if "two" in effect_text else 1

    if "strategy" in effect_text:
        drawn = draw_cards(state, side, "strategy", draw_count)
        result["effects"].append(f"Drew {len(drawn)} Strategy cards")
    elif "character" in effect_text:
        drawn = draw_cards(state, side, "character", draw_count)
        result["effects"].append(f"Drew {len(drawn)} Character cards")


def _dead_men_recruit(state, target_region, choices, result):
    """Dead Men's final optional recruitment, after the attack and movement.

    Card C22 says "may then recruit up to three"; the almanac pp15-16
    confirms recruitment happens after the Shadow's forced retreat and that
    moving companions alone does not recapture an enemy-held settlement.
    """
    from ai import figures_available
    import combat
    target_key = sanitize(target_region)
    available = min(3, figures_available(state, Nation.GONDOR, 'regular'))
    count = choices.get('recruit_count')
    if (count is None and choices.get('_from_pending')
            and combat.FINAL_HITS_HOOK is None and available > 0):
        state.pending_card_option = {
            'card_id': 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow',
            'card_name': 'Dead Men of Dunharrow',
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'options': [
                {'target_region': target_key, '_dmd_recruit_only': True,
                 'recruit_count': n}
                for n in range(available + 1)]}
        result['effects'].append('Free Peoples chooses how many Gondor regulars to recruit')
        return result
    count = 3 if count is None else int(count)
    if choices.get('_dmd_recruit_only') and not 0 <= count <= available:
        raise ValueError('Dead Men recruitment count is not available')
    # Explicit replay choices retain their historical physical-pool cap.
    count = max(0, min(3, count))
    placed = sum(recruit_unit(state, target_region, Nation.GONDOR,
                             UnitKind.REGULAR, ignore_captured=True) is not None
                 for _ in range(count))
    result['effects'].append(f'Recruited {placed} Gondor regular(s) in {target_region}')
    if target_key in state.enemy_controlled.get(Side.SHADOW, set()):
        army = state.armies.get(_find_army_key(state, target_key))
        from combat import side_of_nation
        if (army and any(side_of_nation(u.nation) == Side.FREE_PEOPLES
                         for u in army.units)
                and not any(side_of_nation(u.nation) == Side.SHADOW
                            for u in army.units)):
            state.enemy_controlled[Side.SHADOW].discard(target_key)
            result['effects'].append(f'Regained control of {target_region}')
    return result


def dead_men_arrival_before_roll(state: "GameState", target_region: str,
                                 aragorn_companions, result=None) -> bool:
    """Apply only the printed arrival and attack politics before any dice.

    The live bridge uses this same phase on a clone to render the selected
    Companions before requesting the client roll. No hit, retreat or recruit
    decision is resolved here.
    """
    from game_state import Companion as _Comp
    import game_logic as _gl9
    from combat import side_of_nation as _son
    target_key = sanitize(target_region)
    if target_key not in {"erech", "lamedon", "pelargir"}:
        raise ValueError('Invalid Dead Men destination')
    result = result if result is not None else {"effects": []}
    # Printed sequence: move the selected group FIRST, then roll Shadow hits.
    # Shared-region casualty and destruction helpers filter by the dying side,
    # so arriving Companions cannot become casualties of the Shadow army.
    _lead_names9 = ("Strider", "Aragorn", "Aragorn - Heir to Isildur")
    _src9, _group9 = None, []
    for rk, a in list(state.armies.items()):
        if sanitize(rk) == target_key:
            continue
        comps_here = [l for l in a.leaders if isinstance(l, _Comp)]
        if not any(l.name in _lead_names9 for l in comps_here):
            continue
        if aragorn_companions is not None:
            if (not isinstance(aragorn_companions, (list, tuple))
                    or len(set(aragorn_companions)) != len(aragorn_companions)
                    or not set(aragorn_companions).issubset({l.name for l in comps_here})
                    or not any(name in aragorn_companions for name in _lead_names9)):
                raise ValueError('Dead Men companion group is not available')
            _keep9 = set(aragorn_companions)
            _group9 = [l for l in comps_here if l.name in _keep9]
        else:
            _group9 = comps_here   # take all co-located companions
        _src9 = rk
        break

    if _src9 is not None:
        source = state.armies[_src9]
        for companion in _group9:
            source.leaders.remove(companion)
            _gl9.place_separated_companion(state, companion, target_key)
        if not source.units and not source.leaders:
            state.armies.pop(_src9, None)
        result['effects'].append('Moved ' + ', '.join(c.name for c in _group9)
                                 + f' to {target_region}')

    shadow_army = state.armies.get(_find_army_key(state, target_key))
    attacked = bool(shadow_army and any(_son(u.nation) == Side.SHADOW
                                       for u in shadow_army.units))
    if attacked:
        _gl9.advance_defenders_political(
            state, target_key, attacker_side=Side.FREE_PEOPLES)
    return attacked


def effect_dead_men_of_dunharrow(state: "GameState", target_region: str,
                                 aragorn_companions: List[str] = None,
                                 choices: dict = None, card=None) -> dict:
    """Dead Men of Dunharrow: Move Aragorn, roll against Shadow army.

    USER RULING 1.254: the hits are the Shadow OWNER's learned
    allocation (_route_card_casualties); when that opens a pending the
    card's remaining legs (forced retreat and recruitment) resume
    through _complete_card_casualty with choices['_dmd_cas_resumed']."""
    choices = choices or {}
    # Chained retreat stage (learned path, amendment 1.129): the Shadow
    # player already picked WHICH legal region the forced-retreat army
    # withdraws to.  The attack (political advance, roll, hits) resolved
    # in the prior stage, after the selected Companions arrived. This
    # option retains their target for recruitment and control handling.
    _dmd_dest = choices.get('destination')
    if _dmd_dest is not None:
        target_region = choices.get('_dmd_target', target_region)
        aragorn_companions = choices.get('companions', aragorn_companions)
    valid_regions = {"erech", "lamedon", "pelargir"}
    target_key = sanitize(target_region)

    if target_key not in valid_regions:
        return {"success": False, "message": f"Invalid target: {target_region}"}

    result = {"success": True, "effects": []}
    if choices.get('_dmd_recruit_only'):
        return _dead_men_recruit(state, target_key, choices, result)

    from game_state import Companion as _Comp
    import game_logic as _gl9
    from combat import side_of_nation as _son

    if _dmd_dest is None and not choices.get('_dmd_cas_resumed'):
        dead_men_arrival_before_roll(state, target_key,
                                    aragorn_companions, result)

    # Attack a Shadow army in the target: political advance, hits on
    # the SHADOW portion, then FORCE RETREAT (destroy if impossible,
    # along with Nazgul and Minions).
    shadow_army = state.armies.get(_find_army_key(state, target_key))
    from combat import reconcile_destroyed_army as _rda9
    if shadow_army and any(_son(u.nation) == Side.SHADOW
                           for u in shadow_army.units):
        if _dmd_dest is None and not choices.get('_dmd_cas_resumed'):
            roll = _d6_batch(1)[0]
            _ann_rng(state, 'card effect roll', faces=[roll])
            import combat as _cmb_dmd
            if _cmb_dmd.FINAL_HITS_HOOK is None:
                # USER RULING 1.254: owner-allocated (learned pending);
                # the teacher keeps its certified logged-composition
                # path below.
                result["effects"].append(f"Rolled {roll}")
                if not _route_card_casualties(
                        state, target_key, Side.SHADOW, roll,
                        card or _card_by_name('Dead Men of Dunharrow'),
                        choices, result):
                    return result
                shadow_army = state.armies.get(
                    _find_army_key(state, target_key))
            else:
                casualties = _apply_card_hits_to_shadow(
                    state, shadow_army, roll)
                result["effects"].append(
                    f"Rolled {roll}, {len(casualties)} Shadow casualties")
        _sh_units9 = [u for u in (shadow_army.units if shadow_army else [])
                      if _son(u.nation) == Side.SHADOW]
        _sh_ldrs9 = [l for l in (shadow_army.leaders if shadow_army else [])
                     if _gl9._leader_side(l) == Side.SHADOW]
        if _sh_units9:
            # Retreat destination = the SHADOW player's choice (1.129): a
            # learned cross-side pending when >1 legal region; teacher/GUI
            # keep the first legal pick.  USER 2026-07-16 item 4: never into
            # an FP-CONTROLLED settlement (native or captured) - use the
            # settlement controller.
            from ai import _settlement_controller as _sc9
            _dests9 = []
            for adj in state.board.adjacent(target_key):
                aj = sanitize(adj)
                if _sc9(state, aj) == Side.FREE_PEOPLES:
                    continue
                other = state.armies.get(_find_army_key(state, aj))
                if other and any(_son(u.nation) == Side.FREE_PEOPLES
                                 for u in other.units):
                    continue
                _dests9.append(aj)
            if _dmd_dest is not None:
                dest = (_dmd_dest if _dmd_dest in _dests9
                        else (_dests9[0] if _dests9 else None))
            elif choices.get('_from_pending') and len(_dests9) > 1:
                state.pending_card_option = {
                    'card_id':
                        'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow',
                    'card_name': 'Dead Men of Dunharrow',
                    'side': 'fp', 'decider': 'sh', 'direct': True,
                    'options': [{'destination': _d9,
                                 '_dmd_target': target_key,
                                 # Recorded counts can be carried through;
                                 # normal play chooses only AFTER retreat.
                                 **({'recruit_count': choices['recruit_count']}
                                    if 'recruit_count' in choices else {})}
                                for _d9 in _dests9]}
                result["effects"].append(
                    "Shadow army must retreat - the Shadow player picks "
                    "the destination (pending)")
                return result
            else:
                dest = _dests9[0] if _dests9 else None
            if dest is not None:
                _dst9 = _army_entry_for(state, dest)
                for u in _sh_units9:
                    shadow_army.units.remove(u)
                    _dst9.units.append(u)
                for l in _sh_ldrs9:
                    shadow_army.leaders.remove(l)
                    _dst9.leaders.append(l)
                # an occupied retreat destination can exceed the cap - record
                # it for the learned overstack resolution (1.112.7)
                _own_d9 = sum(1 for u in _dst9.units
                              if _son(u.nation) == Side.SHADOW)
                if _own_d9 > _stack_cap(state, sanitize(dest),
                                        Side.SHADOW):
                    _ovr9 = getattr(state, '_overstack_regions', None)
                    if _ovr9 is None:
                        _ovr9 = set()
                        state._overstack_regions = _ovr9
                    _ovr9.add(sanitize(dest))
                result["effects"].append(f"Shadow army retreated to {dest}")
                _k9r = _find_army_key(state, target_key)
                if _k9r and not state.armies[_k9r].units \
                        and not state.armies[_k9r].leaders:
                    state.armies.pop(_k9r, None)
            else:
                for u in _sh_units9:
                    shadow_army.units.remove(u)
                # USER RULING 2026-08-15: the ONE reconciler books the
                # eliminated figures
                _rda9(state, target_key, Side.SHADOW,
                      card_name='Dead Men of Dunharrow')
                result["effects"].append(
                    "Shadow army could not retreat - destroyed "
                    "(with Nazgûl and Minions)")
        else:
            # army wiped by the hits: Nazgul/Minions go with it
            _rda9(state, target_key, Side.SHADOW,
                  card_name='Dead Men of Dunharrow')
            result["effects"].append(
                "Shadow army destroyed (with Nazgûl and Minions)")

    return _dead_men_recruit(state, target_key, choices, result)


# ---------------------------------------------------------------------------
# Registered Card Effect Handlers - Free Peoples Character Cards
# ---------------------------------------------------------------------------

@register_effect("Elven Cloaks")
def _effect_elven_cloaks(state: "GameState", card, choices: dict) -> dict:
    """Add Elven Cloaks hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "elven cloaks"):
        result["effects"].append("Added Elven Cloaks tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Elven Cloaks tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Elven Rope")
def _effect_elven_rope(state: "GameState", card, choices: dict) -> dict:
    """Add Elven Rope hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "elven rope"):
        result["effects"].append("Added Elven Rope tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Elven Rope tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Phial of Galadriel")
def _effect_phial_of_galadriel(state: "GameState", card, choices: dict) -> dict:
    """Add Phial of Galadriel hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "phial of galadriel"):
        result["effects"].append("Added Phial of Galadriel tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Phial of Galadriel tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Sméagol Helps Nice Master")
def _effect_smeagol_helps(state: "GameState", card, choices: dict) -> dict:
    """Add Sméagol Helps Nice Master hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "smeagol helps nice master"):
        result["effects"].append("Added Sméagol Helps Nice Master tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Sméagol tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Mithril Coat and Sting")
def _effect_mithril_coat(state: "GameState", card, choices: dict) -> dict:
    """Play on table - allows re-drawing hunt tiles."""
    if choices and 'mithril_redraw' in choices:
        # QA 2026-07-11 cluster G: the interposed FP decision
        import game_logic as _gl9
        result = {"effects": []}
        stash = getattr(state, '_mithril_stash', None)
        if stash and stash.get('return_pending'):
            raise ValueError('Mithril redraw already resolved; original tile awaits action end')
        state._mithril_stash = None
        if stash is None:
            return result
        def _apply9(tile9):
            # hunt-card-draws (QA 2026-07-11): the FP redraw may also
            # interpose on the SHADOW hunt-DRAWING cards (Orc Patrol /
            # Foul Thing from the Deep / Isildur's Bane).  Each stashes
            # its own 'apply_kind'; the apply-continuation re-enters the
            # card handler with the (kept or redrawn) tile forced.
            _ak9 = stash.get('apply_kind')
            if _ak9 == 'stronghold':
                return _gl9._apply_shadow_stronghold_tile(state, tile9)
            if _ak9 in ('orc_patrol', 'foul_thing', 'isildur', 'breaking'):
                _hmap9 = {'orc_patrol': _effect_orc_patrol,
                          'foul_thing': _effect_foul_thing,
                          'isildur': _effect_isildurs_bane,
                          'breaking': _effect_breaking_of_fellowship}
                _ch9 = dict(stash.get('choices') or {})
                _ch9['_forced_tile'] = tile9
                _r9 = _hmap9[_ak9](state, None, _ch9)
                for _e9 in (_r9.get('effects') or []):
                    result["effects"].append(_e9)
                return _r9
            if stash.get('mordor_track'):
                r9 = _gl9._apply_mordor_hunt_tile(
                    state, tile9,
                    {"damage": 0, "reveal": False,
                     "stops_movement": False},
                    stash.get('skip_auto_sacrifice', True))
                if r9.get('stops_movement'):
                    # replicate the caller's stop retraction
                    # (amendment 1.116: the step is not gained)
                    state.fellowship.mordor_step = max(
                        0, state.fellowship.mordor_step - 1)
                    result["effects"].append(
                        "Stop tile - the Mordor step is retracted")
                return r9
            if stash.get('eye_no_effect') \
                    and getattr(tile9, 'is_eye', False):
                # Balrog extra draw: an Eye is discarded WITHOUT effect
                state.hunt_discard.append(tile9)
                result["effects"].append(
                    "Eye tile - discarded without effect")
                return {}
            return _gl9._apply_hunt_tile(state, tile9,
                                         stash['successes'],
                                         mordor=stash['mordor'],
                                         eye_no_effect=stash.get('eye_no_effect', False))

        if not choices['mithril_redraw']:
            _apply9(stash['tile'])
            result["effects"].append("Mithril kept - tile applies")
            return result
        # discard the table card, the Shadow redraws
        from game_state import PersistentCardType as _PCTm9
        _pc = getattr(state, 'persistent_cards', None)
        _mc = (_pc.get_by_type(_PCTm9.MITHRIL_COAT) if _pc else None)
        if _mc is not None:
            # CARD CONSERVATION (2026-08-15): a table card removed from
            # play is DISCARDED face-up, never deleted from the game
            from persistent_effects import discard_table_card as _dtc9
            _dtc9(state, _pc.remove_card(_mc.card_id) or _mc)
        result["effects"].append(
            "Mithril Coat and Sting discarded - Shadow redraws")
        # rebuild only if the pool is EMPTY and a tile is required
        # (almanac; the original tile returns during this action).
        # 1.136: the ONE recycler - pre-Mordor special tiles stay out.
        if not state.hunt_pool and _gl9._recycle_hunt_pool(state):
            result["effects"].append("Hunt Pool rebuilt for the redraw")
        # Preserve the publicly known tile across asynchronous damage/reveal
        # choices, including additional draws within the same action.
        state._mithril_stash = dict(stash, return_pending=True)
        if state.hunt_pool:
            t2 = _tile_pop(state)
            try:
                from game_trace import emit_annotation
                emit_annotation(state, 'hunt_tile_draw', {
                    'tile_name': t2.special_name if t2.special_name
                    else repr(t2), 'ctx': 'mithril redraw'})
            except Exception:
                pass
            _apply9(t2)
        _gl9.return_mithril_tile_if_settled(state)
        return result
    result = {"effects": []}
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Mithril Coat and Sting now in play - can force Hunt tile redraw")
    return result


@register_effect("Axe and Bow")
def _effect_axe_and_bow(state: "GameState", card, choices: dict) -> dict:
    """Play on table - reduces hunt damage by 1."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Axe and Bow now in play - can reduce Hunt damage by 1")
    return result


@register_effect("Horn of Gondor")
def _effect_horn_of_gondor(state: "GameState", card, choices: dict) -> dict:
    """Play on table - reduces hunt damage by 1."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Horn of Gondor now in play - can reduce Hunt damage by 1")
    return result


@register_effect("Wizard's Staff")
def _effect_wizards_staff(state: "GameState", card, choices: dict) -> dict:
    """Play on table - can prevent hunt tile draw.

    USER 2026-07-16 (item 10): the interposed FP decision (mirrors the
    Mithril redraw).  When a Shadow hunt-tile draw is pending, FP either
    discards Wizard's Staff to prevent the draw, or lets it proceed."""
    if choices and 'wizardstaff_prevent' in choices:
        import game_logic as _gl9
        result = {"effects": []}
        stash = getattr(state, '_wizardstaff_stash', None)
        state._wizardstaff_stash = None
        if stash is None:
            return result
        if choices['wizardstaff_prevent']:
            # discard the table card; the Shadow draws NO tile this hunt
            from game_state import PersistentCardType as _PCTw9
            _pc = getattr(state, 'persistent_cards', None)
            _ws = (_pc.get_by_type(_PCTw9.WIZARD_STAFF) if _pc else None)
            if _ws is not None:
                # CARD CONSERVATION (2026-08-15): discarded face-up
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, _pc.remove_card(_ws.card_id) or _ws)
            result["effects"].append(
                "Wizard's Staff discarded - Shadow draws no Hunt tile")
            return result
        # declined: resume the draw (bypassing the Wizard's Staff guard).
        # 1.217: a hunt-DRAWING card (Isildur's Bane / Foul Thing from the
        # Deep) stashes its apply_kind + play choices - re-enter THAT
        # handler with _skip_wizard=True so it proceeds to the draw. The
        # normal hunt path stashes successes/mordor and resumes
        # _resolve_hunt_tile (mirrors _effect_mithril_coat's apply_kind
        # routing).
        _ak9 = stash.get('apply_kind')
        if _ak9 == 'stronghold':
            _gl9._draw_shadow_stronghold_tile(
                state, stash['region'], stash['reason'], _skip_wizard=True, _resume=True)
            result['effects'].append("Wizard's Staff kept - stronghold Hunt draw resumes")
            return result
        if _ak9 in ('orc_patrol', 'foul_thing', 'isildur', 'breaking'):
            _hmap9 = {'orc_patrol': _effect_orc_patrol,
                      'foul_thing': _effect_foul_thing,
                      'isildur': _effect_isildurs_bane,
                      'breaking': _effect_breaking_of_fellowship}
            _ch9 = dict(stash.get('choices') or {})
            _ch9['_skip_wizard'] = True
            _r9 = _hmap9[_ak9](state, None, _ch9)
            for _e9 in (_r9.get('effects') or []):
                result["effects"].append(_e9)
            result["effects"].append("Wizard's Staff kept - Hunt tile drawn")
            return result
        if stash.get('mordor_track'):
            # 1.232: Mordor-track decline resumes the track draw; the
            # caller already banked the step, so the stop retraction is
            # replicated here (the Mithril mordor resume's pattern)
            r9 = _gl9._resolve_mordor_hunt_tile(
                state,
                skip_auto_sacrifice=stash.get('skip_auto_sacrifice',
                                              True),
                _skip_wizard=True)
            if r9.get('stops_movement'):
                state.fellowship.mordor_step = max(
                    0, state.fellowship.mordor_step - 1)
                result["effects"].append(
                    "Stop tile - the Mordor step is retracted")
            result["effects"].append(
                "Wizard's Staff kept - Hunt tile drawn")
            return result
        _gl9._resolve_hunt_tile(
            state, stash['successes'], mordor=stash['mordor'],
            _skip_wizard=True,
            eye_no_effect=stash.get('eye_no_effect', False))
        result["effects"].append("Wizard's Staff kept - Hunt tile drawn")
        return result
    result = {"effects": []}
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Wizard's Staff now in play - can prevent Hunt tile draw")
    return result


@register_effect("Athelas")
def _effect_athelas(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice, heal corruption on 5+ (3+ if Strider is guide)."""
    result = {"effects": []}
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""
    threshold = 3 if "Strider" in guide_name else 5

    rolls = _d6_batch(3)
    _ann_rng(state, 'Athelas roll', faces=rolls)
    heals = sum(1 for r in rolls if r >= threshold)

    state.fellowship.corruption = max(0, state.fellowship.corruption - heals)
    result["effects"].append(f"Rolled {rolls}, healed {heals} Corruption (threshold {threshold}+)")
    return result


@register_effect("There is Another Way")
def _effect_there_is_another_way(state: "GameState", card, choices: dict) -> dict:
    """USER RE-RULING 2026-08-24 (SUPERSEDES 2026-07-21f): the heal
    ALWAYS applies (1 Corruption), and with Gollum as Guide the FP may
    ALSO move the Fellowship with the same card - heal + move
    together, NOT either/or.  The card move places NO die in the Hunt
    Box (the die->box step lives only in the fellowship_move
    executor; the USER pinned this explicitly).  Movement still
    requires hidden (rulebook p.38): a revealed Fellowship gets the
    heal only ('heal kept, illegal move skipped')."""
    result = {"effects": []}
    state.fellowship.corruption = max(
        0, state.fellowship.corruption - 1)
    result["effects"].append("Healed 1 Corruption")
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""
    gollum = "Gollum" in guide_name
    _act9 = str((choices or {}).get('action', ''))
    if gollum and _act9 == 'move':
        if not state.fellowship.hidden:
            result["effects"].append(
                "Gollum move skipped: Fellowship revealed "
                "(heal kept)")
            return result
        import game_logic as _gl9
        _mv9 = _gl9.move_fellowship(state, skip_hunt=False,
                                    skip_auto_sacrifice=True)
        result["effects"].append(
            "Fellowship moved (Gollum guide, no die to the Hunt "
            "Box): " + str(_mv9.get('message', '')))
    elif gollum and _act9 == 'hide':
        # USER RULING 2026-08-25 (verbatim: "move or hide depending
        # on whether fellowship is revealed") - the revealed
        # fellowship may HIDE with the card instead
        if state.fellowship.hidden:
            result["effects"].append(
                "Gollum hide skipped: already hidden (heal kept)")
            return result
        state.fellowship.hidden = True
        state.fellowship_acted_in_mordor = True
        result["effects"].append(
            "Fellowship hidden (Gollum guide, There is Another Way)")
    return result


@register_effect("I Will Go Alone")
def _effect_i_will_go_alone(state: "GameState", card, choices: dict) -> dict:
    """Separate 1+ companions to ONE region up to progress + highest
    level of the group + 1 extra region; heal 1 corruption (USER RULE
    2026-07-08; extra=1 matches the card_options spec)."""
    from game_state import Army, Companion
    result = {"effects": []}

    companions_to_separate = choices.get("companions", [])
    destination = choices.get("destination", "")

    # LOW-8: with no companion/destination and a separable target the
    # separation cannot happen, so neither does its trailing heal - refund; a
    # Gollum-only Fellowship has nothing to separate and heals only.
    if (
            not companions_to_separate or not destination):
        if any(c.name != "Gollum" and c.level > 0
               for c in state.fellowship.companions):
            result["message"] = ("I Will Go Alone: no companion/"
                                 "destination chosen - not played")
            return _selfplay_refund(result)
        # no legal separation target -> historical heal-only

    if companions_to_separate and destination == '__remove_from_play__' \
            and state.fellowship.mordor_entered:
        # almanac (QA 2026-07-11 cluster B): Mordor-Track separation
        # removes the companions from play; the heal still applies
        _remove_companions_from_play(state, companions_to_separate,
                                     result)
    elif companions_to_separate and destination:
        dest_san = sanitize(destination)
        picked = [c for c in state.fellowship.companions
                  if c.name in companions_to_separate]
        # Range check: the whole group moves together up to
        # progress + the HIGHEST level among the separating companions
        import game_logic as _gl
        from ai import _companion_reachable
        cur = sanitize(_gl.fellowship_region(state))
        rng = state.fellowship.progress + (
            max((c.level for c in picked), default=0)) + 1  # +1 EXTRA region
        if dest_san != cur and dest_san not in _companion_reachable(
                state, cur, max(0, rng)):
            result["effects"].append(
                f"{destination} out of range (max {rng}) - "
                "not played")
            return _selfplay_refund(result)

        separated_names = []
        for comp in list(state.fellowship.companions):
            if comp.name in companions_to_separate:
                state.fellowship.companions.remove(comp)
                # 1.149b: place via the standard helper so the arrival
                # activates its nation (p.34 4th trigger)
                _gl.place_separated_companion(state, comp, dest_san)
                separated_names.append(comp.name)

        if separated_names:
            result["effects"].append(f"Separated {', '.join(separated_names)} to {destination}")

            # a separated Guide re-picks by HIGHEST Level; a Level tie
            # is the learned pending_guide_change (CL-219) - never a
            # swallowed RuntimeError leaving a stale guide
            from game_logic import change_guide
            change_guide(state, allow_pending=True)
            _gl.check_companion_persistent_discards(state)

    # Heal 1 corruption
    state.fellowship.corruption = max(0, state.fellowship.corruption - 1)
    result["effects"].append("Healed 1 Corruption")
    return result


@register_effect("Bilbo's Song")
def _effect_bilbos_song(state: "GameState", card, choices: dict) -> dict:
    """Heal 1 corruption (2 if Gollum is guide)."""
    result = {"effects": []}
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""
    heal_amount = 2 if "Gollum" in guide_name else 1
    state.fellowship.corruption = max(0, state.fellowship.corruption - heal_amount)
    result["effects"].append(f"Healed {heal_amount} Corruption")
    return result


@register_effect("Mirror of Galadriel")
def _effect_mirror_of_galadriel(state: "GameState", card, choices: dict) -> dict:
    """Change Character die to Will of the West. Heal 1 if in Lórien."""
    result = {"effects": []}
    from game_logic import fellowship_region
    from game_state import ActionDieFace
    # Perform the die change HERE.  SELF-PLAY ONLY: the teacher's schedule
    # applies the logged die_change records itself.
    _changed = False
    for _d in state.dice.get(Side.FREE_PEOPLES, []):
        if not _d.used and _d.current == ActionDieFace.CHARACTER:
            _d.current = ActionDieFace.WILL_OF_WEST
            _changed = True
            break
    if _changed:
        result["effects"].append("Changed Character die to Will of the West")
    else:
        result["message"] = "no unused Character die to change"

    fsp_region = fellowship_region(state)
    if sanitize(fsp_region) == "lorien":
        # almanac (QA 2026-07-11 cluster J): 'not if Lorien is
        # captured' - no heal at a Shadow-controlled Lorien
        if 'lorien' in state.enemy_controlled.get(Side.SHADOW, set()):
            result["effects"].append(
                "Lorien is captured - no Corruption heal")
        else:
            state.fellowship.corruption = max(
                0, state.fellowship.corruption - 1)
            result["effects"].append(
                "Fellowship in Lórien - healed 1 Corruption")
    return result


@register_effect("Challenge of the King")
def _effect_challenge_of_king(state: "GameState", card, choices: dict) -> dict:
    """Draw 3 hunt tiles - eliminate Strider if all Eyes, else remove Eye tiles."""
    result = {"effects": []}
    # Draw 3 DISTINCT tiles; an emptied pool is refilled by the ONE recycler
    # (1.136; pre-Mordor special tiles stay parked in the used pile).
    import game_logic as _glc9
    tiles = []
    for _ in range(3):
        if not state.hunt_pool and _glc9._recycle_hunt_pool(state):
            result["effects"].append(
                "Hunt Pool rebuilt from the used pile mid-draw")
        if state.hunt_pool:
            tile = _tile_choice(state)
            state.hunt_pool.remove(tile)
            tiles.append(tile)

    eye_count = sum(1 for t in tiles if getattr(t, 'is_eye', False))

    if eye_count == 3:
        # All Eyes: the tiles return to the pool and Strider/Aragorn is
        # eliminated - he is SEPARATED by the play requirement, so search the
        # BOARD as well as the Fellowship - through
        # combat.book_eliminated_characters (companion-casualty hooks),
        # never a bare leaders.remove.
        result["effects"].append("All 3 tiles show Eyes - Strider/Aragorn eliminated!")
        from combat import book_eliminated_characters as _bec9
        _dead9 = None
        for c in state.fellowship.companions[:]:
            if "Strider" in c.name or "Aragorn" in c.name:
                state.fellowship.companions.remove(c)
                _dead9 = c
                _glc9.change_guide(state, allow_pending=True)
                break
        if _dead9 is None:
            from game_state import Companion as _CompC
            for _rk9, _a9 in list(state.armies.items()):
                for l in list(_a9.leaders):
                    if isinstance(l, _CompC) and (
                            "Strider" in l.name or "Aragorn" in l.name):
                        _a9.leaders.remove(l)
                        if not _a9.units and not _a9.leaders:
                            state.armies.pop(_rk9, None)
                        _dead9 = l
                        break
                if _dead9 is not None:
                    break
            _glc9.remove_orphaned_leaders(state)
        if _dead9 is not None:
            _bec9(state, [_dead9])
        state.hunt_pool.extend(tiles)
    else:
        # USER RULE 2026-07-08: Challenge is the ONE card that removes Eyes
        # PERMANENTLY (hunt_removed); non-Eye tiles return to the pool.
        discarded = 0
        for t in tiles:
            if getattr(t, 'is_eye', False):
                state.hunt_removed.append(t)
                discarded += 1
            else:
                state.hunt_pool.append(t)
        if discarded > 0:
            result["effects"].append(
                f"Drew {len(tiles)} tiles, permanently removed "
                f"{discarded} Eye tile(s)")
        else:
            result["message"] = (
                f"Drew {len(tiles)} tiles, no Eye tiles to remove")

    random.shuffle(state.hunt_pool)
    return result


def _group_separate(state, names, dest_san, rng, result,
                    allow_besieged=True):
    """1.149c-class GROUP separation for Gwaihir / We Prove Swifter:
    separate every named Fellowship companion together to dest_san
    within `rng`, activating each arrival nation, then re-pick the guide
    ONCE after the whole batch (mirrors There and Back Again).  Refunds
    (self-play) when a name is not in the Fellowship or dest is out of
    range."""
    import game_logic as _gl
    names = [str(n) for n in (names or [])]
    picked = [c for c in state.fellowship.companions if c.name in names]
    if not names or len(picked) != len(names):
        result["message"] = "Group separation: companion not in Fellowship"
        return _selfplay_refund(result)
    if not _separation_dest_ok(state, dest_san, rng,
                               allow_besieged=allow_besieged):
        result["message"] = "Group destination out of range"
        return _selfplay_refund(result)
    sep_names = []
    for comp in list(state.fellowship.companions):
        if comp.name in names:
            state.fellowship.companions.remove(comp)
            _gl.place_separated_companion(state, comp, dest_san)
            sep_names.append(comp.name)
    if sep_names:
        result["effects"].append(
            f"Separated {', '.join(sep_names)} to {dest_san} (group)")
        _gl.change_guide(state, allow_pending=True)
        _gl.check_companion_persistent_discards(state)
    return result


@register_effect("Gwaihir the Windlord")
def _effect_gwaihir(state: "GameState", card, choices: dict) -> dict:
    """Move/separate companions as if level 4, can end in besieged stronghold."""
    from game_state import Companion, Army
    from game_logic import place_separated_companion
    result = {"effects": []}

    action = choices.get("companion_action", "")
    destination = choices.get("destination", "")

    # TEACHER vocabulary: a bare companions+destination choice set IS the
    # logged group separation (GAME19 T2); self-play always supplies
    # companion_action.
    if not action and destination and choices.get('companions'):
        action = 'group'

    if not action or not destination:
        result["message"] = "No companion action selected"  # no-op: NOT an effect (iter 39)
        return _selfplay_refund(result)   # LOW-6: no-op must not consume

    if destination == '__remove_from_play__' \
            and state.fellowship.mordor_entered \
            and (action == 'group' or action.startswith('sep_')):
        names = (choices.get('companions') or []
                 if action == 'group' else [action[4:]])
        _remove_companions_from_play(state, names, result)
        return result

    dest_san = sanitize(destination)
    if action == 'group':
        # 1.149c: a whole group flies together as if Level 4
        return _group_separate(state, choices.get('companions'),
                               dest_san, state.fellowship.progress + 4,
                               result, allow_besieged=True)
    if action.startswith('sep_'):
        # separate as if Level 4: range = progress + 4, may end in a
        # besieged stronghold
        if not _separation_dest_ok(state, dest_san,
                                   state.fellowship.progress + 4,
                                   allow_besieged=True):
            result["message"] = f"{destination} out of range (Level 4)"
            return _selfplay_refund(result)   # LOW-6
    elif action == 'move_group' or action.startswith('move_'):
        # QA 2026-07-11 (Gwaihir CASE 8): a previously separated
        # companion (or a co-located GROUP - card: "one Companion or one
        # group of Companions") moves up to 4 regions FROM ITS OWN
        # REGION; ONE body for the single and the group arm.
        _names9 = (list(choices.get('companions') or [])
                   if action == 'move_group' else [action[5:]])
        return _card_move_board_group(state, _names9, dest_san, 4, result,
                                      '(as Level 4)')

    if action.startswith("sep_"):
        # Separating from fellowship
        comp_name = action[4:]  # Remove "sep_"
        for comp in list(state.fellowship.companions):
            if comp.name == comp_name:
                state.fellowship.companions.remove(comp)
                # 1.149b: place via the standard helper so the arrival
                # activates its nation (_check_companion_activates_nation)
                place_separated_companion(state, comp, dest_san)
                result["effects"].append(f"Separated {comp_name} to {destination} (as Level 4)")

                # a separated Guide re-picks by HIGHEST Level; a Level
                # tie is the learned pending_guide_change (CL-219)
                from game_logic import change_guide
                change_guide(state, allow_pending=True)
                from game_logic import check_companion_persistent_discards
                check_companion_persistent_discards(state)
                break

    return result


def _card_move_board_group(state, names, dest_san, rng, result, label):
    """Card-granted board move of one or more CO-LOCATED separated
    companions (Gwaihir / We Prove Swifter 'move' arm): validated with
    _dest_ok_from from the group's own region (besieged end allowed),
    placed via place_separated_companion, and the emptied source entry
    is dropped (never an empty Army entry left behind)."""
    from game_state import Companion as _CompG
    from game_logic import place_separated_companion
    names = [str(n) for n in (names or [])]
    _cur9, _key9 = None, None
    for _rk9, _a9 in state.armies.items():
        _here9 = {l.name for l in _a9.leaders if isinstance(l, _CompG)}
        if names and set(names) <= _here9:
            _cur9, _key9 = sanitize(_rk9), _rk9
            break
    if _cur9 is None:
        result["message"] = "Companion group not co-located on the board"
        return _selfplay_refund(result)
    if not _dest_ok_from(state, _cur9, dest_san, rng, allow_besieged=True):
        result["message"] = f"{dest_san} out of range {label}"
        return _selfplay_refund(result)   # LOW-6 / LOW-20
    army = state.armies[_key9]
    movers = [l for l in list(army.leaders)
              if isinstance(l, _CompG) and l.name in names]
    for l in movers:
        army.leaders.remove(l)
    if not army.units and not army.leaders:
        state.armies.pop(_key9, None)
    for l in movers:
        place_separated_companion(state, l, dest_san)
    result["effects"].append(
        f"Moved {', '.join(l.name for l in movers)} to {dest_san} {label}")
    return result


@register_effect("We Prove Swifter")
def _effect_we_prove_swifter(state: "GameState", card, choices: dict) -> dict:
    """Move/separate companions +2 regions, can end in besieged stronghold."""
    from game_state import Companion, Army
    from game_logic import place_separated_companion
    result = {"effects": []}

    action = choices.get("companion_action", "")
    destination = choices.get("destination", "")

    if not action or not destination:
        result["message"] = "No companion action selected"  # no-op: NOT an effect (iter 39)
        return _selfplay_refund(result)   # LOW-20: no-op must not consume

    if destination == '__remove_from_play__' \
            and state.fellowship.mordor_entered \
            and (action == 'group' or action.startswith('sep_')):
        names = (choices.get('companions') or []
                 if action == 'group' else [action[4:]])
        _remove_companions_from_play(state, names, result)
        return result

    dest_san = sanitize(destination)
    if action == 'group':
        # 1.149c: a whole group flies together; range from the GROUP's
        # own max level + 2 (independent of Fellowship members)
        _names_g = [str(n) for n in (choices.get('companions') or [])]
        _lvls_g = [c.level for c in state.fellowship.companions
                   if c.name in _names_g]
        _rng_g = state.fellowship.progress + (
            max(_lvls_g) if _lvls_g else 0) + 2
        return _group_separate(state, _names_g, dest_san, _rng_g,
                               result, allow_besieged=True)
    if action.startswith('sep_'):
        # separate up to progress + the SEPARATING companion's OWN level
        # + 2 extra regions (1.211/CL-255: NOT the Fellowship max - a
        # lone L1 hobbit does not borrow Gandalf L3's range); may end in
        # a besieged stronghold
        _sep_comp = next((c for c in state.fellowship.companions
                          if c.name == action[4:]), None)
        _own_lvl = _sep_comp.level if _sep_comp is not None else 0
        _rng = state.fellowship.progress + _own_lvl + 2
        if not _separation_dest_ok(state, dest_san, _rng,
                                   allow_besieged=True):
            result["message"] = f"{destination} out of range (+2)"
            return _selfplay_refund(result)   # LOW-20
    elif action == 'move_group' or action.startswith('move_'):
        # QA 2026-07-11 (almanac C16): a previously separated companion
        # (or a co-located GROUP - card: "one Companion or one group of
        # Companions") moves up to ITS level (+ the group's max level)
        # + 2 regions FROM ITS OWN REGION; ONE body for both arms.
        from game_state import Companion as _CompW
        _names9 = (list(choices.get('companions') or [])
                   if action == 'move_group' else [action[5:]])
        from ai import _companion_group_movement
        _lvl9 = _companion_group_movement([
            l for a in state.armies.values() for l in a.leaders
            if isinstance(l, _CompW) and l.name in _names9])
        return _card_move_board_group(state, _names9, dest_san, _lvl9 + 2,
                                      result, '(+2 regions)')

    if action.startswith("sep_"):
        # Separating from fellowship
        comp_name = action[4:]  # Remove "sep_"
        for comp in list(state.fellowship.companions):
            if comp.name == comp_name:
                state.fellowship.companions.remove(comp)
                # 1.149b: arrival activates its nation via the helper
                place_separated_companion(state, comp, dest_san)
                result["effects"].append(f"Separated {comp_name} to {destination} (+2 regions)")

                # a separated Guide re-picks by HIGHEST Level; a Level
                # tie is the learned pending_guide_change (CL-219)
                from game_logic import change_guide
                change_guide(state, allow_pending=True)
                from game_logic import check_companion_persistent_discards
                check_companion_persistent_discards(state)
                break

    return result


@register_effect("There and Back Again")
def _effect_there_and_back_again(state: "GameState", card, choices: dict) -> dict:
    """Separate companions +1 region. If Gimli/Legolas in Dale/Erebor/Woodland, activate nations."""
    from game_state import Companion, Army
    result = {"effects": []}

    companions_to_separate = choices.get("companions", [])
    destination = choices.get("destination", "")

    # BL-90 (2026-08-25): the printed "Then, if Gimli or Legolas ARE in Dale,
    # Erebor or the Woodland Realm ..." clause is a BOARD test after the
    # separation, reached by BOTH arms (a Mordor-track separation must not
    # drop it); the Mordor arm leaves dest_san/separated_names empty.
    dest_san = ''
    separated_names = []

    if companions_to_separate and (
            destination == '__remove_from_play__'
            or state.fellowship.mordor_entered):
        # almanac (QA 2026-07-11 cluster B): on the Mordor Track the
        # separated companions are REMOVED FROM PLAY - the old path
        # bypassed separate_companions and placed them on a board army
        _remove_companions_from_play(state, companions_to_separate,
                                     result)
    elif companions_to_separate and destination:
        dest_san = sanitize(destination)
        # There and Back Again: move the separated group up to progress +
        # highest level among them + 1 extra region (USER FIX 2026-07-08).
        _picked = [c for c in state.fellowship.companions
                   if c.name in companions_to_separate]
        _rng = state.fellowship.progress + (
            max((c.level for c in _picked), default=0)) + 1
        if not _separation_dest_ok(state, dest_san, _rng):
            result["message"] = f"{destination} out of range (+1)"
            return _selfplay_refund(result)   # LOW-18: no-op must not consume

        for comp in list(state.fellowship.companions):
            if comp.name in companions_to_separate:
                state.fellowship.companions.remove(comp)
                # 1.149b: place via the standard helper so the arrival
                # activates its nation (p.34 4th trigger)
                from game_logic import place_separated_companion
                place_separated_companion(state, comp, dest_san)
                separated_names.append(comp.name)

        if separated_names:
            result["effects"].append(f"Separated {', '.join(separated_names)} to {destination}")

            # a separated Guide re-picks by HIGHEST Level; a Level tie
            # is the learned pending_guide_change (CL-219)
            from game_logic import change_guide
            change_guide(state, allow_pending=True)
            from game_logic import check_companion_persistent_discards
            check_companion_persistent_discards(state)

    elif not (getattr(state.fellowship.guide, 'name', None) == 'Gollum'
              and not any(c.name != 'Gollum' and c.level > 0
                          for c in state.fellowship.companions)):
        result["message"] = "No companions separated"
        return _selfplay_refund(result)

    # Almanac14: this board test also applies when Gollum needs no separation.
    # Check for Gimli/Legolas in Dale/Erebor/Woodland Realm
    trigger_regions = {"dale", "erebor", "woodland realm", "woodlandrealm", "woodland_realm"}
    gimli_legolas_in_trigger = False

    # almanac p32 (QA 2026-07-11 cluster L): a CAPTURED region
    # does not activate/advance - exclude Shadow-controlled regions
    _cap_sh9 = state.enemy_controlled.get(Side.SHADOW, set())
    # Check if Gimli or Legolas just went to a trigger region
    if dest_san in trigger_regions and dest_san not in _cap_sh9 \
            and any(n in ("Gimli", "Legolas")
                    for n in separated_names):
        gimli_legolas_in_trigger = True

    # Also check if Gimli/Legolas were already in a trigger region
    for region_key, army in state.armies.items():
        _rs9 = sanitize(region_key)
        if _rs9 in trigger_regions and _rs9 not in _cap_sh9:
            for leader in army.leaders:
                if isinstance(leader, Companion) and leader.name in ("Gimli", "Legolas"):
                    gimli_legolas_in_trigger = True
                    break

    if gimli_legolas_in_trigger:
        # Card text: activate the Dwarven and North Nations and advance the
        # Dwarven, Elven and North Nations one step.  USER RULING (2026-07,
        # revised): this card DOES activate; activate BEFORE advancing so the
        # passive clamp does not swallow the step.
        for nation in [Nation.DWARVES, Nation.THE_NORTH]:
            activate_nation(state, nation)
        for nation in [Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH]:
            advance_nation(state, nation, 1)
        result["effects"].append("Gimli/Legolas in Dale/Erebor/Woodland - activated Dwarves+North, advanced Dwarves, Elves, North!")

    return result


@register_effect("The Eagles are Coming")
def _effect_eagles_are_coming(state: "GameState", card, choices: dict) -> dict:
    """Roll vs Nazgul (5+), eliminate and move survivors to a Sauron
    stronghold (Shadow picks the destination via a chained stage)."""
    if choices and choices.get('eagles_dest'):
        # chained stage: the SHADOW player picked the survivors' dest
        from game_state import Nazgul as _NazD, Army as _ArmyD
        result = {"effects": []}
        _frm9 = sanitize(str(choices.get('_eagles_from', '')))
        _dst9 = sanitize(str(choices['eagles_dest']))
        src_key = _find_army_key(state, _frm9)
        src = state.armies.get(src_key) if src_key else None
        if src is not None:
            _mv9 = [l for l in src.leaders if isinstance(l, _NazD)]
            for l in _mv9:
                src.leaders.remove(l)
            if not src.units and not src.leaders:
                state.armies.pop(src_key, None)
            if _mv9:
                dk9 = _find_army_key(state, _dst9)
                if dk9 is None or dk9 not in state.armies:
                    state.armies[_dst9] = _ArmyD(
                        region=state.board.region(_dst9))
                    dk9 = _dst9
                state.armies[dk9].leaders.extend(_mv9)
                result["effects"].append(
                    f"{len(_mv9)} surviving Nazgul moved to {_dst9} "
                    f"(Shadow's choice)")
        return result
    result = {"effects": []}

    # Target: the Shadow army (with a Nazgul) in contact with an FP army
    # containing a Companion (the card requirement).
    from game_state import Nazgul as _Naz, Companion as _Comp, Army as _Army

    def _fp_with_companion(a):
        return (a.units and _is_army_fp(a)
                and any(isinstance(l, _Comp) for l in a.leaders))

    target = None
    _want_t9 = sanitize(str(choices.get('target_region', '') or ''))
    if _want_t9:
        _tk9 = _find_army_key(state, _want_t9)
        if _tk9 and _tk9 in state.armies:
            target = (_tk9, state.armies[_tk9])
    for rk, a in state.armies.items():
        if target is not None:
            break
        if not _army_contains_nazgul(a):
            continue
        rs = sanitize(rk)
        near = [rs] + [sanitize(x) for x in state.board.adjacent(rs)]
        for r in near:
            other = next((aa for kk, aa in state.armies.items()
                          if sanitize(kk) == r), None)
            if other is not None and _fp_with_companion(other):
                target = (rk, a)
                break
        if target:
            break
    if target is None:
        # fall back to any army holding regular Nazgûl
        for rk, a in state.armies.items():
            if any(isinstance(l, _Naz) for l in a.leaders):
                target = (rk, a)
                break
    if target is None:
        result["effects"].append("No Nazgûl to target")
        return result

    rk, army = target
    nazgul = [l for l in army.leaders if isinstance(l, _Naz)]
    dice_count = min(5, len(nazgul))
    rolls = _d6_batch(dice_count)
    _ann_rng(state, 'The Eagles are Coming roll', faces=rolls)
    kills = sum(1 for r in rolls if r >= 5)
    for _ in range(min(kills, len(nazgul))):
        army.leaders.remove(nazgul.pop())
    result["effects"].append(f"Rolled {rolls}, eliminated {kills} Nazgûl")

    # Survivors "must all be moved together (by the SHADOW player) to another
    # uncaptured Sauron Stronghold" - a learned cross-side pick on the
    # pending path.
    survivors = [l for l in army.leaders if isinstance(l, _Naz)]
    if survivors:
        _cap_fp9 = state.enemy_controlled.get(Side.FREE_PEOPLES, set())
        cands9 = []
        for k, m in board.regions.items():
            if not m.get('is_stronghold'):
                continue
            ks9 = sanitize(k)
            if _get_region_nation(ks9) != 'sauron':
                continue  # Orthanc/Umbar/FP strongholds all invalid
            if ks9 in _cap_fp9:
                continue  # captured Sauron strongholds invalid
            if ks9 == sanitize(rk):
                # must be a DIFFERENT stronghold when the target is itself a
                # Shadow stronghold (Eagles LOW-11 adjudication 2026-07-21)
                continue
            cands9.append(ks9)
        cands9 = sorted(set(cands9))
        if not cands9:
            result["effects"].append(
                "No uncaptured Sauron Stronghold available - "
                "survivors stay")
        elif choices.get('_from_pending') and len(cands9) > 1:
            state.pending_card_option = {
                'card_id': getattr(card, 'card_id',
                                   'The Eagles are Coming'),
                'card_name': 'The Eagles are Coming',
                'side': 'fp', 'decider': 'sh', 'direct': True,
                'options': [{'eagles_dest': c9,
                             '_eagles_from': sanitize(rk)}
                            for c9 in cands9]}
            result["effects"].append(
                f"{len(survivors)} surviving Nazgul must relocate - "
                f"Shadow picks the Sauron Stronghold (pending)")
            return result
        else:
            dest = cands9[0]
            for l in survivors:
                army.leaders.remove(l)
            if dest not in state.armies:
                state.armies[dest] = _Army(region=state.board.region(dest))
            state.armies[dest].leaders.extend(survivors)
            result["effects"].append(
                f"{len(survivors)} surviving Nazgûl moved to {dest}")
    if not army.units and not army.leaders:
        state.armies.pop(rk, None)
    return result


@register_effect("The Ents Awake: Treebeard")
def _effect_ents_awake_treebeard(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice, hit Orthanc army on 4+. May chain another Character Event."""
    return _effect_ents_awake_common(state, card, choices)


@register_effect("The Ents Awake: Huorns")
def _effect_ents_awake_huorns(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice, hit Orthanc army on 4+. May chain another Character Event."""
    return _effect_ents_awake_common(state, card, choices)


@register_effect("The Ents Awake: Entmoot")
def _effect_ents_awake_entmoot(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice, hit Orthanc army on 4+. May chain another Character Event."""
    return _effect_ents_awake_common(state, card, choices)


def _effect_ents_awake_common(state: "GameState", card, choices: dict) -> dict:
    """Common logic for all Ents Awake cards (QA 2026-07-11 cluster D:
    side-filtered hits, real army key, Nazgul/Minion elimination with a
    destroyed army, siege capture, political attack, Saruman-only when
    no Shadow Army, and the learned free-chain)."""
    from combat import side_of_nation as _son
    import game_logic as _gl9
    result = {"effects": []}

    # chained free-play stages ------------------------------------------------
    if choices.get('_ents_chain_done'):
        state._ents_chain_n = 0
        result["effects"].append("Ents Awake: no further free card")
        return result
    if choices.get('ents_chain_play'):
        _cid9 = choices['ents_chain_play']
        import ai as _ai9
        _ai9._execute_action(state, _ai9.Action(
            die_idx=-1, action_type='play_event',
            params={'card_id': _cid9, 'side': 'fp'}))
        # honest reporting: a nested play that did not resolve leaves
        # the card in hand - never claim it was played
        if _cid9 in state.hands[Side.FREE_PEOPLES].hand:
            result["effects"].append(
                f"Ents Awake chain: {_cid9} could not be played")
        else:
            result["effects"].append(
                f"Ents Awake chain: played {_cid9} without a die")
        return result

    # USER RULING 1.254: the hits are the Shadow OWNER's learned
    # allocation; '_ents_cas_resumed' re-enters after that pending with
    # the roll already spent (continuation via _complete_card_casualty).
    _resumed9 = bool(choices.get('_ents_cas_resumed'))
    hits = 0
    if not _resumed9:
        rolls = _d6_batch(3)
        _ann_rng(state, 'The Ents Awake: Entmoot roll', faces=rolls)
        hits = sum(1 for r in rolls if r >= 4)
        result["effects"].append(
            f"Rolled {rolls}, {hits} hits against Orthanc")

    _okey9 = _find_army_key(state, "orthanc")
    orthanc_army = state.armies.get(_okey9) if _okey9 else None
    _sh_units9 = ([u for u in orthanc_army.units
                   if u.nation in SHADOW_NATIONS]
                  if orthanc_army else [])

    def _eliminate_saruman():
        # Guard on Saruman's ACTUAL figure at Orthanc (LOW-14): a phantom
        # elimination permanently corrupted the in-play state.
        if orthanc_army is None:
            return
        _sar9 = [l for l in list(orthanc_army.leaders)
                 if _gl9._is_saruman(l)]
        if not _sar9:
            return
        for l in _sar9:
            orthanc_army.leaders.remove(l)
        # Route through combat._handle_eliminated_minions (in_play flags + the
        # Saruman table-card discards).  SELF-PLAY ONLY: the VASSAL app kept
        # Saruman's die after an Ents kill and the teacher replays that
        # (same sanctioned app-vs-rulebook class as 1.188).
        import combat as _cmb_ents9
        if _cmb_ents9.FINAL_HITS_HOOK is None:
            _cmb_ents9._handle_eliminated_minions(state, _sar9)
        result["effects"].append("Saruman eliminated")

    if _sh_units9 or _resumed9:
        if not _resumed9:
            # a Shadow Army is present: the play is an ATTACK for the
            # Political Track (almanac)
            _gl9.advance_defenders_political(
                state, "orthanc", attacker_side=Side.FREE_PEOPLES)
        if hits > 0:
            import combat as _cmb_ents7
            if _cmb_ents7.FINAL_HITS_HOOK is None:
                # 1.254: owner-allocated; the teacher keeps its
                # certified logged-composition path below
                if not _route_card_casualties(
                        state, 'orthanc', Side.SHADOW, hits,
                        card or _card_by_name('The Ents Awake: Treebeard'),
                        choices, result):
                    return result
                _okey9 = _find_army_key(state, "orthanc")
                orthanc_army = (state.armies.get(_okey9)
                                if _okey9 else None)
            else:
                casualties = _apply_card_hits_to_shadow(
                    state, orthanc_army, hits)
                result["effects"].append(
                    f"{len(casualties)} Shadow casualties in Orthanc")
        _sh_left9 = orthanc_army is not None and any(
            u.nation in SHADOW_NATIONS for u in orthanc_army.units)
        if not _sh_left9 and orthanc_army is None:
            # learned path: the ONE reconciler (shared casualty
            # aftermath) already booked the wipe and any besieger capture
            result["effects"].append("Orthanc Shadow army destroyed")
        elif not _sh_left9:
            # USER RULING 2026-08-15: the ONE classifier decides which figures
            # die (The Ents Awake is not a sparing card); bookkeeping keeps
            # the teacher gate below.
            import combat as _cmb_ents8
            _gone_objs9, _stay_objs9 = _cmb_ents8.classify_wipe_characters(
                orthanc_army.leaders, Side.SHADOW,
                card_name='The Ents Awake')
            orthanc_army.leaders = _stay_objs9
            _gone9 = [getattr(l, 'name', type(l).__name__)
                      for l in _gone_objs9]
            # clear the CANONICAL in_play flags + Saruman table-card discards
            # via the combat helper; self-play only (teacher replays the app's
            # historical die/table state)
            if _gone_objs9 and _cmb_ents8.FINAL_HITS_HOOK is None:
                _cmb_ents8.book_eliminated_characters(state, _gone_objs9)
            result["effects"].append(
                "Orthanc Shadow army destroyed"
                + (f" (with {', '.join(_gone9)})" if _gone9 else ""))
            if not orthanc_army.units and not orthanc_army.leaders:
                state.armies.pop(_okey9, None)
            _bs9 = getattr(state, 'besieged_strongholds', set())
            if 'orthanc' in _bs9 and orthanc_army is not None \
                    and any(_son(u.nation) == Side.FREE_PEOPLES
                            for u in orthanc_army.units):
                # FP besiegers capture the emptied stronghold at once
                _bs9.discard('orthanc')
                from combat import _check_settlement_capture
                _check_settlement_capture(state, 'orthanc',
                                          Side.FREE_PEOPLES)
                result["effects"].append(
                    "Free Peoples besiegers capture Orthanc")
    else:
        # no Shadow Army: eliminate ONLY Saruman; any Nazgul/Minions
        # present are unaffected (almanac)
        _eliminate_saruman()

    # free-chain (almanac): with Gandalf the White in Fangorn or a
    # Rohan region, another FP Character Event may be played without a
    # die; the chain continues through Ents cards, 4 plays total
    if choices.get('_from_pending'):
        _n9 = int(getattr(state, '_ents_chain_n', 0) or 0) + 1
        state._ents_chain_n = _n9
        _gtw_here9 = False
        for _k9, _a9 in state.armies.items():
            if any(getattr(l, 'name', '') == 'Gandalf the White'
                   for l in _a9.leaders):
                _ks9 = sanitize(_k9)
                _meta9 = board.regions.get(_ks9, {})
                if _ks9 == 'fangorn' or board.fid_to_name.get(
                        _meta9.get('parent_id')) == 'rohan':
                    _gtw_here9 = True
                break
        if _gtw_here9 and _n9 < 4:
            _hand9 = state.hands[Side.FREE_PEOPLES].hand
            # "play ANOTHER Character Event card" (LOW-13/15/17): exclude the
            # card being resolved by card_id (a different Ents copy still
            # chains)
            _res_cid9 = getattr(card, 'card_id', None)
            # a 'play' still carries the card's own play condition:
            # only cards whose requirement holds are legal chain picks
            _opts9 = [{'ents_chain_play': c9}
                      for c9 in _hand9
                      if isinstance(c9, str) and '_Character_' in c9
                      and c9 != _res_cid9
                      and check_event_requirement(state, c9)[0]]
            if _opts9:
                _opts9.append({'_ents_chain_done': True})
                state.pending_card_option = {
                    'card_id': getattr(card, 'card_id',
                                       'The Ents Awake: Treebeard'),
                    'card_name': getattr(card, 'card_name',
                                         'The Ents Awake: Treebeard'),
                    'side': 'fp', 'decider': 'fp', 'direct': True,
                    'options': _opts9}
                result["effects"].append(
                    "Gandalf the White in Fangorn/Rohan: may play "
                    "another Character Event free (pending)")
                return result
        state._ents_chain_n = 0
    return result


@register_effect("Dead Men of Dunharrow")
def _effect_dead_men(state: "GameState", card, choices: dict) -> dict:
    """Move Strider, attack/retreat the Shadow army, then choose 0..3 recruits."""
    target = choices.get("target_region", "erech")
    return effect_dead_men_of_dunharrow(state, target,
                                        choices.get("companions"), choices,
                                        card=card)


def _validate_stewards_recruit_choice(state: "GameState", choices: dict,
                                      require_complete=False) -> None:
    """Reject a forged region/kind before payment, placement or either draw.

    Public actions require a complete legal choice. Direct legacy callers
    may omit the kind but still must name Boromir's actual Gondor region.
    """
    if not isinstance(choices, dict):
        raise ValueError("House of the Stewards choices must be a dictionary")
    from card_options import house_of_the_stewards_options
    available = house_of_the_stewards_options(state, Side.FREE_PEOPLES)
    region = sanitize(str(choices.get('boromir_region', '')))
    if not available or region != available[0]['boromir_region']:
        raise ValueError("House of the Stewards requires Boromir's Gondor region")
    if 'unit_type' in choices and choices['unit_type'] not in {
            o.get('unit_type') for o in available if 'unit_type' in o}:
        raise ValueError("House of the Stewards unit_type is not available")
    if require_complete:
        clean = {k: v for k, v in choices.items() if not k.startswith('_')}
        clean['boromir_region'] = region
        if clean not in available:
            raise ValueError("House of the Stewards choice is not available")


@register_effect("House of the Stewards")
def _effect_house_of_stewards(state: "GameState", card, choices: dict) -> dict:
    """Recruit Gondor unit in Boromir's region, draw 2 Strategy cards.

    USER 2026-07-16: works ONLY in a Gondor region (guarded here as well as
    at the play requirement). 1.211d (2026-07-21, leg ii): the muster
    CANNOT take over captured settlements - a plain recruit_unit (with the
    normal captured-gate) blocks placement into a Shadow-held Gondor
    region, so NO recapture/control-flip. Boromir alone (no army with him)
    still musters via get_or_create_army inside recruit_unit."""
    _validate_stewards_recruit_choice(state, choices)
    result = {"effects": []}
    region = choices.get("boromir_region")
    if region:
        rkey = sanitize(region)
        if _get_region_nation(rkey) != "gondor":
            result["effects"].append(
                f"{region} is not a Gondor region - no recruit")
        else:
            unit_type = _default_unit_type(state, Nation.GONDOR, choices)
            kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
            # 1.211d leg (ii): "cannot take over captured settlements" -
            # the normal captured-gate in recruit_unit blocks a
            # Shadow-held Gondor region (no ignore_captured, no recapture).
            placed = recruit_unit(state, region, Nation.GONDOR, kind)
            if placed is not None:
                result["effects"].append(
                    f"Recruited Gondor {unit_type} in {region}")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 2)
    result["effects"].append(f"Drew {len(drawn)} Strategy cards")
    return result


@register_effect("The Grey Company")
def _effect_grey_company(state: "GameState", card, choices: dict) -> dict:
    """Upgrade regular to elite in Strider's army, draw 2 Strategy cards."""
    result = {"effects": []}
    region = choices.get("strider_region")
    nation = choices.get("nation")

    if region and nation:
        army = state.armies.get(_find_army_key(state, sanitize(region)))
        if army:
            # QA 2026-07-11 cluster I: only FP-side regulars (a shared
            # siege entry must never let this card buff the Shadow),
            # only when an Elite is physically available (skip the
            # upgrade otherwise, almanac), and the replaced Regular
            # goes to CASUALTIES (permanent), not reinforcements.
            from combat import side_of_nation as _son, \
                _track_fp_casualties
            from ai import _can_recruit
            for i, unit in enumerate(army.units):
                if unit.kind == UnitKind.REGULAR \
                        and unit.nation.value.lower() == nation.lower() \
                        and _son(unit.nation) == Side.FREE_PEOPLES:
                    if not _can_recruit(state, unit.nation, 'elite'):
                        result["effects"].append(
                            f"No {nation} Elite available - upgrade "
                            f"skipped (Regular kept)")
                        break
                    _track_fp_casualties(state, [unit],
                                         Side.FREE_PEOPLES)
                    army.units[i] = Unit(unit.nation, UnitKind.ELITE)
                    result["effects"].append(
                        f"Upgraded {nation} Regular to Elite "
                        f"(Regular to casualties)")
                    break

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 2)
    result["effects"].append(f"Drew {len(drawn)} Strategy cards")
    return result


# ---------------------------------------------------------------------------
# Registered Card Effect Handlers - Free Peoples Strategy Cards
# ---------------------------------------------------------------------------

@register_effect("The Last Battle")
def _effect_last_battle(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Fellowship movement dice not added to Hunt Box."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("The Last Battle in play - FSP movement dice bypass Hunt Box")
    return result


@register_effect("A Power too Great")
def _effect_power_too_great(state: "GameState", card, choices: dict) -> dict:
    """Advance Elves, protect Elven strongholds from attack."""
    result = {"effects": []}
    new_pos = advance_nation(state, Nation.ELVES, 1)
    result["effects"].append(f"Advanced Elves to {new_pos}")
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Shadow cannot attack Lórien/Rivendell/Grey Havens")
    return result


@register_effect("The Power of Tom Bombadil")
def _effect_tom_bombadil(state: "GameState", card, choices: dict) -> dict:
    """Advance North on the track, protect Old Forest/Shire/Buckland.

    USER RULING (2026-07): this card must NOT activate the North
    (activation comes only from the legitimate mechanisms or from
    cards whose printed text activates - see advance_nation).
    The printed card advances the North political counter one step
    (advance_nation no longer auto-activates) and blocks Shadow armies
    from The Old Forest / The Shire / Buckland while on the table.
    """
    result = {"effects": []}
    new_pos = advance_nation(state, Nation.THE_NORTH, 1)
    result["effects"].append(f"Advanced North to {new_pos} (not activated)")
    _handle_persistent_card(state, Side.FREE_PEOPLES, card, result)
    result["effects"].append("Shadow cannot attack Old Forest/Shire/Buckland")
    return result


def _move_companion_ranged(state, comp_key, dest, result):
    """Move a separated companion up to its LEVEL regions (USER RULE
    2026-07-08: Book of Mazarbul / Fear! Fire! Foes! "let companions
    move equal to their movement speed").  Returns the destination key
    if moved."""
    from game_state import Companion, Army
    for region_key, army in list(state.armies.items()):
        for leader in list(army.leaders):
            if not isinstance(leader, Companion):
                continue
            if leader.name.replace(' ', '_').lower() != comp_key:
                continue
            cur = sanitize(region_key)
            dest_san = sanitize(dest)
            _rng = max(0, leader.level)
            if leader.name == "Gandalf the White":
                # almanac M4: Shadowfax - GtW moves at level 4 when
                # alone (these cards move companions singly)
                _rng = max(_rng, 4)
            if dest_san != cur:
                from ai import _companion_reachable
                if dest_san not in _companion_reachable(
                        state, cur, _rng):
                    result["effects"].append(
                        f"{leader.name}: {dest} out of range "
                        f"(level {leader.level}) - not moved")
                    return None
            army.leaders.remove(leader)
            if not army.units and not army.leaders:
                state.armies.pop(region_key, None)
            destination_army = _army_entry_for(state, dest_san)
            destination_army.leaders.append(leader)
            from game_logic import _check_companion_activates_nation
            _check_companion_activates_nation(state, dest_san, [leader])
            result["effects"].append(f"Moved {leader.name} to {dest}")
            return dest_san
    return None


def _army_entry_for(state, dest_san):
    """Army entry for a sanitized region key WITHOUT minting duplicate
    keys (QA 2026-07-11: separation handlers created a second
    'helm s deep' entry beside the display-keyed one)."""
    from game_state import Army as _ArmyH
    k = _find_army_key(state, dest_san)
    if k is None or k not in state.armies:
        state.armies[dest_san] = _ArmyH(region=state.board.region(dest_san))
        k = dest_san
    return state.armies[k]


def _dest_ok_from(state, cur, dest_san, rng, allow_besieged=False):
    """Is `dest_san` a legal card-move destination within `rng` regions
    of `cur`?  `allow_besieged` permits a besieged stronghold as the
    END region (Gwaihir / We Prove Swifter)."""
    from ai import _companion_reachable, _friendly_besieged_strongholds
    cur = sanitize(cur)
    if dest_san == cur:
        return True
    # These cards permit entering, but never leaving, a friendly siege.
    # Keep this before the destination-only exception's fallback walk.
    if cur in _friendly_besieged_strongholds(state):
        return False
    if dest_san in _companion_reachable(state, cur, max(0, rng)):
        return True
    bs = getattr(state, 'besieged_strongholds', set())
    if allow_besieged and dest_san in bs:
        # reachable ignoring the besieged-exclusion for the END region
        # only: the path still obeys _companion_reachable's rules - no
        # passing THROUGH a besieged stronghold, and a Shadow-controlled
        # stronghold may be ENTERED but stops the movement (standing
        # ruling: stop-on-entry is per movement action)
        from ai import _is_shadow_controlled_stronghold
        from collections import deque
        seen, q = {cur}, deque([(cur, 0)])
        while q:
            r, d = q.popleft()
            if d >= rng:
                continue
            for adj in state.board.adjacent(r):
                a = sanitize(adj)
                if a in seen:
                    continue
                seen.add(a)
                if a == dest_san:
                    return True
                if a in bs or _is_shadow_controlled_stronghold(state, a):
                    continue
                q.append((a, d + 1))
    return False


def _separation_dest_ok(state, dest_san, rng, allow_besieged=False):
    """Is `dest_san` a legal separation destination within `rng` regions
    of the Fellowship? (USER FIX 2026-07-08 — the companion-separation
    cards teleported with no range check.)  `allow_besieged` permits a
    besieged stronghold as the END region (Gwaihir / We Prove Swifter)."""
    import game_logic as _gl
    return _dest_ok_from(state, _gl.fellowship_region(state), dest_san,
                         rng, allow_besieged=allow_besieged)


def _remove_companions_from_play(state, names, result):
    """Mordor-Track separation (almanac, QA 2026-07-11 cluster B): the
    separated companions are REMOVED FROM PLAY - not placed on the
    board (e.g. done to bring Gollum in as Guide)."""
    import game_logic as _gl
    removed = []
    for comp in list(state.fellowship.companions):
        if comp.name in names:
            state.fellowship.companions.remove(comp)
            removed.append(comp.name)
    if removed:
        _gl.change_guide(state, allow_pending=True)
        _gl.check_companion_persistent_discards(state)
        result["effects"].append(
            "Removed from play (Mordor Track separation): "
            + ", ".join(removed))
    return removed


def _companion_group_move_options(state, exclude=None) -> list:
    """Group-move options for Book of Mazarbul / Fear! Fire! Foes! AND
    the regular-board '__companion_moves__' chain (P3, USER ruling
    1.112.4; rulebook p.20): every non-empty subset of the separated
    companions in a region with the GROUP's movement allowance.
    `exclude` = names already moved this play/die (each companion moves
    once).

    Every subset remains available: a low-Level companion may need to
    travel with an escort while the rest stay behind. Reusing legal
    destinations by movement allowance avoids repeated path searches
    without removing groups or imposing a separate destination rule."""
    from ai import (Action, _companion_group_movement, _companion_reachable,
                    is_legal_companion_action)
    from game_state import Companion
    from itertools import combinations
    exclude = exclude or set()
    army_by_san = {sanitize(k): a for k, a in state.armies.items()}
    validator_cache = {}
    opts = []
    for src, army in sorted(army_by_san.items()):
        comps = [l for l in army.leaders if isinstance(l, Companion)
                 and l.name not in exclude]
        if not comps:
            continue
        subs = (c for r in range(1, len(comps) + 1)
                for c in combinations(comps, r))
        legal_by_movement = {}
        for sub in subs:
            mv = _companion_group_movement(sub)
            if mv <= 0:
                continue
            names = sorted(c.name for c in sub)
            if mv not in legal_by_movement:
                legal_by_movement[mv] = [
                    dest for dest in sorted(_companion_reachable(state, src, mv))
                    if is_legal_companion_action(
                        state, Action(-1, 'companion_move', {
                            'from': src, 'to': dest, 'companions': names}),
                        validator_cache)[0]]
            destinations = list(legal_by_movement[mv])
            if is_legal_companion_action(
                    state, Action(-1, 'companion_move', {
                        'from': src, 'to': src, 'companions': names}),
                    validator_cache)[0]:
                destinations.append(src)
                destinations.sort()
            for dest in destinations:
                opts.append({'move_group': {
                    'from': src, 'to': dest, 'companions': list(names)}})
    return opts


def _move_companion_group(state, grp: dict, result: dict) -> bool:
    """Move a named companion group (P3).  Range = the group's
    allowance (max level; GtW-4 alone/with one hobbit)."""
    from ai import _companion_group_movement, _companion_reachable
    from game_state import Companion, Army
    src = sanitize(str(grp.get('from', '')))
    dst = sanitize(str(grp.get('to', '')))
    names = {str(n) for n in (grp.get('companions') or [])}
    src_key = next((k for k in state.armies if sanitize(k) == src), None)
    if src_key is None or not names:
        return False
    army = state.armies[src_key]
    comps = [l for l in army.leaders if isinstance(l, Companion)
             and l.name in names]
    if not comps:
        return False
    mv = _companion_group_movement(comps)
    if dst != src and dst not in _companion_reachable(state, src, mv):
        result["effects"].append(
            f"Group {sorted(names)}: {dst} out of range - not moved")
        return False
    if dst == src:
        from game_logic import _check_companion_activates_nation
        _check_companion_activates_nation(state, dst, comps)
        return True
    for c in comps:
        army.leaders.remove(c)
    if not army.units and not army.leaders:
        state.armies.pop(src_key, None)
    if dst not in state.armies:
        dst_key = next((k for k in state.armies
                        if sanitize(k) == dst), None)
        if dst_key is None:
            state.armies[dst] = Army(region=state.board.region(dst))
            dst_key = dst
    else:
        dst_key = dst
    state.armies[dst_key].leaders.extend(comps)
    # Event movement and ordinary continuation both activate on arrival.
    # Existing destination Companions are not part of this move.
    from game_logic import _check_companion_activates_nation
    _check_companion_activates_nation(state, dst, comps)
    result["effects"].append(
        f"Moved group {', '.join(sorted(c.name for c in comps))} "
        f"to {dst}")
    return True


def _companions_war_trigger(state, result, regions: set,
                            nation, label: str) -> None:
    """'Then, if a Companion is in <regions>, move <nation> to At War'
    - evaluated on the CURRENT board (after all moves; almanac p32: a
    captured region does not trigger)."""
    from game_state import Companion
    _ec_sh = state.enemy_controlled.get(Side.SHADOW, set())
    for region_key, army in state.armies.items():
        rsan = sanitize(region_key)
        if rsan in regions and rsan not in _ec_sh                 and any(isinstance(l, Companion) for l in army.leaders):
            set_at_war(state, nation)
            result["effects"].append(label)
            return


def _companion_move_card_pending(state, card, choices, result,
                                 regions: set, nation, label: str):
    """Shared chained group-move flow for Mazarbul / FFF (P3): each
    decision moves one group; the terminator (or exhaustion) fires the
    war trigger.  Returns True when the pending path handled the call."""
    if not choices.get('_from_pending'):
        return False
    if choices.get('_ffm_done'):
        _companions_war_trigger(state, result, regions, nation, label)
        return True
    grp = choices.get('move_group')
    if not grp:
        return False  # legacy 'move_<name>' keys - old path handles
    _move_companion_group(state, grp, result)
    moved = set(str(n) for n in (choices.get('_ffm_moved') or []))
    moved.update(str(n) for n in (grp.get('companions') or []))
    opts = _companion_group_move_options(state, exclude=moved)
    if opts:
        for o in opts:
            o['_ffm_moved'] = sorted(moved)
        opts.append({'_ffm_done': True})
        state.pending_card_option = {
            'card_id': getattr(card, 'card_id', card.card_name),
            'card_name': card.card_name,
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'options': opts}
    else:
        _companions_war_trigger(state, result, regions, nation, label)
    return True


@register_effect("Book of Mazarbul")
def _effect_book_of_mazarbul(state: "GameState", card, choices: dict) -> dict:
    """Move non-Fellowship companions. Dwarves to At War if companion in Erebor/Ered Luin."""
    from game_state import Companion, Army
    result = {"effects": []}

    # P3 (USER ruling 1.112.4): chained GROUP moves on the pending path
    if _companion_move_card_pending(
            state, card, choices, result,
            {"erebor", "ered luin", "eredluin", "ered_luin"},
            Nation.DWARVES,
            "Companion in Erebor/Ered Luin - Dwarves moved to At War!"):
        return result

    # Movement is RANGE-LIMITED to each companion's level; there is no
    # teleport default — self-play chooses via the card-option spec
    companion_final_regions = []
    for key, dest in choices.items():
        if key.startswith("move_") and dest:  # dest empty = "Stay"
            moved = _move_companion_ranged(state, key[5:], dest, result)
            if moved:
                companion_final_regions.append(moved)

    # Also track companions that stayed in place
    for region_key, army in state.armies.items():
        for leader in army.leaders:
            if isinstance(leader, Companion):
                if sanitize(region_key) not in companion_final_regions:
                    companion_final_regions.append(sanitize(region_key))

    # Check for Dwarves At War condition
    dwarven_regions = {"erebor", "ered luin", "eredluin", "ered_luin"}
    # almanac p32: a Companion ending at a Shadow-controlled region does not
    # activate/advance
    _ec_sh = state.enemy_controlled.get(Side.SHADOW, set())
    if any(sanitize(r) in dwarven_regions and sanitize(r) not in _ec_sh
           for r in companion_final_regions):
        set_at_war(state, Nation.DWARVES)
        result["effects"].append("Companion in Erebor/Ered Luin - Dwarves moved to At War!")

    # Empty effects list lets the AI's no-op dry-run filter skip the card
    # when there is genuinely nothing useful to do
    return result


def _apply_card_hits_to_shadow(state, army, hits):
    """Apply card-effect hits to the SHADOW portion of an army entry
    only - a besieged region's shared entry also holds FP units, which
    a card targeting a Shadow army must never kill (QA 2026-07-09)."""
    from combat import apply_casualties
    sh_units = [u for u in army.units if u.nation in SHADOW_NATIONS]
    fp_units = [u for u in army.units if u.nation not in SHADOW_NATIONS]
    tmp = Army(region=army.region)
    tmp.units = sh_units
    casualties = apply_casualties(tmp, hits)
    army.units = fp_units + tmp.units
    return casualties


def _side_filtered_kills(army, dying_side, hits, eligible_nations=None):
    """preserve_elites kills restricted to one side of a (possibly
    shared siege) entry — the no-spec teacher fallback of
    _route_card_casualties."""
    from combat import apply_casualties, side_of_nation
    def eligible(unit):
        return (side_of_nation(unit.nation) == dying_side
                and (eligible_nations is None or unit.nation in eligible_nations))
    mine = [u for u in army.units if eligible(u)]
    other = [u for u in army.units if not eligible(u)]
    tmp = Army(region=army.region)
    tmp.units = mine
    cas = apply_casualties(tmp, hits)
    army.units = other + tmp.units
    return cas


# USER RULING 1.254 (all casualties are learned decisions): cards whose
# remaining legs must RESUME once the owner's allocation pending
# resolves.  The ai.py card_casualty executor applies the allocation and
# calls _complete_card_casualty(card_name=...), which re-enters the
# handler with the marker choices below (keyed by the pending's
# card_name; the marker names what the roll already settled).
_CARD_CASUALTY_RESUME = {
    'dead men of dunharrow': lambda region: {
        'target_region': region, '_dmd_cas_resumed': True},
    'the ents awake: treebeard': lambda region: {'_ents_cas_resumed': True},
    'the ents awake: huorns': lambda region: {'_ents_cas_resumed': True},
    'the ents awake: entmoot': lambda region: {'_ents_cas_resumed': True},
    "faramir's rangers": lambda region: {
        'target_region': region, '_far_cas_resumed': True},
    'return to valinor': lambda region: {'_rtv_cas_resumed': region},
}


def _resume_card_after_casualty(state, card_name, region, result):
    """Re-enter the card handler after its owner-allocated casualty
    pending resolved (1.254 continuation).  Same dispatch choke point
    as the direct-pending executor: recruit overstack + active-state
    published around the handler, overstacks re-scanned after."""
    key = str(card_name or '').lower()
    mk = _CARD_CASUALTY_RESUME.get(key)
    if mk is None:
        return
    handler = CARD_EFFECT_HANDLERS.get(key)
    if handler is None:
        raise RuntimeError(
            "card casualty continuation: no handler for %r" % (card_name,))
    card = _card_by_name(card_name)
    global CARD_RECRUIT_OVERSTACK
    from combat import set_active_state as _sas_r
    CARD_RECRUIT_OVERSTACK = True
    _prev_r = _sas_r(state)
    try:
        r2 = handler(state, card, dict(mk(region), _from_pending=True))
    finally:
        CARD_RECRUIT_OVERSTACK = False
        _sas_r(_prev_r)
    result["effects"].extend(r2.get("effects") or [])
    resolve_overstacks(state, learned=True)


def _complete_card_casualty(state, region_key, dying_side, result,
                            card_name=None, resume=True):
    """Shared aftermath for card-inflicted casualties: the
    post-destruction CHARACTER reconciler + the standing orphan-leader
    sweep + the wiped-besieged-garrison capture, in both directions
    (extracted from the Spirit of Mordor / Dreadful Spells inline
    blocks; USER 2026-07-27 casualty-routing wave).

    Card-specific character survival is owned by
    combat.reconcile_destroyed_army and CHARACTER_SPARING_CARDS.
    Return to Valinor removes plain Leaders only when the army is
    eliminated; Companions survive (Almanac p31).
    `card_name` (or, when omitted, combat's published wipe context)
    selects which.  It also reaches the shapes remove_orphaned_leaders
    cannot: a SHARED besieged entry stays non-empty (the enemy's units
    keep it alive), so the standing sweep never sees the wipe.

    resume=True (the ai.py card_casualty executor's call, i.e. AFTER an
    owner-allocation pending resolved): the card's remaining legs
    resume through _CARD_CASUALTY_RESUME (1.254).  _route_card_casualties
    passes resume=False - there the handler is still on the stack."""
    _card_casualty_aftermath(state, region_key, dying_side, result,
                             card_name)
    if resume:
        _resume_card_after_casualty(state, card_name,
                                    sanitize(str(region_key)), result)


def _card_casualty_aftermath(state, region_key, dying_side, result,
                             card_name=None):
    import game_logic as _glcc
    from combat import (garrison_side_for, _check_settlement_capture,
                        reconcile_destroyed_army,
                        side_of_nation as _soncc)
    rsan = sanitize(str(region_key))
    army = next((a for k, a in state.armies.items()
                 if sanitize(k) == rsan), None)
    if army is not None and not any(_soncc(u.nation) == dying_side
                                    for u in army.units):
        # the dying side's units reached ZERO -> destruction event
        reconcile_destroyed_army(state, rsan, dying_side,
                                 card_name=card_name)
        army = next((a for k, a in state.armies.items()
                     if sanitize(k) == rsan), None)
    _glcc.remove_orphaned_leaders(state)
    if army is None:
        return
    bs = getattr(state, 'besieged_strongholds', set())
    if rsan not in bs:
        return
    if dying_side == Side.SHADOW:
        # almanac A5: FP besieger captures a wiped Shadow garrison
        if not any(u.nation in SHADOW_NATIONS for u in army.units) \
                and garrison_side_for(state, rsan) == Side.SHADOW:
            bs.discard(rsan)
            _check_settlement_capture(state, rsan, Side.FREE_PEOPLES)
            result["effects"].append(
                "Shadow garrison eliminated - Free Peoples capture "
                + rsan)
    else:
        # almanac C7/C8/C19: wiped besieged FP garrison falls (the
        # reconciler above already booked the garrison's characters)
        if not any(u.nation not in SHADOW_NATIONS
                   for u in army.units) \
                and garrison_side_for(state, rsan) == Side.FREE_PEOPLES:
            bs.discard(rsan)
            _check_settlement_capture(state, rsan, Side.SHADOW)
            result["effects"].append(
                "Garrison eliminated - Shadow captures " + rsan)


def _card_casualty_options(state, region, side, hits):
    """Card-option wrapper around the ordinary exact casualty vocabulary."""
    import ai
    return [{'casualty_operation': action.params['operation'],
             'nation': action.params['nation'], 'target_region': sanitize(str(region))}
            for action in ai._casualty_operations(state, region, side, hits)]


def _route_card_casualties(state, region_key, dying_side, hits, card,
                           choices, result):
    """Allocate card hits as explicit owner-chosen casualty operations.

    Teacher-hook replay retains its historical literal-spec path. Ordinary
    play uses the shared sequential operations and completes the card's
    aftermath once after all hits are allocated. The calling card handler
    resumes its own remaining legs when this returns True.
    """
    import combat as _cmbcc
    rsan = sanitize(str(region_key))
    if hits <= 0:
        return True
    army = next((a for k, a in state.armies.items()
                 if sanitize(k) == rsan), None)
    if army is None:
        return True
    # USER RULING 2026-08-15: the CARD identity decides whether a wipe
    # spares the characters (Spirit of Mordor / Dreadful Spells) — carry
    # it explicitly, the pending dict already re-supplies it on resume.
    _cnm9 = getattr(card, 'card_name', None)
    if _cmbcc.FINAL_HITS_HOOK is not None:      # teacher replay
        spec = (choices or {}).get('card_cas_spec')
        if spec:
            cas = _cmbcc.apply_casualties_spec(army, spec,
                                               side=dying_side)
        else:
            cas = _side_filtered_kills(army, dying_side, hits)
        if dying_side == Side.FREE_PEOPLES:
            _cmbcc._track_fp_casualties(state, cas, Side.FREE_PEOPLES)
        result["effects"].append("%d casualties" % len(cas))
        _complete_card_casualty(state, rsan, dying_side, result,
                                card_name=_cnm9, resume=False)
        return True
    import ai as _aicc
    pending = {
        'card_id': getattr(card, 'card_id', str(card)),
        'card_name': getattr(card, 'card_name', 'card'),
        'card_casualty': True, 'direct': True,
        'side': 'sh' if dying_side == Side.SHADOW else 'fp',
        'decider': 'sh' if dying_side == Side.SHADOW else 'fp',
        'cas_region': rsan, 'cas_side': dying_side,
        'cas_hits': int(hits)}
    if str(_cnm9 or '').lower() == 'return to valinor':
        # Almanac p31: other nations neither contribute dice nor take damage.
        pending['cas_eligible_nations'] = (Nation.ELVES,)
    # Publish card context while forced operations execute as well, so their
    # gross-flow witnesses identify card timing rather than a combat stage.
    state.pending_card_option = pending
    remaining = _aicc._forced_casualty_allocation(state, rsan, dying_side, int(hits))
    if remaining <= 0:
        state.pending_card_option = None
        result["effects"].append("%d hits applied (forced allocation)" % hits)
        _complete_card_casualty(state, rsan, dying_side, result,
                                card_name=_cnm9, resume=False)
        return True
    pending['cas_hits'] = remaining
    pending['options'] = _card_casualty_options(state, rsan, dying_side, remaining)
    result["effects"].append(
        "%d hits pending owner allocation at %s" % (remaining, rsan))
    return False


@register_effect("The Spirit of Mordor")
def _effect_spirit_of_mordor(state: "GameState", card, choices: dict) -> dict:
    """Roll 5 dice vs multi-nation Shadow army, hit on 5+.

    Official requirement: the target must be a Shadow Army composed of units
    from at least two different Shadow Nations.
    """
    result = {"effects": []}

    def _find_army(name):
        san = sanitize(name)
        for k, a in state.armies.items():
            if sanitize(k) == san:
                return a
        return None

    def _is_multi_nation_shadow(a):
        # QA 2026-07-09 (almanac A5): only SHADOW Army units count; FP
        # bystanders in a shared siege entry must not disqualify the
        # target (a besieged multi-nation garrison IS targetable)
        sh_nations = {u.nation for u in a.units
                      if u.nation in SHADOW_NATIONS}
        return len(sh_nations) >= 2

    target_region = choices.get("target_region")
    army = None
    if target_region:
        army = _find_army(target_region)
        if not army or not _is_multi_nation_shadow(army):
            # No effect (empty effects list lets the AI's no-op filter skip it)
            result["message"] = (f"Invalid target {target_region}: requires a "
                                 f"Shadow army with units from 2+ Shadow Nations")
            return result
    else:
        valid = [(k, a) for k, a in state.armies.items()
                 if _is_multi_nation_shadow(a)]
        if not valid:
            result["message"] = "No Shadow army with units from 2+ Shadow Nations"
            return result
        import combat as _cmb_som7
        if len(valid) > 1 and _cmb_som7.FINAL_HITS_HOOK is None:
            # SQF F4 (1.248-class, decision-ownership doctrine): with
            # 2+ valid targets the target_region pick is the CARD
            # PLAYER's learned decision - a silent default substitutes
            # it.  The card leaves the hand on return (play_event_card
            # tail); the chosen option re-dispatches here through the
            # 'direct' card_option lane.  Teacher replay
            # (FINAL_HITS_HOOK) keeps the documented default: log =
            # truth, pendings never open there (same guard as
            # _route_card_casualties).
            _own7 = ('fp' if getattr(card, 'side', None)
                     == Side.FREE_PEOPLES else 'sh')
            state.pending_card_option = {
                'card_id': getattr(card, 'card_id', str(card)),
                'card_name': getattr(card, 'card_name',
                                     'The Spirit of Mordor'),
                'side': _own7, 'decider': _own7, 'direct': True,
                'options': [{'target_region': sanitize(k)}
                            for k, _a in valid]}
            result["effects"].append(
                "target choice pending (%d valid multi-nation Shadow "
                "armies)" % len(valid))
            return result
        # forced pick (single valid target) / teacher default: the
        # largest valid multi-nation Shadow army
        target_region, army = max(valid, key=lambda kv: len(kv[1].units))

    rolls = _d6_batch(5)
    _ann_rng(state, 'The Spirit of Mordor roll', faces=rolls)
    hits = sum(1 for r in rolls if r >= 5)
    result["effects"].append(f"Rolled {rolls}, {hits} hits against {target_region}")
    if hits > 0:
        # USER 2026-07-27: owner-allocated via the shared casualty
        # decision matrix (teacher: logged spec; self-play: pending).
        # USER RULING 2026-08-15: publish the wipe context so ANY
        # destruction reconciled underneath takes the Spirit-of-Mordor
        # exception (companions/nazgul/minions survive, plain Leaders
        # die) even on a path that does not thread card_name.
        import combat as _cmb_som9
        _prevw9 = _cmb_som9.set_wipe_card_context('The Spirit of Mordor')
        try:
            _route_card_casualties(state, target_region, Side.SHADOW,
                                   hits, card, choices, result)
        finally:
            _cmb_som9.set_wipe_card_context(_prevw9)
    return result


def _faramir_targets(state):
    """Printed strike regions containing actual Shadow Army units."""
    return [r for r in ('osgiliath', 'south ithilien', 'north ithilien')
            if any(sanitize(k) == r and any(u.nation in SHADOW_NATIONS for u in a.units)
                   for k, a in state.armies.items())]


def _faramir_fp_army_present(state):
    """Almanac18: any FP Army in Osgiliath permits the recruitment leg."""
    return any(sanitize(k) == 'osgiliath' and any(u.nation in FP_NATIONS for u in a.units)
               for k, a in state.armies.items())


def _faramir_recruit_options(state):
    if not _faramir_fp_army_present(state):
        return []
    from ai import figures_available
    return [{'_far_recruit': True, 'unit_type': kind}
            for kind in ('regular', 'elite')
            if figures_available(state, Nation.GONDOR, kind) > 0] or [{'_far_recruit': True}]


def _validate_faramir_choices(state, choices, *, recruitment_stage=False,
                             require_complete=False):
    """Validate this card before payment or clearing its paid pending.

    Positional legality is independent of supplies. Recruitment cannot be
    selected before a required interactive strike: that later choice belongs
    after its casualty allocation. Complete teacher records may carry both.
    """
    if choices is not None and not isinstance(choices, dict):
        raise ValueError("Faramir's Rangers choices must be a dictionary")
    selected = dict(choices or {})
    selected.pop('_from_pending', None)
    if recruitment_stage:
        options = _faramir_recruit_options(state)
        if selected.get('_far_recruit') is not True or selected not in options:
            raise ValueError("Faramir's Rangers recruitment choice is not available")
        return
    if '_far_recruit' in selected or '_far_cas_resumed' in selected:
        raise ValueError("Faramir's Rangers continuation cannot start a card play")
    targets = _faramir_targets(state)
    if not targets and not _faramir_fp_army_present(state):
        raise ValueError("Faramir's Rangers requires a Shadow target or an FP Army in Osgiliath")
    target = selected.get('target_region')
    if target is not None and sanitize(str(target)) not in targets:
        raise ValueError("Faramir's Rangers target is not available")
    if require_complete and targets and target is None:
        raise ValueError("Faramir's Rangers requires the selected Shadow target")
    if 'unit_type' in selected:
        kinds = {o.get('unit_type') for o in _faramir_recruit_options(state)
                 if 'unit_type' in o}
        if selected['unit_type'] not in kinds:
            raise ValueError("Faramir's Rangers unit_type is not available")
        import combat
        if targets and combat.FINAL_HITS_HOOK is None:
            raise ValueError("Choose Faramir's Rangers recruitment after the strike")


@register_effect("Faramir's Rangers")
def _effect_faramirs_rangers(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice vs Shadow in Osgiliath area, recruit if FP army there."""
    result = {"effects": []}
    recruit_stage = choices.get('_far_recruit') is True
    resumed = bool(choices.get('_far_cas_resumed'))
    if recruit_stage:
        _validate_faramir_choices(state, choices, recruitment_stage=True,
                                  require_complete=True)
    elif not resumed:
        _validate_faramir_choices(state, choices)
    target = choices.get("target_region", "osgiliath")
    # sanitize() produces SPACE forms; the strike targets only the Shadow army
    # at a valid region (no friendly fire on a recruitment-only play)
    valid_targets = {"osgiliath", "south ithilien", "north ithilien"}

    def _sh_army_at(r):
        a = state.armies.get(_find_army_key(state, sanitize(r)))
        if a and any(u.nation in SHADOW_NATIONS for u in a.units):
            return a
        return None

    tsan = sanitize(target)
    army = _sh_army_at(tsan) if tsan in valid_targets else None
    if not recruit_stage and not resumed and not choices.get("target_region"):
        # FLC C10 (1.248-class, decision-ownership doctrine): the card
        # text says CHOOSE a Shadow Army - with 2+ valid targets the
        # target_region pick is the CARD PLAYER's learned decision, so
        # a choiceless play opens pending_card_option (direct lane),
        # exactly like The Spirit of Mordor.  Teacher replay
        # (FINAL_HITS_HOOK, log = truth) keeps the documented default
        # and never opens a pending.
        import combat as _cmb_far7
        _cands7 = [r for r in ("osgiliath", "south ithilien",
                               "north ithilien")
                   if _sh_army_at(r) is not None]
        if len(_cands7) > 1 and _cmb_far7.FINAL_HITS_HOOK is None:
            _own7 = ('fp' if getattr(card, 'side', None)
                     == Side.FREE_PEOPLES else 'sh')
            state.pending_card_option = {
                'card_id': getattr(card, 'card_id', str(card)),
                'card_name': getattr(card, 'card_name',
                                     "Faramir's Rangers"),
                'side': _own7, 'decider': _own7, 'direct': True,
                'options': [{'target_region': sanitize(r)}
                            for r in _cands7]}
            result["effects"].append(
                "target choice pending (%d valid Shadow-army targets)"
                % len(_cands7))
            return result
        # forced single target / teacher default: the valid target
        # holding the largest Shadow army (the strike lands; the log
        # records the casualty, never the target choice)
        if army is None and _cands7:
            tsan = max(_cands7, key=lambda r: sum(
                1 for u in _sh_army_at(r).units
                if u.nation in SHADOW_NATIONS))
            army = _sh_army_at(tsan)
    # USER RULING 1.254: the hits are the Shadow OWNER's learned
    # allocation; '_far_cas_resumed' re-enters after that pending with
    # the roll already spent (continuation via _complete_card_casualty)
    # and runs the recruit leg.
    if army is not None and not resumed and not recruit_stage:
        rolls = _d6_batch(3)
        _ann_rng(state, "Faramir's Rangers roll", faces=rolls)
        hits = sum(1 for r in rolls if r >= 5)
        result["effects"].append(f"Rolled {rolls}, {hits} hits against {tsan}")
        if hits > 0:
            import combat as _cmb_far
            if _cmb_far.FINAL_HITS_HOOK is None:
                if not _route_card_casualties(
                        state, tsan, Side.SHADOW, hits,
                        card or _card_by_name("Faramir's Rangers"),
                        choices, result):
                    return result
            else:
                # teacher: certified logged-composition path
                casualties = _apply_card_hits_to_shadow(state, army, hits)
                result["effects"].append(f"{len(casualties)} casualties")
                # USER RULING 2026-08-15: a wiped army takes its
                # characters with it (Faramir's Rangers is a sparing
                # card only for Companions/Nazgul/Minions per combat's
                # CHARACTER_SPARING_CARDS; the reconciler decides).
                if not any(u.nation in SHADOW_NATIONS for u in army.units):
                    from combat import reconcile_destroyed_army as _rda9f
                    _rda9f(state, tsan, Side.SHADOW,
                           card_name="Faramir's Rangers")

    # Recruit in Osgiliath if FP army there
    if _faramir_fp_army_present(state):
        # The unit kind is a human decision AFTER the strike and any Shadow
        # casualty choice. AI policy projects this menu to Elite if legal.
        options = _faramir_recruit_options(state)
        import combat as _cmb_far_recruit
        if not recruit_stage and 'unit_type' not in choices:
            if len(options) > 1 and _cmb_far_recruit.FINAL_HITS_HOOK is None:
                state.pending_card_option = {
                    'card_id': getattr(card, 'card_id', None) or
                               "FREE_PEOPLES_Strategy_Faramir's_Rangers",
                    'card_name': "Faramir's Rangers", 'side': 'fp',
                    'decider': 'fp', 'direct': True, 'faramir_recruit': True,
                    'options': options}
                result['effects'].append('Gondor recruitment choice pending in Osgiliath')
                return result
            if len(options) == 1:
                choices = dict(choices, **options[0])
        # 1.211a: elite-preferred by default (honors an explicit logged
        # choices['unit_type'] first; regular only if the pool is dry).
        unit_type = _default_unit_type(state, Nation.GONDOR, choices)
        kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
        # honest per-piece reporting (same family as LOW-24/27/29/30)
        _u = recruit_unit(state, "Osgiliath", Nation.GONDOR, kind)
        _l = recruit_leader(state, "Osgiliath", Nation.GONDOR)
        if _u is not None:
            result["effects"].append(
                f"Recruited Gondor {unit_type} in Osgiliath")
        if _l is not None:
            result["effects"].append(
                "Recruited Gondor Leader in Osgiliath")
    # RB22 permits supply-limited effects once the positional requirement
    # is met, including no available unit or Leader. Missing position was
    # rejected before execution; it is not converted to a legal no-op here.
    return result


@register_effect("Fear! Fire! Foes!")
def _effect_fear_fire_foes(state: "GameState", card, choices: dict) -> dict:
    """Move non-Fellowship companions. North to At War if companion in Shire/Bree."""
    from game_state import Companion, Army
    result = {"effects": []}

    # P3 (USER ruling 1.112.4): chained GROUP moves on the pending path
    if _companion_move_card_pending(
            state, card, choices, result,
            {"the shire", "shire", "bree"},
            Nation.THE_NORTH,
            "Companion in The Shire/Bree - North moved to At War!"):
        return result

    # RANGE-LIMITED companion moves (USER RULE 2026-07-08, same as
    # Book of Mazarbul)
    companion_final_regions = []
    for key, dest in choices.items():
        if key.startswith("move_") and dest:  # dest empty = "Stay"
            moved = _move_companion_ranged(state, key[5:], dest, result)
            if moved:
                companion_final_regions.append(moved)

    # Also track companions that stayed in place
    for region_key, army in state.armies.items():
        for leader in army.leaders:
            if isinstance(leader, Companion):
                if sanitize(region_key) not in companion_final_regions:
                    companion_final_regions.append(sanitize(region_key))

    # Check for North At War condition
    # sanitize() outputs are SPACE forms only
    north_regions = {"the shire", "shire", "bree"}
    # almanac p32 (QA 2026-07-09): no activation at captured regions
    _ec_sh = state.enemy_controlled.get(Side.SHADOW, set())
    if any(sanitize(r) in north_regions and sanitize(r) not in _ec_sh
           for r in companion_final_regions):
        set_at_war(state, Nation.THE_NORTH)
        result["effects"].append("Companion in Shire/Bree - North moved to At War!")

    if not result["effects"]:
        result["message"] = "No companions moved"  # no-op: NOT an effect (iter 39)

    return result


@register_effect("Wisdom of Elrond")
def _effect_wisdom_of_elrond(state: "GameState", card, choices: dict) -> dict:
    """Activate a nation, advance it 1 step."""
    result = {"effects": []}
    nation_name = choices.get("nation")
    if not nation_name:
        # Rules-honest default ("Activate one FP Nation of your choice"): the
        # first FP nation that can still benefit
        for nat in (Nation.ROHAN, Nation.GONDOR, Nation.THE_NORTH,
                    Nation.DWARVES, Nation.ELVES):
            if state.political.boxes.get(nat, 0) < 3:
                nation_name = ('north' if nat == Nation.THE_NORTH
                               else nat.value.lower())
                break
    if nation_name:
        nation = NATION_KEYWORDS.get(nation_name.lower())
        if nation:
            if nation not in FP_NATIONS:
                result["message"] = ("Wisdom of Elrond: %s is not a Free "
                                     "Peoples Nation - not played"
                                     % nation.value)
                return _selfplay_refund(result)
            from persistent_effects import can_activate_nation
            _ok, _blocker = can_activate_nation(state, nation,
                                                "event_card")
            if _ok:
                activate_nation(state, nation)
                new_pos = advance_nation(state, nation, 1)
                result["effects"].append(
                    f"Activated and advanced {nation.value} to {new_pos}")
            else:
                # almanac M8: Wormtongue keeps Rohan passive - the card still
                # advances the track, clamped at box 2 while passive
                new_pos = advance_nation(state, nation, 1)
                result["effects"].append(
                    f"Advanced {nation.value} to {new_pos} "
                    f"(activation blocked by {_blocker})")
    return result


@register_effect("The Red Arrow")
def _effect_red_arrow(state: "GameState", card, choices: dict) -> dict:
    """Advance Rohan, recruit in Edoras.  Honest per-piece reporting:
    a blocked placement/clamped advance is never claimed as an effect
    (no-op plays refund)."""
    result = {"effects": []}
    _old_pos = state.political.boxes.get(Nation.ROHAN, 0)
    new_pos = advance_nation(state, Nation.ROHAN, 1)
    if new_pos != _old_pos:
        result["effects"].append(f"Advanced Rohan to {new_pos}")

    unit_type = _default_unit_type(state, Nation.ROHAN, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    _u = recruit_unit(state, "Edoras", Nation.ROHAN, kind)
    _l = recruit_leader(state, "Edoras", Nation.ROHAN)
    if _u is not None:
        result["effects"].append(f"Recruited Rohan {unit_type} in Edoras")
    if _l is not None:
        result["effects"].append("Recruited Rohan Leader in Edoras")
    if not result["effects"]:
        result["message"] = "Red Arrow: no effect (blocked/clamped)"
        return _selfplay_refund(result)
    return result


def _direct_army_move(state, frm_san, to_san, side,
                      exclude_not_at_war=False,
                      n_regular=None, n_elite=None,
                      allow_overstack=False, via_san=None, cargo=None,
                      defer_overstack=False):
    """Move the acting side's army pieces DIRECTLY between two regions
    (card-granted movement), replicating game_logic.move_army's side
    effects: enemy/Saruman pieces stay behind, stacking cap 10 own
    units at the destination (overflow stays behind, regulars first),
    political entry check, companion activation, orphan-leader removal,
    settlement capture, and siege bookkeeping when a garrison departs.
    Returns the number of units moved.  (QA 2026-07-09.)

    exclude_not_at_war: the voluntary-move border rule, per UNIT
    (amendment 1.222(b) via _unit_may_enter_nation) at the destination
    and at `via_san` (the walked-through region of a 2-region move)."""
    import game_logic as _gl
    from combat import side_of_nation as _son, _check_settlement_capture, \
        garrison_side_for
    # Almanac p17: card-granted moves obey A Power too Great / Tom Bombadil
    # like die-driven moves (audit 2026-07-26)
    if side == Side.SHADOW:
        from persistent_effects import can_shadow_move_to as _csm26
        _ok26, _ = _csm26(state, to_san)
        if not _ok26:
            return 0
    src_key = _find_army_key(state, frm_san)
    src = state.armies.get(src_key)
    if not src:
        return 0
    mv_units, stay_units = [], []
    for u in src.units:
        ok = _son(u.nation) == side
        if ok and exclude_not_at_war:
            # 1.222(b) per unit: At War, or the region is free / the
            # unit's own nation (passive units never cross ANOTHER
            # nation's border - destination AND walked-through region)
            ok = _unit_may_enter_nation(state, u, to_san) and (
                via_san is None
                or _unit_may_enter_nation(state, u, via_san))
        (mv_units if ok else stay_units).append(u)
    if n_regular is not None or n_elite is not None:
        # P4 (USER ruling 1.112.7): an INITIAL SPLIT - move only the
        # chosen composition, the remainder stays behind
        _take_r9 = n_regular if n_regular is not None else 10 ** 6
        _take_e9 = n_elite if n_elite is not None else 10 ** 6
        _mv9 = []
        for u in mv_units:
            if u.kind == UnitKind.ELITE and _take_e9 > 0:
                _mv9.append(u)
                _take_e9 -= 1
            elif u.kind != UnitKind.ELITE and _take_r9 > 0:
                _mv9.append(u)
                _take_r9 -= 1
            else:
                stay_units.append(u)
        mv_units = _mv9
    mv_leaders, stay_leaders = [], []
    for l in src.leaders:
        if (_gl._is_saruman(l) or _gl._leader_side(l) != side
                or (exclude_not_at_war and _gl.leader_must_follow(l)
                    and (not _unit_may_enter_nation(state, l, to_san)
                         or (via_san is not None
                             and not _unit_may_enter_nation(state, l, via_san))))):
            stay_leaders.append(l)
        else:
            mv_leaders.append(l)
    if cargo is not None:
        from army_card_cargo import selected
        if side != Side.SHADOW or n_regular is not None or n_elite is not None:
            raise ValueError('Exact Shadow cargo cannot mix with aggregate splits')
        exact_units, exact_leaders = selected(state, frm_san, cargo)
        if not set(map(id, exact_units)) <= set(map(id, mv_units)):
            raise ValueError('Selected cargo cannot enter destination')
        unit_ids, leader_ids = set(map(id, exact_units)), set(map(id, exact_leaders))
        mv_units, mv_leaders = exact_units, exact_leaders
        stay_units = [u for u in src.units if id(u) not in unit_ids]
        stay_leaders = [l for l in src.leaders if id(l) not in leader_ids]
    if not mv_units:
        return 0  # leaders alone cannot make the move
    if (any(_gl.leader_must_follow(l) and _gl._leader_side(l) == side
            for l in stay_leaders)
            and not any(_son(u.nation) == side for u in stay_units)):
        raise ValueError('Mandatory plain leader needs a unit left behind')
    dst_key = next((k for k in state.armies if sanitize(k) == to_san),
                   None)
    dst = state.armies.get(dst_key) if dst_key else None
    own_at_dst = sum(1 for u in (dst.units if dst else [])
                     if _son(u.nation) == side)
    # QUEUE_2 (2026-08-17): the cap is the ONE siege-aware body -
    # the flat 10 here let a card move overfill a besieged garrison
    room = _gl.army_stack_cap(state, to_san, side) - own_at_dst
    if allow_overstack:
        # P4 (USER ruling 1.112.7): the card move may overstack; the
        # region is recorded and resolve_overstacks() (learned pending
        # on the pending path, regulars-first heuristic otherwise) runs
        # right after the card effect completes
        if len(mv_units) > room and not defer_overstack:
            _ovr9 = getattr(state, '_overstack_regions', None)
            if _ovr9 is None:
                _ovr9 = set()
                state._overstack_regions = _ovr9
            _ovr9.add(to_san)
    else:
        if room <= 0:
            return 0
        if len(mv_units) > room:
            overflow = len(mv_units) - room
            keep = [u for u in mv_units
                    if u.kind == UnitKind.REGULAR][:overflow]
            if len(keep) < overflow:
                keep += [u for u in mv_units
                         if u.kind == UnitKind.ELITE][:overflow - len(keep)]
            _kb = set(map(id, keep))
            stay_units.extend(u for u in mv_units if id(u) in _kb)
            mv_units = [u for u in mv_units if id(u) not in _kb]
    # siege bookkeeping BEFORE mutating: garrison side of the source
    bs = getattr(state, 'besieged_strongholds', set())
    _g_side = garrison_side_for(state, frm_san) if frm_san in bs else None
    src.units = stay_units
    src.leaders = stay_leaders
    if not src.units and not src.leaders:
        state.armies.pop(src_key, None)
    if dst is None:
        dst_key = to_san
        state.armies[dst_key] = Army(region=state.board.region(to_san))
        dst = state.armies[dst_key]
    dst.units.extend(mv_units)
    dst.leaders.extend(mv_leaders)
    if _g_side == side:
        rem = state.armies.get(_find_army_key(state, frm_san))
        if not any(_son(u.nation) == side
                   for u in (rem.units if rem else [])):
            # the whole garrison left: the siege ends and the besieger
            # takes the empty stronghold
            bs.discard(frm_san)
            other = (Side.SHADOW if side == Side.FREE_PEOPLES
                     else Side.FREE_PEOPLES)
            if rem and any(_son(u.nation) == other for u in rem.units):
                _check_settlement_capture(state, frm_san, other)
    _gl._check_political_activation_for_entry(state, to_san, side)
    _gl._check_companion_activates_nation(state, to_san, mv_leaders)
    _gl.remove_orphaned_leaders(state)
    if not any(_son(u.nation) != side for u in dst.units):
        _check_settlement_capture(state, to_san, side)
    return len(mv_units)


def _army_reachable_free(state, start, max_steps, side):
    """Reachable destinations without crossing enemy armies or sieges.
    Friendly intermediate armies remain untouched: the moving group's cargo
    is fixed at action start. Besieger joins are terminal destinations."""
    from collections import deque
    from combat import side_of_nation as _son
    bs = getattr(state, 'besieged_strongholds', set())
    out = set()
    start_san = sanitize(start)
    seen = {start_san}
    q = deque([(start_san, 0)])
    while q:
        cur, d = q.popleft()
        if d >= max_steps:
            continue
        for adj in state.board.adjacent(cur):
            asan = sanitize(adj)
            if asan in seen:
                continue
            seen.add(asan)
            a = state.armies.get(_find_army_key(state, asan))
            has_enemy = bool(a and any(_son(u.nation) != side
                                       for u in a.units))
            has_own = bool(a and any(_son(u.nation) == side
                                     for u in a.units))
            # a besieged region where OUR side is the BESIEGER is a
            # legal JOIN destination - the enemy garrison is walled in
            # and does not block entry (GAME19 T5 TSL: Moria elites
            # reinforce the Lorien siege).  Our side being the walled-in
            # GARRISON is NOT joinable: that would reinforce a besieged
            # stronghold through the siege lines.
            from combat import garrison_side_for as _gsf_join
            _bes_join9 = (asan in bs and has_own
                          and _gsf_join(state, asan) != side)
            if (has_enemy or asan in bs) and not _bes_join9:
                continue  # cannot enter at all
            out.add(asan)
            if asan not in bs:
                q.append((asan, d + 1))
    return out


def _card_move_needs_guard(state, army, to_san, via_san=None):
    """A politically barred FP plain leader must retain an Army unit."""
    import game_logic as _gl
    return any(_gl.leader_must_follow(l)
               and _gl._leader_side(l) == Side.FREE_PEOPLES
               and (not _unit_may_enter_nation(state, l, to_san)
                    or (via_san is not None
                        and not _unit_may_enter_nation(state, l, via_san)))
               for l in army.leaders)


def _paths_of_woses_options(state):
    """Legal {from_region, to_region} choices per almanac A11: source =
    any Rohan region with FP units (a besieged Helm's Deep counts);
    destination = Minas Tirith when it is neither besieged, captured,
    nor Shadow-occupied - else whichever of Druadan Forest / Osgiliath /
    Lossarnach holds no Shadow Army units (empty Shadow-CONTROLLED
    regions are legal and are captured by the move)."""
    _ec_sh = state.enemy_controlled.get(Side.SHADOW, set())
    bs = getattr(state, 'besieged_strongholds', set())

    def _sh_units_at(san):
        a = state.armies.get(_find_army_key(state, san))
        return bool(a and any(u.nation in SHADOW_NATIONS
                              for u in a.units))

    mt = 'minas tirith'
    if mt not in bs and mt not in _ec_sh and not _sh_units_at(mt):
        dests = [mt]  # "must move to Minas Tirith if possible"
    else:
        dests = [r for r in ('druadan forest', 'osgiliath', 'lossarnach')
                 if r in board.regions and r not in bs
                 and not _sh_units_at(r)]
    if not dests:
        return []
    opts = []
    for k, a in state.armies.items():
        san = sanitize(k)
        meta = board.regions.get(san, {})
        if board.fid_to_name.get(meta.get('parent_id')) != 'rohan':
            continue
        if not any(u.nation in FP_NATIONS for u in a.units):
            continue
        # (the per-unit 1.222(b) border rule lives in the MOVER; an army
        # none of whose units may enter completes as a zero-effect play
        # per 1.230 - never pruned from the surface, CL-239)
        fp_units = [u for u in a.units if u.nation in FP_NATIONS]
        for d in dests:
            if d == san:
                continue
            pool = [u for u in fp_units if _unit_may_enter_nation(state, u, d)]
            if (pool and len(pool) == len(fp_units)
                    and _card_move_needs_guard(state, a, d)):
                # The mandatory leader stays; expose the legal initial splits
                # instead of a whole-stack choice that would orphan it.
                nr = sum(u.kind != UnitKind.ELITE for u in pool)
                ne = len(pool) - nr
                for mr in range(nr + 1):
                    for me in range(ne + 1):
                        if (mr + me == 0 and len(pool) > 1) or (mr == nr and me == ne):
                            continue
                        opts.append(dict(from_region=san, to_region=d,
                                         n_regular=mr, n_elite=me))
            else:
                opts.append({"from_region": san, "to_region": d})
    return opts


def _tdaan_path(state, frm, to, units, n_regular=None, n_elite=None):
    """Through a Day and a Night path for the FP `units` at `frm` moving
    to `to` (<= 2 regions): returns (mid, pool) - `pool` is the subset
    that may make the move under the per-unit 1.222(b) border rule at
    the destination AND the walked-through region, `mid` the free
    (friendly or empty, unbesieged) intermediate (None when adjacent). (None,
    []) when no legal selected group may move. Each route is checked
    against the chosen composition and any plain leader left behind;
    ties prefer routes carrying more of the Army's leaders. ONE body
    for the option surface and the handler."""
    frm, to = sanitize(frm), sanitize(to)
    adj_frm = {sanitize(x) for x in state.board.adjacent(frm)}
    candidates = []
    if to in adj_frm:
        candidates.append(None)
    else:
        bs = getattr(state, 'besieged_strongholds', set())
        for m in sorted(adj_frm):
            if m in bs:
                continue
            a_m = state.armies.get(_find_army_key(state, m))
            if a_m and any(u.nation not in FP_NATIONS for u in a_m.units):
                continue
            if to in {sanitize(x) for x in state.board.adjacent(m)}:
                candidates.append(m)
    import game_logic as _gl
    army = state.armies.get(_find_army_key(state, frm))
    best, best_score = (None, []), (-1, -1)
    for m in candidates:
        pool = [u for u in units
                if _unit_may_enter_nation(state, u, to)
                and (m is None or _unit_may_enter_nation(state, u, m))]
        nr = sum(u.kind != UnitKind.ELITE for u in pool)
        ne = len(pool) - nr
        mr = nr if n_regular is None else n_regular
        me = ne if n_elite is None else n_elite
        if mr < 0 or me < 0 or mr > nr or me > ne or mr + me == 0:
            continue
        if (army and mr + me == len(units)
                and _card_move_needs_guard(state, army, to, m)):
            continue
        leaders = sum(_gl.leader_must_follow(l)
                      and _gl._leader_side(l) == Side.FREE_PEOPLES
                      and _unit_may_enter_nation(state, l, to)
                      and (m is None or _unit_may_enter_nation(state, l, m))
                      for l in (army.leaders if army else []))
        score = (len(pool), leaders)
        if score > best_score:
            best, best_score = (m, pool), score
    return best


def _tdaan_options(state):
    """Through a Day and a Night choices: FP armies containing at least
    one Companion (never a besieged garrison - 1.211h, the siege rule
    the Shadow movement cards already honour); destinations within 2
    regions via friendly or empty, unbesieged intermediates, destination
    free of enemy units; the composition grid is built from the pool
    that may legally make the move (1.222(b))."""
    from game_state import Companion as _Comp
    opts = []
    for k, a in state.armies.items():
        if not a.units:
            continue
        if not any(u.nation in FP_NATIONS for u in a.units):
            continue
        if not any(isinstance(l, _Comp) for l in a.leaders):
            continue
        frm = sanitize(k)
        if _besieged_garrison_source(state, frm, Side.FREE_PEOPLES):
            continue
        fp_units = [u for u in a.units if u.nation in FP_NATIONS]
        for dest in sorted(_army_reachable_free(
                state, frm, 2, Side.FREE_PEOPLES)):
            _mid, pool = _tdaan_path(state, frm, dest, fp_units)
            full_r = sum(u.kind != UnitKind.ELITE for u in pool)
            full_e = len(pool) - full_r
            if pool:
                opts.append({"from_region": frm, "to_region": dest})
            # P4 (USER ruling 1.112.7): initial splits - the full
            # composition grid, like army_move; the moving group takes
            # the leaders/companions with it, the remainder stays
            n_r = sum(u.kind != UnitKind.ELITE for u in fp_units)
            n_e = len(fp_units) - n_r
            for mr in range(n_r + 1):
                for me in range(n_e + 1):
                    if mr + me == 0 or (pool and mr == full_r and me == full_e):
                        continue
                    if not _tdaan_path(state, frm, dest, fp_units, mr, me)[1]:
                        continue
                    opts.append({"from_region": frm, "to_region": dest,
                                 "n_regular": mr, "n_elite": me})
    return opts


@register_effect("Help Unlooked For")
def _effect_help_unlooked_for(state: "GameState", card, choices: dict) -> dict:
    """Attack a Shadow Army BESIEGING a stronghold with an adjacent FP
    Army; the Shadow player's combat dice are reduced by one per FP
    Army unit inside the stronghold, minimum 1 (almanac A10).

    QA 2026-07-09 root fixes: the roles were INVERTED (the engine had
    the Shadow army attack the stronghold) and the dice reduction was
    passed under a choices key nothing consumed.  Now the FP army
    ATTACKS the besieging Shadow army - setup_combat's away-attack
    split makes the Shadow portion of the shared siege entry the
    defender and leaves the FP garrison out of the fight - with
    defender_dice_reduction wired through combat.  The attacking army
    must contain at least one At-War unit.  Multi-round interactive
    field battle per USER 1.210(a) ("absolutely does not resolve in a
    single round"): the dice reduction lasts until the combat ends;
    full press/retreat/advance set.
    """
    from combat import garrison_side_for
    result = {"effects": []}

    # TEACHER path: the schedule's own army_attack record synthesizes the
    # relief battle; opening it here would pre-empt that cursor (no board
    # mutation precedes this gate).  Self-play keeps the interactive field
    # battle (USER 1.210(a)).
    import combat as _cmbx9
    if _cmbx9.FINAL_HITS_HOOK is not None:
        return result

    def_region = choices.get("stronghold")
    atk_region = choices.get("attack_from")

    def _atwar_fp_army(a):
        from game_logic import can_attack_with_army
        return bool(a and can_attack_with_army(state, a, side=Side.FREE_PEOPLES)[0])

    # --- heuristic: FP-garrisoned besieged stronghold + adjacent At-War
    # FP relief army (the card only applies to armies BESIEGING) -------
    if not def_region or not atk_region:
        best = None
        for rsan in sorted(getattr(state, 'besieged_strongholds', set())):
            if garrison_side_for(state, rsan) != Side.FREE_PEOPLES:
                continue
            entry = state.armies.get(_find_army_key(state, rsan))
            if not entry or not any(u.nation in SHADOW_NATIONS
                                    for u in entry.units):
                continue
            for adj in state.board.adjacent(rsan):
                adj_san = sanitize(adj)
                fp_a = state.armies.get(_find_army_key(state, adj_san))
                if _atwar_fp_army(fp_a):
                    sc = sum(1 for u in fp_a.units
                             if u.nation not in SHADOW_NATIONS)
                    if best is None or sc > best[0]:
                        best = (sc, rsan, adj_san)
        if best:
            _, def_region, atk_region = best

    if not def_region or not atk_region:
        result["message"] = "No besieged FP stronghold with an adjacent At-War FP army"  # no-op: NOT an effect (iter 39)
        return result

    def_san = sanitize(def_region)
    atk_san = sanitize(atk_region)

    atk_army = state.armies.get(_find_army_key(state, atk_san))
    if not _atwar_fp_army(atk_army):
        result["message"] = "Attacking FP army must contain At-War units"
        return result

    # dice reduction = FP ARMY UNITS inside the besieged stronghold
    # (Leaders/Companions are not Army units and do not count)
    sh_entry = state.armies.get(_find_army_key(state, def_san))
    fp_unit_count = sum(1 for u in (sh_entry.units if sh_entry else [])
                        if u.nation not in SHADOW_NATIONS)

    # 1.210(a) USER: Help Unlooked For "absolutely does not resolve in a single
    # round" - a MULTI-ROUND field battle whose Shadow dice reduction (one per
    # FP unit inside the stronghold, min 1) lasts until the combat ends; full
    # press/retreat/advance set; "if shadow cannot retreat, free people can
    # keep going as long as they want."
    import ai as _ai9
    _pc9 = {
        'attacker_region': atk_san,
        'defender_region': def_san,
        'is_siege': False,
        'stronghold_region': None,
        'phase': 'combat_setup',
        'deciding_side': Side.FREE_PEOPLES,
        'attacker_side': Side.FREE_PEOPLES,
        'defender_side': Side.SHADOW,
        'rounds_fought': 0,
        'def_dice_reduction': fp_unit_count,
    }
    state.pending_combat = _pc9
    _ai9._run_combat_round(state, _pc9)
    result["effects"].append(
        f"Help Unlooked For: interactive field battle {atk_san} "
        f"-> besiegers at {def_san} (Shadow dice -{fp_unit_count})")
    return result


@register_effect("Paths of the Woses")
def _effect_paths_of_woses(state: "GameState", card, choices: dict) -> dict:
    """Move one FP Army DIRECTLY from a Rohan region to Minas Tirith
    (fallback: Druadan Forest / Osgiliath / Lossarnach when MT is
    besieged, captured or Shadow-occupied) - almanac A11: "as if a
    Rohan region is adjacent ... it moves directly".  QA 2026-07-09:
    the old 2-hop BFS was the wrong mechanic and the old gate (FP army
    IN Minas Tirith) the wrong requirement.  Units of not-At-War
    non-Rohan nations stay behind (A11)."""
    result = {"effects": []}
    frm = choices.get("from_region")
    to = choices.get("to_region")
    opts = _paths_of_woses_options(state)
    if frm is None or to is None:
        if not opts:
            result["message"] = "No legal Paths of the Woses move"
            return result
        choices = dict(choices, **opts[0])
        frm, to = choices["from_region"], choices["to_region"]
    frm, to = sanitize(frm), sanitize(to)
    if not any(o["from_region"] == frm and o["to_region"] == to
               for o in opts):
        result["message"] = f"Illegal Paths of the Woses move {frm} to {to}"
        return result
    moved = _direct_army_move(state, frm, to, Side.FREE_PEOPLES,
                              exclude_not_at_war=True, allow_overstack=True,
                              n_regular=choices.get('n_regular'),
                              n_elite=choices.get('n_elite'))
    if moved:
        result["effects"].append(
            f"Paths of the Woses: army moved from {frm} to {to}")
    else:
        result["message"] = f"No movable units in {frm}"
    return result


@register_effect("Through a Day and a Night")
def _effect_through_day_and_night(state: "GameState", card, choices: dict) -> dict:
    """Move one FP Army CONTAINING A COMPANION up to two regions
    (almanac A12).  QA 2026-07-09 root fixes: the gate wrongly looked
    for Rohan units and never for a Companion; the BFS ignored
    enemy-occupied intermediates; the 2-step cap was off-by-one; and a
    mid-path failure stranded the army.  The move now validates against
    _army_reachable_free and executes as one atomic direct transfer."""
    result = {"effects": []}
    frm = choices.get("from_region")
    to = choices.get("to_region")
    if frm is None or to is None:
        opts = _tdaan_options(state)
        if not opts:
            result["message"] = "No FP army with a Companion can move"
            return result
        choices = dict(choices, **opts[0])
        frm, to = choices["from_region"], choices["to_region"]
    frm, to = sanitize(frm), sanitize(to)
    from game_state import Companion as _Comp
    src = state.armies.get(_find_army_key(state, frm))
    if not src or not any(isinstance(l, _Comp) for l in src.leaders):
        result["message"] = f"No FP army with a Companion in {frm}"
        return result
    # 1.211h: a besieged FP GARRISON cannot march out through the siege
    # lines (sortie is a battle) - the guard the Shadow cards use
    if _besieged_garrison_source(state, frm, Side.FREE_PEOPLES):
        result["message"] = (
            f"{frm} is a besieged garrison - cannot move out")
        return result
    if to not in _army_reachable_free(state, frm, 2, Side.FREE_PEOPLES):
        result["message"] = (
            f"{to} not reachable within 2 free regions from {frm}")
        return result
    # walk-through capture (USER RULE 2026-07-08: walking through empty
    # settlements captures them) - the free intermediate, if any, and
    # the pool that may legally move (1.222(b)); ONE body with the
    # option surface (_tdaan_path)
    _nr9 = choices.get('n_regular')
    _ne9 = choices.get('n_elite')
    _nr9 = int(_nr9) if _nr9 is not None else None
    _ne9 = int(_ne9) if _ne9 is not None else None
    _mid, _pool = _tdaan_path(
        state, frm, to, [u for u in src.units if u.nation in FP_NATIONS],
        _nr9, _ne9)
    if not _pool:
        result["message"] = (
            f"No unit in {frm} may enter {to} (non-belligerent border "
            f"rule, 1.222b)")
        return result
    moved = _direct_army_move(
        state, frm, to, Side.FREE_PEOPLES,
        exclude_not_at_war=True, via_san=_mid,
        n_regular=(int(_nr9) if _nr9 is not None else None),
        n_elite=(int(_ne9) if _ne9 is not None else None),
        # P4/P1: overstack-then-learned-resolution on the PENDING path
        # only; the teacher keeps _direct_army_move's certified clamp
        # (excess stays at the source - same lesson as TSiM/amendment
        # 1.123: changing teacher bookkeeping reshapes collection)
        allow_overstack=bool(choices.get('_from_pending')))
    if moved:
        if _mid is not None:
            from combat import _check_settlement_capture
            import game_logic as _gl_td
            _check_settlement_capture(state, _mid, Side.FREE_PEOPLES)
            # rulebook 370-371: a walked-through region is ENTERED
            _gl_td._check_political_activation_for_entry(
                state, _mid, Side.FREE_PEOPLES)
        result["effects"].append(
            f"Through a Day and a Night: army moved from {frm} to {to}"
            + (f" ({_nr9}R+{_ne9}E split)" if _nr9 is not None else ""))
    else:
        result["message"] = f"No movable units in {frm}"
    return result


def _cirdan_unit_combos(state):
    """Supply-valid full recruitment alternatives; AI projects Elite-first."""
    from ai import figures_available
    regular = figures_available(state, Nation.ELVES, 'regular')
    elite = figures_available(state, Nation.ELVES, 'elite')
    total = min(2, regular + elite)
    if total == 0:
        return []
    if total == 1:
        return ['2elite'] if elite else ['2regular']
    return [name for name, nr, ne in (('2regular', 2, 0), ('mixed', 1, 1), ('2elite', 0, 2))
            if regular >= nr and elite >= ne]


@register_effect("Cirdan's Ships")
def _effect_cirdans_ships(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 Elven units in coastal region with FP army."""
    result = {"effects": []}
    region = choices.get("region")
    # LOW-25/26 + USER RULING 2026-07-21: self-play validates the target (an
    # authoritative coastal region holding an FP army) and refunds a no-op;
    # the teacher trusts the logged region.
    if not region:
        result["message"] = ("Cirdan's Ships: no coastal region "
                             "chosen - not played")
        return _selfplay_refund(result)
    from card_options import _COASTAL
    _rs = sanitize(str(region))
    _a9 = next((a for k, a in state.armies.items()
                if sanitize(k) == _rs), None)
    # USER 2026-07-21g: test FP PRESENCE (any FP unit) - a besieged FP
    # stronghold shares its entry with the Shadow besiegers (recruit under
    # siege is legal; the overstack machinery trims with the owner's choice).
    _fp_present = bool(_a9 and _a9.units and any(
        u.nation not in SHADOW_NATIONS for u in _a9.units))
    if _rs not in _COASTAL or not _fp_present:
        result["message"] = ("Cirdan's Ships: %s is not a coastal "
                             "region with a Free Peoples army - "
                             "not played" % region)
        return _selfplay_refund(result)
    combo = _default_unit_combo(state, Nation.ELVES, choices)
    if choices.get('unit_combo') is not None and combo not in _cirdan_unit_combos(state):
        raise ValueError("Cirdan's Ships recruitment combination unavailable")
    _got = []
    if combo == "2regular":
        _got = [recruit_unit(state, region, Nation.ELVES,
                             UnitKind.REGULAR) for _ in range(2)]
        _n = sum(1 for g in _got if g is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} Elven Regular(s) in {region}")
    elif combo == "2elite":
        _got = [recruit_unit(state, region, Nation.ELVES,
                             UnitKind.ELITE) for _ in range(2)]
        _n = sum(1 for g in _got if g is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} Elven Elite(s) in {region}")
    else:  # mixed
        _got = [recruit_unit(state, region, Nation.ELVES,
                             UnitKind.REGULAR),
                recruit_unit(state, region, Nation.ELVES,
                             UnitKind.ELITE)]
        _n = sum(1 for g in _got if g is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} Elven unit(s) in {region}")
    if not result["effects"]:
        result["message"] = ("Cirdan's Ships: Elven pool exhausted "
                             "- nothing recruited")
        return _selfplay_refund(result)
    return result


def _validate_guards_recruit_choice(state: "GameState", choices: dict) -> None:
    """Validate an explicit Guards choice before payment or placement."""
    if not isinstance(choices, dict):
        raise ValueError("Guards of the Citadel choices must be a dictionary")
    if 'region' in choices and sanitize(str(choices['region'])) != 'minas tirith':
        raise ValueError("Guards of the Citadel recruits only in Minas Tirith")
    if 'unit_type' in choices:
        from card_options import guards_of_the_citadel_options
        available = {o.get('unit_type') for o in
                     guards_of_the_citadel_options(state, Side.FREE_PEOPLES)
                     if o.get('unit_type') is not None}
        if choices['unit_type'] not in ('regular', 'elite') \
                or choices['unit_type'] not in available:
            raise ValueError("Guards of the Citadel unit_type is not available")


@register_effect("Guards of the Citadel")
def _effect_guards_of_citadel(state: "GameState", card, choices: dict) -> dict:
    """Recruit the chosen Gondor unit and independent Leader in Minas Tirith.

    The public action path exposes Regular/Elite through the card spec.
    Direct legacy callers without a choice retain their default; an explicit
    malformed or unavailable kind is never silently replaced or omitted.
    """
    _validate_guards_recruit_choice(state, choices)
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.GONDOR, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # Report only what actually happened (recruit_unit returns None when the
    # cap or pool blocks).  LOW-30: unit and Leader are placed independently -
    # unit FIRST so the Leader's no-stand-alone guard passes.
    _u = recruit_unit(state, "Minas Tirith", Nation.GONDOR, kind)
    _l = recruit_leader(state, "Minas Tirith", Nation.GONDOR)
    if _u is not None:
        result["effects"].append(
            f"Recruited Gondor {unit_type} in Minas Tirith")
    if _l is not None:
        result["effects"].append("Recruited Gondor Leader in Minas Tirith")
    if not result["effects"]:
        result["message"] = ("Minas Tirith placement blocked "
                             "(stacking cap or pool)")
        return _selfplay_refund(result)
    return result


@register_effect("Celeborn's Galadhrim")
def _effect_celeborns_galadhrim(state: "GameState", card, choices: dict) -> dict:
    """Recruit Elven unit in Lórien, draw 1 Strategy card."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.ELVES, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # CARD-FLEET LOW-24: honest per-piece reporting; refund when EVERY
    # sub-effect no-ops (self-play only)
    _u = recruit_unit(state, "Lorien", Nation.ELVES, kind)
    if _u is not None:
        result["effects"].append(f"Recruited Elven {unit_type} in Lórien")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 1)
    if drawn:
        result["effects"].append(f"Drew {len(drawn)} Strategy card")
    if not result["effects"]:
        result["message"] = ("Celeborn's Galadhrim: no legal effect "
                             "(Elven pool exhausted, no card to draw)")
        return _selfplay_refund(result)
    return result


@register_effect("Riders of Théoden")
def _effect_riders_of_theoden(state: "GameState", card, choices: dict) -> dict:
    """Recruit Rohan unit and Leader in Edoras or companion region."""
    result = {"effects": []}
    region = choices.get("region", "Edoras")
    # 1.211c: "or any Rohan region CONTAINING a Companion" is a PER-REGION gate:
    # Edoras (if free) UNION every Rohan region holding a BOARD Companion -
    # Helm's Deep qualifies even under siege, any other must be free for the
    # FP.  Fellowship companions are not on the board (1.211l).
    from game_state import Companion as _Comp9
    from card_options import _free_for_side as _ffs9
    _rsan9 = sanitize(region)
    _hd9 = sanitize("Helm's Deep")
    if _rsan9 == sanitize("Edoras"):
        _ok9 = _ffs9(state, _rsan9, Side.FREE_PEOPLES)
    else:
        _a9 = state.armies.get(_find_army_key(state, _rsan9))
        _rohan9 = board.fid_to_name.get(
            board.regions.get(_rsan9, {}).get('parent_id')) == 'rohan'
        _has_comp9 = bool(_a9 and any(
            isinstance(l, _Comp9) for l in _a9.leaders))
        _ok9 = (_rohan9 and _has_comp9
                and (_rsan9 == _hd9
                     or _ffs9(state, _rsan9, Side.FREE_PEOPLES)))
    if not _ok9:
        result["message"] = ("Riders of Theoden: target must be Edoras "
                             "or a Rohan region with a Companion")
        return _selfplay_refund(result)
    unit_type = _default_unit_type(state, Nation.ROHAN, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _u = recruit_unit(state, region, Nation.ROHAN, kind)
    _l = recruit_leader(state, region, Nation.ROHAN)
    if _u is not None:
        result["effects"].append(f"Recruited Rohan {unit_type} in {region}")
    if _l is not None:
        result["effects"].append(f"Recruited Rohan Leader in {region}")
    if not result["effects"]:
        result["message"] = "Riders of Theoden: recruitment blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Grimbeorn the Old Son of Beorn")
def _effect_grimbeorn(state: "GameState", card, choices: dict) -> dict:
    """Recruit North unit and Leader in Carrock."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.THE_NORTH, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # CARD-FLEET LOW-29: honest per-piece reporting + refund when both
    # pieces are blocked (self-play only)
    _u = recruit_unit(state, "Carrock", Nation.THE_NORTH, kind)
    _l = recruit_leader(state, "Carrock", Nation.THE_NORTH)
    if _u is not None:
        result["effects"].append(f"Recruited North {unit_type} in Carrock")
    if _l is not None:
        result["effects"].append("Recruited North Leader in Carrock")
    if not result["effects"]:
        result["message"] = ("Grimbeorn: Carrock recruitment blocked "
                             "(pool exhausted or captured)")
        return _selfplay_refund(result)
    return result


@register_effect("Imrahil of Dol Amroth")
def _effect_imrahil(state: "GameState", card, choices: dict) -> dict:
    """Recruit Gondor unit and Leader in Dol Amroth."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.GONDOR, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _u = recruit_unit(state, "Dol Amroth", Nation.GONDOR, kind)
    _l = recruit_leader(state, "Dol Amroth", Nation.GONDOR)
    if _u is not None:
        result["effects"].append(
            f"Recruited Gondor {unit_type} in Dol Amroth")
    if _l is not None:
        result["effects"].append("Recruited Gondor Leader in Dol Amroth")
    if not result["effects"]:
        result["message"] = "Imrahil: Dol Amroth recruitment blocked"
        return _selfplay_refund(result)
    return result


@register_effect("King Brand's Men")
def _effect_king_brands_men(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 North regulars in Dale, draw 1 Strategy card."""
    result = {"effects": []}
    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _n = sum(1 for _ in range(2)
             if recruit_unit(state, "Dale", Nation.THE_NORTH,
                             UnitKind.REGULAR) is not None)
    if _n:
        result["effects"].append(f"Recruited {_n} North Regular(s) in Dale")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 1)
    if drawn:
        result["effects"].append(f"Drew {len(drawn)} Strategy card")
    if not result["effects"]:
        result["message"] = "King Brand's Men: no legal effect"
        return _selfplay_refund(result)
    return result


@register_effect("Swords in Eriador")
def _effect_swords_in_eriador(state: "GameState", card, choices: dict) -> dict:
    """Recruit North in Shire, Dwarven in Ered Luin, draw 1 Strategy card."""
    result = {"effects": []}
    north_type = _default_unit_type(state, Nation.THE_NORTH, choices, key="north_unit_type")
    dwarf_type = _default_unit_type(state, Nation.DWARVES, choices, key="dwarf_unit_type")

    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _n1 = recruit_unit(state, "The Shire", Nation.THE_NORTH,
                       UnitKind.ELITE if north_type == "elite"
                       else UnitKind.REGULAR)
    _n2 = recruit_unit(state, "Ered Luin", Nation.DWARVES,
                       UnitKind.ELITE if dwarf_type == "elite"
                       else UnitKind.REGULAR)
    if _n1 is not None:
        result["effects"].append(f"Recruited North {north_type} in Shire")
    if _n2 is not None:
        result["effects"].append(
            f"Recruited Dwarven {dwarf_type} in Ered Luin")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 1)
    if drawn:
        result["effects"].append(f"Drew {len(drawn)} Strategy card")
    if not result["effects"]:
        result["message"] = "Swords in Eriador: no legal effect"
        return _selfplay_refund(result)
    return result


@register_effect("Kindred of Glorfindel")
def _effect_kindred_of_glorfindel(state: "GameState", card, choices: dict) -> dict:
    """Recruit Elven unit in Rivendell, draw 1 Strategy card."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.ELVES, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _u = recruit_unit(state, "Rivendell", Nation.ELVES, kind)
    if _u is not None:
        result["effects"].append(
            f"Recruited Elven {unit_type} in Rivendell")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 1)
    if drawn:
        result["effects"].append(f"Drew {len(drawn)} Strategy card")
    if not result["effects"]:
        result["message"] = "Kindred of Glorfindel: no legal effect"
        return _selfplay_refund(result)
    return result


@register_effect("Dain Ironfoot's Guard")
def _effect_dain_ironfoots_guard(state: "GameState", card, choices: dict) -> dict:
    """Recruit Dwarven unit and Leader in Erebor."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.DWARVES, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # CARD-FLEET LOW-27: honest per-piece reporting + refund when both
    # pieces are blocked (self-play only)
    _u = recruit_unit(state, "Erebor", Nation.DWARVES, kind)
    _l = recruit_leader(state, "Erebor", Nation.DWARVES)
    if _u is not None:
        result["effects"].append(f"Recruited Dwarven {unit_type} in Erebor")
    if _l is not None:
        result["effects"].append("Recruited Dwarven Leader in Erebor")
    if not result["effects"]:
        result["message"] = ("Dain Ironfoot's Guard: Erebor recruitment "
                             "blocked (pool exhausted or captured)")
        return _selfplay_refund(result)
    return result


@register_effect("Éomer son of Éomund")
def _effect_eomer(state: "GameState", card, choices: dict) -> dict:
    """Recruit Rohan unit and Leader in free Rohan settlement.
    Exception: cannot recruit into Helm's Deep while it is under siege."""
    result = {"effects": []}
    region = choices.get("region", "Edoras")
    # 1.211b: a besieged Helm's Deep is not a legal Eomer target - refund an
    # illegal pick, never redirect it to Edoras; the teacher keeps log truth.
    if sanitize(region) not in {
            sanitize(r) for r in _get_free_rohan_settlements(
                state, exclude_besieged=True)}:
        result["message"] = "Eomer: target must be a free Rohan settlement"
        return _selfplay_refund(result)
    unit_type = _default_unit_type(state, Nation.ROHAN, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # honest per-piece reporting + fully-blocked refund (self-play only)
    _u = recruit_unit(state, region, Nation.ROHAN, kind)
    _l = recruit_leader(state, region, Nation.ROHAN)
    if _u is not None:
        result["effects"].append(f"Recruited Rohan {unit_type} in {region}")
    if _l is not None:
        result["effects"].append(f"Recruited Rohan Leader in {region}")
    if not result["effects"]:
        result["message"] = "Eomer: recruitment blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Thranduil's Archers")
def _effect_thranduils_archers(state: "GameState", card, choices: dict) -> dict:
    """Recruit Elven unit in Woodland Realm, draw 1 Strategy card."""
    result = {"effects": []}
    unit_type = _default_unit_type(state, Nation.ELVES, choices)
    kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
    # honest per-piece reporting (LOW-24/27/29/30 family, 2026-07-21)
    _u = recruit_unit(state, "Woodland Realm", Nation.ELVES, kind)
    if _u is not None:
        result["effects"].append(
            f"Recruited Elven {unit_type} in Woodland Realm")

    drawn = draw_cards(state, Side.FREE_PEOPLES, "strategy", 1)
    if drawn:
        result["effects"].append(f"Drew {len(drawn)} Strategy card")
    if not result["effects"]:
        result["message"] = "Thranduil's Archers: no legal effect"
        return _selfplay_refund(result)
    return result


# ---------------------------------------------------------------------------
# Registered Card Effect Handlers - Shadow Character Cards
# ---------------------------------------------------------------------------

@register_effect("Shelob's Lair")
def _effect_shelobs_lair(state: "GameState", card, choices: dict) -> dict:
    """Add Shelob's Lair hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "shelob's lair"):
        result["effects"].append(
            "Added special Hunt tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Special Hunt tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("The Ring is Mine!")
def _effect_ring_is_mine(state: "GameState", card, choices: dict) -> dict:
    """Add The Ring is Mine! hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "the ring is mine!"):
        result["effects"].append(
            "Added special Hunt tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Special Hunt tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("On On They Went")
def _effect_on_on_they_went(state: "GameState", card, choices: dict) -> dict:
    """Add On On They Went hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "on on they went"):
        result["effects"].append(
            "Added special Hunt tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Special Hunt tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Give it to Uss!")
def _effect_give_it_to_uss(state: "GameState", card, choices: dict) -> dict:
    """Add Give it to Uss! hunt tile to pool when Fellowship on Mordor Track."""
    result = {"effects": []}
    if stage_special_hunt_tile(state, "give it to uss!"):
        result["effects"].append(
            "Added special Hunt tile to the Hunt Pool")
    else:
        result["effects"].append(
            "Special Hunt tile set aside (enters the pool on Mordor entry)")
    return result


@register_effect("Orc Patrol")
def _effect_orc_patrol(state: "GameState", card, choices: dict) -> dict:
    """Draw hunt tile; discard if Eye or FP special, else apply hunt."""
    result = {"effects": []}
    # hunt-card-draws (QA 2026-07-11): tile source is either a redrawn
    # tile forced by the Mithril continuation, or a fresh Shadow draw.
    _ftile = choices.get('_forced_tile') if choices else None
    if (_ftile is None and not (choices or {}).get('_skip_wizard')
            and _wizardstaff_interposes(state)):
        _wizardstaff_stash_card(state, 'orc_patrol', choices)
        result['effects'].append("Wizard's Staff prevent decision pending")
        return result
    if _ftile is None and not state.hunt_pool:
        import game_logic as _gl136
        _gl136._recycle_hunt_pool(state)   # 1.136 (USER RULE)
    if state.hunt_pool or _ftile is not None:
        if _ftile is not None:
            tile = _ftile
        else:
            tile = _tile_choice(state)
            state.hunt_pool.remove(tile)
            # Mithril Coat and Sting redraw interposes on THIS Shadow
            # draw (learned path only); the apply-continuation runs from
            # _effect_mithril_coat keyed by apply_kind='orc_patrol'.
            if _mithril_interposes(state):
                _mithril_stash_card(state, tile, 'orc_patrol', choices)
                return result
        state.hunt_discard.append(tile)  # drawn = used pile (returns on entry)
        try:
            from game_trace import emit_annotation as _ea132
            _ea132(state, 'hunt_tile_draw', {
                'tile_name': (tile.special_name if tile.special_name
                              else repr(tile)),
                'ctx': 'Orc Patrol',
                'damage': int(getattr(tile, 'damage', 0) or 0)})
        except Exception:
            pass


        # Closed loop passes the LOGGED tile (the random draw is self-play only)
        _fdmg = choices.get("damage")
        if _fdmg is not None:
            damage, reveal = int(_fdmg), bool(choices.get("reveal"))
            result["effects"].append(
                f"Forced hunt tile: {damage} damage, reveal={reveal}")
            vname = str(choices.get("victim", "") or "").strip().lower()
            victim = next((c for c in state.fellowship.companions
                           if c.name.strip().lower() == vname), None)                 if vname else None
            if victim is not None and damage > 0:
                import game_logic as _gl
                _gl._apply_fellowship_damage(state, damage,
                                             kill_name=victim.name)
                result["effects"].append(f"{victim.name} sacrificed")
            else:
                state.fellowship.corruption = min(12, state.fellowship.corruption + damage)  # CL-217a: 12-ceiling clamp (2026-07-21)
            if reveal:
                state.fellowship.hidden = False
        elif getattr(tile, 'is_eye', False):
            result["effects"].append("Drew Eye tile - discarded without effect")
        elif getattr(tile, 'is_fellowship_special', False):
            result["effects"].append("Drew Fellowship special tile - discarded without effect")
        else:
            # Apply hunt
            damage = _card_tile_damage(tile)
            reveal = getattr(tile, 'reveal', False)
            result["effects"].append(f"Drew hunt tile: {damage} damage, reveal={reveal}")
            # Closed loop passes the LOGGED resolution: the humans may
            # sacrifice a companion instead of taking corruption
            # (_apply_fellowship_damage handles level absorption; no RNG
            # either way)
            vname = str(choices.get("victim", "") or "").strip().lower()
            victim = next((c for c in state.fellowship.companions
                           if c.name.strip().lower() == vname), None)                 if vname else None
            import game_logic as _gl
            if victim is not None and damage > 0:
                _gl._apply_fellowship_damage(state, damage,
                                             kill_name=victim.name)
                result["effects"].append(f"{victim.name} sacrificed")
            elif choices.get("damage") is None and choices.get(
                    "victim") is None and "victim" not in choices                     and not choices.get("_from_log"):
                # SELF-PLAY (no logged resolution): route through the learned
                # hunt-damage decision instead of raw corruption (USER
                # 2026-07-09)
                state._pending_hunt_damage = getattr(
                    state, '_pending_hunt_damage', 0) + damage
            else:
                state.fellowship.corruption = min(12, state.fellowship.corruption + damage)  # CL-217a: 12-ceiling clamp (2026-07-21)
            _gl.stage_hunt_reveal(state, tile, reveal)
            _gl.settle_hunt_reveal(state)
    return result


@register_effect("Isildur's Bane")
def _effect_isildurs_bane(state: "GameState", card, choices: dict) -> dict:
    """Draw hunt tile; if valid, all damage as Corruption."""
    result = {"effects": []}
    # hunt-card-draws (QA 2026-07-11): forced (Mithril redraw) or drawn.
    _ftile = choices.get('_forced_tile') if choices else None
    # 1.217 (USER 2026-07-23): the FP may discard "Wizard's Staff" to
    # PREVENT this card's Hunt-tile DRAW entirely (a PRE-draw nullify -
    # distinct from post-draw damage reduction, which Isildur's Bane never
    # offers). Learned path only; mirrors game_logic._resolve_hunt_tile.
    # Skip on a forced-tile continuation (Mithril redraw) or after the
    # staff was already declined (_skip_wizard).
    if (_ftile is None and not (choices or {}).get('_skip_wizard')
            and _wizardstaff_interposes(state)):
        _wizardstaff_stash_card(state, 'isildur', choices)
        result["effects"].append("Wizard's Staff prevent decision pending")
        return result
    if _ftile is None and not state.hunt_pool:
        import game_logic as _gl136
        _gl136._recycle_hunt_pool(state)   # 1.136 (USER RULE)
    if state.hunt_pool or _ftile is not None:
        if _ftile is not None:
            tile = _ftile
        else:
            tile = _tile_choice(state)
            state.hunt_pool.remove(tile)
            # Mithril Coat and Sting redraw interposes on THIS Shadow
            # draw (learned path only; almanac: any Shadow tile draw).
            if _mithril_interposes(state):
                _mithril_stash_card(state, tile, 'isildur', choices)
                return result
        state.hunt_discard.append(tile)  # drawn = used pile (returns on entry)
        try:
            from game_trace import emit_annotation as _ea132
            _ea132(state, 'hunt_tile_draw', {
                'tile_name': (tile.special_name if tile.special_name
                              else repr(tile)),
                'ctx': "Isildur's Bane",
                'damage': int(getattr(tile, 'damage', 0) or 0)})
        except Exception:
            pass


        # Teacher: the LOGGED resolution is ground truth - consume the
        # forced damage exactly like Orc Patrol / Foul Thing do (the
        # 1.112 eye ruling below governs unforced/self-play draws)
        _fdmg = choices.get("damage") if choices else None
        if _fdmg is not None:
            damage = int(_fdmg)
            result["effects"].append(
                f"Hunt tile (logged): {damage} damage as Corruption")
            state.fellowship.corruption = min(12, state.fellowship.corruption + damage)  # CL-217a: 12-ceiling clamp (2026-07-21)
            return result
        # USER RULING 1.112: card-drawn EYE tiles = 0 damage, no reveal, EVEN on
        # the Mordor track; they still leave the pool.
        if (getattr(tile, 'is_eye', False)
                or getattr(tile, 'is_fellowship_special', False)):
            result["effects"].append("Drew Eye/FP tile - discarded without effect")
        else:
            # Shelob's Lair rolls a d6 for damage (the closed loop forces the
            # logged roll); Eyes never reach this branch (1.112.1).
            damage = _card_tile_damage(tile)    # ONE body with Foul Thing
            result["effects"].append(f"Hunt tile: {damage} damage taken as Corruption only")
            state.fellowship.corruption = min(12, state.fellowship.corruption + damage)  # CL-217a: 12-ceiling clamp (2026-07-21)
            from game_logic import stage_hunt_reveal, settle_hunt_reveal
            stage_hunt_reveal(state, tile, getattr(tile, 'reveal', False))
            settle_hunt_reveal(state)
    return result


@register_effect("Foul Thing from the Deep")
def _effect_foul_thing(state: "GameState", card, choices: dict) -> dict:
    """Draw a hunt tile; the damage eliminates a RANDOM companion; a
    drawn HOBBIT uses his ability and SEPARATES instead of dying (USER
    2026-07-09); damage beyond the victim's level goes to Frodo as
    corruption (3 tile + hobbit L1 = +2). Eyes/FP-special tiles do
    nothing. Teacher passes the logged tile/victim via choices."""
    import game_logic as _gl
    from game_trace import emit_annotation
    result = {"effects": []}
    # tile source: forced (Mithril redraw continuation) or a fresh draw
    _ftile = choices.get('_forced_tile') if choices else None
    # 1.217 (USER 2026-07-23): the FP may discard "Wizard's Staff" to
    # PREVENT this card's Hunt-tile DRAW entirely (a PRE-draw nullify -
    # separate from the post-draw Axe/Horn -1 reduction offered below).
    # Learned path only; mirrors game_logic._resolve_hunt_tile. Skip on a
    # forced-tile continuation (Mithril redraw) or after the staff was
    # already declined (_skip_wizard).
    if (_ftile is None and not (choices or {}).get('_skip_wizard')
            and _wizardstaff_interposes(state)):
        _wizardstaff_stash_card(state, 'foul_thing', choices)
        result["effects"].append("Wizard's Staff prevent decision pending")
        return result
    if _ftile is not None:
        tile = _ftile
        state.hunt_discard.append(tile)
    else:
        if not state.hunt_pool:
            # 1.136: the used pile recycles into an empty pool (the
            # ONE recycler; Orc Patrol / Isildur's Bane / BotF match)
            _gl._recycle_hunt_pool(state)
        if not state.hunt_pool:
            return result
        tile = _tile_choice(state)
        state.hunt_pool.remove(tile)
        # gap (a): the Mithril Coat and Sting redraw interposes on THIS
        # Shadow draw (learned path only; almanac: any Shadow tile draw)
        if _mithril_interposes(state):
            _mithril_stash_card(state, tile, 'foul_thing', choices)
            return result
        state.hunt_discard.append(tile)
    _fdmg = choices.get("damage")
    if _fdmg is not None:
        damage = int(_fdmg)
        tname = str(choices.get("tile", "") or repr(tile))
    elif getattr(tile, 'is_eye', False) or             getattr(tile, 'is_fellowship_special', False):
        damage = 0
        tname = tile.special_name or repr(tile)
    else:
        # Shelob's Lair rolls its d6 here exactly as under Isildur's
        # Bane (ONE body: _card_tile_damage)
        damage = int(_card_tile_damage(tile))
        tname = tile.special_name or repr(tile)
    try:
        emit_annotation(state, 'hunt_tile_draw', {
            'tile_name': tname, 'ctx': 'Foul Thing from the Deep',
            'damage': damage})
    except Exception:
        pass
    # USER RULING 2026-07-20: an EYE (or FP-special) tile drawn by a card-forced
    # draw has NO EFFECT - its reveal flag must not fire (almanac C18: reveal
    # icons otherwise apply even with no damage; the Gollum guide may ignore
    # them).
    _gl.stage_hunt_reveal(
        state, tile, bool(getattr(tile, 'reveal', False)
                          and not getattr(tile, 'is_eye', False)
                          and not getattr(tile, 'is_fellowship_special', False)))
    if damage <= 0:
        if not result["effects"]:
            result["effects"].append("Tile has no effect (eye/FP special)")
        return result
    # almanac E13: FP may discard Axe and Bow / Horn of Gondor to reduce the
    # damage BEFORE the forced random casualty (one card per tile).  USER
    # RULING 2026-07-21: a LEARNED FP decision - self-play suspends into the
    # hunt_reduction interpose (_foul_apply_after_reduction resumes); the
    # teacher forces the logged use/decline via choices['hunt_reduction_used'].
    import combat as _cmbft
    _pc9 = getattr(state, 'persistent_cards', None)
    if _cmbft.FINAL_HITS_HOOK is None and _pc9 is not None \
            and getattr(state, 'pending_card_option', None) is None:
        from game_state import PersistentCardType as _PCT9
        for _pt9 in (_PCT9.AXE_AND_BOW, _PCT9.HORN_OF_GONDOR):
            _c9 = _pc9.get_by_type(_pt9)
            if _c9 is not None:
                state.pending_card_option = {
                    'card_id': _c9.card_id,
                    'card_name': _c9.card_name,
                    'side': 'fp', 'decider': 'fp', 'direct': True,
                    'hunt_reduction': True,
                    'foul_resume': {'damage': damage,
                                    'victim': choices.get('victim')},
                    'options': _gl._hunt_reduction_options(state)}
                result["effects"].append(
                    "Axe/Horn reduction decision pending")
                return result
    elif _cmbft.FINAL_HITS_HOOK is not None and _pc9 is not None \
            and choices.get('hunt_reduction_used'):
        # TEACHER: the log shows the human discarded the card here
        from game_state import PersistentCardType as _PCT9
        for _pt9 in (_PCT9.AXE_AND_BOW, _PCT9.HORN_OF_GONDOR):
            _c9 = _pc9.get_by_type(_pt9)
            if _c9 is not None:
                # CARD CONSERVATION (2026-08-15): discarded face-up
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, _pc9.remove_by_name(_c9.card_name) or _c9)
                damage -= 1
                result["effects"].append(
                    f"{_c9.card_name} discarded: damage reduced by 1")
                break
    if damage <= 0:
        result["effects"].append("Damage fully neutralised - no casualty")
        return result
    if _cmbft.FINAL_HITS_HOOK is None:
        # USER 1.208l: the guide-ability reduction (hobbit separates -1
        # / Gollum reveals -1) precedes the forced random casualty even
        # when no Axe/Horn interpose was available; _foul_confront
        # degrades to the identical forced-casualty tail when the
        # guide cannot reduce.
        return _foul_confront(state, {'victim': choices.get('victim')},
                              damage)
    vname = str(choices.get("victim", "") or "").strip().lower()
    victim = next((c for c in state.fellowship.companions
                   if c.name.strip().lower() == vname), None)         if vname else None
    if victim is None:
        cands = [c for c in state.fellowship.companions
                 if c.name != 'Gollum']
        # Almanac p.43: the victim is a RANDOM companion draw (never an
        # optimizable pick); the teacher supplies the logged victim above.
        if cands:
            from game_logic import pick_random_companion
            victim = pick_random_companion(cands, 'Foul Thing victim')
            try:
                emit_annotation(state, 'random_draw', {
                    'ctx': 'Foul Thing victim draw',
                    'result': victim.name,
                    'pool': [c.name for c in cands]})
            except Exception:
                pass
    if victim is None:
        state.fellowship.corruption = min(
            12, state.fellowship.corruption + damage)
        result["effects"].append(f"No companions: +{damage} Corruption")
        return result
    return _foul_apply_casualty(state, victim.name, damage, result)


def _foul_apply_after_reduction(state: "GameState", foul_resume: dict,
                                damage: int) -> dict:
    """Resume Foul Thing's forced-casualty tail after the learned
    Axe/Horn reduction decision (USER RULING 2026-07-21). Mirrors the
    original tail: named victim, else random draw among non-Gollum
    companions, else straight corruption."""
    result = {"effects": []}
    if damage <= 0:
        result["effects"].append(
            "Damage fully neutralised - no casualty")
        # A guide may already have separated while reducing this event.
        # Even without a casualty, its placement remains a learned choice.
        import combat as _foul_combat
        if _foul_combat.FINAL_HITS_HOOK is None:
            from game_logic import pop_hobbit_place_queue
            pop_hobbit_place_queue(state)
        return result
    vname = str((foul_resume or {}).get("victim", "") or "") \
        .strip().lower()
    victim = next((c for c in state.fellowship.companions
                   if c.name.strip().lower() == vname), None) \
        if vname else None
    if victim is None:
        cands = [c for c in state.fellowship.companions
                 if c.name != 'Gollum']
        if cands:
            from game_logic import pick_random_companion
            victim = pick_random_companion(cands, 'Foul Thing victim')
            try:
                from game_trace import emit_annotation
                emit_annotation(state, 'random_draw', {
                    'ctx': 'Foul Thing victim draw',
                    'result': victim.name,
                    'pool': [c.name for c in cands]})
            except Exception:
                pass
    if victim is None:
        state.fellowship.corruption = min(
            12, state.fellowship.corruption + damage)
        result["effects"].append(f"No companions: +{damage} Corruption")
        # A guide may already have separated while reducing this event.
        # Even without a casualty, its placement remains a learned choice.
        import combat as _foul_combat
        if _foul_combat.FINAL_HITS_HOOK is None:
            from game_logic import pop_hobbit_place_queue
            pop_hobbit_place_queue(state)
        return result
    return _foul_apply_casualty(state, victim.name, damage, result)


def _foul_guide_can_reduce(state, g):
    """USER 1.208l: the current Foul Thing guide can shave 1 damage iff a
    hobbit guide can separate (Merry/Pippin) or Gollum can reveal while
    the Fellowship is still hidden (one-shot)."""
    if g is None:
        return False
    ab = getattr(g, 'guide_ability', None)
    if ab == 'sacrifice_reduce_hunt':          # Merry/Pippin separate -1
        return True
    if ab == 'no_reveal_on_damage':
        from game_logic import _gollum_reveal_available
        return _gollum_reveal_available(state)
    return False


def _foul_confront(state, foul_resume, damage):
    """Foul Thing post-Axe/Horn tail. USER 1.208l: BEFORE the forced
    random casualty, the FP may use guide abilities (hobbit guide
    separates -1 / Gollum reveals -1), chained: after a hobbit separates
    a NEW guide may also act. Self-play opens a 2-option foul_guide
    interpose (one ability per step); decline / no reducing guide /
    damage<=0 -> existing forced-casualty tail."""
    import combat as _cmbfg
    if damage > 0 and _cmbfg.FINAL_HITS_HOOK is None \
            and getattr(state, 'pending_card_option', None) is None:
        g = state.fellowship.guide
        if _foul_guide_can_reduce(state, g):
            state.pending_card_option = {
                'card_id': '__foulguide__', 'card_name': 'Foul Thing guide',
                'side': 'fp', 'decider': 'fp', 'direct': True,
                'foul_guide': True,
                'foul_resume': dict(foul_resume or {}, damage=damage),
                'options': [{'use_foul_guide': 1}, {'use_foul_guide': 0}]}
            return {"effects": ["Foul Thing guide-ability decision pending"]}
    return _foul_apply_after_reduction(state, foul_resume, damage)


@register_effect("Candles of Corpses")
def _effect_candles_of_corpses(state: "GameState", card, choices: dict) -> dict:
    """Roll 3 dice, add Corruption on 4+ (6 if Gollum guide)."""
    result = {"effects": []}
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""
    threshold = 6 if "Gollum" in guide_name else 4

    rolls = _d6_batch(3)
    corruption = sum(1 for r in rolls if r >= threshold)

    state.fellowship.corruption = min(12, state.fellowship.corruption + corruption)  # CL-217a: 12-ceiling clamp (2026-07-21)
    result["effects"].append(f"Rolled {rolls}, added {corruption} Corruption (threshold {threshold}+)")
    # visible in traces/GUI — the silent roll made the card LOOK dead
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'dice_roll', {
            'faces': rolls,
            'ctx': f'Candles of Corpses ({threshold}+): '
                   f'+{corruption} corruption', 'side': 'SH'})
    except Exception:
        pass
    return result


def _move_all_nazgul_to_fellowship(state, result):
    """Relocate every Nazgul-figure (incl. the Witch-king) to the
    Ring-bearers' region (Nazgul Search / Nazgul Strike move-half)."""
    import game_logic as _gl9
    from game_state import Army as _Army9
    from ai import _stronghold_controller as _sc9
    _fsp9 = sanitize(_gl9.fellowship_region(state))
    # rulebook 2128-2131: Nazgul never enter a Free Peoples-controlled
    # stronghold unless a Shadow army besieges it (the rule the learned
    # per-figure chain applies in ai._get_nazgul_move_actions)
    if _sc9(state, _fsp9) == Side.FREE_PEOPLES and _fsp9 not in getattr(
            state, 'besieged_strongholds', set()):
        result["effects"].append(
            f"Nazgul cannot enter the unbesieged Free Peoples stronghold "
            f"at {_fsp9} - no Nazgul moved")
        return 0
    _moved9 = 0
    for rk, a in list(state.armies.items()):
        if sanitize(rk) == _fsp9:
            continue
        for l in list(a.leaders):
            if _gl9.is_nazgul_figure(l):
                a.leaders.remove(l)
                if _fsp9 not in state.armies:
                    state.armies[_fsp9] = _Army9(
                        region=state.board.region(_fsp9))
                state.armies[_fsp9].leaders.append(l)
                _moved9 += 1
        if not a.units and not a.leaders:
            state.armies.pop(rk, None)
    if _moved9:
        result["effects"].append(
            f"{_moved9} Nazgul moved to the Ring-bearers")
    return _moved9


@register_effect("Nazgûl Search")
def _effect_nazgul_search(state: "GameState", card, choices: dict) -> dict:
    """Move Nazgûl. If any in Fellowship region, reveal Fellowship."""
    result = {"effects": []}
    result["message"] = "nazgul moves are choices-driven"  # iter 39

    # LEARNED PATH (amendment 1.131): 'Move any or all of the Nazgul'
    # is the shared per-figure flight chain; the reveal check is
    # deferred to the chain's end so it sees the post-move positions.
    # Teacher mode replays the log's own flights as fx exactly as
    # before (FINAL_HITS_HOOK gate, same pattern as the die chain).
    import combat as _cmb131
    if _cmb131.FINAL_HITS_HOOK is None \
            and not choices.get('_from_pending') \
            and not choices.get('move_to_fellowship'):
        if _seed_card_nazgul_chain(state, card, post_key='search'):
            result["effects"].append("Nazgul movement (learned)")
            return result

    # TEACHER pure-mode: the log's own flights replay as engine
    # decisions (fx-derived); the reveal check below then sees the
    # post-move positions.  Self-play never sets this key.
    _nmv9 = choices.get("nazgul_moves")
    if _nmv9:
        _exec_logged_nazgul_flights(state, _nmv9, result)

    # 'Move any or all of the Nazgul' (QA 2026-07-11 r3 E3: the move
    # half was unimplemented - the card only revealed when a Nazgul
    # was already co-located)
    if choices.get("move_to_fellowship"):
        _move_all_nazgul_to_fellowship(state, result)

    # Check if any Nazgûl now in Fellowship region
    _nazgul_search_reveal_check(state, result)
    return result


@register_effect("Cruel Weather")
def _effect_cruel_weather(state: "GameState", card, choices: dict) -> dict:
    """Shadow moves Ring-bearers one region (no hunt triggered).

    The Shadow player picks an adjacent region to move the Fellowship figure
    to.  This does NOT trigger Hunt, does NOT cost a die, and is NOT normal
    Fellowship movement.
    """
    from game_logic import fellowship_region, is_shadow_region
    result = {"effects": []}

    fsp_region = fellowship_region(state)
    target = choices.get("target_region")

    if not target:
        # Heuristic: pick the adjacent region most favourable to Shadow
        # (prefer Shadow-controlled or away from Mordor)
        adj_regions = state.board.adjacent(fsp_region)
        best, best_score = None, -999
        for adj in adj_regions:
            adj_san = sanitize(adj)
            score = 0
            # Prefer Shadow regions
            if is_shadow_region(adj_san):
                score += 2
            # Prefer regions with Shadow armies
            army = state.armies.get(_find_army_key(state, adj_san))
            if army and _is_army_shadow(army):
                score += 3
            # Penalise regions closer to Mordor (we want to move FB away)
            mordor_names = {"morannon", "minas morgul", "gorgoroth", "mount doom", "barad dur"}
            if adj_san in mordor_names:
                score -= 5
            if score > best_score:
                best, best_score = adj_san, score
        target = best

    if not target:
        result["message"] = "No adjacent region to move Ring-bearers"
        return _selfplay_refund(result)

    target_san = sanitize(target)

    # Verify adjacency (self-play only; teacher logs the exact figure move)
    if target_san not in [
            sanitize(a) for a in state.board.adjacent(fsp_region)]:
        result["message"] = f"Cruel Weather: {target} is not adjacent to {fsp_region}"
        return _selfplay_refund(result)

    # Move fellowship figure (no hunt, no progress change)
    try:
        new_region = state.board.region(target_san)
        state.fellowship.region = new_region
        result["effects"].append(f"Cruel Weather: Ring-bearers moved from {fsp_region} to {target_san}")
    except Exception as e:
        result["effects"].append(f"Cruel Weather movement failed: {e}")
    return result


@register_effect("Nazgûl Strike!")
def _effect_nazgul_strike(state: "GameState", card, choices: dict) -> dict:
    """
    Move any/all Nazgûl. Then, if at least one Nazgûl is with Ring-bearers:
    - Discard one FP Character Event card from the table, OR
    - Roll for Hunt (as if FP moved Fellowship)
    """
    result = {"effects": []}
    result["message"] = "nazgul moves are choices-driven"  # iter 39

    # LEARNED PATH (amendment 1.131): per-figure flight chain; the
    # either/or ('Then, if at least one Nazgul is in the region with
    # the Ring-bearers...') fires at the chain's end as its own
    # learned pending.  Teacher mode unchanged (choices-driven).
    import combat as _cmb131
    if _cmb131.FINAL_HITS_HOOK is None \
            and not choices.get('_from_pending') \
            and not choices.get('move_to_fellowship') \
            and not choices.get('action'):
        if _seed_card_nazgul_chain(state, card, post_key='strike'):
            result["effects"].append("Nazgul movement (learned)")
            return result

    # 'Move any or all of the Nazgul' (QA 2026-07-11: previously a GUI
    # stub - self-play options request moving them to the Ring-bearers).
    # ONE implementation: shared with Nazgul Search.
    if choices.get("move_to_fellowship"):
        _move_all_nazgul_to_fellowship(state, result)

    # TEACHER pure-mode (engine-driven-resolution): execute the human's
    # LOGGED Nazgul flights (fx-derived list of {from,to,n}) so the
    # strike's Hunt rolls with a genuinely co-located Nazgul.  The
    # destinations are DECISIONS fed to the engine; only the Hunt roll /
    # tile are RNG-forced.  Self-play never sets this key.
    _nmoves131 = choices.get("nazgul_moves")
    if _nmoves131:
        import game_logic as _glnm
        from game_state import Nazgul as _NazNM
        for _mvn in _nmoves131:
            _frm = sanitize(str(_mvn.get('from', '')))
            _to = sanitize(str(_mvn.get('to', '')))
            _cnt = int(_mvn.get('n', 0) or 0)
            _cmin = int(_mvn.get('nmin', 0) or 0)
            if not _frm or not _to or (_cnt <= 0 and _cmin <= 0):
                continue
            _srck = _find_army_key(state, _frm)
            _srca = state.armies.get(_srck) if _srck else None
            if _srca is None:
                continue
            _movable = [l for l in _srca.leaders
                        if isinstance(l, _NazNM)]
            for _fig in _movable[:_cnt]:
                _srca.leaders.remove(_fig)
                _glnm.get_or_create_army(state, _to).leaders.append(_fig)
            # RB-02/RB-04 (C2, 2026-07-22): named minions (Witch-king /
            # Mouth) ride the flight — the parser tallies them as
            # moved['leader'], distinct from the plain-Nazgûl count above.
            # The engine models them as Minion objects (not Nazgul), so the
            # isinstance(Nazgul) sweep alone stranded the Witch-king at the
            # source.  Saruman never leaves Orthanc (official rule) — never
            # sweep him.  Mirrors _sweep_minions_after_nazgul (the die-driven
            # nazgul_move path already carries minions this way).
            if _cmin > 0:
                _minmov = [l for l in _srca.leaders
                           if type(l).__name__ == 'Minion'
                           and not _glnm._is_saruman(l)][:_cmin]
                for _fig in _minmov:
                    _srca.leaders.remove(_fig)
                    _glnm.get_or_create_army(state, _to).leaders.append(_fig)
            if not _srca.units and not _srca.leaders:
                state.armies.pop(_srck, None)

    action = choices.get("action")  # "discard_table_card" or "hunt"
    target_card = choices.get("target_card")  # Card ID to discard

    # 1.211j: BOTH arms of the either/or require a Nazgul CO-LOCATED
    # with the Ring-bearers (not merely a Nazgul somewhere on the board).
    from game_logic import fellowship_region as _fr9
    _fsp_naz9 = sanitize(_fr9(state))
    _fsp_army9 = state.armies.get(_find_army_key(state, _fsp_naz9))
    _naz_colocated9 = (_fsp_army9 is not None
                       and _army_contains_nazgul(_fsp_army9))
    # TEACHER log-forced bypass (canary game 5, 2024-07-27 Columbo T8):
    # VASSAL does not enforce the co-location condition and the corpus
    # contains plays where the human rolled the Hunt (or discarded)
    # without a Nazgul at the Ring-bearers' region. The log is truth on
    # the teacher path - replay the misplay rather than desync. The
    # strict 1.211j gate stays authoritative for self-play (the flag is
    # only honored under the teacher hook).
    import combat as _cmbns9
    if (choices or {}).get('_log_forced_then') \
            and _cmbns9.FINAL_HITS_HOOK is not None:
        _naz_colocated9 = True
    if action == "discard_table_card" and target_card:
        if not _naz_colocated9:
            # 1.211j: no Nazgul with the Ring-bearers -> the either/or is
            # unreachable, the discard must not happen
            result["message"] = "No Nazgul with the Ring-bearers"
            return result
        # 1.211j: the target MUST be an FP Character-deck table card
        from cards import get_card as _gc9
        _tc9 = _gc9(target_card)
        _deck9 = str(getattr(_tc9, 'deck', '') or '').lower()
        _sd9 = getattr(_tc9, 'side', None)
        _sd9s = str(getattr(_sd9, 'value', _sd9) or '').lower().replace(
            '_', ' ')
        _is_fp9 = (_sd9 == Side.FREE_PEOPLES
                   or _sd9s in ('free peoples', 'fp', 'free_peoples'))
        if _tc9 is None or _deck9 != 'character' or not _is_fp9:
            result["message"] = ("Nazgul Strike target must be an FP "
                                 "Character-deck table card")
            return result
        # Discard a FP Character card from the table
        from game_state import PersistentCardState
        if hasattr(state, 'persistent_cards') and isinstance(state.persistent_cards, PersistentCardState):
            removed = state.persistent_cards.remove_card(target_card)
            if removed:
                # CARD CONSERVATION (2026-08-15): the discard is PHYSICAL
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, removed)
                result["effects"].append(f"Discarded {removed.card_name} from table")
            else:
                result["effects"].append(f"Card {target_card} not found on table")
        else:
            result["effects"].append("No persistent cards in play")

    elif action == "hunt":
        # 1.211j: the Hunt branch requires a Nazgul CO-LOCATED with the
        # Ring-bearers
        if not _naz_colocated9:
            result["message"] = ("No Nazgul with the Ring-bearers - "
                                 "Hunt branch not available")
            return result
        # Roll for Hunt as if FP moved Fellowship (card text).  Real
        # mechanics, not a GUI stub (2024-07-27 Columbo T8 L757: the
        # play's hunt drew the '2' tile at L764 and Boromir guide-
        # sacrificed at L787 -- the old no-op left the tile undrawn and
        # the sacrifice record stranded).  The closed loop forces the
        # roll and tile through the game_logic hooks; damage is left in
        # _pending_hunt_damage for the caller's learned sacrifice
        # decision (same contract as move_fellowship with
        # skip_auto_sacrifice).
        import game_logic as _gl
        from game_state import Side as _Side
        result["effects"].append("Rolling for Hunt...")
        # Hunt Level = Shadow dice in the Hunt Box (rulebook p.40, the
        # same count the fellowship_move path rolls): 0 dice -> no roll
        hunt_level = sum(1 for d in state.hunt_box
                         if d.side is _Side.SHADOW)
        num_dice = min(5, hunt_level)
        fp_dice_in_box = sum(1 for d in state.hunt_box
                             if d.side is _Side.FREE_PEOPLES)
        if num_dice <= 0:
            result["effects"].append("Hunt Level 0 - no Hunt dice to roll")
            successes = 0
        else:
            successes = _gl._hunt_roll(state, num_dice, 6,
                                       fp_modifier=fp_dice_in_box)
        # (1.136: _resolve_hunt_tile recycles an empty pool itself - no
        # pool guard here)
        if successes:
            _gl._resolve_hunt_tile(state, successes, mordor=False)
            result["effects"].append(
                f"Hunt hit ({successes}): tile drawn")
            # PURE-MODE teacher (engine-driven-resolution): open the
            # sacrifice decision so the logged hunt_sacrifice_random /
            # _guide record resolves the drawn damage (mirrors
            # move_fellowship's skip_auto_sacrifice + _open_pending_hunt_
            # after_move).  Gated on the pure wrapper's _pure_open_hunt flag
            # ONLY - in normal-teacher mode _replicate_card_fx_inner still
            # owns the strike-hunt pending_hunt (double-open avoided); self-
            # play keeps the learned-pending contract (damage untouched).
            _dmgph = getattr(state, '_pending_hunt_damage', 0)
            if _dmgph > 0 \
                    and getattr(state, 'pending_hunt', None) is None \
                    and choices.get('_pure_open_hunt'):
                _sacrph = [c for c in state.fellowship.companions
                           if c.name != 'Gollum' and c.level > 0]
                if _sacrph:
                    state.pending_hunt = {'damage': _dmgph,
                                          'phase': 'hunt_sacrifice'}
                else:
                    _gl._apply_fellowship_damage(state, _dmgph)
                    state._pending_hunt_damage = 0
        else:
            result["effects"].append("Hunt missed")

    else:
        # no-op: NOT an effect (the option spec provides real choices)
        result["message"] = ("Choose: discard FP Character card from "
                             "table or roll Hunt")

    return result


@register_effect("Morgul Wound")
def _effect_morgul_wound(state: "GameState", card, choices: dict) -> dict:
    """Add Corruption: 2 if current ≤3, else 1."""
    result = {"effects": []}
    corruption_add = 2 if state.fellowship.corruption <= 3 else 1
    state.fellowship.corruption = min(12, state.fellowship.corruption + corruption_add)  # CL-217a: 12-ceiling clamp (2026-07-21)
    result["effects"].append(f"Added {corruption_add} Corruption (now {state.fellowship.corruption})")
    return result


def _recycle_bof136(state) -> bool:
    import game_logic as _gl136
    return _gl136._recycle_hunt_pool(state) > 0


@register_effect("Lure of the Ring")
def _effect_lure_of_ring(state: "GameState", card, choices: dict) -> dict:
    """Random companion: FP chooses Corruption = Level or eliminate."""
    result = {"effects": []}
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""

    if "Gollum" in guide_name:
        state.fellowship.corruption = min(12, state.fellowship.corruption + 1)  # CL-217a: 12-ceiling clamp (2026-07-21)
        result["effects"].append("Gollum is guide - added 1 Corruption instead")
    elif state.fellowship.companions:
        # Teacher: the logged draw names the victim.  STAGE 2 ('_lure_drawn',
        # the FP's cross-side pick): the draw already happened - no new draw
        # and no stream-compat consume.
        victim = None
        _stage2 = bool(choices.get('_lure_drawn'))
        vname = str(choices.get("victim", "") or "").strip().lower()
        if vname:
            victim = next((c for c in state.fellowship.companions
                           if c.name.strip().lower() == vname), None)
            if victim is not None and not _stage2:
                # stream-compat: consume the draw the unforced path
                # would have made (seeded-RNG alignment, see ai.py
                # hunt_sacrifice_random)
                random.choice(state.fellowship.companions)
        if victim is None and _stage2:
            result["message"] = (
                f"Lure of the Ring: drawn victim {vname!r} is no longer "
                f"in the Fellowship")
            return result
        if victim is None:
            from game_logic import pick_random_companion
            victim = pick_random_companion(state.fellowship.companions,
                                           'Lure of the Ring victim')
        if not _stage2:
            try:
                from game_trace import emit_annotation as _ea132
                _ea132(state, 'random_draw', {
                    'ctx': 'Lure of the Ring victim',
                    'result': victim.name,
                    'pool': [c.name for c in state.fellowship.companions]})
            except Exception:
                pass
        action = choices.get("action")  # "corruption" or "eliminate"
        if action is None:
            import combat as _cmb_lure
            if _cmb_lure.FINAL_HITS_HOOK is not None:
                action = "corruption"      # teacher: historical default
            else:
                # STAGE 1 (audit EE2-3): card text - the Companion is
                # drawn FIRST, then "the Free Peoples player must
                # choose".  The FP decides SEEING the draw: a direct
                # cross-side pending (Stormcrow pattern, decider fp)
                # whose options carry the drawn victim's name and Level.
                _lvl9 = int(getattr(victim, 'level', 0) or 0)
                state.pending_card_option = {
                    'card_id': getattr(card, 'card_id',
                                       'SHADOW_Character_Lure_of_the_Ring'),
                    'card_name': 'Lure of the Ring',
                    'side': 'sh', 'decider': 'fp', 'direct': True,
                    'options': [
                        {'action': 'corruption', 'victim': victim.name,
                         'level': _lvl9, '_lure_drawn': True},
                        {'action': 'eliminate', 'victim': victim.name,
                         'level': _lvl9, '_lure_drawn': True}]}
                result["effects"].append(
                    f"Lure of the Ring drew {victim.name} (Level {_lvl9}) "
                    f"- Free Peoples choose Corruption or elimination "
                    f"(pending)")
                return result
        if action == "eliminate":
            import game_logic as _gl9
            state.fellowship.companions.remove(victim)
            # USER RULING 1.257: a Lure elimination is NOT a Hunt casualty - WWSAT
            # does not fire; the companion still leaves play through the ONE
            # bookkeeper (casualty=False).  A departing guide re-picks by
            # HIGHEST Level (tie -> pending_guide_change).
            _gl9._on_companion_casualty(state, victim.name, casualty=False)
            _gl9.change_guide(state, allow_pending=True)
            if _gl9.hobbit_hunt_placement_applies(state, victim.name):
                # almanac (QA 2026-07-11 C9): a drawn Hobbit uses
                # Take-Them-Alive - placed on the board, not eliminated
                if getattr(state, 'pending_hunt', None) is None:
                    state.pending_hunt = {
                        'phase': 'hobbit_place',
                        'companion': victim.name,
                        'level': int(getattr(victim, 'level', 1) or 1),
                        'companion_obj': victim, 'damage': 0}
                result["effects"].append(
                    f"{victim.name} separates (hobbit ability)")
            else:
                result["effects"].append(f"{victim.name} eliminated")
        else:
            state.fellowship.corruption = min(12, state.fellowship.corruption + victim.level)  # CL-217a: 12-ceiling clamp (2026-07-21)
            result["effects"].append(f"Added {victim.level} Corruption for {victim.name}")
    return result


@register_effect("The Breaking of the Fellowship")
def _effect_breaking_of_fellowship(state: "GameState", card, choices: dict) -> dict:
    """Draw hunt tile; separate companions = tile number."""
    result = {"effects": []}
    choices = dict(choices or {})
    card_id = getattr(card, 'card_id', None) or choices.get('_breaking_card_id')
    guide_name = state.fellowship.guide.name if state.fellowship.guide else ""

    if "Gollum" in guide_name:
        state.fellowship.corruption = min(12, state.fellowship.corruption + 1)  # CL-217a: 12-ceiling clamp (2026-07-21)
        result["effects"].append("Gollum is guide - added 1 Corruption instead")
    else:
        # Almanac p.25: this is not a Hunt, but both protection cards
        # apply to its Shadow draw. Gollum's alternative above draws no tile.
        forced = choices.get('_forced_tile')
        continuation = dict(choices, _breaking_card_id=card_id)
        if (forced is None and not choices.get('_skip_wizard')
                and _wizardstaff_interposes(state)):
            _wizardstaff_stash_card(state, 'breaking', continuation)
            result['effects'].append("Wizard's Staff prevent decision pending")
            return result
        if forced is None:
            if not state.hunt_pool and not _recycle_bof136(state):
                return result
            tile = _tile_choice(state)
            state.hunt_pool.remove(tile)
            if _mithril_interposes(state):
                _mithril_stash_card(state, tile, 'breaking', continuation)
                return result
        else:
            tile = forced
        state.hunt_discard.append(tile)
        try:
            from game_trace import emit_annotation as _ea132
            _ea132(state, 'hunt_tile_draw', {
                'tile_name': (tile.special_name if tile.special_name
                              else repr(tile)),
                'ctx': 'The Breaking of the Fellowship',
                'damage': int(getattr(tile, 'damage', 0) or 0)})
        except Exception:
            pass

        if getattr(tile, 'is_eye', False) or getattr(tile, 'is_fellowship_special', False):
            result["effects"].append("Drew Eye/FP tile - discarded without effect")
        else:
            # Separate `damage` companions into the Ring-bearers' region (ignore
            # Reveal/Stop).  WHICH companions leave is the FREE PEOPLES
            # player's decision: replay = the logged companion_separate
            # records (the handler separates no one); explicit choices =
            # 'separate'; self-play = the chained learned pending.
            import combat as _cmb_bof
            import game_logic as _gl
            from game_state import Army as _Army
            n = _card_tile_damage(tile)
            cur = sanitize(_gl.fellowship_region(state))
            movable = [c for c in state.fellowship.companions
                       if c.name not in ("Frodo", "Sam", "Gollum")]
            _named = (choices.get('separate')
                      or choices.get('companions') or [])
            if _named:
                # the logged/explicit decision (teacher: derived from
                # the play's card_effect'd companion_separate records
                # via the fx-choices block; GUI: passed directly)
                _want = [sanitize(str(x)) for x in _named]
                sep = [c for c in movable
                       if sanitize(str(c.name)) in _want][:n]
            elif _cmb_bof.FINAL_HITS_HOOK is not None:
                # teacher/replay with NO logged separations: the human
                # separated no one (or the records were retracted) -
                # replay that reality, never pick for them
                sep = []
            else:
                # USER 2026-07-27: a chained LEARNED pending (one pick per
                # option, like the nazgul chain), never an auto-pick
                if n > 0 and movable:
                    state.pending_card_option = {
                        'card_id': card_id,
                        'card_name': 'The Breaking of the Fellowship',
                        'side': 'sh', 'decider': 'fp',
                        'bof_separation': True,
                        'bof_n_left': min(n, len(movable)),
                        'bof_region': cur,
                        'bof_mordor': bool(
                            state.fellowship.mordor_entered),
                        'options': [{'companions': [c.name]}
                                    for c in movable],
                    }
                    result["effects"].append(
                        "Breaking: %d separation pick(s) pending (FP "
                        "decides)" % min(n, len(movable)))
                    return result
                sep = []
            if sep:
                _mordor9 = state.fellowship.mordor_entered
                # Armies ONLY via get_or_create_army (standing rule; a raw
                # sanitized-key write minted a duplicate entry)
                _host9 = (None if _mordor9
                          else _gl.get_or_create_army(state, cur))
                for c in sep:
                    state.fellowship.companions.remove(c)
                    if _mordor9:
                        # almanac (QA 2026-07-11 C7): companions
                        # separated on the Mordor Track are REMOVED
                        # from play, never placed
                        pass
                    else:
                        _host9.leaders.append(c)
                # Adjudication #3 (2026-07-21): a separated guide re-picks
                # by HIGHEST Level (Level tie -> pending_guide_change), not
                # arbitrary companions[0].  Every composition change opens
                # this opportunity (Rulebook p37), retaining the old guide if chosen.
                _gl.change_guide(state, allow_pending=True)
                # card texts (Horn of Gondor / Axe and Bow / Wizard's
                # Staff): discard when the required companion leaves
                # the Fellowship - the ONE implementation
                _gl.check_companion_persistent_discards(state)
                result["effects"].append(
                    f"Separated {len(sep)} companion(s) to {cur}: "
                    + ', '.join(c.name for c in sep))
            else:
                result["message"] = "no companions to separate"
    return result


def _bof_apply_separation(state, pco, name):
    """Apply ONE chained Breaking-of-the-Fellowship separation pick
    (USER 2026-07-27: which companions leave is the FP player's LEARNED
    decision — the old self-play path auto-picked lowest-level first).
    Called by the ai.py card_option executor; re-opens the pending
    while picks remain, then runs the guide re-pick tail."""
    import game_logic as _gl
    from board import sanitize as _s
    cur = pco.get('bof_region')
    mordor = bool(pco.get('bof_mordor'))
    comp = next((c for c in state.fellowship.companions
                 if _s(str(c.name)) == _s(str(name or ''))), None)
    if comp is not None:
        state.fellowship.companions.remove(comp)
        if not mordor:
            # almanac C7: Mordor-track separations are REMOVED from
            # play; otherwise place with the Ring-bearers
            _gl.get_or_create_army(state, cur).leaders.append(comp)
        # companion-linked table cards (Horn / Axe and Bow / Wizard's
        # Staff) discard when their companion leaves - ONE implementation
        _gl.check_companion_persistent_discards(state)
    n_left = int(pco.get('bof_n_left', 1)) - 1
    movable = [c for c in state.fellowship.companions
               if c.name not in ('Frodo', 'Sam', 'Gollum')]
    if n_left > 0 and movable:
        state.pending_card_option = dict(
            pco, bof_n_left=n_left,
            options=[{'companions': [c.name]} for c in movable])
        return
    _gl.change_guide(state, allow_pending=True)


@register_effect("Worn with Sorrow and Toil")
def _effect_worn_with_sorrow(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Discard FP Character card when companion casualty."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Worn with Sorrow and Toil in play - discard FP card on companion casualty")
    return result


@register_effect("Flocks of Crebain")
def _effect_flocks_of_crebain(state: "GameState", card, choices: dict) -> dict:
    if choices and 'crebain_discard' in choices:
        # P1 (USER ruling 1.112): the Shadow's hunt-time choice -
        # discard the table card for +1 to all Hunt dice, or keep it.
        # Set by the fellowship_move interposition in ai._execute_action
        result = {"effects": []}
        if choices['crebain_discard']:
            from game_state import PersistentCardType as _PCT9
            _pc = getattr(state, 'persistent_cards', None)
            _cb = (_pc.get_by_type(_PCT9.FLOCKS_OF_CREBAIN)
                   if _pc else None)
            if _cb is not None:
                # CARD CONSERVATION (2026-08-15): discarded face-up
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, _pc.remove_card(_cb.card_id) or _cb)
                state._crebain_hunt_bonus = 1
                result["effects"].append(
                    "Flocks of Crebain discarded: +1 to all Hunt dice")
        else:
            result["effects"].append("Flocks of Crebain kept in play")
        return result
    return _effect_flocks_of_crebain_play(state, card, choices)


def _effect_flocks_of_crebain_play(state, card, choices):
    """Persistent: +1 to Hunt dice."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Flocks of Crebain in play - may add +1 to Hunt dice")
    return result


@register_effect("Dreadful Spells")
def _effect_dreadful_spells(state: "GameState", card, choices: dict) -> dict:
    """Roll dice = Nazgûl count, hit FP army on 5+."""
    result = {"effects": []}
    choices = dict(choices or {})
    target_region = choices.get("target_region")

    if not target_region:
        # Default (the card requirement guarantees one exists): a Shadow
        # ARMY containing Nazgul (units + Nazgul, USER 2026-07-16 item 5)
        # in or adjacent to a region with FP units
        for k, a in state.armies.items():
            if not _is_shadow_army_with_nazgul(a):
                continue
            k_san = sanitize(k)
            for r in [k_san] + [sanitize(x)
                                for x in state.board.adjacent(k_san)]:
                fp_a = None
                for k2, a2 in state.armies.items():
                    if sanitize(k2) == r:
                        fp_a = a2
                        break
                if fp_a and any(u.nation not in SHADOW_NATIONS
                                for u in fp_a.units):
                    target_region = k
                    choices.setdefault("target_fp_army", r)
                    break
            if target_region:
                break

    if target_region:
        # sanitized scan — armies are keyed by display names
        t_san = sanitize(str(target_region))
        army = next((a for k, a in state.armies.items()
                     if sanitize(k) == t_san), None)
        # "that Army" = a Shadow Army (units) containing the Nazgul - a
        # lone Nazgul entry is not an Army (ONE predicate with the gate)
        if _is_shadow_army_with_nazgul(army):
            import game_logic as _gl9
            # the Witch-king provides a die too (almanac, QA C6)
            nazgul_count = sum(1 for l in army.leaders
                               if _gl9.is_nazgul_figure(l))
            dice_count = min(5, nazgul_count)

            rolls = _d6_batch(dice_count)
            _ann_rng(state, 'Dreadful Spells roll', faces=rolls)
            hits = sum(1 for r in rolls if r >= 5)
            result["effects"].append(f"Rolled {rolls} ({dice_count} Nazgûl), {hits} hits")

            # Apply to FP army
            fp_army = choices.get("target_fp_army")
            if fp_army:
                f_san = sanitize(str(fp_army))
                target_army = next((a for k, a in state.armies.items()
                                    if sanitize(k) == f_san), None)
                if target_army and hits > 0:
                    # USER 2026-07-27: owner-allocated via the shared
                    # casualty decision matrix (FP tracking + wiped-
                    # garrison capture live in the router/aftermath).
                    # USER RULING 2026-08-15: publish the wipe context
                    # so any destruction reconciled underneath takes the
                    # Dreadful-Spells exception (companions/nazgul/
                    # minions survive, plain FP Leaders die).
                    import combat as _cmb_ds9
                    _prevw9 = _cmb_ds9.set_wipe_card_context(
                        'Dreadful Spells')
                    try:
                        _route_card_casualties(
                            state, f_san, Side.FREE_PEOPLES, hits, card,
                            choices, result)
                    finally:
                        _cmb_ds9.set_wipe_card_context(_prevw9)
    return result


@register_effect("Grond Hammer of the Underworld")
def _effect_grond(state: "GameState", card, choices: dict) -> dict:
    """Attack stronghold: 3 rounds, FP can't play card first round unless companion.

    Requires the Witch-king with a Shadow Army besieging a native
    Free Peoples Stronghold. The battle has 3 mandatory rounds.
    Round-1 FP combat card blocked UNLESS a Companion is in the battle
    (printed exception; pc key no_def_card_r1_unless_companion on the
    learned path - the legacy gl_battle fallback passes no defender
    card at all, whole-battle).
    """
    from game_logic import battle as gl_battle
    result = {"effects": []}

    # TEACHER path: the schedule's own army_attack record synthesizes this
    # assault; opening it here would pre-empt that cursor.  Self-play keeps
    # the interactive assault.
    import combat as _cmbg9
    if _cmbg9.FINAL_HITS_HOOK is not None and \
            not choices.get('_from_pending'):
        return result

    atk_region = choices.get("attack_from")
    def_region = choices.get("attack_target")

    # --- co-located target selection when AI has no explicit choices ---
    # Share the printed prerequisite and option predicate, preferring the
    # largest eligible besieging Army when no target was supplied.
    if not atk_region or not def_region:
        _targets = set(_grond_siege_regions(state))
        best, best_score = None, -1
        for rkey, army in state.armies.items():
            rsan = sanitize(rkey)
            if rsan not in _targets:
                continue
            score = len(army.units)
            if score > best_score:
                best, best_score = rsan, score
        if best is not None:
            atk_region = best
            def_region = best

    if not atk_region or not def_region:
        result["effects"].append("No co-located Witch-king siege for Grond")
        return _selfplay_refund(result)

    # Almanac p17: event-card attacks cannot target A Power too Great /
    # Tom Bombadil protected regions (audit 2026-07-26).
    from persistent_effects import can_shadow_attack as _csa26g
    _ok26g, _blk26g = _csa26g(state, sanitize(def_region))
    if not _ok26g:
        result["effects"].append(
            f"Grond blocked: {def_region} protected by {_blk26g}")
        return _selfplay_refund(result)

    # P5 (USER ruling 1.112.5, almanac C20): counts as an ATTACK for
    # the Political Track (see Fighting Uruk-hai note)
    import game_logic as _glp9
    _glp9.advance_defenders_political(state, sanitize(def_region),
                                      attacker_side=Side.SHADOW)

    if choices.get('_from_pending'):
        # P2 (USER ruling 1.112.2): identical machinery to Fighting
        # Uruk-hai - 3 mandatory rounds, round-1 FP card block unless
        # a Companion is in the battle, then normal press rules.
        import ai as _ai9
        from combat import garrison_side_for as _gsf9
        _dsan9 = sanitize(def_region)
        _pc9 = {
            'attacker_region': sanitize(atk_region),
            'defender_region': _dsan9,
            'is_siege': True,
            'stronghold_region': _dsan9,
            'phase': 'combat_setup',
            'deciding_side': Side.SHADOW,
            'attacker_side': Side.SHADOW,
            'defender_side': _gsf9(state, _dsan9),
            'rounds_fought': 0,
            'mandatory_rounds': 3,
            'no_def_card_r1_unless_companion': True,
        }
        state.pending_combat = _pc9
        _ai9._run_combat_round(state, _pc9, is_siege=True)
        result["effects"].append(
            f"Grond: interactive assault on {_dsan9} "
            f"(3 mandatory rounds)")
        return result

    try:
        combat_result = gl_battle(
            state,
            sanitize(atk_region),
            sanitize(def_region),
            defender_card_id=None,   # FP cannot play combat card round 1
            is_siege=True,
            max_rounds=3,
            auto_press=True,
        )
        result["effects"].append(
            f"Grond siege: {atk_region} -> {def_region}, "
            f"{combat_result.rounds_fought} rounds, "
            f"atk hits={combat_result.attacker_total_hits}, "
            f"def hits={combat_result.defender_total_hits}"
        )
        if combat_result.defender_destroyed:
            result["effects"].append(f"Stronghold {def_region} conquered!")
        elif combat_result.attacker_destroyed:
            result["effects"].append("Attacking army destroyed")
    except Exception as e:
        result["effects"].append(f"Grond battle failed: {e}")
    return result


@register_effect("Balrog of Moria")
def _effect_balrog_of_moria(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Extra hunt tile when FSP moves through Moria."""
    result = {"effects": []}
    if choices and 'balrog_declare_draw' in choices:
        # P5 (USER ruling 1.112): the DECLARATION trigger - the Shadow
        # MAY discard the table card to draw a hunt tile after a
        # declaration from/into Moria (almanac C17: declaration fully
        # resolves first; Eye = discarded without effect; a Reveal icon
        # is ignored in an originally FP City/Stronghold, captured or not)
        if not choices['balrog_declare_draw']:
            result["effects"].append("Balrog of Moria kept in play")
            return result
        import game_logic as _gl9
        from game_state import PersistentCardType as _PCTb9
        _pcb = getattr(state, 'persistent_cards', None)
        _cb = (_pcb.get_by_type(_PCTb9.BALROG_OF_MORIA)
               if _pcb else None)
        if _cb is None:
            return result
        # ONE draw through the funnel (_resolve_hunt_tile with
        # eye_no_effect=True): the live bridge patches pop_hunt_tile, so a
        # pool-tail PEEK and the popped tile can differ (BRIDGE-9 halt).
        # The eye-without-effect rule applies to the tile ACTUALLY drawn.
        _t9 = _gl9._resolve_hunt_tile(state, 1, mordor=False,
                                     eye_no_effect=True)
        # The funnel returns a resolved tile. Counting discard entries is
        # unreliable when an empty-pool refill removes the previous pile.
        if _t9 is None:
            result["effects"].append(
                "Balrog draw: resolution deferred (interposed card decision pending)"
                if state.pending_card_option is not None else
                "Balrog draw: no eligible Hunt tile available")
        elif getattr(_t9, 'is_eye', False):
            result["effects"].append(
                "Balrog draw: Eye tile - discarded without effect")
        else:
            # HuntTile has no .name (HIGH-4)
            _tdesc9 = (getattr(_t9, 'special_name', None)
                       or ('damage %s%s' % (
                           getattr(_t9, 'damage', '?'),
                           ' reveal' if getattr(_t9, 'reveal',
                                                False) else '')))
            result["effects"].append(
                f"Balrog draw: hunt tile ({_tdesc9}) resolved")
        # CARD CONSERVATION (2026-08-15): discarded face-up
        from persistent_effects import discard_table_card as _dtc9
        _dtc9(state, _pcb.remove_card(_cb.card_id) or _cb)
        result["effects"].append("Balrog of Moria discarded (used)")
        return result
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Balrog of Moria in play - extra Hunt tile for Moria movement")
    return result


@register_effect("The Palantír of Orthanc")
def _effect_palantir_of_orthanc(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Draw card when playing Event die."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Palantír of Orthanc in play - draw card when playing Event")
    return result


@register_effect("Wormtongue")
def _effect_wormtongue(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Rohan cannot activate (with exceptions)."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Wormtongue in play - Rohan cannot activate")
    return result


def _grant_card_followup(state, side, kind, moves=1):
    """Card-granted free army action(s) after the card effect (USER
    audit 2026-07-08: RWA/BCC 'then move/attack' never happened)."""
    try:
        import ai as _ai   # importability guard: no pending without ai
    except Exception:
        return
    state.pending_card_followup = {
        'kind': kind,
        'side': 'fp' if side == Side.FREE_PEOPLES else 'sh',
        'moves_left': moves, 'exclude': set(), 'moved_any': False}


def _exec_logged_nazgul_flights(state, moves, result):
    """TEACHER pure-mode executor: replay the log's per-figure Nazgul
    flights (list of {'from','to','n','nmin'}) through the engine's own
    piece model - plain Nazgul by count, named minions (Witch-king /
    Mouth; never Saruman) by nmin, mirroring the Nazgul Strike!
    executor (RB-02/RB-04).  Self-play never builds these choices."""
    import game_logic as _glnm
    from game_state import Nazgul as _NazNM
    moved = 0
    for _mvn in moves or []:
        frm = sanitize(str(_mvn.get('from', '')))
        to = sanitize(str(_mvn.get('to', '')))
        cnt = int(_mvn.get('n', 0) or 0)
        cmin = int(_mvn.get('nmin', 0) or 0)
        if not frm or not to or (cnt <= 0 and cmin <= 0):
            continue
        srck = _find_army_key(state, frm)
        srca = state.armies.get(srck) if srck else None
        if srca is None:
            continue
        movable = [l for l in srca.leaders if isinstance(l, _NazNM)]
        for fig in movable[:cnt]:
            srca.leaders.remove(fig)
            _glnm.get_or_create_army(state, to).leaders.append(fig)
            moved += 1
        if cmin > 0:
            minmov = [l for l in srca.leaders
                      if type(l).__name__ == 'Minion'
                      and not _glnm._is_saruman(l)][:cmin]
            for fig in minmov:
                srca.leaders.remove(fig)
                _glnm.get_or_create_army(state, to).leaders.append(fig)
                moved += 1
        if not srca.units and not srca.leaders:
            state.armies.pop(srck, None)
    if moved:
        result["effects"].append(
            "Moved %d Nazgul (logged flights)" % moved)
    return moved


@register_effect("The Ringwraiths Are Abroad")
def _effect_ringwraiths_abroad(state: "GameState", card, choices: dict) -> dict:
    """Move ANY OR ALL of the Nazgul (per-figure chain, 1.131 - the
    old cap of 4 left the 5th+ behind, removed QA 2026-07-11 C5);
    then move two Armies each containing a Nazgul OR attack with one
    (the 'nazgul_armies' follow-up grant).

    Heuristic: gather all Nazgul into the Shadow army closest to the
    strongest FP army; the card's attack is the nazgul_armies follow-up
    grant (the internal heuristic battle was removed, QA 2026-07-11 C11).
    """
    result = {"effects": []}

    dest_region = choices.get("nazgul_destination")

    # LEARNED PATH (amendment 1.131): 'Move any or all of the Nazgul'
    # is the per-figure flight chain (the old single-destination
    # gather-all spec collapsed every split); the 'Then, you may
    # either move two Armies each containing a Nazgul, or attack with
    # one Army containing a Nazgul' half is the existing
    # 'nazgul_armies' follow-up, deferred to the chain's end.
    import combat as _cmb131
    if _cmb131.FINAL_HITS_HOOK is None \
            and not choices.get('_from_pending') \
            and not dest_region:
        if _seed_card_nazgul_chain(
                state, card,
                followup={'kind': 'nazgul_armies', 'moves': 2}):
            result["effects"].append("Nazgul movement (learned)")
            return result

    # TEACHER pure-mode (engine-driven-resolution directive): the
    # human's LOGGED per-figure flights and the card's two army hops
    # are DECISIONS fed to the engine - no board patching.  The hops
    # execute through the certified The-Shadow-is-Moving machinery
    # (adjacency / enemy / besieged gates, overstack resolution), one
    # region each, per the card's 'move two Armies each containing a
    # Nazgul' half.  Self-play never sets these keys.
    _nmv9 = choices.get("nazgul_moves")
    _amv9 = choices.get("army_moves")
    if _nmv9 or _amv9:
        _exec_logged_nazgul_flights(state, _nmv9 or [], result)
        for _hop9 in list(_amv9 or [])[:2]:
            _frm9, _dst9 = _hop9[0], _hop9[1]
            _cmp9 = _hop9[2] if len(_hop9) > 2 else None
            _ch9 = {'from_region': _frm9, 'to_region': _dst9}
            if _cmp9:
                _ch9['moved'] = _cmp9
            _r2 = _effect_shadow_is_moving(state, card, _ch9)
            for _e9 in (_r2.get("effects") or []):
                result["effects"].append("RWA army move: %s" % _e9)
        return result

    # --- collect all Nazgul currently on the map ----------------------------
    import game_logic as _gl9
    nazgul_sources = []  # list of (region_key, nazgul_obj)
    for rkey, army in list(state.armies.items()):
        for ldr in list(army.leaders):
            if _gl9.is_nazgul_figure(ldr):
                nazgul_sources.append((rkey, ldr))

    if not nazgul_sources:
        result["effects"].append("No Nazgul on the map to move")
        return result

    # --- pick destination heuristically if not supplied ---------------------
    if not dest_region:
        # Find Shadow army adjacent to the largest FP army
        best_dest, best_score = None, -1
        for rkey, army in state.armies.items():
            if not _is_army_shadow(army) or not army.units:
                continue
            rsan = sanitize(rkey)
            for adj in state.board.adjacent(rsan):
                adj_san = sanitize(adj)
                dfn = state.armies.get(_find_army_key(state, adj_san))
                if dfn and _is_army_fp(dfn):
                    score = len(dfn.units) + len(army.units)
                    if score > best_score:
                        best_dest, best_score = rsan, score
        # Fallback: any Shadow army region
        if not best_dest:
            for rkey, army in state.armies.items():
                if _is_army_shadow(army) and army.units:
                    best_dest = sanitize(rkey)
                    break
        dest_region = best_dest

    if not dest_region:
        result["effects"].append("No valid destination for Nazgul")
        return result

    dest_san = sanitize(dest_region)

    # --- move up to 4 Nazgul to destination ---------------------------------
    moved = 0
    for src_rkey, naz in nazgul_sources:
        # "any or ALL of the Nazgul" - no cap (C5)
        src_san = sanitize(src_rkey)
        if src_san == dest_san:
            continue  # already there
        src_army = state.armies.get(_find_army_key(state, src_rkey))
        if src_army and naz in src_army.leaders:
            src_army.leaders.remove(naz)
            # Clean up empty armies
            if not src_army.units and not src_army.leaders:
                state.armies.pop(src_rkey, None)
        # Place in destination
        _army_entry_for(state, dest_san).leaders.append(naz)
        moved += 1

    result["effects"].append(f"Moved {moved} Nazgul to {dest_region}")

    # the follow-up grant IS the card's attack (C11: no internal heuristic
    # battle)
    _grant_card_followup(state, Side.SHADOW, 'nazgul_armies', moves=2)
    return result


@register_effect("The Black Captain Commands")
def _effect_black_captain_commands(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 Nazgûl with Witch-king OR move Nazgûl. Then move/attack with Witch-king army."""
    result = {"effects": []}
    action = choices.get("action")
    if action is None:
        # TEACHER pure-mode: logged flights with no explicit arm = the
        # card's MOVE half (GAME19 T4: WK+Nazgul Lorien -> Woodland
        # Realm; the blanket 'recruit' default recruited at Lorien
        # instead and stranded the Witch-king for the rest of the game)
        action = 'move' if choices.get('nazgul_moves') else 'recruit'

    if action == "recruit":
        wk_region = choices.get("witch_king_region")
        if not wk_region:
            # AI default: recruit at the Witch-king's current region
            for region_key, army in state.armies.items():
                if any(type(l).__name__ == 'Minion'
                       and 'witch' in getattr(l, 'name', '').lower()
                       for l in army.leaders):
                    wk_region = region_key
                    break
        if wk_region:
            # honest per-piece Nazgul count (honest-reporting wave 2026-07-21)
            _nz9 = sum(1 for _ in range(2)
                       if recruit_nazgul(state, wk_region) is not None)
            if _nz9:
                result["effects"].append(
                    f"Recruited {_nz9} Nazgul in {wk_region}")
            else:
                result["message"] = "Nazgul pool empty - no Nazgul recruited"
        else:
            result["message"] = "Witch-king not on the board"
    elif choices.get('_from_pending'):
        # LEARNED PATH (1.128): "move any or all of the Nazgul" is the shared
        # '__nazgul_moves__' per-figure chain (the Witch-king rides the
        # follow-up half); the "Then, move or attack with the Witch-king's
        # Army" half is DEFERRED to the chain's end via '_then_followup'.
        _fu = {'kind': 'wk_army', 'moves': 1}
        _opts = _nazgul_chain_options(state, {'__wk_ok__': 1})
        if _opts:
            for _o in _opts:
                _o['_then_followup'] = _fu
            _opts.append({'_naz_done': True, '_then_followup': _fu})
            state.pending_card_option = {
                'card_id': '__nazgul_moves__',
                'card_name': '__nazgul_moves__',
                'side': 'sh', 'decider': 'sh', 'direct': True,
                'options': _opts}
            result["effects"].append("Nazgul movement (learned)")
            return result  # follow-up granted when the chain ends
        # No plain Nazgul anywhere to move: fall through to grant the
        # Witch-king army follow-up directly (the move half was empty).
    else:
        # TEACHER / legacy fx: replay the logged per-piece flights; with no
        # choices this branch is a NO-OP and claims no effect.
        moves = choices.get("nazgul_moves") or []
        moved = 0
        for mv in moves:
            try:
                frm, to = mv.get("from"), mv.get("to")
                src_a = state.armies.get(_find_army_key(state, frm))
                if not src_a:
                    continue
                from game_state import Nazgul as _Naz
                import game_logic as _glnz
                nz = [l for l in src_a.leaders
                      if _glnz.is_nazgul_figure(l)]
                if not nz and not mv.get('nmin'):
                    continue
                dst = state.armies.get(_find_army_key(state, to))
                if dst is None:
                    from game_state import Army as _Army
                    dk = sanitize(str(to))
                    state.armies[dk] = _Army(region=state.board.region(dk))
                    dst = state.armies[dk]
                # honor the LOGGED plain-Nazgul count ('n') when present; legacy
                # callers without it keep all-at-source semantics
                _nwant9 = int(mv.get('n', len(nz)) or len(nz))
                for l in nz[:_nwant9]:
                    src_a.leaders.remove(l)
                    dst.leaders.append(l)
                    moved += 1
                # The Witch-king (a Minion, not a Nazgul object) rides the
                # logged flight when the fx counted him as moved['leader'];
                # Saruman never leaves Orthanc.
                _nmin9 = int(mv.get('nmin', 0) or 0)
                if _nmin9 > 0:
                    _minmov9 = [l for l in src_a.leaders
                                if type(l).__name__ == 'Minion'
                                and not _glnz._is_saruman(l)][:_nmin9]
                    for l in _minmov9:
                        src_a.leaders.remove(l)
                        dst.leaders.append(l)
                        moved += 1
                if not src_a.units and not src_a.leaders:
                    _sk9 = _find_army_key(state, frm)
                    if _sk9:
                        state.armies.pop(_sk9, None)
            except Exception:
                continue
        if moved:
            result["effects"].append(f"Moved {moved} Nazgûl")
        else:
            result["message"] = "no Nazgûl moves chosen"

    # Empty effects (Witch-king not on the board) lets the AI's no-op
    # filter exclude the card instead of burning a die on nothing
    _grant_card_followup(state, Side.SHADOW, 'wk_army', moves=1)
    return result


def _lidless_options(state):
    """Distinct face multisets: equal-face physical dice are exchangeable."""
    from itertools import combinations
    from game_state import ActionDieFace
    faces = sorted(d.current.name for d in state.dice.get(Side.SHADOW, [])
                   if not d.used and d.current is not None and d.current != ActionDieFace.EYE)
    return [{"dice_count": count, "dice_faces": list(selected)}
            for count in range(min(3, len(faces)) + 1)
            for selected in sorted(set(combinations(faces, count)))]


@register_effect("The Lidless Eye")
def _effect_lidless_eye(state: "GameState", card, choices: dict) -> dict:
    """Convert the selected unused faces, excluding the funding die."""
    from game_state import ActionDieFace
    result = {"effects": []}

    options = _lidless_options(state)
    requested = choices.get('dice_faces')
    if requested is not None:
        if not isinstance(requested, (list, tuple)) or any(not isinstance(f, str) for f in requested):
            raise ValueError('Lidless Eye requires a face multiset')
        selected = sorted(requested)
        count = choices.get('dice_count', len(selected))
        matches = [o for o in options if o['dice_faces'] == selected and o['dice_count'] == count]
    else:
        # Old callers may specify only a count, but it cannot choose between
        # strategically different remaining dice on the player's behalf.
        count = choices.get('dice_count', 0)
        matches = [o for o in options if o['dice_count'] == count]
    if isinstance(count, bool) or not isinstance(count, int) or len(matches) != 1:
        raise ValueError('Lidless Eye selection unavailable or ambiguous')
    selected_dice = []
    for face in matches[0]['dice_faces']:
        die = next(d for d in state.dice[Side.SHADOW]
                   if not d.used and d.current.name == face and d not in selected_dice)
        selected_dice.append(die)
    for die in selected_dice:
        die.current = ActionDieFace.EYE
        die.mark_used()
        state.hunt_box.append(die)
    converted = len(selected_dice)

    if converted > 0:
        result["effects"].append(f"Converted {converted} Shadow dice to Eyes and placed in Hunt Box")
    else:
        result["effects"].append("Converted no Shadow dice")
    return result


# ---------------------------------------------------------------------------
# Registered Card Effect Handlers - Shadow Strategy Cards
# ---------------------------------------------------------------------------

_RETURN_VALINOR_REGIONS = ('rivendell', 'lorien', 'grey_havens', 'woodland_realm')


def _return_valinor_roll_targets(state):
    """Ordered public regions and dice counts for this card's resolution."""
    targets = []
    besieged = getattr(state, 'besieged_strongholds', set())
    for name in _RETURN_VALINOR_REGIONS:
        region = sanitize(name)
        if region in besieged:
            continue
        army = next((a for key, a in state.armies.items() if sanitize(key) == region), None)
        if army is not None and _is_army_fp(army):
            count = min(5, sum(unit.nation == Nation.ELVES for unit in army.units))
            if count:
                targets.append((region, count))
    return targets


@register_effect("Return to Valinor")
def _effect_return_to_valinor(state: "GameState", card, choices: dict) -> dict:
    """Roll vs Elven armies in non-besieged Elven strongholds, hit on 6."""
    result = {"effects": []}
    elven_strongholds = _RETURN_VALINOR_REGIONS

    # USER RULING 1.254: each stronghold's hits are the FP OWNER's
    # learned allocation; when that pending opens the remaining
    # strongholds resume through _complete_card_casualty with
    # '_rtv_cas_resumed' = the stronghold just settled.
    _resume_after9 = choices.get('_rtv_cas_resumed')
    _skipping9 = _resume_after9 is not None
    _bsg9 = getattr(state, 'besieged_strongholds', set())
    for sh in elven_strongholds:
        sh_san = sanitize(sh)
        if _skipping9:
            if sh_san == sanitize(str(_resume_after9)):
                _skipping9 = False
            continue
        if sh_san in _bsg9:
            continue  # card text: only strongholds NOT under siege
        army = next((a for k, a in state.armies.items()
                     if sanitize(k) == sh_san), None)
        if army and _is_army_fp(army):
            elven_units = sum(1 for u in army.units if u.nation == Nation.ELVES)
            dice_count = min(5, elven_units)
            if dice_count > 0:
                rolls = _d6_batch(dice_count)
                _ann_rng(state, 'Return to Valinor roll', faces=rolls)
                hits = sum(1 for r in rolls if r == 6)
                import combat as _cmb_rtv
                if hits > 0 and _cmb_rtv.FINAL_HITS_HOOK is None:
                    result["effects"].append(
                        f"{sh}: Rolled {rolls}, {hits} hits")
                    if not _route_card_casualties(
                            state, sh_san, Side.FREE_PEOPLES, hits,
                            card or _card_by_name('Return to Valinor'),
                            choices, result):
                        return result
                elif hits > 0:
                    # teacher: certified logged-composition path
                    from combat import (_track_fp_casualties,
                                        reconcile_destroyed_army
                                        as _rda9v)
                    casualties = _side_filtered_kills(
                        army, Side.FREE_PEOPLES, hits, (Nation.ELVES,))
                    # FP losses are PERMANENT: book them or the units
                    # silently return to the Elven reinforcement pool
                    _track_fp_casualties(state, casualties,
                                         Side.FREE_PEOPLES)
                    result["effects"].append(f"{sh}: Rolled {rolls}, {len(casualties)} casualties")
                    # Almanac p31: a wiped army loses plain Leaders,
                    # while Companions remain. Mixed-nation survivors
                    # still support the army's Elven Leaders.
                    if not army.units:
                        _rda9v(state, sh_san, Side.FREE_PEOPLES,
                               card_name='Return to Valinor')
                    import game_logic as _gl9
                    _gl9.remove_orphaned_leaders(state)
                else:
                    # The roll IS the effect — report it even on a miss,
                    # or the dry-run no-op filter flickers with the RNG
                    # (2023-12-06 T8: the play was legal in one teacher
                    # run and filtered in the next)
                    result["effects"].append(f"{sh}: Rolled {rolls}, no hits")
    return result


@register_effect("The Fighting Uruk-hai")
def _effect_fighting_uruk_hai(state: "GameState", card, choices: dict) -> dict:
    """Isengard army besieging a stronghold assaults it: 3 MANDATORY
    rounds (USER 1.210d - after them the battle routes through NORMAL
    siege_press decisions, never auto-press), and the round-1 FP
    combat card is blocked UNLESS a Companion is in the battle (the
    printed exception, ruling 1.121 - pc key
    no_def_card_r1_unless_companion)."""
    result = {"effects": []}

    # TEACHER path: the schedule's own army_attack record synthesizes this
    # assault; opening it here would pre-empt that cursor.  Self-play keeps
    # the interactive assault (USER 1.210(d)).
    import combat as _cmbx9
    if _cmbx9.FINAL_HITS_HOOK is not None:
        return result

    atk_region = choices.get("attack_from")
    def_region = choices.get("attack_target")

    # Card text: 'if an Army containing an Isengard unit is BESIEGING a
    # Stronghold ... Attack THAT Stronghold.' The old heuristic attacked
    # any adjacent FP field army (USER 2026-07-09: it shelled a random
    # army in Dimrill Dale). Target = the besieged stronghold region
    # where the Isengard army stands (co-located siege), nothing else.
    if not atk_region or not def_region:
        bs = getattr(state, 'besieged_strongholds', set()) or set()
        for rkey, army in state.armies.items():
            rsan = sanitize(rkey)
            if rsan not in bs:
                continue
            if any(u.nation == Nation.ISENGARD for u in army.units):
                atk_region = def_region = rsan
                break

    if not atk_region or not def_region:
        result["effects"].append(
            "No Isengard army besieging a stronghold — card has no target")
        return result

    # Almanac p17: event-card attacks cannot target A Power too Great /
    # Tom Bombadil protected regions (audit 2026-07-26).
    from persistent_effects import can_shadow_attack as _csa26u
    _ok26u, _blk26u = _csa26u(state, sanitize(def_region))
    if not _ok26u:
        result["effects"].append(
            f"Fighting Uruk-hai blocked: {def_region} protected by "
            f"{_blk26u}")
        return _selfplay_refund(result)

    # P5 (USER ruling 1.112.5, almanac A2): the play counts as an
    # ATTACK for the Political Track - the defenders activate and
    # advance at declaration, BOTH modes (no campaign log plays this
    # card, so collection is unaffected; the old Adobim break came
    # from a broader placement of the advance)
    import game_logic as _glp9
    _glp9.advance_defenders_political(state, sanitize(def_region),
                                      attacker_side=Side.SHADOW)
    # 1.210(d) USER: the Shadow "must NOT press beyond the 3 mandatory
    # siege rounds."  Full-fidelity interactive siege - 3 mandatory rounds
    # (mandatory_rounds), per-round combat cards (FP blocked in round 1
    # unless a Companion is in the battle), THEN route through NORMAL combat
    # decisions (siege_press pending, Shadow decides); NEVER auto-press.
    # BOTH the teacher and self-play run the interactive machine
    # (supersedes the old auto_press=True gl_battle default).
    import ai as _ai9
    from combat import garrison_side_for as _gsf9
    _dsan9 = sanitize(def_region)
    _pc9 = {
        'attacker_region': sanitize(atk_region),
        'defender_region': _dsan9,
        'is_siege': True,
        'stronghold_region': _dsan9,
        'phase': 'combat_setup',
        'deciding_side': Side.SHADOW,
        'attacker_side': Side.SHADOW,
        'defender_side': _gsf9(state, _dsan9),
        'rounds_fought': 0,
        'mandatory_rounds': 3,
        'no_def_card_r1_unless_companion': True,
    }
    state.pending_combat = _pc9
    _ai9._run_combat_round(state, _pc9, is_siege=True)
    result["effects"].append(
        f"Fighting Uruk-hai: interactive assault on {_dsan9} "
        f"(3 mandatory rounds)")
    return result


def _denethor_targets(state):
    """Printed target is a FP Leader of any nation, never a Companion."""
    from game_state import Leader
    army = state.armies.get(_find_army_key(state, "minas tirith"))
    return [leader for leader in getattr(army, 'leaders', [])
            if type(leader) is Leader and leader.nation not in SHADOW_NATIONS]


@register_effect("Denethor's Folly")
def _effect_denethors_folly(state: "GameState", card, choices: dict) -> dict:
    """Persistent: Eliminate FP leader in MT, no FP combat cards in MT."""
    result = {"effects": []}

    # Eliminate one FP LEADER from Minas Tirith (almanac A3: Companions
    # are NOT Leaders and cannot be eliminated; a mixed siege entry also
    # holds Shadow Nazgul/Minions which must never be the target)
    mt_army = state.armies.get(_find_army_key(state, "minas tirith"))
    targets = _denethor_targets(state)
    nation = choices.get('nation')
    if nation is not None:
        requested = str(nation).lower()
        targets = [leader for leader in targets
                   if requested in (leader.nation.name.lower(), leader.nation.value.lower())]
        if not targets:
            raise ValueError("Denethor's Folly leader target unavailable")
    elif len({leader.nation for leader in targets}) > 1:
        raise ValueError("Denethor's Folly requires a leader nation")
    if targets:
        leader = targets[0]
        mt_army.leaders.remove(leader)
        from combat import _track_fp_leader_deaths
        _track_fp_leader_deaths(state, [leader])
        result['effects'].append(f'Eliminated a {leader.nation.value} Leader in Minas Tirith')

    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Denethor's Folly in play - no FP Combat cards in MT battles")
    return result


@register_effect("The Day Without Dawn")
def _effect_day_without_dawn(state: "GameState", card, choices: dict) -> dict:
    """Discard all unused FP Will of the West dice.  Removal is correct
    (the RESULTS are lost this turn; calculate_bonus_dice rebuilds the
    pool to target count at the next turn boundary) but it SHIFTS the
    indices of every later FP die, and two pendings store an integer
    die index across applies (pending_army_move_die /
    pending_muster_die, B1 2026-08-24) - re-point them."""
    from game_state import ActionDieFace
    result = {"effects": []}
    fp_dice = state.dice.get(Side.FREE_PEOPLES, [])
    removed_idx = [i for i, die in enumerate(fp_dice)
                   if die.current == ActionDieFace.WILL_OF_WEST
                   and not die.used]
    for i in reversed(removed_idx):
        fp_dice.pop(i)
    removed = len(removed_idx)
    if removed:
        for _attr, _sattr in (
                ('pending_army_move_die', 'pending_army_move_side'),
                ('pending_muster_die', 'pending_muster_side')):
            _v = getattr(state, _attr, None)
            if isinstance(_v, int) \
                    and getattr(state, _sattr, None) \
                    == Side.FREE_PEOPLES:
                setattr(state, _attr,
                        _v - sum(1 for i in removed_idx if i < _v))
        result["effects"].append(
            f"Discarded {removed} Will of the West dice")
    else:
        result["message"] = "no unused Will of the West dice"  # iter 39
    return result


@register_effect("Threats and Promises")
def _effect_threats_and_promises(state: "GameState", card, choices: dict) -> dict:
    """Persistent: FP cannot advance passive nations with Muster dice."""
    result = {"effects": []}
    _handle_persistent_card(state, Side.SHADOW, card, result)
    result["effects"].append("Threats and Promises in play - FP cannot advance passive nations")
    return result


@register_effect("Stormcrow")
def _effect_stormcrow(state: "GameState", card, choices: dict) -> dict:
    """Retreat FP nation 1 step, FP must eliminate 1 unit/leader of that nation."""
    result = {"effects": []}
    # STAGE 2 (P1, USER ruling 1.112.3): the FP player picked the piece
    # to eliminate (cross-side direct pending set by stage 1)
    _elim = choices.get("eliminate")
    if _elim:
        nation = NATION_KEYWORDS.get(str(_elim.get('nation', '')).lower())
        kind = str(_elim.get('kind', ''))
        region = sanitize(str(_elim.get('region', '')))
        done = False
        for k, a in state.armies.items():
            if sanitize(k) != region:
                continue
            if kind == 'leader':
                for l in list(a.leaders):
                    if getattr(l, 'nation', None) == nation \
                            and type(l).__name__ == 'Leader':
                        a.leaders.remove(l)
                        from combat import _track_fp_leader_deaths
                        _track_fp_leader_deaths(state, [l])  # W6 BUG-2
                        done = True
                        break
            else:
                want = (UnitKind.ELITE if kind == 'elite'
                        else UnitKind.REGULAR)
                from combat import _track_fp_casualties
                for u in list(a.units):
                    if u.nation == nation and u.kind == want:
                        a.units.remove(u)
                        _track_fp_casualties(state, [u],
                                             Side.FREE_PEOPLES)
                        done = True
                        break
            break
        if done:
            import game_logic as _gl9
            _gl9.remove_orphaned_leaders(state)
            result["effects"].append(
                f"FP eliminated a {nation.value if nation else '?'} "
                f"{kind} in {region}")
        else:
            result["message"] = f"Elimination target gone ({region})"
        return result
    nation_name = choices.get("nation")
    if not nation_name:
        # Rules-honest default: the requirement names the nation — the
        # one whose borders hold the Ring-bearers (or a Companion) and
        # which is not yet At War (2023-12-06 T3: fellowship inside
        # Elven borders -> Elves).  No-op plays get filtered.
        from game_logic import fellowship_region
        try:
            fsp_nat = _get_region_nation(fellowship_region(state))
        except Exception:
            fsp_nat = None
        if fsp_nat:
            if fsp_nat.lower().strip() == 'the north':
                fsp_nat = 'north'
            nat = NATION_KEYWORDS.get(fsp_nat.lower()) or parse_nation(fsp_nat)
            if nat in FP_NATIONS and \
                    state.political.boxes.get(nat, 0) < 3:
                nation_name = fsp_nat
    if nation_name and nation_name.lower().strip() == 'the north':
        nation_name = 'north'
    if nation_name:
        nation = NATION_KEYWORDS.get(nation_name.lower())
        if nation and nation in FP_NATIONS:
            if nation not in _stormcrow_valid_nations(state):
                result["message"] = (
                    f"{nation.value} holds neither the Ring-bearers nor "
                    f"a Companion (or is At War) - invalid Stormcrow "
                    f"target")
                return result
            new_pos = retreat_nation(state, nation, 1)
            result["effects"].append(
                f"Retreated {nation.value} to {new_pos}")
            # almanac M6: the FP player eliminates one Leader or Army
            # unit of that nation.  P1 (USER ruling 1.112.3): via the
            # pending path this is the FP player's LEARNED cross-side
            # decision (stage-2 direct pending); direct calls (teacher
            # fx / GUI) keep the preserve-elites heuristic.
            if choices.get('_from_pending'):
                _opts9 = []
                _tag9 = ('north' if nation == Nation.THE_NORTH
                         else nation.value.lower())
                for k, a in state.armies.items():
                    rsan9 = sanitize(k)
                    kinds9 = set()
                    for u in a.units:
                        if u.nation == nation:
                            kinds9.add('elite' if u.kind == UnitKind.ELITE
                                       else 'regular')
                    for l in a.leaders:
                        if getattr(l, 'nation', None) == nation \
                                and type(l).__name__ == 'Leader':
                            kinds9.add('leader')
                    for kd9 in sorted(kinds9):
                        _opts9.append({'eliminate': {
                            'nation': _tag9, 'kind': kd9,
                            'region': rsan9}})
                if _opts9:
                    state.pending_card_option = {
                        'card_id': getattr(card, 'card_id', 'Stormcrow'),
                        'card_name': 'Stormcrow',
                        'side': 'sh', 'decider': 'fp', 'direct': True,
                        'options': _opts9}
                    result["effects"].append(
                        "FP must eliminate one piece of that nation "
                        "(pending choice)")
                return result
            from combat import _track_fp_casualties
            for kind_pick in ('regular', 'leader', 'elite'):
                done = False
                for k, a in state.armies.items():
                    if kind_pick == 'leader':
                        for l in list(a.leaders):
                            if getattr(l, 'nation', None) == nation                                     and type(l).__name__ == 'Leader':
                                a.leaders.remove(l)
                                from combat import _track_fp_leader_deaths
                                _track_fp_leader_deaths(state, [l])  # W6 BUG-2
                                result["effects"].append(
                                    f"FP eliminated a {nation.value} "
                                    f"Leader in {sanitize(k)}")
                                done = True
                                break
                    else:
                        want = (UnitKind.REGULAR if kind_pick == 'regular'
                                else UnitKind.ELITE)
                        for u in list(a.units):
                            if u.nation == nation and u.kind == want:
                                a.units.remove(u)
                                _track_fp_casualties(state, [u],
                                                     Side.FREE_PEOPLES)
                                result["effects"].append(
                                    f"FP eliminated a {nation.value} "
                                    f"{kind_pick} in {sanitize(k)}")
                                done = True
                                break
                    if done:
                        break
                if done:
                    import game_logic as _gl9
                    _gl9.remove_orphaned_leaders(state)
                    break
    return result


def _stormcrow_valid_nations(state):
    """Not-At-War FP nations whose borders hold the Ring-bearers or a
    separated Companion (the card's own requirement text)."""
    from game_state import Companion as _Comp
    import game_logic as _gl9
    regions = [_gl9.fellowship_region(state)]
    for k, a in state.armies.items():
        if any(isinstance(l, _Comp) for l in a.leaders):
            regions.append(sanitize(k))
    out = set()
    for r in regions:
        nm = _get_region_nation(r)
        if nm:
            # board faction 'the north' vs NATION_KEYWORDS key 'north'
            nm = nm.lower().strip()
            nm = 'north' if nm == 'the north' else nm
        nat = NATION_KEYWORDS.get(nm) if nm else None
        if nat and nat in FP_NATIONS                 and not state.political.is_at_war(nat):
            out.add(nat)
    return out


@register_effect("Shadows Gather")
def _effect_shadows_gather(state: "GameState", card, choices: dict) -> dict:
    """Move Shadow army up to 3 regions to join another Shadow army."""
    result = {"effects": []}
    from_region = choices.get("from_region")
    to_region = choices.get("to_region")
    if from_region and to_region:
        from army_card_cargo import stage
        choices = stage(state, card, choices, from_region)
        if choices is None:
            return {'effects': ['Choose the figures moving with this army']}

    if not (from_region and to_region):
        # LOG = TRUTH: the teacher never auto-picks a join move (the log carries
        # the move or it was declined); self-play auto-picks (the card should
        # act).
        import combat as _cmb_sg
        if _cmb_sg.FINAL_HITS_HOOK is not None:
            return result
        opts = _sg_join_options(state, 3)
        if not opts:
            result["message"] = "No legal Shadows Gather move"
            return result
        from_region, to_region = (opts[0]['from_region'],
                                  opts[0]['to_region'])
    bs = getattr(state, 'besieged_strongholds', set())
    dests0 = {sanitize(k) for k, a in state.armies.items()
              if any(u.nation in SHADOW_NATIONS for u in a.units)
              and (sanitize(k) not in bs
                   or not _besieged_garrison_source(
                       state, sanitize(k), Side.SHADOW))}
    # (besieged-region entries qualify when the Shadow army there is
    # the BESIEGER - joining besiegers outside the walls is a normal
    # move; only a besieged SH garrison is unjoinable.  GAME19 T5 TSL:
    # Moria's two fresh elites reinforce the Lorien siege.)
    # USER 1.208n: honor the learned initial split (whole army when the
    # keys are absent, keeping the legacy path byte-identical).
    _nr = choices.get("n_regular")
    _ne = choices.get("n_elite")
    _sg_join_move(state, from_region, to_region, 3, result, dests0,
                  n_regular=(int(_nr) if _nr is not None else None),
                  n_elite=(int(_ne) if _ne is not None else None),
                  cargo=choices.get('cargo'))
    if not result["effects"]:
        result["message"] = "No legal Shadows Gather move"
    return result


@register_effect("The Shadow Lengthens")
def _effect_shadow_lengthens(state: "GameState", card, choices: dict) -> dict:
    """Move 2 Shadow armies up to 2 regions each to join other Shadow armies."""
    result = {"effects": []}

    # LEARNED path (self-play / pending pick): each of the (up to two)
    # moves is an INDEPENDENT chained pending (amendment 1.117, the TSiM
    # pattern).  The choice arrives via the card_option executor which
    # injects '_from_pending'; the TEACHER path (logged choices / GUI
    # direct calls) uses the same simultaneous mover below.
    if choices.get('_from_pending'):
        return _tsl_learned_chain(state, card, choices)

    moves = []
    for suffix in ('', '_2'):
        source = choices.get('from_region' + suffix)
        target = choices.get('to_region' + suffix)
        if not suffix:
            source = source or choices.get('from_region_1')
            target = target or choices.get('to_region_1')
        if source and target:
            moves.append([source, target, choices.get('n_regular' + suffix),
                          choices.get('n_elite' + suffix), choices.get('cargo' + suffix)])
    if not moves:
        result['message'] = 'No movements selected'
        return _selfplay_refund(result)
    _tsl_execute_staged(state, moves, result)
    return result


@register_effect("The Shadow is Moving")
def _effect_shadow_is_moving(state: "GameState", card, choices: dict) -> dict:
    """Move up to 4 Shadow armies 1 region each."""
    from staged_army_cards import handler, execute_staged
    if choices.get('_from_pending'):
        if not choices.get('_tsim_done'):
            from army_card_cargo import stage
            choices = stage(state, card, choices, choices['from_region'])
            if choices is None:
                return {'effects': ['Choose the figures moving with this army']}
        return handler(state, card, choices)
    result = {"effects": []}
    # An explicit aggregate caller may end without movement.
    if choices.get('_tsim_done'):
        result["effects"].append("The Shadow is Moving: movement ends")
        return result
    # collect up to four (from, to) single-region moves; the primary
    # option keys double as move 1 so the option spec stays uniform.
    # An optional per-move composition ('moved'/'moved_i': counts dict)
    # is the TEACHER pure-mode key - the humans may move a SUBSET of
    # the stack (GAME19 T3: one S&E regular stayed at Near Harad);
    # self-play never builds it, whole-army semantics unchanged.
    mvs = []
    f0 = choices.get("from_region")
    t0 = choices.get("to_region")
    if f0 and t0:
        mvs.append((sanitize(f0), sanitize(t0), choices.get("moved")))
    for i in range(1, 5):
        f = choices.get(f"from_region_{i}")
        t = choices.get(f"to_region_{i}")
        if f and t:
            mvs.append((sanitize(f), sanitize(t),
                        choices.get(f"moved_{i}")))
    mvs = mvs[:4]
    if not mvs:
        result["message"] = "No movements selected"  # no-op (iter 39)
        return result
    result = execute_staged(state, mvs, result)
    resolve_overstacks(state, learned=False)
    if not result['effects']:
        result['message'] = 'No movements selected'
    return result


@register_effect("Corsairs of Umbar")
def _effect_corsairs_of_umbar(state: "GameState", card, choices: dict) -> dict:
    """Land selected Umbar cargo, then resolve the Almanac A10 battle."""
    if '_corsairs_landing_pc' in choices:
        from corsairs_landing import remove
        return remove(state, card, choices)
    result = {"effects": []}
    target = choices.get("target_region")
    if not target:
        result["message"] = "No target region selected"  # no-op (iter 39)
        return _selfplay_refund(result)
    from army_card_cargo import stage
    choices = stage(state, card, choices, 'umbar')
    if choices is None:
        return {'effects': ['Choose the figures sailing from Umbar']}
    tsan = sanitize(target)
    from combat import garrison_side_for
    import game_logic as _gl9
    umb = sanitize("Umbar")
    # almanac A10a: Umbar must be free for movement - a Shadow army
    # BESIEGED inside Umbar by an FP army may not move (besieging an
    # FP garrison in Umbar is fine)
    if umb in getattr(state, 'besieged_strongholds', set())             and garrison_side_for(state, umb) == Side.SHADOW:
        result["message"] = "Shadow army besieged inside Umbar - "                             "the move is not permitted"
        return _selfplay_refund(result)
    if not _shadow_units_at(state, umb):
        result["message"] = "No Shadow army in Umbar"
        return _selfplay_refund(result)
    _gc = board.regions.get(tsan, {})
    if board.fid_to_name.get(_gc.get('parent_id')) != 'gondor' \
            or tsan not in ("anfalas", "dol amroth", "lamedon",
                            "pelargir"):
        # adversarial repair 2026-07-11: COASTAL Gondor only (rulebook
        # map coastal set filtered to Gondor-parent regions)
        result["message"] = f"{tsan} is not a Gondor coastal region"
        return _selfplay_refund(result)
    tkey = _find_army_key(state, tsan)
    tarmy = state.armies.get(tkey) if tkey else None
    fp_there = bool(tarmy and any(u.nation in FP_NATIONS
                                  for u in tarmy.units))
    if fp_there:
        # 1.210(b) USER: movement FULLY COMPLETES first (the corsairs move INTO
        # the target), THEN the co-located battle runs through pending_combat:
        # the FP DEFENDER must be offered a retreat (field, or withdraw into
        # the stronghold) and THE ATTACKER CANNOT VOLUNTARILY CEASE
        # (attacker_no_retreat).  EXCEPTION (example c): landing into a
        # stronghold ALREADY UNDER SIEGE starts a normal siege battle
        # (no-cease does not apply).
        _pre_siege9 = (tsan in getattr(state, 'besieged_strongholds', set()))
        _relief9 = _pre_siege9 and garrison_side_for(state, tsan) == Side.SHADOW
        _garrison9 = None
        if _relief9:
            from army_card_cargo import inventory, freeze
            _garrison9 = freeze(inventory(state, tsan))
        moved = _direct_army_move(state, umb, tsan, Side.SHADOW,
                                 cargo=choices.get('cargo'), allow_overstack=True,
                                 defer_overstack=_relief9)
        if not moved:
            # A blocked landing means NO attack was declared - refund with no
            # political advance (review P2-1)
            result["message"] = (
                "Corsairs landing at %s blocked (protected or full "
                "destination)" % tsan)
            return _selfplay_refund(result)
        # 1.112.5: the battle portion is an ATTACK for the Political Track -
        # advanced only once the landing REALLY happened
        _gl9.advance_defenders_political(state, tsan,
                                         attacker_side=Side.SHADOW)
        # TEACHER path: gate ONLY the battle-open (the schedule's army_attack
        # record synthesizes it); the LANDING is real and must still run.
        # Self-play keeps the interactive co-located battle (USER 1.210(b)).
        import combat as _cmbx9
        if _cmbx9.FINAL_HITS_HOOK is not None:
            result["effects"].append(
                f"Corsairs land at {tsan} (battle synthesized by schedule)")
            return result
        import ai as _ai9
        _pc9 = {
            'attacker_region': tsan,
            'defender_region': tsan,   # co-located (moved in)
            'is_siege': bool(_pre_siege9 and not _relief9),
            'stronghold_region': tsan if _gc.get('is_stronghold')
                                  else None,
            'phase': 'combat_setup',
            'deciding_side': Side.SHADOW,
            'attacker_side': Side.SHADOW,
            'defender_side': Side.FREE_PEOPLES,
            'rounds_fought': 0,
            'attacker_no_retreat': (not _pre_siege9 or _relief9),
        }
        if _relief9:
            # A10(ii): the original SH garrison remains inside. Only the
            # transported force fights the FP besiegers outside the walls.
            _pc9['corsairs_relief_garrison'] = _garrison9
        # A10(iii): joining existing besiegers resolves the ten-unit limit
        # before either player selects a combat card or the first round.
        getattr(state, '_overstack_regions', set()).discard(tsan)
        from corsairs_landing import begin
        begin(state, card, _pc9)
        result["effects"].append(
            f"Corsairs land at {tsan}: co-located battle "
            f"({'siege relief' if _pre_siege9 else 'Shadow cannot retreat'})")
        return result
    moved = _direct_army_move(state, umb, tsan, Side.SHADOW,
                             cargo=choices.get('cargo'), allow_overstack=True)
    if not moved:
        # A blocked landing means the card had NO effect - refund (review P2-1)
        result["message"] = (
            "Corsairs landing at %s blocked (no movable units, "
            "protected or full destination)" % tsan)
        return _selfplay_refund(result)
    result["effects"].append(f"Corsairs moved from Umbar to {tsan}")
    # almanac A10b: entering an empty Gondor region activates Gondor
    # (guaranteed here); the settlement CAPTURE-advance (rulebook p.35,
    # ONE step) already fired natively inside _direct_army_move ->
    # _check_settlement_capture - never advanced a second time here.
    from persistent_effects import can_activate_nation as _can9
    _ok9, _ = _can9(state, Nation.GONDOR, "attack", region=tsan)
    if _ok9:
        state.political.active[Nation.GONDOR] = True
        result["effects"].append("Gondor activated")
    return result


def _is_army_fp(army) -> bool:
    """Check if an army belongs to Free Peoples."""
    if not army.units:
        return False
    return army.units[0].nation in FP_NATIONS


def _region_free_for_shadow(state: "GameState", rkey: str) -> bool:
    """No Free Peoples units at rkey and not an unconquered FP stronghold
    ('free region' for Shadow recruitment purposes)."""
    san = sanitize(rkey)
    # P5 (QA Horde CASE 7, almanac M21): an FP-CAPTURED Shadow
    # settlement is NOT free (the control marker suffices - no FP army
    # required); EXCEPTION: a captured region where a Shadow army
    # besieges the FP garrison inside is free again (almanac Moria
    # rule, same semantics as _rage_regions._ok9).
    _cap_fp9 = state.enemy_controlled.get(Side.FREE_PEOPLES, set())
    if san in _cap_fp9 and san not in getattr(
            state, 'besieged_strongholds', set()):
        return False
    for k, a in state.armies.items():
        if sanitize(k) != san:
            continue
        if any(u.nation not in SHADOW_NATIONS for u in a.units):
            return False
    if _is_fp_nation_region(san) and \
            san not in state.enemy_controlled.get(Side.SHADOW, set()):
        meta = board.regions.get(san, {})
        if meta.get('is_stronghold') or meta.get('is_city'):
            return False
    return True


def _shadow_units_at(state, san):
    a = state.armies.get(_find_army_key(state, san))
    return bool(a and any(u.nation in SHADOW_NATIONS for u in a.units))


def _sg_join_options(state, max_steps):
    """Shadows Gather / The Shadow Lengthens choices: move a Shadow army
    up to max_steps regions through friendly or empty, unbesieged
    intermediates to a region ALREADY holding another Shadow army that
    is not under siege (qualification at the start of the action,
    almanac A7/A8)."""
    # ONE body with the TSL chain (_tsl_dests0): a Shadow BESIEGER stack
    # qualifies; only a besieged SH garrison is unjoinable (1.211h).
    from shadow_join_rules import select as select_join
    dests0 = _tsl_dests0(state)
    singles = []
    for k, a in state.armies.items():
        frm = sanitize(k)
        # Only a besieged GARRISON may not march out; a BESIEGER
        # lifting its siege is legal (matches _sg_join_move's guard).
        if _besieged_garrison_source(state, frm, Side.SHADOW):
            continue
        if not any(u.nation in SHADOW_NATIONS for u in a.units):
            continue
        reach = _army_reachable_free(state, frm, max_steps, Side.SHADOW)
        for d in sorted(reach & dests0):
            if d != frm and select_join(state, frm, d, max_steps,
                                        [u for u in a.units if u.nation in SHADOW_NATIONS])[0]:
                singles.append({"from_region": frm, "to_region": d})
    return singles


def _sg_join_move(state, frm, to, max_steps, result, dests0,
                  n_regular=None, n_elite=None, cargo=None):
    """Validated single join-move + walk-through capture of empty
    settlements (almanac A7/A8 headline: control is taken of empty FP
    settlements AND FP-controlled Shadow settlements moved through).
    USER 1.208n: an optional (n_regular, n_elite) initial split moves
    only the chosen composition; the remainder stays behind."""
    from combat import _check_settlement_capture
    frm, to = sanitize(frm), sanitize(to)
    # 1.211h / row 43: a besieged Shadow garrison cannot march out of
    # the walls via a join-move (covers TSL single + Shadows Gather).
    if _besieged_garrison_source(state, frm, Side.SHADOW):
        result["effects"].append(
            f"{frm} is a besieged garrison - cannot move out")
        return False
    if to not in dests0:
        result["effects"].append(
            f"{to} has no (unbesieged) Shadow army - move refused")
        return False
    if to not in _army_reachable_free(state, frm, max_steps,
                                      Side.SHADOW):
        result["effects"].append(
            f"{to} not reachable within {max_steps} free regions "
            f"from {frm}")
        return False
    # identify the free intermediate(s) for walk-through capture (ONE BFS
    # body: _free_path_mids)
    from shadow_join_rules import select as select_join
    from army_card_cargo import figures, selected, key, freeze
    from collections import Counter
    units, leaders = figures(state, frm)
    if cargo is not None:
        units, leaders = selected(state, frm, cargo)
    units, mids = select_join(state, frm, to, max_steps, units, exact=cargo is not None)
    if not units:
        if cargo is not None:
            raise ValueError('Selected cargo cannot cross political borders')
        result['effects'].append(f'{frm} has no army units allowed across the route borders')
        return False
    if n_regular is not None or n_elite is not None:
        limits = {UnitKind.REGULAR: n_regular, UnitKind.ELITE: n_elite}
        chosen = []
        for unit in units:
            left = limits[unit.kind]
            if left is None or left > 0:
                chosen.append(unit)
                if left is not None:
                    limits[unit.kind] -= 1
        units = chosen
    if not units:
        return False
    cargo = freeze(Counter([key(u, True) for u in units] + [key(l) for l in leaders]))
    moved = _direct_army_move(state, frm, to, Side.SHADOW,
                              allow_overstack=True, cargo=cargo)
    if moved:
        import game_logic as _gl_sg
        for m in mids:
            _check_settlement_capture(state, m, Side.SHADOW)
            # rulebook 370-371: a walked-through region is ENTERED
            _gl_sg._check_political_activation_for_entry(state, m,
                                                         Side.SHADOW)
        result["effects"].append(f"Army moved from {frm} to join {to}")
        return True
    result["effects"].append(f"No movable units in {frm}")
    return False


def _free_path_mids(state, frm, to, max_steps):
    """Shadow join-path intermediates, excluding enemy armies and sieges.
    Friendly armies are traversable without joining or picking up figures.
    Destination legality is checked by the caller; [] also means no path.
    """
    from collections import deque
    from combat import side_of_nation
    frm, to = sanitize(frm), sanitize(to)
    blocked = getattr(state, 'besieged_strongholds', set())
    seen = {frm}
    queue = deque([(frm, [], 0)])
    while queue:
        current, intermediates, distance = queue.popleft()
        if distance >= max_steps:
            continue
        for adjacent in state.board.adjacent(current):
            region = sanitize(adjacent)
            if region in seen:
                continue
            seen.add(region)
            if region == to:
                return intermediates
            army = state.armies.get(_find_army_key(state, region))
            if region in blocked or (army and any(side_of_nation(unit.nation) != Side.SHADOW for unit in army.units)):
                continue
            queue.append((region, intermediates + [region], distance + 1))
    return []


def _rage_split_options(state, dest: str) -> list:
    """Every legal source and Regular/Elite composition, including no move."""
    from combat import side_of_nation
    dsan = sanitize(dest)

    def available(src):
        army = state.armies.get(_find_army_key(state, src))
        units = army.units if army is not None and src != dsan else []
        return tuple(sum(u.nation == Nation.ISENGARD and u.kind == kind
                         for u in units)
                     for kind in (UnitKind.REGULAR, UnitKind.ELITE))

    nr, ne = available('north dunland')
    sr, se = available('south dunland')
    army = state.armies.get(_find_army_key(state, dsan))
    occupied = sum(side_of_nation(u.nation) == Side.SHADOW
                   for u in army.units) if army else 0
    room = max(0, _stack_cap(state, dsan, Side.SHADOW) - occupied)
    options = []
    for nn in range(min(nr + ne, 4, room) + 1):
        for ns in range(min(sr + se, 4 - nn, room - nn) + 1):
            for en in range(max(0, nn - nr), min(ne, nn) + 1):
                for es in range(max(0, ns - sr), min(se, ns) + 1):
                    options.append({'rage_split': {'n_north': nn, 'n_south': ns,
                                    'n_north_elite': en, 'n_south_elite': es},
                                    '_rage_dest': dsan})
    return options


def _rage_apply_split(state, rsplit: dict, dst: str) -> int:
    """Validate the entire transfer before moving any unit.

    Historical count-only callers retain Regulars-first composition. New legal
    actions specify both Elite counts, so recorded and learned choices agree.
    """
    import numbers
    import game_logic
    dst = sanitize(str(dst))
    normalized = {}
    selections = []
    for src, nkey, ekey in (('north dunland', 'n_north', 'n_north_elite'),
                            ('south dunland', 'n_south', 'n_south_elite')):
        want = rsplit.get(nkey, 0)
        if isinstance(want, bool) or not isinstance(want, numbers.Integral) or not 0 <= want <= 4:
            raise RuntimeError('Invalid Rage troop count')
        key = _find_army_key(state, src)
        army = state.armies.get(key)
        units = army.units if army is not None and src != dst else []
        regulars = [u for u in units if u.nation == Nation.ISENGARD and u.kind == UnitKind.REGULAR]
        elites = [u for u in units if u.nation == Nation.ISENGARD and u.kind == UnitKind.ELITE]
        elite_count = rsplit.get(ekey, max(0, want - len(regulars)))
        if (isinstance(elite_count, bool) or not isinstance(elite_count, numbers.Integral)
                or not 0 <= elite_count <= want):
            raise RuntimeError('Invalid Rage Elite count')
        if elite_count > len(elites) or want - elite_count > len(regulars):
            raise RuntimeError('Rage composition unavailable: refusing the partial split')
        normalized[nkey] = int(want)
        normalized[ekey] = int(elite_count)
        selections.append((key, army, regulars[:want - elite_count] + elites[:elite_count]))
    if {'rage_split': normalized, '_rage_dest': dst} not in _rage_split_options(state, dst):
        raise RuntimeError('Rage split exceeds movement or stacking limit')
    for key, army, selected in selections:
        for unit in selected:
            army.units.remove(unit)
            game_logic.get_or_create_army(state, dst).units.append(unit)
        if army is not None and not army.units and not army.leaders:
            state.armies.pop(key, None)
    game_logic.remove_orphaned_leaders(state)
    return normalized['n_north'] + normalized['n_south']


@register_effect("Rage of the Dunlendings")
def _effect_rage_of_dunlendings(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 Isengard regulars adjacent to Dunland, then optionally move
    up to 4 Isengard units from N/S Dunland to that region (1.210(c): a
    single learned split decision, including the option NOT to move)."""
    result = {"effects": []}
    _rsplit = choices.get('rage_split')
    _region = choices.get("recruit_region")
    # STAGE 2 (self-play grid resolution): rage_split arrived via the
    # pending path with NO recruit_region - execute the chosen split move.
    if _rsplit is not None and not _region:
        dst = sanitize(str(choices.get('_rage_dest', '')))
        _mvd = _rage_apply_split(state, _rsplit, dst)
        result["effects"].append(
            f"Rage: moved {_mvd} Isengard unit(s) to {dst}")
        return result
    # STAGE 1: recruit.  almanac: with no Isengard Regular in
    # reinforcements, NO part of the card can be completed.
    from ai import _can_recruit
    if not _can_recruit(state, Nation.ISENGARD, 'regular'):
        result["message"] = ("No Isengard Regulars in reinforcements - "
                             "no part of the card can be completed")
        return result
    region = _region
    if not region:
        cands = _rage_regions(state)
        region = cands[0] if cands else None
    if region and sanitize(region) not in _rage_regions(state):
        result["message"] = f"{region} is not a legal Rage region"
        return result
    if region:
        _u1 = recruit_unit(state, region, Nation.ISENGARD,
                           UnitKind.REGULAR)
        _u2 = recruit_unit(state, region, Nation.ISENGARD,
                           UnitKind.REGULAR)
        _n = sum(1 for u in (_u1, _u2) if u is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} Isengard Regular(s) in {region}")
            # 1.210(c): the 'move up to 4' clause is ONE learned grid
            # decision.  SELF-PLAY: open the split pending (resolved by
            # STAGE 2 above).  TEACHER: the single play carries rage_split
            # (its presence marks the teacher single-call), so the pending
            # is NOT opened - the piece moves replay from the log's own fx
            # records and the grid sample is emitted by rage_chain_cb.
            if _rsplit is None and choices.get('_from_pending'):
                _opts9 = _rage_split_options(state, sanitize(region))
                if _opts9:
                    state.pending_card_option = {
                        'card_id': getattr(card, 'card_id',
                                           'Rage of the Dunlendings'),
                        'card_name': 'Rage of the Dunlendings',
                        'side': 'sh', 'decider': 'sh', 'direct': True,
                        'options': _opts9}
            elif _rsplit is not None \
                    and choices.get('_pure_exec_split'):
                # PURE MODE teacher single-call: the Dunland sweep is
                # part of the same decision and fx replication is OFF -
                # execute the split through the engine (GAME19 T7: the
                # Dunlands stayed full and Gap of Rohan ran 4 short)
                _rmv9 = choices.get('rage_moves')
                if _rmv9:
                    # exact logged sweep (incl. between-Dunland
                    # transfers the n_north/n_south grid cannot carry)
                    from combat import side_of_nation as _son_rg
                    import game_state as _gs_rg
                    _tot9 = 0
                    for _f9r, _t9r, _nr9, _ne9 in _rmv9:
                        _sk9 = _find_army_key(state, _f9r)
                        _sa9 = state.armies.get(_sk9) if _sk9 else None
                        if _sa9 is None:
                            continue
                        _dd9 = _army_entry_for(state, sanitize(_t9r))
                        _wr9, _we9 = int(_nr9), int(_ne9)
                        for _u9 in list(_sa9.units):
                            if _u9.nation != Nation.ISENGARD:
                                continue
                            if _u9.kind == _gs_rg.UnitKind.REGULAR \
                                    and _wr9 > 0:
                                _sa9.units.remove(_u9)
                                _dd9.units.append(_u9)
                                _wr9 -= 1
                                _tot9 += 1
                            elif _u9.kind == _gs_rg.UnitKind.ELITE \
                                    and _we9 > 0:
                                _sa9.units.remove(_u9)
                                _dd9.units.append(_u9)
                                _we9 -= 1
                                _tot9 += 1
                        if not _sa9.units and not _sa9.leaders:
                            state.armies.pop(_sk9, None)
                    if _tot9:
                        result["effects"].append(
                            "Rage: moved %d Isengard unit(s) (logged "
                            "sweep)" % _tot9)
                else:
                    _mvd9 = _rage_apply_split(state, _rsplit,
                                              sanitize(str(region)))
                    if _mvd9:
                        result["effects"].append(
                            "Rage: moved %d Isengard unit(s) to %s"
                            % (_mvd9, region))
        else:
            result["message"] = f"Recruitment blocked in {region}"
    else:
        result["message"] = "No free Rage region"
    return result


def _rage_regions(state):
    """Legal Rage of the Dunlendings recruit regions: free regions
    adjacent to North/South Dunland (impassable borders excluded -
    Orthanc is NOT valid, almanac), plus Moria and N/S Dunland
    themselves when free."""
    _ec_fp = state.enemy_controlled.get(Side.FREE_PEOPLES, set())
    _bs9 = getattr(state, 'besieged_strongholds', set())

    def _ok9(r):
        if r in _ec_fp:
            # captured region: legal only while a Shadow army besieges
            # the FP garrison inside (almanac Moria exception)
            return r in _bs9
        return _region_free_for_shadow(state, r)

    cands = []
    for d in ("north dunland", "south dunland", "moria"):
        if _ok9(d) and d not in cands:
            cands.append(d)
    for d in ("north dunland", "south dunland"):
        for nb in sorted(board.adj.get(d, ())):
            nbs = sanitize(nb)
            if nbs == 'orthanc':
                continue  # impassable Dunland border (almanac)
            if nbs not in cands and _ok9(nbs):
                cands.append(nbs)
    return cands


@register_effect("Return of the Witch-king")
def _effect_return_of_witch_king(state: "GameState", card, choices: dict) -> dict:
    """Move Witch-king to Angmar, recruit 2 regulars + 1 elite there."""
    result = {"effects": []}
    # USER FIX 2026-07-08: actually MOVE the Witch-king to Angmar.
    from game_state import Army as _Army
    ang = sanitize("Angmar")
    wk = None
    for rk, a in list(state.armies.items()):
        for l in list(a.leaders):
            _nm9 = re.sub(r'[^a-z]', '',
                          getattr(l, 'name', '').lower())
            if 'witchking' in _nm9:
                wk = l
                a.leaders.remove(l)
                if not a.units and not a.leaders:
                    state.armies.pop(rk, None)
                break
        if wk is not None:
            break
    if wk is not None:
        if ang not in state.armies:
            state.armies[ang] = _Army(region=state.board.region(ang))
        state.armies[ang].leaders.append(wk)
        result["effects"].append("Witch-king moved to Angmar")
    # Honest per-piece counts (a blocked placement is not an effect)
    _nr9 = sum(1 for _ in range(2)
               if recruit_unit(state, "Angmar", Nation.SAURON,
                               UnitKind.REGULAR) is not None)
    _ne9 = 1 if recruit_unit(state, "Angmar", Nation.SAURON,
                             UnitKind.ELITE) is not None else 0
    if _nr9 or _ne9:
        result["effects"].append(
            "Recruited %d Sauron Regular(s) and %d Elite(s) in Angmar"
            % (_nr9, _ne9))
    if not result["effects"]:
        result["message"] = "Return of the Witch-king: no effect"
        return _selfplay_refund(result)
    return result


@register_effect("Half-orcs and Goblin-men")
def _effect_half_orcs(state: "GameState", card, choices: dict) -> dict:
    """Recruit Isengard unit where Shadow army is present."""
    result = {"effects": []}
    region = choices.get("region")
    if not region:
        # QA 2026-07-11 (a): membership, not units[0] - a mixed besieged
        # entry with the FP garrison first still HAS a Shadow besieger,
        # so _is_army_shadow (units[0]) hid the legal target and the
        # direct/empty-choices play recruited nothing.
        region = next((k for k, a in state.armies.items()
                       if any(u.nation in SHADOW_NATIONS for u in a.units)),
                      None)
    if region and not _shadow_units_at(state, sanitize(region)):
        result["message"] = f"No Shadow army in {region}"
        return result
    if region:
        unit_type = _default_unit_type(state, Nation.ISENGARD, choices)
        kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
        # Honest reporting: a blocked recruit is not an effect (mirrors
        # Olog-hai)
        _u9 = recruit_unit(state, region, Nation.ISENGARD, kind)
        if _u9 is not None:
            result["effects"].append(
                f"Recruited Isengard {unit_type} in {region}")
        else:
            result["message"] = (
                f"Half-orcs: recruitment blocked in {region}")
            return _selfplay_refund(result)
    return result


@register_effect("Olog-hai")
def _effect_olag_hai(state: "GameState", card, choices: dict) -> dict:
    """Recruit Sauron unit where Shadow army is present."""
    result = {"effects": []}
    region = choices.get("region")
    if not region:
        # Membership, not units[0]: a mixed besieged entry with the FP
        # garrison first still HAS a Shadow besieger (Half-orcs twin).
        region = next((k for k, a in state.armies.items()
                       if any(u.nation in SHADOW_NATIONS
                              for u in a.units)), None)
    if region and not _shadow_units_at(state, sanitize(region)):
        result["message"] = f"No Shadow army in {region}"
        return _selfplay_refund(result)
    # honest per-piece reporting + refund on no-op (honest-reporting wave)
    if region:
        unit_type = _default_unit_type(state, Nation.SAURON, choices)
        kind = UnitKind.ELITE if unit_type == "elite" else UnitKind.REGULAR
        _u9 = recruit_unit(state, region, Nation.SAURON, kind)
        if _u9 is not None:
            result["effects"].append(
                f"Recruited Sauron {unit_type} in {region}")
        else:
            result["message"] = f"Olog-hai: recruitment blocked in {region}"
            return _selfplay_refund(result)
    else:
        result["message"] = "Olog-hai: no Shadow army on the board"
        return _selfplay_refund(result)
    return result


@register_effect("Hill-trolls")
def _effect_hill_trolls(state: "GameState", card, choices: dict) -> dict:
    """Upgrade 2 Sauron regulars to elites anywhere."""
    result = {"effects": []}
    upgraded = 0
    # Get regions from multi-select (returns list)
    regions = choices.get("regions", [])

    for region in regions:
        # Find army with case-insensitive matching
        army = None
        for k, a in state.armies.items():
            if sanitize(k) == sanitize(region):
                army = a
                break

        if army:
            for i, unit in enumerate(army.units):
                if unit.nation == Nation.SAURON and unit.kind == UnitKind.REGULAR and upgraded < 2:
                    # QA 2026-07-10: an upgrade consumes an Elite from
                    # the reinforcement pool ("to the maximum extent
                    # possible" - no elite available, no upgrade)
                    from ai import _can_recruit
                    if not _can_recruit(state, Nation.SAURON, 'elite'):
                        result["effects"].append(
                            "No Sauron Elite in reinforcements - "
                            "upgrade skipped")
                        break
                    army.units[i] = Unit(Nation.SAURON, UnitKind.ELITE)
                    upgraded += 1
                    result["effects"].append(f"Upgraded Sauron Regular to Elite in {region}")
                    break  # Only one per region from selection

    if upgraded == 0:
        result["message"] = "No Sauron regulars found to upgrade"  # no-op: NOT an effect (iter 39)
        return _selfplay_refund(result)
    elif upgraded < 2:
        result["effects"].append(f"Only upgraded {upgraded} unit(s)")
    return result


@register_effect("A New Power is Rising")
def _effect_new_power_is_rising(state: "GameState", card, choices: dict) -> dict:
    """Mass Isengard recruitment: 2 in each Dunland, 2 in Orthanc."""
    if "recruit_counts" in choices:
        from shadow_recruit_allocations import apply_allocation
        return apply_allocation(state, card.card_name, choices)
    result = {"effects": []}
    # honest per-piece counts (honest-reporting wave 2026-07-21)
    _nd9 = sum(1 for _ in range(2)
               if recruit_unit(state, "North Dunland", Nation.ISENGARD,
                               UnitKind.REGULAR) is not None)
    _sd9 = sum(1 for _ in range(2)
               if recruit_unit(state, "South Dunland", Nation.ISENGARD,
                               UnitKind.REGULAR) is not None)

    # Get unit combination for Orthanc
    combo = _default_unit_combo(state, Nation.ISENGARD, choices)
    _or9 = []
    if combo == "2regular":
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.REGULAR))
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.REGULAR))
        orthanc_text = "2 Regulars"
    elif combo == "2elite":
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.ELITE))
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.ELITE))
        orthanc_text = "2 Elites"
    else:  # mixed
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.REGULAR))
        _or9.append(recruit_unit(state, "Orthanc", Nation.ISENGARD, UnitKind.ELITE))
        orthanc_text = "1 Regular + 1 Elite"
    _or9n = sum(1 for u in _or9 if u is not None)

    if _nd9:
        result["effects"].append(
            f"Recruited {_nd9} Isengard Regular(s) in North Dunland")
    if _sd9:
        result["effects"].append(
            f"Recruited {_sd9} Isengard Regular(s) in South Dunland")
    if _or9n:
        result["effects"].append(
            f"Recruited Isengard {orthanc_text} in Orthanc")
    if not result["effects"]:
        result["message"] = "A New Power is Rising: all recruitment blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Many Kings to the Service of Mordor")
def _effect_many_kings(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 SE regulars in each of 3 SE settlements."""
    if "recruit_counts" in choices:
        from shadow_recruit_allocations import apply_allocation
        return apply_allocation(state, card.card_name, choices)
    result = {"effects": []}
    settlements = choices.get("settlements")
    if not settlements:
        # max-extent default: the first 3 FREE S&E settlements
        settlements = _free_se_settlements(state)[:3]
    if choices.get("settlements"):
        _san = [sanitize(str(s)) for s in settlements[:3]]
        _legal = set(_free_se_settlements(state))
        if len(set(_san)) != len(_san) or any(s not in _legal for s in _san):
            result["message"] = ("Many Kings to the Service of Mordor: "
                                 "choose 3 different free S&E settlements "
                                 "- not played")
            return _selfplay_refund(result)
    for settlement in settlements[:3]:
        _u1 = recruit_unit(state, settlement,
                           Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR)
        _u2 = recruit_unit(state, settlement,
                           Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR)
        _n = sum(1 for u in (_u1, _u2) if u is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} S&E Regular(s) in {settlement}")
    return result


def _free_se_settlements(state):
    """Free Southron & Easterling settlements (for Many Kings)."""
    out = []
    for san, meta in sorted(board.regions.items()):
        if board.fid_to_name.get(meta.get('parent_id')) not in (
                'southrons', 'southron', 'southrons and easterlings'):
            continue
        if not (meta.get('is_town') or meta.get('is_city')
                or meta.get('is_stronghold')):
            continue
        if san in getattr(state, 'besieged_strongholds', set()):
            from combat import garrison_side_for as _gsf9
            if _gsf9(state, san) == Side.SHADOW:
                out.append(san)  # own garrison under siege (cap 5)
            elif _shadow_units_at(state, san):
                # captured (FP-held) S&E stronghold whose FP garrison
                # a Shadow army now besieges: FREE again for
                # recruitment into the besieging army (almanac Moria
                # rule + "recruit at a Stronghold when a Free Peoples
                # Army is inside a Stronghold besieged by a Shadow
                # Army"; same semantics as _rage_regions._ok9 /
                # _region_free_for_shadow).  recruit_unit already
                # places S&E into the besieging army at the open cap
                # of 10.
                out.append(san)
            continue
        if _region_free_for_shadow(state, san) and \
                san not in state.enemy_controlled.get(
                    Side.FREE_PEOPLES, set()):
            out.append(san)
    return out


@register_effect("The King is Revealed")
def _effect_king_is_revealed(state: "GameState", card, choices: dict) -> dict:
    """Recruit 5 Sauron regulars and 1 Nazgûl in Minas Morgul."""
    result = {"effects": []}
    _n = sum(1 for _ in range(5)
             if recruit_unit(state, "Minas Morgul", Nation.SAURON,
                             UnitKind.REGULAR) is not None)
    _nz = recruit_nazgul(state, "Minas Morgul")
    if _n or _nz:
        result["effects"].append(
            f"Recruited {_n} Sauron Regular(s)"
            f"{' and 1 Nazgul' if _nz else ''} in Minas Morgul")
    else:
        result["message"] = "Minas Morgul recruitment fully blocked"
    return result


@register_effect("Shadows on the Misty Mountains")
def _effect_shadows_on_misty_mountains(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 Sauron units and 1 Nazgûl in Mount Gram or Moria."""
    result = {"effects": []}
    region = choices.get("region", "Mount Gram")
    if sanitize(str(region)) not in ("mount gram", "moria"):
        result["message"] = ("Shadows on the Misty Mountains: %s is "
                             "not Mount Gram or Moria - not played"
                             % region)
        return _selfplay_refund(result)

    # Get unit combination
    combo = _default_unit_combo(state, Nation.SAURON, choices)
    _placed9 = []
    if combo == "2regular":
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.REGULAR))
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.REGULAR))
        unit_text = "2 Regulars"
    elif combo == "2elite":
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.ELITE))
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.ELITE))
        unit_text = "2 Elites"
    else:  # mixed
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.REGULAR))
        _placed9.append(recruit_unit(state, region, Nation.SAURON, UnitKind.ELITE))
        unit_text = "1 Regular + 1 Elite"
    _np9 = sum(1 for u in _placed9 if u is not None)

    # honest combined string when anything placed; fully blocked -> refund
    _nz = recruit_nazgul(state, region)
    if _np9 or _nz is not None:
        _naz9 = ' and 1 Nazgul' if _nz is not None else ' (Nazgul blocked)'
        _ut9 = unit_text if _np9 else 'no units'
        result["effects"].append(
            f"Recruited Sauron {_ut9}{_naz9} in {region}")
    else:
        result["message"] = ("Shadows on the Misty Mountains: "
                             "recruitment fully blocked")
        return _selfplay_refund(result)
    return result


@register_effect("Orcs Multiplying Again")
def _effect_orcs_multiplying_again(state: "GameState", card, choices: dict) -> dict:
    """Recruit 3 Sauron regulars in Dol Guldur and 3 in Mount Gundabad."""
    if "recruit_counts" in choices:
        from shadow_recruit_allocations import apply_allocation
        return apply_allocation(state, card.card_name, choices)
    result = {"effects": []}
    # honest per-piece counts (a blocked placement is not an effect)
    _dg9 = sum(1 for _ in range(3)
               if recruit_unit(state, "Dol Guldur", Nation.SAURON,
                               UnitKind.REGULAR) is not None)
    _mg9 = sum(1 for _ in range(3)
               if recruit_unit(state, "Mount Gundabad", Nation.SAURON,
                               UnitKind.REGULAR) is not None)
    if _dg9:
        result["effects"].append(
            f"Recruited {_dg9} Sauron Regular(s) in Dol Guldur")
    if _mg9:
        result["effects"].append(
            f"Recruited {_mg9} Sauron Regular(s) in Mount Gundabad")
    if not result["effects"]:
        result["message"] = "Orcs Multiplying Again: all blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Horde from the East")
def _effect_horde_from_east(state: "GameState", card, choices: dict) -> dict:
    """Recruit 5 SE regulars in free SE region adjacent to east map edge."""
    result = {"effects": []}
    region = choices.get("region")
    if not region:
        # Rules-honest default (card text: "a free region adjacent to
        # the eastern edge of the map") — no-op plays get filtered
        # almanac: East Rhun / South Rhun / Khand / Far Harad only
        for cand in ("east rhun", "south rhun", "khand", "far harad"):
            if _region_free_for_shadow(state, cand):
                region = cand
                break
    if region and sanitize(region) not in ("east rhun", "south rhun",
                                            "khand", "far harad"):
        result["message"] = f"{region} is not a legal Horde region"
        return result
    if region:
        _n = sum(1 for _ in range(5)
                 if recruit_unit(state, region,
                                 Nation.SOUTHRON_EASTERLINGS,
                                 UnitKind.REGULAR) is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} S&E Regular(s) in {region}")
        else:
            result["message"] = f"Recruitment blocked in {region}"
            return _selfplay_refund(result)
    else:
        result["message"] = "No free Horde region"
        return _selfplay_refund(result)
    return result


@register_effect("Monsters Roused")
def _effect_monsters_roused(state: "GameState", card, choices: dict) -> dict:
    """Recruit Sauron regulars in Angmar/Ettenmoors/Weather Hills, elite in Trollshaws."""
    if "recruit_counts" in choices:
        from shadow_recruit_allocations import apply_allocation
        return apply_allocation(state, card.card_name, choices)
    result = {"effects": []}
    # honest per-piece counts (a blocked placement is not an effect)
    _placed9 = [r for r in ("Angmar", "Ettenmoors", "Weather Hills")
                if recruit_unit(state, r, Nation.SAURON,
                                UnitKind.REGULAR) is not None]
    _el9 = recruit_unit(state, "Trollshaws", Nation.SAURON,
                        UnitKind.ELITE)
    if _placed9:
        result["effects"].append(
            "Recruited Sauron Regulars in " + ", ".join(_placed9))
    if _el9 is not None:
        result["effects"].append("Recruited Sauron Elite in Trollshaws")
    if not result["effects"]:
        result["message"] = "Monsters Roused: all blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Musterings of Long-planned War")
def _effect_musterings_of_long_planned_war(state: "GameState", card, choices: dict) -> dict:
    """Recruit 5 SE regulars in Gorgoroth, 5 Sauron regulars in Nurn."""
    result = {"effects": []}
    # honest per-piece counts (honest-reporting wave 2026-07-21)
    _g9 = sum(1 for _ in range(5)
              if recruit_unit(state, "Gorgoroth",
                              Nation.SOUTHRON_EASTERLINGS,
                              UnitKind.REGULAR) is not None)
    _n9 = sum(1 for _ in range(5)
              if recruit_unit(state, "Nurn", Nation.SAURON,
                              UnitKind.REGULAR) is not None)
    if _g9:
        result["effects"].append(
            f"Recruited {_g9} S&E Regular(s) in Gorgoroth")
    if _n9:
        result["effects"].append(
            f"Recruited {_n9} Sauron Regular(s) in Nurn")
    if not result["effects"]:
        result["message"] = "Musterings of Long-planned War: all blocked"
        return _selfplay_refund(result)
    return result


@register_effect("Pits of Mordor")
def _effect_pits_of_mordor(state: "GameState", card, choices: dict) -> dict:
    """Recruit 2 Sauron regulars in each of 3 Sauron strongholds."""
    if "recruit_counts" in choices:
        from shadow_recruit_allocations import apply_allocation
        return apply_allocation(state, card.card_name, choices)
    result = {"effects": []}
    strongholds = choices.get("strongholds")
    if not strongholds:
        # max-extent default: the first 3 FREE Sauron strongholds
        strongholds = _free_sauron_strongholds(state)[:3]
    if choices.get("strongholds"):
        _san = [sanitize(str(s)) for s in strongholds[:3]]
        _legal = set(_free_sauron_strongholds(state))
        if len(set(_san)) != len(_san) or any(s not in _legal for s in _san):
            result["message"] = ("Pits of Mordor: choose 3 different free "
                                 "Sauron strongholds - not played")
            return _selfplay_refund(result)
    for sh in strongholds[:3]:
        _u1 = recruit_unit(state, sh, Nation.SAURON, UnitKind.REGULAR)
        _u2 = recruit_unit(state, sh, Nation.SAURON, UnitKind.REGULAR)
        _n = sum(1 for u in (_u1, _u2) if u is not None)
        if _n:
            result["effects"].append(
                f"Recruited {_n} Sauron Regular(s) in {sh}")
        if _n < 2:
            # FLE-13b (g3#13): a committed stronghold that gets fewer
            # than 2 Regulars (reinforcement pool exhausted - physical
            # component limits, USER 2026-08-18) must SAY so, never
            # skip silently.
            result["effects"].append(
                f"Pits of Mordor: only {_n} of 2 Sauron Regular(s) "
                f"placed in {sh} (reinforcement pool exhausted)")
    return result


def _free_sauron_strongholds(state):
    """Free Sauron-nation strongholds (for Pits of Mordor)."""
    out = []
    for san, meta in sorted(board.regions.items()):
        if not meta.get('is_stronghold'):
            continue
        if board.fid_to_name.get(meta.get('parent_id')) != 'sauron':
            continue
        if san in getattr(state, 'besieged_strongholds', set()):
            from combat import garrison_side_for as _gsf9
            if _gsf9(state, san) == Side.SHADOW:
                out.append(san)  # own garrison under siege
            elif _shadow_units_at(state, san):
                # captured (FP-held) Sauron stronghold whose FP
                # garrison a Shadow army now besieges: FREE again for
                # recruitment into the besieging army (same semantics
                # as _free_se_settlements / _region_free_for_shadow).
                out.append(san)
            continue
        if _region_free_for_shadow(state, san) and \
                san not in state.enemy_controlled.get(
                    Side.FREE_PEOPLES, set()):
            out.append(san)
    return out


# ---------------------------------------------------------------------------
# Event Requirement Checking - Helper Functions
# ---------------------------------------------------------------------------

def _companion_in_fellowship(state: "GameState", name: str) -> bool:
    """Check if a specific companion is in the Fellowship."""
    return any(c.name == name for c in state.fellowship.companions)


def _find_companion_region(state: "GameState", companion_name: str) -> Optional[str]:
    """Find which region a separated companion is in."""
    for region_key, army in state.armies.items():
        for leader in army.leaders:
            if hasattr(leader, 'name') and leader.name == companion_name:
                return region_key
    return None


def _find_character_with_army(state: "GameState", char_names: List[str],
                              units_required: bool = True) -> Optional[Tuple[str, str]]:
    """Find if any named characters are with a Free Peoples Army. Returns
    (char_name, region).  units_required=False also matches leader-only
    entries (QA 2026-07-11 Dead Men CASE 6: a LONE separated Strider in
    a Rohan region is a legal play - 'in any Rohan region' needs no
    army)."""
    for region_key, army in state.armies.items():
        # A separated Companion may share a region with only Shadow units.
        # Those units never satisfy a printed Free Peoples Army requirement;
        # mixed siege entries qualify whenever an FP unit is also present.
        if units_required and not any(u.nation in FP_NATIONS for u in army.units):
            continue
        for leader in army.leaders:
            if hasattr(leader, 'name') and leader.name in char_names:
                return (leader.name, region_key)
    return None


def _get_region_nation(region_key: str) -> Optional[str]:
    """Get the nation a region belongs to."""
    rkey = sanitize(region_key)

    # board.name_to_meta maps nation names to nation data with 'regions' list
    for nation_name, nation_data in board.name_to_meta.items():
        if 'regions' in nation_data:
            for reg in nation_data['regions']:
                if sanitize(reg.get('name', '')) == rkey:
                    return nation_name.lower()
    return None


def _is_fp_nation_region(region_key: str) -> bool:
    """Check if region is in a Free Peoples nation."""
    nation = _get_region_nation(region_key)
    return nation in ("gondor", "rohan", "elves", "dwarves", "the north", "north")


def _army_contains_nazgul(army: Army) -> bool:
    """Check if army contains at least one Nazgul (the Witch-king IS a
    Nazgul for all rules - QA 2026-07-11)."""
    import game_logic as _gl
    return any(_gl.is_nazgul_figure(leader) for leader in army.leaders)


def _army_contains_witch_king(army: Army) -> bool:
    """Check if army contains the Witch-king."""
    from game_state import Nazgul
    for leader in army.leaders:
        if isinstance(leader, Nazgul) and getattr(leader, 'is_witch_king', False):
            return True
        # normalize to letters only (same as is_nazgul_figure) so 'Witch King' /
        # 'WitchKing' both match
        import re as _re
        nm = _re.sub(r'[^a-z]', '',
                     str(getattr(leader, 'name', '') or '').lower())
        if 'witchking' in nm:
            return True
    return False


def _grond_siege_regions(state: "GameState") -> List[str]:
    """Grond needs the Witch-king outside an FP-held native FP Stronghold."""
    from combat import garrison_side_for
    besieged = getattr(state, 'besieged_strongholds', set()) or set()
    out = []
    for key, army in state.armies.items():
        region = sanitize(key)
        if region not in besieged or garrison_side_for(state, region) != Side.FREE_PEOPLES:
            continue
        meta = _get_region_metadata(region)
        if not meta.get("is_stronghold") or meta.get("owner") != "free":
            continue
        if (any(unit.nation in SHADOW_NATIONS for unit in army.units)
                and _army_contains_witch_king(army)):
            out.append(region)
    return out


def _army_contains_companion(state: "GameState", army: Army) -> bool:
    """Check if army contains a companion."""
    for leader in army.leaders:
        if hasattr(leader, 'name'):
            # Check if this is a known companion
            companion_names = {"Gandalf the Grey", "Gandalf the White", "Strider",
                            "Aragorn", "Aragorn - Heir to Isildur", "Boromir",
                            "Legolas", "Gimli", "Meriadoc", "Peregrin"}
            if leader.name in companion_names or any(cn in leader.name for cn in companion_names):
                return True
    return False


def _is_army_shadow(army: Army) -> bool:
    """Check if army belongs to Shadow."""
    if not army.units:
        return False
    return army.units[0].nation in SHADOW_NATIONS


def _get_adjacent_regions(region_key: str) -> List[str]:
    """Get adjacent regions (module-level adjacency map; never
    constructs a new Board — this runs inside per-army loops)."""
    return sorted(board.adj.get(sanitize(region_key), ()))


def _get_region_metadata(region_key: str) -> dict:
    """Get metadata for a region (isStronghold, isCity, etc)."""
    rkey = sanitize(region_key)
    for nation_name, nation_data in board.name_to_meta.items():
        if 'regions' in nation_data:
            for reg in nation_data['regions']:
                if sanitize(reg.get('name', '')) == rkey:
                    return {
                        'nation': nation_name.lower(),
                        'is_stronghold': reg.get('isStronghold', False),
                        'is_city': reg.get('isCity', False),
                        'is_town': reg.get('isTown', False),
                        'is_fortification': reg.get('isFortification', False),
                        'is_settlement': reg.get('isCity', False) or reg.get('isTown', False) or reg.get('isStronghold', False),
                        'is_shadow': nation_data.get('isShadows', False),
                        'owner': 'shadow' if nation_data.get('isShadows', False) else 'free'
                    }
    return {}


def _stronghold_controlled_by(state: "GameState", region_key: str) -> Optional[str]:
    """Return 'free' or 'shadow' based on who controls the stronghold."""
    rkey = sanitize(region_key)
    meta = _get_region_metadata(rkey)

    # Captured settlements are tracked in enemy_controlled (keyed by the
    # CAPTOR side) — definitive, and covers captured-but-empty regions
    # (2023-12-06 T8: Return to Valinor with SH-captured Lorien)
    if rkey in state.enemy_controlled.get(Side.SHADOW, set()):
        return "shadow"
    if rkey in state.enemy_controlled.get(Side.FREE_PEOPLES, set()):
        return "free"

    # RB31: the besieged player still controls the Stronghold. Partial
    # besieger movement can put its remaining units first in the shared
    # Army entry; that list order cannot transfer control.
    if rkey in getattr(state, 'besieged_strongholds', set()):
        from combat import garrison_side_for
        return "shadow" if garrison_side_for(state, rkey) == Side.SHADOW else "free"

    # Check if there's an army there (sanitized scan — engine army keys
    # are display names like 'Lôrien', never bare sanitized keys)
    army = None
    for k, a in state.armies.items():
        if sanitize(k) == rkey:
            army = a
            break
    if army and army.units:
        if _is_army_fp(army):
            return "free"
        elif _is_army_shadow(army):
            return "shadow"

    # Default to original owner
    return meta.get("owner", "").lower()


# ---------------------------------------------------------------------------
# Event Requirement Checking - Main Function
# ---------------------------------------------------------------------------

def check_event_requirement(state: "GameState", card_id: str) -> Tuple[bool, str]:
    """Check if an event card's requirements are met."""
    from cards import get_card
    from game_logic import fellowship_region

    card = get_card(card_id)
    if not card:
        return False, "Card not found"

    req_text = card.event_req.lower() if card.event_req else ""

    if getattr(card, 'card_name', '') == "Faramir's Rangers":
        if not _faramir_targets(state) and not _faramir_fp_army_present(state):
            return False, "Requires a Shadow target or an FP Army in Osgiliath"
        return True, "OK"

    # 1.210(e): The Spirit of Mordor has an EMPTY event_req but is only
    # playable when a legal target exists - a Shadow army containing units
    # of 2+ Shadow Nations.  Gate BEFORE the empty-req early return.
    if getattr(card, 'card_name', '') == 'The Spirit of Mordor':
        if not _get_multi_nation_shadow_armies(state):
            return False, ("No Shadow army with units from 2+ Shadow "
                           "Nations")
        return True, "OK"

    # row 42: Stormcrow also has an EMPTY event_req but is only playable
    # when a not-At-War FP nation holds the Ring-bearers or a separated
    # Companion.  Gate BEFORE the empty-req early return.
    if getattr(card, 'card_name', '') == 'Stormcrow':
        if not _stormcrow_valid_nations(state):
            return False, ("No not-At-War FP nation holds the "
                           "Ring-bearers or a separated Companion")
        return True, "OK"

    # 1.211l (row 27): Through a Day and a Night moves an FP Army
    # CONTAINING A COMPANION - a BOARD companion (an army-leader figure).
    # Companions inside the Fellowship do not exist on the board, so a
    # Fellowship parked anywhere never satisfies this gate. Gate BEFORE
    # the empty-req early return (the handler guard stays as backstop).
    if getattr(card, 'card_name', '') == 'Through a Day and a Night':
        from game_state import Companion as _CompT
        ok = any(a.units and any(u.nation in FP_NATIONS for u in a.units)
                 and any(isinstance(l, _CompT) for l in a.leaders)
                 for a in state.armies.values())
        if not ok:
            return False, "No Free Peoples Army containing a Companion"
        return True, "OK"

    # No requirement
    if not req_text or req_text.strip() == "":
        return True, "OK"

    # Paths of the Woses (QA 2026-07-09): the DB req text ("FP army in
    # Minas Tirith") contradicts almanac A11 - playability = Rohan At
    # War + an FP army in a Rohan region + a legal destination (MT if
    # free, else Druadan Forest / Osgiliath / Lossarnach), checked via
    # the same option generator the play itself uses.
    if getattr(card, 'card_name', '') == 'Paths of the Woses':
        if not state.political.is_at_war(Nation.ROHAN):
            return False, "Rohan must be At War"
        if not _paths_of_woses_options(state):
            return False, ("No legal move: no FP army in a Rohan region "
                           "or no free destination")
        return True, "OK"

    # -------------------------------------------------------------------------
    # Companion in Fellowship requirements
    # -------------------------------------------------------------------------

    if "gimli or legolas are in the fellowship" in req_text:
        if not (_companion_in_fellowship(state, "Gimli") or
                _companion_in_fellowship(state, "Legolas")):
            return False, "Gimli or Legolas must be in Fellowship"

    if "boromir is in the fellowship" in req_text:
        if not _companion_in_fellowship(state, "Boromir"):
            return False, "Boromir must be in Fellowship"

    if "gandalf the grey is in the fellowship" in req_text:
        if not _companion_in_fellowship(state, "Gandalf the Grey"):
            return False, "Gandalf the Grey must be in Fellowship"

    if "at least one companion is in the fellowship" in req_text:
        if len(state.fellowship.companions) == 0:
            return False, "At least one Companion must be in Fellowship"

    # -------------------------------------------------------------------------
    # Character with Army requirements
    # -------------------------------------------------------------------------

    if "strider/aragorn is with a free peoples army in gondor or rohan" in req_text:
        result = _find_character_with_army(state, ["Strider", "Aragorn", "Aragorn - Heir to Isildur"])
        if not result:
            return False, "Strider/Aragorn must be with a Free Peoples Army"
        char_name, region = result
        nation = _get_region_nation(region)
        if nation not in ("gondor", "rohan"):
            return False, "Strider/Aragorn must be in Gondor or Rohan"

    if "strider/aragorn is in any rohan region" in req_text:
        # USER 2026-07-16 (item 2): Strider/Aragorn must be SEPARATED on the
        # map (a lone figure counts) in a Rohan region.  A Strider/Aragorn
        # still INSIDE the Fellowship does NOT satisfy this - the Fellowship
        # being in Rohan is not enough (W7_ep00001 illegal DMoD).  The old
        # code accepted the in-Fellowship case, so Dead Men could fire with
        # Strider never separated.
        result = _find_character_with_army(
            state, ["Strider", "Aragorn", "Aragorn - Heir to Isildur"],
            units_required=False)
        if not result:
            return False, ("Strider/Aragorn must be SEPARATED in a Rohan "
                           "region (not in the Fellowship)")
        _, region = result
        if _get_region_nation(region) != "rohan":
            return False, ("Strider/Aragorn must be SEPARATED in a Rohan "
                           "region (not in the Fellowship)")

    if "strider/aragorn is with a free peoples army" in req_text and "gondor or rohan" not in req_text:
        result = _find_character_with_army(state, ["Strider", "Aragorn", "Aragorn - Heir to Isildur"])
        if not result:
            return False, "Strider/Aragorn must be with a Free Peoples Army"

    if "boromir is in a gondor region" in req_text:
        # USER RULING 2026-07-21e: a Companion INSIDE the Fellowship
        # is never "in" the Fellowship's region — a Fellowship parked
        # in Gondor does NOT satisfy House of the Stewards. Boromir
        # must be SEPARATED on the board (an army-leader figure) in a
        # Gondor region. (The card's effect + option-gen already act
        # only on a board Boromir; this aligns the play gate.)
        boromir_region = _find_companion_region(state, "Boromir")
        if not boromir_region or _get_region_nation(boromir_region) != "gondor":
            return False, "Boromir must be in a Gondor region"

    if "aragorn is with a free peoples army in a region outside a free peoples nation" in req_text:
        result = _find_character_with_army(state, ["Aragorn", "Aragorn - Heir to Isildur"])
        if not result:
            return False, "Aragorn must be with a Free Peoples Army"
        _, region = result
        if _is_fp_nation_region(region):
            return False, "Aragorn must be outside Free Peoples Nation borders"

    if "aragorn is in play" in req_text:
        # Promotion markers survive elimination to forbid a second promotion.
        # The printed "in play" requirement needs the living map figure;
        # a lone Companion qualifies without any Army units.
        has_aragorn = _find_character_with_army(
            state, ["Aragorn", "Aragorn - Heir to Isildur"],
            units_required=False) is not None
        if not has_aragorn:
            return False, "Aragorn must be in play"

    # -------------------------------------------------------------------------
    # Gandalf the White / Saruman / Witch-king requirements
    # -------------------------------------------------------------------------

    if "gandalf the white is in play" in req_text:
        has_gandalf_white = _find_character_with_army(
            state, ["Gandalf the White"], units_required=False) is not None
        if not has_gandalf_white:
            return False, "Gandalf the White must be in play"

    if "saruman is in play" in req_text:
        # consult the real figure state (in_play + map presence) - the single
        # source of truth
        _sar9 = getattr(state, 'saruman', None)
        _sar_in = bool(getattr(_sar9, 'in_play', False))
        if not _sar_in:
            _sar_in = any(
                type(l).__name__ == 'Minion'
                and 'saruman' in str(getattr(l, 'name', '')).lower()
                for a in state.armies.values() for l in a.leaders)
        if not _sar_in:
            return False, "Saruman must be in play"

    if "the witch-king is in play" in req_text or "witch-king is in play" in req_text:
        # WK piece names vary ('The Witch-king', 'Witch King', 'WitchKing') -
        # normalized containment match
        _wk_found = any(
            'witchking' in re.sub(r'[^a-z]', '',
                                  getattr(l, 'name', '').lower())
            for a in state.armies.values() for l in a.leaders)
        if not _wk_found:
            return False, "The Witch-king must be in play"

    if "a companion is in fangorn" in req_text:
        # Ents Awake trilogy (QA 2026-07-11 cluster D, almanac):
        # 'Gandalf the White must be in play AND a Companion must be
        # in Fangorn' - the earlier OR reading was wrong (the gate
        # leaked whenever GtW was in play).
        gtw_ok = _find_character_with_army(
            state, ["Gandalf the White"], units_required=False) is not None
        # USER RULING 2026-07-21e: a Companion INSIDE the Fellowship is
        # never "in" the Fellowship's region — a Fellowship parked in
        # Fangorn does NOT satisfy "a Companion is in Fangorn". Only a
        # SEPARATED companion on the board counts (GtW separated in
        # Fangorn is such a figure; _army_contains_companion matches).
        companions_in_fangorn = False
        for _k, _a in state.armies.items():
            if sanitize(_k) == "fangorn" \
                    and _army_contains_companion(state, _a):
                companions_in_fangorn = True
                break

        if not (gtw_ok and companions_in_fangorn):
            return False, "Gandalf the White in play and a Companion in Fangorn required"

    # -------------------------------------------------------------------------
    # Army adjacency requirements
    # -------------------------------------------------------------------------

    if "a free peoples army containing a companion is adjacent to" in req_text and "shadow army containing a nazgûl" in req_text:
        # Membership tests, not units[0]: mixed besieged entries hold
        # BOTH sides in one shared army entry (_is_shadow_army_with_nazgul).
        found = False
        for region_key, army in state.armies.items():
            if any(u.nation not in SHADOW_NATIONS for u in army.units) \
                    and _army_contains_companion(state, army):
                # Check adjacent regions for Shadow army with Nazgul
                for adj in _get_adjacent_regions(region_key):
                    adj_army = state.armies.get(_find_army_key(state, adj))
                    if _is_shadow_army_with_nazgul(adj_army):
                        found = True
                        break
                # Also check same region (incl. the SAME shared entry)
                if not found and _is_shadow_army_with_nazgul(army):
                    found = True
                for other_key, other_army in state.armies.items():
                    if found or other_key == region_key:
                        continue
                    if sanitize(other_key) == sanitize(region_key):
                        if _is_shadow_army_with_nazgul(other_army):
                            found = True
                            break
            if found:
                break
        if not found:
            return False, "Need FP Army with Companion adjacent to Shadow Army with Nazgul"

    if "a shadow army containing a nazgûl is adjacent to" in req_text and "free peoples army" in req_text:
        # NOTE: sanitized army-key lookup (armies are keyed by display
        # names — .get(sanitized) always missed) and per-unit FP
        # membership (a besieged region holds BOTH sides in ONE shared
        # entry, so the same-region case must scan the entry's own units)
        found = False
        for region_key, army in state.armies.items():
            # USER 2026-07-16 (item 5): the Nazgûl must be part of a Shadow
            # ARMY - a lone Nazgûl figure (an entry with a Nazgûl leader but
            # NO Shadow units) is not "a Shadow Army containing a Nazgûl".
            if not _is_shadow_army_with_nazgul(army):
                continue
            k_san = sanitize(region_key)
            cand = {k_san}
            cand.update(sanitize(x) for x in _get_adjacent_regions(region_key))
            for k2, a2 in state.armies.items():
                if sanitize(k2) not in cand:
                    continue
                if any(u.nation not in SHADOW_NATIONS for u in a2.units):
                    found = True
                    break
            if found:
                break
        if not found:
            return False, "Need Shadow Army with Nazgûl adjacent to Free Peoples Army"

    # -------------------------------------------------------------------------
    # Siege requirements
    # -------------------------------------------------------------------------

    # Siege entries are MIXED (garrison first, besiegers co-located): use
    # membership tests and sanitized keys, never units[0] or exact keys.
    if "witch-king is in play and is with a shadow army besieging a free peoples stronghold" in req_text:
        if not _grond_siege_regions(state):
            return False, "Witch-king must be with Shadow Army besieging FP Stronghold"

    if "minas tirith is besieged by a shadow army" in req_text:
        # almanac A3 (QA 2026-07-10): may not be played when a SHADOW
        # army is besieged inside Minas Tirith - the garrison must be FP
        from combat import garrison_side_for as _gsf9
        _mt9 = 'minas tirith'
        _mt_ok = (_mt9 in getattr(state, 'besieged_strongholds', set())
                  and _gsf9(state, _mt9) == Side.FREE_PEOPLES)
        if not _mt_ok:
            return False, ("Minas Tirith must be besieged by a Shadow "
                           "Army (FP garrison inside)")

    if "an army containing an isengard unit is besieging a stronghold" in req_text:
        # ONE gate, both modes (USER directive): a live siege with
        # Isengard units on the BESIEGER side
        from combat import garrison_side_for as _gsf9
        found = False
        for _bs9 in getattr(state, 'besieged_strongholds', set()):
            if _gsf9(state, _bs9) != Side.FREE_PEOPLES:
                continue
            _e9 = state.armies.get(_find_army_key(state, _bs9))
            if _e9 and any(u.nation == Nation.ISENGARD
                           for u in _e9.units):
                found = True
                break
        if not found:
            return False, "Isengard Army must be besieging a Stronghold"

    if "there is a free peoples army in minas tirith" in req_text or "there is a free peoples army in minis tirith" in req_text:
        mt_army = next((a for k, a in state.armies.items()
                        if sanitize(k) == 'minas tirith'), None)
        # a besieged MT holds BOTH sides in one entry — FP presence is a
        # membership test, not an army-side test
        if not mt_army or not any(u.nation in FP_NATIONS
                                  for u in mt_army.units):
            return False, "Free Peoples Army must be in Minas Tirith"

    # -------------------------------------------------------------------------
    # Stronghold control requirements
    # -------------------------------------------------------------------------

    if "you control at least one elven stronghold" in req_text:
        # Return to Valinor. USER RULING 1.208o (SUPERSEDES the earlier
        # besieged-qualifies amendment): gate is Shadow CONTROL (capture)
        # of an Elven Stronghold; merely BESIEGED does NOT qualify.
        elven_strongholds = ["rivendell", "lorien", "grey havens",
                             "woodland realm"]
        controls_one = any(
            _stronghold_controlled_by(state, sh) == "shadow"
            for sh in elven_strongholds)
        if not controls_one:
            return False, "Must control at least one Elven Stronghold"

    # -------------------------------------------------------------------------
    # Fellowship location requirements
    # -------------------------------------------------------------------------

    if "fellowship is not in a region containing a free peoples settlement" in req_text:
        fsp_region = fellowship_region(state)
        meta = _get_region_metadata(fsp_region)
        if meta.get("owner") == "free" and meta.get("is_settlement"):
            return False, "Fellowship is in FP settlement"

    if "fellowship is revealed" in req_text:
        if state.fellowship.hidden:
            return False, "Fellowship must be revealed"

    if "fellowship is on step 1 or higher" in req_text:
        if state.fellowship.progress < 1:
            return False, "Fellowship needs progress"
        # almanac R3 (QA 2026-07-11): on the Mordor Track the
        # Fellowship is OFF the Fellowship Track - Cruel Weather may
        # not be played (progress keeps its pre-Mordor value, so the
        # numeric gate alone leaked)
        if state.fellowship.mordor_entered:
            return False, "Fellowship is on the Mordor Track"

    # -------------------------------------------------------------------------
    # Political track requirements
    # -------------------------------------------------------------------------

    if "gondor is active" in req_text:
        if not state.political.active.get(Nation.GONDOR, False):
            return False, "Gondor must be Active"

    if "the elves are \"at war\"" in req_text or "elves are \"at war\"" in req_text:
        if not state.political.is_at_war(Nation.ELVES):
            return False, "Elves must be At War"

    if "rohan is \"at war\"" in req_text:
        if not state.political.is_at_war(Nation.ROHAN):
            return False, "Rohan must be At War"

    if "isengard is \"at war\"" in req_text:
        if not state.political.is_at_war(Nation.ISENGARD):
            return False, "Isengard must be At War"

    if "sauron is \"at war\"" in req_text:
        if not state.political.is_at_war(Nation.SAURON):
            return False, "Sauron must be At War"

    if "the southrons & easterlings are \"at war\"" in req_text or "southrons & easterlings are \"at war\"" in req_text:
        if not state.political.is_at_war(Nation.SOUTHRON_EASTERLINGS):
            return False, "Southrons & Easterlings must be At War"

    # Adjudication #7 (2026-07-21): Horde from the East is UNPLAYABLE when
    # none of its four legal recruit regions (East/South Rhun, Khand, Far
    # Harad) is free for the Shadow - there is no free region to place the
    # 5 S&E Regulars.  Mirrors the effect handler's own region validator.
    if getattr(card, 'card_name', '') == 'Horde from the East':
        if not any(_region_free_for_shadow(state, r)
                   for r in ("east rhun", "south rhun", "khand",
                             "far harad")):
            return False, "All Horde regions are FP-controlled/occupied"

    if "all shadow nations are \"at war\"" in req_text:
        for nation in SHADOW_NATIONS:
            if not state.political.is_at_war(nation):
                return False, "All Shadow nations must be At War"

    # -------------------------------------------------------------------------
    # Ring-bearers/Companion in nation requirements
    # -------------------------------------------------------------------------

    if "either the ring-bearers or a companion is inside the borders of a free peoples nation not \"at war\"" in req_text:
        fsp_region = fellowship_region(state)
        nation_str = _get_region_nation(fsp_region)

        # Map region nation to Nation enum
        nation_map = {
            "gondor": Nation.GONDOR,
            "rohan": Nation.ROHAN,
            "elves": Nation.ELVES,
            "dwarves": Nation.DWARVES,
            "the north": Nation.THE_NORTH,
            "north": Nation.THE_NORTH,
        }

        found = False
        # Check Fellowship location
        if nation_str in nation_map:
            nation = nation_map[nation_str]
            if not state.political.is_at_war(nation):
                found = True

        # Check separated companions
        if not found:
            for region_key, army in state.armies.items():
                if _army_contains_companion(state, army):
                    comp_nation_str = _get_region_nation(region_key)
                    if comp_nation_str in nation_map:
                        nation = nation_map[comp_nation_str]
                        if not state.political.is_at_war(nation):
                            found = True
                            break

        if not found:
            return False, "Ring-bearers or Companion must be in FP Nation not At War"

    return True, "OK"


# ---------------------------------------------------------------------------
# Card Choice Requirements
# ---------------------------------------------------------------------------

def _get_free_rohan_settlements(state: "GameState", exclude_besieged: bool = False) -> List[str]:
    """Get free Rohan settlements.
    If exclude_besieged=True, filters out Helm's Deep when under siege (Eomer rule)."""
    settlements = ["Edoras", "Helm's Deep", "Westemnet", "Folde"]
    if exclude_besieged:
        bs = getattr(state, 'besieged_strongholds', set())
        settlements = [s for s in settlements if sanitize(s) not in bs]
    return settlements


def _get_multi_nation_shadow_armies(state: "GameState") -> List[str]:
    """Get Shadow armies with units from multiple nations."""
    regions = []
    for region_key, army in state.armies.items():
        if not army.units:
            continue
        nations = set(u.nation for u in army.units)
        shadow_nations = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}
        if len(nations & shadow_nations) >= 2:
            regions.append(region_key)
    return regions


# ============================================================================
# The Shadow Lengthens - learned-path chained single-move flow
# (amendment 1.117 TSiM pattern; QA fix "shadow-lengthens-chains" 2026-07-11).
#
# Each of the (up to two) card moves is an INDEPENDENT chained pending on the
# learned path.  The pending SELECTS moves (progress = a list of [from, to]
# region-name pairs carried in every option dict - clone-safe strings); the
# relocation is DEFERRED and executed with staged move-OUTs at chain end, so
# the cargo/qualification of every move is each source's army as it stood at
# the START of the Action (almanac A8) and units delivered by one move are
# never re-picked-up by another - regardless of the order the moves were
# picked.  The TEACHER path (logged choices / GUI direct calls) never carries
# '_from_pending' and keeps the original staged/sequential handler body
# untouched (byte-identical).
# ============================================================================
def _tsl_dests0(state):
    """Regions holding a Shadow army that is not a besieged GARRISON -
    the destination qualification for The Shadow Lengthens / Shadows
    Gather, established at the start of the Action (almanac A8).  A
    Shadow BESIEGER stack qualifies (1.211h / row 43: the region around
    a besieged stronghold is free for the besieger, GAME19 T5); only a
    walled-in Shadow garrison is unjoinable.  Because relocation is
    deferred to chain end, the live board is still the start-of-action
    board throughout selection, so this recomputes the same set at
    every stage."""
    return {sanitize(k) for k, a in state.armies.items()
            if any(u.nation in SHADOW_NATIONS for u in a.units)
            and not _besieged_garrison_source(state, sanitize(k),
                                              Side.SHADOW)}


def _tsim_dest_ok(state, to):
    """The Shadow is Moving 1-region destination test (ONE predicate for
    the spec, the chain surface and the handler): a besieged region is
    enterable only as the BESIEGER (Shadow units outside the walls, FP
    garrison inside - a normal join, GAME19); a besieged Shadow garrison
    is unjoinable; any other region must hold no enemy units."""
    from combat import side_of_nation as _son9d, garrison_side_for
    to = sanitize(to)
    a_to = state.armies.get(_find_army_key(state, to))
    has_enemy = bool(a_to and any(_son9d(u.nation) != Side.SHADOW
                                  for u in a_to.units))
    has_own = bool(a_to and any(_son9d(u.nation) == Side.SHADOW
                                for u in a_to.units))
    if to in getattr(state, 'besieged_strongholds', set()):
        return has_own and garrison_side_for(state, to) != Side.SHADOW
    return not has_enemy


def _tsim_next_options(state, moved_set, n_done):
    """The Shadow is Moving's remaining single-army move options
    (amendment 1.133: extracted from the handler so the teacher chain
    emitter enumerates EXACTLY the learned path's options).  1.211h:
    only a besieged Shadow GARRISON may not move; a besieger stack may
    (lifting the siege) and may be joined (_tsim_dest_ok)."""
    _opts9 = []
    for k, a in state.armies.items():
        frm9 = sanitize(k)
        if frm9 in moved_set \
                or _besieged_garrison_source(state, frm9, Side.SHADOW):
            continue
        if not any(u.nation in SHADOW_NATIONS for u in a.units):
            continue
        for adj in state.board.adjacent(frm9):
            to9 = sanitize(adj)
            if not _tsim_dest_ok(state, to9):
                continue
            _opts9.append({'from_region': frm9,
                           'to_region': to9,
                           '_tsim_moved': sorted(moved_set),
                           '_tsim_n': n_done})
    return _opts9


def _tsl_next_options(state, exclude_src):
    """Single-move options for the next chained pending: every Shadow
    army NOT already selected this play and NOT besieged, paired with
    each qualifying destination it can reach through free regions.
    Sources already selected are excluded via exclude_src (the progress
    data)."""
    return [option for option in _sg_join_options(state, 2)
            if option['from_region'] not in exclude_src]


def _tsl_execute_staged(state, moves, result):
    """Execute the accumulated joins with all move-OUTs staged BEFORE any
    merge.  The cargo of each move is its source's army at the start of
    the Action; a region that received a delivery therefore never has
    those delivered figures picked up by a later move (almanac A8).
    Destination qualification and cargo are fixed at the action start.
    Land all selected figures after every move-out, then offer the owner
    any required overstack removals. Capacity never changes the selection."""
    from card_movement_witness import row as _witness_row
    _gross_witness = []
    state._card_army_move_witness = ('SHADOW_Strategy_The_Shadow_Lengthens', ())
    if not moves:
        result["message"] = "No movements selected"
        return
    from combat import side_of_nation as _son9, _check_settlement_capture
    import game_logic as _gl9
    dests0 = _tsl_dests0(state)      # qualification fixed at start
    if len(moves) > 2 or len({sanitize(m[0]) for m in moves}) != len(moves):
        raise ValueError('Lengthens requires up to two distinct original armies')
    plans = []
    for mv in moves:
        frm, to = sanitize(mv[0]), sanitize(mv[1])
        # USER 1.208n: optional per-move initial split (2-tuple = whole
        # army, byte-identical legacy)
        _nr = mv[2] if len(mv) > 2 else None
        _ne = mv[3] if len(mv) > 3 else None
        if to not in dests0:
            result["effects"].append(
                "%s has no joinable Shadow army (none, or a besieged "
                "garrison) - move refused" % to)
            continue
        if to not in _army_reachable_free(state, frm, 2, Side.SHADOW):
            result["effects"].append(
                "%s not reachable within 2 free regions from %s" % (to, frm))
            continue
        mids = _free_path_mids(state, frm, to, 2)
        key = _find_army_key(state, frm)
        src = state.armies.get(key)
        if src is None:
            result["effects"].append("No movable units in %s" % frm)
            continue
        mv_u = [u for u in src.units if _son9(u.nation) == Side.SHADOW]
        mv_l = [l for l in src.leaders
                if not _gl9._is_saruman(l)
                and _gl9._leader_side(l) == Side.SHADOW]
        from shadow_join_rules import select as select_join
        exact = len(mv) > 4 and mv[4] is not None
        if not exact:
            mv_u, mids = select_join(state, frm, to, 2, mv_u)
        if _nr is not None or _ne is not None:
            _tr = _nr if _nr is not None else 10 ** 6
            _te = _ne if _ne is not None else 10 ** 6
            _sel = []
            for u in mv_u:
                if u.kind == UnitKind.ELITE and _te > 0:
                    _sel.append(u)
                    _te -= 1
                elif u.kind != UnitKind.ELITE and _tr > 0:
                    _sel.append(u)
                    _tr -= 1
            mv_u = _sel
        if len(mv) > 4 and mv[4] is not None:
            from army_card_cargo import selected
            mv_u, mv_l = selected(state, frm, mv[4])
        if exact:
            eligible, mids = select_join(state, frm, to, 2, mv_u, exact=True)
            if len(eligible) != len(mv_u):
                raise ValueError('Selected cargo cannot cross political borders')
        if not mv_u:
            result["effects"].append("No movable units in %s" % frm)
            continue
        plans.append((frm, to, key, mv_u, mv_l, mids))
    # stage all move-OUTs first (start-of-action cargo)
    for frm, to, key, mv_u, mv_l, mids in plans:
        src = state.armies.get(key)
        if src is None:
            continue
        _mu = set(map(id, mv_u))
        _ml = set(map(id, mv_l))
        src.units = [u for u in src.units if id(u) not in _mu]
        src.leaders = [l for l in src.leaders if id(l) not in _ml]
        if not src.units and not src.leaders:
            state.armies.pop(key, None)
    # then merge each group into its destination + walk-through capture
    for frm, to, key, mv_u, mv_l, mids in plans:
        dst_key = _find_army_key(state, to)
        dst = state.armies.get(dst_key)
        if dst is None:
            dst = Army(region=state.board.region(to))
            state.armies[to] = dst
        land = mv_u
        dst.units.extend(land)
        dst.leaders.extend(mv_l)
        _gross_witness.append(_witness_row(frm, to, land, mv_l))
        own = sum(1 for u in dst.units if _son9(u.nation) == Side.SHADOW)
        if own > _gl9.MAX_ARMY_UNITS:
            overstack = getattr(state, '_overstack_regions', None)
            if overstack is None:
                overstack = set()
                state._overstack_regions = overstack
            overstack.add(to)
        for m in mids:
            _check_settlement_capture(state, m, Side.SHADOW)
            # rulebook 370-371: a walked-through region is ENTERED
            _gl9._check_political_activation_for_entry(state, m,
                                                       Side.SHADOW)
        result["effects"].append("Army moved from %s to join %s" % (frm, to))
    _gl9.remove_orphaned_leaders(state)
    state._card_army_move_witness = ('SHADOW_Strategy_The_Shadow_Lengthens', tuple(_gross_witness))


def _tsl_learned_chain(state, card, choices):
    """Chained single-move flow for The Shadow Lengthens on the learned
    path (amendment 1.117, the TSiM pattern).  Each pick appends one
    join to the '_tsl_moves' progress list and re-offers a direct
    pending with the remaining single-move options (already-selected
    sources excluded) plus a terminator.  Relocation is deferred to the
    end of the chain (terminator picked, both armies selected, or no
    further legal move), then executed with staged move-OUTs."""
    result = {"effects": []}
    _cid = getattr(card, 'card_id', 'The Shadow Lengthens')
    if not choices.get('_tsl_done'):
        from army_card_cargo import stage
        choices = stage(state, card, choices, choices['from_region'])
        if choices is None:
            return {'effects': ['Choose the figures moving with this army']}
    moves = [list(m) for m in (choices.get('_tsl_moves') or [])]
    if not choices.get('_tsl_done'):
        f = choices.get('from_region')
        t = choices.get('to_region')
        if f and t:
            # USER 1.208n: carry the learned initial split (clone-safe
            # ints/None) alongside the [from, to] pair.
            _nr = choices.get('n_regular')
            _ne = choices.get('n_elite')
            moves.append([sanitize(f), sanitize(t),
                          (int(_nr) if _nr is not None else None),
                          (int(_ne) if _ne is not None else None),
                          choices.get('cargo')])
    picked_src = {m[0] for m in moves}
    # offer another INDEPENDENT move while under the two-army limit
    if not choices.get('_tsl_done') and len(moves) < 2:
        nxt = _tsl_next_options(state, picked_src)
        if nxt:
            for o in nxt:
                o['_tsl_moves'] = [list(m) for m in moves]
            nxt.append({'_tsl_done': True,
                        '_tsl_moves': [list(m) for m in moves]})
            state.pending_card_option = {
                'card_id': _cid, 'card_name': 'The Shadow Lengthens',
                'side': 'sh', 'decider': 'sh', 'direct': True,
                'options': nxt}
            result["effects"].append(
                "Move %d of up to 2 (pending choice)" % (len(moves) + 1))
            return result
    # chain complete: execute the accumulated moves with staged move-OUTs
    _tsl_execute_staged(state, moves, result)
    if choices.get('_tsl_done') and not moves:
        return {'effects': ['The Shadow Lengthens: movement ends']}
    if not result["effects"]:
        return _selfplay_refund(result)
    return result


# ---------------------------------------------------------------------------
# hunt-card-draws (QA 2026-07-11): shared machinery for the Mithril Coat and
# Sting redraw interpose on the SHADOW hunt-DRAWING cards (Orc Patrol / Foul
# Thing from the Deep / Isildur's Bane) and the Foul Thing learned victim pick.
# ---------------------------------------------------------------------------

def _mithril_interposes(state) -> bool:
    """True iff the FP 'Mithril Coat and Sting' redraw decision must
    interpose BEFORE a Shadow card's freshly drawn hunt tile is applied.

    Gated exactly like the existing sites (game_logic._resolve_hunt_tile /
    _resolve_mordor_hunt_tile): the LEARNED path only
    (combat.FINAL_HITS_HOOK is None) - the teacher replays the humans'
    own Mithril use from the log.  Also requires no other pending
    decision to be open and Mithril to actually be on the table."""
    import combat as _cmb9
    if _cmb9.FINAL_HITS_HOOK is not None:
        return False
    if getattr(state, 'pending_card_option', None) is not None:
        return False
    _pc9 = getattr(state, 'persistent_cards', None)
    if _pc9 is None:
        return False
    try:
        from game_state import PersistentCardType as _PCT9
        return _pc9.get_by_type(_PCT9.MITHRIL_COAT) is not None
    except Exception:
        return False


def _mithril_stash_card(state, tile, apply_kind, choices) -> None:
    """Stash a Shadow card's drawn tile and open the FP Mithril redraw
    pending.  The card's apply-continuation runs from
    _effect_mithril_coat, keyed by apply_kind ('orc_patrol' /
    'foul_thing' / 'isildur'), which re-enters the card handler with the
    kept-or-redrawn tile forced via choices['_forced_tile']."""
    from game_state import PersistentCardType as _PCT9
    _pc9 = getattr(state, 'persistent_cards', None)
    _mc9 = _pc9.get_by_type(_PCT9.MITHRIL_COAT) if _pc9 is not None else None
    # keep only the play choices the continuation needs; drop routing keys
    _stash_choices = {k: v for k, v in (choices or {}).items()
                      if k not in ('_forced_tile',)}
    state._mithril_stash = {'tile': tile, 'apply_kind': apply_kind,
                            'choices': _stash_choices}
    state.pending_card_option = {
        'card_id': getattr(_mc9, 'card_id', '__mithril__'),
        'card_name': 'Mithril Coat and Sting',
        'side': 'fp', 'decider': 'fp', 'direct': True,
        'options': [{'mithril_redraw': True},
                    {'mithril_redraw': False}]}


def _wizardstaff_interposes(state) -> bool:
    """True iff the FP 'Wizard's Staff' PRE-DRAW prevent decision must
    interpose BEFORE a Shadow hunt-DRAWING card (Isildur's Bane / Foul
    Thing from the Deep) draws its Hunt tile.

    Amendment 1.217 (USER 2026-07-23): "wizard's staff can be used against
    isildur's bane."  The staff (card_list.csv:9) PREVENTS the tile DRAW
    itself - a pre-draw nullify - distinct from the post-draw Axe/Horn
    damage REDUCTION (which Isildur's Bane never offers, Foul Thing does).

    Gated exactly like game_logic._resolve_hunt_tile's staff site: the
    LEARNED path only (combat.FINAL_HITS_HOOK is None) - the teacher
    replays the humans' own staff use from the log.  Requires no other
    pending decision and the Wizard's Staff actually on the table."""
    import combat as _cmb9
    if _cmb9.FINAL_HITS_HOOK is not None:
        return False
    if getattr(state, 'pending_card_option', None) is not None:
        return False
    _pc9 = getattr(state, 'persistent_cards', None)
    if _pc9 is None:
        return False
    try:
        from game_state import PersistentCardType as _PCT9
        return _pc9.get_by_type(_PCT9.WIZARD_STAFF) is not None
    except Exception:
        return False


def _wizardstaff_stash_card(state, apply_kind, choices) -> None:
    """Stash a Shadow hunt-drawing card's play context and open the FP
    Wizard's Staff PRE-DRAW prevent pending (amendment 1.217).  The
    continuation runs from _effect_wizards_staff, keyed by apply_kind
    ('foul_thing' / 'isildur'): PREVENT discards the staff and NO tile is
    drawn (no damage); DECLINE re-enters the card handler with
    choices['_skip_wizard']=True so it proceeds to the draw.  Mirrors
    _mithril_stash_card (the sibling POST-draw redraw interpose)."""
    from game_state import PersistentCardType as _PCT9
    _pc9 = getattr(state, 'persistent_cards', None)
    _ws9 = _pc9.get_by_type(_PCT9.WIZARD_STAFF) if _pc9 is not None else None
    # keep only the play choices the continuation needs; drop routing keys
    _stash_choices = {k: v for k, v in (choices or {}).items()
                      if k not in ('_skip_wizard',)}
    state._wizardstaff_stash = {'apply_kind': apply_kind,
                                'choices': _stash_choices}
    state.pending_card_option = {
        'card_id': getattr(_ws9, 'card_id', '__wizardstaff__'),
        'card_name': "Wizard's Staff",
        'side': 'fp', 'decider': 'fp', 'direct': True,
        'options': [{'wizardstaff_prevent': True},
                    {'wizardstaff_prevent': False}]}


def _foul_apply_casualty(state, victim_name, damage, result) -> dict:
    """Apply 'Foul Thing from the Deep' damage to a NAMED companion
    casualty (extracted so both the direct path and the learned
    victim-pick re-entry share one implementation).  Hobbits SEPARATE
    (Take Them Alive) off the Mordor track; excess over the victim's
    level becomes Ring-bearer Corruption."""
    import game_logic as _gl
    victim = next((c for c in state.fellowship.companions
                   if c.name == victim_name), None)
    if victim is None:
        if damage > 0:
            state.fellowship.corruption = min(
                12, state.fellowship.corruption + damage)
            result["effects"].append(f"+{damage} Corruption")
        return result
    absorbed = min(damage, int(getattr(victim, 'level', 1) or 1))
    remainder = damage - absorbed
    # GL-4 class (rulebook p.42 new-Guide note, 1.83, standing ruling
    # 2026-08-16): the forced pick is the casualty-TAKEN row (WWSAT
    # fires; a hobbit separates outside Mordor) and the REMAINDER
    # continues the SAME damage event through game_logic's hunt chain
    # with the NEW guide's ability live - exactly hunt_sacrifice_random.
    # The teacher (FINAL_HITS_HOOK) keeps its certified inline body.
    import combat as _cmb_ft
    _learned = _cmb_ft.FINAL_HITS_HOOK is None
    if _gl.hobbit_hunt_placement_applies(state, victim.name):
        if _learned:
            # separates, alive; the learned hobbit_place decision opens
            # AFTER the chain (queue drained by _finish_hunt_decision)
            _gl.remove_hobbit_for_placement(state, victim, absorbed)
            q = getattr(state, '_hobbit_place_queue', None)
            if q is None:
                q = state._hobbit_place_queue = []
            q.append(victim)
        else:
            # hobbit uses his ability: SEPARATES (placement decision),
            # not eliminated.  A random pick is still a CASUALTY for
            # 'Worn with Sorrow and Toil' (almanac, QA 2026-07-11 E12b).
            state.fellowship.companions.remove(victim)
            _gl._on_companion_casualty(state, victim.name)
            _gl.change_guide(state, allow_pending=True)
            if getattr(state, 'pending_hunt', None) is None:
                state.pending_hunt = {
                    'phase': 'hobbit_place',
                    'companion': victim.name,
                    'level': int(getattr(victim, 'level', 1) or 1),
                    'companion_obj': victim, 'damage': 0}
        result["effects"].append(
            f"{victim.name} separates (hobbit ability), absorbs "
            f"{absorbed}")
    else:
        _gl._apply_fellowship_damage(state, absorbed,
                                     kill_name=victim.name)
        result["effects"].append(f"{victim.name} eliminated, absorbs "
                                 f"{absorbed}")
    if _learned:
        # the card's damage is fully carried by `damage` here; the
        # reduction interpose's staging scalar must not be re-settled
        # by the chain's close (the executor zeroes it right after)
        state._pending_hunt_damage = 0
        _gl.continue_hunt_chain(state, remainder, casualty_taken=True)
        if remainder > 0:
            result["effects"].append(
                f"{remainder} remaining damage continues with the new "
                f"guide (hunt chain)")
    elif remainder > 0:
        state.fellowship.corruption = min(
            12, state.fellowship.corruption + remainder)
        result["effects"].append(
            f"+{remainder} Corruption to the Ring-bearer (excess)")
    return result


# ---------------------------------------------------------------------------
# Regular-board group companion movement chain (rulebook p.20, "Moving
# Companions"): one FP Character die may move ANY number of companion
# GROUPS from different regions, each group with its own allowance (max
# level; Gandalf the White 4 alone / with one hobbit).  The engine moves
# one region's group/subset per companion_move action; on the LEARNED
# path the executor then chains a '__companion_moves__' pending offering
# further group moves, mirroring the '__nazgul_moves__' chain (amendment
# 1.128).  Companions are NOT fungible, so the already-moved NAMES are
# excluded (per-companion once-per-die) and the exclusion set rides every
# option.  Teacher path is untouched (amendment 1.122: logged per-region
# walks + die-less continuations replay as before).
# ---------------------------------------------------------------------------

def _companion_chain_options(state, moved=None) -> list:
    """Further companion GROUP-move options for the current FP Character
    die.  Mirrors _nazgul_chain_options but companions are NOT fungible:
    `moved` (a set/list of NAMES already moved this die) is excluded, so
    the remaining companions - whether left behind at the first move's
    source region or standing elsewhere - are still offered as their own
    groups.  Reuses _companion_group_move_options (the group-lattice
    single source of truth) and re-keys each option to the top-level
    from_region/to_region/companions slots so the action encoder sees
    the region + character multi-hot directly.  The moved-names set is
    embedded in every option (_comp_moved) so the next chain stage knows
    what has already flown."""
    moved = set(str(n) for n in (moved or []))
    opts = []
    for o in _companion_group_move_options(state, exclude=moved):
        grp = o.get('move_group') or {}
        if not (grp.get('from') and grp.get('to') and grp.get('companions')):
            continue
        opts.append({'from_region': sanitize(str(grp['from'])),
                     'to_region': sanitize(str(grp['to'])),
                     'companions': list(grp['companions']),
                     '_comp_moved': sorted(moved)})
    return opts


def _effect_companion_chain(state: "GameState", card, choices: dict) -> dict:
    """One more companion GROUP move on the same FP Character die, then
    re-offer (pseudo-handler '__companion_moves__', rulebook p.20).
    Mirrors _effect_nazgul_chain; the moved-names exclusion grows
    monotonically so the chain always terminates (finite companions)."""
    result = {"effects": []}
    if choices.get('_comp_done'):
        result["effects"].append("Companion movement ends")
        return result
    moved = set(str(n) for n in (choices.get('_comp_moved') or []))
    grp = {'from': choices.get('from_region'),
           'to': choices.get('to_region'),
           'companions': list(choices.get('companions') or [])}
    if grp['from'] and grp['to'] and grp['companions']:
        if _move_companion_group(state, grp, result):
            moved.update(str(n) for n in grp['companions'])
    opts = _companion_chain_options(state, moved)
    if opts:
        opts.append({'_comp_done': True})
        state.pending_card_option = {
            'card_id': '__companion_moves__',
            'card_name': '__companion_moves__',
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'options': opts}
    return result


CARD_EFFECT_HANDLERS["__companion_moves__"] = _effect_companion_chain


