# combat.py - Combat resolution engine for War of the Ring
"""Full combat system per WotR rulebook with cards, rounds, and leadership."""

from __future__ import annotations
import random
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState

from game_state import (
    Side, Nation, UnitKind, Unit, Army, Companion,
    CombatState, CombatCard, CombatResult
)
from board import sanitize
import board


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHADOW_NATIONS = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}


def roll_d6_batch(n: int, ctx: str = '', side=None) -> List[int]:
    """RANDOMNESS FUNNEL (session 33): every combat d6 batch goes through
    here.  Default body = the exact inline code it replaced (same
    `random.randint` calls, same order — seeded runs and the corpus
    canary are unaffected).  The LIVE bridge monkey-patches this so real
    game advances consume the Java client's own rolls (roll1..5
    handlers); MCTS clones keep this default.

    `side` (G1, USER live 2026-08-18: "Each side rolls its own combat
    dice.  The AI rolls its own; the human rolls their own; the
    attacker rolls first.") is WHOSE dice these are.  The engine already
    asked per side, attacker first — only the DIRECTIVE was side-blind,
    so the AI ended up rolling the human's dice for him.  Optional and
    ignored by the default body, so every offline caller is unchanged."""
    return [random.randint(1, 6) for _ in range(n)]


def side_of_nation(nation: Nation) -> Side:
    """Determine which side a nation belongs to."""
    return Side.SHADOW if nation in SHADOW_NATIONS else Side.FREE_PEOPLES


def side_of_army(army: Army) -> Side:
    """Determine the side of an army based on its units."""
    if not army.units:
        return Side.FREE_PEOPLES  # Default
    return side_of_nation(army.units[0].nation)


def garrison_side_for(state: "GameState", region_key: str) -> Side:
    """Side of the garrison inside a (besieged) stronghold region.

    The controller of the settlement: whoever captured it, else the
    native side of the region."""
    from game_logic import is_fp_region
    san = sanitize(region_key)
    if san in state.enemy_controlled.get(Side.SHADOW, set()):
        return Side.SHADOW
    if san in state.enemy_controlled.get(Side.FREE_PEOPLES, set()):
        return Side.FREE_PEOPLES
    return Side.FREE_PEOPLES if is_fp_region(san) else Side.SHADOW


def _combat_leader_side(ld) -> Side:
    from game_state import Nazgul, Companion
    if isinstance(ld, Nazgul):
        return Side.SHADOW
    if isinstance(ld, Companion):
        return Side.FREE_PEOPLES
    if type(ld).__name__ == 'Minion':
        return Side.SHADOW
    if hasattr(ld, 'nation'):
        return side_of_nation(ld.nation)
    return Side.FREE_PEOPLES


# ---------------------------------------------------------------------------
# Combat Strength & Leadership Calculation
# ---------------------------------------------------------------------------

# Captains of the West: each adds +1 combat die (total still capped at 5)
CAPTAINS_OF_THE_WEST = {'Gandalf the Grey', 'Strider', 'Legolas',
                        'Gimli', 'Boromir', 'Aragorn'}


def calc_combat_strength(army: Army, extra_companions: List[Companion] = None,
                         cancelled_captain=None) -> int:
    """
    Combat dice: min(5, units + Captains of the West present).
    Each unit (Regular or Elite) contributes 1 combat die; Gandalf the
    Grey / Strider / Legolas / Gimli / Boromir / Aragorn add +1 each.
    `extra_companions` = companions in the battle but not in army.leaders
    (the fellowship defending its region).
    `cancelled_captain`: USER 1.208i: a Companion whose Captain-of-the-West
    COMBAT DIE Words of Power cancelled for this round - dropped BEFORE the
    min(5) cap (a 6+-raw army stays 5; 4R+Boromir 5->4).
    """
    strength = len(army.units)
    # 1.233: leaders serve only their own side — a stranded enemy
    # leader in this army grants no captain die (side derived from
    # units; leader-only entries keep legacy unfiltered behavior)
    _own = side_of_army(army) if army.units else None
    for ld in army.leaders:
        if _own is not None and _combat_leader_side(ld) != _own:
            continue
        if isinstance(ld, Companion) and ld.name in CAPTAINS_OF_THE_WEST:
            if cancelled_captain is not None and ld is cancelled_captain:
                continue
            strength += 1
    for comp in (extra_companions or []):
        if getattr(comp, 'name', '') in CAPTAINS_OF_THE_WEST:
            if cancelled_captain is not None and comp is cancelled_captain:
                continue
            strength += 1
    return min(5, strength)


def calc_leadership(army: Army, companions: List[Companion] = None,
                    cancel_nazgul: bool = False) -> int:
    """
    Calculate leadership (for re-rolls, capped at 5 by the caller).

    Counts ONCE from army.leaders by type. (The old version counted
    special characters via len(leaders) AND again via the extracted
    params — systematic double-counting.)
    - plain Leader: 1            - Nazgul: 1 (Witch-king: 2)
    - Companion: its leadership rating (crowned Aragorn: 2)
    - Saruman: 1, Mouth of Sauron: 2

    `companions` = EXTRA companions not in army.leaders (fellowship
    defending its region). `cancel_nazgul`: Gandalf the White nullifies
    ALL Nazgul leadership (incl. the Witch-king) in his battle.
    """
    from game_state import Nazgul as _Nazgul
    leadership = 0
    # 1.233: leaders serve only their own side — a stranded enemy
    # leader contributes no Leadership (side from units; leader-only
    # entries unfiltered so elimination bookkeeping stays untouched)
    _own = side_of_army(army) if army.units else None
    for ld in army.leaders:
        if _own is not None and _combat_leader_side(ld) != _own:
            continue
        if isinstance(ld, Companion):
            leadership += getattr(ld, 'leadership', 1)
        elif isinstance(ld, _Nazgul):
            leadership += 0 if cancel_nazgul else 1
        elif type(ld).__name__ == 'Minion':
            nm = getattr(ld, 'name', '').lower()
            if 'witch' in nm:
                leadership += 0 if cancel_nazgul else 2
            elif 'mouth' in nm:
                leadership += 2
            else:
                leadership += 1  # Saruman
        else:
            leadership += 1
    for comp in (companions or []):
        leadership += getattr(comp, 'leadership', 1)
    return leadership


def count_elite_units(army: Army) -> int:
    """Count elite units in army."""
    return sum(1 for u in army.units if u.kind == UnitKind.ELITE)


# ---------------------------------------------------------------------------
# Combat Setup
# ---------------------------------------------------------------------------

def setup_combat(state: "GameState", attacker_region: str, defender_region: str,
                 attacker_card: Optional[CombatCard] = None,
                 defender_card: Optional[CombatCard] = None,
                 is_siege: bool = False,
                 attacker_side: Optional[Side] = None) -> CombatState:
    """
    Initialize combat state for a battle.
    """
    atk_key = sanitize(attacker_region)
    def_key = sanitize(defender_region)

    # Try both sanitized and original keys - armies may be stored either way
    atk_army = state.armies.get(attacker_region) or state.armies.get(atk_key)
    def_army = state.armies.get(defender_region) or state.armies.get(def_key)

    # Also try case-insensitive lookup
    if not atk_army:
        for key, army in state.armies.items():
            if sanitize(key) == atk_key:
                atk_army = army
                attacker_region = key
                break

    if not def_army:
        for key, army in state.armies.items():
            if sanitize(key) == def_key:
                def_army = army
                defender_region = key
                break

    if not atk_army:
        raise RuntimeError(f"No army in attacker region: {attacker_region}")
    if not def_army:
        raise RuntimeError(f"No army in defender region: {defender_region}")

    # Co-located siege assault (from == to): a besieged region holds BOTH
    # sides' units in ONE army entry (garrison inside the stronghold,
    # besiegers outside). Split it into two virtual armies for the battle;
    # resolve_combat merges the survivors back into the shared entry.
    _coloc_parent = None
    if atk_army is def_army:
        garrison = garrison_side_for(state, def_key)
        besieger = Side.SHADOW if garrison == Side.FREE_PEOPLES \
            else Side.FREE_PEOPLES
        # Direction (2026-07-10 sortie fix): the default attacker is
        # the besieger (siege assault); a SORTIE - the garrison
        # attacking its besiegers, rulebook p.32, a field battle -
        # inverts it.  attacker_side arrives from the acting side via
        # battle choices; None keeps the historical assault default.
        _atk_s = attacker_side if attacker_side is not None else besieger
        parent = atk_army
        v_gar = Army(region=parent.region)
        v_bes = Army(region=parent.region)
        for u in parent.units:
            (v_gar if side_of_nation(u.nation) == garrison
             else v_bes).units.append(u)
        for l in parent.leaders:
            (v_gar if _combat_leader_side(l) == garrison
             else v_bes).leaders.append(l)
        # Rulebook p.31 caps a besieged garrison at FIVE Army units;
        # USER 1.181 overrules the printed removed-from-the-game
        # destination — the excess returns to REINFORCEMENTS (an
        # untracked removal IS the pool return), never casualties.
        # (game0 T9: Cirdan's Ships raised the DA garrison to 7; the
        # human removed 2 Gondor Regulars.)  Regulars removed first.
        # The cap binds the GARRISON portion whichever combat role it
        # takes (assault defender or sortie attacker).
        if (len(v_gar.units) > 5
                and def_key in getattr(state, 'besieged_strongholds', set())):
            # The 5-unit cap is a BESIEGED-STRONGHOLD rule (rulebook p.31).
            # It must NOT fire for a co-located FIELD battle where the
            # region is not (yet) besieged - e.g. Corsairs of Umbar lands
            # a Shadow army onto an FP army in an open Gondor coastal
            # region (almanac A10).  Every pre-existing co-located battle
            # is a besieged stronghold, so this gate is behavior-neutral
            # for them.
            excess = len(v_gar.units) - 5
            removable = ([u for u in v_gar.units
                          if u.kind == UnitKind.REGULAR]
                         + [u for u in v_gar.units
                            if u.kind == UnitKind.ELITE])
            # WHICH box the excess goes to is the USER 1.181 house
            # ruling (2026-07-20, CL-147, game_logic siege re-clamp):
            # REINFORCEMENTS, never casualties — an untracked removal
            # IS a pool return (reinforcements = max - deployed -
            # fp_casualties).
            _cap_removed9 = list(removable[:excess])
            _cap_removed_side9 = garrison
            for u in _cap_removed9:
                v_gar.units.remove(u)
        v_atk = v_gar if _atk_s == garrison else v_bes
        v_def = v_bes if _atk_s == garrison else v_gar
        if not v_atk.units or not v_def.units:
            raise RuntimeError(
                f"Co-located battle at {def_key}: missing "
                f"{'attacker' if not v_atk.units else 'defender'} units")
        atk_army, def_army = v_atk, v_def
        _coloc_parent = parent

    # Away-attack from / into a co-located siege region (from != to):
    # the shared entry mixes both sides' units. Only the right portion
    # fights — the bystanders (e.g. the garrison inside the stronghold)
    # stay home; resolve_combat writes survivors back. (v9 showcase:
    # besiegers at Erebor attacking Iron Hills crashed side_of_army.)
    _away_atk_parent = _away_def_parent = None
    _away_atk_rest = _away_def_rest = None
    if _coloc_parent is None:
        def _mixed(a):
            return len({side_of_nation(u.nation) for u in a.units}) > 1

        def _split(parent, my_side):
            v = Army(region=parent.region)
            ru, rl = [], []
            for u in parent.units:
                (v.units if side_of_nation(u.nation) == my_side
                 else ru).append(u)
            for l in parent.leaders:
                (v.leaders if _combat_leader_side(l) == my_side
                 else rl).append(l)
            return v, ru, rl

        if _mixed(atk_army):
            # a garrison cannot attack out through the besiegers — the
            # attacking portion is the besieger side at the source
            _bes = (Side.SHADOW
                    if garrison_side_for(state, atk_key)
                    == Side.FREE_PEOPLES else Side.FREE_PEOPLES)
            _v, _ru, _rl = _split(atk_army, _bes)
            if _v.units:
                _away_atk_parent, _away_atk_rest = atk_army, (_ru, _rl)
                atk_army = _v
        if _mixed(def_army):
            # the target portion is whoever is hostile to the attacker
            _atk_s = side_of_army(atk_army)
            _tgt = (Side.SHADOW if _atk_s == Side.FREE_PEOPLES
                    else Side.FREE_PEOPLES)
            _v, _ru, _rl = _split(def_army, _tgt)
            if _v.units:
                _away_def_parent, _away_def_rest = def_army, (_ru, _rl)
                def_army = _v

    atk_side = side_of_army(atk_army)
    def_side = side_of_army(def_army)

    if atk_side == def_side:
        raise RuntimeError("Cannot attack friendly army")

    # Check for defensive positions
    # Use board.regions for region metadata (cities, forts, strongholds)
    reg_meta = board.regions.get(def_key, {})
    stronghold_defense = reg_meta.get("is_stronghold", False)
    is_city = reg_meta.get("is_city", False)
    fortification = reg_meta.get("is_fort", False)
    # Per rulebook: Cities and Fortifications give first-round 6+ penalty to attacker.
    # Strongholds do NOT have this penalty in field battles — both sides hit on 5+.
    # Strongholds allow siege option (attacker always 6+), but field battle = normal.
    city_defense = (is_city or fortification) and not is_siege

    # Get special characters.  1.233: battle-role rosters come from
    # OWN-side leaders only — a stranded enemy companion/nazgul/minion
    # contributes NOTHING (no shine candidacy, no forfeit pool, no
    # card-gate satisfaction); the army entries themselves are never
    # mutated, so elimination/capture bookkeeping is untouched.
    from game_state import Nazgul as _Nz233
    _atk_own_ldrs = [l for l in atk_army.leaders
                     if _combat_leader_side(l) == atk_side]
    _def_own_ldrs = [l for l in def_army.leaders
                     if _combat_leader_side(l) == def_side]
    atk_companions = [l for l in _atk_own_ldrs
                      if isinstance(l, Companion)]
    def_companions = [l for l in _def_own_ldrs
                      if isinstance(l, Companion)]
    atk_nazgul = sum(1 for l in _atk_own_ldrs if isinstance(l, _Nz233))
    def_nazgul = sum(1 for l in _def_own_ldrs if isinstance(l, _Nz233))
    atk_minions = [getattr(l, 'name', '') for l in _atk_own_ldrs
                   if type(l).__name__ == 'Minion']
    def_minions = [getattr(l, 'name', '') for l in _def_own_ldrs
                   if type(l).__name__ == 'Minion']

    # Nazgul Leadership available for FORFEIT (Dread & Despair, They are
    # Terrible, Cruel as Death) and the Foul Stench re-roll-cancel check.
    # USER 2026-07-18: "witch king counts as a nazgul so he can forfeit
    # his leadership for cruel as death, they are terrible, dread and
    # despair, etc."  The Witch-king is modelled as a Shadow Minion (2
    # leadership), not a Nazgul instance; add his leadership to the
    # forfeitable pool alongside the plain Nazgul (each worth 1) so the
    # pending-effect enumeration (ai._maybe_combat_effect_choice_phase /
    # apply_dread_and_despair / apply_they_are_terrible) can offer it.
    def _wk_leadership(ldrs_):
        # 1.233: own-side leaders only (stranded WK forfeits nothing)
        return sum(getattr(l, 'leadership', 2)
                   for l in ldrs_
                   if type(l).__name__ == 'Minion'
                   and 'witch' in getattr(l, 'name', '').lower())
    atk_nazgul_ldr = atk_nazgul + _wk_leadership(_atk_own_ldrs)
    def_nazgul_ldr = def_nazgul + _wk_leadership(_def_own_ldrs)

    # Fellowship CO-LOCATION fact only (USER 1.209, 2026-07-21): a
    # Fellowship co-located with the battling army contributes NOTHING to
    # combat -- no companion combat dice (no Captain-of-the-West die), no
    # Leadership re-rolls, no Gandalf-the-White shine candidacy, and its
    # members do NOT satisfy any combat-card play gate (requires_companion,
    # requires_hobbit) nor appear in companions_in_battle.  We keep ONLY the
    # boolean "is the Fellowship in the defender's region", which still
    # matters for (a) FP combat-card play eligibility bookkeeping and (b)
    # "Denethor's Folly": FP cannot play combat cards in Minas Tirith.
    # (The Fellowship-in-region gate for It is a Gift / One for the
    # Dark Lord is enforced independently via
    # card.requires_fellowship_region.)
    from persistent_effects import can_fp_play_combat_card
    fp_card_ok, _folly = can_fp_play_combat_card(state, def_key)
    if not fp_card_ok:
        if atk_side == Side.FREE_PEOPLES:
            attacker_card = None
        else:
            defender_card = None

    # Gandalf 'shining' (USER 2026-07-16 item 11): Gandalf the White may
    # forfeit HIS Leadership to cancel ALL Nazgul leadership (incl. the
    # Witch-king) in his battle.  This is a DECISION - it no longer happens
    # automatically.  The pending-combat machine records the FP choice in
    # pending_combat['gtw_shine']; a direct resolve / test with no machine
    # (no 'gtw_shine' key) keeps the legacy "shines" default.
    # NB (USER 1.209 + 2026-07-21e): Gandalf the White can NEVER be in the
    # Fellowship (he arrives already separated), so scanning only the two
    # map armies loses no real shine candidate.
    _gtw_comp = next(
        (c for c in atk_companions + def_companions
         if getattr(c, 'can_cancel_nazgul_leadership', False)
         or getattr(c, 'name', '') == 'Gandalf the White'), None)
    gtw_in_battle = _gtw_comp is not None
    _pcst = getattr(state, 'pending_combat', None)
    _shine_choice = (_pcst.get('gtw_shine')
                     if isinstance(_pcst, dict) and 'gtw_shine' in _pcst
                     else None)
    gtw_shines = gtw_in_battle and (_shine_choice is not False)
    atk_cancel = gtw_shines and atk_side == Side.SHADOW
    def_cancel = gtw_shines and def_side == Side.SHADOW

    # Saruman's ability: while Saruman is in play, Isengard Elite units
    # also count as Leaders (leadership 1 each). Verified vs game0:
    # T5 Fords of Isen (Isengard 2R+5E rerolled 3) and T7 Helm's Deep
    # (leaderless Isengard 5R+4E rerolled 2).
    def _isengard_elite_leadership(army_):
        return sum(1 for u in army_.units
                   if u.nation == Nation.ISENGARD
                   and u.kind == UnitKind.ELITE)
    _saruman_in_play = any(
        type(l).__name__ == 'Minion'
        and 'saruman' in getattr(l, 'name', '').lower()
        for a in state.armies.values() for l in a.leaders)
    saruman_atk_bonus = _isengard_elite_leadership(atk_army) \
        if (_saruman_in_play and atk_side == Side.SHADOW) else 0
    saruman_def_bonus = _isengard_elite_leadership(def_army) \
        if (_saruman_in_play and def_side == Side.SHADOW) else 0

    combat = CombatState(
        attacker_region=atk_key,
        defender_region=def_key,
        attacker_army=atk_army,
        defender_army=def_army,
        attacker_side=atk_side,
        defender_side=def_side,
        is_siege=is_siege,
        stronghold_defense=stronghold_defense,
        city_defense=city_defense,
        attacker_card=attacker_card,
        defender_card=defender_card,
        attacker_combat_strength=calc_combat_strength(atk_army),
        defender_combat_strength=calc_combat_strength(def_army),
        attacker_leadership=calc_leadership(atk_army, cancel_nazgul=atk_cancel)
        + saruman_atk_bonus,
        defender_leadership=calc_leadership(def_army,
                                            cancel_nazgul=def_cancel)
        + saruman_def_bonus,
        companions_in_battle=def_companions + atk_companions,
        nazgul_in_battle=atk_nazgul_ldr + def_nazgul_ldr,
        minions_in_battle=atk_minions + def_minions,
    )
    # When Gandalf shines he FORFEITS his own Leadership (the cost of the
    # cancel).  Remove his contribution from the FP side and mark him as
    # already spent so Mighty Attack / Daring Defiance cannot forfeit it a
    # second time (USER 2026-07-16 item 11).  Only on an EXPLICIT shine
    # decision - the legacy default (no pending machine / teacher replay)
    # keeps cancelling Nazgul leadership without the forfeit cost, so the
    # certified battle math is unchanged.
    combat._shined_companion = None
    if _shine_choice is True and _gtw_comp is not None:
        _g_ldr = getattr(_gtw_comp, 'leadership', 1)
        if _gtw_comp in atk_companions:
            combat.attacker_leadership = max(
                0, combat.attacker_leadership - _g_ldr)
        else:
            combat.defender_leadership = max(
                0, combat.defender_leadership - _g_ldr)
        combat._shined_companion = _gtw_comp

    # Words-of-Power round markers (USER 1.208i/j). Default inert; set by
    # apply_words_of_power (Phase 1b) / apply_reroll_cancellation (Phase 5).
    combat._wop_cancelled = None
    combat._wop_cancelled_is_attacker = False
    combat._foul_stench_cancel_atk = False
    combat._foul_stench_cancel_def = False
    # Shine bookkeeping so WoP can RESTORE the Nazgul Leadership the
    # shine removed (1.208i(2)).
    combat._shine_companion = _gtw_comp if gtw_shines else None
    combat._shine_cancelled_leadership = (
        (atk_nazgul_ldr if atk_side == Side.SHADOW else def_nazgul_ldr)
        if gtw_shines else 0)

    combat.coloc_parent = _coloc_parent
    # 5-unit siege-cap removals (informational: excess is a pool
    # return per USER 1.181 — never booked as casualties, B2)
    combat.cap_removed = locals().get('_cap_removed9') or []
    combat.cap_removed_side = locals().get('_cap_removed_side9')
    combat.away_atk_parent = _away_atk_parent
    combat.away_atk_rest = _away_atk_rest
    combat.away_def_parent = _away_def_parent
    combat.away_def_rest = _away_def_rest

    # RULING 1.228c (USER 2026-08-03): keep a handle to the live
    # GameState so combat-card Minion kills (Fateful Strike / Blade of
    # Westernesse) can do full character-sheet bookkeeping mid-round
    # (in_play flip + Saruman elimination discards) — see
    # _card_kill_minion_bookkeeping.  Follows the established
    # post-construction attr pattern above; direct CombatState
    # constructions fall back to _ACTIVE_STATE (game_logic.battle).
    combat._live_state = state

    return combat


# ---------------------------------------------------------------------------
# Dice Rolling
# ---------------------------------------------------------------------------

def combat_roll(dice_count: int, threshold: int, modifier: int = 0,
                cap: Optional[int] = None,
                side=None) -> Tuple[List[int], int]:
    """
    Roll combat dice and count hits.

    Args:
        dice_count: Number of dice to roll (before cap)
        threshold: Hit threshold (typically 5 for normal, 6 for stronghold attack)
        modifier: Add to each die result
        cap: Maximum dice allowed (from card effects)

    Returns:
        (list of roll results, number of hits)
    """
    if cap is not None:
        dice_count = min(dice_count, cap)
    dice_count = max(1, dice_count)  # Always roll at least 1

    rolls = roll_d6_batch(dice_count, 'combat roll', side=side)
    modified = [r + modifier for r in rolls]

    # Per rulebook p.30: natural 6 always hits, natural 1 always misses,
    # regardless of any modifiers
    hits = 0
    for i, r in enumerate(rolls):
        if r == 1:
            continue  # Natural 1 always misses
        if r == 6 or modified[i] >= threshold:
            hits += 1

    return rolls, hits


def leader_reroll(rolls: List[int], threshold: int, leadership: int,
                  modifier: int = 0, confused: bool = False,
                  miss_modifier: int = None,
                  exclude_misses: int = 0,
                  side=None) -> Tuple[List[int], int]:
    """
    Perform leader re-roll on misses.

    - Re-roll up to min(5, Leadership) misses
    - Same hit threshold and modifier as normal combat roll
    - Natural 1 always miss, natural 6 always hit
    - exclude_misses: misses already converted to hits by Mighty
      Attack / Anduril (Phase 5b) — they are no longer misses, so they
      leave the re-roll pool (a die must not score twice)

    Returns:
        (list of reroll results, number of additional hits)
    """
    if leadership <= 0:
        return [], 0

    # Find misses (dice that didn't hit) — judged with the COMBAT
    # roll's modifier, NOT the re-roll modifier: a re-roll-only bonus
    # (They are Terrible) must not retroactively turn an original miss
    # into a hit and shrink the re-roll pool (USER 2026-07-09: Lorien
    # siege, [2,5,3] at 6+ = 3 misses -> 3 re-roll dice, engine gave 2).
    # Under Confusion, natural 1s already penalized the roller and should
    # NOT be re-rolled per the card text ("cannot be rolled again").
    _mm = modifier if miss_modifier is None else miss_modifier
    misses = []
    for i, r in enumerate(rolls):
        if confused and r == 1:
            continue  # Confused 1s are excluded from re-roll pool
        modified = r + _mm
        if r != 6 and modified < threshold:
            misses.append(i)

    # Misses converted by Mighty Attack / Anduril are hits now
    if exclude_misses > 0:
        misses = misses[:max(0, len(misses) - exclude_misses)]

    # Re-roll up to Leadership misses (leadership capped at 5)
    reroll_count = min(5, leadership, len(misses))
    if reroll_count == 0:
        return [], 0

    rerolls = roll_d6_batch(reroll_count, 'leader reroll', side=side)
    # Re-rolls use the same hit threshold + modifier; natural 1 always miss, natural 6 always hit
    hits = 0
    for r in rerolls:
        if r == 1:
            continue
        if r == 6 or (r + modifier) >= threshold:
            hits += 1

    return rerolls, hits


# ---------------------------------------------------------------------------
# Casualty Application
# ---------------------------------------------------------------------------

def apply_casualties(army: Army, hits: int,
                     policy: str = 'preserve_elites') -> List[Unit]:
    """
    Remove casualties from army.

    In real WotR the owning player chooses per hit: remove a Regular OR
    downgrade an Elite to a Regular. The two orderings below span the
    practical strategies, and the choice is exposed to the AI as a
    learned combat decision (casualty_preserve_units /
    casualty_preserve_elites):

    - 'preserve_elites' (default, original behavior): Regulars are
      removed first; Elites then absorb hits (downgrade for 1 hit,
      full removal for 2). Keeps Elites on the board (siege pressing,
      card requirements) at the cost of unit count.
    - 'preserve_units': Elites downgrade FIRST — each downgrade absorbs
      a hit without losing a figure — then Regulars are removed.
      Maximizes remaining unit count (combat dice).

    Returns list of removed units.
    """
    # per-nation count of Regulars minted-by-downgrade. Within a
    # battle, SHARE the battle-scoped counter so downgrades accumulate
    # across rounds (temp-army additions are invisible to
    # _count_deployed); standalone/card calls get a fresh per-call dict.
    _dgc = _BATTLE_DG if _BATTLE_DG is not None else {}
    # Standalone (non-battle) scope: freeze the reinforcement baseline
    # NOW, before the loop kills anything (task 164). In a battle
    # _DG_AVAIL0 already holds the battle-start baseline and wins; here it
    # is None, so a card/unit-op call that removes regulars then downgrades
    # an elite still checks against the pre-call pool, not a kill-deflated
    # live count. None when no state is published (direct unit use).
    _av0 = None if _DG_AVAIL0 is not None else _capture_reg_baseline(_ACTIVE_STATE)
    casualties = []
    remaining_hits = hits

    if policy == 'spend_elites':
        # Spend Elites FIRST, as full 2-hit removals when possible
        # (downgrade + kill the resulting Regular), keeping Regulars on
        # the board. Legal per rulebook (owner allocates each hit).
        while remaining_hits > 0 and army.units:
            elites = [u for u in army.units if u.kind == UnitKind.ELITE]
            if elites:
                elite = elites[-1]
                army.units.remove(elite)
                casualties.append(elite)
                if remaining_hits >= 2:
                    remaining_hits -= 2
                else:
                    if _consume_downgrade_source(_ACTIVE_STATE,
                                                 elite.nation, _dgc,
                                                 _av0):
                        army.units.append(Unit(elite.nation,
                                               UnitKind.REGULAR))
                        elite._downgraded = True
                    remaining_hits -= 1
            else:
                unit = army.units[-1]
                army.units.remove(unit)
                casualties.append(unit)
                remaining_hits -= 1
        return casualties

    if policy == 'preserve_units':
        while remaining_hits > 0 and army.units:
            elites = [u for u in army.units if u.kind == UnitKind.ELITE]
            if elites:
                elite = elites[-1]
                army.units.remove(elite)
                if _consume_downgrade_source(_ACTIVE_STATE,
                                             elite.nation, _dgc, _av0):
                    army.units.append(Unit(elite.nation,
                                           UnitKind.REGULAR))
                    elite._downgraded = True
                casualties.append(elite)
            else:
                unit = army.units[-1]
                army.units.remove(unit)
                casualties.append(unit)
            remaining_hits -= 1
        return casualties

    # --- 'preserve_elites' (original behavior) ---
    # Sort units: regulars first, then elites
    regulars = [u for u in army.units if u.kind == UnitKind.REGULAR]
    elites = [u for u in army.units if u.kind == UnitKind.ELITE]

    # Remove regulars first
    while remaining_hits > 0 and regulars:
        unit = regulars.pop()
        army.units.remove(unit)
        casualties.append(unit)
        remaining_hits -= 1

    # Then elites (downgrade first, then remove)
    while remaining_hits > 0 and elites:
        elite = elites.pop()
        army.units.remove(elite)

        if remaining_hits >= 2:
            # Full removal
            casualties.append(elite)
            remaining_hits -= 2
        else:
            # Downgrade to regular (box-first then pool, USER
            # 2026-07-20d; in-call counter for the pool leg)
            if _consume_downgrade_source(_ACTIVE_STATE,
                                         elite.nation, _dgc, _av0):
                army.units.append(Unit(elite.nation, UnitKind.REGULAR))
                elite._downgraded = True
            remaining_hits -= 1
            # The elite is (at least partially) lost
            casualties.append(elite)

    return casualties


def is_army_destroyed(army: Army) -> bool:
    """Check if army has been destroyed (no units left)."""
    return len(army.units) == 0


# ---------------------------------------------------------------------------
# Combat Effects Application
# ---------------------------------------------------------------------------

def apply_cancel_effects(combat: CombatState) -> None:
    """Apply card cancellation effects (Swarm of Bats, Daring Defiance) in
    INITIATIVE ORDER (lower combat_value first; the DEFENDER's card first
    on a tie).  Both cancel cards are initiative 0, so on the Swarm-of-Bats
    (Shadow) vs Daring-Defiance (Free Peoples) collision the defender's
    card resolves first — and, being a cancel card, it cancels the
    opponent's card BEFORE that card can act (so the loser's cancel, and
    Daring Defiance's Leadership-forfeit cost, never happen).  USER
    interaction B: Shadow attacking -> DD (defender) first cancels SoB and
    forfeits Companion Leadership; FP attacking -> SoB (defender) first
    cancels DD and Companions RETAIN their Leadership.

    Daring Defiance additionally forfeits ALL participating Companions'
    Leadership RATING (crowned Aragorn = 2) as its cost.  ONCE-ONLY
    LEADERSHIP FORFEIT (USER core rule 5): Leadership already forfeited /
    committed this battle cannot be forfeited a second time.  If ANY
    participating Companion has already forfeited its Leadership — Gandalf
    the White having 'shined' (White Rider) is the pre-cancel-phase case —
    Daring Defiance can no longer pay its all-Companion forfeit cost, so it
    is LEGALLY PLAYED but has NO effect: the opponent's card is NOT
    cancelled (USER interactions A and B-with-White-Rider).
    """
    # Per-round set of Companions whose Leadership is already forfeited /
    # committed.  Cards are per-round, so this resets each round (Phase 1);
    # a battle-scoped White Rider 'shine' commitment persists via
    # _shined_companion and re-seeds it every round.
    _sh0 = getattr(combat, '_shined_companion', None)
    combat._leadership_forfeited = {id(_sh0)} if _sh0 is not None else set()

    def _resolve_side(is_attacker):
        if is_attacker:
            card, cancelled = combat.attacker_card, combat.attacker_card_cancelled
        else:
            card, cancelled = combat.defender_card, combat.defender_card_cancelled
        if not card or cancelled:
            return
        effect = card.combat_effect.lower()
        if "cancel the effects of the combat card" not in effect:
            return
        is_forfeit = "forfeit" in effect and "leadership" in effect
        if is_forfeit:
            # Daring Defiance: if a participating Companion already
            # forfeited its Leadership (White Rider shine), the
            # all-Companion forfeit cost cannot be paid -> NO effect.
            forfeited = getattr(combat, '_leadership_forfeited', set())
            if any(id(c) in forfeited for c in combat.companions_in_battle):
                return
        # Fire: cancel the OPPONENT's card.
        if is_attacker:
            if combat.defender_card:
                combat.defender_card_cancelled = True
        else:
            if combat.attacker_card:
                combat.attacker_card_cancelled = True
        # Daring Defiance cost: forfeit each participating Companion's
        # Leadership RATING (crowned Aragorn = 2), not a flat 1-per-head.
        # companions_in_battle is board-only (1.209/CL-228).  A Companion
        # who already shined (_shined_companion) or was cancelled by Words
        # of Power (_wop_cancelled) is excluded (his Leadership is already
        # gone) and each forfeited Companion is recorded once-only.
        if is_forfeit:
            _shined = getattr(combat, '_shined_companion', None)
            _wopc = getattr(combat, '_wop_cancelled', None)
            forfeited = combat._leadership_forfeited
            companion_leadership = 0
            for c in combat.companions_in_battle:
                if c is _shined or c is _wopc:
                    continue
                companion_leadership += getattr(c, 'leadership', 1)
                forfeited.add(id(c))
            if is_attacker:
                combat.attacker_leadership = max(
                    0, combat.attacker_leadership - companion_leadership)
            else:
                combat.defender_leadership = max(
                    0, combat.defender_leadership - companion_leadership)

    # Initiative order: lower combat_value first, DEFENDER first on tie.
    if _defender_first_initiative(combat):
        _resolve_side(is_attacker=False)
        _resolve_side(is_attacker=True)
    else:
        _resolve_side(is_attacker=True)
        _resolve_side(is_attacker=False)


def apply_dice_modifiers(combat: CombatState) -> None:
    """Apply dice modifier effects from cards.

    1.208h (supersedes 1.79): combat-roll bonuses and leader-re-roll
    bonuses are FULLY SEPARATE. dice_mod is combat-only; reroll_mod is
    re-roll-only. Each card routes by the EXACT target its text names:
      - "combat roll AND leader re-roll" cards (It is a Gift / One for
        the Dark Lord / Deadly Strife / Desperate Battle) fill BOTH
        buckets;
      - combat-only cards (Cruel as Death, Devilry of Orthanc, Mumakil,
        Relentless Assault, Servant of the Secret Fire) touch dice_mod
        ONLY and must NOT touch reroll_mod;
      - re-roll-only cards ("add N to all dice on your Leader re-roll",
        e.g. Swarm of Bats' no-card clause, They are Terrible) touch
        reroll_mod ONLY and must NOT touch dice_mod.
    Advantageous Position's -1 to the Shadow Combat roll must NOT depress
    Shadow re-rolls. The old generic "add 1 to all dice" branch matched
    the re-roll phrasing and wrongly boosted the combat roll (USER audit
    2026-07-09: Swarm of Bats); combat-roll bonuses used to leak into the
    re-roll via reroll_mod = dice_mod + reroll_mod (REMOVED under 1.208h).
    """
    import re as _re

    def _apply(effect, is_attacker):
        # convenience refs
        if is_attacker:
            def add_dice(n): combat.attacker_dice_mod += n
            def add_reroll(n): combat.attacker_reroll_mod += n
            def cost_leadership(n):
                combat.attacker_leadership = max(
                    0, combat.attacker_leadership - n)
            def add_enemy_dice(n): combat.defender_dice_mod += n
            opp_card = combat.defender_card
        else:
            def add_dice(n): combat.defender_dice_mod += n
            def add_reroll(n): combat.defender_reroll_mod += n
            def cost_leadership(n):
                combat.defender_leadership = max(
                    0, combat.defender_leadership - n)
            def add_enemy_dice(n): combat.attacker_dice_mod += n
            opp_card = combat.attacker_card

        # "Both Armies add N" cards (Deadly Strife +2, Desperate Battle
        # +1) name BOTH the combat roll AND the leader re-roll, so under
        # 1.208h they must fill BOTH buckets on both sides (previously
        # dice_mod only, relied on the removed dice_mod->re-roll flow).
        if "both armies add 2" in effect:
            combat.attacker_dice_mod += 2
            combat.defender_dice_mod += 2
            combat.attacker_reroll_mod += 2
            combat.defender_reroll_mod += 2
        elif "both armies add 1" in effect:
            combat.attacker_dice_mod += 1
            combat.defender_dice_mod += 1
            combat.attacker_reroll_mod += 1
            combat.defender_reroll_mod += 1
        # They are Terrible: "Subtract 1 from Nazgul Leadership to add 1
        # to all dice on your Leader re-roll" — handled entirely by
        # apply_they_are_terrible (reroll_mod + the nazgul cost); do NOT
        # also process here or it double-applies.
        elif "subtract 1" in effect and "nazg" in effect \
                and "leader re-roll" in effect:
            pass
        # Cruel as Death: "Subtract 2 from Nazgul Leadership to add 1 to
        # all dice on your Combat roll" — leadership cost then combat +1.
        # USER interaction I: with a shining Gandalf the White (White
        # Rider) present, ALL Nazgul Leadership is negated for the whole
        # battle, so there is nothing to forfeit -> the card has NO effect
        # (the play-legality gate in cards.can_play_combat_card should also
        # refuse it; this resolution self-gate nullifies it either way).
        elif "subtract 2" in effect and "nazg" in effect \
                and "add 1" in effect:
            if getattr(combat, '_shine_companion', None) is None:
                cost_leadership(2)
                add_dice(1)
        # Relentless Assault (also The King is Revealed / Rage of the
        # Dunlendings combat sides): "add 1 to all dice ... FOR EACH HIT
        # you inflicted" — the magnitude is per self-inflicted hit and
        # is applied by apply_relentless_assault. The old generic branch
        # ALSO added a phantom flat +1 here; skip it.
        elif "inflict" in effect and "up to two hits" in effect:
            pass
        else:
            # Generic "add N to all dice on your/their <target>": route
            # by whether the target is the combat roll or ONLY the
            # leader re-roll.  Mûmakil's "if, after the leader re-roll"
            # is a separate sentence, so [^.]* stops at the period and
            # its target is correctly just "combat roll".
            m = _re.search(
                r"add ([12]) to all dice on (?:your|their) ([^.]*)",
                effect)
            if m:
                amt = int(m.group(1))
                target = m.group(2)
                # 1.208h: route by the EXACT target named. INDEPENDENT ifs
                # so a "combat roll and leader re-roll" target fills BOTH
                # buckets; a combat-only target touches dice_mod ONLY; a
                # re-roll-only target touches reroll_mod ONLY.
                if "combat" in target:
                    add_dice(amt)          # combat roll ONLY
                if "re-roll" in target:
                    # re-roll ONLY.  Swarm of Bats gates it on the
                    # opponent having played no combat card.
                    if "did not play a card" in effect:
                        if opp_card is None:
                            add_reroll(amt)
                    else:
                        add_reroll(amt)

        # Subtract from enemy (Swords in Eriador: "Subtract 1 from all
        # dice on the Combat roll of the Shadow player").
        if "subtract 1 from all dice" in effect and "shadow" in effect:
            add_enemy_dice(-1)

    if combat.attacker_card and not combat.attacker_card_cancelled:
        _apply(combat.attacker_card.combat_effect.lower(), True)
    if combat.defender_card and not combat.defender_card_cancelled:
        _apply(combat.defender_card.combat_effect.lower(), False)

    # NOTE: no defender dice bonus for fortifications — per rulebook the
    # only City/Fortification effect is the attacker's first-round 6+
    # threshold (city_defense). The old +1 defender modifier was a house
    # rule (verified vs game0: T5 Fords of Isen, FP rolled at plain 5+).


def _companions_present(combat: CombatState) -> int:
    """Number of Companions PARTICIPATING (separated on the map, in an
    army's leaders).  Companions hidden in the co-located Fellowship do
    not count as 'present' (USER 2026-07-16, same principle as the card
    requirement rule).  1.233: battle rosters are OWN-SIDE only — a
    stranded FP Companion inside the SHADOW army participates in
    nothing, so use the side-filtered roster setup_combat built."""
    return len(getattr(combat, 'companions_in_battle', []) or [])


def apply_dice_count_modifiers(combat: CombatState) -> None:
    """Apply dice count restrictions from cards."""
    # Check attacker card
    if combat.attacker_card and not combat.attacker_card_cancelled:
        effect = combat.attacker_card.combat_effect.lower()

        if "maximum of two dice" in effect:
            combat.defender_dice_cap = 2
        elif "maximum of three dice" in effect:
            combat.defender_dice_cap = 3

        # Brave Stand (USER 2026-07-16: previously non-functional): "The
        # Shadow player rolls one die less in his Combat roll for each
        # Companion present (to a minimum of one)."  Played by FP; the
        # Shadow player is the OPPONENT.  Every Companion in the battle is
        # a Free Peoples figure, so "each Companion present" =
        # len(companions_in_battle).  The old code counted companions in
        # the WRONG army (the Shadow defender, always 0) and used a
        # 5-minus-count cap that ignored the Shadow's real dice, so it
        # never reduced anything.  Route through dice_reduction (subtracted
        # from combat strength; combat_roll floors the count at 1).
        if "rolls one die less" in effect and "companion" in effect:
            companion_count = _companions_present(combat)
            combat.defender_dice_reduction = getattr(
                combat, 'defender_dice_reduction', 0) + companion_count

    # Check defender card
    if combat.defender_card and not combat.defender_card_cancelled:
        effect = combat.defender_card.combat_effect.lower()

        if "maximum of two dice" in effect:
            combat.attacker_dice_cap = 2
        elif "maximum of three dice" in effect:
            combat.attacker_dice_cap = 3

        # Brave Stand played by the FP defender: reduce the Shadow
        # attacker's Combat roll by the number of Companions present.
        if "rolls one die less" in effect and "companion" in effect:
            companion_count = _companions_present(combat)
            combat.attacker_dice_reduction = getattr(
                combat, 'attacker_dice_reduction', 0) + companion_count


def _apply_pre_combat_for_side(combat: CombatState, is_attacker: bool) -> None:
    """Apply pre-combat attack effects for one side."""
    if is_attacker:
        card = combat.attacker_card
        cancelled = combat.attacker_card_cancelled
        leadership = combat.attacker_leadership
        army = combat.attacker_army
    else:
        card = combat.defender_card
        cancelled = combat.defender_card_cancelled
        leadership = combat.defender_leadership
        army = combat.defender_army

    if not card or cancelled:
        return

    effect = card.combat_effect.lower()
    if "before the combat roll" not in effect or "additional attack" not in effect:
        return

    # rulebook p.29: an additional attack with no printed hit number uses the
    # ROUND's base hit number, not a hardcoded 5+. The attacker needs 6+ when
    # assaulting a stronghold/fortification -- a full siege every round, a
    # city/fortification only in the first round; otherwise (and for the
    # defender's own additional attack) the base is 5+. Durin's Bane prints
    # its OWN hit number and keeps it.
    if is_attacker and (getattr(combat, 'is_siege', False) or
                        (getattr(combat, 'city_defense', False)
                         and getattr(combat, 'siege_round', 1) == 1)):
        base_threshold = 6
    else:
        base_threshold = 5

    hits = 0
    rolls = []
    if "leadership" in effect:
        # Sudden Strike: dice = leadership (max 5), at the round's base number
        dice = min(5, leadership)
        rolls, hits = combat_roll(dice, base_threshold, 0)
    elif "elite units" in effect:
        # Charge: dice = elite count (max 5), at the round's base number
        dice = min(5, count_elite_units(army))
        rolls, hits = combat_roll(dice, base_threshold, 0)
    elif "three combat dice" in effect:
        # Durin's Bane: prints its own number (3 dice, hit on 4+)
        rolls, hits = combat_roll(3, 4, 0)

    if is_attacker:
        combat.attacker_pre_combat_hits += hits
        # keep the rolls so self-play traces can show them (they were
        # rolled and previously discarded — coordinator item 2026-07-15)
        combat.attacker_pre_combat_rolls.extend(rolls)
    else:
        combat.defender_pre_combat_hits += hits
        combat.defender_pre_combat_rolls.extend(rolls)


def _get_card_initiative(card, cancelled) -> int:
    """Return the combat_value (initiative) of a card, or 99 if no card."""
    if card and not cancelled:
        return card.combat_value
    return 99


def _defender_first_initiative(combat: CombatState) -> bool:
    """Return True if defender's card has lower or equal initiative (resolves first)."""
    atk_init = _get_card_initiative(combat.attacker_card, combat.attacker_card_cancelled)
    def_init = _get_card_initiative(combat.defender_card, combat.defender_card_cancelled)
    return def_init <= atk_init


def _resolve_pre_combat(combat: CombatState, choices: dict = None) -> None:
    """Pre-combat additional attacks (Sudden Strike / Charge / Durin's
    Bane — all initiative 2) resolved in initiative order (lower first;
    the DEFENDER first on the tie), INTERLEAVED so that (USER interaction
    J):

      * each side's attack magnitude is computed AT RESOLUTION TIME, i.e.
        AFTER the higher-initiative (or defender-on-tie) side has already
        inflicted its casualties — so if Durin's Bane kills Free Peoples
        Elites before Charge resolves, Charge rolls only the Elites still
        present; and
      * the battle ENDS IMMEDIATELY the instant an Army is eliminated: the
        not-yet-resolved side's attack never fires, and the caller skips
        the Combat roll (and roll-dependent cards such as Confusion — USER
        interaction D).
    """
    _pc = choices or {}
    order = [False, True] if _defender_first_initiative(combat) else [True, False]
    for is_attacker in order:
        if combat.battle_over:
            return
        if is_attacker:
            _before = combat.attacker_pre_combat_hits
            _apply_pre_combat_for_side(combat, is_attacker=True)
            _delta = combat.attacker_pre_combat_hits - _before
            if _delta > 0:
                combat.defender_casualties.extend(
                    apply_casualties(combat.defender_army, _delta,
                                     _pc.get('def_casualty_policy',
                                             'preserve_elites')))
                if is_army_destroyed(combat.defender_army):
                    combat.defender_destroyed = True
                    combat.battle_over = True
        else:
            _before = combat.defender_pre_combat_hits
            _apply_pre_combat_for_side(combat, is_attacker=False)
            _delta = combat.defender_pre_combat_hits - _before
            if _delta > 0:
                combat.attacker_casualties.extend(
                    apply_casualties(combat.attacker_army, _delta,
                                     _pc.get('atk_casualty_policy',
                                             'preserve_elites')))
                if is_army_destroyed(combat.attacker_army):
                    combat.attacker_destroyed = True
                    combat.battle_over = True


def apply_reroll_cancellation(combat: CombatState) -> None:
    """Check for Foul Stench effect that cancels leader re-roll.

    USER 1.208j: Foul Stench also cancels that side's Mighty Attack /
    Anduril for the round; the block flag raised here is consumed by
    apply_mighty_attack at Phase 5b.
    """
    # Check if attacker's card cancels defender re-roll
    if combat.attacker_card and not combat.attacker_card_cancelled:
        effect = combat.attacker_card.combat_effect.lower()
        if "leader re-roll is cancelled" in effect:
            # Check if nazgul leadership >= FP leadership
            if combat.nazgul_in_battle >= combat.defender_leadership:
                combat.defender_reroll_cancelled = True
                combat._foul_stench_cancel_def = True

    # Check if defender's card cancels attacker re-roll
    if combat.defender_card and not combat.defender_card_cancelled:
        effect = combat.defender_card.combat_effect.lower()
        if "leader re-roll is cancelled" in effect:
            if combat.nazgul_in_battle >= combat.attacker_leadership:
                combat.attacker_reroll_cancelled = True
                combat._foul_stench_cancel_atk = True


def _apply_bonus_hits_for_side(combat: CombatState, is_attacker: bool) -> None:
    """Apply post-reroll bonus hit effects for one side.

    The running totals INCLUDE bonus hits already scored this phase by the
    higher-priority (earlier-resolving) side, so the Mumakil "more total
    hits" test at initiative 5 sees a No Quarter / Nameless Wood bonus that
    the DEFENDER (resolving first on the init-5 tie) has already added, but
    NOT one the attacker has yet to add (USER interaction F: Nameless Wood /
    No Quarter hits count toward the Mumakil determination ONLY when the FP
    Army is defending).  Sudden Strike / Charge / Durin's Bane pre-combat
    hits are counted too (they are already in *_pre_combat_hits)."""
    atk_total = (combat.attacker_hits + combat.attacker_pre_combat_hits
                 + combat.attacker_bonus_hits)
    def_total = (combat.defender_hits + combat.defender_pre_combat_hits
                 + combat.defender_bonus_hits)

    if is_attacker:
        card = combat.attacker_card
        cancelled = combat.attacker_card_cancelled
        my_total = atk_total
        enemy_total = def_total
        my_army = combat.attacker_army
        enemy_army = combat.defender_army
    else:
        card = combat.defender_card
        cancelled = combat.defender_card_cancelled
        my_total = def_total
        enemy_total = atk_total
        my_army = combat.defender_army
        enemy_army = combat.attacker_army

    if not card or cancelled:
        return

    effect = card.combat_effect.lower()

    # No Quarter: +1 if scored at least 1 hit
    if ("score at least one hit" in effect or "scores at least one hit" in effect) and "score one additional hit" in effect:
        if my_total >= 1:
            if is_attacker:
                combat.attacker_bonus_hits += 1
            else:
                combat.defender_bonus_hits += 1

    # Nameless Wood: +2 if scored at least 1 hit
    if ("score at least one hit" in effect or "scores at least one hit" in effect) and "score two additional hits" in effect:
        if my_total >= 1:
            if is_attacker:
                combat.attacker_bonus_hits += 2
            else:
                combat.defender_bonus_hits += 2

    # Great Host ("twice as many"): NOT here.  The 2x-army auto-hit is
    # checked AFTER this round's casualties are removed (card text +
    # almanac p43) — moved to _apply_post_casualty_for_side (audit
    # 2026-07-26; prior flags 40/41 were mislabeled FIXED).

    # Mumakil: +1 if more hits than enemy
    if "more total hits" in effect and "score one additional hit" in effect:
        if my_total > enemy_total:
            if is_attacker:
                combat.attacker_bonus_hits += 1
            else:
                combat.defender_bonus_hits += 1


def apply_bonus_hits(combat: CombatState) -> None:
    """Apply post-reroll bonus hit effects (No Quarter, Nameless Wood,
    Mumakil "more total hits").

    Every effect resolved here fires at the SAME timing window — after the
    Leader re-roll — which the spec assigns initiative 5 (the Mumakil
    hit-test is calculated "at 5", and No Quarter / Nameless Wood are also
    5).  With both sides tied at 5 the DEFENDER always resolves first, so a
    defending FP Army's No Quarter / Nameless Wood bonus is scored before
    the attacking Shadow's Mumakil test reads the totals, while an
    ATTACKING FP Army's bonus is added only after Mumakil (defender) has
    already tested (USER interaction F).  The Mumakil card's printed
    combat_value (3, for its +1 combat-roll modifier applied back at Phase
    2) does NOT reorder this window."""
    _apply_bonus_hits_for_side(combat, is_attacker=False)
    _apply_bonus_hits_for_side(combat, is_attacker=True)


def apply_casualty_prevention(combat: CombatState, choices: dict = None) -> None:
    """Apply casualty prevention effects (Shield-Wall, Heroic Death)."""
    choices = choices or {}

    # Attacker prevention
    if combat.attacker_card and not combat.attacker_card_cancelled:
        effect = combat.attacker_card.combat_effect.lower()

        # Shield-Wall: cancel 1 if opponent scored 2+. The printed trigger
        # reads the opponent's COMBAT roll + Leader re-roll ONLY -- and
        # combat.defender_hits is exactly that scope post-1.208h. Pre-combat
        # additional attacks (Sudden Strike/Charge/Durin's Bane) and
        # post-reroll bonus hits (No Quarter/Nameless Wood/Great Host/
        # Mumakil) are OUT of the card's printed scope.
        if "two or more hits" in effect and "cancel one hit" in effect:
            if combat.defender_hits >= 2:
                combat.attacker_hits_cancelled += 1

    # Defender prevention
    if combat.defender_card and not combat.defender_card_cancelled:
        effect = combat.defender_card.combat_effect.lower()

        if "two or more hits" in effect and "cancel one hit" in effect:
            if combat.attacker_hits >= 2:
                combat.defender_hits_cancelled += 1


def apply_confusion_effect(combat: CombatState) -> None:
    """Apply Confusion effect where 1s hurt the roller."""
    # Check if Confusion is active (FP card effect)
    if combat.defender_card and not combat.defender_card_cancelled:
        effect = combat.defender_card.combat_effect.lower()
        if "result of" in effect and "1" in effect and "shadow army" in effect:
            combat.attacker_confusion = True

    if combat.attacker_card and not combat.attacker_card_cancelled:
        effect = combat.attacker_card.combat_effect.lower()
        if "result of" in effect and "1" in effect and "shadow army" in effect:
            combat.defender_confusion = True


def apply_words_of_power(combat: CombatState) -> None:
    """Words of Power: cancel one Companion for this Combat round.

    SH card (val=1): 'Choose a Companion. That Companion's Leadership
    and special abilities are cancelled for this Combat round.'
    AI heuristic: pick the highest-Leadership companion. USER 1.208i, the
    cancel spans (a) that Companion's Leadership re-roll contribution, (b)
    his Captain-of-the-West COMBAT die (applied at the roll site, pre-cap;
    4R+Boromir 5->4), and (c) his special abilities: Gandalf's shine (the
    Nazgul Leadership it removed is restored) and the Mighty Attack /
    Anduril payer (refused in apply_mighty_attack). Gandalf the White is
    not a Captain of the West -> no combat die is removed for him; only the
    shine restore and the Leadership subtract fire.
    """
    for card, cancelled, is_atk in [
        (combat.attacker_card, combat.attacker_card_cancelled, True),
        (combat.defender_card, combat.defender_card_cancelled, False),
    ]:
        if card and not cancelled:
            effect = card.combat_effect.lower()
            if "companion" in effect and "leadership" in effect and "cancelled" in effect:
                # CARD-FLEET flags 5/34 (2026-07-21): Words of Power's
                # play condition is "Play if a Nazgul is in the
                # battle". Defense-in-depth resolver gate (siblings
                # apply_they_are_terrible / apply_dread_and_despair
                # already self-gate): a direct resolve/test/bridge
                # path must never cancel Leadership with no Nazgul
                # present. Runs Phase 1b, before any forfeit
                # decrements nazgul_in_battle.
                if combat.nazgul_in_battle < 1:
                    continue
                if combat.companions_in_battle:
                    target = max(combat.companions_in_battle,
                                 key=lambda c: getattr(c, 'leadership', 0) or 0)
                    # FP companions fight OPPOSITE the Shadow card's player.
                    fp_is_attacker = not is_atk
                    # (c) special ability: shining GtW -> cancel shine,
                    # restore the Nazgul Leadership it removed from the
                    # Shadow side.
                    if target is getattr(combat, '_shine_companion', None):
                        _restore = getattr(
                            combat, '_shine_cancelled_leadership', 0)
                        if combat.attacker_side == Side.SHADOW:
                            combat.attacker_leadership += _restore
                        else:
                            combat.defender_leadership += _restore
                    # (a) Leadership subtract from the FP side, UNLESS he
                    # already forfeited it by shining (would double-count).
                    if target is not getattr(combat, '_shined_companion', None):
                        _lead = getattr(target, 'leadership', 0) or 0
                        if fp_is_attacker:
                            combat.attacker_leadership = max(
                                0, combat.attacker_leadership - _lead)
                        else:
                            combat.defender_leadership = max(
                                0, combat.defender_leadership - _lead)
                    # (b)/(c) markers: roll site drops the Captain die;
                    # apply_mighty_attack refuses the cancelled payer.
                    combat._wop_cancelled = target
                    combat._wop_cancelled_is_attacker = fp_is_attacker


def apply_relentless_assault(combat: CombatState, choices: dict = None) -> None:
    """Relentless Assault: self-damage for +1 dice modifier per hit.

    SH card (val=3): 'Before the Combat roll, you may inflict and apply
    up to two hits against your units. Add 1 to all dice on your Combat
    roll for each hit you inflicted.'

    The magnitude is a player choice. When the owner decided it via the
    learned 'combat_effect_choice' phase, choices carries the count as
    'atk_effect_count'/'def_effect_count' (0 = decline). Without a
    forced count (direct resolve_combat callers, GUI, tests) the old
    heuristic applies: sacrifice min(2, units-2) to leave at least 2.
    """
    choices = choices or {}
    for card, cancelled, is_atk in [
        (combat.attacker_card, combat.attacker_card_cancelled, True),
        (combat.defender_card, combat.defender_card_cancelled, False),
    ]:
        if card and not cancelled:
            effect = card.combat_effect.lower()
            if "inflict" in effect and "up to two hits" in effect and "add 1" in effect:
                army = combat.attacker_army if is_atk else combat.defender_army
                fc = choices.get('atk_effect_count' if is_atk
                                 else 'def_effect_count')
                if fc is not None:
                    # Learned decision: inflict exactly fc hits, clamped
                    # to the card max and the army's HIT CAPACITY
                    # (2*elites + regulars).  Almanac p44 permits wiping
                    # the whole army (audit 2026-07-26; the old len-1
                    # keep-one mirrored the pre-fix Onslaught bug).
                    _e_ra = count_elite_units(army)
                    _cap_ra = 2 * _e_ra + (len(army.units) - _e_ra)
                    k = max(0, min(int(fc), 2, _cap_ra))
                    if k > 0:
                        casualties = apply_casualties(army, k)
                        if is_atk:
                            combat.attacker_casualties.extend(casualties)
                            combat.attacker_dice_mod += k
                        else:
                            combat.defender_casualties.extend(casualties)
                            combat.defender_dice_mod += k
                        # review P2 (2026-07-27): a FULL self-wipe must
                        # set the destroyed flags here — the emptied
                        # army otherwise rolled a phantom floor-1 die
                        # before Phase 8 caught the destruction
                        if is_army_destroyed(army):
                            if is_atk:
                                combat.attacker_destroyed = True
                            else:
                                combat.defender_destroyed = True
                            combat.battle_over = True
                    continue
                max_sacrifice = min(2, max(0, len(army.units) - 2))
                if max_sacrifice > 0:
                    casualties = apply_casualties(army, max_sacrifice)
                    # card text: +1 per HIT inflicted — an elite absorbs
                    # 2 hits as ONE removal, so count hits, not figures
                    if is_atk:
                        combat.attacker_casualties.extend(casualties)
                        combat.attacker_dice_mod += max_sacrifice
                    else:
                        combat.defender_casualties.extend(casualties)
                        combat.defender_dice_mod += max_sacrifice


def apply_dread_and_despair(combat: CombatState, choices: dict = None) -> None:
    """Dread and Despair: forfeit Nazgul leadership to reduce FP dice.

    SH card (val=3): 'Before the Combat roll, forfeit one or more points
    of Nazgul Leadership. The FP player rolls one die less (min 1) per
    point forfeited.'

    How many points to forfeit is a player choice (game0 T9: the human
    announced 'ff3' twice — forfeit 3, FP rolled 2 instead of 5). When
    decided via the learned 'combat_effect_choice' phase, choices carries
    the count as 'atk_effect_count'/'def_effect_count'. Fallback
    heuristic: forfeit enough to reduce FP to 1 die, keep 1 for re-roll.
    """
    choices = choices or {}
    for card, cancelled, is_atk in [
        (combat.attacker_card, combat.attacker_card_cancelled, True),
        (combat.defender_card, combat.defender_card_cancelled, False),
    ]:
        if card and not cancelled:
            effect = card.combat_effect.lower()
            if "forfeit" in effect and "nazg" in effect and "one die less" in effect:
                # USER interaction I: a shining Gandalf the White (White
                # Rider) negates ALL Nazgul Leadership for the battle, so
                # there is none to forfeit -> the card has NO effect.
                if getattr(combat, '_shine_companion', None) is not None:
                    continue
                # Only Shadow plays this card
                nazgul_available = combat.nazgul_in_battle
                if nazgul_available <= 0:
                    continue
                # Calculate FP combat strength
                fp_army = combat.defender_army if is_atk else combat.attacker_army
                fp_strength = calc_combat_strength(fp_army)
                fc = choices.get('atk_effect_count' if is_atk
                                 else 'def_effect_count')
                if fc is not None:
                    forfeit = max(0, min(int(fc), nazgul_available))
                else:
                    # Forfeit enough to reduce FP to 1 die, keep at least
                    # 1 for re-roll
                    desired_reduction = max(0, fp_strength - 1)
                    forfeit = min(desired_reduction,
                                  max(0, nazgul_available - 1))
                if forfeit > 0:
                    # Reduce leadership
                    if is_atk:
                        combat.attacker_leadership -= forfeit
                        combat.defender_dice_cap = max(1, fp_strength - forfeit)
                    else:
                        combat.defender_leadership -= forfeit
                        combat.attacker_dice_cap = max(1, fp_strength - forfeit)
                    combat.nazgul_in_battle -= forfeit


def apply_they_are_terrible(combat: CombatState) -> None:
    """They are Terrible: forfeit 1 Nazgul leadership for +1 to re-roll dice.

    SH card (val=4): 'Subtract 1 from your total Nazgul Leadership to
    add 1 to all dice on your Leader re-roll.'
    The effect is delivered via combat.*_reroll_mod.
    """
    for card, cancelled, is_atk in [
        (combat.attacker_card, combat.attacker_card_cancelled, True),
        (combat.defender_card, combat.defender_card_cancelled, False),
    ]:
        if card and not cancelled:
            effect = card.combat_effect.lower()
            if "subtract 1" in effect and "nazg" in effect and "leader re-roll" in effect:
                # USER interaction I: a shining Gandalf the White (White
                # Rider) negates ALL Nazgul Leadership for the battle, so
                # there is none to forfeit -> the card has NO effect.
                if getattr(combat, '_shine_companion', None) is not None:
                    continue
                if combat.nazgul_in_battle >= 1:
                    combat.nazgul_in_battle -= 1
                    if is_atk:
                        combat.attacker_leadership -= 1
                        combat.attacker_reroll_mod += 1
                    else:
                        combat.defender_leadership -= 1
                        combat.defender_reroll_mod += 1


def apply_mighty_attack(combat: CombatState, rolls: List[int],
                        threshold: int, modifier: int,
                        is_attacker: bool) -> int:
    """Mighty Attack / Anduril: forfeit companion leadership to auto-convert misses.

    FP card (val=4): 'Forfeit the Leadership of one Companion to automatically
    change one missed die roll to a hit.'
    Anduril: 'Forfeit Strider's Leadership (1 hit) or Aragorn's Leadership (2 hits).'
    Returns number of auto-converted hits.
    """
    card = combat.attacker_card if is_attacker else combat.defender_card
    cancelled = combat.attacker_card_cancelled if is_attacker else combat.defender_card_cancelled
    if not card or cancelled:
        return 0

    # USER 1.208j: Foul Stench also cancels this side's Mighty Attack /
    # Anduril for the round (flag from apply_reroll_cancellation).
    if is_attacker and getattr(combat, '_foul_stench_cancel_atk', False):
        return 0
    if (not is_attacker) and getattr(combat, '_foul_stench_cancel_def', False):
        return 0

    effect = card.combat_effect.lower()
    auto_hits = 0

    if "forfeit" in effect and "leadership" in effect and "change" in effect and "missed" in effect:
        # Count misses in the combat roll
        misses = sum(1 for r in rolls if r != 6 and (r + modifier) < threshold)
        if misses == 0:
            return 0

        # A companion who already forfeited his Leadership by SHINING
        # (Gandalf the White) cannot forfeit it again (USER 2026-07-16
        # item 11). USER 1.208i: a Companion cancelled by Words of Power
        # this round likewise cannot pay Mighty Attack / Anduril.
        _shined = getattr(combat, '_shined_companion', None)
        _wop = getattr(combat, '_wop_cancelled', None)
        companions = [c for c in combat.companions_in_battle
                      if c is not _shined and c is not _wop]
        if not companions:
            return 0

        # USER 2026-07-16: "forfeit the Leadership of ONE Companion" removes
        # only THAT character's Leadership contribution (Strider 1, crowned
        # Aragorn 2, most companions 1) — NOT the companion's Level and NOT
        # the side's total Leadership.  The old code subtracted comp.level
        # (3 for Strider/Aragorn, 2 for Boromir/Legolas/Gimli), which ate
        # into every other Leader's contribution and effectively forfeited
        # the whole re-roll.
        def _forfeit(amt):
            nonlocal_amt = max(0, int(amt))
            if is_attacker:
                combat.attacker_leadership = max(
                    0, combat.attacker_leadership - nonlocal_amt)
            else:
                combat.defender_leadership = max(
                    0, combat.defender_leadership - nonlocal_amt)

        # Andúril (Strider/Aragorn): forfeit Strider's Leadership (1) to
        # convert one miss, OR crowned Aragorn's Leadership (2) for up to two
        if "aragorn" in effect or "strider" in effect:
            aragorn = [c for c in companions if "aragorn" in c.name.lower() or "strider" in c.name.lower()]
            if aragorn:
                comp = aragorn[0]
                crowned = (getattr(comp, 'crowned', False)
                           or getattr(comp, 'leadership', 1) >= 2)
                max_convert = 2 if crowned else 1
                auto_hits = min(max_convert, misses)
                _forfeit(getattr(comp, 'leadership', 1))
                return auto_hits
            # CARD-FLEET LOW-1 (2026-07-20): Anduril is paid by
            # STRIDER's Leadership (1 miss) or crowned ARAGORN's (up
            # to 2). With neither hero in the battle the card has NO
            # effect (combat_req: "Play if Strider/Aragorn is in the
            # battle"). Refuse HERE so a bypassed legality gate cannot
            # fall through to the generic Mighty Attack path below and
            # convert a miss off another Companion (probe: Gimli-only
            # battle forfeited Gimli's Leadership for a hit).
            return 0

        # Generic Mighty Attack: forfeit one Companion's Leadership, convert
        # one miss.  Pick the companion whose Leadership loss costs least.
        target = min(companions, key=lambda c: getattr(c, 'leadership', 1))
        _forfeit(getattr(target, 'leadership', 1))
        return 1

    return 0


def _card_kill_minion_bookkeeping(combat: CombatState, leader) -> None:
    """RULING 1.228c (USER 2026-08-03): a Minion eliminated by a combat
    card (Fateful Strike / Blade of Westernesse) needs the SAME
    character-sheet bookkeeping as one lost with a destroyed army
    (_handle_eliminated_minions): flip state.witch_king / .saruman /
    .mouth_of_sauron .in_play to False and, for Saruman, fire the
    elimination discards (The Palantir of Orthanc / Wormtongue leave
    the table).  Without this a card-killed minion kept granting his
    bonus action die and end-of-battle abilities.  Feeds the ai.py
    gates: 1.228b kills a card-killed Witch-king's queued 'Sorcerer'
    draw via the in_play flag.

    State comes from combat._live_state (stashed by setup_combat) or
    the module _ACTIVE_STATE published by game_logic.battle(); with
    neither (bare unit-test CombatState) this is a no-op."""
    state = getattr(combat, '_live_state', None)
    if state is None:
        state = _ACTIVE_STATE
    if state is None:
        return
    _handle_eliminated_minions(state, [leader])


def apply_fateful_strike(combat: CombatState, reroll_hits: int,
                         is_attacker: bool) -> None:
    """Fateful Strike: eliminate Nazgul on leader re-roll hit.

    FP card (val=6): 'If your Leader re-roll scores at least one hit,
    additionally eliminate a Nazgul (if present). If 2+ hits, can eliminate
    a Minion instead.'

    RULING 1.228 (USER 2026-08-03): resolving here at Phase 5d — before
    the Phase-7 Black Breath / Heroic Death window — is CORRECT even
    though FS and Black Breath are both initiative 6: FS activates when
    its play condition (the Leader re-roll) is met; the tied-initiative
    defender-first rule is not relevant to this pairing.
    """
    card = combat.attacker_card if is_attacker else combat.defender_card
    cancelled = combat.attacker_card_cancelled if is_attacker else combat.defender_card_cancelled
    if not card or cancelled:
        return

    effect = card.combat_effect.lower()
    if "re-roll" not in effect or "nazg" not in effect:
        return

    if reroll_hits >= 1:
        enemy_army = combat.defender_army if is_attacker else combat.attacker_army
        from game_state import Nazgul
        from game_logic import Minion
        # With 2+ re-roll hits the card lets you eliminate a MINION instead
        # of a Nazgûl (card text).  The USER house rule (item 8) requires a
        # Shadow Minion in the battle for this card, so on 2+ hits prefer
        # removing that Minion (it is the whole point of the play), falling
        # back to a Nazgûl only if no Minion is present.
        if reroll_hits >= 2:
            for i, leader in enumerate(enemy_army.leaders):
                if isinstance(leader, Minion):
                    enemy_army.leaders.pop(i)
                    if leader.name in combat.minions_in_battle:
                        combat.minions_in_battle.remove(leader.name)
                    # RULING 1.228c: the Witch-king is modelled as a
                    # Minion figure (NOT a Nazgul instance), so THIS
                    # branch is the one that can eliminate him — flip
                    # the character sheet (feeds the ai.py 1.228b
                    # Sorcerer gate); a Saruman kill also discards
                    # Palantir of Orthanc / Wormtongue.
                    _card_kill_minion_bookkeeping(combat, leader)
                    return
        # 1 hit (or no Minion present): eliminate a Nazgûl.  Plain
        # Nazgul have no character-sheet entry (the Witch-king is a
        # Minion, handled above) — no 1.228c bookkeeping needed here.
        for i, leader in enumerate(enemy_army.leaders):
            if isinstance(leader, Nazgul):
                enemy_army.leaders.pop(i)
                combat.nazgul_in_battle = max(0, combat.nazgul_in_battle - 1)
                return


def apply_blade_of_westernesse(combat: CombatState, reroll_hits: int,
                                is_attacker: bool) -> None:
    """Blade of Westernesse: use one hit to eliminate one Minion.

    FP card (val=6): 'Use one hit during the Leader re-roll to eliminate
    one Minion of your choice that is participating in the battle.'
    """
    card = combat.attacker_card if is_attacker else combat.defender_card
    cancelled = combat.attacker_card_cancelled if is_attacker else combat.defender_card_cancelled
    if not card or cancelled:
        return

    effect = card.combat_effect.lower()
    if "eliminate" not in effect or "minion" not in effect or "westernesse" not in card.combat_name.lower():
        return

    if reroll_hits >= 1:
        enemy_army = combat.defender_army if is_attacker else combat.attacker_army
        from game_logic import Minion
        for i, leader in enumerate(enemy_army.leaders):
            if isinstance(leader, Minion):
                enemy_army.leaders.pop(i)
                if leader.name in combat.minions_in_battle:
                    combat.minions_in_battle.remove(leader.name)
                # RULING 1.228c: full character-sheet bookkeeping —
                # in_play flip (Witch-king / Saruman / Mouth) and, for
                # Saruman, the Palantir of Orthanc / Wormtongue table
                # discards, so the dead minion's abilities cannot
                # activate later.
                _card_kill_minion_bookkeeping(combat, leader)
                return


def apply_black_breath(combat: CombatState, reroll_hits: int,
                       is_attacker: bool) -> None:
    """Black Breath: eliminate FP leader or companion on re-roll hit.

    SH card (val=6): 'If your Leader re-roll scores at least one hit, you may
    additionally eliminate one FP Leader. Or eliminate a Companion if hits
    >= Companion level.'
    AI heuristic: eliminate highest-level viable companion, else a leader.
    """
    card = combat.attacker_card if is_attacker else combat.defender_card
    cancelled = combat.attacker_card_cancelled if is_attacker else combat.defender_card_cancelled
    if not card or cancelled:
        return
    # 2026-07-21 resolver-gate parity (same class as Words of Power):
    # Black Breath's play condition is "Play if a Nazgul is in the
    # battle"; a re-roll can come from non-Nazgul leadership (Saruman,
    # Isengard elites, Mouth), so self-gate here too.
    # RULING 1.228 (USER 2026-08-03): vs Fateful Strike (also init 6)
    # BOTH cards resolve — FS at Phase 5d on its own trigger, BB here
    # at Phase 7; defender-first tie-breaking does not apply.  Kept
    # edge case: FS killing the SOLE Nazgul at 5d drops
    # nazgul_in_battle to 0 and this gate then blocks BB (certified
    # CL-216).
    if getattr(combat, 'nazgul_in_battle', 0) < 1:
        return

    effect = card.combat_effect.lower()
    if "re-roll" not in effect or "eliminate" not in effect or "leader" not in effect:
        return
    if "black breath" not in card.combat_name.lower():
        return

    if reroll_hits < 1:
        return

    # Try to eliminate highest-level companion whose level <= hits
    enemy_army = combat.defender_army if is_attacker else combat.attacker_army
    viable_companions = [
        c for c in combat.companions_in_battle if c.level <= reroll_hits
    ]
    if viable_companions:
        target = max(viable_companions, key=lambda c: c.level)
        if target in combat.companions_in_battle:
            combat.companions_in_battle.remove(target)
            # Remove from army leaders
            for i, l in enumerate(enemy_army.leaders):
                if isinstance(l, Companion) and l.name == target.name:
                    enemy_army.leaders.pop(i)
                    break
            # RULING 1.228 wave (USER 2026-08-03): the Leadership POOL
            # drops by the Companion's LEADERSHIP rating (e.g. Gandalf:
            # level 3, leadership 1), NOT his level.  The viability
            # gate above stays LEVEL-based per the card text ("if the
            # number of hits equals or exceeds that Companion's
            # Level").  Only visible in the engine-internal multi-round
            # loop — the per-round ai.py flow rebuilds leadership via
            # setup_combat each round.
            _ldr_drop = getattr(target, 'leadership', 1)
            if is_attacker:
                combat.defender_leadership = max(
                    0, combat.defender_leadership - _ldr_drop)
            else:
                combat.attacker_leadership = max(
                    0, combat.attacker_leadership - _ldr_drop)
            return

    # Else eliminate a regular FP leader.  OWN-SIDE plain Leaders only
    # (1.233): a stranded Shadow Minion in the FP army must never be
    # the Black Breath target (it contributes no FP leadership and its
    # in_play bookkeeping would be skipped).
    if enemy_army.leaders:
        for i, leader in enumerate(enemy_army.leaders):
            if (type(leader).__name__ == 'Leader'
                    and _combat_leader_side(leader)
                    == Side.FREE_PEOPLES):
                enemy_army.leaders.pop(i)
                if is_attacker:
                    combat.defender_leadership = max(0, combat.defender_leadership - 1)
                else:
                    combat.attacker_leadership = max(0, combat.attacker_leadership - 1)
                return


def _heroic_death_sacrifice(combat: CombatState, is_atk: bool, army: Army,
                            fc: int, total_enemy: int) -> None:
    """Apply a chosen Heroic Death sacrifice of magnitude fc.

    fc semantics (mirrors the learned choice enumeration): 1 = eliminate
    a plain Leader (cancel 1 hit); >= 2 = eliminate a Companion whose
    level covers fc (cancel up to fc hits). fc == 1 falls back to a
    level-1 companion when no plain leader is present.
    """
    from game_state import Nazgul
    if fc <= 0 or total_enemy <= 0:
        return

    def _cancel(n):
        if is_atk:
            combat.attacker_hits_cancelled += n
        else:
            combat.defender_hits_cancelled += n

    # fc == 1 prefers an OWN-SIDE plain Leader (1.233: a stranded
    # Shadow Minion in the FP army is never a legal FP sacrifice)
    if fc == 1:
        for i, leader in enumerate(army.leaders):
            if (type(leader).__name__ == 'Leader'
                    and _combat_leader_side(leader)
                    == Side.FREE_PEOPLES):
                army.leaders.pop(i)
                _cancel(1)
                if is_atk:
                    combat.attacker_leadership = max(
                        0, combat.attacker_leadership - 1)
                else:
                    combat.defender_leadership = max(
                        0, combat.defender_leadership - 1)
                return
    # Companion sacrifice: smallest level covering fc, else best available
    viable = sorted((c for c in combat.companions_in_battle
                     if c.level >= max(1, fc)), key=lambda c: c.level)
    if not viable:
        viable = sorted(combat.companions_in_battle,
                        key=lambda c: -c.level)
    if not viable:
        return
    target = viable[0]
    combat.companions_in_battle.remove(target)
    for i, l in enumerate(army.leaders):
        if isinstance(l, Companion) and l.name == target.name:
            army.leaders.pop(i)
            break
    # Hit-CANCEL magnitude stays LEVEL-based per the card text ("cancel
    # a number of hits equal to or less than the Companion's Level") —
    # verified against card_list.csv, USER 2026-08-03.
    _cancel(min(max(fc, 1), target.level, total_enemy))
    # RULING 1.228 wave (USER 2026-08-03): the Leadership POOL drops by
    # the Companion's LEADERSHIP rating, not his level (engine-internal
    # multi-round loop only; the per-round ai.py flow rebuilds
    # leadership each round).
    _ldr_drop = getattr(target, 'leadership', 1)
    if is_atk:
        combat.attacker_leadership = max(
            0, combat.attacker_leadership - _ldr_drop)
    else:
        combat.defender_leadership = max(
            0, combat.defender_leadership - _ldr_drop)


def _heroic_death_for_side(combat: CombatState, is_atk: bool,
                           choices: dict = None) -> None:
    """Heroic Death for ONE side (the Free Peoples side holding the card).

    FP card (val=6): 'Before removing casualties, eliminate a Leader to
    cancel one hit, or eliminate one Companion to cancel hits <= level.'

    Which figure to sacrifice (or declining) is a player choice. When
    decided via the learned 'combat_effect_choice' phase, choices carries
    the magnitude as 'atk_effect_count'/'def_effect_count' (0 = decline,
    1 = leader, >=2 = companion of that level). Fallback heuristic: use
    only if total enemy hits would kill >50% of army.

    Extracted so the initiative-6 Black Breath (Shadow) vs Heroic Death
    (Free Peoples) window can resolve each side in defender-first order
    against the shared pool of Free Peoples leaders (see _apply_bb_hd).
    """
    choices = choices or {}
    card = combat.attacker_card if is_atk else combat.defender_card
    cancelled = (combat.attacker_card_cancelled if is_atk
                 else combat.defender_card_cancelled)
    if not card or cancelled:
        return
    if "heroic death" not in card.combat_name.lower():
        return

    # Calculate total CANCELLABLE hits against this side.  Pre-combat
    # attack hits (Sudden Strike/Charge/Durin's Bane) were already
    # applied as casualties at Phase 3 and can no longer be cancelled
    # (same scope discipline as Shield-Wall).
    if is_atk:
        total_enemy = (combat.defender_hits +
                       combat.defender_bonus_hits - combat.attacker_hits_cancelled)
    else:
        total_enemy = (combat.attacker_hits +
                       combat.attacker_bonus_hits - combat.defender_hits_cancelled)
    total_enemy = max(0, total_enemy)

    army = combat.attacker_army if is_atk else combat.defender_army
    army_size = len(army.units)

    fc = choices.get('atk_effect_count' if is_atk else 'def_effect_count')
    if fc is not None:
        # Learned decision: 0 declines, otherwise apply the chosen
        # sacrifice at the printed timing (before casualties are removed)
        _heroic_death_sacrifice(combat, is_atk, army, int(fc), total_enemy)
        return

    # Only sacrifice if hits would destroy >50% of army
    if total_enemy <= army_size // 2:
        return

    # Try companion sacrifice first (cancels up to companion's level hits)
    best_comp = None
    best_value = 0
    for c in combat.companions_in_battle:
        cancel = min(c.level, total_enemy)
        if cancel > best_value:
            best_comp = c
            best_value = cancel

    if best_comp and best_value >= 2:
        combat.companions_in_battle.remove(best_comp)
        for i, l in enumerate(army.leaders):
            if isinstance(l, Companion) and l.name == best_comp.name:
                army.leaders.pop(i)
                break
        # RULING 1.228 wave (USER 2026-08-03): pool drops by LEADERSHIP
        # rating, not level (same fix as _heroic_death_sacrifice; the
        # cancel magnitude best_value stays level-based per card text).
        _ldr_drop = getattr(best_comp, 'leadership', 1)
        if is_atk:
            combat.attacker_hits_cancelled += best_value
            combat.attacker_leadership = max(
                0, combat.attacker_leadership - _ldr_drop)
        else:
            combat.defender_hits_cancelled += best_value
            combat.defender_leadership = max(
                0, combat.defender_leadership - _ldr_drop)
        return

    # Else sacrifice a regular leader to cancel 1 hit.  OWN-SIDE plain
    # Leaders only (1.233): never pay the FP card cost with a stranded
    # Shadow Minion.
    for i, leader in enumerate(army.leaders):
        if (type(leader).__name__ == 'Leader'
                and _combat_leader_side(leader) == Side.FREE_PEOPLES):
            army.leaders.pop(i)
            if is_atk:
                combat.attacker_hits_cancelled += 1
                combat.attacker_leadership = max(0, combat.attacker_leadership - 1)
            else:
                combat.defender_hits_cancelled += 1
                combat.defender_leadership = max(0, combat.defender_leadership - 1)
            return


def _apply_bb_hd(combat: CombatState, atk_reroll_hits: int,
                 def_reroll_hits: int, choices: dict = None) -> None:
    """Resolve the initiative-6 post-reroll Free-Peoples-leader window —
    Black Breath (Shadow) and Heroic Death (Free Peoples) — in initiative
    order, DEFENDER first on the tie, so the defender's card consumes the
    shared pool of Free Peoples leaders/companions BEFORE the attacker's
    (USER interaction H):
      * Shadow defending  -> Black Breath first; if it eliminates all FP
        leadership, Heroic Death has nothing left to sacrifice.
      * FP defending      -> Heroic Death first; if it sacrifices the last
        FP leader, Black Breath has no target left to eliminate.
    Both effects trigger 'after the Leader re-roll, before casualties are
    removed'; each per-side call is self-guarding, so only the side that
    actually holds the card fires."""
    def _side(is_attacker):
        rh = atk_reroll_hits if is_attacker else def_reroll_hits
        apply_black_breath(combat, rh, is_attacker=is_attacker)
        _heroic_death_for_side(combat, is_attacker, choices)

    if _defender_first_initiative(combat):
        _side(False)
        _side(True)
    else:
        _side(True)
        _side(False)


def _apply_post_casualty_for_side(combat: CombatState, is_attacker: bool,
                                  choices: dict = None) -> None:
    """Apply post-casualty effects (Onslaught, We Come to Kill) for one side."""
    choices = choices or {}
    if combat.battle_over:
        # fix 78 (game-14 canary): the deferred-Onslaught path must still
        # be reachable when THIS round's casualties ended the battle — a
        # spec-forced round folds the onslaught-caused kills into the
        # main casualty step, so the enemy is already destroyed while
        # the surviving owner's sacrifice (a logged decision) still
        # needs its pending phase.  Only the deferral escapes this gate;
        # auto-resolved effects on a finished battle stay moot (the
        # enemy-destroyed guards below skip them).
        _own_dead78 = (combat.attacker_destroyed if is_attacker
                       else combat.defender_destroyed)
        if not (choices.get('defer_onslaught') and not _own_dead78):
            return

    if is_attacker:
        card = combat.attacker_card
        cancelled = combat.attacker_card_cancelled
    else:
        card = combat.defender_card
        cancelled = combat.defender_card_cancelled

    if not card or cancelled:
        return

    effect = card.combat_effect.lower()

    # Onslaught: self-damage + roll for hits at 4+
    is_onslaught = ("up to four additional hits" in effect and
                    "4+" in effect) or \
                   ("four additional hits" in effect and "score one hit" in effect)
    if is_onslaught:
        army = combat.attacker_army if is_attacker else combat.defender_army
        enemy_army = combat.defender_army if is_attacker else combat.attacker_army

        # rulebook + almanac: "up to four additional hits against your
        # units" -- the cap is on HITS, and an Elite absorbs 2 hits, so
        # the own-side HIT CAPACITY is 2*elites + regulars (Shadow may
        # eliminate its ENTIRE army; the old len-1 keep-one was wrong).
        _e = count_elite_units(army)
        _r = len(army.units) - _e
        hit_capacity = 2 * _e + _r
        max_sacrifice = min(4, hit_capacity)
        if max_sacrifice > 0 and choices.get('defer_onslaught'):
            # How many units to sacrifice is a player choice — defer
            # it to the learned 'combat_effect_choice' pending phase
            # (game0 T3 MT assault: SP sacrificed 4, rolled 1 6 4 3
            # = 2 hits). apply_onslaught_choice() resumes the effect.
            # fix 78 (game-14 canary): the deferral must NOT be gated
            # on the enemy surviving — with spec-forced casualties the
            # onslaught-caused kills land in the main casualty step,
            # so by this point the enemy can already be destroyed while
            # the human's sacrifice record still needs its phase (the
            # sacrificed units really die; extra hits just fizzle).
            combat.deferred_effect = ('attacker' if is_attacker
                                      else 'defender')
            return
        if not is_army_destroyed(enemy_army):
            if max_sacrifice > 0:
                self_casualties = apply_casualties(army, max_sacrifice)
                if is_attacker:
                    combat.attacker_casualties.extend(self_casualties)
                else:
                    combat.defender_casualties.extend(self_casualties)

                # "Roll one die for each HIT you inflicted" -- one die per
                # HIT, NOT per figure removed (an Elite absorbs 2 hits).
                for roll in roll_d6_batch(max_sacrifice, 'sacrifice hits'):
                    if roll >= 4:
                        enemy_casualties = apply_casualties(enemy_army, 1)
                        if is_attacker:
                            combat.defender_casualties.extend(enemy_casualties)
                        else:
                            combat.attacker_casualties.extend(enemy_casualties)

                if is_army_destroyed(army):
                    if is_attacker:
                        combat.attacker_destroyed = True
                    else:
                        combat.defender_destroyed = True
                    combat.battle_over = True
                if is_army_destroyed(enemy_army):
                    if is_attacker:
                        combat.defender_destroyed = True
                    else:
                        combat.attacker_destroyed = True
                    combat.battle_over = True
        return  # Onslaught handled; don't also check We Come to Kill on same card

    # We Come to Kill: extra elite attack at 5+ after casualties
    if "elite" in effect and "additional" in effect and "after removing casualties" in effect:
        army = combat.attacker_army if is_attacker else combat.defender_army
        enemy_army = combat.defender_army if is_attacker else combat.attacker_army

        if not is_army_destroyed(enemy_army):
            elite_count = min(5, count_elite_units(army))
            if elite_count > 0:
                _, hits = combat_roll(elite_count, 5, 0)
                if hits > 0:
                    extra_casualties = apply_casualties(enemy_army, hits)
                    if is_attacker:
                        combat.defender_casualties.extend(extra_casualties)
                        combat.attacker_bonus_hits += hits
                    else:
                        combat.attacker_casualties.extend(extra_casualties)
                        combat.defender_bonus_hits += hits

                if is_army_destroyed(enemy_army):
                    if is_attacker:
                        combat.defender_destroyed = True
                    else:
                        combat.attacker_destroyed = True
                    combat.battle_over = True

    # Great Host (A New Power is Rising / Many Kings / Stormcrow):
    # "score one automatic hit if your Army has at least twice as many
    # units" — evaluated AFTER removing this round's casualties (card
    # text + almanac p43; audit 2026-07-26, moved from the pre-casualty
    # Phase-6 bonus-hits pass where flags 40/41 left it).
    if "twice as many" in effect and "automatic hit" in effect:
        army = combat.attacker_army if is_attacker else combat.defender_army
        enemy_army = combat.defender_army if is_attacker else combat.attacker_army
        if not is_army_destroyed(enemy_army) and \
                len(army.units) >= 2 * len(enemy_army.units):
            extra = apply_casualties(enemy_army, 1)
            if is_attacker:
                combat.defender_casualties.extend(extra)
                combat.attacker_bonus_hits += 1
            else:
                combat.attacker_casualties.extend(extra)
                combat.defender_bonus_hits += 1
            if is_army_destroyed(enemy_army):
                if is_attacker:
                    combat.defender_destroyed = True
                else:
                    combat.attacker_destroyed = True
                combat.battle_over = True


def _apply_post_casualty_effects(combat: CombatState,
                                 choices: dict = None) -> None:
    """Apply post-casualty effects in initiative order (defender first on tie)."""
    if _defender_first_initiative(combat):
        _apply_post_casualty_for_side(combat, is_attacker=False,
                                      choices=choices)
        _apply_post_casualty_for_side(combat, is_attacker=True,
                                      choices=choices)
    else:
        _apply_post_casualty_for_side(combat, is_attacker=True,
                                      choices=choices)
        _apply_post_casualty_for_side(combat, is_attacker=False,
                                      choices=choices)


# ---------------------------------------------------------------------------
# Main Combat Resolution
# ---------------------------------------------------------------------------

# Fidelity-test hook: fn(combat, final_atk_hits, final_def_hits) ->
# (atk, def).  Installed (with save/restore) by rendered_census's
# combat drive so logged human casualties stay authoritative during
# extraction; None in normal play / self-play.
FINAL_HITS_HOOK = None

# apply_casualties_spec below applies an EXACT logged casualty
# composition; live callers: event_effects._route_card_casualties
# (teacher replay) and test_clause_matrix CL-257.


def apply_casualties_spec(army: Army, spec: dict,
                          side: Optional[Side] = None) -> List[Unit]:
    """Forced casualty OUTCOME (closed-loop v5): remove the exact
    composition the log shows — kill_regular regulars, kill_elite
    elites, downgrade elite->regular swaps.  Downgrades reuse the
    _downgraded tag so pool accounting (downgrade regulars return from
    the casualties pool) stays correct.  side filters mixed siege
    entries.  Shortfalls (composition drift) remove what exists and
    leave the rest — the hook notes the delta."""
    casualties = []
    # Freeze the standalone reinforcement baseline BEFORE the kill loops
    # (task 164): a spec that kills regulars then downgrades an elite must
    # not let those kills inflate the apparent pool. In a battle
    # _DG_AVAIL0 already holds the battle-start baseline and wins.
    _av0 = None if _DG_AVAIL0 is not None else _capture_reg_baseline(_ACTIVE_STATE)

    def _own(u):
        return side is None or side_of_nation(u.nation) == side

    _want = {'kill_regular': int(spec.get('kill_regular', 0) or 0),
             'kill_elite': int(spec.get('kill_elite', 0) or 0),
             'downgrade': int(spec.get('downgrade', 0) or 0)}
    _got = {'kill_regular': 0, 'kill_elite': 0, 'downgrade': 0}
    # NATION-EXACT preference (2026-07-25): the parser's kill lines
    # name the nations ('kills S&E Elite ...'); prefer removing those
    # exact (nation, kind) units before the legacy last-in pick
    # (GAME19 T8: the engine killed SAURON where the client killed
    # ISENGARD - fungible counts, wrong nation).
    _byn9 = dict(spec.get('_by_nation') or {})

    def _pick9(kind_enum, kind_str):
        for x in reversed(army.units):
            if x.kind == kind_enum and _own(x):
                _k9 = '%s|%s' % (getattr(x.nation, 'value', x.nation),
                                 kind_str)
                if _byn9.get(_k9, 0) > 0:
                    _byn9[_k9] = _byn9[_k9] - 1
                    return x
        return next((x for x in reversed(army.units)
                     if x.kind == kind_enum and _own(x)), None)

    for _ in range(_want['kill_regular']):
        u = _pick9(UnitKind.REGULAR, 'regular')
        if u is None:
            break
        army.units.remove(u)
        casualties.append(u)
        _got['kill_regular'] += 1
    for _ in range(_want['kill_elite']):
        u = _pick9(UnitKind.ELITE, 'elite')
        if u is None:
            break
        army.units.remove(u)
        casualties.append(u)
        _got['kill_elite'] += 1
    _dgc = _BATTLE_DG if _BATTLE_DG is not None else {}   # battle-scoped when in a battle
    for _ in range(_want['downgrade']):
        u = next((x for x in reversed(army.units)
                  if x.kind == UnitKind.ELITE and _own(x)), None)
        if u is None:
            break
        army.units.remove(u)
        # ONE chokepoint for downgrade sourcing (ruling 2026-07-20d):
        # _consume_downgrade_source decrements the casualty box on a
        # box pull and charges _dgc only for pool pulls — the old
        # check-only gate minted the Regular box-blind.
        if _consume_downgrade_source(_ACTIVE_STATE, u.nation,
                                     _dgc, _av0) is not None:
            army.units.append(Unit(u.nation, UnitKind.REGULAR))
            u._downgraded = True
        casualties.append(u)
        _got['downgrade'] += 1
    return casualties


def resolve_combat_round(combat: CombatState, choices: dict = None) -> None:
    """
    Resolve one round of combat.

    Args:
        combat: The combat state to resolve
        choices: Dict of player choices for interactive effects
    """
    choices = choices or {}
    # the round's FINAL hits (after prevention) - the ONE quantity the
    # casualty allocation and the hit announcement both read; a round
    # that ends before the roll leaves 0
    combat.attacker_final_hits = 0
    combat.defender_final_hits = 0

    # Phase 1: Cancel effects (Swarm of Bats, Daring Defiance)
    apply_cancel_effects(combat)

    # Phase 1b: Words of Power - cancel companion leadership
    apply_words_of_power(combat)

    # Phase 2: Dice modifiers (+1/+2, both armies, subtract)
    apply_dice_modifiers(combat)
    apply_dice_count_modifiers(combat)

    # Phase 2b: Dread and Despair - forfeit Nazgul to reduce FP dice
    apply_dread_and_despair(combat, choices)

    # Phase 2c: Relentless Assault - self-damage for dice boost
    apply_relentless_assault(combat, choices)

    # Phase 3: Pre-combat attacks (Sudden Strike, Charge, Durin's Bane) in
    # initiative order, INTERLEAVED so each side's magnitude reflects the
    # casualties the higher/defender-on-tie side already inflicted, and the
    # battle ends IMMEDIATELY if an Army is eliminated (USER interactions
    # D + J).  If the battle ended here, skip the Combat roll (and its
    # roll-dependent cards, e.g. Confusion).
    _resolve_pre_combat(combat, choices)
    if combat.battle_over:
        return

    # Phase 4: Combat roll
    if combat.is_siege:
        atk_threshold = 6
    elif combat.city_defense and combat.siege_round == 1:
        atk_threshold = 6
    else:
        atk_threshold = 5
    def_threshold = 5

    # Apply confusion tracking
    apply_confusion_effect(combat)

    # Recalculate strength after pre-combat casualties. WoP (USER 1.208i)
    # cancels the chosen Companion's Captain die for THIS round; the drop
    # happens pre-cap inside calc_combat_strength.
    _wop = getattr(combat, '_wop_cancelled', None)
    _wop_atk = getattr(combat, '_wop_cancelled_is_attacker', False)
    atk_strength = calc_combat_strength(
        combat.attacker_army,
        cancelled_captain=(_wop if _wop_atk else None))
    def_strength = calc_combat_strength(
        combat.defender_army,
        cancelled_captain=(None if _wop_atk else _wop))
    # Battle-scoped card reductions (Help Unlooked For) - combat_roll's
    # max(1, n) provides the almanac's minimum-of-one-die floor
    atk_strength -= getattr(combat, 'attacker_dice_reduction', 0)
    def_strength -= getattr(combat, 'defender_dice_reduction', 0)

    # Roll
    # G1: whose dice these are - attacker first, then defender.
    combat.attacker_combat_rolls, atk_hits = combat_roll(
        atk_strength, atk_threshold, combat.attacker_dice_mod, combat.attacker_dice_cap,
        side=getattr(combat, 'attacker_side', None)
    )
    combat.defender_combat_rolls, def_hits = combat_roll(
        def_strength, def_threshold, combat.defender_dice_mod, combat.defender_dice_cap,
        side=getattr(combat, 'defender_side', None)
    )

    # Confusion: 1s hit the roller. CARD-FLEET HIGH-1 (2026-07-20):
    # natural 1s NEVER counted as hits (combat_roll skips them), so
    # the old `hits -= ones` cancelled one LEGITIMATE 5/6 hit per 1
    # rolled — an invented penalty. The card's whole effect is the
    # hits AGAINST the roller (+ the no-reroll exclusion handled in
    # leader_reroll).
    if combat.attacker_confusion:
        ones = sum(1 for r in combat.attacker_combat_rolls if r == 1)
        combat.defender_hits += ones

    if combat.defender_confusion:
        ones = sum(1 for r in combat.defender_combat_rolls if r == 1)
        combat.attacker_hits += ones

    combat.attacker_hits += max(0, atk_hits)
    combat.defender_hits += max(0, def_hits)

    # Phase 5: Leader re-roll
    apply_reroll_cancellation(combat)

    # Phase 5a: They are Terrible - forfeit Nazgul for +1 re-roll mod
    apply_they_are_terrible(combat)

    # Phase 5b: Mighty Attack / Anduril - auto-convert misses before re-roll
    atk_auto = apply_mighty_attack(combat, combat.attacker_combat_rolls,
                                   atk_threshold, combat.attacker_dice_mod,
                                   is_attacker=True)
    combat.attacker_hits += atk_auto

    def_auto = apply_mighty_attack(combat, combat.defender_combat_rolls,
                                   def_threshold, combat.defender_dice_mod,
                                   is_attacker=False)
    combat.defender_hits += def_auto

    # Phase 5c: Actual leader re-roll (with modified leadership and re-roll mods).
    # 1.208h supersedes 1.79: combat-roll bonuses are fully separate and do
    # NOT reach the re-roll. reroll_mod = reroll_mod ONLY; the combat-roll
    # bonus (dice_mod) is passed as miss_modifier to size the miss pool.
    atk_reroll_hits = 0
    if not combat.attacker_reroll_cancelled and combat.attacker_leadership > 0:
        reroll_mod = combat.attacker_reroll_mod
        combat.attacker_rerolls, atk_reroll_hits = leader_reroll(
            combat.attacker_combat_rolls, atk_threshold,
            combat.attacker_leadership, reroll_mod,
            confused=combat.attacker_confusion,
            # miss pool judged at the COMBAT modifier (re-roll-only
            # bonuses must not shrink it)
            miss_modifier=combat.attacker_dice_mod,
            # Mighty Attack / Anduril conversions leave the pool
            exclude_misses=atk_auto,
            side=getattr(combat, 'attacker_side', None)
        )
        combat.attacker_hits += atk_reroll_hits

    def_reroll_hits = 0
    if not combat.defender_reroll_cancelled and combat.defender_leadership > 0:
        reroll_mod = combat.defender_reroll_mod
        combat.defender_rerolls, def_reroll_hits = leader_reroll(
            combat.defender_combat_rolls, def_threshold,
            combat.defender_leadership, reroll_mod,
            confused=combat.defender_confusion,
            miss_modifier=combat.defender_dice_mod,
            exclude_misses=def_auto,
            side=getattr(combat, 'defender_side', None)
        )
        combat.defender_hits += def_reroll_hits

    # Phase 5d: Post-reroll specials that eliminate the ENEMY's leaders —
    # Fateful Strike (-> Nazgul) and Blade of Westernesse (-> Minion). They
    # act on the opposing (Shadow) leader pool, independent of the Black
    # Breath / Heroic Death Free-Peoples-leader window resolved at Phase 7.
    # RULING 1.228 (USER 2026-08-03): this ordering is CORRECT for the
    # tied-initiative-6 Fateful Strike vs Black Breath pairing — the
    # defender-first tie rule is NOT relevant: FS activates on its play
    # condition (the Leader re-roll), which still takes place even if
    # Black Breath eliminates a Leader/Companion at Phase 7, and BB
    # remains playable after FS fires (unless FS killed the sole Nazgul,
    # tripping BB's resolver gate — CL-216).  The adversarial-review
    # defender-first flag on this block is dismissed.
    apply_fateful_strike(combat, atk_reroll_hits, is_attacker=True)
    apply_fateful_strike(combat, def_reroll_hits, is_attacker=False)
    apply_blade_of_westernesse(combat, atk_reroll_hits, is_attacker=True)
    apply_blade_of_westernesse(combat, def_reroll_hits, is_attacker=False)

    # Phase 6: Post-reroll bonus hits (No Quarter, Nameless Wood, Great Host, Mumakil)
    apply_bonus_hits(combat)

    # Phase 7: Casualty prevention (Shield-Wall), then the initiative-6
    # Black Breath (Shadow) vs Heroic Death (Free Peoples) window resolved
    # in initiative order (DEFENDER first on tie) so the defender's card
    # consumes the shared Free-Peoples-leader pool first (USER interaction
    # H).  Both trigger "before casualties are removed".
    apply_casualty_prevention(combat, choices)
    _apply_bb_hd(combat, atk_reroll_hits, def_reroll_hits, choices)

    # Calculate final hits (pre-combat hits already applied as casualties above,
    # so do NOT add them again here)
    final_atk_hits = (combat.attacker_hits +
                      combat.attacker_bonus_hits - combat.defender_hits_cancelled)
    final_def_hits = (combat.defender_hits +
                      combat.defender_bonus_hits - combat.attacker_hits_cancelled)

    final_atk_hits = max(0, final_atk_hits)
    final_def_hits = max(0, final_def_hits)

    # Fidelity-test hook (rendered_census combat drive): allows a
    # per-battle cap on applied hits when a replayed log shows the humans
    # applied fewer (logged rules deviations).  None in normal play — no
    # behavior change.
    if FINAL_HITS_HOOK is not None:
        final_atk_hits, final_def_hits = FINAL_HITS_HOOK(
            combat, final_atk_hits, final_def_hits)
    combat.attacker_final_hits = final_atk_hits
    combat.defender_final_hits = final_def_hits

    # POST-ROLL allocation (USER 2026-07-09): stash the hits and stop —
    # the owner CHOOSES the allocation after seeing the roll (ai.py
    # pending 'casualty_allocation'); armies stay untouched here so the
    # siege/away merge-backs write them back unchanged.
    if (choices or {}).get('defer_casualties'):
        combat.casualties_deferred = True
        combat.deferred_hits_on_defender = final_atk_hits
        combat.deferred_hits_on_attacker = final_def_hits
        combat.battle_over = True
        return

    # Phase 8: Apply casualties (owner-chosen allocation policy).
    _ch = choices or {}
    combat.defender_casualties.extend(
        apply_casualties(combat.defender_army, final_atk_hits,
                         _ch.get('def_casualty_policy',
                                 'preserve_elites'))
    )
    combat.attacker_casualties.extend(
        apply_casualties(combat.attacker_army, final_def_hits,
                         _ch.get('atk_casualty_policy',
                                 'preserve_elites'))
    )

    # Check for destruction
    if is_army_destroyed(combat.attacker_army):
        combat.attacker_destroyed = True
        combat.battle_over = True
    if is_army_destroyed(combat.defender_army):
        combat.defender_destroyed = True
        combat.battle_over = True

    # Phase 9: Post-casualty effects (Onslaught, We Come to Kill)
    # Resolve in initiative order; defender first on tie. A choice-bearing
    # Onslaught is deferred to the learned pending phase when the caller
    # passes 'defer_onslaught' (see combat.deferred_effect).
    _apply_post_casualty_effects(combat, choices)


def siege_press(army: Army, prefer_nation=None) -> bool:
    """
    Attempt to "press" a siege by downgrading an elite to a regular.

    Per rulebook: after each siege round, the attacker can extend
    the siege by 1 additional round by reducing an elite unit to
    a regular unit. Returns True if press succeeded.

    prefer_nation (fix 76, GAME19 T8): the elite the human paid with is
    log evidence (parser stamps the press record with its nation); pick
    that nation's elite first — first-in-list order silently paid with
    the wrong nation whenever the army mixed elites.

    Replacement-Regular sourcing (F-8 / USER RULING 2026-07-20d, same
    discipline as the learned press path in ai.py): CASUALTY BOX first,
    else pool, else the elite dies outright — the replacement is never
    minted.  FP downgraded elites are booked as permanent casualties.
    """
    def _press_at9(i, unit):
        st = _ACTIVE_STATE
        if st is None:
            army.units[i] = Unit(unit.nation, UnitKind.REGULAR)
            return True
        _dgc9 = _BATTLE_DG if _BATTLE_DG is not None else {}
        src = _consume_downgrade_source(st, unit.nation, _dgc9)
        if src is None:
            # no replacement source: the elite dies outright
            dead = army.units.pop(i)
            _track_fp_casualties(st, [dead],
                                 side_of_nation(dead.nation))
            return True
        army.units[i] = Unit(unit.nation, UnitKind.REGULAR)
        if side_of_nation(unit.nation) == Side.FREE_PEOPLES:
            # the paid elite is a permanent FP casualty
            _track_fp_casualties(st, [unit], Side.FREE_PEOPLES)
        return True

    if prefer_nation is not None:
        _pn = str(getattr(prefer_nation, 'value', prefer_nation)).lower()
        for i, unit in enumerate(army.units):
            if (unit.kind == UnitKind.ELITE
                    and str(getattr(unit.nation, 'value',
                                    unit.nation)).lower() == _pn):
                return _press_at9(i, unit)
    for i, unit in enumerate(army.units):
        if unit.kind == UnitKind.ELITE:
            return _press_at9(i, unit)
    return False


def resolve_combat(state: "GameState", attacker_region: str, defender_region: str,
                   attacker_card_id: Optional[str] = None,
                   defender_card_id: Optional[str] = None,
                   is_siege: bool = False,
                   max_rounds: int = 1,
                   auto_press: bool = False,
                   choices: dict = None) -> CombatResult:
    """
    Resolve a complete battle between two regions.

    Args:
        state: The game state
        attacker_region: Region attacking from
        defender_region: Region being attacked
        attacker_card_id: Card ID for attacker's combat card
        defender_card_id: Card ID for defender's combat card
        is_siege: Whether this is a siege assault
        max_rounds: Maximum rounds (usually 1; cards like Grond add extra)
        auto_press: If True and is_siege, auto-press by downgrading elites
        choices: Player choices for interactive effects

    Returns:
        CombatResult with battle outcome
    """
    from cards import get_card

    # BATTLE-SCOPED downgrade counter (2026-07-20, Feanor teacher-path
    # mint): a multi-round battle operates on TEMP split-armies not
    # written back to state.armies between rounds, so _downgrade_allowed
    # reads STALE pre-battle deployed counts all battle long and the
    # per-apply_casualties-call _dgc resets each round -> more downgrades
    # than the pre-battle reinforcement pool supplies. A single counter
    # shared across every apply_casualties call in THIS battle fixes it
    # (avail0 - battle_downgrades = true remaining reinforcements; kills
    # don't change reinforcement availability). Save/restore for nesting.
    global _BATTLE_DG, _DG_AVAIL0
    _prev_bdg = _BATTLE_DG
    _prev_av0 = _DG_AVAIL0
    _BATTLE_DG = {}
    # Freeze the reinforcement baseline for the WHOLE battle BEFORE any
    # round removes a unit (task 164, Feanor/Hynch teacher replay). A
    # regular KILLED mid-battle vacates a deployment slot but becomes a
    # PERMANENT casualty — NOT free reinforcement capacity. Without this
    # snapshot _downgrade_allowed read a LIVE deployed count deflated by
    # those same-scope kills (Dale: 3 Elf regulars removed first by the
    # preserve_elites policy), computed avail=3 where the true pool was 0,
    # and green-lit a downgrade that MINTED a 6th Elf regular. Downgrades
    # are the ONLY legitimate consumer of this baseline (tracked via
    # _BATTLE_DG / already_consumed). Save/restore for nesting (a card
    # handler whose battle triggers another battle).
    _DG_AVAIL0 = _capture_reg_baseline(state)
    try:
        return _resolve_combat_body(
            state, attacker_region, defender_region,
            attacker_card_id, defender_card_id, is_siege, max_rounds,
            auto_press, choices, get_card)
    finally:
        _BATTLE_DG = _prev_bdg
        _DG_AVAIL0 = _prev_av0


_BATTLE_DG = None
_DG_AVAIL0 = None


def _resolve_combat_body(state, attacker_region, defender_region,
                         attacker_card_id, defender_card_id, is_siege,
                         max_rounds, auto_press, choices, get_card):
    # Load combat cards
    atk_card = None
    def_card = None

    if attacker_card_id:
        card = get_card(attacker_card_id)
        if card:
            atk_card = card.to_combat_card()

    if defender_card_id:
        card = get_card(defender_card_id)
        if card:
            def_card = card.to_combat_card()

    # Setup combat
    combat = setup_combat(state, attacker_region, defender_region,
                          atk_card, def_card, is_siege,
                          attacker_side=(choices or {}).get(
                              'attacker_side'))
    # Battle-scoped dice reductions (Help Unlooked For, QA 2026-07-09):
    # "reduced by one die per FP unit inside the Stronghold, min 1, for
    # this ENTIRE battle" - applied at the roll site every round, never
    # reset by the per-round state reset.
    combat.attacker_dice_reduction = int(
        (choices or {}).get('attacker_dice_reduction', 0) or 0)
    combat.defender_dice_reduction = int(
        (choices or {}).get('defender_dice_reduction', 0) or 0)

    # After round 1, city/fortification defense no longer applies
    if choices and choices.get('disable_city_defense'):
        combat.city_defense = False

    total_atk_hits = 0
    total_def_hits = 0
    rounds_fought = 0

    # For siege: base is 1 round, +extra from cards (max_rounds), +presses
    effective_max = max_rounds

    # Resolve rounds
    while not combat.battle_over and combat.siege_round <= effective_max:
        resolve_combat_round(combat, choices)
        # iter 38 (user directive): combat rolls + leader re-rolls become
        # visible annotation rows in SELF-PLAY traces (no-op otherwise)
        try:
            from game_trace import emit_annotation
            _ctx = (f"{'siege ' if is_siege else ''}battle at "
                    f"{defender_region} r{combat.siege_round}")
            _atk_tag = 'SH' if combat.attacker_side.name == 'SHADOW' \
                else 'FP'
            _def_tag = 'FP' if _atk_tag == 'SH' else 'SH'
            if combat.attacker_combat_rolls:
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.attacker_combat_rolls),
                    'ctx': f'{_ctx} attacker', 'side': _atk_tag})
            if combat.attacker_rerolls:
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.attacker_rerolls),
                    'ctx': f'{_ctx} attacker re-roll', 'side': _atk_tag})
            if combat.defender_combat_rolls:
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.defender_combat_rolls),
                    'ctx': f'{_ctx} defender', 'side': _def_tag})
            if combat.defender_rerolls:
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.defender_rerolls),
                    'ctx': f'{_ctx} defender re-roll', 'side': _def_tag})
            # Pre-combat attack rolls (Sudden Strike / Charge / Durin's
            # Bane) — previously rolled and discarded, invisible in traces
            # (coordinator item 2026-07-15).
            if combat.attacker_pre_combat_rolls:
                _acn = getattr(combat.attacker_card, 'combat_name',
                               'pre-combat') or 'pre-combat'
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.attacker_pre_combat_rolls),
                    'ctx': f'{_acn} pre-combat attack', 'side': _atk_tag})
            if combat.defender_pre_combat_rolls:
                _dcn = getattr(combat.defender_card, 'combat_name',
                               'pre-combat') or 'pre-combat'
                emit_annotation(state, 'dice_roll', {
                    'faces': list(combat.defender_pre_combat_rolls),
                    'ctx': f'{_dcn} pre-combat attack', 'side': _def_tag})
        except Exception:
            pass
        # the totals are the hits that REACH casualties (after Shield-
        # Wall / Heroic Death / Black Breath prevention): what the
        # allocation asks for and what the hit announcement states.
        # Pre-combat additional-attack hits are applied as casualties
        # inside the round (not allocated) and are not counted here.
        total_atk_hits += int(getattr(combat, 'attacker_final_hits', 0))
        total_def_hits += int(getattr(combat, 'defender_final_hits', 0))
        rounds_fought += 1
        combat.siege_round += 1

        # Siege press: after each round, attacker can extend by downgrading elite
        if (is_siege and not combat.battle_over
                and combat.siege_round > effective_max):
            should_press = False
            if auto_press:
                # AI auto-presses if it has elites and defender survives
                should_press = count_elite_units(combat.attacker_army) > 0
            elif choices and choices.get('press_siege'):
                should_press = True

            if should_press and siege_press(combat.attacker_army):
                effective_max += 1
                # Recalculate strength after downgrade
                combat.attacker_combat_strength = calc_combat_strength(combat.attacker_army)

        # Reset round-specific state for multi-round battles
        if not combat.battle_over and combat.siege_round <= effective_max:
            # per-round card effects (Brave Stand's reduction) must not
            # leak into rounds 2+; the battle-scoped Help-Unlooked-For
            # baseline from `choices` is re-seeded, never lost
            combat.attacker_dice_reduction = int(
                (choices or {}).get('attacker_dice_reduction', 0) or 0)
            combat.defender_dice_reduction = int(
                (choices or {}).get('defender_dice_reduction', 0) or 0)
            combat.attacker_hits = 0
            combat.defender_hits = 0
            combat.attacker_pre_combat_hits = 0
            combat.defender_pre_combat_hits = 0
            combat.attacker_bonus_hits = 0
            combat.defender_bonus_hits = 0
            combat.attacker_hits_cancelled = 0
            combat.defender_hits_cancelled = 0
            combat.attacker_combat_rolls = []
            combat.defender_combat_rolls = []
            combat.attacker_rerolls = []
            combat.defender_rerolls = []
            combat.attacker_pre_combat_rolls = []
            combat.defender_pre_combat_rolls = []
            combat.attacker_dice_mod = 0
            combat.defender_dice_mod = 0
            combat.attacker_reroll_mod = 0
            combat.defender_reroll_mod = 0
            combat.attacker_dice_cap = None
            combat.defender_dice_cap = None
            combat.attacker_reroll_cancelled = False
            combat.defender_reroll_cancelled = False
            combat.attacker_confusion = False
            combat.defender_confusion = False
            # Clear combat cards after each round — per rulebook, cards are
            # played per-round and discarded. Prevents re-applying pre-combat
            # attacks, Daring Defiance, etc. in subsequent rounds.
            combat.attacker_card = None
            combat.defender_card = None
            combat.attacker_card_cancelled = False
            combat.defender_card_cancelled = False
            # USER 1.208i/j per-round markers; the battle-scoped
            # _shine_*/_shined_companion bookkeeping persists.
            combat._wop_cancelled = None
            combat._wop_cancelled_is_attacker = False
            combat._foul_stench_cancel_atk = False
            combat._foul_stench_cancel_def = False
            # Recalculate combat strength and leadership for next round
            combat.attacker_combat_strength = calc_combat_strength(combat.attacker_army)
            combat.defender_combat_strength = calc_combat_strength(combat.defender_army)

    # Build result
    result = CombatResult(
        attacker_region=attacker_region,
        defender_region=defender_region,
        rounds_fought=rounds_fought,
        attacker_total_hits=total_atk_hits,
        defender_total_hits=total_def_hits,
        attacker_casualties=combat.attacker_casualties,
        defender_casualties=combat.defender_casualties,
        attacker_destroyed=combat.attacker_destroyed,
        defender_destroyed=combat.defender_destroyed,
        attacker_retreated=combat.attacker_retreated,
        defender_retreated=combat.defender_retreated,
        siege_continues=is_siege and not combat.battle_over,
        deferred_effect=combat.deferred_effect,
    )
    # post-roll allocation flags ride the result wrapper (ai.py routes
    # them into the pending 'casualty_allocation' phase)
    result.casualties_deferred = getattr(combat, 'casualties_deferred',
                                         False)
    result.deferred_hits_on_defender = getattr(
        combat, 'deferred_hits_on_defender', 0)
    result.deferred_hits_on_attacker = getattr(
        combat, 'deferred_hits_on_attacker', 0)
    # cancellation state rides the result so the deferred-allocation
    # caller can resume Phase 9 with the round's true card status
    result.attacker_card_cancelled = combat.attacker_card_cancelled
    result.defender_card_cancelled = combat.defender_card_cancelled

    # Track FP permanent casualties (rulebook p.30: FP casualties never
    # return) — but do it AFTER the survivors are written back to
    # state.armies, at each exit below. In a SPLIT battle (co-located
    # siege / away-attack) the fighting army is a TEMP copy not yet merged
    # back, so booking casualties here left state.armies showing the STALE
    # pre-battle deployed count: deployed + casualties momentarily crossed
    # max at the tracking site even though the merge-back immediately
    # reconciles it (Giampy/Hynch lorien siege — a real Elf regular dies,
    # dep 3->2 cas 2->3, but _track ran while dep still read 3 => transient
    # 6/5). The conservation tripwire wraps _track, so that ordering
    # artifact read as a mint. Deferring to post-write-back keeps deployed
    # live at booking time (field battles are unaffected: their combat army
    # IS the live state.armies entry). Call EXACTLY once per exit path.
    def _publish_fp_casualties():
        _track_fp_casualties(state, combat.attacker_casualties,
                             combat.attacker_side)
        _track_fp_casualties(state, combat.defender_casualties,
                             combat.defender_side)
        # 5-unit siege-cap removals are NOT casualties: USER 1.181
        # house ruling (2026-07-20, pinned CL-147) — the excess
        # returns to reinforcements, and an untracked removal IS the
        # return (reinforcements = max - deployed - fp_casualties).

    # Update game state - remove destroyed armies
    # Find the original army key matching each region (state.armies uses original names)
    def _find_army_key(region_name):
        san = sanitize(region_name)
        # Try original name first
        if region_name in state.armies:
            return region_name
        # Try sanitized match
        for k in state.armies:
            if sanitize(k) == san:
                return k
        return None

    _parent = getattr(combat, 'coloc_parent', None)
    if _parent is not None:
        # Co-located siege: write the survivors of both virtual armies
        # back into the shared army entry (never pop it unless empty).
        # A destroyed side's CHARACTERS are eliminated with the army
        # (USER RULING 2026-08-15, eliminate_army_characters) — they
        # previously survived in the shared entry, leaving phantom enemy
        # leaders inside captured strongholds.
        if combat.defender_destroyed:
            _, def_leaders = eliminate_army_characters(
                state, combat.defender_army.leaders)
        else:
            def_leaders = list(combat.defender_army.leaders)
        if combat.attacker_destroyed:
            _, atk_leaders = eliminate_army_characters(
                state, combat.attacker_army.leaders)
        else:
            atk_leaders = list(combat.attacker_army.leaders)
        _parent.units = list(combat.defender_army.units) + \
            list(combat.attacker_army.units)
        _parent.leaders = def_leaders + atk_leaders
        if not _parent.units and not _parent.leaders:
            dkey = _find_army_key(defender_region)
            if dkey:
                state.armies.pop(dkey, None)
        if combat.defender_destroyed and not combat.attacker_destroyed:
            # 1.152 (USER): mutual elimination -> original controller
            # keeps the region; capture needs a surviving attacker
            _check_settlement_capture(state, sanitize(defender_region),
                                      combat.attacker_side)
        _publish_fp_casualties()   # after the shared-entry merge-back
        return result

    # Away-attack splits: write the fighting portion's survivors back
    # into the shared (mixed) entry; the bystanders never left.
    _aap = getattr(combat, 'away_atk_parent', None)
    if _aap is not None:
        _ru, _rl = combat.away_atk_rest
        if combat.attacker_destroyed:
            _, _sl = eliminate_army_characters(
                state, combat.attacker_army.leaders)
            _su = []
        else:
            _su = list(combat.attacker_army.units)
            _sl = list(combat.attacker_army.leaders)
        _aap.units = _su + _ru
        _aap.leaders = _sl + _rl
        if not _aap.units and not _aap.leaders:
            akey = _find_army_key(attacker_region)
            if akey:
                state.armies.pop(akey, None)
    _adp = getattr(combat, 'away_def_parent', None)
    if _adp is not None:
        _ru, _rl = combat.away_def_rest
        if combat.defender_destroyed:
            _, _sl = eliminate_army_characters(
                state, combat.defender_army.leaders)
            _su = []
        else:
            _su = list(combat.defender_army.units)
            _sl = list(combat.defender_army.leaders)
        _adp.units = _su + _ru
        _adp.leaders = _sl + _rl
        if not _adp.units and not _adp.leaders:
            dkey = _find_army_key(defender_region)
            if dkey:
                state.armies.pop(dkey, None)
        # no settlement capture here: with bystanders present the
        # region is not cleared (relief attack vs the besiegers)

    # Normal (unsplit) battle: the destroyed side's characters die with
    # it through the SAME reconciler (USER RULING 2026-08-15).  Only the
    # dying side's figures are taken — a co-located ENEMY character
    # (e.g. a separated Companion in a region a Shadow army occupies)
    # is not collateral, and the entry is dropped only once empty.
    if combat.attacker_destroyed and _aap is None:
        akey = _find_army_key(attacker_region)
        if akey:
            reconcile_destroyed_army(state, akey, combat.attacker_side)
    if combat.defender_destroyed and _adp is None:
        dkey = _find_army_key(defender_region)
        if dkey:
            reconcile_destroyed_army(state, dkey, combat.defender_side)
        # Check for settlement capture (for VP tracking).  1.152:
        # mutual elimination captures nothing - the attacker must live
        if not combat.attacker_destroyed:
            _check_settlement_capture(state, sanitize(defender_region),
                                      combat.attacker_side)

    _publish_fp_casualties()   # after away-split merge-back / destroyed pops
    return result


def _handle_eliminated_minions(state: "GameState", leaders) -> None:
    """Character-sheet bookkeeping for Minion figures eliminated with a
    destroyed army (e.g. Saruman when Orthanc falls — he never leaves it):
    flip the matching ShadowMinion out of play and fire elimination
    discard triggers (Saruman -> The Palantír of Orthanc / Wormtongue via
    check_saruman_eliminated_discards)."""
    from game_logic import Minion, _handle_saruman_elimination_discards
    for ld in leaders:
        if not isinstance(ld, Minion):
            continue
        nm = str(getattr(ld, 'name', '')).lower()
        if 'saruman' in nm:
            if getattr(state, 'saruman', None) is not None:
                state.saruman.in_play = False
            _handle_saruman_elimination_discards(state)
        elif 'witch' in nm:
            if getattr(state, 'witch_king', None) is not None:
                state.witch_king.in_play = False
        elif 'mouth' in nm:
            if getattr(state, 'mouth_of_sauron', None) is not None:
                state.mouth_of_sauron.in_play = False



# OVERSIGHT #2 (2026-07-20): downgrade replacements must come from a
# real supply slot — the four inline downgrade sites minted regulars
# pool-blind. game_logic.battle() publishes the live GameState here;
# _downgrade_allowed answers "does a replacement regular exist for
# this nation" (FP: casualties box OR computed pool; SH: pool).
# With no active state (standalone/unit use) downgrades are allowed.
_ACTIVE_STATE = None


def set_active_state(state):
    """Publish the live GameState for _downgrade_allowed's supply
    check. Returns the PREVIOUS value so callers can save/restore
    (nesting: a card handler may trigger battle(), which must not
    clear the handler's published state on exit)."""
    global _ACTIVE_STATE
    _prev = _ACTIVE_STATE
    _ACTIVE_STATE = state
    return _prev


def _live_reg_avail(state, nation) -> int:
    """Regulars currently free in REINFORCEMENTS for `nation`
    (max - deployed - permanent FP casualties).  POOL ONLY BY DESIGN:
    this counts reinforcements; the CASUALTY-BOX leg — which sources
    downgrades FIRST (USER 2026-07-20d: "the downgrade is the ONLY
    way casualties return") — is checked separately and live in
    _downgrade_source_of / _cas_box_regulars."""
    # ONE ANSWERER (USER ruling 2026-08-18, strict component limits):
    # ai.figures_available carries this arithmetic now.  It used to be
    # copied here, in _capture_reg_baseline, in ai._can_recruit, in
    # game_logic._clamp_place and inline in two card paths - six
    # answerers to one question.
    import ai as _ai9
    return _ai9.figures_available(state, nation, 'regular')


def _capture_reg_baseline(state) -> dict:
    """Snapshot per-nation regular reinforcement availability for the
    whole upcoming casualty scope (a battle, or one standalone
    apply_casualties* call). Captured BEFORE any unit is removed so that
    later same-scope KILLS — which vacate a deployment slot but only
    become permanent casualties at write-back — cannot masquerade as free
    reinforcement capacity and green-light a mint. Downgrades are the ONLY
    legitimate consumer, decremented via already_consumed / _BATTLE_DG."""
    base = {}
    if state is None:
        return base
    import ai as _ai9
    dep_all = _ai9._count_deployed_all(state)
    for nation in _ai9.NATION_MAX_FIGURES:
        base[nation] = _ai9.figures_available(state, nation, 'regular',
                                              dep_all)
    return base


def _downgrade_allowed(nation, already_consumed=0, avail_override=None) -> bool:
    """Is a replacement Regular available to reduce an Elite of
    `nation` (rulebook p.30)?

    USER RULING 2026-07-20d: the replacement may come from the
    REINFORCEMENT pool OR be pulled back from the CASUALTY box — the
    downgrade is the ONLY way casualties return (bar Gandalf the
    White). Box-sourcing is conservation-clean by construction (the
    figure physically exists in the box: box -1 / deployed +1), so the
    box is checked LIVE; only the pool needs the scope-start baseline
    discipline below.

    Pool availability is read from a SCOPE-START baseline, never a live
    deployed count, so mid-scope events cannot distort it:
    - `_DG_AVAIL0` (battle-wide, resolve_combat) takes precedence — it is
      frozen before round 1 and shared across every round/apply call.
    - else `avail_override` (a per-call baseline the standalone
      apply_casualties* caller froze at its own entry).
    - else the live pool (no scope published, e.g. direct unit use).
    `already_consumed`: Regulars already minted-by-downgrade EARLIER in
    this scope (pool-sourced ones only — box pulls decrement the box
    immediately)."""
    st = _ACTIVE_STATE
    if st is None:
        return True
    try:
        if _cas_box_regulars(st, nation) > 0:
            return True
        if _DG_AVAIL0 is not None and nation in _DG_AVAIL0:
            avail = _DG_AVAIL0[nation]
        elif avail_override is not None and nation in avail_override:
            avail = avail_override[nation]
        else:
            avail = _live_reg_avail(st, nation)
        return (avail - already_consumed) > 0
    except Exception as _dg_err9:
        # permissive on a supply-computation failure, but LOUD — a
        # silent True here disables the whole mint-prevention discipline
        print('[combat] downgrade supply check failed for %s: %r'
              % (getattr(nation, 'value', nation), _dg_err9))
        return True


def _cas_box_regulars(state, nation) -> int:
    """Live count of `nation` Regulars in the FP casualty box (Shadow
    has no box — its casualties return to the pool)."""
    if state is None or side_of_nation(nation) != Side.FREE_PEOPLES:
        return 0
    return ((getattr(state, 'fp_casualties', None) or {})
            .get(nation, {}) or {}).get('regular', 0)


def _downgrade_source_of(state, nation, dgc,
                         avail_override=None) -> Optional[str]:
    """THE replacement-Regular source rule, read-only: which source
    would the next downgrade of `nation` draw on?  BOX FIRST (USER
    2026-07-20d: recovering a dead figure dominates - the pool Regular
    stays musterable), else pool (counted against the scope baseline
    via `dgc`), else None (no replacement: the elite is ELIMINATED
    outright, hit absorbed - USER ruling 2026-08-18).  Returns 'cas' |
    'pool' | None.  _consume_downgrade_source applies the answer."""
    if state is None:
        return 'pool'
    if _cas_box_regulars(state, nation) > 0:
        return 'cas'
    try:
        if _DG_AVAIL0 is not None and nation in _DG_AVAIL0:
            avail = _DG_AVAIL0[nation]
        elif avail_override is not None and nation in avail_override:
            avail = avail_override[nation]
        else:
            avail = _live_reg_avail(state, nation)
    except Exception as _dg_err9:
        # permissive-consistent with _downgrade_allowed's fallback
        # (avail covers everything consumed so far + this one), and
        # LOUD — silently flipping downgrade outcomes hid real breaks
        print('[combat] downgrade source resolve failed for %s: %r'
              % (getattr(nation, 'value', nation), _dg_err9))
        avail = (dgc.get(nation, 0) if dgc is not None else 0) + 1
    consumed = dgc.get(nation, 0) if dgc is not None else 0
    if avail - consumed > 0:
        return 'pool'
    return None


def _consume_downgrade_source(state, nation, dgc,
                              avail_override=None) -> Optional[str]:
    """Consume the replacement-Regular source for one downgrade (the
    ONE answerer every downgrade path routes through: the battle
    press, ai._apply_allocation, the siege_press executor).  Source
    per _downgrade_source_of; the box decrement / pool consumption and
    the _dg_src_log stamp happen HERE.  Returns 'cas' | 'pool' | None
    (None = no replacement: the caller ELIMINATES the elite)."""
    src = _downgrade_source_of(state, nation, dgc, avail_override)
    if state is None:
        if dgc is not None:
            dgc[nation] = dgc.get(nation, 0) + 1
        return 'pool'
    if src == 'cas':
        state.fp_casualties[nation]['regular'] -= 1
    elif src == 'pool':
        if dgc is not None:
            dgc[nation] = dgc.get(nation, 0) + 1
    _dg_stamp(state, nation, src)
    return src


def _dg_stamp(state, nation, src):
    """AUTO-96 game-4 (2026-08-17, the physical-supply phantom):
    the engine's downgrade-source decision (cas -> pool -> None) is
    THE answer; the emitter re-implemented it as a blind zone-order
    pick ('fp dead' first, any nation) and, when they disagreed
    (engine 'pool', emitter picked a dead-pile piece), the client
    pool ran one HIGH forever - pool fp rohan/regular engine=4
    client=5 PHANTOM.  Every consumption stamps this ledger; the
    AUTO server threads the entries into the emission params so the
    emitter OBEYS the engine instead of guessing.  One rule body,
    one witness."""
    if state is None:
        return
    log = getattr(state, '_dg_src_log', None)
    if log is None:
        log = state._dg_src_log = []
    log.append((nation, src))


def _track_fp_leader_deaths(state, leaders):
    """OVERSIGHT #3 (2026-07-20): FP plain-Leader deaths were never
    tracked — dead leaders silently returned to the muster pool
    (rulebook p.30: FP losses are permanent)."""
    if state is None:
        return
    for ld in leaders or []:
        if type(ld).__name__ != 'Leader':
            continue
        nat = getattr(ld, 'nation', None)
        if nat is None or side_of_nation(nat) != Side.FREE_PEOPLES:
            continue
        if not hasattr(state, 'fp_casualties') or state.fp_casualties is None:
            state.fp_casualties = {}
        state.fp_casualties.setdefault(
            nat, {'regular': 0, 'elite': 0, 'leader': 0})
        state.fp_casualties[nat]['leader'] =             state.fp_casualties[nat].get('leader', 0) + 1


# ---------------------------------------------------------------------------
# ARMY-DESTRUCTION CHARACTER RECONCILER   (USER RULING 2026-08-15)
# ---------------------------------------------------------------------------
# "if the army is wiped out, leaders/minions die with it.  the only
#  exception is if the army dies due to spirit of mordor or dreadful
#  spells, in which case companions/nazgul/minions survive, but free
#  people leaders die."
#
# ONE implementation, invoked from EVERY path where an army's units
# reach zero: battle (field / siege / sortie / co-located shared entry /
# away-attack split / deferred-onslaught resume) and card effects
# (Spirit of Mordor, Dreadful Spells, Dead Men of Dunharrow, The Ents
# Awake, Faramir's Rangers, Return to Valinor, ...).
#
# The rule fires on a DESTRUCTION EVENT, never on the mere absence of
# units: a separated Companion (game_logic.place_separated_companion
# builds a leaders-only army), a lone Nazgul and a lone named Minion are
# all LEGAL standing figures.  That is why game_logic.
# remove_orphaned_leaders -- the standing p.8 sweep ("a Leader on the
# map without an Army is immediately removed") -- keeps them and must
# stay exactly as it is: it is the NON-destruction cleanup, this is the
# destruction reconciler.

# Lower-cased card names whose casualties spare the characters.
# USER RULINGS 2026-08-15: army destruction kills the characters
# stacked with it, EXCEPT these cards.  First ruling: Spirit of Mordor
# + Dreadful Spells spare companions/nazgul/minions (plain FP Leaders
# still die by p.8).  Case-by-case follow-up: "Dead men of dunharrow
# and ents kill nazgul/minions. Return to valinor and faramir's
# rangers does not" -> the latter two join the sparing set; Dead Men
# and The Ents Awake stay OUT (their wipes kill characters).
CHARACTER_SPARING_CARDS = frozenset({
    'the spirit of mordor',
    'dreadful spells',
    'return to valinor',
    "faramir's rangers",
})

_WIPE_CARD_CONTEXT = None


def set_wipe_card_context(name):
    """Publish the event card whose casualties are being applied so any
    wipe reconciled underneath it takes the card exception.  Returns the
    PREVIOUS value so callers can save/restore (same nesting contract as
    set_active_state).  Paths that resume ACROSS a pending decision
    (self-play card_casualty allocation) cannot rely on a module context
    and pass card_name= explicitly instead."""
    global _WIPE_CARD_CONTEXT
    prev = _WIPE_CARD_CONTEXT
    _WIPE_CARD_CONTEXT = name
    return prev


def wipe_spares_characters(card_name=None) -> bool:
    """True when this destruction was caused by one of the
    CHARACTER_SPARING_CARDS (Spirit of Mordor, Dreadful Spells, Return
    to Valinor, Faramir's Rangers - USER rulings 2026-08-15, first
    ruling + case-by-case follow-up).  card_name=None falls back to
    the published context (None outside those cards, i.e. battle
    rules: everybody dies)."""
    nm = card_name if card_name is not None else _WIPE_CARD_CONTEXT
    if not nm:
        return False
    return str(nm).strip().lower() in CHARACTER_SPARING_CARDS


def _is_plain_leader(ld) -> bool:
    """A plain Army Leader figure -- NOT a Nazgul / Minion / Companion
    (Nazgul and Minion both subclass Leader, so this is an EXACT type
    test, matching _track_fp_leader_deaths)."""
    return type(ld).__name__ == 'Leader'


def _handle_eliminated_companions(state: "GameState", leaders) -> None:
    """Character-sheet bookkeeping for Companion figures eliminated with
    a destroyed army.  Each is a real CASUALTY, so 'Worn with Sorrow and
    Toil' and the companion persistent-card discards fire exactly as
    they do for a hunt casualty (game_logic._on_companion_casualty; it
    self-defers the learned WWSAT source pick onto state._wwsat_owed, so
    it is safe to call mid-battle)."""
    if state is None:
        return
    from game_state import Companion
    import game_logic as _glec
    for ld in leaders or []:
        if not isinstance(ld, Companion):
            continue
        try:
            _glec._on_companion_casualty(state, ld.name)
        except Exception as _cc_err9:
            # LOUD but non-fatal: the figure is still eliminated
            print('[combat] companion casualty bookkeeping failed for '
                  '%s: %r' % (getattr(ld, 'name', '?'), _cc_err9))


def classify_wipe_characters(leaders, dying_side=None, card_name=None):
    """Split `leaders` into (dead, survivors) for a DESTROYED army.

    dying_side: only figures on that side are candidates (a shared
    besieged entry holds BOTH sides, and an enemy Companion may stand in
    a region an army occupies).  None = every figure in the list belongs
    to the destroyed army.
    """
    spare = wipe_spares_characters(card_name)
    dead, survivors = [], []
    for ld in leaders or []:
        if dying_side is not None and _combat_leader_side(ld) != dying_side:
            survivors.append(ld)
            continue
        if spare and not _is_plain_leader(ld):
            # CHARACTER_SPARING_CARDS (SoM / Dreadful Spells / Return
            # to Valinor / Faramir's Rangers): Companions, Nazgul and
            # named Minions SURVIVE; only plain Leaders die (they may
            # never stand without an Army, rulebook p.8).
            survivors.append(ld)
            continue
        dead.append(ld)
    return dead, survivors


def book_eliminated_characters(state: "GameState", dead) -> None:
    """Send every eliminated figure to its proper home: named Minions
    flip out of play (+ Saruman's table-card discards), FP plain Leaders
    book as PERMANENT casualties, Companions fire the companion-casualty
    triggers.  Nazgul return to the Shadow pool by convention (Shadow
    figures are pool-recycled, so there is nothing to record)."""
    if not dead:
        return
    _handle_eliminated_minions(state, dead)
    _track_fp_leader_deaths(state, dead)
    _handle_eliminated_companions(state, dead)


def eliminate_army_characters(state: "GameState", leaders, dying_side=None,
                              *, card_name=None):
    """Classify + book a destroyed army's characters.  Returns
    (dead, survivors); the caller writes `survivors` back."""
    dead, survivors = classify_wipe_characters(leaders, dying_side, card_name)
    book_eliminated_characters(state, dead)
    return dead, survivors


def reconcile_destroyed_army(state: "GameState", region_key, dying_side=None,
                             *, card_name=None, army=None):
    """THE post-destruction reconciler: an army at `region_key` has just
    had its units reduced to zero -- eliminate the characters that stood
    with it per the ruling, write the survivors back, and drop the entry
    once nothing is left.  Returns the list of eliminated figures."""
    if state is None:
        return []
    key = None
    if region_key is not None:
        san = sanitize(str(region_key))
        for k in state.armies:
            if sanitize(k) == san:
                key = k
                break
    entry = army if army is not None else (
        state.armies.get(key) if key is not None else None)
    if entry is None:
        return []
    dead, survivors = eliminate_army_characters(
        state, entry.leaders, dying_side, card_name=card_name)
    entry.leaders = survivors
    if key is not None and not entry.units and not entry.leaders \
            and state.armies.get(key) is entry:
        state.armies.pop(key, None)
    return dead


def _track_fp_casualties(state: "GameState", casualties: List[Unit], army_side: Side):
    """Track Free Peoples casualties as permanent losses (rulebook p.30).

    FP casualties are permanently out of the game. Shadow casualties return
    to the reinforcement pool. We only need to track FP losses explicitly.
    """
    if army_side != Side.FREE_PEOPLES:
        return  # Shadow casualties return to pool — no tracking needed
    if not hasattr(state, 'fp_casualties') or state.fp_casualties is None:
        state.fp_casualties = {}
    for unit in casualties:
        nat = unit.nation
        if nat not in state.fp_casualties:
            state.fp_casualties[nat] = {'regular': 0, 'elite': 0, 'leader': 0}
        # USER RULING 2026-07-20 (supersedes the 1.188-era pool-swap
        # doctrine): a downgraded FREE PEOPLES elite is a PERMANENT
        # elite casualty — it does NOT return to reinforcements. Only
        # the replacement Regular is sourced from the pool (it entered
        # `deployed` at downgrade time). Conservation holds: elite
        # deployed -1 / cas elite +1; regular deployed +1 / pool -1.
        # SHADOW never reaches here (early-returned above -> its
        # downgraded elite returns to the pool as before). A downgraded
        # Unit keeps kind == ELITE, so it books in the elite branch.
        if unit.kind == UnitKind.REGULAR:
            state.fp_casualties[nat]['regular'] += 1
        else:
            state.fp_casualties[nat]['elite'] += 1


def _check_settlement_capture(state: "GameState", region_key: str, victor_side: Side):
    """
    Check if a settlement was captured and update enemy_controlled tracking.

    Called after defender is destroyed in combat.
    """
    from game_logic import is_fp_region, is_shadow_region

    meta = board.regions.get(region_key, {})

    # USER RULE (2026-07-08): ALL settlements are capturable — towns
    # included, not just cities and strongholds ("when a settlement
    # contains a unit of the opposite side, it gets captured")
    if not (meta.get("is_city") or meta.get("is_stronghold")
            or meta.get("is_town")):
        return

    # Use helper functions to determine original alignment
    is_fp_settlement = is_fp_region(region_key)
    is_sh_settlement = is_shadow_region(region_key)

    # wave-E fix 5 (2026-07-23): rulebook p.35 (_rulebook.txt L349-350) —
    # "Every time a Nation's Settlement (Town, City, or Stronghold) is
    # captured by the opponent" that Nation activates AND advances one
    # political step (an ADDITIONAL step beyond the attack-advance).  The
    # engine only advanced on ATTACK (advance_defenders_political); the
    # capture-advance was previously supplied ONLY by the teacher's
    # political_sync ground truth (RB-07) and NEVER happened in self-play.
    # Now the engine performs it natively.  Minimal, teacher-neutral: it
    # advances the OWNING nation's box one step (+activates), mirroring the
    # political_sync applier's effect EXACTLY, WITHOUT the heavier
    # activate_nation_for_attack path (its early persistent-card discard
    # cascade regressed the certified teacher replay).  political_sync is a
    # conditional absolute-set (`if want > cur`), so on the teacher path it
    # stays idempotent with this native advance (zero gate-1 shift, verified).
    # Fire once, only on a GENUINE opponent capture (region not already in
    # the captor's control set); recapture of one's own settlement never
    # advances.
    def _capture_advance():
        try:
            nat = None
            parent_id = meta.get("parent_id")
            if parent_id:
                from board import fid_to_name
                _fac = (fid_to_name.get(parent_id, "") or "").lower()
                _nm = {
                    'gondor': Nation.GONDOR, 'rohan': Nation.ROHAN,
                    'elves': Nation.ELVES, 'dwarves': Nation.DWARVES,
                    'the north': Nation.THE_NORTH, 'north': Nation.THE_NORTH,
                    'isengard': Nation.ISENGARD, 'sauron': Nation.SAURON,
                    'southrons': Nation.SOUTHRON_EASTERLINGS,
                    'southron': Nation.SOUTHRON_EASTERLINGS,
                    'southrons and easterlings':
                        Nation.SOUTHRON_EASTERLINGS,
                }
                nat = _nm.get(_fac)
            if nat is None:
                return
            cur = state.political.boxes.get(nat, 0)
            if cur < 3:
                # audit A11 (2026-07-26): activation must go through
                # game_logic.activate_nation so the Wormtongue gate and
                # the Rohan-activation discard triggers apply to
                # capture-advances too (was a direct active=True).
                if not state.political.active.get(nat, False):
                    import game_logic as _gl_cap
                    try:
                        _gl_cap.activate_nation(state, nat,
                                                method="attack",
                                                region=region_key)
                    except RuntimeError:
                        pass       # blocked (Wormtongue)
                # review P2 (2026-07-27): the passive->At-War clamp the
                # attacked-branch enforces applies to capture-advances
                # too — a still-passive nation may not be pushed to 3.
                if cur + 1 == 3 and not state.political.active.get(
                        nat, False):
                    pass
                else:
                    state.political.boxes[nat] = cur + 1
        except Exception:
            pass

    # Shadow capturing FP settlement
    if victor_side == Side.SHADOW and is_fp_settlement:
        _new_capture = region_key not in state.enemy_controlled[Side.SHADOW]
        state.enemy_controlled[Side.SHADOW].add(region_key)
        # Remove from FP controlled if they had recaptured it
        state.enemy_controlled[Side.FREE_PEOPLES].discard(region_key)
        if _new_capture:
            _capture_advance()

    # FP capturing Shadow settlement
    elif victor_side == Side.FREE_PEOPLES and is_sh_settlement:
        _new_capture = (region_key
                        not in state.enemy_controlled[Side.FREE_PEOPLES])
        state.enemy_controlled[Side.FREE_PEOPLES].add(region_key)
        # Remove from Shadow controlled if they had recaptured it
        state.enemy_controlled[Side.SHADOW].discard(region_key)
        if _new_capture:
            _capture_advance()

    # FP recapturing their own settlement from Shadow
    elif victor_side == Side.FREE_PEOPLES and is_fp_settlement:
        state.enemy_controlled[Side.SHADOW].discard(region_key)

    # Shadow recapturing their own settlement from FP
    elif victor_side == Side.SHADOW and is_sh_settlement:
        state.enemy_controlled[Side.FREE_PEOPLES].discard(region_key)


# ---------------------------------------------------------------------------
# Deferred combat-card effect choices (learned 'combat_effect_choice' phase)
# ---------------------------------------------------------------------------

def combat_effect_kind(combat_name: str) -> Optional[str]:
    """Choice-bearing combat-card effect kinds, by printed combat name."""
    name = (combat_name or '').strip().lower()
    return {
        'onslaught': 'onslaught',                    # post-casualty sacrifice
        'dread and despair': 'dread_despair',        # pre-roll forfeit
        'relentless assault': 'relentless_assault',  # pre-roll self-hits
        'heroic death': 'heroic_death',              # cancel via sacrifice
    }.get(name)


def _army_entry(state: "GameState", region: str):
    """(key, army) for a region, tolerant of case-variant keys."""
    san = sanitize(str(region))
    for k, a in state.armies.items():
        if sanitize(k) == san:
            return k, a
    return None, None


def _side_units(army: Army, side: Side) -> List[Unit]:
    return [u for u in army.units if side_of_nation(u.nation) == side]


def onslaught_max_sacrifice(state: "GameState", region: str,
                            side: Side) -> int:
    """Max HITS *side* may inflict on its own army via Onslaught at *region*.

    The card caps at four additional HITS; an Elite absorbs 2, so the own
    HIT CAPACITY is 2*elites + regulars. The Shadow may eliminate its ENTIRE
    army (almanac), so there is no keep-one floor.
    """
    _, army = _army_entry(state, region)
    if army is None:
        return 0
    units = _side_units(army, side)
    elites = sum(1 for u in units if u.kind == UnitKind.ELITE)
    hit_capacity = 2 * elites + (len(units) - elites)
    return max(0, min(4, hit_capacity))


def _write_back_side_units(state: "GameState", key: str, army: Army,
                           side: Side, new_units: List[Unit]) -> None:
    """Replace *side*'s units within a (possibly shared siege) army entry.

    Preserves the co-location convention (garrison units first)."""
    others = [u for u in army.units if side_of_nation(u.nation) != side]
    try:
        garrison = garrison_side_for(state, sanitize(key))
    except Exception:
        garrison = None
    if garrison is not None and garrison == side:
        army.units = list(new_units) + others
    else:
        army.units = others + list(new_units)


def resume_deferred_post_casualty(state: "GameState", atk_region: str,
                                  def_region: str, atk_side: Side,
                                  def_side: Side,
                                  atk_card_id: Optional[str] = None,
                                  def_card_id: Optional[str] = None,
                                  atk_cancelled: bool = False,
                                  def_cancelled: bool = False,
                                  is_siege: bool = False,
                                  choices: dict = None
                                  ) -> Tuple[bool, bool, Optional[str]]:
    """Phase 9 for the deferred-casualties round path: once post-roll
    allocation has resolved, run the SAME _apply_post_casualty_effects
    the standalone path runs (We Come to Kill, Great Host, Onslaught),
    on a thin CombatState wrapping the live post-allocation armies.
    Returns (attacker_destroyed, defender_destroyed, deferred_effect).
    """
    from cards import get_card
    atk_key, atk_parent = _army_entry(state, atk_region)
    def_key, def_parent = _army_entry(state, def_region)
    a_units = _side_units(atk_parent, atk_side) if atk_parent else []
    d_units = _side_units(def_parent, def_side) if def_parent else []

    def _cc(cid):
        if not cid:
            return None
        card = get_card(cid)
        return card.to_combat_card() if card else None

    ta = Army(region=(atk_parent.region if atk_parent
                      else sanitize(str(atk_region))),
              units=list(a_units), leaders=[])
    td = Army(region=(def_parent.region if def_parent
                      else sanitize(str(def_region))),
              units=list(d_units), leaders=[])
    cs = CombatState(
        attacker_region=sanitize(str(atk_region)),
        defender_region=sanitize(str(def_region)),
        attacker_army=ta, defender_army=td,
        attacker_side=atk_side, defender_side=def_side,
        is_siege=bool(is_siege))
    cs.attacker_card = _cc(atk_card_id)
    cs.defender_card = _cc(def_card_id)
    cs.attacker_card_cancelled = bool(atk_cancelled)
    cs.defender_card_cancelled = bool(def_cancelled)
    cs.attacker_destroyed = not ta.units
    cs.defender_destroyed = not td.units
    cs.battle_over = cs.attacker_destroyed or cs.defender_destroyed

    # publish the live state (battle() parity): downgrade replacements
    # must be SOURCED (casualty box first, else pool, else the elite
    # dies unreplaced) — an unpublished state let apply_casualties
    # mint replacement Regulars from thin air (batch-0 A4 breach)
    _prev_as = set_active_state(state)
    try:
        _apply_post_casualty_effects(cs, dict(choices or {}))
    finally:
        set_active_state(_prev_as)

    if atk_parent is not None:
        _write_back_side_units(state, atk_key, atk_parent, atk_side,
                               ta.units)
    if def_parent is not None:
        _write_back_side_units(state, def_key, def_parent, def_side,
                               td.units)
    _track_fp_casualties(state, cs.attacker_casualties, atk_side)
    _track_fp_casualties(state, cs.defender_casualties, def_side)
    return (bool(cs.attacker_destroyed), bool(cs.defender_destroyed),
            cs.deferred_effect)


def apply_onslaught_choice(state: "GameState", own_region: str,
                           enemy_region: str, own_side: Side,
                           enemy_side: Side, count: int,
                           own_policy: str = 'preserve_elites',
                           enemy_policy: str = 'preserve_elites',
                           enemy_is_defender: bool = True
                           ) -> Tuple[bool, bool, List[int], int]:
    """Resume a deferred Onslaught with the chosen sacrifice magnitude.

    'After removing casualties ... inflict and apply up to four additional
    hits against your units. Roll one die for each hit you inflicted ...
    score one hit against the enemy on each result of 4+.'

    Verified vs game0 T3 MT assault1: SP sacrificed 4 Regulars, rolled
    1 6 4 3 -> 2 hits (one die per inflicted hit, 4+ scores).

    Returns (own_destroyed, enemy_destroyed, rolls, hits).
    """
    own_key, own_army = _army_entry(state, own_region)
    enemy_key, enemy_army = _army_entry(state, enemy_region)
    if own_army is None or enemy_army is None:
        return False, False, [], 0
    own_units = _side_units(own_army, own_side)
    # cap on HITS: an Elite absorbs 2, so capacity = 2*elites + regulars
    # (full self-wipe permitted -- almanac); apply_casualties absorbs the k
    # hits via the standard elite-downgrade rules.
    _own_elites = sum(1 for u in own_units if u.kind == UnitKind.ELITE)
    hit_capacity = 2 * _own_elites + (len(own_units) - _own_elites)
    k = max(0, min(int(count), 4, hit_capacity))
    if k <= 0 or not _side_units(enemy_army, enemy_side):
        return False, False, [], 0

    # publish the live state (battle() parity): downgrade replacements
    # must be SOURCED (box first / pool / refuse) — an unpublished
    # state minted replacement Regulars (batch-0 A4 breach)
    _prev_as = set_active_state(state)
    try:
        # Self-inflicted hits (owner's casualty policy applies)
        tmp = Army(region=own_army.region, units=list(own_units),
                   leaders=[])
        self_cas = apply_casualties(tmp, k, own_policy)
        _write_back_side_units(state, own_key, own_army, own_side,
                               tmp.units)
        _track_fp_casualties(state, self_cas, own_side)

        # One die per inflicted hit; each 4+ scores a hit on the enemy
        rolls, hits = combat_roll(k, 4, 0)

        enemy_destroyed = False
        if hits > 0:
            etmp = Army(region=enemy_army.region,
                        units=_side_units(enemy_army, enemy_side),
                        leaders=[])
            enemy_cas = apply_casualties(etmp, hits, enemy_policy)
            _write_back_side_units(state, enemy_key, enemy_army,
                                   enemy_side, etmp.units)
            _track_fp_casualties(state, enemy_cas, enemy_side)
    finally:
        set_active_state(_prev_as)
    if hits > 0:
        if not etmp.units:
            enemy_destroyed = True
            # Destroyed side's CHARACTERS die with the army (USER RULING
            # 2026-08-15) — same reconciler as every other wipe path
            reconcile_destroyed_army(state, enemy_key, enemy_side,
                                     army=enemy_army)
            if enemy_is_defender \
                    and _side_units(own_army, own_side):
                # 1.152: no capture on mutual elimination
                _check_settlement_capture(state, sanitize(enemy_region),
                                          own_side)
    own_destroyed = not _side_units(own_army, own_side)
    if own_destroyed:
        # A self-wiped side loses its characters with the army; the
        # shared (siege) entry is only popped once BOTH sides are gone.
        reconcile_destroyed_army(state, own_key, own_side, army=own_army)
        # self-wiped DEFENDER loses the settlement to the surviving
        # attacker; a self-wiped ATTACKER captures nothing (1.152
        # mutual-elimination guard).
        if not enemy_is_defender and not enemy_destroyed \
                and _side_units(enemy_army, enemy_side):
            _check_settlement_capture(state, sanitize(own_region),
                                      enemy_side)
    return own_destroyed, enemy_destroyed, rolls, hits
