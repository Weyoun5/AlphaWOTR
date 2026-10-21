# persistent_effects.py - Persistent card effect hooks for War of the Ring
"""
Handles all persistent card effects that need to be checked during game flow:
- Movement blocking (A Power too Great, The Power of Tom Bombadil)
- Hunt modifications (Flocks of Crebain, Balrog of Moria)
- Combat restrictions (Denethor's Folly)
- Political blocking (Wormtongue, Threats and Promises)
- Card draw bonuses (The Palantír of Orthanc)
- Fellowship protection (The Last Battle)
"""

from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from game_state import GameState

from game_state import (
    Side, Nation, PersistentCardType, PersistentCard
)
from board import sanitize


def discard_table_card(state, pcard) -> None:
    """A persistent/table card removed from play goes to its deck's
    face-up discard pile (physical pile in the app). Pass the object
    returned by PersistentCardState.remove_card (or the card from the
    check_* list). BOTH paths - the app physically discarded it. CL-217."""
    if pcard is None:
        return
    cid = getattr(pcard, 'card_id', '') or ''
    pfx = 'fp' if getattr(pcard, 'side', None) == Side.FREE_PEOPLES else 'sh'
    deck = 'character' if 'Character' in cid else 'strategy'
    dk = "%s_%s" % (pfx, deck)
    d = getattr(state, 'discard_face_up', None)
    if isinstance(d, dict) and dk in d:
        d[dk].append(cid)


# ---------------------------------------------------------------------------
# Movement Blocking
# ---------------------------------------------------------------------------

# Regions protected by "A Power too Great".  Membership is tested on
# sanitize() output (lowercase alnum + single spaces), with the
# leading-article display variant defended like the sibling set.
POWER_TOO_GREAT_REGIONS = {"lorien", "rivendell", "grey havens", "the grey havens"}

# Regions protected by "The Power of Tom Bombadil"
TOM_BOMBADIL_REGIONS = {"the old forest", "old forest", "the shire", "shire", "buckland"}


def can_shadow_move_to(state: "GameState", region: str) -> Tuple[bool, Optional[str]]:
    """
    Check if Shadow can move an army into a region.
    Returns (can_move, blocking_card_name).
    """
    region_key = sanitize(region)

    # Check A Power too Great
    if state.persistent_cards.has_card_type(PersistentCardType.POWER_TOO_GREAT):
        if region_key in POWER_TOO_GREAT_REGIONS:
            return False, "A Power too Great"

    # Check The Power of Tom Bombadil
    if state.persistent_cards.has_card_type(PersistentCardType.TOM_BOMBADIL):
        if region_key in TOM_BOMBADIL_REGIONS:
            return False, "The Power of Tom Bombadil"

    return True, None


def can_shadow_attack(state: "GameState", region: str) -> Tuple[bool, Optional[str]]:
    """
    Check if Shadow can attack a region (including siege assault).
    Same restrictions as movement.
    """
    return can_shadow_move_to(state, region)


# ---------------------------------------------------------------------------
# Hunt Modifications
# ---------------------------------------------------------------------------

def should_draw_extra_hunt_tile(state: "GameState", from_region: str, to_region: str) -> Tuple[bool, Optional[str]]:
    """
    Check if an extra hunt tile should be drawn for this movement.

    Balrog of Moria: Extra tile when Fellowship moves into/out of/through Moria.

    Returns (should_draw, card_name).
    """
    if state.persistent_cards.has_card_type(PersistentCardType.BALROG_OF_MORIA):
        from_key = sanitize(from_region)
        to_key = sanitize(to_region)
        if from_key == "moria" or to_key == "moria":
            return True, "Balrog of Moria"
    return False, None


def should_add_dice_to_hunt_box(state: "GameState") -> bool:
    """
    Check if dice used to move Fellowship should be added to Hunt Box.

    The Last Battle: When this persistent card is in play, FP dice used
    for Fellowship movement are NOT placed in the Hunt Box.
    """
    if not state.persistent_cards.has_card_type(PersistentCardType.LAST_BATTLE):
        return True  # Normal: add dice to hunt box

    # The Last Battle is in play -- Fellowship movement dice bypass the Hunt Box
    return False


# ---------------------------------------------------------------------------
# Combat Restrictions
# ---------------------------------------------------------------------------

def can_fp_play_combat_card(state: "GameState", region: str) -> Tuple[bool, Optional[str]]:
    """
    Check if Free Peoples can play a combat card in this battle.

    Denethor's Folly: FP cannot use combat cards in Minas Tirith battles.

    Returns (can_play, blocking_card_name).
    """
    if state.persistent_cards.has_card_type(PersistentCardType.DENETHORS_FOLLY):
        if sanitize(region) == "minas tirith":
            return False, "Denethor's Folly"

    return True, None


def get_denethors_folly_discard_options(state: "GameState") -> Dict:
    """
    Get options for FP to discard Denethor's Folly.

    - Will of the West Action Die result, OR
    - Any Action Die if Gandalf or Aragorn is in Minas Tirith
    """
    can_use_any_die = False

    # sanitize-aware lookup; almanac A3: Gandalf (Grey or White) or Aragorn
    # (NOT Strider) must be with the besieged army in Minas Tirith
    mt_army = None
    for _k, _a in state.armies.items():
        if sanitize(_k) == 'minas tirith':
            mt_army = _a
            break
    if mt_army:
        for leader in mt_army.leaders:
            nm = getattr(leader, 'name', '')
            # accept the crowned name variant "Aragorn - Heir to Isildur"
            if (nm in ('Gandalf the Grey', 'Gandalf the White')
                    or nm.startswith('Aragorn')):
                can_use_any_die = True
                break

    # (fellowship-internal companions are NOT on the board — only the
    #  army.leaders scan above counts, doctrine 1.211l)
    return {
        "will_of_west_die": True,
        "any_die": can_use_any_die,
        "reason": "Gandalf or Aragorn in Minas Tirith" if can_use_any_die else None
    }


# ---------------------------------------------------------------------------
# Political Restrictions
# ---------------------------------------------------------------------------

def can_activate_nation(state: "GameState", nation: "Nation", method: str,
                        region: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if a nation can be activated.

    Wormtongue: Rohan cannot activate except by:
    - Companion's special ability
    - Fellowship declared in Edoras or Helm's Deep
    - Attack on Edoras or Helm's Deep

    Args:
        nation: The nation to activate
        method: "muster_die", "companion_ability", "companion_move"
            (Edoras/Helm's Deep companion entry, CASE4b),
            "fellowship_declared", "attack", "event_card".  Callers
            also pass "witch_king_entry" / "aragorn_crowned"; any
            method NOT whitelisted below is blocked for Rohan under
            Wormtongue by design.

    Returns (can_activate, blocking_card_name).
    """
    if state.persistent_cards.has_card_type(PersistentCardType.WORMTONGUE):
        if nation == Nation.ROHAN:
            # Wormtongue card text: the attack / declaration exceptions apply
            # ONLY at Edoras or Helm's Deep
            allowed = method == "companion_ability"
            if method == "companion_move":
                # card text: Rohan may activate via a Companion entering
                # Edoras or Helm's Deep (CASE4b)
                allowed = sanitize(region or '') in ("edoras",
                                                     "helm s deep")
            elif method == "fellowship_declared":
                # the declare path only activates the DECLARED settlement's
                # nation, and Edoras/Helm's Deep are the only Rohan
                # settlements - permissive here (CASE5b/5c; callers pass
                # no region)
                allowed = True
            elif method == "attack":
                allowed = sanitize(region or '') in ("edoras",
                                                     "helm s deep")
            if not allowed:
                return False, "Wormtongue"

    return True, None


def can_advance_passive_nation(state: "GameState", nation: "Nation", method: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a passive nation can advance on the political track.

    Threats and Promises: FP cannot advance passive nations using Muster die.
    (Still can advance via attack or companion ability)

    Returns (can_advance, blocking_card_name).
    """
    if state.persistent_cards.has_card_type(PersistentCardType.THREATS_AND_PROMISES):
        if not state.political.active[nation]:  # Nation is passive
            if method == "muster_die":
                return False, "Threats and Promises"

    return True, None


# ---------------------------------------------------------------------------
# Card Draw Bonuses
# ---------------------------------------------------------------------------

def should_draw_extra_card_on_event(state: "GameState") -> bool:
    """
    Check if Shadow should draw an extra card when playing Event die.

    The Palantír of Orthanc: Draw card from either deck when playing Event.
    """
    return state.persistent_cards.has_card_type(PersistentCardType.PALANTIR_OF_ORTHANC)


def check_fellowship_declared_discards(state: "GameState", region: str) -> List["PersistentCard"]:
    """
    Check which cards should be discarded when Fellowship is declared in a region.

    - Worn with Sorrow and Toil: Discard if declared in FP City/Stronghold
    - Flocks of Crebain: Discard if declared in FP City/Stronghold
    - QA 2026-07-11 (Worn CASE 15): a Shadow-CAPTURED settlement is not
      FP-controlled - no discard there (guard added below).

    Returns list of cards to discard.
    """
    if sanitize(region) in state.enemy_controlled.get(
            Side.SHADOW, set()):
        return []  # captured settlement is not FP-controlled
    to_discard = []
    region_key = sanitize(region)

    # Check if it's an FP City or Stronghold
    import board
    from game_logic import is_fp_region
    meta = board.regions.get(region_key, {})
    is_fp_settlement = (
        (meta.get("is_city") or meta.get("is_stronghold")) and
        is_fp_region(region_key)
    )

    if is_fp_settlement:
        # Check Worn with Sorrow and Toil
        card = state.persistent_cards.get_by_type(PersistentCardType.WORN_WITH_SORROW)
        if card:
            to_discard.append(card)

        # Check Flocks of Crebain
        card = state.persistent_cards.get_by_type(PersistentCardType.FLOCKS_OF_CREBAIN)
        if card:
            to_discard.append(card)

    return to_discard


def check_rohan_activated_discards(state: "GameState") -> List["PersistentCard"]:
    """
    Check which cards should be discarded when Rohan is activated.

    - Wormtongue: Discard when Rohan activates

    Returns list of cards to discard.
    """
    to_discard = []

    if state.political.active[Nation.ROHAN]:
        card = state.persistent_cards.get_by_type(PersistentCardType.WORMTONGUE)
        if card:
            to_discard.append(card)

    return to_discard


def check_nation_advanced_discards(state: "GameState", nation: "Nation", method: str) -> List["PersistentCard"]:
    """
    Check which cards should be discarded when a nation advances.

    - Threats and Promises: Discard when FP nation advances via attack or companion ability

    Returns list of cards to discard.
    """
    to_discard = []

    if method in ("attack", "companion_ability"):
        if nation in (Nation.ELVES, Nation.DWARVES, Nation.GONDOR, Nation.THE_NORTH, Nation.ROHAN):
            card = state.persistent_cards.get_by_type(PersistentCardType.THREATS_AND_PROMISES)
            if card:
                to_discard.append(card)

    return to_discard


def check_saruman_eliminated_discards(state: "GameState") -> List["PersistentCard"]:
    """
    Check which cards should be discarded when Saruman is eliminated.

    - The Palantír of Orthanc: Discard if Saruman eliminated
    - Wormtongue: Discard if Saruman eliminated

    Returns list of cards to discard.
    """
    to_discard = []

    # Check if Saruman is no longer in play
    if state.saruman and not state.saruman.in_play:
        card = state.persistent_cards.get_by_type(PersistentCardType.PALANTIR_OF_ORTHANC)
        if card:
            to_discard.append(card)

        card = state.persistent_cards.get_by_type(PersistentCardType.WORMTONGUE)
        if card:
            to_discard.append(card)

    return to_discard


def check_companion_casualty_effect(state: "GameState") -> Optional[Dict]:
    """
    Check if Worn with Sorrow and Toil should trigger (it fires on ANY
    companion casualty, so the fallen companion's identity is not
    consulted).

    Returns dict describing the effect if it triggers, None otherwise.
    """
    if state.persistent_cards.has_card_type(PersistentCardType.WORN_WITH_SORROW):
        return {
            "card_name": "Worn with Sorrow and Toil",
            "effect": "discard_fp_character_card",
            "description": "Discard a FP Character card from hand (random) or table"
        }
    return None


# ---------------------------------------------------------------------------
# Helper: Create PersistentCard from card data
# ---------------------------------------------------------------------------

def create_persistent_card(card_id: str, card_name: str, side: "Side") -> PersistentCard:
    """
    Create a PersistentCard with appropriate type and requirements.
    """
    # Determine card type
    card_type = None
    requires_companion = None

    name_lower = card_name.lower()

    # Free Peoples cards
    if "mithril coat" in name_lower:
        card_type = PersistentCardType.MITHRIL_COAT
    elif "axe and bow" in name_lower:
        card_type = PersistentCardType.AXE_AND_BOW
        requires_companion = "Gimli, Legolas"
    elif "horn of gondor" in name_lower:
        card_type = PersistentCardType.HORN_OF_GONDOR
        requires_companion = "Boromir"
    elif "wizard's staff" in name_lower or "wizards staff" in name_lower:
        card_type = PersistentCardType.WIZARD_STAFF
        requires_companion = "Gandalf the Grey"
    elif "last battle" in name_lower:
        card_type = PersistentCardType.LAST_BATTLE
    elif "power too great" in name_lower:
        card_type = PersistentCardType.POWER_TOO_GREAT
    elif "tom bombadil" in name_lower:
        card_type = PersistentCardType.TOM_BOMBADIL

    # Shadow cards
    elif "worn with sorrow" in name_lower:
        card_type = PersistentCardType.WORN_WITH_SORROW
    elif "flocks of crebain" in name_lower or "crebain" in name_lower:
        card_type = PersistentCardType.FLOCKS_OF_CREBAIN
    elif "balrog" in name_lower:
        card_type = PersistentCardType.BALROG_OF_MORIA
    elif "palantír" in name_lower or "palantir" in name_lower:
        card_type = PersistentCardType.PALANTIR_OF_ORTHANC
    elif "wormtongue" in name_lower:
        card_type = PersistentCardType.WORMTONGUE
    elif "denethor" in name_lower:
        card_type = PersistentCardType.DENETHORS_FOLLY
    elif "threats and promises" in name_lower:
        card_type = PersistentCardType.THREATS_AND_PROMISES

    if card_type is None:
        # This shouldn't happen for known persistent cards
        raise ValueError(f"Unknown persistent card: {card_name}")

    return PersistentCard(
        card_id=card_id,
        card_name=card_name,
        side=side,
        card_type=card_type,
        requires_companion=requires_companion
    )
