# game_state.py – WotR data model
"""WotR data model: pieces, Fellowship, Hunt tracking, and Combat state."""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set

import board


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Side(Enum):
    FREE_PEOPLES = "Free Peoples"
    SHADOW = "Shadow"


class Nation(Enum):
    # Free Peoples
    ELVES = "Elves"
    DWARVES = "Dwarves"
    GONDOR = "Gondor"
    THE_NORTH = "The North"
    ROHAN = "Rohan"
    # Shadow
    ISENGARD = "Isengard"
    SAURON = "Sauron"
    SOUTHRON_EASTERLINGS = "Southrons & Easterlings"


class UnitKind(Enum):
    REGULAR = "Regular"
    ELITE = "Elite"


class ActionDieFace(Enum):
    CHARACTER = "Character"
    ARMY = "Army"
    MUSTER = "Muster"
    EVENT = "Event"
    MUSTER_ARMY = "Muster/Army"
    EYE = "Eye of Sauron"
    WILL_OF_WEST = "Will of the West"

# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------
# Die-face art (FP_PNG / SH_PNG): renderer helper; no in-tree reader as of 2026-08-29.

FP_PNG: Dict[ActionDieFace, str] = {
    ActionDieFace.CHARACTER: "ADFPcharacter.png",
    ActionDieFace.MUSTER: "ADFPmuster.png",
    ActionDieFace.EVENT: "ADFPevent.png",
    ActionDieFace.MUSTER_ARMY: "ADFParmymuster.png",
    ActionDieFace.WILL_OF_WEST: "ADFPwill.png",
}

SH_PNG: Dict[ActionDieFace, str] = {
    ActionDieFace.CHARACTER: "ADSAcharacter.png",
    ActionDieFace.ARMY: "ADSAarmy.png",
    ActionDieFace.MUSTER: "ADSAmuster.png",
    ActionDieFace.EVENT: "ADSAevent.png",
    ActionDieFace.MUSTER_ARMY: "ADSAarmymuster.png",
    ActionDieFace.EYE: "ADSAeye.png",
}

# ---------------------------------------------------------------------------
# Map pieces
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    nation: Nation
    kind: UnitKind


@dataclass
class Leader:
    nation: Nation


@dataclass
class Nazgul(Leader):
    pass


@dataclass
class Army:
    region: "board.Region"
    units: List[Unit] = field(default_factory=list)
    leaders: List[Leader] = field(default_factory=list)

    def add(self, *objs):
        for o in objs:
            if isinstance(o, Unit):
                self.units.append(o)
            else:
                self.leaders.append(o)

# ---------------------------------------------------------------------------
# Fellowship & tracks
# ---------------------------------------------------------------------------

@dataclass
class Companion:
    """Fellowship companion with abilities."""
    name: str
    level: int
    leadership: int = 1
    captain_of_the_west: bool = False
    # Guide abilities (only active while in fellowship as guide)
    guide_ability: Optional[str] = None  # e.g., "palantir_draw", "hide_any_die", "sacrifice_reduce_hunt"
    # Nation activation - which nations can this companion activate
    activates_nations: List[Nation] = field(default_factory=list)  # Empty = activates any
    activation_regions: List[str] = field(default_factory=list)  # If set, must be in these regions
    # Special muster ability
    can_muster_nation: Optional[Nation] = None  # Can use any die to muster this nation
    muster_regions: List[str] = field(default_factory=list)  # Must be in these regions to muster
    # Special movement (for Gandalf the White)
    bonus_movement: int = 0  # Added to level for movement when alone/with hobbit
    solo_hobbit_movement: bool = False  # If True, bonus_movement only when alone or with 1 hobbit


@dataclass
class ShadowMinion:
    """Special Shadow characters: Witch King, Saruman, Mouth of Sauron."""
    name: str
    level: int
    leadership: int
    in_play: bool = False
    was_ever_in_play: bool = False  # Tracks if minion ever entered play (for Gandalf White check)
    # Special abilities
    immobile: bool = False  # Saruman can't leave Orthanc
    infinite_movement: bool = False  # Witch King
    # Combat abilities
    combat_card_draw: bool = False  # Witch King first round draw
    isengard_elite_leadership: bool = False  # Saruman grants elite leadership


@dataclass
class GandalfTheWhite:
    """Upgraded Gandalf with special abilities."""
    level: int = 3
    leadership: int = 1
    captain_of_the_west: bool = False  # Note: Gandalf White does NOT have captain
    in_play: bool = False
    # Movement = 4 when alone or with 1 hobbit
    base_movement: int = 3
    solo_hobbit_movement: int = 4
    # Can forfeit leadership to cancel all Nazgûl leadership
    can_cancel_nazgul_leadership: bool = True


@dataclass
class Aragorn:
    """Crowned Aragorn with enhanced abilities."""
    level: int = 3
    leadership: int = 2
    captain_of_the_west: bool = True
    crowned: bool = False
    # Crown regions
    crown_regions: List[str] = field(default_factory=lambda: ["minas tirith", "pelargir", "dol amroth"])


@dataclass
class Fellowship:
    # Companions still alive in the Company
    companions: List[Companion]
    # Map location of the hidden Company pawn
    region: "board.Region"
    # Current guide (must always be in `companions`, except for Gollum)
    guide: Companion
    # Progress & corruption
    progress: int = 0
    corruption: int = 0
    hidden: bool = True
    # Mordor track
    mordor_entered: bool = False
    mordor_step: int = 0  # 0-5 once in Mordor

@dataclass
class PoliticalTrack:
    boxes: Dict[Nation, int]   # 0-3 (3 = At War)
    active: Dict[Nation, bool] # True = coloured (active) side

    def is_active(self, nation: Nation) -> bool:
        """Nation has been activated (flipped to coloured/active side).

        A passive nation cannot advance to 'At War' until activated.
        Note: activation is separate from disposition (box position).
        All nations can move armies within their own borders regardless
        of active/passive status.
        """
        return self.active.get(nation, False)

    def is_at_war(self, nation: Nation) -> bool:
        """Nation is at war (box >= 3). Can attack, recruit, cross borders."""
        return self.boxes[nation] >= 3

@dataclass
class VictoryPoints:
    free_peoples: int = 0
    shadow: int = 0

# ---------------------------------------------------------------------------
# Event cards & player hands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Combat Card & Combat State
# ---------------------------------------------------------------------------

@dataclass
class CombatCard:
    """A card being used in combat."""
    card_id: str
    side: Side
    deck: str
    combat_name: str
    combat_value: int
    combat_req: str
    combat_effect: str

    # Parsed requirements (populated by cards.py)
    requires_companion: bool = False
    requires_nazgul: bool = False
    requires_elite: bool = False
    requires_leader: bool = False
    min_nazgul_leadership: int = 0
    requires_fellowship_region: bool = False
    requires_stronghold: bool = False
    requires_fp_nation: bool = False
    region_restrictions: List[str] = field(default_factory=list)
    requires_isengard_unit: bool = False
    requires_se_elite: bool = False
    requires_shadow_elite: bool = False
    requires_fp_elite: bool = False


@dataclass
class CombatState:
    """Tracks all state during a battle."""
    attacker_region: str
    defender_region: str
    attacker_army: "Army"
    defender_army: "Army"
    attacker_side: Side
    defender_side: Side

    is_siege: bool = False
    siege_round: int = 1
    stronghold_defense: bool = False
    city_defense: bool = False          # Defending in city or fortification (attacker 6+ first round)

    # Cards
    attacker_card: Optional[CombatCard] = None
    defender_card: Optional[CombatCard] = None
    attacker_card_cancelled: bool = False
    defender_card_cancelled: bool = False

    # Calculated values
    attacker_combat_strength: int = 0
    defender_combat_strength: int = 0
    attacker_leadership: int = 0
    defender_leadership: int = 0

    # Dice modifiers from card effects
    attacker_dice_mod: int = 0
    defender_dice_mod: int = 0
    attacker_reroll_mod: int = 0
    defender_reroll_mod: int = 0
    attacker_dice_cap: Optional[int] = None
    defender_dice_cap: Optional[int] = None

    # Roll tracking
    attacker_combat_rolls: List[int] = field(default_factory=list)
    defender_combat_rolls: List[int] = field(default_factory=list)
    attacker_rerolls: List[int] = field(default_factory=list)
    defender_rerolls: List[int] = field(default_factory=list)
    # Pre-combat attack rolls (Sudden Strike / Charge / Durin's Bane) — kept
    # so self-play traces can show them (they were rolled and discarded)
    attacker_pre_combat_rolls: List[int] = field(default_factory=list)
    defender_pre_combat_rolls: List[int] = field(default_factory=list)
    attacker_hits: int = 0
    defender_hits: int = 0
    attacker_pre_combat_hits: int = 0
    defender_pre_combat_hits: int = 0
    attacker_bonus_hits: int = 0
    defender_bonus_hits: int = 0

    # Reroll cancellation
    attacker_reroll_cancelled: bool = False
    defender_reroll_cancelled: bool = False

    # Special characters present
    companions_in_battle: List["Companion"] = field(default_factory=list)
    nazgul_in_battle: int = 0
    minions_in_battle: List[str] = field(default_factory=list)

    # Confusion effect (1s hurt roller)
    attacker_confusion: bool = False
    defender_confusion: bool = False

    # Casualties prevented
    attacker_hits_cancelled: int = 0
    defender_hits_cancelled: int = 0

    # A choice-bearing post-casualty effect (Onslaught) deferred to the
    # learned 'combat_effect_choice' phase: 'attacker'/'defender'/None
    deferred_effect: Optional[str] = None

    # Outcome
    attacker_casualties: List["Unit"] = field(default_factory=list)
    defender_casualties: List["Unit"] = field(default_factory=list)
    battle_over: bool = False
    attacker_retreated: bool = False
    defender_retreated: bool = False
    attacker_destroyed: bool = False
    defender_destroyed: bool = False


@dataclass
class CombatResult:
    """Result of a completed battle."""
    attacker_region: str
    defender_region: str
    rounds_fought: int
    attacker_total_hits: int
    defender_total_hits: int
    attacker_casualties: List["Unit"]
    defender_casualties: List["Unit"]
    attacker_destroyed: bool
    defender_destroyed: bool
    attacker_retreated: bool
    defender_retreated: bool
    siege_continues: bool = False
    # A choice-bearing post-casualty combat-card effect (Onslaught) was
    # deferred to a learned pending decision: 'attacker'/'defender'/None
    deferred_effect: "Optional[str]" = None

    def __str__(self) -> str:
        result = f"Battle: {self.attacker_region} → {self.defender_region}\n"
        result += f"  Rounds: {self.rounds_fought}\n"
        result += f"  Attacker hits: {self.attacker_total_hits}, casualties: {len(self.attacker_casualties)}\n"
        result += f"  Defender hits: {self.defender_total_hits}, casualties: {len(self.defender_casualties)}\n"
        if self.attacker_destroyed:
            result += "  Attacker army destroyed!\n"
        if self.defender_destroyed:
            result += "  Defender army destroyed!\n"
        if self.attacker_retreated:
            result += "  Attacker retreated.\n"
        if self.defender_retreated:
            result += "  Defender retreated.\n"
        return result


@dataclass
class DeckStatus:
    """Card counts remaining in each event deck."""
    fp_character: int
    fp_strategy: int
    sh_character: int
    sh_strategy: int


# ---------------------------------------------------------------------------
# Persistent Card Tracking
# ---------------------------------------------------------------------------

class PersistentCardType(Enum):
    """Types of persistent cards with their discard conditions."""
    # Free Peoples cards
    MITHRIL_COAT = "Mithril Coat and Sting"
    AXE_AND_BOW = "Axe and Bow"
    HORN_OF_GONDOR = "Horn of Gondor"
    WIZARD_STAFF = "Wizard's Staff"
    LAST_BATTLE = "The Last Battle"
    POWER_TOO_GREAT = "A Power too Great"
    TOM_BOMBADIL = "The Power of Tom Bombadil"

    # Shadow cards
    WORN_WITH_SORROW = "Worn with Sorrow and Toil"
    FLOCKS_OF_CREBAIN = "Flocks of Crebain"
    BALROG_OF_MORIA = "Balrog of Moria"
    PALANTIR_OF_ORTHANC = "The Palantír of Orthanc"
    WORMTONGUE = "Wormtongue"
    DENETHORS_FOLLY = "Denethor's Folly"
    THREATS_AND_PROMISES = "Threats and Promises"


@dataclass
class PersistentCard:
    """A card that stays on the table with ongoing effects."""
    card_id: str
    card_name: str
    side: Side
    card_type: Optional[PersistentCardType] = None  # None for generic tracking

    # Tracking for conditional discard
    requires_companion: Optional[str] = None  # Discard if companion leaves (e.g., Axe and Bow needs Gimli/Legolas)

    def __hash__(self):
        return hash(self.card_id)


@dataclass
class PersistentCardState:
    """Tracks all persistent cards in play and provides helper methods."""
    fp_cards: List[PersistentCard] = field(default_factory=list)
    sh_cards: List[PersistentCard] = field(default_factory=list)

    def add_card(self, card: PersistentCard):
        """Add a persistent card to play."""
        if card.side == Side.FREE_PEOPLES:
            self.fp_cards.append(card)
        else:
            self.sh_cards.append(card)

    def remove_card(self, card_id: str) -> Optional[PersistentCard]:
        """Remove a card from play by ID."""
        for card in self.fp_cards:
            if card.card_id == card_id:
                self.fp_cards.remove(card)
                return card
        for card in self.sh_cards:
            if card.card_id == card_id:
                self.sh_cards.remove(card)
                return card
        return None

    def remove_by_name(self, card_name: str) -> Optional[PersistentCard]:
        """Remove a card from play by name."""
        for card in self.fp_cards:
            if card.card_name == card_name:
                self.fp_cards.remove(card)
                return card
        for card in self.sh_cards:
            if card.card_name == card_name:
                self.sh_cards.remove(card)
                return card
        return None

    def has_card(self, card_name: str) -> bool:
        """Check if a card is in play by name."""
        return any(c.card_name == card_name for c in self.fp_cards + self.sh_cards)

    def has_card_type(self, card_type: PersistentCardType) -> bool:
        """Check if a card type is in play."""
        return any(c.card_type == card_type for c in self.fp_cards + self.sh_cards)

    def get_by_type(self, card_type: PersistentCardType) -> Optional[PersistentCard]:
        """Get a card by type."""
        for card in self.fp_cards + self.sh_cards:
            if card.card_type == card_type:
                return card
        return None

    def get_fp_character_cards(self) -> List[PersistentCard]:
        """Get all FP character cards on the table (for Nazgul Strike targeting)."""
        return [c for c in self.fp_cards if "Character" in c.card_id]

    def check_companion_requirements(self, companions: List["Companion"]) -> List[PersistentCard]:
        """Check which cards should be discarded due to companion leaving."""
        companion_names = {c.name for c in companions}
        to_discard = []

        for card in self.fp_cards:
            if card.requires_companion:
                # Parse required companions (comma-separated)
                required = [n.strip() for n in card.requires_companion.split(",")]
                # If ALL required companions are gone, discard
                if not any(r in companion_names for r in required):
                    to_discard.append(card)

        return to_discard

@dataclass
class PlayerInfo:
    """Hidden per-side info."""
    hand: List[str] = field(default_factory=list)             # card IDs
    discarded_face_down: List[str] = field(default_factory=list)
    needs_discard: bool = False        # hand > 6-card limit pending

# ---------------------------------------------------------------------------
# Hunt tiles & hunt related containers
# ---------------------------------------------------------------------------

@dataclass
class HuntTile:
    damage: int = 0           # corruption value (-2..3; negatives =
                              # beneficial FP special tiles)
    is_eye: bool = False      # Eye tiles (always reveal per rulebook)
    reveal: bool = False      # "Reveal" modifier

    def __post_init__(self):
        # Per rulebook: Eye tiles ALWAYS reveal the fellowship
        if self.is_eye:
            self.reveal = True

    # Special tile properties
    is_special: bool = False  # True for card-added tiles
    special_name: Optional[str] = None  # Name of the special tile
    stops_movement: bool = False  # Stop tiles cancel fellowship movement
    is_dice_roll: bool = False    # Shelob's Lair: roll d6 for damage instead of fixed
    is_fellowship_special: bool = False  # True for FP special tiles (beneficial)
    image_file: Optional[str] = None  # Image filename for display

    def __repr__(self):
        # A NAMED special keeps its identity even when is_eye is set
        # ("The Ring is Mine!" must never render as a plain Eye).
        if self.special_name:
            return self.special_name
        if self.is_eye:
            return "Eye"
        s = str(self.damage)
        if self.reveal:
            s += "R"
        return s


# Special hunt tiles from cards (added when entering Mordor)
SPECIAL_HUNT_TILES = {
    # Free Peoples special tiles (beneficial)
    "Sméagol Helps Nice Master": HuntTile(
        damage=-1, is_special=True, special_name="Sméagol Helps Nice Master",
        is_fellowship_special=True, image_file="b-1.png"
    ),
    "Elven Cloaks": HuntTile(
        damage=0, is_special=True, special_name="Elven Cloaks",
        is_fellowship_special=True, image_file="b0.png"
    ),
    "Elven Rope": HuntTile(
        damage=0, is_special=True, special_name="Elven Rope",
        is_fellowship_special=True, image_file="b0.png"
    ),
    "Phial of Galadriel": HuntTile(
        damage=-2, is_special=True, special_name="Phial of Galadriel",
        is_fellowship_special=True, image_file="b-2.png"
    ),
    # Shadow special tiles (stop tiles)
    "On On They Went": HuntTile(
        damage=3, reveal=False, is_special=True, special_name="On On They Went",
        stops_movement=True, image_file="r3s.png"
    ),
    "Shelob's Lair": HuntTile(
        damage=0, reveal=False, is_special=True, special_name="Shelob's Lair",
        stops_movement=True, is_dice_roll=True, image_file="rds.png"  # d6 roll for damage
    ),
    "The Ring is Mine!": HuntTile(
        damage=0, reveal=True, is_special=True, special_name="The Ring is Mine!",
        stops_movement=True, is_eye=True, image_file="reyers.png"  # Eye + stop
    ),
    "Give it to Uss!": HuntTile(
        damage=1, reveal=True, is_special=True, special_name="Give it to Uss!",
        stops_movement=True, image_file="r1rs.png"
    ),
}



def _initial_hunt_pool() -> List[HuntTile]:
    """Populate and shuffle the 16-tile hunt pool."""
    tiles = [
        HuntTile(3), HuntTile(3), HuntTile(3),
        HuntTile(2), HuntTile(2),
        HuntTile(1), HuntTile(1),
        HuntTile(is_eye=True), HuntTile(is_eye=True),
        HuntTile(is_eye=True), HuntTile(is_eye=True),
        HuntTile(2, reveal=True),
        HuntTile(1, reveal=True), HuntTile(1, reveal=True),
        HuntTile(0, reveal=True), HuntTile(0, reveal=True),
    ]
    random.shuffle(tiles)
    return tiles

# ---------------------------------------------------------------------------
# Dice objects
# ---------------------------------------------------------------------------

class ActionDie:
    __slots__ = ("faces", "current", "side", "used")

    def __init__(self, faces: List[ActionDieFace], side: Side):
        self.faces = faces
        self.current: Optional[ActionDieFace] = None
        self.side = side
        self.used: bool = False  # False = unused / available; True = spent

    # --- gameplay helpers --------------------------------------------------

    def roll(self) -> ActionDieFace:
        if self.used:
            raise RuntimeError("Cannot roll a used die")
        self.current = random.choice(self.faces)
        return self.current

    def mark_used(self):
        """Flag this die as spent so UI can hide / grey it out."""
        self.used = True

    # --- renderer helper ---------------------------------------------------

    def png(self) -> str:
        if self.used:
            return "blank.png"
        if self.current is None:
            # Default fallback image when unrolled
            return "ADSAeye.png" if self.side is Side.SHADOW else "ADFPwill.png"
        table = SH_PNG if self.side is Side.SHADOW else FP_PNG
        return table[self.current]

# ---------------------------------------------------------------------------
# Constants for dice faces
# ---------------------------------------------------------------------------

FP_FACES = [
    ActionDieFace.CHARACTER,
    ActionDieFace.CHARACTER,
    ActionDieFace.MUSTER,
    ActionDieFace.EVENT,
    ActionDieFace.MUSTER_ARMY,
    ActionDieFace.WILL_OF_WEST,
]
SH_FACES = [
    ActionDieFace.CHARACTER,
    ActionDieFace.ARMY,
    ActionDieFace.MUSTER,
    ActionDieFace.EVENT,
    ActionDieFace.MUSTER_ARMY,
    ActionDieFace.EYE,
]

# ---------------------------------------------------------------------------
# Complete game state container
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board: "board.Board"
    armies: Dict[str, Army]
    enemy_controlled: Dict[Side, Set[str]]

    fellowship: Fellowship
    political: PoliticalTrack
    vp: VictoryPoints

    dice: Dict[Side, List[ActionDie]]
    turn: int

    # Hunt
    hunt_pool: List[HuntTile]
    hunt_discard: List[HuntTile]
    hunt_removed: List[HuntTile]
    hunt_box: List[ActionDie]

    # Cards
    decks: DeckStatus
    hands: Dict[Side, PlayerInfo]
    discard_face_up: Dict[str, List[str]]
    # One-shot resources
    fp_elven_rings: int = 3
    fp_dwarven_rings: int = 0
    fp_political_token: bool = False     # Hunt for the Ring (Political) available
    fp_card_token: bool = False          # Hunt for the Ring (Card Draw) available
    sh_elven_rings: int = 0
    sh_dwarven_rings: int = 0
    sh_action_tokens: int = 0

    # Hunt box breakdown (reset each turn)
    eyes_allocated: int = 0   # Shadow dice voluntarily allocated as eyes pre-roll

    # Ring usage tracking (reset at start of each turn)
    fp_ring_used_this_turn: bool = False
    sh_ring_used_this_turn: bool = False
    # A7 (USER 2026-08-19): an Elven Ring is a LEGALITY EXPANSION, not a face
    # change - while up, ONE action of that side may be funded by any die the
    # ring permits (FP: any non-Will die; Shadow: any) and the die KEEPS its
    # rolled face.  Cleared by game_logic.ring_waiver_spend and at every turn
    # boundary.  (The client's ring command renders no die face, so a face
    # change is unreconstructible from any log.)
    fp_ring_waiver: bool = False
    sh_ring_waiver: bool = False
    # Mouth of Sauron ability (user ruling 2026-07-03): once per turn,
    # while the Mouth is in play, Shadow may use a Muster die as an Army
    # die (move pair / attack).  Reset at start of each turn.
    sh_mouth_muster_used_this_turn: bool = False
    # Action-token usage (bid compensation; user ruling 2026-07-03):
    # tokens are single-use and at most ONE token may be used per turn.
    fp_token_used_this_turn: bool = False
    sh_token_used_this_turn: bool = False

    # Dice rolled tracking (for recovery button)
    dice_rolled: bool = False

    # Special Characters
    # Shadow minions
    witch_king: Optional[ShadowMinion] = None
    saruman: Optional[ShadowMinion] = None
    mouth_of_sauron: Optional[ShadowMinion] = None

    # Free Peoples upgrades
    gandalf_the_white: Optional[GandalfTheWhite] = None
    aragorn_crowned: Optional[Aragorn] = None

    # Gollum becomes guide when all companions leave fellowship
    gollum: Optional[Companion] = None

    # Track bonus dice from special characters (calculated at turn start)
    fp_bonus_dice: int = 0
    sh_bonus_dice: int = 0

    # Persistent cards on the table
    persistent_cards: PersistentCardState = field(default_factory=PersistentCardState)

    # VESTIGIAL (always []): kept only because the encoder occupies a frozen
    # dim for it - remove field + dim together at the next encoding bump.
    pending_hunt_tiles: List[str] = field(default_factory=list)

    # Pending multi-step die actions (army double-move, muster double-recruit)
    pending_army_move_die: Optional[int] = None   # die_idx for first army move
    pending_army_move_from: Optional[str] = None   # region key of first move (can't reuse)
    pending_army_move_side: Optional[Side] = None  # which side is doing the double-move
    # Own units already at the first move's destination BEFORE it arrived —
    # those figures did NOT move and MAY move out with the second half
    # (rulebook p.25 forbids moving the same FIGURE twice, not the region):
    # Aggregate counts plus immutable 'figures' nation/kind/character inventory
    # and the Army-die pair's 'waived' payment flag, or None.
    pending_army_move_preexist: Optional[dict] = None
    pending_muster_die: Optional[int] = None       # die_idx for first muster recruit
    pending_muster_first_type: Optional[str] = None  # 'regular' or 'leader' (first recruit)
    pending_muster_region: Optional[str] = None    # settlement of first recruit (must differ)
    pending_muster_side: Optional[Side] = None     # which side is doing the double-recruit

    # FP casualty box.  FP casualties never return EXCEPT the one
    # sanctioned exit: an elite-downgrade replacement Regular is drawn
    # CASUALTY-BOX first (combat's ['regular'] -= 1 sites).  Also caps
    # muster availability (max - deployed - fp_casualties).
    # Dict[Nation, Dict[str, int]] e.g. {Nation.GONDOR: {'regular': 2, 'elite': 1, 'leader': 0}}
    fp_casualties: Dict[Nation, Dict[str, int]] = field(default_factory=dict)

    # Pending multi-phase combat (siege/retreat decisions)
    pending_combat: Optional[dict] = None
    # A Wormtongue-blocked Rohan reaction is reconsidered at action end:
    # 1 = activation only; 2 = activation and a still-owed advance.
    _wormtongue_attack_pending: Optional[int] = None
    # Pending hunt sacrifice / Gollum reveal decision
    pending_hunt: Optional[dict] = None
    # Pending guide change (tie-breaking)
    pending_guide_change: Optional[dict] = None
    # An Event-funded FP card may earn Gandalf's optional same-deck draw
    # only after the entire action and its first hand-limit discard settle.
    # Carries source_card_id/deck/die_face; the later choice is a card option.
    _gandalf_draw_pending: Optional[dict] = None
    # Strongholds currently under siege (set of sanitized region keys)
    besieged_strongholds: set = field(default_factory=set)

    # Mordor inactivity penalty: tracks whether Fellowship moved or hid this turn
    # If Fellowship is on Mordor Track and did NOT move or hide, take 1 corruption at end of turn
    fellowship_acted_in_mordor: bool = False


# ---------------------------------------------------------------------------
# Standard setup helper
# ---------------------------------------------------------------------------

def setup_initial_game(b: "board.Board") -> GameState:
    """Return a fully-populated `GameState` in turn-one setup."""

    # ------------------------------ Armies -----------------------------
    armies: Dict[str, Army] = {}

    def region(name: str):
        return b.region(name)

    def add(r: str, nat: Nation, spec: Dict[UnitKind, int],
            *, leaders: int = 0, nazgul: int = 0):
        army = armies.setdefault(r, Army(region(r)))
        for kind, qty in spec.items():
            for _ in range(qty):
                army.add(Unit(nat, kind))
        army.add(*(Leader(nat) for _ in range(leaders)))
        army.add(*(Nazgul(nat) for _ in range(nazgul)))

    # --------- Free Peoples forces (rulebook setup) --------------------
    add("Erebor",        Nation.DWARVES, {UnitKind.REGULAR: 1, UnitKind.ELITE: 2}, leaders=1)
    add("Ered Luin",     Nation.DWARVES, {UnitKind.REGULAR: 1})
    add("Iron Hills",    Nation.DWARVES, {UnitKind.REGULAR: 1})

    add("Grey Havens",   Nation.ELVES, {UnitKind.REGULAR: 1, UnitKind.ELITE: 1}, leaders=1)
    add("Rivendell",     Nation.ELVES, {UnitKind.ELITE: 2}, leaders=1)
    add("Woodland Realm",Nation.ELVES, {UnitKind.REGULAR: 1, UnitKind.ELITE: 1}, leaders=1)
    add("Lórien",        Nation.ELVES, {UnitKind.REGULAR: 1, UnitKind.ELITE: 2}, leaders=1)

    add("Minas Tirith",  Nation.GONDOR, {UnitKind.REGULAR: 3, UnitKind.ELITE: 1}, leaders=1)
    add("Dol Amroth",    Nation.GONDOR, {UnitKind.REGULAR: 3})
    add("Osgiliath",     Nation.GONDOR, {UnitKind.REGULAR: 2})
    add("Pelargir",      Nation.GONDOR, {UnitKind.REGULAR: 1})

    add("Bree",          Nation.THE_NORTH, {UnitKind.REGULAR: 1})
    add("Carrock",       Nation.THE_NORTH, {UnitKind.REGULAR: 1})
    add("Dale",          Nation.THE_NORTH, {UnitKind.REGULAR: 1}, leaders=1)
    add("North Downs",   Nation.THE_NORTH, {UnitKind.ELITE: 1})
    add("The Shire",     Nation.THE_NORTH, {UnitKind.REGULAR: 1})

    add("Edoras",        Nation.ROHAN, {UnitKind.REGULAR: 1, UnitKind.ELITE: 1})
    add("Fords of Isen", Nation.ROHAN, {UnitKind.REGULAR: 2}, leaders=1)
    add("Helm's Deep",   Nation.ROHAN, {UnitKind.REGULAR: 1})

    # --------- Shadow forces (unchanged) -------------------------------
    add("Orthanc",        Nation.ISENGARD, {UnitKind.REGULAR: 4, UnitKind.ELITE: 1})
    add("North Dunland",  Nation.ISENGARD, {UnitKind.REGULAR: 1})
    add("South Dunland",  Nation.ISENGARD, {UnitKind.REGULAR: 1})

    add("Barad-Dûr",      Nation.SAURON, {UnitKind.REGULAR: 4, UnitKind.ELITE: 1}, nazgul=1)
    add("Dol Guldur",     Nation.SAURON, {UnitKind.REGULAR: 5, UnitKind.ELITE: 1}, nazgul=1)
    add("Gorgoroth",      Nation.SAURON, {UnitKind.REGULAR: 3})
    add("Minas Morgul",   Nation.SAURON, {UnitKind.REGULAR: 5}, nazgul=1)
    add("Moria",          Nation.SAURON, {UnitKind.REGULAR: 2})
    add("Mount Gundabad", Nation.SAURON, {UnitKind.REGULAR: 2})
    add("Nurn",           Nation.SAURON, {UnitKind.REGULAR: 2})
    add("Morannon",       Nation.SAURON, {UnitKind.REGULAR: 5}, nazgul=1)

    add("Far Harad",      Nation.SOUTHRON_EASTERLINGS, {UnitKind.REGULAR: 3, UnitKind.ELITE: 1})
    add("Near Harad",     Nation.SOUTHRON_EASTERLINGS, {UnitKind.REGULAR: 3, UnitKind.ELITE: 1})
    add("North Rhûn",     Nation.SOUTHRON_EASTERLINGS, {UnitKind.REGULAR: 2})
    add("South Rhûn",     Nation.SOUTHRON_EASTERLINGS, {UnitKind.REGULAR: 3, UnitKind.ELITE: 1})
    add("Umbar",          Nation.SOUTHRON_EASTERLINGS, {UnitKind.REGULAR: 3})

    # ---------------- Fellowship & tracks ------------------------------
    start_region = b.region("Rivendell")   # rulebook setup: Fellowship
                                           # starts in Rivendell

    # Create companions with full abilities
    gandalf_grey = Companion(
        name="Gandalf the Grey", level=3, leadership=1,
        captain_of_the_west=True,
        guide_ability="palantir_draw",  # annotation only: the Gandalf-guide
        # draw is implemented BY NAME (ai.py play_event, gui; USER 2026-07-08)
        activates_nations=[],  # Empty = activates ANY nation
        activation_regions=[],  # Empty = anywhere
    )
    strider = Companion(
        name="Strider", level=3, leadership=1,
        captain_of_the_west=True,
        guide_ability="hide_any_die",  # Can hide with any die face
        activates_nations=[Nation.THE_NORTH],
        activation_regions=["dale", "the shire"],  # Only activates North in these regions
    )
    legolas = Companion(
        name="Legolas", level=2, leadership=1,
        captain_of_the_west=True,
        can_muster_nation=Nation.ELVES,  # Can muster elves with any die
        muster_regions=["grey havens", "rivendell", "lorien", "woodland realm"],  # Elven strongholds
    )
    gimli = Companion(
        name="Gimli", level=2, leadership=1,
        captain_of_the_west=True,
        can_muster_nation=Nation.DWARVES,
        muster_regions=["erebor"],
        activates_nations=[Nation.DWARVES],
        activation_regions=["erebor"],
    )
    boromir = Companion(
        name="Boromir", level=2, leadership=1,
        captain_of_the_west=True,
        can_muster_nation=Nation.GONDOR,  # Muster Gondor 1 step toward war
        muster_regions=["minas tirith", "pelargir", "dol amroth"],
        activates_nations=[Nation.GONDOR],
        activation_regions=["minas tirith", "pelargir", "dol amroth"],
    )
    merry = Companion(
        name="Meriadoc", level=1, leadership=1,
        guide_ability="sacrifice_reduce_hunt",  # Separate to reduce hunt damage by 1
        activates_nations=[],  # Activates any nation
    )
    pippin = Companion(
        name="Peregrin", level=1, leadership=1,
        guide_ability="sacrifice_reduce_hunt",  # Same as Merry
        activates_nations=[],  # Activates any nation
    )

    companions = [gandalf_grey, strider, legolas, gimli, boromir, merry, pippin]

    # Gollum - becomes guide when all companions leave fellowship
    gollum = Companion(
        name="Gollum", level=0, leadership=0,
        guide_ability="no_reveal_on_damage",  # Reveal tiles don't reveal fellowship
    )

    fellowship = Fellowship(
        companions=companions,
        guide=companions[0],
        region=start_region,
    )

    # ---------------- Special Characters (not in play at start) -------
    witch_king = ShadowMinion(
        name="Witch King", level=99, leadership=2,
        in_play=False,
        infinite_movement=True,
        combat_card_draw=True,
    )
    saruman = ShadowMinion(
        name="Saruman", level=0, leadership=1,
        in_play=False,
        immobile=True,  # Cannot leave Orthanc
        isengard_elite_leadership=True,  # All Isengard elites get +1 leadership
    )
    mouth = ShadowMinion(
        name="Mouth of Sauron", level=3, leadership=2,
        in_play=False,
    )
    gandalf_white = GandalfTheWhite(in_play=False)
    aragorn = Aragorn(crowned=False)

    political = PoliticalTrack(
        boxes={
            Nation.ROHAN: 0,  Nation.THE_NORTH: 0,  Nation.ELVES: 0,
            Nation.DWARVES: 0, Nation.GONDOR: 1,
            Nation.SOUTHRON_EASTERLINGS: 1, Nation.ISENGARD: 2, Nation.SAURON: 2,
        },
        active={
            Nation.ROHAN: False, Nation.THE_NORTH: False,
            Nation.ELVES: True,  Nation.DWARVES: False, Nation.GONDOR: False,
            Nation.SOUTHRON_EASTERLINGS: True, Nation.ISENGARD: True, Nation.SAURON: True,
        },
    )

    # ---------------- VP, dice, decks, etc. ----------------------------
    vp = VictoryPoints()
    dice = {
        Side.FREE_PEOPLES: [ActionDie(FP_FACES, Side.FREE_PEOPLES) for _ in range(4)],
        Side.SHADOW:      [ActionDie(SH_FACES, Side.SHADOW)        for _ in range(7)],
    }
    decks = DeckStatus(24, 24, 24, 24)
    hands  = {Side.FREE_PEOPLES: PlayerInfo(), Side.SHADOW: PlayerInfo()}
    discard_face_up = {k: [] for k in ("fp_character", "fp_strategy",
                                       "sh_character", "sh_strategy")}

    # ---------------- Assemble GameState --------------------------------
    return GameState(
        board=b,
        armies=armies,
        enemy_controlled={Side.FREE_PEOPLES: set(), Side.SHADOW: set()},
        fellowship=fellowship,
        political=political,
        vp=vp,
        dice=dice,
        turn=1,
        # ---- Hunt ------------------------------------------------------
        hunt_pool=_initial_hunt_pool(),
        hunt_discard=[],
        hunt_removed=[],
        hunt_box=[],
        # ---- Cards -----------------------------------------------------
        decks=decks,
        hands=hands,
        discard_face_up=discard_face_up,
        fp_elven_rings=3, fp_dwarven_rings=0,
        sh_elven_rings=0, sh_dwarven_rings=0, sh_action_tokens=0,
        # ---- Special Characters ------------------------------------------
        witch_king=witch_king,
        saruman=saruman,
        mouth_of_sauron=mouth,
        gandalf_the_white=gandalf_white,
        aragorn_crowned=aragorn,
        gollum=gollum,
        fp_bonus_dice=0,
        sh_bonus_dice=0,
    )

