# cards.py - Card loading and registry for War of the Ring
"""Load and parse cards from CSV, providing a registry for combat and event effects."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from game_state import Side, CombatCard


_HOBBIT_IDENTITIES = frozenset({
    'meriadoc', 'meriadoc brandybuck', 'peregrin', 'peregrin took',
    'frodo', 'frodo baggins', 'sam', 'samwise', 'sam gamgee', 'samwise gamgee',
})


def is_hobbit_identity(companion_or_name) -> bool:
    """Recognize explicit Hobbit names without substring matching.

    Accept either a name or a figure-like object. Participation remains the
    caller's responsibility: a Hobbit hidden in the Fellowship is not an army
    participant merely because its identity matches.
    """
    name = (companion_or_name if isinstance(companion_or_name, str)
            else getattr(companion_or_name, 'name', None))
    return isinstance(name, str) and name.strip().casefold() in _HOBBIT_IDENTITIES


# ---------------------------------------------------------------------------
# Full Card Definition (event + combat)
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """Complete card definition with both event and combat properties."""
    card_id: str
    side: Side
    deck: str  # "Character" or "Strategy"
    card_type: str  # "Character", "Army", "Muster"
    card_name: str

    # Event properties
    event_req: str
    event_effect: str

    # Combat properties
    combat_name: str
    combat_value: int
    combat_req: str
    combat_effect: str

    # Parsed combat requirements
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
    requires_hobbit: bool = False
    # named-character gate ('Gandalf is in the battle'); empty = none
    requires_character_in_battle: List[str] = field(default_factory=list)

    def to_combat_card(self) -> CombatCard:
        """Convert to a CombatCard for use in battle.  CombatCard is
        RESOLUTION-only: requires_hobbit and
        requires_character_in_battle are play-legality gates checked
        on the full Card (can_play_combat_card) and deliberately have
        no CombatCard fields."""
        return CombatCard(
            card_id=self.card_id,
            side=self.side,
            deck=self.deck,
            combat_name=self.combat_name,
            combat_value=self.combat_value,
            combat_req=self.combat_req,
            combat_effect=self.combat_effect,
            requires_companion=self.requires_companion,
            requires_nazgul=self.requires_nazgul,
            requires_elite=self.requires_elite,
            requires_leader=self.requires_leader,
            min_nazgul_leadership=self.min_nazgul_leadership,
            requires_fellowship_region=self.requires_fellowship_region,
            requires_stronghold=self.requires_stronghold,
            requires_fp_nation=self.requires_fp_nation,
            region_restrictions=self.region_restrictions.copy(),
            requires_isengard_unit=self.requires_isengard_unit,
            requires_se_elite=self.requires_se_elite,
            requires_shadow_elite=self.requires_shadow_elite,
            requires_fp_elite=self.requires_fp_elite,
        )


# ---------------------------------------------------------------------------
# Requirement Parsing
# ---------------------------------------------------------------------------

def _parse_combat_requirements(req_text: str, card: Card) -> None:
    """Parse combat requirement text and set boolean flags on card."""
    if not req_text:
        return

    req_lower = req_text.lower()

    # Named-character requirements ("Gandalf is in the battle",
    # "Strider/Aragorn is in the battle")
    if "gandalf is in the battle" in req_lower:
        card.requires_character_in_battle = ["gandalf"]
    elif "strider/aragorn is in the battle" in req_lower or \
            "strider or aragorn" in req_lower:
        card.requires_character_in_battle = ["strider", "aragorn"]

    # "a Leader or a Companion" is an OR requirement covered by the leader
    # check (which accepts companions as leaders); a strict companion flag
    # would wrongly block Sudden Strike / Paths of the Woses.
    if "companion" in req_lower:
        if not re.search(r"leader or (a )?companion", req_lower):
            card.requires_companion = True

    # Hobbit requirement
    if "hobbit" in req_lower:
        card.requires_hobbit = True
        card.requires_companion = True  # Hobbits are companions

    # Nazgul requirements
    if "nazgûl" in req_lower or "nazgul" in req_lower:
        card.requires_nazgul = True

    # Leadership requirements
    match = re.search(r"nazgûl leadership is (\d+)", req_lower)
    if match:
        card.min_nazgul_leadership = int(match.group(1))
        card.requires_nazgul = True

    # Elite requirements
    if "elite" in req_lower:
        card.requires_elite = True
        if "free peoples elite" in req_lower:
            card.requires_fp_elite = True
        elif "shadow elite" in req_lower:
            card.requires_shadow_elite = True
        elif "southrons" in req_lower or "easterlings" in req_lower:
            card.requires_se_elite = True

    # Leader requirements ('leadership' has no boundary after
    # 'leader', so the Nazgul-Leadership cards no longer mis-flag)
    if re.search(r"\bleader\b", req_lower):
        card.requires_leader = True

    # Fellowship region
    if "same region as the fellowship" in req_lower:
        card.requires_fellowship_region = True

    # Stronghold
    if "stronghold" in req_lower:
        card.requires_stronghold = True

    # Free Peoples Nation
    if "free peoples nation" in req_lower:
        card.requires_fp_nation = True

    # Isengard
    if "isengard" in req_lower:
        card.requires_isengard_unit = True

    # Region restrictions
    regions = []
    if "rohan" in req_lower:
        regions.append("rohan")
    if "fangorn" in req_lower:
        regions.append("fangorn")
    if "orthanc" in req_lower:
        regions.append("orthanc")
    if "moria" in req_lower:
        regions.append("moria")
    card.region_restrictions = regions


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def load_cards(csv_path: str) -> Dict[str, Card]:
    """Load all cards from CSV file.

    The CSV format has each row wrapped in outer quotes, with "" for escaped quotes.
    Fields are comma-separated. We find the combat_val (a number) to anchor parsing.
    """
    cards: Dict[str, Card] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Skip header line
        if i == 0:
            continue

        # Each line is wrapped in outer quotes - strip them
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        # Replace escaped quotes "" with a placeholder
        QUOTE_PLACEHOLDER = "\x00"
        line = line.replace('""', QUOTE_PLACEHOLDER)

        # Split by comma
        raw_fields = line.split(',')

        # Find combat_val - it's a single digit number (0-7) that appears
        # near the end of the field list. Work backwards from end.
        # Structure: side, deck, type, name, event_req, event_effect,
        #            combat_name, combat_val, combat_req, combat_effect

        # Find indices of numeric single values (potential combat_val)
        numeric_indices = []
        for j, f in enumerate(raw_fields):
            f_stripped = f.strip()
            if f_stripped.isdigit() and len(f_stripped) == 1:
                numeric_indices.append(j)

        # combat_val should have exactly 2 fields after it (combat_req, combat_effect)
        # but combat_effect might have commas too, so we need the LAST numeric
        # that has at least 1 field after it
        combat_val_idx = None
        for idx in reversed(numeric_indices):
            fields_after = len(raw_fields) - idx - 1
            if fields_after >= 1:  # At least combat_req and possibly more for combat_effect
                combat_val_idx = idx
                break

        if combat_val_idx is None:
            # Fallback: just split into 10 parts as best we can
            fields = raw_fields[:10] if len(raw_fields) >= 10 else raw_fields
        else:
            # Now reconstruct the 10 fields:
            # Fields 0-3: first 4 raw fields (side, deck, type, name)
            # Field 4-5: everything from raw[4] to raw[combat_val_idx-2]
            # Field 6: raw[combat_val_idx-1] (combat_name)
            # Field 7: raw[combat_val_idx] (combat_val)
            # Field 8-9: everything after combat_val

            fields = []
            # First 4 fields
            fields.extend(raw_fields[:4])

            # event_req and event_effect: fields 4 through combat_val_idx-2
            event_fields = raw_fields[4:combat_val_idx-1]
            if len(event_fields) >= 2:
                # Split roughly in half, but event_req is usually shorter
                # Look for a period followed by capital letter as split point
                # Find good split point - usually after first sentence
                split_point = None
                for k, ef in enumerate(event_fields):
                    if ef.endswith('.') and k < len(event_fields) - 1:
                        split_point = k + 1
                        break
                if split_point:
                    fields.append(','.join(event_fields[:split_point]))
                    fields.append(','.join(event_fields[split_point:]))
                else:
                    # Just take first as event_req, rest as event_effect
                    fields.append(event_fields[0])
                    fields.append(','.join(event_fields[1:]))
            elif len(event_fields) == 1:
                fields.append(event_fields[0])
                fields.append('')
            else:
                fields.append('')
                fields.append('')

            # combat_name
            fields.append(raw_fields[combat_val_idx - 1])

            # combat_val
            fields.append(raw_fields[combat_val_idx])

            # combat_req and combat_effect: everything after combat_val
            combat_fields = raw_fields[combat_val_idx + 1:]
            if len(combat_fields) >= 2:
                # combat_req usually ends with a period
                split_point = None
                for k, cf in enumerate(combat_fields[:-1]):
                    if cf.endswith('.'):
                        split_point = k + 1
                        break
                if split_point:
                    fields.append(','.join(combat_fields[:split_point]))
                    fields.append(','.join(combat_fields[split_point:]))
                else:
                    fields.append(combat_fields[0])
                    fields.append(','.join(combat_fields[1:]))
            elif len(combat_fields) == 1:
                fields.append(combat_fields[0])
                fields.append('')
            else:
                fields.append('')
                fields.append('')

        # Restore the quotes in each field
        fields = [f.replace(QUOTE_PLACEHOLDER, '"') for f in fields]

        if len(fields) < 10:
            raise ValueError('card_list.csv row %d: cannot reconstruct '
                             '10 fields (%r)' % (i + 1, line[:60]))

        side_str = fields[0].strip()
        deck = fields[1].strip()
        card_type = fields[2].strip()
        card_name = fields[3].strip()
        event_req = fields[4].strip()
        event_effect = fields[5].strip()
        combat_name = fields[6]
        combat_val_str = fields[7]
        combat_req = fields[8].strip()
        combat_effect = fields[9].strip()

        # ---------------------------------------------------------------------------
        # Fix requirement/effect parsing issues
        # ---------------------------------------------------------------------------
        # A REQUIREMENT is ONLY a condition for playing the card: "Play if [condition]"
        # "Play on the table" is NOT a requirement - it's an instruction
        # Anything else (Roll, Heal, Move, Recruit, Change, etc.) is an EFFECT
        # ---------------------------------------------------------------------------

        # Combine event_req and event_effect, then re-parse.  The join MUST
        # guarantee a sentence boundary (HIGH-2): a req missing its period
        # made the non-greedy split swallow the effect text.
        def _dot(s):
            s = s.strip()
            return s if (not s or s.endswith(('.', '!', '?'))) \
                else s + '.'
        full_event_text = ""
        if event_req and event_effect:
            full_event_text = _dot(event_req) + " " + event_effect
        elif event_req:
            full_event_text = event_req
        elif event_effect:
            full_event_text = event_effect

        # Reset and re-parse
        event_req = ""
        event_effect = full_event_text

        if full_event_text:
            # Case 1: "Play if [condition]." - extract condition as requirement
            match = re.match(r"Play if (.+?\.)\s*(.*)", full_event_text, re.I | re.DOTALL)
            if match:
                event_req = match.group(1).strip().rstrip('.')
                event_effect = match.group(2).strip() if match.group(2) else ""
            else:
                # Case 2: "Play on the table if [condition]." - extract condition
                match = re.match(r"Play on the table if (.+?\.)\s*(.*)", full_event_text, re.I | re.DOTALL)
                if match:
                    event_req = match.group(1).strip().rstrip('.')
                    remaining = match.group(2).strip() if match.group(2) else ""
                    event_effect = ("Play on the table. " + remaining).strip()
                else:
                    # Case 3: "Play on the table." alone or "Play on [target]." - no requirement
                    # Case 4: Anything else (Roll, Heal, Move, etc.) - no requirement
                    event_req = ""
                    event_effect = full_event_text

        # COMBAT: same join (same sentence-boundary guarantee)
        full_combat_text = ""
        if combat_req and combat_effect:
            full_combat_text = _dot(combat_req) + " " + combat_effect
        elif combat_req:
            full_combat_text = combat_req
        elif combat_effect:
            full_combat_text = combat_effect

        combat_req = ""
        combat_effect = full_combat_text

        if full_combat_text:
            # Combat cards only use "Play if [condition]." pattern
            match = re.match(r"Play if (.+?\.)\s*(.*)", full_combat_text, re.I | re.DOTALL)
            if match:
                combat_req = match.group(1).strip().rstrip('.')
                combat_effect = match.group(2).strip() if match.group(2) else ""
            else:
                combat_req = ""
                combat_effect = full_combat_text

        # Parse side (loud on unknown values: a silent SHADOW
        # default would flip a typo'd card's deck membership)
        try:
            side = {"Free Peoples": Side.FREE_PEOPLES,
                    "Shadow": Side.SHADOW}[side_str]
        except KeyError:
            raise ValueError('card_list.csv row %d: unknown side %r'
                             % (i + 1, side_str))

        # Parse combat value
        try:
            combat_value = int(combat_val_str) if combat_val_str else 0
        except ValueError:
            raise ValueError('card_list.csv row %d (%s): unparseable '
                             'combat value %r'
                             % (i + 1, card_name, combat_val_str))

        # Generate unique card ID
        card_id = f"{side.name}_{deck}_{card_name.replace(' ', '_')}"

        card = Card(
            card_id=card_id,
            side=side,
            deck=deck,
            card_type=card_type,
            card_name=card_name,
            event_req=event_req,
            event_effect=event_effect,
            combat_name=combat_name,
            combat_value=combat_value,
            combat_req=combat_req,
            combat_effect=combat_effect,
        )

        # Parse requirements
        _parse_combat_requirements(combat_req, card)

        if card_id in cards:
            raise ValueError('duplicate card id ' + card_id)
        cards[card_id] = card

    return cards


# ---------------------------------------------------------------------------
# Combat Effect Categories
# ---------------------------------------------------------------------------

# Map combat effect text patterns to effect categories
EFFECT_PATTERNS = {
    # Dice modifiers
    "add_1_dice": re.compile(r"add 1 to all dice", re.I),
    "add_2_dice": re.compile(r"add 2 to all dice", re.I),
    "subtract_1_dice": re.compile(r"subtract 1 from all dice", re.I),
    "both_add_1": re.compile(r"both armies add 1", re.I),
    "both_add_2": re.compile(r"both armies add 2", re.I),

    # Dice count
    "max_2_dice": re.compile(r"maximum of two dice", re.I),
    "max_3_dice": re.compile(r"maximum of three dice", re.I),
    "reduce_dice_per_companion": re.compile(r"rolls one die less.*for each companion", re.I),
    "reduce_dice_forfeit": re.compile(r"forfeit one or more points.*rolls one die less", re.I),

    # Cancel
    "cancel_card": re.compile(r"cancel the effects of the combat card", re.I),
    "cancel_reroll": re.compile(r"leader re-roll is cancelled", re.I),

    # Pre-combat
    "sudden_strike": re.compile(r"before the combat roll.*additional attack.*leadership", re.I),
    "charge": re.compile(r"before the combat roll.*additional attack.*elite units", re.I),
    "durins_bane": re.compile(r"before the combat roll.*additional attack.*three combat dice", re.I),

    # Auto-hit
    "mighty_attack": re.compile(r"forfeit the leadership.*automatically change.*missed.*to a hit", re.I),
    "anduril": re.compile(r"forfeit.*aragorn.*leadership.*two missed", re.I),

    # Bonus hits
    "no_quarter": re.compile(r"scores? at least one hit.*score one additional hit", re.I),
    "great_host": re.compile(r"twice as many.*score one automatic hit", re.I),
    "mumakil": re.compile(r"more total hits than.*score one additional hit", re.I),
    "nameless_wood": re.compile(r"score at least one hit.*score two additional hits", re.I),

    # Post-reroll attacks
    "onslaught": re.compile(r"after removing casualties.*inflict.*additional hits.*roll one die.*4\+", re.I),
    "we_come_to_kill": re.compile(r"after removing casualties.*additional attack.*elite units.*5\+", re.I),

    # Casualty prevention
    "shield_wall": re.compile(r"two or more hits.*cancel one hit", re.I),
    "heroic_death": re.compile(r"eliminate a leader to cancel one hit.*eliminate one companion to cancel", re.I),

    # Special elimination
    "black_breath": re.compile(r"eliminate one free peoples leader.*alternatively.*eliminate a companion", re.I),
    "fateful_strike": re.compile(r"eliminate a nazgûl", re.I),

    # Self-damage
    "relentless_assault": re.compile(r"before the combat roll.*inflict.*hits against your units.*add 1", re.I),

    # Confusion
    "confusion": re.compile(r"result of.*1.*scores one hit against the shadow army", re.I),

    # Retreat
    "scouts": re.compile(r"before the combat roll.*retreat", re.I),

    # Words of Power (cancel companion abilities)
    "words_of_power": re.compile(r"companion.*leadership and special abilities are cancelled", re.I),
}


def categorize_effect(effect_text: str) -> List[str]:
    """Return list of effect category names that match this effect text."""
    categories = []
    for name, pattern in EFFECT_PATTERNS.items():
        if pattern.search(effect_text):
            categories.append(name)
    return categories


# ---------------------------------------------------------------------------
# Global Card Registry (lazy loaded)
# ---------------------------------------------------------------------------

_CARDS: Optional[Dict[str, Card]] = None


def get_cards() -> Dict[str, Card]:
    """Get the global card registry, loading from CSV if needed."""
    global _CARDS
    if _CARDS is None:
        import os
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _data_dir = os.path.join(os.path.dirname(_src_dir), 'data')
        csv_path = os.path.join(_data_dir, "card_list.csv")
        _CARDS = load_cards(csv_path)
    return _CARDS


def get_card(card_id: str) -> Optional[Card]:
    """Get a specific card by ID."""
    return get_cards().get(card_id)


def get_cards_by_deck(side: Side, deck: str) -> List[Card]:
    """Get all cards from a specific deck."""
    return [c for c in get_cards().values() if c.side == side and c.deck == deck]


# ---------------------------------------------------------------------------
# Card Validation for Combat
# ---------------------------------------------------------------------------

def can_play_combat_card(card: Card, combat_state: "CombatState",
                         is_attacker: bool,
                         state: "GameState" = None) -> Tuple[bool, str]:
    """Check if a card can legally be played in the current combat state.
    `state` (the full GameState) enables the fellowship-region / FP-nation
    / region-restriction requirements (USER audit 2026-07-08)."""
    # Get the relevant army and state info
    if is_attacker:
        army = combat_state.attacker_army
        side = combat_state.attacker_side
    else:
        army = combat_state.defender_army
        side = combat_state.defender_side

    # Card must match side
    if card.side != side:
        return False, "Card belongs to opponent"

    # Almanac p41: Scouts follows defender retreat rules in a field battle.
    if (card.combat_name or '').strip().lower() == 'scouts' and (
            is_attacker or combat_state.is_siege):
        return False, "Scouts requires defending in a field battle"

    # Check companion requirement
    if card.requires_companion:
        if not combat_state.companions_in_battle:
            return False, "No Companion in battle"

    # Check Hobbit requirement
    if card.requires_hobbit:
        if not any(is_hobbit_identity(c) for c in combat_state.companions_in_battle):
            return False, "No Hobbit in battle"

    # Check Nazgul requirement
    if card.requires_nazgul:
        if combat_state.nazgul_in_battle < 1:
            return False, "No Nazgul in battle"

    # Check minimum Nazgul leadership.  USER RULING 2026-08-03
    # (initiative spec, case I): Gandalf the White's White Rider
    # negates ALL Nazgul Leadership for the whole battle, so cards
    # REQUIRING Nazgul Leadership (Cruel as Death / They are Terrible
    # / Dread and Despair) may NOT be played at all — a play-legality
    # restriction, unlike Daring Defiance's playable-but-inert case.
    # (requires_nazgul PRESENCE above stays on the raw count so Words
    # of Power / Black Breath remain playable under White Rider.)
    if card.min_nazgul_leadership > 0:
        _nl = (0 if getattr(combat_state, '_shine_companion', None)
               is not None else combat_state.nazgul_in_battle)
        if _nl < card.min_nazgul_leadership:
            return False, f"Need {card.min_nazgul_leadership} Nazgul Leadership"

    # Check elite requirement
    if card.requires_elite or card.requires_fp_elite or card.requires_shadow_elite:
        from game_state import UnitKind
        has_elite = any(u.kind == UnitKind.ELITE for u in army.units)
        if not has_elite:
            return False, "No Elite unit in battle"

    # Check leader requirement
    if card.requires_leader:
        if not army.leaders and not combat_state.companions_in_battle:
            return False, "No Leader in battle"

    # Named character in battle (Gandalf / Strider-Aragorn class).
    # companions_in_battle is BOARD-ONLY (USER 1.209 / CL-228:
    # in-Fellowship companions are INVISIBLE to combat and never
    # satisfy any gate); it holds both armies' separated companions.
    _req_chars = getattr(card, 'requires_character_in_battle', None)
    if _req_chars:
        _names = ' '.join(
            str(getattr(c, 'name', '')).lower()
            for c in combat_state.companions_in_battle)
        if not any(rc in _names for rc in _req_chars):
            return False, ("Requires " + "/".join(_req_chars)
                           + " in the battle")

    # Southrons & Easterlings Elite (Shadows Gather / The Shadow
    # Lengthens): the generic elite check passes on ANY elite — this
    # one must be an S&E elite in my army.
    if getattr(card, 'requires_se_elite', False):
        from game_state import UnitKind, Nation
        if not any(u.kind == UnitKind.ELITE
                   and u.nation == Nation.SOUTHRON_EASTERLINGS
                   for u in army.units):
            return False, "No Southrons & Easterlings Elite in battle"

    # Isengard unit in battle (Dreadful Spells / Denethor's Folly /
    # Threats and Promises; the stronghold half of the requirement is
    # covered by requires_stronghold).
    if getattr(card, 'requires_isengard_unit', False):
        from game_state import Nation
        if not any(u.nation == Nation.ISENGARD for u in army.units):
            return False, "No Isengard unit in battle"

    # "Defending Army is in a Stronghold" means inside a besieged
    # stronghold, not a field battle or sortie in a stronghold region.
    if card.requires_stronghold:
        if not combat_state.is_siege:
            return False, "Defending army is not inside a besieged Stronghold"

    # Fellowship-region requirement ("the defending Army is in the same
    # region as the Fellowship") — Phial/Elven Cloaks/It is a Gift class
    if getattr(card, 'requires_fellowship_region', False) and state is not None:
        from board import sanitize as _s
        import game_logic as _gl
        # Rulebook p43: the Mordor Track is not part of any map region.
        # region retains the last gateway for bookkeeping after entry.
        if state.fellowship.mordor_entered:
            return False, "The Fellowship is on the Mordor Track, not in a map region"
        if _s(str(combat_state.defender_region)) != \
                _s(str(_gl.fellowship_region(state))):
            return False, "Defender not in the Fellowship's region"

    # FP-nation requirement ("the defending Army is inside the borders
    # of a Free Peoples Nation")
    if getattr(card, 'requires_fp_nation', False) and state is not None:
        import game_logic as _gl
        if not _gl.is_fp_nation_region(str(combat_state.defender_region)):
            return False, "Defender not inside a Free Peoples Nation"

    # Check region restrictions (Ents class: Rohan/Fangorn/Orthanc;
    # Balrog: within two regions of Moria)
    if card.region_restrictions and state is not None:
        from board import sanitize as _s
        import board as _b
        dreg = _s(str(combat_state.defender_region))
        req_low = (card.combat_req or '').lower()
        ok_region = False
        if 'moria' in card.region_restrictions and \
                'within two regions' in req_low:
            from collections import deque
            seen, q = {'moria'}, deque([('moria', 0)])
            near = set()
            while q:
                r, d = q.popleft()
                near.add(r)
                if d >= 2:
                    continue
                for adj in _b.adj.get(r, ()):
                    a = _s(adj)
                    if a not in seen:
                        seen.add(a)
                        q.append((a, d + 1))
            ok_region = dreg in near
        else:
            if 'fangorn' in card.region_restrictions and dreg == 'fangorn':
                ok_region = True
            if 'orthanc' in card.region_restrictions and dreg == 'orthanc':
                ok_region = True
            if 'rohan' in card.region_restrictions:
                meta = _b.regions.get(dreg, {})
                if _b.fid_to_name.get(meta.get('parent_id')) == 'rohan':
                    ok_region = True
            if 'moria' in card.region_restrictions and dreg == 'moria':
                ok_region = True
        if not ok_region:
            return False, "Defending region does not satisfy the card req"

    return True, "OK"
