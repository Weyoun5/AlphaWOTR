"""Card-identity decoding from VASSAL draw events (commitment brute-force).

This is the ONLY sanctioned production import from the legacy raw-log
parser family: hidden-card identities cannot be carried by the rendered
board text, so the rendered pipeline decodes them here.  Moved verbatim
out of log_replay.py; log_replay re-exports these names so existing
importers keep working.
NOTE: rendered_extract.py should switch its `import log_replay as lr`
card-decoder uses to `import card_decoder` at its next touch.
"""

import hashlib
import re
from typing import Dict, List, Optional, Tuple

# Card databases - 0-indexed reversed alphabetical from messages.properties
# per deck type (the game numbers cards in reversed alphabetical order of
# base game card names).
FP_CHARACTER_CARDS = {
    0: "Wizard's Staff", 1: "We Prove Swifter", 2: "There is Another Way",
    3: "There and Back Again", 4: "The Ents Awake: Entmoot", 5: "The Grey Company",
    6: "The Eagles are Coming", 7: "Sméagol Helps Nice Master",
    8: "The Ents Awake: Treebeard", 9: "Phial of Galadriel",
    10: "Mithril Coat and Sting", 11: "Mirror of Galadriel", 12: "I Will Go Alone",
    13: "The Ents Awake: Huorns", 14: "House of the Stewards", 15: "Horn of Gondor",
    16: "Gwaihir the Windlord", 17: "Elven Rope", 18: "Elven Cloaks",
    19: "Dead Men of Dunharrow", 20: "Challenge of the King", 21: "Bilbo's Song",
    22: "Axe and Bow", 23: "Athelas",
}
FP_STRATEGY_CARDS = {
    0: "The Wisdom of Elrond", 1: "Through a Day and a Night", 2: "Thranduil's Archers",
    3: "The Spirit of Mordor", 4: "The Red Arrow", 5: "The Power of Tom Bombadil",
    6: "The Last Battle", 7: "Swords in Eriador", 8: "Riders of Theoden",
    9: "Paths of the Woses", 10: "King Brand's Men", 11: "Kindred of Glorfindel",
    12: "Imrahil of Dol Amroth", 13: "Help Unlooked For", 14: "Guards of the Citadel",
    15: "Grimbeorn the Old", 16: "Fear! Fire! Foes!", 17: "Faramir's Rangers",
    18: "Eomer, Son of Eomund", 19: "Dain Ironfoot's Guard", 20: "Cirdan's Ships",
    21: "Celeborn's Galadrim", 22: "Book of Mazarbul", 23: "A Power Too Great",
}
SA_CHARACTER_CARDS = {
    0: "Worn with Sorrow and Toil", 1: "Wormtongue", 2: "The Ringwraiths are Abroad",
    3: "The Ring is Mine!", 4: "The Palantir of Orthanc", 5: "The Nazgul Strike!",
    6: "The Lidless Eye", 7: "The Breaking of the Fellowship",
    8: "The Black Captain Commands", 9: "Shelob's Lair", 10: "Orc Patrol",
    11: "On, On They Went", 12: "Nazgul Search", 13: "Morgul Wound",
    14: "Lure of the Ring", 15: "Isildur's Bane",
    16: "Grond, Hammer of the Underworld", 17: "Give it to Uss!",
    18: "Foul Thing From the Deep", 19: "Flocks of Crebain", 20: "Dreadful Spells",
    21: "Cruel Weather", 22: "Candles of Corpses", 23: "A Balrog is Come!",
}
SA_STRATEGY_CARDS = {
    0: "Threats and Promises", 1: "The Shadow Lengthens", 2: "The Shadow is Moving",
    3: "The King is Revealed", 4: "The Fighting Uruk-Hai", 5: "The Day Without Dawn",
    6: "Corsairs of Umbar", 7: "Stormcrow",
    8: "Shadows on the Misty Mountains", 9: "Shadows Gather",
    10: "Return to Valinor", 11: "Return of the Witch-King",
    12: "Rage of the Dunlendings", 13: "Pits of Mordor",
    14: "Orcs Multiplying Again", 15: "Olog-Hai",
    16: "Mustering of a Long-planned War", 17: "Monsters Roused",
    18: "Many Kings to the Service of Mordor", 19: "Horde from the East",
    20: "Hill-Trolls", 21: "Half-Orcs and Goblin-Men", 22: "Denethor's Folly",
    23: "A New Power is Rising",
}


def sha1_hex(s: str) -> str:
    """Compute SHA1 hash and return uppercase hex string."""
    return hashlib.sha1(s.encode('utf-8')).hexdigest().upper()


# Piece ID type mapping derived from messages.properties alphabetical order.
# FP cards (pids 106-157) and SA cards (pids 158-209) are created in alphabetical
# order of card names. Types: C=Character, S=Strategy, X=Expansion.
_FP_PID_TYPES = (
    'S',  # 106: A Power Too Great
    'C',  # 107: Athelas
    'C',  # 108: Axe and Bow
    'C',  # 109: Bilbo's Song
    'S',  # 110: Book of Mazarbul
    'S',  # 111: Celeborn's Galadrim
    'C',  # 112: Challenge of the King
    'S',  # 113: Cirdan's Ships
    'S',  # 114: Dain Ironfoot's Guard
    'C',  # 115: Dead Men of Dunharrow
    'C',  # 116: Elven Cloaks
    'C',  # 117: Elven Rope
    'S',  # 118: Eomer, Son of Eomund
    'S',  # 119: Faramir's Rangers
    'S',  # 120: Fear! Fire! Foes!
    'S',  # 121: Grimbeorn the Old
    'S',  # 122: Guards of the Citadel
    'C',  # 123: Gwaihir the Windlord
    'S',  # 124: Help Unlooked For
    'C',  # 125: Horn of Gondor
    'C',  # 126: House of the Stewards
    'C',  # 127: The Ents Awake: Huorns
    'C',  # 128: I Will Go Alone
    'S',  # 129: Imrahil of Dol Amroth
    'S',  # 130: Kindred of Glorfindel
    'S',  # 131: King Brand's Men
    'C',  # 132: Mirror of Galadriel
    'C',  # 133: Mithril Coat and Sting
    'S',  # 134: Paths of the Woses
    'C',  # 135: Phial of Galadriel
    'C',  # 136: The Ents Awake: Treebeard
    'X',  # 137: Rangers of the North
    'S',  # 138: Riders of Theoden
    'C',  # 139: Smeagol Helps Nice Master
    'S',  # 140: Swords in Eriador
    'C',  # 141: The Eagles are Coming
    'C',  # 142: The Grey Company
    'S',  # 143: The Last Battle
    'C',  # 144: The Ents Awake: Entmoot
    'S',  # 145: The Power of Tom Bombadil
    'S',  # 146: The Red Arrow
    'S',  # 147: The Spirit of Mordor
    'C',  # 148: There and Back Again
    'C',  # 149: There is Another Way
    'S',  # 150: Thranduil's Archers
    'X',  # 151: Threat to the Fiefs of the South
    'X',  # 152: Three Rings for Elven-Kings
    'S',  # 153: Through a Day and a Night
    'C',  # 154: We Prove the Swifter
    'X',  # 155: Wind From the West
    'S',  # 156: The Wisdom of Elrond
    'C',  # 157: Wizard's Staff
)

_SA_PID_TYPES = (
    'C',  # 158: A Balrog is Come!
    'S',  # 159: A New Power is Rising
    'C',  # 160: Candles of Corpses
    'X',  # 161: Captain of Despair
    'C',  # 162: Cruel Weather
    'X',  # 163: Deep Trenches and Great Engines
    'S',  # 164: Denethor's Folly
    'C',  # 165: Dreadful Spells
    'C',  # 166: Flocks of Crebain
    'C',  # 167: Foul Thing From the Deep
    'C',  # 168: Give it to Uss!
    'C',  # 169: Grond, Hammer of the Underworld
    'S',  # 170: Half-Orcs and Goblin-Men
    'S',  # 171: Hill-Trolls
    'S',  # 172: Horde from the East
    'C',  # 173: Isildur's Bane
    'C',  # 174: Lure of the Ring
    'S',  # 175: Many Kings to the Service of Mordor
    'S',  # 176: Monsters Roused
    'C',  # 177: Morgul Wound
    'S',  # 178: Mustering of a Long-planned War
    'C',  # 179: Nazgul Search
    'S',  # 180: Olog-Hai
    'C',  # 181: On, On They Went
    'C',  # 182: Orc Patrol
    'S',  # 183: Orcs Multiplying Again
    'S',  # 184: Pits of Mordor
    'S',  # 185: Rage of the Dunlendings
    'S',  # 186: Return of the Witch-King
    'S',  # 187: Return to Valinor
    'X',  # 188: Shadow of Dol Guldur
    'S',  # 189: Shadows Gather
    'S',  # 190: Shadows on the Misty Mountains
    'C',  # 191: Shelob's Lair
    'S',  # 192: Stormcrow
    'C',  # 193: The Black Captain Commands
    'C',  # 194: The Breaking of the Fellowship
    'S',  # 195: The Corsairs are Upon Us!
    'S',  # 196: The Day Without Dawn
    'S',  # 197: The Fighting Uruk-Hai
    'X',  # 198: The Gates are Closed
    'S',  # 199: The King is Revealed
    'C',  # 200: The Lidless Eye
    'C',  # 201: The Nazgul Strike!
    'C',  # 202: The Palantir of Orthanc
    'C',  # 203: The Ring is Mine!
    'C',  # 204: The Ringwraiths are Abroad
    'S',  # 205: The Shadow is Moving
    'S',  # 206: The Shadow Lengthens
    'S',  # 207: Threats and Promises
    'C',  # 208: Wormtongue
    'C',  # 209: Worn with Sorrow and Toil
)

def _build_pid_to_ki():
    """Build piece_id -> commitment position (k.i) mapping.

    Within each deck type (FP Char, FP Strat, SA Char, SA Strat), the game
    assigns commitment positions in DESCENDING piece_id order: the highest
    pid gets k.i=0, next highest gets k.i=1, etc.
    """
    pid_to_ki = {}
    # Group pids by deck type
    groups = {'FC': [], 'FS': [], 'SC': [], 'SS': []}
    for i, t in enumerate(_FP_PID_TYPES):
        if t != 'X':
            groups['F' + t].append(106 + i)
    for i, t in enumerate(_SA_PID_TYPES):
        if t != 'X':
            groups['S' + t].append(158 + i)
    # Sort each group descending and assign k.i = rank
    for pids in groups.values():
        for ki, pid in enumerate(sorted(pids, reverse=True)):
            pid_to_ki[pid] = ki
    return pid_to_ki

_PID_TO_KI = _build_pid_to_ki()


class CardDecoder:
    """Decodes ALL cards (including unrevealed) using commitment brute-force.

    The approach:
    1. Parse commitment data (hash, salt pairs) from the log
    2. Brute-force each commitment position using the known password to get
       the full shuffled deck order: commitment[k.i] -> card_num
    3. Map each piece_id to its commitment position (k.i) using the known
       pid-to-type mapping derived from the messages.properties alphabetical order
    4. For any drawn card: card_num = commitment[k.i] where k.i = _PID_TO_KI[pid]
    """

    # Class-level password defaults for the common corpus convention;
    # per-game passwords vary and are injected by the caller PER
    # INSTANCE (reports CSV; the bridge sets both to 'ai' — AiBoot
    # claims both hands with that password).
    DEFAULT_FP_PASSWORD = "lembas"
    DEFAULT_SA_PASSWORD = "free"

    def __init__(self):
        self.fp_password = CardDecoder.DEFAULT_FP_PASSWORD
        self.sa_password = CardDecoder.DEFAULT_SA_PASSWORD

        # Brute-forced commitment decks: k.i -> card_num
        self.fp_char_deck = {}   # 24 positions
        self.fp_strat_deck = {}  # 24 positions
        self.sa_char_deck = {}   # 24 positions
        self.sa_strat_deck = {}  # 24 positions

        # Cards in hand: list of card info dicts
        self.fp_hand = []
        self.sa_hand = []

        # Turn tracking
        self.turn = 1

    def _parse_commitment_pairs(self, line: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Parse commitment line into two lists of (hash, salt) pairs.

        Returns (char_pairs, strat_pairs) - group1 is Character, group2 is Strategy.
        """
        match = re.search(r'Controls\.(464|479)~>(.+?)(?:\s+0\s+0\s+[A-F0-9]+)?$', line)
        if not match:
            return [], []

        tokens = match.group(2).strip().split()
        groups = [[], []]
        idx = 2  # Skip master_hash and master_salt

        for g in range(2):
            if idx >= len(tokens) or not tokens[idx].isdigit():
                break
            count = int(tokens[idx])
            idx += 1
            for _ in range(count):
                if idx + 1 < len(tokens) and len(tokens[idx]) == 40 and len(tokens[idx+1]) == 32:
                    groups[g].append((tokens[idx], tokens[idx+1]))
                    idx += 2
                else:
                    break

        return groups[0], groups[1]  # char_pairs, strat_pairs

    def _brute_force_deck(self, pairs: List[Tuple[str, str]], password: str) -> Dict[int, int]:
        """Brute-force all commitment positions to find card_num at each position.

        For each position, tries card_nums 0-23 with the formula:
        intermediate = SHA1(password + salt)
        commit_hash == SHA1(intermediate + str(card_num))
        """
        deck = {}
        for pos, (commit_hash, salt) in enumerate(pairs):
            intermediate = sha1_hex(password + salt)
            for card_num in range(24):
                if sha1_hex(intermediate + str(card_num)) == commit_hash:
                    deck[pos] = card_num
                    break
        if pairs and len(deck) != len(pairs):
            # wrong password / corrupt pair: silence here degrades the
            # whole game to 'Unknown' cards with no diagnostic
            print('[card_decoder] WARNING: brute-force recovered %d/%d '
                  'positions (wrong password?)' % (len(deck), len(pairs)))
        return deck

    def get_card_name(self, card_num: Optional[int], deck_type: str) -> str:
        """Get human-readable card name."""
        if card_num is None:
            return "Unknown"
        if 'FP' in deck_type:
            if 'Character' in deck_type:
                return FP_CHARACTER_CARDS.get(card_num, f"FP Char #{card_num}")
            else:
                return FP_STRATEGY_CARDS.get(card_num, f"FP Strat #{card_num}")
        else:
            if 'Character' in deck_type:
                return SA_CHARACTER_CARDS.get(card_num, f"SA Char #{card_num}")
            else:
                return SA_STRATEGY_CARDS.get(card_num, f"SA Strat #{card_num}")

    def _get_deck_for_type(self, deck_type: str) -> Dict[int, int]:
        """Get the brute-forced commitment deck for a given deck type."""
        return {
            'FP Character': self.fp_char_deck,
            'FP Strategy': self.fp_strat_deck,
            'SA Character': self.sa_char_deck,
            'SA Strategy': self.sa_strat_deck,
        }.get(deck_type, {})

    def decode_piece(self, piece_id: int, deck_type: str) -> Optional[int]:
        """Decode a piece_id to its card_num using commitment position lookup."""
        ki = _PID_TO_KI.get(piece_id)
        if ki is None:
            return None
        deck = self._get_deck_for_type(deck_type)
        return deck.get(ki)

    def preprocess_log(self, log_lines: List[str]):
        """First pass: parse commitments and brute-force all deck positions."""
        for line in log_lines:
            # only assign when the payload regex actually parsed pairs:
            # a substring hit whose regex fails would otherwise clobber
            # previously brute-forced decks with {} (every later decode
            # silently 'Unknown')
            if 'Controls.464' in line:
                fp_char_pairs, fp_strat_pairs = self._parse_commitment_pairs(line)
                if fp_char_pairs or fp_strat_pairs:
                    self.fp_char_deck = self._brute_force_deck(fp_char_pairs, self.fp_password)
                    self.fp_strat_deck = self._brute_force_deck(fp_strat_pairs, self.fp_password)
                else:
                    print('[card_decoder] WARNING: Controls.464 line '
                          'failed commitment parse; decks kept')
            if 'Controls.479' in line:
                sa_char_pairs, sa_strat_pairs = self._parse_commitment_pairs(line)
                if sa_char_pairs or sa_strat_pairs:
                    self.sa_char_deck = self._brute_force_deck(sa_char_pairs, self.sa_password)
                    self.sa_strat_deck = self._brute_force_deck(sa_strat_pairs, self.sa_password)
                else:
                    print('[card_decoder] WARNING: Controls.479 line '
                          'failed commitment parse; decks kept')

    def handle_draw(self, piece_id: str, deck_code: str, hand_code: str) -> Optional[dict]:
        """Handle a card draw and return card info."""
        deck_type = {
            '2004': 'FP Character', '2005': 'FP Strategy',
            '2006': 'SA Character', '2007': 'SA Strategy'
        }.get(deck_code)

        if not deck_type:
            return None

        side = 'FP' if hand_code == '45' else 'SA'

        # Decode card using commitment position lookup
        card_num = self.decode_piece(int(piece_id), deck_type)
        card_name = self.get_card_name(card_num, deck_type)

        card_info = {
            'piece_id': piece_id,
            'side': side,
            'deck_type': deck_type,
            'drawn_turn': self.turn,
            'card_number': card_num,
            'card_name': card_name,
            'decoded': card_num is not None,
        }

        if side == 'FP':
            self.fp_hand.append(card_info)
        else:
            self.sa_hand.append(card_info)

        return card_info

    def handle_reveal(self, piece_id: str, card_num: int, is_fp: bool) -> Optional[dict]:
        """Handle a card being played/revealed."""
        hand = self.fp_hand if is_fp else self.sa_hand

        # Find and remove the card from hand
        for i, card in enumerate(hand):
            if card['piece_id'] == piece_id:
                card['card_number'] = card_num
                card['decoded'] = True
                return hand.pop(i)

        # VASSAL re-creates card pieces on reveal with FRESH ids — match
        # the decoded identity (number + deck type from the fresh pid)
        # instead
        want_dt = None
        try:
            pid = int(piece_id)
            if is_fp and 106 <= pid <= 157:
                t = _FP_PID_TYPES[pid - 106]
            elif not is_fp and 158 <= pid <= 209:
                t = _SA_PID_TYPES[pid - 158]
            else:
                t = None
            if t == 'X':
                # expansion pid: matching by number would pop an
                # unrelated same-numbered BASE card from the hand
                return None
            want_dt = {'C': 'Character', 'S': 'Strategy'}.get(t)
        except (ValueError, IndexError):
            pass
        for i, card in enumerate(hand):
            if card.get('decoded') and card.get('card_number') == card_num \
                    and (want_dt is None
                         or str(card.get('deck_type', '')).endswith(want_dt)):
                return hand.pop(i)

        # Card not found in hand - might be direct play
        return None
