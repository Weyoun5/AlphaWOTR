#!/usr/bin/env python3
"""
WOTR Log Replay - Parses WOTR-Gandalf log files and replays them in the Python GUI.

=====================================================================
LEGACY / DIAGNOSTIC-ONLY (USER DOCTRINE 2026-08-03): "only the
rendered text matters. we should not be using the raw piece moves
anywhere."  Raw piece-move replay must NEVER feed training, extraction,
or any production decision path.  The v32+ corpus chain is
rendered_extract/rendered_replay/rendered_census over client-rendered
dumps.  The ONLY sanctioned production import from this module was the
CardDecoder family (card-identity decoding from draw events — hidden
information the rendered board text cannot carry; NOT piece-move
parsing); it now lives in card_decoder.py and is RE-EXPORTED here for
legacy importers.  Known raw-replay defects P1-P8 (see
data/log_census_report.json parser_defects) are documented, NOT fixed
— the machinery is retired from live paths.  Remaining consumers:
diagnostics (log_census.py) and legacy test coverage.
=====================================================================

This script reads log files from the WOTR-Gandalf board game client and translates
the actions into direct game state modifications for replay in the Python implementation.

IMPORTANT: This does NOT call game_logic functions that have randomness.
Instead it directly sets state values based on the historical log data.
"""

import re
import os
import sys
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import board
from board import sanitize
import game_state as gs
from game_state import Side, Nation, UnitKind, ActionDieFace, Nazgul
try:
    from location_codes_authoritative import \
        AUTH_LOCATION_CODES as _AUTH_CODES
except ImportError:          # generated file absent: fallback-free decode
    _AUTH_CODES = {}
try:
    from piece_table_authoritative import AUTH_ARMY_PIECES
    _AUTH_PIECES = {}
    for _pid, (_nat, _kind) in AUTH_ARMY_PIECES.items():
        _n = getattr(Nation, _nat) if _nat else None
        _k = ('LEADER' if _kind == 'LEADER'
              else 'NAZGUL' if _kind == 'NAZGUL'
              else getattr(UnitKind, _kind))
        _AUTH_PIECES[_pid] = (_n, _k)
except ImportError:
    _AUTH_PIECES = {}

# Card databases + commitment brute-force decoder: MOVED to
# card_decoder.py (the one sanctioned production surface of this legacy
# module).  Re-exported here so every existing `log_replay.X` /
# `from log_replay import X` consumer keeps working unchanged.
from card_decoder import (          # noqa: F401  (re-export)
    FP_CHARACTER_CARDS, FP_STRATEGY_CARDS,
    SA_CHARACTER_CARDS, SA_STRATEGY_CARDS,
    sha1_hex, _FP_PID_TYPES, _SA_PID_TYPES,
    _build_pid_to_ki, _PID_TO_KI, CardDecoder,
)

# ---------------------------------------------------------------------------
# Location Code Mappings (Game.XXXX -> region/area names)
# ---------------------------------------------------------------------------

# (W8 D1: the unused SPECIAL_AREAS dict was deleted 2026-07-20 —
# zero code callers, only a docstring mention in
# location_codes_authoritative.py; it conflicted with LOCATION_CODES
# on Game.1999/2000/2033/2034 which is the authoritative map.)

LOCATION_CODES = {
    # Mordor track. Codes verified by tracing marker-247 walks across the
    # corpus (2023-2026 module versions): 524 -> 525 -> 528 -> 529 -> 532
    # -> 533.  (530/531 never appear in any sampled log; the old guesses
    # 'Game.530/531' = Mordor 4/5 were wrong.)
    'Game.524': 'Mordor 0', 'Game.525': 'Mordor 1',
    'Game.528': 'Mordor 2', 'Game.529': 'Mordor 3',
    'Game.532': 'Mordor 4', 'Game.533': 'Mordor 5',
    # Additional Eriador / western regions
    'Game.1883': 'Minhiriath', 'Game.1884': 'South Ered Luin', 'Game.1885': 'Cardolan',
    'Game.1886': 'Harlindon', 'Game.1887': 'Grey Havens', 'Game.1888': 'Forlindon',
    'Game.1889': 'Tower Hills', 'Game.1890': 'The Shire', 'Game.1891': 'Ered Luin',
    'Game.1892': 'Old Forest', 'Game.1893': 'North Ered Luin', 'Game.1894': 'Gladden Fields',
    'Game.1895': 'Enedwaith', 'Game.1896': 'Arnor', 'Game.1897': 'Buckland',
    'Game.1898': 'Angmar', 'Game.1899': 'North Downs',
    # Eriador regions
    'Game.1900': 'Tharbad', 'Game.1901': 'Ettenmoors', 'Game.1902': 'Bree',
    'Game.1903': 'Mount Gram', 'Game.1904': 'Weather Hills', 'Game.1905': 'Druwaith Iaur',
    'Game.1906': 'South Downs', 'Game.1907': 'Andrast', 'Game.1908': 'Trollshaws',
    'Game.1909': 'Anfalas',
    # Map regions
    'Game.1910': 'North Dunland', 'Game.1911': 'Rivendell', 'Game.1912': 'Hollin',
    'Game.1913': 'Erech', 'Game.1914': 'South Dunland', 'Game.1915': 'Fords of Bruinen',
    'Game.1916': 'Gap of Rohan', 'Game.1917': 'Moria', 'Game.1918': 'Fords of Isen',
    'Game.1919': 'Orthanc', 'Game.1920': 'Dol Amroth', 'Game.1921': 'Fangorn',
    'Game.1922': 'Mount Gundabad', 'Game.1923': "Helm's Deep", 'Game.1924': 'Lamedon',
    'Game.1925': 'Westemnet', 'Game.1926': 'High Pass', 'Game.1927': 'Dimrill Dale',
    'Game.1928': 'Lorien', 'Game.1929': 'Edoras', 'Game.1930': 'Parth Celebrant',
    'Game.1931': 'Pelargir', 'Game.1932': 'Umbar', 'Game.1933': 'Folde',
    'Game.1934': "Eagles' Eyrie", 'Game.1935': 'Lossarnach', 'Game.1936': "Goblin's Gate",
    'Game.1937': 'Eastemnet', 'Game.1938': 'West Harondor', 'Game.1939': 'Evendim',
    'Game.1940': 'Old Ford', 'Game.1941': 'Carrock', 'Game.1942': 'Druadan Forest',
    'Game.1943': 'Osgiliath', 'Game.1944': 'Minas Tirith', 'Game.1945': 'Rhosgobel',
    'Game.1946': 'Near Harad', 'Game.1947': 'Northern Mirkwood', 'Game.1948': 'Western Emyn Muil',
    'Game.1949': 'North Anduin Vale', 'Game.1950': 'Western Mirkwood', 'Game.1951': 'Western Brown Lands',
    'Game.1952': 'East Harondor', 'Game.1953': 'South Anduin Vale', 'Game.1954': 'Woodland Realm',
    'Game.1955': 'South Ithilien', 'Game.1956': 'Dol Guldur', 'Game.1957': 'Old Forest Road',
    'Game.1958': 'Narrows of the Forest', 'Game.1959': 'Dead Marshes', 'Game.1960': 'Eastern Brown Lands',
    'Game.1961': 'Eastern Mirkwood', 'Game.1962': 'Withered Heath', 'Game.1963': 'Southern Mirkwood',
    'Game.1964': 'North Ithilien', 'Game.1965': 'Erebor', 'Game.1966': 'Northern Rhovanion',
    'Game.1967': 'Eastern Emyn Muil', 'Game.1968': 'Far Harad', 'Game.1969': 'Minas Morgul',
    'Game.1970': 'Dale', 'Game.1971': 'Nurn', 'Game.1972': 'Noman-Lands',
    'Game.1973': 'Iron Hills', 'Game.1974': 'Morannon', 'Game.1975': 'Southern Rhovanion',
    'Game.1976': 'Dagorlad', 'Game.1977': 'Gorgoroth', 'Game.1978': 'Khand',
    'Game.1979': 'Barad-Dur', 'Game.1980': 'Vale of the Carnen', 'Game.1981': 'Ash Mountains',
    'Game.1982': 'Vale of the Celduin', 'Game.1983': 'Southern Dorwinion', 'Game.1984': 'East Rhun',
    'Game.1985': 'Northern Dorwinion', 'Game.1986': 'South Rhun', 'Game.1987': 'North Rhun',
    # Free Peoples stronghold boxes (siege retreat areas — distinct from outer regions)
    'Game.1988': 'Erebor Stronghold', 'Game.1989': 'Grey Havens Stronghold',
    'Game.1990': 'Rivendell Stronghold', 'Game.1991': 'Woodland Realm Stronghold',
    'Game.1992': 'Lorien Stronghold', 'Game.1993': "Helm's Deep Stronghold",
    'Game.1994': 'Minas Tirith Stronghold', 'Game.1995': 'Dol Amroth Stronghold',
    'Game.1996': 'Shadow Stronghold',
    # Shadow stronghold boxes
    'Game.20': 'Moria Stronghold', 'Game.21': 'Dol Guldur Stronghold',
    'Game.22': 'Orthanc Stronghold', 'Game.23': 'Morannon Stronghold',
    'Game.24': 'Barad-Dur Stronghold', 'Game.25': 'Minas Morgul Stronghold',
    'Game.26': 'Umbar Stronghold',
    # Victory Points track. Corpus evidence (VP-marker piece 423 moves):
    # transitions are 33->388->389->391->392 with single-step captures, so
    # 388=7, 389=8, 391=9, 392=10. Game.390 never appears in any log
    # (kept mapped defensively). Game 0 ends 389->392 = 8->10 per the
    # readable reference.
    'Game.27': 'VP 0', 'Game.28': 'VP 1', 'Game.29': 'VP 2',
    'Game.30': 'VP 3', 'Game.31': 'VP 4', 'Game.32': 'VP 5', 'Game.33': 'VP 6',
    'Game.388': 'VP 7', 'Game.389': 'VP 8', 'Game.390': 'VP 9',
    'Game.391': 'VP 9', 'Game.392': 'VP 10',
    # Game areas
    'Game.1997': 'Hunt Box', 'Game.1998': 'Fellowship Box', 'Game.1999': 'Guide Box',
    'Game.2000': 'Political 3', 'Game.2001': 'Political 2', 'Game.2002': 'Political 1', 'Game.2003': 'Political 0',
    'Game.2004': 'FP Char Deck', 'Game.2005': 'FP Strat Deck', 'Game.2006': 'SA Char Deck', 'Game.2007': 'SA Strat Deck',
    'Game.2008': 'Vilya Slot', 'Game.2009': 'Nenya Slot', 'Game.2010': 'Narya Slot',
    'Game.2011': 'Used Ring 1', 'Game.2012': 'Used Ring 2', 'Game.2013': 'Used Ring 3',
    'Game.2014': 'Track 0', 'Game.2015': 'Track 1', 'Game.2016': 'Track 2', 'Game.2017': 'Track 3',
    'Game.2018': 'Track 4', 'Game.2019': 'Track 5', 'Game.2020': 'Track 6', 'Game.2021': 'Track 7',
    'Game.2022': 'Track 8', 'Game.2023': 'Track 9', 'Game.2024': 'Track 10', 'Game.2025': 'Track 11',
    'Game.2026': 'Track 12',
    # Board areas (card play, reinforcements, etc.)
    'Game.2030': 'Board2', 'Game.2031': 'Board3', 'Game.2032': 'Board4',
    'Game.2033': 'Board5', 'Game.2034': 'Board6',
    'Game.2035': 'Used', 'Game.2036': 'SA Card Play', 'Game.2037': 'FP Card Play',
    'Game.2038': 'FP Reinf', 'Game.2039': 'FP Dead', 'Game.2040': 'SA Reinf', 'Game.2041': 'SA Dead',
    'Game.2042': 'FP Dice', 'Game.2043': 'SA Dice', 'Game.2044': 'Hunt Pool',
    'Game.2045': 'Used Hunt Tiles', 'Game.2046': 'Removed Hunt Tiles',
    'Controls.45': 'FP Hand', 'Controls.46': 'SA Hand',
    # authoritative-name overrides: '<replay> Turn' lacks an area
    # keyword and would pass is_map_region as a phantom region
    'Game.7': 'Turn Track',
    'Game.19': 'Mount Gundabad Stronghold',
}

# Shadow die faces mapping (verified from game log):
# A = Army, P = Palantir/Event, E = Eye, M = Muster, H = Muster/Army, C = Character
# Values 0-5 or 10-15 (with base offset 10)
SHADOW_DICE_VALUES = {
    0: ActionDieFace.MUSTER, 10: ActionDieFace.MUSTER,           # M
    1: ActionDieFace.MUSTER_ARMY, 11: ActionDieFace.MUSTER_ARMY, # H (Hybrid)
    2: ActionDieFace.ARMY, 12: ActionDieFace.ARMY,               # A
    3: ActionDieFace.CHARACTER, 13: ActionDieFace.CHARACTER,     # C
    4: ActionDieFace.EVENT, 14: ActionDieFace.EVENT,             # P (Palantir)
    5: ActionDieFace.EYE, 15: ActionDieFace.EYE,                 # E (Eye)
}

# FP die faces mapping (verified from game log):
# W = Will of the West, C = Character, M = Muster, H = Muster/Army, P = Event/Palantir
FP_DICE_VALUES = {0: ActionDieFace.WILL_OF_WEST,   # W
                  1: ActionDieFace.CHARACTER,       # C
                  2: ActionDieFace.EVENT,           # P (Palantir)
                  3: ActionDieFace.MUSTER,          # M
                  4: ActionDieFace.MUSTER_ARMY}     # H (Hybrid)

# Die face letters used by "changes <die> to <letter>" log lines
# (Elven Ring usage / manual die correction in the VASSAL module)
DIE_CHANGE_LETTERS = {
    'A': ActionDieFace.ARMY,
    'C': ActionDieFace.CHARACTER,
    'M': ActionDieFace.MUSTER,
    'H': ActionDieFace.MUSTER_ARMY,
    'P': ActionDieFace.EVENT,
    'E': ActionDieFace.EYE,
    'W': ActionDieFace.WILL_OF_WEST,
}

# Political marker piece IDs to nations (256-263, matching messages.properties order)
POLITICAL_MARKERS = {
    '256': Nation.ELVES, '257': Nation.DWARVES, '258': Nation.THE_NORTH,
    '259': Nation.ROHAN, '260': Nation.GONDOR, '261': Nation.SOUTHRON_EASTERLINGS,
    '262': Nation.SAURON, '263': Nation.ISENGARD,
}

# FP action token piece IDs
TOKEN_POLITICAL = '426'   # Hunt for the Ring (Political) - freeaction1.png
TOKEN_CARD_DRAW = '427'   # Hunt for the Ring (Card Draw) - freeaction2.png

# Settlement control markers (capture bookkeeping, NOT army units;
# authoritative piece table: 405-410 = Free Control from Game.2038,
# 411-420 = Shadow Control from Game.2040).  Before these were known a
# control-marker placement parsed as an unknown piece ARRIVING from
# off-board — a phantom regular in the settlement.  ONE family of
# definitions; the *_PIECES names are aliases kept for their call sites.
FP_CONTROL_MARKERS = frozenset(str(i) for i in range(405, 411))
SA_CONTROL_MARKERS = frozenset(str(i) for i in range(411, 421))
SH_CONTROL_MARKERS = SA_CONTROL_MARKERS
CONTROL_MARKERS = FP_CONTROL_MARKERS | SA_CONTROL_MARKERS
CONTROL_MARKER_PIECES = CONTROL_MARKERS

# Victory point track markers: piece -> side whose score they track
VP_MARKERS = {'422': Side.FREE_PEOPLES, '423': Side.SHADOW}
VP_MARKER_PIECES = frozenset(VP_MARKERS)

# Companion piece IDs (240-246)
COMPANIONS = {
    '240': 'Gandalf the Grey', '241': 'Strider', '242': 'Legolas',
    '243': 'Gimli', '244': 'Boromir', '245': 'Meriadoc', '246': 'Peregrin',
    '364': 'Gollum',
}

# Fellowship/Guide markers (247-252 only; 253-255 are minions, 256-263 are political)
FELLOWSHIP_MARKERS = {'247', '248', '249', '250', '251', '252'}

# Dwarven Ring (bid compensation) piece IDs — piece 428 verified in
# 2026-01-05 kapcerz (readable log "moves Dwarven Ring ..."; the grant is
# also announced as an explicit "adds 1 Dwarven Ring(s)" text line)
# t138 (engine-check family 2): 429-431 are ring pieces #2-#4 — group
# grants ('adds N Dwarven Ring(s)' + '430 429 428 -> FP Dice') use them
DWARVEN_RING_PIECES = {'428', '429', '430', '431'}

# Elven Ring piece IDs and locations
RING_PIECES = {'249': 'Vilya', '250': 'Nenya', '251': 'Narya'}
RING_ACTIVE_LOCS = {'Game.2008', 'Game.2009', 'Game.2010'}
RING_USED_LOCS = {'Game.2011', 'Game.2012', 'Game.2013'}

# --- Table-card KILL detection (USER 2026-07-19, ruling 1.149) ----------
# The three deliberately-killable play-on-table cards, keyed by piece id.
# 106/145 are FP Strategy cards (killed by SHADOW: any die + one ARMY-type
# card + one Character card from hand); 202 is the SA Palantir (killed by
# FP: a Will of the West die alone, or any die + one Elven Ring).
_TABLE_KILL_PIDS = {'106', '145', '202'}
# Board (table) slots a played persistent card sits in.
_TABLE_SLOT_ZONES = {'Game.2029', 'Game.2030', 'Game.2031', 'Game.2032',
                     'Game.2033', 'Game.2034'}
# SA Strategy cards of card_type 'Army'.  Only an ARMY-type Strategy card
# (NOT a Muster-type one) plus a Character-deck card pays for the Shadow
# kill of 'A Power too Great' / 'The Power of Tom Bombadil'.  Pids verified
# against cards.get_card(cid).card_type (messages.properties alpha order):
#  164 Denethor's Folly       170 Half-Orcs and Goblin-Men  171 Hill-Trolls
#  180 Olog-Hai               187 Return to Valinor         189 Shadows Gather
#  195 Corsairs of Umbar      196 The Day Without Dawn       197 The Fighting Uruk-Hai
#  205 The Shadow is Moving   206 The Shadow Lengthens
_SA_ARMY_CARD_PIDS = frozenset(
    {164, 170, 171, 180, 187, 189, 195, 196, 197, 205, 206})
# FP / SH action-die piece ids (base + bonus) — for the kill payment scan.
_FP_DIE_PIDS = frozenset({'210', '211', '212', '213', '358', '359'})

_TABLE_KILL_INFO_CACHE = None


def _table_kill_card_info(pid: str):
    """piece id -> (card_id, card_name, killer_side) for the three
    play-on-table killable cards, or None.  card_id/name come from the
    cards registry so the id matches persistent_cards.remove_card()
    exactly (the Palantír's accented name included)."""
    global _TABLE_KILL_INFO_CACHE
    if _TABLE_KILL_INFO_CACHE is None:
        # A failed registry scan must be LOUD and must not cache the
        # empty dict (that silently disabled ALL 1.149 table-card kill
        # detection for the rest of the process).
        from cards import get_cards
        info = {}
        for cid, c in get_cards().items():
            n = c.card_name
            if n == 'A Power too Great':
                info['106'] = (cid, n, Side.SHADOW)
            elif n == 'The Power of Tom Bombadil':
                info['145'] = (cid, n, Side.SHADOW)
            elif 'Palant' in n and 'Orthanc' in n:
                info['202'] = (cid, n, Side.FREE_PEOPLES)
        if len(info) != 3:
            raise RuntimeError(
                'table-kill card scan incomplete: %r' % (info,))
        _TABLE_KILL_INFO_CACHE = info
    return _TABLE_KILL_INFO_CACHE.get(pid)

# Shadow minion piece IDs (identified from log behavior)
MINION_PIECES = {
    '253': 'Witch King',
    '254': 'Saruman',
    '255': 'Mouth of Sauron',
}

# Named-character alias sets (t133 fix F5): the dual-sided figures
# change NAME mid-game (240 Grey->White, 241 Strider->Aragorn) while
# the drag pid stays constant.  Every name-match against army.leaders
# must accept BOTH faces — matching only the base name made the walk
# handler steal an anonymous leader instead of moving the named object
# (triage F5: GtW's Grey Havens -> Ered Luin walk moved the setup
# Elven leader; ~85-90 games).
PIECE_NAME_ALIASES = {
    '240': ('Gandalf the Grey', 'Gandalf the White'),
    '241': ('Strider', 'Aragorn'),
}

# Official companion card values (level, leadership) for re-created
# objects (revival drags / drift recovery)
_COMPANION_LEVELS = {
    'Gandalf the Grey': (3, 1), 'Gandalf the White': (3, 1),
    'Strider': (3, 1), 'Aragorn': (3, 2), 'Boromir': (2, 1),
    'Legolas': (2, 1), 'Gimli': (2, 1),
    'Meriadoc': (1, 1), 'Peregrin': (1, 1),
}


def names_for_pid(pid):
    """All names the pid's figure can carry (tuple, possibly empty)."""
    a = PIECE_NAME_ALIASES.get(pid)
    if a is not None:
        return a
    n = COMPANIONS.get(pid) or MINION_PIECES.get(pid)
    return (n,) if n else ()

# AUTHORITATIVE move-template triples (messages.properties is the log
# format spec; see gen_move_templates.py).  Different client builds
# encode the SAME piece move with different key triples — the
# hand-collected six missed e.g. HighlightLabel.96/97/98 political
# marker walks (2023-03-10 Terminatorhenk T2: Rohan mustered to war
# invisibly; O.4).  Triples with DEDICATED handlers that run AFTER the
# generic move match in _process_line are excluded here so those
# handlers keep receiving their lines: Controls.790 = fellowship track,
# Controls.795 = hunt-box die, Controls.366 = elven-ring use button.
from move_templates_authoritative import MOVE_TRIPLES

_MOVE_TRIPLE_EXCLUDE = {'Controls.790', 'Controls.795', 'Controls.366'}
_GEN_MOVE_TRIPLES = [t for t in MOVE_TRIPLES
                     if t[0] not in _MOVE_TRIPLE_EXCLUDE]
_MOVE_K1_ALT = '|'.join(re.escape(t[0]) for t in _GEN_MOVE_TRIPLES)
_MOVE_K2_ALT = '|'.join(re.escape(t[1]) for t in _GEN_MOVE_TRIPLES)
_MOVE_K3_ALT = '|'.join(re.escape(t[2]) for t in _GEN_MOVE_TRIPLES)

# Piece ID to unit type mapping based on VASSAL module structure.
# Piece IDs 1-58 are Shadow units, 59-98 are FP units (mixed nations),
# and 264-323 are additional FP units / replacements.

# Shadow nations set for determining side from nation
SHADOW_NATIONS = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}

# Specific piece ID to unit type mappings (highest priority).
# Pieces 59-98 are NOT in neat per-nation ranges; mapped individually from log evidence.
SPECIFIC_PIECE_TYPES = {
    # === ON-BOARD STARTING PIECES (inferred from starting regions) ===
    # --- Gondor ---
    '065': (Nation.GONDOR, UnitKind.REGULAR),   # Pelargir
    '075': (Nation.GONDOR, UnitKind.REGULAR),   # Dol Amroth
    '076': (Nation.GONDOR, UnitKind.REGULAR),   # Dol Amroth
    '077': (Nation.GONDOR, UnitKind.REGULAR),   # Dol Amroth
    '079': (Nation.GONDOR, UnitKind.REGULAR),   # Osgiliath
    '080': (Nation.GONDOR, UnitKind.REGULAR),   # Osgiliath
    '081': (Nation.GONDOR, UnitKind.REGULAR),   # Minas Tirith
    '082': (Nation.GONDOR, UnitKind.REGULAR),   # Minas Tirith
    '083': (Nation.GONDOR, UnitKind.REGULAR),   # Minas Tirith
    '084': (Nation.GONDOR, UnitKind.ELITE),     # Minas Tirith
    '085': (Nation.GONDOR, 'LEADER'),           # Minas Tirith
    '086': (Nation.GONDOR, 'LEADER'),           # spare/reserve
    # --- Rohan ---
    '071': (Nation.ROHAN, UnitKind.REGULAR),    # spare/mustered
    '072': (Nation.ROHAN, UnitKind.REGULAR),    # Edoras
    '073': (Nation.ROHAN, UnitKind.ELITE),      # Edoras
    '078': (Nation.ROHAN, UnitKind.REGULAR),    # Helm's Deep
    # --- Elves ---
    '059': (Nation.ELVES, UnitKind.REGULAR),    # Grey Havens
    '060': (Nation.ELVES, UnitKind.ELITE),      # Grey Havens
    '061': (Nation.ELVES, UnitKind.REGULAR),    # Rivendell
    '062': (Nation.ELVES, UnitKind.REGULAR),    # Rivendell
    '063': (Nation.ELVES, UnitKind.ELITE),      # Rivendell
    '067': (Nation.ELVES, 'LEADER'),
    '069': (Nation.ELVES, UnitKind.REGULAR),    # Woodland Realm
    '070': (Nation.ELVES, UnitKind.ELITE),      # Woodland Realm
    '074': (Nation.ELVES, UnitKind.ELITE),      # Woodland Realm
    '087': (Nation.ELVES, UnitKind.REGULAR),    # Lorien
    '089': (Nation.ELVES, UnitKind.REGULAR),
    '090': (Nation.ELVES, UnitKind.REGULAR),
    '091': (Nation.ELVES, UnitKind.REGULAR),
    '092': (Nation.ELVES, UnitKind.ELITE),      # Lorien
    '093': (Nation.ELVES, 'LEADER'),            # Lorien
    '095': (Nation.ELVES, UnitKind.REGULAR),
    '096': (Nation.ELVES, UnitKind.REGULAR),
    '097': (Nation.ELVES, UnitKind.REGULAR),
    # --- Dwarves ---
    '064': (Nation.DWARVES, UnitKind.REGULAR),  # Iron Hills
    '088': (Nation.DWARVES, UnitKind.REGULAR),  # Erebor
    '094': (Nation.DWARVES, UnitKind.ELITE),    # Erebor
    '098': (Nation.DWARVES, 'LEADER'),          # Erebor
    # --- The North ---
    '066': (Nation.THE_NORTH, UnitKind.REGULAR),
    '068': (Nation.THE_NORTH, UnitKind.REGULAR),
    # --- Shadow on-board ---
    '006': (Nation.ISENGARD, UnitKind.ELITE),   # Orthanc
    '035': (Nation.SAURON, UnitKind.ELITE),     # Dol Guldur
    '036': (Nation.SAURON, UnitKind.ELITE),     # Barad-Dur
    '037': (Nation.SAURON, 'NAZGUL'),
    '038': (Nation.SAURON, 'NAZGUL'),
    '039': (Nation.SAURON, 'NAZGUL'),
    '040': (Nation.SAURON, 'NAZGUL'),
    '055': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),  # Far Harad
    '056': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),  # South Rhun
    '057': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),  # Near Harad
    # === FP REINFORCEMENT POOL (empirically verified from identification game) ===
    # Elves pool
    '264': (Nation.ELVES, UnitKind.ELITE),
    '265': (Nation.ELVES, UnitKind.ELITE),
    '266': (Nation.ELVES, UnitKind.ELITE),
    '267': (Nation.ELVES, UnitKind.ELITE),
    '268': (Nation.ELVES, UnitKind.REGULAR),
    '269': (Nation.ELVES, UnitKind.REGULAR),
    # Gondor pool
    '270': (Nation.GONDOR, 'LEADER'),
    '271': (Nation.GONDOR, 'LEADER'),
    '272': (Nation.GONDOR, 'LEADER'),
    '273': (Nation.GONDOR, UnitKind.ELITE),
    '274': (Nation.GONDOR, UnitKind.ELITE),
    '275': (Nation.GONDOR, UnitKind.ELITE),
    '276': (Nation.GONDOR, UnitKind.ELITE),
    '277': (Nation.GONDOR, UnitKind.REGULAR),
    '278': (Nation.GONDOR, UnitKind.REGULAR),
    '279': (Nation.GONDOR, UnitKind.REGULAR),
    '280': (Nation.GONDOR, UnitKind.REGULAR),
    '281': (Nation.GONDOR, UnitKind.REGULAR),
    '282': (Nation.GONDOR, UnitKind.REGULAR),
    # Rohan pool
    '283': (Nation.ROHAN, 'LEADER'),
    '284': (Nation.ROHAN, 'LEADER'),
    '285': (Nation.ROHAN, 'LEADER'),
    '286': (Nation.ROHAN, UnitKind.ELITE),
    '287': (Nation.ROHAN, UnitKind.ELITE),
    '288': (Nation.ROHAN, UnitKind.ELITE),
    '289': (Nation.ROHAN, UnitKind.ELITE),
    '290': (Nation.ROHAN, UnitKind.REGULAR),
    '291': (Nation.ROHAN, UnitKind.REGULAR),
    '292': (Nation.ROHAN, UnitKind.REGULAR),
    '293': (Nation.ROHAN, UnitKind.REGULAR),
    '294': (Nation.ROHAN, UnitKind.REGULAR),
    '295': (Nation.ROHAN, UnitKind.REGULAR),
    # The North pool
    '296': (Nation.THE_NORTH, 'LEADER'),
    '297': (Nation.THE_NORTH, 'LEADER'),
    '298': (Nation.THE_NORTH, 'LEADER'),
    '299': (Nation.THE_NORTH, UnitKind.ELITE),
    '300': (Nation.THE_NORTH, UnitKind.ELITE),
    '301': (Nation.THE_NORTH, UnitKind.ELITE),
    '302': (Nation.THE_NORTH, UnitKind.ELITE),
    '303': (Nation.THE_NORTH, UnitKind.REGULAR),
    '304': (Nation.THE_NORTH, UnitKind.REGULAR),
    '305': (Nation.THE_NORTH, UnitKind.REGULAR),
    '306': (Nation.THE_NORTH, UnitKind.REGULAR),
    '307': (Nation.THE_NORTH, UnitKind.REGULAR),
    '308': (Nation.THE_NORTH, UnitKind.REGULAR),
    # Dwarves pool
    '309': (Nation.DWARVES, 'LEADER'),
    '310': (Nation.DWARVES, 'LEADER'),
    '311': (Nation.DWARVES, 'LEADER'),
    '312': (Nation.DWARVES, UnitKind.REGULAR),
    '313': (Nation.DWARVES, UnitKind.ELITE),
    '314': (Nation.DWARVES, UnitKind.ELITE),
    '315': (Nation.DWARVES, UnitKind.ELITE),
    '316': (Nation.DWARVES, UnitKind.REGULAR),
    '317': (Nation.DWARVES, UnitKind.REGULAR),
    # === SA REINFORCEMENT POOL (empirically verified from identification game) ===
    # Nazgul pool
    '318': (Nation.SAURON, 'NAZGUL'),
    '319': (Nation.SAURON, 'NAZGUL'),
    '320': (Nation.SAURON, 'NAZGUL'),
    '321': (Nation.SAURON, 'NAZGUL'),
    # Isengard pool
    '322': (Nation.ISENGARD, UnitKind.ELITE),
    '323': (Nation.ISENGARD, UnitKind.ELITE),
    '324': (Nation.ISENGARD, UnitKind.ELITE),
    '325': (Nation.ISENGARD, UnitKind.ELITE),
    '326': (Nation.ISENGARD, UnitKind.ELITE),
    '327': (Nation.ISENGARD, UnitKind.REGULAR),
    '328': (Nation.ISENGARD, UnitKind.REGULAR),
    '329': (Nation.ISENGARD, UnitKind.REGULAR),
    '330': (Nation.ISENGARD, UnitKind.REGULAR),
    '331': (Nation.ISENGARD, UnitKind.REGULAR),
    '332': (Nation.ISENGARD, UnitKind.REGULAR),
    # Sauron pool
    '333': (Nation.SAURON, UnitKind.ELITE),
    '334': (Nation.SAURON, UnitKind.ELITE),
    '335': (Nation.SAURON, UnitKind.ELITE),
    '336': (Nation.SAURON, UnitKind.ELITE),
    '337': (Nation.SAURON, UnitKind.REGULAR),
    '338': (Nation.SAURON, UnitKind.REGULAR),
    '339': (Nation.SAURON, UnitKind.REGULAR),
    '340': (Nation.SAURON, UnitKind.REGULAR),
    '341': (Nation.SAURON, UnitKind.REGULAR),
    '342': (Nation.SAURON, UnitKind.REGULAR),
    '343': (Nation.SAURON, UnitKind.REGULAR),
    '344': (Nation.SAURON, UnitKind.REGULAR),
    # S&E pool
    '345': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),
    '346': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),
    '347': (Nation.SOUTHRON_EASTERLINGS, UnitKind.ELITE),
    '348': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '349': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '350': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '351': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '352': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '353': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '354': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '355': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '356': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
    '357': (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
}

def _normalize_name(name) -> str:
    """Lowercase alnum-only card-name key (mirrors ai._normalize_card_name
    — kept local so the parser never imports the torch-heavy ai
    module).  Accents
    FOLD rather than strip ('Théoden' -> 'theoden', not 'thoden') so
    accented card ids match their plain-ASCII log names."""
    import unicodedata
    folded = unicodedata.normalize('NFKD', str(name or ''))
    folded = ''.join(ch for ch in folded if not unicodedata.combining(ch))
    return ''.join(ch for ch in folded.lower() if ch.isalnum())


# Event cards whose COMBAT side is 'Onslaught' — the only combat cards
# with a units-onto-card sacrifice choice (see combat.combat_effect_kind)
_ONSLAUGHT_EVENT_CARDS = {
    'orcsmultiplyingagain', 'shadowsonthemistymountains',
    'thefightingurukhai',
}

# Event cards whose COMBAT side is 'Relentless Assault' ('Before the
# Combat roll, you may inflict and apply up to two hits against your
# units. Add 1 to all dice ... for each hit').  The self-hit units are
# parked on the owner's Card Play area BEFORE the round's combat roll —
# that drag IS the magnitude decision (2023-12-06 T10: 'I kill two' +
# 2 Sauron regulars onto 'The Day Without Dawn' before roll 4-1-6-2-4
# = 3 hits at 6+ and Minas Tirith falls; same battle pattern for 'The
# King is Revealed' at Dol Amroth, 'i take two hits').  Units parked
# AFTER a roll are ordinary casualties (game-1 T5 'Monsters Roused').
_RELENTLESS_EVENT_CARDS = {
    'thedaywithoutdawn', 'rageofthedunlendings', 'thekingisrevealed',
}

# Event cards that can REVEAL the Fellowship: the FP pawn walk that
# follows such a play is the reveal PLACEMENT, not a declaration
# (2023-12-07_001647 T7: Nazgul Search reveal, pawn walked NR->Dagorlad
# BEFORE the marker flip; chat: "you need to reveal")
_REVEALING_EVENT_CARDS = {
    'nazgulsearch', 'thenazgulstrike', 'nazgulstrike',
}


HUNT_TILE_PIECES = {
    '214': '0r', '215': '0r',
    '216': '1r', '217': '1r',
    '218': '2r',
    '219': 'Eye', '220': 'Eye', '221': 'Eye', '222': 'Eye',
    '223': '1', '224': '1',
    '225': '2', '226': '2',
    '227': '3', '228': '3', '229': '3',
    # Special hunt tiles
    '232': 'Phial of Galadriel',
    '233': 'Smeagol Helps Nice Master',
    '234': 'Elven Cloaks',
    '235': 'Elven Rope',
    '236': 'The Ring is Mine!',
    '237': "Shelob's Lair",
    '238': 'On, On They Went',
    '239': 'Give it to Uss!',
}

# Shadow unit piece ID ranges for ON-BOARD starting pieces only (000-057).
# Pieces 006, 035-040, 055-057 are in SPECIFIC_PIECE_TYPES above.
# All reinforcement pool pieces (264+) are in SPECIFIC_PIECE_TYPES.
SA_PIECE_RANGES = {
    # Isengard Regular (000-005; 006 is Elite in SPECIFIC)
    range(0, 6): (Nation.ISENGARD, UnitKind.REGULAR),
    # Sauron Regular (007-034; 035-036 are Elite in SPECIFIC)
    range(7, 35): (Nation.SAURON, UnitKind.REGULAR),
    # Southrons Regular (041-054; 055-057 are Elite in SPECIFIC)
    range(41, 55): (Nation.SOUTHRON_EASTERLINGS, UnitKind.REGULAR),
}

def get_unit_type_from_piece_id(piece_id: str, dynamic_types: dict = None,
                                side_hint: str = None) -> tuple:
    """Determine unit nation and type from piece ID.

    Checks dynamic_types (runtime swap assignments) first, then SPECIFIC_PIECE_TYPES,
    then range tables (filtered by side_hint if provided).

    Args:
        piece_id: Piece ID string.
        dynamic_types: Runtime type assignments from swap/muster events.
        side_hint: 'shadow' or 'fp' to restrict range table lookup. Prevents
                   SA reinforcement pool pieces (318+) from matching FP ranges.
    """
    # Check dynamic types (from Controls.313 piece swap events)
    if dynamic_types and piece_id in dynamic_types:
        return dynamic_types[piece_id]

    # AUTHORITATIVE table (derived from the app's construction order,
    # universal across module versions) beats the hand-guessed static
    # tables: e.g. piece 074 is the ROHAN LEADER, not an Elves elite.
    if piece_id in _AUTH_PIECES:
        return _AUTH_PIECES[piece_id]

    # Check specific mappings (highest priority for static IDs)
    if piece_id in SPECIFIC_PIECE_TYPES:
        return SPECIFIC_PIECE_TYPES[piece_id]

    try:
        pid = int(piece_id)
    except ValueError:
        return None, None

    # Check SA on-board piece ranges (000-057 only)
    for id_range, (nation, unit_type) in SA_PIECE_RANGES.items():
        if pid in id_range:
            return nation, unit_type

    return None, None


def decode_location(code: str) -> str:
    """Decode Game.XXXX code to location name.

    Resolution order: the hand-tuned LOCATION_CODES (names aligned to
    board keys / area keywords) -> the AUTHORITATIVE map generated from
    the Java app's messages.properties -> the raw code.  Before the
    authoritative fallback, unmapped codes silently fell through and
    is_map_region() treated the drag as off-board (silent data loss —
    e.g. Game.19 'Mount Gundabad Stronghold' garrison drags)."""
    v = LOCATION_CODES.get(code)
    if v is not None:
        return v
    v = _AUTH_CODES.get(code)
    if v is not None:
        return v
    return code


def is_map_region(loc: str) -> bool:
    """Check if location is a map region (not a game area).

    EXACT board membership — the old substring-keyword test classified
    KHAND ('Hand'), CARDOLAN ('Card') and DEAD MARSHES ('Dead') as
    off-board, silently vanishing every army that entered them and
    inventing phantom 'returns from casualties' when pieces left
    (2025-04-06 demanray T14: Gandalf the White moving Dead Marshes ->
    Morannon parsed as a SECOND White transformation).  Audited: no
    name that passed the keyword test resolves outside board.regions."""
    if not loc:
        return False
    if loc.startswith('Game.'):
        return False
    return sanitize(loc) in board.regions


def stronghold_base_region(loc: Optional[str]) -> Optional[str]:
    """Map a stronghold-box location ('Minas Tirith Stronghold') to its
    base map region ('Minas Tirith'). Returns None for anything else
    (including 'Shadow Stronghold', whose base is not a region)."""
    if loc and loc.endswith(' Stronghold'):
        base = loc[:-len(' Stronghold')]
        if sanitize(base) in board.regions:
            return base
    return None


class LogReplayer:
    """Replays WOTR log files by directly modifying game state."""

    def __init__(self, state: gs.GameState, board_obj, verbose: bool = True):
        self.state = state
        self.board = board_obj

        # Piece 241 is the dual-sided Strider/Aragorn figure; a crowning
        # in a previous game (in-process) may have flipped the module
        # mapping — reset per replayer instance.
        COMPANIONS['241'] = 'Strider'

        # Track fellowship map position (piece 247)
        self.fellowship_location = "Rivendell"  # Starting location
        self.verbose = verbose
        self.line_num = 0
        self.turn = 1
        self.errors: List[str] = []

        # Track piece locations (piece_id -> region_key)
        self.piece_region: Dict[str, str] = {}

        # Track dynamically assigned piece types (from Controls.313 swap events)
        # piece_id -> (Nation, unit_type) for replacement pieces
        self.piece_types: Dict[str, tuple] = {}

        # Card decoder for tracking cards in hand
        self.card_decoder = CardDecoder()

        # Console output capture
        self.console_output = []

        # Track pre-allocated eyes (eyes allocated before dice roll)
        self.preallocated_eyes = set()

        # Die piece ID -> (side, pool_index) mapping.
        # Base dice are known upfront; bonus dice (from Gandalf White, Aragorn,
        # Saruman, Witch King, Mouth of Sauron) are added dynamically.
        self.die_id_map: Dict[int, Tuple[Side, int]] = {}
        for i in range(7):
            self.die_id_map[99 + i] = (Side.SHADOW, i)
        for i in range(4):
            self.die_id_map[210 + i] = (Side.FREE_PEOPLES, i)

        # Action log for AI training: list of (action_type, params_dict, side)
        self.action_log: List[Tuple[str, dict, Side]] = []

        # Die consumption tracking: pending die face to attach to next action (per side)
        self._pending_die: Dict[Side, Optional[Tuple[str, str]]] = {
            Side.FREE_PEOPLES: None, Side.SHADOW: None
        }

        # Combat tracking: region where combat is happening
        self._combat_region: Optional[str] = None

        # Hunt resolution tracking: set True after hunt_tile_draw, cleared on next turn action
        self._in_hunt_resolution: bool = False
        # Reveal-tile placement pending (survives the damage resolution
        # clearing _in_hunt_resolution; see the pawn-walk classifier)
        self._hunt_reveal_pending: bool = False

        # Player name -> side mapping (built from $ prefix lines)
        self._player_to_side: Dict[str, Side] = {}
        self._current_line_player: Optional[str] = None

        # Table-card KILL pre-pass results (ruling 1.149): drag line_num ->
        # {'params': {...}, 'side': Side, 'pid': str}.  Elven-ring drag
        # lines whose use is a Palantir-kill payment (their standalone
        # elven_ring record is folded into the discard_table_card).
        self._table_card_kills: Dict[int, dict] = {}
        self._table_kill_ring_lines: set = set()

        # Last human-readable (%) line for unit type / tile name parsing
        self._last_readable_line: Optional[str] = None

        # Combat rolls seen since the last die consumption (a battle
        # window without rolls = pre-battle-retreat convention)
        self._rolls_since_die: bool = False

        # Combat state tracking for siege press detection
        self._combat_casualties_this_round: bool = False
        self._last_swap_remove_elite: bool = False
        self._last_swap_remove_side: Optional[Side] = None
        self._last_swap_remove_nation = None
        self._last_press_chat_line: int = -99
        self._last_tile_draw_params = None
        self._last_tile_draw_line: int = -99

        # Siege bookkeeping: which side's units retreated into each
        # stronghold box (garrison), and whether the besieger's army has
        # been seen in the region yet.  Without this,
        # _update_region_control lifted sieges from region composition
        # alone: during the window between retreat_into_siege and the
        # besieger's advance the region holds ONLY the garrison side and
        # the old heuristic read that as "siege over" (2025-04-06
        # demanray T16 Erebor L1470: a garrison casualty-return drag ran
        # the control update and the Erebor siege vanished while the
        # T17 assault still had three press rounds to fight).
        self._siege_garrison: Dict[str, Side] = {}
        self._siege_arrived: set = set()

        # Pending combat card (from Controls.840 selection)
        self._pending_combat_card: Optional[Tuple[str, Side]] = None

        # Army-box pre-pass rewrites: line_num -> ('entry'|'exit'|'noop',
        # region_name_or_None).  See _preprocess_army_boxes.
        self._box_rewrites: Dict[int, Tuple[str, Optional[str]]] = {}

        # Most recent combat-card commitment per side (this battle round):
        # attaches the card to combat-effect-choice records (Onslaught
        # sacrifices, 'ffN' Dread-and-Despair forfeit chats)
        self._last_combat_card_name: dict = {}

        # True per side from a combat-card reveal until the next dice
        # roll: Relentless-Assault self-hits are decided BEFORE the
        # round's combat roll (units onto the card in that window = the
        # magnitude choice; after a roll they are parked casualties)
        self._cc_pre_roll: dict = {}

        # Battle die-window recency (line of the last die-to-Current-Action
        # or combat-card selection): a chat 'nc' within this window is a
        # combat-card DECLINE decision
        self._battle_window_line: int = -999

        # Attack-decision synthesis: attacks never appear as explicit log
        # lines — they're reconstructed from (die consumed) -> (battle
        # events at region R). One army_attack per die-consumption window.
        self._battle_synth_region: Optional[str] = None
        self._last_die_side: Optional[Side] = None
        self._synth_attack_idx: Optional[int] = None  # action_log index w/ to='?'

        # Hunt-damage decision synthesis (take corruption vs sacrifice)
        self._hunt_ctx: Optional[dict] = None

        # Fellowship hidden-marker sync: direct hidden assignments consume
        # the player's subsequent marker flip(s) (prevents toggle
        # inversion). Integer: declares produce a reveal+hide flip PAIR.
        self._skip_next_fsp_flip: int = 0

        # Gandalf the Grey death tracking (his later "return" to the map
        # is the Gandalf the White transformation)
        self._gandalf_dead: bool = False
        self._control_marker_at: dict = {}   # marker piece id -> region

        # Fixed-effect card window (placements are effects, not decisions)
        self._fixed_effect_card: bool = False

        # Card window: side that played an event card and hasn't consumed
        # a die since (die-less moves in this window are card effects)
        self._card_window_side: Optional[Side] = None

        # AI-bridge event-play announce: (normalized card name, line)
        # armed by a '<nick> plays <name>' chat; consumed by the next
        # name-matching reveal (see the reveal handler, t121)
        self._announced_play = None

        # Attacker side of the current synthesized battle (advance vs
        # retreat disambiguation)
        self._battle_attacker_side: Optional[Side] = None

        # Last piece-drag (multi-hop moves collapse into one decision)
        self._last_drag: Optional[dict] = None

        # ---- t125 fix 1: replay-side FP dead-pool bookkeeping -------
        # pid -> (Nation, 'regular'|'elite'|'leader') counted into
        # state.fp_casualties; refunded when the client board revives
        # the piece (map region / stronghold box / reinforcement box).
        self._fp_dead_pids: Dict[str, tuple] = {}

        # ---- t125 fix 2: hunt-tile replay model ---------------------
        # state.hunt_pool/discard/removed mirror the client's tile
        # zones; piece_region tracks every tile pid.  Client setup
        # (base2): 16 standard tiles in the Hunt Pool, the 8 special
        # tiles in Removed Hunt Tiles, 2 spare tiles.  hunt_removed
        # stays [] at start — the engine introduces specials on card
        # play; the Removed-zone specials are modeled piece-level only.
        self._tile_in_flight: Dict[str, object] = {}
        for _tp in range(214, 230):
            self.piece_region[str(_tp)] = 'Hunt Pool'
        for _tp in range(232, 240):
            self.piece_region[str(_tp)] = 'Removed Hunt Tiles'
        self.piece_region['230'] = 'Spare'
        self.piece_region['231'] = 'Spare'

        # ---- t125 fix 5: client-unparseable line bookkeeping --------
        self.checksumless_dropped: List[int] = []

        # Dice removed from play (dragged from a dice pool back to a
        # reinforcement box — the die-loss when a minion dies; Adobim
        # L645: die 099 SA Dice -> SA Reinf after Saruman falls).  The
        # replay dice lists are fixed-size, so removal is tracked by
        # die id; censuses exclude these (t125).
        self._dice_out_of_play: set = set()

        # ---- t125 fix 7 (ruling 1.157): end-of-game noise -----------
        # _game_over latches on a DEFINITE, irreversible signal (ring
        # destroyed / ring claimed), or lazily on an armed pending
        # signal (military VP / concession chat) once inside the
        # precomputed pure-noise log tail.  After the latch, noise-kind
        # lines (hand reveals, hunt draws, card draws, track walks,
        # tile drags, card discards) are flagged and SKIPPED; anything
        # else still processes (no real play is ever skipped).
        self._game_over: Optional[dict] = None
        self._eog_pending: Optional[str] = None
        self._eog_suffix_start: int = 1 << 30
        # Cracks-of-Doom window: the step-5 arrival does NOT end the
        # game by itself — its own hunt draw + damage resolution still
        # decide destroyed-vs-claimed (Wirbelwind L486/488: the '2r'
        # draw and +2 corruption walk FOLLOW the step-5 walk).  The
        # ring_destroyed latch waits for that resolution; a premature
        # latch de-forced the decisive draw (nondeterministic final
        # corruption in the teacher, canary catch).
        self._final_hunt_pending: bool = False
        self._final_hunt_drawn: bool = False

        # (piece locations are tracked lazily as the log mentions them)

    # ---- t125 fix 1: FP dead-pool piece bookkeeping -----------------
    # Zones that mean an army piece is DEAD / permanently off-board.
    # Card-play areas are included: card-killed / sacrificed pieces are
    # displayed there mid-sequence and in the dead pile afterwards
    # (replay_verify DEATH-ZONE EQUIVALENCES).
    _DEAD_PIECE_ZONES = frozenset({
        'dead', 'fp dead', 'sa dead', 'fp casualties', 'sa casualties',
        'fp dead pool', 'sa dead pool', 'fp card play', 'sa card play'})

    def _zone_is_dead(self, zone) -> bool:
        return sanitize(str(zone or '')) in self._DEAD_PIECE_ZONES

    # Destination KEYWORDS that mean a named figure dragged there is a
    # casualty.  Kept as substrings ONLY because card-play targets decode
    # to open-ended CARD NAMES ('Paths of the Dead', 'Horde from the
    # East/Deadly Strife') that no fixed set can enumerate.  Every use
    # must go through _zone_is_companion_death, which gates them.
    _COMPANION_DEATH_AREAS = ('Card Play', 'Dead', 'Casualties',
                              'Current Action')

    def _zone_is_companion_death(self, zone) -> bool:
        """THE resolver for "does this destination mean the companion
        DIED?".  Single answer to a single question — every companion
        arrival/departure classifier must call this, never re-roll its
        own keyword test.

        A MAP REGION IS NEVER A CASUALTY PILE.  is_map_region() was
        rewritten to exact board membership for exactly this reason (see
        its docstring: KHAND/'Hand', CARDOLAN/'Card', DEAD MARSHES/
        'Dead'), but _handle_fellowship_companions kept a private
        substring test, so 'Dead' in 'Dead Marshes' classified every
        companion SEPARATION into Dead Marshes as a death: the companion
        was popped from the fellowship, no army was ever created at the
        destination, and a phantom companion_death / hunt_sacrifice_*
        was appended to the corpus action log (specimen
        auto72_game02 L565, '245 Guide Box -> Dead Marshes' recorded as
        hunt_sacrifice_guide).  Nothing downstream ever corrects it.
        Dead Marshes is the ONLY collision across all 964 decoded
        locations, but the gate is stated as the invariant so a future
        region name cannot reopen the family."""
        if not zone:
            return False
        if is_map_region(zone) or stronghold_base_region(zone) is not None:
            return False
        if self._zone_is_dead(zone):
            return True
        return any(area in str(zone)
                   for area in self._COMPANION_DEATH_AREAS)

    def _zone_is_revive(self, zone) -> bool:
        """Unambiguous revival destinations: the piece is back on the
        map (or a stronghold box) or back in a reinforcement box."""
        if not zone:
            return False
        if is_map_region(zone) or stronghold_base_region(zone) is not None:
            return True
        return 'Reinf' in str(zone) or 'reinforcements' in str(zone)

    def _set_piece_zone(self, pid: str, zone, dead=None):
        """piece_region update + state.fp_casualties bookkeeping.

        FP casualties are permanent (rulebook p.30) and ai.py derives
        muster availability from state.fp_casualties — during replay
        the CLIENT dead piles are ground truth.  *dead*=True forces
        counting when the caller knows the piece was killed even though
        the zone is ambiguous (Controls.644 sends kills to 'Used');
        dead=None infers from the zone.  Refunds fire only on
        unambiguous revivals (downgrade returns 'from casualties',
        Dead->Reinf tidy drags, dead pieces re-mustered)."""
        if dead is None:
            dead = self._zone_is_dead(zone)
        if pid in self._fp_dead_pids:
            if not dead and self._zone_is_revive(zone):
                nat, kind = self._fp_dead_pids.pop(pid)
                fc = getattr(self.state, 'fp_casualties', None) or {}
                nd = fc.get(nat)
                if nd and nd.get(kind, 0) > 0:
                    nd[kind] -= 1
                self.log(f"FP dead-pool refund: piece {pid} {kind} "
                         f"-> {zone}")
        elif dead and pid not in COMPANIONS and pid not in MINION_PIECES:
            nation, ut = get_unit_type_from_piece_id(
                pid, self.piece_types)
            kind = None
            if nation is not None and nation not in SHADOW_NATIONS:
                if ut == 'LEADER':
                    kind = 'leader'
                elif ut == UnitKind.ELITE:
                    kind = 'elite'
                elif ut == UnitKind.REGULAR:
                    kind = 'regular'
            if kind is not None:
                fc = getattr(self.state, 'fp_casualties', None)
                if fc is None:
                    fc = self.state.fp_casualties = {}
                nd = fc.setdefault(nation, {'regular': 0, 'elite': 0,
                                            'leader': 0})
                nd[kind] = nd.get(kind, 0) + 1
                self._fp_dead_pids[pid] = (nation, kind)
        self.piece_region[pid] = zone

    # ---- t125 fix 2: hunt-tile replay model -------------------------
    _TILE_ZONE_LISTS = {'Hunt Pool': 'hunt_pool',
                        'Used Hunt Tiles': 'hunt_discard',
                        'Removed Hunt Tiles': 'hunt_removed'}

    @staticmethod
    def _tile_fold_name(name: str) -> str:
        import unicodedata
        s = unicodedata.normalize('NFKD', str(name))
        s = s.encode('ascii', 'ignore').decode('ascii').lower()
        return re.sub(r'[^a-z0-9]', '', s)

    def _tile_template(self, pid: str):
        """Fresh HuntTile matching the pid's printed face, or None."""
        name = HUNT_TILE_PIECES.get(pid)
        if name is None:
            return None
        low = name.strip().lower()
        if low == 'eye':
            return gs.HuntTile(is_eye=True)
        m = re.fullmatch(r'(-?\d+)\s*(r?)', low)
        if m:
            return gs.HuntTile(int(m.group(1)),
                               reveal=bool(m.group(2)))
        fold = self._tile_fold_name(name)
        for k, t in gs.SPECIAL_HUNT_TILES.items():
            if self._tile_fold_name(k) == fold:
                import copy
                return copy.deepcopy(t)
        return None

    def _tile_matches(self, tile, tmpl) -> bool:
        if tmpl is None or tile is None:
            return False
        if getattr(tmpl, 'is_special', False):
            return (getattr(tile, 'is_special', False)
                    and self._tile_fold_name(tile.special_name or '')
                    == self._tile_fold_name(tmpl.special_name or ''))
        return (not getattr(tile, 'is_special', False)
                and tile.is_eye == tmpl.is_eye
                and tile.damage == tmpl.damage
                and tile.reveal == tmpl.reveal)

    def _tile_pop(self, pid: str):
        """Detach the pid's HuntTile from wherever the model holds it
        (in-flight display, or the state list of its tracked zone)."""
        if pid in self._tile_in_flight:
            return self._tile_in_flight.pop(pid)
        zone = self.piece_region.get(pid)
        lst_name = self._TILE_ZONE_LISTS.get(str(zone))
        if lst_name is None:
            return None
        tmpl = self._tile_template(pid)
        lst = getattr(self.state, lst_name, None) or []
        for i, t in enumerate(lst):
            if self._tile_matches(t, tmpl):
                return lst.pop(i)
        return None

    def _tile_to_zone(self, pid: str, zone: str):
        """Move a hunt-tile pid to *zone*, keeping piece_region and the
        state.hunt_* lists in sync.  Non-list zones (card-play display,
        'Used', map drops) hold the tile in flight."""
        t = self._tile_pop(pid)
        lst_name = self._TILE_ZONE_LISTS.get(str(zone))
        if lst_name is not None:
            if t is None:
                t = self._tile_template(pid)
            if t is not None:
                lst = getattr(self.state, lst_name, None)
                if lst is None:
                    lst = []
                    setattr(self.state, lst_name, lst)
                lst.append(t)
        elif t is not None:
            self._tile_in_flight[pid] = t
        self.piece_region[pid] = zone
        self.log(f"Hunt tile {pid} "
                 f"({HUNT_TILE_PIECES.get(pid, '?')}) -> {zone}")

    def _mordor_tile_repool(self):
        """Client-mirrored Mordor-entry re-pool: EYE tiles return from
        the used pile to the Hunt Pool silently — no log lines
        (GenericPie dumps: eyes 219/221 used->pool at entry).  Special
        tiles do NOT auto-return in the client (Adobim dumps: 'Elven
        Rope' 235 stays in Used Hunt Tiles through entry); when humans
        want them back they drag them explicitly (Wirbelwind L305-307).
        Standard numbered tiles stay used; Removed is permanent.  NOTE:
        the ENGINE's own _restore_hunt_pool_for_mordor also re-pools
        specials — that is the engine's model for self-play and is not
        touched; the replay mirrors the client board."""
        st = self.state
        n = 0
        for t in list(st.hunt_discard or []):
            if t.is_eye:
                st.hunt_discard.remove(t)
                st.hunt_pool.append(t)
                n += 1
        if n:
            self.log(f"Mordor entry: {n} Eye tile(s) re-pooled")
        for pid, z in list(self.piece_region.items()):
            if pid not in HUNT_TILE_PIECES:
                continue
            if sanitize(str(z)) != 'used hunt tiles':
                continue
            tmpl = self._tile_template(pid)
            if tmpl is not None and tmpl.is_eye:
                self.piece_region[pid] = 'Hunt Pool'

    # Action types that participate in resolving hunt damage — anything
    # else recorded while a hunt is unresolved means the player chose to
    # absorb the damage as corruption (synthesized as a decision).
    _HUNT_RESOLUTION_TYPES = frozenset({
        'hunt_tile_draw', 'hunt_sacrifice_guide', 'hunt_sacrifice_random',
        'hunt_take_corruption', 'choose_guide', 'companion_death',
        'companion_move', 'fellowship_reveal', 'fellowship_hide',
        'fellowship_reveal_place',
    })

    def _record_action(self, action_type: str, params: dict, side: Side):
        """Record a player decision for AI training extraction."""
        # Unresolved hunt damage + a non-hunt action following it means
        # the FP player took the damage as corruption — that WAS a
        # decision (vs sacrificing the guide or a random companion)
        ctx = self._hunt_ctx
        if (ctx is not None and not ctx.get('resolved')
                and action_type == 'companion_separate'):
            # SACRIFICE-BY-SEPARATION dialect (rendered ruling fix 29,
            # ported 2026-07-25): a companion walking OUT of the
            # fellowship while hunt damage is UNRESOLVED is the
            # random-victim sacrifice absorbing it - the client renders
            # it as a bare fellowship-box move with no kill gesture
            # (GAME19 T7 L598-599: Merry -> Fangorn absorbs the '1'
            # tile; the old synthesizer fabricated hunt_take_corruption
            # +1 AND kept an ordinary separation).  No corruption walk
            # exists in these logs, so corruption was demonstrably NOT
            # taken; the separation stays as the placement record.
            ctx['resolved'] = True
            self._hunt_ctx = None
            self.action_log.append((
                'hunt_sacrifice_random',
                {'companion': str((params or {}).get('companion', ''))},
                Side.FREE_PEOPLES))
            self.log("Hunt damage absorbed by separation sacrifice "
                     "(%s)" % (params or {}).get('companion'))
        elif (ctx is not None and not ctx.get('resolved')
                and action_type not in self._HUNT_RESOLUTION_TYPES):
            ctx['resolved'] = True
            self.action_log.append((
                'hunt_take_corruption',
                {'amount': ctx.get('damage', 1)}, Side.FREE_PEOPLES))
            self.log(f"Hunt damage taken as corruption "
                     f"({ctx.get('damage', 1)}) [synthesized decision]")
            self._hunt_ctx = None
        # Attach pending die consumption from the same side.
        # Combat-card commitments are FREE (no die) — a pending die near
        # them belongs to the battle's own attack (2023-12-07_001647 T8:
        # the Helm's Deep assault's Army die landed on the Cruel Weather
        # commitment, starving the assault) — leave it pending.
        if self._pending_die[side] is not None \
                and action_type not in ('combat_card', 'combat_card_none'):
            params['die_used'] = self._pending_die[side]
            self._pending_die[side] = None
        elif (action_type in ('army_move', 'muster', 'companion_move',
                              'nazgul_move', 'political')
                and self._card_window_side == side):
            # Die-less placement/move inside a card window = card EFFECT,
            # not a standalone die decision (Celeborn's elite, Ringwraiths
            # moves, Corsairs' sea move, Wisdom of Elrond's political
            # advance, ...)
            params['card_effect'] = True
        elif (action_type == 'companion_separate'
                and self._card_window_side is not None):
            # A standalone separation always spends a Character die
            # (die_used caught above) - a DIE-LESS separation inside ANY
            # open card window is that card's effect.  ANY side: The
            # Breaking of the Fellowship is a SHADOW play whose
            # separations are FP records (cross-side), so the same-side
            # gate above can never catch them (GAME21 T4: Merry+Pippin
            # became phantom expected decisions).
            params['card_effect'] = True
        # fix 91: stamp the emission line (underscore key, same class
        # as _pieces/_by_nation) so gesture-adjacency checks (advance
        # drag-through fold) can verify records belong to ONE drag
        if isinstance(params, dict) and '_rec_line' not in params:
            params['_rec_line'] = self.line_num
        self.action_log.append((action_type, params, side))
        # t125 fix 7: a hunt-damage resolution inside the final-hunt
        # window (Cracks of Doom) decides the game — sacrifice paths
        # resolve without a corruption walk, so latch here too.
        if self._final_hunt_pending and self._game_over is None \
                and action_type in ('hunt_take_corruption',
                                    'hunt_sacrifice_guide',
                                    'hunt_sacrifice_random') \
                and self.state.fellowship.mordor_step == 5 \
                and self.state.fellowship.corruption < 12:
            self._final_hunt_pending = False
            self._game_over = {'line': self.line_num,
                               'reason': 'ring_destroyed'}
            self.log("GAME OVER: ring destroyed (final hunt resolved) "
                     "[ruling 1.157]")

    def _declare_region_key(self, loc) -> str:
        """Canonical sanitized BOARD key for a declare destination.

        Declare-record enrichment (2026-08-02): every fellowship_declare
        record carries params['region'] = the resolved destination in the
        engine-menu vocabulary (_get_declare_actions emits
        params={'region': <sanitized board key>}), so consumers that
        match on 'region' never see a raw client location name.
        Board.region() resolves the leading-article dialect ('The Grey
        Havens' vs 'grey havens'); unresolvable names fall back to the
        plain sanitize so the key is never empty when a name exists.
        """
        if loc is None:
            return ''
        txt = str(loc)
        if not txt or txt == '?':
            return ''
        try:
            return self.board.region(txt).key
        except Exception:
            return sanitize(txt)

    def _sync_corruption(self, pos: int):
        """Corruption marker (252) walk: state sync — and a corruption
        INCREASE while a hunt-damage decision is unresolved IS that
        decision (the FP player used the Ring: took the damage as
        corruption).  Evidence 2023-12-07 T7 line 609: '1r' tile +1
        corruption BEFORE the stronghold '2' tile whose Meriadoc
        sacrifice previously got mis-consumed by the '1r' hunt."""
        old = self.state.fellowship.corruption
        self.state.fellowship.corruption = pos
        self.log(f"Corruption: {old} -> {pos}")
        # RE-DECLARATION heal (rulebook p.34; GAME19 T5 L389): a lone
        # -1 corruption drag while the fellowship is HIDDEN with the
        # pawn on an unconquered FP City/Stronghold is the humans
        # re-declaring IN PLACE for the heal (the client has no formal
        # gesture for a stay-declare - just the marker walk).  Recorded
        # as the fellowship_declare DECISION; the engine's declare path
        # applies the heal itself (declare-time heal, game_logic).
        # FELLOWSHIP-PHASE window only: the stay-heal drag precedes the
        # turn's action-dice roll (dice_rolled resets at the turn
        # marker).  Mid-turn heals (Athelas class, during a battle at
        # the fellowship's settlement) must NOT classify - gate-1
        # regression 2026-07-24: Ruellor/Elessar each lost a kept
        # decision to a phantom mid-turn re-declare.
        # ... and NOT the heal leg of a just-recorded DECLARATION
        # gesture (the client renders pawn-move first, heal after:
        # Ruellor T? Eastemnet->Minas Tirith produced a duplicate
        # in-place declare and cost the teacher a kept decision,
        # gate-1 2026-07-24) - the engine's declare path already heals.
        _last_rec9 = self.action_log[-1] if getattr(
            self, 'action_log', None) else None
        _last_ty9 = (_last_rec9[0] if isinstance(_last_rec9, tuple)
                     else None)
        if pos == old - 1 and self.state.fellowship.hidden \
                and not self.state.dice_rolled \
                and _last_ty9 != 'fellowship_declare' \
                and not self.state.fellowship.mordor_entered \
                and self._hunt_ctx is None:
            try:
                import game_logic as _gl_rd
                _pr9 = sanitize(_gl_rd.fellowship_region(self.state))
                _m9 = board.regions.get(_pr9, {})
                _cap9 = self.state.enemy_controlled.get(
                    Side.SHADOW, set())
                if (_m9.get('is_city') or _m9.get('is_stronghold')) \
                        and _gl_rd.is_fp_region(_pr9) \
                        and _pr9 not in _cap9:
                    self._record_action(
                        'fellowship_declare',
                        {'to': _pr9, 'region': self._declare_region_key(_pr9),
                         'in_place_heal': True,
                         'progress_before': int(
                             self.state.fellowship.progress),
                         'progress_consumed': 0}, Side.FREE_PEOPLES)
                    self.log(f"  in-place re-declaration heal at {_pr9}")
            except Exception:
                pass
        ctx = self._hunt_ctx
        if pos > old and ctx is not None and not ctx.get('resolved'):
            ctx['resolved'] = True
            self._hunt_ctx = None
            self._record_action('hunt_take_corruption',
                                {'amount': pos - old}, Side.FREE_PEOPLES)
            self.log(f"Hunt damage taken as corruption (+{pos - old}) "
                     f"[marker walk]")
        # Lure of the Ring: a corruption walk while the drawn victim is
        # unresolved = the FP chose the CORRUPTION arm (the piece will
        # return to the fellowship) - the pending must not become a
        # companion_death when the card cleanup later kills the CARD
        if pos > old and getattr(self, '_lure_pending', None):
            self.log(f"Lure resolved as corruption "
                     f"({self._lure_pending['companion']} returns)")
            self._lure_pending = None
        # t125 fix 7 (ruling 1.157): corruption 12 = the Ring claimed —
        # a DEFINITE, irreversible Shadow victory
        if pos >= 12 and self._game_over is None:
            self._final_hunt_pending = False
            self._game_over = {'line': self.line_num,
                               'reason': 'ring_claimed'}
            self.log("GAME OVER: ring claimed (corruption 12) "
                     "[ruling 1.157]")
        elif self._final_hunt_pending and self._game_over is None \
                and self.state.fellowship.mordor_entered \
                and self.state.fellowship.mordor_step == 5:
            # the decisive final-hunt corruption walk resolved < 12:
            # the Ring is destroyed (FP victory, irreversible)
            self._final_hunt_pending = False
            self._game_over = {'line': self.line_num,
                               'reason': 'ring_destroyed'}
            self.log("GAME OVER: ring destroyed (final hunt resolved, "
                     "corruption < 12) [ruling 1.157]")

    def _track_die_consumed(self, die_id: int):
        """Track that a die was consumed (moved from pool to used)."""
        mapping = self.die_id_map.get(die_id)
        if not mapping:
            return
        side, die_idx = mapping
        face_name = '?'
        if die_idx < len(self.state.dice[side]):
            die = self.state.dice[side][die_idx]
            if die.current:
                face_name = die.current.name
            elif side == Side.FREE_PEOPLES:
                # Die consumed WITHOUT a logged roll (2023-03-15 T2:
                # sharpz never presses the roll button, then spends all
                # four FP dice on moves/an event/an army pair).  The
                # true faces are unknowable — reconstruct the whole
                # skipped roll as WILL OF THE WEST, the FP wildcard face
                # that legalizes every action the log shows the dice
                # performing (and keeps pass-legality die counts sane).
                # (No Shadow case seen yet; SH has no wildcard face —
                # handle when a log needs it.)
                _n = 0
                for _d in self.state.dice[side]:
                    if _d.current is None:
                        _d.current = ActionDieFace.WILL_OF_WEST
                        _n += 1
                face_name = die.current.name
                self.log(f"Un-rolled FP dice consumed - skipped roll "
                         f"reconstructed as WILL_OF_WEST faces")
                # Emit the reconstructed roll as a dice_roll record so
                # the closed loop FORCES these faces (otherwise it fills
                # them with unforced RNG — a zero-RNG-gate violation)
                self._record_action('dice_roll', {
                    'faces': [ActionDieFace.WILL_OF_WEST.value] * _n,
                    'count': _n, 'reconstructed': True}, side)
            die.used = True
        self._pending_die[side] = (side.name, face_name)
        # Monotonic die-consumption counter (O.1 flip-pair detector: a
        # fellowship flip pair with NO die consumed between them is
        # tracker-reset bookkeeping, not a reveal+hide)
        self._die_consume_seq = getattr(self, '_die_consume_seq', 0) + 1
        # New die = new action window: the next battle events belong to a
        # NEW attack declaration, and any fixed-effect card window closes
        self._rolls_since_die = False
        self._last_die_side = side
        self._battle_synth_region = None
        self._battle_attacker_side = None
        # RB-07 (2026-07-22): reset the per-battle-window "already
        # auto-advanced" nation set (the engine advances each attacked
        # nation exactly ONCE per battle; a SECOND advance in the same
        # window is the settlement-capture advance the engine does not
        # model — see _handle_political).
        self._battle_auto_pol_nations = set()
        self._synth_attack_idx = None
        self._fixed_effect_card = False
        self._card_window_side = None
        self._announced_play = None
        self._last_drag = None
        self._battle_hint_region = None
        self._battle_hint_from = None
        self._battle_window_line = getattr(self, 'line_num', 0)
        self.log(f"Die consumed: {side.name} die {die_idx} ({face_name})")

    def _emit_casualty_mark(self, side, region_san, hits, cause,
                            spec=None):
        """RC4 casualty-forcing v3: record one casualty MARK with its
        CAUSE, tagged at emission time where the live parser context
        exists.  Causes:
          'round'       unambiguous combat-round casualty (to dead/reinf)
          'parked'      removal parked on a Card Play area (usually the
                        round's real casualties at tables that park dead
                        on the revealed card — 2025-06-19 StellaRossa)
          'effect_cost' Onslaught/Relentless sacrifice (a decision cost,
                        never applied hits — combat_effect_choice emitted)
          'pre_cc'      removal between a combat-card reveal and the
                        round's roll (Durin's Bane-class pre-combat hits
                        applied OUTSIDE the final-hits path)
          'card_window' removal inside an event-card effect window
          'downgrade'   field elite->regular swap that was NOT detected
                        as a siege press (ambiguity sentinel: hit
                        weighting unknowable, window must not be forced)
        Consumed by log_census's casualty-mark census and
        test_clause_matrix's forcing."""
        if hits <= 0 or side is None:
            return
        if not hasattr(self, 'casualty_marks'):
            self.casualty_marks = []
        self.casualty_marks.append({
            'line': self.line_num, 'side': side, 'region': region_san,
            'hits': hits, 'cause': cause,
            # v5: the exact removal COMPOSITION when known
            # ({'kill_regular': R, 'kill_elite': E, 'downgrade': D})
            'spec': spec})

    def _mark_cause(self, region_san=None, to_loc=None):
        """Default cause resolution from live context (see
        _emit_casualty_mark).  A card window only makes a removal a card
        COST when it happens AWAY from the live battle (card-driven
        attacks keep their own window open — their in-battle casualties
        are round hits, 2025-06-19 StellaRossa T6 Lorien 'A Balrog is
        Come')."""
        if self._card_window_side is not None:
            _b = self._battle_synth_region
            _bf = None
            # 139 (2026-07-25 PeterM T16 West Harondor, Opus-agent
            # verified by fix-simulation): the positional
            # action_log[_synth_attack_idx] read went STALE when later
            # records shifted around the synth attack, so the
            # ATTACKER's own-region casualties (kills at WH while the
            # battle window sat on 'pelargir') failed the (_b, _bf)
            # membership test and were mis-caused 'card_window' - the
            # deliberate v5 blocker then left a phantom 12-piece SH
            # stack at WH and the T18 Osgiliath move overstacked.
            # Resolve the attacker-from ROBUSTLY: the nearest
            # army_attack at/before the stored index whose 'to'
            # matches the live battle region.  The card_window safety
            # gate itself is unchanged.
            if self._synth_attack_idx is not None:
                try:
                    _lo139 = min(self._synth_attack_idx,
                                 len(self.action_log) - 1)
                    for _j139 in range(_lo139, -1, -1):
                        _at9, _p9, _s9 = self.action_log[_j139]
                        if _at9 == 'army_attack':
                            if _b in (None, '?') or sanitize(
                                    str(_p9.get('to', ''))) == _b:
                                _bf = sanitize(
                                    str(_p9.get('from', '')))
                                break
                except Exception:
                    _bf = None
            if _b in (None, '?') or region_san not in (_b, _bf):
                return 'card_window'
        if any(self._cc_pre_roll.values()):
            return 'pre_cc'
        if to_loc in ('SA Card Play', 'FP Card Play'):
            return 'parked'
        return 'round'

    def _resolve_open_synth_attack(self):
        """Content-anchored lookup of the still-open ('?'-target)
        synthesized attack this window (mirrors fix 139).  _synth_attack_idx
        can go STALE across action_log mutations (combat-card retag
        pop()+re-append; cross-line 'del self.action_log[...]'), so a raw
        positional read can name the WRONG record.  Scan BACK from the
        clamped index for the army_attack whose target is not yet
        resolved; return (params, side) or None."""
        idx = self._synth_attack_idx
        if idx is None or not self.action_log:
            return None
        try:
            lo = min(idx, len(self.action_log) - 1)
            for j in range(lo, -1, -1):
                at0, p0, s0 = self.action_log[j]
                if at0 == 'army_attack' and p0.get('to') in (None, '?'):
                    return p0, s0
        except Exception:
            return None
        return None

    def _maybe_synth_attack(self, region_loc: Optional[str]):
        """Synthesize the army_attack decision for the current battle.

        Attacks never appear as explicit log lines — battles surface only
        as combat cards, casualties, and swaps. Emit one army_attack per
        die-consumption window, attributed to the side that consumed the
        last die; patch the target region in when it becomes known.
        Chat callouts ("MT", "osg time", "from SI") override the casualty
        location, which for attacker losses names the WRONG region.
        """
        side_pre = self._last_die_side or Side.SHADOW
        # A battle target must contain the DEFENDER's units. This rejects
        # both bogus chat hints ("mm" as a thinking noise = Minas Morgul)
        # and attacker-side casualty locations (attacker losses are
        # removed from the attacker's own region).
        hint = getattr(self, '_battle_hint_region', None)
        if hint and self._region_holds_enemy_of(hint, side_pre):
            region_loc = hint
        elif region_loc and not self._region_holds_enemy_of(
                sanitize(region_loc), side_pre):
            # the hint test above already failed and nothing mutated
            region_loc = None
        rs = sanitize(region_loc) if region_loc else None
        if self._battle_synth_region is not None:
            # Attack already emitted this window — patch '?' target
            if (rs and self._battle_synth_region == '?'
                    and self._synth_attack_idx is not None):
                # 1.218 (content-anchored, mirrors fix 139): the stored
                # _synth_attack_idx read went STALE when a later
                # action_log mutation (combat-card retag pop()+re-append
                # ~L3757, or a cross-line 'del self.action_log[...]'
                # record retraction) shifted the record the index named,
                # so the raw positional read wrote params['to'] onto the
                # WRONG record — a combat_card, not the synth attack
                # (reproduced in a 200-game direct-parse sweep: 1 hit,
                # idx->combat_card).  Resolve the still-open '?'-target
                # synth attack by scanning BACK from the (clamped) index
                # for the army_attack whose target is not yet known,
                # exactly as 139 hardened _mark_cause's attacker-from
                # lookup.
                _sa = self._resolve_open_synth_attack()
                if _sa is None:
                    return
                params, s = _sa
                params['to'] = region_loc
                if params.get('from') in (None, '?'):
                    params['from'] = self._find_attacker_from(rs, s)
                self._battle_synth_region = rs
                self._combat_region = rs
                self._auto_tag_defender_political(rs, s)
            return
        side = self._last_die_side or Side.SHADOW
        from_hint = getattr(self, '_battle_hint_from', None)
        from_loc = from_hint or (self._find_attacker_from(rs, side)
                                 if rs else '?')
        _atk_params = {'from': from_loc, 'to': region_loc or '?',
                       'die_used': self._pending_die[side] or None,
                       '_atk_line': getattr(self, 'line_num', 0),
                       '_decl_line': getattr(self, '_battle_window_line', 0)}
        # A die-less attack inside the attacker's own card window is the
        # CARD's attack (Corsairs of Umbar sea assault, The Fighting
        # Uruk-hai...), not a separate army-die decision — the decision
        # was playing the card
        if self._pending_die[side] is None \
                and self._card_window_side == side:
            _atk_params['card_effect'] = True
        self.action_log.append(('army_attack', _atk_params, side))
        if self._pending_die[side] is not None:
            self._pending_die[side] = None
        self._synth_attack_idx = len(self.action_log) - 1
        self._battle_synth_region = rs or '?'
        self._battle_attacker_side = side
        if rs:
            self._combat_region = rs
            self._auto_tag_defender_political(rs, side)
        self.log(f"Attack synthesized: {side.name} -> {region_loc or '?'} "
                 f"(from {from_loc})")

    # region -> owning Nation (local copy of ai._region_nation — kept
    # local so the parser never imports the torch-heavy ai module)
    _FACTION_TO_NATION = {
        'gondor': Nation.GONDOR, 'rohan': Nation.ROHAN,
        'elves': Nation.ELVES, 'dwarves': Nation.DWARVES,
        'the north': Nation.THE_NORTH, 'north': Nation.THE_NORTH,
        'isengard': Nation.ISENGARD, 'sauron': Nation.SAURON,
        'southrons': Nation.SOUTHRON_EASTERLINGS,
        'southron': Nation.SOUTHRON_EASTERLINGS,
        'southrons and easterlings': Nation.SOUTHRON_EASTERLINGS,
    }

    def _auto_tag_defender_political(self, region_san: str, attacker_side):
        """An attack on nation N's region makes N's political advance
        AUTOMATIC (rulebook).  The marker drag sometimes lands BEFORE the
        attack callout in the log — retro-tag the nearest recent die-less
        political record of the defending nation as auto so it is not
        treated as a separate decision."""
        meta = board.regions.get(region_san, {})
        fac = str(board.fid_to_name.get(meta.get('parent_id'), '')).lower()
        nat = self._FACTION_TO_NATION.get(fac)
        if nat is None:
            return
        # RB-07: the engine advances each attacked nation ONCE per battle;
        # if it has already been auto-advanced this window, a further
        # advance is the capture-advance (engine does not model it) and
        # must NOT be retro-tagged auto.
        _seen = getattr(self, '_battle_auto_pol_nations', None)
        if _seen is None:
            _seen = self._battle_auto_pol_nations = set()
        if nat in _seen:
            return
        for at0, p0, _s0 in reversed(self.action_log[-8:]):
            if at0 != 'political':
                continue
            if p0.get('die_used') or p0.get('auto') or p0.get('card_effect'):
                continue
            if str(p0.get('nation', '')).lower() != nat.value.lower():
                continue
            p0['auto'] = True
            _seen.add(nat)
            self.log(f"  (political {nat.value} retro-tagged auto: "
                     f"defender advance of the {region_san} attack)")
            return

    # Community abbreviations used in chat attack callouts (user-supplied)
    _REGION_ABBREVS = {
        'ofr': 'old forest road', 'mt': 'minas tirith', 'da': 'dol amroth',
        'pel': 'pelargir', 'mm': 'minas morgul', 'gorg': 'gorgoroth',
        'mg': 'mount gundabad', 'dg': 'dol guldur', 'lor': 'lorien',
        'riv': 'rivendell', 'wlr': 'woodland realm', 'osg': 'osgiliath',
        'hd': 'helm s deep', 'si': 'south ithilien', 'west': 'westemnet',
        'fords': 'fords of isen',  # the battle site (vs Bruinen, never attacked)
    }

    def _scan_chat_for_battle_hint(self, msg: str) -> None:
        """Extract an attack-target hint from a chat declaration.

        Players call the target before rolling: "MT", "DA", "osg time",
        "Lossarnach", "Fords from Orthanc". Kept until the next die is
        consumed; _maybe_synth_attack prefers it over the (sometimes
        wrong-side) casualty location.
        """
        words = [w.strip('.,!?:;()') for w in msg.split()]
        # 'from X' names the attacker's source region
        if 'from' in words:
            fi = words.index('from')
            tail = ' '.join(words[fi + 1:])
            src = self._match_region_text(tail)
            if src:
                self._battle_hint_from = src
            head = ' '.join(words[:fi])
            tgt = self._match_region_text(head)
            if tgt:
                self._battle_hint_region = tgt
            return
        tgt = self._match_region_text(msg)
        if tgt:
            self._battle_hint_region = tgt
            return
        # Word-level scan: "osg time", "MT then", ... — first word that is
        # an abbreviation or region name wins
        for w in words:
            tgt = self._match_region_text(w)
            if tgt:
                self._battle_hint_region = tgt
                return

    def _match_region_text(self, text: str) -> Optional[str]:
        """Match chat text to a region (abbreviation or [partial] name)."""
        t = sanitize(text)
        if not t or len(t) < 2:
            return None
        if t in self._REGION_ABBREVS:
            return self._REGION_ABBREVS[t]
        try:
            regions = board.regions
        except Exception:
            return None
        if t in regions:
            return t
        # Whole-message equals a region name modulo minor decoration
        # ("osg time" -> osgiliath handled via abbrevs; here full names)
        for rk in regions:
            if t == rk:
                return rk
        # Single long word matching the start of a unique region name
        if ' ' not in t and len(t) >= 4:
            hits = [rk for rk in regions if rk.startswith(t)]
            if len(hits) == 1:
                return hits[0]
            # fix 90 (jmelder T6): multiple prefix hits — take the
            # HIGHEST-COVERAGE completion (shortest name) when the
            # abbreviation covers most of it ('westem' -> westemnet,
            # 6/9, not 'western brown lands' 6/19; the failed hint let
            # the attack synth guess Druadan Forest for a 'folde from
            # westem' declaration).
            if len(hits) > 1:
                _best90 = min(hits, key=len)
                if len(t) / len(_best90) >= 0.5:
                    return _best90
        return None

    def _region_holds_enemy_of(self, region_san: str, side: Side) -> bool:
        """Does the region contain units belonging to `side`'s enemy?"""
        try:
            shadow = side == Side.SHADOW
            for key, army in self.state.armies.items():
                if sanitize(key) != region_san:
                    continue
                for u in army.units:
                    if (u.nation in SHADOW_NATIONS) != shadow:
                        return True
            return False
        except Exception:
            return True  # Fail open

    def _piece_side(self, pieces: List[str]) -> Side:
        """Side owning the given pieces (FP if any FP-nation piece)."""
        for p in pieces:
            if p in COMPANIONS:
                return Side.FREE_PEOPLES
            nation, _ = get_unit_type_from_piece_id(p, self.piece_types)
            if nation is not None:
                return (Side.SHADOW if nation in SHADOW_NATIONS
                        else Side.FREE_PEOPLES)
        return Side.FREE_PEOPLES

    def _find_attacker_from(self, region_san: str, side: Side) -> str:
        """Best-effort attacker source: the side's largest army in or
        adjacent to the battle region (sieges have both armies in-region).

        BESIEGED target (AI-BRIDGE logs only, t123): a stronghold under
        siege can only be battled by the army already IN its region
        (siege assault / sortie, from==to — rulebook; adjacent armies
        must first move in and join).  The largest-army heuristic
        wrongly picked a bigger ADJACENT army (wotr2026_7_17_11_26 T10
        L715: Dale 6R out-sized the 5 Shadow besiegers at Erebor; the
        live battle was the Erebor->Erebor assault).  Prefer the
        in-region army whenever the target is besieged and the
        attacking side has units there.  Gated on _ai_bridge_log: the
        corpus campaign canary must stay byte-identical."""
        try:
            shadow = side == Side.SHADOW
            best, best_n = '?', -1
            in_region_key = None
            candidates = [region_san] + [
                sanitize(a) for a in self.board.adjacent(region_san)]
            for key, army in self.state.armies.items():
                ks = sanitize(key)
                if ks not in candidates:
                    continue
                n = sum(1 for u in army.units
                        if (u.nation in SHADOW_NATIONS) == shadow)
                if n > 0 and ks == region_san:
                    in_region_key = key
                if n > best_n and n > 0:
                    best, best_n = key, n
            if in_region_key is not None \
                    and getattr(self, '_ai_bridge_log', False) \
                    and region_san in getattr(
                        self.state, 'besieged_strongholds', set()):
                return in_region_key
            return best
        except Exception:
            return '?'

    def _find_or_create_army(self, to_loc: str):
        """Find existing army by sanitized key, or create a new one.
        Returns (key, army) tuple.  Avoids duplicate keys with different casing."""
        to_san = sanitize(to_loc)
        # Check for existing army with same sanitized key
        for k in self.state.armies:
            if sanitize(k) == to_san:
                return k, self.state.armies[k]
        # Create new army
        try:
            to_region = self.board.region(to_san)
            to_key = to_region.orig_name if hasattr(to_region, 'orig_name') else to_loc
        except:
            to_key = to_loc
            to_region = None
        if to_key not in self.state.armies:
            self.state.armies[to_key] = gs.Army(region=to_region)
        return to_key, self.state.armies[to_key]

    def _update_region_control(self, region_key: str):
        """Update enemy_controlled for a specific region after army changes."""
        from game_logic import is_shadow_region, is_fp_region

        san = sanitize(region_key)
        info = board.regions.get(san, {})
        # RB-08 (2026-07-22): TOWNS capture on walk-in too (amendment
        # 1.55a: "ALL settlements capture - towns included; capture on
        # walk-in").  The engine's enforce_settlement_control already
        # includes is_town; the raw truth skipped towns, so a Shadow
        # walk-in onto an undefended enemy TOWN (Skaarj iron hills T4,
        # 2023-11-09 westemnet T6) left the raw truth a control-box
        # behind the engine until the human's late marker gesture.
        if not info.get('is_city') and not info.get('is_stronghold') \
                and not info.get('is_town'):
            return

        region_is_shadow = is_shadow_region(san)
        region_is_fp = is_fp_region(san)
        if not region_is_shadow and not region_is_fp:
            return

        # Find army at this region
        army = None
        for k, a in self.state.armies.items():
            if sanitize(k) == san:
                army = a
                break

        # Known Shadow minion names (stored as Companion objects without proper nation)
        shadow_minion_names = {'Witch King', 'Saruman', 'Mouth of Sauron'}

        has_fp = False
        has_sh = False
        if army:
            for u in army.units:
                if u.nation in SHADOW_NATIONS:
                    has_sh = True
                else:
                    has_fp = True
            for l in army.leaders:
                name = getattr(l, 'name', '')
                if isinstance(l, Nazgul) or name in shadow_minion_names or \
                   (hasattr(l, 'nation') and l.nation in SHADOW_NATIONS):
                    has_sh = True
                else:
                    has_fp = True

        besieged = getattr(self.state, 'besieged_strongholds', None)

        # Active siege: composition heuristics must key off WHO is the
        # garrison (inside the stronghold box) vs the besieger.  The old
        # side-symmetric logic lifted sieges during the window between
        # retreat_into_siege and the besieger's advance (the region then
        # holds ONLY the garrison side) and flipped control while the
        # besieger co-located with a live garrison (2025-04-06 demanray
        # T16 Erebor: a garrison casualty-return drag at L1470 read
        # "only Shadow present at an FP settlement" as garrison-
        # eliminated and the T17 assault's three press rounds stranded).
        if besieged is not None and san in besieged:
            gside = getattr(self, '_siege_garrison', {}).get(san)
            if gside is not None:
                g_has = has_sh if gside == Side.SHADOW else has_fp
                b_has = has_fp if gside == Side.SHADOW else has_sh
                if g_has:
                    if b_has:
                        self._siege_arrived.add(san)
                        return  # siege in progress: control frozen
                    if san not in self._siege_arrived:
                        return  # setup window: besieger not advanced yet
                    # besiegers eliminated after arriving: siege over,
                    # the garrison side holds — resolve control below
                    besieged.discard(san)
                    self._siege_garrison.pop(san, None)
                    self._siege_arrived.discard(san)
                    self.log(f"Siege at {region_key} over "
                             f"(besiegers eliminated)")
                else:
                    # garrison eliminated: siege over, capture resolves
                    # through the composition logic below
                    besieged.discard(san)
                    self._siege_garrison.pop(san, None)
                    self._siege_arrived.discard(san)
                    self.log(f"Siege at {region_key} over "
                             f"(garrison eliminated)")

        # FP settlement: Shadow captures if only Shadow units present
        if region_is_fp:
            if has_sh and not has_fp:
                self.state.enemy_controlled[Side.SHADOW].add(san)
                if besieged is not None:
                    besieged.discard(san)  # garrison eliminated: siege over
            elif has_fp:
                # Original owner reclaims
                self.state.enemy_controlled[Side.SHADOW].discard(san)
                if besieged is not None and not has_sh:
                    besieged.discard(san)  # besiegers eliminated: siege over

        # Shadow settlement: FP captures if only FP units present
        if region_is_shadow:
            if has_fp and not has_sh:
                self.state.enemy_controlled[Side.FREE_PEOPLES].add(san)
                if besieged is not None:
                    besieged.discard(san)
            elif has_sh:
                # Original owner reclaims
                self.state.enemy_controlled[Side.FREE_PEOPLES].discard(san)
                if besieged is not None and not has_fp:
                    besieged.discard(san)

    def log(self, msg: str):
        """Log a message if verbose."""
        formatted = f"[{self.line_num:4d}] {msg}"
        self.console_output.append(formatted)
        if self.verbose:
            print(formatted)

    def error(self, msg: str):
        """Log an error."""
        err = f"[{self.line_num:4d}] ERROR: {msg}"
        self.errors.append(err)
        if self.verbose:
            print(err)

    # Mordor-track box codes: on the physical board these boxes DOUBLE as
    # ARMY BOXES (oversized stacks parked off-map, a numbered counter
    # marking the true region).  The counter placement is not decodable
    # from the log, so army parking is resolved by a pre-pass instead.
    _ARMY_BOX_CODES = {'Game.524', 'Game.525', 'Game.528', 'Game.529',
                       'Game.532', 'Game.533'}
    _TRACK_MARKER_PIECES = {'247', '248', '252'}
    _SCRIBE_MOVE_RE = re.compile(
        r'<~Scribe\.141~>([\d\s]+?)<~Scribe\.142~><~([\w.]+)~>'
        r'<~Scribe\.143~><~([\w.]+)~><~Scribe\.144~>')

    # Die-face change line ('$GJB changes 105  to A 85B43B').  The
    # client interpreter does NOT snapshot these (ay_0.k() runs only
    # before moves/kills/flips/allocates dispatches), so an undo after
    # one restores the snapshot of the PRECEDING snapshotted command —
    # reverting that command AND the face change (t125 fix 4).
    _DIE_CHANGE_LINE_RE = re.compile(
        r'^\$\w+\s+changes\s+\d+\s+to\s+[A-Z]\b')

    @staticmethod
    def drop_client_unparseable(log_lines: List[str]) -> List[int]:
        """CLIENT-REPLAY ALIGNMENT ONLY (t125 fix 5): blank the $ command
        lines the client's own replay strip mangles and drops.

        The client replay applies `game.z.c(line)` to every line, which
        unconditionally cuts the LAST space-separated token (normally
        the 6-hex network checksum or the literal 'null').  A corpus
        line that carries NO trailing checksum (Wirbelwind/GJB line 38,
        a Controls.281 bid-token move; whole client-build spans in
        demanray) therefore loses its real last token and the
        interpreter throws — the client counts an error and drops the
        line.  This mirror is NOT applied in normal corpus parsing:
        those lines were real, applied moves in the LIVE game (Columbo
        T8 uses the very token its line 41 grants), so the teacher/BC
        pipeline must keep them.  replay_verify's prefix replayer calls
        this explicitly to reproduce the client board exactly.

        sendTokens commitment lines are kept: they carry no board
        effect on either side, and blanking them would break the
        CardDecoder (deck brute-force input).

        Mutates *log_lines* IN PLACE; returns the 1-based indices of
        the blanked lines."""
        dropped = []
        for i, line in enumerate(log_lines):
            ls = line.strip()
            if not ls.startswith('$') or ' sendTokens ' in ls:
                continue
            toks = ls.split()
            if len(toks) >= 2 and (
                    toks[-1] == 'null'
                    or re.fullmatch(r'[A-F0-9]{6}', toks[-1])):
                continue
            log_lines[i] = ''
            dropped.append(i + 1)
        return dropped

    @staticmethod
    def dedupe_adjacent_lines(log_lines: List[str]) -> List[int]:
        """USER RULING 1.162 (t133, triage F11): ADJACENT byte-identical
        action lines (checksum-stripped) are transport echoes — the
        module re-emits the command with a fresh checksum (25,223 such
        lines across 5,920 corpus logs).  The pid-addressed client
        applies the repeat as a NO-OP (the pieces are already at the
        destination / already dead); the parser's count-based grammars
        double-applied it (dup muster doubled units, dup kill
        over-killed).  Blank the repeats IN PLACE (numbering stays
        stable); chains (triples) blank every repeat.  Runs AFTER the
        undo pre-pass: an undo between the pair consumes one real
        application, so a blanked line breaks the adjacency chain.
        sendTokens lines are never touched (CardDecoder input).

        SCOPE: only the grammars where pid semantics make the repeat a
        provable no-op — piece moves (every _ANY_MOVE_RE triple incl.
        Scribe.141/Controls.281), kills/cleanups (Controls.296/644)
        and tile draws (Controls.624); exactly the triage F11 echo
        histogram.  TOGGLES are excluded: an adjacent duplicated
        Controls.323 FLIP is a real show-then-hide pair, not an echo
        (Columbo L652/653 double-flips marker 247 deliberately).
        Returns the 1-based indices of the blanked lines."""
        def _core(ls):
            ls = re.sub(r'\s+[A-F0-9]{6}$', '', ls)
            return re.sub(r'\s+null$', '', ls)

        def _echo_grammar(ls):
            return (LogReplayer._ANY_MOVE_RE.search(ls) is not None
                    or '<~Controls.296~>' in ls
                    or '<~Controls.644~>' in ls
                    or '<~Controls.624~>' in ls)
        dropped = []
        last_core = None
        for i, line in enumerate(log_lines):
            ls = line.strip()
            if not ls or not ls.startswith('$') or ' sendTokens ' in ls \
                    or not _echo_grammar(ls):
                last_core = None
                continue
            core = _core(ls)
            if core == last_core:
                log_lines[i] = ''
                dropped.append(i + 1)
                continue        # keep last_core: chains dedupe fully
            last_core = core
        return dropped

    @staticmethod
    def undo_prepass(log_lines: List[str]) -> List[str]:
        """Apply VASSAL undo buttons (<~Controls.819~>) to the line list.

        Each press cancels the nearest preceding player command
        ($-prefixed line).  Undone lines and the undo lines themselves
        are BLANKED (not deleted) so line numbers stay stable for
        curation keys and diagnostics; _process_line('') is a no-op.
        Without this, undone actions replay as real decisions AND their
        redos replay again (2023-12-06 T6 lines 550-560: three undos,
        chat "can I redo? the movement?", then different moves —
        double-applied piece drags cascaded tracking drift for the rest
        of the game).

        t125 fix 4: die-face change lines are never snapshotted by the
        client, so an undo whose nearest preceding command is a
        'changes X to Y' line actually restores the snapshot of the
        command BEFORE it — reverting the die spend AND the face change
        (Wirbelwind lines 125-129: spend, ring-collect, change, undo,
        undo left the parser one die over-spent).  Blank the change
        line(s) AND keep walking to the real snapshot target.

        t138 wave-3 (triage N8 root): <~Controls.16~> is the client's
        REDO button ('redoes last action', messages.properties 64).
        Each press re-applies the most recently undone command — the
        old pass left it unhandled, so an undo+redo pair NET-DELETED
        the command (Mol/Columbo 2024-10-20 L267-276: Gandalf the
        White's arrival drag+flip undone x3 then redone x3; the client
        keeps GtW at Fangorn, the parser kept him dead for 19
        checkpoints).  Redo restores the blanked line group LIFO; a new
        $-command clears the redo stack (VASSAL semantics — chat and
        <game>/<replay> lines do not).  1,568/17,180 corpus logs carry
        at least one redo press.

        Mutates *log_lines* IN PLACE (idempotent) and returns it, so the
        preprocess() call covers every replay path's own line list."""
        redo_stack = []          # [[(idx, original), ...], ...]
        for i, line in enumerate(log_lines):
            if '<~Controls.819~>' in line:
                group = []
                for j in range(i - 1, -1, -1):
                    lj = log_lines[j]
                    if lj and lj.startswith('$') \
                            and '<~Controls.819~>' not in lj \
                            and '<~Controls.16~>' not in lj:
                        group.append((j, lj))
                        log_lines[j] = ''
                        if LogReplayer._DIE_CHANGE_LINE_RE.match(lj):
                            continue  # not snapshotted: keep walking
                        break
                log_lines[i] = ''
                if group:
                    redo_stack.append(group)
            elif '<~Controls.16~>' in line and line.startswith('$'):
                if redo_stack:
                    for j, original in redo_stack.pop():
                        log_lines[j] = original
                log_lines[i] = ''
            elif line and line.startswith('$') and redo_stack:
                # a fresh command invalidates the redo stack
                redo_stack = []
        return log_lines

    # Piece-move line in ANY encoding — built from the authoritative
    # move-template table (messages.properties): Scribe drags,
    # Controls.281 moves, the HighlightLabel client variants (81/86
    # dice, 91/96/101 political-marker walks), dice-pool bookkeeping
    # (Controls.726/731/757/762), CardBuffet.3 card retractions, ...
    _ANY_MOVE_RE = re.compile(
        r'<~(?:%s)~>([\d\s]+?)'
        r'<~(?:%s)~><~([\w.]+)~>'
        r'<~(?:%s)~><~([\w.]+)~>'
        % (_MOVE_K1_ALT, _MOVE_K2_ALT, _MOVE_K3_ALT))

    def preprocess(self, log_lines: List[str]):
        """All pre-passes over the full (stripped, non-empty) line list.
        Line numbers = 1-based indices into *log_lines*, matching the
        caller's subsequent _process_line() numbering.  Applies the undo
        pre-pass IN PLACE first (blanked lines keep numbering stable)."""
        self.undo_prepass(log_lines)
        # F11 dedupe (ruling 1.162) — after undo blanking so an undone
        # line breaks the pair, before every other pre-pass
        self.dedupe_adjacent_lines(log_lines)
        self.card_decoder.preprocess_log(log_lines)
        # All die piece ids (base map + bonus dice appearing in rolls) —
        # shared by the eye-drag and inverse-drag pre-passes
        self._all_die_ids = {str(k) for k in self.die_id_map}
        _roll_re = re.compile(r'<~Controls\.345~>([\d\s]+)<~Controls\.346~>')
        for _pl in log_lines:
            _rm = _roll_re.search(_pl)
            if _rm:
                self._all_die_ids.update(_rm.group(1).split())
        self._preprocess_army_boxes(log_lines)
        self._preprocess_eye_drags(log_lines)
        self._preprocess_inverse_drags(log_lines)
        self._preprocess_vp_waypoints(log_lines)
        self._preprocess_table_card_kills(log_lines)
        # t125 fix 7 (USER RULING 1.157): precompute the pure-noise log
        # TAIL — the earliest line index from which every remaining $
        # command is a noise-kind line (hand reveal, hunt draw, card
        # draw, track walk, tile drag, card discard) or chat.  The
        # military-victory / concession game-over signals only latch
        # inside this tail, so a false signal can never skip real play.
        n = len(log_lines)
        start = n + 1
        for i in range(n, 0, -1):
            ln = log_lines[i - 1].strip()
            if not ln or not ln.startswith('$') \
                    or ' sendTokens ' in ln \
                    or self._postgame_noise_kind(ln) is not None:
                start = i
                continue
            break
        self._eog_suffix_start = start

    def _is_die_piece(self, piece_id: str) -> bool:
        """True when the piece id is an action die (base or bonus)."""
        try:
            return int(piece_id) in self.die_id_map
        except (ValueError, TypeError):
            return False

    # ---- t125 fix 7 (USER RULING 1.157): end-of-game noise ----------
    _EOG_RE_REVEAL = re.compile(r'\breveal \d+ [A-F0-9]{8,} \d+')
    _EOG_RE_TILE_DRAW = re.compile(r'<~Controls\.624~>')
    _EOG_RE_CARD_DRAW = re.compile(
        r'<~Controls\.(?:509|491|0|549|557)~>\d+'
        r'<~Controls\.\d+~><~Game\.200[4-7]~>')
    _EOG_RE_COMP_DRAW = re.compile(r'<~Controls\.637~>')
    _EOG_RE_CARD_DISCARD = re.compile(
        r'<~Controls\.(?:281|296)~>[\d ]+<~Controls\.(?:282|297)~>'
        r'<~Controls\.(?:45|46)~>')
    _EOG_RE_TRACK_790 = re.compile(r'<~Controls\.790~>(?:247|248|252)<~')

    def _postgame_noise_kind(self, line: str):
        """Classify a line as an end-of-game noise kind, else None.

        These are the 'would we have destroyed the ring' curiosity
        actions humans take after the game is decided: revealing the
        remaining hand, drawing extra hunt tiles and walking the
        corruption for them, drawing cards, tidying tiles.  Anything
        NOT classified here still processes normally after the
        game-over latch (no real play is ever skipped)."""
        if self._EOG_RE_REVEAL.search(line):
            return 'hand_reveal'
        if self._EOG_RE_TILE_DRAW.search(line):
            return 'hunt_tile_draw'
        if self._EOG_RE_CARD_DRAW.search(line):
            return 'card_draw'
        if self._EOG_RE_COMP_DRAW.search(line):
            return 'companion_draw'
        if self._EOG_RE_CARD_DISCARD.search(line):
            return 'card_discard'
        if self._EOG_RE_TRACK_790.search(line):
            return 'track_walk'
        m = self._ANY_MOVE_RE.search(line)
        if m:
            ps = m.group(1).split()
            if ps and all(p in self._TRACK_MARKER_PIECES for p in ps):
                return 'track_walk'
            if ps and all(p in HUNT_TILE_PIECES or p in ('230', '231')
                          for p in ps):
                return 'tile_drag'
        return None

    def _preprocess_eye_drags(self, log_lines: List[str]):
        """Retracted eye allocations.

        A Shadow die dragged into the Hunt Box and dragged OUT again
        (to a dice pool or Used) before any dice roll / hunt event is an
        allocation the player took back.  Both drags are noise — without
        this the die face is corrupted to EYE(used) for the whole turn
        and the roll skips the die, losing its true face (2023-12-06 T7
        lines 695/696: dice 101+105 in, straight back out via
        Controls.281, then 103+361 allocated for real; T5 lines 468/470:
        die 103 in, dragged to Used two lines later)."""
        self._eye_drag_noise = set()  # (line_num, die_id_str)
        # 131 (2026-07-25 board-is-set T9 L662-666): BUTTON (Controls.372)
        # allocations retracted via a die return from the Hunt Box
        # (Controls.281 Game.1997 -> SA Dice, '1 eye max' correction).
        # The drag-based pending never saw them, so the turn eye count
        # ran 2 vs the true 1 and the engine's roll went a die short.
        self._eye_btn_noise = set()   # (alloc_line_num, die_id_str)
        die_ids = self._all_die_ids   # built once in preprocess()
        pool_codes = {'Game.2042', 'Game.2043'}
        out_codes = {'Game.2035', 'Game.2042', 'Game.2043'}
        pending: Dict[str, int] = {}
        pending372: Dict[str, int] = {}
        _btn_re = re.compile(r'<~Controls\.372~>(\d+)')
        for n, line in enumerate(log_lines, 1):
            if ('<~Controls.345~>' in line or '<~Controls.624~>' in line
                    or '<~Controls.795~>' in line
                    or '<~Controls.372~>' in line
                    or '<~Game.7~>' in line):
                pending.clear()  # roll/hunt/turn activity: allocations real
                _bm = _btn_re.search(line)
                if _bm is not None:
                    pending372[_bm.group(1)] = n
                else:
                    pending372.clear()
                continue
            m = self._ANY_MOVE_RE.search(line)
            if m is None:
                continue
            pieces = m.group(1).split()
            frm, to = m.group(2), m.group(3)
            if to == 'Game.1997' and frm in pool_codes:
                for p in pieces:
                    if p in die_ids:
                        pending[p] = n
            elif frm == 'Game.1997' and to in out_codes:
                for p in pieces:
                    ent = pending.pop(p, None)
                    if ent is not None:
                        self._eye_drag_noise.add((ent, p))
                        self._eye_drag_noise.add((n, p))
                    ent372 = pending372.pop(p, None)
                    if ent372 is not None:
                        self._eye_btn_noise.add((ent372, p))

    def _preprocess_inverse_drags(self, log_lines: List[str]):
        """Aborted piece drags.

        The same piece set dragged A -> B and straight back B -> A (only
        chat between) is a retracted move, not two decisions
        (2023-12-06 T3 lines 266/267: army into Lorien and back, then
        chat 'attack lorien' and the proper declaration; T7 735/736:
        Angmar <-> Ettenmoors).  Both lines are skipped; the state is
        untouched (net zero)."""
        self._skip_move_lines = set()
        # base line -> ordered [('add'|'drop', piece), ...] replayed by
        # _handle_move: walk-backs and split-drag handfuls can
        # INTERLEAVE (demanray L921-924: second handful 026 dragged in,
        # then straight back out — net absent; TBIS-style went-back-
        # went-again — net present), so naive add/drop SETS lose the
        # ordering and double-count pieces.
        self._move_piece_ops = {}

        def _map_reg(tok):
            # walk-back netting and split-drag merging apply to MAP
            # moves only: same-route handfuls to dead piles /
            # reinforcement boxes are separate records by design
            # (casualty counting, per-die musters) — merging them ate
            # dice (2025-04-06 demanray: census 9 -> 11 with the
            # ungated merge)
            try:
                return sanitize(decode_location(tok)) in board.regions
            except Exception:
                return False

        last = None  # (line_num, pieces_tuple, frm, to)
        for n, line in enumerate(log_lines, 1):
            if not line:
                continue
            m = self._ANY_MOVE_RE.search(line)
            if m is None:
                if line.startswith('<'):
                    continue  # chat doesn't break the pair window
                last = None
                continue
            pieces = tuple(sorted(m.group(1).split()))
            if any(p in self._TRACK_MARKER_PIECES for p in pieces):
                last = None  # never pair fellowship/corruption walks
                continue
            if pieces and all(p in getattr(self, '_all_die_ids', ())
                              for p in pieces):
                # dice-only drags are pool bookkeeping (turn-start
                # hunt-box retrieval via Controls.726/757 followed by a
                # re-allocation of the same die must NOT pair as an
                # aborted drag; retracted eye allocations have their own
                # pre-pass)
                last = None
                continue
            frm, to = m.group(2), m.group(3)
            if last is not None:
                ln, lp, lf, lt = last
                if pieces == lp and frm == lt and to == lf and frm != to:
                    self._skip_move_lines.add(ln)
                    self._skip_move_lines.add(n)
                    last = None
                    continue
                if (set(pieces).isdisjoint(lp) and frm == lf and to == lt
                        and frm != to and n - ln <= 3
                        and _map_reg(frm) and _map_reg(to)):
                    # SPLIT DRAG: a second handful of pieces dragged
                    # over the SAME route within a couple of lines is
                    # the same move, not a second action (2024-07-27
                    # Columbo T8 L665/666: 6R Osgiliath -> Minas Tirith,
                    # then the nazgul one line later — one Army die in
                    # the log; parsing two records burned two teacher
                    # dice and starved the L748 Cardolan move).  A
                    # genuine second same-route action has its die
                    # spend (a dice-piece drag) between, which resets
                    # the pair window.
                    self._skip_move_lines.add(n)
                    self._move_piece_ops.setdefault(ln, []).extend(
                        ('add', p) for p in pieces)
                    last = (ln, tuple(sorted(set(lp) | set(pieces))),
                            lf, lt)
                    continue
                if (set(pieces) < set(lp) and frm == lt and to == lf
                        and frm != to and _map_reg(frm) and _map_reg(to)):
                    # PARTIAL walk-back: a strict subset of the pieces
                    # dragged straight back with nothing but chat
                    # between is a drag-size correction, not a second
                    # action (2024-07-27 Columbo T7 L621/622: 10R+2N
                    # Minas Morgul -> South Ithilien, then 1R back one
                    # line later — the humans meant 9R+2N with ONE die;
                    # keeping both records burned a second Character
                    # die and starved the L629 Osgiliath move).  A
                    # genuine split-back would have a die-spend record
                    # between (which resets the pair window).  The
                    # walk-back line is skipped and its pieces dropped
                    # from the first move.
                    self._skip_move_lines.add(n)
                    self._move_piece_ops.setdefault(ln, []).extend(
                        ('drop', p) for p in pieces)
                    last = (ln, tuple(sorted(set(lp) - set(pieces))),
                            lf, lt)
                    continue
            last = (n, pieces, frm, to)

    def _preprocess_vp_waypoints(self, log_lines: List[str]):
        """A drag that WAYPOINTS through a VP-track box (map -> 'VP n' ->
        map, same pieces, within a few lines) is ONE map->map move: the
        player dropped the pieces on the VP track mid-drag (2025-06-19
        StellaRossa T6 L579-580: 6R+2E West Harondor -> VP 7 -> Pelargir
        — parsed as 8 phantom casualties + an 8-unit 'muster', hiding
        the battle advance and the Pelargir capture).  The first hop is
        skipped; the second hop's FROM is rewritten to the origin.
        Genuine VP-MARKER walks live on VP boxes (never map->VP) and are
        untouched; there-and-back pairs are already blanked by the
        inverse-drag pre-pass (lines here skip those)."""
        self._move_from_rewrites = {}
        pending = {}  # pieces_tuple -> (line, raw_from)
        _vp = re.compile(r'VP \d+$')
        for n, line in enumerate(log_lines, 1):
            if n in self._skip_move_lines:
                continue
            m = self._ANY_MOVE_RE.search(line)
            if m is None:
                continue
            pieces = tuple(sorted(m.group(1).split()))
            frm, to = m.group(2), m.group(3)
            if _vp.match(str(decode_location(to))) \
                    and is_map_region(decode_location(frm)):
                pending[pieces] = (n, frm)
            elif _vp.match(str(decode_location(frm))) \
                    and is_map_region(decode_location(to)):
                ent = pending.pop(pieces, None)
                if ent is not None and n - ent[0] <= 4:
                    self._skip_move_lines.add(ent[0])
                    self._move_from_rewrites[n] = ent[1]

    # Table-card-kill pre-pass regexes (ruling 1.149)
    _TCK_MOVE_RE = re.compile(
        r'^\$(\w+).*?<~(?:Scribe\.141|Controls\.281)~>([\d\s]+?)'
        r'<~(?:Scribe\.142|Controls\.282)~><~([\w.]+)~>'
        r'<~(?:Scribe\.143|Controls\.283)~><~([\w.]+)~>')
    _TCK_644_RE = re.compile(
        r'<~Controls\.644~>([\d\s]+?)<~Controls\.645~><~([\w.]+)~>')
    _TCK_DISC_RE = re.compile(
        r'<~Controls\.281~>([\d\s]+?)<~Controls\.282~><~Controls\.(45|46)~>')
    _TCK_ROLL_RE = re.compile(r'^\$(\w+)<~Controls\.345~>([\d ]+)')

    def _preprocess_table_card_kills(self, log_lines: List[str]):
        """USER 2026-07-19 (ruling 1.149): stage deliberate OPPONENT
        table-card kills as single discard_table_card records.

        A kill = a play-on-table card (106 A Power too Great / 145 The
        Power of Tom Bombadil / 202 The Palantír of Orthanc) dragged off
        its Board slot (2029-2034) to a non-slot zone BY THE OPPONENT of
        its owner, with the rule's PAYMENT signature in a small +/- line
        window:
          * SHADOW kills 106/145: one ARMY-type + one Character card
            discarded from the Shadow hand (Controls.281 -> Controls.46).
            (A Muster-type strategy card does NOT pay — only Army-type.)
          * FP kills 202: the card rides a Controls.644 kill to 'Used'
            bundled with an FP action die (the payment die), so the removal
            is provably die-PAID; payment='ring' when an Elven Ring goes
            active->used in the tight window, else 'will' (Will of the West).

        The destination-not-a-slot + drag-by-opponent tests reject the
        owner shuffling his own card between slots or removing it on Saruman
        elimination; the die-paid-644 test rejects retracted drags (dragged
        then back, e.g. Matzerotti has NO 644 on the card) and turn-boundary
        card-alone tidies ('644 202' with no die).  Results land in
        self._table_card_kills (keyed by the drag line number) and
        self._table_kill_ring_lines (the folded ring drags)."""
        n = len(log_lines)
        # Player name -> side, from the roll lines' die ids.
        pside: Dict[str, Side] = {}
        for ln in log_lines:
            m = self._TCK_ROLL_RE.match(ln)
            if not m or m.group(1) in pside:
                continue
            for _i in m.group(2).split():
                try:
                    v = int(_i)
                except ValueError:
                    continue
                if 99 <= v <= 108:
                    pside[m.group(1)] = Side.SHADOW
                    break
                if 210 <= v <= 213:
                    pside[m.group(1)] = Side.FREE_PEOPLES
                    break
        for idx in range(n):
            line = log_lines[idx]
            if 'Game.20' not in line:
                continue
            if (idx + 1) in self._skip_move_lines:
                continue  # retracted (inverse-drag pre-pass)
            m = self._TCK_MOVE_RE.match(line)
            if not m:
                continue
            pids = m.group(2).split()
            if len(pids) != 1 or pids[0] not in _TABLE_KILL_PIDS:
                continue
            frm, to = m.group(3), m.group(4)
            if frm not in _TABLE_SLOT_ZONES or to in _TABLE_SLOT_ZONES:
                continue
            if to in ('Controls.45', 'Controls.46'):
                continue  # returned to a hand = a play RETRACT, not a kill
            tgt = pids[0]
            info = _table_kill_card_info(tgt)
            if info is None or info[0] is None:
                continue
            card_id, card_name, killer = info
            if pside.get(m.group(1)) != killer:
                continue  # only the OPPONENT of the owner can kill it
            lo, hi = max(0, idx - 10), min(n, idx + 11)
            if killer == Side.SHADOW:
                has_army = has_char = False
                for w in log_lines[lo:hi]:
                    dm = self._TCK_DISC_RE.search(w)
                    if not dm or dm.group(2) != '46':
                        continue
                    for p in dm.group(1).split():
                        try:
                            ip = int(p)
                        except ValueError:
                            continue
                        if ip in _SA_ARMY_CARD_PIDS:
                            has_army = True
                        elif 158 <= ip <= 209 and _SA_PID_TYPES[ip - 158] == 'C':
                            has_char = True
                if not (has_army and has_char):
                    continue
                params = {'card_id': card_id, 'card_name': card_name}
            else:  # FP kills the Palantir: a Will of the West die alone, or
                # any die + one Elven Ring.  Require the die-PAID removal
                # signal: the card rides a Controls.644 kill to 'Used'
                # bundled WITH an FP action die (the payment die, e.g.
                # '644 211 202').  This rejects the ambiguous card-alone
                # removals ('644 202') — Saruman-elimination auto-discards
                # and turn-boundary tidies where no die was spent.
                confirmed = False
                for w in log_lines[lo:hi]:
                    km = self._TCK_644_RE.search(w)
                    if not km:
                        continue
                    _kp = km.group(1).split()
                    if tgt in _kp and any(x in _FP_DIE_PIDS for x in _kp):
                        confirmed = True
                        break
                if not confirmed:
                    continue
                # payment: an Elven Ring active->used TIGHTLY around the
                # drag is the ring cost; otherwise a Will of the West die.
                rlo, rhi = max(0, idx - 4), min(n, idx + 5)
                ring_line = None
                for j in range(rlo, rhi):
                    mj = self._TCK_MOVE_RE.match(log_lines[j])
                    if not mj:
                        continue
                    rp = mj.group(2).split()
                    if len(rp) == 1 and rp[0] in RING_PIECES \
                            and mj.group(3) in RING_ACTIVE_LOCS \
                            and mj.group(4) in RING_USED_LOCS:
                        ring_line = j + 1
                        break
                payment = 'ring' if ring_line is not None else 'will'
                params = {'card_id': card_id, 'card_name': card_name,
                          'payment': payment}
                if ring_line is not None:
                    self._table_kill_ring_lines.add(ring_line)
            self._table_card_kills[idx + 1] = {
                'params': params, 'side': killer, 'pid': tgt}

    def _preprocess_army_boxes(self, log_lines: List[str]):
        """Resolve army parking in the Mordor-track/army boxes.

        UNIT pieces Scribe-moved from map region A into a track box, and
        later out of the box to B, are one parked army whose true map
        location X is invisible (the numbered counter isn't decodable).
        Inference, verified vs 2023-12-07_001647 (Barad-Dur -> box5 ->
        Morannon = via Gorgoroth, the unique common neighbor):
          - A == B: the army never moved (both lines = state noise).
          - B adjacent to A: entry was pure parking (army stayed at A);
            exit is the real move A -> B.
          - else X = the UNIQUE common neighbor of A and B: entry was the
            move A -> X, exit is X -> B.  (No unique X -> unresolved; the
            lines replay via the no-rewrite guard in _handle_move.)
        Exits to non-map areas (casualties etc.) resolve to the entry
        region A."""
        self._box_rewrites = {}
        # box_code -> {'line': n, 'from': A, 'pieces': set, 'exits': []}
        # or None (dirty).  Boxes support PARTIAL exits: the Barad-Dur
        # army parked at T1 loses its Nazgul to a T3 flight and the
        # units leave for Morannon at T5 (2023-12-06 lines 100/245/464)
        # — the army's true location X is inferred from the UNIT exits
        # (nazgul-only exits fly anywhere and place no constraint).
        pending = {}

        def _finalize(ent):
            a_name = ent['from']
            a_san = sanitize(a_name)
            adj_a = {sanitize(x) for x in self.board.adjacent(a_san)}
            cand = None
            for (_n2, ps, b_name) in ent['exits']:
                b_san = sanitize(b_name) if b_name else None
                if b_san is None or b_san not in board.regions:
                    continue  # off-map exit (casualties): no constraint
                if ps and all(get_unit_type_from_piece_id(
                        p, self.piece_types)[1] == 'NAZGUL' for p in ps):
                    continue  # nazgul fly anywhere
                if b_san == a_san or b_san in adj_a:
                    c = {a_san}
                else:
                    adj_b = {sanitize(x) for x in self.board.adjacent(b_san)}
                    c = adj_a & adj_b
                cand = c if cand is None else (cand & c)
            if cand is None:
                # USER RULING 1.156 (t125): no unit-exit constraint
                # (nazgul-only / off-map exits) — box units are IN
                # GORGOROTH.  The old stay-at-A guess contradicted the
                # client board and was prefix-unstable (Rizino1 cp250:
                # only the nazgul exit in the prefix -> army wrongly
                # kept at Barad-Dur).  Fall back to A only when the
                # entry region cannot reach Gorgoroth.
                x_san = 'gorgoroth' if (a_san == 'gorgoroth'
                                        or 'gorgoroth' in adj_a) \
                    else a_san
            elif len(cand) == 1:
                x_san = next(iter(cand))
            elif 'gorgoroth' in cand:
                x_san = 'gorgoroth'   # ruling 1.156 tiebreak
            else:
                return  # ambiguous — leave all lines unrewritten
            if x_san == a_san:
                self._box_rewrites[ent['line']] = ('noop', None)
            else:
                self._box_rewrites[ent['line']] = ('entry', x_san)
            for (n2, _ps, b_name) in ent['exits']:
                b_san = sanitize(b_name) if b_name else None
                if b_san == x_san:
                    self._box_rewrites[n2] = ('noop', None)
                else:
                    self._box_rewrites[n2] = ('exit', x_san)

        for n, line in enumerate(log_lines, 1):
            m = self._SCRIBE_MOVE_RE.search(line)
            if m is None:
                continue
            pieces = m.group(1).split()
            if not pieces or all(p in self._TRACK_MARKER_PIECES
                                 for p in pieces):
                continue
            frm, to = m.group(2), m.group(3)
            if to in self._ARMY_BOX_CODES and frm not in self._ARMY_BOX_CODES:
                a_name = decode_location(frm)
                if sanitize(a_name) in board.regions \
                        and to not in pending:
                    pending[to] = {'line': n, 'from': a_name,
                                   'pieces': set(pieces), 'exits': []}
                else:
                    pending[to] = None  # non-map source / re-entry: dirty
            elif frm in self._ARMY_BOX_CODES \
                    and to not in self._ARMY_BOX_CODES:
                ent = pending.get(frm)
                if ent is None:
                    pending.pop(frm, None)
                    continue
                b_name = decode_location(to)
                ent['exits'].append((n, list(pieces), b_name))
                ent['pieces'] -= set(pieces)
                if not ent['pieces']:
                    _finalize(pending.pop(frm))
        for ent in pending.values():
            if isinstance(ent, dict) and ent['exits']:
                _finalize(ent)

    def parse_and_replay(self, log_path: str):
        """Parse log file and replay all events."""
        # First pass: preprocess for card decoding
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_lines = [line.strip() for line in f if line.strip()]

        # Undo pre-pass (blanks undone lines, keeps numbering stable)

        self.preprocess(log_lines)

        # Second pass: replay all events
        for line_num, line in enumerate(log_lines, 1):
            self.line_num = line_num
            self._process_line(line)

        if self.verbose:
            print(f"\n=== REPLAY COMPLETE ===")
            print(f"Final turn: {self.state.turn}")
            print(f"Errors: {len(self.errors)}")

    def _process_line(self, line: str):
        """Process a single log line."""
        # Strip checksum
        line = re.sub(r'\s+[A-F0-9]{6}$', '', line)
        line = re.sub(r'\s+null$', '', line)

        # ---- t125 fix 7 (USER RULING 1.157): end-of-game noise ------
        # Lazy latch: an armed military-victory/concession signal only
        # fires inside the precomputed pure-noise tail.
        if self._game_over is None and self._eog_pending is not None \
                and not self._final_hunt_pending \
                and self.line_num >= self._eog_suffix_start:
            self._game_over = {'line': self.line_num,
                               'reason': self._eog_pending}
            self.log(f"GAME OVER ({self._eog_pending}), line "
                     f"{self.line_num}: trailing hand-reveals/hunt "
                     f"draws are noise [ruling 1.157]")
        if self._game_over is not None and line:
            _nk = self._postgame_noise_kind(line)
            if _nk is not None:
                self.log(f"POST-GAME NOISE skipped ({_nk}) "
                         f"[ruling 1.157]")
                return

        # Card DISCARD (USER-verified anchors 2026-07-09, Skaarj game):
        # two face-down hand-exit grammars — Controls.281/282/283
        # (piece(s), from hand 45/46, to zone) and Controls.296/297
        # (piece, hand). A CARD piece (FP 106-157, SA 158-209) leaving
        # a hand with no reveal on the line = discard. Face-up exits
        # ride the reveal grammars to Current FP/SA Card = PLAYS.
        for _dm in re.finditer(
                r'<~Controls\.281~>([\d ]+)\s*<~Controls\.282~>'
                r'<~Controls\.(45|46)~>'
                r'(?:<~Controls\.283~><~([\w.]+)~>)?', line):
            _pids = [p for p in _dm.group(1).split() if p.isdigit()]
            _frm = _dm.group(2)
            _lo, _hi = (106, 157) if _frm == '45' else (158, 209)
            for _pid in _pids:
                if not (_lo <= int(_pid) <= _hi):
                    continue
                if ('reveal ' + _pid + ' ') in line:
                    continue  # played face-up, not a discard
                self._handle_card_zone_discard(_pid, _frm)

        # Card RETRACTION: a committed card taken BACK out of the play
        # area into a hand (he changed his mind before revealing the
        # one he meant).  The hand EXIT above has been booked since the
        # discard grammars went in; this is its RE-ENTRY TWIN, without
        # which a card he physically holds stays modelled as played.
        # SOURCE ZONE IS THE DISCRIMINATOR: a Current Card zone
        # (Game.2036/2037) is a retraction, a DECK zone is an ordinary
        # draw and is handled by the draw grammars.  See
        # _handle_card_retraction for the specimen and blast radius.
        for _rm in re.finditer(
                r'<~[A-Za-z]+\.\d+~>([\d ]+?)\s*<~[A-Za-z]+\.\d+~>'
                r'<~Game\.(203[67])~><~[A-Za-z]+\.\d+~>'
                r'<~Controls\.(45|46)~>', line):
            _hand = _rm.group(3)
            _lo, _hi = (106, 157) if _hand == '45' else (158, 209)
            for _pid in _rm.group(1).split():
                if _pid.isdigit() and _lo <= int(_pid) <= _hi:
                    self._handle_card_retraction(_pid, _hand)

        # OFFLINE PLAY of a revealing card (AI-emitted logs only): the
        # bridge renders event plays as Controls.452 pid -> the Current
        # Card area (Game.2036/2037) with no reveal grammar, so the
        # _REVEALING_EVENT_CARDS arming in _handle_card_reveal never
        # runs and a card-caused fellowship reveal's pawn walk was
        # classified as a DECLARE (hidden stuck + flip pair eaten;
        # seed-42 tick 79).  Human logs use 452 only for face-down
        # discards to the card DEAD piles (2039/2041) — the play-area
        # target makes this pattern corpus-inert.  Card identity is
        # fixed by piece id offline: 179 = Nazgul Search, 201 = The
        # Nazgul Strike! (the two _REVEALING_EVENT_CARDS).
        if re.search(r'<~Controls\.452~>(?:179|201)<~Controls\.453~>'
                     r'<~Controls\.\d+~><~Controls\.454~>'
                     r'<~Game\.203[67]~>', line):
            self._hunt_reveal_pending = True


        # Capture human-readable % lines (used by muster/hunt tile parsing)
        if line.startswith('%'):
            self._last_readable_line = line
            return

        # Skip auto-setup lines (initial game state setup by engine)
        # BUT allow <auto> lines that are CARD PLAYS.
        #
        # 2026-08-18: this was a hand-written list of three op numbers
        # (Controls.3 / 866 / 861) and Controls.888 - the same grammar,
        # a SHADOW card leaving the hand for the Current SA Card slot -
        # was not on it.  An unstripped <auto> line RETURNS EARLY, so
        # the whole line was invisible: no reveal, and no hand exit.
        # MEASURED: 3,879 corpus games / 5,556 lines carry one, and
        # replaying six specimens with and without the strip changed
        # the Shadow hand in FIVE of them - always one card too HIGH
        # as shipped, because the play was never booked
        # (streak/probe_888_auto_strip.py).
        #
        # THE CLASS, NOT THE INSTANCE.  A card play is identifiable by
        # what it DOES: it says 'reveal' and it moves a piece from a
        # HAND zone to a CURRENT CARD zone.  Censused over 155,532
        # <auto> lines (streak/census_auto_reveal_family.py): that
        # predicate admits exactly the four ops of this family - 866,
        # 3, 861 and 888 - and NOTHING else, so a future sibling is
        # covered without another edit here.  Every other <auto> line
        # (Scribe.141 43,731, Controls.323, Controls.281 ...) is
        # auto-SETUP and must keep being skipped, which the zone test
        # guarantees: setup lines carry no hand->Current-Card pair.
        if line.startswith('<auto>'):
            _hand9 = ('<~Controls.45~>' in line
                      or '<~Controls.46~>' in line)
            _play9 = ('<~Game.2036~>' in line
                      or '<~Game.2037~>' in line)
            if 'reveal' in line and _hand9 and _play9:
                line = line[len('<auto>'):]  # Strip prefix so it won't hit chat handler
            else:
                return
        if line.startswith('$base') or line.startswith('$TEST'):
            return

        # Skip game/system messages (but note dice rolls: a battle window
        # with NO rolls is the pre-battle-retreat convention's signature)
        if line.startswith('<game>') or line.startswith('<replay>'):
            if '<~Controls.1~>' in line:
                self._rolls_since_die = True
                # 130: remember WHERE the last roll happened (the
                # battle-noise move-in gate needs "rolls since attack")
                self._last_roll_line = getattr(self, 'line_num', 0)
                # A roll closes the pre-roll combat-card window (units
                # onto a card after this are casualties, not choices)
                self._cc_pre_roll.clear()
            return

        # Extract $PlayerName prefix for player->side mapping
        player_prefix = re.match(r'^\$(\w+)', line)
        if player_prefix:
            self._current_line_player = player_prefix.group(1)

        # Turn marker
        turn_match = re.search(r'<~Game\.7~>(\d+)', line)
        if turn_match:
            new_turn = int(turn_match.group(1))
            if new_turn != self.state.turn:
                self._handle_turn_change(new_turn)
            return

        # Chat message - check for "pass" before skipping
        chat_match = re.match(r'^<(\w+)>\s*(.*)', line)
        if chat_match:
            player = chat_match.group(1)
            msg = chat_match.group(2).strip().lower()
            if player in ('auto', 'game', 'replay'):
                return
            if msg == 'pass':
                side = self._player_to_side.get(player)
                if side:
                    self.log(f"{side.name} player passes")
                    self._record_action('pass', {}, side)
                return
            # Combat-card decline: '<player> nc' while a battle die-window
            # is open (a die was just devoted to the action / a combat
            # card was just selected). This is a real decision the engine
            # models via the attacker/defender combat-card phases.
            toks = msg.split()
            if toks and re.fullmatch(r'[0-9a-f]{6}', toks[-1]):
                toks = toks[:-1]  # strip the VASSAL line hash
            # t125 fix 7 (ruling 1.157): concession chat ARMS the
            # game-over signal (latches only in the pure-noise tail)
            if toks and ('gg' in toks or 'concede' in msg
                         or 'resign' in toks):
                self._eog_pending = self._eog_pending or 'concession'
            # AI-bridge EVENT-PLAY announce (t121): the live bridge
            # chats '<nick> plays <card name>' immediately before the
            # client's choose/playchosen reveal.  Arm the announce so
            # the reveal handler keeps a name-matching commit as the
            # EVENT play (the playchosen button emits the combat-commit
            # grammar even for event plays).  Human 'plays ...' chats
            # that never precede a name-matching reveal are inert.
            if len(toks) > 1 and toks[0] == 'plays':
                _pn9 = {_normalize_name(' '.join(toks[1:]))}
                if len(toks) > 2 and toks[1] in ('character',
                                                 'strategy'):
                    # early bridge builds announced 'plays strategy X'
                    # (deck word from the engine card id) — accept the
                    # deck-word-stripped variant too
                    _pn9.add(_normalize_name(' '.join(toks[2:])))
                self._announced_play = (_pn9,
                                        getattr(self, 'line_num', 0))
            if toks and toks[-1] == 'nc' and \
                    getattr(self, 'line_num', 0) - \
                    self._battle_window_line <= 10:
                side = self._player_to_side.get(player)
                if side is not None:
                    # The nc chat itself may CARRY the attack declaration
                    # ('fords from orthanc NC', 2025-06-19 StellaRossa T5
                    # L408) — register the hint FIRST so the decline (and
                    # the following defender reveal) bind to the RIGHT
                    # battle instead of a stale _combat_region.
                    if len(toks) > 1:
                        self._scan_chat_for_battle_hint(
                            ' '.join(toks[:-1]))
                    region = (getattr(self, '_battle_hint_region', None)
                              or self._battle_synth_region
                              or self._combat_region or '?')
                    # (129 v1 nc-synth REVERTED 2026-07-25: synthesizing
                    # at the nc chat re-ordered attack records vs
                    # adjacent moves in 27 windows of PeterM alone and
                    # regressed it 0->9.  v2 synthesizes at the swap-
                    # removal site instead - see _handle_..304.)
                    self.action_log.append(('combat_card_none',
                                            {'region': region}, side))
                    self.log(f"{side.name} declines combat card (nc)")
                return
            # Combat-card effect forfeit: '<player> ffN' during a battle
            # window = 'forfeit N' (Dread and Despair: forfeit N points
            # of Nazgul leadership; game0 T9 'ff3' twice — FP rolled 2
            # dice instead of 5). A learned combat_effect_choice decision.
            _ff = re.fullmatch(r'ff\s*(\d)', ' '.join(toks)) \
                if toks else None
            if _ff and getattr(self, 'line_num', 0) - \
                    self._battle_window_line <= 10:
                side = self._player_to_side.get(player)
                if side is not None:
                    region = (getattr(self, '_battle_hint_region', None)
                              or self._battle_synth_region
                              or self._combat_region or '?')
                    if region == '?':
                        # t138 (engine-check family 3): the preceding
                        # combat_card/army_attack record usually knows
                        # the battle region the ff chat forgot
                        for _at8, _p8, _s8 in reversed(
                                self.action_log[-8:]):
                            _r8 = (_p8 or {}).get('region') \
                                or (_p8 or {}).get('to')
                            if _at8 in ('combat_card', 'army_attack') \
                                    and _r8 and _r8 != '?':
                                region = _r8
                                self.log(f"  (ff region from last "
                                         f"{_at8}: {region})")
                                break
                    self.action_log.append((
                        'combat_effect_choice',
                        {'count': int(_ff.group(1)),
                         'card_name':
                             self._last_combat_card_name.get(side, ''),
                         'region': region}, side))
                    self.log(f"{side.name} combat-effect forfeit "
                             f"{_ff.group(1)} (ff)")
                return
            # fix 76b (GAME19 T8): players announce presses in chat
            # ("press", "press draw") right before paying the elite —
            # stamp the line so the 307-sourced press path (replacement
            # regular from REINFORCEMENTS, which Block-1's 308-only
            # detection misses) can emit a nation-stamped record.
            if 'press' in msg.lower().split():
                self._last_press_chat_line = self.line_num
            # fix 82 (game-16 canary): an UNAMBIGUOUS tile-token chat
            # right after a hunt-tile draw names the tile — override the
            # version-unstable static piece map ('3r' etc.; plain digits
            # are NOT used, they collide with hit-count announcements).
            _tt82 = msg.strip().lower().rstrip('.!')
            if (re.fullmatch(r'-?\d+r', _tt82) or _tt82 == 'eye') \
                    and getattr(self, '_last_tile_draw_params', None) \
                    is not None \
                    and self.line_num - getattr(
                        self, '_last_tile_draw_line', -99) <= 4:
                _tp82c = self._last_tile_draw_params
                if str(_tp82c.get('tile_name', '')).lower() != _tt82:
                    self.log(f"  (tile name corrected by chat: "
                             f"{_tp82c.get('tile_name')} -> {_tt82})")
                    _tp82c['tile_name'] = ('Eye' if _tt82 == 'eye'
                                           else _tt82)
                    _tp82c['is_reveal'] = (_tt82 == 'eye'
                                           or _tt82.endswith('r'))
                self._last_tile_draw_params = None
            # Attack declarations in chat: players call the target region
            # ("MT", "DA", "osg time", "Lossarnach", "Fords from Orthanc")
            # before rolling — the most reliable battle-target signal
            self._scan_chat_for_battle_hint(msg)
            return

        # Die consumption (Shadow): Controls.276 moves die from pool to used
        die_consume_match = re.search(
            r'<~Controls\.276~>(\d+)<~Controls\.277~><~[\w.]+~><~Controls\.278~><~Game\.2035~>', line)
        if die_consume_match:
            die_id = int(die_consume_match.group(1))
            self._track_die_consumed(die_id)
            return

        # Die consumption (FP): HighlightLabel.86 moves die from pool to used
        hl_die_match = re.search(
            r'<~HighlightLabel\.86~>(\d+)<~HighlightLabel\.87~><~[\w.]+~><~HighlightLabel\.88~><~Game\.2035~>', line)
        if hl_die_match:
            die_id = int(hl_die_match.group(1))
            self._track_die_consumed(die_id)
            return

        # Die consumption (SH): HighlightLabel.81 variant (krechet's and
        # Upward's clients, 2025-04-15 line 49 / 2025-01-10 line 52).
        # Unhandled, the SH die never attaches to its action record —
        # the T1 'political Isengard' advance became die-less, was
        # dropped by the auto-advance heuristic, Isengard never reached
        # At War in replay, and the (correct) Saruman entry rule then
        # rejected the logged muster.
        hl_sh_die_match = re.search(
            r'<~HighlightLabel\.81~>(\d+)<~HighlightLabel\.82~><~[\w.]+~><~HighlightLabel\.83~><~Game\.2035~>', line)
        if hl_sh_die_match:
            die_id = int(hl_sh_die_match.group(1))
            self._track_die_consumed(die_id)
            return

        # Die consumption (FP variant): Controls.271 moves die from the
        # FP pool to Used (2023-12-06: Rizino1's client encodes ALL 42
        # FP die usages this way; unhandled, every FP die stays 'unused'
        # and the pass-legality comparison is wrong all game)
        fp_die_match = re.search(
            r'<~Controls\.271~>(\d+)<~Controls\.272~><~[\w.]+~>'
            r'<~Controls\.273~><~Game\.2035~>', line)
        if fp_die_match:
            self._track_die_consumed(int(fp_die_match.group(1)))
            return

        # Die face change (Elven Ring usage, or manual correction):
        # "$Sirrus233 changes 362  to C". Without this, ring-changed dice
        # keep their rolled face and actions taken with the new face look
        # illegal to the engine (e.g. Nazgul moved with an Army->Character
        # die).
        die_change_match = re.search(
            r'\bchanges\s+(\d+)\s+to\s+([ACEHMPW])\b', line)
        if die_change_match:
            die_id = int(die_change_match.group(1))
            face = DIE_CHANGE_LETTERS.get(die_change_match.group(2))
            mapping = self.die_id_map.get(die_id)
            if face is not None and mapping is not None:
                side, die_idx = mapping
                if die_idx < len(self.state.dice[side]):
                    old = self.state.dice[side][die_idx].current
                    self.state.dice[side][die_idx].current = face
                    self.log(f"Die face changed: {side.name} die {die_idx} "
                             f"{old.name if old else '?'} -> {face.name} "
                             f"(ring/manual)")
            return

        # Hunt tile draw: Controls.624 moves tile from hunt pool to display
        hunt_tile_match = re.search(
            r'<~Controls\.624~>(\d+)<~Controls\.625~><~([\w.]+)~><~Controls\.626~><~([\w.]+)~>', line)
        if hunt_tile_match:
            # t125 fix 7: inside the final-hunt window the FIRST draw
            # is the decisive Cracks-of-Doom hunt (real play); a SECOND
            # draw means the decisive tile resolved without a
            # corruption walk (0 damage) — the ring is destroyed and
            # this draw is curiosity noise.
            if self._final_hunt_pending and self._final_hunt_drawn \
                    and self._game_over is None:
                self._final_hunt_pending = False
                self._game_over = {'line': self.line_num,
                                   'reason': 'ring_destroyed'}
                self.log("GAME OVER: ring destroyed (decisive tile "
                         "resolved damage-free) [ruling 1.157]")
                self.log("POST-GAME NOISE skipped (hunt_tile_draw) "
                         "[ruling 1.157]")
                return
            if self._final_hunt_pending:
                self._final_hunt_drawn = True
            tile_id = hunt_tile_match.group(1)
            from_loc = hunt_tile_match.group(2)
            dest = hunt_tile_match.group(3)
            # Look up tile name from empirically verified mapping
            tile_name = HUNT_TILE_PIECES.get(tile_id, f"tile {tile_id}")
            # Reveal-ness comes from the TILE itself when known (the old
            # destination heuristic — Game.2036 vs 2037 card areas — is
            # just where the player happened to drop the tile; game
            # 2023-12-07_001647 put the '2r' reveal tile on 2037)
            low = str(tile_name).strip().lower()
            sp = next((t for k, t in gs.SPECIAL_HUNT_TILES.items()
                       if k.lower() == low), None)
            if sp is not None:
                is_reveal = bool(getattr(sp, 'reveal', False))
            elif re.fullmatch(r'-?\d+\s*r', low):
                is_reveal = True
            elif low.startswith('eye'):
                # the Eye tile IS a reveal (its artwork is the reveal
                # icon; engine tile.reveal=True; Gollum cannot block it).
                # The old False classification was masked in the corpus
                # because humans walk the pawn while _in_hunt_resolution
                # is still set — the engine-master bridge defers the
                # placement past the damage decision, which exposed it.
                is_reveal = True
            elif re.fullmatch(r'-?\d+', low):
                is_reveal = False
            else:
                is_reveal = '2036' in dest
            self.log(f"Hunt tile drawn: {tile_name} (reveal={is_reveal})")
            _tp82 = {
                'tile_id': tile_id,
                'tile_name': tile_name,
                'is_reveal': is_reveal,
            }
            self._record_action('hunt_tile_draw', _tp82, Side.SHADOW)
            # fix 82 (game-16 canary): tile piece IDs are NOT stable
            # across VASSAL module versions (2025 module: piece 219 =
            # '3r', but the census-era static map says 'Eye').  Players
            # announce the tile in chat right after the draw — keep a
            # handle so an unambiguous tile-token chat ('3r', '0r',
            # 'eye') within a few lines can correct the record.
            self._last_tile_draw_params = _tp82
            self._last_tile_draw_line = self.line_num
            # t125 fix 2: the draw takes the tile out of state.hunt_pool
            # (in-flight on the display area until discarded/removed)
            self._tile_to_zone(tile_id, decode_location(dest))
            self._in_hunt_resolution = True
            # The reveal PLACEMENT (pawn walk) may come after the damage
            # resolution (guide sacrifice etc.) clears the hunt flag —
            # track it separately.  AI-BRIDGE LOGS ONLY (t121, gated on
            # _ai_bridge_log): the bridge batches several client draws
            # BEFORE the placement lines (collapsed hunt chains — g1
            # T6: '1r' then '2' drawn back-to-back, walk after), so a
            # later non-reveal tile must not clear an armed reveal
            # whose pawn walk has not landed yet.  Corpus logs keep the
            # original overwrite semantics — humans resolve each tile
            # before the next draw, and the OR variant leaked an armed
            # flag across chains in certified Columbo/Adobim T8
            # (canary catch); the gate keeps them byte-identical.
            if getattr(self, '_ai_bridge_log', False):
                self._hunt_reveal_pending = (
                    getattr(self, '_hunt_reveal_pending', False)
                    or bool(is_reveal))
            else:
                self._hunt_reveal_pending = bool(is_reveal)
            # Damage tiles open a hunt-damage decision for FP: take the
            # corruption, sacrifice the guide, or a random companion.
            # Resolution is synthesized from what happens next.
            try:
                dmg = int(str(tile_name).strip().rstrip('r!'))
            except (ValueError, TypeError):
                dmg = 0
            # The Breaking of the Fellowship: the tile drawn inside the
            # card window counts SEPARATIONS, not hunt damage - opening
            # a damage ctx here FABRICATED a hunt_take_corruption
            # decision the game never contained (GAME21 rendered
            # L261-264: draw '2' -> Merry+Pippin separate -> discard,
            # NO corruption walk; the synthesized +2 was then force-fed
            # to the engine by the teacher)
            # The gate applies only while the BoF card window is still
            # OPEN (_card_window_side is cleared on die consumption);
            # _card_window_card alone is never cleared and would
            # suppress every later real hunt draw.
            if (getattr(self, '_card_window_side', None) is not None
                    and getattr(self, '_card_window_card', None)
                    == 'The Breaking of the Fellowship'):
                self.log("BoF tile draw: separation count, "
                         "no hunt-damage ctx")
                return
            if dmg > 0:
                self._hunt_ctx = {'damage': dmg, 'resolved': False}
            elif low.startswith('eye') or 'ring is mine' in low:
                # Eye tiles: damage = hunt-roll successes (>= 1 whenever
                # a tile is drawn) — the FP damage decision is real, with
                # dynamic damage the engine recomputes (2023-03-15 T1
                # line 74/77: Eye tile drawn, Gandalf guide-sacrificed;
                # without a ctx the kill recorded no decision)
                self._hunt_ctx = {'damage': None, 'resolved': False}
            return

        # Dice roll: <~Controls.345~>dice_ids <~Controls.346~>values
        dice_match = re.search(r'<~Controls\.345~>([\d\s]+)<~Controls\.346~>([\d\s]+)', line)
        if dice_match:
            dice_ids = dice_match.group(1).strip().split()
            values = dice_match.group(2).strip().split()
            self._handle_dice_roll(dice_ids, values)
            return

        # Piece movement in ANY encoding — alternation generated from the
        # authoritative move-template table (messages.properties, see
        # gen_move_templates.py).  Client builds encode the SAME piece
        # move with different key triples (Scribe.141, Controls.281,
        # HighlightLabel.81/86 dice, HighlightLabel.91/96/101 political
        # marker walks, dice-pool bookkeeping Controls.726/731/757/762,
        # die refunds Controls.736/767, CardBuffet.3 card moves);
        # the hand-collected alternation missed the HighlightLabel
        # political walks (2023-03-10 Terminatorhenk T2 Rohan-to-war,
        # O.4).  Die/marker/card piece ids are routed by the dedicated
        # branches inside _handle_move.
        move_match = self._ANY_MOVE_RE.search(line)
        if move_match:
            pieces = move_match.group(1).strip().split()
            raw_from = move_match.group(2)
            raw_to = move_match.group(3)
            if raw_from == raw_to:
                # no-op tidy drag (same source and destination box):
                # 2024-07-27 Columbo T6 L494-496 dragged dice+token
                # 2042->2042 three times — each spawned a PHANTOM
                # use_token record and flipped the token state off,
                # so the REAL T8 card-draw token use found no token
                return
            # TABLE-CARD KILL (USER 2026-07-19, ruling 1.149): a deliberate
            # removal of an OPPONENT's play-on-table card.  The whole event
            # (drag off the table + its payment signature — 2 Shadow hand
            # discards, or a Will/Elven-Ring FP spend) is resolved by the
            # _preprocess_table_card_kills pre-pass (it needs the +/- line
            # window the single-pass handler cannot see).  Emit ONE
            # discard_table_card here; the pending die attaches as usual and
            # any Elven-Ring payment drag is folded below so nothing double-
            # counts.
            _tck = self._table_card_kills.get(self.line_num)
            if _tck is not None:
                self._record_action('discard_table_card',
                                     dict(_tck['params']), _tck['side'])
                self.log(f"Table-card KILL: piece {_tck['pid']} by "
                         f"{_tck['side'].name} "
                         f"(payment={_tck['params'].get('payment', 'discards')})")
                return
            # Elven-ring pieces dragged between ring slots (Scribe-drag
            # clients, e.g. 2023-12-06 Rizino1): active slot -> used slot
            # = the FP player USES that ring (user ruling 2026-07-03 —
            # a real rule, a learned decision).  Used slot -> anywhere =
            # the Shadow player collecting the used ring ('collect'
            # record: only a die-face change nearby proves a Shadow USE;
            # the closed-loop schedule build decides).  Ring pieces are
            # never units.
            if len(pieces) == 1 and pieces[0] in RING_PIECES:
                ring_name = RING_PIECES[pieces[0]]
                # Elven Ring spent AS THE PAYMENT for a Palantir kill
                # (ruling 1.149): the ring use is recorded by that
                # composite discard_table_card (payment='ring'), so fold
                # this drag — update the ring counters for parser state
                # but emit NO standalone elven_ring record (else the ring
                # is consumed twice in the engine replay).
                if self.line_num in self._table_kill_ring_lines \
                        and raw_from in RING_ACTIVE_LOCS \
                        and raw_to in RING_USED_LOCS:
                    self.state.fp_elven_rings = max(
                        0, self.state.fp_elven_rings - 1)
                    self.state.sh_elven_rings += 1
                    self.log(f"Elven Ring used (Palantir-kill payment, "
                             f"folded into discard_table_card): {ring_name}")
                    return
                _flight = getattr(self, '_ring_in_flight', None)
                if _flight is None:
                    _flight = self._ring_in_flight = {}
                if raw_from in RING_ACTIVE_LOCS and raw_to in RING_USED_LOCS:
                    self.state.fp_elven_rings = max(
                        0, self.state.fp_elven_rings - 1)
                    self.state.sh_elven_rings += 1
                    self.log(f"Elven Ring used: {ring_name}")
                    self._record_action('elven_ring', {'ring': ring_name},
                                        Side.FREE_PEOPLES)
                elif (raw_from in RING_ACTIVE_LOCS
                        and raw_to not in RING_ACTIVE_LOCS):
                    # two-hop use (2026-01-05 kapcerz L634: Narya went
                    # 2010 -> 2037 -> 2011 in two drags, and the
                    # fellowship-move it paid for logged BETWEEN the
                    # hops): the use is recorded at the FIRST hop so the
                    # ring's die change precedes its beneficiary in the
                    # schedule.  A genuine retraction (active -> X ->
                    # active) is already swallowed by the inverse-drag
                    # pre-pass, never reaching this handler.  An
                    # active->active re-seat is a display shuffle, never
                    # a use (guarded by the raw_to test above).
                    _flight[pieces[0]] = self.line_num
                    self.state.fp_elven_rings = max(
                        0, self.state.fp_elven_rings - 1)
                    self.state.sh_elven_rings += 1
                    self.log(f"Elven Ring used (hop 1): {ring_name}")
                    self._record_action('elven_ring', {'ring': ring_name},
                                        Side.FREE_PEOPLES)
                elif raw_to in RING_USED_LOCS and pieces[0] in _flight \
                        and self.line_num - _flight[pieces[0]] <= 12:
                    del _flight[pieces[0]]  # second hop: already recorded
                elif raw_from in RING_USED_LOCS:
                    self.log(f"Elven Ring collected by Shadow: {ring_name}")
                    self._record_action('elven_ring',
                                        {'ring': ring_name, 'collect': True},
                                        Side.SHADOW)
                return
            # VP-track waypoint collapse (see _preprocess_vp_waypoints):
            # the second hop carries the ORIGINAL source region
            raw_from = getattr(self, '_move_from_rewrites', {}).get(
                self.line_num, raw_from)
            from_loc = decode_location(raw_from)
            to_loc = decode_location(raw_to)
            self._handle_move(pieces, from_loc, to_loc)
            return

        # Fellowship track: <~Controls.790~>marker <~Controls.791~>from <~Controls.792~>to
        fsp_match = re.search(
            r'<~Controls\.790~>(\d+)<~Controls\.791~><~([\w.]+)~><~Controls\.792~><~([\w.]+)~>',
            line)
        if fsp_match:
            marker_id = fsp_match.group(1)
            from_loc = decode_location(fsp_match.group(2))
            to_loc = decode_location(fsp_match.group(3))
            self._handle_fellowship_track(from_loc, to_loc, marker_id)
            return

        # Hunt box die: <~Controls.795~>die <~Controls.796~>from <~Controls.797~>to
        hunt_match = re.search(
            r'<~Controls\.795~>(\d+)<~Controls\.796~><~([\w.]+)~><~Controls\.797~><~([\w.]+)~>',
            line)
        if hunt_match:
            die_id = hunt_match.group(1)
            to_loc = decode_location(hunt_match.group(3))
            if 'Hunt' in to_loc:
                self._handle_die_to_hunt(die_id)
            return

        # Eye allocation before rolling: <~Controls.372~>die_id
        # This tracks eyes allocated to hunt pool BEFORE dice are rolled
        eye_alloc_match = re.search(r'<~Controls\.372~>(\d+)', line)
        if eye_alloc_match:
            die_id = int(eye_alloc_match.group(1))
            # 131: a button allocation the player RETRACTED (die returned
            # from the Hunt Box before the roll) is noise - skip it whole
            if (self.line_num, str(die_id)) in getattr(
                    self, '_eye_btn_noise', set()):
                self.log(f"Retracted eye allocation skipped "
                         f"(button, die {die_id})")
                return
            mapping = self.die_id_map.get(die_id)
            if mapping is None:
                # A bonus die allocated as an Eye BEFORE its first roll
                # has no mapping yet — register it (Controls.372 is a
                # Shadow-only button), mirroring _handle_die_to_hunt.
                self._register_bonus_die(die_id, Side.SHADOW)
                mapping = self.die_id_map.get(die_id)
            if mapping and mapping[0] == Side.SHADOW:
                die_idx = mapping[1]
                # Track this die as pre-allocated (not rolled)
                self.preallocated_eyes.add(die_id)
                if die_idx < len(self.state.dice[Side.SHADOW]):
                    die = self.state.dice[Side.SHADOW][die_idx]
                    die.current = ActionDieFace.EYE
                    die.used = True
                    if die not in self.state.hunt_box:
                        self.state.hunt_box.append(die)
                    self.state.eyes_allocated = getattr(self.state, 'eyes_allocated', 0) + 1
                    self.log(f"Shadow pre-allocates Eye (die {die_idx}, id={die_id}) to Hunt Box (total: {len(self.state.hunt_box)})")
                    self._record_action('eye_allocate', {
                        'die_idx': die_idx,
                        'hunt_box_size': len(self.state.hunt_box)
                    }, Side.SHADOW)
            return

        for _dm in re.finditer(
                r'<~Controls\.296~>(\d+)\s*<~Controls\.297~>'
                r'<~Controls\.(45|46)~>', line):
            _pid, _frm = _dm.group(1), _dm.group(2)
            _lo, _hi = (106, 157) if _frm == '45' else (158, 209)
            if _lo <= int(_pid) <= _hi \
                    and ('reveal ' + _pid + ' ') not in line:
                self._handle_card_zone_discard(_pid, _frm)

        # Card draw: <~Controls.509~>card <~Controls.510~>deck <~Controls.511~>hand
        # GRAMMAR VARIANTS (session 33, live AI-vs-AI build): the menu
        # draw handlers emit PER-DECK token sets — A.a() fdrawc uses
        # Controls.491/492/493, A.c() sdrawc uses Controls.0/549/550,
        # A.d() sdraws uses Controls.557/558/559 (A.b() fdraws uses the
        # familiar 509/510/511).  Corpus logs show only 509 (different
        # producer path/build), so accepting the variants adds coverage
        # without touching corpus behavior.  The deck ZONE token is the
        # authoritative deck id in every variant.
        draw_match = re.search(
            r'<~Controls\.(?:509|491|0|549|557)~>(\d+)'
            r'<~Controls\.\d+~><~Game\.(200[4-7])~>'
            r'<~Controls\.\d+~><~Controls\.(45|46)~>',
            line)
        if draw_match:
            piece_id = draw_match.group(1)
            deck_code = draw_match.group(2)
            hand_code = draw_match.group(3)
            self._handle_card_draw_with_decoder(piece_id, deck_code, hand_code)
            return

        # Fallback card draw (less specific pattern)
        draw_match2 = re.search(
            r'<~Controls\.509~>(\d+)<~Controls\.510~><~([\w.]+)~><~Controls\.511~><~([\w.]+)~>',
            line)
        if draw_match2:
            deck = decode_location(draw_match2.group(2))
            self._handle_card_draw(deck)
            return

        # Combat card selection: Controls.840
        combat_sel_match = re.search(r'<~Controls\.840~>(\d+)', line)
        if combat_sel_match:
            piece_id = combat_sel_match.group(1)
            try:
                pid = int(piece_id)
                side = Side.FREE_PEOPLES if 106 <= pid <= 157 else Side.SHADOW
            except ValueError:
                side = Side.SHADOW
            self._pending_combat_card = (piece_id, side)
            self._battle_window_line = getattr(self, 'line_num', 0)
            # An OPPONENT's combat-card selection proves a battle is
            # being declared: any open card window of the OTHER side is
            # exhausted (2023-12-06 T3: FP committed a combat card
            # against the Dimrill Dale attack while SH's 'Ringwraiths
            # are Abroad' window was still open — the attack mis-tagged
            # card_effect and its rolls were skipped as effect noise).
            # The window-owner's OWN selection keeps the window (card-
            # driven attacks: Corsairs, The Fighting Uruk-hai).
            if self._card_window_side is not None \
                    and self._card_window_side != side:
                self._card_window_side = None
            return

        # Combat card reveal (FP): Controls.884/885 or Controls.861-863
        # Combat card reveal (SA): Controls.867 or Controls.868 present
        # ALSO: a manual Scribe-drag reveal (card dragged from hand to the
        # play area, no combat-reveal button) while a FULL chat attack
        # declaration ('X from Y') is live in the current die window IS
        # that battle's combat-card commitment (2025-06-19 StellaRossa T5
        # L411: Faramir's Rangers dragged right after 'fords from orthanc
        # NC' — misparsed as play_event, the battle never synthesized and
        # its combat rolls expired as card-effect noise).
        _hint_cc = False
        _hr = getattr(self, '_battle_hint_region', None)
        if _hr is not None:
            if getattr(self, '_battle_hint_from', None) is not None:
                # full 'X from Y' declaration (StellaRossa T5 L408)
                _hint_cc = True
            elif _hr in getattr(self.state, 'besieged_strongholds', set()):
                # bare region callout of a BESIEGED stronghold = the
                # assault declaration (StellaRossa T4 L346 'MT'; the
                # Return to Valinor drag at L347 is its combat card —
                # played die-first, no combat-reveal button)
                _hint_cc = True
            elif self._last_die_side is not None and \
                    self._region_holds_enemy_of(_hr, self._last_die_side):
                # region callout validated against the die-consumer's
                # enemy (same validation _maybe_synth_attack applies)
                _hint_cc = True
        is_combat_reveal = ('Controls.884' in line or 'Controls.885' in line or
                            'Controls.867' in line or 'Controls.868' in line or
                            'Controls.861' in line or _hint_cc)

        # Card reveal: reveal piece_id hash card_number
        reveal_match = re.search(r'reveal (\d+) ([A-F0-9]+) (\d+)', line)
        if reveal_match:
            piece_id = reveal_match.group(1)
            card_num = int(reveal_match.group(3))
            # Determine side from context
            is_fp = 'Controls.45' in line or 'Controls.884' in line or 'Controls.885' in line
            # Controls.861-863 combat-card commitments carry a HAND SLOT
            # in the number field, not the card number — decode the true
            # number from the deck commitments (2023-12-06 line 250:
            # 'reveal 131 ... 1' = King Brand's Men, not slot 1; the
            # bogus decode also popped the WRONG card from the tracked
            # hand, cascading hand drift)
            if 'Controls.861' in line:
                try:
                    pid = int(piece_id)
                    dt = None
                    if 106 <= pid <= 157:
                        dt = ('FP Character'
                              if _FP_PID_TYPES[pid - 106] == 'C'
                              else 'FP Strategy')
                    elif 158 <= pid <= 209:
                        dt = ('SA Character'
                              if _SA_PID_TYPES[pid - 158] == 'C'
                              else 'SA Strategy')
                    if dt is not None:
                        dec = self.card_decoder.decode_piece(pid, dt)
                        if dec is not None:
                            card_num = dec
                except (ValueError, IndexError):
                    pass
            # Combat-card commitments are NOT event plays: the effect
            # window _handle_card_reveal opens must not survive the
            # retag (2023-12-06 T9: the FP 'Grimbeorn' commitment's
            # window mis-tagged the FP defenders' retreat card_effect)
            _cw_before = self._card_window_side
            # SOURCE ZONE of the reveal: a card revealed out of a
            # BOARD/TABLE zone (Game.*) was never in a hand, so it must
            # not remove one.  See _handle_card_reveal's from_hand gate.
            # TWO grammars carry it: Controls.867 (the play/reveal form,
            # 159,781 in the corpus) and Controls.862 (the combat-card
            # COMMITMENT form, 31,203) — the 862 variant is how the
            # auto36 break slipped past the first gate.
            _rev_src9 = None
            _m_src9 = re.search(r'Controls\.86[27]~><~([^~]+)~>', line)
            if _m_src9:
                _rev_src9 = _m_src9.group(1)
            self._handle_card_reveal(card_num, is_fp, piece_id,
                                     src_zone=_rev_src9)
            # AI-BRIDGE EVENT PLAY (t121): the live bridge plays event
            # cards through the client's choose/playchosen handler,
            # whose reveal line carries the combat-commit markers
            # (Controls.866/867/868 or 3/884/885) even for EVENT plays.
            # The bridge always chats '<nick> plays <name>' right
            # before.  An armed, NAME-MATCHING announce within 6 lines
            # marks this reveal as the event play itself: keep the
            # play_event record and its effect window; no combat-card
            # retag, no attack synthesis.  Corpus-inert (campaign
            # canary): human combat commits are never announced with a
            # 'plays <exact card name>' chat.
            _ann9 = getattr(self, '_announced_play', None)
            if (is_combat_reveal
                    and getattr(self, '_ai_bridge_log', False)
                    and _ann9 is not None
                    and self.action_log
                    and self.action_log[-1][0] == 'play_event'
                    and _normalize_name(self.action_log[-1][1].get(
                        'card_name', '')) in _ann9[0]
                    and 0 <= getattr(self, 'line_num', 0)
                    - _ann9[1] <= 6):
                self._announced_play = None
                return
            # If this is a combat card reveal, re-tag the last action
            if is_combat_reveal and self.action_log and self.action_log[-1][0] == 'play_event':
                act_type, params, act_side = self.action_log.pop()
                # Combat cards are FREE: a die attached to the play
                # record belongs to the battle's own attack — put it
                # back as pending so the synthesized attack takes it
                # (2023-12-07_001647 T8: the HD assault's Army die had
                # landed on the Cruel Weather commitment)
                _du = params.pop('die_used', None)
                if _du is not None \
                        and self._pending_die.get(act_side) is None:
                    self._pending_die[act_side] = _du
                # Combat cards imply a battle: synthesize the attack
                # decision first. Pass None — a STALE _combat_region from
                # the previous battle must not name this one's target;
                # the region resolves via chat hint or first casualties.
                self._maybe_synth_attack(None)
                # FIX D1: an unresolved combat-card region tags '?' -- NEVER
                # the PREVIOUS battle's region.  A stale _combat_region
                # fallback stamped cards onto the wrong battle (game 11: a
                # 'Pelagir' typo left _combat_region at 'old forest road',
                # so the cards + their casualties landed there).  '?' lets
                # the closed loop's episode reconciler bind by evidence.
                params['region'] = (self._battle_synth_region
                                    if self._battle_synth_region
                                    and self._battle_synth_region != '?'
                                    else '?')
                self._record_action('combat_card', params, act_side)
                self._last_combat_card_name[act_side] = \
                    params.get('card_name', '')
                self._cc_pre_roll[act_side] = True
                # Restore the pre-reveal card window (the synthesis
                # above already consumed the transient one)
                self._card_window_side = _cw_before
            return

        # Random-companion hunt-casualty draw:
        # <~Controls.637~>piece<~Controls.638~><~from~><~Controls.639~><~to~>
        # pulls a companion out of the Fellowship Box after a damage tile
        # (2023-12-07 lines 430/615: Legolas then Meriadoc, chat 'Random,
        # u draw'; 2023-12-06 line 775: Strider the guide).  Resolves the
        # open hunt-damage decision; a hobbit's subsequent walk onto the
        # map is his PLACEMENT (user rule) via the companion_move path.
        rand_draw = re.search(
            r'<~Controls\.637~>(\d+)<~Controls\.638~><~([\w.]+)~>'
            r'<~Controls\.639~><~([\w.]+)~>', line)
        if rand_draw:
            piece_id = rand_draw.group(1)
            from_code = rand_draw.group(2)
            to_loc = decode_location(rand_draw.group(3))
            comp_name = COMPANIONS.get(piece_id)
            if comp_name:
                # t138 Q9 gate (USER ruling 1.169): a 637 companion draw
                # with NO open hunt, no recent corruption take and no
                # card window is a DISPLAY gesture (showing the guide
                # during bidding — Mol/Rizino1 L44 killed Gandalf at
                # cp50 of the SETUP phase), never a casualty.
                _ctx9 = self._hunt_ctx
                _ctx_open = _ctx9 is not None and not _ctx9.get('resolved')
                _recent_corr = any(
                    _at9 == 'hunt_take_corruption'
                    for _at9, _p9, _s9 in reversed(self.action_log[-6:]))
                if not (_ctx_open or _recent_corr
                        or getattr(self, '_in_hunt_resolution', False)
                        or self._card_window_side is not None):
                    self.piece_region[piece_id] = to_loc
                    self.log(f"637 draw outside any hunt/card window: "
                             f"display no-op ({comp_name}) [Q9]")
                    return
                for i, comp in enumerate(self.state.fellowship.companions):
                    if comp.name == comp_name:
                        # AUTHORITATIVE guide-vs-random discriminator
                        # (USER 2026-07-09): the module pulls a RANDOM
                        # companion from the Fellowship Box (Game.1998)
                        # but the GUIDE piece from the Guide Box
                        # (Game.1999). The engine's tracked guide can
                        # drift; the zone cannot.
                        if from_code == 'Game.1999':
                            was_guide = True
                        elif from_code == 'Game.1998':
                            _eng_guide = (
                                self.state.fellowship.guide is not None
                                and self.state.fellowship.guide.name
                                == comp_name)
                            if _eng_guide:
                                self.log(
                                    f"637 zone says RANDOM but engine "
                                    f"guide is {comp_name} — zone wins")
                            was_guide = False
                        else:
                            was_guide = (
                                self.state.fellowship.guide is not None
                                and self.state.fellowship.guide.name
                                == comp_name)
                        self.state.fellowship.companions.pop(i)
                        if was_guide:
                            self.state.fellowship.guide = None
                        if comp_name == 'Gandalf the Grey':
                            self._gandalf_dead = True
                        self.log(f"Random hunt-casualty draw: {comp_name}")
                        ctx = self._hunt_ctx
                        if ctx is not None and not ctx.get('resolved'):
                            ctx['resolved'] = True
                            self._hunt_ctx = None
                            self._record_action(
                                'hunt_sacrifice_guide' if was_guide
                                else 'hunt_sacrifice_random',
                                {'companion': comp_name},
                                Side.FREE_PEOPLES)
                        else:
                            # ctx already resolved by a corruption walk
                            # the humans then RETRACTED (2025-05-16
                            # Adobim L675-679: +2 corruption, walked
                            # back, 'random' chat, Strider drawn) —
                            # emit the record anyway; build_schedules'
                            # retracted-take-corruption prune pairs it
                            # with the stale corruption record.
                            _emitted = False
                            for at0, p0, sd0 in reversed(
                                    self.action_log[-6:]):
                                if at0 == 'hunt_take_corruption':
                                    self._record_action(
                                        'hunt_sacrifice_guide' if was_guide
                                        else 'hunt_sacrifice_random',
                                        {'companion': comp_name},
                                        Side.FREE_PEOPLES)
                                    _emitted = True
                                    break
                            # CARD-driven companion draw (no hunt at
                            # all): Lure of the Ring's random victim.
                            # The draw names the victim (logged RNG -
                            # legitimate forcing) but does NOT prove the
                            # eliminate choice: emitting companion_death
                            # here FABRICATED an elimination when the
                            # human chose corruption (GAME21 rendered
                            # L1021-1024: Legolas drawn, +2 corruption,
                            # RETURNED to the Guide box - engine killed
                            # him for the rest of the game).  Emit a
                            # non-decision lure_draw carrying the victim;
                            # companion_death is emitted only on the
                            # eliminate EVIDENCE (kill/dead-zone of the
                            # drawn pid - 2023-11-03 ketchuptime T2
                            # L159-162 Gimli, chat 'kil'), and a
                            # corruption walk clears the pending.
                            if not _emitted \
                                    and self._card_window_side is not None:
                                self._record_action(
                                    'lure_draw',
                                    {'companion': comp_name,
                                     'card_effect': True, 'auto': True},
                                    Side.FREE_PEOPLES)
                                self._lure_pending = {
                                    'companion': comp_name,
                                    'pid': piece_id,
                                    'line': self.line_num}
                        break
            self.piece_region[piece_id] = to_loc
            return

        # Kill/cleanup event: <~Controls.644~>pieces <~Controls.645~><~Game.XXXX~>
        # Sends pieces to "Used" area - handles dice cleanup, card discard, companion death, etc.
        kill_match = re.search(
            r'<~Controls\.644~>([\d\s]+)<~Controls\.645~><~([\w.]+)~>',
            line)
        if kill_match:
            pieces = kill_match.group(1).strip().split()
            to_loc = decode_location(kill_match.group(2))
            self._handle_kill_event(pieces, to_loc)
            return

        # Card-area sweep: <~Controls.662~>pids <~Controls.663~><~zone~>
        # (client verb 'kills', messages.properties 477).  t138 wave-3
        # (triage N3): 85% of logs use it to clear card areas after the
        # pieces were already dealt with by 296/644/637 — but when the
        # sweep is the ONLY removal record it must route pids by
        # identity (Jed/Mirdain L266: tile 218 + Gandalf + Gimli swept
        # from FP Card Play; the tile stayed 'fp card play' for 15
        # checkpoints).  Idempotent per pid: only pieces whose TRACKED
        # zone is still an in-flight card/current area are routed
        # through the same identity router as Controls.644; cards,
        # dice, and already-resolved pieces are untouched.
        sweep_match = re.search(
            r'<~Controls\.662~>([\d\s]+)<~Controls\.663~><~([\w.]+)~>',
            line)
        if sweep_match:
            _sw_zones = ('sa card play', 'fp card play', 'used',
                         'current action', 'current sa card',
                         'current fp card')
            _sw_hit = []
            for _sp in sweep_match.group(1).strip().split():
                _tz = sanitize(str(self.piece_region.get(_sp) or ''))
                if _tz in _sw_zones and (
                        _sp in HUNT_TILE_PIECES or _sp in ('230', '231')
                        or _sp in COMPANIONS or _sp in MINION_PIECES
                        or get_unit_type_from_piece_id(
                            _sp, self.piece_types)[0] is not None):
                    _sw_hit.append(_sp)
            if _sw_hit:
                self.log(f"Card-area sweep routes in-flight pieces: "
                         f"{_sw_hit}")
                self._handle_kill_event(
                    _sw_hit, decode_location(sweep_match.group(2)))
            return

        # Battle casualties: <~Controls.296~>pieces <~Controls.297~><~location~>
        # Selects pieces as combat casualties at a map region.
        casualty_match = re.search(
            r'<~Controls\.296~>([\d\s]+)<~Controls\.297~><~([\w.]+)~>', line)
        if casualty_match:
            pieces = casualty_match.group(1).strip().split()
            loc = decode_location(casualty_match.group(2))
            # t133 fix F2: hunt-tile pids in kill/cleanup grammars are
            # DISCARDED to Used Hunt Tiles (the client's tile resting
            # place; 193 games left tiles in flight at card-play areas
            # or marked them dead alongside hunt casualties).  A
            # special tile killed out of Removed Hunt Tiles is its
            # ACTIVATION — it parks in Used ("in play, set aside",
            # USER RULING A1 2026-07-17; rulebook p.42) until the
            # Mordor-entry re-pool.
            _tile_kill = [p for p in pieces
                          if p in HUNT_TILE_PIECES or p in ('230', '231')]
            if _tile_kill:
                for p in _tile_kill:
                    self._tile_to_zone(p, 'Used Hunt Tiles')
                pieces = [p for p in pieces if p not in _tile_kill]
                if not pieces:
                    return
            base = stronghold_base_region(loc)
            if base is not None:
                # Garrison casualties reported in the stronghold BOX
                # ("kills ... in Minas Tirith Stronghold") belong to the
                # base region's army — previously dropped entirely,
                # leaving immortal garrisons / wrong control / wrong VP.
                # Removals in a card window with no active battle are card
                # effects (e.g. Cirdan's Ships pulling 2 Gondor Regulars
                # out of besieged Dol Amroth) — no attack to synthesize.
                _card_fx = (self._card_window_side is not None
                            and self._battle_synth_region is None)
                self._handle_battle_casualties(pieces, base,
                                               synth_attack=not _card_fx)
            elif is_map_region(loc):
                self._handle_battle_casualties(pieces, loc)
            elif any(p in COMPANIONS for p in pieces):
                # Companions killed in the fellowship boxes (Guide Box /
                # Fellowship Box; 2023-03-15 line 77: Gandalf 240 killed
                # in Game.1999 = the T1 hunt guide sacrifice) —
                # fellowship casualties, never a battle
                self._handle_battle_casualties(pieces, loc,
                                               synth_attack=False)
            else:
                # t125: dice 'killed' at a non-map area — e.g. an
                # allocated eye confirmed/cleared in the Hunt Box
                # (Columbo L56: 296 kill of die 101 at Game.1997).  The
                # client moves the die piece out of the zone; mirror by
                # retiring it from the hunt box.
                _die_ps9 = [p for p in pieces if self._is_die_piece(p)]
                for p in _die_ps9:
                    self._retire_die_from_hunt(p)
                _rest9 = [p for p in pieces if p not in _die_ps9]
                if _rest9 and 'Hunt' not in str(loc) \
                        and 'Dice' not in str(loc) \
                        and 'Hand' not in str(loc):
                    # t133 fix F6 (+ off-board parking kills): kills
                    # reported at battle/parking boxes (Game.2029 =
                    # Board1: SlowCobra/Raster L311 Gondor leader 085;
                    # VP-box parking: SlowCobra/Wey L445 kills the
                    # staged Lorien garrison at 'VP 0') resolve each
                    # piece from its TRACKED zone exactly like
                    # Controls.644 — the line carries no source region.
                    self._handle_kill_event(_rest9, 'Dead')
            return

        # Piece swap/removal: <~Controls.304~>pieces <~Controls.305~><~location~>
        # Removes pieces from board during battle (downgrade, casualty replacement).
        swap_remove_match = re.search(
            r'<~Controls\.304~>([\d\s]+)<~Controls\.305~><~([\w.]+)~>', line)
        if swap_remove_match:
            pieces = swap_remove_match.group(1).strip().split()
            loc = decode_location(swap_remove_match.group(2))
            self._handle_piece_swap_remove(pieces, loc)
            return

        # Piece swap/placement: <~Controls.313~>pieces <~Controls.30X~><~Controls.314~><~location~>
        # AUTHORITATIVE decode (WOTR-Gandalf messages.properties):
        # Controls.313=" moves ", 307/309="from reinforcements",
        # 308="from casualties", 314=" to ".  The old comment
        # (307=Elite/308=Regular) was wrong — the digit is the SOURCE
        # POOL, not the unit type.
        # t125: tolerate a literal mixed-source annotation between the
        # source-pool token and Controls.314 — Columbo L433:
        # 'Controls.313 295 069 Controls.309 (1)/casualties(1)
        # Controls.314 Game.1993' places 295 from reinforcements AND
        # 069 from casualties in ONE line; the strict grammar dropped
        # the whole placement (069 stayed dead, phantom HD garrison
        # gap).  The piece-level dead-pool bookkeeping refunds the
        # from-casualties pids regardless of the leading source token.
        swap_place_match = re.search(
            r'<~Controls\.313~>([\d\s]+)<~Controls\.(307|308|309)~>'
            r'[^<]*<~Controls\.314~><~([\w.]+)~>',
            line)
        if swap_place_match:
            pieces = swap_place_match.group(1).strip().split()
            swap_type = swap_place_match.group(2)  # source pool key
            loc = decode_location(swap_place_match.group(3))
            self._handle_piece_swap_place(pieces, swap_type, loc)
            return

        # Elven ring usage: Controls.366 moves ring from active to used slot.
        # Used slot -> anywhere = the Shadow player collecting the used FP
        # ring ('collect' record; an adjacent die-face change proves an
        # actual Shadow USE — the closed-loop schedule build decides;
        # counts are NOT double-moved here, the FP-use handler already
        # passed the ring to the Shadow).
        ring_match = re.search(
            r'<~Controls\.366~>(\d+)<~Controls\.367~><~([\w.]+)~><~Controls\.368~><~([\w.]+)~>', line)
        if ring_match:
            ring_pid = ring_match.group(1)
            from_loc_code = ring_match.group(2)
            to_loc_code = ring_match.group(3)
            ring_name = RING_PIECES.get(ring_pid)
            if ring_name and from_loc_code in RING_ACTIVE_LOCS and to_loc_code in RING_USED_LOCS:
                self.state.fp_elven_rings = max(0, self.state.fp_elven_rings - 1)
                self.state.sh_elven_rings += 1
                self.log(f"Elven Ring used: {ring_name}")
                self._record_action('elven_ring', {'ring': ring_name}, Side.FREE_PEOPLES)
            elif ring_name and from_loc_code in RING_USED_LOCS:
                self.log(f"Elven Ring collected by Shadow: {ring_name}")
                self._record_action('elven_ring',
                                    {'ring': ring_name, 'collect': True},
                                    Side.SHADOW)
            return

        # Flip token: <~Controls.323~>piece
        flip_match = re.search(r'<~Controls\.323~>(\d+)', line)
        if flip_match:
            piece_id = flip_match.group(1)
            self._handle_flip(piece_id)
            return

    def _handle_turn_change(self, new_turn: int):
        """Handle turn change - reset dice, clear hunt box."""
        self.log(f"\n{'='*50}")
        self.log(f"TURN {new_turn}")
        self.log(f"{'='*50}")

        # Reset all dice
        for d in self.state.hunt_box:
            d.used = False
            d.current = None
        self.state.hunt_box.clear()
        self.state.eyes_allocated = 0

        for side in [Side.FREE_PEOPLES, Side.SHADOW]:
            for d in self.state.dice[side]:
                d.used = False
                d.current = None

        self.state.turn = new_turn
        self.state.dice_rolled = False

        # Reset preallocated eyes for new turn
        self.preallocated_eyes.clear()

        # Reset combat state for new turn
        self._combat_casualties_this_round = False
        self._last_swap_remove_elite = False
        self._last_swap_remove_side = None
        self._last_swap_remove_nation = None
        self._pending_combat_card = None
        self._in_hunt_resolution = False
        self._hunt_reveal_pending = False

        # Update card decoder turn
        self.card_decoder.turn = new_turn

    def _register_bonus_die(self, die_id: int, side: Side):
        """Register a bonus die (from a special character) in the dice pool.

        Bonus dice have piece IDs outside the base ranges (99-105 for Shadow,
        210-213 for FP). When first encountered in a roll, we add a new die
        to the pool and record the mapping.
        """
        if die_id in self.die_id_map:
            return  # Already registered
        faces = gs.SH_FACES if side == Side.SHADOW else gs.FP_FACES
        new_die = gs.ActionDie(faces, side)
        pool = self.state.dice[side]
        idx = len(pool)
        pool.append(new_die)
        self.die_id_map[die_id] = (side, idx)
        self.log(f"Bonus die registered: id={die_id} -> {side.name}[{idx}]")

    def _handle_dice_roll(self, dice_ids: List[str], values: List[str]):
        """Handle dice roll - SET dice faces from log values (no randomness)."""
        if not dice_ids or not values:
            return

        # Determine side from the first KNOWN die id — the first LISTED
        # id may be a bonus die not registered yet (2023-08-13
        # HombreLobo T2 L128: the Witch-king's die 362 leads the roll,
        # the old first-id-only test defaulted to FREE_PEOPLES and the
        # whole Shadow roll parsed empty, leaving T2 faces unforced).
        side = None
        for _ds in dice_ids:
            _di = int(_ds)
            if _di in self.die_id_map:
                side = self.die_id_map[_di][0]
                break
        if side is None:
            first_id = int(dice_ids[0])
            if 99 <= first_id <= 108:
                side = Side.SHADOW
            elif 210 <= first_id <= 213:
                side = Side.FREE_PEOPLES
            else:
                side = self._player_to_side.get(
                    self._current_line_player, Side.FREE_PEOPLES)

        # Dwarven-ring reroll: a SINGLE-die roll right after a ring use is
        # the ring's outcome (2026-01-05 kapcerz line 50: "rolls C" for
        # die 212).  Fold the face into the pending dwarven_ring record —
        # no separate dice_roll record, so schedule/chunk alignment stays
        # clean — and apply the new face to the die in place.
        _dr = getattr(self, '_dwarven_use_params', None)
        if _dr is not None:
            self._dwarven_use_params = None
            if len(dice_ids) == 1:
                try:
                    _rid = int(dice_ids[0])
                    _val = int(values[0])
                    _vm = SHADOW_DICE_VALUES if side == Side.SHADOW \
                        else FP_DICE_VALUES
                    _face = _vm.get(_val)
                    _mp = self.die_id_map.get(_rid)
                    if _face is not None and _mp is not None:
                        _sd, _ix = _mp
                        if _ix < len(self.state.dice[_sd]):
                            self.state.dice[_sd][_ix].current = \
                                ActionDieFace(_face) \
                                if not isinstance(_face, ActionDieFace) \
                                else _face
                        _dr['new_face'] = _face.value \
                            if isinstance(_face, ActionDieFace) else _face
                        self.log(f"Dwarven Ring reroll: die {_rid} -> "
                                 f"{_dr['new_face']}")
                        return
                except (ValueError, KeyError):
                    pass

        # 1.149 (USER rule): a SINGLE known action die re-rolled after
        # it already shows a face is a dwarven-ring reroll even without
        # the ring piece drag ('dwarven ring placeholder' clients -
        # 2026-01-23 kapcerz T1: H rerolled to P as three bare single-
        # die roll lines).  Synthesize the same record the piece-428
        # path emits; consecutive rerolls of one die collapse to the
        # NET face change.
        if len(dice_ids) == 1:
            _mp2 = None
            try:
                _rid2 = int(dice_ids[0])
                _mp2 = self.die_id_map.get(_rid2)
            except (ValueError, KeyError):
                _mp2 = None
            if _mp2 is not None:
                _sd2, _ix2 = _mp2
                if _ix2 < len(self.state.dice[_sd2]):
                    _die2 = self.state.dice[_sd2][_ix2]
                    _vm2 = SHADOW_DICE_VALUES if _sd2 == Side.SHADOW \
                        else FP_DICE_VALUES
                    _nf2 = None
                    try:
                        _nf2 = _vm2.get(int(values[0]))
                    except (ValueError, IndexError):
                        _nf2 = None
                    if _die2.current is not None and _nf2 is not None:
                        _oldv = _die2.current.value
                        _newv = _nf2.value \
                            if isinstance(_nf2, ActionDieFace) else _nf2
                        _die2.current = _nf2 \
                            if isinstance(_nf2, ActionDieFace) \
                            else ActionDieFace(_nf2)
                        _last = self.action_log[-1] \
                            if self.action_log else None
                        # t138 (family 2, USER ruling 1.168): consecutive
                        # rerolls of ONE die are ONE ring decision
                        # ("reroll a single die until it is different")
                        # — fold into ANY adjacent same-side
                        # dwarven_ring record whose net face chain
                        # continues (the old test required
                        # placeholder_reroll, so a piece-428 use
                        # followed by a second reroll emitted TWO
                        # records).
                        if _last is not None \
                                and _last[0] == 'dwarven_ring' \
                                and _last[2] == _sd2 \
                                and _last[1].get('new_face') == _oldv:
                            _last[1]['new_face'] = _newv
                        else:
                            # placeholder clients (1.149) never show a
                            # grant: infer one when the counter is 0 so
                            # the use is legal (grant-on-evidence)
                            _attr2 = ('fp_dwarven_rings'
                                      if _sd2 == Side.FREE_PEOPLES
                                      else 'sh_dwarven_rings')
                            if getattr(self.state, _attr2, 0) <= 0:
                                setattr(self.state, _attr2, 1)
                                self.log(f"Dwarven Ring inferred grant "
                                         f"({_attr2}=1, placeholder "
                                         f"client)")
                            self._record_action('dwarven_ring', {
                                'side': 'fp'
                                if _sd2 == Side.FREE_PEOPLES else 'sh',
                                'old_face': _oldv, 'new_face': _newv,
                                'placeholder_reroll': True}, _sd2)
                        self.log(f"Dwarven Ring placeholder reroll: "
                                 f"die {_rid2} {_oldv} -> {_newv}")
                        return

        # Map player name to side (for chat message "pass" detection)
        if self._current_line_player and self._current_line_player not in self._player_to_side:
            self._player_to_side[self._current_line_player] = side

        value_map = SHADOW_DICE_VALUES if side == Side.SHADOW else FP_DICE_VALUES

        # Register any bonus dice we haven't seen before
        for die_id_str in dice_ids:
            die_id = int(die_id_str)
            if die_id not in self.die_id_map:
                self._register_bonus_die(die_id, side)

        faces = []
        rolled_eyes = []
        for die_id_str, val_str in zip(dice_ids, values):
            try:
                die_id = int(die_id_str)
                val = int(val_str)
                face = value_map.get(val)

                mapping = self.die_id_map.get(die_id)
                if not mapping:
                    faces.append(f"?{val_str}")
                    continue
                die_side, die_idx = mapping

                # Skip Shadow dice that were pre-allocated as eyes (already in hunt box)
                if die_side == Side.SHADOW and die_id in self.preallocated_eyes:
                    faces.append(f"E*")  # Mark as pre-allocated eye
                    continue

                if face and die_idx < len(self.state.dice[die_side]):
                    # DIRECTLY SET the die face - no randomness!
                    self.state.dice[die_side][die_idx].current = face
                    self.state.dice[die_side][die_idx].used = False
                    faces.append(face.value)

                    # If Shadow rolled Eye, mark used and add to hunt box
                    if die_side == Side.SHADOW and face == ActionDieFace.EYE:
                        self.state.dice[die_side][die_idx].used = True
                        if self.state.dice[die_side][die_idx] not in self.state.hunt_box:
                            self.state.hunt_box.append(self.state.dice[die_side][die_idx])
                            rolled_eyes.append(die_idx)
            except (ValueError, IndexError) as e:
                faces.append(f"?{val_str}")

        self.state.dice_rolled = True
        side_name = "Shadow" if side == Side.SHADOW else "FP"
        self.log(f"{side_name} dice ({len(faces)}): {', '.join(faces)}")

        # Log rolled eyes separately
        if rolled_eyes:
            self.log(f"Shadow rolled Eyes (dice {rolled_eyes}) -> Hunt Box (total: {len(self.state.hunt_box)})")

        self._record_action('dice_roll', {'faces': list(faces), 'count': len(faces)}, side)

    def _handle_move(self, pieces: List[str], from_loc: str, to_loc: str):
        """Handle piece movement between locations."""
        if not from_loc or not to_loc:
            return

        # Aborted drag pair (see _preprocess_inverse_drags)
        if self.line_num in getattr(self, '_skip_move_lines', ()):
            self.log(f"Aborted drag skipped: {from_loc} -> {to_loc} "
                     f"({len(pieces)} pieces)")
            return
        # Walk-back netting / split-drag merging (see
        # _preprocess_inverse_drags): replay the ordered add/drop ops so
        # interleaved handfuls net to the humans' final intent.
        _ops = getattr(self, '_move_piece_ops', {}).get(self.line_num)
        if _ops:
            pieces = list(pieces)
            for _op, _pc in _ops:
                if _op == 'add' and _pc not in pieces:
                    pieces.append(_pc)
                elif _op == 'drop' and _pc in pieces:
                    pieces.remove(_pc)
            self.log(f"Move ops netted ({len(_ops)} ops): "
                     f"{from_loc} -> {to_loc} now {len(pieces)} pieces")
            if not pieces:
                return

        # EDGE-2 (2026-08-17): a CARD piece dragged TO a deck zone =
        # the card RETURNS to the deck (the refusal corrective for an
        # un-owed draw is exactly this, in all-Game grammar; 'put it
        # back on the deck' effects share the shape).  Remove it from
        # the hand model the same way the face-down hand exit does —
        # without this the corrective was grammar-performable but
        # EFFECT-nil and the un-owed card haunted the parser hand.
        if len(pieces) == 1 and to_loc in (
                'FP Char Deck', 'FP Strat Deck',
                'SA Char Deck', 'SA Strat Deck'):
            try:
                _pidr9 = int(pieces[0])
            except ValueError:
                _pidr9 = -1
            if 106 <= _pidr9 <= 209:
                _hc9 = '45' if _pidr9 <= 157 else '46'
                self._handle_card_zone_discard(
                    str(_pidr9), _hc9, deck_return=to_loc)
                return

        # A card piece returning from the play area to a HAND retracts
        # the play (2025-04-06 demanray T4: Horde from the East played
        # illegally, taken back via manual reverse-drag after undos) —
        # drop the pending play_event record so the teacher never
        # executes a retracted play.
        if len(pieces) == 1 and to_loc in ('FP Hand', 'SA Hand')                 and (from_loc in ('FP Card Play', 'SA Card Play')
                     or from_loc.startswith('Board')):
            try:
                _pid = int(pieces[0])
            except ValueError:
                _pid = -1
            if 106 <= _pid <= 209:
                for k in range(len(self.action_log) - 1,
                               max(-1, len(self.action_log) - 16), -1):
                    at, ap, aside = self.action_log[k]
                    if at == 'play_event':
                        self.log(f"Play RETRACTED: "
                                 f"{ap.get('card_name')}")
                        self.action_log.pop(k)
                        break
                return

        # Hunt-tile drags (t125 fix 2): tiles shuttled between the tile
        # zones (Mordor-entry re-pooling by hand, specials introduced
        # from Removed, discards) or parked on display areas.  Zone- or
        # tracking-gated so numeric pid collisions can't misroute.
        _tile_ps = [
            p for p in pieces
            if (p in HUNT_TILE_PIECES or p in ('230', '231'))
            and (from_loc in self._TILE_ZONE_LISTS
                 or to_loc in self._TILE_ZONE_LISTS
                 or p in self._tile_in_flight
                 or str(self.piece_region.get(p))
                 in self._TILE_ZONE_LISTS)]
        if _tile_ps:
            for p in _tile_ps:
                self._tile_to_zone(p, to_loc)
            pieces = [p for p in pieces if p not in _tile_ps]
            if not pieces:
                return

        # Settlement-control markers: capture/restore bookkeeping, never
        # units.  Marker onto a map region = that side claims it (or
        # restores its own); marker leaving = release.  VP markers are
        # pure track bookkeeping.
        if all(p in CONTROL_MARKER_PIECES or p in VP_MARKER_PIECES
               for p in pieces):
            for p in pieces:
                if p in VP_MARKER_PIECES:
                    # VP-track sync: a PURE marker drag returns from THIS
                    # branch, so it must route through _handle_marker_move
                    # here — the later marker branch (before the muster
                    # path) only sees markers MIXED with unit pieces.
                    # Without this, state.vp stayed frozen at 0 for every
                    # corpus game (GenericPie 423 Game.389->Game.392 =
                    # SH 8->10 was swallowed).
                    self._handle_marker_move(p, from_loc, to_loc)
                    continue
                side = (Side.FREE_PEOPLES if p in FP_CONTROL_MARKERS
                        else Side.SHADOW)
                other = (Side.SHADOW if side == Side.FREE_PEOPLES
                         else Side.FREE_PEOPLES)
                dest = stronghold_base_region(to_loc) or to_loc
                src = stronghold_base_region(from_loc) or from_loc
                if is_map_region(dest):
                    dsan = sanitize(dest)
                    try:
                        nat = None
                        import ai as _ai_mod  # circular-safe local
                        nat = _ai_mod._region_nation(dsan)
                    except Exception:
                        nat = None
                    native_shadow = nat is not None and nat.name in (
                        'SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS')
                    if (side == Side.SHADOW) == native_shadow:
                        # own marker on own settlement = restoration
                        self.state.enemy_controlled.setdefault(
                            other, set()).discard(dsan)
                        self.log(f"Control restored: {dest}")
                    else:
                        self.state.enemy_controlled.setdefault(
                            side, set()).add(dsan)
                        self.log(f"Captured: {dest} by {side.name}")
                    self._control_marker_at[p] = dsan
                elif is_map_region(src) or p in self._control_marker_at:
                    ssan = self._control_marker_at.pop(
                        p, sanitize(src) if src else None)
                    if ssan:
                        self.state.enemy_controlled.setdefault(
                            side, set()).discard(ssan)
                        self.log(f"Control released: {ssan}")
            return

        # Army-box parking (see _preprocess_army_boxes): substitute the
        # resolved true region for the box location.  Track-marker pieces
        # (fellowship/corruption) are never rewritten.
        if not all(p in self._TRACK_MARKER_PIECES for p in pieces):
            rw = self._box_rewrites.get(self.line_num)
            if rw is not None and ('Mordor ' in from_loc
                                   or 'Mordor ' in to_loc):
                kind, x_name = rw
                if kind == 'noop':
                    self.log(f"Army-box parking (state noise): "
                             f"{from_loc} -> {to_loc}")
                    return
                if kind == 'entry':
                    self.log(f"Army-box entry rewrite: {to_loc} -> {x_name}")
                    to_loc = x_name
                elif kind == 'exit':
                    self.log(f"Army-box exit rewrite: {from_loc} -> {x_name}")
                    from_loc = x_name
            elif 'Mordor ' in from_loc or 'Mordor ' in to_loc:
                # USER RULING 1.156 (t125 fix 6): army units dragged
                # onto Mordor-track boxes are a common human shortcut
                # for being IN GORGOROTH — the army state joins/creates
                # the Gorgoroth army.  The adjacency-inference pre-pass
                # above still resolves the classic parking pairs
                # (canary-certified); this branch handles the residue
                # the pre-pass could not resolve (sharpz L104: the
                # whole Barad-Dur garrison dragged onto 'Mordor 5' and
                # never dragged out — previously skipped, leaving a
                # phantom Barad-Dur army the client board disproves).
                unit_pieces = [p for p in pieces
                               if p not in self._TRACK_MARKER_PIECES]
                if unit_pieces:
                    nf = 'Gorgoroth' if 'Mordor ' in from_loc else from_loc
                    nt = 'Gorgoroth' if 'Mordor ' in to_loc else to_loc
                    if 'Mordor ' in from_loc:
                        # Box EXIT: humans re-drag the same handful out
                        # of the box repeatedly (Vayze/ProniX L69-71:
                        # 524->Morannon, 524->Morannon Stronghold,
                        # 524->Morannon again; the client moves BY
                        # PIECE ID so the extra drags are no-ops from
                        # wherever the pieces already are).  Only the
                        # drag that actually takes the pieces out of
                        # Gorgoroth is a move; later box-sourced drags
                        # of pieces tracked elsewhere are noise.
                        _zs = {sanitize(str(self.piece_region.get(p)))
                               for p in unit_pieces
                               if self.piece_region.get(p) is not None}
                        _known = {z for z in _zs if z in board.regions}
                        if _known and 'gorgoroth' not in _known:
                            self.log(f"Army-box re-drag noise "
                                     f"(ruling 1.156, pieces already at "
                                     f"{sorted(_known)}): "
                                     f"{from_loc} -> {to_loc}")
                            return
                    if sanitize(nf) == sanitize(nt):
                        self.log(f"Army-box drag within Gorgoroth "
                                 f"(ruling 1.156, state noise): "
                                 f"{from_loc} -> {to_loc}")
                        return
                    self.log(f"Army-box move resolved as Gorgoroth "
                             f"(ruling 1.156): {from_loc} -> {to_loc}")
                    from_loc, to_loc = nf, nt

        # FP action tokens: check before any early returns.  The
        # all-tokens early return must come AFTER the loop — returning
        # inside it dropped every token after the first in a GROUP
        # placement (2025-05-16 line 38: Controls.281 moves '427 426'
        # from FP Reinf to FP Dice; only the card token registered and
        # the political-token bid compensation was lost).
        if any(p in (TOKEN_POLITICAL, TOKEN_CARD_DRAW) for p in pieces):
            for p in pieces:
                if p == TOKEN_POLITICAL or p == TOKEN_CARD_DRAW:
                    self._handle_token_move(p, from_loc, to_loc)
            # handled tokens never continue into the unit paths
            pieces = [p for p in pieces
                      if p not in (TOKEN_POLITICAL, TOKEN_CARD_DRAW)]
            if not pieces:
                return

        # Dwarven Ring (bid compensation; user ruling 2026-07-03).
        # Arrival into a dice pool = the bid grant; leaving the pool
        # (-> Current FP Card / Current Action) = the USE (start of turn,
        # reroll one die to a different face-class).  The single-die
        # re-roll line that follows carries the outcome and is folded
        # into this record as new_face (see _handle_dice_roll).
        _ring_ps = [p for p in pieces if p in DWARVEN_RING_PIECES]
        if _ring_ps and len(_ring_ps) == len(pieces):
            # t138 (engine-check family 2): grants arrive as GROUP drags
            # of ring pieces 428-431 ('adds 2 Dwarven Ring(s)' then
            # '429 428 [storage] -> [FP Dice]') — the old branch only
            # accepted a single piece 428, so multi-ring grants never
            # incremented the counter and every later use was illegal
            # at the certifier (27/42 misses, rings=0).  SIDE comes from
            # the POOL zone ('FP Dice'/'SA Dice'), never from 'SA'
            # appearing anywhere in the pair (a use dragged onto
            # SA Card Play misattributed an FP ring to 'sh').
            if 'Dice' in to_loc and 'Dice' not in from_loc:
                ring_side = Side.SHADOW if 'SA' in to_loc \
                    else Side.FREE_PEOPLES
                attr = 'fp_dwarven_rings' if ring_side == Side.FREE_PEOPLES \
                    else 'sh_dwarven_rings'
                setattr(self.state, attr,
                        getattr(self.state, attr, 0) + len(_ring_ps))
                self.log(f"Dwarven Ring granted (bid, x{len(_ring_ps)}): "
                         f"{attr}={getattr(self.state, attr)}")
            elif 'Dice' in from_loc:
                ring_side = Side.SHADOW if 'SA' in from_loc \
                    else Side.FREE_PEOPLES
                attr = 'fp_dwarven_rings' if ring_side == Side.FREE_PEOPLES \
                    else 'sh_dwarven_rings'
                if getattr(self.state, attr, 0) <= 0:
                    # use with no parsed grant: infer one (belt-and-
                    # braces; some clients park rings without a drag)
                    setattr(self.state, attr, 1)
                    self.log(f"Dwarven Ring inferred grant ({attr}=1)")
                params = {'side': 'fp' if ring_side == Side.FREE_PEOPLES
                          else 'sh'}
                self._dwarven_use_params = params
                self._record_action('dwarven_ring', params, ring_side)
                self.log("Dwarven Ring used (reroll follows)")
            return

        # Control markers / VP markers: explicit ground-truth bookkeeping.
        # Must intercept BEFORE the muster path (previously a Shadow
        # control marker from SA Reinforcements to Pelargir mustered a
        # phantom unit, and the VP marker's undecoded 'Game.392' target
        # produced a phantom 'muster at Game.392').
        # t138 Q11 (per-drag context inference): clients also use the
        # 405-420 marker pids as OVERFLOW UNIT STOCK when the normal
        # pieces run out (9 proven games, side- and class-agnostic).
        # Unambiguous discriminator: a marker dragged from a REINF box
        # onto a region that still holds units HOSTILE to the marker's
        # side cannot be a control capture (you never control a region
        # with enemy units in it, rulebook) — route it to the muster
        # path instead (Wems/Gileforn L133 '416 SA Reinf -> Dale' cost
        # one Sauron regular for 22 checkpoints).  Anything else keeps
        # capture semantics; ambiguous shapes are curation-flagged.
        if any(p in CONTROL_MARKERS or p in VP_MARKERS for p in pieces):
            _q11 = []
            if 'Reinf' in from_loc and is_map_region(to_loc):
                for p in pieces:
                    if p not in CONTROL_MARKERS:
                        continue
                    _mk_side = (Side.SHADOW if p in SA_CONTROL_MARKERS
                                else Side.FREE_PEOPLES)
                    if self._region_holds_enemy_of(sanitize(to_loc),
                                                   _mk_side):
                        _q11.append(p)
            if _q11:
                self.log(f"CURATION NOTE (Q11): marker pid(s) {_q11} "
                         f"from {from_loc} onto contested {to_loc} = "
                         f"overflow unit stock (context-inferred "
                         f"muster)")
                _n_bef11 = len(self.action_log)
                self._handle_muster(_q11, to_loc, 'SA' in from_loc)
                if (len(self.action_log) > _n_bef11
                        and self.action_log[-1][0] == 'muster'):
                    # overflow restock rides the ongoing action — never
                    # its own muster decision
                    self.action_log[-1][1]['auto'] = True
            for p in pieces:
                if p in _q11:
                    continue
                if p in CONTROL_MARKERS or p in VP_MARKERS:
                    self._handle_marker_move(p, from_loc, to_loc)
            pieces = [p for p in pieces
                      if p not in CONTROL_MARKERS and p not in VP_MARKERS]
            if not pieces:
                return

        # Action-die pieces are never units.  Dice ride Scribe moves when
        # players drop a die on a map region as an action marker
        # (2023-12-06 T5 line 462: SH die 105 dropped on Morannon
        # mustered a phantom FP regular whose later 'kill' synthesized a
        # phantom Gorgoroth->Morannon battle) or tidy dice between UI
        # boxes.  Route them through dice bookkeeping and drop them from
        # unit handling.  Dice-location moves keep the dedicated branch.
        if 'Dice' not in from_loc and 'Dice' not in to_loc:
            _die_ps = [p for p in pieces if self._is_die_piece(p)]
            if _die_ps:
                for p in _die_ps:
                    if 'Used' in to_loc or 'Current' in to_loc:
                        mp = self.die_id_map.get(int(p))
                        if mp is not None:
                            sd_, ix_ = mp
                            if ix_ < len(self.state.dice[sd_]):
                                d_ = self.state.dice[sd_][ix_]
                                if not d_.used:
                                    d_.used = True
                                    self.log(f"Die {p} marked used "
                                             f"({from_loc} -> {to_loc})")
                    elif 'Hunt' in to_loc:
                        # t133 (helios2678 L173): a die dragged from
                        # Used/Current INTO the Hunt Box — the FP
                        # fellowship-move die placement (rules p.36:
                        # the action die used to move the Fellowship
                        # is added to the Hunt Box).  No 'Dice' token
                        # on either side, so the dice branch never saw
                        # it and the parser's hunt-box census ran
                        # short.  _handle_die_to_hunt is idempotent
                        # for dice already in the box.
                        try:
                            self._handle_die_to_hunt(p)
                        except ValueError:
                            pass
                    elif 'Reinf' in to_loc:
                        # t133 fix F4b: a die retired to a REINF box
                        # from ANY zone is removed from play — the
                        # rulebook's die-loss on character elimination
                        # is applied by tidying the die out of the DEAD
                        # pile at turn start (p.18; Kaeldar L452 'SA
                        # Dead -> SA Reinf' after Saruman fell).  The
                        # old guard required 'Dice' in from_loc, so the
                        # dead-pile variant kept a phantom die forever.
                        _di = int(p)
                        self._dice_out_of_play.add(_di)
                        mp = self.die_id_map.get(_di)
                        if mp is not None:
                            sd_, ix_ = mp
                            if ix_ < len(self.state.dice[sd_]):
                                d_ = self.state.dice[sd_][ix_]
                                if d_ in self.state.hunt_box:
                                    self.state.hunt_box.remove(d_)
                        self.piece_region[p] = to_loc
                        self.log(f"Die {p} removed from play "
                                 f"({from_loc} -> {to_loc})")
                pieces = [p for p in pieces if p not in _die_ps]
                self.log(f"Die piece(s) {_die_ps}: {from_loc} -> "
                         f"{to_loc} (not units)")
                if not pieces:
                    return

        # Muster from reinforcements to map
        if 'Reinf' in from_loc and is_map_region(to_loc):
            is_shadow = 'SA' in from_loc
            # ruling 1.162: pids already tracked on the map re-drag as
            # moves/no-ops, never as second musters
            pieces = self._redirect_tracked_musters(pieces, to_loc,
                                                    to_loc)
            if pieces:
                self._handle_muster(pieces, to_loc, is_shadow)
            return

        # Reinforcements placed directly into a besieged stronghold BOX
        # (card effects: Imrahil, Cirdan's Ships elves, ...). Units live in
        # the base region's army like every other siege participant.
        # Previously dropped entirely (Reinf -> non-map skip).
        _reinf_sh_base = (stronghold_base_region(to_loc)
                          if 'Reinf' in from_loc else None)
        if _reinf_sh_base is not None:
            pieces = self._redirect_tracked_musters(pieces,
                                                    _reinf_sh_base,
                                                    to_loc)
            if not pieces:
                return
            n_before = len(self.action_log)
            self._handle_muster(pieces, _reinf_sh_base, 'SA' in from_loc)
            # Placement into a besieged stronghold is only possible via
            # card effects — never a muster-die decision
            if (len(self.action_log) > n_before
                    and self.action_log[-1][0] == 'muster'):
                self.action_log[-1][1]['auto'] = True
            return

        # Companion moving from reinforcements to Guide/Fellowship box (e.g. Gollum)
        if 'Reinf' in from_loc and ('Guide' in to_loc or 'Fellowship' in to_loc):
            self._handle_fellowship_companions(pieces, from_loc, to_loc)
            return

        # t133 (Alexey'a family): a companion piece dragged from an
        # OFF-BOARD zone (dead pile / card play / Used) back INTO the
        # Fellowship or Guide Box is a revival/rejoin — the players
        # retracting a mis-clicked hunt sacrifice (helios2678 L120/121:
        # Controls.644 kills 241, then '241 FP Dead -> Guide Box' one
        # line later; Strider then guides, crowns and walks for the
        # rest of the game).  Zone wins: mirror the client board.
        if (('Guide' in to_loc or 'Fellowship' in to_loc)
                and not is_map_region(from_loc)
                and any(p in COMPANIONS for p in pieces)):
            for p in pieces:
                comp_name = COMPANIONS.get(p)
                if not comp_name or comp_name == 'Gollum':
                    continue
                if not any(c.name == comp_name
                           for c in self.state.fellowship.companions):
                    lv, ld = _COMPANION_LEVELS.get(comp_name, (1, 1))
                    self.state.fellowship.companions.append(
                        gs.Companion(name=comp_name, level=lv,
                                     leadership=ld))
                    if comp_name == 'Gandalf the Grey':
                        self._gandalf_dead = False
                    self.log(f"Companion rejoined fellowship "
                             f"(revival drag): {comp_name}")
                self._set_piece_zone(p, to_loc, dead=False)
            self._handle_fellowship_companions(
                [p for p in pieces if p in COMPANIONS],
                from_loc, to_loc)
            return

        # Skip other reinforcement moves (Reinf → non-map)
        if 'Reinf' in from_loc:
            return

        # Fellowship progress (247/248) / Corruption (252) markers dragged
        # along the track via Scribe moves. Controls.790 covers normal
        # progress moves; group drags move FSP + Corruption together
        # ("moves group (FSP, Corruption) from Fellowship Track 2 to 3").
        # Corruption value = track number. State sync only — the actual
        # decisions (hunt damage etc.) are recorded elsewhere.
        if 'Track ' in from_loc and 'Track ' in to_loc:
            m_to = re.search(r'Track (\d+)', to_loc)
            if m_to:
                pos = int(m_to.group(1))
                for p in pieces:
                    if p == '252':
                        self._sync_corruption(pos)
                    elif p in ('247', '248'):
                        self.state.fellowship.progress = pos
                        self.log(f"Fellowship progress (marker drag) -> {pos}")
            return

        # Die movements: track consumption when die goes from pool to used
        if 'Dice' in from_loc or 'Dice' in to_loc:
            # t138 (triage 4.6 / pass-5 3.4): NAMED pieces riding a
            # dice-zone leg are NOT dice — Saruman entered play via
            # 'SA Reinf -> SA Dice' (Controls.281 bounce) then
            # 'SA Dice -> Orthanc'; the old dispatch consumed both legs
            # and Saruman never mustered (25 checkpoints).  A named pid
            # leaving a dice zone for the MAP musters/returns; entering
            # a dice zone just zone-parks.
            _named9 = [p for p in pieces if not self._is_die_piece(p)
                       and (p in MINION_PIECES or p in COMPANIONS)]
            if _named9:
                for p in _named9:
                    if is_map_region(to_loc) and p in MINION_PIECES \
                            and not self._pid_zone_dead(
                                self.piece_region.get(p)):
                        self._handle_muster([p], to_loc, True)
                    elif is_map_region(to_loc) and p in COMPANIONS:
                        _tk9b, _ta9b = self._find_or_create_army(to_loc)
                        self._return_named_piece(p, _tk9b, _ta9b,
                                                 from_loc, to_loc)
                    else:
                        self.piece_region[p] = to_loc
                        self.log(f"Named piece {p} parked via dice "
                                 f"zone: {from_loc} -> {to_loc}")
                pieces = [p for p in pieces if p not in _named9]
                if not pieces:
                    return
            if 'Dice' in from_loc and ('Used' in to_loc or 'Current' in to_loc):
                for p in pieces:
                    try:
                        die_id = int(p)
                        if die_id in self.die_id_map:
                            self._track_die_consumed(die_id)
                    except ValueError:
                        pass
            elif 'Dice' in from_loc and 'Reinf' in to_loc:
                # t125: a die dragged from the pool back to a
                # reinforcement box = REMOVED FROM PLAY (the Shadow
                # loses a die when a minion dies; Adobim L645 after
                # Saruman's elimination).  The client censuses simply
                # stop counting the piece.
                for p in pieces:
                    if self._is_die_piece(p):
                        self._dice_out_of_play.add(int(p))
                        self.piece_region[p] = to_loc
                        self.log(f"Die {p} removed from play "
                                 f"(-> {to_loc})")
            elif 'Hunt' in from_loc and 'Dice' in to_loc:
                # t125: turn-start hunt-box retrieval (Controls.757
                # 'die from Hunt Box to SA Dice' and friends).  The
                # client's die is physically back in the pool; the
                # parser kept it in state.hunt_box until the turn
                # marker, so every checkpoint landing in that window
                # showed phantom hunt-box dice (Adobim: one window per
                # turn boundary, 9 of 33 checkpoints).
                for p in pieces:
                    try:
                        mp = self.die_id_map.get(int(p))
                    except ValueError:
                        mp = None
                    if mp is None:
                        continue
                    sd_, ix_ = mp
                    if ix_ < len(self.state.dice[sd_]):
                        d_ = self.state.dice[sd_][ix_]
                        if d_ in self.state.hunt_box:
                            self.state.hunt_box.remove(d_)
                        d_.used = False
                        self.log(f"Die {p} retrieved from Hunt Box "
                                 f"to pool")
                    self.piece_region[p] = to_loc
            elif 'Dice' in to_loc and 'Dice' not in from_loc:
                # A removed die re-entering a pool is back in play
                for p in pieces:
                    if self._is_die_piece(p):
                        self._dice_out_of_play.discard(int(p))
                # Die RETURNED to the pool from ANY off-pool zone —
                # dwarven-ring reroll flow (Used/Current -> pool,
                # 2026-01-05 kapcerz T1 L48-51) AND the turn-start
                # retrieval grammars from the DEAD piles (t125:
                # Controls.731 'FP Dead -> FP Dice', Controls.762
                # 'SA Dead -> SA Dice'; unhandled, retrieved dice
                # stayed 'used' until the turn marker and every
                # checkpoint in that window showed a short pool —
                # Adobim/Rizino1 dumps).  Un-consume it.
                for p in pieces:
                    try:
                        die_id = int(p)
                    except ValueError:
                        continue
                    mp = self.die_id_map.get(die_id)
                    if mp is None:
                        continue
                    sd_, ix_ = mp
                    if ix_ < len(self.state.dice[sd_]):
                        d_ = self.state.dice[sd_][ix_]
                        if d_.used:
                            d_.used = False
                            self.log(f"Die {p} returned to pool (unused)")
            elif 'Dice' in from_loc and 'Hunt' in to_loc:
                # Manual Scribe drag of dice from the pool into the Hunt
                # Box = eye pre-allocation (some players drag instead of
                # using the Controls.372 button; verified in
                # 2023-12-07_001647 lines 54/96/151/205/306/...)
                side_hint = Side.SHADOW if 'SA' in from_loc else None
                noise = getattr(self, '_eye_drag_noise', set())
                for p in pieces:
                    if (self.line_num, p) in noise:
                        self.log(f"Retracted eye allocation skipped "
                                 f"(die {p})")
                        continue
                    try:
                        self._handle_die_to_hunt(p, side_hint=side_hint)
                    except ValueError:
                        pass
            return

        # Political track
        if 'Political' in from_loc or 'Political' in to_loc:
            self._handle_political(pieces, from_loc, to_loc)
            return

        # t138 N7 exit leg: army pids PARKED in a Fellowship/Guide box
        # waypoint walk back out to the map (Mirdain/buondiavolo
        # L391-395: a Shadow army walked out of the Mount Gundabad box
        # THROUGH Game.1998 to Eagles' Eyrie) — re-materialize the
        # stashed objects at the destination.  Must precede the
        # fellowship routing below, which would swallow the exit.
        _park9 = getattr(self, '_army_park', None)
        if _park9 and is_map_region(to_loc) \
                and any(p in _park9 for p in pieces):
            _tk9, _ta9 = self._find_or_create_army(to_loc)
            _n_out = {'regular': 0, 'elite': 0, 'leader': 0, 'nazgul': 0}
            _orig9 = None
            _done9 = set()
            for p in pieces:
                if p not in _park9:
                    continue
                obj, kind, _orig9 = _park9.pop(p)
                if kind in ('regular', 'elite'):
                    _ta9.units.append(obj)
                else:
                    _ta9.leaders.append(obj)
                _n_out[kind] += 1
                self.piece_region[p] = _tk9
                _done9.add(p)
            self.log(f"Army un-parked -> {to_loc} "
                     f"({sum(_n_out.values())} pcs from {_orig9})")
            if _orig9:
                self._record_action('army_move', {
                    'from': _orig9, 'to': to_loc, 'moved': _n_out,
                    '_pieces': sorted(_done9)},
                    self._piece_side(sorted(_done9)))
            pieces = [p for p in pieces if p not in _done9]
            if not pieces:
                return

        # Fellowship/Guide movements
        if 'Fellowship' in from_loc or 'Guide' in from_loc:
            self._handle_fellowship_companions(pieces, from_loc, to_loc)
            return

        # --- Stronghold boxes (siege mechanics) -------------------------
        # "X Stronghold" is an off-map VASSAL box. Moves into it are the
        # defender's RETREAT INTO SIEGE (previously misparsed as
        # casualties, deleting the garrison!); moves from it back to X are
        # the siege lifting (previously phantom musters); moves from it to
        # Dead/Used are the garrison's battle casualties (previously
        # invisible). Replay keeps besieged units in the region.
        def _stronghold_of(loc: str) -> Optional[str]:
            if loc and loc.endswith(' Stronghold'):
                return loc[:-len(' Stronghold')]
            return None

        sh_to = _stronghold_of(to_loc)
        sh_from = _stronghold_of(from_loc)
        if sh_from is not None and is_map_region(to_loc)                 and any(p in ('247', '248') for p in pieces):
            if sanitize(sh_from) == sanitize(to_loc):
                # t133 fix F3: the ring-bearers figure dragged between
                # a region and its OWN stronghold box (siege display
                # shuffles, courtesy drag-backs) is NEVER a declare or
                # reveal — no region change happened, so the progress
                # counter keeps its step (rulebook p.37: only a declare
                # or a hunt reveal resets the counter; 97 games reset
                # progress to 0 on this shuffle).
                for p in pieces:
                    if p in ('247', '248'):
                        self.piece_region[p] = to_loc
                self.log(f"Fellowship figure box shuffle "
                         f"({from_loc} -> {to_loc}): display only, "
                         f"no declare/reveal")
            else:
                # The FELLOWSHIP leaving a besieged stronghold's BOX for
                # a DIFFERENT map region is a reveal/declare placement,
                # never units (2023-08-13 HombreLobo T7 L811: 247 from
                # Minas Tirith Stronghold -> Minas Morgul — the gateway
                # reveal that ENTERS MORDOR — used to parse as a
                # phantom 1R muster at Minas Morgul, so the teacher
                # never entered the track and every T8+ mordor_move
                # went stranded).
                self._handle_army_move(
                    [p for p in pieces if p in ('247', '248')],
                    sh_from, to_loc)
            pieces = [p for p in pieces if p not in ('247', '248')]
            if not pieces:
                return
        if sh_to is not None and is_map_region(from_loc) \
                and sanitize(sh_to) == sanitize(from_loc):
            self._maybe_synth_attack(from_loc)
            _g_side = self._piece_side(pieces)
            self._record_action('retreat_into_siege',
                                {'region': from_loc},
                                _g_side)
            if not hasattr(self.state, 'besieged_strongholds'):
                self.state.besieged_strongholds = set()
            self.state.besieged_strongholds.add(sanitize(from_loc))
            # Garrison-side tracking (see __init__): control/siege-over
            # decisions during the siege key off WHO is inside the box.
            _san_g = sanitize(from_loc)
            self._siege_garrison[_san_g] = _g_side
            self._siege_arrived.discard(_san_g)
            # Piece-level tracking mirrors the client: the garrison now
            # physically sits in the stronghold SUB-BOX zone (army state
            # is untouched — the kill handler already maps a box-tracked
            # prev back to the base region via stronghold_base_region).
            for p in pieces:
                self.piece_region[p] = to_loc
            self.log(f"Retreat into siege at {from_loc} "
                     f"(units stay in region)")
            return
        if sh_to is not None and is_map_region(from_loc) \
                and sanitize(sh_to) != sanitize(from_loc) \
                and sanitize(sh_to) in board.regions:
            # t133 fix F15 (+F9 geography): a drag from a DIFFERENT
            # region into a stronghold BOX joins the garrison inside
            # that stronghold — nazgul flights into a friendly besieged
            # stronghold are plain printed rules (Moving Nazgul: "can
            # never move this way into a region that contains a
            # Stronghold controlled by the Free Peoples player, unless
            # that Stronghold is besieged by a Shadow Army");
            # companion/GtW walks mirror the client drag (zone wins,
            # Q6/A6).  The old handlers only accepted a box paired with
            # ITS OWN region, so these pieces were deleted as
            # leaving-the-board casualties (74 games).  t138: side_lock
            # so a contested source region never donates enemy units.
            self._handle_army_move(pieces, from_loc, sh_to,
                                   side_lock=self._piece_side(pieces))
            for p in pieces:
                self.piece_region[p] = to_loc
            self.log(f"Cross-region move into stronghold box: "
                     f"{from_loc} -> {to_loc} (garrison of {sh_to})")
            return
        if sh_to is not None and not is_map_region(from_loc) \
                and sanitize(sh_to) in board.regions:
            # t133 fix F12: a NAMED character pid leaving a dead/
            # casualty/card-play zone for the stronghold box is a
            # character appearance/revival, never a unit refund.
            # (Stronghold-box SOURCES are excluded: box->box garrison
            # transfers are a human MISTAKE per USER RULING 2026-07-17
            # Q7 — "those are mistakes, do not implement those" — and
            # keep the old muster-at-destination behavior, flagged for
            # curation.)
            if _stronghold_of(from_loc) is None:
                _comp_ps = [p for p in pieces
                            if p in COMPANIONS or p in MINION_PIECES]
                if _comp_ps:
                    _bk, _ba = self._find_or_create_army(sh_to)
                    for p in _comp_ps:
                        self._return_named_piece(p, _bk, _ba,
                                                 from_loc, to_loc)
                    pieces = [p for p in pieces if p not in _comp_ps]
                    if not pieces:
                        return
            # Units returning from casualties/dead into a besieged
            # stronghold box (elite downgrade return: "moves group
            # ([2]Gondor Regular) from casualties to Minas Tirith
            # Stronghold"). A consequence of the casualty choice, never a
            # standalone decision.
            if _stronghold_of(from_loc) is None:
                # ruling 1.162 idempotency (box SOURCES excluded per
                # the Q7 negative ruling — box->box drags keep the old
                # behavior and stay curation candidates)
                pieces = self._redirect_tracked_musters(pieces, sh_to,
                                                        to_loc)
                if not pieces:
                    return
            n_before = len(self.action_log)
            self._handle_muster(pieces, sh_to,
                                self._piece_side(pieces) == Side.SHADOW)
            if (len(self.action_log) > n_before
                    and self.action_log[-1][0] == 'muster'):
                self.action_log[-1][1]['auto'] = True
            self.log(f"Downgrade return into {sh_to} (from {from_loc})")
            return
        if sh_from is not None:
            region = sh_from
            if is_map_region(to_loc) and sanitize(to_loc) == sanitize(region):
                getattr(self.state, 'besieged_strongholds',
                        set()).discard(sanitize(region))
                self._siege_garrison.pop(sanitize(region), None)
                self._siege_arrived.discard(sanitize(region))
                # survivors leave the sub-box: track them back in the
                # region zone (mirrors the client's drag)
                for p in pieces:
                    self.piece_region[p] = to_loc
                self.log(f"Siege lifted at {region} (units already in region)")
                return
            if is_map_region(to_loc) and sanitize(region) in board.regions:
                # t133 fix F15 (exit leg): pieces leaving a stronghold
                # BOX for a DIFFERENT map region move out of the box's
                # base-region army (nazgul flying out of the besieged
                # garrison, Mount Gundabad L570 '039 040 Game.19 ->
                # Moria').  Previously fell through to the off-board-
                # return branch and RE-mustered pieces the parser had
                # already lost.  t138 (pass-4 5.1 regression): at a
                # CONTESTED region the drag must select by the pids'
                # OWN side, never the besieger's army — side_lock.
                self._handle_army_move(pieces, region, to_loc,
                                       side_lock=self._piece_side(pieces))
                self.log(f"Cross-region move out of stronghold box: "
                         f"{from_loc} -> {to_loc}")
                return
            if not is_map_region(to_loc):
                if to_loc in self._ARMY_PARK_BOXES:
                    # t138 N7: box garrison walked out THROUGH a
                    # Fellowship/Guide box waypoint — transit-park
                    self._park_army_pids(pieces, region, to_loc)
                    return
                # Garrison casualties during the siege
                self._maybe_synth_attack(region)
                self._remove_pieces_from_armies(pieces, region, to_loc)
                self._combat_casualties_this_round = True
                return

        # Mordor track marker moved via Scribe.141.  MUST precede the
        # map->non-map branch below: the ENTRY drag (map region ->
        # 'Mordor 0', e.g. Morannon -> Game.524) was previously swallowed
        # as "units leaving the board", so the replay state's
        # mordor_entered stayed False past the entry moment and the
        # Mouth-of-Sauron legality check failed at the muster line
        # (game 1 line 644 entry vs line 665 muster; game 2 line 801
        # entry vs line 817 muster).  Between-box drags (both locs
        # 'Mordor N') were unaffected.
        if '247' in pieces and ('Mordor ' in from_loc or 'Mordor ' in to_loc):
            self._handle_fellowship_track(from_loc, to_loc)
            return

        # t138 Q9 (USER ruling 1.169): a companion dragged from the MAP
        # back into the Fellowship/Guide Box is the player's UNDO of
        # the prior separation/casualty staging — "if things are undone
        # in the log, then we should undo it as well".  Restore the
        # state: pop the object from whatever army holds it, reinsert
        # into fs.companions, retract the recent separation record.
        # (22 games; the printed no-rejoin rule p.39 loses to the
        # explicit human correction under zone-wins.)  Mixed drags
        # (GoSagan/Webbely L503: pawn + 4 companions + a stray unit in
        # one drag) resolve per-pid; leftovers fall through.
        if to_loc in ('Fellowship Box', 'Guide Box',
                      'Guide Of The Fellowship', 'Fellowship Of The Ring') \
                and is_map_region(from_loc) \
                and any(p in COMPANIONS for p in pieces):
            _left = []
            for p in pieces:
                comp_name = COMPANIONS.get(p)
                if not comp_name or comp_name == 'Gollum':
                    if p in ('247', '248'):
                        self.piece_region[p] = to_loc
                    else:
                        _left.append(p)
                    continue
                _names = set(names_for_pid(p)) or {comp_name}
                popped = None
                for _k in list(self.state.armies):
                    _a = self.state.armies[_k]
                    for i, l in enumerate(_a.leaders):
                        if getattr(l, 'name', None) in _names:
                            popped = _a.leaders.pop(i)
                            if not _a.units and not _a.leaders:
                                del self.state.armies[_k]
                            break
                    if popped is not None:
                        break
                if not any(c.name in _names
                           for c in self.state.fellowship.companions):
                    if isinstance(popped, gs.Companion):
                        comp = popped
                    else:
                        lv, ld = _COMPANION_LEVELS.get(comp_name, (1, 1))
                        comp = gs.Companion(name=comp_name, level=lv,
                                            leadership=ld)
                    self.state.fellowship.companions.append(comp)
                else:
                    comp = next(c for c in
                                self.state.fellowship.companions
                                if c.name in _names)
                self._set_piece_zone(p, to_loc, dead=False)
                if comp_name == 'Gandalf the Grey':
                    self._gandalf_dead = False
                if 'Guide' in to_loc:
                    self.state.fellowship.guide = comp
                self.log(f"Companion re-entry (separation undo, Q9): "
                         f"{comp_name} -> {to_loc}")
                # retract the recent separation/casualty record
                for _b9 in range(1, min(11, len(self.action_log)) + 1):
                    _at9, _ap9, _s9 = self.action_log[-_b9]
                    if _at9 in ('companion_separate', 'companion_death',
                                'hunt_sacrifice_guide',
                                'hunt_sacrifice_random') \
                            and str((_ap9 or {}).get('companion')) \
                            in _names:
                        del self.action_log[-_b9]
                        self.log("  (separation/casualty record "
                                 "retracted, Q9 undo)")
                        break
                # and drop the name from a recent group walk record
                for _b9 in range(1, min(11, len(self.action_log)) + 1):
                    _at9, _ap9, _s9 = self.action_log[-_b9]
                    if _at9 == 'companion_move' and comp_name in (
                            (_ap9 or {}).get('companions') or []):
                        _ap9['companions'] = [
                            c for c in _ap9['companions']
                            if c != comp_name]
                        if not _ap9['companions']:
                            del self.action_log[-_b9]
                        self.log("  (walked-back name dropped from "
                                 "group walk, Q9 undo)")
                        break
            # a Gollum guide with companions present is engine-illegal
            # (p.39: Gollum guides only when alone) — reassign
            _g9 = self.state.fellowship.guide
            if _g9 is not None and _g9.name == 'Gollum' \
                    and self.state.fellowship.companions:
                self._reassign_guide()
            if not _left:
                return
            pieces = _left

        # Units leaving the board (map → non-map: Dead, Used, Card Play, Board, etc.)
        if is_map_region(from_loc) and not is_map_region(to_loc):
            if to_loc in self._ARMY_PARK_BOXES:
                # t138 N7: map army into a Fellowship/Guide box =
                # transit-parking, never casualties
                self._park_army_pids(pieces, from_loc, to_loc)
                return
            # (Separated companions killed from a map region are now
            # handled BY IDENTITY inside _remove_pieces_from_armies —
            # t133 fix F10, alias-matched so GtW/Aragorn faces resolve.)
            from_san = sanitize(from_loc)
            # Combat-card effect cost: army units moved onto the active
            # combat card ("moves group ([4]Sauron Regular) from Minas
            # Tirith to Current SA Card") during a battle window = the
            # owner's chosen effect magnitude (game0 T3: Onslaught —
            # sacrifice 4, roll 4 dice at 4+). The units ARE removed
            # below as the effect cost; the record captures the DECISION
            # (count) without double-counting the casualties.
            if to_loc in ('SA Card Play', 'FP Card Play') \
                    and self._battle_synth_region is not None:
                n_units = sum(
                    1 for pp in pieces
                    if get_unit_type_from_piece_id(pp, self.piece_types)[1]
                    in (UnitKind.REGULAR, UnitKind.ELITE))
                fx_side = (Side.SHADOW if to_loc == 'SA Card Play'
                           else Side.FREE_PEOPLES)
                # Only Onslaught-class combat cards have a units-onto-card
                # SACRIFICE choice.  Units dragged onto any other card
                # during a battle are just casualties parked on the play
                # area (2023-12-07_001647 T5: SH dead on 'Monsters
                # Roused'/Desperate Battle, which has no choice) — the
                # removal below still applies, but it is not a decision.
                _cc_name = _normalize_name(
                    self._last_combat_card_name.get(fx_side, ''))
                _is_choice = (
                    _cc_name in _ONSLAUGHT_EVENT_CARDS
                    # Relentless Assault: only the PRE-ROLL window drag
                    # is the self-hit choice (see _RELENTLESS_EVENT_CARDS)
                    or (_cc_name in _RELENTLESS_EVENT_CARDS
                        and self._cc_pre_roll.get(fx_side, False)))
                if n_units > 0 and _is_choice:
                    # 134 (2026-07-25 board-is-set T8 L558): the
                    # sacrificed PIECES are the human's decision - a
                    # bare count let the engine pick its own units
                    # (policy) and 2 paid SAUR elites survived at MT
                    # for the rest of the game.  Carry the kind mix.
                    _ne134 = sum(
                        1 for pp in pieces
                        if get_unit_type_from_piece_id(
                            pp, self.piece_types)[1] == UnitKind.ELITE)
                    _nr134 = n_units - _ne134
                    self.action_log.append((
                        'combat_effect_choice',
                        {'count': n_units,
                         'kinds': {'regular': _nr134, 'elite': _ne134},
                         'card_name':
                             self._last_combat_card_name.get(fx_side, ''),
                         'region': from_loc}, fx_side))
                    self.log(f"{fx_side.name} combat-effect sacrifice: "
                             f"{n_units} unit(s) from {from_loc} onto card")
                    # v3: these removals are the effect's COST, never
                    # applied combat hits
                    self._fx_cost_drag_line = self.line_num
            # '?'-target battle: casualties leaving the board name the
            # battle region (2023-12-06 T6: the Fighting Uruk-hai siege
            # assault's attack stayed '?->?' — its only board evidence
            # was SH casualties parked on the card from Rivendell).
            # _maybe_synth_attack only PATCHES here (synth exists) and
            # validates the region holds the attacker's enemy.
            if self._battle_synth_region == '?':
                self._maybe_synth_attack(from_loc)
            elif (self._battle_synth_region is None
                    and getattr(self, '_battle_hint_region', None)
                    == sanitize(from_loc)):
                # Chat-declared battle whose only board evidence is this
                # casualty drag (2025-06-19 StellaRossa T6 'Pel NC': the
                # defender's single-unit kill arrives as a map->dead
                # MOVE, not via the Controls.296 casualty path, so no
                # attack was ever synthesized and the Pelargir capture
                # vanished).  _maybe_synth_attack validates the hint
                # region holds the die-consumer's enemy.
                self._maybe_synth_attack(from_loc)
            self._remove_pieces_from_armies(
                pieces, from_loc, to_loc,
                mark_cause=('effect_cost'
                            if getattr(self, '_fx_cost_drag_line', None)
                            == self.line_num else None))
            return

        # Units returning to the board from off-board (Dead/Card Play → map)
        if not is_map_region(from_loc) and is_map_region(to_loc):
            # Handle companion pieces returning to map (e.g., Gandalf resurrection)
            to_key, to_army = self._find_or_create_army(to_loc)

            for p in pieces:
                # (t138 note: minion pids arriving from off-board boxes
                # are handled by this branch's muster TAIL below — the
                # Saruman dice-zone bounce is covered by the named-
                # piece split in the dice-zone dispatch above.)
                comp_name = COMPANIONS.get(p)
                if comp_name:
                    # t138 (StregoRe family): a separation STAGED via a
                    # Board box completes here — place the stashed
                    # object and patch the deferred '?' region
                    _stash9 = getattr(self, '_board_sep_stash', None)
                    if _stash9 and p in _stash9:
                        comp = _stash9.pop(p)
                        to_army.leaders.append(comp)
                        self.piece_region[p] = to_key
                        for _b9 in range(1,
                                         min(9, len(self.action_log)) + 1):
                            _at9, _ap9, _s9 = self.action_log[-_b9]
                            if _at9 == 'companion_separate' \
                                    and _ap9.get('companion') == comp.name \
                                    and _ap9.get('to_region') == '?':
                                _ap9['to_region'] = sanitize(to_loc)
                                break
                        self.log(f"Separation completed via board box: "
                                 f"{comp.name} -> {to_loc}")
                        continue
                    if comp_name == 'Gandalf the Grey' and self._gandalf_dead:
                        # Dead Gandalf returning to the map IS the Gandalf
                        # the White transformation (W-die decision, incl.
                        # WHERE to place him) — previously never extracted
                        comp = gs.Companion(name='Gandalf the White',
                                            level=3, leadership=1)
                        to_army.leaders.append(comp)
                        self.piece_region[p] = to_key
                        if self.state.gandalf_the_white:
                            self.state.gandalf_the_white.in_play = True
                        self._record_action('transform_gandalf',
                                            {'region': to_loc},
                                            Side.FREE_PEOPLES)
                        self.log(f"Gandalf the White appears -> {to_loc}")
                        continue
                    # Add as a Companion leader (official card values; a
                    # hardcoded level=1 understated Legolas/Boromir for
                    # the rest of the game)
                    _lv9, _ld9 = _COMPANION_LEVELS.get(comp_name, (1, 1))
                    comp = gs.Companion(name=comp_name, level=_lv9,
                                        leadership=_ld9)
                    to_army.leaders.append(comp)
                    self.piece_region[p] = to_key
                    self.log(f"Companion returns: {comp_name} -> {to_loc}")
                    # A companion pulled from the fellowship while hunt
                    # damage is unresolved = the random-casualty decision
                    ctx = self._hunt_ctx
                    if ctx is not None and not ctx.get('resolved'):
                        ctx['resolved'] = True
                        self._record_action('hunt_sacrifice_random',
                                            {'companion': comp_name},
                                            Side.FREE_PEOPLES)
                        self._hunt_ctx = None
                    # A hunt-drawn Hobbit walking onto the map = his
                    # PLACEMENT decision (user rule: placed as if
                    # separated; 2023-12-07 line 617: Meriadoc ->
                    # Dagorlad after the Controls.637 draw at 615)
                    if comp_name in ('Meriadoc', 'Peregrin'):
                        for at2, p2, _sd2 in reversed(self.action_log[-3:]):
                            if at2 == 'hunt_sacrifice_random' and \
                                    sanitize(str(p2.get('companion', ''))) \
                                    == sanitize(comp_name):
                                self._record_action(
                                    'companion_move',
                                    {'from': from_loc, 'to': to_loc,
                                     'companions': [comp_name],
                                     'hunt_placement': True},
                                    Side.FREE_PEOPLES)
                                break

            # Remaining non-companion pieces treated as muster
            # 137 (2026-07-25 demanray L620/L623, client-dump proven):
            # the FELLOWSHIP PAWN (247) and its marker family are NEVER
            # army units - when the humans parked the pawn on a card
            # zone mid-battle and dragged it back, this muster tail
            # minted one phantom region-nation REGULAR per return trip
            # (Rivendell held 2 phantom Elven regulars vs the client
            # from cp625 for the rest of the game).  Dice pieces are
            # excluded for the same reason.
            non_comp = [p for p in pieces
                        if p not in COMPANIONS
                        and p not in FELLOWSHIP_MARKERS
                        and not self._is_die_piece(p)]
            if non_comp:
                # ruling 1.162: pids tracked on the map re-drag as
                # moves/no-ops (stale off-board SOURCE — the client
                # moves BY PID from wherever the piece really is)
                non_comp = self._redirect_tracked_musters(
                    non_comp, to_loc, to_key)
            if non_comp:
                # Determine side from piece nation lookup
                is_shadow = any(
                    get_unit_type_from_piece_id(p, self.piece_types)[0] in SHADOW_NATIONS
                    for p in non_comp
                    if get_unit_type_from_piece_id(p, self.piece_types)[0] is not None
                )
                self._handle_muster(non_comp, to_loc, is_shadow)
            return

        # (Mordor track marker branch hoisted above the map->non-map
        # branch, 2026-07-04 — see comment there.)

        # Army movement between map regions
        if is_map_region(from_loc) and is_map_region(to_loc):
            self._handle_army_move(pieces, from_loc, to_loc)
            return

        # Unclaimed off-board bookkeeping drags (t125 fix 1): keep
        # piece-level zones in client sync for tracked unit pids —
        # e.g. dead regulars tidied from FP Dead to FP Reinf
        # (GenericPie pids 078/082: the client board shows them in
        # FP Reinf, the parser froze them at 'Dead' and over-counted
        # the dead pile).  _set_piece_zone refunds the FP dead pool.
        for p in pieces:
            if self._is_die_piece(p) or p in COMPANIONS \
                    or p in MINION_PIECES or p in FELLOWSHIP_MARKERS:
                continue
            try:
                ip = int(p)
            except ValueError:
                continue
            if 106 <= ip <= 239:
                continue  # cards / FP dice / hunt tiles
            if p in self.piece_region:
                self._set_piece_zone(p, to_loc)

    _ARMY_PARK_BOXES = ('Fellowship Box', 'Guide Box')

    def _park_army_pids(self, pieces: List[str], region: str,
                        to_loc: str):
        """t138 N7: army pids dragged into a Fellowship/Guide box are
        TRANSIT-PARKED (waypoint walk), not casualties — pop the
        objects from the region army into a stash keyed by pid; the
        exit leg re-materializes them (Mirdain/buondiavolo L391-395).
        Dead-tracked pids and unknown pids just zone-park."""
        if not hasattr(self, '_army_park'):
            self._army_park = {}
        reg_san = sanitize(str(region))
        src_key = next((k for k in self.state.armies
                        if sanitize(k) == reg_san), None)
        army = self.state.armies.get(src_key) if src_key else None
        n = 0
        for p in pieces:
            if p in COMPANIONS or p in MINION_PIECES \
                    or self._is_die_piece(p) or p in CONTROL_MARKERS \
                    or p in VP_MARKERS or self._pid_zone_dead(
                        self.piece_region.get(p)):
                self.piece_region[p] = to_loc
                continue
            nation, ut = get_unit_type_from_piece_id(p, self.piece_types)
            obj = kind = None
            if army is not None:
                if ut == 'NAZGUL':
                    for i, l in enumerate(army.leaders):
                        if isinstance(l, Nazgul):
                            obj, kind = army.leaders.pop(i), 'nazgul'
                            break
                elif ut == 'LEADER':
                    for i, l in enumerate(army.leaders):
                        # type(l).__name__ != 'Minion' preserves the prior
                        # exclusion of WK/Saruman/Mouth: before the minion-
                        # typing fix they were Companions (had 'level') and
                        # never matched a generic-LEADER casualty here.
                        if not isinstance(l, Nazgul) \
                                and type(l).__name__ != 'Minion' \
                                and not hasattr(l, 'level') \
                                and (nation is None
                                     or getattr(l, 'nation', None)
                                     == nation):
                            obj, kind = army.leaders.pop(i), 'leader'
                            break
                elif ut in (UnitKind.REGULAR, UnitKind.ELITE):
                    for i, u in enumerate(army.units):
                        if u.kind == ut and (nation is None
                                             or u.nation == nation):
                            obj, kind = army.units.pop(i), str(
                                ut.value).lower()
                            break
            if obj is not None:
                self._army_park[p] = (obj, kind, region)
                n += 1
            self.piece_region[p] = to_loc
        if army is not None and not army.units and not army.leaders \
                and src_key in self.state.armies:
            del self.state.armies[src_key]
        self.log(f"Army transit-parked in {to_loc}: {n} pcs from "
                 f"{region} (N7 waypoint)")

    def _pid_display_name(self, pid: str) -> Optional[str]:
        """Current display name of a companion/minion piece, honouring an
        in-play TRANSFORMATION (Gandalf 240 Grey->White, Strider 241
        ->Aragorn).  The static COMPANIONS map only stores the base face,
        so a companion_move fx for a transformed figure carried the STALE
        name (Book of Mazarbul: 'Gandalf the Grey' vs the engine's
        'Gandalf the White') -> the teacher's move derivation found no
        matching companion and the figure never moved (PeterM T7: Gandalf
        never reached Erebor, so the Dwarves-to-war trigger never fired).
        Resolve to whichever alias face the engine state currently holds;
        fall back to the base name."""
        aliases = PIECE_NAME_ALIASES.get(pid)
        base = COMPANIONS.get(pid) or MINION_PIECES.get(pid)
        if not aliases:
            return base
        # Which alias face is presently on the board / in the fellowship?
        present = set()
        for c in self.state.fellowship.companions:
            nm = getattr(c, 'name', None)
            if nm in aliases:
                present.add(nm)
        for a in self.state.armies.values():
            for l in a.leaders:
                nm = getattr(l, 'name', None)
                if nm in aliases:
                    present.add(nm)
        # Prefer the transformed (second) face when it is the one in play.
        if aliases[1] in present:
            return aliases[1]
        if aliases[0] in present:
            return aliases[0]
        # Not found on the board (drift): fall back to the tracked Gandalf
        # death flag, else the base face.
        if pid == '240' and getattr(self, '_gandalf_dead', False):
            return aliases[1]
        return base

    def _handle_army_move(self, pieces: List[str], from_loc: str,
                          to_loc: str, side_lock=None):
        """Move army units between regions.

        *side_lock* (t138 wave-3, triage pass-4 5.1 regression fix):
        when set to a Side, unit/leader selection NEVER crosses sides —
        the box-to-map transfer at a contested (besieged) region used
        to resolve 'the army at the region' and dragged the BESIEGER's
        units out when the box garrison broke out (andre_crc/
        SlickRick441 L325: FP elves exit the Woodland Realm box, the
        parser moved 2 SAURON regulars instead).  With side_lock, a
        missing same-side object is CREATED from the pid's own
        nation/kind (the client demonstrably had it; zone wins) instead
        of stealing an enemy piece."""
        # Filter out fellowship markers - they're not army units!
        army_pieces = [p for p in pieces if p not in FELLOWSHIP_MARKERS]
        if not army_pieces:
            # This is a fellowship movement, not an army movement
            if '247' in pieces:
                # Fellowship pawn moved on the map. Either a voluntary
                # DECLARATION (hidden becomes True) or a hunt-forced REVEAL
                # placement (hidden becomes False) — distinguish via the
                # hunt-resolution flag.
                progress_consumed = self.state.fellowship.progress
                stash = getattr(self, '_pre_reset_progress', None)
                if stash is not None:
                    # Track reset line preceded the pawn move
                    progress_consumed = max(progress_consumed, stash)
                    self._pre_reset_progress = None
                self.fellowship_location = to_loc
                # Update the game state fellowship region
                try:
                    to_region = self.board.region(sanitize(to_loc))
                    self.state.fellowship.region = to_region
                except Exception as e:
                    # Fallback: store as string
                    self.state.fellowship.region = to_loc
                # Reset progress (declare/reveal consumes accumulated progress)
                self.state.fellowship.progress = 0
                # Multi-hop drag continuation: extend whichever record
                # (reveal placement or declare) this hop continues —
                # the DECISION is the final region of the drag
                last = self.action_log[-1] if self.action_log else None
                if (last is not None
                        and last[0] in ('fellowship_reveal_place',
                                        'fellowship_declare')
                        and sanitize(last[1].get('to', '')) ==
                            sanitize(from_loc)):
                    last[1]['to'] = to_loc
                    last[1]['progress_consumed'] = \
                        last[1].get('progress_consumed', 0) + progress_consumed
                    if last[0] == 'fellowship_declare':
                        # declare-record enrichment: keep the canonical
                        # region key + pre-reset progress on the FINAL
                        # region of the drag
                        last[1]['region'] = self._declare_region_key(to_loc)
                        last[1]['progress_before'] = \
                            last[1].get('progress_consumed', 0)
                    self.log(f"  (fellowship drag extended -> {to_loc})")
                    return
                if getattr(self, '_in_hunt_resolution', False) \
                        or getattr(self, '_hunt_reveal_pending', False):
                    self._hunt_reveal_pending = False
                    # AUTO-94 game-1 specimen (2026-08-17): the AUTO
                    # emitter's reveal choreography can put the marker
                    # FLIP BEFORE the walk drags (flip -> die kill ->
                    # 248 reset -> 247 hops).  The old unconditional
                    # skip-arm assumed the flip was still AHEAD; when
                    # it had already passed, the armed skip went STALE
                    # and ate the NEXT genuine flip (the post-entry
                    # fellowship_hide) - phase inversion, self-healing
                    # on the overworld (walk handlers re-sync hidden)
                    # but TERMINAL in Mordor where nothing re-syncs:
                    # engine ended revealed, roundtrip hidden, at the
                    # ring_corruption tick.  If a flip already landed
                    # in this gesture window, RETRACT its record (the
                    # declare branch's mirror rule) instead of arming.
                    _pf9 = getattr(self, '_last_fsp_flip', None)
                    _preflip9 = (_pf9 is not None
                                 and self.line_num - _pf9['line'] <= 8)
                    if _preflip9:
                        for _back in range(
                                1, min(3, len(self.action_log)) + 1):
                            _at0 = self.action_log[-_back][0]
                            if _at0 in ('fellowship_hide',
                                        'fellowship_reveal'):
                                del self.action_log[-_back]
                                self.log('  (pre-walk reveal flip '
                                         'retracted: same hunt '
                                         'reveal)')
                                break
                            if _at0 == 'fellowship_reveal_place':
                                break
                    elif self.state.fellowship.hidden:
                        # We set hidden directly; the player's marker flip
                        # for the same reveal must not toggle again
                        self._skip_next_fsp_flip = 1
                        self._fsp_skip_line = self.line_num
                    self.state.fellowship.hidden = False
                    self.log(f"Fellowship REVEALED: {from_loc} -> {to_loc}")
                    # Reveal placement drags collapse (path retraced hop
                    # by hop; the DECISION is the final region)
                    last = self.action_log[-1] if self.action_log else None
                    if (last is not None
                            and last[0] == 'fellowship_reveal_place'
                            and sanitize(last[1].get('to', '')) ==
                                sanitize(from_loc)):
                        last[1]['to'] = to_loc
                        last[1]['progress_consumed'] = \
                            last[1].get('progress_consumed', 0) + progress_consumed
                    else:
                        self._record_action('fellowship_reveal_place', {
                            'from': from_loc, 'to': to_loc,
                            'progress_consumed': progress_consumed
                        }, Side.FREE_PEOPLES)
                else:
                    self.state.fellowship.hidden = True
                    # The declare's marker dance produces a reveal+hide
                    # flip pair — both are the same declare, not decisions.
                    # The flip can also come BEFORE the walk (2023-11-03
                    # ketchuptime lines 102-103): retract the reveal
                    # record — declaring keeps the Fellowship HIDDEN
                    # (rulebook: "The Fellowship Progress counter remains
                    # Hidden side up").
                    for _back in range(1, min(3, len(self.action_log)) + 1):
                        _at0 = self.action_log[-_back][0]
                        if _at0 == 'fellowship_reveal':
                            del self.action_log[-_back]
                            self.log("  (pre-declare reveal flip "
                                     "retracted: declaring stays hidden)")
                            break
                        if _at0 == 'fellowship_declare':
                            break
                    self._skip_next_fsp_flip = 2
                    self._fsp_skip_line = self.line_num
                    self.log(f"Fellowship declared: {from_loc} -> {to_loc} "
                             f"(consumed {progress_consumed} progress)")
                    # Declare drags collapse: Hollin->Moria->Dimrill Dale
                    # is ONE declaration into Dimrill Dale
                    last = self.action_log[-1] if self.action_log else None
                    if (last is not None and last[0] == 'fellowship_declare'
                            and sanitize(last[1].get('to', '')) ==
                                sanitize(from_loc)):
                        last[1]['to'] = to_loc
                        last[1]['progress_consumed'] = \
                            last[1].get('progress_consumed', 0) + progress_consumed
                        last[1]['region'] = self._declare_region_key(to_loc)
                        last[1]['progress_before'] = \
                            last[1].get('progress_consumed', 0)
                    else:
                        self._record_action('fellowship_declare', {
                            'from': from_loc, 'to': to_loc,
                            'region': self._declare_region_key(to_loc),
                            'progress_before': progress_consumed,
                            'progress_consumed': progress_consumed
                        }, Side.FREE_PEOPLES)
            return
        pieces = army_pieces

        from_san = sanitize(from_loc)
        to_san = sanitize(to_loc)

        _mv_side = Side.SHADOW
        for p in pieces:
            if p in COMPANIONS:
                _mv_side = Side.FREE_PEOPLES
                break
            _nat, _ = get_unit_type_from_piece_id(p, self.piece_types)
            if _nat and _nat not in SHADOW_NATIONS:
                _mv_side = Side.FREE_PEOPLES
                break

        # Battle with no rolled evidence: a die was consumed, the attack
        # target was called in chat, and the DEFENDER drags units out of
        # the named region before any combat roll — the humans'
        # pre-battle-retreat convention (2023-12-06 T7: 'attack iron
        # hills' -> the lone Dwarf regular walks to Erebor).  Synthesize
        # the attack so the drag classifies as the retreat it is.
        if (self._battle_synth_region is None
                and getattr(self, '_battle_hint_region', None)
                and from_san == self._battle_hint_region
                and self._last_die_side is not None
                and not getattr(self, '_rolls_since_die', False)):
            if _mv_side != self._last_die_side \
                    and self._region_holds_enemy_of(from_san,
                                                    self._last_die_side):
                self._maybe_synth_attack(from_loc)

        # '?'-target synthesized attack (combat-card commitment with no
        # usable chat hint): the DEFENDER dragging units out of a region
        # right after it names the battle region (2023-12-06 T9:
        # 'attack osgilitah'/'north itilien' typos left the Grimbeorn
        # commitment's attack unpatched; the FP retreat from North
        # Ithilien resolves it)
        if (self._battle_synth_region == '?'
                and self._synth_attack_idx is not None):
            at0, p0, s0 = self.action_log[self._synth_attack_idx]
            if (at0 == 'army_attack' and str(p0.get('to')) == '?'
                    and _mv_side != s0
                    and self._region_holds_enemy_of(from_san, s0)):
                p0['to'] = from_loc
                if p0.get('from') in (None, '?'):
                    p0['from'] = self._find_attacker_from(from_san, s0)
                self._battle_synth_region = from_san
                self._combat_region = from_san
                self._battle_attacker_side = s0
                self.log(f"  (synth attack target patched by defender "
                         f"vacate: {from_loc})")

        # Find source army
        src_key = None
        for key in self.state.armies:
            if sanitize(key) == from_san:
                src_key = key
                break

        if not src_key or src_key not in self.state.armies:
            # t138 stale-source root (triage pass-4 4.6, R5): the
            # line's DECLARED source goes stale on re-drags/fans (the
            # client moves pids from wherever they actually are) — the
            # tracked piece_region is the source of truth.  Redirect to
            # the pieces' common tracked map region when the declared
            # one holds no army (Gileforn/Sambalam L349-352: the same
            # group dragged out of the Guide Box twice, then fanned
            # from the SECOND landing; Gandalf stranded one region
            # short for the rest of the game).
            _tzs = set()
            for p in pieces:
                _tb = self._tracked_map_base(p)
                if _tb:
                    _tzs.add(sanitize(str(_tb)))
            if len(_tzs) == 1:
                _tz1 = next(iter(_tzs))
                if _tz1 != from_san and _tz1 in board.regions and any(
                        sanitize(k) == _tz1 for k in self.state.armies):
                    self.log(f"  (stale source {from_loc} redirected "
                             f"to tracked {_tz1})")
                    return self._handle_army_move(pieces, _tz1, to_loc,
                                                  side_lock=side_lock)
            if all(p in COMPANIONS or p in MINION_PIECES
                   for p in pieces) and pieces:
                # named-only drag: proceed with an empty source so the
                # per-piece global drift-recovery still relocates them
                src_key, _ = self._find_or_create_army(from_loc)
                _named_only_src9 = True
            else:
                self.log(f"Army move: {from_loc} -> {to_loc} "
                         f"(no source army)")
                return
        else:
            _named_only_src9 = False

        src_army = self.state.armies[src_key]
        if not _named_only_src9 and not src_army.units \
                and not src_army.leaders:
            # (the named-only fallback deliberately proceeds with an
            # empty source: the per-piece drift recovery relocates the
            # named figures — returning here stranded them for good)
            return

        # Get/create destination army (use helper to avoid duplicate keys)
        to_key, dst_army = self._find_or_create_army(to_loc)

        # Count unit types being moved based on piece IDs
        moved = {'regular': 0, 'elite': 0, 'leader': 0, 'nazgul': 0}
        _prev_regions = {p: self.piece_region.get(p) for p in pieces}

        def _lock_ok(obj_nation):
            """side_lock: candidate must be on the locked side."""
            if side_lock is None:
                return True
            return (obj_nation in SHADOW_NATIONS) == \
                (side_lock == Side.SHADOW)

        for piece_id in pieces:
            self.piece_region[piece_id] = to_key

            # Companion pieces (240-246) and minion pieces (253-255) are named leaders
            named_char = COMPANIONS.get(piece_id) or MINION_PIECES.get(piece_id)
            if named_char:
                # t133 fix F5: match by ALIAS SET — the figure's current
                # name may be the flipped face (Gandalf the White,
                # Aragorn) while the pid decodes to the base name.
                _names = set(names_for_pid(piece_id)) or {named_char}
                for i, l in enumerate(src_army.leaders):
                    if hasattr(l, 'name') and l.name in _names:
                        dst_army.leaders.append(src_army.leaders.pop(i))
                        moved['leader'] += 1
                        break
                else:
                    # Not at from_loc: VASSAL logs every waypoint of a
                    # multi-hop drag with from = the ORIGINAL pick-up
                    # region ("moves group (Strider, Boromir) from Moria
                    # to Parth Celebrant" while they already stand in
                    # Dimrill Dale — 2026-01-05 kapcerz T1 lines 72-74;
                    # Strider then never reached Minas Tirith and the
                    # coronation region was wrong).  Move the NAMED
                    # figure from wherever it actually is; if it is
                    # nowhere, CREATE the named object at the
                    # destination — never move an anonymous leader in
                    # a named figure's place (t133 fix F5: the old
                    # pop(0) fallback stole setup army leaders on
                    # GtW/companion walks; ~85-90 games).
                    _found_c = False
                    for _ck in list(self.state.armies):
                        _ca = self.state.armies[_ck]
                        for i, l in enumerate(_ca.leaders):
                            if hasattr(l, 'name') and l.name in _names:
                                dst_army.leaders.append(_ca.leaders.pop(i))
                                moved['leader'] += 1
                                _found_c = True
                                self.log(f"  ({named_char} drift-recovered"
                                         f" from {_ck})")
                                break
                        if _found_c:
                            break
                    if not _found_c:
                        _cur = named_char
                        if piece_id == '240' and self._gandalf_dead:
                            _cur = 'Gandalf the White'
                        if piece_id in MINION_PIECES:
                            # Minion-typed, Shadow-nation figure —
                            # mirrors _return_named_piece (a Companion-
                            # typed Saruman failed every
                            # isinstance(Minion) check downstream)
                            import game_logic as _gl_mn9
                            _mn_nat9 = (gs.Nation.ISENGARD
                                        if _cur == 'Saruman'
                                        else gs.Nation.SAURON)
                            _nc = _gl_mn9.Minion(nation=_mn_nat9,
                                                 name=_cur)
                        else:
                            _lv9, _ld9 = _COMPANION_LEVELS.get(
                                _cur, (1, 1))
                            _nc = gs.Companion(name=_cur, level=_lv9,
                                               leadership=_ld9)
                        dst_army.leaders.append(_nc)
                        moved['leader'] += 1
                        self.log(f"  ({_cur} object re-created at "
                                 f"{to_loc}: lost to earlier drift)")
                continue

            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)
            is_piece_shadow = nation in SHADOW_NATIONS if nation else \
                (side_lock == Side.SHADOW if side_lock is not None
                 else False)
            # NATION-EXACT composition tally (2026-07-25): the client
            # names every moved piece; downstream split/advance
            # executors need per-(nation, kind) counts, not just kind
            # totals (GAME19: an ELVES regular advanced where the
            # client moved GONDOR - fungible counts, wrong nation)
            if unit_type in (UnitKind.REGULAR, UnitKind.ELITE) \
                    and nation is not None:
                _mnk9 = '%s|%s' % (
                    getattr(nation, 'value', str(nation)),
                    'elite' if unit_type == UnitKind.ELITE
                    else 'regular')
                _bn9 = moved.setdefault('_by_nation', {})
                _bn9[_mnk9] = _bn9.get(_mnk9, 0) + 1

            if unit_type == 'LEADER':
                # Prefer moving a leader of matching nation
                found = False
                for i, l in enumerate(src_army.leaders):
                    if not isinstance(l, Nazgul) and hasattr(l, 'nation') and l.nation == nation:
                        dst_army.leaders.append(src_army.leaders.pop(i))
                        moved['leader'] += 1
                        found = True
                        break
                if not found and src_army.leaders:
                    # Fallback: move any non-Nazgul leader (same side
                    # only under side_lock)
                    for i, l in enumerate(src_army.leaders):
                        if not isinstance(l, Nazgul) and _lock_ok(
                                getattr(l, 'nation', None)):
                            dst_army.leaders.append(src_army.leaders.pop(i))
                            moved['leader'] += 1
                            found = True
                            break
                if not found and side_lock is not None \
                        and nation is not None:
                    # box-exit: the client demonstrably moved this
                    # leader — create it rather than steal cross-side
                    dst_army.leaders.append(gs.Leader(nation))
                    moved['leader'] += 1
                    self.log(f"  (leader {piece_id} re-created at "
                             f"{to_loc}: side-locked box exit)")
            elif unit_type == 'NAZGUL':
                _nz_done = False
                for i, l in enumerate(src_army.leaders):
                    if isinstance(l, Nazgul):
                        dst_army.leaders.append(src_army.leaders.pop(i))
                        _nz_done = True
                        break
                if not _nz_done:
                    # Piece-tracking drift: the nazgul object lives in
                    # another army entry — pull it from its tracked
                    # region so the flight still syncs state
                    prev = _prev_regions.get(piece_id)
                    if prev and sanitize(str(prev)) != sanitize(src_key):
                        pk = next((k for k in self.state.armies
                                   if sanitize(k) == sanitize(str(prev))),
                                  None)
                        if pk is not None:
                            pa = self.state.armies[pk]
                            for i, l in enumerate(pa.leaders):
                                if isinstance(l, Nazgul):
                                    dst_army.leaders.append(
                                        pa.leaders.pop(i))
                                    _nz_done = True
                                    break
                            if not pa.units and not pa.leaders:
                                del self.state.armies[pk]
                # The drag demonstrably moved a nazgul PIECE: count it
                # even when the tracked object was lost to earlier drift
                # (the record describes the human's decision)
                moved['nazgul'] += 1
            elif unit_type == UnitKind.ELITE:
                # Prefer moving an Elite of matching nation
                found = False
                if nation:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE and u.nation == nation:
                            dst_army.units.append(src_army.units.pop(i))
                            moved['elite'] += 1
                            found = True
                            break
                if not found:
                    # Fallback: any Elite of same side, then any Elite
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            dst_army.units.append(src_army.units.pop(i))
                            moved['elite'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE and _lock_ok(u.nation):
                            dst_army.units.append(src_army.units.pop(i))
                            moved['elite'] += 1
                            found = True
                            break
                if not found and side_lock is not None:
                    # box-exit under side_lock: never steal cross-side;
                    # create the unit the client demonstrably moved
                    if nation is not None:
                        dst_army.units.append(
                            gs.Unit(nation, UnitKind.ELITE))
                        moved['elite'] += 1
                        self.log(f"  (elite {piece_id} re-created at "
                                 f"{to_loc}: side-locked box exit)")
                    found = True
                if not found and src_army.units:
                    dst_army.units.append(src_army.units.pop(0))
                    moved['regular'] += 1
            else:
                # Regular unit - prefer matching nation, then same side
                found = False
                if nation:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR and u.nation == nation:
                            dst_army.units.append(src_army.units.pop(i))
                            moved['regular'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            dst_army.units.append(src_army.units.pop(i))
                            moved['regular'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR and _lock_ok(u.nation):
                            dst_army.units.append(src_army.units.pop(i))
                            moved['regular'] += 1
                            found = True
                            break
                if not found and side_lock is not None:
                    if nation is not None:
                        dst_army.units.append(
                            gs.Unit(nation, UnitKind.REGULAR))
                        moved['regular'] += 1
                        self.log(f"  (regular {piece_id} re-created at "
                                 f"{to_loc}: side-locked box exit)")
                    found = True
                if not found and src_army.units:
                    dst_army.units.append(src_army.units.pop(0))
                    moved['elite'] += 1

        # Rulebook p.25: "Free Peoples Leaders can never be in a region
        # without combat units — if a moving Army completely vacates a
        # region, all such Leaders must follow."  (2025-05-16 Adobim T5:
        # the FoI vacate dragged 2 regulars; the unlogged leader must
        # follow or the region never empties and the Isengard follow-up
        # is misread as an attack.)
        if not src_army.units and src_army.leaders:
            for _l in list(src_army.leaders):
                # type().__name__ != 'Minion' preserves prior behavior:
                # WK/Saruman/Mouth (Companions with 'level' before the
                # typing fix) never auto-followed a full vacate here, and
                # Saruman must NEVER move (p.stays in Orthanc).
                if not isinstance(_l, Nazgul) and \
                        type(_l).__name__ != 'Minion' and \
                        not hasattr(_l, 'level') and \
                        _lock_ok(getattr(_l, 'nation', None)):
                    src_army.leaders.remove(_l)
                    dst_army.leaders.append(_l)
                    moved['leader'] += 1
                    self.log("  (leader follows full vacate, p.25)")

        # Clean up empty source army
        if not src_army.units and not src_army.leaders:
            del self.state.armies[src_key]

        # Log with detail
        details = []
        if moved['regular']:
            details.append(f"{moved['regular']}R")
        if moved['elite']:
            details.append(f"{moved['elite']}E")
        if moved['leader']:
            details.append(f"{moved['leader']}L")
        if moved['nazgul']:
            details.append(f"{moved['nazgul']}N")
        total = sum(v for k, v in moved.items()
                    if not str(k).startswith('_'))
        self.log(f"Army: {from_loc} -> {to_loc} ({'+'.join(details) if details else total})")

        # Determine side from moved pieces. Companion pieces have no nation
        # mapping, so check them explicitly — otherwise companion walks get
        # misattributed to the Shadow (default).
        move_side = Side.SHADOW
        for piece_id in pieces:
            if piece_id in COMPANIONS:
                move_side = Side.FREE_PEOPLES
                break
            nation, _ = get_unit_type_from_piece_id(piece_id, self.piece_types)
            if nation and nation not in SHADOW_NATIONS:
                move_side = Side.FREE_PEOPLES
                break
        # Companion-only piece movement is NOT an army decision. Placement
        # drags (e.g. a hunt-casualty companion walked Lorien->...->Bree)
        # previously produced chains of phantom 1L army_move rows that
        # polluted the BC data. Classify as companion_move and collapse
        # multi-hop drags into one decision (first origin -> final stop).
        comp_names = [self._pid_display_name(p) for p in pieces
                      if p in COMPANIONS]
        comp_only = (len(comp_names) == len(pieces) and len(pieces) > 0
                     and not moved['regular'] and not moved['elite']
                     and not moved['nazgul'])
        # Nazgul/minion-only drags: character-die nazgul repositioning.
        # Classify by the PIECE IDS as well as the moved counts — after
        # tracking drift the objects may be unfound at the source, but a
        # drag of nazgul pieces is still a nazgul flight (2023-12-06 T8
        # line 843: nazgul 040 Buckland -> Erebor, moved counts zero)
        _ids_all_nazgul = bool(pieces) and all(
            p in MINION_PIECES
            or get_unit_type_from_piece_id(p, self.piece_types)[1]
            == 'NAZGUL'
            for p in pieces)
        nazgul_only = (not comp_only and len(pieces) > 0
                       and not moved['regular'] and not moved['elite']
                       and (moved['nazgul'] > 0 or moved['leader'] > 0
                            or _ids_all_nazgul)
                       and move_side == Side.SHADOW)
        _battle = self._battle_synth_region
        _same_region = (_battle is not None and _battle != '?'
                        and sanitize(to_loc) == _battle)

        if comp_only:
            _prev_cm = self.action_log[-1] if self.action_log else None
            if (_prev_cm is not None
                    and _prev_cm[0] == 'companion_move'
                    and _prev_cm[2] == move_side
                    and _prev_cm[1].get('companions') == comp_names
                    and (sanitize(_prev_cm[1].get('to', '')) ==
                         sanitize(from_loc)
                         # VASSAL multi-hop drags log every waypoint with
                         # from = the ORIGINAL pick-up region (2026-01-05
                         # kapcerz lines 72-74: Moria->DD, Moria->PC,
                         # Moria->EM are ONE walk to Eastemnet)
                         or sanitize(_prev_cm[1].get('from', '')) ==
                         sanitize(from_loc))):
                # Extend the drag: same piece(s) continuing — update target
                self.action_log[-1][1]['to'] = to_loc
                self.log(f"  (companion drag extended -> {to_loc})")
            else:
                cm_params = {
                    'from': from_loc, 'to': to_loc,
                    'companions': comp_names,
                }
                # A Hobbit walked onto the map right after a random hunt
                # casualty drew him = the hunt-casualty PLACEMENT decision
                # (user-verified rule: hunt-drawn Hobbits are placed as if
                # separated, not eliminated)
                if len(comp_names) == 1 and \
                        comp_names[0] in ('Meriadoc', 'Peregrin'):
                    for at2, p2, _sd2 in reversed(self.action_log[-3:]):
                        if at2 == 'hunt_sacrifice_random' and \
                                sanitize(str(p2.get('companion', ''))) == \
                                sanitize(comp_names[0]):
                            cm_params['hunt_placement'] = True
                            break
                self._record_action('companion_move', cm_params, move_side)
        elif nazgul_only:
            # Chain-collapse multi-hop nazgul/minion drags; one die moves
            # ALL nazgul, so subsequent same-window nazgul moves are
            # continuations of one decision
            last = self.action_log[-1] if self.action_log else None
            # MINION PLACEMENT CORRECTION: the minion just mustered is
            # still being dragged in the SAME gesture (no die consumed
            # since the muster, within 3 lines) — the muster's TRUE
            # region is the final drop, not a separate move decision
            # (2023-11-03 ketchuptime T8 L783-784: the Mouth staged at
            # Minas Morgul then dropped at Dol Guldur; the phantom
            # die-less nazgul_move matched a die-BEARING engine flight
            # and stole the Character die the Dol Amroth attack needed).
            _mm = getattr(self, '_minion_muster_ctx', None)
            _mn = sorted(MINION_PIECES[p] for p in pieces
                         if p in MINION_PIECES)
            if (_mm is not None and _mn and len(_mn) == len(pieces)
                    and last is not None and last[0] == 'muster'
                    and _mm['idx'] == len(self.action_log) - 1
                    and _mm['minions'] == _mn
                    and self.line_num - _mm['line'] <= 3
                    and getattr(self, '_die_consume_seq', 0)
                    == _mm['die_seq']
                    and sanitize(str(last[1].get('region', '')))
                    == sanitize(from_loc)):
                last[1]['region'] = to_loc
                self.log(f"  (minion placement corrected -> {to_loc})")
            elif (last is not None and last[0] == 'nazgul_move'
                    and last[2] == move_side
                    and sanitize(last[1].get('to', '')) == sanitize(from_loc)
                    and last[1].get('_pieces') == sorted(pieces)):
                last[1]['to'] = to_loc
                self.log(f"  (nazgul drag extended -> {to_loc})")
            else:
                nz_params = {'from': from_loc, 'to': to_loc, 'moved': moved,
                             '_pieces': sorted(pieces)}
                if (self._pending_die[move_side] is None
                        and self._card_window_side != move_side
                        and last is not None and last[0] == 'nazgul_move'
                        and last[2] == move_side):
                    nz_params['continuation'] = True
                self._record_action('nazgul_move', nz_params, move_side)
        elif _same_region and move_side == self._battle_attacker_side:
            # Attacker moving INTO the battle region = post-battle advance
            # (a decision — how many units move in — but not a die action)
            # fix 91 (jmelder T6): DRAG-THROUGH fold — the human drags
            # the advancing stack through a waypoint region in one
            # gesture (Westemnet -> Druadan -> Folde, raw L514/515).
            # The immediately-preceding record then is a same-side
            # army_move whose 'to' equals this advance's 'from' on an
            # adjacent line: it is the SAME drag, not a real move.
            # Fold: drop the staging record, advance from the CHAIN
            # ORIGIN (which also stops the source-patcher below from
            # overriding a chat-hinted attack source with the
            # waypoint).
            # the waypoint drop may also fabricate a phantom
            # combat_start when the waypoint holds enemy units
            # (Druadan's FP regular) — pop it together with the leg
            _phantom_cs = None
            _lastA = self.action_log[-1] if self.action_log else None
            if (_lastA is not None and _lastA[0] == 'combat_start'
                    and sanitize(str(_lastA[1].get('region', ''))) ==
                    sanitize(str(from_loc))
                    and len(self.action_log) >= 2):
                _phantom_cs = _lastA
                _lastA = self.action_log[-2]
            _adv_pieces = set(pieces or [])
            _leg_pieces = set((_lastA[1].get('_pieces') or [])
                              if _lastA is not None else [])
            if (_lastA is not None and _lastA[0] == 'army_move'
                    and _lastA[2] == move_side
                    and sanitize(str(_lastA[1].get('to', ''))) ==
                    sanitize(str(from_loc))
                    and _lastA[1].get('die_used') is None
                    and self.line_num - int(
                        _lastA[1].get('_rec_line', self.line_num)) <= 2
                    # SAME physical pieces both legs = one drag; real
                    # repositioning legs move different units (game 5
                    # T10 post-battle shuffle was wrongly folded by v1)
                    and _adv_pieces and _adv_pieces == _leg_pieces):
                from_loc = _lastA[1].get('from', from_loc)
                if _phantom_cs is not None:
                    self.action_log.pop()   # the phantom combat_start
                self.action_log.pop()       # the staging leg
                self.log(f"  (advance drag-through folded: origin "
                         f"{from_loc})")
            self._record_action('battle_advance', {
                'from': from_loc, 'to': to_loc, 'moved': moved
            }, move_side)
            self.log(f"  (battle advance into {to_loc})")
            # The advance reveals the attacker's source region: patch the
            # synthesized attack's 'from' (card-driven attacks like
            # Corsairs of Umbar have no adjacent army at synthesis time —
            # the sea move is only logged as this advance).  The advance
            # is the PHYSICAL drag: it also corrects a mis-GUESSED
            # source (2023-12-06 T10: the Dale attack was resolved to
            # Northern Rhovanion, but the advance came from Erebor).
            if self._synth_attack_idx is not None:
                at0, p0, _s0 = self.action_log[self._synth_attack_idx]
                if at0 == 'army_attack' \
                        and sanitize(str(p0.get('to', ''))) == \
                        sanitize(to_loc) \
                        and sanitize(str(p0.get('from', '?'))) != \
                        sanitize(from_loc) \
                        and sanitize(str(p0.get('from', '?'))) != \
                        sanitize(to_loc):
                    old_from = p0.get('from')
                    p0['from'] = from_loc
                    self.log(f"  (attack source patched: {old_from} -> "
                             f"{from_loc})")
        # Detect retreat: the DEFENDER moving away from the combat region
        # DURING an active battle window (the attacker repositioning
        # afterwards is an ordinary move)
        elif (self._combat_region and
                self._battle_attacker_side is not None and
                sanitize(from_loc) == sanitize(self._combat_region) and
                sanitize(to_loc) != sanitize(self._combat_region) and
                move_side != self._battle_attacker_side):
            # Multi-hop retreat drags collapse into one decision
            last = self.action_log[-1] if self.action_log else None
            if (last is not None and last[0] == 'combat_retreat'
                    and last[2] == move_side
                    and sanitize(last[1].get('to', '')) == sanitize(from_loc)):
                last[1]['to'] = to_loc
                self.log(f"  (retreat drag extended -> {to_loc})")
            else:
                # carry _pieces so a multi-hop retreat drag merges into
                # ONE retreat (2023-02-03 Wirbelwind T4: Dale -> Iron
                # Hills -> Erebor; unmerged, the second hop matched a
                # died army_move and burned the die the T4 hide needed)
                self._record_action('combat_retreat', {
                    'from': from_loc, 'to': to_loc, 'moved': moved,
                    '_pieces': sorted(pieces)
                }, move_side)
                self.log(f"  (retreat from combat at {self._combat_region})")
        else:
            # Drag INTO an enemy-occupied map region = the ATTACK
            # declaration itself (2023-08-13 HombreLobo T1: Old Forest
            # Road -> Woodland Realm vs the Elven garrison; no chat
            # callout, so the roll-side synth never fired and the battle
            # was misparsed as an army_move).  Reinforcing one's own
            # siege stays a move (besieged guard); an already-open battle
            # window means the synth owns the attack.
            _to_san = sanitize(to_loc)
            if (is_map_region(to_loc)
                    and self._region_holds_enemy_of(_to_san, move_side)):
                # An attack on this region was already recorded (the live
                # battle window may have been RESET by an interleaved die
                # consumption, and the defender may ALREADY have retreated
                # into the stronghold box making the region 'besieged' —
                # 2023-08-13 HombreLobo T1: attack at line 102, siege
                # retreat at 102, move-in drag at 109): the drag is
                # battle noise.  Checked BEFORE the besieged guard.
                # 130 (2026-07-25 board-is-set T8 L603): the plain
                # 10-record lookback swallowed a GENUINE A-die siege
                # reinforcement (Osgiliath -> besieged Minas Tirith) as
                # "battle noise" of the PREVIOUS assault, which had
                # fully resolved (rolls thrown, its die killed, another
                # die action complete) - the 5 Sauron regulars never
                # moved in-engine and every later Osgiliath/Harondor
                # decision cascaded.  A recent attack still OWNS a
                # move-in only while it is the live action: no rolls
                # since the attack (pre-battle siege-retreat class,
                # HombreLobo T1) OR no new die staged since it
                # (same-window drag).  Both present = the battle is
                # over; the drag is a fresh decision.
                _recent_atk = False
                for at0, p0, s0 in self.action_log[-10:]:
                    if (at0 == 'army_attack' and s0 == move_side
                            and sanitize(str(p0.get('to', '')))
                            == _to_san):
                        _al0 = p0.get('_atk_line')
                        if _al0 is None:
                            _recent_atk = True
                            continue
                        _rolled0 = getattr(self, '_last_roll_line', 0) \
                            > int(_al0)
                        _staged0 = getattr(self, '_battle_window_line',
                                           0) > int(_al0)
                        if not (_rolled0 and _staged0):
                            _recent_atk = True
                _besieged = _to_san in getattr(
                    self.state, 'besieged_strongholds', set())
                if _recent_atk:
                    # The attacker's physical move-in carries the die that
                    # actually paid for the attack.  When a defender
                    # PRE-EMPTIVELY retreats into siege, the attack is
                    # synthesized (_maybe_synth_attack) BEFORE the real die
                    # is known, taking whatever die is pending -- which may
                    # be a MIS-PICKED die the human then BOUNCED back to the
                    # pool (Controls.281) and replaced (2023-08-13
                    # HombreLobo T1: GJB picks the Muster/Army die, the
                    # Elves retreat into Woodland Realm's stronghold, GJB
                    # returns the Muster/Army die, pays the attack with a
                    # Character die, then musters Saruman with the freed
                    # Muster/Army die).  If a LIVE die is pending on the
                    # physical move-in and it differs from the synthesized
                    # attack's die_used, re-point the attack to the live die
                    # so the stale provisional pick cannot steal a die-class
                    # a later decision (the minion muster) needs.  The
                    # move-in is battle noise -- the die belongs to the
                    # attack, so consume the pending here (no second record).
                    _live_du = self._pending_die[move_side]
                    if _live_du is not None:
                        for _bi in range(
                                len(self.action_log) - 1,
                                max(-1, len(self.action_log) - 11), -1):
                            _ba, _bp, _bs = self.action_log[_bi]
                            if (_ba == 'army_attack' and _bs == move_side
                                    and sanitize(str(_bp.get('to', '')))
                                    == _to_san):
                                if _bp.get('die_used') != _live_du:
                                    self.log(
                                        f"  (attack die corrected "
                                        f"{_bp.get('die_used')} -> "
                                        f"{_live_du} from physical move-in)")
                                    _bp['die_used'] = _live_du
                                self._pending_die[move_side] = None
                                break
                    self._record_action('army_move', {
                        'from': from_loc, 'to': to_loc, 'moved': moved,
                        '_pieces': sorted(pieces), 'auto': True}, move_side)
                    self.log(f"  (attacker move-in at {to_loc}: battle "
                             f"noise, recent attack)")
                    self._update_region_control(from_loc)
                    self._update_region_control(to_loc)
                    return
                if not _besieged and self._battle_synth_region is None:
                    _atk = {'from': from_loc, 'to': to_loc, 'moved': moved,
                            '_pieces': sorted(pieces),
                            'die_used': self._pending_die[move_side] or None,
                            '_atk_line': getattr(self, 'line_num', 0),
                            '_decl_line': getattr(self,
                                                  '_battle_window_line', 0)}
                    # AI-BRIDGE grammar (t123): a die-less drag-attack
                    # inside the mover's own card window is the CARD's
                    # attack (Corsairs of Umbar landing at Dol Amroth,
                    # wotr2026_7_17_11_26 L393), not a standalone die
                    # decision — same rule the roll-side synth applies
                    # in _maybe_synth_attack.  Gated: ungated it
                    # re-tagged corpus drag-attacks and shifted the
                    # campaign canary (StellaRossa 132->130 et al).
                    if self._pending_die[move_side] is None \
                            and self._card_window_side == move_side \
                            and getattr(self, '_ai_bridge_log', False):
                        _atk['card_effect'] = True
                    self.action_log.append(('army_attack', _atk, move_side))
                    if self._pending_die[move_side] is not None:
                        self._pending_die[move_side] = None
                    self._synth_attack_idx = len(self.action_log) - 1
                    self._battle_synth_region = _to_san
                    self._battle_attacker_side = move_side
                    self._combat_region = _to_san
                    self._auto_tag_defender_political(_to_san, move_side)
                    self.log(f"  (drag into enemy-held {to_loc} recorded "
                             f"as ATTACK)")
                    self._update_region_control(from_loc)
                    self._update_region_control(to_loc)
                    return
                if (not _besieged
                        and self._battle_synth_region in (_to_san, '?')
                        and self._battle_attacker_side in (None, move_side)):
                    # Attacker physically moving onto the battle region it
                    # is already attacking (siege co-location convention,
                    # 2023-08-13 HombreLobo T1 line 109 after the line-102
                    # synth attack) — battle noise, not a decision
                    self._record_action('army_move', {
                        'from': from_loc, 'to': to_loc, 'moved': moved,
                        '_pieces': sorted(pieces), 'auto': True}, move_side)
                    self.log(f"  (attacker move-in at {to_loc}: battle "
                             f"noise)")
                    self._update_region_control(from_loc)
                    self._update_region_control(to_loc)
                    return
                # besieged without a recent attack: reinforcing one's own
                # siege — falls through to the normal move record
            # Multi-hop drags of the same pieces collapse (e.g. a retreat
            # walked two regions across log lines — possibly interleaved
            # with the opponent's actions). Look back a few entries.
            merged = False
            if self._pending_die[move_side] is None:
                for back in range(1, min(4, len(self.action_log)) + 1):
                    prev = self.action_log[-back]
                    if prev[2] != move_side:
                        continue  # opponent's action interleaved — keep looking
                    if (prev[0] in ('army_move', 'combat_retreat')
                            and sanitize(prev[1].get('to', '')) ==
                                sanitize(from_loc)
                            and prev[1].get('_pieces') == sorted(pieces)):
                        # O.2 RESOLVED (user ruling 2026-07-07): an A->B,
                        # B->C chain of SEPARATE drag gestures (a true
                        # one-gesture multi-hop logs from=pick-up region
                        # on EVERY waypoint) is TWO army moves under one
                        # army-class die whenever the intermediate region
                        # still holds the mover's units — units are
                        # fungible, so the legal reading (an army die
                        # moves up to two DIFFERENT armies) wins.
                        # 2023-08-13 HombreLobo T4 L412-413: OFR->Dale->
                        # Erebor with 2 other Sauron R at Dale; OFR and
                        # Erebor are NOT adjacent (map untouched).
                        # Retreat chains still collapse (one decision).
                        if prev[0] == 'army_move':
                            _mv_sh = move_side == Side.SHADOW
                            _src_holds = any(
                                (u.nation in SHADOW_NATIONS) == _mv_sh
                                for k9, a9 in self.state.armies.items()
                                if sanitize(k9) == sanitize(from_loc)
                                for u in a9.units)
                            if _src_holds:
                                self.log(f"  (no merge: {from_loc} still "
                                         f"holds mover units — second "
                                         f"army of the pair)")
                                break
                        prev[1]['to'] = to_loc
                        self.log(f"  (move drag extended -> {to_loc})")
                        merged = True
                    break  # only the nearest same-side action counts
            if not merged:
                self._record_action('army_move', {
                    'from': from_loc, 'to': to_loc, 'moved': moved,
                    '_pieces': sorted(pieces)}, move_side)

        # Update region control for both source and destination
        self._update_region_control(from_loc)
        self._update_region_control(to_loc)

        # Combat detection: check if destination has both sides
        shadow_minion_names = {'Witch King', 'Saruman', 'Mouth of Sauron'}
        if to_key in self.state.armies:
            dst = self.state.armies[to_key]
            has_fp = any(u.nation not in SHADOW_NATIONS for u in dst.units) or \
                     any(not isinstance(l, Nazgul) and getattr(l, 'name', '') not in shadow_minion_names
                         and hasattr(l, 'nation') and l.nation not in SHADOW_NATIONS for l in dst.leaders)
            has_sh = any(u.nation in SHADOW_NATIONS for u in dst.units) or \
                     any(isinstance(l, Nazgul) or getattr(l, 'name', '') in shadow_minion_names
                         or (hasattr(l, 'nation') and l.nation in SHADOW_NATIONS) for l in dst.leaders)
            # A BESIEGED stronghold is mixed by construction (garrison +
            # besiegers co-located): arrivals joining the siege are NOT
            # new battles (2025-06-19 StellaRossa T6 L509-512: three
            # nazgul converging on besieged Lorien emitted THREE
            # combat_start records — one matched a second die-bearing
            # attack and stole the MUSTER die the T6 pair-move needed).
            # Same-window dedup: one combat_start per battle window.
            _to_san2 = sanitize(to_key)
            if has_fp and has_sh \
                    and _to_san2 not in getattr(
                        self.state, 'besieged_strongholds', set()) \
                    and self._battle_synth_region != _to_san2:
                self._combat_region = to_key
                self._record_action('combat_start', {'region': to_loc, 'attacker': move_side.name}, move_side)

    def _kill_named_piece(self, pid: str, zone, prefer_san=None) -> bool:
        """Eliminate a named companion/minion object BY IDENTITY (t133
        fix F10, rulebook p.29: leaders including Characters die with
        their army).  Pops the alias-matched object from the preferred
        army if named there, else from any army (staged/box-tracked
        figures); updates minion in_play flags; zone-tracks the piece
        as dead.  Returns True when an object was actually removed."""
        names = set(names_for_pid(pid))
        popped = None
        keys = list(self.state.armies)
        if prefer_san:
            keys.sort(key=lambda k: 0 if sanitize(k) == prefer_san else 1)
        for k in keys:
            a = self.state.armies[k]
            for i, l in enumerate(a.leaders):
                if getattr(l, 'name', None) in names:
                    popped = a.leaders.pop(i)
                    if not a.units and not a.leaders:
                        del self.state.armies[k]
                    break
            if popped is not None:
                break
        nm = MINION_PIECES.get(pid)
        if nm == 'Saruman' and self.state.saruman:
            self.state.saruman.in_play = False
        elif nm == 'Witch King' and self.state.witch_king:
            self.state.witch_king.in_play = False
        elif nm == 'Mouth of Sauron' and self.state.mouth_of_sauron:
            self.state.mouth_of_sauron.in_play = False
        # A killed Gandalf the White must drop the in_play flag too
        # (pid 240 aliases Grey/White; nothing else ever clears it)
        if (getattr(popped, 'name', None) == 'Gandalf the White'
                and self.state.gandalf_the_white):
            self.state.gandalf_the_white.in_play = False
        self._set_piece_zone(pid, zone, dead=True)
        if popped is not None:
            self.log(f"Named piece killed: "
                     f"{getattr(popped, 'name', pid)} -> {zone}")
        return popped is not None

    def _return_named_piece(self, pid: str, to_key: str, to_army,
                            from_loc: str, to_zone: str):
        """A named companion/minion pid leaving an off-board zone for a
        stronghold BOX (t133 fix F12): a character appearance/revival,
        NEVER a unit refund — the old dead-pile-exit path credited the
        region an anonymous unit when GtW was placed straight into a
        besieged Elven stronghold box (GtW card p.24: '...or in an
        Elven Stronghold controlled by the Free Peoples (even if it is
        under siege)')."""
        comp_name = COMPANIONS.get(pid) or MINION_PIECES.get(pid)
        if pid == '240' and self._gandalf_dead:
            comp = gs.Companion(name='Gandalf the White', level=3,
                                leadership=1)
            to_army.leaders.append(comp)
            self.piece_region[pid] = to_zone
            if self.state.gandalf_the_white:
                self.state.gandalf_the_white.in_play = True
            self._record_action('transform_gandalf',
                                {'region': to_key}, Side.FREE_PEOPLES)
            self.log(f"Gandalf the White appears -> {to_zone}")
            return
        # Relocate the alias-matched object if it exists somewhere;
        # otherwise create it fresh.
        names = set(names_for_pid(pid))
        for k in list(self.state.armies):
            a = self.state.armies[k]
            for i, l in enumerate(a.leaders):
                if getattr(l, 'name', None) in names:
                    to_army.leaders.append(a.leaders.pop(i))
                    if not a.units and not a.leaders \
                            and sanitize(k) != sanitize(to_key):
                        del self.state.armies[k]
                    self.piece_region[pid] = to_zone
                    self.log(f"{comp_name} relocated -> {to_zone}")
                    return
        if pid in MINION_PIECES:
            # FIX (minion typing): Minion-typed, Shadow-nation figure
            import game_logic as _gl_mn
            _mn_nat = (gs.Nation.ISENGARD if comp_name == 'Saruman'
                       else gs.Nation.SAURON)
            comp = _gl_mn.Minion(nation=_mn_nat, name=comp_name)
            if comp_name == 'Saruman' and self.state.saruman:
                self.state.saruman.in_play = True
            elif comp_name == 'Witch King' and self.state.witch_king:
                self.state.witch_king.in_play = True
            elif comp_name == 'Mouth of Sauron' \
                    and self.state.mouth_of_sauron:
                self.state.mouth_of_sauron.in_play = True
        else:
            # official card values (one implementation: the sibling
            # revival paths use the same table)
            _lv9, _ld9 = _COMPANION_LEVELS.get(comp_name, (1, 1))
            comp = gs.Companion(name=comp_name, level=_lv9,
                                leadership=_ld9)
        to_army.leaders.append(comp)
        self.piece_region[pid] = to_zone
        self.log(f"Companion returns: {comp_name} -> {to_zone}")

    # ---- t133 ruling 1.162 (F11): pid-level idempotency helpers -----
    def _pid_zone_dead(self, zone) -> bool:
        """True when a TRACKED zone means the piece is already off the
        board as a casualty ('identical pieces are identical': a
        re-applied kill line is a client no-op)."""
        if zone is None:
            return False
        zs = sanitize(str(zone))
        return zs in ('dead', 'used') or self._zone_is_dead(zone)

    def _tracked_map_base(self, pid) -> Optional[str]:
        """Sanitized base region of the pid's tracked zone when that
        zone is a map region or a stronghold box; else None."""
        z = self.piece_region.get(pid)
        if z is None:
            return None
        zs = str(z)
        b = stronghold_base_region(zs)
        if b is not None:
            return sanitize(b)
        if is_map_region(zs):
            return sanitize(zs)
        return None

    def _redirect_tracked_musters(self, pieces: List[str], to_base: str,
                                  to_zone: str) -> List[str]:
        """USER RULING 1.162 (t133, F11 durable tier): a muster-grammar
        drag whose pid is ALREADY TRACKED on the map cannot re-muster —
        the client (pid semantics) moves the piece from wherever it is,
        so a same-region re-drag is a display no-op and a stale-source
        re-drag is a MOVE from the tracked region
        (2024-09-05 igforce77/Columbo L140-142: elite 094 dragged from
        the stale 'FP Dead' source three times = +3 phantom musters
        under count semantics).  Returns the pids still eligible for a
        fresh muster."""
        fresh = []
        by_src = {}
        to_san = sanitize(to_base)
        for p in pieces:
            # markers/dice/named figures are never redirected units
            if p in FELLOWSHIP_MARKERS or p in COMPANIONS \
                    or p in MINION_PIECES or self._is_die_piece(p):
                fresh.append(p)
                continue
            base = self._tracked_map_base(p)
            if base is None:
                fresh.append(p)
            elif base == to_san:
                self.piece_region[p] = to_zone
                self.log(f"Re-drag no-op (ruling 1.162): piece {p} "
                         f"already at {to_base}")
            else:
                by_src.setdefault(base, []).append(p)
        for src, ps in by_src.items():
            self.log(f"Stale-source re-drag (ruling 1.162): {ps} "
                     f"move {src} -> {to_base}")
            self._handle_army_move(ps, src, to_base)
            for p in ps:
                self.piece_region[p] = to_zone
        return fresh

    def _remove_pieces_from_armies(self, pieces: List[str], from_loc: str,
                                   to_loc: str, mark_cause: str = None):
        """Remove pieces from their army when they leave the board (casualties, kills, card effects)."""
        # Filter out non-army pieces (companions, fellowship markers, tokens, cards, dice)
        # Note: minion pieces (253-255) are NOT filtered - they need removal + in_play update
        army_pieces = []
        _comp_kill_ps = []
        for p in pieces:
            if p in FELLOWSHIP_MARKERS:
                continue
            if p in COMPANIONS:
                # t133 fix F10: companions leaving the board resolve BY
                # IDENTITY below (the old filter dropped them entirely,
                # so a companion staged out of a stronghold box was
                # never eliminated; rulebook p.29)
                _comp_kill_ps.append(p)
                continue
            if p == TOKEN_POLITICAL or p == TOKEN_CARD_DRAW:
                continue
            if p in CONTROL_MARKER_PIECES or p in VP_MARKER_PIECES:
                # marker kill = control release, never a unit casualty
                side = (Side.FREE_PEOPLES if p in FP_CONTROL_MARKERS
                        else Side.SHADOW)
                reg = self._control_marker_at.pop(p, None)
                if reg:
                    self.state.enemy_controlled.setdefault(
                        side, set()).discard(reg)
                    # RB-08 (2026-07-22): control follows OCCUPATION, not
                    # the marker alone (amendment 1.55a).  If the capturing
                    # side STILL solely occupies the settlement, killing the
                    # marker does not release it - the engine keeps control
                    # (2023-11-09 westemnet T6 place-then-remove misplay).
                    # _update_region_control re-adds only when the region is
                    # a settlement solely held by one side, so a genuine
                    # reclaim (enemy army present) still releases correctly.
                    self._update_region_control(reg)
                    self.log(f"Control marker killed: released {reg}")
                continue
            # Skip card piece IDs (106-209) and dice (099-105, 210-213, 358-362)
            try:
                pid = int(p)
                if 99 <= pid <= 105 or 106 <= pid <= 209 or 210 <= pid <= 213 or 358 <= pid <= 362:
                    continue
            except ValueError:
                continue
            # ruling 1.162 (F11): a piece already tracked dead cannot
            # leave the board again — re-applied drag-kills are client
            # no-ops (count semantics over-killed the army)
            if self._pid_zone_dead(self.piece_region.get(p)):
                self.log(f"Re-kill no-op (ruling 1.162): piece {p} "
                         f"already dead")
                continue
            army_pieces.append(p)

        _rm_from_san = sanitize(from_loc)
        for p in _comp_kill_ps:
            self._kill_named_piece(p, to_loc, prefer_san=_rm_from_san)

        if not army_pieces:
            return

        from_san = sanitize(from_loc)

        # Find source army
        src_key = None
        for key in self.state.armies:
            if sanitize(key) == from_san:
                src_key = key
                break

        if not src_key or src_key not in self.state.armies:
            return

        src_army = self.state.armies[src_key]

        removed = {'regular': 0, 'elite': 0, 'leader': 0, 'nazgul': 0}
        _to_death = not any(w in (to_loc or '')
                            for w in ('Reinf', 'reinforcements'))
        for piece_id in army_pieces:
            # t125 fix 1: piece_region + FP dead-pool bookkeeping in one
            # place (the old inline fp_casualties increment is subsumed;
            # _to_death keeps its semantics — any non-Reinf exit counts,
            # incl. 'Used'/card-play parking).
            self._set_piece_zone(piece_id, to_loc, dead=_to_death)

            # Handle minion pieces being killed
            minion_name = MINION_PIECES.get(piece_id)
            if minion_name:
                for i, l in enumerate(src_army.leaders):
                    if hasattr(l, 'name') and l.name == minion_name:
                        src_army.leaders.pop(i)
                        removed['leader'] += 1
                        break
                if minion_name == 'Saruman' and self.state.saruman:
                    self.state.saruman.in_play = False
                elif minion_name == 'Witch King' and self.state.witch_king:
                    self.state.witch_king.in_play = False
                elif minion_name == 'Mouth of Sauron' and self.state.mouth_of_sauron:
                    self.state.mouth_of_sauron.in_play = False
                self.log(f"MINION KILLED: {minion_name}")
                continue

            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)
            is_piece_shadow = nation in SHADOW_NATIONS if nation else False
            # NATION-EXACT casualty tally (2026-07-25, mirrors the
            # moves-loop moved['_by_nation']): the kill lines NAME the
            # nations; nation-blind casualty specs killed SAURON where
            # the client killed ISENGARD (GAME19 T8 Minas Morgul)
            if unit_type in (UnitKind.REGULAR, UnitKind.ELITE) \
                    and nation is not None:
                _rnk9 = '%s|%s' % (
                    getattr(nation, 'value', str(nation)),
                    'elite' if unit_type == UnitKind.ELITE
                    else 'regular')
                _rbn9 = removed.setdefault('_by_nation', {})
                _rbn9[_rnk9] = _rbn9.get(_rnk9, 0) + 1

            if unit_type == 'LEADER':
                # Prefer removing a leader of matching nation
                found = False
                for i, l in enumerate(src_army.leaders):
                    if not isinstance(l, Nazgul) and hasattr(l, 'nation') and l.nation == nation:
                        src_army.leaders.pop(i)
                        removed['leader'] += 1
                        found = True
                        break
                if not found:
                    for i, l in enumerate(src_army.leaders):
                        if not isinstance(l, Nazgul):
                            src_army.leaders.pop(i)
                            removed['leader'] += 1
                            break
            elif unit_type == 'NAZGUL':
                for i, l in enumerate(src_army.leaders):
                    if isinstance(l, Nazgul):
                        src_army.leaders.pop(i)
                        removed['nazgul'] += 1
                        break
            elif unit_type == UnitKind.ELITE:
                # Prefer removing an Elite of matching nation, then same side
                found = False
                if nation:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE and u.nation == nation:
                            src_army.units.pop(i)
                            removed['elite'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            src_army.units.pop(i)
                            removed['elite'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.ELITE:
                            src_army.units.pop(i)
                            removed['elite'] += 1
                            found = True
                            break
                # Final fallback: remove any same-side unit
                if not found and src_army.units:
                    for i, u in enumerate(src_army.units):
                        if (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            src_army.units.pop(i)
                            if u.kind == UnitKind.ELITE:
                                removed['elite'] += 1
                            else:
                                removed['regular'] += 1
                            break
            else:
                # Regular or unknown - prefer matching nation, then same side
                found = False
                if nation:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR and u.nation == nation:
                            src_army.units.pop(i)
                            removed['regular'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            src_army.units.pop(i)
                            removed['regular'] += 1
                            found = True
                            break
                if not found:
                    for i, u in enumerate(src_army.units):
                        if u.kind == UnitKind.REGULAR:
                            src_army.units.pop(i)
                            removed['regular'] += 1
                            found = True
                            break
                if not found and src_army.units:
                    u = src_army.units.pop(0)
                    if u.kind == UnitKind.ELITE:
                        removed['elite'] += 1
                    else:
                        removed['regular'] += 1

        # Clean up empty army
        if not src_army.units and not src_army.leaders:
            del self.state.armies[src_key]

        # Log
        details = []
        if removed['regular']:
            details.append(f"{removed['regular']}R")
        if removed['elite']:
            details.append(f"{removed['elite']}E")
        if removed['leader']:
            details.append(f"{removed['leader']}L")
        if removed['nazgul']:
            details.append(f"{removed['nazgul']}N")
        self.log(f"Casualties: {'+'.join(details) if details else len(army_pieces)} from {from_loc} -> {to_loc}")

        # RC4 sequel: per-round casualty MARKS - inside a battle window
        # the log's removals per side are ground truth for that round's
        # applied hits (regular=1, elite killed outright=2; downgrades
        # marked separately at the swap handler).  Consumed by the
        # closed loop's FINAL_HITS_HOOK.  v3: CAUSE tagged at emission.
        if self._battle_synth_region is not None and                 (removed['regular'] or removed['elite']):
            # PER-PIECE side resolution (casualty-forcing failure mode
            # #5's precision limit): a mixed-side multi-piece drag used
            # to label ALL pieces with army_pieces[0]'s nation — resolve
            # each piece via the authoritative table and emit one mark
            # PER SIDE with per-side composition.
            _per = {}
            for _pc9 in army_pieces:
                _n9, _k9 = get_unit_type_from_piece_id(
                    _pc9, self.piece_types)
                if _n9 is None or _k9 not in (UnitKind.REGULAR,
                                              UnitKind.ELITE):
                    continue
                _sd9 = (Side.SHADOW if _n9 in SHADOW_NATIONS
                        else Side.FREE_PEOPLES)
                _d9 = _per.setdefault(_sd9, {'kill_regular': 0,
                                             'kill_elite': 0,
                                             'downgrade': 0})
                _d9['kill_regular' if _k9 == UnitKind.REGULAR
                    else 'kill_elite'] += 1
            if not _per:
                _mk_nat, _ = get_unit_type_from_piece_id(
                    army_pieces[0], self.piece_types)
                _mk_side = (Side.SHADOW if _mk_nat in SHADOW_NATIONS
                            else Side.FREE_PEOPLES) if _mk_nat else None
                _per = {_mk_side: {'kill_regular': removed['regular'],
                                   'kill_elite': removed['elite'],
                                   'downgrade': 0}}
            _rbn0 = removed.get('_by_nation') or {}
            _shv0 = {str(getattr(n9, 'value', n9))
                     for n9 in SHADOW_NATIONS}
            for _sd9, _sp9 in _per.items():
                if _rbn0 and '_by_nation' not in _sp9:
                    # NATION-EXACT spec (2026-07-25): partition the
                    # named kills to this mark's side
                    _sp9['_by_nation'] = {
                        k9: v9 for k9, v9 in _rbn0.items()
                        if ((k9.split('|')[0] in _shv0)
                            == (_sd9 == Side.SHADOW))}
                self._emit_casualty_mark(
                    _sd9, from_san,
                    _sp9['kill_regular'] + 2 * _sp9['kill_elite'],
                    mark_cause or self._mark_cause(from_san, to_loc),
                    spec=_sp9)

        # Update region control after casualties
        self._update_region_control(from_loc)

    def _parse_readable_muster_info(self) -> Optional[Tuple[str, str]]:
        """Extract nation and unit type from the last % readable line."""
        line = self._last_readable_line
        if not line:
            return None
        # Match patterns like: ([2]Isengard Elite), (Rohan Regular), (Gondor Leader)
        m = re.search(r'\[?\d*\]?(\w[\w\s]*?)\s+(Regular|Elite|Leader)\b', line)
        if m:
            nation_str = m.group(1).strip()
            unit_str = m.group(2)
            return (nation_str, unit_str)
        return None

    def _handle_muster(self, pieces: List[str], to_loc: str, is_shadow: bool = False):
        """Muster new units to a region."""
        # Action-die, card and hunt-tile pieces are never units (defense
        # in depth: the _handle_move filter covers Scribe drags; card/
        # kill paths call this directly).  Ids 106-239 are cards
        # (106-209), FP dice (210-213) and hunt tiles (214-239).
        def _non_unit(p):
            if self._is_die_piece(p):
                return True
            try:
                return 106 <= int(p) <= 239
            except (ValueError, TypeError):
                return False
        pieces = [p for p in pieces if not _non_unit(p)]
        if not pieces:
            return
        to_san = sanitize(to_loc)

        # Use helper to avoid duplicate army keys with different casing
        to_key, army = self._find_or_create_army(to_loc)

        # Determine nation from reinforcement source (FP vs SA) and region
        region_meta = board.regions.get(to_san, {})
        parent_id = region_meta.get("parent_id")
        faction = board.fid_to_name.get(parent_id, "").lower() if parent_id else ""

        # For Shadow reinforcements, try to match Shadow factions
        if is_shadow:
            shadow_nation_map = {
                'isengard': Nation.ISENGARD,
                'sauron': Nation.SAURON,
                'southrons and easterlings':
                    Nation.SOUTHRON_EASTERLINGS,
                'southrons': Nation.SOUTHRON_EASTERLINGS,
                'easterlings': Nation.SOUTHRON_EASTERLINGS,
            }
            default_nation = shadow_nation_map.get(faction, Nation.SAURON)
        else:
            # Free Peoples
            fp_nation_map = {
                'gondor': Nation.GONDOR, 'rohan': Nation.ROHAN,
                'elves': Nation.ELVES, 'dwarves': Nation.DWARVES,
                'the north': Nation.THE_NORTH, 'north': Nation.THE_NORTH,
            }
            default_nation = fp_nation_map.get(faction, Nation.GONDOR)

        # Process each piece and determine its unit type
        units_added = {'regular': 0, 'elite': 0, 'leader': 0, 'nazgul': 0}
        minions_mustered = []
        dup_minions_skipped = 0
        self._muster_by_nation_tmp = {}   # 119: per-line nation tally
        for piece_id in pieces:
            # Track the piece (t125 fix 1: refunds the FP dead pool if
            # a dead-counted pid is manually returned to the board —
            # downgrade returns done by hand instead of Controls.308)
            self._set_piece_zone(piece_id, to_key)

            # Check for Shadow minion pieces
            minion_name = MINION_PIECES.get(piece_id)
            if minion_name:
                if self._muster_minion(piece_id, minion_name, army,
                                       to_loc):
                    minions_mustered.append(minion_name)
                else:
                    # False = the minion was already in play (re-applied
                    # / drag-jitter placement line, 1.162 idempotency):
                    # board already correct, and NO record contribution
                    # - the duplicate line must not become a second
                    # muster decision (Vayze L103+L105: two
                    # '254 ...->1919' legs doubled Saruman at Orthanc
                    # for 87 parity batches).
                    dup_minions_skipped += 1
                continue

            side_hint = 'shadow' if is_shadow else 'fp'
            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types, side_hint)
            if nation is None or unit_type is None:
                readable_info = self._parse_readable_muster_info()
                if readable_info:
                    nation_str, unit_str = readable_info
                    # keys must be producible by _parse_readable_muster_info's
                    # \w[\w\s]* capture (no '&'); plural forms included
                    nation_map = {
                        'Gondor': Nation.GONDOR, 'Rohan': Nation.ROHAN,
                        'Elven': Nation.ELVES, 'Elves': Nation.ELVES,
                        'Dwarven': Nation.DWARVES, 'Dwarves': Nation.DWARVES,
                        'North': Nation.THE_NORTH, 'Isengard': Nation.ISENGARD,
                        'Sauron': Nation.SAURON,
                        'Southron': Nation.SOUTHRON_EASTERLINGS,
                        'Southrons': Nation.SOUTHRON_EASTERLINGS,
                        'Easterling': Nation.SOUTHRON_EASTERLINGS,
                        'Easterlings': Nation.SOUTHRON_EASTERLINGS,
                    }
                    if nation is None:
                        nation = nation_map.get(nation_str, default_nation)
                    if unit_type is None:
                        unit_type = (UnitKind.ELITE if unit_str == 'Elite' else
                                     UnitKind.REGULAR if unit_str == 'Regular' else 'LEADER')
                else:
                    if nation is None:
                        nation = default_nation
                    if unit_type is None:
                        unit_type = UnitKind.REGULAR

            # A Nazgul ALWAYS belongs to Sauron (data-model invariant).
            # The AUTHORITATIVE piece table types Nazgul pids as
            # (None, 'NAZGUL') by construction, so a Nazgul RETURNING from
            # off-board (SA Casualties/Reinforcements) with an unresolved
            # nation reached the FP default_nation=GONDOR when the return
            # branch computed is_shadow=False for the lone Nazgul piece —
            # minting an impossible (NAZGUL, GONDOR) figure that then
            # poisoned piece_types for every later lookup (2024-02-05
            # Polijuna-TBIS T19 pid 039 SA Casualties -> Barad-Dur).
            if unit_type == 'NAZGUL':
                nation = Nation.SAURON

            # Record dynamic type so future lookups work
            self.piece_types[piece_id] = (nation, unit_type)

            if unit_type == 'LEADER':
                army.leaders.append(gs.Leader(nation))
                units_added['leader'] += 1
            elif unit_type == 'NAZGUL':
                army.leaders.append(gs.Nazgul(nation))
                units_added['nazgul'] += 1
            elif unit_type == UnitKind.ELITE:
                army.units.append(gs.Unit(nation, UnitKind.ELITE))
                units_added['elite'] += 1
                # 119 (2026-07-25): the pieces NAME their nation — carry
                # it so card-effect musters at no-nation regions replay
                # the logged nation instead of a guess (demanray T6:
                # Rage of the Dunlendings' 2 ISENGARD at Hollin planted
                # as SAURON; per-nation casualty specs then forked)
                _byn_m9 = self._muster_by_nation_tmp
                _k9 = f"{nation.value}|elite"
                _byn_m9[_k9] = _byn_m9.get(_k9, 0) + 1
            else:
                army.units.append(gs.Unit(nation, UnitKind.REGULAR))
                units_added['regular'] += 1
                _byn_m9 = self._muster_by_nation_tmp
                _k9 = f"{nation.value}|regular"
                _byn_m9[_k9] = _byn_m9.get(_k9, 0) + 1

        side_name = "SA" if is_shadow else "FP"
        details = []
        if units_added['regular']:
            details.append(f"{units_added['regular']}R")
        if units_added['elite']:
            details.append(f"{units_added['elite']}E")
        if units_added['leader']:
            details.append(f"{units_added['leader']}L")
        if units_added['nazgul']:
            details.append(f"{units_added['nazgul']}N")
        if minions_mustered:
            details.extend(minions_mustered)
        if details:
            self.log(f"Muster ({side_name}): {'+'.join(details)} to {to_loc}")
        else:
            self.log(f"Muster ({side_name}): {len(pieces)} to {to_loc}")

        if dup_minions_skipped and not minions_mustered \
                and not any(units_added.values()):
            # every piece on the line was a suppressed duplicate minion
            # placement - emitting the (empty) muster record would
            # fabricate a decision for a client no-op line
            return

        muster_side = Side.SHADOW if is_shadow else Side.FREE_PEOPLES
        # Minions (Saruman/Witch King/Mouth) come from a neutral VASSAL box
        # so is_shadow misses them — they are ALWAYS Shadow. The wrong side
        # also glued the Shadow die onto the next unrelated Shadow row.
        if minions_mustered:
            muster_side = Side.SHADOW
        muster_params = {'region': to_loc, 'units': units_added}
        if getattr(self, '_muster_by_nation_tmp', None):
            # 119: the placed pieces name their nation — ground truth
            # for card-effect musters at no-nation regions
            muster_params['_by_nation'] = dict(self._muster_by_nation_tmp)
        if minions_mustered:
            muster_params['minions'] = minions_mustered
        # Placements dictated by a fixed-effect card are not decisions
        if (getattr(self, '_fixed_effect_card', False)
                and self._pending_die[muster_side] is None):
            muster_params['auto'] = True
        # Undecodable target location (VASSAL box token) — untrackable
        if sanitize(to_loc) not in board.regions:
            muster_params['auto'] = True
        self._record_action('muster', muster_params, muster_side)
        # Remember minion musters so an immediate same-gesture drag of
        # the minion (no die consumed between) can correct the placement
        # region instead of emitting a phantom minion move (2023-11-03
        # ketchuptime T8 L783-784: Mouth staged at Minas Morgul, dropped
        # at Dol Guldur).
        if minions_mustered:
            self._minion_muster_ctx = {
                'idx': len(self.action_log) - 1,
                'line': self.line_num,
                'die_seq': getattr(self, '_die_consume_seq', 0),
                'minions': sorted(minions_mustered),
            }

    def _muster_minion(self, piece_id: str, minion_name: str, army: gs.Army, to_loc: str):
        """Handle mustering a Shadow minion (Saruman, Witch King, Mouth of Sauron).

        Returns True on a genuine first entry.  Returns False when the
        minion is ALREADY on the board: base-game minions never legally
        re-enter play after entering once, so a second placement line is
        either client drag-jitter (same region - pure no-op) or a
        same-gesture placement correction (different region - RELOCATE
        the existing object).  Either way no new object and no new
        muster record (1.162 idempotency doctrine applied to
        placements)."""
        for k in list(self.state.armies):
            a0 = self.state.armies[k]
            for i, l in enumerate(a0.leaders):
                if getattr(l, 'name', None) == minion_name:
                    if sanitize(k) not in board.regions:
                        # parked in an off-board/box zone (staging leg
                        # of the muster gesture, e.g. Vayze 2040->2037
                        # before 2037->Orthanc): NOT in play - absorb
                        # the box object and fall through to the
                        # genuine first-entry path below
                        a0.leaders.pop(i)
                        if not a0.units and not a0.leaders:
                            del self.state.armies[k]
                        break
                    if a0 is not army:
                        army.leaders.append(a0.leaders.pop(i))
                        if not a0.units and not a0.leaders:
                            del self.state.armies[k]
                        self.log(f"MINION placement corrected: "
                                 f"{minion_name} -> {to_loc}")
                    return False
        # Set in_play flag on the game state minion
        if minion_name == 'Saruman' and self.state.saruman:
            self.state.saruman.in_play = True
        elif minion_name == 'Witch King' and self.state.witch_king:
            self.state.witch_king.in_play = True
        elif minion_name == 'Mouth of Sauron' and self.state.mouth_of_sauron:
            self.state.mouth_of_sauron.in_play = True

        # Add as a named leader to the army.  FIX (minion typing): create
        # a game_logic.Minion (a Leader carrying a proper Shadow nation),
        # NOT a Companion - so side-detection / drift / encode classify
        # these as Shadow exactly as the engine's own muster does.
        import game_logic as _gl_mn
        _mn_nat = (gs.Nation.ISENGARD if minion_name == 'Saruman'
                   else gs.Nation.SAURON)
        minion_leader = _gl_mn.Minion(nation=_mn_nat, name=minion_name)
        army.leaders.append(minion_leader)
        self.log(f"MINION MUSTERED: {minion_name} -> {to_loc}")
        return True

    def _handle_political(self, pieces: List[str], from_loc: str, to_loc: str):
        """Handle political track movement.

        VASSAL numbering: 0 = At War, 3 = most passive.
        Internal numbering: 0 = most passive, 3 = At War.
        Conversion: internal = 3 - vassal.
        """
        from_match = re.search(r'Political (\d)', from_loc)
        to_match = re.search(r'Political (\d)', to_loc)

        # t138 wave-3 (triage N1): political positioning is ABSOLUTE —
        # the counter's box is the whole state.  A drag ENDING outside
        # the track (marker parked on a card being played / bounced
        # through the dice box) only zone-parks the pid; a drag ENDING
        # in a political box sets that nation's counter to that box
        # regardless of the declared source (savedbyhim01/sharpz L65-66:
        # Isengard advance staged through FP Card Play — both legs were
        # dropped by the old both-boxes-required gate; Kaeldar/gus833
        # L414-416: an advance taken back via a Controls.281 bounce
        # stayed applied).
        if not to_match:
            for p in pieces:
                if p in POLITICAL_MARKERS:
                    self.piece_region[p] = to_loc
                    self.log(f"Political marker parked off-track: "
                             f"{p} -> {to_loc}")
            return

        # Convert from VASSAL positions (0=At War) to internal (3=At War)
        to_internal = 3 - int(to_match.group(1))
        if from_match:
            from_internal = 3 - int(from_match.group(1))
            if from_internal == to_internal:
                for p in pieces:
                    if p in POLITICAL_MARKERS:
                        self.piece_region[p] = to_loc
                return
        else:
            # absolute drop-in from a non-political zone: each nation's
            # OWN current box is its transition source — a group drop-in
            # must never let the FIRST marker's already-at-target box
            # swallow the OTHER nations' advances (F8 class).
            from_internal = None
            if not any(p in POLITICAL_MARKERS for p in pieces):
                return

        # t133 fix F8: a GROUP drag of several political markers applies
        # the box transition to EVERY nation in the piece list (the old
        # loop broke after the first marker — 2025-06-26 kapcerz L210
        # moved THE_NORTH + ELVES together and Elves stayed a box
        # behind for 6 checkpoints; 39 games).  The action record stays
        # ONE row (first marker), matching the old schedule shape; the
        # extra nations get the BOX update only — their active flag is
        # untouched, so the human's separate activation FLIP (the
        # Controls.323 toggle) still lands passive->ACTIVE (setting
        # active here inverted the later real flip: AngryBird canary,
        # The North ended passive after the human activated it).
        _pol_recorded = False
        for piece_id in pieces:
            nation = POLITICAL_MARKERS.get(piece_id)
            if nation:
                _src_box9 = (from_internal if from_internal is not None
                             else self.state.political.boxes.get(nation,
                                                                 0))
                if from_internal is None and _src_box9 == to_internal:
                    # drop-in no-op for THIS nation only
                    self.piece_region[piece_id] = to_loc
                    continue
                self.state.political.boxes[nation] = to_internal
                if _pol_recorded:
                    direction = ("advanced" if to_internal > _src_box9
                                 else "retreated")
                    self.log(f"Political: {nation.value} {direction} to "
                             f"{to_internal} (group drag, box only)")
                    continue
                # WAVE-6 (2026-07-25) INVERTED-RULE FIX: a manual counter
                # DRAG on the political track (muster die / event advancing
                # a nation's disposition) NEVER activates the nation
                # (session-21 rule; the engine's advance_political has
                # asserted this for years - game_logic L1603-1615).  The
                # old heuristic set active=True on EVERY advance, so the
                # parser TRUTH MIRROR ran ahead of the rule-correct engine
                # and manufactured ~45 false 'engine-behind' political
                # parity events across 17/20 held-out audit games (each
                # box-matched, active-only: e.g. pol@GONDOR eng=(2,False)
                # truth=(2,True)), mis-colouring the pure-mode triage.
                # Real activations are captured at their own sources: the
                # explicit marker FLIP (pieces 256-263, _POL_FLIP below ->
                # activate aux), an attack on the nation's region
                # (_auto_tag_defender_political + the engine's
                # activate_nation_for_attack), fellowship declare/reveal,
                # companions, event cards.  The box-advance RECORD itself
                # is unchanged (decision schedule byte-identical); only the
                # spurious active-flag write is removed.  A now-unmasked
                # real flip legitimately emits its activate aux (was_active
                # False) - a ground-truth idempotent aux, not a BC
                # decision, so extraction stays decision-preserving.
                direction = ("advanced" if to_internal > _src_box9
                             else "retreated")
                self.log(f"Political: {nation.value} {direction} to {to_internal}")
                pol_side = Side.SHADOW if nation in SHADOW_NATIONS else Side.FREE_PEOPLES
                # During an active battle the defender's track advance is
                # AUTOMATIC (rulebook p.35: attacked nation activates/
                # advances) — mark it so extraction doesn't treat it as a
                # decision.  RB-07 (2026-07-22): the engine advances each
                # attacked nation exactly ONCE per battle
                # (advance_defenders_political); the SECOND advance in the
                # same battle window is the capture-advance.  wave-E fix 5
                # (2026-07-23) makes the ENGINE do the capture-advance too
                # (_check_settlement_capture -> activate_nation_for_attack
                # captured=True, rulebook L349-350), but the capture-advance
                # is kept as a non-auto political_sync GROUND-TRUTH aux (NOT a
                # decision) rather than marked `auto`: the political_sync
                # applier is a CONDITIONAL absolute-set (`if want > cur`) so it
                # is IDEMPOTENT with the engine's now-automatic advance (no
                # double-advance), AND leaving it aux keeps the extraction
                # SCHEDULE byte-identical (marking it auto shifted 22/40
                # harness games).  So: FIRST advance -> auto (engine attack-
                # advance); subsequent same-window advance -> political_sync,
                # which the engine now also produces and the sync confirms.
                _pol_params = {'nation': nation.value,
                               'from': _src_box9, 'to': to_internal}
                if self._battle_synth_region is not None \
                        and to_internal > _src_box9:
                    _seen = getattr(self, '_battle_auto_pol_nations', None)
                    if _seen is None:
                        _seen = self._battle_auto_pol_nations = set()
                    if nation not in _seen:
                        _pol_params['auto'] = True
                        _seen.add(nation)
                self._record_action('political', _pol_params, pol_side)
                _pol_recorded = True

    def _handle_token_move(self, piece_id: str, from_loc: str, to_loc: str):
        """Handle FP action token movement."""
        if piece_id == TOKEN_POLITICAL:
            name = "Political"
        else:
            name = "Card Draw"

        if 'Dice' in to_loc and ('Reinf' in from_loc or 'Board' in from_loc
                                 or 'Hunt' in from_loc):
            # Token placed into dice area.  Tokens ONLY come from the
            # pre-game bid, so ANY placement marks a bid resource — the
            # bookkeeping drag can happen mid-turn (2023-11-03 ketchuptime
            # places at line 83, AFTER the T1 roll, so a pre-roll state
            # snapshot misses it).
            if piece_id == TOKEN_POLITICAL:
                self.state.fp_political_token = True
            else:
                self.state.fp_card_token = True
            self.log(f"FP token available: Hunt for the Ring ({name})")
        elif 'Dice' in from_loc:
            # Token used from dice area
            if piece_id == TOKEN_POLITICAL:
                self.state.fp_political_token = False
            else:
                self.state.fp_card_token = False
            self.log(f"FP uses Hunt for the Ring ({name})")
            self._record_action('use_token', {
                'token': name.lower(),
            }, Side.FREE_PEOPLES)

    def _handle_marker_move(self, piece_id: str, from_loc: str, to_loc: str):
        """Control-marker / VP-marker moves: explicit ground truth.

        Control marker onto map region R = R captured by the marker's
        side; marker off R = control released. VP marker onto 'VP N' =
        that side's score is N. Pure bookkeeping — records NO action row.
        """
        if piece_id in VP_MARKERS:
            side = VP_MARKERS[piece_id]
            m = re.search(r'VP (\d+)', to_loc or '')
            if m:
                n = int(m.group(1))
                if side == Side.SHADOW:
                    self.state.vp.shadow = n
                else:
                    self.state.vp.free_peoples = n
                self.piece_region[piece_id] = to_loc
                self.log(f"VP marker: {side.name} -> {n}")
                # t125 fix 7 (ruling 1.157): a military-victory VP
                # threshold ARMS the game-over signal; it latches only
                # inside the pure-noise log tail (VP can drop again —
                # FP retakes — so this is never definite mid-play)
                if (self.state.vp.shadow >= 10
                        or self.state.vp.free_peoples >= 4):
                    self._eog_pending = (self._eog_pending
                                         or 'military_victory')
            return

        side = (Side.SHADOW if piece_id in SA_CONTROL_MARKERS
                else Side.FREE_PEOPLES)

        def _region_of(loc):
            if not loc:
                return None
            base = stronghold_base_region(loc)
            san = sanitize(base if base is not None else loc)
            return san if san in board.regions else None

        src = _region_of(from_loc)
        dst = _region_of(to_loc)
        if src is not None:
            self.state.enemy_controlled[side].discard(src)
            # RB-08 (1.55a): control follows OCCUPATION — if the
            # capturing side still solely occupies, removing the marker
            # does not release control (idempotent re-add).
            self._update_region_control(src)
            self.log(f"Control marker removed: {side.name} off {from_loc}")
        if dst is not None:
            self.state.enemy_controlled[side].add(dst)
            self.log(f"Control marker: {side.name} controls {to_loc}")
        self.piece_region[piece_id] = to_loc

    def _handle_fellowship_companions(self, pieces: List[str], from_loc: str, to_loc: str):
        """Handle fellowship and companion movements."""
        # Check for companion death (moving to Card Play, Dead, Casualties
        # areas).  Location codes decode to: 'FP Card Play', 'FP Dead',
        # 'SA Card Play', etc.  Classification is delegated to the single
        # resolver _zone_is_companion_death — the private substring test
        # this used to run swallowed every separation into DEAD MARSHES
        # ('Dead' is a substring of it) as a companion death.
        for piece_id in pieces:
            comp_name = COMPANIONS.get(piece_id)
            if comp_name:
                # Check if companion is dying
                is_dying = self._zone_is_companion_death(to_loc)

                if is_dying:
                    # t133: keep the piece-level view in client sync
                    # (dead-zone equivalences count card-play parking)
                    self._set_piece_zone(piece_id, to_loc, dead=True)
                    # Remove companion from fellowship
                    for i, comp in enumerate(self.state.fellowship.companions):
                        if comp.name == comp_name:
                            # Check if this was the guide (for hunt sacrifice distinction)
                            was_guide = (self.state.fellowship.guide and
                                         self.state.fellowship.guide.name == comp_name)
                            self.state.fellowship.companions.pop(i)
                            # If this was the guide, the guide passes to
                            # the highest-level remaining companion
                            # (mirror the engine — previously left None,
                            # a stale/absent guide)
                            if was_guide:
                                self._reassign_guide()
                            self.log(f"Companion DIED: {comp_name}")
                            if comp_name == 'Gandalf the Grey':
                                # A later return of this piece to the map
                                # is the Gandalf the White transformation
                                self._gandalf_dead = True
                            # Distinguish hunt sacrifice from other deaths
                            if self._in_hunt_resolution:
                                if was_guide:
                                    self._record_action('hunt_sacrifice_guide', {
                                        'companion': comp_name
                                    }, Side.FREE_PEOPLES)
                                else:
                                    self._record_action('hunt_sacrifice_random', {
                                        'companion': comp_name
                                    }, Side.FREE_PEOPLES)
                                self._in_hunt_resolution = False
                                if self._hunt_ctx is not None:
                                    self._hunt_ctx['resolved'] = True
                                    self._hunt_ctx = None
                            else:
                                self._record_action('companion_death', {
                                    'companion': comp_name
                                }, Side.FREE_PEOPLES)
                            break
                elif 'Guide' in to_loc:
                    # Becoming guide — may already be in fellowship or joining fresh (e.g. Gollum)
                    found = False
                    for comp in self.state.fellowship.companions:
                        if comp.name == comp_name:
                            self.state.fellowship.guide = comp
                            found = True
                            break
                    if not found:
                        if comp_name == 'Gollum':
                            # GOLLUM TAKEOVER (piece 364 -> Guide Box,
                            # e.g. GenericPie L941 '364 Game.2038 ->
                            # Game.1999').  The ENGINE
                            # (game_logic.change_guide) makes Gollum the
                            # guide with fellowship.companions EMPTY —
                            # he is never a companion.  Appending him
                            # here (pre-fix behavior) opened a permanent
                            # engine-vs-parser composition diff, which
                            # in turn forced the AI bridge to hide the
                            # real corpus drag.  Mirror _reassign_guide
                            # exactly: guide=Gollum (with his hunt
                            # ability), companions stay empty.
                            self.state.fellowship.companions.clear()
                            _g = self.state.fellowship.guide
                            if _g is None or _g.name != 'Gollum':
                                self.state.fellowship.guide = gs.Companion(
                                    'Gollum', 0, 0,
                                    guide_ability='no_reveal_on_damage')
                        else:
                            # New companion joining as guide from
                            # off-fellowship (official card values — a
                            # level-0 guide lost every max-level pick in
                            # _reassign_guide)
                            _lv9, _ld9 = _COMPANION_LEVELS.get(
                                comp_name, (1, 1))
                            comp = gs.Companion(name=comp_name,
                                                level=_lv9,
                                                leadership=_ld9)
                            self.state.fellowship.companions.append(comp)
                            self.state.fellowship.guide = comp
                    self.log(f"Guide: {comp_name}")
                    self._record_action('choose_guide', {
                        'companion': comp_name
                    }, Side.FREE_PEOPLES)
                elif 'Guide' in from_loc and 'Fellowship' in to_loc:
                    # t138 Q2 (ruling 1.159 gap): players park SEVERAL
                    # companions in the Guide Box then take all but the
                    # pick back to the Fellowship Box — the REMAINING
                    # occupant is the chosen guide (StellaRossa/Meldar
                    # L264-267: 242 243 245 in, 243+245 back, guide is
                    # 242; the parser kept the last-staged 245).
                    self.piece_region[piece_id] = to_loc
                    # retract this companion's staged choose_guide
                    # (walk-back of a group staging drag)
                    for _b2 in range(1, min(7, len(self.action_log))
                                     + 1):
                        _at2, _ap2, _s2 = self.action_log[-_b2]
                        if _at2 == 'choose_guide' and \
                                _ap2.get('companion') == comp_name:
                            del self.action_log[-_b2]
                            break
                    _g2 = self.state.fellowship.guide
                    if _g2 is not None and _g2.name == comp_name:
                        _rem2 = [q for q, z in self.piece_region.items()
                                 if q != piece_id and q in COMPANIONS
                                 and sanitize(str(z)) == 'guide box']
                        _pick2 = None
                        if len(_rem2) == 1:
                            _pn2 = COMPANIONS[_rem2[0]]
                            _pick2 = next(
                                (c for c in
                                 self.state.fellowship.companions
                                 if c.name == _pn2), None)
                        if _pick2 is not None:
                            self.state.fellowship.guide = _pick2
                            self.log(f"Guide walk-back: {comp_name} "
                                     f"out, guide = {_pick2.name} "
                                     f"(remaining Guide-Box occupant)")
                        else:
                            self._reassign_guide()
                            self.log(f"Guide walk-back: {comp_name} "
                                     f"out, guide reassigned")
                    else:
                        self.log(f"Guide-Box walk-back: {comp_name} "
                                 f"-> Fellowship Box")
                elif to_loc in ('Board2', 'Board3', 'Board4', 'Board5',
                                'Board6'):
                    # t138 wave-3 (triage pass-4 3.6 / pass-5 3.1,
                    # StregoRe family): companions separate by dragging
                    # into a BOARD box, then fan onward to the map
                    # (kapcerz/Jan L500-501: '243 245 246 Game.1998 ->
                    # Game.2034 -> Game.1969').  Stage the separation:
                    # pop from the fellowship, stash the object, record
                    # companion_separate with a deferred '?' region the
                    # exit leg patches.
                    for i, comp in enumerate(
                            self.state.fellowship.companions):
                        if comp.name == comp_name:
                            separated = \
                                self.state.fellowship.companions.pop(i)
                            if not hasattr(self, '_board_sep_stash'):
                                self._board_sep_stash = {}
                            self._board_sep_stash[piece_id] = separated
                            self.piece_region[piece_id] = to_loc
                            _g = self.state.fellowship.guide
                            if _g is not None and _g.name == comp_name:
                                self._reassign_guide()
                            self._record_action('companion_separate', {
                                'companion': comp_name,
                                'to_region': '?'
                            }, Side.FREE_PEOPLES)
                            self.log(f"Separation staged via {to_loc}: "
                                     f"{comp_name} (destination pending)")
                            break
                elif is_map_region(to_loc) \
                        or stronghold_base_region(to_loc) is not None:
                    # Separating to a region — or to a stronghold BOX
                    # zone (t133 fix F1b, rulebook p.38: with the
                    # Ring-bearers in a region whose FP stronghold is
                    # under siege, companions "separate into the
                    # Stronghold and may not leave"; USER RULING A6
                    # 2026-07-17 mirrors the client drag).  Box
                    # destinations resolve to the base region's army;
                    # the piece stays zoned in the box like the
                    # siege-garrison path.
                    _sep_base = stronghold_base_region(to_loc)
                    _dest = _sep_base if _sep_base is not None else to_loc
                    to_san = sanitize(_dest)
                    # Narrow guard: only the region lookup may fail
                    # (unknown region); the separation body below must
                    # run unguarded — a swallowed mid-body error used
                    # to delete the popped companion with no record.
                    try:
                        to_region = self.board.region(to_san)
                    except Exception:
                        self.log(f"Separation skipped: unknown region "
                                 f"{to_san}")
                        continue
                    to_key = to_region.orig_name if hasattr(to_region, 'orig_name') else _dest

                    # Remove from fellowship
                    for i, comp in enumerate(self.state.fellowship.companions):
                        if comp.name == comp_name:
                            separated = self.state.fellowship.companions.pop(i)
                            # Add to army (sanitized lookup —
                            # duplicate-army-key bug)
                            _, _sep_army = self._find_or_create_army(
                                to_key)
                            _sep_army.leaders.append(separated)
                            # t133 fix F1a: mirror the drag into the
                            # piece-level view (rulebook p.38: the
                            # figure moves to the map) — without
                            # this the piece stays zoned
                            # 'Fellowship Box' and the verifier
                            # counts a phantom dead companion
                            # (largest sweep family, 462 games).
                            self.piece_region[piece_id] = (
                                to_loc if _sep_base is not None
                                else to_key)
                            if _sep_base is not None:
                                self.log(
                                    f"CURATION NOTE (A6): separation "
                                    f"into stronghold box {to_loc} "
                                    f"(legal via card or fellowship "
                                    f"co-location; not gated)")
                            self.log(f"Separated: {comp_name} -> {to_loc}")
                            # Guide reassignment is IMPLICIT in the
                            # client (no line emitted): if the GUIDE
                            # just left, the guide passes to the
                            # highest-level remaining companion —
                            # mirror the engine's change_guide so the
                            # parser doesn't keep a stale guide
                            # (caught by cloned-state verification
                            # 2026-07-16).
                            _g = self.state.fellowship.guide
                            if _g is not None and _g.name == comp_name:
                                self._reassign_guide()
                            # t133 fix F1b: a Guide-Box STAGING drag
                            # right before a group separation is not
                            # a guide pick — retract the just-made
                            # choose_guide for the companion now
                            # leaving (SlowCobra-era pattern: 246
                            # Fellowship Box -> Guide Box, then
                            # '241 246 Guide Box -> destination').
                            if 'Guide' in from_loc:
                                for _b9 in range(
                                        1, min(4, len(self.action_log))
                                        + 1):
                                    _at9, _ap9, _ = \
                                        self.action_log[-_b9]
                                    if (_at9 == 'choose_guide'
                                            and _ap9.get('companion')
                                            == comp_name):
                                        # a die glued onto the
                                        # staging drag belongs to
                                        # the separation itself —
                                        # re-pend it so the record
                                        # below takes it
                                        _du9 = _ap9.get('die_used')
                                        if _du9 is not None and \
                                                self._pending_die[
                                                    Side.FREE_PEOPLES] \
                                                is None:
                                            self._pending_die[
                                                Side.FREE_PEOPLES] = _du9
                                        del self.action_log[-_b9]
                                        self.log(
                                            "  (staging choose_guide "
                                            "retracted: group "
                                            "separation)")
                                        break
                            self._record_action('companion_separate', {
                                'companion': comp_name,
                                'to_region': to_san
                            }, Side.FREE_PEOPLES)
                            break

    def _reassign_guide(self):
        """Guide passes to the highest-level remaining companion (ties
        keep whichever is first — the AI's tie choice arrives as a
        separate choose_guide line and overwrites it).  Gollum takes
        over when no companions remain.  Mirrors game_logic.change_guide
        so a guide-separation/casualty never leaves a stale guide."""
        comps = self.state.fellowship.companions
        if not comps:
            self.state.fellowship.guide = gs.Companion(
                'Gollum', 0, 0, guide_ability='no_reveal_on_damage')
            return
        best_lvl = max(c.level for c in comps)
        self.state.fellowship.guide = next(
            c for c in comps if c.level == best_lvl)

    def _handle_fellowship_track(self, from_loc: str, to_loc: str,
                                 marker_id: Optional[str] = None):
        """Handle fellowship progress track or Mordor track movement."""
        # Progress track: 'Track N' -> 'Track N'
        from_match = re.search(r'Track (\d+)', from_loc)
        to_match = re.search(r'Track (\d+)', to_loc)

        # Corruption marker (252) on the same physical track: value = track
        # number. Not a fellowship move — never touches progress.
        if marker_id == '252' and to_match:
            self._sync_corruption(int(to_match.group(1)))
            return

        if from_match and to_match:
            from_pos = int(from_match.group(1))
            to_pos = int(to_match.group(1))
            self.state.fellowship.progress = to_pos
            self.log(f"Fellowship track: {from_pos} -> {to_pos}")
            if to_pos == from_pos + 1 \
                    and not self.state.fellowship.hidden \
                    and getattr(self, '_ai_bridge_log', False):
                # AI-BRIDGE grammar (t123): a REVEALED fellowship cannot
                # move (official rule, engine-enforced) — a +1 advance
                # while revealed is marker RE-PLACEMENT bookkeeping,
                # never a decision (wotr2026_7_17_11_26 T7 L464-467 /
                # T8 L540-544: an old bridge build walked the counter
                # back up right after the reveal flip; parsed as 9
                # phantom fellowship_moves).  Progress is already
                # synced above.  Corpus path unchanged (campaign canary:
                # ungated, branstonpickle flipped CLEAN->MISMATCH).
                self.log("  (790 advance while revealed: marker "
                         "re-placement, not a move)")
            elif to_pos == from_pos + 1:
                # A real fellowship movement (+1 progress)
                self._record_action('fellowship_move',
                                    {'from': from_pos, 'to': to_pos},
                                    Side.FREE_PEOPLES)
            elif to_pos < from_pos:
                # Track RESET (declare/reveal consumes progress) — not a
                # move. Stash the consumed progress for the pawn-move line
                # that follows, so the declare decision keeps its true
                # pre-decision progress.
                self._pre_reset_progress = from_pos
            return

        # Mordor track: 'Mordor N' -> 'Mordor N'
        from_mordor = re.search(r'Mordor (\d+)', from_loc)
        to_mordor = re.search(r'Mordor (\d+)', to_loc)

        if to_mordor:
            to_step = int(to_mordor.group(1))
            from_step = int(from_mordor.group(1)) if from_mordor else 0
            _was_entered = self.state.fellowship.mordor_entered
            self.state.fellowship.mordor_entered = True
            self.state.fellowship.mordor_step = to_step
            if not _was_entered:
                # t125 fix 3: entry consumes the progress counter; the
                # CLIENT's silent 248 reset actually lands at the first
                # fellowship flip after entry (see _handle_flip), so
                # the marker itself is synced there — here only the
                # entry housekeeping (hunt-tile re-pool) runs.
                self._on_mordor_entry()
            # t125 fix 7 (ruling 1.157): reaching the Cracks of Doom
            # opens the FINAL-HUNT window — the arrival's own hunt
            # draw and damage resolution are still REAL play (they
            # decide ring destroyed vs claimed); the latch fires when
            # that resolution lands (_sync_corruption / second draw /
            # sacrifice record).
            if to_step == 5 and self._game_over is None:
                self._final_hunt_pending = True
                self.log("Cracks of Doom reached: final-hunt window "
                         "open [ruling 1.157]")
            mordor_names = {0: 'Cirith Ungol / The Black Gate',
                            1: 'Morgai / Udun', 2: 'Carach Angren',
                            3: 'Gorgoroth Plain',
                            4: 'On the Slopes of Mount Doom',
                            5: 'The Cracks of Doom'}
            to_name = mordor_names.get(to_step, f'Mordor step {to_step}')
            from_name = mordor_names.get(from_step, f'Mordor step {from_step}')
            if from_mordor is None:
                # ENTRY drag (map region / elsewhere -> a Mordor box):
                # housekeeping for the automatic Mordor entry (the rules
                # enter the Fellowship when it is revealed or declared at
                # Morannon/Minas Morgul — the engine does this inside
                # fellowship_reveal_place/declare/move execution).  State
                # sync only when the pawn was already AT a gateway; from
                # any other region the drag IMPLIES an unlogged DECLARE
                # at the reachable gateway (2023-02-03 Wirbelwind line
                # 304: Dagorlad -> Mordor 0, i.e. declare at Morannon
                # spending the progress counter) — emit that decision or
                # the closed loop never enters Mordor.
                from_san = sanitize(from_loc) if from_loc else ''
                gateways = ('morannon', 'minas morgul')
                if is_map_region(from_loc) and from_san not in gateways:
                    gate = None
                    try:
                        adj = board.adjacency(from_san) if hasattr(
                            board, 'adjacency') else board.adj.get(
                            from_san, set())
                    except Exception:
                        adj = set()
                    for g in gateways:
                        if g in adj:
                            gate = g
                            break
                    gate = gate or 'morannon'
                    # progress_before: the counter still holds its
                    # pre-entry value here (the client's silent 248
                    # reset lands at the first flip AFTER entry, t125
                    # fix 3) - but a track-reset line may already have
                    # stashed it; peek WITHOUT clearing (the stash
                    # lifecycle belongs to the pawn-move branch).
                    _stash_m9 = getattr(self, '_pre_reset_progress', None)
                    self._record_action('fellowship_declare', {
                        'region': gate,
                        'progress_before': max(
                            int(self.state.fellowship.progress),
                            int(_stash_m9 or 0)),
                        'mordor_entry': True}, Side.FREE_PEOPLES)
                    self.log(f"Fellowship ENTERS Mordor via implied "
                             f"declare at {gate} (was {from_loc})")
                else:
                    self.log(f"Fellowship ENTERS Mordor: {from_loc} -> "
                             f"{to_name} (step {to_step})")
                return
            self.log(f"Fellowship Mordor: {from_name} (step {from_step}) -> {to_name} (step {to_step})")
            self._record_action('mordor_move', {
                'from_step': from_step, 'to_step': to_step,
                'from_name': from_name, 'to_name': to_name
            }, Side.FREE_PEOPLES)

    def _retire_die_from_hunt(self, piece_id: str):
        """A die piece cleared out of the Hunt Box by a kill/cleanup
        command (t125): the client renders it in Used; drop it from
        state.hunt_box so dice censuses agree.  Keeps 'used' status."""
        try:
            mp = self.die_id_map.get(int(piece_id))
        except (ValueError, TypeError):
            mp = None
        if mp is None:
            return
        sd_, ix_ = mp
        if ix_ >= len(self.state.dice[sd_]):
            return
        d_ = self.state.dice[sd_][ix_]
        if d_ in self.state.hunt_box:
            self.state.hunt_box.remove(d_)
            self.log(f"Die {piece_id} retired from Hunt Box "
                     f"(kill/cleanup)")
        d_.used = True
        self.piece_region[piece_id] = 'Used'

    def _on_mordor_entry(self):
        """Client-mirrored Mordor-entry housekeeping beyond the marker
        walk itself (t125).  Extended by the hunt-tile model (fix 2):
        the client silently re-pools Eye/special tiles on entry."""
        self._mordor_tile_repool()

    def _handle_die_to_hunt(self, die_id: str, side_hint=None):
        """Handle die being added to hunt box."""
        die_int = int(die_id)

        if die_int not in self.die_id_map and side_hint is Side.SHADOW:
            # A bonus die (Saruman/WK/Mouth) allocated as an eye BEFORE
            # its first roll has no mapping yet — register it now
            self._register_bonus_die(die_int, side_hint)
        mapping = self.die_id_map.get(die_int)
        if mapping:
            side, idx = mapping
            if idx < len(self.state.dice[side]):
                die = self.state.dice[side][idx]
                already = die in self.state.hunt_box
                die.used = True
                if not already:
                    self.state.hunt_box.append(die)
                side_name = "Shadow" if side == Side.SHADOW else "FP"
                self.log(f"{side_name} die {idx} (id={die_int}) -> Hunt Box")
                # A SHADOW die entering the Hunt Box outside a roll is an
                # EYE allocation — same bookkeeping as the Controls.372
                # button path (roll alignment + the allocation decision)
                if side == Side.SHADOW and not already:
                    die.current = ActionDieFace.EYE
                    self.preallocated_eyes.add(die_int)
                    self.state.eyes_allocated = getattr(
                        self.state, 'eyes_allocated', 0) + 1
                    self._record_action('eye_allocate', {
                        'die_idx': idx,
                        'hunt_box_size': len(self.state.hunt_box)
                    }, Side.SHADOW)

    def _handle_card_draw_with_decoder(self, piece_id: str, deck_code: str, hand_code: str):
        """Handle card draw using the card decoder."""
        card_info = self.card_decoder.handle_draw(piece_id, deck_code, hand_code)

        if card_info:
            side = Side.FREE_PEOPLES if card_info['side'] == 'FP' else Side.SHADOW
            deck_type = card_info['deck_type']

            # Update deck counts
            if 'FP' in deck_type:
                if 'Character' in deck_type:
                    self.state.decks.fp_character = max(0, self.state.decks.fp_character - 1)
                else:
                    self.state.decks.fp_strategy = max(0, self.state.decks.fp_strategy - 1)
            else:
                if 'Character' in deck_type:
                    self.state.decks.sh_character = max(0, self.state.decks.sh_character - 1)
                else:
                    self.state.decks.sh_strategy = max(0, self.state.decks.sh_strategy - 1)

            # Store card in hand state
            self.state.hands[side].hand.append(card_info)

            side_name = card_info['side']
            if card_info['decoded']:
                self.log(f"{side_name} drew: {card_info['card_name']}")
            else:
                self.log(f"{side_name} drew: {deck_type} (unknown)")
            _draw_params = {'deck_type': deck_type,
                            'card_name': card_info.get('card_name', 'Unknown')}
            # Die-less draws during a battle window are ability draws
            # (Witch-King: draw after a battle he fights) — not decisions
            if (self._pending_die[side] is None
                    and self._battle_synth_region is not None):
                _draw_params['auto'] = True
            self._record_action('draw_card', _draw_params, side)

    def _handle_card_retraction(self, piece_id: str, hand_code: str):
        """A card coming BACK from a play zone into a hand.

        THE RE-ENTRY TWIN of _handle_card_zone_discard, and it lives
        next to it so both directions are visible in one place.

        MEASURED SPECIMEN (USER live 2026-08-18, his own log, pid 108):
            L212  Controls.840  108                     committed
            L213  <auto> 861 reveal 108  hand -> 2037   PLAYED
            L214  CardBuffet.3  108  Game.2037 -> 45    RETRACTED
            L216  <auto> 861 reveal 106 ...             plays 106 instead
            L239  Controls.296  108  Controls.45        leaves for good
        He committed a card, took it back, and played a different one.
        The exit at L213 was booked; the return at L214 was not, so for
        the next 25 lines a card he was PHYSICALLY HOLDING was modelled
        as gone.  That is E1 ("when I play a card, it does not seem to
        be registered") and the phantom discard prompt that followed it
        is hand-limit logic running on a wrong count.
        The COUNT self-heals only by luck: the real removal at L239
        finds the card already absent and no-ops - which is why this
        never showed up as a final-state mismatch and why E2's "RESYNC
        does not fix it" is consistent with a correct end state.

        BLAST RADIUS, measured before writing it
        (streak/census_card_retraction.py, 1,206 logs): 300 lines - a
        card piece moving from a Current Card zone (Game.2036/2037)
        into a hand.  Draws (60,357 lines, deck zones) are a different
        grammar and were already handled; keeping the source-zone test
        tight is what separates them."""
        side = Side.FREE_PEOPLES if hand_code == '45' else Side.SHADOW
        pid = int(piece_id)
        if hand_code == '45':
            dt = 'FP Character' if _FP_PID_TYPES[pid - 106] == 'C' \
                else 'FP Strategy'
        else:
            dt = 'SA Character' if _SA_PID_TYPES[pid - 158] == 'C' \
                else 'SA Strategy'
        h = self.state.hands[side].hand
        already = any(isinstance(e, dict)
                      and str(e.get('piece_id')) == str(piece_id)
                      for e in h)
        if already:
            return                      # idempotent: never double-add
        num = self.card_decoder.decode_piece(pid, dt)
        name = self.card_decoder.get_card_name(num, dt)
        info = {'piece_id': str(piece_id),
                'side': 'FP' if hand_code == '45' else 'SA',
                'deck_type': dt,
                'drawn_turn': getattr(self.card_decoder, 'turn', None),
                'card_number': num,
                'card_name': name,
                'decoded': num is not None}
        h.append(info)
        dec_hand = (self.card_decoder.fp_hand if hand_code == '45'
                    else self.card_decoder.sa_hand)
        if not any(str(c.get('piece_id')) == str(piece_id)
                   for c in dec_hand):
            dec_hand.append(info)
        _who = 'FP' if hand_code == '45' else 'SA'
        self.log(f"{_who} card RETRACTED back to hand: {name}")

    def _handle_card_zone_discard(self, piece_id: str, hand_code: str,
                                  deck_return=None):
        """Face-down card exit from a hand zone = discard. Decode via
        the COMMITMENT brute-force (works for opening-hand cards — the
        drawn-list lookup returned Unknown for dealt cards and dropped
        them).  EDGE-2 (2026-08-17): deck_return names a deck zone
        the card goes BACK to (the un-owed draw refusal corrective) —
        same hand-exit bookkeeping, deck count restored, different
        log line."""
        side = Side.FREE_PEOPLES if hand_code == '45' else Side.SHADOW
        pid = int(piece_id)
        if hand_code == '45':
            dt = 'FP Character' if _FP_PID_TYPES[pid - 106] == 'C' \
                else 'FP Strategy'
        else:
            dt = 'SA Character' if _SA_PID_TYPES[pid - 158] == 'C' \
                else 'SA Strategy'
        num = self.card_decoder.decode_piece(pid, dt)
        name = self.card_decoder.get_card_name(num, dt)
        # keep decoder + state hands in sync (best effort)
        dec_hand = (self.card_decoder.fp_hand if hand_code == '45'
                    else self.card_decoder.sa_hand)
        for i, c in enumerate(dec_hand):
            if str(c.get('piece_id')) == str(piece_id):
                dec_hand.pop(i)
                break
        h = self.state.hands[side].hand
        for match in (
                lambda e: isinstance(e, dict)
                and str(e.get('piece_id')) == str(piece_id),
                lambda e: isinstance(e, dict)
                and str(e.get('card_name', '')) == str(name),
                lambda e: isinstance(e, str) and e.startswith('_unknown_')):
            hit = next((i for i, e in enumerate(h) if match(e)), None)
            if hit is not None:
                h.pop(hit)
                break
        _who = 'FP' if hand_code == '45' else 'SA'
        if deck_return:
            # restore the deck count the draw decremented
            try:
                _dk = self.state.decks
                if deck_return == 'FP Char Deck':
                    _dk.fp_character += 1
                elif deck_return == 'FP Strat Deck':
                    _dk.fp_strategy += 1
                elif deck_return == 'SA Char Deck':
                    _dk.sh_character += 1
                elif deck_return == 'SA Strat Deck':
                    _dk.sh_strategy += 1
            except Exception:
                pass
            self.log(f"{_who} card returned to {deck_return}: {name}")
        else:
            self.log(f"{_who} discards (face-down): {name}")

    def _handle_card_draw(self, deck: str):
        """Handle card draw (fallback method)."""
        if 'FP' in deck:
            side = Side.FREE_PEOPLES
            if 'Char' in deck:
                self.state.decks.fp_character = max(0, self.state.decks.fp_character - 1)
                deck_type = 'Character'
            else:
                self.state.decks.fp_strategy = max(0, self.state.decks.fp_strategy - 1)
                deck_type = 'Strategy'
        else:
            side = Side.SHADOW
            if 'Char' in deck:
                self.state.decks.sh_character = max(0, self.state.decks.sh_character - 1)
                deck_type = 'Character'
            else:
                self.state.decks.sh_strategy = max(0, self.state.decks.sh_strategy - 1)
                deck_type = 'Strategy'

        self.state.hands[side].hand.append({'deck_type': deck_type, 'card_name': 'Unknown'})
        side_name = "FP" if side == Side.FREE_PEOPLES else "SA"
        self.log(f"{side_name} drew {deck_type}")

    def _handle_card_reveal(self, card_num: int, is_fp: bool, piece_id: str = None,
                            src_zone: str = None):
        """Handle card being played/revealed.

        Uses piece_id to determine Character vs Strategy deck via _FP_PID_TYPES / _SA_PID_TYPES.
        This fixes the bug where SA Character #10 ("Orc Patrol") was decoded as
        SA Strategy #10 ("Return to Valinor") because both dicts have entries for all card_nums.

        `src_zone` is the reveal's SOURCE zone (from Controls.867).  A
        reveal out of a BOARD/TABLE zone (Game.*) is an ON-TABLE card
        being used, NOT a hand play -- see the from_hand gate below.
        """
        # Determine deck type from piece_id using PID type arrays
        deck_type = None
        if piece_id:
            try:
                pid = int(piece_id)
                if is_fp and 106 <= pid <= 157:
                    ptype = _FP_PID_TYPES[pid - 106]
                    deck_type = 'Character' if ptype == 'C' else 'Strategy'
                elif not is_fp and 158 <= pid <= 209:
                    ptype = _SA_PID_TYPES[pid - 158]
                    deck_type = 'Character' if ptype == 'C' else 'Strategy'
            except (ValueError, IndexError):
                pass

        if is_fp:
            side = Side.FREE_PEOPLES
            if deck_type == 'Character':
                card_name = FP_CHARACTER_CARDS.get(card_num, f"FP Character #{card_num}")
            elif deck_type == 'Strategy':
                card_name = FP_STRATEGY_CARDS.get(card_num, f"FP Strategy #{card_num}")
            else:
                # Fallback: try both (only if piece_id unavailable)
                card_name = FP_CHARACTER_CARDS.get(card_num) or FP_STRATEGY_CARDS.get(card_num) or f"FP Card #{card_num}"
        else:
            side = Side.SHADOW
            if deck_type == 'Character':
                card_name = SA_CHARACTER_CARDS.get(card_num, f"SA Character #{card_num}")
            elif deck_type == 'Strategy':
                card_name = SA_STRATEGY_CARDS.get(card_num, f"SA Strategy #{card_num}")
            else:
                card_name = SA_CHARACTER_CARDS.get(card_num) or SA_STRATEGY_CARDS.get(card_num) or f"SA Card #{card_num}"

        # Update card decoder
        if piece_id:
            self.card_decoder.handle_reveal(piece_id, card_num, is_fp)

        # Remove a card from hand.
        #
        # ROOT-CAUSE FIX (2026-08-15, auto35 census break): the reveal's
        # SOURCE zone decides whether anything leaves the hand.  Hand
        # zones are Controls.* (FP Controls.45 / SH Controls.46); a
        # Game.* source means the card was ALREADY on the board/table --
        # e.g. the 1.149b on-table Balrog of Moria discarded to play its
        # combat side (ai.py combat_card_play from_table), whose reveal
        # reads '<~Controls.867~><~Game.2034~>'.  Parsing that as a hand
        # play matched no pid and no name, fell through to the blind
        # `hand.pop(0)` below and deleted an UNRELATED card from the
        # tracked hand; every later genuine play of that phantom-deleted
        # card then missed again and popped the NEXT one, walking a
        # permanent one-card deficit through the whole hand (auto35:
        # Flocks of Crebain -> Wormtongue -> The Lidless Eye -> Cruel
        # Weather, 533 census ticks of 'hand SH: engine-only=[...]').
        # The ENGINE was right throughout; this tracked hand was wrong.
        _from_hand9 = not (src_zone or '').startswith('Game.')
        # CROSS-SIDE GUARD (2026-08-15, auto36 break): `side` here comes
        # from marker sniffing (Controls.45/884/885 in the line).  A
        # reveal sourced from a BOARD zone carries no hand marker, so an
        # FP commitment (`reveal 156 ... <~Controls.862~><~Game.2037~>`,
        # Current FP Card) sniffed as SHADOW and blind-popped pid 183 —
        # The Day Without Dawn — out of the SHADOW hand.  Piece-id RANGE
        # is authoritative (106-157 FP, 158-209 SH) and is already what
        # this method trusts for deck typing: never let a reveal touch
        # the hand of the side its own pid contradicts.
        if _from_hand9 and piece_id:
            try:
                _p9 = int(piece_id)
            except (TypeError, ValueError):
                _p9 = None
            if _p9 is not None and (106 <= _p9 <= 209):
                _pid_is_fp9 = _p9 <= 157
                if _pid_is_fp9 != (side == Side.FREE_PEOPLES):
                    _from_hand9 = False
                    self.log("  [hand-guard] reveal pid=%s is %s but the "
                             "line sniffed as %s - hand untouched"
                             % (piece_id, 'FP' if _pid_is_fp9 else 'SH',
                                'FP' if side == Side.FREE_PEOPLES else 'SH'))
        hand = self.state.hands[side].hand
        if hand and _from_hand9:
            # Try to find the specific card by piece_id
            found = False
            if piece_id:
                for i, card in enumerate(hand):
                    if isinstance(card, dict) and card.get('piece_id') == piece_id:
                        hand.pop(i)
                        found = True
                        break
            if not found:
                # VASSAL re-creates card pieces on reveal with FRESH ids
                # (2023-12-06: Eomer drawn as pid 106, revealed as 143) —
                # match the decoded NAME so the wrong card isn't popped
                # (pop(0) offsets cascaded the tracked hand all game).
                # Hands may hold dicts OR card-id strings (the extraction
                # harness syncs them via ai._sync_hands_to_card_ids).
                want = _normalize_name(card_name)
                for i, card in enumerate(hand):
                    if isinstance(card, dict) \
                            and card.get('card_name') == card_name:
                        hand.pop(i)
                        found = True
                        break
                    if isinstance(card, str) and want \
                            and not card.startswith('_unknown_') \
                            and want in _normalize_name(card):
                        hand.pop(i)
                        found = True
                        break
            if not found and hand:
                # Last resort: burn an unknown placeholder before a
                # known different card
                for i, card in enumerate(hand):
                    if isinstance(card, str) \
                            and card.startswith('_unknown_'):
                        hand.pop(i)
                        found = True
                        break
                if not found:
                    # LOUD (2026-08-15): this fallback pops an ARBITRARY
                    # card and is the cascade engine described above.
                    # It is kept only as a hand-SIZE correction for a
                    # genuine hand play whose identity we failed to
                    # decode; if it ever fires again the log names the
                    # victim so the class is visible instead of silent.
                    _victim9 = hand[0]
                    hand.pop(0)
                    self.log("  [hand-drift] reveal pid=%s (%s) matched "
                             "no tracked hand card - blind-popped %r"
                             % (piece_id, card_name, _victim9))

        side_name = "FP" if is_fp else "SA"
        self.log(f"{side_name} plays: {card_name}")
        self._record_action('play_event', {'card_name': card_name}, side)
        # Cards whose placements are FIXED by the card text — subsequent
        # musters/moves are effects, not decisions (flagged auto so
        # extraction excludes them from policy training)
        _FIXED_EFFECT_CARDS = {
            'A New Power is Rising',
        }
        self._fixed_effect_card = (card_name in _FIXED_EFFECT_CARDS)
        # Open the card-effect window: die-less moves/placements that
        # follow are effects of this card, not standalone decisions
        self._card_window_side = side
        self._card_window_card = card_name
        # a Lure pending never survives into the NEXT play (undo /
        # retraction safety: a stale pending must not mislabel a later
        # unrelated kill as the Lure eliminate)
        self._lure_pending = None
        # Revealing cards: a following FP pawn walk is the reveal
        # PLACEMENT (the marker flip often lands after the walk)
        if _normalize_name(card_name) in _REVEALING_EVENT_CARDS:
            self._hunt_reveal_pending = True

    def _handle_kill_event(self, pieces: List[str], to_loc: str):
        """Handle Controls.644 kill/cleanup events.

        These send pieces to "Used" area. Handles companion death, token
        cleanup, piece tracking updates, control-marker cleanup, AND army
        unit kills. The raw command carries only piece ids + destination
        (no source region), so unit kills resolve via self.piece_region;
        units whose tracked location is off-map were already removed by a
        preceding move and are skipped.
        """
        for piece_id in pieces:
            # Lure of the Ring eliminate EVIDENCE: the drawn victim's
            # pid is killed with the card cleanup (ketchuptime T2
            # L159-162 Gimli, chat 'kil') -> NOW the elimination is
            # proven; emit the companion_death the draw itself must
            # never fabricate (the corruption arm returns the piece
            # and clears _lure_pending at the corruption walk).
            _lp9 = getattr(self, '_lure_pending', None)
            if _lp9 is not None and piece_id == _lp9.get('pid'):
                self._record_action(
                    'companion_death',
                    {'companion': _lp9['companion'],
                     'card_effect': True}, Side.FREE_PEOPLES)
                self.log(f"Lure resolved as ELIMINATE: "
                         f"{_lp9['companion']}")
                self._lure_pending = None
                self._set_piece_zone(piece_id, to_loc, dead=True)
                continue
            # Control markers cleaned up = control released
            if piece_id in CONTROL_MARKERS:
                side = (Side.SHADOW if piece_id in SA_CONTROL_MARKERS
                        else Side.FREE_PEOPLES)
                prev = self.piece_region.get(piece_id)
                if prev:
                    base = stronghold_base_region(prev)
                    san = sanitize(base if base is not None else prev)
                    if san in board.regions:
                        self.state.enemy_controlled[side].discard(san)
                        # RB-08 (1.55a): occupation re-check — sole
                        # occupation by the capturer keeps control
                        self._update_region_control(san)
                        self.log(f"Control marker killed: {side.name} off {prev}")
                self.piece_region[piece_id] = to_loc
                continue
            if piece_id in VP_MARKERS:
                # t138 (triage pass-6 3.2, Q8 wrinkle): endgame rituals
                # also KILL the VP marker (Bomber/jzw27 L689) — same
                # negative ruling 1.165 class; stays by-design
                # divergent, flagged for curation
                self.log(f"CURATION NOTE (Q8/1.165): kill grammar "
                         f"aimed at VP marker {piece_id} (marker "
                         f"ritual, by design)")
                continue
            # Hunt tiles killed/cleaned up = discarded to the used pile
            # (t125 fix 2; the client renders them in Used Hunt Tiles:
            # Wirbelwind L475 kills tile 228 -> dump shows 'used hunt
            # tiles')
            if piece_id in HUNT_TILE_PIECES or piece_id in ('230', '231'):
                self._tile_to_zone(piece_id, 'Used Hunt Tiles')
                continue
            # Companion death
            comp_name = COMPANIONS.get(piece_id)
            if comp_name:
                _in_fell = False
                for i, comp in enumerate(self.state.fellowship.companions):
                    if comp.name == comp_name:
                        _in_fell = True
                        was_guide = (self.state.fellowship.guide is not None and
                                     self.state.fellowship.guide.name == comp_name)
                        self.state.fellowship.companions.pop(i)
                        if was_guide:
                            self.state.fellowship.guide = None
                        self.log(f"Kill event: companion {comp_name} removed")
                        if comp_name == 'Gandalf the Grey':
                            self._gandalf_dead = True
                        # A companion killed while hunt damage is
                        # unresolved = the FP hunt-damage decision
                        ctx = self._hunt_ctx
                        if ctx is not None and not ctx.get('resolved'):
                            ctx['resolved'] = True
                            self._hunt_ctx = None
                            self._record_action(
                                'hunt_sacrifice_guide' if was_guide
                                else 'hunt_sacrifice_random',
                                {'companion': comp_name}, Side.FREE_PEOPLES)
                        break
                if not _in_fell:
                    # t133 fix F10: a SEPARATED/placed companion listed
                    # in the cleanup kill (army destroyed) dies with
                    # his army (rulebook p.29: "all Leaders (including
                    # Characters) ... are also immediately removed from
                    # play") — resolve the object by alias from its
                    # tracked zone (region, stronghold box, or staged
                    # at a card-play area).
                    prev = self.piece_region.get(piece_id)
                    _b9 = stronghold_base_region(str(prev)) if prev else None
                    _pref = None
                    if _b9:
                        _pref = sanitize(_b9)
                    elif prev and is_map_region(str(prev)):
                        _pref = sanitize(str(prev))
                    self._kill_named_piece(piece_id, 'Dead',
                                           prefer_san=_pref)
                    # t138: a separated Gandalf killed here still arms
                    # the GtW arrival gate (the in-fellowship branch
                    # above already did; the F10 branch never set it)
                    if comp_name == 'Gandalf the Grey':
                        self._gandalf_dead = True
                continue

            # Token cleanup
            if piece_id == TOKEN_POLITICAL:
                self.state.fp_political_token = False
                continue
            if piece_id == TOKEN_CARD_DRAW:
                self.state.fp_card_token = False
                continue

            # Action-die pieces: the 644-confirm may be the die's only
            # 'used' signal (its Scribe usage drag can be retracted
            # noise, 2023-12-06 T5 die 103).  Mark used WITHOUT touching
            # the pending-die window; never treat dice as units.
            if self._is_die_piece(piece_id):
                if 'Used' in str(to_loc) or 'Current' in str(to_loc):
                    mp = self.die_id_map.get(int(piece_id))
                    if mp is not None:
                        sd_, ix_ = mp
                        if ix_ < len(self.state.dice[sd_]):
                            d_ = self.state.dice[sd_][ix_]
                            if not d_.used:
                                d_.used = True
                                self.log(f"Die {piece_id} marked used "
                                         f"(kill-event confirm)")
                self.piece_region[piece_id] = to_loc
                continue

            # Army units killed via Controls.644 (e.g. garrison kills):
            # resolve the source region from tracked piece locations.
            # Pieces tracked off-map (dice/cards/already-removed units)
            # fall through to plain tracking updates.
            nation, unit_type = get_unit_type_from_piece_id(
                piece_id, self.piece_types)
            if nation is not None and unit_type is not None:
                prev = self.piece_region.get(piece_id)
                region = None
                if prev:
                    region = stronghold_base_region(prev) or (
                        prev if is_map_region(prev) else None)
                if region is not None:
                    _card_fx = (self._card_window_side is not None
                                and self._battle_synth_region is None)
                    self._handle_battle_casualties(
                        [piece_id], region, synth_attack=not _card_fx)
                # t125: the 644 command's destination token is Game.2035
                # 'Used', but the CLIENT renders killed army units in
                # their DEAD pile (Adobim pid 065 / Rizino1 086 dumps:
                # 'fp dead' while the parser froze them at 'Used')
                self._set_piece_zone(piece_id, 'Dead', dead=True)
                continue

            # Update piece tracking for any piece we know about
            self.piece_region[piece_id] = to_loc

    def _handle_battle_casualties(self, pieces: List[str], loc: str,
                                  synth_attack: bool = True):
        """Handle Controls.296: remove battle casualties from army at location."""
        # Control/VP markers and dice 'killed' at a region are
        # bookkeeping, never casualties — and must not synthesize a
        # battle (2023-12-06 line 747: FP control marker 410 killed at
        # Rivendell Stronghold synthesized a phantom '?' attack)
        _marker_ps = [p for p in pieces
                      if p in CONTROL_MARKERS or p in VP_MARKERS]
        for p in _marker_ps:
            if p in CONTROL_MARKERS:
                m_side = (Side.SHADOW if p in SA_CONTROL_MARKERS
                          else Side.FREE_PEOPLES)
                base = stronghold_base_region(loc) or loc
                san = sanitize(base)
                if san in board.regions:
                    self.state.enemy_controlled[m_side].discard(san)
                    # RB-08 (1.55a): occupation re-check
                    self._update_region_control(san)
                    self.log(f"Control marker removed: {m_side.name} "
                             f"off {loc}")
        pieces = [p for p in pieces
                  if p not in _marker_ps and not self._is_die_piece(p)]
        # t133: capture tracked zones BEFORE the dead-marking loop so
        # named-piece kills can resolve their true source army
        _prev_zone9 = {p: self.piece_region.get(p) for p in pieces}
        _comp_leader_kills9 = 0
        # ruling 1.162 (F11): pids already tracked in a dead zone were
        # killed by an earlier application — the re-kill is a client
        # no-op; count-based removal would over-kill the army
        _rekill9 = [p for p in pieces
                    if p not in COMPANIONS
                    and self._pid_zone_dead(_prev_zone9.get(p))]
        if _rekill9:
            self.log(f"Re-kill no-op (ruling 1.162): {_rekill9} "
                     f"already dead")
            pieces = [p for p in pieces if p not in _rekill9]
            if not pieces:
                return
        # Casualties leave the board: without this, piece_region kept
        # dead pieces AT the region and the bridge emitter re-selected
        # them for later moves (a killed regular marched off Osgiliath
        # while the engine had it dead).  'Dead' is not a map region, so
        # region listings exclude it; zone inversion falls to defaults.
        # t125 fix 1: routed through _set_piece_zone so FP kills feed
        # state.fp_casualties (ai.py muster availability).
        for p in pieces:
            self._set_piece_zone(p, 'Dead', dead=True)
        # Companions killed while IN THE FELLOWSHIP are fellowship
        # casualties, not army casualties (2023-03-15 line 77: the
        # Controls.296 kill of Gandalf 240 in the Guide Box Game.1999 =
        # the T1 hunt guide sacrifice, chat 'l lose gandalf').  Mirrors
        # the kill-event companion handling; SEPARATED companions are
        # eliminated BY IDENTITY from their tracked army (t133 fix
        # F10 — they used to fall through to the count-based army path
        # and survive whenever the region lookup failed).
        _rest = []
        for p in pieces:
            comp_name = COMPANIONS.get(p)
            if not comp_name or not any(
                    c.name == comp_name
                    for c in self.state.fellowship.companions):
                if comp_name:
                    prev = _prev_zone9.get(p)
                    _b9 = stronghold_base_region(str(prev)) \
                        if prev else None
                    _pref = None
                    if _b9:
                        _pref = sanitize(_b9)
                    elif prev and is_map_region(str(prev)):
                        _pref = sanitize(str(prev))
                    elif is_map_region(loc):
                        _pref = sanitize(loc)
                    if self._kill_named_piece(p, 'Dead',
                                              prefer_san=_pref):
                        _comp_leader_kills9 += 1
                    continue
                _rest.append(p)
                continue
            for i, comp in enumerate(self.state.fellowship.companions):
                if comp.name == comp_name:
                    was_guide = (self.state.fellowship.guide is not None and
                                 self.state.fellowship.guide.name == comp_name)
                    self.state.fellowship.companions.pop(i)
                    if was_guide:
                        self.state.fellowship.guide = None
                    self.log(f"Companion casualty: {comp_name} removed "
                             f"from fellowship")
                    if comp_name == 'Gandalf the Grey':
                        self._gandalf_dead = True
                    ctx = self._hunt_ctx
                    if ctx is not None and not ctx.get('resolved'):
                        ctx['resolved'] = True
                        self._hunt_ctx = None
                        self._record_action(
                            'hunt_sacrifice_guide' if was_guide
                            else 'hunt_sacrifice_random',
                            {'companion': comp_name}, Side.FREE_PEOPLES)
                    break
        pieces = _rest
        if not pieces:
            if _comp_leader_kills9:
                # lone separated-companion kill: keep the OLD record
                # shape (synth + combat_casualties leader count) so the
                # schedule is unchanged where the old name-match at the
                # kill region used to succeed
                if synth_attack:
                    self._maybe_synth_attack(loc)
                removed = {'regular': 0, 'elite': 0,
                           'leader': _comp_leader_kills9, 'nazgul': 0}
                self.log(f"Battle casualties at {loc}: "
                         f"{_comp_leader_kills9}L")
                self._record_action('combat_casualties', {
                    'region': loc, 'removed': removed,
                    'side': Side.SHADOW.name}, Side.SHADOW)
                self._update_region_control(loc)
                self._combat_casualties_this_round = True
            return
        # Synthesize the attack decision BEFORE removing units — target
        # validation must see the defender's units still in the region.
        # (synth_attack=False for card-effect removals: no battle exists.)
        if synth_attack:
            self._maybe_synth_attack(loc)
        loc_san = sanitize(loc)

        src_key = None
        for key in self.state.armies:
            if sanitize(key) == loc_san:
                src_key = key
                break

        if not src_key or src_key not in self.state.armies:
            return

        army = self.state.armies[src_key]
        removed = {'regular': 0, 'elite': 0, 'leader': 0, 'nazgul': 0}
        # Per-SIDE partition for the casualty MARKS (failure mode #5:
        # a mixed 296 line must not pool both sides' hits into one
        # mark; the single combat_casualties record keeps its
        # historical shape for schedule parity).
        _rm_side9 = {
            Side.SHADOW: {'regular': 0, 'elite': 0, 'leader': 0,
                          'nazgul': 0},
            Side.FREE_PEOPLES: {'regular': 0, 'elite': 0, 'leader': 0,
                                'nazgul': 0}}

        def _tally9(is_shadow, kind):
            _rm_side9[Side.SHADOW if is_shadow
                      else Side.FREE_PEOPLES][kind] += 1

        def _fallback_same_side(is_piece_shadow):
            """Piece-identity mismatch fallback: remove ANY unit of the
            piece's side, then any leader of that side. Same-side ONLY —
            popping the other side's units would kill besiegers in mixed
            siege armies. The side's casualty count is what matters."""
            for i, u in enumerate(army.units):
                if (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                    u = army.units.pop(i)
                    _k9 = ('elite' if u.kind == UnitKind.ELITE
                           else 'regular')
                    removed[_k9] += 1
                    _tally9(is_piece_shadow, _k9)
                    return True
            shadow_minions = {'Witch King', 'Saruman', 'Mouth of Sauron'}
            for i, l in enumerate(army.leaders):
                l_shadow = (isinstance(l, Nazgul)
                            or getattr(l, 'name', '') in shadow_minions
                            or (hasattr(l, 'nation')
                                and l.nation in SHADOW_NATIONS))
                if l_shadow == is_piece_shadow:
                    army.leaders.pop(i)
                    _k9 = ('nazgul' if isinstance(l, Nazgul)
                           else 'leader')
                    removed[_k9] += 1
                    _tally9(is_piece_shadow, _k9)
                    return True
            return False

        for piece_id in pieces:
            # Handle minion pieces
            minion_name = MINION_PIECES.get(piece_id)
            if minion_name:
                for i, l in enumerate(army.leaders):
                    if hasattr(l, 'name') and l.name == minion_name:
                        army.leaders.pop(i)
                        removed['leader'] += 1
                        _tally9(True, 'leader')
                        break
                continue

            # Handle companion pieces
            comp_name = COMPANIONS.get(piece_id)
            if comp_name:
                for i, l in enumerate(army.leaders):
                    if hasattr(l, 'name') and l.name == comp_name:
                        army.leaders.pop(i)
                        removed['leader'] += 1
                        _tally9(False, 'leader')
                        break
                continue

            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)
            is_piece_shadow = nation in SHADOW_NATIONS if nation else False
            # NATION-EXACT casualty tally (2026-07-25, combat kills)
            if unit_type in (UnitKind.REGULAR, UnitKind.ELITE) \
                    and nation is not None:
                _rnk9 = '%s|%s' % (
                    getattr(nation, 'value', str(nation)),
                    'elite' if unit_type == UnitKind.ELITE
                    else 'regular')
                _rbn9 = removed.setdefault('_by_nation', {})
                _rbn9[_rnk9] = _rbn9.get(_rnk9, 0) + 1

            if unit_type == 'LEADER':
                for i, l in enumerate(army.leaders):
                    if not isinstance(l, Nazgul) and hasattr(l, 'nation') and l.nation == nation:
                        army.leaders.pop(i)
                        removed['leader'] += 1
                        _tally9(is_piece_shadow, 'leader')
                        break
                else:
                    for i, l in enumerate(army.leaders):
                        if not isinstance(l, Nazgul):
                            army.leaders.pop(i)
                            removed['leader'] += 1
                            _tally9(is_piece_shadow, 'leader')
                            break
            elif unit_type == 'NAZGUL':
                for i, l in enumerate(army.leaders):
                    if isinstance(l, Nazgul):
                        army.leaders.pop(i)
                        removed['nazgul'] += 1
                        _tally9(True, 'nazgul')
                        break
            elif unit_type == UnitKind.ELITE:
                found = False
                if nation:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.ELITE and u.nation == nation:
                            army.units.pop(i)
                            removed['elite'] += 1
                            _tally9(is_piece_shadow, 'elite')
                            found = True
                            break
                if not found:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.ELITE and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            army.units.pop(i)
                            removed['elite'] += 1
                            _tally9(is_piece_shadow, 'elite')
                            found = True
                            break
                # Piece identity mismatch (e.g. elite-typed pieces used
                # for a regular-only garrison): same-side fallback
                if not found:
                    _fallback_same_side(is_piece_shadow)
            else:
                # Regular or unknown
                found = False
                if nation:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.REGULAR and u.nation == nation:
                            army.units.pop(i)
                            removed['regular'] += 1
                            _tally9(is_piece_shadow, 'regular')
                            found = True
                            break
                if not found:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.REGULAR and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            army.units.pop(i)
                            removed['regular'] += 1
                            _tally9(is_piece_shadow, 'regular')
                            found = True
                            break
                # Fallback: promoted unit (Regular->Elite via swap) or
                # identity-mismatched piece — same-side unit, then leader
                # (never the other side's units; the old blind pop(0)
                # could kill besiegers out of mixed siege armies)
                if not found:
                    _fallback_same_side(is_piece_shadow)

        # Clean up empty army
        if not army.units and not army.leaders:
            del self.state.armies[src_key]

        # t133: separated companions killed BY IDENTITY above still
        # count as leader casualties in the record (parity with the old
        # in-loop companion pop)
        removed['leader'] += _comp_leader_kills9

        # Log
        details = []
        if removed['regular']:
            details.append(f"{removed['regular']}R")
        if removed['elite']:
            details.append(f"{removed['elite']}E")
        if removed['leader']:
            details.append(f"{removed['leader']}L")
        if removed['nazgul']:
            details.append(f"{removed['nazgul']}N")
        if any(removed.values()):
            self.log(f"Battle casualties at {loc}: {'+'.join(details)}")
            # (attack synthesized at handler entry, pre-removal)
            # Determine side of casualties
            casualty_side = Side.SHADOW
            for piece_id in pieces:
                nation, _ = get_unit_type_from_piece_id(piece_id, self.piece_types)
                if nation and nation not in SHADOW_NATIONS:
                    casualty_side = Side.FREE_PEOPLES
                    break
            self._record_action('combat_casualties', {
                'region': loc, 'removed': removed, 'side': casualty_side.name
            }, casualty_side)
            # v3: this kill-path (Controls.296/281/644) never emitted
            # casualty marks — whole sides' hits were invisible to the
            # per-round forcing (2023-11-09 AngryBird T8 L714: the FP
            # besieger loss at Minas Morgul had no mark)
            if self._battle_synth_region is not None:
                # Per-SIDE marks: a mixed 296 line pooled into one mark
                # over-forced one side's round window and hid the
                # other's hits (failure mode #5); the _by_nation split
                # keys each nation tally to its own side.
                _bn9 = dict(removed.get('_by_nation') or {})
                for _msd9, _mrm9 in _rm_side9.items():
                    if not any(_mrm9.values()):
                        continue
                    _bn_side9 = {}
                    for _bk9, _bv9 in _bn9.items():
                        _bnat9 = _bk9.split('|', 1)[0]
                        _b_is_sh9 = any(
                            getattr(_n9, 'value', '') == _bnat9
                            for _n9 in SHADOW_NATIONS)
                        if (_msd9 == Side.SHADOW) == _b_is_sh9:
                            _bn_side9[_bk9] = _bv9
                    self._emit_casualty_mark(
                        _msd9, loc_san,
                        _mrm9['regular'] + 2 * _mrm9['elite'],
                        self._mark_cause(loc_san),
                        spec={'kill_regular': _mrm9['regular'],
                              'kill_elite': _mrm9['elite'],
                              'downgrade': 0,
                              '_by_nation': _bn_side9})

        # Update region control after casualties
        self._update_region_control(loc)

        # Track that casualties occurred this combat round (for siege press detection)
        self._combat_casualties_this_round = True

    def _handle_piece_swap_remove(self, pieces: List[str], loc: str):
        """Handle Controls.304: remove pieces from army during battle swap (promotion/demotion)."""
        # Garrison downgrades are reported in the stronghold BOX
        # ("kills [2]Gondor Elite in Minas Tirith Stronghold") — map them
        # to the base region where the units actually live.
        _box_base = stronghold_base_region(loc)
        in_stronghold_box = _box_base is not None
        if in_stronghold_box:
            loc = _box_base
        elif not is_map_region(loc):
            return
        loc_san = sanitize(loc)

        src_key = None
        for key in self.state.armies:
            if sanitize(key) == loc_san:
                src_key = key
                break

        if not src_key or src_key not in self.state.armies:
            return

        army = self.state.armies[src_key]
        for piece_id in pieces:
            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)
            is_piece_shadow = nation in SHADOW_NATIONS if nation else False

            # t125 fix 1: the Controls.304 'kills' removal physically
            # sends the piece to its dead pile (client dumps show the
            # downgraded elites in FP Dead — GenericPie 344/564/750/
            # 1172/1196); the parser left piece_region stale at the
            # region, so dead-pile censuses under-counted all game.
            self._set_piece_zone(piece_id, 'Dead', dead=True)

            if unit_type == 'LEADER':
                for i, l in enumerate(army.leaders):
                    if not isinstance(l, Nazgul) and hasattr(l, 'nation') and l.nation == nation:
                        army.leaders.pop(i)
                        break
                else:
                    for i, l in enumerate(army.leaders):
                        if not isinstance(l, Nazgul):
                            army.leaders.pop(i)
                            break
            elif unit_type == UnitKind.ELITE:
                found = False
                if nation:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.ELITE and u.nation == nation:
                            army.units.pop(i)
                            found = True
                            break
                if not found:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.ELITE and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            army.units.pop(i)
                            break
            else:
                # Regular or unknown
                found = False
                if nation:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.REGULAR and u.nation == nation:
                            army.units.pop(i)
                            found = True
                            break
                if not found:
                    for i, u in enumerate(army.units):
                        if u.kind == UnitKind.REGULAR and (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            army.units.pop(i)
                            found = True
                            break
                # Fallback: unit may have been promoted via swap
                if not found:
                    for i, u in enumerate(army.units):
                        if (u.nation in SHADOW_NATIONS) == is_piece_shadow:
                            army.units.pop(i)
                            break

        # Update region control after battle casualties
        self._update_region_control(loc)

        # Track what was removed for siege press detection. Garrison
        # downgrades inside the stronghold box are the DEFENDER's casualty
        # choice — never an attacker press — so they must not arm it.
        if in_stronghold_box:
            self._last_swap_remove_elite = False
            self._last_swap_remove_side = None
            self._last_swap_remove_nation = None
            # RC4 sequel: a garrison elite downgraded in the box = one
            # applied hit (casualty choice) — mark it for the per-round
            # hit forcing (see _remove_pieces_from_armies)
            if self._battle_synth_region is not None:
                for piece_id in pieces:
                    _n9, _k9 = get_unit_type_from_piece_id(
                        piece_id, self.piece_types)
                    if _k9 == UnitKind.ELITE and _n9 is not None:
                        self._emit_casualty_mark(
                            Side.SHADOW if _n9 in SHADOW_NATIONS
                            else Side.FREE_PEOPLES,
                            loc_san, 1, self._mark_cause(loc_san),
                            spec={'kill_regular': 0, 'kill_elite': 0,
                                  'downgrade': 1})
        else:
            for piece_id in pieces:
                nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)
                self._last_swap_remove_elite = (unit_type == UnitKind.ELITE)
                is_shadow = nation in SHADOW_NATIONS if nation else False
                self._last_swap_remove_side = Side.SHADOW if is_shadow else Side.FREE_PEOPLES
                # fix 76 (GAME19 T8): the press PAYMENT nation is log
                # evidence — carry it so the teacher downgrades the exact
                # elite the human paid with (engine default = first elite
                # in list order, wrong nation half the time)
                self._last_swap_remove_nation = nation
            # 129 v2 (2026-07-25 Vayze T8 Dale L797): a CHAT-ONLY battle
            # (both sides 'nc', no explicit kills before the swap) never
            # synthesized its attack, so _battle_synth_region stayed
            # None and the downgrade-pair casualty mark below was gated
            # off -> v5 forced a ZERO defender spec and the dead elite
            # survived, blocking the victor advance.  Mirror the
            # Controls.296 kill path: an ELITE swap-removal with combat
            # rolls already thrown in this die window and no card
            # window open is battle evidence - synthesize here (the
            # synth's own hint/enemy validation names the region).
            # Emitting at the swap line (not the nc chat) keeps record
            # order stable for battles that synthesize normally.
            if (self._battle_synth_region is None
                    and self._card_window_side is None
                    and getattr(self, '_rolls_since_die', False)
                    and self._last_swap_remove_elite):
                self._maybe_synth_attack(loc)
        # Outside a battle window this removal may be the first half of
        # a MUSTER UPGRADE pair (regulars out, elites in — see
        # _handle_piece_swap_place)
        if self._battle_synth_region is None:
            self._pending_304 = (self.line_num, sanitize(loc),
                                 len(pieces))

    def _handle_piece_swap_place(self, pieces: List[str], swap_type: str, loc: str):
        """Handle Controls.313: place replacement pieces during battle swap.
        swap_type: '307'=Elite, '308'=Regular, '309'=Special
        """
        # Garrison downgrade returns arrive in the stronghold BOX ("moves
        # [2]Gondor Regular from casualties to Minas Tirith Stronghold")
        _box_base = stronghold_base_region(loc)
        in_stronghold_box = _box_base is not None
        if in_stronghold_box:
            loc = _box_base
        elif not is_map_region(loc):
            return
        loc_san = sanitize(loc)

        # explicit region-validity gate (the old bare-except around an
        # unused board.region() call swallowed unrelated errors too)
        if loc_san not in board.regions:
            return

        # SANITIZED lookup (duplicate-army-key bug): the region already
        # holds an army under a differently-cased key ("Helm's Deep"
        # from decode_location vs orig_name) — an exact-key existence
        # check creates a SECOND entry and every sanitized-first-match
        # scan (composition counts, control updates) sees only one of
        # them (2025-04-06 demanray T16: press downgrades placed into a
        # phantom second Helm's Deep entry, the besieger count hit 0 and
        # the siege dissolved mid-battle).
        to_key, army = self._find_or_create_army(loc)
        _placed_counts = {'regular': 0, 'elite': 0, 'leader': 0,
                          'nazgul': 0}
        _placed_pieces9 = []   # (kind, nation) per placed piece

        # Determine nation from region context
        region_meta = board.regions.get(loc_san, {})
        parent_id = region_meta.get("parent_id")
        faction = board.fid_to_name.get(parent_id, "").lower() if parent_id else ""

        for piece_id in pieces:
            # t125 fix 1: _set_piece_zone refunds the replay-side FP
            # dead pool when a counted piece returns to the board
            # (Controls.308 'from casualties' downgrade returns — the
            # old inline refund is subsumed piece-exactly).
            self._set_piece_zone(piece_id, to_key)
            nation, unit_type = get_unit_type_from_piece_id(piece_id, self.piece_types)

            # Override unit type from swap_type if piece type is unknown.
            # W8 B1 fix (2026-07-20): swap_type 307/308 are SOURCE pools
            # ('from reinforcements' / 'from casualties'), NOT unit
            # kinds — the old 307->ELITE mapping was wrong. Source
            # doesn't determine kind; default to REGULAR (dormant: every
            # placed pid resolved to a known type across 2000 logs, so
            # this fallback never fires — fixed to prevent future drift).
            if unit_type is None:
                unit_type = UnitKind.REGULAR

            # Determine nation from existing army if not known from piece
            if nation is None:
                # Use the nation of existing units in the army
                for u in army.units:
                    if (u.nation in SHADOW_NATIONS) == ('SA' in loc or faction in ('isengard', 'sauron', 'southrons and easterlings', 'southrons', 'easterlings')):
                        nation = u.nation
                        break
                if nation is None:
                    # Fall back to region-based detection
                    fp_map = {'gondor': Nation.GONDOR, 'rohan': Nation.ROHAN,
                              'elves': Nation.ELVES, 'dwarves': Nation.DWARVES,
                              'the north': Nation.THE_NORTH, 'north': Nation.THE_NORTH}
                    sa_map = {'isengard': Nation.ISENGARD, 'sauron': Nation.SAURON,
                              'southrons and easterlings': Nation.SOUTHRON_EASTERLINGS,
                              'southrons': Nation.SOUTHRON_EASTERLINGS, 'easterlings': Nation.SOUTHRON_EASTERLINGS}
                    nation = fp_map.get(faction) or sa_map.get(faction) or Nation.GONDOR

            # A Nazgul ALWAYS belongs to Sauron (data-model invariant) —
            # never let a region/faction fallback mint a (NAZGUL, GONDOR)
            # figure on the piece-placement path (mirror of the muster fix).
            if unit_type == 'NAZGUL':
                nation = Nation.SAURON

            # Record the dynamic type so future lookups work correctly
            self.piece_types[piece_id] = (nation, unit_type)

            if unit_type == 'LEADER':
                army.leaders.append(gs.Leader(nation))
                _placed_counts['leader'] += 1
                _placed_pieces9.append(('leader', nation))
            elif unit_type == 'NAZGUL':
                army.leaders.append(gs.Nazgul(nation))
                _placed_counts['nazgul'] += 1
                _placed_pieces9.append(('nazgul', nation))
            elif unit_type == UnitKind.ELITE:
                army.units.append(gs.Unit(nation, UnitKind.ELITE))
                _placed_counts['elite'] += 1
                _placed_pieces9.append(('elite', nation))
            else:
                army.units.append(gs.Unit(nation, UnitKind.REGULAR))
                _placed_counts['regular'] += 1
                _placed_pieces9.append(('regular', nation))

        # ROOT-CAUSE FIX (un-recorded musters): outside a battle window
        # and outside a card-effect window, a 304(remove)+313(place)
        # pair — or a standalone 313 placement — at a map region is a
        # MUSTER (elite upgrade / recruitment) encoded via the swap
        # controls (2025-05-16 Adobim L416-417: 2 Isengard regulars out,
        # elites 325/326 in at Orthanc; present in EVERY campaign log).
        # Battle swaps and card placements stay state-only.  One record
        # per unit; only the first carries a die window — siblings are
        # 'free_die' (the human spent one Muster die).
        if (self._battle_synth_region is None
                and self._card_window_side is None
                and swap_type in ('307', '309')
                and _placed_pieces9):
            _p304 = getattr(self, '_pending_304', None)
            _pair = (_p304 is not None and _p304[1] == loc_san
                     and self.line_num - _p304[0] <= 4)
            _first = True
            # one row per placed PIECE, each stamped with ITS OWN
            # nation (119 doctrine: pieces name their nation — the old
            # shared last-piece nation mislabeled mixed placements);
            # kind-grouped order keeps the historical row schedule
            for _kind in ('elite', 'regular', 'leader', 'nazgul'):
                for _pk9, _pn9 in _placed_pieces9:
                    if _pk9 != _kind:
                        continue
                    _mp = {'region': to_key,
                           'units': {'regular': 0, 'elite': 0,
                                     'leader': 0, 'nazgul': 0},
                           'upgrade': _pair}
                    _mp['units'][_kind] = 1
                    if _kind in ('elite', 'regular'):
                        # 119: nation from the placed piece itself
                        _mp['_by_nation'] = {
                            f"{_pn9.value}|{_kind}": 1}
                    if not _first:
                        _mp['free_die'] = True
                    _mside9 = (Side.SHADOW if _pn9 in SHADOW_NATIONS
                               else Side.FREE_PEOPLES)
                    self._record_action('muster', _mp, _mside9)
                    _first = False
            self._pending_304 = None
            self.log(f"  (swap-muster: {dict(_placed_counts)} at "
                     f"{to_key}, upgrade={_pair})")

        # Update region control after battle replacements
        self._update_region_control(loc)

        # A battle window with unknown target: this swap (elite downgrade
        # casualty) reveals the region — patch the synthesized attack
        if self._battle_synth_region == '?':
            self._maybe_synth_attack(loc)

        # Siege press detection: Elite→Regular downgrade while in combat
        # with prior casualties. Stronghold-box swaps (garrison downgrades)
        # are defender casualty choices, never presses.  Presses only
        # exist at STRONGHOLDS — a downgrade mid-battle at a city/town
        # region (2024-07-27 Columbo T3 Edoras: the FP defender absorbs
        # a hit by downgrading an elite) is a casualty choice, never a
        # press.
        if (not in_stronghold_box and
                board.regions.get(sanitize(loc), {}).get('is_stronghold') and
                self._combat_region and self._combat_casualties_this_round and
                self._last_swap_remove_elite and
                (swap_type == '308'
                 # fix 76b: a 307-sourced press (replacement regular from
                 # REINFORCEMENTS) is normally ambiguous with a casualty
                 # downgrade (wave-E exclusion) — but an announced
                 # "press" in chat within the last few lines settles it.
                 # Record-only: the sentinel/no-mark treatment of 307
                 # weight ambiguity is unchanged.
                 or (swap_type == '307'
                     and self.line_num
                     - getattr(self, '_last_press_chat_line', -99) <= 8))
                and sanitize(loc) == sanitize(self._combat_region)):
            press_side = self._last_swap_remove_side or Side.SHADOW
            self.log(f"Siege press detected at {loc}")
            _pp9 = {'region': loc}
            _pn9 = getattr(self, '_last_swap_remove_nation', None)
            if _pn9 is not None:
                _pp9['nation'] = _pn9.value if hasattr(_pn9, 'value') \
                    else str(_pn9)
            self._record_action('siege_press', _pp9, press_side)
        elif (not in_stronghold_box and self._last_swap_remove_elite
                and self._battle_synth_region is not None
                and (swap_type == '308'
                     or (swap_type == '307'
                         and not board.regions.get(
                             sanitize(loc), {}).get('is_stronghold')))):
            # wave-E fix 2 (2026-07-23): extend the FIELD (non-stronghold)
            # elite-downgrade casualty mark to swap_type 307 (replacement
            # Regular 'from reinforcements') in addition to 308 ('from
            # casualties').  A Shadow FIELD elite downgraded mid-battle whose
            # returning Regular came from reinforcements (307) was previously
            # unparsed here (Block-1 press needs 308+is_stronghold; Block-1's
            # muster path needs battle_synth_region is None), so it fell
            # through unrecorded -> engine kept the un-downgraded elite.  The
            # source pool is a VASSAL bookkeeping detail; a field elite->
            # regular downgrade is the same casualty either way.  The
            # STRONGHOLD 307 case is deliberately EXCLUDED: at a stronghold a
            # 307 elite reduction is AMBIGUOUS between a defender casualty and
            # an attacker siege-PRESS (Block-1 catches only 308 presses), and
            # forcing the weight-ambiguity sentinel there wrongly over-removes
            # 307-sourced presses (regressed Columbo/Skaarj Lorien sieges).
            # v3 AMBIGUITY SENTINEL: a field elite->regular swap inside a
            # battle window that was NOT detected as a siege press — it
            # may be a casualty downgrade (1 hit) or a missed press (0
            # hits); the weighting is unknowable, so the round window
            # containing it must not be hit-forced.
            # PATCH A (RC4 v5): split by region type.  A NON-stronghold
            # field elite->regular swap here is an unambiguous casualty
            # DOWNGRADE (a field defender cannot press), so emit the exact
            # removal COMPOSITION {R:0,E:0,D:1} with the live cause -> v5
            # can reproduce the unit mix.  A stronghold swap that failed
            # press-detection KEEPS the weight-ambiguity sentinel
            # (downgrade=1 hit vs missed press=0 hits is unknowable), so
            # spec=None and its round window is not hit-forced.
            if board.regions.get(
                    sanitize(loc), {}).get('is_stronghold'):
                self._emit_casualty_mark(
                    self._last_swap_remove_side or Side.SHADOW,
                    loc_san, 1, 'downgrade')
            else:
                self._emit_casualty_mark(
                    self._last_swap_remove_side or Side.SHADOW,
                    loc_san, 1, self._mark_cause(loc_san),
                    spec={'kill_regular': 0, 'kill_elite': 0,
                          'downgrade': 1})

        # Reset swap tracking
        self._last_swap_remove_elite = False
        self._last_swap_remove_side = None
        self._last_swap_remove_nation = None

    def _handle_flip(self, piece_id: str):
        """Handle token flip (fellowship hidden/revealed, coronation)."""
        if piece_id == '241':
            # Strider/Aragorn figure flip = CROWNING (Aragorn — Heir to
            # Isildur, Will of the West die).  2026-01-05 kapcerz line 88:
            # Controls.323 on 241 right after the W-die spend — readable
            # log "flips Aragorn in Minas Tirith".  Rename the leader so
            # requirement checks ("Aragorn is in play") see him, keep the
            # piece mapping in sync for later drags of piece 241.
            # t138 (engine-check family 1): REPLACE the object instead
            # of renaming in place — clone_state shares Companion refs
            # (ai.py 'share refs'), so the in-place rename retro-mutated
            # every already-cloned pre-state and the certifier's
            # can_crown_aragorn saw 'Strider not found' at the very
            # decision being matched (38/38 crown misses).  The GtW
            # transform and the engine's own game_logic.crown_aragorn
            # both create a NEW object; mirror that.
            region = None
            for rk, army in self.state.armies.items():
                for _ci, ld in enumerate(army.leaders):
                    if isinstance(ld, gs.Companion) and \
                            ld.name in ('Strider', 'Aragorn'):
                        _new = gs.Companion(
                            name='Aragorn', level=3, leadership=2,
                            captain_of_the_west=True)
                        army.leaders[_ci] = _new
                        if self.state.fellowship.guide is ld:
                            self.state.fellowship.guide = _new
                        region = rk
                        break
                if region:
                    break
            if region is None:
                for _ci, comp in enumerate(
                        self.state.fellowship.companions):
                    if comp.name == 'Strider':
                        # 1.137 (USER): the fellowship path renamed
                        # WITHOUT upgrading - Aragorn kept Strider's
                        # leadership 1 (Aragorn: level 3, leadership 2)
                        _new = gs.Companion(
                            name='Aragorn', level=3, leadership=2,
                            captain_of_the_west=True,
                            guide_ability=comp.guide_ability)
                        self.state.fellowship.companions[_ci] = _new
                        if self.state.fellowship.guide is comp:
                            self.state.fellowship.guide = _new
                        region = self.fellowship_location
                        break
            if getattr(self.state, 'aragorn_crowned', None) is not None:
                self.state.aragorn_crowned.crowned = True
            COMPANIONS['241'] = 'Aragorn'
            self.log(f"Strider crowned as Aragorn ({region})")
            self._record_action('crown_aragorn',
                                {'region': region}, Side.FREE_PEOPLES)
            return
        if piece_id in ['247', '248']:
            # t125 fix 3 (client-verified timing): the client silently
            # returns the progress marker 248 to Track 0 at the first
            # fellowship FLIP after Mordor entry, not at the entry drag
            # itself (Wirbelwind dumps: 248 stays on Track 1 through
            # cp350, entry at L304, and zeroes with the L361 flip; the
            # L345 Mordor advance did not reset it).  Progress is
            # engine-inert inside Mordor — this is client-board sync.
            if self.state.fellowship.mordor_entered \
                    and self.state.fellowship.progress:
                self.log(f"Mordor flip: progress "
                         f"{self.state.fellowship.progress} -> 0 "
                         f"(client marker reset)")
                self.state.fellowship.progress = 0
            # If a hunt/declare handler already set hidden directly, the
            # player's marker flip is the SAME event — blind toggling here
            # double-counted it and inverted every hide/reveal afterwards
            # (hides were logged as reveals).
            if self._skip_next_fsp_flip > 0:
                # AUTO-95 game-5 (2026-08-17, the routed stale-skip
                # SIBLING): the declare branch armed skip=2 for the
                # corpus flip-and-unflip dance, but the AUTO emitter
                # sends ONE dance flip (or none) - the leftover skip
                # ate the NEXT REAL flips one by one (an engine
                # reveal at +48 lines, then - via a re-arm the eaten
                # flip made invisible - the post-entry hide), and
                # the phase inversion rode the Mordor track to the
                # ring end, exactly like AUTO-94.  CONSUMPTION IS
                # WINDOW-TRUTH (coordinator rule): a skip may only
                # eat a flip landing NEAR its arming; a stale skip
                # EXPIRES and the flip processes normally.
                _arm9 = getattr(self, '_fsp_skip_line', None)
                if _arm9 is not None                         and self.line_num - _arm9 <= 10:
                    self._skip_next_fsp_flip -= 1
                    self.log(f"Fellowship marker flip (already synced)")
                    # eaten flips STAMP the window ledger too - the
                    # walk branch's pre-flip check must see EVERY
                    # flip, toggled or eaten (AUTO-95: the invisible
                    # eat let the branch re-arm)
                    self._last_fsp_flip = {
                        'line': self.line_num,
                        'die_seq': getattr(self, '_die_consume_seq', 0),
                        'alog': -1}
                    return
                self._skip_next_fsp_flip = 0
                self.log("Fellowship flip: stale skip EXPIRED "
                         "(armed %s line(s) ago) - processing the "
                         "flip normally"
                         % ('?' if _arm9 is None
                            else self.line_num - _arm9))
            # O.1 RESOLVED (user ruling 2026-07-07): Mordor entry does NOT
            # re-hide a revealed Fellowship — but players commonly
            # FLIP-AND-UNFLIP the marker (a quick pair, no die spent)
            # purely to reset the fellowship tracker display when
            # declaring/entering Mordor.  A flip pair in close succession
            # with NO die consumed between them is tracker-reset
            # BOOKKEEPING: cancel the first flip's toggle+record, emit
            # nothing.  (Every genuine reveal has an attributable source;
            # a die-less flip pair has none.)
            _pf = getattr(self, '_last_fsp_flip', None)
            if (_pf is not None
                    and self.line_num - _pf['line'] <= 6
                    and getattr(self, '_die_consume_seq', 0)
                        == _pf['die_seq']
                    and self.action_log
                    and len(self.action_log) - 1 == _pf['alog']
                    and self.action_log[-1][0] in ('fellowship_hide',
                                                   'fellowship_reveal')
                    and not (self.action_log[-1][1] or {}).get('die_used')):
                del self.action_log[-1]
                self.state.fellowship.hidden = \
                    not self.state.fellowship.hidden
                self._last_fsp_flip = None
                self.log("Fellowship flip pair = tracker reset "
                         "(bookkeeping, O.1)")
                return
            self.state.fellowship.hidden = not self.state.fellowship.hidden
            status = "hidden" if self.state.fellowship.hidden else "revealed"
            self.log(f"Fellowship {status}")
            if not self.state.fellowship.hidden:
                # reveal handled via the flip (no pawn placement)
                self._hunt_reveal_pending = False
            self._record_action('fellowship_hide' if self.state.fellowship.hidden else 'fellowship_reveal', {
                'hidden': self.state.fellowship.hidden
            }, Side.FREE_PEOPLES)
            self._last_fsp_flip = {
                'line': self.line_num,
                'die_seq': getattr(self, '_die_consume_seq', 0),
                'alog': len(self.action_log) - 1}
            return
        # POLITICAL ACTIVATION: nations are activated by FLIPPING their
        # political marker to the colored side (auth piece table
        # 256-263; 'p*' images = flippable active side).  Previously
        # dropped — the replay's political.active never flipped unless
        # a box DRAG happened, so 'Gondor must be Active' requirements
        # and at-war advances failed corpus-wide (2025-01-10
        # SilmarillWill L188/215/218: north/dwarves/rohan/gondor/
        # southron flips).
        #
        # t142 (engine-check political family, 11 games): a political
        # flip is an ACTIVATION, never a de-activation — WOTR nations
        # never revert to passive (rulebook p.33: passive -> active is
        # the only transition).  The old TOGGLE double-counted against
        # the box-advance heuristic (_handle_political sets active=True
        # on any advance): Gondor advanced 1->2 (heuristic set active),
        # the human's real activation flip then TOGGLED it back to
        # passive, and the later 2->3 At-War advance was illegal
        # (Jorrick L163/169/276; +9 twins).  SET active=True
        # (idempotent) instead: the flip can only ever activate, so it
        # can never re-introduce an 'advance not offered' miss.  Genuine
        # flip-undo pairs are already removed by the undo pre-pass
        # (1.169); a stray manual re-flip leaving a nation active is
        # harmless (an activated nation not yet at war changes nothing).
        _POL_FLIP = {'256': gs.Nation.ELVES, '257': gs.Nation.DWARVES,
                     '258': gs.Nation.THE_NORTH, '259': gs.Nation.ROHAN,
                     '260': gs.Nation.GONDOR,
                     '261': gs.Nation.SOUTHRON_EASTERLINGS,
                     '262': gs.Nation.SAURON, '263': gs.Nation.ISENGARD}
        if piece_id in _POL_FLIP:
            nat = _POL_FLIP[piece_id]
            was_active = bool(self.state.political.active.get(nat, False))
            self.state.political.active[nat] = True
            self.log(f"Political marker flip: {nat.value} -> ACTIVE")
            if not was_active:
                box = self.state.political.boxes.get(nat, 0)
                self._record_action('political', {
                    'nation': nat.value, 'from': box, 'to': box,
                    'activate': True},
                    Side.SHADOW if nat in SHADOW_NATIONS
                    else Side.FREE_PEOPLES)
            return


def create_replay_gui(state: gs.GameState, board_obj, replayer: LogReplayer):
    """Create GUI with replay controls."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    from gui import WotrGUI, COLORS

    gui = WotrGUI(state, board_obj)

    # Store replayer state
    gui._replayer = replayer
    gui._replay_running = False
    gui._replay_speed = 100
    gui._log_lines = []
    gui._log_index = 0

    # Load log lines and preprocess for card decoding
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'game_logs', 'thisone.log')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        gui._log_lines = [line.strip() for line in f if line.strip()]

    # Preprocess for card decoding (first pass)
    replayer.preprocess(gui._log_lines)

    # Add console log display frame at top (shows parsed output, not raw log)
    console_frame = tk.Frame(gui.root, bg=COLORS["bg"], pady=5)
    console_frame.pack(side=tk.TOP, fill=tk.X, before=gui.main_frame)

    tk.Label(console_frame, text="Console Log:", bg=COLORS["bg"],
             fg=COLORS["highlight"], font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)

    console_text = tk.Text(console_frame, height=6, width=100, bg="#1a1a2e", fg="#00ff88",
                           font=("Monaco", 9), wrap=tk.WORD)
    console_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    console_scroll = tk.Scrollbar(console_frame, orient=tk.VERTICAL, command=console_text.yview)
    console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    console_text.config(yscrollcommand=console_scroll.set)
    gui._console_text = console_text

    # Add replay frame
    replay_frame = tk.Frame(gui.root, bg=COLORS["bg"], pady=5)
    replay_frame.pack(side=tk.TOP, fill=tk.X, before=gui.main_frame)

    tk.Label(replay_frame, text="Log Replay", bg=COLORS["bg"],
             fg=COLORS["highlight"], font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=10)

    def update_console_display():
        """Update the console text to show recent parsed output."""
        gui._console_text.config(state=tk.NORMAL)
        gui._console_text.delete(1.0, tk.END)
        # Show last 10 lines of console output
        recent = replayer.console_output[-10:] if replayer.console_output else []
        for line in recent:
            gui._console_text.insert(tk.END, line + "\n")
        gui._console_text.see(tk.END)
        gui._console_text.config(state=tk.DISABLED)

    def sync_hands_to_gui():
        """Sync card decoder hands to the GUI's hand display."""
        # Build card ID strings that match GUI format
        # The GUI uses card IDs like "FP_Character_1" and looks them up in cards.py
        # For replay, we'll use descriptive names directly in the hand listbox

        # Update FP hand in game state
        fp_cards = []
        for card in replayer.card_decoder.fp_hand:
            if card.get('decoded') and card.get('card_name'):
                # Use format: "REPLAY_FP_cardname" so GUI shows it properly
                fp_cards.append(f"REPLAY_{card['card_name']}")
            else:
                fp_cards.append(f"REPLAY_{card['deck_type']} (unknown)")
        state.hands[Side.FREE_PEOPLES].hand = fp_cards

        # Update SA hand in game state
        sa_cards = []
        for card in replayer.card_decoder.sa_hand:
            if card.get('decoded') and card.get('card_name'):
                sa_cards.append(f"REPLAY_{card['card_name']}")
            else:
                sa_cards.append(f"REPLAY_{card['deck_type']} (unknown)")
        state.hands[Side.SHADOW].hand = sa_cards

        # Sync GUI hands with game state
        gui._sync_hands()
        gui._update_hand_display()

    # Override the hand display update to show replay cards nicely
    original_update_hand = gui._update_hand_display
    def replay_update_hand_display():
        """Update hand display for replay mode."""
        gui.hand_listbox.delete(0, tk.END)
        hand = gui.hands[gui.current_side]
        for card_id in hand:
            if card_id.startswith("REPLAY_"):
                # Display the card name directly
                display = card_id[7:]  # Remove "REPLAY_" prefix
            else:
                # Fall back to original behavior
                from cards import get_card
                card = get_card(card_id)
                display = card.card_name if card else card_id
            gui.hand_listbox.insert(tk.END, display)
    gui._update_hand_display = replay_update_hand_display

    def step():
        if gui._log_index < len(gui._log_lines):
            line = gui._log_lines[gui._log_index]
            gui._replayer.line_num = gui._log_index + 1
            gui._replayer._process_line(line)
            gui._log_index += 1
            progress_lbl.config(text=f"{gui._log_index}/{len(gui._log_lines)}")
            update_console_display()
            sync_hands_to_gui()
            gui.refresh()
        else:
            event_lbl.config(text="Complete!")
            gui._replay_running = False
            play_btn.config(text="Play")

    def step_n(n):
        for _ in range(n):
            if gui._log_index >= len(gui._log_lines):
                break
            line = gui._log_lines[gui._log_index]
            gui._replayer.line_num = gui._log_index + 1
            gui._replayer._process_line(line)
            gui._log_index += 1
        progress_lbl.config(text=f"{gui._log_index}/{len(gui._log_lines)}")
        update_console_display()
        sync_hands_to_gui()
        gui.refresh()

    def toggle_play():
        gui._replay_running = not gui._replay_running
        play_btn.config(text="Pause" if gui._replay_running else "Play")
        if gui._replay_running:
            auto_step()

    def auto_step():
        if not gui._replay_running:
            return
        if gui._log_index < len(gui._log_lines):
            step()
            gui.root.after(gui._replay_speed, auto_step)
        else:
            gui._replay_running = False
            play_btn.config(text="Play")

    def update_speed(evt=None):
        try:
            gui._replay_speed = int(speed_var.get())
        except:
            pass

    # Buttons
    tk.Button(replay_frame, text="Step", command=step,
              bg=COLORS["button_bg"], fg=COLORS["button_fg"]).pack(side=tk.LEFT, padx=2)
    play_btn = tk.Button(replay_frame, text="Play", command=toggle_play,
                         bg=COLORS["button_bg"], fg=COLORS["button_fg"])
    play_btn.pack(side=tk.LEFT, padx=2)
    tk.Button(replay_frame, text="+10", command=lambda: step_n(10),
              bg=COLORS["button_bg"], fg=COLORS["button_fg"]).pack(side=tk.LEFT, padx=2)
    tk.Button(replay_frame, text="+100", command=lambda: step_n(100),
              bg=COLORS["button_bg"], fg=COLORS["button_fg"]).pack(side=tk.LEFT, padx=2)

    tk.Label(replay_frame, text="Speed:", bg=COLORS["bg"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=(10,2))
    speed_var = tk.StringVar(value="100")
    ttk.Combobox(replay_frame, textvariable=speed_var, values=["50","100","250","500"], width=5).pack(side=tk.LEFT)
    speed_var.trace('w', lambda *a: update_speed())

    progress_lbl = tk.Label(replay_frame, text=f"0/{len(gui._log_lines)}",
                            bg=COLORS["bg"], fg=COLORS["text"])
    progress_lbl.pack(side=tk.RIGHT, padx=10)

    event_lbl = tk.Label(replay_frame, text="Ready", bg=COLORS["bg"], fg=COLORS["text_dim"])
    event_lbl.pack(side=tk.RIGHT, padx=10)

    # Initialize displays
    update_console_display()
    sync_hands_to_gui()

    gui.run()


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="WOTR Log Replay")
    parser.add_argument("log_file", nargs="?",
                       default=None)
    parser.add_argument("-g", "--gui", action="store_true", help="Launch GUI")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Load game
    print("Loading board...")
    b = board.load_board()
    state = gs.setup_initial_game(b)
    print(f"Board: {len(b._regions)} regions")

    # Create replayer
    replayer = LogReplayer(state, b, verbose=args.verbose)

    if args.gui:
        print("Launching GUI...")
        create_replay_gui(state, b, replayer)
    else:
        print(f"Replaying {args.log_file}...")
        replayer.parse_and_replay(args.log_file)

        print(f"\n=== FINAL STATE ===")
        print(f"Turn: {state.turn}")
        print(f"Fellowship progress: {state.fellowship.progress}")
        print(f"Hunt box: {len(state.hunt_box)} dice")
        print(f"Armies: {len(state.armies)}")


if __name__ == "__main__":
    main()
