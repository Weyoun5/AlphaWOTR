#!/usr/bin/env python3
"""
War of the Ring - AI core (2026-07-27 state)

Single-file AI infrastructure:
  1. State cloning (clone_state)
  2. State-to-vector encoding (encode_state, 3742 dims, tail-append rule)
  3. Action encoding (encode_action, ACTION_ENCODING_DIM dims; col 22
     dead by design)
  4. Nets (WOTRValueNet: arch registry base/wide/xformer; value head +
     spatial action scorer)
  5. Legal move enumeration (get_legal_actions)
  6. Action application (apply_action)
  7. MCTS (mcts_search, chance nodes, tree reuse, WOTR_TURN_HORIZON
     cross-turn search)
  8. Self-play episodes (_run_single_episode; the td_self_play twin was
     GUI opponent)

Training lives in train_corpus.py (full-menu BC + TD-lambda value);
extraction in rendered_extract.py.  The old CSV-export / legacy-npz
training paths were removed 2026-07-27.
"""

import dataclasses
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# Ensure project directory is on path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
sys.path.insert(0, _SRC_DIR)


def _resolve_model_path(path: str) -> str:
    """Resolve model path relative to project root (board.py changes CWD to src/)."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)

# Set matplotlib backend before any imports that might trigger Tkinter
import matplotlib
matplotlib.use('Agg')

import board
from board import sanitize
import game_state as gs
from game_state import (
    Side, Nation, UnitKind, ActionDieFace,
    GameState, Army, Unit, Leader, Nazgul,
    Fellowship, PoliticalTrack, VictoryPoints, DeckStatus,
    PlayerInfo, ActionDie, HuntTile, PersistentCardState,
    ShadowMinion, GandalfTheWhite, Aragorn, Companion,
)
import game_logic
import combat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _armies_lookup(state, key):
    """state.armies lookup tolerant of key-casing drift ('Pelargir' vs
    'pelargir') — exact hit first, then a sanitized scan.  The exact-key
    class has bitten repeatedly (Fangorn, Pelargir retreat, ketchuptime
    Lamedon; iterations 2/20/22) — use this instead of raw .get() when
    the key's PROVENANCE is a record/pending dict rather than an
    armies-iteration key."""
    if key is None:
        return None
    a = state.armies.get(key)
    if a is not None:
        return a
    ks = sanitize(str(key))
    for k, v in state.armies.items():
        if sanitize(k) == ks:
            return v
    return None


SHADOW_NATIONS = {Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS}
FP_NATIONS = {Nation.ELVES, Nation.DWARVES, Nation.GONDOR, Nation.THE_NORTH, Nation.ROHAN}
ALL_NATIONS = list(Nation)

# Shared action type vocabulary (used in training data / encode_action one-hot)
ACTION_TYPE_CATEGORIES = [
    'army_move', 'army_attack', 'advance_political', 'muster',
    'draw_card', 'play_event', 'fellowship_move', 'fellowship_hide',
    'fellowship_declare', 'special_character', 'pass', 'eye_allocate',
    'other', 'combat_card',
]
_ACTION_TYPE_TO_IDX = {cat: i for i, cat in enumerate(ACTION_TYPE_CATEGORIES)}

# (Per-side category policy heads removed — the spatial action scorer,
#  trained by behavioral cloning, is the single policy pathway.)

# Card embedding constants
MAX_HAND_SIZE = 6
NUM_CARD_IDS = 96       # 96 real cards (0-95)
CARD_PAD_IDX = 96       # Embedding padding index (index 96)

_CARD_ID_TO_IDX = None

# (RETREAT_ADVANCE_HOOK + VICTOR_ADVANCE_SAMPLE_CB deleted 2026-07-27
#  — the teacher that installed them is retired; all gated branches
#  collapsed to the self-play behavior)

def _build_card_id_index():
    from cards import get_cards
    return {cid: i for i, cid in enumerate(sorted(get_cards().keys()))}

def _get_card_idx_map():
    global _CARD_ID_TO_IDX
    if _CARD_ID_TO_IDX is None:
        _CARD_ID_TO_IDX = _build_card_id_index()
    return _CARD_ID_TO_IDX

# Map log_replay action types to canonical action type categories
LOG_ACTION_TO_CANONICAL = {
    'army_move': 'army_move',
    'political': 'advance_political',
    'muster': 'muster',
    'draw_card': 'draw_card',
    'play_event': 'play_event',
    # 1.149: deliberate opponent table-card kill (die + payment)
    'discard_table_card': 'play_event',
    'fellowship_move': 'fellowship_move',
    'fellowship_hide': 'fellowship_hide',
    'fellowship_declare': 'fellowship_declare',
    'eye_allocate': 'eye_allocate',
    'pass': 'pass',
    'skip': 'pass',  # Skip (discard die) maps to same category as pass
    'mordor_move': 'fellowship_move',
    'companion_separate': 'special_character',
    'fellowship_reveal_place': 'fellowship_move',
    # Non-decisions: map to None to exclude from policy training
    'dice_roll': None,
    'combat_casualties': None,
    'hunt_tile_draw': None,
    'combat_start': None,
    # True 'other' actions (player decisions but no dedicated category)
    'fellowship_reveal': 'other',
    'companion_death': 'other',
    'use_token': 'other',
    'combat_card': 'combat_card',
    # Combat-card decisions (engine pending phases; logged 'combat_card'
    # rows = plays, parser-emitted 'combat_card_none' rows = 'nc' declines)
    'combat_card_play': 'combat_card',
    'combat_card_none': 'combat_card',
    # Combat-card effect choices (Onslaught sacrifice etc.) — parser rows
    # from units-to-Current-SA/FP-Card moves and 'ffN' forfeit chats
    'combat_effect_choice': 'combat_card',
    # Gandalf 'shining' pre-combat-card decision (item 11)
    'shine': 'combat_card',
    'shine_none': 'combat_card',
    # Learned decisions (mapped to proper categories)
    'elven_ring': 'special_character',
    'dwarven_ring': 'other',
    'siege_press': 'army_attack',
    'combat_retreat': 'army_move',
    'choose_guide': 'special_character',
    'hunt_sacrifice_guide': 'special_character',
    'hunt_sacrifice_random': 'special_character',
    'hunt_guide_separate': 'special_character',
    # Parser-overhaul additions (previously missing/mislabeled)
    'army_attack': 'army_attack',
    'retreat_into_siege': 'army_move',
    'companion_move': 'special_character',
    'transform_gandalf': 'special_character',
    'crown_aragorn': 'special_character',
    'hunt_take_corruption': 'special_character',
    'nazgul_move': 'special_character',
    # Post-battle move-in (a real decision — how many units advance —
    # but no discrete engine action yet, so value-only)
    'battle_advance': 'other',
}

def _find_mandatory_draws(action_log):
    """Return set of action_log indices that are mandatory turn-start draws.

    Every turn both players draw 1 Character + 1 Strategy card (= 4 draws
    per turn) before dice are rolled.  These are not player decisions.

    Detection: walk backwards from each dice_roll within a window of 12
    actions, skipping turn-phase actions and tolerating small gaps (players
    sometimes do a quick muster or other action between drawing cards and
    rolling dice).  Collects up to 4 draw_card per turn boundary.
    """
    mandatory = set()
    n = len(action_log)
    if n == 0:
        return mandatory

    # Find first dice_roll of each turn.  dice_roll typically appears in
    # pairs (SH then FP) -- we only need the first of each pair.
    dice_roll_indices = []
    prev_was_dice = False
    for i, (at, _, _) in enumerate(action_log):
        if at == 'dice_roll':
            if not prev_was_dice:
                dice_roll_indices.append(i)
            prev_was_dice = True
        else:
            prev_was_dice = False

    for dr_idx in dice_roll_indices:
        draws_found = []
        # Walk backwards up to 12 actions, collecting draw_card actions.
        # Allow any actions in between (muster, fellowship_declare, etc.)
        # since players sometimes interleave small actions with draws.
        j = dr_idx - 1
        limit = max(dr_idx - 12, -1)
        while j > limit and len(draws_found) < 4:
            at = action_log[j][0]
            if at == 'draw_card':
                draws_found.append(j)
            j -= 1
        for idx in draws_found:
            mandatory.add(idx)

    return mandatory


# Per rulebook p.7: maximum figures available per nation (regular, elite, leader/nazgul)
NATION_MAX_FIGURES = {
    Nation.DWARVES:               {'regular': 5,  'elite': 5,  'leader': 4},
    Nation.ELVES:                 {'regular': 5,  'elite': 10, 'leader': 4},
    Nation.GONDOR:                {'regular': 15, 'elite': 5,  'leader': 4},
    Nation.THE_NORTH:             {'regular': 10, 'elite': 5,  'leader': 4},
    Nation.ROHAN:                 {'regular': 10, 'elite': 5,  'leader': 4},
    Nation.ISENGARD:              {'regular': 12, 'elite': 6,  'leader': 0},
    Nation.SAURON:                {'regular': 36, 'elite': 6,  'leader': 8},  # 8 Nazgul
    Nation.SOUTHRON_EASTERLINGS:  {'regular': 24, 'elite': 6,  'leader': 0},
}

# Regions that are settlements (for mustering). Built lazily from board.regions.
_SETTLEMENT_REGIONS: Optional[Dict[str, dict]] = None

def _get_settlement_regions() -> Dict[str, dict]:
    """Return {sanitized_key: region_meta} for all settlements."""
    global _SETTLEMENT_REGIONS
    if _SETTLEMENT_REGIONS is None:
        _SETTLEMENT_REGIONS = {}
        for key, info in board.regions.items():
            if info.get("is_city") or info.get("is_stronghold") or info.get("is_town"):
                _SETTLEMENT_REGIONS[key] = info
    return _SETTLEMENT_REGIONS


_NATION_SIDE = {}


def _side_of_nation(nation: Nation) -> Side:
    # PROFILE (session 34 scale work): 1.1M calls/search — a dict memo
    # replaces enum-set membership (result identical; Nation is finite)
    s = _NATION_SIDE.get(nation)
    if s is None:
        s = (Side.SHADOW if nation in SHADOW_NATIONS
             else Side.FREE_PEOPLES)
        _NATION_SIDE[nation] = s
    return s


# Map sanitized faction names (from board.fid_to_name) to Nation enums
_FACTION_TO_NATION = {
    'gondor': Nation.GONDOR,
    'rohan': Nation.ROHAN,
    'elves': Nation.ELVES,
    'dwarves': Nation.DWARVES,
    'the north': Nation.THE_NORTH,
    'north': Nation.THE_NORTH,
    'isengard': Nation.ISENGARD,
    'sauron': Nation.SAURON,
    'southrons': Nation.SOUTHRON_EASTERLINGS,
    'southron': Nation.SOUTHRON_EASTERLINGS,
    'southrons and easterlings': Nation.SOUTHRON_EASTERLINGS,
    # the board splits S&E geographically into TWO faction names:
    # 'southrons' (Harad) and 'easterlings' (Rhun) — without this alias
    # every Rhun region resolved to nation=None (no musters, no
    # defender-auto, no border checks; census: North Rhun x9)
    'easterlings': Nation.SOUTHRON_EASTERLINGS,
}


def _region_nation(rkey: str) -> Optional[Nation]:
    """Return the Nation that owns a region, or None if unknown."""
    meta = board.regions.get(rkey, {})
    parent_id = meta.get("parent_id")
    if not parent_id:
        return None
    faction_name = board.fid_to_name.get(parent_id, "")
    return _FACTION_TO_NATION.get(faction_name)


def _count_deployed(state: GameState, nation: Nation) -> Dict[str, int]:
    """Count units of a nation currently on the board.

    Returns {'regular': N, 'elite': N, 'leader': N}.
    """
    counts = {'regular': 0, 'elite': 0, 'leader': 0}
    for army in state.armies.values():
        for u in army.units:
            if u.nation == nation:
                if u.kind == UnitKind.REGULAR:
                    counts['regular'] += 1
                else:
                    counts['elite'] += 1
        for ld in army.leaders:
            # Conservation fix (2026-07-20): pool figures only —
            # Minions/Companions are not leader-pool pieces.
            if (type(ld).__name__ in ('Leader', 'Nazgul')
                    and hasattr(ld, 'nation') and ld.nation == nation):
                counts['leader'] += 1
    return counts


def _count_deployed_all(state: GameState) -> Dict[Nation, Dict[str, int]]:
    """One army pass producing deployed counts for EVERY nation.

    Equivalent to {nat: _count_deployed(state, nat)} but O(armies) instead
    of O(armies x nations) — recruit enumeration and pool encoding call
    this once per state instead of per (nation, type) candidate.
    """
    counts: Dict[Nation, Dict[str, int]] = {
        n: {'regular': 0, 'elite': 0, 'leader': 0} for n in ALL_NATIONS}
    for army in state.armies.values():
        for u in army.units:
            c = counts.get(u.nation)
            if c is not None:
                if u.kind == UnitKind.REGULAR:
                    c['regular'] += 1
                else:
                    c['elite'] += 1
        for ld in army.leaders:
            # Conservation fix (2026-07-20): only POOL figures count —
            # plain Leaders and Nazgul. Minions (Witch-king/Mouth/
            # Saruman) and Companions carry a nation but are NOT
            # leader-pool figures; counting them blocked the last
            # nazgul/leader from ever mustering while they lived.
            _tn = type(ld).__name__
            if _tn not in ('Leader', 'Nazgul'):
                continue
            nat = getattr(ld, 'nation', None)
            c = counts.get(nat)
            if c is not None:
                c['leader'] += 1
    return counts


def figures_available(state: GameState, nation: Nation, typ: str,
                      deployed_all: Optional[Dict] = None) -> int:
    """THE ONE ANSWERER for "is a physical piece available?".

    USER RULING 2026-08-18: "component limits are ENFORCED STRICTLY
    EVERYWHERE - the engine may never place more units on the board
    than there are physical pieces (musters, downgrades, card effects,
    reinforcements, everything)."

    Returns the number of `typ` figures of `nation` that still exist
    OFF the board and may therefore be placed:

        cap - deployed - (permanent FP casualties)

    FP casualties are permanent and never return (rulebook p.30), so
    they are spent capacity; Shadow casualties go back to the pool and
    are not subtracted.  Never negative: an over-cap board (which is
    itself a defect) reports 0 headroom rather than a negative one.

    WHY THIS EXISTS.  The same arithmetic was written SIX times -
    ai._can_recruit, game_logic._clamp_place, combat._live_reg_avail,
    combat._capture_reg_baseline, and inline copies in event_effects
    and card_options.  Six answerers to one question is how a supply
    rule gets enforced in five places and skipped in the sixth; every
    one of them is now a thin wrapper over this.  DO NOT ADD A SEVENTH -
    call this.

    `deployed_all`: precomputed _count_deployed_all(state) to avoid a
    full army scan per candidate.
    """
    maxes = NATION_MAX_FIGURES.get(nation)
    if not maxes:
        return 0
    cap = maxes.get(typ, 0)
    if cap <= 0:
        return 0
    if deployed_all is not None:
        deployed = deployed_all.get(nation, {})
    else:
        deployed = _count_deployed(state, nation)
    used = deployed.get(typ, 0)
    if _side_of_nation(nation) == Side.FREE_PEOPLES:
        fp_cas = getattr(state, 'fp_casualties', None) or {}
        used += (fp_cas.get(nation, {}) or {}).get(typ, 0)
    return max(0, cap - used)


def _can_recruit(state: GameState, nation: Nation, typ: str,
                 deployed_all: Optional[Dict] = None) -> bool:
    """Check if a nation has available reinforcements of the given type.

    Per rulebook p.30: FP casualties are permanent (out of game forever),
    Shadow casualties return to reinforcement pool. For FP, available =
    max - deployed - casualties. For Shadow, available = max - deployed.

    `deployed_all`: precomputed _count_deployed_all(state) to avoid a
    full army scan per candidate.
    """
    # ONE ANSWERER (USER ruling 2026-08-18): this used to carry its own
    # copy of the cap arithmetic.  It is now a predicate over
    # figures_available, so "can I recruit" and "how many are left"
    # can never disagree.
    return figures_available(state, nation, typ, deployed_all) > 0


# All region keys from the board (built at module load)
ALL_REGION_KEYS = sorted(board.regions.keys())
REGION_INDEX = {k: i for i, k in enumerate(ALL_REGION_KEYS)}
NUM_REGIONS = len(ALL_REGION_KEYS)

# Precomputed board graph edges for GNN (static, symmetric)
NODE_FEAT_DIM = 30  # per-region features in encode_state (iter 36)
# [0-8] unit/control features, [9] fellowship-here, [10] my named
# characters here /3, [11] enemy named characters here /3,
# [12] my nazgul here /8, [13] enemy nazgul here /8 (nazgul are
# distinctly visible figures, previously lumped into leader counts),
# [14:30] per-NATION unit composition (8 nations x regular/15, elite/5
# in ALL_NATIONS order — "everything the human sees": mixed-nation
# stacks were previously indistinguishable from single-nation ones)

def _build_edge_index():
    """Build directed edge lists from board adjacency graph.

    v26 port-hardening (HPC audit 2026-07-20): iterate neighbors
    SORTED — the raw set order is PYTHONHASHSEED-dependent, and the
    GNN scatter-add over edges is float-non-associative, so edge
    order must be platform-invariant (same fix class as the
    Board.adjacent hash-order flake). Landed pre-v26-training so the
    new nets are trained and certified under the sorted order."""
    src, dst = [], []
    for rkey in ALL_REGION_KEYS:
        i = REGION_INDEX[rkey]
        for nb in sorted(board.adj.get(rkey, set())):
            j = REGION_INDEX.get(nb)
            if j is not None:
                src.append(i)
                dst.append(j)
    return src, dst

_EDGE_SRC, _EDGE_DST = _build_edge_index()


# ===================================================================
# Precomputed BFS distances for military threat features
# ===================================================================

def _precompute_settlement_distances():
    """Compute BFS distances from every settlement to all regions.

    Returns:
        fp_targets: list of (region_key, vp_value) for FP settlements
        sh_targets: list of (region_key, vp_value) for Shadow settlements
        dist_to_fp: dict[region_key] -> min distance to any FP VP target
        dist_to_sh: dict[region_key] -> min distance to any Shadow VP target
    """
    from collections import deque

    b_obj = board.Board()

    # FP settlements (Shadow captures these for VP)
    _FP_SETTLEMENTS = {
        'dol amroth': 2, 'minas tirith': 2, 'helm s deep': 2,
        'erebor': 2, 'grey havens': 2, 'lorien': 2,
        'rivendell': 2, 'woodland realm': 2,
        'dale': 1, 'edoras': 1, 'pelargir': 1, 'the shire': 1,
    }
    # Shadow settlements (FP captures these for VP)
    _SH_SETTLEMENTS = {
        'barad dur': 2, 'dol guldur': 2, 'minas morgul': 2,
        'morannon': 2, 'moria': 2, 'mount gundabad': 2,
        'orthanc': 2, 'umbar': 2,
        'angmar': 1, 'far harad': 1,
    }

    fp_targets = [(k, v) for k, v in _FP_SETTLEMENTS.items()]
    sh_targets = [(k, v) for k, v in _SH_SETTLEMENTS.items()]

    def bfs_min_dist(sources):
        """Multi-source BFS from all sources at once."""
        dist = {}
        queue = deque()
        for src, _ in sources:
            dist[src] = 0
            queue.append(src)
        while queue:
            node = queue.popleft()
            for nb in b_obj.adjacent(node):
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    dist_to_fp = bfs_min_dist(fp_targets) if fp_targets else {}
    dist_to_sh = bfs_min_dist(sh_targets) if sh_targets else {}

    return fp_targets, sh_targets, dist_to_fp, dist_to_sh

_FP_VP_TARGETS, _SH_VP_TARGETS, _DIST_TO_FP_TARGET, _DIST_TO_SH_TARGET = \
    _precompute_settlement_distances()


def _precompute_mordor_distances():
    """BFS distance from every region to nearest Mordor gate (morannon/minas morgul)."""
    from collections import deque
    b_obj = board.Board()
    gates = [('morannon', 0), ('minas morgul', 0)]
    dist = {}
    queue = deque()
    for g, _ in gates:
        dist[g] = 0
        queue.append(g)
    while queue:
        node = queue.popleft()
        for nb in b_obj.adjacent(node):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist

_DIST_TO_MORDOR = _precompute_mordor_distances()
_MAX_MORDOR_DIST = max(_DIST_TO_MORDOR.values()) if _DIST_TO_MORDOR else 15.0


def _compute_military_threat_features(
        my_armies: 'Dict[str, Tuple]',
        en_armies: 'Dict[str, Tuple]',
        my_side: Side) -> 'List[float]':
    """Compute spatial military threat features.

    Returns 8 features:
    - my_threat_1: my units within 1 step of enemy VP targets (normalized)
    - my_threat_2: my units within 2 steps of enemy VP targets
    - my_threat_3: my units within 3 steps of enemy VP targets
    - en_threat_1: enemy units within 1 step of my VP targets
    - en_threat_2: enemy units within 2 steps of my VP targets
    - en_threat_3: enemy units within 3 steps of my VP targets
    - my_targets_threatened: how many of my VP targets have enemy adjacent
    - en_targets_threatened: how many of enemy VP targets I'm adjacent to
    """
    if my_side == Side.FREE_PEOPLES:
        dist_to_my = _DIST_TO_FP_TARGET
        dist_to_en = _DIST_TO_SH_TARGET
        my_targets = _FP_VP_TARGETS
        en_targets = _SH_VP_TARGETS
    else:
        dist_to_my = _DIST_TO_SH_TARGET
        dist_to_en = _DIST_TO_FP_TARGET
        my_targets = _SH_VP_TARGETS
        en_targets = _FP_VP_TARGETS

    # Count my units within 1, 2, 3 steps of enemy VP targets
    my_t1 = my_t2 = my_t3 = 0
    for rkey, (regs, elites, leaders, nazgul) in my_armies.items():
        strength = regs + elites
        d = dist_to_en.get(rkey, 99)
        if d <= 1:
            my_t1 += strength
        if d <= 2:
            my_t2 += strength
        if d <= 3:
            my_t3 += strength

    # Count enemy units within 1, 2, 3 steps of my VP targets
    en_t1 = en_t2 = en_t3 = 0
    for rkey, (regs, elites, leaders, nazgul) in en_armies.items():
        strength = regs + elites
        d = dist_to_my.get(rkey, 99)
        if d <= 1:
            en_t1 += strength
        if d <= 2:
            en_t2 += strength
        if d <= 3:
            en_t3 += strength

    # Count targets under threat (enemy within 1 step)
    my_threatened = 0
    for rkey, _ in my_targets:
        for en_rkey in en_armies:
            d_to_target = _bfs_distance_cached(en_rkey, rkey)
            if d_to_target is not None and d_to_target <= 1:
                my_threatened += 1
                break
    en_threatened = 0
    for rkey, _ in en_targets:
        for my_rkey in my_armies:
            d_to_target = _bfs_distance_cached(my_rkey, rkey)
            if d_to_target is not None and d_to_target <= 1:
                en_threatened += 1
                break

    return [
        my_t1 / 20.0, my_t2 / 30.0, my_t3 / 40.0,
        en_t1 / 20.0, en_t2 / 30.0, en_t3 / 40.0,
        my_threatened / 10.0,
        en_threatened / 10.0,
    ]


_FP_TARGET_KEYS = frozenset(t[0] for t in _FP_VP_TARGETS)
_SH_TARGET_KEYS = frozenset(t[0] for t in _SH_VP_TARGETS)


def _bfs_distance_cached(src: str, dst: str) -> 'Optional[int]':
    """Get BFS distance between two regions (cached)."""
    if src == dst:
        return 0
    # Use precomputed distances if available
    if dst in _FP_TARGET_KEYS:
        return _DIST_TO_FP_TARGET.get(src, 99)
    if dst in _SH_TARGET_KEYS:
        return _DIST_TO_SH_TARGET.get(src, 99)
    return None


# ===================================================================
# Strategic planning features
# ===================================================================

# Precompute per-settlement 1-hop and 2-hop neighbor sets
_FP_SETTLE_NEIGHBORS_1: Dict[str, set] = {}  # key -> set of 1-hop neighbors
_FP_SETTLE_NEIGHBORS_2: Dict[str, set] = {}  # key -> set of 2-hop neighbors
_SH_SETTLE_NEIGHBORS_1: Dict[str, set] = {}
_SH_SETTLE_NEIGHBORS_2: Dict[str, set] = {}

def _precompute_settlement_neighbors():
    for skey, _ in _FP_VP_TARGETS:
        n1 = set(board.adj.get(skey, set()))
        n2 = set(n1)
        for nb in n1:
            n2.update(board.adj.get(nb, set()))
        n2.discard(skey)
        _FP_SETTLE_NEIGHBORS_1[skey] = n1
        _FP_SETTLE_NEIGHBORS_2[skey] = n2
    for skey, _ in _SH_VP_TARGETS:
        n1 = set(board.adj.get(skey, set()))
        n2 = set(n1)
        for nb in n1:
            n2.update(board.adj.get(nb, set()))
        n2.discard(skey)
        _SH_SETTLE_NEIGHBORS_1[skey] = n1
        _SH_SETTLE_NEIGHBORS_2[skey] = n2

_precompute_settlement_neighbors()


def _compute_strategic_features(
        army_by_region: Dict[str, 'Army'],
        state: 'GameState',
        my_side: Side,
        enemy_side: Side,
        vp: dict) -> List[float]:
    """Compute 12 strategic planning features for the value network.

    These give the network enough information to reason about:
    - How close each side is to a military VP win (VP-weighted threat proximity)
    - Defensive vulnerabilities (undefended settlements, weakest defense)
    - Fellowship timeline (total steps remaining to destroy ring)
    - Game urgency (corruption margin, game progress, VP distance to win)
    """
    # Determine which settlements are "mine" and "enemy's" (original ownership)
    if my_side == Side.FREE_PEOPLES:
        my_targets = _FP_VP_TARGETS    # FP settlements (I defend these)
        en_targets = _SH_VP_TARGETS    # Shadow settlements (I attack these)
        my_n1, my_n2 = _FP_SETTLE_NEIGHBORS_1, _FP_SETTLE_NEIGHBORS_2
        en_n1, en_n2 = _SH_SETTLE_NEIGHBORS_1, _SH_SETTLE_NEIGHBORS_2
    else:
        my_targets = _SH_VP_TARGETS
        en_targets = _FP_VP_TARGETS
        my_n1, my_n2 = _SH_SETTLE_NEIGHBORS_1, _SH_SETTLE_NEIGHBORS_2
        en_n1, en_n2 = _FP_SETTLE_NEIGHBORS_1, _FP_SETTLE_NEIGHBORS_2

    # --- VP-weighted threat to MY settlements (enemy attacking me) ---
    en_vp_threat_1 = 0  # VP of my settlements with enemy army within 1 hop
    en_vp_threat_2 = 0  # VP of my settlements with enemy army within 2 hops
    en_max_atk = 0      # Max enemy army strength adjacent to my settlements
    my_min_def = 5      # Min defense strength of my settlements
    my_undef = 0        # Count of my undefended settlements

    for skey, svp in my_targets:
        # Check defense
        def_army = army_by_region.get(skey)
        if def_army:
            def_str = 0
            for u in def_army.units:
                if _side_of_nation(u.nation) == my_side:
                    def_str += 1  # Each unit = 1 combat die
            def_str = min(5, def_str)
        else:
            def_str = 0
        if def_str == 0:
            my_undef += 1
        my_min_def = min(my_min_def, def_str)

        # Check 1-hop enemy threat
        found_1 = False
        for nbr in my_n1.get(skey, set()):
            a = army_by_region.get(nbr)
            if a:
                atk_str = 0
                for u in a.units:
                    if _side_of_nation(u.nation) == enemy_side:
                        atk_str += 1  # Each unit = 1 combat die
                if atk_str > 0:
                    found_1 = True
                    en_max_atk = max(en_max_atk, min(5, atk_str))
        if found_1:
            en_vp_threat_1 += svp

        # Check 2-hop enemy threat
        found_2 = found_1
        if not found_2:
            for nbr in my_n2.get(skey, set()):
                a = army_by_region.get(nbr)
                if a:
                    for u in a.units:
                        if _side_of_nation(u.nation) == enemy_side:
                            found_2 = True
                            break
                if found_2:
                    break
        if found_2:
            en_vp_threat_2 += svp

    # --- VP-weighted threat to ENEMY settlements (me attacking them) ---
    my_vp_threat_1 = 0
    my_vp_threat_2 = 0
    en_undef = 0

    for skey, svp in en_targets:
        # Check if enemy settlement is undefended
        def_army = army_by_region.get(skey)
        has_def = False
        if def_army:
            for u in def_army.units:
                if _side_of_nation(u.nation) == enemy_side:
                    has_def = True
                    break
        if not has_def:
            en_undef += 1

        # Check 1-hop my threat
        found_1 = False
        for nbr in en_n1.get(skey, set()):
            a = army_by_region.get(nbr)
            if a:
                for u in a.units:
                    if _side_of_nation(u.nation) == my_side:
                        found_1 = True
                        break
            if found_1:
                break
        if found_1:
            my_vp_threat_1 += svp

        # Check 2-hop my threat
        found_2 = found_1
        if not found_2:
            for nbr in en_n2.get(skey, set()):
                a = army_by_region.get(nbr)
                if a:
                    for u in a.units:
                        if _side_of_nation(u.nation) == my_side:
                            found_2 = True
                            break
                if found_2:
                    break
        if found_2:
            my_vp_threat_2 += svp

    # --- Fellowship timeline ---
    f = state.fellowship
    if f.mordor_entered:
        steps_rem = 5 - f.mordor_step
    else:
        cur_region = game_logic.fellowship_region(state)
        dist_to_gate = _DIST_TO_MORDOR.get(cur_region, 15)
        steps_rem = dist_to_gate + 5  # distance to gate + 5 mordor steps

    # --- VP to win (enemy perspective: how far is my enemy from winning) ---
    # FIX (full-pass review 2026-07-27): the enemy's OWN victory
    # threshold — Shadow needs 10, FP needs 4 (FP_VP_VICTORY).  The old
    # code used the Shadow threshold for both perspectives, overstating
    # FP's distance by 6 from the Shadow side's view.
    if my_side == Side.FREE_PEOPLES:
        en_vp_to_win = max(0, game_logic.SHADOW_VP_VICTORY - vp['shadow'])
    else:
        en_vp_to_win = max(0, game_logic.FP_VP_VICTORY
                           - vp['free_peoples'])

    return [
        en_vp_threat_1 / 10.0,             # VP of my settlements enemy can reach in 1
        en_vp_threat_2 / 10.0,             # VP of my settlements enemy can reach in 2
        my_vp_threat_1 / 10.0,             # VP of enemy settlements I can reach in 1
        my_vp_threat_2 / 10.0,             # VP of enemy settlements I can reach in 2
        en_max_atk / 5.0,                  # Strongest enemy army near my settlements
        my_min_def / 5.0,                  # Weakest defense of my settlements
        my_undef / 12.0,                   # My undefended settlements
        en_undef / 10.0,                   # Enemy undefended settlements
        steps_rem / 20.0,                  # Fellowship total steps to ring destruction
        (12 - f.corruption) / 12.0,        # Corruption margin (1=healthy, 0=about to lose)
        state.turn / 25.0,                 # Game progress (urgency signal)
        en_vp_to_win / 10.0,              # How far enemy is from military VP win
    ]

STRATEGIC_FEATURE_COUNT = 12


# ===================================================================
# Step 1: State Cloning
# ===================================================================

def clone_state(state: GameState) -> GameState:
    """Deep-copy a GameState for search.

    Custom shallow-copy with selective deep-copy of mutable containers.
    The Board object is shared (immutable). Most scalar fields are copied
    by value automatically. Only containers (dicts, lists, sets) that
    could be mutated during action application are deep-copied.
    """
    board_obj = state.board

    # Deep-copy armies (the most complex mutable structure)
    new_armies = {}
    for k, army in state.armies.items():
        new_army = Army(region=army.region)
        new_army.units = [Unit(u.nation, u.kind) for u in army.units]
        new_army.leaders = list(army.leaders)  # Leader/Nazgul are small, share refs
        new_armies[k] = new_army

    # Deep-copy dice
    new_dice = {}
    for side, dice_list in state.dice.items():
        new_list = []
        for d in dice_list:
            nd = ActionDie(d.faces, d.side)
            nd.current = d.current
            nd.used = d.used
            new_list.append(nd)
        new_dice[side] = new_list

    # Rebuild hunt_box referencing new dice objects
    new_hunt_box = []
    for hd in state.hunt_box:
        # Find the corresponding new die
        for side, old_list in state.dice.items():
            for i, od in enumerate(old_list):
                if od is hd:
                    new_hunt_box.append(new_dice[side][i])
                    break

    # Deep-copy fellowship
    new_companions = list(state.fellowship.companions)  # Share Companion refs
    new_guide = state.fellowship.guide
    new_fellowship = Fellowship(
        companions=new_companions,
        guide=new_guide,
        region=state.fellowship.region,
        progress=state.fellowship.progress,
        corruption=state.fellowship.corruption,
        hidden=state.fellowship.hidden,
        mordor_entered=state.fellowship.mordor_entered,
        mordor_step=state.fellowship.mordor_step,
    )

    # Deep-copy hands
    new_hands = {}
    for side, pi in state.hands.items():
        new_pi = PlayerInfo()
        new_pi.hand = list(pi.hand)
        new_pi.discarded_face_down = list(pi.discarded_face_down)
        new_hands[side] = new_pi

    # Deep-copy political track
    new_political = PoliticalTrack(
        boxes=dict(state.political.boxes),
        active=dict(state.political.active),
    )

    # Deep-copy VP
    new_vp = VictoryPoints(
        free_peoples=state.vp.free_peoples,
        shadow=state.vp.shadow,
    )

    # Shallow-copy sets and simple containers
    new_enemy_controlled = {
        s: set(v) for s, v in state.enemy_controlled.items()
    }

    s2 = GameState(
        board=board_obj,
        armies=new_armies,
        enemy_controlled=new_enemy_controlled,
        fellowship=new_fellowship,
        political=new_political,
        vp=new_vp,
        dice=new_dice,
        turn=state.turn,
        hunt_pool=list(state.hunt_pool),
        hunt_discard=list(state.hunt_discard),
        hunt_removed=list(state.hunt_removed),
        hunt_box=new_hunt_box,
        decks=DeckStatus(
            state.decks.fp_character, state.decks.fp_strategy,
            state.decks.sh_character, state.decks.sh_strategy,
        ),
        hands=new_hands,
        discard_face_up={k: list(v) for k, v in state.discard_face_up.items()},
        fp_elven_rings=state.fp_elven_rings,
        fp_dwarven_rings=state.fp_dwarven_rings,
        fp_political_token=state.fp_political_token,
        fp_card_token=state.fp_card_token,
        sh_elven_rings=state.sh_elven_rings,
        sh_dwarven_rings=state.sh_dwarven_rings,
        sh_action_tokens=state.sh_action_tokens,
        fp_ring_used_this_turn=state.fp_ring_used_this_turn,
        sh_ring_used_this_turn=state.sh_ring_used_this_turn,
        # A7: the Elven-Ring legality waiver is pending-shaped, and this
        # function copies attributes ONE BY ONE - a field missing here
        # vanishes on every MCTS clone, silently.
        fp_ring_waiver=getattr(state, 'fp_ring_waiver', False),
        sh_ring_waiver=getattr(state, 'sh_ring_waiver', False),
        sh_mouth_muster_used_this_turn=getattr(
            state, 'sh_mouth_muster_used_this_turn', False),
        fp_token_used_this_turn=getattr(
            state, 'fp_token_used_this_turn', False),
        sh_token_used_this_turn=getattr(
            state, 'sh_token_used_this_turn', False),
        dice_rolled=state.dice_rolled,
        witch_king=dataclasses.replace(state.witch_king) if state.witch_king else None,
        saruman=dataclasses.replace(state.saruman) if state.saruman else None,
        mouth_of_sauron=dataclasses.replace(state.mouth_of_sauron) if state.mouth_of_sauron else None,
        gandalf_the_white=dataclasses.replace(state.gandalf_the_white) if state.gandalf_the_white else None,
        # 1.137 (USER bug report): aragorn_crowned was a SHARED
        # reference - an MCTS clone considering crown_aragorn set
        # crowned=True on the LIVE state: Aragorn appeared in the GUI
        # before any crowning action, the real crown action vanished
        # from the menu ('already crowned') so no die was ever spent,
        # and calculate_bonus_dice granted FP a spurious +1 die.
        aragorn_crowned=dataclasses.replace(state.aragorn_crowned)
        if state.aragorn_crowned else None,
        gollum=state.gollum,
        fp_bonus_dice=state.fp_bonus_dice,
        sh_bonus_dice=state.sh_bonus_dice,
        # Own PersistentCardState per clone — sharing it let in-tree card
        # plays (Threats and Promises etc.) land on the TRUE game's table
        persistent_cards=PersistentCardState(
            fp_cards=list(state.persistent_cards.fp_cards),
            sh_cards=list(state.persistent_cards.sh_cards),
        ),
        pending_hunt_tiles=list(state.pending_hunt_tiles),
    )

    # Copy dynamically added deck attributes if they exist
    for attr in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        deck = getattr(state, attr, None)
        if deck is not None:
            setattr(s2, attr, list(deck))

    # Copy pending multi-step die action state
    s2.pending_army_move_die = state.pending_army_move_die
    s2.pending_army_move_from = state.pending_army_move_from
    s2.pending_army_move_side = state.pending_army_move_side
    _pco = getattr(state, 'pending_card_option', None)
    s2.pending_card_option = (
        dict(_pco, options=list(_pco['options'])) if _pco else None)
    if _pco and _pco.get('resume_action'):
        s2.pending_card_option['resume_action'] = dict(
            _pco['resume_action'],
            params=dict(_pco['resume_action'].get('params') or {}))
    s2._crebain_hunt_bonus = getattr(state, '_crebain_hunt_bonus', 0)
    s2._reveal_place_due = getattr(state, '_reveal_place_due', False)
    # (stale-state sweep 2026-07-27: legacy _saruman/_wk/_mouth
    # in-play flags DELETED — canonical .in_play + map presence are
    # the single source of truth)
    s2._ents_chain_n = getattr(state, '_ents_chain_n', 0)
    _ms9 = getattr(state, '_mithril_stash', None)
    s2._mithril_stash = dict(_ms9) if _ms9 else None
    _wss9 = getattr(state, '_wizardstaff_stash', None)
    s2._wizardstaff_stash = dict(_wss9) if _wss9 else None
    _ovr9 = getattr(state, '_overstack_regions', None)
    if _ovr9:
        s2._overstack_regions = set(_ovr9)
    _pcf = getattr(state, 'pending_card_followup', None)
    # the exclude set is mutated by the followup executor: a shared set
    # would let in-tree moves pollute the real follow-up menu
    s2.pending_card_followup = (
        dict(_pcf, exclude=set(_pcf.get('exclude') or ()))
        if _pcf else None)
    # engine-declared draw debt (game_logic.grant_draw): the live driver
    # reads it off the clone apply_action returns - copy, never share
    s2._owed_draws = dict(getattr(state, '_owed_draws', None) or {})
    # free-action routing marker (elven_ring): _next_acting_side keeps
    # the ring user's seat until its funded action lands
    s2._ring_free_side = getattr(state, '_ring_free_side', None)
    s2.pending_army_move_preexist = (
        dict(state.pending_army_move_preexist)
        if getattr(state, 'pending_army_move_preexist', None) else None)
    s2.pending_muster_die = state.pending_muster_die
    s2.pending_muster_first_type = state.pending_muster_first_type
    s2.pending_muster_region = state.pending_muster_region
    s2.pending_muster_side = state.pending_muster_side

    # Copy FP permanent casualties tracking
    fp_cas = getattr(state, 'fp_casualties', None)
    if fp_cas:
        s2.fp_casualties = {nat: dict(counts) for nat, counts in fp_cas.items()}

    # Copy hunt box breakdown tracking
    s2.eyes_allocated = getattr(state, 'eyes_allocated', 0)


    # Copy pending multi-phase states (dicts of primitives — shallow copy suffices,
    # except rearguard_units/rearguard_leaders which are lists of dataclasses)
    pc = getattr(state, 'pending_combat', None)
    if pc is not None:
        s2.pending_combat = dict(pc)
        if 'rearguard_units' in pc:
            s2.pending_combat['rearguard_units'] = list(pc['rearguard_units'])
        if 'rearguard_leaders' in pc:
            s2.pending_combat['rearguard_leaders'] = list(pc['rearguard_leaders'])
        ec = pc.get('effect_choice')
        if isinstance(ec, dict):
            ec2 = dict(ec)
            if isinstance(ec.get('counts'), list):
                ec2['counts'] = list(ec['counts'])
            s2.pending_combat['effect_choice'] = ec2
        # hidden combat-card commits (K1): own dict per clone
        for _hk9 in ('atk_hidden_cc', 'def_hidden_cc'):
            if isinstance(pc.get(_hk9), dict):
                s2.pending_combat[_hk9] = dict(pc[_hk9])
    else:
        s2.pending_combat = None
    ph = getattr(state, 'pending_hunt', None)
    s2.pending_hunt = dict(ph) if ph is not None else None
    # W11 EDGE1 (2026-07-20): clone the scalar hunt-damage pending too
    # (session-27 invariant: every pending field must survive a clone).
    # Safe today via pending_hunt['damage'], but explicit is robust.
    s2._pending_hunt_damage = getattr(state, '_pending_hunt_damage', 0)
    pg = getattr(state, 'pending_guide_change', None)
    s2.pending_guide_change = dict(pg) if pg is not None else None
    pd = getattr(state, 'pending_discard', None)
    s2.pending_discard = dict(pd) if pd is not None else None
    # dynamic decision-phase flags (eye allocation broke silently in
    # MCTS because the root clone lost the phase flag)
    s2._eye_allocation_phase = getattr(state, '_eye_allocation_phase',
                                       False)
    s2._declare_phase = getattr(state, '_declare_phase', False)
    # v30 cross-turn search: the crossing budget MUST survive clones or
    # in-tree lookahead would cross turn boundaries without limit
    s2._search_turn_crossings = getattr(state, '_search_turn_crossings', 0)
    s2._hobbit_place_queue = list(getattr(state, '_hobbit_place_queue',
                                          None) or [])
    # review P2 (2026-07-27): the once-per-region stronghold hunt-tile
    # guard must survive clones or an in-tree second pass through the
    # same stronghold re-draws the tile, inflating hunt damage on that
    # search branch (getattr-None masked the loss).  Preserve None vs
    # set: None means "no reveal event open" and must stay None.
    _stdr9 = getattr(state, '_sh_tile_drawn_regions', None)
    s2._sh_tile_drawn_regions = set(_stdr9) if _stdr9 is not None else None
    # 1.211f: Worn with Sorrow deferred-discard queue is a pending
    # decision list - it MUST survive a clone (MCTS root determinization
    # / tree reuse) or an owed discard opened mid-search would vanish.
    s2._wwsat_owed = list(getattr(state, '_wwsat_owed', None) or [])
    # DOWNGRADE-SOURCE LEDGER (BRIDGE-9 2026-08-18).  combat._dg_stamp
    # appends (nation, 'cas'|'pool'|None) per downgrade and the emitter
    # slices `post[len(pre):]` to get the ones THIS transition made.
    # Without it on the clone, `pre` always looked empty and the slice
    # took every stamp on the post state - over-attributing an earlier
    # round's downgrades to a later one in a multi-round battle.  The
    # per-clone cost was MEASURED before adding it: 0.14 us on a 40.3 us
    # clone (0.34%), i.e. negligible against the correctness of a
    # per-round boundary, and it keeps the field handled the same way
    # as every other pending field here.
    s2._dg_src_log = list(getattr(state, '_dg_src_log', None) or [])
    s2._fp_used_hunt_box_last_turn = getattr(
        state, '_fp_used_hunt_box_last_turn', False)
    # USER 1.208m: the deferred Palantir bonus-draw flag is a pending
    # decision field - it must survive a clone (MCTS root determinization
    # / tree reuse) or a Palantir draw opened mid-battle would vanish.
    s2._palantir_draw_pending = getattr(state, '_palantir_draw_pending',
                                        False)

    # Copy besieged strongholds set
    bs = getattr(state, 'besieged_strongholds', set())
    s2.besieged_strongholds = set(bs)

    # Copy Mordor activity tracking
    s2.fellowship_acted_in_mordor = getattr(state, 'fellowship_acted_in_mordor', False)

    # Turn-routing marker (Side or None): set when a BATTLE resolves so the
    # next die-action passes to the NON-attacker (the whole battle was the
    # attacker's single action, regardless of who made the last combat
    # sub-decision). See _execute_action / _next_acting_side.
    s2._battle_resolved_attacker = getattr(
        state, '_battle_resolved_attacker', None)

    return s2


# ===================================================================
# Step 2: State-to-Vector Encoding
# ===================================================================

def encode_state(state: GameState, side: Side) -> np.ndarray:
    """Encode a GameState from the perspective of `side` into a fixed-length
    numeric feature vector.

    The same network evaluates both sides; the encoding flips perspective
    so "my" always refers to the encoding side.
    """
    feats: List[float] = []

    my_side = side
    enemy_side = Side.SHADOW if side == Side.FREE_PEOPLES else Side.FREE_PEOPLES

    # --- Pre-index armies by sanitized key ---
    army_by_region: Dict[str, Army] = {}
    for akey, army in state.armies.items():
        army_by_region[sanitize(akey)] = army

    my_controlled = state.enemy_controlled.get(my_side, set())
    en_controlled = state.enemy_controlled.get(enemy_side, set())

    # Fellowship pawn location (critical for hunt rerolls, healing, etc.)
    try:
        _fship_rkey = game_logic.fellowship_region(state)
    except Exception:
        _fship_rkey = None

    # --- Armies per region (12 features × NUM_REGIONS) ---
    for rkey in ALL_REGION_KEYS:
        my_reg = my_elite = my_lead = 0
        en_reg = en_elite = en_lead = 0
        my_naz = en_naz = 0
        fp_named = sh_named = 0  # Companions / Minions present here
        nat_counts = [0.0] * (2 * len(ALL_NATIONS))  # per-nation reg/elite
        in_siege = 1.0 if rkey in getattr(state, 'besieged_strongholds', set()) \
            else 0.0

        army = army_by_region.get(rkey)
        if army:
            for u in army.units:
                s = _side_of_nation(u.nation)
                if s == my_side:
                    if u.kind == UnitKind.REGULAR:
                        my_reg += 1
                    else:
                        my_elite += 1
                else:
                    if u.kind == UnitKind.REGULAR:
                        en_reg += 1
                    else:
                        en_elite += 1
                # nation-identity composition (iter 36): what the human
                # SEES on the board — which nation's figures stand here
                ni = ALL_NATIONS.index(u.nation)
                if u.kind == UnitKind.REGULAR:
                    nat_counts[2 * ni] += 1.0 / 15.0
                else:
                    nat_counts[2 * ni + 1] += 1.0 / 5.0
            for ld in army.leaders:
                if getattr(ld, 'name', '') in ('Witch King', 'Saruman',
                                               'Mouth of Sauron'):
                    # FIX (minion typing): Shadow, even if a stray
                    # Companion-typed minion slips through
                    s = Side.SHADOW
                    sh_named += 1
                elif isinstance(ld, Companion):
                    s = Side.FREE_PEOPLES
                    fp_named += 1
                elif isinstance(ld, Nazgul):
                    s = Side.SHADOW
                    if my_side == Side.SHADOW:
                        my_naz += 1
                    else:
                        en_naz += 1
                elif type(ld).__name__ == 'Minion':
                    s = Side.SHADOW
                    sh_named += 1
                elif hasattr(ld, 'nation'):
                    s = _side_of_nation(ld.nation)
                else:
                    continue
                if s == my_side:
                    my_lead += 1
                else:
                    en_lead += 1

        my_ctrl = 1.0 if rkey in my_controlled else 0.0
        en_ctrl = 1.0 if rkey in en_controlled else 0.0
        my_named, en_named = (fp_named, sh_named) \
            if my_side == Side.FREE_PEOPLES else (sh_named, fp_named)

        feats.extend([
            my_reg / 10.0, my_elite / 10.0, my_lead / 7.0,
            en_reg / 10.0, en_elite / 10.0, en_lead / 7.0,
            in_siege, my_ctrl, en_ctrl,
            1.0 if rkey == _fship_rkey else 0.0,
            my_named / 3.0, en_named / 3.0,
            my_naz / 8.0, en_naz / 8.0,
        ])
        feats.extend(nat_counts)

    # --- Political track (4 features × 8 nations) ---
    for nat in ALL_NATIONS:
        pos = state.political.boxes.get(nat, 0)
        active = 1.0 if state.political.active.get(nat, False) else 0.0
        at_war = 1.0 if pos >= 3 else 0.0
        is_mine = 1.0 if _side_of_nation(nat) == my_side else 0.0
        feats.extend([pos / 3.0, active, at_war, is_mine])

    # --- Fellowship (8 features: 7 original + 1 distance to Mordor) ---
    f = state.fellowship
    if f.mordor_entered:
        fship_dist = 0.0  # Already in Mordor
    else:
        cur_region = game_logic.fellowship_region(state)
        fship_dist = _DIST_TO_MORDOR.get(cur_region, _MAX_MORDOR_DIST) / _MAX_MORDOR_DIST
    feats.extend([
        f.progress / 12.0,
        f.corruption / 12.0,
        1.0 if f.hidden else 0.0,
        1.0 if f.mordor_entered else 0.0,
        f.mordor_step / 5.0,
        len(f.companions) / 8.0,
        (f.guide.level if f.guide else 0) / 3.0,
        1.0 - fship_dist,  # Proximity to Mordor: 0=far, 1=at gate
    ])

    # --- VP (computed live from enemy_controlled, not stale state.vp) ---
    vp = game_logic.calculate_victory_points(state)
    my_vp = vp['free_peoples'] if my_side == Side.FREE_PEOPLES else vp['shadow']
    en_vp = vp['shadow'] if my_side == Side.FREE_PEOPLES else vp['free_peoples']
    feats.extend([my_vp / 10.0, en_vp / 10.0])

    # --- Hunt pool composition ---
    pool = state.hunt_pool
    n_eye = sum(1 for t in pool if t.is_eye)
    n_special_shadow = sum(1 for t in pool if getattr(t, 'is_special', False)
                           and not getattr(t, 'is_fellowship_special', False))
    n_special_fp = sum(1 for t in pool if getattr(t, 'is_fellowship_special', False))
    n_stop = sum(1 for t in pool if getattr(t, 'stops_movement', False))
    dmg_tiles = [t.damage for t in pool if not t.is_eye and t.damage > 0]
    avg_damage = sum(dmg_tiles) / max(1, len(dmg_tiles)) if dmg_tiles else 0.0

    # Hunt box breakdown: distinguish allocated eyes, rolled eyes, FP dice
    n_allocated_eyes = getattr(state, 'eyes_allocated', 0)
    n_sh_in_box = sum(1 for d in state.hunt_box if d.side == Side.SHADOW)
    n_rolled_eyes = max(0, n_sh_in_box - n_allocated_eyes)
    n_fp_in_box = sum(1 for d in state.hunt_box if d.side == Side.FREE_PEOPLES)
    feats.extend([
        n_allocated_eyes / 5.0,         # Shadow voluntary eye allocation (0-5)
        n_rolled_eyes / 4.0,            # Shadow dice that rolled Eye face (0-4)
        n_fp_in_box / 5.0,              # FP dice committed to hunt (fellowship moves)
        len(pool) / 16.0,
        n_eye / 4.0,                    # Eye tiles (max 4 in base game)
        n_special_shadow / 4.0,         # Shadow special tiles
        n_special_fp / 4.0,             # FP special tiles (beneficial)
        n_stop / 4.0,                   # Stop tiles (critical in Mordor)
        avg_damage / 3.0,               # Average non-eye damage
        1.0 if state.fellowship.mordor_entered else 0.0,  # Mordor pool constructed
    ])

    # --- Dice (mine AND the enemy's visible faces) ---
    my_dice = state.dice.get(my_side, [])
    en_dice = state.dice.get(enemy_side, [])
    my_unused = sum(1 for d in my_dice if not d.used and d.current is not None)
    en_unused = sum(1 for d in en_dice if not d.used and d.current is not None)
    feats.extend([my_unused / 10.0, en_unused / 10.0])

    # Per-face counts for BOTH sides (dice are public information)
    for dice_pool in (my_dice, en_dice):
        face_counts = {f: 0 for f in ActionDieFace}
        for d in dice_pool:
            if not d.used and d.current is not None:
                face_counts[d.current] += 1
        for face in ActionDieFace:
            feats.append(face_counts[face] / 4.0)

    # --- Cards: hand sizes, enemy hand composition, ALL four deck counts ---
    my_hand = state.hands.get(my_side, PlayerInfo())
    en_hand = state.hands.get(enemy_side, PlayerInfo())
    # Enemy hand composition by deck type (which deck each draw came from
    # is public knowledge)
    # deck composition via the ONE classifier: live human-hand
    # placeholders (HU_<side>_CHAR_n / _STRAT_n) count by deck too
    en_char_in_hand = sum(1 for c in en_hand.hand
                          if game_logic.card_deck_of(c) == 'character')
    en_strat_in_hand = sum(1 for c in en_hand.hand
                           if game_logic.card_deck_of(c) == 'strategy')
    feats.extend([
        len(my_hand.hand) / 6.0,
        len(en_hand.hand) / 6.0,
        en_char_in_hand / 6.0,
        en_strat_in_hand / 6.0,
        # stale-state sweep 2026-07-27: DeckStatus was NEVER updated
        # (four constant 1.0 features since day one) — encode the REAL
        # deck depletion from the live deck lists
        len(getattr(state, 'fp_char_deck', []) or []) / 24.0,
        len(getattr(state, 'fp_strat_deck', []) or []) / 24.0,
        len(getattr(state, 'sh_char_deck', []) or []) / 24.0,
        len(getattr(state, 'sh_strat_deck', []) or []) / 24.0,
    ])

    # --- Card hand: raw card indices for embedding layer (6 features) ---
    # Each slot is a card index (0-95) or -1 for empty/unknown.
    # The network's embedding layer will look up learned per-card representations.
    card_map = _get_card_idx_map()
    my_card_indices = []
    for cid in my_hand.hand[:MAX_HAND_SIZE]:
        my_card_indices.append(float(card_map.get(cid, -1)))
    while len(my_card_indices) < MAX_HAND_SIZE:
        my_card_indices.append(-1.0)
    feats.extend(my_card_indices)

    # --- Card status: 96 dims, one per card ---
    # 0 = unseen, 0.5 = seen (played/discarded), 1.0 = on the table now.
    # Card-counting: knowing what has been played narrows what remains.
    card_status = [0.0] * NUM_CARD_IDS
    for _deck_list in getattr(state, 'discard_face_up', {}).values():
        for cid in _deck_list:
            idx = card_map.get(cid, -1)
            if idx >= 0:
                card_status[idx] = 0.5
    pc_state = state.persistent_cards
    if pc_state is not None:
        for _pcard in (list(getattr(pc_state, 'fp_cards', []))
                       + list(getattr(pc_state, 'sh_cards', []))):
            idx = card_map.get(getattr(_pcard, 'card_id', ''), -1)
            if idx >= 0:
                card_status[idx] = 1.0
    feats.extend(card_status)

    # --- Special characters ---
    gw = 1.0 if (state.gandalf_the_white and state.gandalf_the_white.in_play) else 0.0
    ac = 1.0 if (state.aragorn_crowned and state.aragorn_crowned.crowned) else 0.0
    wk = 1.0 if (state.witch_king and state.witch_king.in_play) else 0.0
    sa = 1.0 if (state.saruman and state.saruman.in_play) else 0.0
    mo = 1.0 if (state.mouth_of_sauron and state.mouth_of_sauron.in_play) else 0.0

    if my_side == Side.FREE_PEOPLES:
        feats.extend([gw, ac, wk, sa, mo])
    else:
        # Flip perspective
        feats.extend([wk, sa, mo, gw, ac])

    # --- Resources ---
    if my_side == Side.FREE_PEOPLES:
        feats.extend([
            state.fp_elven_rings / 3.0,
            state.fp_dwarven_rings / 1.0 if state.fp_dwarven_rings else 0.0,
            (1.0 if state.fp_political_token else 0.0),
            state.sh_elven_rings / 3.0,
            state.sh_dwarven_rings / 1.0 if state.sh_dwarven_rings else 0.0,
            state.sh_action_tokens / 3.0,
        ])
    else:
        feats.extend([
            state.sh_elven_rings / 3.0,
            state.sh_dwarven_rings / 1.0 if state.sh_dwarven_rings else 0.0,
            state.sh_action_tokens / 3.0,
            state.fp_elven_rings / 3.0,
            state.fp_dwarven_rings / 1.0 if state.fp_dwarven_rings else 0.0,
            (1.0 if state.fp_political_token else 0.0),
        ])

    # --- Turn ---
    feats.append(state.turn / 15.0)

    # --- Nations at war (normalised counts) ---
    feats.append(sum(1 for n in FP_NATIONS if state.political.is_at_war(n)) / 5.0)
    feats.append(sum(1 for n in SHADOW_NATIONS if state.political.is_at_war(n)) / 3.0)

    # --- Companion details (10 slots: alive, in_fellowship, level,
    # location) --- (iter 36: + Aragorn and Gandalf the White — the
    # crowned/transformed figures are Companions with these names and
    # previously had NO identity/location encoding, only in_play flags)
    companion_names = [
        "Gandalf the Grey", "Strider", "Legolas", "Gimli",
        "Boromir", "Meriadoc", "Peregrin", "Gollum",
        "Aragorn", "Gandalf the White",
    ]
    # Build alive set: fellowship companions + separated (in army leaders)
    fellowship_names = {c.name for c in state.fellowship.companions}
    alive_companions = set(fellowship_names)
    separated_levels = {}    # name -> level for separated companions
    separated_regions = {}   # name -> region index on the board
    for akey, army in state.armies.items():
        for ld in army.leaders:
            if isinstance(ld, Companion):
                alive_companions.add(ld.name)
                separated_levels[ld.name] = ld.level
                separated_regions[ld.name] = REGION_INDEX.get(sanitize(akey), -1)
    for cname in companion_names:
        alive = 1.0 if cname in alive_companions else 0.0
        in_fellowship = 1.0 if cname in fellowship_names else 0.0
        level = 0.0
        for c in state.fellowship.companions:
            if c.name == cname:
                level = c.level / 3.0
                break
        # Separated companions: still alive with level but not in fellowship
        if alive > 0 and level == 0 and cname in separated_levels:
            level = separated_levels[cname] / 3.0
        # Board location for separated companions ((idx+1)/105, 0 = not on map)
        loc = 0.0
        ridx = separated_regions.get(cname, -1)
        if ridx >= 0 and not in_fellowship:
            loc = (ridx + 1) / float(NUM_REGIONS)
        feats.extend([alive, in_fellowship, level, loc])

    # --- Minion locations (3: Witch-king, Saruman, Mouth of Sauron) ---
    minion_locs = {'Witch King': 0.0, 'Saruman': 0.0, 'Mouth of Sauron': 0.0}
    for akey, army in state.armies.items():
        for ld in army.leaders:
            if type(ld).__name__ == 'Minion' or \
                    getattr(ld, 'name', '') in minion_locs:
                nm = getattr(ld, 'name', '')
                if nm in minion_locs:
                    ridx = REGION_INDEX.get(sanitize(akey), -1)
                    if ridx >= 0:
                        minion_locs[nm] = (ridx + 1) / float(NUM_REGIONS)
    feats.extend([minion_locs['Witch King'], minion_locs['Saruman'],
                  minion_locs['Mouth of Sauron']])

    # --- Nation units on board (as proxy for remaining pool) ---
    # Count units on the board per nation
    nation_unit_counts: Dict[Nation, List[int]] = {
        n: [0, 0] for n in ALL_NATIONS
    }
    for akey, army in state.armies.items():
        for u in army.units:
            idx = 0 if u.kind == UnitKind.REGULAR else 1
            nation_unit_counts[u.nation][idx] += 1

    for nat in ALL_NATIONS:
        r, e = nation_unit_counts[nat]
        # v26 (W10 F3): normalise by the nation's OWN max pool — the flat
        # /15, /5 left Sauron's 28 setup regulars reading 1.87 (>1) and
        # compressed small nations (same class as the historical /7-vs-/8
        # count bugs). Values change, dims do not.
        _mx26 = NATION_MAX_FIGURES.get(nat, {})
        feats.extend([r / float(max(1, _mx26.get('regular', 15))),
                      e / float(max(1, _mx26.get('elite', 5)))])

    # --- Military threat features (8) ---
    # Build per-region army dicts in (reg, elite, lead, nazgul) format
    my_army_dict = {}
    en_army_dict = {}
    for rkey in ALL_REGION_KEYS:
        army = army_by_region.get(rkey)
        if not army:
            continue
        mr = me = ml = mn = 0
        er = ee = el = en_naz = 0
        for u in army.units:
            s = _side_of_nation(u.nation)
            if s == my_side:
                if u.kind == UnitKind.REGULAR:
                    mr += 1
                else:
                    me += 1
            else:
                if u.kind == UnitKind.REGULAR:
                    er += 1
                else:
                    ee += 1
        for ld in army.leaders:
            if getattr(ld, 'name', '') in ('Witch King', 'Saruman',
                                           'Mouth of Sauron'):
                s = Side.SHADOW  # FIX (minion typing): Shadow by name
            elif hasattr(ld, 'nation'):
                s = _side_of_nation(ld.nation)
            elif isinstance(ld, Companion):
                s = Side.FREE_PEOPLES
            else:
                continue
            if s == my_side:
                ml += 1
            else:
                el += 1
        if mr + me + ml > 0:
            my_army_dict[rkey] = (mr, me, ml, 0)
        if er + ee + el > 0:
            en_army_dict[rkey] = (er, ee, el, 0)
    feats.extend(_compute_military_threat_features(my_army_dict, en_army_dict, my_side))

    # --- Strategic planning features (12) ---
    feats.extend(_compute_strategic_features(
        army_by_region, state, my_side, enemy_side, vp))

    # --- Reinforcement pools (8 nations × 3: remaining / max) ---
    # FP casualties are permanent; Shadow units cycle back to the pool, so
    # remaining = max - on_board (- permanent dead for FP)
    # ONE ANSWERER (USER ruling 2026-08-18): the numerator was an
    # inline copy of the availability arithmetic.  figures_available
    # returns exactly max(0, cap - used) with the same `used`, so this
    # is value-identical - pinned by leg (2) of
    # streak/test_supply_one_answerer.py, which characterises
    # figures_available against the old formula for every nation and
    # type.  Feature count and order are unchanged.
    _deployed_all = _count_deployed_all(state)
    for nat in ALL_NATIONS:
        maxes = NATION_MAX_FIGURES.get(nat, {})
        for typ in ('regular', 'elite', 'leader'):
            total = maxes.get(typ, 0)
            feats.append(figures_available(state, nat, typ,
                                           _deployed_all)
                         / max(total, 1))

    # --- Immediate threat & availability (7) ---
    # VP each side could plausibly capture THIS turn: enemy settlements
    # where an adjacent army outnumbers the garrison, capped by dice left
    def _vp_capturable(att_dict, def_dict, att_side, dice_left):
        total = 0.0
        n_targets = 0
        att_ctrl = state.enemy_controlled.get(att_side, set())
        for rk in ALL_REGION_KEYS:
            meta = board.regions.get(rk, {})
            if not (meta.get('is_city') or meta.get('is_stronghold')):
                continue
            nat_r = _region_nation(rk)
            if nat_r is None or _side_of_nation(nat_r) == att_side:
                continue
            if rk in att_ctrl:
                continue  # already captured
            d = def_dict.get(rk, (0, 0, 0, 0))
            def_units = d[0] + d[1]
            best_att = 0
            for nb in board.adj.get(rk, ()):
                a = att_dict.get(nb, (0, 0, 0, 0))
                best_att = max(best_att, a[0] + a[1])
            if best_att > def_units and n_targets < dice_left:
                total += 2.0 if meta.get('is_stronghold') else 1.0
                n_targets += 1
        return total / 10.0

    feats.append(_vp_capturable(my_army_dict, en_army_dict, my_side, my_unused))
    feats.append(_vp_capturable(en_army_dict, my_army_dict, enemy_side, en_unused))

    # Which special characters could enter play right now
    def _safe_can(fn):
        try:
            return 1.0 if fn(state)[0] else 0.0
        except Exception:
            return 0.0
    feats.append(_safe_can(game_logic.can_muster_witch_king))
    feats.append(_safe_can(game_logic.can_muster_saruman))
    feats.append(_safe_can(game_logic.can_muster_mouth_of_sauron))
    feats.append(_safe_can(game_logic.can_transform_gandalf_white))
    try:
        feats.append(1.0 if game_logic.can_crown_aragorn(state)[0] else 0.0)
    except Exception:
        # Fallback: Strider alive and outside the fellowship
        feats.append(1.0 if any(
            isinstance(l, Companion) and l.name == 'Strider'
            for a in state.armies.values() for l in a.leaders) else 0.0)

    # --- Guide IDENTITY one-hot (iter 36: which companion guides the
    # fellowship — previously only the guide's LEVEL was encoded; the
    # guide's ability is identity-specific and fully visible) ---
    guide_onehot = [0.0] * len(companion_names)
    g = state.fellowship.guide
    if g is not None:
        gname = sanitize(str(getattr(g, 'name', '')))
        for gi_, cn in enumerate(companion_names):
            if sanitize(cn) == gname:
                guide_onehot[gi_] = 1.0
                break
    feats.extend(guide_onehot)

    # --- PENDING-DECISION block (iter 40, user finding: the T1 H-die
    # second army move was passed — the net had NO feature saying it is
    # mid-double-move with a free move left; same for the muster
    # double-recruit and a deferred card option) ---
    _pam = state.pending_army_move_die is not None
    _pmu = getattr(state, 'pending_muster_die', None) is not None
    _pco = getattr(state, 'pending_card_option', None) is not None
    feats.append(1.0 if _pam else 0.0)
    feats.append(1.0 if _pmu else 0.0)
    feats.append(1.0 if _pco else 0.0)
    pending_from = [0.0] * NUM_REGIONS
    _pfrom = None
    if _pam:
        _pfrom = state.pending_army_move_from
    elif _pmu:
        _pfrom = getattr(state, 'pending_muster_region', None)
    if _pfrom:
        _pi = REGION_INDEX.get(sanitize(str(_pfrom)))
        if _pi is not None:
            pending_from[_pi] = 1.0
    feats.extend(pending_from)

    # --- FULL DECISION CONTEXT (USER DIRECTIVE 2026-07-08: "for every
    # single decision that is made by the net, the entire game state
    # needs to be exposed — the context matters for every decision no
    # matter how small").  The pending-battle state was encoded in ZERO
    # dims: combat cards / casualties / retreats / presses were chosen
    # blind to which battle was being fought.  This block exposes every
    # pending decision's situation. ---
    pc = getattr(state, 'pending_combat', None)
    battle_region = [0.0] * NUM_REGIONS
    # every live pending_combat phase gets its OWN slot (capping five
    # phases into one slot left retreat/siege/casualty/advance decisions
    # indistinguishable to the value head; attacker_rearguard dropped —
    # transition-only, never a live decision)
    _PC_PHASES = ('defender_siege_choice',
                  'attacker_combat_card', 'defender_combat_card',
                  'combat_effect_choice', 'combat_round_result',
                  'defender_retreat', 'attacker_siege_decision',
                  'attacker_casualty_policy', 'defender_casualty_policy',
                  'victor_advance', 'casualty_allocation',
                  'fp_shine')
    combat_ctx = [0.0] * (10 + len(_PC_PHASES) + 5)  # +4 v14 P2, +1 v16
    if pc is not None:
        _bi = REGION_INDEX.get(sanitize(str(pc.get('defender_region', ''))))
        if _bi is not None:
            battle_region[_bi] = 1.0
        combat_ctx[0] = 1.0
        combat_ctx[1] = 1.0 if pc.get('is_siege') else 0.0
        combat_ctx[2] = 1.0 if pc.get('attacker_side') == my_side else 0.0
        _aa = _armies_lookup(state, str(pc.get('attacker_region', '')))
        _da = _armies_lookup(state, str(pc.get('defender_region', '')))
        _asd = pc.get('attacker_side')
        for _off, _army, _sd in ((3, _aa, _asd),
                                 (6, _da, pc.get('defender_side'))):
            if _army is not None and _sd is not None:
                _u = [u for u in _army.units
                      if _side_of_nation(u.nation) == _sd]
                combat_ctx[_off + 0] = sum(
                    1 for u in _u if u.kind == UnitKind.REGULAR) / 10.0
                combat_ctx[_off + 1] = sum(
                    1 for u in _u if u.kind == UnitKind.ELITE) / 10.0
                # Leaders side-filtered like the units: mixed siege
                # entries hold BOTH sides' figures in one Army object.
                _nl = 0
                for _ld in _army.leaders:
                    try:
                        _ls = (Side.FREE_PEOPLES
                               if isinstance(_ld, Companion)
                               else Side.SHADOW
                               if (isinstance(_ld, Nazgul)
                                   or type(_ld).__name__ == 'Minion')
                               else _side_of_nation(_ld.nation))
                    except Exception:
                        continue
                    if _ls == _sd:
                        _nl += 1
                combat_ctx[_off + 2] = _nl / 7.0
        combat_ctx[9] = min(pc.get('rounds_fought', 0), 5) / 5.0
        _ph = pc.get('phase')
        if _ph in _PC_PHASES:
            combat_ctx[10 + _PC_PHASES.index(_ph)] = 1.0
        # v14 (P2 card battles): mandatory rounds remaining, round-1
        # FP combat-card block, battle-scoped dice reductions
        _mro9 = 10 + len(_PC_PHASES)
        combat_ctx[_mro9 + 0] = max(
            0, pc.get('mandatory_rounds', 0)
            - pc.get('rounds_fought', 0)) / 3.0
        combat_ctx[_mro9 + 1] = (1.0 if pc.get(
            'no_def_card_r1_unless_companion') else 0.0)
        combat_ctx[_mro9 + 2] = min(
            pc.get('atk_dice_reduction', 0), 5) / 5.0
        combat_ctx[_mro9 + 3] = min(
            pc.get('def_dice_reduction', 0), 5) / 5.0
        # v16 (Corsairs): the attacker cannot retreat/cease this battle
        combat_ctx[_mro9 + 4] = (1.0 if pc.get('attacker_no_retreat')
                                 else 0.0)
    feats.extend(battle_region)
    feats.extend(combat_ctx)

    # pending hunt: damage + phase
    ph = getattr(state, 'pending_hunt', None)
    hunt_ctx = [0.0] * 6
    if ph is not None:
        hunt_ctx[0] = 1.0
        hunt_ctx[1] = min(ph.get('damage', 0), 6) / 6.0
        _hph = str(ph.get('phase', ''))
        for _k, _nm in enumerate(('hunt_sacrifice', 'hunt_gollum',
                                  'hobbit_place')):
            if _hph == _nm:
                hunt_ctx[2 + _k] = 1.0
        # hobbit_place: the level drives the reachable placement set
        hunt_ctx[5] = min(int(ph.get('level', 0) or 0), 4) / 4.0
    feats.extend(hunt_ctx)

    # pending card option: which card + how many options.  v14 adds
    # the DECIDER bit (cross-side decisions: Lure/Stormcrow), the
    # 'direct' chained-stage flag, and the resume_action bit (an
    # interposed decision - Crebain - stashes the original action)
    pco = getattr(state, 'pending_card_option', None)
    co_ctx = [0.0] * 6
    if pco is not None:
        co_ctx[0] = (_get_card_idx_map().get(pco.get('card_id', ''), -1)
                     + 1) / 97.0
        co_ctx[1] = min(len(pco.get('options') or []), 40) / 40.0
        _dec9 = pco.get('decider', pco.get('side'))
        co_ctx[2] = 1.0 if (_dec9 == 'fp') == (
            my_side == Side.FREE_PEOPLES) else 0.0
        co_ctx[3] = 1.0 if pco.get('direct') else 0.0
        co_ctx[4] = 1.0 if pco.get('resume_action') else 0.0
        # owner side (may differ from decider - cross-side decisions)
        co_ctx[5] = 1.0 if (pco.get('side') == 'fp') == (
            my_side == Side.FREE_PEOPLES) else 0.0
    feats.extend(co_ctx)

    # pending discard / guide change / eye-allocation phase
    _pd = getattr(state, 'pending_discard', None)
    feats.append(1.0 if _pd is not None else 0.0)
    feats.append(1.0 if _pd is not None
                 and _pd.get('side') == my_side else 0.0)
    feats.append(1.0 if getattr(state, 'pending_guide_change', None)
                 is not None else 0.0)
    feats.append(1.0 if getattr(state, '_eye_allocation_phase', False)
                 else 0.0)
    feats.append(1.0 if getattr(state, '_declare_phase', False) else 0.0)
    # card-granted follow-up context (kind + moves left)
    _pcf = getattr(state, 'pending_card_followup', None)
    feats.append(1.0 if _pcf is not None else 0.0)
    feats.append((1.0 if _pcf and _pcf.get('kind') == 'wk_army' else 0.0))
    feats.append((_pcf.get('moves_left', 0) / 2.0) if _pcf else 0.0)
    # special hunt tiles queued for the pool by played cards
    feats.append(min(len(getattr(state, 'pending_hunt_tiles', None)
                         or []), 3) / 3.0)
    # v14 (P1): pending Crebain +1 hunt bonus; unresolved overstack
    # regions (count + total excess own-side units past the caps)
    feats.append(1.0 if getattr(state, '_crebain_hunt_bonus', 0) else 0.0)
    _ovr9 = getattr(state, '_overstack_regions', None) or ()
    feats.append(min(len(_ovr9), 3) / 3.0)
    _exc9 = 0
    if _ovr9:
        from combat import side_of_nation as _son9e
        from event_effects import _stack_cap as _scap9
        for _rr9 in _ovr9:
            _ar9 = _armies_lookup(state, _rr9)
            if _ar9 is None:
                continue
            for _sd9 in (Side.FREE_PEOPLES, Side.SHADOW):
                _mine9 = sum(1 for u in _ar9.units
                             if _son9e(u.nation) == _sd9)
                _exc9 += max(0, _mine9 - _scap9(state, sanitize(_rr9),
                                                _sd9))
    feats.append(min(_exc9, 6) / 6.0)
    # v15 (FP Character round, amendment 1.126): Ents free-chain
    # progress; the Mithril interpose must SHOW the stashed tile -
    # the FP decision is about ITS properties
    feats.append(min(int(getattr(state, '_ents_chain_n', 0) or 0), 4)
                 / 4.0)
    _msv9 = getattr(state, '_mithril_stash', None)
    if _msv9:
        _mt9 = _msv9.get('tile')
        feats.append(1.0)
        feats.append(min(max(int(getattr(_mt9, 'damage', 0) or 0), -2),
                         3) / 3.0)
        feats.append(1.0 if getattr(_mt9, 'is_eye', False) else 0.0)
        feats.append(1.0 if getattr(_mt9, 'reveal', False) else 0.0)
        feats.append(1.0 if getattr(_mt9, 'stops_movement', False)
                     else 0.0)
        feats.append(1.0 if _msv9.get('mordor') else 0.0)
    else:
        feats.extend([0.0] * 6)

    # --- v26 tail (W10 F1/F2, session 35) — new features are APPENDED
    # here, at the very END, so the hardcoded _CARD_IDX_START layout sum
    # and every earlier offset stay valid for ALL arches (base/wide/
    # xformer absorb extra dims in the global token: _global_dim is
    # computed from input_dim, global_post = x[:, _CARD_IDX_END:]). ---
    # F1: Mordor-inactivity flag — drives the +1 end-of-turn corruption
    # when the fellowship is on the Mordor track and did NOT move/hide
    # this turn (game_logic.py:188). Was probe-proven invisible (W10).
    feats.append(1.0 if getattr(state, 'fellowship_acted_in_mordor',
                                False) else 0.0)
    # F2: double-move second-leg context — how many OWN figures at the
    # destination pre-date the first half (only they may continue), and
    # the double-recruit's first type (constrains the second recruit).
    _pre26 = getattr(state, 'pending_army_move_preexist', None) or {}
    feats.append(min(int(_pre26.get('regular', 0) or 0), 10) / 10.0)
    feats.append(min(int(_pre26.get('elite', 0) or 0), 10) / 10.0)
    feats.append(min(int(_pre26.get('leaders', 0) or 0), 7) / 7.0)
    _ft26 = getattr(state, 'pending_muster_first_type', None)
    feats.append(1.0 if _ft26 == 'regular' else 0.0)
    feats.append(1.0 if _ft26 == 'elite' else 0.0)
    feats.append(1.0 if _ft26 == 'leader' else 0.0)
    # F2: second-leg OWNERSHIP — pending_*_side was unencoded; the bare
    # pending flag could not say WHOSE leftover half-move it is.
    _ps26 = (getattr(state, 'pending_army_move_side', None)
             if state.pending_army_move_die is not None
             else (getattr(state, 'pending_muster_side', None)
                   if getattr(state, 'pending_muster_die', None)
                   is not None else None))
    feats.append(1.0 if _ps26 == my_side else 0.0)
    feats.append(1.0 if (_ps26 is not None and _ps26 != my_side)
                 else 0.0)

    return np.array(feats, dtype=np.float32)


def get_pending_context_start() -> int:
    """First state-encoding dim of the PENDING/TRANSIENT context tail
    (battle_region one-hot onward: combat/hunt/card-option/discard/
    guide/eye/declare/followup/tiles/crebain/overstack/ents/mithril).
    Used by the value-head context-dropout regularizer (amendment
    1.130: the value head learned a +0.23 'mid-battle' bonus from
    context correlations the corpus never counterfactuals)."""
    global _PENDING_CTX_START
    if _PENDING_CTX_START is None:
        import board as _b9
        import game_state as _gs9
        import game_logic as _gl9
        _st9 = _gs9.setup_initial_game(_b9.load_board())
        _gl9.initialize_decks(_st9)
        v0 = encode_state(_st9, Side.FREE_PEOPLES)
        _st9.pending_combat = {
            'attacker_region': 'eregion', 'defender_region': 'moria',
            'attacker_side': Side.SHADOW,
            'defender_side': Side.FREE_PEOPLES,
            'phase': 'attacker_combat_card', 'is_siege': False,
            'rounds_fought': 1}
        v1 = encode_state(_st9, Side.FREE_PEOPLES)
        import numpy as _np9
        # The first differing dim is battle_region[REGION_INDEX[probe
        # region]], NOT the block start — subtract the probe region's
        # one-hot offset so the returned boundary is the true block
        # start (the value context-dropout must mask the WHOLE
        # battle_region one-hot, not just regions >= the probe's index).
        _PENDING_CTX_START = (int(_np9.nonzero(v0 != v1)[0][0])
                              - REGION_INDEX['moria'])
    return _PENDING_CTX_START


_PENDING_CTX_START: Optional[int] = None

# Compute feature dimension once
_FEATURE_DIM: Optional[int] = None

def get_feature_dim() -> int:
    """Return the feature vector length (computed once from a dummy state)."""
    global _FEATURE_DIM
    if _FEATURE_DIM is None:
        b = board.load_board()
        dummy = gs.setup_initial_game(b)
        vec = encode_state(dummy, Side.FREE_PEOPLES)
        _FEATURE_DIM = len(vec)
    return _FEATURE_DIM


# ===================================================================
# Step 3: Log Trajectory Extraction
# ===================================================================



def _bfs_distance_to_mordor(state: 'GameState') -> int:
    """Compute shortest-path distance (in regions) from the Fellowship's
    current position to the nearest Mordor entry point (Morannon or Minas
    Morgul) using BFS on the board adjacency graph.

    Returns 0 if already at a Mordor entry or inside Mordor.
    Returns a large number (99) if unreachable (should not happen on a
    connected map).
    """
    if state.fellowship.mordor_entered:
        return 0

    from board import sanitize as _san
    cur = game_logic.fellowship_region(state)
    mordor_gates = {_san("morannon"), _san("minas morgul")}

    if cur in mordor_gates:
        return 0

    # BFS
    from collections import deque
    visited = {cur}
    queue = deque([(cur, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in state.board.adjacent(node):
            if neighbor in mordor_gates:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 99  # unreachable fallback


# Cache the max distance at module level (computed once)
_MAX_MORDOR_DISTANCE: Optional[int] = None

def _get_max_mordor_distance(state: 'GameState') -> int:
    """Get the maximum possible BFS distance to Mordor (for normalization).
    Computed once from the board graph and cached."""
    global _MAX_MORDOR_DISTANCE
    if _MAX_MORDOR_DISTANCE is not None:
        return _MAX_MORDOR_DISTANCE

    from board import sanitize as _san
    from collections import deque
    mordor_gates = {_san("morannon"), _san("minas morgul")}

    # BFS from each mordor gate outward to find furthest region
    max_dist = 0
    for gate in mordor_gates:
        visited = {gate}
        queue = deque([(gate, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist > max_dist:
                max_dist = dist
            for neighbor in state.board.adjacent(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    _MAX_MORDOR_DISTANCE = max(max_dist, 1)
    return _MAX_MORDOR_DISTANCE


def _total_corruption_soak(state: 'GameState') -> float:
    """Compute total corruption soak from companions in the fellowship.

    Each companion absorbs corruption equal to their level when eliminated
    by hunt damage (instead of taking corruption on Frodo). This measures
    the fellowship's remaining 'buffer' against corruption.

    Returns sum of companion levels in the fellowship.
    """
    return sum(c.level for c in state.fellowship.companions)


def state_value_fp(state: 'GameState') -> float:
    """Heuristic value of a game state from FP perspective.

    Returns a float in [-1, +1]:
      positive = favors FP, negative = favors Shadow.

    FP value function (what FP wants to maximize):
      Primary: distance to Mordor (closer = better) and dunking the ring
               (mordor_step progress once inside Mordor).
      Secondary: FP military victory points.

    Shadow value function (what Shadow wants to maximize, shown as negative
    from FP perspective):
      - Military VP (distance to 10)
      - FP distance to Mordor (positive for Shadow = farther is better)
      - Total FSP corruption = frodo_corruption + companion soak used
        (corruption / (12 + total_companion_levels))
    """
    # VP computed live from enemy_controlled (state.vp is display-only and
    # never updated by the engine)
    _vp = game_logic.calculate_victory_points(state)

    # ---- FP Score ----
    # Distance to Mordor: normalize to [0, 1] where 1 = at Mordor gate
    if state.fellowship.mordor_entered:
        # Inside Mordor: progress toward Mount Doom (5 steps)
        # Full distance credit (1.0) + mordor progress bonus
        mordor_proximity = 1.0
        mordor_progress = state.fellowship.mordor_step / 5.0
    else:
        # Outside Mordor: BFS distance to nearest gate
        dist = _bfs_distance_to_mordor(state)
        max_dist = _get_max_mordor_distance(state)
        mordor_proximity = 1.0 - (dist / max_dist)  # 1.0 at gate, 0.0 at farthest
        mordor_progress = 0.0

    fp_vp_progress = min(_vp['free_peoples'], 4) / 4.0

    # FP score: Mordor progress + military defense
    fp_score = (0.35 * mordor_proximity
                + 0.30 * mordor_progress
                + 0.35 * fp_vp_progress)  # Military VP matters for FP too

    # ---- Shadow Score ----
    # Military VP progress (Shadow needs 10)
    sh_vp_progress = min(_vp['shadow'], 10) / 10.0

    # Shadow benefits from fellowship being far from Mordor (rush strategy)
    # This is the inverse of FP's mordor_proximity
    sh_distance_bonus = 1.0 - mordor_proximity  # 1.0 when far, 0.0 at gate
    if state.fellowship.mordor_entered:
        # Shadow gets penalty as fellowship progresses in Mordor
        sh_distance_bonus = -(mordor_progress)

    # Total corruption = frodo corruption + corruption soak consumed
    # Max corruption before ring is lost = 12
    # Companion soak = sum of companion levels (each can sacrifice to absorb)
    companion_soak = _total_corruption_soak(state)
    # Effective corruption progress: how close to losing the ring
    # Total "HP" = 12 (frodo max) but companions extend effective HP
    total_hp = 12.0 + companion_soak
    effective_corruption = min(state.fellowship.corruption, 12) / total_hp
    # Also factor in remaining companions: fewer companions = less soak left
    # Normalize companion_soak to initial max (~8 for full fellowship)
    soak_remaining = companion_soak / 8.0  # 8 = approximate sum of all starting companion levels

    # Hunt pool danger: in Mordor, the composition of the hunt pool matters
    hunt_danger = 0.0
    if state.fellowship.mordor_entered and state.hunt_pool:
        pool = state.hunt_pool
        pool_size = len(pool)
        n_bad = sum(1 for t in pool if t.is_eye
                    or getattr(t, 'stops_movement', False)
                    or (t.damage >= 2 and not t.is_eye))
        n_good = sum(1 for t in pool
                     if getattr(t, 'is_fellowship_special', False))
        hunt_danger = max(0.0, min(1.0, (n_bad - n_good) / max(pool_size, 1)))

    shadow_score = (0.65 * sh_vp_progress          # VP is THE primary Shadow objective
                    + 0.10 * sh_distance_bonus
                    + 0.10 * effective_corruption
                    + 0.10 * (1.0 - soak_remaining)
                    + 0.05 * hunt_danger)

    return float(np.clip(fp_score - shadow_score, -1, 1))





def _is_shadow_stronghold(state, region_key):
    """Check if region is an unconquered shadow stronghold."""
    meta = board.regions.get(region_key, {})
    fp_ctrl = state.enemy_controlled.get(gs.Side.FREE_PEOPLES, set())
    return (meta.get("is_stronghold")
            and game_logic.is_shadow_region(region_key)
            and region_key not in fp_ctrl)


def _get_reveal_placement_actions(state: 'GameState') -> 'List[Action]':
    """Generate valid reveal placement actions after hunt reveals fellowship.

    The FP player chooses where to place the fellowship (within progress
    distance).  Revelation cannot end in an UNCONQUERED FP
    city/stronghold (a Shadow-captured one is legal — mode="reveal" is a
    static-metadata memo, so the conquest exception is applied here).
    This is a learned MCTS decision — not automatic.
    """
    actions = []
    if state.fellowship.mordor_entered:
        return actions

    cur = game_logic.fellowship_region(state)
    progress = state.fellowship.progress

    if progress <= 0:
        # No progress — must stay in place (auto-handled, no actions needed)
        return actions

    reachable = game_logic.get_reachable_regions(
        state, cur, progress, mode="reveal")
    # Conquest exception (mirrors _auto_reveal_place._legal_places): FP
    # settlements CAPTURED by Shadow are legal reveal destinations —
    # the static memo excluded every FP settlement.
    _sh_ctrl9 = state.enemy_controlled.get(Side.SHADOW, set())
    if _sh_ctrl9:
        _all9 = game_logic.get_reachable_regions(
            state, cur, progress, mode="normal")
        for _r9 in (_all9 - reachable):
            if _r9 in _sh_ctrl9:
                reachable.add(_r9)

    for dest in reachable:
        # USER RULING 2026-07-20b (w5_g2 T9): the pawn never
        # region-steps into Mordor's interior — reached only via the
        # track after a turn-start gateway declaration (1.155).
        if sanitize(dest) in game_logic._MORDOR_INTERIOR:
            continue
        actions.append(Action(
            die_idx=-1,
            action_type='fellowship_reveal_place',
            params={'region': dest}
        ))

    # If nowhere valid (all reachable are FP settlements), stay in place
    if not actions:
        actions.append(Action(
            die_idx=-1,
            action_type='fellowship_reveal_place',
            params={'region': cur}
        ))

    return actions


def _apply_reveal_placement(state: 'GameState', dest: str):
    """Execute a reveal placement to *dest* — the ONE implementation
    shared by the fellowship_reveal_place executor (teacher + learned
    picks) and open_reveal_placement's forced single-option case.
    Leaving/entering Shadow-stronghold draws, pawn move, progress 0."""
    cur = game_logic.fellowship_region(state)
    if getattr(state, '_sh_tile_drawn_regions', None) is None:
        state._sh_tile_drawn_regions = set()

    # Shadow stronghold: "leaving" tile
    if cur != dest and _is_shadow_stronghold(state, cur):
        game_logic._draw_shadow_stronghold_tile(state, cur, "leaving")

    # Shadow stronghold: "entering" tile at destination
    if cur != dest and _is_shadow_stronghold(state, dest):
        game_logic._draw_shadow_stronghold_tile(state, dest, "entering")

    # Move fellowship to chosen destination
    try:
        state.fellowship.region = state.board.region(dest)
    except Exception:
        pass
    state.fellowship.progress = 0
    state._reveal_place_due = False
    # No immediate Mordor entry on a gateway reveal: the pawn goes
    # onto the track only via the turn-start declaration (1.155).


def open_reveal_placement(state: 'GameState') -> bool:
    """Settle-point for a due hunt-reveal placement (USER RULING
    2026-08-16, closing the charter's OPEN_reveal_placement row:
    reveal placement is a LEARNED FP DECISION — net in self-play,
    human in sided live play).

    Returns True when the placement decision is OPEN (menu =
    _get_reveal_placement_actions, FP to act via
    _pending_decision_side); False when nothing was due or the
    forced/no-choice case resolved here.

    Forced cases keep the certified auto path byte-for-byte
    (_auto_reveal_place): revealed at a Mordor gateway (pawn stays,
    USER 2026-07-20b), progress<=0, or in Mordor.  A single-option
    menu applies that option directly (no decision to learn).

    Housekeeping: the 'revealed in' Shadow-stronghold draw precedes
    the choice (human log order, 1.87 one draw per region per event —
    idempotent via _sh_tile_drawn_regions, armed when the reveal set
    the flag).  1.132 ordering holds: the reveal tile's own damage
    settled before this runs, and a prep-draw's damage resolves as
    its OWN decision before the menu opens (the flag stays due)."""
    if not getattr(state, '_reveal_place_due', False):
        return False
    if state.pending_hunt is not None \
            or getattr(state, '_pending_hunt_damage', 0):
        return False          # damage first (1.132); still due
    cur = game_logic.fellowship_region(state)
    if (state.fellowship.mordor_entered
            or state.fellowship.progress <= 0
            or sanitize(cur) in (sanitize('morannon'),
                                 sanitize('minas morgul'))):
        # forced: the certified auto path (it re-arms its own event
        # set; the prep draw below never ran for these states —
        # classification is stable across re-entry, so no double draw)
        state._reveal_place_due = False
        try:
            game_logic._auto_reveal_place(state)
        except Exception:
            pass
        if getattr(state, '_pending_hunt_damage', 0) > 0:
            if not game_logic._offer_hunt_reduction(state):
                game_logic._settle_hunt_damage(state)
        return False
    # prep: revealed IN a Shadow stronghold draws before the choice
    if _is_shadow_stronghold(state, cur):
        game_logic._draw_shadow_stronghold_tile(state, cur,
                                                "revealed in")
        if getattr(state, '_pending_hunt_damage', 0) > 0:
            if not game_logic._offer_hunt_reduction(state):
                game_logic._settle_hunt_damage(state)
        if state.pending_hunt is not None \
                or getattr(state, '_pending_hunt_damage', 0):
            return False      # that damage decides first; still due
    legal = _get_reveal_placement_actions(state)
    if len(legal) <= 1:
        state._reveal_place_due = False
        if legal:
            _apply_reveal_placement(state,
                                    legal[0].params['region'])
            if getattr(state, '_pending_hunt_damage', 0) > 0:
                if not game_logic._offer_hunt_reduction(state):
                    game_logic._settle_hunt_damage(state)
        return False
    return True               # decision OPEN: FP places the pawn


def _build_card_name_lookup() -> Dict[str, str]:
    """Build card_name -> card_id lookup from cards.py registry."""
    from cards import get_cards
    return {card.card_name: card_id for card_id, card in get_cards().items()}


def _sync_hands_to_card_ids(state: 'GameState', lookup: Dict[str, str]):
    """Convert any card dicts in hands to card_id strings using the lookup."""
    for side in (Side.FREE_PEOPLES, Side.SHADOW):
        hand = state.hands[side].hand
        for i, item in enumerate(hand):
            if isinstance(item, str) and item.startswith('_unknown_'):
                # Retry placeholders with the normalized (accent/case
                # tolerant) lookup
                cid = _lookup_card_id(lookup, item[len('_unknown_'):])
                if cid:
                    hand[i] = cid
                continue
            if isinstance(item, dict):
                card_name = item.get('card_name', '')
                cid = _lookup_card_id(lookup, card_name) if card_name else None
                if cid:
                    hand[i] = cid
                elif card_name and card_name != 'Unknown':
                    # Known name but not in lookup — keep as placeholder
                    hand[i] = f"_unknown_{card_name}"
                else:
                    # Truly unknown card — keep as placeholder for hand size
                    hand[i] = "_unknown_card"


# ===================================================================
# Behavioral cloning: match logged human decisions to engine actions
# ===================================================================

BC_NUM_ALTERNATIVES = 15            # sampled legal alternatives per decision
BC_NUM_SLOTS = BC_NUM_ALTERNATIVES + 1  # slot 0 = the human-chosen action

# Logged action types the matcher supports (others are not BC attempts)
_BC_SUPPORTED_LOG_TYPES = {
    'army_move', 'combat_start', 'combat_retreat', 'muster', 'political',
    'fellowship_move', 'mordor_move', 'fellowship_hide', 'fellowship_declare',
    'draw_card', 'play_event', 'companion_separate', 'pass',
    'army_attack', 'nazgul_move',
    # Combat-card decisions (matched against a synthesized candidate set —
    # the replayed state has no pending_combat, see _synthetic_decision_actions)
    'combat_card', 'combat_card_none',
    # Combat-card effect choices (Onslaught sacrifice count, Dread and
    # Despair forfeit, ...) — also matched against a synthesized set
    'combat_effect_choice',
    # Hunt-casualty hobbit placement (companion_move rows tagged
    # hunt_placement by the parser)
    'companion_move',
    # Elven-ring use is a learned free-action decision (user ruling
    # 2026-07-03); 'collect'-tagged records (Shadow picking up a used FP
    # ring) are excluded in _bc_is_attemptable
    'elven_ring',
    # Bid-compensation action-token use (free action, die_idx=-1)
    'use_token',
    # Strider -> Aragorn coronation (piece-241 flip in the logs;
    # engine action existed but the record type was never emitted)
    'crown_aragorn',
    # Dwarven-ring reroll (bid compensation; piece-428 drag out of the
    # dice pool, outcome folded in as new_face by the parser)
    'dwarven_ring',
}


def _bc_is_attemptable(act_type: str, params: dict, act_side=None) -> bool:
    """Is this logged action a matchable die-action decision?"""
    p0 = params or {}
    # Card effects, forced auto-advances, and multi-op continuations are
    # not standalone decisions
    if p0.get('card_effect') or p0.get('auto') or p0.get('continuation'):
        return False
    if act_type not in _BC_SUPPORTED_LOG_TYPES:
        return False
    if act_type == 'elven_ring':
        # Shadow picking a used FP ring up off the board is bookkeeping,
        # not a decision; only ring USES (die-face changes) are learnable
        return not p0.get('collect')
    if act_type == 'companion_move':
        # Hunt-casualty hobbit placements and DIED companion walks are
        # decisions; die-less walks are continuations of the same die
        # (rulebook p.36: one Character die moves ALL on-map companion
        # groups) — the closed loop keeps those as continuation entries
        # and force-executes them (2026-01-05 kapcerz T1: Legolas ->
        # Lorien on the die, Strider/Boromir -> Eastemnet die-less).
        return bool(p0.get('hunt_placement') or p0.get('die_used'))
    if act_type == 'muster':
        p = params or {}
        # Minion summons (Saruman / Witch-king / Mouth) ARE muster-die
        # decisions — and strategically critical ones (+1 action die)
        if p.get('minions'):
            return True
        units = p.get('units') or {}
        # Zero-unit "musters" are card-driven placements
        if not any(units.get(k, 0) for k in ('regular', 'elite', 'leader', 'nazgul')):
            return False
        # Placements in the OTHER side's settlements (e.g. Corsairs of
        # Umbar putting Shadow units in Gondor ports) are card effects,
        # never muster-die decisions.
        if act_side is not None:
            nat = _region_nation(sanitize(str(p.get('region', ''))))
            if nat is not None and _side_of_nation(nat) != act_side:
                return False
    return True


def _bc_params_incomplete(act_type: str, params: dict) -> bool:
    """Synthesized attacks are logged BEFORE the battle region is known
    ('from'/'to' = '?'); the replayer patches the shared params dict in
    place when a later log line reveals the region. Matching must be
    deferred until after the full replay for those actions."""
    p = params or {}
    if act_type == 'army_attack':
        return str(p.get('from', '')) == '?' or str(p.get('to', '')) == '?'
    if act_type == 'political':
        # A die-less political record may be the defender advance of an
        # attack whose callout lands LATER in the log — the replayer
        # retro-tags it 'auto' at attack synthesis. Defer so the retry
        # sees the patched flag (re-check attemptability at retry).
        return p.get('die_used') is None and not p.get('auto') \
            and not p.get('card_effect')
    return False


# Extraction-only die-face fallback: the replayer's die->action attribution
# is heuristic (fuzzy consumption windows, unlogged ring changes), so a
# demonstrably-taken army/character action may find no die of the right
# face in the replayed state. Granting one synthetic die of the required
# face keeps the human decision learnable without weakening engine rules.
_BC_FALLBACK_FACE = {
    'army_attack': ActionDieFace.ARMY,
    'combat_start': ActionDieFace.ARMY,
    'nazgul_move': ActionDieFace.CHARACTER,
}


_NORM_CARD_LOOKUP = None


def _normalize_card_name(name: str) -> str:
    import unicodedata
    # Fold accents (Nazgûl -> Nazgul) then keep alphanumerics only
    folded = unicodedata.normalize('NFKD', str(name))
    folded = ''.join(ch for ch in folded if not unicodedata.combining(ch))
    return ''.join(ch for ch in folded.lower() if ch.isalnum())


def _lookup_card_id(card_lookup: 'Dict[str, str]', name: str) -> 'Optional[str]':
    """Card name -> card_id, tolerant of accents/punctuation differences."""
    cid = card_lookup.get(name)
    if cid:
        return cid
    global _NORM_CARD_LOOKUP
    if _NORM_CARD_LOOKUP is None:
        _NORM_CARD_LOOKUP = {_normalize_card_name(k): v
                             for k, v in card_lookup.items()}
    norm = _normalize_card_name(name)
    # Known log-vs-card spelling variants (VASSAL piece titles vs the
    # official card list the DB uses)
    norm = {
        'celebornsgaladrim': 'celebornsgaladhrim',
        'musteringofalongplannedwar': 'musteringsoflongplannedwar',
        'abalrogiscome': 'balrogofmoria',
        # t138 (engine-check family 4): deck-table short label vs the
        # registry's full title — 16/16 Grimbeorn plays undecodable
        'grimbeorntheold': 'grimbeorntheoldsonofbeorn',
    }.get(norm, norm)
    hit = _NORM_CARD_LOOKUP.get(norm)
    if hit:
        return hit
    # Logs sometimes prefix 'The ' where the card name lacks it (or vice
    # versa): 'The Nazgul Strike!' vs card 'Nazgûl Strike!'
    if norm.startswith('the'):
        hit = _NORM_CARD_LOOKUP.get(norm[3:])
        if hit:
            return hit
    return _NORM_CARD_LOOKUP.get('the' + norm)


def _bc_param_str(v) -> str:
    """Normalize an action param value for matching (enums, region names)."""
    if v is None:
        return ''
    v = getattr(v, 'value', v)  # Nation enum etc. -> string value
    return sanitize(str(v))




def _synthetic_decision_actions(state, act_side, act_type, params,
                                card_lookup):
    """Legal-action set for logged decisions whose engine pending phase is
    not modeled at the replayed state.

    - 'combat_card' / 'combat_card_none': one combat_card_play per distinct
      combat card in the side's hand + a decline action (the engine's
      attacker/defender_combat_card phases offer exactly this shape).
    - 'companion_move' tagged hunt_placement: the hobbit-placement action
      set (companion_separate semantics) anchored at the current fellowship
      position, separation movement = progress + hobbit level.

    Returns a list of Actions, or None when act_type is not synthetic.
    """
    p = params or {}
    if act_type in ('combat_card', 'combat_card_none'):
        from cards import get_card
        tag = 'fp' if act_side == Side.FREE_PEOPLES else 'sh'
        actions = []
        seen = set()
        hand = state.hands.get(act_side)
        for cid in (hand.hand if hand else []):
            if not isinstance(cid, str) or cid in seen:
                continue
            seen.add(cid)
            card = get_card(cid)
            # Every card has a combat side (dual-purpose cards);
            # combat_value is only a strength hint, 0 is still playable
            # (e.g. Swarm of Bats)
            if card is None or not (card.combat_name or '').strip():
                continue
            actions.append(Action(die_idx=-1, action_type='combat_card_play',
                                  params={'card_id': cid, 'side': tag}))
        actions.append(Action(die_idx=-1, action_type='combat_card_none',
                              params={'side': tag}))
        return actions
    if act_type == 'companion_move' and p.get('hunt_placement'):
        comps = p.get('companions') or []
        comp = str(comps[0]) if comps else str(p.get('companion', ''))
        level = 1  # hunt-placed hobbits (Meriadoc / Peregrin) are level 1
        try:
            cur = sanitize(game_logic.fellowship_region(state))
        except Exception:
            return []
        # GENERATOR/VALIDATOR SPLIT (USER 2026-08-17 + EDGE-3): this
        # was the 7th site with its own rule copy (no terminus
        # filter - a hobbit could be placed to END on a Shadow
        # army).  It now only PROPOSES; is_legal_companion_action is
        # the sole gate.
        movement = state.fellowship.progress + level
        reachable = _companion_reachable(state, cur, movement) \
            if movement > 0 else set()
        reachable.add(cur)
        _cch9 = {}
        out9 = []
        for dest in sorted(reachable):
            a9 = Action(die_idx=-1,
                        action_type='companion_separate',
                        params={'companion': comp,
                                'to_region': dest})
            if is_legal_companion_action(
                    state, a9, _cch9, hunt_place_level=level)[0]:
                out9.append(a9)
        return out9
    if act_type == 'combat_effect_choice':
        cid = p.get('card_id') or _lookup_card_id(
            card_lookup, p.get('card_name', '') or '')
        counts = _combat_effect_choice_counts(
            state, cid, p.get('region', ''), act_side)
        return [Action(die_idx=-1, action_type='combat_effect_choice',
                       params={'card_id': cid or '', 'count': int(k)})
                for k in counts]
    if act_type == 'combat_retreat':
        # Replayed states carry no pending_combat — synthesize the
        # retreat candidate set from the rules: any adjacent region free
        # of enemy units and not an enemy-controlled settlement
        # (2023-12-06 T7: the Iron Hills defender retreats to Erebor).
        frm = sanitize(str(p.get('from', '')))
        if not frm:
            return []
        actions = []
        for nb in state.board.adjacent(frm):
            nb_army = None
            for k, a in state.armies.items():
                if sanitize(k) == sanitize(nb):
                    nb_army = a
                    break
            if nb_army and nb_army.units and any(
                    _side_of_nation(u.nation) != act_side
                    for u in nb_army.units):
                continue  # enemy units present
            other = Side.SHADOW if act_side == Side.FREE_PEOPLES \
                else Side.FREE_PEOPLES
            # CENSUS 2026-08-17 (EDGE-3's shape, retreat family): this
            # synthetic branch still carried the PRE-2026-07-15 raw
            # captured-set test; _settlement_controller is the ONE
            # rule body (a unit may not retreat into a region holding
            # an enemy-CONTROLLED settlement, native included, p.30)
            # - same answerer as _get_retreat_actions.
            if _settlement_controller(state, sanitize(nb)) == other:
                continue
            actions.append(Action(
                die_idx=-1, action_type='combat_retreat',
                params={'from': frm, 'to': nb}))
        return actions
    return None


def _combat_effect_choice_counts(state, card_id, region, act_side):
    """Legal choice magnitudes for a combat-card effect at a replayed
    state (no pending_combat modeled): mirrors the engine's pending-phase
    enumeration per effect kind."""
    from combat import (combat_effect_kind, _army_entry, _side_units)
    from game_state import Nazgul
    kind = None
    try:
        from cards import get_card
        card = get_card(card_id) if card_id else None
        kind = combat_effect_kind(getattr(card, 'combat_name', '') or '')
    except Exception:
        pass
    _, army = _army_entry(state, str(region or ''))
    own_units = _side_units(army, act_side) if army is not None else []
    if kind == 'onslaught':
        # t138: at a REPLAYED state the sacrifice cost may already be
        # applied (kill line precedes the choice record), so allow up
        # to the full sacrifice range rather than len-1.
        # 1.211 cluster-B alignment: the count is HITS and an Elite
        # absorbs 2, so enumerate to hit capacity (2E+R, max 4) --
        # a logged 4-hit sacrifice on a 2-elite army must match.
        from game_state import UnitKind as _UK9
        _cap9 = sum(2 if u.kind == _UK9.ELITE else 1 for u in own_units)
        mx = max(0, min(4, _cap9))
        return list(range(0, mx + 1)) if mx > 0 else [0]
    if kind == 'relentless_assault':
        mx = max(0, min(2, len(own_units) - 1))
        return list(range(0, mx + 1)) if mx > 0 else [0]
    if kind == 'dread_despair':
        # t138 (engine-check family 3): humans forfeit Nazgul
        # leadership INCLUDING the Witch-king's 2 (4 corpus samples),
        # and in a FIELD battle the Nazgul ride the ATTACKER's army
        # adjacent to the battle region (sieges co-locate, fields do
        # not).  Enumerate the max over the battle-region army and
        # same-side armies adjacent to it.  (The live engine pending
        # phase now also counts the Witch-king's leadership — USER ruling
        # 2026-07-18, amendment 1.173, combat.setup_combat.nazgul_in_battle.)
        def _dd_points(a):
            if a is None:
                return 0
            pts = sum(1 for l in a.leaders if isinstance(l, Nazgul))
            for l in a.leaders:
                if getattr(l, 'name', '') in ('Witch King',
                                              'Witch-king'):
                    pts += getattr(l, 'leadership', 2)
            return pts
        n = _dd_points(army)
        try:
            _reg_san = sanitize(str(region or ''))
            _b = board.load_board()
            for _adj in _b.adjacent(_reg_san):
                for _k2, _a2 in state.armies.items():
                    if sanitize(_k2) != sanitize(str(_adj)):
                        continue
                    if _side_units(_a2, act_side):
                        n = max(n, _dd_points(_a2))
        except Exception:
            pass
        return list(range(1, n + 1)) if n >= 1 else [1]
    if kind == 'heroic_death':
        counts = {0}
        if army is not None:
            if any(not isinstance(l, (Companion, Nazgul))
                   for l in army.leaders):
                counts.add(1)
            for l in army.leaders:
                if isinstance(l, Companion) and l.level >= 1:
                    counts.add(min(3, l.level))
        for c in state.fellowship.companions:
            if sanitize(str(region or '')) == sanitize(
                    game_logic.fellowship_region(state)) and c.level >= 1:
                counts.add(min(3, c.level))
        return sorted(counts)
    # Unknown effect: offer decline + the generic small magnitudes
    return [0, 1, 2, 3, 4]


_HOPS_CACHE = {}
_HOPS_BOARD = None


def _region_hops(src, dst):
    """BFS hops between two regions on the static board.  Label
    matching only (no state available); None if either key unknown."""
    global _HOPS_BOARD
    s0, d0 = sanitize(str(src)), sanitize(str(dst))
    if s0 == d0:
        return 0
    dmap = _HOPS_CACHE.get(d0)
    if dmap is None:
        if _HOPS_BOARD is None:
            _HOPS_BOARD = board.load_board()
        from collections import deque
        dmap = {d0: 0}
        dq = deque([d0])
        while dq:
            cur = dq.popleft()
            try:
                nbs = _HOPS_BOARD.adjacent(cur)
            except Exception:
                continue
            for nb in nbs:
                nk = sanitize(nb)
                if nk not in dmap:
                    dmap[nk] = dmap[cur] + 1
                    dq.append(nk)
        _HOPS_CACHE[d0] = dmap
    return dmap.get(s0)


def _match_logged_action(actions: 'List[Action]', act_type: str, params: dict,
                         card_lookup: 'Dict[str, str]') -> int:
    """Find the engine legal action that corresponds to a logged human decision.

    The VASSAL logs record decisions in their own vocabulary; this maps them
    onto the engine's legal-action list at the (replayed) pre-action state.
    Returns an index into `actions`, or -1 when no match exists (replay
    drift, unsupported decision type, engine restriction).
    """
    p = params or {}
    s = _bc_param_str

    if act_type == 'army_move':
        f, t = s(p.get('from')), s(p.get('to'))
        moved = p.get('moved') or p.get('units') or {}
        leaders_only = (moved.get('regular', 0) == 0
                        and moved.get('elite', 0) == 0
                        and (moved.get('nazgul', 0) > 0
                             or moved.get('leader', 0) > 0))
        ok_types = ('nazgul_move', 'companion_move', 'army_move') \
            if leaders_only else ('army_move',)
        logged_n = moved.get('regular', 0) + moved.get('elite', 0)
        # Nazgul-aware tie-break (amendment 1.132, same defect class as
        # the 1.131 nazgul_move count fix): humans carry the nazgul on
        # 90% of moves out of nazgul-hosting regions (21-game census
        # 299/333), but (r,e)-tied variants always resolved to the
        # first-generated carry-all - the 10% leave-behinds trained as
        # carries.  Only applies when the menu actually offers nazgul
        # variants (Shadow moves from nazgul-hosting sources).
        want_nz = int(moved.get('nazgul', 0) or 0)
        _has_nz_vars = any(a.action_type == 'army_move'
                           and a.params.get('n_nazgul') is not None
                           for a in actions)

        def _nz_pen(a):
            if not _has_nz_vars or a.action_type != 'army_move':
                return 0.0
            nz = a.params.get('n_nazgul')
            if nz is not None:
                return 0.0 if int(nz) == want_nz else 0.5
            # legacy variant: take_leaders=True carries everything
            if want_nz > 0:
                return 0.0 if a.params.get('take_leaders', True) else 0.5
            return 0.5 if a.params.get('take_leaders', True) else 0.0
        best_i, best_diff = -1, 10 ** 9
        for i, a in enumerate(actions):
            if (a.action_type in ok_types
                    and s(a.params.get('from')) == f
                    and s(a.params.get('to')) == t):
                # Among split options, pick the closest to the human's split
                if a.action_type == 'army_move' and a.params.get('n_regular') is not None:
                    # 126b (2026-07-25 Vayze T8 Dale): PER-KIND
                    # distance — the total-count diff let a 2R+1E
                    # variant tie a logged 3R+0E move and win on
                    # enumeration order, walking an Elves elite out of
                    # Dale the humans left behind.
                    a_r9 = int(a.params.get('n_regular') or 0)
                    a_e9 = int(a.params.get('n_elite') or 0)
                    diff = ((abs(a_r9 - int(moved.get('regular', 0)
                                            or 0))
                             + abs(a_e9 - int(moved.get('elite', 0)
                                              or 0)))
                            if logged_n else 0) + _nz_pen(a)
                else:
                    diff = _nz_pen(a)
                if diff < best_diff:
                    best_i, best_diff = i, diff
        if best_i < 0 and not leaders_only:
            # The parser recorded a MOVE but the destination holds an
            # enemy rearguard (vacate-convention leftovers, 2025-05-16
            # Adobim T5: 1 Rohan regular left at Fords of Isen): the
            # drag IS the attack declaration.
            for i, a in enumerate(actions):
                if a.action_type == 'army_attack' \
                        and s(a.params.get('to')) == t \
                        and (not f or f == '?'
                             or s(a.params.get('from')) == f):
                    return i
        return best_i
    elif act_type == 'combat_start':
        t = s(p.get('region'))
        for i, a in enumerate(actions):
            if a.action_type == 'army_attack' and s(a.params.get('to')) == t:
                return i
    elif act_type == 'army_attack':
        f, t = s(p.get('from')), s(p.get('to'))
        for i, a in enumerate(actions):
            if (a.action_type == 'army_attack'
                    and s(a.params.get('to')) == t
                    and (not f or f == '?'
                         or s(a.params.get('from')) == f)):
                return i
    elif act_type == 'nazgul_move':
        f, t = s(p.get('from')), s(p.get('to'))
        # Count-aware (amendment 1.131): the menu lists the full-group
        # move first and COUNT subsets after; matching by (from,to)
        # alone always returned the full variant, so a human moving 1
        # of 3 nazgul trained as "move all 3" (and the 1.128 count
        # variants could never be slot-0).  The log's moved-count picks
        # the exact subset; k == source total falls through to the
        # full variant (subsets only enumerate 1..n-1).
        k = int((p.get('moved') or {}).get('nazgul', 0) or 0)
        best = -1
        for i, a in enumerate(actions):
            if (a.action_type == 'nazgul_move'
                    and s(a.params.get('from')) == f
                    and s(a.params.get('to')) == t):
                if k > 0 and a.params.get('count') == k:
                    return i
                if best < 0 and a.params.get('count') is None \
                        and not a.params.get('include_wk'):
                    best = i
        if best >= 0:
            return best
        # Relaxed: same destination (engine moves ALL nazgul in a region)
        for i, a in enumerate(actions):
            if a.action_type == 'nazgul_move' and s(a.params.get('to')) == t:
                return i
    elif act_type == 'combat_retreat':
        f, t = s(p.get('from')), s(p.get('to'))
        for i, a in enumerate(actions):
            if (a.action_type in ('combat_retreat', 'army_move')
                    and s(a.params.get('from')) == f
                    and s(a.params.get('to')) == t):
                return i
    elif act_type == 'muster':
        region = s(p.get('region'))
        # Minion summons match the engine's muster_minion actions
        minions = p.get('minions') or []
        if minions:
            want = _normalize_card_name(str(minions[0]))
            best = -1
            for i, a in enumerate(actions):
                if a.action_type != 'muster_minion':
                    continue
                if _normalize_card_name(str(a.params.get('minion', ''))) == want:
                    # Prefer the logged muster REGION (minions can enter at
                    # several valid regions; matching by name alone put the
                    # Mouth of Sauron at the wrong stronghold)
                    if region and s(a.params.get('region')) == region:
                        return i
                    if best < 0:
                        best = i
            return best
        units = p.get('units') or {}
        if not any(units.get(k, 0) for k in ('regular', 'elite', 'leader', 'nazgul')):
            return -1  # Card-driven placement, not a muster-die decision
        if units.get('elite', 0) > 0:
            prefs = [('muster_elite', None), ('muster_recruit', 'elite')]
            if 'upgrade' in p and not p.get('upgrade'):
                # swap-encoded PLAIN recruitment (no regular removed):
                # placing the elite must not replace a regular
                prefs = [('muster_recruit', 'elite'),
                         ('muster_elite', None)]
        elif units.get('leader', 0) > 0 or units.get('nazgul', 0) > 0:
            prefs = [('muster_recruit', 'leader'), ('muster_recruit', 'nazgul')]
        else:
            prefs = [('muster_recruit', 'regular')]
        for atype, typ in prefs:
            for i, a in enumerate(actions):
                if a.action_type != atype:
                    continue
                if s(a.params.get('region')) != region:
                    continue
                if typ is not None and s(a.params.get('type')) != typ:
                    continue
                return i
        # Fallback: any recruit action at that settlement
        for i, a in enumerate(actions):
            if (a.action_type in ('muster_recruit', 'muster_elite')
                    and s(a.params.get('region')) == region):
                # 126a (2026-07-25 demanray T10 Erebor): kind-CLASS
                # guard — a UNIT record must never fall back onto a
                # leader/nazgul recruit (a '1 Dwarves regular' muster
                # matched type=leader when the regular pool ran dry;
                # the phantom leader + missing regular poisoned the
                # Erebor comps for the rest of the game).  Let it
                # fail instead — the teacher's force-place salvage
                # applies the exact logged pieces.
                _atype9 = s(a.params.get('type'))
                if (units.get('regular', 0) or units.get('elite', 0)) \
                        and _atype9 in ('leader', 'nazgul'):
                    continue
                if (units.get('leader', 0) or units.get('nazgul', 0)) \
                        and not (units.get('regular', 0)
                                 or units.get('elite', 0)) \
                        and _atype9 in ('regular', 'elite'):
                    continue
                return i
    elif act_type == 'political':
        nat = s(p.get('nation'))
        for i, a in enumerate(actions):
            if (a.action_type == 'advance_political'
                    and s(a.params.get('nation')) == nat):
                return i
    elif act_type in ('fellowship_move', 'mordor_move'):
        for i, a in enumerate(actions):
            if a.action_type == 'fellowship_move':
                return i
    elif act_type == 'fellowship_hide':
        for i, a in enumerate(actions):
            if a.action_type == 'fellowship_hide':
                return i
    elif act_type == 'fellowship_declare':
        # 2026-08-02 (declare-region enrichment): 'region' is now the
        # board-resolved canonical key at every parser dialect —
        # prefer it over the raw client name in 'to' (article dialect:
        # 'The Grey Havens' vs board key 'grey havens')
        t = s(p.get('region')) or s(p.get('to'))
        for i, a in enumerate(actions):
            if (a.action_type == 'fellowship_declare'
                    and s(a.params.get('region')) == t):
                return i
        if p.get('mordor_entry'):
            # Implied entry declare: the parser guessed the gateway from
            # the REPLAYER's pawn position, which can differ from the
            # teacher's — ANY reachable gateway enters Mordor
            # (2023-02-03 Wirbelwind T4: record says minas morgul, the
            # teacher's pawn reaches morannon)
            for i, a in enumerate(actions):
                if a.action_type == 'fellowship_declare' and \
                        s(a.params.get('region')) in ('morannon',
                                                      'minas morgul'):
                    return i
    elif act_type == 'draw_card':
        deck = 'character' if 'character' in str(p.get('deck_type', '')).lower() \
            else 'strategy'
        for i, a in enumerate(actions):
            if a.action_type == 'draw_card' and s(a.params.get('deck')) == deck:
                return i
    elif act_type == 'play_event':
        card_id = _lookup_card_id(card_lookup, p.get('card_name', ''))
        if card_id:
            for i, a in enumerate(actions):
                if (a.action_type == 'play_event'
                        and a.params.get('card_id') == card_id):
                    return i
    elif act_type == 'companion_separate':
        comp = p.get('companion', '')
        t = s(p.get('to_region'))
        # 1.147: drag-collapsed records put the FINAL destination in
        # to_region (separate + walks in one drag) - out of separation
        # range.  Fall back to the in-range variant NEAREST the logged
        # destination (the human's actual waypoint), never the
        # alphabetical first ('-> angmar' label noise).
        best = -1
        best_d = None
        for i, a in enumerate(actions):
            if (a.action_type == 'companion_separate'
                    and a.params.get('companion') == comp):
                cand = s(a.params.get('to_region'))
                if cand == t:
                    return i
                d = _region_hops(cand, t)
                if best < 0 or (d is not None
                                and (best_d is None or d < best_d)):
                    best, best_d = i, d
        return best
    elif act_type == 'pass':
        # Passing (when holding fewer dice) is a real, important FP decision.
        # A logged pass WITH a die is a SKIP (discard a die without effect —
        # rulebook: an action die may be used to do nothing); match the skip
        # action for that face (2024-12-18 gus833 T1).
        du = p.get('die_used')
        if du:
            want_face = None
            if isinstance(du, (list, tuple)) and len(du) > 1:
                want_face = str(du[1]).upper().replace('/', '_').replace(
                    ' ', '_')
            best_any_skip = -1
            for i, a in enumerate(actions):
                if a.action_type != 'skip':
                    continue
                if best_any_skip < 0:
                    best_any_skip = i
                if want_face and str(a.params.get('face', '')).upper() == \
                        want_face:
                    return i
            if best_any_skip >= 0:
                return best_any_skip
        for i, a in enumerate(actions):
            if a.action_type == 'pass':
                return i
        # neither a skip nor a plain pass in the menu: no match (a
        # die-bearing pass prefers a skip, else the plain pass above)
    elif act_type == 'combat_card':
        # Logged combat-card commitment -> the play action for that card
        cid = _lookup_card_id(card_lookup, p.get('card_name', '') or '')
        if cid:
            for i, a in enumerate(actions):
                if (a.action_type == 'combat_card_play'
                        and a.params.get('card_id') == cid):
                    return i
    elif act_type == 'combat_card_none':
        # 'nc' chat declaration -> the decline action
        for i, a in enumerate(actions):
            if a.action_type == 'combat_card_none':
                return i
    elif act_type == 'crown_aragorn':
        # Piece-241 flip -> the coronation action (engine validates the
        # crown region itself; there is at most one such action)
        for i, a in enumerate(actions):
            if a.action_type == 'crown_aragorn':
                return i
    elif act_type == 'dwarven_ring':
        # Piece-428 drag out of the dice pool -> the reroll action for
        # the die the log spent just before (die_used face = old_face)
        du = p.get('die_used')
        want = None
        if isinstance(du, (list, tuple)) and len(du) > 1:
            want = getattr(ActionDieFace, str(du[1]), None)
        if want is None and p.get('old_face'):
            # 2026-08-02 corpus sweep: 5/157 records carry old_face/
            # new_face params but no die_used — the matcher never read
            # them, silently falling back to the first menu variant
            try:
                want = ActionDieFace(p['old_face'])
            except ValueError:
                want = None
        best = -1
        for i, a in enumerate(actions):
            if a.action_type != 'dwarven_ring':
                continue
            if best < 0:
                best = i
            if want is not None and a.params.get('old_face') == want.value:
                return i
        if best >= 0:
            return best
    elif act_type == 'combat_effect_choice':
        # Combat-card effect magnitude: match the logged count (and the
        # card when both sides know it)
        want = int(p.get('count', 0) or 0)
        cid = p.get('card_id') or _lookup_card_id(
            card_lookup, p.get('card_name', '') or '')
        for i, a in enumerate(actions):
            if a.action_type != 'combat_effect_choice':
                continue
            if cid and a.params.get('card_id') and \
                    a.params.get('card_id') != cid:
                continue
            if int(a.params.get('count', -1)) == want:
                return i
    elif act_type == 'companion_move':
        comps = p.get('companions') or []
        comp = str(comps[0]) if comps else str(p.get('companion', ''))
        t = s(p.get('to'))
        f = s(p.get('from'))
        if not p.get('hunt_placement'):
            # P3: when the log names the walked companions and a subset
            # action carries exactly that group, prefer it (split walks
            # replay exactly); otherwise the legacy (from,to) pass picks
            # the full-group action and the closed loop's die-less
            # continuations reconcile splits as before.
            _lg9 = {str(c) for c in (p.get('companions') or [])}
            if _lg9:
                for i, a in enumerate(actions):
                    if (a.action_type == 'companion_move'
                            and a.params.get('companions')
                            and s(a.params.get('to')) == t
                            and s(a.params.get('from')) == f
                            and {str(c) for c in a.params['companions']}
                            == _lg9):
                        return i
            best = -1
            for i, a in enumerate(actions):
                if a.action_type != 'companion_move':
                    continue
                if a.params.get('companions'):
                    continue  # subset actions only match by exact group
                if s(a.params.get('to')) == t:
                    if s(a.params.get('from')) == f:
                        return i
                    if best < 0:
                        best = i
            if best >= 0:
                return best
        # Hunt-casualty hobbit placement: the DECISION is the final region
        # (1.147: nearest-waypoint fallback, same drag-collapse class)
        best = -1
        best_d = None
        for i, a in enumerate(actions):
            if (a.action_type == 'companion_separate'
                    and (not comp
                         or s(a.params.get('companion')) == s(comp))):
                cand = s(a.params.get('to_region'))
                if cand == t:
                    return i
                d = _region_hops(cand, t)
                if best < 0 or (d is not None
                                and (best_d is None or d < best_d)):
                    best, best_d = i, d
        return best
    elif act_type == 'elven_ring':
        # Learned ring decision (user ruling 2026-07-03): prefer the exact
        # old-face -> new-face change (schedule-enriched params); fall back
        # to new-face only, then any ring action (parser records carry only
        # the ring name)
        def _fv(x):
            return getattr(x, 'value', x)
        want_old, want_new = _fv(p.get('old_face')), _fv(p.get('new_face'))
        if not want_old:
            # 1.148: the logged die_used face IS the converted die's old
            # face (Scribe-drag records carry no explicit faces).  The
            # blind ring_idxs[0] fallback picked the FIRST variant - the
            # Will die - teaching FP to burn W in the exact line where W
            # crowns Aragorn (Adobim T1: human converted MUSTER).
            _du = p.get('die_used')
            if isinstance(_du, (tuple, list)) and len(_du) == 2:
                try:
                    want_old = ActionDieFace[str(_du[1])].value
                except Exception:
                    want_old = None
        ring_idxs = [i for i, a in enumerate(actions)
                     if a.action_type == 'elven_ring']
        if not ring_idxs:
            return -1
        if want_new:
            exact = [i for i in ring_idxs
                     if actions[i].params.get('new_face') == want_new
                     and (not want_old
                          or actions[i].params.get('old_face') == want_old)]
            if exact:
                return exact[0]
            by_new = [i for i in ring_idxs
                      if actions[i].params.get('new_face') == want_new]
            if by_new:
                return by_new[0]
        if want_old:
            by_old = [i for i in ring_idxs
                      if actions[i].params.get('old_face') == want_old]
            if by_old:
                # generation orders new faces CHARACTER first - the
                # deterministic pick for a faceless record
                return by_old[0]
        return ring_idxs[0]
    elif act_type == 'use_token':
        # Parser reports 'political' / 'card draw'; engine tokens are
        # 'muster' / 'draw'
        tk = str(p.get('token', p.get('token_type', ''))).lower()
        want_type = 'muster' if 'politic' in tk or 'muster' in tk else 'draw'
        want_nat = s(p.get('nation')) if p.get('nation') else ''
        want_deck = str(p.get('deck', '')).lower()
        best = -1
        for i, a in enumerate(actions):
            if a.action_type != 'use_token' or \
                    a.params.get('token_type') != want_type:
                continue
            if want_type == 'muster' and want_nat:
                if s(a.params.get('nation')) == want_nat:
                    return i
            elif want_type == 'draw' and want_deck:
                if str(a.params.get('deck', '')).lower() == want_deck:
                    return i
            if best < 0:
                best = i
        return best
    return -1


def _build_bc_candidates(state_pre: 'GameState', act_side: Side, act_type: str,
                         params: dict, card_lookup: 'Dict[str, str]',
                         rng) -> 'Optional[Tuple[np.ndarray, np.ndarray, int]]':
    """Build a behavioral-cloning candidate set for one logged decision.

    Enumerates the engine's legal actions at the pre-action state, matches the
    human's logged choice, and samples up to BC_NUM_ALTERNATIVES distinct
    alternatives. Returns (encs, regions, n_cands) with the chosen action in
    slot 0, or None when the decision cannot be matched (counted by caller).
    """
    # Ground-truth hand correction: the human PLAYED this card, so it was
    # in their hand — tracked-hand drift (undecodable ability draws) must
    # not block the match. Swap a placeholder for it, or insert it.
    # (Same ground truth applies to combat-card commitments.)
    _gt_play_cid = None
    if act_type in ('play_event', 'combat_card'):
        cid = _lookup_card_id(card_lookup, (params or {}).get('card_name', ''))
        if cid:
            _gt_play_cid = cid
            hand = state_pre.hands[act_side].hand
            if cid not in hand:
                for i, item in enumerate(hand):
                    if isinstance(item, str) and item.startswith('_unknown_'):
                        hand[i] = cid
                        break
                else:
                    hand.append(cid)
    def _enumerate(st):
        synth = _synthetic_decision_actions(st, act_side, act_type, params,
                                            card_lookup)
        if synth is not None:
            return synth
        if act_type == 'fellowship_declare':
            # Declarations are a separate free-action phase, not part of
            # get_legal_actions. The alternatives the human rejected are the
            # other reachable regions plus not declaring at all.
            # VASSAL logs reset the progress track BEFORE the pawn move, so
            # restore the true pre-decision progress from the logged value.
            # 2026-08-02: mordor_entry declares carry no
            # progress_consumed — progress_before (parser enrichment,
            # always present now) is the correct fallback; without it a
            # standstill gateway entry enumerated an EMPTY declare menu.
            _p9 = params or {}
            consumed = int(_p9.get('progress_consumed')
                           or _p9.get('progress_before') or 0)
            if consumed > st.fellowship.progress:
                st = clone_state(st)
                st.fellowship.progress = consumed
            acts = _get_declare_actions(st)
            acts.append(Action(die_idx=-1, action_type='pass', params={}))
            return acts
        return get_legal_actions(st, act_side)

    try:
        legal = _enumerate(state_pre)
    except Exception:
        return None
    idx = _match_logged_action(legal, act_type, params, card_lookup) \
        if legal else -1
    if idx < 0:
        # Retry with relaxed dice: the human demonstrably took this action,
        # but the replayer often marks the spent die as used BEFORE the
        # action line, so the engine sees no legal die for it.
        relaxed = clone_state(state_pre)
        for d in relaxed.dice.get(act_side, []):
            if d.current is not None:
                d.used = False
        try:
            legal = _enumerate(relaxed)
        except Exception:
            return None
        idx = _match_logged_action(legal, act_type, params, card_lookup)
        state_used = relaxed
        if idx < 0 and _gt_play_cid is not None:
            # Ground truth: the human played this card. The engine's no-op
            # dry-run filter (empty-choices) may have excluded it — the
            # real play had choices. Add a synthetic play action so the
            # decision is learnable.
            _synth_type = ('play_event' if act_type == 'play_event'
                           else 'combat_card_play')
            legal = list(legal) + [Action(
                die_idx=-1, action_type=_synth_type,
                params={'card_id': _gt_play_cid,
                        'side': 'fp' if act_side == Side.FREE_PEOPLES
                        else 'sh'})]
            idx = len(legal) - 1
        if idx < 0:
            # Final retry: grant one synthetic unused die of the face this
            # action class requires (see _BC_FALLBACK_FACE). The human took
            # the action; only the replay's die-face bookkeeping disagrees.
            face_need = _BC_FALLBACK_FACE.get(act_type)
            if face_need is not None:
                forced = clone_state(relaxed)
                extra = ActionDie([face_need], act_side)
                extra.current = face_need
                forced.dice.setdefault(act_side, []).append(extra)
                try:
                    legal_f = _enumerate(forced)
                except Exception:
                    legal_f = []
                idx_f = _match_logged_action(legal_f, act_type, params,
                                             card_lookup)
                if idx_f >= 0:
                    legal, idx, state_used = legal_f, idx_f, forced
        if idx < 0:
            return None
        state_pre = state_used  # encode candidates against the retry state
    chosen = legal[idx]
    # Army moves: encode the EXACT human split (the menu option matched is
    # the nearest approximation; the executor supports arbitrary counts)
    if act_type == 'army_move' and chosen.action_type == 'army_move':
        moved = (params or {}).get('moved') or {}
        mr, me = moved.get('regular', 0), moved.get('elite', 0)
        if mr or me:
            # USER 2026-07-11: exact figure split - nazgul counted
            # separately (the old bool lumped them with leaders, so a
            # human split taking a nazgul but not the leader, or vice
            # versa, was unrepresentable)
            take = moved.get('leader', 0) > 0
            chosen = Action(die_idx=chosen.die_idx, action_type='army_move',
                            params=dict(chosen.params, n_regular=mr,
                                        n_elite=me, take_leaders=take,
                                        n_nazgul=moved.get('nazgul', 0)))
    try:
        chosen_enc = encode_action(chosen, state_pre, act_side).astype(np.float16)
        chosen_reg = _get_action_region_index(chosen)
    except Exception:
        return None
    # Distinct alternatives, deduplicated by ENCODING (+region): two actions
    # the encoder cannot distinguish must not appear as chosen-vs-alternative
    # (identical inputs with opposite targets are irreducible label noise).
    seen = {(chosen_enc.tobytes(), chosen_reg)}
    alt_pool = []
    for a in legal:
        if a is chosen:
            continue
        try:
            enc = encode_action(a, state_pre, act_side).astype(np.float16)
            reg = _get_action_region_index(a)
        except Exception:
            continue
        key = (enc.tobytes(), reg)
        if key in seen:
            continue
        seen.add(key)
        alt_pool.append((enc, reg))
    if not alt_pool:
        return None  # Forced move — nothing to learn from
    if len(alt_pool) > BC_NUM_ALTERNATIVES:
        alt_pool = rng.sample(alt_pool, BC_NUM_ALTERNATIVES)
    encs = np.zeros((BC_NUM_SLOTS, ACTION_ENCODING_DIM), dtype=np.float16)
    regions = np.full(BC_NUM_SLOTS, -1, dtype=np.int16)
    encs[0] = chosen_enc
    regions[0] = chosen_reg
    for j, (enc, reg) in enumerate(alt_pool, start=1):
        encs[j] = enc
        regions[j] = reg
    return encs, regions, 1 + len(alt_pool)

















# ===================================================================
# Step 4: Value Network
# ===================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
    # Auto-detect CUDA device (device_count guard: with
    # CUDA_VISIBLE_DEVICES='' this torch build reports is_available()
    # True with ZERO devices and get_device_name(0) raises)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        DEVICE = torch.device('cuda')
        print(f"[AI] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("[AI] CUDA not available, using CPU")
except ImportError:
    HAS_TORCH = False
    DEVICE = None


if HAS_TORCH:
    # ---------------------------------------------------------------
    # GNN Board Encoder
    # ---------------------------------------------------------------

    class GCNLayer(nn.Module):
        """Single graph convolution layer with residual connection."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.residual = (nn.Linear(in_dim, out_dim, bias=False)
                             if in_dim != out_dim else nn.Identity())

        def forward(self, x, edge_src, edge_dst, deg_inv_sqrt):
            """Message passing on batched node features.

            Args:
                x: (batch, num_nodes, in_dim)
                edge_src, edge_dst: (num_edges,) LongTensor on same device
                deg_inv_sqrt: (num_nodes,) float on same device
            Returns:
                (batch, num_nodes, out_dim)
            """
            B, N, _ = x.shape
            # Gather source node features -> (batch, num_edges, in_dim)
            src_feats = x[:, edge_src, :]
            # Normalize by degree: D^{-1/2} A D^{-1/2}
            src_norm = deg_inv_sqrt[edge_src].unsqueeze(0).unsqueeze(-1)  # (1, E, 1)
            dst_norm = deg_inv_sqrt[edge_dst].unsqueeze(0).unsqueeze(-1)  # (1, E, 1)
            msg = src_feats * src_norm * dst_norm  # (B, E, in_dim)
            # Scatter-add to destination nodes
            out = torch.zeros(B, N, msg.shape[-1], device=x.device, dtype=x.dtype)
            dst_exp = edge_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, msg.shape[-1])
            out.scatter_add_(1, dst_exp, msg)
            # Transform + BN + ReLU + residual
            out = self.linear(out)
            out = self.bn(out.reshape(-1, out.shape[-1])).reshape(B, N, -1)
            out = torch.relu(out + self.residual(x))
            return out

    class BoardGNN(nn.Module):
        """2-layer GCN over the static WOTR board graph with mean-pool readout."""

        def __init__(self, node_feat_dim: int = NODE_FEAT_DIM,
                     hidden_dim: int = 64, num_layers: int = 2,
                     output_dim: int = 64):
            super().__init__()
            self.num_nodes = NUM_REGIONS
            # Register static graph structure as buffers (moved with .to(device))
            edge_src = torch.tensor(_EDGE_SRC, dtype=torch.long)
            edge_dst = torch.tensor(_EDGE_DST, dtype=torch.long)
            self.register_buffer('edge_src', edge_src)
            self.register_buffer('edge_dst', edge_dst)
            # Precompute degree normalization: D^{-1/2}
            deg = torch.zeros(NUM_REGIONS, dtype=torch.float32)
            for d in _EDGE_DST:
                deg[d] += 1.0
            deg = deg.clamp(min=1.0)
            deg_inv_sqrt = deg.pow(-0.5)
            self.register_buffer('deg_inv_sqrt', deg_inv_sqrt)
            # GCN layers
            dims = [node_feat_dim] + [hidden_dim] * num_layers
            self.layers = nn.ModuleList([
                GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
            ])
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, node_features):
            """Forward pass.

            Args:
                node_features: (batch, num_nodes, node_feat_dim)
            Returns:
                graph_out: (batch, output_dim)
                node_emb: (batch, num_nodes, hidden_dim) — per-node GCN embeddings
            """
            x = node_features
            for layer in self.layers:
                x = layer(x, self.edge_src, self.edge_dst, self.deg_inv_sqrt)
            # x is (B, 105, 64) — rich per-node embeddings
            node_emb = x
            pooled = x.mean(dim=1)  # (B, 64)
            graph_out = self.output_proj(pooled)  # (B, output_dim)
            return graph_out, node_emb

    # ---------------------------------------------------------------
    # Action encoding for hierarchical policy
    # ---------------------------------------------------------------
    # 25 base features + 8-dim nation one-hot [25:33] + card-index slot [33]
    # + decision-subtype one-hot [34:34+len(_DECISION_SUBTYPES)] for the
    # pending-phase decisions (hunt / combat / rearguard / combat-card /
    # combat-card effect choices) that previously all encoded
    # identically within their category.  INVARIANT: the one-hot width
    # is len(_DECISION_SUBTYPES) — never quote a hardcoded count.
    _DECISION_SUBTYPES = [
        'hunt_take_corruption', 'hunt_sacrifice_guide', 'hunt_sacrifice_random',
        'hunt_gollum_reveal', 'hunt_gollum_no_reveal',
        'combat_no_retreat', 'combat_continue', 'combat_cease',
        'combat_retreat', 'retreat_into_siege',
        'siege_assault', 'siege_continue', 'siege_press', 'siege_no_press',
        'rearguard_none', 'hunt_guide_separate',
        'casualty_preserve_units', 'casualty_preserve_elites',
        'combat_card_play', 'combat_card_none',
        # Choice-bearing combat-card effect magnitude (Onslaught sacrifice,
        # Dread and Despair forfeit, Relentless Assault self-hits, Heroic
        # Death sacrifice); count scalar in slot [23], card idx in [33]
        'combat_effect_choice',
        # Gandalf 'shining' pre-combat-card decision (item 11); the yes/no
        # magnitude rides the count scalar [23] (1 = shine, 0 = decline)
        'combat_shine',
        # FORCED hunt-casualty hobbit placement (v32 fix 2026-07-28):
        # rides the companion_separate action type (die_idx=-1, pending
        # hunt phase 'hobbit_place') but is a DIFFERENT decision.
        # Without this bit ~85% of the corpus 'separation' class was
        # forced Merry/Pippin placements, contaminating the voluntary-
        # separation prior (turn-1 trollshaws separations, USER catch).
        'hobbit_place',
    ]
    _SUBTYPE_TO_IDX = {t: i for i, t in enumerate(_DECISION_SUBTYPES)}
    _SUBTYPE_TO_IDX['shine'] = _SUBTYPE_TO_IDX['combat_shine']
    _SUBTYPE_TO_IDX['shine_none'] = _SUBTYPE_TO_IDX['combat_shine']
    # post-roll allocation rides the casualty subtype slot; the
    # count scalar [23] distinguishes the magnitudes
    _SUBTYPE_TO_IDX['casualty_allocate'] = \
        _SUBTYPE_TO_IDX['casualty_preserve_units']
    # iter 36 FULL-INFORMATION extension (user directive: "literally
    # everything possible that the human sees") — ranges corrected
    # 2026-07-27 (+1 vs the original comments; symbols are the truth):
    # [56:60]  action-moved composition: n_regular/10, n_elite/10,
    #          n_leaders/7, n_nazgul/8 (moves, attacks, musters,
    #          retreats — R/E splits previously conflated in one total)
    # [60:73]  CHARACTER IDENTITY multi-hot (13): which specific
    #          companions/minions this action moves/separates/sacrifices
    # [73:76]  source-army own composition reg/10, elite/10, ldr+naz/8
    # [76:79]  target-region enemy composition reg/10, elite/10, ldr+naz/8
    _ACTION_CHARACTERS = [
        'Gandalf the Grey', 'Strider', 'Legolas', 'Gimli', 'Boromir',
        'Meriadoc', 'Peregrin', 'Aragorn', 'Gandalf the White', 'Gollum',
        'Witch King', 'Saruman', 'Mouth of Sauron',
    ]
    _ACTION_CHAR_IDX = {sanitize(n): i
                        for i, n in enumerate(_ACTION_CHARACTERS)}
    # common log/param aliases -> canonical slot
    for _al, _cn in (('merry', 'Meriadoc'), ('pippin', 'Peregrin'),
                     ('witchking', 'Witch King'),
                     ('witch king', 'Witch King'),
                     ('mouthofsauron', 'Mouth of Sauron')):
        _ACTION_CHAR_IDX.setdefault(sanitize(_al),
                                    _ACTION_CHAR_IDX[sanitize(_cn)])
    # NOTE: offsets are COMPUTED from list lengths - the symbols are
    # the truth; never quote absolute indices in comments.
    _ACT_COMP_OFF = 34 + len(_DECISION_SUBTYPES)
    _ACT_CHAR_OFF = _ACT_COMP_OFF + 4
    _ACT_SRC_OFF = _ACT_CHAR_OFF + len(_ACTION_CHARACTERS)
    _ACT_TGT_OFF = _ACT_SRC_OFF + 3
    # MORDOR-ENTRY flag (iter 39, user finding: the net could not
    # distinguish the entry-declare at a gateway from an ordinary
    # declare -- same class as the composition/identity gaps)
    _ACT_MORDOR_OFF = _ACT_TGT_OFF + 3
    # CARD-OPTION decision flag (user finding 2026-07-07: optioned
    # cards always took the handler's first-listed default; options are
    # now pending decisions — see card_options.py)
    _ACT_CARDOPT_OFF = _ACT_MORDOR_OFF + 1
    # acting-die FACE (decision-audit 2026-07-08: skip carries a
    # 'face' param the encoder ignored, and W-wild means the same action
    # type can ride different faces — which die an action burns was
    # invisible).  Face index /7, 0 for die-less actions.
    # Then SKIP flag (skip=discard a specific die vs pass=do nothing —
    # they encoded IDENTICALLY; 15 collapses in one audited game).
    _ACT_DIEFACE_OFF = _ACT_CARDOPT_OFF + 1
    _ACT_SKIP_OFF = _ACT_DIEFACE_OFF + 1
    # own units LEFT BEHIND at the source; then empties-source
    # flag — the leave-a-garrison decision existed only implicitly
    # (USER 2026-07-09: expose the context, no weighting)
    _ACT_LEAVE_OFF = _ACT_SKIP_OFF + 1
    _ACT_EMPTIES_OFF = _ACT_LEAVE_OFF + 1
    # v14 (P1-P5 card-option exposure): either-or pick value
    # (+1/-1: crebain discard/keep, balrog draw/keep, nazgul-search
    # move, Lure eliminate/corruption); chain-terminator flag
    # (_tsim/_ffm/_rage done); unit-kind scalar (regular/elite/
    # leader picks in eliminate / overstack_remove / rage_move);
    # option group size (move_group companions, split totals)
    _ACT_OPTVAL_OFF = _ACT_EMPTIES_OFF + 1
    _ACT_OPTTERM_OFF = _ACT_OPTVAL_OFF + 1
    _ACT_OPTKIND_OFF = _ACT_OPTTERM_OFF + 1
    _ACT_OPTSIZE_OFF = _ACT_OPTKIND_OFF + 1
    # v29 (2026-07-26): target region natively ENEMY-side / MY-side
    # bits (actor perspective; neutral regions leave both 0).  The
    # WHERE-ranking had no ownership signal for the target — the
    # is_stronghold bit [18] scored Moria exactly like Rivendell, and
    # v28 self-play declared the Fellowship INTO Moria 23x in 42 games.
    _ACT_TGT_ENEMYSIDE_OFF = _ACT_OPTSIZE_OFF + 1
    _ACT_TGT_MYSIDE_OFF = _ACT_TGT_ENEMYSIDE_OFF + 1

    # --- 1.149 TABLE-KILL CATEGORY (USER 2026-08-20: "give it its own
    # category and extract the training signal") ------------------------
    # The type one-hot is feat[0:14] and feat[14] is already the source
    # region, so a 15th category CANNOT be added in place - it would
    # shift every later feature and invalidate both the checkpoints and
    # the 7.5GB pre-encoded corpus.  APPENDED AT THE TAIL instead (the
    # standing convention for state features), so columns 0..91 keep
    # their exact meaning: the bit is 1 only for `discard_table_card`,
    # which still also lights the play_event bit it has always lit.
    # That gives the net a dedicated input to learn a table-kill
    # preference from, without a renumbering.
    #
    # GATED, DEFAULT OFF.  Turning it on changes ACTION_ENCODING_DIM,
    # and load_net's action_dim guard then REFUSES every existing
    # checkpoint (wotr_{fp,sh}_corpus_live.pt are action_dim=92) - i.e.
    # live play would stop dead.  The USER is testing on the current
    # net right now, so the default stays 92 until a v41-generation net
    # exists; set WOTR_ACT_GEN=v41 for extraction/training runs.
    ACT_GEN = os.environ.get('WOTR_ACT_GEN', '').strip().lower()
    _ACT_TABLEKILL_ON = (ACT_GEN == 'v41')
    _ACT_TABLEKILL_OFF = _ACT_TGT_MYSIDE_OFF + 1
    ACTION_ENCODING_DIM = (_ACT_TABLEKILL_OFF + 1 if _ACT_TABLEKILL_ON
                           else _ACT_TGT_MYSIDE_OFF + 1)  # 93 / 92
    _FACE_ORDER = {'CHARACTER': 1, 'ARMY': 2, 'MUSTER': 3,
                   'MUSTER_ARMY': 4, 'EVENT': 5, 'WILL_OF_WEST': 6,
                   'EYE': 7}
    _ACTION_CARD_IDX_DIM = 33  # raw card index (0-95) or -1

    def encode_action(action, state, side):
        """Encode an action as a fixed-length feature vector for the action scorer."""
        feat = np.zeros(ACTION_ENCODING_DIM, dtype=np.float32)
        # [0:14] action type one-hot (14 categories)
        canonical = _ATYPE_TO_CAT.get(action.action_type, 'other')
        feat[_ACTION_TYPE_TO_IDX.get(canonical, 12)] = 1.0
        # the 1.149 table kill's OWN bit (tail-appended; see the
        # ACTION_ENCODING_DIM block).  Without it the net scored
        # "spend a die + 2 cards to destroy an opponent's table card"
        # through its play-event head, with no way to tell the two
        # apart - USER 2026-08-20.
        if _ACT_TABLEKILL_ON \
                and action.action_type == 'discard_table_card':
            feat[_ACT_TABLEKILL_OFF] = 1.0
        p = action.params
        # [14] source region index (normalized)
        if 'from' in p:
            feat[14] = REGION_INDEX.get(sanitize(str(p['from'])), 0) / float(NUM_REGIONS)
        # [15] target region index
        target = p.get('to') or p.get('region') or p.get('to_region')
        if target:
            feat[15] = REGION_INDEX.get(sanitize(str(target)), 0) / float(NUM_REGIONS)
        # [16] unit count at source — OWN side only (mixed siege
        # entries hold both sides; counting the enemy garrison made
        # the source size wrong exactly where garrisons matter)
        if 'from' in p:
            army = _armies_lookup(state, p['from'])
            if army:
                _own_src = sum(
                    1 for u in army.units
                    if _side_of_nation(u.nation) == side)
                feat[16] = _own_src / 10.0
                if action.action_type in ('army_move',
                                          'victor_advance') \
                        and p.get('n_regular') is not None:
                    _mv9 = (int(p.get('n_regular') or 0)
                            + int(p.get('n_elite') or 0))
                    _left9 = max(0, _own_src - _mv9)
                    feat[_ACT_LEAVE_OFF] = _left9 / 10.0
                    feat[_ACT_EMPTIES_OFF] = 1.0 if _left9 == 0 else 0.0
        # [17] nation index (scalar, kept for continuity) + [25:33] one-hot
        if 'nation' in p:
            nat_list = ['gondor', 'rohan', 'the_north', 'dwarves', 'elves',
                        'isengard', 'sauron', 'southron_easterlings']
            n = str(p['nation']).lower().replace(' ', '_')
            if hasattr(p['nation'], 'name'):
                n = p['nation'].name.lower()
            if n in nat_list:
                nat_idx = nat_list.index(n)
                feat[17] = (nat_idx + 1) / 8.0
                feat[25 + nat_idx] = 1.0
        # [18] target is stronghold, [19] target is city
        if target:
            tkey = sanitize(str(target))
            meta = board.regions.get(tkey, {})
            feat[18] = 1.0 if meta.get('is_stronghold') else 0.0
            feat[19] = 1.0 if meta.get('is_city') else 0.0
            # [88]/[89] native side of the target (fid 6-9 Shadow,
            # 1-5 FP, 10 free/neutral)
            _pid9 = meta.get('parent_id')
            if _pid9 in (6, 7, 8, 9):
                _tside9 = Side.SHADOW
            elif _pid9 in (1, 2, 3, 4, 5):
                _tside9 = Side.FREE_PEOPLES
            else:
                _tside9 = None
            if _tside9 is not None:
                if _tside9 == side:
                    feat[_ACT_TGT_MYSIDE_OFF] = 1.0
                else:
                    feat[_ACT_TGT_ENEMYSIDE_OFF] = 1.0
        # [20] distance to VP target from destination
        if target:
            tkey = sanitize(str(target))
            if side == Side.SHADOW:
                feat[20] = _DIST_TO_FP_TARGET.get(tkey, 10) / 10.0
            else:
                feat[20] = _DIST_TO_SH_TARGET.get(tkey, 10) / 10.0
        # [21] is_attack flag
        feat[21] = 1.0 if action.action_type in ('army_attack', 'siege_assault',
                                                    'siege_press') else 0.0
        # [22] die_idx / 7 (which die is used)
        # [22] DEAD (v30 skew fix, 2026-07-26): die INDEX is an
        # arbitrary pool position — extraction stages its one die at
        # idx 0 while live menus use real indices, so this column
        # train/serve-skewed every action.  The die FACE column carries
        # the semantics.  Kept as a column (layout stability), always 0.
        # [23] special: elven ring new_face / eye count / companion index
        if action.action_type == 'elven_ring':
            # Encode target die face as index
            faces = ['Character', 'Army', 'Muster', 'Event', 'Muster/Army',
                     'Eye of Sauron', 'Will of the West']
            face_str = p.get('new_face', '')
            feat[23] = (faces.index(face_str) + 1) / 7.0 if face_str in faces else 0.0
            # target_die_idx NOT encoded (col 22 dead — skew fix
            # 2026-07-26; extraction pool order != live pool order).
            # [24] the target die's OLD face (USER 2026-07-27): the
            # semantic identity of WHICH die is being changed — without
            # it, ring options changing different dice to the same new
            # face collapsed.  Params-sourced: the shared generator
            # stamps old_face on both the extraction and live paths.
            _of9 = p.get('old_face', '')
            feat[24] = ((faces.index(_of9) + 1) / 7.0
                        if _of9 in faces else 0.0)
        elif action.action_type == 'eye_allocate':
            feat[23] = p.get('count', 0) / 3.0
        elif action.action_type == 'casualty_allocate':
            feat[23] = p.get('count', 0) / 5.0
        elif action.action_type == 'nazgul_move':
            # v28 (USER 2026-07-24): the flight SIZE and the WK rider
            # were invisible - count/include_wk variants of one
            # (from,to) route encoded byte-identically (the v11
            # victor_advance collapse, one action type over).  The
            # no-count action means ALL plain Nazgul at the source, so
            # encode the EFFECTIVE count.
            _nz9 = p.get('count')
            if _nz9 is None:
                _na9 = _armies_lookup(state, p.get('from'))
                _nz9 = (sum(1 for l in _na9.leaders
                            if isinstance(l, Nazgul))
                        if _na9 else 0)
            feat[23] = min(int(_nz9), 8) / 8.0
            feat[24] = 1.0 if p.get('include_wk') else 0.0
        elif action.action_type == 'use_token':
            # token type + deck choice (nation rides the [25:33] one-hot)
            feat[23] = 0.4 if p.get('token_type') == 'muster' else 0.8
            if p.get('deck'):
                feat[24] = 0.5 if p.get('deck') == 'character' else 1.0
        elif action.action_type == 'dwarven_ring':
            # die_idx NOT encoded (col 22 dead — skew fix 2026-07-26);
            # the OLD face below carries the choice semantics.
            faces = ['Character', 'Army', 'Muster', 'Event', 'Muster/Army',
                     'Eye of Sauron', 'Will of the West']
            of = p.get('old_face', '')
            feat[23] = (faces.index(of) + 1) / 7.0 if of in faces else 0.0
        elif action.action_type == 'choose_guide':
            # review P2 (2026-07-27): real params carry 'Meriadoc' /
            # 'Peregrin' (never 'Merry'/'Pippin') and Gollum / Gandalf
            # the White are legal guides — the stale list zeroed
            # feat[23] for all four (identity still reached the net via
            # the character multi-hot, so this sub-feature was
            # redundant-wrong, not a hard collapse)
            companion_names = ['Gandalf the Grey', 'Strider', 'Legolas',
                               'Gimli', 'Boromir', 'Meriadoc', 'Peregrin',
                               'Aragorn', 'Gandalf the White', 'Gollum']
            cname = p.get('companion', '')
            feat[23] = ((companion_names.index(cname) + 1)
                        / float(len(companion_names))
                        if cname in companion_names else 0.0)
        # [23]/[24] for army moves: split size + leaders flag
        if (action.action_type in ('army_move', 'victor_advance')
                and p.get('n_regular') is not None):
            # total in the integer-ish range, elite count in the second
            # decimal — (1r,1e) and (2r,0e) must encode differently
            # (v11 audit: victor_advance 1e vs 2e collapsed)
            feat[23] = ((int(p.get('n_regular') or 0)
                         + int(p.get('n_elite') or 0)) / 10.0
                        + int(p.get('n_elite') or 0) / 100.0)
            feat[24] = 1.0 if p.get('take_leaders', True) else 0.0
        # [24] deck choice for draws (otherwise unused)
        if action.action_type == 'draw_card':
            feat[24] = 0.5 if p.get('deck') == 'character' else 1.0
        # [23]/[24] for musters: what is being recruited (without this,
        # "Saruman at Orthanc" was indistinguishable from "elite at Orthanc")
        if action.action_type == 'muster_recruit':
            feat[24] = 0.5 if p.get('type') == 'leader' else 0.25
        elif action.action_type == 'muster_elite':
            feat[24] = 0.75
        elif action.action_type == 'muster_minion':
            feat[24] = 1.0
            _minions = {'witch king': 1, 'witchking': 1, 'saruman': 2,
                        'mouth of sauron': 3, 'mouthofsauron': 3}
            mname = str(p.get('minion', '')).lower().replace('_', ' ').strip()
            feat[23] = _minions.get(mname, 0) / 3.0
        # [33] card identity for event plays, hand-limit discards and
        # combat-card commitments (embedded by the scorer — the net sees
        # WHICH card is involved)
        feat[_ACTION_CARD_IDX_DIM] = -1.0
        if action.action_type in ('play_event', 'discard_card',
                                  'combat_card_play',
                                  'combat_effect_choice', 'card_option',
                                  'discard_table_card'):
            cidx = _get_card_idx_map().get(p.get('card_id', ''), -1)
            feat[_ACTION_CARD_IDX_DIM] = float(cidx)
        if action.action_type == 'card_option':
            feat[_ACT_CARDOPT_OFF] = 1.0
            from card_options import option_region as _optreg
            _cho = p.get('choices') or {}
            _r = _optreg(_cho)
            if _r:
                feat[15] = (REGION_INDEX.get(sanitize(str(_r)), 0)
                            / float(NUM_REGIONS))
            # v17 (amendment 1.131): chain-flight SOURCE region - two
            # options flying a nazgul to the same destination from
            # different regions were byte-identical (option_region
            # returns to_region first; feat[14] is free for options)
            _fr9 = _cho.get('from_region')
            if _fr9:
                feat[14] = (REGION_INDEX.get(sanitize(str(_fr9)), 0)
                            / float(NUM_REGIONS))
            # decision-audit 2026-07-08: NON-region option keys were
            # invisible (Wisdom of Elrond nation options collapsed).
            _cn = _cho.get('nation')
            if _cn:
                # Slot order MUST match nat_list ([25:33] one meaning
                # per slot): isengard=5, sauron=6.
                nat_list2 = ['gondor', 'rohan', 'the_north', 'dwarves',
                             'elves', 'isengard', 'sauron',
                             'southrons_easterlings']
                _cns = sanitize(str(_cn)).replace(' ', '_')
                _alias = {'north': 'the_north',
                          'southrons_and_easterlings':
                              'southrons_easterlings'}
                _cns = _alias.get(_cns, _cns)
                if _cns in nat_list2:
                    feat[25 + nat_list2.index(_cns)] = 1.0
            for _nm in (_cho.get('companions') or []):
                _i3 = _ACTION_CHAR_IDX.get(sanitize(str(_nm)))
                if _i3 is not None:
                    feat[_ACT_CHAR_OFF + _i3] = 1.0
            # v16: learned hunt-victim picks (Orc Patrol / Foul Thing)
            if _cho.get('victim'):
                _i3 = _ACTION_CHAR_IDX.get(sanitize(str(_cho['victim'])))
                if _i3 is not None:
                    feat[_ACT_CHAR_OFF + _i3] = 1.0
            _ca = str(_cho.get('companion_action') or '')
            if _ca.startswith(('sep_', 'move_')):
                _i3 = _ACTION_CHAR_IDX.get(
                    sanitize(_ca.split('_', 1)[1].replace('_', ' ')))
                if _i3 is not None:
                    feat[_ACT_CHAR_OFF + _i3] = 1.0
            if _cho.get('dice_count') is not None:
                feat[23] = float(_cho['dice_count']) / 3.0
            if _cho.get('count') is not None:
                feat[23] = float(_cho['count']) / 8.0
            # v28 (USER 2026-07-24): a WK chain hop and a plain-nazgul
            # hop on the same route encoded identically
            if _cho.get('include_wk'):
                feat[24] = 1.0
            # LOW-5 (2026-07-20): Dead Men recruit magnitude — without
            # this, options differing only in recruit_count encode
            # byte-identically (the SoMM spatial-invisibility class)
            if _cho.get('recruit_count') is not None:
                feat[23] = float(_cho['recruit_count']) / 3.0
            # v14: nested option specs surface their fields
            _nest9 = None
            for _nk9 in ('overstack_remove', 'move_group', 'rage_move',
                         'eliminate'):
                if isinstance(_cho.get(_nk9), dict):
                    _nest9 = _cho[_nk9]
                    break
            if _nest9 is not None:
                _nr9 = _nest9.get('region') or _nest9.get('to')
                if _nr9 and not feat[15]:
                    feat[15] = (REGION_INDEX.get(sanitize(str(_nr9)), 0)
                                / float(NUM_REGIONS))
                _nn9 = _nest9.get('nation')
                if _nn9:
                    _nns9 = sanitize(str(_nn9)).replace(' ', '_')
                    _nns9 = {'north': 'the_north',
                             'southrons_and_easterlings':
                             'southrons_easterlings'}.get(_nns9, _nns9)
                    # Slot order MUST match nat_list: isengard=5,
                    # sauron=6.
                    _nl9 = ['gondor', 'rohan', 'the_north', 'dwarves',
                            'elves', 'isengard', 'sauron',
                            'southrons_easterlings']
                    if _nns9 in _nl9:
                        feat[25 + _nl9.index(_nns9)] = 1.0
                feat[_ACT_OPTKIND_OFF] = {
                    'regular': 0.34, 'elite': 0.67,
                    'leader': 1.0}.get(str(_nest9.get('kind', '')), 0.0)
                _gc9 = _nest9.get('companions') or []
                for _nm9 in _gc9:
                    _i9 = _ACTION_CHAR_IDX.get(sanitize(str(_nm9)))
                    if _i9 is not None:
                        feat[_ACT_CHAR_OFF + _i9] = 1.0
                if _gc9:
                    feat[_ACT_OPTSIZE_OFF] = min(len(_gc9), 9) / 9.0
            # v14/v15: either-or boolean picks (True=+1 / False=-1)
            for _bk9 in ('crebain_discard', 'balrog_declare_draw',
                         'move_to_fellowship', 'mithril_redraw'):
                if _bk9 in _cho:
                    feat[_ACT_OPTVAL_OFF] = 1.0 if _cho[_bk9] else -1.0
                    break
            _la9 = str(_cho.get('action', ''))
            if _la9 == 'eliminate':
                feat[_ACT_OPTVAL_OFF] = 1.0
            elif _la9 == 'corruption':
                feat[_ACT_OPTVAL_OFF] = -1.0
            elif _la9 == 'hide':
                feat[_ACT_OPTVAL_OFF] = 1.0
            elif _la9 == 'move':
                feat[_ACT_OPTVAL_OFF] = -1.0
            elif _la9 == 'heal_only':
                feat[_ACT_OPTVAL_OFF] = 0.5
            # v17 (1.131): Nazgul Strike! post-move either/or - hunt
            # roll vs table-card discard vs decline must not collapse
            elif _la9 == 'hunt':
                feat[_ACT_OPTVAL_OFF] = 1.0
            elif _la9 == 'discard_table_card':
                feat[_ACT_OPTVAL_OFF] = -1.0
            elif _la9 == 'none':
                feat[_ACT_OPTVAL_OFF] = 0.25
            # v17 (1.131): the discard TARGET's identity outranks the
            # pending card's constant idx (several table cards may be
            # discardable; without this they collapse pairwise)
            if _cho.get('target_card'):
                _tci9 = _get_card_idx_map().get(
                    str(_cho['target_card']), -1)
                if _tci9 >= 0:
                    feat[_ACTION_CARD_IDX_DIM] = float(_tci9)
            # v14/v15: chain terminators; a '__remove_from_play__'
            # destination (Mordor-track separation) rides the same slot
            # (no board destination - the card idx disambiguates)
            if any(_cho.get(_tk9) for _tk9 in
                   ('_tsim_done', '_ffm_done', '_rage_done',
                    '_ents_chain_done', '_naz_done', '_comp_done',
                    '_tsl_done')):
                feat[_ACT_OPTTERM_OFF] = 1.0
            if str(_cho.get('destination', '')) == '__remove_from_play__':
                feat[_ACT_OPTTERM_OFF] = 1.0
            # v15: the Ents free-chain PICK - the played card's identity
            # IS the decision; the pending's own card is constant
            _ecp9 = _cho.get('ents_chain_play')
            if _ecp9:
                feat[_ACTION_CARD_IDX_DIM] = float(
                    _get_card_idx_map().get(str(_ecp9), -1))
            # v14: TDaaN split composition rides the standard slots
            _cnr9 = _cho.get('n_regular')
            _cne9 = _cho.get('n_elite')
            if _cnr9 is not None or _cne9 is not None:
                feat[_ACT_COMP_OFF + 0] = float(_cnr9 or 0) / 10.0
                feat[_ACT_COMP_OFF + 1] = float(_cne9 or 0) / 10.0
                feat[_ACT_OPTSIZE_OFF] = min(
                    float(_cnr9 or 0) + float(_cne9 or 0), 10) / 10.0
        # [80] acting-die face; [81] freed by the skip removal — now
        # the nazgul_move include_wk flag (else the two variants would
        # encode identically = an encoding collapse)
        if p.get('include_wk'):
            feat[_ACT_SKIP_OFF] = 1.0
        if action.action_type == 'skip':
            feat[_ACT_SKIP_OFF] = 1.0
            _fname = str(p.get('face', '') or '')
            feat[_ACT_DIEFACE_OFF] = _FACE_ORDER.get(_fname, 0) / 7.0
        elif action.action_type == 'pass':
            # v30 skew fix (2026-07-26): pass is canonically FACE-LESS.
            # Extraction passes ride the staged die (idx 0, face set) so
            # the die-idx branch stamped col 81 on them, while live
            # passes (die_idx=-1) encoded 0 — the only face-less-pass
            # positives were the decline-declare samples, and the live
            # prior collapsed onto pass (=1.0000 at every root).
            pass
        elif action.die_idx is not None and action.die_idx >= 0:
            try:
                _dl = state.dice.get(side, [])
                if action.die_idx < len(_dl) and                         _dl[action.die_idx].current is not None:
                    feat[_ACT_DIEFACE_OFF] = _FACE_ORDER.get(
                        _dl[action.die_idx].current.name, 0) / 7.0
            except Exception:
                pass
        # [34:55] decision subtype one-hot (hunt / combat-phase / rearguard
        # / combat-card / combat-card effect choice)
        sub = _SUBTYPE_TO_IDX.get(action.action_type)
        if sub is not None:
            feat[34 + sub] = 1.0
            if action.action_type == 'rearguard_split':
                feat[23] = p.get('leave_behind', 1) / 10.0
            elif action.action_type == 'combat_effect_choice':
                # Choice magnitude (like rearguard size): sacrifice /
                # forfeit count, 0 = decline
                feat[23] = p.get('count', 0) / 10.0
            elif action.action_type in ('shine', 'shine_none'):
                # binary shine decision rides the count slot (item 11)
                feat[23] = 1.0 if action.action_type == 'shine' else 0.0
        elif action.action_type == 'companion_separate':
            # v32 (2026-07-28): a FORCED hunt-casualty hobbit placement
            # reuses the companion_separate type — mark it with its own
            # subtype so the net can tell it from a VOLUNTARY separation
            _ph9e = getattr(state, 'pending_hunt', None)
            if isinstance(_ph9e, dict) \
                    and _ph9e.get('phase') == 'hobbit_place':
                feat[34 + _SUBTYPE_TO_IDX['hobbit_place']] = 1.0

        # ---- iter 36 full-information features ----
        at = action.action_type

        def _army_comp(region, want_side):
            """(reg, elite, leaders+nazgul) of want_side at region."""
            if not region:
                return 0, 0, 0
            a = _armies_lookup(state, str(region))
            if a is None:
                return 0, 0, 0
            r = e = l = 0
            for u in a.units:
                if _side_of_nation(u.nation) == want_side:
                    if u.kind == UnitKind.REGULAR:
                        r += 1
                    else:
                        e += 1
            for ld in a.leaders:
                try:
                    ls = (Side.FREE_PEOPLES if isinstance(ld, Companion)
                          else Side.SHADOW if (isinstance(ld, Nazgul)
                                               or type(ld).__name__
                                               == 'Minion')
                          else _side_of_nation(ld.nation))
                except Exception:
                    continue
                if ls == want_side:
                    l += 1
            return r, e, l

        # [55:59] action-moved composition
        n_r = n_e = n_l = n_z = 0
        src_r, src_e, src_l = _army_comp(p.get('from'), side)
        if at == 'army_move':
            n_r = int(p.get('n_regular') or 0)
            n_e = int(p.get('n_elite') or 0)
            if p.get('take_leaders', True):
                a0 = _armies_lookup(state, str(p.get('from', '')))
                if a0 is not None:
                    for ld in a0.leaders:
                        if isinstance(ld, Nazgul):
                            if side == Side.SHADOW:
                                n_z += 1
                        else:
                            try:
                                ls = (Side.FREE_PEOPLES
                                      if isinstance(ld, Companion)
                                      else Side.SHADOW
                                      if type(ld).__name__ == 'Minion'
                                      else _side_of_nation(ld.nation))
                            except Exception:
                                continue
                            if ls == side:
                                n_l += 1
            # v16.1 (amendment 1.130): the 1.128 n_nazgul split was
            # documented but never wired here - '(L)' and '(L nz=0)'
            # variants encoded byte-identically (label noise + a
            # non-choice for the net).  The explicit count wins.
            if p.get('n_nazgul') is not None:
                n_z = int(p.get('n_nazgul') or 0)
        elif at == 'victor_advance':
            # 1.150: the advance composition (incl. n_nazgul) is a
            # decision dimension; without this the nz variants encoded
            # byte-identically.
            n_r = int(p.get('n_regular') or 0)
            n_e = int(p.get('n_elite') or 0)
            n_z = int(p.get('n_nazgul') or 0)
        elif at in ('army_attack', 'combat_retreat'):
            # the whole own force at the source/battle region moves/fights
            n_r, n_e, n_l = _army_comp(p.get('from') or p.get('to'), side)
        elif at == 'muster_recruit':
            typ = str(p.get('type', 'regular'))
            n_r = 1 if typ == 'regular' else 0
            n_e = 1 if typ == 'elite' else 0
            n_l = 1 if typ == 'leader' else 0
            n_z = 1 if typ == 'nazgul' else 0
        elif at == 'muster_elite':
            n_e = 1
        elif at == 'muster_minion':
            n_l = 1
        elif at == 'nazgul_move':
            a0 = _armies_lookup(state, str(p.get('from', '')))
            n_z = sum(1 for ld in (a0.leaders if a0 else [])
                      if isinstance(ld, Nazgul))
            # v16.1 (amendment 1.130): count subsets were invisible -
            # 'count' caps the moved figure count
            if p.get('count') is not None:
                n_z = min(n_z, int(p.get('count') or 0))
        elif at == 'mouth_move':
            n_l = 1   # one minion figure (P2-02 ruling 2026-07-20)
        elif at == 'siege_trim':
            # k elites returned to reinforcements (ruling 2026-07-20)
            n_e = int(p.get('elites_back', 0) or 0)
            n_r = max(0, int(p.get('excess', 0) or 0) - n_e)
        feat[_ACT_COMP_OFF + 0] = n_r / 10.0
        feat[_ACT_COMP_OFF + 1] = n_e / 10.0
        feat[_ACT_COMP_OFF + 2] = n_l / 7.0
        feat[_ACT_COMP_OFF + 3] = n_z / 8.0

        # [59:72] character identity multi-hot
        def _mark_char(nm):
            if not nm:
                return
            i2 = _ACTION_CHAR_IDX.get(sanitize(str(nm)))
            if i2 is not None:
                feat[_ACT_CHAR_OFF + i2] = 1.0

        if at in ('companion_separate', 'choose_guide'):
            _mark_char(p.get('companion'))
            for nm in (p.get('companions') or []):
                _mark_char(nm)
        elif at == 'companion_move':
            names = list(p.get('companions') or [])
            if p.get('companion'):
                names.append(p.get('companion'))
            if not names:
                # engine group semantics: moves every Companion at 'from'
                a0 = _armies_lookup(state, str(p.get('from', '')))
                names = [ld.name for ld in (a0.leaders if a0 else [])
                         if isinstance(ld, Companion)]
            for nm in names:
                _mark_char(nm)
        elif at == 'hunt_sacrifice_guide':
            nm = p.get('forced_companion') or p.get('companion')
            if not nm and state.fellowship.guide is not None:
                nm = state.fellowship.guide.name
            _mark_char(nm)
        # hunt_sacrifice_random: the victim is a FUTURE random draw.
        # Encoding the log's drawn name (forced_companion) LEAKED the
        # label — teacher samples had the character bit lit, live
        # candidates never do, so the trained prior for random was
        # floored to 0.0 in every live scenario while scoring 9/9 on
        # (leaked) corpus samples (USER 2026-07-09 root cause). The
        # random candidate encodes NO character identity.
        elif at == 'muster_minion':
            _mark_char(p.get('minion'))
        elif at == 'crown_aragorn':
            _mark_char('Aragorn')
        elif at == 'special_character':
            _mark_char(p.get('character') or p.get('minion'))

        # [72:75] source own composition; [75:78] target enemy composition
        feat[_ACT_SRC_OFF + 0] = src_r / 10.0
        feat[_ACT_SRC_OFF + 1] = src_e / 10.0
        feat[_ACT_SRC_OFF + 2] = src_l / 8.0
        tgt_reg = p.get('to') or p.get('region') or p.get('to_region')
        en_side = (Side.SHADOW if side == Side.FREE_PEOPLES
                   else Side.FREE_PEOPLES)
        tr, te, tl = _army_comp(tgt_reg, en_side)
        feat[_ACT_TGT_OFF + 0] = tr / 10.0
        feat[_ACT_TGT_OFF + 1] = te / 10.0
        feat[_ACT_TGT_OFF + 2] = tl / 8.0
        # [78] Mordor-entry declare flag (iter 39)
        if at == 'fellowship_declare':
            _dreg = sanitize(str(p.get('region', '') or p.get('to', '')))
            if _dreg in ('morannon', 'minas morgul')                     or p.get('mordor_entry'):
                feat[_ACT_MORDOR_OFF] = 1.0
        return feat

    def _get_action_region_index(action) -> int:
        """Return primary region index (0-104) for an action, or -1 if non-regional."""
        p = action.params
        # card_option: the region lives inside the choices dict (USER
        # 2026-07-08: SoMM Mount-Gram-vs-Moria was spatially INVISIBLE —
        # both options returned -1 and the GNN scorer could not
        # distinguish them; priors came out exactly uniform)
        if action.action_type == 'card_option':
            from card_options import option_region
            r = option_region(p.get('choices') or {})
            if r:
                idx = REGION_INDEX.get(sanitize(str(r)))
                if idx is not None:
                    return idx
        for key in ('to', 'to_region', 'region'):
            val = p.get(key, '')
            if val:
                idx = REGION_INDEX.get(sanitize(str(val)))
                if idx is not None:
                    return idx
        val = p.get('from', '')
        if val:
            idx = REGION_INDEX.get(sanitize(str(val)))
            if idx is not None:
                return idx
        return -1  # pass, draw_card, fellowship, etc.

    # -----------------------------------------------------------------
    # Cached encoding for the MCTS hot path. encode_action depends on the
    # state ONLY through the unit count of the source army (feat[16]), so
    # results are globally cacheable by action identity + that count.
    # -----------------------------------------------------------------
    _ENC_CACHE: dict = {}
    _RIDX_CACHE: dict = {}

    def _action_cache_key(action, state, side, comp_memo=None):
        p = action.params or {}
        if action.action_type == 'companion_separate'                 and p.get('companions') and comp_memo is not None:
            # FAST PATH (profiled 2026-08-22, uncapped lattice: the
            # ~35-element key was built ~800k times per search).  The
            # elements the full key carries that can DIFFER for a group
            # separation: die face, the hobbit_place subtype, the
            # companions, the destination and the ENEMY composition at
            # the destination ([75:78]); everything else is None for
            # this shape.  The guard test (key-equality implies
            # encoding-equality) covers this path through test_rules.
            _face = None
            if action.die_idx is not None and action.die_idx >= 0:
                _dl = state.dice.get(side, [])
                if action.die_idx < len(_dl)                         and _dl[action.die_idx].current is not None:
                    _face = _dl[action.die_idx].current.name
            _phk9 = getattr(state, 'pending_hunt', None)
            _hplace = (isinstance(_phk9, dict)
                       and _phk9.get('phase') == 'hobbit_place')
            _tgt9 = p.get('to_region')
            _ek9 = ('__enemy__', _tgt9)
            _tgt_en = comp_memo.get(_ek9)
            if _tgt_en is None:
                _tgt_en = _enemy_comp_at(state, side, _tgt9)
                comp_memo[_ek9] = _tgt_en
            return ('sep', _hplace, side, action.die_idx, _face,
                    tuple(p['companions']), _tgt9, _tgt_en)

        def _comp_at(rname):
            # PROFILE (session 34): within one expansion the state is
            # fixed and ~500 candidates share a handful of regions —
            # comp_memo (fresh dict per expansion) collapses 150K+
            # army scans per search to a few dozen.  KEY CONTENT is
            # unchanged (memo only avoids recomputation).
            if comp_memo is not None and rname in comp_memo:
                return comp_memo[rname]
            r9 = _comp_at_raw(rname)
            if comp_memo is not None:
                comp_memo[rname] = r9
            return r9

        def _comp_at_raw(rname):
            # complete own-side composition (r, e, l, z) - covers
            # feat[16]/leave-off AND the [55:59] composition slots
            # (1.132: the old total-unit count missed the r/e split,
            # leaders and nazgul, so attacks/moves cached at one
            # composition were reused at another)
            if rname is None:
                return None
            a = _armies_lookup(state, str(rname))
            if a is None:
                return (0, 0, 0, 0)
            r = e = l = z = 0
            for u in a.units:
                if _side_of_nation(u.nation) != side:
                    continue
                if u.kind == UnitKind.REGULAR:
                    r += 1
                else:
                    e += 1
            for ld in a.leaders:
                if isinstance(ld, Nazgul):
                    if side == Side.SHADOW:
                        z += 1
                    continue
                try:
                    ls = (Side.FREE_PEOPLES if isinstance(ld, Companion)
                          else Side.SHADOW
                          if type(ld).__name__ == 'Minion'
                          else _side_of_nation(ld.nation))
                except Exception:
                    continue
                if ls == side:
                    l += 1
            return (r, e, l, z)

        frm = p.get('from')
        n_at_from = _comp_at(frm)
        if n_at_from is None and action.action_type in (
                'army_attack', 'combat_retreat'):
            n_at_from = _comp_at(p.get('to'))
        # 1.132 (user QA, serial guide-burn root cause): encode_action
        # is state-dependent BEYOND the source count -
        # (a) the acting die's FACE rides feat[80]; the die at an index
        #     changes between action windows AND episodes (the cache is
        #     process-wide), so a stale face poisoned die-action priors;
        # (b) hunt_sacrifice_guide encodes the VICTIM (current guide) at
        #     the character one-hot - a cached Gandalf-victim encoding
        #     served at a Strider decision made the net answer the wrong
        #     question (humans burn Gandalf 73%, Strider 6%), driving
        #     Gandalf->Strider->Legolas->Gimli serial sacrifices.
        _face = None
        if action.die_idx is not None and action.die_idx >= 0:
            _dl = state.dice.get(side, [])
            if action.die_idx < len(_dl) \
                    and _dl[action.die_idx].current is not None:
                _face = _dl[action.die_idx].current.name
        _guide = None
        if action.action_type in ('hunt_sacrifice_guide',
                                  'hunt_guide_separate'):
            _guide = getattr(state.fellowship.guide, 'name', None)
        nation = p.get('nation')
        ch = p.get('choices')
        ch_key = (tuple(sorted((k, str(v)) for k, v in ch.items()))
                  if isinstance(ch, dict) else None)
        # 1.138 (review agent C, probe-confirmed): the key MUST mirror
        # every param encode_action reads - missing fields collided
        # companion GROUP moves (full group vs subsets shared one slot,
        # 'move Aragorn alone' served the whole-group encoding into the
        # live MCTS priors), minion identities, n_nazgul escorts,
        # include_wk, token types.  Guard test in test_rules asserts
        # key-equality implies encoding-equality.
        _comps = p.get('companions')
        _comps_k = tuple(sorted(str(c) for c in _comps))             if _comps else None
        # v32 (2026-07-28): the hobbit_place subtype bit reads the
        # pending-hunt phase — the key must mirror it (a cached
        # voluntary-separation encoding must not serve a forced
        # placement, or vice versa)
        _hplace = None
        if action.action_type == 'companion_separate':
            _phk9 = getattr(state, 'pending_hunt', None)
            _hplace = (isinstance(_phk9, dict)
                       and _phk9.get('phase') == 'hobbit_place')
        # Full-group companion_move (params carry only from/to) encodes
        # the identities of every Companion at 'from' — the key must
        # mirror them or equal-count groups collide (1.138 class).
        if action.action_type == 'companion_move' and _comps_k is None \
                and not p.get('companion'):
            _a9k = _armies_lookup(state, str(frm or ''))
            _comps_k = tuple(sorted(
                ld.name for ld in (_a9k.leaders if _a9k else [])
                if isinstance(ld, Companion)))
        # encode_action embeds the ENEMY composition of the target
        # region ([75:78]); the key must mirror it or stale garrison
        # strengths are served across states (1.132 class).
        _tgt9 = p.get('to') or p.get('region') or p.get('to_region')
        _tgt_en = None
        if _tgt9 is not None:
            _ek9 = ('__enemy__', _tgt9)
            if comp_memo is not None and _ek9 in comp_memo:
                _tgt_en = comp_memo[_ek9]
            else:
                _a9e = _armies_lookup(state, str(_tgt9))
                if _a9e is None:
                    _tgt_en = (0, 0, 0)
                else:
                    _er = _ee = _el = 0
                    for u in _a9e.units:
                        if _side_of_nation(u.nation) == side:
                            continue
                        if u.kind == UnitKind.REGULAR:
                            _er += 1
                        else:
                            _ee += 1
                    for ld in _a9e.leaders:
                        if isinstance(ld, Nazgul):
                            if side == Side.FREE_PEOPLES:
                                _el += 1
                            continue
                        try:
                            ls = (Side.FREE_PEOPLES
                                  if isinstance(ld, Companion)
                                  else Side.SHADOW
                                  if type(ld).__name__ == 'Minion'
                                  else _side_of_nation(ld.nation))
                        except Exception:
                            continue
                        if ls != side:
                            _el += 1
                    _tgt_en = (_er, _ee, _el)
                if comp_memo is not None:
                    comp_memo[_ek9] = _tgt_en
        return (_hplace, side, action.action_type, action.die_idx, frm,
                p.get('to'), p.get('region'), p.get('to_region'),
                getattr(nation, 'value', nation), p.get('type'),
                p.get('count'), p.get('card_id'), p.get('companion'),
                p.get('deck'), p.get('new_face'), p.get('target_die_idx'),
                p.get('n_regular'), p.get('n_elite'), p.get('take_leaders'),
                ch_key, n_at_from, _face, _guide,
                _comps_k, p.get('minion'), p.get('n_nazgul'),
                p.get('include_wk'), p.get('token_type'),
                p.get('character'), p.get('old_face'), p.get('face'),
                p.get('leave_behind'), p.get('forced_companion'),
                p.get('elites_back'), p.get('excess'), _tgt_en)

    def _enemy_comp_at(state, side, tgt):
        """(regulars, elites, leaders) of the ENEMY of `side` at `tgt` -
        the same arithmetic the full key inlines for _tgt_en."""
        _a9e = _armies_lookup(state, str(tgt))
        if _a9e is None:
            return (0, 0, 0)
        _er = _ee = _el = 0
        for u in _a9e.units:
            if _side_of_nation(u.nation) == side:
                continue
            if u.kind == UnitKind.REGULAR:
                _er += 1
            else:
                _ee += 1
        for ld in _a9e.leaders:
            if isinstance(ld, Nazgul):
                if side == Side.FREE_PEOPLES:
                    _el += 1
                continue
            try:
                ls = (Side.FREE_PEOPLES if isinstance(ld, Companion)
                      else Side.SHADOW if type(ld).__name__ == 'Minion'
                      else _side_of_nation(ld.nation))
            except Exception:
                continue
            if ls != side:
                _el += 1
        return (_er, _ee, _el)

    def _encode_action_cached(action, state, side, comp_memo=None):
        _ad = action.__dict__
        if '_lattice' in _ad and comp_memo is not None:
            # memoized lattice Action (shared across the search's
            # nodes): its encoding varies only with the die face, the
            # hobbit_place subtype and the enemy composition at the
            # destination - stash per Action under that small key
            # instead of hashing the 35-element key into the global
            # cache ~800k times per search
            _fk = ('__face__', action.die_idx)
            _face = comp_memo.get(_fk, _ad)   # sentinel: not computed
            if _face is _ad:
                _face = None
                _dl = state.dice.get(side, [])
                if 0 <= action.die_idx < len(_dl)                         and _dl[action.die_idx].current is not None:
                    _face = _dl[action.die_idx].current.name
                comp_memo[_fk] = _face
            _hk = '__hplace__'
            _hp = comp_memo.get(_hk, _ad)
            if _hp is _ad:
                _phk9 = getattr(state, 'pending_hunt', None)
                _hp = (isinstance(_phk9, dict)
                       and _phk9.get('phase') == 'hobbit_place')
                comp_memo[_hk] = _hp
            _tgt9 = action.params.get('to_region')
            _ek9 = ('__enemy__', _tgt9)
            _tgt_en = comp_memo.get(_ek9)
            if _tgt_en is None:
                _tgt_en = _enemy_comp_at(state, side, _tgt9)
                comp_memo[_ek9] = _tgt_en
            _sk = (_face, _hp, _tgt_en)
            _st = _ad.get('_encs')
            if _st is None:
                _st = _ad['_encs'] = {}
            enc = _st.get(_sk)
            if enc is None:
                enc = encode_action(action, state, side)
                if len(_st) > 64:
                    _st.clear()
                _st[_sk] = enc
            return enc
        key = _action_cache_key(action, state, side, comp_memo)
        enc = _ENC_CACHE.get(key)
        if enc is None:
            enc = encode_action(action, state, side)
            if len(_ENC_CACHE) > 500_000:
                _ENC_CACHE.clear()
            _ENC_CACHE[key] = enc
        return enc

    def _region_index_cached(action):
        _ad = action.__dict__
        if '_lattice' in _ad:
            _ri = _ad.get('_ri')
            if _ri is None:
                _ri = _ad['_ri'] = _get_action_region_index(action)
            return _ri
        p = action.params or {}
        ch = p.get('choices')
        ch_key = (tuple(sorted((k, str(v)) for k, v in ch.items()))
                  if isinstance(ch, dict) else None)
        key = (action.action_type, p.get('to'), p.get('to_region'),
               p.get('region'), p.get('from'), ch_key)
        idx = _RIDX_CACHE.get(key)
        if idx is None:
            idx = _get_action_region_index(action)
            if len(_RIDX_CACHE) > 500_000:
                _RIDX_CACHE.clear()
            _RIDX_CACHE[key] = idx
        return idx

    # ---------------------------------------------------------------
    # Value Network with GNN + Card Embeddings + Action Scorer
    # ---------------------------------------------------------------
    # Feature layout (stale-doc fix 2026-07-13; the numbers below are
    # COMPUTED, never hardcode them - get_feature_dim() is the truth):
    #   [0 : 105*NODE_FEAT_DIM]                per-region grid -> GNN
    #   [_CARD_IDX_START : +MAX_HAND_SIZE]     card indices -> Embedding
    #   everything else                        global features
    #   [get_pending_context_start() :]        pending/transient context
    #                                          tail (VALUE_CTX_DROPOUT
    #                                          masks this in BC value
    #                                          training)
    # Current totals (v31): state 3742 dims, board 105x30, action 91
    # (all COMPUTED, never hardcoded - see the NOTE above).
    _CARD_IDX_START = NUM_REGIONS * NODE_FEAT_DIM + 32 + 8 + 2 + 10 + 16 + 8
    _CARD_IDX_END = _CARD_IDX_START + MAX_HAND_SIZE  # derived, no constant

    # SCALE (USER 2026-07-19): architecture registry.  'base' is the
    # historical net (~1.3M params) — every existing checkpoint loads
    # byte-identically under it.  'wide' is the bitter-lesson scale-up
    # (~25x params, ~62MB fp32).  Selected ONCE at import via
    # WOTR_NET_ARCH; a checkpoint must be loaded under the arch it was
    # trained with.
    _NET_ARCHS = {
        'base': dict(gnn_hidden=64, gnn_layers=2, gnn_out=64,
                     card_dim=16, card_proj=32,
                     bb1=512, bb2=256, vh=128, sc1=192, sc2=64),
        'wide': dict(gnn_hidden=256, gnn_layers=3, gnn_out=256,
                     card_dim=48, card_proj=96,
                     bb1=4096, bb2=2048, vh=1024, sc1=1024, sc2=256),
        # SCALE step 2 (USER 2026-07-19): graph-transformer over entity
        # tokens — 1 CLS + 105 region tokens (adjacency-distance
        # attention bias) + 1 global token + 6 card tokens.  No mean
        # pooling: theaters stay separate entities until the CLS
        # readout attends over them.  bb2/gnn_out double as the token
        # dim so score_actions_spatial / value_head / the inference
        # server work unchanged.
        'xformer': dict(kind='transformer', d=384, layers=8, heads=8,
                        gnn_hidden=384, gnn_layers=0, gnn_out=384,
                        card_dim=48, card_proj=96,
                        bb1=0, bb2=384, vh=512, sc1=1024, sc2=256),
    }
    NET_ARCH_NAME = os.environ.get('WOTR_NET_ARCH', 'base')
    NET_ARCH_CFG = _NET_ARCHS[NET_ARCH_NAME]

    _REGION_DIST_CACHE = None

    def _region_distance_matrix():
        """(105, 105) BFS hop distances over the board adjacency, in
        REGION_INDEX order (for the xformer attention bias).  Cached."""
        global _REGION_DIST_CACHE
        if _REGION_DIST_CACHE is not None:
            return _REGION_DIST_CACHE
        from collections import deque
        b9 = board.load_board()
        inv = [None] * NUM_REGIONS
        for key, idx in REGION_INDEX.items():
            if 0 <= idx < NUM_REGIONS and inv[idx] is None:
                inv[idx] = key
        D = np.full((NUM_REGIONS, NUM_REGIONS), 15, dtype=np.int16)
        for i, key in enumerate(inv):
            if key is None:
                continue
            D[i, i] = 0
            seen = {key}
            q = deque([(key, 0)])
            while q:
                cur, dist = q.popleft()
                for adj in b9.adjacent(cur):
                    a = sanitize(adj)
                    if a not in seen:
                        seen.add(a)
                        j = REGION_INDEX.get(a)
                        if j is not None and 0 <= j < NUM_REGIONS:
                            D[i, j] = dist + 1
                        q.append((a, dist + 1))
        _REGION_DIST_CACHE = D
        return D

    class _ScaledTanhHead(nn.Module):
        """VALUE package 2026-08-09 (run-7 autopsy): v = S*tanh(z/S).
        Slope 1 at 0 (Tanh-compatible mid-range, existing checkpoints
        keep their calibration near zero), saturates at +-S.  The
        training currency legitimately spans ~+-2 (opponent-axis
        denial deltas on top of the +-1 graded terminal, 1.224/1.225);
        plain Tanh made 13-36% of TD targets per batch UNREACHABLE ->
        MSE rail-pinning, q-spread collapse (b5: root Q -1.11 flat to
        0.001 by T10).  Parameter-free: state_dicts are unchanged;
        the activation is reconstructed from the checkpoint's
        'head_act' field (see load_value_net)."""

        def __init__(self, scale: float = 2.0):
            super().__init__()
            self.scale = float(scale)

        def forward(self, x):
            return self.scale * torch.tanh(x / self.scale)

    def _value_head_act():
        """Head activation for NEWLY-CONSTRUCTED nets.  Env-gated,
        default = legacy Tanh (byte-identical behavior until the
        package flips the flag); checkpoints saved by the v2 trainer
        carry 'head_act' and load deterministically regardless of
        this env."""
        if os.environ.get('WOTR_VALUE_HEAD_ACT', 'tanh') == 'scaled2':
            return _ScaledTanhHead(2.0)
        return nn.Tanh()

    class WOTRValueNet(nn.Module):
        """GNN + Card Embedding + MLP backbone -> value head + spatial action scorer.

        Architecture:
            Board:     (B, 105, NODE_FEAT_DIM) -> GCN(2 layers) -> mean_pool -> (B, 64)
            Cards:     (B, 6) card indices -> Embedding(97,16) -> masked_mean -> proj -> (B, 32)
            Global:    (B, global_dim) -> passthrough  (computed from input_dim)
            Backbone:  concat(board 64 + cards 32 + global) -> 512 (BN+ReLU+Drop) -> 256 (BN+ReLU+Drop)
            Value:     256 -> 128 -> 1 -> Tanh
            Policy:    spatial action scorer (256 + region_emb 64 + action enc) -> 1,
                       trained by behavioral cloning on human decisions
        """

        def __init__(self, input_dim: int, side: Optional[str] = None):
            super().__init__()
            self.input_dim = input_dim
            self.side = side  # 'fp' or 'sh'
            # SCALE (USER 2026-07-19, bitter-lesson directive): all
            # widths come from WOTR_NET_ARCH ('base' = the historical
            # shapes, every existing checkpoint loads unchanged;
            # 'wide' = ~25x parameters for the scale experiments).
            cfg = NET_ARCH_CFG
            self.arch_name = NET_ARCH_NAME
            self.arch_kind = cfg.get('kind', 'gnn')
            if self.arch_kind == 'transformer':
                self._build_transformer(cfg, input_dim)
                return
            # GNN for spatial features
            self.board_gnn = BoardGNN(node_feat_dim=NODE_FEAT_DIM,
                                      hidden_dim=cfg['gnn_hidden'],
                                      num_layers=cfg['gnn_layers'],
                                      output_dim=cfg['gnn_out'])
            # Card embeddings
            self.card_embedding = nn.Embedding(NUM_CARD_IDS + 1,
                                               cfg['card_dim'],
                                               padding_idx=CARD_PAD_IDX)
            self.card_proj = nn.Sequential(
                nn.Linear(cfg['card_dim'], cfg['card_proj']), nn.ReLU())
            # Global features dimension
            _spatial_dim = NUM_REGIONS * NODE_FEAT_DIM  # 945
            _global_dim = input_dim - _spatial_dim - MAX_HAND_SIZE  # 125
            self._spatial_dim = _spatial_dim
            self._global_dim = _global_dim
            # Backbone: GNN output + card proj + global feats
            backbone_in = cfg['gnn_out'] + cfg['card_proj'] + _global_dim
            self.backbone = nn.Sequential(
                nn.Linear(backbone_in, cfg['bb1']),
                nn.BatchNorm1d(cfg['bb1']),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(cfg['bb1'], cfg['bb2']),
                nn.BatchNorm1d(cfg['bb2']),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.value_head = nn.Sequential(
                nn.Linear(cfg['bb2'], cfg['vh']),
                nn.BatchNorm1d(cfg['vh']),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(cfg['vh'], 1),
                _value_head_act(),
            )
            # Spatial action scorer: uses GNN per-node embeddings for region-aware priors
            # Input: backbone(bb2) + region_node_emb(gnn_out)
            #        + action features (ACTION_ENCODING_DIM - 1, card-idx
            #          column replaced by its card_dim embedding)
            self.spatial_action_scorer = nn.Sequential(
                nn.Linear(cfg['bb2'] + cfg['gnn_out']
                          + (ACTION_ENCODING_DIM - 1) + cfg['card_dim'],
                          cfg['sc1']),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg['sc1'], cfg['sc2']),
                nn.ReLU(),
                nn.Linear(cfg['sc2'], 1),
            )
            # 1.226a RESIDUAL POLICY DELTA: consumes the SAME per-action
            # concat features as the scorer (single path via
            # _action_combined).  FINAL layer weight AND bias are
            # ZERO-INIT so a fresh head is an exact no-op (total logits
            # == BC scorer logits bit-for-bit).  NO BatchNorm/Dropout —
            # 1.224e: BN running stats are buffers that drift in any
            # train-mode forward regardless of requires_grad.
            self.policy_delta = nn.Sequential(
                nn.Linear(cfg['bb2'] + cfg['gnn_out']
                          + (ACTION_ENCODING_DIM - 1) + cfg['card_dim'],
                          128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.policy_delta[2].weight)
            nn.init.zeros_(self.policy_delta[2].bias)

        def _build_transformer(self, cfg, input_dim):
            """Graph-transformer body (arch 'xformer').  Interface-
            compatible with the GNN body: _process_input returns
            (CLS readout (B, d), region token outputs (B, 105, d))."""
            d = cfg['d']
            self._spatial_dim = NUM_REGIONS * NODE_FEAT_DIM
            self._global_dim = (input_dim - self._spatial_dim
                                - MAX_HAND_SIZE)
            self.xf_d = d
            self.xf_heads = cfg['heads']
            self.tok_region = nn.Linear(NODE_FEAT_DIM, d)
            self.tok_global = nn.Linear(self._global_dim, d)
            self.card_embedding = nn.Embedding(NUM_CARD_IDS + 1,
                                               cfg['card_dim'],
                                               padding_idx=CARD_PAD_IDX)
            self.tok_card = nn.Linear(cfg['card_dim'], d)
            self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
            self.region_pos = nn.Parameter(
                torch.zeros(1, NUM_REGIONS, d))
            nn.init.normal_(self.region_pos, std=0.02)
            # attention bias by graph-distance bucket (0..7 hops region
            # to region; bucket 8 = any pair involving a non-region
            # token), learned per head
            self.attn_bias = nn.Parameter(
                torch.zeros(cfg['heads'], 9))
            L = 1 + NUM_REGIONS + 1 + MAX_HAND_SIZE
            buckets = torch.full((L, L), 8, dtype=torch.long)
            dist = torch.from_numpy(
                _region_distance_matrix()).long().clamp(0, 7)
            buckets[1:1 + NUM_REGIONS, 1:1 + NUM_REGIONS] = dist
            self.register_buffer('xf_buckets', buckets)
            self.xf_attn = nn.ModuleList()
            self.xf_ln1 = nn.ModuleList()
            self.xf_ln2 = nn.ModuleList()
            self.xf_mlp = nn.ModuleList()
            for _ in range(cfg['layers']):
                self.xf_attn.append(nn.MultiheadAttention(
                    d, cfg['heads'], dropout=0.1, batch_first=True))
                self.xf_ln1.append(nn.LayerNorm(d))
                self.xf_ln2.append(nn.LayerNorm(d))
                self.xf_mlp.append(nn.Sequential(
                    nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(4 * d, d)))
            self.xf_ln_out = nn.LayerNorm(d)
            self.value_head = nn.Sequential(
                nn.Linear(d, cfg['vh']), nn.LayerNorm(cfg['vh']),
                nn.GELU(), nn.Dropout(0.2), nn.Linear(cfg['vh'], 1),
                _value_head_act())
            self.spatial_action_scorer = nn.Sequential(
                nn.Linear(d + d + (ACTION_ENCODING_DIM - 1)
                          + cfg['card_dim'], cfg['sc1']),
                nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(cfg['sc1'], cfg['sc2']), nn.ReLU(),
                nn.Linear(cfg['sc2'], 1))
            # 1.226a residual policy delta (xformer twin — same rules:
            # zero-init final layer, no BatchNorm/Dropout, shared
            # per-action concat via _action_combined)
            self.policy_delta = nn.Sequential(
                nn.Linear(d + d + (ACTION_ENCODING_DIM - 1)
                          + cfg['card_dim'], 128),
                nn.ReLU(),
                nn.Linear(128, 1))
            nn.init.zeros_(self.policy_delta[2].weight)
            nn.init.zeros_(self.policy_delta[2].bias)

        def _process_input_xf(self, x):
            B = x.shape[0]
            spatial = x[:, :self._spatial_dim].reshape(
                B, NUM_REGIONS, NODE_FEAT_DIM)
            reg_tok = self.tok_region(spatial) + self.region_pos
            card_raw = x[:, _CARD_IDX_START:_CARD_IDX_END]
            card_idx = card_raw.long()
            card_idx = torch.where(
                card_idx < 0,
                torch.tensor(CARD_PAD_IDX, device=x.device), card_idx)
            card_idx = card_idx.clamp(0, CARD_PAD_IDX)
            card_tok = self.tok_card(self.card_embedding(card_idx))
            global_pre = x[:, self._spatial_dim:_CARD_IDX_START]
            global_post = x[:, _CARD_IDX_END:]
            glob_tok = self.tok_global(
                torch.cat([global_pre, global_post], dim=1)
            ).unsqueeze(1)
            toks = torch.cat([self.cls_tok.expand(B, -1, -1), reg_tok,
                              glob_tok, card_tok], dim=1)
            L = toks.shape[1]
            bias = self.attn_bias[:, self.xf_buckets]      # (H, L, L)
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1) \
                .reshape(B * self.xf_heads, L, L)
            h = toks
            for attn, ln1, ln2, mlp in zip(self.xf_attn, self.xf_ln1,
                                           self.xf_ln2, self.xf_mlp):
                hn = ln1(h)
                a, _ = attn(hn, hn, hn, attn_mask=bias,
                            need_weights=False)
                h = h + a
                h = h + mlp(ln2(h))
            h = self.xf_ln_out(h)
            return h[:, 0], h[:, 1:1 + NUM_REGIONS]

        def _embed_action_encs(self, action_encodings):
            """Split raw encodings into (features, card_embedding).

            The last column carries a raw card index for play_event actions
            (-1 otherwise); it is looked up in the shared card_embedding so
            the scorer can tell WHICH card a play action refers to.
            """
            feats = torch.cat(
                [action_encodings[..., :_ACTION_CARD_IDX_DIM],
                 action_encodings[..., _ACTION_CARD_IDX_DIM + 1:]], dim=-1)
            cidx = action_encodings[..., _ACTION_CARD_IDX_DIM].long()
            cidx = torch.where(cidx < 0,
                               torch.tensor(CARD_PAD_IDX, device=cidx.device),
                               cidx).clamp(0, CARD_PAD_IDX)
            return feats, self.card_embedding(cidx)

        def _process_input(self, x):
            """Split flat input into GNN, card, and global components.

            Returns (backbone_out, node_emb):
                backbone_out: (B, 256)
                node_emb: (B, 105, 64) per-node GCN embeddings
            """
            if self.arch_kind == 'transformer':
                return self._process_input_xf(x)
            B = x.shape[0]
            # 1) Spatial features -> GNN
            spatial = x[:, :self._spatial_dim]
            node_feats = spatial.reshape(B, NUM_REGIONS, NODE_FEAT_DIM)
            gnn_out, node_emb = self.board_gnn(node_feats)  # (B,64), (B,105,64)
            # 2) Card indices -> Embedding
            card_raw = x[:, _CARD_IDX_START:_CARD_IDX_END]  # (B, 6)
            card_idx = card_raw.long()
            card_idx = torch.where(card_idx < 0, torch.tensor(CARD_PAD_IDX, device=x.device), card_idx)
            card_idx = card_idx.clamp(0, CARD_PAD_IDX)
            card_emb = self.card_embedding(card_idx)  # (B, 6, 16)
            # Masked mean pool (exclude padding)
            mask = (card_raw >= 0).unsqueeze(-1).float()  # (B, 6, 1)
            card_sum = (card_emb * mask).sum(dim=1)  # (B, 16)
            card_count = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            card_pooled = card_sum / card_count  # (B, 16)
            card_out = self.card_proj(card_pooled)  # (B, 32)
            # 3) Global features (everything except spatial and card indices)
            global_pre = x[:, self._spatial_dim:_CARD_IDX_START]  # (B, 63)
            global_post = x[:, _CARD_IDX_END:]  # (B, 62)
            global_feats = torch.cat([global_pre, global_post], dim=1)  # (B, 125)
            # Concat and run backbone
            combined = torch.cat([gnn_out, card_out, global_feats], dim=1)
            return self.backbone(combined), node_emb

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """Returns value predictions only (backward-compatible)."""
            h, _ = self._process_input(x)
            return self.value_head(h).squeeze(-1)

        def forward_backbone(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """Returns (backbone_out, node_emb) for action scoring."""
            return self._process_input(x)

        def forward_spatial(self, x: 'torch.Tensor'):
            """Returns (value, backbone_out, node_emb) for spatial action scoring."""
            h, node_emb = self._process_input(x)
            value = self.value_head(h).squeeze(-1)
            return value, h, node_emb

        def _action_combined(self, backbone_out, node_emb,
                             action_encodings, region_indices):
            """Shared per-action feature concat — the BC scorer AND the
            1.226a policy delta consume EXACTLY this tensor (single
            construction path; never duplicate it).

            Args:
                backbone_out: (1, 256) backbone features
                node_emb: (1, 105, 64) per-region GCN embeddings
                action_encodings: (N, ACTION_ENCODING_DIM) from encode_action()
                region_indices: (N,) int tensor, -1 = non-regional (use mean-pooled)
            """
            N = action_encodings.shape[0]
            bb = backbone_out.expand(N, -1)  # (N, 256)
            # Vectorized region embedding lookup
            mean_emb = node_emb.mean(dim=1)  # (1, 64)
            # Clamp negative indices to 0 for gathering, fix up after
            safe_idx = region_indices.clamp(min=0)  # (N,)
            # Gather: node_emb[0, safe_idx[i]] for each i
            region_embs = node_emb[0, safe_idx]  # (N, 64)
            # Replace non-regional (index was -1) with mean-pooled
            # (torch.where, not in-place: the delta head trains through
            # this path and in-place indexed writes break autograd)
            non_regional = (region_indices < 0)
            if non_regional.any():
                region_embs = torch.where(
                    non_regional.unsqueeze(-1),
                    mean_emb[0].unsqueeze(0).expand_as(region_embs),
                    region_embs)
            feats, card_emb = self._embed_action_encs(action_encodings)
            return torch.cat([bb, region_embs, feats, card_emb], dim=1)

        def action_scores(self, backbone_out, node_emb,
                          action_encodings, region_indices,
                          with_delta=None):
            """Raw TOTAL per-action logits = BC scorer + 1.226a residual
            policy delta.  with_delta=None reads the WOTR_POLICY_DELTA
            kill-switch ('1' default; '0' -> base-only for A/B).  A
            fresh zero-init delta contributes exact 0.0, so old
            checkpoints behave bit-identically."""
            combined = self._action_combined(
                backbone_out, node_emb, action_encodings, region_indices)
            scores = self.spatial_action_scorer(combined).squeeze(-1)  # (N,)
            use_delta = (_policy_delta_enabled() if with_delta is None
                         else with_delta)
            if use_delta and getattr(self, 'policy_delta', None) is not None:
                scores = scores + self.policy_delta(combined).squeeze(-1)
            return scores

        def score_actions_spatial(self, backbone_out, node_emb, action_encodings, region_indices):
            """Softmax priors over action_scores (scorer + delta).

            Returns:
                priors: (N,) normalized probabilities
            """
            scores = self.action_scores(backbone_out, node_emb,
                                        action_encodings, region_indices)
            priors = torch.softmax(scores, dim=0)
            return priors


# ===================================================================
# Holdout Evaluation
# ===================================================================



def _policy_delta_enabled() -> bool:
    """1.226a kill-switch: WOTR_POLICY_DELTA env, '1' default.  Read
    per call so A/B flips need no re-import (cheap: one getenv per
    node expansion)."""
    return os.environ.get('WOTR_POLICY_DELTA', '1') != '0'


def _load_state_dict_compat(net, sd, path=''):
    """Non-strict state-dict load with EXPLICIT key accounting (1.226a
    checkpoint compat): checkpoints predating the policy_delta head
    load with the fresh ZERO-INIT delta (an exact behavioral no-op);
    any OTHER missing/unexpected key raises loudly.  Returns the list
    of delta keys that were left at fresh init (empty for new-format
    checkpoints)."""
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing
                   if not k.startswith('policy_delta')]
    if bad_missing or unexpected:
        raise RuntimeError(
            'checkpoint %s key mismatch: missing=%s unexpected=%s'
            % (path, bad_missing, list(unexpected)))
    return [k for k in missing if k.startswith('policy_delta')]


def load_value_net(path: str, side: str = None) -> Optional['WOTRValueNet']:
    """Load a CURRENT-GENERATION value net (strict, stale sweep
    2026-07-27): the old MLP->GNN / spatial-migration / very-old-format
    branches targeted deleted pre-v31 generations and silently
    zero-padded obsolete semantics.  Accepts both 'model_state'
    (corpus trainers) and legacy 'model_state_dict' key names; REFUSES
    cross-generation checkpoints (input_dim mismatch) loudly."""
    if not HAS_TORCH:
        return None
    path = _resolve_model_path(path)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=True, map_location=DEVICE)
    sd = ckpt.get('model_state') or ckpt.get('model_state_dict') or ckpt
    current_dim = get_feature_dim()
    ck_dim = ckpt.get('input_dim')
    if ck_dim is not None and int(ck_dim) != int(current_dim):
        raise RuntimeError(
            'checkpoint %s is generation input_dim=%s but the engine '
            'is %d — retrain (run_v31_pipeline)' % (path, ck_dim,
                                                    current_dim))
    ck_adim = ckpt.get('action_dim')
    if ck_adim is not None and int(ck_adim) != int(ACTION_ENCODING_DIM):
        raise RuntimeError(
            'checkpoint %s is generation action_dim=%s but the engine '
            'is %d — retrain (run_v31_pipeline)'
            % (path, ck_adim, ACTION_ENCODING_DIM))
    if side is None:
        side = ckpt.get('side')
    if side is None:
        b = os.path.basename(path).lower()
        side = 'fp' if ('_fp_' in b or b.startswith('fp_')) else (
            'sh' if ('_sh_' in b or b.startswith('sh_')) else None)
    net = WOTRValueNet(current_dim, side=side).to(DEVICE)
    # VALUE package: the head activation is CHECKPOINT-DESCRIBED —
    # 'head_act' absent means legacy Tanh (older checkpoints keep
    # their exact eval behavior whatever WOTR_VALUE_HEAD_ACT says;
    # the env only shapes freshly-constructed nets).  _ScaledTanhHead
    # is parameter-free, so the swap never touches the state_dict.
    _ck_act = ckpt.get('head_act', 'tanh') if isinstance(
        ckpt, dict) else 'tanh'
    net.value_head[-1] = (_ScaledTanhHead(2.0)
                          if _ck_act == 'scaled2' else nn.Tanh())
    # 1.226a: pre-delta checkpoints keep the fresh zero-init delta
    # (exact no-op); any other key mismatch still raises
    _load_state_dict_compat(net, sd, path)
    net.eval()
    # CONFIDENCE AUTO-REGISTRATION (comms\30 sec 0 contract): a
    # checkpoint stamped with conf_* fields reconstructs its
    # mechanism and registers the dispatching provider.  Consumed by
    # the WOTR_CONF_TRACK / cfg.conf_play root-decision path (leaf
    # shrinkage WOTR_CONF_LEAF is retired from the play path);
    # kill-switch WOTR_CONF_AUTOREG=0; never fatal (a seed must load
    # even if the confidence stamp is malformed — the battery catches
    # that).
    try:
        if isinstance(ckpt, dict) and ckpt.get('conf_mech') and \
                os.environ.get('WOTR_CONF_AUTOREG', '1') == '1':
            import confidence as _conf9
            _conf9.register_from_checkpoint(ckpt, side)
    except Exception:
        pass
    return net


def load_dual_nets(fp_path: str = 'wotr_fp_value.pt',
                   sh_path: str = 'wotr_sh_value.pt') -> Optional[Tuple['WOTRValueNet', 'WOTRValueNet']]:
    """Load separate FP and Shadow networks. Returns (fp_net, sh_net) or None."""
    fp_net = load_value_net(fp_path, side='fp')
    sh_net = load_value_net(sh_path, side='sh')
    if fp_net is None or sh_net is None:
        return None
    return fp_net, sh_net


# ===================================================================
# Step 6: Legal Move Enumeration
# ===================================================================

@dataclass
class Action:
    """An atomic action the AI can take."""
    die_idx: int              # Which die to consume (-1 for no-die actions)
    action_type: str          # e.g., 'army_move', 'fellowship_move', etc.
    params: dict = field(default_factory=dict)

    def __repr__(self):
        p = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"Action({self.action_type}, die={self.die_idx}, {p})"


def _get_unused_dice(state: GameState, side: Side) -> List[Tuple[int, ActionDie]]:
    """Return list of (index, die) for unused dice with a rolled face."""
    result = []
    for i, d in enumerate(state.dice.get(side, [])):
        if not d.used and d.current is not None:
            result.append((i, d))
    return result


def _die_faces_for_action(face: ActionDieFace) -> Set[str]:
    """Map die face to the action types it allows."""
    if face == ActionDieFace.CHARACTER:
        return {'character'}
    elif face == ActionDieFace.ARMY:
        return {'army'}
    elif face == ActionDieFace.MUSTER:
        return {'muster'}
    elif face == ActionDieFace.EVENT:
        return {'event'}
    elif face == ActionDieFace.MUSTER_ARMY:
        return {'muster', 'army'}
    elif face == ActionDieFace.WILL_OF_WEST:
        return {'character', 'army', 'muster', 'event', 'special'}
    elif face == ActionDieFace.EYE:
        return set()  # Eyes go to hunt box automatically
    return set()


def _is_shadow_controlled_stronghold(state: GameState, rkey: str) -> bool:
    """Stronghold currently controlled by the Shadow (original Shadow
    stronghold not captured by FP, or FP stronghold captured by Shadow)."""
    meta = board.regions.get(rkey, {})
    if not meta.get('is_stronghold'):
        return False
    nat = _region_nation(rkey)
    if nat is None:
        return False
    if _side_of_nation(nat) == Side.SHADOW:
        return rkey not in state.enemy_controlled.get(Side.FREE_PEOPLES, set())
    return rkey in state.enemy_controlled.get(Side.SHADOW, set())


def _companion_terminus_ok(state: GameState, dest, army_by_san=None) -> bool:
    """May a companion group END its move/separation at *dest*?
    ONE answerer for all six generation sites (USER ruling 2026-08-17,
    complete, the moria specimen): a Shadow-controlled STRONGHOLD is a
    LEGAL terminus — entering it is exactly ending there, because
    stop-on-entry is per movement action (_companion_reachable stops
    expansion at it); leaving on a LATER action is free, so the
    rivendell -> moria -> east emnet -> minas tirith crown route is
    legal across actions.  The W5 B-2 p.20 rule keeps governing
    NON-stronghold regions: a Shadow ARMY may be passed through but
    not ended on."""
    d = sanitize(str(dest))
    if _is_shadow_controlled_stronghold(state, d):
        return True
    if army_by_san is None:
        army_by_san = {sanitize(k): a for k, a in state.armies.items()}
    a = army_by_san.get(d)
    return not (a and any(_side_of_nation(u.nation) == Side.SHADOW
                          for u in a.units))


def is_legal_companion_action(state: GameState, action,
                              cache=None, hunt_place_level=None):
    """THE companion-family VALIDATOR (USER structural order
    2026-08-17, verbatim: "why are there 6 seperate rules for
    companion movement?? why aren't we just deferring everything to
    the engine").  ONE rule body answers every companion
    movement/placement legality question; the menu GENERATORS only
    PROPOSE candidates and every candidate passes through here as
    the sole gate - game rules may not live in generators (EDGE-3
    died of exactly that: a 7th generation site with its own rule
    copy).  Covers companion_move and companion_separate (voluntary
    separation AND the hobbit_place / hunt_placement reuse).
    Returns (ok, reason); *cache* is an optional per-mask dict that
    memoizes reachable-set BFS across candidates.

    The rule organs (_companion_reachable = stop-on-entry path,
    _companion_terminus_ok = terminus ruling 1.249,
    _companion_group_movement = allowance) are private to this
    validator + the explainer's unreachable-why probe."""
    if cache is None:
        cache = {}

    def _reach(cur, movement):
        k = ('reach', cur, movement)
        if k not in cache:
            cache[k] = _companion_reachable(state, cur, movement) \
                if movement > 0 else set()
        return cache[k]

    at = action.action_type
    p = action.params or {}
    d = sanitize(str(p.get('to_region', p.get('to', ''))))
    if not d:
        return False, 'no destination'

    def _place_check(level):
        """Hunt-placed hobbit: placed as if separated from the
        CURRENT fellowship position, movement = progress + level;
        in-place always legal; terminus ruling applies."""
        try:
            cur = sanitize(game_logic.fellowship_region(state))
        except Exception:
            return False, 'fellowship position unknown'
        if d == cur:
            return True, None
        mv = state.fellowship.progress + int(level or 1)
        if d not in _reach(cur, mv):
            return False, _companion_unreachable_why(
                state, cur, d, mv)
        if not _companion_terminus_ok(state, d):
            return False, ('%s holds a Shadow army - companions '
                           'may pass through but not end there' % d)
        return True, None

    if at == 'companion_separate':
        if hunt_place_level is not None:
            # explicit hunt-placement context (the replay-candidate
            # synthesizer validates without an armed pending)
            return _place_check(hunt_place_level)
        ph = getattr(state, 'pending_hunt', None)
        if isinstance(ph, dict) and ph.get('phase') == 'hobbit_place':
            comp = str(p.get('companion', ''))
            if comp != str(ph.get('companion', '')):
                return False, ('%s is not the hobbit being placed'
                               % comp)
            return _place_check(ph.get('level', 1))
        # voluntary separation
        if state.fellowship.mordor_entered:
            return False, ('companions cannot separate on the '
                           'Mordor track')
        names = p.get('companions') or (
            [p['companion']] if p.get('companion') else [])
        want = set(str(n) for n in names)
        comps = [c for c in state.fellowship.companions
                 if c.name in want and c.name != 'Gollum'
                 and c.level > 0]
        if not comps or len(comps) != len(want):
            missing = sorted(want - {c.name for c in comps})
            return False, ('%s not in the Fellowship'
                           % '/'.join(missing or sorted(want)))
        cur = sanitize(str(game_logic.fellowship_region(state)))
        if cur in getattr(state, 'besieged_strongholds', set()):
            # under siege: may only separate INTO the stronghold
            if d == cur:
                return True, None
            return False, ('the fellowship is besieged - companions '
                           'may only separate into %s' % cur)
        if d == cur:
            return True, None          # separating in place
        mv = state.fellowship.progress + max(c.level for c in comps)
        if mv <= 0:
            return False, 'group movement allowance is 0'
        if d not in _reach(cur, mv):
            return False, _companion_unreachable_why(
                state, cur, d, mv)
        if not _companion_terminus_ok(state, d):
            return False, ('%s holds a Shadow army - companions may '
                           'pass through but not end there' % d)
        return True, None
    if at == 'companion_move':
        src = sanitize(str(p.get('from', '')))
        if not src or d == src:
            return False, 'move needs distinct from/to'
        army = None
        for k, a in state.armies.items():
            if sanitize(k) == src:
                army = a
                break
        comps_all = [l for l in (army.leaders if army else [])
                     if isinstance(l, Companion)]
        names = p.get('companions')
        if names:
            wn = set(str(n) for n in names)
            group = [c for c in comps_all if c.name in wn]
            if len(group) != len(wn):
                return False, ('not all of %s stand at %s'
                               % (sorted(wn), src))
        else:
            group = comps_all
        if not group:
            return False, 'no companions at %s' % src
        mv = _companion_group_movement(group)
        if mv <= 0:
            return False, 'group movement allowance is 0'
        if d not in _reach(src, mv):
            return False, _companion_unreachable_why(
                state, src, d, mv)
        if not _companion_terminus_ok(state, d):
            return False, ('%s holds a Shadow army - companions may '
                           'pass through but not end there' % d)
        return True, None
    return False, 'not a companion-family action'


def _companion_reachable(state: GameState, start: str, max_steps: int) -> set:
    """Regions a Companion group can reach (user-confirmed rules):

    - CAN enter regions with Shadow armies and Shadow-controlled
      strongholds — but a Shadow stronghold STOPS movement (enter, not
      pass through)
    - Cannot enter strongholds currently under siege
    - Cannot leave a besieged stronghold (empty result)
    """
    besieged = getattr(state, 'besieged_strongholds', set())
    if start in besieged:
        return set()  # cannot leave a besieged stronghold
    reachable = set()
    frontier = {start}
    visited = {start}
    for _ in range(max_steps):
        nxt = set()
        for r in frontier:
            for adj in state.board.adjacent(r):
                a = sanitize(adj)
                if a in visited:
                    continue
                visited.add(a)
                if a in besieged:
                    continue  # cannot enter a besieged stronghold
                reachable.add(a)
                if _is_shadow_controlled_stronghold(state, a):
                    continue  # may ENTER but movement stops here
                nxt.add(a)
        frontier = nxt
    return reachable


def _companion_group_movement(comps) -> int:
    """Movement allowance of a companion group (max level; Gandalf the
    White moves 4 — but only alone or with a single Hobbit)."""
    max_level = max((c.level for c in comps), default=0)
    names = {c.name for c in comps}
    if 'Gandalf the White' in names:
        others = names - {'Gandalf the White'}
        hobbits = {'Meriadoc', 'Peregrin'}
        if not others or (len(others) == 1 and others <= hobbits):
            return 4
    return max_level


def _get_companion_move_actions(state: GameState, die_idx: int) -> List[Action]:
    """FP Character die: move Companions on the map (rulebook p.20).

    Each group moves up to its movement allowance (highest Level; GtW=4
    alone/with one hobbit). The full-group action moves ALL Companions in
    a region together; PROPER SUBSETS are explicit group actions with
    their own allowances (P3, amendment 1.122). May pass THROUGH
    Shadow-army regions but not END there (_dest_ok filters
    destinations); a Shadow-controlled stronghold is enterable but
    STOPS movement; besieged strongholds are impassable and cannot be
    left (see _companion_reachable).
    """
    actions = []
    army_by_san = {sanitize(k): a for k, a in state.armies.items()}
    # GENERATOR/VALIDATOR SPLIT (USER 2026-08-17): this generator
    # only PROPOSES (reachable universe per group); every candidate
    # passes is_legal_companion_action - the ONE rule body.
    _vc9 = {}

    def _dest_ok(a9):
        return is_legal_companion_action(state, a9, _vc9)[0]

    for src, army in army_by_san.items():
        comps = [l for l in army.leaders if isinstance(l, Companion)]
        if not comps:
            continue
        movement = _companion_group_movement(comps)
        if movement > 0:
            # full group first (legacy params shape - the teacher
            # matcher's (from,to) fallback keeps picking this one)
            for dest in sorted(_companion_reachable(state, src, movement)):
                a9 = Action(die_idx=die_idx,
                            action_type='companion_move',
                            params={'from': src, 'to': dest})
                if _dest_ok(a9):
                    actions.append(a9)
        # P3 (USER ruling 1.112.4): PROPER subsets are explicit group
        # choices - each group moves with ITS OWN allowance (max level;
        # GtW-4 alone/with one hobbit via _companion_group_movement).
        # Bounded: full subset lattice up to 4 companions, singles only
        # beyond that.  CAP justification (task: companion-multigroup):
        # only 7 companions exist, so a stack of >4 SEPARATED companions
        # in one region is essentially never reached; the full lattice
        # is 2**n (n=5 -> 30 proper subsets, n=7 -> 126) times reachable
        # destinations, which makes an uncapped menu explode.  The
        # '__companion_moves__' chain (rulebook p.20, added with this
        # task) composes multi-group moves ACROSS decisions: after any
        # group moves, the pending re-offers subsets of the still-
        # unmoved companions, so partitions of a large stack stay
        # reachable even though a single menu never enumerates them all.
        if len(comps) > 1:
            from itertools import combinations
            # USER 2026-08-22: NO CAP on the legal surface ("why is
            # there a cap at all for anything, I don't want one") -
            # every proper subset is a legal group choice (p.20);
            # the old <=4 lattice / singles-beyond bound hid legal
            # gestures from the human seat and choices from the AI.
            _subs9 = [list(c9) for r9 in range(1, len(comps))
                      for c9 in combinations(comps, r9)]
            # COST (2026-08-22, uncapped lattice): a companion move's
            # legality depends on (allowance, src, dest) - never on
            # WHICH members (the validator reads the group's allowance
            # via _companion_group_movement, the reachable set and the
            # terminus).  Ask the ONE validator once per distinct
            # allowance with a representative group; every subset with
            # that allowance takes its destinations from the answer.
            _legal_by_mv9 = {}
            for sub9 in _subs9:
                mv9 = _companion_group_movement(sub9)
                if mv9 <= 0:
                    continue
                nm9 = sorted(c9.name for c9 in sub9)
                if mv9 not in _legal_by_mv9:
                    ok9 = []
                    for dest in sorted(_companion_reachable(state, src,
                                                            mv9)):
                        a9 = Action(
                            die_idx=die_idx,
                            action_type='companion_move',
                            params={'from': src, 'to': dest,
                                    'companions': nm9})
                        if _dest_ok(a9):
                            ok9.append(dest)
                    _legal_by_mv9[mv9] = ok9
                for dest in _legal_by_mv9[mv9]:
                    actions.append(Action(
                        die_idx=die_idx,
                        action_type='companion_move',
                        params={'from': src, 'to': dest,
                                'companions': list(nm9)}))
    return actions


def _get_mouth_move_actions(state: GameState, die_idx: int) -> List[Action]:
    """USER RULING 2026-07-20 (P2-02): the Mouth of Sauron moves on a
    Character die with RANGE 3 and no other movement restrictions
    (unlike Nazgul he is not barred from FP-army regions).  Before
    this, only true Nazgul figures were enumerated, so a Mouth left
    alone (his army marched off, no nazgul to ride) was stranded
    forever.  Self-play path only — the teacher replays logged
    ride-alongs unchanged."""
    if not getattr(getattr(state, 'mouth_of_sauron', None),
                   'in_play', False):
        return []
    src = None
    for k, a in state.armies.items():
        for ld in a.leaders:
            if (type(ld).__name__ == 'Minion'
                    and 'mouth' in str(getattr(ld, 'name', '')).lower()):
                src = sanitize(k)
                break
        if src is not None:
            break
    if src is None:
        return []
    seen = {src}
    frontier = {src}
    dests = set()
    for _ in range(3):
        nxt = set()
        for r in frontier:
            for adj in state.board.adjacent(r):
                a9 = sanitize(adj)
                if a9 not in seen:
                    seen.add(a9)
                    nxt.add(a9)
        dests.update(nxt)
        frontier = nxt
    return [Action(die_idx=die_idx, action_type='mouth_move',
                   params={'from': src, 'to': d})
            for d in sorted(dests)]


def _get_nazgul_move_actions(state: GameState, die_idx: int) -> List[Action]:
    """Shadow Character die: move Nazgul anywhere on the map (rulebook p.20).

    Full rule (USER 2026-07-11): one Character die moves ANY number of
    Nazgul, each independently.  The menu offers the region group and
    COUNT subsets (nazgul are fungible); on the learned path the
    executor then chains a '__nazgul_moves__' pending so further
    flights (one nazgul per decision) may pick DIFFERENT destinations
    within the same die.  Destinations exclude regions containing Free
    Peoples Army units and FP-controlled Strongholds not under siege.
    """
    actions = []
    army_by_san = {sanitize(k): a for k, a in state.armies.items()}

    def _has_wk(a):
        return any(type(l).__name__ == 'Minion'
                   and 'witch' in str(getattr(l, 'name', '')).lower()
                   for l in a.leaders)

    sources = [k for k, a in army_by_san.items()
               if any(isinstance(l, Nazgul) for l in a.leaders)]
    # The Witch-king moves like a Nazgul (infinite movement). In
    # self-play he is offered as an explicit include_wk variant (and a
    # WK-only region is a valid source). Teacher generation unchanged.
    wk_sources = {k for k, a in army_by_san.items() if _has_wk(a)}
    for k in wk_sources:
        if k not in sources:
            sources.append(k)
    if not sources:
        return actions

    # Restrict destinations to the strategically meaningful set instead of
    # all 105 regions (full enumeration made Shadow's action list ~600 long
    # and dominated MCTS expansion cost). Covers the human uses: hunt
    # support at the fellowship, leadership for armies, siege support, and
    # screening at enemy settlements.
    dests = set()
    try:
        fs = game_logic.fellowship_region(state)
        dests.add(fs)
        dests.update(sanitize(a) for a in state.board.adjacent(fs))
    except Exception:
        pass
    for k, a in army_by_san.items():
        if a.units:  # any army region (join/besiege/screen)
            dests.add(k)
    dests.update(getattr(state, 'besieged_strongholds', set()))
    for rkey, meta in board.regions.items():
        if meta.get('is_stronghold') or meta.get('is_city'):
            dests.add(rkey)

    sh_captured = state.enemy_controlled.get(Side.SHADOW, set())
    besieged = getattr(state, 'besieged_strongholds', set())
    for dest in sorted(dests):
        meta = board.regions.get(dest)
        if meta is None:
            continue
        d_army = army_by_san.get(dest)
        # Cannot end movement in a region with enemy Army units —
        # UNLESS the region is under siege (nazgul join the besiegers;
        # the FP units there are the garrison inside the stronghold)
        if (d_army and dest not in besieged
                and any(_side_of_nation(u.nation) == Side.FREE_PEOPLES
                        for u in d_army.units)):
            continue
        # FP-controlled Stronghold: only if a Shadow army is besieging it
        if meta.get('is_stronghold'):
            nat = _region_nation(dest)
            fp_stronghold = (nat is not None
                             and _side_of_nation(nat) == Side.FREE_PEOPLES
                             and dest not in sh_captured)
            if fp_stronghold and dest not in besieged:
                continue
        for src in sources:
            if src == dest:
                continue
            _src_army = army_by_san.get(src)
            _n_naz9 = (sum(1 for l in _src_army.leaders
                           if isinstance(l, Nazgul))
                       if _src_army is not None else 0)
            if _n_naz9 > 0:
                actions.append(Action(
                    die_idx=die_idx,
                    action_type='nazgul_move',
                    params={'from': src, 'to': dest}
                ))
                # USER 2026-07-11: COUNT subsets (fungible figures)
                for _k9 in range(1, _n_naz9):
                    actions.append(Action(
                        die_idx=die_idx,
                        action_type='nazgul_move',
                        params={'from': src, 'to': dest,
                                'count': _k9}
                    ))
            if src in wk_sources:
                actions.append(Action(
                    die_idx=die_idx,
                    action_type='nazgul_move',
                    params={'from': src, 'to': dest,
                            'include_wk': True}
                ))
                # USER 2026-07-24: ANY combination must be expressible -
                # the Witch-king plus a COUNT subset of the co-located
                # Nazgul (count=0 = the WK flying alone).  The no-count
                # include_wk variant above stays the all+WK move; the
                # executor already honors count+include_wk together.
                for _k9 in range(0, _n_naz9):
                    actions.append(Action(
                        die_idx=die_idx,
                        action_type='nazgul_move',
                        params={'from': src, 'to': dest,
                                'include_wk': True, 'count': _k9}
                    ))
    return actions


def _get_army_move_actions(state: GameState, side: Side, die_idx: int,
                           require_leader: bool = False,
                           exclude_from: Optional[str] = None) -> List[Action]:
    """Enumerate army movement actions for a given die.

    Args:
        require_leader: If True, only armies with leaders/characters can move
                       (required for Character die).
        exclude_from: If set, skip armies from this region (for army die
                     second move — can't move same army twice).
    """
    actions = []

    # Pre-index armies by sanitized key once (the previous inner scan made
    # this function O(armies^2 x neighbors) — the top hot spot in MCTS)
    army_by_san = {sanitize(k): a for k, a in state.armies.items()}
    exclude_san = sanitize(exclude_from) if exclude_from else None
    _bs144 = getattr(state, 'besieged_strongholds', set())
    # 1.144b: certified-teacher escape - the teacher replays LEGAL
    # human moves against an engine state that can carry small
    # besieger-composition drift (stale flags); LOG = ground truth.
    if _bs144 and combat.FINAL_HITS_HOOK is not None:
        _bs144 = set()

    # "A Power too Great" / "The Power of Tom Bombadil" (QA 2026-07-09):
    # the protection blocks Shadow MOVES into the listed regions, not
    # just attacks - the GUI checked this, the AI enumeration/executor
    # never did (validate_army_move had no callers).
    _csmt = None
    if side == Side.SHADOW:
        from game_state import PersistentCardType as _PCT
        _pc = getattr(state, 'persistent_cards', None)
        if _pc is not None and (
                _pc.has_card_type(_PCT.POWER_TOO_GREAT)
                or _pc.has_card_type(_PCT.TOM_BOMBADIL)):
            from persistent_effects import can_shadow_move_to as _csmt

    _preexist = getattr(state, 'pending_army_move_preexist', None)

    for akey, army in state.armies.items():
        # The acting side must have units here (membership test, not
        # units[0] — a besieged region holds BOTH sides' units in one
        # army entry and either side may move its own out)
        if not army.units:
            continue
        own_units = [u for u in army.units
                     if _side_of_nation(u.nation) == side]
        if not own_units:
            continue

        # All nations can move armies (even passive/box-0 nations can move
        # within their own borders). Border-crossing is restricted below.
        # Only the acting side's OWN nations matter for restrictions.
        army_nations = set(u.nation for u in own_units)

        # Character die: army must have a leader or character (Saruman's
        # ability: Isengard Elites count as Leaders while he is in play)
        if require_leader and not game_logic.army_has_leader_figure(
                state, army,
                side=side):
            continue

        # Army die second move: rulebook p.25 forbids moving the same
        # FIGURE twice with one action — units that were ALREADY at the
        # first move's destination did not move and may leave with the
        # second half (verified 2023-12-07_001647 T3: Nurn->Gorgoroth 2R,
        # then the pre-existing Gorgoroth army -> Morannon on the same
        # Army die). Cap the movable pool at the pre-existing counts.
        from_san = sanitize(akey)
        # 1.144 (USER): a besieged garrison may NOT move out of its
        # stronghold - its only offensive option is the sortie
        # (from == to attack).  The BESIEGER may still leave (thinning
        # or lifting the siege).
        if from_san in _bs144 and \
                _stronghold_controller(state, from_san) == side:
            continue
        _cap = None
        if exclude_san and from_san == exclude_san:
            if not _preexist:
                continue  # no snapshot — conservative old behavior
            _cap = _preexist
            if _cap.get('regular', 0) + _cap.get('elite', 0) <= 0:
                continue

        # Rulebook p.33 (iter 38): the non-belligerent restriction applies
        # "if at least one of the units in the Army belongs to a
        # non-belligerent Nation" — cross-border movement into another
        # nation's territory needs ALL unit nations At War (the old
        # ANY-at-war test was inverted for mixed stacks).
        all_at_war = all(state.political.is_at_war(n) for n in army_nations)
        neighbors = state.board.adjacent(from_san)

        for to_san in neighbors:
            # protected-region move block (see _csmt above)
            if _csmt is not None and not _csmt(state, to_san)[0]:
                continue
            # 1.144 (USER): no slipping units through the siege lines
            # - the garrison's side cannot MOVE into its own besieged
            # stronghold region; relief comes only via an adjacent
            # attack on the besiegers.
            if to_san in _bs144 and \
                    _stronghold_controller(state, to_san) == side:
                continue
            # Non-belligerent border restriction (rulebook p.33).  1.213
            # (USER 2026-07-22): legality keys on the MOVING SUBSET's
            # nations, NOT the whole source army — a mixed stack may still
            # send an ALL-AT-WAR detachment across into another nation's
            # territory (2025-02-05 Vayze/ProniX T8: an Elven-only group,
            # Elves always At War, left a Dale stack that also held not-
            # At-War North units, Dale -> Erebor, and was wrongly refused).
            # Voluntary moves only; retreats/advances are exempt (they may
            # cross freely).  For such a foreign crossing, restrict the
            # movable pool to the at-war-nation units.
            _move_pool = own_units
            if not all_at_war:
                dest_nation = _region_nation(to_san)
                if dest_nation is not None:
                    # METROLOGY 2026-07-27 (13 moves / 11 games, two
                    # agents independently): the old `dest_nation not
                    # in army_nations` skip let PASSIVE units piggyback
                    # wholesale into an at-war teammate's home nation
                    # (p.35: an army with ANY not-at-war units may not
                    # enter another Nation's borders).  1.213's
                    # moving-subset principle applied per unit: a unit
                    # may enter nation D's territory only if its nation
                    # is at war OR it is D's own unit (passive units
                    # move freely inside their OWN borders).
                    _move_pool = [u for u in own_units
                                  if state.political.is_at_war(u.nation)
                                  or u.nation == dest_nation]
                    if not _move_pool:
                        continue

            existing = army_by_san.get(to_san)

            # Check if enemy army (would trigger combat — that's an attack
            # action) — UNLESS own units are also there (joining one's own
            # besieging force outside an enemy garrison is a move)
            if existing and existing.units:
                has_enemy = any(_side_of_nation(u.nation) != side
                                for u in existing.units)
                has_own = any(_side_of_nation(u.nation) == side
                              for u in existing.units)
                if has_enemy and not has_own:
                    # This is an attack, not a move
                    continue

            # Split menu (rulebook p.25 "Splitting an Army"): moving any
            # portion of the units is legal. Offer a compact menu — full
            # stack, half, and a single-unit detachment — each validated
            # against the destination stacking limit independently.
            # Stacking counts OWN units only: a besieged region's shared
            # army entry holds the enemy garrison too, which must not
            # count against the moving side's 10-unit limit.
            existing_units = sum(
                1 for u in existing.units
                if _side_of_nation(u.nation) == side) if existing else 0
            n_r = sum(1 for u in _move_pool if u.kind == UnitKind.REGULAR)
            n_e = sum(1 for u in _move_pool if u.kind == UnitKind.ELITE)
            # own units that stay put no matter what this split takes:
            # on a capped second leg the leg-1 arrivals cannot re-move,
            # so they still hold any left-behind leaders (review P1
            # 2026-07-27: the old `< n_total` guard made a leaderless
            # pre-existing garrison impossible to move out WHOLE)
            n_pool = n_r + n_e
            if _cap is not None:
                # second-half move out of the first move's destination:
                # only the pre-existing (unmoved) figures may go
                n_r = min(n_r, int(_cap.get('regular', 0)))
                n_e = min(n_e, int(_cap.get('elite', 0)))
            n_total = n_r + n_e
            if n_total <= 0:
                continue

            take_ok = _cap is None or int(_cap.get('leaders', 0)) > 0
            # FULL composition grid (USER finding 2026-07-08: the old
            # full/half/single menu could not express e.g. 4R+1E out of
            # 7R+1E — the half split's n_e//2 dropped lone elites, human
            # splits were unrepresentable -> extraction dropped them and
            # self-play left elites behind).  Moving any portion is
            # legal (p.25); the composition IS the decision.
            options = []
            for mr in range(n_r + 1):
                for me in range(n_e + 1):
                    if mr + me == 0:
                        continue
                    if take_ok:
                        options.append((mr, me, True))
                    # leaders stay only if units remain to hold them
                    # (and never on Character-die moves: the moving
                    # group must contain the leader figure)
                    if not require_leader and mr + me < n_pool:
                        options.append((mr, me, False))

            seen_opts = set()
            # USER 2026-07-11: Nazgul may be split from an army move -
            # 'n_nazgul' variants let the AI leave some/all behind or
            # take a nazgul-only escort (Nazgul may stand alone; plain
            # leaders keep the take_leaders all-or-none bool + the
            # units-remain guard)
            _mv_naz9 = (sum(1 for l in army.leaders
                            if isinstance(l, Nazgul))
                        if side == Side.SHADOW else 0)
            # QUEUE_2 (2026-08-17): the destination cap is the ONE
            # siege-aware body (menu == executor == card path)
            _dcap9 = game_logic.army_stack_cap(state, to_san, side)
            for mr, me, take_leaders in options:
                if (mr, me, take_leaders) in seen_opts:
                    continue
                seen_opts.add((mr, me, take_leaders))
                if existing_units + mr + me > _dcap9:
                    continue
                actions.append(Action(
                    die_idx=die_idx,
                    action_type='army_move',
                    params={'from': akey, 'to': to_san,
                            'n_regular': mr, 'n_elite': me,
                            'take_leaders': take_leaders,
                            'side': 'fp' if side == Side.FREE_PEOPLES
                            else 'sh'}
                ))
                if _mv_naz9 > 0:
                    for _nn9 in range(_mv_naz9 + 1):
                        # skip the legacy-equivalent variant (all naz
                        # ride when take_leaders, none when not)
                        if _nn9 == (_mv_naz9 if take_leaders else 0):
                            continue
                        actions.append(Action(
                            die_idx=die_idx,
                            action_type='army_move',
                            params={'from': akey, 'to': to_san,
                                    'n_regular': mr, 'n_elite': me,
                                    'take_leaders': take_leaders,
                                    'n_nazgul': _nn9,
                                    'side': 'sh'}
                        ))

    return actions


def _stronghold_controller(state: GameState, rkey: str) -> Optional[Side]:
    """Side controlling the stronghold at rkey (None if no stronghold):
    the native side unless the region was captured (enemy_controlled)."""
    san = sanitize(rkey)
    meta = board.regions.get(san, {})
    if not meta.get('is_stronghold'):
        return None
    nat = _region_nation(san)
    if nat is None:
        return None
    native = _side_of_nation(nat)
    other = Side.SHADOW if native == Side.FREE_PEOPLES else Side.FREE_PEOPLES
    if san in state.enemy_controlled.get(other, set()):
        return other
    return native


def _settlement_controller(state: GameState, rkey: str) -> Optional[Side]:
    """Side controlling the SETTLEMENT (town / city / stronghold) at rkey,
    or None if the region has no settlement (open regions and bare
    fortifications like Osgiliath are never 'controlled').  Native side
    unless captured (enemy_controlled). Used for retreat legality — a unit
    may not retreat into a region holding an enemy-controlled settlement,
    even when empty (rulebook p.30; USER 2026-07-15)."""
    san = sanitize(rkey)
    meta = board.regions.get(san, {})
    if not (meta.get('is_town') or meta.get('is_city')
            or meta.get('is_stronghold')):
        return None
    nat = _region_nation(san)
    if nat is None:
        return None
    native = _side_of_nation(nat)
    other = Side.SHADOW if native == Side.FREE_PEOPLES else Side.FREE_PEOPLES
    if san in state.enemy_controlled.get(other, set()):
        return other
    if san in state.enemy_controlled.get(native, set()):
        # recaptured by native side (shouldn't normally be listed, but be safe)
        return native
    return native


def _get_army_attack_actions(state: GameState, side: Side, die_idx: int,
                              require_leader: bool = False) -> List[Action]:
    """Enumerate army attack actions for a given die.

    Args:
        require_leader: If True, only armies with leaders/characters can attack
                       (required for Character die).
    """
    actions = []
    _attack_index = {sanitize(k): (k, a) for k, a in state.armies.items()}

    for akey, army in state.armies.items():
        if not army.units:
            continue
        # Membership test (not units[0]): a besieged region holds BOTH
        # sides' units in one army entry — the attacker must still be
        # able to act from it
        if not any(_side_of_nation(u.nation) == side for u in army.units):
            continue

        # Rulebook p.33 (iter 38, user finding: SH attacked Lorien with a
        # stack of 8 Sauron + 1 Isengard before Sauron was At War): the
        # non-belligerent restriction applies "if at least one of the
        # units in the Army belongs to a non-belligerent Nation" — an
        # army may ATTACK only when ALL its units' nations are At War.
        # QUEUE_4 (2026-08-17): game_logic.can_attack_with_army(side=)
        # is THE at-war body - menu and GUI share it.
        if not game_logic.can_attack_with_army(state, army,
                                               side=side)[0]:
            continue

        # Character die: army must have a leader or character (Saruman's
        # ability: Isengard Elites count as Leaders while he is in play)
        if require_leader and not game_logic.army_has_leader_figure(
                state, army,
                side=side):
            continue

        from_san = sanitize(akey)
        # Siege assault: attacker and garrison CO-LOCATED in one region
        # (defender withdrew into the stronghold). The assault is an
        # attack with from == to.
        _own_and_enemy_here = (
            any(_side_of_nation(u.nation) == side for u in army.units)
            and any(_side_of_nation(u.nation) != side for u in army.units))
        if _own_and_enemy_here:
            # from == to covers BOTH the besieger's assault and the
            # garrison's SORTIE (p.32) - stamp the acting side so the
            # executor never has to infer it (2026-07-10 sortie fix)
            actions.append(Action(
                die_idx=die_idx,
                action_type='army_attack',
                params={'from': akey, 'to': akey, 'siege': True,
                        'side': 'fp' if side == Side.FREE_PEOPLES
                        else 'sh'}))
        # 1.144 (USER): a besieged garrison's ONLY attack is the
        # sortie appended above - it cannot strike adjacent regions
        # through the siege lines.  (1.144b: learned path only - the
        # teacher follows the log against possible flag drift.)
        if from_san in getattr(state, 'besieged_strongholds', set()) \
                and combat.FINAL_HITS_HOOK is None \
                and _stronghold_controller(state, from_san) == side:
            continue
        neighbors = state.board.adjacent(from_san)

        for to_san in neighbors:
            # Check for enemy army (indexed lookup, not a full scan)
            entry = _attack_index.get(to_san)
            if entry is None:
                continue
            # Persistent cards (A Power too Great / Tom Bombadil) block
            # Shadow attacks into protected regions
            if side == Side.SHADOW:
                from persistent_effects import can_shadow_attack
                ok_atk, _ = can_shadow_attack(state, to_san)
                if not ok_atk:
                    continue
            k2, a2 = entry
            if not a2.units:
                continue
            # Membership test, not units[0]: a shared siege entry lists
            # the garrison first and hid the enemy FIELD army from
            # adjacent attackers (2023-12-06 T5: SH relief attack
            # Gorgoroth -> Morannon against the FP besiegers)
            has_enemy = any(_side_of_nation(u.nation) != side
                            for u in a2.units)
            if not has_enemy:
                continue
            has_own = any(_side_of_nation(u.nation) == side
                          for u in a2.units)
            if has_own:
                # Mixed entry = a siege in progress there.  If the enemy
                # portion is the GARRISON (we are the besiegers), it can
                # only be assaulted from inside the region (the
                # from == to action above) — an adjacent army cannot
                # attack into the stronghold.  If the enemy portion is
                # the besieging field army (OUR garrison is under
                # siege), an adjacent relief attack is legal.
                if _stronghold_controller(state, to_san) != side:
                    continue
            actions.append(Action(
                die_idx=die_idx,
                action_type='army_attack',
                params={'from': akey, 'to': k2,
                        'side': 'fp' if side == Side.FREE_PEOPLES
                        else 'sh'}
            ))

    return actions


def _get_table_kill_actions(state: GameState, side: Side, die_idx: int,
                            die_face) -> List[Action]:
    """1.149 (USER 2026-07-19): deliberate removal of an OPPONENT's
    play-on-table card — the rules existed as GUI-only helpers and were
    never offered to the AI (nor parsed from the corpus).

    SHADOW: any Action Die + discard 1 ARMY-type card + 1 Character
    card from hand (USER 2026-07-19: NOT any Strategy card — Muster
    cards don't pay) kills 'A Power too Great' / 'The Power of Tom
    Bombadil'.
    FP: a Will of the West result alone, OR any die + one Elven Ring,
    kills 'The Palantír of Orthanc'; Denethor's Folly has its own
    any-die alternates (Aragorn/Gandalf in Minas Tirith, P2-03)."""
    from game_state import PersistentCardType as _PCT
    pc = getattr(state, 'persistent_cards', None)
    if pc is None:
        return []
    out = []
    if side == Side.SHADOW:
        # USER 2026-07-19: the payment is one ARMY card (card_type
        # 'Army' — the 11-card SH list: Return to Valinor, Fighting
        # Uruk-hai, Denethor's Folly, Day Without Dawn, Shadows
        # Gather, Shadow Lengthens, Shadow is Moving, Corsairs,
        # Half-orcs, Olog-hai, Hill-trolls) + ANY Character card —
        # NOT any Strategy card (Muster cards don't pay).
        from cards import get_card as _gc149
        hand = state.hands[Side.SHADOW].hand
        has_army = any(
            isinstance(c, str)
            and getattr(_gc149(c), 'card_type', None) == 'Army'
            for c in hand)
        has_chr = any(isinstance(c, str) and '_Character_' in c
                      for c in hand)
        if has_army and has_chr:
            for t in (_PCT.POWER_TOO_GREAT, _PCT.TOM_BOMBADIL):
                card = pc.get_by_type(t)
                if card is not None:
                    out.append(Action(
                        die_idx=die_idx,
                        action_type='discard_table_card',
                        params={'card_id': card.card_id,
                                'card_name': getattr(
                                    card, 'card_name', card.card_id)}))
    else:
        # USER RULING 2026-07-20 (P2-03): Denethor's Folly is removed
        # by a Will of the West die alone, or ANY die while Aragorn /
        # Gandalf (Grey or White) is in Minas Tirith.  (It also
        # auto-discards when the siege of Minas Tirith ends — that
        # half lives in game_logic.cleanup_sieges.)
        folly = pc.get_by_type(_PCT.DENETHORS_FOLLY)
        if folly is not None:
            _fname = getattr(folly, 'card_name', folly.card_id)
            if die_face == ActionDieFace.WILL_OF_WEST:
                out.append(Action(
                    die_idx=die_idx,
                    action_type='discard_table_card',
                    params={'card_id': folly.card_id,
                            'card_name': _fname, 'payment': 'will'}))
            else:
                try:
                    from persistent_effects import \
                        get_denethors_folly_discard_options as _gdf
                    _any_ok = bool(_gdf(state).get('any_die'))
                except Exception:
                    _any_ok = False
                if _any_ok:
                    out.append(Action(
                        die_idx=die_idx,
                        action_type='discard_table_card',
                        params={'card_id': folly.card_id,
                                'card_name': _fname,
                                'payment': 'character'}))
        card = pc.get_by_type(_PCT.PALANTIR_OF_ORTHANC)
        if card is not None:
            if die_face == ActionDieFace.WILL_OF_WEST:
                out.append(Action(
                    die_idx=die_idx,
                    action_type='discard_table_card',
                    params={'card_id': card.card_id,
                            'card_name': getattr(card, 'card_name',
                                                 card.card_id),
                            'payment': 'will'}))
            elif state.fp_elven_rings > 0:
                out.append(Action(
                    die_idx=die_idx,
                    action_type='discard_table_card',
                    params={'card_id': card.card_id,
                            'card_name': getattr(card, 'card_name',
                                                 card.card_id),
                            'payment': 'ring'}))
    return out


def _get_political_advance_actions(state: GameState, side: Side, die_idx: int) -> List[Action]:
    """Enumerate political advancement actions (Muster or Character
    die).  QUEUE_3 (2026-08-17): PROPOSE-only - the rule body is
    game_logic.can_advance_political (shared with the companion-
    ability menu and the executor)."""
    actions = []
    my_nations = FP_NATIONS if side == Side.FREE_PEOPLES else SHADOW_NATIONS
    for nat in my_nations:
        if not game_logic.can_advance_political(
                state, nat, "muster_die")[0]:
            continue
        actions.append(Action(
            die_idx=die_idx,
            action_type='advance_political',
            params={'nation': nat}
        ))
    return actions


# Companions that can advance their nation's political track with ANY
# action die while inside a CITY OR STRONGHOLD of their nation
# (rulebook character abilities; user ruling 2026-07-03 — the old code
# required a stronghold for Legolas/Gimli, see data/rules_amendments.md).
_COMPANION_MUSTER_NATION = {
    'Legolas': Nation.ELVES,
    'Gimli': Nation.DWARVES,
    'Boromir': Nation.GONDOR,
}


def _get_companion_muster_advances(state: GameState, die_idx: int) -> List[Action]:
    """Legolas/Gimli/Boromir on-board ability: any die advances their
    nation's track while they stand in a city or stronghold of their
    nation."""
    actions = []
    seen: Set[Nation] = set()
    for rkey, army in state.armies.items():
        rk = sanitize(rkey)
        for ld in army.leaders:
            if not isinstance(ld, Companion):
                continue
            nat = _COMPANION_MUSTER_NATION.get(ld.name)
            if nat is None or nat in seen:
                continue
            meta = board.regions.get(rk, {})
            if not ((meta.get('is_stronghold') or meta.get('is_city'))
                    and _region_nation(rk) == nat):
                continue
            # A stronghold captured by the Shadow can't be used
            if rk in state.enemy_controlled.get(Side.SHADOW, set()):
                continue
            # QUEUE_3 (2026-08-17): the political core is ONE body -
            # game_logic.can_advance_political (P5 stands: the
            # companion_ability method keeps T&P from gating this
            # menu; the positional predicates above are this
            # ability's own rule, implemented only here)
            if not game_logic.can_advance_political(
                    state, nat, "companion_ability")[0]:
                continue
            seen.add(nat)
            actions.append(Action(
                die_idx=die_idx,
                action_type='advance_political',
                params={'nation': nat, 'companion_ability': ld.name}
            ))
    return actions


def is_legal_muster_action(state: GameState, action, cache=None):
    """THE muster-family VALIDATOR (QUEUE round 2026-08-17, census
    QUEUE_1: the recruit and elite generators carried FIVE copied
    predicates plus a raw captured-set test).  Generators PROPOSE,
    this judges: at-war, own-nation settlement, no dice-muster into
    a besieged stronghold, controller via _settlement_controller
    (the one settlement-control answerer), stacking cap (units
    only), the leader rules (1.140 FP lone leader; p.25 nazgul
    stronghold-only, self-play path - the teacher keeps matching the
    VASSAL app's loose placements, gate-1 regression), and the
    reinforcement pool.  Covers muster_recruit and muster_elite;
    the two-recruits-in-separate-settlements rule is a MULTI-action
    constraint and stays with the pending context (exclude_region).
    Returns (ok, reason); *cache* memoizes the per-mask scans."""
    if cache is None:
        cache = {}
    at = action.action_type
    p = action.params or {}
    if at not in ('muster_recruit', 'muster_elite'):
        return False, 'not a muster-family action'
    nat = p.get('nation')
    typ = 'elite' if at == 'muster_elite' \
        else str(p.get('type', 'regular'))
    rkey = sanitize(str(p.get('region', '')))
    if nat is None or not rkey:
        return False, 'no nation/region'
    side = _side_of_nation(nat)
    if state.political.boxes.get(nat, 0) < 3:
        return False, '%s is not At War' % getattr(nat, 'value', nat)
    if 'settlements' not in cache:
        cache['settlements'] = _get_settlement_regions()
    meta = cache['settlements'].get(rkey)
    if meta is None:
        return False, '%s is not a settlement' % rkey
    if _region_nation(rkey) != nat:
        return False, ('%s is not a %s settlement'
                       % (rkey, getattr(nat, 'value', nat)))
    if rkey in getattr(state, 'besieged_strongholds', set()):
        return False, ('no dice muster into the besieged '
                       'stronghold at %s' % rkey)
    enemy = Side.SHADOW if side == Side.FREE_PEOPLES \
        else Side.FREE_PEOPLES
    if _settlement_controller(state, rkey) == enemy:
        return False, '%s is enemy-controlled' % rkey
    if 'armies' not in cache:
        cache['armies'] = {sanitize(k): a
                           for k, a in state.armies.items()}
    army = cache['armies'].get(rkey)
    if typ in ('regular', 'elite'):
        if (len(army.units) if army else 0) \
                >= game_logic.MAX_ARMY_UNITS:
            return False, ('stacking limit (%d) at %s'
                           % (game_logic.MAX_ARMY_UNITS, rkey))
    if typ == 'leader' and side == Side.FREE_PEOPLES:
        # 1.140: a lone FP leader is removed immediately (p.20)
        if not (army is not None and any(
                _side_of_nation(u.nation) == side
                for u in army.units)):
            return False, ('a lone leader at %s would be removed '
                           '(1.140, p.20)' % rkey)
    if typ == 'leader' and side == Side.SHADOW \
            and not meta.get('is_stronghold') \
            and combat.FINAL_HITS_HOOK is None:
        # p.25: nazgul recruit at Sauron STRONGHOLDS only
        # (self-play; the teacher is log-truth)
        return False, 'nazgul recruit only at strongholds (p.25)'
    if 'deployed' not in cache:
        cache['deployed'] = _count_deployed_all(state)
    if not _can_recruit(state, nat, typ, cache['deployed']):
        return False, ('the %s %s pool is exhausted'
                       % (getattr(nat, 'value', nat), typ))
    return True, None


def _get_recruit_actions(state: GameState, side: Side, die_idx: int,
                         allowed_types: Optional[List[str]] = None,
                         exclude_region: Optional[str] = None) -> List[Action]:
    """Enumerate recruit actions at settlements of At War nations.

    Per rulebook: units can only be placed in settlements belonging to the
    recruiting nation (not any same-side settlement).

    Args:
        allowed_types: List of unit types to generate ('regular', 'leader').
                      Defaults to ['regular', 'leader'] for muster first recruit.
        exclude_region: If set, skip this region (for muster second recruit —
                       two figures must be in separate settlements).
    """
    if allowed_types is None:
        allowed_types = ['regular', 'leader']
    actions = []
    my_nations = FP_NATIONS if side == Side.FREE_PEOPLES else SHADOW_NATIONS
    # GENERATOR/VALIDATOR SPLIT (QUEUE round 2026-08-17): PROPOSE
    # only - is_legal_muster_action is the ONE rule body (at-war /
    # own-settlement / besieged / controller / stacking / leader
    # rules / pool all live THERE).  exclude_region stays here: the
    # two-recruits-in-separate-settlements rule is a MULTI-action
    # constraint of the pending context, not single-action legality.
    _vc9 = {}
    for nat in my_nations:
        for rkey in _get_settlement_regions():
            if exclude_region and sanitize(rkey) ==                     sanitize(exclude_region):
                continue
            for typ in allowed_types:
                a9 = Action(
                    die_idx=die_idx,
                    action_type='muster_recruit',
                    params={'nation': nat, 'region': rkey,
                            'type': typ})
                if is_legal_muster_action(state, a9, _vc9)[0]:
                    actions.append(a9)
    return actions


def _get_muster_elite_actions(state: GameState, side: Side, die_idx: int) -> List[Action]:
    """Enumerate elite recruitment actions (place 1 new elite unit).

    Per rulebook p.26: one of the muster options is placing one Elite unit
    in a friendly settlement of the recruiting nation.
    This consumes the entire die (no second recruit).
    """
    actions = []
    my_nations = FP_NATIONS if side == Side.FREE_PEOPLES else SHADOW_NATIONS
    # GENERATOR/VALIDATOR SPLIT (QUEUE round 2026-08-17): PROPOSE
    # only - is_legal_muster_action is the ONE rule body.
    _vc9 = {}
    for nat in my_nations:
        for rkey in _get_settlement_regions():
            a9 = Action(
                die_idx=die_idx,
                action_type='muster_elite',
                params={'nation': nat, 'region': rkey})
            if is_legal_muster_action(state, a9, _vc9)[0]:
                actions.append(a9)
    return actions


def _get_play_event_for_die(state: GameState, side: Side, die_idx: int,
                            die_category: str,
                            effect_memo: Optional[Dict] = None) -> List[Action]:
    """Enumerate event cards playable with a given die type.

    Args:
        die_category: 'character', 'army', 'muster', or 'event' (event = any card).

    Each typed die plays ONLY its own card type (rulebook p.22); the
    Event/Palantir die plays any type.  (The old docstring claimed a
    character>=army>=muster hierarchy the rules do not have.)
    """
    from cards import get_card
    from event_effects import check_event_requirement

    # Each die plays ONLY its matching card type (per rulebook p.22):
    # "by using an Action die result whose icon matches the symbol
    #  on the upper right corner of the card."
    if die_category == 'event':
        allowed_card_types = {'Character', 'Army', 'Muster'}
    elif die_category == 'character':
        allowed_card_types = {'Character'}
    elif die_category == 'army':
        allowed_card_types = {'Army'}
    elif die_category == 'muster':
        allowed_card_types = {'Muster'}
    else:
        allowed_card_types = set()

    actions = []
    hand = state.hands.get(side, PlayerInfo())
    side_tag = 'fp' if side == Side.FREE_PEOPLES else 'sh'

    for card_id in hand.hand:
        card = get_card(card_id)
        if card and card.card_type in allowed_card_types:
            can_play, _ = check_event_requirement(state, card_id)
            if not can_play:
                continue
            # The dry-run clones the full state — memoize within one
            # enumeration (state is constant), since the same card is
            # checked for both the Event face and its matching face
            if effect_memo is not None:
                has_eff = effect_memo.get(card_id)
                if has_eff is None:
                    has_eff = _event_card_has_effect(state, side, card_id)
                    effect_memo[card_id] = has_eff
            else:
                has_eff = _event_card_has_effect(state, side, card_id)
            if not has_eff:
                continue
            actions.append(Action(
                die_idx=die_idx,
                action_type='play_event',
                params={'card_id': card_id, 'side': side_tag}
            ))

    return actions


def _event_state_signature(state: GameState, side: Side):
    """A dict snapshot of everything an event card could change — used to
    detect whether a dry-run play actually did anything (state delta),
    rather than trusting the handler's effects strings."""
    f = state.fellowship
    armies = tuple(sorted(
        (sanitize(k),
         sum(1 for u in a.units if u.kind == UnitKind.REGULAR),
         sum(1 for u in a.units if u.kind == UnitKind.ELITE),
         len(a.leaders))
        for k, a in state.armies.items()))
    pol = tuple(sorted((n.value, b) for n, b in state.political.boxes.items()))
    act = tuple(sorted((n.value, bool(v))
                       for n, v in state.political.active.items()))
    ec = tuple(sorted(len(state.enemy_controlled.get(s, set()))
                      for s in (Side.FREE_PEOPLES, Side.SHADOW)))
    # QA 2026-07-10: PersistentCardState stores fp_cards/sh_cards (no
    # 'cards' attr) — the old getattr silently read 0 forever, so every
    # "play on the table" card (The Last Battle, A Power too Great, Tom
    # Bombadil...) looked like a no-op and was pruned from self-play.
    pcards = 0
    try:
        _pcs = state.persistent_cards
        pcards = (len(getattr(_pcs, 'fp_cards', []) or [])
                  + len(getattr(_pcs, 'sh_cards', []) or []))
    except Exception:
        pass
    # RED EX-1 (2026-08-17): the signature hashed dice as (count,
    # unused) - BLIND TO FACES - so Mirror of Galadriel's face-change
    # effect compared as a no-op and the prune made the card
    # AI-unplayable in every self-play game ever (0 offers in 162
    # chaos games; apply_action plays it fine).  Same class as the
    # 2026-07-10 pcards sibling.  The audit below adds every
    # dimension the old signature omitted vs full state: FACES,
    # companion/guide IDENTITY, elven RINGS, hand CONTENT, deck
    # lengths, besieged set.
    _dice_sig = tuple(
        (len(state.dice.get(sd, [])),
         sum(1 for d in state.dice.get(sd, []) if not d.used),
         tuple(sorted(str(getattr(d.current, 'name', None))
                      for d in state.dice.get(sd, []) if not d.used)))
        for sd in (Side.FREE_PEOPLES, Side.SHADOW))
    return {
        'dice': _dice_sig,
        'corr': f.corruption, 'prog': f.progress, 'hid': bool(f.hidden),
        'comps': tuple(sorted(str(c.name) for c in f.companions)),
        'guide': str(getattr(f.guide, 'name', None)),
        'mord': f.mordor_entered,
        'mstep': f.mordor_step, 'armies': armies, 'pol': pol, 'act': act,
        'ec': ec,
        'rings': (int(getattr(state, 'fp_elven_rings', 3)),
                  int(getattr(state, 'sh_elven_rings', 0))),
        'hand_fp': tuple(sorted(
            str(c) for c in state.hands[Side.FREE_PEOPLES].hand
            if isinstance(c, str))),
        'hand_sh': tuple(sorted(
            str(c) for c in state.hands[Side.SHADOW].hand
            if isinstance(c, str))),
        'decks': tuple(len(getattr(state, a9, []) or []) for a9 in
                       ('fp_char_deck', 'fp_strat_deck',
                        'sh_char_deck', 'sh_strat_deck')),
        'besieged': tuple(sorted(
            str(x) for x in getattr(state, 'besieged_strongholds',
                                    set()))),
        'pool': len(state.hunt_pool),
        'disc': len(getattr(state, 'hunt_discard', [])),
        'box': len(state.hunt_box), 'pcards': pcards,
        'pco': getattr(state, 'pending_card_option', None) is not None,
        'vpfp': state.vp.free_peoples if hasattr(state, 'vp') else 0,
        'vpsh': state.vp.shadow if hasattr(state, 'vp') else 0,
    }


def _event_card_has_effect(state: GameState, side: Side, card_id: str) -> bool:
    """Dry-run an event card on a cloned state with empty choices.

    The AI cannot supply interactive choices yet, so handlers that require
    them silently do nothing. Playing such a card burns a die for zero
    effect and teaches the net that play_event is a no-op — exclude those
    cards from the legal action set instead.
    """
    # TEACHER MODE (iter 39): never prune — the humans DID play the card
    # and supplied choices; the closed loop replicates them from the
    # logged fx records.  Pruning re-routed teacher plays through the
    # forced-play chain and lost the fx-derived choices (AngryBird
    # canary caught it when the honest no-op reporting fix landed).
    # Optioned cards (QA 2026-07-09, Cirdan's Ships / Paths of the
    # Woses): the empty-choices dry-run below no-ops any handler that
    # needs a region choice, silently pruning legally playable cards
    # from self-play.  If the option spec yields choices, the real play
    # defers into pending_card_option (or auto-picks a single option) -
    # it is NOT a no-op.
    try:
        from cards import get_card as _gc_eff
        from card_options import get_card_options as _gco_eff
        _cn_eff = getattr(_gc_eff(card_id), 'card_name', '') or ''
        if _cn_eff and _gco_eff(state, side, _cn_eff):
            return True
    except Exception:
        pass
    try:
        from event_effects import play_event_card
        import random as _rng_eff
        import game_logic as _gl_eff
        clone = clone_state(state)
        before = _event_state_signature(clone, side)
        # USER RULING 2026-07-18 (amendment 1.174): for a card whose effect
        # includes a die roll (Return to Valinor class), THE ROLL IS THE
        # EFFECT for legality.  Legality must be a DETERMINISTIC function
        # of preconditions (phase, valid targets exist) — never of a roll
        # outcome — or the no-op filter FLICKERS with the RNG (Return to
        # Valinor legal in one enumeration, pruned in the next: the
        # 12-vs-13 flicker) and enumeration silently burns randomness.
        # Dry-run every d6 at its BEST case (all 6s — "cannot possibly
        # matter even on a good roll" still gates illegal: no targets ->
        # no roll -> no delta -> not offered) and REWIND the global RNG so
        # enumeration consumes ZERO randomness.  roll_d6_batch is the sole
        # d6 funnel (hunt rolls via _hunt_roll route through it too), so
        # this one patch covers the class.  The REAL roll still happens
        # exactly once, live, inside play_event_card on ACTUAL play.
        _rng_state = _rng_eff.getstate()
        _orig_roll = _gl_eff.roll_d6_batch
        _orig_pick = _gl_eff.pick_random_companion
        _orig_pcard = _gl_eff.pick_random_card
        _orig_draw = _gl_eff.draw_event_card
        try:
            _gl_eff.roll_d6_batch = lambda n, ctx='', side=None: [6] * n
            # deterministic picks: enumeration must consume ZERO
            # randomness and never reach a patched (client-owned)
            # funnel — a dry-run is not a real draw
            _gl_eff.pick_random_companion = \
                lambda cands, ctx='': cands[0]
            _gl_eff.pick_random_card = lambda cands, ctx='': cands[0]
            # a granted draw IS an effect, but a dry-run must not touch
            # a live draw funnel (WAVE-F: NeedClientCall escaped
            # get_legal_actions through this path) nor real decks: a
            # dummy hand append keeps the delta visible
            _gl_eff.draw_event_card = \
                lambda st9, sd9, dk9: st9.hands[sd9].hand.append(
                    '__dryrun_draw__')
            result = play_event_card(clone, side, card_id)
        finally:
            _gl_eff.roll_d6_batch = _orig_roll
            _gl_eff.pick_random_companion = _orig_pick
            _gl_eff.pick_random_card = _orig_pcard
            _gl_eff.draw_event_card = _orig_draw
            _rng_eff.setstate(_rng_state)
        if not result.get("success"):
            return False
        # USER FIX 2026-07-08: judge by the ACTUAL STATE DELTA, not the
        # handler's effects strings.  Many handlers append "Healed 1
        # Corruption" / "Recruited ..." even when nothing changed (heal
        # at corruption 0; blocked recruit) — the old effects-list check
        # let the net play those useless cards (e.g. There is Another Way
        # T1 at corruption 0 with no Gollum).  Discount the played card's
        # OWN removal from the actor's hand (that is not an "effect").
        after = _event_state_signature(clone, side)
        _hkey = 'hand_fp' if side == Side.FREE_PEOPLES else 'hand_sh'
        after[_hkey] += 1
        return after != before
    except Exception:
        return True  # On dry-run failure, fall back to offering the card


def _get_fellowship_actions(state: GameState, die_idx: int) -> List[Action]:
    """Enumerate fellowship actions (FP only, character die)."""
    actions = []

    # Rulebook p.38: "Once the Fellowship has been revealed, it cannot be
    # moved by the Free Peoples player using a Character Action die until
    # it is Hidden again."
    can_move = state.fellowship.hidden

    # Move fellowship (pre-Mordor)
    if can_move and not state.fellowship.mordor_entered:
        if state.fellowship.progress < 12:
            actions.append(Action(
                die_idx=die_idx,
                action_type='fellowship_move',
                params={}
            ))

    # In Mordor: move on mordor track.  RULING 1.227 (USER 2026-08-02,
    # closes the 2026-07-11 open question): a stop tile ends ONLY THAT
    # ONE MOVEMENT, never the whole turn — the stop's step-retraction
    # stands and further fellowship moves the same turn stay legal
    # (certified GenericPie T7: Shelob stop at L816, second move at
    # L929).  Current behavior confirmed correct.
    if (can_move and state.fellowship.mordor_entered
            and state.fellowship.mordor_step < 5):
        actions.append(Action(
            die_idx=die_idx,
            action_type='fellowship_move',
            params={}
        ))

    # Hide fellowship
    if not state.fellowship.hidden:
        actions.append(Action(
            die_idx=die_idx,
            action_type='fellowship_hide',
            params={}
        ))

    return actions




def _get_declare_actions(state: GameState,
                         allow_in_place: bool = False) -> List[Action]:
    """Enumerate fellowship declaration actions (no die cost).

    Declaration does not consume a die -- it is a free action available
    whenever the fellowship has accumulated progress > 0 and is not yet
    inside Mordor.  Per the rulebook, the pawn may be placed in any region
    at distance UP TO the progress counter (including staying in place).
    The behavioral-cloning-trained policy decides WHEN and WHERE — no
    hand-coded direction filter.

    Returns actions with die_idx=-1 (no die consumed) and action_type
    'fellowship_declare', with params={'region': <destination>}.
    """
    actions = []
    # Only a HIDDEN fellowship may declare (USER 2026-07-09: declaring
    # while revealed was a free hide cheat) — a revealed fellowship
    # must spend a die to hide first.
    if not state.fellowship.hidden:
        return actions
    if state.fellowship.mordor_entered:
        return actions
    # iter 39 (user finding: fellowship dithers at the Mordor doorstep):
    # declaring moves the figure UP TO progress regions — declaring IN
    # PLACE with progress 0 is legal (RAW; it is how a fellowship at
    # Morannon/Minas Morgul ENTERS Mordor from a standstill, and how it
    # heals by declaring in an FP city/stronghold).  allow_in_place is
    # passed by the SELF-PLAY drivers only: the teacher/replay paths keep
    # the historical progress>0 gate (enabling it there regressed the
    # kapcerz canary via the replay-sample enumeration).
    if state.fellowship.progress <= 0 and not allow_in_place:
        return actions
    if state.fellowship.progress < 0:
        return actions

    cur = game_logic.fellowship_region(state)
    progress = state.fellowship.progress

    # All regions within `progress` steps (official: up to, not exactly)
    reachable = {cur}
    frontier = {cur}
    for _ in range(progress):
        frontier = {sanitize(adj)
                    for r in frontier
                    for adj in state.board.adjacent(r)}
        reachable |= frontier

    # USER RULING 2026-07-20b (w5_g2 T9): never DECLARE into Mordor's
    # interior — the pawn declares AT a gateway and enters via the
    # track at the next turn start (1.155). Self-play only (humans
    # never declare into gorgoroth, so the teacher loses nothing).
    reachable -= game_logic._MORDOR_INTERIOR

    # Rarity-audit fix (2026-07-28, skip-in-disguise doctrine — same
    # class as the siege_continue removal): declaring IN PLACE at
    # progress 0 is a literal no-op (pawn stays, progress 0->0, still
    # hidden, no die) UNLESS it has an effect: arming Mordor entry at
    # a gateway (1.155), healing in an unconquered FP city/stronghold
    # with corruption > 0, or ACTIVATING the settlement's passive
    # nation (rulebook p.34, audit A10 — activation has no
    # enemy-control exception).  The corpus has no coverage of the
    # progress-0 declare window (decline twins are gated progress>=1
    # by USER ruling), so the net was blind there and declared-in-place
    # at Rivendell in 81/84 self-play games — a no-op menu entry, not
    # exploration.  Effectful in-place declares stay in the menu.
    if progress == 0 and cur in reachable:
        _keep_ip = cur in ('morannon', 'minas morgul')
        if not _keep_ip:
            _meta9 = board.regions.get(cur, {})
            _is_settle9 = bool(_meta9.get('is_city')
                               or _meta9.get('is_stronghold'))
            if _is_settle9 and game_logic.is_fp_region(cur):
                # heal effect (mirrors declare_fellowship's heal check)
                if (state.fellowship.corruption > 0
                        and cur not in state.enemy_controlled.get(
                            Side.SHADOW, set())):
                    _keep_ip = True
                else:
                    # activation effect (mirrors
                    # _activate_nation_for_fellowship_declaration)
                    from board import fid_to_name as _f2n9
                    _fac9 = _f2n9.get(_meta9.get('parent_id'),
                                      '').lower()
                    _nm9 = {'gondor': Nation.GONDOR,
                            'rohan': Nation.ROHAN,
                            'elves': Nation.ELVES,
                            'dwarves': Nation.DWARVES,
                            'the north': Nation.THE_NORTH,
                            'north': Nation.THE_NORTH}.get(_fac9)
                    if _nm9 is not None and not \
                            state.political.active.get(_nm9, False):
                        _keep_ip = True
        if not _keep_ip:
            reachable.discard(cur)

    for region in sorted(reachable):
        actions.append(Action(
            die_idx=-1,
            action_type='fellowship_declare',
            params={'region': region}
        ))

    return actions


def companion_separate_action_for(state: GameState, die_idx: int,
                                  names, dest) -> Optional[Action]:
    """The NAMED-subset separation action IF legal, else None.  ONE
    implementation of the p.38 group-separation rule (movement =
    progress + the group's highest level; terminus per
    _companion_terminus_ok — USER 2026-08-17: a Shadow stronghold IS a
    legal terminus, stop-on-entry per action; Mordor bars separation),
    shared by the menu's bounded lattice and the live matcher's
    construct-then-verify (USER live 2026-08-17: a 7-companion
    fellowship put pair-separations outside the menu's <=4 explosion
    cap, and the LEGAL gesture refused - legality must not depend on
    menu enumeration)."""
    act, _why = companion_separate_check(state, die_idx, names, dest)
    return act


def companion_separate_check(state: GameState, die_idx: int,
                             names, dest):
    """(action, None) if legal else (None, reason).  DELEGATES to
    is_legal_companion_action - THE one companion-family rule body
    (USER structural order 2026-08-17) - and the reason string feeds
    the live matcher's refusal explainer (P1: refusals must NAME the
    failed precondition)."""
    nl = sorted(str(n) for n in names)
    d = sanitize(str(dest))
    if len(nl) == 1:
        act = Action(die_idx=die_idx,
                     action_type='companion_separate',
                     params={'companion': nl[0], 'to_region': d})
    else:
        act = Action(die_idx=die_idx,
                     action_type='companion_separate',
                     params={'companions': nl, 'to_region': d})
    ok, why = is_legal_companion_action(state, act)
    return (act, None) if ok else (None, why)


def _companion_unreachable_why(state: GameState, cur: str, d: str,
                               movement: int) -> str:
    """Distinguish 'too far' from 'a Shadow stronghold on the way
    stops the move' for the refusal explainer: re-run the walk WITHOUT
    the stop-on-entry rule; if the target becomes reachable, the stop
    rule is what cut the path."""
    import collections as _c9
    dist = {sanitize(cur): 0}
    q = _c9.deque([sanitize(cur)])
    besieged = getattr(state, 'besieged_strongholds', set())
    while q:
        x = q.popleft()
        if dist[x] >= movement:
            continue
        for adj9 in state.board.adjacent(x):
            a9 = sanitize(adj9)
            if a9 in dist or a9 in besieged:
                continue
            dist[a9] = dist[x] + 1
            q.append(a9)
    if d in dist:
        return ('the path passes THROUGH a Shadow stronghold - '
                'entering one ends the move there (enter is legal, '
                'through is not)')
    return ('out of range (group movement %d = progress + highest '
            'level)' % movement)


def _get_companion_separate_actions(state: GameState, die_idx: int) -> List[Action]:
    """Generate companion separation actions (one companion at a time).

    Costs a Character or Will die.  Movement = fellowship progress +
    highest level of separating companion.  Cannot separate in Mordor.
    """
    actions = []
    if state.fellowship.mordor_entered:
        return actions
    # Must keep at least ring-bearers (Gollum stays automatically)
    separable = [c for c in state.fellowship.companions
                 if c.name not in ("Gollum",) and c.level > 0]
    if not separable:
        return actions

    cur = game_logic.fellowship_region(state)
    progress = state.fellowship.progress
    # GENERATOR/VALIDATOR SPLIT (USER 2026-08-17): this menu only
    # PROPOSES (per-group reachable universe + in-place); every
    # candidate passes is_legal_companion_action - the ONE rule body
    # (mordor bar / besieged-only-into-cur / range / stop-on-entry /
    # terminus all live THERE).  Under siege _companion_reachable is
    # empty, so the proposals collapse to in-place and the validator
    # keeps exactly those - the old dedicated besieged branch is
    # absorbed.
    _vc9 = {}

    def _prop9(names, mv, dest):
        if len(names) == 1:
            a9 = Action(die_idx=die_idx,
                        action_type='companion_separate',
                        params={'companion': names[0],
                                'to_region': dest})
        else:
            a9 = Action(die_idx=die_idx,
                        action_type='companion_separate',
                        params={'companions': sorted(names),
                                'to_region': dest})
        if is_legal_companion_action(state, a9, _vc9)[0]:
            actions.append(a9)

    for comp in separable:
        movement = progress + comp.level
        if movement <= 0:
            continue
        reachable = _companion_reachable(state, cur, movement)
        reachable.add(cur)  # separating in place is always allowed
        for dest in sorted(reachable):
            _prop9([comp.name], movement, dest)
    # Rulebook p.38: ONE die separates "one Companion or one GROUP of
    # Companions" — group movement = progress + HIGHEST level in the
    # group (audit A15, 2026-07-26; singles-only burned N dice for N
    # companions in self-play).  One group action (all separable) per
    # destination; encode_action already carries the multi-hot.
    if len(separable) > 1:
        _gmv15 = progress + max(c.level for c in separable)
        if _gmv15 > 0:
            _grch15 = _companion_reachable(state, cur, _gmv15)
            _grch15.add(cur)
            _gnames15 = [c.name for c in separable]
            for dest in sorted(_grch15):
                _prop9(_gnames15, _gmv15, dest)
        # PROPER SUBSETS (USER live session 2026-08-17: the human
        # separated Strider+Boromir out of a larger fellowship and
        # the singles+all-group menu had no matching action - the
        # LEGAL gesture refused).  Rulebook p.38 'one Companion or
        # one GROUP' means ANY group; 1.112.4/1.122 already settled
        # the same question for companion MOVES with the same
        # explosion bound: full subset lattice while the separable
        # set is <= 4, singles only beyond (7 companions max; a
        # subset's movement = progress + ITS OWN highest level).
        # USER 2026-08-22: NO CAP - every proper subset of the
        # separable companions, whatever the fellowship size (the
        # <=4 bound refused the legal Strider+Boromir pair out of a
        # 7-companion fellowship on the real client, twice)
        if len(separable) >= 2:
            from itertools import combinations
            # COST (profiled 2026-08-22 after the cap removal: 693k
            # validator calls inside one 300-sim search, 44 s): a
            # separation's legality depends on (allowance, dest) only
            # - never on WHICH members form the group (the validator's
            # separation branch reads progress + max level, the
            # reachable set and the terminus).  So the validator - still
            # the ONE answerer - is asked once per distinct allowance
            # with a representative group, and every subset with that
            # allowance takes its destination set from the answer.
            _legal_by_mv = {}

            def _dests_for(mv, rep_names):
                if mv not in _legal_by_mv:
                    rch = _companion_reachable(state, cur, mv)
                    rch.add(cur)
                    ok = []
                    for dest in sorted(rch):
                        a9 = Action(die_idx=die_idx,
                                    action_type='companion_separate',
                                    params={'companions': rep_names,
                                            'to_region': dest})
                        if is_legal_companion_action(state, a9, _vc9)[0]:
                            ok.append(dest)
                    _legal_by_mv[mv] = ok
                return _legal_by_mv[mv]
            # LATTICE MEMO: the proper-subset block is a pure function
            # of (die, position, progress, the companions, the legal
            # destinations per allowance) - computed once per distinct
            # context and REUSED across the search's nodes (the same
            # Action objects; nothing mutates an Action).  Before this
            # every FP expansion rebuilt ~3,000 Action objects.
            _mvs = sorted(set(progress + max(c.level for c in _sub)
                              for _r in range(2, len(separable))
                              for _sub in combinations(separable, _r)))
            _dest_sig = tuple((mv, tuple(_dests_for(
                mv, sorted(c.name for c in separable)))) for mv in _mvs
                if mv > 0)
            _lkey = (die_idx, cur, progress,
                     tuple(sorted(c.name for c in separable)), _dest_sig)
            _block = _SEP_LATTICE_MEMO.get(_lkey)
            if _block is None:
                _block = []
                for _r16 in range(2, len(separable)):
                    for _sub16 in combinations(separable, _r16):
                        _mv16 = progress + max(c.level for c in _sub16)
                        if _mv16 <= 0:
                            continue
                        _nm16 = sorted(c.name for c in _sub16)
                        for dest in _dests_for(_mv16, _nm16):
                            _a16 = Action(
                                die_idx=die_idx,
                                action_type='companion_separate',
                                params={'companions': list(_nm16),
                                        'to_region': dest})
                            _a16._lattice = True   # memoized: may stash
                            _block.append(_a16)
                _bdests = sorted(set(
                    a16.params['to_region'] for a16 in _block))
                for a16 in _block:
                    a16._block_key = _lkey
                    a16._block_dests = _bdests
                if len(_SEP_LATTICE_MEMO) > 256:
                    _SEP_LATTICE_MEMO.clear()
                _SEP_LATTICE_MEMO[_lkey] = _block
            actions.extend(_block)
    return actions


# proper-subset separation blocks, keyed by their full context (see
# _get_companion_separate_actions); bounded, process-wide
_SEP_LATTICE_MEMO = {}


_WIPE_GUARD_PHASES = (
    'attacker_combat_card', 'defender_combat_card',
    'combat_round_result', 'attacker_continue', 'defender_retreat',
    'defender_siege_choice', 'attacker_siege_decision')


def _pending_battle_wiped(state: GameState, pc: dict):
    """(atk_dead, def_dead) when the pending battle's entry has lost
    a whole side (the CHAOS-1 stale-siege class), else None.
    Read-only - shared by the menu guard (mutation-free) and the
    executor's auto-finish."""
    try:
        _ar = pc['attacker_region']
        _dr = pc['defender_region']
        _as = pc['attacker_side']
        _ds = pc['defender_side']
        ad = not _alloc_side_units(state, _ar, _as)
        dd = not _alloc_side_units(state, _dr, _ds)
    except Exception:
        return None
    return (ad, dd) if (ad or dd) else None


def _get_pending_combat_actions(state: GameState) -> List[Action]:
    """Generate actions for multi-phase combat (siege/retreat).

    Dispatches based on state.pending_combat['phase'].
    """
    pc = state.pending_combat
    phase = pc['phase']
    actions = []

    # K1: hidden commits are being revealed by the client - nobody may
    # act (a 'pass' here would dissolve the battle)
    if phase == _CC_REVEAL_PHASE:
        return []

    # CHAOS-1 (RED 2026-08-17, chaos seed 95376): a pending battle
    # whose entry lost ONE SIDE ENTIRELY outside the round machinery
    # (the stale-siege class: besieger vacated/died elsewhere while
    # the flag stayed) must not build card menus - setup_combat
    # raises 'missing attacker units' and every menu degrades with a
    # [STATE-INVARIANT].  MUTATION-FREE here (generators never
    # mutate): offer the decline; the EXECUTOR's twin guard finishes
    # the battle as a wipe.  Decision windows that legally hold
    # one-sided compositions (casualty allocation, victor_advance,
    # post-casualty effects) are exempt.
    if phase in _WIPE_GUARD_PHASES:
        _wg9 = _pending_battle_wiped(state, pc)
        if _wg9 is not None:
            _tag9 = ('fp' if pc.get('deciding_side')
                     == Side.FREE_PEOPLES else 'sh')
            return [Action(die_idx=-1,
                           action_type='combat_card_none',
                           params={'side': _tag9})]

    if phase == 'attacker_rearguard':
        # USER RULE 2026-07-08: rearguard REMOVED — this phase should be
        # transitioned through at attack time; if reached (legacy state),
        # offer only the no-rearguard continuation.
        actions.append(Action(die_idx=-1, action_type='rearguard_none',
                              params={}))

    elif phase == 'defender_siege_choice':
        # Stronghold defender, before round 1: withdraw into the stronghold
        # (siege) or give field battle. NO adjacent-region retreat pre-round.
        actions.append(Action(die_idx=-1, action_type='retreat_into_siege', params={}))
        actions.append(Action(die_idx=-1, action_type='combat_no_retreat', params={}))

    elif phase == 'attacker_continue':
        # After a fought field round: attacker ceases or continues (p.29)
        actions.append(Action(die_idx=-1, action_type='combat_continue', params={}))
        # Corsairs of Umbar (almanac A10): the attacking Shadow army has
        # already MOVED IN and CANNOT retreat/disengage - while co-located
        # it must fight on (ceasing would leave two opposing armies stacked
        # outside a siege).  Suppress the cease option for such battles.
        if not pc.get('attacker_no_retreat'):
            actions.append(Action(die_idx=-1, action_type='combat_cease', params={}))

    elif phase == 'defender_retreat':
        # Post-round, attacker is continuing: defender may retreat to an
        # adjacent free region, withdraw into a stronghold here, or fight on
        actions.extend(_get_retreat_actions(state, pc, defending=True))
        # Withdrawing into the siege is only for the side that CONTROLS the
        # stronghold (the garrison). A besieger defending a SORTIE cannot
        # retreat into the stronghold it besieges (USER 2026-07-15).
        if not pc.get('is_siege') \
                and _defender_controls_stronghold(state, pc):
            actions.append(Action(die_idx=-1, action_type='retreat_into_siege', params={}))
        # A retreat combat card (Scouts) COMMITS the retreat — playing
        # it and standing was a wasted card + rules violation (USER
        # 2026-07-09 T7 Fords of Isen)
        if not pc.get('_card_retreat_mandatory'):
            actions.append(Action(die_idx=-1, action_type='combat_no_retreat', params={}))

    elif phase == 'victor_advance':
        # Victor chooses the advancing portion (free split, may stay).
        atk_r = pc['attacker_region']
        def_r = pc['defender_region']
        army = _armies_lookup(state, atk_r)
        side_v = pc['attacker_side']
        if army:
            own = [u for u in army.units
                   if _side_of_nation(u.nation) == side_v]
            n_r = sum(1 for u in own if u.kind == UnitKind.REGULAR)
            n_e = sum(1 for u in own if u.kind == UnitKind.ELITE)
            # 1.150: nazgul are an explicit advance dimension (1.131
            # parity) and leaders may advance with ANY portion; plain
            # leaders can never be left standing alone
            n_naz = (sum(1 for l in army.leaders
                         if isinstance(l, Nazgul))
                     if side_v == Side.SHADOW else 0)
            # review P2 (2026-07-27): board minions carry no 'immobile'
            # attr (only the off-board template does), so the old
            # getattr check was dead AND a redundant `or any(...)`
            # clause re-admitted every non-Nazgul anyway.  Saruman is
            # the one immobile figure — exclude him by identity.
            has_pl = any(
                not isinstance(l, Nazgul)
                and not game_logic._is_saruman(l)
                for l in army.leaders)
            _stag = 'fp' if side_v == Side.FREE_PEOPLES else 'sh'
            for mr in range(n_r + 1):
                for me in range(n_e + 1):
                    if mr + me == 0:
                        continue
                    # USER RULING 2026-07-27 (supersedes the same-day
                    # C1 menu cap): the advance MAY overstack — the
                    # relief example: 10 attackers may all advance
                    # into a 5-unit garrison (15) and then TRIM to 10
                    # as the learned decision (excess to
                    # reinforcements), or advance fewer.  The
                    # victor_advance EXECUTOR registers + resolves.
                    full = (mr + me == n_r + n_e)
                    if has_pl:
                        tl_opts = [True] if full else [True, False]
                    else:
                        tl_opts = [full]
                    for nz in range(n_naz + 1):
                        for tl in tl_opts:
                            prm = {'from': atk_r, 'to': def_r,
                                   'n_regular': mr, 'n_elite': me,
                                   'take_leaders': tl, 'side': _stag}
                            if n_naz:
                                prm['n_nazgul'] = nz
                            actions.append(Action(
                                die_idx=-1,
                                action_type='victor_advance',
                                params=prm))
        actions.append(Action(die_idx=-1, action_type='victor_stay',
                              params={}))

    elif phase == 'attacker_siege_decision':
        # USER AUDIT 2026-07-08: paying an attack die and then choosing
        # 'siege_continue' (do nothing) is a die burned for nothing —
        # skip-in-disguise.  In SELF-PLAY the only continuation is the
        # assault; 'maintaining' a siege costs no die (just don't act).
        # The TEACHER keeps siege_continue: logs contain attack drags
        # with no rolls after them, which it consumes gracefully.
        actions.append(Action(die_idx=-1, action_type='siege_assault', params={}))

    elif phase == 'combat_round_result':
        # Siege combat only: attacker decides to press (downgrade an elite)
        # or stop after the round. The co-located siege entry holds BOTH
        # sides' units — only ATTACKER-side elites enable a press.
        _atk_san = sanitize(pc['attacker_region'])
        atk_army = next((a for k, a in state.armies.items()
                         if sanitize(k) == _atk_san), None)
        atk_side_pc = pc.get('attacker_side')
        has_elites = bool(atk_army) and any(
            u.kind == UnitKind.ELITE and (
                atk_side_pc is None
                or _side_of_nation(u.nation) == atk_side_pc)
            for u in atk_army.units)
        if has_elites:
            actions.append(Action(die_idx=-1, action_type='siege_press', params={}))
            actions.append(Action(die_idx=-1, action_type='siege_no_press', params={}))
        else:
            # No elites — siege combat ends (can't press)
            actions.append(Action(die_idx=-1, action_type='siege_no_press', params={}))

    elif phase == 'attacker_card_retreat':
        # W5 G-2 (2026-07-20): attacker committed a 'retreat your
        # army' combat card — the retreat is MANDATORY (2026-07-09
        # ruling: playing it and standing was a wasted card), so only
        # destinations are offered; the generic pass fallback covers
        # the no-legal-destination corner.
        actions.extend(_get_retreat_actions(state, pc, defending=False))

    elif phase in ('attacker_casualty_policy', 'defender_casualty_policy'):
        # Owner chooses casualty allocation for this battle: downgrade
        # elites first (preserve unit count / combat dice) or spend
        # regulars first (preserve elites for sieges and cards)
        actions.append(Action(die_idx=-1,
                              action_type='casualty_preserve_units', params={}))
        actions.append(Action(die_idx=-1,
                              action_type='casualty_preserve_elites', params={}))

    elif phase == 'fp_shine':
        # Gandalf 'shining' (item 11): FP decides, before combat-card
        # declaration, whether Gandalf the White forfeits his Leadership to
        # cancel ALL Nazgul leadership (incl. the Witch-king).
        actions.append(Action(die_idx=-1, action_type='shine', params={}))
        actions.append(Action(die_idx=-1, action_type='shine_none', params={}))

    elif phase in ('attacker_combat_card', 'defender_combat_card'):
        # Per-round combat-card commitment (learned): play one of the
        # playable combat cards in hand, or decline ('nc')
        actions.extend(_get_combat_card_actions(state, pc))

    elif phase == 'casualty_allocation':
        # POST-ROLL allocation: k = elite downgrades absorbing 1 hit
        # each; the rest kills regulars first (USER 2026-07-09 — chosen
        # AFTER the roll, replacing the pre-battle policy coin-flip)
        stage = pc.get('alloc_stage')
        if stage == 'attacker':
            _reg9, _side9 = pc['attacker_region'], pc['attacker_side']
            _hits9 = int(pc.get('atk_hits', 0) or 0)
        else:
            _reg9, _side9 = pc['defender_region'], pc['defender_side']
            _hits9 = int(pc.get('def_hits', 0) or 0)
        _n_e = len(_alloc_side_units(state, _reg9, _side9,
                                     UnitKind.ELITE))
        # k ranges over the elites regardless of replacement supply
        # (K6 / USER ruling 2026-08-18): an elite downgraded with no
        # replacement Regular is ELIMINATED outright, hit absorbed -
        # a legal allocation, so the menu must carry it.  The executor
        # (_apply_allocation) resolves the source per downgrade.
        for k in range(0, min(_hits9, _n_e) + 1):
            # 'count' = k elite-downgrades (the learned DECISION — consumed
            # by the executor + encoder + teacher matcher, must stay k).
            # 'casualties' = total hits actually being allocated: the old
            # trace only showed count=k, which reads as "0 casualties"
            # whenever the owner kills regulars instead of downgrading
            # (USER 2026-07-15 logging bug). Display side reads 'casualties'.
            actions.append(Action(die_idx=-1,
                                  action_type='casualty_allocate',
                                  params={'count': k, 'casualties': _hits9}))

    elif phase == 'siege_trim':
        # USER RULING 2026-07-20: withdrawing >5 units into a siege,
        # the OWNER chooses which excess units return to
        # REINFORCEMENTS (not casualties — untracked removal is a
        # pool return, same doctrine as resolve_overstacks).
        # k = number of ELITES sent back; the rest are regulars
        # (k=0 keeps all elites — the human-observed default,
        # game0 T9 Cirdan's Ships).
        _rsan9 = pc.get('trim_region')
        _dside9 = pc.get('defender_side')
        _gar9 = _alloc_side_units(state, _rsan9, _dside9)
        _exc9 = len(_gar9) - game_logic.MAX_SIEGE_UNITS
        _ne9 = sum(1 for u in _gar9 if u.kind == UnitKind.ELITE)
        for k in range(0, min(max(_exc9, 0), _ne9) + 1):
            actions.append(Action(die_idx=-1,
                                  action_type='siege_trim',
                                  params={'elites_back': k,
                                          'excess': max(_exc9, 0)}))

    elif phase == 'combat_effect_choice':
        # Choice-bearing combat-card effect: one action per legal
        # magnitude (Onslaught: sacrifice K units; Dread and Despair:
        # forfeit K Nazgul leadership; Relentless Assault: K self-hits;
        # Heroic Death: 0=decline / 1=leader / level=companion)
        ec = pc.get('effect_choice') or {}
        cid = ec.get('card_id', '')
        if ec.get('stage') == 'post':
            counts = list(range(0, int(ec.get('max_count', 0) or 0) + 1))
        else:
            counts = list(ec.get('counts') or [0])
        for k in counts:
            actions.append(Action(
                die_idx=-1, action_type='combat_effect_choice',
                params={'card_id': cid, 'count': int(k)}))

    if not actions:
        # Fallback: allow pass to clear stuck state
        actions.append(Action(die_idx=-1, action_type='pass', params={}))

    return actions


def _get_retreat_actions(state: GameState, pc: dict, defending: bool) -> List[Action]:
    """Generate retreat destination actions for combat retreat.

    Returns combat_retreat actions for each valid adjacent region the
    retreating side can move to.
    """
    actions = []
    if defending:
        retreat_region = pc['defender_region']
        retreat_side = pc.get('defender_side', Side.SHADOW)
    else:
        retreat_region = pc['attacker_region']
        retreat_side = pc.get('attacker_side', Side.FREE_PEOPLES)

    _foe = (Side.SHADOW if retreat_side == Side.FREE_PEOPLES
            else Side.FREE_PEOPLES)
    # USER RULING 2026-07-27 (supersedes the same-day C2 menu ban):
    # a retreat MAY overstack the destination — the excess is resolved
    # ONCE THE MOVE ENDS as the learned trim decision (excess to
    # reinforcements, same decision as stronghold overstacking).  The
    # combat_retreat EXECUTOR registers + resolves the overstack.
    # Find adjacent regions the retreating army can move to
    for nb_key in state.board.adjacent(sanitize(retreat_region)):
        nb_army = _armies_lookup(state, nb_key)
        # Can retreat to: empty region, or friendly region (never into a
        # region holding ANY enemy unit)
        if nb_army and nb_army.units:
            if any(_side_of_nation(u.nation) == _foe
                   for u in nb_army.units):
                continue  # Can't retreat into enemy-occupied region
        # USER 2026-07-15: cannot retreat into an enemy-CONTROLLED
        # settlement — captured OR the enemy's native town/city/stronghold
        # — even when empty of units (the old check only caught the
        # enemy_controlled CAPTURED set, so retreats into an uncaptured
        # Shadow stronghold/city/town slipped through).
        if _settlement_controller(state, nb_key) == _foe:
            continue
        actions.append(Action(
            die_idx=-1,
            action_type='combat_retreat',
            params={'to_region': nb_key}
        ))

    return actions


def _get_pending_hunt_actions(state: GameState) -> List[Action]:
    """Generate actions for pending hunt decisions (Gollum reveal / sacrifice).

    Dispatches based on state.pending_hunt['phase'].
    """
    ph = state.pending_hunt
    phase = ph['phase']
    actions = []

    if phase in ('hunt_gollum', 'hunt_sacrifice'):
        # 1.230: game_logic.hunt_decision_menu is the ONE menu
        # implementation (kernel choice-validation reads the same list)
        for at9, pr9 in game_logic.hunt_decision_menu(state):
            actions.append(Action(die_idx=-1, action_type=at9,
                                  params=dict(pr9)))

    elif phase == 'hobbit_place':
        # User-verified rule: a randomly drawn Hobbit hunt casualty is
        # PLACED by FP as if separated: reachable set from the CURRENT
        # fellowship position with movement = progress + level (same
        # machinery as companion separation; reuses companion_separate
        # so encoding/matching already work).
        comp = ph.get('companion', '')
        level = int(ph.get('level', 1) or 1)
        try:
            cur = sanitize(game_logic.fellowship_region(state))
        except Exception:
            cur = None
        if cur is not None:
            # GENERATOR/VALIDATOR SPLIT (USER 2026-08-17): propose
            # the reachable universe; is_legal_companion_action (the
            # armed hobbit_place pending routes it to the placement
            # branch) is the sole rule gate.
            movement = state.fellowship.progress + level
            reachable = _companion_reachable(state, cur, movement) \
                if movement > 0 else set()
            reachable.add(cur)  # placing in place is always allowed
            _vch9 = {}
            for dest in sorted(reachable):
                a9 = Action(
                    die_idx=-1,
                    action_type='companion_separate',
                    params={'companion': comp, 'to_region': dest})
                if is_legal_companion_action(state, a9, _vch9)[0]:
                    actions.append(a9)

    if not actions:
        actions.append(Action(die_idx=-1, action_type='pass', params={}))

    return actions


def _get_elven_ring_actions(state: GameState, side: Side) -> List[Action]:
    """Generate elven ring actions (free action, no die consumed).

    Each ring use changes one unused die to a different face.
    FP: can't change TO Will_of_West.
    Shadow: can't change FROM Eye. CAN change TO Eye (moves to hunt box).
    Only 1 ring per side per turn.
    """
    actions = []
    if side == Side.FREE_PEOPLES:
        if state.fp_elven_rings <= 0 or state.fp_ring_used_this_turn:
            return actions
    else:
        if state.sh_elven_rings <= 0 or state.sh_ring_used_this_turn:
            return actions

    dice_pool = state.dice.get(side, [])
    # All possible faces (excluding constraints per side)
    all_faces = [ActionDieFace.CHARACTER, ActionDieFace.ARMY,
                 ActionDieFace.MUSTER, ActionDieFace.EVENT,
                 ActionDieFace.MUSTER_ARMY]
    if side == Side.SHADOW:
        all_faces.append(ActionDieFace.EYE)
    # F-4 (fuzz seed 9317, 2026-07-20): the old `side != FP` branch
    # appended WILL_OF_WEST to the SHADOW menu — Shadow ringed a die
    # into Will of the West and spent it as a wild. The ring may NEVER
    # produce Will of the West (rulebook: any result except Will), and
    # Will is not a Shadow face at all. No side gets it.

    for i, die in enumerate(dice_pool):
        if die.used or die.current is None:
            continue
        # Shadow cannot change an Eye die
        if side == Side.SHADOW and die.current == ActionDieFace.EYE:
            continue
        for new_face in all_faces:
            if new_face == die.current:
                continue  # No-op change
            actions.append(Action(
                die_idx=-1,
                action_type='elven_ring',
                params={'target_die_idx': i, 'new_face': new_face.value,
                        'old_face': die.current.value,
                        'side': 'fp' if side == Side.FREE_PEOPLES else 'sh'}
            ))

    return actions


def _get_event_actions(state: GameState, side: Side, die_idx: int,
                       effect_memo: Optional[Dict] = None) -> List[Action]:
    """Enumerate event die actions: draw card or play event."""
    actions = []
    # NOTE: drawing at the 6-card hand limit IS legal (rulebook: draw,
    # then immediately discard down) — the discard is a learned decision
    # via pending_discard. The old early-return here blocked legal draws.

    # Draw a card — check actual deck lists (initialized via initialize_decks)
    if side == Side.FREE_PEOPLES:
        fp_char = getattr(state, 'fp_char_deck', [])
        fp_strat = getattr(state, 'fp_strat_deck', [])
        if fp_char:
            actions.append(Action(die_idx=die_idx, action_type='draw_card',
                                  params={'deck': 'character', 'side': 'fp'}))
        if fp_strat:
            actions.append(Action(die_idx=die_idx, action_type='draw_card',
                                  params={'deck': 'strategy', 'side': 'fp'}))
    else:
        sh_char = getattr(state, 'sh_char_deck', [])
        sh_strat = getattr(state, 'sh_strat_deck', [])
        if sh_char:
            actions.append(Action(die_idx=die_idx, action_type='draw_card',
                                  params={'deck': 'character', 'side': 'sh'}))
        if sh_strat:
            actions.append(Action(die_idx=die_idx, action_type='draw_card',
                                  params={'deck': 'strategy', 'side': 'sh'}))

    # Play any event card from hand (event die plays all types; uses the
    # same requirement + no-op filtering as the typed-die path)
    actions.extend(_get_play_event_for_die(state, side, die_idx, 'event',
                                           effect_memo))

    return actions


def _get_special_actions(state: GameState, side: Side, die_idx: int) -> List[Action]:
    """Enumerate special actions available with Will of the West (FP) die."""
    actions = []

    if side == Side.FREE_PEOPLES:
        # Transform Gandalf the White
        can_gtw, _, regions = game_logic.can_transform_gandalf_white(state)
        if can_gtw:
            for r in regions:
                actions.append(Action(
                    die_idx=die_idx,
                    action_type='transform_gandalf',
                    params={'region': r}
                ))

        # Crown Aragorn
        can_crown, _, regions = game_logic.can_crown_aragorn(state)
        if can_crown:
            actions.append(Action(
                die_idx=die_idx,
                action_type='crown_aragorn',
                params={}
            ))

    return actions


def _dedup_regions_sanitized(regions):
    """Dedup a region list by sanitized key (decision-audit 2026-07-08:
    can_muster_* returned 'Mount Gundabad' AND 'mount gundabad' — the
    same region offered twice as distinct actions)."""
    seen, out = set(), []
    for r in regions or []:
        k = sanitize(str(r))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _get_muster_minion_actions(state: GameState, die_idx: int) -> List[Action]:
    """Enumerate Shadow minion mustering actions."""
    actions = []

    can_wk, _, wk_regions = game_logic.can_muster_witch_king(state)
    wk_regions = _dedup_regions_sanitized(wk_regions)
    if can_wk:
        for r in wk_regions:
            actions.append(Action(
                die_idx=die_idx,
                action_type='muster_minion',
                params={'minion': 'Witch King', 'region': r}
            ))

    can_sa, _, sa_regions = game_logic.can_muster_saruman(state)
    sa_regions = _dedup_regions_sanitized(sa_regions)
    if can_sa:
        for r in sa_regions:
            actions.append(Action(
                die_idx=die_idx,
                action_type='muster_minion',
                params={'minion': 'Saruman', 'region': r}
            ))

    can_mo, _, mo_regions = game_logic.can_muster_mouth_of_sauron(state)
    mo_regions = _dedup_regions_sanitized(mo_regions)
    if can_mo:
        for r in mo_regions:
            actions.append(Action(
                die_idx=die_idx,
                action_type='muster_minion',
                params={'minion': 'Mouth of Sauron', 'region': r}
            ))

    return actions


def get_legal_actions(state: GameState, side: Side) -> List[Action]:
    """Enumerate all legal atomic actions for the given side.

    Handles multi-step die actions (Army die double-move, Muster die
    double-recruit) via pending state on the GameState.

    Returns a list of Action objects that can be passed to apply_action().
    """
    # (Declaration and reveal phases removed — using auto-declare/auto-reveal)

    # --- Pending multi-phase states (combat, hunt, guide) ---
    if state.pending_combat is not None:
        return _get_pending_combat_actions(state)

    # guide choice BEFORE hunt options: mid-hunt guide ties (e.g. a
    # sacrifice leaves Merry+Pippin) must resolve first — the new
    # guide's ability determines the remaining hunt options
    if state.pending_guide_change is not None:
        return [Action(die_idx=-1, action_type='choose_guide',
                       params={'companion': name})
                for name in state.pending_guide_change['candidates']]

    if state.pending_hunt is not None:
        return _get_pending_hunt_actions(state)

    # --- Hunt-reveal placement (USER RULING 2026-08-16: LEARNED FP
    # decision).  Reachable only after the reveal's damage fully
    # settled AND open_reveal_placement left a real choice open (the
    # forced cases resolve at the settle point and clear the flag).
    # Branch position mirrors _pending_decision_side EXACTLY. ---
    if getattr(state, '_reveal_place_due', False) \
            and not getattr(state, '_pending_hunt_damage', 0):
        _rp_acts9 = _get_reveal_placement_actions(state)
        if _rp_acts9:
            return _rp_acts9

    # --- Hand-limit discard (learned decision: which card to give up) ---
    _pd = getattr(state, 'pending_discard', None)
    if _pd is not None:
        _d_side = _pd['side']
        if side != _d_side:
            # v9 showcase bug: FP was handed SH's hand-limit discard and
            # determinization made it discard imaginary cards. The
            # decision belongs to the hand's owner — a wrong-side ask is
            # a routing bug upstream; surface it loudly.
            _pco9 = getattr(state, 'pending_card_option', None)
            _pcf9 = getattr(state, 'pending_card_followup', None)
            raise RuntimeError(
                f"pending_discard belongs to {_d_side.name}, "
                f"but {side.name} was asked for legal actions "
                f"(coexisting: card_option="
                f"{(_pco9 or {}).get('card_name')}/"
                f"{(_pco9 or {}).get('side')}, followup="
                f"{(_pcf9 or {}).get('card_id')}/"
                f"{(_pcf9 or {}).get('side')}, "
                f"combat={state.pending_combat is not None}, "
                f"hunt={state.pending_hunt is not None})")
        _d_tag = 'fp' if _d_side == Side.FREE_PEOPLES else 'sh'
        _seen = set()
        _acts = []
        for c in state.hands[_d_side].hand:
            if c not in _seen:
                _seen.add(c)
                _acts.append(Action(die_idx=-1, action_type='discard_card',
                                    params={'card_id': c, 'side': _d_tag}))
        if _acts:
            return _acts
        state.pending_discard = None  # Empty hand — nothing to discard

    # --- Fellowship declaration phase (FP, start of turn) ---
    if getattr(state, '_declare_phase', False):
        _decl = _get_declare_actions(state, allow_in_place=True)
        _decl.append(Action(die_idx=-1, action_type='pass', params={}))
        return _decl

    # --- Eye allocation phase (Shadow pre-roll) ---
    if getattr(state, '_eye_allocation_phase', False):
        # Rulebook p.17 sidebar: the maximum is the number of Companions
        # currently in the Fellowship (Ring-bearers do not count), and
        # 'the Shadow player may always place at least one die in the
        # Hunt Box, even if all the Companions have left'.  At least 1
        # is REQUIRED if the FP retrieved a die from the Hunt Box.
        # 1.134 (USER QA): the old 'p.39 Mordor exception' lifting the
        # cap on the Mordor track was an INVENTION - no such rule
        # exists; TD self-play exploited it (7 eyes vs 3 companions)
        # into a degenerate zero-VP hunt-camp strategy.
        n_companions = len(state.fellowship.companions)
        unused = len([d for d in state.dice.get(Side.SHADOW, []) if not d.used])
        max_eyes = min(max(n_companions, 1), unused)
        min_eyes = 1 if getattr(state, '_fp_used_hunt_box_last_turn', False) else 0
        min_eyes = min(min_eyes, max_eyes)
        return [Action(die_idx=-1, action_type='eye_allocate',
                       params={'count': n})
                for n in range(min_eyes, max_eyes + 1)]

    # --- Card-granted FOLLOW-UP army action (USER audit 2026-07-08:
    # Ringwraiths Are Abroad / Black Captain Commands grant a free
    # move/attack afterwards — 'the whole purpose of the card') ---
    _pcf = getattr(state, 'pending_card_followup', None)
    if _pcf is not None:
        _f_side = Side.FREE_PEOPLES if _pcf['side'] == 'fp' else Side.SHADOW
        if side != _f_side:
            raise RuntimeError(
                f"pending_card_followup belongs to {_f_side.name}, "
                f"but {side.name} was asked for legal actions")
        acts_f = []
        _needs_wk = _pcf['kind'] == 'wk_army'
        from game_state import Nazgul as _NazF
        for _k, _a in state.armies.items():
            if not _a.units:
                continue
            if not any(_side_of_nation(u.nation) == _f_side
                       for u in _a.units):
                continue
            if _needs_wk and not any(
                    game_logic.is_nazgul_figure(l)
                    and not isinstance(l, _NazF)
                    for l in _a.leaders):
                # QA 2026-07-11 (BCC C6): the engine WK is named 'Witch
                # King' - the exact-name gate made the follow-up
                # unreachable; match the WK Minion via the shared helper
                continue
            if not _needs_wk and not any(
                    game_logic.is_nazgul_figure(l) for l in _a.leaders):
                # QA 2026-07-11 (RWA C9): the WK Minion counts too
                continue
            if sanitize(_k) in _pcf.get('exclude', ()):
                continue
            for _mv in _get_army_move_actions(state, _f_side, -1):
                if sanitize(str(_mv.params.get('from'))) == sanitize(_k):
                    _mv.params['card_followup'] = True
                    acts_f.append(_mv)
            if not _pcf.get('moved_any'):
                for _at2 in _get_army_attack_actions(state, _f_side, -1):
                    if sanitize(str(_at2.params.get('from'))) == sanitize(_k):
                        _at2.params['card_followup'] = True
                        _at2.params['followup_attack'] = True
                        acts_f.append(_at2)
        acts_f.append(Action(die_idx=-1, action_type='followup_done',
                             params={}))
        return acts_f

    # --- Pending card-option decision (deferred optioned event card) ---
    _pco = getattr(state, 'pending_card_option', None)
    if _pco is not None:
        _o_side = (Side.FREE_PEOPLES
                   if _pco.get('decider', _pco['side']) == 'fp'
                   else Side.SHADOW)
        if side != _o_side:
            # owner-guard (2026-07-10, same class as the discard guard):
            # the option choice belongs to the card's player — a
            # wrong-side ask means a routing bug upstream, and the
            # wrong net would pick the option / the trajectory sample
            # would be attributed to the wrong side.
            raise RuntimeError(
                f"pending_card_option ({_pco.get('card_name')}) belongs"
                f" to {_o_side.name}, but {side.name} was asked for "
                f"legal actions")
        return [Action(die_idx=-1, action_type='card_option',
                       params={'idx': i, 'card_id': _pco['card_id'],
                               'card_name': _pco['card_name'],
                               'side': _pco['side'], 'choices': opt})
                for i, opt in enumerate(_pco['options'])]

    # --- Pending Army die double-move (second move) ---
    # F-2 guard (USER 2026-07-19): the chain belongs to the side that
    # spent the die — never offer the leftover second half to the
    # opponent (pre-guard, a chain surviving a turn boundary handed
    # Shadow's second army move to Free Peoples).
    if (state.pending_army_move_die is not None
            and state.pending_army_move_side in (None, side)):
        actions = _get_army_move_actions(
            state, side, state.pending_army_move_die,
            exclude_from=state.pending_army_move_from)
        # Can always end the double-move early (pass consumes the die)
        actions.append(Action(
            die_idx=state.pending_army_move_die,
            action_type='pass',
            params={}
        ))
        return actions

    # --- Pending Muster die double-recruit (second recruit) ---
    # F-2 guard: same ownership rule as the army-move chain above.
    if (state.pending_muster_die is not None
            and getattr(state, 'pending_muster_side', None) in (None, side)):
        # Per rulebook p.26: valid combos are 2 regulars, 2 leaders/nazgul,
        # or 1 regular + 1 leader. All three allow any second type.
        # Second figure must be in a DIFFERENT settlement.
        actions = _get_recruit_actions(
            state, side, state.pending_muster_die,
            allowed_types=['regular', 'leader'],
            exclude_region=state.pending_muster_region)
        actions.append(Action(
            die_idx=state.pending_muster_die,
            action_type='pass',
            params={}
        ))
        return actions

    # --- Normal action generation ---
    actions = []
    unused_dice = _get_unused_dice(state, side)

    # Deduplicate: only generate actions once per unique face
    faces_seen: Set[ActionDieFace] = set()
    # Shared across faces: event-card dry-run results (state is constant
    # for the whole enumeration; the Event face and a card's matching
    # face would otherwise dry-run the same card twice)
    effect_memo: Dict = {}

    # A7 (USER ruling 2026-08-19): while an Elven-Ring legality waiver
    # is up, the dice it permits may fund ANY of the four ordinary
    # action categories.  Without this the ring is paid and buys
    # NOTHING - the menu would still refuse the move the human paid for,
    # which is the very symptom (USER live C2).  'special' is excluded
    # on purpose: it is granted only by Will of the West, and the ring
    # may never produce Will of the West.
    import game_logic as _gl7m
    _ring_waived = _gl7m.ring_waiver_active(state, side)
    _WAIVED_CATS = frozenset(('character', 'army', 'muster', 'event'))

    for die_idx, die in unused_dice:
        face = die.current
        if face in faces_seen:
            continue
        faces_seen.add(face)

        allowed = _die_faces_for_action(face)
        if _ring_waived and _gl7m.ring_waiver_permits(side, face):
            allowed = set(allowed) | _WAIVED_CATS

        if 'army' in allowed:
            # Army die: 2 moves (pending) or 1 attack
            actions.extend(_get_army_move_actions(state, side, die_idx))
            actions.extend(_get_army_attack_actions(state, side, die_idx))
            # Army die can play Army event cards
            actions.extend(_get_play_event_for_die(state, side, die_idx,
                                                   'army', effect_memo))

        if 'muster' in allowed:
            # Muster die: advance political, or recruit (2 units), or elite upgrade
            actions.extend(_get_political_advance_actions(state, side, die_idx))
            actions.extend(_get_recruit_actions(state, side, die_idx))
            actions.extend(_get_muster_elite_actions(state, side, die_idx))
            if side == Side.SHADOW:
                actions.extend(_get_muster_minion_actions(state, die_idx))
            # Muster die can play Muster event cards
            actions.extend(_get_play_event_for_die(state, side, die_idx,
                                                   'muster', effect_memo))
            # Mouth of Sauron ability (user ruling 2026-07-03): once per
            # turn, while the Mouth is in play, Shadow may use a plain
            # Muster die as an Army die (army move pair / attack).
            if (side == Side.SHADOW and 'army' not in allowed
                    and state.mouth_of_sauron is not None
                    and state.mouth_of_sauron.in_play
                    and not getattr(state,
                                    'sh_mouth_muster_used_this_turn',
                                    False)):
                mouth_actions = (
                    _get_army_move_actions(state, side, die_idx)
                    + _get_army_attack_actions(state, side, die_idx))
                for a in mouth_actions:
                    a.params['mouth_muster'] = True
                actions.extend(mouth_actions)

        # 1.149: table-card kills ride ANY action die (the Palantír
        # variant self-gates on WotW face / available Elven Ring)
        actions.extend(_get_table_kill_actions(state, side, die_idx,
                                               face))

        # Legolas/Gimli/Boromir: any die musters their nation from its
        # stronghold (skip when a muster die already generated the action)
        if side == Side.FREE_PEOPLES and 'muster' not in allowed:
            actions.extend(_get_companion_muster_advances(state, die_idx))

        if 'character' in allowed:
            if side == Side.FREE_PEOPLES:
                actions.extend(_get_fellowship_actions(state, die_idx))
                actions.extend(_get_companion_separate_actions(state, die_idx))
                # FP Character die: move separated Companions (rulebook p.20)
                actions.extend(_get_companion_move_actions(state, die_idx))
            else:
                # Shadow Character die: move Nazgul anywhere (rulebook p.20)
                actions.extend(_get_nazgul_move_actions(state, die_idx))
                actions.extend(_get_mouth_move_actions(state, die_idx))
            # Character die: move ONE army with leader, or attack with leader
            if 'army' not in allowed:  # Avoid re-enumerating if army already done
                actions.extend(_get_army_move_actions(
                    state, side, die_idx, require_leader=True))
                actions.extend(_get_army_attack_actions(
                    state, side, die_idx, require_leader=True))
            # Character die: play Character/Army/Muster event cards
            if 'event' not in allowed:  # Avoid duplicates
                actions.extend(_get_play_event_for_die(
                    state, side, die_idx, 'character', effect_memo))

        if 'event' in allowed:
            actions.extend(_get_event_actions(state, side, die_idx,
                                              effect_memo))

        if 'special' in allowed:
            actions.extend(_get_special_actions(state, side, die_idx))
            # ('special' is granted only by WILL_OF_WEST, an FP-only
            # face; Shadow minion musters ride the 'muster' branch.)

    # Fellowship declaration removed from action resolution --
    # per rulebook, declaration happens during Fellowship Phase (Phase 2)
    # at the START of a turn, before dice are rolled.  Handled separately
    # in the turn loop via _fellowship_phase_declare().

    # Strider's Guide ability: hide the Fellowship using ANY Action die
    # result (not just Character)
    if (side == Side.FREE_PEOPLES and not state.fellowship.hidden
            and state.fellowship.guide is not None
            and 'Strider' in getattr(state.fellowship.guide, 'name', '')):
        seen_hide_faces = set()
        for die_idx2, die2 in unused_dice:
            if die2.current in (ActionDieFace.CHARACTER,
                                ActionDieFace.WILL_OF_WEST):
                continue  # Character/WoW hide already generated
            if die2.current in seen_hide_faces:
                continue
            seen_hide_faces.add(die2.current)
            actions.append(Action(
                die_idx=die_idx2,
                action_type='fellowship_hide',
                params={'strider_ability': True}
            ))

    # Elven ring (free action, no die consumed — changes an existing die face)
    actions.extend(_get_elven_ring_actions(state, side))

    # USER RULE 2026-07-08: 'skip' (burn a die for nothing) is REMOVED
    # from the action space — humans never kill dice for no reason; any
    # parser detection of a skip is likely a mistake.  (The executor
    # branch remains for legacy traces; nothing generates it.)

    # Action tokens + Dwarven rings (bid compensation; user ruling
    # 2026-07-03 — free actions outside the dice)
    actions.extend(_get_token_actions(state, side))
    actions.extend(_get_dwarven_ring_actions(state, side))

    # Pass: do nothing, don't consume a die — legal when you hold FEWER
    # unused ACTION DICE than the opponent (printed rule counts dice).
    # AMENDMENT 1.223 (USER 2026-07-28): the pass-eligibility comparison
    # is DICE-ONLY.  The old token-inflated totals (2026-07-03 token
    # ruling) wrongly suppressed 313 real human tempo passes in the t138
    # pilot ("token-equalized" misclassified as table courtesy).  Tokens
    # still grant their extra turn-order actions elsewhere; they no
    # longer block a pass.
    other_side = Side.SHADOW if side == Side.FREE_PEOPLES else Side.FREE_PEOPLES
    my_unused = len(unused_dice)
    other_unused = sum(1 for d in state.dice.get(other_side, [])
                       if not d.used and d.current is not None)
    if my_unused > 0 and my_unused < other_unused:
        actions.append(Action(
            die_idx=-1,  # No die consumed
            action_type='pass',
            params={}
        ))

    return actions


def _token_actions_left(state: GameState, side: Side) -> int:
    """Extra turn-order actions from unused action tokens (max 1/turn)."""
    import game_logic
    used = getattr(state, 'fp_token_used_this_turn'
                   if side == Side.FREE_PEOPLES
                   else 'sh_token_used_this_turn', False)
    if used:
        return 0
    if game_logic.has_action_token(state, side, 'muster') or \
            game_logic.has_action_token(state, side, 'draw'):
        return 1
    return 0


def _get_token_actions(state: GameState, side: Side) -> List[Action]:
    """Bid-compensation action tokens as learned free actions
    (die_idx=-1; single-use, max one token use per side per turn)."""
    import game_logic
    if _token_actions_left(state, side) == 0:
        return []
    actions = []
    side_tag = 'fp' if side == Side.FREE_PEOPLES else 'sh'
    if game_logic.has_action_token(state, side, 'muster'):
        # one political advance, equivalent to a muster die
        for pa in _get_political_advance_actions(state, side, -1):
            actions.append(Action(
                die_idx=-1, action_type='use_token',
                params={'token_type': 'muster',
                        'nation': pa.params.get('nation'),
                        'side': side_tag}))
    if game_logic.has_action_token(state, side, 'draw'):
        # draw from EITHER deck — the deck choice is a decision
        for deck, dl in (('character',
                          getattr(state, f'{side_tag}_char_deck', [])),
                         ('strategy',
                          getattr(state, f'{side_tag}_strat_deck', []))):
            if dl:
                actions.append(Action(
                    die_idx=-1, action_type='use_token',
                    params={'token_type': 'draw', 'deck': deck,
                            'side': side_tag}))
    return actions


def _get_dwarven_ring_actions(state: GameState, side: Side) -> List[Action]:
    """Dwarven-ring rerolls as learned free actions (die_idx=-1).

    User ruling 2026-07-03: usable at the START of a turn or not at all;
    reroll one own unused die until a different face-class results."""
    import game_logic
    if not game_logic.can_use_dwarven_ring(state, side):
        return []
    actions = []
    side_tag = 'fp' if side == Side.FREE_PEOPLES else 'sh'
    for i, die in enumerate(state.dice.get(side, [])):
        if die.used or die.current is None:
            continue
        # a die with no valid reroll result is not a legal target
        orig_classes = game_logic._DWARVEN_FACE_CLASSES.get(
            die.current, frozenset())
        if not any(not (game_logic._DWARVEN_FACE_CLASSES.get(
                f, frozenset()) & orig_classes) for f in die.faces):
            continue
        actions.append(Action(
            die_idx=-1, action_type='dwarven_ring',
            params={'die_idx': i, 'old_face': die.current.value,
                    'side': side_tag}))
    return actions


# ===================================================================
# Step 7: Action Application
# ===================================================================

def apply_action(state: GameState, action: Action) -> GameState:
    """Apply an action to a cloned state and return the new state.

    This clones the state, applies the action, and returns the result.
    The original state is not modified.
    """
    s2 = clone_state(state)
    _execute_action(s2, action)
    return s2


def _get_die_face(state: GameState, die_idx: int,
                   side: Optional[Side] = None) -> Optional[ActionDieFace]:
    """Look up the face of a die by index for a specific side.

    die_idx < 0 (free/card-followup actions) has NO die: returns None.
    Python negative indexing would otherwise read the LAST die and a
    card-followup army_move whose side's last die showed Army opened a
    phantom free double-move chain (pending_army_move_die=-1).

    If side is provided, looks up in that side's dice pool only.
    Otherwise falls back to searching both (legacy behavior).
    """
    if die_idx is None or die_idx < 0:
        return None
    if side is not None:
        dice = state.dice.get(side, [])
        if die_idx < len(dice):
            return dice[die_idx].current
        return None
    for s in [Side.FREE_PEOPLES, Side.SHADOW]:
        dice = state.dice.get(s, [])
        if die_idx < len(dice):
            return dice[die_idx].current
    return None


def _defender_controls_stronghold(state: GameState, pc: dict) -> bool:
    """True when the DEFENDER is the side that CONTROLS the stronghold in
    the defender's region — i.e. the garrison, the only side that may
    withdraw into a siege there.  In a SORTIE the defender is the BESIEGER
    (they do not control the stronghold) and must fight a FIELD battle:
    retreat_into_siege must never be offered to them (USER 2026-07-15)."""
    if not pc.get('stronghold_region'):
        return False
    return _stronghold_controller(state, pc['defender_region']) \
        == pc.get('defender_side')


def _post_rearguard_transition(state: GameState, pc: dict) -> None:
    """Route combat after the rearguard decision (official sequence p.29-30).

    - Already-besieged stronghold: attacker chooses assault vs. maintain.
    - Stronghold region the DEFENDER CONTROLS (not yet besieged): the
      defender first chooses to withdraw into the stronghold (siege) or
      give field battle.  A besieger defending a SORTIE does NOT control
      the stronghold — it fights a plain field battle.
    - Plain field battle: the first round is fought immediately — there is
      NO pre-round retreat to an adjacent region.
    """
    if pc.get('is_siege'):
        pc['phase'] = 'attacker_siege_decision'
        pc['deciding_side'] = pc['attacker_side']
    elif _defender_controls_stronghold(state, pc):
        pc['phase'] = 'defender_siege_choice'
        pc['deciding_side'] = pc['defender_side']
    else:
        _run_combat_round(state, pc)


def _is_siege_battle(state: GameState, defender_region: str) -> bool:
    """Check if attacking this region would be a siege.

    Siege if defender is in a stronghold (per board metadata).
    """
    def_key = sanitize(defender_region)
    meta = board.regions.get(def_key, {})
    return meta.get('is_stronghold', False)


def _restore_rearguard(state: GameState, pc: dict) -> None:
    """Restore rearguard units to the attacker's original region after combat.

    Called when combat ends (pending_combat about to be cleared).  The
    rearguard units are added back to whatever army exists in the
    attacker's region, or a new army is created there.
    """
    rearguard_units = pc.get('rearguard_units')
    if not rearguard_units:
        return
    atk_region = pc['attacker_region']
    atk_army = _armies_lookup(state, atk_region)
    if atk_army is None:
        # Army may have been fully destroyed or moved -- create a fresh one
        region_obj = state.board.region(sanitize(atk_region))
        atk_army = Army(region=region_obj, units=[], leaders=[])
        state.armies[atk_region] = atk_army
    atk_army.units.extend(rearguard_units)
    # Also restore any rearguard leaders
    rearguard_leaders = pc.get('rearguard_leaders', [])
    if rearguard_leaders:
        atk_army.leaders.extend(rearguard_leaders)


def _maybe_casualty_policy_phase(state: GameState, pc: dict) -> bool:
    """Seed default casualty policies; never opens a decision phase.

    The up-front per-battle policy pick was superseded by the per-round
    learned casualty ALLOCATION (_enter_casualty_allocation) — both
    sides just get the default here and the real decision happens at
    allocation time.  Always returns False (caller runs the round)."""
    pc.setdefault('atk_casualty_policy', 'preserve_elites')
    pc.setdefault('def_casualty_policy', 'preserve_elites')
    return False


def _maybe_shine_phase(state: GameState, pc: dict) -> bool:
    """USER 2026-07-16 (item 11): before combat-card declaration, if Gandalf
    the White is in the battle and there is Nazgul leadership to cancel, the
    FP player decides whether Gandalf SHINES - forfeiting his Leadership to
    cancel ALL Nazgul leadership (incl. the Witch-king).  Fires once per
    battle; the decision is recorded in pc['gtw_shine']."""
    if pc.get('shine_asked'):
        return False
    # Self-play / learned path only (like the Mithril interpose and the
    # casualty-policy phase): the teacher replays certified battles whose
    # logs carry no 'shine' row, and its combat math relies on the legacy
    # auto-cancel default.  FINAL_HITS_HOOK is set only under the teacher.
    import combat as _cmbsh
    if _cmbsh.FINAL_HITS_HOOK is not None:
        pc['shine_asked'] = True
        return False
    from combat import setup_combat
    try:
        tmp = setup_combat(state, pc['attacker_region'],
                           pc['defender_region'],
                           is_siege=pc.get('is_siege', False),
                           attacker_side=pc.get('attacker_side'))
    except Exception:
        pc['shine_asked'] = True
        return False
    gtw = any(getattr(c, 'can_cancel_nazgul_leadership', False)
              or getattr(c, 'name', '') == 'Gandalf the White'
              for c in tmp.companions_in_battle)
    sh_army = (tmp.attacker_army if tmp.attacker_side == Side.SHADOW
               else tmp.defender_army)
    has_naz = any(game_logic.is_nazgul_figure(l)
                  for l in getattr(sh_army, 'leaders', []) or [])
    if not (gtw and has_naz):
        pc['shine_asked'] = True   # nothing to shine at
        return False
    pc['phase'] = 'fp_shine'
    pc['deciding_side'] = Side.FREE_PEOPLES
    return True


def _maybe_combat_card_phase(state: GameState, pc: dict) -> bool:
    """Enter the per-round combat-card decision phases (attacker first,
    then defender) before the round is fought.

    Returns True if a decision phase was entered (the caller must NOT run
    the combat round yet — the combat_card_play/none executor resumes
    _run_combat_round once both sides have committed). The phases always
    fire; with no playable cards the only legal action is
    'combat_card_none' (a forced decline, still consumed uniformly).
    """
    if 'atk_combat_card' not in pc:
        pc['phase'] = 'attacker_combat_card'
        pc['deciding_side'] = pc['attacker_side']
        return True
    if 'def_combat_card' not in pc:
        if (pc.get('no_def_card_r1_unless_companion')
                and pc.get('rounds_fought', 0) == 0):
            _da9 = _armies_lookup(state, pc['defender_region'])
            if not (_da9 and any(isinstance(l, Companion)
                                 for l in _da9.leaders)):
                # P2 (Grond / Fighting Uruk-hai): FP cannot play a
                # Combat card in round 1 unless a Companion is in the
                # battle - a forced decline, not a decision
                pc['def_combat_card'] = None
                return False
        pc['phase'] = 'defender_combat_card'
        pc['deciding_side'] = pc['defender_side']
        return True
    return False


def _commit_combat_card(state: GameState, pc: dict, role: str,
                        cid: 'Optional[str]', from_table: bool = False
                        ) -> None:
    """ONE commit body for a combat-card decision of `role` ('atk' |
    'def'): the card leaves the hand (or the table for the on-table
    Balrog, 1.149b) to the face-up pile (CL-217), the Witch-king
    Sorcerer redraw is queued for a round-1 Shadow play (1.152), and
    pc[role + '_combat_card'] records the commitment (None = declined).
    Shared by the combat_card_play / combat_card_none executor and the
    live reveal of a hidden commit (reveal_combat_card)."""
    cc_side = pc['attacker_side'] if role == 'atk' else pc['defender_side']
    if cid is not None and from_table:
        # the on-table Balrog of Moria used as a combat card - pay by
        # discarding the TABLE card, never a hand card
        _pcb9 = getattr(state, 'persistent_cards', None)
        if _pcb9 is not None:
            _rm9 = _pcb9.remove_card(cid)
            from persistent_effects import discard_table_card
            discard_table_card(state, _rm9)
    elif cid is not None:
        hand = state.hands[cc_side].hand
        if cid in hand:
            hand.remove(cid)
        else:
            # the player held the card under a placeholder: tracked-hand
            # drift ('_unknown_*') or a live human-hand placeholder of
            # the card's deck (HU_*); the placeholder leaves instead
            _deck9 = game_logic.card_deck_of(cid)
            _ph9 = next((i for i, item in enumerate(hand)
                         if isinstance(item, str)
                         and (item.startswith('_unknown_')
                              or (item.startswith('HU_')
                                  and game_logic.card_deck_of(item)
                                  == _deck9))), None)
            if _ph9 is not None:
                hand.pop(_ph9)
        _discard_played_combat_card(state, cid)
    # Witch-king 'Sorcerer': when the WK is in the battle and SHADOW
    # plays a combat card in the FIRST round, Shadow draws an event card
    # of the same deck type after the round (1.152 ordering).
    if cid is not None and pc.get('rounds_fought', 0) == 0 \
            and cc_side == Side.SHADOW:
        _wk_here = False
        for _reg9 in (pc.get('attacker_region'), pc.get('defender_region')):
            _ar9 = _armies_lookup(state, str(_reg9))
            if _ar9 and any(
                    type(l).__name__ == 'Minion'
                    and 'witch' in str(getattr(l, 'name', '')).lower()
                    for l in _ar9.leaders):
                _wk_here = True
                break
        if _wk_here:
            pc['_wk_sorcerer_deck'] = (game_logic.card_deck_of(cid)
                                       or 'strategy')
    pc[role + '_combat_card'] = cid


# K1 (R4): a HUMAN's combat card is committed face-down on the client
# and revealed later; the engine records the commit by TYPE only until
# the reveal line names it.  The sentinel occupies pc[<role>_combat_card]
# so every "has this side committed?" reader keeps working; the round is
# parked in the reveal phase (no deciding side, no menu) until every
# hidden commit is revealed.
_HIDDEN_CC = '_hidden_combat_card_'
_CC_REVEAL_PHASE = 'combat_card_reveal'


class IllegalPlay(Exception):
    """A play the engine refuses.  The state is untouched when this is
    raised, so the caller may refuse it to the client (undo + notice)."""


def combat_cards_unrevealed(state: GameState) -> List[Side]:
    """Sides whose combat card for this round is committed but still
    hidden from the engine (attacker first).  Empty when no battle is
    open or every commit is known."""
    pc = getattr(state, 'pending_combat', None)
    if not pc:
        return []
    out = []
    if pc.get('atk_hidden_cc'):
        out.append(pc['attacker_side'])
    if pc.get('def_hidden_cc'):
        out.append(pc['defender_side'])
    return out


def _cc_hidden_role(pc: dict, side: Side) -> 'Optional[str]':
    if side == pc.get('attacker_side'):
        return 'atk'
    if side == pc.get('defender_side'):
        return 'def'
    return None


def commit_hidden_combat_card(state: GameState, side: Side,
                              deck: str) -> None:
    """Record that `side` committed an UNREVEALED combat card of deck
    type `deck` ('character' | 'strategy') in its open card window and
    advance the phases exactly as a combat_card_play would (attacker ->
    defender -> reveal gate) WITHOUT resolving any effect.  Raises
    IllegalPlay (state untouched) when no window of that side is open,
    the deck is unknown, the side may not play a combat card here, or
    the engine hand holds no card of that deck."""
    pc = getattr(state, 'pending_combat', None)
    if not pc:
        raise IllegalPlay('no battle is open')
    phase = pc.get('phase')
    if phase == 'attacker_combat_card':
        role, owner = 'atk', pc.get('attacker_side')
    elif phase == 'defender_combat_card':
        role, owner = 'def', pc.get('defender_side')
    else:
        raise IllegalPlay('no combat-card window is open (phase %s)'
                          % phase)
    if side != owner:
        raise IllegalPlay('%s may not commit in the %s window'
                          % (side.name, phase))
    if deck not in ('character', 'strategy'):
        raise IllegalPlay('unknown deck %r' % (deck,))
    _tmp9, _fp_ok9 = _combat_card_check_ctx(state, pc, role == 'atk')
    if not _fp_ok9:
        raise IllegalPlay('%s may not play a combat card in this battle'
                          % side.name)
    hand = state.hands[side].hand
    if not any(game_logic.card_deck_of(c) == deck for c in hand):
        _balrog9 = False
        if side == Side.SHADOW and deck == 'character':
            from game_state import PersistentCardType as _PCT9
            _pcb9 = getattr(state, 'persistent_cards', None)
            _balrog9 = bool(_pcb9 is not None
                            and _pcb9.get_by_type(_PCT9.BALROG_OF_MORIA))
        if not _balrog9:
            raise IllegalPlay('%s holds no %s card to commit'
                              % (side.name, deck))
    pc[role + '_hidden_cc'] = {'deck': deck}
    pc[role + '_combat_card'] = _HIDDEN_CC
    _run_combat_round(state, pc)


def reveal_combat_card(state: GameState, side: Side, card_id: str) -> None:
    """Replace `side`'s hidden commit with the real card and, once no
    hidden commit remains, resume the round.  Validates the card for
    THIS battle: wrong side/deck, not playable, or not held -> raises
    IllegalPlay with the state untouched.  Idempotent for a commit the
    engine already saw (the AI's own card echoed by the reveal line):
    a card that already left the hand is a no-op."""
    pc = getattr(state, 'pending_combat', None)
    role = _cc_hidden_role(pc, side) if pc else None
    hidden = pc.get(role + '_hidden_cc') if (pc and role) else None
    hand = state.hands[side].hand
    if hidden is None:
        if card_id in hand:
            raise IllegalPlay('%s never committed %s' % (side.name,
                                                         card_id))
        return
    from cards import get_card
    card = get_card(card_id)
    if card is None:
        raise IllegalPlay('unknown card %r' % (card_id,))
    if card.side != side:
        raise IllegalPlay('%s is not a %s card' % (card_id, side.name))
    if str(card.deck).lower() != hidden['deck']:
        raise IllegalPlay('%s is a %s card, the commit was %s'
                          % (card_id, str(card.deck).lower(),
                             hidden['deck']))
    is_attacker = (role == 'atk')
    tmp9, fp_ok9 = _combat_card_check_ctx(state, pc, is_attacker)
    if not fp_ok9 or not _combat_card_playable(state, card, tmp9,
                                                is_attacker):
        raise IllegalPlay('%s is not playable in this battle' % card_id)
    from_table = False
    if card_id not in hand:
        _deck9 = hidden['deck']
        _ph9 = any(isinstance(c, str)
                   and (c.startswith('_unknown_')
                        or (c.startswith('HU_')
                            and game_logic.card_deck_of(c) == _deck9))
                   for c in hand)
        if not _ph9:
            from game_state import PersistentCardType as _PCT9
            _pcb9 = getattr(state, 'persistent_cards', None)
            _cb9 = (_pcb9.get_by_type(_PCT9.BALROG_OF_MORIA)
                    if _pcb9 is not None else None)
            if _cb9 is not None and _cb9.card_id == card_id:
                from_table = True
            else:
                raise IllegalPlay('%s holds no %s (or %s placeholder)'
                                  % (side.name, card_id, _deck9))
    pc.pop(role + '_hidden_cc', None)
    _commit_combat_card(state, pc, role, card_id, from_table)
    if not combat_cards_unrevealed(state):
        _run_combat_round(state, pc)


def cancel_hidden_combat_card(state: GameState, side: Side) -> None:
    """Withdraw `side`'s UNREVEALED commit (the client refused the
    named card and returned it to the hand): the sentinel leaves and
    that side's card window re-opens through the regular phase opener;
    the OTHER side's commit (card or decline) stands.  Raises
    IllegalPlay (state untouched) when `side` holds no hidden commit."""
    pc = getattr(state, 'pending_combat', None)
    role = _cc_hidden_role(pc, side) if pc else None
    if not role or not pc.get(role + '_hidden_cc'):
        raise IllegalPlay('%s has no hidden combat card to withdraw'
                          % side.name)
    pc.pop(role + '_hidden_cc', None)
    pc.pop(role + '_combat_card', None)
    _run_combat_round(state, pc)


def _maybe_combat_effect_choice_phase(state: GameState, pc: dict) -> bool:
    """Enter the pre-roll combat-card effect-choice phase when a committed
    card carries a choice-bearing 'Before the Combat roll' effect (Dread
    and Despair: forfeit K Nazgul leadership; Relentless Assault: K
    self-hits; Heroic Death: which figure to sacrifice, applied at its
    printed pre-casualty timing).

    Returns True if a decision phase was entered (the combat_effect_choice
    executor stores the count and resumes _run_combat_round). Onslaught is
    NOT handled here — its choice point is after casualties, deferred via
    combat.deferred_effect (see _run_combat_round's post-battle handling).
    """
    from cards import get_card
    from combat import combat_effect_kind, setup_combat
    for role in ('atk', 'def'):
        if pc.get(f'{role}_effect_asked'):
            continue
        pc[f'{role}_effect_asked'] = True   # consider each side once/round
        cid = pc.get(f'{role}_combat_card')
        if not cid:
            continue
        try:
            card = get_card(cid)
        except Exception:
            card = None
        kind = combat_effect_kind(getattr(card, 'combat_name', '') or '') \
            if card else None
        if kind not in ('dread_despair', 'relentless_assault',
                        'heroic_death'):
            continue
        side = pc['attacker_side'] if role == 'atk' else pc['defender_side']
        counts = None
        try:
            tmp = setup_combat(state, pc['attacker_region'],
                               pc['defender_region'],
                               is_siege=pc.get('is_siege', False),
                               attacker_side=pc.get('attacker_side'))
        except Exception:
            continue  # battle setup failed — heuristic fallback applies
        if kind == 'dread_despair':
            n = tmp.nazgul_in_battle
            # USER RULE 2026-07-08: forfeiting reduces the FP roll one die
            # per point but only DOWN TO A MINIMUM OF 1 — forfeiting more
            # than (FP dice - 1) is pointless and was offered (count=3 vs
            # a 2-unit defender).  Cap at the enemy's dice minus one.
            _enemy = tmp.defender_army if role == 'atk' else tmp.attacker_army
            from combat import calc_combat_strength as _ccs
            _cap9 = max(0, _ccs(_enemy) - 1)
            n = min(n, _cap9)
            if n >= 1:
                counts = list(range(1, n + 1))  # 'forfeit one or more'
        elif kind == 'relentless_assault':
            army = tmp.attacker_army if role == 'atk' else tmp.defender_army
            mx = max(0, min(2, len(army.units) - 1))
            if mx >= 1:
                counts = list(range(0, mx + 1))
        elif kind == 'heroic_death':
            army = tmp.attacker_army if role == 'atk' else tmp.defender_army
            opts = {0}
            if any(not isinstance(l, (Companion, Nazgul))
                   for l in army.leaders):
                opts.add(1)
            for c in tmp.companions_in_battle:
                if c.level >= 1:
                    opts.add(min(3, c.level))
            if len(opts) > 1:
                counts = sorted(opts)
        if counts:
            pc['effect_choice'] = {'stage': 'pre', 'owner': role,
                                   'effect': kind, 'card_id': cid,
                                   'counts': counts}
            pc['phase'] = 'combat_effect_choice'
            pc['deciding_side'] = side
            return True
    return False


def _battle_map_companions(combat) -> List[Companion]:
    """Companions physically SEPARATED on the map in this battle (present
    in an army's leaders).  EXCLUDES in-Fellowship companions: per USER
    2026-07-16 a companion hidden in the Fellowship does NOT count toward a
    combat card's companion requirement (setup_combat folds fellowship
    co-location into companions_in_battle, which we must not use here)."""
    out = []
    for a in (getattr(combat, 'attacker_army', None),
              getattr(combat, 'defender_army', None)):
        for l in getattr(a, 'leaders', []) or []:
            if isinstance(l, Companion):
                out.append(l)
    return out


def _battle_has_shadow_minion(combat) -> bool:
    """A Shadow Minion (Witch-king, Saruman, Mouth of Sauron) is in the
    battle.  The engine models all three as game_logic.Minion figures."""
    from game_logic import Minion
    for a in (getattr(combat, 'attacker_army', None),
              getattr(combat, 'defender_army', None)):
        for l in getattr(a, 'leaders', []) or []:
            if isinstance(l, Minion):
                return True
    return False


def _extra_combat_card_ok(state, card, combat) -> bool:
    """USER house rules / corrections layered on cards.can_play_combat_card
    (which counts Fellowship co-location):
      item 3 - a card's companion requirement (named character, Hobbit, or
        'a Companion in the battle') must be met by a SEPARATED companion on
        the map, not one hidden in the Fellowship.
      item 8 - Mithril Coat and Sting / Bilbo's Song may be played ONLY when
        a Shadow Minion is in the battle.

    Applied on the SELF-PLAY / learned path only: the teacher replays the
    humans' certified, rules-legal plays from the log (whose combat cards
    are matched directly), so this stricter menu must not perturb it."""
    import combat as _cmbx
    if _cmbx.FINAL_HITS_HOOK is not None:
        return True
    map_comps = _battle_map_companions(combat)
    if getattr(card, 'requires_hobbit', False):
        hobbit_names = {"Meriadoc", "Peregrin", "Frodo", "Sam"}
        if not any(getattr(c, 'name', '') in hobbit_names for c in map_comps):
            return False
    _req_chars = getattr(card, 'requires_character_in_battle', None)
    if _req_chars:
        _names = ' '.join(str(getattr(c, 'name', '')).lower()
                          for c in map_comps)
        if not any(rc in _names for rc in _req_chars):
            return False
    if getattr(card, 'requires_companion', False):
        if not map_comps:
            return False
    if getattr(card, 'card_name', '') in ("Mithril Coat and Sting",
                                          "Bilbo's Song"):
        if not _battle_has_shadow_minion(combat):
            return False
    # USER catch 2026-07-28 (rarity-audit batch, lorien sieges): a
    # retreat-effect combat card (Scouts — The Red Arrow / The Spirit
    # of Mordor / Grimbeorn) is UNPLAYABLE in a siege battle: the
    # besieged garrison is already withdrawn and has no retreat leg
    # (the F-5 execution gate already refuses the effect there, so the
    # play only burned the card).  Menu must mirror the executor.
    if getattr(combat, 'is_siege', False):
        _ce9 = (getattr(card, 'combat_effect', '') or '').lower()
        if 'retreat your army' in _ce9:
            return False
    return True


def _combat_card_check_ctx(state: GameState, pc: dict,
                           is_attacker: bool):
    """(tmp_combat, fp_ok) for combat-card play legality — the ONE
    context both the menu and the opponent card model use."""
    from combat import setup_combat
    from persistent_effects import can_fp_play_combat_card
    side = pc['attacker_side'] if is_attacker else pc['defender_side']
    tmp_combat = setup_combat(state, pc['attacker_region'],
                              pc['defender_region'],
                              is_siege=pc.get('is_siege', False),
                              attacker_side=pc.get('attacker_side'))
    fp_ok = True
    if side == Side.FREE_PEOPLES:
        # Denethor's Folly etc.: FP combat cards can be blocked
        fp_ok, _ = can_fp_play_combat_card(
            state, sanitize(pc['defender_region']))
    return tmp_combat, fp_ok


def _combat_card_playable(state: GameState, card, tmp_combat,
                          is_attacker: bool) -> bool:
    """One card's play legality in this battle (menu semantics)."""
    from cards import can_play_combat_card
    # Every card has a combat side; combat_value 0 is still playable
    # (e.g. Swarm of Bats)
    if card is None or not (card.combat_name or '').strip():
        return False
    ok, _ = can_play_combat_card(card, tmp_combat,
                                 is_attacker=is_attacker, state=state)
    return bool(ok and _extra_combat_card_ok(state, card, tmp_combat))


def _get_combat_card_actions(state: GameState, pc: dict) -> List[Action]:
    """Enumerate the combat-card decision for the current phase:
    one 'combat_card_play' per playable combat card in the deciding
    side's hand (per cards.can_play_combat_card) + 'combat_card_none'.
    """
    is_attacker = (pc.get('phase') == 'attacker_combat_card')
    side = pc['attacker_side'] if is_attacker else pc['defender_side']
    tag = 'fp' if side == Side.FREE_PEOPLES else 'sh'
    actions: List[Action] = []

    hand = state.hands.get(side)
    cand_ids = []
    if hand:
        seen = set()
        for cid in hand.hand:
            if isinstance(cid, str) and cid not in seen:
                seen.add(cid)
                cand_ids.append(cid)

    # 1.149b (USER): while 'Balrog of Moria' sits ON THE TABLE, Shadow
    # may discard it to play its combat side (Durin's Bane) as if from
    # hand — enumeration previously drew from the hand only, so the
    # option never existed.  can_play_combat_card already enforces the
    # near-Moria requirement.
    _balrog_table_cid = None
    if side == Side.SHADOW:
        from game_state import PersistentCardType as _PCT9b
        _pcb9 = getattr(state, 'persistent_cards', None)
        _cb9 = (_pcb9.get_by_type(_PCT9b.BALROG_OF_MORIA)
                if _pcb9 is not None else None)
        if _cb9 is not None and _cb9.card_id not in cand_ids:
            _balrog_table_cid = _cb9.card_id
            cand_ids.append(_cb9.card_id)

    if cand_ids:
        try:
            from cards import get_card
            tmp_combat, fp_ok = _combat_card_check_ctx(state, pc,
                                                       is_attacker)
            if fp_ok:
                for cid in cand_ids:
                    card = get_card(cid)
                    ok = _combat_card_playable(state, card, tmp_combat,
                                               is_attacker)
                    if ok:
                        _prm9 = {'card_id': cid, 'side': tag}
                        if cid == _balrog_table_cid:
                            _prm9['from_table'] = True
                        actions.append(Action(
                            die_idx=-1, action_type='combat_card_play',
                            params=_prm9))
        except Exception as _cce9:
            # review P2 (2026-07-27): NEVER silent — a throw here made
            # every legal combat card vanish and forced a decline,
            # changing game results with no signal.  Menu still
            # degrades to decline-only (robustness), but the failure is
            # logged as a HARD invariant.
            try:
                import state_invariants as _si9
                _si9.VIOLATIONS.append({
                    'code': 'combat_card_menu_failed',
                    'ctx': 'T%s side=%s' % (getattr(state, 'turn', '?'),
                                            tag),
                    'detail': '%s: %s' % (type(_cce9).__name__,
                                          str(_cce9)[:160]),
                })
            except Exception:
                pass

    actions.append(Action(die_idx=-1, action_type='combat_card_none',
                          params={'side': tag}))
    return actions


def _alloc_side_units(state, region, side, kind=None):
    army = _armies_lookup(state, region)
    if army is None:
        return []
    return [u for u in army.units
            if _side_of_nation(u.nation) == side
            and (kind is None or u.kind == kind)]


def _apply_allocation(state, region, side, hits, k_downgrades):
    """Apply `hits` to side's portion at region: k elite downgrades
    (1 hit each), then regular kills, then elite 2-hit removals —
    apply_casualties semantics with the split CHOSEN post-roll."""
    army = _armies_lookup(state, region)
    if army is None or hits <= 0:
        return
    h, k = int(hits), int(k_downgrades)
    from combat import (_capture_reg_baseline as _crb9,
                        _downgrade_source_of as _dso9,
                        _consume_downgrade_source as _cds9,
                        _track_fp_casualties as _tfc9)
    # ONE downgrade-supply answerer (R7 / K6): combat._consume_downgrade_
    # source decides box -> pool -> none against a baseline frozen at
    # this allocation's entry (this scope's own kills never masquerade
    # as free supply) and stamps _dg_src_log for the renderer.
    _base9 = _crb9(state)
    _dgc9 = {}

    def _downgrade_one(e):
        """Downgrade elite `e` (1 hit absorbed).  No replacement source
        -> the elite is ELIMINATED outright (USER ruling 2026-08-18),
        never skipped.  A downgraded FP elite is a permanent casualty."""
        _src9 = _cds9(state, e.nation, _dgc9, _base9)
        army.units.remove(e)
        if _src9 is not None:
            army.units.append(gs.Unit(e.nation, UnitKind.REGULAR))
        if side == Side.FREE_PEOPLES:
            _tfc9(state, [e], side)

    while k > 0 and h > 0:
        elites = _alloc_side_units(state, region, side, UnitKind.ELITE)
        if not elites:
            break
        # prefer an elite whose nation still has a replacement source
        e = next((u for u in reversed(elites)
                  if _dso9(state, u.nation, _dgc9, _base9) is not None),
                 elites[-1])
        _downgrade_one(e)
        h -= 1
        k -= 1
    while h > 0:
        regs = _alloc_side_units(state, region, side, UnitKind.REGULAR)
        if regs:
            army.units.remove(regs[-1])
            if side == Side.FREE_PEOPLES:
                # track IMMEDIATELY: a batched death would be invisible
                # to the leftover-downgrade supply read below
                _tfc9(state, [regs[-1]], side)
            h -= 1
            continue
        elites = _alloc_side_units(state, region, side, UnitKind.ELITE)
        if not elites:
            break
        e = elites[-1]
        if h >= 2:
            # elite fully REMOVED (2 hits) = real casualty
            army.units.remove(e)
            if side == Side.FREE_PEOPLES:
                _tfc9(state, [e], side)
            h -= 2
        else:
            # leftover single hit DOWNGRADES the elite (same answerer)
            _downgrade_one(e)
            h -= 1
    # (tracking is incremental above)


def _alloc_choice_exists(state, region, side, hits):
    if hits <= 0:
        return False
    n_e = len(_alloc_side_units(state, region, side, UnitKind.ELITE))
    n_r = len(_alloc_side_units(state, region, side, UnitKind.REGULAR))
    # AUTO-WIPE (USER 2026-07-20 "no decision if everyone dies"): when
    # hits meet/exceed the side's whole absorption capacity (2/elite +
    # 1/regular) every figure dies regardless of the downgrade split;
    # under the FP downgrade ruling a doomed-army downgrade only burns
    # extra pool Regulars. No real choice — caller auto-applies k=0
    # (minimal loss). SELF-PLAY only: the teacher replays the logged
    # human composition via apply_casualties_spec.
    if hits >= 2 * n_e + n_r:
        return False
    if not n_e:
        return False
    if n_r:
        return True
    # an all-ELITE army (no regulars) below a wipe still has a real
    # kill-vs-downgrade choice when 2 hits can kill ONE elite or
    # downgrade TWO.  Supply never gates the choice (K6): an elite
    # downgraded with no replacement is eliminated outright, which is
    # still the owner's legal allocation.
    return hits >= 2 and n_e >= 2


def _enter_casualty_allocation(state, pc):
    """Advance the post-roll allocation stages; pause on a pending
    decision when a side has a real choice, auto-apply otherwise."""
    while True:
        stage = pc.get('alloc_stage')
        if stage == 'defender':
            reg, side = pc['defender_region'], pc['defender_side']
            hits = int(pc.get('def_hits', 0) or 0)
        elif stage == 'attacker':
            reg, side = pc['attacker_region'], pc['attacker_side']
            hits = int(pc.get('atk_hits', 0) or 0)
        else:
            break
        if _alloc_choice_exists(state, reg, side, hits):
            pc['phase'] = 'casualty_allocation'
            pc['deciding_side'] = side
            return
        _apply_allocation(state, reg, side, hits, 0)
        pc['alloc_stage'] = ('attacker' if stage == 'defender' else None)
    _complete_allocated_round(state, pc)


def _complete_allocated_round(state, pc):
    """Round aftermath on the live state after post-roll allocation:
    Phase-9 post-casualty card effects (shared engine implementation),
    then destroyed portions cleaned, capture, sieges, and the normal
    post-round transitions."""
    from combat import (_check_settlement_capture, garrison_side_for,
                        reconcile_destroyed_army,
                        resume_deferred_post_casualty)
    atk_r, def_r = pc['attacker_region'], pc['defender_region']
    atk_s, def_s = pc['attacker_side'], pc['defender_side']
    atk_dead = not _alloc_side_units(state, atk_r, atk_s)
    def_dead = not _alloc_side_units(state, def_r, def_s)
    # Phase 9 resumes here on the deferred path: the standalone
    # resolve_combat_round runs it right after Phase-8 casualties; the
    # deferred path reaches that point only once allocation resolved.
    _p9 = pc.pop('_p9', None) or {}
    _p9_role = None
    if (_p9.get('atk_card') or _p9.get('def_card')) \
            and not (atk_dead and def_dead):
        _a_d9, _d_d9, _p9_role = resume_deferred_post_casualty(
            state, atk_r, def_r, atk_s, def_s,
            _p9.get('atk_card'), _p9.get('def_card'),
            _p9.get('atk_cancelled', False),
            _p9.get('def_cancelled', False),
            is_siege=pc.get('is_siege', False),
            choices={'defer_onslaught': True})
        atk_dead = atk_dead or _a_d9
        def_dead = def_dead or _d_d9
    for reg, side, dead in ((atk_r, atk_s, atk_dead),
                            (def_r, def_s, def_dead)):
        if not dead:
            continue
        # USER RULING 2026-08-15: the destroyed side's CHARACTERS die
        # with it — one implementation for every wipe path (this is the
        # deferred-onslaught / deferred-allocation resume of a battle,
        # so no card exception applies).
        reconcile_destroyed_army(state, reg, side)
    # 1.152: mutual elimination captures nothing (the original
    # controller keeps the region) - same rule as _finish_wiped_battle
    if def_dead and not atk_dead \
            and garrison_side_for(state, sanitize(str(def_r))) == def_s:
        _check_settlement_capture(state, sanitize(str(def_r)), atk_s)
    try:
        game_logic.cleanup_sieges(state)
    except Exception:
        pass
    _p9_cid = (_p9.get('atk_card') if _p9_role == 'attacker'
               else _p9.get('def_card'))
    if atk_dead or def_dead:
        if _open_deferred_onslaught(state, pc, _p9_role, _p9_cid,
                                    atk_dead, def_dead):
            return
        _finish_battle(state, pc, atk_dead, def_dead)
    else:
        if _open_deferred_onslaught(state, pc, _p9_role, _p9_cid,
                                    False, False):
            return
        _set_post_round_phase(state, pc, pc.get('is_siege', False))


def _register_move_overstack(state: GameState, region, side) -> None:
    """USER RULING 2026-07-27: combat moves (retreat merge, victor
    advance) MAY overstack the destination — the excess resolves ONCE
    THE MOVE ENDS as the learned trim decision (excess units return to
    reinforcements; the same __overstack__ pending as stronghold
    overstacking).  Self-play only — teacher compositions are
    log-truth.  Call AFTER pending_combat is cleared (the trim pending
    would otherwise be shadowed by the combat menu)."""
    try:
        from event_effects import _stack_cap, resolve_overstacks
        san = sanitize(str(region))
        army = _armies_lookup(state, san)
        if army is None:
            return
        n_own = sum(1 for u in army.units
                    if _side_of_nation(u.nation) == side)
        if n_own > _stack_cap(state, san, side):
            ovr = getattr(state, '_overstack_regions', None)
            if ovr is None:
                ovr = state._overstack_regions = set()
            ovr.add(san)
            resolve_overstacks(state, learned=True)
    except Exception:
        pass


def _resolve_wk_sorcerer(state: GameState, pc: dict) -> None:
    """Resolve a queued Witch-king 'Sorcerer' draw (set at combat-card
    play).  USER bug report 2026-07-27: the draw only resolved in the
    post-round section of _run_combat_round, so a battle terminated
    BEFORE any round (Scouts retreat-before-round, siege maintained
    without assault) silently dropped it — the card WAS played, the
    draw is owed on every termination path.  Pop-based: safe to call
    from multiple sites."""
    _wk_deck9 = pc.pop('_wk_sorcerer_deck', None)
    if _wk_deck9 is not None:
        # RULING 1.228b (USER 2026-08-03): if Fateful Strike / Blade
        # of Westernesse eliminated the Witch-king during the battle,
        # his Sorcerer ability cannot activate — the queued draw dies
        # with him.
        _wk9 = getattr(state, 'witch_king', None)
        if _wk9 is not None and not getattr(_wk9, 'in_play', True):
            return
        try:
            # engine-granted draw (A13 source 5): grant_draw stamps the
            # owed-draw debt when a client-owned draw defers (a
            # BaseException deferral passes through this except
            # untouched; only deck-empty RuntimeError is swallowed here)
            game_logic.grant_draw(state, Side.SHADOW, _wk_deck9)
            from game_trace import emit_annotation
            emit_annotation(state, 'wk_sorcerer_draw',
                            {'deck': _wk_deck9})
        except Exception:
            pass
        _check_hand_limits(state)


def _open_deferred_onslaught(state: GameState, pc: dict, role,
                             card_id, atk_dead: bool,
                             def_dead: bool) -> bool:
    """Open the learned Onslaught-magnitude pending for a deferred
    post-casualty Onslaught.  fix-78 semantics: reachable even when the
    round's casualties ended the battle — the surviving owner's
    sacrifice is still a decision (extra hits fizzle); the stashed
    destruction flags make the executor re-finish the battle.
    Returns True iff the pending phase was opened."""
    if role not in ('attacker', 'defender'):
        return False
    own_dead = atk_dead if role == 'attacker' else def_dead
    if own_dead:
        return False
    from combat import onslaught_max_sacrifice
    own_side = pc[f'{role}_side']
    own_region = pc[f'{role}_region']
    mx = onslaught_max_sacrifice(state, own_region, own_side)
    if mx <= 0:
        return False
    if atk_dead or def_dead:
        try:
            from game_trace import emit_annotation
            emit_annotation(state, 'onslaught_deferred_on_destroyed',
                            {'region': pc.get('defender_region'),
                             'card': str(card_id), 'role': role})
        except Exception:
            pass
        pc['_finish_atk_dead'] = atk_dead
        pc['_finish_def_dead'] = def_dead
    pc['effect_choice'] = {
        'stage': 'post',
        'owner': 'atk' if role == 'attacker' else 'def',
        'effect': 'onslaught', 'card_id': card_id or '',
        'max_count': mx}
    pc['phase'] = 'combat_effect_choice'
    pc['deciding_side'] = own_side
    return True


def _finish_wiped_battle(state: GameState, pc: dict,
                         atk_dead: bool, def_dead: bool) -> None:
    """ONE wipe-finish implementation (CHAOS-1 root + guard, seeds
    96327/96806: a siege_press killed the last attacker unit - the
    press downgrade can pop an elite outright - and the round
    machinery walked straight into the next card window).
    Characters die with the army (USER 2026-08-15), capture settles,
    _finish_battle runs cleanup_sieges so the stale flag lifts."""
    from combat import (reconcile_destroyed_army as _rda9,
                        _check_settlement_capture as _csc9,
                        garrison_side_for as _gsf9)
    for _rg9, _sd9, _dead9 in (
            (pc['attacker_region'], pc['attacker_side'], atk_dead),
            (pc['defender_region'], pc['defender_side'], def_dead)):
        if _dead9:
            try:
                _rda9(state, _rg9, _sd9)
            except Exception:
                pass
    if def_dead and not atk_dead:
        try:
            _drs9 = sanitize(str(pc['defender_region']))
            if _gsf9(state, _drs9) == pc['defender_side']:
                _csc9(state, _drs9, pc['attacker_side'])
        except Exception:
            pass
    _finish_battle(state, pc, atk_dead, def_dead)


def _run_combat_round(state: GameState, pc: dict,
                      is_siege: bool = None) -> None:
    """Run 1 round of combat and update pending_combat state.

    Sets phase to 'combat_round_result' if both armies survive,
    or clears pending_combat if one army is destroyed.
    """
    # CHAOS-1 ROOT (measured, seed 96327 T24 rivendell): the entry
    # never checked that BOTH sides still exist - a siege_press whose
    # downgrade killed the attacker's last unit (no regular in the
    # casualty box or pool: the elite dies outright) opened the NEXT
    # card window on a one-sided battle, leaving the stale besieged
    # flag + the combat_card_menu_failed degradation.  A wiped side =
    # the battle is over, before any phase opener.
    _wgr9 = _pending_battle_wiped(state, pc)
    if _wgr9 is not None:
        print('[combat] battle over before the round: a side is '
              'wiped (%s)' % pc.get('defender_region'))
        _finish_wiped_battle(state, pc, _wgr9[0], _wgr9[1])
        return
    # Seeds per-battle casualty-policy DEFAULTS only (never opens a
    # decision phase; the real decision is the per-round learned
    # casualty allocation)
    if _maybe_casualty_policy_phase(state, pc):
        return

    # Combat cards are chosen per ROUND (learned decisions): the attacker
    # commits first, then the defender. The executor for
    # combat_card_play / combat_card_none resumes this function.
    if is_siege is not None:
        pc['is_siege'] = bool(is_siege)  # persist for phase resumption
    # Gandalf's 'shine' decision (item 11) precedes combat-card declaration
    if _maybe_shine_phase(state, pc):
        return
    if _maybe_combat_card_phase(state, pc):
        return
    # K1 reveal gate: a hidden (human) commit parks the round - no
    # effect choice and no roll until reveal_combat_card names every
    # card.  No side decides here (the client's reveal does).
    if combat_cards_unrevealed(state):
        pc['phase'] = _CC_REVEAL_PHASE
        pc['deciding_side'] = None
        return

    # Choice-bearing pre-roll card effects (Dread and Despair, Relentless
    # Assault, Heroic Death): the owner picks the magnitude (learned).
    # The combat_effect_choice executor stores the count and resumes here.
    if _maybe_combat_effect_choice_phase(state, pc):
        return

    # Retreat combat card (Grimbeorn / The Red Arrow / The Spirit of
    # Mordor — "Before the Combat roll, retreat your Army to an adjacent
    # free region or withdraw into a siege").  When the defender commits
    # such a card, the retreat is a LEARNED decision (destination) taken
    # BEFORE round 1 via the existing defender_retreat phase: the teacher
    # forces it from the log's retreat record and it becomes a training
    # sample; self-play's net chooses.  Guarded to fire once per battle.
    if (pc.get('rounds_fought', 0) == 0
            and not pc.get('_card_retreat_offered')):
        _dcid = pc.get('def_combat_card')
        _deff = ''
        if _dcid:
            try:
                from cards import get_card as _gc
                _deff = (getattr(_gc(_dcid), 'combat_effect', '')
                         or '').lower()
            except Exception:
                _deff = ''
        if 'retreat your army' in _deff and not pc.get('is_siege'):
            # F-5 gate (fuzz seed 9322, 2026-07-20): the card's siege
            # half is "or withdraw into a siege" — a besieged garrison
            # is already withdrawn and has NO retreat leg. Pre-gate,
            # the siege-defender's card retreat moved the CO-LOCATED
            # entry: the six Woodland Realm besiegers "retreated" into
            # FP-held Dale (mixed army, phantom attacks downstream).
            pc['_card_retreat_offered'] = True
            pc['_card_retreat_mandatory'] = True
            pc['phase'] = 'defender_retreat'
            pc['deciding_side'] = pc['defender_side']
            return
        # W5 G-2 fix (2026-07-20): the ATTACKER's committed
        # 'retreat your army' card was silently inert — only
        # def_combat_card was checked. Same gates (once per battle,
        # never from a siege), attacker decides.
        _acid = pc.get('atk_combat_card')
        _aeff = ''
        if _acid:
            try:
                from cards import get_card as _gca
                _aeff = (getattr(_gca(_acid), 'combat_effect', '')
                         or '').lower()
            except Exception:
                _aeff = ''
        if 'retreat your army' in _aeff and not pc.get('is_siege'):
            pc['_card_retreat_offered'] = True
            pc['_card_retreat_mandatory'] = True
            pc['phase'] = 'attacker_card_retreat'
            pc['deciding_side'] = pc['attacker_side']
            return

    if is_siege is None:
        is_siege = pc.get('is_siege', False)

    atk_region = pc['attacker_region']
    def_region = pc['defender_region']

    # Combat cards committed in the decision phases (consumed per round —
    # the next round asks again); same for the per-round effect choices
    atk_card_id = pc.pop('atk_combat_card', None)
    def_card_id = pc.pop('def_combat_card', None)
    atk_effect_count = pc.pop('atk_effect_count', None)
    def_effect_count = pc.pop('def_effect_count', None)
    pc.pop('atk_effect_asked', None)
    pc.pop('def_effect_asked', None)

    # Store card picks for GUI observability
    state._last_combat_cards = (atk_card_id, def_card_id)

    # Run exactly 1 round.
    # After round 1, city/fortification defense no longer applies
    # (attacker hits on 5+ instead of 6+). Pass disable flag for rounds 2+.
    rounds_fought = pc.get('rounds_fought', 0)
    _disable_city_def = (rounds_fought > 0)

    _round_choices = {
        'attacker_side': pc.get('attacker_side'),
        'disable_city_defense': _disable_city_def,
        'atk_casualty_policy': pc.get('atk_casualty_policy',
                                      'preserve_elites'),
        'def_casualty_policy': pc.get('def_casualty_policy',
                                      'preserve_elites'),
        'atk_effect_count': atk_effect_count,
        'def_effect_count': def_effect_count,
        # P2 (Help Unlooked For): 'for this entire battle' - in pc
        'attacker_dice_reduction': pc.get('atk_dice_reduction', 0),
        'defender_dice_reduction': pc.get('def_dice_reduction', 0),
        'defer_casualties': True,
        # Onslaught's sacrifice count is a learned post-casualty
        # decision
        'defer_onslaught': True,
    }
    # kernel-node capture: everything the in-tree exchange kernel needs
    # to reproduce THIS round (read post-apply; never inherited across
    # clones — clone_state whitelists attrs)
    state._last_round_call = {
        'atk_region': atk_region, 'def_region': def_region,
        'atk_side': pc.get('attacker_side'),
        'def_side': pc.get('defender_side'),
        'atk_card': atk_card_id, 'def_card': def_card_id,
        'is_siege': bool(is_siege), 'gtw_shine': pc.get('gtw_shine'),
        'choices': dict(_round_choices)}
    result = game_logic.battle(state, atk_region, def_region,
                               attacker_card_id=atk_card_id,
                               defender_card_id=def_card_id,
                               is_siege=is_siege, max_rounds=1,
                               auto_press=False,
                               choices=_round_choices)
    state._last_battle_result = result
    pc['rounds_fought'] = pc.get('rounds_fought', 0) + 1

    _resolve_wk_sorcerer(state, pc)

    if getattr(result, 'casualties_deferred', False):
        pc['def_hits'] = int(result.deferred_hits_on_defender)
        pc['atk_hits'] = int(result.deferred_hits_on_attacker)
        pc['alloc_stage'] = 'defender'
        if atk_card_id or def_card_id:
            # Phase-9 payload: allocation completion resumes the
            # post-casualty card effects with the round's true
            # cancellation state (_complete_allocated_round)
            pc['_p9'] = {
                'atk_card': atk_card_id, 'def_card': def_card_id,
                'atk_cancelled': bool(getattr(
                    result, 'attacker_card_cancelled', False)),
                'def_cancelled': bool(getattr(
                    result, 'defender_card_cancelled', False))}
        else:
            pc.pop('_p9', None)
        _enter_casualty_allocation(state, pc)
        return

    _role = getattr(result, 'deferred_effect', None)
    _cid = atk_card_id if _role == 'attacker' else def_card_id
    if result.attacker_destroyed or result.defender_destroyed:
        if _open_deferred_onslaught(state, pc, _role, _cid,
                                    result.attacker_destroyed,
                                    result.defender_destroyed):
            return
        _finish_battle(state, pc, result.attacker_destroyed,
                       result.defender_destroyed)
    else:
        # Both armies survive. A deferred Onslaught ('after removing
        # casualties, inflict up to four hits against your units...')
        # opens its choice phase BEFORE the round-result decisions.
        if _open_deferred_onslaught(state, pc, _role, _cid,
                                    False, False):
            return
        _set_post_round_phase(state, pc, is_siege)


def _finish_battle(state: GameState, pc: dict, attacker_destroyed: bool,
                   defender_destroyed: bool) -> None:
    """Battle decided: victor advance, siege cleanup, rearguard restore.

    USER RULE 2026-07-08: the victor chooses HOW MANY units advance
    (free split — 0, some, or all), it is not an automatic full-stack
    move.  The advance becomes a pending 'victor_advance' decision
    (composition grid + stay)."""
    atk_region = pc['attacker_region']
    def_region = pc['defender_region']
    if (defender_destroyed and not attacker_destroyed
            and sanitize(atk_region) != sanitize(def_region)):
        atk_army = _armies_lookup(state, atk_region)
        if atk_army and atk_army.units:
            try:
                game_logic.cleanup_sieges(state)
            except Exception:
                pass
            _restore_rearguard(state, pc)
            state.pending_combat = {
                'phase': 'victor_advance',
                'attacker_region': atk_region,
                'defender_region': def_region,
                'attacker_side': pc['attacker_side'],
                'defender_side': pc['defender_side'],
                'deciding_side': pc['attacker_side'],
                'is_siege': False, 'rounds_fought':
                    pc.get('rounds_fought', 0),
            }
            return
    # F-7 (fuzz seed 9531, 2026-07-20): the old unconditional
    # `bs.discard(def_region)` LIFTED THE SIEGE when a failed RELIEF
    # attack ended (the relief battle's def_region IS the besieged
    # stronghold — the besiegers defend there); garrison + besiegers
    # were left co-located with no siege flag: phantom from==to
    # attacks, illegal muster/move offers. cleanup_sieges implements
    # the correct condition (flag lifts only when one side no longer
    # holds the region).
    try:
        game_logic.cleanup_sieges(state)
    except Exception:
        pass
    _restore_rearguard(state, pc)
    state.pending_combat = None
    _maybe_open_palantir_draw(state)   # USER 1.208m: deferred Palantir draw


def _set_post_round_phase(state: GameState, pc: dict, is_siege: bool) -> None:
    """Both armies survive: route to the post-round decision phase."""
    if is_siege:
        if pc.get('mandatory_rounds', 0) > pc.get('rounds_fought', 0):
            # P2 (Grond / Fighting Uruk-hai, almanac A2/C20): the
            # first three rounds are REQUIRED - neither side may end
            # the battle early and no elite is downgraded; per-round
            # decisions (combat cards, effect choices) still interpose
            # via the phase machinery before each round is rolled.
            _run_combat_round(state, pc, is_siege=True)
            return
        # Siege: attacker decides whether to press (no retreat option)
        pc['phase'] = 'combat_round_result'
        pc['deciding_side'] = pc['attacker_side']
    else:
        # Field battle (rulebook p.29): after each round the ATTACKER
        # first decides to cease or continue; if continuing, the
        # DEFENDER then gets the retreat window.
        pc['phase'] = 'attacker_continue'
        pc['deciding_side'] = pc['attacker_side']


def _retreat_army(state: GameState, from_region: str,
                  to_region: str, side: 'Optional[Side]' = None) -> None:
    """Move a retreating army to the target region.

    SANITIZED-SCAN key resolution (2026-07-07): pc regions are
    sanitized while parser-created entries keep original casing
    ('Pelargir' vs 'pelargir') — the exact/sanitized .get pair missed
    the entry and the retreat silently NO-OPED, leaving the defender
    in place and the battle region mixed (2023-11-03 ketchuptime T6:
    the Pelargir defenders never left, the victor advance then failed
    'Not enough elites in stack' against the mixed entry, and the T7
    Lamedon->Dol Amroth attack found no army)."""
    _fk = next((k for k in state.armies
                if sanitize(str(k)) == sanitize(str(from_region))), None)
    army = state.armies.get(_fk) if _fk is not None else None
    if not army or not army.units:
        return
    # F-5 hardening (fuzz seed 9322, 2026-07-20): on a MIXED entry
    # (co-located siege model) move ONLY the retreating side's units
    # and leaders — the whole-entry move_army dragged the opposing
    # army along with the retreat.
    if side is not None:
        _other = [u for u in army.units
                  if _side_of_nation(u.nation) != side]
        if _other:
            from combat import _combat_leader_side as _cls5
            _mv_u = [u for u in army.units
                     if _side_of_nation(u.nation) == side]
            _mv_l = [l for l in army.leaders
                     if _cls5(l) == side
                     and not game_logic._is_saruman(l)]
            if not _mv_u:
                return
            army.units = [u for u in army.units if u not in _mv_u]
            army.leaders = [l for l in army.leaders if l not in _mv_l]
            dest = game_logic.get_or_create_army(state, sanitize(to_region))
            dest.units.extend(_mv_u)
            dest.leaders.extend(_mv_l)
            return
    try:
        game_logic.move_army(state, _fk, to_region)
    except Exception:
        # Fallback: manually move (Saruman never leaves Orthanc — keep
        # him behind exactly like move_army does)
        from_key = _fk
        to_key = sanitize(to_region)
        if from_key in state.armies:
            _stay = [l for l in army.leaders if game_logic._is_saruman(l)]
            if _stay:
                army.leaders = [l for l in army.leaders
                                if not game_logic._is_saruman(l)]
            existing = _armies_lookup(state, to_key)
            if existing and existing.units:
                existing.units.extend(army.units)
                existing.leaders.extend(army.leaders)
            else:
                state.armies[to_key] = army
            del state.armies[from_key]
            if _stay:
                back = game_logic.get_or_create_army(state, from_key)
                back.leaders.extend(_stay)


def _pop_hobbit_place_queue(state: GameState) -> None:
    """Delegate: game_logic.pop_hobbit_place_queue is the ONE
    implementation (1.230 factoring)."""
    game_logic.pop_hobbit_place_queue(state)


def _open_palantir_deck_choice(state: GameState) -> None:
    """USER 1.208m(1): the Palantir of Orthanc bonus draw is a LEARNED
    which-deck decision (Character / Strategy).  Rides pending_card_option
    (frozen encode).  Self-play only; the teacher replays the logged
    draw's deck via forced_draw."""
    if getattr(state, 'pending_card_option', None) is not None:
        # I3 (USER live 2026-08-18, verbatim: "Palantir of Orthanc was
        # in play (engine knew) but the AI never drew from it -
        # multiple occurrences").  This early return used to DROP the
        # bonus draw on the floor, and it fired every time, because the
        # collision it guards against is the NORMAL case: the card the
        # Shadow just played usually opens its OWN card option first.
        # MEASURED on his session - all three owed Palantir draws were
        # option-bearing plays (The Shadow Lengthens client 310, The
        # Ringwraiths are Abroad 357, Grond 422) and all three were
        # lost, while the two Witch-king draws that owed nothing to
        # this path both fired.
        # The deferral mechanism already exists for exactly this shape
        # (the P-die-still-resolving case), so reuse it rather than
        # inventing a second one: _maybe_open_palantir_draw re-offers
        # the choice from nine resolution sites once the way is clear.
        state._palantir_draw_pending = True
        return
    # 2026-08-16: only offer decks that can actually be drawn from —
    # WOTR has no reshuffle, so an exhausted deck is simply not a
    # choice (offering it produced an unguarded RuntimeError that
    # halted a live game).  No drawable deck at all = no decision.
    _decks9 = [d for d, _a9 in (('character', 'sh_char_deck'),
                                ('strategy', 'sh_strat_deck'))
               if getattr(state, _a9, None)]
    if not _decks9:
        return
    state.pending_card_option = {
        'card_id': '__palantir_draw__', 'card_name': 'Palantir draw',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'palantir_draw': True,
        'options': [{'palantir_deck': d} for d in _decks9]}


def _maybe_open_palantir_draw(state: GameState) -> None:
    """USER 1.208m: fire the deferred Palantir bonus draw once the P-die
    action FULLY resolves (battle ended, all presses done).  Self-play
    only (the teacher replays the logged draw).

    1.228d (USER 2026-08-03): if Saruman was eliminated DURING the
    battle (Fateful Strike / Blade of Westernesse card kill), the
    Palantir of Orthanc was discarded with him (1.228c) and the
    queued draw dies too — re-check the table card at fire time,
    mirroring the 1.228b Witch-king Sorcerer gate."""
    if getattr(state, '_palantir_draw_pending', False) \
            and state.pending_combat is None:
        state._palantir_draw_pending = False
        try:
            from persistent_effects import should_draw_extra_card_on_event
            if not should_draw_extra_card_on_event(state):
                return
        except Exception:
            pass
        _open_palantir_deck_choice(state)


def _record_exec_failure(state: GameState, action: Action,
                         err: Exception) -> None:
    """Log a swallowed execution exception as a HARD invariant
    violation (code 'action_exec_failed') so showcases surface it."""
    try:
        import state_invariants as _si
        _si.VIOLATIONS.append({
            'code': 'action_exec_failed',
            'ctx': f"T{getattr(state, 'turn', '?')} "
                   f"{action.action_type} {action.params}",
            'detail': f"{type(err).__name__}: {err}",
        })
    except Exception:
        pass
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'exec_failure', {
            'action': action.action_type,
            'error': f"{type(err).__name__}: {str(err)[:120]}"})
    except Exception:
        pass


def _recover_from_failed_action(state: GameState, action: Action,
                                active: Side) -> None:
    """After an action raised during execution, restore the game to a
    playable state: abandon any stuck pending decision phase and consume
    the offending die so the same invalid action can't be re-selected
    forever (turn progress requires dice to run out).
    """
    if (state.pending_combat is not None or state.pending_hunt is not None
            or state.pending_guide_change is not None
            or getattr(state, '_reveal_place_due', False)):
        if state.pending_combat is not None:
            try:
                _restore_rearguard(state, state.pending_combat)
            except Exception:
                pass
        state.pending_combat = None
        state.pending_hunt = None
        state.pending_guide_change = None
        # abandon a stuck reveal placement like any other stuck
        # decision phase (USER RULING 2026-08-16 pending family)
        state._reveal_place_due = False
        return
    # Card pendings carry die_idx=-1 (no die to consume): they must be
    # cleared here too, or the identical failing menu is re-offered
    # every iteration until the episode cap burns out.
    if (getattr(state, 'pending_card_option', None) is not None
            or getattr(state, 'pending_card_followup', None) is not None
            or getattr(state, 'pending_discard', None) is not None):
        state.pending_card_option = None
        state.pending_card_followup = None
        state.pending_discard = None
        return
    state.pending_army_move_die = None
    state.pending_army_move_from = None
    state.pending_army_move_side = None
    state.pending_army_move_preexist = None
    state.pending_muster_die = None
    state.pending_muster_first_type = None
    state.pending_muster_region = None
    state.pending_muster_side = None
    dice = state.dice.get(active, [])
    if 0 <= action.die_idx < len(dice):
        dice[action.die_idx].used = True


def _dedupe_armies(state: GameState) -> None:
    """Merge army entries whose keys sanitize to the same region.

    Some creation sites keyed state.armies by original names ('Orthanc'),
    others by sanitized ones ('orthanc'); once both exist for a region,
    sanitized-match lookups become insertion-order dependent — phantom
    'no Nazgul at X' / 'no companions at X' failures that silently killed
    self-play episodes. One O(armies) pass restores the invariant; called
    before every action execution as a self-healing net over the ~15
    historical creation sites.
    """
    seen: dict = {}
    dupes = []
    for k in state.armies.keys():
        san = sanitize(k)
        if san in seen:
            dupes.append((seen[san], k))
        else:
            seen[san] = k
    for keep, extra in dupes:
        a, b = state.armies[keep], state.armies[extra]
        a.units.extend(b.units)
        a.leaders.extend(b.leaders)
        del state.armies[extra]


def _traced_execute(state: GameState, action: Action, tracer):
    """Execute a REAL self-play action with engine roll/tile trace
    annotations active (iter 38).  Scoped to the single call so MCTS
    lookahead (apply_action on clones) never spams simulated rolls into
    the trace."""
    if tracer is not None:
        try:
            tracer.begin_action_buffer()
        except Exception:
            pass
    if tracer is None:
        return _execute_action(state, action)
    import game_trace as _gt
    _gt.ACTIVE_TRACER = tracer
    try:
        return _execute_action(state, action)
    finally:
        _gt.ACTIVE_TRACER = None


def _execute_action(state: GameState, action: Action):
    """Execute an action on a state (mutates in place).

    Dedupes armies before AND after the action: combat resolution and
    event effects create army entries under non-canonical keys, and the
    next enumeration must see canonical armies or stacking checks and
    unit lookups go wrong (enumeration/execution mismatches).
    """
    _dedupe_armies(state)
    # CHAOS-1 (RED 2026-08-17) EXECUTOR TWIN of the menu guard: a
    # pending battle whose entry lost one whole side outside the
    # round machinery FINISHES as a wipe here (the menu offered only
    # the decline; executing it against the ghost battle raised
    # action_exec_failed forever).  Characters die with the army
    # (USER 2026-08-15, reconcile_destroyed_army = the one wipe
    # implementation), capture settles, cleanup_sieges lifts the
    # stale flag inside _finish_battle.  The incoming action was
    # generated against the ghost menu - absorbed.
    _pcw9 = getattr(state, 'pending_combat', None)
    if isinstance(_pcw9, dict) \
            and _pcw9.get('phase') in _WIPE_GUARD_PHASES:
        _wg9 = _pending_battle_wiped(state, _pcw9)
        if _wg9 is not None:
            print('[combat] stale battle auto-finished: a side was '
                  'wiped outside the round machinery (%s)'
                  % _pcw9.get('defender_region'))
            _finish_wiped_battle(state, _pcw9, _wg9[0], _wg9[1])
            return
    # Bug-1 turn-routing (SH attacking FP with combat cards + a FP retreat
    # used to hand SH two actions in a row): a whole battle is ONE action
    # spent by the ATTACKER, but its multi-step resolution (combat cards,
    # retreat, cease) flips the acting side through the defender.  When a
    # battle RESOLVES, remember the attacker so the next die-action routes
    # to the NON-attacker.  A die-less forced reaction (hand-limit discard,
    # hunt, guide change) that follows must NOT drop the marker — it is
    # cleared only when a real die-spending action lands.
    _pre_pc = state.pending_combat
    _pre_atk = (_pre_pc.get('attacker_side')
                if isinstance(_pre_pc, dict) else None)

    def _unused_die_total(_s):
        _n = 0
        for _sd in (Side.FREE_PEOPLES, Side.SHADOW):
            for _d in _s.dice.get(_sd, []):
                if not _d.used and _d.current is not None:
                    _n += 1
        return _n
    _pre_unused = _unused_die_total(state)
    # the elven_ring free-action marker lives exactly until the NEXT
    # executed action (ring + funded action = one move)
    if action.action_type != 'elven_ring':
        state._ring_free_side = None
    try:
        _execute_action_inner(state, action)
    finally:
        _dedupe_armies(state)
        _check_hand_limits(state)
        # iter 36 siege upkeep (SELF-PLAY only — the teacher replay
        # manages siege state from the humans' logged drags): at stable
        # points, a stronghold stays besieged only while both sides
        # occupy the region; a dead/departed besieger lifts the siege.
        if state.pending_combat is None:
            try:
                game_logic.cleanup_sieges(state)
            except Exception:
                pass
        # "Play on the table" upkeep (The Last Battle, Denethor's
        # Folly) - BOTH modes (USER directive 2026-07-10: nothing
        # self-play only); teacher's logged discard finds the card
        # already gone, which remove_card tolerates
        if state.pending_combat is None:
            _recheck_last_battle(state)
        # 1.211f: drain any owed Worn-with-Sorrow-and-Toil discard as a
        # learned Shadow SOURCE pending once the action fully settled.
        # SELF-PLAY only (the teacher forces the logged source inline
        # via state._wwsat_forced); never while another pending is open.
        if (state.pending_combat is None
                and getattr(state, 'pending_card_option', None) is None
                and state.pending_hunt is None
                and getattr(state, '_wwsat_owed', None)):
            _offer_wwsat_discard(state)
        # 1.132 (USER RULE): deferred reveal placement - the revealing
        # tile's own damage decision resolved first.  USER RULING
        # 2026-08-16: the placement itself is a LEARNED FP DECISION —
        # open_reveal_placement resolves the forced/no-choice cases
        # via the certified auto path and otherwise leaves
        # _reveal_place_due set as an OPEN FP pending (menu served by
        # get_legal_actions, routed by _pending_decision_side, applied
        # by the fellowship_reveal_place executor).
        open_reveal_placement(state)

        # Bug-7 guide invariant: a Guide must always exist while any
        # Companion remains in the Fellowship. Every companion-leaving path
        # (separation, hunt casualty, guide sacrifice) already re-picks via
        # change_guide; this net catches any OTHER path (e.g. a card effect
        # that removes a companion directly) so the Fellowship is never left
        # guideless. Re-pick the highest-level remaining companion; a
        # genuine top-level tie surfaces as a choose_guide decision.
        try:
            _fs7 = state.fellowship
            if _fs7.companions \
                    and getattr(state, 'pending_guide_change', None) is None \
                    and (_fs7.guide is None
                         or _fs7.guide not in _fs7.companions):
                game_logic.change_guide(state, allow_pending=True)
        except Exception:
            pass

        # Bug-1 turn-routing marker upkeep (see the pre-try capture above):
        if _pre_pc is not None and state.pending_combat is None:
            # a battle just resolved via THIS action -> the next die-action
            # belongs to the non-attacker
            state._battle_resolved_attacker = _pre_atk
        elif _unused_die_total(state) < _pre_unused \
                or action.action_type == 'pass':
            # a real die-spending action landed OR the routed side chose
            # to PASS (its voluntary turn-order action) -> the battle
            # marker has done its job.  Passing consumes no die, so the
            # old die-count-only test left the marker set and
            # _post_action_active bounced control back to the SAME side
            # after every pass — an infinite pass loop that silently
            # burned self-play's iteration cap (and stalled live games).
            state._battle_resolved_attacker = None
        # else: a die-less forced reaction (discard / hunt / guide) -> keep
        # the marker so it still routes the post-reaction die-action



def _recheck_last_battle(state: GameState) -> None:
    """'Play on the table' upkeep for The Last Battle (almanac A1, QA
    2026-07-09): discard as soon as the play condition stops holding at
    the end of an action - Aragorn's army eliminated / moved into a
    Free Peoples Nation / Aragorn gone, or the Fellowship declared or
    revealed.  Self-play only: the teacher replays the humans' own
    logged discard."""
    pc = getattr(state, 'persistent_cards', None)
    if pc is None:
        return
    from game_state import PersistentCardType as _PCT
    if pc.has_card_type(_PCT.LAST_BATTLE):
        card = pc.get_by_type(_PCT.LAST_BATTLE)
        ok = bool(state.fellowship.hidden)
        if ok:
            from event_effects import check_event_requirement
            ok = check_event_requirement(state, card.card_id)[0]
        if not ok:
            from persistent_effects import discard_table_card as _dtc9
            _dtc9(state, pc.remove_card(card.card_id) or card)
    # Denethor's Folly (almanac A3, QA 2026-07-10): discarded as soon
    # as Minas Tirith is no longer under siege (captured or relieved)
    if pc.has_card_type(_PCT.DENETHORS_FOLLY):
        card = pc.get_by_type(_PCT.DENETHORS_FOLLY)
        if 'minas tirith' not in getattr(state, 'besieged_strongholds',
                                         set()):
            from persistent_effects import discard_table_card as _dtc9
            _dtc9(state, pc.remove_card(card.card_id) or card)


def _offer_wwsat_discard(state: GameState) -> None:
    """1.211f: drain ONE owed 'Worn with Sorrow and Toil' discard.

    When both a table FP Character card AND an FP hand Character card
    still exist, open a learned Shadow SOURCE pending (kill a named
    table card, or discard a random FP hand Character card - NO decline
    per 1.210g).  When only one source remains, resolve it directly.
    Self-play only (drained from the _execute_action finally block)."""
    owed = getattr(state, '_wwsat_owed', None)
    if not owed:
        return
    owed.pop()
    hand = state.hands[Side.FREE_PEOPLES].hand
    # one deck classifier with _wwsat_resolve (live placeholders too)
    chars = [c for c in hand
             if game_logic.card_deck_of(c) == 'character']
    pcards = getattr(state, 'persistent_cards', None)
    table_chars = (pcards.get_fp_character_cards()
                   if pcards is not None else [])
    if not table_chars:
        game_logic._wwsat_resolve(state, 'hand')
        return
    if not chars:
        game_logic._wwsat_resolve(state, 'table')
        return
    options = [{'wwsat_source': 'table', 'card_id': c.card_id}
               for c in table_chars]
    options.append({'wwsat_source': 'hand'})
    state.pending_card_option = {
        'card_id': 'Worn with Sorrow and Toil',
        'card_name': '__wwsat_discard__',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': options,
    }


def _check_hand_limits(state: GameState) -> None:
    """Open a pending hand-limit discard decision if any hand exceeds 6.

    Catches every overdraw source (event effects, the Palantír extra
    draw, the turn rollover's two draws, ...) in one place — including
    inside MCTS lookahead, where the discard then becomes part of the
    searched decision sequence.

    Delegates to game_logic.open_hand_limit_discard: the RULES layer
    owns "the hand limit is 6" and the turn path (game_logic.next_turn)
    calls the same opener, so there is exactly ONE implementation
    (USER RULING 2026-08-15 — the discard is a learned decision and no
    path may auto-trim).
    """
    game_logic.open_hand_limit_discard(state)


def _discard_played_combat_card(state, cid):
    """A combat card committed in battle leaves the hand and goes FACE-UP
    to its deck's discard pile (physical pile in the app) - it must not
    vanish. Call ONLY when this site actually removed the card from a
    hand (prevents double-discard vs the closed_loop flush). Applies on
    BOTH paths: log = truth includes the physical discard. CL-217."""
    try:
        from cards import get_card as _gc
        c = _gc(cid)
        if c is None:
            return
        pfx = 'fp' if c.side == Side.FREE_PEOPLES else 'sh'
        dk = "%s_%s" % (pfx, str(c.deck).lower())
        d = getattr(state, 'discard_face_up', None)
        if isinstance(d, dict) and dk in d:
            d[dk].append(cid)
    except Exception:
        pass


def _execute_action_inner(state: GameState, action: Action):
    """Execute an action on a state (mutates in place).

    Handles multi-step die actions:
    - Army die: first army_move sets pending (second move), second consumes die.
    - Muster die: first muster_recruit sets pending (second recruit), second consumes.
    - pass clears any pending state and consumes the die.
    """
    # the declare phase is exactly one decision deep — ANY executed
    # action (declare or pass) ends it, incl. inside MCTS clones
    if getattr(state, '_declare_phase', False):
        state._declare_phase = False
    # v30 cross-turn search: the in-tree turn boundary.  Only ever
    # injected by _settle_new_node during search — never enumerated by
    # get_legal_actions, never executed by the episode loops.
    if action.action_type == 'end_turn':
        _advance_turn_for_search(state)
        return
    atype = action.action_type
    p = action.params
    die_idx = action.die_idx

    if atype == 'skip':
        # Skip: discard one die without effect (always available)
        _consume_die(state, die_idx)

    elif atype == 'pass':
        # Pass: do nothing, no die consumed (only when fewer dice than opponent)
        # OR: consume pending multi-step die
        # If pending multi-step action, consume that die
        if state.pending_army_move_die is not None:
            # fix 86 (Feanor T6 gate-1): if the stored pending die is
            # ALREADY USED (a teacher forcer settled the second leg and
            # consumed it directly), closing the window must not fall
            # back onto a FRESH die — that double-charged the pair and
            # starved the next Muster/Army record.
            _pd86 = state.pending_army_move_die
            _pdice86 = state.dice.get(state.pending_army_move_side, [])
            if not (0 <= _pd86 < len(_pdice86)
                    and _pdice86[_pd86].used):
                # (MUSTER included for the Mouth's muster-as-army ability)
                _consume_die_for_face(
                    state, state.pending_army_move_side,
                    state.pending_army_move_die,
                    (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY,
                     ActionDieFace.WILL_OF_WEST, ActionDieFace.MUSTER))
            state.pending_army_move_die = None
            state.pending_army_move_from = None
            state.pending_army_move_side = None
            state.pending_army_move_preexist = None
        elif state.pending_muster_die is not None:
            _consume_die_for_face(
                state, state.pending_muster_side,
                state.pending_muster_die,
                (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.WILL_OF_WEST))
            state.pending_muster_die = None
            state.pending_muster_first_type = None
            state.pending_muster_region = None
            state.pending_muster_side = None
        elif state.pending_combat is not None:
            # Pass during combat clears pending combat
            _restore_rearguard(state, state.pending_combat)
            state.pending_combat = None
            _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw
        elif state.pending_hunt is not None:
            # Pass during hunt: apply damage directly
            dmg = state.pending_hunt.get('damage', 0)
            if dmg > 0:
                game_logic._apply_fellowship_damage(state, dmg)
            state._pending_hunt_damage = 0
            state.pending_hunt = None
        elif state.pending_guide_change is not None:
            state.pending_guide_change = None
        else:
            _consume_die(state, die_idx)

    elif atype == 'army_move':
        from_key = p['from']
        to_key = p['to']
        # card-granted follow-up bookkeeping (RWA: up to 2 moves OR 1
        # attack; BCC: one move/attack with the WK's army)
        _pcf9 = getattr(state, 'pending_card_followup', None)
        if p.get('card_followup') and _pcf9 is not None:
            # rebind, never add in place: the set may still be shared
            # with a parent state
            _pcf9['exclude'] = (set(_pcf9.get('exclude') or ())
                                | {sanitize(str(to_key))})
            _pcf9['moved_any'] = True
            _pcf9['moves_left'] = _pcf9.get('moves_left', 1) - 1
            if _pcf9['moves_left'] <= 0 or _pcf9['kind'] == 'wk_army':
                state.pending_card_followup = None

        # Mouth of Sauron ability: this army action rides a Muster die
        # (once per turn while the Mouth is in play)
        if p.get('mouth_muster'):
            state.sh_mouth_muster_used_this_turn = True

        # Track army as moved this turn (rulebook: each army once per turn)
        # Mark both source and destination — army is the same whether it's at
        # the old or new location

        # Split support (rulebook p.25): n_regular/n_elite/take_leaders params
        # select a portion of the army; absent params move the whole stack.
        def _split_kwargs():
            _nn9 = p.get('n_nazgul')
            _nl9 = p.get('n_leaders')
            if p.get('n_regular') is None and p.get('n_elite') is None \
                    and _nn9 is None and _nl9 is None:
                return {}
            _army = next((a for k, a in state.armies.items()
                          if sanitize(k) == sanitize(from_key)), None)
            if _army is None:
                return {}
            n_r = sum(1 for u in _army.units if u.kind == UnitKind.REGULAR)
            n_e = sum(1 for u in _army.units if u.kind == UnitKind.ELITE)
            mr = min(int(p.get('n_regular') or 0), n_r)
            me = min(int(p.get('n_elite') or 0), n_e)
            if mr >= n_r and me >= n_e and _nn9 is None and _nl9 is None:
                return {}  # whole stack — simple merge path
            take = p.get('take_leaders', True)
            # C (1.211): an explicit n_leaders count (teacher leader-PARTIAL
            # split) pins the non-Nazgul leader figures precisely - an int,
            # never None, so take() cannot fall to its None->0 stranding
            # path.  Only the teacher sets n_leaders; self-play never does.
            _ml9 = (int(_nl9) if _nl9 is not None
                    else (None if take else 0))
            if _nn9 is not None:
                # USER 2026-07-11: explicit nazgul count; plain leaders
                # follow n_leaders (or take_leaders when unset)
                return {'move_units': mr, 'move_elite': me,
                        'move_leaders': _ml9,
                        'move_nazgul': int(_nn9)}
            if _nl9 is not None:
                return {'move_units': mr, 'move_elite': me,
                        'move_leaders': int(_nl9)}
            return {'move_units': mr, 'move_elite': me,
                    'move_leaders': len(_army.leaders) if take else 0}

        mv_kwargs = _split_kwargs()
        # Acting side: from the enumerated action's tag when present —
        # the units[0] fallback misattributes shared besieged entries
        # (both sides in one army entry) and then consumes the WRONG
        # side's die and units.
        _side_tag = p.get('side')
        if _side_tag in ('fp', 'sh'):
            _act_side = Side.FREE_PEOPLES if _side_tag == 'fp' \
                else Side.SHADOW
        else:
            _a0 = _armies_lookup(state, from_key)
            _act_side = _side_of_nation(_a0.units[0].nation) \
                if (_a0 and _a0.units) else Side.FREE_PEOPLES
        mv_kwargs['side'] = _act_side

        if state.pending_army_move_die is not None:
            # Second move of Army die double-move (MUSTER included for
            # the Mouth of Sauron's muster-as-army ability).  Consume
            # BEFORE moving so a die-accounting failure cannot leave a
            # half-applied move (free-action leak fix, 2026-07-08).
            # fix 86: unless the stored die is ALREADY USED (a teacher
            # forcer settled it directly) — the fallback would burn a
            # fresh die and double-charge the pair.
            _pd86b = state.pending_army_move_die
            _pdice86b = state.dice.get(state.pending_army_move_side, [])
            if not (0 <= _pd86b < len(_pdice86b)
                    and _pdice86b[_pd86b].used):
                _consume_die_for_face(
                    state, state.pending_army_move_side,
                    state.pending_army_move_die,
                    (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY,
                     ActionDieFace.WILL_OF_WEST, ActionDieFace.MUSTER))
            game_logic.move_army(state, from_key, to_key, **mv_kwargs)
            state.pending_army_move_die = None
            state.pending_army_move_from = None
            state.pending_army_move_side = None
            state.pending_army_move_preexist = None
        else:
            # Determine die face to decide if this is an Army die (2 moves)
            # or Character die (1 move)
            side = _act_side
            face = _get_die_face(state, die_idx, side=side)

            if face in (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY,
                        ActionDieFace.WILL_OF_WEST) \
                    or (face == ActionDieFace.MUSTER
                        and p.get('mouth_muster')):
                # Army die mode: first of 2 moves, set pending.
                # Rulebook p.25: "no figure is moved twice using the same
                # action" — the second move must exclude the units that just
                # moved (now at the DESTINATION). Units left behind by a
                # split at the origin are a different army and MAY move —
                # and so may units that were ALREADY at the destination
                # before the first move arrived (figures, not regions):
                # snapshot the pre-existing counts for the second half.
                _pre = next((a for k, a in state.armies.items()
                             if sanitize(k) == sanitize(to_key)), None)
                _pre_own = [u for u in (_pre.units if _pre else [])
                            if _side_of_nation(u.nation) == side]
                state.pending_army_move_preexist = {
                    'regular': sum(1 for u in _pre_own
                                   if u.kind == UnitKind.REGULAR),
                    'elite': sum(1 for u in _pre_own
                                 if u.kind == UnitKind.ELITE),
                    'leaders': len(_pre.leaders) if _pre else 0,
                }
                game_logic.move_army(state, from_key, to_key, **mv_kwargs)
                state.pending_army_move_die = die_idx
                state.pending_army_move_from = to_key
                state.pending_army_move_side = side
            else:
                # Character die: single move, consume immediately
                _consume_die_for_face(state, side, die_idx,
                                      (ActionDieFace.CHARACTER,
                                       ActionDieFace.WILL_OF_WEST))
                game_logic.move_army(state, from_key, to_key, **mv_kwargs)
        # Walking into an undefended enemy settlement CAPTURES it
        # (rulebook: enemy-controlled settlements free of enemy units are
        # valid move destinations — and occupying one takes control).
        # SIDE-AWARE (2026-07-04): the capturing side is the MOVER
        # (_act_side), never _occ.units[0] — a shared entry with an enemy
        # residue unit sitting first credited the WRONG side (game 2
        # T9/T10: Shadow moves into Morannon / Minas Morgul / Umbar were
        # recorded as FP captures -> phantom FP military victory at 6
        # VP).  And walking in only captures when NO enemy units remain
        # at the destination (otherwise it is a siege/assault situation;
        # combat resolution handles those captures).
        try:
            from combat import _check_settlement_capture
            _occ = next((a for k, a in state.armies.items()
                         if sanitize(k) == sanitize(to_key)), None)
            if _occ and _occ.units:
                _enemy_here = any(
                    _side_of_nation(u.nation) != _act_side
                    for u in _occ.units)
                if not _enemy_here:
                    _check_settlement_capture(state, sanitize(to_key),
                                              _act_side)
        except Exception:
            pass
        # Siege lift after a move is settled by game_logic.cleanup_sieges
        # in the post-action block (the one implementation; a besieged
        # flag lifts when one side no longer holds the region).

    elif atype == 'army_attack':
        from_key = p['from']
        to_key = p['to']
        # METROLOGY 2026-07-27: heal any stale conquest flag BEFORE the
        # battle forms — a non-combat reoccupation (card recruit,
        # retreat) left enemy_controlled stale, and once the region
        # went two-sided the standing invariant skipped it forever
        # (garrison_side_for then mis-answered for the whole siege).
        # Self-play only; the teacher keeps its certified call pattern.
        try:
            game_logic.enforce_settlement_control(state)
        except Exception:
            pass
        if p.get('card_followup'):
            state.pending_card_followup = None  # attack ends the follow-up

        # Mouth of Sauron ability: this attack rides a Muster die
        # (once per turn while the Mouth is in play)
        if p.get('mouth_muster'):
            state.sh_mouth_muster_used_this_turn = True

        army = _armies_lookup(state, from_key)
        if army is None:
            army = next((a for k, a in state.armies.items()
                         if sanitize(k) == sanitize(from_key)), None)
        if army and army.units:
            _p_side9 = p.get('side')
            if _p_side9 in ('fp', 'sh'):
                # 2026-07-10 sortie fix: the enumeration stamped the
                # acting side - never infer (the garrison-opposite
                # inference misattributed garrison SORTIES to the
                # besieger and consumed the wrong side's dice)
                side = (Side.FREE_PEOPLES if _p_side9 == 'fp'
                        else Side.SHADOW)
            else:
                # legacy/teacher actions without the stamp:
                # membership-aware inference - a co-located siege entry
                # (from == to) holds BOTH sides; assume the assault
                # direction (attacker opposes the garrison)
                _sides_here = {_side_of_nation(u.nation)
                               for u in army.units}
                if len(_sides_here) == 1:
                    side = next(iter(_sides_here))
                else:
                    from combat import garrison_side_for
                    _gar = garrison_side_for(state, to_key)
                    side = (Side.SHADOW if _gar == Side.FREE_PEOPLES
                            else Side.FREE_PEOPLES)
            _attack_faces = (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY,
                             ActionDieFace.CHARACTER,
                             ActionDieFace.WILL_OF_WEST)
            if p.get('mouth_muster'):
                # Mouth ability: the attack legitimately rides a Muster die
                _attack_faces = _attack_faces + (ActionDieFace.MUSTER,)
            _consume_die_for_face(state, side, die_idx, _attack_faces)
        else:
            # No army found: still honor the enumeration's side stamp —
            # a stale 'sh' attack must not form a pending_combat with
            # attacker_side=FP (every combat decision would route to
            # the wrong side's net).
            _p_side9 = p.get('side')
            side = (Side.SHADOW if _p_side9 == 'sh'
                    else Side.FREE_PEOPLES)
        # Track region attacked

        # Stronghold region? (defender MAY withdraw into a siege there.)
        # An attack is a siege from the start only when the defenders are
        # ALREADY besieged inside the stronghold.
        stronghold_region = _is_siege_battle(state, to_key)
        already_besieged = sanitize(to_key) in getattr(
            state, 'besieged_strongholds', set())
        # SORTIE (rulebook p.32): the garrison attacking its besiegers
        # fights a FIELD battle - no 6+ assault threshold.  Same for a
        # RELIEF attack from an adjacent region (GAME19 T8 fix 75): when
        # the attacker is the garrison's own side, the hostile units are
        # the BESIEGERS standing in the region - a field battle against
        # them, never an assault on the (friendly) stronghold.  Before
        # this gate the from!=to relief case inherited is_siege=True and
        # parked forever at attacker_siege_decision (zero rounds).
        _is_sortie9 = False
        if already_besieged:
            from combat import garrison_side_for as _gsf9
            _is_sortie9 = _gsf9(state, to_key) == side
        def_army = _armies_lookup(state, to_key)
        if def_army and def_army.units:
            _def_sides = {_side_of_nation(u.nation) for u in def_army.units}
            if len(_def_sides) == 1:
                def_side = next(iter(_def_sides))
            else:
                def_side = (Side.SHADOW if side == Side.FREE_PEOPLES
                            else Side.FREE_PEOPLES)
        else:
            def_side = (Side.SHADOW if side == Side.FREE_PEOPLES
                        else Side.FREE_PEOPLES)

        # Set up pending_combat for multi-phase resolution.
        # USER RULE 2026-07-08: the REARGUARD decision is REMOVED — humans
        # go hundreds of games without leaving one; the phase routes
        # straight to the post-rearguard flow (siege choice / field round).
        state.pending_combat = {
            'attacker_region': from_key,
            'defender_region': to_key,
            'is_siege': already_besieged and not _is_sortie9,
            'stronghold_region': stronghold_region,
            'phase': 'attacker_rearguard',  # transitioned immediately below
            'deciding_side': side,
            'attacker_side': side,
            'defender_side': def_side,
            'rounds_fought': 0,
        }
        # Rulebook p.35: the defending nation activates + advances when
        # the attack is DECLARED (fires even for sieges established
        # without an assault round)
        game_logic.advance_defenders_political(state, to_key,
                                               attacker_side=side)
        _post_rearguard_transition(state, state.pending_combat)

    elif atype == 'victor_advance':
        pc = state.pending_combat
        state.pending_combat = None
        _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw
        if pc:
            _vside9 = (Side.FREE_PEOPLES if p.get('side') == 'fp'
                       else Side.SHADOW)
            try:
                # 1.150: move_army's take() treats qty=None as ZERO
                # when unit counts are supplied - 'None means all' was
                # a silent no-op and every advance stranded all
                # leadership at the attack region
                game_logic.move_army(
                    state, p['from'], p['to'],
                    move_units=int(p.get('n_regular') or 0),
                    move_elite=int(p.get('n_elite') or 0),
                    move_leaders=(99 if p.get('take_leaders') else 0),
                    move_nazgul=(int(p['n_nazgul'])
                                 if p.get('n_nazgul') is not None
                                 else (99 if p.get('take_leaders')
                                       else 0)),
                    side=_vside9)
            except Exception:
                if pc.get('post_retreat'):
                    raise   # live/learned path: a failed advance is LOUD
                pass  # stacking or lookup issue — victor stays
            else:
                if pc.get('post_retreat'):
                    # entering the region the defender vacated CAPTURES
                    # its settlement (a won battle registers the
                    # capture at the wipe; a retreat only on entry)
                    from combat import _check_settlement_capture
                    _check_settlement_capture(
                        state, sanitize(str(p['to'])), _vside9)
                # USER RULING 2026-07-27: the advance may overstack
                # (relieve-and-merge play) — trim is the learned
                # decision once the move ends
                _register_move_overstack(state, p['to'], _vside9)

    elif atype == 'victor_stay':
        state.pending_combat = None
        _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'followup_done':
        state.pending_card_followup = None

    elif atype == 'rearguard_none':
        # Attacker commits full army
        pc = state.pending_combat
        if pc:
            _post_rearguard_transition(state, pc)

    elif atype == 'combat_no_retreat':
        _pcx = state.pending_combat
        if _pcx is not None:
            _pcx.pop('_card_retreat_mandatory', None)
        # Defender stands and fights
        pc = state.pending_combat
        if pc:
            if pc['is_siege']:
                # Defender stays in stronghold — attacker decides
                pc['phase'] = 'attacker_siege_decision'
                pc['deciding_side'] = pc['attacker_side']
            else:
                # Field battle: fight a round
                _run_combat_round(state, pc)

    elif atype == 'retreat_into_siege':
        # Stronghold defender withdraws into the stronghold (rulebook p.30:
        # allowed at the beginning of any combat round). Max 5 units inside;
        # excess units are removed.
        pc = state.pending_combat
        if pc:
            def_san = sanitize(pc['defender_region'])
            if not hasattr(state, 'besieged_strongholds'):
                state.besieged_strongholds = set()
            state.besieged_strongholds.add(def_san)
            pc['is_siege'] = True
            d_army = _armies_lookup(state, pc['defender_region'])
            # A CO-LOCATED entry - Corsairs of Umbar lands the attacking
            # Shadow army INTO the region BEFORE the FP withdraws (almanac
            # A10) - holds BOTH sides.  The 5-unit siege cap (rulebook p.31)
            # binds only the GARRISON (defender-side); the besiegers must
            # not be touched.  Every OTHER caller reaches this with the
            # attacker in a separate region, so d_army is garrison-only and
            # the pre-existing single-side branch below runs byte-identical.
            _dside = pc.get('defender_side')
            _others = [u for u in (d_army.units if d_army else [])
                       if _dside is not None
                       and _side_of_nation(u.nation) != _dside]
            # USER RULING 2026-07-20: withdrawal excess returns to
            # REINFORCEMENTS, never casualties (untracked removal =
            # computed-pool return, same doctrine as
            # resolve_overstacks). WHICH units return is the OWNER'S
            # choice: on the self-play path a 'siege_trim' learned
            # decision opens below; the teacher path auto-trims
            # elites-kept (the log's own piece rows are ground truth
            # — game0 T9: the human removed 2 Gondor Regulars).
            _gar = [u for u in (d_army.units if d_army else [])
                    if _dside is None
                    or _side_of_nation(u.nation) == _dside]
            _exc = len(_gar) - game_logic.MAX_SIEGE_UNITS
            _trim_pending = False
            if _exc > 0:
                pc['phase'] = 'siege_trim'
                pc['deciding_side'] = _dside
                pc['trim_region'] = def_san
                _trim_pending = True
            # iter 36 (rulebook p.31): the attacking army MOVES INTO the
            # region — the siege is maintained by the co-located besieger
            # (the engine's from==to assault model).  Without this the
            # attacker stayed adjacent forever and cleanup_sieges would
            # (correctly) dissolve the siege it just created.
            # (fix-92 EXPERIMENT REVERTED 2026-07-25: unconditional
            # move-in cured nothing for game 20 — its Isengard unit
            # diverges upstream — and broke game 16 by dragging leaders
            # the humans left behind; game 20's deduped arrival drags
            # remain an open parser-side item, see rules_amendments.)
            if sanitize(pc['attacker_region']) != def_san:
                try:
                    game_logic.move_army(state, pc['attacker_region'],
                                         pc['defender_region'])
                    pc['attacker_region'] = pc['defender_region']
                except Exception:
                    pass  # stacking or lookup issue: attacker stays put
            # iter 38 SIEGE TWO-STEP (user ruling + rulebook p.30: "If
            # the defending Army retreats into a Siege, the battle ends
            # immediately"): the attack die bought the declaration; the
            # defender's withdrawal ENDS the action — no combat round
            # against the besieged army on this die.  Damaging units
            # inside the stronghold requires a SEPARATE siege-assault
            # action (its own die) later.  The old flow continued into
            # 'attacker_siege_decision' and could resolve assault rounds
            # within the same action.
            # (USER RULING 2026-07-20: if a siege_trim decision opened
            # above, the battle end waits for it — its executor runs
            # the restore/clear.)
            if not _trim_pending:
                _restore_rearguard(state, pc)
                state.pending_combat = None
                _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'siege_trim':
        # USER RULING 2026-07-20: the withdrawing owner chose how many
        # ELITES (k) go back to reinforcements; the rest of the excess
        # are regulars. Untracked removal = pool return (computed
        # reinforcements), NEVER casualties.
        pc = state.pending_combat
        if pc and pc.get('phase') == 'siege_trim':
            _rsan = pc.get('trim_region')
            _dside = pc.get('defender_side')
            d_army = _armies_lookup(state, _rsan)
            if d_army is not None:
                _gar = [u for u in d_army.units
                        if _side_of_nation(u.nation) == _dside]
                _exc = len(_gar) - game_logic.MAX_SIEGE_UNITS
                if _exc > 0:
                    k = int(p.get('elites_back', 0) or 0)
                    elites = [u for u in _gar
                              if u.kind == UnitKind.ELITE]
                    regs = [u for u in _gar
                            if u.kind != UnitKind.ELITE]
                    k = max(0, min(k, len(elites), _exc))
                    back = elites[:k] + regs[:_exc - k]
                    if len(back) < _exc:   # not enough regulars
                        back += elites[k:k + (_exc - len(back))]
                    _bk = set(map(id, back))
                    d_army.units = [u for u in d_army.units
                                    if id(u) not in _bk]
            _restore_rearguard(state, pc)
            state.pending_combat = None
            _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'combat_cease':
        # Attacker ceases the attack — surviving units stay where they were
        pc = state.pending_combat
        if pc:
            _restore_rearguard(state, pc)
        state.pending_combat = None
        _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'combat_retreat':
        # Army retreats to chosen region
        pc = state.pending_combat
        if pc:
            to_region = p.get('to_region')
            retreating_side = pc['deciding_side']
            _advance_pending = None
            if retreating_side == pc['defender_side']:
                # Defender retreats; the attacker's advance is OPTIONAL
                # and a LEARNED decision (R6): the existing
                # victor_advance pending (composition grid + stay)
                # opens for the attacker, exactly as after a won
                # battle.  SANITIZED lookups: pc regions are sanitized
                # keys while parser/engine-created entries may carry
                # original casing.
                _retreat_army(state, pc['defender_region'], to_region,
                              side=pc.get('defender_side'))
                _atk_k = next((k for k in state.armies if sanitize(str(k))
                               == sanitize(str(pc['attacker_region']))),
                              pc['attacker_region'])
                _def_k = next((k for k in state.armies if sanitize(str(k))
                               == sanitize(str(pc['defender_region']))),
                              pc['defender_region'])
                atk_army = _armies_lookup(state, _atk_k)
                if atk_army and atk_army.units:
                    from combat import _check_settlement_capture
                    if sanitize(str(_atk_k)) == sanitize(str(_def_k)):
                        # CO-LOCATED (Corsairs of Umbar A10): the attacker
                        # already stands in the region the defender just
                        # vacated - no move to decide; the abandoned
                        # settlement is captured.
                        try:
                            _check_settlement_capture(
                                state, sanitize(pc['defender_region']),
                                pc['attacker_side'])
                        except Exception:
                            pass
                    else:
                        _advance_pending = {
                            'phase': 'victor_advance',
                            'attacker_region': _atk_k,
                            'defender_region': _def_k,
                            'attacker_side': pc['attacker_side'],
                            'defender_side': pc['defender_side'],
                            'deciding_side': pc['attacker_side'],
                            'is_siege': False,
                            'rounds_fought': pc.get('rounds_fought', 0),
                            # the victor_advance executor captures the
                            # vacated settlement on entry
                            'post_retreat': True,
                        }
            else:
                # Attacker retreats — defender keeps region
                _retreat_army(state, pc['attacker_region'], to_region,
                              side=pc.get('attacker_side'))
            _resolve_wk_sorcerer(state, pc)   # Scouts pre-round retreat
            _restore_rearguard(state, pc)
            state.pending_combat = None
            # USER RULING 2026-07-27: a retreat merge may overstack —
            # open the learned trim now that the combat pending is down
            _register_move_overstack(state, to_region, retreating_side)
            if _advance_pending is not None:
                # the attacker decides next; a retreat-merge trim pending
                # (card_option) is served once this pending closes
                state.pending_combat = _advance_pending
            else:
                _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'siege_assault':
        # Attacker assaults the stronghold
        pc = state.pending_combat
        if pc:
            pc['is_siege'] = True
            _run_combat_round(state, pc, is_siege=True)

    elif atype == 'siege_continue':
        # Attacker maintains siege without combat — stronghold is now besieged
        pc = state.pending_combat
        if pc:
            if not hasattr(state, 'besieged_strongholds'):
                state.besieged_strongholds = set()
            state.besieged_strongholds.add(sanitize(pc['defender_region']))
            _resolve_wk_sorcerer(state, pc)   # no-assault termination
            _restore_rearguard(state, pc)
        state.pending_combat = None
        _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'combat_continue':
        # Field battle: ATTACKER chooses to continue (rulebook p.29) — the
        # DEFENDER then gets the retreat window before the next round.
        pc = state.pending_combat
        if pc:
            pc['phase'] = 'defender_retreat'
            pc['deciding_side'] = pc['defender_side']

    elif atype == 'siege_press':
        # Attacker presses siege (downgrade 1 elite). The co-located siege
        # entry holds BOTH sides' units (garrison first), so downgrade an
        # elite of the ATTACKER's side specifically — siege_press() on the
        # shared entry hit the garrison's elite instead.
        pc = state.pending_combat
        if pc:
            _atk_san = sanitize(pc['attacker_region'])
            atk_army = next((a for k, a in state.armies.items()
                             if sanitize(k) == _atk_san), None)
            # no_reduce (2025-04-06 demanray T19 Umbar): a teacher-
            # synthesized siege press that carries NO log press evidence
            # (no siege_press record, no attacker downgrade mark — the
            # extra siege rounds were fought on CASUALTIES, not paid
            # reductions) advances the round WITHOUT reducing an elite.
            # The parser recorded the attacker's per-round elite losses as
            # kills (kill_elite) and the client census shows the survivors
            # keeping their Elite rank; a press-downgrade here double-
            # removes those elites (4E -> 0E instead of the true 2E).
            if atk_army and not p.get('no_reduce'):
                atk_side_pc = pc.get('attacker_side')
                # fix 76 (GAME19 T8): the parser stamps the press record
                # with the PAID elite's nation (log evidence) — prefer
                # that nation's elite; first-in-list order paid with the
                # wrong nation whenever the attacker mixed elites.
                _want_nat9 = p.get('nation')
                _want_nat9 = (str(_want_nat9).lower()
                              if _want_nat9 is not None else None)
                _press_i9 = None
                for _i9, _u9 in enumerate(atk_army.units):
                    if _u9.kind == UnitKind.ELITE and (
                            atk_side_pc is None
                            or _side_of_nation(_u9.nation) == atk_side_pc):
                        if _press_i9 is None:
                            _press_i9 = _i9
                        if (_want_nat9 is None
                                or str(getattr(_u9.nation, 'value',
                                               _u9.nation)).lower()
                                == _want_nat9):
                            _press_i9 = _i9
                            break
                if _press_i9 is not None:
                    i, u = _press_i9, atk_army.units[_press_i9]
                    # ONE downgrade-supply answerer (R7 / K6):
                    # box -> pool -> none, stamped on _dg_src_log; no
                    # replacement = the elite is ELIMINATED outright.
                    from combat import (
                        _consume_downgrade_source as _cds8,
                        _track_fp_casualties as _tfc8)
                    _nat8 = u.nation
                    _src8 = _cds8(state, _nat8, {})
                    if _src8 is None:
                        atk_army.units.pop(i)
                    else:
                        atk_army.units[i] = Unit(_nat8,
                                                 UnitKind.REGULAR)
                    if _side_of_nation(_nat8) == Side.FREE_PEOPLES:
                        # the paid FP elite is a permanent casualty
                        _tfc8(state, [u], Side.FREE_PEOPLES)
            _run_combat_round(state, pc, is_siege=True)

    elif atype == 'siege_no_press':
        # Attacker stops siege pressing
        if state.pending_combat:
            _restore_rearguard(state, state.pending_combat)
        state.pending_combat = None
        _maybe_open_palantir_draw(state)   # USER 1.208m deferred draw

    elif atype == 'discard_card':
        # Hand-limit discard: the owner chose which card to give up
        # (USER RULING 2026-08-15 — a real learned decision).  The
        # removal, the face-down pile and the one-at-a-time pending
        # re-derivation all live in the shared primitive so the GUI and
        # the extractor behave identically.
        d_side = (Side.FREE_PEOPLES if p.get('side') == 'fp'
                  else Side.SHADOW)
        cid = p.get('card_id')
        if not game_logic.discard_hand_card(state, d_side, cid):
            raise RuntimeError(
                f"discard_card {cid!r} not in {d_side.name} hand")

    elif atype == 'casualty_allocate':
        pc = state.pending_combat
        if pc:
            stage = pc.get('alloc_stage')
            if stage == 'attacker':
                _reg9, _side9 = pc['attacker_region'], pc['attacker_side']
                _hits9 = int(pc.get('atk_hits', 0) or 0)
            else:
                _reg9, _side9 = pc['defender_region'], pc['defender_side']
                _hits9 = int(pc.get('def_hits', 0) or 0)
            _apply_allocation(state, _reg9, _side9, _hits9,
                              int(p.get('count', 0) or 0))
            pc['alloc_stage'] = ('attacker' if stage == 'defender'
                                 else None)
            _enter_casualty_allocation(state, pc)

    elif atype in ('casualty_preserve_units', 'casualty_preserve_elites'):
        # Owner locks in casualty allocation for this battle, then combat
        # resumes (next policy decision or the round itself)
        pc = state.pending_combat
        if pc:
            pol = ('preserve_units' if atype == 'casualty_preserve_units'
                   else 'preserve_elites')
            if pc.get('phase') == 'attacker_casualty_policy':
                pc['atk_casualty_policy'] = pol
            else:
                pc['def_casualty_policy'] = pol
            _run_combat_round(state, pc)

    elif atype in ('shine', 'shine_none'):
        # Gandalf 'shining' decision (item 11): record the FP choice, then
        # resume into the combat-card phases.  gtw_shine feeds setup_combat
        # (cancel all Nazgul leadership + forfeit Gandalf's own Leadership).
        pc = state.pending_combat
        if pc:
            pc['gtw_shine'] = (atype == 'shine')
            pc['shine_asked'] = True
            _run_combat_round(state, pc)

    elif atype in ('combat_card_play', 'combat_card_none'):
        # Per-round combat-card commitment; combat resumes (defender's
        # commitment next, or the round itself)
        pc = state.pending_combat
        if pc:
            cid = p.get('card_id') if atype == 'combat_card_play' else None
            _role9 = ('atk' if pc.get('phase') == 'attacker_combat_card'
                      else 'def')
            _commit_combat_card(state, pc, _role9, cid,
                                bool(p.get('from_table')))
            _run_combat_round(state, pc)

    elif atype == 'combat_effect_choice':
        # Choice-bearing combat-card effect magnitude. Pre-roll effects
        # (Dread and Despair / Relentless Assault / Heroic Death) store
        # the count and resume the round; the post-casualty Onslaught
        # applies its sacrifice + extra rolls now, then continues to the
        # normal post-round decisions.
        pc = state.pending_combat
        if pc:
            ec = pc.pop('effect_choice', None) or {}
            count = int(p.get('count', 0) or 0)
            owner = ec.get('owner', 'atk')
            if ec.get('stage') == 'post':
                from combat import apply_onslaught_choice
                role = 'attacker' if owner == 'atk' else 'defender'
                other = 'defender' if role == 'attacker' else 'attacker'
                own_dead, enemy_dead, rolls, hits = apply_onslaught_choice(
                    state,
                    pc[f'{role}_region'], pc[f'{other}_region'],
                    pc[f'{role}_side'], pc[f'{other}_side'], count,
                    own_policy=pc.get(
                        'atk_casualty_policy' if owner == 'atk'
                        else 'def_casualty_policy', 'preserve_elites'),
                    enemy_policy=pc.get(
                        'def_casualty_policy' if owner == 'atk'
                        else 'atk_casualty_policy', 'preserve_elites'),
                    enemy_is_defender=(other == 'defender'))
                # fix 78: the round that opened this phase may already
                # have destroyed an army (stashed flags) — combine
                _pf_atk78 = pc.pop('_finish_atk_dead', False)
                _pf_def78 = pc.pop('_finish_def_dead', False)
                if own_dead or enemy_dead or _pf_atk78 or _pf_def78:
                    atk_dead = (own_dead if role == 'attacker'
                                else enemy_dead) or _pf_atk78
                    def_dead = (enemy_dead if role == 'attacker'
                                else own_dead) or _pf_def78
                    _finish_battle(state, pc, atk_dead, def_dead)
                else:
                    _set_post_round_phase(state, pc,
                                          pc.get('is_siege', False))
            else:
                pc[f'{owner}_effect_count'] = count
                _run_combat_round(state, pc)

    elif atype == 'advance_political':
        nation = p['nation']
        side = _side_of_nation(nation)
        # QA 2026-07-11 (T&P CASE 7): a COMPANION-ability advance may
        # use ANY Action die (Boromir/Legolas/Gimli rule) - the
        # Muster-family consumption raised on a Character die
        _adv_faces = (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                      ActionDieFace.WILL_OF_WEST)
        if p.get('companion_ability'):
            _adv_faces = (ActionDieFace.CHARACTER, ActionDieFace.ARMY,
                          ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                          ActionDieFace.EVENT, ActionDieFace.WILL_OF_WEST)
        _consume_die_for_face(state, side, die_idx, _adv_faces)
        # QA 2026-07-10 round 2: a companion-ability advance must carry
        # its method - Threats and Promises discards on it and must not
        # block it as a muster-die advance of a passive nation
        game_logic.advance_political(
            state, nation,
            method='companion_ability' if p.get('companion_ability')
            else 'muster_die')

    elif atype == 'discard_table_card':
        # 1.149 (USER 2026-07-19): deliberate kill of an opponent's
        # play-on-table card.  SH: any die + discard 1 Strategy + 1
        # Character card from hand (A Power too Great / Tom Bombadil).
        # FP: WotW die alone, or any die + one Elven Ring (Palantír —
        # a spent ring flips to its Flame side and goes to Shadow).
        _any_faces = (ActionDieFace.CHARACTER, ActionDieFace.ARMY,
                      ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                      ActionDieFace.EVENT, ActionDieFace.EYE,
                      ActionDieFace.WILL_OF_WEST)
        _kill_side = (Side.SHADOW
                      if str(p.get('card_id', '')).startswith('FREE')
                      else Side.FREE_PEOPLES)
        if _kill_side == Side.SHADOW:
            _consume_die_for_face(state, Side.SHADOW, die_idx,
                                  _any_faces)
            # payment = one ARMY card (card_type 'Army') + one
            # Character card (USER 2026-07-19 correction: Muster-type
            # Strategy cards cannot pay)
            from cards import get_card as _gc149x
            _hand9 = state.hands[Side.SHADOW].hand
            # CARD CONSERVATION (2026-08-15): the two payment cards are
            # DISCARDED, not annihilated - they used to leave the hand
            # and reach no pile at all, deleting them from the game.
            for _c9 in list(_hand9):
                if isinstance(_c9, str) and getattr(
                        _gc149x(_c9), 'card_type', None) == 'Army':
                    _hand9.remove(_c9)
                    _discard_played_combat_card(state, _c9)
                    break
            for _c9 in list(_hand9):
                if isinstance(_c9, str) and '_Character_' in _c9:
                    _hand9.remove(_c9)
                    _discard_played_combat_card(state, _c9)
                    break
        else:
            if p.get('payment') == 'will':
                _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                                      (ActionDieFace.WILL_OF_WEST,))
            elif p.get('payment') == 'character':
                # Denethor's Folly any-die path (P2-03 ruling
                # 2026-07-20): Aragorn/Gandalf in Minas Tirith, no
                # ring cost
                _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                                      _any_faces)
            else:
                _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                                      _any_faces)
                if state.fp_elven_rings > 0:
                    state.fp_elven_rings -= 1
                    state.sh_elven_rings += 1
        _pc9 = getattr(state, 'persistent_cards', None)
        if _pc9 is not None and p.get('card_id'):
            # CARD CONSERVATION (2026-08-15): a table card killed by
            # this action is physically DISCARDED face-up, not deleted
            # (auto35 live game lost The Palantir of Orthanc here).
            from persistent_effects import discard_table_card as _dtc9
            _dtc9(state, _pc9.remove_card(p['card_id']))

    elif atype == 'muster_recruit':
        nation = p['nation']
        region = p['region']
        typ = p['type']
        side = _side_of_nation(nation)

        if state.pending_muster_die is not None:
            # Second recruit of Muster die double-recruit
            game_logic._muster_one(state, side, region, typ, nation=nation)
            _consume_die_for_face(
                state, state.pending_muster_side,
                state.pending_muster_die,
                (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                 ActionDieFace.WILL_OF_WEST))
            state.pending_muster_die = None
            state.pending_muster_first_type = None
            state.pending_muster_region = None
            state.pending_muster_side = None
        else:
            # First recruit: set pending for second recruit
            game_logic._muster_one(state, side, region, typ, nation=nation)
            state.pending_muster_die = die_idx
            state.pending_muster_first_type = typ
            state.pending_muster_region = region
            state.pending_muster_side = side

    elif atype == 'muster_elite':
        # Place one new elite unit (consumes full die, no second recruit)
        nation = p['nation']
        region = p['region']
        side = _side_of_nation(nation)
        _consume_die_for_face(state, side, die_idx,
                              (ActionDieFace.MUSTER, ActionDieFace.MUSTER_ARMY,
                               ActionDieFace.WILL_OF_WEST))
        game_logic._muster_one(state, side, region, 'elite', nation=nation)

    elif atype == 'fellowship_move':
        # P1 (USER ruling 1.112): with Flocks of Crebain on the table
        # and hunt dice about to roll, the SHADOW first decides whether
        # to discard it for +1 to all Hunt dice - a learned cross-turn
        # pending; the move itself is stashed and resumes after the
        # decision (resume_action re-dispatch in the card_option
        # executor).  Mordor hunts draw tiles without dice - no choice.
        if not p.get('_crebain_resolved'):
            from game_state import PersistentCardType as _PCT9
            _pc9 = getattr(state, 'persistent_cards', None)
            _cb9 = (_pc9.get_by_type(_PCT9.FLOCKS_OF_CREBAIN)
                    if _pc9 else None)
            if (_cb9 is not None
                    and not state.fellowship.mordor_entered
                    and any(d.side is Side.SHADOW
                            for d in state.hunt_box)):
                state.pending_card_option = {
                    'card_id': getattr(_cb9, 'card_id', '__crebain__'),
                    'card_name': 'Flocks of Crebain',
                    'side': 'sh', 'decider': 'sh', 'direct': True,
                    'options': [{'crebain_discard': True},
                                {'crebain_discard': False}],
                    'resume_action': {
                        'die_idx': die_idx,
                        'action_type': 'fellowship_move',
                        'params': dict(p, _crebain_resolved=True)}}
                return
        _fm_used_idx = _consume_die_for_face(
            state, Side.FREE_PEOPLES, die_idx,
            (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST))
        # Move fellowship with hunt — skip auto-sacrifice so AI can decide.
        # Per rulebook: the FP die goes into hunt box AFTER the hunt resolves,
        # not before.  The hunt uses only dice already in the box.
        was_hidden = state.fellowship.hidden
        game_logic.move_fellowship(state, skip_hunt=False,
                                   skip_auto_sacrifice=True)
        # The Last Battle upkeep FIRST (almanac A1 ordering): if this
        # very move revealed the Fellowship, the card is removed before
        # the die placement - so the die DOES go into the Hunt Box as
        # the last step.  BOTH modes (USER directive 2026-07-10).
        _recheck_last_battle(state)
        # NOW add die to hunt box (after hunt resolution).
        # "The Last Battle" persistent card: FP movement dice bypass the box.
        # The die placed in the box must be the die actually CONSUMED
        # (the consume fallback may have marked a different index) —
        # otherwise an unused die lands in the box and stays spendable.
        from persistent_effects import should_add_dice_to_hunt_box
        if should_add_dice_to_hunt_box(state) and _fm_used_idx is not None \
                and _fm_used_idx < len(state.dice[Side.FREE_PEOPLES]):
            d = state.dice[Side.FREE_PEOPLES][_fm_used_idx]
            if d not in state.hunt_box:
                state.hunt_box.append(d)
        # Check for pending hunt damage that needs AI decision.
        # USER RULING 2026-07-21: offer the LEARNED Axe/Horn reduction
        # FIRST (card text order: reduce, then confront); the interpose
        # resumes _settle_hunt_damage from the card_option executor.
        # The settle body (Gollum reveal / sacrifice / apply) is
        # byte-identical, factored into game_logic._settle_hunt_damage.
        if getattr(state, '_pending_hunt_damage', 0) > 0:
            if not game_logic._offer_hunt_reduction(state):
                game_logic._settle_hunt_damage(state)

    elif atype == 'fellowship_declare':
        # Declaration is a free action (die_idx=-1, no die consumed).
        # The AI learns when and where to declare.
        region = p['region']
        _balrog_from9 = sanitize(game_logic.fellowship_region(state))
        _balrog_prog9 = state.fellowship.progress
        try:
            game_logic.declare_fellowship(state, region=region)
        except Exception:
            # TEACHER ONLY: log-truth declares may disagree with the
            # reachability model, so the logged position is forced.
            # Self-play must NEVER force an engine-rejected declare
            # (the raise IS the legality guard: unreachable region,
            # no-move turn, Mordor-interior ban) — propagate so the
            # driver rejects the action.
            import combat as _cmb_fd
            if _cmb_fd.FINAL_HITS_HOOK is None:
                raise
            try:
                state.fellowship.region = state.board.region(region)
                state.fellowship.progress = 0
                state.fellowship.hidden = True
            except Exception:
                pass
        # No Mordor entry here (USER FINAL 2026-08-16): declaring only
        # PLACES the pawn.  Entry resolves in the turn-start sequence's
        # entry check, which runs AFTER the declaration opportunity in
        # the same turn start - never inside this action, never
        # mid-turn.  (The same-action entry variant was built and
        # REVERTED 2026-08-16.)
        # P5 (USER ruling 1.112, almanac C17): a declaration FROM,
        # INTO or THROUGH Moria with 'Balrog of Moria' on the table
        # gives the SHADOW a discard-for-hunt-tile decision (fully
        # after the declaration resolves).  THROUGH = Moria lies on at
        # least one legal declaration path of length <= the
        # pre-declaration progress (BFS over board adjacency - the same
        # reachability model declare_fellowship uses; see
        # _balrog_moria_on_declare_path).  The stationary turn-start
        # trigger is handled at the Phase 2 pre-step.
        import combat as _cmb_balg
        if (_cmb_balg.FINAL_HITS_HOOK is None
                and _balrog_moria_on_declare_path(
                    state, _balrog_from9, sanitize(region),
                    _balrog_prog9)):
            from game_state import PersistentCardType as _PCT9
            _pcb9 = getattr(state, 'persistent_cards', None)
            _cb9 = (_pcb9.get_by_type(_PCT9.BALROG_OF_MORIA)
                    if _pcb9 else None)
            if _cb9 is not None and getattr(state, 'pending_card_option',
                                            None) is None:
                state.pending_card_option = {
                    'card_id': getattr(_cb9, 'card_id', '__balrog__'),
                    'card_name': 'Balrog of Moria',
                    'side': 'sh', 'decider': 'sh', 'direct': True,
                    'options': [{'balrog_declare_draw': True},
                                {'balrog_declare_draw': False}]}
        elif _cmb_balg.FINAL_HITS_HOOK is not None \
                and _balrog_moria_on_declare_path(
                    state, _balrog_from9, sanitize(region),
                    _balrog_prog9):
            # fix 100 (game 11 T5): the P5 declare-trigger had NO
            # teacher branch — the Rizino1 Moria declare's Balrog draw
            # (tiles L339/L359, Legolas guide-sacrifice L360) never
            # fired and the whole chain orphaned.  Log-driven: fire
            # exactly when the evidence gate (fix 98) shows a queued
            # tile belonging to this event.
            from game_state import PersistentCardType as _PCT9
            _pcb9 = getattr(state, 'persistent_cards', None)
            _cb9 = (_pcb9.get_by_type(_PCT9.BALROG_OF_MORIA)
                    if _pcb9 else None)
            if _cb9 is not None and (
                    game_logic.BALROG_TILE_GATE is None
                    or game_logic.BALROG_TILE_GATE(state)):
                from event_effects import _effect_balrog_of_moria
                try:
                    _effect_balrog_of_moria(
                        state, _cb9, {'balrog_declare_draw': True})
                except Exception:
                    pass

    elif atype == 'fellowship_hide':
        if p.get('strider_ability'):
            # Strider's Guide ability: any die result may hide the Fellowship
            _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                                  tuple(ActionDieFace))
        else:
            _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                                  (ActionDieFace.CHARACTER,
                                   ActionDieFace.WILL_OF_WEST))
        state.fellowship.hidden = True
        # Hiding counts as Fellowship activity (rulebook p.39: the Mordor
        # inactivity corruption applies only if the Fellowship neither
        # moved NOR hid during the turn).  Set UNCONDITIONALLY: a hide
        # (or move) BEFORE entering Mordor the same turn still counts —
        # the flag is only read at turn end when in Mordor.
        state.fellowship_acted_in_mordor = True

    elif atype == 'draw_card':
        deck = p['deck']
        side = Side.FREE_PEOPLES if p.get('side', 'fp') == 'fp' else Side.SHADOW
        _consume_die_for_face(state, side, die_idx,
                              (ActionDieFace.EVENT, ActionDieFace.WILL_OF_WEST))
        try:
            game_logic.draw_event_card(state, side, deck)
        except RuntimeError:
            pass  # Deck empty — silently skip
        # The Palantir of Orthanc grants on Event-die event PLAYS only
        # (card text) - a draw never triggers it (play_event branch).

    elif atype == 'play_event':
        card_id = p['card_id']
        side = Side.FREE_PEOPLES if p.get('side', 'fp') == 'fp' else Side.SHADOW
        # PHANTOM-PLAY GUARD (2026-07-20, defense-in-depth behind the
        # tree-reuse own-hand check): a play_event for a card NOT in
        # the real hand must never consume a die — the old flow burned
        # the die first, then play_event_card rejected 'Card not in
        # hand' as a silent no-op (10/23 plays in trace w5_g2).
        # SELF-PLAY only: the teacher replays logged plays (log = truth
        # for hand contents there).
        if card_id not in state.hands[side].hand:
            return
        # Capture the die face + card deck BEFORE consuming the die — needed
        # for the Gandalf-the-Grey / Palantír same-deck redraw below.
        _pe_face = _get_die_face(state, die_idx, side=side) if die_idx >= 0 \
            else None
        try:
            from cards import get_card as _gc_deck
            _pe_deck = (getattr(_gc_deck(card_id), 'deck', '') or '').lower()
        except Exception:
            _pe_deck = ''
        # Snapshot die.used BEFORE consuming the action die: a no-op
        # play (out-of-range dest / absent choice) must refund the
        # action die too, not just dice spent inside the effect.
        _used_before = [d.used for d in state.dice.get(side, [])]
        _consume_die_for_face(state, side, die_idx,
                              (ActionDieFace.EVENT, ActionDieFace.CHARACTER,
                               ActionDieFace.ARMY, ActionDieFace.MUSTER,
                               ActionDieFace.MUSTER_ARMY, ActionDieFace.WILL_OF_WEST))
        _defer = False
        _play_choices = p.get('choices')
        if _play_choices is None and CARD_OPTION_DEFER:
            # AUTO-RESOLVER CENSUS 2026-08-16 (systemic finding): this
            # guard used to be `except Exception: _opts0 = []` - ANY
            # option-spec bug silently converted a learned decision
            # into the card's in-effect fallback heuristic (no halt,
            # no log line).  A spec failure on the LIVE path now halts
            # loudly instead of quietly un-learning the decision; the
            # teacher (log-forced choices) never enters this branch.
            try:
                from cards import get_card as _gc0
                from card_options import get_card_options as _gco
                _cn0 = getattr(_gc0(card_id), 'card_name', '') or ''
                _opts0 = _gco(state, side, _cn0) if _cn0 else []
            except NeedClientCall:
                raise
            except Exception as _ose0:
                import combat as _cmbose
                if _cmbose.FINAL_HITS_HOOK is None:
                    raise RuntimeError(
                        'card option spec FAILED for %r - a learned '
                        'decision would have silently fallen to the '
                        'in-effect heuristic (census 2026-08-16): %r'
                        % (card_id, _ose0))
                _opts0 = []
            if len(_opts0) >= 2:
                # optioned card: the choice is a decision, not a
                # default.  P1 (USER ruling 1.112.3): some cards'
                # choices belong to the OPPONENT (Lure of the Ring:
                # the FP player picks corruption-vs-eliminate) - the
                # decider routes/owns the pending, the card owner
                # still plays the card.
                _own0 = 'fp' if side == Side.FREE_PEOPLES else 'sh'
                try:
                    from card_options import CARD_OPTION_DECIDER as _COD
                    _dec0 = _COD.get(str(_cn0).lower())
                except Exception:
                    _dec0 = None
                if _dec0 == 'opponent':
                    _dec0 = 'sh' if _own0 == 'fp' else 'fp'
                state.pending_card_option = {
                    'card_id': card_id, 'card_name': _cn0,
                    'side': _own0,
                    'decider': _dec0 or _own0,
                    'options': _opts0}
                _defer = True
            elif len(_opts0) == 1:
                # the auto-pick IS the pending path (teacher-supplied
                # choices never reach this branch) - chained stages
                # (Stormcrow stage 2, TSiM moves 2-4) key on the marker
                _play_choices = dict(_opts0[0])
                _play_choices['_from_pending'] = True
        if not _defer:
            # CARD-FLEET LOW-6/LOW-8 (2026-07-21): on a refund restore
            # the pre-consumption die snapshot (taken above) — a play
            # doing nothing must not burn the action die.
            _pe_res = None
            try:
                from event_effects import play_event_card
                _pe_res = play_event_card(state, side, card_id,
                                          choices=_play_choices)
            except Exception as _pe_err:
                # Handler raised: remove the card from hand AND file it
                # to the face-up discard so the hand+deck+discard union
                # multiset stays conserved (the overlay recon depends
                # on it); log loudly — a raise here is a handler bug.
                print(f"[play_event] handler raised for {card_id}: "
                      f"{_pe_err!r}")
                hand = state.hands.get(side, PlayerInfo())
                if card_id in hand.hand:
                    hand.hand.remove(card_id)
                    try:
                        _dfu = getattr(state, 'discard_face_up', None)
                        if isinstance(_dfu, dict):
                            _pfx = ('fp' if side == Side.FREE_PEOPLES
                                    else 'sh')
                            _dk = ('%s_%s' % (_pfx, _pe_deck)
                                   if _pe_deck in ('character', 'strategy')
                                   else '%s_character' % _pfx)
                            _dfu.setdefault(_dk, []).append(card_id)
                    except Exception:
                        pass
            if _pe_res is not None and _pe_res.get("refund"):
                for _d, _u in zip(state.dice.get(side, []),
                                  _used_before):
                    _d.used = _u
        # USER FIX 2026-07-08: the SAME-DECK REDRAW that already worked in
        # the GUI but not the AI.  Gandalf the Grey (guide) draws a card of
        # the SAME DECK when FP plays a card with an Event/Will die; The
        # Palantír of Orthanc draws same-deck each Shadow event play.
        if _pe_deck in ('character', 'strategy'):
            try:
                if side == Side.FREE_PEOPLES:
                    g = state.fellowship.guide
                    if (g is not None
                            and getattr(g, 'name', '') == 'Gandalf the Grey'
                            and _pe_face in (ActionDieFace.EVENT,
                                             ActionDieFace.WILL_OF_WEST)):
                        # engine-granted draw: grant_draw is the one
                        # writer of the owed-draw ledger
                        game_logic.grant_draw(state, side, _pe_deck)
                else:
                    from persistent_effects import \
                        should_draw_extra_card_on_event as _pal
                    # USER RULING 2026-08-15: the Palantir does NOT
                    # self-trigger — playing the card itself grants no
                    # bonus draw; the ability starts with the NEXT
                    # Shadow event play.  REGRESSION FIX 2026-08-16:
                    # the original guard compared against a spaced
                    # ascii name but the canonical id is
                    # SHADOW_Character_The_Palantír_of_Orthanc
                    # (underscores + i-acute) — it could NEVER match
                    # and the ruling was never enforced (born dead).
                    # Normalize both differences; the live-registry
                    # pin lives in test_palantir_no_self_trigger.py.
                    _is_pal_self9 = ('palantir of orthanc'
                                     in str(card_id).lower()
                                     .replace('_', ' ')
                                     .replace('í', 'i'))
                    # card text: "use an Event Action Die result to
                    # play an Event card" - the rolled face decides
                    # (a ring waiver never changes the face)
                    if (_pal(state) and not _is_pal_self9
                            and _pe_face == ActionDieFace.EVENT):
                        if state.pending_combat is not None:
                            # P-die action still resolving: defer until
                            # the action fully resolves (USER 1.208m)
                            state._palantir_draw_pending = True
                        else:
                            _open_palantir_deck_choice(state)
            except Exception:
                pass
        # A card-driven hunt (Nazgul Strike!'s roll-for-the-Hunt option)
        # leaves its tile damage in _pending_hunt_damage — convert it to
        # the learned sacrifice decision exactly like the
        # fellowship_move path (2024-07-27 Columbo T8: Boromir guide-
        # sacrifice at L787 resolves the L764 '2' tile).
        _phd = getattr(state, '_pending_hunt_damage', 0)
        if _phd > 0 and state.pending_hunt is None:
            # 2026-07-21: learned Axe/Horn reduction on the self-play
            # card-driven hunt path (reduce, then confront; the shared
            # settle also aligns the Gollum reveal with the
            # fellowship_move twin). TEACHER: historical
            # sacrifice-or-apply kept byte-identical (log = truth).
            if not game_logic._offer_hunt_reduction(state):
                game_logic._settle_hunt_damage(state)

    elif atype == 'card_option':
        _pco = getattr(state, 'pending_card_option', None)
        state.pending_card_option = None
        # P1 (USER ruling 1.112): Flocks of Crebain interpose for the
        # Nazgul Strike! HUNT option - the SHADOW first decides whether to
        # discard Crebain for +1 to ALL Hunt dice, mirroring the
        # fellowship_move interpose above.
        # The Crebain decision resumes THIS card_option via resume_action,
        # restoring (via _restore_pco) the Nazgul Strike pending it carries
        # so the hunt re-dispatches with the bonus set and NO re-interpose
        # (the _crebain_resolved marker rides the resumed choices).
        if _pco is None and p.get('_restore_pco') is not None:
            _pco = p['_restore_pco']
        if _pco is not None:
            _cho9 = dict(p.get('choices') or {})
            if (_cho9.get('action') == 'hunt'
                    and not _cho9.get('_crebain_resolved')
                    and 'nazgul strike' in sanitize(
                        str(_pco.get('card_name', '')))):
                from game_state import PersistentCardType as _PCT8
                _pc8 = getattr(state, 'persistent_cards', None)
                _cb8 = (_pc8.get_by_type(_PCT8.FLOCKS_OF_CREBAIN)
                        if _pc8 else None)
                if (_cb8 is not None
                        and not state.fellowship.mordor_entered
                        and any(d.side is Side.SHADOW
                                for d in state.hunt_box)):
                    state.pending_card_option = {
                        'card_id': getattr(_cb8, 'card_id', '__crebain__'),
                        'card_name': 'Flocks of Crebain',
                        'side': 'sh', 'decider': 'sh', 'direct': True,
                        'options': [{'crebain_discard': True},
                                    {'crebain_discard': False}],
                        'resume_action': {
                            'die_idx': die_idx,
                            'action_type': 'card_option',
                            'params': {
                                'choices': dict(
                                    _cho9, _crebain_resolved=True),
                                '_restore_pco': _pco}}}
                    return
        if _pco is not None and _pco.get('card_casualty'):
            # USER 2026-07-27: card-inflicted casualties (Dreadful
            # Spells / Spirit of Mordor) resolve through the SAME
            # allocation machinery as combat — the chosen option
            # carries k elite downgrades; _apply_allocation does the
            # casualty-box-first accounting + FP tracking, then the
            # shared aftermath handles orphans / wiped-garrison capture.
            _kcc = int((p.get('choices') or {}).get('cas_downgrades', 0)
                       or 0)
            _apply_allocation(state, _pco['cas_region'],
                              _pco['cas_side'], int(_pco['cas_hits']),
                              _kcc)
            import event_effects as _eecc
            _rescc = {'effects': []}
            # USER RULING 2026-08-15: the wipe exception is keyed on the
            # CARD — the pending carries its name across the decision
            # boundary (combat's published context cannot survive it).
            _eecc._complete_card_casualty(
                state, _pco['cas_region'], _pco['cas_side'], _rescc,
                card_name=_pco.get('card_name'))
            return
        if _pco is not None and _pco.get('bof_separation'):
            # USER 2026-07-27: Breaking of the Fellowship — WHICH
            # companion leaves is the FP player's chained learned pick
            _nmb = ((p.get('choices') or {}).get('companions')
                    or [None])[0]
            import event_effects as _eeb
            _eeb._bof_apply_separation(state, _pco, _nmb)
            return
        if _pco is not None and _pco.get('hunt_reduction'):
            # USER RULING 2026-07-21: Axe&Bow / Horn of Gondor
            # discard-to-reduce interpose (game_logic.
            # _offer_hunt_reduction). Resolve the FP's yes/no, then
            # confront the (possibly reduced) remaining damage.
            _chr = p.get('choices') or {}
            if _chr.get('use_hunt_reduction'):
                _pcs = getattr(state, 'persistent_cards', None)
                _cid = _chr.get('card_id') or _pco.get('card_id')
                if _pcs is not None and _cid is not None:
                    # CARD CONSERVATION (2026-08-15): the spent Horn of
                    # Gondor / Axe and Bow is DISCARDED face-up, not
                    # deleted (auto35 live game lost Horn of Gondor).
                    from persistent_effects import (discard_table_card
                                                    as _dtc9)
                    _dtc9(state, _pcs.remove_card(_cid))
                state._pending_hunt_damage = max(
                    0, getattr(state, '_pending_hunt_damage', 0) - 1)
            _fres = _pco.get('foul_resume')
            if _fres is not None:
                # Foul Thing tail: BEFORE the forced random casualty the
                # FP may use guide abilities (USER 1.208l). Route through
                # _foul_confront which opens the learned foul_guide
                # interpose (self-play) or falls through to the casualty.
                from event_effects import _foul_confront
                _red9 = max(0, int(_fres.get('damage', 0))
                            - (1 if _chr.get('use_hunt_reduction')
                               else 0))
                _foul_confront(state, _fres, _red9)
                state._pending_hunt_damage = 0
            else:
                game_logic._settle_hunt_damage(state)
            return
        if _pco is not None and _pco.get('foul_guide'):
            # USER 1.208l: Foul Thing pre-pick guide stage. The FP may
            # use the current guide's reducing ability (hobbit separates
            # / Gollum reveals) to shave 1 damage before the forced
            # random casualty; a hobbit separation re-checks the NEW
            # guide (chain). Self-play only (foul_guide pending is never
            # opened on the teacher path). Mirrors hunt_guide_separate.
            _chr = p.get('choices') or {}
            _fres = _pco.get('foul_resume') or {}
            _dmg = int(_fres.get('damage', 0))
            from event_effects import (_foul_confront,
                                       _foul_apply_after_reduction)
            if _chr.get('use_foul_guide'):
                _g = state.fellowship.guide
                _ab = getattr(_g, 'guide_ability', None)
                if _ab == 'no_reveal_on_damage':
                    state.fellowship.hidden = False        # Gollum reveals
                else:                                       # hobbit separates
                    _f = state.fellowship
                    if _g in _f.companions:
                        _f.companions.remove(_g)
                    if _f.mordor_entered:
                        # USER 2026-08-15: separation-death, not a hunt
                        # casualty — no WWSAT (Foul Thing's WWSAT comes
                        # from its forced PICK, not the guide reduce)
                        game_logic._on_companion_casualty(
                            state, _g.name, casualty=False)
                    else:
                        q = getattr(state, '_hobbit_place_queue', None) or []
                        q.append(_g)
                        state._hobbit_place_queue = q
                    game_logic.change_guide(state, allow_pending=True)
                _foul_confront(state, dict(_fres, victim=_fres.get('victim')),
                               max(0, _dmg - 1))            # re-check new guide
            else:
                # decline -> straight to the forced casualty. Must NOT
                # re-enter _foul_confront: the current reducing guide is
                # unchanged (e.g. Gollum still hidden), so it would just
                # re-open the same interpose forever (spec 1.208l control-
                # flow fix; the spec's "decline -> pick" comment confirms
                # the intent is the pick, not another offer).
                _foul_apply_after_reduction(state, _fres, _dmg)
            state._pending_hunt_damage = 0
            return
        if _pco is not None and _pco.get('palantir_draw'):
            # USER 1.208m: Palantir of Orthanc bonus draw - the SHADOW
            # picks WHICH deck (Character / Strategy) to draw from after
            # the P-die event action fully resolved. Self-play only.
            _chp = p.get('choices') or {}
            _deck = str(_chp.get('palantir_deck', 'character') or
                        'character')
            # 2026-08-16 (live-play halt): WOTR has NO reshuffle — an
            # exhausted deck simply means no draw.  This caller drew
            # unguarded, so a late-game empty Shadow deck raised
            # RuntimeError out of apply_action and HALTED the game
            # (the comment below claimed self-play only; it fired on
            # the live bridge path too).
            try:
                # engine-granted draw: grant_draw stamps the owed-draw
                # debt when the client owns the physical draw
                game_logic.grant_draw(state, Side.SHADOW, _deck)
            except RuntimeError as _pde9:
                print('[palantir] no draw: %s' % (_pde9,), flush=True)
            state._palantir_draw_pending = False
            return
        if _pco is not None and _pco.get('card_name') == '__wwsat_discard__':
            # 1.211f: resolve the Shadow's learned Worn-with-Sorrow
            # SOURCE pick (kill the named table FP Character card, or
            # discard a random FP hand Character card). 1.210g: MANDATORY
            # - the option list never carries a decline. pco already
            # cleared above.
            _chw = p.get('choices') or {}
            game_logic._wwsat_resolve(
                state, source=_chw.get('wwsat_source'),
                card_id=_chw.get('card_id'))
            return
        if _pco is not None:
            _side = (Side.FREE_PEOPLES if _pco['side'] == 'fp'
                     else Side.SHADOW)
            _ch9 = dict(p.get('choices') or {})
            # chained stages know they arrived via the pending path
            _ch9['_from_pending'] = True
            if _pco.get('direct'):
                # chained stage: the card already left the hand - call
                # the handler directly (play_event_card would refuse
                # 'Card not in hand'). P1 chaining machinery.
                try:
                    from event_effects import (CARD_EFFECT_HANDLERS,
                                               resolve_overstacks)
                    try:
                        from cards import get_card as _gcd
                        _cd9 = _gcd(_pco['card_id'])
                    except Exception:
                        # pseudo-pendings ('__overstack__') have no
                        # real card behind them
                        _cd9 = None
                    _h9 = CARD_EFFECT_HANDLERS.get(
                        str(_pco.get('card_name', '')).lower()) or \
                        CARD_EFFECT_HANDLERS.get(
                            getattr(_cd9, 'card_name', '').lower())
                    if _h9 is not None:
                        # Same conservation choke point as
                        # play_event_card's dispatch: card recruits may
                        # overstack, and casualty application inside a
                        # handler must be supply-checked against THIS
                        # state (combat._ACTIVE_STATE).
                        import event_effects as _ee9
                        from combat import set_active_state as _sas9
                        _ee9.CARD_RECRUIT_OVERSTACK = True
                        _prev_as9 = _sas9(state)
                        try:
                            _h9(state, _cd9, _ch9)
                        finally:
                            _ee9.CARD_RECRUIT_OVERSTACK = False
                            _sas9(_prev_as9)
                    # a direct-stage effect may itself overstack (card
                    # moves); scan unless the handler already chained
                    resolve_overstacks(state, learned=True)
                except Exception:
                    # the pending is already cleared: a swallowed
                    # failure silently drops the decision AND its
                    # effect.  Self-play / live apply on a clone, so the
                    # raise rejects the action; only the teacher (log =
                    # truth, FINAL_HITS_HOOK set) keeps the lenient path.
                    import combat as _cmbds9
                    if _cmbds9.FINAL_HITS_HOOK is None:
                        raise
            else:
                try:
                    from event_effects import play_event_card
                    play_event_card(state, _side, _pco['card_id'],
                                    choices=_ch9)
                except Exception:
                    hand = state.hands.get(_side, PlayerInfo())
                    if _pco['card_id'] in hand.hand:
                        hand.hand.remove(_pco['card_id'])
            # P1 (Crebain): a decision interposed on another action
            # (fellowship_move) resumes that stashed action now
            _ra9 = _pco.get('resume_action')
            if _ra9 is not None:
                _execute_action(state, Action(
                    die_idx=_ra9['die_idx'],
                    action_type=_ra9['action_type'],
                    params=dict(_ra9['params'])))
            # 1.230: a card_option resolution may leave hunt-tile damage
            # (Mithril redraw, There is Another Way's move) - it runs
            # the SAME full settle as the fellowship_move tail (Gollum
            # offer + Axe/Horn offer + sacrifice chains); the old inline
            # sacrifice-or-apply copy diverged
            if getattr(state, '_pending_hunt_damage', 0) > 0 \
                    and state.pending_hunt is None:
                if not game_logic._offer_hunt_reduction(state):
                    game_logic._settle_hunt_damage(state)

    elif atype == 'transform_gandalf':
        region = p['region']
        game_logic.transform_gandalf_white(state, region, die_idx)

    elif atype == 'crown_aragorn':
        game_logic.crown_aragorn(state, die_idx)

    elif atype == 'companion_move':
        from_san = sanitize(p['from'])
        to_san = sanitize(p['to'])
        _consume_die_for_face(state, Side.FREE_PEOPLES, die_idx,
                              (ActionDieFace.CHARACTER,
                               ActionDieFace.WILL_OF_WEST))
        src_key = next((k for k in state.armies if sanitize(k) == from_san), None)
        if src_key is None:
            raise RuntimeError(f"companion_move: no army at {p['from']}")
        src_army = state.armies[src_key]
        comps = [l for l in src_army.leaders if isinstance(l, Companion)]
        _want9 = p.get('companions')
        if _want9:
            # P3: a named GROUP moves; the rest stay behind
            _wset9 = {str(n) for n in _want9}
            comps = [c for c in comps if c.name in _wset9]
        if not comps:
            raise RuntimeError(f"companion_move: no companions at {p['from']}")
        for c in comps:
            src_army.leaders.remove(c)
        if not src_army.units and not src_army.leaders:
            del state.armies[src_key]
        dst_key = next((k for k in state.armies if sanitize(k) == to_san), None)
        if dst_key is None:
            orig = board.regions.get(to_san, {}).get('orig', p['to'])
            state.armies[orig] = Army(region=state.board.region(to_san))
            dst_key = orig
        state.armies[dst_key].leaders.extend(comps)
        # Companion ending movement in own nation's city/stronghold
        # activates that nation (rulebook p.34)
        game_logic._check_companion_activates_nation(
            state, to_san, state.armies[dst_key])
        # USER (rulebook p.20 "Moving Companions"): the SAME Character
        # die may move ANY number of companion GROUPS from different
        # regions, each with its own allowance.  On the learned path
        # chain a '__companion_moves__' pending offering further group
        # moves - mirrors the '__nazgul_moves__' chain (amendment
        # 1.128).  Companions are NOT fungible, so the names that just
        # moved are excluded BY NAME (per-companion once-per-die); a
        # companion left behind at this move's source is still offered.
        # Teacher path unchanged (amendment 1.122: the log's per-region
        # walks + die-less continuations replay as before) - same
        # FINAL_HITS_HOOK gate as the nazgul chain.
        import combat as _cmbc9
        if _cmbc9.FINAL_HITS_HOOK is None \
                and getattr(state, 'pending_card_option', None) is None:
            _moved9 = set(c.name for c in comps)
            from event_effects import _companion_chain_options
            _optsc9 = _companion_chain_options(state, _moved9)
            if _optsc9:
                _optsc9.append({'_comp_done': True})
                state.pending_card_option = {
                    'card_id': '__companion_moves__',
                    'card_name': '__companion_moves__',
                    'side': 'fp', 'decider': 'fp', 'direct': True,
                    'options': _optsc9}

    elif atype == 'nazgul_move':
        from_san = sanitize(p['from'])
        to_san = sanitize(p['to'])
        _consume_die_for_face(state, Side.SHADOW, die_idx,
                              (ActionDieFace.CHARACTER,))
        # Locate source army (keys may be orig names)
        src_key = next((k for k in state.armies if sanitize(k) == from_san), None)
        if src_key is None:
            raise RuntimeError(f"nazgul_move: no army at {p['from']}")
        src_army = state.armies[src_key]
        nazgul = [l for l in src_army.leaders if isinstance(l, Nazgul)]
        _cnt9 = p.get('count')
        if _cnt9 is not None:
            # USER 2026-07-11: move only a COUNT subset (fungible)
            nazgul = nazgul[:max(0, int(_cnt9))]
        if p.get('include_wk'):
            nazgul += [l for l in src_army.leaders
                       if type(l).__name__ == 'Minion'
                       and 'witch' in str(getattr(l, 'name', '')).lower()]
        if not nazgul:
            raise RuntimeError(f"nazgul_move: no Nazgul at {p['from']}")
        for nz in nazgul:
            src_army.leaders.remove(nz)
        if not src_army.units and not src_army.leaders:
            del state.armies[src_key]
        # Destination army (create a leader-only stack if needed)
        dst_key = next((k for k in state.armies if sanitize(k) == to_san), None)
        if dst_key is None:
            orig = board.regions.get(to_san, {}).get('orig', p['to'])
            state.armies[orig] = Army(region=state.board.region(to_san))
            dst_key = orig
        state.armies[dst_key].leaders.extend(nazgul)
        # USER 2026-07-11 (printed rule): the SAME die may distribute
        # more Nazgul to OTHER destinations - a chained learned pending
        # on the self-play path (the teacher replays the log's own
        # per-piece flights; same gating pattern as the Mithril
        # interpose).  One nazgul per chained decision; figures that
        # already flew this die are excluded via the arrivals map.
        import combat as _cmbn9
        if _cmbn9.FINAL_HITS_HOOK is None \
                and getattr(state, 'pending_card_option', None) is None:
            _arr9 = {to_san: len(nazgul), '__wk_ok__': 1}
            if p.get('include_wk'):
                _arr9['__wk__'] = 1  # the WK already flew this die
            from event_effects import _nazgul_chain_options
            _opts9 = _nazgul_chain_options(state, _arr9)
            if _opts9:
                _opts9.append({'_naz_done': True})
                state.pending_card_option = {
                    'card_id': '__nazgul_moves__',
                    'card_name': '__nazgul_moves__',
                    'side': 'sh', 'decider': 'sh', 'direct': True,
                    'options': _opts9}

    elif atype == 'mouth_move':
        # USER RULING 2026-07-20 (P2-02): Mouth moves on a Character
        # die, range 3, no destination restrictions.
        from_san = sanitize(p['from'])
        to_san = sanitize(p['to'])
        _consume_die_for_face(state, Side.SHADOW, die_idx,
                              (ActionDieFace.CHARACTER,))
        src_key = next((k for k in state.armies
                        if sanitize(k) == from_san), None)
        if src_key is None:
            raise RuntimeError(f"mouth_move: no army at {p['from']}")
        src_army = state.armies[src_key]
        mouth = next((l for l in src_army.leaders
                      if type(l).__name__ == 'Minion'
                      and 'mouth' in str(getattr(l, 'name', '')).lower()),
                     None)
        if mouth is None:
            raise RuntimeError(f"mouth_move: no Mouth at {p['from']}")
        src_army.leaders.remove(mouth)
        if not src_army.units and not src_army.leaders:
            del state.armies[src_key]
        dst_key = next((k for k in state.armies
                        if sanitize(k) == to_san), None)
        if dst_key is None:
            orig = board.regions.get(to_san, {}).get('orig', p['to'])
            state.armies[orig] = Army(region=state.board.region(to_san))
            dst_key = orig
        state.armies[dst_key].leaders.append(mouth)

    elif atype == 'muster_minion':
        minion = p['minion']
        region = p['region']
        game_logic.muster_minion(state, minion, region, die_idx)

    elif atype == 'companion_separate':
        companion_name = p.get('companion')
        to_region = p['to_region']
        ph = state.pending_hunt
        if ph is not None and ph.get('phase') == 'hobbit_place':
            # Hunt-casualty Hobbit placement (die-less): the hobbit was
            # already removed from the Fellowship by the hunt resolution;
            # FP now places him as if separated (1.230 ONE impl).
            game_logic.resolve_hunt_decision(state, 'hobbit_place',
                                             place_region=to_region)
        else:
            # separate_companions() handles die consumption via _pick_die()
            # (audit A15: a 'companions' list = ONE-die group separation)
            _grp15 = p.get('companions')
            game_logic.separate_companions(
                state, list(_grp15) if _grp15 else [companion_name],
                to_region, die_idx=die_idx)

    elif atype == 'fellowship_reveal_place':
        # Learned reveal placement after hunt reveals fellowship
        # (teacher-forced from logs since v-early; the LEARNED pick in
        # self-play/live since USER RULING 2026-08-16).  "Revealed in"
        # shadow stronghold tile already drawn (teacher: in
        # fellowship_move; learned: open_reveal_placement's prep).
        # _apply_reveal_placement is the ONE implementation:
        # leaving/entering tiles + pawn move + progress 0.
        _apply_reveal_placement(state, p['region'])
        # Learned path only: a leaving/entering draw's damage settles
        # through the shared learned machinery.  TEACHER keeps its
        # certified behavior (logged records force damage separately).
        import combat as _cmbrp9
        if _cmbrp9.FINAL_HITS_HOOK is None \
                and getattr(state, '_pending_hunt_damage', 0) > 0:
            if not game_logic._offer_hunt_reduction(state):
                game_logic._settle_hunt_damage(state)

    elif atype in ('hunt_gollum_reveal', 'hunt_gollum_no_reveal',
                   'hunt_take_corruption', 'hunt_sacrifice_guide',
                   'hunt_guide_separate'):
        # 1.230: game_logic.resolve_hunt_decision is the ONE
        # implementation (hunt_dist kernel consumes the same body)
        game_logic.resolve_hunt_decision(state, atype)

    elif atype == 'hunt_sacrifice_random':
        # Closed-loop ground truth: the log's draw NAMES the drawn
        # companion - forced_companion honors it (1.24); 1.230 ONE impl
        game_logic.resolve_hunt_decision(
            state, atype, forced_companion=p.get('forced_companion'))

    elif atype == 'choose_guide':
        companion_name = p['companion']
        game_logic.change_guide(state, chosen=companion_name)
        state.pending_guide_change = None

    elif atype == 'eye_allocate':
        n_eyes = p.get('count', 0)
        if n_eyes > 0:
            indices = [i for i, d in enumerate(state.dice[Side.SHADOW])
                       if not d.used][:n_eyes]
            game_logic.allocate_eyes(state, n_eyes, indices=indices)
        # the allocation decision is over (clones keep the flag now, so
        # the executor must end the phase or MCTS loops on it forever)
        state._eye_allocation_phase = False

    elif atype == 'elven_ring':
        target_idx = p['target_die_idx']
        new_face_str = p['new_face']
        new_face = ActionDieFace(new_face_str)
        ring_side = Side.FREE_PEOPLES if p.get('side') == 'fp' else Side.SHADOW
        game_logic.use_elven_ring(state, ring_side, target_idx, new_face)
        # a ring use is not an action: the ring user acts next
        # (_next_acting_side); the marker dies with the next action
        state._ring_free_side = ring_side

    elif atype == 'use_token':
        t_side = Side.FREE_PEOPLES if p.get('side', 'fp') == 'fp' \
            else Side.SHADOW
        game_logic.use_action_token(
            state, t_side, token_type=p['token_type'],
            nation=p.get('nation'), deck=p.get('deck', 'strategy'))

    elif atype == 'dwarven_ring':
        r_side = Side.FREE_PEOPLES if p.get('side', 'fp') == 'fp' \
            else Side.SHADOW
        forced = p.get('forced_face')
        if forced is not None and not isinstance(forced, ActionDieFace):
            forced = ActionDieFace(forced)
        game_logic.use_dwarven_ring(state, r_side, die_idx=p['die_idx'],
                                    forced_face=forced)

    else:
        raise ValueError(f"Unknown action type: {atype}")


def _consume_die(state: GameState, die_idx: int):
    """Mark a die as used without caring about face.

    If the specified die is already used, finds another unused die.
    """
    if die_idx < 0:
        return  # No-die action (e.g., fellowship_declare)
    for side in [Side.FREE_PEOPLES, Side.SHADOW]:
        dice = state.dice.get(side, [])
        if die_idx < len(dice) and not dice[die_idx].used:
            dice[die_idx].mark_used()
            return
    # Fallback: consume any unused die
    for side in [Side.FREE_PEOPLES, Side.SHADOW]:
        for d in state.dice.get(side, []):
            if not d.used and d.current is not None:
                d.mark_used()
                return


def _consume_die_for_face(state: GameState, side: Side, die_idx: int,
                          valid_faces: Tuple[ActionDieFace, ...]):
    """Mark a die as used for the given side with a compatible face.

    If the specified die_idx is already used, finds another unused die
    showing one of the valid faces.  Returns the index of the die
    actually consumed (None for die-less actions) so callers that
    relocate the die (hunt box) always move the CONSUMED die.
    """
    if die_idx < 0:
        return None  # No-die action (e.g., fellowship_declare)
    dice = state.dice.get(side, [])
    # Try the specified die first
    if die_idx < len(dice):
        d = dice[die_idx]
        if not d.used and d.current in valid_faces:
            d.mark_used()
            return die_idx
    # A7 (USER ruling 2026-08-19): an ELVEN RING paid for this action is
    # a LEGALITY EXPANSION - the STAGED die pays, keeping its face.
    #
    # THIS BRANCH MUST SIT ABOVE THE FALLBACK, and that ordering is the
    # whole fix.  Below it, the fallback would quietly consume a
    # DIFFERENT die than the one the human physically dragged into
    # Current Action - and `fellowship_move` drops the CONSUMED die into
    # the Hunt Box, so the wrong die leaves his pool while the staged
    # one stays spendable.  That is the client-vs-engine desync behind
    # USER live items C2/C4 ("shadow kept the elven ring I spent").
    #
    # The ring may never produce Will of the West (rulebook Ch.IV), so a
    # Will-ONLY requirement - crown_aragorn's `payment='will'` - is not
    # waivable.  A widened permit is not a conservative change either.
    if (die_idx < len(dice)
            and set(valid_faces) != {ActionDieFace.WILL_OF_WEST}):
        import game_logic as _gl7
        d = dice[die_idx]
        if (not d.used and _gl7.ring_waiver_active(state, side)
                and _gl7.ring_waiver_permits(side, d.current)):
            d.mark_used()
            _gl7.ring_waiver_spend(state, side)
            return die_idx
    # The named die cannot pay.  Consuming a DIFFERENT die silently is
    # the client-vs-engine desync class (the human dragged die X, the
    # engine spends die Y and fellowship_move drops the wrong die into
    # the Hunt Box): self-play / live refuse LOUDLY.  Only the teacher
    # (log = truth, FINAL_HITS_HOOK set) keeps the lenient any-die
    # fallback for its own die accounting.
    import combat as _cmbdie9
    if _cmbdie9.FINAL_HITS_HOOK is not None:
        for i, d in enumerate(dice):
            if not d.used and d.current in valid_faces:
                d.mark_used()
                return i
    # a silent no-op here means the action executes FOR FREE (an
    # illegal extra action) - fail loudly so the driver rejects it
    raise RuntimeError(
        f"die {die_idx} of {side.name} cannot pay for faces "
        f"{[f.name for f in valid_faces]} (used/absent/wrong face) "
        f"- refusing to consume a different die")


# ===================================================================
# Step 8a: MCTS (Monte Carlo Tree Search)
# ===================================================================

class MCTSNode:
    """A node in the MCTS search tree.

    1.148 (USER 2026-07-19): a node whose action consumed RNG on
    application becomes a CHANCE node (is_chance=True, state stays
    None).  Its `outcomes` list holds independently-sampled result
    branches; selection there is least-visited (progressive widening),
    NOT PUCT — so its Q converges to the EXPECTATION over the random
    event instead of freezing one sampled outcome ("the net should
    calculate expectation values in MCTS for anything affected by
    RNG")."""
    __slots__ = ('state', 'side', 'parent', 'action', 'prior',
                 'children', 'visit_count', 'value_sum', 'value_sq',
                 'term_visits',
                 'is_terminal', 'terminal_value',
                 'is_chance', 'outcomes', 'out_sig', 'drops',
                 'proven', 'proven_value', 'kernel_cover',
                 'w3', 'ov_seen', 'ov_win', 'forced_visits',
                 'conf_sum', 'phi_sum')

    def __init__(self, state, side, parent=None, action=None, prior=0.0):
        self.state = state           # GameState (cloned), None for lazy expansion
        self.side = side             # Side that acts from this node
        self.parent = parent
        self.action = action         # Action that led here from parent
        self.prior = prior           # P(a) from policy network
        self.children = []           # List of MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0         # Sum of values from this side's perspective
        self.value_sq = 0.0          # 1.226e: sum of squared backup values
        self.term_visits = 0         # 1.226f: backups from RESOLVED terminals
        self.is_terminal = False
        self.terminal_value = 0.0
        self.is_chance = False       # True: outcomes[] are sampled results
        self.outcomes = None         # list[MCTSNode] for chance nodes
        self.out_sig = None          # outcome fingerprint (branch dedup);
                                     # branch.prior doubles as draw weight
        self.drops = 0               # cap-overflow draws (count toward the
                                     # bank or high-entropy nodes redraw
                                     # every visit forever)
        self.proven = None           # 'W'/'L' game-theoretic proof from
                                     # the ROOT side's view; minimax over
                                     # proven children, chance BLOCKS
        self.proven_value = 0.0      # graded value of the proof (kappa
                                     # grading arbitrates all-lost roots)
        self.kernel_cover = None     # kernel-exact chance node: kept
                                     # branch mass (1.0 = full support;
                                     # None = sampled node)
        self.w3 = None               # opponent combat-card window:
                                     # closed-form P(plays X) map over
                                     # the full unseen pool (design 3B)
                                     # — the 3A lazy-add mass source;
                                     # None = BC-prior node
        self.ov_seen = None          # 3A lazy-add memo: card_ids
                                     # already considered at this node
        self.ov_win = None           # cached card-window kind
                                     # ('none'/'discard'/'combat'/
                                     # 'event'); None = not classified
        self.forced_visits = 0       # task-14 FIX 2: forced-floor
                                     # visits (NODE attribute so the
                                     # count SURVIVES tree reuse —
                                     # a per-search array leaked old
                                     # forced visits into 'earned'
                                     # on reused roots)
        self.conf_sum = 0.0          # task-15 LAW 2: sum of leaf
                                     # confidences backed up through
                                     # this node (terminal/proven
                                     # backups count 1.0 — exact)
        self.phi_sum = 0.0           # task-17 LAW 1R: sum of leaf
                                     # scaffold components (root-
                                     # currency) — per-arm phi mean
                                     # for shadow values

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


# Self-play defers optioned event cards into pending_card_option
# decisions.  Extraction bypasses the defer by passing explicit
# choices, so this stays True everywhere.
CARD_OPTION_DEFER = True

_ATYPE_TO_CAT = {
    'army_move': 'army_move', 'nazgul_move': 'army_move',
    'mouth_move': 'army_move',   # P2-02 ruling 2026-07-20
    'siege_trim': 'army_attack',  # siege-withdrawal excess choice (2026-07-20)
    'companion_move': 'special_character',
    'retreat_into_siege': 'army_move', 'combat_cease': 'pass',
    'army_attack': 'army_attack',
    'advance_political': 'advance_political',
    'muster_recruit': 'muster', 'muster_elite': 'muster',
    'muster_minion': 'muster', 'muster': 'muster',
    'draw_card': 'draw_card', 'play_event': 'play_event',
    'card_option': 'play_event',
    'discard_table_card': 'play_event',   # 1.149 table-card kill
    'victor_advance': 'army_move', 'victor_stay': 'pass',
    'discard_card': 'draw_card',  # Hand-limit discard (card economy)
    'fellowship_move': 'fellowship_move',
    'fellowship_hide': 'fellowship_hide',
    'fellowship_declare': 'fellowship_declare',
    'fellowship_reveal_place': 'fellowship_move',
    'special_character': 'special_character',
    'pass': 'pass', 'skip': 'pass', 'eye_allocate': 'eye_allocate',
    'combat_card': 'combat_card',
    # Combat-card decision phases (learned: play a specific card / decline)
    'combat_card_play': 'combat_card',
    'combat_card_none': 'combat_card',
    # Choice-bearing combat-card effect magnitude (Onslaught / Dread and
    # Despair / Relentless Assault / Heroic Death)
    'combat_effect_choice': 'combat_card',
    # Companion separation
    'companion_separate': 'special_character',
    # Rearguard
    'rearguard_none': 'army_attack',
    'rearguard_split': 'army_attack',
    # Casualty allocation (owner decision, once per battle)
    'casualty_preserve_units': 'army_attack',
    'casualty_preserve_elites': 'army_attack',
    # Combat: siege + retreat
    'combat_retreat': 'army_move',
    'combat_no_retreat': 'pass',
    'combat_continue': 'pass',
    'siege_assault': 'army_attack',
    'siege_continue': 'pass',
    'siege_press': 'army_attack',
    'siege_no_press': 'pass',
    # Elven rings (FP -> special_character; Shadow falls through in policy)
    'elven_ring': 'special_character',
    # Will-of-the-West specials
    'transform_gandalf': 'special_character',
    'crown_aragorn': 'special_character',
    # Bid-compensation free actions (user ruling 2026-07-03)
    'use_token': 'other',
    'dwarven_ring': 'other',
    # Hunt: Gollum reveal + companion sacrifice
    'hunt_gollum_reveal': 'special_character',
    'hunt_gollum_no_reveal': 'special_character',
    'hunt_sacrifice_guide': 'special_character',
    'hunt_guide_separate': 'special_character',
    'hunt_sacrifice_random': 'special_character',
    'casualty_allocate': 'special_character',
    'hunt_take_corruption': 'special_character',
    # Guide change
    'choose_guide': 'special_character',
}

# ---------------------------------------------------------------------
# EPISTEMIC CONFIDENCE (task 15, USER order 2026-08-10; reset-
# architecture wave).  VALUE owns the MECHANISM (how c is produced:
# head-ensemble disagreement / RND / embedding density — eventually a
# confidence-native net).  SEARCH owns CONSUMPTION, defined here.
#
# Contract: a PROVIDER callable registered via set_confidence_fn:
#     fn(vec: np.float32[D], side: Side, v_raw: float) -> c in [0, 1]
# where v_raw is the net's RESIDUAL value (pre-phi).  c = 1 means
# "this state is inside the net's supported experience"; c = 0 means
# "the net has no basis here".  When VALUE's confidence-native net
# lands, the provider becomes a thin adapter over the net's own
# confidence output; the consumption laws below do not change.
#
# LAW 1 (leaf shrinkage, WOTR_CONF_LEAF=1): the leaf value becomes
#     v_leaf = c * v_net + sgn * phi
# — the net RESIDUAL shrinks toward the phi scaffold (the hand-built
# prior), never the whole value toward an arbitrary zero.  Exact
# values are exempt BY CONSTRUCTION: terminals (_search_terminal) and
# solver proofs never pass through a net eval, and the heuristic
# fallback is itself scaffold-class.  Run-7 postmortem framing: an
# UNSUPPORTED +1.68 must not steamroll a SUPPORTED +0.08.
#
# LAW 2 (per-arm support): each simulation's leaf confidence is
# accumulated on the path (conf_sum); a root arm's support is
#     s_a = tf_a + (1 - tf_a) * mean_leaf_conf_a
# (terminal-resolved backups are exact = full support).  Exposed in
# visit_info['conf'] and payload 'cf' for meters and for LAW 3.
#
# LAW 3 (quantified 1.230, cfg conf_margin > 0): the binary
# terminal-grounded override generalizes to
#     q_gap >= term_margin + conf_margin * (1 - s_eff),
#     s_eff = max(s_arm, s_default)
# — at full support the margin is term_margin; with no support the
# required gap is term_margin + conf_margin (conf_margin large ==
# today's binary rule).  Default 0.0 = rule unchanged.
# ---------------------------------------------------------------------
_CONF_LEAF = os.environ.get('WOTR_CONF_LEAF', '0') == '1'
# LAW 1R (task 17, seeded re-adjudication comms\36): TRACK-ONLY mode.
# The erebor class proved the honest theorem: on a deterministic
# knife-edge (true margin 0.0025, interiors mid-confidence with
# nonzero spread) ANY leaf-value transformation — absolute or
# contrast-relative — reprices the order: the perturbation scale
# (O(c*|v|) ~ 0.1-1.0) exceeds the margin ~100x, and making it
# exactly zero on "trustworthy" states requires knowing which states
# those are — the confidence question itself, circular.  Therefore
# leaf shrinkage is RETIRED FROM THE PLAY PATH (WOTR_CONF_LEAF stays
# off; code kept for TD-side use and for the record).  Consumption
# moves to the ROOT DECISION: WOTR_CONF_TRACK=1 calls the provider
# at every leaf eval and records confidence WITHOUT touching the
# value — search, values, and RNG stream are BIT-IDENTICAL to
# conf-OFF by construction — and select_root_action applies a
# support-and-margin-gated override (cfg.conf_play) where margins
# are explicit and auditable.
_CONF_TRACK = os.environ.get('WOTR_CONF_TRACK', '0') == '1'
_CONF_FN = None
_LAST_LEAF_CONF = 1.0     # set by _conf_shrink, read at backprop
_LAST_LEAF_PHI = 0.0      # scaffold component of the last leaf

# LAW 1B (task 16, R4 LEG-3b fix at source): CONFIDENCE REFERENCE
# MODE.  R4 finding: at erebor_recapture the correct arm wins by a
# 0.0025 Q-margin and the whole neighborhood reads NEAR-UNIFORM
# mid-confidence (cf 0.38-0.42); absolute shrinkage (LAW 1 'abs')
# compresses the entire Q table ~2.5x toward per-state scaffold
# noise, destroying thin LEARNED margins and letting a 1-visit arm
# win the cert judge's argmax-Q.  The epistemic truth: a UNIFORM
# confidence level carries NO ordering information — only CONTRAST
# does.  'root' mode consumes the differential component only:
#     c_eff = min(1, c_leaf / max(c_root, eps))
# (c_root = the provider's read of the ROOT state, once per search;
# reference calls pass v_raw = 0.0 — basis-type providers ignore it
# by contract).  Near-uniform subtrees -> c_eff ~ 1 -> orderings
# preserved (erebor); genuinely less-supported lines than the root
# -> shrinkage stands (mordor sit-lines 0.15/0.85 differential).
# Never amplifies (clip at 1).  'mean' mode uses the search-running
# mean as reference (measured alternative; nonstationary early).
# Default 'abs' = R4 behavior until the battery adopts.
_CONF_REF = os.environ.get('WOTR_CONF_REF', 'abs')
_CONF_ROOT_C = None       # per-search root reference ('root' mode)
_CONF_MEAN = [0.0, 0]     # running [sum, n] ('mean' mode)
# Near-uniform DEADBAND ('root' mode): when c_leaf/c_root >=
# 1 - deadband, c_eff snaps to EXACTLY 1 — the leaf value is
# bit-untouched, so a near-uniform-confidence search is BIT-IDENTICAL
# to conf-OFF (same values -> same descent -> same RNG stream).
# This is what makes no-regression PROVABLE on fixtures whose
# margins sit below value noise (erebor's 0.0025: ANY perturbation,
# however small, is a coin flip there — measured: the R4 conf-OFF
# pass itself flips under a different RNG stream).
_CONF_DEADBAND = float(os.environ.get('WOTR_CONF_DEADBAND', '0.25'))


def set_confidence_fn(fn):
    """Register the confidence provider (None clears)."""
    global _CONF_FN
    _CONF_FN = fn


def _conf_search_reset(root_vec, side):
    """Per-search reference init (called by mcts_search when the
    wave is live).  'root': one provider read of the root state."""
    global _CONF_ROOT_C, _CONF_MEAN
    _CONF_MEAN = [0.0, 0]
    _CONF_ROOT_C = None
    if _CONF_REF in ('root', 'peak') and _CONF_FN is not None \
            and root_vec is not None:
        try:
            c = float(_CONF_FN(root_vec, side, 0.0))
            if c == c:
                _CONF_ROOT_C = min(1.0, max(1e-3, c))
        except Exception:
            _CONF_ROOT_C = None


def _conf_shrink(vec, side, v_raw):
    """LAW 1 core: returns c_eff * v_raw and records c_eff for
    LAW 2.  Inactive (flag off / no provider / provider error) ->
    identity with c = 1.

    Reference modes (LAW 1B):
      'abs'  — c_eff = c (R4 behavior; uniform mid-confidence
               compresses thin margins — the measured erebor break).
      'root' — c_eff = c / c_root (measured and REJECTED for the
               rescue case: a poisoned root reads unfamiliar itself,
               so the unsupported lines land in its deadband and the
               mordor flip is lost).
      'peak' — c_eff = c / running-max-confidence (initialized from
               the root read).  The reference is the BEST-SUPPORTED
               alternative seen this search: near-uniform subtrees
               (erebor 0.38-0.46) sit inside the deadband -> EXACT
               identity with conf-OFF; lines less supported than the
               best line (mordor sit 0.15 vs push 0.85) shrink.
               Transient: leaves read before the peak establishes go
               unshrunk (bounded by early-visit dilution; measured).
      'mean' — running-mean reference (measured alternative)."""
    global _LAST_LEAF_CONF, _CONF_ROOT_C
    if (not _CONF_LEAF and not _CONF_TRACK) or _CONF_FN is None:
        return v_raw
    try:
        c = float(_CONF_FN(vec, side, v_raw))
    except Exception:
        return v_raw
    if not (c == c):              # NaN guard
        return v_raw
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    if not _CONF_LEAF:
        # LAW 1R track-only: record ABSOLUTE support (the root
        # override consumes support LEVELS, not contrast) and return
        # the value bit-untouched — provably regression-free.
        _LAST_LEAF_CONF = c
        return v_raw
    if _CONF_REF == 'root' and _CONF_ROOT_C is not None:
        ratio = c / _CONF_ROOT_C
        c = 1.0 if ratio >= (1.0 - _CONF_DEADBAND) \
            else min(1.0, ratio)
    elif _CONF_REF == 'peak':
        if _CONF_ROOT_C is None:
            _CONF_ROOT_C = max(1e-3, c)
        elif c > _CONF_ROOT_C:
            _CONF_ROOT_C = c
        ratio = c / _CONF_ROOT_C
        c = 1.0 if ratio >= (1.0 - _CONF_DEADBAND) \
            else min(1.0, ratio)
    elif _CONF_REF == 'mean':
        _CONF_MEAN[0] += c
        _CONF_MEAN[1] += 1
        if _CONF_MEAN[1] >= 20:
            ref = _CONF_MEAN[0] / _CONF_MEAN[1]
            c = min(1.0, c / max(1e-3, ref))
    _LAST_LEAF_CONF = c
    return c * v_raw


# (lattice key, die face, hobbit_place, enemy comps per dest) -> the
# (N, ACTION_ENCODING_DIM) encoding matrix of a memoized separation
# block; bounded, process-wide.  One lookup replaces N per-action
# encodings at every FP expansion (profiled 2026-08-22: ~3,000 lattice
# actions x ~200 expansions per 300-sim search).
_LATTICE_ENC_CACHE = {}


def _encode_actions_batch(actions, state, side, comp_memo):
    """Encodings + region indexes for a menu: memoized lattice
    separation actions come as ONE cached block matrix (the block's
    Action objects are shared across nodes; its encodings vary only
    with the die face, the hobbit_place subtype and the enemy
    composition at each destination); every other action goes through
    _encode_action_cached as before.  Returns (encs, region_idx)."""
    n = len(actions)
    lat = [i for i in range(n) if '_lattice' in actions[i].__dict__]
    if len(lat) < 32:
        encs = np.array([_encode_action_cached(a, state, side, comp_memo)
                         for a in actions], dtype=np.float32)
        ridx = np.array([_region_index_cached(a) for a in actions],
                        dtype=np.int64)
        return encs, ridx
    # the menu may carry SEVERAL blocks (one per Character die): the
    # cache key names every block (in order) with its die's face
    _dl = state.dice.get(side, [])
    blocks = []
    dests = set()
    last_bk = None
    for i in lat:
        ad = actions[i].__dict__
        bk = ad.get('_block_key')
        if bk is not last_bk:
            die_idx = actions[i].die_idx
            _face = None
            if 0 <= die_idx < len(_dl)                     and _dl[die_idx].current is not None:
                _face = _dl[die_idx].current.name
            blocks.append((bk, die_idx, _face))
            dests.update(ad.get('_block_dests') or ())
            last_bk = bk
    _phk9 = getattr(state, 'pending_hunt', None)
    _hp = (isinstance(_phk9, dict) and _phk9.get('phase') == 'hobbit_place')
    en_sig = []
    for d in sorted(dests):
        _ek9 = ('__enemy__', d)
        _te = comp_memo.get(_ek9)
        if _te is None:
            _te = _enemy_comp_at(state, side, d)
            comp_memo[_ek9] = _te
        en_sig.append((d, _te))
    ck = (tuple(blocks), len(lat), _hp, tuple(en_sig))
    mat = _LATTICE_ENC_CACHE.get(ck)
    if mat is None or mat.shape[0] != len(lat):
        mat = np.array([_encode_action_cached(actions[i], state, side,
                                              comp_memo) for i in lat],
                       dtype=np.float32)
        if len(_LATTICE_ENC_CACHE) > 512:
            _LATTICE_ENC_CACHE.clear()
        _LATTICE_ENC_CACHE[ck] = mat
    encs = np.empty((n, mat.shape[1]), dtype=np.float32)
    encs[lat] = mat
    rest = [i for i in range(n) if '_lattice' not in actions[i].__dict__]
    if rest:
        encs[rest] = np.array([_encode_action_cached(actions[i], state,
                                                     side, comp_memo)
                               for i in rest], dtype=np.float32)
    ridx = np.array([_region_index_cached(a) for a in actions],
                    dtype=np.int64)
    return encs, ridx


def _get_priors_and_value(state: GameState, side: Side,
                          net: 'WOTRValueNet',
                          actions: 'List[Action]',
                          features_vec: 'Optional[np.ndarray]' = None
                          ) -> 'Tuple[List[float], float]':
    """Get policy priors + value from the network.

    Priors come from the spatial action scorer (behavioral-cloning-trained,
    GNN region embeddings per action); value from the value head.
    Falls back to uniform priors + heuristic value on any failure.
    """
    vec = features_vec if features_vec is not None else encode_state(state, side)
    # SCALE: remote batched-GPU inference path (per-net client wins —
    # lets one process arena nets served by DIFFERENT servers/archs)
    _icli = getattr(net, '_client', None) or INFER_CLIENT
    if _icli is not None and getattr(net, 'remote_key', None):
        try:
            _cm9 = {}   # per-expansion composition memo (profile fix)
            action_encs, region_idx = _encode_actions_batch(
                actions, state, side, _cm9)
            priors, v = _icli.pv(
                net.remote_key, np.asarray(vec, dtype=np.float32),
                action_encs, region_idx)
            priors = np.asarray(priors, dtype=np.float32)
            priors = priors / (priors.sum() + 1e-10)
            v = _conf_shrink(np.asarray(vec, dtype=np.float32),
                             side, float(v))     # LAW 1 (remote)
            # gate widened + phi TRACKED (ultrareview bug_001/003):
            # the remote branch applied shaping without recording
            # _LAST_LEAF_PHI — same-side remote leaves backed up
            # phi_sum 0.0, corrupting the ph payload LAW 1R and the
            # delta trainer consume; and the profile-only gate
            # dropped the rho ring term when profiles were unset
            if (_PHI_RINGS_RHO or PLAYER_PROFILES.get('fp')
                    or PLAYER_PROFILES.get('sh')):
                _shp17 = (_profile_shaping(state)
                          if side == Side.FREE_PEOPLES
                          else -_profile_shaping(state))
                if _CONF_LEAF or _CONF_TRACK:
                    global _LAST_LEAF_PHI
                    _LAST_LEAF_PHI = _shp17
                v = float(v) + _shp17
            return priors.tolist(), float(v)
        except Exception:
            pass
    net.eval()

    try:
        with torch.no_grad():
            x = torch.from_numpy(np.array([vec])).to(DEVICE)
            backbone_out, node_emb = net.forward_backbone(x)
            v = net.value_head(backbone_out).squeeze(-1).item()
            _cm9 = {}   # per-expansion composition memo (profile fix)
            action_encs, region_idx = _encode_actions_batch(
                actions, state, side, _cm9)
            ae_t = torch.from_numpy(action_encs).to(DEVICE)
            ri_t = torch.from_numpy(region_idx).to(DEVICE)
            priors_t = net.score_actions_spatial(
                backbone_out, node_emb, ae_t, ri_t)
            priors = priors_t.cpu().numpy().astype(np.float32)
        # (skip/rearguard prior floor removed 2026-07-08 — those actions
        # are no longer generated)
        priors = priors / (priors.sum() + 1e-10)
        v = _conf_shrink(np.asarray(vec, dtype=np.float32),
                         side, float(v))         # LAW 1 (local)
        if (_PHI_RINGS_RHO or PLAYER_PROFILES.get('fp')
                or PLAYER_PROFILES.get('sh')):
            _shp17 = (_profile_shaping(state)
                      if side == Side.FREE_PEOPLES
                      else -_profile_shaping(state))
            if _CONF_LEAF or _CONF_TRACK:
                _LAST_LEAF_PHI = _shp17   # global decl in remote block
            v = float(v) + _shp17
        return priors.tolist(), float(v)
    except Exception:
        pass

    # Fallback (heuristic leaf: net inference failed)
    v_fp = state_value_fp(state)
    v = v_fp if side == Side.FREE_PEOPLES else -v_fp
    if (_PHI_RINGS_RHO or PLAYER_PROFILES.get('fp')
            or PLAYER_PROFILES.get('sh')):
        _shp17 = (_profile_shaping(state) if side == Side.FREE_PEOPLES
                  else -_profile_shaping(state))
        if _CONF_LEAF or _CONF_TRACK:
            # Same recording contract as the net branches: the phi
            # component of every leaf feeds the ph/cf payloads.  A
            # heuristic leaf has no net support estimate; conf 1.0 =
            # "unshrunk / fully counted", the tracked invariant.
            # (_LAST_LEAF_PHI is declared global in the remote branch.)
            global _LAST_LEAF_CONF
            _LAST_LEAF_PHI = _shp17
            _LAST_LEAF_CONF = 1.0
        v += _shp17
    priors = np.ones(len(actions), dtype=np.float32) / len(actions)
    return priors.tolist(), v


# ---------------------------------------------------------------------
# Player strategy PROFILES (session 29, USER directive): six styles -
# FP ring / FP military / FP balanced, SH corruption / SH military /
# SH balanced.  A profile adds a small zero-sum potential to the MCTS
# leaf VALUE (search-behavior shaping only; replay outcomes stay pure
# +/-1/0, so the nets still learn true win/loss).  Self-play workers
# draw a profile pair per episode; arena games cycle all six.
# ---------------------------------------------------------------------
PLAYER_PROFILES = {'fp': None, 'sh': None}

# SCALE (USER 2026-07-19): per-worker handle to the batched-GPU
# inference server (src/infer_server.py).  None = local inference.
# Set by workers via infer_server.attach_client(); the MCTS helper
# functions route through it when the net object is a RemoteNet.
INFER_CLIENT = None
PROFILE_WEIGHT = 0.15
# v2 (USER, session 29): 'balanced' profiles removed - four styles,
# each trained against its own WIN AXIS (see the outcome-alignment
# block in _run_single_episode).
# 'mixed' (USER 2026-07-30): ALL outcomes valued equally — blended
# convex currency (the reward split's else-branch), full +/-1 terminal
# for ANY win (on-mission covers both victory types), no leaf shaping,
# phi channels included (military currencies count in the blend).
FP_PROFILES = ('ring', 'military', 'mixed')
SH_PROFILES = ('corruption', 'military', 'mixed')

# F2/V2a phi ring price (2026-08-04 A/B validated; VALUE rebase
# amendment 2026-08-09).  Flag default OFF -> byte-identical
# behavior.  Potential phi_rings = RHO * (fp_rings - sh_rings - 3):
# REBASED to game-start zero — the raw (fp-sh) form carries +0.3 at
# start, invisible to play (constant shifts never change in-tree
# comparisons; the A/B evidence transfers exactly) but it displaces
# the residual training currency (targets = G - phi): measured
# saturation 27-38% raw vs 0.1-4.0% rebased at the S=2 head
# (experiments\VALUE_rig RESULTS.md sec 7).  Spending a ring costs
# RHO immediately; potential-based -> telescopes.
_PHI_RINGS_RHO = float(os.environ.get('WOTR_PHI_RINGS_RHO', '0') or 0)


def _profile_shaping(state: GameState) -> float:
    """FP-positive zero-sum shaping term from both sides' profiles.
    'balanced'/None contribute 0 (pure value)."""
    _ring_phi9 = 0.0
    if _PHI_RINGS_RHO:
        _ring_phi9 = _PHI_RINGS_RHO * (
            float(state.fp_elven_rings)
            - float(state.sh_elven_rings) - 3.0)
    fp_p = PLAYER_PROFILES.get('fp')
    sh_p = PLAYER_PROFILES.get('sh')
    if not fp_p and not sh_p:
        return _ring_phi9
    b_fp = b_sh = 0.0
    f = state.fellowship
    try:
        if fp_p == 'ring':
            if f.mordor_entered:
                b_fp = 0.6 + 0.4 * min(f.mordor_step, 5) / 5.0
            else:
                # 1.145: truthful metric (1.143b) - the raw counter
                # resets on every declare/reveal, so mid-search the
                # shaping confiscated progress and dodged declares
                _d0s = _gate_distance(state, 'rivendell') or 8
                b_fp = 0.3 * min(_ring_progress_metric(state),
                                 _d0s) / float(_d0s)
            # 1.146: effective corruption (companion buffer) in
            # search too - protecting companions IS protecting the ring
            b_fp -= 0.25 * _effective_corruption(state) / 12.0
        elif fp_p == 'military':
            vp = game_logic.calculate_victory_points(state)
            b_fp = min(vp['free_peoples'] / 4.0, 1.0)
        if sh_p == 'corruption':
            b_sh = (f.corruption / 12.0
                    + 0.15 * min(len(state.hunt_box), 8) / 8.0)
        elif sh_p == 'military':
            vp = game_logic.calculate_victory_points(state)
            b_sh = min(vp['shadow'] / 10.0, 1.0)
    except Exception:
        return _ring_phi9
    return PROFILE_WEIGHT * (b_fp - b_sh) + _ring_phi9


def _mcts_eval_state(state: GameState, side: Side,
                     fp_net: 'WOTRValueNet',
                     sh_net: 'WOTRValueNet') -> float:
    """Evaluate a state from the given side's perspective.

    Uses the side's trained value network; falls back to the heuristic
    if no net is available.
    """
    _shp = _profile_shaping(state)
    _sgn = 1.0 if side == Side.FREE_PEOPLES else -1.0
    if _CONF_LEAF or _CONF_TRACK:
        # task-17 LAW 1R: record the scaffold component of this leaf
        # (feeds per-arm phi-mean for the root shadow values)
        global _LAST_LEAF_PHI
        _LAST_LEAF_PHI = _sgn * _shp
    net = fp_net if side == Side.FREE_PEOPLES else sh_net
    # SCALE: remote batched-GPU inference (infer_server) when the
    # worker holds a RemoteNet handle instead of local weights
    _icli = getattr(net, '_client', None) or INFER_CLIENT
    if _icli is not None and getattr(net, 'remote_key', None):
        try:
            vec = encode_state(state, side)
            v = _icli.eval(
                net.remote_key,
                np.asarray([vec], dtype=np.float32))[0]
            return _conf_shrink(np.asarray(vec, dtype=np.float32),
                                side, float(v)) + _sgn * _shp
        except Exception:
            pass
    if net is not None:
        try:
            vec = encode_state(state, side)
            net.eval()
            with torch.no_grad():
                x = torch.from_numpy(np.array([vec])).to(DEVICE)
                return _conf_shrink(
                    np.asarray(vec, dtype=np.float32), side,
                    float(net(x).item())) + _sgn * _shp
        except Exception:
            pass
    v_fp = state_value_fp(state)
    return (v_fp if side == Side.FREE_PEOPLES else -v_fp) + _sgn * _shp


def _pending_decision_kind_side(state: GameState,
                                fallback: 'Optional[Side]' = None):
    """(kind, side) of the highest-priority pending decision — in
    EXACTLY the branch order of get_legal_actions.  THE one cascade;
    _pending_decision_side and describe_pending are thin consumers.
    kind is a stable label ('combat:<phase>' | 'guide_change' |
    'hunt' | 'reveal_place' | 'discard' | 'declare' | 'eye' |
    'card_followup' | 'card_option'); (None, None) when no pending
    decision exists."""
    if state.pending_combat is not None:
        return ('combat:%s' % state.pending_combat.get('phase', '?'),
                state.pending_combat.get('deciding_side', fallback))
    if state.pending_guide_change is not None:
        return ('guide_change', Side.FREE_PEOPLES)
    if state.pending_hunt is not None:
        return ('hunt', Side.FREE_PEOPLES)
    # hunt-reveal placement (USER RULING 2026-08-16: learned FP
    # decision) — branch position mirrors get_legal_actions EXACTLY
    if getattr(state, '_reveal_place_due', False) \
            and not getattr(state, '_pending_hunt_damage', 0):
        return ('reveal_place', Side.FREE_PEOPLES)
    _pd = getattr(state, 'pending_discard', None)
    if _pd is not None:
        return ('discard', _pd['side'])
    if getattr(state, '_declare_phase', False):
        return ('declare', Side.FREE_PEOPLES)
    if getattr(state, '_eye_allocation_phase', False):
        return ('eye', Side.SHADOW)
    _pcf = getattr(state, 'pending_card_followup', None)
    if _pcf is not None:
        return ('card_followup',
                Side.FREE_PEOPLES if _pcf['side'] == 'fp'
                else Side.SHADOW)
    _pco = getattr(state, 'pending_card_option', None)
    if _pco is not None:
        # the DECIDER owns the choice (cross-side decisions: Lure of
        # the Ring / Stormcrow give the pick to the opponent - P1)
        return ('card_option',
                Side.FREE_PEOPLES
                if _pco.get('decider', _pco['side']) == 'fp'
                else Side.SHADOW)
    return (None, None)


def _pending_decision_side(state: GameState,
                           fallback: 'Optional[Side]' = None
                           ) -> 'Optional[Side]':
    """The side that owns the highest-priority pending decision — in
    EXACTLY the branch order of get_legal_actions.

    Single source of truth (v13 showcase crash, 2026-07-10): three
    routers each carried their own copy of this cascade and drifted —
    _next_acting_side put pending_card_option ahead of a coexisting
    FP hand-limit discard, get_legal_actions resolves the discard
    first, and its owner-guard rightly raised.  Any router that picks
    an acting side MUST use this helper; never inline the cascade.
    `fallback` is returned when pending_combat carries no
    deciding_side.  Returns None when no pending decision exists.
    (The cascade itself lives in _pending_decision_kind_side.)
    """
    return _pending_decision_kind_side(state, fallback)[1]


def describe_pending(state: GameState,
                     fallback_side: 'Optional[Side]' = None
                     ) -> 'Optional[dict]':
    """THE packaging of "what decision is pending": a dict
    {'kind', 'side', 'actions'} for the highest-priority open pending,
    or None when no pending decision exists (the caller is between
    actions — enumerate the active side directly).  Composes the two
    existing single-source answerers (_pending_decision_kind_side +
    get_legal_actions); drivers consume this instead of growing a
    third cascade (the old sided bridge's _pending_phase_name mirror
    is the defect this replaces)."""
    kind, side = _pending_decision_kind_side(state, fallback_side)
    if kind is None or side is None:
        return None
    return {'kind': kind, 'side': side,
            'actions': get_legal_actions(state, side)}


def _post_action_active(state: GameState, active: Side,
                        other: Side) -> Side:
    """Next side to act after `active` completed an action, when NO pending
    decision is open (shared by both self-play drivers — single source of
    truth). A just-resolved battle passes control to the NON-attacker (the
    whole battle was the attacker's single die); otherwise plain
    alternation to `other`."""
    _bra = getattr(state, '_battle_resolved_attacker', None)
    if _bra is not None:
        return (Side.SHADOW if _bra == Side.FREE_PEOPLES
                else Side.FREE_PEOPLES)
    return other


def _next_acting_side(state: GameState,
                      just_acted: Side) -> 'Optional[Side]':
    """Determine who acts next after just_acted took an action.

    Returns None if both sides have exhausted their dice.
    """
    # Pending decision phases dictate the acting side directly — these
    # override dice alternation. Without these checks, MCTS lookahead
    # assigned hunt/combat decisions to the wrong side: wrong net
    # evaluated the node and the value backup sign flipped for the
    # whole subtree (the outer game loop always got this right; only
    # in-tree search was affected).
    _ps = _pending_decision_side(state, fallback=just_acted)
    if _ps is not None:
        return _ps

    # An Elven Ring use is not an action (rulebook: the ring user "then
    # proceeds to take a regular action"): the marker set by the
    # elven_ring executor keeps that side's seat until its funded action
    # lands, so the live driver and in-tree routing agree with the
    # episode loop.
    _rfs = getattr(state, '_ring_free_side', None)
    if _rfs is not None:
        return _rfs

    # Multi-step actions keep the same side
    if (state.pending_army_move_die is not None or
            state.pending_muster_die is not None):
        return just_acted

    # Bug-1: a whole battle is ONE action spent by the ATTACKER — its
    # multi-step resolution may have flipped `just_acted` to the defender
    # (the last combat sub-decision). Alternate from the ATTACKER (the
    # true die-spender) so control passes to the non-attacker.
    _bra = getattr(state, '_battle_resolved_attacker', None)
    effective = _bra if _bra is not None else just_acted

    other = (Side.SHADOW if effective == Side.FREE_PEOPLES
             else Side.FREE_PEOPLES)

    def _unused(side):
        return sum(1 for d in state.dice.get(side, [])
                   if not d.used and d.current is not None)

    other_left = _unused(other)
    my_left = _unused(effective)

    if other_left > 0:
        return other
    if my_left > 0:
        return effective
    return None  # Both exhausted


def _expand_node(node: MCTSNode,
                 fp_net: 'WOTRValueNet',
                 sh_net: 'WOTRValueNet',
                 root_side: Side,
                 max_children: int = None) -> float:
    """Expand a node: create children using category-balanced pruning.

    Guarantees at least MIN_PER_CAT actions from each present action category
    make it into the children list (army_move, army_attack, fellowship_move,
    etc.). Remaining slots filled by highest-prior actions overall.
    This ensures MCTS always explores military and fellowship actions.
    """
    net = fp_net if node.side == Side.FREE_PEOPLES else sh_net
    _ov35 = _SIM_OVERLAY
    if _ov35 is not None and node.side == _ov35.opp_side \
            and node.state is not None:
        # 3A patch point: the opponent's menu is enumerated from
        # their (sampled) hand — the overlay must be live here
        try:
            _ov35.patch(node.state)
        except Exception:
            OVERLAY_STATS['patch_error'] += 1
            if OVERLAY_ASSERT:
                raise
    actions = get_legal_actions(node.state, node.side)

    if not actions:
        return _mcts_eval_state(node.state, root_side, fp_net, sh_net)

    priors, value = _get_priors_and_value(node.state, node.side, net, actions)
    # ROOT-CURRENCY INVARIANT: priors are the node side's opponent
    # model, but every value backed up by one search is the ROOT
    # side's net from the ROOT side's perspective — one currency per
    # tree.  Same-side expansions already got exactly that from
    # _get_priors_and_value; opponent expansions re-evaluate.
    if node.side != root_side:
        value = _mcts_eval_state(node.state, root_side, fp_net, sh_net)
        # OPPONENT combat-card window: the BC prior is replaced by the
        # closed-form posterior x best-response model (own-side card
        # windows keep net priors untouched)
        _pcw = getattr(node.state, 'pending_combat', None)
        if isinstance(_pcw, dict) and _pcw.get('phase') in (
                'attacker_combat_card', 'defender_combat_card') \
                and _pcw.get('deciding_side') == node.side:
            _pri35 = _opponent_card_priors(node.state, _pcw, actions,
                                           node.side)
            if _pri35 is not None:
                priors, node.w3 = _pri35

    if max_children is None:
        max_children = MCTS_MAX_CHILDREN or (1 << 30)   # 0 = no cap
    if len(actions) <= max_children:
        for a, p in zip(actions, priors):
            child = MCTSNode(state=None, side=node.side,
                             parent=node, action=a, prior=p)
            node.children.append(child)
    else:
        selected_list = _prune_candidates(actions, priors,
                                          max_children)
        total_prior = sum(priors[i] for i in selected_list)
        for i in selected_list:
            p = priors[i] / total_prior if total_prior > 0 else 1.0 / len(selected_list)
            child = MCTSNode(state=None, side=node.side,
                             parent=node, action=actions[i], prior=p)
            node.children.append(child)

    return value


def _prune_candidates(actions, priors, max_children=40,
                      free_guarantee=None):
    """Category-balanced candidate pruning (factored VERBATIM from
    _expand_node: min 3 per category, then fill by prior; returns
    sorted selected indices).

    RING-STARVATION FIX 1 (LAB task 14, 2026-08-09): free_guarantee
    (default = WOTR_FREE_CANDS) also guarantees the top-prior FREE-
    ACTION arms (<= _FREE_CANDS_N) a slot.  Mechanism: elven_ring
    shares the 'special_character' category with companion_move /
    companion_separate, whose priors always outrank its ~1e-4, so at
    big menus rings were excluded from the candidate set outright
    (measured: 53/76 = 70%% of ring-legal fsp_present probe states).
    The guarantee runs BEFORE the top-prior fill, so the children
    count still respects max_children.  Flag off = byte-identical
    legacy selection."""
    if free_guarantee is None:
        free_guarantee = _FREE_CANDS
    MIN_PER_CAT = 3
    cat_indices = {}  # cat_idx -> [(prior, action_idx)] sorted desc
    for i, a in enumerate(actions):
        canonical = _ATYPE_TO_CAT.get(a.action_type, 'other')
        idx = _ACTION_TYPE_TO_IDX.get(canonical,
                                      _ACTION_TYPE_TO_IDX['other'])
        cat_indices.setdefault(idx, []).append((priors[i], i))
    for cat in cat_indices:
        cat_indices[cat].sort(reverse=True)

    selected = set()
    # Phase 1: guarantee min slots per category
    for cat, items in cat_indices.items():
        for _, action_idx in items[:MIN_PER_CAT]:
            selected.add(action_idx)

    # Phase 1b (FIX 1): guarantee top-prior free-action arms
    if free_guarantee:
        free = sorted(
            [(priors[i], i) for i, a in enumerate(actions)
             if a.action_type in _FREE_ACTION_TYPES],
            reverse=True)
        for _, action_idx in free[:_FREE_CANDS_N]:
            selected.add(action_idx)

    # Phase 2: fill remaining slots with highest-prior unselected actions
    remaining = sorted(
        [(priors[i], i) for i in range(len(actions))
         if i not in selected],
        reverse=True)
    for _, action_idx in remaining:
        if len(selected) >= max_children:
            break
        selected.add(action_idx)

    return sorted(selected)


# SEARCH-WIDTH SETTINGS (USER 2026-08-22: "cut those caps but add extra
# sliders with those set as default").  Not legality - search budget:
# how many children an MCTS node expands (by prior, min 3 per action
# category) and how many distinct outcomes a chance node tracks.
# 0 = NO CAP.  The live panel's sliders set them through
# set_search_limits(); the defaults are the tuned values.
MCTS_MAX_CHILDREN = 40
CHANCE_K_MAX = 24


def set_search_limits(max_children=None, chance_k=None):
    """Set the search-width settings (None = leave; 0 = no cap)."""
    global MCTS_MAX_CHILDREN, CHANCE_K_MAX, _CHANCE_K_MAX, _KERNEL_K_MAX
    if max_children is not None:
        MCTS_MAX_CHILDREN = max(0, int(max_children))
    if chance_k is not None:
        CHANCE_K_MAX = max(0, int(chance_k))
        _CHANCE_K_MAX = CHANCE_K_MAX or (1 << 30)
        _KERNEL_K_MAX = CHANCE_K_MAX or (1 << 30)
    return MCTS_MAX_CHILDREN, CHANCE_K_MAX


_CHANCE_K_MAX = 24     # max distinct outcome branches per chance node
                       # (sampled nodes: support ~9-10, cap inert;
                       # kernel-exact combat nodes: p90 dropped mass
                       # 18-24% at 12 vs 1-3.5% at 24, measured)
_CHANCE_DRAWS = 16     # RNG draws whose frequencies weight the branches

# v30 (2026-07-26): CROSS-TURN SEARCH.  The old horizon ended every
# simulation at the end of die-spending, so hunt/corruption bleed,
# dice economics and the end-of-turn Military Victory check were
# INVISIBLE to search (the value net had to carry all cross-turn
# credit).  WOTR_TURN_HORIZON = number of turn boundaries a simulation
# may cross (0 restores the old behavior).  The boundary is a CHANCE
# edge (next-turn dice rolls + phase-1 draws consume RNG, so
# _apply_action_detect_rng turns it into a chance node whose branches
# sample independent next-turn rolls); the declare / eye-allocation /
# hand-limit phases inside the transition are resolved GREEDILY by the
# policy prior (mirroring the episode loop's noise-free phase picks) —
# the searched decisions stay the die-spends.
_TURN_HORIZON = int(os.environ.get('WOTR_TURN_HORIZON', '1') or 0)

# F1 (LAB 2026-08-09, proposal P-D): root Dirichlet-noise exemption
# for FREE actions (zero-die-cost same-side actions).  Mechanism: a
# free action's subtree re-contains every root continuation plus the
# converted die, so its Q is systematically inflated (+0.09 measured,
# ring-usage memory) and noise-forced visits at ~1e-4-prior big-menu
# ring arms hand it the visits PUCT would never give it.  Exempting
# free arms from the noise mix removes the noise channel WITHOUT
# touching their clean priors (thin-menu deliberate conversions keep
# their honest BC prior mass).  Default OFF = current behavior.
_FREE_ACTION_TYPES = frozenset(
    {'elven_ring', 'use_token', 'dwarven_ring'})
_FREE_NOISE_EXEMPT = os.environ.get(
    'WOTR_FREE_NOISE_EXEMPT', '0') == '1'

# RING-STARVATION FIXES (LAB task 14, 2026-08-09).  Post-launch
# finding: F1 + visit_tf removed the pathologies that MANUFACTURED
# ring usage (Dirichlet subsidy + option-value bias + temp sampling),
# leaving the honest ~1e-4 BC prior as the only exploration source —
# u-term at N=310 is ~0.02 (an order below typical Q-gaps), so free
# arms get ZERO visits -> zero target mass -> zero delta pressure:
# permanent starvation, the exact mirror of the old amplification.
# FIX 1 _FREE_CANDS: candidate-set guarantee (see _prune_candidates).
# FIX 2 _FREE_FORCE: KataGo-style ROOT forced-playout floor — the
#   top-_FREE_CANDS_N-prior free arms each get >= _FREE_FORCE visits
#   (honest MEAN estimates at fixed n, which per the F5 theorem
#   REDUCE the max-over-noise bias; the option-value component is
#   priced by F2's rho).  Budget <= _FREE_FORCE * _FREE_CANDS_N sims.
# FIX 3 target discount: forced visits are SUBTRACTED from the
#   visit_tf pi target (played arm keeps raw n), so the floor itself
#   is never distilled — target mass must be EARNED beyond the floor
#   (the F4 no-amplification guarantee survives).
# All default OFF; adoption values: _FREE_CANDS=1, _FREE_FORCE=12.
_FREE_CANDS = os.environ.get('WOTR_FREE_CANDS', '0') == '1'
_FREE_CANDS_N = 3
_FREE_FORCE = int(os.environ.get('WOTR_FREE_FORCE', '0') or 0)


def _apply_root_noise(children, rng_noise, frac=0.25,
                      exempt_free=None):
    """Mix Dirichlet noise into root children priors IN PLACE.
    exempt_free (default: _FREE_NOISE_EXEMPT): free-action arms keep
    their clean prior; the noise vector is renormalized over the
    non-exempt arms and the final prior vector renormalized to sum 1
    (exempt arms' RELATIVE priors are exactly preserved).  Returns the
    exempt mask for testability."""
    if exempt_free is None:
        exempt_free = _FREE_NOISE_EXEMPT
    k = len(children)
    ex = np.zeros(k, dtype=bool)
    if exempt_free:
        for i, c in enumerate(children):
            a = getattr(c, 'action', None)
            if a is not None and a.action_type in _FREE_ACTION_TYPES:
                ex[i] = True
        if ex.all():
            ex[:] = False      # degenerate all-free menu: noise as usual
    noise = np.asarray(rng_noise, dtype=np.float64)
    if not ex.any():
        # exemption inactive: EXACT legacy arithmetic (byte-preserved
        # default path — no renormalization pass)
        for i, c in enumerate(children):
            c.prior = (1.0 - frac) * c.prior + frac * noise[i]
        return ex
    noise = noise.copy()
    noise[ex] = 0.0
    s = float(noise.sum())
    if s > 0.0:
        noise = noise / s
    for i, c in enumerate(children):
        if not ex[i]:
            c.prior = (1.0 - frac) * c.prior + frac * float(noise[i])
    tot = float(sum(c.prior for c in children))
    if tot > 0.0:
        for c in children:
            c.prior = c.prior / tot
    return ex
# nets for the transition's greedy phase picks; set by mcts_search
# (single-threaded per worker process)
_SEARCH_NETS = (None, None)

# Lever 1 (session 34 scale work): tree reuse across consecutive
# decisions.  After a search picks action A, the chosen child's subtree
# is kept; the NEXT search reuses it as root iff (a) A was
# deterministic (chance edges drop the tree — the real outcome is one
# sample the tree can't claim), and (b) the incoming state's
# fingerprint EXACTLY matches the child state's fingerprint (any
# divergence — pendings, opponent action, engine side effects — means
# a fresh tree).  Reused roots skip re-determinization and fresh root
# noise (visit mass carried over dominates anyway).
_TREE_REUSE = os.environ.get('WOTR_TREE_REUSE', '1') != '0'
_TREE_CACHE = {'child': None, 'sig': None, 'root_side': None,
               'hits': 0, 'misses': 0}


def _state_sig(state, sort_dice=False):
    """Strong cheap fingerprint of everything search-relevant.

    sort_dice: canonicalize each side's dice by face — die ORDER never
    matters to game logic, but roll order splits identical multisets
    (chance-outcome dedup wants the multiset; tree reuse keeps order)."""
    try:
        armies = tuple(sorted(
            (k, len(a.units),
             sum(1 for u in a.units if u.kind == UnitKind.ELITE),
             len(a.leaders))
            for k, a in state.armies.items()))
        _per = [[(d.current.name if d.current else '-', d.used)
                 for d in state.dice.get(s9, [])]
                for s9 in (Side.FREE_PEOPLES, Side.SHADOW)]
        dice = tuple(x for sub in _per
                     for x in (sorted(sub) if sort_dice else sub))
        f = state.fellowship
        pol = tuple((n.value, state.political.boxes.get(n, 0),
                     bool(state.political.active.get(n)))
                    for n in Nation)
        return hash((
            state.turn, armies, dice,
            sanitize(game_logic.fellowship_region(state)),
            f.progress, f.corruption, f.hidden, f.mordor_entered,
            f.mordor_step, len(f.companions),
            len(state.hands[Side.FREE_PEOPLES].hand),
            len(state.hands[Side.SHADOW].hand),
            len(getattr(state, 'hunt_box', ())), pol,
            state.pending_combat is not None,
            state.pending_hunt is not None,
            getattr(state, 'pending_card_option', None) is not None,
            getattr(state, 'pending_discard', None) is not None,
            # full-pass review 2026-07-27: the remaining two pendings
            # _pending_decision_side routes on (defense-in-depth vs the
            # same reuse-hijack class as the phase flags below)
            getattr(state, 'pending_guide_change', None) is not None,
            getattr(state, 'pending_card_followup', None) is not None,
            # USER bug 2026-07-27 (eye_allocs list short vs turns): the
            # transient phase flags CHANGE the legal-action space — a
            # subtree stashed from the just-drained SH hand-discard
            # search sig-matched the eye-phase root (same state, same
            # hands) and served the NORMAL menu, so MCTS returned
            # 'pass' and the turn's eye allocation was silently
            # SKIPPED (same class could skip declares).
            bool(getattr(state, '_declare_phase', False)),
            bool(getattr(state, '_eye_allocation_phase', False)),
            # USER RULING 2026-08-16: an open reveal placement CHANGES
            # the legal-action space — same reuse-hijack class as the
            # phase flags above
            bool(getattr(state, '_reveal_place_due', False))))
    except Exception:
        return None


def _apply_action_detect_rng(parent_state: GameState, action: Action):
    """apply_action + did-it-consume-RNG flag (random-module state
    comparison; roll_d6_batch and every engine random path route
    through `random`)."""
    if OVERLAY_ASSERT:
        _overlay_assert_state(parent_state)
    pre_rng = random.getstate()
    new_state = apply_action(parent_state, action)
    return new_state, random.getstate() != pre_rng


def _chance_sig(state):
    """Outcome fingerprint: _state_sig plus the RNG-bearing content it
    deliberately omits (pending rolls/tiles, hand contents, terminal
    result) — distinct RNG outcomes must never merge."""
    def _prj(d):
        if not isinstance(d, dict):
            return d is not None
        out = []
        for k in sorted(d, key=str):
            v = d[k]
            if isinstance(v, (int, float, str, bool)) or v is None:
                out.append((str(k), v))
            elif isinstance(v, (list, tuple)):
                out.append((str(k), tuple(
                    x if isinstance(x, (int, float, str, bool))
                    else str(getattr(x, 'name', type(x).__name__))
                    for x in v)))
            else:
                out.append((str(k),
                            str(getattr(v, 'name', type(v).__name__))))
        return tuple(out)
    try:
        _rs9 = _SEARCH_ROOT_SIDE
        if _rs9 is not None:
            # 3A root-relative hands (design Section 4): the root
            # side's hand by identity (we observe our own draws) +
            # opponent per-deck COUNTS only — per-sim overlay hands
            # must never split RNG branches or corrupt exact weights.
            # Revealed opponent combat cards still distinguish via
            # the pending_combat projection (public once played).
            _os9 = (Side.SHADOW if _rs9 == Side.FREE_PEOPLES
                    else Side.FREE_PEOPLES)
            _oh9 = state.hands[_os9].hand
            _nc9 = sum(1 for c in _oh9
                       if isinstance(c, str) and '_Character_' in c)
            hands = (tuple(sorted(map(str, state.hands[_rs9].hand))),
                     _nc9, len(_oh9) - _nc9)
        else:
            hands = tuple(
                tuple(sorted(map(str, state.hands[s9].hand)))
                for s9 in (Side.FREE_PEOPLES, Side.SHADOW))
    except Exception:
        hands = None
    return (_state_sig(state, sort_dice=True),
            _prj(state.pending_combat),
            _prj(state.pending_hunt),
            _prj(getattr(state, 'pending_discard', None)),
            hands,
            getattr(state, '_search_roll_ms', None),
            bool(getattr(state, 'game_over', False)),
            str(getattr(state, 'winner', None) or ''),
            # hunt-resolution pendings: kernel branches split exactly
            # on these — distinct outcomes must never merge
            _prj(getattr(state, 'pending_card_option', None)),
            int(getattr(state, '_pending_hunt_damage', 0) or 0),
            _prj(getattr(state, 'pending_guide_change', None)),
            len(getattr(state, '_hobbit_place_queue', None) or ()),
            bool(getattr(state, '_reveal_place_due', False)))


def _select_outcome(chance: MCTSNode) -> 'Optional[MCTSNode]':
    """Sample/descend an outcome branch of a chance node.

    One fresh RNG draw per visit until _CHANCE_DRAWS are banked;
    draws dedup by _chance_sig into branches whose `prior` holds the
    draw count.  Descent then keeps each branch's visit share at its
    draw-frequency share, so the node's Q is the probability-weighted
    expectation (uniform least-visited descent weighted a 5/8-likely
    outcome at 1/4 and sat on won games — mordor_push fixture)."""
    outs = chance.outcomes
    drawn = sum(b.prior for b in outs) + chance.drops
    if chance.drops >= _CHANCE_DRAWS:
        drawn = _CHANCE_DRAWS          # exact-weight mode banks attempts
    if drawn < _CHANCE_DRAWS:
        st = None
        _overlay_prepare_apply(chance.parent.state, chance.side,
                               chance.action)
        try:
            st, _ = _apply_action_detect_rng(chance.parent.state,
                                             chance.action)
        except Exception:
            pass
        if st is not None:
            sg = _chance_sig(st)
            ms = getattr(st, '_search_roll_ms', None) \
                if chance.action is not None \
                and chance.action.action_type == 'end_turn' else None
            hit = next((b for b in outs if b.out_sig == sg), None)
            if hit is not None:
                if ms is None:
                    hit.prior += 1.0
                else:
                    chance.drops += 1  # attempt banked; weight exact
            elif len(outs) < _CHANCE_K_MAX:
                branch = MCTSNode(state=st, side=chance.side,
                                  parent=chance, action=None,
                                  prior=(_roll_ms_prob(ms)
                                         if ms is not None else 1.0))
                branch.out_sig = sg
                outs.append(branch)
                if ms is not None:
                    chance.drops += 1
                return branch
            else:
                # cap overflow: weights cover the tracked set, but the
                # draw still fills the bank (see MCTSNode.drops)
                chance.drops += 1
        if not outs:
            return None
    tot = sum(b.prior for b in outs)
    return min(outs, key=lambda b: (b.visit_count
                                    - chance.visit_count
                                    * (b.prior / tot)))


def _greedy_prior_action(state: GameState, side: Side, net,
                         actions) -> 'Optional[Action]':
    """Argmax-prior pick for the transition's phase decisions (declare /
    eyes / hand-limit) — the in-tree twin of the episode loop's
    noise-free phase MCTS, collapsed to one net call.

    1.226 note: deliberately UNCHANGED by the Gumbel selection swap —
    the priors here come from _get_priors_and_value, whose total
    logits already include the residual policy delta, so this pick
    inherits the corrected priors as the delta trains."""
    if not actions:
        return None
    try:
        priors, _v = _get_priors_and_value(state, side, net, actions)
        return actions[int(np.argmax(priors))]
    except Exception:
        return None


def _advance_turn_for_search(state: GameState) -> None:
    """In-tree turn transition (WOTR_TURN_HORIZON, v30).

    Order mirrors the episode loop exactly: end-of-turn Victory Check
    (Military included — previously invisible to search) -> phase 1
    via _safe_next_turn (recover, draws, hand-limit pending) ->
    hand-limit greedy resolve -> Mordor-entry pre-step + Balrog ->
    auto/greedy declare -> greedy eye allocation -> both action rolls.
    Draws + rolls consume RNG, so the caller's edge becomes a chance
    node sampling independent next-turn rolls.  On an end-of-turn
    victory the state is flagged and NOT advanced."""
    res = game_logic.check_all_victory_conditions(state)
    if game_logic.should_game_end(res, end_of_turn=True):
        state._search_endturn_victory = dict(res)
        return
    state._search_turn_crossings = getattr(
        state, '_search_turn_crossings', 0) + 1
    fpn, shn = _SEARCH_NETS
    if _safe_next_turn(state):
        # P0 2026-08-15: rollover ring victory — flag, don't play on
        state._search_endturn_victory = dict(
            game_logic.check_all_victory_conditions(state))
        return
    # hand-limit discards opened by the turn-start draws: resolve
    # greedily here (the episode loop searches them; in-tree they are
    # part of the chance transition)
    for _hl9 in range(4):
        _pd9 = getattr(state, 'pending_discard', None)
        if _pd9 is None:
            break
        _dside9 = _pd9['side']
        _dacts9 = get_legal_actions(state, _dside9)
        _pick9 = _greedy_prior_action(
            state, _dside9,
            fpn if _dside9 == Side.FREE_PEOPLES else shn, _dacts9)
        if _pick9 is None and _dacts9:
            _pick9 = _dacts9[0]
        if _pick9 is None:
            state.pending_discard = None
            break
        try:
            _execute_action_inner(state, _pick9)
        except Exception:
            state.pending_discard = None
            break
    # Phase 2 pre-steps (twin of the episode loop).
    # FINAL RULING (USER 2026-08-16): Mordor entry is TURN-START ONLY
    # and POSITIONAL, and the DECLARATION OPPORTUNITY COMES FIRST -
    # the entry check runs AFTER the Fellowship phase below.  The old
    # order (entry check before the declaration) was the day-one bug:
    # a fellowship declared onto a gate waited a full turn to enter.
    # Balrog stationary trigger stays BEFORE the declaration (almanac
    # C17: 'remaining stationary at turn start' - a declaration away
    # must not have happened yet).
    if not state.fellowship.mordor_entered:
        _balrog_stationary_turn_start(state)
    _auto9 = False
    if not state.fellowship.mordor_entered:
        _auto9 = _auto_enter_mordor(state)
    # Fellowship phase: greedy declare (pass = do nothing).
    # In-place declares are legal at progress 0 (allow_in_place).
    if not _auto9 and not state.fellowship.mordor_entered:
        if _get_declare_actions(state, allow_in_place=True):
            state._declare_phase = True
            try:
                _dacts = get_legal_actions(state, Side.FREE_PEOPLES)
                _pick = _greedy_prior_action(
                    state, Side.FREE_PEOPLES, fpn, _dacts)
            finally:
                state._declare_phase = False
            if _pick is not None and \
                    _pick.action_type == 'fellowship_declare':
                try:
                    _execute_action_inner(state, _pick)
                except Exception:
                    pass
    # Mordor entry check - AFTER the declaration opportunity, same
    # turn start (USER FINAL 2026-08-16): a gate position reached by a
    # fresh declaration, standing from an earlier one, or by a REVEAL
    # enters NOW.  _check_mordor_entry is positional and idempotent -
    # the single answerer for all three triggers.
    if not state.fellowship.mordor_entered:
        game_logic._check_mordor_entry(state)
    # Hunt allocation: greedy eye count
    _shu = [d for d in state.dice.get(Side.SHADOW, []) if not d.used]
    if _shu:
        state._eye_allocation_phase = True
        try:
            _eacts = get_legal_actions(state, Side.SHADOW)
            _epick = _greedy_prior_action(state, Side.SHADOW, shn, _eacts)
        finally:
            state._eye_allocation_phase = False
        _ne = (int(_epick.params.get('count', 0) or 0)
               if _epick is not None else 1)
        if _ne > 0:
            _idx = [i for i, d in enumerate(state.dice[Side.SHADOW])
                    if not d.used][:_ne]
            try:
                game_logic.allocate_eyes(state, _ne, indices=_idx)
            except Exception:
                pass
    _hb0 = len(getattr(state, 'hunt_box', ()) or ())
    game_logic.roll_action_dice(state, pool=Side.FREE_PEOPLES)
    game_logic.roll_action_dice(state, pool=Side.SHADOW)
    _hb1 = len(getattr(state, 'hunt_box', ()) or ())
    # rolled multiset stash: rolled Eyes migrate to the hunt box, so
    # the post-roll pool alone under-counts the SH roll
    state._search_roll_ms = (
        tuple(sorted(d.current.name for d in
                     state.dice.get(Side.FREE_PEOPLES, [])
                     if d.current is not None)),
        tuple(sorted(d.current.name for d in
                     state.dice.get(Side.SHADOW, [])
                     if d.current is not None)),
        max(0, _hb1 - _hb0))


def _roll_ms_prob(ms) -> float:
    """Exact multinomial probability of a stashed roll multiset
    (uniform 1/6 faces, sides independent)."""
    fpf, shf, eyes = ms

    def _mprob(faces, extra_eyes=0):
        n = len(faces) + extra_eyes
        if n <= 0:
            return 1.0
        cnt = {}
        for f in faces:
            cnt[f] = cnt.get(f, 0) + 1
        if extra_eyes:
            cnt['EYE'] = cnt.get('EYE', 0) + extra_eyes
        num = math.factorial(n)
        den = 1
        for c in cnt.values():
            den *= math.factorial(c)
        return (num / den) * (1.0 / 6.0) ** n

    return _mprob(fpf) * _mprob(shf, eyes)


# ------------------------------------------------------------------
# ISMCTS 3A (design_ismcts_determinization.md, USER-certified):
# per-SIMULATION opponent-hand re-determinization.  Each simulation
# samples a fresh opponent hand from the exchangeable-uniform
# posterior over U_d = deck + opp hand + opp FACE-DOWN discards per
# deck (L1 fix), reshuffles the OWN deck orders (L2 fix — own HAND
# never touched) and the hunt-pool ORDER (composition public, order
# private).  The sample lives in a per-sim overlay; every in-search
# state read is patch-preceded, so stale hidden zones in stashed
# tree states are inert.  Replaces root-once determinization.
# ------------------------------------------------------------------

OVERLAY_ASSERT = False   # TEST builds: verify every in-search apply
                         # reads a patch-preceded state (module flag,
                         # set by validation scripts — never an env)
OVERLAY_STATS = {
    'samples': 0, 'sample_error': 0, 'patches': 0, 'patch_error': 0,
    'cond_swaps': 0, 'require_fail': 0, 'mask_empty': 0,
    'mask_error': 0, 'lazy_event': 0, 'lazy_discard': 0,
    'lazy_combat': 0, 'lazy_no_template': 0, 'lazy_rejected': 0,
    'det_draw_chance': 0, 'assert_checks': 0}

_SIM_OVERLAY = None       # active _SimOverlay (per-worker; MCTS is
                          # single-threaded per process)
_SEARCH_ROOT_SIDE = None  # root side of the running search (feeds the
                          # root-relative _chance_sig hands component)

_OV_DECKS = ('char', 'strat')
_OV_LAZY_PRIOR = 0.02     # floor prior for 3A lazily-added children
_OV_LAZY_MIN_VISITS = 16  # event-window additions need a dry-run
                          # (~ms) per candidate: only nodes with real
                          # visit mass pay it — the mask itself stays
                          # universal, so no sim ever descends an
                          # edge its sample cannot play


def overlay_stats_summary() -> dict:
    return dict(OVERLAY_STATS)


def _ov_ok(c) -> bool:
    """Card entries the overlay may move (tracked-hand placeholders
    stay pinned in place)."""
    return isinstance(c, str) and not c.startswith('_unknown_')


def _ov_deck_of(cid: str) -> str:
    # one deck classifier (game_logic.card_deck_of): live placeholders
    # sort into their minted deck instead of defaulting to strategy
    return ('char' if game_logic.card_deck_of(cid) == 'character'
            else 'strat')


def _ov_deck_attr(tag: str, d: str) -> str:
    return tag + ('_char_deck' if d == 'char' else '_strat_deck')


def _ov_tile_key(t) -> tuple:
    return (t.damage, t.is_eye, t.reveal,
            bool(getattr(t, 'is_special', False)),
            getattr(t, 'special_name', None),
            bool(getattr(t, 'stops_movement', False)),
            bool(getattr(t, 'is_dice_roll', False)),
            bool(getattr(t, 'is_fellowship_special', False)))


def combat_card_pool(state: GameState, pc: dict, is_attacker: bool,
                     deck_type=None, extra=()):
    """THE ONE ANSWERER for "which combat cards could this side play,
    from what is UNSEEN?"  (USER, stated twice: "the AI should NOT be
    allowed to see my hand.")

    The pool is hand + face-down discards + both decks - i.e. every
    card of that side's that the opponent has NOT seen - filtered to
    the cards that are actually PLAYABLE in this combat context.
    Because it unions the hand with the decks, it is an INVARIANT of
    the opponent's information set: moving a card between hand and
    deck (which is all a hand swap does) leaves the multiset alone.
    Copy-weighted; returns sorted list[(card_id, copies)].

    deck_type  restrict to one deck ('character'/'strategy'), e.g. a
               committed card whose TYPE is public but identity is not
    extra      card ids to fold back in (the committed copy, so the
               pool is the PRE-commit one)

    Two consumers, one body: _atk_commit_mixture (the D3 opponent-model
    blinding, 2026-08-16) and the bridge's 'playable:' chat line (live
    0818 F3) - the latter used to enumerate the human's REAL hand."""
    from cards import get_card
    side9 = pc['attacker_side'] if is_attacker else pc['defender_side']
    tag9 = 'fp' if side9 == Side.FREE_PEOPLES else 'sh'
    tmp9, ok9 = _combat_card_check_ctx(state, pc, is_attacker)
    if not ok9:
        return []
    hp9 = state.hands.get(side9)
    cards9 = (
        [c for c in (hp9.hand if hp9 else []) if _ov_ok(c)]
        + [c for c in (hp9.discarded_face_down
                       if hp9 else []) if _ov_ok(c)]
        + [c for c in (getattr(state, tag9 + '_char_deck',
                               None) or []) if _ov_ok(c)]
        + [c for c in (getattr(state, tag9 + '_strat_deck',
                               None) or []) if _ov_ok(c)])
    for c in (extra or ()):
        if _ov_ok(c):
            cards9.append(c)
    mix9, rej9 = {}, set()
    for c in cards9:
        if deck_type is not None and _ov_deck_of(c) != deck_type:
            continue
        if c in mix9:
            mix9[c] += 1
            continue
        if c in rej9:
            continue
        try:
            if _combat_card_playable(state, get_card(c), tmp9,
                                     is_attacker):
                mix9[c] = 1
            else:
                rej9.add(c)
        except Exception:
            rej9.add(c)
    return sorted(mix9.items())


def _atk_commit_mixture(state: GameState, pc: dict):
    """What could the attacker have COMMITTED?  The D3 blinding
    distribution (USER 2026-08-16: identity hidden, TYPE visible):
    playable cards of the committed TYPE from the attacker's
    PRE-COMMIT unseen pool - hand + face-down + decks PLUS the
    committed copy restored.  Restoring the committed copy makes the
    pool an invariant of the defender's information set: swapping
    which same-type card was committed swaps one pool copy for
    another, leaving the multiset identical.  Copy-weighted.
    Returns a sorted list[(card_id, copies)]; [] when nothing of the
    committed type is representable/playable.
    Body factored out to combat_card_pool (live 0818): the bridge's
    'playable:' line needs the same pool without the type filter."""
    _committed9 = pc.get('atk_combat_card')
    if not _committed9:
        return []
    _hid9 = pc.get('atk_hidden_cc')
    if _hid9:
        # K1 hidden commit: the TYPE is the record; the card itself is
        # still part of the attacker's unseen pool (no copy to restore)
        return combat_card_pool(
            state, pc, True,
            deck_type='char' if _hid9.get('deck') == 'character'
            else 'strat')
    return combat_card_pool(state, pc, True,
                            deck_type=_ov_deck_of(_committed9),
                            extra=(_committed9,))


class _SimOverlay:
    """One simulation's determinization: opponent per-deck (hand,
    deck-order, face-down) partition of U_d, own deck orders, hunt
    pool order.  Deck/pool orders are stored in DRAW order (index 0 =
    next draw); the engine pops list ENDS, so writes reverse."""

    __slots__ = ('root_side', 'opp_side', 'opp_tag', 'own_tag',
                 'hand', 'deck', 'fdown', 'own_deck', 'pool',
                 'commit_cid', 'commit_true', 'commit_bkey',
                 'commit_slot', 'commit_hidden')

    def __init__(self, root_state, root_side):
        self.root_side = root_side
        self.opp_side = (Side.SHADOW if root_side == Side.FREE_PEOPLES
                         else Side.FREE_PEOPLES)
        self.opp_tag = ('fp' if self.opp_side == Side.FREE_PEOPLES
                        else 'sh')
        self.own_tag = 'sh' if self.opp_tag == 'fp' else 'fp'
        # ROOT/SEARCH BLINDING (USER RULING 2026-08-16, verbatim: "NO
        # YOU CANNOT SEE THE CARD THAT IS COMMITTED").  When the ROOT
        # player is the DEFENDER and the attacker's combat card lies
        # committed face-down, its IDENTITY is hidden information -
        # sample one of the committed TYPE from the D3 mixture
        # (_atk_commit_mixture, the same one answerer the opponent
        # model uses) BEFORE the zone build, so the zones below can be
        # built from the identity-invariant PRE-COMMIT pool minus the
        # sample.  TYPE stays visible, commitment fact stays visible;
        # root=attacker never blinds (their own commit is known).
        self.commit_cid = None
        self.commit_true = None
        self.commit_bkey = None
        self.commit_hidden = False
        self.commit_slot = getattr(root_state, '_commit_slot', None)
        _commit_true9 = None
        _pcb9 = getattr(root_state, 'pending_combat', None)
        if (_pcb9 and _pcb9.get('atk_combat_card')
                and _pcb9.get('attacker_side') == self.opp_side):
            _mixb9 = _atk_commit_mixture(root_state, _pcb9)
            # K1: a hidden (human) commit is the same type-only world;
            # the sampled identity stands in for the unrevealed card so
            # the in-tree round can resolve.  Nothing is restored to the
            # pool (the hidden card never left the engine's unseen set).
            self.commit_hidden = bool(_pcb9.get('atk_hidden_cc'))
            if self.commit_hidden:
                self.commit_bkey = _pcb9.get('rounds_fought', 0)
            if _mixb9:
                self.commit_cid = random.choices(
                    [c for c, _w in _mixb9],
                    weights=[_w for _c, _w in _mixb9], k=1)[0]
                # only THIS battle's THIS card window: the root clone's
                # pc carries a nonce (stamped by mcts_search) that a
                # fresh in-tree battle's pc lacks, and rounds_fought
                # pins the window (an in-tree commit in a later window
                # is search-imagination and keeps the identity the
                # search chose)
                self.commit_bkey = _pcb9.get('rounds_fought', 0)
                if not self.commit_hidden:
                    _commit_true9 = _pcb9.get('atk_combat_card')
                    self.commit_true = _commit_true9
                OVERLAY_STATS['commit_blind'] = OVERLAY_STATS.get(
                    'commit_blind', 0) + 1
        info = root_state.hands.get(self.opp_side)
        o_hand = [c for c in (info.hand if info else []) if _ov_ok(c)]
        o_fd = [c for c in (getattr(info, 'discarded_face_down', None)
                            or []) if _ov_ok(c)]
        self.hand, self.deck, self.fdown = {}, {}, {}
        for d in _OV_DECKS:
            dl = [c for c in (getattr(root_state,
                                      _ov_deck_attr(self.opp_tag, d),
                                      None) or []) if _ov_ok(c)]
            hd = [c for c in o_hand if _ov_deck_of(c) == d]
            fd = [c for c in o_fd if _ov_ok(c)
                  and _ov_deck_of(c) == d]
            pool = dl + hd + fd
            if (_commit_true9 is not None
                    and _ov_deck_of(_commit_true9) == d
                    and self.commit_cid != _commit_true9):
                # this sim's world: the attacker committed commit_cid,
                # not the true card - the unseen pool is the PRE-COMMIT
                # pool (post-commit + true) minus the sample.  Same
                # multiset in every same-type-swap world.
                pool.append(_commit_true9)
                if self.commit_cid in pool:
                    pool.remove(self.commit_cid)
            elif (self.commit_hidden and self.commit_cid is not None
                    and _ov_deck_of(self.commit_cid) == d
                    and self.commit_cid in pool):
                # hidden commit: the sampled identity is the committed
                # card, so it leaves this sim's unseen pool
                pool.remove(self.commit_cid)
            # canonical order before the shuffle: equal multisets must
            # shuffle identically whatever list order the state
            # carried (the blinding invariance depends on it; the
            # sampling distribution is unchanged - a uniform shuffle
            # of a sorted list is a uniform shuffle)
            pool.sort()
            random.shuffle(pool)
            self.hand[d] = pool[:len(hd)]
            self.deck[d] = pool[len(hd):len(hd) + len(dl)]
            self.fdown[d] = pool[len(hd) + len(dl):]
        self.own_deck = {}
        for d in _OV_DECKS:
            dl = [c for c in (getattr(root_state,
                                      _ov_deck_attr(self.own_tag, d),
                                      None) or []) if _ov_ok(c)]
            random.shuffle(dl)
            self.own_deck[d] = dl
        keys = [_ov_tile_key(t)
                for t in (getattr(root_state, 'hunt_pool', None) or [])]
        random.shuffle(keys)
        self.pool = keys

    def has(self, cid: str) -> bool:
        return cid in self.hand[_ov_deck_of(cid)]

    def require(self, cid: str) -> bool:
        """Condition the sample on 'cid held' (a descended public play
        implies holding): swap cid into the hand from deck/face-down,
        displacing a random same-deck hand card.  False = cid outside
        the unseen pool (inconsistent edge — counted by caller)."""
        d = _ov_deck_of(cid)
        h = self.hand[d]
        if cid in h:
            return True
        if not h:
            return False
        vic = h[random.randrange(len(h))]
        if cid in self.deck[d]:
            self.deck[d][self.deck[d].index(cid)] = vic
        elif cid in self.fdown[d]:
            self.fdown[d][self.fdown[d].index(cid)] = vic
        else:
            return False
        h.remove(vic)
        h.append(cid)
        return True

    def _recon(self, d, s_hand, s_deck, s_fd):
        """Reconcile one opponent deck's overlay zones to the state's
        public zone counts: multiset diffs are path reveals (leave the
        pool, hand first) / pool entries (into the deck); count
        rebalances are hidden draws (deck top -> hand) and hidden
        discards (uniform hand card -> face-down, the exchangeable
        choice under the Q1 posterior)."""
        from collections import Counter
        oh, od, of = self.hand[d], self.deck[d], self.fdown[d]
        sz = Counter(s_hand)
        sz.update(c for c in s_deck if _ov_ok(c))
        sz.update(s_fd)
        oz = Counter(oh)
        oz.update(od)
        oz.update(of)
        for cid, n in (oz - sz).items():
            for _ in range(n):
                if cid in oh:
                    oh.remove(cid)
                elif cid in od:
                    od.remove(cid)
                elif cid in of:
                    of.remove(cid)
        for cid, n in (sz - oz).items():
            for _ in range(n):
                od.insert(random.randrange(len(od) + 1), cid)
        h = len(s_hand)
        m = sum(1 for c in s_deck if _ov_ok(c))
        while len(oh) < h and od:
            oh.append(od.pop(0))
        while len(oh) < h and of:
            oh.append(of.pop(random.randrange(len(of))))
        while len(oh) > h:
            of.append(oh.pop(random.randrange(len(oh))))
        while len(od) < m and of:
            od.insert(random.randrange(len(od) + 1),
                      of.pop(random.randrange(len(of))))
        while len(od) > m:
            of.append(od.pop(random.randrange(len(od))))

    def patch(self, state) -> None:
        """Write this simulation's sampled hidden zones into `state`
        after reconciling to its public zone counts.  Idempotent; the
        union multiset of (deck + hand + face-down) per deck is
        preserved, so the W3 closed-form posterior computed on a
        patched state equals the true-state one exactly."""
        from collections import Counter
        OVERLAY_STATS['patches'] += 1
        info = state.hands.get(self.opp_side)
        if info is not None:
            for d in _OV_DECKS:
                sdeck = getattr(state, _ov_deck_attr(self.opp_tag, d),
                                None)
                if sdeck is None:
                    continue
                s_hand = [c for c in info.hand
                          if _ov_ok(c) and _ov_deck_of(c) == d]
                s_fd = [c for c in info.discarded_face_down
                        if _ov_ok(c) and _ov_deck_of(c) == d]
                # ROOT BLINDING (USER 2026-08-16): the recon target is
                # the BLINDED world's multiset (state + true committed
                # - sampled committed).  Without this the reconciler
                # forced the TRUE identity back into the zones and the
                # state re-leaked it from the second simulation on
                # (probe: sim-1 overlay divergence).
                s_deck_l = sdeck
                if (self.commit_cid is not None
                        and self.commit_true is not None
                        and self.commit_cid != self.commit_true
                        and _ov_deck_of(self.commit_true) == d):
                    for _sl9 in (s_hand, s_fd):
                        if self.commit_cid in _sl9:
                            _sl9[_sl9.index(self.commit_cid)] = \
                                self.commit_true
                            break
                    else:
                        if self.commit_cid in sdeck:
                            s_deck_l = list(sdeck)
                            s_deck_l[s_deck_l.index(
                                self.commit_cid)] = self.commit_true
                self._recon(d, s_hand, s_deck_l, s_fd)
                sdeck[:] = list(reversed(self.deck[d]))
            info.hand[:] = (
                [c for c in info.hand if not _ov_ok(c)]
                + self.hand['char'] + self.hand['strat'])
            info.discarded_face_down[:] = (
                [c for c in info.discarded_face_down
                 if not _ov_ok(c)]
                + self.fdown['char'] + self.fdown['strat'])
        for d in _OV_DECKS:
            sdeck = getattr(state, _ov_deck_attr(self.own_tag, d),
                            None)
            if sdeck is None:
                continue
            o = self.own_deck[d]
            sc = Counter(c for c in sdeck if _ov_ok(c))
            oc = Counter(o)
            for cid, n in (oc - sc).items():
                for _ in range(n):
                    o.remove(cid)
            for cid, n in (sc - oc).items():
                for _ in range(n):
                    o.insert(random.randrange(len(o) + 1), cid)
            sdeck[:] = list(reversed(o))
        pool = getattr(state, 'hunt_pool', None)
        if pool is not None:
            skeys = Counter(_ov_tile_key(t) for t in pool)
            oc = Counter(self.pool)
            for k, n in (oc - skeys).items():
                for _ in range(n):
                    self.pool.remove(k)
            for k, n in (skeys - oc).items():
                for _ in range(n):
                    self.pool.insert(
                        random.randrange(len(self.pool) + 1), k)
            buckets = {}
            for t in pool:
                buckets.setdefault(_ov_tile_key(t), []).append(t)
            pool[:] = list(reversed(
                [buckets[k].pop() for k in self.pool]))
        # ROOT/SEARCH BLINDING (USER RULING 2026-08-16): the sampled
        # committed-card identity replaces the true one before the
        # engine can resolve with it — only for THIS battle's THIS
        # card window (bkey); a popped key (revealed card) or an
        # in-tree later-window commit is never touched.
        if self.commit_cid is not None or self.commit_hidden:
            _pcb9 = getattr(state, 'pending_combat', None)
            if (_pcb9 and _pcb9.get('atk_combat_card')
                    and _pcb9.get('_commit_nonce')
                    and _pcb9.get('rounds_fought', 0)
                    == self.commit_bkey):
                # K1: the hidden commit becomes THIS sim's sampled
                # identity (no representable candidate -> no card), so
                # the reveal gate never parks an in-tree round
                if _pcb9.get('atk_hidden_cc'):
                    _pcb9.pop('atk_hidden_cc', None)
                    _pcb9['atk_combat_card'] = self.commit_cid
                elif self.commit_cid is not None:
                    _pcb9['atk_combat_card'] = self.commit_cid
            # the blanked face-up-discard slot (root blinding) shows
            # THIS simulation's sampled identity, so the encoding's
            # card-status plane reads a sampled world, never the truth
            if self.commit_slot is not None:
                _dk9, _ix9 = self.commit_slot
                _dfu9 = getattr(state, 'discard_face_up', None)
                if (isinstance(_dfu9, dict) and _dk9 in _dfu9
                        and _ix9 < len(_dfu9[_dk9])):
                    _dfu9[_dk9][_ix9] = self.commit_cid


def _overlay_prepare_apply(parent_state, actor_side, action) -> None:
    """3A patch point: sync the active overlay to this state's public
    zone counts and write the sampled hidden zones, conditioning on
    the edge's card requirement (an opponent hand-consuming action
    must hold its card in THIS determinization)."""
    ov = _SIM_OVERLAY
    if ov is None:
        return
    try:
        ov.patch(parent_state)
        if action is None or actor_side != ov.opp_side:
            return
        at = action.action_type
        if at in ('play_event', 'discard_card') \
                or (at == 'combat_card_play'
                    and not (action.params or {}).get('from_table')):
            cid = (action.params or {}).get('card_id')
            if cid and _ov_ok(cid) and not ov.has(cid):
                if ov.require(cid):
                    OVERLAY_STATS['cond_swaps'] += 1
                    ov.patch(parent_state)
                else:
                    OVERLAY_STATS['require_fail'] += 1
    except Exception:
        OVERLAY_STATS['patch_error'] += 1
        if OVERLAY_ASSERT:
            raise


def _overlay_assert_state(state) -> None:
    """TEST-build invariant (design 3A): every in-search apply reads a
    patch-preceded state — its opponent hand must BE the active
    overlay's hand."""
    ov = _SIM_OVERLAY
    if ov is None or not OVERLAY_ASSERT:
        return
    OVERLAY_STATS['assert_checks'] += 1
    info = state.hands.get(ov.opp_side)
    got = sorted(c for c in (info.hand if info else [])
                 if _ov_ok(c))
    want = sorted(ov.hand['char'] + ov.hand['strat'])
    if got != want:
        raise AssertionError(
            'overlay fingerprint mismatch: state=%r overlay=%r'
            % (got, want))


def _hidden_zone_lens(state) -> tuple:
    """Sizes of every hidden-ORDER zone.  A size change with no RNG
    consumed is a draw/pop of a re-determinized order (own draw, hunt
    pool pop): a CHANCE event under 3A — branches must sample the
    per-sim orders instead of fusing one."""
    return (len(getattr(state, 'fp_char_deck', None) or ()),
            len(getattr(state, 'fp_strat_deck', None) or ()),
            len(getattr(state, 'sh_char_deck', None) or ()),
            len(getattr(state, 'sh_strat_deck', None) or ()),
            len(getattr(state, 'hunt_pool', None) or ()))


def _ov_window_kind(node) -> str:
    """Classify (once) whether this opponent node is a card window the
    overlay must mask/extend."""
    k = node.ov_win
    if k is None:
        k = 'none'
        for c in node.children:
            at = (c.action.action_type if c.action is not None else '')
            if at == 'discard_card':
                k = 'discard'
                break
            if at in ('combat_card_play', 'combat_card_none'):
                k = 'combat'
                break
            if at == 'play_event':
                k = 'event'
        node.ov_win = k
    return k


def _ov_child_ok(child, node, ov) -> bool:
    """Compatibility mask: an opponent card edge is descendable this
    simulation only if the overlay hand holds its card.  At combat
    windows this is the W3 composition (mission item 5): the overlay
    hand IS the sample — per-sim PUCT maximizes over HELD candidates,
    so the node's Q averages to exactly posterior x best-response
    across simulations; the W3 priors remain exploration guidance +
    the lazy-add mass source (overlay-invariant, and never branch
    WEIGHTS, so no membership probability is double-counted)."""
    a = child.action
    if a is None:
        return True
    at = a.action_type
    if at in ('play_event', 'discard_card') \
            or (at == 'combat_card_play'
                and not (a.params or {}).get('from_table')):
        cid = (a.params or {}).get('card_id')
        return (not cid) or (not _ov_ok(cid)) or ov.has(cid)
    return True


def _overlay_lazy_children(node, ov) -> str:
    """3A lazy child addition at opponent card windows: overlay-hand
    cards missing from the stored enumeration (it was built under a
    DIFFERENT simulation's sample) get children so no determinization
    is silently card-starved.  Bounded by hand size; per-node memo.
    Combat windows draw their mass from the node's W3 map (the decline
    arm gives that mass back); event windows dry-run playability and
    reuse a same-deck template child's die binding."""
    kind = _ov_window_kind(node)
    if kind == 'none':
        return kind
    ov.patch(node.state)
    if node.ov_seen is None:
        node.ov_seen = set()
    hand_cards = ov.hand['char'] + ov.hand['strat']
    if kind == 'discard':
        have = {(c.action.params or {}).get('card_id')
                for c in node.children if c.action is not None
                and c.action.action_type == 'discard_card'}
        for cid in hand_cards:
            if cid in have or cid in node.ov_seen:
                continue
            node.ov_seen.add(cid)
            node.children.append(MCTSNode(
                state=None, side=node.side, parent=node,
                action=Action(die_idx=-1, action_type='discard_card',
                              params={'card_id': cid,
                                      'side': ov.opp_tag}),
                prior=_OV_LAZY_PRIOR))
            OVERLAY_STATS['lazy_discard'] += 1
    elif kind == 'combat':
        w3 = node.w3
        if w3 is None:
            return kind
        have = {(c.action.params or {}).get('card_id')
                for c in node.children if c.action is not None
                and c.action.action_type == 'combat_card_play'}
        decline = next(
            (c for c in node.children if c.action is not None
             and c.action.action_type == 'combat_card_none'), None)
        for cid in hand_cards:
            if cid in have or cid in node.ov_seen:
                continue
            node.ov_seen.add(cid)
            pm = float(w3.get(cid, 0.0))
            if pm <= 0.0:
                continue
            node.children.append(MCTSNode(
                state=None, side=node.side, parent=node,
                action=Action(die_idx=-1,
                              action_type='combat_card_play',
                              params={'card_id': cid,
                                      'side': ov.opp_tag}),
                prior=max(pm, 1e-3)))
            if decline is not None:
                decline.prior = max(decline.prior - pm, 1e-3)
            OVERLAY_STATS['lazy_combat'] += 1
    else:
        if node.visit_count < _OV_LAZY_MIN_VISITS:
            return kind
        tmpl = {}
        have = set()
        for c in node.children:
            if c.action is not None \
                    and c.action.action_type == 'play_event':
                cid0 = (c.action.params or {}).get('card_id')
                have.add(cid0)
                if cid0 and _ov_ok(cid0):
                    tmpl.setdefault(_ov_deck_of(cid0), c)
        for cid in hand_cards:
            if cid in have or cid in node.ov_seen:
                continue
            node.ov_seen.add(cid)
            t = tmpl.get(_ov_deck_of(cid))
            if t is None:
                OVERLAY_STATS['lazy_no_template'] += 1
                continue
            try:
                ok = _event_card_has_effect(node.state, node.side, cid)
            except Exception:
                ok = False
            if not ok:
                OVERLAY_STATS['lazy_rejected'] += 1
                continue
            node.children.append(MCTSNode(
                state=None, side=node.side, parent=node,
                action=Action(die_idx=t.action.die_idx,
                              action_type='play_event',
                              params={'card_id': cid,
                                      'side': (t.action.params or {})
                                      .get('side', ov.opp_tag)}),
                prior=_OV_LAZY_PRIOR))
            OVERLAY_STATS['lazy_event'] += 1
    return kind


# ------------------------------------------------------------------
# Combat exchange kernel nodes (design 3B): combat-round chance nodes
# get EXACT branch sets from combat_dist.exact_distribution; branch
# states are engine-built via witness-plan forced applies.
# ------------------------------------------------------------------

_KERNEL_PARENTS = frozenset(('combat_card_play', 'combat_card_none',
                             'combat_effect_choice'))
# kernel-exact nodes carry exact masses, so they afford a wider branch
# cap than sampled nodes (cap-tuning: p90 dropped mass 23% at 12,
# <=3.5% at 24); sampled 1.229 nodes keep _CHANCE_K_MAX
_KERNEL_K_MAX = 24
# cover-targeted extension: pre-combat-card sieges are TRUE state
# entropy (measured: DB/Charge siege = 384 state-distinct outcomes,
# top-24 = 33.8% mass) — extend the kept set toward the target, hard
# ceiling bounds the forced-apply cost
_KERNEL_K_HARD = 96
_KERNEL_COVER_TARGET = 0.90
_KERNEL_MEMO = {}
_KERNEL_MEMO_MAX = 4096
KERNEL_STATS = {'combat_chance_nodes': 0, 'kernel_nodes': 0,
                'fallback': {}, 'dropped_mass': [], 'support': [],
                'opp_card_nodes': 0, 'opp_card_fallback': 0,
                'hunt_class_applies': 0, 'hunt_nodes': 0,
                'hunt_det': 0, 'hunt_walks': 0}


def _kstat_highdrop(entry):
    """Bounded top-20-by-dropped-mass capture of kernel nodes whose
    kept branch set leaves > 25% of the outcome mass on the floor —
    full context, so cap tuning argues from real node identities."""
    lst = KERNEL_STATS.setdefault('high_drop', [])
    lst.append(entry)
    if len(lst) > 20:
        lst.sort(key=lambda e: -e.get('dropped', 0.0))
        del lst[20:]


def _kstat_fallback(reason, detail=None):
    KERNEL_STATS['fallback'][reason] = \
        KERNEL_STATS['fallback'].get(reason, 0) + 1
    if detail is not None:
        lst = KERNEL_STATS.setdefault('fallback_detail', [])
        if len(lst) < 20000:
            lst.append((reason, detail))


def kernel_stats_summary() -> dict:
    """Aggregate kernel-node statistics (dropped mass, fallbacks)."""
    def _pcts(vals):
        v = sorted(vals)

        def _p(q):
            return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0
        return {'n': len(v),
                'mean': (sum(v) / len(v)) if v else 0.0,
                'p50': _p(0.50), 'p90': _p(0.90),
                'max': (v[-1] if v else 0.0)}

    dm = _pcts(KERNEL_STATS['dropped_mass'])
    return {'combat_chance_nodes': KERNEL_STATS['combat_chance_nodes'],
            'kernel_nodes': KERNEL_STATS['kernel_nodes'],
            'fallback': dict(KERNEL_STATS['fallback']),
            'dropped_mass_mean': dm['mean'],
            'dropped_mass_p50': dm['p50'],
            'dropped_mass_p90': dm['p90'],
            'dropped_mass_max': dm['max'],
            'dropped_by_class': {
                k: _pcts(v) for k, v in
                (KERNEL_STATS.get('by_class') or {}).items()},
            'dropped_at_cap': {
                k: _pcts(v) for k, v in
                (KERNEL_STATS.get('dropped_at') or {}).items()},
            'support_max': max(KERNEL_STATS['support'], default=0),
            'opp_card_nodes': KERNEL_STATS['opp_card_nodes'],
            'opp_card_fallback': KERNEL_STATS['opp_card_fallback'],
            'hunt_class_applies': KERNEL_STATS.get(
                'hunt_class_applies', 0),
            'hunt_nodes': KERNEL_STATS.get('hunt_nodes', 0),
            'hunt_det': KERNEL_STATS.get('hunt_det', 0),
            'hunt_walks': KERNEL_STATS.get('hunt_walks', 0),
            'fallback_detail': list(
                KERNEL_STATS.get('fallback_detail', ()))[:400],
            'high_drop': sorted(
                KERNEL_STATS.get('high_drop', ()),
                key=lambda e: -e.get('dropped', 0.0))}


def _kernel_army_spec(state: GameState, region, side: Side):
    """ArmySpec of `side`'s portion at `region` (Isengard elites split
    out — their nation is math-bearing under Saruman's leadership
    rule; everything else folds to one nation per side)."""
    from combat_dist import ArmySpec, LeaderSpec
    from combat import _combat_leader_side
    army = _armies_lookup(state, region)
    r = e_o = e_i = 0
    specs = []
    if army is not None:
        for u in army.units:
            if _side_of_nation(u.nation) != side:
                continue
            if u.kind == UnitKind.REGULAR:
                r += 1
            elif u.nation == Nation.ISENGARD:
                e_i += 1
            else:
                e_o += 1
        for l in army.leaders:
            # 1.233: stranded enemy leaders contribute nothing —
            # side-filter matches the fixed engine exactly
            if _combat_leader_side(l) != side:
                continue
            if isinstance(l, gs.Nazgul):
                specs.append(LeaderSpec('nazgul'))
            elif isinstance(l, gs.Companion):
                specs.append(LeaderSpec(
                    'companion', str(l.name),
                    int(getattr(l, 'level', 1) or 1),
                    int(getattr(l, 'leadership', 1) or 1)))
            elif type(l).__name__ == 'Minion':
                specs.append(LeaderSpec(
                    'minion', str(getattr(l, 'name', '')),
                    nation=Nation.SAURON))
            else:
                specs.append(LeaderSpec('leader'))
    base = Nation.SAURON if side == Side.SHADOW else Nation.GONDOR
    return ArmySpec(regulars=r, elites=e_o, nation=base,
                    elite_nation=base, leaders=tuple(specs),
                    elites2=e_i, elite2_nation=Nation.ISENGARD)


def _kernel_ctx(state: GameState, info: dict, atk_card_id, def_card_id,
                defer: bool):
    """ExchangeContext for a pending exchange on `state`: armies read
    live (side-filtered), settlement KIND carried by fixture regions
    (the round math depends on kind, never the name)."""
    from combat_dist import ExchangeContext
    from cards import get_card
    atk_side = info['atk_side']
    def_side = info['def_side']
    atk = _kernel_army_spec(state, info['atk_region'], atk_side)
    dfn = _kernel_army_spec(state, info['def_region'], def_side)
    if atk.regulars + atk.elites + atk.elites2 == 0 \
            or dfn.regulars + dfn.elites + dfn.elites2 == 0:
        raise ValueError('empty battle side')
    meta = board.regions.get(sanitize(str(info['def_region'])), {})
    is_siege = bool(info.get('is_siege'))
    city = bool((meta.get('is_city') or meta.get('is_fort'))
                and not is_siege)

    def _cname(cid):
        if not cid:
            return None
        c = get_card(cid)
        nm = getattr(c, 'combat_name', None) if c else None
        if not (nm or '').strip():
            raise ValueError('no combat side for %r' % (cid,))
        return nm

    ch = {}
    src = dict(info.get('choices') or {})
    for k in ('atk_casualty_policy', 'def_casualty_policy',
              'atk_effect_count', 'def_effect_count'):
        v = src.get(k)
        if v is not None:
            ch[k] = v
    for k in ('attacker_dice_reduction', 'defender_dice_reduction'):
        v = int(src.get(k, 0) or 0)
        if v:
            ch[k] = v
    if src.get('disable_city_defense'):
        ch['disable_city_defense'] = True
    sar = any(type(l).__name__ == 'Minion'
              and 'saruman' in str(getattr(l, 'name', '')).lower()
              for a in state.armies.values() for l in a.leaders)
    return ExchangeContext(
        attacker=atk, defender=dfn,
        attacker_card=_cname(atk_card_id),
        defender_card=_cname(def_card_id),
        is_siege=is_siege, city=city,
        shine=info.get('gtw_shine'),
        choices=tuple(sorted(ch.items())),
        defer_casualties=bool(defer),
        saruman_in_play=bool(sar))


def _kernel_dist(ctx, want_witness: bool = False):
    """Memoized exact_distribution; RNG state shielded (the kernel's
    lazy scratch fixture must never perturb worker streams)."""
    import combat_dist as cd
    ent = _KERNEL_MEMO.get(ctx)
    if ent is not None and (not want_witness
                            or ent.witness is not None):
        return ent
    _rs = random.getstate()
    try:
        d = cd.exact_distribution(ctx, want_witness=want_witness)
    finally:
        random.setstate(_rs)
    if len(_KERNEL_MEMO) >= _KERNEL_MEMO_MAX:
        _KERNEL_MEMO.clear()
    _KERNEL_MEMO[ctx] = d
    return d


def _kernel_expand_combat(node: 'MCTSNode', probe_state) -> bool:
    """Try to install EXACT kernel branches on a combat-round chance
    node: top-N outcomes by exact mass (N = _CHANCE_K_MAX, or the full
    support), branch.prior = exact mass, branch states built by
    forcing each outcome's sufficient statistics through the REAL
    apply.  False = caller keeps the 1.229 sampled path (counted)."""
    import combat_dist as cd
    if OVERLAY_ASSERT:
        _overlay_assert_state(node.parent.state)
    KERNEL_STATS['combat_chance_nodes'] += 1
    info = getattr(probe_state, '_last_round_call', None)
    if not isinstance(info, dict):
        _pcd = getattr(node.parent.state, 'pending_combat', None) or {}
        _ecd = _pcd.get('effect_choice') or {}
        if (node.action.action_type == 'combat_effect_choice'
                and _ecd.get('stage') == 'post'):
            # deferred-Onslaught magnitude resume: the only RNG is
            # apply_onslaught_choice's k-dice 4+ batch — exact
            # binomial branches, engine-built states
            return _kernel_expand_onslaught_post(node)
        _kstat_fallback('no_round_call', {
            'a': node.action.action_type,
            'params': dict(node.action.params or {}),
            'pc_phase': _pcd.get('phase'),
            'ec_stage': _ecd.get('stage'),
            'ec_effect': _ecd.get('effect')})
        return False
    _cur_obs = [None]
    try:
        ctx = _kernel_ctx(node.parent.state, info, info.get('atk_card'),
                          info.get('def_card'), defer=True)
        dist = _kernel_dist(ctx, want_witness=True)
        # OUTCOME-CLASS COARSENING (user-approved): merge outcomes by
        # the state-faithful projection obs[8:19] (rosters, destroyed,
        # fx, deferred stash) BEFORE the top-N cut — the diagnostic
        # hit-counter fields 0-7 never reach the child GameState, so
        # class members are search-state identical; the max-mass
        # member witnesses the class (any member's plan is valid).
        classes = {}
        for _ob9, _m9 in dist.outcomes.items():
            _ck9 = _ob9[8:19]
            _e9 = classes.get(_ck9)
            if _e9 is None:
                classes[_ck9] = [_m9, _ob9, _m9]
            else:
                _e9[0] += _m9
                if _m9 > _e9[2]:
                    _e9[1] = _ob9
                    _e9[2] = _m9
        outs = sorted(((e[1], e[0]) for e in classes.values()),
                      key=lambda kv: -kv[1])
        total = sum(v for _k, v in outs)
        if not outs or abs(total - 1.0) > 1e-6:
            _kstat_fallback('mass_error')
            return False
        k_eff = min(_KERNEL_K_MAX, len(outs))
        cover = sum(v for _k, v in outs[:k_eff])
        _k_hard = _KERNEL_K_HARD if CHANCE_K_MAX else len(outs)
        while k_eff < min(len(outs), _k_hard) \
                and cover < _KERNEL_COVER_TARGET:
            cover += outs[k_eff][1]
            k_eff += 1
        kept = outs[:k_eff]
        if 1.0 - cover > 0.25:
            _kstat_highdrop({
                'class': node.action.action_type,
                'dropped': round(1.0 - cover, 6),
                'support': len(outs),
                'support_raw': len(dist.outcomes),
                'kept': len(kept),
                'ctx': repr(ctx)[:900]})
        conf = cd.plan_conf_case(ctx)
        # We Come to Kill: the ONLY Phase-9 batch that can ride the
        # SAME apply (auto-allocation case).  Cancellation-exact
        # activity via the engine's own stage0.
        wctk_side = None
        for _nm9, _who9 in ((ctx.attacker_card, 'attacker'),
                            (ctx.defender_card, 'defender')):
            if _nm9 and 'we come to kill' in _nm9.lower():
                wctk_side = _who9
        if wctk_side is not None:
            _p0, _c0 = cd.stage0(ctx)
            _sp0 = _p0.atk if wctk_side == 'attacker' else _p0.dfn
            if 'we come to kill' not in _sp0.name:
                wctk_side = None
        branches = []
        for obs, mass in kept:
            _cur_obs[0] = obs
            plan = dist.witness.get(obs)
            if plan is None:
                _kstat_fallback('no_witness')
                return False
            if wctk_side is not None and obs[16] \
                    and not _predict_alloc_pause(node.parent.state,
                                                 info, obs):
                # forced allocation resolves in-apply -> the WCtK
                # elite attack rolls in the SAME apply: extend the
                # plan and split the branch by its exact binomial
                _ra2, _ea2 = cd._apply_hits(
                    obs[8], obs[9], int(obs[18] or 0),
                    'preserve_elites')
                _rd2, _ed2 = cd._apply_hits(
                    obs[10], obs[11], int(obs[17] or 0),
                    'preserve_elites')
                _own_post = ((_ra2, _ea2) if wctk_side == 'attacker'
                             else (_rd2, _ed2))
                _en_post = ((_rd2, _ed2) if wctk_side == 'attacker'
                            else (_ra2, _ea2))
                _ec9 = min(5, _own_post[1])
                if (_ra2 + _ea2 > 0 and _rd2 + _ed2 > 0
                        and _ec9 > 0):
                    if wctk_side == 'attacker':
                        _ro, _so = info['atk_region'], info['atk_side']
                        _re9, _se = info['def_region'], info['def_side']
                    else:
                        _ro, _so = info['def_region'], info['def_side']
                        _re9, _se = info['atk_region'], info['atk_side']
                    for _h9, _ph9 in cd._binom_terms(
                            _ec9, cd._hit_faces(5, 0) / 6.0):
                        st_b = cd.run_with_plan(
                            plan + (_h9,), conf,
                            lambda: apply_action(node.parent.state,
                                                 node.action))
                        _en_pred = cd._apply_hits(
                            _en_post[0], _en_post[1], _h9,
                            'preserve_elites')

                        def _re_cnt(stx, reg, sid):
                            _us = _alloc_side_units(stx, reg, sid)
                            return (sum(1 for u in _us
                                        if u.kind == UnitKind.REGULAR),
                                    sum(1 for u in _us
                                        if u.kind == UnitKind.ELITE))
                        if _re_cnt(st_b, _re9, _se) != _en_pred \
                                or _re_cnt(st_b, _ro, _so) != _own_post:
                            _kstat_fallback('branch_verify', {
                                'a': node.action.action_type,
                                'cards': (ctx.attacker_card,
                                          ctx.defender_card),
                                'obs': obs[8:19], 'wctk_h': _h9})
                            return False
                        branches.append((st_b, mass * _ph9))
                    continue
            st_b = cd.run_with_plan(
                plan, conf,
                lambda: apply_action(node.parent.state, node.action))
            pcb = getattr(st_b, 'pending_combat', None)
            if isinstance(pcb, dict) \
                    and pcb.get('phase') == 'casualty_allocation':
                # engine-vs-kernel invariant: the stashed hits of the
                # forced apply must equal the outcome's
                if (int(pcb.get('def_hits', 0) or 0) != obs[17]
                        or int(pcb.get('atk_hits', 0) or 0) != obs[18]):
                    _kstat_fallback('branch_verify', {
                        'a': node.action.action_type,
                        'cards': (ctx.attacker_card,
                                  ctx.defender_card),
                        'obs': obs[8:19]})
                    return False
            branches.append((st_b, mass))
    except cd.PlanMismatch as _pm:
        _o9 = _cur_obs[0]

        def _army_dump(reg):
            _a9 = _armies_lookup(node.parent.state, reg)
            if _a9 is None:
                return None
            return ([(str(u.nation.value), u.kind.name)
                     for u in _a9.units],
                    [(type(l).__name__, str(getattr(l, 'name', '')),
                      str(getattr(l, 'nation', '')))
                     for l in _a9.leaders])
        _kstat_fallback('plan_mismatch', {
            'a': node.action.action_type,
            'err': str(_pm)[:200],
            'cards': (ctx.attacker_card, ctx.defender_card),
            'obs': (_o9[8:19] if _o9 else None),
            'ctx': repr(ctx)[:700],
            'plan': repr(dist.witness.get(_o9))[:120] if _o9 else None,
            'atk_army': repr(_army_dump(info['atk_region']))[:400],
            'def_army': repr(_army_dump(info['def_region']))[:400],
            'regions': (str(info['atk_region']),
                        str(info['def_region']))})
        return False
    except Exception as _ex:
        _kstat_fallback('error', {
            'a': node.action.action_type,
            'err': '%s: %s' % (type(_ex).__name__, str(_ex)[:200]),
            'info': {k: str(info.get(k)) for k in
                     ('atk_region', 'def_region', 'atk_side',
                      'def_side', 'atk_card', 'def_card',
                      'is_siege')}})
        return False
    merged = {}
    for st_b, mass in branches:
        sg = _chance_sig(st_b)
        ent = merged.get(sg)
        if ent is not None:
            ent[1] += mass
        else:
            merged[sg] = [st_b, mass]
    node.outcomes = []
    for st_b, mass in sorted(merged.values(), key=lambda v: -v[1]):
        br = MCTSNode(state=st_b, side=node.side, parent=node,
                      action=None, prior=float(mass))
        br.out_sig = _chance_sig(st_b)
        node.outcomes.append(br)
    node.drops = _CHANCE_DRAWS     # bank full: weights are exact
    node.kernel_cover = float(cover)
    KERNEL_STATS['kernel_nodes'] += 1
    if len(KERNEL_STATS['dropped_mass']) < 500000:
        KERNEL_STATS['dropped_mass'].append(max(0.0, 1.0 - cover))
        KERNEL_STATS['support'].append(len(outs))
        _cls = node.action.action_type
        KERNEL_STATS.setdefault('by_class', {}).setdefault(
            _cls, []).append(max(0.0, 1.0 - cover))
        # counterfactual coverage at larger caps (cap-tuning data)
        _da = KERNEL_STATS.setdefault('dropped_at', {})
        _run = 0.0
        for _i, (_o9, _m9) in enumerate(outs):
            _run += _m9
            if _i + 1 in (16, 24, 32):
                _da.setdefault(_i + 1, []).append(
                    max(0.0, 1.0 - _run))
        for _k9 in (16, 24, 32):
            if len(outs) < _k9:
                _da.setdefault(_k9, []).append(0.0)
    return True


def _kernel_expand_onslaught_post(node: 'MCTSNode') -> bool:
    """Exact branches for a deferred-Onslaught magnitude resume: the
    apply rolls k dice at 4+ (apply_onslaught_choice); branch set =
    Binomial(k, hit faces/6) with engine-built states via single-batch
    witness plans."""
    import combat_dist as cd
    if OVERLAY_ASSERT:
        _overlay_assert_state(node.parent.state)
    k = int((node.action.params or {}).get('count', 0) or 0)
    if k <= 0:
        # count 0 rolls nothing; RNG here means an unmodeled consumer
        _kstat_fallback('onslaught_post_k0',
                        {'a': node.action.action_type})
        return False
    try:
        terms = cd._binom_terms(k, cd._hit_faces(4, 0) / 6.0)
        branches = []
        for h, ph in terms:
            st_b = cd.run_with_plan(
                (h,), False,
                lambda: apply_action(node.parent.state, node.action))
            branches.append((st_b, ph))
    except cd.PlanMismatch as _pm:
        _kstat_fallback('plan_mismatch', {
            'a': node.action.action_type, 'err': str(_pm)[:200],
            'cards': ('onslaught_post', k)})
        return False
    except Exception as _ex:
        _kstat_fallback('error', {
            'a': node.action.action_type,
            'err': '%s: %s' % (type(_ex).__name__, str(_ex)[:200])})
        return False
    merged = {}
    for st_b, mass in branches:
        sg = _chance_sig(st_b)
        ent = merged.get(sg)
        if ent is not None:
            ent[1] += mass
        else:
            merged[sg] = [st_b, mass]
    node.outcomes = []
    for st_b, mass in sorted(merged.values(), key=lambda v: -v[1]):
        br = MCTSNode(state=st_b, side=node.side, parent=node,
                      action=None, prior=float(mass))
        br.out_sig = _chance_sig(st_b)
        node.outcomes.append(br)
    node.drops = _CHANCE_DRAWS
    node.kernel_cover = 1.0
    KERNEL_STATS['kernel_nodes'] += 1
    if len(KERNEL_STATS['dropped_mass']) < 500000:
        KERNEL_STATS['dropped_mass'].append(0.0)
        KERNEL_STATS['support'].append(len(merged))
        KERNEL_STATS.setdefault('by_class', {}).setdefault(
            'onslaught_post', []).append(0.0)
    return True


def _predict_alloc_pause(state: GameState, info: dict,
                         obs: tuple) -> bool:
    """Mirror of _alloc_choice_exists over a kernel defer outcome's
    post-round rosters (defender stage then attacker) — predicts
    whether the apply pauses at a casualty_allocation pending.  A
    wrong prediction can only cost a counted plan-mismatch fallback,
    never a wrong branch."""
    from combat import _cas_box_regulars
    for region, side, r, e, hits in (
            (info['def_region'], info['def_side'],
             obs[10], obs[11], obs[17]),
            (info['atk_region'], info['atk_side'],
             obs[8], obs[9], obs[18])):
        hits = int(hits or 0)
        if hits <= 0:
            continue
        if hits >= 2 * e + r:
            continue
        if not e:
            continue
        if r:
            return True
        if hits < 2 or e < 2:
            continue
        if any(_cas_box_regulars(state, u.nation) > 0
               or _can_recruit(state, u.nation, 'regular')
               for u in _alloc_side_units(state, region, side,
                                          UnitKind.ELITE)):
            return True
    return False


def _kernel_chance_proof(n: 'MCTSNode') -> bool:
    """1.232 sound extension: a kernel-exact chance node whose branch
    set covers 100% of the outcome mass MAY prove when every branch is
    proven the SAME class (proof value = mass-weighted expectation).
    Sampled chance nodes still block proofs."""
    cover = getattr(n, 'kernel_cover', None)
    if not n.is_chance or cover is None or cover < 1.0 - 1e-9:
        return False
    outs = n.outcomes or []
    if not outs or any(b.proven is None for b in outs):
        return False
    cls = {b.proven for b in outs}
    if len(cls) != 1:
        return False
    tot = sum(b.prior for b in outs)
    if tot <= 0:
        return False
    n.proven = cls.pop()
    n.proven_value = sum(b.prior * b.proven_value
                         for b in outs) / tot
    return True


# ------------------------------------------------------------------
# Hunt kernel nodes: hunt-bearing applies get EXACT branch sets by
# enumerating the REAL apply under hunt_dist's fork driver (funnels
# over hunt roll / rerolls / tile picks / Shelob d6 / companion
# draws).  The deterministic hunt-pool pop becomes an enumerated
# uniform pick — pool ORDER stops leaking into the tree.  Learned
# pendings (gollum / sacrifice / hobbit place / card options) are the
# natural apply boundaries, so branch states OPEN those decisions and
# expansion below stays normal decision nodes — never collapsed.
# ------------------------------------------------------------------

_HUNT_PARENTS = frozenset((
    'fellowship_move', 'fellowship_reveal_place', 'hobbit_place',
    'hunt_gollum_reveal', 'hunt_gollum_no_reveal',
    'hunt_take_corruption', 'hunt_sacrifice_guide',
    'hunt_sacrifice_random', 'hunt_guide_separate', 'choose_guide',
    'card_option'))


def _hunt_action_gate(node: 'MCTSNode') -> bool:
    """Attempt hunt-kernel expansion for this class-map action?
    card_option only when the pending option is hunt-owned."""
    at = node.action.action_type
    if at != 'card_option':
        return True
    pco = getattr(node.parent.state, 'pending_card_option', None)
    if not isinstance(pco, dict):
        return False
    if pco.get('hunt_reduction'):
        return True
    for o in (pco.get('options') or []):
        if isinstance(o, dict) and ('mithril_redraw' in o
                                    or 'wizardstaff_prevent' in o):
            return True
    return False


def _mk_walk_choice(real_choice):
    """random.choice funnel for driven hunt walks: ANY choice is a
    uniform draw, so routing it through the fork driver's option
    machinery is exact (covers non-Companion consumers the gl proxy
    passes through: WWSAT hand discards, Foul Thing casualties, card
    handlers).  Outside a walk, delegates to the real choice."""
    import hunt_dist as hd

    def _walk_choice(seq):
        drv9 = hd._DRIVER
        if drv9 is None:
            return real_choice(seq)
        n9 = len(seq)
        if n9 <= 0:
            return real_choice(seq)      # engine IndexError parity
        if n9 == 1:
            return seq[0]
        i9 = drv9.choose([(j, 1.0 / n9) for j in range(n9)])
        return seq[i9]
    return _walk_choice


def _hunt_rng_culprit(parent, action, plan, src) -> str:
    """Diagnostic replay of a raw-RNG hunt walk: wrap the random-module
    entry points to record the FIRST unfunneled consumer's callsite,
    then replay the failing plan.  Restores everything; RNG-neutral."""
    import hunt_dist as hd
    import traceback as _tb
    hit = []
    names = ('choice', 'randint', 'random', 'sample', 'uniform',
             'randrange')
    real = {n: getattr(random, n) for n in names}

    def _wrap(name):
        fn = real[name]

        def w(*a, **k):
            if not hit:
                frames = _tb.extract_stack()[-6:-1]
                hit.append('%s <- %s' % (name, ' | '.join(
                    '%s:%d %s' % (os.path.basename(f.filename),
                                  f.lineno, f.name) for f in frames)))
            return fn(*a, **k)
        return w

    rng0 = random.getstate()
    drv = hd.HuntDriver(list(plan), src)
    hd._install(drv)
    for n in names:
        setattr(random, n, _wrap(n))
    # mirror the walk's choice funnel so only genuinely raw
    # consumers (non-choice, or unfunnelable) are reported
    random.choice = _mk_walk_choice(_wrap('choice'))
    try:
        try:
            apply_action(parent, action)
        except BaseException:
            pass
    finally:
        for n in names:
            setattr(random, n, real[n])
        hd._uninstall()
        random.setstate(rng0)
    return hit[0] if hit else 'unknown'


def _hunt_expand_try(node: 'MCTSNode'):
    """Enumerate the REAL apply of a hunt-bearing action under the
    hunt_dist fork driver.  Returns ('chance', None) with kernel
    branches installed, ('det', state) when the apply is genuinely
    stochastic-free, or (None, None) -> caller keeps the 1.229
    sampled path (counted fallback)."""
    import hunt_dist as hd
    KERNEL_STATS['hunt_class_applies'] = \
        KERNEL_STATS.get('hunt_class_applies', 0) + 1
    parent = node.parent.state
    if OVERLAY_ASSERT:
        _overlay_assert_state(parent)
    fp_dice = sum(1 for d in getattr(parent, 'hunt_box', [])
                  if getattr(d, 'side', None) == Side.FREE_PEOPLES)
    src = hd.AnalyticSource(hd.HuntContext(
        fp_dice=fp_dice,
        crebain_bonus=int(getattr(parent, '_crebain_hunt_bonus', 0)
                          or 0)))
    rng0 = random.getstate()
    leaves = []
    stack = [[]]
    walks = 0
    forked = False
    _real_choice = random.choice
    _walk_choice = _mk_walk_choice(_real_choice)
    try:
        while stack:
            plan = stack.pop()
            walks += 1
            if walks > 4096:
                raise RuntimeError('hunt walk budget exceeded')
            drv = hd.HuntDriver(plan, src)
            hd._install(drv)
            random.choice = _walk_choice
            try:
                st_b = apply_action(parent, node.action)
            except hd.HuntFork:
                forked = True
                if random.getstate() != rng0:
                    raise RuntimeError('raw RNG in driven walk')
                for i in range(len(drv.pending)):
                    stack.append(plan + [i])
                continue
            finally:
                random.choice = _real_choice
                hd._uninstall()
            if random.getstate() != rng0:
                raise RuntimeError('raw RNG in driven walk')
            leaves.append((st_b, drv.prob))
    except hd.HuntFork:
        hd._uninstall()
        random.setstate(rng0)
        _kstat_fallback('hunt_fork_leak',
                        {'a': node.action.action_type})
        return None, None
    except Exception as e:
        hd._uninstall()
        random.setstate(rng0)
        if 'raw RNG' in str(e):
            _kstat_fallback('hunt_rng', {
                'a': node.action.action_type,
                'params': dict(node.action.params or {}),
                'culprit': _hunt_rng_culprit(parent, node.action,
                                             plan, src)})
        else:
            _kstat_fallback('hunt_error', {
                'a': node.action.action_type,
                'err': '%s: %s' % (type(e).__name__, str(e)[:200])})
        return None, None
    KERNEL_STATS['hunt_walks'] = \
        KERNEL_STATS.get('hunt_walks', 0) + walks
    if not forked:
        if len(leaves) == 1 and abs(leaves[0][1] - 1.0) < 1e-9:
            KERNEL_STATS['hunt_det'] = \
                KERNEL_STATS.get('hunt_det', 0) + 1
            return 'det', leaves[0][0]
        _kstat_fallback('hunt_shape')
        return None, None
    total = sum(p for _s, p in leaves)
    if not leaves or abs(total - 1.0) > 1e-6:
        _kstat_fallback('hunt_mass')
        return None, None
    merged = {}
    for st_b, mass in leaves:
        sg = _chance_sig(st_b)
        ent = merged.get(sg)
        if ent is not None:
            ent[1] += mass
        else:
            merged[sg] = [st_b, mass]
    ranked = sorted(merged.values(), key=lambda v: -v[1])
    kept = ranked[:_KERNEL_K_MAX]
    cover = sum(m for _s, m in kept)
    if 1.0 - cover > 0.25:
        try:
            import hunt_dist as _hd9
            _pool9 = sorted(_hd9.tile_key(t)
                            for t in getattr(parent, 'hunt_pool', []))
        except Exception:
            _pool9 = None
        _kstat_highdrop({
            'class': 'hunt:' + node.action.action_type,
            'dropped': round(1.0 - cover, 6),
            'support': len(ranked), 'kept': len(kept),
            'desc': {'params': dict(node.action.params or {}),
                     'eyes': len(getattr(parent, 'hunt_box', ())),
                     'pool': _pool9}})
    node.is_chance = True
    node.outcomes = []
    for st_b, mass in kept:
        br = MCTSNode(state=st_b, side=node.side, parent=node,
                      action=None, prior=float(mass))
        br.out_sig = _chance_sig(st_b)
        node.outcomes.append(br)
    node.drops = _CHANCE_DRAWS     # bank full: weights are exact
    node.kernel_cover = float(cover)
    KERNEL_STATS['hunt_nodes'] = KERNEL_STATS.get('hunt_nodes', 0) + 1
    if len(KERNEL_STATS['dropped_mass']) < 500000:
        KERNEL_STATS['dropped_mass'].append(max(0.0, 1.0 - cover))
        KERNEL_STATS['support'].append(len(ranked))
        KERNEL_STATS.setdefault('by_class', {}).setdefault(
            'hunt:' + node.action.action_type, []).append(
                max(0.0, 1.0 - cover))
    return 'chance', None


def _capture_bank_swing(state: GameState, def_region, atk_side: Side,
                        def_side: Side) -> float:
    """Bank delta (1.224 convex VP win-fraction currency) if the
    attacker takes def_region: city 1 VP / stronghold 2 VP, capture on
    an enemy-native settlement raises the capturer's track, re-capture
    strips the dispossessed holder's."""
    try:
        san = sanitize(str(def_region))
        meta = board.regions.get(san, {})
        val = 2 if meta.get('is_stronghold') else (
            1 if meta.get('is_city') else 0)
        if val <= 0:
            return 0.0
        from combat import garrison_side_for
        if garrison_side_for(state, san) != def_side:
            return 0.0
        vp = game_logic.calculate_victory_points(state)
        nat = _region_nation(san)
        native = _side_of_nation(nat) if nat is not None else None

        def _phi(side, v):
            if side == Side.SHADOW:
                return _win_frac_phi(v, game_logic.SHADOW_VP_VICTORY)
            return _win_frac_phi(v, game_logic.FP_VP_VICTORY)

        if native == def_side:
            v0 = vp['shadow' if atk_side == Side.SHADOW
                    else 'free_peoples']
            return _phi(atk_side, v0 + val) - _phi(atk_side, v0)
        if native == atk_side:
            v0 = vp['shadow' if def_side == Side.SHADOW
                    else 'free_peoples']
            return _phi(def_side, v0) - _phi(def_side, v0 - val)
        return 0.0
    except Exception:
        return 0.0


def _opp_exchange_value(state: GameState, info: dict, ctx, dist,
                        opp_is_attacker: bool):
    """Opponent's exchange value for best-response ranking:
    (capture win-fraction swing, hit-capacity differential) — the
    1.143/1.224 bank currency first, engine 2-hit elite capacity as
    tie-break.  Existing constants only; ordering is all the closed
    form consumes."""
    a0, d0 = ctx.attacker, ctx.defender
    cap_a0 = a0.regulars + 2 * (a0.elites + a0.elites2)
    cap_d0 = d0.regulars + 2 * (d0.elites + d0.elites2)
    e_cap_a = e_cap_d = 0.0
    p_destD_only = 0.0
    for obs, pr in dist.outcomes.items():
        e_cap_a += pr * (obs[8] + 2 * obs[9])
        e_cap_d += pr * (obs[10] + 2 * obs[11])
        if obs[13] and not obs[12]:
            p_destD_only += pr
    lost_a = cap_a0 - e_cap_a
    lost_d = cap_d0 - e_cap_d
    swing = 0.0
    if p_destD_only > 0.0:
        swing = p_destD_only * _capture_bank_swing(
            state, info['def_region'], info['atk_side'],
            info['def_side'])
    if opp_is_attacker:
        return (swing, lost_d - lost_a)
    return (-swing, lost_a - lost_d)


def _opponent_card_priors(state: GameState, pc: dict, actions,
                          opp_side: Side):
    """Closed-form opponent combat-card model (design 3B): P(plays X) =
    P(X held AND no better-for-them candidate held) under the
    exchangeable-uniform hand posterior over the unseen pool (deck +
    hand + face-down discards, per deck); 'better' ranks by the kernel
    exchange value under the opponent's functional; combat_card_none
    takes the residual (plus the mass of candidates the sampled hand
    cannot represent).  Returns (priors aligned with `actions`,
    {card_id: P(plays)} over the FULL pool — the 3A lazy-add mass
    source), or None (counted fallback -> BC priors).
    OVERLAY-INVARIANT: the pool below is the per-deck union
    deck + hand + face-down, which 3A patching permutes but never
    changes as a multiset — this posterior is identical on true and
    patched states (no double-conditioning with the overlay sample).
    D3-BLINDED (USER 2026-08-16): the defender ranking never reads the
    attacker's committed card IDENTITY - only its public TYPE, via a
    copy-weighted mixture over the attacker's playable cards of that
    type (see the atk_mix block below)."""
    import combat_dist as cd
    from cards import get_card
    KERNEL_STATS['opp_card_nodes'] += 1
    try:
        is_attacker = pc.get('phase') == 'attacker_combat_card'
        tag = 'fp' if opp_side == Side.FREE_PEOPLES else 'sh'
        # '_unknown_*' tracked-hand placeholders (live-play/teacher
        # states) carry no deck identity: drop them from BOTH the pool
        # and the hand count (classing them as 'strat' skewed the
        # per-deck hand posterior).  _ov_ok/_ov_deck_of are the one
        # implementation.
        hand_pi = state.hands.get(opp_side)
        hand = [c for c in (hand_pi.hand if hand_pi else [])
                if _ov_ok(c)]
        fdown = [c for c in (hand_pi.discarded_face_down
                             if hand_pi else []) if _ov_ok(c)]
        deck_c = [c for c in (getattr(state, tag + '_char_deck', None)
                              or []) if _ov_ok(c)]
        deck_s = [c for c in (getattr(state, tag + '_strat_deck', None)
                              or []) if _ov_ok(c)]

        _deck_of = _ov_deck_of

        pool = {'char': [], 'strat': []}
        for c in deck_c + deck_s + hand + fdown:
            pool[_deck_of(c)].append(c)
        pool_sizes = {d: len(v) for d, v in pool.items()}
        n_hand = {'char': sum(1 for c in hand
                              if _deck_of(c) == 'char'),
                  'strat': sum(1 for c in hand
                               if _deck_of(c) == 'strat')}
        tmp, fp_ok = _combat_card_check_ctx(state, pc, is_attacker)
        cand_copies = {}
        rejected = set()
        if fp_ok:
            for d in ('char', 'strat'):
                for cid in pool[d]:
                    key = (cid, d)
                    if key in cand_copies:
                        cand_copies[key] += 1
                        continue
                    if cid in rejected:
                        continue
                    if _combat_card_playable(state, get_card(cid), tmp,
                                             is_attacker):
                        cand_copies[key] = 1
                    else:
                        rejected.add(cid)
            # a table combat card (Balrog) is a CERTAIN holding: model
            # it as its own single-card pool the hand always contains
            for a in actions:
                if a.action_type == 'combat_card_play' \
                        and a.params.get('from_table'):
                    _tcid = a.params.get('card_id')
                    if _tcid:
                        cand_copies[(_tcid, 'table')] = 1
                        pool_sizes['table'] = 1
                        n_hand['table'] = 1
        info = {'atk_region': pc['attacker_region'],
                'def_region': pc['defender_region'],
                'atk_side': pc['attacker_side'],
                'def_side': pc['defender_side'],
                'is_siege': pc.get('is_siege', False),
                'gtw_shine': pc.get('gtw_shine'),
                'choices': {
                    'atk_casualty_policy':
                        pc.get('atk_casualty_policy'),
                    'def_casualty_policy':
                        pc.get('def_casualty_policy'),
                    'attacker_dice_reduction':
                        pc.get('atk_dice_reduction', 0),
                    'defender_dice_reduction':
                        pc.get('def_dice_reduction', 0),
                    'disable_city_defense':
                        pc.get('rounds_fought', 0) > 0}}
        # D3 BLINDING (rules-audit discrepancy 3, USER-ruled
        # 2026-08-16): combat cards are committed SECRETLY - at the
        # defender's decision the attacker's card lies face-down, so
        # only its TYPE (char/strat card back) and the fact of
        # commitment are public; the IDENTITY is not.  The old code
        # ranked defender candidates in a context built from
        # pc['atk_combat_card'] - the exact committed card - an
        # information leak into the opponent model.  USER: "it can see
        # the type of card (character/ strategy) just not the
        # identity."  So the defender ranking now marginalizes over
        # the attacker's PLAYABLE cards of the COMMITTED TYPE from the
        # attacker's unseen pool (deck + hand + face-down,
        # copy-weighted, the same _ov machinery as the opponent pool
        # above).  Type visible, identity dissolved.  The attacker
        # model is unchanged (the defender has not committed at
        # attacker-decision time - nothing to blind).
        atk_mix = [(None, 1)]
        if not is_attacker and pc.get('atk_combat_card'):
            # _atk_commit_mixture is the ONE answerer (shared with the
            # root/search blinding in _SimOverlay, USER 2026-08-16):
            # playable same-type candidates from the attacker's
            # PRE-COMMIT unseen pool, copy-weighted, sorted.
            try:
                _amix9 = _atk_commit_mixture(state, pc)
                if _amix9:
                    atk_mix = _amix9
                # else: no representable candidate of the committed
                # type - identity-free None context
            except Exception:
                atk_mix = [(None, 1)]

        def _v(cand_cid):
            if is_attacker:
                vctx = _kernel_ctx(state, info, cand_cid, None,
                                   defer=False)
                return _opp_exchange_value(state, info, vctx,
                                           _kernel_dist(vctx),
                                           opp_is_attacker=True)
            # defender model: EXPECTED exchange value over the
            # type-only attacker mixture (component-wise mean of the
            # ranking tuple - ordering is all the posterior consumes)
            _tot9 = float(sum(w for _c9, w in atk_mix)) or 1.0
            _s09 = _s19 = 0.0
            for _acid9, _w9 in atk_mix:
                vctx = _kernel_ctx(state, info, _acid9, cand_cid,
                                   defer=False)
                _ev9 = _opp_exchange_value(state, info, vctx,
                                           _kernel_dist(vctx),
                                           opp_is_attacker=False)
                _s09 += _w9 * _ev9[0]
                _s19 += _w9 * _ev9[1]
            return (_s09 / _tot9, _s19 / _tot9)

        v_none = _v(None)
        cands = {}
        for (cid, d), copies in cand_copies.items():
            try:
                cands[cid] = (d, copies, _v(cid))
            except Exception:
                cands[cid] = (d, copies, v_none)   # unrankable: as none
        p_play, _p_none = cd.card_play_posterior(
            pool_sizes, n_hand, cands, v_none)
        priors = []
        seen = 0.0
        for a in actions:
            if a.action_type == 'combat_card_play':
                pm = float(p_play.get(a.params.get('card_id'), 0.0))
                priors.append(pm)
                seen += pm
            else:
                priors.append(-1.0)
        # decline arm = residual, incl. unrepresentable candidates
        rest = max(0.0, 1.0 - seen)
        priors = [rest if x < 0 else x for x in priors]
        floor = 1e-3               # PUCT never explores a 0-prior arm
        priors = [max(floor, x) for x in priors]
        s = sum(priors)
        return [x / s for x in priors], \
            {k: float(v) for k, v in p_play.items()}
    except Exception:
        KERNEL_STATS['opp_card_fallback'] += 1
        return None


# ------------------------------------------------------------------
# Root-action selection: visit-argmax with evidence-gated overrides.
# ONE implementation for every consumer — the search's final pick and
# the trainer's policy-delta targets (run_td_v33loop).
# ------------------------------------------------------------------

@dataclass
class SearchSelectConfig:
    """Root-selection knobs; override via WOTR_SELECT_CFG='k=v,k=v'."""
    min_visits: int = 8         # visits an arm needs to be considered
    term_frac: float = 0.95     # arm terminal-resolved at/above this
    term_margin: float = 0.03   # Q edge required for an override
    c_visit: float = 50.0       # pi sharpness: sigma=(c_visit+max_n)*c_scale
    c_scale: float = 1.0
    completed_q_min_n: int = 3  # below this, an arm's Q = pooled mean
    # pi_target mode (LAB 2026-08-09, proposal P-A):
    #   'sigma'    — run-7 baseline: p*exp(sigma*q_hat) over eligible
    #                arms.  PROVEN KILLER: sigma = 50+max_n (~238-2386
    #                at 1000 sims) one-hots 71.9% of targets on Q
    #                noise / free-action bias; one delta update crushed
    #                fellowship_move prior 0.149 -> 0.0006 (250x).
    #   'visit_tf' — P3: visit distribution over alive arms
    #                (AlphaZero-style; target-side amplification of
    #                any arm == 1.0x its visit share, unit-tested) +
    #                w_tf-weighted one-hot blend ONLY when the
    #                1.230 terminal-grounded override fired (play !=
    #                visit argmax).  Proof-forced pv rows unchanged.
    #   Play pick is IDENTICAL under both modes (targets change,
    #   play semantics never do).
    pi_mode: str = 'sigma'
    w_tf: float = 0.5           # override-arm blend weight (visit_tf)
    # task-15 LAW 3 (quantified 1.230): 0.0 = binary tf rule only
    # (unchanged).  > 0 adds the graded path: an arm may override on
    # q_gap >= term_margin + conf_margin * (1 - support_arm), where
    # support = tf + (1 - tf) * mean leaf confidence ('cf') of the
    # CHALLENGER arm (its Q is the estimate being trusted; the
    # exact-default case stays with the binary tf path).  conf_margin
    # is the extra Q-gap demanded of a fully UNSUPPORTED arm.
    conf_margin: float = 0.0
    # task-17 LAW 1R (root-side confidence override; the play-path
    # replacement for retired leaf shrinkage).  Fires ONLY when ALL
    # of: (i) the current pick's (post-tf-override) support s <= s_low and
    # it is not terminal-grounded; (ii) an eligible alternative with
    # s >= s_high exists; (iii) the SHADOW ordering — q_shadow =
    # s*(q_hat - phi_mean) + phi_mean, per-arm — favors the
    # alternative by >= conf_play_margin.  Knife-edge states (uniform
    # support) fail (i)/(ii) and pass through BIT-untouched; the
    # rescue case (unsupported default vs supported alternative)
    # flips on explicit, auditable margins.  Default off.
    conf_play: bool = False
    s_low: float = 0.30
    s_high: float = 0.60
    conf_play_margin: float = 0.10
    # task-20 (v37 smoke-refusal supply fix): the earned-visit
    # discount (FIX 3) protected the TRAINING target from forced-
    # floor visits but the play-time TEMPERATURE SAMPLER drew on raw
    # n — measured on the refused smoke: 20/24 ring plays were
    # temp-sampled from ~4% visit shares that were PURE floor
    # (ring-arm visits p50 exactly 36 = 12x3, earned p50 = 0).
    # temp_earned=True subtracts forced visits from the sampling
    # weights too (mirror of FIX 3; argmax play and pi targets
    # unaffected; all-forced fallback keeps raw n).  Default off.
    temp_earned: bool = False


def _select_cfg_from_env() -> 'SearchSelectConfig':
    """Build SearchSelectConfig with WOTR_SELECT_CFG overrides applied
    ('key=val,key=val'; unknown keys are fatal — a typo must not
    silently run a different experiment)."""
    cfg = SearchSelectConfig()
    for part in os.environ.get('WOTR_SELECT_CFG', '').split(','):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition('=')
        key = key.strip()
        if not hasattr(cfg, key):
            raise ValueError('WOTR_SELECT_CFG: unknown key %r' % key)
        cast = type(getattr(cfg, key))
        if cast is str:
            setattr(cfg, key, val.strip())
        else:
            setattr(cfg, key, cast(float(val)))
    return cfg


SELECT_CFG = _select_cfg_from_env()


def select_root_action(children_stats, temperature, cfg=None):
    """Pick the root action from per-arm search stats
    ({'p' clean priors, 'q' parent-perspective child Q, 'n' raw visit
    counts, optional 'tf' resolved-terminal fraction}) and return
    (index, pi_target float64[K], diagnostics dict).

    1. DEFAULT: the most-visited arm wins.
    2. OVERRIDE (terminal-grounded ONLY): an arm with
       n >= cfg.min_visits and a completed-Q lead >= cfg.term_margin
       replaces the default iff the ARM or the DEFAULT is
       terminal-resolved (tf >= cfg.term_frac — its value is exact).
       Head-valued Q gaps never override: net error is systematic and
       does not shrink with visits, so no gap between two head
       opinions is ever significant (an 0.15 "lead" from a tf=0 arm
       sat won games out at the Crack of Doom — mordor_push fixture).
    3. TEMPERATURE: pi_target weights arms with n >= cfg.min_visits by
       prior * exp(sigma * completed-Q), sigma=(c_visit+max_n)*c_scale.
       temp 0 plays the default/override pick; temp > 0 samples
       pi_target^(1/T).

    pi_target is the delta-head training target; ineligible arms hold
    exactly 0 (the trainer masks its CE to the eligible set).
    Diagnostics: q_hat float64[K], sigma float, elig bool[K], regret
    float (best eligible completed-Q minus the pick's)."""
    if cfg is None:
        cfg = SELECT_CFG
    p = np.asarray(children_stats['p'], dtype=np.float64)
    q = np.asarray(children_stats['q'], dtype=np.float64)
    n = np.asarray(children_stats['n'], dtype=np.float64)
    tf = children_stats.get('tf')
    pv = children_stats.get('pv')

    # completed-Q: arms too thin to trust take the pooled mean Q
    tot = float(n.sum())
    mean_q = float((q * n).sum() / tot) if tot > 0 else 0.0
    q_hat = np.where(n >= cfg.completed_q_min_n, q, mean_q)

    # proof dominance: a proven WIN beats everything; a proven LOSS
    # has zero upside so NO estimate ever justifies playing it while
    # any unproven arm exists; all-lost -> best graded value (die
    # closest to victory, never resign)
    if pv is not None and len(n):
        pv_arr = np.asarray(pv, dtype=np.int8)
        if (pv_arr == 1).any():
            play = int(np.argmax(np.where(pv_arr == 1, q_hat, -np.inf)))
            pi = (pv_arr == 1).astype(np.float64)
            pi /= pi.sum()
            return play, pi, {'q_hat': q_hat, 'sigma': 0.0,
                              'elig': pv_arr == 1, 'regret': 0.0}
        if (pv_arr == -1).all():
            # 1.225 graded closeness: pick the EXACT proven graded
            # value (die closest to victory), not q_hat — pre-proof
            # net backups freeze into q and can invert the ordering.
            _pvv = children_stats.get('pvv')
            if _pvv is not None and len(_pvv) == len(n):
                play = int(np.argmax(np.asarray(_pvv,
                                                dtype=np.float64)))
            else:
                play = int(np.argmax(q_hat))
            pi = np.zeros(len(n))
            pi[play] = 1.0
            return play, pi, {'q_hat': q_hat, 'sigma': 0.0,
                              'elig': pv_arr == -1, 'regret': 0.0}
        alive = pv_arr != -1
    else:
        alive = np.ones(len(n), dtype=bool)

    max_n = float(n.max()) if len(n) else 0.0
    elig = (n >= float(cfg.min_visits)) & alive
    n_play = np.where(alive, n, -1.0)
    play = default = int(np.argmax(n_play)) if len(n) else 0
    if len(n) and tf is not None:
        tf_arr = np.asarray(tf, dtype=np.float64)
        cand = elig.copy()
        cand[default] = False
        ok = (cand
              & ((q_hat - q_hat[default]) >= cfg.term_margin)
              & ((tf_arr >= cfg.term_frac)
                 | (tf_arr[default] >= cfg.term_frac)))
        cf = children_stats.get('cf')
        if cfg.conf_margin > 0.0 and cf is not None:
            # task-15 LAW 3: support-graded override margin, gated on
            # the CHALLENGER's support only — the challenger's Q is
            # the estimate being trusted; a well-supported DEFAULT
            # never licenses a noisy challenger (the binary tf path
            # above already covers the exact-default case).
            cf_arr = np.asarray(cf, dtype=np.float64)
            sup = tf_arr + (1.0 - tf_arr) * np.clip(cf_arr, 0.0,
                                                    1.0)
            ok = ok | (cand & ((q_hat - q_hat[default])
                               >= cfg.term_margin
                               + cfg.conf_margin * (1.0 - sup)))
        if ok.any():
            play = int(np.argmax(np.where(ok, q_hat, -np.inf)))
    tf_founded = (play != default)   # only tf/LAW-3 overrides so far
    # ultrareview bug_005: remember WHICH arm the tf/LAW-3 override
    # picked — if the LAW 1R block below moves play again, the
    # sharpening must still target the tf-picked arm (its value is
    # exact), never the confidence rescue
    tf_pick = play if tf_founded else None
    # task-17 LAW 1R: root-side confidence override (see cfg)
    if cfg.conf_play and len(n):
        cf17 = children_stats.get('cf')
        if cf17 is not None:
            tf17 = (np.asarray(tf, dtype=np.float64)
                    if tf is not None else np.zeros(len(n)))
            cf17 = np.clip(np.asarray(cf17, dtype=np.float64),
                           0.0, 1.0)
            sup17 = tf17 + (1.0 - tf17) * cf17
            ph17 = np.asarray(
                children_stats.get('ph', np.zeros(len(n))),
                dtype=np.float64)
            if sup17[play] <= cfg.s_low \
                    and tf17[play] < cfg.term_frac:
                qs17 = sup17 * (q_hat - ph17) + ph17
                cand17 = elig & alive & (sup17 >= cfg.s_high)
                cand17[play] = False
                if cand17.any():
                    j17 = int(np.argmax(np.where(cand17, qs17,
                                                 -np.inf)))
                    if qs17[j17] - qs17[play] \
                            >= cfg.conf_play_margin:
                        play = j17
                        elig[play] = True
    if not elig.any():
        # degenerate tiny search: trust only the most-visited arm(s)
        elig = (n >= max_n) & alive
        if not elig.any():
            elig = n >= max_n
    elig[play] = True

    # pi_target: mode-dependent (see SearchSelectConfig.pi_mode)
    sigma = (cfg.c_visit + max_n) * cfg.c_scale
    if cfg.pi_mode == 'visit_tf':
        # P3: visit distribution over ALIVE arms (proven-loss arms
        # hold exactly 0 however many exploration visits they ate);
        # bounded by construction — no arm can receive more target
        # mass than the search actually allocated, except the
        # terminal-grounded override arm below.
        w = np.where(alive, np.maximum(n, 0.0), 0.0)
        # FIX 3 (task 14): forced-playout visits ('fn') are NOT
        # earned — subtract them so the floor is never distilled
        # into the prior (played arm keeps raw n: playing it IS the
        # strongest endorsement)
        fn = children_stats.get('fn')
        if fn is not None:
            w_e = np.maximum(
                w - np.asarray(fn, dtype=np.float64), 0.0)
            w_e[play] = w[play]
            if float(w_e.sum()) > 0.0:
                w = w_e
        s = float(w.sum())
        if s > 0.0 and np.isfinite(s):
            pi = w / s
        else:
            pi = elig.astype(np.float64)
            pi /= max(1.0, float(pi.sum()))
        if tf_founded:
            # the 1.230 terminal-grounded override fired: sharpen
            # toward the override arm, bounded by w_tf (its value is
            # exact — the ONE sanctioned Q-gap sharpening).  task-17:
            # LAW 1R confidence overrides deliberately do NOT sharpen
            # the training target (play-side rescue only; the prior
            # learns from earned visits, never from a rescue) — so
            # sharpen toward tf_pick, which survives a LAW 1R
            # play reassignment (ultrareview bug_005).
            one = np.zeros_like(pi)
            one[tf_pick] = 1.0
            pi = cfg.w_tf * one + (1.0 - cfg.w_tf) * pi
    else:
        # run-7 baseline ('sigma'): over the eligible set only;
        # ineligible arms get exactly 0
        logits = np.log(np.clip(p, 1e-12, None)) + sigma * q_hat
        logits = np.where(elig, logits, -np.inf)
        logits -= logits[elig].max() if elig.any() else 0.0
        with np.errstate(over='ignore', invalid='ignore'):
            pi = np.exp(logits)
        pi[~elig] = 0.0
        s = float(pi.sum())
        if not np.isfinite(s) or s <= 0.0:
            pi = elig.astype(np.float64)
            pi /= max(1.0, float(pi.sum()))
        else:
            pi = pi / s

    # play policy: visit counts (PUCT's own prior+value blend); pi is
    # the TRAINING target only — sampling it plays raw head opinions
    if temperature < 0.01 or play != default:
        idx = play
    else:
        n_samp = np.where(alive, n, 0.0)
        if cfg.temp_earned:
            fn20 = children_stats.get('fn')
            if fn20 is not None:
                n_e20 = np.maximum(
                    n_samp - np.asarray(fn20, dtype=np.float64),
                    0.0)
                if float(n_e20.sum()) > 0.0:
                    n_samp = n_e20   # task-20: sample EARNED visits
        w = np.power(np.maximum(n_samp, 0.0),
                     1.0 / temperature)
        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= 0.0:
            idx = play
        else:
            idx = int(np.random.choice(len(n), p=w / wsum))
    diag = {'q_hat': q_hat, 'sigma': float(sigma), 'elig': elig,
            'regret': float(np.max(q_hat[elig]) - q_hat[idx])}
    return idx, pi, diag


def _search_terminal(result: dict, root_side: 'Side',
                     state: 'GameState' = None) -> float:
    """Search-terminal value in the ROOT side's currency: the ROOT
    profile's _terminal_for PLUS the same shaping term net leaves get
    (_mcts_eval_state adds sgn*phi to every leaf; a terminal without it
    is paid in a richer currency — graded losses outranked surviving,
    erebor_recapture fixture). Search-only; training labels unchanged."""
    fp_won = (result.get('victory') == 'free_peoples')
    won_side = Side.FREE_PEOPLES if fp_won else Side.SHADOW
    tag = 'fp' if root_side == Side.FREE_PEOPLES else 'sh'
    v = float(_terminal_for(tag, PLAYER_PROFILES.get(tag),
                            result.get('victory_type'),
                            root_side == won_side, state))
    if state is not None:
        _sgn = 1.0 if root_side == Side.FREE_PEOPLES else -1.0
        v += _sgn * _profile_shaping(state)
    return v


def _proof_mark(result: dict, root_side: 'Side') -> str:
    """'W'/'L' from the root side's view for a resolved terminal."""
    fp_won = (result.get('victory') == 'free_peoples')
    won = Side.FREE_PEOPLES if fp_won else Side.SHADOW
    return 'W' if won == root_side else 'L'


def _propagate_proofs(path, root_side: 'Side') -> None:
    """Minimax proof propagation up a just-backed-up path.

    A certain loss has no upside; no Q estimate outranks that fact —
    proofs are the dominance layer above graded values.  Chance nodes
    BLOCK proofs (sampled outcomes prove nothing) EXCEPT kernel-exact
    full-cover combat nodes (_kernel_chance_proof); 'all children'
    means all TRACKED children (the prune's commitment).  Grading of
    short-circuit proofs uses the known-proven children only."""
    for n in reversed(path):
        if n.is_chance or n.proven is not None or not n.children:
            if n.proven is None and not n.is_terminal:
                # kernel-exact full-cover chance nodes may prove
                # (all branches proven, same class); sampled block
                if _kernel_chance_proof(n):
                    continue
                return
            continue
        mover_root = (n.side == root_side)
        want = 'W' if mover_root else 'L'
        got = [c.proven for c in n.children]
        vals = [c.proven_value for c in n.children
                if c.proven is not None]
        if want in got:
            n.proven = want
            pick = [c.proven_value for c in n.children
                    if c.proven == want]
            n.proven_value = (max(pick) if mover_root else min(pick))
        elif all(g is not None for g in got):
            n.proven = 'L' if mover_root else 'W'
            n.proven_value = (max(vals) if mover_root else min(vals))
        else:
            return


def _settle_new_node(node: MCTSNode, pre_side: Side,
                     fp_net: 'WOTRValueNet',
                     sh_net: 'WOTRValueNet',
                     root_side: Side) -> float:
    """Terminal-check / acting-side / expansion for a node whose state
    was just materialized.  Returns the leaf value in the ROOT side's
    currency (root perspective, root profile terminal)."""
    _etv = getattr(node.state, '_search_endturn_victory', None)
    if _etv is not None:
        node.is_terminal = True
        node.terminal_value = _search_terminal(_etv, root_side,
                                               node.state)
        node.proven = _proof_mark(_etv, root_side)
        node.proven_value = node.terminal_value
        node.side = pre_side
        return node.terminal_value
    result = game_logic.check_all_victory_conditions(node.state)
    if game_logic.should_game_end(result, end_of_turn=False):
        node.is_terminal = True
        node.terminal_value = _search_terminal(result, root_side,
                                               node.state)
        node.proven = _proof_mark(result, root_side)
        node.proven_value = node.terminal_value
        node.side = pre_side
        return node.terminal_value
    next_side = _next_acting_side(node.state, pre_side)
    if next_side is None:
        node.side = pre_side
        _v9 = _mcts_eval_state(node.state, root_side, fp_net, sh_net)
        if (_TURN_HORIZON > 0
                and getattr(node.state, '_search_turn_crossings', 0)
                < _TURN_HORIZON):
            node.children.append(MCTSNode(
                state=None, side=pre_side, parent=node,
                action=Action(die_idx=-1, action_type='end_turn',
                              params={}),
                prior=1.0))
        return _v9
    node.side = next_side
    return _expand_node(node, fp_net, sh_net, root_side)


def _select_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Select the child with highest PUCT score.

    Proven children are excluded — their value is exact, so visits
    there are pure waste; the freed budget deepens unproven lines.
    3A: at opponent card windows the compatibility mask restricts the
    candidate set to edges this simulation's overlay hand can play
    (prior mass renormalized over the compatible set), after lazy
    child addition for overlay-legal actions the stored enumeration
    lacks."""
    ov = _SIM_OVERLAY
    masked = (ov is not None and node.side == ov.opp_side
              and node.state is not None)
    if masked:
        try:
            masked = _overlay_lazy_children(node, ov) != 'none'
        except Exception:
            OVERLAY_STATS['mask_error'] += 1
            if OVERLAY_ASSERT:
                raise
            masked = False
    pscale = 1.0
    if masked:
        cand = [c for c in node.children
                if c.proven is None and _ov_child_ok(c, node, ov)]
        if not cand:
            cand = [c for c in node.children
                    if _ov_child_ok(c, node, ov)]
        if not cand:
            # every child incompatible with this sample: fall back
            # unmasked (the opponent must still act) — counted
            OVERLAY_STATS['mask_empty'] += 1
            cand = [c for c in node.children if c.proven is None] \
                or node.children
        else:
            pmass = sum(c.prior for c in cand)
            if pmass > 1e-9:
                pscale = 1.0 / pmass
    else:
        cand = [c for c in node.children if c.proven is None]
        if not cand:
            cand = node.children
    sqrt_parent = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0
    best_score = -float('inf')
    best_child = None

    for child in cand:
        # Q from parent's perspective: same side → use q directly,
        # opponent → negate
        if child.side == node.side:
            q = child.q_value
        else:
            q = -child.q_value
        # For unexpanded children (visit_count=0), Q defaults to 0
        # so exploration term dominates, which is desired
        u = c_puct * child.prior * pscale * sqrt_parent \
            / (1.0 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


# ===================================================================
# Area-Focused MCTS: Hot Zone Detection
# ===================================================================

def _get_hot_regions(action: Action, state: GameState) -> set:
    """Return regions affected by an action + their 1-hop neighbors."""
    hot = set()
    params = action.params if action.params else {}
    for key in ('from', 'to', 'region', 'target'):
        val = params.get(key, '')
        if val:
            r = sanitize(val)
            if r in board.adj:
                hot.add(r)
                hot.update(board.adj.get(r, set()))
    if action.action_type in ('fellowship_move', 'fellowship_declare'):
        try:
            r = sanitize(state.fellowship.region.key)
            hot.add(r)
            hot.update(board.adj.get(r, set()))
        except (AttributeError, TypeError):
            pass
    return hot


def _select_child_from_subset(children: list, parent: 'MCTSNode',
                               c_puct: float) -> 'MCTSNode':
    """Select best child from a subset using PUCT (same formula as _select_child)."""
    sqrt_parent = math.sqrt(parent.visit_count) if parent.visit_count > 0 else 1.0
    best_score = -float('inf')
    best_child = None
    for child in children:
        if child.visit_count > 0:
            if child.side == parent.side:
                q = child.q_value
            else:
                q = -child.q_value
        else:
            q = 0.0
        u = c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _action_in_regions(action: Action, hot_regions: set) -> bool:
    """Check if action involves any hot-zone region."""
    if action.action_type in ('pass', 'draw_card', 'fellowship_hide',
                               'advance_political', 'play_event'):
        return True  # Non-regional or hard-to-determine region
    params = action.params if action.params else {}
    for key in ('from', 'to', 'region', 'target'):
        val = params.get(key, '')
        if val and sanitize(val) in hot_regions:
            return True
    return False


# R10 counter-knob (LAB 2026-08-09): the honest recalibrated value
# currency widens/lowers in-tree Q levels; at fixed c_puct the PUCT
# exploration term weakens RELATIVELY and prior-disfavored survival
# lines starve (erebor_recapture adjudication probe: seeds lift the
# iron-hills muster to top-Q at 400 sims, recal nets never give it
# the visits).  Env-pinned, default 4.0 = byte-identical; adoption
# value comes from fixture-validated A/B only.
_C_PUCT = float(os.environ.get('WOTR_C_PUCT', '4.0') or 4.0)


def mcts_search(state: GameState, side: Side,
                fp_net: 'WOTRValueNet', sh_net: 'WOTRValueNet',
                num_simulations: int = 100, c_puct: float = _C_PUCT,
                temperature: float = 1.0,
                add_noise: bool = True,
                hot_regions: set = None,
                bonus_sims: int = 0,
                determinize: bool = True
                ) -> 'Optional[Tuple[Action, dict]]':
    """Run MCTS from the given state and return the selected action.

    Args:
        state: Current game state
        side: Side to move
        fp_net, sh_net: Value networks
        num_simulations: Number of MCTS simulations
        c_puct: Exploration constant for PUCT
        temperature: Temperature for action selection (0 = greedy)
        add_noise: Add Dirichlet noise to root priors
        determinize: ISMCTS 3A — every simulation samples a fresh
            opponent hand from the exchangeable-uniform posterior
            over deck + hand + face-down discards (per deck) and
            re-shuffles own-deck + hunt-pool ORDERS, so lookahead
            neither cheats with perfect information nor fuses one
            imaginary hand across the whole search

    Returns:
        (selected Action, visit_info dict), or None if no legal actions.
    """
    if not HAS_TORCH:
        return None

    # v30 cross-turn search: the turn-transition edge needs the nets
    # for its greedy phase picks (declare/eyes/hand-limit)
    global _SEARCH_NETS, _SIM_OVERLAY, _SEARCH_ROOT_SIDE
    _SEARCH_NETS = (fp_net, sh_net)
    _SEARCH_ROOT_SIDE = side

    # ROOT/SEARCH BLINDING trigger (USER RULING 2026-08-16, verbatim:
    # "NO YOU CANNOT SEE THE CARD THAT IS COMMITTED"): the root player
    # is the DEFENDER while the attacker's combat card lies committed
    # face-down.  Tree reuse and stashing are disabled for these
    # (rare) roots — a stashed subtree carries sample-specific
    # committed-card worlds and the blinded root clone's blanked
    # discard slot must never leak back as a "real" state.
    _blind9 = False
    _pcr9 = getattr(state, 'pending_combat', None)
    if (_pcr9 and _pcr9.get('atk_combat_card')
            and _pcr9.get('attacker_side') != side):
        _blind9 = True

    # Lever 1: try to reuse the previous search's chosen subtree
    root = None
    _reused = False
    if _TREE_REUSE:
        _cand9 = _TREE_CACHE.get('child')
        _TREE_CACHE['child'] = None      # single-shot
        if (not _blind9
                and _cand9 is not None and _cand9.state is not None
                and not _cand9.is_chance and _cand9.children
                and _cand9.side == side
                # ROOT-CURRENCY INVARIANT: a stashed subtree's stored
                # values are in the PREVIOUS search's root currency —
                # reuse only when that currency is this root's
                and _TREE_CACHE.get('root_side') == side):
            _sig_now = _state_sig(state)
            # PHANTOM-PLAY FIX (USER trace w5_g2 T8 Morgul Wound,
            # 2026-07-20): _state_sig fingerprints hands by COUNT only,
            # so a cached child whose state carries the PREVIOUS
            # search's DETERMINIZED (imaginary) hand for `side` could
            # be reused as this side's root — get_legal_actions then
            # offered play_event for cards not really held (10/23 card
            # plays in one game were phantoms, each burning a die on a
            # no-op). The mover's OWN hand is perfect information: it
            # must match reality EXACTLY or the tree is stale.
            if (_sig_now is not None
                    and _sig_now == _TREE_CACHE.get('sig')
                    and sorted(map(str, _cand9.state.hands[side].hand))
                    == sorted(map(str, state.hands[side].hand))):
                root = _cand9
                root.parent = None
                _reused = True
                _TREE_CACHE['hits'] += 1
        if not _reused:
            _TREE_CACHE['misses'] += 1

    if root is None:
        # 3A: NO root-once hand swap — the root clone keeps the true
        # state (root menu/priors are determinization-invariant: own
        # menu + count-only opponent encoding); each simulation
        # overlays its own posterior sample before any hidden read.
        root = MCTSNode(state=clone_state(state), side=side)
        if _blind9:
            # ROOT BLINDING: the committed card was physically
            # discarded face-up at COMMIT time (CL-217), so the root
            # encode's card-status plane (0.5 = seen) would read the
            # very identity the defender cannot see.  Blank that one
            # copy to a placeholder ('unseen') on the SEARCH-SCOPED
            # root clone and record the slot; each simulation's
            # overlay writes ITS sampled identity into the slot
            # (patch), so every in-search encode/resolution sees a
            # sampled world, never the true identity.
            try:
                _cidb9 = _pcr9.get('atk_combat_card')
                _dfu9 = getattr(root.state, 'discard_face_up',
                                None) or {}
                for _dk9, _lst9 in _dfu9.items():
                    for _ix9 in range(len(_lst9) - 1, -1, -1):
                        if _lst9[_ix9] == _cidb9:
                            _lst9[_ix9] = '_committed_face_down_'
                            root.state._commit_slot = (_dk9, _ix9)
                            break
                    if getattr(root.state, '_commit_slot',
                               None) is not None:
                        break
                # nonce: lets the per-sim patch tell THIS battle's
                # THIS card window apart from a fresh in-tree battle
                # (whose pc dict is built without it)
                if root.state.pending_combat is not None:
                    root.state.pending_combat['_commit_nonce'] = True
            except Exception:
                pass
        _expand_node(root, fp_net, sh_net, side)
        root.visit_count = 1  # Root counts as visited

    if not root.children:
        return None

    # capture CLEAN priors BEFORE root noise — final selection and the
    # stored search stats must see the BC prior, never the Dirichlet
    # mix (reused roots were expanded interior and never noised)
    priors_clean = np.array([c.prior for c in root.children],
                            dtype=np.float32)

    # task-16 LAW 1B: per-search confidence reference (root read)
    if _CONF_LEAF and _CONF_FN is not None \
            and _CONF_REF in ('root', 'peak', 'mean'):
        try:
            _cv16 = encode_state(root.state, side)
        except Exception:
            _cv16 = None
        _conf_search_reset(_cv16, side)

    # Add Dirichlet noise to root priors for exploration (in-tree
    # search only; final selection uses priors_clean).  F1: free-
    # action arms are exempt under WOTR_FREE_NOISE_EXEMPT (default
    # off; exact legacy arithmetic when inactive).
    if add_noise and not _reused and len(root.children) > 0:
        noise = np.random.dirichlet([0.3] * len(root.children))
        _apply_root_noise(root.children, noise)

    # FIX 2 (task 14): root forced-playout floor for the top-prior
    # free-action arms — restores an exploration source for ~1e-4-
    # prior free arms without the Dirichlet subsidy's junk-wide blast.
    # Forced counts live ON THE NODE (forced_visits) so they survive
    # tree reuse; a reused child already at/above the floor is not
    # re-forced and its old forced visits stay excluded from 'earned'.
    _force_idx = []
    if _FREE_FORCE > 0:
        _fr9 = sorted(
            [(float(priors_clean[i]), i)
             for i, c in enumerate(root.children)
             if c.action is not None
             and c.action.action_type in _FREE_ACTION_TYPES],
            reverse=True)
        _force_idx = [i for _p9, i in _fr9[:_FREE_CANDS_N]]

    # small menus need visit mass per arm, not exhaustive search:
    # cap the budget at 60 sims/child (floor 120)
    if len(root.children) <= 6:
        num_simulations = min(num_simulations,
                              max(120, 60 * len(root.children)))

    def _simulate_from(node, path):
        """One simulation starting at `node` (already appended to
        `path`): walk through decision AND chance nodes to a leaf,
        materialize/settle it, backprop.  An action that consumed RNG
        becomes a chance node whose Q averages over independently
        sampled outcomes; leaf_term marks backups whose value came
        from a RESOLVED terminal (exact, not a net estimate).
        ROOT-CURRENCY INVARIANT: every leaf value here is in the ROOT
        side's currency (root perspective, root profile).
        3A: one fresh posterior overlay per simulation (k=1), sampled
        from the root state's unseen pool; every hidden-zone read
        below is patch-preceded."""
        global _SIM_OVERLAY, _LAST_LEAF_CONF, _LAST_LEAF_PHI
        _LAST_LEAF_CONF = 1.0     # task-15 LAW 2: per-sim leaf conf
        _LAST_LEAF_PHI = 0.0      # task-17 LAW 1R: per-sim leaf phi
        if determinize:
            try:
                _SIM_OVERLAY = _SimOverlay(root.state, side)
                OVERLAY_STATS['samples'] += 1
            except Exception:
                _SIM_OVERLAY = None
                OVERLAY_STATS['sample_error'] += 1
                if OVERLAY_ASSERT:
                    raise
        else:
            _SIM_OVERLAY = None
        leaf_term = False
        while True:
            if node.is_terminal:
                leaf_value = node.terminal_value
                leaf_term = True
                break
            if node.proven is not None:
                # proven interior node: exact value, no descent
                leaf_value = node.proven_value
                leaf_term = True
                break
            if node.is_chance:
                branch = _select_outcome(node)
                if branch is None:
                    # every sampling attempt failed — loss for the
                    # acting side, signed into root perspective
                    node.is_terminal = True
                    node.terminal_value = (-1.0 if node.side == side
                                           else 1.0)
                    leaf_value = node.terminal_value
                    leaf_term = True
                    break
                path.append(branch)
                node = branch
                if (not node.is_terminal and not node.children
                        and node.visit_count == 0):
                    # freshly sampled outcome: settle it now
                    leaf_value = _settle_new_node(
                        node, node.parent.side, fp_net, sh_net, side)
                    leaf_term = node.is_terminal
                    break
                continue
            if node.children:
                if node is root and _force_idx:
                    # FIX 2: serve the forced floor before PUCT
                    _fp9 = None
                    for _i9 in _force_idx:
                        _c9 = root.children[_i9]
                        if _c9.proven is None \
                                and _c9.visit_count < _FREE_FORCE \
                                and (_fp9 is None
                                     or _c9.visit_count
                                     < root.children[_fp9]
                                     .visit_count):
                            _fp9 = _i9
                    if _fp9 is not None:
                        root.children[_fp9].forced_visits += 1
                        node = root.children[_fp9]
                        path.append(node)
                        continue
                node = _select_child(node, c_puct)
                path.append(node)
                continue
            if node.state is None:
                # 3A patch point: this simulation's overlay must be
                # live for the materializing apply AND the kernel
                # forced applies below (combat witness plans / hunt
                # fork-driver walks pop decks during branch build)
                _overlay_prepare_apply(node.parent.state, node.side,
                                       node.action)
                # Hunt-bearing applies: exact kernel branches FIRST —
                # the Mordor tile pop consumes no RNG, so the sampled
                # detector below would (deterministically) leak the
                # pool order instead of branching
                if (node.action is not None
                        and node.action.action_type in _HUNT_PARENTS
                        and _hunt_action_gate(node)):
                    _hkind, _hst = _hunt_expand_try(node)
                    if _hkind == 'chance':
                        branch = node.outcomes[0]
                        path.append(branch)
                        leaf_value = _settle_new_node(
                            branch, node.side, fp_net, sh_net, side)
                        leaf_term = branch.is_terminal
                        break
                    if _hkind == 'det':
                        node.state = _hst
                        leaf_value = _settle_new_node(
                            node, node.parent.side, fp_net, sh_net,
                            side)
                        leaf_term = node.is_terminal
                        break
                # Lazy materialization + RNG detection
                _pl35 = (_hidden_zone_lens(node.parent.state)
                         if _SIM_OVERLAY is not None else None)
                try:
                    new_state, used_rng = _apply_action_detect_rng(
                        node.parent.state, node.action)
                except Exception:
                    # Action failed — loss for the acting side,
                    # signed into root perspective
                    node.is_terminal = True
                    node.terminal_value = (-1.0 if node.side == side
                                           else 1.0)
                    leaf_value = node.terminal_value
                    leaf_term = True
                    break
                if (not used_rng and _pl35 is not None
                        and _hidden_zone_lens(new_state) != _pl35):
                    # RNG-free draw/pop of a hidden-ORDER zone: a
                    # chance event under 3A — one materialization
                    # must not fuse this simulation's sampled order
                    used_rng = True
                    OVERLAY_STATS['det_draw_chance'] += 1
                if used_rng:
                    node.is_chance = True
                    node.outcomes = []
                    if (node.action is not None
                            and node.action.action_type
                            in _KERNEL_PARENTS
                            and _kernel_expand_combat(node, new_state)):
                        # exact combat branches installed (no sampling);
                        # this visit descends the top-mass branch
                        branch = node.outcomes[0]
                        path.append(branch)
                        leaf_value = _settle_new_node(
                            branch, node.side, fp_net, sh_net, side)
                        leaf_term = branch.is_terminal
                        break
                    _ms0 = getattr(new_state, '_search_roll_ms', None) \
                        if node.action is not None \
                        and node.action.action_type == 'end_turn' \
                        else None
                    branch = MCTSNode(state=new_state, side=node.side,
                                      parent=node, action=None,
                                      prior=(_roll_ms_prob(_ms0)
                                             if _ms0 is not None
                                             else 1.0))
                    if _ms0 is not None:
                        node.drops += 1
                    branch.out_sig = _chance_sig(new_state)
                    node.outcomes.append(branch)
                    path.append(branch)
                    leaf_value = _settle_new_node(
                        branch, node.side, fp_net, sh_net, side)
                    leaf_term = branch.is_terminal
                    break
                node.state = new_state
                leaf_value = _settle_new_node(
                    node, node.parent.side, fp_net, sh_net, side)
                leaf_term = node.is_terminal
                break
            # Already expanded node with no children (no legal actions)
            leaf_value = _mcts_eval_state(
                node.state, side, fp_net, sh_net)
            break

        # BACKPROP: leaf_value is in root perspective, so a node's sum
        # gains +value iff the node acts for the root side (per-node
        # sums stay in the node side's perspective, one currency)
        for n in reversed(path):
            n.visit_count += 1
            if n.side == side:
                n.value_sum += leaf_value
            else:
                n.value_sum -= leaf_value
            # squared backup (diagnostic spread) + resolved-terminal
            # count (feeds the terminal-resolved override)
            n.value_sq += leaf_value * leaf_value
            if leaf_term:
                n.term_visits += 1
            if _CONF_LEAF or _CONF_TRACK:
                # task-15 LAW 2: exact backups carry full support
                n.conf_sum += 1.0 if leaf_term else _LAST_LEAF_CONF
                # task-17 LAW 1R: scaffold component (terminals pay
                # phi via _search_terminal — same tracked value)
                n.phi_sum += _LAST_LEAF_PHI
        if leaf_term:
            _propagate_proofs(path, side)

    for _sim in range(num_simulations):
        if root.proven is not None:
            break
        _simulate_from(root, [root])

    # --- Bonus simulations for hot-zone focused search ---
    if hot_regions and bonus_sims > 0 and root.children:
        hot_children = [c for c in root.children
                        if _action_in_regions(c.action, hot_regions)]
        if not hot_children:
            hot_children = root.children
        for _bsim in range(bonus_sims):
            node = _select_child_from_subset(hot_children, root, c_puct)
            _simulate_from(node, [root, node])

    # Per-action root statistics: visit distribution (spatial-scorer
    # target) + stored-search-stats instrumentation (clean priors,
    # parent-perspective child Q, raw visit counts)
    child_visits_raw = np.array([c.visit_count for c in root.children],
                                dtype=np.float32)
    total_v = child_visits_raw.sum()
    child_visits = (child_visits_raw / total_v if total_v > 0
                    else child_visits_raw.copy())
    q_values = np.array(
        [(c.q_value if c.side == root.side else -c.q_value)
         for c in root.children], dtype=np.float32)
    # per-arm empirical variance of backup values (diagnostic only;
    # sign-invariant, so no perspective flip needed)
    q_var = np.array(
        [max(0.0, c.value_sq / c.visit_count
             - (c.value_sum / c.visit_count) ** 2)
         if c.visit_count > 0 else 0.0
         for c in root.children], dtype=np.float32)
    # fraction of backups from resolved terminals (exact values)
    term_frac = np.array(
        [c.term_visits / c.visit_count if c.visit_count > 0 else 0.0
         for c in root.children], dtype=np.float32)
    visit_info = {
        'action_encs': np.array([encode_action(c.action, state, side)
                                 for c in root.children], dtype=np.float32),
        'region_indices': np.array([_get_action_region_index(c.action)
                                    for c in root.children], dtype=np.int32),
        'visits': child_visits,
        'visit_counts': child_visits_raw.astype(np.int32),
        'q_values': q_values,
        'q_var': q_var,
        'term_frac': term_frac,
        'priors_clean': priors_clean,
        'proven': np.array([{'W': 1, 'L': -1}.get(c.proven, 0)
                            for c in root.children], dtype=np.int8),
        # exact graded proof values (1.225): the all-proven-loss pick
        # must use these, not the visit-frozen q means
        'proven_value': np.array([float(getattr(c, 'proven_value', 0.0))
                                  for c in root.children],
                                 dtype=np.float32),
        'forced': np.array([getattr(c, 'forced_visits', 0)
                            for c in root.children],
                           dtype=np.int32),
        # task-15 LAW 2: per-arm mean leaf confidence (1.0 when the
        # confidence wave is off — full support, laws inert)
        'conf': np.array(
            [(c.conf_sum / c.visit_count)
             if ((_CONF_LEAF or _CONF_TRACK) and c.visit_count > 0)
             else 1.0
             for c in root.children], dtype=np.float32),
        # task-17 LAW 1R: per-arm mean leaf scaffold (root currency)
        'phi': np.array(
            [(c.phi_sum / c.visit_count)
             if ((_CONF_LEAF or _CONF_TRACK) and c.visit_count > 0)
             else 0.0
             for c in root.children], dtype=np.float32),
    }

    def _stash_tree(child):
        # Lever 1: remember the chosen subtree + its predicted
        # post-state fingerprint for possible reuse next search
        # (never from a blinded root — its states carry sampled
        # committed-card worlds + the blanked discard slot)
        if _TREE_REUSE and not _blind9 and child is not None \
                and child.state is not None and not child.is_chance:
            _TREE_CACHE['child'] = child
            _TREE_CACHE['sig'] = _state_sig(child.state)
            _TREE_CACHE['root_side'] = side

    # final pick: visit-argmax with evidence-gated overrides (see
    # select_root_action; regret ~ 0 marks deliberate picks)
    idx, pi_sel, sel_diag = select_root_action(
        {'p': priors_clean, 'q': q_values, 'n': child_visits_raw,
         'tf': term_frac, 'pv': visit_info['proven'],
         'pvv': visit_info['proven_value'],
         'fn': visit_info['forced'], 'cf': visit_info['conf'],
         'ph': visit_info['phi']},
        temperature)
    visit_info['pi_sel'] = pi_sel.astype(np.float32)
    visit_info['q_hat'] = sel_diag['q_hat'].astype(np.float32)
    visit_info['sigma'] = sel_diag['sigma']
    visit_info['played'] = idx
    visit_info['regret'] = sel_diag['regret']
    _stash_tree(root.children[idx])
    _SIM_OVERLAY = None      # overlays are search-scoped
    return root.children[idx].action, visit_info


def mcts_action(state: GameState, side: Side,
                fp_net: 'WOTRValueNet', sh_net: 'WOTRValueNet',
                num_simulations: int = 100, c_puct: float = _C_PUCT,
                temperature: float = 1.0,
                add_noise: bool = True) -> 'Optional[Action]':
    """Select an action via MCTS (action only, no visit info).

    Returns the selected Action, or None if no legal actions.
    """
    result = mcts_search(state, side, fp_net, sh_net,
                         num_simulations=num_simulations,
                         c_puct=c_puct, temperature=temperature,
                         add_noise=add_noise)
    if result is None:
        return None
    action, _policy = result
    return action


def mcts_action_with_policy(state: GameState, side: Side,
                            fp_net: 'WOTRValueNet', sh_net: 'WOTRValueNet',
                            num_simulations: int = 100, c_puct: float = _C_PUCT,
                            temperature: float = 1.0,
                            add_noise: bool = True,
                            hot_regions: set = None,
                            bonus_sims: int = 0):
    """MCTS returning both action and visit info for policy training.

    Returns (Action, visit_info dict) or (None, None).
    """
    # hunt_sacrifice/hunt_gollum flow through the normal search —
    # prior-first bypasses are retired (select_root_action already
    # makes the prior's line win by default)
    result = mcts_search(state, side, fp_net, sh_net,
                         num_simulations=num_simulations,
                         c_puct=c_puct, temperature=temperature,
                         add_noise=add_noise,
                         hot_regions=hot_regions,
                         bonus_sims=bonus_sims)
    if result is None:
        return None, None
    return result


# ===================================================================
# Step 9: Phase 2 TD(λ) Self-Play
# ===================================================================

def _safe_next_turn(state: GameState):
    """Advance to next turn, handling empty decks gracefully.

    Returns True if the ROLLOVER itself decided the game (ring
    victory at corruption 12 via the Mordor inactivity penalty —
    P0 2026-08-15: the engine used to play ON past a won game; ring
    victories are IMMEDIATE, rulebook p.17, wherever corruption
    increments).  Callers must stop the game when this returns True.
    """
    # Mordor inactivity penalty: +1 corruption if Fellowship didn't move/hide
    if (state.fellowship.mordor_entered
            and not getattr(state, 'fellowship_acted_in_mordor', False)
            and state.fellowship.corruption < 12):
        state.fellowship.corruption = min(12, state.fellowship.corruption + 1)
        if state.fellowship.corruption >= 12:
            return True

    state.turn += 1
    state.eyes_allocated = 0
    # USER AUDIT 2026-07-08: the eye-allocation MINIMUM (1 if the FP
    # player retrieved a die from the Hunt Box = moved the Fellowship)
    # depends on this flag — the self-play turn path cleared the box
    # WITHOUT setting it, so the mandatory minimum never applied.
    state._fp_used_hunt_box_last_turn = any(
        d.side is Side.FREE_PEOPLES for d in state.hunt_box)
    for d in state.hunt_box:
        d.used, d.current = False, None
    state.hunt_box.clear()
    for pool in state.dice.values():
        for d in pool:
            d.used, d.current = False, None
    state.fp_ring_used_this_turn = False
    state.sh_ring_used_this_turn = False
    # A7: an unspent Elven-Ring legality waiver never crosses a turn
    # boundary (mirrors game_logic.begin_turn_reset - this is the
    # FOURTH turn implementation and AUTO is the caller that runs it).
    state.fp_ring_waiver = False
    state.sh_ring_waiver = False
    state._ring_free_side = None
    state.sh_mouth_muster_used_this_turn = False
    state.fp_token_used_this_turn = False
    state.sh_token_used_this_turn = False
    state.dice_rolled = False
    # Reset Mordor activity tracking for the new turn
    state.fellowship_acted_in_mordor = False
    # NOTE (2026-07-24): the stay-heal (rulebook p.34) requires RE-
    # DECLARING in place - a decision, offered via the declare phase;
    # never an auto-heal here (see game_logic.next_turn note).
    # Minions mustered mid-game grant bonus action dice — the closed
    # loop's turn path never recalculated them (2023-11-03 ketchuptime
    # T9: the Mouth's T8 muster left the SH pool at 8 while the humans
    # rolled 9; the Foul Thing play starved and its Eye's +5 corruption
    # never applied — outcome flipped to ring-destroyed).  Recalculated
    # ONCE below, after pendings are cleared (nothing in between reads
    # the dice pools).
    # Clear pending multi-phase states (restore rearguard if combat was pending)
    if state.pending_combat is not None:
        _restore_rearguard(state, state.pending_combat)
    state.pending_combat = None
    state.pending_hunt = None
    state.pending_guide_change = None
    # F-2 (USER 2026-07-19): the Army-die double-move / Muster
    # double-recruit chains MUST die with the turn — a dangling chain
    # used to survive into the next turn, where get_legal_actions
    # handed the leftover second half to whichever side acted first
    # (the OPPONENT: fuzz seeds 9101/9140, FP played free moves on
    # Shadow's stale die).
    state.pending_army_move_die = None
    state.pending_army_move_from = None
    state.pending_army_move_side = None
    state.pending_army_move_preexist = None
    state.pending_muster_die = None
    state.pending_muster_first_type = None
    state.pending_muster_region = None
    state.pending_muster_side = None
    # OVERSIGHT #4 (2026-07-20): the card/discard pendings also die
    # with the turn — probe-proven to survive before this (stale
    # owner's menu leaked into the next turn via the decision-cap
    # valve or empty-menu fallthrough).
    state.pending_card_option = None
    if hasattr(state, 'pending_card_followup'):
        state.pending_card_followup = None
    if hasattr(state, 'pending_discard'):
        state.pending_discard = None
    # F-3: a battle abandoned by the per-turn decision cap may have
    # left a wiped garrison unconquered — settle control/siege state
    # now that the pendings are cleared (stable point).
    try:
        game_logic.cleanup_sieges(state)
    except Exception:
        pass
    game_logic.calculate_bonus_dice(state)
    # Draw cards, silently skipping empty decks
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for deck in ("character", "strategy"):
            try:
                game_logic.draw_event_card(state, s, deck)
            except RuntimeError:
                pass
    # Hand limit is 6 (rulebook): discarding down is a LEARNED decision —
    # open a pending phase instead of discarding randomly. The self-play
    # loop resolves it via MCTS before the turn's other phases.
    _check_hand_limits(state)


def _ai_eye_allocation(state: 'GameState', net: 'WOTRValueNet'):
    """Shadow AI pre-allocates eyes before rolling using value net evaluation.

    Capped at 0-2 eyes max — real games almost never allocate more than 2.
    The encoding now distinguishes allocated eyes vs rolled eyes vs FP hunt
    dice, so the value net can properly evaluate this tradeoff.
    """
    shadow_dice = [d for d in state.dice[Side.SHADOW] if not d.used]
    n_dice = len(shadow_dice)
    if n_dice <= 2:
        return
    max_eyes = min(2, n_dice - 2)  # Cap at 2, leave at least 2 for actions
    best_eyes = 0
    best_val = None
    net.eval()
    for n_eyes in range(max_eyes + 1):
        s2 = clone_state(state)
        if n_eyes > 0:
            indices = [i for i, d in enumerate(s2.dice[Side.SHADOW])
                       if not d.used][:n_eyes]
            game_logic.allocate_eyes(s2, n_eyes, indices=indices)
        vec = encode_state(s2, Side.SHADOW)
        with torch.no_grad():
            val = net(torch.from_numpy(np.array(vec, dtype=np.float32)).unsqueeze(0).to(DEVICE)).item()
        if best_val is None or val > best_val:
            best_val = val
            best_eyes = n_eyes
    if best_eyes > 0:
        indices = [i for i, d in enumerate(state.dice[Side.SHADOW])
                   if not d.used][:best_eyes]
        game_logic.allocate_eyes(state, best_eyes, indices=indices)


_MILITARY_ACTION_TYPES = frozenset({
    'army_move', 'army_attack', 'nazgul_move',
    'muster_recruit', 'muster_elite', 'muster_minion',
    'advance_political',
    'combat_card_play', 'combat_card_none', 'combat_continue',
    'combat_cease', 'combat_retreat', 'combat_no_retreat',
    'combat_effect_choice', 'casualty_allocate',
    'casualty_preserve_elites', 'casualty_preserve_units',
    'retreat_into_siege', 'siege_assault', 'siege_continue',
    'siege_press', 'siege_no_press', 'rearguard_split',
    'rearguard_none', 'victor_advance', 'victor_stay'})

_MILITARY_DIE_FACES = frozenset({'Army', 'Muster', 'Muster/Army'})


def _is_military_action(action_type, params=None, die_face=None):
    """1.143 (USER): is this a decision that could have defended /
    attacked?  Only these carry military reward deltas (VP + phi) -
    a fellowship move is structurally isolated from stronghold events.
    Military = army/muster/combat action families, a muster token, or
    ANY action spending an Army/Muster-faced die (e.g. an event card
    played with a muster die is spent defense capacity)."""
    if action_type in _MILITARY_ACTION_TYPES:
        return True
    p = params or {}
    # (ring rerolls carry no result face in params — the reroll outcome
    #  is random and unknown at classification time, so rings are not
    #  classified military here; the die they produce is classified when
    #  it is SPENT via the die_face argument below.)
    if action_type == 'use_token'             and 'muster' in str(p.get('token_type', '')).lower():
        return True
    if die_face in _MILITARY_DIE_FACES:
        return True
    return False


def _threat_potential(state: 'GameState') -> int:
    """1.142: count of FP-controlled strongholds that are UNDEFENDED
    (zero own army units) while a Shadow army stands in or adjacent.
    Used as a potential-based shaping term - the moment a garrison is
    stripped under threat, the potential drops (spatial credit)."""
    n = 0
    sh_cap = state.enemy_controlled.get(Side.SHADOW, set())
    for rkey, meta in board.regions.items():
        if not meta.get('is_stronghold'):
            continue
        nat = _region_nation(rkey)
        if nat is None or _side_of_nation(nat) != Side.FREE_PEOPLES:
            continue
        if rkey in sh_cap:
            continue
        army = _armies_lookup(state, rkey)
        own = sum(1 for u in (army.units if army else [])
                  if _side_of_nation(u.nation) == Side.FREE_PEOPLES)
        if own > 0:
            continue
        threat = False
        if army is not None and any(
                _side_of_nation(u.nation) == Side.SHADOW
                for u in army.units):
            threat = True
        else:
            for adj in state.board.adjacent(rkey):
                a2 = _armies_lookup(state, sanitize(adj))
                if a2 is not None and any(
                        _side_of_nation(u.nation) == Side.SHADOW
                        for u in a2.units):
                    threat = True
                    break
        if threat:
            n += 1
    return n


_GATE_DIST_CACHE = None


def _gate_distance(state: 'GameState', rkey: str):
    """BFS hops from `rkey` to the nearest Mordor gate (morannon /
    minas morgul).  Board adjacency is static -> cached per process."""
    global _GATE_DIST_CACHE
    if _GATE_DIST_CACHE is None:
        from collections import deque
        dist = {}
        dq = deque()
        for g in ('morannon', 'minas morgul'):
            gk = sanitize(g)
            dist[gk] = 0
            dq.append(gk)
        while dq:
            cur = dq.popleft()
            for nb in state.board.adjacent(cur):
                nk = sanitize(nb)
                if nk not in dist:
                    dist[nk] = dist[cur] + 1
                    dq.append(nk)
        _GATE_DIST_CACHE = dist
    return _GATE_DIST_CACHE.get(sanitize(rkey))


def _ring_progress_metric(state: 'GameState') -> float:
    """1.143b: hidden counter + TRUE pawn advancement toward the
    Mordor gates.  The raw counter resets on declare/reveal, which
    made every reveal confiscate the whole cycle's progress reward
    (a -0.56 spike on a fellowship_move) while net map progress
    earned nothing.  With the pawn term, a straight-toward-Mordor
    reveal nets 0 and only sideways wandering is clawed back."""
    try:
        d = _gate_distance(state, game_logic.fellowship_region(state))
        d0 = _gate_distance(state, 'rivendell')
    except Exception:
        d = d0 = None
    if d is None or d0 is None:
        return float(state.fellowship.progress)
    adv = d0 if state.fellowship.mordor_entered else d0 - d
    return float(state.fellowship.progress + adv)


def _effective_corruption(state: 'GameState') -> float:
    """1.146 (USER): corruption with companion health as a buffer.
    Companions still IN the fellowship can absorb hunt damage, so a
    level-3 companion is worth -3 effective corruption; losing him
    (death, sacrifice or separation) surfaces that risk immediately."""
    buf = sum(int(getattr(c, 'level', 0) or 0)
              for c in state.fellowship.companions)
    return float(state.fellowship.corruption) - buf


def _dice_potential(state: 'GameState') -> float:
    """1.144 (USER 2026-07-19, phi_dice): action-dice-economy potential,
    SHADOW-POSITIVE, in extra-dice units (weight applied at use).

    Counts each side's action dice beyond its base pool (SH 7 / FP 4):
    Saruman / Witch-king / Mouth for Shadow, crowned Aragorn / Gandalf
    the White for FP.  Potential-based shaping (gamma*phi' - phi) is
    policy-invariant and telescopes along the trajectory — it pays the
    MUSTER/CROWNING/TRANSFORM at the decision that earns the die, the
    credit the terminal-only view never attributed ("his extra die nets
    actions that are beneficial")."""
    sh_extra = max(0, len(state.dice.get(Side.SHADOW, [])) - 7)
    fp_extra = max(0, len(state.dice.get(Side.FREE_PEOPLES, [])) - 4)
    return float(sh_extra - fp_extra)


def _reward_stats(state: 'GameState') -> dict:
    """Snapshot of every quantity the step-reward shaping reads."""
    vp = game_logic.calculate_victory_points(state)
    return {'progress': _ring_progress_metric(state),
            'corruption': state.fellowship.corruption,
            'eff_corr': _effective_corruption(state),
            'sh_vp': vp['shadow'], 'fp_vp': vp['free_peoples'],
            'phi': _threat_potential(state),
            'dice_phi': _dice_potential(state),
            # F2 phi ring price inputs (inert while the flag is off)
            'fp_rings': int(state.fp_elven_rings),
            'sh_rings': int(state.sh_elven_rings)}


_RING_TRACK_TAIL = 5.0     # Mordor-track steps after the gate


def _win_frac_phi(x, total) -> float:
    """1.224 convex win-fraction potential: (progress/threshold)^2,
    clamped to [0,1].  A full track integrates to exactly 1.0 (one
    victory, commensurate with the +/-1 terminal); late progress is
    worth more than early (SH VP 9->10 pays 0.19, 0->1 pays 0.01;
    the last Mordor steps pay ~30x a Shire step)."""
    if total is None or total <= 0:
        return 0.0
    f = max(0.0, min(1.0, float(x) / float(total)))
    return f * f


def _compute_step_reward_split(pre: dict, state: 'GameState',
                               acting_side: 'Side') -> tuple:
    """(axis_reward, military_reward) for one step, INTERVAL semantics.

    1.224 (USER 2026-07-29): rewards are CONVEX WIN-FRACTION deltas —
    every axis is Delta((x/threshold)^2) so all four profiles trade in
    one currency ("fractions of a victory"), late progress outweighs
    early, and opponent-axis terms are SYMMETRIC (blocking Shadow's
    9th VP is automatically worth 0.19 — the old hand-tuned -0.30
    defense coefficient is superseded by the currency's convexity).
    Still potential-based (deltas telescope) -> policy-invariant.

    1.143 (USER): military terms are returned SEPARATELY so the
    episode loop can BANK them and pay them only on the side's next
    MILITARY-class action.  A fellowship move is never blamed or
    credited for stronghold events.  Axis terms land on every step.
    1.140 profile-purity and 1.146 effective corruption carry over.
    """
    axis = 0.0
    mil = 0.0
    vp = game_logic.calculate_victory_points(state)
    try:
        ring_total = _gate_distance(state, 'rivendell') \
            + _RING_TRACK_TAIL
    except Exception:
        ring_total = 13.0
    d_ring = (_win_frac_phi(_ring_progress_metric(state), ring_total)
              - _win_frac_phi(pre['progress'], ring_total))
    d_fpvp = (_win_frac_phi(vp['free_peoples'],
                            game_logic.FP_VP_VICTORY)
              - _win_frac_phi(pre['fp_vp'], game_logic.FP_VP_VICTORY))
    d_shvp = (_win_frac_phi(vp['shadow'], game_logic.SHADOW_VP_VICTORY)
              - _win_frac_phi(pre['sh_vp'],
                              game_logic.SHADOW_VP_VICTORY))
    d_corr = (_win_frac_phi(state.fellowship.corruption, 12.0)
              - _win_frac_phi(pre['corruption'], 12.0))
    if pre.get('eff_corr') is not None:
        d_ecorr = (_win_frac_phi(_effective_corruption(state), 12.0)
                   - _win_frac_phi(pre['eff_corr'], 12.0))
    else:
        d_ecorr = d_corr
    fp_prof = PLAYER_PROFILES.get('fp')
    sh_prof = PLAYER_PROFILES.get('sh')
    # 1.142 threat potential + 1.144 phi_dice: unchanged (already
    # potential-based, zero-sum, banked)
    d_phi = None
    if pre.get('phi') is not None:
        d_phi = _threat_potential(state) - pre['phi']
    d_dice = None
    if pre.get('dice_phi') is not None:
        d_dice = _dice_potential(state) - pre['dice_phi']
    # F2 phi_rings IN-STREAM: acting side's OWN ring-count delta pays
    # on the axis stream every step (never banked) — same potential
    # currency as the search leaf term, acting-side-positive sign.
    # (Own-count only: the FP->SH transfer credits SH's stream at
    # SH's next sample; asymmetric but conserving — A/B semantics.)
    if _PHI_RINGS_RHO and pre.get('fp_rings') is not None:
        if acting_side == Side.FREE_PEOPLES:
            axis += _PHI_RINGS_RHO * (int(state.fp_elven_rings)
                                      - int(pre['fp_rings']))
        else:
            axis += _PHI_RINGS_RHO * (int(state.sh_elven_rings)
                                      - int(pre['sh_rings']))
    # 1.224b (USER 2026-07-30): threat-phi and dice-phi are MILITARY
    # currencies — paying them profile-blind was the residual leakage
    # channel that had fp_ring nets capturing strongholds (a capture
    # drops Shadow threat potential -> banked mil credit -> paid on
    # the military action; batch-6 exemplar: fp_ring won MILITARY in
    # 6 turns, fellowship never past mdist 8).  Non-military profiles
    # get NO phi terms; their only military-bucket term is the
    # symmetric opponent-VP delta (defense, capture-neutral).
    if acting_side == Side.FREE_PEOPLES:
        if fp_prof in (None, 'military', 'mixed'):
            if d_phi is not None:
                mil -= 0.15 * d_phi
            if d_dice is not None:
                mil -= 0.20 * d_dice
        if fp_prof == 'ring':
            axis += d_ring - d_ecorr
            mil -= d_shvp
        elif fp_prof == 'military':
            mil += d_fpvp - d_shvp
        else:
            # profiles off: blended convex currency
            axis += d_ring - d_ecorr
            mil += d_fpvp - d_shvp
    else:
        if sh_prof in (None, 'military', 'mixed'):
            if d_phi is not None:
                mil += 0.15 * d_phi
            if d_dice is not None:
                mil += 0.20 * d_dice
        if sh_prof == 'corruption':
            axis += d_corr - d_ring
            axis -= 0.005
        elif sh_prof == 'military':
            mil += d_shvp - d_fpvp
            axis -= 0.01
        else:
            mil += d_shvp - d_fpvp
            axis += d_corr - d_ring
            axis -= 0.005
    return axis, mil




# 1.145 (USER 2026-07-19): a profile net's terminal is conditioned on
# HOW the game was won.  On-mission victory keeps the full +/-1;
# an OFF-mission win pays 0.2 (w10_ep00023: the fp-ring net parked the
# fellowship at Hollin for 20 turns and militarized because the
# profile-blind terminal paid +1.0 for the military win — the 1.140
# profile-pure SHAPING was being out-voted 5-10x by the terminal).
# Losses stay -1 in full (losing off-mission is still losing).
_PROFILE_ON_MISSION = {
    ('fp', 'ring'): ('ring_destroyed',),
    ('fp', 'military'): ('military',),
    ('sh', 'corruption'): ('ring_corruption',),
    ('sh', 'military'): ('military',),
    # mixed: EVERY win is on-mission — full terminal either way
    ('fp', 'mixed'): ('ring_destroyed', 'military'),
    ('sh', 'mixed'): ('ring_corruption', 'military'),
}


# 1.225b (USER 2026-08-03 "zero off mission for fp but keep it for
# shadow"): under the focus run, fp_ring was chasing military wins in
# 21% of games — the off-mission +0.2 was the only jackpot INSIDE the
# search horizon while the ring win sits far outside it.  FP now gets
# ZERO for off-mission wins; Shadow keeps 0.2 (a corruption-profile
# Shadow stumbling into a military win still denied FP the game).
_OFFMISSION_FP = float(os.environ.get('WOTR_OFFMISSION_FP', '0.0')
                       or 0.0)
_OFFMISSION_SH = float(os.environ.get('WOTR_OFFMISSION_SH', '0.2')
                       or 0.0)


def _profile_terminal(base: float, side_tag: str, profile,
                      victory_type) -> float:
    """Scale a WON terminal by mission fit; losses/truncations pass."""
    if base <= 0.0 or profile is None:
        return float(base)
    on_mission = str(victory_type or '') in _PROFILE_ON_MISSION.get(
        (side_tag, str(profile)), ())
    if on_mission:
        return float(base)
    off = _OFFMISSION_FP if side_tag == 'fp' else _OFFMISSION_SH
    return off * float(base)


# 1.225 (USER 2026-08-02): GRADED CLOSENESS TERMINALS.  "We should not
# resign or lose hope in a lost endgame.  Being at 8/10 VP is better
# than 2/10; 1 step from destroying the ring is better than 10 steps."
# The LOSER's terminal is -1 + KAPPA * closeness(final_state) instead
# of a flat -1; closeness is PROFILE-PURE and priced in the SAME convex
# win-fraction currency as the step rewards (_win_frac_phi), so a loss
# ranges [-1, -1+KAPPA].  Winner terminals are UNCHANGED
# (+1 x _profile_terminal mission fit, 1.145/1.224c).
# USER 2026-08-02: "losing at the last step should be more like -.3"
# -> KAPPA 0.7: a maximally-close loss floors at -1 + 0.7 = -0.30
_TERM_CLOSENESS = float(os.environ.get('WOTR_TERM_CLOSENESS', '0.7')
                        or 0.0)


def _closeness_to_victory(side_tag: str, profile,
                          final_state: 'GameState') -> float:
    """[0, 1] profile-pure closeness of `side_tag` to ITS OWN victory
    at `final_state`, in the convex _win_frac_phi currency.
    mixed / profiles-off blend with MAX semantics (closest axis)."""
    st = final_state
    prof = str(profile) if profile is not None else None
    try:
        vp = game_logic.calculate_victory_points(st)
        if side_tag == 'fp':
            try:
                ring_total = _gate_distance(st, 'rivendell') \
                    + _RING_TRACK_TAIL
            except Exception:
                ring_total = 13.0
            ring_c = _win_frac_phi(_ring_progress_metric(st), ring_total)
            mil_c = _win_frac_phi(vp['free_peoples'],
                                  game_logic.FP_VP_VICTORY)
            if prof == 'ring':
                return ring_c
            if prof == 'military':
                return mil_c
            return max(ring_c, mil_c)      # mixed / profiles off
        else:
            corr_c = _win_frac_phi(st.fellowship.corruption, 12.0)
            mil_c = _win_frac_phi(vp['shadow'],
                                  game_logic.SHADOW_VP_VICTORY)
            if prof == 'corruption':
                return corr_c
            if prof == 'military':
                return mil_c
            return max(corr_c, mil_c)      # mixed / profiles off
    except Exception:
        return 0.0


def _terminal_for(side_tag: str, profile, victory_type, won: bool,
                  final_state: 'GameState' = None) -> float:
    """THE terminal for every consumer (1.225 single implementation):
    episode outcome labels, td_value_update targets, and both MCTS
    search-terminal twins route through here — terminal semantics must
    never fork again (1.224c lesson).

    won  -> +1 scaled by _profile_terminal mission fit (unchanged).
    lost -> -1 + KAPPA * closeness (KAPPA = WOTR_TERM_CLOSENESS,
            default 0.7 -> best-case loss -0.30, USER 2026-08-02);
            final_state None (state unavailable at the call site)
            degrades to the old flat -1.0."""
    if won:
        return float(_profile_terminal(1.0, side_tag, profile,
                                       victory_type))
    if final_state is None or _TERM_CLOSENESS <= 0.0:
        return -1.0
    return -1.0 + _TERM_CLOSENESS * _closeness_to_victory(
        side_tag, profile, final_state)








def _auto_enter_mordor(state) -> bool:
    """USER RULE (2026-07-08): "we can hardcode a check to always enter
    mordor when you have enough movement to."  Self-play only (the
    teacher follows the logged declares).  When the fellowship is
    hidden, not yet in Mordor, and its progress reaches a gateway
    (Morannon / Minas Morgul), declare straight into the gateway.

    PERMANENT SANCTIONED HARDCODE (USER 2026-08-02, comment-archaeology
    review): "auto enter mordor will be one time we hardcode."  This
    deliberately pre-empts the declare-phase search and is NOT a
    revisit candidate — do not flag in future deferred-decision
    sweeps.

    USER RULING 1.155 (2026-07-17), timing superseded by the FINAL
    ruling (USER 2026-08-16): this helper DECLARES at the gateway
    only — "you can only enter at the start of the turn, either by
    declaration or by being revealed on morannon or minas morgul."
    Entry itself resolves in the SAME turn start's entry check
    (_check_mordor_entry), which the drivers run AFTER the
    declaration opportunity — never mid-turn, and never a full-turn
    wait.  The teacher path is unaffected: it mirrors the logged
    humans' own entry housekeeping."""
    f = state.fellowship
    if f.mordor_entered or not f.hidden or f.progress <= 0:
        return False
    try:
        cur = sanitize(game_logic.fellowship_region(state))
    except Exception:
        return False
    from collections import deque
    for gw in ('morannon', 'minas morgul'):
        if gw not in board.regions:
            continue
        # plain adjacency BFS: declare places the fellowship up to
        # `progress` regions from its current position
        seen, q, dist = {cur}, deque([(cur, 0)]), None
        while q:
            r, d = q.popleft()
            if r == gw:
                dist = d
                break
            if d >= f.progress:
                continue
            for adj in state.board.adjacent(r):
                a = sanitize(adj)
                if a not in seen:
                    seen.add(a)
                    q.append((a, d + 1))
        if dist is None or dist > f.progress:
            continue
        try:
            # DECLARE only (ruling 1.155) — the fellowship sits at the
            # gateway declared; _check_mordor_entry fires at the next
            # turn's Fellowship-phase pre-step.
            game_logic.declare_fellowship(state, region=gw)
            # 2026-07-20c: trace-visible (the w5_g2 T9 declare was a
            # silent unrecorded state jump attributed to the next
            # recorded action — the auditor and the user must SEE it)
            try:
                from game_trace import emit_annotation
                emit_annotation(state, 'auto_declare_gateway',
                                {'region': gw}, side='FP')
            except Exception:
                pass
            return True
        except Exception:
            return False
    return False




# ===================================================================
# Parallel Self-Play
# ===================================================================

def _game_stats_finalize(state, ep_meta) -> dict:
    """Compact end-of-game statistics for the TD reporting system
    (session 29, USER spec).  Queue-safe primitives only."""
    try:
        vp = game_logic.calculate_victory_points(state)
    except Exception:
        vp = {'free_peoples': 0, 'shadow': 0}
    f = state.fellowship
    # distance from the fellowship figure to the nearest Mordor gate
    # (negative = steps taken ON the Mordor track)
    if f.mordor_entered:
        mdist = -int(getattr(f, 'mordor_step', 0))
    else:
        try:
            from collections import deque
            cur = sanitize(game_logic.fellowship_region(state))
            seen, q = {cur}, deque([(cur, 0)])
            mdist = 99
            while q:
                r, d = q.popleft()
                if r in ('morannon', 'minas morgul'):
                    mdist = d
                    break
                for adj in state.board.adjacent(r):
                    a = sanitize(adj)
                    if a not in seen:
                        seen.add(a)
                        q.append((a, d + 1))
        except Exception:
            mdist = 99
    def _settls(side):
        out = []
        for r in state.enemy_controlled.get(side, set()):
            meta = board.regions.get(sanitize(str(r)), {})
            if meta.get('is_stronghold') or meta.get('is_city'):
                out.append(sanitize(str(r)))
        return sorted(out)
    return {
        'profiles': dict(PLAYER_PROFILES),
        'turns': ep_meta.get('turns', 0),
        'winner': ep_meta.get('winner'),
        'victory_type': ep_meta.get('victory_type'),
        'fp_vp': int(vp.get('free_peoples', 0)),
        'sh_vp': int(vp.get('shadow', 0)),
        'sh_captured': _settls(Side.SHADOW),
        'fp_captured': _settls(Side.FREE_PEOPLES),
        'aragorn_crowned': bool(getattr(
            getattr(state, 'aragorn_crowned', None), 'crowned', False)),
        'corruption': int(f.corruption),
        'mordor_entered': bool(f.mordor_entered),
        'mordor_dist': int(mdist),
        'separations': list(ep_meta.get('separations', [])),
        'eye_allocs': list(ep_meta.get('eye_allocs', [])),
        'trace_path': ep_meta.get('trace_path'),
    }


def _apply_axis_outcomes(state, winner, vtype, outcome_fp, outcome_sh,
                         turn_count):
    """WIN-AXIS terminal targets (USER, profiles v2/v3/1.145).

    military profiles: outcome = 2*clip(VP/target)-1 (VP only).
    ring / corruption profiles (1.145): +/-1 by win-loss, but an
    OFF-MISSION win pays 0.2*base (fp_ring's mission is ring_destroyed,
    sh_corruption's is ring_corruption); losses stay -1 in full.
    Shadow win-fast decay applies to positive SH outcomes.
    Timeout games (winner='timeout') keep their bootstrapped targets.
    """
    try:
        if PLAYER_PROFILES.get('fp') == 'military' \
                or PLAYER_PROFILES.get('sh') == 'military':
            _vp_ax = game_logic.calculate_victory_points(state)
            if PLAYER_PROFILES.get('fp') == 'military':
                outcome_fp = 2.0 * min(
                    _vp_ax.get('free_peoples', 0) / 4.0, 1.0) - 1.0
            if PLAYER_PROFILES.get('sh') == 'military':
                outcome_sh = 2.0 * min(
                    _vp_ax.get('shadow', 0) / 10.0, 1.0) - 1.0
        # 1.145: ring/corruption profiles — a WON off-mission terminal
        # is scaled to 0.2*base ('ring assaults Mordor' VP-grab leak,
        # w10_ep00023) without flipping to a loss (the run-15 -1.0
        # override is superseded).  Review P1 (2026-07-27): the W10 B1
        # comment claimed _profile_terminal "already" did this, but it
        # had ZERO call sites — the scaling must be applied HERE.
        # Winner-gated: timeout targets are bootstrapped (possibly
        # positive) truncation values, never "wins" to scale.
        if winner == 'fp' and outcome_fp is not None \
                and PLAYER_PROFILES.get('fp') not in (None, 'military'):
            outcome_fp = _profile_terminal(
                outcome_fp, 'fp', PLAYER_PROFILES.get('fp'), vtype)
        if winner == 'sh' and outcome_sh is not None \
                and PLAYER_PROFILES.get('sh') not in (None, 'military'):
            outcome_sh = _profile_terminal(
                outcome_sh, 'sh', PLAYER_PROFILES.get('sh'), vtype)
    except Exception:
        pass
    # 1.140 (USER): Shadow must win AS FAST AS POSSIBLE - positive SH
    # outcomes decay with game length (T8 win keeps full value, each
    # later turn costs 4%, floor 0.4).  Losses are NOT scaled.
    try:
        if PLAYER_PROFILES.get('sh') in ('corruption', 'military') \
                and outcome_sh is not None and outcome_sh > 0:
            outcome_sh *= max(0.4, 1.0 - 0.04 * max(0, turn_count - 8))
    except Exception:
        pass
    return outcome_fp, outcome_sh


# item 5 instrumentation (USER 2026-08-02): per-decision root search
# stats stored in traces (K<=40 root children each) — the raw material
# for the 1.226b delta targets and the regret view.
_STORE_SEARCH_STATS = os.environ.get('WOTR_STORE_SEARCH_STATS',
                                     '1') != '0'


def _search_stats_payload(visit_info):
    """(search_stats dict, regret float) for trace storage, or
    (None, None) when disabled / when the caller passed a bare prior
    array instead of the visit_info dict (no such path remains since
    1.226d retired the policy-first hunt picks; guard kept as a
    defensive contract).  Layout: 'ae' float16[K, ACTION_ENCODING_DIM],
    'ri' int16[K] (region indices — needed to REBUILD the scorer's
    concat path at training time), 'p' float32[K] clean priors,
    'q' float32[K] parent-perspective child Q, 'n' uint16[K] visits,
    'played' int index of the executed action."""
    if not _STORE_SEARCH_STATS or not isinstance(visit_info, dict) \
            or 'q_values' not in visit_info:
        return None, None
    try:
        ss = {'ae': np.asarray(visit_info['action_encs'],
                               dtype=np.float16),
              'ri': np.asarray(visit_info['region_indices'],
                               dtype=np.int16),
              'p': np.asarray(visit_info['priors_clean'],
                              dtype=np.float32),
              'q': np.asarray(visit_info['q_values'],
                              dtype=np.float32),
              'n': np.clip(np.asarray(visit_info['visit_counts']),
                           0, 65535).astype(np.uint16),
              'played': int(visit_info['played'])}
        # per-arm empirical variance (diagnostic only)
        if 'q_var' in visit_info:
            ss['v'] = np.asarray(visit_info['q_var'],
                                 dtype=np.float32)
        # resolved-terminal fraction (feeds the terminal-resolved
        # override in select_root_action)
        if 'term_frac' in visit_info:
            ss['tf'] = np.asarray(visit_info['term_frac'],
                                  dtype=np.float16)
        # solver proof marks (LAB 2026-08-09, P-A extension): stored
        # so TRAINING targets can reproduce the proof-forced pi the
        # play side used — proven-win one-hots are the exact,
        # terminal-grounded sharpening 1.230 sanctions, and they were
        # previously invisible to the trainer (payload lacked pv)
        if 'proven' in visit_info:
            ss['pv'] = np.asarray(visit_info['proven'],
                                  dtype=np.int8)
        # forced-playout counts (task 14 FIX 3): stored so the
        # trainer's visit_tf target applies the SAME earned-visits
        # discount the play side used
        if 'forced' in visit_info \
                and np.any(visit_info['forced']):
            ss['fn'] = np.clip(np.asarray(visit_info['forced']),
                               0, 65535).astype(np.uint16)
        # task-15 LAW 2: per-arm mean leaf confidence (stored only
        # when the wave is live — all-1.0 arrays teach nothing)
        if 'conf' in visit_info \
                and np.any(np.asarray(visit_info['conf']) < 1.0):
            ss['cf'] = np.asarray(visit_info['conf'],
                                  dtype=np.float16)
            # task-17 LAW 1R: per-arm scaffold means ride with cf
            if 'phi' in visit_info:
                ss['ph'] = np.asarray(visit_info['phi'],
                                      dtype=np.float16)
        return ss, float(visit_info['regret'])
    except Exception:
        return None, None


def _run_single_episode(fp_net, sh_net, config: dict) -> dict:
    """Run a single self-play episode and return trajectory data.

    Args:
        fp_net, sh_net: Value networks (on CPU for workers, GPU for main)
        config: dict with keys: max_turns, mcts_sims, shadow_vp

    Returns dict with:
        states, sides, rewards, visit_dists, action_encs,
        outcome_fp, turns, winner, victory_type
    """
    _ep_stats = {'eye_allocs': [], 'separations': []}
    # leak hygiene (USER 2026-07-30 hunt): the single-shot tree-reuse
    # cache must NEVER survive an episode boundary — a stale subtree
    # carries the PREVIOUS game's profile-valued Q estimates (the sig
    # guard makes a cross-game hit near-impossible, but the class is
    # removed outright here)
    _TREE_CACHE['child'] = None
    _TREE_CACHE['sig'] = None
    # 1.142: per-side interval anchors - each side's step reward
    # covers everything since its OWN previous step, so opponent-step
    # events (captures, hunt corruption, progress) finally reach the
    # side that must react to them
    _last_post = {}
    # 1.143: per-side bank of military reward terms, paid only on the
    # side's next military-class action (structural isolation)
    _mil_bank = {}
    b = board.load_board()
    state = gs.setup_initial_game(b)
    game_logic.initialize_decks(state)
    # Turn 1 draw phase: each player draws 1 Character + 1 Strategy card
    # (next_turn handles turns 2+; without this, turn 1 was played
    # cardless — human logs show both players drawing at turn 1 start)
    for _side in (Side.FREE_PEOPLES, Side.SHADOW):
        for _deck in ('character', 'strategy'):
            try:
                game_logic.draw_event_card(state, _side, _deck)
            except Exception:
                pass

    max_turns = config.get('max_turns', 100)
    mcts_sims = config.get('mcts_sims', 400)
    shadow_vp = config.get('shadow_vp', 10)
    bonus_sims = config.get('bonus_sims', 0)

    # Optional game trace recording (replayable in the GUI)
    tracer = None
    if config.get('trace_path'):
        from game_trace import TraceRecorder
        tracer = TraceRecorder(config['trace_path'])
        tracer.record_initial(state)

    # Override shadow VP threshold
    _orig_vp = game_logic.SHADOW_VP_VICTORY
    game_logic.SHADOW_VP_VICTORY = shadow_vp

    # Optional: Pre-advance nations (disabled by default)
    if config.get('pre_advance_nations', False):
        for n in (gs.Nation.SAURON, gs.Nation.ISENGARD, gs.Nation.SOUTHRON_EASTERLINGS):
            while state.political.boxes[n] < 3:
                try:
                    game_logic.advance_political(state, n, method="attack")
                except Exception:
                    state.political.boxes[n] = 3
                    state.political.active[n] = True
                    break
        for n in (gs.Nation.GONDOR, gs.Nation.ROHAN, gs.Nation.THE_NORTH):
            while state.political.boxes[n] < 3:
                try:
                    game_logic.advance_political(state, n, method="attack")
                except Exception:
                    state.political.boxes[n] = 3
                    state.political.active[n] = True
                    break

    # Optional: Reinforce Shadow armies (disabled by default)
    if config.get('reinforce_armies', False):
        for rkey, (extra_r, extra_e) in {
            'Minas Morgul': (4, 1), 'Dol Guldur': (3, 1),
            'Orthanc': (3, 0), 'Far Harad': (3, 0), 'Morannon': (3, 0),
        }.items():
            army = _armies_lookup(state, rkey)
            if army:
                for _ in range(extra_r):
                    army.units.append(gs.Unit(army.units[0].nation, gs.UnitKind.REGULAR))
                for _ in range(extra_e):
                    army.units.append(gs.Unit(army.units[0].nation, gs.UnitKind.ELITE))

    trajectory = []  # (state_vec, side, reward, visit_dist, action_enc)
    game_over = False
    turn_count = 0
    hot_regions = set()  # Regions affected by previous action (for focused MCTS)

    # Arena/evaluation overrides: fixed low temperature, no root noise
    _temp_override = config.get('temperature_override')
    _use_noise = config.get('add_noise', True)

    while not game_over and turn_count < max_turns:
        turn_count += 1
        # Symmetric temperature schedule
        if turn_count <= 3:
            fp_temp = sh_temp = 0.6
        elif turn_count <= 8:
            fp_temp = sh_temp = 0.3
        else:
            fp_temp = sh_temp = 0.15
        if _temp_override is not None:
            fp_temp = sh_temp = _temp_override

        # Phase 1b: hand-limit discards (learned decision) — resolve
        # before the turn's other phases; each iteration discards 1 card
        _d_guard = 0
        while (getattr(state, 'pending_discard', None) is not None
               and _d_guard < 12):
            _d_guard += 1
            d_side = state.pending_discard['side']
            d_temp = fp_temp if d_side == Side.FREE_PEOPLES else sh_temp
            d_action, d_visit = mcts_action_with_policy(
                state, d_side, fp_net, sh_net,
                num_simulations=max(20, mcts_sims // 4),
                temperature=d_temp, add_noise=_use_noise)
            if d_action is None or d_action.action_type != 'discard_card':
                # Safety valve — never stall the game on a stuck phase.
                # DETERMINISTIC and conserving (2026-08-15): this used
                # to `random.choice` the card and drop it into NO pile,
                # which both annihilated it and made the episode depend
                # on the global RNG stream. It fires only when MCTS
                # failed to return a discard at all, so it is never the
                # decision path — but it must still be a legal discard.
                _h = state.hands[d_side].hand
                if len(_h) > game_logic.HAND_LIMIT:
                    game_logic.discard_hand_card(state, d_side, _h[-1])
                else:
                    state.pending_discard = None
                continue
            vec_d = encode_state(state, d_side)
            enc_d = encode_action(d_action, state, d_side)
            try:
                _traced_execute(state, d_action, tracer)
            except Exception:
                state.pending_discard = None
                break
            trajectory.append((vec_d, d_side, 0.0, d_visit, enc_d))
            if tracer:
                _ss9d, _rg9d = _search_stats_payload(d_visit)
                tracer.record_action(state, d_side, d_action,
                                     search_stats=_ss9d, regret=_rg9d)

        # Phase 2 pre-step.  FINAL RULING (USER 2026-08-16): Mordor
        # entry is TURN-START ONLY and POSITIONAL, and the DECLARATION
        # OPPORTUNITY COMES FIRST - the entry check now runs AFTER the
        # Fellowship phase below (the old order, entry check first,
        # was the day-one bug: a fellowship declared onto a gate
        # waited a full turn to enter).
        if not state.fellowship.mordor_entered:
            # Balrog of Moria STATIONARY trigger (almanac C17, second
            # bullet): the Fellowship remaining stationary IN Moria at
            # the start of a turn also lets the Shadow discard the
            # table card to draw a Hunt tile.  Same pending shape and
            # 'balrog_declare_draw' handler branch as the declaration
            # trigger; it resolves in the main action loop below, i.e.
            # AFTER the FP Fellowship phase (declaration first, per the
            # almanac).  LEARNED PATH ONLY (the helper self-gates on
            # combat.FINAL_HITS_HOOK is None) so the certified teacher
            # never stalls on a fresh interactive pending.  The trigger
            # check itself stays BEFORE the declaration (stationary
            # means no declaration away has happened yet).
            _balrog_stationary_turn_start(state)
        # USER RULING 2026-07-20c: declarations are TURN-START ONLY —
        # the gateway auto-declare fires HERE (Fellowship phase), never
        # mid-turn (w5_g2 T9 silent declare removed).
        _auto_declared9 = False
        if not state.fellowship.mordor_entered:
            _auto_declared9 = _auto_enter_mordor(state)
        # Phase 2: Fellowship Phase — FP may declare fellowship position
        if not _auto_declared9 and state.fellowship.progress >= 0 and not state.fellowship.mordor_entered:  # iter 39: in-place declare (entry/heal) legal at progress 0
            declare_actions = _get_declare_actions(state, allow_in_place=True)
            if declare_actions:
                state._declare_phase = True
                try:
                    # USER 2026-07-18 (Ithilien/Dunland declare
                    # ping-pong): NO root noise on the declare pick —
                    # declare roots are value-flat (every child ~= same
                    # eval), so Dirichlet spikes + early-turn
                    # temperature sampled 1-step back-and-forth
                    # declares instead of the gateway push the prior
                    # already encodes (v22: minas morgul 0.49 /
                    # morannon 0.26 from north ithilien).
                    # 1.142c (USER 2026-08-02 'declaring at rivendell
                    # and resetting progress for no reason'): NOISE-
                    # FREE means temperature too.  add_noise=False
                    # killed the Dirichlet spikes but fp_temp (0.6
                    # turns 1-3) kept SAMPLING the value-flat declare
                    # children — 3.61 declares/game vs human 1.9, 192
                    # progress-burning rivendell declares in 8 batches,
                    # all at t2-t3, reward-negative every one.  Argmax.
                    dec_action, dec_visit = mcts_action_with_policy(
                        state, Side.FREE_PEOPLES, fp_net, sh_net,
                        num_simulations=max(30, mcts_sims // 3),
                        temperature=0.0, add_noise=False)
                finally:
                    state._declare_phase = False
                if dec_action and dec_action.action_type == 'fellowship_declare':
                    vec = encode_state(state, Side.FREE_PEOPLES)
                    enc = encode_action(dec_action, state, Side.FREE_PEOPLES)
                    try:
                        _traced_execute(state, dec_action, tracer)
                    except Exception:
                        pass
                    trajectory.append((vec, Side.FREE_PEOPLES, 0.0, dec_visit, enc))
                    if tracer:
                        _ss9e, _rg9e = _search_stats_payload(dec_visit)
                        tracer.record_action(state, Side.FREE_PEOPLES,
                                             dec_action,
                                             search_stats=_ss9e,
                                             regret=_rg9e)

        # Mordor entry check - AFTER the declaration opportunity, same
        # turn start (USER FINAL 2026-08-16): a gate position reached
        # by a fresh declaration (auto or learned), standing from an
        # earlier one, or by a REVEAL enters NOW.  _check_mordor_entry
        # is positional and idempotent - the single answerer for all
        # three triggers.
        if not state.fellowship.mordor_entered:
            game_logic._check_mordor_entry(state)

        # Shadow eye allocation via MCTS (before dice roll per WOTR rules)
        shadow_unused = [d for d in state.dice.get(Side.SHADOW, [])
                         if not d.used]
        if len(shadow_unused) > 0:
            state._eye_allocation_phase = True
            try:
                # 1.142 REVISED (USER 2026-07-18, baseline_w9_g3 4-eyes
                # T1): NO root noise on the eye-allocation decision AT
                # ALL — the old split (noisy pick, clean re-searched
                # target) still EXECUTED noise-spiked counts.  One clean
                # search now serves both the played action and the
                # policy target.
                # 1.142c: temperature=0 — same fix as the declare
                # pick above; sh_temp sampling was the residual noise
                # channel 1.142's add_noise=False left open
                eye_action, eye_visit = mcts_action_with_policy(
                    state, Side.SHADOW, fp_net, sh_net,
                    num_simulations=mcts_sims,
                    temperature=0.0, add_noise=False)
            finally:
                state._eye_allocation_phase = False
            if eye_action is not None \
                    and eye_action.action_type != 'eye_allocate':
                # never silent (2026-07-27): a non-eye result here was
                # the tree-reuse phase-flag hijack — sig now separates
                # phase states, so any recurrence is a regression
                try:
                    import state_invariants as _si9e
                    _si9e.VIOLATIONS.append({
                        'code': 'eye_phase_hijacked',
                        'ctx': 'T%s' % getattr(state, 'turn', '?'),
                        'detail': 'eye-phase search returned %s'
                                  % eye_action.action_type})
                except Exception:
                    pass
            if eye_action and eye_action.action_type == 'eye_allocate':
                n_eyes = eye_action.params.get('count', 0)
                _ep_stats['eye_allocs'].append(int(n_eyes))
                state_vec = encode_state(state, Side.SHADOW)
                action_enc = encode_action(eye_action, state, Side.SHADOW)
                if n_eyes > 0:
                    indices = [i for i, d in enumerate(state.dice[Side.SHADOW])
                               if not d.used][:n_eyes]
                    game_logic.allocate_eyes(state, n_eyes, indices=indices)
                trajectory.append((state_vec, Side.SHADOW, 0.0,
                                   eye_visit, action_enc))
                if tracer:
                    _ss9f, _rg9f = _search_stats_payload(eye_visit)
                    tracer.record_action(state, Side.SHADOW, eye_action,
                                         search_stats=_ss9f,
                                         regret=_rg9f)

        game_logic.roll_action_dice(state, pool=Side.FREE_PEOPLES)
        game_logic.roll_action_dice(state, pool=Side.SHADOW)

        def _unused(side):
            return sum(1 for d in state.dice.get(side, [])
                       if not d.used and d.current is not None)

        active = Side.FREE_PEOPLES
        # F-3 (USER 2026-07-19): 40 was a per-turn DECISION cap, but
        # battle sub-decisions (cards, allocations, retreats) burn
        # iterations too — a long siege turn hit the cap mid-battle
        # and _safe_next_turn abandoned the fight AFTER the garrison
        # died but BEFORE capture ran (fuzz seed 9133, Lorien). 200
        # keeps the runaway-loop valve far above any legal turn.
        for _ in range(200):
            # Check for pending multi-phase states first
            has_pending = (state.pending_combat is not None or
                           state.pending_hunt is not None or
                           state.pending_guide_change is not None or
                           getattr(state, 'pending_card_option',
                                   None) is not None or
                           getattr(state, 'pending_discard',
                                   None) is not None or
                           getattr(state, 'pending_card_followup',
                                   None) is not None or
                           # USER RULING 2026-08-16: open reveal
                           # placement is an FP pending
                           getattr(state, '_reveal_place_due', False))

            fp_left = _unused(Side.FREE_PEOPLES)
            sh_left = _unused(Side.SHADOW)
            if fp_left == 0 and sh_left == 0 and not has_pending:
                break

            if not has_pending:
                if active == Side.FREE_PEOPLES and fp_left == 0:
                    active = Side.SHADOW
                elif active == Side.SHADOW and sh_left == 0:
                    active = Side.FREE_PEOPLES
                my_left = _unused(active)
                if my_left == 0:
                    break

            # pre-ask routing safety via the SHARED cascade (v13 crash:
            # this override forced active to the card-option side while
            # an FP hand-limit discard was open)
            _ps0 = _pending_decision_side(state)
            if _ps0 is not None:
                active = _ps0
            # USER RULING 2026-07-20c: declarations are TURN-START ONLY
            # — mid-turn _auto_enter_mordor removed (w5_g2 T9); it now
            # fires in the Phase-2 Fellowship pre-step.
            actions = get_legal_actions(state, active)
            if not actions:
                break

            temp = fp_temp if active == Side.FREE_PEOPLES else sh_temp
            visit_dist = None
            action, visit_dist = mcts_action_with_policy(
                state, active, fp_net, sh_net,
                num_simulations=mcts_sims, temperature=temp, add_noise=_use_noise,
                hot_regions=hot_regions, bonus_sims=bonus_sims)
            if action is None:
                break

            other = (Side.SHADOW if active == Side.FREE_PEOPLES
                     else Side.FREE_PEOPLES)

            pre = _last_post.get(active)
            if pre is None:
                pre = _reward_stats(state)

            # 1.143: classify before execution - the die is consumed
            _die_face9 = None
            try:
                if action.die_idx is not None and action.die_idx >= 0:
                    _cur9 = state.dice[active][action.die_idx].current
                    _die_face9 = _cur9.value if _cur9 is not None else None
            except Exception:
                _die_face9 = None
            _mil9 = _is_military_action(action.action_type,
                                        action.params, _die_face9)

            state_vec = encode_state(state, active)
            action_enc = encode_action(action, state, active)
            # review P2 (2026-07-27): capture the hunt-place phase
            # BEFORE execution — the hobbit_place executor clears
            # pending_hunt, so a post-execute read missed the common
            # single-hobbit case and mislabeled it a chosen separation
            _hunt_place9 = bool(
                state.pending_hunt is not None
                and (state.pending_hunt or {}).get('phase')
                == 'hobbit_place')
            _exec_failed9 = False
            try:
                _traced_execute(state, action, tracer)
            except Exception as _exec_err:
                # Rare enumeration/execution mismatch (e.g. stacking after
                # mid-combat army changes). Recover by consuming the die /
                # clearing the stuck phase rather than killing the episode
                # — a raise here used to discard the whole game's data.
                # Never silent: every recovery is a HARD invariant flag.
                _record_exec_failure(state, action, _exec_err)
                _recover_from_failed_action(state, action, active)
                action = Action(die_idx=action.die_idx,
                                action_type='pass', params={})
                action_enc = encode_action(action, state, active)
                # the substituted pass no longer matches the search's
                # 'played' index — drop the stats for this row
                _exec_failed9 = True

            # Update hot zone for next MCTS call
            if bonus_sims > 0:
                hot_regions = _get_hot_regions(action, state)

            _axis9, _milr9 = _compute_step_reward_split(pre, state,
                                                        active)
            # 1.143 (USER): military deltas are banked and paid only on
            # this side's next military-class action.  1.143b flush:
            # residue at game end is NO LONGER dropped — it lands on
            # the side's LAST sample after the game loop (a stronghold
            # captured after the side's last military action used to
            # leave the -VP nowhere).
            _mil_bank[active] = _mil_bank.get(active, 0.0) + _milr9
            step_reward = _axis9
            if _mil9:
                step_reward += _mil_bank[active]
                _mil_bank[active] = 0.0
            _last_post[active] = _reward_stats(state)
            trajectory.append((state_vec, active, step_reward, visit_dist, action_enc))
            if action.action_type in ('companion_separate',
                                      'hunt_guide_separate'):
                # stats fix (USER 2026-07-26): label the three distinct
                # mechanics — a hunt-casualty hobbit PLACEMENT is not a
                # chosen separation, and group separations list all names
                _grp9s = action.params.get('companions')
                if _grp9s:
                    _ep_stats['separations'].extend(str(x) for x in _grp9s)
                else:
                    _nm9 = str(action.params.get('companion', '') or '?')
                    if _hunt_place9:
                        _nm9 += ' (hunt-placed)'
                    _ep_stats['separations'].append(_nm9)
            elif action.action_type in ('play_event', 'card_option'):
                # card-driven separations were invisible in the report
                # (I Will Go Alone / Breaking of the Fellowship) —
                # USER caught a crowned-Aragorn game showing no Strider
                # separation because the card did it
                _cn9 = str(action.params.get('card_name', '') or '')
                _ch9 = (action.params.get('choices') or {})
                if 'I Will Go Alone' in _cn9 or 'Breaking' in _cn9:
                    for _c9 in (_ch9.get('companions') or ['?']):
                        _ep_stats['separations'].append(
                            '%s (card: %s)' % (_c9, _cn9 or 'separation'))
            if tracer:
                _ss9m, _rg9m = ((None, None) if _exec_failed9
                                else _search_stats_payload(visit_dist))
                tracer.record_action(state, active, action,
                                     search_stats=_ss9m, regret=_rg9m)

            vc = game_logic.check_all_victory_conditions(state)
            # Ring victories end the game immediately; military victory is
            # only checked in the end-of-turn Victory Check (rulebook p.17).
            if game_logic.should_game_end(vc, end_of_turn=False):
                game_over = True
                break

            if action.action_type == 'elven_ring':
                pass  # ring use is NOT an action — same side acts
            elif (state.pending_army_move_die is not None or
                    state.pending_muster_die is not None):
                pass  # Same side continues (multi-step die action)
            else:
                # single source of truth for pending-decision routing
                # (v13 crash: inlined cascades drifted)
                _ps9 = _pending_decision_side(state, fallback=active)
                active = (_ps9 if _ps9 is not None
                          else _post_action_active(state, active, other))

        if game_over:
            break
        # Victory Check phase: military victory at end of turn only
        _end_vc = game_logic.check_all_victory_conditions(state)
        if game_logic.should_game_end(_end_vc, end_of_turn=True):
            game_over = True
            break
        if _safe_next_turn(state):
            # P0 2026-08-15: rollover corruption hit 12 — immediate
            # ring victory; never play into a won game
            game_over = True
            if tracer:
                tracer.record_new_turn(state)
            break
        if tracer:
            tracer.record_new_turn(state)

    # 1.143b TERMINAL FLUSH: residual military bank lands on the
    # side's LAST emitted sample (mirrors run_td_v33loop's
    # _replay_reward_stream flush — the two live reward-accumulation
    # paths must stay twins)
    for _fl_side9, _fl_res9 in list(_mil_bank.items()):
        if abs(_fl_res9) > 1e-9:
            for _fl_i9 in range(len(trajectory) - 1, -1, -1):
                if trajectory[_fl_i9][1] == _fl_side9:
                    _t9 = trajectory[_fl_i9]
                    trajectory[_fl_i9] = (_t9[0], _t9[1],
                                          _t9[2] + _fl_res9,
                                          _t9[3], _t9[4])
                    _mil_bank[_fl_side9] = 0.0
                    break

    # Determine outcome
    result = game_logic.check_all_victory_conditions(state)
    outcome_sh = None  # Shadow terminal target; None => use -outcome_fp
    if result['victory'] in ('free_peoples', 'shadow'):
        winner = 'fp' if result['victory'] == 'free_peoples' else 'sh'
        vtype = result.get('victory_type', 'ring')
        # 1.225: BOTH outcome labels route through _terminal_for (the
        # single terminal implementation) — winner keeps the 1.145
        # mission-fit scaling, loser gets the graded closeness
        # terminal.  Computed BEFORE the SHADOW_VP_VICTORY restore so
        # closeness uses the game's actual threshold.
        outcome_fp = _terminal_for('fp', PLAYER_PROFILES.get('fp'),
                                   vtype, winner == 'fp', state)
        outcome_sh = _terminal_for('sh', PLAYER_PROFILES.get('sh'),
                                   vtype, winner == 'sh', state)
        # 1.140 (USER): Shadow win-fast decay kept — positive SH
        # outcomes only (was inside _apply_axis_outcomes, which now
        # serves the timeout path alone)
        try:
            if PLAYER_PROFILES.get('sh') in ('corruption', 'military') \
                    and outcome_sh > 0:
                outcome_sh *= max(0.4, 1.0 - 0.04
                                  * max(0, turn_count - 8))
        except Exception:
            pass
    else:
        # Timeout = TRUNCATION, not termination. The turn cap is a
        # training artifact — real WotR games always finish — so the
        # final position is a censored observation, not a draw. Labeling
        # it 0.0 taught "8 VP at the cap is worthless" (no credit for
        # almost winning); a hand-crafted VP formula would re-poison the
        # labels. Standard RL treatment: bootstrap the terminal target
        # from each net's OWN estimate of the final position, clamped
        # away from false certainty.
        winner = 'timeout'
        vtype = 'timeout'
        outcome_fp = 0.0
        try:
            with torch.no_grad():
                xf = torch.from_numpy(np.asarray(
                    [encode_state(state, Side.FREE_PEOPLES)],
                    dtype=np.float32)).to(DEVICE)
                xs = torch.from_numpy(np.asarray(
                    [encode_state(state, Side.SHADOW)],
                    dtype=np.float32)).to(DEVICE)
                outcome_fp = float(np.clip(fp_net(xf).item(), -0.9, 0.9))
                outcome_sh = float(np.clip(sh_net(xs).item(), -0.9, 0.9))
        except Exception:
            outcome_fp, outcome_sh = 0.0, 0.0

    game_logic.SHADOW_VP_VICTORY = _orig_vp

    if tracer:
        try:
            # Make the outcome visible in the filename: ep00042.trace.gz
            # -> ep00042_sh_military.trace.gz
            base = tracer.path
            if base.endswith('.trace.gz'):
                tracer.path = base[:-len('.trace.gz')] + f'_{winner}_{vtype}.trace.gz'
            tracer.save(winner=winner, victory_type=vtype, turns=turn_count,
                        mcts_sims=mcts_sims, max_turns=max_turns,
                        outcome_fp=outcome_fp)
        except Exception as e:
            print(f"[trace] failed to save {tracer.path}: {e}")
        finally:
            import game_trace as _gt
            _gt.ACTIVE_TRACER = None

    # Package trajectory as numpy arrays
    if trajectory:
        states_arr = np.array([t[0] for t in trajectory], dtype=np.float32)
        sides_arr = np.array([0 if t[1] == Side.FREE_PEOPLES else 1
                              for t in trajectory], dtype=np.int8)
        rewards_arr = np.array([t[2] for t in trajectory], dtype=np.float32)
        visit_dists = [t[3] for t in trajectory]
        action_encs = np.array([t[4] for t in trajectory], dtype=np.float32)
    else:
        states_arr = np.zeros((0, get_feature_dim()), dtype=np.float32)
        sides_arr = np.zeros(0, dtype=np.int8)
        rewards_arr = np.zeros(0, dtype=np.float32)
        visit_dists = []
        action_encs = np.zeros((0, ACTION_ENCODING_DIM), dtype=np.float32)

    # --- WIN-AXIS reward alignment (USER, session 29 profiles v2) ---
    # military profiles train on VICTORY POINTS only: outcome =
    # 2*clip(vp/target) - 1 (an off-axis win - e.g. SH corruption
    # victory under SH-military - earns whatever its VP say, i.e. a
    # loss-grade target at low VP).  ring/corruption profiles keep the
    # standard +/-1 outcome.  Search shaping (_profile_shaping) is
    # unchanged; this only changes the TRAINING target.
    #
    # v3.1 BUG FIX (runs 5-7, user-diagnosed): outcomes must be
    # DECOUPLED before any axis override.  The trainer derived
    # outcome_sh = -outcome_fp when unset; the FP-military override
    # (-1.0 for a 0-VP ring win) handed the SHADOW net +1.0 FOR A GAME
    # IT LOST - and those are exactly the hunt races, so hunt-heavy
    # states trained as wins for EVERY SH net (29%/23% of
    # sh_corruption/sh_military games in run 7).  outcome_sh is now
    # always explicit from the RAW result; overrides apply per side.
    if outcome_sh is None:
        outcome_sh = -outcome_fp
    # 1.225: decided games are fully labeled by _terminal_for above;
    # _apply_axis_outcomes now serves TRUNCATIONS only (military-axis
    # VP measure over the bootstrap + SH decay on the censored tail)
    if winner == 'timeout':
        outcome_fp, outcome_sh = _apply_axis_outcomes(
            state, winner, vtype, outcome_fp, outcome_sh, turn_count)

    return {
        'states': states_arr,
        'sides': sides_arr,
        'rewards': rewards_arr,
        'visit_dists': visit_dists,
        'action_encs': action_encs,
        'outcome_fp': outcome_fp,
        # Shadow terminal target: None for decided games (use -outcome_fp);
        # own-net bootstrap for truncated games
        'outcome_sh': outcome_sh,
        'turns': turn_count,
        'winner': winner,
        'victory_type': vtype,
        'game_stats': _game_stats_finalize(state, {
            'turns': turn_count, 'winner': winner,
            'victory_type': vtype,
            'separations': _ep_stats['separations'],
            'eye_allocs': _ep_stats['eye_allocs'],
            'trace_path': config.get('trace_path'),
        }),
    }




def resolve_trace_path(tp):
    """The tracer appends the outcome to the filename on save - find
    the real file for a configured trace path."""
    import glob as _glob
    tp = str(tp or '')
    if not tp:
        return None
    if os.path.exists(tp):
        return tp
    # newest match, NOT alphabetical: stale traces from earlier runs
    # shared the same w{N}_ep{M} base and hits[0] linked episode stats
    # to another run's game (USER report 2026-07-15, run 17 ep 7)
    hits = _glob.glob(tp.replace('.trace.gz', '') + '*.trace.gz')
    return max(hits, key=os.path.getmtime) if hits else None


def append_game_log(games_log_path, game_stats, batch, episode=None):
    """One JSONL line per game for the interactive TD dashboard."""
    import json as _json
    rec = dict(game_stats)
    rec['batch'] = batch
    if episode is not None:
        rec['episode'] = int(episode)
    rec['trace_path'] = resolve_trace_path(rec.get('trace_path'))
    try:
        with open(games_log_path, 'a', encoding='utf-8') as f:
            f.write(_json.dumps(rec) + '\n')
    except Exception:
        pass


def write_td_window_report(window_stats, label, report_path,
                           save_dir=None, rng=None):
    """USER spec (session 29): per-50-game report - each statistic
    broken down by profile - plus ONE random saved game per window
    (annotated with its profile pair).  Appends markdown to
    report_path; returns the saved-game path (or None)."""
    import json as _json
    import random as _rnd
    import shutil as _sh
    rng = rng or _rnd
    lines = ['', '## %s  (n=%d games)' % (label, len(window_stats)), '']
    if not window_stats:
        lines.append('(no games)')
    groups = []
    for side_key, profs in (('fp', FP_PROFILES), ('sh', SH_PROFILES)):
        for prof in profs:
            sel = [g for g in window_stats
                   if (g.get('profiles') or {}).get(side_key) == prof]
            groups.append(('%s_%s' % (side_key.upper(), prof), sel))
    groups.append(('ALL', list(window_stats)))

    def _avg(sel, fn):
        vals = [fn(g) for g in sel]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    lines.append('| profile | n | fp-w/sh-w/to | turns | SH VP | FP VP | '
                 'crowned% | corr | mordor-dist | entered% | eyes/turn |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for name, sel in groups:
        if not sel:
            lines.append('| %s | 0 | - | - | - | - | - | - | - | - | - |'
                         % name)
            continue
        wf = sum(1 for g in sel if g.get('winner') == 'fp')
        ws = sum(1 for g in sel if g.get('winner') == 'sh')
        wt = len(sel) - wf - ws
        lines.append(
            '| %s | %d | %d/%d/%d | %.1f | %.2f | %.2f | %.0f%% | %.2f |'
            ' %.1f | %.0f%% | %.2f |' % (
                name, len(sel), wf, ws, wt,
                _avg(sel, lambda g: g.get('turns', 0)),
                _avg(sel, lambda g: g.get('sh_vp', 0)),
                _avg(sel, lambda g: g.get('fp_vp', 0)),
                100 * _avg(sel, lambda g: 1.0 if g.get('aragorn_crowned')
                           else 0.0),
                _avg(sel, lambda g: g.get('corruption', 0)),
                _avg(sel, lambda g: g.get('mordor_dist', 99)),
                100 * _avg(sel, lambda g: 1.0 if g.get('mordor_entered')
                           else 0.0),
                _avg(sel, lambda g: (sum(g.get('eye_allocs') or [0])
                                     / max(1, len(g.get('eye_allocs')
                                                  or [0])))),
            ))
    # strongholds captured (count of games each settlement ended captured)
    from collections import Counter as _C
    cap_sh = _C()
    cap_fp = _C()
    seps = _C()
    for g in window_stats:
        for r in g.get('sh_captured') or []:
            cap_sh[r] += 1
        for r in g.get('fp_captured') or []:
            cap_fp[r] += 1
        for n in g.get('separations') or []:
            seps[n] += 1
    lines.append('')
    lines.append('- Shadow-captured settlements (games ended held): '
                 + (', '.join('%s x%d' % kv
                              for kv in cap_sh.most_common()) or 'none'))
    lines.append('- FP-captured settlements: '
                 + (', '.join('%s x%d' % kv
                              for kv in cap_fp.most_common()) or 'none'))
    lines.append('- Companions separated: '
                 + (', '.join('%s x%d' % kv
                              for kv in seps.most_common()) or 'none'))
    saved = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cands = [(g, resolve_trace_path(g.get('trace_path')))
                 for g in window_stats]
        cands = [(g, t) for g, t in cands if t]
        if cands:
            pick, pick_trace = rng.choice(cands)
            pp = pick.get('profiles') or {}
            base = '%s_fp-%s_sh-%s.trace.gz' % (
                label.replace(' ', '_').replace('@', 'ep'),
                pp.get('fp') or 'none', pp.get('sh') or 'none')
            saved = os.path.join(save_dir, base)
            try:
                _sh.copyfile(pick_trace, saved)
                lines.append('- Saved game: %s  (FP profile=%s, '
                             'SH profile=%s, winner=%s %s)' % (
                                 base, pp.get('fp'), pp.get('sh'),
                                 pick.get('winner'),
                                 pick.get('victory_type')))
            except Exception as e:
                lines.append('- Saved game: copy failed (%s)' % e)
                saved = None
    txt = '\n'.join(lines) + '\n'
    try:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(txt)
    except Exception:
        pass
    print(txt, flush=True)
    return saved








# ===================================================================
# CLI Interface
# ===================================================================

def main():
    """CLI: engine/AI self-checks + the GUI-facing self-play entry.
    (The legacy extract/train/export-csv/batch-export/evaluate
    subcommands were removed 2026-07-27 — extraction lives in
    rendered_extract.py, training in train_corpus.py, self-play runs
    in run_td_v*.py.)"""
    import argparse
    parser = argparse.ArgumentParser(description='WOTR AI core')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('test-clone', help='Test state cloning')
    sub.add_parser('test-encode', help='Test state encoding')
    sub.add_parser('test-legal', help='Test legal action enumeration')
    sub.add_parser('watch',
                   help='Watch AI vs AI game in GUI (step with Space)')

    # ('selfplay' subcommand deleted 2026-07-27 with the
    #  td_self_play twin — production runs use run_td_v33sp.py)

    args = parser.parse_args()

    if args.command == 'test-clone':
        _test_clone()
    elif args.command == 'test-encode':
        _test_encode()
    elif args.command == 'test-legal':
        _test_legal_actions()
    elif args.command == 'watch':
        b = board.load_board()
        state = gs.setup_initial_game(b)
        from gui import launch_gui_selfplay
        launch_gui_selfplay(state, b)
    else:
        parser.print_help()


def _test_clone():
    """Test state cloning correctness and speed."""
    print("=== Test: State Cloning ===")
    b = board.load_board()
    state = gs.setup_initial_game(b)

    # Correctness
    s2 = clone_state(state)
    assert s2.turn == state.turn
    assert s2.vp.free_peoples == state.vp.free_peoples
    assert s2.board is state.board, "Board should be shared"

    # Mutate original
    state.turn = 99
    state.vp.shadow = 10
    assert s2.turn == 1, "Clone should not be affected by original mutation"
    assert s2.vp.shadow == 0, "Clone VP should be unchanged"
    state.turn = 1
    state.vp.shadow = 0

    # Army mutation
    s2 = clone_state(state)
    first_key = next(iter(state.armies))
    state.armies[first_key].units.append(Unit(Nation.SAURON, UnitKind.REGULAR))
    assert len(s2.armies[first_key].units) != len(state.armies[first_key].units), \
        "Clone armies should be independent"
    state.armies[first_key].units.pop()

    print("Correctness: PASSED")

    # Speed
    n = 1000
    t0 = time.time()
    for _ in range(n):
        clone_state(state)
    elapsed = time.time() - t0
    print(f"Speed: {n} clones in {elapsed:.2f}s ({n/elapsed:.0f} clones/sec)")
    print()


def _test_encode():
    """Test state encoding."""
    print("=== Test: State Encoding ===")
    b = board.load_board()
    state = gs.setup_initial_game(b)

    vec_fp = encode_state(state, Side.FREE_PEOPLES)
    vec_sh = encode_state(state, Side.SHADOW)

    print(f"Feature dim: {len(vec_fp)}")
    assert len(vec_fp) == len(vec_sh), "Feature dims should match"
    assert not np.any(np.isnan(vec_fp)), "No NaN in FP encoding"
    assert not np.any(np.isnan(vec_sh)), "No NaN in Shadow encoding"
    assert not np.any(np.isinf(vec_fp)), "No inf in FP encoding"

    # Check that perspectives differ
    assert not np.allclose(vec_fp, vec_sh), "FP and Shadow encodings should differ"

    print(f"FP vec range: [{vec_fp.min():.3f}, {vec_fp.max():.3f}]")
    print(f"SH vec range: [{vec_sh.min():.3f}, {vec_sh.max():.3f}]")
    print("Encoding: PASSED")
    print()


def _test_legal_actions():
    """Test legal action enumeration on initial state with dice."""
    print("=== Test: Legal Action Enumeration ===")
    b = board.load_board()
    state = gs.setup_initial_game(b)

    # Roll dice
    game_logic.roll_action_dice(state, pool=Side.FREE_PEOPLES)
    game_logic.roll_action_dice(state, pool=Side.SHADOW)

    # Print dice
    print("FP dice:", [d.current.value for d in state.dice[Side.FREE_PEOPLES] if d.current and not d.used])
    print("SH dice:", [d.current.value for d in state.dice[Side.SHADOW] if d.current and not d.used])

    fp_actions = get_legal_actions(state, Side.FREE_PEOPLES)
    sh_actions = get_legal_actions(state, Side.SHADOW)

    print(f"\nFP legal actions: {len(fp_actions)}")
    for a in fp_actions[:20]:
        print(f"  {a}")
    if len(fp_actions) > 20:
        print(f"  ... and {len(fp_actions) - 20} more")

    print(f"\nSH legal actions: {len(sh_actions)}")
    for a in sh_actions[:20]:
        print(f"  {a}")
    if len(sh_actions) > 20:
        print(f"  ... and {len(sh_actions) - 20} more")

    print("\nLegal actions: PASSED")
    print()


def _balrog_moria_on_declare_path(state, from_key, to_key, steps):
    """True if 'moria' lies on at least one legal fellowship-declaration
    path of length <= *steps* from *from_key* to *to_key*.

    Declaration reachability (game_logic.declare_fellowship) is a plain
    BFS over board adjacency: the pawn may be placed in any region at a
    distance up to the progress counter.  Moria lies on such a path iff
    dist(from,moria) + dist(moria,to) <= steps (the shortest path through
    a node; valid because board adjacency is undirected).  FROM / INTO
    Moria are the dist==0 endpoints and short-circuit to True.

    APPROXIMATION (documented, amendment): the model is board adjacency
    ONLY - fellowship declaration has no per-region legality beyond
    distance (the exact model declare_fellowship itself uses to validate
    the target), so within the declaration rules this test is exact.  It
    deliberately does NOT reconstruct the SECRET step-by-step route the
    Ring-bearers actually took (hidden information); the almanac trigger
    is 'moves through Moria while being declared', i.e. it is enough that
    a legal declaration path through Moria EXISTS.
    """
    moria = 'moria'
    if from_key == moria or to_key == moria:
        return True
    # A pure through-path needs dist(from,moria) >= 1 and
    # dist(moria,to) >= 1, so steps < 2 can never pass through.
    if steps is None or steps < 2:
        return False

    def _dists(src):
        seen = {src: 0}
        frontier = [src]
        for hop in range(1, steps + 1):
            nxt = []
            for r in frontier:
                for adj in state.board.adjacent(r):
                    a = sanitize(adj)
                    if a not in seen:
                        seen[a] = hop
                        nxt.append(a)
            frontier = nxt
            if not frontier:
                break
        return seen

    d_from = _dists(from_key).get(moria)
    if d_from is None:
        return False
    d_to = _dists(to_key).get(moria)
    return d_to is not None and (d_from + d_to) <= steps


def _balrog_stationary_turn_start(state) -> bool:
    """Balrog of Moria STATIONARY trigger (almanac C17, second bullet):
    if the Fellowship remains stationary IN Moria at the start of a turn
    and the 'Balrog of Moria' card is on the table, open the Shadow's
    discard-for-hunt-tile pending - the SAME pending shape and
    'balrog_declare_draw' handler branch used by the declaration trigger.

    LEARNED PATH ONLY: the certified closed-loop teacher replays logged
    choices and must never stall on a fresh interactive pending, so this
    is gated on combat.FINAL_HITS_HOOK is None (== self-play / learned
    mode; the teacher sets the hook).  No campaign log plays Balrog, so
    the canary is unaffected, but the gate is the mandatory pattern.

    Returns True iff a pending was opened.
    """
    import combat as _cmb_bal
    if _cmb_bal.FINAL_HITS_HOOK is not None:
        return False
    if getattr(state, 'pending_card_option', None) is not None:
        return False
    if state.fellowship.mordor_entered:
        return False
    try:
        if sanitize(game_logic.fellowship_region(state)) != 'moria':
            return False
    except Exception:
        return False
    # adversarial repair 2026-07-11 (MAJOR): 'remains STATIONARY' -
    # progress > 0 means the Fellowship has hidden-moved off the
    # marker; only a genuinely unmoved Fellowship triggers
    if state.fellowship.progress != 0:
        return False
    from game_state import PersistentCardType as _PCT_bal
    _pcb_bal = getattr(state, 'persistent_cards', None)
    _cb_bal = (_pcb_bal.get_by_type(_PCT_bal.BALROG_OF_MORIA)
               if _pcb_bal else None)
    if _cb_bal is None:
        return False
    state.pending_card_option = {
        'card_id': getattr(_cb_bal, 'card_id', '__balrog__'),
        'card_name': 'Balrog of Moria',
        'side': 'sh', 'decider': 'sh', 'direct': True,
        'options': [{'balrog_declare_draw': True},
                    {'balrog_declare_draw': False}]}
    return True


# ---------------------------------------------------------------------------
# WOTR_NATIVE=1 (PERF 2026-08-09; re-wired 2026-08-11 after the F10
# clobber - the task-15 confidence patch, generated against a base
# without this block, deleted it on apply; see lab comms 44_PERF).
# PURE performance swap, byte-identical outputs (parity-certified;
# perf/native in the lab tree; re-verified against THIS ai.py, whose
# post-8/9 changes touch no ported function).  Rebinding module
# globals here (the last statements executed at import) retargets
# every call site: internal callers resolve these names through
# module globals at call time, and external `from ai import X` /
# `ai.X` reads happen only after import completes.  Default OFF:
# without the env flag this block is inert and the module behaves
# byte-identically to production.  ACTIVATION IS GATED: run
# src/test_native_activation.py (exit 4 = flag requested but the
# native path is not actually taken - the exact F10 failure mode).
# NOTE for patch generators: diff against a base that CONTAINS this
# block, or your patch will delete it again (the F10 root cause).
if os.environ.get('WOTR_NATIVE') == '1':
    try:
        import wotr_native as _wotr_native
    except ImportError as _e:
        raise ImportError(
            'WOTR_NATIVE=1 but the wotr_native extension is not '
            'built/importable: %r' % (_e,))
    _wotr_native.init_ai(sys.modules[__name__])
    _wotr_native.init_state_encoder(sys.modules[__name__])
    _wotr_native.init_clone(sys.modules[__name__])
    _side_of_nation = _wotr_native.side_of_nation
    _armies_lookup = _wotr_native.armies_lookup
    encode_state = _wotr_native.encode_state
    clone_state = _wotr_native.clone_state
    _state_sig = _wotr_native.state_sig
    if HAS_TORCH:
        encode_action = _wotr_native.encode_action
        _action_cache_key = _wotr_native.action_cache_key
        _encode_action_cached = _wotr_native.encode_action_cached
        _region_index_cached = _wotr_native.region_index_cached
        _get_action_region_index = _wotr_native.get_action_region_index


if __name__ == '__main__':
    main()
