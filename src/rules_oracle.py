# -*- coding: ascii -*-
"""
rules_oracle.py -- CLEAN-ROOM legality oracle for War of the Ring (2nd ed).

This module was written WITHOUT reading the engine's rules code
(game_logic.py, ai.py, combat.py, event_effects.py, etc.).  It implements
WOTR legality from (a) the author's knowledge of the official 2nd-edition
rulebook, (b) data/CURRENT_RULES.md (house rulings, authoritative), and
(c) the interface shapes in data/action_interface_spec.md +
src/game_state.py + src/board.py.  A separate differential fuzzer diffs
this oracle against the engine; nothing here imports or trusts the engine.

--------------------------------------------------------------------------
SOURCES USED (for audit)
--------------------------------------------------------------------------
Rulebook chapters implemented from memory:
  * Action dice / die faces (Character/Army/Muster/Event/Muster-Army/
    Eye/Will-of-the-West) and which action category each face permits.
  * Movement & attack political restrictions (rulebook p.33): a unit of a
    Nation that is not "At War" cannot cross another Nation's border nor
    attack; ALL units in the acting Army must be At War (confirmed by
    rules_amendments.md line 1036-1042 citing p.33).
  * Political track (p.34): a passive (non-active) Nation can never be
    advanced INTO the "At War" step.
  * Hunt / eye allocation, Fellowship declaration & reveal distance,
    Nazgul movement restriction, minion entry conditions, coronation.

CURRENT_RULES.md lines followed (authoritative house rulings):
  * s1  Hunt: eye cap = # Companions in Fellowship; reveal/declare figure
        moves up to Progress and may NOT end in an unconquered FP city/
        stronghold.
  * s2  Fellowship: movement requires hidden=True; declaration hidden-only
        up to Progress regions; Crown Aragorn = Strider in unconquered
        Minas Tirith/Pelargir/Dol Amroth on a Will die.
  * s3  Nazgul/minions: Nazgul cannot end in a region with FP Army units
        except a besieged stronghold; Witch-king entry = Sauron At War +
        >=1 FP Nation At War; Mouth entry = all 5 FP Nations At War OR
        Fellowship on Mordor track (amendment 1.3); Saruman = Orthanc,
        Isengard At War; Saruman immobile.
  * s4  Combat/siege: assaulting a besieged stronghold is a co-located
        (from==to) attack; siege presses only exist at strongholds.
  * amendment 1.1: Elven Ring changes one UNUSED die to another face, not
    to Will of the West, once per side per turn.
  * USER 2026-07-20 rulings (CURRENT_RULES updated): (P2-03) FP table-kill
    payments are 'will' (Will die alone), 'ring' (any die + Elven Ring,
    Palantir), 'character' (any die while Aragorn/Gandalf is in Minas
    Tirith, Denethor's Folly only); (1.181) siege_trim: withdrawing owner
    chooses how many elites of the >5-cap excess return to reinforcements
    (excess is never casualties); (P2-02) mouth_move: a lone in-play Mouth
    moves up to 3 regions on a Character die with NO destination
    restrictions (unlike nazgul).

RULEBOOK vs CURRENT_RULES conflicts / uncertainties noticed (reported to
user; treat related diffs skeptically):
  1. Recruiting (muster_recruit / muster_elite) political gate: the
     rulebook does NOT clearly gate RECRUITING reinforcements on a Nation
     being "At War" (only movement/attack are gated).  rules_amendments.md
     (1.22) shows the engine BLOCKING musters for a Nation lagging below
     At War.  Because I cannot confirm the recruiting-requires-At-War rule
     from the rulebook, this oracle does NOT flag not-At-War musters as
     illegal (would risk false FALSE).  Muster diffs should be reviewed by
     hand.
  2. Moving a not-At-War Army into a NEUTRAL ("free") wilderness region:
     unclear whether this counts as leaving one's borders.  Oracle does
     NOT flag it (only crossing into another NAMED Nation is flagged).
  3. Political track box count: game_state models boxes 0..3 (3 = At War);
     the physical track has more steps.  Oracle uses the engine's 0..3
     model for the "already At War / cannot advance further" check.
  4. Leader-led Army movement/attack on a Character die is legal per
     rulebook; because a co-located Companion could also enable it, the
     oracle does not FLAG a Character-die army action even when no leader
     is visible (returns permissive / None instead of False).
  5. Denethor's Folly 'character' payment (USER 2026-07-20 P2-03) names
     "Aragorn / Gandalf".  This oracle also accepts a figure named
     'Strider' in Minas Tirith (the physical piece is the same and may be
     uncrowned); if the ruling means CROWNED Aragorn strictly, diffs where
     only Strider is present should be user-adjudicated, not trusted.

The rest is 'none' coverage -> check_action returns None (cannot
adjudicate) rather than guessing.
"""

from collections import deque

import board
from game_state import Side, Nation, ActionDieFace, UnitKind, Nazgul


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FP = Side.FREE_PEOPLES
SH = Side.SHADOW

FP_NATIONS = frozenset({
    Nation.ELVES, Nation.DWARVES, Nation.GONDOR,
    Nation.THE_NORTH, Nation.ROHAN,
})
SH_NATIONS = frozenset({
    Nation.ISENGARD, Nation.SAURON, Nation.SOUTHRON_EASTERLINGS,
})

CROWN_REGIONS = frozenset({"minas tirith", "pelargir", "dol amroth"})

# Faces that clearly CANNOT move/attack an Army under any circumstance.
_NON_ARMY_FACES = frozenset({ActionDieFace.EVENT})


ORACLE_COVERAGE = {
    # priority: die-face gating + high-traffic legality
    "advance_political": "partial",   # political-status legality (not face)
    "army_attack": "partial",         # face, all-At-War, adjacency, target
    "army_move": "partial",           # face, units, adjacency, enemy-occupied, border
    "muster_recruit": "partial",      # nation-owns-region, settlement, occupation
    "muster_elite": "partial",
    "muster_minion": "partial",       # entry conditions (Saruman/WK/Mouth)
    "eye_allocate": "partial",        # cap = companions; upper bound
    "fellowship_move": "partial",     # hidden + Character/Will face
    "fellowship_declare": "partial",  # hidden + distance <= progress
    "fellowship_reveal_place": "partial",  # distance + FP city/stronghold bar
    "nazgul_move": "partial",         # Character face + FP-army restriction
    "elven_ring": "partial",          # ownership + not-to-Will + unused die
    "crown_aragorn": "partial",       # Will die + not-already-crowned
    "draw_card": "partial",           # Event/Will face + deck side/non-empty
    "play_event": "partial",          # empty-hand
    "combat_retreat": "partial",      # adjacency + not-into-enemy
    "choose_guide": "partial",        # target must be in Fellowship
    "discard_table_card": "partial",  # target on table + FP payment face
    "companion_move": "partial",      # Character/Will face
    "siege_assault": "partial",       # Army face
    "retreat_into_siege": "partial",  # stronghold required (if determinable)
    "siege_trim": "partial",          # pending combat + 0<=elites_back<=excess
    "mouth_move": "partial",          # SH + in-play + Character face + range<=3
    # not adjudicated
    "companion_separate": "none",
    "fellowship_hide": "none",
    "dwarven_ring": "none",
    "card_option": "none",
    "casualty_allocate": "none",
    "casualty_preserve_elites": "none",
    "casualty_preserve_units": "none",
    "combat_card_play": "none",
    "combat_card_none": "none",
    "combat_effect_choice": "none",
    "combat_continue": "none",
    "combat_cease": "none",
    "combat_no_retreat": "none",
    "rearguard_split": "none",
    "rearguard_none": "none",
    "siege_press": "none",
    "siege_no_press": "none",
    "siege_continue": "none",
    "transform_gandalf": "none",
    "use_token": "none",
    "victor_advance": "none",
    "victor_stay": "none",
    "discard_card": "none",
    "pass": "none",
    "skip": "none",
    "hunt_take_corruption": "none",
    "hunt_sacrifice_guide": "none",
    "hunt_sacrifice_random": "none",
    "hunt_guide_separate": "none",
    "hunt_gollum_reveal": "none",
    "hunt_gollum_no_reveal": "none",
    "followup_done": "none",
    "shine": "none",
    "shine_none": "none",
}


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------
def _params(action):
    p = getattr(action, "params", None)
    return p if isinstance(p, dict) else {}


def _nation_side(nation):
    return FP if nation in FP_NATIONS else SH


def _nation_board_key(nation):
    """board faction key ('gondor','the north','southrons and easterlings')."""
    try:
        return board._canon_faction(nation.value)
    except Exception:
        try:
            return board.sanitize(nation.value)
        except Exception:
            return None


def _canon_key(state, name):
    """Canonical sanitized region key, robust to article variants."""
    if name is None:
        return None
    if hasattr(name, "key"):
        return name.key
    if not isinstance(name, str):
        return None
    try:
        return state.board.region(name).key
    except Exception:
        try:
            return board.sanitize(name)
        except Exception:
            return None


def _region_meta(key):
    try:
        return board.regions.get(key)
    except Exception:
        return None


def _region_nation_key(key):
    meta = _region_meta(key)
    if not meta:
        return None
    try:
        return board.fid_to_name.get(meta.get("parent_id"))
    except Exception:
        return None


def _is_city(key):
    meta = _region_meta(key)
    return bool(meta and meta.get("is_city"))


def _is_stronghold(key):
    meta = _region_meta(key)
    return bool(meta and meta.get("is_stronghold"))


def _is_settlement(key):
    meta = _region_meta(key)
    if not meta:
        return False
    return bool(meta.get("is_city") or meta.get("is_stronghold")
                or meta.get("is_fort") or meta.get("is_town"))


def _army_at(state, canon_key):
    """Find the Army object at a region, robust to dict-key convention."""
    if canon_key is None:
        return None
    try:
        a = state.armies.get(canon_key)
    except Exception:
        a = None
    if a is not None:
        return a
    try:
        for av in state.armies.values():
            try:
                if av.region.key == canon_key:
                    return av
            except Exception:
                continue
    except Exception:
        return None
    return None


def _side_unit_counts(army, side):
    reg = el = 0
    for u in getattr(army, "units", []):
        try:
            if _nation_side(u.nation) == side:
                if u.kind == UnitKind.ELITE:
                    el += 1
                else:
                    reg += 1
        except Exception:
            continue
    return reg, el


def _side_units_total(state, key, side):
    army = _army_at(state, key)
    if army is None:
        return 0
    reg, el = _side_unit_counts(army, side)
    return reg + el


def _side_nazgul(army):
    n = 0
    for l in getattr(army, "leaders", []):
        if isinstance(l, Nazgul):
            n += 1
    return n


def _side_nations_in_army(army, side):
    out = set()
    for u in getattr(army, "units", []):
        try:
            if _nation_side(u.nation) == side:
                out.add(u.nation)
        except Exception:
            continue
    return out


def _is_besieged(state, key):
    if key is None:
        return False
    try:
        if key in (state.besieged_strongholds or set()):
            return True
    except Exception:
        pass
    return False


def _adjacent(state, a, b):
    try:
        return b in state.board.adjacent(a)
    except Exception:
        return False


def _bfs_within(state, start_key, goal_key, max_d):
    """True if goal is reachable from start within max_d region-steps."""
    if start_key is None or goal_key is None:
        return None
    if start_key == goal_key:
        return True
    if max_d is None or max_d < 1:
        return False
    seen = {start_key}
    dq = deque([(start_key, 0)])
    while dq:
        k, d = dq.popleft()
        if d >= max_d:
            continue
        try:
            neigh = state.board.adjacent(k)
        except Exception:
            neigh = []
        for n in neigh:
            if n == goal_key:
                return True
            if n not in seen:
                seen.add(n)
                dq.append((n, d + 1))
    return False


def _get_die(state, side, idx):
    try:
        dl = state.dice.get(side)
    except Exception:
        dl = None
    if dl is None:
        return None, "no dice for this side"
    if not isinstance(idx, int) or idx < 0 or idx >= len(dl):
        return None, "die_idx out of range"
    return dl[idx], None


def _die_face(die):
    return getattr(die, "current", None)


def _as_face(x):
    """Coerce a face-ish value (enum or str) to ActionDieFace or None."""
    if isinstance(x, ActionDieFace):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        for f in ActionDieFace:
            if s == f.value.lower() or s == f.name.lower():
                return f
        if "will" in s:
            return ActionDieFace.WILL_OF_WEST
        if "eye" in s:
            return ActionDieFace.EYE
        if "muster" in s and "army" in s:
            return ActionDieFace.MUSTER_ARMY
    return None


# --------------------------------------------------------------------------
# Generic die precondition (used, rolled, eye-face, ownership)
# --------------------------------------------------------------------------
def _die_precheck(state, side, action):
    """Return (verdict, reason) if the die itself is clearly illegal to
    spend; return (None, None) if the die is fine or face unknown."""
    di = getattr(action, "die_idx", -1)
    if not isinstance(di, int) or di < 0:
        return None, None  # decision-phase action, no die
    die, err = _get_die(state, side, di)
    if die is None:
        return False, "no such action die (%s)" % err
    if getattr(die, "used", False):
        return False, "die already spent (rulebook: each die used once)"
    face = _die_face(die)
    if face == ActionDieFace.EYE:
        return False, "Eye dice are hunt-only (rulebook Hunt; CURRENT_RULES s1)"
    # unrolled die (face None): cannot confirm face -> stay silent
    return None, None


# --------------------------------------------------------------------------
# check_action dispatch
# --------------------------------------------------------------------------
def check_action(state, side, action):
    """-> (verdict, reason).  True=legal, False=ILLEGAL (cites rule),
    None=cannot adjudicate."""
    try:
        return _check_action_inner(state, side, action)
    except Exception as exc:  # never crash a caller -- but stay LOUD:
        # an internal oracle defect must be distinguishable from a
        # legitimate cannot-adjudicate (the fuzzer keys on this prefix)
        import traceback
        print('[rules_oracle] INTERNAL ERROR adjudicating %r: %r'
              % (getattr(action, 'action_type', None), exc))
        traceback.print_exc()
        return None, "oracle-internal (%s): %s" % (
            type(exc).__name__, exc)


def _check_action_inner(state, side, action):
    at = getattr(action, "action_type", None)
    if not isinstance(at, str):
        return None, "no action_type"

    # 1) generic die precheck first (used/eye/out-of-range)
    v, r = _die_precheck(state, side, action)
    if v is False:
        return False, r

    cov = ORACLE_COVERAGE.get(at)
    if cov is None or cov == "none":
        return None, "action_type '%s' not adjudicated" % at

    p = _params(action)
    di = getattr(action, "die_idx", -1)
    die = None
    face = None
    if isinstance(di, int) and di >= 0:
        die, _err = _get_die(state, side, di)
        face = _die_face(die) if die is not None else None

    pol = state.political

    # ---- eye_allocate --------------------------------------------------
    if at == "eye_allocate":
        cnt = p.get("count")
        if not isinstance(cnt, int):
            return None, "eye_allocate count not an int"
        cap = len(getattr(state.fellowship, "companions", []) or [])
        if cnt < 0:
            return False, "eye_allocate count < 0"
        # CURRENT_RULES s1: "Minimum 1 die is always *permitted* even at
        # zero companions" -- count=1 is ALWAYS legal regardless of the
        # companion cap; the cap only binds counts >= 2.  The cap applies
        # on the Mordor track too (no Mordor exception).
        if cnt <= 1:
            return (True,
                    "eye allocation of %d always permitted "
                    "(CURRENT_RULES s1: minimum 1 die always allowed)" % cnt)
        if cnt > cap:
            return (False,
                    "eye allocation %d exceeds cap %d (= Companions in "
                    "Fellowship; CURRENT_RULES s1)" % (cnt, cap))
        return True, "eye allocation within cap (CURRENT_RULES s1)"

    # ---- advance_political --------------------------------------------
    if at == "advance_political":
        nation = p.get("nation")
        if not isinstance(nation, Nation):
            return None, "advance_political: nation not resolvable"
        box = None
        try:
            box = pol.boxes.get(nation)
        except Exception:
            box = None
        if box is None:
            return None, "advance_political: box unknown"
        if box >= 3:
            return (False,
                    "%s already At War (box 3); cannot advance further "
                    "(rulebook p.34)" % nation.value)
        if box == 2 and not pol.is_active(nation):
            return (False,
                    "passive %s cannot advance into At War (rulebook p.34; "
                    "CURRENT_RULES s6)" % nation.value)
        return True, "political advance permitted (rulebook p.34)"

    # ---- army_move -----------------------------------------------------
    if at == "army_move":
        return _check_army_move(state, side, p, face)

    # ---- army_attack ---------------------------------------------------
    if at == "army_attack":
        return _check_army_attack(state, side, p, face)

    # ---- muster_recruit / muster_elite --------------------------------
    if at in ("muster_recruit", "muster_elite"):
        return _check_muster(state, side, p)

    # ---- muster_minion -------------------------------------------------
    if at == "muster_minion":
        return _check_muster_minion(state, side, p)

    # ---- fellowship_move ----------------------------------------------
    if at == "fellowship_move":
        fs = state.fellowship
        if not getattr(fs, "hidden", False):
            return (False,
                    "Fellowship must be hidden to move (CURRENT_RULES s2)")
        if face is not None and face not in (ActionDieFace.CHARACTER,
                                             ActionDieFace.WILL_OF_WEST):
            return (False,
                    "Fellowship move is a Character action; die face %s "
                    "cannot move it (rulebook)" % face.value)
        return True, "Fellowship hidden + Character/Will die (CURRENT_RULES s2)"

    # ---- fellowship_declare -------------------------------------------
    if at == "fellowship_declare":
        fs = state.fellowship
        if not getattr(fs, "hidden", False):
            return (False,
                    "Fellowship must be hidden to declare (CURRENT_RULES s2)")
        region = _region_key(state, p.get("region"))
        if region is None:
            return None, "declare: region not resolvable"
        start = _region_key(state, getattr(fs, "region", None))
        prog = getattr(fs, "progress", 0) or 0
        within = _bfs_within(state, start, region, prog)
        if within is False:
            return (False,
                    "declared region beyond Progress (%d) from last known "
                    "position (CURRENT_RULES s2)" % prog)
        if within is None:
            return None, "declare: distance not resolvable"
        return True, "declaration within Progress (CURRENT_RULES s2)"

    # ---- fellowship_reveal_place --------------------------------------
    if at == "fellowship_reveal_place":
        return _check_reveal_place(state, p)

    # ---- nazgul_move ---------------------------------------------------
    if at == "nazgul_move":
        return _check_nazgul_move(state, side, p, face)

    # ---- elven_ring ----------------------------------------------------
    if at == "elven_ring":
        return _check_elven_ring(state, side, p)

    # ---- crown_aragorn -------------------------------------------------
    if at == "crown_aragorn":
        ac = getattr(state, "aragorn_crowned", None)
        if ac is not None and getattr(ac, "crowned", False):
            return False, "Aragorn already crowned (CURRENT_RULES s2)"
        if face is not None and face != ActionDieFace.WILL_OF_WEST:
            return (False,
                    "Crowning Aragorn requires a Will of the West die "
                    "(CURRENT_RULES s2)")
        # location check when Strider is inside the Fellowship
        fs = state.fellowship
        names = {getattr(c, "name", "") for c in getattr(fs, "companions", [])}
        if "Strider" in names:
            here = _region_key(state, getattr(fs, "region", None))
            if here is not None and here not in CROWN_REGIONS:
                return None, "crown: Strider may be separated; unverified"
            if here in CROWN_REGIONS and _side_units_total(state, here, SH) > 0:
                return None, "crown: city may be contested; unverified"
        return None, "crown preconditions plausible; location unverified"

    # ---- draw_card -----------------------------------------------------
    if at == "draw_card":
        return _check_draw_card(state, side, p, face)

    # ---- play_event ----------------------------------------------------
    if at == "play_event":
        try:
            hand = state.hands[side].hand
        except Exception:
            hand = None
        if hand is not None and len(hand) == 0:
            return False, "no cards in hand to play (rulebook events)"
        return None, "play_event: card/requirement not fully adjudicated"

    # ---- combat_retreat ------------------------------------------------
    if at == "combat_retreat":
        return _check_combat_retreat(state, side, p)

    # ---- choose_guide --------------------------------------------------
    if at == "choose_guide":
        comp = p.get("companion")
        cname = comp if isinstance(comp, str) else getattr(comp, "name", None)
        fs = state.fellowship
        names = {getattr(c, "name", "") for c in getattr(fs, "companions", [])}
        if cname is None:
            return None, "choose_guide: companion not resolvable"
        if names and cname not in names and cname.lower() != "gollum":
            return (False,
                    "guide must be a Companion in the Fellowship "
                    "(rulebook Fellowship)")
        return None, "choose_guide plausible; not fully adjudicated"

    # ---- discard_table_card -------------------------------------------
    if at == "discard_table_card":
        return _check_discard_table_card(state, side, p, face)

    # ---- companion_move ------------------------------------------------
    if at == "companion_move":
        if face is not None and face not in (ActionDieFace.CHARACTER,
                                             ActionDieFace.WILL_OF_WEST):
            return (False,
                    "Companion movement is a Character action; die face %s "
                    "cannot do it (rulebook)" % face.value)
        return None, "companion_move face ok; path not fully adjudicated"

    # ---- siege_assault -------------------------------------------------
    if at == "siege_assault":
        if face is not None and face not in (ActionDieFace.ARMY,
                                             ActionDieFace.MUSTER_ARMY,
                                             ActionDieFace.WILL_OF_WEST):
            return (False,
                    "Siege assault costs an Army die; face %s cannot "
                    "(rulebook Siege; CURRENT_RULES s4)" % face.value)
        return None, "siege_assault face ok; siege context not adjudicated"

    # ---- siege_trim (USER 2026-07-20, 1.181) --------------------------
    if at == "siege_trim":
        eb = p.get("elites_back")
        ex = p.get("excess")
        if isinstance(eb, int) and isinstance(ex, int):
            if eb < 0 or ex < 0 or eb > ex:
                return (False,
                        "siege_trim: elites_back must be between 0 and the "
                        "excess unit count (CURRENT_RULES s4; USER 1.181)")
        if getattr(state, "pending_combat", None):
            return (True,
                    "siege_trim: owner's choice of elites returned to "
                    "reinforcements during siege withdrawal "
                    "(CURRENT_RULES s4; USER 1.181)")
        return None, "siege_trim: pending combat context not visible"

    # ---- mouth_move (USER 2026-07-20, P2-02) --------------------------
    if at == "mouth_move":
        if side != SH:
            return False, "Mouth of Sauron is a Shadow minion (CURRENT_RULES s3)"
        mouth = getattr(state, "mouth_of_sauron", None)
        if mouth is not None and not getattr(mouth, "in_play", False):
            return False, "Mouth of Sauron is not in play (CURRENT_RULES s3)"
        if face is not None and face != ActionDieFace.CHARACTER:
            return (False,
                    "mouth_move is a Character-die action; face %s cannot "
                    "(CURRENT_RULES s3; USER P2-02)" % face.value)
        frm = _region_key(state, p.get("from"))
        to = _region_key(state, p.get("to"))
        if frm is None or to is None:
            return None, "mouth_move: from/to not resolvable"
        within = _bfs_within(state, frm, to, 3)
        if within is False:
            return (False,
                    "mouth_move destination beyond range 3 "
                    "(CURRENT_RULES s3; USER 2026-07-20 P2-02)")
        if within is None:
            return None, "mouth_move: range not resolvable"
        # NO destination restrictions per the ruling (unlike nazgul):
        # deliberately no FP-army / stronghold bar here.
        return (True,
                "mouth_move: Character die + range<=3; no destination "
                "restrictions (CURRENT_RULES s3; USER P2-02)")

    # ---- retreat_into_siege -------------------------------------------
    if at == "retreat_into_siege":
        pc = getattr(state, "pending_combat", None)
        reg = None
        if isinstance(pc, dict):
            for k in ("defender_region", "to", "region", "attacker_region"):
                if pc.get(k):
                    reg = _region_key(state, pc.get(k))
                    break
        if reg is not None and not _is_stronghold(reg):
            return (False,
                    "can only withdraw into siege at a Stronghold "
                    "(rulebook Siege; CURRENT_RULES s4)")
        return None, "retreat_into_siege: stronghold context not resolvable"

    return None, "action_type '%s' not adjudicated" % at


# --------------------------------------------------------------------------
# army_move
# --------------------------------------------------------------------------
def _check_army_move(state, side, p, face):
    frm = _region_key(state, p.get("from"))
    to = _region_key(state, p.get("to"))
    if frm is None or to is None:
        return None, "army_move: from/to not resolvable"

    # --- die face gating (permissive where exceptions may apply) -------
    fv = _army_face_verdict(state, side, face, frm)
    if fv is not None:
        return fv  # a (False, reason)

    # --- source army exists with the claimed units ---------------------
    army = _army_at(state, frm)
    if army is None:
        return False, "no Army at source region '%s' (rulebook movement)" % frm
    reg, el = _side_unit_counts(army, side)
    naz = _side_nazgul(army)
    nreg = int(p.get("n_regular", 0) or 0)
    nel = int(p.get("n_elite", 0) or 0)
    nnaz = int(p.get("n_nazgul", 0) or 0)
    if nreg > reg:
        return (False,
                "source has %d regulars, move claims %d (rulebook movement)"
                % (reg, nreg))
    if nel > el:
        return (False,
                "source has %d elites, move claims %d (rulebook movement)"
                % (el, nel))
    if nnaz > naz:
        return (False,
                "source has %d Nazgul, move claims %d (rulebook)" % (naz, nnaz))
    if nreg + nel + nnaz == 0:
        if not p.get("take_leaders"):
            return None, "army_move: empty move; not adjudicated"
        # a leaders-only 'army move' moves no Army units -- the
        # rulebook forbids it (leaders may never move alone as an
        # army; Nazgul use nazgul_move, companions companion_move)
        return (False,
                "army_move with 0 units (leaders-only) is not a legal "
                "army movement (rulebook p.25)")

    # --- from == to is not a plain move --------------------------------
    if frm == to:
        return None, "army_move from==to; not adjudicated"

    # --- adjacency -----------------------------------------------------
    if not _adjacent(state, frm, to):
        return (False,
                "destination '%s' not adjacent to '%s' (rulebook movement)"
                % (to, frm))

    # --- cannot move into a region containing enemy units --------------
    # (the besieged exemption is for JOINING one's own besiegers: the
    # walled-in garrison does not block entry.  It must NOT exempt the
    # garrison side reinforcing its own besieged stronghold from
    # outside -- stay silent there rather than blessing it.)
    enemy = SH if side == FP else FP
    if _side_units_total(state, to, enemy) > 0:
        if not _is_besieged(state, to):
            return (False,
                    "cannot move into region '%s' occupied by enemy "
                    "units; must attack instead (rulebook p.33)" % to)
        return None, ("army_move into besieged region '%s'; "
                      "garrison-vs-besieger legality not adjudicated"
                      % to)

    # --- cross-border requires At War (single-nation source only) ------
    src_nats = _side_nations_in_army(army, side)
    if len(src_nats) == 1:
        mover = next(iter(src_nats))
        dst_nat_key = _region_nation_key(to)
        mover_key = _nation_board_key(mover)
        if (dst_nat_key and dst_nat_key not in ("free", None)
                and dst_nat_key != mover_key):
            if not state.political.is_at_war(mover):
                return (False,
                        "%s is not At War; cannot cross border into %s "
                        "territory (rulebook p.33)"
                        % (mover.value, dst_nat_key))

    return True, "army_move: face/units/adjacency/border checks pass"


def _army_face_verdict(state, side, face, frm):
    """Return (False, reason) if the die face clearly cannot move/attack an
    Army; else None."""
    if face is None:
        return None
    if face in (ActionDieFace.ARMY, ActionDieFace.MUSTER_ARMY):
        return None
    if side == FP and face == ActionDieFace.WILL_OF_WEST:
        return None
    if face == ActionDieFace.CHARACTER:
        return None  # leader/companion-led move may be legal -> stay silent
    if face == ActionDieFace.MUSTER:
        # only legal for Shadow while the Mouth of Sauron is in play
        mouth = getattr(state, "mouth_of_sauron", None)
        if side == SH and mouth is not None and getattr(mouth, "in_play", False):
            return None
        return (False,
                "a plain Muster die cannot move/attack an Army "
                "(rulebook; no Mouth-of-Sauron exception available)")
    if face in _NON_ARMY_FACES:
        return (False,
                "an Event die cannot move/attack an Army (rulebook)")
    return None


# --------------------------------------------------------------------------
# army_attack
# --------------------------------------------------------------------------
def _check_army_attack(state, side, p, face):
    frm = _region_key(state, p.get("from"))
    to = _region_key(state, p.get("to"))
    if frm is None or to is None:
        return None, "army_attack: from/to not resolvable"

    fv = _army_face_verdict(state, side, face, frm)
    if fv is not None:
        return fv

    army = _army_at(state, frm)
    if army is None:
        return False, "no attacking Army at '%s' (rulebook combat)" % frm

    # --- ALL attacking units must belong to a Nation At War (p.33) -----
    src_nats = _side_nations_in_army(army, side)
    if not src_nats:
        return (False,
                "no %s units at '%s' to attack with (rulebook combat)"
                % (side.value, frm))
    for n in src_nats:
        if not state.political.is_at_war(n):
            return (False,
                    "%s is not At War; every unit in an attacking Army must "
                    "be At War (rulebook p.33)" % n.value)

    # --- co-located battle (from == to) --------------------------------
    # a siege assault OR a non-besieged co-located FIELD battle
    # (Corsairs of Umbar landing, almanac A10) - both legal; the engine
    # offers from==to whenever own and enemy units share the region
    if frm == to:
        if _is_besieged(state, to):
            return True, "co-located siege assault (CURRENT_RULES s4)"
        return True, "co-located field battle (almanac A10)"

    # --- adjacency -----------------------------------------------------
    if not _adjacent(state, frm, to):
        return (False,
                "attack target '%s' not adjacent to '%s' (rulebook combat)"
                % (to, frm))

    # --- defender must have enemy units --------------------------------
    enemy = SH if side == FP else FP
    if _side_units_total(state, to, enemy) <= 0:
        return (False,
                "no enemy Army in '%s' to attack (rulebook combat)" % to)

    return True, "army_attack: face/At-War/adjacency/target checks pass"


# --------------------------------------------------------------------------
# muster
# --------------------------------------------------------------------------
def _check_muster(state, side, p):
    nation = p.get("nation")
    region = _region_key(state, p.get("region"))
    if not isinstance(nation, Nation):
        return None, "muster: nation not resolvable"
    if region is None:
        return None, "muster: region not resolvable"

    # recruiting must be for one of the acting side's Nations
    if _nation_side(nation) != side:
        return (False,
                "%s cannot recruit %s units (wrong side; rulebook muster)"
                % (side.value, nation.value))

    # region must belong to the recruiting Nation
    reg_nat = _region_nation_key(region)
    want = _nation_board_key(nation)
    if reg_nat is not None and want is not None and reg_nat != want:
        return (False,
                "'%s' is not a %s settlement; reinforcements go in that "
                "Nation's own Settlements (rulebook muster)"
                % (region, nation.value))

    # region must be a Settlement
    if not _is_settlement(region):
        return (False,
                "'%s' is not a Settlement; reinforcements are placed only in "
                "Settlements (rulebook muster)" % region)

    # region must not be enemy-occupied (unless a besieged friendly hold)
    enemy = SH if side == FP else FP
    if _side_units_total(state, region, enemy) > 0 and not _is_besieged(state, region):
        return (False,
                "cannot recruit in '%s'; it is occupied by enemy units "
                "(rulebook muster)" % region)

    return (None,
            "muster nation/settlement/occupation ok; political-status gate "
            "and reinforcement pool not adjudicated")


def _check_muster_minion(state, side, p):
    minion = str(p.get("minion", "")).lower()
    pol = state.political
    if "saruman" in minion:
        m = getattr(state, "saruman", None)
        if m is not None and getattr(m, "in_play", False):
            return False, "Saruman already in play (rulebook)"
        if not pol.is_at_war(Nation.ISENGARD):
            return (False,
                    "Saruman requires Isengard At War (rulebook character)")
        reg = _region_key(state, p.get("region"))
        if reg is not None and reg != "orthanc":
            return (False,
                    "Saruman enters at Orthanc only (CURRENT_RULES s2/s3)")
        return None, "Saruman entry conditions met; placement unverified"
    if "witch" in minion or "wk" == minion:
        m = getattr(state, "witch_king", None)
        if m is not None and getattr(m, "in_play", False):
            return False, "Witch-king already in play (rulebook)"
        if not pol.is_at_war(Nation.SAURON):
            return (False,
                    "Witch-king requires Sauron At War (CURRENT_RULES s3)")
        if not any(pol.is_at_war(n) for n in FP_NATIONS):
            return (False,
                    "Witch-king requires >=1 FP Nation At War "
                    "(CURRENT_RULES s3)")
        return None, "Witch-king entry conditions met; placement unverified"
    if "mouth" in minion:
        m = getattr(state, "mouth_of_sauron", None)
        if m is not None and getattr(m, "in_play", False):
            return False, "Mouth of Sauron already in play (rulebook)"
        all_fp_war = all(pol.is_at_war(n) for n in FP_NATIONS)
        on_mordor = getattr(state.fellowship, "mordor_entered", False)
        if not (all_fp_war or on_mordor):
            return (False,
                    "Mouth requires all 5 FP Nations At War or Fellowship on "
                    "the Mordor track (CURRENT_RULES s3; amendment 1.3)")
        return None, "Mouth entry conditions met; placement unverified"
    return None, "muster_minion: unknown minion"


# --------------------------------------------------------------------------
# fellowship reveal placement
# --------------------------------------------------------------------------
def _check_reveal_place(state, p):
    region = _region_key(state, p.get("region"))
    if region is None:
        return None, "reveal_place: region not resolvable"
    fs = state.fellowship
    start = _region_key(state, getattr(fs, "region", None))
    prog = getattr(fs, "progress", 0) or 0
    within = _bfs_within(state, start, region, prog)
    if within is False:
        return (False,
                "reveal placement beyond Progress (%d) from last known "
                "position (CURRENT_RULES s1)" % prog)
    # cannot end in an unconquered FP city/stronghold
    reg_nat = _region_nation_key(region)
    is_fp_nation = False
    if reg_nat:
        for n in FP_NATIONS:
            if _nation_board_key(n) == reg_nat:
                is_fp_nation = True
                break
    if is_fp_nation and (_is_city(region) or _is_stronghold(region)):
        if _side_units_total(state, region, SH) == 0 and not _is_besieged(state, region):
            return (False,
                    "cannot reveal in an unconquered FP city/stronghold '%s' "
                    "(CURRENT_RULES s1)" % region)
    if within is None:
        return None, "reveal_place: distance not resolvable"
    return True, "reveal placement within Progress and not in FP hold"


# --------------------------------------------------------------------------
# nazgul_move
# --------------------------------------------------------------------------
def _check_nazgul_move(state, side, p, face):
    if face is not None and face != ActionDieFace.CHARACTER:
        return (False,
                "Nazgul movement is a Character action; die face %s cannot "
                "(rulebook)" % face.value)
    frm = _region_key(state, p.get("from"))
    to = _region_key(state, p.get("to"))
    cnt = int(p.get("count", 1) or 1)
    if frm is not None:
        army = _army_at(state, frm)
        avail = _side_nazgul(army) if army is not None else 0
        # Witch-king can add to the flight; be lenient by +1 when requested
        if getattr(p, "get", None) and p.get("include_wk"):
            avail += 1
        if cnt > avail and army is not None:
            return (False,
                    "source '%s' has %d Nazgul, move claims %d (rulebook)"
                    % (frm, avail, cnt))
    if to is not None:
        # Nazgul cannot end in a region with FP Army units unless it is a
        # besieged stronghold (they join the besiegers).
        if _side_units_total(state, to, FP) > 0 and not _is_besieged(state, to):
            return (False,
                    "Nazgul cannot end in '%s' which holds FP Army units "
                    "(not besieged) (CURRENT_RULES s3)" % to)
    return True, "nazgul_move: face + FP-army restriction checks pass"


# --------------------------------------------------------------------------
# elven_ring
# --------------------------------------------------------------------------
def _check_elven_ring(state, side, p):
    if side == FP:
        rings = getattr(state, "fp_elven_rings", 0) or 0
        used = getattr(state, "fp_ring_used_this_turn", False)
        who = "FP"
    else:
        rings = getattr(state, "sh_elven_rings", 0) or 0
        used = getattr(state, "sh_ring_used_this_turn", False)
        who = "Shadow"
    if rings <= 0:
        return False, "%s has no Elven Ring available (amendment 1.1)" % who
    if used:
        return (False,
                "%s already used an Elven Ring this turn (amendment 1.1: "
                "once per side per turn)" % who)
    new_face = _as_face(p.get("new_face"))
    old_face = _as_face(p.get("old_face"))
    # the not-to-Will ban has NO side qualifier (amendment 1.1 / F-4:
    # a ring may never produce Will of the West for anyone)
    if new_face == ActionDieFace.WILL_OF_WEST:
        return (False,
                "an Elven Ring cannot change a die to Will of the West "
                "(amendment 1.1)")
    if side == FP and new_face == ActionDieFace.EYE:
        return (False,
                "the Eye is not a Free Peoples face -- an FP ring cannot "
                "produce it")
    if new_face is not None and old_face is not None and new_face == old_face:
        return (False,
                "an Elven Ring must change the die to a DIFFERENT face "
                "(amendment 1.1)")
    if side != FP and old_face == ActionDieFace.EYE:
        return (False,
                "Shadow cannot change a die showing the Eye with a ring")
    tdi = p.get("target_die_idx")
    if isinstance(tdi, int):
        die, err = _get_die(state, side, tdi)
        if die is None:
            return False, "Elven Ring target die invalid (%s)" % err
        if getattr(die, "used", False):
            return (False,
                    "Elven Ring must target an UNUSED die (amendment 1.1)")
        if side != FP and getattr(die, "current", None) \
                == ActionDieFace.EYE:
            return (False,
                    "Shadow cannot change a die showing the Eye with a "
                    "ring")
    return True, "elven_ring: ownership/target/face-change checks pass"


# --------------------------------------------------------------------------
# draw_card
# --------------------------------------------------------------------------
def _check_draw_card(state, side, p, face):
    if face is not None:
        ok = (face == ActionDieFace.EVENT
              or (side == FP and face == ActionDieFace.WILL_OF_WEST))
        if not ok:
            return (False,
                    "drawing a card requires an Event die (or Will for FP); "
                    "face %s cannot (rulebook)" % face.value)
    # (deck is 'character'/'strategy' with the side in params['side']:
    #  side-legality and deck exhaustion are the engine's own draw
    #  path's business, not the oracle's)
    return True, "draw_card: face checks pass"


# --------------------------------------------------------------------------
# combat_retreat
# --------------------------------------------------------------------------
def _check_combat_retreat(state, side, p):
    to = _region_key(state, p.get("to") or p.get("to_region"))
    frm = _region_key(state, p.get("from"))
    if to is None:
        return None, "combat_retreat: destination not resolvable"
    if frm is not None and not _adjacent(state, frm, to):
        return (False,
                "retreat destination '%s' not adjacent to battle region '%s' "
                "(rulebook retreat)" % (to, frm))
    enemy = SH if side == FP else FP
    if _side_units_total(state, to, enemy) > 0:
        return (False,
                "cannot retreat into '%s'; it contains enemy units "
                "(rulebook retreat)" % to)
    return None, "combat_retreat: control status not fully adjudicated"


# --------------------------------------------------------------------------
# discard_table_card (FP payments per USER 2026-07-20, P2-03)
# --------------------------------------------------------------------------
_MT_HERO_TOKENS = ("aragorn", "strider", "gandalf")


def _hero_in_minas_tirith(state):
    """Is Aragorn or Gandalf (Grey/White) visibly in Minas Tirith?
    -> True (in the MT army), None (maybe: inside a Fellowship parked
    there), False (not visible anywhere in MT)."""
    army = _army_at(state, "minas tirith")
    if army is not None:
        for l in getattr(army, "leaders", []):
            nm = str(getattr(l, "name", "")).lower()
            if nm and any(t in nm for t in _MT_HERO_TOKENS):
                return True
    try:
        fs = state.fellowship
        if _region_key(state, getattr(fs, "region", None)) == "minas tirith":
            for c in getattr(fs, "companions", []):
                nm = str(getattr(c, "name", "")).lower()
                if any(t in nm for t in _MT_HERO_TOKENS):
                    return None
    except Exception:
        pass
    return False


def _check_discard_table_card(state, side, p, face):
    pc = getattr(state, "persistent_cards", None)
    if pc is None:
        return None, "discard_table_card: no persistent-card state"
    opp_cards = pc.sh_cards if side == FP else pc.fp_cards
    if not opp_cards:
        return (False,
                "no opponent table cards in play to discard "
                "(CURRENT_RULES s5)")

    # resolve the named target on the opponent's table
    target = None
    cid = p.get("card_id")
    cname = p.get("card_name")
    if cid or cname:
        for c in opp_cards:
            if ((cid and getattr(c, "card_id", None) == cid)
                    or (cname and getattr(c, "card_name", None) == cname)):
                target = c
                break
        if target is None:
            return (False,
                    "target card is not on the opponent's table "
                    "(CURRENT_RULES s5)")
    tname = str(getattr(target, "card_name", cname or "") or "").lower()
    is_folly = "denethor" in tname

    if side == SH:
        # SH pays any die + one ARMY-type strategy card + one Character-deck
        # card; hand card TYPES are not visible to the oracle -> None.
        return (None,
                "SH table-kill hand payment (Army strategy + Character "
                "card) not adjudicated")

    payment = str(p.get("payment", "") or "").lower()
    rings = getattr(state, "fp_elven_rings", 0) or 0

    if payment == "will":
        if face is not None and face != ActionDieFace.WILL_OF_WEST:
            return (False,
                    "'will' payment requires a Will of the West die alone "
                    "(CURRENT_RULES s5)")
        return True, "FP Will-die table kill (CURRENT_RULES s5)"

    if payment == "ring":
        if rings <= 0:
            return (False,
                    "'ring' payment needs an Elven Ring; FP has none "
                    "(CURRENT_RULES s5)")
        return True, "FP any-die + Elven Ring table kill (CURRENT_RULES s5)"

    if payment == "character":
        if target is not None and not is_folly:
            return (False,
                    "'character' payment removes only Denethor's Folly "
                    "(USER 2026-07-20, P2-03)")
        hero = _hero_in_minas_tirith(state)
        if hero is True:
            return (True,
                    "any-die Folly kill: Aragorn/Gandalf in Minas Tirith "
                    "(USER 2026-07-20, P2-03)")
        if hero is None:
            return (None,
                    "hero may be in Minas Tirith inside the Fellowship; "
                    "unverified")
        return (False,
                "'character' payment needs Aragorn or Gandalf in Minas "
                "Tirith (USER 2026-07-20, P2-03)")

    # payment route unspecified: is ANY route open?
    if face == ActionDieFace.WILL_OF_WEST:
        return True, "Will die alone pays for an FP table kill (CURRENT_RULES s5)"
    if rings > 0:
        return None, "ring payment possible; payment route unspecified"
    if is_folly or target is None:
        if _hero_in_minas_tirith(state) is not False:
            return None, "character payment may apply; route unspecified"
    if face is None:
        return None, "die face unknown; payment route unverifiable"
    return (False,
            "no FP payment route available: non-Will die, no Elven Ring, "
            "and no Aragorn/Gandalf-in-Minas-Tirith Folly case "
            "(CURRENT_RULES s5; USER 2026-07-20)")


# --------------------------------------------------------------------------
# region-key coercion for action params
# --------------------------------------------------------------------------
def _region_key(state, val):
    if val is None:
        return None
    if hasattr(val, "key"):
        return val.key
    if isinstance(val, str):
        return _canon_key(state, val)
    return None


# --------------------------------------------------------------------------
# required_offers
# --------------------------------------------------------------------------
def required_offers(state, side, offered_actions):
    """Return ascii strings for rules-mandated options MISSING from the
    menu.  CONSERVATIVE: only unconditional obligations given visible
    state.  Empty list = no violation found."""
    out = []
    try:
        offered_types = set()
        for a in (offered_actions or []):
            t = getattr(a, "action_type", None)
            if isinstance(t, str):
                offered_types.add(t)

        # Obligation: a hidden Fellowship with an unused Character/Will die
        # and no pending decision MUST be able to move.
        if side == FP:
            fs = state.fellowship
            hidden = getattr(fs, "hidden", False)
            mordor_done = (getattr(fs, "mordor_entered", False)
                           and getattr(fs, "mordor_step", 0) >= 5)
            corrupt_out = getattr(fs, "corruption", 0) >= 12
            # Generic pending-decision detection: mid-chain decision states
            # (combat, hunt, card options, discards, double-move/muster
            # halves, guide changes, ...) offer only the chain's own
            # continuation actions, so NO obligation applies while ANY
            # 'pending*' attribute on the state is populated (truthy).
            # A hardcoded field list here proved too narrow (fuzz round 1:
            # pending_card_option / pending_discard / pending_army_move_*
            # / pending_muster_* states drew false obligations).
            no_pending = True
            for attr_name in dir(state):
                if not attr_name.startswith("pending"):
                    continue
                try:
                    val = getattr(state, attr_name, None)
                except Exception:
                    continue
                if callable(val):
                    continue
                if val:  # truthy = a pending decision/chain is active
                    no_pending = False
                    break
            has_char_die = False
            try:
                for d in state.dice.get(FP, []):
                    if getattr(d, "used", False):
                        continue
                    f = getattr(d, "current", None)
                    if f in (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST):
                        has_char_die = True
                        break
            except Exception:
                has_char_die = False

            if (hidden and not mordor_done and not corrupt_out
                    and no_pending and has_char_die
                    and "fellowship_move" not in offered_types):
                out.append(
                    "fellowship_move must be offered: Fellowship is hidden, "
                    "an unused Character/Will die exists, and no pending "
                    "decision blocks it (CURRENT_RULES s2)")
    except Exception:
        return out
    return out


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import game_state

    class A(object):
        """Duck-typed Action (NOT imported from ai.py)."""
        def __init__(self, die_idx, action_type, params=None):
            self.die_idx = die_idx
            self.action_type = action_type
            self.params = params or {}

    b = board.load_board()
    st = game_state.setup_initial_game(b)

    # give FP/SH some rolled faces for die-face tests
    st.dice[FP][0].current = ActionDieFace.CHARACTER
    st.dice[FP][1].current = ActionDieFace.MUSTER
    st.dice[FP][2].current = ActionDieFace.EVENT
    st.dice[FP][3].current = ActionDieFace.WILL_OF_WEST
    st.dice[SH][0].current = ActionDieFace.ARMY
    st.dice[SH][1].current = ActionDieFace.CHARACTER
    st.dice[SH][2].current = ActionDieFace.EYE

    tests = [
        ("eye_allocate legal (0<=cap)",
         SH, A(-1, "eye_allocate", {"count": 0})),
        ("eye_allocate ILLEGAL (99>cap)",
         SH, A(-1, "eye_allocate", {"count": 99})),
        ("advance_political Elves (box0)",
         FP, A(1, "advance_political", {"nation": Nation.ELVES})),
        ("advance_political Sauron (box2 active -> allowed)",
         SH, A(-1, "advance_political", {"nation": Nation.SAURON})),
        ("army_attack Gondor NOT at war (ILLEGAL)",
         FP, A(3, "army_attack",
               {"from": "osgiliath", "to": "minas morgul", "side": "Free Peoples"})),
        ("army_move Event die (ILLEGAL face)",
         FP, A(2, "army_move",
               {"from": "minas tirith", "to": "osgiliath",
                "n_regular": 1})),
        ("fellowship_move hidden + Character (legal)",
         FP, A(0, "fellowship_move", {})),
        ("muster_recruit Gondor in Minas Tirith (legal region)",
         FP, A(1, "muster_recruit",
               {"nation": Nation.GONDOR, "region": "minas tirith",
                "type": "regular"})),
        ("muster_recruit Gondor in Edoras (ILLEGAL: Rohan region)",
         FP, A(1, "muster_recruit",
               {"nation": Nation.GONDOR, "region": "edoras",
                "type": "regular"})),
        ("muster_minion Witch-king too early (ILLEGAL)",
         SH, A(1, "muster_minion", {"minion": "witch_king",
                                    "region": "morannon"})),
        ("nazgul_move into FP-held region (ILLEGAL)",
         SH, A(1, "nazgul_move",
               {"from": "minas morgul", "to": "minas tirith", "count": 1})),
        ("crown_aragorn on non-Will die (ILLEGAL)",
         FP, A(0, "crown_aragorn", {})),
        ("elven_ring change to Will (ILLEGAL)",
         FP, A(-1, "elven_ring",
               {"target_die_idx": 1, "old_face": "Muster",
                "new_face": "Will of the West"})),
        ("draw_card with Event die (legal)",
         FP, A(2, "draw_card", {"deck": "fp_character"})),
        ("play_event with EMPTY hand (ILLEGAL)",
         FP, A(2, "play_event", {"card_id": "x"})),
        ("unknown action_type -> None",
         FP, A(-1, "totally_made_up", {})),
    ]

    print("=== check_action smoke ===")
    for label, sd, act in tests:
        v, r = check_action(st, sd, act)
        print("%-52s -> %-5s : %s" % (label, str(v), r))

    # now give FP a card so play_event reaches the 'partial/None' path
    st.hands[FP].hand.append("fp_char_gandalf")
    v, r = check_action(st, FP, A(2, "play_event", {"card_id": "fp_char_gandalf"}))
    print("%-52s -> %-5s : %s"
          % ("play_event with a card in hand (-> None)", str(v), r))

    print()
    print("=== 2026-07-20 rulings: table-kill payments / siege_trim / mouth_move ===")
    folly = game_state.PersistentCard(
        card_id="sh_strat_folly", card_name="Denethor's Folly", side=SH)
    worm = game_state.PersistentCard(
        card_id="sh_strat_worm", card_name="Wormtongue", side=SH)
    st.persistent_cards.add_card(folly)
    st.persistent_cards.add_card(worm)
    p20 = [
        ("Folly 'character' pay, NO hero in MT (ILLEGAL)",
         FP, A(0, "discard_table_card",
               {"card_name": "Denethor's Folly", "payment": "character"})),
    ]
    for label, sd, act in p20:
        v, r = check_action(st, sd, act)
        print("%-52s -> %-5s : %s" % (label, str(v), r))
    # put Gandalf the Grey into the Minas Tirith army
    mt_army = None
    for _k, _a in st.armies.items():
        if _a.region.key == "minas tirith":
            mt_army = _a
            break
    mt_army.leaders.append(game_state.Companion(
        name="Gandalf the Grey", level=3, leadership=1))
    p20b = [
        ("Folly 'character' pay, Gandalf in MT (legal)",
         FP, A(0, "discard_table_card",
               {"card_name": "Denethor's Folly", "payment": "character"})),
        ("'character' pay vs non-Folly card (ILLEGAL)",
         FP, A(0, "discard_table_card",
               {"card_name": "Wormtongue", "payment": "character"})),
        ("'will' pay on Character die (ILLEGAL)",
         FP, A(0, "discard_table_card",
               {"card_name": "Wormtongue", "payment": "will"})),
        ("'will' pay on Will die (legal)",
         FP, A(3, "discard_table_card",
               {"card_name": "Wormtongue", "payment": "will"})),
        ("'ring' pay with rings available (legal)",
         FP, A(0, "discard_table_card",
               {"card_name": "Wormtongue", "payment": "ring"})),
        ("siege_trim without pending combat (-> None)",
         FP, A(-1, "siege_trim", {"elites_back": 1, "excess": 2})),
        ("siege_trim elites_back > excess (ILLEGAL)",
         FP, A(-1, "siege_trim", {"elites_back": 3, "excess": 2})),
        ("mouth_move Mouth not in play (ILLEGAL)",
         SH, A(1, "mouth_move",
               {"from": "minas morgul", "to": "gorgoroth"})),
    ]
    for label, sd, act in p20b:
        v, r = check_action(st, sd, act)
        print("%-52s -> %-5s : %s" % (label, str(v), r))
    st.pending_combat = {"defender_region": "erebor"}
    v, r = check_action(st, FP, A(-1, "siege_trim",
                                  {"elites_back": 1, "excess": 2}))
    print("%-52s -> %-5s : %s"
          % ("siege_trim during pending combat (legal)", str(v), r))
    st.pending_combat = None
    st.mouth_of_sauron.in_play = True
    for label, act in (
            ("mouth_move range 2 on Character die (legal)",
             A(1, "mouth_move", {"from": "minas morgul", "to": "gorgoroth"})),
            ("mouth_move range >3 (ILLEGAL)",
             A(1, "mouth_move", {"from": "minas morgul", "to": "grey havens"})),
            ("mouth_move on Army die (ILLEGAL)",
             A(0, "mouth_move", {"from": "minas morgul", "to": "gorgoroth"})),
            ("mouth_move into FP-army region w/in 3 (legal: no dest bar)",
             A(1, "mouth_move", {"from": "minas morgul", "to": "osgiliath"}))):
        v, r = check_action(st, SH, act)
        print("%-52s -> %-5s : %s" % (label, str(v), r))
    st.mouth_of_sauron.in_play = False

    print()
    print("=== eye_allocate min-1 carve-out (0 companions) ===")
    saved_comps = st.fellowship.companions
    st.fellowship.companions = []
    for cnt in (0, 1, 2):
        v, r = check_action(st, SH, A(-1, "eye_allocate", {"count": cnt}))
        print("eye_allocate count=%d, 0 companions -> %-5s : %s"
              % (cnt, str(v), r))
    st.fellowship.companions = saved_comps

    print()
    print("=== required_offers smoke ===")
    # Fellowship hidden, unused Character die (idx0), no pending -> expect
    # fellowship_move obligation when menu omits it.
    menu_missing = [A(1, "muster_recruit", {}), A(0, "advance_political", {})]
    print("menu missing fellowship_move:", required_offers(st, FP, menu_missing))
    menu_has = [A(0, "fellowship_move", {})]
    print("menu has fellowship_move   :", required_offers(st, FP, menu_has))
    # ANY truthy pending_* attribute suppresses obligations entirely.
    for pf, pv in (("pending_card_option", {"card_id": "x"}),
                   ("pending_army_move_from", "osgiliath"),
                   ("pending_muster_region", "minas tirith")):
        setattr(st, pf, pv)
        print("with %s set             :" % pf,
              required_offers(st, FP, menu_missing))
        setattr(st, pf, None)
    print()
    counts = {"full": 0, "partial": 0, "none": 0}
    for lvl in ORACLE_COVERAGE.values():
        counts[lvl] = counts.get(lvl, 0) + 1
    print("ORACLE_COVERAGE counts:", counts)
