"""Encoding generation v53 (R2-D5-V53; USER D5 2026-10-22 "implement all
d5, including retraining" - retraining itself waited for D6).

v53 = v52 with two APPENDED tails (the project's append-at-tail
convention): every v52 column keeps its exact position, meaning and value,

    encode_state (v53)  == encode_state (v52)  ++ state_tail   (98)
    encode_action (v53) == encode_action (v52) ++ action_tail  (42)

so a v52 vector is the bit-identical prefix of its v53 vector - with ONE
documented exception (R3-V53-FOLLOWONS, rescan v040): the prefix's
overstack-excess column uses the real stacking cap under v53
(event_effects._stack_cap = game_logic.army_stack_cap) and the pre-v040
flag-only cap under v52 (event_effects._stack_cap_flag_only); the two differ
only while a besieged-stronghold flag is stale (the entry holds no besieger),
which cleanup_sieges prevents in play.  Selected by
WOTR_ACT_GEN=v53 (ai.py reads it once at import).

CHECKPOINTS (R17-DOCSTRINGS, 2026-10-27).  v53 checkpoints exist: the D6
retrain made them, and the D7 release (2026-10-26) deployed a v53 pair (the
live head's pair and generation: wotr_bridge/launch_head.py NET_TAG /
NET_GEN).  A v52 checkpoint under v53: by default ai.load_value_net refuses
it (observation / action contracts and dims differ; it raises, never pads).
Under the explicit opt-in WOTR_ADMIT_GENERATIONS=v52, a v52 checkpoint whose
contract stamp matches v52 exactly loads as a
mixed_generation.GenerationViewNet at its own v52 widths and reads the v52
prefix of every v53 vector (the one exception above included).

STATE TAIL (encoding side's perspective; offsets relative to the tail start,
absolute = 4647 + offset)
  [HAND_OFF : +96]   A1-03  1.0 for every card index (ai._get_card_idx_map,
                     0..95) in the ENCODING side's own hand - all of it,
                     so cards beyond the six embedding slots are visible.
                     Own private knowledge only (ruling 1.261): placeholder
                     ids that are not real cards (live human-hand
                     HU_<side>_... ids) light nothing, exactly like the six
                     slots.
  [THREAT_OFF : +2]  A1-02  exact per-target threat counts:
                     [0] my VP settlements with an ENEMY army unit in the
                         settlement or in a region adjacent to it,
                         / number of my VP settlements (12 FP, 10 Shadow);
                     [1] enemy VP settlements with one of MY army units in
                         or adjacent to it, / number of enemy settlements.
                     (The v52 slots 'my/en targets threatened' used the
                     nearest-ANY-target distance and saturate; they stay
                     unchanged in the prefix.)

ACTION TAIL (offsets relative to the tail start, absolute = 157 + offset)
  [ABILITY_OFF]      A1-01  1.0 on an advance_political row carrying
                     params['companion_ability'] (the Legolas / Gimli /
                     Boromir advance, which discards Threats and Promises -
                     rulings 1.106 / 1.107 / 1.125); 0 on the plain advance.
  [FIGURE_OFF : +13] named figures (ai._ACTION_CHARACTERS order):
                     advance_political -> the companion whose ability it is;
                     army_move / victor_advance -> every Companion / Minion
                     the action actually moves (E5-01).
  [LEADER_OFF : +8]  E5-01 plain Leaders the action moves, per nation
                     (NATIONS order), count / LEADER_NORM.
  [REGULAR_OFF : +8] E5-01 Regulars the action moves, per nation, / UNIT_NORM.
  [ELITE_OFF : +8]   E5-01 Elites the action moves, per nation, / UNIT_NORM.
                     The moved figures are read from the EXECUTOR's own
                     selection: army_move -> army_move_participants.selected
                     with the same border predicate and second-leg
                     preexisting view the executor builds; victor_advance ->
                     the battle participants (army_participants.
                     partition_battle_army) filled exactly as
                     game_logic.move_army fills the counts (unit_set /
                     leader_set rows name their figures).  Nazgul stay on
                     the v52 composition column (fungible, ruling 1.128).
                     R4-TDAAN-SPLIT-UNIT-IDENTITY (D5 E3-02): a Through a
                     Day and a Night card_option row fills these four lanes
                     too, with the figures it moves - the handler's own
                     route and selection (event_effects._tdaan_move_roster:
                     _tdaan_row_route + _direct_move_pick, unit_set rows
                     name their units).  Same lanes, same widths; the v52
                     prefix is untouched (v52 merges two rows that differ
                     only in unit_set - the 'E3-02' alias class).
                     R4B-WOSES-SPLIT-ROWS: a Paths of the Woses card_option
                     row fills them the same way (event_effects.
                     _woses_move_roster: the handler's _direct_move_pick,
                     its split grid's unit_set rows name their units).
                     R17-ENG-SIEGETRIM (ruling 1.181): a siege_trim row
                     naming its returned units ('unit_set', one row per
                     nation/kind set when a count fills more than one way)
                     fills the REGULAR / ELITE lanes with those units,
                     / UNIT_NORM (ai._v53_siege_trim_lanes, called by
                     ai.encode_action after action_tail; native
                     encoding_v53.inc v53_action_tail).  A siege_trim row
                     without a set keeps the all-zero tail; the v52 prefix
                     merges the rows of one k (the pre-v53 alias).
  [VIA_OFF : +4]     E3-02 Through a Day and a Night route rows
                     (card_option with choices['via_region']):
                     [0] 1.0 (the row names its walked-through region);
                     [1] (REGION_INDEX[via] + 1) / NUM_REGIONS;
                     [2] 1.0 when walking through captures an empty
                         enemy-controlled settlement;
                     [3] 1.0 when the entry activates a Shadow nation
                     ([2]/[3] = event_effects._tdaan_walk_effects, the two
                     consequences the handler applies).  All four 0 on a
                     row without via_region.

All other action types (and siege_trim rows without a unit_set) keep an
all-zero tail.  Values survive the corpus's float16 packing (multiples of
1/4, 1/10, 1/105 steps).
"""
import numpy as np

OBSERVATION_CONTRACT = 'v53_hand_threat_context_v1'
ACTION_CONTRACT = 'v53_identity_route_choices_v1'

HAND_OFF, HAND_DIM = 0, 96
THREAT_OFF, THREAT_DIM = HAND_OFF + HAND_DIM, 2
STATE_TAIL_DIM = THREAT_OFF + THREAT_DIM            # 98

CHARACTERS = (
    'Gandalf the Grey', 'Strider', 'Legolas', 'Gimli', 'Boromir',
    'Meriadoc', 'Peregrin', 'Aragorn', 'Gandalf the White', 'Gollum',
    'Witch King', 'Saruman', 'Mouth of Sauron',
)   # == ai._ACTION_CHARACTERS (asserted by ai at import under v53)
NATIONS = ('GONDOR', 'ROHAN', 'THE_NORTH', 'DWARVES', 'ELVES',
           'ISENGARD', 'SAURON', 'SOUTHRON_EASTERLINGS')
LEADER_NORM = 4.0      # at most 4 plain Leaders of one nation exist
UNIT_NORM = 10.0       # army stacking limit
TDAAN_CARD = 'FREE_PEOPLES_Strategy_Through_a_Day_and_a_Night'
WOSES_CARD = 'FREE_PEOPLES_Strategy_Paths_of_the_Woses'   # R4B-WOSES-SPLIT-ROWS

ABILITY_OFF = 0
FIGURE_OFF = ABILITY_OFF + 1
LEADER_OFF = FIGURE_OFF + len(CHARACTERS)
REGULAR_OFF = LEADER_OFF + len(NATIONS)
ELITE_OFF = REGULAR_OFF + len(NATIONS)
VIA_OFF = ELITE_OFF + len(NATIONS)
VIA_DIM = 4
ACTION_TAIL_DIM = VIA_OFF + VIA_DIM                 # 42

# action types whose tail can be non-zero (the rest are all-zero)
TAIL_ACTION_TYPES = frozenset(('advance_political', 'army_move',
                               'victor_advance', 'card_option'))


# ---------------------------------------------------------------------------
# state tail
# ---------------------------------------------------------------------------

def _unit_regions(ai, state):
    regions = {ai.Side.FREE_PEOPLES: set(), ai.Side.SHADOW: set()}
    for akey, army in state.armies.items():
        for u in army.units:
            regions[ai._side_of_nation(u.nation)].add(ai.sanitize(akey))
    return regions


def threat_counts(ai, state, side):
    """A1-02: (my targets threatened, enemy targets threatened) as exact
    counts - a target is threatened when an opposing army UNIT stands in
    it or in a region adjacent to it (pairwise, per target)."""
    enemy = ai.Side.SHADOW if side == ai.Side.FREE_PEOPLES else ai.Side.FREE_PEOPLES
    regions = _unit_regions(ai, state)
    adj = ai.board.adj
    if side == ai.Side.FREE_PEOPLES:
        mine, theirs = ai._FP_VP_TARGETS, ai._SH_VP_TARGETS
    else:
        mine, theirs = ai._SH_VP_TARGETS, ai._FP_VP_TARGETS

    def count(targets, occupied):
        return sum(1 for t, _ in targets
                   if t in occupied or any(nb in occupied for nb in adj.get(t, ())))
    return count(mine, regions[enemy]), count(theirs, regions[side]), len(mine), len(theirs)


def state_tail(ai, state, side):
    out = np.zeros(STATE_TAIL_DIM, dtype=np.float32)
    card_map = ai._get_card_idx_map()
    hand = state.hands.get(side)
    for cid in (hand.hand if hand is not None else ()):
        idx = card_map.get(cid, -1)
        if idx >= 0:
            out[HAND_OFF + idx] = 1.0
    mine, theirs, n_mine, n_theirs = threat_counts(ai, state, side)
    out[THREAT_OFF] = mine / n_mine
    out[THREAT_OFF + 1] = theirs / n_theirs
    return out


# ---------------------------------------------------------------------------
# action tail
# ---------------------------------------------------------------------------

def _figure_index(ai, name):
    idx = ai._ACTION_CHAR_IDX.get(ai.sanitize(str(name)))
    if idx is None:
        raise ValueError('v53 encoding: unknown named figure %r' % (name,))
    return idx


def _acting_side(ai, params, side):
    tag = params.get('side')
    if tag == 'fp':
        return ai.Side.FREE_PEOPLES
    if tag == 'sh':
        return ai.Side.SHADOW
    return side


def _army_move_context(ai, state, side, params):
    """The count-independent half of army_move_roster: (source view, acting
    side, border predicate), or None when the source has no army.  Reads
    only params 'from' / 'to' / 'side' and the state."""
    from army_move_participants import preexisting
    src = ai._armies_lookup(state, params.get('from'))
    if src is None:
        return None
    act_side = _acting_side(ai, params, side)
    frm = ai.sanitize(str(params.get('from')))
    to = ai.sanitize(str(params.get('to')))
    nation_ok = None
    if frm != to:
        nation_ok = ai._border_pred9(
            state, [u for u in src.units if ai._side_of_nation(u.nation) == act_side],
            src.leaders, act_side, to)
    source = src
    if (state.pending_army_move_die is not None
            and frm == ai.sanitize(state.pending_army_move_from or '')):
        source = preexisting(source, state.pending_army_move_preexist or {})
    return source, act_side, nation_ok


def army_move_roster(ai, state, side, params):
    """The (units, leaders) an army_move moves - the executor's own
    selection (ai._execute_action 'army_move': _border_pred9 over the full
    source army, the preexisting view on the second Army-die leg, then
    army_move_participants.selected).  None when the source has no army."""
    from army_move_participants import selected
    context = _army_move_context(ai, state, side, params)
    if context is None:
        return None
    source, act_side, nation_ok = context
    moving = selected(source, act_side, params, nation_ok)
    return list(moving.units), list(moving.leaders)


def army_move_variant_tails(ai, state, side, params, counts):
    """R3-V53-FOLLOWONS batch fast path (search_action_encoding_v53): the
    action tails of army_move rows equal to `params` (a row naming no
    figure set) except n_regular / n_elite = each (n_regular, n_elite) of
    `counts`.  Row i equals action_tail of that row bit for bit:
      * the count-independent context is built once;
      * the executor's own selection (army_move_participants.selected) runs
        once, on `params`: its leaders are every row's leaders (selected
        picks leaders independently of the unit counts);
      * selected takes the first n_elite eligible Elites and the first
        n_regular eligible Regulars in army order, so a row's per-nation
        unit lanes are prefix counts over those two pools.  The pools use
        selected's own filter and are cross-checked against its units for
        `params` (a divergence raises, never encodes)."""
    from army_move_participants import selected
    from combat import side_of_nation
    if params.get('leader_set') is not None or params.get('unit_set') is not None:
        raise ValueError('v53 fast path: rows naming figures are never grouped')
    out = np.zeros((len(counts), ACTION_TAIL_DIM), dtype=np.float32)
    context = _army_move_context(ai, state, side, params)
    if context is None:
        return out
    source, act_side, nation_ok = context
    template = selected(source, act_side, params, nation_ok)
    lanes = np.zeros(ACTION_TAIL_DIM, dtype=np.float32)
    _write_roster(ai, lanes, [], list(template.leaders))
    out[:] = lanes
    units = [u for u in source.units if side_of_nation(u.nation) == act_side
             and (nation_ok is None or nation_ok(u.nation))]
    pools = []
    for kind in (ai.UnitKind.ELITE, ai.UnitKind.REGULAR):
        pool = [u for u in units if u.kind == kind]
        prefix = np.zeros((len(pool) + 1, len(NATIONS)), dtype=np.int64)
        for j, u in enumerate(pool):
            prefix[j + 1] = prefix[j]
            prefix[j + 1, NATIONS.index(u.nation.name)] += 1
        pools.append((pool, prefix))
    (elites, elite_prefix), (regulars, regular_prefix) = pools
    want = elites[:int(params['n_elite'])] + regulars[:int(params['n_regular'])]
    if [id(u) for u in template.units] != [id(u) for u in want]:
        raise RuntimeError('v53 fast path: unit pools disagree with army_move_participants.selected')
    for n_regular, n_elite in counts:
        if not (0 <= n_elite <= len(elites) and 0 <= n_regular <= len(regulars)):
            raise ValueError('Movement exceeds eligible army units')
        if n_regular + n_elite == 0:
            raise ValueError('Army movement requires an eligible unit')
    n_regular = np.fromiter((c[0] for c in counts), dtype=np.int64, count=len(counts))
    n_elite = np.fromiter((c[1] for c in counts), dtype=np.int64, count=len(counts))
    out[:, REGULAR_OFF:REGULAR_OFF + len(NATIONS)] = regular_prefix[n_regular] / UNIT_NORM
    out[:, ELITE_OFF:ELITE_OFF + len(NATIONS)] = elite_prefix[n_elite] / UNIT_NORM
    return out


def victor_advance_roster(ai, state, side, params):
    """The (units, leaders) a victor_advance moves: the battle participants
    filled exactly as game_logic.move_army fills the counts the executor
    passes (ai._execute_action 'victor_advance': n_regular / n_elite,
    move_leaders = n_leaders or 99/0 by take_leaders, move_nazgul = n_nazgul
    or 99/0, chosen_units / chosen_leaders from unit_set / leader_set).
    None when the source has no army or no battle is pending."""
    import game_logic
    from army_participants import partition_battle_army
    from army_move_participants import pick
    pc = getattr(state, 'pending_combat', None)
    src = ai._armies_lookup(state, params.get('from'))
    if src is None or not pc:
        return None
    vside = ai.Side.FREE_PEOPLES if params.get('side') == 'fp' else ai.Side.SHADOW
    frm = ai.sanitize(str(params.get('from')))
    bu, bl, _, _ = partition_battle_army(state, src, frm, vside, pc)
    unit_ids = {id(u) for u in bu}
    leader_ids = {id(l) for l in bl}
    chosen_u = chosen_l = None
    if params.get('unit_set') is not None:
        chosen_u = {id(u) for u in pick(
            [u for u in bu if ai._side_of_nation(u.nation) == vside],
            params['unit_set'], unit=True)}
    if params.get('leader_set') is not None:
        chosen_l = {id(l) for l in pick(
            [l for l in bl if not isinstance(l, ai.Nazgul)
             and not game_logic._is_saruman(l)
             and game_logic._leader_side(l) == vside],
            params['leader_set'])}

    def take(seq, qty, pred):
        moved = []
        for obj in seq:
            if qty and pred(obj) and len(moved) < qty:
                moved.append(obj)
        return moved

    own = [u for u in src.units
           if ai._side_of_nation(u.nation) == vside and id(u) in unit_ids]
    unit_ok = (lambda u: True) if chosen_u is None else (lambda u: id(u) in chosen_u)
    units = (take([u for u in own if u.kind == ai.UnitKind.ELITE],
                  int(params.get('n_elite') or 0), unit_ok)
             + take([u for u in own if u.kind == ai.UnitKind.REGULAR],
                    int(params.get('n_regular') or 0), unit_ok))

    def movable(l):
        return (not game_logic._is_saruman(l) and id(l) in leader_ids
                and game_logic._leader_side(l) == vside
                and (chosen_l is None or isinstance(l, ai.Nazgul) or id(l) in chosen_l))
    take_all = 99 if params.get('take_leaders') else 0
    move_leaders = (int(params['n_leaders']) if params.get('n_leaders') is not None
                    else take_all)
    move_nazgul = (int(params['n_nazgul']) if params.get('n_nazgul') is not None
                   else take_all)
    if move_leaders:
        move_leaders = min(move_leaders, sum(
            1 for l in src.leaders if movable(l) and not isinstance(l, ai.Nazgul)))
    naz_pool = [l for l in src.leaders if movable(l) and isinstance(l, ai.Nazgul)]
    nazgul = naz_pool[:max(0, min(move_nazgul, len(naz_pool)))]
    naz_ids = {id(l) for l in nazgul}
    rest = [l for l in src.leaders if id(l) not in naz_ids]
    ordered = ([l for l in rest if game_logic.leader_must_follow(l)]
               + [l for l in rest if not game_logic.leader_must_follow(l)])
    others = take(ordered, move_leaders,
                  lambda l: movable(l) and not isinstance(l, ai.Nazgul))
    other_ids = {id(l) for l in others}
    leaders = nazgul + [l for l in rest if id(l) in other_ids]
    return units, leaders


def _write_roster(ai, out, units, leaders):
    from army_card_cargo import key
    reg = [0] * len(NATIONS)
    eli = [0] * len(NATIONS)
    ldr = [0] * len(NATIONS)
    for u in units:
        i = NATIONS.index(u.nation.name)
        if u.kind == ai.UnitKind.REGULAR:
            reg[i] += 1
        else:
            eli[i] += 1
    for l in leaders:
        k = key(l)
        if k[0] == 'nazgul':
            continue                      # v52 composition column
        if k[0] == 'named':
            out[FIGURE_OFF + _figure_index(ai, k[1])] = 1.0
        elif k[0] == 'leader':
            ldr[NATIONS.index(k[1])] += 1
        else:
            raise ValueError('v53 encoding: unexpected leader figure %r' % (k,))
    for i in range(len(NATIONS)):
        out[LEADER_OFF + i] = ldr[i] / LEADER_NORM
        out[REGULAR_OFF + i] = reg[i] / UNIT_NORM
        out[ELITE_OFF + i] = eli[i] / UNIT_NORM


def _write_via(ai, out, params, state):
    choices = params.get('choices') or {}
    via = choices.get('via_region') if isinstance(choices, dict) else None
    if via is None:
        return
    if params.get('card_id') != TDAAN_CARD:
        raise ValueError('v53 encoding: via_region on an unsupported card %r'
                         % (params.get('card_id'),))
    vs = ai.sanitize(str(via))
    idx = ai.REGION_INDEX.get(vs)
    if idx is None:
        raise ValueError('v53 encoding: unknown via_region %r' % (via,))
    from event_effects import _tdaan_walk_effects
    captures, activates = _tdaan_walk_effects(state, vs)
    out[VIA_OFF] = 1.0
    out[VIA_OFF + 1] = (idx + 1) / float(ai.NUM_REGIONS)
    out[VIA_OFF + 2] = 1.0 if captures else 0.0
    out[VIA_OFF + 3] = 1.0 if activates is not None else 0.0


def tdaan_roster(ai, state, params):
    """R4-TDAAN-SPLIT-UNIT-IDENTITY: (units, leaders) a Through a Day and a
    Night card_option row moves (event_effects._tdaan_move_roster, the
    handler's own route and selection), or None for any other card_option
    row, a row naming no from/to region, or a row that moves nothing."""
    choices = params.get('choices')
    if params.get('card_id') != TDAAN_CARD or not isinstance(choices, dict):
        return None
    if choices.get('from_region') is None or choices.get('to_region') is None:
        return None
    from event_effects import _tdaan_move_roster
    roster = _tdaan_move_roster(state, choices)
    if roster is None:
        return None
    _mid, units, leaders = roster
    return list(units), list(leaders)


def woses_roster(ai, state, params):
    """R4B-WOSES-SPLIT-ROWS: (units, leaders) a Paths of the Woses
    card_option row moves (event_effects._woses_move_roster, the handler's
    own selection), or None for any other card_option row, a row naming no
    from/to region, or a row that moves nothing."""
    choices = params.get('choices')
    if params.get('card_id') != WOSES_CARD or not isinstance(choices, dict):
        return None
    if choices.get('from_region') is None or choices.get('to_region') is None:
        return None
    from event_effects import _woses_move_roster
    roster = _woses_move_roster(state, choices)
    if roster is None:
        return None
    units, leaders = roster
    return list(units), list(leaders)


def action_tail(ai, action, state, side):
    out = np.zeros(ACTION_TAIL_DIM, dtype=np.float32)
    at = action.action_type
    if at not in TAIL_ACTION_TYPES:
        return out
    p = action.params or {}
    if at == 'advance_political':
        name = p.get('companion_ability')
        if name:
            out[ABILITY_OFF] = 1.0
            out[FIGURE_OFF + _figure_index(ai, name)] = 1.0
    elif at in ('army_move', 'victor_advance'):
        roster = (army_move_roster(ai, state, side, p) if at == 'army_move'
                  else victor_advance_roster(ai, state, side, p))
        if roster is not None:
            _write_roster(ai, out, *roster)
    else:
        roster = tdaan_roster(ai, state, p)
        if roster is None:
            roster = woses_roster(ai, state, p)   # R4B-WOSES-SPLIT-ROWS
        if roster is not None:
            _write_roster(ai, out, *roster)
        _write_via(ai, out, p, state)
    return out


def cache_component(ai, action, state, side):
    """Extra encoding-cache key for the state-dependent v53 lanes: () for
    an action whose tail is constant zero, else the tail bytes (the moved
    roster depends on WHICH figures stand at the source, the border
    predicate and the battle partition; the via consequences on settlement
    control and political activity - none of which the v52 key carries)."""
    if action.action_type not in TAIL_ACTION_TYPES:
        return ()
    return (action_tail(ai, action, state, side).tobytes(),)
