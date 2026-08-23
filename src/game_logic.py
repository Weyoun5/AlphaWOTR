# game_logic.py – explicit-die-required version (2025-06-29)
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import board
import game_state as gs
from board import sanitize

# Army stacking limit: max 10 units per army (leaders don't count)
MAX_ARMY_UNITS = 10

# fix 98: teacher-installed evidence gate for the Balrog-of-Moria
# one-shot extra hunt tile (None = self-play, rules-default fire)
BALROG_TILE_GATE = None

# Siege stacking limit: max 5 army units inside a besieged stronghold
MAX_SIEGE_UNITS = 5
# Military victory thresholds (rulebook p.17). Restored here: the
# 1.179 dead-code deletion ranges accidentally swallowed the
# originals that sat next to the removed functions.
SHADOW_VP_VICTORY = 10
FP_VP_VICTORY = 4

# Mordor's interior regions (USER RULING 2026-07-20b, w5_g2 T9): the
# FELLOWSHIP pawn never region-steps into these — they are reached
# only via the Mordor track after a turn-start gateway declaration
# (1.155). Armies DO legitimately occupy them (1.156) — this set must
# only ever filter fellowship placement/declare menus, never
# get_reachable_regions (shared with army movement).
_MORDOR_INTERIOR = frozenset((sanitize('gorgoroth'),
                              sanitize('barad dur'),
                              sanitize('nurn')))


# ---------------------------------------------------------------------------
# RANDOMNESS FUNNELS (session 33, client-built-ins rework).
#
# Every d6 batch and every hunt-tile pick in the rules engine goes through
# these two module-level helpers.  Their default bodies are byte-for-byte
# the code they replaced (same `random` calls in the same order, so seeded
# runs and the corpus canary are unaffected).  The LIVE bridge
# (wotr_bridge/ai_move_server.py) monkey-patches them so that, during a
# REAL game advance, the values come from the Java client's own RNG
# (roll1..5 / hunt handlers) instead of Python's — per the user mandate
# that all randomness be executed by the client's built-in functions.
# MCTS simulation clones keep using the default bodies (the patch is
# scoped to real advances only).
def roll_d6_batch(n: int, ctx: str = '', side=None) -> List[int]:
    """Roll *n* six-sided dice.  ctx is a human label for the batch
    (hunt roll / combat roll / ...) used by the live bridge's logs.
    `side` is WHOSE dice these are (G1, USER live 2026-08-18: each side
    rolls its own).  Optional and ignored here; the live bridge's
    funnel forwards it into the directive so the human is asked to roll
    his own instead of the AI ordering his client to."""
    return [random.randint(1, 6) for _ in range(n)]


def pick_hunt_tile(state: gs.GameState):
    """Pick (but do not remove) one random tile from state.hunt_pool.
    Callers remove it themselves — exactly the random.choice they used
    to inline."""
    return random.choice(state.hunt_pool)


def pop_hunt_tile(state: gs.GameState):
    """Draw the TOP tile of the (shuffled-at-init) hunt pool — exactly
    the state.hunt_pool.pop() the hunt paths used to inline.
    INVARIANT: every hunt draw must route through this funnel — the
    rendered teacher forces tiles by pool replacement, the hunt_dist
    kernel wraps it to enumerate outcomes, and the LIVE bridge patches
    it so the client's own hunt handler supplies the tile."""
    return state.hunt_pool.pop()


def pick_random_companion(candidates, ctx: str = ''):
    """Pick one random companion from *candidates* (hunt casualty draw,
    sacrifice, Foul Thing victim, Lure of the Ring...).  Default body is
    byte-for-byte the random.choice these sites inlined (same call, same
    order — seeded runs unaffected).  A NAMED funnel so an external
    randomness owner (the live driver: the CLIENT performs random
    companion draws) patches ONE site instead of sniffing random.choice
    by sequence type.  `ctx` labels the draw for the driver's directive.
    Teacher-side stream-compat consumes (the forced-victim paths that
    burn one draw for seeded alignment) stay on raw random.choice on
    purpose: they are not real picks and must never reach a client."""
    return random.choice(candidates)


def pick_random_card(candidates, ctx: str = ''):
    """Pick one random CARD id from *candidates* (WWSAT random FP-hand
    Character discard; any future random card pick).  Same contract as
    pick_random_companion: default body byte-for-byte the random.choice
    it replaced; a named funnel so the live driver patches ONE site
    (the client performs the random discard and its rendered line
    carries NO identity — blinding holds by construction)."""
    return random.choice(candidates)

# Shadow minion helper class
@dataclass
class Minion(gs.Leader):
    # Canonical names: "Saruman", "Witch King", "Mouth of Sauron" —
    # remove_orphaned_leaders matches these EXACTLY.
    name: str


def _is_saruman(ld) -> bool:
    """True if this board figure is Saruman.

    Official rule: Saruman is always in Orthanc and NEVER moves. Every
    path that transports leaders (army moves, battle advances, retreats,
    nazgul/minion sweeps) must exclude him. He is only removed from the
    board when eliminated (Orthanc falls / card effect)."""
    return (type(ld).__name__ == 'Minion'
            and 'saruman' in str(getattr(ld, 'name', '')).lower())


def saruman_in_play(state: gs.GameState) -> bool:
    """Saruman figure present on the board (same test combat.setup_combat
    uses for the Isengard-elite leadership bonus)."""
    return any(_is_saruman(l)
               for a in state.armies.values() for l in a.leaders)


def army_has_leader_figure(state: gs.GameState, army: gs.Army, side=None) -> bool:
    """Does this army count as 'containing a Leader' (Character-die army
    action requirement)?

    Saruman's official ability: while Saruman is in play, Isengard ELITE
    units are considered to be both Army units AND Leaders — so a
    leaderless Isengard-elite army may move/attack with a Character die
    (and gets leader re-rolls, handled in combat.setup_combat).
    Previously this only worked by the ENGINE BUG that carried Saruman
    out of Orthanc with the army."""
    # W5 B-1 fix (2026-07-20): side-filter — in a CO-LOCATED besieged
    # entry the leaders list holds BOTH sides' figures, and the old
    # bare truthiness let a leaderless side borrow the ENEMY's leader
    # for Character-die actions. side=None keeps legacy behavior.
    if side is None:
        if army.leaders:
            return True
    else:
        from combat import _combat_leader_side
        if any(_combat_leader_side(l) == side for l in army.leaders):
            return True
    if saruman_in_play(state):
        return any(u.nation == gs.Nation.ISENGARD
                   and u.kind == gs.UnitKind.ELITE
                   for u in army.units)
    return False

# Side helpers & mandatory die-picker
_SHADOW_NATIONS = {
    gs.Nation.ISENGARD, gs.Nation.SAURON, gs.Nation.SOUTHRON_EASTERLINGS
}
def _side_of(n: gs.Nation) -> gs.Side:
    # ONE implementation: combat.side_of_nation is the authority
    # (lazy import dodges the module cycle; _SHADOW_NATIONS above is
    # kept only for set-membership call sites).
    from combat import side_of_nation
    return side_of_nation(n)


def _pick_die(state: gs.GameState,
              side: gs.Side,
              wanted: Tuple[gs.ActionDieFace, ...],
              die_idx: Optional[int]) -> gs.ActionDie:
    """
    Consume *exactly* the die at index *die_idx* in the side’s pool.
    • The die must be unused and show one of *wanted* faces
      (FP WILL is wildcard).
    """
    if die_idx is None:
        raise RuntimeError("die_idx must be supplied – explicit die use is required")

    dice = state.dice[side]
    try:
        die = dice[die_idx]
    except IndexError:
        raise RuntimeError("die_idx out of range")

    valid = (die.current in wanted) or (
        side is gs.Side.FREE_PEOPLES and die.current == gs.ActionDieFace.WILL_OF_WEST)
    waiver_rescue = False
    if not valid and not die.used \
            and ring_waiver_active(state, side) \
            and ring_waiver_permits(side, die.current):
        # A7 legality expansion: a paid Elven Ring widens what this die
        # may fund.  ai._consume_die_for_face (ai.py:10806) already
        # honors the waiver; _pick_die refusing the die its own menu
        # offered was the WAVE-F engine gap (muster_minion path,
        # game_logic.py:4198).  The waiver is spent ONLY when it
        # actually rescued the consumption (ring_waiver_spend contract).
        valid = True
        waiver_rescue = True
    if die.used or not valid:
        raise RuntimeError("Selected die cannot be used for this action")

    die.used = True
    if waiver_rescue:
        ring_waiver_spend(state, side)
    return die

# Dice rolling / Eye allocation / turn reset
def roll_action_dice(state: gs.GameState, *, pool: gs.Side) -> List[gs.ActionDieFace]:
    faces = []
    for d in state.dice[pool]:
        if not d.used:
            faces.append(d.roll())
            if pool is gs.Side.SHADOW and d.current == gs.ActionDieFace.EYE:
                d.used = True  # Mark eye as used so it's removed from active dice
                state.hunt_box.append(d)

    # Mark that dice have been rolled (prevents recovery)
    state.dice_rolled = True
    return faces


def adopt_rolled_faces(state: gs.GameState, assignments):
    """Adopt EXTERNALLY-rolled action-die faces (client-owned randomness,
    sided play).  `assignments`: iterable of (side, die_index, face).

    Owns the rolled-Eye rule exactly as roll_action_dice does: a Shadow
    Eye is marked used and routed to the Hunt Box (a rolled Eye left in
    the action pool has no legal actions and wedges the search).  Dice
    already carrying a face or already used are SKIPPED — idempotent
    under re-scan of a feed window.  Sets state.dice_rolled once any
    face lands.  Returns [(side, die_index)] of dice routed to the
    Hunt Box.  Face-value/grammar mapping is the DRIVER's job; this
    takes gs.ActionDieFace values only."""
    eyes = []
    adopted = 0
    for side, idx, face in assignments:
        dice = state.dice.get(side, [])
        if not 0 <= idx < len(dice):
            raise RuntimeError(
                "adopt_rolled_faces: die index %r out of range for %s"
                % (idx, side))
        d = dice[idx]
        if d.used or d.current is not None:
            continue
        d.current = face
        adopted += 1
        if side is gs.Side.SHADOW and face is gs.ActionDieFace.EYE:
            d.used = True
            if d not in state.hunt_box:
                state.hunt_box.append(d)
            eyes.append((side, idx))
    if adopted:
        state.dice_rolled = True
    return eyes

def allocate_eyes(state: gs.GameState, eyes: int, *, indices: List[int]):
    """
    Phase-3 allocation: flip the specific Shadow dice in *indices* to Eye
    faces and move them to the Hunt Box.

    • len(indices) must equal *eyes*
    • Each referenced die must be currently unused.
    """
    if len(indices) != eyes:
        raise RuntimeError("indices length must equal eyes argument")

    dice_pool = state.dice[gs.Side.SHADOW]

    for idx in indices:
        try:
            die = dice_pool[idx]
        except IndexError:
            raise RuntimeError(f"Die index {idx} out of range")

        if die.used:
            raise RuntimeError(f"Die {idx} already used")

        die.current = gs.ActionDieFace.EYE
        die.used = True
        state.hunt_box.append(die)
    state.eyes_allocated += eyes


def begin_turn_reset(state: gs.GameState, new_turn, deal_cards=True,
                     bonus_dice=True):
    """THE TURN BOUNDARY'S RULES, WITH THE TURN NUMBER SUPPLIED BY THE
    CALLER.  Returns True when the Mordor corruption penalty just ended
    the game (ring victory at corruption 12).

    WHY THIS EXISTS (A12, 2026-08-19).  There were THREE implementations
    of the turn boundary - `next_turn` (engine, reached only by AUTO),
    `LogReplayer._handle_turn_change` (parser), and
    `AiBrain._sided_turn_rollover` (the bridge, i.e. THE MODE THE USER
    PLAYS).  Driven from one identical position, the bridge's failed to
    reset or apply EIGHT things and the parser's SIX:

        fp/sh_ring_used_this_turn, sh_mouth_muster_used_this_turn,
        fp/sh_token_used_this_turn   -> once-per-turn permissions that,
             once spent, never cleared again for the rest of the game;
        dice_rolled, eyes_allocated  -> 3 eyes carried into the next
             turn's hunt;
        the Mordor inactivity corruption penalty (rulebook p.19) ->
             the Fellowship sits in Mordor for free;
        and `calculate_bonus_dice`, measured as Shadow pool 9 (engine)
             vs 7 (bridge) with Saruman and the Witch-king in play -
             which is USER live bug J1 verbatim.
    The stuck ring flag is USER bug C2's second mechanism: driven at
    `use_elven_ring`'s own interface, the stuck flag REFUSES ("Already
    used an Elven Ring this turn") where the cleared flag ACCEPTS.

    WHY IT IS A SPLIT AND NOT "PATCH THE BRIDGE TO RESET EIGHT MORE
    THINGS".  That is the member-by-member repair this project has been
    paid for twice, and it would leave three implementations of one
    rule in place - which is the actual violation.  `next_turn` couples
    jobs that are separable, and the reason a third implementation grew
    is that the other two callers need SOME of them and must not have
    the others:

      * the per-turn RESET            - all three need it;
      * `state.turn += 1`             - **only AUTO may do this**.  In
        sided play the human's marker gesture is the source of truth
        for the turn number, and the parser takes it from the log.  So
        the number is a PARAMETER here, never an increment.
      * the Mordor penalty, which can END THE GAME - all three need it,
        and the boolean is returned rather than acted on, because each
        caller delivers an end-of-game differently.
      * the four turn-start DRAWS and the hand-limit opener - AUTO only.
        In sided and in replay the CLIENT draws and the log carries it;
        an engine draw here would double every turn-start card.  Hence
        `deal_cards`, which is the only behavioural switch.

    `enforce_settlement_control` deliberately stays in `next_turn`: it
    is guarded to self-play there ("the teacher keeps its certified
    post-combat call pattern") and moving it would change AUTO.
    """
    # (3) MORDOR INACTIVITY PENALTY - rulebook p.19.  It must run
    # BEFORE `fellowship_acted_in_mordor` is cleared below, and before
    # anything else, exactly where `next_turn` always ran it.
    # P0 2026-08-15: corruption 12 here = IMMEDIATE ring victory, so
    # the caller is told and stops the game; ring victories never wait.
    _rollover_won9 = False
    if (state.fellowship.mordor_entered
            and not getattr(state, 'fellowship_acted_in_mordor', False)
            and state.fellowship.corruption < 12):
        state.fellowship.corruption = min(
            12, state.fellowship.corruption + 1)
        _rollover_won9 = state.fellowship.corruption >= 12

    # (2) THE NUMBER IS THE CALLER'S.  AUTO passes turn+1, so its
    # behaviour is unchanged; sided and the parser pass what the client
    # told them.
    state.turn = int(new_turn)

    # (1) THE PER-TURN RESET
    state.eyes_allocated = 0
    # Rulebook p.17 (Hunt Allocation): the Shadow MUST allocate at least one
    # die next turn if the FP player retrieves a die from the Hunt Box now.
    state._fp_used_hunt_box_last_turn = any(
        d.side is gs.Side.FREE_PEOPLES for d in state.hunt_box)
    for d in state.hunt_box:
        d.used, d.current = False, None
    state.hunt_box.clear()
    for pool in state.dice.values():
        for d in pool:
            d.used, d.current = False, None

    # Reset ring usage tracking
    state.fp_ring_used_this_turn = False
    state.sh_ring_used_this_turn = False
    # A7: an unspent legality waiver never crosses a turn boundary.
    state.fp_ring_waiver = False
    state.sh_ring_waiver = False
    state._ring_free_side = None
    state.sh_mouth_muster_used_this_turn = False
    state.fp_token_used_this_turn = False
    state.sh_token_used_this_turn = False
    state.dice_rolled = False

    # Reset Mordor activity tracking for the new turn
    state.fellowship_acted_in_mordor = False

    # Calculate bonus dice from special characters.
    #
    # `bonus_dice=False` IS NOT AN OPTIONAL RULE - IT IS THE HALF THIS
    # FUNCTION CANNOT DO.  Growing the roster is only correct if the
    # matching PHYSICAL die piece is dragged out of the reinforcement
    # box in the same breath, and only the caller that talks to the
    # client can do that.  The bridge already does it properly at its
    # recover step (`_pool_growth_lines`, and OWN SIDE ONLY, because
    # forging the human's drags is exactly what the live protocol
    # forbids).  MEASURED the hard way: my first cut ran this here for
    # the sided caller too, which grew BOTH pools with no piece for
    # either, and the soak wedged on
    #     "STUCK: Shadow dice never rolled: slot 7"
    # - live, that is the engine believing the human has 8 dice while
    # his client shows 7, and the AI nagging him forever.  A rule
    # applied without the emission it depends on is a live defect, not
    # a partial fix.
    if bonus_dice:
        calculate_bonus_dice(state)

    if deal_cards:
        for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            try:
                draw_event_card(state, s, "character")
            except Exception:
                pass  # deck empty
            try:
                draw_event_card(state, s, "strategy")
            except Exception:
                pass
        # USER RULING 2026-08-15: "wtf it is supposed to be a learned
        # decision."  The hand-limit discard is a REAL decision (which
        # card to give up) and must never be taken by the engine.  This
        # site used to auto-trim - first with an unseeded
        # `random.choice`, then deterministically - and either way it
        # PRE-EMPTED the learned pending, so the nets never saw the
        # decision at all on this path.  Now the rollover only OPENS
        # the pending; whoever drives the turn resolves it (self-play
        # MCTS, EngineSession, the GUI's Discard button, or the
        # extractor replaying the human's logged pick).
        open_hand_limit_discard(state)
    return _rollover_won9


def next_turn(state: gs.GameState):
    # METROLOGY 2026-07-27 (w0_g0 WR): a reoccupation of a conquered
    # settlement via a NON-combat path (event-card recruit, retreat)
    # left the enemy_controlled flag stale — once the enemy re-besieged
    # it, enforce_settlement_control's single-side contract skipped the
    # region forever, garrison_side_for kept answering the old
    # conqueror, and the 1.144 no-slipping-through-siege-lines block
    # was defeated.  Heal single-side stale flags at every turn
    # boundary.  Self-play only — the teacher keeps its certified
    # post-combat call pattern.
    import combat as _cmb_nt
    if _cmb_nt.FINAL_HITS_HOOK is None:
        try:
            enforce_settlement_control(state)
        except Exception:
            pass
    # EVERYTHING ELSE IS `begin_turn_reset` (A12, 2026-08-19) - see its
    # docstring for why the boundary was split and which caller needs
    # which half.  AUTO passes turn+1, so this function's behaviour is
    # unchanged: the number is the same, the statement order is the
    # same, and `deal_cards=True` keeps the four turn-start draws and
    # the hand-limit opener exactly where they were.
    #
    # stale-state sweep 2026-07-27: regions_attacked_this_turn AND
    # armies_moved_this_turn DELETED (USER order) - both were legacy
    # trackers for rules that do not exist; the fix-105 teacher branch
    # that read the latter guarded a long-deleted menu gate and is
    # deleted with it.
    #
    # NOTE (2026-07-24): the stay-heal (rulebook p.34, GAME19 T5 L389)
    # is NOT unconditional - healing requires RE-DECLARING in place,
    # a decision.  The parser classifies the lone -1 corruption drag
    # as a fellowship_declare (log_replay._sync_corruption) and the
    # declare path applies the heal; an auto-heal here would heal
    # fellowships that chose not to declare.
    return begin_turn_reset(state, int(state.turn) + 1, deal_cards=True)


HAND_LIMIT = 6            # rulebook: 6 cards in hand at end of turn


def open_hand_limit_discard(state: gs.GameState) -> bool:
    """Open the LEARNED hand-limit discard pending if any hand is over
    the limit.  ONE implementation, shared by every turn path
    (game_logic.next_turn, ai._check_hand_limits and everything that
    calls it).  Resolves ONE card at a time: the executor re-opens the
    pending while a hand is still over, so a 2-card overdraw is two
    separate decisions.  FP is offered first when both sides are over.

    Returns True if a pending is now open."""
    if getattr(state, 'pending_discard', None) is not None:
        return True
    for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        hand = state.hands[s].hand
        state.hands[s].needs_discard = len(hand) > HAND_LIMIT
        if len(hand) > HAND_LIMIT:
            state.pending_discard = {'side': s}
            return True
    return False


def discard_hand_card(state: gs.GameState, side: gs.Side, card_id) -> bool:
    """THE hand-discard primitive: take `card_id` out of `side`'s hand
    and put it in that side's face-down discard pile, then re-derive the
    hand-limit pending.  Every path funnels through here — the learned
    `discard_card` action (ai), the GUI's Discard button
    (discard_card_from_hand) and any card effect that discards from hand
    — so a discarded card can never be annihilated (2026-08-15 card
    conservation wave) and the pending can never be left stale.

    Returns True if the card was found and discarded."""
    hand = state.hands[side].hand
    if card_id not in hand:
        return False
    hand.remove(card_id)
    state.hands[side].discarded_face_down.append(card_id)
    state.hands[side].needs_discard = len(hand) > HAND_LIMIT
    # the pending is CONSUMED, then re-derived: still over the limit (or
    # the other side is) -> a fresh one-card decision, else cleared.
    state.pending_discard = None
    open_hand_limit_discard(state)
    return True


def recover_dice(state: gs.GameState):
    """
    Reset all dice to unrolled state (before rolling).
    Also clears the hunt box.
    Used for the "Recover Dice" button to allow eye allocation before rolling.
    """
    if state.dice_rolled:
        raise RuntimeError("Cannot recover dice after rolling - use for pre-roll allocation only")

    for d in state.hunt_box:
        d.used, d.current = False, None
    state.hunt_box.clear()
    state.eyes_allocated = 0

    for pool in state.dice.values():
        for d in pool:
            d.used, d.current = False, None


_RING_WAIVER_ATTR = {gs.Side.FREE_PEOPLES: 'fp_ring_waiver',
                     gs.Side.SHADOW: 'sh_ring_waiver'}


def ring_waiver_active(state: gs.GameState, side: gs.Side) -> bool:
    """THE ONE ANSWERER for 'does this side hold an unspent Elven-Ring
    legality waiver right now'.  Every component that has to know -
    `ai.get_legal_actions`, `ai._consume_die_for_face`, and the live
    bridge's gesture check - reads it here, so they cannot disagree
    about whether the human paid (A7: one rule had THREE
    implementations, in three different components)."""
    return bool(getattr(state, _RING_WAIVER_ATTR[side], False))


def ring_waiver_permits(side: gs.Side, face) -> bool:
    """USER RULE 2026-08-18, verbatim: the ring expands legality "to any
    non Will of the west die for free people / any die for shadow".
    `face` is the die's ROLLED face (the ring never changes it)."""
    if face is None:
        return False
    if side == gs.Side.FREE_PEOPLES:
        return face != gs.ActionDieFace.WILL_OF_WEST
    return True


def ring_waiver_spend(state: gs.GameState, side: gs.Side) -> None:
    """Consume the waiver.  Called ONLY when the waiver actually rescued
    a die consumption - never on an action the side could pay for
    anyway, because "a ring that is spent and buys nothing" is the
    defect being fixed, not the fix."""
    setattr(state, _RING_WAIVER_ATTR[side], False)


def use_elven_ring(state: gs.GameState, side: gs.Side, die_idx: int,
                   new_face: gs.ActionDieFace) -> str:
    """
    Pay an Elven Ring.

    A7 / USER RULING 2026-08-19: THIS IS A LEGALITY EXPANSION, NOT A
    FACE CHANGE.  The die keeps the face it rolled; the ring raises a
    one-shot waiver that widens which dice may fund the side's next
    action - any non-Will-of-the-West die for the Free Peoples, any die
    for the Shadow.  Anything downstream that reads the die's face still
    sees the original face.

    WHY THE FACE MODEL WAS WRONG, measured rather than argued: the
    client's ring command (`A.java` ringVilya/ringNenya/ringNarya) emits
    a marker walk and an optional flip - no die pid and no face.  Over
    300 real client logs, 540 ring uses produced ONE die-face re-render
    within 6 lines against 2.6 expected by chance, in a client that
    re-renders a die's face 254 times elsewhere.  A face change the
    client never emits is not reconstructible from the rendered log by
    ANY parser, so the parser and the bridge - which both tracked the
    ring as a legality grant with no face change - were reading the
    client correctly and the engine was the component out of step.
    `new_face` is therefore only a RECORD of what the player asked for,
    with ONE exception below.

    THE ONE EXCEPTION - ring to the Eye.  That use has a PHYSICAL
    consequence the client does render: the die walks into the Hunt Box.
    It keeps its face change, because here the face is emitted.

    Rules that survive unchanged:
    - the ring may NEVER produce Will of the West, for either side
      (rulebook Ch.IV; defended twice already - F-4 fuzz seed 9317 and
      amendment 1.178a).  Under the waiver model that reads: a ringed
      die may not fund a Will-of-the-West-only action.
    - the Eye is not a Free Peoples face
    - Shadow cannot ring a die showing an Eye
    - only 1 ring per side per turn
    - after FP uses a ring, Shadow gets it

    Returns a description of what the ring bought.
    """
    # Check if side has a ring
    if side == gs.Side.FREE_PEOPLES:
        if state.fp_elven_rings <= 0:
            raise RuntimeError("No Elven Rings available")
        if state.fp_ring_used_this_turn:
            raise RuntimeError("Already used an Elven Ring this turn")
        if new_face == gs.ActionDieFace.WILL_OF_WEST:
            raise RuntimeError("Cannot change to Will of the West with ring")
        # The Eye is not an FP face: an FP die ringed to Eye would be
        # unspendable (no action validates it) until next_turn.
        if new_face == gs.ActionDieFace.EYE:
            raise RuntimeError("Eye is not a Free Peoples face")
    else:
        if state.sh_elven_rings <= 0:
            raise RuntimeError("No Elven Rings available")
        if state.sh_ring_used_this_turn:
            raise RuntimeError("Already used an Elven Ring this turn")
        # W3 BUG-4 / 1.178a defense-in-depth: Will of the West is not
        # a Shadow face and the ring may never produce it for anyone
        # (the 1.178a fix covered only the ai.py menu).
        if new_face == gs.ActionDieFace.WILL_OF_WEST:
            raise RuntimeError("Cannot change to Will of the West with ring")

    # Get the die
    dice_pool = state.dice[side]
    if die_idx < 0 or die_idx >= len(dice_pool):
        raise RuntimeError(f"Invalid die index {die_idx}")

    die = dice_pool[die_idx]
    if die.used:
        raise RuntimeError("Cannot change a used die")
    if die.current is None:
        raise RuntimeError("Cannot change an unrolled die")

    # Shadow cannot change an Eye
    if side == gs.Side.SHADOW and die.current == gs.ActionDieFace.EYE:
        raise RuntimeError("Shadow cannot change an Eye die with ring")

    old_face = die.current

    # Mark ring as used
    if side == gs.Side.FREE_PEOPLES:
        state.fp_elven_rings -= 1
        state.fp_ring_used_this_turn = True
        # Transfer ring to Shadow
        state.sh_elven_rings += 1
    else:
        state.sh_elven_rings -= 1
        state.sh_ring_used_this_turn = True

    # THE ONE FACE CHANGE THAT SURVIVES: ring to the Eye.  The die walks
    # into the Hunt Box, which is a zone move the client DOES render, so
    # the engine may model it without inventing information.
    if side == gs.Side.SHADOW and new_face == gs.ActionDieFace.EYE:
        die.current = new_face
        die.used = True
        state.hunt_box.append(die)
        return f"Changed {old_face.value} to Eye (moved to Hunt Box)"

    # LEGALITY EXPANSION.  The die keeps its rolled face.
    setattr(state, _RING_WAIVER_ATTR[side], True)
    return (f"Elven Ring paid ({old_face.value} die staged, face kept) "
            f"- any {'non-Will ' if side == gs.Side.FREE_PEOPLES else ''}"
            f"die may fund this action")

# Movement / combat helpers (logic unchanged)

def remove_orphaned_leaders(state: gs.GameState):
    """Remove plain Leaders (EITHER side) alone without combat units
    (rulebook p.8): 'If at any time a Leader is on the map without an
    Army, he is immediately removed.'  Nazgul, Companions and named
    Minions are exempt and may stand alone.
    """
    for rkey in list(state.armies.keys()):
        army = state.armies[rkey]
        if not army.units and army.leaders:
            # Keep Nazgul and Companions/Minions (they can be alone)
            # Remove only FP Leader objects that aren't characters
            remaining = []
            for ld in army.leaders:
                if isinstance(ld, gs.Nazgul):
                    remaining.append(ld)  # Nazgul can be alone
                elif isinstance(ld, gs.Companion):
                    remaining.append(ld)  # Companions move independently
                elif hasattr(ld, 'name') and ld.name in ('Saruman', 'Witch King', 'Mouth of Sauron'):
                    remaining.append(ld)  # Minions can be alone
                else:
                    pass  # Regular Leader removed (no combat units)
            army.leaders = remaining
            # Clean up empty armies
            if not army.units and not army.leaders:
                del state.armies[rkey]


def enforce_settlement_control(state: gs.GameState):
    """Standing settlement-control invariant (F-1/F-3 fix, USER
    2026-07-19): a settlement region occupied by exactly ONE side's
    army units belongs to that side — if the current controller
    differs, run the capture bookkeeping.  Catches every path the
    per-site `_check_settlement_capture` calls missed: a garrison
    wiped out ATTACKING in a sortie (capture only fired on
    defender_destroyed), and battles abandoned mid-allocation when
    the per-turn decision cap ended the turn.  Both left fallen
    strongholds unconquered: no VP, owner kept mustering there.
    Two-side (besieged) and empty regions are untouched; recapture
    and neutral settlements are handled inside
    _check_settlement_capture.  Same stability contract as
    cleanup_sieges (which calls this first).
    """
    from combat import (_check_settlement_capture, garrison_side_for,
                        side_of_nation as _son)
    for k, a in list(state.armies.items()):
        if not a.units:
            continue
        san = sanitize(k)
        meta = board.regions.get(san, {})
        if not (meta.get("is_city") or meta.get("is_stronghold")
                or meta.get("is_town")):
            continue
        sides = {_son(u.nation) for u in a.units}
        if len(sides) != 1:
            continue
        occupier = next(iter(sides))
        if garrison_side_for(state, san) != occupier:
            _check_settlement_capture(state, san, occupier)


def cleanup_sieges(state: gs.GameState):
    """Siege upkeep (iteration 36, self-play engine fix): a stronghold
    stays besieged only while units of BOTH sides occupy its region (the
    engine's co-located siege model — garrison inside the stronghold,
    besieging army in the region).  When the besieging army leaves or
    dies (or the garrison is gone), the siege is OVER and the flag must
    lift — before this fix nothing ever removed a stale flag outside
    _finish_battle, so a besieger dying in another battle left a
    permanent 'siege with no besieger' (2026-07-07 showcase: Minas
    Tirith T5 fp4/sh0 besieged, no adjacent SH army) and later adjacent
    attacks on the garrison inherited is_siege=True ('siege assault
    from adjacent region').  Call at STABLE points only (no
    pending_combat — mid-battle windows legally hold transient
    one-sided compositions).
    """
    enforce_settlement_control(state)
    # fix 93 (game-20 canary, gus833 T4 Lorien/WR): the flag is
    # DERIVABLE state — a stronghold region held by BOTH sides IS a
    # siege in the co-located model.  The teacher's retreat-then-
    # arrival ordering could drop it forever: the retreat sets the
    # flag, the pre-arrival stable point sees a one-sided region and
    # lifts it, and the arrival aux had nothing to re-add.  Re-derive
    # the flag here (ADD direction) before the lift pass.
    from combat import side_of_nation as _son
    bs = getattr(state, 'besieged_strongholds', None)
    if bs is None:
        bs = state.besieged_strongholds = set()
    from board import regions as _regs93
    for _k93, _a93 in state.armies.items():
        _san93 = sanitize(_k93)
        if _san93 in bs or not _a93.units:
            continue
        if not _regs93.get(_san93, {}).get('is_stronghold'):
            continue
        if len({_son(u.nation) for u in _a93.units}) == 2:
            bs.add(_san93)
    for r in list(bs):
        entry = None
        for k, a in state.armies.items():
            if sanitize(k) == sanitize(r):
                entry = a
                break
        sides = {_son(u.nation) for u in entry.units} if entry else set()
        if len(sides) < 2:
            bs.discard(r)
            continue
        # P2-10 (USER sweep 2026-07-20): the 5-unit besieged-garrison
        # cap is now a STANDING invariant, not just a setup/withdrawal
        # check — card effects that add units to a besieged garrison
        # (Cirdan's-Ships class once put one at 7) are re-clamped at
        # the next stable point. Elites kept first, excess removed —
        # excess returns to REINFORCEMENTS (1.181), never casualties.
        from combat import garrison_side_for as _gsf10
        _gside = _gsf10(state, sanitize(r))
        _gar = [u for u in entry.units if _son(u.nation) == _gside]
        if len(_gar) > MAX_SIEGE_UNITS:
            # USER RULING 2026-07-27 (Cirdan/Dol-Amroth double-trim):
            # WHICH excess units return to reinforcements is the
            # OWNER'S learned decision on the self-play path — the
            # R1 stable-point detector below registers the region and
            # opens the '__overstack__' pending; clamping heuristically
            # here raced that pending (7 -> clamp 5 -> stale pending
            # -> 4).  The heuristic clamp remains for the TEACHER
            # (log = truth).
            import combat as _cmb10
            if _cmb10.FINAL_HITS_HOOK is None:
                continue
            kept = sorted(_gar, key=lambda u:
                          0 if u.kind == gs.UnitKind.ELITE else 1)
            excess = kept[MAX_SIEGE_UNITS:]
            _ex_ids = set(map(id, excess))
            entry.units = [u for u in entry.units
                           if id(u) not in _ex_ids]
            # USER RULING 2026-07-20: siege-cap excess returns to
            # REINFORCEMENTS, not casualties — an untracked removal
            # IS a pool return (reinforcements are computed as
            # max - deployed - fp_casualties; same doctrine as
            # resolve_overstacks). Deliberately NO casualty tracking.
    # USER RULING 2026-07-20 (P2-03): Denethor's Folly leaves play the
    # moment Minas Tirith is not under siege — "removed if the siege
    # ends for any reason".  Checked AFTER the re-derive/lift passes so
    # a transiently-dropped flag never discards the card early (audit
    # A4, 2026-07-26); a fully-lifted siege still clears it here.
    _pcards = getattr(state, 'persistent_cards', None)
    if _pcards is not None:
        if sanitize('minas tirith') not in bs:
            _folly = _pcards.get_by_type(
                gs.PersistentCardType.DENETHORS_FOLLY)
            if _folly is not None:
                # CARD CONSERVATION (2026-08-15): discarded face-up
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, _pcards.remove_card(_folly.card_id) or _folly)
    # METROLOGY R1 (2026-07-27, 13 instances of 11 besiegers): the
    # 10-cap trim only ran for producer paths that REGISTERED their
    # region in _overstack_regions — mass-muster/card-move paths that
    # never registered left persistent >10 stacks fighting at full
    # strength.  Standing DETECTOR (mirror of the P2-10 garrison
    # clamp): any side over its per-side cap at a stable point is
    # registered and resolved through the certified 1.112.7 machinery
    # (learned pending on the self-play path).  Self-play only — the
    # teacher's log-truth compositions are ground truth.
    import combat as _cmb_r1
    if _cmb_r1.FINAL_HITS_HOOK is None:
        try:
            from event_effects import (_stack_cap, resolve_overstacks)
            _found_r1 = False
            for _k1, _a1 in state.armies.items():
                _s1 = sanitize(_k1)
                for _sd1 in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
                    _n1 = sum(1 for u in _a1.units
                              if _son(u.nation) == _sd1)
                    if _n1 > _stack_cap(state, _s1, _sd1):
                        _ovr1 = getattr(state, '_overstack_regions',
                                        None)
                        if _ovr1 is None:
                            _ovr1 = state._overstack_regions = set()
                        _ovr1.add(_s1)
                        _found_r1 = True
            if _found_r1 and state.pending_combat is None:
                resolve_overstacks(state, learned=True)
        except Exception:
            pass


def _activate_nation_for_fellowship_declaration(state: gs.GameState, region_key: str):
    """Activate an FP nation when fellowship is declared in its city/stronghold (rulebook p.34)."""
    from board import regions as board_regions, fid_to_name
    reg_info = board_regions.get(region_key)
    if not reg_info:
        return
    parent_id = reg_info.get("parent_id")
    if not parent_id:
        return
    faction_name = fid_to_name.get(parent_id, "").lower()
    nation_map = {
        'gondor': gs.Nation.GONDOR, 'rohan': gs.Nation.ROHAN,
        'elves': gs.Nation.ELVES, 'dwarves': gs.Nation.DWARVES,
        'the north': gs.Nation.THE_NORTH, 'north': gs.Nation.THE_NORTH,
    }
    nation = nation_map.get(faction_name)
    if not nation:
        return
    if not state.political.active.get(nation, False):
        try:
            activate_nation(state, nation, method="fellowship_declared")
        except RuntimeError:
            pass  # Blocked by persistent effect


# Companion-to-nation mapping for political activation (rulebook p.34;
# user ruling 2026-07-03, see data/rules_amendments.md):
# - Gandalf (Grey or White), Aragorn and the hobbits (Merry/Pippin)
#   activate ANY Free Peoples nation (value None = any).
# - Strider activates only The North (Strider is NOT Aragorn).
# - Boromir -> Gondor, Legolas -> Elves, Gimli -> Dwarves.
# A companion ending movement in a city/stronghold of a (matching) nation
# activates that nation.
COMPANION_NATIONS = {
    'Gandalf the Grey': None,
    'Gandalf the White': None,
    'Gandalf': None,
    'Aragorn': None,
    'Meriadoc': None,
    'Merry': None,
    'Peregrin': None,
    'Pippin': None,
    'Strider': [gs.Nation.THE_NORTH],
    'Boromir': [gs.Nation.GONDOR],
    'Legolas': [gs.Nation.ELVES],
    'Gimli': [gs.Nation.DWARVES],
}


def companion_activates(comp_name: str, nation) -> bool:
    """Does this companion activate *nation* by ending movement in its
    city/stronghold?  None entry = activates any FP nation."""
    nats = COMPANION_NATIONS.get(comp_name, [])
    if nats is None:
        return True
    return nation in nats


def _check_companion_activates_nation(state: gs.GameState, to_san: str, army: gs.Army):
    """Activate FP nation when companion ends movement in its city/stronghold (rulebook p.34).

    The 4th political activation trigger: a Companion ends movement in a
    City or Stronghold of their nation.
    """
    # Check if destination is a city or stronghold
    region_meta = board.regions.get(to_san, {})
    if not (region_meta.get("is_city") or region_meta.get("is_stronghold")):
        return

    # Determine which FP nation owns this region
    parent_id = region_meta.get("parent_id")
    if not parent_id:
        return
    faction_name = board.fid_to_name.get(parent_id, "").lower()
    nation_map = {
        'gondor': gs.Nation.GONDOR, 'rohan': gs.Nation.ROHAN,
        'elves': gs.Nation.ELVES, 'dwarves': gs.Nation.DWARVES,
        'the north': gs.Nation.THE_NORTH, 'north': gs.Nation.THE_NORTH,
    }
    region_nation = nation_map.get(faction_name)
    if not region_nation:
        return  # Not an FP region

    # Rulebook p.34: "...unless it is controlled by the enemy."  A
    # companion ending movement in a Shadow-held FP settlement does NOT
    # activate the nation (audit A9, 2026-07-26).
    if to_san in state.enemy_controlled.get(gs.Side.SHADOW, set()):
        return

    # Check if any companion in the army activates this nation
    for ld in army.leaders:
        if isinstance(ld, gs.Companion):
            if companion_activates(ld.name, region_nation):
                if not state.political.active.get(region_nation, False):
                    try:
                        activate_nation(state, region_nation,
                                        method="companion_move",
                                        region=to_san)
                    except RuntimeError:
                        pass
                return  # Only need to activate once


def _check_political_activation_for_entry(state: gs.GameState, to_san: str,
                                           moving_side: gs.Side):
    """Activate a nation when an enemy army enters one of its regions (rulebook p.34).

    This triggers only for the nation that owns the destination region,
    and only if the moving army is from the opposing side.
    """
    from board import regions as board_regions, fid_to_name
    reg_info = board_regions.get(to_san)
    if not reg_info:
        return
    parent_id = reg_info.get("parent_id")
    if not parent_id:
        return
    faction_name = fid_to_name.get(parent_id, "").lower()
    nation_map = {
        'gondor': gs.Nation.GONDOR, 'rohan': gs.Nation.ROHAN,
        'elves': gs.Nation.ELVES, 'dwarves': gs.Nation.DWARVES,
        'the north': gs.Nation.THE_NORTH, 'north': gs.Nation.THE_NORTH,
        'isengard': gs.Nation.ISENGARD, 'sauron': gs.Nation.SAURON,
        'southrons': gs.Nation.SOUTHRON_EASTERLINGS,
        'southron': gs.Nation.SOUTHRON_EASTERLINGS,
        'southrons and easterlings': gs.Nation.SOUTHRON_EASTERLINGS,
    }
    nation = nation_map.get(faction_name)
    if not nation:
        return
    # Check if the entering army is from the opposing side
    from combat import side_of_nation
    region_side = side_of_nation(nation)
    if region_side == moving_side:
        return  # Friendly entry — no activation
    # Activate the nation whose region was entered
    if not state.political.active.get(nation, False):
        try:
            activate_nation(state, nation, method="attack")
        except RuntimeError:
            pass  # Blocked by Wormtongue or similar


def army_stack_cap(state: gs.GameState, to_san: str,
                   moving_side) -> int:
    """THE destination stacking cap (QUEUE round 2026-08-17, census
    QUEUE_2: the siege-aware body lived only in move_army while the
    ai.py menu and event_effects._direct_army_move used the flat 10).
    Rulebook p.31: no more than FIVE army units inside a besieged
    stronghold - the 5-cap binds when the destination is besieged
    and the mover JOINS THE GARRISON side (a besieger reinforcing
    the siege lines outside stays at 10)."""
    besieged = getattr(state, 'besieged_strongholds', set())
    to_san = sanitize(str(to_san))
    if to_san in besieged and moving_side is not None:
        dst = None
        for k, a in state.armies.items():
            if sanitize(k) == to_san:
                dst = a
                break
        if dst is not None and dst.units:
            from combat import side_of_nation as _son3
            if not any(_son3(u.nation) != moving_side
                       for u in dst.units):
                return MAX_SIEGE_UNITS
    return MAX_ARMY_UNITS


def move_army(state: gs.GameState,
              fr: str,
              to: str,
              *,
              move_units: Optional[int] = None,
              move_elite: Optional[int] = None,
              move_leaders: Optional[int] = None,
              move_nazgul: Optional[int] = None,
              side: Optional[gs.Side] = None):
    """
    Move pieces from *fr* to *to*.

    • If no counts are supplied, the **entire** stack moves (old behaviour).
    • Otherwise supply how many of each category to move.
    • *side*: the ACTING side.  Required for correctness when the source
      is a shared besieged-region entry holding BOTH sides' units — the
      old units[0] heuristic misattributed the mover (and then the wrong
      side's units/dice were consumed).
    """
    b = state.board

    # Normalize keys - find actual army key and board key
    fr_san = sanitize(fr)
    to_san = sanitize(to)

    # Find the actual army key (original name with special chars)
    fr_army_key = None
    for key in state.armies:
        if sanitize(key) == fr_san:
            fr_army_key = key
            break

    if not fr_army_key:
        raise RuntimeError(f"No army in {fr}")

    # Check adjacency using sanitized keys
    if to_san not in b.adjacent(fr_san):
        raise RuntimeError(f"{to} not adjacent to {fr}")

    src = state.armies.get(fr_army_key)
    if not src:
        raise RuntimeError(f"No army in {fr}")

    # Get the destination region's original name for consistent army key
    to_region = b.region(to_san)
    to_army_key = to_region.orig_name if to_region else to
    # KEY-CASING DRIFT (duplicate-key factory): an army may already
    # exist under a differently-cased key ("Helm's Deep" parser naming
    # vs "Helm's deep" board orig_name) — reuse the existing key or the
    # setdefault below forges a DUPLICATE entry that splits the real
    # army (2024-07-27 Columbo T3: the Helm's Deep advance landed in a
    # phantom entry; the die-less pull-backs then found only the
    # garrison and Westemnet stayed empty for the Edoras attack).
    to_army_key = next((k for k in state.armies
                        if sanitize(k) == to_san), to_army_key)

    # Determine moving army's side for political activation check
    _moving_side = side
    if _moving_side is None and src.units:
        from combat import side_of_nation as _son
        _moving_side = _son(src.units[0].nation)

    # Determine effective stacking limit for destination.
    # Rulebook: the 10-unit stacking limit counts an army's OWN units.
    # A besieged region holds BOTH sides in one army entry — the garrison
    # inside the stronghold (max 5) must not count against the besieging
    # army outside, and vice versa. The 5-unit siege cap only applies to
    # units joining the garrison side inside a besieged stronghold.
    besieged = getattr(state, 'besieged_strongholds', set())
    _dst_probe = state.armies.get(to_army_key)
    if _dst_probe is None:
        for _k, _a in state.armies.items():
            if sanitize(_k) == to_san:
                _dst_probe = _a
                break

    def _own_units_at_dst():
        if _dst_probe is None:
            return 0
        if _moving_side is None:
            return len(_dst_probe.units)
        from combat import side_of_nation as _son2
        return sum(1 for u in _dst_probe.units
                   if _son2(u.nation) == _moving_side)

    # QUEUE_2 (2026-08-17): ONE destination-cap body - the ai.py
    # menu, this executor and the card-move path all read it
    _stack_cap = army_stack_cap(state, to_san, _moving_side)

    # A shared besieged-region entry holds BOTH sides: "whole stack" for
    # the acting side means ALL of ITS units — convert to a full split so
    # the enemy garrison/besiegers stay behind.
    if move_units is move_elite is move_leaders is None \
            and move_nazgul is None \
            and _moving_side is not None and src.units:
        from combat import side_of_nation as _son_ws
        if any(_son_ws(u.nation) != _moving_side for u in src.units):
            move_units = sum(1 for u in src.units
                             if u.kind is gs.UnitKind.REGULAR
                             and _son_ws(u.nation) == _moving_side)
            move_elite = sum(1 for u in src.units
                             if u.kind is gs.UnitKind.ELITE
                             and _son_ws(u.nation) == _moving_side)
            move_leaders = sum(1 for l in src.leaders
                               if not _is_saruman(l)
                               and _leader_side(l) == _moving_side)

    # ----- default: move whole stack (an explicit move_nazgul count is
    # a split, never a whole-stack move)
    if move_units is move_elite is move_leaders is None \
            and move_nazgul is None:
        # Check stacking limit before merge (own-side units only)
        dst_existing = state.armies.get(to_army_key)
        if dst_existing:
            combined_units = _own_units_at_dst() + len(src.units)
            if combined_units > _stack_cap:
                raise RuntimeError(f"Move would exceed stacking limit ({_stack_cap} units)")

        # Enemy-side characters sharing the region (e.g. a Companion in a
        # region a Shadow army occupies) must NOT be dragged along.
        # Saruman never moves (stays in Orthanc — official rule).
        _stay = [l for l in src.leaders if _is_saruman(l)]
        if _moving_side is not None:
            _stay += [l for l in src.leaders
                      if _leader_side(l) != _moving_side
                      and not _is_saruman(l)]
        if _stay:
            src.leaders = [l for l in src.leaders if l not in _stay]
        _fr_region_obj = src.region

        state.armies.pop(fr_army_key)
        dst = state.armies.setdefault(to_army_key, src)
        src.region = to_region
        if dst is not src:
            dst.units.extend(src.units); dst.leaders.extend(src.leaders)
        if _stay:
            _back = state.armies.setdefault(
                fr_army_key, gs.Army(region=_fr_region_obj))
            if _back is src:  # src became dst object; make a fresh army
                _back = gs.Army(region=_fr_region_obj)
                state.armies[fr_army_key] = _back
            _back.leaders.extend(_stay)

        # Political activation: enemy army entering a region (rulebook p.34)
        if _moving_side:
            _check_political_activation_for_entry(state, to_san, _moving_side)
        # Companion in city/stronghold activates nation (rulebook p.34)
        _check_companion_activates_nation(state, to_san, dst)
        # Remove FP Leaders left alone without combat units (rulebook p.8)
        remove_orphaned_leaders(state)
        return

    # ----- split logic ----------------------------------------------------
    def take(lst: list, qty: int | None, predicate=lambda x: True):
        moved, kept = [], []
        for obj in lst:
            (moved if qty and predicate(obj) and len(moved) < qty else kept).append(obj)
        return moved, kept

    # Only the ACTING side's units are movable (shared besieged entries
    # hold both sides — the enemy's units always stay behind)
    from combat import side_of_nation as _son_sp
    _own_units, _enemy_units = [], []
    for u in src.units:
        if _moving_side is None or _son_sp(u.nation) == _moving_side:
            _own_units.append(u)
        else:
            _enemy_units.append(u)

    m_elite , keep_elite  = take([u for u in _own_units if u.kind is gs.UnitKind.ELITE],
                                 move_elite)
    m_regular, keep_reg   = take([u for u in _own_units if u.kind is gs.UnitKind.REGULAR],
                                 move_units)
    # Only the moving side's own leaders may travel with the army;
    # Saruman never moves (stays in Orthanc — official rule)
    def _movable_leader(l):
        return (not _is_saruman(l)
                and (_moving_side is None
                     or _leader_side(l) == _moving_side))
    if move_leaders:
        # Callers commonly ask for len(src.leaders); immobile (Saruman)
        # and enemy-side figures are not movable — clamp instead of
        # failing the whole move.  With an explicit move_nazgul count,
        # move_leaders governs NON-Nazgul figures only, so the clamp
        # must not count Nazgul.
        if move_nazgul is None:
            move_leaders = min(
                move_leaders,
                sum(1 for l in src.leaders if _movable_leader(l)))
        else:
            move_leaders = min(
                move_leaders,
                sum(1 for l in src.leaders
                    if _movable_leader(l)
                    and not isinstance(l, gs.Nazgul)))
    if move_nazgul is None:
        m_lead, keep_lead = take(
            src.leaders, move_leaders, predicate=_movable_leader)
    else:
        # USER 2026-07-11: nazgul ride separately from plain leaders -
        # move_nazgul picks how many Nazgul travel (fungible), while
        # move_leaders governs the non-Nazgul figures as before.
        # Nazgul left behind may stand alone (no unit guard needed).
        _naz_pool9 = [l for l in src.leaders
                      if _movable_leader(l) and isinstance(l, gs.Nazgul)]
        _mv_naz9 = _naz_pool9[:max(0, min(move_nazgul, len(_naz_pool9)))]
        _mv_naz_ids9 = set(map(id, _mv_naz9))
        _rest9 = [l for l in src.leaders if id(l) not in _mv_naz_ids9]
        # non-moving nazgul stay with the keep pile (take() keeps
        # predicate-failing objects)
        m_oth9, keep_lead = take(
            _rest9, move_leaders,
            predicate=lambda l: (_movable_leader(l)
                                 and not isinstance(l, gs.Nazgul)))
        m_lead = _mv_naz9 + m_oth9

    if move_elite and len(m_elite) < move_elite:
        raise RuntimeError("Not enough elites in stack")
    if move_units and len(m_regular) < move_units:
        raise RuntimeError("Not enough regulars in stack")
    if move_leaders:
        # With an explicit move_nazgul, the shortfall check covers the
        # NON-Nazgul figures the count governs (m_lead also holds the
        # moved Nazgul).
        _got_lead = len(m_lead) if move_nazgul is None else len(m_oth9)
        if _got_lead < move_leaders:
            raise RuntimeError("Not enough leaders in stack")

    # Check stacking limit before merge (own-side units only; siege cap
    # only when reinforcing the garrison side inside a besieged stronghold)
    moving_units = len(m_elite) + len(m_regular)
    dst_existing = state.armies.get(to_army_key)
    if dst_existing and _own_units_at_dst() + moving_units > _stack_cap:
        raise RuntimeError(f"Move would exceed stacking limit ({_stack_cap} units)")

    # Moved figures LEAVE the source (previously they were duplicated:
    # written back to src AND extended into dst); the enemy side's units
    # in a shared entry always stay
    src.units   = keep_elite + keep_reg + _enemy_units
    src.leaders = keep_lead
    if not src.units and not src.leaders:
        state.armies.pop(fr_army_key, None)

    # create / merge destination
    dst = state.armies.setdefault(
        to_army_key, gs.Army(region=to_region))
    dst.units.extend(m_elite + m_regular)
    dst.leaders.extend(m_lead)

    # Political activation: enemy army entering a region (rulebook p.34)
    if _moving_side:
        _check_political_activation_for_entry(state, to_san, _moving_side)
    # Companion in city/stronghold activates nation (rulebook p.34)
    _check_companion_activates_nation(state, to_san, dst)
    # Remove FP Leaders left alone without combat units (rulebook p.8)
    remove_orphaned_leaders(state)

    # USER RULE (2026-07-08): entering an undefended enemy settlement
    # captures it (towns included) — "if settlements are empty and I
    # walk through them, they are now captured"
    if _moving_side and dst.units:
        from combat import side_of_nation as _son2, \
            _check_settlement_capture as _csc2
        if not any(_son2(u.nation) != _moving_side for u in dst.units):
            _csc2(state, to_san, _moving_side)



#  Helper: find board-key + meta for a sanitised region key
def _find_region_key_and_meta(state: gs.GameState,
                              sanitised_key: str) -> Tuple[str, Optional[dict]]:
    """
    Robust lookup using *state.board*.

    Returns (original_key, meta_dict or None).
    """

    # 1) exhaustive scan first (cheap) ----------------------------------
    for k, v in board.name_to_meta.items():
        if sanitize(k) == sanitised_key:
            return k, v

    # 2) ask the live Board instance ------------------------------------
    try:
        reg_obj = state.board.region(sanitised_key)
    except Exception as exc:
        raise RuntimeError("Region exists for movement but not for metadata")

    orig_key = getattr(reg_obj, "key", None) or getattr(reg_obj, "name", None)
    if orig_key is None:
        orig_key = str(reg_obj).split()[0]

    meta = board.name_to_meta.get(orig_key)

    return orig_key, meta

#  DECLARE Fellowship phase flag (K2): the turn-start declare window is
#  an FP-owned learned decision served by ai.get_legal_actions under
#  state._declare_phase (declares + the 'pass' decline).  Drivers arm
#  and settle it through these two calls only; any executed action
#  (declare or pass) closes it too.
def open_declare_phase(state: gs.GameState) -> None:
    state._declare_phase = True


def close_declare_phase(state: gs.GameState) -> None:
    state._declare_phase = False


#  DECLARE Fellowship: reachability validation + pawn placement
def declare_fellowship(state: gs.GameState, *, region: str):

    region_key = sanitize(region)

    cur_key = fellowship_region(state)

    steps = state.fellowship.progress

    if steps == 0 and region_key != cur_key:
        raise RuntimeError("Fellowship did not move this turn")

    # ---- reachability search ------------------------------------------
    # Official rule: the pawn may be placed in any region at distance UP TO
    # the progress counter (including staying in place).
    reachable = {cur_key}
    frontier = {cur_key}
    for hop in range(steps):
        frontier = {
            sanitize(adj)
            for r in frontier
            for adj in state.board.adjacent(r)
        }
        reachable |= frontier

    if region_key not in reachable:
        raise RuntimeError("Region not reachable with current progress")

    # USER RULING 2026-07-20b: never declare INTO Mordor's interior —
    # the pawn declares AT a gateway. Self-play executor only: the
    # teacher's logged declares are ground truth (humans never declare
    # into gorgoroth).
    #
    # ENTRY TIMING, settled 2026-08-16 (USER, final): "you can only
    # enter at the start of the turn, either by declaration or by being
    # revealed on morannon or minas morgul."  So entry is TURN-START
    # ONLY and POSITIONAL - the qualifying test is where the fellowship
    # stands, reached by declaration OR by reveal (hunt-forced
    # included).  There is no re-hide requirement and no entry inside
    # the declare action; declaring here only PLACES the pawn.
    #
    # An intermediate reading had entry happen in this action.  It was
    # implemented and REVERTED the same hour: the corpus says otherwise
    # and so does the USER.  Recorded because the wrong version is the
    # kind that looks tidier than the rule.
    #
    # CORPUS CONFIRMATION (17,692 human logs, 8,939 gate-placement ->
    # entry sequences), measured because the engine and its comparator
    # share one model and agreement-gates cannot see a shared wrong
    # rule:
    #   * entries in the SAME turn as the placement: the placement sits
    #     a MEDIAN OF 8 LINES after the turn marker (77% within 40) -
    #     i.e. both events are in the turn-START window, not mid-turn.
    #   * entries in a LATER turn: 5,662 of 5,672 (100%) land within 40
    #     lines of the NEXT turn marker - next-turn-start, exactly.
    #   * archived engine games: all 46 declare->entry sequences cross
    #     exactly one turn boundary; none entered mid-turn.
    # Turn-start-only is what the corpus shows and what the four
    # turn-start call sites (bridge _eng_declare -> _eng_mordor,
    # ai._run_single_episode, ai._advance_turn_for_search,
    # engine_play EngineSession) implement - all positionally, with
    # the DECLARATION OPPORTUNITY FIRST and the entry check AFTER it
    # in the same turn start (ordering fix 2026-08-16).
    import combat as _cmb_mord
    if (_cmb_mord.FINAL_HITS_HOOK is None
            and region_key in _MORDOR_INTERIOR):
        raise RuntimeError(
            "Fellowship cannot declare into Mordor's interior (1.155)")

    # ---- locate original map key --------------------------------------
    try:
        orig_key, meta = _find_region_key_and_meta(state, region_key)

    except RuntimeError as exc:
        raise

    # ---- place pawn & reset -------------------------------------------
    state.fellowship.region = state.board.region(orig_key)
    state.fellowship.progress = 0
    state.fellowship.hidden = True

    # Declaring in an FP City/Stronghold discards "Worn with Sorrow and
    # Toil" / "Flocks of Crebain" (persistent card auto-discard triggers)
    from persistent_effects import (check_fellowship_declared_discards,
                                     discard_table_card)
    for _pc in check_fellowship_declared_discards(state, region_key):
        _rm = state.persistent_cards.remove_card(_pc.card_id)
        discard_table_card(state, _rm or _pc)

    # 1.211i: The Last Battle discards on DECLARATION as well as reveal
    # (Mordor entry is a declaration, 1.155).  _recheck_last_battle only
    # catches the REVEAL (hidden False); a declaration keeps hidden=True.
    # Un-gated is safe: remove_card is idempotent, the rule is mandatory,
    # and it sits beside the un-gated sibling discards above.
    _tlb = state.persistent_cards.get_by_type(
        gs.PersistentCardType.LAST_BATTLE)
    if _tlb is not None:
        _rm = state.persistent_cards.remove_card(_tlb.card_id)
        discard_table_card(state, _rm or _tlb)

    # ---- heal check ---------------------------------------------------
    # Use region metadata from board.regions, not faction metadata
    # Healing requires: FP settlement (city/stronghold) NOT captured by Shadow
    region_meta = board.regions.get(region_key, {})
    is_settlement = region_meta.get("is_city") or region_meta.get("is_stronghold")
    shadow_captured = state.enemy_controlled.get(gs.Side.SHADOW, set())
    if is_settlement and is_fp_region(region_key):
        if region_key not in shadow_captured:
            before = state.fellowship.corruption
            state.fellowship.corruption = max(0, before - 1)
        # Political activation: fellowship declared in an FP city/
        # stronghold (rulebook p.34).  The activation trigger has NO
        # enemy-control exception (audit A10, 2026-07-26) - only the
        # HEALING is lost when Shadow holds the settlement.
        _activate_nation_for_fellowship_declaration(state, region_key)

    # USER RULING 2026-07-27 (supersedes fix 101): a DECLARATION draws
    # NO Shadow-stronghold hunt tile — the printed rule (marker p.37:
    # "only if the Fellowship is REVEALED by the Shadow player") is
    # reveal-only, and the fix-101 extension made declaring through
    # Moria-class regions wrongly punishing.  TEACHER-ONLY EXCEPTION:
    # the closed loop's wrapped _draw_shadow_stronghold_tile applies
    # log-evidence gates, so historical games whose humans DID resolve
    # a tile on a declare (the original game-11 evidence) still replay
    # exactly — the draw fires only when a teacher hook is installed
    # AND the log shows it.
    import combat as _cmb101
    if _cmb101.FINAL_HITS_HOOK is not None:
        def _is_sh_sh101(r):
            _m101 = board.regions.get(sanitize(r), {})
            return (_m101.get('is_stronghold')
                    and is_shadow_region(sanitize(r))
                    and sanitize(r) not in state.enemy_controlled[
                        gs.Side.FREE_PEOPLES])
        try:
            state._sh_tile_drawn_regions = set()
            if sanitize(cur_key) != region_key and _is_sh_sh101(cur_key):
                _draw_shadow_stronghold_tile(state, cur_key, "leaving")
            if sanitize(cur_key) != region_key and _is_sh_sh101(region_key):
                _draw_shadow_stronghold_tile(state, region_key, "entering")
        except Exception:
            pass


def hide_fellowship(state: gs.GameState) -> dict:
    """
    Hide the fellowship (uses Character die action).
    Returns result dict with success status.
    """
    if state.fellowship.hidden:
        return {"success": False, "message": "Fellowship already hidden"}

    state.fellowship.hidden = True
    # Track Fellowship activity (avoids end-of-turn Mordor corruption
    # penalty).  Unconditional: acting BEFORE entering Mordor the same
    # turn still counts (flag only read at turn end when in Mordor).
    state.fellowship_acted_in_mordor = True
    return {"success": True, "message": "Fellowship hidden"}


def has_action_token(state: gs.GameState, side: gs.Side,
                     token_type: str) -> bool:
    """Is a token of *token_type* ('muster'/'draw') available to *side*?

    FP tokens are the two named bid-compensation pieces (Hunt for the
    Ring Political / Card Draw); the Shadow side uses the generic count.
    """
    if side is gs.Side.FREE_PEOPLES:
        if token_type == "muster":
            return bool(state.fp_political_token)
        return bool(state.fp_card_token)
    return state.sh_action_tokens > 0


def use_action_token(state: gs.GameState,
                     side: gs.Side,
                     *,
                     token_type: str,
                     nation: Optional[gs.Nation] = None,
                     deck: str = "strategy"):
    """
    Bid-compensation action token (user ruling 2026-07-03, see
    data/rules_amendments.md).

    `token_type` : "muster" or "draw"

    • muster – one political advance, equivalent to a muster die
               (requires nation param).
    • draw   – draw from EITHER deck (`deck` = 'character'/'strategy';
               the deck choice is a player decision).

    Tokens are SINGLE-USE, at most ONE token use per side per turn, and
    exist OUTSIDE the action dice: using one counts as an action in the
    turn-order alternation WITHOUT consuming a die.
    """
    used_attr = ("fp_token_used_this_turn" if side is gs.Side.FREE_PEOPLES
                 else "sh_token_used_this_turn")
    if getattr(state, used_attr, False):
        raise RuntimeError("Already used an action token this turn")
    if not has_action_token(state, side, token_type):
        raise RuntimeError(f"No {token_type} action token left")
    if token_type == "muster":
        if nation is None:
            raise RuntimeError("Must supply nation")
        advance_political(state, nation, method="muster_die")
    elif token_type == "draw":
        if deck not in ("character", "strategy"):
            raise RuntimeError("deck must be character/strategy")
        # engine-granted draw (A13 source 6): routes through the debt
        # ledger so a deferred client-side draw is recognized as owed
        grant_draw(state, side, deck)
    else:
        raise RuntimeError("token_type must be muster/draw")
    # consume the single-use token
    if side is gs.Side.FREE_PEOPLES:
        if token_type == "muster":
            state.fp_political_token = False
        else:
            state.fp_card_token = False
    else:
        state.sh_action_tokens -= 1
    setattr(state, used_attr, True)


# Dwarven-ring face classes: the reroll result must not share a face
# class with the original face (user ruling 2026-07-03: 'reroll one die
# until a DIFFERENT face-class than the original; a Muster original may
# not end on Muster or Muster/Army').  The hybrid Muster/Army face
# belongs to both the muster and army classes (symmetric interpretation,
# documented in data/rules_amendments.md).
_DWARVEN_FACE_CLASSES = {
    gs.ActionDieFace.CHARACTER: frozenset({'character'}),
    gs.ActionDieFace.ARMY: frozenset({'army'}),
    gs.ActionDieFace.MUSTER: frozenset({'muster'}),
    gs.ActionDieFace.MUSTER_ARMY: frozenset({'muster', 'army'}),
    gs.ActionDieFace.EVENT: frozenset({'event'}),
    gs.ActionDieFace.EYE: frozenset({'eye'}),
    gs.ActionDieFace.WILL_OF_WEST: frozenset({'will'}),
}


def can_use_dwarven_ring(state: gs.GameState, side: gs.Side) -> bool:
    """Dwarven rings are used at the START of a turn or not at all (user
    ruling 2026-07-03): only before any action-die/token use this turn.
    Eye dice pre-allocated to the Hunt Box are marked used at roll time
    and do not count as 'the turn has started'."""
    attr = ("fp_dwarven_rings" if side is gs.Side.FREE_PEOPLES
            else "sh_dwarven_rings")
    if getattr(state, attr, 0) <= 0:
        return False
    for pool_side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        for d in state.dice.get(pool_side, []):
            if d.used and d.current != gs.ActionDieFace.EYE:
                return False
    if getattr(state, 'fp_token_used_this_turn', False) or \
            getattr(state, 'sh_token_used_this_turn', False):
        return False
    return True


def use_dwarven_ring(state: gs.GameState,
                     side: gs.Side,
                     *,
                     die_idx: int,
                     forced_face: Optional[gs.ActionDieFace] = None):
    """
    Reroll one unused die (user ruling 2026-07-03, see
    data/rules_amendments.md): usable at the START of a turn or not at
    all; the reroll repeats until the result is of a DIFFERENT face
    class than the original (Muster original -> result may not be
    Muster or Muster/Army; hybrid Muster/Army blocks both classes).

    `forced_face`: closed-loop forcing (zero unforced RNG) — the logged
    result face; must satisfy the class constraint.
    """
    attr = "fp_dwarven_rings" if side is gs.Side.FREE_PEOPLES else "sh_dwarven_rings"
    if getattr(state, attr) == 0:
        raise RuntimeError("No Dwarven Rings left")
    if not can_use_dwarven_ring(state, side):
        raise RuntimeError("Dwarven Ring only usable at the start of a turn")

    die = state.dice[side][die_idx]
    if die.used:
        raise RuntimeError("Die already used")
    if die.current is None:
        raise RuntimeError("Cannot reroll an unrolled die")

    orig_classes = _DWARVEN_FACE_CLASSES.get(die.current, frozenset())

    if forced_face is not None:
        if _DWARVEN_FACE_CLASSES.get(forced_face, frozenset()) & orig_classes:
            raise RuntimeError(
                f"Forced face {forced_face.value} shares a class with "
                f"the original {die.current.value}")
        die.current = forced_face
    else:
        valid = [f for f in die.faces
                 if not (_DWARVEN_FACE_CLASSES.get(f, frozenset())
                         & orig_classes)]
        if not valid:
            raise RuntimeError("No valid reroll result exists for this die")
        die.current = random.choice(valid)
    # "All dice showing the Eye must be placed in the Hunt Box" (marker
    # p.20; audit A3, 2026-07-26) - mirror roll_action_dice/use_elven_ring.
    # Dormant in the base game (no Shadow dwarven ring) but correct.
    if side is gs.Side.SHADOW and die.current is gs.ActionDieFace.EYE:
        die.used = True
        state.hunt_box.append(die)
    setattr(state, attr, getattr(state, attr) - 1)


def _leader_side(ld) -> gs.Side:
    """Side a leader figure belongs to (Companions FP; Nazgul/Minions SH;
    plain Leaders by nation)."""
    if isinstance(ld, gs.Companion):
        return gs.Side.FREE_PEOPLES
    if isinstance(ld, gs.Nazgul) or type(ld).__name__ == 'Minion':
        return gs.Side.SHADOW
    nat = getattr(ld, 'nation', None)
    if nat is None:
        return gs.Side.FREE_PEOPLES
    from combat import side_of_nation
    return side_of_nation(nat)


def get_or_create_army(state: gs.GameState, region_key: str) -> gs.Army:
    """Return the army at region_key (matching by SANITIZED name),
    creating an empty one under the sanitized key if absent.

    Army-creation sites historically keyed state.armies by either the
    original name ('Orthanc') or the sanitized one ('orthanc'); exact-key
    existence checks then created duplicates, making sanitized-match
    lookups order-dependent (phantom 'no Nazgul at X' failures). Always
    create through this helper.
    """
    san = sanitize(region_key)
    for k, a in state.armies.items():
        if sanitize(k) == san:
            return a
    region = state.board.region(san)
    # canonicalize the dict key against the board ('the grey havens'
    # from a VASSAL location name vs board key 'grey havens' — a
    # non-canonical key would be a fresh duplicate-army-key bug)
    key = getattr(region, 'key', san) or san
    for k, a in state.armies.items():
        if sanitize(k) == key:
            return a
    army = gs.Army(region=region)
    state.armies[key] = army
    return army


def advance_defenders_political(state: gs.GameState, dfn_r: str,
                                attacker_side=None) -> None:
    """Rulebook p.35: a nation is activated AND advanced one step when an
    army containing its units is attacked. Fires once, at attack
    declaration (including sieges that never see an assault round).
    attacker_side filters the attacker's own nations out of mixed
    co-located siege entries."""
    dfn_san = sanitize(dfn_r)
    def_army = state.armies.get(dfn_r) or state.armies.get(dfn_san)
    if def_army is None:
        for k, a in state.armies.items():
            if sanitize(k) == dfn_san:
                def_army = a
                break
    if not def_army:
        return
    nations = set(u.nation for u in def_army.units)
    if attacker_side is not None:
        nations = {n for n in nations if _side_of(n) is not attacker_side}
    for nation in nations:
        try:
            if not state.political.active.get(nation, False):
                # W3 BUG-1 fix (2026-07-20): route through
                # activate_nation so persistent-card blockers
                # (Wormtongue: Rohan) apply on the ATTACK path too —
                # the raw flip here bypassed can_activate_nation.
                try:
                    activate_nation(state, nation, method="attack",
                                    region=dfn_san)
                except RuntimeError:
                    pass   # blocked (e.g. Wormtongue) — no flip
            if state.political.boxes[nation] < 3:
                advance_political(state, nation, method="attack")
        except Exception:
            pass  # Already at war or blocked by persistent card


def battle(state: gs.GameState, atk_r: str, dfn_r: str,
           attacker_card_id: Optional[str] = None,
           defender_card_id: Optional[str] = None,
           is_siege: bool = False,
           max_rounds: int = 1,
           auto_press: bool = False,
           choices: Optional[Dict] = None) -> "gs.CombatResult":
    """
    Resolve combat between two regions.

    Args:
        state: Game state
        atk_r: Attacking region
        dfn_r: Defending region
        attacker_card_id: Combat card ID for attacker
        defender_card_id: Combat card ID for defender
        is_siege: Whether this is a siege assault
        max_rounds: Number of combat rounds (1 normally, 3 for Grond/Fighting Uruk-hai)
        auto_press: If True and is_siege, auto-press by downgrading elites
        choices: Player choices for interactive card effects

    Returns:
        CombatResult with battle outcome
    """
    # NOTE: no per-turn attack restriction — the official rules allow
    # attacking the same region repeatedly with separate Action dice
    # (verified vs rulebook + almanac; the old limit was a house rule).
    dfn_san = sanitize(dfn_r)

    # Per rulebook p.35 the defending nation activates/advances when the
    # attack is DECLARED (see advance_defenders_political, called from the
    # attack executor) — advancing here fired once per ROUND, double-
    # advancing multi-round battles and never firing for sieges that
    # began without an assault.

    # OVERSIGHT #2 / W1-audit fix (2026-07-20): publish the live state
    # so combat's _downgrade_allowed supply checks work. (The original
    # 1.182 wiring landed in dead _apply_hits by a patch-anchor bug —
    # W1 caught the no-op.)
    from combat import resolve_combat, set_active_state
    _prev_as = set_active_state(state)   # save/restore (nesting-safe)
    try:
        return resolve_combat(
            state, atk_r, dfn_r,
            attacker_card_id=attacker_card_id,
            defender_card_id=defender_card_id,
            is_siege=is_siege,
            max_rounds=max_rounds,
            auto_press=auto_press,
            choices=choices or {}
        )
    finally:
        # OVERSIGHT (d) + nesting fix: RESTORE the prior state (not
        # None). A card handler that published its own state before
        # triggering this battle must keep it for its later casualty
        # applications.
        set_active_state(_prev_as)


# Fellowship / region helpers (unchanged)
def fellowship_region(state: gs.GameState) -> str:
    r = state.fellowship.region
    if isinstance(r, str):
        return sanitize(r)
    for attr in ("key", "name", "orig_name"):
        if hasattr(r, attr):
            return sanitize(getattr(r, attr))
    txt = str(r).lower()
    if " key " in txt:
        return sanitize(txt.split(" key ")[1].split()[0])
    return sanitize(txt)

def _army_at_fellowship(s: gs.GameState):
    """Army at the fellowship's region, tolerant of original-case army keys
    ('Hollin' vs 'hollin') — exact-key lookups silently missed re-roll
    sources for armies stored under unsanitized keys."""
    want = fellowship_region(s)
    a = s.armies.get(want)
    if a is not None:
        return a
    for k, army in s.armies.items():
        if sanitize(k) == want:
            return army
    return None

def _nazgul_present(s: gs.GameState) -> bool:
    a = _army_at_fellowship(s);  return bool(a and any(isinstance(l, gs.Nazgul) for l in a.leaders))
def _shadow_army_present(s: gs.GameState) -> bool:
    a = _army_at_fellowship(s);  return bool(a and any(_side_of(u.nation) is gs.Side.SHADOW for u in a.units))

# OPT (stateless memo): is_shadow_region/is_fp_region are pure functions
# of an immutable input (region name) and immutable board geography
# (board.regions/fid_to_name/name_to_meta are built once at import and
# never mutated). They were called per-settlement on every victory-point
# snapshot (28k+ calls/300-sim search), each doing a sanitize + 3 dict
# lookups. Memoizing on the raw region_key is byte-identical and bounds
# the key space to the ~200 distinct region/location names seen.
_IS_SHADOW_REGION_CACHE: dict = {}
_IS_FP_REGION_CACHE: dict = {}


def is_shadow_region(region_key: str) -> bool:
    """Check if a region belongs to Shadow based on its parent faction."""
    cached = _IS_SHADOW_REGION_CACHE.get(region_key)
    if cached is not None:
        return cached
    meta = board.regions.get(sanitize(region_key), {})
    result = False
    if meta:
        parent_id = meta.get("parent_id")
        if parent_id:
            # Look up faction name and check isShadows flag
            faction_name = board.fid_to_name.get(parent_id)
            if faction_name:
                faction_meta = board.name_to_meta.get(faction_name, {})
                result = faction_meta.get("isShadows", False)
    _IS_SHADOW_REGION_CACHE[region_key] = result
    return result


def is_fp_region(region_key: str) -> bool:
    """Check if a region belongs to Free Peoples based on its parent faction."""
    cached = _IS_FP_REGION_CACHE.get(region_key)
    if cached is not None:
        return cached
    meta = board.regions.get(sanitize(region_key), {})
    result = False
    if meta:
        parent_id = meta.get("parent_id")
        if parent_id:
            faction_name = board.fid_to_name.get(parent_id)
            if faction_name:
                faction_meta = board.name_to_meta.get(faction_name, {})
                # FP regions are those where isShadows is False
                result = not faction_meta.get("isShadows", False)
    _IS_FP_REGION_CACHE[region_key] = result
    return result

def is_fp_nation_region(region_key: str) -> bool:
    """Strictly inside a Free Peoples NATION's borders (Gondor, Rohan,
    Elves, Dwarves, North) — neutral 'free' wilderness does NOT count.
    is_fp_region() treats neutral as FP, which is right for friendliness
    checks but wrong for card requirements like 'the defending Army is
    inside the borders of a Free Peoples Nation'."""
    meta = board.regions.get(sanitize(region_key), {})
    parent_id = meta.get("parent_id")
    if not parent_id:
        return False
    faction_name = board.fid_to_name.get(parent_id)
    if not faction_name or faction_name == 'free':
        return False
    faction_meta = board.name_to_meta.get(faction_name, {})
    return not faction_meta.get("isShadows", False)


def _in_shadow_stronghold(s: gs.GameState) -> bool:
    """Check if fellowship is in a stronghold controlled by Shadow.

    Re-roll source requires:
    - The region is a stronghold (not just a city), AND
    - Shadow controls it: either
      (a) Shadow-native stronghold NOT captured by FP, or
      (b) FP-native stronghold captured by Shadow.
    """
    region_key = fellowship_region(s)
    meta = board.regions.get(region_key, {})
    if not meta.get("is_stronghold"):
        return False
    fp_captured = s.enemy_controlled.get(gs.Side.FREE_PEOPLES, set())
    sh_captured = s.enemy_controlled.get(gs.Side.SHADOW, set())
    if is_shadow_region(region_key):
        # Shadow stronghold: gives re-roll unless FP captured it
        return region_key not in fp_captured
    elif is_fp_region(region_key):
        # FP stronghold: gives re-roll only if Shadow captured it
        return region_key in sh_captured
    return False


#  Legacy hunt damage resolution (used by tests and simple cases)
def apply_hunt_damage_simple(state: gs.GameState,
                              dmg: int,
                              *,
                              kill_random: bool = False,
                              kill_guide: bool = False,
                              kill_name: Optional[str] = None):
    """
    Apply *dmg* corruption **after** FP has decided how to mitigate.

    This is a simpler version for tests. Full damage resolution runs
    through _settle_hunt_damage / the pending_hunt decision (AI) or
    gui._resolve_hunt_damage (GUI).

    • kill_random  – remove a random companion (not Gollum)
    • kill_guide   – remove current guide (ignored if guide is Gollum)
    • kill_name    – remove the named companion

    Only **one** of the three flags may be set.
    """
    flags = sum(bool(x) for x in (kill_random, kill_guide, kill_name))
    if flags > 1:
        raise RuntimeError("Specify only one mitigation option")

    if kill_random:
        candidates = [c for c in state.fellowship.companions
                      if c.name != "Gollum"]
        if candidates:
            victim = pick_random_companion(candidates,
                                           'hunt casualty draw')
            if hobbit_hunt_placement_applies(state, victim.name):
                # Randomly drawn Hobbit: placed, not eliminated
                remove_hobbit_for_placement(state, victim, dmg)
                place_separated_companion(state, victim,
                                          fellowship_region(state))
                return
            kill_name = victim.name    # fall through to common path

    _apply_fellowship_damage(
        state, dmg,
        sacrifice_guide=kill_guide,
        kill_name=kill_name)


# Guide management (unchanged)
def change_guide(state: gs.GameState, *, chosen: Optional[str] = None,
                 allow_pending: bool = False):
    comps = state.fellowship.companions
    if not comps:
        # Gollum takes over as guide WITH his ability (reveal icons on
        # ordinary damage tiles do not reveal the Fellowship) — the ad-hoc
        # object previously created here lacked guide_ability entirely
        state.fellowship.guide = gs.Companion(
            "Gollum", 0, 0, guide_ability="no_reveal_on_damage")
        return
    best_lvl = max(c.level for c in comps)
    best = [c for c in comps if c.level == best_lvl]
    if len(best) == 1:
        state.fellowship.guide = best[0]; return
    if chosen is not None:
        for c in best:
            if c.name == chosen:
                state.fellowship.guide = c; return
        raise RuntimeError(f"{chosen} not among tied companions")
    # Tie with no chosen: set pending state for AI decision
    if allow_pending:
        state.pending_guide_change = {
            'candidates': [c.name for c in best]
        }
        # Temporarily keep current guide (or pick first) until AI decides
        if state.fellowship.guide not in best:
            state.fellowship.guide = best[0]
        return
    raise RuntimeError("Guide tie - supply chosen")

# Companion separation  (explicit die)
#  NEW: move several companions with one Character / Will die
def separate_companions(state: gs.GameState,
                         names: List[str],
                         target: str,
                         *,
                         die_idx: int):
    """
    Spend one FP Character (or Will) die to move *names* to *target*.

    • Removes each named companion from the Fellowship.
    • Guide is re-picked automatically if he/she leaves.
    """
    if not names:
        raise RuntimeError("No companion names supplied")

    # Rulebook p.43: companions can never be separated on Mordor track
    if state.fellowship.mordor_entered:
        raise RuntimeError("Cannot separate companions while on the Mordor Track")

    # Validate EVERY name before mutating anything: the die burn and
    # the per-companion moves below must be all-or-nothing (a mid-loop
    # raise would leave the state half-mutated with the die spent).
    _fs_names = {c.name for c in state.fellowship.companions}
    for name in names:
        if name not in _fs_names:
            raise RuntimeError(f"{name} not in Fellowship")

    _pick_die(state, gs.Side.FREE_PEOPLES,
              (gs.ActionDieFace.CHARACTER,), die_idx=die_idx)

    for name in names:
        for i, comp in enumerate(state.fellowship.companions):
            if comp.name == name:
                state.fellowship.companions.pop(i)
                # 1.148: get_or_create_army, NEVER setdefault - a raw
                # setdefault('moria') beside an existing 'Moria' entry
                # created a shadowed duplicate; the separated companion
                # was invisible to sanitized-index lookups (group-move
                # menus, crown checks) and vanished on forced walks.
                army = get_or_create_army(state, target)
                army.leaders.append(comp)
                # 1.149b: separation IS movement - ending it in the
                # companion's own city/stronghold activates the nation
                # (rulebook p.34, 4th trigger)
                _check_companion_activates_nation(
                    state, sanitize(str(target)), army)
                break
        else:
            raise RuntimeError(f"{name} not in Fellowship")

    if state.fellowship.guide not in state.fellowship.companions:
        change_guide(state, allow_pending=True)
    check_companion_persistent_discards(state)

# Army / Character / Political / Card actions
def can_advance_political(state: gs.GameState, nation: gs.Nation,
                          method: str = "muster_die"):
    """THE political-advance rule body (QUEUE round 2026-08-17,
    census QUEUE_3: three copies lived in the two ai.py menus and
    the executor).  Menus propose, the executor enforces - both
    through here.  Returns (ok, reason)."""
    from persistent_effects import can_advance_passive_nation
    box = state.political.boxes.get(nation, 0)
    if box >= 3:
        return False, f"{nation.value} is already At War"
    if box == 2 and not state.political.is_active(nation):
        return False, (f"{nation.value} is not active "
                       f"(cannot advance to At War)")
    ok, blocker = can_advance_passive_nation(state, nation, method)
    if not ok:
        return False, f"blocked by {blocker}"
    return True, None


def advance_political(state: gs.GameState, nation: gs.Nation,
                      method: str = "muster_die") -> int:
    """
    Advance a nation one step on the political track.
    Returns the new position.

    Args:
        method: "muster_die", "attack", "companion_ability", "event_card"
    """
    from persistent_effects import check_nation_advanced_discards

    box = state.political.boxes[nation]
    # QUEUE_3 (2026-08-17): ONE rule body - can_advance_political.
    # Executor tolerance preserved exactly: an already-At-War advance
    # stays the historical silent no-op (teacher logs carry
    # duplicate advances); every other illegality raises loudly.
    _ok9, _why9 = can_advance_political(state, nation, method)
    if not _ok9 and box < 3:
        raise RuntimeError(f"Cannot advance {nation.value}: {_why9}")

    if box < 3:
        state.political.boxes[nation] = box + 1
        # SESSION-21 USER RULE, RESTORED (session 34, w1_ep00007
        # Gondor-active-T1 regression): ADVANCING THE POLITICAL TRACK
        # NEVER ACTIVATES THE NATION.  Activation happens only through
        # its own triggers — attacked/entered with army units
        # (activate_nation_for_attack / the attack path's explicit
        # flip), fellowship declared or revealed in the nation's
        # city/stronghold, a companion ending movement there, or an
        # event card that says so (activate_nation).  A card-QA wave
        # (2026-07-11 CASE11) had reinstated advance-activates behind a
        # Wormtongue gate — inverting the rule; with it, one muster die
        # took a passive nation to Active and a second to At War.

        # Check for nation advancement discards (Threats and Promises)
        from persistent_effects import discard_table_card
        for card in check_nation_advanced_discards(state, nation, method):
            _rm = state.persistent_cards.remove_card(card.card_id)
            discard_table_card(state, _rm or card)

    return state.political.boxes[nation]


def activate_nation(state: gs.GameState, nation: gs.Nation,
                    method: str = "muster_die",
                    region: str = None) -> None:
    """
    Activate a nation (flip to colored side).

    Args:
        method: "muster_die", "attack", "companion_ability", "fellowship_declared", "event_card"
    """
    from persistent_effects import can_activate_nation, check_rohan_activated_discards

    # Check for Wormtongue blocking Rohan activation
    can_activate, blocker = can_activate_nation(state, nation, method,
                                                 region=region)
    if not can_activate:
        raise RuntimeError(f"Cannot activate {nation.value}: blocked by {blocker}")

    state.political.active[nation] = True

    # Check for Rohan activation discards (Wormtongue)
    if nation == gs.Nation.ROHAN:
        from persistent_effects import discard_table_card
        for card in check_rohan_activated_discards(state):
            _rm = state.persistent_cards.remove_card(card.card_id)
            discard_table_card(state, _rm or card)


def activate_nation_for_attack(state: gs.GameState, region: str, attacked: bool = False,
                                captured: bool = False) -> List[str]:
    """
    Handle political activation when a region is entered/attacked/captured.

    - Enemy entering a region: activate nation
    - Being attacked: activate + advance 1 step
    - Settlement captured: activate + advance 1 step (additional)

    Returns list of effects that occurred.
    """
    from board import sanitize, regions as board_regions, fid_to_name

    effects = []
    rkey = sanitize(region)

    # Find the nation that owns this region
    nation = None
    is_settlement = False

    if rkey in board_regions:
        reg_info = board_regions[rkey]
        parent_id = reg_info.get("parent_id")
        if parent_id:
            faction_name = fid_to_name.get(parent_id, "")
            # Map faction name to nation enum
            nation_map = {
                'gondor': gs.Nation.GONDOR,
                'rohan': gs.Nation.ROHAN,
                'elves': gs.Nation.ELVES,
                'dwarves': gs.Nation.DWARVES,
                'the north': gs.Nation.THE_NORTH,
                'north': gs.Nation.THE_NORTH,
                'isengard': gs.Nation.ISENGARD,
                'sauron': gs.Nation.SAURON,
                'southrons': gs.Nation.SOUTHRON_EASTERLINGS,
                'southron': gs.Nation.SOUTHRON_EASTERLINGS,
                'southrons and easterlings': gs.Nation.SOUTHRON_EASTERLINGS,
            }
            nation = nation_map.get(faction_name.lower())
            is_settlement = reg_info.get("is_city") or reg_info.get("is_town") or reg_info.get("is_stronghold")

    if not nation:
        return effects

    from persistent_effects import check_rohan_activated_discards, check_nation_advanced_discards

    # Activation when attacked - Wormtongue's attack exception applies
    # only at Edoras / Helm's Deep (QA 2026-07-10, region-aware check)
    if attacked or captured:
        from persistent_effects import can_activate_nation as _can9
        _ok9, _blk9 = _can9(state, nation, "attack", region=rkey)
        if not _ok9:
            effects.append(f"{nation.value} activation blocked by {_blk9}")
        elif not state.political.active.get(nation, False):
            state.political.active[nation] = True
            effects.append(f"{nation.value} activated")

            # Check for Rohan activation discards (Wormtongue)
            if nation == gs.Nation.ROHAN:
                from persistent_effects import discard_table_card
                for card in check_rohan_activated_discards(state):
                    _rm = state.persistent_cards.remove_card(card.card_id)
                    discard_table_card(state, _rm or card)
                    effects.append("Wormtongue discarded (Rohan activated)")

    # Advance for being attacked (a PASSIVE nation never advances into
    # At War - rulebook p.34 clamp, relevant when Wormtongue blocked
    # the activation above)
    if attacked:
        old_pos = state.political.boxes.get(nation, 0)
        _cap9 = 3 if state.political.active.get(nation, False) else 2
        if old_pos < _cap9:
            state.political.boxes[nation] = old_pos + 1
            effects.append(f"{nation.value} advanced to {old_pos + 1} (attacked)")

            # Check for nation advancement discards (Threats and Promises)
            from persistent_effects import discard_table_card
            for card in check_nation_advanced_discards(state, nation, "attack"):
                _rm = state.persistent_cards.remove_card(card.card_id)
                discard_table_card(state, _rm or card)
                effects.append("Threats and Promises discarded")

    # Additional advance for settlement captured
    if captured and is_settlement:
        old_pos = state.political.boxes.get(nation, 0)
        if old_pos < 3:
            state.political.boxes[nation] = old_pos + 1
            effects.append(f"{nation.value} advanced to {old_pos + 1} (settlement captured)")

    return effects


def is_nation_at_war(state: gs.GameState, nation: gs.Nation) -> bool:
    """Check if a nation is at war (position 3 on political track)."""
    return state.political.boxes.get(nation, 0) >= 3


def can_attack_with_army(state: gs.GameState, army,
                         side: "gs.Side" = None) -> Tuple[bool, str]:
    """
    Check if an army can attack (all units must be from nations at
    war).  Returns (can_attack, reason).

    QUEUE round (2026-08-17, census QUEUE_4): `side` filters to the
    ACTING side's units - the old all-units loop misjudged shared
    co-located siege entries (asking about the garrison judged the
    besieger's nations too, so the GUI's answer diverged from the
    ai.py menu).  ONE at-war rule body: the ai.py attack menu and
    the GUI both call here.
    """
    from combat import side_of_nation as _son
    units = [u for u in army.units
             if side is None or _son(u.nation) == side]
    if not units:
        return False, ("No units in army" if side is None
                       else "No units of the acting side in army")
    for unit in units:
        if not is_nation_at_war(state, unit.nation):
            return False, f"{unit.nation.value} is not at war"

    return True, "OK"



# (Hunt logic, move_fellowship, validate_recruit etc. remain unchanged)
#  Hunt logic, Fellowship movement, damage resolution
# Hunt Dice Functions - for GUI-driven hunt resolution
def get_hunt_params(state: gs.GameState) -> dict:
    """
    Get hunt parameters for the current fellowship position.

    Hunt Level = Shadow Action dice in the Hunt Box (rulebook p.40).
    Shadow rolls min(5, hunt_level) dice; success on 6+.
    Each FP die in the Hunt Box adds +1 modifier to each roll result.
    A natural 1 always misses regardless of modifier.
    """
    fp_dice_in_box = sum(1 for d in state.hunt_box
                         if d.side is gs.Side.FREE_PEOPLES)
    # Success on 6+ (rulebook p.40); FP dice in box give +1 per die to each
    # roll (fp_modifier), they never change the base threshold and the
    # Fellowship being revealed has no effect on the Hunt roll.
    base_threshold = 6

    # Hunt Level = Shadow Action dice in the Hunt Box only (rulebook p.40);
    # FP dice raise the modifier, not the number of dice rolled.
    hunt_level = sum(1 for d in state.hunt_box if d.side is gs.Side.SHADOW)
    num_dice = min(5, hunt_level)

    reroll_sources = []
    try:
        if _nazgul_present(state):
            reroll_sources.append("Nazgul present")
        if _shadow_army_present(state):
            reroll_sources.append("Shadow army present")
        if _in_shadow_stronghold(state):
            reroll_sources.append("Shadow stronghold")
    except Exception:
        pass

    return {
        "threshold": base_threshold,
        "num_dice": num_dice,
        "hunt_level": hunt_level,
        "fp_dice_in_box": fp_dice_in_box,
        "fp_modifier": fp_dice_in_box,
        "eyes": sum(1 for d in state.hunt_box
                    if d.current == gs.ActionDieFace.EYE),
        "reroll_sources": reroll_sources,
        "reroll_available": len(reroll_sources) > 0
    }


def roll_hunt_dice(num_dice: int, threshold: int,
                   fp_modifier: int = 0) -> dict:
    """
    Roll hunt dice and count hits.

    Each die result gets +fp_modifier added.
    A hit occurs when (roll + fp_modifier) >= threshold.
    A natural 1 always misses regardless of modifier.

    Returns rolls, modified results, and hit count.
    """
    rolls = roll_d6_batch(num_dice, 'hunt roll')
    modified = [r + fp_modifier for r in rolls]
    # Natural 1 always misses regardless of modifier
    hits = sum(1 for r, m in zip(rolls, modified)
               if r > 1 and m >= threshold)
    return {
        "rolls": rolls,
        "modified": modified,
        "hits": hits,
        "threshold": threshold,
        "fp_modifier": fp_modifier
    }


def reroll_hunt_dice(num_rerolls: int, threshold: int,
                     fp_modifier: int = 0) -> dict:
    """
    Shadow re-rolls missed hunt dice.
    Each re-roll gets +fp_modifier. Natural 1 always misses.
    Returns new rolls and additional hits.
    """
    rolls = roll_d6_batch(num_rerolls, 'hunt reroll')
    modified = [r + fp_modifier for r in rolls]
    hits = sum(1 for r, m in zip(rolls, modified)
               if r > 1 and m >= threshold)
    return {
        "rolls": rolls,
        "modified": modified,
        "hits": hits
    }


def draw_hunt_tile(state: gs.GameState, hunt_hits: int) -> dict:
    """
    Draw a hunt tile from the pool. Returns tile info without applying
    damage; the GUI resolves mitigation via gui._resolve_hunt_damage.
    The drawn tile is filed into hunt_discard HERE (drawn = used pile;
    Eyes included — they return to the pool on Mordor entry).
    """
    if not state.hunt_pool:
        # 1.136: refill at EVERY draw site — an emptied pool recycles
        # the used pile before failing the draw.
        _recycle_hunt_pool(state)
    if not state.hunt_pool:
        return {"success": False, "message": "Hunt pool empty"}

    tile = pick_hunt_tile(state)
    state.hunt_pool.remove(tile)
    state.hunt_discard.append(tile)   # drawn = used pile

    # Calculate base damage
    if tile.is_eye:
        damage = hunt_hits
    else:
        damage = tile.damage

    # Get tile name - use special_name if set, otherwise repr (e.g., "2R", "Eye")
    tile_name = tile.special_name if tile.special_name else repr(tile)

    # iter 38: hunt tile draws visible in SELF-PLAY traces
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'hunt_tile_draw',
                        {'tile_name': tile_name, 'ctx': 'hunt'})
    except Exception:
        pass

    return {
        "success": True,
        "tile": tile,
        "damage": damage,
        "reveal": tile.reveal,
        "is_eye": tile.is_eye,
        "tile_name": tile_name,
        "stops_movement": getattr(tile, 'stops_movement', False)
    }


_REACHABLE_CACHE: dict = {}   # (start, steps, mode) -> frozenset (static
                              # board graph memo; was referenced but never
                              # DEFINED - a latent NameError on the only
                              # code path that calls with this cache,
                              # exposed by the reveal-placement emitter
                              # 2026-07-24; self-play auto-place never hit
                              # it)


def get_reachable_regions(state: gs.GameState, start: str, max_steps: int,
                          mode: str = "normal") -> set:
    """
    Get regions reachable from start within max_steps.

    mode:
    - "normal": Standard movement
    - "reveal": Fellowship reveal - excludes ALL FP cities/strongholds
      (STATIC metadata only; the caller must add back Shadow-conquered
      FP settlements, which are legal reveal destinations)

    Result depends only on the static board graph + region metadata, so it
    is memoized globally (this was a per-MCTS-expansion BFS hot spot).
    """
    from collections import deque

    start_san = sanitize(start)
    _ck = (start_san, max_steps, mode)
    cached = _REACHABLE_CACHE.get(_ck)
    if cached is not None:
        return set(cached)
    reachable = set()
    visited = {start_san: 0}
    queue = deque([(start_san, 0)])

    while queue:
        current, steps = queue.popleft()

        if steps > 0:
            # Check if this is a valid destination based on mode
            meta = board.regions.get(current, {})
            is_fp_settlement = (is_fp_region(current) and
                               (meta.get("is_city") or meta.get("is_stronghold")))

            if mode == "reveal" and is_fp_settlement:
                pass  # Cannot end here when revealed
            else:
                reachable.add(current)

        if steps >= max_steps:
            continue

        for neighbor in state.board.adjacent(current):
            neighbor_san = sanitize(neighbor)
            if neighbor_san not in visited or visited[neighbor_san] > steps + 1:
                visited[neighbor_san] = steps + 1
                queue.append((neighbor_san, steps + 1))

    _REACHABLE_CACHE[_ck] = frozenset(reachable)
    return reachable


# Legacy function - still used by Mordor hunt and internal AI resolution
def _hunt_roll(state: gs.GameState, num_dice: int, thresh: int,
               fp_modifier: int = 0) -> int:
    """Roll hunt dice with FP modifier. Natural 1 always misses."""
    # P1: a hunt-time Flocks of Crebain discard adds +1 to ALL dice of
    # THIS hunt (rerolls included) - consumed here, single call site.
    # Consumed BEFORE the 0-dice early return: the bonus belongs to
    # THIS hunt and must never leak into a later one.
    _cb9 = getattr(state, '_crebain_hunt_bonus', 0)
    if _cb9:
        fp_modifier += _cb9
        state._crebain_hunt_bonus = 0
    if num_dice <= 0:
        return 0
    rolls = roll_d6_batch(num_dice, 'hunt roll')
    # Natural 1 always misses; otherwise hit if (roll + modifier) >= threshold
    successes = sum(1 for r in rolls if r > 1 and (r + fp_modifier) >= thresh)
    # iter 38: hunt rolls visible in SELF-PLAY traces (no-op otherwise)
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'dice_roll',
                        {'faces': list(rolls), 'ctx': 'hunt roll',
                         'side': 'SH'})
    except Exception:
        pass

    # One re-roll per source (Nazgul, Shadow army, Shadow stronghold)
    reroll_sources = 0
    try:
        if _nazgul_present(state):
            reroll_sources += 1
        if _shadow_army_present(state):
            reroll_sources += 1
        if _in_shadow_stronghold(state):
            reroll_sources += 1
    except Exception:
        pass

    # Re-roll misses (one per source); natural 1 still always misses
    misses = [i for i, r in enumerate(rolls)
              if r == 1 or (r + fp_modifier) < thresh]
    rerolls_to_do = min(reroll_sources, len(misses))
    _rr_list = []
    for j in range(rerolls_to_do):
        rr = roll_d6_batch(1, 'hunt reroll')[0]
        _rr_list.append(rr)
        if rr > 1 and (rr + fp_modifier) >= thresh:
            successes += 1
    if _rr_list:
        try:
            from game_trace import emit_annotation
            emit_annotation(state, 'dice_roll',
                            {'faces': _rr_list, 'ctx': 'hunt re-roll',
                             'side': 'SH'})
        except Exception:
            pass

    return successes


def _auto_sacrifice_companion(state: gs.GameState, dmg: int):
    """POLICY FALLBACK - NOT AN ENGINE RESOLVER (auto-resolver purge
    2026-08-16).  This heuristic RESOLVES A LEARNED FP DECISION
    (corruption vs sacrifice vs which companion).  It may be invoked
    ONLY by an explicit skip_auto_sacrifice=False request from a
    caller that owns the decision surface (legacy GUI manual play);
    the decision framework (settle -> pending_hunt -> owning side) is
    the production path.  The Merry incident (2026-08-16): a bare
    move_fellowship from the sided bridge reached this heuristic,
    which picked and killed the USER's companion silently.  The
    defaults are now inverted so a bare call can never land here;
    test_autoresolver_purge.py pins the caller census (exactly two
    call sites, both behind explicit False).

    Heuristic: pick the cheapest companion whose level >= damage for full
    absorption, otherwise pick the highest-level companion for maximum
    absorption.  Skips Gollum (level 0, can't absorb).
    """
    if dmg <= 0:
        return
    candidates = [c for c in state.fellowship.companions
                  if c.name != "Gollum" and c.level > 0]
    if not candidates:
        _apply_fellowship_damage(state, dmg)
        return
    # Prefer exact/cheap absorption: smallest level >= damage
    full_absorb = [c for c in candidates if c.level >= dmg]
    if full_absorb:
        victim = min(full_absorb, key=lambda c: c.level)
    else:
        # No companion can fully absorb; pick highest level for max absorption
        victim = max(candidates, key=lambda c: c.level)
    if victim == state.fellowship.guide:
        _apply_fellowship_damage(state, dmg, sacrifice_guide=True)
    else:
        _apply_fellowship_damage(state, dmg, kill_name=victim.name)


def is_nazgul_figure(l) -> bool:
    """The Witch-king is a Nazgul for ALL rules (almanac) - the engine
    models him as a Minion, so isinstance(Nazgul) checks missed him
    across four cards (QA 2026-07-11): Dreadful Spells dice/gate,
    Ringwraiths moves, Black Captain Commands, Nazgul Search reveal."""
    if isinstance(l, gs.Nazgul):
        return True
    import re as _re
    return 'witchking' in _re.sub(
        r'[^a-z]', '', str(getattr(l, 'name', '')).lower())


def check_companion_persistent_discards(state: gs.GameState) -> list:
    """FP table cards with a requires_companion clause (Axe and Bow:
    Gimli+Legolas, Horn of Gondor: Boromir, Wizard's Staff: Gandalf
    the Grey) discard IMMEDIATELY when all required companions have
    left the Fellowship.  QA 2026-07-11 cluster A: PersistentCardState.
    check_companion_requirements existed but was never called."""
    pc = getattr(state, 'persistent_cards', None)
    if pc is None:
        return []
    try:
        gone = pc.check_companion_requirements(
            state.fellowship.companions)
    except Exception:
        return []
    names = []
    from persistent_effects import discard_table_card
    for c in list(gone):
        _rm = pc.remove_card(c.card_id)
        discard_table_card(state, _rm or c)
        names.append(c.card_name)
    return names


def card_deck_of(cid) -> Optional[str]:
    """THE deck classifier for one hand entry: 'character' | 'strategy'
    | None (not a card).  Card-table ids carry their deck in the id
    (cards.load_cards builds `<SIDE>_<Deck>_<Name>`); the live
    driver's human-hand placeholders (HU_<side>_CHAR_n /
    HU_<side>_STRAT_n) answer by their minted shape.  Every consumer
    that must sort a hand by deck (WWSAT, the opponent overlay) reads
    it here - a second classifier cannot disagree about a placeholder."""
    if not isinstance(cid, str):
        return None
    if '_Character_' in cid:
        return 'character'
    if '_Strategy_' in cid:
        return 'strategy'
    if cid.startswith('HU_'):
        if '_CHAR_' in cid:
            return 'character'
        if '_STRAT_' in cid:
            return 'strategy'
    return None


def _wwsat_resolve(state: gs.GameState, source, card_id=None):
    """Apply ONE 'Worn with Sorrow and Toil' FP-Character discard.

    1.211f: the Shadow chooses the SOURCE - kill a played FP Character
    card from the TABLE, or discard a RANDOM FP Character card from the
    FP hand.  source='table' -> the table card matching card_id (else
    the first FP Character persistent); source='hand' -> a random FP
    hand Character card.  source=None is the LEGACY hand-first heuristic
    (hand random if any, else table) EXCEPT a teacher-forced source on
    state._wwsat_forced wins when present (closed_loop faithfulness)."""
    from persistent_effects import discard_table_card
    import random as _rnd
    forced = getattr(state, '_wwsat_forced', None)
    if source is None and forced is not None:
        source = forced.get('source')
        if card_id is None:
            card_id = forced.get('card_id')
    state._wwsat_forced = None            # consume any teacher force
    hand = state.hands[gs.Side.FREE_PEOPLES].hand
    chars = [c for c in hand if card_deck_of(c) == 'character']
    pcards = getattr(state, 'persistent_cards', None)
    table_chars = (pcards.get_fp_character_cards()
                   if pcards is not None else [])
    use_table = (source == 'table') or (source is None and not chars)
    if use_table and table_chars:
        target = None
        if card_id is not None:
            target = next((c for c in table_chars
                           if c.card_id == card_id), None)
        if target is None:
            target = table_chars[0]
        _rm = pcards.remove_card(target.card_id)
        discard_table_card(state, _rm or target)
        return
    if chars:
        _gone = pick_random_card(chars, 'WWSAT hand discard')
        hand.remove(_gone)
        # CL-217: physical face-up discard pile (the app discarded it)
        _dfu = getattr(state, 'discard_face_up', None)
        if isinstance(_dfu, dict) and 'fp_character' in _dfu:
            _dfu['fp_character'].append(_gone)


def _on_companion_casualty(state: gs.GameState, name: str,
                           casualty: bool = True):
    """'Worn with Sorrow and Toil': each companion CASUALTY forces the
    Shadow to discard an FP Character card - randomly from the FP hand
    OR from the table (1.211f: a LEARNED Shadow SOURCE decision when
    BOTH sources exist; 1.210g: MANDATORY, no decline path).

    USER RULING 2026-08-15: SEPARATION-death is legally distinct from
    being taken as a hunt casualty — a hobbit GUIDE absorbing hunt
    damage separates (which in Mordor kills), and WWSAT does NOT
    trigger on it; with a hobbit guide WWSAT fires only via a real
    casualty-taking (e.g. Foul Thing from the Deep's forced pick).
    casualty=False = leaves-play cleanup only, no WWSAT.

    RECORD-THEN-DRAIN: casualty resolution cannot suspend for a pending,
    so a both-sources SELF-PLAY decision is recorded on
    state._wwsat_owed and drained as a learned pending after the action
    (ai._offer_wwsat_discard).  The teacher forces the LOGGED source via
    state._wwsat_forced and resolves inline."""
    check_companion_persistent_discards(state)
    if not casualty:
        return
    from persistent_effects import check_companion_casualty_effect
    # ARITY REPAIR 2026-08-15: the dead `companion_name` parameter was
    # dropped from check_companion_casualty_effect (WWSAT fires on ANY
    # companion casualty) without updating this sole caller, so EVERY
    # companion casualty raised TypeError (CL-222 / CL-247 ERROR).
    if not check_companion_casualty_effect(state):
        return
    hand = state.hands[gs.Side.FREE_PEOPLES].hand
    # ONE deck classifier (card_deck_of) - live human-hand placeholders
    # (HU_FR_CHAR_n) are Character cards too
    chars = [c for c in hand if card_deck_of(c) == 'character']
    pcards = getattr(state, 'persistent_cards', None)
    table_chars = (pcards.get_fp_character_cards()
                   if pcards is not None else [])
    if not chars and not table_chars:
        return
    import combat as _cmbwwsat
    if (chars and table_chars
            and _cmbwwsat.FINAL_HITS_HOOK is None
            and getattr(state, '_wwsat_forced', None) is None):
        owed = getattr(state, '_wwsat_owed', None)
        if owed is None:
            owed = []
            state._wwsat_owed = owed
        owed.append(True)
        return
    _wwsat_resolve(state, None)


# User-verified rule: a RANDOM hunt casualty that draws a Hobbit
# (Meriadoc / Peregrin) is NOT eliminated — the Free Peoples player places
# him on the board as if he had separated from the Fellowship (separation
# movement = progress + level; reference game: Merry drawn T2, placed at
# Bree). Separation is impossible on the Mordor track, so there the Hobbit
# is eliminated normally.
HUNT_PLACEABLE_HOBBITS = ("Meriadoc", "Peregrin")


def hobbit_hunt_placement_applies(state: gs.GameState, victim_name) -> bool:
    return (str(victim_name) in HUNT_PLACEABLE_HOBBITS
            and not state.fellowship.mordor_entered)


def remove_hobbit_for_placement(state: gs.GameState, victim, dmg: int):
    """Remove a hunt-drawn Hobbit from the Fellowship WITHOUT eliminating
    him: he absorbs his level from the Hunt damage (excess becomes
    corruption) and awaits placement on the board via the separation
    rules.

    USER RULING 2026-08-16 (verbatim: "wwsat still fires on hobbits
    being taken as randoms"): being DRAWN as a hunt casualty IS
    casualty-taking - WWSAT fires even though the hobbit physically
    separates (Foul Thing's forced pick is the same row).  The old
    'not a casualty' comment described guide ABSORPTION, the other
    row.  Self-play/live only: the TEACHER's WWSAT discards are
    forced from the log's own records (certified behavior kept).
    Returns the Companion object to place."""
    if victim in state.fellowship.companions:
        state.fellowship.companions.remove(victim)
    import combat as _cmbrhp
    if _cmbrhp.FINAL_HITS_HOOK is None:
        _on_companion_casualty(state, victim.name, casualty=True)
    if state.fellowship.guide not in state.fellowship.companions:
        change_guide(state, allow_pending=True)
    net = max(0, dmg - victim.level)
    state.fellowship.corruption = min(12, state.fellowship.corruption + net)
    return victim


def place_separated_companion(state: gs.GameState, comp, region: str):
    """Put a companion figure on the board at *region* (army leader slot),
    as separation does."""
    rk = sanitize(str(region))
    # 1.148: same duplicate-key class as separate_companions
    army = get_or_create_army(state, rk)
    army.leaders.append(comp)
    # 1.149b: arrival activation applies here too (forced walks,
    # hunt-hobbit placements, card-effect separations)
    _check_companion_activates_nation(state, rk, army)


def _apply_fellowship_damage(state: gs.GameState,
                             dmg: int,
                             *,
                             sacrifice_guide: bool = False,
                             kill_name: Optional[str] = None):
    if dmg <= 0: return
    absorbed = 0
    guide = state.fellowship.guide

    # THE TWO-ROW HOBBIT RULE (USER 2026-08-15 + final scope
    # 2026-08-16, "foul thing = random casualty, just forced"):
    #   row 1  GUIDE ABSORPTION (sacrifice_guide): the hobbit guide
    #          soaks via his SEPARATION ability - not casualty-taking,
    #          WWSAT never fires; separates ALIVE outside Mordor,
    #          dies on the Mordor track.
    #   row 2  TAKEN AS A CASUALTY (kill_name: random draw, teacher-
    #          forced name, Foul Thing's forced pick, legacy heuristic
    #          pick): WWSAT FIRES - but a HOBBIT still physically
    #          SEPARATES outside Mordor ("if the hobbits are killed
    #          as a random they still seperate except in mordor");
    #          only Mordor makes it a death.  Non-hobbits die.
    # Teacher keeps its certified behavior for row-2 placement (the
    # log's own drags place the hobbit); WWSAT for the teacher is
    # forced from records inside _on_companion_casualty.
    def _is_hobbit(c):
        return c.name in ('Meriadoc', 'Peregrin')

    def _remove_absorbing(c, *, absorption):
        """Remove companion c absorbing damage; returns level."""
        import combat as _cmbrow
        state.fellowship.companions.remove(c)
        _no_wwsat = absorption and _is_hobbit(c)   # row 1 only
        _on_companion_casualty(state, c.name, casualty=not _no_wwsat)
        if _is_hobbit(c) and not state.fellowship.mordor_entered \
                and (absorption or _cmbrow.FINAL_HITS_HOOK is None):
            # separation, alive: place at the fellowship's region
            place_separated_companion(state, c, fellowship_region(state))
        return c.level

    if sacrifice_guide and guide.name != "Gollum":
        absorbed = _remove_absorbing(guide, absorption=True)
        change_guide(state, allow_pending=True)
    elif kill_name:
        for c in list(state.fellowship.companions):
            if c.name == kill_name and c.name != "Gollum":
                absorbed = _remove_absorbing(c, absorption=False)
                if c == guide:
                    change_guide(state, allow_pending=True)
                break

    net = max(0, dmg - absorbed)
    # USER RULING 2026-07-21: the 'Axe and Bow'/'Horn of Gondor'
    # discard-to-reduce is now a LEARNED FP decision resolved BEFORE
    # this point (see _offer_hunt_reduction + the card_option
    # executor); the old auto-consume here is removed. The teacher
    # forces the LOGGED use/decline (closed_loop hunt forcing).
    state.fellowship.corruption = min(
        12, state.fellowship.corruption + max(0, net))


def _offer_hunt_reduction(state: gs.GameState) -> bool:
    """USER RULING 2026-07-21: 'Axe and Bow'/'Horn of Gondor'
    discard-to-reduce-Hunt-damage is a LEARNED FP yes/no decision
    (card text: 'you may discard ... to reduce the Hunt damage by one
    (to a minimum of zero). Any remaining Hunt damage must be
    confronted normally.').

    Opens a pending_card_option interpose riding the FROZEN encode
    (card-idx + option-count + direct/decider channels already expose
    it; NO new encoded field, NO new pending-phase subtype — the
    Wizard's Staff interpose is the template). Returns True iff the
    pending opened -> the caller must SUSPEND and not settle.

    Self-play only (combat.FINAL_HITS_HOOK is None); the teacher
    forces the LOGGED use/decline and never calls this. ONE card per
    settle (almanac Orc Patrol C12: one per Hunt tile)."""
    if getattr(state, '_pending_hunt_damage', 0) <= 0:
        return False
    if getattr(state, 'pending_card_option', None) is not None:
        return False
    import combat as _cmbhr
    if _cmbhr.FINAL_HITS_HOOK is not None:
        return False                      # teacher forces from the log
    pcards = getattr(state, 'persistent_cards', None)
    if pcards is None:
        return False
    card = None
    for ptype in (gs.PersistentCardType.AXE_AND_BOW,
                  gs.PersistentCardType.HORN_OF_GONDOR):
        card = pcards.get_by_type(ptype)
        if card is not None:
            break
    if card is None:
        return False
    state.pending_card_option = {
        'card_id': card.card_id, 'card_name': card.card_name,
        'side': 'fp', 'decider': 'fp', 'direct': True,
        'hunt_reduction': True,
        'options': [{'use_hunt_reduction': 1, 'card_id': card.card_id},
                    {'use_hunt_reduction': 0}]}
    return True


def _settle_hunt_damage(state: gs.GameState) -> None:
    """Confront the accumulated _pending_hunt_damage: optional Gollum
    reveal, else the companion-sacrifice decision, else direct
    corruption. Factored out of the fellowship_move executor
    (2026-07-21) so the Axe/Horn reduction interpose can resume it
    after the FP's choice. No-op when no damage pends or a decision is
    already open."""
    pending_dmg = getattr(state, '_pending_hunt_damage', 0)
    if pending_dmg <= 0 or state.pending_hunt is not None:
        return
    guide = state.fellowship.guide
    _in_fp_settlement = False
    if guide is not None:
        _cur = sanitize(fellowship_region(state))
        _rinfo = board.regions.get(_cur, {})
        _is_stype = _rinfo.get('is_city') or _rinfo.get('is_stronghold')
        _not_conq = _cur not in state.enemy_controlled.get(
            gs.Side.SHADOW, set())
        _in_fp_settlement = bool(_is_stype and is_fp_region(_cur)
                                 and _not_conq)
    gollum_can_reveal = (
        getattr(guide, 'guide_ability', None) == 'no_reveal_on_damage'
        and state.fellowship.hidden and not _in_fp_settlement)
    if gollum_can_reveal:
        # 1.229: opening a decision CONSUMES the accumulator - from here
        # _pending_hunt_damage holds only damage not yet under decision
        # (tiles drawn DURING the decision), settled as its own decision
        # at finish (1.132: never pooled)
        state.pending_hunt = {'damage': pending_dmg,
                              'phase': 'hunt_gollum'}
        state._pending_hunt_damage = 0
        return
    sacrificable = [c for c in state.fellowship.companions
                    if c.name != "Gollum" and c.level > 0]
    if sacrificable:
        state.pending_hunt = {'damage': pending_dmg,
                              'phase': 'hunt_sacrifice'}
        state._pending_hunt_damage = 0
    else:
        _apply_fellowship_damage(state, pending_dmg)
        state._pending_hunt_damage = 0


def pop_hobbit_place_queue(state: gs.GameState) -> None:
    """Open the next queued hobbit_place decision (one per separated
    hobbit); no-op while another hunt decision is in flight."""
    if getattr(state, 'pending_hunt', None) is not None:
        return
    q = getattr(state, '_hobbit_place_queue', None)
    if not q:
        return
    hob = q.pop(0)
    state.pending_hunt = {
        'phase': 'hobbit_place',
        'companion': getattr(hob, 'name', str(hob)),
        'level': int(getattr(hob, 'level', 1) or 1),
        'companion_obj': hob,
        'damage': 0,
    }


def _finish_hunt_decision(state: gs.GameState) -> None:
    """Close the in-flight hunt decision; damage accumulated DURING it
    (stronghold draws on a reveal placement) opens its OWN decision
    (1.229/1.132), else the hobbit-place queue drains."""
    state.pending_hunt = None
    if getattr(state, '_pending_hunt_damage', 0) > 0:
        _settle_hunt_damage(state)
    else:
        pop_hobbit_place_queue(state)


def continue_hunt_chain(state: gs.GameState, dmg: int,
                        casualty_taken: bool) -> None:
    """Route REMAINING hunt damage to the next guide's ability or a
    further decision; apply as corruption when no option remains
    (USER 2026-07-09 chained guide abilities)."""
    if dmg <= 0:
        _finish_hunt_decision(state)
        return
    g2 = state.fellowship.guide
    _can_reveal = (g2 is not None
                   and getattr(g2, 'guide_ability', None)
                   == 'no_reveal_on_damage' and state.fellowship.hidden)
    _hob_guide = (g2 is not None and getattr(g2, 'guide_ability', None)
                  == 'sacrifice_reduce_hunt')
    _sacrificable = (not casualty_taken and any(
        c.name != 'Gollum' and c.level > 0
        for c in state.fellowship.companions))
    if _can_reveal:
        state.pending_hunt = {'damage': dmg, 'phase': 'hunt_gollum',
                              'casualty_taken': casualty_taken}
    elif _sacrificable or _hob_guide:
        state.pending_hunt = {'damage': dmg, 'phase': 'hunt_sacrifice',
                              'casualty_taken': casualty_taken}
    else:
        _apply_fellowship_damage(state, dmg)
        _finish_hunt_decision(state)


def hunt_decision_menu(state: gs.GameState) -> list:
    """Legal (action_type, params) pairs for the open gollum/sacrifice
    hunt decision; hobbit_place region menus stay with the caller."""
    ph = state.pending_hunt
    if ph is None:
        return []
    phase = ph.get('phase')
    out = []
    if phase == 'hunt_gollum':
        out.append(('hunt_gollum_reveal', {}))
        out.append(('hunt_gollum_no_reveal', {}))
    elif phase == 'hunt_sacrifice':
        # rulebook p.42: the casualty is OPTIONAL; corruption always legal
        out.append(('hunt_take_corruption', {}))
        if not ph.get('casualty_taken') and any(
                c.name != 'Gollum' and c.level > 0
                for c in state.fellowship.companions):
            out.append(('hunt_sacrifice_guide', {}))
            out.append(('hunt_sacrifice_random', {}))
        g9 = state.fellowship.guide
        if g9 is not None and getattr(g9, 'guide_ability', None) \
                == 'sacrifice_reduce_hunt':
            out.append(('hunt_guide_separate', {'companion': g9.name}))
    return out


def resolve_hunt_decision(state: gs.GameState, kind: str, *,
                          forced_companion=None,
                          place_region: Optional[str] = None) -> None:
    """Execute one hunt decision against state.pending_hunt.  The ONE
    implementation behind the ai.py executors and the hunt_dist kernel
    (drift structurally impossible)."""
    ph = state.pending_hunt
    if not ph:
        return

    if kind == 'hunt_gollum_reveal':
        state.fellowship.hidden = False
        # 1.229: accumulator is 0 here (consumed at open) - stronghold
        # tiles drawn by this placement land in it and MUST survive to
        # their own decision, never be zeroed with this one.
        # AUTO-RESOLVER PURGE (2026-08-16): reveal placement is a
        # LEARNED FP DECISION - on the learned path DEFER to
        # ai.open_reveal_placement (runs at the apply_action tail
        # once the chained damage decisions settle; forced cases
        # resolve via the certified auto path there).  TEACHER keeps
        # the immediate log-driven placement.
        import combat as _cmbgr9
        if _cmbgr9.FINAL_HITS_HOOK is None \
                and not state.fellowship.mordor_entered:
            state._reveal_place_due = True
            state._sh_tile_drawn_regions = set()
        else:
            try:
                _auto_reveal_place(state)
            except Exception:
                pass
        new_dmg = max(0, ph['damage'] - 1)
        sacrificable = [c for c in state.fellowship.companions
                        if c.name != "Gollum" and c.level > 0]
        if sacrificable and new_dmg > 0:
            # thread casualty_taken (one-casualty-per-hunt invariant)
            state.pending_hunt = {'damage': new_dmg,
                                  'phase': 'hunt_sacrifice',
                                  'casualty_taken':
                                      ph.get('casualty_taken', False)}
        else:
            if new_dmg > 0:
                _apply_fellowship_damage(state, new_dmg)
            _finish_hunt_decision(state)

    elif kind == 'hunt_gollum_no_reveal':
        dmg = ph['damage']
        sacrificable = [c for c in state.fellowship.companions
                        if c.name != "Gollum" and c.level > 0]
        if sacrificable and dmg > 0:
            state.pending_hunt = {'damage': dmg,
                                  'phase': 'hunt_sacrifice',
                                  'casualty_taken':
                                      ph.get('casualty_taken', False)}
        else:
            if dmg > 0:
                _apply_fellowship_damage(state, dmg)
            _finish_hunt_decision(state)

    elif kind == 'hunt_take_corruption':
        _apply_fellowship_damage(state, ph['damage'])
        _finish_hunt_decision(state)

    elif kind == 'hunt_sacrifice_guide':
        dmg = int(ph['damage'])
        g9 = state.fellowship.guide
        absorbed = min(dmg, int(getattr(g9, 'level', 0) or 0))
        _apply_fellowship_damage(state, absorbed, sacrifice_guide=True)
        continue_hunt_chain(state, dmg - absorbed, casualty_taken=True)

    elif kind == 'hunt_guide_separate':
        f9 = state.fellowship
        hob = f9.guide
        if hob is not None and hob in f9.companions:
            f9.companions.remove(hob)
        if hob is not None:
            if f9.mordor_entered:
                # USER 2026-07-09: separation on the Mordor track kills
                # USER 2026-08-15: ...but it is a SEPARATION-death, not
                # a hunt casualty — WWSAT must not trigger
                _on_companion_casualty(state, hob.name, casualty=False)
                try:
                    from game_trace import emit_annotation
                    emit_annotation(state, 'random_draw', {
                        'ctx': 'separation on Mordor track',
                        'result': f'{hob.name} eliminated'})
                except Exception:
                    pass
            else:
                q = getattr(state, '_hobbit_place_queue', None)
                if q is None:
                    q = state._hobbit_place_queue = []
                q.append(hob)
        change_guide(state, allow_pending=True)
        continue_hunt_chain(state, max(0, int(ph.get('damage', 0)) - 1),
                            casualty_taken=bool(ph.get('casualty_taken')))

    elif kind == 'hunt_sacrifice_random':
        dmg = ph['damage']
        candidates = [c for c in state.fellowship.companions
                      if c.name != "Gollum" and c.level > 0]
        if candidates:
            victim = None
            if forced_companion:
                victim = next(
                    (c for c in candidates
                     if sanitize(c.name)
                     == sanitize(str(forced_companion))), None)
                if victim is not None:
                    # stream-compat: consume the unforced path's draw
                    random.choice(candidates)
            if victim is None:
                victim = pick_random_companion(candidates,
                                               'hunt casualty draw')
            try:
                from game_trace import emit_annotation
                emit_annotation(state, 'random_draw', {
                    'ctx': 'hunt casualty draw',
                    'result': victim.name,
                    'pool': [c.name for c in candidates]})
            except Exception:
                pass
            if hobbit_hunt_placement_applies(state, victim.name):
                remove_hobbit_for_placement(state, victim, dmg)
                state.pending_hunt = {
                    'phase': 'hobbit_place',
                    'companion': victim.name,
                    'level': victim.level,
                    'companion_obj': victim,
                }
                return
            if victim == state.fellowship.guide:
                _apply_fellowship_damage(state, dmg,
                                         sacrifice_guide=True)
            else:
                _apply_fellowship_damage(state, dmg,
                                         kill_name=victim.name)
        else:
            _apply_fellowship_damage(state, dmg)
        _finish_hunt_decision(state)

    elif kind == 'hobbit_place':
        comp_obj = ph.get('companion_obj')
        if comp_obj is None:
            comp_obj = gs.Companion(name=str(ph.get('companion', '')),
                                    level=1, leadership=1)
        target = place_region if place_region is not None \
            else fellowship_region(state)
        place_separated_companion(state, comp_obj, target)
        _finish_hunt_decision(state)

    else:
        raise RuntimeError('unknown hunt decision %r' % kind)


def _recycle_hunt_pool(state: gs.GameState) -> int:
    """USER RULE (1.136): an empty Hunt Pool is refilled from the USED
    pile - ALL used tiles return, except special (card/Mordor) tiles
    before Mordor entry.  REMOVED tiles (Challenge of the King class)
    live in state.hunt_removed and never return.  Returns the number
    of tiles recycled."""
    if state.hunt_pool:
        return 0
    back = []
    for tile in state.hunt_discard[:]:
        if not state.fellowship.mordor_entered \
                and getattr(tile, 'is_special', False):
            continue
        back.append(tile)
        state.hunt_discard.remove(tile)
    if back:
        import random as _r136
        _r136.shuffle(back)
        state.hunt_pool.extend(back)
        try:
            from game_trace import emit_annotation
            emit_annotation(state, 'hunt_pool_recycled',
                            {'count': len(back)})
        except Exception:
            pass
    return len(back)


def _offer_wizard_staff(state: gs.GameState, stash: dict) -> bool:
    """Open the Wizard's Staff prevent-the-draw interpose (self-play
    only); True iff the pending opened and the caller must SUSPEND.
    1.232: the ONE implementation for both the outside-Mordor and the
    Mordor-track draw paths."""
    import combat as _cmbw9
    if _cmbw9.FINAL_HITS_HOOK is not None:
        return False
    _pcw9 = getattr(state, 'persistent_cards', None)
    if _pcw9 is None:
        return False
    try:
        _wsw9 = _pcw9.get_by_type(gs.PersistentCardType.WIZARD_STAFF)
    except Exception:
        _wsw9 = None
    if _wsw9 is None or getattr(state, 'pending_card_option',
                                None) is not None:
        return False
    state._wizardstaff_stash = stash
    state.pending_card_option = {
        'card_id': getattr(_wsw9, 'card_id', '__wizardstaff__'),
        'card_name': "Wizard's Staff",
        'side': 'fp', 'decider': 'fp', 'direct': True,
        'options': [{'wizardstaff_prevent': True},
                    {'wizardstaff_prevent': False}]}
    return True


def _resolve_hunt_tile(state: gs.GameState,
                       successes: int,
                       *, mordor: bool, _skip_wizard: bool = False,
                       eye_no_effect: bool = False):
    # Wizard's Staff (USER 2026-07-16 item 10): before the Shadow draws a
    # Hunt tile, the FP player may discard "Wizard's Staff" to PREVENT the
    # draw entirely.  _skip_wizard=True on the resume avoids re-asking.
    # eye_no_effect: Balrog-of-Moria extra draw — an Eye is discarded
    # WITHOUT effect (almanac QA 2026-07-11); the eye test runs on the
    # tile the funnel ACTUALLY drew (never a pool[-1] peek, which
    # diverges under the hunt_dist kernel / live-bridge pop patches).
    if not _skip_wizard and _offer_wizard_staff(
            state, {'successes': successes, 'mordor': mordor,
                    'eye_no_effect': eye_no_effect}):
        return
    if not state.hunt_pool:
        # 1.136 (USER RULE, supersedes 1.135's fizzle): recycle the
        # used pile back into the pool.  Only if NOTHING is
        # recyclable either: teacher keeps the drift tripwire,
        # self-play fizzles.
        _recycle_hunt_pool(state)
    if not state.hunt_pool:
        import combat as _cmb135
        if _cmb135.FINAL_HITS_HOOK is not None:
            raise RuntimeError("Hunt pool exhausted")
        return
    tile = pop_hunt_tile(state)

    # iter 38: hunt tile draws visible in SELF-PLAY traces
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'hunt_tile_draw', {
            'tile_name': tile.special_name if tile.special_name
            else repr(tile),
            'ctx': ('balrog extra draw' if eye_no_effect
                    else 'mordor hunt' if mordor else 'hunt')})
    except Exception:
        pass

    if eye_no_effect and getattr(tile, 'is_eye', False):
        # almanac (QA 2026-07-11): an Eye drawn for the Balrog is
        # discarded WITHOUT effect (no interposes needed).
        state.hunt_discard.append(tile)
        return

    # 'Mithril Coat and Sting' (QA 2026-07-11 cluster G): after the
    # SHADOW draws a hunt tile the FP player may discard the table
    # card to force a redraw - a learned FP decision interposed
    # BEFORE the tile applies (so the choice precedes Shelob's damage
    # die, per the almanac).  All _resolve_hunt_tile draws are Shadow
    # draws (regular hunts, Mordor track, Shadow card effects).
    # TEACHER walks replay the humans' own Mithril usage from the log
    # (their redraw is its own discard + tile row; the forcing hooks
    # already schedule the actual tiles) - the interposed pending is
    # the LEARNED path only (same pattern as the casualty-policy
    # phase; canary Wirbelwind/kapcerz stalled on it 2026-07-11).
    import combat as _cmbm9
    _pcm9 = (getattr(state, 'persistent_cards', None)
             if _cmbm9.FINAL_HITS_HOOK is None else None)
    _mc9 = None
    if _pcm9 is not None:
        try:
            _mc9 = _pcm9.get_by_type(
                gs.PersistentCardType.MITHRIL_COAT)
        except Exception:
            _mc9 = None
    if _mc9 is not None and getattr(state, 'pending_card_option',
                                    None) is None:
        state._mithril_stash = {'tile': tile, 'successes': successes,
                                'mordor': mordor,
                                'eye_no_effect': eye_no_effect}
        state.pending_card_option = {
            'card_id': getattr(_mc9, 'card_id', '__mithril__'),
            'card_name': 'Mithril Coat and Sting',
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'options': [{'mithril_redraw': True},
                        {'mithril_redraw': False}]}
        return

    _apply_hunt_tile(state, tile, successes, mordor=mordor)


def _apply_hunt_tile(state: gs.GameState, tile, successes: int,
                     *, mordor: bool):
    """Apply a drawn hunt tile (split from _resolve_hunt_tile for the
    Mithril interpose - QA 2026-07-11 cluster G)."""
    dmg = tile.damage
    if tile.is_eye:
        dmg = successes if not mordor else len(state.hunt_box)

    # W3 BUG-3 FIX (2026-07-20, W11 hunt wave): ACCUMULATE, never
    # assign.  A single fellowship move can draw more than one damage
    # tile before the FP player's damage decision is reached — the
    # Balrog-of-Moria extra draw (move_fellowship) and the
    # Shadow-stronghold extra draws (_draw_shadow_stronghold_tile,
    # which already used +=).  The old `=` re-assigned _pending_hunt_
    # damage on the second tile, SILENTLY DROPPING the first tile's
    # damage (probe: primary 2 + Balrog 3 resolved as 3, not 5).
    # CURRENT_RULES sec.1: "each tile resolves its own damage decision
    # first" and damage is never lost between tiles — accumulation into
    # the single pending scalar keeps every tile's damage (consistent
    # with the shadow-stronghold path).
    # NEGATIVE damage (FP special tiles: Smeagol Helps -1, Phial -2) is
    # a HEAL and applies immediately: every _pending_hunt_damage
    # consumer treats <=0 as "nothing pending" and never clears it, so
    # a negative accumulator would eat the heal, poison the FP ring
    # victory gates, and offset the next hunt's real damage.
    if dmg < 0:
        state.fellowship.corruption = max(
            0, state.fellowship.corruption + dmg)
    else:
        state._pending_hunt_damage = getattr(
            state, '_pending_hunt_damage', 0) + dmg  # UI/AI reads this

    state.hunt_discard.append(tile)  # drawn = used pile
    if tile.reveal:
        # Gollum guide ability: ignore reveals on non-eye, non-special
        # tiles (USER 2026-07-11: applies on the Mordor track too -
        # tiles DO reveal there, Gollum blocks only ordinary ones)
        guide = state.fellowship.guide
        if (guide and getattr(guide, 'guide_ability', None) == 'no_reveal_on_damage'
                and not tile.is_eye and not getattr(tile, 'is_special', False)):
            pass  # Gollum blocks the reveal
        else:
            state.fellowship.hidden = False
            # Per rulebook: when revealed, FP must move Ring-bearers figure
            # from last known position, up to progress steps. Cannot end in
            # unconquered FP City/Stronghold. Auto-place toward Mordor.
            if state.fellowship.progress > 0 and not state.fellowship.mordor_entered:
                # 1.132 (USER RULE): the tile's own damage decision
                # resolves FIRST; only then is the figure placed and any
                # Shadow-stronghold tile drawn as its OWN new decision
                # (each tile = a separate casualty/corruption choice -
                # damage is NEVER pooled, and the stronghold tile is not
                # even seen until the first tile fully resolves).
                # Self-play defers the placement to the damage
                # resolution point; teacher keeps the certified
                # immediate order (logged records drive its walk).
                import combat as _cmb132
                if _cmb132.FINAL_HITS_HOOK is None:
                    # USER RULING 2026-08-16: reveal placement is a
                    # LEARNED FP DECISION.  Defer BOTH cases (damage
                    # or not) to the post-settle decision point
                    # (ai.open_reveal_placement: forced cases resolve
                    # via the certified auto path, a real choice opens
                    # an FP pending).  Arm the one-draw-per-region-
                    # per-event guard NOW so the deferred 'revealed
                    # in' draw and the placement executor share one
                    # event scope.
                    state._reveal_place_due = True
                    state._sh_tile_drawn_regions = set()
                else:
                    _auto_reveal_place(state)


def _auto_reveal_place(state: gs.GameState):
    """When fellowship is revealed, move Ring-bearers figure toward Mordor.

    Per rulebook: move from last known position, up to progress steps.
    Cannot end in unconquered FP City/Stronghold (unlike declaration).
    Resets progress to 0.
    """
    cur = fellowship_region(state)
    progress = state.fellowship.progress
    import board as board_mod

    def _is_sh_stronghold(r):
        meta = board_mod.regions.get(r, {})
        return (meta.get('is_stronghold') and is_shadow_region(r)
                and r not in state.enemy_controlled[
                    gs.Side.FREE_PEOPLES])

    # Revealed IN a Shadow stronghold (Moria class): extra tile draw
    # before any movement — eyes do nothing (USER 2026-07-09: was
    # nonfunctional on the hunt-reveal path; only the manual walk API
    # drew these)
    state._sh_tile_drawn_regions = set()   # one draw per region per event
    _started_in_sh = _is_sh_stronghold(cur)
    if _started_in_sh:
        _draw_shadow_stronghold_tile(state, cur, "revealed in")

    if progress <= 0:
        return

    # USER RULING 2026-07-20b (trace w5_g2 T9): revealed AT a Mordor
    # gateway, the pawn STAYS on the gate (progress resets) — it must
    # never region-step INTO Mordor's interior; entry happens at the
    # next turn-start Fellowship pre-step (1.155). Self-play only
    # (teacher placement is log-driven).
    import combat as _cmb_gate
    if (_cmb_gate.FINAL_HITS_HOOK is None
            and cur in (sanitize('morannon'), sanitize('minas morgul'))):
        state.fellowship.progress = 0
        return

    # USER RULING 2026-07-20 (P2-04): reveal movement is UP TO the
    # Progress counter — place at exactly Progress when legal, and
    # when every region at that distance is an unconquered FP city/
    # stronghold (forbidden), fall back to SHORTER distances instead
    # of placing inside a settlement (the old fallback revealed the
    # Fellowship INSIDE Grey Havens from Forlindon).  Distance 0
    # (stay put) is handled by the not-valid branch below.
    shadow_controlled = state.enemy_controlled.get(gs.Side.SHADOW, set())

    def _frontier_at(d):
        fr = {cur}
        for _ in range(d):
            nxt = set()
            for r in fr:
                for adj in state.board.adjacent(r):
                    nxt.add(sanitize(adj))
            fr = nxt
        return fr

    def _legal_places(fr):
        out = []
        _self_play = _cmb_gate.FINAL_HITS_HOOK is None
        for region in fr:
            if region == cur:
                continue
            # USER RULING 2026-07-20b (w5_g2 T9 Gorgoroth): the pawn
            # never region-steps into Mordor's interior — that space
            # is reached only via the Mordor track after a turn-start
            # gateway declaration (1.155). Armies are NOT affected
            # (this filter is reveal-placement only). Self-play only.
            if _self_play and region in _MORDOR_INTERIOR:
                continue
            meta = board_mod.regions.get(region, {})
            is_fp_settlement = ((meta.get('is_city')
                                 or meta.get('is_stronghold'))
                                and is_fp_region(region)
                                and region not in shadow_controlled)
            if not is_fp_settlement:
                out.append(region)
        return out

    valid = []
    for _d in range(progress, 0, -1):
        valid = _legal_places(_frontier_at(_d))
        if valid:
            break

    if not valid:
        # Can't move anywhere — stay put but still reset progress.
        # (Mordor entry is NOT immediate — it happens at the start of
        # the next turn's Fellowship phase; USER 2026-07-09.)
        state.fellowship.progress = 0
        return

    # Pick the one closest to Mordor
    mordor_gates = {sanitize("morannon"), sanitize("minas morgul")}

    def _bfs_dist(start, targets):
        # A GATEWAY ITSELF is distance 0.  Without this check the
        # gate was scored by its distance to the OTHER gate (2 via
        # gorgoroth), so an adjacent region (south ithilien = 1,
        # dagorlad = 1) always outranked landing ON the gate — the
        # reveal placement bounced the fellowship around Minas
        # Morgul/Morannon forever (baseline_w4_g0 T8, USER report
        # 2026-07-18).
        if start in targets:
            return 0
        visited = {start}
        queue = [start]
        dist = 0
        while queue:
            dist += 1
            next_q = []
            for r in queue:
                for adj in state.board.adjacent(r):
                    a = sanitize(adj)
                    if a in targets:
                        return dist
                    if a not in visited:
                        visited.add(a)
                        next_q.append(a)
            queue = next_q
        return 999

    # sorted(): 'valid' is built from a SET — an unsorted min() made
    # equal-distance tie-breaks hash-order dependent (the survey-flake
    # class of bug)
    best = min(sorted(valid), key=lambda r: _bfs_dist(r, mordor_gates))

    # Walk a shortest path to the placement and draw the extra tile for
    # every Shadow stronghold crossed or entered (Moria class), plus
    # the leaving draw if the reveal started in one.
    _par = {cur: None}
    _q = [cur]
    while _q and best not in _par:
        _nq = []
        for _r in _q:
            for _adj in state.board.adjacent(_r):
                _a = sanitize(_adj)
                if _a not in _par:
                    _par[_a] = _r
                    _nq.append(_a)
        _q = _nq
    _path = []
    _n = best
    while _n is not None and _n != cur:
        _path.append(_n)
        _n = _par.get(_n)
    _path.reverse()
    _left_done = False
    for _stepr in _path:
        if _started_in_sh and not _left_done and _stepr != cur:
            _draw_shadow_stronghold_tile(state, cur, "leaving")
            _left_done = True
        if _is_sh_stronghold(_stepr):
            _draw_shadow_stronghold_tile(state, _stepr,
                                         "passing through")

    state.fellowship.region = state.board.region(best)
    state.fellowship.progress = 0
    # 1.132: the placement is invisible in traces without this row
    # (user QA: 'revealed but the GUI shows nothing happened')
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'fellowship_reveal_place',
                        {'from': cur, 'to': best, 'steps': progress})
    except Exception:
        pass
    # Mordor entry is NOT immediate on reveal — it happens at the start
    # of the next turn's Fellowship phase (USER 2026-07-09).


def _restore_hunt_pool_for_mordor(state: gs.GameState):
    """
    Restore hunt pool for Mordor entry:
    1. Return all eye tiles to pool (except Challenge of the King removed)
    2. Return other tiles from discard
    3. Add special tiles for cards that have been played
    4. Shuffle the pool
    """
    from game_state import SPECIAL_HUNT_TILES, HuntTile

    # USER SPEC (2026-07-07): of the USED tiles, only Eye tiles and the
    # blue/red special tiles return to the pool on Mordor entry;
    # standard numbered tiles stay used.  hunt_removed is permanent
    # (e.g. Challenge of the King) and never returns.
    for tile in state.hunt_discard[:]:
        if tile.is_eye or getattr(tile, 'is_special', False):
            state.hunt_pool.append(tile)
            state.hunt_discard.remove(tile)

    # (full-pass review 2026-07-27: the 'pending_hunt_tiles old system'
    # loop deleted — the field is never written in production; special
    # tiles enter via event_effects.stage_special_hunt_tile into
    # hunt_pool/hunt_discard and the eye/special restore above.)
    random.shuffle(state.hunt_pool)


def _check_mordor_entry(state: gs.GameState) -> bool:
    """
    Check if Fellowship enters Mordor from Morannon or Minas Morgul.
    Returns True if Mordor was entered this call.
    """
    if state.fellowship.mordor_entered:
        return False
    if fellowship_region(state) in {"minas morgul", "morannon"}:
        state.fellowship.mordor_entered = True
        state.fellowship.mordor_step = 0
        # P0 2026-08-15 (found by the live-play census): entry must
        # ZERO the Fellowship progress — the corpus entry choreography
        # walks the progress marker home, and the client/parser stop
        # reconciling progress once mordor_entered, so a non-zero
        # progress here left the ENGINE permanently ahead of the board
        # (and put a phantom progress into every in-Mordor training
        # state, since self-play/gui/engine_play share this function).
        state.fellowship.progress = 0
        # NOTE (open rules question, data/rules_amendments.md): whether
        # entry re-hides a revealed Fellowship is interpreted BOTH ways
        # in the logs — four certified games spend a die hiding after a
        # reveal-entry (entry does NOT hide), 2026-01-05 kapcerz T8
        # moves immediately without hiding (entry hides).  The engine
        # keeps the certified-majority reading; the kapcerz deviation is
        # force-consumed in the harness ('mordor_entry_hide_forced').
        _restore_hunt_pool_for_mordor(state)
        return True
    return False


def move_fellowship(state: gs.GameState,
                    skip_hunt: bool = False,
                    skip_auto_sacrifice: bool = True) -> dict:
    # DEFAULT INVERTED (auto-resolver purge 2026-08-16, USER: "THESE
    # ARE ALL LEARNED DECISIONS"; the Merry incident is specimen zero):
    # the DANGEROUS direction — letting _auto_sacrifice_companion
    # consume the hunt-damage decision inside this call — must now be
    # asked for EXPLICITLY (skip_auto_sacrifice=False).  A bare call
    # leaves damage in _pending_hunt_damage for the decision framework
    # (settle -> pending_hunt -> owning side).
    """
    Advance Fellowship pawn one region and optionally perform Hunt resolution.

    Args:
        skip_hunt: If True, skip automatic hunt resolution (for GUI to handle
                   interactively using get_hunt_params(), roll_hunt_dice(), etc.)

    Returns:
        dict with results: {"success": bool, "message": str, "victory": str or None,
                           "hunt_params": dict if skip_hunt}
    """
    result = {"success": True, "message": "", "victory": None}

    # Teacher only: logs place the Mordor marker mid-turn as housekeeping.
    # SELF-PLAY / live drivers enter ONLY at a turn start, via the
    # entry check that runs AFTER the Fellowship-phase declaration
    # opportunity (USER FINAL 2026-08-16) — never through this call.
    import combat as _cmb9
    if _cmb9.FINAL_HITS_HOOK is not None:
        _check_mordor_entry(state)
    # The FP player moved the Fellowship this turn: the Mordor inactivity
    # penalty (next_turn) applies only when the Fellowship neither moved
    # nor hid during the turn — a move BEFORE entering Mordor the same
    # turn counts (game 1 T7: move + reveal at Morannon, no further
    # fellowship action, and the humans applied no penalty).
    state.fellowship_acted_in_mordor = True

    #  Outside Mordor 
    if not state.fellowship.mordor_entered:
        state.fellowship.progress += 1
        result["message"] = f"Fellowship progress: {state.fellowship.progress}"

        if skip_hunt:
            # Return hunt params for GUI to handle interactively
            result["hunt_params"] = get_hunt_params(state)
        else:
            # Perform hunt resolution internally
            # Hunt Level = Shadow Action dice in the Hunt Box (rulebook p.40)
            hunt_level = sum(1 for d in state.hunt_box
                             if d.side is gs.Side.SHADOW)
            num_dice = min(5, hunt_level)
            fp_dice_in_box = sum(1 for d in state.hunt_box
                                 if d.side is gs.Side.FREE_PEOPLES)
            # Success on 6+; FP dice add +1 per die via fp_modifier
            base_thresh = 6

            successes = _hunt_roll(state, num_dice, base_thresh,
                                   fp_modifier=fp_dice_in_box) if num_dice else 0
            # "Balrog of Moria" persistent card: an extra hunt tile is
            # drawn whenever the Fellowship moves while located in Moria
            from persistent_effects import should_draw_extra_hunt_tile
            _fsp_reg = fellowship_region(state)
            _balrog, _ = should_draw_extra_hunt_tile(state, _fsp_reg, _fsp_reg)
            # fix 98 (game 11 T4): teacher-only evidence gate — the
            # humans fire the Balrog trigger at their own chosen event
            # (Rizino1/Pallando applied it at the T5 DECLARE, chat
            # 'might not trigger...seams like yes', not at the T4
            # move); the one-shot draw must follow the LOGGED tile,
            # not the engine's first eligible event.  Hook is None in
            # self-play (rules-default behavior unchanged).
            if _balrog and BALROG_TILE_GATE is not None \
                    and not BALROG_TILE_GATE(state):
                _balrog = False
            # GAP-1 (USER 2026-08-20 "fix balrog").  The card reads
            # "YOU MAY discard 'Balrog of Moria' to draw an additional
            # Hunt tile if the Fellowship moves into, out of, or through
            # Moria WHILE BEING DECLARED OR REVEALED" (card_list.csv:68).
            # This site honoured NEITHER half: it drew the tile and
            # force-discarded the one-shot card with no Shadow decision,
            # and it keyed on should_draw_extra_hunt_tile(cur, cur) --
            # the Fellowship's LAST KNOWN region -- so a HIDDEN move
            # while the marker sat in Moria fired it too.  The Shadow
            # lost a card it never chose to spend.  Its two sibling
            # triggers (declaration ai.py:10049, stationary turn start
            # ai.py:16562) already model this correctly as a learned
            # 'balrog_declare_draw' pending; this one now joins them.
            #
            # TEACHER PATH UNTOUCHED: with combat.FINAL_HITS_HOOK set,
            # the closed-loop teacher replays the human's LOGGED choice
            # through BALROG_TILE_GATE above and must never stall on a
            # fresh interactive pending -- exactly the gate the
            # stationary trigger documents as the mandatory pattern.
            import combat as _cmb_bal9
            _balrog_learned = (_balrog
                               and _cmb_bal9.FINAL_HITS_HOOK is None)
            if _balrog_learned:
                _balrog = False        # nothing fires inline any more
            if successes:
                _resolve_hunt_tile(state, successes, mordor=False)
            if _balrog_learned:
                # "declared or revealed" is tested AFTER the primary
                # tile resolves, because resolving it is what reveals:
                # a Fellowship must be hidden to move, so the only way
                # a MOVE happens "while being revealed" is that this
                # move's own hunt revealed it.
                if not state.fellowship.hidden \
                        and getattr(state, 'pending_card_option',
                                    None) is None:
                    _pcb9 = getattr(state, 'persistent_cards', None)
                    _cb9 = (_pcb9.get_by_type(
                        gs.PersistentCardType.BALROG_OF_MORIA)
                        if _pcb9 else None)
                    if _cb9 is not None:
                        # the SAME pending shape + handler branch the
                        # other two triggers use; the draw, the
                        # eye-without-effect rule, the Reveal-icon
                        # exception and the discard all live in
                        # event_effects._effect_balrog_of_moria.
                        state.pending_card_option = {
                            'card_id': getattr(_cb9, 'card_id',
                                               '__balrog__'),
                            'card_name': 'Balrog of Moria',
                            'side': 'sh', 'decider': 'sh',
                            'direct': True,
                            'options': [{'balrog_declare_draw': True},
                                        {'balrog_declare_draw': False}]}
            if _balrog:
                # W11 GAP1 fix (2026-07-20): refill an emptied pool
                # before the Balrog extra draw — "refill at EVERY draw
                # site" (rule); the primary draw above may have emptied
                # it, and the old `and state.hunt_pool` silently skipped
                # the guaranteed extra tile.
                _recycle_hunt_pool(state)
            if _balrog and state.hunt_pool:
                # ONE draw through the funnel; the eye-without-effect
                # rule applies to the tile actually drawn (a pool[-1]
                # peek diverged from the hunt_dist / live-bridge pops).
                _resolve_hunt_tile(state, max(successes, 1),
                                   mordor=False, eye_no_effect=True)
                # one-shot: the Balrog card is discarded when used
                _pcb = getattr(state, 'persistent_cards', None)
                _cb = (_pcb.get_by_type(
                    gs.PersistentCardType.BALROG_OF_MORIA)
                    if _pcb else None)
                if _cb is not None:
                    # CARD CONSERVATION (2026-08-15): discarded face-up
                    from persistent_effects import (discard_table_card
                                                    as _dtc9)
                    _dtc9(state, _pcb.remove_card(_cb.card_id) or _cb)
        # W3 BUG-2 fix (2026-07-20): this block sat mis-indented
        # INSIDE the Balrog branch, so the legacy auto-sacrifice
        # (skip_auto_sacrifice=False, GUI/non-AI callers) never ran
        # for ordinary hunt damage. De-indented to the move level.
        # W3 BUG-3 fix (2026-07-20, W11 hunt wave): the Balrog extra
        # draw used to OVERWRITE _pending_hunt_damage in _apply_hunt_
        # tile, dropping the first tile's damage; _apply_hunt_tile now
        # ACCUMULATES, so `pending` here correctly holds BOTH tiles.
        pending = getattr(state, '_pending_hunt_damage', 0)
        if pending > 0 and not skip_auto_sacrifice:
            # Legacy auto-sacrifice path (for GUI/non-AI callers)
            _auto_sacrifice_companion(state, pending)
            state._pending_hunt_damage = 0
        # If skip_auto_sacrifice: damage stays in _pending_hunt_damage
        # for caller (AI) to handle via pending_hunt state

    # Inside Mordor
    else:
        if state.fellowship.mordor_step >= 5:
            result["success"] = False
            result["message"] = "Fellowship already on final step!"
            return result

        # the stopped-this-turn rule gates the ACTION MENU
        # (ai._get_fellowship_actions), not execution: an explicitly
        # requested move proceeds so the teacher can replay logged
        # walks that moved again after a Stop tile (GenericPie T7)
        state.fellowship.mordor_step += 1
        result["message"] = f"Mordor step: {state.fellowship.mordor_step}/5"

        # Draw hunt tile and resolve (Mordor always uses internal resolution)
        tile_result = _resolve_mordor_hunt_tile(
            state, skip_auto_sacrifice=skip_auto_sacrifice)

        import os as _os_flag
        if (tile_result.get("stops_movement")
                and _os_flag.environ.get('MORDOR_STOP_KEEPS_STEP')):
            # ESCAPE HATCH (bisection only): 1.30's keep-step behavior.
            result["message"] = (f"Movement stopped for this turn "
                                 f"(step {state.fellowship.mordor_step} kept)")
        elif tile_result.get("stops_movement"):
            # USER RULE 2026-07-08 (supersedes 1.30): a Stop-icon tile
            # means the Fellowship does NOT gain the step — on step 2,
            # drawing a stop tile leaves them on step 2 — and movement
            # ends for the turn.  The TEACHER still follows each log's
            # own recorded walk (ground-truth syncs apply logged steps
            # in either direction), so certified replays are unaffected
            # by this default.
            state.fellowship.mordor_step -= 1
            result["message"] = (f"Stop tile: step not gained "
                                 f"(stays {state.fellowship.mordor_step}), "
                                 f"movement stopped this turn")

        # Check for corruption victory
        if state.fellowship.corruption >= 12:
            result["victory"] = "shadow"
            result["message"] = "Ring-bearers corrupted! Shadow wins!"

        # Check for FP victory (reached step 5) — only when the final
        # step's hunt damage has RESOLVED (see check_ring_victory)
        elif (state.fellowship.mordor_step >= 5
                and state.fellowship.corruption < 12
                and not getattr(state, '_pending_hunt_damage', 0)
                and state.pending_hunt is None
                and getattr(state, '_mithril_stash', None) is None
                and getattr(state, '_wizardstaff_stash', None) is None):
            # 1.132 (user QA): a Mithril-interposed tile is UNRESOLVED
            # hunt damage - the stash must apply before the win check
            # (1.232: a Staff-interposed DRAW is equally unresolved)
            result["victory"] = "free_peoples"
            result["message"] = "Ring destroyed! Free Peoples win!"

    return result


def _resolve_mordor_hunt_tile(state: gs.GameState,
                              skip_auto_sacrifice: bool = True,
                              _skip_wizard: bool = False) -> dict:
    # DEFAULT INVERTED (auto-resolver purge 2026-08-16): same contract
    # as move_fellowship — auto-resolution of the damage decision must
    # be requested explicitly; a bare call leaves the decision pending.
    """
    Resolve hunt tile in Mordor.

    Rules:
    - Always draw a tile (no dice roll needed)
    - Eye damage = total dice in hunt box (including FP character dice)
    - Stop tiles (On On They Went, Shelob's Lair, Ring is Mine, Give it to Us)
      cancel the fellowship's movement
    """
    result = {"damage": 0, "reveal": False, "stops_movement": False}

    # 1.232 (USER RULE): the Wizard's Staff may cancel a Mordor-track
    # tile draw too - same interpose, BEFORE recycle/draw; the caller
    # receives the neutral result until the FP decides
    if not _skip_wizard and _offer_wizard_staff(
            state, {'mordor_track': True,
                    'skip_auto_sacrifice': skip_auto_sacrifice}):
        return result

    if not state.hunt_pool:
        _recycle_hunt_pool(state)   # 1.136 (USER RULE)
    if not state.hunt_pool:
        return result

    # pop() like _resolve_hunt_tile so the teacher's _arrange_pool_last
    # hook forces the LOGGED tile (2026-07-11: this draw bypassed every
    # forcing hook - GenericPie T9 drew a phantom damage-4 tile where
    # the log has '1r', the root of the Isildur false-green chain)
    tile = pop_hunt_tile(state)

    # iter 38: hunt tile draws visible in SELF-PLAY traces
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'hunt_tile_draw', {
            'tile_name': tile.special_name if tile.special_name
            else repr(tile),
            'ctx': 'mordor hunt'})
    except Exception:
        pass

    # 'Mithril Coat and Sting' interpose (QA 2026-07-11 Mithril CASE
    # 5): the FP redraw decision precedes the tile's application -
    # including Shelob's d6 damage roll.  The caller receives a
    # neutral result; the mithril resolution applies the tile and
    # replicates the stop-tile step retraction.
    import combat as _cmbm9
    _pcm9 = (getattr(state, 'persistent_cards', None)
             if _cmbm9.FINAL_HITS_HOOK is None else None)
    _mc9 = None
    if _pcm9 is not None:
        try:
            _mc9 = _pcm9.get_by_type(
                gs.PersistentCardType.MITHRIL_COAT)
        except Exception:
            _mc9 = None
    if _mc9 is not None and getattr(state, 'pending_card_option',
                                    None) is None:
        state._mithril_stash = {
            'tile': tile, 'successes': 0, 'mordor': True,
            'mordor_track': True,
            'skip_auto_sacrifice': skip_auto_sacrifice}
        state.pending_card_option = {
            'card_id': getattr(_mc9, 'card_id', '__mithril__'),
            'card_name': 'Mithril Coat and Sting',
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'options': [{'mithril_redraw': True},
                        {'mithril_redraw': False}]}
        return result  # neutral: no damage/stop until the decision

    return _apply_mordor_hunt_tile(state, tile, result,
                                   skip_auto_sacrifice)


def _apply_mordor_hunt_tile(state: gs.GameState, tile, result: dict,
                            skip_auto_sacrifice: bool) -> dict:
    """Apply a drawn Mordor-track tile (split from
    _resolve_mordor_hunt_tile for the Mithril interpose)."""
    # Calculate damage
    if tile.is_eye:
        # Eye damage = total dice in hunt box
        total_dice = len(state.hunt_box)
        damage = total_dice
        result["damage"] = damage
        result["reveal"] = True  # Eyes reveal
    elif getattr(tile, 'is_dice_roll', False):
        # Shelob's Lair: roll 1d6 for damage — VISIBLE in self-play
        # traces (USER 2026-07-08: 'we roll a dice for how much
        # corruption, that needs to be shown')
        damage = roll_d6_batch(1, 'shelob d6 damage')[0]
        try:
            from game_trace import emit_annotation
            emit_annotation(state, 'dice_roll',
                            {'faces': [damage],
                             'ctx': f"{getattr(tile, 'special_name', 'Shelob')}"
                                    f" d6 damage", 'side': 'SH'})
        except Exception:
            pass
        result["damage"] = damage
        result["reveal"] = tile.reveal
    else:
        damage = tile.damage
        result["damage"] = damage
        result["reveal"] = tile.reveal

    # Check for stop tile
    if getattr(tile, 'stops_movement', False):
        result["stops_movement"] = True

    # Apply damage (handle negative damage from FP special tiles)
    # Auto-sacrifice companion in Mordor too — unless the caller (AI
    # executor) wants to resolve the casualty/Gollum decision itself
    # (mirrors the outside-Mordor skip_auto_sacrifice path; previously
    # Mordor hunts NEVER offered the sacrifice or Gollum-reveal choice)
    if damage > 0:
        if skip_auto_sacrifice:
            # ACCUMULATE, never assign (W3 BUG-3 invariant): damage
            # already pending across an interpose resume must never be
            # silently dropped.
            state._pending_hunt_damage = getattr(
                state, '_pending_hunt_damage', 0) + damage
        else:
            _auto_sacrifice_companion(state, damage)
    elif damage < 0:
        # Negative damage = healing
        state.fellowship.corruption = max(0, state.fellowship.corruption + damage)

    # Mordor Track reveal rule (USER 2026-07-11 correction): tiles DO
    # reveal on the track - Gollum's guide ability blocks only ORDINARY
    # damage-tile reveals; Eyes and special tiles reveal through him.
    # (GenericPie T9 stayed hidden on '1r'/'2r' because GOLLUM was the
    # guide from T7 ev269 - not because track tiles never reveal; Mol
    # T8's Eye revealed.)
    if result["reveal"] and state.fellowship.hidden:
        guide = state.fellowship.guide
        if (guide is not None
                and getattr(guide, 'guide_ability', None)
                == 'no_reveal_on_damage'
                and not tile.is_eye
                and not getattr(tile, 'is_special', False)):
            pass  # Gollum keeps the Fellowship hidden
        else:
            state.fellowship.hidden = False

    # Move tile to the used pile (drawn tiles are 'used', not removed;
    # eyes + specials return to the pool on Mordor entry — user spec)
    state.hunt_discard.append(tile)

    return result


def _draw_shadow_stronghold_tile(state: gs.GameState, region: str, reason: str):
    """
    Draw an extra hunt tile for passing through/being revealed in a shadow stronghold.
    If the tile is an Eye, nothing happens (no damage).
    """
    # ONE extra draw per stronghold region per reveal EVENT (USER
    # 2026-07-09).  1.231: the guard runs BEFORE the 1.136 recycle -
    # a suppressed duplicate draw must not fold the used pile back
    # into the pool (it is not a draw site).
    _seen = getattr(state, '_sh_tile_drawn_regions', None)
    if _seen is not None:
        _rk = sanitize(str(region))
        if _rk in _seen:
            return
        _seen.add(_rk)

    if not state.hunt_pool:
        _recycle_hunt_pool(state)   # 1.136 (USER RULE)
    if not state.hunt_pool:
        return

    tile = pick_hunt_tile(state)
    state.hunt_pool.remove(tile)

    if tile.is_eye:
        # Eye tiles do nothing when drawn for shadow stronghold
        state.hunt_discard.append(tile)  # used pile (returns on entry)
    else:
        # Route the damage through the SAME learned hunt-damage
        # decision as a normal hunt (sacrifice guide / random / take
        # corruption / chained guide abilities) — the old
        # _auto_sacrifice_companion silently killed a companion with
        # no decision (USER 2026-07-09).  NEGATIVE damage (FP special
        # tiles) is a HEAL and applies immediately — the pending
        # accumulator must stay >= 0 (see _apply_hunt_tile).
        if tile.damage < 0:
            state.fellowship.corruption = max(
                0, state.fellowship.corruption + tile.damage)
        else:
            state._pending_hunt_damage = getattr(
                state, '_pending_hunt_damage', 0) + tile.damage
        state.hunt_discard.append(tile)
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'stronghold_tile_draw', {
            'tile_name': tile.special_name if tile.special_name
            else repr(tile), 'region': str(region), 'ctx': reason,
            'damage': 0 if tile.is_eye else tile.damage})
    except Exception:
        pass


def _muster_one(state: gs.GameState,
                side: gs.Side,
                region: str,
                typ: str,
                nation: Optional[gs.Nation] = None):
    """
    Place a single unit/leader in *region*.
    • *typ* is 'elite', 'regular', or 'leader'
    • *nation* is the recruiting nation (defaults to Sauron/Gondor for backward compat)
    • Region must be a city / stronghold / town.
    • Army cannot exceed 10 units (stacking limit).
    """
    rkey = sanitize(region)
    meta = board.regions.get(rkey)
    if not meta:
        raise RuntimeError(f"{region} not recognised")
    if not (meta["is_city"] or meta["is_stronghold"] or meta.get("is_town")):
        raise RuntimeError(f"{region} is not a settlement")

    # Default nation if not specified
    if nation is None:
        nation = gs.Nation.SAURON if side is gs.Side.SHADOW else gs.Nation.GONDOR

    # review P1-1 (2026-07-27): route through get_or_create_army — the
    # old raw setdefault(orig_name) keyed by the BOARD's original name,
    # which can case-mismatch the setup/parser key ("Helm's Deep" vs
    # board "Helm's deep") and forge a DUPLICATE army entry: musters
    # split the garrison across two keys, battles fight half the army.
    army = get_or_create_army(state, rkey)

    # FIGURE-POOL guard (2026-07-20, enforce-by-construction): a unit
    # may only be mustered if the nation's physical reinforcement pool
    # has one. Every legitimate caller pre-checks _can_recruit (so this
    # is behavior-neutral for them), but an UNGUARDED caller (a card
    # handler, a teacher forcing path) could otherwise mint a figure
    # beyond the pool (W9 F2 / kapcerz Dwarves 6/5). Return without
    # placing when exhausted — the excess stays in reinforcements,
    # matching event_effects.recruit_unit's semantics.
    if typ in ("elite", "regular"):
        try:
            from ai import _can_recruit
        except ImportError:
            # circular-import window only; a raising _can_recruit must
            # stay loud (silently disabling the pool cap re-opens the
            # W9 F2 over-pool minting)
            _can_recruit = None
        if _can_recruit is not None and not _can_recruit(state, nation,
                                                         typ):
            return None

    # Check stacking limit for units (leaders don't count)
    # Besieged strongholds have a 5-unit limit; otherwise 10
    besieged = getattr(state, 'besieged_strongholds', set())
    unit_cap = MAX_SIEGE_UNITS if rkey in besieged else MAX_ARMY_UNITS
    if typ in ("elite", "regular") and len(army.units) >= unit_cap:
        raise RuntimeError(f"Army stacking limit ({unit_cap} units) reached in {region}")

    if typ == "elite":
        army.units.append(gs.Unit(nation, gs.UnitKind.ELITE))
    elif typ == "regular":
        army.units.append(gs.Unit(nation, gs.UnitKind.REGULAR))
    elif typ == "leader":
        if nation is gs.Nation.SAURON:
            # Sauron's "leaders" are Nazgul (winged — they get nazgul_move)
            army.leaders.append(gs.Nazgul(nation))
        else:
            army.leaders.append(gs.Leader(nation))
    else:
        raise RuntimeError("typ must be elite/regular/leader")

#  Event-card draw helper (with hand-size check)
def draw_event_card(state: gs.GameState,
                    side: gs.Side,
                    deck: str):
    """
    Draw one card from *deck* for *side*.

    • *deck* is "character" or "strategy".
    • Draws from actual card decks (initialized with real card IDs).
    • If hand size > 6, sets `needs_discard = True` on that side's PlayerInfo.
    """
    hands = state.hands[side]

    # Use actual card decks with real card IDs
    if side is gs.Side.FREE_PEOPLES:
        if deck == "character":
            card_deck = getattr(state, 'fp_char_deck', [])
            if not card_deck:
                raise RuntimeError("FP character deck empty")
            card_id = card_deck.pop()
        else:  # strategy
            card_deck = getattr(state, 'fp_strat_deck', [])
            if not card_deck:
                raise RuntimeError("FP strategy deck empty")
            card_id = card_deck.pop()
    else:  # Shadow
        if deck == "character":
            card_deck = getattr(state, 'sh_char_deck', [])
            if not card_deck:
                raise RuntimeError("SH character deck empty")
            card_id = card_deck.pop()
        else:  # strategy
            card_deck = getattr(state, 'sh_strat_deck', [])
            if not card_deck:
                raise RuntimeError("SH strategy deck empty")
            card_id = card_deck.pop()

    # ---- add to hand & check limit ---------------------------------------
    hands.hand.append(card_id)
    hands.needs_discard = len(hands.hand) > 6
    # every draw visible in the trace/GUI (USER 2026-07-09: 'the cards
    # drawn and the cards discarded visible in the GUI log')
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'card_drawn', {
            'deck': str(deck), 'card_id': str(card_id),
            'side': 'FP' if side is gs.Side.FREE_PEOPLES else 'SH'})
    except Exception:
        pass


def grant_draw(state: gs.GameState, side: gs.Side, deck: str):
    """An ENGINE-GRANTED draw (Witch-king Sorcerer, bid-compensation
    token, card-effect draws — the sources a driver cannot derive from
    the play itself).  Executes the draw; when execution is DEFERRED by
    an external randomness owner (the live driver's draw funnel raising
    its client-call out of draw_event_card), the grant is recorded in
    ``state._owed_draws`` so the driver can recognize the side's
    physical draw as owed.  This is A13's missing writer: the debt
    channel had 2 readers and none — a rule the engine could not
    express.  Keyed ``(side.name, deck)`` to match the deployed reader.

    Semantics:
    • success        -> draw done; settle one unit of any debt this same
                        grant stamped on an earlier deferred attempt
                        (driver retry loops re-run the grant site).
    • RuntimeError   -> deck empty: the grant is unfulfillable — NO debt
                        is recorded and the error propagates unchanged.
    • anything else  -> deferred/aborted mid-draw: stamp the debt, then
                        re-raise.  A transactional caller (apply on a
                        clone, commit on success) gets exact debt; a
                        by-reference caller may retain a residual stamp,
                        which errs toward accepting one owed-shaped draw
                        — never toward refusing a legitimate one."""
    try:
        draw_event_card(state, side, deck)
    except RuntimeError:
        raise
    except BaseException:
        d = getattr(state, '_owed_draws', None)
        if d is None:
            d = {}
            state._owed_draws = d
        key = (side.name, deck)
        d[key] = d.get(key, 0) + 1
        raise
    else:
        settle_owed_draw(state, side, deck)


def settle_owed_draw(state: gs.GameState, side: gs.Side,
                     deck: str) -> bool:
    """Consume one unit of declared draw debt (the driver calls this
    when the owed physical draw arrives; grant_draw calls it on a
    successful retry).  True iff debt existed and was consumed."""
    d = getattr(state, '_owed_draws', None) or {}
    key = (side.name, deck)
    if d.get(key, 0) > 0:
        d[key] -= 1
        return True
    return False


#  SPECIAL CHARACTERS: Minions, Gandalf the White, Aragorn

def can_muster_witch_king(state: gs.GameState) -> Tuple[bool, str, List[str]]:
    """
    Check if Witch King can be mustered.
    Returns (can_muster, reason, valid_regions)

    Requirement (USER RULE 2026-07-08 — the stronghold condition does
    not exist): Sauron at war AND any FP nation at war -> muster in a
    region with a Sauron unit.
    """
    if state.witch_king and state.witch_king.in_play:
        return False, "Witch King already in play", []
    # USER RULING 2026-07-17: minions cannot be re-mustered once killed
    if state.witch_king and state.witch_king.was_ever_in_play:
        return False, "Witch King was eliminated", []

    sauron_at_war = state.political.is_at_war(gs.Nation.SAURON)

    # Check if any FP nation is at war
    fp_nations = [gs.Nation.ELVES, gs.Nation.DWARVES, gs.Nation.GONDOR, gs.Nation.THE_NORTH, gs.Nation.ROHAN]
    any_fp_at_war = any(state.political.is_at_war(n) for n in fp_nations)

    valid_regions = []

    # Condition 1: Sauron at war + any FP at war -> muster in region with Sauron unit
    if sauron_at_war and any_fp_at_war:
        for region_key, army in state.armies.items():
            for unit in army.units:
                if unit.nation == gs.Nation.SAURON:
                    valid_regions.append(region_key)
                    break

    # USER RULE 2026-07-08: there is NO stronghold condition for the
    # Witch-king — he musters ONLY with a Sauron army, and only when
    # Sauron is At War AND at least one FP nation is At War.  (The
    # removed 'all shadow at war OR fellowship in Mordor -> any shadow
    # stronghold' branch belonged to no card at all — the MOUTH's real
    # entry is ALL FP NATIONS at war OR fellowship in Mordor, which
    # can_muster_mouth_of_sauron implements correctly.)

    if not valid_regions:
        return False, "No valid muster location", []

    return True, "", valid_regions


def can_muster_saruman(state: gs.GameState) -> Tuple[bool, str, List[str]]:
    """
    Check if Saruman can be mustered.
    Returns (can_muster, reason, valid_regions)

    Requirements:
    - Isengard at war
    - Muster in Orthanc only
    """
    if state.saruman and state.saruman.in_play:
        return False, "Saruman already in play", []
    # USER RULING 2026-07-17: minions cannot be re-mustered once killed
    if state.saruman and state.saruman.was_ever_in_play:
        return False, "Saruman was eliminated", []

    if not state.political.is_at_war(gs.Nation.ISENGARD):
        return False, "Isengard not at war", []

    orthanc = sanitize("orthanc")
    # Check Orthanc not conquered by FP
    if orthanc in state.enemy_controlled.get(gs.Side.FREE_PEOPLES, set()):
        return False, "Orthanc is conquered", []

    return True, "", [orthanc]


def can_muster_mouth_of_sauron(state: gs.GameState) -> Tuple[bool, str, List[str]]:
    """
    Check if Mouth of Sauron can be mustered.
    Returns (can_muster, reason, valid_regions)

    Requirements (user ruling 2026-07-03, see data/rules_amendments.md):
    - ALL FIVE Free Peoples nations at war OR fellowship in Mordor
      (the old check wrongly required all SHADOW nations at war)
    - Muster in unconquered shadow stronghold
    """
    if state.mouth_of_sauron and state.mouth_of_sauron.in_play:
        return False, "Mouth of Sauron already in play", []
    # USER RULING 2026-07-17: minions cannot be re-mustered once killed
    if state.mouth_of_sauron and state.mouth_of_sauron.was_ever_in_play:
        return False, "Mouth of Sauron was eliminated", []

    all_fp_at_war = all(state.political.is_at_war(n) for n in (
        gs.Nation.GONDOR, gs.Nation.ROHAN, gs.Nation.ELVES,
        gs.Nation.DWARVES, gs.Nation.THE_NORTH))

    fellowship_in_mordor = state.fellowship.mordor_entered

    if not (all_fp_at_war or fellowship_in_mordor):
        return False, "Not all FP nations at war and fellowship not in Mordor", []

    valid_regions = []
    shadow_strongholds = _get_shadow_strongholds(state)
    for sh in shadow_strongholds:
        if sh not in state.enemy_controlled.get(gs.Side.FREE_PEOPLES, set()):
            valid_regions.append(sh)

    if not valid_regions:
        return False, "No unconquered shadow strongholds", []

    return True, "", valid_regions


def _get_shadow_strongholds(state: gs.GameState) -> List[str]:
    """Return list of shadow stronghold region keys."""
    from board import regions as board_regions, fid_to_name
    shadow_strongholds = []
    for key, info in board_regions.items():
        if info.get("is_stronghold"):
            parent_id = info.get("parent_id")
            if parent_id:
                nation_name = fid_to_name.get(parent_id, "")
                if nation_name in ["sauron", "isengard",
                                   "southrons and easterlings",
                                   "southrons", "easterlings"]:
                    shadow_strongholds.append(key)
    return shadow_strongholds


def muster_minion(state: gs.GameState, minion_name: str, region: str, die_idx: int):
    """
    Muster a shadow minion (Witch King, Saruman, or Mouth of Sauron).
    Consumes a muster die.
    """
    wanted = (gs.ActionDieFace.MUSTER, gs.ActionDieFace.MUSTER_ARMY)
    _pick_die(state, gs.Side.SHADOW, wanted, die_idx)

    region_san = sanitize(region)
    minion_name_lower = minion_name.lower()

    if minion_name_lower == "witch king":
        can, reason, valid = can_muster_witch_king(state)
        if not can:
            raise RuntimeError(f"Cannot muster Witch King: {reason}")
        if region_san not in [sanitize(v) for v in valid]:
            raise RuntimeError("Invalid muster location for Witch King")
        state.witch_king.in_play = True
        state.witch_king.was_ever_in_play = True
        # Add as leader to the region
        _add_minion_to_region(state, state.witch_king, region_san)
        # WK card: "Activate all Free Peoples Nations" on entry
        # (rulebook marker p.24; audit A8, 2026-07-26).  Activation only
        # (the active flag) - never a track advance.  Idempotent, so
        # teacher replays of logs whose humans flipped the markers
        # explicitly are unaffected.
        for _fpn in (gs.Nation.ELVES, gs.Nation.DWARVES, gs.Nation.GONDOR,
                     gs.Nation.THE_NORTH, gs.Nation.ROHAN):
            if not state.political.active.get(_fpn, False):
                try:
                    activate_nation(state, _fpn, method="witch_king_entry")
                except RuntimeError:
                    pass

    elif minion_name_lower == "saruman":
        can, reason, valid = can_muster_saruman(state)
        if not can:
            raise RuntimeError(f"Cannot muster Saruman: {reason}")
        if region_san not in [sanitize(v) for v in valid]:
            raise RuntimeError("Invalid muster location for Saruman")
        state.saruman.in_play = True
        state.saruman.was_ever_in_play = True
        _add_minion_to_region(state, state.saruman, region_san)

    elif minion_name_lower in ["mouth of sauron", "mouth"]:
        can, reason, valid = can_muster_mouth_of_sauron(state)
        if not can:
            raise RuntimeError(f"Cannot muster Mouth of Sauron: {reason}")
        if region_san not in [sanitize(v) for v in valid]:
            raise RuntimeError("Invalid muster location for Mouth of Sauron")
        state.mouth_of_sauron.in_play = True
        state.mouth_of_sauron.was_ever_in_play = True
        _add_minion_to_region(state, state.mouth_of_sauron, region_san)
    else:
        raise RuntimeError(f"Unknown minion: {minion_name}")


def _add_minion_to_region(state: gs.GameState, minion: gs.ShadowMinion, region_san: str):
    """Add a minion as a leader to a region's army."""
    # Create minion leader object
    minion_leader = Minion(nation=gs.Nation.SAURON, name=minion.name)

    # Get or create army at region (sanitized-key aware)
    get_or_create_army(state, region_san).leaders.append(minion_leader)


def can_transform_gandalf_white(state: gs.GameState) -> Tuple[bool, str, List[str]]:
    """
    Check if Gandalf the Grey can become Gandalf the White.
    Returns (can_transform, reason, valid_regions)

    Requirements:
    - At least one Shadow minion must be in play (Saruman, Witch-King, or Mouth of Sauron)
    - Gandalf the Grey is dead or left the fellowship
    - Use Will of the West die
    - If dead: bring to Fangorn or unconquered elven stronghold
    """
    if state.gandalf_the_white and state.gandalf_the_white.in_play:
        return False, "Gandalf the White already in play", []

    # Check if Gandalf the Grey is in fellowship (check this first for
    # clearer error messages — Grey must leave before White can appear)
    gandalf_in_fellowship = any(
        c.name == "Gandalf the Grey" for c in state.fellowship.companions
    )

    if gandalf_in_fellowship:
        return False, "Gandalf the Grey is still in the fellowship", []

    # Check that at least one Shadow minion is or WAS in play.
    # Per rulebook: "any valid Minion is or has been in play (even if
    # eliminated)" — so we check was_ever_in_play, not just in_play.
    minion_was_in_play = False
    if state.witch_king and state.witch_king.was_ever_in_play:
        minion_was_in_play = True
    if state.saruman and state.saruman.was_ever_in_play:
        minion_was_in_play = True
    if state.mouth_of_sauron and state.mouth_of_sauron.was_ever_in_play:
        minion_was_in_play = True

    if not minion_was_in_play:
        return False, "At least one Shadow minion must be or have been in play", []

    # Check if Gandalf is on the board (separated)
    gandalf_on_board = False
    gandalf_region = None
    for region_key, army in state.armies.items():
        for leader in army.leaders:
            if isinstance(leader, gs.Companion) and leader.name == "Gandalf the Grey":
                gandalf_on_board = True
                gandalf_region = region_key
                break

    # Valid regions: Fangorn or elven strongholds
    elven_strongholds = ["grey havens", "rivendell", "lorien", "woodland realm"]
    valid_regions = ["fangorn"] + elven_strongholds

    # Filter to unconquered
    valid_regions = [
        r for r in valid_regions
        if sanitize(r) not in state.enemy_controlled.get(gs.Side.SHADOW, set())
    ]

    if gandalf_on_board:
        # If on board, can transform where he is (no movement needed)
        return True, "", [gandalf_region] if gandalf_region else valid_regions

    # If dead, must place in valid region
    if not valid_regions:
        return False, "No valid regions for Gandalf the White", []

    return True, "", valid_regions


def transform_gandalf_white(state: gs.GameState, region: str, die_idx: int):
    """
    Transform Gandalf the Grey into Gandalf the White.
    Requires Will of the West die.
    """
    wanted = (gs.ActionDieFace.WILL_OF_WEST,)
    _pick_die(state, gs.Side.FREE_PEOPLES, wanted, die_idx)

    can, reason, valid = can_transform_gandalf_white(state)
    if not can:
        raise RuntimeError(f"Cannot transform Gandalf: {reason}")

    region_san = sanitize(region)
    if region_san not in [sanitize(v) for v in valid]:
        raise RuntimeError("Invalid region for Gandalf the White")

    # Remove Gandalf the Grey from anywhere
    for region_key, army in list(state.armies.items()):
        army.leaders = [
            l for l in army.leaders
            if not (isinstance(l, gs.Companion) and l.name == "Gandalf the Grey")
        ]

    # Mark Gandalf the White as in play
    state.gandalf_the_white.in_play = True

    # Create Gandalf the White as companion
    gandalf_white_companion = gs.Companion(
        name="Gandalf the White", level=3, leadership=1,
        captain_of_the_west=False,  # No captain of the west
        bonus_movement=1,  # Movement 4 when alone/with hobbit
        solo_hobbit_movement=True,
    )

    # Add to region (sanitized-key aware)
    get_or_create_army(state, region_san).leaders.append(gandalf_white_companion)

    # QA 2026-07-11 cluster A (amendment 1.126): Gandalf the Grey has left
    # play (upgraded to White via the Will-of-the-West die from a separated
    # or eliminated Grey).  Wizard's Staff (and any other requires_companion
    # FP table card keyed on Grey) must discard immediately.  The card-play
    # upgrade path (effect_gandalf_white_upgrade) already re-checks; this
    # completes the invariant for the board/die path.
    check_companion_persistent_discards(state)


def can_crown_aragorn(state: gs.GameState) -> Tuple[bool, str, List[str]]:
    """
    Check if Strider can be crowned as Aragorn.
    Returns (can_crown, reason, valid_regions)

    Requirements:
    - Strider is in Minas Tirith, Pelargir, or Dol Amroth
    - The region is unconquered by Shadow
    - Use Will of the West die
    """
    if state.aragorn_crowned and state.aragorn_crowned.crowned:
        return False, "Aragorn already crowned", []

    crown_regions = ["minas tirith", "pelargir", "dol amroth"]

    # Find Strider
    strider_region = None
    for region_key, army in state.armies.items():
        for leader in army.leaders:
            if isinstance(leader, gs.Companion) and leader.name == "Strider":
                strider_region = region_key
                break

    # Strider's card (rulebook marker p.4): "If Strider is NOT in the
    # Fellowship, he can be replaced by Aragorn."  An in-Fellowship
    # Strider can never be crowned - he must separate onto the map
    # first (audit A12, 2026-07-26; the old branch crowned through the
    # fellowship pawn's location).
    if any(c.name == "Strider" for c in state.fellowship.companions):
        return False, "Strider is still in the Fellowship", []

    if not strider_region:
        return False, "Strider not found", []

    strider_region_san = sanitize(strider_region)
    if strider_region_san not in [sanitize(r) for r in crown_regions]:
        return False, "Strider must be in Minas Tirith, Pelargir, or Dol Amroth", []

    if strider_region_san in state.enemy_controlled.get(gs.Side.SHADOW, set()):
        return False, f"{strider_region} is conquered", []

    return True, "", [strider_region]


def crown_aragorn(state: gs.GameState, die_idx: int):
    """
    Crown Strider as Aragorn.
    Requires Will of the West die.
    """
    wanted = (gs.ActionDieFace.WILL_OF_WEST,)
    _pick_die(state, gs.Side.FREE_PEOPLES, wanted, die_idx)

    can, reason, valid = can_crown_aragorn(state)
    if not can:
        raise RuntimeError(f"Cannot crown Aragorn: {reason}")

    # Find and upgrade Strider
    for region_key, army in state.armies.items():
        for i, leader in enumerate(army.leaders):
            if isinstance(leader, gs.Companion) and leader.name == "Strider":
                # Replace Strider with Aragorn
                aragorn_companion = gs.Companion(
                    name="Aragorn", level=3, leadership=2,
                    captain_of_the_west=True,
                    activates_nations=[],  # Activates any
                )
                army.leaders[i] = aragorn_companion
                state.aragorn_crowned.crowned = True
                # Aragorn enters play in a Gondor city/stronghold ->
                # activate Gondor (rulebook marker p.33; audit A13,
                # 2026-07-26).  Same trigger as any companion entering
                # play in its nation's settlement.
                if not state.political.active.get(gs.Nation.GONDOR, False):
                    try:
                        activate_nation(state, gs.Nation.GONDOR,
                                        method="aragorn_crowned",
                                        region=sanitize(region_key))
                    except RuntimeError:
                        pass
                return
    raise RuntimeError("Cannot crown Aragorn: Strider not on the map "
                       "(must separate from the Fellowship first)")


def calculate_bonus_dice(state: gs.GameState):
    """
    Calculate bonus dice from special characters.
    Called at the start of each turn.

    FP bonus dice: +1 for Gandalf the White, +1 for Aragorn crowned
    Shadow bonus dice: +1 for each minion in play
    """
    fp_bonus = 0
    sh_bonus = 0

    # "Both ... lose the additional die if the corresponding Character
    # is eliminated" (marker p.18; audit A14, 2026-07-26).  The Shadow
    # mirror clears minion in_play flags on elimination; the FP side
    # never cleared anything, so verify liveness directly: the
    # character must still be on the map (or, defensively, in the
    # fellowship) for the die to count.
    def _fp_alive(name):
        for _a14 in state.armies.values():
            for _l14 in _a14.leaders:
                if isinstance(_l14, gs.Companion) and _l14.name == name:
                    return True
        return any(c.name == name for c in state.fellowship.companions)

    # Free Peoples
    if state.gandalf_the_white and state.gandalf_the_white.in_play \
            and _fp_alive("Gandalf the White"):
        fp_bonus += 1
    if state.aragorn_crowned and state.aragorn_crowned.crowned \
            and _fp_alive("Aragorn"):
        fp_bonus += 1

    # Shadow
    if state.witch_king and state.witch_king.in_play:
        sh_bonus += 1
    if state.saruman and state.saruman.in_play:
        sh_bonus += 1
    if state.mouth_of_sauron and state.mouth_of_sauron.in_play:
        sh_bonus += 1

    state.fp_bonus_dice = fp_bonus
    state.sh_bonus_dice = sh_bonus

    # Adjust actual dice pools (add new dice if needed)
    _adjust_dice_pool(state, gs.Side.FREE_PEOPLES, 4 + fp_bonus)
    _adjust_dice_pool(state, gs.Side.SHADOW, 7 + sh_bonus)


def _adjust_dice_pool(state: gs.GameState, side: gs.Side, target_count: int):
    """Adjust dice pool to match target count."""
    current = len(state.dice[side])
    if current < target_count:
        # Add dice
        faces = gs.FP_FACES if side == gs.Side.FREE_PEOPLES else gs.SH_FACES
        for _ in range(target_count - current):
            state.dice[side].append(gs.ActionDie(faces, side))
    elif current > target_count:
        # Remove unused dice from end
        while len(state.dice[side]) > target_count:
            state.dice[side].pop()


def _handle_saruman_elimination_discards(state: gs.GameState):
    """
    Handle automatic card discards when Saruman is eliminated.

    Cards that require Saruman:
    - The Palantír of Orthanc
    - Wormtongue
    """
    from persistent_effects import check_saruman_eliminated_discards
    from game_state import PersistentCardState

    # Ensure persistent_cards exists
    if not hasattr(state, 'persistent_cards') or state.persistent_cards is None:
        return

    if not isinstance(state.persistent_cards, PersistentCardState):
        return

    from persistent_effects import discard_table_card
    for card in check_saruman_eliminated_discards(state):
        _rm = state.persistent_cards.remove_card(card.card_id)
        discard_table_card(state, _rm or card)


#  COMPANION ABILITIES

def get_companion_movement(companion: gs.Companion, state: gs.GameState,
                           army: Optional[gs.Army] = None) -> int:
    """
    Get movement distance for a companion.
    - Normal: level
    - Gandalf the White: 4 when alone or with 1 hobbit
    """
    if companion.name == "Gandalf the White" and companion.solo_hobbit_movement:
        # Check if alone or with only hobbits
        if army:
            companions_in_army = [
                l for l in army.leaders if isinstance(l, gs.Companion)
            ]
            hobbits = [c for c in companions_in_army
                       if c.name in ["Meriadoc", "Peregrin", "Frodo", "Sam"]]
            non_hobbits = [c for c in companions_in_army if c not in hobbits and c != companion]
            if len(non_hobbits) == 0 and len(hobbits) <= 1:
                return 4  # Solo or with 1 hobbit
        else:
            return 4  # Assume solo
    return companion.level


_VP_TABLES = None   # (restored: the P2-09 dead-code deletion range
                    # accidentally swallowed this memo global)


def _vp_settlement_tables():
    """(fp_settlements, sh_settlements) built once from static board data.

    fp_settlements: (sanitized_key, vp_value) for FP-owned settlements —
        these score for SHADOW when Shadow-controlled.
    sh_settlements: (sanitized_key, vp_value) for Shadow-owned settlements
        — these score for FREE PEOPLES when FP-controlled.

    Mirrors the original per-region branch exactly (region_is_fp wins over
    region_is_shadow; a region matching neither faction scores for no one).
    """
    global _VP_TABLES
    if _VP_TABLES is None:
        from board import regions as board_regions, sanitize
        fp_list = []
        sh_list = []
        for region_key, region_info in board_regions.items():
            is_city = region_info.get("is_city", False)
            is_stronghold = region_info.get("is_stronghold", False)
            if not is_city and not is_stronghold:
                continue
            region_is_shadow = is_shadow_region(region_key)
            region_is_fp = is_fp_region(region_key)
            vp_value = 2 if is_stronghold else 1
            skey = sanitize(region_key)
            if region_is_fp:
                fp_list.append((skey, vp_value))
            elif region_is_shadow:
                sh_list.append((skey, vp_value))
        _VP_TABLES = (tuple(fp_list), tuple(sh_list))
    return _VP_TABLES


def calculate_victory_points(state: gs.GameState) -> dict:
    """
    Calculate victory points for both sides based on settlements controlled.

    VP Rules:
    - Cities = 1 VP
    - Strongholds = 2 VP
    - Only settlements controlled by the ENEMY side count

    Returns dict with "free_peoples": int, "shadow": int
    """
    fp_settlements, sh_settlements = _vp_settlement_tables()

    # Hoisted out of the per-settlement loop (was 2 enum-keyed dict.get
    # per settlement per call). `x in ()` is False, matching the original
    # `in ...get(side, set())` when a side has captured nothing.
    fp_captured = state.enemy_controlled.get(gs.Side.FREE_PEOPLES, ())
    sh_captured = state.enemy_controlled.get(gs.Side.SHADOW, ())

    # Shadow gets VP for FP settlements they control
    sh_vp = 0
    for skey, vp_value in fp_settlements:
        if skey in sh_captured:
            sh_vp += vp_value

    # FP gets VP for Shadow settlements they control
    fp_vp = 0
    for skey, vp_value in sh_settlements:
        if skey in fp_captured:
            fp_vp += vp_value

    return {"free_peoples": fp_vp, "shadow": sh_vp}


def check_military_victory(state: gs.GameState) -> dict:
    """
    Check if either side has achieved military victory.

    Returns dict with:
    - "victory": "shadow", "free_peoples", or None
    - "free_peoples_vp": int
    - "shadow_vp": int
    - "message": str or None
    """
    vp = calculate_victory_points(state)

    result = {
        "victory": None,
        "free_peoples_vp": vp["free_peoples"],
        "shadow_vp": vp["shadow"],
        "message": None
    }

    if vp["shadow"] >= SHADOW_VP_VICTORY:
        result["victory"] = "shadow"
        result["message"] = f"Shadow military victory! ({vp['shadow']} VP)"

    elif vp["free_peoples"] >= FP_VP_VICTORY:
        result["victory"] = "free_peoples"
        result["message"] = f"Free Peoples military victory! ({vp['free_peoples']} VP)"

    return result


def check_ring_victory(state: gs.GameState) -> dict:
    """
    Check for Ring-based victory conditions.

    Returns dict with:
    - "victory": "shadow", "free_peoples", or None
    - "message": str or None
    """
    result = {"victory": None, "message": None}

    # Shadow wins if corruption reaches 12
    if state.fellowship.corruption >= 12:
        result["victory"] = "shadow"
        result["message"] = f"Ring-bearers corrupted! ({state.fellowship.corruption} corruption)"

    # FP wins if fellowship reaches step 5 in Mordor with < 12 corruption.
    # The FINAL step's hunt damage must resolve FIRST: with a pending
    # hunt open, corruption may still reach 12 (2023-11-03 ketchuptime
    # T9: the step-5 Eye tile's +5 took the humans to 12 — Shadow Ring,
    # not ring-destroyed; the premature check flipped the outcome).
    elif state.fellowship.mordor_entered and state.fellowship.mordor_step >= 5:
        if state.fellowship.corruption < 12                 and not getattr(state, '_pending_hunt_damage', 0)                 and state.pending_hunt is None                 and getattr(state, '_mithril_stash', None) is None                 and getattr(state, '_wizardstaff_stash', None) is None:
            result["victory"] = "free_peoples"
            result["message"] = "The Ring is destroyed! Free Peoples win!"

    return result


def should_game_end(vc: dict, end_of_turn: bool) -> bool:
    """Rulebook p.17 victory TIMING (iter 38, single source of truth for
    the self-play loops): ring victories (destruction at Mordor step 5 /
    corruption 12) end the game IMMEDIATELY, mid-action; MILITARY
    victories (SH 10 VP / FP 4 VP) are only checked in the end-of-turn
    Victory Check phase."""
    vt = vc.get('victory_type')
    if not vc.get('victory'):
        return False
    if vt in ('ring_destroyed', 'ring_corruption'):
        return True
    return bool(end_of_turn)


def check_all_victory_conditions(state: gs.GameState) -> dict:
    """
    Check all victory conditions.

    Priority:
    1. Ring victory (corruption or destruction)
    2. Military victory

    Returns dict with:
    - "victory": "shadow", "free_peoples", or None
    - "victory_type": "ring_corruption", "ring_destroyed", "military", or None
    - "message": str or None
    - "free_peoples_vp": int
    - "shadow_vp": int
    """
    # Check ring victory first (takes priority)
    ring = check_ring_victory(state)
    if ring["victory"]:
        vp = calculate_victory_points(state)
        return {
            "victory": ring["victory"],
            "victory_type": "ring_corruption" if ring["victory"] == "shadow" else "ring_destroyed",
            "message": ring["message"],
            "free_peoples_vp": vp["free_peoples"],
            "shadow_vp": vp["shadow"]
        }

    # Check military victory
    military = check_military_victory(state)
    if military["victory"]:
        return {
            "victory": military["victory"],
            "victory_type": "military",
            "message": military["message"],
            "free_peoples_vp": military["free_peoples_vp"],
            "shadow_vp": military["shadow_vp"]
        }

    # No victory yet
    return {
        "victory": None,
        "victory_type": None,
        "message": None,
        "free_peoples_vp": military["free_peoples_vp"],
        "shadow_vp": military["shadow_vp"]
    }



# CARD MANAGEMENT - GUI-Independent Functions

def draw_cards_for_side(state: gs.GameState, side: gs.Side) -> dict:
    """
    Draw 1 character + 1 strategy card for a side.
    All deck manipulation happens here.

    Returns dict with drawn card info.
    """
    drawn = []
    hand = state.hands[side].hand

    if side == gs.Side.FREE_PEOPLES:
        char_deck = state.fp_char_deck if hasattr(state, 'fp_char_deck') else []
        strat_deck = state.fp_strat_deck if hasattr(state, 'fp_strat_deck') else []
    else:
        char_deck = state.sh_char_deck if hasattr(state, 'sh_char_deck') else []
        strat_deck = state.sh_strat_deck if hasattr(state, 'sh_strat_deck') else []

    # Draw character card
    if char_deck:
        card_id = char_deck.pop()
        hand.append(card_id)
        from cards import get_card
        card = get_card(card_id)
        drawn.append({
            "card_id": card_id,
            "card_name": card.card_name if card else card_id,
            "deck": "character"
        })

    # Draw strategy card
    if strat_deck:
        card_id = strat_deck.pop()
        hand.append(card_id)
        from cards import get_card
        card = get_card(card_id)
        drawn.append({
            "card_id": card_id,
            "card_name": card.card_name if card else card_id,
            "deck": "strategy"
        })

    return {
        "drawn": drawn,
        "hand_size": len(hand),
        "over_limit": len(hand) > 6
    }


def discard_card_from_hand(state: gs.GameState, side: gs.Side, card_idx: int) -> dict:
    """
    Discard a card from hand by index.
    Returns info about discarded card.
    """
    hand = state.hands[side].hand

    if card_idx < 0 or card_idx >= len(hand):
        return {"success": False, "message": "Invalid card index"}

    card_id = hand[card_idx]

    # CARD CONSERVATION + LEARNED-DISCARD ROUTING (2026-08-15): this
    # used to `hand.pop(card_idx)` and return — the card reached NO pile
    # (annihilated) and an open hand-limit `pending_discard` was left
    # stale, which would have stalled the GUI once next_turn started
    # opening the pending instead of auto-trimming.  Both are fixed HERE
    # in the shared engine, so gui.py needs no change: its Discard
    # button IS the human's learned pick.
    discard_hand_card(state, side, card_id)

    from cards import get_card
    card = get_card(card_id)
    card_name = card.card_name if card else card_id

    return {
        "success": True,
        "discarded": card_id,
        "card_name": card_name,
        "hand_size": len(hand)
    }


def initialize_decks(state: gs.GameState) -> None:
    """
    Initialize and shuffle all card decks.
    Called at game start.
    Stores card IDs (strings), not Card objects.
    """
    from cards import get_cards_by_deck

    # Free Peoples decks - extract card_id strings
    fp_char = [c.card_id for c in get_cards_by_deck(gs.Side.FREE_PEOPLES, "Character")]
    fp_strat = [c.card_id for c in get_cards_by_deck(gs.Side.FREE_PEOPLES, "Strategy")]
    random.shuffle(fp_char)
    random.shuffle(fp_strat)
    state.fp_char_deck = fp_char
    state.fp_strat_deck = fp_strat

    # Shadow decks - extract card_id strings
    sh_char = [c.card_id for c in get_cards_by_deck(gs.Side.SHADOW, "Character")]
    sh_strat = [c.card_id for c in get_cards_by_deck(gs.Side.SHADOW, "Strategy")]
    random.shuffle(sh_char)
    random.shuffle(sh_strat)
    state.sh_char_deck = sh_char
    state.sh_strat_deck = sh_strat


# PERSISTENT CARD EFFECTS - GUI-Independent Functions

def remove_persistent_card(state: gs.GameState, card_name: str) -> dict:
    """
    Remove a persistent card from play by name.
    Returns success status.
    """
    from game_state import PersistentCardState

    if not hasattr(state, 'persistent_cards') or state.persistent_cards is None:
        return {"success": False, "message": "No persistent cards"}

    if isinstance(state.persistent_cards, PersistentCardState):
        removed = state.persistent_cards.remove_by_name(card_name)
        # CARD CONSERVATION (2026-08-15): the GUI's table-card removals
        # (Crebain / Wizard's Staff / Mithril / Axe and Bow / Horn of
        # Gondor) route through here - discard face-up, never delete.
        if removed:
            from persistent_effects import discard_table_card as _dtc9
            _dtc9(state, removed)
        return {"success": removed, "removed": card_name if removed else None}

    # Legacy dict format
    for side in [gs.Side.FREE_PEOPLES, gs.Side.SHADOW]:
        cards = state.persistent_cards.get(side, [])
        for card_id in list(cards):
            from cards import get_card
            card = get_card(card_id)
            if card and card.card_name == card_name:
                cards.remove(card_id)
                return {"success": True, "removed": card_name}

    return {"success": False, "message": f"Card '{card_name}' not found"}



# RANDOM COMPANION SACRIFICE

def sacrifice_random_companion(state: gs.GameState) -> dict:
    """
    Sacrifice a random companion (not Gollum) to absorb hunt damage.

    Returns:
        {"success": bool, "victim": str or None, "message": str}
    """
    import random
    companions = [c for c in state.fellowship.companions if c.name != "Gollum"]

    if not companions:
        return {"success": False, "victim": None, "message": "No companions to sacrifice"}

    victim = pick_random_companion(companions, 'sacrifice draw')

    if hobbit_hunt_placement_applies(state, victim.name):
        # Randomly drawn Hobbit: placed on the board (as if separated),
        # not eliminated. Damage absorption is handled by the caller.
        state.fellowship.companions.remove(victim)
        # almanac (QA 2026-07-11 Worn CASE 9): a random Hunt pick is
        # ALWAYS a casualty for 'Worn with Sorrow and Toil', even when
        # Take-Them-Alive keeps the hobbit on the board
        _on_companion_casualty(state, victim.name)
        try:
            if state.fellowship.guide not in state.fellowship.companions:
                change_guide(state, allow_pending=True)
        except Exception:
            pass
        place_separated_companion(state, victim, fellowship_region(state))
        return {"success": True, "victim": victim.name,
                "message": f"Random casualty: {victim.name} is a Hobbit — "
                           f"placed on the board (not eliminated)"}

    state.fellowship.companions.remove(victim)
    _on_companion_casualty(state, victim.name)

    # Update guide if needed
    try:
        change_guide(state, allow_pending=True)
    except Exception:
        pass

    return {"success": True, "victim": victim.name, "message": f"Random casualty: {victim.name}"}

# EXTRA SHADOW STRONGHOLD TILE

def draw_extra_stronghold_tile(state: gs.GameState) -> dict:
    """
    Draw an extra hunt tile when revealed in/passing through shadow stronghold.
    Eye tiles have no effect when drawn this way.

    Returns:
        {"success": bool, "tile": HuntTile or None, "damage": int, "is_eye": bool, "message": str}
    """
    if not state.hunt_pool:
        return {"success": False, "tile": None, "damage": 0, "is_eye": False,
                "message": "Hunt pool empty - no extra tile"}

    tile = pick_hunt_tile(state)
    state.hunt_pool.remove(tile)

    # Get tile name - use special_name if set, otherwise repr (e.g., "2R", "Eye")
    tile_name = tile.special_name if tile.special_name else repr(tile)

    # Eye tiles do nothing when drawn as extra stronghold tiles
    if tile.is_eye:
        state.hunt_discard.append(tile)  # used pile (returns on entry)
        return {"success": True, "tile": tile, "damage": 0, "is_eye": True,
                "message": f"Extra tile: {tile_name} (Eye tile - no effect)"}

    damage = tile.damage
    if damage > 0:
        state.fellowship.corruption += damage
    state.hunt_discard.append(tile)

    return {"success": True, "tile": tile, "damage": damage, "is_eye": False,
            "message": f"Extra tile: {tile_name}, +{damage} corruption"}

# END OF FILE


# Moved from the deleted closed_loop_test (teacher retired 2026-07-27):
# a pure rules helper — legal piece-add headroom for a nation.
def _clamp_place(state, nation, kind, want):
    """How many pieces of `kind` may be LEGALLY added to `nation` now
    without exceeding ai.NATION_MAX_FIGURES.

    Conservation doctrine (1.188/1.189, extended to the teacher in 1.194):
    a log-alignment FORCING path must never MINT an over-max board.  When
    a forced placement would push deployed (+ permanent FP casualties for
    Free Peoples) over the nation cap, place only up to the cap and DROP
    the excess -- keeping the board rules-legal so subsequent samples train
    on valid states, at the cost of a sanctioned trajectory divergence
    (parity with the casualty-path decision).  Returns
    min(want, remaining_capacity), clamped at >= 0.

    `kind` accepts a gs.UnitKind (REGULAR/ELITE) or a string
    ('regular'/'elite'/'leader'/'nazgul'); nazgul/leader map to the shared
    leader pool (SAURON's cap of 8 = the 8 Nazgul).  Mirrors
    ai._can_recruit's exact accounting, but returns the count.
    """
    import ai
    from game_state import Side as _S_cp
    FP = _S_cp.FREE_PEOPLES

    try:
        want = int(want)
    except Exception:
        return 0
    if want <= 0:
        return 0
    if isinstance(kind, gs.UnitKind):
        typ = 'elite' if kind == gs.UnitKind.ELITE else 'regular'
    else:
        _k = str(kind).lower()
        typ = ('leader' if _k in ('leader', 'nazgul')
               else 'elite' if _k == 'elite' else 'regular')
    # ONE ANSWERER (USER ruling 2026-08-18, strict component limits):
    # the cap arithmetic lives in ai.figures_available; this is the
    # "clamp a requested count to it" wrapper.
    return max(0, min(want, ai.figures_available(state, nation, typ)))
