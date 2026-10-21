# game_logic.py - rules engine (every die-spending action names its die index)
from __future__ import annotations
import random
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import board
import game_state as gs
from board import sanitize

# Army stacking limit: max 10 units per army (leaders don't count)
MAX_ARMY_UNITS = 10

# Teacher-installed evidence gate for the Balrog-of-Moria extra hunt tile
# (None = self-play, rules-default fire)
BALROG_TILE_GATE = None

# Siege stacking limit: max 5 army units inside a besieged stronghold
MAX_SIEGE_UNITS = 5
# Military victory thresholds (rulebook p.17)
SHADOW_VP_VICTORY = 10
FP_VP_VICTORY = 4

# Geographic map regions, separate from the Mordor Track (rulebook p. 43).
# Ordinary Fellowship map travel may enter or cross them; Track entry is
# checked separately at Minas Morgul or Morannon during the Fellowship phase.
_MORDOR_INTERIOR = frozenset((sanitize('gorgoroth'),
                              sanitize('barad dur'),
                              sanitize('nurn')))


# ---------------------------------------------------------------------------
# RANDOMNESS FUNNELS: every d6 batch, hunt-tile pick and random companion/
# card draw in the rules engine goes through these helpers.  Default
# bodies are the plain random calls (seeded runs unchanged); the LIVE
# bridge patches them so a real advance consumes the client's own rolls;
# MCTS clones keep the defaults.
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


def leader_must_follow(ld) -> bool:
    """A plain Leader figure that may never stand alone (rulebook
    L783-785 / L2436-2439) and therefore must follow a full vacate.
    Nazgul, Companions and the named Minions may stay behind on their own
    (W4-035, ruling W4-Q12 2026-09-03).  The set is EXACTLY the one
    remove_orphaned_leaders removes."""
    if isinstance(ld, (gs.Nazgul, gs.Companion)):
        return False
    if getattr(ld, 'name', None) in ('Saruman', 'Witch King',
                                     'Mouth of Sauron'):
        return False
    return isinstance(ld, gs.Leader)


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
    # Side filter: a co-located besieged entry holds BOTH sides' figures;
    # side=None keeps legacy truthiness (W5 B-1).
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
def _side_of(n: gs.Nation) -> gs.Side:
    # ONE implementation: combat.side_of_nation (lazy import dodges the cycle)
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
    # 1.178a / 1.215 (rulebook 1712-1714): an Elven Ring never produces
    # Will of the West - a Will-ONLY requirement is not waivable
    # (mirrors ai._consume_die_for_face; GL-2).
    _will_only = set(wanted) == {gs.ActionDieFace.WILL_OF_WEST}
    if not valid and not die.used and not _will_only \
            and ring_waiver_active(state, side) \
            and ring_waiver_permits(side, die.current):
        # A7: a paid Elven Ring widens which die may fund the action; the
        # waiver is spent ONLY when it actually rescued this consumption.
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


def _settle_end_of_turn(state: gs.GameState) -> bool:
    """Finish Action Resolution, then check victory, before any turn reset.

    Call exactly once immediately before a nonterminal reset. A terminal
    result is retained so repeated rollover requests cannot charge again.
    """
    if getattr(state, '_search_endturn_victory', None) is not None:
        return True
    result = check_all_victory_conditions(state)
    if not should_game_end(result, end_of_turn=False):
        # Rulebook pp43-44: inactivity is part of phase 5; military victory
        # is checked in phase 6. An existing ring victory is already final.
        if (state.fellowship.mordor_entered
                and not getattr(state, 'fellowship_acted_in_mordor', False)
                and state.fellowship.corruption < 12):
            state.fellowship.corruption = min(
                12, state.fellowship.corruption + 1)
        result = check_all_victory_conditions(state)
    if should_game_end(result, end_of_turn=True):
        state._search_endturn_victory = dict(result)
        return True
    return False


def begin_turn_reset(state: gs.GameState, new_turn, deal_cards=True,
                     bonus_dice=True):
    """THE TURN BOUNDARY'S RULES, WITH THE TURN NUMBER SUPPLIED BY THE
    CALLER. Resolve inactivity before the Victory Check. Return True on
    any terminal outcome, before changing the turn, dice, or decks.

    ONE implementation of the boundary (A12, 2026-08-19) shared by AUTO
    (`next_turn`), the parser and the sided bridge; the callers need
    different halves, hence the split:

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
    # The optional guide draw belongs to the preceding action, even when
    # its funding die was the last die. Resolve it (or explicitly decline)
    # before recovering dice or drawing the next turn's unknown cards.
    if (getattr(state, '_gandalf_draw_pending', None) is not None
            or (getattr(state, 'pending_card_option', None) or {}).get(
                'gandalf_draw')):
        raise RuntimeError('Resolve the pending Gandalf draw before the next turn')
    if _settle_end_of_turn(state):
        return True

    # The turn number is the CALLER's (AUTO: turn+1; sided/parser: the
    # client's marker).
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
    # EX-13: the ring's requested face is cleared with the waiver (runtime
    # attributes, not dataclass fields)
    state.fp_ring_waiver_face = None
    state.sh_ring_waiver_face = None
    state._ring_free_side = None
    # EX-5: the battle-attacker marker is per-action state; the new turn's
    # first actor is the Free Peoples (rulebook 1406-1407)
    state._battle_resolved_attacker = None
    state.sh_mouth_muster_used_this_turn = False
    state.fp_token_used_this_turn = False
    state.sh_token_used_this_turn = False
    state.dice_rolled = False

    # Reset Mordor activity tracking for the new turn
    state.fellowship_acted_in_mordor = False

    # bonus_dice=False: growing the roster is legal only when the caller also
    # drags the physical die piece; the sided bridge does that itself at
    # recover (_pool_growth_lines, own side only).  Growing both pools with
    # no piece wedges live play ("Shadow dice never rolled: slot 7").
    if bonus_dice:
        calculate_bonus_dice(state)

    if deal_cards:
        for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            for _deck9 in ("character", "strategy"):
                try:
                    draw_event_card(state, s, _deck9)
                except RuntimeError as _e9:
                    # GL-9: only draw_event_card's documented
                    # empty-deck refusal skips the draw; anything else
                    # is a defect and propagates
                    if 'deck empty' not in str(_e9):
                        raise
        # USER RULING 2026-08-15: the hand-limit discard is a LEARNED decision -
        # the rollover only OPENS the pending; whoever drives the turn
        # resolves it.
        open_hand_limit_discard(state)
    return False


def next_turn(state: gs.GameState):
    # Heal stale single-side enemy_controlled flags at the turn boundary
    # (non-combat reoccupation paths run no capture bookkeeping).  Self-play
    # only - the teacher keeps its certified post-combat call pattern.
    import combat as _cmb_nt
    if _cmb_nt.FINAL_HITS_HOOK is None:
        # GL-9: no guard - a failing control heal is a defect, never a
        # silently stale flag
        enforce_settlement_control(state)
    # Everything else is begin_turn_reset (A12): AUTO passes turn+1 and
    # deal_cards=True, so this function's behaviour is unchanged.
    # The stay-heal (rulebook p.34) is NOT applied here: healing requires
    # re-declaring in place; the declare path applies it.
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
    _pend = getattr(state, 'pending_discard', None)
    if _pend is not None:
        # BL-150 E2 (2026-08-28): RE-DERIVE instead of trusting an open pending -
        # one opened before an optioned card left the hand goes stale; keep it
        # only while that side is still over the limit.
        _ps = _pend.get('side') if isinstance(_pend, dict) else None
        if _ps is not None and _ps in state.hands \
                and len(state.hands[_ps].hand) > HAND_LIMIT:
            return True
        state.pending_discard = None
        if _ps is not None and _ps in state.hands:
            state.hands[_ps].needs_discard = False
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
    """Reject obsolete snapshots that lost the ring's physical target.

    New states change die.current and never create a payment waiver.
    An old active waiver cannot be safely reconstructed from its flag.
    """
    if getattr(state, _RING_WAIVER_ATTR[side], False):
        raise RuntimeError('Legacy Elven Ring waiver has no reliable physical target; resync or replay this game')
    return False


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
    # EX-13: a spent waiver also clears the requested face
    setattr(state, 'fp_ring_waiver_face'
            if side is gs.Side.FREE_PEOPLES else 'sh_ring_waiver_face',
            None)


def use_elven_ring(state: gs.GameState, side: gs.Side, die_idx: int,
                   new_face: gs.ActionDieFace) -> str:
    """Change one unused die's actual result (rulebook p.21).

    The changed die stays unused and retains its new face even if the
    player uses another die or passes (Almanac, The Day Without Dawn).
    A Shadow change to Eye is immediately placed in the Hunt Box.
    """
    if side not in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        raise RuntimeError("Invalid Elven Ring side")
    if not isinstance(new_face, gs.ActionDieFace):
        raise RuntimeError("Invalid Elven Ring result")
    ring_waiver_active(state, side)
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
        if new_face == gs.ActionDieFace.ARMY:
            raise RuntimeError("Free Peoples Army result is Muster/Army")
    else:
        if state.sh_elven_rings <= 0:
            raise RuntimeError("No Elven Rings available")
        if state.sh_ring_used_this_turn:
            raise RuntimeError("Already used an Elven Ring this turn")
        # 1.178a: the ring may never produce Will of the West, for either side
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
    if old_face == new_face:
        raise RuntimeError("Elven Ring must change the die result")

    # Mark ring as used
    if side == gs.Side.FREE_PEOPLES:
        state.fp_elven_rings -= 1
        state.fp_ring_used_this_turn = True
        # Transfer ring to Shadow
        state.sh_elven_rings += 1
    else:
        state.sh_elven_rings -= 1
        state.sh_ring_used_this_turn = True

    die.current = new_face
    ring_waiver_spend(state, side)  # obsolete flags carry no new permission
    if side == gs.Side.SHADOW and new_face == gs.ActionDieFace.EYE:
        die.used = True
        state.hunt_box.append(die)
        return f"Changed {old_face.value} to Eye (moved to Hunt Box)"

    return f"Changed {old_face.value} to {new_face.value} with Elven Ring"

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
    differs, run the capture bookkeeping.
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
    lift.  Call at STABLE points only (no pending_combat — mid-battle
    windows legally hold transient one-sided compositions).
    """
    enforce_settlement_control(state)
    # fix 93: the siege flag is DERIVABLE state - a stronghold held by BOTH
    # sides IS a siege (co-located model); re-derive (ADD) before the lift
    # pass.
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
        # P2-10 (USER 2026-07-20): the 5-unit besieged-garrison cap is a
        # standing invariant re-clamped at every stable point; excess returns
        # to REINFORCEMENTS (1.181), never casualties.
        from combat import garrison_side_for as _gsf10
        _gside = _gsf10(state, sanitize(r))
        _gar = [u for u in entry.units if _son(u.nation) == _gside]
        if len(_gar) > MAX_SIEGE_UNITS:
            # USER RULING 2026-07-27: WHICH excess units return is the owner's
            # learned decision on the self-play path (the R1 detector below
            # opens '__overstack__'); the heuristic clamp is TEACHER-only
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
            # 1.181: an untracked removal IS a pool return - no casualty
            # tracking
    # P2-03 (USER 2026-07-20): Denethor's Folly leaves play once Minas Tirith
    # is not under siege; checked AFTER the re-derive/lift passes (audit A4).
    _pcards = getattr(state, 'persistent_cards', None)
    if _pcards is not None:
        if sanitize('minas tirith') not in bs:
            _folly = _pcards.get_by_type(
                gs.PersistentCardType.DENETHORS_FOLLY)
            if _folly is not None:
                # CARD CONSERVATION (2026-08-15): discarded face-up
                from persistent_effects import discard_table_card as _dtc9
                _dtc9(state, _pcards.remove_card(_folly.card_id) or _folly)
    # R1 standing detector: a side over its per-side cap at a stable point is
    # registered and resolved through the 1.112.7 overstack machinery
    # (learned pending); self-play only - teacher compositions are log truth.
    import combat as _cmb_r1
    if _cmb_r1.FINAL_HITS_HOOK is None:
        # GL-9: no guard - an overstack that cannot be detected or
        # resolved is a defect, never a silently full-strength stack
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
        # Siege trimming or battle losses can satisfy an earlier overflow.
        # Reconcile its ledger at this stable point even when the scan found
        # no newly oversized army. Open combat/card choices retain ownership.
        if (_found_r1 or getattr(state, '_overstack_regions', None)) and state.pending_combat is None:
            resolve_overstacks(state, learned=True)


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


def _companion_activation_nation(state: gs.GameState, to_san: str, arriving_leaders):
    """Passive nation activated by these Companions ending movement here.

    Also identifies useful stationary movement (Almanac p.41).
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

    # Only a Companion who ended this movement can activate the nation.
    # A figure already at the destination may have arrived by retreat.
    for ld in arriving_leaders:
        if isinstance(ld, gs.Companion):
            if companion_activates(ld.name, region_nation):
                if not state.political.active.get(region_nation, False):
                    return region_nation
                return  # Only need to activate once


def _check_companion_activates_nation(state: gs.GameState, to_san: str, arriving_leaders):
    """Apply ordinary arrival activation using only the moving Companions."""
    nation = _companion_activation_nation(state, to_san, arriving_leaders)
    if nation is not None:
        try:
            activate_nation(state, nation, method="companion_move", region=to_san)
        except RuntimeError:
            pass


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
    """THE destination stacking cap.
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
              side: Optional[gs.Side] = None,
              nation_ok: Optional[Callable[[gs.Nation], bool]] = None,
              combat_context: Optional[dict] = None,
              movement_roster=None,
              defer_stacking: bool = False):
    """
    Move pieces from *fr* to *to*.

    • If no counts are supplied, the **entire** stack moves (old behaviour).
    • Otherwise supply how many of each category to move.
    • *side*: the ACTING side.  Required for correctness when the source
      is a shared besieged-region entry holding BOTH sides' units — the
      old units[0] heuristic misattributed the mover (and then the wrong
      side's units/dice were consumed).
    • *nation_ok* (W4-038, rulebook L3093-3096 / L4189-4191): 'may Nation
      n cross into *to*' on a VOLUNTARY move over another Nation's border.
      Units and plain Leaders whose nation fails it are not movable (they
      stay, exactly like the enemy side's figures of a shared entry);
      Companions / Minions / Nazgul are not nation figures.  None (the
      default, retreats / advances / card moves) = no restriction.
    """
    b = state.board

    # Combat advance resolves excess through the caller's existing learned
    # trim window after entry. Ordinary army movement retains its hard cap.
    if defer_stacking and (not combat_context
                           or combat_context.get('phase') != 'victor_advance'):
        raise ValueError('Deferred stacking requires a victory-advance context')

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
    if to_san != fr_san and to_san not in b.adjacent(fr_san):
        raise RuntimeError(f"{to} not adjacent to {fr}")

    src = state.armies.get(fr_army_key)
    if not src:
        raise RuntimeError(f"No army in {fr}")

    # Get the destination region's original name for consistent army key
    to_region = b.region(to_san)
    to_army_key = to_region.orig_name if to_region else to
    # Reuse an existing key that sanitizes to the same region ("Helm's Deep" vs
    # "Helm's deep"); a fresh setdefault forges a duplicate army entry.
    to_army_key = next((k for k in state.armies
                        if sanitize(k) == to_san), to_army_key)

    # Determine moving army's side for political activation check
    _moving_side = side
    if _moving_side is None and src.units:
        from combat import side_of_nation as _son
        _moving_side = _son(src.units[0].nation)

    # A victory advance / siege entry transports only the battle's frozen
    # participants. Keep the physical rear in src throughout orphan checks.
    _battle_unit_ids = _battle_leader_ids = None
    if movement_roster is not None:
        _mu, _ml = movement_roster
        _battle_unit_ids = {id(u) for u in _mu}
        _battle_leader_ids = {id(l) for l in _ml}
        if move_units is move_elite is move_leaders is move_nazgul is None:
            move_units = sum(u.kind == gs.UnitKind.REGULAR for u in _mu)
            move_elite = sum(u.kind == gs.UnitKind.ELITE for u in _mu)
            move_leaders = len(_ml)
    if combat_context is not None:
        from army_participants import partition_battle_army
        _bu, _bl, _, _ = partition_battle_army(
            state, src, fr_san, _moving_side, combat_context)
        _battle_unit_ids = {id(u) for u in _bu}
        _battle_leader_ids = {id(l) for l in _bl}
        if move_units is move_elite is move_leaders is move_nazgul is None:
            move_units = sum(u.kind == gs.UnitKind.REGULAR for u in _bu)
            move_elite = sum(u.kind == gs.UnitKind.ELITE for u in _bu)
            move_leaders = len(_bl)

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

    # ONE destination-cap body (army_stack_cap), shared with the ai.py menu
    # and the card-move path
    _stack_cap = army_stack_cap(state, to_san, _moving_side)

    # A shared besieged-region entry holds BOTH sides: "whole stack" for
    # the acting side means ALL of ITS units — convert to a full split so
    # the enemy garrison/besiegers stay behind.
    if move_units is move_elite is move_leaders is None \
            and move_nazgul is None and src.units \
            and (_moving_side is not None or nation_ok is not None):
        from combat import side_of_nation as _son_ws
        # W4-038: with a border predicate the "whole stack" is every own
        # figure that MAY cross - the rest stays (a split, never a merge)
        if nation_ok is not None or \
                any(_son_ws(u.nation) != _moving_side for u in src.units):
            _own_ws = [u for u in src.units
                       if (_moving_side is None
                           or _son_ws(u.nation) == _moving_side)
                       and (nation_ok is None or nation_ok(u.nation))]
            move_units = sum(1 for u in _own_ws
                             if u.kind is gs.UnitKind.REGULAR)
            move_elite = sum(1 for u in _own_ws
                             if u.kind is gs.UnitKind.ELITE)
            move_leaders = sum(1 for l in src.leaders
                               if not _is_saruman(l)
                               and (_moving_side is None
                                    or _leader_side(l) == _moving_side))

    # ----- default: move whole stack (an explicit move_nazgul count is
    # a split, never a whole-stack move)
    if move_units is move_elite is move_leaders is None \
            and move_nazgul is None:
        if fr_san == to_san:
            arrivals = [l for l in src.leaders
                        if _moving_side is None or _leader_side(l) == _moving_side]
            if not src.units or _companion_activation_nation(state, to_san, arrivals) is None:
                raise RuntimeError("Stationary army movement must activate a nation")
            _check_companion_activates_nation(state, to_san, arrivals)
            return
        # Check stacking limit before merge (own-side units only)
        dst_existing = state.armies.get(to_army_key)
        if dst_existing:
            combined_units = _own_units_at_dst() + len(src.units)
            if not defer_stacking and combined_units > _stack_cap:
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
        _arriving_leaders = list(src.leaders)

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
        _check_companion_activates_nation(state, to_san, _arriving_leaders)
        # Remove FP Leaders left alone without combat units (rulebook p.8)
        remove_orphaned_leaders(state)
        # Rulebook p.32: entry captures an undefended settlement immediately,
        # whether the player moved a whole army or split it. Do not depend on
        # a later AI/turn cleanup to perform this part of the move.
        if _moving_side and dst.units:
            from combat import side_of_nation, _check_settlement_capture
            if not any(side_of_nation(u.nation) != _moving_side
                       for u in dst.units):
                _check_settlement_capture(state, to_san, _moving_side)
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
        if (_moving_side is None or _son_sp(u.nation) == _moving_side) \
                 and (_battle_unit_ids is None or id(u) in _battle_unit_ids):
            _own_units.append(u)
        else:
            _enemy_units.append(u)
    # W4-038 (c): on a voluntary foreign crossing the non-belligerent
    # nations' units may not cross (rulebook L3093-3096, L4189-4191) -
    # they are held back like the enemy side's units, whatever the list
    # order, so a count never resolves to a figure that may not move.
    _held38 = []
    if nation_ok is not None:
        _held38 = [u for u in _own_units if not nation_ok(u.nation)]
        _own_units = [u for u in _own_units if nation_ok(u.nation)]

    m_elite , keep_elite  = take([u for u in _own_units if u.kind is gs.UnitKind.ELITE],
                                 move_elite)
    m_regular, keep_reg   = take([u for u in _own_units if u.kind is gs.UnitKind.REGULAR],
                                 move_units)
    # Only the moving side's own leaders may travel with the army;
    # Saruman never moves (stays in Orthanc — official rule)
    def _movable_leader(l):
        return (not _is_saruman(l)
                and (_battle_leader_ids is None or id(l) in _battle_leader_ids)
                and (_moving_side is None
                     or _leader_side(l) == _moving_side)
                # W4-038 (c): a non-belligerent Nation's plain Leader
                # never crosses another Nation's border (L3093-3096);
                # Companions / Minions / Nazgul are not nation figures
                and (nation_ok is None
                     or not leader_must_follow(l)
                     or nation_ok(l.nation)))
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
    # W4-035 constraint (b): under a leader COUNT the plain Leaders (the
    # figures that may not be left behind) are CHOSEN first, whatever the
    # list order - [Companion, Leader, Leader] with move_leaders=2 moves
    # both Leaders and leaves the Companion; both the moved and the kept
    # lists keep their original order (only the choice changes).
    def _must_first(lst):
        return ([l for l in lst if leader_must_follow(l)]
                + [l for l in lst if not leader_must_follow(l)])

    if move_nazgul is None:
        _chosen9, _ = take(
            _must_first(src.leaders), move_leaders,
            predicate=_movable_leader)
        _mid9 = set(map(id, _chosen9))
        m_lead = [l for l in src.leaders if id(l) in _mid9]
        keep_lead = [l for l in src.leaders if id(l) not in _mid9]
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
        _chosen9, _ = take(
            _must_first(_rest9), move_leaders,
            predicate=lambda l: (_movable_leader(l)
                                 and not isinstance(l, gs.Nazgul)))
        _mid9 = set(map(id, _chosen9))
        m_oth9 = [l for l in _rest9 if id(l) in _mid9]
        keep_lead = [l for l in _rest9 if id(l) not in _mid9]
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

    if fr_san == to_san:
        if not (m_elite or m_regular) or _companion_activation_nation(state, to_san, m_lead) is None:
            raise RuntimeError("Stationary army movement must activate a nation")
        _check_companion_activates_nation(state, to_san, m_lead)
        return

    # A plain Leader cannot be voluntarily stranded without own Army units.
    # In particular, a peaceful Leader forbidden to cross another nation
    # needs a unit left behind; do not route this illegal move through orphan cleanup.
    if (any(leader_must_follow(l) and (_moving_side is None
            or _leader_side(l) == _moving_side) for l in keep_lead)
            and not (keep_elite or keep_reg or _held38
                     or any(_moving_side is None or _son_sp(u.nation) == _moving_side
                            for u in _enemy_units))):
        raise RuntimeError("mandatory plain leader needs a unit left behind")

    # Check stacking limit before merge (own-side units only; siege cap
    # only when reinforcing the garrison side inside a besieged stronghold)
    moving_units = len(m_elite) + len(m_regular)
    dst_existing = state.armies.get(to_army_key)
    if not defer_stacking and dst_existing and _own_units_at_dst() + moving_units > _stack_cap:
        raise RuntimeError(f"Move would exceed stacking limit ({_stack_cap} units)")

    # Moved figures leave the source; the enemy side's units in a shared
    # entry always stay
    src.units   = keep_elite + keep_reg + _held38 + _enemy_units
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
    _check_companion_activates_nation(state, to_san, m_lead)
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

    # rulebook 3361-3363: only a HIDDEN Fellowship may declare - the
    # primitive refuses rather than hand a revealed one a free hide (GL-11)
    if not state.fellowship.hidden:
        raise RuntimeError("Fellowship is revealed - cannot declare")

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

    # 1.211i: The Last Battle discards on DECLARATION as well as reveal (Mordor
    # entry is a declaration, 1.155); remove_card is idempotent.
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

    # USER RULING 2026-07-27: a DECLARATION draws NO Shadow-stronghold tile
    # (marker p.37: reveal-only).  TEACHER-ONLY: the log-evidence-gated draw
    # replays historical games whose humans resolved one.
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
        # Almanac p30/37/39: Saruman can die during this battle. The attack's
        # blocked Rohan reaction must then resolve at action end, even if the
        # defending Army has since been eliminated. Lower-box advancement
        # already happens below; only a blocked box-two advance is still owed.
        from persistent_effects import can_activate_nation
        if (nation == gs.Nation.ROHAN
                and not state.political.active.get(nation, False)
                and not can_activate_nation(state, nation, "attack",
                                            region=dfn_san)[0]):
            state._wormtongue_attack_pending = max(
                getattr(state, '_wormtongue_attack_pending', None) or 0,
                2 if state.political.boxes[nation] == 2 else 1)
        try:
            if not state.political.active.get(nation, False):
                # Route through activate_nation so persistent-card blockers
                # (Wormtongue) apply on the attack path too.
                try:
                    activate_nation(state, nation, method="attack",
                                    region=dfn_san)
                except RuntimeError:
                    pass   # blocked (e.g. Wormtongue) — no flip
            if state.political.boxes[nation] < 3:
                advance_political(state, nation, method="attack")
        except RuntimeError as _e9:
            # GL-9: only advance_political's documented refusal is
            # tolerated (passive nation at box 2 / persistent-card
            # block); anything else propagates
            if 'Cannot advance' not in str(_e9):
                raise


def settle_wormtongue_attack(state: gs.GameState) -> None:
    """Settle one recorded attack reaction once all its choices complete."""
    debt = getattr(state, '_wormtongue_attack_pending', None)
    if not debt:
        return
    if any(getattr(state, key, None) is not None for key in (
            'pending_combat', 'pending_card_option', 'pending_card_followup',
            'pending_discard')) or getattr(state, '_overstack_regions', None):
        return
    state._wormtongue_attack_pending = None
    if state.persistent_cards.has_card_type(gs.PersistentCardType.WORMTONGUE):
        return
    activate_nation(state, gs.Nation.ROHAN, method="attack")
    if debt == 2:
        if state.political.boxes[gs.Nation.ROHAN] < 3:
            advance_political(state, gs.Nation.ROHAN, method="attack")
        else:
            # A later capture may already have filled the last box. The
            # attack's owed advance still triggers its own table-card expiry;
            # never grant an additional box or remove the capture's advance.
            from persistent_effects import (
                check_nation_advanced_discards, discard_table_card)
            for card in check_nation_advanced_discards(
                    state, gs.Nation.ROHAN, "attack"):
                removed = state.persistent_cards.remove_card(card.card_id)
                discard_table_card(state, removed or card)


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
    # No per-turn attack restriction: the rules allow attacking the same
    # region repeatedly with separate Action dice.
    dfn_san = sanitize(dfn_r)

    # Defenders activate/advance at attack DECLARATION
    # (advance_defenders_political, from the attack executor), never per round.
    # OVERSIGHT #2: publish the live state so combat's downgrade supply
    # checks work.
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
        # RESTORE the prior state (a card handler's published state must
        # survive the battle it triggers)
        set_active_state(_prev_as)
        settle_wormtongue_attack(state)


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
    # rulebook 2131-2135: the Witch-king IS a Nazgul (is_nazgul_figure is
    # the single source of truth, 1.109) - Hunt re-roll source (GL-3)
    a = _army_at_fellowship(s)
    return bool(a and any(is_nazgul_figure(l) for l in a.leaders))
def _shadow_army_present(s: gs.GameState) -> bool:
    a = _army_at_fellowship(s);  return bool(a and any(_side_of(u.nation) is gs.Side.SHADOW for u in a.units))

# Memo: pure functions of an immutable region name and the import-time
# board tables; byte-identical, ~200 distinct keys.
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
        # Gollum takes over as guide WITH his ability (ordinary reveal tiles do
        # not reveal the Fellowship)
        state.fellowship.guide = gs.Companion(
            "Gollum", 0, 0, guide_ability="no_reveal_on_damage")
        return
    best_lvl = max(c.level for c in comps)
    best = [c for c in comps if c.level == best_lvl]
    if chosen is not None:
        for c in best:
            if c.name == chosen:
                state.fellowship.guide = c; return
        raise RuntimeError(f"{chosen} not among highest-level companions")
    if len(best) == 1:
        state.fellowship.guide = best[0]; return
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

def fellowship_tail_pending(state):
    """Earlier Fellowship effects that must finish before entry and Guide choice."""
    names = ('pending_combat', 'pending_hunt', 'pending_guide_change',
             'pending_card_option', 'pending_card_followup', 'pending_discard')
    pending = [name for name in names if getattr(state, name, None) is not None]
    pending.extend(name for name in ('_pending_hunt_damage', '_hunt_reveal_pending', '_reveal_place_due',
                                     '_mithril_stash', '_hobbit_place_queue',
                                     '_pending_reveal_strongholds', '_fellowship_route_event')
                   if getattr(state, name, None))
    return pending


def resume_fellowship_tail(state):
    """Finish the declaration/Hunt chain, then entry/refill and optional Guide.

    Drivers resume after each real pending decision; they retain ownership of
    action-dice rolls. The marker is encoded because an otherwise identical
    Hunt or compulsory Guide choice has a different continuation mid-turn.
    """
    if not getattr(state, '_fellowship_tail_phase', False):
        return True
    if state.dice_rolled:
        raise RuntimeError('Fellowship tail must precede action dice rolls')
    if fellowship_tail_pending(state):
        return False
    if should_game_end(check_all_victory_conditions(state), end_of_turn=False):
        state._fellowship_tail_phase = False
        return True
    # Almanac C17: finish every Balrog tile and its consequences before the
    # entering-Mordor Hunt Pool refill, including any compulsory new Guide.
    if not state.fellowship.mordor_entered:
        _check_mordor_entry(state)
    state._fellowship_tail_phase = False
    open_fellowship_guide_choice(state)
    return True


def begin_fellowship_tail(state):
    """Close declaration and expose its resumable effects before Guide/eyes."""
    if state.dice_rolled:
        raise RuntimeError('Fellowship tail must precede action dice rolls')
    close_declare_phase(state)
    if getattr(state, '_eye_allocation_phase', False):
        return True
    if not getattr(state, '_fellowship_tail_phase', False):
        from ai import _balrog_stationary_turn_start
        _balrog_stationary_turn_start(state)
        state._fellowship_tail_phase = True
    return resume_fellowship_tail(state)


def open_fellowship_guide_choice(state: gs.GameState):
    """End Fellowship once: highest-level guide choice, then Hunt allocation.

    The existing eye flag means allocation remains due. Guide obligations
    have higher priority, so this also identifies their preaction context.
    Repeated phase-close signals cannot reopen an already completed choice.
    """
    if state.dice_rolled:
        raise RuntimeError('Fellowship guide phase must precede action dice rolls')
    if getattr(state, '_fellowship_tail_phase', False):
        raise RuntimeError('Earlier Fellowship effects must resolve before Guide choice')
    if getattr(state, '_eye_allocation_phase', False):
        return state.pending_guide_change is not None
    if state.pending_guide_change is not None:
        raise RuntimeError('Earlier compulsory guide choice must resolve first')
    change_guide(state, allow_pending=True)
    state._eye_allocation_phase = True
    return state.pending_guide_change is not None


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
                # 1.148: armies only via get_or_create_army (a raw setdefault
                # forges a shadowed duplicate key)
                army = get_or_create_army(state, target)
                army.leaders.append(comp)
                # 1.149b: separation IS movement - ending it in the
                # companion's own city/stronghold activates the nation
                # (rulebook p.34, 4th trigger)
                _check_companion_activates_nation(
                    state, sanitize(str(target)), [comp])
                break
        else:
            raise RuntimeError(f"{name} not in Fellowship")

    # Rulebook p37: every composition change permits a highest-level Guide choice.
    change_guide(state, allow_pending=True)
    check_companion_persistent_discards(state)

# Army / Character / Political / Card actions
def can_advance_political(state: gs.GameState, nation: gs.Nation,
                          method: str = "muster_die"):
    """THE political-advance rule body.  Menus propose, the executor
    enforces - both through here.  Returns (ok, reason)."""
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
    # ONE rule body (can_advance_political).  An already-At-War advance stays a
    # silent no-op (teacher logs carry duplicate advances); every other
    # illegality raises.
    _ok9, _why9 = can_advance_political(state, nation, method)
    if not _ok9 and box < 3:
        raise RuntimeError(f"Cannot advance {nation.value}: {_why9}")

    if box < 3:
        state.political.boxes[nation] = box + 1
        # USER RULE (session 21, restored session 34): ADVANCING THE TRACK NEVER
        # ACTIVATES.  Activation comes only from its own triggers - attacked/
        # entered with army units, fellowship declared/revealed in the
        # nation's city/stronghold, a companion ending movement there, or a
        # card that says so (activate_nation).

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

    # Almanac p.29 (Wormtongue): occupying empty Edoras/Helm's Deep is
    # movement, not an attack. Only an actual attack earns that exception.
    if attacked or captured:
        from persistent_effects import can_activate_nation as _can9
        _activation_method = "attack" if attacked else "enemy_entry"
        _ok9, _blk9 = _can9(state, nation, _activation_method, region=rkey)
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

    # Additional advance for settlement captured (rulebook 3009-3011: a
    # PASSIVE nation never reaches At War - same clamp as 'attacked'; GL-10)
    if captured and is_settlement:
        old_pos = state.political.boxes.get(nation, 0)
        _cap10 = 3 if state.political.active.get(nation, False) else 2
        if old_pos < _cap10:
            state.political.boxes[nation] = old_pos + 1
            effects.append(f"{nation.value} advanced to {old_pos + 1} (settlement captured)")

    return effects


def is_nation_at_war(state: gs.GameState, nation: gs.Nation) -> bool:
    """Check if a nation is at war (position 3 on political track)."""
    return state.political.boxes.get(nation, 0) >= 3


def can_attack_with_army(state: gs.GameState, army,
                         side: "gs.Side" = None) -> Tuple[bool, str]:
    """Whether a legal attacking portion remains after compulsory rear figures.

    RB28/35: peaceful units AND Leaders stay behind; the rear needs a unit.
    The selected portion contains every other eligible figure. Character
    payment additionally requires leadership in that participating portion.
    """
    from army_participants import select_forced_attack
    if army is None or not army.units:
        return False, 'No units in army'
    if side is None:
        from combat import side_of_nation
        sides = {side_of_nation(u.nation) for u in army.units}
        if len(sides) != 1:
            return False, 'Mixed army requires an acting side'
        side = next(iter(sides))
    formation = select_forced_attack(state, army, side)
    return formation.available, formation.unavailable_reason or 'OK'


# Hunt logic, Fellowship movement, damage resolution
# Hunt dice functions - GUI-driven hunt resolution
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

    # GL-9: pure lookups - a failure here is a defect, never a silently
    # dropped Shadow re-roll
    reroll_sources = []
    if _nazgul_present(state):
        reroll_sources.append("Nazgul present")
    if _shadow_army_present(state):
        reroll_sources.append("Shadow army present")
    if _in_shadow_stronghold(state):
        reroll_sources.append("Shadow stronghold")

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


_REACHABLE_CACHE: dict = {}   # (start, steps, mode) -> frozenset (static board memo)


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


# Internal hunt roll (self-play, card hunts); the GUI uses get_hunt_params +
# roll_hunt_dice
def _hunt_roll(state: gs.GameState, num_dice: int, thresh: int,
               fp_modifier: int = 0) -> int:
    """Roll hunt dice with FP modifier. Natural 1 always misses."""
    # P1: a hunt-time Flocks of Crebain discard adds +1 to ALL dice of THIS
    # hunt (rerolls included); consumed here, before the 0-dice return, so
    # it can never leak into a later hunt.
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
    # GL-9: pure lookups - no guard (see get_hunt_params)
    reroll_sources = 0
    if _nazgul_present(state):
        reroll_sources += 1
    if _shadow_army_present(state):
        reroll_sources += 1
    if _in_shadow_stronghold(state):
        reroll_sources += 1

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
    the production path.  test_autoresolver_purge.py pins the caller
    census (exactly two call sites, both behind explicit False).

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
    # GL-9: a failing requirement check must not keep Axe and Bow / Horn
    # / Staff on the table - no guard
    gone = pc.check_companion_requirements(state.fellowship.companions)
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
    FP hand, or declines this optional effect. Explicit table targets
    must still exist; an invalid target must not discard another card.
    source=None is the LEGACY hand-first heuristic
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
    if source == 'decline':
        return
    if source not in (None, 'hand', 'table'):
        raise ValueError('Invalid Worn with Sorrow and Toil source: %r' % source)
    hand = state.hands[gs.Side.FREE_PEOPLES].hand
    chars = [c for c in hand if card_deck_of(c) == 'character']
    pcards = getattr(state, 'persistent_cards', None)
    table_chars = (pcards.get_fp_character_cards()
                   if pcards is not None else [])
    if source == 'table' and not table_chars:
        raise ValueError('Worn with Sorrow and Toil table target is unavailable')
    if source == 'hand' and not chars:
        raise ValueError('Worn with Sorrow and Toil hand source is unavailable')
    use_table = (source == 'table') or (source is None and not chars)
    if use_table and table_chars:
        target = None
        if card_id is not None:
            target = next((c for c in table_chars
                           if c.card_id == card_id), None)
            if target is None:
                raise ValueError('Worn with Sorrow and Toil table target is unavailable: %r' % card_id)
        if target is None:
            if source == 'table' and len(table_chars) > 1:
                raise ValueError('Worn with Sorrow and Toil requires a named table target')
            target = table_chars[0]
        _rm = pcards.remove_card(target.card_id)
        discard_table_card(state, _rm or target)
        return
    if chars:
        _gone = pick_random_card(chars, 'WWSAT hand discard')
        hand.remove(_gone)
        # 1.261: the WWSAT hand discard is FACE-DOWN - the identity is
        # FP private knowledge (the client renders no name); the card
        # stays tracked in the owner's pile (conservation)
        state.hands[gs.Side.FREE_PEOPLES].discarded_face_down.append(_gone)


def _on_companion_casualty(state: gs.GameState, name: str,
                           casualty: bool = True):
    """Offer Worn with Sorrow and Toil after a Fellowship HUNT casualty.

    The Shadow may discard a random hand Character card, choose a named
    table Character card, or decline. Ordinary army-combat eliminations
    and non-Hunt event eliminations must pass casualty=False.

    USER RULING 2026-08-15: SEPARATION-death is legally distinct from
    being taken as a hunt casualty — a hobbit GUIDE absorbing hunt
    damage separates (which in Mordor kills), and WWSAT does NOT
    trigger on it; with a hobbit guide WWSAT fires only via a real
    casualty-taking (e.g. Foul Thing from the Deep's forced pick).
    casualty=False = leaves-play cleanup only, no WWSAT.

    RECORD-THEN-DRAIN: casualty resolution cannot suspend for a pending,
    so the optional SELF-PLAY decision is recorded on
    state._wwsat_owed and drained as a learned pending after the action
    (ai._offer_wwsat_discard).  The teacher forces the LOGGED source via
    state._wwsat_forced and resolves inline."""
    check_companion_persistent_discards(state)
    if not casualty:
        return
    from persistent_effects import check_companion_casualty_effect
    # Only callers resolving a Fellowship Hunt casualty pass casualty=True.
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
    if (_cmbwwsat.FINAL_HITS_HOOK is None
            and getattr(state, '_wwsat_forced', None) is None):
        owed = getattr(state, '_wwsat_owed', None)
        if owed is None:
            owed = []
            state._wwsat_owed = owed
        owed.append(True)
        return
    _wwsat_resolve(state, None)


# User-verified rule: a RANDOM hunt casualty that draws a Hobbit is NOT
# eliminated - the FP player places him as if separated (progress +
# level).  Separation is impossible on the Mordor track: there he dies.
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
    separates (Foul Thing's forced pick is the same row).
    Self-play/live only: the TEACHER's WWSAT discards are
    forced from the log's own records (certified behavior kept).
    GL-4: the hunt decision passes min(dmg, level) and chains the
    remainder through continue_hunt_chain (new guide's ability live).
    Returns the Companion object to place."""
    if victim in state.fellowship.companions:
        state.fellowship.companions.remove(victim)
    import combat as _cmbrhp
    if _cmbrhp.FINAL_HITS_HOOK is None:
        _on_companion_casualty(state, victim.name, casualty=True)
    # Rulebook p37: every composition change permits a highest-level Guide choice.
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
    _check_companion_activates_nation(state, rk, [comp])


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
                change_guide(state, allow_pending=True)
                break

    net = max(0, dmg - absorbed)
    # USER RULING 2026-07-21: the Axe/Horn discard-to-reduce is a LEARNED FP
    # decision resolved BEFORE this point (_offer_hunt_reduction); the
    # teacher forces the logged use/decline.
    state.fellowship.corruption = min(
        12, state.fellowship.corruption + max(0, net))


def _hunt_reduction_options(state):
    """Choose either protection card, or neither; never both for one tile."""
    pcards = getattr(state, 'persistent_cards', None)
    cards = [pcards.get_by_type(ptype) for ptype in
             (gs.PersistentCardType.AXE_AND_BOW,
              gs.PersistentCardType.HORN_OF_GONDOR)] if pcards else []
    return [{'use_hunt_reduction': 1, 'card_id': card.card_id}
            for card in cards if card is not None] + [{'use_hunt_reduction': 0}]


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
        'options': _hunt_reduction_options(state)}
    return True


def reveal_fellowship_from_hunt(state: gs.GameState) -> None:
    """Commit a Hunt reveal and its map placement, including voluntary Gollum."""
    state._hunt_reveal_pending = 0
    was_hidden = state.fellowship.hidden
    state.fellowship.hidden = False
    if was_hidden and not state.fellowship.mordor_entered:
        state._reveal_place_due = True
        state._sh_tile_drawn_regions = set()


def stage_hunt_reveal(state: gs.GameState, tile, reveal: bool = True) -> None:
    """Rulebook pp.41-42: resolve damage and new Guide abilities before reveal."""
    if not reveal or not state.fellowship.hidden:
        return
    unavoidable = bool(tile.is_eye or getattr(tile, 'is_special', False))
    guide = state.fellowship.guide
    if (not unavoidable and guide is not None
            and getattr(guide, 'guide_ability', None) == 'no_reveal_on_damage'):
        return
    # The drawn tile is public. 1 is suppressible by a new Gollum; 2 is not.
    state._hunt_reveal_pending = 2 if unavoidable else 1


def settle_hunt_reveal(state: gs.GameState) -> bool:
    """Finish this tile's reveal only after its protection/damage/Guide chain."""
    pending = getattr(state, '_hunt_reveal_pending', 0)
    if not pending:
        return False
    if (getattr(state, '_pending_hunt_damage', 0)
            or any(getattr(state, key, None) is not None for key in
                   ('pending_hunt', 'pending_guide_change', 'pending_card_option'))
            or getattr(state, '_hobbit_place_queue', None)):
        return False
    state._hunt_reveal_pending = 0
    guide = state.fellowship.guide
    if (pending == 1 and guide is not None
            and getattr(guide, 'guide_ability', None) == 'no_reveal_on_damage'):
        return True
    reveal_fellowship_from_hunt(state)
    return True


def _gollum_reveal_available(state: gs.GameState) -> bool:
    """THE predicate for Gollum's reveal-to-reduce (Gollum card, rulebook
    660-664: 'If the Fellowship is not revealed as an effect of the
    Hunt, you may choose to reveal it to reduce the Hunt damage by
    one') - shared by _settle_hunt_damage and continue_hunt_chain
    (GL-1: the two settle paths disagreed).  Offered whenever Gollum
    guides and the Fellowship is hidden; the ONLY bar is a pawn that
    cannot leave an unconquered FP city/stronghold: at progress 0 the
    Fellowship stays where it is (rulebook 3352-3355) and a reveal may
    never END in such a settlement (3401-3405)."""
    guide = state.fellowship.guide
    if guide is None or getattr(guide, 'guide_ability', None) \
            != 'no_reveal_on_damage':
        return False
    if not state.fellowship.hidden:
        return False
    # Almanac p.6: an Eye/special reveal cannot be used as a voluntary reveal
    # to reduce damage, even though the figure is placed after damage settles.
    if getattr(state, '_hunt_reveal_pending', 0) == 2:
        return False
    if state.fellowship.progress > 0 or state.fellowship.mordor_entered:
        return True
    _cur = sanitize(fellowship_region(state))
    _rinfo = board.regions.get(_cur, {})
    _is_stype = _rinfo.get('is_city') or _rinfo.get('is_stronghold')
    _not_conq = _cur not in state.enemy_controlled.get(
        gs.Side.SHADOW, set())
    return not (_is_stype and is_fp_region(_cur) and _not_conq)


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
    protection = getattr(state, 'pending_card_option', None) or {}
    if (protection.get('hunt_reduction')
            or any(key in option for option in protection.get('options', ())
                   for key in ('use_hunt_reduction', 'wizardstaff_prevent', 'mithril_redraw'))):
        # The player must resolve protection before choosing a casualty. The
        # protection executor clears its menu and then resumes this function.
        return
    if _gollum_reveal_available(state):
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


def return_mithril_tile_if_settled(state: gs.GameState) -> bool:
    """Return the rejected tile once the entire action has finished.

    Almanac p.11: further required draws recycle the used pile without this
    tile. Keep it in the existing public Mithril stash while decisions remain;
    that stash is cloned and encoded for both players. No eager pool refill.
    """
    stash = getattr(state, '_mithril_stash', None)
    if not stash or not stash.get('return_pending'):
        return False
    if any(getattr(state, key, None) is not None for key in (
            'pending_combat', 'pending_hunt', 'pending_guide_change',
            'pending_card_option', 'pending_card_followup', 'pending_discard',
            'pending_army_move_die', 'pending_muster_die')):
        return False
    if any(getattr(state, key, None) for key in (
            '_pending_hunt_damage', '_hunt_reveal_pending', '_reveal_place_due', '_pending_reveal_strongholds', '_fellowship_route_event', '_hobbit_place_queue',
            '_overstack_regions', '_wwsat_owed')):
        return False
    state.hunt_pool.append(stash['tile'])
    state._mithril_stash = None
    random.shuffle(state.hunt_pool)
    return True


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
    _can_reveal = _gollum_reveal_available(state)     # GL-1: ONE predicate
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
        g9 = state.fellowship.guide
        _hob9 = (g9 is not None and getattr(g9, 'guide_ability', None)
                 == 'sacrifice_reduce_hunt')
        if not ph.get('casualty_taken') and any(
                c.name != 'Gollum' and c.level > 0
                for c in state.fellowship.companions):
            # GL-5 (1.83 + standing ruling 2026-08-16): a hobbit guide
            # SEPARATES - never a casualty, placement is the learned
            # hobbit_place decision - and that act is the
            # hunt_guide_separate row below; the sacrifice row
            # (immediate placement + casualty_taken) is not his
            if not _hob9:
                out.append(('hunt_sacrifice_guide', {}))
            out.append(('hunt_sacrifice_random', {}))
        if _hob9:
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
        reveal_fellowship_from_hunt(state)
        # 1.229: accumulator is 0 here (consumed at open) - stronghold
        # tiles drawn by this placement land in it and MUST survive to
        # their own decision, never be zeroed with this one.
        # USER 2026-08-16: reveal placement is a LEARNED FP decision - the
        # learned path defers to ai.open_reveal_placement (apply_action tail);
        # the teacher keeps the immediate log-driven placement.
        import combat as _cmbgr9
        if _cmbgr9.FINAL_HITS_HOOK is not None:
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
            # GL-4 (rulebook p.42 new-Guide note, 1.83, standing ruling
            # 2026-08-16): the victim absorbs min(damage, level) and the
            # REMAINDER continues the SAME damage event with the NEW
            # guide's ability live (exactly as hunt_sacrifice_guide).
            # A random draw is the casualty-TAKEN row for every victim,
            # the guide included (WWSAT fires; a hobbit separates
            # outside Mordor, dies on the track) - never the
            # guide-absorption row.
            absorbed = min(int(dmg), int(victim.level))
            if hobbit_hunt_placement_applies(state, victim.name):
                remove_hobbit_for_placement(state, victim, absorbed)
                q = getattr(state, '_hobbit_place_queue', None)
                if q is None:
                    q = state._hobbit_place_queue = []
                q.append(victim)     # learned hobbit_place after the chain
            else:
                _apply_fellowship_damage(state, absorbed,
                                         kill_name=victim.name)
            continue_hunt_chain(state, int(dmg) - absorbed,
                                casualty_taken=True)
            return
        _apply_fellowship_damage(state, dmg)
        _finish_hunt_decision(state)

    elif kind == 'hobbit_place':
        comp_obj = ph.get('companion_obj')
        if comp_obj is None:
            comp_obj = gs.Companion(name=str(ph.get('companion', '')),
                                    level=1, leadership=1)
        target = place_region if place_region is not None \
            else fellowship_region(state)
        target = sanitize(target)
        bound = ph.get('group_destination')
        if bound is not None and target != bound:
            raise ValueError('Companions separating on one Hunt tile must move together')
        # Almanac pp6-7: all Companions separating on this tile move together.
        # Public reveal/route drivers drain this queue before placing the
        # Fellowship or drawing the next tile, so its members share one tile.
        same_tile = bool(getattr(state, '_hobbit_place_queue', None))
        place_separated_companion(state, comp_obj, target)
        _finish_hunt_decision(state)
        if same_tile and (state.pending_hunt or {}).get('phase') == 'hobbit_place':
            state.pending_hunt['group_destination'] = target

    else:
        raise RuntimeError('unknown hunt decision %r' % kind)


def _recycle_hunt_pool(state: gs.GameState) -> int:
    """Refill an empty Hunt Pool with discarded standard tiles only.

    Rulebook p. 40 excludes special blue/red tiles and permanently removed
    tiles at every stage, including Mordor. Return the number recycled.
    """
    if state.hunt_pool:
        return 0
    back = []
    for tile in state.hunt_discard[:]:
        if getattr(tile, 'is_special', False):
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

    # Mithril Coat and Sting (QA cluster G): after the SHADOW draws a hunt tile
    # the FP may discard the table card to force a redraw - a learned FP
    # decision interposed BEFORE the tile applies (before Shelob's die).
    # Learned path only: the teacher replays the humans' own Mithril use.
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

    _apply_hunt_tile(state, tile, successes, mordor=mordor,
                     eye_no_effect=eye_no_effect)
    return tile


def _apply_hunt_tile(state: gs.GameState, tile, successes: int,
                     *, mordor: bool, eye_no_effect: bool = False):
    """Apply a drawn hunt tile (split from _resolve_hunt_tile for the
    Mithril interpose - QA 2026-07-11 cluster G)."""
    # Balrog's Eye has no effect, but Mithril may first replace any Shadow
    # draw (Almanac p11). The source survives both protection continuations.
    if eye_no_effect and tile.is_eye:
        state.hunt_discard.append(tile)
        return
    # Almanac C17: declaration into an originally FP city/stronghold ignores
    # this tile's reveal icon even after Shadow captures that settlement.
    # Derive this from the public source and destination before scheduling
    # any reveal effects; the physical tile retains its printed metadata.
    # A Balrog drawn after a completed reveal cannot reveal again or restart
    # that same route's stronghold history (rulebook p41, Almanac p27).
    ignore_reveal = False
    if eye_no_effect:
        destination = sanitize(fellowship_region(state))
        meta = board.regions.get(destination, {})
        ignore_reveal = (not state.fellowship.hidden or
                         bool(is_fp_region(destination) and
                              (meta.get('is_city') or meta.get('is_stronghold'))))
    dmg = tile.damage
    if tile.is_eye:
        dmg = successes if not mordor else len(state.hunt_box)

    # ACCUMULATE, never assign (W3 BUG-3): several tiles can be drawn before the
    # FP damage decision (Balrog extra draw, stronghold draws) and each tile's
    # damage must survive.  NEGATIVE damage (FP special tiles) is a heal and
    # applies immediately - the accumulator must stay >= 0.
    if dmg < 0:
        state.fellowship.corruption = max(
            0, state.fellowship.corruption + dmg)
    else:
        state._pending_hunt_damage = getattr(
            state, '_pending_hunt_damage', 0) + dmg  # UI/AI reads this

    state.hunt_discard.append(tile)  # drawn = used pile
    stage_hunt_reveal(state, tile, tile.reveal and not ignore_reveal)
    settle_hunt_reveal(state)


def _auto_reveal_place(state: gs.GameState):
    """When fellowship is revealed, move Ring-bearers figure toward Mordor.

    Per rulebook: move from last known position, up to progress steps.
    Cannot end in unconquered FP City/Stronghold (unlike declaration).
    Resets progress to 0.
    """
    # On the Mordor Track, revealing never revisits the old map gateway.
    if state.fellowship.mordor_entered:
        return
    import combat as _cmb_reveal
    learned = _cmb_reveal.FINAL_HITS_HOOK is None
    if learned:
        from fellowship_reveal_route import place, drain
        if (getattr(state, '_fellowship_route_event', None)
                or getattr(state, '_pending_reveal_strongholds', None)):
            drain(state)
            return
        if _reveal_hunt_pending(state) or getattr(state, '_hobbit_place_queue', None):
            state._reveal_place_due = True
            return
    cur = fellowship_region(state)
    progress = state.fellowship.progress
    import board as board_mod

    def _is_sh_stronghold(r):
        meta = board_mod.regions.get(r, {})
        return (meta.get('is_stronghold') and is_shadow_region(r)
                and r not in state.enemy_controlled[
                    gs.Side.FREE_PEOPLES])

    # Teacher log replay retains its historical draw ordering. Self-play
    # places first, then delegates the Balrog/stronghold sequence to the
    # shared route event (Almanac p27; rulebook p41).
    if (_cmb_reveal.FINAL_HITS_HOOK is not None
            or getattr(state, '_sh_tile_drawn_regions', None) is None):
        state._sh_tile_drawn_regions = set()
    _started_in_sh = _is_sh_stronghold(cur)
    if _started_in_sh and not learned:
        _draw_shadow_stronghold_tile(state, cur, "revealed in")
        if _reveal_hunt_pending(state):
            state._reveal_place_due = True
            return

    if progress <= 0:
        if learned:
            place(state, cur)
        return

    # USER 2026-07-20 (P2-04): reveal movement is UP TO Progress - place at
    # exactly Progress when legal, else fall back to shorter distances; never
    # inside an unconquered FP city/stronghold.
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
        for region in fr:
            if region == cur:
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
        if learned:
            place(state, cur)
        else:
            state.fellowship.progress = 0
        return

    # Pick the one closest to Mordor
    mordor_gates = {sanitize("morannon"), sanitize("minas morgul")}

    def _bfs_dist(start, targets):
        # A gateway itself is distance 0 (else the gate scored by its distance
        # to the OTHER gate and adjacent regions always outranked landing
        # on it)
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
    if learned:
        place(state, best)
        return

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
            if _reveal_hunt_pending(state):
                state._reveal_place_due = True
                return

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
    random.shuffle(state.hunt_pool)


def _check_mordor_entry(state: gs.GameState) -> bool:
    """
    Check if Fellowship enters Mordor from Morannon or Minas Morgul.
    Returns True if Mordor was entered this call.
    """
    if state.fellowship.mordor_entered:
        return False
    if getattr(state, '_fellowship_tail_phase', False) and fellowship_tail_pending(state):
        raise RuntimeError('Fellowship effects must finish before Mordor entry/refill')
    if fellowship_region(state) in {"minas morgul", "morannon"}:
        state.fellowship.mordor_entered = True
        state.fellowship.mordor_step = 0
        # Entry ZEROES progress (USER 2026-08-15: the client stops reconciling
        # progress once in Mordor).  Entry does NOT re-hide a revealed
        # Fellowship (USER O.1 2026-07-07, final ruling 2026-08-16).
        state.fellowship.progress = 0
        _restore_hunt_pool_for_mordor(state)
        return True
    return False


def move_fellowship(state: gs.GameState,
                    skip_hunt: bool = False,
                    skip_auto_sacrifice: bool = True) -> dict:
    # skip_auto_sacrifice defaults True (USER 2026-08-16: hunt damage is a
    # LEARNED decision); only an explicit False lets
    # _auto_sacrifice_companion consume it.
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
    # The Fellowship acted this turn (Mordor inactivity exemption; a move before
    # entering Mordor the same turn counts)
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
            # Live/search play opens Balrog after actual reveal placement.
            # Retain the separate log-evidence draw gate for legacy replay.
            import combat as _cmb_bal9
            _balrog = False
            if _cmb_bal9.FINAL_HITS_HOOK is not None:
                from persistent_effects import should_draw_extra_hunt_tile
                _fsp_reg = fellowship_region(state)
                _balrog, _ = should_draw_extra_hunt_tile(state, _fsp_reg, _fsp_reg)
                if _balrog and BALROG_TILE_GATE is not None and not BALROG_TILE_GATE(state):
                    _balrog = False
            if successes:
                _resolve_hunt_tile(state, successes, mordor=False)
            if _balrog:
                # refill an emptied pool before the extra draw ("refill at
                # EVERY draw site")
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
        # Legacy auto-sacrifice runs at the move level (not inside the Balrog
        # branch); `pending` holds every tile's accumulated damage.
        pending = getattr(state, '_pending_hunt_damage', 0)
        if pending > 0 and not skip_auto_sacrifice:
            # Legacy auto-sacrifice path (for GUI/non-AI callers)
            _auto_sacrifice_companion(state, pending)
            state._pending_hunt_damage = 0
            settle_hunt_reveal(state)
        # If skip_auto_sacrifice: damage stays in _pending_hunt_damage
        # for caller (AI) to handle via pending_hunt state

    # Inside Mordor
    else:
        if state.fellowship.mordor_step >= 5:
            result["success"] = False
            result["message"] = "Fellowship already on final step!"
            return result

        # Attempt one step. A Stop tile cancels this attempt; another
        # funded Fellowship movement remains legal in the same turn.
        state.fellowship.mordor_step += 1
        result["message"] = f"Mordor step: {state.fellowship.mordor_step}/5"

        # Draw hunt tile and resolve (Mordor always uses internal resolution)
        tile_result = _resolve_mordor_hunt_tile(
            state, skip_auto_sacrifice=skip_auto_sacrifice)

        import os as _os_flag
        if (tile_result.get("stops_movement")
                and _os_flag.environ.get('MORDOR_STOP_KEEPS_STEP')):
            # ESCAPE HATCH (bisection only): 1.30's keep-step behavior.
            result["message"] = (f"Movement attempt stopped "
                                 f"(step {state.fellowship.mordor_step} kept)")
        elif tile_result.get("stops_movement"):
            # A Stop tile prevents this step without ending later movement.
            state.fellowship.mordor_step -= 1
            result["message"] = (f"Stop tile: step not gained "
                                 f"(stays {state.fellowship.mordor_step}), "
                                 f"movement attempt ended")

        # Check for corruption victory
        if state.fellowship.corruption >= 12:
            result["victory"] = "shadow"
            result["message"] = "Ring-bearers corrupted! Shadow wins!"

        # Check for FP victory (reached step 5) — only when the final
        # step's hunt damage has RESOLVED (see check_ring_victory)
        elif (state.fellowship.mordor_step >= 5
                and state.fellowship.corruption < 12
                and not getattr(state, '_hunt_reveal_pending', 0)
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
    # skip_auto_sacrifice defaults True (same contract as move_fellowship)
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

    # pop() through the funnel so the teacher's _arrange_pool_last hook forces
    # the LOGGED tile
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

    # Apply damage (handle negative damage from FP special tiles).
    # Auto-sacrifice only when explicitly requested (mirrors the
    # outside-Mordor path).
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

    # Mordor Track reveal rule (USER 2026-07-11): tiles DO reveal on the track;
    # Gollum blocks only ordinary damage-tile reveals - Eyes and specials
    # reveal through him.
    stage_hunt_reveal(state, tile, result["reveal"])
    settle_hunt_reveal(state)

    # Move tile to the used pile (drawn tiles are 'used', not removed;
    # eyes + specials return to the pool on Mordor entry — user spec)
    state.hunt_discard.append(tile)

    return result


def _reveal_hunt_pending(state):
    """A reveal cannot proceed through an unfinished Hunt or guide choice."""
    return (state.pending_hunt is not None
            or getattr(state, 'pending_card_option', None) is not None
            or getattr(state, 'pending_guide_change', None) is not None
            or bool(getattr(state, '_hunt_reveal_pending', 0))
            or bool(getattr(state, '_pending_hunt_damage', 0)))


def _apply_shadow_stronghold_tile(state, tile):
    """Extra stronghold tiles ignore Eyes and reveal icons; damage still decides."""
    damage = 0 if tile.is_eye else tile.damage
    if damage < 0:
        state.fellowship.corruption = max(0, state.fellowship.corruption + damage)
    elif damage:
        state._pending_hunt_damage = getattr(state, '_pending_hunt_damage', 0) + damage
    state.hunt_discard.append(tile)
    return {'damage': damage}


def _draw_shadow_stronghold_tile(state: gs.GameState, region: str, reason: str,
                                *, _skip_wizard=False, _resume=False):
    """One Shadow draw per stronghold per reveal, with both protection windows."""
    seen = getattr(state, '_sh_tile_drawn_regions', None)
    if seen is not None and not _resume:
        region_key = sanitize(str(region))
        if region_key in seen:
            return
        seen.add(region_key)
    if not _skip_wizard and _offer_wizard_staff(state, {
            'apply_kind': 'stronghold', 'region': region, 'reason': reason}):
        return
    if not state.hunt_pool:
        _recycle_hunt_pool(state)
    if not state.hunt_pool:
        return
    tile = pick_hunt_tile(state)
    state.hunt_pool.remove(tile)
    try:
        from game_trace import emit_annotation
        emit_annotation(state, 'stronghold_tile_draw', {
            'tile_name': tile.special_name if tile.special_name else repr(tile),
            'region': str(region), 'ctx': reason,
            'damage': 0 if tile.is_eye else tile.damage})
    except Exception:
        pass
    from event_effects import _mithril_interposes, _mithril_stash_card
    if _mithril_interposes(state):
        _mithril_stash_card(state, tile, 'stronghold', {'region': region, 'reason': reason})
        return
    return _apply_shadow_stronghold_tile(state, tile)


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
    • Stacking: 10 units normally, 5 when mustering into a BESIEGED
      stronghold (MAX_SIEGE_UNITS below; rulebook p.31).
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

    # Armies only via get_or_create_army (a raw setdefault can forge a
    # case-variant duplicate entry)
    army = get_or_create_army(state, rkey)

    # FIGURE-POOL guard: a unit may only be mustered if the nation's physical
    # reinforcement pool has one; an exhausted pool returns without placing
    # (matches event_effects.recruit_unit).
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

    # Check stacking limit for units (leaders don't count).  GL-8
    # (rulebook 2870-2871): the 5-cap binds the GARRISON of a besieged
    # stronghold; in the co-located siege entry only *side*'s own units
    # count against its cap (the besiegers outside stay at 10).
    if typ in ("elite", "regular"):
        from combat import (garrison_side_for as _gsf8,
                            side_of_nation as _son8)
        besieged = getattr(state, 'besieged_strongholds', set())
        unit_cap = (MAX_SIEGE_UNITS
                    if rkey in besieged and _gsf8(state, rkey) == side
                    else MAX_ARMY_UNITS)
        _mine8 = sum(1 for u in army.units if _son8(u.nation) == side)
        if _mine8 >= unit_cap:
            raise RuntimeError(
                f"Army stacking limit ({unit_cap} units) reached in {region}")

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

    # USER RULE 2026-07-08: there is NO stronghold condition - the Witch-king
    # musters only with a Sauron army, only while Sauron and at least one FP
    # nation are At War (the Mouth has the all-FP-at-war / Mordor entry).

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

    Requirements (USER final ruling 2026-07-04, amendment 1.3):
    - ALL FIVE Free Peoples nations at war OR fellowship in Mordor
    - Muster in an unconquered Sauron Stronghold (Almanac p.9)
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
        # His entry card names Sauron, not every Shadow nation.
        # Orthanc and Umbar belong to Isengard and Southrons respectively.
        parent_id = board.regions.get(sh, {}).get("parent_id")
        if board.fid_to_name.get(parent_id, "").lower() != "sauron":
            continue
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

    region_san = sanitize(region)
    minion_name_lower = minion_name.lower()

    # GL-7: validate BEFORE the die is consumed (all-or-nothing - a
    # refused muster must leave the die spendable)
    _label7 = {"witch king": "Witch King", "saruman": "Saruman",
               "mouth of sauron": "Mouth of Sauron",
               "mouth": "Mouth of Sauron"}.get(minion_name_lower)
    if _label7 is None:
        raise RuntimeError(f"Unknown minion: {minion_name}")
    _check7 = {"Witch King": can_muster_witch_king,
               "Saruman": can_muster_saruman,
               "Mouth of Sauron": can_muster_mouth_of_sauron}[_label7]
    can, reason, valid = _check7(state)
    if not can:
        raise RuntimeError(f"Cannot muster {_label7}: {reason}")
    if region_san not in [sanitize(v) for v in valid]:
        raise RuntimeError(f"Invalid muster location for {_label7}")
    _pick_die(state, gs.Side.SHADOW, wanted, die_idx)

    if minion_name_lower == "witch king":
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
        state.saruman.in_play = True
        state.saruman.was_ever_in_play = True
        _add_minion_to_region(state, state.saruman, region_san)

    elif minion_name_lower in ["mouth of sauron", "mouth"]:
        state.mouth_of_sauron.in_play = True
        state.mouth_of_sauron.was_ever_in_play = True
        _add_minion_to_region(state, state.mouth_of_sauron, region_san)


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
    - At least one Shadow minion IS OR HAS BEEN in play (Saruman,
      Witch-King, or Mouth of Sauron - was_ever_in_play, even if
      eliminated; NOT the stricter in_play)
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

    # GL-7: validate BEFORE the die is consumed (all-or-nothing)
    can, reason, valid = can_transform_gandalf_white(state)
    if not can:
        raise RuntimeError(f"Cannot transform Gandalf: {reason}")

    region_san = sanitize(region)
    if region_san not in [sanitize(v) for v in valid]:
        raise RuntimeError("Invalid region for Gandalf the White")
    _pick_die(state, gs.Side.FREE_PEOPLES, wanted, die_idx)

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

    # 1.126: Gandalf the Grey has left play - requires_companion FP table cards
    # keyed on Grey (Wizard's Staff) discard immediately.
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

    # Strider's card (marker p.4): only a Strider NOT in the Fellowship can be
    # crowned - he must separate onto the map first (audit A12).
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

    # GL-7: validate BEFORE the die is consumed (all-or-nothing)
    can, reason, valid = can_crown_aragorn(state)
    if not can:
        raise RuntimeError(f"Cannot crown Aragorn: {reason}")
    _pick_die(state, gs.Side.FREE_PEOPLES, wanted, die_idx)

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

    # "Both ... lose the additional die if the corresponding Character is
    # eliminated" (marker p.18, audit A14): the FP character must still be on
    # the map or in the Fellowship for the die to count.
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


_VP_TABLES = None   # memo: (fp_settlements, sh_settlements)


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

    # hoisted out of the loop; `x in ()` is False, matching an absent side
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
        if (state.fellowship.corruption < 12
                and not getattr(state, '_hunt_reveal_pending', 0)
                and not getattr(state, '_pending_hunt_damage', 0)
                and state.pending_hunt is None
                and getattr(state, '_mithril_stash', None) is None
                and getattr(state, '_wizardstaff_stash', None) is None):
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

    # CARD CONSERVATION (2026-08-15): route through discard_hand_card so the
    # card reaches the face-down pile and the hand-limit pending is re-derived
    # (the GUI's Discard button IS the human's learned pick).
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


# Pure rules helper: legal piece-add headroom for a nation.
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
