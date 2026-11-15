"""Exact enumeration of the chance outcomes of applying one action.

The one-ply scoring used to estimate the value of a chance-triggering move
from a few sampled clones. This module instead walks the REAL engine apply
under a fork driver: every time the engine reaches a random draw the driver
either follows a planned option or records the options and aborts the walk,
so repeating the walk for every option enumerates all outcomes with their
exact probabilities:

    value(move) = sum over outcomes c of P(c) * V(state after c)

What is enumerated (each as ONE draw, taking the statistic that alone
reaches the successor state):
  * hunt roll: the TOTAL number of successes (dice + re-rolls, closed form);
  * hunt tile / stronghold tile: one option per tile KIND with probability
    count / pool size, read from the pool multiset and never from its order
    (the order is hidden information);
  * Shelob's die, random companion, random card: uniform options;
  * a card drawn by the MOVING side from its own deck: one option per distinct
    card of that deck (the mover knows its deck's content, never its order);
    a draw by the other side stays one sample of the resampled hidden world
    (its identity never reaches the mover's observation);
  * a combat round with no active card and no Confusion: each side's TOTAL
    hits (roll + leader re-roll, closed form).
  * event-card dice (ctx 'card effect', up to 3 dice): one option per
    MULTISET of faces. Every such site in event_effects.py reads the faces
    only through a count (faces >= a threshold, or == 6), so the order of
    the dice never reaches the successor state.
Rounds with an active combat card, paused rounds, larger dice batches and
any raw RNG use raise Unsupported; expand() then asks the engine's own exact
combat kernel (carded rounds) and samples only when that declines too.

RESOLVING A BATTLE (resolve()). Chance does not always sit inside the apply
of the move that causes it: an attack only OPENS a battle, the dice are
rolled when the last combat card is committed, usually by the other player.
Scoring such a move on the state right after it asks the value function to
price an unresolved battle, and that is what the Shadow's decisions mostly
are (measured 2026-10-04 on 220,229 Shadow decisions: the credit for WHICH
attack to make had validity +0.02 +- .37 while "attack or not" had +0.80).
resolve() therefore keeps walking while a battle waits for its dice: the
pre-roll decisions (withdraw into the siege, combat cards, retreat window)
are taken by the policies (the mover's own policy for its decisions, its
model of the opponent on the resampled hidden hand for the others), the
round's dice are enumerated, and the value is taken where the round has
been rolled or the battle is over. The same holds for the Shadow's eye
allocation, which is followed at once by the action-dice roll of both
sides: its value is the mean over sampled rolls (the outcome space is far
too large to enumerate; every candidate of the decision sees the same
sampled dice), not the value of the state before the roll.

The patches are process-global module attributes (like the live bridge's):
safe in the single-threaded collection workers, not across threads. Walks
run on clones and consume no RNG, so the result does not depend on a seed.

Derived from the probe verified on 2026-10-04 (scratch chance_probe.py):
every enumerated leaf re-forced in a fresh clone reproduced the same state
(124 of 124), no sampled state fell outside the enumerated set, and
sum P*V matched 3,000 sampled clones within sampling error for hunts
(2-4 dice, re-roll sources, a forced second tile on a gateway, the Mordor
track) and for plain combat rounds.
"""
import collections
import contextlib
from copy import deepcopy
import itertools
import math
import random

import numpy as np

_ENGINE = None
_ACTIVE = None


class Fork(BaseException):
    """BaseException: the engine's own `except Exception` blocks must not swallow it."""


class Unsupported(Exception):
    """This apply holds chance the enumerator does not model: sample instead."""


class ChanceDriver:
    def __init__(self, plan=(), *, side=None, tag=None):
        self.plan = tuple(plan)
        self.side, self.tag = side, tag      # the moving side (engine enum, "fp"/"sh"): own draws are enumerated
        self.ptr = 0
        self.prob = 1.0
        self.pending = None          # (kind, options) of the first unplanned draw
        self.labels = []
        self.gl_script = []          # scripted game_logic d6 batches [(ctx, faces)]
        self.cb_script = []          # scripted combat d6 batches [(ctx, faces)]

    def choose(self, kind, options):
        """options: [(label, value, probability)] in an order that does not depend on the clone."""
        if self.pending is not None:      # engine upkeep running while a Fork unwinds
            raise Fork()
        if self.ptr < len(self.plan):
            label, value, p = options[self.plan[self.ptr]]
            self.ptr += 1
            self.prob *= p
            self.labels.append((kind, label))
            return value
        self.pending = (kind, options)
        raise Fork()


def _engine():
    """Engine modules and the original funnels (bound once, after src is importable)."""
    global _ENGINE
    if _ENGINE is None:
        import combat as cb
        import combat_dist as cd
        import combat_round_resume as crr
        import game_logic as gl
        import hunt_dist as hd
        originals = dict(hunt_roll=gl._hunt_roll, gl_roll=gl.roll_d6_batch, pop=gl.pop_hunt_tile, pick=gl.pick_hunt_tile,
                         companion=gl.pick_random_companion, card=gl.pick_random_card, cb_roll=cb.roll_d6_batch,
                         primary=crr._primary_side, shuffle=random.shuffle, draw=gl.draw_event_card)
        _ENGINE = dict(gl=gl, cb=cb, cd=cd, crr=crr, hd=hd, original=originals)
    return _ENGINE


def _binom(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def total_hits_pmf(n, p_roll, rerolls, p_reroll):
    """P(total hits) of n dice where up to `rerolls` misses are rolled again once."""
    out = [0.0] * (n + 1)
    for h in range(n + 1):
        k = min(rerolls, n - h)
        for h2 in range(k + 1):
            out[h + h2] += _binom(n, h, p_roll) * _binom(k, h2, p_reroll)
    return out


def _p_hunt_roll(state, num_dice, thresh, fp_modifier=0):
    e, drv = _engine(), _ACTIVE
    gl, original = e["gl"], e["original"]["hunt_roll"]
    if drv is None or num_dice <= 0:
        return original(state, num_dice, thresh, fp_modifier)
    mod = fp_modifier + int(getattr(state, "_crebain_hunt_bonus", 0) or 0)
    faces = sum(1 for r in range(2, 7) if r + mod >= thresh)       # a natural 1 never hits
    sources = (int(bool(gl._nazgul_present(state))) + int(bool(gl._shadow_army_present(state)))
               + int(bool(gl._in_shadow_stronghold(state))))
    pmf = total_hits_pmf(num_dice, faces / 6.0, sources, faces / 6.0)
    h = drv.choose("hunt_successes", [(k, k, p) for k, p in enumerate(pmf) if p > 0.0])
    # h hits on the first roll and every re-roll misses: exactly h successes.
    drv.gl_script = [("hunt roll", [6] * h + [1] * (num_dice - h))]
    drv.gl_script += [("hunt reroll", [1])] * min(sources, num_dice - h)
    out = original(state, num_dice, thresh, fp_modifier)
    if out != h or drv.gl_script:
        raise RuntimeError("hunt script mismatch: engine %r, forced %r, left %r" % (out, h, drv.gl_script))
    return out


def _p_gl_roll(n, ctx="", side=None):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["gl_roll"](n, ctx, side)
    if drv.gl_script:
        want, faces = drv.gl_script[0]
        if want != ctx or len(faces) != n:
            raise RuntimeError("hunt script mismatch at %r n=%d (expected %r n=%d)" % (ctx, n, want, len(faces)))
        drv.gl_script.pop(0)
        return list(faces)
    if ctx == "shelob d6 damage" and n == 1:
        return [drv.choose("shelob_d6", [(v, v, 1.0 / 6.0) for v in range(1, 7)])]
    if ctx == "card effect" and 1 <= n <= 3:
        return list(drv.choose("card_dice", card_dice_options(n)))
    raise Unsupported("game_logic.roll_d6_batch ctx=%r n=%d" % (ctx, n))


def card_dice_options(n):
    """One option per multiset of n faces with its multinomial probability."""
    options = []
    for faces in itertools.combinations_with_replacement(range(1, 7), n):
        ways = math.factorial(n)
        for count in collections.Counter(faces).values():
            ways //= math.factorial(count)
        options.append((faces, faces, ways / float(6 ** n)))
    return options


def tile_options(pool):
    """One option per tile kind, probability count / size: the pool's ORDER is never read."""
    hd = _engine()["hd"]
    first, count = {}, collections.Counter()
    for i, tile in enumerate(pool):
        key = hd.tile_key(tile)
        count[key] += 1
        first.setdefault(key, i)
    return [(key, first[key], count[key] / float(len(pool))) for key in sorted(count)]


def _p_pop(state):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["pop"](state)
    return state.hunt_pool.pop(drv.choose("tile", tile_options(state.hunt_pool)))


def _p_pick(state):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["pick"](state)
    return state.hunt_pool[drv.choose("tile", tile_options(state.hunt_pool))]


def _p_companion(candidates, ctx=""):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["companion"](candidates, ctx)
    n = len(candidates)
    return candidates[drv.choose("companion", [(getattr(c, "name", str(c)), i, 1.0 / n) for i, c in enumerate(candidates)])]


def _p_card(candidates, ctx=""):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["card"](candidates, ctx)
    n = len(candidates)
    return candidates[drv.choose("card", [(str(c), i, 1.0 / n) for i, c in enumerate(candidates)])]


def _p_draw(state, side, deck):
    drv = _ACTIVE
    original = _engine()["original"]["draw"]
    if drv is None or drv.tag is None or side != drv.side:
        return original(state, side, deck)
    cards = getattr(state, drv.tag + ("_char_deck" if deck == "character" else "_strat_deck"), None)
    if cards and len(cards) > 1 and all(isinstance(c, str) and not c.startswith("_unknown_") for c in cards):
        count = collections.Counter(cards)
        size = float(len(cards))
        card = drv.choose("draw", [(c, c, count[c] / size) for c in sorted(count)])
        cards.append(cards.pop(len(cards) - 1 - cards[::-1].index(card)))      # the engine draws the last card
    return original(state, side, deck)


def plain_round(combat_state):
    """No active (uncancelled) combat card on either side and no Confusion:
    the successor depends on each side's TOTAL hits only."""
    crr = _engine()["crr"]
    return (crr._name(combat_state, True) == "" and crr._name(combat_state, False) == ""
            and not combat_state.attacker_confusion and not combat_state.defender_confusion)


def side_roll_parameters(c, attacker):
    e = _engine()
    cb, cd = e["cb"], e["cd"]
    role, short = ("attacker", "atk") if attacker else ("defender", "def")
    cancelled = getattr(c, "_wop_cancelled", None)
    cancelled_atk = getattr(c, "_wop_cancelled_is_attacker", False)
    strength = cb.calc_combat_strength(getattr(c, role + "_army"),
                                       cancelled_captain=cancelled if cancelled_atk == attacker else None)
    strength -= getattr(c, role + "_dice_reduction", 0)
    cap = getattr(c, role + "_dice_cap")
    n = max(1, min(strength, cap) if cap is not None else strength)
    threshold = getattr(c, "_resume_" + short + "_threshold")
    leadership = 0 if getattr(c, role + "_reroll_cancelled") else max(0, getattr(c, role + "_leadership"))
    p_roll = cd._hit_faces(threshold, getattr(c, role + "_dice_mod")) / 6.0
    p_reroll = cd._hit_faces(threshold, getattr(c, role + "_reroll_mod")) / 6.0
    return n, p_roll, min(5, leadership), p_reroll


def _p_primary_side(cur, attacker):
    drv = _ACTIVE
    original = _engine()["original"]["primary"]
    role = "attacker" if attacker else "defender"
    if drv is None or ("combat", role) in cur.observed_roll_sides:
        return original(cur, attacker)
    c = cur.combat
    if not plain_round(c):
        raise Unsupported("combat round with an active card or Confusion")
    n, p_roll, rerolls, p_reroll = side_roll_parameters(c, attacker)
    pmf = total_hits_pmf(n, p_roll, rerolls, p_reroll)
    h = drv.choose(role + "_hits", [(k, k, p) for k, p in enumerate(pmf) if p > 0.0])
    k = min(rerolls, n - h)
    drv.cb_script.append(("combat roll", [6] * h + [1] * (n - h)))           # a 6 always hits, a 1 never
    if k > 0:
        drv.cb_script.append(("leader reroll", [1] * k))
    return original(cur, attacker)


def _p_cb_roll(n, ctx="", side=None):
    drv = _ACTIVE
    if drv is None:
        return _engine()["original"]["cb_roll"](n, ctx, side=side)
    if drv.cb_script:
        want, faces = drv.cb_script[0]
        if want != ctx or len(faces) != n:
            raise RuntimeError("combat script mismatch at %r n=%d (expected %r n=%d)" % (ctx, n, want, len(faces)))
        drv.cb_script.pop(0)
        return list(faces)
    raise Unsupported("combat.roll_d6_batch ctx=%r n=%d" % (ctx, n))


@contextlib.contextmanager
def driven(driver):
    """Install the fork driver (process-global patches, single-threaded use). Always restored."""
    global _ACTIVE
    e = _engine()
    gl, cb, crr, original = e["gl"], e["cb"], e["crr"], e["original"]
    if _ACTIVE is not None:
        raise RuntimeError("chance driver is not re-entrant")
    _ACTIVE = driver
    gl._hunt_roll, gl.roll_d6_batch = _p_hunt_roll, _p_gl_roll
    gl.pop_hunt_tile, gl.pick_hunt_tile = _p_pop, _p_pick
    gl.pick_random_companion, gl.pick_random_card = _p_companion, _p_card
    cb.roll_d6_batch, crr._primary_side = _p_cb_roll, _p_primary_side
    gl.draw_event_card = _p_draw
    random.shuffle = lambda seq, *a, **k: None       # pool order is irrelevant once tiles are picked by kind
    try:
        yield driver
    finally:
        _ACTIVE = None
        gl._hunt_roll, gl.roll_d6_batch = original["hunt_roll"], original["gl_roll"]
        gl.pop_hunt_tile, gl.pick_hunt_tile = original["pop"], original["pick"]
        gl.pick_random_companion, gl.pick_random_card = original["companion"], original["card"]
        cb.roll_d6_batch, crr._primary_side = original["cb_roll"], original["primary"]
        gl.draw_event_card = original["draw"]
        random.shuffle = original["shuffle"]


def private_frame(frame, choice):
    """A fresh copy of the CHOSEN action per hypothetical world (never alias the live menu)."""
    actions = list(frame.actions)
    a = actions[choice]
    actions[choice] = type(a)(a.die_idx, a.action_type, deepcopy(a.params))
    return type(frame)(frame.side, frame.phase, frame.turn, tuple(actions), frame.inputs, frame.fixed_reason)


def _rng_state(world):
    py, npst = world.rng.python, world.rng.numpy
    return py, (npst[0], npst[1].tobytes(), npst[2], npst[3], npst[4])


def apply_forced(world, frame, choice, plan):
    """Realise ONE chance outcome (a plan returned by enumerate_chance) in `world`."""
    driver = ChanceDriver(plan, side=world.sides[frame.side], tag=frame.side)
    before = _rng_state(world)
    with driven(driver):
        world.apply(frame, choice)
    if driver.ptr != len(driver.plan) or driver.gl_script or driver.cb_script:
        raise RuntimeError("plan not fully consumed")
    if _rng_state(world) != before:
        raise Unsupported("raw RNG consumed outside the funnels")
    return driver


# Phases of a pending battle in which the dice of the coming round have not
# been rolled yet and a decision is awaited before they are.
PRE_ROLL_PHASES = frozenset({"attacker_siege_decision", "defender_siege_choice", "fp_shine", "attacker_combat_card",
                             "defender_combat_card", "defender_retreat"})


def battle_marker(world):
    """(waits for its dice?, rounds fought) of the pending battle; (False, -1) without one."""
    pending = getattr(world.state, "pending_combat", None)
    if not pending or world.end is not None:
        return False, -1
    return pending.get("phase") in PRE_ROLL_PHASES, int(pending.get("rounds_fought", 0) or 0)


def kernel_outcomes(world, frame, choice):
    """Exact outcomes of one combat-card commitment that resolves a round the
    fork driver does not model (an active combat card), from the engine's own
    exact combat kernel (ai._kernel_expand_combat, the production search's
    chance node). `world` must already hold the resampled hidden information.
    Returns [(probability, successor world)] or None when the kernel declines
    (adaptive cards, a paused round) or the action is not a kernel parent."""
    from .search import clone_environment
    ai = world.ai
    action = private_frame(frame, choice).actions[choice]
    if action.action_type not in ai._KERNEL_PARENTS:
        return None
    side = world.sides[frame.side]
    parent = ai.MCTSNode(state=world.state, side=side)
    node = ai.MCTSNode(state=None, side=side, parent=parent, action=action, prior=1.0)
    saved, saved_numpy, cap = random.getstate(), np.random.get_state(), ai.CHANCE_K_MAX
    ai.set_search_limits(chance_k=0)                      # keep the whole support
    try:
        probe, used = ai._apply_action_detect_rng(world.state, action)
        if not used or not ai._kernel_expand_combat(node, probe):
            return None
        outcomes = [(float(branch.prior), branch.state) for branch in node.outcomes]
    except Exception:
        return None
    finally:
        ai.set_search_limits(chance_k=cap)
        random.setstate(saved)
        np.random.set_state(saved_numpy)
    total = sum(p for p, _ in outcomes)
    if not outcomes or abs(total - 1.0) > 1e-6 or any(state is None for _, state in outcomes):
        return None
    leaves = []
    for probability, state in outcomes:
        leaf = clone_environment(world, 0, extra_transitions=4, state=state)
        leaf.committed = world.committed + 1
        # The kernel built the state, not leaf.apply(): run apply()'s stage/seat
        # bookkeeping, or a leaf whose battle ended keeps the parent's seat (the
        # attacker) and prepare() hands it a second action (fix wave S1-01).
        with leaf._engine_scope():
            leaf.after_action(frame, action)
        leaves.append((probability / total, leaf))
    return leaves


def expand(env, frame, choice, *, determinize=None, max_walks=96, samples=8, seed=0):
    """One apply as weighted successor worlds.

    Exact by the fork driver; else exact by the engine's combat kernel; else
    `samples` sampled clones (common seeds: sample k of every candidate of a
    decision uses the same clone seed). Returns ([(probability, world)], how)
    with how in "exact", "kernel", "sampled"."""
    from .search import clone_environment
    try:
        leaves = enumerate_chance(env, frame, choice, max_walks=max_walks, determinize=determinize)
        return [(p, world) for p, _, _, world in leaves], "exact"
    except Unsupported:
        pass
    base = env
    if determinize is not None:
        base = clone_environment(env, 0, extra_transitions=4)
        determinize(base)
    leaves = kernel_outcomes(base, frame, choice)
    if leaves is not None:
        return leaves, "kernel"
    leaves = []
    for k in range(samples):
        world = clone_environment(base, (int(seed) + 0x9E3779B97F4A7C15 * (k + 1)) % (2 ** 62), extra_transitions=4)
        world.apply(private_frame(frame, choice), choice)
        leaves.append((1.0 / samples, world))
    return leaves, "sampled"


def roll_outcomes(world, samples, seed):
    """The action-dice roll that follows the eye allocation: `samples` equally
    weighted worlds. Sample k uses the same private stream for every
    candidate, so two allocations are compared on the same dice."""
    from .search import clone_environment
    out = []
    for k in range(samples):
        rolled = clone_environment(world, (int(seed) + 0xD1B54A32D192ED03 * (k + 1)) % (2 ** 62), extra_transitions=4)
        with rolled._engine_scope():
            rolled._roll()
        out.append((1.0 / samples, rolled))
    return out


def resolve(env, frame, choice, *, determinize=None, policies=None, max_walks=96, samples=8, seed=0, max_steps=6,
            max_leaves=96, roll_samples=0):
    """The weighted states at which frame.actions[choice] is valued
    (_resolve_battle), followed by the pending action-dice roll when
    roll_samples > 0 and the move was the eye allocation."""
    leaves, counts = _resolve_battle(env, frame, choice, determinize=determinize, policies=policies, max_walks=max_walks,
                                     samples=samples, seed=seed, max_steps=max_steps, max_leaves=max_leaves)
    if roll_samples > 0 and any(world.end is None and world.stage == "post_allocation" for _, world in leaves):
        rolled = []
        for probability, world in leaves:
            if world.end is None and world.stage == "post_allocation":
                rolled.extend((probability * share, child) for share, child in roll_outcomes(world, roll_samples, seed))
                counts["rolls"] += 1
            else:
                rolled.append((probability, world))
        leaves = rolled
    return leaves, counts


def lookahead(leaves, policies, side, *, decisions, max_walks=96, samples=8, seed=0, max_leaves=96, max_rounds=None):
    """Walk every valued state on through the next `decisions` decisions of the OTHER side.

    Decisions of either side on the way are taken by that side's policy (its most probable move; a forced
    decision needs no policy) and expanded like a move in _resolve_battle: exactly by the fork driver or the
    combat kernel; a sampled expansion ends that walk (its samples are kept, nothing is compounded on them).
    A walk also ends at the end of the game, at a decision of a side without a policy, or when the leaf cap is
    reached. The point: a move that leaves something open is valued after the opponent has had the chance to
    take it. Returns ([(probability, world)], counts)."""
    counts = collections.Counter()
    other = "fp" if side == "sh" else "sh"
    final, frontier = [], [(probability, world, 0) for probability, world in leaves]
    rounds = 0
    while frontier and rounds < (max_rounds or 8 * decisions):
        rounds += 1
        following = []
        for probability, world, taken in frontier:
            if taken >= decisions or world.end is not None or len(final) + len(frontier) + len(following) > max_leaves:
                final.append((probability, world))
                continue
            try:
                awaited = world.prepare()
            except Exception:
                awaited = None
            if awaited is None or (len(awaited.actions) > 1 and policies.get(awaited.side) is None):
                final.append((probability, world))
                continue
            move = 0 if len(awaited.actions) == 1 else int(np.argmax(policies[awaited.side].predict(awaited.inputs).log_probabilities))
            children, how = expand(world, awaited, move, max_walks=max_walks, samples=samples, seed=seed)
            counts[how] += 1
            counts["decisions"] += 1
            advanced = taken + (1 if awaited.side == other and len(awaited.actions) > 1 else 0)
            if how == "sampled":
                final.extend((probability * share, child) for share, child in children)
                continue
            following.extend((probability * share, child, advanced) for share, child in children)
        frontier = following
    return final + [(probability, world) for probability, world, _ in frontier], counts


def _resolve_battle(env, frame, choice, *, determinize, policies, max_walks, samples, seed, max_steps, max_leaves):
    """The weighted states at which frame.actions[choice] is valued.

    The apply itself is expanded (expand()). While a successor holds a battle
    that waits for its dice, the awaited pre-roll decision is taken by
    policies[side] (its most probable action; a forced decision needs no
    policy) and expanded in turn, until the round has been rolled, the battle
    is over, a decision of a side without a policy is reached, or the limits
    are hit. policies=None values every move right after its own apply.
    Returns ([(probability, world)], counts) with counts = {"exact": n,
    "kernel": n, "sampled": n, "decisions": n} over the expansions made."""
    counts = collections.Counter()
    leaves, how = expand(env, frame, choice, determinize=determinize, max_walks=max_walks, samples=samples, seed=seed)
    counts[how] += 1
    if not policies or how != "exact":
        # Kernel successors are bare engine states and sampled ones would
        # compound sampling: neither is walked further.
        return leaves, counts
    final, frontier = [], leaves
    for _ in range(max_steps):
        following = []
        for probability, world in frontier:
            waits, rounds = battle_marker(world)
            if not waits or len(final) + len(frontier) + len(following) > max_leaves:
                final.append((probability, world))
                continue
            try:
                awaited = world.prepare()
            except Exception:
                awaited = None
            if awaited is None or (len(awaited.actions) > 1 and policies.get(awaited.side) is None):
                final.append((probability, world))
                continue
            if len(awaited.actions) == 1:
                move = 0
            else:
                move = int(np.argmax(policies[awaited.side].predict(awaited.inputs).log_probabilities))
            children, how = expand(world, awaited, move, max_walks=max_walks, samples=samples, seed=seed)
            counts[how] += 1
            counts["decisions"] += 1
            for share, child in children:
                still_waits, child_rounds = battle_marker(child)
                rolled = (not still_waits) or child_rounds != rounds or how != "exact"
                (final if rolled else following).append((probability * share, child))
        frontier = following
        if not frontier:
            break
    return final + frontier, counts


def enumerate_chance(env, frame, choice, *, max_walks=96, determinize=None):
    """Every chance outcome of applying frame.actions[choice] in `env`.

    Returns [(probability, plan, labels, world)], probabilities summing to 1,
    world = the successor environment. `determinize(world)` is applied to each
    fresh clone and must be the SAME deterministic function for every walk (one
    hidden world per enumeration, so plan indices stay valid). Raises
    Unsupported when the apply holds chance the driver does not model or needs
    more than max_walks walks; the live env and its RNG are never touched."""
    from .search import clone_environment
    leaves, stack, walks = [], [()], 0
    while stack:
        plan = stack.pop()
        walks += 1
        if walks > max_walks:
            raise Unsupported("walk budget exceeded")
        world = clone_environment(env, 0, extra_transitions=4)
        if determinize is not None:
            determinize(world)
        driver = ChanceDriver(plan, side=world.sides[frame.side], tag=frame.side)
        before = _rng_state(world)
        try:
            with driven(driver):
                world.apply(private_frame(frame, choice), choice)
        except Unsupported:
            raise
        except BaseException:
            # A Fork, or an engine error raised by the action's finally-upkeep while the
            # Fork was unwinding through a half-applied state: both mean "fork here".
            if driver.pending is None:
                raise
            for i in reversed(range(len(driver.pending[1]))):
                stack.append(plan + (i,))
            continue
        if driver.gl_script or driver.cb_script:
            raise RuntimeError("scripted dice not consumed")
        if _rng_state(world) != before:
            raise Unsupported("raw RNG consumed outside the funnels")
        leaves.append((driver.prob, plan, tuple(driver.labels), world))
    total = sum(p for p, _, _, _ in leaves)
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError("probability mass %.12f" % total)
    return leaves
