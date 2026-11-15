"""Gumbel-style root search that produces a per-decision policy target.

The search is the learning signal, not a strength setting. A small budget
(tens of simulations) over a Gumbel-sampled subset of the full legal menu
yields an improved policy over the WHOLE menu (completed-Q softmax), plus a
mixed root value. Both are recorded as supervised targets for the learner.

Semantics:
  * The full legal menu is scored by the prior; nothing is capped. Only the
    tree expansion is restricted to the k Gumbel-sampled root actions.
  * Simulations run on cloned engine states with a private RNG, so chance
    (dice, hunt tiles, draws) is sampled, never read from the live game.
  * Opponent decisions inside the tree are sampled from a frozen opponent
    MODEL (the public human-imitation policy), never the live opponent's
    weights. Own decisions inside the tree form the open-loop tree.
  * The live game's RNG streams and state are never touched by a search.
  * Optional determinization resamples the opponent's hidden hand, both
    deck orders and the hunt pool in the clone before each simulation.

References: Danihelka et al. 2022, "Policy improvement by planning with
Gumbel" (Sequential Halving + completed Q-values), and the mctx reference
qtransform_completed_by_mix_value defaults (value_scale .1, maxvisit_init 50).
"""
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
import math
import os
import time

import numpy as np
import torch

from .environment import GameRandom, PythonEnvironment


@dataclass(frozen=True)
class SearchConfig:
    # "tree": Gumbel Sequential Halving over sampled continuations (opponent
    #   model + chance inside the tree). Measured: at 32-128 simulations the
    #   target's deviation from the prior equals its deviation between two
    #   search seeds, i.e. the target is leaf-sampling noise.
    # "afterstate": one-ply TD-Gammon style. Each Gumbel-sampled root action
    #   is applied on a clone and the DETERMINISTIC afterstate (before any
    #   chance, reply or upkeep) is scored by the own critic (+/-1 if it ends
    #   the game). Reproducible by construction; chance-triggering actions
    #   (combat, hunt) average chance_samples clones.
    mode: str = "afterstate"
    simulations: int = 32
    considered_actions: int = 16
    max_own_depth: int = 3
    max_simulation_steps: int = 300
    determinize: bool = True
    chance_samples: int = 4
    # Afterstate mode: how the value of a move whose apply holds chance is
    # computed. "sample" = the average of chance_samples sampled clones (only
    # for candidates flagged combat/hunt; one clone otherwise). "exact" = the
    # engine apply is walked under a fork driver (chance.py) and the value is
    #     sum over outcomes of P(outcome) * V(state after the outcome),
    # for EVERY candidate, flagged or not: hunt successes x tile kinds from
    # the remaining pool, plain combat rounds, random companions/cards, own
    # card draws. A candidate without chance costs one apply; chance the
    # driver does not model (event dice, rounds with an active combat card)
    # or more than chance_walks walks falls back to chance_samples clones.
    chance: str = "sample"
    chance_walks: int = 96
    # chance="exact" only. "pending": a move whose chance is still PENDING
    # when its own apply ends is valued where that chance has been resolved.
    # (1) A move that leaves a battle waiting for its dice (an attack, a
    # combat card committed first, "continue", "do not retreat") is valued
    # where that round has been ROLLED: the pre-roll decisions in between are
    # taken by the policies (own policy; the opponent model on the resampled
    # hidden hand) and the dice are enumerated (chance.resolve). (2) The eye
    # allocation is valued as the mean over roll_samples sampled action-dice
    # rolls (the same dice for every candidate). "none": every move is valued
    # right after its own apply. Measured 2026-10-04 on one game: for the
    # Shadow's attack candidates the two values differ by sd 0.08 and rank the
    # attacks of a decision differently (correlation -0.25).
    resolve: str = "none"
    roll_samples: int = 16
    # Afterstate mode: trust factors ((kind of move, weight), ..., ("*",
    # default)) from credit_validity.py --trust-output. An evaluated
    # candidate keeps weight * (Q - sibling mix); one that ends the game keeps
    # it all. Empty = full credit for everything. Only the target and the move
    # are affected; the recorded Q stays raw.
    credit_gate: tuple = ()
    # Afterstate mode: calibrated credit ((kind, within, between), ..., ("*",
    # within, between)) from validity_bench.py calibrate. The credit of an
    # evaluated candidate becomes
    #   within[kind] * (Q - mean Q of the evaluated candidates of its kind)
    # + between[kind] * (that mean - mean Q of all evaluated candidates)
    # (prior-weighted means): what the value function's opinion of a move has
    # been MEASURED to be worth on held-out games, per kind of move, instead
    # of taking every opinion at face value. Takes precedence over
    # credit_gate. The recorded Q stays raw.
    credit_calibration: tuple = ()
    # Multiplies sigma when credit_calibration is used. A calibrated credit is
    # worth what it says (and is small), so it supports a larger step than the
    # raw credit; validity_bench.py calibrate measures the gain per multiple.
    credit_scale: float = 1.
    # Afterstate mode: this many additional legal actions sampled UNIFORMLY
    # (not from the prior) are evaluated too, so moves the prior underrates
    # can enter the target when the critic sees a large edge. 0 = none.
    exploration_candidates: int = 0
    value_scale: float = .1
    maxvisit_init: float = 50.
    # mctx's default min-max rescaling stretches ANY sibling Q spread to the
    # full [0,1] range, so a 0.02 noise spread from a weak critic sharpens the
    # target as much as a decisive 1.0 spread (measured: 5.8 nats at every
    # turn in generation 1). Off: sigma = visit scale * value_scale * Q with
    # Q in [-1,1], so target sharpness is proportional to the evidence.
    rescale_values: bool = False
    # Engine rules notices (e.g. zero-effect card plays) print per call;
    # hypothetical worlds would repeat them dozens of times per decision.
    quiet_simulations: bool = True
    # Afterstate mode, play/evaluation only: the move played is the argmax of
    # prior logit + sigma(Q) over the evaluated actions (no Gumbel noise in
    # the final choice), i.e. the search policy's best evaluated move rather
    # than a sample from it. The CANDIDATE set stays Gumbel-sampled: measured
    # 2026-10-03, restricting candidates to the prior's top 24 loses most of
    # the gain (FP 58/128 vs 80/128), because the critic's best corrections
    # are moves the prior ranks low. Training collection keeps the noise.
    greedy: bool = False
    # Afterstate mode: which actions are evaluated. "gumbel" = a Gumbel
    # sample without replacement from the prior (training collection; also
    # brings low-prior actions in). "prior" = the prior's top
    # considered_actions, a trust region: the critic may only re-rank moves
    # the prior already rates highly. The one regime-valid gain measured so
    # far used "prior" with greedy selection (FP 110/256 vs 88/256, p=.033).
    candidates: str = "gumbel"
    # Afterstate mode: how the PLAYED move is chosen. "improved" samples from
    # prior x exp(credit), so the value function steers play. "prior" samples
    # from the policy itself: the value function only SCORES the candidates
    # (credit for training) and never influences which move is played, which
    # also keeps collection exactly on-policy and the credit-validity
    # measurement free of selection on the credit.
    play: str = "improved"
    # Temperature T of the sampled move: the move is the argmax of
    # prior logit (+ credit under play="improved") + T * Gumbel noise, i.e. a
    # sample from that policy at temperature T (1 = sample it as is; lower =
    # closer to its greedy play, still exploring).
    # The value function is fitted on the games that are played, so its
    # credit describes THAT way of playing: credit measured under fully
    # sampled play (T=1) did not transfer to greedy play (Shadow eye
    # allocation: valid +2.0 under T=1, greedy panel 184 -> 184/181/179).
    play_temperature: float = 1.
    algorithm: str = "gumbel_search_target_v1"

    def __post_init__(self):
        if (not math.isfinite(self.credit_scale)
                or any(not math.isfinite(value) for _, within, between in self.credit_calibration
                       for value in (within, between))):
            raise ValueError("Credit calibration slopes and scale must be finite")
        if self.chance not in ("sample", "exact") or (self.chance == "exact" and self.mode != "afterstate"):
            raise ValueError("chance is 'sample' or 'exact'; 'exact' exists only in afterstate mode")
        if self.resolve not in ("none", "pending") or (self.resolve == "pending" and self.chance != "exact"):
            raise ValueError("resolve is 'none' or 'pending'; 'pending' needs chance='exact'")
        if type(self.roll_samples) is not int or self.roll_samples < 1:
            raise ValueError("roll_samples must be a positive integer")
        for name in ("simulations", "considered_actions", "max_own_depth", "max_simulation_steps", "chance_samples",
                     "chance_walks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("value_scale", "maxvisit_init"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if (type(self.rescale_values) is not bool or self.mode not in ("tree", "afterstate")
                or type(self.exploration_candidates) is not int or self.exploration_candidates < 0
                or self.algorithm != "gumbel_search_target_v1"):
            raise ValueError("Unsupported search algorithm, mode, exploration count or rescale flag")
        if type(self.greedy) is not bool or (self.greedy and self.mode != "afterstate"):
            raise ValueError("greedy selection is a bool and exists only in afterstate mode")
        if self.candidates not in ("gumbel", "prior") or (self.candidates == "prior" and self.mode != "afterstate"):
            raise ValueError("candidates is 'gumbel' or 'prior'; 'prior' exists only in afterstate mode")
        if self.play not in ("improved", "prior") or (self.play == "prior" and self.mode != "afterstate"):
            raise ValueError("play is 'improved' or 'prior'; 'prior' exists only in afterstate mode")
        if not (math.isfinite(self.play_temperature) and 0. < self.play_temperature <= 1.) or (
                self.play_temperature != 1. and self.mode != "afterstate"):
            raise ValueError("play_temperature lies in (0, 1] and exists only in afterstate mode")


@dataclass(frozen=True)
class SearchResult:
    selected: int
    target: np.ndarray          # improved policy over the complete menu, float64, sums to 1
    visits: np.ndarray          # root visit counts, int64
    prior_logp: np.ndarray      # behavior prior over the complete menu, float64
    root_value: float           # raw network value at the root
    search_value: float         # mixed root value after search
    considered: int
    simulations: int
    elapsed: float
    terminal_leaves: int
    max_depth: int
    steps: int
    completed: np.ndarray = field(default_factory=lambda: np.zeros(0))  # completed Q per root action
    # Ensemble critic only: what EACH member says, [members, actions] (0 where
    # not evaluated) and [members] at the root, so that any subset of the
    # members can be compared afterwards on the same decisions.
    member_completed: np.ndarray | None = None
    member_root: np.ndarray | None = None


def considered_visit_sequence(max_considered: int, simulations: int) -> tuple[int, ...]:
    """Sequential Halving schedule (mctx get_sequence_of_considered_visits).

    Element t is the visit count an action must currently have to be eligible
    at simulation t. Halving is implicit: after each round only the better
    half keeps receiving the next required visit count.
    """
    if max_considered <= 1:
        return tuple([0] * simulations)
    log2max = int(math.ceil(math.log2(max_considered)))
    sequence, visits, considered = [], [0] * max_considered, max_considered
    while len(sequence) < simulations:
        extra = max(1, int(simulations / (log2max * considered)))
        for _ in range(extra):
            sequence.extend(visits[:considered])
            for i in range(considered):
                visits[i] += 1
        considered = max(2, considered // 2)
    return tuple(sequence[:simulations])


def completed_q(logp, visits, totals, raw_value, *, value_scale, maxvisit_init, rescale=False):
    """Completed and visit-scaled Q over the complete menu.

    Unvisited actions receive the mixed value (prior-weighted visited Q mixed
    with the raw value). rescale=True applies mctx's min-max normalization
    (sharpness independent of the Q spread); False keeps raw Q in [-1,1].
    Returns (sigma_q, completed, mixed_value).
    """
    logp = np.asarray(logp, np.float64)
    visits = np.asarray(visits, np.int64)
    totals = np.asarray(totals, np.float64)
    visited = visits > 0
    q = np.zeros(len(logp), np.float64)
    q[visited] = totals[visited] / visits[visited]
    prior = np.exp(logp)
    prior = np.maximum(prior, np.finfo(np.float64).tiny)
    total_visits = int(visits.sum())
    if visited.any():
        mass = prior[visited].sum()
        weighted = float((prior[visited] * q[visited]).sum() / mass)
    else:
        weighted = 0.
    mixed = (float(raw_value) + total_visits * weighted) / (total_visits + 1)
    completed = np.where(visited, q, mixed)
    if rescale:
        low, high = completed.min(), completed.max()
        scaled = (completed - low) / max(high - low, 1e-8)
    else:
        scaled = completed
    sigma = (maxvisit_init + visits.max()) * value_scale * scaled
    return sigma, completed, mixed


def calibrated_credit(logp, evaluated, q, kinds, calibration):
    """Calibrated credit per action (0 where not evaluated).

    logp: prior log-probabilities over the menu; evaluated: bool mask; q:
    value per action (used where evaluated); kinds: kind name per action;
    calibration: ((kind, within, between), ...) with a "*" default.
    A candidate that ends the game (|Q| = 1) keeps its exact credit."""
    logp, q = np.asarray(logp, np.float64), np.asarray(q, np.float64)
    evaluated = np.asarray(evaluated, bool)
    credit = np.zeros(len(logp), np.float64)
    if not evaluated.any():
        return credit
    table = {str(k): (float(w), float(b)) for k, w, b in calibration}
    default = table.get("*", (1., 1.))
    prior = np.maximum(np.exp(logp), np.finfo(np.float64).tiny)
    mix = float((prior[evaluated] * q[evaluated]).sum() / prior[evaluated].sum())
    kinds = np.asarray([str(k) for k in kinds])
    for kind in set(kinds[evaluated].tolist()):
        members = evaluated & (kinds == kind)
        mean = float((prior[members] * q[members]).sum() / prior[members].sum())
        within, between = table.get(kind, default)
        inside = (q[members] - mean) if members.sum() >= 2 else 0.
        credit[members] = within * inside + between * (mean - mix)
    decided = evaluated & (np.abs(q) >= .999)
    credit[decided] = q[decided] - mix
    return credit


def improved_policy(logp, sigma):
    scores = np.asarray(logp, np.float64) + sigma
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def _log_softmax(scores):
    scores = np.asarray(scores, np.float64)
    shifted = scores - scores.max()
    return shifted - math.log(np.exp(shifted).sum())


class _Node:
    __slots__ = ("logp", "visits", "totals", "raw_value", "children")

    def __init__(self, logp, raw_value):
        self.logp = np.asarray(logp, np.float64)
        self.visits = np.zeros(len(self.logp), np.int64)
        self.totals = np.zeros(len(self.logp), np.float64)
        self.raw_value = float(raw_value)
        self.children = {}


def clone_environment(env: PythonEnvironment, seed: int, *, extra_transitions: int, state=None) -> PythonEnvironment:
    """A hypothetical world: cloned state (or the given successor state),
    private RNG, no replay sink."""
    clone = PythonEnvironment.__new__(PythonEnvironment)
    clone.ai, clone.gs, clone.gl = env.ai, env.gs, env.gl
    clone.episode_id, clone.seed = env.episode_id + "#search", int(seed)
    clone.rng = GameRandom(int(seed))
    clone.samplers = {}
    clone.sides, clone.tags = env.sides, env.tags
    clone.max_turns = env.max_turns
    clone.max_transitions = env.committed + int(extra_transitions)
    clone.committed = env.committed
    clone.end = None
    clone.active = env.active
    clone.stage = env.stage
    clone.first_turn = env.first_turn
    clone.state = env.ai.clone_state(env.state) if state is None else state
    return clone


def _movable(card, gl, deck_name):
    return isinstance(card, str) and not card.startswith("_unknown_") and gl.card_deck_of(card) == deck_name


def resample_committed_card(state, ai, own_side, opponent_tag: str, sides: dict, rng: np.random.Generator):
    """The attacker's committed, unrevealed combat card as the DEFENDER knows it.

    While the defender chooses its own combat card the attacker's card is
    committed face down: its deck type is public, its identity is not
    (rulings 2026-08-16 / 1.246 / 1.261). The engine state holds the real
    card (pending_combat["atk_combat_card"], and its copy already lies in the
    face-up discards), and applying the defender's choice resolves the round
    with it, so a clone scored without this step has seen the hidden card
    (found 2026-10-04: the defender's observation was identical under five
    different committed cards while the value of "play no card" moved from
    +0.28 to +0.34). The identity is redrawn from the same distribution the
    production search uses (ai._atk_commit_mixture: playable cards of the
    committed type from everything of the attacker's the defender has not
    seen, the committed copy included) and exchanged with that card's hidden
    copy, so the clone stays a consistent world with the same card counts.
    Returns the sampled card id, or None when nothing is hidden.
    """
    true = ai._hidden_commit_card(state, own_side)
    if true is None:
        return None
    pending = state.pending_combat
    if pending.get("atk_hidden_cc"):
        return None                       # a live human commitment: the engine never held the identity
    mixture = ai._atk_commit_mixture(state, pending)
    if not mixture:
        return None
    weights = np.asarray([copies for _, copies in mixture], np.float64)
    card = mixture[int(rng.choice(len(mixture), p=weights / weights.sum()))][0]
    if card == true:
        return card
    info = state.hands.get(sides[opponent_tag])
    zones = [info.hand, getattr(info, "discarded_face_down", None)] if info is not None else []
    zones += [getattr(state, opponent_tag + "_char_deck", None), getattr(state, opponent_tag + "_strat_deck", None)]
    source = next((zone for zone in zones if zone is not None and card in zone), None)
    slot = None
    for name in sorted(getattr(state, "discard_face_up", None) or {}):
        pile = state.discard_face_up[name]
        if true in pile:
            slot = (pile, len(pile) - 1 - pile[::-1].index(true))
            break
    if source is None or slot is None:
        raise RuntimeError("Committed combat card cannot be exchanged consistently")
    source[source.index(card)] = true
    slot[0][slot[1]] = card
    pending["atk_combat_card"] = card
    return card


def determinize_state(state, gl, own_tag: str, sides: dict, rng: np.random.Generator, *, ai=None):
    """Resample what the searcher cannot see, inside a clone only.

    Opponent hand cards are exchanged with their unseen deck and face-down
    discards per deck type (zone sizes preserved); both own deck orders and
    the hunt pool are shuffled; with `ai`, an unrevealed committed combat card
    of the opponent is redrawn first. Face-up discards, the table and known
    own cards are untouched.
    """
    opponent_tag = "sh" if own_tag == "fp" else "fp"
    if ai is not None:
        resample_committed_card(state, ai, sides[own_tag], opponent_tag, sides, rng)
    info = state.hands.get(sides[opponent_tag])
    hand = info.hand if info is not None else []
    face_down = getattr(info, "discarded_face_down", None) if info is not None else None
    face_down = face_down if isinstance(face_down, list) else []
    for deck_name, suffix in (("character", "_char_deck"), ("strategy", "_strat_deck")):
        deck = getattr(state, opponent_tag + suffix, None)
        if deck is None:
            continue
        zones = [(zone, [i for i, c in enumerate(zone) if _movable(c, gl, deck_name)]) for zone in (hand, deck, face_down)]
        # Canonical order first (sorted), then the permutation: the result
        # depends on the hidden MULTISET and the search RNG only, never on the
        # hidden order or on which of these cards the opponent really holds.
        pool = sorted([zone[i] for zone, index in zones for i in index], key=str)
        if len(pool) < 2:
            continue
        order = rng.permutation(len(pool))
        pool = iter([pool[j] for j in order])
        for zone, index in zones:
            for slot in index:
                zone[slot] = next(pool)
    for suffix in ("_char_deck", "_strat_deck"):
        deck = getattr(state, own_tag + suffix, None)
        if deck is not None and len(deck) > 1:
            canonical = sorted(deck, key=str)
            order = rng.permutation(len(deck))
            deck[:] = [canonical[j] for j in order]
    pool = getattr(state, "hunt_pool", None)
    if pool is not None and len(pool) > 1:
        canonical = sorted(pool, key=repr)
        order = rng.permutation(len(pool))
        pool[:] = [canonical[j] for j in order]


class GumbelSearcher:
    """Owns no game. Each call searches one prepared decision of one live game."""

    def __init__(self, config: SearchConfig):
        self.config = config

    @torch.inference_mode()
    def _value_only(self, actor, world, side_enum):
        """Own-side value of an arbitrary hypothetical state (no menu needed);
        from the actor's private critic when it has one."""
        observation = np.asarray(world.ai.encode_state(world.state, side_enum), dtype=np.float32)
        return float(actor.value_of(observation)[0])

    def search(self, env: PythonEnvironment, frame, actor, opponent_model, rng: np.random.Generator,
               *, prior=None) -> SearchResult:
        """Search one learned decision. `prior` may supply the already computed
        root prediction (log probabilities, value) to avoid a second forward."""
        started = time.perf_counter()
        config = self.config
        side = frame.side
        actions = frame.actions
        count = len(actions)
        if count < 2:
            raise ValueError("Search requires a learned decision with at least two options")
        if prior is None:
            prior = actor.predict(frame.inputs)
        if config.mode == "afterstate":
            return self._search_afterstate(env, frame, actor, rng, prior, started, opponent_model)
        root = _Node(prior.log_probabilities, prior.value)
        considered = min(config.considered_actions, count)
        gumbel = rng.gumbel(size=count)
        top = np.argsort(-(gumbel + root.logp), kind="stable")[:considered]
        eligible = np.zeros(count, bool)
        eligible[top] = True
        schedule = considered_visit_sequence(considered, config.simulations)
        side_enum = env.sides[side]
        terminal_leaves = max_depth = steps_total = 0
        # Clone-side actions are private copies: a retained pending payload
        # inside a hypothetical world must never alias the live menu.
        private_actions = [type(a)(a.die_idx, a.action_type, deepcopy(a.params)) for a in actions]
        with contextlib.ExitStack() as stack:
            if config.quiet_simulations:
                sink = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(sink))
                stack.enter_context(contextlib.redirect_stderr(sink))
            for simulation, required in enumerate(schedule):
                sigma, _, _ = completed_q(root.logp, root.visits, root.totals, root.raw_value,
                                          value_scale=config.value_scale, maxvisit_init=config.maxvisit_init,
                                          rescale=config.rescale_values)
                scores = gumbel + root.logp + sigma
                scores[~eligible | (root.visits != required)] = -np.inf
                if not np.isfinite(scores).any():
                    # The schedule asked for a visit count no eligible action has.
                    scores = np.where(eligible, gumbel + root.logp + sigma, -np.inf)
                choice = int(np.argmax(scores))
                value, depth, steps, terminal = self._simulate(env, frame, private_actions, choice, root, actor,
                                                               opponent_model, rng, side, side_enum, simulation)
                terminal_leaves += terminal
                max_depth = max(max_depth, depth)
                steps_total += steps
        sigma, completed, mixed = completed_q(root.logp, root.visits, root.totals, root.raw_value,
                                              value_scale=config.value_scale, maxvisit_init=config.maxvisit_init,
                                              rescale=config.rescale_values)
        target = improved_policy(root.logp, sigma)
        final = gumbel + root.logp + sigma
        final[root.visits != root.visits.max()] = -np.inf
        selected = int(np.argmax(final))
        return SearchResult(selected, target, root.visits.copy(), root.logp.copy(), root.raw_value, float(mixed),
                            considered, config.simulations, time.perf_counter() - started,
                            terminal_leaves, max_depth, steps_total, np.asarray(completed, np.float64))

    def _search_afterstate(self, env, frame, actor, rng, prior, started, opponent_model=None):
        """One-ply afterstate improvement: score each Gumbel-sampled root action
        by the critic's value of the deterministic position right after it.

        No opponent model, no deeper simulation. Chance-triggering actions
        (combat or hunt resolved during the apply) average chance_samples
        clones; everything else is evaluated once and is exactly reproducible.
        With chance="exact" every candidate's value is the exact expectation
        over the chance outcomes of its apply (chance.py), and sampling is
        only the fallback for chance the enumerator does not model.
        """
        from .chance import resolve
        from .probability_schema import INDEX as PROBABILITY_INDEX
        config = self.config
        side = frame.side
        actions = frame.actions
        count = len(actions)
        logp = np.asarray(prior.log_probabilities, np.float64)
        considered = min(config.considered_actions, count)
        gumbel = rng.gumbel(size=count)
        ranking = logp if config.candidates == "prior" else gumbel + logp
        top = np.argsort(-ranking, kind="stable")[:considered]
        if config.exploration_candidates and count > considered:
            # Uniform exploration quota over the actions the prior did not put
            # forward; they enter the completed Q and the target like any other
            # evaluated action (their low prior still weights the target).
            rest = np.setdiff1d(np.arange(count), top, assume_unique=True)
            extra = rng.choice(rest, size=min(config.exploration_candidates, len(rest)), replace=False)
            top = np.concatenate((top, extra))
            considered = len(top)
        visits = np.zeros(count, np.int64)
        totals = np.zeros(count, np.float64)
        side_enum = env.sides[side]
        chance_columns = [PROBABILITY_INDEX["combat_available"], PROBABILITY_INDEX["hunt_available"]]
        chance_flags = frame.inputs.probabilities[:, chance_columns].sum(axis=1) > 0
        private_actions = [type(a)(a.die_idx, a.action_type, deepcopy(a.params)) for a in actions]
        private_frame = _frame_with(frame, private_actions)
        terminal = evaluations = 0
        members = (None, None)
        with contextlib.ExitStack() as stack:
            if config.quiet_simulations:
                sink = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(sink))
                stack.enter_context(contextlib.redirect_stderr(sink))
            if config.chance == "exact":
                # ONE hidden world per decision, the same for every candidate
                # and every walk (determinize_state is a function of the
                # hidden multisets and this seed only): candidates are
                # compared in the same resampled world, so what the mover
                # cannot see never adds noise to the DIFFERENCE between two
                # candidates, and the walks of one candidate differ in the
                # chance outcomes alone. Sampled fallbacks share their seeds
                # across candidates for the same reason.
                decision_seed = int(rng.integers(0, 2 ** 62))

                def hidden(world):
                    if config.determinize:
                        determinize_state(world.state, world.gl, side, world.sides,
                                          np.random.default_rng(decision_seed), ai=world.ai)
                policies, roll_samples = None, 0
                if config.resolve == "pending":
                    policies = {side: actor, ("sh" if side == "fp" else "fp"): opponent_model}
                    roll_samples = config.roll_samples
                owners, weights, observations = [], [], []
                member_root = actor.member_values_of(frame.inputs.observation)
                if member_root is not None:
                    member_root = member_root.cpu().numpy().astype(np.float64).reshape(-1)
                    member_totals = np.zeros((len(member_root), count), np.float64)
                for choice in top:
                    choice = int(choice)
                    leaves, _ = resolve(env, frame, choice, determinize=hidden, policies=policies,
                                        max_walks=config.chance_walks, samples=config.chance_samples, seed=decision_seed,
                                        roll_samples=roll_samples)
                    for probability, world in leaves:
                        if world._victory():
                            totals[choice] += probability * world.end.terminal_return(side)
                            if member_root is not None:
                                member_totals[:, choice] += probability * world.end.terminal_return(side)
                            terminal += 1
                        else:
                            owners.append(choice)
                            weights.append(probability)
                            observations.append(np.asarray(world.ai.encode_state(world.state, side_enum), dtype=np.float32))
                    evaluations += len(leaves)
                    # One visit per evaluated candidate (an expectation, or
                    # the mean of a sampled fallback): the target scale
                    # (maxvisit_init + visits.max()) is the same for every
                    # decision.
                    visits[choice] = 1
                if observations and member_root is not None:
                    each = actor.member_values_of(np.stack(observations)).cpu().numpy().astype(np.float64)
                    for k in range(len(each)):
                        np.add.at(member_totals[k], np.asarray(owners, np.int64), np.asarray(weights, np.float64) * each[k])
                    np.add.at(totals, np.asarray(owners, np.int64), np.asarray(weights, np.float64) * each.mean(axis=0))
                elif observations:
                    values = actor.value_of(np.stack(observations)).cpu().numpy().astype(np.float64).reshape(-1)
                    np.add.at(totals, np.asarray(owners, np.int64), np.asarray(weights, np.float64) * values)
                if member_root is not None:
                    members = (member_totals.astype(np.float32), member_root.astype(np.float32))
                top = ()
            for choice in top:
                choice = int(choice)
                samples = config.chance_samples if chance_flags[choice] else 1
                for _ in range(samples):
                    world = clone_environment(env, int(rng.integers(0, 2 ** 62)), extra_transitions=4)
                    if config.determinize:
                        # HIDDEN INFORMATION. The clone is a copy of the true
                        # state: the hunt pool is a pre-shuffled list whose
                        # top tile is the tile that WILL be drawn, and the
                        # decks are in their real order. Until 2026-10-04
                        # this mode applied the move on that copy directly,
                        # so every afterstate value of a Fellowship move (and
                        # of any draw) had seen the real next tile / card.
                        # Found because that credit predicted results (+0.72)
                        # yet could not be predicted from the game state
                        # (held-out R^2 .07). Everything unseen is resampled
                        # in every clone, as the tree mode always did.
                        determinize_state(world.state, world.gl, side, world.sides, rng, ai=world.ai)
                    world.apply(private_frame, choice)
                    evaluations += 1
                    if world._victory():
                        value = world.end.terminal_return(side)
                        terminal += 1
                    else:
                        value = self._value_only(actor, world, side_enum)
                    visits[choice] += 1
                    totals[choice] += value
        sigma, completed, mixed = completed_q(logp, visits, totals, prior.value, value_scale=config.value_scale,
                                              maxvisit_init=config.maxvisit_init, rescale=config.rescale_values)
        if config.credit_calibration:
            names = [str(getattr(a.action_type, "name", a.action_type)) for a in actions]
            sigma = (config.maxvisit_init + visits.max()) * config.value_scale * config.credit_scale * calibrated_credit(
                logp, visits > 0, completed, names, config.credit_calibration)
        elif config.credit_gate:
            gate = dict(config.credit_gate)
            default = float(gate.get("*", 0.))
            weight = np.array([float(gate.get(str(getattr(a.action_type, "name", a.action_type)), default)) for a in actions])
            weight = np.where((visits > 0) & (np.abs(completed) >= .999), 1., weight)
            gated = np.where(visits > 0, mixed + weight * (completed - mixed), mixed)
            sigma = (config.maxvisit_init + visits.max()) * config.value_scale * gated
        target = improved_policy(logp, sigma)
        steer = sigma if config.play == "improved" else 0.
        # argmax(logp + T * gumbel) is a sample from the policy at temperature T.
        final = (logp + steer) if config.greedy else (config.play_temperature * gumbel + logp + steer)
        final[visits == 0] = -np.inf
        selected = int(np.argmax(final))
        return SearchResult(selected, target, visits, logp.copy(), float(prior.value), float(mixed),
                            considered, evaluations, time.perf_counter() - started, terminal, 1, evaluations,
                            np.asarray(completed, np.float64), members[0], members[1])

    def _simulate(self, env, frame, private_actions, choice, root, actor, opponent_model, rng,
                  side, side_enum, simulation):
        config = self.config
        seed = int(rng.integers(0, 2 ** 62))
        world = clone_environment(env, seed, extra_transitions=config.max_simulation_steps + 2)
        if config.determinize:
            determinize_state(world.state, world.gl, side, world.sides, rng, ai=world.ai)
        world.apply(_frame_with(frame, private_actions), choice)
        path = [(root, choice)]
        node = root.children.get(choice)
        depth = steps = 0
        terminal = 0
        value = None
        while True:
            current = world.prepare()
            if current is None:
                end = world.end
                if end is not None and end.status == "terminated":
                    value = end.terminal_return(side)
                    terminal = 1
                else:
                    value = self._value_only(actor, world, side_enum)
                break
            steps += 1
            if steps > config.max_simulation_steps:
                value = self._value_only(actor, world, side_enum)
                break
            if len(current.actions) == 1:
                world.apply(current, 0)
                continue
            if current.side != side:
                prediction = opponent_model.predict(current.inputs)
                world.apply(current, prediction.sample(rng))
                continue
            # Own decision: expand, evaluate, or descend.
            if node is None or len(node.logp) != len(current.actions) or depth >= config.max_own_depth:
                prediction = actor.predict(current.inputs)
                if node is None and depth < config.max_own_depth:
                    parent, index = path[-1]
                    parent.children[index] = _Node(prediction.log_probabilities, prediction.value)
                value = prediction.value
                break
            sigma, _, _ = completed_q(node.logp, node.visits, node.totals, node.raw_value,
                                      value_scale=config.value_scale, maxvisit_init=config.maxvisit_init,
                                      rescale=config.rescale_values)
            policy = improved_policy(node.logp, sigma)
            pick = int(np.argmax(policy - node.visits / (1 + node.visits.sum())))
            path.append((node, pick))
            world.apply(current, pick)
            node = node.children.get(pick)
            depth += 1
        for parent, index in path:
            parent.visits[index] += 1
            parent.totals[index] += value
        return value, depth, steps, terminal


def _frame_with(frame, actions):
    return type(frame)(frame.side, frame.phase, frame.turn, tuple(actions), frame.inputs, frame.fixed_reason)
