"""Search-target learner: cross-entropy to the search's improved policy,
a bootstrapped value target, and a KL anchor to the frozen human-imitation
policy. No terminal-return policy gradient, no advantage, no entropy bonus.

Credit is LOCAL: each searched decision carries its own supervised target.
The terminal outcome enters only through the value target's lambda-return,
bootstrapped from later search root values (horizon ~1/(1-lambda) own
decisions), so one game's result no longer has to explain 200 decisions.

BatchNorm buffers stay fixed and dropout stays disabled (eval mode), as in
collection; gradients still train the backbone and every head.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .contracts import CONTRACT, BehaviorPolicy, ContractError
from .policy import (ActorInput, PolicyActor, actor_digest, build_critic_model, critic_scalar_names, forward_ragged,
                     weights_digest, _install_chance_head, _install_credit_head_from_state, _install_kind_bias_from_state)
from .race_critic import UNLABELED, RaceCritic, race_label, race_value_loss
from .search_rollout_io import SearchGame, load_search_arrays


@dataclass(frozen=True)
class SearchLearnerConfig:
    learning_rate: float = 1e-4
    epochs: int = 2
    max_decisions: int = 128
    max_candidates: int = 131072
    action_chunk_size: int = 16384
    value_coefficient: float = .5
    anchor_coefficient: float = .1
    # The KL anchor to the frozen human-imitation policy is a cold-start
    # safeguard, not a permanent tether: its coefficient is scaled by
    # max(0, 1 - generation / anchor_decay_generations), reaching zero at
    # that generation. 0 keeps the anchor constant.
    anchor_decay_generations: int = 4
    value_lambda: float = .95
    # The imported seed's critic is zero, so generation-0 search values carry
    # no information to bootstrap from. The FIRST update uses this lambda
    # (1 = Monte Carlo outcome) so the critic starts from the global signal.
    cold_start_value_lambda: float = 1.
    # Policy terms (CE + anchor) start at this generation; earlier updates are
    # critic-only. In generation 0 the targets equal the prior, and even a
    # no-signal update perturbed near-tied argmaxes (45% of decisions with a
    # top-2 margin < .02 flipped) and cost 10-15 greedy win points.
    policy_from_generation: int = 1
    # Policy terms apply only to decisions whose search target disagrees with
    # the collection-time prior by at least this KL; rows without a search
    # opinion must not erode the prior's structure. Value terms use all rows.
    min_target_kl: float = .01
    max_gradient_norm: float = 1.
    # Value-only rows (afterstates of the fitted games, pre-states and
    # afterstates of replay games) trained per optimizer step alongside the
    # decision batch.
    value_batch_size: int = 256
    # Random human-corpus rows (observation, +/-1 outcome) added to the value
    # pool each update, so a corpus-trained critic (60% explained variance)
    # is not pulled back toward a few hundred noisy self-play outcomes.
    corpus_replay_rows: int = 50000
    # Learning rate of the private critic network (whole network, Adam).
    critic_learning_rate: float = 3e-5
    # Train the private critic's backbone on self-play outcomes? ~500
    # independent labels per generation cannot train 1.3M parameters:
    # end to end it memorized game identities (in-sample .15, held-out .60-.72).
    # False keeps the corpus-pretrained backbone and trains the head only.
    critic_train_backbone: bool = False
    # Outcome labels are one constant per game. Letting their gradient reach
    # the shared body wrecked the pretrained representation in generation 0
    # (policy drifted with no policy signal). Default: value head only.
    value_through_backbone: bool = False
    # Whole-game shuffling window; MC labels are constant within a game, so a
    # small window makes consecutive steps chase a few outcomes. A game is
    # ~450 MB uncompressed and two chunks are resident (prefetch).
    games_per_chunk: int = 24
    # Whole games held out from fitting; their metrics are the honest report.
    holdout_fraction: float = .1
    measure_decisions: int = 4096
    # Credit gate: ((action kind, weight), ...). The value function's credit
    # is only used for kinds of move where it has been shown to predict
    # results (credit_validity.py); the weight shrinks a candidate's value
    # toward the sibling mix (0 = no credit, 1 = full). Kinds not listed get
    # the "*" weight (default 0). Empty = no gate (every kind, full weight).
    # Measured 2026-10-04: with race_critic_r1 the credit for a move other
    # than the policy's favourite was valid for Fellowship moves and
    # two-option decisions, zero for army moves / musters / events, and
    # INVERTED for attacks and political advances. Training on all of it
    # taught noise, which is why no generation improved.
    credit_gate: tuple = ()
    # Games scored by an ensemble critic store every member's value of every
    # candidate. A nonempty tuple of member indices recomputes the target
    # from the mean of THOSE members (member_validity.py picks them); empty =
    # the stored target (mean of all members).
    credit_members: tuple = ()
    # ((kind, within, between), ...) from validity_bench.py calibrate: the
    # target is recomputed from the stored Q with the calibrated credit
    # (search.calibrated_credit). Takes precedence over credit_gate.
    credit_calibration: tuple = ()
    credit_scale: float = 1.
    # False: a policy-only fit builds no value pool (pre-states and
    # afterstates of every fitted game: ~10 GB and one extra pass over the
    # games for 2,000 games) and measures no held-out values.
    value_pool: bool = True
    algorithm: str = "gumbel_search_target_v1"

    def __post_init__(self):
        object.__setattr__(self, "credit_gate", tuple((str(k), float(w)) for k, w in self.credit_gate))
        object.__setattr__(self, "credit_members", tuple(int(k) for k in self.credit_members))
        object.__setattr__(self, "credit_calibration", tuple((str(k), float(w), float(b)) for k, w, b in self.credit_calibration))
        if (not math.isfinite(self.credit_scale)
                or any(not math.isfinite(value) for _, within, between in self.credit_calibration
                       for value in (within, between))):
            raise ValueError("Credit calibration slopes and scale must be finite")
        if type(self.value_pool) is not bool or any(k < 0 for k in self.credit_members):
            raise ValueError("value_pool is a bool; credit members are nonnegative indices")
        if any(not math.isfinite(w) or not 0 <= w <= 2 for _, w in self.credit_gate):
            raise ValueError("credit gate weights must lie in [0, 2]")
        for name in ("epochs", "max_decisions", "max_candidates", "action_chunk_size", "games_per_chunk", "measure_decisions",
                     "value_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("learning_rate", "max_gradient_norm", "critic_learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("value_coefficient", "anchor_coefficient"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if (not 0 <= self.value_lambda <= 1 or not 0 <= self.cold_start_value_lambda <= 1
                or not 0 <= self.holdout_fraction < 1 or type(self.value_through_backbone) is not bool
                or type(self.anchor_decay_generations) is not int or self.anchor_decay_generations < 0
                or type(self.policy_from_generation) is not int or self.policy_from_generation < 0
                or not math.isfinite(self.min_target_kl) or self.min_target_kl < 0
                or type(self.corpus_replay_rows) is not int or self.corpus_replay_rows < 0
                or type(self.critic_train_backbone) is not bool
                or self.algorithm != "gumbel_search_target_v1"):
            raise ValueError("Unsupported value lambda, holdout fraction, anchor decay, policy gate or algorithm")


@dataclass(frozen=True)
class DecisionSample:
    episode_id: str
    index: int
    inputs: ActorInput
    target: np.ndarray
    value_target: float
    selected: int
    target_kl: float = 0.  # KL(target || collection-time prior): how much the search disagreed
    outcome: float = 0.    # terminal +/-1 for the owner (diagnostic only; the value target may bootstrap)

    @property
    def action_count(self):
        return len(self.target)


def value_targets(search_values, terminal_return, value_lambda):
    """Lambda-return over ONE owner's searched decisions, gamma 1, no reward.

    G[T-1] = z (the terminal result). G[t] = (1-lambda) v[t+1] + lambda G[t+1],
    where v[t+1] is the search root value at the owner's NEXT searched
    decision. lambda=1 is Monte Carlo; lambda=0 bootstraps one decision ahead.
    """
    values = np.asarray(search_values, np.float64)
    if terminal_return not in (-1., 1.) or not 0 <= value_lambda <= 1:
        raise ValueError("Value targets require a decided outcome and lambda in [0,1]")
    targets = np.empty_like(values)
    following = float(terminal_return)
    for t in range(len(values) - 1, -1, -1):
        if t == len(values) - 1:
            targets[t] = terminal_return
        else:
            targets[t] = (1 - value_lambda) * values[t + 1] + value_lambda * following
        following = targets[t]
    return targets


# Reporting threshold: a decision counts as "the credit moved the target" when
# KL(target || prior at collection) is at least this (independent of min_target_kl).
REPORT_TARGET_KL = .005


def gated_target(game: SearchGame, arrays: dict, row_index: int, start: int, stop: int, credit_gate, *,
                 return_sigma=False, members=(), calibration=(), credit_scale=1.) -> np.ndarray:
    """The search's improved policy recomputed with per-kind credit weights.

    Stored per candidate: completed Q (afterstate value if evaluated, the
    sibling mix otherwise), visits and the prior. A candidate of kind k keeps
    weight w_k of its distance from the sibling mix; an evaluated candidate
    that ended the game (|Q| = 1) keeps full weight, its credit being exact.
    With every weight and credit_scale equal to 1 this reproduces the stored target."""
    if not game.kind_names or "kinds" not in arrays:
        raise ContractError("A credit gate needs games recorded with candidate kinds")
    search = game.search
    if search.get("rescale_values", False):
        raise ContractError("The credit gate is defined for unrescaled Q only")
    gate = dict(credit_gate)
    default = gate.get("*", 0.)
    by_code = np.array([gate.get(name, default) for name in game.kind_names], np.float64)
    weight = by_code[np.asarray(arrays["kinds"][start:stop], np.int64)]
    completed = np.asarray(arrays["completed_q"][start:stop], np.float64)
    visits = np.asarray(arrays["visits"][start:stop], np.int64)
    logp = np.asarray(arrays["prior_logp"][start:stop], np.float64)
    mixed = float(game.rows[row_index].search_value)
    if len(members):
        # The same completed Q and sibling mix as search.completed_q, from the chosen members only.
        if "member_q" not in arrays:
            raise ContractError("Credit members need games scored by an ensemble critic")
        order = getattr(game, "_searched_order", None)
        if order is None:
            order = {i: k for k, i in enumerate(game.searched_rows)}
            object.__setattr__(game, "_searched_order", order)
        chosen = list(members)
        evaluated = visits > 0
        q = np.asarray(arrays["member_q"][start:stop], np.float64)[:, chosen].mean(axis=1)
        root = float(np.asarray(arrays["member_root"][order[row_index]], np.float64)[chosen].mean())
        prior = np.maximum(np.exp(logp), np.finfo(np.float64).tiny)
        weighted = float((prior[evaluated] * q[evaluated]).sum() / prior[evaluated].sum()) if evaluated.any() else 0.
        total = int(visits.sum())
        mixed = (root + total * weighted) / (total + 1)
        completed = np.where(evaluated, q, mixed)
    weight = np.where((visits > 0) & (np.abs(completed) >= .999), 1., weight)
    gated = np.where(visits > 0, mixed + weight * (completed - mixed), mixed)
    scale = (float(search["maxvisit_init"]) + int(visits.max())) * float(search["value_scale"])
    if len(calibration):
        from .search import calibrated_credit
        names = [game.kind_names[int(k)] for k in np.asarray(arrays["kinds"][start:stop], np.int64)]
        sigma = scale * credit_scale * calibrated_credit(logp, visits > 0, completed, names, calibration)
        if return_sigma:
            return sigma
        scores = logp + sigma
        scores -= scores.max()
        target = np.exp(scores)
        return target / target.sum()
    if return_sigma:
        # The logit correction itself, zero for unevaluated and gated-out candidates.
        return scale * credit_scale * (gated - mixed)
    scores = logp + scale * credit_scale * gated
    scores -= scores.max()
    target = np.exp(scores)
    return target / target.sum()


def game_samples(game: SearchGame, arrays: dict, identity: BehaviorPolicy, value_lambda: float, *, credit_gate=(),
                 credit_members=(), credit_calibration=(), credit_scale=1.):
    """Fresh on-policy searched decisions of one owner, with value targets."""
    side = identity.side
    if game.end is None or game.end.status != "terminated":
        return []
    own = [i for i, row in enumerate(game.rows) if row.side == side and row.searched]
    if not own:
        return []
    for i in own:
        row = game.rows[i]
        if row.behavior != identity:
            if row.behavior.origin == "historical":
                return []
            raise ContractError("Stale learner weights/generation; fresh on-policy collection is required")
        if row.kind != "learned" or row.search_value is None:
            raise ContractError("Searched rows must be learned decisions with a search value")
    own.sort(key=lambda i: game.rows[i].side_index)
    targets = value_targets([game.rows[i].search_value for i in own], game.end.terminal_return(side), value_lambda)
    offsets = arrays["offsets"]
    samples = []
    for i, value_target in zip(own, targets):
        start, stop = int(offsets[i]), int(offsets[i + 1])
        kinds = None
        if game.kind_names:
            codes = np.asarray(arrays.get("kinds", ()))[start:stop]
            if (codes.shape != (stop - start,) or codes.dtype.kind not in "iu"
                    or np.any(codes < 0) or np.any(codes >= len(game.kind_names))):
                raise ContractError("Recorded candidate kind codes do not match the menu vocabulary")
            kinds = tuple(game.kind_names[int(code)] for code in codes)
        inputs = ActorInput(arrays["observations"][i], arrays["candidates"][start:stop],
                            arrays["regions"][start:stop], arrays["probabilities"][start:stop], kinds)
        target = np.asarray(arrays["target"][start:stop], np.float64)
        if not np.isfinite(target).all() or target.min() < 0 or not np.isclose(target.sum(), 1., rtol=0, atol=1e-5):
            raise ContractError("Recorded search target is not a normalized distribution")
        if credit_gate or credit_members or credit_calibration:
            target = gated_target(game, arrays, i, start, stop, credit_gate or (("*", 1.),), members=credit_members,
                                  calibration=credit_calibration, credit_scale=credit_scale)
        prior = np.asarray(arrays["prior_logp"][start:stop], np.float64)
        positive = target > 0
        target_kl = float(np.sum(target[positive] * (np.log(target[positive]) - prior[positive])))
        samples.append(DecisionSample(game.episode_id, i, inputs, target, float(value_target), game.rows[i].selected,
                                      max(0., target_kl), float(game.end.terminal_return(side))))
    return samples


@dataclass(frozen=True)
class ValueSample:
    """A value-only training row: an observation with its return (no menu).

    clock / turns_left are the race critic's survival label (race_critic.py):
    which clock ended the game (-1 none, -2 unlabeled) and how many turns
    after this row's turn it ended."""
    episode_id: str
    index: int
    observation: np.ndarray
    value_target: float
    outcome: float
    afterstate: bool
    clock: int = UNLABELED
    turns_left: float = float("nan")


def game_end_turn(game: SearchGame) -> int:
    """Turn the game ended on: the last recorded row's turn (rows cover
    every decision up to the terminal transition)."""
    return max(int(row.turn) for row in game.rows)


def value_samples(game: SearchGame, arrays: dict, identity: BehaviorPolicy, value_lambda: float, *,
                  include_prestates: bool):
    """Value-only rows of one owner: recorded afterstates (same return as their
    decision) and, for replay games, the pre-decision observations too.
    Replay games are older policies, so their search values are stale: callers
    pass value_lambda=1 (Monte-Carlo outcome) for them."""
    side = identity.side
    if game.end is None or game.end.status != "terminated":
        return []
    own = [i for i, row in enumerate(game.rows) if row.side == side and row.searched]
    if not own:
        return []
    own.sort(key=lambda i: game.rows[i].side_index)
    targets = value_targets([game.rows[i].search_value for i in own], game.end.terminal_return(side), value_lambda)
    outcome = float(game.end.terminal_return(side))
    searched_order = {i: j for j, i in enumerate(k for k, row in enumerate(game.rows) if row.searched)}
    afterstates = arrays.get("afterstates")
    end_turn = game_end_turn(game)
    samples = []
    for i, value_target in zip(own, targets):
        row = game.rows[i]
        label = race_label(game.end.winner, game.end.victory_route, end_turn, row.turn)
        if include_prestates:
            samples.append(ValueSample(game.episode_id, i, arrays["observations"][i], float(value_target), outcome, False,
                                       label.clock, label.turns_left))
        if row.has_afterstate and afterstates is not None and searched_order[i] < len(afterstates):
            samples.append(ValueSample(game.episode_id, i, afterstates[searched_order[i]], float(value_target), outcome, True,
                                       label.clock, label.turns_left))
    return samples


def corpus_value_samples(source, rows, rng):
    """Random human-corpus rows as value samples (outcome = target). A
    four-part source (observations, outcomes, clocks, turns_left) also
    carries the race critic's survival labels."""
    if source is None or rows <= 0:
        return []
    observations_path, outcomes_path = source[0], source[1]
    outcomes = np.load(outcomes_path, mmap_mode="r")
    count = len(outcomes)
    index = np.sort(rng.choice(count, size=min(rows, count), replace=False))
    observations = np.load(observations_path, mmap_mode="r")
    block = np.ascontiguousarray(observations[index], dtype=np.float32)
    labels = np.asarray(outcomes[index], np.float32)
    if len(source) >= 4:
        clocks = np.asarray(np.load(source[2], mmap_mode="r")[index], np.int64)
        turns_left = np.asarray(np.load(source[3], mmap_mode="r")[index], np.float32)
        return [ValueSample("corpus", int(i), block[j], float(labels[j]), float(labels[j]), False,
                            int(clocks[j]), float(turns_left[j])) for j, i in enumerate(index)]
    return [ValueSample("corpus", int(i), block[j], float(labels[j]), float(labels[j]), False)
            for j, i in enumerate(index)]


def value_only_loss(model, value_samples, config: SearchLearnerConfig, *, through_backbone=None):
    """Half-MSE of the critic on observation-only rows (value head only when
    value_through_backbone is False, as in search_loss; a private critic
    passes through_backbone=True because its backbone is its own). A race
    critic trains on its survival + outcome loss instead (same metric keys)."""
    if isinstance(model, RaceCritic):
        return race_value_loss(model, value_samples)
    device = next(model.parameters()).device
    if through_backbone is None:
        through_backbone = config.value_through_backbone
    observations = torch.from_numpy(np.stack([s.observation for s in value_samples])).to(device)
    backbone, _ = model.forward_backbone(observations)
    values = model.value_head(backbone if through_backbone else backbone.detach()).reshape(-1)
    target = torch.tensor([s.value_target for s in value_samples], dtype=values.dtype, device=device)
    outcome = torch.tensor([s.outcome for s in value_samples], dtype=values.dtype, device=device)
    loss = .5 * (values - target).square()
    return loss.mean(), dict(value_only_loss=loss.mean(), value_only_outcome_mse=(.5 * (values.detach() - outcome).square()).mean())


def decision_batches(samples, order, *, max_decisions, max_candidates):
    """Bound memory by candidates without truncating any decision's menu."""
    batch, count = [], 0
    for raw in order:
        i = int(raw)
        n = samples[i].action_count
        if batch and (len(batch) >= max_decisions or count + n > max_candidates):
            yield batch
            batch, count = [], 0
        batch.append(i)
        count += n
    if batch:
        yield batch


def _log_softmax_ragged(logits):
    return [torch.log_softmax(scores.double(), dim=0) for scores in logits]


def _accumulate_metrics(sums, counts, metrics, rows):
    """Conditional KL means count their own rows, including across microbatches."""
    values = {key: float(value.detach()) for key, value in metrics.items()}
    moved = values.get("target_moved_fraction", 0.) * rows
    for key, value in values.items():
        count = (moved if key == "kl_where_target_moved" else rows - moved
                 if key == "kl_where_target_is_prior" else rows)
        sums[key] = sums.get(key, 0.) + value * count
        counts[key] = counts.get(key, 0.) + count


def _metric_means(sums, counts):
    return {key: value / counts[key] if counts[key] else 0. for key, value in sums.items()}


def episode_evidence_keys(game):
    """Run-local names are labels; directories and completed evidence identify games."""
    keys = {"path:" + os.path.normcase(str(game.path.resolve()))}
    if game.arrays_sha256:
        content = dict(seed=game.seed, arrays_sha256=game.arrays_sha256,
                       rows=[dict(row._asdict(), behavior=asdict(row.behavior)) for row in game.rows],
                       end=None if game.end is None else asdict(replace(game.end, episode_id="content")))
        keys.add("content:" + hashlib.sha256(json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest())
    return keys


def require_unique_episodes(games):
    """Keep repeated directories/copied evidence out of whole-game holdouts.

    Episode IDs such as b000-train-0 are local to a run and may repeat
    legitimately. Equal seeds also remain valid when the completed arrays
    differ (for example games against different opponents).
    """
    seen = set()
    for game in games:
        keys = episode_evidence_keys(game)
        if seen.intersection(keys):
            raise ContractError("Duplicate episode evidence cannot enter a whole-game fitting cohort")
        seen.update(keys)


def search_loss(model, anchor_model, samples, config: SearchLearnerConfig, *, anchor_scale=1., policy_weight=1.,
                policy_normalizer=None):
    """Per-decision CE to the search target + value MSE + KL(policy || anchor).

    Policy terms (CE, anchor) are averaged over INFORMATIVE decisions only:
    those whose target_kl >= config.min_target_kl. Value terms use every
    decision. policy_weight 0 (generation gate) removes the policy terms.
    anchor_scale multiplies the anchor coefficient (the learner's decay
    schedule); reported metrics are unscaled and cover all decisions.
    A microbatch can supply policy_normalizer so that its externally weighted
    sum retains the logical batch's informative-row mean."""
    values, logits = forward_ragged(model, [s.inputs for s in samples], max_action_rows=config.action_chunk_size,
                                    detach_value=not config.value_through_backbone)
    device = values.device
    log_policies = _log_softmax_ragged(logits)
    targets = [torch.from_numpy(s.target).to(device) for s in samples]
    cross_entropy = torch.stack([-(t * lp).sum() for t, lp in zip(targets, log_policies)])
    entropy = torch.stack([-(lp.exp() * lp).sum() for lp in log_policies])
    target_entropy = torch.stack([-(t * torch.log(t.clamp_min(1e-300))).sum() for t in targets])
    agreement = torch.stack([(lp.argmax() == t.argmax()).double() for t, lp in zip(targets, log_policies)])
    informative = torch.tensor([s.target_kl >= config.min_target_kl for s in samples], dtype=torch.float64, device=device)
    count = informative.sum()
    denominator = count if policy_normalizer is None else policy_normalizer
    value_target = torch.tensor([s.value_target for s in samples], dtype=values.dtype, device=device)
    value_loss = .5 * (values - value_target).square()
    policy_term = (cross_entropy * informative).sum() / denominator if float(count) > 0 else cross_entropy.sum() * 0.
    loss = policy_weight * policy_term + config.value_coefficient * value_loss.mean()
    outcome = torch.tensor([s.outcome for s in samples], dtype=values.dtype, device=device)
    metrics = dict(cross_entropy=cross_entropy.mean(), value_loss=value_loss.mean(), entropy=entropy.mean(),
                   target_entropy=target_entropy.mean(), argmax_agreement=agreement.mean(),
                   informative_fraction=informative.mean(),
                   target_kl=torch.tensor(float(np.mean([s.target_kl for s in samples])), device=device),
                   # Critic accuracy against the actual terminal result (0.5 = zero critic),
                   # independent of the bootstrapped value target.
                   outcome_mse=(.5 * (values.detach() - outcome).square()).mean())
    # The two things an update must do, reported separately: get closer to the
    # target where the credit moved it, and stay put where the target is the
    # policy's own distribution (drift there is damage, not learning).
    kl_each = (cross_entropy - target_entropy).detach()
    moved = torch.tensor([s.target_kl >= REPORT_TARGET_KL for s in samples], dtype=torch.float64, device=device)
    metrics["kl_where_target_moved"] = (kl_each * moved).sum() / moved.sum().clamp(min=1.)
    metrics["kl_where_target_is_prior"] = (kl_each * (1 - moved)).sum() / (1 - moved).sum().clamp(min=1.)
    metrics["target_moved_fraction"] = moved.mean()
    if anchor_model is not None and config.anchor_coefficient > 0:
        with torch.no_grad():
            _, anchor_logits = forward_ragged(anchor_model, [s.inputs for s in samples], max_action_rows=config.action_chunk_size)
            anchor_log = _log_softmax_ragged(anchor_logits)
        kl = torch.stack([(lp.exp() * (lp - al)).sum() for lp, al in zip(log_policies, anchor_log)])
        if anchor_scale > 0 and policy_weight > 0 and float(count) > 0:
            loss = loss + policy_weight * config.anchor_coefficient * anchor_scale * (kl * informative).sum() / denominator
        metrics["anchor_kl"] = kl.mean()
    return loss, metrics


class SearchLearner:
    def __init__(self, actor: PolicyActor, config=SearchLearnerConfig(), *, seed=52, anchor: PolicyActor | None = None,
                 device="cpu", critic=None):
        attached = critic if critic is not None else actor.critic
        if actor.identity.origin != "learner" or actor_digest(
                actor.model.state_dict(), None if attached is None else attached.state_dict()) != actor.identity.weights_sha256:
            raise ContractError("Learner initialization requires matching owned actor weights (policy and private critic)")
        if config.anchor_coefficient > 0 and anchor is None:
            raise ValueError("A positive anchor coefficient needs the frozen anchor policy")
        if anchor is not None and anchor.identity.side != actor.identity.side:
            raise ContractError("Anchor policy belongs to another side")
        self.device = torch.device(device)
        self.model = deepcopy(actor.model).to(self.device).eval().requires_grad_(True)
        self.anchor_model = None if anchor is None else deepcopy(anchor.model).to(self.device).eval().requires_grad_(False)
        self.anchor_sha256 = None if anchor is None else anchor.identity.weights_sha256
        self.identity = actor.identity
        self.source_sha256 = actor.source_sha256
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        # Private critic: a full value network trained only on outcomes (its
        # own backbone, never shared with the policy). When present, every
        # value sample trains it, the policy model's value head is idle, and
        # exported actors carry it for the search.
        critic = critic if critic is not None else actor.critic
        self.critic = None if critic is None else deepcopy(critic).to(self.device).eval()
        self.critic_optimizer = None
        if self.critic is not None:
            for name, p in self.critic.named_parameters():
                if config.critic_train_backbone:
                    p.requires_grad_(not name.startswith(("spatial_action_scorer.", "policy_delta.", "chance_policy.")))
                else:
                    p.requires_grad_(name.startswith("value_head."))
            trainable = [p for p in self.critic.parameters() if p.requires_grad]
            # An ensemble critic (parameters under members.*) has no head of its
            # own here: it is used exactly as it was built, never trained by the learner.
            if trainable:
                self.critic_optimizer = torch.optim.Adam(trainable, lr=config.critic_learning_rate, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.consumed_episode_ids = set()
        self.consumed_episode_keys = set()
        # Old checkpoints record only run-local names; retain their existing
        # conservative admission behavior without applying it to new evidence.
        self.legacy_consumed_episode_ids = set()
        self.updates = self.optimizer_steps = 0
        self.failed = False
        # Optional callable returning earlier generations' SearchGame objects
        # whose observations/afterstates train the critic (Monte-Carlo
        # outcome targets; their search values are stale). Set by the runner.
        self.replay_source = None
        # Optional (observations .npy path, outcomes .npy path) of this side's
        # human-corpus rows; corpus_replay_rows random rows join the value pool.
        self.corpus_source = None

    def export_actor(self, *, device="cpu", origin="learner"):
        if self.failed:
            raise RuntimeError("Failed learner must be restored from a completed checkpoint")
        model = deepcopy(self.model).to(device).eval().requires_grad_(False)
        identity = BehaviorPolicy(self.identity.side, self.identity.weights_sha256, self.identity.generation, origin)
        critic = None if self.critic is None else deepcopy(self.critic).to(device).eval().requires_grad_(False)
        return PolicyActor(model, identity, source_sha256=self.source_sha256, critic=critic)

    @property
    def active_value_lambda(self):
        return self.config.cold_start_value_lambda if self.updates == 0 else self.config.value_lambda

    @property
    def anchor_scale(self):
        """Anchor decay by the generation being trained (0 at the start)."""
        decay = self.config.anchor_decay_generations
        if decay == 0:
            return 1.
        return max(0., 1. - self.identity.generation / decay)

    @property
    def policy_weight(self):
        """Generation gate: critic-only updates before policy_from_generation."""
        return 1. if self.identity.generation >= self.config.policy_from_generation else 0.

    def _chunks(self, games):
        """Whole-cohort game permutation, loaded in chunks with ONE chunk of
        prefetch so decompression (which releases the GIL) overlaps the
        optimizer steps on the current chunk. Peak memory is two chunks."""
        from concurrent.futures import ThreadPoolExecutor
        order = self.rng.permutation(len(games))
        value_lambda = self.active_value_lambda
        size = self.config.games_per_chunk
        chunks = [[games[int(i)] for i in order[start:start + size]] for start in range(0, len(order), size)]

        def load(chunk):
            # The games of a chunk are read side by side (hashing and
            # decompression release the GIL): a fit on 2,300 games spent its
            # time here with the GPU at 1% (2026-10-04). Order is preserved.
            def one(game):
                return game_samples(game, load_search_arrays(game), self.identity, value_lambda,
                                    credit_gate=self.config.credit_gate, credit_members=self.config.credit_members,
                                    credit_calibration=self.config.credit_calibration, credit_scale=self.config.credit_scale)
            with ThreadPoolExecutor(max_workers=4) as readers:
                parts = list(readers.map(one, chunk))
            return [sample for part in parts for sample in part]

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(load, chunks[0]) if chunks else None
            for index in range(len(chunks)):
                samples = pending.result()
                pending = pool.submit(load, chunks[index + 1]) if index + 1 < len(chunks) else None
                if samples:
                    yield samples
                del samples

    @torch.no_grad()
    def measure_values(self, samples):
        if not samples:
            return None
        sums, count = {}, 0
        model = self.critic if self.critic is not None else self.model
        for start in range(0, len(samples), self.config.value_batch_size):
            batch = samples[start:start + self.config.value_batch_size]
            _, metrics = value_only_loss(model, batch, self.config, through_backbone=self.critic is not None or None)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.) + float(value) * len(batch)
            count += len(batch)
        return {key: value / count for key, value in sums.items()}

    @torch.no_grad()
    def measure(self, samples):
        sums, counts = {}, {}
        for indices in decision_batches(samples, range(len(samples)), max_decisions=self.config.max_decisions,
                                        max_candidates=self.config.max_candidates):
            batch = [samples[i] for i in indices]
            _, metrics = search_loss(self.model, self.anchor_model, batch, self.config, anchor_scale=self.anchor_scale,
                                     policy_weight=self.policy_weight)
            if self.critic is not None:
                # The reported critic accuracy must be the critic's, not the
                # policy model's idle value head.
                rows = [ValueSample(s.episode_id, s.index, s.inputs.observation, s.value_target, s.outcome, False) for s in batch]
                _, critic_metrics = value_only_loss(self.critic, rows, self.config, through_backbone=True)
                metrics["outcome_mse"] = critic_metrics["value_only_outcome_mse"]
                metrics["value_loss"] = critic_metrics["value_only_loss"]
            _accumulate_metrics(sums, counts, metrics, len(batch))
        return _metric_means(sums, counts) if sums else None

    def update(self, games):
        if self.failed:
            raise RuntimeError("Failed learner must be restored from a completed checkpoint")
        games = [g for g in games if g.end is not None and g.end.status == "terminated"]
        require_unique_episodes(games)
        eligible, decisions = [], 0
        for game in games:
            if (game.episode_id in self.legacy_consumed_episode_ids
                    or self.consumed_episode_keys.intersection(episode_evidence_keys(game))):
                raise ContractError("Episode already consumed by this learner")
            own = [i for i in game.searched_rows if game.rows[i].side == self.identity.side
                   and game.rows[i].behavior == self.identity]
            if own:
                eligible.append(game)
                decisions += len(own)
        if not eligible:
            return dict(status="no_training_data", side=self.identity.side, optimizer_steps=0, decisions=0)
        value_lambda = self.active_value_lambda
        anchor_scale = self.anchor_scale
        policy_weight = self.policy_weight
        # Whole-game holdout: never fitted, measured before and after. With a
        # tiny cohort the holdout is empty and only the in-sample probe exists.
        holdout_count = int(len(eligible) * self.config.holdout_fraction)
        if holdout_count >= len(eligible):
            holdout_count = len(eligible) - 1
        order = self.rng.permutation(len(eligible))
        holdout = [eligible[int(i)] for i in order[:holdout_count]]
        fitted = [eligible[int(i)] for i in order[holdout_count:]]
        holdout_samples = []
        for game in holdout:
            # Only the first measure_decisions held-out decisions are measured. Loading every
            # held-out game first held them all in memory at once (each sample keeps its
            # game's arrays alive): 230 games = 46 GB, which is what made a 2,300-game
            # policy fit commit 30+ GB and ran the machine out of memory on 2026-10-04.
            if len(holdout_samples) >= self.config.measure_decisions:
                break
            holdout_samples.extend(game_samples(game, load_search_arrays(game), self.identity, value_lambda,
                                                credit_gate=self.config.credit_gate,
                                                credit_members=self.config.credit_members,
                                                credit_calibration=self.config.credit_calibration,
                                                credit_scale=self.config.credit_scale))
        holdout_samples = holdout_samples[:self.config.measure_decisions]
        probe = next(self._chunks(fitted))[:self.config.measure_decisions]
        # Value-only pool: afterstates of the fitted games (same return as the
        # decision) plus pre-states and afterstates of replay games with
        # Monte-Carlo outcome targets. Held-out games' afterstates are measured.
        value_pool, replay_games = [], []
        # With a private critic the policy model's value head is idle, so the
        # fitted games' pre-decision states join the value pool explicitly.
        for game in (fitted if self.config.value_pool else ()):
            value_pool.extend(value_samples(game, load_search_arrays(game), self.identity, value_lambda,
                                            include_prestates=self.critic is not None))
        if self.replay_source is not None:
            current_evidence = set().union(*(episode_evidence_keys(g) for g in eligible))
            replay_games = [g for g in self.replay_source()
                            if not current_evidence.intersection(episode_evidence_keys(g))]
            for game in replay_games:
                value_pool.extend(value_samples(game, load_search_arrays(game), self.identity, 1., include_prestates=True))
        corpus_rows = corpus_value_samples(self.corpus_source, self.config.corpus_replay_rows, self.rng)
        value_pool.extend(corpus_rows)
        holdout_values = []
        for game in (holdout if self.config.value_pool else ()):
            holdout_values.extend(value_samples(game, load_search_arrays(game), self.identity, value_lambda, include_prestates=False))
        holdout_values = holdout_values[:self.config.measure_decisions]
        before = dict(train=self.measure(probe), holdout=self.measure(holdout_samples),
                      holdout_afterstates=self.measure_values(holdout_values))
        start_steps = self.optimizer_steps
        last, epochs_completed, contributing = {}, 0, 0
        value_order = self.rng.permutation(len(value_pool)) if value_pool else np.zeros(0, np.int64)
        value_cursor = 0
        # With a private critic the policy loss carries no value term.
        policy_config = replace(self.config, value_coefficient=0.) if self.critic is not None else self.config
        try:
            for _ in range(self.config.epochs):
                for samples in self._chunks(fitted):
                    order = self.rng.permutation(len(samples))
                    for start in range(0, len(order), self.config.max_decisions):
                        logical = order[start:start + self.config.max_decisions]
                        informative_count = sum(samples[int(i)].target_kl >= self.config.min_target_kl for i in logical)
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.critic_optimizer is not None:
                            self.critic_optimizer.zero_grad(set_to_none=True)
                        metric_sums, metric_counts = {}, {}
                        for indices in decision_batches(samples, logical, max_decisions=self.config.max_decisions,
                                                        max_candidates=self.config.max_candidates):
                            batch = [samples[i] for i in indices]
                            weight = len(indices) / len(logical)
                            loss, metrics = search_loss(self.model, self.anchor_model, batch, policy_config,
                                                        anchor_scale=anchor_scale, policy_weight=policy_weight,
                                                        policy_normalizer=informative_count * weight)
                            (loss * weight).backward()
                            _accumulate_metrics(metric_sums, metric_counts, metrics, len(indices))
                            del loss, metrics
                        averaged = _metric_means(metric_sums, metric_counts)
                        if len(value_pool):
                            if value_cursor + self.config.value_batch_size > len(value_order):
                                value_order = self.rng.permutation(len(value_pool))
                                value_cursor = 0
                            slice_ = value_order[value_cursor:value_cursor + self.config.value_batch_size]
                            value_cursor += self.config.value_batch_size
                            rows = [value_pool[int(i)] for i in slice_]
                            if self.critic is not None:
                                loss, metrics = value_only_loss(self.critic, rows, self.config,
                                                                through_backbone=self.config.critic_train_backbone)
                                if self.critic_optimizer is not None:
                                    loss.backward()
                            else:
                                loss, metrics = value_only_loss(self.model, rows, self.config)
                                (self.config.value_coefficient * loss).backward()
                            for key, value in metrics.items():
                                averaged[key] = float(value.detach())
                            del loss, metrics
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm,
                                                             error_if_nonfinite=True)
                        self.optimizer.step()
                        if self.critic_optimizer is not None:
                            torch.nn.utils.clip_grad_norm_([p for p in self.critic.parameters() if p.requires_grad],
                                                           self.config.max_gradient_norm, error_if_nonfinite=True)
                            self.critic_optimizer.step()
                        self.optimizer_steps += 1
                        contributing += len(logical)
                        last = dict(averaged, gradient_norm=float(norm))
                epochs_completed += 1
                hook = getattr(self, "epoch_hook", None)
                if hook is not None:
                    hook(epochs_completed)
            after = dict(train=self.measure(probe), holdout=self.measure(holdout_samples),
                         holdout_afterstates=self.measure_values(holdout_values))
            if not all(math.isfinite(v) for part in after.values() if part for v in part.values()):
                raise FloatingPointError("Nonfinite post-update learner metrics")
            self.updates += 1
            self.consumed_episode_ids.update(g.episode_id for g in eligible)
            self.consumed_episode_keys.update(set().union(*(episode_evidence_keys(g) for g in eligible)))
            self.identity = BehaviorPolicy(self.identity.side,
                                           actor_digest(self.model.state_dict(),
                                                        None if self.critic is None else self.critic.state_dict()),
                                           self.identity.generation + 1, "learner")
        except Exception:
            self.failed = True
            raise
        return dict(status="updated", side=self.identity.side, generation=self.identity.generation,
                    decisions=decisions, episodes=len(eligible), fitted_games=len(fitted), holdout_games=len(holdout),
                    holdout_decisions=len(holdout_samples), optimizer_steps=self.optimizer_steps - start_steps,
                    epochs_completed=epochs_completed, contributing_decisions=contributing,
                    value_lambda=value_lambda, value_through_backbone=self.config.value_through_backbone,
                    anchor_scale=anchor_scale, policy_weight=policy_weight, min_target_kl=self.config.min_target_kl,
                    value_rows=len(value_pool), replay_games=len(replay_games), corpus_rows=len(corpus_rows),
                    before=before, after=after, last_minibatch=last)

    def save_snapshot(self, destination):
        """A checkpoint of the weights as they are in the middle of an update
        (epoch_hook): saved under the identity those weights would have, the
        learner's own identity is left as it is."""
        kept = self.identity
        self.identity = BehaviorPolicy(kept.side, actor_digest(self.model.state_dict(),
                                                               None if self.critic is None else self.critic.state_dict()),
                                       kept.generation + 1, "learner")
        try:
            self.save_checkpoint(destination)
        finally:
            self.identity = kept

    def save_checkpoint(self, destination):
        if self.failed:
            raise RuntimeError("Refusing to save a failed learner as a resumable update")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        payload = dict(schema="search_target_learner_v1", contract_id=CONTRACT.identity, config=asdict(self.config),
            behavior=asdict(self.identity), source_sha256=self.source_sha256, anchor_sha256=self.anchor_sha256,
            model_state={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            optimizer_state=self.optimizer.state_dict(), rng=self.rng.bit_generator.state,
            consumed_episode_ids=sorted(self.consumed_episode_ids), updates=self.updates, optimizer_steps=self.optimizer_steps,
            consumed_episode_keys=sorted(self.consumed_episode_keys),
            legacy_consumed_episode_ids=sorted(self.legacy_consumed_episode_ids),
            critic_state=None if self.critic is None else {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
            critic_scalar_names=critic_scalar_names(self.critic),
            critic_optimizer_state=None if self.critic_optimizer is None else self.critic_optimizer.state_dict())
        path = destination / "state.pt"
        torch.save(payload, path)
        receipt = dict(schema="search_target_learner_completion_v1", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (destination / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, ai, destination, side, *, device="cpu", anchor: PolicyActor | None = None):
        destination = Path(destination)
        receipt = json.loads((destination / "complete.json").read_text(encoding="utf-8"))
        path = destination / "state.pt"
        if receipt["schema"] != "search_target_learner_completion_v1" or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ContractError("Incomplete or modified learner checkpoint")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["schema"] != "search_target_learner_v1" or payload["contract_id"] != CONTRACT.identity:
            raise ContractError("Wrong learner checkpoint algorithm/contract")
        identity = BehaviorPolicy(**payload["behavior"])
        if identity.side != side:
            raise ContractError("Learner checkpoint belongs to another side")
        if payload["anchor_sha256"] is not None and (anchor is None or anchor.identity.weights_sha256 != payload["anchor_sha256"]):
            raise ContractError("Learner checkpoint was trained against a different anchor policy")
        with torch.random.fork_rng(devices=[]):
            model = ai.WOTRValueNet(CONTRACT.observation_width, side=side)
            model.value_head[-1] = torch.nn.Tanh()
            _install_chance_head(model)
            _install_credit_head_from_state(ai, model, payload["model_state"])
            _install_kind_bias_from_state(model, payload["model_state"])
            model.load_state_dict(payload["model_state"], strict=True)
        critic = None
        if payload.get("critic_state") is not None:
            critic = build_critic_model(ai, side, payload["critic_state"], scalar_names=payload.get("critic_scalar_names"))
            critic.load_state_dict(payload["critic_state"], strict=True)
        actor = PolicyActor(model, identity, source_sha256=payload["source_sha256"], critic=critic)
        learner = cls(actor, SearchLearnerConfig(**payload["config"]), anchor=anchor, device=device, critic=critic)
        learner.optimizer.load_state_dict(payload["optimizer_state"])
        if learner.critic_optimizer is not None and payload.get("critic_optimizer_state") is not None:
            saved = payload["critic_optimizer_state"]
            saved_params = sum(len(g["params"]) for g in saved.get("param_groups", []))
            live_params = sum(len(g["params"]) for g in learner.critic_optimizer.param_groups)
            if saved_params == live_params:
                learner.critic_optimizer.load_state_dict(saved)
            # else: the trainable set changed (e.g. backbone frozen); fresh moments.
        learner.rng.bit_generator.state = payload["rng"]
        learner.consumed_episode_ids = set(payload["consumed_episode_ids"])
        learner.consumed_episode_keys = set(payload.get("consumed_episode_keys", ()))
        learner.legacy_consumed_episode_ids = set(payload.get(
            "legacy_consumed_episode_ids", payload["consumed_episode_ids"] if "consumed_episode_keys" not in payload else ()))
        learner.updates, learner.optimizer_steps = payload["updates"], payload["optimizer_steps"]
        return learner
