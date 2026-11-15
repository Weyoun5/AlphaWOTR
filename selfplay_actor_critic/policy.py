"""Direct categorical inference using the existing trained spatial scorer.

Actors receive encoded inputs only. No game state, search, opponent network,
optimizer, or reward-profile code is reachable through this interface.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .compatibility import assess_legacy_artifact
from .contracts import CONTRACT, BehaviorPolicy, ContractError, canonical_json
from .probability_schema import INDEX as PROBABILITY_INDEX


@dataclass(frozen=True)
class ActorInput:
    observation: np.ndarray
    candidates: np.ndarray
    regions: np.ndarray
    probabilities: np.ndarray
    # The engine kind of each candidate (its action_type: army_move, nazgul_move, muster_minion, siege_press ...),
    # finer than the encoding's 14 categories. Required by a policy whose kind bias is not all zero.
    kinds: tuple = None


@dataclass(frozen=True)
class Prediction:
    log_probabilities: np.ndarray
    value: float

    def sample(self, rng: np.random.Generator, temperature: float = 1.) -> int:
        """temperature 1 = the recorded distribution; 0 = argmax (greedy);
        otherwise softmax(log p / temperature). Collection uses 1."""
        if temperature == 1.:
            return int(rng.choice(len(self.log_probabilities), p=np.exp(self.log_probabilities)))
        if temperature <= 0.:
            return int(np.argmax(self.log_probabilities))
        scaled = self.log_probabilities / temperature
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        return int(rng.choice(len(weights), p=weights / weights.sum()))


def weights_digest(state_dict, contract_identity=None) -> str:
    """Once per immutable policy snapshot, never once per action.
    contract_identity: the contract the weights belong to (default: the running one; a net of an older generation
    admitted into this engine - D6P-MIXED-GEN-ADAPTER - is fingerprinted under its own generation's contract)."""
    h = hashlib.sha256((CONTRACT.identity if contract_identity is None else contract_identity).encode("ascii"))
    for key, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(canonical_json([key, str(array.dtype), list(array.shape)]).encode("ascii"))
        h.update(array.tobytes())
    return h.hexdigest()


def actor_digest(model_state, critic_state=None, contract_identity=None) -> str:
    """Identity of an actor's BEHAVIOUR: the policy weights, and the private
    critic's weights when one is attached (it decides what the search plays).
    Without a critic this is exactly weights_digest(model_state)."""
    if critic_state is None:
        return weights_digest(model_state, contract_identity)
    h = hashlib.sha256(b"actor_with_private_critic_v1")
    h.update(weights_digest(model_state, contract_identity).encode("ascii"))
    h.update(weights_digest(critic_state, contract_identity).encode("ascii"))
    return h.hexdigest()


# Action-type one-hot at the front of every candidate encoding (src/ai.py
# ACTION_TYPE_CATEGORIES, checked on install). Every engine action kind maps
# to exactly one category (e.g. eye_allocate -> eye_allocate, nazgul_move ->
# army_move, card_option -> play_event).
ACTION_CATEGORIES = ("army_move", "army_attack", "advance_political", "muster", "draw_card", "play_event",
                     "fellowship_move", "fellowship_hide", "fellowship_declare", "special_character", "pass",
                     "eye_allocate", "other", "combat_card")


class KindBias(torch.nn.Module):
    """One logit offset per ENGINE action kind (WP5, 2026-10-05).

    pg_fit's per-kind regressors (one-hot of the played kind minus the policy's kind mass) are the exact score
    function of this parameter, so a fitted per-kind step can be written into the policy. The vocabulary of
    kinds travels inside the state dict (as bytes), so the seed format, the digest and the loaders need no
    new field. Zero-initialised: installing it changes no logit. Kinds outside the vocabulary get 0."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = tuple(str(k) for k in vocab)
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("kind vocabulary has duplicates")
        self.index = {k: i for i, k in enumerate(self.vocab)}
        self.bias = torch.nn.Parameter(torch.zeros(len(self.vocab)))
        self.register_buffer("vocab_bytes", torch.tensor(list(json.dumps(list(self.vocab)).encode("utf-8")), dtype=torch.uint8))

    def active(self) -> bool:
        return bool(self.bias.detach().abs().max() > 0) if len(self.vocab) else False

    def extended(self, kinds):
        """A copy whose vocabulary also holds `kinds` (new ones at zero)."""
        new = KindBias(self.vocab + tuple(k for k in dict.fromkeys(str(k) for k in kinds) if k not in self.index))
        with torch.no_grad():
            new.bias[:len(self.vocab)] = self.bias.detach().cpu()
        return new.to(self.bias.device)

    def forward(self, kinds):
        """kinds: sequence of action_type strings, one per candidate -> offset [N]."""
        index = torch.tensor([self.index.get(k, -1) for k in kinds], dtype=torch.long, device=self.bias.device)
        return torch.where(index >= 0, self.bias[index.clamp(min=0)], torch.zeros((), dtype=self.bias.dtype, device=self.bias.device))

    @staticmethod
    def from_state(state: dict, prefix="kind_bias."):
        vocab = json.loads(bytes(state[prefix + "vocab_bytes"].cpu().numpy().tolist()).decode("utf-8"))
        return KindBias(vocab)


def install_kind_bias(model, vocab):
    """Attach a per-kind bias to a policy network (it becomes part of its state); extends an existing one."""
    present = getattr(model, "kind_bias", None)
    model.kind_bias = KindBias(vocab) if present is None else present.extended(vocab)
    return model.kind_bias


class KindContextBias(torch.nn.Module):
    """Per-kind logit offsets CONDITIONED on the position (kind_context.py; Codex's representation experiment,
    2026-10-11): offset(candidate) = bias[kind] . rows(context of the side's own observation), WIDTH numbers per
    kind (the plain offset and one per context level). The vocabulary, the side (the context reads side-relative
    slots) and the context spec travel inside the state dict. Zero-initialised: installing it changes no logit."""

    def __init__(self, vocab, side):
        super().__init__()
        from .kind_context import SPEC, WIDTH
        self.vocab = tuple(str(k) for k in vocab)
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("kind vocabulary has duplicates")
        if side not in ("fp", "sh"):
            raise ValueError("side must be 'fp' or 'sh'")
        self.index = {k: i for i, k in enumerate(self.vocab)}
        self.side, self.spec = side, SPEC
        self.bias = torch.nn.Parameter(torch.zeros(len(self.vocab), WIDTH))
        meta = json.dumps(dict(vocab=list(self.vocab), side=side, spec=SPEC))
        self.register_buffer("meta_bytes", torch.tensor(list(meta.encode("utf-8")), dtype=torch.uint8))

    def active(self) -> bool:
        return bool(self.bias.detach().abs().max() > 0) if len(self.vocab) else False

    def extended(self, kinds):
        """A copy whose vocabulary also holds `kinds` (new ones at zero)."""
        new = KindContextBias(self.vocab + tuple(k for k in dict.fromkeys(str(k) for k in kinds) if k not in self.index), self.side)
        with torch.no_grad():
            new.bias[:len(self.vocab)] = self.bias.detach().cpu()
        return new.to(self.bias.device)

    def contexts(self, observations):
        """Regressor rows [N, WIDTH] of observation rows in this side's view (numpy)."""
        from .kind_context import levels, rows
        return rows(levels(observations, self.side))

    def forward(self, kinds, context_rows):
        """kinds: one action_type per candidate; context_rows: [N, WIDTH] tensor, the row of each candidate's
        decision -> offset [N] (0 for kinds outside the vocabulary)."""
        index = torch.tensor([self.index.get(k, -1) for k in kinds], dtype=torch.long, device=self.bias.device)
        picked = self.bias[index.clamp(min=0)] * context_rows.to(self.bias.dtype)
        return torch.where(index >= 0, picked.sum(-1), torch.zeros((), dtype=self.bias.dtype, device=self.bias.device))

    @staticmethod
    def from_state(state: dict, prefix="kind_ctx_bias."):
        from .kind_context import SPEC
        meta = json.loads(bytes(state[prefix + "meta_bytes"].cpu().numpy().tolist()).decode("utf-8"))
        if meta["spec"] != SPEC:
            raise ContractError(f"This policy's kind context is {meta['spec']}; this code computes {SPEC}")
        return KindContextBias(meta["vocab"], meta["side"])


def install_kind_ctx_bias(model, vocab, side):
    """Attach a context-conditioned per-kind bias to a policy network; extends an existing one."""
    present = getattr(model, "kind_ctx_bias", None)
    if present is not None and present.side != side:
        raise ContractError("The policy's kind context bias belongs to the other side")
    model.kind_ctx_bias = KindContextBias(vocab, side) if present is None else present.extended(vocab)
    return model.kind_ctx_bias


def _install_kind_bias_from_state(model, state):
    if any(k.startswith("kind_bias.") for k in state):
        model.kind_bias = KindBias.from_state(state)
    if any(k.startswith("kind_ctx_bias.") for k in state):
        model.kind_ctx_bias = KindContextBias.from_state(state)


class CreditHead(torch.nn.Module):
    """Additive logit correction that carries validated decision credit.

    It is the only trained part when credit is put into the policy: the
    imitation network stays frozen. The correction is computed from explicit
    game-state features (the race read-out of the observation) and the
    candidate's encoding, and it is MASKED by action category: for a candidate
    of any other category it is exactly zero, so credit for one kind of
    decision cannot change the policy anywhere else. (The network's own
    policy_delta head could learn the credit but leaked small corrections
    onto unrelated decisions and changed the best move in ~5% of them.)
    Starts at zero: a freshly installed head leaves the policy unchanged."""

    def __init__(self, layout, categories, *, action_width, hidden=64):
        super().__init__()
        from .race_critic import RaceFeatures
        unknown = [c for c in categories if c not in ACTION_CATEGORIES]
        if unknown or not categories:
            raise ValueError("credit head categories must be action categories: " + str(unknown))
        self.features = RaceFeatures(layout)
        gate = torch.zeros(len(ACTION_CATEGORIES))
        for c in categories:
            gate[ACTION_CATEGORIES.index(c)] = 1.
        self.register_buffer("gate", gate)
        self.body = torch.nn.Sequential(torch.nn.Linear(layout.feature_count + action_width, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.out = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    @property
    def categories(self):
        return tuple(c for c, g in zip(ACTION_CATEGORIES, self.gate.tolist()) if g > 0)

    def forward(self, observations, owner, encodings):
        """observations [B, W], owner [N] (decision of each candidate), encodings [N, A] -> correction [N]."""
        out = torch.zeros(len(encodings), dtype=encodings.dtype, device=encodings.device)
        index = torch.nonzero(encodings[:, :len(ACTION_CATEGORIES)] @ self.gate > 0).reshape(-1)
        if len(index):
            state = self.features(observations)[owner[index]]
            out = out.index_add(0, index, self.out(self.body(torch.cat((state, encodings[index]), dim=-1))).reshape(-1))
        previous = getattr(self, "previous", None)
        if previous is not None:
            out = out + previous(observations, owner, encodings)
        return out

    @staticmethod
    def from_state(state: dict, prefix="credit_head."):
        from .race_critic import RaceFeatures
        layout = RaceFeatures.from_state(state, prefix=prefix + "features.").layout()
        gate = state[prefix + "gate"]
        hidden, width = state[prefix + "body.0.weight"].shape
        categories = tuple(c for c, g in zip(ACTION_CATEGORIES, gate.tolist()) if g > 0)
        head = CreditHead(layout, categories, action_width=int(width) - layout.feature_count, hidden=int(hidden))
        if any(key.startswith(prefix + "previous.") for key in state):
            head.previous = CreditHead.from_state(state, prefix=prefix + "previous.")
        return head


def install_credit_head(ai, model, head):
    """Attach a credit head to a policy network (it becomes part of its state)."""
    if tuple(getattr(ai, "ACTION_TYPE_CATEGORIES", ACTION_CATEGORIES)) != ACTION_CATEGORIES:
        raise ContractError("Action categories of the encoder differ from the credit head's")
    model.credit_head = head
    return model


def _install_credit_head_from_state(ai, model, state):
    """Loaders call this before load_state_dict: a saved policy that carries a
    credit head gets the same head installed."""
    if any(k.startswith("credit_head.") for k in state):
        install_credit_head(ai, model, CreditHead.from_state(state))


def forward_ragged(model, inputs: list[ActorInput], *, max_action_rows=16384, detach_value=False, return_delta=False):
    """Shared differentiable scoring for collection and learning.

    Caller selects autograd mode. Model stays in eval mode in both paths:
    Dropout/BatchNorm must not change a policy just because it is being trained.
    Returns values and one complete-menu logit tensor per decision.
    detach_value=True computes the value from detached backbone features, so a
    value loss trains the value head only and never the shared representation.
    return_delta=True also returns, per decision, the policy_delta head's
    additive correction alone (computed from DETACHED features, so a loss on
    it trains that head only and never the imitation policy).
    """
    # D6P-MIXED-GEN-ADAPTER: a net of an older generation admitted into this engine (src/mixed_generation.py) reads
    # its own generation's prefix of every input, before any module (credit head, kind biases included) sees it.
    view = getattr(model, "view_actor_input", None)
    if view is not None:
        inputs = [view(item) for item in inputs]
    device = next(model.parameters()).device
    counts = [len(item.candidates) for item in inputs]
    observations = torch.from_numpy(np.stack([item.observation for item in inputs])).to(device)
    backbone, nodes = model.forward_backbone(observations)
    values = model.value_head(backbone.detach() if detach_value else backbone).reshape(-1)
    encodings = np.concatenate([item.candidates for item in inputs])
    regions = np.concatenate([item.regions for item in inputs])
    probabilities = np.concatenate([item.probabilities for item in inputs])
    owners = np.repeat(np.arange(len(inputs), dtype=np.int64), counts)
    means = nodes.mean(dim=1)
    kind_bias = getattr(model, "kind_bias", None)
    ctx_bias = getattr(model, "kind_ctx_bias", None)
    kinds = None
    # A zero offset is still a trainable parameter: excluding it from the
    # graph would prevent a newly installed head from ever learning. Legacy
    # inputs without kind metadata remain usable while both heads are zero.
    known_kinds = all(item.kinds is not None and len(item.kinds) == len(item.candidates) for item in inputs)
    train_kinds = torch.is_grad_enabled() and known_kinds
    use_kind = kind_bias is not None and (kind_bias.active() or (
        train_kinds and kind_bias.bias.requires_grad and bool(kind_bias.vocab)))
    use_context = ctx_bias is not None and (ctx_bias.active() or (
        train_kinds and ctx_bias.bias.requires_grad and bool(ctx_bias.vocab)))
    if use_kind or use_context:
        if not known_kinds:
            raise ContractError("This policy carries a per-kind bias: every ActorInput must name its candidates' kinds")
        kinds = [k for item in inputs for k in item.kinds]
    context_rows = None
    if use_context:
        # the context of every decision from the side's OWN observation (the one the policy is scoring)
        context_rows = torch.from_numpy(ctx_bias.contexts(np.stack([item.observation for item in inputs]))).to(device)
    scores, deltas = [], []
    for start in range(0, len(encodings), max_action_rows):
        stop = start + max_action_rows
        features = torch.from_numpy(encodings[start:stop]).to(device)
        region = torch.from_numpy(regions[start:stop]).to(device)
        owner = torch.from_numpy(owners[start:stop]).to(device)
        action_features, cards = model._embed_action_encs(features)
        spatial = torch.where((region < 0).unsqueeze(-1), means[owner], nodes[owner, region.clamp(min=0)])
        combined = torch.cat((backbone[owner], spatial, action_features, cards), dim=-1)
        logits = model.spatial_action_scorer(combined).reshape(-1)
        logits = logits + model.policy_delta(combined).reshape(-1)
        if return_delta:
            deltas.append(model.policy_delta(combined.detach()).reshape(-1))
        chance = probabilities[start:stop]
        active = np.flatnonzero(chance[:, PROBABILITY_INDEX["combat_available"]]
                               + chance[:, PROBABILITY_INDEX["hunt_available"]])
        if len(active):
            index = torch.from_numpy(active).to(device)
            chance = torch.from_numpy(chance[active]).to(device)
            adjustment = model.chance_policy(torch.cat((combined[index], chance), dim=-1)).reshape(-1)
            logits = logits.index_add(0, index, adjustment)
        credit = getattr(model, "credit_head", None)
        if credit is not None:
            logits = logits + credit(observations, owner, features)
        if use_kind:
            logits = logits + kind_bias(kinds[start:stop]).to(logits.dtype)
        if context_rows is not None:
            logits = logits + ctx_bias(kinds[start:stop], context_rows[owner]).to(logits.dtype)
        scores.append(logits)
    if return_delta:
        return values, torch.cat(scores).split(counts), torch.cat(deltas).split(counts)
    return values, torch.cat(scores).split(counts)


class PolicyActor:
    def __init__(self, model, identity: BehaviorPolicy, *, source_sha256=None, max_action_rows=16384, critic=None):
        if max_action_rows < 1:
            raise ValueError("max_action_rows must be positive")
        self.model = model.eval()
        self.identity = identity
        self.source_sha256 = source_sha256
        self.device = next(model.parameters()).device
        self.max_action_rows = max_action_rows
        # Optional PRIVATE critic network (own backbone + value head, trained
        # on corpus and self-play outcomes, never on policy targets). When
        # present it supplies every value the search and collector record;
        # the policy model's own value head is then unused.
        self.critic = None if critic is None else critic.eval()

    @torch.inference_mode()
    def value_of(self, observations) -> torch.Tensor:
        """Values of raw observations (numpy [N, W]) from the critic if present."""
        x = torch.from_numpy(np.asarray(observations, np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        model = self.critic if self.critic is not None else self.model
        hidden, _ = model.forward_backbone(x)
        return model.value_head(hidden).reshape(-1)

    @torch.inference_mode()
    def member_values_of(self, observations):
        """[members, N] values when the critic is an ensemble (ensemble_critic.py), else None."""
        members = getattr(self.critic, "member_values", None)
        if members is None:
            return None
        x = torch.from_numpy(np.asarray(observations, np.float32)).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return members(x)

    @torch.inference_mode()
    def predict_many(self, inputs: list[ActorInput]) -> list[Prediction]:
        """One backbone batch, ragged action scoring, full-menu normalization.

        Chunking bounds temporary GPU memory; it never truncates a legal menu.
        All GPU outputs cross to the host together after scoring the batch.
        """
        if not inputs:
            return []
        counts = [len(item.candidates) for item in inputs]
        values, scores = forward_ragged(self.model, inputs, max_action_rows=self.max_action_rows)
        if self.critic is not None:
            values = self.value_of(np.stack([item.observation for item in inputs]))
        host = torch.cat([values, *scores]).cpu().numpy()
        results, offset = [], len(inputs)
        for i, count in enumerate(counts):
            logits = host[offset:offset + count].astype(np.float64)
            shifted = logits - logits.max()
            log_probabilities = shifted - np.log(np.exp(shifted).sum())
            results.append(Prediction(log_probabilities, float(host[i])))
            offset += count
        return results

    def predict(self, inputs: ActorInput) -> Prediction:
        return self.predict_many([inputs])[0]


def _install_chance_head(model):
    """Zero contribution at warm start, with trainable context-sensitive weights."""
    width = model.spatial_action_scorer[0].in_features + CONTRACT.probability_width
    model.chance_policy = torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    torch.nn.init.zeros_(model.chance_policy[-1].weight)
    torch.nn.init.zeros_(model.chance_policy[-1].bias)


# D6P-MIXED-GEN-ADAPTER (D6 plan S1(e), section 6): a net of an OLDER encoding generation inside a newer engine (a
# deployed v52 net in a v53 evaluation). src/mixed_generation.py owns the generation table and the view net; these
# helpers give it the actor-critic contract side. Only reached after the running contract refused a checkpoint, and
# only under the explicit opt-in WOTR_ADMIT_GENERATIONS: without it every loader refuses exactly as before.
def generation_contract(spec):
    """The actor-critic contract record of generation `spec` (a mixed_generation.GenerationSpec): the running contract
    with its five generation-keyed fields (contracts.Contract: observation, observation_width, actions, action_width,
    ai_policy) set to that generation's values."""
    record = CONTRACT.to_dict()
    record.update(observation=spec.observation, observation_width=spec.observation_width, actions=spec.actions,
                  action_width=spec.action_width, ai_policy=spec.ai_policy)
    return record


def generation_contract_identity(spec) -> str:
    """contracts.Contract.identity as a process of generation `spec` computes it."""
    return hashlib.sha256(canonical_json(generation_contract(spec)).encode("ascii")).hexdigest()


def _admitted_view(ai, stamp, where):
    """The older generation a checkpoint is played through in this engine, or None (mixed_generation.checkpoint_view;
    the opt-in is read before anything is imported)."""
    if not os.environ.get("WOTR_ADMIT_GENERATIONS", "").strip():
        return None
    import mixed_generation  # src/, beside the engine module `ai` the caller imported
    return mixed_generation.checkpoint_view(ai, stamp, where)


def _build_view_net(ai, view, side):
    import mixed_generation
    return mixed_generation.build_view_net(ai, view, side)


def _seed_contract_stamp(contract):
    """An actor seed's contract record as the checkpoint stamp mixed_generation reads."""
    return {"observation_contract": contract.get("observation"), "action_encoding_contract": contract.get("actions"),
            "ai_policy_contract": contract.get("ai_policy"), "input_dim": contract.get("observation_width"),
            "action_dim": contract.get("action_width")}


def _view_critic(ai, view, side, critic_state):
    """A private critic of an admitted older generation: plain value networks only (built at that generation's widths,
    reading the prefix like the policy). Race, full-state and ensemble critics read layouts of their own and refuse.
    Weights as constructed, like build_critic_model: the caller loads critic_state (load_actor_seed, strictly)."""
    from .ensemble_critic import EnsembleCritic
    from .full_critic import FullStateCritic
    from .race_critic import RaceCritic
    if (EnsembleCritic.is_ensemble_state(critic_state) or RaceCritic.is_race_state(critic_state)
            or FullStateCritic.is_full_state(critic_state)):
        raise ContractError("Only a plain value-network critic can play across generations (D6P-MIXED-GEN-ADAPTER)")
    with torch.random.fork_rng(devices=[]):
        model = _build_view_net(ai, view, side)
        model.value_head[-1] = torch.nn.Tanh()
        _install_chance_head(model)
    return model


def load_policy_seed(ai, path, side, *, device="cpu", generation=0, origin="learner", critic_seed=52) -> PolicyActor:
    """Strict load once; retain all policy weights and replace the old critic.

    This is a warm start, never a resume. No optimizer state is imported.
    The final critic layer starts at zero; its activation is an ordinary tanh.
    A checkpoint of an older generation admitted by WOTR_ADMIT_GENERATIONS loads at its own widths and reads the
    prefix of this engine's inputs (D6P-MIXED-GEN-ADAPTER); its identity uses its own generation's contract.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    owned = checkpoint.get("side") == side
    admitted = owned and assess_legacy_artifact("checkpoint", checkpoint).policy_warm_start
    view = _admitted_view(ai, checkpoint, str(path)) if owned and not admitted else None
    if not admitted and view is None:
        raise ContractError("Incompatible policy seed or wrong checkpoint owner")
    # Construction and critic initialization cannot perturb gameplay or sampler RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(critic_seed)
        model = (ai.WOTRValueNet(CONTRACT.observation_width, side=side) if view is None
                 else _build_view_net(ai, view, side))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        for module in model.value_head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        model.value_head[-1] = torch.nn.Tanh()
        final = [m for m in model.value_head.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        _install_chance_head(model)
    identity = BehaviorPolicy(side, weights_digest(model.state_dict(),
                                                   None if view is None else generation_contract_identity(view)),
                              generation, origin)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return PolicyActor(model.to(device), identity, source_sha256=source_sha)


def save_actor_seed(actor: PolicyActor, path):
    """Export reproducible actor weights, explicitly without trainer/resume state.

    A private critic, when attached, travels with the seed (critic_state) so
    collection workers search with the same critic the controller trained."""
    payload = dict(schema="actor_critic_seed_v1", contract=CONTRACT.to_dict(), contract_id=CONTRACT.identity,
        behavior=asdict(actor.identity), source_sha256=actor.source_sha256,
        model_state={k: v.detach().cpu() for k, v in actor.model.state_dict().items()},
        head_act="tanh", critic_initialization="reset_hidden_zero_final_v1", optimizer_state=None,
        critic_state=None if actor.critic is None else {k: v.detach().cpu() for k, v in actor.critic.state_dict().items()},
        critic_scalar_names=critic_scalar_names(actor.critic))
    from .durability import file_lock
    from .runtime_io import _atomic_write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_name(path.name + ".lock")):
        if path.exists():
            raise FileExistsError(path)

        _atomic_write(path, lambda stream: torch.save(payload, stream))


def build_critic_model(ai, side, critic_state=None, *, scalar_names=None):
    """A critic of the saved state's kind: the race critic (race_critic.py)
    when the state carries its feature buffers, else a full value network of
    the actor's shape (its policy heads unused). Callers load the state."""
    from .ensemble_critic import EnsembleCritic
    from .full_critic import FullStateCritic
    from .race_critic import RaceCritic
    if critic_state is not None and EnsembleCritic.is_ensemble_state(critic_state):
        # Mean of several value functions (ensemble_critic.py): every member is rebuilt as its own kind.
        return EnsembleCritic([build_critic_model(ai, side, part, scalar_names=scalar_names)
                               for part in EnsembleCritic.split_state(critic_state)])
    if critic_state is not None and RaceCritic.is_race_state(critic_state):
        return RaceCritic.from_state(critic_state, side, scalar_names=scalar_names)
    with torch.random.fork_rng(devices=[]):
        model = ai.WOTRValueNet(CONTRACT.observation_width, side=side)
        model.value_head[-1] = torch.nn.Tanh()
        _install_chance_head(model)
    if critic_state is not None and FullStateCritic.is_full_state(critic_state):
        # Full-state value network with auxiliary heads (full_critic.py).
        return FullStateCritic.from_state(model, critic_state, side, scalar_names=scalar_names)
    return model


def critic_scalar_names(critic):
    """The race read-out's scalar feature names (not tensors, so payloads carry them)."""
    from .ensemble_critic import EnsembleCritic
    from .full_critic import FullStateCritic
    from .race_critic import RaceCritic
    if isinstance(critic, EnsembleCritic):
        names = {critic_scalar_names(m) for m in critic.members} - {None}
        if len(names) > 1:
            raise ContractError("Ensemble members read different scalar layouts")
        return names.pop() if names else None
    return tuple(critic.layout.scalar_names) if isinstance(critic, (RaceCritic, FullStateCritic)) else None


def load_critic(ai, path, side, *, device="cpu"):
    """Load a private critic (corpus_critic.py --private, private_critic_v1)
    or a race critic (race_fit.py, race_critic_v1)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") not in ("private_critic_v1", "race_critic_v1", "full_state_critic_v2",
                                     "critic_ensemble_v1") or payload.get("side") != side:
        raise ContractError("Unsupported critic file or wrong side")
    model = build_critic_model(ai, side, payload["model_state"], scalar_names=payload.get("scalar_names"))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_actor_seed(ai, path, side, *, device="cpu") -> PolicyActor:
    """Load the new seed format as-is; never reset its critic a second time.
    A seed of an older generation admitted by WOTR_ADMIT_GENERATIONS loads at its own widths and reads the prefix of
    this engine's inputs (D6P-MIXED-GEN-ADAPTER); its contract and fingerprint are checked as its own generation's."""
    from .contracts import Contract
    payload = torch.load(path, map_location="cpu", weights_only=True)
    recorded, view = payload.get("contract"), None
    if isinstance(recorded, dict) and recorded != CONTRACT.to_dict():
        view = _admitted_view(ai, _seed_contract_stamp(recorded), str(path))
        if view is not None and canonical_json(dict(recorded)) != canonical_json(generation_contract(view)):
            raise ContractError("Actor seed contract differs from its generation's beyond the generation fields")
    if view is None:
        Contract.from_dict(payload["contract"])
    contract_id = CONTRACT.identity if view is None else generation_contract_identity(view)
    if (payload["schema"] != "actor_critic_seed_v1" or payload["contract_id"] != contract_id
            or payload["head_act"] != "tanh" or payload["optimizer_state"] is not None):
        raise ContractError("Unsupported actor seed format")
    identity = BehaviorPolicy(**payload["behavior"])
    if identity.side != side or actor_digest(payload["model_state"], payload.get("critic_state"),
                                             None if view is None else contract_id) != identity.weights_sha256:
        raise ContractError("Actor seed owner or weight fingerprint differs")
    with torch.random.fork_rng(devices=[]):
        model = (ai.WOTRValueNet(CONTRACT.observation_width, side=side) if view is None
                 else _build_view_net(ai, view, side))
        model.value_head[-1] = torch.nn.Tanh()
        _install_chance_head(model)
        _install_credit_head_from_state(ai, model, payload["model_state"])
        _install_kind_bias_from_state(model, payload["model_state"])
        model.load_state_dict(payload["model_state"], strict=True)
    critic = None
    if payload.get("critic_state") is not None:
        critic = (_view_critic(ai, view, side, payload["critic_state"]) if view is not None
                  else build_critic_model(ai, side, payload["critic_state"], scalar_names=payload.get("critic_scalar_names")))
        # Both builders return constructor weights; the saved critic is what the identity above vouches for. An admitted
        # older generation's critic used to skip this and served its random constructor weights (R17-SAC rows 62/64).
        critic.load_state_dict(payload["critic_state"], strict=True)
        critic = critic.to(device).eval().requires_grad_(False)
    return PolicyActor(model.to(device), identity, source_sha256=payload["source_sha256"], critic=critic)
