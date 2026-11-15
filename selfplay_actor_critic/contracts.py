"""Admission-time contracts, independent of the engine and training libraries.

These records describe actual player transitions, not simulations. Validation
does not prove rules legality: the collector must obtain menus from the engine.
No old trace, reward profile, or search visit target is implicitly converted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import os
from typing import Any, Mapping
from .probability_schema import VERSION as PROBABILITY_VERSION, WIDTH as PROBABILITY_WIDTH

# R2-D5-V53 (USER D5 2026-10-22): the engine encoding generation is chosen
# once from WOTR_ACT_GEN (src/ai.py reads it at import).  v53 = the v52
# vectors + appended tails (src/encoding_v53.py).  Every other value keeps the
# historical v52 schema below, so the v52 contract identity is unchanged.
# EngineAdapter cross-checks these against the running engine (loud on any
# mismatch); no v53 data or checkpoint exists until the D6 retrain.
_ENGINE_V53 = os.environ.get("WOTR_ACT_GEN", "").strip().lower() == "v53"


class ContractError(ValueError):
    pass


SIDES = frozenset(("fp", "sh"))
FIXED_REASONS = frozenset((
    "bonus_draw_always_take", "recruit_elite_if_available",
    "combat_victim_priority", "mighty_payer_lowest_leadership", "mordor_gateway_declaration",
))
VICTORY_ROUTES = {"fp": frozenset(("military", "ring_destroyed")),
                  "sh": frozenset(("military", "ring_corruption"))}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _text(value: Any, field: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), field + " must be nonempty text")


def _integer(value: Any, field: str, minimum: int = 0) -> None:
    _require(type(value) is int and value >= minimum, field + " must be an integer >= " + str(minimum))


def _number(value: Any, field: str) -> None:
    try:
        valid = type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, field + " must be finite numeric data")


def _side(value: Any) -> None:
    _require(isinstance(value, str) and value in SIDES, "side must be fp or sh")


def _sha(value: Any, field: str) -> None:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), field + " must be a lowercase SHA256")


def _json_value(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _number(value, "JSON number")
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, "JSON object keys must be strings")
            _json_value(item)
    else:
        raise ContractError("Unsupported JSON value: " + type(value).__name__)


def canonical_json(value: Any) -> str:
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _exact_fields(value: Mapping, fields: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), label + " must be an object")
    _require(set(value) == fields, label + " has missing or unknown fields")


@dataclass(frozen=True)
class Contract:
    schema: str = "wotr_actor_critic_contract_v1"
    observation: str = ("v53_hand_threat_context_v1" if _ENGINE_V53
                        else "v52_observed_combat_round_v3")
    observation_width: int = 4745 if _ENGINE_V53 else 4647
    observation_dtype: str = "float32"
    actions: str = ("v53_identity_route_choices_v1" if _ENGINE_V53
                    else "observed_combat_choices_v8")
    action_width: int = 199 if _ENGINE_V53 else 157
    action_dtype: str = "float32"
    # D6P-POLICY-VERSION (2026-10-23): v53 carries the A2-05 Ring-twin policy
    # version (src/ai_policies.policy_version_for_generation); v52 keeps v3.
    ai_policy: str = ("elite_bonus_draw_combat_victim_mighty_ring_twin_v4" if _ENGINE_V53
                      else "elite_bonus_draw_combat_victim_mighty_v3")
    turn_policy: str = "gateway_declaration_phase_boundaries_v3"
    visibility: str = "player_information_snapshot_v1"
    probabilities: str = PROBABILITY_VERSION
    probability_width: int = PROBABILITY_WIDTH
    probability_dtype: str = "float32"
    rewards: str = "all_legal_wins_zero_sum_v1"
    decisions: str = "actual_player_transitions_v2"
    selector: str = "direct_masked_categorical_v1"
    discount: float = 1.0
    intermediate_reward: float = 0.0
    recurrent: bool = False

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            _require(type(value) is type(field.default) and value == field.default,
                     "Unsupported contract field: " + field.name)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping) -> Contract:
        expected = cls().to_dict()
        _exact_fields(value, set(expected), "contract")
        # Compare canonical data, not Python equality (True == 1).
        _require(canonical_json(dict(value)) == canonical_json(expected), "Unsupported contract; explicit migration required")
        return cls()


CONTRACT = Contract()


def _contract(identity: str) -> None:
    _require(identity == CONTRACT.identity, "Unsupported actor-critic contract identity")


@dataclass(frozen=True)
class ActionSpec:
    """Stable execution-bearing identity; not an encoding or a legality claim."""
    action_type: str
    die_index: int
    params_json: str = "{}"

    def __post_init__(self):
        _text(self.action_type, "action_type")
        _integer(self.die_index, "die_index", -1)
        _require(isinstance(self.params_json, str), "params_json must be text")
        try:
            params = json.loads(self.params_json)
        except (ValueError, TypeError) as exc:
            raise ContractError("Invalid action parameters") from exc
        _require(type(params) is dict, "Action parameters must be an object")
        _require(canonical_json(params) == self.params_json, "Action parameters must use canonical JSON")

    @classmethod
    def create(cls, action_type: str, die_index: int, params: dict) -> ActionSpec:
        return cls(action_type, die_index, canonical_json(params))

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Observation:
    side: str
    features: tuple[float, ...]
    contract_id: str = CONTRACT.identity

    def __post_init__(self):
        _contract(self.contract_id)
        _side(self.side)
        _require(type(self.features) is tuple and len(self.features) == CONTRACT.observation_width,
                 "Observation requires the exact immutable feature vector")
        for value in self.features:
            _number(value, "Observation feature")


@dataclass(frozen=True)
class BehaviorPolicy:
    side: str
    weights_sha256: str
    generation: int
    origin: str  # learner or historical; historical rows never become PPO data

    def __post_init__(self):
        _side(self.side)
        _sha(self.weights_sha256, "weights_sha256")
        _integer(self.generation, "generation")
        _require(type(self.origin) is str and self.origin in ("learner", "historical"), "Unsupported behavior policy origin")


def decision_kind(menu_size: int, fixed_reason: str | None = None) -> str:
    _integer(menu_size, "menu_size", 1)
    if fixed_reason is not None:
        _require(isinstance(fixed_reason, str) and fixed_reason in FIXED_REASONS, "Unknown fixed policy reason")
        _require(menu_size == 1, "A fixed decision cannot discard remaining learned choices")
        return "fixed_policy"
    return "forced" if menu_size == 1 else "learned"


@dataclass(frozen=True)
class DecisionRecord:
    episode_id: str
    transition_index: int  # counts actual player actions; chance is inside the engine interval
    side_index: int        # counts this player's actions, including forced/fixed transitions
    observation: Observation
    menu: tuple[ActionSpec, ...]  # complete menu AFTER approved fixed-policy projection
    selected: str
    behavior: BehaviorPolicy
    behavior_log_probability: float
    behavior_value: float | None
    kind: str
    fixed_reason: str | None = None
    reward: float = 0.0
    schema: str = "wotr_actor_critic_decision_v1"

    def __post_init__(self):
        _require(self.schema == "wotr_actor_critic_decision_v1", "Unsupported decision schema")
        _text(self.episode_id, "episode_id")
        _integer(self.transition_index, "transition_index")
        _integer(self.side_index, "side_index")
        _require(isinstance(self.observation, Observation), "Missing observation contract")
        _require(isinstance(self.behavior, BehaviorPolicy), "Missing behavior policy identity")
        _require(self.behavior.side == self.observation.side, "Behavior policy belongs to another side")
        _require(type(self.menu) is tuple and all(isinstance(a, ActionSpec) for a in self.menu),
                 "Menu must be an immutable sequence of actions")
        expected = decision_kind(len(self.menu), self.fixed_reason)
        _require(self.kind == expected, "Decision kind disagrees with menu or fixed-policy provenance")
        ids = [a.identity for a in self.menu]
        _require(len(set(ids)) == len(ids), "Duplicate actions would change sampling probabilities")
        _sha(self.selected, "selected action identity")
        _require(self.selected in ids, "Selected action is absent from the supplied menu")
        _number(self.behavior_log_probability, "behavior_log_probability")
        _require(self.behavior_log_probability <= 0, "Log probability must be <= 0")
        if expected != "learned":
            _require(self.behavior_log_probability == 0, "Deterministic transitions must have probability one")
        else:
            _require(self.behavior_value is not None, "Learned decisions require the pre-decision value estimate")
        if self.behavior_value is not None:
            _number(self.behavior_value, "behavior_value")
            _require(-1 <= self.behavior_value <= 1, "Value uses the [-1,1] win/loss scale")
        _number(self.reward, "reward")
        _require(self.reward == 0, "Intermediate shaping and per-decision penalties are forbidden")

    @property
    def policy_candidate(self) -> bool:
        """Still requires freshness, rollout-integrity and generation admission."""
        return self.kind == "learned" and self.behavior.origin == "learner"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["observation"]["features"] = list(self.observation.features)
        data["menu"] = [asdict(a) for a in self.menu]
        return data

    @classmethod
    def from_dict(cls, value: Mapping) -> DecisionRecord:
        _exact_fields(value, set(cls.__dataclass_fields__), "decision")
        data = dict(value)
        obs = data["observation"]
        _exact_fields(obs, set(Observation.__dataclass_fields__), "observation")
        _require(type(obs["features"]) is list, "Serialized features must be an array")
        data["observation"] = Observation(**{**obs, "features": tuple(obs["features"])})
        _exact_fields(data["behavior"], set(BehaviorPolicy.__dataclass_fields__), "behavior")
        data["behavior"] = BehaviorPolicy(**data["behavior"])
        _require(type(data["menu"]) is list, "Serialized menu must be an array")
        for action in data["menu"]:
            _exact_fields(action, set(ActionSpec.__dataclass_fields__), "action")
        data["menu"] = tuple(ActionSpec(**a) for a in data["menu"])
        return cls(**data)


@dataclass(frozen=True)
class EpisodeEnd:
    episode_id: str
    status: str  # terminated, truncated, interrupted, failed
    transition_count: int = field(kw_only=True)  # captured by the collector at the end boundary
    winner: str | None = None
    victory_route: str | None = None
    reason: str | None = None
    contract_id: str = CONTRACT.identity
    schema: str = "wotr_actor_critic_episode_end_v2"

    def __post_init__(self):
        _contract(self.contract_id)
        _require(self.schema == "wotr_actor_critic_episode_end_v2", "Unsupported end schema")
        _text(self.episode_id, "episode_id")
        _integer(self.transition_count, "transition_count")
        _require(self.status in ("terminated", "truncated", "interrupted", "failed"), "Unknown episode end status")
        if self.status == "terminated":
            _side(self.winner)
            _require(type(self.victory_route) is str and self.victory_route in VICTORY_ROUTES[self.winner], "Victory route does not belong to winner")
            _require(self.reason is None, "Decided game requires a victory route, not an interruption reason")
        else:
            _require(self.winner is None and self.victory_route is None, "Unfinished games cannot acquire terminal labels")
            _text(self.reason, "Nonterminal end reason")

    def terminal_return(self, side: str) -> float:
        _side(side)
        _require(self.status == "terminated", "No terminal return for unfinished or failed games")
        return 1.0 if side == self.winner else -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> EpisodeEnd:
        _exact_fields(value, set(cls.__dataclass_fields__), "episode end")
        return cls(**value)


def validate_episode(records: tuple[DecisionRecord, ...], end: EpisodeEnd) -> dict[str, tuple[DecisionRecord, ...]]:
    """Validate a complete recorded prefix; group by owner without alternating signs.

    An interrupted prefix remains valid evidence but has no terminal labels.
    A resumed full episode must include its prefix exactly once.
    """
    _require(type(records) is tuple and isinstance(end, EpisodeEnd), "Typed episode records required")
    _require(len(records) == end.transition_count, "Episode end transition count differs from recorded rows")
    grouped: dict[str, list[DecisionRecord]] = {"fp": [], "sh": []}
    for index, row in enumerate(records):
        _require(isinstance(row, DecisionRecord), "Untyped decision row")
        _require(row.episode_id == end.episode_id, "Cross-episode data splice")
        _require(row.transition_index == index, "Missing, duplicated or reordered player transition")
        side = row.observation.side
        _require(row.side_index == len(grouped[side]), "Incorrect player decision sequence")
        grouped[side].append(row)
    return {side: tuple(rows) for side, rows in grouped.items()}
