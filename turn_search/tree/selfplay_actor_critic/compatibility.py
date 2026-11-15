"""Metadata admission decisions. No checkpoint is loaded, relabeled, or trained."""
from dataclasses import asdict, dataclass
from typing import Mapping

from .contracts import CONTRACT, SIDES


@dataclass(frozen=True)
class ReuseAssessment:
    policy_warm_start: bool
    imitation_data: bool
    fresh_actor_data: bool
    training_resume: bool
    required_actions: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def assess_legacy_artifact(kind: str, metadata: Mapping) -> ReuseAssessment:
    """Only explicitly versioned legacy policy weights/data may warm start.

    This does not certify shapes of actual tensors, source integrity, completeness
    of a checkpoint, or corpus admission. Those remain required at load time.
    No existing artifact is admitted as an actor-critic resume or rollout.
    """
    if kind not in ("checkpoint", "human_policy_arrays", "search_trace", "search_targets", "response_policy"):
        raise ValueError("Unknown legacy artifact kind")
    if not isinstance(metadata, Mapping):
        raise ValueError("Artifact metadata must be a mapping")
    if kind in ("search_trace", "search_targets", "response_policy"):
        return ReuseAssessment(False, False, False, False, (), (
            "Keep for historical evaluation/debugging; not fresh direct-policy experience.",
            "Search visits and modeled-opponent targets are not sampled actor advantages.",))
    expected = {"observation_contract": CONTRACT.observation,
                "action_encoding_contract": CONTRACT.actions,
                "ai_policy_contract": CONTRACT.ai_policy,
                "input_dim": CONTRACT.observation_width,
                "action_dim": CONTRACT.action_width}
    mismatches = tuple(key for key, value in expected.items()
                       if type(metadata.get(key)) is not type(value) or metadata[key] != value)
    reasons = []
    if mismatches:
        reasons.append("Missing or incompatible metadata: " + ", ".join(mismatches))
    if kind == "checkpoint" and (type(metadata.get("side")) is not str or metadata["side"] not in SIDES):
        reasons.append("Explicit checkpoint side is required.")
    if kind == "human_policy_arrays" and metadata.get("admission_verified") is not True:
        reasons.append("Corpus admission and train/held-out separation must be verified.")
    okay = not reasons
    if kind == "checkpoint":
        actions = ("Validate actual weight layout before loading.",
                   "Import compatible policy/encoder weights with source provenance.",
                   "Reset the value head for the new win/loss objective.",
                   "Create fresh actor-critic optimizer state and counters.")
    else:
        actions = ("Use human action labels only for imitation/anchoring.",
                   "Preserve game-level train/held-out split and admission evidence.",
                   "Do not reuse stored z/phi/value labels or fabricate behavior log probabilities.")
    reasons.append("Legacy artifacts cannot resume the new learning algorithm or supply fresh PPO rollouts.")
    return ReuseAssessment(okay and kind == "checkpoint", okay and kind == "human_policy_arrays",
                           False, False, actions if okay else (), tuple(reasons))
