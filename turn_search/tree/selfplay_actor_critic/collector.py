"""Batched actual-game collection with compact arrays and no per-row audits.

The coordinator holds both players, but each PolicyActor sees only its own
encoded decisions. Expensive contract conversion/audits are explicit offline
operations in rollout_io, never part of this loop.
"""
import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .contracts import BehaviorPolicy, EpisodeEnd
from .policy import ActorInput, PolicyActor


class StoredAction(NamedTuple):
    die_idx: int
    action_type: str
    params: dict


@dataclass(frozen=True)
class Transition:
    side: str
    side_index: int
    turn: int
    phase: str
    inputs: ActorInput
    actions: tuple[StoredAction, ...]
    selected: int
    log_probabilities: np.ndarray
    value: float | None
    behavior: BehaviorPolicy
    kind: str
    fixed_reason: str | None

    @property
    def action_count(self):
        return len(self.actions)


@dataclass
class Rollout:
    episode_id: str
    seed: int
    rows: list[Transition] = field(default_factory=list)
    end: EpisodeEnd | None = None
    error: dict | None = None
    # Search targets aligned with rows (None for rows decided without
    # search). Written only by the search collector; ignored elsewhere.
    search: list = field(default_factory=list)
    # The OTHER side's observation of the position before each row's move
    # (written by the search collector). With it each side's value function
    # can be trained on EVERY decision point of a game, not only on its own:
    # the one-ply scoring evaluates states in which the opponent moves next,
    # and a critic fitted on its own decision states alone priced those
    # inconsistently (2026-10-04).
    watch: list = field(default_factory=list)


class CollectionFailure(RuntimeError):
    def __init__(self, rollout, partial_rollouts):
        super().__init__(rollout.error["message"])
        self.rollout = rollout
        self.partial_rollouts = partial_rollouts


class CollectionPaused(Exception):
    """A saved collection boundary, with games still live rather than truncated."""
    pass


class Collector:
    def __init__(self, actors: dict[str, PolicyActor]):
        if set(actors) != {"fp", "sh"} or any(a.identity.side != side for side, a in actors.items()):
            raise ValueError("Collector requires one actor owned by each side")
        self.actors = dict(actors)

    def collect(self, environment, *, stop_requested=lambda: False):
        return self.collect_many([environment], stop_requested=stop_requested)[0]

    def collect_many(self, environments, *, episode_actors=None, resume_rollouts=None,
                     stop_requested=lambda: False, on_commit=None, on_boundary=None):
        environments = list(environments)
        assignments = ([dict(self.actors) for _ in environments] if episode_actors is None
                       else [dict(actors) for actors in episode_actors])
        if len(assignments) != len(environments) or any(
                set(actors) != {"fp", "sh"} or any(a.identity.side != side for side, a in actors.items())
                for actors in assignments):
            raise ValueError("Every episode requires one correctly owned actor per side")
        rollouts = ([Rollout(e.episode_id, e.seed) for e in environments] if resume_rollouts is None
                    else list(resume_rollouts))
        if len(rollouts) != len(environments) or any(
                e.episode_id != r.episode_id or e.seed != r.seed or e.committed != len(r.rows)
                for e, r in zip(environments, rollouts)):
            raise ValueError("Environment and saved rollout boundaries disagree")
        if len({r.episode_id for r in rollouts}) != len(rollouts):
            raise ValueError("Episode IDs must be unique within a collection batch")
        side_counts = [{s: sum(row.side == s for row in r.rows) for s in ("fp", "sh")} for r in rollouts]
        if any(row.behavior != actors[row.side].identity for actors, r in zip(assignments, rollouts) for row in r.rows):
            raise ValueError("A resumed game must keep its assigned policies")
        active = {i for i, r in enumerate(rollouts) if r.end is None}
        current, frame, selected = None, None, None
        try:
            while active:
                # Games can have different frozen opponents. Batch only inputs
                # assigned to the SAME actor, never all models of a given side.
                ready = {}
                for current in sorted(active):
                    env, rollout = environments[current], rollouts[current]
                    frame, selected = None, None
                    if stop_requested():
                        env.finish("interrupted", reason="user_stop")
                    frame = env.prepare()
                    if frame is None:
                        rollout.end = env.end
                        active.remove(current)
                        continue
                    if len(frame.actions) == 1:
                        selected = 0
                        self._commit(env, rollout, frame, selected, np.zeros(1, dtype=np.float64), None,
                                     side_counts[current], assignments[current][frame.side])
                        if on_commit is not None:
                            on_commit(env, rollout.rows[-1])
                    else:
                        actor = assignments[current][frame.side]
                        ready.setdefault(actor, []).append((current, frame))
                for actor, batch in ready.items():
                    current, frame = batch[0]
                    selected = None
                    predictions = actor.predict_many([frame.inputs for _, frame in batch])
                    for (current, frame), prediction in zip(batch, predictions):
                        env, rollout = environments[current], rollouts[current]
                        selected = prediction.sample(env.samplers[frame.side])
                        self._commit(env, rollout, frame, selected, prediction.log_probabilities,
                                     prediction.value, side_counts[current], actor)
                        if on_commit is not None:
                            on_commit(env, rollout.rows[-1])
                if on_boundary is not None and on_boundary(environments, rollouts) is False:
                    raise CollectionPaused("Collection checkpoint saved; games remain resumable")
            return rollouts
        except CollectionPaused:
            raise
        except Exception as exc:
            env, rollout = environments[current], rollouts[current]
            env.finish("failed", reason="collection_exception")
            rollout.end = env.end
            rollout.error = dict(type=type(exc).__name__, message=str(exc), phase=env.stage,
                                 turn=env.state.turn, attempted_index=selected,
                                 action_type=(frame.actions[selected].action_type if frame is not None and selected is not None else None))
            for i in active:
                if i != current:
                    environments[i].finish("interrupted", reason="peer_episode_failed")
                    rollouts[i].end = environments[i].end
            raise CollectionFailure(rollout, rollouts) from exc

    def _commit(self, env, rollout, frame, selected, log_probabilities, value, side_counts, actor):
        # Snapshot descriptors before execution: some rules retain choice dicts
        # as pending payloads. Do not retain mutable aliases into later states.
        actions = tuple(StoredAction(a.die_idx, a.action_type, copy.deepcopy(a.params)) for a in frame.actions)
        row = Transition(frame.side, side_counts[frame.side], frame.turn, frame.phase, frame.inputs,
                         actions, selected, log_probabilities, value, actor.identity,
                         "learned" if len(actions) > 1 else "fixed_policy" if frame.fixed_reason else "forced",
                         frame.fixed_reason)
        env.apply(frame, selected)
        rollout.rows.append(row)
        side_counts[frame.side] += 1
