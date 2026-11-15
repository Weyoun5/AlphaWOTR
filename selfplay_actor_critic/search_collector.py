"""Actual-game collection where learner-owned decisions are chosen by search.

Every learner-owned learned decision records the search's improved policy
over the COMPLETE menu and its mixed root value as supervised targets.
Frozen historical opponents and evaluation games keep direct sampling.
The live game's RNG streams are never consumed by a search; each search
draws its Gumbel noise and clone seeds from the owner's per-episode sampler.
"""
import copy
from dataclasses import dataclass

import numpy as np

from .collector import (CollectionFailure, CollectionPaused, Collector, Rollout, StoredAction, Transition)
from .policy import PolicyActor
from .search import GumbelSearcher, SearchConfig, SearchResult


@dataclass(frozen=True)
class SearchTarget:
    target: np.ndarray        # float32, complete menu, sums to one
    visits: np.ndarray        # int32, complete menu
    prior_logp: np.ndarray    # float64 behavior prior over the complete menu
    root_value: float         # raw network value at the root
    search_value: float       # mixed root value after search
    considered: int
    simulations: int
    elapsed: float
    terminal_leaves: int
    max_depth: int
    steps: int
    completed_q: np.ndarray = None  # float32, completed Q per root action (diagnostic)
    # Own-side observation of the position right after the selected action
    # was applied (float32 [W]); the critic is trained on it with the same
    # return as the decision, so afterstate search values are in-distribution.
    afterstate: np.ndarray = None
    # Ensemble critic: every member's value of every evaluated action
    # ([members, menu], 0 where not evaluated) and of the root ([members]).
    member_q: np.ndarray = None
    member_root: np.ndarray = None

    @classmethod
    def from_result(cls, result: SearchResult):
        completed = np.asarray(result.completed, np.float32)
        if completed.shape != np.shape(result.target):
            completed = np.zeros(len(result.target), np.float32)
        return cls(np.asarray(result.target, np.float32), np.asarray(result.visits, np.int32),
                   np.asarray(result.prior_logp, np.float64), float(result.root_value), float(result.search_value),
                   int(result.considered), int(result.simulations), float(result.elapsed),
                   int(result.terminal_leaves), int(result.max_depth), int(result.steps), completed,
                   member_q=getattr(result, "member_completed", None), member_root=getattr(result, "member_root", None))


class SearchCollector(Collector):
    """Collector whose learner-owned learned decisions run a Gumbel search.

    opponent_models: frozen PolicyActor per side, used ONLY inside the tree
    as a model of the other player. They are never the live opponent.
    """
    def __init__(self, actors: dict[str, PolicyActor], *, search: SearchConfig | None,
                 opponent_models: dict[str, PolicyActor] | None, temperature: float = 1.,
                 search_sides=("fp", "sh"), side_temperature: dict | None = None,
                 search_focus: dict[str, str] | None = None):
        super().__init__(actors)
        # Optional per-side override of `temperature` for decisions not chosen
        # by search (diagnostics: a sampling focal side against a greedy opponent).
        self.side_temperature = dict(side_temperature or {})
        if any(s not in ("fp", "sh") or not 0. <= float(t) <= 1. for s, t in self.side_temperature.items()):
            raise ValueError("side_temperature maps fp/sh to a temperature in [0, 1]")
        # Sides whose learner-owned decisions are searched. A side left out
        # plays its raw policy (and so produces no search targets): used when
        # the critic is a measured improvement for one side only.
        self.search_sides = tuple(search_sides)
        if not self.search_sides or any(s not in ("fp", "sh") for s in self.search_sides):
            raise ValueError("search_sides must name fp and/or sh")
        # Evaluation identities stay historical (they must never own learning
        # rows). Explicit match focus selects the agent that searches without
        # also turning search on for its frozen panel opponent.
        self.search_focus = None if search_focus is None else dict(search_focus)
        if self.search_focus is not None and any(
                not isinstance(e, str) or not e or s not in ("fp", "sh")
                for e, s in self.search_focus.items()):
            raise ValueError("search_focus maps episode IDs to fp/sh")
        self.search = None if search is None else GumbelSearcher(search)
        if self.search is not None:
            if opponent_models is None or set(opponent_models) != {"fp", "sh"} or any(
                    a.identity.side != side for side, a in opponent_models.items()):
                raise ValueError("Search collection requires one frozen opponent model per side")
        self.opponent_models = dict(opponent_models or {})
        # Sampling temperature for decisions NOT chosen by search (frozen
        # opponents, evaluation games). Training collection keeps 1.
        if not (0. <= float(temperature) <= 1.):
            raise ValueError("temperature must lie in [0, 1]")
        self.temperature = float(temperature)

    def _searched(self, actor, episode_id=None):
        owns_search = (actor.identity.origin == "learner" if self.search_focus is None
                       else self.search_focus.get(episode_id) == actor.identity.side)
        return self.search is not None and owns_search and actor.identity.side in self.search_sides

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
                or len(r.search) != len(r.rows)
                for e, r in zip(environments, rollouts)):
            raise ValueError("Environment and saved rollout boundaries disagree")
        if len({r.episode_id for r in rollouts}) != len(rollouts):
            raise ValueError("Episode IDs must be unique within a collection batch")
        if self.search_focus is not None and any(e.episode_id not in self.search_focus for e in environments):
            raise ValueError("Every evaluation episode requires an explicit search focus")
        side_counts = [{s: sum(row.side == s for row in r.rows) for s in ("fp", "sh")} for r in rollouts]
        if any(row.behavior != actors[row.side].identity for actors, r in zip(assignments, rollouts) for row in r.rows):
            raise ValueError("A resumed game must keep its assigned policies")
        active = {i for i, r in enumerate(rollouts) if r.end is None}
        current, frame, selected = None, None, None
        try:
            while active:
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
                    actor = assignments[current][frame.side]
                    if len(frame.actions) == 1:
                        selected = 0
                        self._commit(env, rollout, frame, selected, np.zeros(1, dtype=np.float64), None,
                                     side_counts[current], actor, None)
                        if on_commit is not None:
                            on_commit(env, rollout.rows[-1])
                    elif self._searched(actor, env.episode_id):
                        prior = actor.predict(frame.inputs)
                        opponent = self.opponent_models["sh" if frame.side == "fp" else "fp"]
                        result = self.search.search(env, frame, actor, opponent, env.samplers[frame.side], prior=prior)
                        selected = result.selected
                        log_target = np.log(np.maximum(result.target, np.finfo(np.float64).tiny))
                        self._commit(env, rollout, frame, selected, log_target, result.search_value,
                                     side_counts[current], actor, SearchTarget.from_result(result))
                        if on_commit is not None:
                            on_commit(env, rollout.rows[-1])
                    else:
                        ready.setdefault(actor, []).append((current, frame))
                for actor, batch in ready.items():
                    current, frame = batch[0]
                    selected = None
                    predictions = actor.predict_many([frame.inputs for _, frame in batch])
                    for (current, frame), prediction in zip(batch, predictions):
                        env, rollout = environments[current], rollouts[current]
                        selected = prediction.sample(env.samplers[frame.side],
                                                     self.side_temperature.get(frame.side, self.temperature))
                        self._commit(env, rollout, frame, selected, prediction.log_probabilities,
                                     prediction.value, side_counts[current], actor, None)
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

    def _commit(self, env, rollout, frame, selected, log_probabilities, value, side_counts, actor, target=None):
        actions = tuple(StoredAction(a.die_idx, a.action_type, copy.deepcopy(a.params)) for a in frame.actions)
        row = Transition(frame.side, side_counts[frame.side], frame.turn, frame.phase, frame.inputs,
                         actions, selected, np.asarray(log_probabilities, np.float64), value, actor.identity,
                         "learned" if len(actions) > 1 else "fixed_policy" if frame.fixed_reason else "forced",
                         frame.fixed_reason)
        # The same position as the side NOT to move sees it (its own private
        # knowledge only: the encoder hides the mover's hand from it).
        watcher = env.sides["sh" if frame.side == "fp" else "fp"]
        rollout.watch.append(np.asarray(env.ai.encode_state(env.state, watcher), np.float32))
        env.apply(frame, selected)
        if target is not None:
            # The deterministic afterstate the search would have scored for
            # this action; the critic learns it with the decision's return.
            from dataclasses import replace
            observation = np.asarray(env.ai.encode_state(env.state, env.sides[frame.side]), np.float32)
            target = replace(target, afterstate=observation)
        rollout.rows.append(row)
        rollout.search.append(target)
        side_counts[frame.side] += 1
