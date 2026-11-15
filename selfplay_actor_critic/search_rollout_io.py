"""Packed persistence for search-target rollouts; loaded lazily by the learner.

A game directory holds arrays.npz (inputs, prior, target policy, visits) and
episode.json (row metadata, ending, provenance, search configuration).
Completion metadata is written last so an interrupted write never looks
complete. Replay traces are produced separately by the runner.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .collector import Rollout
from .contracts import CONTRACT, BehaviorPolicy, ContractError, EpisodeEnd
from .inspect_assets import sha

SCHEMA = "search_rollout_v1"


class RowMeta(NamedTuple):
    side: str
    side_index: int
    turn: int
    phase: str
    selected: int
    value: float | None
    behavior: BehaviorPolicy
    kind: str
    fixed_reason: str | None
    action_count: int
    searched: bool
    root_value: float | None
    search_value: float | None
    has_afterstate: bool = False


@dataclass
class SearchGame:
    """Row metadata only; arrays are loaded on demand with load_search_arrays."""
    path: Path
    episode_id: str
    seed: int
    end: EpisodeEnd | None
    error: dict | None
    rows: list
    search: dict
    arrays_sha256: str
    # Names of the action kinds indexed by the per-candidate `kinds` array
    # (empty for games saved before 2026-10-04, which have no such array).
    kind_names: tuple = ()

    @property
    def searched_rows(self):
        return [i for i, row in enumerate(self.rows) if row.searched]


def action_kind(action) -> str:
    """Stable name of an engine action's type (enum name or its string)."""
    kind = getattr(action, "action_type", None)
    return str(getattr(kind, "name", kind))


def save_search_rollout(rollout: Rollout, destination, *, search_config: dict):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows, targets = rollout.rows, rollout.search
    if len(targets) != len(rows):
        raise ContractError("Search targets are not aligned with recorded rows")
    lengths = [len(row.actions) for row in rows]
    total = int(sum(lengths))
    target = np.zeros(total, np.float32)
    visits = np.zeros(total, np.int32)
    prior = np.zeros(total, np.float64)
    completed = np.zeros(total, np.float32)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64))).astype(np.int64)
    # Afterstate observations of searched rows, in row order (zeros when the
    # collector did not record one, flagged per row by has_afterstate).
    afterstates = []
    metadata_rows = []
    # Action kind of every candidate (index into kind_names): the credit
    # gate and the credit-validity instrument work per kind of move.
    kinds = np.zeros(total, np.int16)
    kind_index, kind_names = {}, []
    for i, (row, hit) in enumerate(zip(rows, targets)):
        start, stop = offsets[i], offsets[i + 1]
        for j, action in enumerate(row.actions):
            name = action_kind(action)
            code = kind_index.get(name)
            if code is None:
                code = kind_index[name] = len(kind_names)
                kind_names.append(name)
            kinds[start + j] = code
        if hit is None:
            prior[start:stop] = row.log_probabilities
            searched = False
            extra = dict(root_value=None, search_value=None, elapsed=None, considered=None,
                         simulations=None, terminal_leaves=None, max_depth=None, steps=None)
        else:
            if hit.target.shape != (stop - start,) or hit.visits.shape != (stop - start,) or hit.prior_logp.shape != (stop - start,):
                raise ContractError("Search target width differs from the recorded menu")
            target[start:stop] = hit.target
            visits[start:stop] = hit.visits
            prior[start:stop] = hit.prior_logp
            if getattr(hit, "completed_q", None) is not None and hit.completed_q.shape == (stop - start,):
                completed[start:stop] = hit.completed_q
            searched = True
            after = getattr(hit, "afterstate", None)
            has_afterstate = after is not None and np.shape(after) == (CONTRACT.observation_width,)
            afterstates.append(np.asarray(after, np.float32) if has_afterstate
                               else np.zeros(CONTRACT.observation_width, np.float32))
            extra = dict(root_value=hit.root_value, search_value=hit.search_value, elapsed=hit.elapsed,
                         considered=hit.considered, simulations=hit.simulations,
                         terminal_leaves=hit.terminal_leaves, max_depth=hit.max_depth, steps=hit.steps,
                         has_afterstate=bool(has_afterstate))
        metadata_rows.append(dict(side=row.side, side_index=row.side_index, turn=row.turn, phase=row.phase,
            selected=row.selected, value=row.value, behavior=asdict(row.behavior), kind=row.kind,
            fixed_reason=row.fixed_reason, action_count=len(row.actions), searched=searched, **extra))
    arrays = dict(
        observations=np.stack([r.inputs.observation for r in rows]) if rows else np.empty((0, CONTRACT.observation_width), np.float32),
        candidates=np.concatenate([r.inputs.candidates for r in rows]) if rows else np.empty((0, CONTRACT.action_width), np.float32),
        regions=np.concatenate([r.inputs.regions for r in rows]) if rows else np.empty(0, np.int64),
        probabilities=np.concatenate([r.inputs.probabilities for r in rows]) if rows else np.empty((0, CONTRACT.probability_width), np.float32),
        log_probabilities=np.concatenate([r.log_probabilities for r in rows]) if rows else np.empty(0, np.float64),
        prior_logp=prior, target=target, visits=visits, completed_q=completed, offsets=offsets,
        afterstates=(np.stack(afterstates) if afterstates else np.empty((0, CONTRACT.observation_width), np.float32)),
        kinds=kinds)
    # Ensemble critic: each member's value of every candidate (0 where not evaluated) and of the root of every
    # searched row (same order as afterstates). Written only when every searched row carries them.
    hits = [(i, hit) for i, hit in enumerate(targets) if hit is not None]
    member_counts = {None if getattr(hit, "member_q", None) is None else len(hit.member_q) for _, hit in hits}
    if hits and None not in member_counts and len(member_counts) == 1:
        count = member_counts.pop()
        member_q = np.zeros((total, count), np.float32)
        for i, hit in hits:
            member_q[offsets[i]:offsets[i + 1]] = np.asarray(hit.member_q, np.float32).T
        arrays["member_q"] = member_q
        arrays["member_root"] = np.stack([np.asarray(hit.member_root, np.float32) for _, hit in hits])
    watch = getattr(rollout, "watch", None) or []
    if rows and len(watch) == len(rows):
        # Row i: the position before row i's move as the side NOT to move sees it.
        # Absent for games resumed from a checkpoint written before this existed.
        arrays["watch"] = np.stack(watch).astype(np.float32)
    np.savez_compressed(destination / "arrays.npz", **arrays)
    metadata = dict(schema=SCHEMA, contract_id=CONTRACT.identity, episode_id=rollout.episode_id, seed=rollout.seed,
        end=None if rollout.end is None else rollout.end.to_dict(), error=rollout.error,
        search=dict(search_config), rows=metadata_rows, arrays_sha256=sha(destination / "arrays.npz"),
        kind_names=kind_names)
    (destination / "episode.json").write_text(json.dumps(metadata, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def load_search_game(path) -> SearchGame:
    path = Path(path)
    metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("contract_id") != CONTRACT.identity:
        raise ContractError("Unsupported search rollout")
    rows = []
    for data in metadata["rows"]:
        rows.append(RowMeta(data["side"], data["side_index"], data["turn"], data["phase"], data["selected"],
                            data["value"], BehaviorPolicy(**data["behavior"]), data["kind"], data["fixed_reason"],
                            data["action_count"], bool(data["searched"]), data["root_value"], data["search_value"],
                            bool(data.get("has_afterstate", False))))
    end = None if metadata["end"] is None else EpisodeEnd.from_dict(metadata["end"])
    if end is not None and end.transition_count != len(rows):
        raise ContractError("Search rollout row count differs from its ending")
    return SearchGame(path, metadata["episode_id"], metadata["seed"], end, metadata["error"], rows,
                      metadata["search"], metadata["arrays_sha256"], tuple(metadata.get("kind_names", ())))


def load_search_arrays(game: SearchGame, *, verify=True) -> dict:
    path = game.path / "arrays.npz"
    if verify and sha(path) != game.arrays_sha256:
        raise ContractError("Search rollout array hash differs from completion metadata")
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    offsets = data["offsets"]
    count = len(game.rows)
    if (offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(data["candidates"])
            or len(data["observations"]) != count or len(data["target"]) != len(data["candidates"])
            or any(offsets[i + 1] - offsets[i] != row.action_count for i, row in enumerate(game.rows))):
        raise ContractError("Search rollout offsets/row counts disagree")
    return data
