"""Admission for learned return credits; no work in a game decision loop."""
import json
from pathlib import Path

import numpy as np

from .contracts import CONTRACT, ContractError, canonical_json
from .durability import digest_file, episode_index

SCHEMA = "return_decomp_credit_v1"


def request_for(run, side, form, model, store):
    from .critic_feature_cache import read_completed_store
    if side not in ("fp", "sh") or form not in ("immediate", "cumulative"):
        raise ContractError("Unknown learned-credit owner or form")
    run, model, store = (Path(path).resolve() for path in (run, model, store))
    features, _, _, identity = read_completed_store(store, run)
    if hasattr(features, "_mmap"):
        features._mmap.close()
    episodes = episode_index(run)
    if not episodes:
        raise ContractError("Learned credits require identifiable source episodes")
    return dict(schema=SCHEMA, contract=CONTRACT.to_dict(), run=str(run), side=side, form=form,
        model=str(model), model_sha256=digest_file(model / "model.pt"),
        store=str(store), features=identity, index_sha256=digest_file(store / "index" / (run.name + ".npz")),
        episodes=[dict(seed=seed, folder=str(path.parent.resolve()),
                       metadata=digest_file(path), arrays=digest_file(path.parent / "arrays.npz"))
                  for seed, (path, _) in sorted(episodes.items())],
        sources={name: digest_file(Path(__file__).parent / name) for name in (
            "return_decomp.py", "return_credit_cache.py", "critic_feature_cache.py", "full_state_critic.py", "contracts.py")})


def load_admitted(path, *, run, side):
    with np.load(path, allow_pickle=False) as data:
        if "request" not in data.files:
            raise ContractError("Legacy learned credits have no owner/input identity; regenerate with return_decomp credits")
        request = json.loads(str(data["request"]))
        if request.get("schema") != SCHEMA:
            raise ContractError("Unsupported learned-credit receipt")
        if request.get("side") != side or request.get("run") != str(Path(run).resolve()):
            raise ContractError("Learned credits belong to another side or run")
        current = request_for(run, side, request["form"], request["model"], request["store"])
        if canonical_json(current) != canonical_json(request):
            raise ContractError("Learned-credit inputs changed; regenerate the credits")
        seeds, offsets, credit = data["seeds"], data["offsets"], data["credit"]
        if (seeds.ndim != 1 or seeds.dtype.kind not in "iu" or len(set(seeds.tolist())) != len(seeds)
                or offsets.shape != (len(seeds) + 1,) or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(credit)
                or credit.ndim != 1 or not np.isfinite(credit).all()):
            raise ContractError("Malformed learned-credit rows")
        return {int(seed): credit[offsets[i]:offsets[i + 1]] for i, seed in enumerate(seeds)}
