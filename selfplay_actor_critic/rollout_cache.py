"""Content admission for counterfactual rollout artifacts, outside game play."""
import json
from pathlib import Path

import numpy as np

from .contracts import CONTRACT, ContractError, canonical_json
from .durability import behavior_content, digest_file

ROOT = Path(__file__).resolve().parents[1]


def source_content(root=None):
    """Bind the engine and Python rollout dependencies, including their schemas."""
    root = ROOT if root is None else Path(root)
    files = []
    for folder in (root / "src", root / "selfplay_actor_critic"):
        files.extend(p for p in folder.glob("*.py") if not p.name.startswith("test_"))
    return {p.relative_to(root).as_posix(): digest_file(p) for p in sorted(files)}


def request_for(task):
    (folder, run, decision, side, policies, temperature, candidates, explore,
     continuations, adaptive, max_continuations, close, top_k) = task
    if (candidates < 1 or continuations < 1 or top_k < 1 or explore < 0
            or not np.isfinite(temperature) or temperature <= 0
            or (adaptive and max_continuations < continuations)):
        raise ContractError("Rollout requires positive sampling controls and an adaptive cap covering the fixed round")
    folder, run = Path(folder).resolve(), Path(run).resolve()
    behavior = {s: behavior_content(policies[s], s) for s in ("fp", "sh")}
    metadata = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("behavior_content") is not None and metadata["behavior_content"] != behavior:
        raise ContractError("Rollout policies changed from the recorded episode behavior")
    if "temperature" in metadata and float(metadata["temperature"]) != float(temperature):
        raise ContractError("Rollout temperature differs from the recorded episode")
    return dict(schema="counterfactual_rollout_request_v2", contract=CONTRACT.to_dict(),
        folder=str(folder), run=str(run), decision=int(decision), side=side,
        policies=dict(policies), behavior=behavior,
        settings=dict(temperature=float(temperature), candidates=int(candidates), explore=int(explore),
                      continuations=int(continuations), adaptive=bool(adaptive),
                      max_continuations=int(max_continuations), close=float(close), top_k=int(top_k)),
        episode={name: digest_file(folder / name) for name in ("episode.json", "arrays.npz")},
        run_files={name: digest_file(run / name) if (run / name).is_file() else None
                   for name in ("config.json", "manifest.json")},
        sources=source_content())


def task_for(request):
    s = request["settings"]
    return (request["folder"], request["run"], request["decision"], request["side"], request["policies"],
            s["temperature"], s["candidates"], s["explore"], s["continuations"], s["adaptive"],
            s["max_continuations"], s["close"], s["top_k"])


def load_admitted(path):
    """Read a consumer's cache only when its exact inputs still match."""
    with np.load(path, allow_pickle=False) as archive:
        if "request" not in archive.files:
            raise ContractError(f"{path}: legacy rollout cache needs rebuilding with rollout_targets")
        request = json.loads(str(archive["request"]))
        if request.get("schema") != "counterfactual_rollout_request_v2":
            raise ContractError(f"{path}: unsupported rollout request")
        if canonical_json(request_for(task_for(request))) != canonical_json(request):
            raise ContractError(f"{path}: rollout inputs changed; rebuild the rollout cache")
        if np.any(archive["counts_fixed"] <= 0):
            raise ContractError(f"{path}: candidate lacks terminal fixed-round outcomes; rollout is not a distillation target")
        width = archive["kept_h"].shape[1]
        for name in ("tail_h_min", "tail_h_max"):
            if name not in archive.files or archive[name].shape != (width,) or not np.isfinite(archive[name]).all():
                raise ContractError(f"{path}: omitted-action feature intervals are missing or invalid")
        if np.any(archive["tail_h_min"] > archive["tail_h_max"]):
            raise ContractError(f"{path}: omitted-action feature intervals are reversed")
        return {key: archive[key] for key in archive.files}
