"""Input and owner receipts for optional replay-menu and Q-value fitters."""
import json
from pathlib import Path

import numpy as np

from .contracts import CONTRACT, ContractError, canonical_json
from .durability import behavior_content, critic_content, digest_file, validate_episode
from .rollout_cache import source_content


def menu_path(run, side, seed):
    """<run>/menus/<side>/<seed>.npz. A cache holds one side's decisions only, so the side is part of the key:
    with a side-free path, building one side's menus replaced the other side's valid caches (S3-03)."""
    if side not in ("fp", "sh"):
        raise ContractError(f"Replay menu side must be fp or sh, not {side!r}")
    return Path(run) / "menus" / side / f"{int(seed)}.npz"


def menu_request(task):
    folder, run, side, policies, per_game, check, top_k, temperature = task
    folder, run = Path(folder).resolve(), Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    validate_episode(meta, manifest)
    behavior = {s: behavior_content(policies[s], s) for s in ("fp", "sh")}
    if (side not in ("fp", "sh") or meta["end"]["status"] != "terminated"
            or behavior != meta["behavior_content"] or float(temperature) != float(meta["temperature"])):
        raise ContractError("Replay menu inputs differ from the recorded terminal episode")
    return dict(schema="replay_menu_request_v1", contract=CONTRACT.to_dict(),
                folder=str(folder), run=str(run), side=side, seed=int(meta["seed"]),
                policies=dict(policies), behavior=behavior, per_game=int(per_game), check=bool(check),
                top_k=int(top_k), temperature=float(temperature),
                manifest=digest_file(run / "manifest.json"),
                episode={name: digest_file(folder / name) for name in ("episode.json", "arrays.npz")},
                sources=source_content())


def menu_task(request):
    return tuple(request[name] for name in
                 ("folder", "run", "side", "policies", "per_game", "check", "top_k", "temperature"))


def load_menu(path, *, run, side, seed, lazy=()):
    """One admitted replay-menu cache as {member: array}. `path` may be an open binary file. Members named in
    `lazy` are admitted with the rest but left unread (replay_menus.MenuBatch.load_streamed memory-maps the
    activations from the same open file instead of reading them, R2-RS-v099-S3-01)."""
    with np.load(path, allow_pickle=False) as archive:
        if "request" not in archive.files:
            raise ContractError("Legacy replay menu has no input identity; rebuild with replay_menus")
        request = json.loads(str(archive["request"]))
        if (request.get("schema") != "replay_menu_request_v1"
                or request.get("run") != str(Path(run).resolve()) or request.get("side") != side
                or request.get("seed") != int(seed) or not request.get("check")
                or str(archive["side"]) != side or int(archive["seed"]) != int(seed)
                or float(archive["temperature"]) != request["temperature"]):
            raise ContractError("Replay menu belongs to another owner/input or was built without replay checks")
        if canonical_json(menu_request(menu_task(request))) != canonical_json(request):
            raise ContractError("Replay menu inputs changed; rebuild with replay_menus")
        return {key: archive[key] for key in archive.files if key not in lazy}


def q_request(task):
    folder, run, side, policies, candidates, critic_dir, temperature = task
    folder, run = Path(folder).resolve(), Path(run).resolve()
    meta = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    seed = int(meta["seed"])
    menu_file = menu_path(run, side, seed)
    menu = load_menu(menu_file, run=run, side=side, seed=seed)
    recorded = json.loads(str(menu["request"]))
    if (recorded["folder"] != str(folder) or recorded["policies"] != policies
            or recorded["temperature"] != float(temperature)):
        raise ContractError("Q-value request differs from its admitted menu")
    return dict(schema="menu_q_request_v1", folder=str(folder), run=str(run), side=side, seed=seed,
                policies=dict(policies), candidates=int(candidates), critic_dir=str(Path(critic_dir).resolve()),
                temperature=float(temperature), menu_sha256=digest_file(menu_file),
                critic=critic_content(critic_dir, side))


def q_task(request):
    return tuple(request[name] for name in
                 ("folder", "run", "side", "policies", "candidates", "critic_dir", "temperature"))


def load_q(path, *, run, side, seed):
    with np.load(path, allow_pickle=False) as archive:
        if "request" not in archive.files:
            raise ContractError("Legacy Q values have no input identity; rebuild with allaction_fit qvalues")
        request = json.loads(str(archive["request"]))
        if (request.get("schema") != "menu_q_request_v1"
                or request.get("run") != str(Path(run).resolve()) or request.get("side") != side
                or request.get("seed") != int(seed)
                or str(archive["side"]) != side or int(archive["seed"]) != int(seed)):
            raise ContractError("Q values belong to another side, seed or run")
        if canonical_json(q_request(q_task(request))) != canonical_json(request):
            raise ContractError("Q-value inputs changed; rebuild with allaction_fit qvalues")
        return {key: archive[key] for key in archive.files}
