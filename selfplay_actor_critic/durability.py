"""Admission and publication contracts for offline collection and learning.

These checks run at artifact boundaries, never in per-move inference.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .contracts import CONTRACT, ContractError, canonical_json
from .runtime_io import _atomic_write, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = ROOT / "audit/card_gap_followthrough_sep14/epoch_extension_to8_r1/run_r1"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


@contextmanager
def file_lock(path, timeout=120):
    """Crash-released exclusive lock; wait for another short artifact transaction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        started = time.monotonic()
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Artifact transaction is busy: {path}")
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def actor_source(policy, side):
    """One resolution rule shared by collectors and league admission."""
    if policy == "seed":
        # R17-SAC row 67: the running generation's copy of the epoch-8 pair (the v7 files under SEED_BASE load
        # under no current generation; generation.BC_SEED_DIRS)
        from .generation import bc_seed_dir
        return "raw", bc_seed_dir(8, "durability.actor_source") / f"{side}_epoch1.pt"
    folder = Path(policy)
    if not folder.is_absolute():
        folder = ROOT / folder
    for kind, path in (("actor", folder / f"{side}_seed.pt"),
                       ("raw", folder / f"{side}_epoch1.pt"),
                       ("search", folder / side / "state.pt")):
        if path.is_file():
            if kind == "search":
                receipt = json.loads((path.parent / "complete.json").read_text(encoding="utf-8"))
                if receipt.get("schema") != "search_target_learner_completion_v1" or receipt.get("sha256") != digest_file(path):
                    raise ContractError("Incomplete or modified search checkpoint")
            return kind, path
    raise FileNotFoundError(f"No supported {side} actor in {folder}")


def behavior_content(policy, side):
    kind, path = actor_source(policy, side)
    return dict(format=kind, sha256=digest_file(path))


def critic_content(folder, side):
    if not folder:
        return None
    folder = Path(folder)
    path = folder / f"{side}_critic.pt"
    # The exact marker name is owned by full_state_critic, with no model import.
    from .full_state_critic import is_both_views_dir
    if is_both_views_dir(folder):
        path = folder / "critic.pt"
    return dict(path=str(path.resolve()), sha256=digest_file(path))


def episode_files(run):
    return sorted(Path(run).glob("batches/*/training_workers/worker_*/training/raw-*/episode.json"))


def episode_index(run, *, require_identity=False):
    """Reject duplicates before cache reuse, sampling, or fold assignment."""
    out = {}
    for path in episode_files(run):
        meta = json.loads(path.read_text(encoding="utf-8"))
        seed = int(meta["seed"])
        if seed in out:
            raise ContractError(f"Duplicate episode seed {seed}: {out[seed][0]} and {path}")
        if require_identity and not meta.get("behavior_content"):
            raise ContractError(f"{path}: missing behavior content identity; legacy data needs explicit provenance migration")
        out[seed] = (path, meta)
    return out


def episode_folder(run, seed):
    # Fixed shard assignment is independent of the number of worker processes.
    return Path(run) / "batches/batch_000_raw/training_workers" / f"worker_{seed % 256:03d}" / "training" / f"raw-{seed}"


def ensure_run_manifest(run, config, behaviors):
    """A worker-count change is operational; every trajectory-affecting field is frozen."""
    run = Path(run)
    settings = {k: v for k, v in config.items() if k not in ("workers", "output")}
    body = dict(schema="value_run_v2", contract_id=CONTRACT.identity, settings=settings,
                behavior_content=behaviors, cohort_policy="all_scheduled_terminal_v1")
    body["identity"] = digest_json(body)
    path = run / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != body:
            raise ContractError("Resume changes the frozen collection contract or policy contents; use a new run directory")
    else:
        if (run / "config.json").exists() or episode_files(run):
            raise ContractError("Legacy run has no content manifest; explicit provenance migration is required before resume")
        write_json(path, body)
        write_json(run / "config.json", config)
    return body


def validate_episode(meta, manifest):
    settings = manifest["settings"]
    if (meta.get("run_identity") != manifest["identity"] or
            meta.get("behavior_content") != manifest["behavior_content"] or
            float(meta["temperature"]) != float(settings["temperature"])):
        raise ContractError("Episode does not belong to the frozen collection manifest")
    seed = int(meta["seed"])
    if not settings["first_seed"] <= seed < settings["first_seed"] + settings["games"]:
        raise ContractError("Episode seed is outside the scheduled cohort")


def cohort_status(run, *, write=False):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    entries = episode_index(run, require_identity=True)
    counts = dict(terminated=0, truncated=0, failed=0, interrupted=0, missing=0)
    for _, meta in entries.values():
        validate_episode(meta, manifest)
        status = (meta.get("end") or {}).get("status")
        if status not in counts or status == "missing":
            raise ContractError("Invalid recorded cohort status")
        counts[status] += 1
    counts["missing"] = settings["games"] - len(entries)
    result = dict(schema="value_cohort_v1", run_identity=manifest["identity"], policy=manifest["cohort_policy"],
                  scheduled=settings["games"], counts=counts, complete=counts["terminated"] == settings["games"])
    if write:
        write_json(run / "cohort.json", result)
    return result


def require_terminal_cohort(run):
    status = cohort_status(run)
    if not status["complete"]:
        raise ContractError(f"{run}: all scheduled games must terminate before fitting: {status['counts']}")


def prediction_receipt(run, side, critic_dir, files):
    manifest = Path(run) / "manifest.json"
    return dict(schema="pg_predictions_v2", side=side, critic=critic_content(critic_dir, side),
                manifest=digest_file(manifest) if manifest.exists() else None,
                episodes=[dict(path=str(Path(p).relative_to(run)), meta=digest_file(p),
                               arrays=digest_file(Path(p).parent / "arrays.npz")) for p in files])


def validate_generation(folder, side, *, expected_request=None):
    folder = Path(folder)
    receipt = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    expected = {f"{side}_seed.pt", "pg.json"}
    if receipt.get("schema") != "pg_generation_v1" or receipt.get("side") != side or set(receipt.get("files", {})) != expected:
        raise ContractError("Unsupported generation completion receipt")
    for name, digest in receipt["files"].items():
        if digest_file(folder / name) != digest:
            raise ContractError("Modified or incomplete generation: " + name)
    meta = json.loads((folder / "pg.json").read_text(encoding="utf-8"))
    if meta.get("side") != side or not meta.get("provenance"):
        raise ContractError("Generation lacks fitting provenance")
    if expected_request is not None and meta.get("fit_request") != expected_request:
        raise ContractError("Completed generation does not match the requested fitting settings or input contents; use a distinct output")
    return receipt


def publish_generation(ai, actor, folder, metadata):
    """Publish verified weights and provenance as one directory; retain incomplete attempts."""
    from .policy import load_actor_seed, save_actor_seed
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(folder.parent / (folder.name + ".publish.lock")):
        if (folder / "complete.json").exists():
            validate_generation(folder, actor.identity.side)
            saved = json.loads((folder / "pg.json").read_text(encoding="utf-8"))
            if saved != metadata:
                raise ContractError("Completed generation has different provenance or fitting settings")
            return
        stage = folder.parent / ("." + folder.name + ".staged-" + uuid.uuid4().hex)
        stage.mkdir()
        side = actor.identity.side
        checkpoint = stage / f"{side}_seed.pt"
        save_actor_seed(actor, checkpoint)
        write_json(stage / "pg.json", metadata)
        loaded = load_actor_seed(ai, checkpoint, side)
        if loaded.identity != actor.identity:
            raise ContractError("Published actor does not round trip")
        write_json(stage / "complete.json", dict(schema="pg_generation_v1", side=side,
            files={name: digest_file(stage / name) for name in (checkpoint.name, "pg.json")}))
        validate_generation(stage, side)
        if folder.exists():
            # Never erase partial or legacy evidence when retrying an interrupted fit.
            folder.rename(folder.parent / (folder.name + ".incomplete-" + uuid.uuid4().hex))
        os.replace(stage, folder)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generation", "cohort", "preflight"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--side", choices=("fp", "sh"), default="sh")
    parser.add_argument("--request-file", help="exact PG request captured before the next-step critic refit")
    args = parser.parse_args()
    if args.command == "generation":
        expected = json.loads(Path(args.request_file).read_text(encoding="utf-8")) if args.request_file else None
        validate_generation(args.path, args.side, expected_request=expected)
    elif args.command == "cohort":
        require_terminal_cohort(args.path)
    else:
        from .pg_fit import _engine
        from .offline_eval import _load_actor
        _load_actor(_engine(), args.path, args.side)


if __name__ == "__main__":
    main()
