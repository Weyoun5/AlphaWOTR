"""Bounded compatibility inventory: checkpoint metadata and corpus manifests.

Reads two local weight files on CPU, array headers and compact menu indices,
never game logs or full feature/candidate payloads. Does not load a live model,
modify a checkpoint, or launch a worker. Historical manifest hashes are evidence,
not a fresh hash of all arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .compatibility import assess_legacy_artifact
from .contracts import CONTRACT, canonical_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_arrays(base, side, rows, menu_cap):
    """Validate shapes plus the small label/offset arrays in bounded chunks."""
    import numpy as np
    arrays, headers = {}, {}
    if type(rows) is not int or rows <= 0:
        raise ValueError("Invalid corpus rows in training receipt: " + side)
    if type(menu_cap) is not int or menu_cap <= 0:
        raise ValueError("Invalid corpus menu cap")
    try:
        for suffix, dtype, shape in (
                ("X", "float32", (rows, CONTRACT.observation_width)),
                ("cands", "float16", (None, CONTRACT.action_width)),
                ("human", "int32", (rows,)), ("off", "int64", (rows + 1,))):
            name = f"{side}_{suffix}.npy"
            path = base / "dataset" / name
            array = arrays[suffix] = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.dtype(dtype):
                raise ValueError(f"Corpus {name} dtype must be {dtype}")
            if array.ndim != len(shape) or any(n is not None and n != a for n, a in zip(shape, array.shape)):
                raise ValueError(f"Corpus {name} shape disagrees with receipt/schema: {array.shape}, expected {shape}")
            headers[name] = dict(shape=list(array.shape), dtype=str(array.dtype), bytes=path.stat().st_size)
        off, labels = arrays["off"], arrays["human"]
        candidates = arrays["cands"].shape[0]
        if off[0] != 0 or off[-1] != candidates:
            raise ValueError(f"Corpus {side} offset endpoints do not cover candidate rows")
        for start in range(0, rows, 65536):
            stop = min(rows, start + 65536)
            boundaries = off[start:stop + 1]
            if np.any(boundaries < 0) or np.any(boundaries > candidates):
                raise ValueError(f"Corpus {side} offset outside candidate array")
            counts = np.diff(boundaries)
            if np.any(counts <= 0) or np.any(counts > menu_cap):
                raise ValueError(f"Corpus {side} offset menus empty, reversed, or exceed menu cap")
            chosen = labels[start:stop]
            if np.any(chosen < 0) or np.any(chosen >= counts):
                raise ValueError(f"Corpus {side} human label outside its candidate menu")
    finally:
        # Explicit close also releases Windows file handles on invalid fixtures.
        for array in arrays.values():
            array._mmap.close()
    return headers


def inventory(root: Path) -> dict:
    import torch
    root = root.resolve()
    seeds = root / "audit/card_gap_followthrough_sep14/epoch_extension_to8_r1/run_r1/epoch_008/policy_run"
    base = root / "audit/2026-09-04/performance_pass/card_followup_reextract_sep14_full_r9/consolidated_r1"
    receipt_path = seeds / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("passed") is not True or receipt.get("policy_only") is not True
            or receipt.get("value_labels_used") is not False or receipt.get("search_labels_used") is not False):
        raise ValueError("Invalid completed policy-only training receipt")
    report = dict(schema="actor_critic_asset_inventory_v2", created_utc=datetime.now(timezone.utc).isoformat(),
                  contract=CONTRACT.to_dict(), contract_id=CONTRACT.identity, checkpoints=[],
                  verification_scope="Checkpoint bytes and metadata, corpus manifests, array headers and menu indices/labels; no full feature/candidate rehash or game-log read.")
    for side in ("fp", "sh"):
        path = seeds / f"{side}_epoch1.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        keys = ("side", "input_dim", "action_dim", "observation_contract", "action_encoding_contract",
                "ai_policy_contract", "reward_contract", "reward_profile", "head_act", "training_status")
        metadata = {key: checkpoint.get(key) for key in keys}
        if metadata["side"] != side:
            raise ValueError("Checkpoint owner differs from requested side: " + side)
        completion = receipt["results"][side]["epochs"][-1]
        if (type(completion.get("global_epoch")) is not int or completion["global_epoch"] != 8
                or completion.get("reload_weights_exact") is not True
                or receipt["results"][side].get("value_head_unchanged") is not True):
            raise ValueError("Invalid completed epoch-eight receipt: " + side)
        actual_sha = sha(path)
        expected_sha = receipt["results"][side]["epochs"][-1]["checkpoint_sha256"]
        if actual_sha != expected_sha:
            raise ValueError("Checkpoint differs from completed epoch-eight receipt: " + side)
        state = checkpoint["model_state"]
        layout = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()}
        report["checkpoints"].append(dict(path=str(path.relative_to(root)), sha256=actual_sha,
            metadata=metadata, epoch_in_completion_receipt=receipt["results"][side]["epochs"][-1]["global_epoch"],
            receipt_hash_verified=True, parameter_layout_sha256=hashlib.sha256(canonical_json(layout).encode()).hexdigest(),
            policy_only_receipt=receipt["policy_only"], value_labels_used=receipt["value_labels_used"],
            value_head_unchanged=receipt["results"][side]["value_head_unchanged"],
            assessment=assess_legacy_artifact("checkpoint", metadata).to_dict()))
    admission_path, split_path = base / "policy_admission.json", base / "dataset/split.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha(admission_path) != receipt["admission_sha256"]:
        raise ValueError("Corpus admission differs from the training receipt")
    if sha(split_path) != admission["dataset_hashes"]["split.json"]:
        raise ValueError("Corpus split differs from admission")
    train, heldout = split["train"], split["holdout"]
    if (set(train) & set(heldout) or len(set(train)) != len(train) or len(set(heldout)) != len(heldout)
            or len(train) != admission["train_games"] or len(heldout) != admission["holdout_games"]):
        raise ValueError("Corpus game split has overlap, duplicates or wrong counts")
    if split["learning_contract"] != admission["learning_contract"]:
        raise ValueError("Corpus manifests disagree on semantics")
    headers = {}
    for side in ("fp", "sh"):
        rows = receipt["results"][side]["rows"]
        if rows != receipt["results"][side]["epochs"][-1]["rows"]:
            raise ValueError("Corpus row counts disagree within training receipt: " + side)
        headers.update(_check_arrays(base, side, rows, split["menu_cap"]))
    metadata = {**admission["learning_contract"], "input_dim": CONTRACT.observation_width,
                "action_dim": CONTRACT.action_width, "admission_verified": admission["policy_admitted"] is True}
    report["human_corpus"] = dict(path=str(base.relative_to(root)), admission_sha256=sha(admission_path),
        split_sha256=sha(split_path), train_games=len(train), heldout_games=len(heldout),
        rows_checked_in_original_admission=split["contract_rows_checked"], policy_admitted=admission["policy_admitted"],
        value_admitted=admission["value_admitted"], metadata=metadata, headers=headers,
        menu_cap=split["menu_cap"], capped_menus=split["capped_menus"], array_payloads_rehashed=False,
        menu_indices_and_labels_checked=True,
        assessment=assess_legacy_artifact("human_policy_arrays", metadata).to_dict(),
        limitations=["Full payload integrity must be checked at later admission, not inferred from headers.",
                     "Human candidate menus were capped; they are not complete legal menus for fresh actor sampling."])
    report["legacy_search_artifacts"] = {
        kind: assess_legacy_artifact(kind, {}).to_dict()
        for kind in ("search_trace", "search_targets", "response_policy")}
    report["production_stop_markers"] = {
        str(path): (root / path).is_file() for path in (
            Path("native_backend/runs/own_value/current_policy_sep23_columns_r1/STOP"),
            Path("audit/selfplay_restart_sep21/failure_watcher/STOP"))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), checkpoints=len(report["checkpoints"]),
                         train_games=report["human_corpus"]["train_games"],
                         heldout_games=report["human_corpus"]["heldout_games"],
                         actor_critic_resume_compatible=False)))


if __name__ == "__main__":
    main()
