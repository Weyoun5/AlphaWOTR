"""The engine encoding generation of the self-play / evaluation tools: chosen by the CALLER, read in one place.

D6P-SAC-GEN-FROM-ENV (D6 plan S1(d)). The tools used to force WOTR_ACT_GEN=v52 themselves, in-process and in every
spawned worker, right before `import ai`, so a run launched with WOTR_ACT_GEN=v53 was encoded at v52 without a word
and an unset variable quietly meant v52. Now:
  * the caller sets WOTR_ACT_GEN explicitly (the pre-D6 drivers say v52; the D6 drivers, and everything since the D7
    install of 2026-10-26 made v53 the deployed generation, say v53); a tool never writes it;
  * require_generation() refuses loudly when it is unset, blank or not a self-play generation (contracts.Contract
    knows v52 and v53 only), and when an already-imported engine (`ai`) or contract module was built for another
    generation: both read the variable ONCE, at import, so they cannot serve a different one later;
  * a pool built with generation_pool_kwargs() hands every worker the parent's generation; the worker inherits the
    environment (Windows spawn) and pin_worker_generation() refuses to start under any other (BrokenProcessPool in the
    parent, the GenerationError traceback on stderr);
  * bc_seed_dir() names the caller's generation's copy of the frozen behaviour-cloning pairs the tools default to.
No encoding is touched here; this module only reads.
"""
import os
from pathlib import Path
import sys

ENV = "WOTR_ACT_GEN"
SUPPORTED = ("v52", "v53")
_CONTRACTS = __name__.rpartition(".")[0] + ".contracts"
_ROOT = Path(__file__).resolve().parents[1]
# R17-SAC row 67. The frozen behaviour-cloning pairs the tools default to (epoch 6 = the evaluation opponent, epoch 8 =
# the untouched imitation base, offline_eval's search-tree model of the other player), as folders of
# {fp,sh}_epoch1.pt, one per generation. The 2026-09-15 originals under SEED_BASE
# (audit/card_gap_followthrough_sep14/epoch_extension_to8_r1/run_r1/epoch_00{6,8}/policy_run) carry action contract
# observed_combat_choices_v7, which every current loader refuses since the 2026-10-19 bump to v8. Their copies:
#   v52: the same weights relabelled v7 -> v8 (D6P-V8-SEED-COPIES, scripts/make_v8_seed_copies.py, receipt.json there);
#   v53: those v8 copies widened by zero columns (D6P-CKPT-V52-TO-V53, scripts/migrate_v52_to_v53_checkpoint.py,
#        a receipt beside each file), the folders the FP league plays as E6 / E8 (d6sp/fpsp/fpsp_common.SEEDS_V53).
#        They live in that scratch folder: if it is archived, move these entries (and SEEDS_V53) with it. A missing
#        copy makes bc_seed_dir refuse loudly (FileNotFoundError naming the folder); it never falls back.
BC_SEED_DIRS = {
    "v52": {6: _ROOT / "audit/static_fixes_oct19/models_v8/d6_seed_copies/epoch_006/policy_run",
            8: _ROOT / "audit/static_fixes_oct19/models_v8/d6_seed_copies/epoch_008/policy_run"},
    "v53": {6: _ROOT / "claude_bugscan_2026-10-21/d6sp/scratch/seeds_v53/e6",
            8: _ROOT / "claude_bugscan_2026-10-21/d6sp/scratch/seeds_v53/e8"},
}


class GenerationError(RuntimeError):
    """The encoding generation is unset, unknown or inconsistent with an imported module; it is never defaulted."""


def require_generation(where):
    """The caller's encoding generation ('v52' or 'v53'), normalised as ai.py reads it; refuses instead of defaulting."""
    raw = os.environ.get(ENV)
    if raw is None or not raw.strip():
        # R17-SAC rows 61/78: v53 has been the deployed generation since the D7 install (wotr_bridge/launch_head.py
        # NET_GEN); the test of this item reads NET_GEN, so the next deployment turns it red until this text follows.
        raise GenerationError(
            f"{where}: {ENV} is unset. Set it explicitly to the generation of the nets and data this run uses "
            f"(v53 = the deployed generation since D7, 2026-10-26; v52 = the earlier generation, for its own older nets "
            f"and data); the self-play tools never choose it themselves.")
    generation = raw.strip().lower()
    if generation not in SUPPORTED:
        raise GenerationError(f"{where}: {ENV}={raw!r} is not a self-play generation (supported: "
                              f"{', '.join(SUPPORTED)})")
    engine = sys.modules.get("ai")
    for name in ("ACT_GEN", "_OBSERVATION_GENERATION"):
        loaded = getattr(engine, name, None)
        if isinstance(loaded, str) and loaded != generation:
            raise GenerationError(
                f"{where}: the engine module ai was imported under {ENV}={loaded or '(unset)'} ({name}) but the "
                f"caller asks for {generation}; ai reads the generation once, at import - start a fresh process")
    contracts = sys.modules.get(_CONTRACTS)
    flag = getattr(contracts, "_ENGINE_V53", None)
    if isinstance(flag, bool) and flag != (generation == "v53"):
        raise GenerationError(
            f"{where}: {_CONTRACTS} was imported for {'v53' if flag else 'v52'} but the caller asks for {generation}; "
            f"it reads the generation once, at import - start a fresh process")
    return generation


def pin_worker_generation(parent):
    """ProcessPoolExecutor initializer: this worker must run exactly the parent's generation."""
    generation = require_generation(f"pool worker pid {os.getpid()}")
    if generation != parent:
        raise GenerationError(f"pool worker pid {os.getpid()}: inherited {ENV}={generation} but the parent runs "
                              f"{parent!r}; refusing to mix generations")
    return generation


def generation_pool_kwargs(where):
    """ProcessPoolExecutor(..., **generation_pool_kwargs(where)): refuse in the parent before any worker starts, and
    pin every worker to the parent's generation."""
    return dict(initializer=pin_worker_generation, initargs=(require_generation(where),))


def bc_seed_dir(epoch, where):
    """The folder of {fp,sh}_epoch1.pt holding the epoch-`epoch` (6 or 8) behaviour-cloning pair at the caller's
    generation (BC_SEED_DIRS). Refuses an unset generation (require_generation), another epoch, and a missing copy."""
    generation = require_generation(where)
    folder = BC_SEED_DIRS[generation].get(epoch)
    if folder is None:
        raise GenerationError(f"{where}: no {generation} copy of the epoch-{epoch} behaviour-cloning pair "
                              f"(known epochs: {', '.join(str(e) for e in sorted(BC_SEED_DIRS[generation]))})")
    missing = [side for side in ("fp", "sh") if not (folder / f"{side}_epoch1.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"{where}: the {generation} epoch-{epoch} behaviour-cloning pair is missing "
                                f"{', '.join(missing)}_epoch1.pt under {folder}")
    return folder
