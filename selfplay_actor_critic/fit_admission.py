"""Admission checks for optional fits before models or reports are published."""
import math

import numpy as np

from .contracts import ContractError


def require_fit_schedule(*, epochs, batch_size=1, learning_rate=1.):
    if type(epochs) is not int or epochs < 1:
        raise ContractError("Fitting requires a positive epoch count")
    if type(batch_size) is not int or batch_size < 1:
        raise ContractError("Fitting requires a positive batch size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ContractError("Fitting requires a finite positive learning rate")


def require_game_partitions(train, held, *, context):
    if not len(train) or not len(held):
        raise ContractError(f"{context} requires nonempty training and held-out game partitions")


def require_critic_partitions(train, held):
    require_game_partitions(train, held, context="Critic fitting")
    # Selection scores use own pre-decision rows. Training may also use
    # afterstates or the other player's positions as seen in stored watch rows.
    for code, side in enumerate(("fp", "sh")):
        if not any(np.any((entry["side"] == code) & entry["learned"]) for entry in held):
            raise ContractError(f"Critic fitting requires {side} learned decision rows in the held-out partition")


def require_optimizer_steps(steps, *, context):
    if any(count < 1 for count in steps.values()):
        raise ContractError(f"{context} completed no optimizer steps for one or more sides; refusing publication")


def cross_validation_folds(games, folds, seed=11, eligible=None):
    """Keep usable random assignments; repair empty training complements."""
    if type(folds) is not int or folds < 2:
        raise ContractError("Cross-validation requires at least two folds")
    live = np.ones(games, bool) if eligible is None else np.asarray(eligible, bool)
    if live.shape != (games,) or live.sum() < 2:
        raise ContractError("Cross-validation requires at least two eligible games")
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, games)
    if len(np.unique(fold[live])) < 2:
        # Tiny eligible cohorts can all draw one fold. Split those games
        # reproducibly using indices alone, without consulting any outcomes.
        order = rng.permutation(np.flatnonzero(live))
        moved = order[:len(order) // 2]
        fold[moved] = (fold[moved] + 1) % folds
    return fold, live


def cross_validation_groups(games, folds, seed=11, eligible=None):
    """Fisher cells refine both full-cohort and audit-eligible fold masks."""
    full, _ = cross_validation_folds(games, folds, seed)
    fit, live = cross_validation_folds(games, folds, seed, eligible)
    _, groups = np.unique(np.column_stack((full, fit, live)), axis=0, return_inverse=True)
    return groups
