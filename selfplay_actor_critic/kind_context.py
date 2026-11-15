"""The CONTEXT of a per-kind logit offset (Codex's "cheap representation experiment", 2026-10-11).

A plain per-kind bias (policy.KindBias) lets the policy say "attack more"; conditioned on the position it can say
"defend the exposed stronghold now, attack later". The context is a few integer levels read from the side's OWN
observation at the decision (never the other side's view), so the same function serves the fit (pg_fit.py
--features kindctx, on the stored observations) and play (policy.KindContextBias, on the live observation):

    turn        1-3 | 4-6 | 7-9 | 10+
    sh_vp       0-3 | 4-6 | 7+                    (Shadow victory points, whichever side is looking)
    corruption  0-5 | 6-9 | 10+
    fs_distance 0-3 | 4-7 | 8+                    (regions from the Fellowship's figure to the Mordor gate, less its
                                                   hidden progress; explicit_value.py's definition)
    exposure    0 | 1 | 2+                        (own strongholds/cities, still held, with an enemy army inside or
                                                   in an adjacent region)

rows() turns the levels into the regressor of one kind's offsets: 1 (the plain offset) and an indicator per
(factor, level >= 1), WIDTH = 12 numbers; the offset of a candidate of kind k is bias[k] . rows(context). With 25
kinds that is 300 parameters. The score function of bias[k, j] is rows_j x (1[played kind = k] - P(kind k)), so
pg_fit's kind regressors times the context rows are exact (design(..., ["kindctx"])).

  python -m selfplay_actor_critic.kind_context check --run RUN --side sh --policy seeds/kindctx_sh_a/rms_0.03 --games 20
replays stored games and checks that the exported policy's logit changes equal the fitted step (last-layer part
from pg.json + bias . rows) on every candidate, and that the live context equals the one read from the stored
(float16) observations.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .race_features import CAPTURED_BY_ENEMY, FELLOWSHIP, FP_NATIONS, MY_ENEMY, NATION_COUNT, REGION_STRIDE, SH_NATIONS, TURN

ROOT = Path(__file__).resolve().parents[1]
REGIONS = 105
MAX_MORDOR_DISTANCE = 13          # src/ai.py _MAX_MORDOR_DIST (explicit_value.MAX_MORDOR_DISTANCE)
SPEC = "kindctx-v1"
# (factor, thresholds): the level is the number of thresholds the value exceeds
FACTORS = (("turn", (3, 6, 9)), ("sh_vp", (3, 6)), ("corruption", (5, 9)), ("fs_distance", (3, 7)), ("exposure", (0, 1)))


def _band_label(name, thresholds, level):
    """Label of rows()' column 1[level == l] (a one-hot BAND, not a cumulative 'value > t'): the integer values of the
    band, 'turn 4-6', 'turn 10+', 'exposure 1' (fix wave X4-05: the old 'turn>3' read as every turn above 3)."""
    lo = thresholds[level - 1] + 1
    if level == len(thresholds):
        return f"{name} {lo}+"
    hi = thresholds[level]
    return f"{name} {lo}" if hi == lo else f"{name} {lo}-{hi}"


COLUMNS = ("base",) + tuple(_band_label(name, thresholds, level)
                            for name, thresholds in FACTORS for level in range(1, len(thresholds) + 1))
WIDTH = len(COLUMNS)
_TABLES = {}


def _engine():
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    from .generation import require_generation
    require_generation("kind_context._engine")
    os.environ.setdefault("WOTR_NATIVE", "0")
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    return ai


def tables(side):
    """Static per side: [(region index, neighbour indices)] of the side's own settlements, and the enemy nations."""
    if side not in _TABLES:
        from .race_features import settlement_table
        rows = settlement_table(_engine(), side)
        owner = 1 if side == "sh" else 0
        own = tuple((int(r["region"]), tuple(int(n) for n in r["neighbors"])) for r in rows if r["owner"] == owner)
        _TABLES[side] = (own, list(SH_NATIONS if side == "fp" else FP_NATIONS))
    return _TABLES[side]


def values(observations, side):
    """Raw integer factor values [N, len(FACTORS)] from observation rows in the view of `side`."""
    X = np.asarray(observations, np.float32)
    if X.ndim == 1:
        X = X[None]
    n = len(X)
    my, en = MY_ENEMY["vp"]
    sh_vp = np.rint(X[:, my if side == "sh" else en] * 10.)
    corruption = np.rint(X[:, FELLOWSHIP["corruption"]] * 12.)
    distance = np.rint((1. - X[:, FELLOWSHIP["mordor_proximity"]]) * MAX_MORDOR_DISTANCE)
    progress = np.rint(X[:, FELLOWSHIP["fs_progress"]] * 12.)
    fs_distance = np.where(X[:, FELLOWSHIP["mordor_entered"]] > .5, 0., np.maximum(distance - progress, 0.))
    turn = np.rint(X[:, TURN] * 15.)
    own, enemy_nations = tables(side)
    blocks = X[:, :REGIONS * REGION_STRIDE].reshape(n, REGIONS, REGION_STRIDE)
    counts = blocks[:, :, 14:14 + 2 * NATION_COUNT].reshape(n, REGIONS, NATION_COUNT, 2)[:, :, enemy_nations, :]
    enemy_units = (np.rint(counts[:, :, :, 0] * 15.) + np.rint(counts[:, :, :, 1] * 5.)).sum(2) > 0      # [N, REGIONS]
    exposure = np.zeros(n)
    for region, neighbours in own:
        held = blocks[:, region, CAPTURED_BY_ENEMY] < .5
        threatened = enemy_units[:, region]
        if neighbours:
            threatened = threatened | enemy_units[:, list(neighbours)].any(1)
        exposure += held & threatened
    return np.column_stack([turn, sh_vp, corruption, fs_distance, exposure]).astype(np.int64)


def levels(observations, side):
    """Factor levels [N, len(FACTORS)] int8."""
    v = values(observations, side)
    out = np.zeros(v.shape, np.int8)
    for j, (_, thresholds) in enumerate(FACTORS):
        for t in thresholds:
            out[:, j] += (v[:, j] > t)
    return out


def rows(levels):
    """Regressor rows [N, WIDTH] float32 of the levels: 1, then 1[level_j == l] per factor j and level l >= 1."""
    levels = np.asarray(levels)
    if levels.ndim == 1:
        levels = levels[None]
    out = np.zeros((len(levels), WIDTH), np.float32)
    out[:, 0] = 1.
    col = 1
    for j, (_, thresholds) in enumerate(FACTORS):
        for level in range(1, len(thresholds) + 1):
            out[:, col] = levels[:, j] == level
            col += 1
    return out


def describe(bias, names, scale=1.):
    """Lines of a [kinds, WIDTH] offset table (logits); each non-base column is the offset inside that level band only."""
    w = max(13, max(len(c) for c in COLUMNS) + 1)
    lines = ["    " + f"{'kind':22s}" + "".join(f"{c:>{w}s}" for c in COLUMNS)]
    for name, row in zip(names, np.asarray(bias)):
        lines.append("    " + f"{name:22s}" + "".join(f"{scale * v:+{w}.3f}" for v in row))
    return lines


def check(args):
    """Replay stored games with the base and the exported policy; every candidate's logit change must equal the
    fitted step: theta_last . h_last(a) (+ theta_delta . h_delta(a)) + bias[kind(a)] . rows(context), and the live
    context must equal the stored observations' context."""
    from .offline_eval import _load_actor
    from .policy import forward_ragged
    from .replay_menus import game_folders, load_episode
    from .replay_menus import replay as replay_game
    from .valuedata import LastLayer
    import torch
    ai = _engine()
    folder = Path(args.policy)
    pg = json.load(open(folder / "pg.json", encoding="utf-8"))
    theta = np.asarray(pg["theta"], np.float64)
    features = pg["features"]
    # fix wave S2-07: an explicit --base wins; pg.json's base_policy is only the default (pg_fit always records it)
    base_policy = str(args.base) if args.base else pg.get("base_policy")
    if not base_policy:
        raise ValueError(f"no base policy: pass --base, or export {folder} with pg_fit so pg.json records base_policy")
    base = _load_actor(ai, base_policy, args.side)
    new = _load_actor(ai, str(folder), args.side)
    ctx = getattr(new.model, "kind_ctx_bias", None)
    names = pg.get("kinds", [])
    heads = [("last", base.model.spatial_action_scorer, 64), ("delta", base.model.policy_delta, 128)]
    taps = {name: LastLayer(head) for name, head, _ in heads}
    start = 0
    weights = {}
    for name, head, width in heads:
        if name in features:
            width = int(head[-1].in_features)
            weights[name] = theta[start:start + width]
            start += width
    if "kind" in features:
        kind_theta = dict(zip(names, theta[start:start + len(names)]))
        start += len(names)
    else:
        kind_theta = {}
    ctx_theta = theta[start:start + len(names) * WIDTH].reshape(len(names), WIDTH) if "kindctx" in features else None
    worst, decisions, level_mismatch, games_done = 0., 0, 0, 0
    for game in game_folders(args.run)[:args.games]:
        meta, observations, _ = load_episode(game)
        own_rows = [k for k, r in enumerate(meta["rows"]) if r["side"] == args.side]
        stored_levels = levels(observations[own_rows], args.side) if own_rows else np.zeros((0, len(FACTORS)), np.int8)
        seen = {}

        def on_decision(k, frame, selected):
            nonlocal worst, decisions, level_mismatch
            if frame.side != args.side:
                return
            inputs = frame.inputs
            live = levels(inputs.observation, args.side)[0]
            if not np.array_equal(live, stored_levels[own_rows.index(k)]):
                level_mismatch += 1
            for tap in taps.values():
                tap.parts.clear()
            with torch.inference_mode():
                _, (logits_base,) = forward_ragged(base.model, [inputs])
                h = {name: taps[name].take() for name in taps}
                _, (logits_new,) = forward_ragged(new.model, [inputs])
            predicted = np.zeros(len(inputs.candidates))
            for name in weights:
                predicted += h[name].astype(np.float64) @ weights[name]
            kinds = list(inputs.kinds)
            predicted += np.array([kind_theta.get(kd, 0.) for kd in kinds])
            if ctx_theta is not None:
                row = rows(live)[0].astype(np.float64)
                index = {n: i for i, n in enumerate(names)}
                predicted += np.array([ctx_theta[index[kd]] @ row if kd in index else 0. for kd in kinds])
            actual = logits_new.cpu().numpy().astype(np.float64) - logits_base.cpu().numpy().astype(np.float64)
            worst = max(worst, float(np.abs(actual - predicted).max()))
            decisions += 1

        try:
            replay_game(ai, meta, observations, on_decision)
        except RuntimeError as error:
            print(f"  seed {meta['seed']}: replay differs from the stored game ({str(error)[:80]}); skipped")
            continue
        games_done += 1
    print(f"{games_done} games, {decisions} {args.side} decisions: largest |actual - predicted| logit change {worst:.2e}; "
          f"live context != stored context at {level_mismatch} decisions"
          + (f"; the exported policy's kind_ctx_bias covers {len(ctx.vocab)} kinds" if ctx is not None else "; no kind_ctx_bias in the policy"))
    return worst, level_mismatch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--side", required=True, choices=("fp", "sh"))
    c.add_argument("--policy", required=True, help="an exported pg_fit step folder (rms_<size>) with pg.json")
    c.add_argument("--base", default="", help="the policy the step was added to (default: pg.json's base_policy)")
    c.add_argument("--games", type=int, default=20)
    args = parser.parse_args()
    if args.command == "check":
        check(args)


if __name__ == "__main__":
    main()
