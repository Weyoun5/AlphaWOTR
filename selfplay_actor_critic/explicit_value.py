"""An EXPLICIT value function on the race scalars, fitted on the human corpus.

User, 2026-10-05: "define an explicit value function, something like shadow = a * shadow_vps + b * fp_vps +
c * corruption + d * fsp_companions + e * companions_on_board + f * fsp_dist_to_mordor; our prior is that the
values of each increase the closer to victory they are (8->10 is more than 0->2 Shadow VPs ...). Fit a-f on the
human corpus."

Two models of P(Shadow wins | position), both logistic:
  linear:    logit = bias + sum_f a_f * x_f                      (the form above, plus the turn)
  staircase: logit = bias + sum_f g_f(x_f), g_f a monotone step function of the feature's integer level with
             the sign fixed by the prior (Shadow VPs up, FP VPs down, corruption up, distance to Mordor up,
             Mordor steps down, companions down). Every step is free, so 8->10 may weigh more than 0->2.
Fitted on the training games of the corpus (split.json), measured on its held-out games: log-loss and the share
of the result's variance explained, overall and by turn, beside the neural value network for comparison.

  python -m selfplay_actor_critic.explicit_value --output selfplay_actor_critic/seeds/explicit_value_a.json

The fitted function is saved as JSON (levels and values per feature) so that any observation can be valued
without the networks (value_of(features) below; event credit in event_trace.py reads it).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
from .report_metrics import metric_text, r2_score

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "audit/2026-09-04/performance_pass/card_followup_reextract_sep14_full_r9"
from .race_features import (COMPANION_BASE, COMPANIONS, FELLOWSHIP, FP_NATIONS, MY_ENEMY, NATION_COUNT, REGION_STRIDE,  # noqa: E402
                            REINFORCEMENT_BASE, SH_NATIONS, TURN, UNITS_ON_BOARD_BASE)

MAX_MORDOR_DISTANCE = 13          # src/ai.py _MAX_MORDOR_DIST: mordor_proximity = 1 - distance / 13

# (name, observation index, scale: raw = round(value * scale), sign of the prior for the Shadow, levels)
# The vp slots are side-relative (my, enemy): resolved per row by the mover's side.
FEATURES = [
    ("sh_vp", None, 10, +1, 11),
    ("fp_vp", None, 10, -1, 6),
    ("corruption", FELLOWSHIP["corruption"], 12, +1, 13),
    ("fs_distance", None, 1, +1, 14),           # regions from the Fellowship to the Mordor gate, less its hidden progress
    ("mordor_step", FELLOWSHIP["mordor_step"], 5, -1, 6),
    ("companions_in_fs", FELLOWSHIP["companions_in_fs"], 8, -1, 9),
    ("companions_on_board", None, 1, -1, 10),
    ("turn", TURN, 15, +1, 20),
]
# Optional army terms (--armies): units on the board per side, in pairs (a casualty or a muster of two units
# moves one level). Without them the value prices no army loss, so a failed assault costs nothing.
ARMY_FEATURES = [
    ("sh_army", None, 1, +1, 37),
    ("fp_army", None, 1, -1, 26),
]
ARMY_BIN = 2
REGIONS = 105
# Cost terms (--costs, user 2026-10-05 "add dice spent and units lost as costs"): the action dice each side
# still has this turn (every die spent, on an action or on the hunt, lowers it; both are 0 before the roll),
# and the Free Peoples units permanently dead (Shadow casualties return to the pool, so for the Shadow the
# units-on-board term is the loss term).
COST_FEATURES = [
    ("sh_dice", None, 1, +1, 11),
    ("fp_dice", None, 1, -1, 7),
    ("fp_dead", None, 1, +1, 39),
]
# src/ai.py NATION_MAX_FIGURES for the Free Peoples nations, in nation order (regulars, elites)
FP_MAX_FIGURES = {0: (5, 10), 1: (5, 5), 2: (15, 5), 3: (10, 5), 4: (10, 5)}


def use_armies(enabled=True, costs=False):
    """Switch the optional terms on (or off) for every FEATURES reader in this process."""
    for entry in ARMY_FEATURES + COST_FEATURES:
        if entry in FEATURES:
            FEATURES.remove(entry)
    extra = (ARMY_FEATURES if enabled else []) + (COST_FEATURES if costs else [])
    FEATURES[len(FEATURES) - 1:len(FEATURES) - 1] = extra          # before the turn


def optional_flags():
    """(armies on, costs on) in this process."""
    return any(e in FEATURES for e in ARMY_FEATURES), any(e in FEATURES for e in COST_FEATURES)


def features_of(X, side):
    """Integer-level features (absolute, Shadow's view) from observation rows of one side ('fp'|'sh')."""
    out = {}
    my, en = MY_ENEMY["vp"]
    sh_slot, fp_slot = (my, en) if side == "sh" else (en, my)
    for name, index, scale, _, levels in FEATURES:
        if name == "sh_vp":
            raw = X[:, sh_slot] * scale
        elif name == "fp_vp":
            raw = X[:, fp_slot] * scale
        elif name == "companions_on_board":
            alive = sum(X[:, COMPANION_BASE + 4 * j] for j in range(len(COMPANIONS)))
            in_fs = sum(X[:, COMPANION_BASE + 4 * j + 1] for j in range(len(COMPANIONS)))
            raw = alive - in_fs
        elif name in ("sh_army", "fp_army"):
            blocks = X[:, :REGIONS * REGION_STRIDE].reshape(len(X), REGIONS, REGION_STRIDE)
            counts = blocks[:, :, 14:14 + 2 * NATION_COUNT].reshape(len(X), REGIONS, NATION_COUNT, 2)
            units = np.rint(counts[:, :, :, 0] * 15.) + np.rint(counts[:, :, :, 1] * 5.)
            nations = list(SH_NATIONS) if name == "sh_army" else list(FP_NATIONS)
            raw = units[:, :, nations].sum((1, 2)) // ARMY_BIN
        elif name in ("sh_dice", "fp_dice"):
            my, en = MY_ENEMY["dice_unused"]
            mine = (name == "sh_dice") == (side == "sh")
            raw = X[:, my if mine else en] * 10.
        elif name == "fp_dead":
            raw = np.zeros(len(X))
            for i, (regulars, elites) in FP_MAX_FIGURES.items():
                raw += regulars * np.clip(1. - X[:, UNITS_ON_BOARD_BASE + 2 * i] - X[:, REINFORCEMENT_BASE + 3 * i], 0., 1.)
                raw += elites * np.clip(1. - X[:, UNITS_ON_BOARD_BASE + 2 * i + 1] - X[:, REINFORCEMENT_BASE + 3 * i + 1], 0., 1.)
            raw = np.rint(raw) // ARMY_BIN
        elif name == "fs_distance":
            # The figure on the board (last known position) is `distance` regions from the gate; the hidden
            # Fellowship may already be `progress` regions further, so a reveal moves the figure without
            # changing this. Inside Mordor the distance is 0 (mordor_step counts from there).
            distance = np.rint((1. - X[:, FELLOWSHIP["mordor_proximity"]]) * MAX_MORDOR_DISTANCE)
            progress = np.rint(X[:, FELLOWSHIP["fs_progress"]] * 12.)
            raw = np.where(X[:, FELLOWSHIP["mordor_entered"]] > .5, 0., np.maximum(distance - progress, 0.))
        else:
            raw = X[:, index] * scale
        out[name] = np.clip(np.rint(raw).astype(np.int64), 0, levels - 1)
    return out


def _corpus_blocks(path):
    """(mover 'sh'|'fp', observation rows, Shadow won, turn) per side of one corpus game, Shadow first: the row
    order of load_corpus, shared with _compare_with_network so that both value the same rows."""
    with np.load(path, allow_pickle=True) as d:
        X, side, z, turn = d["X"], d["side"], d["z"], d["turn"]
    blocks = []
    for code, name in ((1, "sh"), (0, "fp")):
        m = side == code
        if m.any():
            blocks.append((name, X[m], (z[m] == 1) if name == "sh" else (z[m] == -1), turn[m]))
    return blocks


def load_game(path):
    rows = {}
    for name, X, sh_wins, turn in _corpus_blocks(path):
        f = features_of(X, name)
        f["sh_wins"] = sh_wins
        f["turn_raw"] = turn
        rows[name] = f
    return rows


def load_corpus(limit=0):
    split = json.loads((CORPUS / "consolidated_r1/dataset/split.json").read_text(encoding="utf-8"))
    sets = {}
    for part in ("train", "holdout"):
        files = [str(CORPUS / "run/npz" / name) for name in split[part] if (CORPUS / "run/npz" / name).exists()]
        if limit:
            files = files[:limit]
        cols = {}
        with ThreadPoolExecutor(6) as pool:
            for g, rows in enumerate(pool.map(load_game, files)):
                for side, f in rows.items():
                    for k, v in f.items():
                        cols.setdefault(k, []).append(v)
                    cols.setdefault("game", []).append(np.full(len(f["sh_wins"]), g))
                    cols.setdefault("side", []).append(np.full(len(f["sh_wins"]), side == "sh"))
        sets[part] = {k: np.concatenate(v) for k, v in cols.items()}
        sets[part]["games"] = len(files)
        sets[part]["sources"] = [("corpus", f) for f in files]      # the rows' origin, in row order
    return sets


def _selfplay_blocks(folder):
    """(mover 'sh'|'fp', observation rows, Shadow won, turn) per side of one self-play game (valuedata.py run),
    Shadow first; None unless the game terminated. The row order of load_selfplay, shared with
    _compare_with_network so that both value the same rows."""
    with open(os.path.join(folder, "episode.json"), encoding="utf-8") as stream:
        meta = json.load(stream)
    end = meta.get("end") or {}
    if end.get("status") != "terminated":
        return None
    with np.load(os.path.join(folder, "arrays.npz")) as d:
        X = d["observations"].astype(np.float32)
    sides = np.array([r["side"] for r in meta["rows"]])
    turn = np.array([r["turn"] for r in meta["rows"]], np.int64)
    blocks = []
    for name in ("sh", "fp"):
        m = sides == name
        if m.any():
            blocks.append((name, X[m], np.full(int(m.sum()), end["winner"] == "sh"), turn[m]))
    return blocks


def _selfplay_game(task):
    """Integer features per decision of one self-play game (valuedata.py run), by the mover's side."""
    folder, (armies, costs) = task
    use_armies(armies, costs)                            # worker processes start with the default list
    blocks = _selfplay_blocks(folder)
    if blocks is None:
        return None
    rows = {}
    for name, X, sh_wins, turn in blocks:
        f = features_of(X, name)
        f["sh_wins"] = sh_wins
        f["turn_raw"] = turn
        rows[name] = f
    return rows


def load_selfplay(runs, holdout_fraction=.1, limit=0):
    """Self-play runs as (train, holdout) sets of the same shape as load_corpus, split by game."""
    from concurrent.futures import ProcessPoolExecutor
    folders = []
    for run in runs:
        folders += sorted(glob.glob(os.path.join(run, "batches", "*", "training_workers", "worker_*", "training", "raw-*")))
    if limit:
        folders = folders[:limit]
    sets = {"train": {}, "holdout": {}}
    counts = {"train": 0, "holdout": 0}
    sources = {"train": [], "holdout": []}
    with ProcessPoolExecutor(8) as pool:
        flags = optional_flags()
        for g, rows in enumerate(pool.map(_selfplay_game, [(f, flags) for f in folders], chunksize=64)):
            if rows is None:
                continue
            part = "holdout" if (g % 1000) < 1000 * holdout_fraction else "train"
            cols = sets[part]
            for side, f in rows.items():
                for k, v in f.items():
                    cols.setdefault(k, []).append(v)
                cols.setdefault("game", []).append(np.full(len(f["sh_wins"]), counts[part]))
                cols.setdefault("side", []).append(np.full(len(f["sh_wins"]), side == "sh"))
            counts[part] += 1
            sources[part].append(("selfplay", folders[g]))
    for part in sets:
        sets[part] = {k: np.concatenate(v) for k, v in sets[part].items()}
        sets[part]["games"] = counts[part]
        sets[part]["sources"] = sources[part]                       # the rows' origin, in row order
    return sets


def design(data, kind):
    """linear: one column per feature (its level); staircase: one indicator per step of each feature."""
    columns, names = [], []
    for name, _, _, sign, levels in FEATURES:
        x = data[name]
        if kind == "linear":
            columns.append(sign * x.astype(np.float64))
            names.append(name)
        else:
            for level in range(1, levels):
                columns.append(sign * (x >= level).astype(np.float64))
                names.append(f"{name}>={level}")
    return np.stack(columns, axis=1), names


def fit_logistic(A, y, *, positive, l2=1e-4, iterations=300):
    """Logistic regression by L-BFGS; `positive` constrains the slope columns to be >= 0 (the prior's direction,
    already folded into the columns' signs) by fitting softplus-parameterised weights."""
    import torch
    A = torch.from_numpy(A)
    y = torch.from_numpy(y.astype(np.float64))
    w = torch.full((A.shape[1],), -4. if positive else 0., dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=iterations, line_search_fn="strong_wolfe")

    def weights():
        return torch.nn.functional.softplus(w) if positive else w          # softplus >= 0: every step in the prior's direction

    def closure():
        optimizer.zero_grad()
        logits = A @ weights() + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + l2 * (weights() ** 2).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return weights().detach().numpy(), float(b.item())


def evaluate(logits, y, turn, label):
    p = 1. / (1. + np.exp(-logits))
    eps = 1e-6
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    z = 2. * y - 1.
    v = 2. * p - 1.
    explained = r2_score(z, v)
    cells = []
    for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 20)):
        m = (turn >= lo) & (turn <= hi)
        if m.sum() > 1000:
            cells.append(f"turns {lo}-{hi} {metric_text(r2_score(z[m], v[m]), '4.1f', scale=100, suffix='%')}")
    print(f"  {label:34s} log-loss {loss:.4f} | result variance explained {metric_text(explained, '4.1f', scale=100, suffix='%')} | " + "  ".join(cells))
    return explained


def configure(model):
    """Set FEATURES to what a saved model was fitted with (the army terms on or off)."""
    use_armies("sh_army" in model["values"], "sh_dice" in model["values"])


def value_of(model, features):
    """V(position) in [-1, 1] for the Shadow from integer-level features (dict of arrays), per the saved JSON."""
    logits = np.full(len(next(iter(features.values()))), model["bias"], np.float64)
    for name, table in model["values"].items():
        logits += np.asarray(table, np.float64)[np.clip(features[name], 0, len(table) - 1)]
    return np.tanh(logits / 2.)                     # 2 * sigmoid - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0, help="games per split (smoke test)")
    parser.add_argument("--armies", action="store_true", help="add the units-on-board terms (ARMY_FEATURES)")
    parser.add_argument("--costs", action="store_true", help="add the cost terms: unused dice per side, Free Peoples units dead (COST_FEATURES)")
    parser.add_argument("--selfplay", nargs="*", default=[], metavar="RUN",
                        help="fit on these self-play runs (valuedata.py) instead of the human corpus; every 10th game held out")
    parser.add_argument("--critic-dir", default=str(ROOT / "selfplay_actor_critic/seeds/full_critic_r2"),
                        help="neural value network scored on the same held-out rows for comparison ('' = skip)")
    args = parser.parse_args()
    use_armies(args.armies, args.costs)
    sets = load_selfplay(args.selfplay, limit=args.limit) if args.selfplay else load_corpus(args.limit)
    train, hold = sets["train"], sets["holdout"]
    print(f"corpus: {train['games']} training games ({len(train['sh_wins'])} decisions), {hold['games']} held-out games "
          f"({len(hold['sh_wins'])} decisions); Shadow won {100 * train['sh_wins'][train['game'] != np.roll(train['game'], 1)].mean():.1f}% of the training games")
    y_train, y_hold = train["sh_wins"].astype(np.float64), hold["sh_wins"].astype(np.float64)
    turn_hold = hold["turn_raw"]
    models = {}
    for kind in ("linear", "staircase"):
        A, names = design(train, kind)
        w, b = fit_logistic(A, y_train, positive=True)
        H, _ = design(hold, kind)
        explained = evaluate(H @ w + b, y_hold, turn_hold, f"{kind} (prior signs enforced)")
        models[kind] = (w, b, names, explained)
    # The staircase, free of the sign prior: does the prior cost anything?
    A, names = design(train, "staircase")
    w_free, b_free = fit_logistic(A, y_train, positive=False)
    H, _ = design(hold, "staircase")
    evaluate(H @ w_free + b_free, y_hold, turn_hold, "staircase, signs free")
    print("linear model (logit of P(Shadow wins), per unit of the feature, prior sign applied):")
    w, b, names, _ = models["linear"]
    for name, coefficient, (_, _, _, sign, _) in zip(names, w, FEATURES):
        print(f"   {name:22s} {sign * coefficient:+.3f}")
    print(f"   bias {b:+.3f}")
    print("staircase model: the value of each level of each feature (logit; 0 at level 0), prior sign applied:")
    w, b, names, _ = models["staircase"]
    values, k = {}, 0
    for name, _, _, sign, levels in FEATURES:
        steps = w[k:k + levels - 1]
        k += levels - 1
        table = np.concatenate(([0.], np.cumsum(sign * steps)))
        values[name] = table.tolist()
        print(f"   {name:22s} " + " ".join(f"{v:+.2f}" for v in table))
    if args.critic_dir:
        _compare_with_network(args, hold, y_hold, turn_hold)
    if args.output:
        saved = dict(schema="explicit_value_v1", features=[dict(name=n, index=i, scale=s, sign=g, levels=l) for n, i, s, g, l in FEATURES],
                     bias=float(models["staircase"][1]), values=values, held_out_explained=models["staircase"][3],
                     linear=dict(bias=float(models["linear"][1]), weights={n: float(g * c) for n, c, (_, _, _, g, _) in zip(models["linear"][2], models["linear"][0], FEATURES)},
                                 held_out_explained=models["linear"][3]))
        Path(args.output).write_text(json.dumps(saved, indent=1), encoding="utf-8")
        print(f"written {args.output}")


def _compare_with_network(args, hold, y_hold, turn_hold):
    """The neural value network on the same held-out rows (its own view per side; sign flipped to the Shadow's).
    The rows are re-read from the held-out set's own sources (self-play game folders with --selfplay, corpus
    files otherwise) in the loader's row order, and must reproduce y_hold and turn_hold exactly."""
    if "sources" not in hold:
        raise ValueError("the held-out set carries no row sources: the network cannot be scored on the same rows")
    readers = {"selfplay": _selfplay_blocks, "corpus": _corpus_blocks}
    from .generation import require_generation
    require_generation("explicit_value._compare_with_network")
    os.environ.setdefault("WOTR_NATIVE", "0")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    import torch
    from .policy import load_critic
    critics = {s: load_critic(ai, Path(args.critic_dir) / f"{s}_critic.pt", s) for s in ("fp", "sh")}
    v, y, t = [], [], []
    with torch.inference_mode():
        for kind, path in hold["sources"]:
            blocks = readers[kind](path)
            if blocks is None:
                raise RuntimeError(f"held-out game {path} is no longer a terminated game")
            for name, X, sh_wins, turn in blocks:
                c = critics[name]
                sign = 1. if name == "sh" else -1.
                x = torch.from_numpy(X.astype(np.float32))
                v.append(sign * c.value_head(c.forward_backbone(x)[0]).reshape(-1).numpy())
                y.append(sh_wins)
                t.append(turn)
    if not v:
        raise ValueError("no held-out rows to score the network on")
    v, y, t = np.concatenate(v), np.concatenate(y).astype(np.float64), np.concatenate(t)
    if not (np.array_equal(y, y_hold) and np.array_equal(t, turn_hold)):
        raise RuntimeError(f"the network's held-out rows ({len(y)}) differ from the explicit model's ({len(y_hold)}): "
                           "results or turns do not match row for row")
    logits = 2. * np.arctanh(np.clip(v, -.999, .999))
    evaluate(logits, y_hold, turn_hold, f"neural {Path(args.critic_dir).name}")


if __name__ == "__main__":
    main()
