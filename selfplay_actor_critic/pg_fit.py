"""A direct policy gradient on the action scorer's last layer, from games alone.

valuedata.py --gradient plays games between two policies and stores, per decision, the centred last-layer
activations of the played move x = h(played) - E_policy[h] (64 numbers; h feeds the final linear layer of
the action scorer). The move is a sample of the policy, so if the logits are tilted by theta . h the
first-order change of the side's expected result is theta . g with

    g = mean over games of sum over the side's decisions of x * (result - baseline) / T

(the score-function identity; baseline = any function of the position, here a linear function of a value
network's V(position), which only lowers the noise). No value function's opinion about MOVES enters: the
only evidence is which sampled moves were in fact followed by wins.

The direction is the natural gradient (F + ridge)^-1 g, F = the covariance of x (so a step has a known
root-mean-square size in logits). How much of it is real is measured on games it was not estimated from
(folds of whole games): the held-out first-order gain of a step of fixed size. A planted effect of known
size (--plant) shows what the same data could have detected.

  python -m selfplay_actor_critic.pg_fit --run selfplay_actor_critic/runs/raw_g5_t05_a --side sh \
      --critic-dir selfplay_actor_critic/seeds/full_critic_r4 --output selfplay_actor_critic/seeds/pg_sh_a

--output writes, per step size, a policy with the tilt added to its last layer (<output>/rms_<size>/<side>_seed.pt,
playable with offline_eval --policy <output>/rms_<size>).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
from .report_metrics import correlation, explained_fraction, metric_text

ROOT = Path(__file__).resolve().parents[1]
from .event_trace import credits_for_run  # noqa: E402
from . import kind_context  # noqa: E402
from .contracts import ContractError, canonical_json
from .durability import (behavior_content, digest_json, episode_index, prediction_receipt,
                         publish_generation, require_terminal_cohort, validate_episode)
from .runtime_io import _atomic_write
from .fit_admission import cross_validation_folds, cross_validation_groups, require_game_partitions


def _bounded_map(pool, fn, items, chunk):
    """pool.map in chunks: Executor.map submits EVERY item at once and holds each result (a read game with both
    views, ~1 MB+) until it is consumed, so a slow consumer (the critic) lets the whole run pile up in memory
    (45 GB on big_g5_g3_a, 10-12 02:55). This keeps at most one chunk of results alive."""
    for start in range(0, len(items), chunk):
        yield from pool.map(fn, items[start:start + chunk])


def _read(path, side):
    with open(path, encoding="utf-8") as stream:
        meta = json.load(stream)
    end = meta.get("end")
    if not end or end.get("status") != "terminated":
        return None
    rows = meta["rows"]
    own = np.array([r["side"] == side for r in rows], bool)
    with np.load(os.path.join(os.path.dirname(path), "arrays.npz")) as d:
        if "centred" not in d.files:
            raise ContractError("Scheduled training episode lacks gradient evidence")
        out = dict(centred=d["centred"][own], kind_played=d["kind_played"][own], kind_mass=d["kind_mass"][own],
                   layout=[int(w) for w in d["centred_layout"]] if "centred_layout" in d.files else [int(d["centred"].shape[1])],
                   kind_names=[str(k) for k in d["kind_names"]], played_logp=d["played_logp"][own], top=d["top"][own],
                   observations=d["observations"][own])
        # the other side's view of the same decisions (valuedata stores it as "watch"): a both-views critic reads it
        other = "observations_other" if "observations_other" in d.files else ("watch" if "watch" in d.files else None)
        if other is not None and len(d[other]) == len(own):
            out["observations_other"] = d[other][own]
    out.update(result=1. if end["winner"] == side else -1., seed=int(meta["seed"]),
               turn=np.array([r["turn"] for r in rows], np.int16)[own], policy=meta.get("policies", {}).get(side, "seed"),
               temperature=float(meta["temperature"]), behavior=canonical_json(meta["behavior_content"][side]))
    return out


def _training_files(run, side, max_games):
    """One admitted episode order for gradients and their context features."""
    require_terminal_cohort(run)
    index = episode_index(run, require_identity=True)
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    identities = set()
    for _, meta in index.values():
        validate_episode(meta, manifest)
        identities.add((canonical_json(meta["behavior_content"][side]), float(meta["temperature"])))
    if len(identities) != 1:
        raise ContractError("Run mixes focal behavior identities or temperatures")
    files = [str(index[seed][0].resolve()) for seed in sorted(index)]
    if max_games:
        files = files[:max_games]
    return files


def load_run(run, side, critic_dir, *, device="cuda", max_games=0, threads=6):
    """Flat per-decision arrays, cached by exact episode and critic content."""
    run = Path(run).resolve()
    files = _training_files(run, side, max_games)
    receipt = prediction_receipt(run, side, critic_dir, files)
    receipt_text = canonical_json(receipt)
    cache = Path(run) / f"pg_{side}_{digest_json(receipt)}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            if str(d["receipt"]) != receipt_text:
                raise ContractError("Prediction cache receipt does not match its inputs")
            return {k: d[k] for k in d.files}
    values = None
    if critic_dir:
        from .generation import require_generation
        require_generation("pg_fit.load_run")
        os.environ.setdefault("WOTR_NATIVE", "0")
        for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
            if _sys_path_entry not in sys.path:
                sys.path.insert(0, _sys_path_entry)
        cwd = os.getcwd()
        import ai
        os.chdir(cwd)
        import torch
        from .full_state_critic import is_both_views_dir, load_full_state_critic
        from .policy import load_critic
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if is_both_views_dir(critic_dir):
            # A both-views (centralised) critic, full_state_critic.py: V from the mover's view AND the other side's.
            both = load_full_state_critic(critic_dir, device, ai=ai)

            def values(part):
                if "observations_other" not in part:
                    raise SystemExit(f"{run}: the games do not store the other side's view; a both-views critic needs it")
                return both.values(part["observations"], part["observations_other"], side, batch=8192)
        else:
            critic = load_critic(ai, Path(critic_dir) / f"{side}_critic.pt", side, device=device)

            @torch.inference_mode()
            def values(part):
                x = part["observations"]
                parts = []
                for start in range(0, len(x), 32768):
                    chunk = torch.from_numpy(x[start:start + 32768].astype(np.float32)).to(device)
                    parts.append(critic.value_head(critic.forward_backbone(chunk)[0]).reshape(-1).double().cpu().numpy())
                return np.concatenate(parts)
    cols = {key: [] for key in ("game", "centred", "kind", "mass", "played_logp", "top", "turn", "value")}
    results, seeds, kind_index, policy, temperature, layout, behavior = [], [], {}, None, None, None, None
    with ThreadPoolExecutor(threads) as pool:
        for part in _bounded_map(pool, lambda p: _read(p, side), files, 4 * max(1, threads)):
            if part is None:
                continue
            if behavior is not None and (behavior != part["behavior"] or temperature != part["temperature"]):
                raise ContractError("Games mix focal behavior identities or temperatures")
            policy, temperature, behavior = part["policy"], part["temperature"], part["behavior"]
            if layout is not None and layout != part["layout"]:
                raise SystemExit(f"{run}: games with different stored features {layout} / {part['layout']}")
            layout = part["layout"]
            local = np.array([kind_index.setdefault(k, len(kind_index)) for k in part["kind_names"]], np.int64)
            n = len(part["top"])
            cols["game"].append(np.full(n, len(results), np.int32))
            cols["centred"].append(part["centred"].astype(np.float32))
            cols["kind"].append(local[part["kind_played"].astype(np.int64)])
            cols["mass"].append((local, part["kind_mass"]))
            cols["played_logp"].append(part["played_logp"])
            cols["top"].append(part["top"])
            cols["turn"].append(part["turn"])
            cols["value"].append(values(part).astype(np.float32) if values and n else np.zeros(n, np.float32))
            results.append(part["result"])
            seeds.append(part["seed"])
            if len(results) % 2000 == 0:
                print(f"  read {len(results)} games", flush=True)
    mass = np.zeros((sum(len(m) for _, m in cols["mass"]), len(kind_index)), np.float32)
    row = 0
    for local, part in cols["mass"]:
        mass[row:row + len(part), local] = part
        row += len(part)
    out = {key: np.concatenate(value) for key, value in cols.items() if key != "mass"}
    out.update(mass=mass, result=np.asarray(results, np.float64), seed=np.asarray(seeds, np.int64), layout=np.asarray(layout, np.int64),
               kinds=np.array(sorted(kind_index, key=kind_index.get)), policy=np.array(str(policy)), temperature=np.float64(temperature),
               behavior=np.array(behavior), receipt=np.array(receipt_text))
    # A concurrent replacement must not label old predictions with a new critic/input digest.
    if prediction_receipt(run, side, critic_dir, files) != receipt:
        raise ContractError("Prediction inputs changed during cache construction")
    _atomic_write(cache, lambda stream: np.savez(stream, **out))
    return out


def _read_observations(path, side):
    """(seed, the side's own observations) of one stored game, with load_run's acceptance rule."""
    with open(path, encoding="utf-8") as stream:
        meta = json.load(stream)
    end = meta.get("end")
    if not end or end.get("status") != "terminated":
        raise ContractError("Scheduled context episode is not terminal")
    own = np.array([r["side"] == side for r in meta["rows"]], bool)
    with np.load(os.path.join(os.path.dirname(path), "arrays.npz")) as d:
        if "centred" not in d.files:
            raise ContractError("Scheduled context episode lacks gradient evidence")
        return int(meta["seed"]), d["observations"][own]


def context_for_run(run, side, *, max_games=0, threads=6):
    """kind_context levels of the side's decisions of the run (its own view), in load_run's game order:
    (levels [decisions, factors] int8, seeds [games], decisions per game). Cached per (side, spec, games)."""
    run = Path(run).resolve()
    files = _training_files(run, side, max_games)
    receipt = dict(inputs=prediction_receipt(run, side, None, files),
                   context_spec=kind_context.SPEC)
    receipt_text = canonical_json(receipt)
    cache = run / f"ctx_{side}_{digest_json(receipt)}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            if str(d['receipt']) != receipt_text:
                raise ContractError("Context cache receipt does not match its inputs")
            return d["levels"], d["seeds"], d["counts"]
    levels, seeds, counts = [], [], []
    # the engine import behind the settlement tables chdirs to src/ for a moment: do it BEFORE the reader threads
    # open relative paths (the 20:40 build lost 29,870 of 30,000 games to FileNotFoundError that way), and read by
    # absolute path anyway
    kind_context.tables(side)
    files = [os.path.abspath(f) for f in files]
    with ThreadPoolExecutor(threads) as pool:
        for part in _bounded_map(pool, lambda f: _read_observations(f, side), files, 4 * max(1, threads)):
            seed, observations = part
            levels.append(kind_context.levels(observations, side))
            seeds.append(seed)
            counts.append(len(observations))
            if len(seeds) % 2000 == 0:
                print(f"  context of {len(seeds)} games", flush=True)
    out = dict(levels=np.concatenate(levels) if levels else np.zeros((0, len(kind_context.FACTORS)), np.int8),
               seeds=np.asarray(seeds, np.int64), counts=np.asarray(counts, np.int64),
               receipt=np.array(receipt_text))
    if prediction_receipt(run, side, None, files) != receipt['inputs']:
        raise ContractError("Context inputs changed during cache construction")
    _atomic_write(cache, lambda stream: np.savez(stream, **out))
    return out["levels"], out["seeds"], out["counts"]


def attach_context(data, run, side, max_games=0):
    """data["ctx"] = the context levels of every decision (kind_context.py), aligned with the loaded games."""
    levels, seeds, counts = context_for_run(run, side, max_games=max_games)
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    if len(seeds) != games or not np.array_equal(seeds, data["seed"]) or not np.array_equal(counts, np.diff(bounds)):
        raise SystemExit(f"{run}: the context cache does not align with the loaded games (delete ctx_{side}_*.npz to rebuild)")
    data = dict(data)
    data["ctx"] = levels
    return data


BANKS = ("last", "delta")            # stored order: the action scorer's last hidden layer, the policy_delta head's


def bank(data, name):
    """Columns of one stored feature bank (games before 2026-10-04 19:40 hold "last" only)."""
    layout = [int(w) for w in data["layout"]] if "layout" in data else [data["centred"].shape[1]]
    i = BANKS.index(name)
    if i >= len(layout):
        raise SystemExit(f"These games do not store the '{name}' activations")
    start = sum(layout[:i])
    return slice(start, start + layout[i])


def design(data, features):
    """Centred regressors per decision: the activations feeding the final layer of the action scorer ("last")
    and of the policy_delta head ("delta"), and/or a one-hot of the kind of move (played minus the policy's
    probability of each kind: the regressor of a per-kind logit offset). Always in that order."""
    parts = []
    for name in BANKS:
        if name in features:
            parts.append(data["centred"][:, bank(data, name)].astype(np.float64))
    if "kind" in features or "kindctx" in features:
        offsets = -data["mass"].astype(np.float64)
        offsets[np.arange(len(offsets)), data["kind"]] += 1.
        offsets = offsets[:, kind_mask(data)]
    if "kind" in features:
        parts.append(offsets)
    if "kindctx" in features:
        # the context-conditioned offsets (kind_context.py): per kept kind k and context column j the regressor
        # rows_j(context) x (1[played kind = k] - P(kind k)); columns kind-major (k * WIDTH + j); float32 (300
        # columns over millions of decisions) and the whole design then float32
        if "ctx" not in data:
            raise SystemExit("--features kindctx needs the games' context (attach_context)")
        rows = kind_context.rows(data["ctx"])
        width = kind_context.WIDTH
        offsets32 = offsets.astype(np.float32)
        block = np.empty((len(offsets32), offsets32.shape[1] * width), np.float32)
        for j in range(width):
            block[:, j::width] = offsets32 * rows[:, j:j + 1]
        parts = [part.astype(np.float32) for part in parts] + [block]
        return np.concatenate(parts, axis=1)
    return np.concatenate(parts, axis=1)


KIND_MIN_DECISIONS = 2000


def kind_mask(data):
    """Which kinds get a per-kind regressor (and a written bias): those played at least KIND_MIN_DECISIONS times."""
    kinds = data["kind"]
    if data.get("_feature_fit_games") is not None:
        kinds = kinds[np.asarray(data["_feature_fit_games"], bool)[data["game"]]]
    return np.bincount(kinds, minlength=data["mass"].shape[1]) >= KIND_MIN_DECISIONS


def kept_kinds(data):
    return [str(k) for k, ok in zip(data["kinds"], kind_mask(data)) if ok]


def feature_slices(data, features):
    """{feature name: slice of the design columns} in design() order."""
    out, start = {}, 0
    for name in BANKS:
        if name in features:
            width = bank(data, name).stop - bank(data, name).start
            out[name] = slice(start, start + width)
            start += width
    if "kind" in features:
        width = int(kind_mask(data).sum())
        out["kind"] = slice(start, start + width)
        start += width
    if "kindctx" in features:
        width = int(kind_mask(data).sum()) * kind_context.WIDTH
        out["kindctx"] = slice(start, start + width)
    return out


def advantages(data, lam, fit_games=None):
    """What multiplies x at each decision, in result units.

    V = the critic's value of the position mapped linearly onto the result (least squares). Over the side's own
    decisions k = d, d+1, ... of the game, with V[after the last] = the result:
        sum over k >= d of lam^(k-d) * (V[k+1] - V[k])
    lam = 1 telescopes to result - V[d]: unbiased whatever V is, and noisy (everything that happens later in
    the game is in it). lam < 1 trusts V's judgement of the positions reached later (never its opinion of
    moves): far less noise, at the price of V's errors.
    fit_games: boolean mask over games on which the two-parameter calibration of V is fitted (None = all; the
    held-out protocol passes the training folds so the evaluation fold never shapes its own baseline)."""
    from scipy.signal import lfilter
    z = data["result"][data["game"]]
    A = np.column_stack((np.ones(len(z)), data["value"].astype(np.float64)))
    rows = slice(None) if fit_games is None else np.asarray(fit_games, bool)[data["game"]]
    v = A @ np.linalg.lstsq(A[rows], z[rows], rcond=None)[0]
    if lam >= 1.:
        return z - v
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    out = np.zeros(len(z))
    for g in range(games):
        a, b = bounds[g], bounds[g + 1]
        if b > a:
            delta = np.append(v[a + 1:b], data["result"][g]) - v[a:b]
            out[a:b] = lfilter([1.], [1., -lam], delta[::-1])[::-1]
    return out


def residual(data, fit_games=None):
    """Result minus its linear prediction from V(position)."""
    return advantages(data, 1., fit_games)


def audit_split(seeds, fraction, audit_seed):
    """A fixed set of whole games held out from EVERYTHING (direction, calibration, luck coefficients, kind
    baselines): boolean mask over games, reproducible from the seeds alone (the same rule as luck.audit_mask)."""
    from .luck import audit_mask
    return audit_mask(np.asarray(seeds, np.int64), fraction, audit_seed)


def nuisance_fit_mask(data, fit_games):
    """Games a nuisance quantity may be fitted on: all but the audit games."""
    return np.ones(len(data["result"]), bool) if fit_games is None else np.asarray(fit_games, bool)


def load_credit_cache(path, *, run, side):
    """Read learned credits only for their bound side, episodes and producer inputs."""
    from .return_credit_cache import load_admitted
    return load_admitted(path, run=run, side=side)


def attach_traced(data, run, side, model, mode, delay_turns=0, lam=1., cache_dir=""):
    """Add the traced event credit (event_trace.py) of every decision as data["traced"], aligned by game seed;
    with cache_dir, the credit comes from <cache_dir>/<run name>.npz instead (a learned decomposition)."""
    if cache_dir:
        credits = load_credit_cache(Path(cache_dir) / f"{Path(run).name}.npz", run=run, side=side)
    else:
        credits = credits_for_run(run, side, model, mode=mode, delay_turns=delay_turns, lam=lam)
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    traced = np.zeros(len(data["game"]))
    for g in range(games):
        part = credits.get(int(data["seed"][g]))
        if part is None or len(part) != bounds[g + 1] - bounds[g]:
            raise SystemExit(f"{run}: game {int(data['seed'][g])} has no traced credit of the right length")
        traced[bounds[g]:bounds[g + 1]] = part
    data = dict(data)
    data["traced"] = traced
    return data


def luck_cache_matches(runs, stem, fraction, audit_seed):
    """Every run's luck cache must have been built with this audit set (its coefficients never saw the audit games)."""
    for run in runs:
        with np.load(Path(run) / f"luck_{stem}.npz") as d:
            if "audit_fraction" not in d.files or float(d["audit_fraction"]) != float(fraction) or int(d["audit_seed"]) != int(audit_seed):
                return False
    return True


def attach_luck(data, run, side, stem):
    """Add the chance control variate (luck.py cache luck_<stem>.npz of the run) per own decision as data["luck"],
    and what refitting its coefficients on a training mask needs (luck_refit_correction): the raw correction of
    each event kind per decision (data["luck_kind"]), each game's total luck per kind (data["luck_totals"]) and
    result in Shadow units (data["luck_z"]), columns in data["luck_kind_names"] order."""
    from .luck import load_corrections, load_kind_terms
    corrections = load_corrections(run, stem, side)
    names, per_kind, totals, z = load_kind_terms(run, stem, side)
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    luck = np.zeros(len(data["game"]))
    luck_kind = np.zeros((len(data["game"]), len(names)))
    luck_totals = np.zeros((games, len(names)))
    luck_z = np.zeros(games)
    for g in range(games):
        seed = int(data["seed"][g])
        part = corrections.get(seed)
        if part is None or len(part) != bounds[g + 1] - bounds[g] or len(per_kind[seed]) != len(part):
            raise SystemExit(f"{run}: game {seed} has no luck correction of the right length")
        luck[bounds[g]:bounds[g + 1]] = part
        luck_kind[bounds[g]:bounds[g + 1]] = per_kind[seed]
        luck_totals[g], luck_z[g] = totals[seed], z[seed]
    data = dict(data)
    data.update(luck=luck, luck_kind=luck_kind, luck_totals=luck_totals, luck_z=luck_z, luck_kind_names=np.array(names))
    return data


def luck_refit_correction(data, train):
    """The per-decision luck correction with its control-variate coefficients refitted on the `train` games only
    (luck.coefficient_rows, the cache's own rule with `train` in place of the non-audit games): a training game is
    scaled by its seed fold's coefficients fitted on the other training games, any other game by their mean. The
    cached data["luck"] used coefficients fitted on every non-audit game, a CV evaluation fold's outcomes included,
    so the held-out refits must not reuse it (S2-04). On the games the cache was fitted on it reproduces data["luck"]."""
    from .luck import coefficient_rows
    if "luck_kind" not in data:
        raise ContractError("The luck correction cannot be refitted: attach_luck supplied no per-kind luck terms")
    _, rows = coefficient_rows(data["luck_totals"], data["luck_z"], data["seed"], np.asarray(train, bool))
    return (data["luck_kind"] * rows[data["game"]]).sum(1)


def traced_advantages(data, baseline="kind", fit_games=None):
    """The traced credit minus the behaviour-weighted mean of the fitted kind means.

    This baseline depends on the position's kind probabilities, not the sampled kind, so it preserves
    both within-kind and between-kind policy gradients.
    fit_games: games the kind means are taken from (None = all; the audit mode passes the non-audit games)."""
    r = data["traced"].astype(np.float64).copy()
    if baseline == "kind":
        rows = slice(None) if fit_games is None else np.asarray(fit_games, bool)[data["game"]]
        means = np.bincount(data["kind"][rows], r[rows], len(data["kinds"])) / np.maximum(np.bincount(data["kind"][rows], minlength=len(data["kinds"])), 1)
        r -= data["mass"] @ means
    return r


def merge(parts):
    """Several runs of the same side and policy as one data set."""
    seeds = [int(seed) for p in parts for seed in p["seed"]]
    if len(set(seeds)) != len(seeds):
        raise ContractError("Duplicate episode seeds cannot enter policy-gradient folds")
    if any(str(p.get("behavior", p["policy"])) != str(parts[0].get("behavior", parts[0]["policy"])) or
           float(p["temperature"]) != float(parts[0]["temperature"]) for p in parts):
        raise ContractError("The runs were not played by the same focal behavior at the same temperature")
    if len(parts) == 1:
        return parts[0]
    kinds = sorted({str(k) for p in parts for k in p["kinds"]})
    index = {k: i for i, k in enumerate(kinds)}
    if len({"luck_kind" in p for p in parts}) > 1:
        raise ContractError("Some runs carry per-kind luck terms and others do not")
    luck_kinds = sorted({str(k) for p in parts for k in p.get("luck_kind_names", [])})
    luck_index = {k: i for i, k in enumerate(luck_kinds)}
    out, offset, mass = {}, 0, []
    for p in parts:
        local = np.array([index[str(k)] for k in p["kinds"]], np.int64)
        m = np.zeros((len(p["kind"]), len(kinds)), np.float32)
        m[:, local] = p["mass"]
        mass.append(m)
        if "luck_kind" in p:
            # each run's per-kind luck terms in the union of the runs' event kinds (a kind a run never saw is zero)
            columns = np.array([luck_index[str(k)] for k in p["luck_kind_names"]], np.int64)
            for key in ("luck_kind", "luck_totals"):
                block = np.zeros((len(p[key]), len(luck_kinds)))
                block[:, columns] = p[key]
                out.setdefault(key, []).append(block)
            out.setdefault("luck_z", []).append(p["luck_z"])
        if [int(w) for w in p.get("layout", [p["centred"].shape[1]])] != [int(w) for w in parts[0].get("layout", [parts[0]["centred"].shape[1]])]:
            raise SystemExit("The runs do not store the same features")
        for key in ("centred", "played_logp", "top", "turn", "value", "result", "seed", "traced", "luck", "ctx"):
            if key in p:
                out.setdefault(key, []).append(p[key])
        out.setdefault("kind", []).append(local[p["kind"]])
        out.setdefault("game", []).append(p["game"].astype(np.int64) + offset)
        offset += len(p["result"])
    out = {key: np.concatenate(value) for key, value in out.items()}
    out.update(mass=np.concatenate(mass), kinds=np.array(kinds), policy=parts[0]["policy"], temperature=parts[0]["temperature"])
    if "luck_kind" in parts[0]:
        out["luck_kind_names"] = np.array(luck_kinds)
    if "behavior" in parts[0]:
        out["behavior"] = parts[0]["behavior"]
        out["source_receipts"] = np.array([str(p["receipt"]) for p in parts])
    if "layout" in parts[0]:
        out["layout"] = parts[0]["layout"]
    return out


class GroupedFisher:
    """The per-game Fisher sums x x^T kept per GROUP of games (a fold x audit cell) instead of per game: with 300+
    regressors the [games, P, P] array would take tens of GB. F[mask].sum(0) and F.sum(0) behave as on the full
    array for every mask that is constant within each group (the fold draws and the audit set the groups were
    built from; anything else raises)."""

    def __init__(self, sums, group):
        self.sums, self.group = sums, np.asarray(group, np.int64)
        self.shape = (len(self.group), sums.shape[1], sums.shape[2])

    def __len__(self):
        return len(self.group)

    def __getitem__(self, mask):
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self.group),):
            raise TypeError("GroupedFisher supports boolean game masks only")
        inside = np.bincount(self.group, mask.astype(np.float64), len(self.sums))
        size = np.bincount(self.group, minlength=len(self.sums)).astype(np.float64)
        if not np.all((inside == 0) | (inside == size)):
            raise ValueError("the mask splits a Fisher group: it is not a union of the fold x audit cells the sums were built from")
        return GroupedFisher(self.sums[inside > 0], np.zeros(int(mask.sum()), np.int64))

    def sum(self, axis=0):
        if axis != 0:
            raise ValueError("GroupedFisher sums over games only")
        return self.sums.sum(0)


def game_gradient(data, X, r, temperature):
    """Per game: sum over decisions of x * residual / T [games, P] (the G of game_sums, without the Fisher sums)."""
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    G = np.zeros((games, X.shape[1]))
    for g in range(games):
        a, b = bounds[g], bounds[g + 1]
        G[g] = X[a:b].T.astype(np.float64) @ (r[a:b] / temperature)
    return G


def game_sums(data, X, r, temperature, groups=None):
    """Per game: sum over decisions of x * residual / T [games, P] and of x x^T [games, P, P]; with `groups` (an
    integer per game) the Fisher sums are kept per group as a GroupedFisher instead (memory)."""
    games = len(data["result"])
    bounds = np.searchsorted(data["game"], np.arange(games + 1))
    G = np.zeros((games, X.shape[1]))
    if groups is not None:
        groups = np.asarray(groups, np.int64)
        sums = np.zeros((int(groups.max()) + 1, X.shape[1], X.shape[1]))
        for g in range(games):
            a, b = bounds[g], bounds[g + 1]
            Xg = X[a:b].astype(np.float64)
            G[g] = Xg.T @ (r[a:b] / temperature)
            sums[groups[g]] += Xg.T @ Xg
        return G, GroupedFisher(sums, groups), np.diff(bounds)
    F = np.zeros((games, X.shape[1], X.shape[1]))
    weighted = X * r[:, None] / temperature
    for g in range(games):
        a, b = bounds[g], bounds[g + 1]
        G[g] = weighted[a:b].sum(0)
        F[g] = X[a:b].T @ X[a:b]
    return G, F, np.diff(bounds)


def direction(G, F, counts, ridge, rms, anchor=None):
    """Natural-gradient step of root-mean-square size `rms` (logits at the played move, temperature 1).

    anchor = (delta, weight): delta is the policy's CUMULATIVE drift from an anchor policy (current weights minus
    the anchor's, in the fitted features' coordinates) and the step is pulled back toward the anchor in the
    Fisher metric: it minimises -g . theta + 1/2 theta F theta + weight/2 (theta + delta) F (theta + delta)
    (plus the ridge), i.e. (1 + weight) (F + ridge) theta = g - weight F delta, before the rescaling to `rms`.
    weight 0 is the plain step; large weights make the step mostly a return toward the anchor."""
    # A valid whole-game fold can have no variation in the selected
    # features, even when another fold does. Its evidence supplies no step.
    width = G.shape[1]
    if width == 0 or counts.sum() == 0:
        return np.zeros(width, np.float64)
    fisher = F.sum(0) / counts.sum()
    trace = float(np.trace(fisher))
    if trace == 0:
        return np.zeros(width, np.float64)
    gradient = G.mean(0)
    scale = 1.
    if anchor is not None:
        delta, weight = anchor
        gradient = gradient - weight * (fisher @ np.asarray(delta, np.float64))
        scale = 1. + weight
    system = scale * (fisher + ridge * trace / width * np.eye(width))
    # With no ridge, redundant coordinates leave the same objective on
    # the Fisher support. Its minimum-norm solution adds no null-space step.
    theta = (np.linalg.lstsq(system, gradient, rcond=None)[0] if ridge == 0
             else np.linalg.solve(system, gradient))
    size = float(np.sqrt(theta @ fisher @ theta))
    return theta * (rms / size) if size > 0 else theta


def drift(F, counts, delta):
    """Root-mean-square logit distance of a cumulative drift `delta` from the anchor, in the Fisher metric."""
    fisher = F.sum(0) / counts.sum()
    delta = np.asarray(delta, np.float64)
    return float(np.sqrt(max(delta @ fisher @ delta, 0.)))


def curvature(S, V, z):
    """Second derivative of the expected result along a step, from the games themselves.

    For the policy tilted by eps along the step, d2/deps2 E[z] = E[z (S^2 - V)] with S the sum over the game
    decisions of the score of the step (the logit change at the played move minus its policy mean, in sampling
    logits) and V the sum over decisions of the policy variance of that logit change. S is a martingale sum, so
    the mean of S^2 - V is 0 and a constant baseline is free. Returns (c, standard error) in result units per
    step^2: the quadratic model of the gain of eps steps is eps * a + eps^2 * c / 2 with a the first-order gain.
    A negative c is the interaction of the step across the game (saturation: the same thing asked for at every
    decision does not add up)."""
    term = (np.asarray(z, np.float64) - np.mean(z)) * (np.asarray(S, np.float64) ** 2 - np.asarray(V, np.float64))
    return float(term.mean()), float(term.std() / np.sqrt(len(term)))


def quadratic_step(a, c, c_error, *, lo=.25, hi=4.):
    """The step multiple (in units of the fitted step) that maximises eps * a + eps^2 * c / 2, between lo and hi;
    None when the curvature is not resolved (c >= 0 or within its error), so the default step stands."""
    if a <= 0 or c >= 0 or abs(c) < c_error:
        return None
    return float(min(hi, max(lo, -a / c)))


def bias_share(plain, G):
    """How much of the difference between the results gradient (lam 1) and a shorter-horizon one is real.

    Per game the two sums differ by what the value network was trusted for. If that trust costs nothing the
    mean difference is noise: chi2 = mean' cov^-1 mean equals its degrees of freedom. Returns (share, chi2,
    dof), share = max(0, 1 - dof / chi2): the James-Stein weight on the difference, 0 when it is all noise
    and toward 1 as a real bias shows. G + share * (plain - G) is the blended gradient: the short horizon's
    precision, corrected by the results as far as the results can be heard."""
    R = plain - G
    live = R.std(0) > 1e-12
    if not live.any():
        return 0., 0., 0
    R = R[:, live]
    mean = R.mean(0)
    covariance = np.atleast_2d(np.cov(R.T)) / len(R)
    solved, _, rank, _ = np.linalg.lstsq(covariance, mean, rcond=None)
    chi2 = max(0., float(mean @ solved))
    # A single live coordinate is scalar covariance; redundant coordinates
    # carry only their covariance rank's degrees of freedom. Equal means
    # supply no evidence for a bias correction.
    return (max(0., 1. - rank / chi2) if chi2 > 0 else 0.), chi2, int(rank)


def _held_out_steps(G, F, counts, ridge, rms, folds, seed, anchor, refit, eligible):
    """Shared fold admission and nuisance refitting for every CV report."""
    fold, live = cross_validation_folds(len(G), folds, seed, eligible)
    for f in range(folds):
        inside = (fold == f) & live
        train = (fold != f) & live
        if not inside.any():
            continue
        G_f, measurement = (G, None) if refit is None else refit(train)
        theta = direction(G_f[train], F[train], counts[train], ridge, rms, anchor)
        yield inside, theta, measurement


def held_out(G, F, counts, ridge, rms, folds, seed=11, measure=None, anchor=None, refit=None, eligible=None):
    """(gain, standard error) in result units of a step estimated on the other folds, evaluated on each fold's
    games. `measure`: per-game sums to evaluate with (default the same G; pass the lam = 1 sums to measure a
    step found with a smaller lam by results alone). `refit(train_mask) -> (G, measure)`: recompute the per-game
    sums with every nuisance quantity (the calibration of V, the luck coefficients) fitted on the training folds
    only; without it the sums passed in are used as they are. `eligible`: games the folds are drawn from (the
    audit games are left out of the cross-validation altogether).

    CAUTION (GPT-6 review 2026-10-11): the fold directions share training games, so the fold-wise scores are not
    independent and this standard error is optimistic (a null simulation of 500 games and 8 coefficients found
    the empirical spread 1.46x the reported SE). The audit mode (audit_gain) gives the honest number."""
    measure = G if measure is None else measure
    per_game = np.zeros(len(G))
    live = np.zeros(len(G), bool)
    for inside, theta, measurement in _held_out_steps(G, F, counts, ridge, rms, folds, seed, anchor, refit, eligible):
        M_f = measure if measurement is None else measurement
        per_game[inside] = M_f[inside] @ theta
        live |= inside
    scores = per_game[live]
    return float(scores.mean()), float(scores.std() / np.sqrt(len(scores)))


def audit_gain(G, F, counts, ridge, rms, audit, anchor=None, refit=None):
    """ONE direction fitted on the non-audit games (with every nuisance quantity fitted there too, via refit),
    evaluated on the untouched audit games: (gain, standard error, number of audit games). The direction and the
    baseline are frozen with respect to the audit games, so their per-game scores are independent and the plain
    standard error is valid."""
    audit = np.asarray(audit, bool)
    train = ~audit
    require_game_partitions(np.flatnonzero(train), np.flatnonzero(audit), context="Audit")
    G_t, M_t = (G, G) if refit is None else refit(train)
    theta = direction(G_t[train], F[train], counts[train], ridge, rms, anchor)
    scores = M_t[audit] @ theta
    return float(scores.mean()), float(scores.std() / np.sqrt(max(len(scores), 1))), int(audit.sum())


def held_out_curvature(G, F, counts, ridge, rms, folds, X, data, temperature, kind_slice=None, seed=11, anchor=None,
                       refit=None, eligible=None):
    """The curvature along the fitted step, each game with the step fitted on the other folds.

    Returns (c_interaction, error, c_kind_within, error). The cross-decision part is estimated for the whole step
    from the played moves alone (the per-decision policy variance replaced by the played move's squared score,
    which leaves exactly the cross terms). The within-decision part is exact for the per-kind block only (its
    policy variance follows from the stored kind mass) and is None when no kind block is fitted."""
    games = len(G)
    fold, live = cross_validation_folds(games, folds, seed, eligible)
    z = data["result"]
    S = np.zeros(games); V_sample = np.zeros(games)
    S_kind = np.zeros(games); V_kind = np.zeros(games)
    mass = data["mass"][:, kind_mask(data)].astype(np.float64) if kind_slice is not None else None
    for f in range(folds):
        inside = (fold == f) & live
        train = (fold != f) & live
        if not inside.any():
            continue
        G_f = G if refit is None else refit(train)[0]
        theta = direction(G_f[train], F[train], counts[train], ridge, rms, anchor)
        s = X @ theta / temperature
        S[inside] = np.bincount(data["game"], s, games)[inside]
        V_sample[inside] = np.bincount(data["game"], s ** 2, games)[inside]
        if kind_slice is not None:
            tk = theta[kind_slice]
            s_kind = X[:, kind_slice] @ tk / temperature
            var_kind = (mass @ (tk ** 2) - (mass @ tk) ** 2) / temperature ** 2
            S_kind[inside] = np.bincount(data["game"], s_kind ** 2, games)[inside]
            V_kind[inside] = np.bincount(data["game"], var_kind, games)[inside]
    c_int, e_int = curvature(S[live], V_sample[live], z[live])
    if kind_slice is None:
        return c_int, e_int, None, None
    term = (z[live] - z[live].mean()) * (S_kind[live] - V_kind[live])
    return c_int, e_int, float(term.mean()), float(term.std() / np.sqrt(live.sum()))


def held_out_by_kind(G, F, counts, ridge, rms, folds, X, r, data, temperature, seed=11, refit=None, eligible=None):
    """The held-out gain (by the per-decision multipliers r) split by the kind of move: which kinds of decision
    the step from G helps or hurts. Returns {kind: (gain, standard error)} in result units per game."""
    # Here the refit's measurement is the row-level result residual, so the
    # same admitted folds can also attribute their scores to move kinds.
    per_decision = np.zeros(len(X))
    live = np.zeros(len(G), bool)
    for inside, theta, measurement in _held_out_steps(G, F, counts, ridge, rms, folds, seed, None, refit, eligible):
        rows = inside[data["game"]]
        residuals = r if measurement is None else measurement
        per_decision[rows] = (X[rows] @ theta) * residuals[rows] / temperature
        live |= inside
    out = {}
    games = len(G)
    for k, name in enumerate(data["kinds"]):
        rows = (data["kind"] == k) & live[data["game"]]
        if rows.sum() < 1000:
            continue
        per_game = np.bincount(data["game"][rows], per_decision[rows], games)
        out[str(name)] = (float(per_game[live].mean()), float(per_game[live].std() / np.sqrt(live.sum())), int(rows.sum()))
    return out


def print_by_kind(label, table):
    print(f"  {label}: held-out gain by results per kind of move (points per game)")
    for name, (gain, error, n) in sorted(table.items(), key=lambda kv: -abs(kv[1][0])):
        print(f"    {name:22s} {50 * gain:+6.2f} +- {50 * error:4.2f}   ({n} decisions)")


def _engine():
    """The encoder module, imported without leaving the working directory changed."""
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    from .generation import require_generation
    require_generation("pg_fit._engine")
    os.environ.setdefault("WOTR_NATIVE", "0")
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    return ai


def feature_weights(actor, data, features):
    """The policy's current weights in the fitted features' coordinates (design() order): the final layer of
    the scorer ("last") and of the policy_delta head ("delta"), and the per-kind bias of the kept kinds."""
    import torch
    parts = []
    with torch.no_grad():
        for name, head in (("last", actor.model.spatial_action_scorer), ("delta", actor.model.policy_delta)):
            if name in features:
                parts.append(head[-1].weight.detach().cpu().double().reshape(-1).numpy())
        if "kind" in features:
            bias = getattr(actor.model, "kind_bias", None)
            values = np.zeros(len(kept_kinds(data)))
            if bias is not None:
                for i, name in enumerate(kept_kinds(data)):
                    if name in bias.index:
                        values[i] = float(bias.bias[bias.index[name]])
            parts.append(values)
        if "kindctx" in features:
            ctx = getattr(actor.model, "kind_ctx_bias", None)
            values = np.zeros((len(kept_kinds(data)), kind_context.WIDTH))
            if ctx is not None:
                for i, name in enumerate(kept_kinds(data)):
                    if name in ctx.index:
                        values[i] = ctx.bias[ctx.index[name]].detach().cpu().double().numpy()
            parts.append(values.reshape(-1))
    return np.concatenate(parts)


def write_step(actor, theta, data, features):
    """Add a fitted step (design() coordinates) to the policy's weights in place."""
    import torch
    from .policy import install_kind_bias, install_kind_ctx_bias
    with torch.no_grad():
        start = 0
        for name, head in (("last", actor.model.spatial_action_scorer), ("delta", actor.model.policy_delta)):
            if name in features:
                final = head[-1]
                width = int(final.in_features)
                final.weight += torch.from_numpy(theta[start:start + width]).to(final.weight.dtype).reshape(1, -1)
                start += width
        if "kind" in features:
            names = kept_kinds(data)
            bias = install_kind_bias(actor.model, names)
            for name, value in zip(names, theta[start:start + len(names)]):
                bias.bias[bias.index[name]] += float(value)
            start += len(names)
        if "kindctx" in features:
            names = kept_kinds(data)
            ctx = install_kind_ctx_bias(actor.model, names, actor.identity.side)
            block = theta[start:start + len(names) * kind_context.WIDTH].reshape(len(names), kind_context.WIDTH)
            for name, row in zip(names, block):
                ctx.bias[ctx.index[name]] += torch.from_numpy(np.asarray(row)).to(ctx.bias.dtype)
            start += len(names) * kind_context.WIDTH
    if start != len(theta):
        raise SystemExit("The step does not match the policy's heads")


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, nargs="+", help="one or more valuedata.py --gradient runs (same policy for the side)")
    parser.add_argument("--lam", type=float, nargs="+", default=[1., .98, .95, .9],
                        help="1 = the result minus V (unbiased); smaller = trust V for the positions reached later (less noise)")
    parser.add_argument("--write", type=float, nargs=2, default=None, metavar=("LAM", "RIDGE"),
                        help="with --output: the setting to write (default: the best held-out gain measured by results)")
    parser.add_argument("--blend", action="store_true",
                        help="correct every lam < 1 gradient by the results gradient as far as their difference is not noise (bias_share)")
    parser.add_argument("--side", required=True, choices=("fp", "sh"))
    parser.add_argument("--critic-dir", default="", help="value network for the baseline (none = the mean result)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--features", nargs="+", default=["last"], choices=("last", "delta", "kind", "kindctx"))
    parser.add_argument("--ridge", type=float, nargs="+", default=[.01, .1, 1., 10.])
    parser.add_argument("--rms", type=float, default=.1, help="step size: root-mean-square logit change at the played move")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--audit-fraction", type=float, default=0.,
                        help="AUDIT: hold this share of whole games out of everything (direction, calibration, luck coefficients, kind baselines),"
                             " fit one direction on the rest and report its gain on the audit games with a plain standard error")
    parser.add_argument("--audit-seed", type=int, default=0, help="which games form the audit set (reproducible from the game seeds)")
    parser.add_argument("--plant", type=float, default=0.,
                        help="control: add an effect of this first-order gain (points, at the step size) along a random direction")
    parser.add_argument("--output", default="", help="write tilted policies <output>/rms_<size>/<side>_seed.pt (last-layer features only)")
    parser.add_argument("--steps", type=float, nargs="+", default=[.05, .1, .2])
    parser.add_argument("--anchor", default="",
                        help="seed folder (or 'seed' = the imitation policy) the step is pulled toward in the Fisher metric: the ridge then acts on "
                             "the cumulative drift from the anchor, not on the step (AlphaStar's KL anchor)")
    parser.add_argument("--anchor-weight", type=float, nargs="+", default=[1.],
                        help="weights of the anchor pull to report; the first one is used for the written step")
    parser.add_argument("--curvature", action="store_true",
                        help="estimate the second derivative along the chosen step from the same games and size the written step by the quadratic model")
    parser.add_argument("--traced", default="", metavar="MODEL.json",
                        help="also the gradient of the event credit traced by event_trace.py with this explicit value function")
    parser.add_argument("--traced-cache", default="", metavar="DIR",
                        help="instead of tracing: per-decision credits from <DIR>/<run name>.npz (return_decomp.py credits); sets --traced to the dir name")
    parser.add_argument("--trace-mode", default="full", choices=("full", "split", "return"))
    parser.add_argument("--trace-baseline", default="kind", choices=("kind", "none"))
    parser.add_argument("--trace-delay", type=int, default=0, help="credit only decisions at least this many turns before the event (hindsight only)")
    parser.add_argument("--trace-lam", type=float, default=1., help="return mode: discount of the value's later changes per turn elapsed")
    parser.add_argument("--write-traced", type=float, default=None, metavar="RIDGE", help="with --output: write the traced-credit step")
    parser.add_argument("--traced-mix", type=float, nargs="*", default=[],
                        help="also the gradients of result residual + MIX x traced credit, for these MIX values")
    parser.add_argument("--luck", default="", metavar="STEM",
                        help="chance control variate: subtract the luck corrections of <run>/luck_<STEM>.npz (luck.py) from the results advantage")
    parser.add_argument("--write-luck", type=float, default=None, metavar="RIDGE", help="with --output: write the luck-corrected results step")
    parser.add_argument("--by-kind", type=float, default=None, metavar="RIDGE",
                        help="split the held-out gain by results of the lam-1 and the traced directions by kind of move, at this ridge")
    parser.add_argument("--check-generation", default="", help="read-only integrity and exact invocation check; never fits")
    parser.add_argument("--request-receipt", default="", help="after a successful check, atomically save the expected request")
    return parser


def main():
    args = argument_parser().parse_args()
    if args.traced_cache:
        args.traced = args.traced or args.traced_cache
    from .fit_request import fitting_request
    request = fitting_request(args) if args.output or args.check_generation else None
    if args.check_generation:
        from .durability import validate_generation
        validate_generation(args.check_generation, args.side, expected_request=request)
        if args.request_receipt:
            from .runtime_io import write_json
            write_json(args.request_receipt, request)
        return
    parts = [load_run(run, args.side, args.critic_dir, device=args.device, max_games=args.max_games) for run in args.run]
    if args.traced:
        parts = [attach_traced(p, run, args.side, args.traced, args.trace_mode, args.trace_delay, args.trace_lam, args.traced_cache)
                 for p, run in zip(parts, args.run)]
    if args.luck:
        parts = [attach_luck(p, run, args.side, args.luck) for p, run in zip(parts, args.run)]
    if "kindctx" in args.features:
        parts = [attach_context(p, run, args.side, args.max_games) for p, run in zip(parts, args.run)]
    data = merge(parts)
    games = len(data["result"])
    audit = None
    if args.audit_fraction > 0:
        if args.audit_fraction >= 1:
            raise ContractError("Audit fraction must be less than one")
        audit = audit_split(data["seed"], args.audit_fraction, args.audit_seed)
        if not audit.any() or (~audit).sum() < 2:
            raise ContractError("Audit requires held-out games and at least two fitting games")
        if args.luck and not luck_cache_matches(args.run, args.luck, args.audit_fraction, args.audit_seed):
            raise SystemExit("the luck cache was not built with this audit set: rerun luck.py with the same --audit-fraction/--audit-seed")
    fit_games = None if audit is None else ~audit
    cross_validation_folds(games, args.folds, eligible=fit_games)
    fit_mask = nuisance_fit_mask(data, fit_games)
    # Feature eligibility must also be independent of the audit trajectories.
    data = dict(data, _feature_fit_games=fit_games)
    temperature = float(data["temperature"])
    X = design(data, args.features)
    slices = feature_slices(data, args.features)
    anchor = None
    if args.anchor:
        from .offline_eval import _load_actor
        ai = _engine()
        current = feature_weights(_load_actor(ai, str(data["policy"]), args.side), data, args.features)
        anchored = feature_weights(_load_actor(ai, args.anchor, args.side), data, args.features)
        anchor = (current - anchored, float(args.anchor_weight[0]))
    win = float((data["result"] > 0).mean())
    print(f"{' + '.join(Path(r).name for r in args.run)} {args.side}: {games} games ({100 * win:.1f}% won), {len(X)} decisions, policy {data['policy']}, "
          f"temperature {temperature:g}, {X.shape[1]} regressors ({'+'.join(args.features)})")
    if args.plant:
        # A known effect: every decision adds c * (v . x) to the result, v a random direction. The gradient is
        # then proportional to F v and the best step of a given size is along v itself; c is set so that this
        # best step gains `plant` points.
        rng = np.random.default_rng(5)
        v = rng.normal(size=X.shape[1])
        fisher = X.T @ X / len(X)
        step = v * args.rms / np.sqrt(v @ fisher @ v)
        per_decision = X @ v
        true_gain = (step @ (fisher @ v)) * len(X) / games / temperature      # of the step, per unit c
        c = (args.plant / 50.) / true_gain
        data = dict(data)
        data["result"] = data["result"] + c * np.bincount(data["game"], per_decision, games)
        print(f"  planted: a direction whose step of rms {args.rms:g} gains {args.plant:+.2f} points")
    if audit is not None:
        print(f"  AUDIT: {int(audit.sum())} of {games} games held out of every fit (audit seed {args.audit_seed}); "
              f"the calibration of V{', the luck coefficients' if args.luck else ''}{', the kind baselines' if args.traced else ''} saw only the other {int((~audit).sum())}")
    r = residual(data, fit_games)
    explained = explained_fraction(r.var(), data["result"][data["game"]].var())
    groups = None
    if X.shape[1] > 128:
        # Refine full-cohort and fitting-only fold masks: a tiny fitting
        # subset may need its own deterministic nonempty-fold repair.
        groups = cross_validation_groups(games, args.folds, eligible=fit_games)
    plain, F, counts = game_sums(data, X, r, temperature, groups)

    def refit_for(lam, extra=None):
        """refit(train_mask) -> (G, measure): the sums with the calibration of V fitted on train_mask only;
        extra(train_mask) -> per-decision term subtracted from the advantage (the luck correction, its
        coefficients refitted on train_mask: luck_refit_correction), or None."""
        def refit(train):
            r_t = residual(data, train)
            measure = game_gradient(data, X, r_t, temperature)
            if lam >= 1.:
                r_l = r_t
            else:
                r_l = advantages(data, lam, train)
            if extra is not None:
                r_l = r_l - extra(train)
            G_t = measure if (lam >= 1. and extra is None) else game_gradient(data, X, r_l, temperature)
            if lam < 1. and args.blend:
                share = bias_share(measure[train], G_t[train])[0]
                G_t = G_t + share * (measure - G_t)
            return G_t, measure
        return refit

    def refit_traced(train):
        return (game_gradient(data, X, traced_advantages(data, args.trace_baseline, train), temperature),
                game_gradient(data, X, residual(data, train), temperature))

    def chosen_refit(setting):
        if setting == "traced":
            return refit_traced
        if setting == "luck":
            return refit_for(1., extra=lambda train: luck_refit_correction(data, train))
        return refit_for(setting)

    def self_measure(refit):
        def fitted(train):
            gradient, _ = refit(train)
            return gradient, gradient
        return fitted

    print(f"  the baseline explains {metric_text(explained, '.0f', scale=100, suffix='%')} of the result's variance; steps of rms {args.rms:g}; gains in points of win probability")
    if anchor is not None:
        print(f"  anchor {args.anchor}: the policy has drifted {drift(F, counts, anchor[0]):.3f} rms logits from it (Fisher metric, in these features)")
    best, sums = None, {}
    for lam in args.lam:
        r_lam = r if lam >= 1. else advantages(data, lam, fit_games)
        G = plain if lam >= 1. else game_gradient(data, X, r_lam, temperature)
        noise = 50 * args.rms * np.sqrt((r_lam ** 2).mean() * counts.mean() / games) / temperature
        note = ""
        if lam < 1.:
            share, chi2, dof = bias_share(plain[fit_mask], G[fit_mask])
            note = f"; what it leaves out of the results gradient: chi2 {chi2:.0f} on {dof} (noise alone {dof} +- {np.sqrt(2 * dof):.0f})"
            if args.blend:
                G = G + share * (plain - G)
                note += f"; blended with the results gradient, weight {share:.2f}"
        sums[lam] = G
        print(f"  lam {lam:g}: noise of a random step +-{noise:.2f}{note}")
        refit = refit_for(lam)
        for ridge in args.ridge:
            gain, error = held_out(G, F, counts, ridge, args.rms, args.folds, measure=plain, refit=refit, eligible=fit_games)
            same, same_error = held_out(G, F, counts, ridge, args.rms, args.folds,
                                        refit=self_measure(refit), eligible=fit_games)
            inside = float(G.mean(0) @ direction(G, F, counts, ridge, args.rms))
            line = (f"    ridge {ridge:<6g} CV held-out gain by results {50 * gain:+.2f}+-{50 * error:.2f} (fold SE, optimistic) | by this lam "
                    f"{50 * same:+.2f}+-{50 * same_error:.2f} | in-sample {50 * inside:+.2f}")
            if audit is not None:
                a_gain, a_error, n_audit = audit_gain(G, F, counts, ridge, args.rms, audit, refit=refit)
                line += f" | AUDIT gain by results {50 * a_gain:+.2f}+-{50 * a_error:.2f} on {n_audit} untouched games"
            print(line, flush=True)
            if anchor is not None:
                for weight in args.anchor_weight:
                    pulled = (anchor[0], float(weight))
                    gain_a, error_a = held_out(G, F, counts, ridge, args.rms, args.folds, measure=plain, anchor=pulled,
                                              refit=refit, eligible=fit_games)
                    after = drift(F, counts, anchor[0] + direction(G, F, counts, ridge, args.rms, pulled))
                    plain_after = drift(F, counts, anchor[0] + direction(G, F, counts, ridge, args.rms))
                    aline = ""
                    if audit is not None:   # Codex round 3: the exported step is the ANCHORED direction, so audit that one too
                        aa_gain, aa_error, n_audit = audit_gain(G, F, counts, ridge, args.rms, audit, anchor=pulled, refit=refit)
                        aline = f" | AUDIT gain by results {50 * aa_gain:+.2f}+-{50 * aa_error:.2f} (anchored direction)"
                    print(f"      anchor weight {weight:<5g} held-out gain by results {50 * gain_a:+.2f}+-{50 * error_a:.2f} | drift after the step "
                          f"{after:.3f} rms logits (unanchored step: {plain_after:.3f}){aline}", flush=True)
                    if weight == anchor[1]:
                        gain = gain_a
            if best is None or gain > best[0]:
                best = (gain, ridge, lam)
    if args.traced:
        r_tr = traced_advantages(data, args.trace_baseline, fit_games)
        G = game_gradient(data, X, r_tr, temperature)
        sums["traced"] = G
        noise = 50 * args.rms * np.sqrt((r_tr ** 2).mean() * counts.mean() / games) / temperature
        corr = correlation(r_tr, r)
        print(f"  traced credit ({Path(args.traced).stem}, {args.trace_mode}, delay {args.trace_delay}, lam {args.trace_lam:g}, baseline {args.trace_baseline}): {100 * float((data['traced'] != 0).mean()):.0f}% of decisions "
              f"carry credit, correlation with the result residual {metric_text(corr, '+.3f')}; noise of a random step +-{noise:.2f} (credit units)")
        for ridge in args.ridge:
            gain, error = held_out(G, F, counts, ridge, args.rms, args.folds, measure=plain,
                                   refit=refit_traced, eligible=fit_games)
            same, same_error = held_out(G, F, counts, ridge, args.rms, args.folds,
                                        refit=self_measure(refit_traced), eligible=fit_games)
            inside = float(G.mean(0) @ direction(G, F, counts, ridge, args.rms))
            print(f"    ridge {ridge:<6g} held-out gain by results {50 * gain:+.2f}+-{50 * error:.2f} | by traced credit {50 * same:+.2f}+-{50 * same_error:.2f}"
                  f" | in-sample {50 * inside:+.2f}", flush=True)
        for mix in args.traced_mix:
            G = game_gradient(data, X, r + mix * r_tr, temperature)
            sums[f"mix{mix:g}"] = G
            print(f"  results + {mix:g} x traced credit:")
            def refit_mixed(train):
                residuals = residual(data, train)
                credit = traced_advantages(data, args.trace_baseline, train)
                return (game_gradient(data, X, residuals + mix * credit, temperature),
                        game_gradient(data, X, residuals, temperature))

            for ridge in args.ridge:
                gain, error = held_out(G, F, counts, ridge, args.rms, args.folds, measure=plain,
                                       refit=refit_mixed, eligible=fit_games)
                print(f"    ridge {ridge:<6g} held-out gain by results {50 * gain:+.2f}+-{50 * error:.2f}", flush=True)
    if args.luck:
        # The written (and in-sample) luck step uses the same rule as its AUDIT/CV refits: coefficients refitted on
        # the fitting games (luck_refit_correction). The cached data["luck"] was fitted per run on that cache's own
        # game set, so with several --run inputs or --max-games the audited step was not the written one (R2-S4-SAC);
        # for one run with the cache's full game set the two corrections are identical.
        correction = luck_refit_correction(data, fit_mask)
        r_cv = r - correction
        G = game_gradient(data, X, r_cv, temperature)
        sums["luck"] = G
        refit_luck = chosen_refit("luck")            # coefficients refitted per training mask, never the cached ones
        noise = 50 * args.rms * np.sqrt((r_cv ** 2).mean() * counts.mean() / games) / temperature
        noise0 = 50 * args.rms * np.sqrt((r ** 2).mean() * counts.mean() / games) / temperature
        print(f"  chance control variate (luck_{args.luck}): per-decision residual sd {r.std():.3f} -> {r_cv.std():.3f}; "
              f"noise of a random step +-{noise0:.2f} -> +-{noise:.2f}; correlation of the correction with the residual {metric_text(correlation(correction, r), '+.3f')}")
        for ridge in args.ridge:
            gain, error = held_out(G, F, counts, ridge, args.rms, args.folds, measure=plain, refit=refit_luck, eligible=fit_games)
            same, same_error = held_out(G, F, counts, ridge, args.rms, args.folds,
                                        refit=self_measure(refit_luck), eligible=fit_games)
            inside = float(G.mean(0) @ direction(G, F, counts, ridge, args.rms))
            line = (f"    ridge {ridge:<6g} CV held-out gain by results {50 * gain:+.2f}+-{50 * error:.2f} (fold SE, optimistic) | by corrected results "
                    f"{50 * same:+.2f}+-{50 * same_error:.2f} | in-sample {50 * inside:+.2f}")
            if audit is not None:
                a_gain, a_error, n_audit = audit_gain(G, F, counts, ridge, args.rms, audit, refit=refit_luck)
                line += f" | AUDIT gain by results {50 * a_gain:+.2f}+-{50 * a_error:.2f} on {n_audit} untouched games"
            print(line, flush=True)
    if args.by_kind is not None:
        def refit_kind_rows(train, traced=False):
            residuals = residual(data, train)
            credit = traced_advantages(data, args.trace_baseline, train) if traced else residuals
            return game_gradient(data, X, credit, temperature), residuals

        print_by_kind("results direction (lam 1)", held_out_by_kind(
            plain, F, counts, args.by_kind, args.rms, args.folds, X, r, data, temperature,
            refit=refit_kind_rows, eligible=fit_games))
        if args.traced:
            print_by_kind("traced-credit direction", held_out_by_kind(
                sums["traced"], F, counts, args.by_kind, args.rms, args.folds, X, r, data, temperature,
                refit=lambda train: refit_kind_rows(train, traced=True), eligible=fit_games))
    if args.write_luck is not None:
        if not args.luck:
            raise SystemExit("--write-luck needs --luck")
        best = (None, args.write_luck, "luck")
    elif args.write_traced is not None:
        if not args.traced:
            raise SystemExit("--write-traced needs --traced")
        best = (None, args.write_traced, "traced")
    elif args.write:
        lam, ridge = args.write
        if lam not in sums:
            sums[lam] = plain if lam >= 1. else game_gradient(data, X, advantages(data, lam, fit_games), temperature)
            if lam < 1. and args.blend:
                sums[lam] = sums[lam] + bias_share(plain[fit_mask], sums[lam][fit_mask])[0] * (plain - sums[lam])
        best = (None, ridge, lam)
    G = sums[best[2]]
    # The receipt describes the exact selected estimator and anchor. In
    # particular, --write must not replace anchored CV with an unanchored,
    # globally calibrated estimate.
    selected_refit = chosen_refit(best[2])
    selected_gain, _ = held_out(G, F, counts, best[1], args.rms, args.folds, measure=plain,
                               anchor=anchor, refit=selected_refit, eligible=fit_games)
    best = (selected_gain, best[1], best[2])
    steps = list(args.steps)
    chosen = None
    quadratic = None
    if "kindctx" in args.features:
        theta_all = direction(G[fit_mask], F[fit_mask], counts[fit_mask], best[1], args.rms, anchor)
        names = kept_kinds(data)
        table = theta_all[slices["kindctx"]].reshape(len(names), kind_context.WIDTH)
        print(f"  context-conditioned kind offsets of the chosen direction (ridge {best[1]:g}, lam {best[2]}, step rms {args.rms:g}; logits;"
              f" fitted on the {int(fit_mask.sum())} non-audit games)")
        for line in kind_context.describe(table, names):
            print(line)
        levels_used = data["ctx"]
        shares = [np.bincount(levels_used[:, j].astype(np.int64), minlength=len(t) + 1) / len(levels_used) for j, (_, t) in enumerate(kind_context.FACTORS)]
        print("    share of decisions per level: " + "; ".join(f"{n} " + "/".join(f"{100 * v:.0f}%" for v in sh) for (n, _), sh in zip(kind_context.FACTORS, shares)))
    if args.curvature:
        c_int, e_int, c_kind, e_kind = held_out_curvature(G, F, counts, best[1], args.rms, args.folds, X, data, temperature,
                                                          kind_slice=slices.get("kind"), anchor=anchor,
                                                          refit=selected_refit, eligible=fit_games)
        c = c_int + (c_kind or 0.)
        e = float(np.sqrt(e_int ** 2 + (e_kind or 0.) ** 2))
        a = best[0]
        print(f"  curvature along the chosen step (ridge {best[1]:g}, lam {best[2]}): first-order a = {50 * a:+.2f} points per step of rms {args.rms:g}; "
              f"c = {50 * c:+.2f}+-{50 * e:.2f} points per step^2 (cross-decision {50 * c_int:+.2f}+-{50 * e_int:.2f}"
              + (f", within-decision of the kind block {50 * c_kind:+.2f}+-{50 * e_kind:.2f}" if c_kind is not None else "") + ")")
        multiple = quadratic_step(a, c, e)
        if multiple is None:
            print(f"  the curvature is not resolved (or not negative): the step size stays {args.rms:g}; quadratic model at 1 step {50 * (a + c / 2):+.2f}")
        else:
            chosen = round(args.rms * multiple, 4)
            print(f"  quadratic model: gain at 1 step {50 * (a + c / 2):+.2f}, at 2 steps {50 * (2 * a + 2 * c):+.2f}; best step {multiple:.2f} x = rms {chosen:g}, "
                  f"predicted gain {50 * (a * multiple + c * multiple ** 2 / 2):+.2f} points")
            if chosen not in steps:
                steps.append(chosen)
        quadratic = dict(a_points=50 * a, c_points=50 * c, c_error_points=50 * e, cross_decision_points=50 * c_int,
                         kind_within_points=None if c_kind is None else 50 * c_kind, best_multiple=multiple, rms=chosen)
    if args.output and not args.plant:
        if fitting_request(args) != request:
            raise ContractError("Fitting inputs or anchor changed during the update; refusing publication")
        ai = _engine()
        from .contracts import BehaviorPolicy
        from .offline_eval import _load_actor
        from .policy import PolicyActor, actor_digest
        if canonical_json(behavior_content(str(data["policy"]), args.side)) != str(data["behavior"]):
            raise ContractError("Focal policy weights changed since collection")
        for size in steps:
            theta = direction(G[fit_mask], F[fit_mask], counts[fit_mask], best[1], size, anchor)
            actor = _load_actor(ai, str(data["policy"]), args.side)
            write_step(actor, theta, data, args.features)
            identity = BehaviorPolicy(args.side, actor_digest(actor.model.state_dict(), None if actor.critic is None else actor.critic.state_dict()),
                                      actor.identity.generation, "learner")
            actor = PolicyActor(actor.model, identity, source_sha256=actor.source_sha256, critic=actor.critic)
            folder = Path(args.output) / f"rms_{size:g}"
            metadata = dict(run=[str(r) for r in args.run], side=args.side, games=games, ridge=best[1], lam=best[2], rms=size,
                                                            fitting_games=int(fit_mask.sum()),
                                                            audit_games=0 if audit is None else int(audit.sum()),
                                                            fit_request=request,
                                                            blend=bool(args.blend),
                                                            features=[n for n in ("last", "delta", "kind", "kindctx") if n in args.features],
                                                            kinds=kept_kinds(data) if ("kind" in args.features or "kindctx" in args.features) else [],
                                                            base_policy=str(data["policy"]),
                                                            anchor=dict(seed=args.anchor, weight=anchor[1]) if anchor is not None else None,
                                                            curvature=quadratic, curvature_step=(size == chosen),
                                                            held_out_gain_points=50 * best[0] * size / args.rms,
                                                            theta=theta.tolist(),
                                                            provenance=dict(behavior=json.loads(str(data["behavior"])),
                                                                inputs=[json.loads(str(r)) for r in data.get("source_receipts", [data.get("receipt")])]))
            publish_generation(ai, actor, folder, metadata)
            print(f"  wrote {folder} (ridge {best[1]:g}{', curvature-chosen step' if size == chosen else ''})")


if __name__ == "__main__":
    main()
