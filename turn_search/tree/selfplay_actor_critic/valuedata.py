"""Cheap games for VALUE training: the raw policies play, nothing is scored.

A value function needs positions and results, not candidate scores. This
plays games between the two raw policies (each samples its own move at
temperature T) and stores, per decision with more than one option, the
position as the mover sees it and as the other side sees it ("watch"),
half precision, in the layout value_fit.py reads (a run folder with
episode.json + arrays.npz per game; no menus, targets or afterstates). About
ten times cheaper than a scored collection and ~50 KB per game.
arrays.npz also holds each side's own view (1.261) of the FINAL position,
final_fp / final_sh (one row each, half precision, the last two members):
what happened after the last stored decision (forced moves, a battle that
resolves without a decision, the end-of-turn settlement) shows in no stored
row. Readers that do not use them ignore them (they read members by name).

  python -m selfplay_actor_critic.valuedata --output selfplay_actor_critic/runs/raw_t05_a --games 20000 --first-seed 30000000

Rerunning the same command continues (finished games are skipped independently
of worker count); policy contents and trajectory settings are frozen by the
run manifest. Legacy runs without content provenance require explicit migration.
A STOP file ends collection after the games in flight. Truncations retain their
evidence and block the all-scheduled-games terminal objective; they are never
silently discarded to admit a survivor-only cohort.

--gradient also stores what a DIRECT policy gradient on the last layer needs
(pg_fit.py). The move played is a sample of the policy, so every decision is
a small randomised experiment. If a side's logits are tilted by theta . h(s, a),
with h the 64 activations feeding the action scorer's final layer, the
first-order change of its expected result is theta . g,
  g = mean over games of sum over decisions of
      (h(played) - E_policy[h]) * (result - baseline) / T.
Stored per decision: that centred vector, the kind of the played move and the
policy's probability mass per kind (the same for a per-kind offset), the
log-probability of the played move. No value function is involved.

--fp-policy / --sh-policy: 'seed' (the epoch-8 policy) or a fitted policy
folder (gated_fit.py output).

--full-side sh also accumulates the policy gradient with respect to EVERY
policy weight (pg_full.py), not the last layer only. Per decision of that
side the worker backpropagates the centred score
  u = d/d theta [ logit(played) - sum_a behaviour(a) logit(a) ]
(about 5 ms on one thread) and runs the two estimators online:
  G = sum_k delta_k e_k,  e_k = lam e_(k-1) + u_k,  delta_k = V(k+1) - V(k)
      (GAE(lam) by eligibility trace; V = --value-map applied to the
      critic of --full-critic-dir, V after the last decision = the result)
  M = result * sum_d u_d - sum_d V(d) u_d   (the results gradient, lam 1)
The parent adds each game's G, M and the diagonal of the Fisher matrix
(sum_d u_d^2) into --full-folds accumulators by seed (fullgrad.npz, saved
every 100 games), so a step estimated on some folds can be measured on the
others. A small sample of decisions (--probe-rate) keeps its whole menu, so
the size of a step in logits can be measured exactly.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from .contracts import ContractError
from .durability import (behavior_content, cohort_status, critic_content, digest_file, ensure_run_manifest,
                         episode_folder, episode_index, file_lock, validate_episode)
from .runtime_io import _atomic_write, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = ROOT / "audit/card_gap_followthrough_sep14/epoch_extension_to8_r1/run_r1"
_STATE = {}


class LastLayer:
    """The activations feeding a policy head's final linear layer, for the menu of the last forward pass.
    Heads tapped: the action scorer ("last", 64) and the additive policy_delta head ("delta", 128); a tilt of
    either final layer's weights changes the logits by theta . activations."""

    def __init__(self, scorer):
        import torch
        if not isinstance(scorer[-1], torch.nn.Linear):
            raise ValueError("The head must end in a linear layer")
        self.width = int(scorer[-1].in_features)
        self.parts = []
        scorer[-2].register_forward_hook(lambda module, inputs, output: self.parts.append(output.detach()))

    def take(self):
        import torch
        h = torch.cat(self.parts).double().cpu().numpy()
        self.parts.clear()
        return h


def full_decision(actor, inputs, sampler, temperature, params):
    """Sample the move exactly as PolicyActor.predict + Prediction.sample would, and return the centred score:
    the gradient of logit(played) - sum_a behaviour(a) * logit(a) in every policy weight (flat float32)."""
    import torch
    from .policy import Prediction, forward_ragged
    with torch.enable_grad():
        _, (logits,) = forward_ragged(actor.model, [inputs], max_action_rows=actor.max_action_rows)
        host = logits.detach().cpu().numpy().astype(np.float64)
        shifted = host - host.max()
        logp = shifted - np.log(np.exp(shifted).sum())
        selected = Prediction(logp, 0.).sample(sampler, temperature)
        scaled = logp / temperature
        behaviour = np.exp(scaled - scaled.max())
        behaviour /= behaviour.sum()
        weights = torch.from_numpy(-behaviour).to(logits.dtype)
        weights[selected] += 1.
        grads = torch.autograd.grad((logits * weights).sum(), params, allow_unused=True)
    vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
    return int(selected), logp, vector


class GradientTrace:
    """One game's policy-gradient sums for one side, computed online.

    step(u, v): u = the centred score of the decision just taken, v = V of its position.
    finish(result) returns
      G = sum_d u_d * A_d,  A_d = sum over k >= d of lam^(k-d) * (V[k+1] - V[k]),  V after the last decision = result
          (GAE(lam); as an eligibility trace: G = sum_k delta_k e_k, e_k = lam e_(k-1) + u_k)
      M = sum_d u_d * (result - V_d)                                  (the results gradient, lam 1)
      fisher = sum_d u_d^2, decisions, result."""

    def __init__(self, size, lam):
        import torch
        self.lam = float(lam)
        self.trace, self.total, self.first, self.second, self.fisher = (torch.zeros(size) for _ in range(5))
        self.previous, self.count = None, 0

    def step(self, vector, value):
        if self.previous is not None:
            self.total.add_(self.trace, alpha=value - self.previous)      # delta of the previous decision times its trace
        self.trace.mul_(self.lam).add_(vector)
        self.first.add_(vector)
        self.second.add_(vector, alpha=value)
        self.fisher.addcmul_(vector, vector)
        self.previous, self.count = float(value), self.count + 1

    def finish(self, result):
        self.total.add_(self.trace, alpha=result - self.previous)
        return dict(G=self.total.numpy(), M=(result * self.first - self.second).numpy(), fisher=self.fisher.numpy(),
                    decisions=self.count, result=float(result))


def policy_parameters(model):
    """(names, tensors) of the weights a policy step may move: everything but the value head."""
    pairs = [(n, p) for n, p in model.named_parameters() if not n.startswith("value_head")]
    return [n for n, _ in pairs], [p for _, p in pairs]


def play(task):
    output, seed, temperature, worker, policies, gradient, full, *rest = task
    chance = rest[0] if rest else None          # --chance-log: chance_log.ChanceLogger settings (dict) or None
    manifest, existing = rest[1:3]
    folder = Path(existing).parent if existing else episode_folder(output, seed)
    # With --full-side the parent asks only for games missing from its accumulators: a game already on disk is
    # played again (same seed, same game) for its gradient and not written twice.
    write = not (folder / "episode.json").exists()
    if not write:
        saved = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        validate_episode(saved, manifest)
        if saved["end"]["status"] != "terminated":
            return seed, saved["end"]["status"], 0, None
    if not write and not full:
        return seed, "exists", 0, None
    if "ai" not in _STATE:
        from .generation import require_generation
        require_generation("valuedata.play")
        os.environ.update(CUDA_VISIBLE_DEVICES="-1", WOTR_NATIVE="0")  # '-1', never '' (Windows)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "src"))
        cwd = os.getcwd()
        import ai
        os.chdir(cwd)
        import torch
        torch.set_num_threads(1)
        from .offline_eval import _load_actor
        if {s: behavior_content(policies[s], s) for s in ("fp", "sh")} != manifest["behavior_content"]:
            raise ContractError("Behavior weights changed after collection admission")
        _STATE["ai"] = ai
        _STATE["run_identity"] = manifest["identity"]
        _STATE["actors"] = {s: _load_actor(ai, policies[s], s) for s in ("fp", "sh")}
        if {s: behavior_content(policies[s], s) for s in ("fp", "sh")} != manifest["behavior_content"]:
            raise ContractError("Behavior weights changed while loading actors")
        _STATE["taps"] = {s: [LastLayer(_STATE["actors"][s].model.spatial_action_scorer), LastLayer(_STATE["actors"][s].model.policy_delta)]
                          for s in ("fp", "sh")} if gradient else {}
        if full:
            from .policy import load_critic
            _STATE["critic"] = load_critic(ai, Path(full["critic_dir"]) / f"{full['side']}_critic.pt", full["side"])
            _STATE["params"] = policy_parameters(_STATE["actors"][full["side"]].model)[1]
    if _STATE["run_identity"] != manifest["identity"]:
        raise ContractError("Worker actor cache belongs to a different collection contract")
    ai, actors, taps = _STATE["ai"], _STATE["actors"], _STATE["taps"]
    if full:
        import torch
        sums = GradientTrace(sum(p.numel() for p in _STATE["params"]), full["lam"])
        probe_rng = np.random.default_rng(seed * 31 + 7)
        probes = dict(row=[], selected=[], count=[], candidates=[], regions=[], probabilities=[], kinds=[])
    from .contracts import CONTRACT
    from .environment import PythonEnvironment
    env = PythonEnvironment(ai, f"raw-{seed}", seed)
    logger = None
    if chance:
        from .chance_log import ChanceLogger
        logger = ChanceLogger(env, ai, seed, **chance)
    rows, observations, watch = [], [], []
    centred, kind_played, kind_mass, played_logp, top, kind_index = [], [], [], [], [], {}
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        while True:
            if logger is not None:
                logger.before_prepare(len(rows))
            frame = env.prepare()
            if frame is None:
                break
            if logger is not None and logger.pending is not None:
                logger.on_frame(frame.side, frame.inputs.observation,
                                ai.encode_state(env.state, env.sides["sh" if frame.side == "fp" else "fp"]))
            if len(frame.actions) == 1:
                if logger is not None:
                    logger.before_apply(frame, 0, len(rows))
                env.apply(frame, 0)
                if logger is not None:
                    logger.after_apply()
                continue
            for group in taps.values():
                for tap in group:
                    tap.parts.clear()
            if full and frame.side == full["side"]:
                selected, logp, vector = full_decision(actors[frame.side], frame.inputs, env.samplers[frame.side], temperature, _STATE["params"])
                with torch.inference_mode():
                    critic = _STATE["critic"]
                    x = torch.from_numpy(np.asarray(frame.inputs.observation, np.float32)).unsqueeze(0)
                    value = full["value_map"][0] + full["value_map"][1] * float(critic.value_head(critic.forward_backbone(x)[0]).reshape(-1)[0])
                sums.step(vector, value)
                if probe_rng.random() < full["probe_rate"]:
                    probes["row"].append(len(rows)); probes["selected"].append(selected); probes["count"].append(len(logp))
                    probes["candidates"].append(np.asarray(frame.inputs.candidates)); probes["regions"].append(np.asarray(frame.inputs.regions))
                    probes["probabilities"].append(np.asarray(frame.inputs.probabilities))
                    probes["kinds"].extend(frame.inputs.kinds)
            else:
                prediction = actors[frame.side].predict(frame.inputs)
                selected = prediction.sample(env.samplers[frame.side], temperature)
                logp = np.asarray(prediction.log_probabilities, np.float64)
            if gradient:
                hidden = np.concatenate([tap.take() for tap in taps[frame.side]], axis=1)
                if len(hidden) != len(logp):
                    raise RuntimeError("Last-layer activations do not match the menu")
                scaled = logp / temperature
                behaviour = np.exp(scaled - scaled.max())
                behaviour /= behaviour.sum()
                kinds = np.fromiter((kind_index.setdefault(a.action_type, len(kind_index)) for a in frame.actions), np.int64, len(logp))
                centred.append((hidden[selected] - behaviour @ hidden).astype(np.float32))
                kind_played.append(int(kinds[selected]))
                kind_mass.append(np.bincount(kinds, behaviour))
                played_logp.append(float(logp[selected]))
                top.append(int(selected) == int(np.argmax(logp)))
            other = env.sides["sh" if frame.side == "fp" else "fp"]
            observations.append(np.asarray(frame.inputs.observation, np.float16))
            # "watch" = the OTHER side's view of the same decision (its private knowledge included). It is the second
            # input of the both-views critic (full_state_critic.py; pg_fit reads it as observations_other).
            watch.append(np.asarray(ai.encode_state(env.state, other), np.float16))
            rows.append(dict(side=frame.side, turn=int(frame.turn), phase=str(frame.phase), selected=int(selected),
                             kind="learned", action_count=len(frame.actions), searched=False, search_value=None))
            if logger is not None:
                logger.before_apply(frame, selected, len(rows))
            env.apply(frame, selected)
            if logger is not None:
                logger.after_apply()
    end = env.end
    if end is None:
        raise ContractError("Environment ended without an accounted episode end")
    payload = None
    if full and sums.count and end.status == "terminated":
        payload = sums.finish(end.terminal_return(full["side"]))
    if not write:
        return seed, "ok", len(rows), payload
    folder.mkdir(parents=True, exist_ok=True)
    extra = {}
    if full and probes["row"]:
        extra.update(probe_row=np.asarray(probes["row"], np.int32), probe_selected=np.asarray(probes["selected"], np.int32),
                     probe_count=np.asarray(probes["count"], np.int32), probe_candidates=np.concatenate(probes["candidates"]),
                     probe_regions=np.concatenate(probes["regions"]), probe_probabilities=np.concatenate(probes["probabilities"]),
                     probe_kinds=np.asarray(probes["kinds"], dtype=str))
    if gradient and rows:
        mass = np.zeros((len(rows), len(kind_index)), np.float32)
        for i, part in enumerate(kind_mass):
            mass[i, :len(part)] = part
        extra.update(centred=np.stack(centred), centred_layout=np.array([tap.width for tap in taps["sh"]], np.int32),
                     kind_played=np.asarray(kind_played, np.int16), kind_mass=mass,
                     played_logp=np.asarray(played_logp, np.float32), top=np.asarray(top, bool),
                     kind_names=np.array([str(getattr(k, "name", k)) for k in sorted(kind_index, key=kind_index.get)]))
    elif gradient:
        widths = [tap.width for tap in taps["sh"]]
        extra.update(centred=np.empty((0, sum(widths)), np.float32), centred_layout=np.array(widths, np.int32),
                     kind_played=np.empty(0, np.int16), kind_mass=np.empty((0, 0), np.float32),
                     played_logp=np.empty(0, np.float32), top=np.empty(0, bool), kind_names=np.array([], dtype=str))
    if logger is not None and observations:
        extra.update(logger.arrays(len(observations[0])))
        extra["chance_summary"] = np.array(json.dumps(logger.summary()))
    # Each side's own view of the final state (round 17, R17-SAC-FINALVIEW): a capture made after the last stored
    # decision of either side shows only here (the league's defend_targets objective reads it instead of replaying
    # the game).  Appended last, so every other member keeps its place.
    extra.update({f"final_{s}": np.asarray(ai.encode_state(env.state, env.sides[s]), np.float16) for s in ("fp", "sh")})
    _atomic_write(folder / "arrays.npz", lambda stream: np.savez_compressed(stream,
        observations=np.stack(observations) if observations else np.empty((0, CONTRACT.observation_width), np.float16),
        watch=np.stack(watch) if watch else np.empty((0, CONTRACT.observation_width), np.float16), **extra))
    # transition_count is the number of STORED rows (forced moves are not stored): what value_fit checks.
    meta = dict(schema="value_data_v1", contract_id=CONTRACT.identity, episode_id=f"raw-{seed}", seed=seed,
                temperature=temperature, policies=policies, rows=rows,
                run_identity=manifest["identity"], behavior_content=manifest["behavior_content"],
                arrays_sha256=digest_file(folder / "arrays.npz"),
                end=dict(status=end.status, transition_count=len(rows), winner=end.winner, victory_route=end.victory_route, reason=end.reason,
                         engine_transitions=end.transition_count))
    write_json(folder / "episode.json", meta)
    return seed, "ok" if end.status == "terminated" else end.status, len(rows), payload


class FullGradient:
    """The parent's accumulators of --full-side: per fold (seed modulo folds) the sums over games of G and M,
    the Fisher diagonal, and which seeds are in them. One writer, saved atomically."""

    def __init__(self, path, folds, meta):
        self.path, self.folds, self.meta = Path(path), int(folds), dict(meta)
        self.G = self.M = self.fisher = None
        self.fisher_by_fold = None
        self.decisions_by_fold = np.zeros(self.folds, np.int64)
        self.games, self.results = np.zeros(self.folds, np.int64), np.zeros(self.folds, np.float64)
        self.decisions, self.seeds, self.unsaved = 0, [], 0
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as d:
                saved = json.loads(str(d["meta"]))
                if saved != self.meta:
                    raise SystemExit(f"{self.path} was accumulated with other settings: {saved}")
                if not {"fisher_by_fold", "decisions_by_fold"} <= set(d.files):
                    raise ContractError("Legacy full-gradient accumulator lacks fold-local Fisher evidence; "
                                        "regenerate this derived accumulator before resuming")
                self.G, self.M, self.fisher = d["G"].copy(), d["M"].copy(), d["fisher"].copy()
                self.fisher_by_fold = d["fisher_by_fold"].copy()
                self.decisions_by_fold = d["decisions_by_fold"].copy()
                self.games, self.results = d["games"].copy(), d["results"].copy()
                self.decisions, self.seeds = int(d["decisions"]), [int(s) for s in d["seeds"]]

    def add(self, seed, payload):
        if seed in self.seeds:
            raise ContractError(f"Duplicate full-gradient seed {seed}")
        if self.G is None:
            size = len(payload["G"])
            self.G, self.M = np.zeros((self.folds, size), np.float32), np.zeros((self.folds, size), np.float32)
            self.fisher = np.zeros(size, np.float64)
            self.fisher_by_fold = np.zeros((self.folds, size), np.float64)
        fold = seed % self.folds
        self.G[fold] += payload["G"]
        self.M[fold] += payload["M"]
        self.fisher += payload["fisher"]
        self.fisher_by_fold[fold] += payload["fisher"]
        self.games[fold] += 1
        self.results[fold] += payload["result"]
        self.decisions += int(payload["decisions"])
        self.decisions_by_fold[fold] += int(payload["decisions"])
        self.seeds.append(int(seed))
        self.unsaved += 1

    def save(self):
        if self.G is None or not self.unsaved:
            return
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez(temporary, G=self.G, M=self.M, fisher=self.fisher, games=self.games, results=self.results,
                 fisher_by_fold=self.fisher_by_fold, decisions_by_fold=self.decisions_by_fold,
                 decisions=np.int64(self.decisions), seeds=np.asarray(self.seeds, np.int64), meta=np.array(json.dumps(self.meta)))
        os.replace(temporary, self.path)
        self.unsaved = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="run folder (its name is the run's name in value_fit)")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--first-seed", type=int, default=30000000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--fp-policy", default="seed", help="'seed' (epoch-8) or a fitted policy folder")
    parser.add_argument("--sh-policy", default="seed", help="'seed' (epoch-8) or a fitted policy folder")
    parser.add_argument("--gradient", action="store_true", help="also store the last-layer policy-gradient terms (pg_fit.py)")
    parser.add_argument("--full-side", default="", choices=("", "fp", "sh"), help="accumulate the gradient in every policy weight of this side (pg_full.py)")
    parser.add_argument("--full-critic-dir", default=str(ROOT / "selfplay_actor_critic/seeds/full_critic_r2"))
    parser.add_argument("--value-map", type=float, nargs=2, default=[0., 1.], metavar=("A", "G"),
                        help="result ~ A + G * critic value (pg_fit prints the fit); used for the baseline and the GAE deltas")
    parser.add_argument("--full-lam", type=float, default=.9)
    parser.add_argument("--full-folds", type=int, default=5)
    parser.add_argument("--probe-rate", type=float, default=.01, help="share of the side's decisions stored with their whole menu")
    parser.add_argument("--chance-log", action="store_true",
                        help="log every chance event with its exact outcome distribution and both views of every alternative (chance_log.py; luck.py)")
    parser.add_argument("--chance-roll-samples", type=int, default=8, help="sampled alternatives of the action-dice roll per turn")
    parser.add_argument("--chance-max-walks", type=int, default=64, help="enumeration budget per apply before falling back to sampling")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.games < 1 or args.temperature <= 0:
        raise ContractError("Collection needs positive games, workers and sampling temperature")
    with file_lock(output / "collection.lock"):
        return collect(args, output)


def collect(args, output):
    policies = dict(fp=args.fp_policy, sh=args.sh_policy)
    behaviors = {s: behavior_content(policies[s], s) for s in ("fp", "sh")}
    settings = vars(args).copy()
    if args.full_side:
        settings["full_critic_content"] = critic_content(args.full_critic_dir, args.full_side)
    manifest = ensure_run_manifest(output, settings, behaviors)
    existing = episode_index(output, require_identity=True)
    for path, meta in existing.values():
        validate_episode(meta, manifest)
        if not (path.parent / "arrays.npz").is_file():
            raise ContractError(f"Missing episode arrays: {path}")
    full, store, already = None, None, 0
    if args.full_side:
        full = dict(side=args.full_side, critic_dir=args.full_critic_dir, value_map=[float(v) for v in args.value_map], lam=float(args.full_lam),
                    probe_rate=float(args.probe_rate))
        store = FullGradient(output / "fullgrad.npz", args.full_folds,
                             dict(side=args.full_side, lam=float(args.full_lam), value_map=[float(v) for v in args.value_map],
                                  policy=policies[args.full_side], temperature=float(args.temperature), folds=int(args.full_folds),
                                  critic=settings["full_critic_content"], run_identity=manifest["identity"]))
        already = len(store.seeds)
    held = set(store.seeds) if store else set()
    chance = dict(roll_samples=args.chance_roll_samples, max_walks=args.chance_max_walks) if args.chance_log else None
    tasks = [(str(output), args.first_seed + i, args.temperature, i % args.workers, policies, args.gradient, full, chance,
              manifest, str(existing[args.first_seed + i][0]) if args.first_seed + i in existing else None) for i in range(args.games)
             if args.first_seed + i not in held]
    started, done, rows, finished = time.perf_counter(), already, 0, already
    stop = output / "STOP"
    from .generation import generation_pool_kwargs
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             **generation_pool_kwargs("valuedata.collect")) as pool:
        for seed, status, count, payload in pool.map(play, tasks, chunksize=1):
            done += 1
            rows += count
            finished += status in ("ok", "exists")
            if store is not None and payload is not None:
                store.add(seed, payload)
                if store.unsaved >= 100:
                    store.save()
            if done % 200 == 0:
                print(f"{done}/{args.games} games ({finished} finished), {rows} rows, {time.perf_counter() - started:.0f}s", flush=True)
            if stop.exists():
                print("STOP file found: leaving after the games in flight", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
    if store is not None:
        store.save()
    print(f"done: {done} games ({finished} finished), {rows} rows, {time.perf_counter() - started:.0f}s", flush=True)
    status = cohort_status(output, write=True)
    if status["complete"]:
        return 0
    if any(status["counts"][s] for s in ("truncated", "failed", "interrupted")):
        print(f"Cohort blocked: unfinished evidence retained; no update admitted: {status['counts']}", file=sys.stderr)
        return 3
    return 75 if stop.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
