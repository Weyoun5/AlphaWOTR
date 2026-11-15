"""FPSP collection: Free Peoples games at v53 whose FP decisions carry search credit, or plain raw games.

Mode 'search' (the generation data):
  * FP = the current generation (a BC-format v53 checkpoint folder with fp_epoch1.pt, or an actor seed folder),
    with an OWN-VIEW critic attached (fpsp critics/*_v53, converted exactly from v52 or refitted at v53).
  * Every FP decision with more than one option: the policy samples its OWN move at --temperature (collection play
    "prior": the critic never steers play, so the games are exactly on-policy) and the --considered Gumbel-sampled
    candidates are each scored by the critic at their afterstate: exact expectation over the chance outcomes of the
    move (chance.py), a battle or eye allocation left pending valued where its dice are rolled (--resolve pending),
    and ONE resampled hidden world per decision (search.determinize_state: opponent hand vs its unseen cards,
    both own deck orders, the hunt pool, an unrevealed committed combat card).  Pre-roll decisions inside that
    resolution are taken by the FP policy and by the opponent MODEL (fixed: the deployed Shadow) on the resampled
    hand.  This is selfplay_actor_critic.search_collector.SearchCollector with search.SearchConfig(mode afterstate);
    nothing in it is re-implemented here.
  * SH = one Shadow net per game from the opponent pool (--opponents NAME:WEIGHT,...), chosen deterministically from
    the seed, raw policy at --temperature (never searches).
  Saved per game (folder g<seed>, written under a temp name and renamed when complete): search_rollout_v1
  (arrays.npz + episode.json, v53 widths) and fpsp.json (opponent, nets and their sha256, timing).
Mode 'raw': both sides raw at --temperature, no critic; for throughput and value data (same folder layout).

Resumable: a seed whose g<seed>/episode.json exists is skipped; a leftover temp folder is moved into _partial/ (never
deleted).  A STOP file in --out (or in --stop-file) ends collection after the games in flight.  One status line per
--status-every games, a ledger line per game (ledger.jsonl), progress.json.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fpsp_common as C  # noqa: E402

_STATE = {}


def parse_pool(text):
    pool = []
    for part in text.split(","):
        name, _, weight = part.partition(":")
        name = name.strip()
        if name not in C.SH_ARMS:
            raise SystemExit(f"unknown Shadow arm {name!r}; known: {sorted(C.SH_ARMS)}")
        pool.append((name, float(weight or 1)))
    total = sum(w for _, w in pool)
    if total <= 0:
        raise SystemExit("opponent weights must be positive")
    return [(n, w / total) for n, w in pool]


def opponent_for(seed, pool):
    """Deterministic opponent of a seed (independent of worker count and resume order)."""
    import numpy as np
    u = np.random.default_rng([int(seed), 7_300_531]).random()
    acc = 0.
    for name, weight in pool:
        acc += weight
        if u < acc:
            return name
    return pool[-1][0]


def policy_file(folder, side):
    folder = Path(folder)
    for name in (f"{side}_epoch1.pt", f"{side}_seed.pt"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{folder}: no {side}_epoch1.pt / {side}_seed.pt")


def _worker_init(generation):
    from selfplay_actor_critic.generation import pin_worker_generation
    pin_worker_generation(generation)


def _load(ai, spec):
    """Actors for one task spec, cached per process."""
    key = json.dumps(spec, sort_keys=True)
    if _STATE.get("key") == key:
        return _STATE["actors"]
    import torch
    torch.set_num_threads(1)
    from selfplay_actor_critic.contracts import BehaviorPolicy
    from selfplay_actor_critic.offline_eval import _load_actor
    from selfplay_actor_critic.policy import PolicyActor, actor_digest, load_critic
    fp = _load_actor(ai, spec["fp_policy"], "fp")
    if spec["critic_dir"]:
        critic = load_critic(ai, Path(spec["critic_dir"]) / "fp_critic.pt", "fp")
        identity = BehaviorPolicy("fp", actor_digest(fp.model.state_dict(), critic.state_dict()), 0, "learner")
        fp = PolicyActor(fp.model, identity, source_sha256=fp.source_sha256, critic=critic)
    shs = {}
    for name in spec["opponents"]:
        loaded = _load_actor(ai, spec["sh_policy"] if name == "custom" else str(C.SH_ARMS[name]), "sh")
        shs[name] = PolicyActor(loaded.model, BehaviorPolicy("sh", loaded.identity.weights_sha256,
                                                             loaded.identity.generation, "historical"),
                                source_sha256=loaded.source_sha256)
    model_name = spec["opponent_model"]
    model = shs.get(model_name)
    if model is None and spec["mode"] == "raw":
        model = next(iter(shs.values()))
    elif model is None:
        loaded = _load_actor(ai, str(C.SH_ARMS[model_name]), "sh")
        model = PolicyActor(loaded.model, BehaviorPolicy("sh", loaded.identity.weights_sha256, 0, "historical"),
                            source_sha256=loaded.source_sha256)
    _STATE.update(key=key, actors=dict(fp=fp, sh=shs, model=model))
    return _STATE["actors"]


def play(task):
    seed, spec = task
    out = Path(spec["out"])
    final = out / f"g{seed}"
    if (final / "episode.json").exists():
        return dict(seed=seed, status="exists")
    from selfplay_actor_critic.generation import require_generation
    require_generation("fp_collect.play")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", WOTR_NATIVE="0")
    ai = _STATE.get("ai")
    if ai is None:
        ai = _STATE["ai"] = C.import_engine()
    actors = _load(ai, spec)
    from dataclasses import asdict
    from selfplay_actor_critic.environment import PythonEnvironment
    from selfplay_actor_critic.search import SearchConfig
    from selfplay_actor_critic.search_collector import SearchCollector
    from selfplay_actor_critic.search_rollout_io import save_search_rollout
    opponent = spec["assignment"][str(seed)]
    sh = actors["sh"][opponent]
    started = time.perf_counter()
    if spec["mode"] == "search":
        config = SearchConfig(mode="afterstate", considered_actions=int(spec["considered"]),
                              chance_samples=int(spec["chance_samples"]), chance="exact", resolve=spec["resolve"],
                              chance_walks=int(spec["chance_walks"]), determinize=True, play="prior",
                              play_temperature=float(spec["temperature"]))
        collector = SearchCollector(dict(fp=actors["fp"], sh=sh), search=config,
                                    opponent_models=dict(fp=actors["fp"], sh=actors["model"]),
                                    temperature=float(spec["temperature"]), search_sides=("fp",))
    else:
        config = None
        collector = SearchCollector(dict(fp=actors["fp"], sh=sh), search=None, opponent_models=None,
                                    temperature=float(spec["temperature"]))
    env = PythonEnvironment(ai, f"fpsp-{spec['tag']}-{seed}", seed)
    with open(os.devnull, "w") as sink:
        import contextlib
        with contextlib.redirect_stdout(sink):
            rollout = collector.collect(env)
    seconds = time.perf_counter() - started
    temp = out / f"_tmp_g{seed}_{uuid.uuid4().hex[:8]}"
    save_search_rollout(rollout, temp, search_config=asdict(config) if config is not None else dict(mode="raw"))
    end = rollout.end
    searched = sum(1 for t in rollout.search if t is not None)
    meta = dict(schema="fpsp_game_v1", seed=seed, tag=spec["tag"], mode=spec["mode"], opponent=opponent,
                opponent_sha256=sh.source_sha256, opponent_model=spec["opponent_model"],
                fp_policy=spec["fp_policy"], fp_source_sha256=actors["fp"].source_sha256,
                fp_identity=asdict(actors["fp"].identity), critic_dir=spec["critic_dir"],
                status=end.status, winner=end.winner, route=end.victory_route, turns=int(env.state.turn),
                rows=len(rollout.rows), fp_rows=sum(r.side == "fp" for r in rollout.rows), searched=searched,
                seconds=seconds, pid=os.getpid(), error=rollout.error)
    (temp / "fpsp.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    try:
        os.replace(temp, final)
    except OSError:
        if (final / "episode.json").exists():          # another process finished the same seed first
            shutil.move(str(temp), str(out / "_partial" / temp.name))
            return dict(seed=seed, status="exists")
        raise
    return meta


def collect(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stop_files = [out / "STOP"] + ([Path(args.stop_file)] if args.stop_file else [])
    pool = [("custom", 1.0)] if args.sh_policy else parse_pool(args.opponents)
    seeds = [C.collect_seed(args.generation, args.first_index + i) if args.seed_block == "collect"
             else C.value_seed(args.generation, args.first_index + i) if args.seed_block == "value"
             else args.first_seed + i for i in range(args.games)]
    # leftovers of an interrupted write are archived, never deleted
    partial = out / "_partial"
    for leftover in out.glob("_tmp_g*"):
        partial.mkdir(exist_ok=True)
        shutil.move(str(leftover), str(partial / leftover.name))
    todo = [s for s in seeds if not (out / f"g{s}" / "episode.json").exists()]
    # round 17 row 70: a game folder compressed by retention (fpz.py: game.fpz in place of episode.json / arrays.npz) is
    # a finished game, not a missing one; replaying it ended in os.replace onto the non-empty folder (WinError 5)
    import fpz
    compressed = [s for s in todo if (out / f"g{s}" / fpz.ARCHIVE).exists()]
    if compressed:
        raise SystemExit(f"{out}: {len(compressed)} game folders are compressed (g{compressed[0]}/{fpz.ARCHIVE} ...); "
                         f"restore them first: python -B {HERE / 'fpz.py'} restore {out}")
    assignment = {str(s): opponent_for(s, pool) for s in seeds}
    spec = dict(out=str(out), mode=args.mode, tag=args.tag, fp_policy=str(Path(args.fp_policy).resolve()),
                critic_dir=str(Path(args.critic_dir).resolve()) if args.critic_dir else "",
                opponents=sorted({n for n, _ in pool}), opponent_model=args.opponent_model,
                sh_policy=str(Path(args.sh_policy).resolve()) if args.sh_policy else "",
                considered=args.considered, chance_samples=args.chance_samples, chance_walks=args.chance_walks,
                resolve=args.resolve, temperature=args.temperature, assignment=assignment)
    manifest = out / "collect_manifest.json"
    fixed = {k: v for k, v in spec.items() if k != "assignment"}
    fixed.update(pool=pool, seeds=[seeds[0], seeds[-1], len(seeds)],
                 fp_sha256=C.sha256_file(policy_file(args.fp_policy, "fp")),
                 critic_sha256=C.sha256_file(Path(args.critic_dir) / "fp_critic.pt") if args.critic_dir else None,
                 opponent_sha256={n: C.sha256_file(policy_file(args.sh_policy if n == "custom" else C.SH_ARMS[n], "sh"))
                                  for n in spec["opponents"]})
    saved = C.read_json(manifest)
    if saved is not None:
        if {k: saved.get(k) for k in fixed} != json.loads(json.dumps(fixed)):
            raise SystemExit(f"{manifest} records other settings; use a new --out")
    else:
        C.write_json(manifest, dict(fixed, created=C.stamp()))
    ledger = out / "ledger.jsonl"
    done = len(seeds) - len(todo)
    print(f"{C.stamp()} collect {args.tag}: {len(todo)} of {len(seeds)} games to play ({done} on disk), "
          f"{args.workers} workers, mode {args.mode}", flush=True)
    started = time.time()
    finished = played = 0
    wins = 0
    seconds = 0.
    stopped = False
    from selfplay_actor_critic.generation import require_generation
    generation = require_generation("fp_collect.collect")
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init, initargs=(generation,)) as ex:
            pending = set()
            queue = list(todo)
            while queue or pending:
                while queue and len(pending) < args.workers * 2 and not stopped:
                    pending.add(ex.submit(play, (queue.pop(0), spec)))
                if not pending:
                    break
                ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in ready:
                    meta = fut.result()
                    if meta.get("status") == "exists":
                        continue
                    played += 1
                    finished += meta["status"] == "terminated"
                    wins += meta.get("winner") == "fp"
                    seconds += meta["seconds"]
                    with open(ledger, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps({k: meta[k] for k in ("seed", "opponent", "status", "winner", "route",
                                                                      "turns", "rows", "fp_rows", "searched",
                                                                      "seconds")}) + "\n")
                    if played % args.status_every == 0 or (not queue and not pending):
                        wall = time.time() - started
                        print(f"{C.stamp()} {args.tag}: {done + played}/{len(seeds)} games | this session {played} "
                              f"in {wall:.0f}s = {3600 * played / max(wall, 1e-9):.0f} games/h | mean "
                              f"{seconds / played:.1f}s/game/worker | FP won {wins}/{played}", flush=True)
                        C.write_json(out / "progress.json", dict(tag=args.tag, games=len(seeds), on_disk=done + played,
                                                                 session_played=played, wall_seconds=wall,
                                                                 games_per_hour=3600 * played / max(wall, 1e-9),
                                                                 mean_game_seconds=seconds / played, fp_wins=wins,
                                                                 updated=C.stamp()))
                if not stopped and any(p.exists() for p in stop_files):
                    stopped = True
                    queue.clear()
                    print(f"{C.stamp()} STOP seen: finishing {len(pending)} games in flight", flush=True)
    on_disk = sum((out / f"g{s}" / "episode.json").exists() for s in seeds)
    summary = dict(tag=args.tag, games=len(seeds), on_disk=on_disk, complete=on_disk == len(seeds), stopped=stopped,
                   session_played=played, wall_seconds=time.time() - started,
                   mean_game_seconds=(seconds / played) if played else None, updated=C.stamp())
    C.write_json(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] else (75 if stopped else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="search", choices=("search", "raw"))
    ap.add_argument("--tag", default="fpsp")
    ap.add_argument("--fp-policy", required=True, help="folder with fp_epoch1.pt (BC format) or fp_seed.pt")
    ap.add_argument("--critic-dir", default="", help="folder with fp_critic.pt (own-view, v53); search mode")
    ap.add_argument("--opponents", default="R2SP:1", help="NAME:WEIGHT,... from fpsp_common.SH_ARMS")
    ap.add_argument("--opponent-model", default="R2SP", help="Shadow net modelling the opponent inside resolution")
    ap.add_argument("--sh-policy", default="", help="one explicit Shadow policy folder for every game (overrides --opponents)")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--first-index", type=int, default=0)
    ap.add_argument("--seed-block", default="collect", choices=("collect", "value", "explicit"))
    ap.add_argument("--first-seed", type=int, default=0, help="with --seed-block explicit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=.5)
    ap.add_argument("--considered", type=int, default=24)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-walks", type=int, default=96)
    ap.add_argument("--resolve", default="pending", choices=("none", "pending"))
    ap.add_argument("--status-every", type=int, default=16)
    ap.add_argument("--stop-file", default="")
    args = ap.parse_args()
    if args.mode == "search" and not args.critic_dir:
        ap.error("search mode needs --critic-dir")
    if args.seed_block == "explicit" and args.first_seed < C.PANEL_FIRST_SEED:
        ap.error("FPSP seeds start at 99300000")
    C.require_recipe_env("fp_collect")
    if not args.sh_policy and (any(n in C.V52_ARMS for n, _ in parse_pool(args.opponents)) or args.opponent_model in C.V52_ARMS):
        if os.environ.get("WOTR_ADMIT_GENERATIONS") != "v52":
            raise SystemExit("a v52 arm in the pool needs WOTR_ADMIT_GENERATIONS=v52")
    sys.path.insert(0, str(C.ROOT))
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
