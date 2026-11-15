"""Offline fixed-panel evaluation of any policy at any temperature.

Plays the focal side (both sides, alternately) against the frozen epoch-6
evaluation opponents on a seed panel, raw policy, no search. Low temperature
keeps games aligned between nearby policies, which the runner's temperature-1
panel does not (two checkpoints within KL 6e-4 differed by 10 points).
The epoch-6 opponents (--opponent-dir default) and the epoch-8 pair the
search uses as its model of the other player are the running generation's
copies of those behaviour-cloning pairs (generation.bc_seed_dir).

  python -m selfplay_actor_critic.offline_eval --policy <fitted policy folder or actor seed> --temperature 0 --seeds 256
  python -m selfplay_actor_critic.offline_eval --policy <run>/checkpoints/<id> --temperature 0 --seeds 256
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# The 2026-09-15 behaviour-cloning run (epoch_00N/policy_run/{fp,sh}_epoch1.pt). Its files carry action contract
# observed_combat_choices_v7 and load under NO current generation (R17-SAC row 67); this module no longer reads them
# and loads the generation's copies through generation.bc_seed_dir. The name stays for its importers (paired_eval).
SEED_BASE = ROOT / "audit/card_gap_followthrough_sep14/epoch_extension_to8_r1/run_r1"


def _bc_pair_file(epoch, side, where):
    """<side>_epoch1.pt of the running generation's epoch-`epoch` behaviour-cloning pair."""
    from .generation import bc_seed_dir
    return bc_seed_dir(epoch, where) / f"{side}_epoch1.pt"


def _load_actor(ai, policy, side, critic_dir=""):
    from .policy import load_policy_seed
    from .durability import actor_source
    kind, path = actor_source(policy, side)
    if kind == "raw":
        actor = load_policy_seed(ai, path, side)
    elif kind == "actor":
        # An actor seed (export_seed.py, pg_fit.py): the policy as saved.
        from .policy import load_actor_seed
        actor = load_actor_seed(ai, path, side)
    else:
        from .search_learner import SearchLearner
        anchor = load_policy_seed(ai, _bc_pair_file(8, side, "offline_eval._load_actor"), side)
        learner = SearchLearner.load_checkpoint(ai, path.parent, side, anchor=anchor)
        actor = learner.export_actor()
    if critic_dir:
        # Attach a private critic (corpus_critic.py --private output) to the
        # policy; the identity then covers both, as the runner does.
        from .policy import PolicyActor, actor_digest, load_critic
        from .contracts import BehaviorPolicy
        critic = load_critic(ai, Path(critic_dir) / f"{side}_critic.pt", side)
        identity = BehaviorPolicy(side, actor_digest(actor.model.state_dict(), critic.state_dict()), actor.identity.generation, "learner")
        actor = PolicyActor(actor.model, identity, source_sha256=actor.source_sha256, critic=critic)
    return actor


_ACTORS = {}
_TREE_MODELS = {}


def _tree_models(ai):
    """The search's frozen model of each player (SearchCollector opponent_models): the running generation's epoch-8
    pair, loaded once per worker process."""
    from .policy import load_policy_seed
    if "models" not in _TREE_MODELS:
        _TREE_MODELS["models"] = {s: load_policy_seed(ai, _bc_pair_file(8, s, "offline_eval.play"), s) for s in ("fp", "sh")}
    return _TREE_MODELS["models"]


def play(task):
    (policy, focus, seed, temperature, opponent_dir, search, critic_dir, greedy, focal_temperature, value_scale, candidates,
     chance, search_temperature, resolve, trust, calibration) = task
    from .generation import require_generation
    require_generation("offline_eval.play")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", WOTR_NATIVE="0")  # '-1', never '' (Windows)
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    import torch
    torch.set_num_threads(1)
    from .policy import load_policy_seed
    from .environment import PythonEnvironment
    from .search_collector import SearchCollector
    other = "sh" if focus == "fp" else "fp"
    # The opponent is FROZEN (origin "historical"): it never searches and always
    # plays its raw policy at `temperature`. Until 2026-10-04 it was loaded with
    # the default origin "learner", so every --search run also searched the
    # opponent with its zero critic: in sampled mode it then played a
    # temperature-1 sample of its policy instead of its greedy move, which made
    # "sampled search vs raw greedy" comparisons invalid (different opponents).
    key = (policy, focus, critic_dir, opponent_dir)
    if key not in _ACTORS:
        # Loaded once per worker process and reused for every game (a policy holds no game state).
        frozen = Path(opponent_dir) / f"{other}_epoch1.pt"
        if frozen.exists():
            opponent = load_policy_seed(ai, frozen, other, origin="historical")
        else:
            # A fitted policy folder or an actor seed as the frozen opponent (--opponent-dir seeds/fit_fp_gen5).
            from .contracts import BehaviorPolicy
            from .policy import PolicyActor
            loaded = _load_actor(ai, opponent_dir, other)
            opponent = PolicyActor(loaded.model, BehaviorPolicy(other, loaded.identity.weights_sha256, loaded.identity.generation, "historical"),
                                   source_sha256=loaded.source_sha256)
        _ACTORS.clear()
        _ACTORS[key] = {focus: _load_actor(ai, policy, focus, critic_dir), other: opponent}
    actors = _ACTORS[key]
    env = PythonEnvironment(ai, f"offline-{focus}-{seed}", seed)
    config, models = None, None
    if search:
        # The focal side searches (afterstate, its attached critic); the
        # frozen opponent samples at `temperature`. The collector requires a
        # model of each player; afterstate mode reads the other player's only
        # with --resolve pending (its pre-roll battle choices, chance.resolve).
        from .search import SearchConfig
        config = SearchConfig(mode="afterstate", considered_actions=int(search), chance_samples=4, greedy=bool(greedy),
                              value_scale=float(value_scale), candidates=candidates, chance=chance,
                              play_temperature=float(search_temperature), resolve=resolve,
                              credit_gate=tuple(sorted((trust or {}).get(focus, {}).items())),
                              credit_calibration=tuple((kind, w, b) for kind, (w, b) in sorted(
                                  ((calibration or {}).get(focus, {}).get("slopes", {})).items())),
                              credit_scale=float((calibration or {}).get(focus, {}).get("scale", 1.)))
        models = _tree_models(ai)
    side_temperature = None if focal_temperature is None else {focus: focal_temperature}
    rollout = SearchCollector(actors, search=config, opponent_models=models, temperature=temperature,
                              side_temperature=side_temperature).collect(env)
    end = rollout.end
    return dict(focus=focus, seed=seed, status=end.status, winner=end.winner, route=end.victory_route,
                turns=env.state.turn, decisions=len(rollout.rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # R17-SAC row 67: 'seed' resolves in durability.actor_source to the generation's copy (generation.bc_seed_dir);
    # test_round17_R17_SAC_policy_help_sac pins this help to that resolution
    parser.add_argument("--policy", required=True,
                        help="a folder holding <side>_seed.pt (actor seed), <side>_epoch1.pt (network) or <side>/state.pt "
                             "(search-learner checkpoint), resolved by durability.actor_source, or 'seed' = the running "
                             "generation's untouched epoch-8 behaviour-cloning net (generation.bc_seed_dir(8))")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--first-seed", type=int, default=91400052)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--opponent-dir", default=None,
                        help="frozen opponent: a folder with <side>_epoch1.pt or an actor seed / fitted policy folder "
                             "(default: the running generation's epoch-6 behaviour-cloning pair, generation.bc_seed_dir)")
    parser.add_argument("--search", type=int, default=0,
                        help="focal side plays with the afterstate search over this many Gumbel candidates (0 = raw policy)")
    parser.add_argument("--critic-dir", default="", help="attach {fp,sh}_critic.pt private critics to the focal policy")
    parser.add_argument("--greedy-search", action="store_true",
                        help="with --search: play the search policy's best move (no Gumbel noise)")
    parser.add_argument("--focus", default="both", choices=("both", "fp", "sh"), help="which side(s) to evaluate")
    parser.add_argument("--candidates", default="gumbel", choices=("gumbel", "prior"),
                        help="with --search: evaluated actions are a Gumbel sample from the prior, or the prior's top N")
    parser.add_argument("--value-scale", type=float, default=.1,
                        help="with --search: weight of the critic's Q against the prior logit (sigma = (50 + visits) * scale * Q)")
    parser.add_argument("--focal-temperature", type=float, default=None,
                        help="sampling temperature of the FOCAL side's non-searched decisions (default: --temperature); "
                             "the opponent always uses --temperature")
    parser.add_argument("--chance", default="sample", choices=("sample", "exact"),
                        help="with --search: exact expectation over each candidate's chance outcomes, or sampled clones")
    parser.add_argument("--resolve", default="none", choices=("none", "pending"),
                        help="with --search and --chance exact: value moves whose chance is still pending where it has been resolved")
    parser.add_argument("--trust", default="",
                        help="with --search: trust-factor file {side: {kind: weight, '*': default}} (credit_validity.py --trust-output)")
    parser.add_argument("--calibration", default="",
                        help="with --search: calibration file from validity_bench.py calibrate (calibrated credit)")
    parser.add_argument("--search-temperature", type=float, default=1.,
                        help="with --search and no --greedy-search: temperature of the sampled move of the searched "
                             "policy (prior x exp(credit)); match it to --temperature for a like-for-like comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.opponent_dir is None:
        from .generation import bc_seed_dir
        args.opponent_dir = str(bc_seed_dir(6, "offline_eval.main"))
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    trust = json.loads(Path(args.trust).read_text(encoding="utf-8")) if args.trust else None
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
    sides = ("sh", "fp") if args.focus == "both" else (args.focus,)
    tasks = [(args.policy, focus, seed, args.temperature, args.opponent_dir, args.search, args.critic_dir, args.greedy_search,
              args.focal_temperature, args.value_scale, args.candidates, args.chance, args.search_temperature, args.resolve,
              trust, calibration)
             for focus in sides for seed in seeds]
    started = time.perf_counter()
    results = []
    from .generation import generation_pool_kwargs
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             **generation_pool_kwargs("offline_eval.main")) as pool:
        for result in pool.map(play, tasks, chunksize=4):
            results.append(result)
    summary = {}
    for focus in sides:
        rows = [r for r in results if r["focus"] == focus and r["status"] == "terminated"]
        wins = sum(r["winner"] == focus for r in rows)
        routes = {}
        for r in rows:
            key = r["winner"] + ":" + r["route"]
            routes[key] = routes.get(key, 0) + 1
        summary[focus] = dict(wins=wins, games=len(rows), win_rate=wins / len(rows) if rows else None, routes=routes,
                              mean_turns=sum(r["turns"] for r in rows) / len(rows) if rows else None)
    payload = dict(policy=args.policy, temperature=args.temperature, seeds=seeds, opponent_dir=args.opponent_dir,
                   search=args.search, greedy_search=args.greedy_search, critic_dir=args.critic_dir,
                   focal_temperature=args.focal_temperature, value_scale=args.value_scale, candidates=args.candidates,
                   opponent_origin="historical", chance=args.chance, search_temperature=args.search_temperature,
                   resolve=args.resolve, trust=trust, calibration=calibration,
                   seconds=time.perf_counter() - started, summary=summary, games=results,
                   # D6P-MIXED-GEN-ADAPTER: the engine generation and the older generations admitted through the
                   # prefix view (src/mixed_generation.py); None = only nets of the engine's own generation load
                   generation=dict(engine=os.environ.get("WOTR_ACT_GEN"),
                                   admitted=os.environ.get("WOTR_ADMIT_GENERATIONS") or None))
    Path(args.output).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for focus, s in summary.items():
        rate = "unavailable" if s["win_rate"] is None else f"{s['win_rate']:.3f}"
        turns = "unavailable" if s["mean_turns"] is None else f"{s['mean_turns']:.1f}"
        print(f"{args.policy} T={args.temperature} focus {focus}: {s['wins']}/{s['games']} = {rate}  turns {turns}  routes {s['routes']}")


if __name__ == "__main__":
    main()
