"""Shared pieces of the cluster self-play league (hpc/remote/league, 2026-10-30).

USER 2026-10-30 (coordinator relay): "fp and shadow are trained identically right? previously we had different training
for each but i want them to be identical here."  HARD RULE, enforced here:
  * ONE configuration block (CONFIG) for every agent of either side: trainer, optimizer, KL limit, anchor schedule,
    advantage estimator, value fit, batch sizes, epochs, decision subsampling, held-out fraction, evidence gate, cycle
    cadence, opponent-pool / PFSP rules and exploiter reset rules.  check_config() refuses any side-specific key.
  * The ONLY per-agent differences are the side it plays and its reward table (OBJECTIVES), plus its role in the
    league (mirrored: every role has the same number of agents on each side, check_agents()).
  * config_hash() (sha256 of the canonical JSON of CONFIG) is printed by every trainer and written into every
    checkpoint, per side, so the report can show the FP and SH agents ran the same configuration.

Reward contracts (USER 2026-10-30; a timeout = the 40-turn cap = 0 for everyone):
    objective        +1                      0                       -1
    sh_win           any Shadow win          -                       any FP win
    sh_corruption    Shadow corruption win   Shadow military win     any FP win
    fp_win           any FP win              -                       any Shadow win
    fp_military      FP military win         Ring destroyed          any Shadow win
Victory routes of the engine (native_env.ROUTES / ai): ring_destroyed (FP), military (either side), ring_corruption (SH).
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
B = "claude_bugscan_2026-10-21"
ARMS = B + "/d6sp/arena/raw_512/arms"
BC_FILES = {"fp": (ARMS + "/R1/fp/fp_epoch1.pt", "beb80c795000a76edc3eccf01548ccf12fab45192fd5c1a3ee53865339bd4ce3"),
            "sh": (ARMS + "/R1/sh/sh_epoch1.pt", "5592c96e6a9641ef0b7833eacf3e59ab089c069fc21d73780753681f6b8a213c")}
SIDES = ("fp", "sh")
ROUTES = ("ring_destroyed", "military", "ring_corruption")

# ------------------------------------------------------------------------------------------------------------ config
# THE configuration of every agent (both sides, every role).  No key may name a side (check_config).
CONFIG = {
    "schema": "wotr_league_config_v1",
    "play": {
        "temperature": 0.5,          # every league game: both nets sample softmax(logits / 0.5) (the evaluation mode)
        "max_turns": 40,             # the 40-turn cap (native_env max_turns); a capped game = timeout = 0 for everyone
    },
    "record": {
        "fraction": 1.0,             # share of each LEARNING side's decisions written (all; disk is cheap, ~1 MB/game)
        "gradient": True,            # batchgen --gradient evidence + the other side's view (the hybrid update needs it)
    },
    # USER decision 2026-10-30 (after the full-network pilot): HYBRID = the race's last-layer policy step (pg_fit,
    # exactly gen/race.py's flags) + a fully trained value head per agent; "full" (the PPO trainer below) stays
    # available but is off.
    "update": "hybrid",
    "policy": {                      # pg_fit (selfplay_actor_critic/pg_fit.py) as fp_league fit() / gen/race.py run it
        "features": ["last"], "lam": [1.0, 0.98], "ridge": [1.0], "folds": 5, "write_lam": 1.0, "write_ridge": 1.0,
        "steps": [0.03], "anchor": "bc_start", "anchor_weight": 0.3,
        "gate": "none",              # coordinator ruling (review S3-1): the race's ungated step; held-out gain recorded
        "keep_pgfit_copy": False,    # pg_fit's published copy of the step is removed once the snapshot is verified;
                                     # its provenance (pg.json, complete.json) is kept in the snapshot as pgfit_*.json
        "critic": {                  # the race's BC-clean critic (gen/race.py + bc_critic.py + full_state_critic refits)
            "initial": "bc_value_heads", "refit": "both_views_frozen_bc_towers_fresh_fusion", "max_games": 6000,
            "epochs": 1, "width": 256, "lr": 1.0e-3, "batch": 4096, "staleness": "newest finished refit (the previous cycle)",
        },
    },
    "train": {
        "policy_temperature": 0.5,   # the policy that is improved is the T=0.5 policy softmax(logits/0.5) (= play)
        "loss": "ppo_clip",          # clipped surrogate, ratio pi_theta^T(a) / mu(a) vs the RECORDED behaviour logp
        "ppo_clip": 0.1,
        "epochs": 2,                 # passes over the cycle's (subsampled) decisions, early-stopped by kl_stop
        "batch_decisions": 256,
        "batch_rows": 196608,        # a minibatch also ends when its full legal menus reach this many rows
        "optimizer": "adam", "lr": 3.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8, "grad_clip": 1.0,
        "optimizer_state": "persistent_per_agent",   # Adam moments carried across cycles (reset with the agent)
        "kl_stop": 0.01,             # early stop when the EMA of KL(mu || pi_theta^T) over minibatches exceeds this
        "kl_ema": 0.8,
        "anchor": {                  # beta * KL(pi_theta^T || pi_BC^T), BC = the agent's side's R1 net (frozen)
            "beta0": 0.05, "floor": 0.01, "decay_learning_games": 250000,   # linear decay in the agent's own games
        },
        "advantage": {"estimator": "mc", "lambda": 1.0, "baseline": "behaviour_value_head", "normalize": True},
        "max_train_decisions": 400000,   # decision subsample per agent per cycle (uniform, deterministic hash)
        "holdout_fraction": 0.05,        # whole games (seed hash), never trained on
        "gate_z": 0.0,               # accept the policy step iff held-out first-order gain by results > gate_z * SE
        "value": {"input": "detached_backbone", "lr": 1.0e-4, "epochs": 4, "batch": 4096, "target_clip": 0.98,
                  "fit_after_policy": True,
                  # review S3-5: the agent's own-to-move positions AND the opponent-to-move positions in its own view
                  # (the shards' view rows), both with the agent's game return - the own-value search's leaves
                  "rows": "own_and_opponent_to_move"},
        "module_mode": "eval_autograd_v1",   # BatchNorm statistics and dropout fixed (full_network_v1)
        "trained": "backbone+gnn+card_embedding+action_scorer+policy_delta (policy loss); value_head (value loss, "
                   "detached features); chance_policy frozen at zero (not part of the BC net; the live search cannot "
                   "carry it)",
    },
    "league": {
        "cycle_learning_games_main": 10000,   # each main's learning games per cycle (a main-vs-main game counts twice)
        "target_learning_games_main": 500000,
        "share_main_vs_main": 0.40,           # of a main's games: current opposing mains (all pairs equally)
        "share_main_vs_main_exploiter": 0.15, # its own main exploiter (current; counts for both)
        "share_main_vs_pool": 0.45,           # PFSP over frozen snapshots of the other side + BC
        "bc_min_share_main": 0.05,            # of a main's games vs BC (inside the pool share): Elo anchor
        "extra_main_exploiter": 500,          # extra games of a main exploiter vs its target's recent snapshots
        "extra_league_exploiter": 1500,       # league exploiter vs PFSP over the whole other-side pool
        "extra_anti": 1300,                   # anti-strategy agent vs its target lineage (current + snapshots)
        "bc_mirror_games": 400,               # BC fp vs BC sh, every cycle (fixes the side term of the Elo)
        "recent_snapshots": 3,                # "recent" = the last 3 snapshots of a lineage
        "pfsp_power": 2.0,                    # weight (1 - p_win)^power, p_win from the current Elo fit
        "pfsp_floor": 0.25,                   # this share of every PFSP block is spread uniformly over the pool
        "exploiter_reset_cycles": 10,         # exploiters / anti agents reset every 10 cycles ...
        "exploiter_reset_win_rate": 0.70,     # ... or as soon as they beat their target at >= 70% in a cycle
        "exploiter_reset_min_games": 500,     #     (over at least 500 games against it)
        "exploiter_reset_to": "latest_same_side_win_main",   # fp_win for FP exploiters, sh_win for SH exploiters
    },
}

# The reward tables (the only per-agent training difference besides the side).
OBJECTIVES = {
    "fp_win": {"side": "fp", "win": {"ring_destroyed": 1.0, "military": 1.0}, "loss": -1.0, "timeout": 0.0},
    "fp_military": {"side": "fp", "win": {"ring_destroyed": 0.0, "military": 1.0}, "loss": -1.0, "timeout": 0.0},
    "sh_win": {"side": "sh", "win": {"ring_corruption": 1.0, "military": 1.0}, "loss": -1.0, "timeout": 0.0},
    "sh_corruption": {"side": "sh", "win": {"ring_corruption": 1.0, "military": 0.0}, "loss": -1.0, "timeout": 0.0},
}
# the live search's terminal profile for each objective (final search league; own_orch_mod PROFILE_CODES -> the search
# box's profile_terminal, native_search/sbox_w2/cpp/search_controller.inc): a loss is -1 for every profile; a win is
# +1 on-target, and off-target 0.0 for FP / 0.2 for SH (hard-coded in the box).  'mixed' = every win on-target.
#   fp_win -> mixed (= the table), sh_win -> mixed (= the table), fp_military -> military (= the table: ring win 0.0),
#   sh_corruption -> corruption: Shadow military win 0.2 in the search vs 0 in the table - the ONE terminal the box
#   cannot express (no profile gives an off-target Shadow win 0); closing it needs a search-box change (agent E /
#   native owner).  search_mismatches() lists it; BC keeps its own (ring / military).
SEARCH_PROFILE = {"fp_win": "mixed", "fp_military": "military", "sh_win": "mixed", "sh_corruption": "corruption"}
_PROFILE_CODE = {"fp": {"ring": 1, "military": 2, "mixed": 3}, "sh": {"corruption": 1, "military": 2, "mixed": 3}}
_ROUTE_KIND = {"military": 1, "ring_destroyed": 2, "ring_corruption": 3}


def search_terminal(side, profile, winner, route):
    """The search box's terminal value for `side` under `profile` (search_controller.inc profile_terminal)."""
    if winner not in SIDES:
        return 0.0
    if winner != side:
        return -1.0
    code = _PROFILE_CODE[side][profile]
    target = 1 if code == 2 else (2 if side == "fp" else 3)
    if code == 3 or _ROUTE_KIND[route] == target:
        return 1.0
    return 0.0 if side == "fp" else 0.2


def search_mismatches():
    """[(objective, winner, route, table value, search terminal)] where the live search's terminal differs from the
    reward table the agent's value head is trained on (decided games only; timeouts are 0 in both)."""
    out = []
    for objective, table in OBJECTIVES.items():
        side = table["side"]
        for winner in SIDES:
            routes = ("ring_destroyed", "military") if winner == "fp" else ("ring_corruption", "military")
            for route in routes:
                a = reward(objective, winner, route)
                b = search_terminal(side, SEARCH_PROFILE[objective], winner, route)
                if abs(a - b) > 1e-12:
                    out.append((objective, winner, route, a, b))
    return out

AGENTS = {
    # the full league (mirrored: 2 mains, 2 main exploiters, 1 league exploiter, 1 anti agent per side)
    "league": [
        dict(name="fp_win", side="fp", role="main", objective="fp_win"),
        dict(name="fp_mil", side="fp", role="main", objective="fp_military"),
        dict(name="sh_win", side="sh", role="main", objective="sh_win"),
        dict(name="sh_cor", side="sh", role="main", objective="sh_corruption"),
        dict(name="mx_fp_win", side="sh", role="main_exploiter", objective="sh_win", target="fp_win"),
        dict(name="mx_fp_mil", side="sh", role="main_exploiter", objective="sh_win", target="fp_mil"),
        dict(name="mx_sh_win", side="fp", role="main_exploiter", objective="fp_win", target="sh_win"),
        dict(name="mx_sh_cor", side="fp", role="main_exploiter", objective="fp_win", target="sh_cor"),
        dict(name="lx_fp", side="fp", role="league_exploiter", objective="fp_win"),
        dict(name="lx_sh", side="sh", role="league_exploiter", objective="sh_win"),
        dict(name="anti_cor", side="fp", role="anti", objective="fp_win", target="sh_cor"),
        dict(name="anti_mil", side="sh", role="anti", objective="sh_win", target="fp_mil"),
    ],
    # the required pilot: one FP and one SH learner, pure win, mirror self-play (the race setup)
    "pilot": [
        dict(name="fp_win", side="fp", role="main", objective="fp_win"),
        dict(name="sh_win", side="sh", role="main", objective="sh_win"),
    ],
}


def _side_tokens(key):
    k = str(key).lower()
    parts = k.replace("-", "_").split("_")
    return any(p in ("fp", "sh", "shadow", "free", "freepeoples") for p in parts)


def check_config(cfg=None):
    """Refuse a configuration with any side-specific key (the USER's identical-training rule)."""
    cfg = CONFIG if cfg is None else cfg
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if _side_tokens(k):
                    bad.append(path + "/" + str(k))
                walk(v, path + "/" + str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + "[%d]" % i)
    walk(cfg, "")
    if bad:
        raise ValueError("league config carries side-specific keys (forbidden: FP and SH train identically): %s" % bad)
    return True


def check_agents(agents):
    """Mirrored structure: every role has as many FP as SH agents; objectives match the agent's side."""
    count = {}
    for a in agents:
        if OBJECTIVES[a["objective"]]["side"] != a["side"]:
            raise ValueError("agent %s plays %s but its objective %s is a %s objective"
                             % (a["name"], a["side"], a["objective"], OBJECTIVES[a["objective"]]["side"]))
        count.setdefault(a["role"], {"fp": 0, "sh": 0})[a["side"]] += 1
    for role, c in count.items():
        if c["fp"] != c["sh"]:
            raise ValueError("league structure not mirrored: role %s has %d FP and %d SH agents" % (role, c["fp"], c["sh"]))
    names = [a["name"] for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("duplicate agent names")
    for a in agents:
        if a.get("target") and a["target"] not in names:
            raise ValueError("agent %s targets unknown agent %s" % (a["name"], a["target"]))
    return count


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hashed_inputs():
    """Everything the hash pins (review S2-1): the training config AND the reward tables, the agent set (objectives,
    roles, targets), the BC files and the search profiles."""
    return dict(config=CONFIG, objectives=OBJECTIVES, agents=AGENTS, bc_files=BC_FILES, search_profile=SEARCH_PROFILE,
                routes=list(ROUTES))


def config_hash(cfg=None):
    """sha256 of the canonical JSON of hashed_inputs() (cfg: a CONFIG to hash in place of the current one)."""
    body = hashed_inputs()
    if cfg is not None:
        body = dict(body, config=cfg)
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def reward(objective, winner, route, status="terminated"):
    """The agent's scalar for one finished game (winner 'fp' / 'sh' / None, route one of ROUTES)."""
    table = OBJECTIVES[objective]
    if status != "terminated" or winner not in SIDES:
        return float(table["timeout"])
    if winner == table["side"]:
        if route not in table["win"]:
            raise ValueError("objective %s: unknown win route %r" % (objective, route))
        return float(table["win"][route])
    return float(table["loss"])


def anchor_beta(learning_games):
    a = CONFIG["train"]["anchor"]
    frac = max(0.0, 1.0 - float(learning_games) / float(a["decay_learning_games"]))
    return max(float(a["floor"]), float(a["beta0"]) * frac + float(a["floor"]) * (1.0 - frac))


def holdout(seed, fraction=None):
    """Whole-game held-out split, the same for every agent and side: a stable hash of the seed."""
    fraction = CONFIG["train"]["holdout_fraction"] if fraction is None else fraction
    h = int(hashlib.sha256(b"league-holdout-%d" % int(seed)).hexdigest()[:8], 16)
    return (h % 10000) < int(round(fraction * 10000))


def keep_decision(seed, ply, agent, cycle, q):
    """Deterministic uniform decision subsample (max_train_decisions) - same rule for every agent."""
    if q >= 1.0:
        return True
    h = int(hashlib.sha256(("%d|%d|%s|%d" % (seed, ply, agent, cycle)).encode()).hexdigest()[:8], 16)
    return (h / float(1 << 32)) < q


# ------------------------------------------------------------------------------------------------------------ helpers
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, body):
    atomic_write_text(path, json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_root():
    """The code tree: $WOTR_TREE, else the repo this folder sits in, else ../tree (cluster: $AW/{league,tree})."""
    env = os.environ.get("WOTR_TREE", "").strip()
    cands = [Path(env)] if env else [HERE.parents[2], HERE.parent / "tree"]
    for c in cands:
        if (c / "selfplay_actor_critic").is_dir() and (c / B / "native_w2").is_dir():
            return Path(os.path.abspath(c))
    raise SystemExit("league: no code tree at %s; set WOTR_TREE" % ", ".join(map(str, cands)))


def gen_dir():
    """hpc/remote/gen (batchgen's planner and the lockstep machinery): beside this folder."""
    return HERE.parent / "gen"


def search_dir():
    return HERE.parent / "search"


def setup_paths(tree=None):
    tree = Path(tree) if tree else tree_root()
    for p in (str(tree / "src"), str(tree / B / "native_w2"), str(tree), str(gen_dir()), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return tree


FLUX_DROP = ("OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "LD_AUDIT", "SPINDLE")


def clean_env(env=None, threads=1, gpu=None):
    """A child environment free of flux run's mpibind / Spindle settings (see hpc/README.md, both race drivers)."""
    env = dict(os.environ if env is None else env)
    for k in list(env):
        if k in FLUX_DROP or k.startswith("LDCS_"):
            env.pop(k, None)
    env["OMP_PROC_BIND"] = "false"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu != "keep":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


class Log:
    def __init__(self, path, t0=None):
        self.path = Path(path)
        self.t0 = time.time() if t0 is None else t0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, text):
        line = "%s +%6.1fm  %s" % (time.strftime("%H:%M:%S"), (time.time() - self.t0) / 60.0, text)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def deep(obj):
    return copy.deepcopy(obj)


check_config()
for _mode, _agents in AGENTS.items():
    check_agents(_agents)


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="export the shared league config (one block for every agent) and its hash")
    _sub = _ap.add_subparsers(dest="cmd", required=True)
    _ex = _sub.add_parser("export")
    _ex.add_argument("--out", default="")
    _a = _ap.parse_args()
    _body = dict(hashed_inputs(), config_hash=config_hash(), search_mismatches=search_mismatches())
    if _a.out:
        write_json(_a.out, _body)
    print(json.dumps(dict(config_hash=_body["config_hash"]), indent=1) if _a.out else json.dumps(_body, indent=1))
