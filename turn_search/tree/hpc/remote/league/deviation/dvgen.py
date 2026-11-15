"""Forced-deviation generation: plans, node runs, checks, status.  Resumable; nothing here submits a job.

  awpy dvgen.py pincheck
  awpy dvgen.py plan-scan --out D --games N --fp-ckpt F --sh-ckpt F [--seed0 S] [--tag-every 10]    exit 10 = nothing left
  awpy dvgen.py plan-roll --out D --stage 1|2 ...                                                     exit 10 = nothing left
  awpy dvgen.py run --out D --stage scan|roll1|roll2 --node K --nodes M --deadline EPOCH <batchgen worker options>
  awpy dvgen.py check --out D --stage scan|roll1|roll2                                                exit 3 = abort the job
  awpy dvgen.py status --out D --job J --resubmit "<command>"

scan   D/scan/shards: base games + forks + candidate positions.  The seeds still to play are all those not in a complete shard, so a
       deadline or a killed job loses only what was in flight.
roll1  D/roll1/positions.json is chosen ONCE from the scan's candidates; chunks of rollouts are the unit of resume.
roll2  D/roll2/positions.json is chosen ONCE from roll1's results; fresh rollouts (other k), so its numbers are unbiased estimates of
       the rule that selected them.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dvcommon as D  # noqa: E402
V, C = D.V, D.C

REFUSAL_LIMIT = 0.005
PANEL_FP_SCORE = 0.462            # fp_win@c200 vs sh_win@c200 at 1 sim, 500 games, 95% .419-.506 (hpc/results/t0panel_20261102/FINAL.md)
CHUNK1, CHUNK2 = 16, 64           # rollouts per chunk: a chunk replays its position once, so this is what a replay is spread over
HANDICAP_MIN = 1000               # handicap forks before its check counts (its effect is about -12 +- 6 on 107 local forks: a pilot or canary only reports it)
AA_SHARE = 0.05                   # share of rollout positions that also get an arm identical to the top (must match it exactly)


def pin_refusal():
    C.setup_paths()
    V.allow_search_patches()
    import native_valuedata
    return native_valuedata.pin_refusal()


def pincheck(_args):
    reason = pin_refusal()
    print("native pin: %s" % ("OK" if reason is None else "REFUSED: %s" % reason), flush=True)
    return 0 if reason is None else 2


# -------------------------------------------------------------------------------------------------------------- scan
def scan_dir(out):
    return Path(out) / "scan"


def recorded(paths, key="seeds", procs=0):
    if procs >= 1 and len(paths) > 8:
        with multiprocessing.get_context("spawn").Pool(procs) as pool:
            metas = pool.map(D.read_meta, paths, chunksize=16)
    else:
        metas = [D.read_meta(p) for p in paths]
    return [x for m in metas for x in m[key]]


def refused_seeds(folder):
    seeds = set()
    for p in sorted(Path(folder).glob("refused_node*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seeds.add(int(json.loads(line)["seed"]))
    return seeds


def plan_scan(args):
    folder = scan_dir(args.out)
    (folder / "shards").mkdir(parents=True, exist_ok=True)
    freq = json.loads(Path(args.census).read_text(encoding="utf-8"))
    wanted = dict(games=int(args.games), seed0=int(args.seed0), temperature=0.0, max_turns=int(C.CONFIG["play"]["max_turns"]),
                  fp_ckpt=str(D.absolute(args.fp_ckpt)), sh_ckpt=str(D.absolute(args.sh_ckpt)), view_keep=float(args.view_keep), tag_every=int(args.tag_every),
                  hazards=D.hazards(freq, json.loads(args.rates) if getattr(args, "rates", "") else None), census_sha256=C.sha_file(args.census), tag_rates=dict(D.TAG_RATES))
    for side in D.SIDES:
        wanted["%s_seed_sha256" % side] = C.sha_file(Path(wanted["%s_ckpt" % side]) / ("%s_seed.pt" % side))
    have = C.read_json(folder / "config.json")
    if have is None:
        C.write_json(folder / "config.json", wanted)
    else:
        if dict(have, games=wanted["games"]) != json.loads(json.dumps(wanted)) or wanted["games"] < have["games"]:
            raise SystemExit("config mismatch in %s (only --games may grow): have %s, want %s" % (folder, have, wanted))
        if wanted["games"] != have["games"]:
            C.write_json(folder / "config.json", wanted)
    seeds = np.array(recorded(D.shard_paths(folder / "shards"), procs=args.procs), np.int64)
    if len(np.unique(seeds)) != len(seeds):
        raise SystemExit("%d duplicate seeds in the shards" % (len(seeds) - len(np.unique(seeds))))
    every = np.arange(wanted["seed0"], wanted["seed0"] + wanted["games"], dtype=np.int64)
    refused = refused_seeds(folder)
    pending = every[~np.isin(every, seeds)]
    pending = pending[~np.isin(pending, np.fromiter(refused, np.int64, len(refused)))]
    tmp = folder / ("pending.tmp%d.npy" % os.getpid())
    np.save(tmp, pending)
    os.replace(tmp, folder / "pending.npy")
    print("%s plan-scan: %d games recorded, %d refused, %d of %d pending" % (C.now(), len(seeds), len(refused), len(pending), wanted["games"]), flush=True)
    return 10 if not len(pending) else 0


# ------------------------------------------------------------------------------------------------------------- rollouts
def read_candidates(out, procs=0):
    paths = D.shard_paths(scan_dir(out) / "shards")
    return recorded(paths, key="candidates", procs=procs)


def choose_positions(cands, n_general, n_named, seed=20261102):
    """Stratified without looking at anything later than the decision itself.  General: equal counts per (side, top band), uniform
    within, so a weighted share over decisions needs only the strata sizes, which are kept.  Named: up to n_named per (tag, side)."""
    r = np.random.default_rng(seed)
    chosen, strata = [], {}
    general = [c for c in cands if c["general"]]
    keys = sorted({(c["side"], c["top_band"]) for c in general})
    for key in keys:
        pool = [c for c in general if (c["side"], c["top_band"]) == key]
        strata["%s/%d" % key] = len(pool)
        take = r.permutation(len(pool))[:max(1, n_general // max(1, len(keys)))]
        chosen += [dict(pool[int(i)], stratum="%s/%d" % key, sample="general") for i in take]
    used = {(c["seed"], c["ply"]) for c in chosen}
    named_rows = {}
    for tag, name in D.TAG_NAMES.items():
        for side in D.SIDES:
            pool = [c for c in cands if c["tags"] & tag and c["side"] == side]
            named_rows["%s/%s" % (name, side)] = len(pool)
            for i in r.permutation(len(pool))[:n_named]:
                c = pool[int(i)]
                if (c["seed"], c["ply"]) in used:
                    for x in chosen:
                        if (x["seed"], x["ply"]) == (c["seed"], c["ply"]):
                            x["sample"] = x["sample"] if "named:" in x["sample"] else x["sample"] + "+named:" + name
                    continue
                used.add((c["seed"], c["ply"]))
                chosen.append(dict(c, stratum="named", sample="named:" + name))
    for pid, c in enumerate(chosen):
        c["pid"] = pid
    return chosen, dict(strata=strata, named_pool=named_rows)


def plan_roll(args):
    stage = int(args.stage)
    folder = Path(args.out) / ("roll%d" % stage)
    (folder / "shards").mkdir(parents=True, exist_ok=True)
    pos_file = folder / "positions.json"
    if not pos_file.is_file():
        if stage == 1:
            cands = read_candidates(args.out, args.procs)
            positions, info = choose_positions(cands, args.n_general, args.n_named)
            r = np.random.default_rng(777)
            for c in positions:
                arms = [dict(a, arm=i) for i, a in enumerate(c["arms"])]
                arms.append(dict(arm=len(arms), row=c["top"], source="top", p=c["p_top"], kind=c["top_kind"], label=c["top_label"]))
                if r.random() < AA_SHARE:
                    arms.append(dict(arm=len(arms), row=c["top"], source="aa", p=c["p_top"], kind=c["top_kind"], label=c["top_label"]))
                c["arms"] = arms
            body = dict(stage=1, k=int(args.k1), k0=0, chunk=CHUNK1, positions=positions, info=info, candidates=len(cands))
        else:
            positions = choose_promoted(args.out, args.k_promote, args.max_promoted, args.random_share, args.procs)
            body = dict(stage=2, k=int(args.k2), k0=1000000, chunk=CHUNK2, positions=positions)
        C.write_json(pos_file, body)
    body = json.loads(pos_file.read_text(encoding="utf-8"))
    done = set(recorded(D.shard_paths(folder / "shards"), key="chunks", procs=args.procs))
    done = {x.split("!")[0] for x in done}
    items = []
    for c in body["positions"]:
        for a in range(0, int(body["k"]), int(body["chunk"])):
            chunk = "%d:%d" % (c["pid"], a)
            if chunk in done:
                continue
            items.append(dict(id=chunk, stage=stage, pid=c["pid"], seed=c["seed"], ply=c["ply"], pos=c["pos"], rows=c["rows"], top=c["top"], side=c["side"],
                              k0=int(body["k0"]) + a, k1=int(body["k0"]) + min(a + int(body["chunk"]), int(body["k"])),
                              arms=[dict(arm=x["arm"], row=x["row"]) for x in c["arms"]]))
    C.write_json(folder / "pending.json", items)
    n_total = len(body["positions"]) * int(np.ceil(int(body["k"]) / int(body["chunk"])))
    print("%s plan-roll %d: %d positions, %d chunks of %d left; %.0f arm-rollouts left" % (
        C.now(), stage, len(body["positions"]), len(items), n_total, sum((i["k1"] - i["k0"]) * len(i["arms"]) for i in items)), flush=True)
    return 10 if not items else 0


def read_rolls(folder, procs=0):
    """{pid: {arm: {k: score}}} (the mover's score) of complete rollouts, and the positions body."""
    folder = Path(folder)
    body = json.loads((folder / "positions.json").read_text(encoding="utf-8"))
    sides = {c["pid"]: D.SIDE_CODE[c["side"]] for c in body["positions"]}
    rows = []
    for p in D.shard_paths(folder / "shards"):
        rows.append(D.read_table(p, "rolls/", D.ROLL_FIELDS))
    if rows:
        all_rows = {k: np.concatenate([r[k] for r in rows]) for k in D.ROLL_FIELDS}
    else:
        all_rows = {k: np.zeros(0, t) for k, t in D.ROLL_FIELDS.items()}
    return body, all_rows, sides


def pair_table(all_rows, sides):
    """{pid: {arm: (k array, score array)}}, refused rollouts left out."""
    out = {}
    ok = all_rows["status"] == D.OK
    pid, arm, k = all_rows["pid"][ok], all_rows["arm"][ok], all_rows["k"][ok]
    win = all_rows["winner"][ok]
    side = np.array([sides[int(x)] for x in pid], np.int64) if len(pid) else np.zeros(0, np.int64)
    s = np.where(win < 0, 0.5, (win == side).astype(np.float64))
    order = np.lexsort((k, arm, pid))
    pid, arm, k, s = pid[order], arm[order], k[order], s[order]
    if not len(pid):
        return out
    starts = np.flatnonzero(np.r_[True, (pid[1:] != pid[:-1]) | (arm[1:] != arm[:-1])])
    ends = np.r_[starts[1:], len(pid)]
    for a, b in zip(starts, ends):
        out.setdefault(int(pid[a]), {})[int(arm[a])] = (k[a:b], s[a:b])
    return out


def paired_diff(table, pid, arm, top_arm):
    """Means and standard errors of alt - top over the rollouts k both finished."""
    ka, sa = table[pid][arm]
    kt, st = table[pid][top_arm]
    _common, ia, it = np.intersect1d(ka, kt, assume_unique=True, return_indices=True)
    d = sa[ia] - st[it]
    return d, (float(d.mean()) if len(d) else float("nan")), (float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf"))


def choose_promoted(out, z_promote, max_promoted, random_share, procs=0):
    """Stage 1 -> stage 2: per position the alternative with the largest stage-1 z, kept when z >= z_promote; the largest z if there
    are too many.  random_share of the slots go to a uniformly chosen alternative of a uniformly chosen position instead: the check
    that selection did something (an unselected comparison at the same K)."""
    body, rows, sides = read_rolls(Path(out) / "roll1", procs)
    table = pair_table(rows, sides)
    r = np.random.default_rng(4242)
    candidates, by_pid = [], {c["pid"]: c for c in body["positions"]}
    for pid, arms in table.items():
        c = by_pid[pid]
        top_arm = next((a["arm"] for a in c["arms"] if a["source"] == "top"), None)
        if top_arm is None or top_arm not in arms:
            continue
        best = None
        for a in c["arms"]:
            if a["source"] in ("top", "aa") or a["arm"] not in arms:
                continue
            _d, m, se = paired_diff(table, pid, a["arm"], top_arm)
            z = m / se if se > 0 else 0.0
            if best is None or z > best[0]:
                best = (z, m, a["arm"])
        if best is not None:
            candidates.append((best[0], best[1], pid, best[2], top_arm))
    selected = sorted([x for x in candidates if x[0] >= z_promote], key=lambda x: -x[0])
    n_random = int(round(max_promoted * random_share))
    selected = selected[:max(0, max_promoted - n_random)]
    positions, new_pid = [], 0
    for z, m, pid, arm, top_arm in selected:
        c = by_pid[pid]
        positions.append(dict(c, source_pid=pid, pid=new_pid, sample2="selected", stage1_z=z, stage1_mean=m,
                              arms=[dict(next(a for a in c["arms"] if a["arm"] == arm), arm=0), dict(next(a for a in c["arms"] if a["arm"] == top_arm), arm=1)]))
        new_pid += 1
    pool = [c for c in body["positions"] if c["sample"].startswith("general") and c["pid"] in table]
    for i in r.permutation(len(pool))[:n_random]:
        c = pool[int(i)]
        alts = [a for a in c["arms"] if a["source"] not in ("top", "aa")]
        tops = [a for a in c["arms"] if a["source"] == "top"]
        if not alts or not tops:
            continue
        alt = alts[int(r.integers(len(alts)))]
        positions.append(dict(c, source_pid=c["pid"], pid=new_pid, sample2="random", stage1_z=None, stage1_mean=None,
                              arms=[dict(alt, arm=0), dict(tops[0], arm=1)]))
        new_pid += 1
    print("promotion: %d positions with a best alternative, %d at z >= %.2f, %d selected + %d random" % (
        len(candidates), sum(1 for x in candidates if x[0] >= z_promote), z_promote, len(selected), n_random), flush=True)
    return positions


# ---------------------------------------------------------------------------------------------------------- a node
def job_for(config, folder, mode, node, session, items):
    return dict(job="dv_%s_%s" % (mode, session), cycle=0, node=node, session=session, mode=mode, items=items,
                ckpts=dict(fp=[config["fp_ckpt"], "fp"], sh=[config["sh_ckpt"], "sh"]), out=str(Path(folder) / "shards"), temperature=0.0,
                max_turns=int(config["max_turns"]), hazards=config.get("hazards"), view_keep=config.get("view_keep", 0.3),
                tag_every=config.get("tag_every", 10), tag_rates=config.get("tag_rates") or {}, view_ks=16, flush_mb=96, flush_games=128,
                gradient=False, stats_module=None)


def run(args):
    import batchgen
    import dvworker
    import lgnode
    import lgworker
    lgworker.serve_main = dvworker.serve_main
    out = Path(args.out).resolve()
    stage = args.stage
    config = json.loads((scan_dir(out) / "config.json").read_text(encoding="utf-8"))
    folder = out / stage
    if stage == "scan":
        pending = np.load(folder / "pending.npy")
        mine = [dict(seed=int(s)) for s in pending[args.node::args.nodes]]
    else:
        items = json.loads((folder / "pending.json").read_text(encoding="utf-8"))
        mine = items[args.node::args.nodes]
    session = "%s%03d" % (time.strftime("%m%d%H%M%S"), os.getpid() % 1000)
    report_file = folder / ("run_node%d.json" % args.node)
    status = dict(node=args.node, host=platform.node(), session=session, share=len(mine), played=0, refusals=0, status="complete", wall_s=0.0)
    print("%s dvgen %s node %d/%d: %d items" % (C.now(), stage, args.node, args.nodes, len(mine)), flush=True)
    if not mine:
        C.write_json(report_file, status)
        return 0
    opts = batchgen.own_parser().parse_args(args.worker_options)
    tree = C.setup_paths()
    os.environ["WOTR_TREE"] = str(tree)
    reason = pin_refusal()
    if reason is not None:
        raise SystemExit("native pin refused (%s)" % reason)
    from selfplay_actor_critic.generation import require_generation
    generation = require_generation("dvgen")
    pool = lgnode.Pool(batchgen.make_plan(opts)["specs"], dict(tree=str(tree), generation=generation, actor_cache=4))
    abandon = out / (".abandon_%s_n%d" % (session, args.node))
    if args.deadline:
        def timer():
            while time.time() < args.deadline:
                time.sleep(1.0)
            abandon.write_text("deadline\n")
        threading.Thread(target=timer, daemon=True).start()
    t0 = time.time()
    try:
        shards, refusals, reports, errors, abandoned = pool.run_job(job_for(config, folder, "scan" if stage == "scan" else "roll", args.node, session, mine), abandon)
    finally:
        pool.close()
        if abandon.is_file():
            abandon.unlink()
    for r in refusals:
        C.append_jsonl(folder / ("refused_node%d.jsonl" % args.node), r)
    played, wall = sum(len(s["games"]) for s in shards), time.time() - t0
    counts = {}
    for rep in reports.values():
        for key, value in (rep.get("counts") or {}).items():
            counts[key] = counts.get(key, 0) + value
    status.update(played=played, refusals=len(refusals), wall_s=round(wall, 1), counts=counts, setup_s=round(pool.setup_seconds, 1),
                  decisions_per_s=round(counts.get("decisions", 0) / wall, 1) if wall > 0 else None,
                  bytes=sum(s["bytes"] for s in shards), status="error" if errors else ("stopped" if abandoned else "complete"),
                  errors={str(k): str(v)[-3000:] for k, v in errors.items()},
                  worker_seconds={k: round(sum((r.get("seconds") or {}).get(k, 0.0) for r in reports.values()), 1)
                                  for k in ("engine", "forward", "record", "write", "load")})
    with open(folder / "run_history.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(status, sort_keys=True, default=str) + "\n")
    C.write_json(report_file, status)
    print("%s dvgen %s node %d: %s, %d items in %.0fs, %.0f decisions/s, %d refusals" % (
        C.now(), stage, args.node, status["status"], played, wall, status["decisions_per_s"] or 0, len(refusals)), flush=True)
    return 1 if errors else (75 if abandoned else 0)


# ------------------------------------------------------------------------------------------------------------- checks
def load_scan_tables(out, kinds=False):
    """(games, forks), with forks["kind_top"] / ["kind_alt"] translated to one table (forks["kind_names"]) when kinds=True (the
    codes in a shard index that shard's own table)."""
    paths = D.shard_paths(scan_dir(out) / "shards")
    parts = [(D.read_table(p, "games/", D.GAME_FIELDS), D.read_table(p, "forks/", D.FORK_FIELDS)) for p in paths]
    names = {}
    if kinds:
        for p, (_g, f) in zip(paths, parts):
            table = np.array([names.setdefault(n, len(names)) for n in D.read_meta(p)["kinds"]] or [0], np.uint16)
            if len(f["seed"]):
                f["kind_top"], f["kind_alt"] = table[f["kind_top"]], table[f["kind_alt"]]
    games = {k: np.concatenate([g[k] for g, _f in parts]) if parts else np.zeros(0, t) for k, t in D.GAME_FIELDS.items()}
    forks = {k: np.concatenate([f[k] for _g, f in parts]) if parts else np.zeros(0, t) for k, t in D.FORK_FIELDS.items()}
    if kinds:
        forks["kind_names"] = sorted(names, key=names.get)
    return games, forks


def attach_base(games, forks):
    """Each fork's baseline result (the game that went on), as arrays aligned with the forks."""
    order = np.argsort(games["seed"])
    seeds = games["seed"][order]
    at = np.searchsorted(seeds, forks["seed"])
    if len(forks["seed"]) and (at.max() >= len(seeds) or np.any(seeds[at] != forks["seed"])):
        raise ValueError("a fork without its base game")
    return {k: games[k][order][at] for k in ("winner", "route", "turns", "plies")}


def check_scan(out, min_games=500):
    """Every reason to stop before spending more.  Returns (failures, warnings, figures)."""
    games, forks = load_scan_tables(out)
    folder = scan_dir(out)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    fails, warns, fig = [], [], {}
    n = len(games["seed"])
    fig["games"] = int(n)
    if n < min_games:
        warns.append("only %d games: most checks are thin" % n)
    if not n:
        fails.append("no games at all")
        return fails, warns, fig
    if len(np.unique(games["seed"])) != n:
        fails.append("duplicate seeds")
    refused = len(refused_seeds(folder))
    fig["base_refused"] = refused
    if refused > REFUSAL_LIMIT * max(n + refused, 200):
        fails.append("%d base refusals of %d games" % (refused, n + refused))
    fp = float(D.score(games["winner"], 0).mean())
    se = float(np.sqrt(max(fp * (1 - fp), 0.01) / n))
    fig.update(fp_score=round(fp, 4), fp_score_se=round(se, 4), panel_fp_score=PANEL_FP_SCORE, mean_plies=round(float(games["plies"].mean()), 1),
               mean_turns=round(float(games["turns"].mean()), 2), forks_per_game=round(len(forks["seed"]) / n, 3),
               cap_share=round(float((games["winner"] < 0).mean()), 5))
    if abs(fp - PANEL_FP_SCORE) > max(0.06, 4 * se):           # the panel's own 95% is about +-.044 on 500 games: catches wrong play, not small gaps
        fails.append("T=0 baseline FP score %.3f is too far from the panel's %.3f (not the same play?)" % (fp, PANEL_FP_SCORE))
    if not 150 <= fig["mean_plies"] <= 400:
        fails.append("mean decisions per game %.0f (expected ~255)" % fig["mean_plies"])
    if not 6.5 <= fig["mean_turns"] <= 12:
        fails.append("mean game length %.1f turns (expected ~8.7)" % fig["mean_turns"])
    r = config["hazards"]["rates"]
    plan = 34 * r["band"] + 2 * D.N_GROUPS * r["kind"] + r["exact_aa"] + r["reseed_aa"] + r["handicap"]
    fig["forks_per_game_planned"] = round(plan, 3)
    if n >= min_games and not 0.4 * plan <= fig["forks_per_game"] <= 1.6 * plan:
        fails.append("%.2f forks per game, planned about %.2f" % (fig["forks_per_game"], plan))
    if not len(forks["seed"]):
        fails.append("no forks")
        return fails, warns, fig
    ok = forks["status"] == D.OK
    fig["fork_refused_share"] = round(float((~ok).mean()), 5)
    if (~ok).mean() > 0.01:
        fails.append("%.2f%% of the forks refused" % (100 * (~ok).mean()))
    if not np.all(np.isfinite(forks["p_top"])) or not np.all(np.isfinite(forks["p_alt"])):
        fails.append("non-finite priors")
    real = np.isin(forks["stream"], (D.BAND, D.KIND))
    if np.any(forks["alt"][real] == forks["top"][real]):
        fails.append("a forced row equal to the top row")
    if np.any(forks["p_alt"][real] > forks["p_top"][real] + 1e-5):
        fails.append("an alternative with more prior than the top")
    band_mask = forks["stream"] == D.BAND
    wrong = np.searchsorted(D.ALT_EDGES, forks["p_alt"][band_mask].astype(np.float64), side="right") - 1 != forks["alt_band"][band_mask]
    if wrong.mean() > 1e-3 if band_mask.any() else False:     # float32 storage may move a row on an edge; no more than that
        fails.append("alt_band disagrees with p_alt on %.3f%%" % (100 * wrong.mean()))
    if np.any(forks["rows"][real] < 2):
        fails.append("a fork at a one-row menu")
    if np.any(forks["ply"] >= attach_base(games, forks)["plies"]):
        fails.append("a fork at a ply the base game never reached")
    base = attach_base(games, forks)
    alt_s, base_s = D.score(forks["winner"], forks["side"]), D.score(base["winner"], forks["side"])
    d = alt_s - base_s
    # exact A/A: the copy with the top move must BE the base game from there on
    aa = (forks["stream"] == D.EXACT_AA) & ok
    if aa.sum():
        same = (forks["trace"][aa] == forks["base_trace"][aa]) & (forks["winner"][aa] == base["winner"][aa])
        fig.update(exact_aa=int(aa.sum()), exact_aa_identical=round(float(same.mean()), 5))
        if aa.sum() >= 20 and same.mean() < 0.98:
            fails.append("exact A/A: only %.2f%% of the copies reproduced their game" % (100 * same.mean()))
    else:
        warns.append("no exact A/A fork yet")
    # reseeded A/A: mean 0 up to noise; the SD is the instrument's noise without pairing
    ra = (forks["stream"] == D.RESEED_AA) & ok
    if ra.sum():
        m, sd = float(d[ra].mean()), float(d[ra].std())
        fig.update(reseed_aa=int(ra.sum()), reseed_aa_mean=round(m, 4), reseed_aa_sd=round(sd, 4))
        if ra.sum() >= 100 and abs(m) > 4.5 * sd / np.sqrt(ra.sum()):
            fails.append("reseeded A/A mean %+.3f on %d (the null is 0)" % (m, ra.sum()))
    # positive control: the handicap arm has to lose points or the instrument cannot see effects
    hc = (forks["stream"] == D.HANDICAP) & ok
    if hc.sum():
        m, sd = float(d[hc].mean()), float(d[hc].std())
        fig.update(handicap=int(hc.sum()), handicap_mean=round(m, 4), handicap_se=round(sd / np.sqrt(hc.sum()), 4))
        if hc.sum() >= HANDICAP_MIN and m + 3 * sd / np.sqrt(hc.sum()) >= 0:
            fails.append("the handicap control (%d worst moves in a row) gave %+.3f on %d: the instrument sees no effect" % (D.HANDICAP_MOVES, m, hc.sum()))
    return fails, warns, fig


def check_roll(out, stage, limit=0.02):
    folder = Path(out) / stage
    body, rows, sides = read_rolls(folder)
    fails, warns = [], []
    fig = dict(positions=len(body["positions"]))
    metas = [D.read_meta(p) for p in D.shard_paths(folder / "shards")]
    chunks = [x for m in metas for x in m["chunks"]]
    bad = [x for x in chunks if "!" in x]
    fig.update(chunks=len(chunks), chunks_lost=len(bad), rollouts=int(len(rows["pid"])))
    if chunks and len(bad) > limit * len(chunks):
        fails.append("%d of %d chunks lost (replay mismatch or refusal): %s" % (len(bad), len(chunks), bad[:5]))
    refused = float((rows["status"] != D.OK).mean()) if len(rows["pid"]) else 0.0
    fig["rollouts_refused_share"] = round(refused, 5)
    if refused > 0.01:
        fails.append("%.2f%% of the rollouts refused" % (100 * refused))
    # rollout A/A: the same k and the same row must be the same game, move for move (not merely the same result)
    trace = {(int(p), int(a), int(k)): int(t) for p, a, k, t, st in zip(rows["pid"], rows["arm"], rows["k"], rows["trace"], rows["status"]) if st == D.OK}
    same = total = 0
    for c in body["positions"]:
        aa = next((a["arm"] for a in c["arms"] if a["source"] == "aa"), None)
        top = next((a["arm"] for a in c["arms"] if a["source"] == "top"), None)
        if aa is None:
            continue
        for (p, a, k), t in trace.items():
            if p == c["pid"] and a == aa and (p, top, k) in trace:
                total += 1
                same += int(t == trace[(p, top, k)])
    if total:
        fig.update(aa_rollouts=total, aa_identical=round(same / total, 5))
        if total >= 50 and same / total < 0.98:
            fails.append("rollout A/A: %d of %d identical (the same k must give the same game)" % (same, total))
    elif stage == "roll1":
        warns.append("no rollout A/A arm finished yet")
    return fails, warns, fig


def check(args):
    fails, warns, fig = (check_scan(args.out) if args.stage == "scan" else check_roll(args.out, args.stage))
    body = dict(stage=args.stage, time=C.now(), ok=not fails, failures=fails, warnings=warns, figures=fig)
    C.write_json(Path(args.out) / ("CHECK_%s.json" % args.stage), body)
    print("CHECK %s" % json.dumps(body, sort_keys=True), flush=True)
    return 3 if fails else 0


def job_status(args):
    out = Path(args.out)
    body = dict(job=args.job, time=C.now(), resubmit=args.resubmit)
    stages = {}
    errors = {}
    for stage in ("scan", "roll1", "roll2"):
        folder = out / stage
        nodes = [C.read_json(p) for p in sorted(folder.glob("run_node*.json"))]
        node_errors = {n["node"]: n.get("errors") for n in nodes if n.get("status") == "error"}
        if node_errors:
            errors[stage] = node_errors
        stages[stage] = dict(shards=len(D.shard_paths(folder / "shards")) if (folder / "shards").is_dir() else 0, nodes=[n.get("status") for n in nodes], setup_s=[n.get("setup_s") for n in nodes][:2])
    checks = {s: C.read_json(out / ("CHECK_%s.json" % s)) for s in ("scan", "roll1", "roll2")}
    failed = {s: c["failures"] for s, c in checks.items() if c and not c["ok"]}
    done = (out / "report" / "report.md").is_file() and (out / "report" / "COMPLETE").is_file()
    body.update(stages=stages, checks={s: (c or {}).get("figures") for s, c in checks.items()}, errors=errors, failed_checks=failed)
    if errors or failed:
        body.update(status="error", next="read the errors above and logs before resubmitting: a failed check means the data is bad")
    elif done:
        body.update(status="done", next="fetch report/report.md")
    else:
        body.update(status="resumable", next="resubmit the same command; finished work is kept")
    C.write_json(out / "JOB_STATUS.json", body)
    print("JOB_STATUS " + json.dumps(body, sort_keys=True, default=str), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pincheck")
    p = sub.add_parser("plan-scan")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed0", type=int, default=300000000, help="above every seed the league, the searches and the value lab used")
    p.add_argument("--fp-ckpt", required=True)
    p.add_argument("--sh-ckpt", required=True)
    p.add_argument("--census", default=str(HERE / "census_freq.json"))
    p.add_argument("--rates", default="", help="JSON overriding dvcommon.RATES (aa.flux)")
    p.add_argument("--view-keep", type=float, default=0.3)
    p.add_argument("--tag-every", type=int, default=10)
    p.add_argument("--procs", type=int, default=32)
    p = sub.add_parser("plan-roll")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", type=int, required=True, choices=(1, 2))
    p.add_argument("--n-general", type=int, default=480)
    p.add_argument("--n-named", type=int, default=20)
    p.add_argument("--k1", type=int, default=64)
    p.add_argument("--k2", type=int, default=1024)
    p.add_argument("--k-promote", type=float, default=1.28)
    p.add_argument("--max-promoted", type=int, default=250)
    p.add_argument("--random-share", type=float, default=0.1)
    p.add_argument("--procs", type=int, default=32)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True, choices=("scan", "roll1", "roll2"))
    p.add_argument("--node", type=int, default=int(os.environ.get("FLUX_TASK_RANK", "0")))
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--deadline", type=float, default=0.0)
    p = sub.add_parser("check")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True, choices=("scan", "roll1", "roll2"))
    p = sub.add_parser("status")
    p.add_argument("--out", required=True)
    p.add_argument("--job", default="")
    p.add_argument("--resubmit", default="")
    args, rest = ap.parse_known_args()
    if hasattr(args, "out"):
        args.out = str(D.absolute(args.out))
    if args.cmd == "run":
        args.worker_options = rest
        try:
            return run(args)
        except BaseException:
            C.write_json(Path(args.out) / args.stage / ("run_node%d.json" % args.node),
                         dict(node=args.node, host=platform.node(), status="error", errors={"run": traceback.format_exc()[-3000:]}))
            raise
    if rest:
        ap.error("unknown arguments %s" % rest)
    return dict(pincheck=pincheck, status=job_status, check=check)[args.cmd](args) if args.cmd in ("pincheck", "status", "check") else (
        plan_scan(args) if args.cmd == "plan-scan" else plan_roll(args))


if __name__ == "__main__":
    raise SystemExit(main())
