"""League profiler: what does each main agent like to do?  Measured habits -> counter-exploiter specs.

ADDENDUM 2 (USER 2026-10-26): "Ideally we have some automated way of seeing what the net likes to do generally and can
design an exploiter opponent to counteract that".  This tool:
  1. PLAYS recorded raw games (T=0.5, the league's regime) of the matchups it is given: the mains' mirror (FP main vs
     SH main) and league references (the pretrained BC nets against each main).  It records every learned decision
     (side, turn, phase, action kind, the action's parameters) and per turn the public game state it needs
     (Fellowship progress / corruption / Mordor, settlements controlled by each side's enemy, victory points,
     political boxes), plus the end.  This is analysis only: recording reads the engine state after the fact and
     never feeds a player.
  2. MEASURES habits per side: victory routes and game length; Shadow capture targets (which Free Peoples
     settlements fall, how often, when) and attack targets; musters by region (both sides); eye allocation per turn;
     Mordor entry turn and final Mordor step; Fellowship moves, declarations, separations; Aragorn / Gandalf the White
     timing; event-card use; political advances; FP settlements lost (defence holes).
  3. COMPARES them with the human corpus (data/gameReports_with_passwords.csv, base game: 13,939 reports: victory
     routes, game length, the settlements captured in each game, final Mordor step, Aragorn turn, turn-1 eyes) and
     with the league references (the BC nets in the same matchups).
  4. FLAGS predictable habits (concentration well above the human / league reference) and EMITS exploiter specs:
     JSON files the generator (exploiter_gen.py) trains, each with its objective, base net, target main, evidence.
  The three types the user named are always emitted (FP military-only, Shadow corruption-only, FP hard-muster
  defender of the Shadow's measured targets), plus the mirrors and the generic exploiters, with priorities from the
  measured habits.

  (fpsp recipe env, WOTR_ADMIT_GENERATIONS=v52 if a v52 net plays)
  python -B profiler.py --fp <FP main folder> --sh <SH main folder> --out <dir> --mirror 512 --refs 192 --workers 4
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
import gzip
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fpsp_common as C  # noqa: E402

_STATE = {}
KEEP_PARAMS = ("to", "from", "region", "to_region", "card", "card_id", "nation", "eyes", "count", "companion", "name",
               "die_face", "face", "unit", "units", "target", "new_face", "dice", "choice", "option", "mode")


def norm(name):
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", text.lower())


def _init(generation):
    from selfplay_actor_critic.generation import pin_worker_generation
    pin_worker_generation(generation)


def _actors(ai, fp, sh):
    key = (fp, sh)
    if _STATE.get("key") != key:
        import torch
        torch.set_num_threads(1)
        from selfplay_actor_critic.offline_eval import _load_actor
        _STATE.update(key=key, actors={"fp": _load_actor(ai, fp, "fp"), "sh": _load_actor(ai, sh, "sh")})
    return _STATE["actors"]


def _compact(params):
    out = {}
    for k, v in (params or {}).items():
        if k in KEEP_PARAMS or "card" in k or "region" in k:
            text = v if isinstance(v, (int, float, bool)) else str(v)[:48]
            out[k] = text
    return out


def _snapshot(ai, state):
    gl = ai.game_logic
    f = state.fellowship
    try:
        vp = gl.calculate_victory_points(state)
    except Exception:
        vp = {}
    # state.enemy_controlled[S] = the ENEMY settlements side S controls (game_logic.calculate_victory_points), so the
    # Free Peoples' LOST settlements are enemy_controlled[SHADOW] and the Shadow's are enemy_controlled[FREE_PEOPLES]
    enemy = {}
    for side_enum, tag in ((ai.gs.Side.SHADOW, "fp"), (ai.gs.Side.FREE_PEOPLES, "sh")):
        try:
            enemy[tag] = sorted(norm(r) for r in state.enemy_controlled.get(side_enum, ()))
        except Exception:
            enemy[tag] = []
    try:
        political = {str(getattr(n, "name", n)): int(b) for n, b in state.political.boxes.items()}
    except Exception:
        political = {}
    return dict(turn=int(state.turn), progress=int(f.progress), corruption=int(f.corruption), hidden=bool(f.hidden),
                mordor=bool(f.mordor_entered), mordor_step=int(getattr(f, "mordor_step", 0)),
                region=norm(getattr(getattr(f, "region", None), "name", getattr(f, "region", ""))),
                companions=len(getattr(f, "companions", []) or []),
                vp_fp=int(vp.get("free_peoples", 0)), vp_sh=int(vp.get("shadow", 0)),
                fp_lost=enemy["fp"], sh_lost=enemy["sh"], political=political)


def play(task):
    seed, label, fp, sh, temperature = task
    from selfplay_actor_critic.generation import require_generation
    require_generation("profiler.play")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", WOTR_NATIVE="0")
    ai = _STATE.get("ai") or C.import_engine()
    _STATE["ai"] = ai
    actors = _actors(ai, fp, sh)
    from selfplay_actor_critic.environment import PythonEnvironment
    env = PythonEnvironment(ai, f"profile-{label}-{seed}", seed)
    actions, turns = [], []
    last_turn = None
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        while True:
            frame = env.prepare()
            if frame is None:
                break
            if env.state.turn != last_turn:
                last_turn = env.state.turn
                turns.append(_snapshot(ai, env.state))
            if len(frame.actions) == 1:
                env.apply(frame, 0)
                continue
            prediction = actors[frame.side].predict(frame.inputs)
            selected = prediction.sample(env.samplers[frame.side], temperature)
            a = frame.actions[selected]
            actions.append(dict(s=frame.side, t=int(frame.turn), ph=str(frame.phase),
                                k=str(getattr(a.action_type, "name", a.action_type)), p=_compact(a.params),
                                n=len(frame.actions)))
            env.apply(frame, selected)
    final = _snapshot(ai, env.state)
    end = env.end
    return dict(seed=seed, label=label, fp=fp, sh=sh, status=end.status, winner=end.winner, route=end.victory_route,
                turn=int(env.state.turn), actions=actions, turns=turns, final=final)


# ------------------------------------------------------------------------------------------------- human reference
def human_reference(path=C.ROOT / "data" / "gameReports_with_passwords.csv"):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")) if not r["expansions"].strip()]
    routes = collections.Counter()
    for r in rows:
        if r["victory"] == "Concession":
            routes[("fp" if r["side"] == "Free" else "sh") + ":concession"] += 1
        elif r["side"] == "Free":
            routes["fp:" + ("ring_destroyed" if r["victory"] == "Ring" else "military")] += 1
        else:
            routes["sh:" + ("ring_corruption" if r["victory"] == "Ring" else "military")] += 1
    captured = collections.Counter(norm(s) for r in rows for s in r["strongholds"].split(",") if s.strip())
    turns = [int(r["turns"]) for r in rows if r["turns"].strip().isdigit()]
    mordor = [int(r["mordor"]) for r in rows if r["mordor"].strip().isdigit()]
    aragorn = [int(r["aragorn_turn"]) for r in rows if r["aragorn_turn"].strip().isdigit()]
    eyes = [int(r["initial_eyes"]) for r in rows if r["initial_eyes"].strip().isdigit()]
    n = len(rows)
    return dict(games=n, routes={k: v / n for k, v in routes.items()},
                captured_share={k: v / n for k, v in captured.items()},
                turns_mean=float(np.mean(turns)), turns_sd=float(np.std(turns)),
                mordor_step_mean=float(np.mean(mordor)) if mordor else None,
                aragorn_crowned_share=len(aragorn) / n, aragorn_turn_mean=float(np.mean(aragorn)) if aragorn else None,
                turn1_eyes_mean=float(np.mean(eyes)) if eyes else None,
                source=str(path))


# ------------------------------------------------------------------------------------------------- measurements
def _share(counter, total):
    return {k: v / total for k, v in counter.most_common()} if total else {}


def _top(d, k):
    return dict(list(sorted(d.items(), key=lambda kv: -kv[1]))[:k])


def entropy_bits(counter):
    total = sum(counter.values())
    if not total:
        return 0.
    return float(-sum((v / total) * math.log2(v / total) for v in counter.values() if v))


def measure(games, settlements):
    """Habit statistics of one matchup's games (both sides)."""
    games = [g for g in games if g["status"] == "terminated"]
    n = len(games)
    if not n:
        return dict(games=0)
    fp_settlements = {norm(k) for k, _, owner, _ in settlements if owner == 0}
    sh_settlements = {norm(k) for k, _, owner, _ in settlements if owner == 1}
    routes = collections.Counter(f"{g['winner']}:{g['route']}" for g in games)
    out = dict(games=n, fp_win=sum(g["winner"] == "fp" for g in games) / n, routes=_share(routes, n),
               turns_mean=float(np.mean([g["turn"] for g in games])), turns_sd=float(np.std([g["turn"] for g in games])))
    # settlements that fell (any time), first turn
    fp_lost, sh_lost, fp_lost_turn = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for g in games:
        seen_fp, seen_sh = {}, set()
        for snap in g["turns"] + [g["final"]]:
            for r in snap["fp_lost"]:
                if r in fp_settlements and r not in seen_fp:
                    seen_fp[r] = snap["turn"]
            for r in snap["sh_lost"]:
                if r in sh_settlements:
                    seen_sh.add(r)
        for r, t in seen_fp.items():
            fp_lost[r] += 1
            fp_lost_turn[r].append(t)
        for r in seen_sh:
            sh_lost[r] += 1
    total_fp_captures = sum(fp_lost.values())
    out["sh_captures"] = dict(per_game=total_fp_captures / n, share_of_games=_share(fp_lost, n),
                              first_turn_mean={r: float(np.mean(v)) for r, v in fp_lost_turn.items()},
                              top4_share_of_captures=(sum(v for _, v in fp_lost.most_common(4)) / total_fp_captures)
                              if total_fp_captures else None,
                              entropy_bits=entropy_bits(fp_lost))
    out["fp_captures"] = dict(per_game=sum(sh_lost.values()) / n, share_of_games=_share(sh_lost, n))
    # per-side action habits
    for side in ("fp", "sh"):
        acts = [a for g in games for a in g["actions"] if a["s"] == side]
        kinds = collections.Counter(a["k"] for a in acts)
        attacks = collections.Counter(norm(a["p"].get("to", "")) for a in acts if a["k"] == "army_attack")
        musters = collections.Counter(norm(a["p"].get("region", "")) for a in acts if a["k"].startswith("muster_"))
        moves_to = collections.Counter(norm(a["p"].get("to", "")) for a in acts if a["k"] == "army_move")
        cards = collections.Counter(str(a["p"].get("card", a["p"].get("card_id", "?"))) for a in acts if a["k"] == "play_event")
        political = collections.Counter(str(a["p"].get("nation", "?")) for a in acts if a["k"] == "advance_political")
        first_attack = [min((a["t"] for a in g["actions"] if a["s"] == side and a["k"] == "army_attack"), default=None) for g in games]
        first_attack = [t for t in first_attack if t is not None]
        out[side] = dict(decisions_per_game=len(acts) / n, kinds_per_game={k: v / n for k, v in kinds.most_common(25)},
                         attack_targets=_share(attacks, sum(attacks.values())), attacks_per_game=sum(attacks.values()) / n,
                         first_attack_turn_mean=float(np.mean(first_attack)) if first_attack else None,
                         attack_entropy_bits=entropy_bits(attacks),
                         muster_regions=_share(musters, sum(musters.values())), musters_per_game=sum(musters.values()) / n,
                         muster_top4_share=(sum(v for _, v in musters.most_common(4)) / sum(musters.values())) if musters else None,
                         reinforce_regions=_top(_share(moves_to, sum(moves_to.values())), 12),
                         event_cards=_top(_share(cards, sum(cards.values())), 15), events_per_game=sum(cards.values()) / n,
                         political=_share(political, sum(political.values())))
    # Shadow eyes, Fellowship and FP timing
    eyes, eyes_t1 = [], []
    for g in games:
        for a in g["actions"]:
            if a["s"] == "sh" and a["k"] == "eye_allocate":
                e = a["p"].get("eyes", a["p"].get("count"))
                try:
                    e = int(e)
                except (TypeError, ValueError):
                    continue
                eyes.append(e)
                if a["t"] == 1:
                    eyes_t1.append(e)
    mordor_turn = [next((s["turn"] for s in g["turns"] if s["mordor"]), None) for g in games]
    entered = [t for t in mordor_turn if t is not None]
    crown = [min((a["t"] for a in g["actions"] if a["k"] == "crown_aragorn"), default=None) for g in games]
    white = [min((a["t"] for a in g["actions"] if a["k"] == "transform_gandalf"), default=None) for g in games]
    out["sh"].update(eyes_mean=float(np.mean(eyes)) if eyes else None, eyes_sd=float(np.std(eyes)) if eyes else None,
                     turn1_eyes_mean=float(np.mean(eyes_t1)) if eyes_t1 else None,
                     final_corruption_mean=float(np.mean([g["final"]["corruption"] for g in games])))
    out["fp"].update(mordor_entered_share=len(entered) / n,
                     mordor_entry_turn_mean=float(np.mean(entered)) if entered else None,
                     mordor_entry_turn_sd=float(np.std(entered)) if entered else None,
                     final_mordor_step_mean=float(np.mean([g["final"]["mordor_step"] for g in games])),
                     final_progress_mean=float(np.mean([g["final"]["progress"] for g in games])),
                     fellowship_moves_per_game=sum(1 for g in games for a in g["actions"] if a["s"] == "fp" and a["k"] == "fellowship_move") / n,
                     separations_per_game=sum(1 for g in games for a in g["actions"] if a["k"] == "companion_separate") / n,
                     aragorn_crowned_share=sum(1 for c in crown if c is not None) / n,
                     aragorn_turn_mean=float(np.mean([c for c in crown if c is not None])) if any(c is not None for c in crown) else None,
                     gandalf_white_share=sum(1 for w in white if w is not None) / n)
    return out


# ------------------------------------------------------------------------------------------------- habits -> specs
def flags_and_specs(stats, human, mains, settlements, *, top_targets=4):
    """Predictable habits of each main (vs the human corpus and the league references) -> exploiter specs."""
    mirror = stats["mirror"]
    flags, specs = [], []
    fp_names = {norm(k): k for k, _, owner, _ in settlements if owner == 0}

    def flag(side, habit, value, reference, note):
        flags.append(dict(main=side, habit=habit, value=value, reference=reference, note=note))

    # -------- the Shadow main's habits (countered by FP exploiters, which train the Shadow main) --------
    caps = mirror["sh_captures"]["share_of_games"]
    targets = [r for r, _ in sorted(caps.items(), key=lambda kv: -kv[1])[:top_targets]]
    human_caps = {r: human["captured_share"].get(r, 0.) for r in targets}
    ref = stats.get("fp_ref_vs_sh_main", {}).get("sh_captures", {}).get("share_of_games", {})
    flag("sh", "capture_targets", {r: round(caps[r], 3) for r in targets}, dict(human=human_caps, fp_bc_vs_sh_main={r: ref.get(r) for r in targets}),
         f"top-{top_targets} FP settlements the Shadow main takes; top-4 share of all its captures "
         f"{mirror['sh_captures']['top4_share_of_captures']}")
    sh_wins = {k: v for k, v in mirror["routes"].items() if k.startswith("sh:")}
    sh_total = sum(sh_wins.values()) or 1e-9
    sh_mil = sh_wins.get("sh:military", 0.) / sh_total
    human_sh_mil = human["routes"].get("sh:military", 0) / max(1e-9, human["routes"].get("sh:military", 0) + human["routes"].get("sh:ring_corruption", 0))
    flag("sh", "win_route_military_share", round(sh_mil, 3), round(human_sh_mil, 3), "share of the Shadow main's wins that are military")
    specs.append(dict(name="fp_defend_targets", side="fp", trains="sh", priority=1.0 + float(mirror["sh_captures"]["top4_share_of_captures"] or 0),
                      objective=dict(type="defend_targets", targets=[fp_names.get(t, t) for t in targets], weight=0.5),
                      prior_bias=dict(targets=[fp_names.get(t, t) for t in targets], kinds=["muster_recruit", "muster_elite", "army_move"], beta=1.5),
                      why="USER-named: hard-muster the Shadow main's habitual military targets so it must switch targets in-game",
                      evidence=dict(targets={t: caps[t] for t in targets}, human=human_caps)))
    specs.append(dict(name="fp_military_only", side="fp", trains="sh", priority=1.0 + sh_mil,
                      objective=dict(type="profile_graded", profile="military", kappa=0.7),
                      why="USER-named: an FP that wins only by military; punishes a Shadow that leaves its own strongholds thin "
                          f"(Shadow strongholds fell in {mirror['fp_captures']['per_game']:.2f} per game)",
                      evidence=dict(sh_military_win_share=sh_mil, fp_captures=mirror["fp_captures"])))
    specs.append(dict(name="fp_ring_only", side="fp", trains="sh", priority=0.5 + sh_mil,
                      objective=dict(type="profile_pure", profile="ring", off_mission=-1.0),
                      why="mirror type: an FP racing the Ring only; punishes a Shadow that spends its dice on armies and neglects the hunt",
                      evidence=dict(sh_military_win_share=sh_mil, sh_eyes_mean=mirror["sh"].get("eyes_mean"))))
    specs.append(dict(name="fp_generic", side="fp", trains="sh", priority=0.5, objective=dict(type="standard"),
                      why="generic main exploiter of the Shadow main"))
    # -------- the FP main's habits (countered by Shadow exploiters, which train the FP main) --------
    fp_wins = {k: v for k, v in mirror["routes"].items() if k.startswith("fp:")}
    fp_total = sum(fp_wins.values()) or 1e-9
    fp_ring = fp_wins.get("fp:ring_destroyed", 0.) / fp_total
    human_fp_ring = human["routes"].get("fp:ring_destroyed", 0) / max(1e-9, human["routes"].get("fp:ring_destroyed", 0) + human["routes"].get("fp:military", 0))
    flag("fp", "win_route_ring_share", round(fp_ring, 3), round(human_fp_ring, 3), "share of the FP main's wins by destroying the Ring")
    sd = mirror["fp"].get("mordor_entry_turn_sd")
    flag("fp", "mordor_entry_turn", dict(mean=mirror["fp"].get("mordor_entry_turn_mean"), sd=sd, entered=mirror["fp"].get("mordor_entered_share")),
         dict(human_final_mordor_step=human.get("mordor_step_mean")), "when the Fellowship enters Mordor (a predictable clock helps the hunt)")
    lost = mirror["sh_captures"]["share_of_games"]
    holes = [r for r, _ in sorted(lost.items(), key=lambda kv: -kv[1])[:top_targets]]
    flag("fp", "defence_holes", {r: round(lost[r], 3) for r in holes}, {r: human["captured_share"].get(r, 0.) for r in holes},
         "FP settlements most often lost (the same captures, read as the FP main's undefended places)")
    specs.append(dict(name="sh_corruption_only", side="sh", trains="fp", priority=1.0 + fp_ring + (0.5 if sd is not None and sd < 1.5 else 0.),
                      objective=dict(type="profile_graded", profile="corruption", kappa=0.7),
                      why="USER-named: a Shadow that wins only by corrupting the Ring-bearer; punishes a Ring-reliant, predictable Fellowship",
                      evidence=dict(fp_ring_win_share=fp_ring, human=human_fp_ring, mordor=mirror["fp"].get("mordor_entry_turn_mean"), mordor_sd=sd)))
    specs.append(dict(name="sh_military_only", side="sh", trains="fp", priority=0.8 + (1 - fp_ring),
                      objective=dict(type="profile_graded", profile="military", kappa=0.7),
                      why="mirror type: a Shadow that wins only by military; punishes the FP's defence holes",
                      evidence=dict(defence_holes={r: lost[r] for r in holes})))
    specs.append(dict(name="sh_generic", side="sh", trains="fp", priority=0.5, objective=dict(type="standard"),
                      why="generic main exploiter of the FP main"))
    for s in specs:
        s.update(base=C.EXPLOITER_BASE[s["side"]], target_main=mains[s["trains"]])
    return flags, sorted(specs, key=lambda s: (-s["priority"], s["name"]))


def markdown(report):
    h, m = report["human"], report["stats"]["mirror"]
    L = [f"# League profile ({report['created']})", "",
         f"Mains: FP `{report['mains']['fp']}`, SH `{report['mains']['sh']}`. Raw policies, T=0.5. Games per matchup: "
         + ", ".join(f"{k} {v.get('games', 0)}" for k, v in report["stats"].items()) + ".",
         f"Human reference: {h['games']} base-game reports ({Path(h['source']).name}).", "",
         "## Outcomes", "", "| | FP win | routes | turns |", "|---|---|---|---|"]
    for k, v in report["stats"].items():
        if v.get("games"):
            L.append(f"| {k} | {100 * v['fp_win']:.1f}% | " + ", ".join(f"{r} {100 * p:.0f}%" for r, p in v["routes"].items())
                     + f" | {v['turns_mean']:.1f} ± {v['turns_sd']:.1f} |")
    L.append(f"| humans | {100 * sum(p for r, p in h['routes'].items() if r.startswith('fp')):.1f}% | "
             + ", ".join(f"{r} {100 * p:.0f}%" for r, p in sorted(h["routes"].items(), key=lambda kv: -kv[1])) + f" | {h['turns_mean']:.1f} ± {h['turns_sd']:.1f} |")
    L += ["", "## Shadow main: settlements it takes (share of games; mirror vs FP BC reference vs humans)", "",
          "| settlement | mirror | FP BC vs SH main | humans | first-capture turn |", "|---|---|---|---|---|"]
    ref = report["stats"].get("fp_ref_vs_sh_main", {}).get("sh_captures", {}).get("share_of_games", {})
    for r, p in list(m["sh_captures"]["share_of_games"].items())[:10]:
        L.append(f"| {r} | {100 * p:.0f}% | {100 * ref.get(r, 0):.0f}% | {100 * h['captured_share'].get(r, 0):.0f}% | {m['sh_captures']['first_turn_mean'].get(r, float('nan')):.1f} |")
    L.append(f"\nCaptures per game {m['sh_captures']['per_game']:.2f}; top-4 share of captures {m['sh_captures']['top4_share_of_captures']}; "
             f"entropy {m['sh_captures']['entropy_bits']:.2f} bits.")
    for side, title in (("sh", "Shadow main"), ("fp", "Free Peoples main")):
        s = m[side]
        L += ["", f"## {title}: habits (mirror)", "",
              f"- decisions per game {s['decisions_per_game']:.0f}; attacks per game {s['attacks_per_game']:.1f} (first attack turn "
              f"{s['first_attack_turn_mean']}); attack targets: " + ", ".join(f"{r} {100 * p:.0f}%" for r, p in list(s["attack_targets"].items())[:6]),
              f"- musters per game {s['musters_per_game']:.1f}, top-4 share {s['muster_top4_share']}: " + ", ".join(f"{r} {100 * p:.0f}%" for r, p in list(s["muster_regions"].items())[:6]),
              f"- army moves into: " + ", ".join(f"{r} {100 * p:.0f}%" for r, p in list(s["reinforce_regions"].items())[:6]),
              f"- event cards per game {s['events_per_game']:.1f}: " + ", ".join(f"{c} {100 * p:.0f}%" for c, p in list(s["event_cards"].items())[:6]),
              f"- political advances: " + ", ".join(f"{c} {100 * p:.0f}%" for c, p in list(s["political"].items())[:6]),
              f"- top action kinds per game: " + ", ".join(f"{k} {v:.1f}" for k, v in list(s["kinds_per_game"].items())[:10])]
        if side == "sh":
            L.append(f"- eyes per allocation {s['eyes_mean']} (sd {s['eyes_sd']}), turn 1 {s['turn1_eyes_mean']} (humans {h['turn1_eyes_mean']:.2f}); final corruption {s['final_corruption_mean']:.1f}")
        else:
            L.append(f"- Mordor entered in {100 * s['mordor_entered_share']:.0f}% of games, turn {s['mordor_entry_turn_mean']} (sd {s['mordor_entry_turn_sd']}); final Mordor step "
                     f"{s['final_mordor_step_mean']:.2f} (humans {h['mordor_step_mean']:.2f}); Fellowship moves per game {s['fellowship_moves_per_game']:.1f}; separations {s['separations_per_game']:.1f}; "
                     f"Aragorn crowned {100 * s['aragorn_crowned_share']:.0f}% (humans {100 * h['aragorn_crowned_share']:.0f}%), Gandalf the White {100 * s['gandalf_white_share']:.0f}%")
    L += ["", "## Flags (predictable habits)", ""] + [f"- **{f['main']} {f['habit']}**: {f['value']} (reference {f['reference']}). {f['note']}" for f in report["flags"]]
    L += ["", "## Exploiter specs emitted (priority order)", "", "| spec | side | trains | objective | base | priority | why |", "|---|---|---|---|---|---|---|"]
    for s in report["specs"]:
        L.append(f"| {s['name']} | {s['side']} | {s['trains']} main | {s['objective_text']} | BC-{s['base']} | {s['priority']:.2f} | {s['why']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp", default=str(C.DEPLOYED_FP), help="FP main policy folder")
    ap.add_argument("--sh", default=str(C.DEPLOYED_SH), help="SH main policy folder")
    ap.add_argument("--fp-ref", default=str(C.FP_ARMS[C.EXPLOITER_BASE["fp"]]), help="league reference FP (BC)")
    ap.add_argument("--sh-ref", default=str(C.SH_ARMS[C.EXPLOITER_BASE["sh"]]), help="league reference SH (BC)")
    ap.add_argument("--mirror", type=int, default=512)
    ap.add_argument("--refs", type=int, default=192)
    ap.add_argument("--first-seed", type=int, default=C.PROFILE_FIRST_SEED)
    ap.add_argument("--temperature", type=float, default=.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queue", default="", help="also write the specs into this generator queue folder")
    ap.add_argument("--tag", default="", help="spec name suffix (e.g. the main's generation)")
    args = ap.parse_args()
    C.require_recipe_env("profiler")
    sys.path.insert(0, str(C.ROOT))
    from selfplay_actor_critic.generation import require_generation
    generation = require_generation("profiler.main")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def rel(p):
        try:
            return Path(p).resolve().relative_to(C.ROOT.resolve()).as_posix()
        except ValueError:
            return str(p)
    mains = dict(fp=rel(args.fp), sh=rel(args.sh))
    if max(args.mirror, args.refs) > 8_000:
        raise SystemExit("at most 8,000 games per matchup (the profiler seed block layout)")
    matchups = [("mirror", mains["fp"], mains["sh"], args.mirror, 0),
                ("fp_ref_vs_sh_main", rel(args.fp_ref), mains["sh"], args.refs, 8_000),
                ("fp_main_vs_sh_ref", mains["fp"], rel(args.sh_ref), args.refs, 16_000)]
    started = time.time()
    stats, all_games = {}, {}
    for label, fp, sh, n, offset in matchups:
        if n <= 0:
            continue
        path = out / f"games_{label}.jsonl.gz"
        if path.exists():
            games = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
            if len(games) != n or games[0]["fp"] != fp or games[0]["sh"] != sh:
                raise SystemExit(f"{path} holds another profile; use a new --out")
        else:
            tasks = [(args.first_seed + offset + i, label, fp, sh, args.temperature) for i in range(n)]
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init, initargs=(generation,)) as ex:
                games = list(ex.map(play, tasks, chunksize=2))
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as stream:
                for g in games:
                    stream.write(json.dumps(g, separators=(",", ":")) + "\n")
            os.replace(tmp, path)
            print(f"{C.stamp()} profile {label}: {n} games ({time.time() - started:.0f}s)", flush=True)
        all_games[label] = games
    ai = C.import_engine()
    import objectives as O
    settlements = O.settlement_table("fp")
    for label, games in all_games.items():
        stats[label] = measure(games, settlements)
    human = human_reference()
    flags, specs = flags_and_specs(stats, human, mains, settlements)
    for s in specs:
        s["objective_text"] = O.describe(s["objective"], s["side"])
        if args.tag:
            s["name"] = f"{s['name']}_{args.tag}"
    report = dict(schema="fpsp_profile_v1", created=C.stamp(), mains=mains, matchups=[m[:4] for m in matchups],
                  temperature=args.temperature, first_seed=args.first_seed, stats=stats, human=human, flags=flags,
                  specs=specs, seconds=time.time() - started, tool_sha256=C.sha256_file(__file__))
    C.write_json(out / "profile.json", report)
    (out / "PROFILE.md").write_text(markdown(report), encoding="utf-8")
    spec_dir = out / "specs"
    spec_dir.mkdir(exist_ok=True)
    for s in specs:
        C.write_json(spec_dir / f"{s['name']}.json", dict(s, profile=str((out / "profile.json").resolve())))
        if args.queue:
            q = Path(args.queue)
            q.mkdir(parents=True, exist_ok=True)
            if not (q / f"{s['name']}.json").exists():
                C.write_json(q / f"{s['name']}.json", dict(s, profile=str((out / "profile.json").resolve()), queued=C.stamp()))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
