"""Event credit traced back along the chain of decisions that produced the event.

User, 2026-10-05: "for anything that increases the value function, we trace the entire route that led to it
happening. So if an army conquers a stronghold, the decisions that receive credit are all decisions that led to
that army's formation + cards that were played and the in-combat decisions. Consider Barad-dur -> merging with
the Morannon army -> Woodland Realm and conquering it: all those decisions should receive credit."

The games on disk (valuedata.py runs) hold, for every decision of either side, the position as the mover sees
it. Between two consecutive stored positions the explicit value function (explicit_value.py) may move: a
settlement captured, corruption taken, the Fellowship advanced, a companion lost. Each such move is an EVENT
with a size (the change of the explicit value) and a kind (which feature moved).

Lineage. Each region's army of a side carries a set of decisions with weights: the decisions that put figures
there (army units, leaders, Nazgul and named characters alike). Figures are followed by their counts per region
between consecutive positions:
  * figures appear in a region with no region losing any (a muster, a card, a minion entering play): the
    decision taken at that position joins the region's lineage with the number of figures placed;
  * figures leave one region and arrive in another (a move, Nazgul flying in, an advance after a battle): the
    arriving figures bring the departing region's lineage with them (a share proportional to the figures moved)
    and the moving decision itself; a merge adds the two lineages;
  * figures vanish with nothing arriving (casualties): the lineage stays with the survivors.
Credit (mode "full", the default). A decision in the lineage of an army is credited with the event's value
change times the fraction of that army it handled (a muster of 4 of the 9 units: 4/9; the move of all 9: 1);
the decision that attacked and the side's in-combat decisions of the battle are credited in full. The losing
side's decisions that moved figures OUT of a lost region earlier are debited by the fraction of the garrison
they took away, its combat decisions of the defence in full. Hunt events (corruption, companions lost,
Fellowship progress) credit or debit the Shadow's eye allocation of the turn in full and the lineage of the
Shadow's figures in the Fellowship's region (they give the re-rolls). An event with no capture, no battle and
no hunt is credited to the side's own decision that was just applied (a card). Mode "split" normalises the
shares of an event to 1 over its responsible decisions instead.

  python -m selfplay_actor_critic.event_trace --run selfplay_actor_critic/runs/big_g5_g3_a --side sh --show 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from .explicit_value import FEATURES, configure, features_of, value_of  # noqa: E402
from .race_features import FP_NATIONS, SH_NATIONS, REGION_STRIDE, NATION_COUNT  # noqa: E402

REGIONS = 105
MY_LEAD, EN_LEAD, MY_CTRL, EN_CTRL, FS_HERE = 2, 5, 7, 8, 9
COMBAT_KINDS = {"combat_card_play", "combat_card_none", "casualty_allocate", "siege_press", "siege_no_press", "combat_continue",
                "combat_cease", "combat_effect_choice", "combat_retreat", "combat_no_retreat", "retreat_into_siege", "victor_advance",
                "victor_stay", "army_attack", "siege_trim"}
HUNT_FEATURES = ("corruption", "companions_in_fs", "fs_distance", "mordor_step")
MODES = ("full", "split", "return")       # return: no tracing, every own decision gets the whole later change of the value


def region_table(obs, side):
    """Per region from one observation row in the view of `side`: figures of the Shadow and of the Free Peoples
    (army units plus leaders, Nazgul and named characters with them), who controls it, and the Fellowship's
    region (its figure on the board, the last known position: where the hunt re-rolls come from)."""
    blocks = obs[:REGIONS * REGION_STRIDE].reshape(REGIONS, REGION_STRIDE)
    counts = blocks[:, 14:14 + 2 * NATION_COUNT].reshape(REGIONS, NATION_COUNT, 2)
    units = np.rint(counts[:, :, 0] * 15.) + np.rint(counts[:, :, 1] * 5.)
    sh = units[:, list(SH_NATIONS)].sum(1)
    fp = units[:, list(FP_NATIONS)].sum(1)
    lead_my, lead_en = np.rint(blocks[:, MY_LEAD] * 7.), np.rint(blocks[:, EN_LEAD] * 7.)
    mine, enemy = blocks[:, MY_CTRL] > .5, blocks[:, EN_CTRL] > .5
    if side == "sh":
        sh, fp, sh_ctrl, fp_ctrl = sh + lead_my, fp + lead_en, mine, enemy
    else:
        sh, fp, sh_ctrl, fp_ctrl = sh + lead_en, fp + lead_my, enemy, mine
    fs = int(np.argmax(blocks[:, FS_HERE])) if blocks[:, FS_HERE].max() > .5 else -1
    return sh, fp, sh_ctrl, fp_ctrl, fs


class Lineage:
    """Per region: {decision index: figures' worth of credit} for one side's army there, and the decisions that
    moved figures OUT of each region with the fraction of the garrison they took (for debiting a later loss)."""

    def __init__(self):
        self.carried = [dict() for _ in range(REGIONS)]
        self.left = [dict() for _ in range(REGIONS)]

    def advance(self, before, after, decision, moving_side_decision):
        """Follow the figures from `before` to `after` (counts per region); `decision` is the index of the
        decision taken between them (credited where figures appear), and whether it was this side's own."""
        delta = after - before
        losses = [(int(r), float(-delta[r])) for r in np.nonzero(delta < 0)[0]]
        gains = [(int(r), float(delta[r])) for r in np.nonzero(delta > 0)[0]]
        lost_total, gained_total = sum(v for _, v in losses), sum(v for _, v in gains)
        moved_total = min(lost_total, gained_total)
        for region, lost in losses:
            share = min(1., moved_total / lost_total) if lost_total else 0.     # the part that moved on (not casualties)
            pool = self.carried[region]
            fraction = lost / before[region] if before[region] > 0 else 0.
            taken = {d: w * fraction for d, w in pool.items()}
            for d in pool:
                pool[d] -= taken[d]
            if moving_side_decision:
                self.left[region][decision] = self.left[region].get(decision, 0.) + fraction
            for target, gained in gains:
                part = gained / gained_total
                for d, w in taken.items():
                    self.carried[target][d] = self.carried[target].get(d, 0.) + w * share * part
        for region, gained in gains:
            if moving_side_decision:
                self.carried[region][decision] = self.carried[region].get(decision, 0.) + gained
        for region in range(REGIONS):
            self.carried[region] = {d: w for d, w in self.carried[region].items() if w > 1e-9}

    def credit(self, region):
        """Shares normalised to 1 over the region's lineage."""
        pool = self.carried[region]
        total = sum(pool.values())
        return {d: w / total for d, w in pool.items()} if total > 0 else {}

    def weights(self, region, count):
        """Fraction of the army of `count` figures each decision handled (at most 1)."""
        return {d: min(1., w / max(float(count), 1.)) for d, w in self.carried[region].items()}

    def debit(self, region):
        """Shares normalised to 1 over the decisions that moved figures out."""
        pool = self.left[region]
        total = sum(pool.values())
        return {d: w / total for d, w in pool.items()} if total > 0 else {}

    def debit_weights(self, region):
        return {d: min(1., w) for d, w in self.left[region].items()}


def load_game(folder):
    with open(os.path.join(folder, "episode.json"), encoding="utf-8") as stream:
        meta = json.load(stream)
    with np.load(os.path.join(folder, "arrays.npz")) as d:
        obs = d["observations"].astype(np.float32)
        kinds = [str(k) for k in d["kind_names"]][:] if "kind_names" in d.files else None
        played = d["kind_played"] if "kind_played" in d.files else None
    rows = meta["rows"]
    kind = [kinds[int(k)] for k in played] if kinds is not None else ["?"] * len(rows)
    end = meta.get("end") or {}
    return dict(seed=meta["seed"], winner=end.get("winner"), route=end.get("victory_route"), status=end.get("status"),
                rows=rows, obs=obs, kind=kind)


def trace(game, model, side="sh", names=(), mode="full", delay_turns=0, lam=1.):
    """Per decision of `side`: traced credit (sum over events of weight x value change, in the Shadow's value for
    the Shadow, negated for the Free Peoples) and the event log [(decision index, label, size, {decision: weight})].
    delay_turns > 0 keeps only the hindsight part: a decision is credited for an event at least that many turns
    later (the immediate effects of a move are left out)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    configure(model)
    rows, obs, kinds = game["rows"], game["obs"], game["kind"]
    n = len(rows)
    sides = [r["side"] for r in rows]
    tables = [region_table(obs[t], sides[t]) for t in range(n)]
    feats = [features_of(obs[t:t + 1], sides[t]) for t in range(n)]
    values = np.array([value_of(model, f)[0] for f in feats])
    own = Lineage()                      # the side's figures
    sign = 1. if side == "sh" else -1.
    credit = np.zeros(n)
    events = []
    eye_of_turn = {}
    if mode == "return":
        # every own decision gets the later change of the value; lam < 1 discounts it by the turns elapsed
        increments = sign * np.diff(values)
        turns = np.array([r["turn"] for r in rows])
        for t in range(n):
            if sides[t] == side:
                later = increments[t:]
                credit[t] = float((later * lam ** (turns[t + 1:] - turns[t])).sum()) if lam < 1. else sign * (values[-1] - values[t])
        return credit, events

    def battle_chain(t):
        """This side's in-combat decisions of the battle running at t, back to its attack (if it attacked)."""
        chain, k = [], t
        while k >= 0 and (sides[k] != side or kinds[k] in COMBAT_KINDS):
            if sides[k] == side:
                chain.append(k)
                if kinds[k] == "army_attack":
                    break
            k -= 1
        return chain

    for t in range(n - 1):
        if sides[t] == side and kinds[t] == "eye_allocate":
            eye_of_turn[rows[t]["turn"]] = t
        sh0, fp0, shc0, fpc0, fs0 = tables[t]
        sh1, fp1, shc1, fpc1, _ = tables[t + 1]
        figures0, figures1 = (sh0, sh1) if side == "sh" else (fp0, fp1)
        own.advance(figures0, figures1, t, sides[t] == side)
        change = sign * (values[t + 1] - values[t])
        if abs(change) < 1e-9:
            continue
        a, b = feats[t], feats[t + 1]
        moved = [name for name, *_ in FEATURES if int(b[name][0]) != int(a[name][0]) and name != "turn"]
        if not moved:
            continue
        responsible = {}

        def add(d, w):
            responsible[d] = responsible.get(d, 0.) + w

        label = ",".join(moved)
        ctrl_own0, ctrl_own1 = (shc0, shc1) if side == "sh" else (fpc0, fpc1)
        ctrl_en0, ctrl_en1 = (fpc0, fpc1) if side == "sh" else (shc0, shc1)
        # a settlement changes hands when a side's "captured" flag is raised, or the other side's is cleared
        # (a home settlement taken back)
        captured = np.nonzero((ctrl_own1 & ~ctrl_own0) | (ctrl_en0 & ~ctrl_en1))[0]
        lost = np.nonzero((ctrl_en1 & ~ctrl_en0) | (ctrl_own0 & ~ctrl_own1))[0]
        if len(captured) and change > 0:
            for region in captured:
                pool = own.weights(int(region), figures1[region]) if mode == "full" else own.credit(int(region))
                for d, w in pool.items():
                    add(d, w)
            for k in battle_chain(t):
                add(k, 1.)
            label = "captured " + ",".join(names[int(r)] if int(r) < len(names) else str(int(r)) for r in captured)
        elif len(lost) and change < 0:
            for region in lost:
                pool = own.debit_weights(int(region)) if mode == "full" else own.debit(int(region))
                for d, w in pool.items():
                    add(d, w)
            for k in battle_chain(t):
                add(k, 1.)
            label = "lost " + ",".join(names[int(r)] if int(r) < len(names) else str(int(r)) for r in lost)
        else:
            if kinds[t] in COMBAT_KINDS:                       # a battle running: a companion killed, a card's effect
                for k in battle_chain(t):
                    add(k, 1.)
            elif sides[t] == side and kinds[t] != "eye_allocate":
                add(t, 1.)                                     # the side's own decision just applied (a card, a move)
            if side == "sh" and any(name in HUNT_FEATURES for name in moved):
                eye = eye_of_turn.get(rows[t]["turn"])
                if eye is not None:
                    add(eye, 1.)
                if fs0 >= 0:
                    pool = own.weights(fs0, figures0[fs0]) if mode == "full" else own.credit(fs0)
                    for d, w in pool.items():
                        add(d, w)
        if delay_turns:
            responsible = {d: w for d, w in responsible.items() if rows[d]["turn"] <= rows[t]["turn"] - delay_turns}
        if not responsible:
            events.append((t, label, change, {}))
            continue
        if mode == "split":
            total = sum(responsible.values())
            responsible = {d: w / total for d, w in responsible.items()}
        for d, w in responsible.items():
            credit[d] += w * change
        events.append((t, label, change, responsible))
    return credit, events


def game_folders(run, max_games=0):
    folders = sorted(glob.glob(os.path.join(run, "batches", "*", "training_workers", "worker_*", "training", "raw-*")))
    return folders[:max_games] if max_games else folders


def _trace_folder(task):
    folder, side, model, mode, delay, lam = task
    game = load_game(folder)
    if game["status"] != "terminated":
        return None
    own = np.array([r["side"] == side for r in game["rows"]])
    if not own.any():
        return None
    credit, _ = trace(game, model, side, mode=mode, delay_turns=delay, lam=lam)
    return int(game["seed"]), credit[own]


def _credit_inputs(run, side, model_path, mode, delay_turns, lam, max_games):
    """Identify every input before using a derived credit cache.

    Include nonterminal episodes too: an existing folder can become ready
    without changing the folder count. Read both metadata and arrays by content
    so edits, replaced episodes, and newly ready games cannot reuse old credit.
    """
    from .durability import digest_file
    folders = game_folders(run, max_games)
    model_bytes = Path(model_path).read_bytes()
    receipt = dict(schema="event_trace_credit_v2", side=side, mode=mode,
                   delay_turns=delay_turns, lam=lam, max_games=max_games,
                   model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                   episodes=[dict(folder=str(Path(folder).relative_to(run)),
                                  metadata=digest_file(Path(folder) / "episode.json"),
                                  arrays=digest_file(Path(folder) / "arrays.npz"))
                             for folder in folders])
    return folders, model_bytes, receipt


def credits_for_run(run, side, model_path, *, mode="full", delay_turns=0, lam=1., max_games=0, workers=8):
    """{seed: traced credit over the side's own decisions, in order} for every finished game of the run; cached
    in the run folder with the exact model, episode contents and tracing settings."""
    from .contracts import ContractError, canonical_json
    from .runtime_io import _atomic_write
    run = os.path.abspath(run)
    model_path = Path(model_path).resolve()
    folders, model_bytes, receipt = _credit_inputs(run, side, model_path, mode, delay_turns, lam, max_games)
    serialized = canonical_json(receipt)
    tag = f"{mode}_delay{delay_turns}" if delay_turns else mode
    if lam < 1.:
        tag += f"_lam{lam:g}"
    cache = Path(run) / f"trace_{side}_{Path(model_path).stem}_{tag}_{len(folders)}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as d:
                if "receipt" in d.files and str(d["receipt"]) == serialized:
                    seeds, offsets, flat = d["seeds"], d["offsets"], d["credit"]
                    return {int(s): flat[offsets[i]:offsets[i + 1]] for i, s in enumerate(seeds)}
        except (OSError, ValueError, KeyError, BadZipFile):
            # An incomplete derived cache can be rebuilt from its source.
            pass
    model = json.loads(model_bytes.decode("utf-8"))
    out = {}
    tasks = [(folder, side, model, mode, delay_turns, lam) for folder in folders]
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(_trace_folder, tasks, chunksize=64)):
            if part is not None:
                out[part[0]] = part[1]
            if (i + 1) % 5000 == 0:
                print(f"  traced {i + 1} of {len(folders)} games", flush=True)
    if _credit_inputs(run, side, model_path, mode, delay_turns, lam, max_games)[2] != receipt:
        raise ContractError("Tracing inputs changed while producing credits; refusing cache publication")
    seeds = np.array(list(out), np.int64)
    parts = [out[int(s)] for s in seeds]
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    _atomic_write(cache, lambda stream: np.savez(
        stream, seeds=seeds, offsets=offsets, credit=np.concatenate(parts) if parts else np.zeros(0),
        receipt=np.array(serialized)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--side", default="sh", choices=("fp", "sh"))
    parser.add_argument("--model", default=str(ROOT / "selfplay_actor_critic/seeds/explicit_value_a.json"))
    parser.add_argument("--mode", default="full", choices=MODES)
    parser.add_argument("--delay-turns", type=int, default=0, help="credit only decisions at least this many turns before the event")
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--show", type=int, default=2, help="print the events and credited decisions of this many games")
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    run = os.path.abspath(args.run)                   # the engine import below changes the working directory
    cwd = os.getcwd()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import board  # noqa: F401
        from ai import ALL_REGION_KEYS
        names = list(ALL_REGION_KEYS)
    except Exception:
        names = [str(i) for i in range(REGIONS)]
    os.chdir(cwd)
    folders = game_folders(run, args.max_games)
    totals = dict(games=0, events=0, credited=0, decisions=0, nonzero=0, positive=0., negative=0.)
    by_kind = {}
    by_label = {}
    for g, folder in enumerate(folders):
        game = load_game(folder)
        credit, events = trace(game, model, args.side, names, mode=args.mode, delay_turns=args.delay_turns)
        own = np.array([r["side"] == args.side for r in game["rows"]])
        totals["games"] += 1
        totals["events"] += len(events)
        totals["credited"] += sum(1 for e in events if e[3])
        totals["decisions"] += int(own.sum())
        totals["nonzero"] += int((credit[own] != 0).sum())
        totals["positive"] += float(credit[own][credit[own] > 0].sum())
        totals["negative"] += float(credit[own][credit[own] < 0].sum())
        for t in np.nonzero(own)[0]:
            k = game["kind"][t]
            by_kind.setdefault(k, [0, 0., 0])
            by_kind[k][0] += 1
            by_kind[k][1] += float(credit[t])
            by_kind[k][2] += int(credit[t] != 0)
        for t, label, change, shares in events:
            key = label.split(" ")[0] if label.startswith(("captured", "lost")) else label
            by_label.setdefault(key, [0, 0, 0.])
            by_label[key][0] += 1
            by_label[key][1] += int(bool(shares))
            by_label[key][2] += abs(change)
        if g < args.show:
            print(f"game {game['seed']}: {game['winner']} wins by {game['route']}, {len(game['rows'])} decisions")
            for t, label, change, shares in events:
                if not shares:
                    continue
                who = sorted(shares.items(), key=lambda kv: -kv[1])[:10]
                print(f"   decision {t:3d} (turn {game['rows'][t]['turn']:2d}, {game['rows'][t]['side']}, {game['kind'][t]:18s}) {label:28s} value {change:+.3f}")
                for d, w in who:
                    print(f"        {w:5.2f}  decision {d:3d} turn {game['rows'][d]['turn']:2d} {game['kind'][d]}")
    print(f"{totals['games']} games: {totals['events']} value-moving transitions, {totals['credited']} traced to decisions; "
          f"{totals['nonzero']} of {totals['decisions']} {args.side} decisions carry credit (sum of positive {totals['positive']:+.1f}, negative {totals['negative']:+.1f})")
    print(f"{'event':34s} {'count':>6s} {'traced':>6s} {'mean |value|':>12s}")
    for key, (count, traced, size) in sorted(by_label.items(), key=lambda kv: -kv[1][0]):
        print(f"{key:34s} {count:6d} {traced:6d} {size / count:12.3f}")
    print(f"{'kind':22s} {'decisions':>9s} {'with credit':>11s} {'mean credit':>12s}")
    for k, (count, s, nz) in sorted(by_kind.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:22s} {count:9d} {nz:11d} {s / count:+12.4f}")


if __name__ == "__main__":
    main()
