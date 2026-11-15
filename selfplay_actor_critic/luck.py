"""The chance control variate: price every logged chance event with a frozen critic and subtract the luck.

For a chance event j of a game (chance_log.py: the position before it, the exact outcome distribution, both
views of every alternative, the realised position) and a frozen critic u (the Shadow's win value in (-1, 1),
exact +-1 where an alternative decides the game):
    L_j = u(realised position) - sum_c p(c) u(alternative c)
has zero conditional mean (the live engine draws from the distribution the enumeration describes), so for a
decision t the corrected return  z - sum over events after t of L_j  has the same expectation and less variance
when u prices the luck. The corrections are cached per run (luck_<critic>.npz) for pg_fit --luck; --check reports
the mean of L (must be zero within noise), the variance of a game's total luck and the share of the result's
variance it removes.

  python -m selfplay_actor_critic.luck --run <valuedata run with --chance-log> --critic-dir selfplay_actor_critic/seeds/both_views_r1 --check
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
from .report_metrics import correlation, explained_fraction, metric_text, ratio

ROOT = Path(__file__).resolve().parents[1]


def load_critic(critic_dir, device="cpu"):
    """A function values(sh_view, fp_view) -> the Shadow's value, from a both-views critic folder (BOTH_VIEWS marker)
    or a single-view Shadow critic folder (sh_critic.pt)."""
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    from .generation import require_generation
    require_generation("luck.load_critic")
    os.environ.setdefault("WOTR_NATIVE", "0")
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    import torch
    critic_dir = Path(critic_dir)
    if (critic_dir / "BOTH_VIEWS").exists():
        from .full_state_critic import load_full_state_critic
        critic = load_full_state_critic(critic_dir, device, ai=ai)
        return lambda sh, fp: np.asarray(critic.value_shadow(np.asarray(sh, np.float32), np.asarray(fp, np.float32)), np.float64), critic_dir.name
    from .policy import load_critic as load_single
    critic = load_single(ai, critic_dir / "sh_critic.pt", "sh", device=device)

    @torch.inference_mode()
    def values(sh, fp):
        out = []
        for start in range(0, len(sh), 8192):
            x = torch.from_numpy(np.asarray(sh[start:start + 8192], np.float32)).to(device)
            out.append(critic.value_head(critic.forward_backbone(x)[0]).reshape(-1).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return values, critic_dir.name


def game_folders(run):
    return sorted(glob.glob(os.path.join(run, "batches", "*", "training_workers", "worker_*", "training", "raw-*")))


def read_game(folder):
    meta = json.load(open(os.path.join(folder, "episode.json"), encoding="utf-8"))
    end = meta.get("end") or {}
    if end.get("status") != "terminated":
        return None
    with np.load(os.path.join(folder, "arrays.npz")) as d:
        if "chance_after" not in d.files:
            return None
        keys = ("chance_after", "chance_kind", "chance_offsets", "chance_probs", "chance_terminal", "chance_obs_sh", "chance_obs_fp",
                "chance_live_sh", "chance_live_fp", "chance_live_terminal")
        arrays = {k: d[k] for k in keys}
    sides = np.array([r["side"] for r in meta["rows"]])
    return dict(seed=int(meta["seed"]), result=1. if end["winner"] == "sh" else -1., sides=sides, **arrays)


def event_terms(game, values):
    """L_j per event (Shadow units): the critic's value of the realised position minus its expectation over the
    enumerated alternatives (exact +-1 where an alternative decides the game)."""
    E = len(game["chance_after"])
    offsets = game["chance_offsets"]
    if E == 0:
        return np.zeros(0)
    u_alt = values(game["chance_obs_sh"], game["chance_obs_fp"])
    u_live = values(game["chance_live_sh"], game["chance_live_fp"])
    exact = ~np.isnan(game["chance_terminal"])
    u_alt = np.where(exact, game["chance_terminal"], u_alt)
    exact_live = ~np.isnan(game["chance_live_terminal"])
    u_live = np.where(exact_live, game["chance_live_terminal"], u_live)
    L = np.zeros(E)
    for j in range(E):
        a, b = offsets[j], offsets[j + 1]
        p = game["chance_probs"][a:b].astype(np.float64)
        L[j] = u_live[j] - float(p @ u_alt[a:b]) / max(float(p.sum()), 1e-12)
    return L


def corrections_from(game, L):
    """Per-decision corrections for each side (mover's perspective) from per-event terms L: decision t (row
    index) is corrected by every event with chance_after > t (the event follows decision t or later)."""
    sides = game["sides"]
    n = len(sides)
    after = game["chance_after"].astype(np.int64)
    per_row = np.zeros(n)
    cumulative = np.zeros(n + 2)                 # cumulative[k] = sum of L over events with chance_after == k
    np.add.at(cumulative, np.clip(after, 0, n + 1), L)
    suffix = np.cumsum(cumulative[::-1])[::-1]   # suffix[k] = sum over events with chance_after >= k
    for t in range(n):
        per_row[t] = suffix[t + 1]
    corrections = {}
    for s in ("sh", "fp"):
        own = sides == s
        corrections[s] = per_row[own] * (1. if s == "sh" else -1.)
    return corrections


def luck_terms(game, values):
    """(L per event, per-decision corrections per side) with the raw (unscaled) terms."""
    L = event_terms(game, values)
    return L, corrections_from(game, L)


def audit_mask(seeds, fraction, audit_seed):
    """The audit set: a reproducible share of whole games chosen from their seeds alone (hash of seed and audit
    seed), so pg_fit and luck.py agree on it without sharing state. fraction 0 = no audit games."""
    seeds = np.asarray(seeds, np.int64)
    if fraction <= 0:
        return np.zeros(len(seeds), bool)
    salt = np.uint64((int(audit_seed) * 0xD1B54A32D192ED03 + 1) % (2 ** 64))
    with np.errstate(over="ignore"):
        h = (seeds.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + salt) % np.uint64(2 ** 61 - 1)
    return (h.astype(np.float64) / float(2 ** 61 - 1)) < fraction


def fit_scales(totals, z, folds, kinds, n_folds=5, include=None):
    """Control-variate coefficients per event kind, cross-fitted by fold: beta_f = argmin |z - T beta|^2 on the
    games outside fold f AND inside `include` (the non-audit games; None = all). A fixed beta keeps the zero
    mean; the cross-fit keeps it honest; excluded games never enter any fit. Returns [n_folds, kinds]."""
    include = np.ones(len(z), bool) if include is None else np.asarray(include, bool)
    beta = np.zeros((n_folds, len(kinds)))
    for f in range(n_folds):
        m = (folds != f) & include
        if m.sum() < 2 * len(kinds) + 2:
            beta[f] = 1.
            continue
        T = totals[m]
        beta[f] = np.linalg.lstsq(T, z[m] - z[m].mean(), rcond=None)[0]
    return beta


def coefficient_rows(totals, z, seeds, include, n_folds=5):
    """The coefficients each game's correction is scaled by, the cache's one rule (build() and pg_fit's held-out
    refits share it): beta per fold fitted (fit_scales) on the `include` games outside that fold, folds = seed %
    n_folds; an included game takes its fold's row, any other game (an audit game, a held-out CV fold) the mean of
    the rows, all fitted without it. Fewer than 20 included games: every coefficient 1.
    Returns (beta [n_folds, kinds], rows [games, kinds])."""
    totals = np.asarray(totals, np.float64)
    include = np.asarray(include, bool)
    folds = np.asarray(seeds, np.int64) % n_folds
    kinds = totals.shape[1]
    if int(include.sum()) >= 20:
        beta = fit_scales(totals, np.asarray(z, np.float64), folds, range(kinds), n_folds, include=include)
    else:
        beta = np.ones((n_folds, kinds))
    rows = np.where(include[:, None], beta[folds], beta.mean(0)[None, :])
    return beta, rows


def build(run, critic_dir, *, device="cpu", threads=4, max_games=0, audit_fraction=0., audit_seed=0):
    values, name = load_critic(critic_dir, device)
    folders = game_folders(run)
    if max_games:
        folders = folders[:max_games]
    games, out = [], dict(seeds=[], result=[], total=[], events=[], L=[], kinds=[])

    def slim(game):  # what the scaling and the corrections need; the chance observations (~35 MB per game) are dropped
        return {k: game[k] for k in ("seed", "result", "sides", "chance_after", "chance_kind")}

    with ThreadPoolExecutor(threads) as pool:
        # bounded read-ahead: pool.map over ALL folders would hold every read game until consumed (3,000 logged games
        # = ~100 GB; the 10-11 19:07 run died of exactly this), so games are read in chunks of 4 x threads
        step = 4 * max(1, threads)
        for start in range(0, len(folders), step):
          for game in pool.map(read_game, folders[start:start + step]):
            if game is None:
                continue
            L = event_terms(game, values)
            games.append((slim(game), L))
            out["seeds"].append(game["seed"])
            out["result"].append(game["result"])
            out["total"].append(float(L.sum()))
            out["events"].append(len(L))
            out["L"].append(L)
            out["kinds"].append(game["chance_kind"])
    kinds = sorted({str(k) for g, _ in games for k in g["chance_kind"].tolist()})
    index = {k: i for i, k in enumerate(kinds)}
    z = np.asarray(out["result"])
    seeds = np.asarray(out["seeds"], np.int64)
    audit = audit_mask(seeds, audit_fraction, audit_seed)
    totals = np.zeros((len(games), len(kinds)))
    for g, (game, L) in enumerate(games):
        for k, l in zip(game["chance_kind"].tolist(), L):
            totals[g, index[str(k)]] += l
    # an audit game gets the mean of the fold coefficients (all fitted without it); a non-audit game its fold's
    # coefficients (fitted without its fold)
    beta, rows = coefficient_rows(totals, z, seeds, ~audit)
    raw = {"sh": [], "fp": []}
    scaled = {"sh": [], "fp": []}
    # the raw corrections of each event kind apart (columns in kind_names order): the scaled correction is linear in
    # the coefficients, so pg_fit's held-out protocol can refit them on its training folds only (S2-04)
    per_kind = {"sh": [], "fp": []}
    scaled_total = np.zeros(len(games))
    for g, (game, L) in enumerate(games):
        c = corrections_from(game, L)
        row = rows[g]
        event_kind = np.array([index[str(k)] for k in game["chance_kind"].tolist()], np.int64)
        factors = row[event_kind] if len(event_kind) else np.zeros(0)
        cs = corrections_from(game, L * factors)
        scaled_total[g] = float((L * factors).sum())
        columns = [corrections_from(game, np.where(event_kind == k, L, 0.)) for k in range(len(kinds))]
        for side in ("sh", "fp"):
            raw[side].append(c[side])
            scaled[side].append(cs[side])
            per_kind[side].append(np.column_stack([col[side] for col in columns]) if columns else np.zeros((len(c[side]), 0)))
    cache = Path(run) / f"luck_{name}.npz"
    arrays = dict(seeds=np.asarray(out["seeds"], np.int64), result=z, total=np.asarray(out["total"]), scaled_total=scaled_total,
                  events=np.asarray(out["events"], np.int64),
                  L=np.concatenate(out["L"]) if out["L"] else np.zeros(0), kinds=np.concatenate(out["kinds"]) if out["kinds"] else np.array([]),
                  critic=np.array(name), kind_names=np.array(kinds), beta=beta, audit=audit,
                  audit_fraction=np.float64(audit_fraction), audit_seed=np.int64(audit_seed), kind_totals=totals)
    for s in ("sh", "fp"):
        arrays[f"offsets_{s}"] = np.concatenate([[0], np.cumsum([len(p) for p in raw[s]])]).astype(np.int64)
        arrays[f"correction_raw_{s}"] = np.concatenate(raw[s]) if raw[s] else np.zeros(0)
        arrays[f"correction_{s}"] = np.concatenate(scaled[s]) if scaled[s] else np.zeros(0)
        arrays[f"correction_kind_{s}"] = np.concatenate(per_kind[s]) if per_kind[s] else np.zeros((0, len(kinds)))
    np.savez(cache, **arrays)
    return cache, arrays


def load_corrections(run, stem, side, raw=False):
    """{seed: per-own-decision correction (mover's perspective)} from a cache written by build(); the scaled
    (cross-fitted control-variate coefficients per event kind) corrections unless raw."""
    with np.load(Path(run) / f"luck_{stem}.npz") as d:
        seeds, offsets, flat = d["seeds"], d[f"offsets_{side}"], d[f"correction_raw_{side}" if raw else f"correction_{side}"]
    return {int(s): flat[offsets[i]:offsets[i + 1]] for i, s in enumerate(seeds)}


def load_kind_terms(run, stem, side):
    """What refitting the coefficients needs (pg_fit's held-out protocol, coefficient_rows): (kind names,
    {seed: [own decisions, kinds] raw correction of each event kind (mover's perspective)}, {seed: [kinds] the
    game's total luck of each kind}, {seed: the game's result in Shadow units}). A cache written before these
    arrays existed (2026-10-22) cannot be refitted and must be rebuilt."""
    path = Path(run) / f"luck_{stem}.npz"
    with np.load(path) as d:
        if f"correction_kind_{side}" not in d.files or "kind_totals" not in d.files:
            raise SystemExit(f"{path} has no per-kind luck terms, so its coefficients cannot be refitted on the training folds "
                             f"(held-out protocol): rebuild it with luck.py --run {run} (same --critic-dir, --audit-fraction, --audit-seed)")
        names = [str(k) for k in d["kind_names"].tolist()]
        seeds, offsets, flat = d["seeds"], d[f"offsets_{side}"], d[f"correction_kind_{side}"]
        totals, result = d["kind_totals"], d["result"]
    corrections = {int(s): flat[offsets[i]:offsets[i + 1]] for i, s in enumerate(seeds)}
    return names, corrections, {int(s): totals[i] for i, s in enumerate(seeds)}, {int(s): float(result[i]) for i, s in enumerate(seeds)}


def check(arrays):
    L, kinds, z, total = arrays["L"], arrays["kinds"], arrays["result"], arrays["total"]
    games = len(z)
    print(f"{games} games, {len(L)} chance events ({len(L) / max(games, 1):.1f} per game); critic {arrays['critic']}")
    if len(L):
        error = L.std() / np.sqrt(len(L))
        print(f"  mean of L {L.mean():+.4f} +- {error:.4f} (zero by construction; t = {metric_text(ratio(L.mean(), error), '+.2f')}); sd {L.std():.3f}")
        for kind in sorted(set(kinds.tolist())):
            m = kinds == kind
            print(f"    {kind:8s} {int(m.sum()):6d} events  mean {L[m].mean():+.4f} +- {L[m].std() / np.sqrt(m.sum()):.4f}  sd {L[m].std():.3f}")
    if games > 2:
        corr = correlation(total, z)
        var_z, var_c = z.var(), (z - total).var()
        print(f"  per game: total luck sd {total.std():.3f}, result sd {z.std():.3f}, correlation {metric_text(corr, '+.3f')}; "
              f"Var(z) {var_z:.3f} -> Var(z - total luck) {var_c:.3f}: {metric_text(explained_fraction(var_c, var_z), '.1f', scale=100, suffix='%')} of the result's variance removed "
              f"(+-{100 * 2 / np.sqrt(games):.0f}% at this size)")
        # the gradient term's noise falls with the per-decision residual's second moment; the game total is the
        # worst case (every decision is corrected by the whole game's luck)
        scaled = arrays["scaled_total"]
        var_s = (z - scaled).var()
        print("  scaled by cross-fitted coefficients per kind " + ", ".join(f"{k} {arrays['beta'][:, i].mean():+.2f}" for i, k in enumerate(arrays["kind_names"].tolist()))
              + f": Var(z - scaled luck) {var_s:.3f}: {metric_text(explained_fraction(var_s, var_z), '.1f', scale=100, suffix='%')} removed")
        noise_ratio = ratio(var_s, var_z)
        print(f"  implied noise of a random step: x{metric_text(None if noise_ratio is None else np.sqrt(noise_ratio))} of the uncorrected (game totals, scaled)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--critic-dir", default=str(ROOT / "selfplay_actor_critic/seeds/both_views_r1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--audit-fraction", type=float, default=0., help="games held out of the coefficient fit (pg_fit --audit-fraction, same rule)")
    parser.add_argument("--audit-seed", type=int, default=0)
    args = parser.parse_args()
    import torch
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    cache, arrays = build(args.run, args.critic_dir, device=device, threads=args.threads, max_games=args.max_games,
                          audit_fraction=args.audit_fraction, audit_seed=args.audit_seed)
    print(f"written {cache}; coefficients fitted on {int((~arrays['audit']).sum())} games"
          + (f" (the {int(arrays['audit'].sum())} audit games, fraction {args.audit_fraction:g} seed {args.audit_seed}, saw none of them)" if args.audit_fraction > 0 else " (no audit set)"))
    if args.check:
        check(arrays)


if __name__ == "__main__":
    main()
