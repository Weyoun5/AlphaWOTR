"""Forced-deviation report: reads the shards and value files, writes report.md and report.json; a pure function of them.

  awpy report.py --out D [--resamples 1000]                 (D/report/report.md, D/report/report.json, D/report/COMPLETE)
  awpy report.py --power forks.jsonl                         (the power table of a list of {"d": ...} rows, for the README)

Numbers are POINTS (100 x the mover's score, win 1 / cap .5 / loss 0).  D = score(forced move) - score(top move), per fork.
Bootstrap: whole seeds resampled (Poisson weights); seeds are pooled into up to 4000 buckets for speed, each a whole-seed block.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dvcommon as D  # noqa: E402
import dvgen  # noqa: E402
C = D.C

DELTA = 0.03                # the effect the design is sized for: +3 points
CI_DECIDE = 0.03            # "no hidden edge" = the whole 95% interval below +3


# ----------------------------------------------------------------------------------------------------------- statistics
def boot_mean(d, seeds, weights=None, resamples=1000, seed=11, buckets=4000):
    """(mean, lo, hi) of a weighted mean, resampling whole seeds."""
    d = np.asarray(d, np.float64)
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    w = np.ones(len(d)) if weights is None else np.asarray(weights, np.float64)
    u, inv = np.unique(seeds, return_inverse=True)
    if len(u) > buckets:
        inv = (u[inv] * 2654435761 % (2 ** 32)) % buckets
        k = buckets
    else:
        k = len(u)
    sums = np.bincount(inv, weights=w * d, minlength=k)
    mass = np.bincount(inv, weights=w, minlength=k)
    point = float(sums.sum() / mass.sum())
    if k < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 100):
        n = min(100, resamples - start)
        draw = rng.poisson(1.0, (n, k)).astype(np.float64)
        means[start:start + n] = (draw @ sums) / np.maximum(draw @ mass, 1e-12)
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def benjamini_hochberg(p, q=0.05):
    """Boolean mask of the rejected hypotheses at FDR q."""
    p = np.asarray(p, np.float64)
    if not len(p):
        return np.zeros(0, bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, len(p) + 1) / len(p))
    out = np.zeros(len(p), bool)
    if passed.any():
        out[order[: int(np.max(np.flatnonzero(passed))) + 1]] = True
    return out


def n_needed(sd, delta=DELTA, alpha=0.05, power=0.8):
    """Pairs for a one-sample z test of mean D = delta against 0, two-sided alpha."""
    nd = NormalDist()
    return int(np.ceil(((nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * sd / delta) ** 2))


def mde(sd, n, alpha=0.05, power=0.8):
    """Smallest effect detectable with this many pairs."""
    nd = NormalDist()
    return float((nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * sd / np.sqrt(max(n, 1)))


def fmt(x, digits=1, sign=True):
    if x is None or not np.isfinite(x):
        return "-"
    return ("%+.*f" if sign else "%.*f") % (digits, x)


def cell(d, base, alt, seeds, weights=None, resamples=1000):
    m, lo, hi = boot_mean(d, seeds, weights, resamples)
    n = len(d)
    if not n:
        return dict(n=0)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    sd = float(np.sqrt(np.sum(w * (d - m) ** 2)))
    se = (hi - lo) / (2 * 1.96) if np.isfinite(hi) else float("nan")
    corr = float(np.corrcoef(alt, base)[0, 1]) if n > 2 and np.std(alt) > 0 and np.std(base) > 0 else float("nan")
    return dict(n=int(n), seeds=int(len(np.unique(seeds))), mean=m * 100, lo=lo * 100, hi=hi * 100, se=se * 100, sd=sd * 100, rho=corr,
                better=float(np.sum(w * (d > 0))) * 100, worse=float(np.sum(w * (d < 0))) * 100, same=float(np.sum(w * (d == 0))) * 100,
                p=float(2 * (1 - NormalDist().cdf(abs(m) / se))) if se and np.isfinite(se) and se > 0 else 1.0,
                mde=mde(sd, n) * 100, needed=n_needed(sd) if sd > 0 else 0)


HEADER = "| %s | n | seeds | mean D | 95%% | D>0 | D<0 | same result | sd D | rho(alt, base) | MDE 80%% |\n|---|---|---|---|---|---|---|---|---|---|---|\n"


def row_text(name, c, extra=""):
    if not c.get("n"):
        return "| %s | 0 | | | | | | | | | |\n" % name
    return "| %s | %d | %d | %s%s | [%s, %s] | %.1f%% | %.1f%% | %.1f%% | %.0f | %s | %.1f |\n" % (
        name, c["n"], c["seeds"], fmt(c["mean"]), extra, fmt(c["lo"]), fmt(c["hi"]), c["better"], c["worse"], c["same"], c["sd"], fmt(c["rho"], 2, False), c["mde"])


# ------------------------------------------------------------------------------------------------------------- mode A
def scan_arrays(out):
    games, forks = dvgen.load_scan_tables(out, kinds=True)
    base = dvgen.attach_base(games, forks) if len(forks["seed"]) else dict(winner=np.zeros(0), route=np.zeros(0), turns=np.zeros(0), plies=np.zeros(0))
    forks["s_alt"] = D.score(forks["winner"], forks["side"]) if len(forks["seed"]) else np.zeros(0)
    forks["s_base"] = D.score(base["winner"], forks["side"]) if len(forks["seed"]) else np.zeros(0)
    forks["d"] = forks["s_alt"] - forks["s_base"]
    forks["base_plies"] = base["plies"]
    forks["base_winner"] = base["winner"]
    return games, forks


def table(lines, title, name, forks, masks, weighted=False, resamples=1000, store=None):
    lines.append("\n%s\n\n" % title)
    lines.append(HEADER % name)
    for key, mask in masks:
        w = 1.0 / forks["hazard"][mask] if weighted else None
        c = cell(forks["d"][mask], forks["s_base"][mask], forks["s_alt"][mask], forks["seed"][mask], w, resamples)
        lines.append(row_text(key, c))
        if store is not None:
            store[key] = c


def mode_a(lines, js, forks, games, resamples):
    ok = forks["status"] == D.OK
    band = (forks["stream"] == D.BAND) & ok
    kind = (forks["stream"] == D.KIND) & ok
    primary = {}
    p_values, keys = [], []
    for code, side in enumerate(D.SIDES):
        lines.append("\n### %s: top band x alternative's band\n\n" % side.upper())
        lines.append(HEADER % "top / alt")
        for tb, top in enumerate(D.TOP_NAMES):
            for ab, alt in enumerate(D.ALT_NAMES):
                m = band & (forks["side"] == code) & (forks["top_band"] == tb) & (forks["alt_band"] == ab)
                if not m.any():
                    continue
                c = cell(forks["d"][m], forks["s_base"][m], forks["s_alt"][m], forks["seed"][m], None, resamples)
                primary["%s %s %s" % (side, top, alt)] = c
                keys.append("%s %s %s" % (side, top, alt))
                p_values.append(c["p"])
                lines.append(row_text("%s / %s" % (top, alt), c))
    reject = benjamini_hochberg(p_values)
    for key, rejected in zip(keys, reject):
        primary[key]["bh05"] = bool(rejected)
    lines.append("\nSignificant after Benjamini-Hochberg at 5%% over these %d cells: %s\n" % (
        len(keys), ", ".join("%s (%s)" % (k, fmt(primary[k]["mean"])) for k, r in zip(keys, reject) if r) or "none"))
    js["primary"] = primary

    marg = {}
    for code, side in enumerate(D.SIDES):
        s = band & (forks["side"] == code)
        conf = s & (forks["top_band"] >= D.CONFIDENT)
        table(lines, "### %s: confident decisions (top >= .8), weighted over decisions (1 / hazard)" % side.upper(), "alt band", forks,
              [(n, conf & (forks["alt_band"] == ab)) for ab, n in enumerate(D.ALT_NAMES)] + [("all", conf)], True, resamples, marg.setdefault(side + "_confident", {}))
        table(lines, "### %s: by top band, all alternatives (weighted)" % side.upper(), "top band", forks,
              [(n, s & (forks["top_band"] == tb)) for tb, n in enumerate(D.TOP_NAMES)], True, resamples, marg.setdefault(side + "_top", {}))
        table(lines, "### %s: by the alternative's rank (2 = runner-up), confident decisions (weighted)" % side.upper(), "rank", forks,
              [("2", conf & (forks["rank"] == 1)), ("3-5", conf & (forks["rank"] >= 2) & (forks["rank"] <= 4)), ("6+", conf & (forks["rank"] >= 5))], True, resamples, marg.setdefault(side + "_rank", {}))
        table(lines, "### %s: by turn, confident decisions (weighted)" % side.upper(), "turn", forks,
              [(("%d-%d" % b), conf & (forks["turn"] >= b[0]) & (forks["turn"] <= b[1])) for b in V_TURNS], True, resamples, marg.setdefault(side + "_turn", {}))
    js["marginals"] = marg

    kinds = {}
    for code, side in enumerate(D.SIDES):
        s = kind & (forks["side"] == code)
        table(lines, "### %s: by the FORCED move's kind (kind stream: uniform over that kind's rows, any prior)" % side.upper(), "forced kind", forks,
              [(g, s & (forks["group_alt"] == i)) for i, g in enumerate(D.GROUPS)], False, resamples, kinds.setdefault(side + "_forced", {}))
        table(lines, "### %s: as above, confident decisions only" % side.upper(), "forced kind", forks,
              [(g, s & (forks["group_alt"] == i) & (forks["top_band"] >= D.CONFIDENT)) for i, g in enumerate(D.GROUPS)], False, resamples, kinds.setdefault(side + "_forced_confident", {}))
        s2 = band & (forks["side"] == code)
        table(lines, "### %s: by the TOP move's kind (band stream, weighted)" % side.upper(), "top kind", forks,
              [(g, s2 & (forks["group_top"] == i)) for i, g in enumerate(D.GROUPS)], True, resamples, kinds.setdefault(side + "_top", {}))
    js["kinds"] = kinds

    # the user's cases, as far as an unselected fork can speak to them
    tags = {}
    real = band | kind
    table(lines, "### The named cases: forks at decisions whose top move matches the query (any forced row)", "case", forks,
          [("%s %s" % (name, side), real & ((forks["tags"] & tag) > 0) & (forks["side"] == code)) for tag, name in D.TAG_NAMES.items() for code, side in enumerate(D.SIDES)],
          False, resamples, tags)
    js["named_forks"] = tags
    return primary, marg


def controls(lines, js, forks, games, resamples):
    ok = forks["status"] == D.OK
    out = {}
    lines.append("\n## Controls\n\n")
    aa = (forks["stream"] == D.EXACT_AA) & ok
    if aa.any():
        same = (forks["trace"][aa] == forks["base_trace"][aa]) & (forks["winner"][aa] == forks["base_winner"][aa])
        out["exact_aa"] = dict(n=int(aa.sum()), identical=float(same.mean()))
        lines.append("- **Exact A/A** (a copy that plays the top move): %d of %d reproduced the game move for move. Anything below 100%% is float noise between batches or a broken copy.\n" % (int(same.sum()), int(aa.sum())))
    ra = (forks["stream"] == D.RESEED_AA) & ok
    if ra.any():
        c = cell(forks["d"][ra], forks["s_base"][ra], forks["s_alt"][ra], forks["seed"][ra], None, resamples)
        out["reseed_aa"] = c
        lines.append("- **Reseeded A/A** (the top move, a fresh engine stream): mean %s [%s, %s] on %d, must cover 0. Its sd %.0f is the noise of an unpaired comparison from the same positions.\n" % (fmt(c["mean"]), fmt(c["lo"]), fmt(c["hi"]), c["n"], c["sd"]))
    hc = (forks["stream"] == D.HANDICAP) & ok
    if hc.any():
        c = cell(forks["d"][hc], forks["s_base"][hc], forks["s_alt"][hc], forks["seed"][hc], None, resamples)
        out["handicap"] = c
        lines.append("- **Handicap** (positive control: the mover's next %d decisions at its lowest eligible prior): mean %s [%s, %s] on %d. Must be clearly negative, or a zero below would mean nothing.\n" % (D.HANDICAP_MOVES, fmt(c["mean"]), fmt(c["lo"]), fmt(c["hi"]), c["n"]))
    real = np.isin(forks["stream"], (D.BAND, D.KIND))
    refused = real & ~ok
    lines.append("- **Refused forks:** %d of %d forced moves (%.3f%%). A refused fork is dropped from its cell; per cell they are at most %.2f%%.\n" % (
        int(refused.sum()), int(real.sum()), 100 * refused.sum() / max(1, real.sum()), max_cell_refusal(forks, real) * 100))
    band = (forks["stream"] == D.BAND) & ok
    if band.any():
        rho = float(np.corrcoef(forks["s_alt"][band], forks["s_base"][band])[0, 1])
        var_d = float(np.var(forks["d"][band]))
        var_ind = float(np.var(forks["s_alt"][band]) + np.var(forks["s_base"][band]))
        out["pairing"] = dict(rho=rho, var_d=var_d, var_independent=var_ind, gain=var_ind / max(var_d, 1e-12), same_result=float((forks["d"][band] == 0).mean()))
        lines.append("- **Pairing:** rho(alt, base) %.3f, so a pair carries %.1fx the information of two independent games; %.1f%% of the pairs end with the same result. It is a variance reduction measured here, never assumed.\n" % (rho, out["pairing"]["gain"], 100 * out["pairing"]["same_result"]))
    js["controls"] = out


def max_cell_refusal(forks, real):
    worst = 0.0
    for key in np.unique(forks["side"][real].astype(np.int64) * 100 + forks["top_band"][real] * 10 + forks["alt_band"][real]):
        m = real & (forks["side"].astype(np.int64) * 100 + forks["top_band"] * 10 + forks["alt_band"] == key)
        if m.sum() >= 50:
            worst = max(worst, float((forks["status"][m] != D.OK).mean()))
    return worst


V_TURNS = ((1, 3), (4, 6), (7, 9), (10, 99))


# ------------------------------------------------------------------------------------------------------------ value
def load_values(out, source, models):
    folder = Path(out) / "values"
    index = folder / ("%s_index.npz" % source)
    if not index.is_file():
        return None
    idx = dict(np.load(index))
    idx["models"] = {}
    for m in models:
        f = folder / ("%s_%s.npy" % (source, m))
        if f.is_file():
            v = np.load(f)
            if len(v) == len(idx["a"]):
                idx["models"][m] = v
    return idx


def model_names(out, source="scan"):
    folder = Path(out) / "values"
    names = sorted(p.name[len(source) + 1:-4] for p in folder.glob("%s_*.npy" % source))
    return [n for n in names if not n.startswith("index")]


def fork_value_gaps(forks, idx, model):
    """V(s after forced) - V(s after top) per fork that kept its views (the mover's model on the mover's view), aligned with `forks`."""
    key = forks["seed"].astype(np.int64) * 10000 + forks["fid"]
    order = np.argsort(key)
    sorted_key = key[order]
    out = {}
    for role in (1, 2):
        m = idx["role"] == role
        k = idx["a"][m].astype(np.int64) * 10000 + idx["b"][m]
        at = np.searchsorted(sorted_key, k)
        at = np.minimum(at, len(sorted_key) - 1)
        good = sorted_key[at] == k
        v = np.full(len(key), np.nan)
        v[order[at[good]]] = idx["models"][model][m][good]
        out[role] = v
    return out[2] - out[1]


def slope(x, y, seeds, resamples=300, seed=5):
    """Slope of y on x (through the means), whole-seed bootstrap (buckets as above)."""
    if len(x) < 3 or np.std(x) == 0:
        return dict(n=int(len(x)))
    def fit(a, b, w):
        xm = np.sum(w * a) / np.sum(w)
        ym = np.sum(w * b) / np.sum(w)
        return float(np.sum(w * (a - xm) * (b - ym)) / np.sum(w * (a - xm) ** 2))
    point = fit(x, y, np.ones(len(x)))
    u, inv = np.unique(seeds, return_inverse=True)
    if len(u) > 2000:
        inv = (u[inv] * 2654435761 % (2 ** 32)) % 2000
        k = 2000
    else:
        k = len(u)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(resamples):
        w = rng.poisson(1.0, k)[inv].astype(np.float64)
        if w.sum() > 2:
            boots.append(fit(x, y, w))
    return dict(n=int(len(x)), slope=point, lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                corr=float(np.corrcoef(x, y)[0, 1]))


def value_section(lines, js, out, forks, models):
    idx = load_values(out, "scan", models)
    if idx is None or not idx["models"]:
        lines.append("\n## Value estimates\n\nNo value files yet (dvvalue.py).\n")
        return
    parity = C.read_json(Path(out) / "values" / "PARITY.json")
    lines.append("\n## Do the value estimates predict which deviations help?\n\n")
    lines.append("dV = V(next decision after the forced move) - V(next decision after the top move), mover's view, one sampled continuation each. The score scale is half the value scale, so **slope 1 = a value gap is worth what it says**. A one-sample gap carries the dice of the move itself, which pulls slopes toward 0; mode B below averages that out.\n\n")
    if parity:
        lines.append("Head parity on %d fork observations: max |recomputed - recorded| %.2e.\n\n" % (parity["rows"], parity["max"]))
    res = {}
    ok = (forks["status"] == D.OK) & np.isin(forks["stream"], (D.BAND, D.KIND))
    for name, v in idx["models"].items():
        gap = fork_value_gaps(forks, idx, name)
        have = ok & np.isfinite(gap)
        res[name] = {}
        lines.append("### %s (%d forks with views)\n\n| set | n | slope of D on dV/2 | 95%% | corr | mean D when dV > +.05 | when dV < -.05 |\n|---|---|---|---|---|---|---|\n" % (name, int(have.sum())))
        for code, side in enumerate(D.SIDES):
            for label, m in (("all", np.ones(len(gap), bool)), ("confident", forks["top_band"] >= D.CONFIDENT)):
                s = have & m & (forks["side"] == code)
                r = slope(gap[s] / 2.0, forks["d"][s], forks["seed"][s])
                hi = forks["d"][s & (gap > 0.05)]
                lo = forks["d"][s & (gap < -0.05)]
                r.update(high_mean=float(hi.mean() * 100) if len(hi) else None, high_n=int(len(hi)), low_mean=float(lo.mean() * 100) if len(lo) else None, low_n=int(len(lo)))
                res[name]["%s_%s" % (side, label)] = r
                if "slope" in r:
                    lines.append("| %s %s | %d | %+.2f | [%+.2f, %+.2f] | %.3f | %s (n %d) | %s (n %d) |\n" % (
                        side, label, r["n"], r["slope"], r["lo"], r["hi"], r["corr"], fmt(r["high_mean"]), r["high_n"], fmt(r["low_mean"]), r["low_n"]))
        lines.append("\n")
    js["value"] = res
    return res


# -------------------------------------------------------------------------------------------------------------- mode B
def stage1_rows(out):
    """One row per (position, alternative): stage-1 mean, se, odd/even k means, source."""
    body, rows, sides = dvgen.read_rolls(Path(out) / "roll1")
    table = dvgen.pair_table(rows, sides)
    out_rows = []
    for c in body["positions"]:
        if c["pid"] not in table:
            continue
        arms = table[c["pid"]]
        top = next((a["arm"] for a in c["arms"] if a["source"] == "top"), None)
        if top not in arms:
            continue
        for a in c["arms"]:
            if a["source"] in ("top", "aa") or a["arm"] not in arms:
                continue
            ka, sa = arms[a["arm"]]
            kt, st = arms[top]
            common, ia, it = np.intersect1d(ka, kt, assume_unique=True, return_indices=True)
            if len(common) < 4:
                continue
            d = sa[ia] - st[it]
            odd = common % 2 == 1
            out_rows.append(dict(pid=c["pid"], arm=a["arm"], source=a["source"], side=c["side"], top_band=c["top_band"], p_top=c["p_top"], p_alt=a.get("p", 0.0),
                                 kind=a.get("kind"), label=a.get("label"), general=c["sample"].startswith("general"), sample=c["sample"], stratum=c.get("stratum"),
                                 tags=c["tags"], n=len(d), mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), odd=float(d[odd].mean()), even=float(d[~odd].mean()),
                                 top_mean=float(st[it].mean()), seed=c["seed"], ply=c["ply"], turn=c["turn"]))
    return body, out_rows


def true_sd(rows, resamples=500, seed=3):
    """Deconvolved spread of the true alt - top difference across alternatives: Cov(odd-k mean, even-k mean) estimates Var(true D),
    position-clustered bootstrap.  Noise cancels because the two halves share no rollout."""
    if len(rows) < 8:
        return dict(n=len(rows))
    odd, even = np.array([r["odd"] for r in rows]), np.array([r["even"] for r in rows])
    pid = np.array([r["pid"] for r in rows])
    u, inv = np.unique(pid, return_inverse=True)
    def cov(w):
        mo = np.sum(w * odd) / np.sum(w)
        me = np.sum(w * even) / np.sum(w)
        return float(np.sum(w * (odd - mo) * (even - me)) / np.sum(w))
    point = cov(np.ones(len(odd)))
    rng = np.random.default_rng(seed)
    boots = np.array([cov(np.maximum(rng.poisson(1.0, len(u))[inv].astype(np.float64), 0) + 1e-9) for _ in range(resamples)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = float(np.mean([r["mean"] for r in rows]))
    sd = np.sqrt(max(point, 0.0))
    nd = NormalDist()
    tail = lambda t: 1 - nd.cdf((t - mean) / sd) if sd > 0 else float(mean > t)
    pos = sd * nd.pdf(mean / sd) + mean * nd.cdf(mean / sd) if sd > 0 else max(mean, 0.0)
    return dict(n=len(rows), positions=int(len(u)), mean=mean * 100, var=point, var_lo=float(lo), var_hi=float(hi), sd=sd * 100, sd_hi=float(np.sqrt(max(hi, 0.0))) * 100,
                normal_share_above_3=tail(0.03), normal_share_above_5=tail(0.05), normal_share_above_10=tail(0.10), normal_mean_positive_part=pos * 100)


def stage2_rows(out):
    folder = Path(out) / "roll2"
    if not (folder / "positions.json").is_file():
        return None, []
    body, rows, sides = dvgen.read_rolls(folder)
    table = dvgen.pair_table(rows, sides)
    out_rows = []
    nd = NormalDist()
    for c in body["positions"]:
        if c["pid"] not in table or 0 not in table[c["pid"]] or 1 not in table[c["pid"]]:
            continue
        d, m, se = dvgen.paired_diff(table, c["pid"], 0, 1)
        if len(d) < 16:
            continue
        z = m / se if se > 0 else 0.0
        alt = c["arms"][0]
        out_rows.append(dict(pid=c["pid"], source_pid=c["source_pid"], sample2=c["sample2"], side=c["side"], top_band=c["top_band"], p_top=c["p_top"], p_alt=alt.get("p"), n=len(d),
                             mean=m, se=se, z=z, p=1 - nd.cdf(z), stage1_z=c.get("stage1_z"), stage1_mean=c.get("stage1_mean"), general=c["sample"].startswith("general"),
                             sample=c["sample"], stratum=c.get("stratum"), tags=c["tags"], turn=c["turn"], top_label=c.get("top_label"), alt_label=alt.get("label"),
                             seed=c["seed"], ply=c["ply"], alt_source=alt.get("source"), alt_kind=alt.get("kind"), top_kind=c.get("top_kind")))
    return body, out_rows


def rescued(lines, js, out, body1, rows1, rows2, resamples=2000):
    """Over the GENERAL sample only, reweighted to decisions: how many confident decisions have a better alternative that survives
    a fresh, independent replication, and what is switching worth?  Promotion happens on stage 1; every number here is stage 2."""
    general = sorted({r["pid"] for r in rows1 if r["general"]})
    if not general:
        return None
    strata_size = (body1.get("info") or {}).get("strata") or {}
    pos = {c["pid"]: c for c in body1["positions"]}
    selected = [r for r in rows2 if r["sample2"] == "selected"]
    sig = benjamini_hochberg([r["p"] for r in selected], 0.05) if selected else np.zeros(0, bool)
    for r, s in zip(selected, sig):
        r["sig_bh05"] = bool(s)
    hit = {r["source_pid"]: r for r in selected}
    weight = {}
    counts = {}
    for pid in general:
        counts[pos[pid]["stratum"]] = counts.get(pos[pid]["stratum"], 0) + 1
    for pid in general:
        st = pos[pid]["stratum"]
        weight[pid] = float(strata_size.get(st, 1)) / counts[st]
    w = np.array([weight[p] for p in general])
    w = w / w.sum()
    in_sig = np.array([bool(hit.get(p, {}).get("sig_bh05")) for p in general], np.float64)
    gain_sig = np.array([max(hit[p]["mean"], 0.0) if hit.get(p, {}).get("sig_bh05") else 0.0 for p in general])
    gain_rule = np.array([hit[p]["mean"] if p in hit else 0.0 for p in general])
    promoted = np.array([p in hit for p in general], np.float64)
    rng = np.random.default_rng(17)
    def ci(x):
        boots = []
        for _ in range(resamples):
            b = rng.poisson(1.0, len(x)).astype(np.float64)
            boots.append(float(np.sum(b * w * x) / max(np.sum(b * w), 1e-12)))
        return float(np.sum(w * x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    res = dict(positions=len(general), promoted=ci(promoted), share_significant=ci(in_sig), points_per_decision_significant=[x * 100 for x in ci(gain_sig)],
               points_per_decision_rule=[x * 100 for x in ci(gain_rule)])
    lines.append("\n## Mode B: positions with top prior > .8\n\n")
    lines.append("Each rollout k re-deals what the mover cannot see and reseeds the dice, the same for every arm, then plays that arm's move and T=0. Stage 1 screens top-5 + 2 random alternatives; stage 2 replays the best one with fresh rollouts, so **stage 2 is not subject to the winner's curse**. Weighted to confident decisions by the strata sizes.\n\n")
    lines.append("| over %d general positions | estimate | 95%% |\n|---|---|---|\n" % len(general))
    lines.append("| share promoted to stage 2 | %.1f%% | [%.1f%%, %.1f%%] |\n" % tuple(100 * x for x in res["promoted"]))
    lines.append("| **share with an alternative significantly better (BH 5%%, stage 2)** | **%.1f%%** | [%.1f%%, %.1f%%] |\n" % tuple(100 * x for x in res["share_significant"]))
    lines.append("| points per confident decision from switching at those | %+.2f | [%+.2f, %+.2f] |\n" % tuple(res["points_per_decision_significant"]))
    lines.append("| points per confident decision of the rule 'switch whenever stage 1 z is high enough' (stage 2 estimate, losses included) | %+.2f | [%+.2f, %+.2f] |\n" % tuple(res["points_per_decision_rule"]))
    rand = [r for r in rows2 if r["sample2"] == "random"]
    if selected:
        m = np.array([r["mean"] for r in selected])
        lines.append("\nSelected alternatives at stage 2: mean %+.1f points over %d (stage 1 said %+.1f: the difference is the winner's curse).\n" % (100 * m.mean(), len(m), 100 * np.mean([r["stage1_mean"] for r in selected])))
    if rand:
        m = np.array([r["mean"] for r in rand])
        lines.append("Random alternatives at stage 2 (unselected, same K): mean %+.1f points over %d.\n" % (100 * m.mean(), len(m)))
    res["selected_mean_stage2"] = float(np.mean([r["mean"] for r in selected])) * 100 if selected else None
    js["rescued"] = res
    return res


def named_section(lines, js, rows1, rows2):
    """Named-query positions: the best alternative's stage 1 and, when promoted, stage 2."""
    lines.append("\n### Named-case positions\n\n| case | positions | mean over alternatives | best alternative's stage-1 mean | promoted | significant at stage 2 |\n|---|---|---|---|---|---|\n")
    res = {}
    for tag, name in D.TAG_NAMES.items():
        for side in D.SIDES:
            mine = [r for r in rows1 if r["tags"] & tag and r["side"] == side]
            pids = sorted({r["pid"] for r in mine})
            if not pids:
                continue
            best = [max(r["mean"] for r in mine if r["pid"] == p) for p in pids]
            s2 = [r for r in rows2 if r["tags"] & tag and r["side"] == side and r["sample2"] == "selected"]
            sig = sum(1 for r in s2 if r.get("sig_bh05"))
            res["%s %s" % (name, side)] = dict(positions=len(pids), mean=float(np.mean([r["mean"] for r in mine])) * 100, best_stage1=float(np.mean(best)) * 100, promoted=len(s2), significant=sig)
            lines.append("| %s %s | %d | %+.1f | %+.1f | %d | %d |\n" % (name, side, len(pids), 100 * np.mean([r["mean"] for r in mine]), 100 * np.mean(best), len(s2), sig))
    named_alts = [r for r in rows1 if r["source"] == "named"]
    if named_alts:
        m = np.array([r["mean"] for r in named_alts])
        lines.append("\nThe query's own alternative (same army move with more units): mean %+.1f over %d positions, stage 1.\n" % (100 * m.mean(), len(m)))
    js["named_positions"] = res
    top = sorted(rows2, key=lambda r: -r["z"])[:15]
    if top:
        lines.append("\n### The largest stage-2 improvements\n\n| side | turn | p top | top move | p alt | alternative | stage-2 D | z | sig |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in top:
            lines.append("| %s | %d | %.3f | %s | %.4f | %s | %+.1f | %.1f | %s |\n" % (r["side"], r["turn"], r["p_top"], short(r["top_label"]), r["p_alt"] or 0.0, short(r["alt_label"]), 100 * r["mean"], r["z"], "yes" if r.get("sig_bh05") else ""))
    js["stage2_rows"] = rows2


def short(label):
    try:
        x = json.loads(label)
    except (TypeError, ValueError):
        return str(label)[:60]
    p = x.get("params") or {}
    keep = {k: v for k, v in p.items() if k in ("from", "to", "n_regular", "n_elite", "card_id", "region", "nation", "count", "companion", "operation")}
    return ("%s %s" % (x.get("kind"), json.dumps(keep, sort_keys=True))).replace("|", "/")[:110]


def spread_section(lines, js, rows1):
    lines.append("\n### True spread of alt - top, stage 1 (position-clustered; noise removed by the odd/even split)\n\n")
    lines.append("| set | alternatives | positions | mean | true sd | true sd 95% upper | share > +3 (normal) | > +5 | > +10 | E[max(0, D)] (normal) |\n|---|---|---|---|---|---|---|---|---|---|\n")
    res = {}
    groups = [("%s %s" % (s, D.TOP_NAMES[b]), [r for r in rows1 if r["general"] and r["side"] == s and r["top_band"] == b]) for s in D.SIDES for b in (2, 3)]
    groups += [("all general", [r for r in rows1 if r["general"]])]
    groups += [("%s, alternative %s" % (s, src), [r for r in rows1 if r["general"] and r["side"] == s and r["source"] == src]) for s in D.SIDES for src in ("top5", "random")]
    for name, rows in groups:
        t = true_sd(rows)
        res[name] = t
        if "sd" in t:
            lines.append("| %s | %d | %d | %+.1f | %.1f | %.1f | %.1f%% | %.1f%% | %.1f%% | %+.2f |\n" % (
                name, t["n"], t["positions"], t["mean"], t["sd"], t["sd_hi"], 100 * t["normal_share_above_3"], 100 * t["normal_share_above_5"], 100 * t["normal_share_above_10"], t["normal_mean_positive_part"]))
    lines.append("\nThe normal columns assume a bell shape for the true differences, which a few very wrong moves would break: read them with the stage-2 share above, not instead of it.\n")
    js["spread"] = res


def value_b(lines, js, out, body, rows1, models):
    idx = load_values(out, "roll1", models)
    if idx is None or not idx["models"]:
        return
    res = {}
    lines.append("\n### Mode B: do the values rank the alternatives?\n\nFor every alternative, dV = mean over up to 16 rollouts of V(first decision after the alternative) - the same for the top, against stage-1 D (64 rollouts). Slope 1 on dV/2 = calibrated. 'Follow V' = per position, the alternative the model likes most, played only when the model prefers it to the top.\n\n")
    lines.append("| model | alternatives | slope | 95% | corr | positions where V prefers an alternative | realised D there |\n|---|---|---|---|---|---|---|\n")
    for name, v in idx["models"].items():
        order = {}
        for a, b, val in zip(idx["a"], idx["b"], v):
            key = int(a)
            order.setdefault(key, []).append((int(b), float(val)))
        xs, ys, pid, follow = [], [], [], {}
        by_pid = {c["pid"]: c for c in body["positions"]}
        for r in rows1:
            if not r["general"]:
                continue
            c = by_pid[r["pid"]]
            top = next((a["arm"] for a in c["arms"] if a["source"] == "top"), None)
            alt_rows = dict(order.get(r["pid"] * 100 + r["arm"] + 1, []))
            top_rows = dict(order.get(r["pid"] * 100 + top + 1, []))
            common = sorted(set(alt_rows) & set(top_rows))
            if not common:
                continue
            gap = float(np.mean([alt_rows[k] - top_rows[k] for k in common]))
            xs.append(gap / 2.0)
            ys.append(r["mean"])
            pid.append(r["pid"])
            best = follow.get(r["pid"])
            if best is None or gap > best[0]:
                follow[r["pid"]] = (gap, r["mean"])
        xs, ys, pid = np.array(xs), np.array(ys), np.array(pid)
        s = slope(xs, ys, pid)
        chosen = [m for g, m in follow.values() if g > 0]
        res[name] = dict(s, follow_positions=len(chosen), follow_mean=float(np.mean(chosen)) * 100 if chosen else None, positions=len(follow))
        if "slope" in s:
            lines.append("| %s | %d | %+.2f | [%+.2f, %+.2f] | %.3f | %d of %d | %s |\n" % (name, s["n"], s["slope"], s["lo"], s["hi"], s["corr"], len(chosen), len(follow), fmt(res[name]["follow_mean"])))
    js["value_b"] = res


# --------------------------------------------------------------------------------------------------------------- verdict
def verdict(lines, js, primary, marg, value, rescue):
    lines.append("\n## What this says (thresholds set before the data)\n\n")
    conf = {k: v for k, v in primary.items() if k.split(" ")[1] in D.TOP_NAMES[D.CONFIDENT:] and v.get("n")}
    up = [k for k, v in conf.items() if v.get("bh05") and v["mean"] > 0]
    down = [k for k, v in conf.items() if v.get("bh05") and v["mean"] < 0]
    open3 = [k for k, v in conf.items() if v["hi"] >= CI_DECIDE * 100]
    lines.append("- **Confident cells:** %d. Significantly better than the top move: %s. Significantly worse: %d. Upper 95%% bound still at or above +3: %s.\n" % (len(conf), ", ".join(up) or "none", len(down), ", ".join(open3) or "none"))
    any_up = bool(up)
    if any_up:
        lines.append("- **Reading A:** whole bands of moves the net rejects are better than its choice on average. The policy is wrong in a way more games of the right kind can see: exploration and a target for those moves are live.\n")
    elif not open3:
        lines.append("- **Reading A:** no confident band hides a +3 edge. Forcing a rejected move does not, on average, beat the top move under this play.\n")
    else:
        lines.append("- **Reading A:** no band is better, but some intervals still reach +3: those stay open.\n")
    if rescue is not None:
        share = rescue["share_significant"][0]
        pts = rescue["points_per_decision_rule"][0]
        if share >= 0.05:
            lines.append("- **Reading B:** %.1f%% of confident decisions have an alternative that replicates, worth %+.2f points per confident decision to the switch rule. Position-specific errors exist; whether the net can learn them is a different question (below).\n" % (100 * share, pts))
        else:
            lines.append("- **Reading B:** only %.1f%% of confident decisions have an alternative that replicates (rule %+.2f points per confident decision). Stage 2 resolves about 4-5 points; smaller edges are not excluded.\n" % (100 * share, pts))
    if value:
        for name, r in value.items():
            parts = []
            for side in D.SIDES:
                x = r.get(side + "_confident") or {}
                if "slope" in x:
                    parts.append("%s slope %+.2f [%+.2f, %+.2f]" % (side.upper(), x["slope"], x["lo"], x["hi"]))
            lines.append("- **%s** on confident forks: %s.\n" % (name, ", ".join(parts)))
    lines.append("\nHow to act on it: **A better or B share >= 5%** means errors are real: if a value's slope is well above 0 there, value-guided search or value-into-policy has something to pick up; if the slopes are near 0 only rollouts see them, so a target built from rollouts is the next test. **A flat and B under 5%** means the confident moves are right against this opponent under this follow-up: stop looking for rejected good moves, and look at follow-through, plans or opponent modelling instead.\n")
    js["verdict"] = dict(confident_cells=len(conf), better=up, worse=len(down), upper_reaches_3=open3)


def power_lines(sd_rows):
    """The README's table: pairs to detect +3 points at 80% power, for measured sds."""
    out = ["| set | pairs in census | sd D (points) | pairs for +3, alpha .05 | pairs for +3, alpha .05/34 |", "|---|---|---|---|---|"]
    for name, x in sd_rows:
        x = np.asarray(x, np.float64)
        sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
        out.append("| %s | %d | %.0f | %d | %d |" % (name, len(x), sd * 100, n_needed(sd), n_needed(sd, alpha=0.05 / 34)))
    return out


def write_report(out, resamples=1000):
    out = Path(out)
    (out / "report").mkdir(parents=True, exist_ok=True)
    lines, js = [], {}
    games, forks = scan_arrays(out)
    config = C.read_json(out / "scan" / "config.json") or {}
    lines.append("# Forced deviation: is the less-likely move better?\n\n")
    lines.append("Generated %s. %d base games (seeds from %s), %d forks. Both sides at T=0 throughout, nets %s and %s. Points = 100 x the mover's score; D = forced move - top move, same game up to the fork.\n" % (
        C.now(), len(games["seed"]), config.get("seed0"), len(forks["seed"]), config.get("fp_ckpt"), config.get("sh_ckpt")))
    fp = float(D.score(games["winner"], 0).mean()) if len(games["seed"]) else float("nan")
    lines.append("\nBaseline: FP score %.3f (panel at 1 sim %.3f on 500 games), %.1f turns, %.0f decisions per game, cap %.2f%%.\n" % (
        fp, dvgen.PANEL_FP_SCORE, float(games["turns"].mean()) if len(games["seed"]) else 0, float(games["plies"].mean()) if len(games["seed"]) else 0, 100 * float((games["winner"] < 0).mean()) if len(games["seed"]) else 0))
    controls(lines, js, forks, games, resamples)
    lines.append("\n## Mode A: one forced move, averaged over decisions\n")
    primary, marg = mode_a(lines, js, forks, games, resamples)
    models = model_names(out)
    value = value_section(lines, js, out, forks, models)
    rescue = None
    if (out / "roll1" / "positions.json").is_file():
        body1, rows1 = stage1_rows(out)
        _b2, rows2 = stage2_rows(out)
        rescue = rescued(lines, js, out, body1, rows1, rows2)
        spread_section(lines, js, rows1)
        named_section(lines, js, rows1, rows2)
        value_b(lines, js, out, body1, rows1, models)
    verdict(lines, js, primary, marg, value, rescue)
    lines.append("\n## What this cannot show\n\n")
    for text in (
            "Only one move is forced, and the net itself plays the rest. A move that needs a follow-up the net never plays looks bad here even if it is good (a plan cannot be tested this way). So a flat result clears the net's moves **given the net's own follow-up**, not against best play.",
            "Only against the installed opponent at T=0. A move can beat this opponent and lose to a human, or the reverse.",
            "Only positions this pair reaches at T=0. Whole lines it never enters, and moves below the prior floor, are not sampled.",
            "Mode A's cells average over positions: a flat cell can hide a few badly wrong moves, which is why mode B exists. Mode B re-deals hidden cards uniformly over what the mover has not seen, not by what the opponent's play suggests.",
            "A better move does not mean the net can learn it: last-layer updates may not represent it. That is another experiment."):
        lines.append("- %s\n" % text)
    (out / "report" / "report.md").write_text("".join(lines), encoding="utf-8")
    C.write_json(out / "report" / "report.json", js)
    (out / "report" / "COMPLETE").write_text(C.now() + "\n")
    return "".join(lines), js


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--power")
    args = ap.parse_args()
    if args.power:
        rows = [json.loads(x) for x in Path(args.power).read_text(encoding="utf-8").splitlines() if x.strip()]
        print("\n".join(power_lines([("all", [r["d"] for r in rows])])))
        return 0
    write_report(D.absolute(args.out), args.resamples)
    print("wrote %s" % (D.absolute(args.out) / "report" / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
