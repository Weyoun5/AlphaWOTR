"""FORCED DEVIATION, shared pieces (hpc/remote/league/deviation, 2026-11-02).

One question: when the league net is confident of a move, is the less-likely move actually better?

Everything here is a pure function or a writer; the engine and nets live in dvworker.py.  Every selection is a STOPPING TIME:
it reads only the past of the game (the mover's priors and the kinds of its menu) and a private random stream keyed by
(seed, ply), never the game's RNG and never the future.  That is what makes "the fork against the game that continued" a
causal comparison: picking a decision by anything the baseline does later (say, uniform over the finished game's decisions)
would condition the baseline arm on its own future and not the forced arm.

Units: a result is the mover's SCORE, win 1 / cap .5 / loss 0, so D = score(alt) - score(top) and +.03 is "+3 points".
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import socket
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
# HERE stays first: valuelab has a report.py of its own
for _p in (HERE.parent.parent / "gen", HERE.parent, HERE.parent / "valuelab", HERE):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))
import vlcommon as V  # noqa: E402
C = V.C

SCHEMA = "wotr_deviation_shard_v1"
SIDES = ("fp", "sh")
SIDE_CODE = {"fp": 0, "sh": 1}
PHASE_CODES = dict(V.PHASE_CODES)

# The cluster's tree moved once more after the value lab (11-02 05:10, VALUENET).  Read 11-02 by diffing the cluster's
# tree/src/ai.py (160c10a9...) against the file the valuelab list accepted (valuenet_1102/cluster_ref/src/ai.py, 5bebafb1...):
# an additive SearchValueModel class, load_search_value_model, bind_search_value_model, and `getattr(net, 'value_model', None)`
# switches inside _get_priors_and_value, _mcts_eval_state, _pv_forward_many and _eval_forward_many (+ head_only=True on two
# callers).  All of it is search code, default off; raw play calls none of it (the native engine plays, PolicyActor.predict_many
# scores).  The other 4 entries of vlcommon.SEARCH_PATCHES are still the cluster's contents, every other pinned file and the DLL
# (a9450639...) match the pin (read-only check 11-02).  vlcommon.py itself is not edited, so the value lab's upload stays as it
# is: this process, and each spawned worker (dvworker imports this module first), accepts exactly this one extra hash.
CLUSTER_AI_SHA256 = "f4f9764c74fb2405f1a78e4e6d859a063f3ec8092b942107ba726389d34ff903"
V.SEARCH_PATCHES["ai.py"] = CLUSTER_AI_SHA256

# prior bands (brief): the alternative's raw prior (softmax T=1) and the top move's raw prior at that decision
FLOOR = 1e-4
ALT_EDGES = np.array([1e-4, 1e-3, 1e-2, 5e-2, 0.2])
ALT_NAMES = ("<.1%", ".1-1%", "1-5%", "5-20%", ">20%")
TOP_EDGES = np.array([0.5, 0.8, 0.95])
TOP_NAMES = ("<.5", ".5-.8", ".8-.95", ">.95")
CONFIDENT = 2                       # top bands >= this index are "the net is confident"

# which arm a fork is
BAND, KIND, EXACT_AA, RESEED_AA, HANDICAP, RESEED_B = 0, 1, 2, 3, 4, 5
STREAMS = ("band", "kind", "exact_aa", "reseed_aa", "handicap", "reseed_b")
OK, REFUSED, ALIAS = 0, 1, 2        # fork status
HANDICAP_MOVES = 8                  # the positive control: the mover's next 8 decisions at its lowest eligible prior

# per-game rates: expected forks per base game (hazard = rate / how many such decisions a game has, from census_freq.json)
RATES = dict(band=0.10, kind=0.08, exact_aa=0.01, reseed_aa=0.03, handicap=0.02, candidate=0.02)   # kind: about half fire with no eligible row

# the kinds the question asks about; every engine action_type maps to one (the raw kind is kept as well)
GROUPS = ("army_move", "army_attack", "muster", "play_event", "fellowship_move", "hide", "separate", "pass", "draw",
          "combat_card", "casualty", "character", "political", "hunt", "combat_flow", "other")
GROUP_OF = dict(
    army_move="army_move", army_attack="army_attack", siege_assault="army_attack",
    muster_elite="muster", muster_recruit="muster", muster_minion="muster", muster_regular="muster",
    play_event="play_event", card_option="play_event", discard_card="play_event", discard_table_card="play_event",
    fellowship_move="fellowship_move", mordor_move="fellowship_move",
    fellowship_hide="hide", companion_separate="separate", hunt_guide_separate="separate",
    draw_card="draw", combat_card_play="combat_card", combat_card_none="combat_card", combat_effect_choice="combat_card",
    casualty_allocate="casualty", siege_trim="casualty",
    companion_move="character", nazgul_move="character", mouth_move="character", transform_gandalf="character",
    crown_aragorn="character", elven_ring="character", choose_guide="character", fellowship_declare="character",
    fellowship_reveal_place="character", eye_allocate="character", shine="character",
    advance_political="political",
    hunt_sacrifice_random="hunt", hunt_sacrifice_guide="hunt", hunt_take_corruption="hunt", hunt_gollum_reveal="hunt",
    hunt_gollum_no_reveal="hunt",
    combat_continue="combat_flow", combat_cease="combat_flow", combat_retreat="combat_flow", combat_no_retreat="combat_flow",
    retreat_into_siege="combat_flow", siege_press="combat_flow", siege_no_press="combat_flow", victor_advance="combat_flow",
    victor_stay="combat_flow", shine_none="character")
GROUP_OF["pass"] = "pass"
GROUP_OF["skip"] = "pass"
GROUP_CODE = {g: i for i, g in enumerate(GROUPS)}
N_GROUPS = len(GROUPS)


def group_code(kind):
    return GROUP_CODE[GROUP_OF.get(str(kind), "other")]


# the user's cases, as queries on the position (position facts and the top row only)
TAG_SPLIT, TAG_SIEGE, TAG_DREADFUL, TAG_T1PASS = 1, 2, 4, 8
TAG_NAMES = {TAG_SPLIT: "top_splits_stack", TAG_SIEGE: "top_lifts_own_siege", TAG_DREADFUL: "top_dreadful_spells",
             TAG_T1PASS: "top_t1_pass"}
TAG_MIN_P = 0.5                     # the query only looks at moves the net likes
TAG_RATES = dict(top_splits_stack=0.05, top_lifts_own_siege=1.0, top_dreadful_spells=1.0, top_t1_pass=0.02)   # kept as named candidates
T1PASS_MIN_P = 0.95

N_U = 5 + N_GROUPS + 5              # uniforms drawn per decision: the same count whatever fires, so the stream never shifts
U_AA, U_RESEED, U_HANDICAP, U_CAND, U_VIEW = range(5 + N_GROUPS, N_U)

FORK_FIELDS = dict(seed=np.int64, ply=np.int32, turn=np.int16, phase=np.uint8, side=np.uint8, stream=np.uint8, status=np.uint8,
                   top_band=np.int8, alt_band=np.int8, group_top=np.uint8, group_alt=np.uint8, kind_top=np.uint16,
                   kind_alt=np.uint16, rows=np.int32, elig=np.int32, rank=np.int32, top=np.int32, alt=np.int32, p_top=np.float32,
                   p_alt=np.float32, hazard=np.float32, value=np.float32, tags=np.uint8, fid=np.int32, winner=np.int8,
                   route=np.uint8, end_turn=np.int16, end_ply=np.int32, trace=np.uint64, base_trace=np.uint64)
GAME_FIELDS = dict(seed=np.int64, winner=np.int8, route=np.uint8, turns=np.int16, plies=np.int32, forks=np.int32)
ROLL_FIELDS = dict(pid=np.int32, arm=np.int16, k=np.int32, status=np.uint8, winner=np.int8, route=np.uint8, end_turn=np.int16,
                   end_ply=np.int32, trace=np.uint64)
ROUTE_CODES = {None: 0, "military": 1, "ring_destroyed": 2, "ring_corruption": 3}
ROUTE_NAMES = ("none", "military", "ring_destroyed", "ring_corruption")


def top_band(p):
    return int(np.searchsorted(TOP_EDGES, p, side="right"))


def alt_band(p):
    """-1 below the floor, else the index into ALT_NAMES."""
    return int(np.searchsorted(ALT_EDGES, p, side="right")) - 1


def score(winner_code, side_code):
    """The mover's score, for scalars or arrays: winner_code -1 = no winner (the 40-turn cap)."""
    w = np.asarray(winner_code)
    return np.where(w < 0, 0.5, (w == side_code).astype(np.float64))


def winner_code(end):
    return -1 if end.status != "terminated" or end.winner not in SIDE_CODE else SIDE_CODE[end.winner]


def selector(seed):
    """The decision-selection stream of a game, unrelated to the game's own RNG and samplers."""
    return np.random.default_rng([int(seed), 0xDE71])


def pick_rng(seed, ply, stream, index=0):
    return np.random.default_rng([int(seed), int(ply), int(stream), int(index), 0xA17])


def stream_state(seed, ply, k):
    """The rollout's engine stream, in the form NativeRandom takes: a function of the position and k only, so every arm of
    rollout k starts from the same one."""
    h = int(hashlib.sha256(b"deviation-stream-%d-%d-%d" % (int(seed), int(ply), int(k))).hexdigest()[:16], 16)
    return random.Random(h).getstate()


def determinize_rng(seed, ply, k):
    return np.random.default_rng([int(seed), int(ply), int(k), 0x77])


def trace_step(h, rows, selected):
    """A cheap running digest of a game's decisions (menu size + chosen row): equal traces = the same game, for A/A checks."""
    return ((int(h) * 1000003) ^ (int(rows) << 20) ^ int(selected)) & 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------------------------------------------------------- hazards
def hazards(freq, rates=None):
    """Per-decision firing probabilities from census frequencies: rate / (how many such decisions a game has), capped at 1.
    freq: {"band": {side: [[decisions per game with a row in that band] per alt band] per top band},
           "kind": {side: {group: decisions per game}}, "decisions": {side: per game}, "confident": {side: per game}}."""
    r = dict(RATES, **(rates or {}))

    def h(rate, f):
        return min(1.0, rate / f) if f > 0 else 0.0
    out = dict(rates=r, floor=FLOOR, band={}, kind={})
    total = sum(freq["decisions"][s] for s in SIDES)
    for s in SIDES:
        out["band"][s] = [[h(r["band"], float(f)) for f in row] for row in freq["band"][s]]
        out["kind"][s] = [h(r["kind"], float(freq["kind"][s].get(g, 0.0))) for g in GROUPS]
    out.update(exact_aa=h(r["exact_aa"], total), reseed_aa=h(r["reseed_aa"], total), handicap=h(r["handicap"], total))
    if r.get("reseed_twin"):                # aa.flux: a second fresh stream at the same decision, so two clones can be compared without the base game
        out["reseed_twin"] = True
    conf = sum(freq["confident"][s] for s in SIDES)
    out["candidate"] = h(r["candidate"], conf)
    return out


def plan_decision(seed, ply, side, lp, top, kinds, vocab, hz, u, view_keep=0.3):
    """What a base game does at one decision.  Pure: (forks, candidate?, view?).  A fork: dict(stream, alt, alt_band, hazard).
    u = N_U uniforms from selector(seed), drawn once per decision by the caller.  Reads priors and kinds only."""
    n = len(lp)
    p = np.exp(np.asarray(lp, np.float64))
    tb = top_band(float(p[top]))
    forks = []
    if n < 2:
        return forks, False, False, tb
    band_h, kind_h = hz["band"][side][tb], hz["kind"][side]
    hit_band = [ab for ab in range(5) if u[ab] < band_h[ab]]
    hit_kind = [g for g in range(N_GROUPS) if u[5 + g] < kind_h[g]]
    is_cand = tb >= CONFIDENT and u[U_CAND] < hz["candidate"]
    aa, reseed, handicap = u[U_AA] < hz["exact_aa"], u[U_RESEED] < hz["reseed_aa"], u[U_HANDICAP] < hz["handicap"]
    if not (hit_band or hit_kind or is_cand or aa or reseed or handicap):
        return forks, False, False, tb
    elig = p >= hz["floor"]
    elig[top] = False
    bands = np.searchsorted(ALT_EDGES, p, side="right") - 1
    for ab in hit_band:
        rows = np.flatnonzero(elig & (bands == ab))
        if len(rows):
            forks.append(dict(stream=BAND, alt=int(rows[pick_rng(seed, ply, BAND, ab).integers(len(rows))]), alt_band=ab,
                              hazard=float(band_h[ab])))
    if hit_kind:
        groups = np.array([group_code(k) for k in vocab], np.int64)[np.asarray(kinds, np.int64)]
        for g in hit_kind:
            rows = np.flatnonzero(elig & (groups == g))
            if len(rows):
                alt = int(rows[pick_rng(seed, ply, KIND, g).integers(len(rows))])
                forks.append(dict(stream=KIND, alt=alt, alt_band=alt_band(float(p[alt])), hazard=float(kind_h[g])))
    if aa:
        forks.append(dict(stream=EXACT_AA, alt=int(top), alt_band=-1, hazard=float(hz["exact_aa"])))
    if reseed:
        forks.append(dict(stream=RESEED_AA, alt=int(top), alt_band=-1, hazard=float(hz["reseed_aa"])))
        if hz.get("reseed_twin"):
            forks.append(dict(stream=RESEED_B, alt=int(top), alt_band=-1, hazard=float(hz["reseed_aa"])))
    if handicap and elig.any():
        forks.append(dict(stream=HANDICAP, alt=int(top), alt_band=-1, hazard=float(hz["handicap"])))
    return forks, bool(is_cand and elig.any()), bool(forks) and float(u[U_VIEW]) < view_keep, tb


def lowest_eligible(lp, top, floor=FLOOR):
    """The handicap control's move: the lowest prior at or above the floor, other than the top; the top when there is none."""
    p = np.exp(np.asarray(lp, np.float64))
    mask = p >= floor
    mask[top] = False
    rows = np.flatnonzero(mask)
    return int(rows[np.argmin(p[rows])]) if len(rows) else int(top)


def candidate_arms(seed, ply, lp, top, n_top=5, n_random=2, floor=FLOOR):
    """The alternatives of a rollout position: the top-5 by prior, plus 2 uniformly random rows at or above the floor."""
    p = np.exp(np.asarray(lp, np.float64))
    order = [int(i) for i in np.argsort(-p, kind="stable") if int(i) != int(top)]
    first = order[:n_top]
    rest = [i for i in order[n_top:] if p[i] >= floor]
    r = pick_rng(seed, ply, 9)
    chosen = [int(rest[j]) for j in r.permutation(len(rest))[:n_random]] if rest else []
    return first, chosen


def named_tags(game, ai, side, turn, first_turn, phase, top, p_top, top_kind):
    """The user's cases as queries on the true position (the mover's own move and public board facts only); a tags bit field.
    Decodes rows and the state, so a game asks only for fork decisions and on a share of the games (see dvworker)."""
    tags = 0
    p_top = float(p_top)
    if p_top < TAG_MIN_P:
        return tags
    if top_kind == "pass":
        if int(turn) == int(first_turn) and str(phase) == "actions" and p_top >= T1PASS_MIN_P:
            tags |= TAG_T1PASS
        return tags
    if top_kind == "play_event":
        _d, _k, params = game.row(ai, int(top))
        if "dreadful_spells" in str(params.get("card_id", "")).lower():
            tags |= TAG_DREADFUL
        return tags
    if top_kind != "army_move":
        return tags
    _d, _k, params = game.row(ai, int(top))
    state = game.state(ai)
    army = state.armies.get(params.get("from"))
    if army is not None:
        mine = ai.gs.Side.FREE_PEOPLES if side == "fp" else ai.gs.Side.SHADOW
        own = [u for u in army.units if ai._side_of_nation(u.nation) == mine]
        if int(params.get("n_regular", 0)) + int(params.get("n_elite", 0)) < len(own):
            tags |= TAG_SPLIT
    if getattr(state, "besieged_strongholds", None):
        fork = game.world_copy()
        try:
            fork.world_apply_from(game, int(top))
            if set(state.besieged_strongholds) - set(fork.state(ai).besieged_strongholds):
                tags |= TAG_SIEGE
        finally:
            fork.close()
    return tags


def named_alternative(game, ai, top, lp, kinds, limit=64):
    """The "2 Regulars instead of 1" row: the top's army move with the most units, among the `limit` likeliest army moves
    (decoding a whole menu of thousands of rows is not worth it); None when there is none."""
    code = game.vocab.index("army_move")
    rows = np.flatnonzero(np.asarray(kinds) == code)
    rows = rows[np.argsort(-np.asarray(lp)[rows], kind="stable")][:limit]
    _d, _k, params = game.row(ai, int(top))
    best, size = None, int(params.get("n_regular", 0)) + int(params.get("n_elite", 0))
    for i in rows:
        if int(i) == int(top):
            continue
        _d, _k, other = game.row(ai, int(i))
        if other.get("from") == params.get("from") and other.get("to") == params.get("to"):
            n = int(other.get("n_regular", 0)) + int(other.get("n_elite", 0))
            if n > size:
                best, size = int(i), n
    return best


# ------------------------------------------------------------------------------------------------------------ shards
class ShardWriter:
    """Atomic zip shards, the league's habit: only a flushed game exists, so a killed worker loses only what is unflushed."""

    def __init__(self, folder, stem, meta, level=1):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stem, self.meta, self.level = stem, dict(meta), level
        self.count = 0
        self.kinds = {}
        self._reset()

    def _reset(self):
        self.games, self.forks, self.rolls, self.views, self.candidates, self.chunks = [], [], [], [], [], []
        self.bytes = 0
        self.kinds = {}

    def __len__(self):
        return len(self.games) + len(self.chunks)

    def kind(self, name):
        return self.kinds.setdefault(str(name), len(self.kinds))

    def add_view(self, a, b, role, side, idx, val, term=0, value=float("nan")):
        """One view row.  Scan: a = seed, b = fid; roll: a = pid * 100 + arm + 1, b = k.  role 0 at the fork, 1 baseline's next
        decision, 2 forced arm's next decision; term 1 = the game ended there (score in `value`, no vector)."""
        self.views.append((int(a), int(b), int(role), int(side), int(term), float(value), idx, val))
        self.bytes += 6 * len(idx) + 40

    def add_game(self, game, forks, candidates):
        self.games.append(game)
        for f in forks:
            f["kind_top"] = self.kind(f.pop("kind_top_name"))
            f["kind_alt"] = self.kind(f.pop("kind_alt_name"))
            self.forks.append(f)
            self.bytes += 100
        self.candidates.extend(candidates)

    def add_chunk(self, chunk_id, rows):
        self.chunks.append(chunk_id)
        self.rolls.extend(rows)
        self.bytes += 40 * len(rows)

    def _arrays(self):
        out = {}
        for prefix, rows, fields in (("games/", self.games, GAME_FIELDS), ("forks/", self.forks, FORK_FIELDS), ("rolls/", self.rolls, ROLL_FIELDS)):
            for k, t in fields.items():
                if k == "trace" or k == "base_trace":
                    out[prefix + k] = np.array([int(r.get(k, 0)) for r in rows], np.uint64)
                else:
                    out[prefix + k] = np.array([r[k] for r in rows], t)
        v = self.views
        out["views/a"] = np.array([x[0] for x in v], np.int64)
        out["views/b"] = np.array([x[1] for x in v], np.int64)
        out["views/role"] = np.array([x[2] for x in v], np.uint8)
        out["views/side"] = np.array([x[3] for x in v], np.uint8)
        out["views/term"] = np.array([x[4] for x in v], np.uint8)
        out["views/value"] = np.array([x[5] for x in v], np.float32)
        out["views/o0"] = np.concatenate([[0], np.cumsum([len(x[6]) for x in v], dtype=np.int64)]).astype(np.int64)
        out["views/oidx"] = np.concatenate([x[6] for x in v]) if v else np.zeros(0, np.int16)
        out["views/oval"] = np.concatenate([x[7] for x in v]) if v else np.zeros(0, np.float32)
        return out

    def flush(self):
        if not len(self):
            return None
        name = "%s_%04d.zip" % (self.stem, self.count)
        path = self.folder / name
        tmp = self.folder / (name + ".tmp%d" % os.getpid())
        meta = dict(self.meta, schema=SCHEMA, seeds=[g["seed"] for g in self.games], chunks=list(self.chunks),
                    candidates=self.candidates, kinds=sorted(self.kinds, key=self.kinds.get))
        with open(tmp, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=self.level) as z:
                z.writestr("meta.json", json.dumps(meta, sort_keys=True, default=str))
                for key, value in self._arrays().items():
                    buf = io.BytesIO()
                    np.lib.format.write_array(buf, np.ascontiguousarray(value), allow_pickle=False)
                    z.writestr(key + ".npy", buf.getvalue())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        out = dict(shard=name, games=list(self.chunks) if self.chunks else [g["seed"] for g in self.games], bytes=path.stat().st_size,
                   host=socket.gethostname(), pid=os.getpid())
        self.count += 1
        self._reset()
        return out


def shard_paths(folder):
    return sorted(p for p in Path(folder).glob("*.zip") if p.is_file())


def read_meta(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError("%s: schema %r, expected %r" % (path, meta.get("schema"), SCHEMA))
    return meta


def read_table(path, prefix, fields):
    out = {}
    with zipfile.ZipFile(path) as z:
        for field in fields:
            with z.open("%s%s.npy" % (prefix, field)) as stream:
                out[field] = np.lib.format.read_array(io.BufferedReader(stream), allow_pickle=False)
    return out


VIEW_FIELDS = ("a", "b", "role", "side", "term", "value", "o0", "oidx", "oval")


def absolute(path):
    """Every path a tool writes to, absolute, BEFORE anything imports the engine (board.py chdirs to src/ on import)."""
    return Path(path).resolve()
