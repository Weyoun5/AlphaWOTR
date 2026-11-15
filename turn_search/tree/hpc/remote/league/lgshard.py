"""Packed league game data: one ZIP shard per worker flush (hpc/remote/league, 2026-10-30).

Never one file per game or decision (file quota).  A shard holds the WHOLE games it lists (a game exists iff it is in a
complete shard: shards are written to a temp name, fsynced and renamed), their records, and every LEARNING side's
decisions grouped by learning agent, so a trainer decompresses only its own agent's members.

Schema "wotr_league_shard_v1" (ZIP_DEFLATED, level 1: ~27x on the menus at ~650 MB/s, measured 2026-10-30):
  meta.json   schema, cycle, node, worker, session, created, phases, games [records], groups {agent: counts}
  <agent>/<field>.npy, per learning agent present in the shard (D decisions, M menu rows, K / Z non-zeros):
    game      int32  [D]    row of meta.games
    ply       int32  [D]    learned-decision number within its game (both sides counted)
    side      int8   [D]    0 FP, 1 SH (the deciding side == the agent's side)
    turn      int16  [D]
    phase     uint8  [D]    index into meta.phases
    sel       int32  [D]    row of the played move within the decision's menu
    logp_sel  float32 [D]   log mu(sel): the BEHAVIOUR policy's tempered log-probability (log softmax(logits / T))
    value     float32 [D]   the behaviour net's value head at the decision (the advantage baseline V_old)
    o0        int64  [D+1]  observation non-zero offsets (observations are sparse: ~260 of 4745 non-zero)
    oidx      int16  [K]    column of each observation non-zero
    oval      float32 [K]   value (exact float32: the trainer reproduces the behaviour logits exactly)
    m0        int64  [D+1]  menu row offsets (the FULL legal menu, never truncated)
    region    int16  [M]    region index per row (-1 non-regional)
    nnz       uint8  [M]    non-zeros per row (action encodings are sparse: ~9 of 199)
    col       uint8  [Z]    column of each row non-zero
    vidx      uint16 [Z]    index into vals
    vals      float32 [V]   distinct non-zero encoding values of the group (exact)
    mu_logp   float16 [M]   tempered behaviour log-probability of every row (KL(mu || pi) telemetry / early stop)
  v2 additions (hybrid update: the race's last-layer policy gradient, pg_fit, and its both-views critic):
    w0, widx, wval          the OTHER side's view at the decision (valuedata "watch"), sparse like o0/oidx/oval
    centred   float32 [D, 192]  batchgen --gradient centred last-layer activations of the played row:
                            h(sel) - sum_a mu(a) h(a) for the taps [spatial_action_scorer last layer (64),
                            policy_delta last layer (128)] (meta groups.<agent>.layout), mu = the tempered behaviour
    kplay     int16  [D]    the played row's engine kind, index into meta groups.<agent>.kinds
    kmass     float32 [D, K]  behaviour probability mass per engine kind (K = len(kinds))
    played_logp float32 [D] UNtempered log-probability of the played row (play_native played_logp)
    top       bool   [D]    played row == argmax
    v0, vidx, vval [V+1]/[*]  VIEW ROWS (review S3-5): the agent's OWN view at the OPPONENT's decisions of its learning
                            games (the watch vector of those decisions), sparse; vgame int32 [V] (row of meta.games),
                            vturn int16 [V], vply int32 [V].  The value head trains on them too (the own-value search
                            evaluates opponent-to-move leaves); meta groups.<agent>.views = V.
  game record (meta.games[i]): seed, fp / sh snapshot ids, learn {fp: agent|None, sh: agent|None}, status, winner,
    route, reason, turns (final game turn), first_turn, decisions {fp, sh}, fp_vp, sh_vp, fp_strongholds (Shadow
    strongholds captured by FP at the end), sh_strongholds (FP strongholds captured by the Shadow), fp_cities,
    sh_cities, corruption, progress, mordor_step, holdout, wall_s, worker, node.
"""
import io
import json
import os
from pathlib import Path
import socket
import zipfile

import numpy as np

SCHEMA = "wotr_league_shard_v2"
SCHEMAS = ("wotr_league_shard_v1", "wotr_league_shard_v2")
FIELDS_D = dict(game=np.int32, ply=np.int32, side=np.int8, turn=np.int16, phase=np.uint8, sel=np.int32,
                logp_sel=np.float32, value=np.float32)
ALL_FIELDS = tuple(FIELDS_D) + ("o0", "oidx", "oval", "m0", "region", "nnz", "col", "vidx", "vals", "mu_logp")
GRADIENT_FIELDS = ("w0", "widx", "wval", "centred", "kplay", "kmass", "played_logp", "top")
VIEW_FIELDS = ("v0", "vidx", "vval", "vgame", "vturn", "vply")


def sparse_rows(dense):
    """dense float32 [n, A] -> (nnz uint8 [n], col uint8 [z], val float32 [z]) in row-major order."""
    rows, cols = np.nonzero(dense)
    nnz = np.bincount(rows, minlength=dense.shape[0])
    if dense.shape[1] > 255 or (len(nnz) and nnz.max() > 255):
        raise ValueError("lgshard: action width / non-zeros per row exceed uint8")
    return nnz.astype(np.uint8), cols.astype(np.uint8), dense[rows, cols].astype(np.float32)


def sparse_vector(vec):
    idx = np.flatnonzero(vec)
    if len(vec) > 32767:
        raise ValueError("lgshard: observation width exceeds int16")
    return idx.astype(np.int16), np.asarray(vec, np.float32)[idx]


def sparsify(x):
    """One recorded decision (dense obs float32 [W], cand float32 [n, A], region [n], mu_logp [n] + the scalar
    fields) -> its compact form (what a worker keeps in memory until the game ends)."""
    oi, ov = sparse_vector(x["obs"])
    nnz, col, val = sparse_rows(np.asarray(x["cand"], np.float32))
    out = {k: x[k] for k in FIELDS_D if k != "game" and k in x}
    out.update(oidx=oi, oval=ov, nnz=nnz, col=col, val=val, region=np.asarray(x["region"], np.int16),
               mu_logp=np.asarray(x["mu_logp"], np.float16))
    if x.get("watch") is not None:
        out["widx"], out["wval"] = sparse_vector(x["watch"])
    for k in ("centred", "kind_played", "kind_mass", "played_logp", "top"):
        if k in x:
            out[k] = x[k]
    return out


class Group:
    def __init__(self):
        self.d = {k: [] for k in FIELDS_D}
        self.oidx, self.oval, self.ocount = [], [], []
        self.region, self.nnz, self.col, self.val, self.mu, self.mcount = [], [], [], [], [], []
        self.widx, self.wval, self.wcount = [], [], []
        self.centred, self.kplay, self.kmass, self.plogp, self.top = [], [], [], [], []
        self.vidx, self.vval, self.vcount, self.vgame, self.vturn, self.vply = [], [], [], [], [], []
        self.kinds = {}                       # engine kind name -> index in this group
        self.gradient = None                  # True when the decisions carry the gradient evidence
        self.bytes = 0

    def add(self, game_row, x):
        """x: a sparsify() result (or a dense decision, sparsified here)."""
        if "oidx" not in x:
            x = dict(sparsify(x), phase=x["phase"])
        for k in FIELDS_D:
            self.d[k].append(game_row if k == "game" else x[k])
        self.oidx.append(x["oidx"])
        self.oval.append(x["oval"])
        self.ocount.append(len(x["oidx"]))
        self.nnz.append(x["nnz"])
        self.col.append(x["col"])
        self.val.append(x["val"])
        self.region.append(x["region"])
        self.mu.append(x["mu_logp"])
        self.mcount.append(len(x["nnz"]))
        grad = "centred" in x
        if self.gradient is None:
            self.gradient = grad
        elif self.gradient != grad:
            raise ValueError("lgshard: decisions with and without gradient evidence in one group")
        if grad:
            self.widx.append(x["widx"])
            self.wval.append(x["wval"])
            self.wcount.append(len(x["widx"]))
            self.centred.append(np.asarray(x["centred"], np.float32))
            self.kplay.append(self.kinds.setdefault(str(x["kind_played"]), len(self.kinds)))
            self.kmass.append({self.kinds.setdefault(str(k), len(self.kinds)): float(m) for k, m in x["kind_mass"]})
            self.plogp.append(float(x["played_logp"]))
            self.top.append(bool(x["top"]))
            self.bytes += 6 * len(x["widx"]) + 4 * len(x["centred"]) + 8 * len(x["kind_mass"])
        self.bytes += 6 * len(x["oidx"]) + 5 * len(x["col"]) + 5 * len(x["nnz"]) + 64

    def add_view(self, game_row, x):
        """One view row: the agent's own view (sparse idx / val) at an opponent decision, its turn and ply."""
        self.vidx.append(x["idx"])
        self.vval.append(x["val"])
        self.vcount.append(len(x["idx"]))
        self.vgame.append(game_row)
        self.vturn.append(int(x["turn"]))
        self.vply.append(int(x["ply"]))
        self.bytes += 6 * len(x["idx"]) + 16

    def arrays(self):
        out = {k: np.asarray(self.d[k], dtype=FIELDS_D[k]) for k in FIELDS_D}
        out["o0"] = np.concatenate([[0], np.cumsum(self.ocount, dtype=np.int64)]).astype(np.int64)
        out["oidx"] = np.concatenate(self.oidx) if self.oidx else np.zeros(0, np.int16)
        out["oval"] = np.concatenate(self.oval) if self.oval else np.zeros(0, np.float32)
        out["m0"] = np.concatenate([[0], np.cumsum(self.mcount, dtype=np.int64)]).astype(np.int64)
        out["region"] = np.concatenate(self.region) if self.region else np.zeros(0, np.int16)
        out["nnz"] = np.concatenate(self.nnz) if self.nnz else np.zeros(0, np.uint8)
        out["col"] = np.concatenate(self.col) if self.col else np.zeros(0, np.uint8)
        allval = np.concatenate(self.val) if self.val else np.zeros(0, np.float32)
        vals, inv = np.unique(allval, return_inverse=True)
        if len(vals) > 65535:
            raise ValueError("lgshard: more than 65535 distinct encoding values in one group")
        out["vals"] = vals.astype(np.float32)
        out["vidx"] = inv.astype(np.uint16).reshape(-1)
        out["mu_logp"] = np.concatenate(self.mu) if self.mu else np.zeros(0, np.float16)
        if self.gradient:
            out["w0"] = np.concatenate([[0], np.cumsum(self.wcount, dtype=np.int64)]).astype(np.int64)
            out["widx"] = np.concatenate(self.widx) if self.widx else np.zeros(0, np.int16)
            out["wval"] = np.concatenate(self.wval) if self.wval else np.zeros(0, np.float32)
            out["centred"] = np.stack(self.centred).astype(np.float32)
            out["kplay"] = np.asarray(self.kplay, np.int16)
            mass = np.zeros((len(self.kmass), len(self.kinds)), np.float32)
            for i, m in enumerate(self.kmass):
                for k, v in m.items():
                    mass[i, k] = v
            out["kmass"] = mass
            out["played_logp"] = np.asarray(self.plogp, np.float32)
            out["top"] = np.asarray(self.top, bool)
            out["v0"] = np.concatenate([[0], np.cumsum(self.vcount, dtype=np.int64)]).astype(np.int64)
            out["vidx"] = np.concatenate(self.vidx) if self.vidx else np.zeros(0, np.int16)
            out["vval"] = np.concatenate(self.vval) if self.vval else np.zeros(0, np.float32)
            out["vgame"] = np.asarray(self.vgame, np.int32)
            out["vturn"] = np.asarray(self.vturn, np.int16)
            out["vply"] = np.asarray(self.vply, np.int32)
        return out


class ShardWriter:
    """Accumulates whole games; flush() writes one shard atomically and returns its manifest row."""

    def __init__(self, folder, stem, meta, level=1):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stem, self.meta, self.level = stem, dict(meta), level
        self.phases = list(self.meta.pop("phases", []) or [])
        self.count = 0
        self._reset()

    def _reset(self):
        self.games, self.groups = [], {}

    def phase_code(self, phase):
        phase = str(phase)
        if phase not in self.phases:
            self.phases.append(phase)
            if len(self.phases) > 255:
                raise RuntimeError("lgshard: more than 255 phases")
        return self.phases.index(phase)

    def add_game(self, record, decisions, views=None):
        """record: the game record (dict); decisions: {agent: [decision dicts]} of the learning sides; views: {agent:
        [view rows]} (the agent's own view at the opponent's decisions)."""
        row = len(self.games)
        self.games.append(dict(record))
        for agent, items in decisions.items():
            g = self.groups.setdefault(agent, Group())
            for x in items:
                x = dict(x)
                x["phase"] = self.phase_code(x["phase"])
                g.add(row, x)
        for agent, items in (views or {}).items():
            g = self.groups.setdefault(agent, Group())
            for x in items:
                g.add_view(row, x)

    def __len__(self):
        return len(self.games)

    @property
    def bytes(self):
        return sum(g.bytes for g in self.groups.values())

    def flush(self):
        if not self.games:
            return None
        groups = {a: g.arrays() for a, g in self.groups.items()}
        meta = dict(self.meta, schema=SCHEMA, created=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
                    phases=self.phases, games=self.games,
                    groups={a: dict(decisions=int(len(arr["sel"])), rows=int(len(arr["region"])),
                                    nnz=int(len(arr["col"])), obs_nnz=int(len(arr["oidx"])),
                                    gradient=bool(self.groups[a].gradient), views=len(self.groups[a].vgame),
                                    kinds=sorted(self.groups[a].kinds, key=self.groups[a].kinds.get),
                                    layout=list(self.meta.get("layout") or []))
                            for a, arr in groups.items()})
        name = "%s_%04d.zip" % (self.stem, self.count)
        path = self.folder / name
        tmp = self.folder / (name + ".tmp%d" % os.getpid())
        with open(tmp, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=self.level) as z:
                z.writestr("meta.json", json.dumps(meta, sort_keys=True, default=str))
                for agent, arr in groups.items():
                    for field, value in arr.items():
                        buf = io.BytesIO()
                        np.lib.format.write_array(buf, np.ascontiguousarray(value), allow_pickle=False)
                        z.writestr("%s/%s.npy" % (agent, field), buf.getvalue())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        row = dict(shard=name, games=[g["seed"] for g in self.games], bytes=path.stat().st_size,
                   decisions={a: int(len(arr["sel"])) for a, arr in groups.items()}, host=socket.gethostname(),
                   pid=os.getpid())
        self.count += 1
        self._reset()
        return row


def read_meta(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
    if meta.get("schema") not in SCHEMAS:
        raise ValueError("%s: schema %r, expected one of %r" % (path, meta.get("schema"), SCHEMAS))
    return meta


def read_group(path, agent, fields=ALL_FIELDS):
    """{field: array} of one agent's group in one shard (only its members are decompressed), plus 'meta'."""
    out = {}
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        if meta.get("schema") not in SCHEMAS:
            raise ValueError("%s: schema %r" % (path, meta.get("schema")))
        if agent not in meta.get("groups", {}):
            return None
        if fields == "all":
            fields = ALL_FIELDS + (GRADIENT_FIELDS if meta["groups"][agent].get("gradient") else ())
        names = set(z.namelist())
        fields = [f for f in fields if "%s/%s.npy" % (agent, f) in names or f not in VIEW_FIELDS]
        for f in fields:
            with z.open("%s/%s.npy" % (agent, f)) as stream:
                out[f] = np.lib.format.read_array(io.BufferedReader(stream), allow_pickle=False)
    out["meta"] = meta
    return out


def shard_paths(folder):
    return sorted(p for p in Path(folder).glob("*.zip") if p.is_file())


def check_group(arr):
    """Structural checks of one group; returns a list of problems."""
    bad = []
    D = len(arr["sel"])
    for k in FIELDS_D:
        if len(arr[k]) != D:
            bad.append("field %s length" % k)
    if arr["m0"].shape != (D + 1,) or arr["m0"][0] != 0 or (D and np.any(np.diff(arr["m0"]) < 1)):
        bad.append("m0 offsets")
    M = int(arr["m0"][-1]) if D else 0
    for k in ("region", "nnz", "mu_logp"):
        if len(arr[k]) != M:
            bad.append("%s rows %d != %d" % (k, len(arr[k]), M))
    Z = int(arr["nnz"].astype(np.int64).sum())
    if len(arr["col"]) != Z or len(arr["vidx"]) != Z:
        bad.append("non-zero counts")
    if Z and int(arr["vidx"].max()) >= len(arr["vals"]):
        bad.append("vidx outside vals")
    if arr["o0"].shape != (D + 1,) or int(arr["o0"][-1]) != len(arr["oidx"]):
        bad.append("o0 offsets")
    sizes = np.diff(arr["m0"]) if D else np.zeros(0)
    if D and (np.any(arr["sel"] < 0) or np.any(arr["sel"] >= sizes)):
        bad.append("sel outside menu")
    return bad


def dense_menu(arr, d, width):
    """Decision d's full menu as dense float32 [n, width] (tests / checks)."""
    lo, hi = int(arr["m0"][d]), int(arr["m0"][d + 1])
    nz0 = np.concatenate([[0], np.cumsum(arr["nnz"].astype(np.int64))])
    out = np.zeros((hi - lo, width), np.float32)
    for r in range(lo, hi):
        a, b = int(nz0[r]), int(nz0[r + 1])
        out[r - lo, arr["col"][a:b].astype(np.int64)] = arr["vals"][arr["vidx"][a:b].astype(np.int64)]
    return out


def dense_obs(arr, d, width):
    a, b = int(arr["o0"][d]), int(arr["o0"][d + 1])
    out = np.zeros(width, np.float32)
    out[arr["oidx"][a:b].astype(np.int64)] = arr["oval"][a:b]
    return out
