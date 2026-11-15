"""VALUE LAB, shared pieces (hpc/remote/league/valuelab, 2026-11-02).

One question: how much better can a dedicated value network predict game outcomes than the current value heads?
This file: the game split, the row subsample, the compact shard format (no menus: a value model never reads them) and
the hashes the within-position check needs.  The league's own code is imported, never changed (lgcommon.reward = the
objective returns, lgworker = the recorder, lgnode.Pool = the persistent workers).

Rows.  One decision of a game gives ONE ROW PER SIDE, in that side's own view:
    own=1   the side moves here (the observation the actor saw; `value` = its head as played, `move` and `pos` are set)
    own=0   the opponent moves here (the native "watch" vector, what the own-value search evaluates at such a leaf)
so a side's rows of a game, in ply order, are exactly the positions its value head is asked about.  ply counts both
sides' learned decisions.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent, HERE.parent.parent / "gen"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lgcommon as C  # noqa: E402

SCHEMA = "wotr_valuelab_shard_v1"
SIDES = ("fp", "sh")
OBJECTIVE = {"fp": "fp_win", "sh": "sh_win"}       # the installed pair's objectives: win +1, loss -1, 40-turn cap 0
BUCKETS = ("train", "val", "test")
TEST_SHARE, VAL_SHARE = 800, 200                   # of 10,000 (whole games, by seed hash)
TURN_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 16), (17, 99))
PHASE_CODES = {"opening": 0, "declaration": 1, "preaction": 2, "actions": 3}
TARGET_CLIP = float(C.CONFIG["train"]["value"]["target_clip"])    # the league's: training targets only, never scoring
ROW_FIELDS = dict(ply=np.int32, turn=np.int16, phase=np.uint8, own=np.uint8, value=np.float32, move=np.uint64,
                  pos=np.uint64, top=np.uint8)


def bucket(seed):
    """'train' | 'val' | 'test': a stable hash of the seed, so whole games stay on one side of the split.  'val' selects
    checkpoints; 'test' is only ever read by report.py."""
    h = int(hashlib.sha256(b"valuelab-split-%d" % int(seed)).hexdigest()[:8], 16) % 10000
    return "test" if h < TEST_SHARE else ("val" if h < TEST_SHARE + VAL_SHARE else "train")


def keep_row(seed, ply, keep_train):
    """Train games keep a deterministic share of their decisions (both sides' rows of a kept decision); val and test keep
    all.  Rows of one game share one result: outcome information grows with GAMES, so disk and epochs buy more spread
    over many games than over all rows of few games."""
    if keep_train >= 1.0 or bucket(seed) != "train":
        return True
    h = int(hashlib.sha256(b"valuelab-row-%d-%d" % (int(seed), int(ply))).hexdigest()[:8], 16)
    return h / float(1 << 32) < keep_train


def handover(seed, spec):
    """EXPLORATION GAMES: (side, first turn, net id) of the ONE stretch of game `seed` in which another net of the same side
    plays that side's moves (spec["turns"] turns long), or None.  A function of the seed alone, so a game replays the same
    and a reader of the shards can tell what its game held.  spec: {"fp": [net ids], "sh": [net ids], "turns", "first",
    "last", "share"}: the side is drawn evenly, the first turn evenly from first..last, the net evenly from the side's list;
    a game that ends before its first turn simply has no stretch."""
    h = hash64(b"valuelab-handover", np.int64(seed).tobytes())
    rng = np.random.default_rng([int(h & 0xFFFFFFFF), int(h >> 32)])
    if rng.random() >= float(spec.get("share", 1.0)):
        return None
    side = SIDES[int(rng.integers(2))]
    nets = spec.get(side) or []
    if not nets:
        return None
    first = int(rng.integers(int(spec["first"]), int(spec["last"]) + 1))
    return side, first, str(nets[int(rng.integers(len(nets)))])


def hash64(*parts):
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest(), "little")


def move_key(candidate, region):
    """The identity of a played move: its action encoding and region (two moves that encode alike count as one)."""
    i = np.flatnonzero(candidate)
    return hash64(i.astype(np.int16).tobytes(), np.asarray(candidate, np.float32)[i].tobytes(), np.int64(region).tobytes())


def sparse(vec):
    i = np.flatnonzero(vec)
    return i.astype(np.int16), np.asarray(vec, np.float32)[i]


def own_row(ply, turn, phase, value, idx, val, move, top):
    return dict(ply=ply, turn=turn, phase=PHASE_CODES[str(phase)], own=1, value=value, idx=idx, val=val, move=move,
                pos=hash64(idx.tobytes(), val.tobytes()), top=int(bool(top)))


def view_row(ply, turn, phase, idx, val):
    return dict(ply=ply, turn=turn, phase=PHASE_CODES[str(phase)], own=0, value=float("nan"), idx=idx, val=val, move=0,
                pos=0, top=0)


class ValueShardWriter:
    """lgshard.ShardWriter's interface and atomic flush for compact rows: <side>/<field>.npy, one row per kept decision
    per side (game, ROW_FIELDS, o0 / oidx / oval = the sparse observation), plus meta.json with the game records, each
    carrying rows {side: [rows, non-zeros]} so the compiler can lay the data out without reading the arrays."""

    def __init__(self, folder, stem, meta, level=1):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stem, self.meta, self.level = stem, dict(meta), level
        self.count = 0
        self._reset()

    def _reset(self):
        self.games, self.rows = [], {s: [] for s in SIDES}
        self.bytes = 0

    def add_game(self, record, decisions, views=None):
        row = len(self.games)
        record = dict(record, rows={})
        for side in SIDES:
            items = sorted(list(decisions.get(side) or []) + list((views or {}).get(side) or []), key=lambda x: x["ply"])
            nnz = 0
            for x in items:
                self.rows[side].append((row, x))
                nnz += len(x["idx"])
            record["rows"][side] = [len(items), nnz]
            self.bytes += 6 * nnz + 40 * len(items)
        self.games.append(record)

    def __len__(self):
        return len(self.games)

    def arrays(self, side):
        rows = self.rows[side]
        out = dict(game=np.array([g for g, _ in rows], np.int32))
        for k, dtype in ROW_FIELDS.items():
            out[k] = np.array([x[k] for _, x in rows], dtype)
        out["o0"] = np.concatenate([[0], np.cumsum([len(x["idx"]) for _, x in rows], dtype=np.int64)]).astype(np.int64)
        out["oidx"] = np.concatenate([x["idx"] for _, x in rows]) if rows else np.zeros(0, np.int16)
        out["oval"] = np.concatenate([x["val"] for _, x in rows]) if rows else np.zeros(0, np.float32)
        return out

    def flush(self):
        if not self.games:
            return None
        name = "%s_%04d.zip" % (self.stem, self.count)
        path = self.folder / name
        tmp = self.folder / (name + ".tmp%d" % os.getpid())
        meta = dict(self.meta, schema=SCHEMA, games=self.games)
        decisions = {s: len(self.rows[s]) for s in SIDES}
        with open(tmp, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=self.level) as z:
                z.writestr("meta.json", json.dumps(meta, sort_keys=True, default=str))
                for side in SIDES:
                    for field, value in self.arrays(side).items():
                        buf = io.BytesIO()
                        np.lib.format.write_array(buf, np.ascontiguousarray(value), allow_pickle=False)
                        z.writestr("%s/%s.npy" % (side, field), buf.getvalue())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        out = dict(shard=name, games=[g["seed"] for g in self.games], bytes=path.stat().st_size, decisions=decisions,
                   host=socket.gethostname(), pid=os.getpid())
        self.count += 1
        self._reset()
        return out


def read_meta(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError("%s: schema %r, expected %r" % (path, meta.get("schema"), SCHEMA))
    return meta


def read_side(path, side):
    """{field: array} of one side's rows in one shard."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for field in ("game", "o0", "oidx", "oval") + tuple(ROW_FIELDS):
            with z.open("%s/%s.npy" % (side, field)) as stream:
                out[field] = np.lib.format.read_array(io.BufferedReader(stream), allow_pickle=False)
    return out


def shard_paths(folder):
    return sorted(p for p in Path(folder).glob("*.zip") if p.is_file())


def game_return(side, record):
    """The objective return of a game record for `side` (lgcommon's table: the league's own rule)."""
    return C.reward(OBJECTIVE[side], record.get("winner"), record.get("route"), record.get("status"))


# The cluster tree no longer matches the native generator's pin (NATIVE_RAW_PIN.json, 10-29): SEARCHCOR (11-01) and SEARCHFIX
# (11-02) patched these five files, and native_env.verify would refuse every raw game.  Read 11-02 (diff against
# searchcor/backup_tree, whose ai.py, reply_evaluation.py and reward_contract.py hash to the pin): search terminals, the FPU
# flag (default off), the reply table flag (default off); own_search.py and reply_selection.py are search-only modules.  None of
# it is on the raw-play path.  Only these EXACT contents are accepted; any other change to any pinned file still refuses,
# and so does a different DLL.
SEARCH_PATCHES = {
    # ai.py: 11-03 cluster tree SEARCHCOR + SEARCHFIX + VALUENET + ROOTSEARCH (search code only, default off; was 5bebafb1);
    #        11-10 + GENSEARCH (search code, and a combat-round tap that is None outside the exact-chance search).  The
    #        second hash was admitted on 11-13 after a replay: the generator on that build re-played the 11-02 run's first
    #        3,000 games; 2,998 ended identically (vlsame.py, runs/explore_20261113/SAME.json).  The 2 others are at the rate
    #        two runs of ONE build differ by float noise between batches (11-02 exact A/A control: 2,016 of 2,020).
    "ai.py": ("f4f9764c74fb2405f1a78e4e6d859a063f3ec8092b942107ba726389d34ff903",
              "4050dc5d1923bd51304fe61cb509ebcedcae5d062eb94fdeea8a987cd447ddb5"),
    "own_search.py": "ab44a4c82ac0e995420a634409fc7e3478bea6f0258b4fed92a18f2a33b5ad9b",
    "reply_evaluation.py": "d6b7c26a707b3f35f831bfedc93c22b90388c9b23d962297fa1fbdd690d480b6",
    "reply_selection.py": "2b44fb7158ad4ae1096996ecd15ba509bbf95d1e64240b178f85f4ef205be5f8",
    "reward_contract.py": "a79214d27e8f607001912401dcd2b953b85f6f4f22eb2d21dd3a1cb1a3fa4a30",
}


def allow_search_patches():
    """native_env.verify, accepting SEARCH_PATCHES on top of the pin in this process (a parent or a spawned worker)."""
    import native_env
    verify = native_env.verify
    if getattr(verify, "valuelab_patches", False):
        return

    def checked(ai, pin):
        src = Path(ai.__file__).resolve().parent
        pin = dict(pin, engine_sha256=dict(pin["engine_sha256"]))
        for name, sha in SEARCH_PATCHES.items():
            got = native_env.sha256_file(src / name)
            if got in ((sha,) if isinstance(sha, str) else sha):
                pin["engine_sha256"][name] = got
        return verify(ai, pin)
    checked.valuelab_patches = True
    native_env.verify = checked


def load_ai():
    """The engine module with the tree on sys.path (a cwd-preserving import, as the league does)."""
    C.setup_paths()
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    return ai
