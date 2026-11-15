"""Regenerate the states and menus of stored games by replaying them.

A valuedata.py game stores, per decision with more than one option, the mover's observation and the index of
the move played (rows[k]["selected"]); forced moves are applied but not stored. The engine's chance is a
function of the game seed alone (environment.GameRandom), so replaying the same seed and the same choices
reproduces the game: every regenerated observation is checked against the stored one (float16, exact), the
menu size against rows[k]["action_count"], and the end against episode.json.

What a policy step needs from a decision is its menu under the policy that played: for every legal action the
activations feeding the two policy heads' final layers (the action scorer's 64, the policy_delta head's 128)
and the old log-probability. New final-layer weights then give the new logits as
    logit'(a) = logit(a) + (w_s' - w_s) . h_s(a) + (w_d' - w_d) . h_d(a)
with no further forward pass, and the new log-probabilities by renormalising over the menu (ppo_fit.py).

The cache: <run>/menus/<side>/<seed>.npz per game (float16 activations, float32 log-probabilities), one side's
decisions only, so both sides' caches of one run coexist. Menus run to thousands of actions (median 52, mean
600) but at the play temperature the mass outside the 32 most probable is below 0.1%, so each decision keeps
its --top-k actions (default 48) plus the played one and the log of the mass left out, at T=1 and at the play temperature. This bounded cache supports
the all-action fitter's documented tail approximation. PPO needs --top-k 0 to retain every action because
the fitted shared weights also change omitted actions. The default takes about 2.5 MB per game for the
Shadow; --decisions-per-game keeps a random subset of the side's decisions to bound the disk further.
Complete menus are large (5,000 Shadow games at 48 decisions each: about 51 GB of activations), so ppo_fit reads
them through MenuBatch.load_streamed, which memory-maps the activations instead of loading them.

  python -m selfplay_actor_critic.replay_menus --run selfplay_actor_critic/runs/pg3_g5_a --side sh --max-games 8 --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import ContractError, canonical_json
from .durability import behavior_content
from .menu_cache import load_menu, menu_path, menu_request

ROOT = Path(__file__).resolve().parents[1]
_STATE = {}
HEADS = ("spatial_action_scorer", "policy_delta")


def game_folders(run, max_games=0):
    from .event_trace import game_folders as _folders
    return _folders(run, max_games)


def load_episode(folder):
    meta = json.loads(Path(folder, "episode.json").read_text(encoding="utf-8"))
    with np.load(Path(folder, "arrays.npz")) as d:
        observations = d["observations"]
        played_logp = d["played_logp"] if "played_logp" in d.files else None
    return meta, observations, played_logp


def _setup(policies):
    """The engine and the actors, once per worker process."""
    behavior = {s: behavior_content(policies[s], s) for s in ("fp", "sh")}
    if "ai" in _STATE and _STATE.get("behavior") == behavior:
        return _STATE
    from .generation import require_generation
    require_generation("replay_menus._setup")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", WOTR_NATIVE="0")  # '-1', never '' (Windows)
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    import torch
    torch.set_num_threads(1)
    from .offline_eval import _load_actor
    from .valuedata import LastLayer
    _STATE["ai"] = ai
    _STATE["policies"] = policies
    _STATE["actors"] = {s: _load_actor(ai, policies[s], s) for s in ("fp", "sh")}
    if {s: behavior_content(policies[s], s) for s in ("fp", "sh")} != behavior:
        raise ContractError("Replay actors changed while loading")
    _STATE["behavior"] = behavior
    _STATE["taps"] = {s: [LastLayer(getattr(_STATE["actors"][s].model, head)) for head in HEADS] for s in ("fp", "sh")}
    return _STATE


def replay(ai, meta, observations, on_decision=None, *, check=True):
    """Replay one stored game. on_decision(k, frame, selected) is called at every stored decision before the
    move is applied. Raises on any mismatch with the stored game when check is set. Returns the EpisodeEnd."""
    from .environment import PythonEnvironment
    rows = meta["rows"]
    seed = int(meta["seed"])
    env = PythonEnvironment(ai, f"raw-{seed}", seed)
    k = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        while True:
            frame = env.prepare()
            if frame is None:
                break
            if len(frame.actions) == 1:
                env.apply(frame, 0)
                continue
            if k >= len(rows):
                raise RuntimeError(f"seed {seed}: the replay reaches a decision {k} the stored game does not have")
            row = rows[k]
            if check:
                if frame.side != row["side"] or int(frame.turn) != int(row["turn"]) or len(frame.actions) != int(row["action_count"]):
                    raise RuntimeError(f"seed {seed} decision {k}: replay {frame.side} turn {frame.turn} {len(frame.actions)} actions, "
                                       f"stored {row['side']} turn {row['turn']} {row['action_count']}")
                regenerated = np.asarray(frame.inputs.observation, np.float16)
                if not np.array_equal(regenerated, observations[k]):
                    bad = np.flatnonzero(regenerated != observations[k])
                    raise RuntimeError(f"seed {seed} decision {k}: {len(bad)} observation features differ (first {bad[:5].tolist()})")
            selected = int(row["selected"])
            if on_decision is not None:
                on_decision(k, frame, selected)
            env.apply(frame, selected)
            k += 1
    if check:
        end = env.end
        stored = meta["end"]
        if k != len(rows) or end is None or end.status != stored["status"] or end.winner != stored["winner"]:
            raise RuntimeError(f"seed {seed}: replay ended {None if end is None else (end.status, end.winner)} after {k} decisions, "
                               f"stored ({stored['status']}, {stored['winner']}) after {len(rows)}")
    return env.end


def menu_of(actor, taps, inputs):
    """(h [actions, 192] float32, logp [actions] float64) of one decision's menu under the actor."""
    for tap in taps:
        tap.parts.clear()
    prediction = actor.predict(inputs)
    h = np.concatenate([tap.take() for tap in taps], axis=1)
    logp = np.asarray(prediction.log_probabilities, np.float64)
    if len(h) != len(logp):
        raise RuntimeError("Last-layer activations do not match the menu")
    return h, logp


def _logsumexp(x):
    if len(x) == 0:
        return -np.inf
    m = float(np.max(x))
    return float(np.log(np.exp(x - m).sum()) + m)


def top_actions(logp, selected, top_k, temperature):
    """The indices (sorted) of the top_k most probable actions plus the played one, and the log of the mass
    left out at T=1 and at the play temperature (-inf when nothing is left out)."""
    if top_k <= 0 or len(logp) <= top_k:
        return np.arange(len(logp)), -np.inf, -np.inf
    order = np.argsort(-logp, kind="stable")
    kept = order[:top_k]
    if selected not in kept:
        kept = np.append(kept, selected)
    out = np.setdiff1d(np.arange(len(logp)), kept)
    return np.sort(kept), _logsumexp(logp[out]), _logsumexp(logp[out] / temperature)


def cache_path(run, seed, side):
    """One side's menu cache of one game (menu_cache.menu_path: the side is part of the key)."""
    return menu_path(run, side, seed)


def build_one(task):
    """Replay one game and write its menu cache for `side`. Returns (seed, status, decisions cached, seconds)."""
    folder, run, side, policies, per_game, check, top_k, temperature = task
    started = time.perf_counter()
    request = menu_request(task)
    meta, observations, played_logp = load_episode(folder)
    seed = int(meta["seed"])
    path = cache_path(run, seed, side)
    if path.exists():
        try:
            cached = load_menu(path, run=run, side=side, seed=seed)
            if json.loads(str(cached["request"])) == request:
                return seed, "exists", 0, 0.
        except ContractError:
            pass  # Replace stale/legacy caches atomically after a checked rebuild.
    state = _setup(policies)
    ai, actor, taps = state["ai"], state["actors"][side], state["taps"][side]
    own = [k for k, r in enumerate(meta["rows"]) if r["side"] == side]
    keep = set(own)
    if per_game and len(own) > per_game:
        keep = set(np.random.default_rng(seed * 13 + 5).choice(own, per_game, replace=False).tolist())
    parts = dict(decision=[], own_index=[], selected=[], count=[], menu=[], h=[], logp=[], tail1=[], tailT=[])
    own_index = {k: i for i, k in enumerate(own)}
    mismatch = []

    def on_decision(k, frame, selected):
        if k not in keep:
            return
        h, logp = menu_of(actor, taps, frame.inputs)
        if played_logp is not None and abs(float(logp[selected]) - float(played_logp[k])) > 2e-3:
            mismatch.append((k, float(logp[selected]), float(played_logp[k])))
        kept, tail1, tailT = top_actions(logp, selected, top_k, temperature)
        parts["decision"].append(k)
        parts["own_index"].append(own_index[k])
        parts["selected"].append(int(np.flatnonzero(kept == selected)[0]))        # position of the played move among the kept
        parts["count"].append(len(kept))
        parts["menu"].append(len(logp))
        parts["h"].append(h[kept].astype(np.float16))
        parts["logp"].append(logp[kept].astype(np.float32))
        parts["tail1"].append(tail1)
        parts["tailT"].append(tailT)
    try:
        replay(ai, meta, observations, on_decision, check=check)
    except RuntimeError as error:
        return seed, f"mismatch: {error}", 0, time.perf_counter() - started
    if mismatch:
        k, new, old = mismatch[0]
        return seed, f"logp mismatch at decision {k}: replayed {new:.4f} stored {old:.4f} ({len(mismatch)} decisions)", 0, time.perf_counter() - started
    if menu_request(task) != request:
        raise ContractError("Replay menu inputs changed during construction")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, decision=np.asarray(parts["decision"], np.int32), own_index=np.asarray(parts["own_index"], np.int32),
             selected=np.asarray(parts["selected"], np.int32), count=np.asarray(parts["count"], np.int32),
             menu=np.asarray(parts["menu"], np.int32), tail1=np.asarray(parts["tail1"], np.float32), tailT=np.asarray(parts["tailT"], np.float32),
             offsets=np.concatenate([[0], np.cumsum(parts["count"])]).astype(np.int64),
             h=np.concatenate(parts["h"]) if parts["h"] else np.zeros((0, 192), np.float16),
             logp=np.concatenate(parts["logp"]) if parts["logp"] else np.zeros(0, np.float32),
             own_decisions=np.int32(len(own)), side=side, seed=np.int64(seed), temperature=np.float64(temperature), top_k=np.int32(top_k),
             request=np.array(canonical_json(request)))
    os.replace(temporary, path)
    return seed, "ok", len(parts["decision"]), time.perf_counter() - started


ACTIVATIONS = ("h", "logp")          # per kept action: memory-mapped by MenuBatch.load_streamed, never read eagerly
DECISION_COLUMNS = dict(own_index=np.int32, selected=np.int32, count=np.int32, menu=np.int32,
                        tail1=np.float32, tailT=np.float32)
H_WIDTH = 192                         # the two policy heads' final-layer inputs (64 + 128)


def available_memory():
    """Bytes of physical memory the operating system reports available now."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                        "virtual_total", "virtual_available", "extended")]
        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.available)
    return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def _mapped_members(stream, names):
    """Memory-map the named arrays of the np.savez archive open in `stream`: read-only, nothing read into memory.
    np.savez stores its members uncompressed; each member's bytes are checked against the archive's CRC-32 (what
    np.load checks while reading) and its .npy header against its size. The arrays keep the mapping alive (the
    mapping holds its own handle, so the stream may be closed)."""
    import io
    import math
    import mmap
    import struct
    import zipfile
    import zlib
    with zipfile.ZipFile(stream) as archive:
        try:
            infos = {name: archive.getinfo(name + ".npy") for name in names}
        except KeyError as error:
            raise ContractError(f"Replay menu has no {error} member; rebuild with replay_menus") from None
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    arrays = {}
    try:
        for name, info in infos.items():
            if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size or info.flag_bits & 1:
                raise ContractError(f"Replay menu member {name} is not stored uncompressed (np.savez), so it cannot be "
                                    "memory-mapped; rebuild with replay_menus")
            local = mapping[info.header_offset:info.header_offset + 30]
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise ContractError(f"Replay menu member {name} has a corrupt zip header; rebuild with replay_menus")
            start = info.header_offset + 30 + sum(struct.unpack("<HH", local[26:30]))
            end = start + info.file_size
            if end > len(mapping):
                raise ContractError(f"Replay menu member {name} is truncated; rebuild with replay_menus")
            with memoryview(mapping) as whole, whole[start:end] as member:
                crc = zlib.crc32(member)
            if crc != info.CRC:
                raise ContractError(f"Replay menu member {name} fails its CRC-32 check (corrupt cache); rebuild with replay_menus")
            header = io.BytesIO(mapping[start:min(end, start + (1 << 16))])
            version = np.lib.format.read_magic(header)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(header)
            else:
                raise ContractError(f"Replay menu member {name} has .npy version {version}; rebuild with replay_menus")
            offset = start + header.tell()
            if dtype.hasobject or offset + dtype.itemsize * math.prod(shape) != end:
                raise ContractError(f"Replay menu member {name}'s header does not match its size; rebuild with replay_menus")
            arrays[name] = np.ndarray(shape, dtype, buffer=mapping, offset=offset, order="F" if fortran else "C")
    except BaseException:
        arrays.clear()
        mapping.close()
        raise
    return arrays


class RaggedRows:
    """Per-game arrays addressed as one flat, read-only array of rows (the parts are memory-mapped menu caches).
    Indexing with an integer, a slice, an integer array or a boolean mask gathers only those rows into memory;
    np.asarray() materialises the whole array (one preallocated copy). It has shape, dtype, ndim, nbytes, len."""

    def __init__(self, parts, *, dtype, row_shape):
        self.parts, self.dtype, self.row_shape = list(parts), np.dtype(dtype), tuple(row_shape)
        for part in self.parts:
            if part.dtype != self.dtype or tuple(part.shape[1:]) != self.row_shape:
                raise ContractError(f"Replay menu activations disagree across games: {part.dtype} {part.shape[1:]} "
                                    f"against {self.dtype} {self.row_shape}; rebuild with replay_menus")
        self.starts = np.concatenate([[0], np.cumsum([len(p) for p in self.parts])]).astype(np.int64)
        self.shape = (int(self.starts[-1]),) + self.row_shape
        self.ndim = len(self.shape)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"RaggedRows(shape={self.shape}, dtype={self.dtype}, parts={len(self.parts)})"

    def _rows(self, index):
        n = self.shape[0]
        if isinstance(index, slice):
            return np.arange(*index.indices(n), dtype=np.int64)
        if isinstance(index, tuple):
            raise IndexError("RaggedRows is indexed by rows only")
        rows = np.asarray(index)
        if rows.dtype == bool:
            if rows.shape != (n,):
                raise IndexError(f"boolean index of shape {rows.shape} for {n} rows")
            return np.flatnonzero(rows)
        if not np.issubdtype(rows.dtype, np.integer):
            raise IndexError("RaggedRows takes integers, slices, integer arrays or a boolean mask")
        rows = np.where(rows < 0, rows + n, rows).astype(np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= n):
            raise IndexError(f"row index out of range for {n} rows")
        return rows

    def __getitem__(self, index):
        rows = self._rows(index)
        flat = rows.reshape(-1)
        out = np.empty((len(flat),) + self.row_shape, self.dtype)
        if len(flat):
            part = np.searchsorted(self.starts, flat, side="right") - 1      # empty parts are never chosen
            order = np.argsort(part, kind="stable")
            for group in np.split(order, np.flatnonzero(np.diff(part[order])) + 1):
                g = int(part[group[0]])
                out[group] = self.parts[g][flat[group] - self.starts[g]]
        out = out.reshape(rows.shape + self.row_shape)
        return out[()] if out.ndim == 0 else out

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("RaggedRows cannot be converted to an array without a copy")
        out = np.empty(self.shape, self.dtype)
        for part, start in zip(self.parts, self.starts):
            out[start:start + len(part)] = part
        return out if dtype is None else out.astype(dtype, copy=False)


class MenuBatch:
    """The cached menus of a batch of games, read lazily per game. load_streamed keeps the per-action activations
    memory-mapped (host memory: the per-decision columns plus what a caller gathers); load_all copies them into
    memory, refusing first when they do not fit."""

    def __init__(self, run, seeds, *, side):
        self.run, self.seeds, self.side = run, [int(s) for s in seeds], side

    def game(self, seed):
        return load_menu(cache_path(self.run, seed, self.side), run=self.run, side=self.side, seed=seed)

    def __iter__(self):
        for seed in self.seeds:
            yield seed, self.game(seed)

    def load_streamed(self, *, complete=False):
        """The batch as flat arrays without reading the activations into memory: h [kept actions, 192] float16 and
        logp [kept actions] float32 as RaggedRows over the admitted caches, memory-mapped (each member checked
        against its CRC-32); per decision, in memory, the game index, own_index, selected (position among the
        kept), the number kept, the full menu size, the left-out log-masses, offsets into the actions; the seeds
        and the temperature. complete=True refuses, before mapping that game's activations, a cache that does
        not keep the whole menu (built without --top-k 0). The caches stay mapped (on Windows: cannot be
        replaced) until the returned h and logp are released."""
        cols = {k: [] for k in ("game", *DECISION_COLUMNS)}
        h, logp = [], []
        temperature = None
        for g, seed in enumerate(self.seeds):
            with open(cache_path(self.run, seed, self.side), "rb") as stream:
                d = load_menu(stream, run=self.run, side=self.side, seed=seed, lazy=ACTIVATIONS)
                count = np.asarray(d["count"], np.int64)
                if complete and (not np.array_equal(count, d["menu"]) or not np.isneginf(d["tailT"]).all()):
                    raise ContractError(f"Replay menu of game {seed} does not keep every action; PPO requires complete "
                                        "action menus: rebuild replay_menus for this run and side with --top-k 0")
                if temperature is not None and float(d["temperature"]) != temperature:
                    raise ContractError(f"Replay menus of one batch were built at temperatures {temperature} and "
                                        f"{float(d['temperature'])}")
                temperature = float(d["temperature"])
                arrays = _mapped_members(stream, ACTIVATIONS)
            total = int(count.sum())
            if (arrays["h"].ndim != 2 or len(arrays["h"]) != total or arrays["logp"].shape != (total,)
                    or ("offsets" in d and not np.array_equal(d["offsets"], np.concatenate([[0], np.cumsum(count)])))):
                raise ContractError(f"Replay menu of game {seed}: activations do not match the kept counts; rebuild with replay_menus")
            h.append(arrays["h"])
            logp.append(arrays["logp"])
            cols["game"].append(np.full(len(d["selected"]), g, np.int64))
            for k in DECISION_COLUMNS:
                cols[k].append(d[k])
        dtypes = dict(game=np.int64, **DECISION_COLUMNS)
        out = {k: np.concatenate(v) if v else np.zeros(0, dtypes[k]) for k, v in cols.items()}
        out.update(h=RaggedRows(h, dtype=h[0].dtype if h else np.float16, row_shape=h[0].shape[1:] if h else (H_WIDTH,)),
                   logp=RaggedRows(logp, dtype=logp[0].dtype if logp else np.float32, row_shape=()),
                   offsets=np.concatenate([[0], np.cumsum(out["count"])]).astype(np.int64),
                   seeds=np.asarray(self.seeds, np.int64), temperature=temperature)
        return out

    def load_all(self, *, memory_budget=None):
        """load_streamed's arrays with h and logp copied into memory, one preallocated array each (about 1x their
        size, never a per-game list plus its concatenation). Refuses loudly, before copying any activation, when
        they need more than `memory_budget` bytes (default: the memory the OS reports available)."""
        out = self.load_streamed()
        need = out["h"].nbytes + out["logp"].nbytes
        budget = available_memory() if memory_budget is None else int(memory_budget)
        if need > budget:
            raise ContractError(f"MenuBatch.load_all needs {need / 1e9:.2f} GB for the activations of {len(out['h'])} kept "
                                f"actions, over the {budget / 1e9:.2f} GB budget; use MenuBatch.load_streamed "
                                "(memory-mapped, as ppo_fit does), fewer games (--max-games) or fewer cached decisions "
                                "per game (replay_menus --decisions-per-game)")
        out["h"] = np.asarray(out["h"])
        out["logp"] = np.asarray(out["logp"])
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a valuedata.py run (its config.json names the policies)")
    parser.add_argument("--side", required=True, choices=("fp", "sh"))
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--decisions-per-game", type=int, default=0, help="cache a random subset of the side's decisions (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-check", action="store_true", help="skip the stored-vs-replayed checks (never for training data)")
    parser.add_argument("--top-k", type=int, default=48, help="actions kept per decision (plus the played one); 0 = the whole menu")
    args = parser.parse_args()
    run = os.path.abspath(args.run)
    config = json.loads(Path(run, "config.json").read_text(encoding="utf-8"))
    policies = {s: config.get(f"{s}_policy", "seed") for s in ("fp", "sh")}
    temperature = float(config.get("temperature", .5))
    folders = game_folders(run, args.max_games)
    tasks = [(f, run, args.side, policies, args.decisions_per_game, not args.no_check, args.top_k, temperature) for f in folders]
    started = time.perf_counter()
    counts = {"ok": 0, "exists": 0, "bad": 0}
    decisions = 0
    from .generation import generation_pool_kwargs
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             **generation_pool_kwargs("replay_menus.main")) as pool:
        for i, (seed, status, n, seconds) in enumerate(pool.map(build_one, tasks, chunksize=2)):
            if status == "ok":
                counts["ok"] += 1; decisions += n
            elif status == "exists":
                counts["exists"] += 1
            else:
                counts["bad"] += 1
                print(f"  seed {seed}: {status}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1} of {len(tasks)} games, {time.perf_counter() - started:.0f} s", flush=True)
    print(f"{run} {args.side}: {counts['ok']} games cached ({decisions} decisions), {counts['exists']} already cached, "
          f"{counts['bad']} mismatched; policies {policies}; {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
