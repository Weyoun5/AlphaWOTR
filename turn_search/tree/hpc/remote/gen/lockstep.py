"""Lockstep multi-game worker of the batched raw-game generator (batchgen.py starts these processes).

One process plays up to G games at once on the native engine (native_w2's native_env.NativeRawGame), each game exactly
as native_valuedata.play_native plays it, interleaved:

  every step:  advance each live game to its next learned decision (game.next, game.encode, chance rows)
               group the pending decisions by side (the fp and sh actors differ)
               score each group with ONE PolicyActor.predict_many call on this process's device (a GPU or the CPU)
               per game, in order: Prediction.sample with THAT game's own samplers[side] at the run temperature, the
               --gradient terms from this game's slice of the batch's last-layer activations, the stored rows, apply
  a finished game is written (the same two files play_native writes) and its slot gets the next unclaimed seed.

Per game the sequence of engine calls, RNG draws and stored values is play_native's; only the policy forward is
shared.  G=1 is byte-identical to gamegen_native.py; with G>1 the CPU forward is not batch-invariant (~1e-7), so
games agree to a tolerance (test_batchgen.py measures how many diverge).

A game whose native step is refused (native_env.NativeRefusal) is replayed by valuedata.play, as play_native does
(one loud line containing "native refusal"); nothing is dropped silently.

Seeds are claimed one at a time from the parent's shared list (a lock-protected counter), so a fast GPU-forward worker
takes more seeds than a CPU-forward one; every seed is played by exactly one worker.
"""
import contextlib
import json
import queue as queue_module
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
_STATE = {}
TIMERS = ("setup", "engine", "forward", "taps", "glue", "write", "fallback")


def split_taps(taps, counts):
    """The batch's last-layer activations, split back per decision.

    taps: the side's valuedata.LastLayer objects (each holds the outputs of every forward chunk of the last
    predict_many call: rows = all candidates of the batch, in input order).  Returns one float64 array per decision,
    [count_i, sum of tap widths], the columns in tap order (what play_native's np.concatenate([tap.take() ...], axis=1)
    gives for that decision alone).  The row total of every tap must equal sum(counts): anything else is a bug and
    raises loudly.  The parts are cleared."""
    import torch
    total = int(sum(counts))
    blocks = []
    for k, tap in enumerate(taps):
        if not tap.parts:
            raise RuntimeError(f"gradient tap {k}: no activations captured for a batch of {len(counts)} decisions")
        h = torch.cat(tap.parts)
        tap.parts.clear()
        if h.ndim != 2 or h.shape[0] != total or h.shape[1] != tap.width:
            raise RuntimeError(f"gradient tap {k}: captured {tuple(h.shape)} rows x width, the batch has {total} "
                               f"candidates over {len(counts)} decisions (counts {list(counts)[:8]}...) x width "
                               f"{tap.width}")
        # float32 -> float64 is exact, on either device; one host copy per tap per batch
        blocks.append(h.cpu().numpy().astype(np.float64))
    hidden = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    out, offset = [], 0
    for count in counts:
        out.append(hidden[offset:offset + count])
        offset += count
    if offset != hidden.shape[0] or any(len(h) != c for h, c in zip(out, counts)):
        raise RuntimeError(f"gradient tap split: {offset} rows assigned of {hidden.shape[0]}")
    return out


def gather_taps(taps, counts):
    """The batch's last-layer activations as ONE tensor on the forward's device, [sum(counts), sum of tap widths]
    (float32, not copied to the host), with split_taps' loud length checks.  The parts are cleared."""
    import torch
    total = int(sum(counts))
    blocks = []
    for k, tap in enumerate(taps):
        if not tap.parts:
            raise RuntimeError(f"gradient tap {k}: no activations captured for a batch of {len(counts)} decisions")
        h = torch.cat(tap.parts)
        tap.parts.clear()
        if h.ndim != 2 or h.shape[0] != total or h.shape[1] != tap.width:
            raise RuntimeError(f"gradient tap {k}: captured {tuple(h.shape)} rows x width, the batch has {total} "
                               f"candidates over {len(counts)} decisions x width {tap.width}")
        blocks.append(h)
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)


def device_centred(stacked, counts, selected, behaviours):
    """play_native's centred terms, (h[selected] - behaviour @ h) in float64 then float32, for every decision of the
    batch, computed on the device that holds the activations: only [decisions, width] crosses to the host instead of
    every candidate row.  Same arithmetic in float64 (a GPU sums in another order than numpy: ~1e-16 relative; GPU runs
    are tolerance-compared anyway).  One small matmul per decision (deterministic, no atomics)."""
    import torch
    device = stacked.device
    h64 = stacked.double()
    weights = torch.from_numpy(np.concatenate(behaviours)).to(device)
    rows, offset = [], 0
    for count, chosen in zip(counts, selected):
        block = h64[offset:offset + count]
        rows.append(block[int(chosen)] - weights[offset:offset + count] @ block)
        offset += count
    return list(torch.stack(rows).float().cpu().numpy())


class Slot:
    """One live game and its accumulators (play_native's locals)."""
    __slots__ = ("seed", "folder", "game", "rows", "observations", "watch", "centred", "kind_played", "kind_mass",
                 "played_logp", "top", "kind_index", "names", "local", "pending")

    def __init__(self, seed, folder, game):
        self.seed, self.folder, self.game = seed, folder, game
        self.rows, self.observations, self.watch = [], [], []
        self.centred, self.kind_played, self.kind_mass, self.played_logp, self.top = [], [], [], [], []
        self.kind_index = {}
        self.names = np.empty(0, dtype=object)     # native vocab code -> kind name (this game's vocab)
        self.local = np.empty(0, np.int64)          # native vocab code -> index in kind_index (-1: not seen yet)
        self.pending = None

    def kind_names(self, kinds):
        """tuple(game.vocab[int(k)] for k in kinds), vectorised."""
        vocab = self.game.vocab
        if len(vocab) != len(self.names):
            self.names = np.array(vocab, dtype=object)
            self.local = np.concatenate([self.local, np.full(len(vocab) - len(self.local), -1, np.int64)])
        return tuple(self.names[kinds].tolist())

    def game_kinds(self, kinds):
        """np.fromiter((kind_index.setdefault(k, len(kind_index)) for k in kind_names), ...), vectorised: new kinds get
        their index in order of first appearance, as setdefault assigns them."""
        local = self.local[kinds]
        if (local < 0).any():
            new = kinds[local < 0]
            _, first = np.unique(new, return_index=True)
            for code in new[np.sort(first)]:
                name = self.names[code]
                index = self.kind_index.get(name)
                if index is None:
                    index = self.kind_index[name] = len(self.kind_index)
                self.local[code] = index
            local = self.local[kinds]
        return local


class Worker:
    def __init__(self, spec, common, channel, stop, abandon, counter):
        self.spec, self.common = spec, common
        self.channel, self.stop, self.abandon, self.counter = channel, stop, abandon, counter
        self.seeds = common.get("seeds", [])
        self.loaded = None                  # (policies, gradient, manifest identity) of the actors in memory
        self.parent_pid = os.getppid()      # an orphaned worker (its agent / launcher died) stops instead of lingering
        self.orphaned = False
        self.reset_stats()

    def reset_stats(self):
        self.t = dict.fromkeys(TIMERS, 0.0)
        self.n = dict(games=0, fallbacks=0, decisions=0, abandoned=0, abandoned_decisions=0, steps=0)
        self.forward = {s: dict(calls=0, decisions=0, rows=0, max_batch=0) for s in ("fp", "sh")}
        self.steady = dict(seconds=0.0, decisions=0)
        # where "engine" goes (no double counting: these are parts of t["engine"])
        self.sub = dict(opening=0.0, next_encode=0.0, chance=0.0, apply=0.0, final=0.0)
        self.marks = []        # (epoch, cumulative decisions, cumulative games) every ~2 s: the node rate in any window

    def send(self, *message):
        self.channel.put(message)

    # ---- setup: play_native's _STATE block, plus the device
    def setup(self):
        """One-shot worker: the process setup and this run's actors."""
        started = time.perf_counter()
        self.setup_process()
        self.load_job(self.common)
        self.t["setup"] = time.perf_counter() - started

    def setup_process(self):
        """Everything that does not depend on the policies: paths, affinity, generation, engine modules, native engine."""
        spec, common = self.spec, self.common
        tree = Path(common["tree"])
        native_dir = tree / "claude_bugscan_2026-10-21" / "native_w2"
        for p in (str(tree), str(native_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        # A launcher's OpenMP binding (LLNL flux run -> mpibind: OMP_PLACES / OMP_PROC_BIND) makes the preloaded libomp
        # bind every worker's main thread to the FIRST place when torch starts OpenMP: all workers on a few CPUs (the
        # 2026-10-29 race smoke ran 25x slow).  Single-threaded workers need no OpenMP binding: dropped before torch loads.
        for name in ("OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
            os.environ.pop(name, None)
        os.environ["OMP_PROC_BIND"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        cpus = spec.get("cpus")
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        from selfplay_actor_critic.generation import pin_worker_generation, require_generation
        pin_worker_generation(common["generation"])
        require_generation("valuedata.play")
        os.environ["WOTR_NATIVE"] = "0"
        if not spec["device"].startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # '-1', never '' (Windows); as play_native
        sys.path.insert(0, str(tree))
        sys.path.insert(0, str(tree / "src"))
        cwd = os.getcwd()
        import ai
        os.chdir(cwd)
        import torch
        torch.set_num_threads(1)
        from selfplay_actor_critic import valuedata
        from selfplay_actor_critic.contracts import CONTRACT, ContractError
        from selfplay_actor_critic.durability import digest_file, episode_folder
        from selfplay_actor_critic.policy import ActorInput
        from selfplay_actor_critic.runtime_io import _atomic_write, write_json
        import native_env
        self.ai, self.torch = ai, torch
        self.valuedata, self.native_env, self.ContractError = valuedata, native_env, ContractError
        self.CONTRACT, self.ActorInput = CONTRACT, ActorInput
        self.digest_file, self.episode_folder, self._atomic_write, self.write_json = (digest_file, episode_folder,
                                                                                     _atomic_write, write_json)
        if spec["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"worker {spec['index']}: device {spec['device']} requested but torch sees no GPU "
                                   f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
            torch.cuda.set_device(torch.device(spec["device"]))
        self.native = native_env.load(ai)
        torch.zeros(4) @ torch.zeros(4)                 # OpenMP / BLAS initialised now, then the affinity re-applied
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        self.affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        self.refuse_at = _test_refusals()
        # --gradient terms reduced on the GPU for GPU workers (unless --gpu-taps host); CPU workers: play_native's numpy
        self.device_taps = ((spec["device"].startswith("cuda") and spec.get("taps", "device") == "device")
                            or os.environ.get("BATCHGEN_TEST_DEVICE_TAPS") == "1")      # tests: the GPU path on a CPU

    def load_job(self, common):
        """This run's actors (play_native's behaviour-content checks, loads and taps), kept while the policies stay."""
        spec, ai, torch = self.spec, self.ai, self.torch
        from selfplay_actor_critic.durability import behavior_content
        from selfplay_actor_critic.offline_eval import _load_actor
        import native_valuedata
        self.common, self.seeds = common, common["seeds"]
        policies, manifest = common["policies"], common["manifest"]
        if {s: behavior_content(policies[s], s) for s in ("fp", "sh")} != manifest["behavior_content"]:
            raise self.ContractError("Behavior weights changed after collection admission")
        key = (json.dumps(policies, sort_keys=True), bool(common["gradient"]), manifest["identity"])
        if self.loaded == key:
            return
        self.actors = self.taps = None
        actors = {s: _load_actor(ai, policies[s], s) for s in ("fp", "sh")}
        if {s: behavior_content(policies[s], s) for s in ("fp", "sh")} != manifest["behavior_content"]:
            raise self.ContractError("Behavior weights changed while loading actors")
        taps = ({s: [self.valuedata.LastLayer(actors[s].model.spatial_action_scorer),
                     self.valuedata.LastLayer(actors[s].model.policy_delta)] for s in ("fp", "sh")}
                if common["gradient"] else {})
        self.chance_rows = native_valuedata.ChanceRows(ai, actors)
        if spec["device"].startswith("cuda"):
            device = torch.device(spec["device"])
            for actor in actors.values():
                actor.model.to(device)
                actor.device = device
                if spec.get("max_rows"):
                    actor.max_action_rows = int(spec["max_rows"])
        elif spec.get("max_rows"):
            for actor in actors.values():
                actor.max_action_rows = int(spec["max_rows"])
        self.actors, self.taps = actors, taps
        self.loaded = key

    # ---- seeds
    def claim(self):
        if self.stop.is_set():
            return None
        with self.counter.get_lock():
            i = self.counter.value
            if i >= len(self.seeds):
                return None
            self.counter.value = i + 1
        return int(self.seeds[i])

    def task(self, seed):
        """The valuedata/play_native task tuple of a new game (for the Python replay)."""
        c = self.common
        return (c["output"], seed, c["temperature"], self.spec["index"], c["policies"], c["gradient"], None, None,
                c["manifest"], None)

    def start(self, seed):
        started = time.perf_counter()
        folder = self.episode_folder(self.common["output"], seed)
        if (folder / "episode.json").exists():          # written meanwhile (never by this run: the parent scheduled
            raise self.ContractError(f"seed {seed}: episode appeared during collection: {folder}")  # only missing seeds)
        try:
            game = self.native_env.NativeRawGame(self.native, self.ai, seed)
        except self.native_env.NativeRefusal as error:
            self.t["engine"] += time.perf_counter() - started
            self.fallback(seed, 0, error)
            return None
        self.t["engine"] += time.perf_counter() - started
        self.sub["opening"] += time.perf_counter() - started
        return Slot(seed, folder, game)

    def fallback(self, seed, rows, error):
        """play_native's except branch: the whole game replayed on the Python generator (valuedata.play)."""
        print(f"[native-raw] seed {seed}: native refusal after {rows} learned decisions ({error}); "
              f"game replayed on the Python generator", file=sys.__stdout__, flush=True)
        started = time.perf_counter()
        # valuedata.play caches its actors per run identity in valuedata._STATE and refuses another identity; a
        # persistent worker (race) plays a new run (manifest) every step, so the cache is dropped before each replay
        # (it reloads this run's actors on the CPU: the cost of a first replay).  Review S2-3, 2026-10-29.
        self.valuedata._STATE.clear()
        seed_, status, count, _ = self.valuedata.play(self.task(seed))
        self.t["fallback"] += time.perf_counter() - started
        self.n["fallbacks"] += 1
        self.n["games"] += 1
        self.send("game", self.spec["index"], seed_, status, count, True)

    # ---- the lockstep loop
    def run(self):
        G = int(self.spec["lockstep"])
        single = self.spec.get("score") == "single"
        T = float(self.common["temperature"])
        gradient = bool(self.common["gradient"])
        slots, exhausted = [], False
        NativeRefusal = self.native_env.NativeRefusal
        actors, taps = self.actors, self.taps
        clock = time.perf_counter
        loop_started = parent_checked = clock()
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            while True:
                if self.abandon.is_set():
                    break
                if self.parent_pid and clock() - parent_checked > 10.0:
                    parent_checked = clock()
                    if os.getppid() != self.parent_pid:      # the agent / launcher died: stop (games not written)
                        self.orphaned = True
                        break
                step_started = clock()
                while len(slots) < G and not exhausted:
                    seed = self.claim()
                    if seed is None:
                        exhausted = True
                        break
                    slot = self.start(seed)
                    if slot is not None:
                        slots.append(slot)
                if not slots:
                    break
                full = len(slots) == G and not exhausted
                groups = {"fp": [], "sh": []}
                done = []
                for slot in slots:
                    game = slot.game
                    t0 = clock()
                    try:
                        if not game.next():
                            self.t["engine"] += clock() - t0
                            self.sub["next_encode"] += clock() - t0
                            done.append(slot)
                            continue
                        obs, other_view, candidates, regions, kinds = game.encode()
                        t1 = clock()
                        names = slot.kind_names(kinds)
                        t2 = clock()
                        actor = actors[game.side]
                        probabilities = self.chance_rows(game, actor, len(candidates))
                        t3 = clock()
                        if self.refuse_at and self.refuse_at.get(slot.seed) == len(slot.rows):
                            raise NativeRefusal("batchgen test hook BATCHGEN_TEST_REFUSE")
                    except NativeRefusal as error:
                        self.t["engine"] += clock() - t0
                        self.drop(slot, error)
                        continue
                    inputs = self.ActorInput(obs, candidates, regions, probabilities, kinds=names)
                    slot.pending = (inputs, kinds, obs, other_view)
                    groups[game.side].append(slot)
                    t4 = clock()
                    self.t["engine"] += (t1 - t0) + (t3 - t2)
                    self.sub["next_encode"] += t1 - t0
                    self.sub["chance"] += t3 - t2
                    self.t["glue"] += (t2 - t1) + (t4 - t3)
                for slot in done:
                    self.finish(slot)
                slots = [s for s in slots if s.pending is not None]
                decided = 0
                for side in ("fp", "sh"):
                    batch = groups[side]
                    if not batch:
                        continue
                    actor = actors[side]
                    parts = [batch] if not single else [[s] for s in batch]
                    for part in parts:
                        t0 = clock()
                        for group in taps.values():
                            for tap in group:
                                tap.parts.clear()
                        predictions = actor.predict_many([s.pending[0] for s in part])
                        t1 = clock()
                        counts = [len(s.pending[0].candidates) for s in part]
                        if gradient and self.device_taps:
                            stacked, hidden = gather_taps(taps[side], counts), None
                        else:
                            hidden = split_taps(taps[side], counts) if gradient else None
                        t2 = clock()
                        stats = self.forward[side]
                        stats["calls"] += 1
                        stats["decisions"] += len(part)
                        stats["rows"] += sum(counts)
                        stats["max_batch"] = max(stats["max_batch"], len(part))
                        self.t["forward"] += t1 - t0
                        self.t["taps"] += t2 - t1
                        choices = [self.choose(slot, prediction, T, gradient) for slot, prediction in zip(part, predictions)]
                        t3 = clock()
                        if not gradient:
                            centred = [None] * len(part)
                        elif hidden is not None:
                            centred = []
                            for (selected, logp, behaviour), h in zip(choices, hidden):
                                if len(h) != len(logp):
                                    raise RuntimeError("Last-layer activations do not match the menu")
                                centred.append((h[selected] - behaviour @ h).astype(np.float32))
                        else:
                            if any(c != len(choice[1]) for c, choice in zip(counts, choices)):
                                raise RuntimeError("Last-layer activations do not match the menu")
                            centred = device_centred(stacked, counts, [c[0] for c in choices], [c[2] for c in choices])
                        t4 = clock()
                        for slot, choice, row in zip(part, choices, centred):
                            self.record(slot, *choice, row, gradient)
                        self.t["taps"] += t4 - t3
                        self.t["glue"] += (t3 - t2) + (clock() - t4)
                    decided += len(batch)
                # apply (engine), in slot order
                survivors = []
                for slot in slots:
                    selected = slot.pending
                    slot.pending = None
                    t0 = clock()
                    try:
                        slot.game.apply(selected)
                    except NativeRefusal as error:
                        self.t["engine"] += clock() - t0
                        self.drop(slot, error)
                        continue
                    self.t["engine"] += clock() - t0
                    self.sub["apply"] += clock() - t0
                    survivors.append(slot)
                slots = survivors
                self.n["decisions"] += decided
                self.n["steps"] += 1
                now = time.time()
                if not self.marks or now - self.marks[-1][0] >= 2.0:
                    self.marks.append((now, self.n["decisions"], self.n["games"]))
                if full:
                    self.steady["seconds"] += clock() - step_started
                    self.steady["decisions"] += decided
        self.marks.append((time.time(), self.n["decisions"], self.n["games"]))
        for slot in slots:                                  # --time-limit: in-flight games abandoned (not written)
            self.n["abandoned"] += 1
            self.n["abandoned_decisions"] += len(slot.rows)
            slot.game.close()
        self.loop_seconds = clock() - loop_started

    def choose(self, slot, prediction, T, gradient):
        """play_native's sampling (this game's own sampler) and the behaviour policy of the --gradient terms."""
        game = slot.game
        selected = prediction.sample(game.samplers[game.side], T)
        logp = np.asarray(prediction.log_probabilities, np.float64)
        behaviour = None
        if gradient:
            scaled = logp / T
            behaviour = np.exp(scaled - scaled.max())
            behaviour /= behaviour.sum()
        return selected, logp, behaviour

    def record(self, slot, selected, logp, behaviour, centred, gradient):
        """play_native's per-decision stores after the forward; leaves the selected index in slot.pending."""
        inputs, kinds, obs, other_view = slot.pending
        game = slot.game
        count = len(inputs.candidates)
        if gradient:
            game_kinds = slot.game_kinds(kinds)
            slot.centred.append(centred)
            slot.kind_played.append(int(game_kinds[selected]))
            slot.kind_mass.append(np.bincount(game_kinds, behaviour))
            slot.played_logp.append(float(logp[selected]))
            slot.top.append(int(selected) == int(np.argmax(logp)))
        slot.observations.append(np.asarray(obs, np.float16))
        slot.watch.append(np.asarray(other_view, np.float16))
        slot.rows.append(dict(side=game.side, turn=int(game.turn), phase=str(game.phase), selected=int(selected),
                              kind="learned", action_count=count, searched=False, search_value=None))
        slot.pending = selected

    def drop(self, slot, error):
        rows = len(slot.rows)
        slot.game.close()
        slot.pending = None
        self.fallback(slot.seed, rows, error)

    def finish(self, slot):
        """play_native from `end = game.end` on (write=True, no --full-side, no --chance-log)."""
        ai, CONTRACT, game = self.ai, self.CONTRACT, slot.game
        t0 = time.perf_counter()
        try:
            end = game.end
            final_state = game.state(ai)
        except self.native_env.NativeRefusal as error:
            self.t["engine"] += time.perf_counter() - t0
            self.drop(slot, error)
            return
        finally:
            game.close()
        if end is None:
            raise self.ContractError("Environment ended without an accounted episode end")
        sides = {"fp": ai.gs.Side.FREE_PEOPLES, "sh": ai.gs.Side.SHADOW}
        finals = {f"final_{s}": np.asarray(ai.encode_state(final_state, sides[s]), np.float16) for s in ("fp", "sh")}
        t1 = time.perf_counter()
        self.t["engine"] += t1 - t0
        self.sub["final"] += t1 - t0
        rows, gradient, taps = slot.rows, self.common["gradient"], self.taps
        folder = slot.folder
        folder.mkdir(parents=True, exist_ok=True)
        extra = {}
        if gradient and rows:
            mass = np.zeros((len(rows), len(slot.kind_index)), np.float32)
            for i, part in enumerate(slot.kind_mass):
                mass[i, :len(part)] = part
            kind_index = slot.kind_index
            extra.update(centred=np.stack(slot.centred), centred_layout=np.array([tap.width for tap in taps["sh"]], np.int32),
                         kind_played=np.asarray(slot.kind_played, np.int16), kind_mass=mass,
                         played_logp=np.asarray(slot.played_logp, np.float32), top=np.asarray(slot.top, bool),
                         kind_names=np.array([str(getattr(k, "name", k)) for k in sorted(kind_index, key=kind_index.get)]))
        elif gradient:
            widths = [tap.width for tap in taps["sh"]]
            extra.update(centred=np.empty((0, sum(widths)), np.float32), centred_layout=np.array(widths, np.int32),
                         kind_played=np.empty(0, np.int16), kind_mass=np.empty((0, 0), np.float32),
                         played_logp=np.empty(0, np.float32), top=np.empty(0, bool), kind_names=np.array([], dtype=str))
        extra.update(finals)
        observations, watch = slot.observations, slot.watch
        self._atomic_write(folder / "arrays.npz", lambda stream: np.savez_compressed(stream,
            observations=np.stack(observations) if observations else np.empty((0, CONTRACT.observation_width), np.float16),
            watch=np.stack(watch) if watch else np.empty((0, CONTRACT.observation_width), np.float16), **extra))
        manifest, seed = self.common["manifest"], slot.seed
        meta = dict(schema="value_data_v1", contract_id=CONTRACT.identity, episode_id=f"raw-{seed}", seed=seed,
                    temperature=self.common["temperature"], policies=self.common["policies"], rows=rows,
                    run_identity=manifest["identity"], behavior_content=manifest["behavior_content"],
                    arrays_sha256=self.digest_file(folder / "arrays.npz"),
                    end=dict(status=end.status, transition_count=len(rows), winner=end.winner, victory_route=end.victory_route, reason=end.reason,
                             engine_transitions=end.transition_count))
        self.write_json(folder / "episode.json", meta)
        self.t["write"] += time.perf_counter() - t1
        self.n["games"] += 1
        self.send("game", self.spec["index"], seed, "ok" if end.status == "terminated" else end.status, len(rows), False)

    def report(self):
        maxrss = None
        try:
            import resource
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KB -> MB
        except Exception:  # noqa: BLE001 - Windows has no resource module
            pass
        memory = {}
        try:                                    # Linux: private (anonymous) vs file-backed (shared libraries) resident
            for line in open("/proc/self/status"):
                if line.split(":")[0] in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM"):
                    memory[line.split(":")[0]] = int(line.split()[1]) / 1024.0
        except OSError:
            pass
        if self.spec["device"].startswith("cuda") and getattr(self, "torch", None) is not None:
            try:
                memory["gpu_max_allocated"] = self.torch.cuda.max_memory_allocated() / 2 ** 20
                memory["gpu_max_reserved"] = self.torch.cuda.max_memory_reserved() / 2 ** 20
            except Exception:  # noqa: BLE001
                pass
        loop = getattr(self, "loop_seconds", 0.0)
        busy = sum(self.t[k] for k in TIMERS if k != "setup")
        return dict(spec={k: v for k, v in self.spec.items() if k != "cpus"}, cpus=len(self.spec.get("cpus") or []),
                    affinity=len(getattr(self, "affinity", []) or []),
                    pid=os.getpid(), seconds=dict(self.t, loop=loop, other=max(0.0, loop - busy)), counts=dict(self.n),
                    forward=self.forward, steady=self.steady, maxrss_mb=maxrss, memory_mb=memory,
                    engine_detail=dict(self.sub), marks=self.marks)


def _test_refusals():
    """BATCHGEN_TEST_REFUSE='seed:decision,...' (tests only): raise NativeRefusal at that learned decision of that
    seed, to exercise the Python replay.  Never set in production."""
    text = os.environ.get("BATCHGEN_TEST_REFUSE", "").strip()
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        seed, decision = item.split(":")
        out[int(seed)] = int(decision)
    return out


def serve_main(spec, base, jobs, channel, stop, abandon, counter):
    """Persistent worker (race_node.py): the process setup once, then one job per message on `jobs` (a common dict
    with its own run folder, manifest, policies and seeds; the parent resets `counter` and the events before each),
    until None.  Messages: ready, game, stats (with the job id), error."""
    worker = Worker(spec, base, channel, stop, abandon, counter)
    try:
        started = time.perf_counter()
        worker.setup_process()
        worker.send("ready", spec["index"], time.perf_counter() - started)
        while True:
            try:
                job = jobs.get(timeout=5.0)
            except queue_module.Empty:
                if os.getppid() != worker.parent_pid:       # the agent died (review S2-1): exit, so flux run ends
                    break
                continue
            if job is None:
                break
            worker.reset_stats()
            started = time.perf_counter()
            worker.load_job(job)
            worker.t["setup"] = time.perf_counter() - started
            worker.marks.append((time.time(), 0, 0))
            worker.run()
            if worker.orphaned:
                break
            worker.send("stats", spec["index"], dict(worker.report(), job=job.get("job")))
    except BaseException:  # noqa: BLE001 - every failure is reported to the parent, then the process fails
        worker.send("error", spec["index"], traceback.format_exc())
        raise SystemExit(1)


def worker_main(spec, common, channel, stop, abandon, counter):
    """Process entry point (spawn).  Messages to the parent: ready, game (one per written game), stats, error."""
    worker = Worker(spec, common, channel, stop, abandon, counter)
    try:
        worker.setup()
        worker.marks.append((time.time(), 0, 0))
        worker.send("ready", spec["index"], worker.t["setup"])
        worker.run()
        worker.send("stats", spec["index"], worker.report())
    except BaseException:  # noqa: BLE001 - every failure is reported to the parent, then the process fails
        worker.send("error", spec["index"], traceback.format_exc())
        try:
            worker.send("stats", spec["index"], worker.report())
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1)
