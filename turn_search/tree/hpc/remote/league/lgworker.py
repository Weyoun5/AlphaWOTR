"""League recorder worker: lockstep multi-matchup raw self-play on the native engine (hpc/remote/league, 2026-10-30).

The batched generator's machinery (hpc/remote/gen/lockstep.py: native_env.NativeRawGame per game, one PolicyActor.
predict_many per net per step on this process's device, each game's own sampler) extended to a LEAGUE:
  * every game has its own pair of nets (an FP snapshot and an SH snapshot); a step scores the pending decisions of all
    live games grouped by NET (one predict_many per net), so one worker plays any mix of pairings;
  * games run under the league's 40-turn cap (native max_turns; a capped game is a timeout);
  * for every LEARNING side of a game the worker records, per decision, what a full-network actor-critic needs: the
    observation, the FULL legal menu (sparse exact encodings + regions), the played row, the behaviour policy's
    tempered log-probabilities (row and whole menu), the behaviour value head, side / turn / phase / ply
    (lgshard.py schema); per game: the outcome, victory route, turns, strongholds / cities captured per side, VP,
    corruption, the nets and seeds;
  * for the hybrid update (the race's last-layer policy gradient, pg_fit) every recorded decision also carries
    batchgen --gradient's evidence, computed by the generator's own tap machinery (gen/lockstep.py split_taps /
    gather_taps / device_centred on valuedata.LastLayer hooks): the centred activations of the two last layers, the
    played kind, the behaviour mass per kind, the untempered played log-probability, argmax flag, and the other
    side's view (valuedata "watch") - the forward pass is shared with play, so this costs no extra forward;
  * finished games go into a packed shard (lgshard.ShardWriter), flushed atomically every --flush-mb / --flush-games;
    only flushed games exist, so an abandoned or killed worker loses only games not yet flushed (replayed on resume
    under the same seeds, never counted twice).
A native refusal drops that game LOUDLY ("native refusal" line + a 'refusal' message the node agent counts; the driver
aborts a cycle above a refusal budget): the Python replay of play_native records no menus, so it cannot stand in.
"""
import collections
import contextlib
import os
from pathlib import Path
import queue as queue_module
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
TIMERS = ("setup", "engine", "forward", "record", "write", "load")


class Slot:
    __slots__ = ("item", "game", "pending", "decisions", "ply", "counts", "t0", "stats", "extra", "views")

    def __init__(self, item, game):
        self.item, self.game = item, game
        self.pending = None
        self.stats = None          # the per-game statistics collector (job["stats_module"]), read-only observer
        self.extra = None          # (kind codes, watch) of the pending decision, for the collector
        self.decisions = {"fp": [], "sh": []}
        self.views = {"fp": [], "sh": []}   # a learning side's own view at the OTHER side's decisions (value head)
        self.ply = 0
        self.counts = {"fp": 0, "sh": 0}
        self.t0 = time.time()


class LeagueWorker:
    def __init__(self, spec, base, channel, stop, abandon, counter):
        self.spec, self.base = spec, base
        self.channel, self.stop, self.abandon, self.counter = channel, stop, abandon, counter
        self.parent_pid = os.getppid()
        self.orphaned = False
        self.actors = collections.OrderedDict()
        self.reset_stats()

    def reset_stats(self):
        self.t = dict.fromkeys(TIMERS, 0.0)
        self.n = dict(games=0, decisions=0, recorded=0, refusals=0, abandoned=0, shards=0, actor_loads=0)

    def send(self, *message):
        self.channel.put(message)

    # ------------------------------------------------------------------------------------------------ setup
    def setup_process(self):
        started = time.perf_counter()
        spec, base = self.spec, self.base
        tree = Path(base["tree"])
        for p in (str(tree), str(tree / "claude_bugscan_2026-10-21" / "native_w2"), str(HERE), str(HERE.parent / "gen")):
            if p not in sys.path:
                sys.path.insert(0, p)
        # flux run -> mpibind binds every worker's main thread to the same place (hpc/README.md): dropped before torch
        for name in ("OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
            os.environ.pop(name, None)
        os.environ["OMP_PROC_BIND"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        cpus = spec.get("cpus")
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        from selfplay_actor_critic.generation import pin_worker_generation, require_generation
        pin_worker_generation(base["generation"])
        require_generation("league worker")
        os.environ["WOTR_NATIVE"] = "0"
        if not spec["device"].startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        sys.path.insert(0, str(tree))
        sys.path.insert(0, str(tree / "src"))
        cwd = os.getcwd()
        import ai
        os.chdir(cwd)
        import torch
        torch.set_num_threads(1)
        import game_logic
        import native_env
        from selfplay_actor_critic.offline_eval import _load_actor
        from selfplay_actor_critic.policy import ActorInput
        from selfplay_actor_critic.probability_schema import WIDTH
        import lgcommon
        import lgshard
        import lockstep
        from selfplay_actor_critic.valuedata import LastLayer
        self.lockstep, self.LastLayer = lockstep, LastLayer
        self.ai, self.torch, self.game_logic, self.native_env = ai, torch, game_logic, native_env
        self._load_actor, self.ActorInput, self.chance_width = _load_actor, ActorInput, WIDTH
        self.lgcommon, self.lgshard = lgcommon, lgshard
        if spec["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("worker %s: device %s requested but torch sees no GPU" % (spec["index"], spec["device"]))
            torch.cuda.set_device(torch.device(spec["device"]))
        self.native = native_env.load(ai)
        torch.zeros(4) @ torch.zeros(4)
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        self.fp_sett, self.sh_sett = game_logic._vp_settlement_tables()
        self.zero = {}
        self.t["setup"] = time.perf_counter() - started

    def load_job(self, job):
        self.job = job
        self.items = job["items"]
        self.ckpts = job["ckpts"]
        self.T = float(job["temperature"])
        self.max_turns = int(job["max_turns"])
        self.record_fraction = float(job.get("record_fraction", 1.0))
        stem = "n%d_w%03d_%s" % (int(job["node"]), int(self.spec["index"]), job["session"])
        self.writer = self.lgshard.ShardWriter(job["out"], stem, dict(cycle=job["cycle"], node=job["node"],
                                                                      worker=self.spec["index"], session=job["session"]))
        self.gradient = bool(job.get("gradient", True))
        self.flush_bytes = int(float(job.get("flush_mb", 96)) * 2 ** 20)
        self.flush_games = int(job.get("flush_games", 64))
        # optional per-game statistics collector (agent F's gamestats.py): module.new_game(**info) -> an object with
        # observe(...) per decision and finish(final_state, ...) -> dict at the end; it only READS (the game, its RNG
        # and every decision are untouched); its errors are recorded in the game record, never raised
        self.stats_mod = None
        name = job.get("stats_module")
        if name:
            import importlib
            self.stats_mod = importlib.import_module(name)

    # ------------------------------------------------------------------------------------------------ nets
    def actor(self, cid):
        a = self.actors.get(cid)
        if a is not None:
            self.actors.move_to_end(cid)
            return a
        t0 = time.perf_counter()
        folder, side = self.ckpts[cid]
        a = self._load_actor(self.ai, folder, side)
        model = a.model
        for name in ("kind_bias", "kind_ctx_bias", "credit_head"):
            if getattr(model, name, None) is not None:
                raise RuntimeError("league net %s carries %s: not supported (inputs carry no kinds)" % (cid, name))
        chance = getattr(model, "chance_policy", None)
        if chance is not None:
            last = [m for m in chance.modules() if isinstance(m, self.torch.nn.Linear)][-1]
            if bool(self.torch.any(last.weight != 0)) or bool(self.torch.any(last.bias != 0)):
                raise RuntimeError("league net %s has an active chance head (frozen at zero in the league)" % cid)
        if self.spec["device"].startswith("cuda"):
            device = self.torch.device(self.spec["device"])
            model.to(device)
            a.device = device
        if self.spec.get("max_rows"):
            a.max_action_rows = int(self.spec["max_rows"])
        self.actors[cid] = a
        self.n["actor_loads"] += 1
        while len(self.actors) > int(self.base.get("actor_cache", 24)):
            self.actors.popitem(last=False)
        self.t["load"] += time.perf_counter() - t0
        return a

    def taps_for(self, actor):
        """batchgen --gradient's taps on this net (the inputs of the scorer's and the policy delta's final linear
        layers), attached once per loaded net; None when the job records no gradient evidence."""
        if not self.gradient:
            return None
        taps = getattr(actor, "taps", None)
        if taps is None:
            taps = actor.taps = [self.LastLayer(actor.model.spatial_action_scorer), self.LastLayer(actor.model.policy_delta)]
        layout = [t.width for t in taps]
        if self.writer.meta.get("layout") not in (None, layout):
            raise RuntimeError("league nets with different last-layer widths in one job")
        self.writer.meta["layout"] = layout
        return taps

    def zeros(self, n):
        """All-zero chance rows (exact for a zero chance head), as a view of ONE growing buffer (a cache per menu size
        held thousands of arrays per worker: ~90 GB per node after a cycle, 2026-10-30 test)."""
        z = self.zero.get("buf")
        if z is None or len(z) < n:
            z = self.zero["buf"] = np.zeros((max(n, 2 * (0 if z is None else len(z)), 1024), self.chance_width), np.float32)
        return z[:n]

    # ------------------------------------------------------------------------------------------------ seeds
    def claim(self):
        if self.stop.is_set():
            return None
        if self.parent_pid and os.getppid() != self.parent_pid:      # the node agent died: claim nothing new
            self.orphaned = True
            return None
        with self.counter.get_lock():
            i = self.counter.value
            if i >= len(self.items):
                return None
            self.counter.value = i + 1
        return self.items[i]

    def start(self, item):
        t0 = time.perf_counter()
        try:
            game = self.native_env.NativeRawGame(self.native, self.ai, int(item[0]), max_turns=self.max_turns)
        except self.native_env.NativeRefusal as error:
            self.refusal(item, 0, error)
            return None
        finally:
            self.t["engine"] += time.perf_counter() - t0
        slot = Slot(item, game)
        if self.stats_mod is not None:
            try:
                slot.stats = self.stats_mod.new_game(seed=int(item[0]), fp=item[1], sh=item[2], learn_fp=item[3],
                                                     learn_sh=item[4], ai=self.ai, first_turn=getattr(game, "first_turn", 1))
            except Exception as error:  # noqa: BLE001 - statistics never stop a game
                slot.stats = None
                self.stats_errors = getattr(self, "stats_errors", 0) + 1
                self.last_stats_error = repr(error)[:300]
        return slot

    def refusal(self, item, rows, error):
        self.n["refusals"] += 1
        print("[league] native refusal: seed %s after %d decisions (%s); game DROPPED (counted against the cycle's "
              "refusal budget)" % (item[0], rows, error), file=sys.__stdout__, flush=True)
        self.send("refusal", self.spec["index"], int(item[0]), str(error)[:500])

    def drop(self, slot, error):
        try:
            slot.game.close()
        except Exception:  # noqa: BLE001
            pass
        slot.pending = None
        self.refusal(slot.item, slot.ply, error)

    # ------------------------------------------------------------------------------------------------ loop
    def run(self):
        G = int(self.spec["lockstep"])
        T = self.T
        NativeRefusal = self.native_env.NativeRefusal
        clock = time.perf_counter
        slots, exhausted = [], False
        parent_checked = clock()
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            while True:
                if self.abandon.is_set():
                    break
                if self.parent_pid and clock() - parent_checked > 10.0:
                    parent_checked = clock()
                    if os.getppid() != self.parent_pid:
                        self.orphaned = True
                        break
                while len(slots) < G and not exhausted:
                    item = self.claim()
                    if item is None:
                        exhausted = True
                        break
                    slot = self.start(item)
                    if slot is not None:
                        slots.append(slot)
                if not slots:
                    break
                groups, done = collections.OrderedDict(), []
                t0 = clock()
                for slot in slots:
                    game = slot.game
                    try:
                        if not game.next():
                            done.append(slot)
                            continue
                        obs, _watch, cand, regions, _kinds = game.encode()
                        slot.extra = (_kinds, _watch)
                    except NativeRefusal as error:
                        self.drop(slot, error)
                        continue
                    side = game.side
                    cid = self.actor_id(slot, side)
                    slot.pending = (self.ActorInput(obs, cand, regions, self.zeros(len(cand))), side, cid)
                    groups.setdefault(cid, []).append(slot)
                self.t["engine"] += clock() - t0
                for slot in done:
                    self.finish(slot)
                slots = [s for s in slots if s.pending is not None]
                for cid, batch in groups.items():
                    actor = self.actor(cid)
                    taps = self.taps_for(actor)
                    if taps:
                        for tap in taps:
                            tap.parts.clear()
                    t0 = clock()
                    predictions = actor.predict_many([s.pending[0] for s in batch])
                    t1 = clock()
                    self.t["forward"] += t1 - t0
                    choices = [self.sample(slot, prediction, T) for slot, prediction in zip(batch, predictions)]
                    centred = [None] * len(batch)
                    if taps and any(c["learner"] for c in choices):
                        centred = self.centred(actor, batch, choices)
                    elif taps:
                        for tap in taps:
                            tap.parts.clear()
                    for slot, prediction, choice, row in zip(batch, predictions, choices, centred):
                        self.choose(slot, prediction, T, choice, row)
                    self.t["record"] += clock() - t1
                survivors = []
                t0 = clock()
                for slot in slots:
                    selected = slot.pending
                    slot.pending = None
                    try:
                        slot.game.apply(selected)
                    except NativeRefusal as error:
                        self.drop(slot, error)
                        continue
                    survivors.append(slot)
                self.t["engine"] += clock() - t0
                self.n["decisions"] += len(slots)
                slots = survivors
        for slot in slots:                     # abandoned in flight: not written (replayed on resume)
            self.n["abandoned"] += 1
            try:
                slot.game.close()
            except Exception:  # noqa: BLE001
                pass
        self.flush()

    def actor_id(self, slot, side):
        """The net that plays `side` at the open decision of this game: the item's.  (The value lab's exploration games
        hand one stretch of a game to another net of the same side: valuelab/vlworker.py.)"""
        return slot.item[1] if side == "fp" else slot.item[2]

    def sample(self, slot, prediction, T):
        """play_native's sampling with this game's own sampler; for a learning side also the behaviour policy
        softmax(logp / T) the --gradient terms are centred with (play_native's arithmetic)."""
        inputs, side, cid = slot.pending
        game = slot.game
        selected = prediction.sample(game.samplers[side], T)
        learner = slot.item[3] if side == "fp" else slot.item[4]
        out = dict(selected=int(selected), learner=learner, record=False)
        if learner and (self.record_fraction >= 1.0 or self.lgcommon.keep_decision(int(slot.item[0]), slot.ply, "rec", 0,
                                                                                   self.record_fraction)):
            logp = np.asarray(prediction.log_probabilities, np.float64)
            scaled = logp / T
            behaviour = np.exp(scaled - scaled.max())
            behaviour /= behaviour.sum()
            m = scaled.max()
            out.update(record=True, logp=logp, behaviour=behaviour, mu=scaled - (m + np.log(np.exp(scaled - m).sum())))
        else:
            out["learner"] = None
        return out

    def centred(self, actor, batch, choices):
        """batchgen's --gradient centred terms h(sel) - behaviour @ h of the recorded decisions of this batch
        (lockstep.split_taps on the host for CPU workers, gather_taps + device_centred on the GPU for GPU workers)."""
        ls = self.lockstep
        counts = [len(s.pending[0].candidates) for s in batch]
        out = [None] * len(batch)
        if self.spec["device"].startswith("cuda"):
            stacked = ls.gather_taps(actor.taps, counts)
            keep = [i for i, c in enumerate(choices) if c["record"]]
            offs = np.concatenate([[0], np.cumsum(counts)])
            parts = [stacked[int(offs[i]):int(offs[i + 1])] for i in keep]
            import torch
            rows = ls.device_centred(torch.cat(parts), [counts[i] for i in keep], [choices[i]["selected"] for i in keep],
                                     [choices[i]["behaviour"] for i in keep])
            for i, r in zip(keep, rows):
                out[i] = np.asarray(r, np.float32)
        else:
            hidden = ls.split_taps(actor.taps, counts)
            for i, (h, c) in enumerate(zip(hidden, choices)):
                if c["record"]:
                    if len(h) != len(c["behaviour"]):
                        raise RuntimeError("Last-layer activations do not match the menu")
                    out[i] = (h[c["selected"]] - c["behaviour"] @ h).astype(np.float32)
        return out

    def choose(self, slot, prediction, T, choice=None, centred=None):
        inputs, side, cid = slot.pending
        game = slot.game
        if choice is None:
            choice = self.sample(slot, prediction, T)
        selected = choice["selected"]
        other = "sh" if side == "fp" else "fp"
        if self.gradient and (slot.item[3] if other == "fp" else slot.item[4]):
            # the opponent of the mover learns in this game: keep ITS view of this position (the watch vector)
            vi, vv = self.lgshard.sparse_vector(slot.extra[1])
            slot.views[other].append(dict(idx=vi, val=vv, turn=int(game.turn), ply=slot.ply))
        if choice["record"]:
            mu, logp, behaviour = choice["mu"], choice["logp"], choice["behaviour"]
            kinds, watch = slot.extra
            x = dict(ply=slot.ply, side=0 if side == "fp" else 1, turn=int(game.turn), phase=str(game.phase),
                     sel=int(selected), logp_sel=float(mu[selected]), value=float(prediction.value),
                     obs=inputs.observation, cand=inputs.candidates, region=inputs.regions, mu_logp=mu)
            if centred is not None:
                vocab = game.vocab
                codes = np.asarray(kinds, np.int64)
                mass = np.bincount(codes, weights=behaviour, minlength=len(vocab))
                x.update(watch=watch, centred=centred, kind_played=vocab[int(codes[selected])],
                         kind_mass=[(vocab[int(c)], float(mass[c])) for c in np.flatnonzero(mass)],
                         played_logp=float(logp[selected]), top=bool(int(selected) == int(np.argmax(logp))))
            sx = self.lgshard.sparsify(x)
            sx["phase"] = x["phase"]
            slot.decisions[side].append(sx)
            self.n["recorded"] += 1
        if slot.stats is not None and not getattr(slot.stats, "failed", None):
            try:
                kinds, watch = slot.extra
                vocab = game.vocab
                slot.stats.observe(side=side, turn=int(game.turn), phase=str(game.phase), ply=slot.ply,
                                   selected=int(selected), selected_kind=vocab[int(kinds[selected])],
                                   kinds=kinds, vocab=vocab, n_rows=len(inputs.candidates), obs=inputs.observation,
                                   watch=watch, candidates=inputs.candidates, game=game)
            except Exception as error:  # noqa: BLE001
                try:
                    slot.stats.failed = repr(error)[:300]
                except Exception:  # noqa: BLE001
                    slot.stats = None
        slot.counts[side] += 1
        slot.ply += 1
        slot.pending = selected

    def finish(self, slot):
        ai, game, item = self.ai, slot.game, slot.item
        t0 = time.perf_counter()
        try:
            end = game.end
            final = game.state(ai)
        except self.native_env.NativeRefusal as error:
            self.drop(slot, error)
            return
        finally:
            try:
                game.close()
            except Exception:  # noqa: BLE001
                pass
        if end is None:
            raise RuntimeError("seed %s: the native game ended without an accounted end" % item[0])
        vp = self.game_logic.calculate_victory_points(final)
        sh_ctrl = final.enemy_controlled.get(ai.gs.Side.SHADOW, ())       # FP settlements the Shadow controls
        fp_ctrl = final.enemy_controlled.get(ai.gs.Side.FREE_PEOPLES, ())  # Shadow settlements FP controls
        fel = final.fellowship
        seed = int(item[0])
        record = dict(seed=seed, fp=item[1], sh=item[2], learn=dict(fp=item[3] or None, sh=item[4] or None),
                      status=end.status, winner=end.winner, route=end.victory_route, reason=end.reason,
                      turns=int(getattr(final, "turn", 0) or 0), first_turn=int(getattr(game, "first_turn", 1) or 1),
                      final_turn=None if game.final_turn is None else int(game.final_turn),
                      decisions=dict(slot.counts), fp_vp=int(vp["free_peoples"]), sh_vp=int(vp["shadow"]),
                      fp_strongholds=sum(1 for k, v in self.sh_sett if v == 2 and k in fp_ctrl),
                      sh_strongholds=sum(1 for k, v in self.fp_sett if v == 2 and k in sh_ctrl),
                      fp_cities=sum(1 for k, v in self.sh_sett if v == 1 and k in fp_ctrl),
                      sh_cities=sum(1 for k, v in self.fp_sett if v == 1 and k in sh_ctrl),
                      corruption=int(getattr(fel, "corruption", 0) or 0), progress=int(getattr(fel, "progress", 0) or 0),
                      mordor_step=int(getattr(fel, "mordor_step", 0) or 0), holdout=bool(self.lgcommon.holdout(seed)),
                      wall_s=round(time.time() - slot.t0, 2), worker=int(self.spec["index"]), node=int(self.job["node"]))
        decisions, views = {}, {}
        for side in ("fp", "sh"):
            learner = item[3] if side == "fp" else item[4]
            if learner:
                decisions[learner] = slot.decisions[side]
                if self.gradient:
                    views[learner] = slot.views[side]
        if slot.stats is not None:
            try:
                if getattr(slot.stats, "failed", None):
                    record["stats_error"] = slot.stats.failed
                else:
                    record["stats"] = slot.stats.finish(final, seed=seed, record=dict(record), ai=ai)
            except Exception as error:  # noqa: BLE001
                record["stats_error"] = repr(error)[:300]
        self.t["engine"] += time.perf_counter() - t0
        self.writer.add_game(record, decisions, views)
        self.n["games"] += 1
        if self.writer.bytes >= self.flush_bytes or len(self.writer) >= self.flush_games:
            self.flush()

    def flush(self):
        if not len(self.writer):
            return
        t0 = time.perf_counter()
        row = self.writer.flush()
        self.t["write"] += time.perf_counter() - t0
        self.n["shards"] += 1
        self.send("shard", self.spec["index"], row)

    def report(self):
        memory = {}
        try:
            for line in open("/proc/self/status"):
                if line.split(":")[0] in ("VmRSS", "VmHWM"):
                    memory[line.split(":")[0]] = int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return dict(index=self.spec["index"], device=self.spec["device"], seconds=dict(self.t), counts=dict(self.n),
                    memory_mb=memory, actors_cached=len(self.actors))


def serve_main(spec, base, jobs, channel, stop, abandon, counter):
    """Persistent worker (lgnode.py): process setup once, then one job per message until None.
    Messages: ready, shard, refusal, stats (with the job id), error."""
    worker = LeagueWorker(spec, base, channel, stop, abandon, counter)
    try:
        worker.setup_process()
        worker.send("ready", spec["index"], worker.t["setup"])
        while True:
            try:
                job = jobs.get(timeout=5.0)
            except queue_module.Empty:
                if os.getppid() != worker.parent_pid:
                    break
                continue
            if job is None:
                break
            worker.reset_stats()
            worker.load_job(job)
            worker.run()
            if worker.orphaned:
                break
            worker.send("stats", spec["index"], dict(worker.report(), job=job.get("job")))
    except BaseException:  # noqa: BLE001 - reported to the parent, then the process fails
        worker.send("error", spec["index"], traceback.format_exc())
        raise SystemExit(1)
