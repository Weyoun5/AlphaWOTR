"""Forced-deviation worker: lgworker.LeagueWorker's process setup, nets, claim, refusal and flush, with its own loop, because one
base game here fans out into several live games (a loop over a fixed slot list cannot grow).

SCAN job (mode A).  A base game plays both sides at T=0 (argmax of the installed pair's raw policy).  At a decision a fork is a
native exact copy of the game (state, its own RNG stream, bookkeeping; native_env world_copy) with the forced row applied; the base
game then plays the top move and so IS the paired baseline.  The arm plays on at T=0 with no further deviation.  Both arms hold the
same stream at the fork; the streams drift apart when the arms consume different numbers of draws (see README).  Nothing here
reads a future: see dvcommon.

ROLLOUT job (mode B).  A chunk replays a position from its seed, verifies it (observation hash, menu size, top row), then runs
rollouts k = k0.. for every arm.  For each k all arms share one world: a clone with a stream that depends on (seed, ply, k), with
what the mover cannot see re-dealt (search.determinize_state, opponent hand and decks, own deck order, hunt pool) by a generator
keyed the same way, then that arm's row.  So a position's rollouts average over the mover's information, not over one hidden deal.
"""
import collections
import contextlib
import os
import sys
import time

import numpy as np

import dvcommon as D
import lgworker
V, C = D.V, D.C


class Arm(lgworker.Slot):
    """One live game: role 'base' | 'alt' | 'handicap' | 'replay' | 'roll'."""


class Record:
    """A base game and its forks: written when the last of them ends."""
    __slots__ = ("seed", "open", "forks", "views", "candidates", "base", "dropped")

    def __init__(self, seed):
        self.seed, self.open = seed, 1
        self.forks, self.views, self.candidates = [], [], []
        self.base, self.dropped = None, False


class Chunk:
    """A block of rollouts of one position."""
    __slots__ = ("id", "item", "open", "rows", "game", "queue", "extra", "views", "lost")

    def __init__(self, item):
        self.id, self.item = item["id"], item
        self.open, self.rows, self.game, self.queue, self.extra, self.views, self.lost = 0, [], None, collections.deque(), 0, [], False


class DeviationWorker(lgworker.LeagueWorker):
    def load_job(self, job):
        self.job = job
        self.items = job["items"]
        self.ckpts = job["ckpts"]
        self.max_turns = int(job["max_turns"])
        if float(job["temperature"]) != 0.0:
            raise RuntimeError("the deviation worker plays T=0 only (see README)")
        stem = "n%d_w%03d_%s" % (int(job["node"]), int(self.spec["index"]), job["session"])
        self.writer = D.ShardWriter(job["out"], stem, dict(node=job["node"], worker=self.spec["index"], session=job["session"], mode=job["mode"]))
        self.mode = job["mode"]
        self.hz = job.get("hazards")
        self.view_keep = float(job.get("view_keep", 0.3))
        self.tag_every = int(job.get("tag_every", 10))
        self.tag_rates = job.get("tag_rates") or {}
        self.view_ks = int(job.get("view_ks", 16))
        self.flush_bytes = int(float(job.get("flush_mb", 96)) * 2 ** 20)
        self.flush_games = int(job.get("flush_games", 128))
        self.n.update(forks=0, fork_refused=0, alias_skipped=0, candidates=0, replay_mismatch=0, arms=0, chunks=0)

    def taps_for(self, actor):
        return None

    # -------------------------------------------------------------------------------------------------------- the loop
    def run(self):
        G = max(1, int(self.spec["lockstep"]))
        NativeRefusal = self.native_env.NativeRefusal
        clock = time.perf_counter
        slots, exhausted = [], False
        self.queue = collections.deque()
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
                # a rollout arm waiting beats a new game: only claim when this worker is idle up to its lockstep
                while len(slots) < G and self.queue:
                    arm = self.spawn_roll(self.queue[0])
                    if arm is not None:
                        slots.append(arm)
                while len(slots) < G and not self.queue and not exhausted:
                    item = self.claim()
                    if item is None:
                        exhausted = True
                        break
                    slot = self.start_item(item)
                    if slot is not None:
                        slots.append(slot)
                if not slots:
                    if self.queue:
                        continue
                    break
                groups, done = collections.OrderedDict(), []
                t0 = clock()
                live = []
                for slot in slots:
                    game = slot.game
                    try:
                        more = game.next()
                        while more and game.forced:         # a clone exposes single-option frames; they are not decisions
                            game.apply(0)
                            more = game.next()
                        if not more:
                            done.append(slot)
                            continue
                        obs, watch, cand, regions, kinds = game.encode()
                    except NativeRefusal as error:
                        self.arm_refused(slot, error)
                        continue
                    slot.extra = (kinds, watch)
                    if slot.views:
                        self.capture(slot, obs, watch, game.side)
                    side = game.side
                    slot.pending = (self.ActorInput(obs, cand, regions, self.zeros(len(cand))), side, side)
                    groups.setdefault(side, []).append(slot)
                    live.append(slot)
                self.t["engine"] += clock() - t0
                for slot in done:
                    self.finish_arm(slot)
                slots = live
                spawned = []
                for cid, batch in groups.items():
                    actor = self.actor(cid)
                    t0 = clock()
                    predictions = actor.predict_many([s.pending[0] for s in batch])
                    t1 = clock()
                    self.t["forward"] += t1 - t0
                    for slot, prediction in zip(batch, predictions):
                        spawned.extend(self.decide(slot, prediction))
                    self.t["record"] += clock() - t1
                survivors = []
                t0 = clock()
                for slot in slots:
                    selected = slot.pending
                    slot.pending = None
                    if selected is None:                # a replay that reached its position: it stays open, unapplied
                        continue
                    try:
                        slot.game.apply(selected)
                    except NativeRefusal as error:
                        self.arm_refused(slot, error)
                        continue
                    survivors.append(slot)
                self.t["engine"] += clock() - t0
                self.n["decisions"] += len(slots)
                slots = survivors + spawned
        for slot in slots:                     # abandoned in flight: not written (replayed on resume)
            self.n["abandoned"] += 1
            try:
                slot.game.close()
            except Exception:  # noqa: BLE001
                pass
        for entry in self.queue:
            if entry["chunk"].game is not None:
                entry["chunk"].game.close()
                entry["chunk"].game = None
        self.flush()

    # ----------------------------------------------------------------------------------------------------- starting
    def new_game(self, seed):
        return self.native_env.NativeRawGame(self.native, self.ai, int(seed), max_turns=self.max_turns)

    def start_item(self, item):
        t0 = time.perf_counter()
        try:
            game = self.new_game(item["seed"])
        except self.native_env.NativeRefusal as error:
            self.refusal([item["seed"]], 0, error)
            return None
        finally:
            self.t["engine"] += time.perf_counter() - t0
        arm = Arm([item["seed"]], game)
        arm.views = []
        arm.trace, arm.tracks, arm.mover, arm.handicap = 0, [], None, 0
        if self.mode == "scan":
            arm.role, arm.rec = "base", Record(int(item["seed"]))
            arm.sel = D.selector(item["seed"])
            arm.tag_scan = self.tag_every > 0 and int(item["seed"]) % self.tag_every == 0
        else:
            arm.role, arm.chunk = "replay", Chunk(item)
        return arm

    def spawn_roll(self, entry):
        """One rollout of a queued chunk (lazy, so a CPU worker holds one world at a time)."""
        chunk, arm_id, row, k = entry["chunk"], entry["arm"], entry["row"], entry["k"]
        self.queue.popleft()
        item = chunk.item
        t0 = time.perf_counter()
        out = None
        try:
            base = chunk.game
            world = base.world_clone(D.stream_state(item["seed"], item["ply"], k), chunk.extra)
            try:
                world.world_determinize(D.SIDE_CODE[item["side"]], D.determinize_rng(item["seed"], item["ply"], k))
                world.world_apply_from(base, row)
            except BaseException:
                world.close()
                raise
            out = Arm([item["seed"]], world)
            out.role, out.chunk, out.arm_id, out.k = "roll", chunk, arm_id, k
            out.mover, out.handicap, out.trace, out.tracks = item["side"], 0, 0, []
            out.ply = int(item["ply"]) + 1
            out.views = [("roll", k)] if int(item.get("stage", 1)) == 1 and k < self.view_ks else []
            chunk.open += 1
            self.n["arms"] += 1
        except self.native_env.NativeRefusal:
            chunk.rows.append(dict(pid=item["pid"], arm=arm_id, k=k, status=D.REFUSED, winner=-1, route=0, end_turn=0, end_ply=0, trace=0))
            self.n["fork_refused"] += 1
        finally:
            self.t["engine"] += time.perf_counter() - t0
        if entry["last"]:
            chunk.game.close()
            chunk.game = None
            if chunk.open == 0:
                self.write_chunk(chunk)
        return out

    # ----------------------------------------------------------------------------------------------------- deciding
    def decide(self, slot, prediction):
        inputs, side, _cid = slot.pending
        game = slot.game
        lp = np.asarray(prediction.log_probabilities, np.float64)
        top = int(np.argmax(lp))
        selected = top
        spawned = []
        if slot.role == "base":
            spawned = self.scan_decision(slot, prediction, lp, top)
        elif slot.role == "replay":
            item = slot.chunk.item
            if slot.ply == int(item["ply"]):
                self.begin_rollouts(slot, inputs, top)
                slot.pending = None
                return []
        elif slot.handicap > 0 and side == slot.mover:
            selected = D.lowest_eligible(lp, top)
            slot.handicap -= 1
        rows = len(inputs.candidates)
        slot.trace = D.trace_step(slot.trace, rows, selected)
        for row in slot.tracks:
            row["base_trace"] = D.trace_step(row.get("base_trace", 0), rows, selected)
        slot.counts[side] += 1
        slot.ply += 1
        slot.pending = selected
        return spawned

    def scan_decision(self, slot, prediction, lp, top):
        inputs, side, _cid = slot.pending
        game = slot.game
        seed, ply = int(slot.item[0]), slot.ply
        kinds, watch = slot.extra
        u = slot.sel.random(D.N_U)
        forks, is_cand, keep_view, tb = D.plan_decision(seed, ply, side, lp, top, kinds, game.vocab, self.hz, u, self.view_keep)
        tags, cand_tags = 0, 0
        p_top = float(np.exp(lp[top]))
        top_kind = game.vocab[int(kinds[top])]
        if p_top >= D.TAG_MIN_P and top_kind in ("pass", "play_event", "army_move") and (forks or is_cand or slot.tag_scan):
            t0 = time.perf_counter()
            tags = D.named_tags(game, self.ai, side, game.turn, game.first_turn, game.phase, top, p_top, top_kind)
            self.t["record"] += time.perf_counter() - t0
            cand_tags = tags if slot.tag_scan else 0   # only tag-scan games offer named positions: their rate is known
        spawned = []
        if not forks and not is_cand and not cand_tags:
            return spawned
        rec = slot.rec
        p = np.exp(lp)
        order = None
        fork_view = None
        if keep_view:
            fork_view = V.sparse(inputs.observation)
        for plan in forks:
            alt = plan["alt"]
            stream = plan["stream"]
            if stream in (D.BAND, D.KIND) and np.array_equal(inputs.candidates[alt], inputs.candidates[top]) and int(inputs.regions[alt]) == int(inputs.regions[top]):
                self.n["alias_skipped"] += 1        # the same move under another row: not a deviation
                continue
            if order is None:
                order = np.argsort(-lp, kind="stable")
            fid = len(rec.forks)
            row = dict(seed=seed, ply=ply, turn=int(game.turn), phase=D.PHASE_CODES[str(game.phase)], side=D.SIDE_CODE[side], stream=stream,
                       status=D.OK, top_band=tb, alt_band=plan["alt_band"], group_top=D.group_code(game.vocab[int(kinds[top])]),
                       group_alt=D.group_code(game.vocab[int(kinds[alt])]), kind_top_name=game.vocab[int(kinds[top])],
                       kind_alt_name=game.vocab[int(kinds[alt])], rows=len(lp), elig=int((p >= self.hz["floor"]).sum() - 1),
                       rank=int(np.flatnonzero(order == alt)[0]), top=top, alt=alt, p_top=float(p[top]), p_alt=float(p[alt]),
                       hazard=plan["hazard"], value=float(prediction.value), tags=tags, fid=fid, winner=-1, route=0, end_turn=0,
                       end_ply=0, trace=0, base_trace=0)
            rec.forks.append(row)
            arm_game = None
            t0 = time.perf_counter()
            try:
                if stream in (D.RESEED_AA, D.RESEED_B):
                    info = game.world_info()
                    arm_game = game.world_clone(D.stream_state(seed, ply, 1 if stream == D.RESEED_B else 0), info[5] - info[2])
                else:
                    arm_game = game.world_copy()
                arm_game.world_apply_from(game, alt)
            except self.native_env.NativeRefusal:
                row["status"] = D.REFUSED
                self.n["fork_refused"] += 1
                if arm_game is not None:
                    arm_game.close()
                continue
            finally:
                self.t["engine"] += time.perf_counter() - t0
            arm = Arm([seed], arm_game)
            arm.role = "handicap" if stream == D.HANDICAP else "alt"
            arm.rec, arm.row, arm.mover = rec, row, side
            arm.handicap = D.HANDICAP_MOVES if stream == D.HANDICAP else 0
            arm.ply, arm.trace, arm.tracks = ply + 1, 0, []
            arm.views = []
            rec.open += 1
            self.n["forks"] += 1
            if stream == D.EXACT_AA:                # the base keeps the running digest this arm must reproduce
                slot.tracks.append(row)
                arm.trace = D.trace_step(0, len(lp), top)
            if keep_view:
                arm.views.append(("alt", fid))
                slot.views.append(("base", fid, side))
                self.writer_view(rec, seed, fid, 0, side, fork_view, value=float(prediction.value))
            spawned.append(arm)
        if is_cand or cand_tags:
            self.add_candidate(slot, rec, side, lp, top, tb, is_cand, cand_tags if cand_tags else tags, inputs)
        return spawned

    def writer_view(self, rec, a, b, role, side, sparse, term=0, value=float("nan")):
        rec.views.append((a, b, role, D.SIDE_CODE[side], sparse[0], sparse[1], term, value))

    def capture(self, slot, obs, watch, side):
        """The first decision after a fork: the mover's own view of it (the observation when the mover moves here, else the
        other-side view), the pair of cases its value model is asked about."""
        if slot.role == "roll":
            vec = obs if side == slot.mover else watch
            item = slot.chunk.item
            slot.chunk.views.append((item["pid"] * 100 + slot.arm_id + 1, slot.k, 2, D.SIDE_CODE[slot.mover], *V.sparse(vec), 0, float("nan")))
            slot.views = []
            return
        keep = []
        for entry in slot.views:
            if entry[0] == "alt":
                vec = obs if side == slot.mover else watch
                self.writer_view(slot.rec, int(slot.item[0]), entry[1], 2, slot.mover, V.sparse(vec))
            else:
                vec = obs if side == entry[2] else watch
                self.writer_view(slot.rec, int(slot.item[0]), entry[1], 1, entry[2], V.sparse(vec))
        slot.views = keep

    def add_candidate(self, slot, rec, side, lp, top, tb, is_cand, tags, inputs):
        game = slot.game
        seed, ply = int(slot.item[0]), slot.ply
        if tags and not is_cand:                 # named cases are thinned per tag so a common one cannot swamp the plan
            rate = max((float(self.tag_rates.get(D.TAG_NAMES[t], 1.0)) for t in D.TAG_NAMES if tags & t), default=1.0)
            if float(D.pick_rng(seed, ply, 8).random()) >= rate:
                return
        first, chosen = D.candidate_arms(seed, ply, lp, top)
        arms = [dict(row=int(r), source="top5") for r in first] + [dict(row=int(r), source="random") for r in chosen]
        ai = self.ai
        kinds = slot.extra[0]
        named = D.named_alternative(game, ai, top, lp, kinds) if tags & D.TAG_SPLIT else None
        if named is not None and named not in [a["row"] for a in arms]:
            arms.append(dict(row=named, source="named"))
        p = np.exp(lp)
        for a in arms:
            d, k, params = game.row(ai, a["row"])
            a.update(p=float(p[a["row"]]), kind=game.vocab[int(kinds[a["row"]])], label=C_label(d, k, params))
        d, k, params = game.row(ai, top)
        rec.candidates.append(dict(seed=seed, ply=ply, turn=int(game.turn), phase=str(game.phase), side=side, top=int(top), p_top=float(p[top]),
                                   top_band=int(tb), rows=int(len(lp)), pos=str(V.hash64(inputs.observation.tobytes())), general=bool(is_cand),
                                   tags=int(tags), top_kind=game.vocab[int(kinds[top])], top_label=C_label(d, k, params), arms=arms))
        self.n["candidates"] += 1

    def begin_rollouts(self, slot, inputs, top):
        chunk, item, game = slot.chunk, slot.chunk.item, slot.game
        if (str(V.hash64(inputs.observation.tobytes())) != str(item["pos"]) or len(inputs.candidates) != int(item["rows"])
                or int(top) != int(item["top"]) or str(game.side) != str(item["side"])):
            self.n["replay_mismatch"] += 1
            chunk.lost = True
            game.close()
            self.writer.add_chunk(chunk.id + "!mismatch", [])
            self.after_write()
            return
        info = game.world_info()
        chunk.game, chunk.extra = game, info[5] - info[2]
        for k in range(int(item["k0"]), int(item["k1"])):
            for a in item["arms"]:
                self.queue.append(dict(chunk=chunk, arm=int(a["arm"]), row=int(a["row"]), k=k, last=False))
        self.queue[-1]["last"] = True

    # ----------------------------------------------------------------------------------------------------- ending
    def arm_refused(self, slot, error):
        try:
            slot.game.close()
        except Exception:  # noqa: BLE001
            pass
        slot.pending = None
        if slot.role == "base":
            slot.rec.dropped = True
            self.refusal(slot.item, slot.ply, error)
            self.close_record(slot.rec)
        elif slot.role in ("alt", "handicap"):
            slot.row["status"] = D.REFUSED
            self.n["fork_refused"] += 1
            self.close_record(slot.rec)
        elif slot.role == "replay":
            self.refusal(slot.item, slot.ply, error)
            self.writer.add_chunk(slot.chunk.id + "!refused", [])
            self.after_write()
        else:
            item = slot.chunk.item
            slot.chunk.rows.append(dict(pid=item["pid"], arm=slot.arm_id, k=slot.k, status=D.REFUSED, winner=-1, route=0, end_turn=0, end_ply=0, trace=0))
            self.n["fork_refused"] += 1
            self.close_roll(slot.chunk)

    def refusal(self, item, rows, error):
        self.n["refusals"] += 1
        print("[deviation] native refusal: seed %s after %d decisions (%s); game dropped" % (item[0], rows, error), file=sys.__stdout__, flush=True)
        self.send("refusal", self.spec["index"], int(item[0]), str(error)[:500])

    def finish_arm(self, slot):
        game = slot.game
        t0 = time.perf_counter()
        try:
            end = game.end
            final_turn = game.final_turn
        finally:
            try:
                game.close()
            except Exception:  # noqa: BLE001
                pass
        if end is None:
            raise RuntimeError("seed %s: a game ended without an accounted end" % (slot.item[0],))
        winner, route = D.winner_code(end), D.ROUTE_CODES[end.victory_route if end.status == "terminated" else None]
        turn = int(final_turn or 0)
        if slot.role == "replay":              # the replay ended before its position: a float tie flipped it (exact A/A .998)
            self.n["replay_mismatch"] += 1
            print("[deviation] replay of chunk %s ended before its position; chunk dropped" % (slot.chunk.id,), file=sys.__stdout__, flush=True)
            self.writer.add_chunk(slot.chunk.id + "!mismatch", [])
            self.after_write()
        elif slot.role == "base":
            slot.rec.base = dict(seed=int(slot.item[0]), winner=winner, route=route, turns=turn, plies=slot.ply, forks=len(slot.rec.forks))
            for entry in slot.views:           # the base ended right after a kept fork
                z = float(D.score(winner, D.SIDE_CODE[entry[2]])) * 2 - 1
                self.writer_view(slot.rec, int(slot.item[0]), entry[1], 1, entry[2], (np.zeros(0, np.int16), np.zeros(0, np.float32)), term=1, value=z)
            self.close_record(slot.rec)
        elif slot.role in ("alt", "handicap"):
            slot.row.update(winner=winner, route=route, end_turn=turn, end_ply=slot.ply, trace=slot.trace)
            for entry in slot.views:
                z = float(D.score(winner, D.SIDE_CODE[slot.mover])) * 2 - 1
                self.writer_view(slot.rec, int(slot.item[0]), entry[1], 2, slot.mover, (np.zeros(0, np.int16), np.zeros(0, np.float32)), term=1, value=z)
            self.close_record(slot.rec)
        else:
            item = slot.chunk.item
            slot.chunk.rows.append(dict(pid=item["pid"], arm=slot.arm_id, k=slot.k, status=D.OK, winner=winner, route=route, end_turn=turn,
                                        end_ply=slot.ply, trace=slot.trace))
            for _entry in slot.views:
                z = float(D.score(winner, D.SIDE_CODE[slot.mover])) * 2 - 1
                slot.chunk.views.append((item["pid"] * 100 + slot.arm_id + 1, slot.k, 2, D.SIDE_CODE[slot.mover],
                                         np.zeros(0, np.int16), np.zeros(0, np.float32), 1, z))
            self.close_roll(slot.chunk)
        self.t["engine"] += time.perf_counter() - t0

    def close_record(self, rec):
        rec.open -= 1
        if rec.open > 0:
            return
        if rec.dropped or rec.base is None:
            return
        self.n["games"] += 1
        rec.base["forks"] = len(rec.forks)
        for a, b, role, side, idx, val, term, value in rec.views:
            self.writer.add_view(a, b, role, side, idx, val, term, value)
        self.writer.add_game(rec.base, rec.forks, rec.candidates)
        self.after_write()

    def close_roll(self, chunk):
        chunk.open -= 1
        if chunk.open == 0 and chunk.game is None:
            self.write_chunk(chunk)

    def write_chunk(self, chunk):
        for a, b, role, side, idx, val, term, value in chunk.views:
            self.writer.add_view(a, b, role, side, idx, val, term, value)
        self.writer.add_chunk(chunk.id, chunk.rows)
        self.n["chunks"] += 1
        self.after_write()

    def after_write(self):
        if self.writer.bytes >= self.flush_bytes or len(self.writer) >= self.flush_games:
            self.flush()


def C_label(die, kind, params):
    import json
    return json.dumps(dict(die=die, kind=kind, params=params), default=str, sort_keys=True)[:400]


_LEAGUE_SERVE = lgworker.serve_main


def serve_main(spec, base, jobs, channel, stop, abandon, counter):
    """lgnode.Pool's process target: the league's serve loop with this worker class."""
    lgworker.LeagueWorker = DeviationWorker
    C.setup_paths(base["tree"])
    V.allow_search_patches()
    return _LEAGUE_SERVE(spec, base, jobs, channel, stop, abandon, counter)
