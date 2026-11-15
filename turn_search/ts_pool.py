"""A pool of worker processes for the turn search: each holds the same position on its own fast engine and runs the
playout batches it is handed (in lockstep).  Used by ts_service.py."""
import multiprocessing as mp
import pickle
import sys
import traceback

import turnsearch as S


def worker_main(conn, ckpts, judges):
    """ckpts: the world's net table, name -> (folder, side); judges: name -> (side, file) (tscommon.net_table)."""
    sys.stdout = sys.stderr                       # nothing of a worker may reach the service's own channel
    try:
        import tscommon as TS
        import turnsearch as S
        w = TS.world(ckpts)
        ai, ne, ctx = w.ai, w.native_env, w.native
        values = TS.load_values(ai, judges)
        for name in ckpts:                        # load every net now, not inside the first decision
            w.actor(name)
        S.warm(w, values, ckpts, {name: side for name, (side, _file) in judges.items()})      # ... and run each once
        conn.send(("ready", None))
    except BaseException:
        conn.send(("failed", traceback.format_exc()[-1500:]))
        return
    base = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg[0]
        try:
            if op == "quit":
                break
            if op == "pos":
                if base is not None:
                    base.close()
                    base = None
                _, blob, stage, active, stream_key = msg
                state = pickle.loads(blob)
                base = ne.NativeRawGame.at(ctx, ai, state, TS.stream(*stream_key), stage, active, max_turns=100)
                more = base.next()
                while more and base.forced:
                    base.apply(0)
                    more = base.next()
                conn.send(("ok", (str(base.side), int(base.turn), int(base.rows)) if more else None))
            elif op == "batch":
                _, cfg_dict, seed_key, side, turn0, specs = msg
                before = dict(S.TIMES)
                out, plies = S.run_batch(w, values, S.Config(**cfg_dict), base, seed_key, side, turn0, specs)
                conn.send(("ok", (out, plies, {k: S.TIMES[k] - before[k] for k in before})))
            elif op == "drop":
                if base is not None:
                    base.close()
                    base = None
                conn.send(("ok", None))
        except BaseException:
            conn.send(("error", traceback.format_exc()[-1500:]))
    if base is not None:
        base.close()


class Pool:
    def __init__(self, n, ckpts, judges):
        ctx = mp.get_context("spawn")
        self.workers = []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker_main, args=(b, ckpts, judges), daemon=True)
            p.start()
            self.workers.append((p, a))
        for p, a in self.workers:
            tag, text = a.recv()
            if tag != "ready":
                raise RuntimeError("a turn-search worker did not start: %s" % text)
        self.cfg = None

    def _all(self, msg):
        for _, a in self.workers:
            a.send(msg)
        out = []
        for _, a in self.workers:
            tag, body = a.recv()
            if tag != "ok":
                raise RuntimeError("turn-search worker: %s" % body)
            out.append(body)
        return out

    def set_position(self, blob, stage, active, stream_key):
        return self._all(("pos", blob, stage, active, stream_key))

    def drop(self):
        self._all(("drop",))

    def executor(self, cfg):
        cfg_dict = dict(cfg.__dict__)

        def run(seed_key, side, turn0, specs):
            n = len(self.workers)
            chunks = [specs[i::n] for i in range(n)]
            used = [(i, c) for i, c in enumerate(chunks) if c]
            for i, c in used:
                self.workers[i][1].send(("batch", cfg_dict, seed_key, side, turn0, c))
            results = [None] * len(specs)
            plies = 0
            for i, c in used:
                tag, body = self.workers[i][1].recv()
                if tag != "ok":
                    raise RuntimeError("turn-search worker: %s" % body)
                out, pl, times = body
                plies += pl
                for name, seconds in times.items():     # the workers' seconds, summed over workers (turnsearch.TIMES)
                    S.TIMES[name] += seconds
                for j, o in enumerate(out):
                    results[i + j * n] = o
            return results, plies
        return run

    def close(self):
        for p, a in self.workers:
            try:
                a.send(("quit",))
            except Exception:
                pass
        for p, a in self.workers:
            p.join(5)
            if p.is_alive():
                p.terminate()
