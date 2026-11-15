"""The turn-search service: a process beside the live AI server.  The server hands it the position at one of the AI's
decisions (a pickled engine state, hidden cards already dealt at random by the server); it answers with the move to play.

    python -B ts_service.py --pipe \\\\.\\pipe\\NAME --workers 6 --policy DIR --values DIR

--policy DIR holds fp_win\\fp_seed.pt and sh_win\\sh_seed.pt; --values DIR holds fp_value.pt and sh_value.pt.
Requests ({'op': 'choose', 'state': bytes, 'side': 'fp'|'sh', 'owner': 'fp'|'sh'|None, 'seed': int, 'setting': name})
and answers travel as pickles over a named pipe (multiprocessing.connection).  Anything the service cannot do it REFUSES
({'ok': False, 'why'}), so the caller plays its ordinary search: it never guesses.

HOW HARD IT SEARCHES is the caller's choice, per decision, by NAME: search_settings.json beside this file says what each
name means (absent = one setting "default": the defaults of turnsearch.Config on the CPU):
  device    "auto" (the default), "cuda" or "cpu": where the nets run.  auto and cuda both mean "this PC's NVIDIA GPU
            if it can really be used, else the processor" (ts_device.py says when and why); --device overrides it;
  gpu       which GPU, as CUDA_VISIBLE_DEVICES names it (default "0");
  seconds   the time one decision may take (turnsearch.Config.seconds; 0 or absent = no limit): a PC too slow for a
            setting's streams gets as many as fit, and an answer says so ("cut", "out_of_time");
  nets      side -> key -> {"policy": folder under --policy, "judge": a file under --values, or "own"}: the nets a
            seat can play (tscommon.net_table).  A request names its seat's net by the key ("net"; default "win"); that
            net plays the seat's moves in the playouts and its judge judges them ("own" = the net's own value head,
            which the league trained on that net's objective); the side's OTHER nets are the seat's advisers in the
            settings that say "advisers": true; the other side is always answered with its "win" net;
  settings  {name: {"advisers": true|false, "config": {any other turnsearch.Config field: kinds, horizon, k_moves,
            plans, keep, r1, r2, r3, ...}}};
  default   the name used for a request that names none.
A request that names a setting the file does not hold is refused.

--workers auto (the default): one per physical core, at most 10, and no more than the free memory holds (ts_device.py).
Exit codes the starter acts on: ts_device.GPU_FAILED (4) = the GPU failed, start me again with --device cpu;
ts_device.CANNOT_RUN (5) = this PC cannot run the turn search now (memory).
--tell FILE: the service writes there what this PC gives as soon as it is decided, before the nets load (the starter's
default strength follows it: GPU or processor).
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

sys.stdout = sys.stderr
sys.path.insert(0, str(Path(__file__).resolve().parent))     # the kit's own modules: a Python with a fixed path file
                                                              # (the game's own) does not add the script's folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", required=True)
    ap.add_argument("--workers", default="auto", help="a number, or auto: what this PC's cores and free memory hold")
    ap.add_argument("--device", default="", help="auto | cuda | cpu (default: the settings file's, else auto)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--values", required=True)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--parent", type=int, default=0)
    ap.add_argument("--tell", default="", help="a file that gets what this PC gives, once it is decided")
    args = ap.parse_args()

    def tell(device, workers, why):
        """{"parent", "device": "cuda" | "cpu" | "" (cannot run), "workers", "why"}, written whole."""
        if args.tell:
            try:
                Path(args.tell + ".tmp").write_text(json.dumps(dict(parent=args.parent, device=device, workers=workers, why=why)),
                                                    encoding="utf-8")
                os.replace(args.tell + ".tmp", args.tell)
            except OSError as error:
                print("turn-search service: could not write %s: %s" % (args.tell, error), flush=True)
    if args.parent:
        # the service lives exactly as long as the AI server that started it
        import threading
        if os.name == "nt":
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            handle = k32.OpenProcess(0x00100000, False, int(args.parent))          # SYNCHRONIZE

            def watch():
                while True:
                    if not handle or k32.WaitForSingleObject(handle, 2000) != 0x102:   # not WAIT_TIMEOUT: the parent is gone
                        os._exit(0)
        else:
            def watch():                # (a process whose parent ended is handed to another parent)
                while True:
                    if os.getppid() != int(args.parent):
                        os._exit(0)
                    time.sleep(2)
        threading.Thread(target=watch, daemon=True).start()
    path = Path(__file__).resolve().parent / "search_settings.json"
    settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # WHAT THIS PC GIVES (ts_device.py): how many workers, and whether the nets run on its GPU.  Decided before anything
    # of the kit is imported; tscommon reads the two variables when it is.
    import ts_device
    workers, why_workers = ts_device.workers(args.workers)
    if workers < 1:
        print("turn-search service: cannot run on this PC now (%s)" % why_workers, flush=True)
        tell("", 0, why_workers)
        os._exit(ts_device.CANNOT_RUN)
    device, workers, why_device = ts_device.device(args.device or settings.get("device", "auto"), workers, settings.get("gpu", "0"))
    os.environ["WOTR_TS_DEVICE"], os.environ["WOTR_TS_GPU"] = device, str(settings.get("gpu", "0"))
    tell(device, workers, why_device)
    print("turn-search service: the nets run on %s (%s); %d workers (%s)" % (
        "the GPU" if device == "cuda" else "the processor", why_device, workers, why_workers), flush=True)
    try:
        import numpy as np
        import tscommon as TS
        import turnsearch as S
        import ts_pool
        ckpts, judges, seat_net = TS.net_table(settings, args.policy, args.values)
        absent = [p for p in [folder for folder, _side in ckpts.values()] + [f for _side, f in judges.values()] if not Path(p).exists()]
        if absent or any((side, "win") not in seat_net for side in ("fp", "sh")):
            raise SystemExit("search_settings.json names nets that are not there (or a side has no \"win\" net): %s" % absent)
        w = TS.world(ckpts)
        ai, ne, ctx = w.ai, w.native_env, w.native
        values = TS.load_values(ai, judges)
        named = settings.get("settings") or {"default": {}}
        default = settings.get("default") or next(iter(named))
        if default not in named:
            raise SystemExit("search_settings.json: the default %r is not one of its settings %s" % (default, sorted(named)))
        pool = ts_pool.Pool(workers, ckpts, judges)
        S.warm(w, values, ckpts, {name: side for name, (side, _file) in judges.items()})
    except Exception:
        text = traceback.format_exc()
        print(text, flush=True)
        if device == "cuda" and ts_device.is_gpu_error(text):
            print("turn-search service: the GPU failed while starting - ending so that it is started on the processor", flush=True)
            os._exit(ts_device.GPU_FAILED)
        raise
    # ONE SEARCH PER (setting, seat): the seat's own net plays its moves and is judged by its own value; the side's
    # other nets are its advisers; the other side is answered with its WIN net.  Every seat of the table is searched
    # the same way - the WIN seats exactly as before there were others.
    searches = {}
    for name, spec in named.items():
        for (side, key), net in seat_net.items():
            cfg = S.Config(**dict(dict(horizon=args.horizon, seconds=float(settings.get("seconds", 0))),
                                  main={side: net}, judge={side: net},
                                  advisers={side: TS.advisers_of(seat_net, side, key)} if spec.get("advisers") else {},
                                  **spec.get("config", {})))
            searches[(name, side, key)] = S.TurnSearch(w, values, cfg, executor=pool.executor(cfg))
    index = {"fp": 0, "sh": 1}
    try:
        from ai_policies import action_policy_key as policy_key
    except Exception:
        policy_key = None
    print("turn-search service ready: %d workers on %s, pipe %s, default setting %s, seats %s, settings %s" % (
        workers, TS.DEVICE, args.pipe, default, sorted("%s %s" % seat for seat in seat_net),
        json.dumps({"%s %s %s" % k: dict(s.cfg.__dict__) for k, s in searches.items()}, sort_keys=True)), flush=True)

    def choose(req):
        t0 = time.perf_counter()
        setting = req.get("setting") or default
        side, key = req["side"], req.get("net") or "win"
        if setting not in named:
            return dict(ok=False, why="no setting named %r (search_settings.json holds %s)" % (setting, sorted(named)))
        search = searches.get((setting, side, key))
        if search is None:
            return dict(ok=False, no_such_net=True, why="no net %r of the side %s (search_settings.json holds %s)" % (
                key, side, sorted(k for s, k in seat_net if s == side)))
        state = pickle.loads(req["state"])
        if not getattr(state, "dice_rolled", False):
            return dict(ok=False, why="not in the action phase")
        seed = int(req.get("seed", 0))
        # WHOSE ACTION IS IT.  The server's "owner" is the side whose action an open follow-up decision belongs to (a battle,
        # a hunt, a discard); at an ordinary action it names the previous actor (or nobody), and the action is the
        # deciding side's.  So: the owner's seat first, the deciding side's seat second, and the seat is taken only when
        # the fast engine then opens the SAME decision (same side to move, same number of options).
        seats = [index[s] for s in dict.fromkeys([req.get("owner") or side, side])]
        base, why = None, None
        for seat in seats:
            try:
                g = ne.NativeRawGame.at(ctx, ai, state, TS.stream(seed, 1), "actions", seat, max_turns=100)
            except BaseException as error:
                why = "the fast engine refuses the position: %s" % (str(error)[:200],)
                continue
            try:
                more = g.next()
                while more and g.forced:
                    g.apply(0)
                    more = g.next()
                if not more:
                    why = "the fast engine sees no decision here"
                elif str(g.side) != side:
                    why = "the fast engine has %s to move, the game %s" % (g.side, side)
                elif req.get("rows") is not None and int(g.rows) != int(req["rows"]):
                    why = "menus differ: fast engine %d rows, game %d" % (g.rows, req["rows"])
                else:
                    base, active = g, seat
                    break
            except BaseException as error:
                why = "the fast engine refuses the position: %s" % (str(error)[:200],)
            g.close()
        if base is None:
            return dict(ok=False, why=why)
        try:
            seen = pool.set_position(req["state"], "actions", active, (seed, 1))
            if any(x != (side, int(base.turn), int(base.rows)) for x in seen):
                return dict(ok=False, why="the workers see another decision: %r" % (seen[:2],))
            ref_row = None
            if req.get("ref") is not None:
                # the move the server would play otherwise (its ordinary search's pick): the move to beat
                rdie, rkind, rparams = req["ref"]
                kinds = base.encode()[4]
                same_kind = [i for i in range(int(base.rows)) if base.vocab[int(kinds[i])] == rkind]
                decoded = [(i, base.row(ai, i)) for i in same_kind]
                ref_row = next((i for i, (d, k, p) in decoded if d == rdie and p == rparams), None)
                if ref_row is None and policy_key is not None:
                    want = policy_key(ai.Action(die_idx=rdie, action_type=rkind, params=rparams))
                    ref_row = next((i for i, (d, k, p) in decoded if policy_key(ai.Action(die_idx=d, action_type=k, params=p)) == want), None)
                if ref_row is None:
                    return dict(ok=False, why="the server's own move (%s) is not in the fast engine's menu" % rkind)
            before = dict(search.stats)
            pick, rep = search.choose(base, (seed, 7), ref_move=ref_row)
            die, kind, params = base.row(ai, int(pick))
            tdie, tkind, tparams = base.row(ai, int(rep["ref"]))
            return dict(ok=True, die=die, kind=kind, params=params, switched=bool(pick != rep["ref"]),
                        top=dict(die=tdie, kind=tkind, params=tparams), gain=rep.get("gain"), se=rep.get("se"),
                        plan=rep.get("plan"), plans=rep.get("n_pairs"), rows=int(base.rows), setting=setting, net=key,
                        playouts=search.stats["playouts"] - before["playouts"], plies=search.stats["plies"] - before["plies"],
                        seconds=time.perf_counter() - t0, device=TS.DEVICE, workers=workers,
                        cut=rep.get("cut") or {}, out_of_time=bool(rep.get("out_of_time")))
        finally:
            base.close()

    with Listener(args.pipe, authkey=b"wotr-turn-search") as listener:       # (a pipe or a socket file: by the address)
        while True:
            conn = listener.accept()
            try:
                while True:
                    req = conn.recv()
                    op = req.get("op")
                    if op == "quit":
                        conn.send(dict(ok=True))
                        pool.close()
                        return
                    if op == "ping":
                        conn.send(dict(ok=True, workers=workers, device=TS.DEVICE))
                        continue
                    try:
                        conn.send(choose(req))
                    except BaseException:
                        text = traceback.format_exc()
                        print(text, flush=True)
                        gpu_failed = device == "cuda" and ts_device.is_gpu_error(text)
                        conn.send(dict(ok=False, why="service error: " + text[-900:], gpu_failed=gpu_failed))
                        if gpu_failed:
                            # the GPU is gone or full: end, and say so with the exit code - the server's client starts
                            # the service again on the processor
                            print("turn-search service: the GPU failed - ending so that it is started on the processor", flush=True)
                            pool.close()
                            os._exit(ts_device.GPU_FAILED)
                        if "turn-search worker" in text:
                            # a worker failed in the middle of a batch: the others' answers are unread, so the pool
                            # cannot be trusted again.  End; the server's client starts a fresh service at the next decision.
                            pool.close()
                            os._exit(3)
            except (EOFError, ConnectionError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
