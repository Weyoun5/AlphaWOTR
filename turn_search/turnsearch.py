"""TURN SEARCH (v1).  At a decision, judge each candidate move by what the board looks like after the REST OF THE TURN
and the whole next turn, not three plies later.

  * candidates  : the net's top moves at this decision;
  * plans       : for each candidate, the mover's remaining moves of THIS turn are either the net's own choice ("greedy")
                  or a sampled different sequence (so other orders of its dice are tried);
  * the opponent: plays its net's move (its dice run out exactly as in the game: no reply once it has none);
  * chance      : every plan is played on the same few dice streams and hidden-card deals (common random numbers);
  * the judge   : the value net, asked at the first decision of turn T + horizon (or the real result if the game ended);
  * the choice  : stage 1 screens (candidate, plan) pairs on a few streams; stage 2 re-plays the best pairs on FRESH
                  streams; the move played is the first move of the best pair, if it beats the net's own move by a margin.

All playouts of one batch run in lockstep, so the nets are called in batches.  A batch is a list of specs
(move, plan, k, follow) -> (value, record, line); `executor` runs it here or on a pool of processes.
"""
import math
import time

import numpy as np
import torch

import tscommon as TS

D = TS.D
# Where this process's playouts spent their time, in seconds since the process began (run_batch adds to it; a pool
# worker sends its own share with every batch, ts_pool sums them here in the service): the engine's steps, the engine's
# encoding of a decision for the nets, the nets, and the batches themselves (wall time, everything included).
TIMES = dict(engine=0.0, encode=0.0, nets=0.0, batch=0.0)


SMALL_MENU = 8          # as the live search's ai._SMALL_MENU


class Config:
    def __init__(self, **kw):
        self.k_moves = 6            # candidate first moves
        self.p_min = 1e-4           # ... each with at least this prior (the top two always)
        self.kinds = 0              # also the net's best move of up to this many KINDS of action the candidates do not
                                    # hold yet, whatever its prior (0 = off): a kind the net never plays gets judged too
        self.advisers = {}          # side -> names of OTHER nets of that side (the world's net table), e.g. the net trained
                                    # for a military win.  Each adds its own top `k_adviser` moves to the candidates, and
                                    # every candidate gets one more plan per adviser: the mover plays THAT net's moves for
                                    # the whole look-ahead (a line the main net would never start gets played out)
        self.k_adviser = 3
        self.main = {}              # side -> the net that plays that side's own moves (the world's net table); a side not
                                    # named plays the net named like the side (its WIN net).  The searching seat's own
                                    # net goes here when it is not the WIN net; the other side is not named: its WIN net
        self.judge = {}             # side -> the value that judges for a searching seat of that side (the table of
                                    # values handed to run_batch); not named = the value named like the side
        self.adviser_turns = 10 ** 6  # an adviser's plan lasts this many turns (the search's look-ahead ends before that;
                                    # a playout to the end of the game sets it to the look-ahead the plan was chosen with)
        self.plans = 3              # plans per candidate (1 greedy + sampled ones)
        self.flat = 3.0             # sampled plans: the mover's priors raised to 1 / flat, over its top `plan_top` rows
        self.plan_top = 4
        self.r1 = 4                 # streams of the screen
        self.keep = 6               # pairs that go on to stage 2
        self.r2 = 12                # fresh streams of stage 2
        self.horizon = 2            # the judge is asked at the first decision of turn T + horizon
        self.r3 = 32                # stage 3: the ONE best challenger against the net's move on fresh streams again
        self.margin = 2.0           # switch away from the net's move only if that last, unselected gain exceeds margin * its se
        self.min_gain = 0.03        # ... and this much win probability
        self.lines = False          # keep each plan's readable line (slow: for reports)
        self.seconds = 0.0          # the time one decision may take (0 = as long as it takes).  A PC too slow for the
                                    # streams above gets as many as fit: the screen's rest may use half of what is left
                                    # after its first stream, then r2 and r3 are cut by one factor; under MIN_R2 / MIN_R3
                                    # streams nothing is decided and the move to beat stands ("out of time")
        self.__dict__.update(kw)


MIN_R2, MIN_R3 = 4, 8               # fewer streams than these say nothing: the decision is given up instead


def warm(w, values, names, judges=None):
    """One pass of every net on this process's device (a GPU's first pass takes seconds; not inside the first decision).
    names: the world's net table, name -> (folder, side).  judges: value name -> side (default: each named like its side)."""
    judges = judges or {s: s for s in ("fp", "sh")}
    g = w.new_game(1)
    try:
        seen = set()
        for _ in range(400):
            if len(seen) == 2 or not g.next():
                break
            side = str(g.side)
            if not g.forced and side not in seen:
                seen.add(side)
                enc = g.encode()
                for name, (_folder, s) in names.items():
                    if s == side:
                        TS.logp(w, g, enc, name)
                with torch.no_grad():
                    for name, s in judges.items():
                        if s == side:
                            net = values[name]
                            net(torch.as_tensor(np.stack([TS.view(g, side)])).to(next(net.parameters()).device))
            g.apply(0 if g.forced else int(np.argmax(TS.logp(w, g))))
    finally:
        g.close()


def adviser_of(cfg, side, plan):
    """The net whose moves the mover plays in this plan: plans 0 .. cfg.plans - 1 are the main net's (None), the ones
    after them belong to the side's advisers in order."""
    names = cfg.advisers.get(side, ())
    j = plan - cfg.plans
    return names[j] if 0 <= j < len(names) else None


def row_key(cand, regions, i):
    return hash((cand[i].tobytes(), int(regions[i])))


def key_ints(seed_key):
    return [abs(int(x)) % (2 ** 31) for x in seed_key]


class Playout:
    __slots__ = ("world", "move", "plan", "k", "own", "follow", "record", "value", "view", "enc", "dt", "line")


def run_batch(w, values, cfg, base, seed_key, side, turn0, specs):
    """specs: [(move, plan, k, follow or None)].  -> [(value, record, line)] in the same order.  `base` stands at the
    open decision; it is not changed."""
    clock = time.perf_counter
    t_batch, before = clock(), dict(TIMES)
    info = base.world_info()
    extra = info[5] - info[2]
    jobs = []
    for move, plan, k, follow in specs:
        p = Playout()
        p.world = base.world_clone(TS.stream(*seed_key, k), extra)
        try:
            p.world.world_determinize(D.SIDE_CODE[side], np.random.default_rng(key_ints(seed_key) + [k, 0x77]))
            p.world.world_apply_from(base, int(move))
        except BaseException:
            p.world.close()
            for q in jobs:
                q.world.close()
            raise
        p.move, p.plan, p.k, p.own, p.follow, p.record, p.value, p.view, p.line = move, plan, k, 0, follow, [], None, None, []
        jobs.append(p)
    active = list(jobs)
    plies = 0
    try:
        while active:
            waiting = {"fp": [], "sh": []}
            leaves = []
            for p in active:
                g = p.world
                more = g.next()
                while more and g.forced:
                    g.apply(0)
                    more = g.next()
                if not more:
                    end = g.end
                    p.value = 0.0 if end.winner is None else (1.0 if end.winner == side else -1.0)
                    continue
                p.dt = int(g.turn) - turn0
                t1 = clock()
                # only what this ply needs is encoded: at the look-ahead's end the judge's view (the searching side's
                # observation) and no menu; before it the mover's observation and the menu's rows
                if p.dt >= cfg.horizon:
                    p.view = TS.view(g, side)
                    leaves.append(p)
                else:
                    p.enc = TS.decision(g)
                    waiting[g.side].append(p)
                TIMES["encode"] += clock() - t1
            t1 = clock()
            if leaves:
                with torch.no_grad():
                    net = values[cfg.judge.get(side, side)]
                    v = net(torch.as_tensor(np.stack([p.view for p in leaves])).to(next(net.parameters()).device)).cpu().numpy()
                for p, x in zip(leaves, v):
                    p.value = float(x)
            answers = []
            for s, group in waiting.items():
                nets = {}                                              # the net that moves in each playout
                for p in group:
                    advised = s == side and p.dt < cfg.adviser_turns
                    nets.setdefault((adviser_of(cfg, side, p.plan) if advised else None) or cfg.main.get(s, s), []).append(p)
                for net, members in nets.items():
                    answers += [(s, p, pr) for p, pr in zip(members, w.actor(net).predict_many(
                        [w.ActorInput(p.enc[0], p.enc[1], p.enc[2], w.zeros(len(p.enc[1]))) for p in members]))]
            TIMES["nets"] += clock() - t1
            for s, p, pr in answers:
                lp = np.asarray(pr.log_probabilities, np.float64)
                sel = int(np.argmax(lp))
                if s == side and p.dt == 0 and 0 < p.plan < cfg.plans:
                    cand, regions = p.enc[1], p.enc[2]
                    if p.follow is not None:                       # a recorded plan: its next move if it is legal here
                        if p.own < len(p.follow):
                            want = p.follow[p.own]
                            for i in range(len(cand)):
                                if row_key(cand, regions, i) == want:
                                    sel = i
                                    break
                    else:                                          # the first realisation: sample and record
                        top = np.argsort(-lp)[:cfg.plan_top]
                        q = np.exp((lp[top] - lp[top].max()) / cfg.flat)
                        q /= q.sum()
                        rng = np.random.default_rng(key_ints(seed_key) + [p.move, p.plan, p.own, 0x51])
                        sel = int(rng.choice(top, p=q))
                        p.record.append(row_key(cand, regions, sel))
                    p.own += 1
                if cfg.lines and p.k == 0 and p.dt == 0:           # the readable line of this plan (stream 0 only)
                    d_, kind_, params_ = p.world.row(w.ai, sel)
                    p.line.append(("me" if s == side else "opp", kind_, params_))
                p.world.apply(sel)
                p.enc = None
                plies += 1
            active = [p for p in active if p.value is None]
    finally:
        for p in jobs:
            if p.world is not None:
                p.world.close()
                p.world = None
        spent = clock() - t_batch
        TIMES["batch"] += spent
        TIMES["engine"] += spent - (TIMES["encode"] - before["encode"]) - (TIMES["nets"] - before["nets"])
    return [(p.value, tuple(p.record), p.line) for p in jobs], plies


class TurnSearch:
    def __init__(self, w, value_nets, cfg=None, executor=None):
        self.w, self.values, self.cfg = w, value_nets, cfg or Config()
        self.stats = dict(playouts=0, plies=0)
        self.executor = executor       # None: run batches in this process on `base`

    def _exec(self, base, seed_key, side, turn0, specs):
        if self.executor is None:
            out, plies = run_batch(self.w, self.values, self.cfg, base, seed_key, side, turn0, specs)
        else:
            out, plies = self.executor(seed_key, side, turn0, specs)
        self.stats["playouts"] += len(specs)
        self.stats["plies"] += plies
        return out

    def kind(self, base, row):
        """What kind of action a menu row is: its type and the piece, card or nation it names (not where or how many)."""
        _die, name, params = base.row(self.w.ai, int(row))
        params = params if isinstance(params, dict) else {}
        return name, str(params.get("minion") or params.get("character") or params.get("card_id") or params.get("nation") or "")

    def choose(self, base, seed_key, forced_moves=(), ref_move=None):
        """base: a game standing at an open decision.  -> (row index to play, report dict).
        ref_move: the row to beat (the move the caller would play otherwise); None = the net's own top move."""
        w, cfg = self.w, self.cfg
        began, cut = time.perf_counter(), {}     # cut: the streams this decision got instead of the configured ones
        side, turn0 = str(base.side), int(base.turn)
        enc = base.encode()
        lp = TS.logp(w, base, enc, cfg.main.get(side))
        order = [int(i) for i in np.argsort(-lp)]
        top = order[0]
        if len(order) == 1:
            return top, dict(side=side, turn=turn0, top=top, ref=top, pick=top, pairs=[], reason="one legal move")
        # (a menu of at most SMALL_MENU rows is searched whole: no prior keeps a row out)
        moves = [i for n, i in enumerate(order[:cfg.k_moves])
                 if n < 2 or len(order) <= SMALL_MENU or lp[i] >= math.log(cfg.p_min)]
        ref_row = top if ref_move is None else int(ref_move)
        for m in tuple(forced_moves) + (ref_row,):
            if int(m) not in moves:
                moves.append(int(m))
        for name in cfg.advisers.get(side, ()):
            lpa = TS.logp(w, base, enc, name)
            for n, i in enumerate(int(i) for i in np.argsort(-lpa)[:cfg.k_adviser]):
                if (n < 1 or lpa[i] >= math.log(cfg.p_min)) and i not in moves:
                    moves.append(i)
        if cfg.kinds:
            held, room = {self.kind(base, m) for m in moves}, len(moves) + cfg.kinds
            for i in order:
                if len(moves) >= room:
                    break
                kind = self.kind(base, i)
                if kind not in held:
                    held.add(kind)
                    moves.append(i)
        pairs = [(m, j) for m in moves for j in range(cfg.plans + len(cfg.advisers.get(side, ())))]

        def follow(pair):                        # a sampled plan is replayed from its record; the others are policies
            return plan_of[pair] if 0 < pair[1] < cfg.plans else None
        # stage 1, stream 0: realise (and record) every plan
        out = self._exec(base, seed_key, side, turn0, [(m, j, 0, None) for m, j in pairs])
        val = {pair: [o[0]] for pair, o in zip(pairs, out)}
        plan_of = {pair: o[1] for pair, o in zip(pairs, out)}
        line_of = {pair: o[2] for pair, o in zip(pairs, out)}
        seen, live = set(), []
        for pair in pairs:                       # a sampled plan that came out as another's sequence adds nothing
            key = (pair[0], "greedy" if pair[1] == 0 else plan_of[pair] if pair[1] < cfg.plans else pair[1])
            if key not in seen:
                seen.add(key)
                live.append(pair)
        r1, r2, r3, played = cfg.r1, cfg.r2, cfg.r3, len(pairs)

        def left():                              # seconds of the decision's time that are left
            return cfg.seconds - (time.perf_counter() - began)

        def pace():                              # seconds one playout took so far
            return (time.perf_counter() - began) / played
        if cfg.seconds and (r1 - 1) * len(live) * pace() > 0.5 * left():
            r1 = cut["r1"] = 1 + max(0, int(0.5 * left() / (len(live) * pace())))
        specs = [(m, j, k, follow((m, j))) for m, j in live for k in range(1, r1)]
        for spec, o in zip(specs, self._exec(base, seed_key, side, turn0, specs) if specs else ()):
            val[(spec[0], spec[1])].append(o[0])
        played += len(specs)
        s1 = {pair: float(np.mean(val[pair])) for pair in live}
        ref = (ref_row, 0)
        ranked = sorted(live, key=lambda pair: -s1[pair])
        keep = [ref] + [pair for pair in ranked if pair != ref][:cfg.keep - 1]
        if cfg.seconds and (len(keep) * r2 + 2 * r3) * pace() > left():
            fits = max(left(), 0.0) / ((len(keep) * r2 + 2 * r3) * pace())
            r2, r3 = cut["r2"], cut["r3"] = int(r2 * fits), int(r3 * fits)
            if r2 < MIN_R2 or r3 < MIN_R3:
                return ref_row, dict(side=side, turn=turn0, top=top, ref=ref_row, pick=ref_row, plan=0, gain=0.0, se=0.0,
                                     follow=None, stage3=None, pairs=[], moves=moves, n_pairs=len(live), rows=len(order),
                                     cut=cut, out_of_time=True)
        # stage 2: fresh streams, the same for every kept pair
        specs = [(m, j, 1000 + k, follow((m, j))) for m, j in keep for k in range(r2)]
        v2 = {pair: np.zeros(r2) for pair in keep}
        for spec, o in zip(specs, self._exec(base, seed_key, side, turn0, specs)):
            v2[(spec[0], spec[1])][spec[2] - 1000] = o[0]
        table = []
        cand, cand_gain = None, 0.0
        for pair in keep:
            d = (v2[pair] - v2[ref]) / 2.0                          # win-probability scale, paired by stream
            gain, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
            table.append(dict(move=pair[0], plan=pair[1], prior=float(math.exp(lp[pair[0]])), s1=s1[pair] / 2 + 0.5,
                              s2=float(v2[pair].mean()) / 2 + 0.5, gain=gain, se=se, steps=len(plan_of[pair]),
                              line=line_of.get(pair), net=adviser_of(cfg, side, pair[1])))
            if pair != ref and gain > cand_gain:
                cand, cand_gain = pair, gain
        # stage 3: the best-looking challenger was CHOSEN on stage 2's dice, so its gain there flatters it.  Play it and
        # the net's move once more on dice nothing was chosen on; only that gain decides.
        best, best_gain, best_se, s3 = ref, 0.0, 0.0, None
        if cand is not None and cand_gain >= cfg.min_gain:
            specs = [(pair[0], pair[1], 2000 + k, follow(pair))
                     for k in range(r3) for pair in (cand, ref)]
            out = self._exec(base, seed_key, side, turn0, specs)
            v3 = np.array([o[0] for o in out]).reshape(r3, 2)
            d = (v3[:, 0] - v3[:, 1]) / 2.0
            gain, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            s3 = dict(move=cand[0], plan=cand[1], gain=gain, se=se, stage2_gain=cand_gain)
            if gain >= cfg.min_gain and gain > cfg.margin * se:
                best, best_gain, best_se = cand, gain, se
        return best[0], dict(side=side, turn=turn0, top=top, ref=ref_row, pick=best[0], plan=best[1], gain=best_gain, se=best_se,
                             follow=follow(best), stage3=s3,
                             pairs=table, moves=moves, n_pairs=len(live), rows=len(order), cut=cut, out_of_time=False)
