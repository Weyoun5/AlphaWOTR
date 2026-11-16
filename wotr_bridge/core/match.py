# -*- coding: utf-8 -*-
"""match - MATCH, board against board (bridge2/DESIGN.md, principle 4).

    The human's decision is the legal option whose OWN piece changes, applied
    to the board as the engine has it when the decision opens, give the board
    as it stands now - with or without what the engine does alone after it.

judge(base, now, menu, opened=None) compares piece tables (boardmodel.Table):
  BASE    the board as the ENGINE STATE has it when the decision opens
          (boardmodel.ops_plan over reconcile.difference + moves),
  NOW     the current client board,
  OPENED  the client board as it was when the decision opened, when that was
          not yet BASE.  Whatever separated it from BASE and has not changed
          since (a piece of the step before still in transit, a marker the
          engine has already turned, something the human did early) is read
          BOTH ways - as arrived and as it lies: it never vetoes a match, it
          may be part of the answer, and a board that shows nothing else is
          "nothing".
and answers with a Verdict:

  commit     NOW == BASE + MOVES(o) + any part of CONSEQ(o) for exactly one
             choice o - or for a chain o1, o2, ... of consecutive decisions of
             the same human.  Verdict.chain = the options to apply, in order.
               .extendable  True: the board is also a strict part of ANOTHER
                            target of the same decision (heal / heal + move,
                            Elite removed / Elite reduced): commit only after
                            Q_SECONDS of quiet, and keep asking again().
                            False: it is not - or the human closed the die /
                            card window on the board BESIDE the moves of a
                            complete answer, or AFTER a move of the answer
                            (`tidied_last`: his closing gesture - the optional
                            rest is declined).  A kill with nothing beside
                            it, or one made BEFORE the moves that only
                            declines an optional rest, ends nothing: the
                            board is a strict part of every other answer
                            that spends that die.  Or he turned the
                            marker he placed to the side the engine has (the
                            flip ends a placement).
                            None: not settled (treat as True) - the scan limit
                            cut it, or the move was found outside the quick
                            filter.
                            A piece that may walk on (the Fellowship,
                            a Companion, a Nazgul) standing in one legal place
                            is a strict part of every other placement of it.
               .more        the same human has a further decision whose moves
                            may still come (the rest is that decision's).
               .ahead       the board holds the moves of the chain AND MORE,
                            and what follows is more decisions of his than one
                            search follows (MAX_DEPTH): commit the chain, judge
                            the rest against the new BASE.  (More beside a move
                            that a roll or the other seat follows is no answer:
                            WHOLE BOARD OR NOTHING - a mismatch that names it.)
               .toward      extendable True: the name of the first such target
                            found (for the game's log)
               .performed   per step: which pieces realised a free choice,
                            out of that step's own share of the board
                            [(zone, kind, +-n)] (the Leader's nation, Regular
                            or Elite, the box a Regular came from).
               .untouched   the board has not changed since the decision
                            opened and was read as it lies: his answer only
                            when he says so (a closing word) - never applied
                            by time (ready() is False)
               .unfinished  the answer ends his action, and his die still lies
                            in Current Action: putting it away is his (see
                            ready)
               .conseq      per step: True when the board also shows (part
                            of) that step's consequences.
               .closing     the last step has no move of its own and was read
                            from the closing kill (see below).
               .closed_early a step's die / card was killed before the
                            decision that follows it was answered.
               .next_staged the board also shows an action die staged for the
                            decision that follows the chain.
  nothing    NOW == BASE: nothing was done - or NOW is what it was when the
             decision opened and, read as it lies, neither an answer nor on
             the way to one (when it is, that verdict is given: what he moved
             ahead of a roll is his answer).
  wait       NOW lies between BASE and the target of some option (or chain):
             the human is still moving.
  undecided  several DIFFERENT choices have the board as their target, or the
             board shows only what options WITHOUT a move of their own leave
             (a staged die, a consequence): Verdict.choices; it needs a word
             or a click.
  mismatch   the board was changed and nothing fits: Verdict.nearest and
             Verdict.sentence name the nearest target and what differs.

An option without a move of its own (no piece change, or only its action die
staged) is never committed from an untouched board or from a staged die;
Verdict.words lists such options when the board shows exactly what they leave.
ONE exception, the CLOSING KILL: when nothing with a move fits, the board
shows a die or card killed out of the play area, and exactly one option
without any move has that kill among its closing changes ("done", "stay", no
second recruit), that option is committed - .closing is True.

No engine apply happens here; everything about the game comes in through the
Menu the caller builds:

  Menu(options, plan, sig=None, key=None, name=None, expand=None,
       conseq=None, outcome=None, side=None, net=None, faults=None, position=None)
    faults     [(option, what failed, the error)]: the Menu's callbacks append
               here when one of them FAILED for an option (a fault, not a
               hold); the board cannot answer that option, and the caller
               says so.  The matcher only carries the list
    options    the legal options (any objects)
    plan(o)    -> list of boardmodel.Change: the piece changes a player
               performing o makes ([] = none); None when it cannot be rendered
    sig(o)     -> Sig or None, WITHOUT rendering:
                 regions  frozenset of the map regions o's moves touch
                          (boardmodel.region_token names), exactly
                 die      the kind of the action die o takes from the dice
                          pool ('die:fp:C'), or None
                 same     a token shared by options whose MOVES are identical
                          (rendered once), or None
                 zones    frozenset of the zone FAMILIES outside the map, the
                          dice zones and the play area ('fp reinf', 'sa hand',
                          'political', 'track', ... - boardmodel.touched) that
                          o's moves, optional changes and consequences MAY
                          touch (a superset is fine), or None = unknown.  A
                          board that touches another family is not on its way.
               None = unknown: the option is always rendered.
    key(o)     -> hashable: options with the same key are the same choice
               (the same action through another die of the same face).
    name(o)    -> words for sentences.
    expand(o)  -> Menu of the NEXT decision of the same human after o;
               None (no further decision of his); ROLL (what follows needs a
               roll that was not made yet, or the other seat's answer).
    conseq(o)  -> list of boardmodel.Change: what the engine does alone after
               o, relative to BASE + MOVES(o) (the reconciler's moves for the
               state after o); None when that needs an unrolled result.
               Asked only of options whose own moves are already on the board
               while more was moved (and, capped, of options that move
               nothing on the map, to tell a consequence done early).
    outcome(o) -> hashable engine-state key after o (after the whole chain
               that led to o's menu), or None.  Chains with the same outcome
               are one choice (the same removals in another order).
    position(o) -> a cheap hashable key of the position after o, as the engine's own
               search tells positions apart, or None.  A position reached again by
               other choices, with the same board still to explain, is not searched
               again (two figures the game cannot tell apart, each gone where the
               other might have).
"""
import collections

from . import boardmodel as bm

COMMIT, WAIT, NOTHING, UNDECIDED, MISMATCH = 'commit', 'wait', 'nothing', 'undecided', 'mismatch'
EXTEND, CHANGE, SAME = 'extend', 'change', 'same'       # again(): kinds of a second commit
ROLL = 'roll'                # Menu.expand: what follows needs a roll / the other seat
MOVES, CONSEQ, CLOSING, EARLY, STAGED = 'moves', 'conseq', 'closing', 'early', 'staged'   # how a step was read
CUT = 'cut'                  # ... and the last step the search followed (MAX_DEPTH): more comes after it
MAX_DEPTH = 6                # decisions of one human in one gesture (corpus maximum: see REPORT)
WAIT_SCAN = 400              # options rendered to learn whether the board is on the way
PLAN_BUDGET = 800            # plans rendered for one board, all decisions of a chain together,
#                              before the on-the-way scans of the LATER decisions stop
#                              (stats['budget'] counts the stops; the first decision's own scan
#                              is never cut by it)
EXTEND_SCAN = 64             # options rendered to learn whether a match can still grow
NEAREST_SCAN = 24            # options rendered to name the nearest target of a mismatch
WORD_SCAN = 64               # options without a move asked for their consequences
CONSEQ_SCAN = 24             # options asked whether the board is their consequence done early
CHAIN_WAIT = 16              # largest decision asked whether the board begins with its NEXT decision
SUM_SCAN = 48                # options rendered to learn whether the board is the SUM of a chain they
SUM_STEPS = 8                # begin, and how many of them are followed into the next decision
MAX_CHOICES = 12
K_POLLS = 2                  # equal boards in a row before a commit is applied
Q_SECONDS = 6.0              # quiet time before a strict-part commit is applied

Sig = collections.namedtuple('Sig', 'regions die same zones n')     # n: the units its own move takes
Sig.__new__.__defaults__ = (None, None, None, None)
_NO_SIG = Sig(None)
# one rendered option: own delta, coarse rules, plan, optional delta (with what lagged), the
# target it can grow to (own + optional, without what lagged), has a move, optional without the lag
_P = collections.namedtuple('_P', 'd coarse plan opt full own soft')
_Step = collections.namedtuple('_Step', 'o menu p how level')


class Menu(object):
    __slots__ = ('options', 'plan', 'sig', 'key', 'name', 'expand', 'conseq', 'outcome', 'side', 'net',
                 'faults', 'position')

    def __init__(self, options, plan, sig=None, key=None, name=None, expand=None, conseq=None,
                 outcome=None, side=None, net=None, faults=None, position=None):
        self.position = position
        self.faults = faults if faults is not None else []      # see the module text
        self.side = side            # 'fp' / 'sh': whose decision it is (None: not told)
        self.net = net              # net(o) -> the changes of o AND of what the engine does alone
        #                             after it, as one change of BASE with named pieces; or None
        self.options = list(options)
        self.plan = plan
        self.sig = sig
        self.key = key or (lambda o: o)
        self.name = name or (lambda o: str(o))
        self.expand = expand
        self.conseq = conseq
        self.outcome = outcome


class Verdict(object):
    __slots__ = ('kind', 'chain', 'choices', 'nearest', 'sentence', 'extendable', 'more', 'ahead',
                 'words', 'performed', 'conseq', 'closing', 'closed_early', 'next_staged', 'key', 'delta',
                 'stats', 'unfinished', 'names', 'untouched', 'toward')

    def __init__(self, kind, chain=None, choices=None, nearest=None, sentence='', extendable=False,
                 more=False, ahead=False, words=None, performed=None, conseq=None, closing=False,
                 closed_early=False, next_staged=False, key=None, delta=None, stats=None,
                 unfinished=False, names=None, toward=None):
        self.unfinished, self.untouched = unfinished, False
        self.names = names or []        # UNDECIDED: the name of every step of every choice (as `choices`)
        self.kind = kind
        self.chain = chain or []
        self.choices = choices or []
        self.nearest = nearest or []
        self.sentence = sentence
        self.extendable = extendable
        self.toward = toward
        self.more = more
        self.ahead = ahead
        self.words = words or []
        self.performed = performed or []
        self.conseq = conseq or []
        self.closing = closing
        self.closed_early = closed_early
        self.next_staged = next_staged
        self.key = key
        self.delta = delta or {}
        self.stats = stats or {}

    def __repr__(self):
        return 'Verdict(%s%s%s)' % (self.kind, ', chain of %d' % len(self.chain) if self.chain else '',
                                    ', %d choices' % len(self.choices) if self.choices else '')


class _Level(object):
    """One decision of a (possible) chain: one menu judged against what is
    still unexplained on the board (d_now, relative to this level's base).
    `carry` = optional changes that may or may not be on the board whatever
    is chosen here: the closing sweep and consequences of the steps before;
    `lag` = what was not yet BASE when the decision opened, as it lies now
    (it excuses a difference, it is never a target to wait for)."""

    def __init__(self, base, d_now, menu, stats, carry=None, raw=None, lag=None):
        self.base, self.d_now, self.menu, self.stats = base, d_now, menu, stats
        self.carry = carry or {}
        self.lag = lag or {}
        self.raw = raw              # d_now as the board lies (what lagged NOT taken as arrived)
        self.later = False          # a later decision of a chain: its scans obey PLAN_BUDGET
        self._plans, self._same, self._conseq = {}, {}, {}
        self._cands = None
        self.ins = []               # steps whose own moves are all on the board
        self._early = 0
        stats['options'] += len(menu.options)

    def plan(self, o):
        """_P of an option; None = it cannot be rendered."""
        oid = id(o)
        if oid in self._plans:
            return self._plans[oid]
        s = self.menu.sig(o) if self.menu.sig else None
        tok = s.same if s is not None else None
        if tok is not None and tok in self._same:
            out = self._same[tok]
        else:
            self.stats['plans'] += 1
            pl = self.menu.plan(o)
            out = None
            if pl is not None:
                d, coarse, opt = bm.plan_delta(self.base, pl)
                d, coarse = bm.bind(d, coarse, self.d_now)
                if self.carry:
                    opt = bm.add_delta(opt, self.carry)
                out = _P(d, coarse, pl, bm.add_delta(opt, self.lag) if self.lag else opt,
                         bm.add_delta(d, opt) if opt else d, bool(d) and not bm.stages_only(d), opt)
            if tok is not None:
                self._same[tok] = out
        self._plans[oid] = out
        return out

    def conseq(self, o, p):
        """The _P of an option with its consequences among the optional
        changes, or None (no consequence, or not known without a roll).
        Asked once per option."""
        oid = id(o)
        if oid not in self._conseq:
            out = None
            if self.menu.conseq is not None:
                self.stats['conseq'] += 1
                q = self.menu.conseq(o)
                if q:
                    d, coarse, opt = bm.plan_delta(self.base, list(p.plan) + [bm.optional(c) for c in q])
                    d, coarse = bm.bind(d, coarse, self.d_now)
                    if self.carry:
                        opt = bm.add_delta(opt, self.carry)
                    # (the plan carries the consequences as optional changes: a die the engine
                    # spends after the move is a die this reading moves, whoever shows it)
                    out = p._replace(d=d, coarse=coarse, soft=opt, full=bm.add_delta(d, opt),
                                     opt=bm.add_delta(opt, self.lag) if self.lag else opt,
                                     plan=list(p.plan) + [bm.optional(c) for c in q])
            self._conseq[oid] = out
        return self._conseq[oid]

    def candidates(self):
        """Options by what their signature allows: (its moves may all be on
        the board; it may be a target the board is on the way to - tightest
        first; it moves nothing on the map itself, so only its consequences
        can put the board on its way; its moves may be on the board but for
        ONE region the board does not show changing; the same with its die
        not yet on the board)."""
        if self._cands is None:
            regions, dice = bm.observed(self.d_now)
            zones, out, less = bm.touched(self.d_now) + (regions,)
            if self.carry or self.lag:
                # what the steps before may have left on the board (their consequences, what
                # lagged) is no evidence against an option: it does not narrow the on-the-way scan
                rest = bm.residue(self.d_now, {}, frozenset(), bm.add_delta(self.carry, self.lag))
                zones, out, less = bm.touched(rest) + (bm.observed(rest)[0],)
            first, longer, cards, gapped, undiced = [], [], [], [], []
            for i, o in enumerate(self.menu.options):
                s = self.menu.sig(o) if self.menu.sig else None
                if s is None or s.regions is None:
                    first.append(o)
                    longer.append((0, i, o))
                    continue
                # (outside its pool a die has no face: 'die:fp:?' stands for any die of the side)
                die_seen = s.die is None or s.die in dice or s.die[:7] + '?' in dice
                unseen = s.regions - regions
                if die_seen and not unseen:
                    first.append(o)
                elif len(unseen) == 1 < len(s.regions):
                    (gapped if die_seen else undiced).append(o)
                if less <= s.regions:
                    # not on its way: another action die left the pool, or the board touches
                    # a zone family the option can never touch
                    if not ((out and s.die is not None and s.die not in out)
                            or (s.zones is not None and not zones <= s.zones)):
                        longer.append((len(s.regions) - len(less) + (0 if die_seen else 1), i, o))
                elif not s.regions:
                    cards.append(o)
            longer.sort(key=lambda t: t[:2])
            # of the options that lack one region, those first that take as many units as the
            # board shows gone or come in a region they name: a scan that is cut meets them first
            units = collections.Counter()
            for (z, k), v in self.d_now.items():
                units[bm.region_of(z)] += abs(v) if k.endswith((' REGULAR', ' ELITE')) else 0

            def off(o):
                s = self.menu.sig(o)
                return min(abs(s.n - units[r]) for r in s.regions & regions) if s.n is not None and s.regions & regions else 0
            self._cands = (first, [o for _n, _i, o in longer], cards, sorted(gapped, key=off), sorted(undiced, key=off))
        return self._cands

    def on_the_way(self, limit, want=1, alone=False):
        """Options (at most `want`) whose target the board lies strictly
        before; second value False when the scan limit cut the search.
        `alone`: the board shows nothing but dice / cards killed - then every
        other option with a move of its own that takes those dice and cards
        out of their pool and hand is such a target, whether or not its own
        plan ends with the kill."""
        found, n = [], 0
        longer, cards = self.candidates()[1:3]
        for o in longer + cards:
            if n >= limit or (self.later and self.stats['plans'] >= PLAN_BUDGET):
                self.stats['budget'] += n < limit
                return found, False
            before = self.stats['plans']
            p = self.plan(o)
            n += self.stats['plans'] - before
            if p is None or not p.own:
                continue
            hit = (bm.between(self.d_now, p.full, p.coarse) or (alone and all(
                p.d.get(k, 0) < 0 for k, v in self.d_now.items()
                if v < 0 and k[0] in bm.DICE_POOLS | bm.HANDS))) \
                and not bm.fits(self.d_now, p.d, p.coarse, p.opt)
            if not hit and self.menu.conseq is not None:
                # a consequence done before the option's own moves are complete (the recruits
                # of a card dragged before the card, the relieved garrison put back before the
                # advance): asked of at most CONSEQ_SCAN options, never of one whose signature
                # says it moves other regions than the board shows
                s = self.menu.sig(o) if self.menu.sig else None
                if (s is None or not s.regions) and (id(o) in self._conseq or self._early < CONSEQ_SCAN):
                    self._early += id(o) not in self._conseq
                    q = self.conseq(o, p)
                    if q is not None:
                        p = q
                        hit = bm.between(self.d_now, p.full, p.coarse) \
                            and not bm.fits(self.d_now, p.d, p.coarse, p.opt)
            if hit:
                found.append((o, bm.distance(bm.project(self.d_now, p.coarse),
                                             bm.project(p.d, p.coarse))))
                if len(found) >= want:
                    break
        return found, True

    def wordless(self):
        """[(option, _P)] of the options without a move of their own whose
        leavings (a staged die, the closing sweep, consequences) are exactly
        what the board shows."""
        out, n = [], 0
        for o in self.menu.options:
            s = self.menu.sig(o) if self.menu.sig else None
            if s is not None and s.regions:
                continue                # it moves pieces on the map
            p = self.plan(o)
            if p is None or p.own:
                continue
            if bm.fits(self.d_now, p.d, p.coarse, p.opt):
                out.append((o, p))
            elif not p.d and n < WORD_SCAN:
                n += 1
                q = self.conseq(o, p)
                if q is not None and bm.fits(self.d_now, q.d, q.coarse, q.opt):
                    out.append((o, q))
        return out


def _faces_fit(table, faces, plan=()):
    """No die the human took out of a pool shows a face that is neither the
    one this reading gives it nor the one it had (a die never turned) - but
    for as many dice as the reading moves whatever they show (a transfer of
    the class 'die:fp' / 'die:sh': the eye allocation before the roll)."""
    if faces is None:
        return True
    root, shown = faces
    want, was = bm.moved_faces(root, table)
    free = collections.Counter()
    for ch in plan:
        if ch.pid is None and not ch.opt and ch.klass in ('die:fp', 'die:sh'):
            free[ch.klass == 'die:fp'] += ch.n
    over = collections.Counter()
    for f, n in shown.items():
        over[f[0]] += max(0, n - want[f] - was[f])
    return all(n <= free[fp] for fp, n in over.items())


def _can_stage(table, faces):
    """Every die the board shows outside the pools is one this reading moved
    itself or still has, with that face, in a dice pool."""
    root, shown = faces
    moved = bm.moved_faces(root, table)[0]
    pool = collections.Counter((pid in bm.FP_DICE, t[1]) for pid, t in table.p.items()
                               if (pid in bm.FP_DICE or pid in bm.SH_DICE) and t[0] in bm.DICE_POOLS)
    return all(n <= moved[f] + pool[f] for f, n in shown.items())


class _Seen(set):
    """What one search has been through: the sets of choices it followed (the same choices in another order are
    not followed again), and what each POSITION it searched gave (`gave`: reached again by other choices, with
    the same board still to explain, its chains are those chains)."""

    def __init__(self):
        set.__init__(self)
        self.gave = {}


def _less(d, own, coarse):
    """A board difference without a step's own move (own: read by its coarse rules, as d is here).  What the
    step took from a CHOICE of zones, or brought to one, that the board does not show there (another step made
    that trip back) is booked on the first zone of the choice - as boardmodel.performed books such a trip."""
    seen = bm.project(d, coarse)
    return {(z if (z, k) in seen else z.partition('|')[0], k): v for (z, k), v in bm.sub_delta(seen, own).items()}


def _search(base, d_now, menu, depth, stats, carry, seen, prefix, raw=None, lag=None, faces=None):
    """-> (full chains, wait candidates, ahead candidates, level).  A full
    chain is a list of _Step: its moves, in order, give the board.

    The shortest reading wins, level by level:
      1. options whose own moves (with or without their optional changes)
         ARE the board;
      2. else, among the options whose own moves are all on the board, the
         largest ones (an option whose moves are a strict part of another
         candidate's is the longer way to the same board): own moves plus
         any part of their consequences;
      3. else those same options as the first step of a chain;
      4. else the closing kill: the one option without any move whose closing
         changes hold the die / card the board shows killed;
      5. else one of those largest options, when all the board shows beyond
         it is its die / card killed early - or an action die staged for the
         decision that follows (which then needs its own move or word);
      6. else a chain whose SUM is the board although its first step is not
         on the board by itself (_summed)."""
    lv = _Level(base, d_now, menu, stats, carry, raw, lag)
    lv.later = bool(prefix)
    full, waits, ahead, keys = [], [], [], set()
    ins = []
    for o in lv.candidates()[0]:
        p = lv.plan(o)
        if p is not None and p.own and (bm.inside(p.d, d_now, p.coarse)
                                        or (raw is not None and bm.inside(p.d, raw, p.coarse))):
            ins.append((o, p))

    def add(o, p, how):
        k = menu.key(o)
        # (a die staged for the decision that follows is not this reading's to explain)
        if k not in keys and (how == STAGED or _faces_fit(base.apply(p.plan, True), faces, p.plan)):
            keys.add(k)
            full.append([_Step(o, menu, p, how, lv)])
    for o, p in ins:
        if bm.fits(d_now, p.d, p.coarse, p.opt):
            add(o, p, MOVES)
    lv.ins = [_Step(o, menu, p, MOVES, lv) for o, p in ins]
    if full:
        return full, waits, ahead, lv
    tops = [(o, p) for o, p in ins
            if not any(p2 is not p and bm.part_of(p.d, p2.d) for _o2, p2 in ins)]
    stats['first_steps'] += len(tops)
    rest_ = []
    for o, p in tops:
        q = lv.conseq(o, p)             # asked only here: own moves on the board, and more
        if q is not None and bm.fits(d_now, q.d, q.coarse, q.opt):
            add(o, q, CONSEQ)
        else:
            rest_.append((o, q if q is not None else p))
    if not full and menu.conseq is not None and len(menu.options) <= CHAIN_WAIT:
        # a move and ALL the engine does alone after it, as the position they end in: a piece of
        # the own move may be back where it was (a Regular removed, then the Elite reduced with
        # that very Regular - at the table: the Elite dragged to the casualties), so the own
        # move is not on the board by itself.  Small decisions only (one probe per option)
        for o in lv.candidates()[0]:
            p = lv.plan(o)
            q = None
            if p is not None and p.own and not any(o is o2 for o2, _p in ins):
                # (Menu.net: that whole change with the pieces the renderer and the engine
                # take; without it the option's plan and its consequences together)
                q = menu.net(o) if menu.net is not None else menu.conseq(o) and (
                    list(p.plan) + list(menu.conseq(o)))
            if q:
                stats['conseq'] += 1
                d, coarse, opt = bm.plan_delta(base, list(q))
                d, coarse = bm.bind(d, coarse, d_now)
                opt = bm.add_delta(opt, lv.carry) if lv.carry else opt
                if bm.fits(d_now, d, coarse, bm.add_delta(opt, lv.lag) if lv.lag else opt):
                    add(o, p._replace(d=d, coarse=coarse, soft=opt, full=bm.add_delta(d, opt),
                                      opt=bm.add_delta(opt, lv.lag) if lv.lag else opt), CONSEQ)
    for o, p in (rest_ if (not full and menu.expand is not None) else ()):
        pk = tuple(sorted(prefix + (repr(menu.key(o)),)))
        if pk in seen:
            continue                    # the same choices in another order: searched already
        seen.add(pk)
        stats['expanded'] += 1
        sub = menu.expand(o)
        if sub is None:
            continue
        step = _Step(o, menu, p, MOVES, lv)
        if sub == ROLL:
            ahead.append([step])
            continue
        if depth <= 1:
            # more decisions in one gesture than are followed: this beginning is an answer
            # "ahead" - the rest is judged from the new BASE when it has been applied
            ahead.append([step._replace(how=CUT)])
            stats['depth_cut'] += 1
            continue
        rest, raw2 = [_less(d, bm.project(p.d, p.coarse), p.coarse) if d is not None else None for d in (d_now, raw)]
        # the same POSITION with the same board still to explain, reached by other choices (two figures the game
        # cannot tell apart, each gone where the other might have): searched already - its chains are these chains
        same = menu.position(o) if menu.position is not None else None
        if same is not None:
            same = (same, depth) + tuple(frozenset(d.items()) if d else None for d in (rest, raw2, p.soft))
        if same in seen.gave:
            stats['same_position'] += 1
        else:
            seen.gave[same if same is not None else pk] = _search(
                base.apply(p.plan), rest, sub, depth - 1, stats, p.soft, seen, pk, raw2, lag, faces)[:3]
        f2, w2, a2 = seen.gave[same if same is not None else pk]
        full.extend([step] + chain for chain in f2)
        waits.extend(([step] + chain, dist) for chain, dist in w2)
        ahead.extend([step] + chain for chain in a2)
    if not full and bm.kills(d_now):
        for o, p in lv.wordless():
            if not p.d:                 # no move at all, and the kill is among its closing changes
                add(o, p, CLOSING)
        stats['closing'] += len(full)
    if not full:
        for o, p in rest_:
            if bm.closes_only(bm.residue(d_now, p.d, p.coarse, p.opt)):
                add(o, p, EARLY)
        stats['closed_early'] += len(full)
    if not full:
        # ... only when ONE choice reads the board so: the gesture is still open (a die waits
        # in the play area), several readings mean "wait", not "ask"
        staged = collections.OrderedDict()
        for o, p in rest_:
            if bm.stages_only(bm.residue(d_now, p.d, p.coarse, p.opt)):
                staged.setdefault(menu.key(o), (o, p))
        if len(staged) > 1 and faces is not None:
            # which face did the Elven Ring give?  the reading whose dice pool can hand out
            # the die the board shows staged
            able = [k for k, (o, p) in staged.items() if _can_stage(base.apply(p.plan, True), faces)]
            staged = collections.OrderedDict((k, staged[k]) for k in able)
        if len(staged) == 1:
            add(*(next(iter(staged.values())) + (STAGED,)))
        stats['next_staged'] += len(full)
    if not full and menu.expand is not None and depth > 1:
        full.extend(_summed(lv, depth, stats, seen, prefix, faces, not ins and len(menu.options) <= CHAIN_WAIT))
    if not full:
        found, _done = lv.on_the_way(WAIT_SCAN, 3, not prefix and bm.kills_only(d_now, hands=True))
        waits.extend(([_Step(o, menu, lv.plan(o), MOVES, lv)], dist) for o, dist in found)
        if not waits and menu.expand is not None and depth > 1:
            # ... or to the sum of a chain, when only the die of its first step is still to come
            waits.extend((c, 1) for c in _summed(lv, depth, stats, seen, prefix, faces, False, undiced=True))
    return full, waits, ahead, lv


def _summed(lv, depth, stats, seen, prefix, faces, small, undiced=False):
    """Reading 6 of _search: an answer is the SUM of its steps - the order of the moves and the
    boards between them decide nothing.  The chains that give the board although their first
    step is not on it by itself (undiced: that the board WOULD be with the die of their first
    step staged - no answer yet: the board is on the way to them):
      an option whose move the board shows but for ONE map region, where another step of the
      answer brings as many pieces back or takes them on (boardmodel.lent: two armies moved one
      behind the other with one die) - of the options the board may hold and at most SUM_SCAN
      that name one region more, SUM_STEPS followed;
      in a `small` decision, an option that moves nothing itself and opens a further decision
      of his (the hobbit Guide leaves the Fellowship: where he is placed is the move), when
      the chain holds a move."""
    menu, out, followed = lv.menu, [], 0
    tried = set(id(s.o) for s in lv.ins)        # (whose moves ARE on the board: the readings before)
    still = [o for o in menu.options if not (menu.sig and (menu.sig(o) or _NO_SIG).regions)] if small else []
    for o in lv.candidates()[4][:SUM_SCAN] if undiced else lv.candidates()[0] + lv.candidates()[3][:SUM_SCAN] + still:
        if id(o) in tried or (lv.later and stats['plans'] >= PLAN_BUDGET):
            continue
        tried.add(id(o))
        p = lv.plan(o)
        die = {k: v for k, v in p.d.items() if k[1].startswith('die:')} if (undiced and p is not None) else {}
        now, lies = [bm.add_delta(d, die) if (die and d is not None) else d for d in (lv.d_now, lv.raw)]
        if p is None or not (bm.lent(p.d, now, p.coarse) and followed < SUM_STEPS if p.d
                             else any(o is x for x in still)) \
                or (pk := tuple(sorted(prefix + (repr(menu.key(o)),)))) in seen:
            continue
        sub = menu.expand(o)
        followed += bool(p.d)
        if sub is None or sub == ROLL:
            continue
        seen.add(pk)
        stats['expanded'] += 1
        own = bm.project(p.d, p.coarse)
        rest, raw = [_less(d, own, p.coarse) if (own and d is not None) else d for d in (now, lies)]
        chains = _search(lv.base.apply(p.plan), rest, sub, depth - 1, stats, p.soft, seen, pk, raw, lv.lag, faces)[0]
        out.extend([_Step(o, menu, p, MOVES, lv)] + c for c in chains if p.own or any(s.p.own for s in c))
    return out


def _before_its_die(lv, depth, lag, faces):
    """The moves of an answer made BEFORE its die is laid out (two armies moved, the die still
    in its pool): no reading of the board as it lies, and no die of the person has left its
    pool - but with the die of an option staged the board is an answer.  It is on the way to
    that answer: -> [(its chain, 1)], as _search's wait candidates."""
    menu, (regions, dice), tried = lv.menu, bm.observed(lv.d_now), set()
    for o in () if dice else menu.options:
        s = menu.sig(o) if menu.sig else None
        if s is not None and (not s.die or s.die in tried or not s.regions or not s.regions <= regions):
            continue                    # (the quick filter: that die is tried, or its moves are not all there)
        p = lv.plan(o)
        stage = frozenset((k, v) for k, v in p.d.items() if k[1].startswith('die:')) if p is not None and p.own else ()
        if not stage or stage in tried:
            continue
        tried |= {stage, s.die if s is not None else None}      # (one search for each die: its die is the question)
        full, _waits, ahead, _lv = _search(lv.base, bm.add_delta(lv.d_now, dict(stage)), menu, depth, lv.stats,
                                           None, _Seen(), (), None, lag, faces)
        if full or ahead:
            return [((full or ahead)[0], 1)]
    return []


def _choice_key(chain):
    """Chains that make the same choices in another order are one choice."""
    return tuple(sorted(repr(s.menu.key(s.o)) for s in chain))


def _by_outcome(groups):
    """Different chains that leave the engine in the same state are one choice."""
    out = collections.OrderedDict()
    for key, chain in groups.items():
        fn = chain[-1].menu.outcome
        k = fn(chain[-1].o) if fn is not None else None
        if k is None:
            return groups
        out.setdefault(k, (key, chain))
    return collections.OrderedDict(out.values())


def _by_faces(groups, base, now):
    """Readings that differ only in the face a staged or used die shows: keep
    those whose dice show what the board shows (all of them when none does)."""
    want = bm.moved_faces(base, now)[0]
    out = collections.OrderedDict()
    for key, chain in groups.items():
        t = base
        for s in chain:
            t = t.apply(s.p.plan, with_optional=True)
        if bm.moved_faces(base, t)[0] == want:
            out[key] = chain
    return out or groups


def _extendable(chain, root):
    """(is the board a strict part of another target of the same decision:
    True / False / None = not settled, can a further decision of the same
    human continue the gesture, the name of the first such target found)."""
    last = level = chain[-1].level
    # (a kill with nothing beside it in the decision it is read in ends nothing there: every
    # OTHER answer of that die and card is bigger - the committed one, its die tidied early, is not)
    found, done = last.on_the_way(EXTEND_SCAN, 2, bm.kills_only(last.d_now, hands=True))
    found = [f for f in found if f[0] is not chain[-1].o]
    if not found and last is not root:
        level = root
        found, d2 = root.on_the_way(EXTEND_SCAN)
        done = done and d2
    toward = str((level.menu.name or repr)(found[0][0])) if found else None
    more = False
    s = chain[-1]
    if s.menu.expand is not None:
        sub = s.menu.expand(s.o)
        if sub is not None and sub != ROLL:
            lv = _Level(last.base.apply(s.p.plan), {}, sub, last.stats)
            for o2 in sub.options[:EXTEND_SCAN]:
                p2 = lv.plan(o2)
                if p2 is not None and p2.own:
                    more = True
                    break
    return (True if found else False if done else None), more, toward


def _commit(chain, root, stats, check_extend, ahead=False, shut=frozenset(), tidied_last=False):
    ext, more, toward = _extendable(chain, root) if check_extend else (False, False, None)
    declines = chain[-1].how == CLOSING or bm.kills_only(chain[-1].p.d, hands=True)
    if (bm.kills(root.d_now) and not bm.kills_only(root.d_now, hands=True) and (tidied_last or not declines)) \
            or any(ch.pid in shut for s in chain for ch in s.p.plan if not ch.opt):
        # the die / card window was closed on the board beside the moves of a complete answer,
        # or the marker the answer places was turned to its side (the reveal flip at the end of
        # a placement): the answer is final.  Where the answer only DECLINES an optional rest by
        # the kill (its last step is the kill and nothing else), the ORDER says how fast it
        # closes, never what is answered: tidied AFTER a move it is his closing gesture and
        # final; tidied BEFORE the moves (the die discarded first, the card played and put away
        # before its own moves) - or with NO piece moved - the kill ends nothing: the board is a
        # strict part of every bigger answer of that die and card, and the quiet time closes it
        ext = False
    # the answer's own plan closes its die / card window (the optional kill) and the board still
    # holds HIS DIE in the play area - put there with this answer, or lying there since the
    # decision opened (the die of a two-step action): the action is not finished on the board.
    # Putting that die away is HIS (owner: "it also killed my die for me without my consent"):
    # whatever step follows, the answer waits for it (ready).  A card alone holds nothing back
    # (his combat card in its slot while he defends: no die of his is out, the battle is not
    # his action - the casualties that end it are taken as they are made)
    last = chain[-1]
    closing = [ch for ch in last.p.plan if ch.opt and ch.pid is not None and ch.zone is not None
               and bm.zone_id(ch.zone) in bm.USED_DICE and bm.kind_of(ch.pid).startswith('die:')]
    waits = (not bm.kills(root.d_now) and bool(closing) and not (root.raw and bm.kills(root.raw))
             and (any(v > 0 and z in bm.PLAY_AREA and k.startswith('die:') for (z, k), v in root.d_now.items())
                  or any((root.base.p.get(ch.pid) or ('',))[0] in bm.PLAY_AREA for ch in closing)))
    done, rest = [], root.d_now         # each step's pick out of its OWN share of the board
    for s in chain:
        done.append(bm.performed(rest, s.p.plan))
        rest = bm.sub_delta(rest, {(z, k): v for z, k, v in done[-1]})
    return Verdict(COMMIT, chain=[s.o for s in chain], extendable=ext, more=more, ahead=ahead,
                   unfinished=waits, performed=done, toward=toward if ext else None,
                   conseq=[s.how == CONSEQ for s in chain], closing=chain[-1].how == CLOSING,
                   closed_early=any(s.how == EARLY for s in chain),
                   next_staged=chain[-1].how == STAGED, key=_choice_key(chain),
                   delta=dict(root.d_now), stats=stats)


def untidied(base, now, opened=None):
    """What ONE board says about the order of a gesture: True - it shows a move of the game
    (more than dice and cards laid into the play area) and no die or card tidied; False -
    neither; None - a die or card is tidied (what stood before it stays the fact).  The caller
    keeps the last answer that was not None with the open gesture: that is judge()'s
    `tidied_last` for the boards that show the tidying."""
    d = bm.delta(base, bm.arrived(now, opened, base) if opened is not None else now)
    return None if bm.kills(d) else bool(d) and not bm.stages_only(d, cards=True)


def _as_it_lies(base, now, menu, depth, check_extend):
    """A board that has not changed since the decision opened and is not BASE, read as it lies:
    what was not yet BASE may be HIS doing - the rest of a gesture that ran ahead of a roll.
    -> the verdict when that is an answer or on the way to one (it is then never "nothing": a
    closing word would commit the option without a move), else None.  Such an answer is
    `untouched`: what lies there may as well be the client's own doing or a piece in transit,
    so it is never applied by time, only when he says so.  A die that only lies killed says
    nothing."""
    if bm.kills_only(bm.delta(base, now)):
        return None
    v = judge(base, now, menu, depth, check_extend)
    if v.kind in (COMMIT, WAIT) or (v.kind == UNDECIDED and not v.words):
        v.untouched = v.kind == COMMIT
        return v
    return None


def _unfiltered(base, now, menu, exact, depth, check_extend, opened):
    """The verdict on the moves that ARE the board and were never tried, judged without the
    quick filter (the filter is no judge); None when a die's face stands against them."""
    v = judge(base, now, only(menu, exact), depth, check_extend, opened)
    if v.kind == MISMATCH:
        return None
    v.extendable = v.extendable or None             # (what else it may grow into was not looked at)
    return v


def only(menu, options):
    """The Menu of a decision narrowed to some of its options, read without the quick filter:
    the board against exactly these answers (an answer he named by a click or a word)."""
    return Menu(options, menu.plan, None, menu.key, menu.name, menu.expand, menu.conseq, menu.outcome,
                menu.side, menu.net, menu.faults, menu.position)


def named(base, now, menu, o, opened=None):
    """The answer the human NAMED (a click, a word) as the board shows it: judge()'s COMMIT for
    that one option (with what follows it on the board) - or, while the board is only on the
    way to it (his pieces lie there, the die or a part of the move is missing), a COMMIT of the
    option alone whose `performed` is what he did of it.  And whatever else lies moved: when he
    took the die o spends out of its pool - laid out, or put away already - o is made with HIS
    die.  None: the board shows none of it.  (now: without what fits no move at all - the caller's.)"""
    v = judge(base, now, only(menu, [o]), opened=opened)
    if v.kind == COMMIT:
        return v
    d = v.delta or bm.delta(base, bm.arrived(now, opened, base) if opened is not None else now)
    done, own = bm.performed(d, menu.plan(o)), bm.plan_delta(base, menu.plan(o))[0]
    # (on the way to o ITSELF - what lies there for the decision after it is not o's -, or the
    # board shows what o, an option without a move of its own, leaves: the die he laid out for it)
    part = (v.kind == WAIT or (v.kind == UNDECIDED and any(w is o for w in v.words))) and (
        done or any(d.get(k, 0) * n > 0 for k, n in own.items()))
    return given(base, menu, o, [done]) if part or any(
        n < 0 and k[1].startswith('die:') and d.get(k, 0) < 0 for k, n in own.items()) else None


def given(base, menu, o, performed=()):
    """A COMMIT of the ONE option the human named - made with what he did of it on the board (named), or with
    nothing of it there - as a later board is read against it (again): its choice, and what its own move changes."""
    return Verdict(COMMIT, chain=[o], performed=list(performed), key=(repr(menu.key(o)),),
                   delta=bm.plan_delta(base, menu.plan(o) or ())[0])


def reads(base, now, menu, pid, opened=None, acts=False):
    """WHAT ONE PIECE of the human means to a decision - `now`: the board on which that piece
    (pid) alone lies moved.  The verdict per piece, one of three: True - an answer or the
    beginning of one (no mismatch); False - it is read and fits no move: at an action decision
    (acts) every other piece, else one taken from where an option's OWN gesture takes a piece
    of its kind (its plan without the optional changes: what the game carries out by itself
    after his answer is no gesture of his), lying where no move puts it; None - neither: the
    decision does not read it."""
    fits = judge(base, now, menu, opened=opened).kind != MISMATCH
    if fits or acts:
        return fits
    left = {base.place(pid): -1}        # (where the game has it: the place and kind it left)
    zone = next(iter(left))[0]
    lv = _Level(base, bm.delta(base, bm.arrived(now, opened, base) if opened is not None else now), menu,
                collections.Counter())
    for o in menu.options:
        s = menu.sig(o) if menu.sig else None
        if s is not None and s.regions is not None and bm.is_map_region(zone) \
                and bm.region_of(zone) not in s.regions:
            continue                    # (its own moves touch other map regions, exactly: Sig)
        p = lv.plan(o)
        if p is not None and any(bm.project(p.d, p.coarse).get(k, 0) < 0 for k in bm.project(left, p.coarse)):
            return False
    return None


def as_it_stands(menu):
    """The ONE option of a small decision that leaves the board exactly as it stands (its plan
    changes nothing), while every other option has a move of its own: what the board answers
    when his time to change it has ended (the Guide who sits in the Guide box).  None: no such
    option, several, or an option whose plan is not known."""
    plans = [(o, menu.plan(o)) for o in menu.options[:CHAIN_WAIT + 1]]
    still = [o for o, pl in plans if pl is not None and not pl]
    if len(plans) > CHAIN_WAIT or len(still) != 1:
        return None
    return still[0] if all(pl and not all(ch.opt for ch in pl) for o, pl in plans if o is not still[0]) else None


def judge(base, now, menu, depth=MAX_DEPTH, check_extend=True, opened=None, tidied_last=False):
    """The verdict for one board (see the module text).  tidied_last: the die / card it shows tidied was
    tidied AFTER a move of the gesture (untidied)."""
    stats, seen = collections.Counter(), bm.arrived(now, opened, base) if opened is not None else now
    d_now = bm.delta(base, seen)                    # what lagged and was not touched: as arrived
    if not d_now:                                   # untouched, or only what was not yet BASE
        return (seen is not now and _as_it_lies(base, now, menu, depth, check_extend)) \
            or Verdict(NOTHING, stats=stats)
    lag, raw = {}, None
    if seen is not now:
        lag, raw = bm.delta(now, seen), bm.delta(base, now)     # ... or as it lies
    shut = bm.turned(now, opened, base) if opened is not None else frozenset()
    faces = (base, bm.moved_faces(base, seen)[0])   # (a die still in transit is not the human's)
    full, waits, ahead, lv = _search(base, d_now, menu, depth, stats, None, _Seen(), (), raw, lag, faces)
    waits = waits if full or waits or ahead else _before_its_die(lv, depth, lag, faces)
    if full:
        groups = collections.OrderedDict()
        for chain in full:
            groups.setdefault(_choice_key(chain), chain)
        if len(groups) > 1:
            groups = _by_outcome(groups)
            stats['same_outcome'] += 1 if len(groups) == 1 else 0
        if len(groups) > 1:
            groups = _by_faces(groups, base, seen)
            stats['by_faces'] += 1 if len(groups) == 1 else 0
        if len(groups) == 1:
            return _commit(next(iter(groups.values())), lv, stats, check_extend, shut=shut, tidied_last=tidied_last)
        chains = list(groups.values())
        closing = all(c[-1].how == CLOSING for c in chains)
        return Verdict(UNDECIDED, choices=[[s.o for s in c] for c in chains[:MAX_CHOICES]],
                       names=[[s.menu.name(s.o) for s in c] for c in chains[:MAX_CHOICES]],
                       words=[c[-1].o for c in chains] if closing else None,
                       sentence='The board fits %d different choices: %s.' % (
                           len(chains), '; '.join(collections.OrderedDict.fromkeys(
                               ' then '.join(s.menu.name(s.o) for s in c) for c in chains[:4]))),
                       stats=stats)
    words = [o for o, _p in lv.wordless()]
    if waits:
        chain, _dist = min(waits, key=lambda w: w[1])
        return Verdict(WAIT, nearest=[s.o for s in chain], words=words, delta=d_now, stats=stats)
    if ahead:
        # WHOLE BOARD OR NOTHING.  The board holds the moves of an option (a chain) AND MORE.  Where
        # what follows is more decisions of his than one search follows (CUT), the beginning is the
        # answer "ahead" - its other first steps on the board are its own later steps - unless a
        # piece of the other side lies moved as well, which no decision of his explains.  Where a
        # roll, a draw or the other seat follows, what would belong beyond it is no chain of his
        # answers: the option is not taken beside it, and the pieces it does not move are named
        top = max(ahead, key=lambda c: (c[-1].how == CUT, len(c), len(c[-1].p.d)))
        last = top[-1]
        rest = bm.residue(last.level.d_now, last.p.d, last.p.coarse, last.p.opt)
        if last.how == CUT and not (menu.side and any(bm.owner(k) not in (None, menu.side) for (_z, k) in rest)):
            return _commit(top, lv, stats, False, ahead=True)
        return Verdict(MISMATCH, nearest=[s.o for s in top], stats=stats, sentence=bm.sentence(
            last.level.d_now, last.p.full, last.menu.name(last.o), last.p.coarse))
    if words:
        return Verdict(UNDECIDED, choices=[[o] for o in words[:MAX_CHOICES]], words=words, delta=d_now,
                       names=[[menu.name(o)] for o in words[:MAX_CHOICES]],
                       sentence='The board shows no move of its own for: %s. It needs the word '
                                'or the click.' % '; '.join(menu.name(o) for o in words[:4]),
                       stats=stats)
    if menu.expand is not None and len(menu.options) <= CHAIN_WAIT:
        # the move of the decision that FOLLOWS shown before this decision's own (the new Guide
        # seated before the old one is taken away): on the way to that chain
        for o in menu.options:
            p = lv.plan(o)
            sub = menu.expand(o) if (p is not None and p.own) else None
            if sub is None or sub == ROLL:
                continue
            stats['expanded'] += 1
            lv2 = _Level(base.apply(p.plan), {}, sub, stats)
            for o2 in sub.options[:CHAIN_WAIT]:
                p2 = lv2.plan(o2)
                if p2 is not None and p2.own and bm.between(d_now, bm.add_delta(p.full, p2.full),
                                                           p.coarse | p2.coarse):
                    return Verdict(WAIT, nearest=[o, o2], words=words, stats=stats)
    # nothing fits: name the nearest target among the most alike options
    regions, _dice = bm.observed(d_now)
    scored = []
    for i, o in enumerate(menu.options):
        s = menu.sig(o) if menu.sig else None
        alike = (len(s.regions & regions) - len(s.regions ^ regions)
                 if s is not None and s.regions is not None else 0)
        scored.append((-alike, i, o))
    scored.sort(key=lambda t: t[:2])
    best = None
    # ... and among every option already rendered for this board (the prefilter let it through),
    # each against its move with and without its optional part and its consequences: a card
    # played and killed is nearer to "play that card" than to an attack that stages the die
    near = [o for _a, _i, o in scored[:NEAREST_SCAN]]
    have = set(id(o) for o in near)
    near += [o for o in menu.options if id(o) in lv._plans and id(o) not in have]
    exact = []
    for o in near:
        p = lv.plan(o)
        if p is None or not p.d:
            continue
        p = lv._conseq.get(id(o)) or p
        dist = min(bm.distance(bm.project(d_now, p.coarse), bm.project(t, p.coarse)) for t in (p.d, p.full))
        exact += [o] if dist == 0 else []
        if best is None or dist < best[0]:
            best = (dist, o, p)
    if best is None:
        return Verdict(MISMATCH, sentence='The board was changed, and no legal move changes it.',
                       stats=stats)
    v = _unfiltered(base, now, menu, exact, depth, check_extend, opened) if (exact and menu.sig) else None
    if v is not None:
        return v
    p = best[2]                 # (said against the move WITH the optional changes when that is nearer)
    far = lambda t: bm.distance(bm.project(d_now, p.coarse), bm.project(t, p.coarse))
    target = p.full if far(p.full) < far(p.d) else p.d
    on, own = bm.project(d_now, p.coarse), bm.project(p.d, p.coarse)
    if not any(v > 0 and on.get(k, 0) > 0 and (not k[1].startswith('die:') or (
            own.get(k, 0) > 0 and k[0] not in bm.PLAY_AREA)) for k, v in bm.project(p.full, p.coarse).items()):
        # a move is NAMED only when a piece lies where that move puts it - a move he has begun.  (An action die
        # begins every move of its face and names none: it counts only where it is the move itself, a discard.)
        # Else only what lies moved is said
        return Verdict(MISMATCH, sentence='No legal move moves: %s.' % bm.moved_words(d_now), stats=stats)
    text = bm.sentence(d_now, target, menu.name(best[1]), p.coarse)
    if best[0] == 0:            # every piece lies where that move puts it: a die's face is against it
        text = ('The pieces are where "%s" puts them, but an action die shows a face that move does '
                'not give it. Turn the die back, or use the word or the click.' % menu.name(best[1]))
    return Verdict(MISMATCH, nearest=[best[1]], sentence=text, stats=stats)


def ready(verdict, polls_equal, seconds_quiet, k=K_POLLS, q=Q_SECONDS):
    """May a COMMIT verdict be applied now?  polls_equal = polls in a row that
    gave this same board; seconds_quiet = time since the last client line.
    THE QUIET TIME (q seconds) is for an answer that can still GROW or whose
    growth is not settled (a strict part of another target: `extendable` is
    not False).  Everything else k polls - a card's option with the card on
    the table, a step that does not end its action.  NEVER by time: an answer
    he has not closed (it ends his action, and his die still lies in the
    play area - Verdict.unfinished): putting the die away is his, and
    makes the answer final at once (as a closing word or a click does); and
    an `untouched` answer: time does not make a board his that he has not
    touched."""
    if verdict.kind != COMMIT or verdict.untouched or verdict.unfinished or polls_equal < k:
        return False
    return verdict.extendable is False or seconds_quiet >= q


def again(base, now, menu, committed, depth=MAX_DEPTH, opened=None, tidied_last=False):
    """The same decision judged again AFTER `committed` (a COMMIT verdict of it) was applied, with
    the BASE and the Menu it was judged with (the AI has not acted on it, no roll was consumed):
      same     the board still shows the committed choice (also when it now
               holds the moves of the decision that follows it as well)
      extend   the board now shows another choice and still holds everything
               the committed board held (Verdict.chain = the choice to apply INSTEAD)
      change   the board now shows another choice, and something the
               committed board held is gone (a placement corrected, an undo)
      else     judge()'s own kinds: wait / undecided / mismatch / nothing."""
    v = judge(base, now, menu, depth, opened=opened, tidied_last=tidied_last)
    if v.kind == COMMIT:
        # the committed choices are all still in the reading: what the board holds beyond them
        # is the NEXT decision's (the core has it open and judges it from its own opening),
        # never a change of the committed answer - made with the same pieces (another pick inside
        # the same choice is a change: the engine holds the pieces of the first pick).  Another
        # choice on an `untouched` board is nothing: he has not said so
        left, made = list(v.key or ()), list(v.performed)
        held = all(k in left and (left.remove(k) or True) for k in (committed.key or ())) \
            and all(p in made and (made.remove(p) or True) for p in committed.performed)
        v.kind = (SAME if held else NOTHING if v.untouched else
                  EXTEND if bm.inside(committed.delta, v.delta) else CHANGE)
    return v
