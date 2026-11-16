"""The answer on the board: the open decision of the person as the matcher sees it (the gesture),
his board answers that were taken while the game has not gone on past them (the window), and
what on the board is HIS meanwhile.

ONE rule, held here and asked by everybody: a piece the person was SEEN to move (link.track) is
his - PER PIECE (His.pieces) - until end() says why it is his no longer.  It is kept where he
put it and read by the decision of his that can read it (aside(): what a decision reads).  The
COMPARISON is the only thing that ever moves a piece he moved: end() tells it that it may, and
the pass that carries the piece says the one sentence of that ending (explain).  settle() is the
one test of whether a piece of his may stay where it lies under the game as it is now, asked by
every pass that looks at the whole board; committed() says what his own answer does to what he
had moved for it.  owned() is the comparison's question; gesture() opens a decision on the
board; close() ends the window.

Judged here, applied by the loop: watch gives the steps of a ready answer; again says when the
board no longer shows his taken answers - they are then taken back, and his decision opens
again on the board as it lies.  A BoardAnswers imports no other part of the loop: it is handed
the Voice, the client link, the person's side, the position and the clock (callables), and the
loop's two probes - apply(state, spec, act) -> the state after one step, opens(position) ->
the decision the step machine opens there.
"""
import collections
import functools

from . import boardmodel as bm
from . import emission, layout, match, panel, reconcile, words
from . import emitter as em
from . import menu as mn
from .randomness import NeedClientCall

RANDOM_CALLS = ('!CALL draw_', '!CALL hunt_tile', '!CALL random_companion', '!CALL combat_roll',
                '!CALL roll_action_dice', '!CALL wwsat')
# what on the board is the person's now (BoardAnswers.owned).  whole: the comparison keeps off
# the board; pieces: what stays where it lies whenever the comparison does look - {piece: the
# place he took it from, or None}; back: what has ceased to be his and waits for the pass that
# carries it - {piece: the place it came from}
Owned = collections.namedtuple('Owned', 'whole pieces back')
# why ONE piece is his: the place he took it from, when the bridge first saw it moved (since:
# the steps committed then; seen: the client's request), beside: it lay beside an answer he
# gave by a word or a click, and his decision that follows has yet to say whether it reads it;
# and turned: he handled it where it lies (turned its side or a die's face, took it away and brought it back) -
# it is his although it lies where he took it from, until a pass finds it where the game has it
Own = collections.namedtuple('Own', 'frm since seen beside turned', defaults=(False,))
# one call of end() whose pieces the comparison may move: its reason, the pieces that ceased to
# be his ({piece: Own}), and `said` when the sentence that tells him is not the pass's to say
Ended = collections.namedtuple('Ended', 'reason pieces said')
# the reasons of end() that owe him no sentence of the pass (those that do are the keys of
# words.PUT_BACK): his answer took the pieces; he took it back by the word or the panel; the
# client started another game; the game is over; the piece lies where the game has one of its
# kind; he put it back where he took it from himself - and, said by what happens with them: a
# result of the client that the game does not use; a card the game refused; his word "back"
ANSWERED, TAKEN_BACK, RESTARTED, OVER, PLACED, HOME = 'answered', 'taken back', 'restarted', 'over', 'placed', 'home'
UNUSED, REFUSED, BACK = 'unused', 'refused', 'back'


class His(object):
    """What on the board is the person's, piece by piece, and what has just ceased to be."""
    __slots__ = ('pieces', 'ended', 'pending')

    def __init__(self):
        self.pieces = {}            # {piece: Own}: every piece he was seen to move, until its ownership ends
        self.ended = []             # [Ended]: until the next pass of the comparison has carried them
        self.pending = None         # after an answer READ from the board: what the comparison wanted elsewhere before it


class Gesture(object):
    """The open decision of the person as the matcher sees it."""
    __slots__ = ('id', 'root', 'side', 'spec', 'pos', 'memo', 'at_pos', 'next', 'opening', 'menu',
                 'elsewhere', 'key', 'equal', 'at', 'v', 'moved', 'told', 'failed', 'noted', 'options', 'taken', 'means')

    def __init__(self, id=None, root=None, side=None, spec=None, pos=None, at=0.0, opening=None,
                 menu=None, key=None, equal=0, v=None, options=()):
        self.id, self.root, self.side, self.spec, self.pos = id, root, side, spec, pos
        self.options = options      # the legal actions of the decision
        self.memo, self.at_pos, self.next = {}, {}, {}      # what the probes found, per option
        self.opening, self.menu = opening, menu     # its tables and its Menu (None: not built)
        self.elsewhere = frozenset()        # pieces this decision does not read (aside, took, read; menu.owned)
        self.key, self.equal, self.at, self.v = key, equal, at, v   # the board, since when, its verdict
        self.means = (None, {})     # (that board, what each piece of his means to this decision: _means)
        self.taken = None           # the verdict his answer to it is made with (his pieces on the board)
        self.moved = False          # the board last seen with nothing tidied showed a move (match.untidied)
        self.told = frozenset()     # the pieces of his, each as it lay, that a sentence was said about
        self.failed = None          # (the board on which an error stopped his answer, the sentence said)
        self.noted = 0              # how many of the Menu's faults are in the log


class Taken(object):
    """One answer of his that was taken: the gesture it was read with (its verdict: Gesture.taken),
    the position before it, how many steps it made, and the pieces of his it was made with (read
    from the board: all the decision read) - {piece: (its Own, where he had put it)}."""
    __slots__ = ('gesture', 'before', 'entries', 'pieces')

    def __init__(self, gesture, before):
        self.gesture, self.before, self.entries, self.pieces = gesture, before, 0, {}


class Window(object):
    """His answers in a row - read from the board, clicked or typed; no step of another seat, no
    roll between them - while the game has not gone on past them: until then the board is read
    from the latest of them that can read it (BoardAnswers.again)."""
    __slots__ = ('answers', 'judged', 'searched')

    def __init__(self):
        self.answers = []           # [Taken], in the order they were taken
        self.judged = None          # the board again() last read against them (menu.board_key)
        self.searched = None        # the poll in which what follows it was first held back


def _took(g, mine, board):
    """Of the pieces he moved for an answer (g: its gesture, with the verdict it is made with):
    those the answer TOOK - the pieces its steps' plans name (as far as they can be followed
    without a roll) that lie where a plan puts them; and, counted by the place and kind the
    matcher counts pieces in, as many as he showed for a plan that names only a class
    (Verdict.performed) or another piece of the same kind (which die of a face, which of
    several units is his to pick) - first those he took from a place a plan takes them from.  The rest is the next
    decision's to read.  -> (those; the die it spends that he had put away already: used, his until its action ends)."""
    took, free, zone, now, menu, v = set(), collections.Counter(), board.zone, mn.table(board), g.menu, g.taken
    base, left, used, faces = g.opening.base if g.opening is not None else None, set(), set(), set()
    for i, o in enumerate(v.chain if v is not None and menu is not None else ()):
        faces.add(mn.die_letter(g.root, g.side, o))
        for ch in [ch for ch in menu.plan(o) or () if ch.pid is not None and ch.zone is not None]:
            if zone.get(ch.pid) == ch.zone:
                took.add(ch.pid)
            else:
                free[base.apply([ch], True).place(ch.pid)] += 1
        done = v.performed[i] if i < len(v.performed) else ()
        free.update({(z, k): n for z, k, n in done if n > 0})
        left.update((z, k) for z, k, n in done if n < 0)
        menu = menu.expand(o) if menu.expand is not None else None
        if menu is None or menu == match.ROLL:
            break
    for pid in sorted((p for p in mine if p not in took and p in now.p),
                      key=lambda p: (not left or p not in base.p or base.place(p) not in left, p)):
        key, staged = now.place(pid), (bm.zone_id(reconcile.STAGED), now.place(pid)[1])
        if free[key] > 0:
            free[key] -= 1
            took.add(pid)
        elif free[staged] > 0 and (board.face.get(pid) or '-') in faces:    # (the die the answer spends, put away by him
            free[staged] -= 1                                       # already: USED by it, and his until its action ends)
            used.add(pid)
    return [p for p in mine if p in took], used


def _about(zone, owns, ops=None):
    """The pieces a sentence is about, as its record holds them (voice.Voice.record): those of
    `owns` ({piece: Own}) as they lie - or the pieces the `ops` of a pass move, each with where
    the pass carries it."""
    out = []
    for op in ops if ops is not None else [('lies', None, p) for p in sorted(owns)]:
        own = owns.get(op[2])
        out.append({'pid': op[2], 'lay': zone.get(op[2]), 'was': own.frm if own is not None else None,
                    'to': op[4] if op[0] == 'move' else None, 'by': 'person' if own is not None else None,
                    'seen': own.seen if own is not None else None})
    return out


def _put_back(voice, board, e, ops):
    """Say the sentence of the ending `e`, whose pieces the `ops` carry back - to another place,
    or turned back where they lie (a side, a die's face) -> its number."""
    return voice.say_about(words.put_back(e.reason, bm.ops_words(mn.table(board), ops)), reason=e.reason,
                           pieces=_about(board.zone, e.pieces, ops), since=min(own.since for own in e.pieces.values()))


def _acts(g):
    """THE one place that says whether the open decision (g: its gesture) is an ACTION decision:
    he chooses what to do with an action die, the game settled.  (Every other decision of his -
    a battle, the second move of a die, a card's choice, the Eye allocation, a discard - cannot
    read what none of its options moves: BoardAnswers.aside.)"""
    return g.pos.phase == 'actions' and reconcile.settled(g.root)


def _means(g, board, mine):
    """WHAT EACH PIECE OF HIS MEANS to the open decision (g: its gesture; mine: the pieces of his
    it reads) - THE verdict per piece, asked ONCE a board, of each piece ALONE, all else of his
    where the game has it (match.reads) -> {piece: True - an answer or the beginning of one;
    False - it is read and fits no move (no answer is taken beside it, he is told once which);
    None - not read: at a decision that is no action decision (_acts) a piece no option's own
    gesture takes from there waits for the first decision of his that can read it}."""
    key = mn.board_key(board)
    if g.means[0] != key and g.opening is not None and g.menu is not None:
        g.means = (key, {p: match.reads(g.opening.base, mn.reading(board, g.opening, g.elsewhere | (set(mine) - {p})),
                                        g.menu, p, g.opening.opened, _acts(g)) for p in mine if p in g.opening.base.p})
    return g.means[1] if g.means[0] == key else {}


def _spots(L, board, owns):
    """Those of the person's pieces (owns: {piece: Own}) that lie at - or that he took from - A PLACE AN ANSWER OF
    HIS CHAIN CHANGED (L: the window of his taken answers), as the GAME sees places - a region's count of a kind, a
    box, a track step: what the verdict each answer was taken with shows changed (Verdict.delta).  A piece he TURNED
    where it lies (a die's face) lay there as another kind: its place is the zone.  A die or card
    he only put away is none: what the game carries out by itself at an action's end is no gesture of his.
    -> {piece: the last answer that changed such a place of it (its number in the window)}"""
    spots, now = {}, mn.table(board)
    for i, a in enumerate(L.answers if L is not None else ()):
        changed = a.gesture.taken.delta if a.gesture.taken is not None else ()
        spots.update(dict.fromkeys(list(changed) + [zone for zone, _kind in changed], i))
    was = now.apply([bm.move(p, own.frm) for p, own in owns.items() if p in now.p and own.frm is not None])
    tidy = lambda p: [op for op in [('move', None, p, owns[p].frm, board.zone[p])] if reconcile.tidies(op) and not reconcile.undoes(op)]
    at = {p: (was.place(p), now.place(p)) + ((now.place(p)[0],) if owns[p].turned else ()) for p in owns if p in now.p and not tidy(p)}
    return {p: max(spots.get(k, -1) for k in places) for p, places in at.items() if any(k in spots for k in places)}


def _ahead(L, g):
    """Is the board AHEAD of the game by his own doing?  The answer he gave last (g: its gesture; L: the window)
    was read off the board as a chain of which only its own step is taken - he named that step by a click or a
    word: it answers ONE decision.  The decision of his that follows reads the rest in its own way; until it has,
    the comparison keeps off the board (owned), and nothing is moved for a step of that chain."""
    return bool(L is not None and L.answers and L.answers[-1].gesture is g and g.taken is not None
                and len(g.taken.chain) > L.answers[-1].entries)


def _choice_names(g, v):
    """The name of each choice of an undecided board (the matcher's own, Verdict.names)."""
    names = getattr(v, 'names', None) or [[g.menu.name(a) for a in chain] for chain in v.choices]
    return [' and '.join(chain) for chain in names]


class BoardAnswers(object):
    def __init__(self, voice, link, human, position, now, apply, opens):
        self.voice, self.link, self.human = voice, link, human
        self.position, self.now = position, now     # callables: the game position, the clock
        self.apply, self.opens = apply, opens       # the loop's probes (see the head of this file)
        self.gest = None            # the open decision as the matcher sees it (a Gesture)
        self.chain = []             # steps of a board answer still to apply: [(option, action, fix)]
        self.window = None          # his taken board answers while a change of the board takes them back
        self.his = His()            # what on the board is his, and what has just ceased to be

    def queue(self, steps):
        """The steps of his board answer: the first is applied now (-> it, or None), the others
        wait for the decisions that follow."""
        self.chain = list(steps[1:])
        return steps[0] if steps else None

    def has(self, o):
        """The pieces the person has moved that his decision `o` (it, or its gesture) READS:
        {piece: Own} - none while the decision has no gesture."""
        g = self.gest
        return {p: own for p, own in self.his.pieces.items() if p not in g.elsewhere} \
            if g is not None and o is not None and g.id == o.id else {}

    # ------------------------------------------------ what is his
    def owned(self, o, pos, built=True):
        """THE question of the comparison: what on the board is the person's now.  o: his
        decision that waits for him, or None; built: the board has equalled the game once
        (one built from a saved position holds nothing he did).  whole - his decision is open,
        or the steps of his answer are being applied.  pieces - what stays where it lies: what
        the game owes him a decision about (menu.PICKS), and what he was seen to move
        (His.pieces).  back - what has ceased to be his and the next pass carries (His.ended)."""
        pieces = dict.fromkeys(mn.owned(pos.state, self.link.board, pos.phase if built else None, self.human))
        pieces.update((p, own.frm) for p, own in self.his.pieces.items())
        return Owned(bool(self.human is not None and (self.chain or o is not None or _ahead(self.window, self.gest))),
                     pieces, {p: own.frm for e in self.his.ended for p, own in e.pieces.items()})

    def _staging(self, pos):
        """The pieces in the play area that an action the person still has in this turn can
        use: a die of his that is rolled and that the game has not spent (by THE binding of the
        dice of this poll: the view's); a card of his hand while he has such a die, or while
        his dice of this turn are still to be rolled."""
        board, state = self.link.board, pos.state
        if pos.phase == 'end' or pos.over is not None:
            return frozenset()
        dice = {p for p, (side, i) in self.link.view.die_id_map.items()
                if side is self.human and not state.dice[side][i].used} if pos.phase == 'actions' else set()
        hand = {str(self.link.view.card_pid_map.get(c)) for c in state.hands[self.human].hand} \
            if dice or pos.phase != 'actions' else ()
        return frozenset(p for p, z in board.zone.items() if z in reconcile.DISPLAY and (p in dice or p in hand))

    def aside(self, g, pieces, staging):
        """WHAT A DECISION READS (g: its gesture): every piece he moved - of either side - with
        two exceptions.  The first, here -> those of `pieces` it sets aside: a die or card he
        laid into the play area for an action he still has (staging), which no option of the
        decision uses - nor his decision that follows an option at once (the die an Elven Ring
        has turned): it is that action's.  The second is _means(): the verdict per piece."""
        def nxt(a):         # (his decision that follows the option a at once: its state and options)
            d = self.probe_next(g, self.probe_apply(g, g.root, [a]))
            return (d[2], d[3]) if d is not None and d[0] is g.side else None
        return frozenset(p for p in pieces if p in staging and not mn.uses(
            g.root, g.side, g.options, self.link.board, p, self.link.view.card_pid_map, nxt))

    def took(self, pieces, o, rolling, late=()):
        """The pieces the person was seen to move since the last poll (link.track: {piece: where
        it lay}): each is his from now.  o: his decision that is open; rolling: a client call is
        open for the step he answered it with (its dice are rolled) - what he moves now lies
        beyond it; late: those he moved AFTER he gave the answer that is being carried out - the
        decision he answered never read them (early pieces).  One back where he
        took it from is his no longer (HOME) - but one he turned there.  A piece of an
        answer of his that was taken stays that answer's when he only puts it away; moved on - or
        turned, or handled again, where it lies -, it is his like any piece he moves (what the
        board then says of his taken answers: again)."""
        h, L, pos, zone = self.his, self.window, self.position(), self.link.board.zone
        g = self.gest if o is not None and self.gest is not None and self.gest.id == o.id else None
        staging = self._staging(pos) if pieces else ()
        for pid, frm in sorted(pieces.items()):
            was, turned = h.pieces.pop(pid, None), frm == zone.get(pid)
            held = [a.pieces for a in L.answers if pid in a.pieces] if L is not None else []
            op = ('move', None, pid, held[-1][pid][1], zone.get(pid)) if held else None
            for taken in held:          # (the answers that took it keep where HE has put it now)
                taken[pid] = (taken[pid][0], zone.get(pid))
            if held and ((op[3] == op[4] and was is None and not turned) or (reconcile.tidies(op) and not reconcile.undoes(op))):
                continue                # (put away by him, or lying untouched where he had put it: the answer's still)
            frm, turned = was.frm if was is not None else frm, turned or (was is not None and was.turned)
            h.pieces[pid] = Own(frm, was.since if was is not None else self.voice.stats['steps'], self.voice.req, False, turned)
            if g is not None and (pid in late or not rolling):
                g.elsewhere = (g.elsewhere - {pid}) | ({pid} if pid in late else self.aside(g, [pid], staging))
        self.end(HOME, [p for p, own in h.pieces.items() if zone.get(p) == own.frm and not own.turned])

    def end(self, reason, pieces=None, said=None):
        """THE one place where ownership ends, with its reason (the constants above, or a key
        of words.PUT_BACK).  pieces: whose ownership ends (None: all; {piece: where it came
        from} for the piece of a result of the client, which nobody moved by hand); said: the
        sentence that tells him is said with what happens (its number, or True).  -> those
        pieces.  Every ending is logged.  Where the piece does not lie where the game has it,
        the next pass of the comparison carries it - and says the sentence of a reason of
        words.PUT_BACK (explain)."""
        h = self.his
        gone = {p: h.pieces.pop(p) for p in sorted(h.pieces if pieces is None else pieces) if p in h.pieces}
        if isinstance(pieces, dict):
            gone.update((p, Own(frm, self.voice.stats['steps'], self.voice.req, False))
                        for p, frm in pieces.items() if p not in gone)
        if gone:        # (the game's log: every ending, with its cause)
            self.voice.log('ended', reason=reason, pieces=sorted(gone), said=None if said is True else said)
            if reason in words.PUT_BACK or said:
                h.ended.append(Ended(reason, gone, said))
        return gone

    def settle(self, fate, settled):
        """THE test of whether a piece of his may stay where it lies under the game as it is
        now - asked by every pass of the comparison that looks at the whole board, on that
        pass's own arithmetic (fate: reconcile._kept).  One that lies where the game has a
        piece of its kind is not "moved" any more (PLACED).  One the game has neither there
        nor where he took it from was NEEDED by a step of the game.  And when the game is
        settled, a die or card of his in the play area that no action he still has can use is
        NOT_OWED.  (Anything else he moved stays, whatever the game did around it.)"""
        self.end(PLACED, [p for p, f in fate.items() if f is True])
        self.end(words.NEEDED, [p for p, f in fate.items() if f is None])
        if settled and self.his.pieces:
            staging, zone = self._staging(self.position()), self.link.board.zone
            self.end(words.NOT_OWED, [p for p in self.his.pieces if zone.get(p) in reconcile.DISPLAY and p not in staging])

    def release(self):
        """A step of the game cannot be made on the board as it lies (a client call failed on it, or its
        result does not fit): what he moved and the game has elsewhere is NEEDED - it goes back."""
        pos, board = self.position(), self.link.board
        ops = reconcile.moves(reconcile.difference(
            pos.state, board, self.link.view.card_pid_map, True,
            self.link.view.die_id_map))[0] if board is not None and self.his.pieces else ()
        self.end(words.NEEDED, [op[2] for op in ops if op[2] in self.his.pieces])

    def back(self):
        """He asked for the board to be put back: nothing he moved is his any more - the pass
        that answers him carries it -, and no answer of his is still to be told about."""
        self.end(BACK, said=True)
        self.his.pending = None

    def close(self):
        """THE one place where the window of his taken answers ends: they stand - the game went
        on, dice were asked for what follows, or he answered otherwise."""
        self.window = None

    def abandon(self, taken_back=False):
        """His board answer is given up - he chose by a click or a word, an error stopped it, or
        he took it back by the word or the panel: the window ends, no further step is applied;
        after such a take-back nothing is his any more (it has its own sentence)."""
        self.close()
        self.chain = []
        if self.gest is not None:
            self.gest.taken = None
        if taken_back:
            self.end(TAKEN_BACK)
            self.his.ended, self.his.pending = [], None

    def failed(self, name):
        """An error of the bridge stopped the answer he made on the board (name: the error's
        file): no answer, his board stays, and ONE sentence for this board - an error of
        mine, never a rule of the game."""
        self.abandon()
        g = self.gest
        if g is not None and (g.failed is None or g.failed[0] != g.key):
            g.failed = (g.key, words.board_answer_failed(name))
            self.voice.stats['board_said'] += 1
            self.voice.say_about(g.failed[1], pieces=_about(self.link.board.zone, self.has(g)))

    def explain(self, ops, lines, whole):
        """What a pass of the comparison owes the person, said here (-> the number of the first
        sentence, or None) - made from what end() was given, and from nothing else.  ops: what
        the pass moves; lines: it sends something; whole: it looked at the whole board.  Said: the
        reason of EACH ending whose pieces the pass takes back (the piece itself: no other
        stands for it), one sentence an ending; or, after an answer READ from the board, that
        the pass moves a piece of his side it did not want elsewhere before.  Every ending is
        spent by the pass: it has carried the pieces."""
        h, zone, tag = self.his, self.link.board.zone, layout.side_tag(self.human)
        gone = {p for e in h.ended for p in e.pieces}
        told = [(e, back) for e in h.ended if not e.said for back in [
            [op for op in ops if op[2] in e.pieces and reconcile.undoes(op)]] if back]
        brought = [op for op in ops if op[1] == tag and op[2] not in h.pending and op[2] not in gone
                   and not reconcile.tidies(op)] if whole and h.pending is not None else []
        h.ended, h.pending = [], None if whole else h.pending
        if lines and told:      # (one sentence for EACH ending whose pieces this pass carries back, each by its reason)
            return [_put_back(self.voice, self.link.board, e, back) for e, back in told][0]
        return self.voice.say_about(words.also_moved(), pieces=_about(zone, {}, brought)) if lines and brought else None

    def on_board(self, g, state, identity, one=False):
        """ONE way in for every answer: the answer he NAMED (a click, a word) in the decision of
        gesture g as the board shows it - its steps made with the pieces he put there for it
        (menu.realise), so the engine takes his pieces and not its own, also when only a part
        of the move is there.  one: the pieces of the NAMED step are laid out first - each
        piece of his that ALONE is (a part of) it; his others lie beside his explicit answer
        (committed).  A board that is then a CHAIN beginning with the named step lies at that
        chain's END: his answer takes its own step, and the rest is the next decision's to read
        (_ahead).  None: nothing of it is on the board, it has no tables, or an error stopped an
        answer of this board - the verdict a later board is read against is kept all the same."""
        board, mine = self.link.board, set(self.has(g)) if one else set()
        opt = next((a for a in g.menu.options if panel.identity(a) == identity), None) \
            if g.opening is not None and g.menu is not None and (
                g.failed is None or g.failed[0] != mn.board_key(board)) else None
        named = lambda off: match.named(g.opening.base, mn.reading(board, g.opening, g.elsewhere | off),
                                        g.menu, opt, g.opening.opened)
        v = (opt is not None and (mine or not one) and named({p for p in mine if not named(mine - {p})})) or None
        g.taken = v or (opt is not None and match.given(g.opening.base, g.menu, opt)) or None    # (what a later board is read against)
        return [(a,) + tuple(step) for a, step in zip(v.chain, mn.realise(state, v.chain, v))] if v else None

    def refuse(self, pid):
        """The game refuses the card the person answered with (it is no combat card here): ONE
        sentence - and one that LIES on the table (whoever laid it there) is nobody's: the
        comparison carries it where the game has it, into his hand."""
        laid = self.link.board.zone.get(pid) in reconcile.DISPLAY
        self.end(REFUSED, {pid: None} if laid else [pid], said=self.voice.say_about(words.card_not_playable(laid), pieces=[
            {'pid': pid, 'lay': None, 'was': None, 'to': None, 'by': 'person', 'seen': None}]))

    def stands(self, g):
        """The identity of the option that leaves the board of gesture g's decision exactly as it
        stands (match.as_it_stands), or None - also while the board shows a move the decision
        reads (WHOLE BOARD OR NOTHING: that is watch's to judge)."""
        keep = match.as_it_stands(g.menu) if g is not None and g.menu is not None and not self.moved(g) else None
        return panel.identity(keep) if keep is not None else None

    def moved(self, g, only=None):
        """What the board shows that the decision of gesture g reads as moved (boardmodel's delta;
        empty: nothing - only then is an answer given FOR him: the one legal answer, the board
        as it stands).  only: of what he moved, these pieces alone."""
        now = mn.reading(self.link.board, g.opening, g.elsewhere | (set(self.has(g)) - set(only or self.has(g))))
        return bm.delta(g.opening.base, bm.arrived(now, g.opening.opened, g.opening.base))

    # ------------------------------------------------ what a move would lead to
    def probe_apply(self, g, root, actions):
        """ctx.apply of menu.py: the state after `actions` (his own steps in a row from the opening
        of gesture g); None when a step needs a roll or draw, the other seat, or is illegal."""
        import game_logic as gl
        key = tuple(id(a) for a in actions)
        if root is not g.root or not actions:
            return None
        if key not in g.memo:
            post = None
            if len(actions) == 1:
                start = (g.root, g.spec, g.pos)
            else:
                nxt = self.probe_next(g, self.probe_apply(g, root, actions[:-1]))
                start = None if nxt is None or nxt[0] is not g.side else (nxt[2], nxt[4], nxt[5])
            if start is not None:
                state, spec, pos = start
                act = actions[-1]
                try:
                    post = self.apply(state, spec, act)
                    change = dict(spec['route'](post, act)) if spec['route'] is not None else {}
                    if spec['phase'] is not None:
                        change['phase'] = spec['phase']
                    result = gl.check_all_victory_conditions(post)
                    if gl.should_game_end(result, end_of_turn=False):
                        change.update(over=dict(result), phase='over')
                    g.at_pos[id(post)] = pos._replace(state=post, **change)
                except (NeedClientCall, Exception):
                    post = None
            g.memo[key] = (post, list(actions))         # the actions stay alive: their ids are the key
        return g.memo[key][0]

    def probe_next(self, g, post):
        """(side, kind, state, menu, spec, position) of the decision the step machine opens after
        `post` with no roll, draw or other automatic step in between; None otherwise."""
        pos = g.at_pos.get(id(post)) if post is not None else None
        if pos is None:
            return None
        if id(post) not in g.next:
            g.next[id(post)] = self.opens(pos) if pos.over is None else None
        return g.next[id(post)]

    def probe_following(self, g, post, side):
        """ctx.following of menu.py: (kind, options) of the same person's next
        decision after `post`, or None."""
        nxt = self.probe_next(g, post)
        return (nxt[1], nxt[3]) if nxt is not None and nxt[0] is side else None

    # ------------------------------------------------ the open decision on the board
    def gesture_lines(self, o, act, after=()):
        """What the person does on the board to answer the open decision `o` with `act`: the
        client lines a click on it would send, as his own actions ('!HUMAN <call>' = a client
        command of his, such as playing a card); [] when the board cannot show that answer.
        Nothing is sent or used.  after: his answers before it in the SAME gesture, already on
        the board (`act` is then an option of the decision that follows them)."""
        g = self.gest
        if o is None or g is None or g.id != o.id or g.menu is None:
            return []
        want = panel.identity(act)
        try:
            menu, pre, kind, chain = g.menu, self.position().state, o.kind, []
            for done in after:
                chain.append(next(a for a in menu.options if panel.identity(a) == panel.identity(done)))
                pre = self.probe_apply(g, g.root, chain)
                nxt, menu = self.probe_next(g, pre), menu.expand(chain[-1])
                if nxt is None or nxt[0] is not o.side or not hasattr(menu, 'options'):
                    return []
                kind = nxt[1]
            opt = next((a for a in menu.options if panel.identity(a) == want), None)
            plan = menu.plan(opt) if opt is not None else None
            if not plan or all(ch.opt for ch in plan):
                return []
            post = self.probe_apply(g, g.root, chain + [opt])
            view = self.link.view.probe(pre if chain else None)
            if chain:       # (another state than the game's: its own binding)
                view.die_id_map = reconcile.bind_dice(pre, self.link.board)
            out = emission.render_step(view, self.link.nick(o.side), o.side, kind, opt, post, pre)
        except Exception:
            return []
        if out.hold:
            return []
        lines = [line for line in out.lines if em.classify_line(em.strip_tail(line))[0]
                 not in (None, 'chat', 'combat_card_pick')]
        return lines + ['!HUMAN ' + c[6:].strip() for c in out.calls       # his own draw is his
                        if c.startswith('!CALL draw_')
                        or not c.startswith(RANDOM_CALLS + ('!CALL reveal_chosen',))]

    def gesture(self, o, spec):
        """The open decision `o` as the matcher sees it, built once: the tables of its opening
        and its Menu (menu.py).  What is his when it opens: the pieces of a decision owed by
        the state of the game stay where they lie in BASE; what he moved before it was asked
        and it READS (aside) is read as MOVED like any move."""
        g, ahead = self.gest, _ahead(self.window, self.gest)
        if g is None or g.id != o.id:
            seen = g if g is not None and g.id is None and g.root is o.state else None      # (taken back by hand: again)
            g = self.gest = Gesture(o.id, o.state, o.side, spec, self.position(), self.now(), options=o.menu)
            owed = functools.partial(mn.owned, phase=g.pos.phase, person=self.human)
            g.elsewhere = aside = self.aside(g, self.his.pieces, self._staging(g.pos))
            g.opening = mn.tables(o.state, self.link.board, self.link.view.card_pid_map, owed,
                                  {p: own.frm for p, own in self.his.pieces.items() if p not in aside},
                                  self.link.view.die_id_map or None, [p for p, own in self.his.pieces.items() if own.turned])
            if ahead:       # (all that separates this board from the game he did himself: read as moved, not as lagging)
                g.opening = g.opening._replace(opened=g.opening.base)
            ctx = mn.Ctx(self.link.view, self.link.nick(o.side), functools.partial(self.probe_apply, g),
                         functools.partial(self.probe_following, g), owed)
            g.menu = mn.build(o.state, o.side, o.kind, o.menu, g.opening, ctx)
            g.elsewhere = aside | owed(o.state, self.link.board, about=o.kind)
            if seen is not None and seen.elsewhere == g.elsewhere:      # (the verdict that took his answers back
                g.key, g.v = seen.key, seen.v                           # read this very board, by this decision)
        return g

    def read(self, g):
        """THE verdict of an open decision (g: its gesture, its Menu built) on the board as it
        lies, judged ONCE a board (watch counts the polls it stands): read whole; and, where that
        is no answer, without the pieces it does not read (_means: None - but one at a place an
        answer of his chain changed, _spots: again reads that board) and without what lay BESIDE
        his answer by a word or a click and is not the beginning of an answer here.
        None: this board could not be judged (noted once)."""
        key = mn.board_key(self.link.board)
        if key != g.key:
            g.key, g.equal, g.at, g.v = key, 0, self.now(), None
            for final in (False, True):
                board = mn.reading(self.link.board, g.opening, g.elsewhere)
                order = match.untidied(g.opening.base, board, g.opening.opened)
                g.moved = g.moved if order is None else order
                v = match.judge(g.opening.base, board, g.menu, opened=g.opening.opened, tidied_last=g.moved)
                chain = _spots(self.window, self.link.board, self.has(g)) if v.kind == match.MISMATCH and not final else ()
                aside = () if v.kind != match.MISMATCH or final else frozenset(
                    p for p, x in _means(g, self.link.board, self.has(g)).items()
                    if (x is None and p not in chain) or (x is not True and self.his.pieces[p].beside))
                if not aside:
                    break
                g.elsewhere |= aside
            g.v = v
            self.voice.log('judge', decision=g.id, verdict=v.kind, chain=len(v.chain), extendable=v.extendable,
                           unfinished=v.unfinished, toward=v.toward)    # (why a commit waits the quiet time: match.ready)
            for a, what, error in g.menu.faults[g.noted:]:      # an option it cannot read
                self.voice.note('"%s" cannot be read from the board: %s failed with %r'
                                % (g.menu.name(a), what, error))
            g.noted = len(g.menu.faults)
        return g.v

    def watch(self, o, spec, confirm=False):
        """The board as the answer to the open decision `o`: the realised steps of a COMMIT
        verdict that is ready (rulings 1 and 7), else None.  WHOLE BOARD OR NOTHING: the
        matcher gives a COMMIT only when all the decision reads is an answer, or a chain of
        answers of his decisions that follow at once.  A board that fits no move, or several,
        is told ONCE for the pieces of his it reads as they lie, after the quiet time."""
        now, h = self.now(), self.his
        g = self.gesture(o, spec)
        if g.menu is None:              # it could not be built (noted once): click or word
            return None
        v, key, zone = self.read(g), g.key, self.link.board.zone
        g.equal += 1
        if v is None:                   # this board could not be judged (noted once)
            return None
        quiet = now - g.at
        beside = [p for p, own in h.pieces.items() if own.beside]
        if beside:      # what lay beside his answer by a word or a click: this decision follows it at once -
            # each piece stays as the beginning of an answer to it, and goes back when it is none (not read)
            self.end(words.BESIDE, [p for p in beside if p in g.elsewhere])
            h.pieces.update((p, h.pieces[p]._replace(beside=False)) for p in beside if p in h.pieces)
        if g.failed is not None and g.failed[0] == key:     # (an error stopped these very pieces: said, not tried again)
            if confirm:                 # ... but a closing word is never met with silence
                self.voice.say_about(g.failed[1], pieces=_about(zone, self.has(g)))
            return None
        if v.kind == 'commit' and (confirm or match.ready(v, g.equal, quiet)):
            g.taken = v
            return [(opt,) + tuple(step) for opt, step in zip(v.chain, mn.realise(o.state, v.chain, v))]
        # a closing word is answered at once, also on a board already told
        lie = frozenset((p, zone.get(p)) for p in self.has(g))
        if v.kind in ('undecided', 'mismatch') and (confirm or (quiet >= match.Q_SECONDS and not lie <= g.told)):
            g.told |= lie
            self.voice.stats['board_said'] += 1
            names = _choice_names(g, v)
            faulty = list(dict.fromkeys(g.menu.name(a) for a, _what, _error in g.menu.faults))
            text = words.board_choices(names) if v.kind == 'undecided' else words.board_unclear(v.sentence, faulty)
            means = _means(g, self.link.board, self.has(g)) if v.kind == 'mismatch' else {}
            means = {p: x for p, x in means.items() if p not in g.elsewhere}
            odd = [p for p, x in means.items() if x is False]
            rival = [p for p in means if not odd and not bm.stages_only(self.moved(g, [p]), True)]
            # (the words are of HIS pieces only: THE pieces of his that fit no move, as they lie - or, with none such,
            # ALL that alone begin a move of their own: more moves than are owed, none called illegal (a die or card
            # only laid out names none); and never a card of his hand by name in chat: by what lies moved)
            if odd or len(rival) > 1 or not words.is_plain(text) or any(words.chat_card_name(
                    words.card_title(c)) in words.chat_card_name(text) for c in o.state.hands[o.side].hand):
                text = words.board_choices(()) if v.kind == 'undecided' else words.board_unclear(
                    None, faulty, bm.moved_words(self.moved(g, odd or rival[1:] and rival)), len(rival) > 1)
            self.voice.say_about(text, pieces=_about(zone, self.has(g)))
        return None

    def named(self, o, word, v):
        """The board is undecided between choices and a typed word names exactly ONE of
        them: the steps of that choice, committed as a board answer (his pieces stay where
        he put them: the figure on Mordor Track step 0 with the gate typed); else []."""
        g = self.gest
        if getattr(v, 'kind', None) != 'undecided' or not v.choices or g is None or g.menu is None:
            return []
        i = words.names_choice(word, _choice_names(g, v))
        if i is None:
            return []
        chain = list(v.choices[i])
        self.abandon()              # a word chose: the board is not watched on
        return self.on_board(g, o.state, panel.identity(chain[0])) or [(opt, opt, None) for opt in chain]

    # ------------------------------------------------ his taken answers: their window
    def again(self, o):
        """WHICH READING, AND WHAT A TAKE-BACK IS - the one place.  His taken answers, while the
        game has not gone on past them (the window), against the board as it lies now (asked
        once a board); o: his decision that is open, or None.  The board is read from the LATEST
        point of his chain that can read it.  First the open decision, at the game as it stands
        (read) - with none open, the decision of his last answer, at the game before it.  When
        that is a mismatch AND a piece of his lies at, or was taken from, a place an answer of
        the chain changed (_spots; a piece elsewhere never sends the reading back), the
        board is read from the game as it stood BEFORE the last answer that changed such a place
        (the answers after it left that place as the first reader found it), by the decision
        that was open then (its gesture, its pieces where HE put them: match.again), and so on
        back to the first.  The first point that reads it - no mismatch - is the reading: when
        that is another answer than it gave, the answers from there on are taken back -> (the
        position to restore, the journal entries to drop), and that decision opens again with
        the verdict just given.  None: his answers stand - also when no point reads the board:
        nothing is taken, nothing put back, and the open decision's own verdict on each piece
        stands (J1: watch tells him once; one it does not read waits).  Pieces are compared as the
        GAME sees them, by place and kind: no record of a piece decides it.  What the answers
        took is his again, read as moved by the decision that opens again; what the bridge did
        for them the comparison takes back."""
        L, board, h, g = self.window, self.link.board, self.his, self.gest
        g = g if o is not None and g is not None and g.id == o.id and g.menu is not None else None
        if L is None or board is None or self.chain or L.judged == mn.board_key(board) or (o is not None and g is None):
            return None
        if o is None and (self.opens(self.position()) or (None,))[0] is self.human:
            return None             # (a decision of his follows at once: it reads first, when it is open)
        L.judged, k, n = mn.board_key(board), None, len(L.answers)
        try:
            v = self.read(g) if g is not None else None
            read = self.has(g) if g is not None else {p: h.pieces[p] for p in set(h.pieces) - self._staging(self.position())}
            mine = _spots(L, board, read) if g is None or (v is not None and v.kind == match.MISMATCH) else {}
            for i in [n - 1] * (g is None and bool(read)) + [i for i in range(max(mine.values(), default=-1), -1, -1)
                                                            if g is not None or i < n - 1]:
                a = L.answers[i].gesture
                if a.taken is None:
                    break
                # (THE WHOLE BOARD: all the first reader reads - a stray it reads spoils every point, J1 - without what
                # it set aside, and with the pieces of his answers that the bridge moved on - the die of the chain,
                # put away for him - where HE had put them: what the bridge did is no move of his)
                put = {p: x[1] for q in L.answers for p, x in q.pieces.items() if p in board.zone and p not in h.pieces}
                off = a.elsewhere | (set(h.pieces) - set(read))
                v = match.again(a.opening.base, mn.reading(reconcile.Board(dict(board.zone, **put), board.face, board.mark),
                                                           a.opening, off), a.menu, a.taken, opened=a.opening.opened)
                if v.kind != match.MISMATCH:
                    k = i if v.kind != match.SAME else None
                    break
            if k is None and g is not None and mine:
                # NO POINT READS IT AS ANOTHER ANSWER: his pieces are the open decision's alone to judge (J0) - one
                # it does not read waits in silence, as a piece elsewhere does
                unread = {p for p, x in _means(g, board, read).items() if x is None and p in mine}
                if unread:
                    g.key, g.elsewhere = None, g.elsewhere | unread
        except Exception as e:      # (it cannot be judged: his answers stand)
            self.voice.stats['match_errors'] += 1
            self.voice.note('the board could not be judged against his taken answers: %r' % (e,))
        if k is None:
            return None
        gone, L.answers, L.searched, was = L.answers[k:], L.answers[:k], None, {}
        for a in reversed(gone):        # (a piece several of them took: as the first of them found it)
            was.update(a.pieces)
        # (a piece of these answers that the bridge moved on since - his die, put away for him - goes
        # back where HE had put it, by the one mover; it is his again like the rest)
        self.end(TAKEN_BACK, {p: at for p, (own, at) in was.items()
                              if p not in h.pieces and board.zone.get(p) not in (None, at)}, said=True)
        h.pieces.update((p, own) for p, (own, at) in was.items())
        base, now = gone[0].gesture.opening.base.p, mn.table(board).p   # (home: as the game before these answers
        self.end(HOME, [p for p in was if p not in now or now[p] == base.get(p)])   # has it - its place, face and side)
        h.pending = None
        self.voice.log('again', decision=gone[0].gesture.id, answers=len(gone))
        # (the decision opens anew and is read anew; the verdict just given for this board is kept for it: gesture)
        v.kind = match.COMMIT if v.chain else v.kind
        self.window, self.gest = L if L.answers else None, Gesture(root=gone[0].gesture.root, key=L.judged, v=v)
        self.gest.elsewhere = frozenset(off)
        return gone[0].before, sum(a.entries for a in gone)

    def committed(self, who, source, random, before, o):
        """A step was committed.  who: 'human', or who else decided it; source: how his answer
        was given (answer.Answer.source); random: dice or a draw went into it; before: the
        position it was taken from; o: his decision it answers.  His answer takes the pieces
        it is made with (_took).  Of the other pieces the answered decision READ, none stays
        where the game does not have it but as the beginning of his answer to the decision
        that follows at once: after an answer read from the board there is none (watch); after
        an answer he gave by a word or a click they lie BESIDE it - they go back now, or, when
        a decision of his follows at once, when that decision does not read them so (watch).
        A step of his with no dice or draw belongs to the window - read from the board, clicked
        or typed: EVERY answer of his chain; any other step closes it."""
        g, pos, board, h = self.gest, self.position(), self.link.board, self.his
        self.end(words.BESIDE, [p for p, own in h.pieces.items() if own.beside])    # (nothing of his read them since)
        if who == 'human' and g is not None and board is not None:
            first = o is not None and g.id == o.id      # (not a further step of the chain his gesture gave)
            read = self.has(g) if first else {}
            took, used = _took(g, read, board)
            early = {p for p, x in _means(g, board, read).items() if x is None} if read and source != 'board' else ()
            if not (first or source == 'board'):
                self.close()    # (an answer the open gesture did not read: nothing can be read against it)
            else:
                self.window = L = self.window or Window()
                if not L.answers or L.answers[-1].gesture is not g:
                    L.answers.append(Taken(g, before))
                L.answers[-1].entries += 1
                L.answers[-1].pieces.update((p, (own, board.zone.get(p))) for p, own in read.items()
                                            if source == 'board' or p in took or p in used)
                L.searched = None if source == 'board' else 0       # (what follows a click or a word is not held back)
            if source == 'board':
                self.voice.stats['board_answers'] += 1
                h.pending = g.opening.pending if first else h.pending
            self.end(ANSWERED, took)
            rest = [p for p in read if p in h.pieces and p not in used and p not in early] if source != 'board' else []
            nxt = self.opens(pos) if rest else None
            if nxt is not None and nxt[0] is self.human:
                h.pieces.update((p, h.pieces[p]._replace(beside=True)) for p in rest)
            else:
                self.end(words.BESIDE, rest)
        if who in ('human', 'forced') and source != 'board':
            h.pending = None    # (he chose by a click or a word, or it was given for him: Answers._made)
        if who != 'human' or random:
            self.close()

    def settling(self):
        """Ruling 2: his board answer may still change.  What follows it and is not his - the AI's step, the
        action roll, the turn end - waits until the board has been seen once more and still shows his answer
        (what follows a click or a word does not wait: committed; nor does a probe of the loop: the server)."""
        L, polls = self.window, self.voice.stats['polls']
        if L is None:
            return False
        L.searched = polls if L.searched is None else L.searched
        return L.searched == polls
