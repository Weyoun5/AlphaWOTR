"""The loop: one handler per client poll (DESIGN principles 1, 3, 5, 6, 8).

    python -u -m core.server --client-dir <dir> --side AUTO|FP|SA [--sims N]
                             [--net-tag T] [--seed N]          (cwd = wotr_bridge)
    --side names the AI's seat (the person has the other); AUTO = AI vs AI.

The Server holds the game and the order of one poll; every other state has its owner:
    pos       the game position: the engine state and the turn cursor the
              engine's own episode loop keeps (phase, side to act, turn owner)
    journal   the committed steps of this turn (the position before each: for a take-back)
    rand      the client-randomness funnel with its one open request
    link      the client as seen this poll: feed, board, nicks, typed words (link.py);
              inbox: its results no step has used yet
    answers   the decision the person is asked now and his answer to it (answer.py)
    boards    his answer on the board: the gesture, and the window in which a
              committed answer may still grow or change (boardanswer.py)
    compare   the comparison of the board with the game (compare.py)
    voice     everything said, noted, logged and published (voice.py)
    asked / shown / used_random   small state of the open step

Per poll: ingest feed events -> read the board at that feed position -> an
open step is retried; otherwise the board is verified against the engine and
what is missing is re-rendered (reconcile); otherwise the engine advances one
step.  A step is applied on a clone and adopted whole.  Whoever decides - the
search or the person's click - the step goes through the same _decide: shown
before its dice are asked, rendered by the same renderer, verified the same way.

Answers on the board: while a decision of the person is open the comparison is
held and every changed board is judged by the matcher (match.judge on the Menu of
menu.py).  A complete answer, stable for K polls (a strict part of a bigger one:
quiet for Q seconds), goes through the same _decide as a click, step by step when
it answers several decisions.  Until the AI's next step is committed a change of
the board takes the answer back (boardanswer.again): the position of the decision's
opening is restored and the decision opens again on the board as it lies.  What on
the board is his meanwhile, piece by piece, is boardanswer's to say; the loop only
tells it what happened.
"""
import contextlib
import os
import random
import time
from collections import Counter, namedtuple

from . import answer, boardanswer, client_protocol, compare, emission, layout, link, live_parser, panel
from . import reconcile, settings, strength, turn_search, voice, words
from . import emitter as em
from . import randomness as rnd
from .feed_reader import FeedRestart
from .transport import FileBridge

# the game position; `state` is never mutated after it is adopted
Position = namedtuple('Position', 'state phase active owner empty over')
# one committed step: who decided it, and whether client dice or draws went into it
Entry = namedtuple('Entry', 'label before who random')

SENT, FAILED, TOLD = 'sent', 'failed', 'told'   # an open client call (Server.asked[1]; see _ask)
SILENT_S = 6.0              # seconds a call the client carried out may give no result before he is told
STEPS_PER_POLL = 12         # silent engine steps one poll may chain
ROLL_ZONE = {'SHADOW': em.SH_DICE_POOL, 'FREE_PEOPLES': em.FP_DICE_POOL}
LETTER_FACE = {v: k for k, v in em._FACE_LETTER.items()}
CARD_PHASES = ('attacker_combat_card', 'defender_combat_card')
# steps whose picture needs the result: never shown before their dice
SHOWN_AFTER = ('hunt_sacrifice_random', 'combat_retreat', 'casualty_allocate', 'siege_press',
               'casualty_preserve_units', 'casualty_preserve_elites')


def _failed(asked):
    """The open client call (Server.asked) FAILED - the client said so, its result fits nothing, or
    an error of mine may have lost it: it is asked again (one he was TOLD of waits for his word)."""
    if asked is not None and asked[1] != TOLD:
        asked[1] = FAILED


def card_box(person, kind, act):
    """The box that governs what a step BRINGS ABOUT (settings): 'effects' for a step of the person
    (person: his side, None when the step is not his) that plays an event card or answers a question
    A CARD OF HIS OWN asked, and None for every other step (nobody's choice: the bridge does it).  A
    card question names who asked it, and a card's id begins with its owner's side (cards.py).  His own
    card, or nobody (a move his card grants: only the card's player is asked) - a card of his own.  The
    other side's card (real client, REAL_R18 F2: the hits of the AI's Dreadful Spells were left to him
    as "your card") or one of the engine's own askers ('__companion_moves__', ...: the follow-up of a
    die's action, of an ability, of a rule; F1) - not his card: the bridge carries it."""
    if person is None:
        return None
    asked_by = str((act.params or {}).get('card_id') or '')
    own = not asked_by or asked_by.startswith(person.name + '_')
    return 'effects' if act.action_type == 'play_event' or (words.family(kind) == 'card' and own) else None


class _Opens(Exception):
    """A probe reached a decision: (side, kind, state, menu, spec)."""


class _Stops(Exception):
    """A probe reached a step that is not a decision (a roll, a draw, a commit)."""


def fresh_state(seed=None):
    import board
    import game_logic
    import game_state as gs
    if seed is not None:
        random.seed(seed)
    state = gs.setup_initial_game(board.load_board())
    game_logic.initialize_decks(state)
    return state


def answered(journal, phase, turn):
    """Did the person answer the decision of `phase` in this turn?  (His word or his move
    closed that part of the turn.)"""
    return any(e.who in ('human', 'forced') and e.before.phase == phase and e.before.state.turn == turn
               for e in journal)


def last_answer(journal):
    """How many steps at the journal's end are the last answer: the answers given for the
    person after it, and the step before them (0: the journal holds none)."""
    n = next((i for i, e in enumerate(reversed(journal)) if e.who != 'forced'), None)
    return 0 if n is None else n + 1


def unseen_slots(state, viewer):
    """THE list of the places of `state` whose card `viewer` cannot see, as (holder, key): the
    other side's decks, hand and face-down discards, and its combat card while it lies committed
    and unrevealed (not one paid with the table card: everyone saw it).  That card is ONE card
    in two places - the battle record and its copy on the face-up pile, where the engine hides
    the entry that EQUALS the record: the place of the copy is the second value (or None)."""
    import ai
    other = next(s for s in state.hands if s is not viewer)
    tag, info = 'fp' if other.name == 'FREE_PEOPLES' else 'sh', state.hands[other]
    slots = [(cards, i) for cards in (getattr(state, tag + '_char_deck'), getattr(
        state, tag + '_strat_deck'), info.hand, info.discarded_face_down) for i in range(len(cards))]
    battle = state.pending_combat if isinstance(state.pending_combat, dict) else {}
    commit = ai._hidden_commit_card(state, viewer)
    if commit is None or battle.get('atk_cc_from_table') == commit:
        return slots, None
    copy_at = next(((pile, i) for pile in state.discard_face_up.values()
                    for i in reversed(range(len(pile))) if pile[i] == commit), None)
    return slots + [(battle, 'atk_combat_card')], copy_at


def assert_blind(state, handed, viewer):
    """The standing check on every hand-off to the search (it raises, and names no card):
    the unseen places of `handed` hold exactly the unseen cards of the game, each once, and
    none of those cards stands where everybody sees it (a face-up pile, the battle record)."""
    def cards(st):
        slots, copy_at = unseen_slots(st, viewer)
        seen = Counter(c for pile in st.discard_face_up.values() for c in pile)
        seen.update(v for k, v in (st.pending_combat or {}).items()
                    if k != 'atk_combat_card' and isinstance(v, str))
        if copy_at is not None:
            seen[copy_at[0][copy_at[1]]] -= 1
        return Counter(str(holder[key]) for holder, key in slots), +seen
    (truth, public), (dealt, shown) = cards(state), cards(handed)
    leaked = sum(((shown - public) & truth).values())
    if leaked or dealt != truth:
        raise AssertionError('the state for the search is not blind: %d unseen card(s) shown, '
                             'the unseen set %s' % (leaked, 'kept' if dealt == truth else 'changed'))


def redact(state, viewer, seed):
    """A clone of `state` in which every card `viewer` cannot see (unseen_slots) is dealt
    again at random, deck by deck.  The result depends only on WHICH cards are unseen and
    on `seed`, never on where they are: the search gets no hint of the hand."""
    import ai
    import game_logic as gl
    clone = ai.clone_state(state)
    rng = random.Random(seed)
    slots, copy_at = unseen_slots(clone, viewer)
    for deck in ('character', 'strategy'):
        places = [(holder, key) for holder, key in slots if gl.card_deck_of(holder[key]) == deck]
        pool = sorted(str(holder[key]) for holder, key in places)
        rng.shuffle(pool)
        for (holder, key), card in zip(places, pool):
            holder[key] = card
    if copy_at is not None:
        copy_at[0][copy_at[1]] = clone.pending_combat['atk_combat_card']
    return clone


class Server(object):
    def __init__(self, client_dir, side='AUTO', sims=300, net_tag='auto', seed=None,
                 load_nets=True):
        import game_state as gs
        self.gs = gs
        FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
        self.client_dir = client_dir
        self.sims = int(sims)
        self.net_tag = net_tag
        self.ai_sides = {'AUTO': (FP, SH), 'WATCH': (FP, SH), 'FP': (FP,)}.get(str(side).upper(), (SH,))
        self.human = None if len(self.ai_sides) == 2 else (SH if self.ai_sides[0] is FP else FP)
        self.voice = voice.Voice(client_dir, self.ai_sides, self.human, lambda: self.pos,
                                 lambda side: self.link.nick(side), lambda: self.answers.open)
        self.rand = rnd.ClientRandomness(human_side=getattr(self.human, 'name', None), seed=seed,
                                         prefers=settings.Settings(client_dir))
        self.rand.install()
        self.inbox = link.Inbox()
        self.link = link.ClientLink(client_dir, self.ai_sides, self.human, self.rand, self.voice,
                                    self.inbox, lambda: self.pos.state, lambda: self.now())
        self.pos = Position(fresh_state(seed), 'draws', FP, None, frozenset(), None)
        self.journal = []
        self.compare = compare.Comparison(
            self.voice, self.link, self.human, lambda: self.pos, lambda: self.owned(),
            lambda *found: self.boards.settle(*found), lambda *found: self.boards.explain(*found),
            lambda: self.on_turn_end is not None and self.on_turn_end(self))
        self.asked = None           # the open client call: [its identity, SENT / FAILED / TOLD, receipt read at,
        #                             the person still owes a part of it that he makes with his own hand]
        self.shown = None           # the open step was shown before its dice: {'reveal': bool}
        self.used_random = False    # the open step asked the client for dice or cards
        self.probing = False        # the matcher asks what a move would open: no step is taken
        self.boards = boardanswer.BoardAnswers(self.voice, self.link, self.human, lambda: self.pos,
                                               lambda: self.now(), self._probe_step, self._probe_open)
        self.answers = answer.Answers(
            self.voice, self.link, self.inbox, self.rand, self.boards, self.human, lambda: self.pos,
            lambda built, o: self.on_decision is not None and self.on_decision(built, o))
        self.now = time.time        # the clock of the quiet time (a test may set its own)
        self.rng = random.Random(seed)
        self.on_turn_end = self.on_decision = self.on_search = None     # test hooks
        self.search = None
        if load_nets:
            from .own_model import load_own_net, OwnModelSearch, SeatSearches, seat_seed
            journal = os.path.join(client_dir, 'own_search_calls.jsonl')
            searches = {s: OwnModelSearch(s, load_own_net(net_tag, s), seed=seat_seed(seed, s),
                                          journal=journal) for s in self.ai_sides}
            self.search = SeatSearches(searches) if self.human is None else searches[self.ai_sides[0]]
            print('  [search] %s; %d simulations' % (
                'AI vs AI' if self.human is None else 'the AI plays the %s'
                % words.side_name(self.ai_sides[0]), self.sims), flush=True)

    # ------------------------------------------------------------ the poll
    def handle(self, request):
        """One transport cycle -> (lines, status words).  Never raises: an exception is logged with
        the engine state saved, and the next poll retries from the same position (principle 8)."""
        self.voice.stats['polls'] += 1
        try:
            lines = self._poll(request)
            self.voice.errors = 0
        except FeedRestart as e:
            if self.link.refusal is None:       # (said, noted and ended once: every later poll is silent)
                self.voice.note('feed restart: %s' % e)
                self.boards.end(boardanswer.RESTARTED)
            lines = self.link.refuse(words.another_game())
        except Exception as e:
            self.voice.stats['exceptions'] += 1
            self.voice.errors += 1
            name = self.voice.save_error(e, self.position)
            # the same step is tried again once, with the dice and cards the client made for
            # it and its request standing.  A choice of the person that failed, or an error
            # that comes again, gives the step up: another answer, or other results, follow
            if self.voice.errors > 1 or self.answers.click is not None:
                self.rand.abort_step('exception')
                self.asked = self.shown = None
            elif self.rand.step_open:
                self.rand.begin_attempt()       # (nothing of the failed attempt counts as used)
                _failed(self.asked)             # (... and a result that came with this poll may be lost)
            self.voice.how = 'The AI hit an internal error and is trying again.'
            lines = [self.voice.chat(self.ai_sides[0], words.internal_error(name))] \
                if self.voice.errors == 3 else []
            if self.answers.click is not None:      # the person's choice failed: he may choose again
                self.voice.result(self.answers.click, 'refused', words.not_carried_out())
                self.voice.how = words.not_carried_out(status=True)
                self.answers.drop(again=True)
                self.link.unpick()
        lines, self.voice.outbox = self.link.replied(list(lines) + self.voice.outbox), []
        self.voice.publish(self.facts)
        return lines, self.voice.how

    def _read(self, request):
        """The request as the client sent it: its feed lines (their events), its failures, its
        board.  -> (what link.read gave, None) - or (None, the reply) when nothing is played on
        it: another seat's request, a client that shows a hand it must not, no board yet."""
        side = str(getattr(request, 'side', None) or '')
        want = ('AUTO',) if self.human is None else (
            'LIVE_' + self.ai_sides[0].name, self.ai_sides[0].name)
        if side and side not in want:
            self.voice.how = ('This AI was started %s. Use the matching entry of the AI menu, or '
                              'start the AI again.' % ('for AI vs AI' if self.human is None else 'to play the %s'
                                                       % words.side_name(self.ai_sides[0])))
            return None, []
        poll = self.link.read(request)
        self.voice.req = poll.header.get('req') if poll.header is not None else None
        why = self.link.unblinded(poll.new)
        if why:
            return None, self.link.refuse(why)
        for i, line in enumerate(poll.new, len(self.link.feed.lines) - len(poll.new)):
            self.link.event(live_parser.parse(line), i)
        # the confirm of the client's own random discard is answered by the
        # client itself; its watchdog line is not a failure
        bad = [l for l in client_protocol.client_error_lines(poll.new)
               if 'Confirm Random Discard' not in l]
        if bad:
            self.voice.stats['client_errors'] += len(bad)
            self.voice.note('client: %s' % bad[0][:200])
            if self.asked is not None:
                # the call failed: it is asked again when the board has been made the game's -
                # what he moved and the call needs (his dice still in the Hunt Box at the action
                # roll) goes back first, with that reason
                _failed(self.asked)
                self.compare.repair = self.shown is None and self.human is not None
                self.boards.release()
        # (the board first: a card drawn in this request is ONE result by what its board shows - link.bank)
        read = self.link.read_board(poll.header, poll.want)
        self.link.raw_events(poll.raw, poll.want)
        return (poll, None) if read else (None, self.link.no_board(poll.header, poll.want))

    def _poll(self, request):
        poll, lines = self._read(request)
        if poll is None:
            return lines
        # what he moved since the last board is his from now (the dice are bound with it: link.bind) - read by
        # his decision that is open.  THE ORDER of his answer and his moves is a fact: what the client dates
        # after his click, and all he moves while the dice of a clicked or typed answer are rolled, the answered
        # decision never read (late); during the roll of an answer READ FROM THE BOARD he still finishes it
        click = self.answers.click
        rolling = self.asked is not None and click is not None
        moved, late, line = self.link.track(poll.raw, poll.want, click.at if click is not None else None)
        self.link.bind(self.pos.state, dict(moved, **self.owned().pieces))
        self.boards.took(moved, self.answers.open, rolling, moved if rolling and click.source != 'board' else late)
        # what the funnel handed back, and what no step used and has left the bank: said once -
        # and the piece of such a result is nobody's: the comparison carries it back with this reply
        self._hand_back()
        # (a die staged and the word typed in one poll: both seen.  THE ORDER of a word that ANSWERS - it gave his
        # answer, or closed his decision - and his moves: what the client's lines hold after it, that decision never read)
        for body, unseen, at in self.link.heard():
            had = (self.answers.click, self.answers.closing)
            self._chat_word(body, unseen)
            after = {p: frm for p, frm in moved.items() if at is not None and line.get(p, -1) > at}
            if after and (self.answers.click, self.answers.closing) != had:
                self.boards.took(after, self.answers.open, rolling, after)
        if self.pos.over is not None:
            self.boards.end(boardanswer.OVER)
            if not self.voice.over_said:         # the winning step's consequences first
                lines = self.compare.verify()
                if lines:
                    return lines
            return self.voice.over_lines()
        back = self.boards.again(self.answers.open)
        if back is not None:    # he changed the board and it no longer shows his taken answers:
            self._restore(*back)                    # they are taken back, his decision opens again
            self.voice.stats['board_changes'] += 1
        for _ in range(STEPS_PER_POLL):
            # ONE mover.  The comparison looks at the whole board when no step is in flight (shown,
            # dice asked: it is finished first, the board is ahead) and no decision of his is open -
            # or when he asked for it ("back": nothing he moved is his any more), or a client call
            # failed.  Else it carries only what has ceased to be his (held)
            repair, self.compare.repair, own = self.compare.repair, False, self.owned()
            idle = self.answers.waits is not None or not self.rand.step_open
            if self.compare.request is not None and idle:
                self.boards.back()
            whole = repair or (idle and (self.compare.request is not None or not own.whole))
            if whole or own.back:
                lines = self.compare.verify() if whole else self.compare.carry()
                if lines:
                    return lines
            if whole and self.compare.waits():
                return []       # (a card of his own left its effects to him: the game waits until his board shows them)
            lines = self._advance()
            if lines is not None:
                return lines
        return []

    def owned(self):
        """What on the board is the person's now (boardanswer.BoardAnswers.owned): the one
        answer for the loop - is the comparison kept off the board - and for the comparison -
        which pieces it leaves where they lie."""
        return self.boards.owned(self.answers.waits, self.pos, not self.compare.fresh)

    # ------------------------------------------------ answers on the board
    @contextlib.contextmanager
    def _probe(self):
        """Engine and step machine may be run inside and nothing is consumed: the funnel (its own
        probe) and the position are as before afterwards (a roll, a draw or a commit reached
        inside a probe stops it before anything is asked or closed)."""
        keep = (self.pos, self.voice.how, self.probing)
        self.probing = True
        try:
            with self.rand.probe():
                yield
        finally:
            self.pos, self.voice.how, self.probing = keep

    def _probe_step(self, state, spec, act):
        """The state after one step of the person (`act`, under the `spec` of its decision) on
        `state`: applied inside a probe, so nothing is consumed."""
        import ai
        with self._probe(), self.rand.advance():
            post = ai.clone_state(state) if spec['skip'] is not None and spec['skip'](act) \
                else ai.apply_action(state, act)
            if spec['flag'] is not None:
                setattr(post, spec['flag'], False)
            if spec['after'] is not None:
                spec['after'](post)
        return post

    def _probe_open(self, pos):
        """(side, kind, state, menu, spec, position) of the decision the step machine opens
        at `pos` with no roll, draw or other automatic step in between; None otherwise."""
        out = None
        try:
            with self._probe():
                self.pos = pos
                try:
                    for _ in range(STEPS_PER_POLL):
                        if self._advance() is not None:
                            break
                except _Opens as opens:
                    out = opens.args + (self.pos,)
        except (_Stops, rnd.NeedClientCall, Exception):
            out = None
        return out

    def _restore(self, before, entries):
        """THE one take-back: the game goes back to the position `before`, the last `entries`
        steps leave the journal, an open step and a search for what followed are thrown away,
        the open decision ends with its answer and its words.  A TAKE-BACK RESTORES THE BOARD:
        what the undone steps moved and the board has followed goes back with them - also a die,
        a card or a tile, which the comparison does not look at while the game is not settled."""
        if self.link.board is not None and self.human is not None:
            # (those the board has where the game as it is now would have them settled, and the game restored
            # would not - but what the restored game's own pass moves, or what is his: his die of a two-step action)
            ops = lambda state, settled: {op[2]: op[4] for op in reconcile.moves(reconcile.difference(
                state, self.link.board, self.link.view.card_pid_map, settled))[0] if op[0] == 'move'}
            was, now, own = ops(before.state, True), ops(self.pos.state, True), ops(before.state, False)
            self.boards.end(boardanswer.TAKEN_BACK, {
                p: z for p, z in was.items() if p not in now and p not in own and p not in self.owned().pieces
                and not reconcile.looks(p, False)}, said=True)
        del self.journal[len(self.journal) - entries:]
        self.pos = before
        self.rand.abort_step('taken back')
        self.answers.close(True)
        self.compare.recon = self.compare.left = None       # (... and what a card of the undone steps left to him)
        self.compare.spent, self.compare.round = set(), set()
        self.asked = self.shown = None
        self.used_random = False
        if self.link.board is not None:     # (the dice binding follows the engine state)
            self.link.bind(before.state, self.owned().pieces)

    # ------------------------------------------------------------ the engine
    def _advance(self):
        """One engine step.  Returns the lines to send ([] = waiting for the client or the
        person), or None when the step was silent and another may follow."""
        import ai
        import game_logic as gl
        gs = self.gs
        FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
        pos = self.pos
        st = pos.state
        if pos.over is not None:
            return self.voice.over_lines()
        if not self.probing:
            self.link.feed_random_discard()
        phase = pos.phase
        if phase == 'draws':
            return self._draws()
        if phase == 'open':                 # decisions the draws opened (hand limit)
            kind, side = ai._pending_decision_kind_side(st)
            if kind is None or kind == 'declare':
                return self._move(phase='declare')
            return self._decide(side, kind)
        if phase == 'declare':
            return self._declare()
        if phase == 'tail0':
            return self._apply('fellowship phase', gl.begin_fellowship_tail, phase='tail')
        if phase == 'tail':
            if not getattr(st, '_fellowship_tail_phase', False):
                if self._victory(st):
                    return None
                return self._move(phase='guide')
            side = ai._pending_decision_side(st)
            if side is None:
                raise RuntimeError('open Fellowship effect has no decision owner')
            return self._decide(side, ai._pending_decision_kind_side(st)[0] or 'fellowship_tail',
                                after=gl.resume_fellowship_tail)
        if phase == 'guide':
            if st.pending_guide_change is None:
                return self._move(phase='eyes')
            # the Fellowship phase ends here: once he has closed its declaration (his word or
            # his move), the Guide the board shows is his answer
            return self._decide(FP, 'guide_change', stands=answered(self.journal, 'declare', st.turn))
        if phase == 'eyes':
            if not any(not d.used for d in st.dice.get(SH, [])):
                return self._move(phase='roll')
            return self._decide(SH, 'eye', flag='_eye_allocation_phase', phase='roll')
        if phase == 'roll':
            return self._roll()
        if phase == 'actions':
            return self._action()
        if phase == 'end':
            return self._turn_end()
        raise RuntimeError('unknown phase %r' % (phase,))

    def _move(self, **change):
        """Advance the cursor only (no engine change, nothing to render)."""
        self.pos = self.pos._replace(**change)
        return None

    def _commit(self, state, label, who='auto', side=None, act=None, source=None, **change):
        """Adopt a whole new position; say what a player needs to follow it; check victory on it
        (after every commit).  source: where the person's answer came from, when the step is his."""
        if self.probing:
            raise _Stops(label)
        before, pre, dud = self.pos, self.pos.state, None
        if act is not None and act.action_type == 'play_event':
            import ai
            try:        # the engine's own test for a play that changes nothing
                with self.rand.pin():
                    if ai._pending_decision_kind_side(state)[0] is None and not \
                            ai._event_card_has_effect(pre, side, str(act.params.get('card_id'))):
                        dud = words.card_title(act.params.get('card_id'))
            except Exception:
                dud = None
        rolled, tiles = self.rand.step_commit()
        self.journal.append(Entry(label, before, who, bool(self.used_random or rolled)))
        self.pos, answered = before._replace(state=state, **change), self.answers.open
        self.answers.close(battle=state.pending_combat is not None)
        self.boards.committed(who, source, self.journal[-1].random, before, answered if side is self.human else None)
        self.asked = self.shown = None
        self.used_random = False
        if (pre.pending_combat is None) != (state.pending_combat is None):
            self.link.unpick()          # (a card he holds chosen answers the battle it was chosen in, and no other)
        self.voice.narrate(pre, state, tiles, act, dud)
        if self.link.board is not None:     # (the dice binding follows the engine state)
            self.link.bind(state, self.owned().pieces)
        self.voice.stats['steps'] += 1
        self.voice.log('step', label=label, turn=int(state.turn), phase=self.pos.phase)
        self._victory(state)

    def _victory(self, state):
        import game_logic as gl
        result = gl.check_all_victory_conditions(state)
        if gl.should_game_end(result, end_of_turn=False):
            self.pos = self.pos._replace(over=dict(result), phase='over')
            return True
        return False

    def _apply(self, label, fn, **change):
        """An automatic engine step: fn(clone) under the randomness funnel."""
        import ai
        if not self.probing and self.boards.settling():
            return []
        self.rand.begin_attempt()
        clone = ai.clone_state(self.pos.state)
        try:
            with self.rand.advance():
                fn(clone)
        except rnd.NeedClientCall as call:
            return self._ask(call)
        self._commit(clone, label, **change)
        return None

    def _draws(self):
        """The turn-start draws of both sides (one per deck, the person's
        first) and the hand limit, as game_logic.begin_turn_reset deals them."""
        import game_logic as gl
        FP, SH = self.gs.Side.FREE_PEOPLES, self.gs.Side.SHADOW
        st = self.pos.state
        order = (FP, SH) if self.human is not SH else (SH, FP)
        decks = [(s, d) for s in order for d in ('character', 'strategy')
                 if getattr(st, '%s_%s_deck' % ('fp' if s is FP else 'sh',
                                                'char' if d == 'character' else 'strat'))]
        self.rand.begin_attempt()
        call = self.rand.draws_request([(s.name, d) for s, d in decks])
        if call is not None:
            return self._ask(call)

        def deal(clone):
            for side, deck in decks:
                gl.draw_event_card(clone, side, deck)
            gl.open_hand_limit_discard(clone)
        return self._apply('draws', deal, phase='open')

    def _declare(self):
        """The Fellowship phase's declaration: the engine's declare menu under
        its own flag; an AI declares a reachable Mordor gateway without search."""
        import ai
        FP = self.gs.Side.FREE_PEOPLES
        st = self.pos.state
        if st.fellowship.mordor_entered or getattr(st, '_fellowship_tail_phase', False) \
                or st.fellowship.progress < 0 \
                or not ai._get_declare_actions(st, allow_in_place=True):
            return self._move(phase='tail0')
        gateway = ai.auto_mordor_gateway(st) if self.human is not FP else None

        def sanctioned(actions):
            from board import sanitize
            return next((a for a in actions if gateway is not None
                         and a.action_type == 'fellowship_declare'
                         and sanitize(str(a.params.get('region', ''))) == gateway), None)
        return self._decide(FP, 'declare', flag='_declare_phase', phase='tail0', sanctioned=sanctioned,
                            skip=lambda a: a.action_type != 'fellowship_declare')

    def _roll(self):
        """The action roll: each pool is rolled by the client; the faces of
        its roll line land on the engine's unrolled dice."""
        import ai
        import game_logic as gl
        if self.probing:
            raise _Stops('action roll')     # (a probe ends at the roll: it uses no result of it)
        st = self.pos.state
        waiting = {}
        for side in (self.gs.Side.SHADOW, self.gs.Side.FREE_PEOPLES):
            idx = [i for i, d in enumerate(st.dice.get(side, []))
                   if not d.used and d.current is None]
            if idx:
                waiting[side] = idx
        assigns, unfit = [], []
        for faces in self.inbox.rolls:
            # a roll belongs to the pool whose dice SHOW its faces on the board (the roll line
            # names no pool): never guessed from its size.  One that the dice of both pools
            # show is booked when both rolls are in - the faces are then the same
            fits = [s for s in waiting if sorted(faces) == reconcile.rolled(
                self.link.board, layout.side_tag(s), len(waiting[s]))]
            if len(fits) == 1 or (fits and self.inbox.rolls.count(faces) >= len(fits)):
                for i, letter in zip(waiting.pop(fits[0]), faces):
                    assigns.append((fits[0], i, getattr(self.gs.ActionDieFace, LETTER_FACE[letter])))
            elif not fits:
                unfit.append(faces)
        if waiting:
            if not self.probing and self.boards.settling():
                return []       # his Eyes may still change: the roll is asked when they stand
            if unfit:       # no pool of that size: no answer - compare the board, ask again in turn
                self.inbox.rolls = [r for r in self.inbox.rolls if r not in unfit]
                _failed(self.asked)
                self.compare.repair = True
                self.boards.release()
            self.rand.begin_attempt()
            return self._ask(rnd.NeedClientCall('action_roll', box='turn', sides={
                s.name: (ROLL_ZONE[s.name], []) for s in waiting}))
        clone = ai.clone_state(st)
        gl.adopt_rolled_faces(clone, assigns)
        clone.dice_rolled = True
        self.inbox.rolls = []
        self.rand.begin_attempt()
        self._commit(clone, 'action roll', phase='actions', active=self.gs.Side.FREE_PEOPLES,
                     owner=None, empty=frozenset())
        return None

    def _action(self):
        """Action resolution: the engine's own router (engine_play's loop)."""
        import ai
        FP, SH = self.gs.Side.FREE_PEOPLES, self.gs.Side.SHADOW
        pos = self.pos
        st = pos.state

        def unused(side):
            return sum(1 for d in st.dice.get(side, []) if not d.used and d.current is not None)
        kind, pending_side = ai._pending_decision_kind_side(st)
        active = pos.active
        if kind is None:
            if unused(FP) == 0 and unused(SH) == 0:
                return self._move(phase='end')
            if active is FP and unused(FP) == 0:
                active = SH
            elif active is SH and unused(SH) == 0:
                active = FP
            if unused(active) == 0:
                return self._move(phase='end')
        else:
            active = pending_side if pending_side is not None else active
        with self.rand.pin():
            actions = ai.get_legal_actions(st, active)
        if not actions:
            # a side with no legal action hands the rest of the round over
            other = SH if active is FP else FP
            empty = pos.empty | {active}
            if kind is None and other not in empty and unused(other) > 0:
                return self._move(active=other, empty=empty)
            return self._move(phase='end')
        owner = ai._turn_owner_after(st, active, pos.owner)
        return self._decide(
            active, kind or 'action',
            route=lambda post, act: dict(
                active=ai._active_after_action(post, act, active, owner),
                owner=owner, empty=frozenset()))

    def _turn_end(self):
        """The turn boundary: the engine's begin_turn_reset on a clone (the
        draws follow as their own step); the client recovers both pools."""
        import ai
        import combat
        import game_logic as gl
        if self.probing:
            raise _Stops('turn end')
        if not self.probing and self.boards.settling():
            return []
        st = self.pos.state
        blockers = ai._turn_end_blockers(st)
        if blockers:
            raise RuntimeError('the turn cannot end: ' + ', '.join(blockers))
        self.rand.begin_attempt()
        clone = ai.clone_state(st)
        with self.rand.advance():
            if combat.FINAL_HITS_HOOK is None:
                gl.enforce_settlement_control(clone)
            over = gl.begin_turn_reset(clone, int(st.turn) + 1, deal_cards=False, bonus_dice=True)
        if over:
            result = dict(getattr(clone, '_search_endturn_victory', None)
                          or gl.check_all_victory_conditions(clone))
            self.journal.append(Entry('turn end', self.pos, 'auto', False))
            self.pos = self.pos._replace(state=clone, over=result, phase='over')
            self.rand.step_commit()
            return None
        self._commit(clone, 'turn end', phase='draws', active=self.gs.Side.FREE_PEOPLES,
                     owner=None, empty=frozenset())
        del self.journal[:-1]           # (the journal is of one turn: nothing is taken back across its end)
        self.inbox.rolls = []
        self.voice.how = 'The turn is over: both dice pools are recovered.'
        self.voice.outbox.insert(0, self.voice.chat(self.ai_sides[0], words.turn_over(int(st.turn))))
        return ['!CALL menu frecover', '!CALL menu srecover']

    # ------------------------------------------------------------ decisions
    def _menu(self, state, side):
        """The options of `side` in `state`: every legal action for a person,
        the policy's menu for the AI."""
        import ai
        from ai_policies import project_ai_actions
        with self.rand.pin():
            legal = list(ai.get_legal_actions(state, side))
        if side is self.human:
            return legal
        return list(project_ai_actions(state, side, legal).actions) or legal

    def _choose(self, state, side, kind, menu, owner):
        """The AI's pick.  The search is handed a state in which the cards it
        may not see are dealt again at random (redact)."""
        import ai
        from ai_policies import action_policy_key
        if len(menu) == 1:
            return menu[0]
        if self.search is None:
            return self.rng.choice(menu)        # test harness without nets
        position = self.rand.prefers['strength'] or strength.default(turn_search.gpu())
        stated = self.rand.prefers['sims']          # (the panel's simulations knob; -1: not moved)
        sims, setting = strength.plan(position, menu, self.sims if stated < 0 else stated)
        self.voice.how = words.ai_thinking(side, sims)
        if self.voice.write_status is not None:
            self.voice.write_status(_enrich=False, state='thinking', waiting_on=self.voice.how)
            self.voice.publish(self.facts)
        seed = self.rng.getrandbits(32)
        handed = redact(state, side, seed)
        assert_blind(state, handed, side)
        if self.on_search is not None:
            self.on_search(state, handed, side, seed)
        with ai.search_turn_owner(owner):
            act = self.search.choose(handed, side, num_simulations=sims)[0] if sims else self.search.raw(handed, side, menu)
        if act is None:
            raise RuntimeError('the search returned no action for %s %s' % (side.name, kind))
        key = action_policy_key(act)
        hit = next((a for a in menu if a.action_type == act.action_type
                    and a.die_idx == act.die_idx and a.params == act.params), None) \
            or next((a for a in menu if action_policy_key(a) == key), None)
        if hit is None:
            raise RuntimeError('the search picked an action outside the menu: %s' % act.action_type)
        # TURN SEARCH (core/turn_search.py): the rest of the turn and the next one are played out for the candidates;
        # a move confirmed as better replaces the pick.  Off, refused or failed: the pick stands.  How hard both
        # searches work at this decision is strength.plan's word (the menu's slider; small decisions cheaply).
        return turn_search.improve(ai, handed, side, owner, seed, menu, hit, setting)

    def _hand_back(self):
        """What the funnel handed back, and what no step used and has left the bank: said once - and the
        piece of such a result is nobody's: the comparison carries it back with this reply."""
        unused, self.inbox.unused = self.inbox.unused + self.rand.take_dropped(), []
        if unused and self.human is not None:
            self.boards.end(boardanswer.UNUSED, *self.link.ignored(unused))

    def _decide(self, side, kind, flag=None, phase=None, after=None, sanctioned=None,
                skip=None, route=None, stands=False):
        """One decision of either seat: choose (once per step), apply on a
        clone, show, adopt.  flag: an engine phase flag held while choosing;
        after(post): the engine's follow-up; skip(act): the choice changes
        nothing on the board; stands: the decision ends a window of the person -
        when the board as it stands answers it, it is not asked."""
        import ai
        base = self.pos.state
        state = base
        human = side is self.human
        if not self.probing and not human and not self.rand.step_open \
                and base.pending_combat is None and base.pending_hunt is None:
            self.link.drop_early()          # the AI acts: they answer nothing of its step
            self._hand_back()
        if flag is not None:
            state = ai.clone_state(base)
            setattr(state, flag, True)
        if not self.probing:
            self.rand.begin_attempt()
        menu = self._menu(state, side)
        if not menu:
            raise RuntimeError('no legal choice for %s %s' % (side.name, kind))
        spec = {'flag': flag, 'phase': phase, 'after': after, 'skip': skip, 'route': route,
                'stands': stands}
        if self.probing:                # the matcher asks what opens here: no step is taken
            raise _Opens(side, kind, state, menu, spec)
        if human and len(menu) == 1 and self.answers.click is None and not self.boards.chain and self.boards.settling():
            return []       # (the one answer left is given when his answer before it stands)
        picked = self.answers.pick(kind, state, side, menu, spec) if human else None
        if human and picked is None:
            self.voice.waits(kind, side, base)
            return []

        def chooser(actions):
            pick = picked if human else sanctioned(actions) if sanctioned is not None else None
            if pick is None:
                pick = self._choose(state, side, kind, actions, self.pos.owner)
            return [pick] + [a for a in actions if a is not pick]
        # a board answer may carry the person's own pieces as parameters: it is not a menu row
        act = self.rand.journal_pick(
            menu if picked is None or any(a is picked for a in menu) else [picked] + menu, chooser)[0]
        if not human and self.boards.settling():
            return []       # (the AI's step is chosen now and committed when his answer stands)
        silent = skip is not None and skip(act)
        click = self.answers.click if human else None
        fix = click.fix if click is not None else None
        try:
            with self.rand.advance():
                post = ai.clone_state(state) if silent else ai.apply_action(state, act)
                if flag is not None:
                    setattr(post, flag, False)
                if after is not None:
                    after(post)
                if fix is not None:     # the engine holds the pieces he moved (menu.realise)
                    fix(post)
        except rnd.NeedClientCall as call:
            call.box = card_box(side if human else None, kind, act)
            return self._show_before(side, kind, act, base, silent) + self._ask(call)
        except Exception as e:
            if not (human and self.answers.shown()) or (fix is None and any(
                    panel.identity(a) == panel.identity(act) for a in menu)):
                raise
            # the move fits by counts, and the step made with the very pieces he took failed.  Which
            # pieces he takes is his to choose: this is an ERROR OF THE BRIDGE (of the engine, of the
            # fix, of the reading), never a rule of the game - logged with the position like any
            # error, counted, and told as that.  No answer; his board stays; a click answers
            self.voice.stats['exceptions'] += 1
            self.rand.abort_step('an answer made on the board failed')
            self.boards.failed(self.voice.save_error(e, self.position))
            self.answers.drop()
            self.voice.waits(kind, side, base)
            return []
        lines = []
        if not silent and not self.shown:
            lines = self._render(side, kind, act, base, post)
        elif self.shown and not self.shown.get('reveal') and self._reveal_due(base, post, act):
            lines = ['!CALL reveal_chosen']
        held = lines is None
        label = words.option_label(base, side, act, self.human)
        change = dict(route(post, act)) if route is not None else {}
        if phase is not None:
            change['phase'] = phase
        given = self.answers.taken() if human else None         # (an answer may be given for him)
        self._commit(post, '%s %s %s' % (layout.side_tag(side), kind, act.action_type),
                     who='forced' if given in ('only', 'stands') else 'human' if human else 'ai',
                     side=side, act=act, source=given, **change)
        self.compare.played(base, act, post)
        if not self.rand.prefers[card_box(side if human else None, kind, act)]:
            self.compare.left = self.compare.left or [0, '']    # (what his card brings about for his figures is his to show)
        opened = any(getattr(base, k, None) is None and getattr(post, k, None) is not None
                     for k in ('pending_combat', 'pending_hunt'))
        self.voice.told(side, label, click)
        if opened:
            self.link.drop_early()          # they answer no combat or Hunt roll
            self._hand_back()
            self.link.battle(not human and lines != [])     # (shown by this reply, or by the comparison)
        return [] if held else (lines or None)

    def _reveal_due(self, pre, post, act):
        """Both combat cards are committed and at least one is a card."""
        battle = pre.pending_combat or {}
        closed = battle.get('phase') == 'defender_combat_card' if post is None else (
            battle.get('phase') in CARD_PHASES
            and (post.pending_combat or {}).get('phase') not in CARD_PHASES)
        return bool(closed and (act.action_type == 'combat_card_play'
                                or battle.get('atk_combat_card') or battle.get('def_combat_card')))

    def _show_before(self, side, kind, act, pre, silent):
        """A step that needs client dice or a draw is shown BEFORE they are asked (the card play,
        the die, the move), once; its result follows from the comparison after the commit."""
        if self.shown is not None or silent:
            return []
        self.shown = {}
        if kind in ('card_option', 'card_followup') or act.action_type in SHOWN_AFTER:
            return []
        holds = self.voice.stats['holds']
        lines = self._render(side, kind, act, pre, None)
        self.voice.stats['holds'] = holds     # not yet a failure: the commit renders it
        if not lines:
            return []
        self.shown = {'shown': True, 'reveal': '!CALL reveal_chosen' in lines}
        return lines

    def _render(self, side, kind, act, pre, post):
        """One step as client lines (reused emission), under the seat's nick.  What it cannot
        render is left to reconcile, which shows the difference.  Of an answer made with pieces
        he put on the board - read from it, a step of a chain or one, or named with a part of the
        move there (Answers.shown: THE answer to "is it already there") - nothing is sent: his
        pieces stand, and the rest - of the move he named, of what the engine did alone - is
        shown by the comparison (ruling 3)."""
        if side is self.human and self.answers.shown():
            return []
        try:
            emitted = emission.render_step(self.link.view, self.link.nick(side), side, kind, act, post, pre)
        except Exception as e:
            self.voice.stats['holds'] += 1
            self.voice.note('render %s %s failed: %r' % (kind, act.action_type, e))
            return None
        if emitted.hold:
            self.voice.stats['holds'] += 1
            self.voice.note('render %s %s held: %s' % (kind, act.action_type, str(emitted.hold)[:200]))
            return None
        lines = []
        own = side is self.human and self.answers.source() == 'chooser'
        # (what a card of his own brings about for HIS FIGURES is his to show when he chose so: no line of
        #  the step moves one - the step's die, its card and every other piece are shown; compare.Comparison._left)
        left = () if self.rand.prefers[card_box(side if side is self.human else None, kind, act)] else layout.FIGURES
        for line in emitted.lines:
            op, f = em.classify_line(line)
            if op == 'combat_card_pick' and own:
                continue                # the person already picked it in the client
            named = [p for key in ('pieces', 'piece', 'die', 'dice')
                     for p in ((f or {}).get(key) if isinstance((f or {}).get(key), list)
                               else str((f or {}).get(key) or '').split())]
            if any(str(p).isdigit() and str(p) not in self.link.board.zone for p in named):
                self.voice.stats['holds'] += 1    # never a line for a piece the board does not have
                self.voice.note('line for a piece that is not on the board, not sent: %s' % line[:80])
                continue
            if left and any(str(p).isdigit() and layout.family(p) in left and layout.side_of(p) == layout.side_tag(side)
                            for p in named):
                continue
            lines.append(line)
        # client randomness is only ever asked through the funnel (_ask): a
        # draw or roll the emission would order again is already consumed
        lines += [c for c in emitted.calls if not c.startswith(boardanswer.RANDOM_CALLS)]
        if self._reveal_due(pre, post, act):
            lines.append('!CALL reveal_chosen')     # both commitments are in: reveal together
        # nothing is said in the person's name: his words in chat are his own
        return lines + [self.voice.chat(side, text) for _k, text in emitted.chat
                        if side is not self.human]

    # ------------------------------------------------------ client randomness
    def _ask(self, call):
        """Ask for a client result ONCE; it is accepted whenever it arrives.  WHO MAKES IT is the
        funnel's to say (randomness.directives_for, by the box the request carries): the client
        on a call, or the person with his own hand - his dice, and his cards and action dice
        where he chose so.  A person is asked for what he makes himself once (the status and the
        panel keep saying it).  The client is asked AGAIN
        only on a fact: it did not carry the reply out (its receipts), or it said the call FAILED
        (and the board has been compared since).  When it carried the call out and no result has
        come for SILENT_S seconds since its receipt was read, nothing more is sent by itself: he is TOLD
        once what leads on (the word "retry" asks again: _chat_word; a result made by hand is
        taken like any other), and the status keeps saying it.  EVERY call has this form: the
        bridge makes no random result of the game itself - dice, draws and lots are the game
        window's.  A step that asks for dice or a draw - the person's own too - ends the window
        of his board answer: it stands."""
        if self.probing:
            raise _Stops(call.kind)
        self.used_random = True
        self.boards.close()
        asks = self.rand.directives_for(call)
        # (his: what he makes with his own hand, as words need it - how many dice, which decks; calls: the rest)
        his, calls = [d.text for d in asks if d.kind == 'hand'], [d for d in asks if d.kind != 'hand']
        need, ctx = int(call.kw.get('need', 1)), call.kw.get('ctx')
        # one open request is ordered once: it is told from the next by the funnel's count of
        # answered requests, never by what has come in for it (a person who rolled too few
        # dice is asked for the rest)
        ident = (call.kind, self.rand.answered, call.kw.get('need') if his else None)
        self.voice.how = words.waiting_for_client(call.kind) if not his else words.waiting_for_roll(need, ctx) \
            if call.kind == 'd6' else words.by_hand(None, call.kind, his)
        again = self.asked is not None and self.asked[0] == ident
        if again:
            state, since, self.asked[3] = self.asked[1], self.asked[2], bool(his)
            # (what he makes himself is asked once and waited for; a call beside it that was lost or FAILED is
            #  ordered again like any call - a FAILED one when the board has been put right first)
            if (his and not (calls and (state == FAILED or self.link.receipts.missing))) or (
                    state == FAILED and self.compare.repair):
                return []
            if state == TOLD or (state == SENT and not self.link.receipts.missing):
                self.asked[2] = since = self.now() if since is None else since  # (the clock: from its receipt's reading)
                if state == SENT and self.now() - since < SILENT_S:
                    return []
                self.voice.how = words.client_silent(call.kind)
                if state == TOLD:
                    return []
                self.asked[1] = TOLD
                self.voice.stats['silent_calls'] += 1
                self.voice.note('the client did not answer: %s' % (ident,))
                return [self.voice.chat(self.ai_sides[0], self.voice.how)]
            self.voice.stats['reasks'] += 1
            self.voice.note('asking the client again: %s' % (ident,))
        else:
            self.voice.stats['asks'] += 1
        self.asked = [ident, SENT, None, bool(his)]
        out = []
        if call.kind == 'tile':
            # the client's pool is made the engine's pool right before the draw
            out += reconcile.render(reconcile.pool_moves(call.pool, call.used, self.link.board),
                                    self.link.nick)
        if call.kind == 'action_roll' and self.link.board is not None:
            # a roll takes every die that lies in a pool: the dice on the table are made the game's first
            out += reconcile.render(reconcile.dice_moves(self.pos.state, self.link.board), self.link.nick)
        out += [d.text if d.kind == 'call' else self.voice.chat(self.ai_sides[-1], d.text) for d in calls]
        if his and not again:
            out.append(self.voice.chat(self.ai_sides[0], words.roll_request(self.human, need, ctx)
                                       if call.kind == 'd6' else words.by_hand(self.human, call.kind, his)))
        return out

    # ----------------------------------------------- the person: panel, words
    def panel_poll(self, text):
        """What the panel's request slot gave (transport.panel_cb, between
        handlers; None: nothing).  True when a request was answered."""
        req = panel.request(text)
        if req is None:
            return False
        if req.get('human_nick') and self.human is not None:
            self.link.human_nick = str(req['human_nick'])
        reply = self._panel_request(req)
        if reply is not None:
            reply.update(token=req.get('token'), op=req.get('op'))
            self.voice.replies.insert(0, reply)
            del self.voice.replies[panel.REPLIES_KEPT:]
            if reply.get('message'):
                self.voice.record('panel', reply['message'])
        self.voice.publish(self.facts)
        return True

    def _panel_request(self, req):
        op = req.get('op')
        if op in ('refresh', 'perform_report'):
            return None
        if op == 'recovery':
            what = req.get('recovery')
            if what in ('put_back', 'resync'):
                text = self._chat_word('back', panel=(req.get('token'),))
            elif what == 'take_back':
                text = self.take_back()
            else:       # AI strength is taken; a lever this bridge has not says so, never silent
                sims = (req.get('args') or {}).get('sims')
                if what == 'set_sims' and isinstance(sims, int):
                    self.sims = max(1, min(1000, sims))
                text = words.lever(what, self.sims)
                self.voice.lever_said = {'ok': what == 'set_sims', 'summary': text, 'ts': time.time()}
                self.voice.log_tail.append({'ts': time.time(), 'kind': 'info', 'text': text})
            return {'status': 'ok', 'message': text}
        return self.answers.panel_request(req)

    def _chat_word(self, body, unseen=False, panel=None):
        """A chat line of the person: 'back' and 'undo', or a word for his
        open decision.  Ordinary talk does nothing.  (unseen: ClientLink.heard; panel: 'back'
        asked with the panel's lever - (the token of that click,) - -> the answer to it.)"""
        text = words.chat_key(body)
        if self.pos.over is not None and panel is None:     # no word does anything now: said, never silent
            if text in ('back', 'resync', 'undo', 'finish') or text in words.ALIASES or words.old_word(text):
                self.voice.say(words.game_over(self.pos.over))
        elif text in ('back', 'resync'):        # ("resync" was the old bridge's word for it)
            waits = self.rand.step_open and self.answers.waits is None
            said = self.compare.put_back(waits, 'panel' if panel else 'chat', panel[0] if panel else None)
            if waits and panel is None:     # it waits for a roll or a draw: said now
                self.voice.say(said)
            return said
        elif text == 'retry' and self.asked is not None and self.asked[1] == TOLD:
            self.voice.say(words.waiting_for_client(self.asked[0][0]))      # (asked again with this reply; no piece
            self.asked = None                                               # is touched: "back" is another word)
        elif text == 'finish':
            # what the bridge had left to his own hand it makes itself now: his part of the open request
            # is handed to the client (asked anew with this reply), and what a card of his left to him
            # the comparison carries like everything else
            call = self.asked is not None and self.asked[3]
            self.voice.say(words.finishing(call or self.compare.left is not None))
            if call:
                self.rand.hand, self.asked = (self.rand.hand[0], False), None
            self.compare.left = None
        elif text == 'undo':
            self.voice.say(self.take_back())
        else:
            self.answers.hear(text, self.answers.open if self.answers.click is None else None, unseen)

    def take_back_reason(self):
        """Why the last decision cannot be taken back (None = it can).  His answer may be
        withdrawn as long as it may be changed on the board: until the next step that is not
        his is committed, or a roll or draw was asked for one."""
        if self.human is None or self.pos.over is not None:
            return words.take_back('nothing')
        if self.answers.click is not None:
            return words.take_back('resolving')
        n = last_answer(self.journal)
        if self.asked is not None or not n or self.journal[-n].who != 'human':
            return words.take_back('moved on')
        if any(e.random for e in self.journal[-n:]):
            return words.take_back('rolled')
        return None

    def take_back(self):
        """'Take back my last decision': the position before it, while the AI
        has not answered it; the board follows through the comparison."""
        why = self.take_back_reason()
        if why is not None:
            return why
        n = last_answer(self.journal)
        self.boards.abandon(taken_back=True)        # (nothing is his any more: no piece he moved,
        self.link.unpick()                          # and no card he holds chosen - the one he played by it too)
        self._restore(self.journal[-n].before, n)
        self.voice.stats['takebacks'] += 1
        self.voice.how = words.take_back('done')
        return self.voice.how

    # ------------------------------------------------------------ small things
    def position(self, blind=False):
        """The game position as a saved-position file holds it: the engine state, the
        turn cursor, and the card pieces whose identity the client has shown.  blind: the
        cards one seat or neither has seen are dealt again, and none of their pieces is named."""
        p = self.pos
        state, cards = p.state, dict(self.link.view.card_pid_map)
        for viewer in (p.state.hands if blind else ()):
            unseen = {holder[key] for holder, key in unseen_slots(p.state, viewer)[0]}
            cards = {c: pid for c, pid in cards.items() if c not in unseen}
            state = redact(state, viewer, 0)
        return {'state': state, 'phase': p.phase, 'active': p.active, 'owner': p.owner,
                'empty': list(p.empty), 'cards': cards}

    def load_position(self, path):
        """Start from a saved position instead of a fresh game (--position, a test
        switch; not in the bat).  Nothing else is needed: the first polls make the
        client's board equal the position through the comparison, from any board.
        LIMIT: the identity of a card piece is the client's commitment, made anew for
        every game.  A card the client of THIS game has not shown cannot be put in a
        hand, and on a real client the position plays only up to the next draw (a drawn
        card may be one the saved engine has elsewhere: the funnel stops).  The stand-in
        of the tests is given the commitment of the game the position was saved in."""
        import pickle
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.pos = Position(d['state'], d.get('phase') or 'actions',
                            d.get('active') or self.gs.Side.FREE_PEOPLES, d.get('owner'),
                            frozenset(d.get('empty') or ()), None)
        self.link.view.card_pid_map.update(d.get('cards') or {})
        self.journal, self.inbox.rolls = [], []
        self.compare.fresh = True
        self.voice.note('started from the saved position %s (turn %d, %s)'
                   % (os.path.basename(path), int(self.pos.state.turn), self.pos.phase))

    def facts(self):
        """What the panel's view is built from (panel.Facts), read off the loop as it is now."""
        human = self.human
        decision = None if self.pos.over else self.answers.decision_view()
        return panel.Facts(
            pos=self.pos, human=human, ai_sides=self.ai_sides, sims=self.sims, net_tag=self.net_tag,
            session=self.link.session, human_nick=self.link.human_nick,
            ai_nick=None if human is None else self.link.nick(next(iter(self.ai_sides))),
            decision=decision, take_back=None if human is None else self.take_back_reason(),
            how=self.voice.how, ai_last=self.voice.ai_last, lever_said=self.voice.lever_said,
            log_tail=self.voice.log_tail, replies=self.voice.replies)


def build(client_dir, side='AUTO', sims=300, net_tag='auto', seed=None, load_nets=True, poll=0.2,
          position=None):
    # the folder's lock is taken BEFORE the nets are loaded: a second server is refused at
    # once, and whoever started this one can tell a loading server from a dead one
    bridge = FileBridge(client_dir, poll=poll)
    bridge.acquire_lock()
    server = Server(client_dir, side=side, sims=sims, net_tag=net_tag, seed=seed,
                    load_nets=load_nets)
    if position:
        server.load_position(position)
    bridge.status_extra = lambda: server.voice.status(server.sims)
    bridge.panel_cb = lambda: server.panel_poll(bridge.panel.take())
    server.voice.write_status = bridge.write_status
    return server, bridge


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--client-dir', required=True)
    ap.add_argument('--side', choices=('AUTO', 'WATCH', 'FP', 'SA', 'SH'), required=True)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--net-tag', default='auto')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--position', default=None, help='test switch: start from a saved position')
    a = ap.parse_args(argv)
    server, bridge = build(a.client_dir, side=a.side, sims=a.sims, net_tag=a.net_tag, seed=a.seed,
                           position=a.position)
    turn_search.start(a.net_tag, a.client_dir)      # its own process; loads while the game is set up
    bridge.serve(server.handle)


if __name__ == '__main__':
    main()
