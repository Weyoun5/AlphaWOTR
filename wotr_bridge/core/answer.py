"""The open decision of the person and his answer to it: which source answered.

pick() asks the sources in order - a click or word already given, the one legal answer, the
board as it stands, what he did on the board (with the word that waits for the board's
verdict), a closing word with no board, the client's own card chooser - and gives the action,
or None while he has not answered.  An Answers imports no other part of the loop: it is handed
the Voice, the client link, the inbox, the funnel, the board answers (boardanswer.BoardAnswers),
the position (a callable) and the test hook for a decision shown.
"""
import contextlib
from collections import namedtuple

from . import boardmodel, layout, panel, reconcile, typed, words

# the person's open decision: its id, the kind of the step machine, his side, the state it is
# asked in (a phase flag set), the legal actions
Decision = namedtuple('Decision', 'id kind side state menu')


class Answer(object):
    """The person's answer to his open decision: the option (identity) and who gave it (source:
    'panel' with the token of that click, 'chat', 'board', 'chooser', or given for him - 'only',
    'stands').  An answer made with the pieces he put on the board - read from it, or named
    with a part of the move there - carries the action made with them and the fix that makes
    the engine hold them (menu.realise).  judged: the board was asked whether it shows a named
    answer (once: Answers._made).  at: the client's own time of a click (seconds; link.track)."""
    __slots__ = ('decision', 'identity', 'source', 'token', 'action', 'fix', 'judged', 'at')

    def __init__(self, decision, identity, source, token=None, action=None, fix=None, at=None):
        self.decision, self.identity, self.source, self.token = decision, identity, source, token
        self.action, self.fix, self.judged, self.at = action, fix, False, at


OPENS = 'opens'     # in the place of a battle: the battle that the attack of this decision opens - (OPENS, its id)


def _battle(state):
    """Which battle is open (its regions), or None."""
    battle = state.pending_combat
    return tuple(str(battle.get(k)) for k in ('attacker_region', 'defender_region', 'region')) \
        if isinstance(battle, dict) else None


def owed_by(call, rand):
    """Does this client call wait for dice the person rolls with his OWN HAND?  (rand: the funnel -
    who makes a request is its to say, ClientRandomness.directives_for.)"""
    return call is not None and call.kind == 'd6' and rand.directives_for(call)[0].kind == 'hand'


class Answers(object):
    def __init__(self, voice, link, inbox, rand, boards, human, position, on_decision):
        self.voice, self.link, self.inbox, self.rand = voice, link, inbox, rand
        self.boards, self.human = boards, human
        self.position, self.on_decision = position, on_decision
        self.open = None            # the person's open decision (a Decision)
        self.click = None           # his answer to it (an Answer)
        self.built = (None, None)   # (id, built view) of the open decision
        self.early = {}             # words given before their question: {'card' | 'round': (word, its battle)}
        self.closing = None         # a closing word waiting for the board's verdict (ruling 8)
        self.typed = None           # a word that may name one choice of the board: (id, word, else say, option named)
        self.later = []             # words typed while another waited for the board's verdict: heard in turn

    def close(self, taken_back=False, battle=True):
        """THE one place where the open decision of the person ends.  taken_back: it is undone
        with its answer (a take-back, his board answer changed) - the answer and every word
        kept for it go with it; battle: a battle is still open (a word kept for a question of
        a battle dies with that battle)."""
        self.open = None
        if taken_back:
            self.click = self.closing = self.typed = None
        if taken_back or not battle:
            self.early = {}
        del self.later[:]

    def source(self):
        """Who gave his answer (Answer.source), or None while there is none."""
        return self.click.source if self.click is not None else None

    def shown(self):
        """Is his answer made with the pieces he put on the board himself?  (Nothing of it is rendered.)"""
        return self.click is not None and self.click.action is not None

    @property
    def waits(self):
        """His open decision while it waits for him (no answer of his yet), else None."""
        return self.open if self.click is None else None

    def drop(self, again=False):
        """His answer is not carried out (the game refused his pieces; again: an error stopped
        it, and the decision is asked anew once the board is the game's again)."""
        self.click = None
        if again:
            self.open = None

    def taken(self):
        """His answer is being committed -> where it came from (Answer.source), or None when
        the decision was not his to answer.  A card he holds chosen that did not give the
        answer to his card decision answers nothing any more: the client is told."""
        given, self.click = self.source(), None
        if given != 'chooser' and self.open is not None and words.early_fits('card', self.open.kind):
            self.link.unpick()
        return given

    # ------------------------------------------------------------ the sources
    def pick(self, kind, state, side, menu, spec=None):
        """The person's answer to the open decision, or None while he has not
        answered: a panel click or chat word (click), for a combat card the
        card he picked with the client's own chooser, or what he did on the
        board (the matcher; one step of it per decision)."""
        dec_id = 'd-%d-%s-%s' % (self.voice.stats['steps'], kind.split(':')[-1].replace('_', '-'),
                                 panel.menu_sig(menu)[:6])
        waited = self.open is not None and self.open.id == dec_id      # (he was asked before this poll)
        o = self.open = Decision(dec_id, kind, side, state, menu)
        click = self.click
        if click is not None and click.decision != dec_id:
            self.voice.result(click, 'superseded', words.stale())
            click = self.click = None
        # (a word kept for another battle than the one that is open answers nothing; a word typed with
        # the attack itself is kept for the battle that attack opens, and for no other decision)
        now, kept = _battle(state) if self.early else None, {}
        for q, (word, b) in self.early.items():
            if b is not None and b[0] == OPENS:
                if now is not None:
                    kept[q] = (word, now)       # the battle his attack opened
                elif b[1] == dec_id:
                    kept[q] = (word, b)         # his attack is not carried out yet
            elif b == now:
                kept[q] = (word, b)
        self.early = kept
        asked = next((q for q in self.early if words.early_fits(q, kind)), None)
        if click is None and asked is not None:
            self.hear(self.early.pop(asked)[0], self.open)     # a word given before this question
            click = self.click
        if click is None and self.closing is None and self.typed is None and self.later:
            self.hear(self.later.pop(0), self.open)             # (the word typed after the one judged last poll)
            click = self.click
        closing, self.closing = self.closing, None
        if closing is not None and (closing[0] != dec_id or click is not None):
            closing = None
        typed, self.typed = self.typed, None
        if typed is not None and (typed[0] != dec_id or click is not None):
            typed = None
        for source in (self._only, self._stands, self._board, self._word):
            if click is None:
                click = source(o, spec, closing, typed)
        if click is not None:
            return self._made(o, spec, click)
        return self._chooser(o, waited)

    def _only(self, o, spec, closing, typed):
        """ONE legal answer is not asked (P-096) - unless the board has a say in it: he picks
        the pieces it is made with (some of several of a class: which Leader, which dice -
        that pick stays his, ruling 3), or pieces he moved lie there that this decision reads
        (WHOLE BOARD OR NOTHING: they are that answer, or he is told - _board); and unless it is
        the only one because of the cards he holds, which nobody across the table sees (owner:
        "it's not supposed to know my hand"): his combat card, while he has a card in hand."""
        menu = o.menu
        hidden = words.early_fits('card', o.kind) and bool(o.state.hands[o.side].hand)
        if len(menu) == 1 and not self.boards.chain and not hidden:
            plan, base, moved = None, None, False
            if self.link.board is not None:
                with contextlib.suppress(Exception):
                    g = self.boards.gesture(o, spec)
                    plan, base = g.menu.plan(menu[0]), g.opening.base
                    moved = self.boards.moved(g)
            if not moved and (not plan or not boardmodel.picks(base, plan)):
                self.click = Answer(o.id, panel.identity(menu[0]), 'only')
        return self.click

    def _stands(self, o, spec, closing, typed):
        """The end of a window of his: the board as it stands is his answer, and nothing
        is asked when it gives one (the Guide who sits in the Guide box)."""
        if (spec or {}).get('stands') and not self.boards.chain and self.link.board is not None:
            keep = self.boards.stands(self.boards.gesture(o, spec))
            if keep is not None:
                self.click = Answer(o.id, keep, 'stands')
        return self.click

    def _board(self, o, spec, closing, typed):
        """What he did on the board, one step of it per decision; a closing word and a typed
        word are answered by the board's verdict."""
        if self.link.board is None:
            return None
        click, menu = None, o.menu
        steps = self.boards.chain
        if not steps:
            judged = None
            try:
                steps = self.boards.watch(o, spec, confirm=closing is not None) or []
                judged = self.boards.gest.v
            except Exception as e:  # the matcher failed: the panel and the words still answer
                self.voice.stats['match_errors'] += 1
                self.voice.note('the board could not be judged: %r' % (e,))
            if closing is not None and not steps:
                # ruling 8: the word commits the option without a move only when the
                # board shows nothing (or allows exactly that option on the way)
                named = any(panel.identity(a) == closing[1]
                            for a in getattr(judged, 'words', None) or () if hasattr(a, 'params'))
                # (... or nothing but a die or card laid down for an action, with a word that
                # names an option: he says what he does, the staging pieces lie as they lay)
                what = words.confirm(getattr(judged, 'kind', None), named or (closing[1] is not None and (
                    boardmodel.stages_only(getattr(judged, 'delta', None) or {}, True))))
                # (a word that closes the decision without naming an option of it: the
                # option that leaves the board as it stands)
                ident = closing[1] or (self.boards.stands(self.boards.gest) if what == 'word' else None)
                if what == 'word' and ident is not None:
                    click = self.click = Answer(o.id, ident, 'chat')
                elif what == 'word':
                    self.voice.say(words.say_choices(o.state, o.side, menu, 'none'))
                elif getattr(judged, 'kind', None) == 'wait':
                    self.voice.say(words.board_unclear('The move on the board is not finished'))
            if typed is not None and not steps:
                steps = self.boards.named(o, typed[1], judged)
                opt = next((a for a in menu if panel.identity(a) == typed[3]), None)
                g = self.boards.gest
                plan = g.menu.plan(opt) if opt is not None and g is not None and g.opening is not None else None
                if not steps and plan is not None and all(ch.opt for ch in plan):
                    # (it names an option that moves nothing: no board can show it, its own word
                    # answers - on any board that is not itself a ready answer)
                    click = self.click = Answer(o.id, typed[3], 'chat')
                elif not steps and typed[2]:
                    self.voice.say(typed[2])
        first = self.boards.queue(steps)
        if first is not None:
            opt, action, fix = first
            want = panel.identity(opt)
            if any(panel.identity(a) == want for a in menu):
                click = self.click = Answer(o.id, want, 'board', action=action, fix=fix)
            else:       # the step machine opened something else: judge the board anew
                self.voice.note('a board answer did not fit the decision that opened: %s'
                                % opt.action_type)
                self.boards.queue(())
        return click

    def _word(self, o, spec, closing, typed):
        """A closing word when there is no board to ask."""
        if closing is not None and self.link.board is None:
            self.click = Answer(o.id, closing[1], 'chat')
        return self.click

    def _made(self, o, spec, click):
        """The action of his answer.  An answer he NAMED - by a click or a word - is made with
        the pieces HE has put on the board for it: what else he moved is the next decision's to
        read, or goes back (boardanswer.committed).  It stays the answer he named and answers
        ONE decision; a chain of a board answer that waited is given up for it, and where the
        board is a chain that begins with the named step, the decision of his that follows
        reads the rest in its own way (boardanswer._ahead).  The board is asked once, when the
        answer is first seen: a step that then waits for its dice stays what it was.  (What the
        bridge laid there for a click is not his, and is no part of it.)"""
        if click.source in ('panel', 'chat') and not click.judged and self.link.board is not None:
            steps = None
            try:
                g = self.boards.gesture(o, spec)
                steps = g.menu is not None and self.boards.on_board(g, o.state, click.identity, True)
            except Exception as e:  # the matcher failed: the click and the word still answer
                self.voice.stats['match_errors'] += 1
                self.voice.note('the board could not be judged: %r' % (e,))
            first = self.boards.queue((steps or [])[:1])
            if first is not None:
                click.action, click.fix = first[1], first[2]
        click.judged = True
        if click.action is not None:
            return click.action
        return next((a for a in o.menu if panel.identity(a) == click.identity), None)

    def _chooser(self, o, waited):
        """For a combat card: the card he holds chosen with the client's own chooser (it stays
        the client's until its reveal) - or the one he chose in this very request while he was
        being asked (waited), also when he, or the client, has turned it up since."""
        pid, self.inbox.picked = self.inbox.own_pick or (waited and self.inbox.picked) or None, None
        if words.early_fits('card', o.kind) and pid is not None:
            pid = str(pid)
            cid = next((c for c, p in self.link.view.card_pid_map.items() if str(p) == pid), None)
            act = next((a for a in o.menu if a.action_type == 'combat_card_play'
                        and str((a.params or {}).get('card_id')) == str(cid)), None)
            if act is not None:     # (the client holds it chosen: at its reveal it is told to turn up THIS card)
                self.click, self.inbox.playing = Answer(o.id, panel.identity(act), 'chooser'), pid
                return act
            # refused: the client is told that the card is chosen no more (link.unpick); one he
            # has laid on the table is his no longer, and the comparison carries it back into
            # his hand (boardanswer.refuse says the one sentence)
            self.link.unpick()
            self.boards.refuse(pid)
        return None

    # ----------------------------------------------- the person: panel, words
    def decision_view(self):
        """The panel's `decision`: the dice the person owes, else his open
        decision (built once per decision), else None."""
        call = self.rand.pending()
        if owed_by(call, self.rand) and not self.link.refusal:
            return panel.roll_decision(self.rand.owed()[None], self.voice.stats['asks'], call.kw.get('ctx'))
        if self.open is None or self.click is not None or self.link.refusal:
            return None
        if self.built[0] != self.open.id:
            o = self.open
            built = panel.decision(o.id, o.kind, o.state, o.side, o.menu)
            self.built = (o.id, built)
            self.on_decision(built, o)
        return self.built[1]

    def panel_request(self, req):
        """A preview or a choice of the panel against the open decision -> its reply."""
        op, o = req.get('op'), self.open
        if self.link.refusal:            # nothing is played on this client any more
            return {'status': 'stale', 'message': self.link.refusal}
        if op not in ('preview', 'choose') or o is None or req.get('decision_id') != o.id:
            return {'status': 'stale', 'message': words.stale()}
        act = next((a for a in o.menu if panel.identity(a) == req.get('identity')), None)
        if act is None:
            return {'status': 'stale', 'message': words.stale()}
        label = words.option_label(o.state, o.side, act, self.human)
        if op == 'preview':
            return {'status': 'ok', 'preview': {
                'title': label, 'lines': [], 'lines_pretty': [words.you_did(label)],
                'changes': [], 'warnings': [], 'can_perform': True,
                'board_lines': self.boards.gesture_lines(o, act)}}
        if self.click is not None:
            return {'status': 'busy', 'message': words.choice_busy()}
        self.click = Answer(o.id, req['identity'], 'panel', token=req.get('token'), at=req.get('ts'))
        return {'status': 'ok', 'message': words.you_did(label),
                'perform': {'steps': [], 'pace_ms': 0}}

    def hear(self, text, o, unseen=False):
        """A word against the open decision `o`.  It commits only when it names
        exactly ONE option whose answer cannot be seen on the board; a word
        that fits none or several gets ONE sentence naming what can be typed;
        'nc', 'cc <card>', 'press' / 'stop' given early are kept for their
        question only - and for none of a battle he had not seen when he typed (unseen)."""
        say = named = None
        early = None if unseen else words.early_kind(text, self.position().state.pending_combat, self.human)
        if o is not None and o is self.open and (self.closing is not None or self.typed is not None):
            self.later.append(text)         # (a word of this poll waits for the board: this one is next)
        elif o is None:
            call = self.rand.pending()
            if early:
                self.early[early] = (text, _battle(self.position().state))
                say = words.word_kept(early)
            elif words.is_word(text):       # (it answers nothing; his own dice may be what is owed)
                say = words.waiting_for_roll(int(call.kw.get('need', 1)), call.kw.get(
                    'ctx')) if owed_by(call, self.rand) else words.nothing_asked()
            else:
                say = words.old_word(text)
        else:
            b = self.link.board      # the faces of his dice that lie in the play area (the staged die),
            mine = self.boards.has(o)       # or that he has put away already for this very decision
            family = layout.FP_DICE if layout.side_tag(o.side) == 'fp' else layout.SH_DICE
            staged = {b.face.get(p) for p in family if b.zone.get(p) in reconcile.DISPLAY or (
                p in mine and b.zone.get(p) in layout.boxes(p))} if hasattr(b, 'zone') else ()
            # (an attack typed with a word for its battle: the attack is heard now, the word waits)
            text, with_it, asks = typed.bundled(text, words.EARLY, o.menu, words.region_key)
            verdict, act = words.heard(text, o.state, o.side, o.menu, staged, o.kind)
            if with_it and verdict == 'pick' and words.by_word(act) and o is self.open:
                self.early[asks] = (with_it, (OPENS, o.id))
            if verdict == 'pick' and words.by_word(act) and words.closes(text) and o is self.open:
                self.closing = (o.id, panel.identity(act))   # ruling 8: the board is asked first
            elif verdict == 'none' and words.ends(text, o.kind) and o is self.open:
                self.closing = (o.id, None)      # it closes the decision as the board stands
            elif verdict == 'pick' and words.by_word(act):
                self.click = Answer(o.id, panel.identity(act), 'chat')
            elif verdict == 'pick':     # (unless the board cannot show that option at all: _board)
                say, named = words.not_by_word(), panel.identity(act)
            elif verdict == 'none' and early and not words.early_fits(early, o.kind):
                self.early[early] = (text, _battle(self.position().state))
                say = words.word_kept(early)
            elif verdict is not None:
                say = words.which_die(act) if verdict == 'dice' else \
                    words.say_choices(o.state, o.side, o.menu, verdict)
            else:
                say = words.old_word(text)
            if self.click is None and self.closing is None and o is self.open:
                # the word may name one of the choices the board cannot tell apart (the two
                # gates): the board is asked first, and `say` is said only when it does not
                self.typed, say = (o.id, text, say, named), None
        if say:
            self.voice.say(say)
