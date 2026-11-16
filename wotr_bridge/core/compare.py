"""The comparison of the client's board with the game, driven: what differs is rendered again; a
difference that stands after the client carried its lines out, or that no line shows, is said
once and tried again when anything of it changes or he asks ("back"); the
answer to "put the board back"; the record of each turn end.  It is the ONE mover of a piece the
person moved: what has ceased to be his (owned().back) goes where the game has it.

A Comparison imports no other part of the loop: the loop hands it its Voice, the client link (the
board of this poll), the position (a callable), and the three things it asks about the person:
what on the board is his now (owned), whether what is his may stay (settle), and which sentence a
pass owes him (explain).
"""
import time
from collections import namedtuple

from . import boardmodel as bm
from . import layout, reconcile, words

# "put the board back" as asked: by whom ('panel': its reply carries `token`; 'chat')
Asked = namedtuple('Asked', 'who token')


class Comparison(object):
    def __init__(self, voice, link, human, position, owned, settle, explain, at_turn_end):
        self.voice, self.link, self.human = voice, link, human
        self.position = position    # the game position now (a callable)
        self.owned = owned          # owned() -> what on the board is the person's now (whole, pieces, back)
        self.settle = settle        # settle(fate, settled): may what is his stay where it lies (asked by a whole pass)
        self.explain = explain      # explain(ops, lines, whole) says the sentence a pass owes him -> its number
        self.at_turn_end = at_turn_end      # called when a turn end was recorded
        self.recon = None           # [signature of the difference, times sent, said: it could not be shown]
        #                             (None again when the game is put back to an earlier position)
        self.request = None         # "put the board back" (Asked): compare now, whatever is held
        self.repair = False         # a client call failed: compare the board before asking again (the loop sets it,
        #                             and takes it when it makes that pass)
        self.held = False           # this pass moves only what has ceased to be the person's (carry)
        self.fresh = False          # started from a saved position: the board is built whole first
        self.turn_ends = []         # [{'turn', 'rows', 'stray'}] verified at each turn end
        self.verdict = (0, 0, 'equal', [])      # (poll, id of the state compared, verdict, rows)
        self.spent = set()          # the combat cards of rounds that have been fought: put away by the next pass
        #                             that looks at the whole board (played)
        self.round = set()          # the combat cards a battle still holds for its round
        self.left = None            # what a card of his own leaves to HIM (the loop opens it; _left, waits): [the number
        #                             of the sentence that told him - 0: not told yet -, what is left now, in words]

    def _left(self, rows):
        """WHAT A CARD OF HIS OWN LEAVES TO HIM (he applies his cards' effects himself: the loop opened `left`
        when a step of such a card was committed).  Of the rows of a pass that looks at the whole board,
        those of HIS OWN FIGURES are his to show: they are no rows of this pass (-> the others, which the
        pass carries as always: the AI's pieces, the markers the rules move, his die and his card put
        away).  Only the STATE of the board is read - which pieces he takes for it, and in what order,
        decides nothing; a piece he puts elsewhere stays his like any piece he moves.  The board shows
        all of it (whoever put it there), the game is over, or no piece can show what is left: nothing is
        left (None).  ("finish" and "back" end it too: the pass then carries everything.)"""
        if self.left is None or self.human is None:
            return rows
        tag = layout.side_tag(self.human)
        mine = [r for r in rows if r.side == tag and any(layout.family(p) in layout.FIGURES for p in r.have)]
        ops, unshown = reconcile.moves(mine)
        mine = [r for r in mine if not any(r is x for x in unshown)]
        what = bm.ops_words(bm.from_board(*self.link.board), [op for op in ops if any(op[2] in r.have for r in mine)])
        if not what or self.position().over is not None:
            self.left = None
            return rows
        self.left[1] = what
        return [r for r in rows if not any(r is x for x in mine)]

    def played(self, base, act, post):
        """The loop took a step of the game: what it did with the combat cards of a battle (reconcile.round_cards)."""
        self.round, spent = reconcile.round_cards(base, act, post, self.round)
        self.spent |= spent

    def waits(self):
        """Asked by the loop after a pass that looked at the whole board, before it takes the next step of
        the game - also a question of his own: must the game wait for his board?  He is told ONCE what
        his card leaves him, and the status keeps saying it as it stands."""
        if self.left is None:
            return False
        self.left[0] = self.left[0] or self.voice.say(words.left_to_you(self.left[1]))
        self.voice.how = words.left_to_you(self.left[1], status=True)
        return True

    def put_back(self, waits, who, token=None):
        """'Put the board back': compare and re-render now.  waits: a roll or a draw is under
        way, and the board follows it."""
        self.recon = None
        self.request = Asked(who, token)
        return words.put_back_asked(waits)

    def _answer(self, asked, text):
        """The answer to a "put the board back": in the log, in the panel's reply to that
        click, and - asked in chat - in chat."""
        self.voice.log_tail.append({'ts': time.time(), 'kind': 'info', 'text': text})
        for reply in self.voice.replies:
            if asked.token is not None and reply.get('token') == asked.token:
                reply['message'] = text
                self.voice.record('panel', text)
        if asked.who == 'chat':
            self.voice.say(text)

    def settled(self):
        """Is the game settled now (reconcile.settled: THE question)?"""
        return reconcile.settled(self.position().state)

    def carry(self):
        """The pass while the comparison is kept off the board (a decision of the person is open,
        or a step is in flight): it moves only what has ceased to be his.  -> the lines"""
        self.held = True
        try:
            return self.verify()
        finally:
            self.held = False

    def verify(self):
        """ONE pass: compare this poll's board with the engine and re-render what is missing
        (see carry: only what has ceased to be the person's).  A difference is sent once - again
        only when the client's receipts say it did not carry the lines out; when it stands all
        the same, or no line shows it: one sentence, and the game goes on.  Sets verdict[2]:
        equal / sent / gave_up (the pass of carry leaves it)."""
        state, board, held = self.position().state, self.link.board, self.held
        if not held and self.verdict[:2] == (self.voice.stats['polls'], id(state)):
            self.turn_end_check(state, [])         # this state was compared in this poll
            return []
        asked, self.request, settled = None if held else self.request, self.request if held else None, self.settled()
        own, fate = self.owned(), {}
        # a board built from a saved position holds no move under way: cards, dice and tiles are
        # placed too, whatever decision the position has open (until equal once).  What is the
        # person's stays where it lies (owned) - after "back" nothing he moved is his any more,
        # and with no decision of his open nothing a decision owns either
        known = (state, board, self.link.view.card_pid_map, settled or self.fresh, self.link.view.die_id_map or None)
        spent = self.spent = {c for c in self.spent if not known[3] and board.zone.get(
            str(known[2].get(c))) in reconcile.DISPLAY}         # (while one still lies in the play area)
        rows = reconcile.difference(*known, own.pieces if asked is None or own.whole else (), fate, spent)
        if not held and asked is None and self.human is not None:
            self.settle(fate, settled)
        back = self.owned().back
        if any(fate.get(p, 0) is not None and p in own.pieces for p in back):   # (kept a moment ago, his no longer)
            rows = reconcile.difference(*known, self.owned().pieces, None, spent)
        self.left = None if asked is not None else self.left    # ("back": the whole board as the game has it)
        rows = rows if held else self._left(rows)
        ops, unshown = reconcile.moves(rows, back)
        # ONE answer to where a piece goes that is his no longer: where the GAME has it - also a die, card or tile
        # this pass does not look at (asked as if the game were settled); only what is his AGAIN goes where HE had put it
        gone = {p: frm for p, frm in back.items() if p not in own.pieces and not reconcile.looks(p, known[3])}
        ops += [op for op in reconcile.moves(reconcile.difference(
            state, board, known[2], True, known[4], self.owned().pieces), gone)[0] if op[2] in gone] if gone else []
        ops += [('move', layout.side_of(p), p, board.zone[p], frm) for p, frm in sorted(back.items())
                if frm and p in own.pieces and board.zone.get(p) not in (None, frm) and not any(op[2] == p for op in ops)]
        if held:        # (rows nothing else of which is this pass's)
            ops, rows = [op for op in ops if op[2] in back], []
        lines = reconcile.render(ops, self.link.nick)
        said = self.explain(ops, bool(lines), not held) if self.human is not None else None
        if asked is not None:       # the answer to "put the board back": what the comparison found
            self._answer(asked, words.board_as_asked(bool(lines)) if own.whole else words.board_compared(
                len(rows) + sum(1 for op in ops if not any(op[2] in r.have for r in rows))))
        if held:
            if lines:
                self.voice.log('reconcile', turn=int(state.turn), tries=1, rows=[], lines=lines, said=said)
            return lines
        texts = [reconcile.row_text(r) for r in rows]
        verdict = 'equal'
        out = []
        if not rows:
            self.recon = None
            self.fresh = False
            out = lines                 # (what came back where the pass does not look)
        else:
            sig = tuple(sorted(texts))
            if self.recon is None or self.recon[0] != sig:
                self.recon = [sig, 0, False]
            if self.recon[2]:
                verdict = 'gave_up'                 # said once; the game goes on
            elif lines and (not self.recon[1] or self.link.receipts.missing):
                self.recon[1] += 1                  # (again: the client did not carry the lines out)
                verdict, out = 'sent', lines
            else:       # no line shows it, or the client carried the lines out and it stands
                self.recon[2] = True
                self.fresh = False
                verdict = 'gave_up'
                self.voice.stats['giveups'] += 1
                shown = [reconcile.row_text(r) for r in (unshown or rows)]
                self.voice.log('giveup', turn=int(state.turn), rows=shown)
                self.voice.note('could not show: %s' % '; '.join(shown)[:400])
                plain = [reconcile.row_words(r) for r in (unshown or rows)]
                out = [self.voice.chat(self.voice.ai_sides[0], words.could_not_show(plain))]
        if out and verdict != 'gave_up':
            self.voice.stats['reconciles'] += 1
            self.voice.log('reconcile', turn=int(state.turn), tries=self.recon[1] if self.recon else 1,
                           rows=texts, lines=lines, said=said)
            self.voice.how = words.putting_in_order()
        self.verdict = (self.voice.stats['polls'], id(state), verdict, texts)
        self.turn_end_check(state, out)
        return out

    def turn_end_check(self, state, out):
        """The record of the comparison at a turn end (once per turn)."""
        if self.position().phase != 'end' or out or self.verdict[2] not in ('equal', 'gave_up') \
                or (self.turn_ends and self.turn_ends[-1]['turn'] == int(state.turn)):
            return
        stray = sorted(p for p, z in self.link.board.zone.items() if z in reconcile.DISPLAY)
        self.turn_ends.append({'turn': int(state.turn), 'rows': self.verdict[3], 'stray': stray})
        self.voice.log('turn_end', turn=int(state.turn), rows=self.verdict[3], verdict=self.verdict[2],
                       stray=stray)
        self.at_turn_end()
