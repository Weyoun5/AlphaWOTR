"""The voice of the loop: the one exit of everything said in chat, noted, logged and published.

A Voice knows the game only through what the loop hands it: the position (a callable) and the
nick a line of a side is sent under (a callable).  It imports no other part of the loop.
"""
import json
import os
import time
import traceback

from . import emitter as em
from . import panel, words

END_MARKER = '#END'


class Voice(object):
    def __init__(self, client_dir, ai_sides, human, position, nick, decision):
        self.client_dir, self.ai_sides, self.human = client_dir, ai_sides, human
        self.position = position    # the game position now (a callable)
        self.nick = nick            # the nick a line of a side is sent under (a callable)
        self.decision = decision    # the open decision of the person, or None (a callable)
        self.req = None             # the number of the client request being handled (in every log record)
        self.outbox = []            # sentences owed (chat answers, what happened), sent with the reply
        self.lever_said = None      # the answer to the last Settings lever (the panel's result card)
        self.errors, self.about = 0, {}     # about: what the sentence `say_about` hands to `chat` is about
        self.stats = {'polls': 0, 'steps': 0, 'reconciles': 0, 'giveups': 0, 'exceptions': 0,
                      'reasks': 0, 'holds': 0, 'client_errors': 0, 'clicks': 0, 'takebacks': 0,
                      'blocked_sentences': 0, 'unplain': 0, 'dropped_dice': 0, 'asks': 0,
                      'silent_calls': 0, 'board_answers': 0, 'board_changes': 0,
                      'board_said': 0, 'match_errors': 0, 'forced': 0, 'said': 0}
        self.notes, self.replies, self.log_tail, self.said = [], [], [], []
        self.ai_last = None
        self.how = words.starting()
        self.write_status = None    # FileBridge.write_status, set by build()
        self.over_said = False
        self.error_file = 'core-error.txt'
        self.sent = [0, '', 0.0]    # the panel view written last: its number, its text, when

    def record(self, channel, text, reason=None, since=None, pieces=()):
        """THE record of one sentence handed to the person (the log's `said`): where he reads it
        ('chat' / 'panel'), its name in words.py, the decision of his it is about, the reason it
        states, and the pieces it is about - each {pid, lay: where it lay on this poll's board,
        was: where its mover took it from, to: where this reply carries it (None: it stays), by:
        who moved it, seen: in which request}; since / steps: the steps committed then, and now."""
        self.stats['said'] += 1
        o, steps = self.decision(), self.stats['steps']
        self.log('said', n=self.stats['said'], channel=channel, text=str(text), reason=reason,
                 id=text.id if isinstance(text, words.Said) else None, steps=steps,
                 decision=o.id if o is not None else None, pieces=list(pieces),
                 since=steps if since is None else since)
        return self.stats['said']

    def chat(self, side, text):
        """THE one exit of everything said in chat (an AI seat never names a card of the person's hand)."""
        import game_state as gs
        side = gs.Side.FREE_PEOPLES if side in ('fp', gs.Side.FREE_PEOPLES) \
            else gs.Side.SHADOW
        name = text.id if isinstance(text, words.Said) else None
        if not words.is_plain(text):
            self.stats['unplain'] += 1
            self.note('sentence put into plain words: %r' % text[:120])
            text = words.plain(text)
        if self.human is not None and side is not self.human:
            low = words.chat_card_name(text)
            for cid in self.position().state.hands[self.human].hand:
                if words.chat_card_name(words.card_title(cid)) in low:
                    self.stats['blocked_sentences'] += 1
                    self.note('a sentence naming a card of the person was not said')
                    text, name = words.chose(), 'chose'
        text = words.Said(text)
        text.id = name
        self.said.append((side.name, text))
        del self.said[:-200]
        about, self.about = self.about, {}
        self.record('chat', text, **about)
        return em.render('chat', {'nick': self.nick(side), 'msg': text}, local=False)

    def say(self, text):
        """One sentence of the AI seat, owed with the next reply -> its number in the game's log."""
        self.outbox.append(self.chat(self.ai_sides[0], text))
        return self.stats['said']

    def say_about(self, text, **about):
        """`say` with what the sentence is about (see record).  (say and chat stay plain: recorders wrap them.)"""
        self.about = about
        return self.say(text)

    def note(self, text):
        text = words.no_cards(text)     # (the console log lies in the game folder)
        self.notes.append(text)
        del self.notes[:-50]
        print('  [note] %s' % text, flush=True)

    def log(self, kind, **fields):
        fields.update(kind=kind, ts=round(time.time(), 3), req=self.req)
        try:
            with open(os.path.join(self.client_dir, 'core_log.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(fields, default=str) + '\n')
        except OSError:
            pass

    def told(self, side, label, click):
        """After a step: the status sentence, the panel's log and last-action
        card, the answer to the click."""
        human = side is self.human
        self.how = words.you_did(label) if human else words.ai_did(side, label)
        self.log_tail.append({'ts': time.time(), 'kind': 'you' if human else 'ai', 'text': self.how})
        del self.log_tail[:-40]
        if human:       # (an answer given for him is none of his: counted apart)
            self.stats['forced' if click is not None and click.source in ('only', 'stands') else 'clicks'] += 1
            self.result(click, 'committed', words.done(label))
        else:
            self.ai_last = {'side': side.name, 'title': self.how, 'text': '', 'changes': [],
                            'notes': [], 'ts': time.time()}

    def result(self, click, result, text):
        """The verdict of a panel click, written into its reply."""
        for reply in self.replies:
            if click is not None and click.token and reply.get('token') == click.token:
                reply.update(result=result, result_text=text)
                self.record('panel', text)

    def narrate(self, pre, post, tiles, act, dud):
        """One plain line per fact a player needs to follow the game, each
        built from the state change of the committed step (words.py)."""
        import game_state as gs
        FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
        card = None
        if act is not None and act.action_type in ('play_event', 'card_option', 'combat_card_play'):
            card = words.card_title(act.params.get('card_id'))
        pile = [t for s in (pre, post) for name in ('hunt_pool', 'hunt_discard', 'hunt_removed')
                for t in getattr(s, name, None) or ()]
        # what a tile did is the engine's: the damage of the Hunt it opened, and a reveal it
        # made or staged for good (2: nothing can prevent it any more)
        hunt = post.pending_hunt if pre.pending_hunt is None and len(tiles) == 1 else None
        shown = pre.fellowship.hidden and (not post.fellowship.hidden or getattr(
            post, '_hunt_reveal_pending', 0) == 2)
        said = [words.tile_drawn(face, pile, (hunt or {}).get('damage'), shown) for face in tiles]
        said.append(words.round_hits(pre, post, self.human))
        said.append(words.card_no_effect(dud) if dud else None)
        said += [words.dice_changed(pre, post, s, card, self.human) for s in (FP, SH)]
        said += words.sieges_ended(pre, post)
        said += [words.enters_mordor(pre, post), words.gollum_guide(pre, post)]
        said += words.table_cards_gone(pre, post)
        self.outbox += [self.chat(self.ai_sides[0], text) for text in said if text]

    def save_error(self, error, position):
        """Principle 8: the exception and the position it happened in.  The person is sent to
        these files: no card is named, no hand or deck order saved (WOTR_CORE_DEBUG_CARDS=1 does).
        position(blind=): the game position as a saved-position file holds it."""
        import pickle
        text = words.no_cards(''.join(
            traceback.format_exception(type(error), error, error.__traceback__)))
        print('! poll error (%d in a row): %s' % (self.errors, text), flush=True)
        if self.errors > 3:
            return self.error_file
        pos = self.position()
        # hyphens: the name is said in chat, where no underscore word may appear
        stem = os.path.join(self.client_dir, 'core-error-%s-T%d-%s' % (
            time.strftime('%Y%m%d-%H%M%S'), int(pos.state.turn), pos.phase))
        self.error_file = os.path.basename(stem) + '.txt'
        try:
            with open(stem + '.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            with open(stem + '.pkl', 'wb') as f:     # a saved position: --position reads it
                pickle.dump(position(blind=self.human is not None and os.environ.get(
                    'WOTR_CORE_DEBUG_CARDS') != '1'), f)
        except Exception as e:
            print('! state not saved: %r' % (e,), flush=True)
        return self.error_file

    def over_lines(self):
        """The closing lines, once; after them nothing more is said.  The end
        marker is the client's AI vs AI watch loop's (it stops on it); a
        client with a person at the board has no such verb."""
        import game_state as gs
        pos = self.position()
        result = pos.over
        end = [] if self.human is not None else [
            '%s %s %s' % (END_MARKER, str(result.get('victory') or 'none'),
                          str(result.get('victory_type') or 'unknown'))]
        self.how = words.game_over(result)
        if self.over_said:
            return end
        self.over_said = True
        self.log('game_over', result={k: str(v) for k, v in result.items()},
                 turn=int(pos.state.turn))
        FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
        loser = SH if str(result.get('victory', '')).startswith('free') else FP
        lines = [self.chat(self.ai_sides[0], self.how)]
        if loser in self.ai_sides:
            lines.append(self.chat(loser, 'gg'))
        return lines + end

    def status(self, sims):
        """Extra fields of ai_bridge_status.json (transport.status_extra)."""
        pos = self.position()
        st = pos.state
        return {'mode': 'ai_vs_ai' if self.human is None else 'sided',
                'turn': int(st.turn), 'phase': pos.phase,
                'acting': getattr(pos.active, 'name', None), 'sims': sims,
                'note': '', 'game_over': ({k: str(v) for k, v in pos.over.items()}
                                          if pos.over else None),
                'core': dict(self.stats)}

    def publish(self, facts):
        """The panel's view of now (facts: a callable giving panel.Facts)."""
        try:
            panel.publish(facts, self.client_dir, self.sent)
        except Exception as e:           # the view never stops the game
            self.note('panel view not written: %r' % (e,))
