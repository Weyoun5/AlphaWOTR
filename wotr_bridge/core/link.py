"""The client as the loop sees it: the feed, the session, the nicks, the board of this poll, the
words the person typed, and the client's results that no step has used yet.

A ClientLink reads and remembers; it decides nothing about the game.  It imports no other part
of the loop: the loop hands it the funnel, its Voice, the engine state and the clock.
"""
import itertools
import json
import os
import re
import time
from collections import Counter, namedtuple

from . import client_protocol, layout, live_parser, reconcile, transport, words
from . import emitter as em
from . import menu as mn
from .card_identity import CardIdentity
from .client_view import ClientView
from .feed_reader import FeedReader, FeedRestart

AI_NICK = reconcile.NICK
FEED_WAIT_S = 1.5
NO_BOARD_S = 30.0           # seconds without an accepted board before the person is told
TALK = ('chat', 'system', 'blank', 'noop')      # feed lines that move no piece
TAG = re.compile(r'<~([\w.]+)~>')               # a tag of a line of the raw log
PICK = re.compile(r'<~Controls\.840~>(\d+)')    # 'chooses a card' (face down): the client's line
UNPICK = re.compile(r'^\$\S+?(?:<~Controls\.186~>| deselects)')    # ... and 'deselects': that nick's card is chosen no more
REVEALS = ('!CALL play_card', '!CALL reveal_chosen')    # the calls at which the client turns up EVERY chosen card

# one request of the client as read: its header, the new feed lines, the raw log, and the feed
# position its board was taken at.  Never kept.
Poll = namedtuple('Poll', 'header new raw want')


def _shape(line):
    """A line of the raw log as the client's feed sidecar states a command: (the nick it
    stands under, its tags in order, its numbers - the piece ids among them)."""
    gaps = TAG.split(line)[::2]
    head = gaps[0].split()
    nick = re.sub('[$%<>]', '', head[0]) if head and head[0][:1] in ('$', '%', '<') else None
    return nick, TAG.findall(line), [t for gap in [' '.join(head[nick is not None:])] + gaps[1:]
                                     for t in gap.split() if t.isdigit()]


class Receipts(object):
    """What the client itself states about the replies of this server.  Its receipts
    (ai_inject_ack.jsonl): one per line and call, written right after it was carried out and
    before the client asks again - which feed paragraphs it wrote, or why it was held or
    refused.  And the feed's sidecar (ai_feed.txt.meta.jsonl): for a feed line the command that
    wrote it, by nick and tags - the raw log of the request holds the commands in that order,
    with their pieces.  So of every command the bridge knows whether a line or call of its own gave it."""

    def __init__(self, client_dir, feed_path, voice):
        self.acks = FeedReader(os.path.join(client_dir, 'ai_inject_ack.jsonl'))
        self.meta = FeedReader(feed_path + '.meta.jsonl')
        self.voice = voice
        self.mine = set()           # the feed paragraphs (by number) a line or call of mine wrote
        self.sent = None            # the reply that is on its way: (the request it answers, its lines and calls)
        self.missing = False        # a line of the last reply has no receipt, or was held: the client did NOT carry it out
        self.cmds = []              # commands joined and not yet taken: [(feed line, line of the raw log)]
        self.done = 0               # feed lines whose record is joined; and where the raw log stands
        self.raw_at = 0

    def read(self, session):
        """The receipts written since the last request.  The reply that was on its way is on
        its way no longer: each line of it has its receipt - carried out, held or refused
        with its reason.  While receipts of that reply are unread it is NOT KNOWN who moved what:
        the board waits for them, as it waits for the feed (the client writes each before it
        asks again; a file held for the moment is read a moment later).  Where one is MISSING
        after that (the client dropped the reply) or says the line was held, the line was not
        carried out: that is noted; a call is asked again, a difference sent again."""
        got, new, end = 0, self.acks.poll(), time.time() + FEED_WAIT_S
        while self.sent is not None and self.acks.lines and len(new) < self.sent[1] and time.time() < end:
            time.sleep(0.03)
            new += self.acks.poll()
        for text in new:
            try:
                rec = json.loads(text)
            except ValueError:
                self.voice.note('a receipt of the client cannot be read: %s' % text[:80])
                continue
            if session is not None and rec.get('session') not in (None, session):
                continue                # (the file is kept across the client's sessions)
            got += self.sent is not None and self.sent[0] in (None, rec.get('req')) and rec.get('kind') != 'held'
            self.mine.update(range(int(rec.get('paragraphs_before', 0)) + 1, int(rec.get('paragraphs_after', 0)) + 1))
            if rec.get('ok') is False or rec.get('real_from'):      # (real_from: the piece did not lie where my line
                self.voice.note('the client %s (%s): %s' % (          # took it for - who had moved it, it does not say)
                    'did not carry out a line' if rec.get('ok') is False else 'took a piece of a line from where it lay',
                    rec.get('error') or rec.get('real_from') or rec.get('kind'), str(rec.get('raw'))[:80]))
        if self.sent is not None and got < self.sent[1] and self.acks.lines:
            self.missing = True         # (until one board has been read with it: ClientLink.replied)
            self.voice.note('the client carried out %d of the %d line(s) of the reply to request %s'
                            % (got, self.sent[1], self.sent[0]))
        self.sent = None

    def join(self, raw, upto):
        """The client's commands up to the feed line `upto`, joined with the raw log: -> the new
        ones, [(feed line, line of the raw log, the pieces the sidecar states for it or None, the
        client's time of it)].
        A record of the sidecar with `raw_new` is the first paragraph of a command; it is the
        next line of the raw log with that nick, those tags and - where the sidecar may state
        them - those pieces."""
        out = []
        while self.done < min(upto, len(self.meta.lines)):
            i, self.done = self.done, self.done + 1
            try:
                rec = json.loads(self.meta.lines[i])
            except ValueError:          # (loud, and the game goes on from the board: the command is nobody's)
                self.voice.note('the client\'s record of feed line %d cannot be read: %s' % (i + 1, self.meta.lines[i][:80]))
                continue
            if not rec.get('raw_new'):
                continue
            want, pids = (rec.get('nick'), list(rec.get('tags') or ())), rec.get('pids')
            pids = [str(p) for p in pids] if isinstance(pids, list) else None
            k = next((k for k in range(self.raw_at, len(raw)) for shape in [_shape(raw[k])] if shape[:2] == want
                      and (pids is None or all(p in shape[2] for p in pids))), None)
            if k is None:               # (never silently: its pieces are then nobody's command's)
                self.voice.note('feed line %d states a command the raw log does not hold' % (i + 1))
                continue
            self.raw_at = k + 1
            out.append((i, k, pids, rec.get('ts')))
        self.cmds += out
        return out

    def take(self, upto):
        """The commands between the board before and the board taken at the feed line `upto`."""
        out = [c for c in self.cmds if c[0] < upto]
        self.cmds = self.cmds[len(out):]
        return out


class Inbox(object):
    """Results of the client that no step has used yet."""

    def __init__(self):
        self.rolls = []             # action-roll results not yet adopted: [letters]
        self.own_pick = None        # the card piece the client holds chosen under the person's nick
        self.picked = None          # ... and the one he chose in this very request (also when it is turned up since)
        self.playing = None         # ... and the one the game took as his answer, until the client has turned it up
        self.random_discard = False     # a random discard the board has yet to name
        self.drawn = {}             # {a deck's box: the piece of the drawn card the funnel keeps for it}
        self.unused = []            # results the game does not use, until said: the piece of each, or None


class ClientLink(object):
    def __init__(self, client_dir, ai_sides, human, rand, voice, inbox, state, now):
        self.ai_sides, self.human, self.rand, self.voice = ai_sides, human, rand, voice
        self.inbox = inbox
        self.state, self.now = state, now       # callables: the engine state now, the clock
        self.human_nick = None      # learned from the client's own lines
        self.dump_path = os.path.join(client_dir, 'ai_client_board.txt')
        self.feed = FeedReader(os.path.join(client_dir, 'ai_feed.txt'))
        self.cards = CardIdentity(None if human is None else ai_sides[0].name, password='ai')
        self.view = ClientView(state)
        self.view.die_id_map = {}
        self.board = None           # reconcile.Board of this poll, or None
        self.bound = {}             # shared Shadow stronghold boxes of this poll
        self.attested = False       # the client's feed stated which hand it keeps from the AI
        self.receipts = Receipts(client_dir, self.feed.path, voice)     # what the client did with my replies
        self.owed = []              # lines owed to the client ahead of everything else of the next reply
        self.seen = None            # the board of the poll before (reconcile.Board)
        self.words = []             # his chat lines of this poll, heard once its board is read
        self.ai_lines = 0           # feed lines under the AI's nick read so far: where the feed stands for it
        self.battle_at = None       # ... and where it stood when a step of the AI opened the battle that is open
        self.feed_off = None        # feed lines the client's count does not hold (written twice); None: not known
        #                             yet - no board's count was the feed's, line for line (that proves 0)
        self.unread = None          # since when no board was accepted, and what was said
        self.refusal = None         # why this client cannot be played on (refuse)
        self.session = None

    def nick(self, side):
        """The nick a line of `side` (a Side or 'fp' / 'sa') is sent under."""
        tag = side if side in ('fp', 'sa') else layout.side_tag(side)
        if self.human is not None and tag == layout.side_tag(self.human):
            return self.human_nick or 'Player'
        return AI_NICK[tag]

    def refuse(self, why):
        """This client cannot be played on, for good: the first reason is kept (nothing more is
        played or offered), said in chat once, when it arises, and stays the status."""
        first, self.refusal = self.refusal is None, self.refusal or why
        self.voice.how = self.refusal
        return [self.voice.chat(self.ai_sides[0], self.refusal)] if first else []

    def read(self, request):
        """One request of the client -> Poll.  Another client session: FeedRestart."""
        header, raw = client_protocol.split_seq_header(getattr(request, 'log_lines', ()) or ())
        if header is not None and header.get('session'):
            if self.session is None:
                self.session = header['session']
            elif header['session'] != self.session:
                raise FeedRestart('client session %s replaced %s' % (header['session'], self.session))
        new, want = self.read_feed(header)
        self.receipts.read(self.session)
        return Poll(header, new, raw, want)

    def unblinded(self, new):
        """Why this client may not be played on (the person's hand would be readable), or None."""
        ai_name = None if self.human is None else self.ai_sides[0].name
        why = self.refusal
        for line in new:
            why = why or words.blinding_violation(line, ai_name)
            self.attested = self.attested or line.startswith('#BLIND ')
        if self.human is not None and self.feed.lines and not self.attested:
            why = why or words.blinding_violation('#BLIND aiside=none hands=none', ai_name)
        return why

    def heard(self):
        """A board was accepted: his chat lines since the last one, to be heard on it, each with
        `unseen` - he typed it before the battle the AI's step opened was on his screen - and
        its feed line."""
        self.unread = None
        said, self.words = self.words, []
        return said

    def battle(self, shown):
        """A step opened a battle.  shown: it was the AI's and its lines are on their way to
        his screen - a word he types before the feed holds one of them is typed unseen."""
        self.battle_at = self.ai_lines if shown else None

    def replied(self, lines):
        """-> the reply as it goes out: what is owed to the client ahead of everything else
        first (owed).  When the reply makes the client reveal, that is also what it must hold
        chosen under his nick at that moment: the card the game holds as his played card of
        the open battle, while it lies face down in his hand - or the card it has just taken
        as his answer, while that step still waits for its dice (playing) - else nothing
        (unpick).  Its lines and calls are on their way: when the client asks again each has
        been carried out, held or refused - its receipt says which (Receipts.read)."""
        if self.human is not None and self.board is not None and any(line.startswith(REVEALS) for line in lines):
            battle = self.state().pending_combat or {}
            hand = reconcile.FP_HAND if layout.side_tag(self.human) == 'fp' else reconcile.SA_HAND
            self.unpick(next((str(p) for c, p in self.view.card_pid_map.items() if c in battle.values()
                              and self.board.zone.get(str(p)) == hand), self.inbox.playing))
        # (never a line for a piece that already lies where the line puts it, whoever made the line: menu.idle)
        lines, self.owed = [x for x in self.owed + list(lines) if self.board is None or not mn.idle(
            self.board, x, self.view.dynamic_types)], []
        n = sum(1 for line in lines if line.strip() and not line.startswith(('PASS', '#')))
        self.receipts.sent = (self.voice.req, n) if n else None
        self.receipts.missing = self.receipts.missing and self.board is None
        return lines

    def mine(self, i):
        """Did a line or call of this server write the feed line `i` (its receipt says so: the
        board of a request is read when the receipts of the reply before it are - Receipts.read)?"""
        return i + 1 - client_protocol.FEED_HEADER_LINES - (self.feed_off or 0) in self.receipts.mine

    def track(self, raw, want, since=None):
        """-> ({piece: where it lay} of the pieces the PERSON moved since the last board: those no
        line or call of this server moved last, that lie elsewhere than he took them from - or
        that a command of his which names them TURNED where they lay (a side, a die's face), or
        that two or more commands of his name and leave where they lay (taken away and brought
        back: he handled them);
        those of them he moved later than `since`, the client's own time, in seconds, of the
        click that is being carried out: the client dates each of his commands; {piece: the
        feed line of his last command that names it} - THE ORDER of his moves and his words).  The
        client states it: the last command since that board that names a piece stands in the
        feed inside one of my receipts, or not.  A command's pieces are those the sidecar
        states - or, where it hides them, the numbers of its raw line that are pieces of the
        board (never a hash or a value).  WHERE he took a piece from: a command that names two
        places states it (from, to); any other took it from where the commands before it had
        put it, else from where it lay on the last board - the one place a command names is
        not taken for it (the client's own lines name the place they PUT a piece too).  A piece
        no command names (the client's Undo) moved with a command of his when one stands there."""
        seen, self.seen, now = self.seen, self.board, self.board.zone
        cmds = self.receipts.take(len(self.feed.lines) if want is None else want)
        if seen is None or self.human is None:
            return {}, set(), {}
        # {piece: its last command was mine} (None: every command was); where he took it; where it lies
        by, was, at, when = {None: True}, dict(seen.zone), dict(seen.zone), {}      # now; when he last named it
        named = Counter()           # how many commands of his name it (two that leave it where it lay: away and back)
        for i, k, pids, ts in cmds:
            if live_parser.parse(self.feed.lines[i]).get('op') not in TALK:
                nick, tags, numbers = _shape(raw[k])
                # (a command under an AI seat's nick is never his: a line of mine, or the result of a
                # call of mine that the client wrote after the call's receipt - a draw, a discard)
                mine = self.mine(i) or nick in AI_NICK.values()
                zones = [t for t in tags if t.startswith('Game.') or t in (reconcile.FP_HAND, reconcile.SA_HAND)]
                for p in pids if pids is not None else [n for n in numbers if n in seen.zone or n in now]:
                    if not mine and by.get(p, True):
                        was[p] = zones[0] if len(zones) == 2 else at.get(p)
                    by[p], at[p], when[p] = mine, zones[1] if len(zones) == 2 else now.get(p), (ts, i)
                    named[p] += not mine
                by[None] = by[None] and mine
        moved = {p: was.get(p) for p in set(was) | set(now) if not by.get(p, by[None]) and (
            was.get(p) != now.get(p) or (by.get(p) is False and (named[p] > 1 or any(
                old.get(p) != new.get(p) for old, new in zip(seen[1:], self.board[1:])))))}
        return moved, {p for p in moved if isinstance(since, (int, float)) and p in when and isinstance(
            when[p][0], (int, float)) and when[p][0] > since * 1000}, {p: when[p][1] for p in moved if p in when}

    def unpick(self, played=None):
        """THE one place where the bridge sets what the client holds chosen under the person's
        nick.  A card he chose is taken from him unplayed - it was refused, his card was
        answered otherwise or taken back, its battle is over (or one opened after it), or the
        client is about to reveal: the client is told by its own line, under the nick his pick
        stood under, ahead of every line that could turn the card up.  played: the card the game
        holds as his played card, face down in his hand - the client is told to hold THAT chosen,
        whatever he chose since.  -> the piece taken from him, or None."""
        pid, self.inbox.own_pick, self.inbox.playing = self.inbox.own_pick or self.inbox.playing, None, None
        nick = {'nick': self.nick(self.human)}
        if played is not None:
            self.owed.append(em.render('combat_card_pick', dict(nick, piece=played), local=False))
        elif pid is not None:
            self.owed.append(em.render('combat_card_unpick', nick, local=False))
        return pid

    # ------------------------------------------------------- feed and board
    def read_feed(self, header):
        """(new feed lines, the feed position the request's dump was taken at or None); waits
        a moment for the feed file - and, with a person, its sidecar - to reach that position."""
        new, want = [], None
        if header is not None and header.get('feed_paragraphs', -1) >= 0:
            want = header['feed_paragraphs'] + client_protocol.FEED_HEADER_LINES + (self.feed_off or 0)
        deadline = time.time() + FEED_WAIT_S
        while True:
            new.extend(self.feed.poll())
            behind = want is not None and len(self.feed.lines) < want
            if self.human is not None:
                self.receipts.meta.poll()
                behind = behind or (want is not None and len(self.receipts.meta.lines) < want)
            if not behind or time.time() >= deadline:
                return new, want
            time.sleep(0.03)

    def no_board(self, header, want):
        """No board of this poll could be tied to the feed.  Never for ever, never in
        silence: (a) no board of this session was tied to the feed line for line (feed_off None;
        read_board: that proves there is no offset), the feed holds more lines than the client counts, and
        EXACTLY that many of its paragraphs from before the first
        command that moves a piece are written twice (the client's boot paragraphs: the one thing
        its count leaves out): that is the session's offset, counted in from now on - a fact of
        the feed's head, which no later line can change; (b) nothing accepted for NO_BOARD_S
        seconds: ONE sentence, in chat once and in the status until a board is accepted again."""
        now, lines = self.now(), self.feed.lines
        self.voice.how = words.waiting_for_board()
        u = self.unread = self.unread or {'since': now, 'said': False}
        extra = len(lines) - want if want is not None and self.feed_off is None else 0
        boot = Counter(itertools.takewhile(lambda l: not l.startswith('$') or live_parser.parse(l).get('op') in TALK,
                                           lines[client_protocol.FEED_HEADER_LINES:]))
        if extra > 0 and sum(n - 1 for n in boot.values()) == extra:
            self.feed_off = extra
            self.voice.note('the feed holds %d line(s) more than the client counts (boot paragraphs '
                            'written twice): taken as the offset of this session' % extra)
        if now - u['since'] >= NO_BOARD_S:
            self.voice.how = words.no_board()
            if not u['said']:
                u['said'] = True
                return [self.voice.chat(self.ai_sides[0], self.voice.how)]
        return []

    def read_board(self, header, want):
        """Bind this poll's board: the dump whose header is the request's, with
        nothing but talk in the feed after it."""
        self.board = None
        text = board = None
        for _ in range(3):              # one read, whole, into memory; again when torn
            text = transport.read_shared(self.dump_path)
            board = reconcile.parse_board(text)
            if board is not None:
                break
            time.sleep(0.02)
        if board is None:
            return False
        dump_header = client_protocol.parse_seq_header(text.split('\n', 1)[0])
        if header is not None and dump_header is not None \
                and dump_header.get('dump') != header.get('dump'):
            return False
        if want is not None:        # (with a person: the sidecar states who gave each of those lines)
            if len(self.feed.lines) < want or (self.human is not None and len(self.receipts.meta.lines) < want):
                return False
            for line in self.feed.lines[want:]:
                if live_parser.parse(line).get('op') not in TALK:
                    return False
            self.feed_off = 0 if self.feed_off is None and want == len(self.feed.lines) else self.feed_off
        state = self.state()
        self.view.piece_region, self.bound = layout.bind_shared_boxes(state, board.zone)
        self.board = board
        self.view.shadow_stronghold_regions = dict(self.bound)
        self.view.piece_mark = dict(board.mark)     # a two-sided piece is turned to a SIDE
        self.bind(state)                # (until the loop knows what the person moved on this board)
        return True

    def bind(self, state, last=()):
        """THE binding of the die pieces of this poll's board to the game's dice (state: the game
        as it is now), kept in the view: the renderer, the comparison and what is said name the
        same die.  last: the dice that are the person's - bound after the others, so that no
        other die is ever moved in the stead of one he moved."""
        self.view.die_id_map = reconcile.bind_dice(state, self.board, last)

    def event(self, rec, at=None):
        """Feed events: dice, tile and companion draws, the random discard,
        the person's chat words (at: the feed line).  Piece lines are never interpreted."""
        op = rec.get('op')
        who = rec.get('nick') if op == 'chat' else rec.get('player')
        if self.human is not None and who and who not in AI_NICK.values() and who != 'Player' \
                and op in ('chat', 'roll_num', 'draw', 'noop'):
            self.human_nick = who
        self.ai_lines += who in AI_NICK.values()
        if op == 'roll_num':
            values = rec.get('ints') or []
            tag = None
            if self.human is not None:
                # the client rolls the AI's own call under the AI's nick and nothing else:
                # every other roll is the person's, whatever its length and whenever it comes
                tag = (self.ai_sides[0] if who in AI_NICK.values() else self.human).name
            self.bank('d6', values, tag)
        elif op == 'roll_dice':
            self.inbox.rolls.append([str(f).upper() for f in rec.get('faces') or ()])
        elif op == 'draw':
            import rendered_replay as rr
            what = str(rec.get('what', '') or '').strip()
            piece = rr.parse_piece(what)
            if piece is not None and piece[0] == 'tile':
                self.bank('tile', what)
            elif piece is not None and piece[0] == 'named':
                self.bank('companion', (piece[1], what))
        elif op == 'random_discard':
            self.inbox.random_discard = True
        elif op == 'chat' and self.human is not None and who not in AI_NICK.values():
            # (heard on the board of its poll; unseen: no line of the AI stands in the feed
            # between the step of the AI that opened the battle and this line of his)
            self.words.append((str(rec.get('body') or ''), self.battle_at == self.ai_lines, at))

    def bank(self, kind, payload, side=None):
        """Hand ONE client result to the funnel (side: who rolled the dice).  What it hands
        back - dice beyond the request, a second result nobody asked for - is unused (said
        with this reply: ignored); and a roll of the person that it keeps for his next roll
        right after the one he was asked for is said (he rolled twice: no request will show
        for those dice)."""
        piece = str(payload[3]) if kind == 'card' and len(payload) > 3 else None
        # (a drawn card is ONE result by what the board shows: one drawn while the card the funnel
        # keeps for that deck lies in its deck again on the board of this very request - the client
        # wrote every line of it before that board: he undid his draw and drew anew - IS the draw;
        # and the kept card drawn again is the result the funnel has)
        deck, zone = layout.home(piece) if piece is not None else None, self.board.zone if self.board is not None else {}
        kept = self.inbox.drawn.get(deck)
        got = self.rand.feed(kind, payload, side=side, undone=kept if kept != piece and zone.get(kept) == deck else None)
        self.inbox.unused += [p for p in [piece] * len(got['dropped']) if p is None or p != kept]
        if piece is not None and not got['dropped']:
            self.inbox.drawn[deck] = piece
        if got['second'] and self.human is not None and side == self.human.name:
            self.voice.say(words.second_roll(len(got['kept'])))

    def ignored(self, results):
        """Results of the client that the game does not use (the piece of each, or None): ONE
        sentence -> ({the piece of such a result: where the client took it from (a card he
        drew too many: its deck)}, the number of the sentence).  Those pieces are nobody's: the
        comparison carries them back with this very reply (boardanswer.end)."""
        zone = self.board.zone if self.board is not None else {}
        back = {p: layout.home(p) for p in sorted(set(results) - {None})
                if p in zone and layout.home(p) not in (None, zone[p])}
        return back, self.voice.say_about(words.unused_results(len(results)), pieces=[
            {'pid': p, 'lay': zone[p], 'was': home, 'to': home, 'by': 'call', 'seen': self.voice.req}
            for p, home in back.items()])

    def raw_events(self, raw, want):
        """The things only the raw log states: the identity of a drawn card (the client's
        commitment), and the card the person holds chosen with the client's own chooser - a
        fact of the CLIENT: from his own "chooses" line (no line or call of this server gave
        it; his nick stands on it) until a later command names that card (a reveal, a move,
        a kill: what he PLAYED is no pick) or his nick un-chooses.  want: the feed line of
        the request's board."""
        drawn = self.cards.harvest(raw)
        for d in drawn:     # (the reader has moved on: what it could name is banked, whatever
            if d['cid'] is not None:        # it could not name)
                self.bank('card', (d['side'], d['deck'], d['cid'], d['pid']))
                self.view.card_pid_map[d['cid']] = d['pid']
        lost = [str(d['pid']) for d in drawn if d['cid'] is None]
        if lost:
            raise RuntimeError('drawn card piece %s cannot be identified' % ', '.join(lost))
        self.inbox.picked = None
        for i, k, _pids, _ts in self.receipts.join(raw, len(self.feed.lines) if want is None else want) \
                if self.human is not None else ():
            nick, _tags, numbers = _shape(raw[k])
            pick = PICK.search(raw[k])
            if pick and not self.mine(i):
                self.inbox.own_pick = self.inbox.picked = pick.group(1)
                self.human_nick = nick
            elif self.inbox.own_pick in numbers or (nick == self.human_nick and (pick or UNPICK.search(raw[k]))):
                self.inbox.own_pick = None      # (... or a line of mine set what his nick holds)
            self.inbox.playing = None if self.inbox.playing in numbers and not pick else self.inbox.playing

    def feed_random_discard(self):
        """The client discarded a random Free Peoples Character card: the
        board names it (the hand card that left the hand)."""
        import game_state as gs
        if not self.inbox.random_discard or self.board is None:
            return
        hand = self.state().hands[gs.Side.FREE_PEOPLES].hand
        gone = [cid for cid in hand
                if self.board.zone.get(self.view.card_pid_map.get(cid)) != reconcile.FP_HAND]
        if gone:
            self.inbox.random_discard = False
            self.bank('card_pick', gone)

    def drop_early(self):
        """Dice the person rolled with nothing owed wait for the card he plays
        next.  They are dropped when the AI acts or when a battle or a Hunt opens - WITHOUT a
        word (that can be many requests after the roll: a sentence about a roll comes with the
        roll, or not at all; the log notes it); dice rolled while a battle or Hunt is open
        are that battle's (an early roll is accepted)."""
        extra = self.rand.drop_early(self.human) if self.human is not None else 0
        if extra:
            self.voice.stats['dropped_dice'] += extra
            self.voice.note('%d dice the person rolled unasked answer nothing any more' % extra)
