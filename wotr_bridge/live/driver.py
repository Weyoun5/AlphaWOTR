"""The thin sided driver: one tick loop over the rendered feed.

COMPOSITION (REBUILD_PLAN architecture):

  FeedReader.poll() -> live_parser.parse -> Segmenter.push
     'phase' records   -> turn-start protocol (below)
     windows           -> gesture_match (menu-sized, vs engine menus)
     'chat'            -> chat.classify_command / match_choice
     'undo'/'redo'     -> history pop (engine snapshots)
  engine (SOLE authority): ai.describe_pending / get_legal_actions /
     apply_action on a CLONE -> COMMIT by adopting the post state
     (design rule 2: the transactional tick — a NeedClientCall discards
     the clone, randomness.feed answers it, the retry re-applies)
  AI decisions: the nets via the existing pick machinery
  emission: engine events -> emitter lines (never state-diff
     re-inference), chat through the ChatOut single exit
  witnesses: client_board.diff_engine_vs_dump on demand (report-only)

TURN-START PROTOCOL (every step is a USER ruling; see
sided-play-rulings + LIVE_0818_BUGLIST):
  1. human draws 2 (one per deck)         [D2]
  2. AI draws 2                           [D2]
  3. hand-limit discard resolves NOW      [D4] (engine pending gates)
  4. FP declaration window ends on the declaration gesture or 'fp ok'
                                          [D2: the Fellowship phase]
  5. eye allocation BEFORE the roll [D3]; AI allocating 0 says '0e'
  6. dice roll: each side rolls its OWN pool [G1]; faces adopt via
     game_logic.adopt_rolled_faces (rolled Eyes -> Hunt Box)
  7. action alternation: engine-owned (_next_acting_side /
     _post_action_active); pass legality is the engine's menu
  8. human 'recover dice' triggers the AI's own recover [D1], then
     game_logic.begin_turn_reset (A12) + bonus dice [J1]

THE HAND MODEL (the E1/E2/E3/E4 class killer):
  The engine is authoritative for COUNTS everywhere and for the AI's
  own identities.  The human's cards are OPAQUE PLACEHOLDERS
  ('HU_<side><deck>_<n>') in the engine hand until the client makes
  one public (a play names its title) — then the placeholder is
  swapped for the real cid and the fictional engine deck loses that
  cid.  The AI process never receives a hidden identity (blinding by
  construction); counts can never drift because every hand-zone
  grammar routes through one pair of enter/leave functions.

RAW-CHANNEL IMPORTS (rendered-baseline directive): exactly TWO client
statements live only on the raw channel and are sanctioned — the
CardDecoder commitment decode (identity of the AI's OWN cards) and the
action-roll pid binding (<pids> <values>: which physical die carries
which rolled face; the rendered letters line has no pids, and a wrong
binding would stage a die whose client face differs from the engine's).
Both are client statements, not inference; everything else raw is dead
to this driver.

PORTED FROM THE SPECTATOR (PHASE3-D mechanisms, hoisted into this base
2026-08-20 so ONE implementation serves both drivers — defect-9 rule;
SpectatorDriver's own copies still shadow these until they are deleted
there):
  1. RNG DEFERRAL + WINDOW-RNG FEEDING — a window's own recorded
     randomness is fed per matching attempt; a window whose rolls
     render AFTER its terminator (hunt rolls, tile draws, combat d6)
     is DEFERRED and retried on each new rng-bearing phase record,
     never mis-refused.  Commits re-run the matched chain in ONE fresh
     attempt and end with the unconsumed-randomness honesty gate.
  2. PENDING PHASE-BUFFER — board phase records arriving while a
     HUMAN-side engine pending is open (hunt casualty / guide /
     reveal-place chains, hand-limit discards) buffer into a
     PhaseWindow and match against the pending's own menu.
  3. d6 SIDE-TAGGING — roller nicks bind to sides from ZONE-attributed
     draw lines (fact 3: the deck names the owner, never the nick); a
     nick that demonstrably draws for BOTH sides (the one-client !CALL
     case: the client stamps the LOCAL seat's nick on AI-ordered draws
     too) is POISONED to untagged — zone-derived or nothing, never
     guessed.
"""
import contextlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import chat as chat_mod
from . import client_board
from . import gesture_match
from . import live_parser
from .client_view import ClientView
from .feed_reader import FeedReader, FeedRestart
from .segmenter import Segmenter

# the raw action-roll pid binding (sanctioned raw import #2; the
# donor's _ACTION_ROLL_RE ai_move_server.py:157 VERBATIM - the first
# lift transcribed it as 34[56]/347 and NEVER matched a real line:
# the packaged client emits Controls.345=pids, Controls.346=values
# (+ trailing checksum), measured on every cbt3/cbt6 roll line.  The
# fabricated R4 test specimen had been written FROM the wrong regex -
# a self-consistent wrong pin, the instrument-catalogue class.)
RAW_ACTION_ROLL_RE = re.compile(
    r'<~Controls\.345~>([\d ]+?) ?<~Controls\.346~>([\d ]+)')

_PHASES = ('draws', 'declare', 'eyes', 'roll', 'actions')


class DriverHalt(RuntimeError):
    """A loud, sticky, NAMED stop with a recovery path (instrument
    rule: fail loudly, never wait silently)."""


class _RandCtl(object):
    """gesture_match's `rng` cursor controller over ClientRandomness:
    probe/apply context split + save/restore bracketing (contract at
    match_window).  ONE implementation, hoisted from the spectator
    (WAVE-F built it there; the live driver's bare-`pin` matching ran
    probes with funnels ON — the route_error class)."""

    def __init__(self, rand):
        self.r = rand

    def root_attempt(self):
        self.r.begin_attempt()

    def apply_cm(self):
        return self.r.advance()

    def probe_cm(self):
        return self.r.pin()

    def save(self):
        r = self.r
        fp = r._feed_pos
        return (set(fp['d6']), fp['tiles'], dict(fp['cards']),
                fp['companion'], fp['card_pick'],
                len(r._journal), r._journal_ix, r._probe_ix,
                r._step_rng_cur, r._pending, r._pending_got,
                len(r.shim_uses))

    def restore(self, s):
        r = self.r
        (d6, tiles, cards, comp, cpick, jlen, jix, pix,
         rng_cur, pend, pgot, shim) = s
        r._feed_pos = {'d6': set(d6), 'tiles': tiles,
                       'cards': dict(cards), 'companion': comp,
                       'card_pick': cpick}
        del r._journal[jlen:]
        r._journal_ix = jix
        r._probe_ix = pix
        r._step_rng_cur = rng_cur
        r._pending = pend
        r._pending_got = pgot
        del r.shim_uses[shim:]


class PhaseWindow(object):
    """Window-shaped container for PHASE records that express an open
    engine pending decision (hunt consequences, hand-limit discards,
    guide changes...).  No die staging exists for these — the pending
    itself scopes the gesture; completeness is decided by the match.
    (Hoisted from spectator.py PHASE3-D; spectator keeps a shadowing
    twin until WAVE-F deletes it.)"""
    die_side = None
    die_face = None
    closed = 'pending'

    def __init__(self):
        self.records = []

    def body_records(self):
        return [r for r in self.records
                if r.get('op') in ('move', 'kill', 'flip')]


class HandModel(object):
    """Placeholder bookkeeping for the HUMAN hand (see module doc).
    Engine hand lists are the one count authority; this class only
    mints/swaps placeholder identities inside them."""

    def __init__(self, state, human_side):
        self.es = state
        self.side = human_side
        self._n = 0

    def _mint(self, deck):
        """Placeholder shape HU_<side>_CHAR_n / HU_<side>_STRAT_n: the
        engine's ONE deck classifier (game_logic.card_deck_of) reads
        the deck back out of it."""
        self._n += 1
        return 'HU_%s_%s_%d' % (self.side.name[:2],
                                'CHAR' if deck == 'character'
                                else 'STRAT', self._n)

    def human_draw(self, deck):
        """The human's physical draw line: engine hand grows by ONE
        placeholder; the fictional engine deck shrinks by one (counts
        stay conserved; identity stays with the client)."""
        attr = '%s_%s_deck' % (
            'fp' if self.side.name == 'FREE_PEOPLES' else 'sh',
            'char' if deck == 'character' else 'strat')
        deck_list = getattr(self.es, attr, [])
        if not deck_list:
            raise DriverHalt('human drew from an EMPTY %s - deck '
                             'count desync' % attr)
        deck_list.pop()
        ph = self._mint(deck)
        hand = self.es.hands[self.side].hand
        hand.append(ph)
        self.es.hands[self.side].needs_discard = len(hand) > 6
        return ph

    def reveal(self, title_or_cid, deck=None):
        """A human card became PUBLIC (played / returned face-up):
        swap one placeholder of that deck for the real cid.  Returns
        the cid, or None if resolution failed (caller holds)."""
        cid = self._resolve_cid(title_or_cid)
        if cid is None:
            return None
        hand = self.es.hands[self.side].hand
        if cid in hand:
            return cid                      # already public
        import game_logic as gl
        dk = deck or gl.card_deck_of(cid) or 'strategy'
        ph = next((c for c in hand if isinstance(c, str)
                   and c.startswith('HU_')
                   and gl.card_deck_of(c) == dk), None)
        if ph is None:
            return None
        hand[hand.index(ph)] = cid
        # the fictional deck must not still hold the real cid
        for attr in ('fp_char_deck', 'fp_strat_deck',
                     'sh_char_deck', 'sh_strat_deck'):
            dl = getattr(self.es, attr, None)
            if dl and cid in dl:
                dl.remove(cid)
                dl.append(self._mint(dk))   # count conserved
                break
        return cid

    @staticmethod
    def _resolve_cid(title_or_cid):
        """Client title ('Elven Rope/It is a Gift') -> engine cid via
        ai's accent/alias-tolerant lookup (the single answerer; the
        first cut iterated a nonexistent cards.CARDS dict and resolved
        NOTHING — caught by the card-play lane, 2026-08-19)."""
        t = str(title_or_cid)
        try:
            import cards
            if hasattr(cards, 'get_card') and cards.get_card(t):
                return t                    # already a cid
        except Exception:
            pass
        try:
            from . import gesture_match
            return gesture_match.title_to_cid(t)
        except Exception:
            return None


def _zone_key(name):
    """'Board1' / ' board 1 ' -> 'board1' (rendered zone names)."""
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


# rendered slot name -> client zone code (log_replay:156-157)
_TABLE_SLOT_CODE = {
    'board1': 'Game.2029', 'board2': 'Game.2030',
    'board3': 'Game.2031', 'board4': 'Game.2032',
    'board5': 'Game.2033', 'board6': 'Game.2034',
}
# and back, for reading the client dump (which is keyed by zone NAME)
_TABLE_SLOT_NAME = {v: k for k, v in _TABLE_SLOT_CODE.items()}

# The client keeps the Guide in its OWN box, separate from the rest of
# the Fellowship (client_board's census treats both as "in fellowship",
# which is why the distinction was never read).  Whoever sits here IS
# the human's guide choice.
_GUIDE_ZONES = frozenset(('guideofthefellowship', 'guidebox',
                          'guideofthefellowshipbox'))


def table_card_key(title):
    """Normalised key for a play-on-table card.

    The feed names a card by its FULL client title -- 'A Power Too
    Great/No Quarter', "Denethor's Folly/Devilry of Orthanc" -- where
    the half before the '/' is the event name and the half after is the
    combat-card name.  The engine's PersistentCard.card_name carries
    only the event half ('A Power too Great').  Key on that half,
    case-folded and de-accented, so 'The Palantír of Orthanc' and 'The
    Palantir of Orthanc' are the same card."""
    import unicodedata as _ud
    s = str(title).split('/')[0]
    s = _ud.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


class LiveDriver(object):
    """One instance per sided game."""

    def __init__(self, engine_state, ai_side, feed_path,
                 randomness=None, chat_out=None, dump_path=None,
                 nick='WOTR_AI'):
        import game_state as gs
        self.es = engine_state                  # sole authority (I1)
        self.ai_side = ai_side
        self.human_side = (gs.Side.SHADOW
                           if ai_side is gs.Side.FREE_PEOPLES
                           else gs.Side.FREE_PEOPLES)
        self.nick = nick
        self.feed = FeedReader(feed_path)
        # WAVE-I: the segmenter recognizes the server's OWN emission
        # echoes (self-echo detection only; attribution stays
        # zone-derived) — replaces the WAVE-H die_stage bypass, which
        # left two holes (false window termination by the AI's owed
        # die kill; AI records polluting an open human window's body)
        self.seg = Segmenter(nick=nick)
        self.rand = randomness
        self.rctl = (_RandCtl(randomness)
                     if randomness is not None else None)
        self.last_actor = None       # who last spent a die action
        # pid -> (Board-slot zone code, normalised card key) for EVERY
        # play-on-table card we have watched being seated, by either
        # player.  Filled from the feed in _meta_track; consumed by
        # live_server._table_mopup, which kills the piece in that slot
        # when the engine drops the card.
        self.table_seats = {}
        self.chat = chat_out or chat_mod.ChatOut()
        self.view = ClientView(lambda: self.es, dump_path)
        # the emission planners see the VIEW, not the driver, so the
        # table-seat lookup rides along with it (same shape as
        # card_pid_map).  Must follow the ClientView construction.
        self.view.table_seats = self.table_seats
        self.view.table_seat_pid = self.table_seat_pid
        # ONE hand-model registry (the spectator carries both seats;
        # live carries the human's — the AI's identities ride the
        # commitment decode, never a placeholder).  self.hands stays
        # as the human-hand alias the existing call sites use.
        self.hand_models = {
            self.human_side: HandModel(engine_state, self.human_side)}
        self.hands = self.hand_models[self.human_side]
        self.phase = 'draws'
        self.turn_draws = {}                    # (side, deck) -> n
        # commit ledger: every engine mutation records its pre-state
        # AND the client lines that expressed it; the client's per-LINE
        # undo consumes those lines tail-first and pops the pre-state
        # only when a commit is fully unwound (fact 1: LIFO per line)
        self.commits = []       # [(pre_state, [norm line texts], label)]
        self.uncommitted = []   # norm lines of refused/held windows
        self.alarms = []
        self.unknown_lines = []
        self.holds = []                         # open matcher holds
        # side name -> raw roll line's pid order, awaiting adoption
        # (sanctioned raw import #2; consumed by _try_bind_roll).
        # BOTH orders happen live (cbt4): the feed's letters line can
        # be polled BEFORE the raw request line arrives (independent
        # channels) - so adoption also PARKS its index order and the
        # bind fires from whichever side completes the pair.
        self._pending_roll_pids = {}
        self._last_adopt_idxs = {}      # side name -> (idxs, n_faces)
        self._adopt_parked_tick = {}      # WAVE-K R6: side -> tick parked
        self._adopt_starve_alarmed = set()
        self._tick_n = 0
        self._seen_roll_lines = set()
        # ---- PHASE3-D ported state (see module doc; the spectator's
        # __init__ re-seeds the same names — identical initial values,
        # its assignments harmlessly overwrite these) ------------------
        self.findings = []          # census rows (kind, detail)
        # WAVE-J (A-09/A-10): the game-end verdict — set by
        # check_victory (ring: mid-action; military: TURN END per
        # RULING 1.212) or a human concession; once set the server
        # stops all AI work and closes the log
        self.game_over = None
        self.window_stats = {'matched': 0, 'hold': 0, 'none': 0,
                             'ambiguous': 0}
        self.pending_stats = {'matched': 0, 'expired': 0}
        # a window whose ONLY failure was rng starvation waits here
        # for the randomness the client renders AFTER the window
        # (hunt rolls / tile draws land post-terminator)
        self.deferred = None        # (window, side) | None
        self.pending_buf = None     # PhaseWindow | None
        # board phase records arriving WHILE a window is deferred:
        # they may express the pendings its commit will open — queued,
        # replayed after the deferral settles
        self.deferred_tail = []
        # roller-nick -> side name | None(=poisoned), bound from the
        # ZONE-attributed draw lines (fact 3 compliant); feeds the d6
        # side lanes so interleaved combat segments stay straight
        self.nick_side = {}
        # WAVE-H (combat leg): a chat-committed human decision whose
        # apply STARVED on client randomness (the defender's 'nc'
        # triggers the combat round's rolls).  Same deferral doctrine
        # as windows: retried on each rng-bearing record, its ask
        # surfaced for live_server to ORDER, expired LOUDLY when the
        # client moves on.  (The old _commit_human applied under pin()
        # at advance-depth 0, where the funnels are INERT - the round
        # would have rolled ENGINE dice, violating the client-owns-
        # randomness mandate silently.)
        self.chat_deferred = None       # (action, via) | None
        # die kills owed AFTER a chain closes (USER invariant 'die ->
        # action -> resolve everything -> kill die'; donor :15760-15829
        # for play_event / army_attack / draw_card).  live_server
        # appends at emission time and flushes render_die_kill once the
        # chain is closed - the lift dropped the flush entirely and the
        # spent die stranded in Current Action forever.
        self.owed_die_kills = []        # [(side_name, die_idx)]
        # WAVE-K (R4, A-07 residue): the AI's COMMITTED COMBAT CARD dies
        # in the Current-Card slot when the combat closes (corpus
        # 'kills <title> in Current Action.'); [(side_name, pid, cid)]
        self.owed_card_kills = []
        # K1: the human committed an UNREVEALED combat card this round
        # (ai.commit_hidden_combat_card); cleared by the reveal line
        self.human_cc_committed = False
        # K1 refusal epoch: bumped when a refused reveal withdraws the
        # human's hidden commit, so the round's reveal directive goes
        # out again for the re-commit (live_server keys its once-per-
        # round dedup on it)
        self.cc_reveal_epoch = 0
        # bumped whenever a human undo pops the engine state: an AI
        # step open on the popped state is stale (live_server abandons
        # it - K3)
        self.undo_epoch = 0
        # monotonic commit-epoch counter (the capped self.commits list
        # cannot count) — live_server keys its window-ask dedup on it
        self.n_commits_total = 0
        # items produced by nested calls (deferral retries fired from
        # deep inside _route) that must still reach the client —
        # refusal undo presses, holding notes; tick() drains this
        self._out_queue = []
        self._last_blocked_kind = None  # blocked-note de-spam
        # WAVE-G port (WAVE-I hoist): second-consecutive-empty-expiry
        # tracker for the pending force (spectator used getattr; the
        # base seeds it so the hoisted body is drop-in)
        self._expire_streak = None
        # ---- WAVE-I state ------------------------------------------------
        # MIRROR-ECHO LEDGER (blocker 1, the cbt10 race): unit-removal
        # rows a committed apply already resolved engine-side.  AI-side
        # rows are EMITTED by the sweep and banked here; HUMAN-side
        # rows the gesture did not express are banked un-emitted (the
        # human renders their own casualties).  An incoming non-self
        # kill/removal record fully covered by open entries is an ECHO
        # of the committed effect — consumed, never a new gesture.
        # Entries: dict(kind='unit'|'naz'|'named', side='fp'|'shadow',
        #               region=<region_key>, tkey/name, n)
        self.echo_ledger = []
        # SETTLEMENT-CAPTURE ledgers (blocker 2): control markers
        # emitted per region (region_key -> pid) and the VP value each
        # side's marker was last dragged to (donor grammar
        # ai_move_server :10094-10236; engine truth enemy_controlled +
        # calculate_victory_points).
        self.ctrl_markers = {'fp': {}, 'sh': {}}
        self.vp_marks = {'fp': 0, 'sh': 0}
        # WAVE-I: the refusal-restore watch, ARMED at every refusal
        # (undo_model.RestoreWatch — its alarm is module-proven; this
        # wires it into the live tick so it can actually fire, the C4
        # class closed END-TO-END, not just at module level)
        from .undo_model import RestoreWatch
        self.restore_watch = RestoreWatch()
        # USER 2026-08-22 'Enforce legal moves for HUMAN player (beta)':
        # True = the legality gate (refusals / undo presses / holds);
        # False = the AI NEVER countermands the human - every gesture
        # the matcher cannot commit is taken AS GIVEN (the staged die
        # spent, the client board adopted into the engine).  The live
        # server sets this from the panel's control file (default OFF
        # for testers); the offline suites keep the default True.
        self.enforce_human = True
        # CLIENT-WAVE sidecar (WAVE-J adoption): live_server attaches
        # a SidecarReader; None = channel absent (corpus soak, old
        # jars) and the heuristic lanes stay - one answerer PER
        # CHANNEL AVAILABILITY
        self.sidecar = None
        self._feed_seq = 0          # 1-based feed line number of the
        #                             line being ingested (= its index
        #                             in feed.lines + 1)
        self._route_errors_seen = set()   # distinct error texts alarmed

    # ------------------------------------------------------------------
    def tick(self, raw_request_lines=()):
        """One cycle: ingest feed + the request's raw lines (sanctioned
        imports only), advance, return response items:
        [('line', s) | ('call', s) | ('chat', kind, text)]."""
        out = []
        # CLIENT-WAVE adoption (defect-9): with a PRESENT sidecar the
        # raw import #2 DIES - the meta die channel answers roll-pid
        # binding; the raw regex lane runs only when the channel is
        # absent (corpus soak / old jars / suites without a meta file)
        sc = self.sidecar
        if sc is not None:
            sc.poll()
        if sc is None or not sc.present:
            for raw in raw_request_lines:
                m = RAW_ACTION_ROLL_RE.search(raw)
                if m and raw not in self._seen_roll_lines:
                    # each raw roll line is consumed ONCE (AiOpp
                    # resends the whole log every tick; re-storing a
                    # consumed binding would poison the NEXT adoption)
                    self._seen_roll_lines.add(raw)
                    self._bind_roll_pids(m.group(1).split(),
                                         m.group(2).split())
        try:
            new = self.feed.poll()
        except FeedRestart as e:
            raise DriverHalt(
                'FEED RESTART: %s - client restarted or a new game '
                'began; restart the driver against the new feed.' % e)
        # ONE answerer for 'feed line number': the reader's own line
        # index (seq is 1-based over feed.lines).  Every line advances
        # it, routed or not, so the sidecar alignment survives any
        # per-line routing error.
        base = len(self.feed.lines) - len(new)
        for i, raw in enumerate(new):
            self._feed_seq = base + i + 1
            try:
                out.extend(self._ingest_line(raw, sc))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # a routing error is isolated to ITS line: the rest
                # of the batch still routes.  LOUD: a finding per
                # line, a sticky alarm + panel note once per distinct
                # error text.
                msg = '%s: %s' % (type(e).__name__, e)
                self.findings.append(('route_error',
                                      'line %d: %s :: %r'
                                      % (self._feed_seq, msg[:120],
                                         raw[:80])))
                if msg not in self._route_errors_seen:
                    self._route_errors_seen.add(msg)
                    self.alarms.append('ROUTE ERROR (feed line %d): '
                                       '%s' % (self._feed_seq, msg))
                    out.append(('note', 'route error on %r: %s'
                                % (raw[:80], msg)))
        if self._out_queue:
            # items minted by nested deferral retries (refusal undo
            # presses, holding notes) — surfaced before the AI acts
            out.extend(self._out_queue)
            self._out_queue = []
        # WAVE-I: refusal-restore verification (armed-only, dump-read)
        self._check_restore_watch()
        # WAVE-K (R6): sidecar-present roll with no die channel -> alarm
        self._tick_n += 1
        self._check_roll_binding_starvation()
        # WAVE-I (wirF live finding): a pending buffer that stops
        # moving is a JAM (the harness's Woodland Realm over-kill sat
        # SILENT for many minutes — no alarm, no counter; instrument
        # catalogue: a stall with no counter is invisible).  One loud
        # note per 25 stalled ticks; the census row keeps it on the
        # books.
        if self.pending_buf is not None and self.pending_buf.records:
            import ai as _ai
            _dp = _ai.describe_pending(self.es)
            if _dp is None:
                # the pending this buffer was collected FOR resolved
                # by another route (chat pick, force): the records can
                # never match anything — expire them LOUDLY, never an
                # eternal silent residue
                norms = [r.get('norm', '')
                         for r in self.pending_buf.records]
                self.uncommitted.extend(norms)
                self.findings.append(('pending_buffer_orphaned',
                                      '%d record(s): %s'
                                      % (len(norms),
                                         (norms[:1] or [''])[0][:80])))
                self.alarms.append('PENDING BUFFER ORPHANED '
                                   '(%d record(s)) - pending resolved '
                                   'by another route' % len(norms))
                # an orphaned buffer never commits: give back any
                # reveal-swaps it made (see _unreveal)
                self._unreveal(self.pending_buf)
                self.pending_buf = None
                self._buf_stall = None
            else:
                fp = (len(self.pending_buf.records),
                      self.n_commits_total)
                if getattr(self, '_buf_stall', None) \
                        and self._buf_stall[0] == fp:
                    self._buf_stall = (fp, self._buf_stall[1] + 1)
                    if self._buf_stall[1] % 25 == 0:
                        msg = ('PENDING BUFFER STALLED: %d record(s) '
                               'unmatched for %d ticks (pending=%s) - '
                               'the gesture does not equal any legal '
                               'option; RESYNC/undo it :: records=%r '
                               ':: matcher=%s'
                               % (fp[0], self._buf_stall[1],
                                  _dp['kind'],
                                  [r.get('norm', '')[:60] for r in
                                   self.pending_buf.records][:3],
                                  str(getattr(self, '_last_pending_match',
                                              ''))[:200]))
                        self.findings.append(('pending_buffer_stall',
                                              msg[:160]))
                        out.append(('note', msg))
                else:
                    self._buf_stall = (fp, 1)
        else:
            self._buf_stall = None
        out.extend(self._ai_turn())
        # a deferred window / pending buffer starving on client
        # randomness surfaces its ask so live_server can ORDER it
        # (!CALL for the AI's own dice, a chat ask for the human's) —
        # without this a live defender-roll starves forever (the
        # spectator reads the rng from the log; live must solicit it).
        # ONLY d6/tile/randomcomp asks are auto-ordered: rand.pending()
        # is the LAST starving candidate's call, not provably the
        # gesture's — an excess roll/draw is quarantined by the
        # honesty gate, but a wrong '!CALL draw_*' would draw a real
        # card (CARD kinds stay excluded).  randomcomp added WAVE-J
        # (wjg10 live: a hunt-casualty random-companion draw starved a
        # deferred fellowship window to loud expiry — the directive
        # existed, the gate never let it out; its ask is USER-mandated
        # chat + a parser-inert builtin).  Other kinds surface as a
        # loud note instead.
        if self.deferred is not None or self.pending_buf is not None \
                or self.chat_deferred is not None:
            p = self.rand.pending() if self.rand is not None else None
            if p is None and self.deferred is not None:
                # the global pending slot was clobbered by an
                # interleaved attempt - the deferral's own captured
                # starve call is the surviving truth (wjg11)
                _dw = (self.deferred[0]
                       if isinstance(self.deferred, tuple)
                       and self.deferred else None)
                p = getattr(_dw, '_starve_call', None)
            if p is not None:
                _items = (list((p.kw or {}).get('items') or ())
                          if p.kind == 'draws' else [])
                if p.kind in ('d6', 'tile', 'randomcomp'):
                    out.append(('rng_ask', p))
                elif p.kind == 'draws' and _items and all(
                        str(sn) == self.ai_side.name for sn, _ in _items):
                    # the AI's OWN deck (an engine-granted draw inside a
                    # human-decided round, e.g. the Witch-king's
                    # Sorcerer draw): unambiguous - ordered like a roll
                    # (epoch dedup + watchdog protect against a double
                    # order); a human-side deck stays the human's hand
                    out.append(('rng_ask', p))
                elif p.kind != self._last_blocked_kind:
                    out.append(('note',
                                'blocked on client %s (%r) - not '
                                'auto-ordered (candidate-starve '
                                'ambiguity); resolve via the client'
                                % (p.kind, p.kw)))
                self._last_blocked_kind = p.kind
        else:
            self._last_blocked_kind = None
        return out

    def _ingest_line(self, raw, sc):
        """Parse + sidecar-attach + segment + route ONE feed line (the
        per-line unit tick() isolates errors around)."""
        out = []
        rec = live_parser.parse(raw)
        if sc is not None and sc.present:
            mrec = sc.rec(self._feed_seq)
            if mrec is not None:
                rec['meta'] = mrec
                self._meta_track(rec)
            else:
                self.findings.append(('sidecar_gap',
                                      'seq %d has no meta record'
                                      % self._feed_seq))
        if rec.get('unknown'):
            self.unknown_lines.append(raw)
            self.alarms.append('UNKNOWN LINE: %r' % raw)
        for ev, payload in self.seg.push(rec):
            out.extend(self._route(ev, payload))
        return out

    # ------------------------------------------------------------------
    def _route(self, ev, payload):
        """Ported PHASE3-D routing shell around the event dispatch:
        phase board records defer behind a starving window / buffer
        into an open HUMAN-side pending; rng-bearing records retry
        whatever is blocked.  (The spectator's _route override layers
        its mirror + both-seat scoping on top and delegates here.)"""
        if ev == 'update' \
                and getattr(self, '_chat_attack_window', None) is payload \
                and not getattr(self, 'spectator', False) \
                and payload.records:
            # A committed chat-attack keeps its die window open ONLY to
            # swallow the terminating die kill (the segmenter closes on
            # that kill, never on an 'update').  Every OTHER record the
            # window collects is LIVE combat traffic - the round's own
            # d6 rolls above all - that must reach the funnel/handlers
            # NOW: a combat roll buffered into the window deadlocks the
            # round (the die kill that would release the buffer is
            # itself gated on the round the roll must resolve; the
            # human-attacker seat wedged here where the AI-attacker
            # seat never does - the AI's staging is a self-echo that
            # opens no window).  The record stays in w.records; the
            # window's close consumes it without re-processing.
            return self._route('phase', payload.records[-1])
        if ev == 'phase' \
                and payload.get('op') in ('move', 'kill') \
                and self._consume_mirror_echo(payload):
            # WAVE-I (the cbt10 race): the client's mirror of an
            # already-committed removal - consumed above, never a
            # gesture, never pending-buffer content.  The AI's OWN
            # forced-row line echoes here too (its nick): the row
            # banked at emission is claimed by that echo, so the
            # human's mirror list (echo_open) carries only the
            # human's rows (D6: an unclaimed AI row polluted the
            # batch and the human's forced casualties went unmirrored)
            return []
        if ev == 'phase' and payload.get('op') == 'move' \
                and not str(payload.get('player', '')
                            ).startswith(self.nick) \
                and not getattr(self, 'spectator', False) \
                and gesture_match.is_ring_track_move(
                    payload.get('obj'), payload.get('from'),
                    payload.get('to')):
            # a ring walked with NO die staged: the standalone ring
            # lane (commit against the elven_ring menu or refuse) -
            # never pending-buffer content, never a silent drop
            return self._on_ring_walk(payload)
        if ev == 'phase' and payload.get('op') == 'move' \
                and not getattr(self, 'spectator', False):
            # a '(chosen)' reveal from a NON-hand zone expresses
            # nothing in fact space (inert to the lanes below) yet it
            # is the human's combat-card reveal gone wrong - routed
            # BEFORE the inert gate, never a silent drop
            _src = gesture_match.zone_key(payload.get('from'))
            if _src[0] == 'chosen_other':
                items = self._on_chosen_nonhand(payload, _src[1])
                if items is not None:
                    return items
        if ev == 'phase' \
                and payload.get('op') in ('move', 'kill', 'flip') \
                and not str(payload.get('player', '')
                            ).startswith(self.nick) \
                and not self._is_dice_bookkeeping(payload) \
                and not self._rec_inert(payload):
            # DICE bookkeeping records (recover drags, pool moves) are
            # never pending content: they must reach the normal
            # handlers or a pending open at turn end eats the rollover
            # (measured: guide_change swallowed the recover and the
            # whole next turn cascaded).  Our own emission echoes are
            # never the human's pending resolution either (nick used
            # for SELF-echo detection only — the attribution ruling
            # forbids guessing the HUMAN's side from a nick).
            # WAVE-J (wjg6 LIVE wedge): the AI's OWN combat-card
            # commitment echoes with '<auto>' attribution — the
            # client's chooser renders '<auto> moves <card> from
            # Shadow Hand (chosen) to Current SA Card.' for OUR
            # '!CALL choose_card', and the nick-based self-echo filter
            # cannot see it.  It buffered as pending content and
            # poisoned the human's casualty match (2-record buffer,
            # stalled 100+ ticks).  An (chosen)-source move from the
            # AI's OWN hand is a SELF echo in live play — consumed,
            # never buffered.  (Spectator: both seats external, its
            # cross-side lane owns these — unchanged.)
            if not getattr(self, 'spectator', False) \
                    and payload.get('op') == 'move':
                src = gesture_match.zone_key(payload.get('from'))
                if src[0] == 'hand_chosen':
                    items = self._on_chosen_reveal(payload, src[1])
                    if items is not None:
                        return items
            if self.deferred is not None:
                # a window is still starving: these records belong
                # AFTER its commit — queue, replay when it settles
                self.deferred_tail.append(payload)
                return []
            import ai
            # K2: a human FP's declaration records must find the
            # declare menu open (the AI's own turn-start actions may
            # have cleared the engine flag since it was armed)
            self._ensure_declare_armed()
            dp = ai.describe_pending(self.es)
            if dp is not None and self._pending_buffer_applies(dp):
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf.records.append(payload)
                self._try_pending()
                return []
        out = self._route_event(ev, payload)
        # rng-bearing phase records may unblock the pending buffer or
        # a deferred window
        if ev == 'phase' and payload.get('op') in ('roll_num', 'draw',
                                                   'random_discard'):
            self._retry_blocked()
        return out

    # the six play-on-table slots, rendered name -> client zone code
    # (log_replay:156-157).  A card seated in one of these is a live
    # persistent card; killing it happens IN the slot (corpus: 6,686
    # 'kills <card> in Board<N>' lines).
    _TABLE_SLOT_CODES = ('Game.2029', 'Game.2030', 'Game.2031',
                         'Game.2032', 'Game.2033', 'Game.2034')

    def _track_table_seat(self, rec, pids):
        """Watch a play-on-table card into (or out of) a Board slot.

        Keyed by the card's NAME, not its pid: the pid is the one thing
        we may NOT have (a card played from a hidden hand is redacted
        by the feed's blinding), while the rendered line always names
        the card and always names the slot.  The pid is resolved later
        from the client's own dump - the client is the authority on
        where its pieces are, and a STATIC pid table is not: it said
        145 was 'The Power of Tom Bombadil' when this build renders 145
        as 'Kindred of Glorfindel', and the AI dutifully dragged the
        wrong card out of the FP Strategy Deck (USER 2026-08-20)."""
        key = table_card_key(rec['obj'])
        if not key:
            return
        slot = _TABLE_SLOT_CODE.get(_zone_key(rec['to']))
        cur = self.table_seats.get(key)
        if slot is None:
            # the card LEFT its slot (dragged to a Current-Card zone
            # for a deliberate kill): the kill lane owns it now
            if cur is not None:
                self.table_seats.pop(key, None)
            return
        pid = str(pids[0]) if (pids and len(pids) == 1) else None
        # ARMED=False: a seat is NOT moppable until the ENGINE has been
        # seen holding that card (the Tom Bombadil killed-on-sight
        # class).  An existing seat keeps whatever it already proved.
        self.table_seats[key] = {
            'slot': slot,
            'pid': pid or (cur or {}).get('pid'),
            'armed': bool((cur or {}).get('armed')),
        }

    def table_seat_pid(self, key):
        """The client piece id of a seated table card, resolved from
        the client's DUMP when the feed never gave us one.  A slot
        holding exactly one piece names it unambiguously; anything else
        returns None and the caller HOLDS rather than guessing."""
        ent = self.table_seats.get(key)
        if not ent:
            return None
        if ent.get('pid'):
            return str(ent['pid'])
        slot = ent.get('slot')
        want = _zone_key(_TABLE_SLOT_NAME.get(slot, ''))
        if not want:
            return None

        def _occupants():
            return [str(p) for p, z in
                    (self.view.piece_region or {}).items()
                    if _zone_key(z) == want]

        occ = _occupants()
        if not occ:
            # ASK THE CLIENT, don't settle for a stale cache (USER
            # 2026-08-20: "why do you need to guess? you can just
            # extract it from the game state").  The board dump IS the
            # client's statement of where its pieces are; a seat that
            # cannot be resolved from a cached copy is usually just a
            # copy older than the drag we watched.
            try:
                self.view.refresh_dump()
            except Exception:
                pass
            occ = _occupants()
        if len(occ) == 1:
            ent['pid'] = occ[0]
            return occ[0]
        if len(occ) > 1:
            # a table slot holds ONE card; more than one means the dump
            # and our slot bookkeeping disagree.  Say so - never pick.
            self.alarms.append(
                'TABLE SLOT %s holds %d pieces (%s) - cannot name the '
                'seated card; refusing to pick one'
                % (slot, len(occ), ','.join(sorted(occ))[:60]))
        return None

    def _forget_table_seat_pid(self, pid):
        for k, e in list(self.table_seats.items()):
            if str(e.get('pid') or '') == str(pid):
                self.table_seats.pop(k, None)

    def _meta_track(self, rec):
        """CLIENT-WAVE (c): pid-TRUE view bookkeeping from the
        sidecar's exact pids - piece positions stop depending on the
        fungible dump-intersection alone (the dump stays the WITNESS;
        this is tracking, never authority).  Redacted records update
        nothing (blinding fail-closed: meta_pids treats 'redacted' as
        absent - no bypass around the feed's blinding)."""
        from . import sidecar as sc_mod
        pids = sc_mod.meta_pids(rec.get('meta'))
        # TABLE-CARD SEATING IS TRACKED EVEN WITHOUT PIDS.  A card
        # played straight from the hand carries `pids: redacted` (the
        # feed blinds anything leaving a hidden hand - correctly), so
        # the old `if not pids: return` above dropped the human's
        # Tom Bombadil seat entirely.  The card is PUBLIC the moment it
        # lands in a Board slot, and the rendered line NAMES it, so the
        # seat is booked by NAME here and its pid is resolved from the
        # client's own dump when it is needed (USER 2026-08-20: "it
        # should see where I placed the power of tom bombadil from the
        # rendered log, and delete it there").
        if rec.get('op') == 'move' and rec.get('to') and rec.get('obj'):
            self._track_table_seat(rec, pids)
        if not pids:
            return
        op = rec.get('op')
        if op == 'move' and rec.get('to'):
            for p in pids:
                self.view.piece_region[str(p)] = str(rec['to'])
            # TABLE-CARD SEATING BY *EITHER* PLAYER (USER 2026-08-20:
            # "you can see where the human player places the table
            # card, so just keep track of that location and kill it
            # there").  live_server._table_book is written ONLY by the
            # AI's own seat line, so a card the HUMAN played was never
            # booked and NOTHING removed it from their board when the
            # engine discarded it - measured 0/14 human-seated vs
            # 14/14 AI-seated.  The sidecar already gives us the exact
            # pid and the feed names the card, so the seat is simply
            # observed here, for both seats, and the mop-up kills the
            # piece IN THE SLOT we saw it go into.
        elif op == 'kill':
            for p in pids:
                self._forget_table_seat_pid(p)
            # mark DEAD, never pop: an untracked pid falls back to its
            # SETUP zone in emitter._pieces_in (the wjg13 dead-pid
            # re-pick class)
            for p in pids:
                self.view.piece_region[str(p)] = 'Dead'

    def _pending_buffer_applies(self, dp):
        """May phase records buffer into this open pending?  Live: only
        the HUMAN's pendings resolve from the feed — the AI's resolve
        through live_server's step machinery.  (The spectator's _route
        override buffers BOTH seats — both are external there.)"""
        return dp['side'] is self.human_side

    # ---- K1: combat-card two-phase protocol, the driver's half --------
    def _combat_cards_open(self):
        """Is a battle open in the engine?  The window/buffer context
        gesture_match reads to classify a chooser '(chosen)' reveal
        (combat card vs CardBuffet.9 event play).  Spectator: unstated
        (None) - both seats are external there."""
        if getattr(self, 'spectator', False):
            return None
        return isinstance(getattr(self.es, 'pending_combat', None),
                          dict)

    def _new_phase_window(self):
        pw = PhaseWindow()
        pw.combat_cards_open = self._combat_cards_open()
        return pw

    def _cc_unrevealed(self):
        """Sides whose committed combat card is still hidden, from
        the ONE engine answerer (ai.combat_cards_unrevealed); empty
        when no battle is open.  The engine surface missing while a
        battle is open is LOUD."""
        import ai
        if not isinstance(getattr(self.es, 'pending_combat', None),
                          dict):
            return set()
        if not hasattr(ai, 'combat_cards_unrevealed'):
            if self.human_cc_committed:
                raise DriverHalt('ENGINE SURFACE MISSING: ai.'
                                 'combat_cards_unrevealed (K1 step 1)')
            return set()
        return set(ai.combat_cards_unrevealed(self.es) or ())

    def _human_cc_decision_open(self):
        """Is the engine's open decision the HUMAN's combat-card
        commitment (attacker_/defender_combat_card)?"""
        import ai
        dp = ai.describe_pending(self.es)
        return (dp is not None and dp['side'] is self.human_side
                and str(dp['kind']).startswith('combat:')
                and str(dp['kind']).endswith('_combat_card'))

    _CHOOSE_LABEL_RE = re.compile(r'\[([^\]]*)\]\s*$')

    @classmethod
    def _deck_from_label(cls, text):
        """'<nick> chooses a card.[<deck back label>]' -> (Side, deck)
        or (None, None).  The deck back names its OWNER side (zone-
        derived attribution, never the nick): 'Free ... Card' ->
        FP, 'Shadow ... Card' -> SHADOW; 'Character' -> character,
        any 'Strategy' variant ('(Army)'/'(Muster)' sub-decks) ->
        strategy."""
        import game_state as gs
        m = cls._CHOOSE_LABEL_RE.search(str(text or ''))
        label = (m.group(1) if m else '').strip().lower()
        side = (gs.Side.FREE_PEOPLES if label.startswith('free')
                else gs.Side.SHADOW if label.startswith('shadow')
                else None)
        deck = ('character' if 'character' in label
                else 'strategy' if 'strategy' in label else None)
        return (side, deck) if (side and deck) else (None, None)

    def _on_choose(self, rec):
        """'<nick> chooses a card.[<label>]' - a HIDDEN combat-card
        commit when the engine's open decision is the human's combat
        card (K1 step 3: ai.commit_hidden_combat_card; identity stays
        with the client until the reveal).  The AI's own choose_card
        echo (the label names the AI's deck) is consumed.  Outside a
        combat-card decision the chooser is the CardBuffet.9 event
        play: the '(chosen)' reveal line that follows is the gesture,
        this line commits nothing."""
        if getattr(self, 'spectator', False):
            return []
        side, deck = self._deck_from_label(rec.get('norm')
                                           or rec.get('raw'))
        if side is None:
            self.alarms.append('CHOOSE line without a deck label: %r'
                               % str(rec.get('raw', ''))[:80])
            return []
        if side is self.ai_side:
            return []               # our own !CALL choose_card echo
        if not self._human_cc_decision_open():
            return []               # event play via the chooser
        import ai
        if not hasattr(ai, 'commit_hidden_combat_card'):
            raise DriverHalt('ENGINE SURFACE MISSING: ai.commit_'
                             'hidden_combat_card (K1 step 1) - the '
                             "human's hidden combat card cannot be "
                             'committed')
        human = self.human_side

        def _commit(state):
            ai.commit_hidden_combat_card(state, human, deck)
        _commit.action_type = 'combat_card_hidden_commit'
        self.findings.append(('cc_hidden_commit',
                              '%s %s' % (human.name, deck)))
        return self._attempt_chat_action(_commit, via='choose')

    def _on_chosen_reveal(self, rec, tag):
        """A '<auto> moves <TITLE> from <Side> Hand (chosen) to
        Current <X> Card.' line.  Returns driver items when the line
        was consumed here, None when it is an ordinary gesture record
        (the chooser event play) for the window/pending lanes.
          AI side:    self echo (cc_reveal_echo); reflected into the
                      engine only if the engine still lists the AI
                      side as unrevealed (idempotent, K1 step 4).
          human side: with a hidden commit outstanding, the reveal -
                      placeholder swapped for the named cid, then
                      ai.reveal_combat_card on a clone (transactional;
                      a starved round defers like a chat decision).
                      An unplayable card is REFUSED: swap given back,
                      chat notice, the engine keeps the hidden commit
                      (the human chooses again or says nc)."""
        import ai
        import game_state as gs
        side = gs.Side.FREE_PEOPLES if tag == 'FP' else gs.Side.SHADOW
        title = str(rec.get('obj', '') or '')
        norm = str(rec.get('norm', ''))[:80]
        if side is self.ai_side:
            self.findings.append(('cc_reveal_echo', norm))
            if self.ai_side in self._cc_unrevealed():
                cid = gesture_match.title_to_cid(title)
                if cid is None or not hasattr(ai, 'reveal_combat_card'):
                    self.alarms.append('AI COMBAT CARD REVEAL not '
                                       'reflected: %r' % title[:60])
                else:
                    ai.reveal_combat_card(self.es, self.ai_side, cid)
            return []
        if side is not self.human_side:
            return None
        if not self.human_cc_committed \
                and side not in self._cc_unrevealed():
            return None             # the chooser EVENT play
        if not hasattr(ai, 'reveal_combat_card'):
            raise DriverHalt('ENGINE SURFACE MISSING: ai.reveal_'
                             'combat_card (K1 step 1)')
        hm = self.hands
        cid = hm.reveal(title)
        if cid is None:
            self.alarms.append('COMBAT CARD REVEAL: cannot resolve '
                               '%r to a hand placeholder' % title[:60])
            return [('note', 'combat card reveal unresolved: %s'
                     % title[:60])]
        human = self.human_side

        def _reveal(state):
            ai.reveal_combat_card(state, human, cid)
        _reveal.action_type = 'combat_card_reveal'
        try:
            items = self._attempt_chat_action(_reveal, via='cc_reveal')
        except DriverHalt as e:
            # unplayable here: the engine refused (untouched).  USER
            # E4 / rule 5: the card RETURNS TO HAND - the hidden
            # commit is withdrawn (that side's card window re-opens,
            # the other side's commit stands) and the client's reveal
            # is undone by ONE client-native press.  A human mistake
            # is a finding + notice, never an alarm.  The identity is
            # public now (the client named it): the hand keeps the
            # real cid.
            self.findings.append(('cc_reveal_refused',
                                  '%s: %s' % (title[:40], str(e)[:80])))
            items = self._withdraw_human_cc(rec)
            items.append(('chat', 'refusal', chat_mod.refusal_illegal(
                'that combat card cannot be played in this battle - '
                'it goes back to your hand; choose another or say nc')))
            return items
        self.findings.append(('cc_reveal', '%s %s' % (human.name, cid)))
        return items

    def _on_chosen_nonhand(self, rec, tag):
        """A '(chosen)' reveal whose source is NOT a hand ('<auto>
        moves TITLE from FP Casualties (chosen) to Current FP Card.'):
        the client's chooser let the human pick a card from the
        casualties pile.  Final3 real-client leg 2026-08-22: the line
        matched no lane, the hidden commit stayed outstanding and the
        battle wedged SILENTLY (0 alarms, pending None forever).  A
        card the side does not hold cannot be its combat card: the
        hidden commit is withdrawn, the client's reveal undone by ONE
        press, the human told - the same refusal as an unplayable
        card.  An AI-side one is a loud alarm (the AI picks from its
        hand only).  None when no commit is outstanding (the lanes
        below own it)."""
        import game_state as gs
        side = (gs.Side.FREE_PEOPLES if tag == 'FP'
                else gs.Side.SHADOW if tag == 'SH' else None)
        title = str(rec.get('obj', '') or '')
        if side is None:
            return None
        if side is self.ai_side:
            if self.ai_side in self._cc_unrevealed():
                self.alarms.append('AI COMBAT CARD REVEAL from a '
                                   'non-hand zone: %r' % title[:60])
            return None
        if side is not self.human_side:
            return None
        if not self.human_cc_committed                 and side not in self._cc_unrevealed():
            return None
        self.findings.append(('cc_reveal_refused',
                              '%s: not in hand (%s)'
                              % (title[:40],
                                 str(rec.get('from', ''))[:30])))
        items = self._withdraw_human_cc(rec)
        items.append(('chat', 'refusal', chat_mod.refusal_illegal(
            'that card is not in your hand - it goes back; choose a '
            'combat card from your hand or say nc')))
        return items

    def _withdraw_human_cc(self, rec):
        """The engine refused the human's revealed card: withdraw the
        hidden commit (ai.cancel_hidden_combat_card) and put the client
        back with ONE undo press.  Invariants: the '<auto>' reveal is
        the client's newest snapshot (the chooser pushes none), so one
        press retracts it; the client's undo also folds every
        adjacent '<auto>' line below it - the AI's own reveal of the
        same directive - which _on_reveal_undone re-shows from the
        undo echo; the refused line is booked uncommitted so its echo
        is bookkeeping; the reveal epoch bumps so the re-commit's
        reveal directive goes out again."""
        import ai
        from emitter import render
        human = self.human_side

        def _cancel(state):
            ai.cancel_hidden_combat_card(state, human)
        _cancel.action_type = 'combat_card_cancel'
        items = self._attempt_chat_action(_cancel, via='cc_cancel')
        self.cc_reveal_epoch += 1
        norm = str(rec.get('norm', '') or '')
        if norm:
            self.uncommitted.append(norm)
        items.append(('line', render('undo', {'nick': self.nick},
                                     local=False)))
        return items

    def _on_reveal_undone(self, urec):
        """A '(chosen)' reveal line came back off the client (an undo
        press).  Returns driver items, or None when the line is not
        this lane's (a human-side reveal the ledger owns).
          our refused line: bookkeeping (uncommitted -> consumed);
          the AI's OWN card, retracted by the same press (the client
          folds adjacent '<auto>' lines): re-shown by its own
          play_card while the engine still holds it played in the
          open battle - its identity is already public; otherwise a
          loud alarm (RESYNC owed)."""
        import game_state as gs
        norm = str(urec.get('norm', '') or '')
        if norm and norm in self.uncommitted:
            self.uncommitted.remove(norm)
            self.findings.append(('cc_refused_returned', norm[:80]))
            return []
        src = gesture_match.zone_key(urec.get('from'))
        side = gs.Side.FREE_PEOPLES if src[1] == 'FP' else gs.Side.SHADOW
        if side is not self.ai_side:
            return None
        title = str(urec.get('obj', '') or '')
        cid = gesture_match.title_to_cid(title)
        pc = getattr(self.es, 'pending_combat', None)
        played = (isinstance(pc, dict) and cid is not None
                  and cid not in self.es.hands[side].hand)
        pid = self.view.card_pid_map.get(cid) if cid else None
        if played and pid is not None:
            self.findings.append(('cc_ai_reveal_restored',
                                  '%s pid %s' % (cid, pid)))
            return [('call', '!CALL play_card %s' % pid)]
        self.alarms.append('AI COMBAT CARD REVEAL RETRACTED (%s) - not '
                           'restorable (played=%s pid=%s); RESYNC owed'
                           % (title[:40], played, pid))
        return [('note', 'AI combat card reveal retracted by the undo: '
                         '%s' % title[:60])]

    def _route_event(self, ev, payload):
        if ev == 'passthrough':
            if payload.get('op') == 'chat':
                return self._on_chat(payload)
            return []
        if ev == 'phase':
            return self._on_phase(payload)
        if ev in ('open', 'update'):
            return []
        if ev == 'closed':
            return self._on_window(payload)
        if ev == 'discarded':
            return []               # self-undo: LEGAL, SILENT (L1)
        if ev == 'withdrawn':
            # open-window tail undone pre-close; a retracted '(chosen)'
            # reveal still owes its bookkeeping / the AI's re-show
            if payload.get('op') == 'move' and gesture_match.zone_key(
                    payload.get('from'))[0] == 'hand_chosen':
                items = self._on_reveal_undone(payload)
                if items is not None:
                    return items
            return []
        if ev == 'dropped_dup':
            # a re-rendered twin the segmenter dropped: on the books
            self.findings.append(('render_dup_dropped',
                                  str(payload.get('norm', ''))[:80]))
            return []
        if ev in ('undo', 'redo'):
            return self._on_undo(ev, payload)
        raise DriverHalt('unrouted segmenter event %r' % ev)

    def _mordor_entry_items(self):
        """WAVE-J (ledger A-29/G-04) + WAVE-K R2 (one helper, two
        callers): the FINAL USER RULING 2026-08-16 entry check runs at
        the close of the declaration opportunity, same turn start:
        declared-on-a-gate OR revealed-on-a-gate enters NOW
        (_check_mordor_entry is positional and idempotent - the one
        answerer for all three triggers; the 46 archived one-turn-gap
        sequences are the OLD bug's fingerprint)."""
        import game_logic as gl
        items = []
        fs = self.es.fellowship
        if not getattr(fs, 'mordor_entered', False) \
                and not getattr(self, 'spectator', False):
            pre_p = int(getattr(fs, 'progress', 0) or 0)
            try:
                gl._check_mordor_entry(self.es)
            except Exception as e:
                self.alarms.append('mordor entry check FAILED: '
                                   '%r' % (e,))
            if getattr(fs, 'mordor_entered', False):
                self.findings.append(('mordor_entry',
                                      'turn %s'
                                      % getattr(self.es, 'turn', '?')))
                items.append(('mordor_entry', {'pre_progress': pre_p}))
        return items

    def after_ai_declare(self):
        """WAVE-K R2: the AI-as-FP closed its declaration opportunity
        (declared or declined, committed by live_server): the same
        turn-start continuation the inbound 'fp ok' path runs - the
        Mordor-entry check - then the eye window belongs to the HUMAN
        Shadow (drag or 'Ne' chat; live_server says 'fp ok')."""
        self._close_declare('ai declare step closed')
        self.phase = 'eyes'
        return self._mordor_entry_items()

    # ---- K2: the Fellowship-phase declare menu, armed for WHICHEVER
    # side is FP (human or AI).  ONE engine answerer (game_logic.open/
    # close_declare_phase); the driver only decides WHEN.
    def _declare_settled(self):
        return getattr(self, '_declare_done_turn', None) \
            == getattr(self.es, 'turn', None)

    def _ensure_declare_armed(self):
        """Idempotent: while the turn start sits at 'declare' and this
        turn's declaration has not settled, the engine's declare menu
        is open (a human FP's declaration records then buffer into it;
        the AI-FP's declare step reads it).  An executed action clears
        the engine flag, so re-arming is what keeps the menu alive
        across the AI's own turn-start draws/discard."""
        if self.phase != 'declare' or getattr(self, 'spectator', False):
            return
        if self._declare_settled():
            return
        fs = self.es.fellowship
        if getattr(fs, 'mordor_entered', False):
            return
        if getattr(self.es, '_declare_phase', False):
            return
        import game_logic as gl
        if not hasattr(gl, 'open_declare_phase'):
            raise DriverHalt('ENGINE SURFACE MISSING: game_logic.'
                             'open_declare_phase (K2) - the FP '
                             'declaration cannot be armed')
        gl.open_declare_phase(self.es)

    def _eye_closes_declare(self):
        """Does a HUMAN Shadow eye gesture close the Fellowship phase?
        Only when the FP is the human (their silence is the decline).
        With the AI as FP the declaration is the AI's own pending
        decision: an early eye lands engine-side, the phase stays
        'declare' and after_ai_declare moves it on."""
        import game_state as gs
        return self.human_side is gs.Side.FREE_PEOPLES \
            or self._declare_settled()

    def _close_declare(self, why):
        """The declaration opportunity closed (declared, declined, or
        the AI step finished): settle a still-open decline and close
        the engine flag for this turn."""
        if getattr(self, 'spectator', False):
            return
        self._settle_declare_decline(why)
        import game_logic as gl
        if getattr(self.es, '_declare_phase', False):
            if not hasattr(gl, 'close_declare_phase'):
                raise DriverHalt('ENGINE SURFACE MISSING: game_logic.'
                                 'close_declare_phase (K2)')
            gl.close_declare_phase(self.es)
        self._declare_done_turn = getattr(self.es, 'turn', None)

    # ---- chat ---------------------------------------------------------
    def _on_chat(self, rec):
        if str(rec.get('nick', '')).startswith(self.nick):
            return []               # our own chat echo
        cmd = chat_mod.classify_command(rec.get('body', ''))
        kind = cmd[0] if cmd else None
        if kind == 'fp_ok' and self.phase == 'declare':
            # 'fp ok' closes the declaration opportunity: a human FP
            # who declared has already committed through the declare
            # menu; one who did not has declined (absence-is-the-
            # choice) - settled BEFORE the entry check reads the
            # position (K2)
            self._close_declare("'fp ok'")
            self.phase = 'eyes'
            # WAVE-J (ledger A-29/G-04): the MORDOR ENTRY check - 'fp
            # ok' closes the declaration opportunity, so the check
            # runs HERE, same turn start (ONE helper, shared with the
            # AI-as-FP declare path - WAVE-K R2)
            items = self._mordor_entry_items()
            import game_state as gs
            if self.ai_side is gs.Side.SHADOW:
                # D3: the AI allocates eyes BEFORE any roll; 0 -> '0e'
                items.append(('ai_eyes', None))
            return items
        if kind == 'eyes' and cmd is not None and self.phase in (
                'declare', 'eyes'):
            # Human SHADOW seat.  USER doctrine: chat only for what
            # pieces cannot show.  '0e' = zero eyes (no piece shows
            # zero): closes the declaration opportunity, no allocation.
            # 'Ne' (N>0) is REFUSED with the drag it stands for: the
            # allocation is the physical SA Dice -> Hunt box move (or
            # the client's allocate-eye button), matched by
            # _on_phase_move/_on_eye into _allocate_one_eye - the ONE
            # allocation path.  Allocating in the engine on chat left
            # the client dice in the pool: the human then rolled 7
            # client dice against 6 engine dice (witness mismatch).
            # The AI-Shadow seat never receives this (its own eyes
            # ride the engine menu).
            import game_state as gs
            if self.human_side is gs.Side.SHADOW:
                try:
                    n = int(cmd[1] or 0)
                except (TypeError, ValueError):
                    n = 0
                if n > 0:
                    self.findings.append(('eye_chat_refused',
                                          '%de: drag owed' % n))
                    return [('chat', 'refusal',
                             chat_mod.refusal_eyes_by_board(n))]
                # '0e' against the ENGINE's eye menu (p.17: at least
                # one eye when the Free Peoples retrieved a Hunt Box
                # die) - refused, the window stays open
                try:
                    _counts = self._eye_menu_counts()
                except Exception as _ece:
                    _counts = None
                    self.findings.append(('eye_menu_unavailable',
                                          repr(_ece)[:80]))
                if _counts and min(_counts) > 0 and self.enforce_human:
                    self.findings.append(('eye_chat_refused',
                                          '0e: minimum %d' % min(_counts)))
                    return [('chat', 'refusal', chat_mod.refusal_illegal(
                        'at least %d eye%s must be allocated this turn '
                        '(the Free Peoples retrieved a Hunt Box die) - '
                        'drag %s to the hunt box'
                        % (min(_counts), '' if min(_counts) == 1 else 's',
                           'an eye die' if min(_counts) == 1
                           else 'the eye dice')))]
                if self.phase == 'declare':
                    if self._eye_closes_declare():
                        self._close_declare("'0e' eye chat")
                        self.phase = 'eyes'
                    else:
                        self.findings.append(('eye_before_ai_declare',
                                              '0e chat'))
                else:
                    self.phase = 'eyes'
                return []
        if kind == 'resync':
            # witness FIRST (evidence), then - the USER's adopt-and-
            # keep-playing lever (C-12) - recovery when the report
            # found drift or expectation machinery is stuck
            items = self._witness_report()
            drift = any('desync' in str(i[1]) for i in items
                        if i[0] == 'note')
            stuck = (self.deferred is not None
                     or self.chat_deferred is not None
                     or self.pending_buf is not None
                     or bool(self.uncommitted))
            if drift or stuck:
                items += self.resync_recover()
            return items
        if kind == 'concede' and not getattr(self, 'spectator',
                                             False):
            # WAVE-J (A-10): 'gg'/'concede'/'resign' ends the game -
            # the conceding HUMAN loses; classified inbound since
            # LIFT-C but never routed (the ledger's MISSING A-10).
            # (Spectator: corpus 'gg' chats are the game's own record,
            # not a live concession - byte-pinned census.)
            import game_state as gs
            self.game_over = {
                'victory': ('shadow' if self.ai_side is gs.Side.SHADOW
                            else 'free_peoples'),
                'victory_type': 'concession',
                'message': 'human conceded'}
            self.findings.append(('game_over', 'concession'))
            return [('game_over', dict(self.game_over))]
        # decision answers (combat magnitudes, nc, field battle...)
        # are matched against the OPEN engine menu — never a driver
        # interpretation
        import ai
        if self.chat_deferred is not None:
            # the previous chat decision is still starving on client
            # randomness — a second decision cannot be matched against
            # a menu the first has not yet advanced (loud, not silent)
            return [('note', 'previous chat decision still awaiting '
                             'client randomness — chat ignored')]
        dp = ai.describe_pending(self.es, self.human_side)
        if dp is not None and dp['side'] is self.human_side:
            act = chat_mod.match_choice(rec.get('body', ''),
                                        dp['actions'])
            if act is not None:
                return self._commit_human(act, via='chat')
        if kind == 'attack' and dp is None:
            return self._human_chat_attack(cmd[1])
        if kind == 'attack_from' and dp is None:
            ask = getattr(self, '_attack_ask', None)
            if ask is None:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'name the region to attack first'))]
            return self._human_chat_attack((ask, cmd[1]))
        if kind == 'region' and dp is None \
                and self._is_region_name(cmd[1]):
            return self._human_chat_attack((cmd[1], None))
        if kind == 'pass' and dp is None:
            # WAVE-I (WAVE-H blocker 4): plain-action 'pass' with no
            # pending open, during the human's alternation turn —
            # matched against the ENGINE menu's own pass action (the
            # old 1.223 pass-legality rule lives IN the menu, never a
            # driver counter) and committed; refused otherwise.
            return self._human_plain_pass()
        return []

    def _region_keys(self):
        regs = getattr(self.es.board, '_regions', None) or {}
        return list(regs.keys())

    def _is_region_name(self, text):
        """Does the text name a region - exactly, by possessive fold or
        by a unique short form (chat_mod.resolve_region, the ONE
        resolver)?  Ambiguous short forms count as a region name so the
        attack lane can list the candidates."""
        key, cands = chat_mod.resolve_region(text, self._region_keys())
        return key is not None or len(cands) > 1

    def _human_out_of_turn(self):
        """Is a PLAIN human action (die window, chat attack, pass)
        out of turn?  Whose-turn comes from the ONE engine answerer
        (ai._next_acting_side over who last acted: pendings, ring
        waiver, multi-step dice, dice exhaustion); the driver only
        contributes last_actor.  Shared by the window lane and the two
        chat lanes so the three cannot disagree."""
        import ai
        if self.phase != 'actions' or self.last_actor is None:
            return False
        return ai._next_acting_side(self.es, self.last_actor) \
            is not self.human_side

    def _human_chat_attack(self, spec):
        """'<target> from <source>' typed with the action die STAGED:
        the human's attack declaration (the AI declares the same way).
        Matched against the ENGINE menu's army_attack actions for the
        staged die's face; the battle then owns the open die window -
        its records route to the battle's pendings as they arrive and
        the closing die kill is consumed without a second match."""
        import ai
        from board import sanitize
        target, source = spec
        if self.phase != 'actions':
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'the action phase is not open'))]
        if self._human_out_of_turn():
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'it is not your action turn'))]
        w = self.seg.window
        if w is None or not getattr(w, 'die_face', None)                 or w.die_side != self.human_side.name:
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'stage the action die first, then declare the attack'))]
        if self.rand is not None:
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, self.human_side)
        else:
            menu = ai.get_legal_actions(self.es, self.human_side)
        # short forms / possessives resolve through the ONE resolver;
        # an ambiguous short form is answered with its candidates
        rk = self._region_keys()
        tn, tc = chat_mod.resolve_region(target, rk)
        if tn is None:
            if len(tc) > 1:
                return [('chat', 'refusal', chat_mod.region_ambiguous(
                    target, tc))]
            tn = sanitize(str(target)).strip().lower()
        sn = None
        if source is not None:
            sn, sc = chat_mod.resolve_region(source, rk)
            if sn is None:
                if len(sc) > 1:
                    return [('chat', 'refusal',
                             chat_mod.region_ambiguous(source, sc))]
                sn = sanitize(str(source)).strip().lower()
        cands = [a for a in menu if a.action_type == 'army_attack'
                 and sanitize(str(a.params.get('to', ''))).strip()
                 .lower() == tn
                 and gesture_match._die_filter(
                     self.es, self.human_side, w.die_face)(a)]
        if sn is not None:
            cands = [a for a in cands
                     if sanitize(str(a.params.get('from', '')))
                     .strip().lower() == sn]
        if not cands:
            self._attack_ask = None
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'no legal attack on %s%s with the staged %s die'
                % (tn, (' from %s' % sn) if sn else '',
                   w.die_face)))]
        sources = sorted(set(sanitize(str(a.params.get('from', '')))
                             .strip().lower() for a in cands))
        if len(sources) > 1:
            # underspecified: ask, listing the legal sources; the
            # human answers 'from <region>'
            self._attack_ask = tn
            return [('chat', 'attack_from_ask',
                     chat_mod.attack_from_ask(sources))]
        self._attack_ask = None
        items = self._commit_human(cands[0], via='chat-attack')
        if self.chat_deferred is None:
            self.last_actor = self.human_side
        self._chat_attack_window = w
        self.findings.append(('chat_attack', '%s from %s' % (tn, sn)))
        return items

    def _human_plain_pass(self):
        import ai
        if self.phase != 'actions':
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass(
                'the action phase is not open'))]
        if self._human_out_of_turn():
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass(
                'it is not your action turn'))]
        if self.rand is not None:
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, self.human_side)
        else:
            menu = ai.get_legal_actions(self.es, self.human_side)
        pa = next((a for a in menu if a.action_type == 'pass'), None)
        if pa is None:
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass(
                'the engine menu offers no pass here'))]
        items = self._commit_human(pa, via='chat-pass')
        if self.chat_deferred is None:
            # a committed plain action alternates the turn
            self.last_actor = self.human_side
        return items

    # ---- phase records (turn-start pipeline) ---------------------------
    def _on_phase(self, rec):
        op = rec.get('op')
        if op == 'draw':
            return self._on_draw(rec)
        if op == 'roll_dice':
            return self._on_action_roll(rec)
        if op == 'roll_num':
            if self.rand is not None:
                r = self.rand.feed('d6', rec.get('ints', []),
                                   side=self._d6_roll_side(rec))
                if r.get('dropped'):
                    return [('chat', 'excess_dice',
                             'using the first %d: %s (ignoring %s)'
                             % (len(r['kept']),
                                ' '.join(str(v) for v in r['kept']),
                                ' '.join(str(v)
                                         for v in r['dropped'])))]
            return []
        if op in ('add_eye', 'alloc_eye'):
            return self._on_eye(rec)
        if op == 'move':
            return self._on_phase_move(rec)
        if op == 'turn':
            # the replay header ('<replay> Turn N') - the live client
            # renders no mid-game marker line, so the turn boundary in
            # live is the recover gesture alone (_on_recover); the
            # spectator's own _route resyncs on this record
            return []
        if op == 'choose':
            return self._on_choose(rec)
        if op in ('flip', 'kill', 'die_change', 'random_discard'):
            return []               # engine-derived or router-later
        return []

    def _on_draw(self, rec):
        """A physical draw line.  Attribution by ZONE content: the
        'what' names the deck ('Free Character Card') - the deck names
        the OWNER side (fact 3: never the nick)."""
        what = str(rec.get('what', '')).lower()
        import game_state as gs
        side = (gs.Side.FREE_PEOPLES if what.startswith('free')
                else gs.Side.SHADOW if what.startswith('shadow')
                else None)
        if side is None:
            # non-deck phase draws (hunt tiles, random companions) are
            # RANDOMNESS the engine will ask for: bank them for the
            # funnels (ported, PHASE3-D — a deferred fellowship-move
            # window settles on exactly these records)
            self._tile_face_announce = None
            self._feed_draw_rec(rec)
            lbl = getattr(self, '_tile_face_announce', None)
            if lbl and not getattr(self, 'spectator', False):
                self._tile_face_announce = None
                return [('chat', 'tile_face', lbl)]
            return []
        deck = 'character' if 'character' in what else 'strategy'
        # d6 side-lane binding (ported mechanism 3): the deck names the
        # OWNER, so THIS nick's later roll lines are that side's dice.
        # Runs BEFORE the own-echo return: the AI's !CALL draws carry
        # the LOCAL seat's nick, and seeing that nick draw for the AI
        # side is what poisons a shared nick to untagged.
        # CLIENT-WAVE: the sidecar's zone-proven side is DATA - it
        # feeds the binding when present (the deck-name inference
        # stays for channel-absent lanes; '<game>' d6 lines carry no
        # raw fields, so the bound-nick lane itself remains load-
        # bearing there - reported, not silent).
        from . import sidecar as _sc_mod
        _mside = _sc_mod.meta_side(rec.get('meta'))
        self._bind_roller_nick(rec.get('player'), _mside or side)
        if side is self.ai_side:
            return []               # our own draw echo (fed via decode)
        import game_logic as gl
        if self._card_starved_gesture_open():
            # a human gesture (deferred window / pending buffer / chat
            # decision) is starving on the CARD funnel: this draw is
            # the card it is waiting for, rendered after the window's
            # terminator - it joins the gesture's records (the draw
            # is a fact the match must cover), is fed (placeholder
            # swap) and retried BEFORE the owed-draw lane can misjudge
            # it
            tgt = (self.deferred[0] if self.deferred is not None
                   else self.pending_buf)
            if tgt is not None:
                tgt.records.append(rec)
            self._feed_draw_rec(rec)
            self._retry_blocked()
            self.findings.append(('post_terminator_draw_fed',
                                  '%s %s' % (side.name, deck)))
            return []
        key = (side, deck)
        n = self.turn_draws.get(key, 0)
        owed = (n == 0                              # turn-start draw
                or gl.settle_owed_draw(self.es, side, deck))
        if not owed:
            # a human mistake handled by the client-native restore is
            # a FINDING + notice, never an alarm (the alarm below is
            # for the un-restorable case only)
            self.findings.append(('unowed_draw_refused',
                                  '%s %s' % (side.name, deck)))
            items = [('chat', 'refusal',
                      chat_mod.refusal_illegal('un-owed %s draw'
                                               % deck))]
            # rule 5: the restore is the CLIENT'S OWN undo of the
            # draw line (the newest line; 'draws' is a snapshot verb)
            # - never a forged corrective drag.  Undo-ineligible ->
            # corrective owed, loudly.  The refused line is booked
            # uncommitted so the undo echo is bookkeeping, never a
            # ledger pop of the earlier OWED draw.
            if not getattr(self, 'spectator', False):
                from .undo_model import presses_to_revert
                from emitter import render
                norm = rec.get('norm', '')
                if norm:
                    self.uncommitted.append(norm)
                shim = PhaseWindow()
                shim.records.append(rec)
                n = presses_to_revert(self.feed.lines,
                                      self._window_span(shim),
                                      self.nick)
                if n is None:
                    self.alarms.append('UN-OWED DRAW undo-ineligible '
                                       '- corrective restore owed')
                    self.findings.append(
                        ('unowed_draw_corrective_owed', norm[:60]))
                else:
                    for _ in range(n):
                        items.append(('line', render(
                            'undo', {'nick': self.nick}, local=False)))
                    self.findings.append(('unowed_draw_undone',
                                          '%d press(es)' % n))
            return items
        self._snapshot([rec.get('norm', '')], 'human_draw')
        self.turn_draws[key] = n + 1
        self.hands.human_draw(deck)
        # D4: the hand-limit discard resolves NOW through the engine's
        # own pending (which card = a learned/human decision)
        gl.open_hand_limit_discard(self.es)
        if self.phase == 'draws' \
                and self.turn_draws.get((side, 'character')) \
                and self.turn_draws.get((side, 'strategy')):
            return self._ai_turn_start_draws()
        return []

    def _card_starved_gesture_open(self):
        """Is a HUMAN-side gesture blocked on the card funnel?  Reads
        the starving call the deferral captured (the global pending
        slot may have been clobbered by an interleaved attempt)."""
        if self.deferred is None and self.pending_buf is None \
                and self.chat_deferred is None:
            return False
        p = None
        if self.deferred is not None:
            p = getattr(self.deferred[0], '_starve_call', None)
        if p is None and self.rand is not None:
            p = self.rand.pending()
        if p is None and self.deferred is not None:
            note = str(getattr(self.deferred[0], '_starve_note', '')
                       or '')
            return 'draws:' in note or 'card' in note
        return p is not None and p.kind in ('card', 'draws')

    def _on_action_roll(self, rec):
        """Rendered letters roll: adopt faces onto the roller's pool in
        order (fungible until bound; pid binding rides the raw roll
        line, fact: rendered letters carry no pids)."""
        import game_state as gs
        import game_logic as gl
        faces = rec.get('faces', [])
        side = (gs.Side.SHADOW if 'E' in faces
                else gs.Side.FREE_PEOPLES if 'W' in faces else None)
        if side is None:
            # ambiguous small rolls: match by unused-dice count
            _match = [s for s in (gs.Side.SHADOW, gs.Side.FREE_PEOPLES)
                      if len([d for d in self.es.dice.get(s, [])
                              if not d.used and d.current is None])
                      == len(faces)]
            if len(_match) > 1:
                # BOTH pools fit and no Eye/Will face disambiguates.
                # Behaviour is unchanged (Shadow first, as always) but
                # it must never be SILENT: since the action phase now
                # waits for BOTH rolls (pools_rolled), a wrong-side
                # adoption leaves the real owner unrolled forever and
                # the game simply stops - a mystery deadlock instead
                # of a named fault.  Reachable when the eyes leave the
                # Shadow pool the same size as the FP pool (7-3 == 4)
                # AND the human rolls before the AI's own !CALL lands.
                self.alarms.append(
                    'ROLL SIDE AMBIGUOUS: %d faces fit BOTH pools '
                    '(%s) and no Eye/Will face names the roller - '
                    'adopting %s. If the wrong pool took these faces, '
                    'the other side can never roll and the turn will '
                    'stall.' % (len(faces),
                                '/'.join(s.name for s in _match),
                                _match[0].name))
            side = _match[0] if _match else None
        if side is None:
            self.alarms.append('ROLL side unresolved: %r' % faces)
            return []
        if side is gs.Side.SHADOW and self.human_side is gs.Side.SHADOW \
                and not getattr(self, 'spectator', False):
            # the roll closes the allocation: the engine menu's
            # MINIMUM (p.17: at least one eye when the FP retrieved a
            # die) cannot be forced on the human - it is a loud alarm
            counts = self._eye_menu_counts()
            have = int(getattr(self.es, 'eyes_allocated', 0) or 0)
            if counts and have < min(counts):
                self.alarms.append(
                    'EYE MINIMUM NOT MET: %d allocated, the engine '
                    'menu requires at least %d (the FP retrieved a '
                    'Hunt Box die last turn)' % (have, min(counts)))
                self.findings.append(('eye_minimum_not_met',
                                      '%d < %d' % (have, min(counts))))
        letter_face = {v: k for k, v in
                       gesture_match.FACE_LETTER.items()}
        assigns = []
        idxs = [i for i, d in enumerate(self.es.dice.get(side, []))
                if not d.used and d.current is None]
        if len(faces) != len(idxs):
            # rule 12: a partial adoption must be LOUD — the engine
            # dice roster disagrees with the client's rolled count
            # (bonus-die musters the engine missed, or vice versa)
            self.alarms.append(
                'ROLL adopted %d of %d faces for %s - dice roster '
                'desync' % (min(len(faces), len(idxs)), len(faces),
                            side.name))
        for i, f in zip(idxs, faces):
            fname = letter_face.get(f)
            if fname:
                assigns.append((side, i,
                                getattr(gs.ActionDieFace, fname)))
        gl.adopt_rolled_faces(self.es, assigns)
        # bind the raw roll line's pid order to THE SAME index order
        # (sanctioned raw #2; cbt3: the pool order changes across
        # turns - pre-seed alone mis-labels every die after recover).
        # PARK this adoption's index order; the bind fires now if the
        # raw line already arrived, or retroactively when it does
        # (cbt4: the letters line was polled first).
        self._last_adopt_idxs[side.name] = (list(idxs), len(faces))
        # WAVE-K (R6 / S10): remember WHEN this adoption parked, so a
        # sidecar-present roll whose record carries no die channel
        # (segs[1]) is a LOUD alarm after a few ticks, never a silent
        # unbound pool (the loud-stall rule)
        self._adopt_parked_tick[side.name] = getattr(self, '_tick_n', 0)
        # CLIENT-WAVE adoption: the sidecar's DIE CHANNEL (segs[1] -
        # pids separate from face values, the H7 lesson structural)
        # feeds the SAME parking spot the raw lane used - one binding
        # implementation, the raw regex retired when the channel is
        # present (tick guard).
        from . import sidecar as _sc_mod
        _mp = _sc_mod.roll_pids(rec.get('meta'))
        if _mp is not None:
            self._pending_roll_pids[side.name] = _mp
        self._try_bind_roll(side.name)
        items = []
        if side is gs.Side.SHADOW \
                and self.ai_side is gs.Side.FREE_PEOPLES \
                and all(d.current is None
                        for d in self.es.dice.get(gs.Side.FREE_PEOPLES,
                                                  [])) \
                and not getattr(self, 'spectator', False):
            # WAVE-K R2 (G11: the SHADOW pool rolls FIRST): the human
            # Shadow's roll landed - the AI-as-FP now orders its OWN
            # pool roll (live_server -> '!CALL roll_action_dice
            # Game.2042'); nothing else ever would
            items.append(('ai_roll', None))
        if self.phase in ('roll', 'eyes') \
                and (getattr(self, 'spectator', False)
                     or self.pools_rolled()):
            self.phase = 'actions'
            # WAVE-J (USER live specimen: 'shadow is not supposed to
            # go first' - the AI mustered before the human's first
            # action).  MECHANISM: last_actor CARRIED ACROSS the turn
            # boundary, so whoever closed turn N-1 handed turn N's
            # first action to the OTHER side.  THE RULE (certified
            # AUTO machine, ai.py:15891: `active = Side.FREE_PEOPLES`
            # at every action-resolution start): the FREE PEOPLES
            # side acts FIRST, side-based never seat-based.  Mirror
            # it through the ONE answerer: seeding last_actor=SHADOW
            # makes ai._next_acting_side answer FP.  An idle FP human
            # surfaces via the stall instruments - the AI WAITS
            # (unsticking by rule violation is the old stack's
            # disease).  (Spectator: byte-pinned census - its window
            # matching does not consume the alternation; unseeded.)
            if not getattr(self, 'spectator', False):
                self.last_actor = gs.Side.SHADOW
        elif self.phase == 'eyes':
            # a roll landed but the other pool still owes one: rolling
            # is UNDERWAY, action resolution is not open.  Say so, so
            # the phase name (and the panel line built from it) names
            # the real state instead of leaving it at 'eyes'.
            self.phase = 'roll'
        return items

    def pools_rolled(self):
        """True once BOTH action pools carry rolled faces.

        THE GATE ON ACTION RESOLUTION (USER 2026-08-20 live report:
        "it's not respecting turn order at all").  The rules finish
        the Action Roll for BOTH players before Action Resolution
        opens; we used to flip to 'actions' on whichever roll landed
        first.  That is not a cosmetic ordering slip, because the
        engine's ONE alternation answerer counts a die as available
        only when it carries a ROLLED face:

            _unused(side) = dice with (not d.used and
                                       d.current is not None)

        so an UNROLLED pool is indistinguishable from an EXHAUSTED
        one.  With the AI as Shadow (the launcher's default) the AI's
        pool rolled, the human's did not, `other_left` was 0, and
        _next_acting_side handed the AI EVERY action of the turn:
        the user's feed shows six Shadow dice spent back to back
        (H A C A A C) with no FP action between them and every FP
        face still None.  AUTO never sees this - it rolls both pools
        together - and the one clean sided game on record (wkf3) was
        AI-as-FP, the branch that chains its own second roll.

        An eye-allocated die is `used=True` (game_logic:276), so the
        Shadow eyes never hold this gate shut.  Spectator replay is
        exempt: a corpus log carries both rolls already and its
        census is byte-pinned.

        Pinned by test_action_phase_waits_for_both_rolls."""
        import game_state as gs
        for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            for d in self.es.dice.get(side, []):
                if not d.used and d.current is None:
                    return False
        return True

    def _try_bind_roll(self, side_name):
        """Complete a (raw pid order, adoption index order) pair for
        one side, whichever half arrived last."""
        import game_state as gs
        pids = self._pending_roll_pids.get(side_name)
        parked = self._last_adopt_idxs.get(side_name)
        if pids is None or parked is None:
            return
        idxs, n_faces = parked
        del self._pending_roll_pids[side_name]
        del self._last_adopt_idxs[side_name]
        if len(pids) != n_faces:
            self.alarms.append(
                'ROLL pid binding count mismatch: %d pid(s) vs %d '
                'rolled %s faces - bindings left as-is'
                % (len(pids), n_faces, side_name))
            return
        side = (gs.Side.SHADOW if side_name == 'SHADOW'
                else gs.Side.FREE_PEOPLES)
        # pid[i] <-> the i-th adopted engine index (zip stops at the
        # engine roster when it is shorter - the partial-adoption
        # alarm already fired at adoption)
        for p, i in zip(pids, idxs):
            self.view.bind_die(p, side, i)

    def _bind_roller_nick(self, nick, side):
        """Bind a roller nick to a side from a ZONE-attributed draw
        line (ported mechanism 3).  A nick observed drawing for BOTH
        sides — the one-client sided game, where the client stamps the
        LOCAL seat's nick on the AI's !CALL draws too — is POISONED to
        None (= untagged forever): attribution is zone-derived or
        nothing, never nick-guessed (the ruling).  A poisoned nick
        keeps today's side-blind d6 behavior; distinct-nick games
        (two clients / spectated pairs) get true side lanes."""
        nick = str(nick or '')
        if not nick:
            return
        if nick not in self.nick_side:
            self.nick_side[nick] = side.name
            return
        prev = self.nick_side[nick]
        if prev is not None and prev != side.name:
            self.nick_side[nick] = None       # both-sides nick: poison

    def _d6_roll_side(self, rec):
        """WHO rolled a '<game> X rolls ...' line, when attributable:
        the draw-line nick binding (zone-derived; ported PHASE3-D —
        the spectator's identical override is now redundant).  None
        (unbound or poisoned nick) = untagged, serves any consumer."""
        return self.nick_side.get(str(rec.get('player', '') or ''))

    def _on_eye(self, rec):
        own = str(rec.get('player', '')).startswith(self.nick)
        if self.phase == 'declare' and not own:
            if self._eye_closes_declare():
                self._close_declare('eye allocation began')
                self.phase = 'eyes'
            else:
                self.findings.append(('eye_before_ai_declare',
                                      str(rec.get('norm', ''))[:60]))
        if self.phase in ('draws', 'eyes'):
            self.phase = 'eyes'
        if rec.get('op') == 'alloc_eye' \
                and not str(rec.get('player', '')
                            ).startswith(self.nick):
            # 'allocates an eye.' = ONE Shadow die flipped to Eye and
            # placed in the Hunt Box BEFORE the roll (D3).  Without
            # this the engine keeps 7 unset dice, the rendered roll
            # carries 6 faces, and _on_action_roll cannot resolve the
            # roller (game 2023-01-02 turn 1: menu empty, EVERY window
            # 'none' — the true earliest failure of the first census).
            import game_state as gs
            if self.human_side is gs.Side.SHADOW:
                self._allocate_one_eye(rec)
        return []

    def _eye_menu_counts(self):
        """The ENGINE's eye menu for this turn - the eye_allocate
        counts get_legal_actions offers under the allocation phase
        (p.17: at most one per Companion, at least one when the FP
        retrieved a die) - read on a clone BEFORE this turn's first
        eye lands and held for the turn.  ONE legality answerer for
        the human's allocation; the driver only counts gestures."""
        import ai
        import game_state as gs
        turn = getattr(self.es, 'turn', None)
        if getattr(self, '_eye_counts_turn', None) == turn \
                and getattr(self, '_eye_counts', None) is not None:
            return self._eye_counts
        clone = ai.clone_state(self.es)
        clone._eye_allocation_phase = True
        # the probe asks for the EYE menu alone: the FP declare flag
        # (K2, still armed while the AI-FP's step is queued) precedes
        # it in get_legal_actions and would answer instead
        clone._declare_phase = False
        if self.rand is not None:
            with self.rand.pin():
                acts = ai.get_legal_actions(clone, gs.Side.SHADOW)
        else:
            acts = ai.get_legal_actions(clone, gs.Side.SHADOW)
        counts = set(int((a.params or {}).get('count', 0))
                     for a in acts if a.action_type == 'eye_allocate')
        if not counts:
            # another open pending answered instead of the eye menu:
            # a defect to name, never a reason to refuse every eye
            self.alarms.append('EYE MENU UNAVAILABLE: the engine '
                               'offered no eye_allocate action (%s) - '
                               'eyes land unjudged this turn'
                               % sorted({a.action_type for a in acts}))
            counts = None
        self._eye_counts = counts
        self._eye_counts_turn = turn
        return counts

    def _allocate_one_eye(self, rec=None):
        """Engine-side Phase-3 allocation of ONE eye onto any unset
        unused Shadow die (dice are fungible until faces bind).  Live:
        the running total must be a count the ENGINE's eye menu
        offers, or the gesture is refused (rule 5: client-native undo
        of `rec`'s line + notice, items via _out_queue) and nothing is
        allocated.  Returns True when an eye landed."""
        import game_logic as gl
        import game_state as gs
        dice = self.es.dice.get(gs.Side.SHADOW, [])
        idx = next((i for i, d in enumerate(dice)
                    if not d.used and d.current is None), None)
        if idx is None:
            self.alarms.append('EYE ALLOCATION with no unset unused '
                               'Shadow die - roster desync')
            return False
        counts = (None if getattr(self, 'spectator', False)
                  or not self.enforce_human
                  else self._eye_menu_counts())
        if counts is not None:
            total = int(getattr(self.es, 'eyes_allocated', 0) or 0) + 1
            if total not in counts:
                self.alarms.append(
                    'EYE ALLOCATION REFUSED: %d eye(s) is not in the '
                    "engine's menu %s (one per Companion in the "
                    'Fellowship)' % (total, sorted(counts)))
                self.findings.append(('eye_alloc_refused',
                                      'total=%d menu=%s'
                                      % (total, sorted(counts))))
                items = [('chat', 'refusal', chat_mod.refusal_illegal(
                    'only %s eye(s) may go to the hunt this turn'
                    % ('/'.join(str(c) for c in sorted(counts))
                       or '0')))]
                if rec is not None:
                    from .undo_model import presses_to_revert
                    from emitter import render
                    norm = rec.get('norm', '')
                    if norm:
                        self.uncommitted.append(norm)
                    shim = PhaseWindow()
                    shim.records.append(rec)
                    n = presses_to_revert(self.feed.lines,
                                          self._window_span(shim),
                                          self.nick)
                    if n is None:
                        self.alarms.append('EYE REFUSAL undo-'
                                           'ineligible - corrective '
                                           'restore owed')
                    else:
                        items.extend(('line', render(
                            'undo', {'nick': self.nick}, local=False))
                            for _ in range(n))
                self._out_queue.extend(items)
                return False
        gl.allocate_eyes(self.es, 1, indices=[idx])
        return True

    def _on_phase_move(self, rec):
        """Non-window board moves: recover-dice detection (D1), hunt
        pool maintenance, human eye-allocation drags (fact 12), the
        STANDALONE Elven-Ring walk (no die staged yet)."""
        to = str(rec.get('to', ''))
        frm = str(rec.get('from', ''))
        obj = str(rec.get('obj', ''))
        # CORPUS eye-allocation form (client fact 12; hoisted from the
        # spectator, WAVE-F lane 1): the plain drag 'SA Dice -> The
        # Hunt For The Ring' is the ONLY observed human allocation
        # gesture (0 of 17,693 human logs used Controls.795).  Phase
        # gate keeps post-roll drags of ROLLED eyes out
        # (adopt_rolled_faces already moved those).
        if to.strip().lower() == 'the hunt for the ring' \
                and frm.strip().lower() == 'sa dice' \
                and 'action die' in obj.lower() \
                and self.phase in ('draws', 'declare', 'eyes'):
            import game_state as gs
            if self.human_side is gs.Side.SHADOW \
                    or getattr(self, 'spectator', False):
                # the first Shadow eye gesture closes a HUMAN FP's
                # declaration opportunity (D3 order); an AI-FP's
                # declaration is its own pending (see
                # _eye_closes_declare)
                self._settle_declare_decline('eye allocation began')
                if self.phase == 'declare' \
                        and not getattr(self, 'spectator', False) \
                        and not self._eye_closes_declare():
                    self.findings.append(('eye_before_ai_declare',
                                          obj[:60]))
                else:
                    if self.phase == 'declare':
                        self._close_declare('eye allocation began')
                    self.phase = 'eyes'
                import rendered_replay as rr
                toks = (rr.split_group(obj[len('group ('):-1])
                        if obj.startswith('group (')
                        and obj.endswith(')') else [obj])
                n = 0
                for t in toks:
                    pp = rr.parse_piece(t)
                    n += (pp[3] if pp is not None and pp[0] == 'die'
                          else 1)
                for _ in range(n):
                    if not self._allocate_one_eye(rec):
                        self.findings.append(('eye_alloc_failed',
                                              obj[:80]))
                        break
            else:
                self.alarms.append('Shadow eye drag seen but the AI '
                                   'is Shadow - unexpected gesture')
            return []
        if 'action die' in obj.lower() and to in ('FP Dice',
                                                  'SA Dice') \
                and self.phase == 'actions':
            # phase guard: pre-roll die arrangement (setup, eye returns)
            # must never read as a turn rollover — recover is a
            # turn-END gesture, reachable only from the action phase.
            # The client renders zone Game.2035 (killed/used dice) as
            # 'Current Action', so a bulk recover echoes as 'from
            # Current Action to FP Dice'.  Disambiguation, SOURCE-
            # INDEPENDENT: the self-undo moves exactly ONE die (from
            # Current Action or the Hunt Box - the fellowship-move
            # die sits there) while its side still has unused dice; a
            # multi-die drag, or any pool-ward drag with the side's
            # dice all used, is the recover.
            import game_state as gs
            import rendered_replay as rr
            side = (gs.Side.FREE_PEOPLES if to == 'FP Dice'
                    else gs.Side.SHADOW)
            dice = self.es.dice.get(side, [])
            all_used = bool(dice) and all(d.used for d in dice)
            n_moved = 0
            toks = (rr.split_group(obj[len('group ('):-1])
                    if obj.startswith('group (') and obj.endswith(')')
                    else [obj])
            for t in toks:
                pp = rr.parse_piece(t)
                n_moved += (pp[3] if pp is not None
                            and pp[0] == 'die' else 1)
            if n_moved > 1 or all_used:
                return self._on_recover(rec)
            # D-T2-6: the self-undo shape is only right for a die that
            # is merely STAGED (an open window, nothing in the engine).
            # A die whose action is COMMITTED (the sidecar pid resolves
            # to a USED engine die) - or any die leaving the Hunt Box,
            # where a die sits only after a committed Fellowship move -
            # cannot be "self-undone" by a drag: the engine has moved
            # on.  Live: the recover click with unused dice put one C
            # die back, the boards split, the witness alarmed 60 s
            # later.  REFUSED like the eye drag: client-native undo of
            # this ONE line + a notice; the engine stays whole.
            used = self._dragged_die_used(rec, side)
            from_hunt = _zone_key(frm) == _zone_key(
                'The Hunt For The Ring')
            if used or (used is None and from_hunt):
                n_unused = sum(1 for d in dice if not d.used)
                self.findings.append(('hunt_box_die_refused',
                                      '%s from %s (unused left %d)'
                                      % (obj[:40], frm[:30], n_unused)))
                return self._refuse_phase_line(
                    rec, chat_mod.refusal_illegal(
                        'a used die returns to the pool only at the '
                        'end of the turn (recover) - you still have '
                        '%d unused %s' % (n_unused, 'die' if n_unused
                                          == 1 else 'dice')),
                    'HUNT-BOX DIE')
            # single die back to the pool with unused dice left =
            # the human's self-undo shape (fact 5) — not a rollover
            self.findings.append(('die_self_undo',
                                  '%s from %s' % (obj[:40], frm[:30])))
        return []

    def _dragged_die_used(self, rec, side):
        """True/False: the engine die the dragged client die piece
        (sidecar pid) maps to is used/unused; None when the record
        carries no single pid or the pid maps to no engine die."""
        from . import sidecar as sc_mod
        pids = sc_mod.meta_pids(rec.get('meta'))
        if not pids or len(pids) != 1:
            return None
        pid = str(pids[0])
        from emitter import die_piece_for
        dice = self.es.dice.get(side, [])
        for i, d in enumerate(dice):
            try:
                if str(die_piece_for(self.view, side.name, i)) == pid:
                    return bool(d.used)
            except LookupError:
                return None
        return None

    def _refuse_phase_line(self, rec, notice, label):
        """Refuse ONE non-window phase record: the line is booked
        uncommitted, the client-native undo presses that revert it are
        emitted (the eye-refusal model), the notice rides chat.  An
        undo-ineligible line is a loud alarm (corrective restore
        owed), never a silent drop."""
        from .undo_model import presses_to_revert
        from emitter import render
        if not self.enforce_human:
            return self._lenient_accept(None, label.lower())
        items = [('chat', 'refusal', notice)]
        norm = rec.get('norm', '')
        if norm:
            self.uncommitted.append(norm)
        shim = PhaseWindow()
        shim.records.append(rec)
        n = presses_to_revert(self.feed.lines, self._window_span(shim),
                              self.nick)
        if n is None:
            self.alarms.append('%s REFUSAL undo-ineligible - corrective '
                               'restore owed' % label)
        else:
            items.extend(('line', render('undo', {'nick': self.nick},
                                         local=False))
                         for _ in range(n))
        return items

    def _on_ring_walk(self, rec):
        """A ring marker walked OUTSIDE a die window - the rulebook-
        natural order (use the ring, THEN stage the die).  USER A7 /
        R6: the ring is the engine's `elven_ring` action (a side-wide
        legality waiver; one ring per turn; the same side continues),
        so the walk COMMITS that action against the engine menu on the
        human's own action turn; any target die fits the waiver (the
        Eye variant alone has a physical consequence and is never the
        match of a bare marker walk).  Out of turn, outside the action
        phase, or with no ring use on the menu (none left / one used
        already): refused with the client-native undo of that one line
        (rule 5) and a notice.  The in-window form (ring walked after
        the die was staged) is gesture_match's ring_moves lane."""
        import ai
        from .undo_model import presses_to_revert
        from emitter import render
        norm = str(rec.get('norm', '') or '')
        why = None
        act = None
        if self.phase != 'actions':
            why = 'the action phase is not open'
        elif self._human_out_of_turn():
            why = 'it is not your action turn'
        elif ai.describe_pending(self.es) is not None:
            why = 'a decision is still open'
        else:
            if self.rand is not None:
                with self.rand.pin():
                    menu = ai.get_legal_actions(self.es, self.human_side)
            else:
                menu = ai.get_legal_actions(self.es, self.human_side)
            act = next((a for a in menu if a.action_type == 'elven_ring'
                        and str(a.params.get('new_face', '')).lower()
                        != 'eye'), None)
            if act is None:
                why = 'no elven ring use is legal now (none left, or ' \
                      'one already used this turn)'
        if why is None:
            if self._commit_chain([act], [norm], 'ring'):
                self.findings.append(('ring_standalone', norm[:80]))
                # R6: the ring user acts next (ai._ring_free_side) -
                # last_actor stays the human's
                self.last_actor = self.human_side
                return []
            why = 'the ring use did not apply'
        self.findings.append(('ring_refused', '%s :: %s' % (why, norm[:60])))
        items = [('chat', 'refusal', chat_mod.refusal_illegal(why))]
        if norm:
            self.uncommitted.append(norm)
        shim = PhaseWindow()
        shim.records.append(rec)
        n = presses_to_revert(self.feed.lines, self._window_span(shim),
                              self.nick)
        if n is None:
            self.alarms.append('RING WALK REFUSED but undo-ineligible - '
                               'corrective restore owed')
        else:
            items.extend(('line', render('undo', {'nick': self.nick},
                                         local=False))
                         for _ in range(n))
        return items

    def _on_recover(self, rec):
        """Human recovered dice -> the turn rolled over (D1: trigger
        the AI's own recover, never wait for the marker)."""
        import ai
        import game_logic as gl
        # the turn is ending: whatever the ported mechanisms still
        # hold open resolves NOW or expires LOUDLY, never a ride
        # across the boundary (hoisted from the spectator's rollover)
        if self.chat_deferred is not None:
            self._retry_blocked()
        if self.chat_deferred is not None:
            self._expire_chat_deferred('unresolved at rollover')
        if self.deferred is not None:
            self._retry_blocked()
        if self.deferred is not None:
            self._expire_deferred('unresolved at rollover')
        self._flush_deferred_tail()
        dp = ai.describe_pending(self.es)
        if dp is not None and self._pending_buffer_applies(dp):
            self._try_pending()
            dp = ai.describe_pending(self.es)
        if dp is not None and self._pending_buffer_applies(dp):
            self._expire_pending('unresolved at rollover')
        # WAVE-I: echo-ledger entries never claimed by a client mirror
        # expire at the boundary — LOUD rows, then a clean slate (a
        # stale entry could eat a NEXT turn's real gesture)
        for e in self.echo_ledger:
            self.findings.append(('mirror_echo_unclaimed',
                                  '%(side)s %(kind)s %(tkey)s x%(n)d '
                                  'at %(region)s' % e))
        self.echo_ledger = []
        self._snapshot([rec.get('norm', '')], 'rollover')
        # WAVE-J (A-09, RULING 1.212): MILITARY victory is a TURN-END
        # check - the human's recover IS the turn end.  Check BEFORE
        # the reset; a won game never rolls into a fresh turn.
        if self.check_victory(end_of_turn=True) is not None:
            return [('game_over', dict(self.game_over))]
        # begin_turn_reset SETS state.turn to the passed number — the
        # first cut ALSO incremented afterwards, double-stepping the
        # turn counter (found in the spectator soak, 2026-08-19)
        ended = gl.begin_turn_reset(self.es, self.es.turn + 1,
                                    deal_cards=False,
                                    bonus_dice=False)
        self.phase = 'draws'
        self.turn_draws = {}
        if ended:
            # turn-start corruption increment reached 12 (the donor's
            # Part-38 rollover ring victory) - deliver it, never play on
            self.check_victory(end_of_turn=True)
            if self.game_over is None:
                self.game_over = {'victory': 'shadow',
                                  'victory_type': 'ring_corruption',
                                  'message': 'corruption at rollover'}
                self.findings.append(('game_over',
                                      'rollover corruption'))
            return [('game_over', dict(self.game_over))]
        # J1 (A-11, THIRD occurrence - USER-verified MAJOR both prior
        # times).  MECHANISM of the regression: A12 made the rollover
        # pass bonus_dice=False (correct - the RESET must not grant)
        # BECAUSE the recover step was the ONE granter; the lift
        # carried the False half and dropped the granter, so a mustered
        # Saruman/WK/Mouth or a crowned Aragorn / Gandalf the White
        # grew NOBODY's pool for the whole game - every later menu,
        # eye count and roll ask ran one die short per figure.  The
        # grant runs HERE, on the LIVE state (the donor's stale-state
        # second-order fix), and the server renders the AI's OWN new
        # die pieces out of its reinforcement box (('bonus_dice',
        # pre_counts) item; the human drags their own - forging human
        # drags is the class the live protocol forbids).
        # calculate_bonus_dice adjusts TO the roster size (idempotent:
        # a second call grants nothing - the B13-21 double-grant pin).
        import game_state as gs
        pre_counts = {s.name: len(self.es.dice.get(s, []))
                      for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)}
        try:
            gl.calculate_bonus_dice(self.es)
        except Exception as e:
            self.alarms.append('J1 bonus-dice recalc FAILED: %r' % e)
        items = []
        if len(self.es.dice.get(self.ai_side, [])) \
                > pre_counts.get(self.ai_side.name, 0):
            items.append(('bonus_dice', dict(pre_counts)))
        items.append(('recover_ai', None))
        return items                    # live_server renders the drags

    # ---- windows -------------------------------------------------------
    def _on_window(self, w):
        import ai
        import game_state as gs
        # our own emission echo: verify, never re-match
        stag = w.records[0] if w.records else {}
        if str(stag.get('player', '')).startswith(self.nick):
            return []
        if getattr(self, '_chat_attack_window', None) is w:
            # the battle this window staged was committed by the
            # chat declaration; the die kill only closes it
            self._chat_attack_window = None
            self.window_stats['matched'] += 1
            return []
        out = []
        if self.phase != 'actions' \
                and not getattr(self, 'spectator', False):
            # action resolution opens only once BOTH pools rolled
            # (pools_rolled -> phase 'actions'); a die staged before
            # that is out of turn order - refused (rule 5: client-
            # native undo + notice), never matched and never an
            # alternation seed
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            self.window_stats['none'] += 1
            self.findings.append(('window_before_action_phase',
                                  'phase=%s :: %s'
                                  % (self.phase,
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:80])))
            res = gesture_match.MatchResult(
                'none', note='the action phase is not open - both '
                             'action pools must roll first')
            out.extend(self._refuse(w, res))
            return out
        if w.die_side != self.human_side.name:
            self.alarms.append('window staged a %s die during the '
                               'human turn' % w.die_side)
        # a NEW window closing means the client moved on: anything
        # still blocked is now attributable — retry once, then expire
        # visibly (ported PHASE3-D deferral discipline)
        if self.chat_deferred is not None:
            self._retry_blocked()
        if self.chat_deferred is not None:
            self._expire_chat_deferred('unresolved at next window')
        if self.deferred is not None:
            self._retry_blocked()
        if self.deferred is not None:
            self._expire_deferred('rng never arrived before the next '
                                  'window')
            self._flush_deferred_tail()
        # WAVE-H cbt5: a human window arriving while the AI's OWN
        # multi-step die is open (pending_army_move_die/muster - the
        # describe_pending coverage gap, GATED engine item) would be
        # matched against the SECOND-LEG-gated menu and mis-judged
        # (live: deferred on an unrelated hunt-roll candidate).  The
        # engine says it is still the AI's turn - HOLD the window
        # loudly instead of matching it against a poisoned menu.
        if self.last_actor is self.ai_side and (
                self.es.pending_army_move_die is not None
                or getattr(self.es, 'pending_muster_die', None)
                is not None):
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            if not self.enforce_human:
                return out + self._lenient_accept(
                    w, 'window while the AI die is still open')
            self.window_stats['hold'] += 1
            self.holds.append((w, gesture_match.MatchResult(
                'hold', note='AI multi-step die still open - out of '
                             'turn')))
            self.alarms.append('WINDOW WHILE AI MULTI-STEP OPEN - '
                               'held (out of turn)')
            return out + [('note', 'window held: the AI is still '
                                   'resolving its die')]
        if self._human_out_of_turn() \
                and not getattr(self, 'spectator', False):
            # rulebook p.20: the players ALTERNATE actions.  A second
            # human die window while the engine names the other side
            # (the AI's reply still in flight, or its pending open) is
            # REFUSED like the chat lanes refuse it - matched, it
            # committed and re-seeded last_actor, so the AI then took
            # two actions in a row (SA, SA, FP, FP) with no note.
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            self.window_stats['none'] += 1
            self.findings.append(('window_out_of_turn',
                                  'last_actor=%s :: %s'
                                  % (self.last_actor.name,
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:80])))
            res = gesture_match.MatchResult(
                'none', note='it is not your action turn')
            out.extend(self._refuse(w, res))
            return out
        dp = ai.describe_pending(self.es)
        if dp is not None and self._pending_buffer_applies(dp):
            self._try_pending()
            dp = ai.describe_pending(self.es)
        if dp is not None:
            # matching against a pending-gated menu would mis-refuse a
            # fine gesture: expire a stuck HUMAN pending (loud), hold
            # the window either way.  An AI-side pending is the step
            # machinery's to resolve — never expired from here.
            human_pending = self._pending_buffer_applies(dp)
            if human_pending:
                self._expire_pending('unresolved at next window')
                dp2 = ai.describe_pending(self.es)
                if dp2 is None:
                    # the expiry settled it (prefix match / declare
                    # decline / unique no-op): the window proceeds
                    dp = None
                else:
                    dp = dp2
        if dp is not None and not self.enforce_human:
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            return out + self._lenient_accept(
                w, 'window while %s pending (%s) is open'
                % (dp['kind'], dp['side'].name))
        if dp is not None:
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            self.window_stats['hold'] += 1
            msg = ('WINDOW WHILE PENDING (%s %s) - held'
                   % (dp['kind'], dp['side'].name))
            if msg not in self.alarms:
                self.alarms.append(msg)
            self.holds.append((w, gesture_match.MatchResult(
                'hold', note='pending %s unresolved' % dp['kind'])))
            out.append(('note', 'window held: pending %s unresolved'
                        % dp['kind']))
            if self._pending_buffer_applies(dp) \
                    and not getattr(self, 'spectator', False):
                out.extend(self._pending_reminder(dp))
            return out
        # a card play NAMES its title: the human card became PUBLIC —
        # swap the hand placeholder for the real cid BEFORE matching,
        # so the engine menu carries the play_event candidate.  Combat
        # cards chosen through the client chooser reveal into WHICHEVER
        # hand the chosen-zone names (the defender picks inside the
        # attacker's window); only modeled hands hold placeholders.
        # The engine's battle context classifies a chooser reveal
        # (K1: combat card vs chooser event play).
        w.combat_cards_open = self._combat_cards_open()
        pre_facts = gesture_match.window_facts(w)
        w._reveals = getattr(w, '_reveals', [])
        for title in pre_facts.cards_played:
            cid = self.hands.reveal(title)
            if cid is None:
                self.alarms.append('card reveal failed for %r'
                                   % (title,))
            else:
                # tracked so a FAILED window rolls the swap back
                # (_unreveal — WAVE-I: the base refusal paths had no
                # reveal rollback at all)
                w._reveals.append((self.human_side, cid))
        for tag, title in pre_facts.combat_cards:
            cc_side = (gs.Side.FREE_PEOPLES if tag == 'FP'
                       else gs.Side.SHADOW)
            hm = self.hand_models.get(cc_side)
            if hm is not None:
                cid = hm.reveal(title)
                if cid is None:
                    self.alarms.append('combat card reveal failed '
                                       'for %r' % (title,))
                else:
                    w._reveals.append((cc_side, cid))
        # the window's own recorded randomness, fed BEFORE matching
        # (ported mechanism 1)
        self._feed_window_rng(w)
        self._match_window_now(w, self.human_side)
        return out

    # ---- rng-fed matching + deferral (ported PHASE3-D, mechanism 1) ----
    def _attempt_pin(self):
        """Context-manager factory for one matching attempt: fresh
        cursors (begin_attempt) + the funnel-diverted real-trajectory
        scope (advance), so every candidate replays the same fed
        randomness."""
        rand = self.rand

        @contextlib.contextmanager
        def _cm():
            rand.begin_attempt()
            with rand.advance():
                yield
        return _cm

    def _feed_window_rng(self, w):
        """The window's own recorded randomness, fed BEFORE matching so
        every candidate attempt replays the same values (begin_attempt
        resets the cursors per candidate)."""
        for rec in w.records:
            op = rec.get('op')
            if op == 'roll_num':
                try:
                    self.rand.feed('d6', rec.get('ints', []),
                                   side=self._d6_roll_side(rec))
                except Exception as e:
                    self.findings.append(('rng_d6_feed', repr(e)))
            elif op == 'move':
                frm = str(rec.get('from', '')).lower()
                if 'hunt pool' in frm:
                    try:
                        self.rand.feed('tile', str(rec.get('obj', '')))
                    except Exception as e:
                        self.findings.append(('rng_tile_feed',
                                              repr(e)))
            elif op == 'draw':
                self._feed_draw_rec(rec)

    def _feed_draw_rec(self, rec):
        """One 'draws X.' record's randomness, by WHAT was drawn:
        deck cards -> minted placeholder fed as 'card' (identity is
        hidden until a later reveal swaps it); hunt tiles -> 'tile';
        random companions -> 'companion'.  The placeholder is SWAPPED
        into the fictional engine deck first — counts stay conserved
        and the funnel's deck-membership check (a real desync witness)
        keeps its teeth.  Deck draws for a side with NO hand model
        here (the AI's own draws in live mode) are SKIPPED: their
        identities ride the commitment decode, never a placeholder."""
        import game_state as gs
        import rendered_replay as rr
        what = str(rec.get('what', '') or '').strip()
        wl = what.lower()
        side = (gs.Side.FREE_PEOPLES if wl.startswith('free')
                else gs.Side.SHADOW if wl.startswith('shadow')
                else None)
        if side is not None and wl.endswith('card'):
            hm = self.hand_models.get(side)
            if hm is None:
                return          # AI-side identity: commitment decode
            deck = 'character' if 'character' in wl else 'strategy'
            ph = hm._mint(deck)
            attr = '%s_%s_deck' % (
                'fp' if side is gs.Side.FREE_PEOPLES else 'sh',
                'char' if deck == 'character' else 'strat')
            deck_list = getattr(self.es, attr, None)
            if not deck_list:
                self.findings.append(('rng_card_feed',
                                      'engine %s empty' % attr))
                return
            # swap: one fictional REAL cid out, the placeholder in —
            # the engine draw then removes the placeholder (deck -1).
            # Never pop a placeholder an earlier draw of the same
            # gesture already swapped in (two draws in one play: the
            # second swap used to eat the first's placeholder and the
            # engine halted 'deck desync' on the real draw).
            real = next((i for i in range(len(deck_list) - 1, -1, -1)
                         if not str(deck_list[i]).startswith('HU_')),
                        None)
            if real is None:
                self.findings.append(('rng_card_feed',
                                      'engine %s holds only '
                                      'placeholders' % attr))
                return
            deck_list.pop(real)
            deck_list.append(ph)
            try:
                self.rand.feed('card', (side.name, deck, ph))
            except Exception as e:
                self.findings.append(('rng_card_feed', repr(e)))
            return
        pp = rr.parse_piece(what)
        if pp is not None and pp[0] == 'tile':
            try:
                self.rand.feed('tile', what)
            except Exception as e:
                self.findings.append(('rng_tile_feed', repr(e)))
            # WAVE-J (ledger A-06, corpus A7): the tile-face announce
            # after an AI-side draw - the AUTHORITATIVE face
            # correction ('1r'/'eye'; plain damage numbers excluded
            # by the grammar, they collide with hit counts).  Banked
            # for the caller (only the live phase-draw path emits;
            # window-replayed feeds never re-announce).
            import re as _re
            lbl = _re.sub(r'[^a-z0-9]+', ' ',
                          str(what).lower()).strip()
            if _re.fullmatch(r'-?\d+r', lbl) or lbl == 'eye':
                self._tile_face_announce = lbl
            return
        if pp is not None and pp[0] == 'named':
            # random companion draw: feed BOTH the canonical name and
            # the rendered alias (dual-sided figures)
            try:
                self.rand.feed('companion', (pp[1], what))
            except Exception as e:
                self.findings.append(('rng_companion_feed', repr(e)))

    def _commit_chain(self, chain, w_norms, label, gesture=None):
        """Re-run a matched chain in ONE fresh attempt so the committed
        state and the consumed-feed cursors belong to the matched
        candidate, not to whichever candidate the matcher probed last.
        Ends with the unconsumed-randomness HONESTY GATE: randomness
        the client produced that the committed step did not call for
        is a finding, and the feed is quarantined (the pregame
        coin flip was feeding hunt rolls before this gate).
        `gesture` (WAVE-I): the committed Window/PhaseWindow — its
        expressed rows subtract from the post-commit forced sweep."""
        import ai
        self.rand.begin_attempt()
        pre = self.es
        post = self.es
        try:
            with self.rand.advance():
                for a in chain:
                    post = ai.apply_action(post, a)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            self.findings.append(('commit_replay_halt',
                                  '%r on %r' % (e, chain)))
            self.alarms.append('COMMIT REPLAY HALT (%s): %r'
                               % (label, e))
            self.uncommitted.extend(w_norms)
            self.rand.reset_feed()
            return False
        self._snapshot(w_norms, label, post=post)
        self.es = post
        for hm in self.hand_models.values():
            hm.es = post
        if label in ('pending:declare', 'declare-decline'):
            # this turn's declaration opportunity is spent (K2)
            self._declare_done_turn = getattr(post, 'turn', None)
        # G2 (WAVE-H): a combat round resolved inside this chain
        # announces its hit counts (engine statement)
        from . import emission as em_mod
        self._out_queue.extend(
            ('chat', k, t) for k, t in
            em_mod.round_hits_chat(post, self.human_side))
        # WAVE-I: engine events the gesture did not express — forced
        # allocations of the AI's own units + settlement-capture
        # markers (live-only inside; no-op for the spectator)
        self._out_queue.extend(
            self.post_commit_sweep(pre, post, gesture=gesture,
                                   label=label))
        exc = self.rand.unconsumed()
        self.rand.step_commit()
        if any(exc.values()):
            self.findings.append(('unconsumed_rng',
                                  '%r after %s' % (
                                      {k: v for k, v in exc.items()
                                       if v}, label)))
            self.rand.reset_feed()
        return True

    # ---- WAVE-I: post-commit engine-event sweep -------------------------
    def post_commit_sweep(self, pre, post, gesture=None, expressed=None,
                          label=''):
        """Blockers 1+2 (one producer per commit path): after ANY
        committed apply,
          (1) AI-SIDE unit-removal/relocation rows the gesture (or the
              AI's own emission ops) did NOT express render as the
              AI's OWN lines via the bounded row renderer
              (emission.render_forced_rows; ambiguity -> HOLD + ALARM,
              never a guess) and are banked on the echo ledger;
          (2) HUMAN-side residual removal rows are BANKED un-emitted
              (the human renders their own casualties; their incoming
              mirror records are then echoes, not gestures);
          (3) settlement-capture control/VP marker walks render from
              the engine's enemy_controlled / calculate_victory_points
              (emission.render_capture_markers, ledger-reconciled).
        Returns driver items.  Spectator: no client to keep honest —
        skipped entirely (census byte-identity)."""
        if getattr(self, 'spectator', False):
            return []
        from . import emission as em_mod
        items = []
        ai_tag = 'shadow' if self.ai_side.name == 'SHADOW' else 'fp'
        hu_tag = 'fp' if ai_tag == 'shadow' else 'shadow'
        try:
            rows_ai = em_mod.removal_rows(pre, post, ai_tag)
            rows_hu = em_mod.removal_rows(pre, post, hu_tag)
        except Exception as e:
            self.alarms.append('FORCED-ROW SWEEP READ FAILED (%s): %r'
                               % (label, e))
            rows_ai = rows_hu = None
        if rows_ai is not None:
            sub_ai = sub_hu = None
            blocked = False
            if gesture is not None:
                sub_ai, blk_a = self._gesture_rows(gesture, ai_tag)
                sub_hu, blk_h = self._gesture_rows(gesture, hu_tag)
                blocked = blk_a
            if expressed is not None:
                sub_ai, dec_fail = em_mod.expressed_rows(
                    self.view, expressed, ai_tag)
                sub_hu, _ = em_mod.expressed_rows(self.view,
                                                  expressed, hu_tag)
                blocked = blocked or dec_fail
            res_ai = em_mod.subtract_rows(rows_ai, sub_ai)
            res_hu = em_mod.subtract_rows(rows_hu, sub_hu)
            if not em_mod.rows_empty(res_ai):
                if blocked:
                    self.alarms.append(
                        'FORCED AI ROWS UNRENDERED (%s): expressed-row '
                        'decode failed - refusing to risk a double '
                        'kill; RESYNC will show the drift' % label)
                    self.findings.append(('forced_rows_hold',
                                          '%s: decode-blocked'
                                          % label))
                else:
                    em, ledger = em_mod.render_forced_rows(
                        self.view, self.nick, self.ai_side, res_ai,
                        pre, label=label)
                    if em.hold:
                        self.alarms.append('FORCED AI ROWS HOLD (%s): '
                                           '%s' % (label, em.hold))
                        self.findings.append(('forced_rows_hold',
                                              str(em.hold)[:200]))
                    else:
                        items.extend(('line', l) for l in em.lines)
                        for row in ledger:
                            self.echo_ledger.append(row)
                        if em.lines:
                            self.findings.append(
                                ('forced_rows_emitted',
                                 '%s: %d line(s)' % (label,
                                                     len(em.lines))))
            # HUMAN residual removals: bank only (their pieces, their
            # gesture) — NET removals per type: a pure RELOCATION
            # (minus offset by an equal plus elsewhere — e.g. a leader
            # riding an army move the generic-leader subtraction
            # cannot attribute) is NOT a removal and must never sit on
            # the ledger (wir3: a banked leader-move minus expired as
            # a spurious mirror_echo_unclaimed row every turn)
            def _bank_net(slot, kind, tkey):
                net = (sum(n for _, n in slot['minus'])
                       - sum(n for _, n in slot['plus']))
                left = max(0, net)
                for reg, n in slot['minus']:
                    if left <= 0:
                        break
                    take = min(n, left)
                    left -= take
                    self.echo_ledger.append(
                        {'kind': kind, 'side': hu_tag,
                         'region': reg, 'tkey': tkey, 'n': take})
                # WAVE-J: NET ARRIVALS bank too (kind '<k>_in') — the
                # forced-downgrade replacement regular.  _bank_net
                # used to DROP the plus rows outright, so the client
                # board could never receive the piece the engine
                # minted: a guaranteed resync desync on every forced
                # downgrade of a human-side unit (reproduction:
                # test_forced_downgrade_arrival_echo).  A pure
                # relocation (minus == plus) still banks NOTHING on
                # either side (the wir3 leader-move rule).
                left_in = max(0, -net)
                for reg, n in slot['plus']:
                    if left_in <= 0:
                        break
                    take = min(n, left_in)
                    left_in -= take
                    self.echo_ledger.append(
                        {'kind': kind + '_in', 'side': hu_tag,
                         'region': reg, 'tkey': tkey, 'n': take})
            for tkey, slot in res_hu['units'].items():
                _bank_net(slot, 'unit', tkey)
            _bank_net(res_hu['naz'], 'naz', 'nazgul')
            for nm in res_hu['named']['leaves']:
                lv = res_hu['named']['leaves'].get(nm, [])
                jn = res_hu['named']['joins'].get(nm, [])
                for reg in lv[len(jn):]:
                    self.echo_ledger.append(
                        {'kind': 'named', 'side': hu_tag,
                         'region': reg, 'tkey': nm, 'n': 1})
        # (3) capture markers — engine event, ledger-reconciled
        em2, notes = em_mod.render_capture_markers(
            self.view, self.nick, post, self.ctrl_markers,
            self.vp_marks)
        for n in notes:
            self.findings.append(('capture_marker', n))
            if 'EXHAUSTED' in n or 'failed' in n:
                self.alarms.append('CAPTURE MARKER: %s' % n)
        items.extend(('line', l) for l in em2.lines)
        return items

    def _gesture_rows(self, gesture, side_tag):
        """The rows a committed gesture's records EXPRESSED, in
        removal_rows space, for subtraction.  Returns (rows, blocked):
        blocked=True when the gesture touched generic-leader rows in a
        way this converter cannot attribute (the sweep then refuses to
        render rather than risk a double kill)."""
        from . import emission as em_mod
        from . import emission_card as emc
        rows = {'units': {}, 'naz': {'minus': [], 'plus': []},
                'named': {'leaves': {}, 'joins': {}},
                'cas': {}, 'unexplained': []}
        try:
            facts = gesture_match.window_facts(gesture)
        except Exception:
            return rows, True
        for (reg, tkey), v in facts.armies.items():
            if not v or em_mod._row_side(tkey) != side_tag:
                continue
            slot = rows['units'].setdefault(
                tkey, {'minus': [], 'plus': []})
            if v < 0:
                slot['minus'].append((reg, -v))
            else:
                slot['plus'].append((reg, v))
        for (nm, reg), v in facts.named.items():
            if not v:
                continue
            _pid, pside = emc._named_pid_side(nm)
            want = 'SHADOW' if side_tag == 'shadow' else 'FREE_PEOPLES'
            if pside != want:
                continue
            b = 'joins' if v > 0 else 'leaves'
            for _ in range(abs(v)):
                rows['named'][b].setdefault(nm, []).append(reg)
        blocked = False
        for reg, v in facts.leaders.items():
            if not v:
                continue
            # generic leader rows (nazgul + plain leaders share the
            # client token space): attribute the SHADOW ones to nazgul
            # (the only fungible shadow leader); FP generic-leader
            # gestures cannot be attributed per nation here — block
            # the sweep for safety when FP leader rows exist
            if side_tag == 'shadow':
                b = 'plus' if v > 0 else 'minus'
                rows['naz'][b].append((reg, abs(v)))
            else:
                blocked = True
        return rows, blocked

    def _consume_mirror_echo(self, rec):
        """A non-self kill/removal record FULLY covered by open echo-
        ledger entries is the client-side mirror of an effect the
        engine already committed (the cbt10 race) — consume it as an
        echo, never a new gesture.  Partial/unknown coverage returns
        False (normal routing; loud paths there)."""
        if not self.echo_ledger or getattr(self, 'spectator', False):
            return False
        import rendered_replay as rr
        op = rec.get('op')
        suffix = ''      # WAVE-J: '' = removal rows, '_in' = arrivals
        if op == 'kill':
            zk = gesture_match.zone_key(rec.get('zone'))
            if zk[0] not in ('region', 'stronghold'):
                return False
            reg = zk[1]
            toks = list(rec.get('pieces', []))
        elif op == 'move':
            src = gesture_match.zone_key(rec.get('from'))
            dst = gesture_match.zone_key(rec.get('to'))
            if src[0] in ('region', 'stronghold') \
                    and dst[0] in ('dead', 'reinf'):
                reg = src[1]
            elif src[0] in ('reinf', 'dead') \
                    and dst[0] in ('region', 'stronghold'):
                # WAVE-J: the human mirrors a banked ARRIVAL (forced-
                # downgrade replacement) by dragging the piece in
                # from a reserve zone — echo-consumable against
                # '<kind>_in' entries ONLY (a region->region move is
                # never an arrival echo)
                reg = dst[1]
                suffix = '_in'
            else:
                return False
            obj = str(rec.get('obj', '') or '')
            toks = (rr.split_group(obj[len('group ('):-1])
                    if obj.startswith('group (') and obj.endswith(')')
                    else [obj])
        else:
            return False
        want = []
        for t in toks:
            pp = rr.parse_piece(t)
            if pp is None:
                return False
            kind, a, b, count = pp
            if kind == 'unit':
                nat = gesture_match._nation_enum(a)
                if nat is None:
                    return False
                import game_state as gs
                uk = (gs.UnitKind.REGULAR if b == 'REGULAR'
                      else gs.UnitKind.ELITE)
                want.append(('unit' + suffix, reg,
                             gesture_match._unit_type_key(nat, uk),
                             count))
            elif kind == 'nazgul':
                want.append(('naz' + suffix, reg, 'nazgul', count))
            elif kind == 'leader':
                nat = gesture_match._nation_enum(a)
                if nat is None:
                    return False
                want.append(('unit' + suffix, reg, '%s:leader'
                             % str(getattr(nat, 'value', nat)).lower(),
                             count))
            elif kind == 'named':
                if suffix:
                    return False   # named arrivals are never banked
                want.append(('named', reg, a, 1))
            else:
                return False
        # coverage check on a COPY; adopt only when every row covers
        ledger = [dict(e) for e in self.echo_ledger]
        for kind, reg, tkey, count in want:
            need = count
            for e in ledger:
                if need <= 0:
                    break
                if e['kind'] == kind and e['region'] == reg \
                        and e['tkey'] == tkey and e['n'] > 0:
                    take = min(need, e['n'])
                    e['n'] -= take
                    need -= take
            if need > 0:
                return False
        self.echo_ledger = [e for e in ledger if e['n'] > 0]
        self.findings.append(('mirror_kill_echo',
                              '%s :: %s' % (op,
                                            str(rec.get('norm',
                                                        ''))[:100])))
        return True

    def _match_window_now(self, w, side):
        """Match (attempt-pinned) and settle one window.  Outcome items
        go through self._out_queue — this runs from nested retry sites
        whose returns are discarded, and refusal undo presses must
        still reach the client."""
        res = gesture_match.match_window(
            self.es, side, w, pin=self._attempt_pin(), rng=self.rctl,
            advance_tolerance=not getattr(self, 'spectator', False),
            fs_prefilter=not getattr(self, 'spectator', False))
        w_norms = [r.get('norm', '') for r in w.records]
        if res.status == 'match':
            if self._commit_chain(res.chain, w_norms, 'window',
                                  gesture=w):
                self.window_stats['matched'] += 1
                self.last_actor = side      # alternation bookkeeping
            else:
                self.window_stats['hold'] += 1
                self._unreveal(w)
                self._out_queue.append(
                    ('note', 'window commit failed - held (see '
                             'findings)'))
            return []
        if res.status == 'hold' and res.starved:
            # the gesture's randomness renders AFTER the window (hunt
            # rolls, tile draws): wait for it, retry on each rng feed.
            # the starve note rides the window so _expire_deferred can
            # pick the right engine-gap shim (WAVE-I hoist)
            w._starve_note = res.note
            # WAVE-J (wjg11 live: the randomcomp ask NEVER went out -
            # the global rand.pending() was CLEARED by an interleaved
            # attempt before the tick's ask gate read it, and the
            # deferral carried only the note TEXT).  The starving CALL
            # itself rides the window; the ask gate and waiting_on
            # fall back to it when the global slot was clobbered.
            w._starve_call = (getattr(res, 'starve_call', None)
                              or (self.rand.pending()
                                  if self.rand is not None else None))
            self.deferred = (w, side)
            self._out_queue.append(
                ('note', 'window deferred: %s' % (res.note or
                                                  'awaiting client '
                                                  'randomness')))
            return []
        self.uncommitted.extend(w_norms)
        if res.status in ('hold', 'ambiguous') and not self.enforce_human:
            self.window_stats['none'] += 1
            self._out_queue.extend(self._lenient_accept(
                w, res.note or res.status))
            self.rand.reset_feed()
            return []
        if res.status in ('hold', 'ambiguous'):
            self.window_stats[res.status if res.status in
                              self.window_stats else 'hold'] += 1
            self.holds.append((w, res))
            self.findings.append((res.status,
                                  '%s die=%s(%s) n_recs=%d :: %s'
                                  % (side.name, w.die_face,
                                     res.note or '', len(w.records),
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:100])))
            self._unreveal(w)
            self._out_queue.append(('note', 'holding: %s'
                                    % (res.note or res.status)))
            self.rand.reset_feed()
            return []
        # 'none': the live refusal doctrine (rule 5) — client-native
        # undo over the gesture window + chat notice
        self.window_stats['none'] += 1
        # EXPLAIN BEFORE THE ROLLBACK.  _unreveal() swaps the played
        # card back out of the hand, so a menu derived after it can
        # never contain the play_event the human actually attempted -
        # the first cut of this explainer ran inside _refuse (after the
        # unreveal) and confidently named an unrelated army move as the
        # 'closest legal move'.  A diagnosis has to be taken from the
        # state the MATCHER judged, not from the state we rolled back to.
        explain = self._explain_refusal(w) if not res.note else None
        self._unreveal(w)
        self._out_queue.extend(self._refuse(w, res, explain))
        self.rand.reset_feed()
        return []

    def _expire_chat_deferred(self, why):
        """A deferred CHAT decision whose randomness never arrived
        before the client moved on: LOUD expiry (the engine still holds
        the pending; nothing was committed)."""
        action, via = self.chat_deferred
        self.chat_deferred = None
        p = self.rand.pending() if self.rand is not None else None
        self.findings.append((
            'chat_deferred_expired',
            '%s %r (%s, pending=%s)'
            % (via, getattr(action, 'action_type', action), why,
               p.kind if p is not None else None)))
        self.alarms.append('DEFERRED CHAT DECISION EXPIRED (%s): %r'
                           % (why, getattr(action, 'action_type',
                                           action)))
        if self.rand is not None:
            self.rand.reset_feed()

    def _reveal_buffer_cards(self, buf):
        """Swap hand placeholders for the real cids the BUFFER names,
        so the pending's menu can offer them.  Idempotent: a cid
        already revealed is not re-revealed, and the tracking list is
        what _unreveal rolls back on a failed match."""
        import game_state as _gs
        buf._reveals = getattr(buf, '_reveals', [])
        try:
            f = gesture_match.window_facts(buf)
        except Exception:
            return
        already = {c for _s, c in buf._reveals}
        for tag, title in list(getattr(f, 'combat_cards', ()) or ()):
            cc_side = (_gs.Side.FREE_PEOPLES if tag == 'FP'
                       else _gs.Side.SHADOW)
            hm = self.hand_models.get(cc_side)
            if hm is None:
                continue
            cid = hm.reveal(title)
            if cid is None:
                self.alarms.append('combat card reveal failed for %r '
                                   '(pending buffer)' % (title,))
            elif cid not in already:
                buf._reveals.append((cc_side, cid))

    def _unreveal(self, w):
        """A reveal-swap belongs to ITS window's commit: when the
        window fails to commit, the real cid must leave the engine
        hand again (a stranded reveal consumed the deck's placeholder
        budget and poisoned every later menu build — game 2023-01-13:
        Kindred stayed in hand after its window failed, Celeborn's
        reveal then found no strategy placeholder left, and the
        stranded cid's dry-run starved the card funnel on every
        subsequent enumeration).  (WAVE-I hoist from the spectator —
        the base refusal paths had NO reveal rollback at all; live
        hand models may cover one side only, hence the .get.)"""
        for r_side, cid in getattr(w, '_reveals', []):
            hm = self.hand_models.get(r_side)
            if hm is None:
                continue
            hand = self.es.hands[r_side].hand
            if cid in hand:
                import game_logic as gl
                dk = gl.card_deck_of(cid) or 'strategy'
                hand[hand.index(cid)] = hm._mint(dk)
        w._reveals = []

    def _retry_skip_draw_grants(self, w, side):
        """ENGINE GAP (reported, engine FROZEN): optional in-battle
        draw grants (the Witch-king 'Sorcerer' draw, ai._resolve_wk_
        sorcerer -> grant_draw) fire UNCONDITIONALLY, but the rules
        say MAY draw — a client that never drew starves every
        candidate chain on the grant's NeedClientCall re-raise.  ONE
        retry with grant_draw stubbed to the decline (ctx-scoped,
        restored in finally); the census row keeps the engine debt on
        the books.  Fix owed: optional draw grants as LEARNED
        decisions (same family as the p.40 hunt re-roll).
        (WAVE-I hoist from the spectator, body verbatim.)"""
        import game_logic as gl
        skipped = []
        orig = gl.grant_draw

        def _decline(state, side_, deck):
            skipped.append((getattr(side_, 'name', side_), deck))
        n_matched = self.window_stats['matched']
        gl.grant_draw = _decline
        try:
            self._match_window_now(w, side)
        finally:
            gl.grant_draw = orig
        if self.window_stats['matched'] > n_matched:
            self.findings.append((
                'engine_gap_draw_grant_declined',
                '%s die=%s skipped grants %r (shim-committed; engine '
                'fix owed: optional draw grant as a learned decision)'
                % (side.name, w.die_face, skipped)))
            return True
        if self.deferred is not None:
            # still starving even without the grants: a real hold
            self.deferred = None
            self.window_stats['hold'] += 1
            self.findings.append((
                'hold_engine_gap_draw_grant',
                '%s die=%s :: %s'
                % (side.name, w.die_face,
                   ([r.get('norm', '') for r in w.records][1:2]
                    or [''])[0][:100])))
            self._unreveal(w)
            self.rand.reset_feed()
            return True
        # a plain 'none' retry recorded itself
        self.rand.reset_feed()
        return True

    def _expire_deferred(self, why):
        """A deferred window whose randomness never arrived before the
        client demonstrably moved on: attribute it LOUDLY.  Shim
        retries for the two engine gaps on the books (optional draw
        grants; the unconditional p.40 re-roll — decline-equivalent
        natural-1 synthesis).  (WAVE-I: ONE implementation — the
        WAVE-F growth the spectator carried is hoisted here; live-only
        alarms stay gated on the spectator flag so the census stays
        byte-identical.)"""
        w, side = self.deferred
        self.deferred = None
        w_norms = [r.get('norm', '') for r in w.records]
        self.uncommitted.extend(w_norms)
        p = self.rand.pending()
        note = getattr(w, '_starve_note', '') or ''
        # R5: the shim retries (declined draw grants, decline-
        # equivalent re-roll synthesis) are CORPUS-CENSUS tools - the
        # live path never commits a human gesture with synthesized
        # randomness; an expired live deferral is a loud uncommitted
        # window, nothing else.
        spectator = getattr(self, 'spectator', False)
        if spectator and 'draws:' in note \
                and self._retry_skip_draw_grants(w, side):
            return
        if spectator and p is not None and p.kind == 'd6' \
                and p.kw.get('ctx') in ('hunt reroll',
                                        'leader reroll'):
            # ENGINE GAP (reported, engine FROZEN): game_logic.
            # _hunt_roll takes the p.40 hunt re-roll UNCONDITIONALLY,
            # but the re-roll is the Shadow player's OPTIONAL choice —
            # this client declined it (the next window already opened
            # with no re-roll line), so the engine starves on a die
            # the client never rolled.  ONE retry with the decline-
            # equivalent shim (natural 1 = byte-identical outcome to
            # declining); the finding row keeps the engine debt on the
            # books.  Fix owed: re-roll take/decline as a LEARNED
            # SHADOW DECISION.
            self.rand._pending = None
            n_matched = self.window_stats['matched']
            self.rand.reroll_decline_shim = True
            try:
                self._match_window_now(w, side)
            finally:
                self.rand.reroll_decline_shim = False
            if self.window_stats['matched'] > n_matched:
                self.findings.append((
                    'engine_gap_hunt_reroll_declined',
                    '%s die=%s :: %s (shim-committed; engine fix '
                    'owed: optional re-roll as a Shadow decision)'
                    % (side.name, w.die_face,
                       (w_norms[1] if len(w_norms) > 1
                        else w_norms)[:100])))
                return
            if self.deferred is not None:
                # still starving even with the shim: a real hold
                self.deferred = None
                self.window_stats['hold'] += 1
                self.findings.append(('hold_engine_gap_hunt_reroll',
                                      '%s die=%s :: %s'
                                      % (side.name, w.die_face,
                                         (w_norms[1] if len(w_norms)
                                          > 1 else w_norms)[:100])))
                self._unreveal(w)
            # a plain 'none' retry already recorded itself
        else:
            self.window_stats['none'] += 1
            self.findings.append(('deferred_expired',
                                  '%s die=%s (%s, pending=%s, %s) :: %s'
                                  % (side.name, w.die_face, why,
                                     p.kind if p is not None else None,
                                     getattr(w, '_starve_note', ''),
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:100])))
            self._unreveal(w)
            if not getattr(self, 'spectator', False):
                # LIVE loudness (beyond the spectator's census row): an
                # expired deferral means the human's gesture is now
                # uncommitted — engine and client have parted
                self.alarms.append('DEFERRED WINDOW EXPIRED (%s): '
                                   '%s die=%s'
                                   % (why, side.name, w.die_face))
        self.rand.reset_feed()

    def _retry_blocked(self):
        """New client randomness arrived: give a deferred chat action,
        then the pending buffer, then a deferred window, another
        attempt."""
        import ai
        if self.chat_deferred is not None:
            # BEFORE the dp check: the starved chat action's OWN
            # pending (e.g. defender_combat_card) is still open, so
            # the dp branch below would swallow the retry forever
            action, via = self.chat_deferred
            self.chat_deferred = None
            items = self._attempt_chat_action(action, via)
            if items:
                self._out_queue.extend(items)
            if self.chat_deferred is not None:
                return              # still starving: re-armed inside
        dp = ai.describe_pending(self.es)
        if dp is not None:
            if self._pending_buffer_applies(dp):
                self._try_pending()
            return
        if self.deferred is not None:
            w, side = self.deferred
            self.deferred = None
            self._match_window_now(w, side)
            # a starved re-hold re-arms the slot (set inside); a
            # settle releases whatever queued behind the deferral
            if self.deferred is None:
                self._flush_deferred_tail()

    def _flush_deferred_tail(self):
        """Replay the records queued behind a deferral: pendings the
        commit opened consume them; the rest reach the normal phase
        handlers.  Non-empty handler returns ride the out queue."""
        import ai
        tail, self.deferred_tail = self.deferred_tail, []
        for rec in tail:
            if self.deferred is not None:
                self.deferred_tail.append(rec)      # re-blocked
                continue
            dp = ai.describe_pending(self.es)
            if dp is not None and self._pending_buffer_applies(dp):
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf.records.append(rec)
                self._try_pending()
            else:
                out = self._route_event('phase', rec)
                if out:
                    self._out_queue.extend(out)

    # ---- pending phase-buffer (ported PHASE3-D, mechanism 2) -----------
    @staticmethod
    def _rec_inert(rec):
        """A record that EXPRESSES nothing in fact space (control-
        marker bookkeeping, 'kills nothing', political-activation
        flips): it can never resolve a pending — buffering it JAMMED
        the declare buffer on game 2023-01-07's turn-1 control-marker
        drag and cascaded the whole game."""
        pw = PhaseWindow()
        pw.records.append(rec)
        f = gesture_match.window_facts(pw)
        return f.is_empty() and not f.unmapped

    @staticmethod
    def _is_dice_bookkeeping(rec):
        """True for move/kill records whose pieces are ALL action
        dice (recover drags, pool maintenance)."""
        def _dieish(tok):
            return 'action die' in str(tok).lower()
        if rec.get('op') == 'move':
            obj = str(rec.get('obj', '') or '')
            if obj.startswith('group (') and obj.endswith(')'):
                import rendered_replay as rr
                toks = rr.split_group(obj[len('group ('):-1])
            else:
                toks = [obj]
            return bool(toks) and all(_dieish(t) for t in toks)
        if rec.get('op') == 'kill':
            pcs = rec.get('pieces', [])
            return bool(pcs) and all(_dieish(p) for p in pcs)
        return False

    def _pending_menu(self, dp):
        """The pending's own action menu, prefiltered by identity when
        the buffered gesture named one (discard lane: identity-less
        deck-label kills prefer PLACEHOLDER discards — a revealed real
        cid would have rendered by name)."""
        menu = dp['actions']
        if dp['kind'] == 'discard' and self.pending_buf is not None:
            f = gesture_match.window_facts(self.pending_buf)
            if f.hand_discards and not f.cards_played:
                import game_logic as gl
                decks = set(dk for (_sd, dk) in f.hand_discards)
                ph = [a for a in menu
                      if str(a.params.get('card_id', ''))
                      .startswith('HU_')
                      and gl.card_deck_of(
                          str(a.params.get('card_id', ''))) in decks]
                if ph:
                    menu = ph
        return self._drop_unevidenced_random(dp, menu)

    # A hunt casualty the CLIENT draws: its victim is generated at
    # apply time, so no already-rendered record can identify it.
    _CLIENT_DRAWN_TYPES = ('hunt_sacrifice_random',)

    def _drop_unevidenced_random(self, dp, menu):
        """Never let a gesture MATCH a client-drawn random casualty.

        USER 2026-08-21: "it also drew a random companion without me
        saying so ... i moved the fellowship to get revealed and then
        was going to kill the guide but the ai decided that meant
        random".

        Mechanism: while a HUMAN-side hunt pending is open, incidental
        board records (the reveal flip, the Fellowship move) buffer
        into a PhaseWindow and are matched against the pending's own
        menu.  hunt_sacrifice_random is on that menu, and merely
        APPLYING it calls game_logic.pick_random_companion -> the
        'randomcomp' funnel -> '!CALL random_companion'.  So a
        speculative match does not just guess the FP's decision, it
        EXECUTES a draw in the client before the human has spoken.

        It is also unmatchable in principle: the victim is drawn by
        the client when the action is applied, so a record that has
        ALREADY rendered cannot evidence it - unless the human used
        the client's built-in draw first, in which case the victim
        DID render and shows up as a companion leaving the Fellowship
        (the engine then adopts it via forced_companion).

        So: keep the entry only when the buffer positively names a
        departing companion.  Otherwise the FP must say it - the chat
        lane already accepts 'kill the guide' / 'random'.
        """
        if dp.get('kind') != 'hunt':
            return menu
        evidenced = False
        if self.pending_buf is not None:
            try:
                f = gesture_match.window_facts(self.pending_buf)
                evidenced = bool(getattr(f, 'comps_gone', None))
            except Exception:
                evidenced = False
        if evidenced:
            return menu
        kept = [a for a in menu
                if a.action_type not in self._CLIENT_DRAWN_TYPES]
        if len(kept) != len(menu):
            self.findings.append(
                ('random_casualty_not_matchable',
                 'held %d client-drawn option(s); the FP must choose'
                 % (len(menu) - len(kept))))
        return kept or menu

    def _try_pending(self):
        """Match the buffered phase records against the open pending's
        menu; commit on full coverage.  Returns True when the pending
        cascade fully settled."""
        import ai
        dp = ai.describe_pending(self.es)
        if dp is None:
            self.pending_buf = None
            return True
        if self.pending_buf is None or not self.pending_buf.records:
            return False
        # REVEAL BEFORE THE MENU IS READ (USER 2026-08-20: "scouts is
        # nonfunctional ... it still waits for my decision").  The
        # human's hand holds HU_ placeholders, and a placeholder has no
        # card identity, so ai.describe_pending could only ever offer
        # `combat_card_none` for a defender-combat-card decision -- the
        # human could NEVER play a combat card, and the buffer waited
        # for a match that was not in the menu.  _on_window already
        # does exactly this swap for gesture windows (:1515-1537); the
        # PENDING-buffer path never got it.  Reveals are tracked on the
        # buffer so a failed match rolls them back (_unreveal), because
        # a stranded reveal poisons every later menu build.
        self.pending_buf.combat_cards_open = self._combat_cards_open()
        self._reveal_buffer_cards(self.pending_buf)
        if getattr(self.pending_buf, '_reveals', None):
            dp = ai.describe_pending(self.es) or dp
        res = gesture_match.match_window(
            self.es, dp['side'], self.pending_buf,
            menu=self._pending_menu(dp), pin=self._attempt_pin(),
            rng=self.rctl,
            advance_tolerance=not getattr(self, 'spectator', False),
            fs_prefilter=not getattr(self, 'spectator', False))
        if res.status != 'match':
            return False    # incomplete buffer / rng still owed: wait
        buf = self.pending_buf
        norms = [r.get('norm', '') for r in buf.records]
        if self._commit_chain(res.chain, norms,
                              'pending:%s' % dp['kind'],
                              gesture=buf):
            self.pending_stats['matched'] += 1
            self.pending_buf = None
            return True
        # the commit failed: the buffer's reveal-swaps must leave the
        # engine hand again, or the consumed placeholder budget
        # poisons every later menu build (see _unreveal)
        self._unreveal(buf)
        self.pending_buf = None
        return False

    def _settle_declare_decline(self, why):
        """The game moved past the declaration window with no declare
        gesture: the FP answer is DECLINE (absence-is-the-choice) —
        commit the declare menu's own 'pass'.  No-op unless the engine
        declare flag is armed (K2: _ensure_declare_armed in live, the
        spectator's own draw handler there)."""
        import ai
        if not getattr(self.es, '_declare_phase', False):
            return
        if self.pending_buf is not None and self.pending_buf.records:
            return          # declare gestures still buffering
        dp = ai.describe_pending(self.es)
        if dp is None or dp['kind'] != 'declare':
            return
        pa = next((a for a in dp['actions']
                   if a.action_type == 'pass'), None)
        if pa is None:
            self.findings.append(('declare_no_pass', why))
            return
        if self._commit_chain([pa], [], 'declare-decline'):
            self.pending_stats['matched'] += 1

    def _try_pending_prefixes(self):
        """PREFIX consumption, at EXPIRY only (the client has moved
        on, so the buffer is COMPLETE): the buffer may hold SEVERAL
        sequential pending cascades' records (game 2023-01-22: two
        hunts' consequences rendered as one cleanup batch) — commit
        the longest prefix that covers the CURRENT pending, keep the
        tail for the next, loop.  Returns True when the cascade fully
        settled.  (WAVE-I hoist from the spectator, body verbatim.)"""
        import ai
        while True:
            dp = ai.describe_pending(self.es)
            if dp is None:
                if self.pending_buf is not None \
                        and not self.pending_buf.records:
                    self.pending_buf = None
                return True
            buf = self.pending_buf
            if buf is None or not buf.records:
                return False
            recs = buf.records
            menu = self._pending_menu(dp)
            hit_k = None
            for k in range(len(recs), 0, -1):
                shim = PhaseWindow()
                shim.records = recs[:k]
                shim.combat_cards_open = getattr(
                    buf, 'combat_cards_open', None)
                res = gesture_match.match_window(
                    self.es, dp['side'], shim, menu=menu,
                    rng=self.rctl, stop_kinds=(),
                    fs_reset_relax=True,
                    advance_tolerance=not getattr(self, 'spectator',
                                                  False),
                    fs_prefilter=not getattr(self, 'spectator',
                                             False))
                self._last_pending_match = '%s:%s' % (
                    res.status, getattr(res, 'note', '') or '')
                if res.status == 'match':
                    norms = [r.get('norm', '') for r in recs[:k]]
                    if self._commit_chain(res.chain, norms,
                                          'pending:%s' % dp['kind'],
                                          gesture=shim):
                        self.pending_stats['matched'] += 1
                        hit_k = k
                    break
            if hit_k is None:
                return False
            buf.records = recs[hit_k:]
            if not buf.records:
                self.pending_buf = None

    def _replay_records(self, recs):
        """Re-route records that were buffered against the WRONG
        pending (jam hostages): pendings now open consume them; the
        rest reach the normal phase handlers.  Mirror-tracking (the
        spectator's) already happened on first arrival — route via
        LiveDriver._route directly.  (WAVE-I hoist, body verbatim.)"""
        import ai
        for rec in recs:
            if ai.describe_pending(self.es) is not None:
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf.records.append(rec)
                self._try_pending()
            else:
                LiveDriver._route(self, 'phase', rec)

    def _expire_pending(self, why):
        """An open pending the feed never resolved before the client
        moved on: prefix consumption first (the buffer is complete),
        then the silence-resolutions (declare decline, unique-empty-
        delta no-op — absence IS the choice), the jam-breaking forces
        (placeholder discard; second-empty-expiry force with draws+d6
        stubbed), and finally the LOUD expiry row.  (WAVE-I: ONE
        implementation, hoisted from the spectator's WAVE-F/G growth;
        the live alarm stays gated on the spectator flag.)"""
        import ai
        # the client has moved on: the buffer is complete — prefix
        # consumption may settle multi-cascade batches now
        if self.pending_buf is not None and self.pending_buf.records:
            if self._try_pending_prefixes():
                return
        dp = ai.describe_pending(self.es)
        buf = self.pending_buf
        if dp is not None and dp['kind'] == 'declare' \
                and not (buf and buf.records):
            self._settle_declare_decline(why)
            if ai.describe_pending(self.es) is None:
                return
            dp = ai.describe_pending(self.es)
        if dp is not None and not (buf and buf.records):
            # the client moved on rendering NO gesture for this
            # pending: the answer is the pending's own NO-OP — the
            # unique candidate whose observable delta is EMPTY (guide
            # KEEP, combat_round_result STOP...).  Absence-is-the-
            # choice, same class as the 0-eye/declare-decline
            # absences; committed only when exactly ONE empty-delta
            # candidate exists (never guess between distinct no-ops).
            base_fp = gesture_match._facts_fingerprint(
                gesture_match.delta_facts(self.es, self.es))
            pin = self._attempt_pin()
            quiet = []
            for a in dp['actions']:
                try:
                    with pin():
                        post = ai.apply_action(self.es, a)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    continue
                if gesture_match._facts_fingerprint(
                        gesture_match.delta_facts(self.es, post)) \
                        == base_fp:
                    quiet.append(a)
            if len(quiet) == 1 and self._commit_chain(
                    quiet, [], 'pending:silent-noop'):
                self.pending_stats['matched'] += 1
                self.findings.append(
                    ('pending_silent_noop',
                     '%s: no client gesture -> %s'
                     % (dp['kind'], quiet[0].action_type)))
                self.pending_buf = None
                return
        buf_facts = (gesture_match.window_facts(buf)
                     if (buf and buf.records) else None)
        # R5: from here on the JAM-BREAKERS - a forced placeholder
        # discard and the second-empty-expiry force with draws + d6
        # stubbed - are CORPUS-CENSUS tools only.  A live human
        # pending is never force-committed: it stays open (the engine
        # still holds it), the buffer's records are reported, and the
        # window lane reminds the human once.
        if not getattr(self, 'spectator', False):
            self._expire_pending_live(dp, buf, why)
            return
        if dp is not None and dp['kind'] == 'discard' \
                and (buf_facts is None
                     or not buf_facts.hand_discards):
            # a hand-limit pending with NO client discard gesture: the
            # ENGINE hand is bigger than the client's real one (unowed
            # grants or upstream cascade).  Left open it JAMS: every
            # later phase record buffers into it and every window rows
            # 'window_while_pending' (measured 30-40 such rows per
            # game; the Hollin reveal chain died hostage in one).
            # Force the least-wrong settle — a PLACEHOLDER discard,
            # restoring hand-size parity toward the client's reality —
            # one loud row per forced card, then REPLAY any hostage
            # records so the chain they express gets its match.
            forced = 0
            for _ in range(8):
                dpf = ai.describe_pending(self.es)
                if dpf is None or dpf['kind'] != 'discard':
                    break
                acts = dpf['actions'] or []
                pick = next(
                    (a for a in acts
                     if str((a.params or {}).get('card_id', ''))
                     .startswith('HU_')), None) \
                    or (acts[0] if acts else None)
                if pick is None or not self._commit_chain(
                        [pick], [], 'pending:discard-forced'):
                    break
                forced += 1
            if forced:
                self.findings.append((
                    'discard_forced',
                    '%d placeholder discard(s) forced - engine hand '
                    'over limit with no client discard gesture (%s)'
                    % (forced, why)))
                hostages = list(buf.records) if buf else []
                self.pending_buf = None
                if hostages:
                    self._replay_records(hostages)
                if ai.describe_pending(self.es) is None \
                        or self.pending_buf is not None:
                    return
        key = (dp['kind'] if dp else '?',
               dp['side'].name if dp else '?')
        if not (buf and buf.records) and dp is not None \
                and getattr(self, '_expire_streak', None) == key:
            # SECOND consecutive empty-buffer expiry of the same
            # pending: it will never resolve from records (upstream
            # cascade or a declined optional grant) and left open it
            # JAMS every later window (game 2023-01-22: one hunt
            # pending held 24 windows hostage).  Force the first
            # commit-able option — draws STUBBED (the client never
            # drew: the Palantir deck-choice pending starved every
            # option and sprayed 40 halt rows, game 2023-01-28) AND
            # d6 STUBBED to natural 1s (a card_option whose EVERY
            # branch rolls dice re-jammed forever: 32 'card effect'
            # halt alarms, game 2023-01-31 'Ringwraiths are Abroad'
            # class — the client resolved the card invisibly, the
            # force must not need randomness the log never carries) —
            # one loud row; the divergence lands in the resync.
            import combat as cb
            import game_logic as gl
            n_find = len(self.findings)
            n_alarm = len(self.alarms)
            g0, d0 = gl.grant_draw, gl.draw_event_card
            r0, r1 = gl.roll_d6_batch, cb.roll_d6_batch

            def _ones(n, *a, **k):
                return [1] * int(n)
            gl.grant_draw = lambda *a, **k: None
            gl.draw_event_card = lambda *a, **k: None
            gl.roll_d6_batch = _ones
            cb.roll_d6_batch = _ones
            try:
                forced_a = None
                for a in dp['actions'] or []:
                    if self._commit_chain([a], [], 'pending:forced'):
                        forced_a = a
                        break
            finally:
                gl.grant_draw, gl.draw_event_card = g0, d0
                gl.roll_d6_batch, cb.roll_d6_batch = r0, r1
            # attempt noise (starved options) is not window evidence
            self.findings[n_find:] = [
                f for f in self.findings[n_find:]
                if f[0] != 'commit_replay_halt']
            self.alarms[n_alarm:] = [
                a for a in self.alarms[n_alarm:]
                if 'COMMIT REPLAY HALT' not in a]
            if forced_a is not None:
                self.findings.append((
                    'pending_forced',
                    'kind=%s side=%s -> %s %s (second empty '
                    'expiry - unjammed; draws+d6 stubbed)'
                    % (key[0], key[1], forced_a.action_type,
                       str(forced_a.params)[:60])))
                self._expire_streak = None
                self.pending_buf = None
                return
        self._expire_streak = (key if not (buf and buf.records)
                               else None)
        self.pending_stats['expired'] += 1
        card = ''
        if dp and dp.get('actions'):
            p0 = dp['actions'][0].params or {}
            nm = p0.get('card_name') or p0.get('card_id')
            if nm:
                card = ' card=%s' % str(nm)[:40]
        self.findings.append((
            'pending_unresolved',
            'kind=%s side=%s buf=%d (%s)%s :: %s'
            % (dp['kind'] if dp else '?',
               dp['side'].name if dp else '?',
               len(buf.records) if buf else 0, why, card,
               [r.get('norm', '')[:60]
                for r in (buf.records[:2] if buf else [])])))
        if buf is not None:
            self.uncommitted.extend(
                r.get('norm', '') for r in buf.records)
        self.pending_buf = None
        self.rand.reset_feed()

    def _expire_pending_live(self, dp, buf, why):
        """LIVE tail of _expire_pending (R5): the human's pending
        STAYS OPEN in the engine; its unmatched buffer records are
        reported (census row + sticky alarm) and released to
        uncommitted.  Nothing is committed on the human's behalf."""
        n_buf = len(buf.records) if (buf and buf.records) else 0
        if dp is None:
            self.pending_buf = None
            return
        self.pending_stats['expired'] += 1
        self.findings.append((
            'pending_unresolved',
            'kind=%s side=%s buf=%d (%s) :: %s'
            % (dp['kind'], dp['side'].name, n_buf, why,
               [r.get('norm', '')[:60]
                for r in (buf.records[:2] if n_buf else [])])))
        msg = ('HUMAN DECISION HELD (%s): %s still open - %d '
               'unmatched gesture record(s) released; resolve it '
               '(board/chat) or undo' % (why, dp['kind'], n_buf))
        if msg not in self.alarms:
            self.alarms.append(msg)
        if n_buf:
            self.uncommitted.extend(
                r.get('norm', '') for r in buf.records)
        self.pending_buf = None
        self.rand.reset_feed()

    def _pending_reminder(self, dp):
        """At most ONE chat reminder per (held pending, commit epoch)
        naming the owed decision, through a registered builder of the
        chat vocabulary (the held window IS a refusal: nothing was
        applied).  Returns driver items."""
        key = (dp['kind'], dp['side'].name, self.n_commits_total)
        if getattr(self, '_reminded_pending', None) == key:
            return []
        self._reminded_pending = key
        kind = str(dp['kind'])
        if kind.endswith('_combat_card'):
            what = 'combat card'
        else:
            what = kind.split(':')[-1].replace('_', ' ')
        try:
            text = chat_mod.pending_reminder(what)
        except Exception as e:
            self.findings.append(('pending_reminder_failed',
                                  repr(e)[:80]))
            return []
        return [('chat', 'refusal', text)]

    # Decision options that COMMIT NOTHING: declining a card, passing,
    # taking no optional effect.  A skip may only ever choose from
    # these (or from a pending that offers exactly one action, which is
    # forced anyway) - it must never spend a card, a die or a unit on
    # the player's behalf.
    _SKIP_SAFE_TYPES = (
        'combat_card_none', 'rearguard_none', 'pass',
        'decline', 'card_option_none', 'no_effect',
        'hunt_effect_none', 'guide_ability_none',
    )

    def skip_open_decision(self):
        """Answer the open pending with its DECLINE option (USER
        2026-08-20: "I want it to be able to skip the decision").

        The turn buttons only move last_actor, so a stuck decision -
        the combat-card pending the USER sat on - could not be got past
        at all.  This is the escape hatch, and it is deliberately
        narrow: it picks a NON-COMMITTAL option, never one that spends
        a card, a die or a unit; if the pending has no such option and
        is not a single forced action, it REFUSES and says why rather
        than guessing an answer for the player.

        Returns (items, message)."""
        import ai
        dp = ai.describe_pending(self.es)
        if dp is None:
            return [], 'nothing to skip - no decision is open'
        acts = list(dp.get('actions') or ())
        if not acts:
            return [], ('the %s decision offers no actions at all'
                        % dp['kind'])
        pick = next((a for a in acts
                     if a.action_type in self._SKIP_SAFE_TYPES), None)
        if pick is None:
            # a card_option pending carries its decline as the option
            # whose choices are ALL falsy ('balrog_declare_draw':
            # False).  There is no '_none' action type to key on, so
            # the decline is recognised by its content - and only a
            # wholly-falsy option qualifies, so a skip can still never
            # choose to spend anything.
            for a in acts:
                ch = (a.params or {}).get('choices')
                if isinstance(ch, dict) and ch \
                        and not any(bool(v) for v in ch.values()):
                    pick = a
                    break
        if pick is None and len(acts) == 1:
            pick = acts[0]          # forced: skipping changes nothing
        if pick is None:
            return [], ('the %s decision has no decline option - it '
                        'needs a real answer (%s)'
                        % (dp['kind'],
                           ', '.join(sorted({a.action_type
                                             for a in acts}))[:80]))
        who = dp['side']
        items = self._commit_human(pick, via='panel-skip')
        msg = ('skipped %s\'s %s decision with %s'
               % (who.name, dp['kind'], pick.action_type))
        return items, msg

    # Every field ai._pending_decision_kind_side inspects.  A reset
    # that missed one would leave the game just as stuck, so this list
    # is taken FROM that answerer, not guessed.
    _PENDING_FIELDS = (
        'pending_combat', 'pending_hunt', 'pending_guide_change',
        'pending_card_option', 'pending_card_followup',
        'pending_discard', '_pending_hunt_damage', '_declare_phase',
        '_eye_allocation_phase', '_reveal_place_due',
        'pending_army_move_die', 'pending_muster_die',
    )

    def reset_and_resync(self):
        """THE UNSTICK BUTTON (USER 2026-08-20: "a button to resync and
        reset and clear all pending decisions, so the AI continues from
        there").

        Order matters and is deliberate:
          1. DECLINE what can legally be declined (skip_open_decision,
             bounded) - a clean answer keeps the engine consistent and
             is always preferable to erasing state;
          2. FORCE-CLEAR whatever pendings remain, naming each one;
          3. drop the driver's own open state (deferred windows, the
             chat deferral, a half-collected pending buffer - with its
             reveals given back, or the placeholder budget leaks);
          4. RESYNC: adopt the client's board into the view;
          5. hand the turn to the AI so play resumes.

        This CAN erase a decision mid-battle - that is what the USER
        asked for and it is their game - so every step is reported and
        the report goes to the panel and the console.  Returns
        (items, report)."""
        import ai
        report = []
        items = []
        for _ in range(12):
            # DEFENSIVE ON PURPOSE: this button exists for games that
            # are already broken, and a malformed pending makes
            # describe_pending itself raise (a guide-change dict with
            # no 'candidates', say).  If asking what is pending throws,
            # there is nothing to decline cleanly - fall straight
            # through to the force-clear, which is what the player
            # pressed the button for.
            try:
                if ai.describe_pending(self.es) is None:
                    break
                got, msg = self.skip_open_decision()
            except Exception as e:
                report.append('could not read the open decision (%s) '
                              '- clearing it outright' % (repr(e)[:60],))
                break
            if not got:
                break               # nothing declinable left
            items.extend(got)
            report.append(msg)
        # 2. anything still owed is erased by name
        for f in self._PENDING_FIELDS:
            v = getattr(self.es, f, None)
            if v:
                report.append('cleared %s' % f)
                setattr(self.es, f,
                        False if isinstance(v, bool) else None)
        # 3. the driver's own open state
        if self.pending_buf is not None:
            self._unreveal(self.pending_buf)
            self.pending_buf = None
            report.append('dropped the pending buffer')
        if self.deferred is not None:
            self.deferred = None
            report.append('dropped the deferred window')
        if self.chat_deferred is not None:
            self.chat_deferred = None
            report.append('dropped the deferred chat decision')
        # USER 2026-08-22 ('if something breaks people can easily reset
        # it and continue'): every other open expectation goes too
        if self.holds:
            report.append('dropped %d held window(s)' % len(self.holds))
            self.holds = []
        if self.deferred_tail:
            report.append('dropped %d deferred record(s)'
                          % len(self.deferred_tail))
            self.deferred_tail = []
        if getattr(self, 'human_cc_committed', False):
            self.human_cc_committed = False
            report.append('dropped the hidden combat-card commit')
        self._attack_ask = None
        self._chat_attack_window = None
        self._eye_counts = None
        self._last_blocked_kind = None
        self._last_pending_match = None
        from .undo_model import RestoreWatch as _RW
        self.restore_watch = _RW()
        if self.alarms:
            report.append('cleared %d alarm(s)' % len(self.alarms))
            self.alarms = []
        if self.rand is not None:
            try:
                self.rand.reset_feed()
            except Exception:
                pass
        # 4. adopt the client's board
        try:
            # raw driver items - live_server dispatches them, exactly
            # as it does for the plain resync lever
            items.extend(self.resync_recover(drain_feed=True))
            report.append('resynced to the client board')
        except Exception as e:
            report.append('resync failed: %r' % (e,))
        # 5. the AI acts next
        self.last_actor = self.human_side
        report.append('turn handed to the AI')
        return items, report

    def _commit_human(self, action, via='gesture'):
        """One chat-committed human decision, under the SAME attempt/
        advance treatment as windows (WAVE-H): the old pin()-scoped
        apply ran at advance-depth 0 where the funnels are INERT - a
        defender's 'nc' that fires the combat round would have rolled
        ENGINE dice silently.  A starved apply defers (retried on each
        rng-bearing record; ask surfaced for live_server to order)."""
        return self._attempt_chat_action(action, via)

    def _attempt_chat_action(self, action, via):
        """`action`: an engine Action (ai.apply_action on a clone) or
        a CALLABLE taking the clone to mutate (the K1 hidden-commit /
        reveal engine calls) - one transactional commit path."""
        import ai
        from . import emission as em_mod
        from . import randomness as rnd_mod

        def _apply(state):
            if callable(action) and not hasattr(action, 'die_idx'):
                clone = ai.clone_state(state)
                action(clone)
                return clone
            return ai.apply_action(state, action)
        if self.rand is None:
            post = _apply(self.es)
        else:
            self.rand.begin_attempt()
            try:
                with self.rand.advance():
                    post = _apply(self.es)
            except rnd_mod.NeedClientCall:
                # the decision needs client randomness that has not
                # rendered yet (combat rolls...): defer, never invent
                self.chat_deferred = (action, via)
                return [('note',
                         'chat action deferred (%s): awaiting client '
                         'randomness' % via)]
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                raise DriverHalt('human %s action failed to apply: %r'
                                 % (via, e))
        self.chat_deferred = None
        pre = self.es
        self._snapshot([], via, post=post)   # chat commits: no client line
        self.es = post
        for hm in self.hand_models.values():
            hm.es = post
        self.hands.es = post
        # K1 flag: ONE writer, at the commit (a deferred retry lands
        # here too)
        _at = getattr(action, 'action_type', '')
        if _at == 'combat_card_hidden_commit':
            self.human_cc_committed = True
        elif _at in ('combat_card_reveal', 'combat_card_cancel'):
            self.human_cc_committed = False
        if self.rand is not None:
            exc = self.rand.unconsumed()
            self.rand.step_commit()
            if any(exc.values()):
                self.findings.append(
                    ('unconsumed_rng', '%r after chat:%s'
                     % ({k: v for k, v in exc.items() if v}, via)))
                self.rand.reset_feed()
        # G2: a round that resolved inside THIS apply announces its
        # hit counts (engine statement - see emission.round_hits_chat)
        out = [('chat', k, t) for k, t in
               em_mod.round_hits_chat(post, self.human_side)]
        # WAVE-I: chat commits have NO gesture rows to subtract — the
        # whole delta is engine-resolved (forced allocations, capture
        # markers) and sweeps here
        out.extend(self.post_commit_sweep(pre, post,
                                          label='chat:%s' % via))
        return out

    def _explain_refusal(self, w):
        """The refusal message, derived from the state the MATCHER
        judged (call before _unreveal).  Returns a chat string or None.

        USER 2026-08-20: "if it says a move isn't legal it should
        specify what rule it thinks was broken."  match_window's 'none'
        branch carries NO note, so _refuse printed the literal word
        'gesture'.  chat.refusal_closest was lifted from the monolith
        but never given a caller; the explainer itself
        (ai_move_server.py:5760-5862) stayed behind.  Refusal-only
        cost: re-deriving the menu and re-applying every candidate is
        far too expensive per gesture and exactly right per REFUSAL."""
        try:
            import ai
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, self.human_side)
            desc, diff = gesture_match.explain_refusal(
                self.es, self.human_side, w, menu, rng=self.rctl)
            if desc and diff:
                return chat_mod.refusal_closest(desc, diff)
            if desc:
                # the closest action produces the SAME board facts, so
                # the disagreement is not a board fact - name the die
                return chat_mod.refusal_closest(desc, ['the action die'])
        except Exception as e:
            self.findings.append(('refusal_explain_failed',
                                  repr(e)[:120]))
        return None

    def _adopt_board_now(self, why):
        """Adopt the client's board into the engine (the resync adopter:
        units / leaders / companions / guide / tracks; census-gated,
        transactional).  Returns note items."""
        items = []
        if not self.view.refresh_dump():
            items.append(('note', '%s: no stable client dump - nothing '
                                  'adopted' % why))
            return items
        try:
            from . import adopt as _adopt
            import ai as _ai9
            _rep, _nb, _na = _adopt.adopt_client_board(
                self.es, self.view.piece_region,
                dynamic_types=self.view.dynamic_types,
                ai_module=_ai9,
                guide_hook=self.reconcile_guide_from_client,
                dump_path=getattr(self.view, 'dump_path', None))
            for _line in _rep:
                items.append(('note', '%s: %s' % (why, _line)))
            self.findings.append(('lenient_adopt',
                                  'divergences %d -> %d' % (_nb, _na)))
        except Exception as _e9:
            items.append(('note', '%s: adopt FAILED (engine left '
                                  'untouched): %r' % (why, _e9)))
            self.alarms.append('LENIENT ADOPT FAILED: %r' % (_e9,))
        return items

    def _lenient_accept(self, w, why):
        """Legality enforcement OFF (USER 2026-08-22): a gesture the
        matcher would refuse or hold is ACCEPTED AS GIVEN - never an
        undo press, never a chat refusal, never a hold.  The staged
        die (if the window had one) is spent on the engine side, the
        human counts as having acted, the client's board is adopted
        into the engine, every expectation of a restore is dropped.
        The panel note says what happened (a finding, not an alarm)."""
        why = str(why or 'gesture')[:120]
        self.findings.append(('lenient_accept', why))
        items = [('note', 'accepted as given (legality enforcement '
                          'OFF): %s' % why)]
        side = self.human_side
        if w is not None and getattr(w, 'die_face', None):
            letter_face = {v: k for k, v in
                           gesture_match.FACE_LETTER.items()}
            want = letter_face.get(str(w.die_face))
            dice = self.es.dice.get(side, [])
            idx = next((i for i, d in enumerate(dice)
                        if not d.used and getattr(d.current, 'name',
                                                  None) == want), None)
            if idx is None:
                idx = next((i for i, d in enumerate(dice)
                            if not d.used), None)
            if idx is not None:
                dice[idx].used = True
                items.append(('note', 'lenient: %s die #%d spent'
                              % (w.die_face, idx)))
            self.last_actor = side
            self.n_commits_total = getattr(self, 'n_commits_total',
                                           0) + 1
        if w is not None:
            try:
                self._unreveal(w)
            except Exception:
                pass
            items.extend(self._lenient_hand_facts(w))
        items.extend(self._adopt_board_now('lenient'))
        from .undo_model import RestoreWatch as _RW
        self.restore_watch = _RW()
        if self.rand is not None:
            try:
                self.rand.reset_feed()
            except Exception:
                pass
        return items

    def _lenient_hand_facts(self, w):
        """A lenient accept takes the window AS GIVEN - including the
        cards the client showed LEAVING the human's hand (a rendered
        title is public information; the count is board truth, and
        _adopt_board_now never adopts hands).  Without this, every
        leniently-accepted play kept its card in the engine's modeled
        hand as a phantom; the next turn's draws then opened a
        hand-limit pending the player could never satisfy (tester
        2026-08-24 'prompted to discard at 6 cards': engine 7, player
        6 - reproduced).  Every removal routes through
        gl.discard_hand_card, THE discard funnel (conservation + the
        pending re-derives).  Only the HUMAN's modeled hand: the AI's
        own plays commit through the engine, never through here."""
        import game_logic as gl
        try:
            f = gesture_match.window_facts(w)
        except Exception as e:
            self.findings.append(('lenient_hand_facts_unread',
                                  repr(e)[:80]))
            return []
        side = self.human_side
        tag = 'FP' if side.name == 'FREE_PEOPLES' else 'SH'
        self.hands.es = self.es
        titles = list(f.cards_played)
        titles.extend(t for (s, t) in f.combat_cards if s == tag)
        items = []
        hand = self.es.hands[side].hand
        for title in titles:
            cid = self.hands.reveal(title)
            if cid is None or cid not in hand:
                # unresolvable title: the COUNT is still board truth -
                # a placeholder of the hand leaves instead, loudly
                cid = next((c for c in hand if isinstance(c, str)
                            and c.startswith('HU_')), None)
                self.findings.append(('lenient_hand_title_unresolved',
                                      str(title)[:60]))
            if cid is None:
                self.findings.append(('lenient_hand_fact_unapplied',
                                      str(title)[:60]))
                continue
            gl.discard_hand_card(self.es, side, cid)
            items.append(('note', 'lenient: %s left the hand'
                          % str(title)[:60]))
        for (sname, deck), n in f.hand_discards.items():
            if sname != side.name:
                continue
            for _ in range(int(n)):
                ph = next((c for c in hand if isinstance(c, str)
                           and c.startswith('HU_')
                           and gl.card_deck_of(c) == deck), None) \
                    or next((c for c in hand
                             if gl.card_deck_of(c) == deck), None)
                if ph is None:
                    self.findings.append(
                        ('lenient_hand_fact_unapplied',
                         'discard %s' % deck))
                    break
                gl.discard_hand_card(self.es, side, ph)
                items.append(('note',
                              'lenient: %s discard applied' % deck))
        return items

    def _refuse(self, w, res, explain=None):
        if not self.enforce_human:
            return self._lenient_accept(
                w, getattr(res, 'note', None) or 'refused gesture')
        from .undo_model import expectations_from_window, \
            presses_to_revert
        from emitter import render
        n = presses_to_revert(self.feed.lines,
                              self._window_span(w), self.nick)
        out = [('chat', 'refusal',
                explain
                or chat_mod.refusal_illegal(res.note or 'gesture'))]
        if n is None:
            self.alarms.append('REFUSAL UNDO-INELIGIBLE - corrective '
                               'restore owed, window held')
            self.holds.append((w, res))
            # WAVE-K (R5 / ledger D-03): the restore watch is ARMED on
            # this path too.  DOCTRINE = HOLD-AND-WITNESS: the client-
            # native undo cannot reach the window (lines were pushed
            # after it - client fact 2), so the board stays unrestored;
            # the watch VERIFIES against the dump and fires the sticky
            # RESTORE WATCH alarm unless someone (human undo / manual
            # edit + resync) restores it.  No reversal is forged.
            expect, notat, _unver = expectations_from_window(
                [r.get('norm', '') for r in w.records],
                pids=self._record_pids(w.records))
            if expect or notat:
                self.restore_watch.arm(expect, notat)
            return out
        for _ in range(n):
            out.append(('line', render('undo', {'nick': self.nick},
                                       local=False)))
        # WAVE-I: ARM the restore watch on the refused window's own
        # rendered lines; tick() verifies against the client dump and
        # the sticky alarm fires if the restore never lands
        expect, notat, _unver = expectations_from_window(
            [r.get('norm', '') for r in w.records],
            pids=self._record_pids(w.records))
        if expect or notat:
            self.restore_watch.arm(expect, notat)
        return out

    @staticmethod
    def _record_pids(records):
        """The sidecar's exact piece ids per window record (None where
        absent or redacted) - the restore watch's first answerer for
        fungible pieces (dice, cards)."""
        from . import sidecar as _sc_mod
        return [_sc_mod.meta_pids(r.get('meta')) for r in records]

    def _check_roll_binding_starvation(self):
        """WAVE-K (R6 / ledger S10): with the sidecar PRESENT the raw
        roll lane is retired, so a roll record that carries no die
        channel (segs[1]) leaves the adoption's index order parked
        forever and the pool unbound - silently.  Alarm ONCE per
        starved adoption after 3 ticks (one answerer: the sidecar; its
        silence is a defect, not a fallback)."""
        sc = getattr(self, 'sidecar', None)
        if sc is None or not getattr(sc, 'present', False):
            return
        for side_name, parked in list(self._adopt_parked_tick.items()):
            if side_name not in self._last_adopt_idxs:
                self._adopt_parked_tick.pop(side_name, None)
                self._adopt_starve_alarmed.discard(side_name)
                continue
            waited = self._tick_n - int(parked)
            if waited >= 3 \
                    and side_name not in self._adopt_starve_alarmed:
                self._adopt_starve_alarmed.add(side_name)
                self.alarms.append(
                    'ROLL PID BINDING STARVED: sidecar present but no '
                    'die channel (segs[1]) for the %s action roll after '
                    '%d ticks - dice pids unbound' % (side_name, waited))

    def _check_restore_watch(self):
        """Verify an armed refusal-restore watch against the client's
        own dump (tick cadence, armed-only); a sticky alarm lands in
        self.alarms ONCE."""
        if not self.restore_watch.armed:
            return
        if not self.view.refresh_dump():
            return
        self.restore_watch.verify(self.view.piece_region)
        if self.restore_watch.alarm:
            msg = 'RESTORE WATCH: %s' % self.restore_watch.alarm
            if msg not in self.alarms:
                self.alarms.append(msg)
            self.restore_watch.clear_alarm()

    def _window_span(self, w):
        """(w0, w1) index span of the window's raw lines within the
        feed's consumed history - the contract presses_to_revert
        actually takes.  (WAVE-H cbt3: _refuse passed the raw-text
        LIST; presses_to_revert's `w0, w1 = window` unpack blew up as
        a server cycle error on the FIRST live refusal.)  (-1, -1)
        when the span cannot be located - presses_to_revert then
        returns None and the refusal takes the corrective-hold path."""
        raws = [r.get('raw', '') for r in w.records if r.get('raw')]
        lines = self.feed.lines
        if not raws or not lines:
            return (-1, -1)
        w1 = None
        for j in range(len(lines) - 1, -1, -1):
            if lines[j] == raws[-1]:
                w1 = j + 1
                break
        if w1 is None:
            return (-1, -1)
        lo = max(-1, w1 - 1 - 400)
        for j in range(w1 - 1, lo, -1):
            if lines[j] == raws[0]:
                return (j, w1)
        return (-1, -1)

    # ---- undo events ----------------------------------------------------
    def _on_undo(self, ev, rec):
        """One client undo press = ONE line restored (fact 1: LIFO,
        whole-board, per line).  The inner text names the undone line;
        consume it tail-first from the newest commit's line ledger and
        pop the engine to the pre-state only when that commit is fully
        unwound.  Undo of a never-committed line (a refused/held
        window our presses are unwinding) is pure bookkeeping.

        A DEFERRED window can be unwound before its randomness ever
        arrives (ported PHASE3-D; real specimen: turn-3 fs move
        staged, closed on die-to-hunt, fully aborted by three undos):
        tail-pop its records; a fully-unwound deferral dissolves —
        the human took the action back."""
        if ev == 'redo':
            # unmodelled: a sticky alarm (once), never a halt that
            # would drop the rest of the poll batch
            msg = 'REDO seen - unmodelled, holding; RESYNC if the ' \
                  'board looks wrong'
            if msg not in self.alarms:
                self.alarms.append(msg)
            return [('note', 'client REDO is not modelled - RESYNC')]
        inner = str(rec.get('inner', '')).strip()
        kind, norm = live_parser.normalize(inner)
        if self.deferred is not None:
            w, _side = self.deferred
            if w.records and str(w.records[-1].get('norm', '')) \
                    == norm:
                w.records.pop()
                if not w.records:
                    self.deferred = None
                    if self.rand is not None:
                        self.rand.reset_feed()
                return []
        urec = live_parser.parse(inner)
        if urec.get('op') == 'move' and gesture_match.zone_key(
                urec.get('from'))[0] == 'hand_chosen':
            items = self._on_reveal_undone(urec)
            if items is not None:
                return items
        if self.uncommitted and norm in self.uncommitted:
            self.uncommitted.remove(norm)
            return []
        if not self.commits:
            self.alarms.append('undo with empty commit ledger '
                               '(pre-game gesture) - ignored')
            return []
        # the client's undo is LIFO over ITS lines; the AI's replies
        # are LINELESS ledger entries (booked the moment they applied,
        # often in the same poll cycle as the human's window), so the
        # human's newest lines may sit BENEATH one or more 'ai:' entries
        k = len(self.commits) - 1
        above = []
        while k >= 0 and not self.commits[k][1] \
                and str(self.commits[k][2]).startswith('ai:'):
            above.append(k)
            k -= 1
        pre, lines, label, turn_state = self.commits[k][:4] \
            if k >= 0 else self.commits[-1][:4]
        if not lines:
            # a chat-committed action has no client lines to unwind:
            # the client's undo cannot reach it (no snapshot of ours)
            self.alarms.append('UNDO reached a lineless commit (%s) '
                               '- holding' % label)
            return [('note', 'undo reached a chat-decided action (%s) '
                             '- RESYNC' % label)]
        if lines[-1] == norm:
            lines.pop()
            if not lines:
                replies = [self.commits[j] for j in reversed(above)]
                del self.commits[k:]
                self.es = pre
                for hm in self.hand_models.values():
                    hm.es = pre
                self.hands.es = pre
                self._restore_turn_state(turn_state)
                self.undo_epoch += 1
                if replies:
                    return self._rebase_ai_replies(replies)
            return []
        self.alarms.append(
            'UNDO ORDER MISMATCH: inner %r vs expected tail %r (%s) '
            '- holding the pop' % (norm, lines[-1], label))
        return [('note', 'undo out of order - RESYNC if the board looks '
                         'wrong')]

    def _rebase_ai_replies(self, replies):
        """THE ENGINE FOLLOWS THE CLIENT: the human retracted a window
        the AI had already answered (its reply lines are on the client
        and cannot be taken back), so each AI commit above the popped
        window is re-applied, in order, onto the restored state; the
        AI's last plain action keeps the alternation (the human is to
        act).  A reply that cannot re-apply - illegal on the restored
        state, or needing client randomness already consumed - stops
        the re-base LOUDLY (alarm naming it; RESYNC owed): the boards
        have parted and no guess narrows it."""
        import ai
        from . import randomness as rnd_mod
        done = 0
        for ent in replies:
            label = ent[2]
            act = ent[4] if len(ent) > 4 else None
            if act is None:
                self.alarms.append('UNDO RE-BASE FAILED: the AI commit '
                                   '%s carries no action to re-apply - '
                                   'RESYNC owed' % label)
                break
            pre = self.es
            try:
                if self.rand is None:
                    post = ai.apply_action(pre, act)
                else:
                    self.rand.begin_attempt()
                    with self.rand.advance():
                        post = ai.apply_action(pre, act)
            except rnd_mod.NeedClientCall:
                self.alarms.append('UNDO RE-BASE FAILED: the AI\'s %s '
                                   '(%s) needs client randomness that '
                                   'was already spent - RESYNC owed'
                                   % (label, getattr(act, 'action_type',
                                                     act)))
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                self.alarms.append('UNDO RE-BASE FAILED: the AI\'s %s '
                                   '(%s) does not apply on the restored '
                                   'state: %r - RESYNC owed'
                                   % (label, getattr(act, 'action_type',
                                                     act), e))
                break
            self._snapshot([], label, post=post, action=act)
            self.es = post
            for hm in self.hand_models.values():
                hm.es = post
            self.hands.es = post
            if label == 'ai:action':
                self.last_actor = self.ai_side
            done += 1
        self.findings.append(('undo_rebased',
                              '%d/%d AI commit(s) re-applied after the '
                              'retraction' % (done, len(replies))))
        if done == len(replies):
            # the registered builder + a vocabulary kind: a raw 'note'
            # chat kind is NOT in the sided vocabulary and raised
            # ChatPolicyError as a server cycle error on the real
            # client (enforced twin leg e1, 2026-08-22)
            return [('chat', 'prompt', chat_mod.undo_rebased())]
        return [('note', 'undo re-base incomplete - RESYNC')]

    # ---- AI turn --------------------------------------------------------
    def _ai_turn_start_draws(self):
        self.phase = 'declare'
        import game_state as gs
        # K2: the Fellowship phase opens for WHICHEVER side is FP
        self._ensure_declare_armed()
        if self.ai_side is gs.Side.FREE_PEOPLES:
            # WAVE-K R2: the AI holds the Fellowship phase - its own
            # declaration decision (learned) follows its draws, then
            # the Mordor-entry check, then ITS 'fp ok' (live_server
            # queues the second step behind the first)
            return [('ai_draws', None), ('ai_declare', None)]
        return [('ai_draws', None)]     # live_server runs the AI draws

    def _ai_turn(self):
        """When the engine says the pending decision (or the active
        turn) is the AI's, decide and emit.  Emission module wires in
        live_server; here we only surface the fact."""
        dp = self.ai_action_dp()
        if dp is not None:
            if dp['kind'] == 'declare':
                # the AI-FP's declaration is the 'ai_declare' step's
                # (queued by _ai_turn_start_draws), never a generic
                # decide item
                return []
            return [('ai_decide', dp)]
        return []

    def ai_action_dp(self):
        """The AI's due decision, or None: an open engine pending owned
        by the AI, OR — PHASE3-B gap closed 2026-08-20 — the plain
        action-turn ALTERNATION after a committed action.  Whose-turn
        comes from the ONE engine answerer ai._next_acting_side
        (pendings, multi-step continuation, the battle-is-one-
        attacker-action rule, dice exhaustion -> None); the driver
        contributes only WHO JUST ACTED (its own commit bookkeeping),
        never a turn rule."""
        import ai
        import game_state as gs
        if self.ai_side is gs.Side.FREE_PEOPLES:
            # K2: the AI-FP's declare menu stays armed through its own
            # turn-start draws/discard until its declare step closes
            self._ensure_declare_armed()
        dp = ai.describe_pending(self.es, self.ai_side)
        if dp is not None:
            return dp if dp['side'] is self.ai_side else None
        if self.phase != 'actions' or self.last_actor is None:
            return None
        nxt = ai._next_acting_side(self.es, self.last_actor)
        if nxt is not self.ai_side:
            return None
        if self.rand is not None:
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, self.ai_side)
        else:
            menu = ai.get_legal_actions(self.es, self.ai_side)
        if not menu:
            return None
        return {'kind': 'action', 'side': self.ai_side,
                'actions': menu}

    # ---- helpers --------------------------------------------------------
    def check_victory(self, end_of_turn=False, state=None):
        """WAVE-J (A-09): the ENGINE's own victory answerers, at the
        commit chokepoint.  RULING 1.212: ring victories (destroyed /
        corruption 12) end the game IMMEDIATELY mid-action; MILITARY
        victories are only checked at TURN END (should_game_end owns
        the timing rule - one answerer)."""
        if self.game_over is not None:
            return self.game_over
        if getattr(self, 'spectator', False):
            return None     # the corpus soak is byte-pinned; the
            #                 spectator's own census owns its endings
        import game_logic as gl
        try:
            vc = gl.check_all_victory_conditions(
                state if state is not None else self.es)
            if gl.should_game_end(vc, end_of_turn):
                self.game_over = vc
                self.findings.append(
                    ('game_over', '%s (%s) fp=%s sh=%s'
                     % (vc.get('victory'), vc.get('victory_type'),
                        vc.get('free_peoples_vp'),
                        vc.get('shadow_vp'))))
        except Exception as e:
            self.alarms.append('victory check FAILED: %r' % e)
        return self.game_over

    # the driver turn state a ledger entry carries with the engine
    # pre-state (K5): a full unwind restores BOTH, or the turn's draw
    # counters / phase / alternation would outlive the undone commit
    _TURN_STATE_FIELDS = ('turn_draws', 'phase', 'last_actor',
                          'owed_die_kills', 'owed_card_kills',
                          'human_cc_committed')

    def _turn_state(self):
        out = {}
        for f in self._TURN_STATE_FIELDS:
            v = getattr(self, f, None)
            if isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, list):
                v = list(v)
            out[f] = v
        return out

    def _restore_turn_state(self, ts):
        for f, v in (ts or {}).items():
            if isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, list):
                v = list(v)
            setattr(self, f, v)

    def _snapshot(self, line_norms, label, post=None, action=None):
        """Ledger entry: (engine PRE-state, client line norms, label,
        driver turn state, the committed engine action or None) -
        callers snapshot BEFORE mutating either the engine or the turn
        state.  `action` lets a lineless AI commit be RE-BASED when the
        human retracts the window beneath it (_on_undo)."""
        import ai
        self.commits.append((ai.clone_state(self.es),
                             [n for n in line_norms if n], label,
                             self._turn_state(), action))
        self.n_commits_total += 1       # monotonic commit epoch
        if len(self.commits) > 100:
            self.commits.pop(0)
        # WAVE-J (A-09): every commit checks for a MID-ACTION ring
        # victory on the adopted post state (military waits for the
        # turn-end check in _on_recover - RULING 1.212).  WAVE-K (R1 /
        # PARITY-AUDIT-2): evaluated on the POST state the caller passes
        # - every commit site snapshots BEFORE `es = post`, so self.es
        # here is still the PRE state and a mid-action ring victory was
        # detected one commit LATE (the AI's next action could emit
        # after the game was won).  post=None = the callers whose es
        # already IS the current state (human draw, rollover).
        self.check_victory(end_of_turn=False, state=post)

    def _bind_roll_pids(self, pids, values):
        """The raw roll line's pid list binds physical dice to engine
        indices IN ORDER of the same adoption (sanctioned raw #2).

        WAVE-H cbt3 LIVE FINDING (turn-2 wedge): the pre-seeded
        index->pid map (fact 11) is only true while the client selects
        the pool in ascending pid order.  After the turn-1 recover the
        client's turn-2 FP roll listed pids REVERSED
        ('Controls.345>213 212 211 210' faces 'C H P C') - the status
        table then published the WRONG pid per engine die, the harness
        staged a die whose client face disagreed with the engine's,
        and the matcher rightly refused an engine-legal gesture.
        Rendered letters carry no pids, and the letters ORDER equals
        the raw line's pid order (measured on all 4 cbt3 roll pairs) -
        so the raw pid list is STORED per side and consumed by
        _on_action_roll's adoption, binding pid[i] -> engine idx[i] of
        the SAME assignment.  This also closes the bonus-die binding
        gap (358/359/360+ bind like every other pid on their line)."""
        clean = [str(p) for p in pids]
        if not clean:
            return
        side_name = None
        for p in clean:
            b = self.view.die_id_map.get(p)
            if b is not None:
                side_name = getattr(b[0], 'name', str(b[0]))
                break
        if side_name is None:
            # a line of only never-bound pids: family heuristic
            side_name = ('SHADOW' if any(p in self._SH_DIE_FAMILY
                                         for p in clean)
                         else 'FREE_PEOPLES')
        self._pending_roll_pids[side_name] = clean
        self._try_bind_roll(side_name)   # adoption may have run first

    _SH_DIE_FAMILY = frozenset('%03d' % (99 + i) for i in range(7))

    def guide_from_client(self):
        """The companion the CLIENT shows in its Guide box, or None.

        Returns the engine-side Companion name so the caller can hand
        it to game_logic.change_guide.  None when the box is empty,
        ambiguous, or holds a piece we cannot name - the caller then
        leaves the engine alone rather than guessing.
        """
        import log_replay as lr
        occ = []
        for pid, z in (self.view.piece_region or {}).items():
            if _zone_key(z) not in _GUIDE_ZONES:
                continue
            # lr.COMPANIONS is keyed by the pid STRING ('241'), not
            # an int - coercing to int here silently matched nothing
            key = str(pid)
            if key in lr.COMPANIONS:
                occ.append(key)
        if len(occ) != 1:
            return None                      # empty or ambiguous
        names = set(lr.names_for_pid(occ[0]) or ())
        for c in (self.es.fellowship.companions or []):
            if str(c.name) in names:
                return str(c.name)
        return None

    def reconcile_guide_from_client(self):
        """Ingest the human's GUIDE CHOICE from the client (I1-safe).

        USER 2026-08-21, reproduced before patching: with Strider as
        guide the stack refused a legal hide as an illegal move, and
        reset+resync could not repair it.

        Root cause: the guide is chosen by the FP player, but setup
        breaks the Gandalf-the-Grey / Strider level-3 tie SILENTLY -
        game_logic.change_guide takes best[0] and never opens
        pending_guide_change - so a human who guides with Strider is
        desynced from turn 1.  Every Strider-ability hide ("hide the
        Fellowship using ANY Action die", ai.py:7334) is then refused,
        because that branch tests `'Strider' in guide.name`.

        Why the button could not fix it: resync_recover adopts the
        client board into the VIEW and leaves ENGINE state untouched
        by design (I1).  The guide lives in engine state, so no amount
        of resyncing reached it.

        Why this does not break I1: a guide choice is a DECISION, not
        client state being adopted - the Guide box is merely where the
        human states it.  It is applied through change_guide, which
        validates against the rules (must be in the Fellowship, must
        be tied for top level) and RAISES on an illegal pick, which we
        report rather than force.  The engine stays the authority.

        Returns a note string when something changed, else None.
        """
        import game_logic as gl
        want = self.guide_from_client()
        if want is None:
            return None
        cur = getattr(getattr(self.es.fellowship, 'guide', None),
                      'name', None)
        if cur is not None and str(cur) == want:
            return None                      # already agreed
        try:
            gl.change_guide(self.es, chosen=want)
        except Exception as e:
            # an illegal pick is the human's to fix, not ours to force
            self.findings.append(('guide_reconcile_refused',
                                  '%s (%s)' % (want, repr(e)[:60])))
            return ('guide: client shows %s but the rules refuse that '
                    'choice (%s) - engine keeps %s'
                    % (want, repr(e)[:60], cur))
        self.findings.append(('guide_reconciled',
                              '%s -> %s' % (cur, want)))
        return ('guide: engine had %s, client shows %s - adopted the '
                "human's choice" % (cur, want))

    def resync_recover(self, drain_feed=False):
        """WAVE-J (ledger C-12, USER redefinition 2026-08-16): RESYNC
        = RECOVERY, the adopt-and-keep-playing lever.  Adopts the
        client's PHYSICAL board into the view model (hidden info stays
        engine-side - and the notes say so), clears stuck expectation
        machinery LOUDLY (census rows, never silence), reopens a
        rules-owed hand-limit pending, and leaves the engine state
        untouched (I1: the engine is never adopted FROM the client).
        drain_feed: skip the un-ingested backlog (manual-edit release:
        the edited sequence must not be matched as gestures)."""
        import game_logic as gl
        items = [('note', 'resync-recovery: adopting the client '
                          'board into the view; hidden info stays '
                          'engine-side; engine state untouched')]
        if drain_feed:
            try:
                skipped = len(self.feed.poll() or ())
                if skipped:
                    self.findings.append(('resync_feed_drained',
                                          '%d line(s)' % skipped))
                    items.append(('note', 'resync-recovery: %d feed '
                                          'line(s) drained unmatched'
                                  % skipped))
            except Exception:
                pass
        if self.view.refresh_dump():
            self.findings.append(('resync_adopt',
                                  '%d piece zones adopted'
                                  % len(self.view.piece_region)))
        else:
            items.append(('note', 'resync-recovery: no stable dump - '
                                  'nothing adopted'))
        # USER RULING 2026-08-21: "the resync is supposed to take the
        # CLIENT state and give it to the engine, not the other way
        # around".  Adopting into the VIEW alone left every divergence
        # that lives in ENGINE state (guide, armies, companions)
        # unreachable, which is why the button never unstuck anything.
        # Transactional + census-gated: adopt.adopt_client_board writes
        # only when the divergence count improves, so a partial adopter
        # can decline but never corrupt.
        try:
            from . import adopt as _adopt
            import ai as _ai9
            _rep, _nb, _na = _adopt.adopt_client_board(
                self.es, self.view.piece_region,
                dynamic_types=self.view.dynamic_types,
                ai_module=_ai9,
                guide_hook=self.reconcile_guide_from_client,
                dump_path=getattr(self.view, 'dump_path', None))
            for _line in _rep:
                items.append(('note', 'resync-adopt: %s' % _line))
            self.findings.append(('resync_engine_adopted',
                                  'divergences %d -> %d' % (_nb, _na)))
        except Exception as _e9:
            items.append(('note', 'resync-adopt FAILED (engine left '
                                  'untouched): %r' % (_e9,)))
        # stuck expectations expire LOUDLY (the census rows name them)
        if self.chat_deferred is not None:
            self._expire_chat_deferred('resync-recovery')
        if self.deferred is not None:
            self._expire_deferred('resync-recovery')
        self._flush_deferred_tail()
        if self.pending_buf is not None:
            self._expire_pending('resync-recovery')
        n_unc = len(self.uncommitted)
        if n_unc:
            self.findings.append(('resync_uncommitted_cleared',
                                  '%d line(s)' % n_unc))
            self.uncommitted = []
        for e in self.echo_ledger:
            self.findings.append(('mirror_echo_unclaimed',
                                  'resync-recovery: %r' % (e,)))
        self.echo_ledger = []
        # a rules-owed hand-limit pending reopens (the client kept
        # playing; the engine still owns the rule)
        try:
            gl.open_hand_limit_discard(self.es)
        except Exception:
            pass
        self.findings.append(('resync_recovery', 'completed'))
        return items

    def _witness_report(self):
        if not self.view.refresh_dump():
            return [('note', 'no stable board dump')]
        rows = client_board.diff_engine_vs_dump(
            self.es, self.view.piece_region,
            dynamic_types=self.view.dynamic_types)
        if rows:
            self.alarms.append('WITNESS: %d discrepancies' % len(rows))
            return [('note',
                     'desync: %d rows (first: %s)'
                     % (len(rows), rows[0]))]
        return [('note', 'engine and client agree')]
