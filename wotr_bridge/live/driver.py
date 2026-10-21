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
  1. client draws 2 for the human (one per deck), automatically
  2. AI draws 2 through the existing client randomness funnel
  3. hand-limit discard resolves NOW      [D4] (engine pending gates)
  4. FP declaration window ends on the declaration gesture or 'fp ok'
                                          [D2: the Fellowship phase]
  5. eye allocation BEFORE the roll [D3]; AI allocating 0 says '0e'
  6. dice roll: each side rolls its OWN pool [G1]; faces adopt via
     game_logic.adopt_rolled_faces (rolled Eyes -> Hunt Box)
  7. action alternation: engine-owned (_next_acting_side /
     _post_action_active); pass legality is the engine's menu
  8. after the last action and its physical consequences, recover both
     pools automatically; native recovery confirms begin_turn_reset
     (A12) + bonus dice [J1]. Manual recovery is also acknowledged.

THE HAND MODEL (the E1/E2/E3/E4 class killer):
  The engine is authoritative for COUNTS everywhere; identities for
  BOTH hands ride the commitment decode (USER 2026-08-25: "the
  engine should see the human hand" - human draws decode to real
  cids and the published menu carries the human's card plays).  A
  human draw still mints an opaque placeholder
  ('HU_<side><deck>_<n>') at the rendered line; the decode swaps it
  for the real cid within the same server cycle (or a play's title
  does, for any card the decode missed - the placeholder path stays
  as the loud fallback).  The AI's CHAT never names the human's
  cards; counts can never drift because every hand-zone grammar
  routes through one pair of enter/leave functions.

RAW-CHANNEL IMPORTS (rendered-baseline directive): exactly TWO client
statements live only on the raw channel and are sanctioned — the
CardDecoder commitment decode (identity of drawn cards) and the
action-roll pid binding (<pids> <values>: which physical die carries
which rolled face; the rendered letters line has no pids, and a wrong
binding would stage a die whose client face differs from the engine's).
Both are client statements, not inference; everything else raw is dead
to this driver.

PORTED FROM THE SPECTATOR (PHASE3-D mechanisms, hoisted into this base
2026-08-20 so ONE implementation serves both drivers — defect-9 rule;
the spectator's shadow copies were DELETED in WAVE-G, it keeps
re-export aliases only):
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

# sanctioned raw import #2: the client's action-roll line pairs
# Controls.345 (pids) with Controls.346 (values) (+ trailing
# checksum); the rendered letters line carries no pids.
RAW_ACTION_ROLL_RE = re.compile(
    r'<~Controls\.345~>([\d ]+?) ?<~Controls\.346~>([\d ]+)')


class DriverHalt(RuntimeError):
    """A loud, sticky, NAMED stop with a recovery path (instrument
    rule: fail loudly, never wait silently)."""


class NativeDieFaceMismatch(DriverHalt):
    """Public die identity disagrees; a paired checkpoint can restore it."""


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

    def outcome_key(self):
        """External obligations after a completed human gesture.

        Probe counters and the choice journal reset at step_commit; they are
        execution details. Which client outcomes were consumed does matter.
        """
        r = self.r
        pending = r._pending
        return gesture_match._typed_value_key({
            'consumed': r._feed_pos,
            'pending': None if pending is None else (pending.kind, pending.kw),
            'pending_got': r._pending_got,
            'shim_uses': r.shim_uses,
        })

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
    (Hoisted from spectator.py PHASE3-D; the spectator's twin was
    deleted in WAVE-G, an alias re-export remains.)"""
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
        occupied = {c for hand in self.es.hands.values() for c in hand.hand}
        for attr in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
            occupied.update(getattr(self.es, attr, ()) or ())
        while True:
            self._n += 1
            ph = 'HU_%s_%s_%d' % (self.side.name[:2],
                                    'CHAR' if deck == 'character'
                                    else 'STRAT', self._n)
            if ph not in occupied:
                return ph

    def human_draw(self, deck, record=None):
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
        from .human_draw_identity import slot
        slot(self, deck, ph, record)
        self.es.hands[self.side].needs_discard = len(hand) > 6
        if len(hand) > 6:
            # BL-47 / BL-74: the hand-limit discard PENDING is opened
            # here by the one implementation
            # (game_logic.open_hand_limit_discard); the [BL-74] print
            # is the ingest instrument (a pinned log line).
            import game_logic as _gl47
            try:
                _opened74 = _gl47.open_hand_limit_discard(self.es)
                # BL-74 instrument: HandModel has no findings list -
                # the print lands in live_server.out
                print('  [BL-74] hand-limit ingest: %s hand=%d '
                      'opened=%s pending=%r'
                      % (self.side.name, len(hand), _opened74,
                         getattr(self.es, 'pending_discard', None)))
            except Exception as _e74:
                print('  [BL-74] hand-limit ingest FAILED: %r'
                      % (_e74,))
        return ph

    def reveal(self, title_or_cid, deck=None, placeholder=None):
        """A human card became PUBLIC (played / returned face-up):
        swap one placeholder of that deck for the real cid.  Returns
        the cid, or None if resolution failed (caller holds)."""
        cid = self._resolve_cid(title_or_cid)
        if cid is None:
            return None
        hand = self.es.hands[self.side].hand
        if placeholder is not None and placeholder not in hand:
            return None                     # this draw has not committed
        if cid in hand:
            return cid                      # already public
        import game_logic as gl
        dk = deck or gl.card_deck_of(cid) or 'strategy'
        ph = placeholder if placeholder is not None else next((c for c in hand if isinstance(c, str)
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

# ---------------------------------------------------------------------
# SELF-REPAIR (USER 2026-08-28): "we should never have generic 'open
# pending'; it needs to be explicit about what it is waiting on.
# furthermore, it should be self repairing."  The stall detectors call
# _propose_repair; the human answers yes/no in chat; the repair runs
# as a small state machine in _repair_tick (kill the staged pieces ->
# wait for the client to show it -> reset + resync client-as-is with
# the political track from the engine -> turn from the last die
# killed -> announce).
# ---------------------------------------------------------------------
_DIE_KILL_SIDE_RE = re.compile(
    r'\bkills\b.*?\b(Shadow|Free) Action Die \(', re.I)
_UNDO_LINE_RE = re.compile(r'\bundoes last action\b', re.I)
# the client zones the repair empties: the staged die and the current
# card slots (the dump names the card slots 'FP/SA Card Play'; the feed
# names them 'Current FP/SA Card' - both spellings are the same zone)
_REPAIR_SLOTS = frozenset(('current action', 'fp card play',
                           'sa card play', 'current fp card',
                           'current sa card'))
# pending kind (substring) -> what physical gesture / chat satisfies it
_PENDING_HINTS = (
    ('hunt', 'move the guide to casualties (it takes the damage), or '
             'say random or corruption'),
    ('reveal', 'drag the Fellowship figure to the region it is revealed '
               'in'),
    ('declare', 'say fp ok, or move the Fellowship marker to declare'),
    ('discard', 'kill the card from your hand'),
    ('casualt', 'remove the units the hits require'),
    ('retreat', 'move the retreating army, or say field to keep fighting'),
    ('combat', 'play or decline a combat card, retreat, or fight - on '
               'the board or in chat'),
    ('guide', 'move the new guide into the guide box'),
    ('card', 'perform the card effect on the board, or decline it'),
    ('muster', 'place the mustered units or leaders on the board'),
)


def side_of_last_die_kill(lines):
    """USER turn rule (2026-08-25): "the last die killed = previous
    player's turn".  Scans raw feed lines oldest->newest for action-die
    kills (either nick; the SIDE comes from the die's own name, 'Shadow
    Action Die' / 'Free Action Die'), honouring an undo of a kill.
    Returns the Side whose die was killed last, or None when no die has
    been killed."""
    import game_state as gs
    kills = []
    for raw in list(lines or ()):
        s = str(raw)
        m = _DIE_KILL_SIDE_RE.search(s)
        if m is None:
            continue
        side = (gs.Side.SHADOW if m.group(1).lower() == 'shadow'
                else gs.Side.FREE_PEOPLES)
        if _UNDO_LINE_RE.search(s):
            if kills and kills[-1] is side:
                kills.pop()              # the undone kill
            continue
        kills.append(side)
    return kills[-1] if kills else None


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
        # the segmenter recognizes the server's OWN emission echoes
        # (self-echo detection only; attribution stays zone-derived)
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
        self._turn_start_draws_open = False
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
        # Either half may arrive first: adoption PARKS its index order
        # and the bind fires from whichever completes the pair.
        self._pending_roll_pids = {}
        self._last_adopt_idxs = {}      # side name -> (idxs, n_faces)
        # side -> tick the adoption parked at (R6 starvation alarm)
        self._adopt_parked_tick = {}
        self._adopt_starve_alarmed = set()
        self._tick_n = 0
        self._seen_roll_lines = set()
        # ---- state shared with the spectator (same names, same
        # initial values) ----
        self.findings = []          # census rows (kind, detail)
        # the game-end verdict (ring: mid-action; military: TURN END
        # per RULING 1.212; or a concession) - once set the server
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
        self._repair = None         # self-repair offer/run (see _repair_tick)
        self._repair_kill_echo = None   # our own kill line owed back
        # board phase records arriving WHILE a window is deferred:
        # they may express the pendings its commit will open — queued,
        # replayed after the deferral settles
        self.deferred_tail = []
        # roller-nick -> side name | None(=poisoned), bound from the
        # ZONE-attributed draw lines (fact 3 compliant); feeds the d6
        # side lanes so interleaved combat segments stay straight
        self.nick_side = {}
        # a chat-committed human decision whose apply STARVED on client
        # randomness (the defender's 'nc' fires the round's rolls):
        # retried on each rng-bearing record, its ask surfaced for
        # live_server to ORDER, expired LOUDLY when the client moves on
        # (never applied outside advance(): the funnels are inert there)
        self.chat_deferred = None       # (action, via) | None
        # die kills owed AFTER a chain closes (USER invariant 'die ->
        # action -> resolve everything -> kill die'; play_event /
        # army_attack / draw_card): live_server appends at emission
        # time and flushes render_die_kill once the chain is closed
        self.owed_die_kills = []        # [(side_name, die_idx)]
        # R4: the AI's COMMITTED COMBAT CARD dies in the Current-Card
        # slot when the combat closes (corpus 'kills <title> in Current
        # Action.'); [(side_name, pid, cid)]
        self.owed_card_kills = []
        # K1: the human committed an UNREVEALED combat card this round
        # (ai.commit_hidden_combat_card); cleared by the reveal line
        self.human_cc_committed = False
        self._cc_battle_serial = 0
        self._cc_reveal_origins = {}
        self._cc_choose_preamble = None
        self._uncommitted_card_selection = None
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
        # second-consecutive-empty-expiry tracker for the pending force
        # (_expire_pending, spectator lane)
        self._expire_streak = None
        # MIRROR-ECHO LEDGER (the cbt10 race): unit-removal rows a
        # committed apply already resolved engine-side.  AI-side rows
        # are EMITTED by the sweep and banked here; HUMAN-side rows the
        # gesture did not express are banked un-emitted (the human
        # renders their own casualties).  An incoming non-self
        # kill/removal record fully covered by open entries is an ECHO
        # of the committed effect - consumed, never a new gesture.
        # Entries: dict(kind='unit'|'naz'|'named', side='fp'|'shadow',
        #               region=<region_key>, tkey/name, n)
        self.echo_ledger = []
        self._post_commit_tail = ()
        self._forced_ai_rows_holds = ()
        self._human_casualty_prefix = None
        self._ai_casualty_prefix = None
        # SETTLEMENT-CAPTURE ledgers: control markers emitted per
        # region (region_key -> pid) and the VP value each side's
        # marker was last dragged to (engine truth: enemy_controlled +
        # calculate_victory_points)
        self.ctrl_markers = {'fp': {}, 'sh': {}}
        self.vp_marks = {'fp': 0, 'sh': 0}
        # the refusal-restore watch (undo_model.RestoreWatch), ARMED at
        # every refusal and verified on the tick against the client dump
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
        # the sidecar: live_server attaches a SidecarReader; None =
        # channel absent (corpus soak, old jars) and the heuristic
        # lanes stay - one answerer PER CHANNEL AVAILABILITY
        self.sidecar = None
        self._feed_seq = 0          # 1-based feed line number of the
        #                             line being ingested (= its index
        #                             in feed.lines + 1)
        self._route_errors_seen = set()   # distinct error texts alarmed
        # ---- BRLANE fix wave (2026-08-29) -------------------------------
        # BRLANE-13: the AI's emitted op lines awaiting the client's echo
        # (pid -> {zone, deadline tick, raw line, expired}); the ack file
        # of the candidate jar (client_protocol) releases them first,
        # the feed echo under the AI nick second, the deadline is LOUD
        self.echo_expect = {}
        self.ack_path = None            # live_server sets ai_inject_ack.jsonl
        self._acks_seen = 0
        # BRLANE-7: the raw op lines the server emitted (BL-26 re-assert)
        self._own_emitted = []
        # BRLANE-12: a region chatted over an open card-play window = the
        # card's target, answered when the play opens its decision
        self._card_target_hint = None
        self._human_draw_asked = None   # BRLANE-14 one ask per demand
        self._retrying_holds = False    # BRLANE-2 re-entrancy guard

    # ------------------------------------------------------------------
    # BRLANE fix wave (2026-08-29): the state the gesture / window /
    # chat / undo lanes PUBLISH, and the lanes that used to be silent.
    # ------------------------------------------------------------------
    _ECHO_DEADLINE_TICKS = 6     # ~2 client cycles at pace 1.6 s + slack
    _HOLD_RETRY_MAX = 3
    _HOLD_EXPIRE_TICKS = 150     # ~4 min of an unclearing condition

    def _human_window_open(self):
        from .staged_dice import pending as selection_pending
        w = getattr(self.seg, 'window', None)
        return (not getattr(self, 'spectator', False)
                and ((w is not None and getattr(w, 'die_side', None) == self.human_side.name)
                     or selection_pending(self)))

    def human_action_completion_pending(self):
        """The next action waits for the human's physical die-window close.

        A chat-declared battle can finish in the engine before its spent die
        leaves Current Action. This transport boundary does not block an AI
        choice *inside* that battle or another ongoing human card effect.
        """
        return (not getattr(self, 'spectator', False)
                and (self._human_window_open()
                     or getattr(self, '_chat_attack_window', None) is not None))

    def action_completion_pending(self, resuming_owed=()):
        """A new action waits for cleanup; a retry owns its exact staged die.

        Only the server's current journaled step supplies resuming_owed.
        Human windows and every other physical die obligation still hold.
        """
        return (self.human_action_completion_pending()
                or any(entry not in resuming_owed
                       for entry in (getattr(self, 'owed_die_kills', None) or ())))

    @property
    def _reconcile_due(self):
        """BRLANE-1 / M1 / M4: the auto-reconcile flag is a PROPERTY - a
        request to arm it while a HUMAN gesture window is open (die
        staged, pieces dragged, die not yet killed) is PARKED in
        `_reconcile_owed` and promoted when the window closes, so the
        AI's own commit (live_server) can never make the comparator
        judge a half-made gesture.  Readers see a plain bool."""
        return bool(self.__dict__.get('_reconcile_due_v', False))

    @_reconcile_due.setter
    def _reconcile_due(self, value):
        if value and self._human_window_open():
            if not self.__dict__.get('_reconcile_owed'):
                self.findings.append(('reconcile_parked_window',
                                      'arm parked: human window open'))
            self.__dict__['_reconcile_owed'] = True
            return
        self.__dict__['_reconcile_due_v'] = bool(value)

    def _promote_reconcile(self):
        """The human window closed: a parked arm becomes due."""
        if self.__dict__.get('_reconcile_owed'):
            self.__dict__['_reconcile_owed'] = False
            self.__dict__['_reconcile_due_v'] = True

    def gesture_open(self):
        """BRLANE-1: True while a human gesture is half-made (its die
        staged and not killed), a refusal restore is in flight, or an
        AI emission echo is outstanding - the comparator's
        quiescence_holds() reads THIS (never re-derives it)."""
        if getattr(self, 'spectator', False):
            return False
        if self._human_window_open():
            return True
        if getattr(getattr(self, 'restore_watch', None), 'armed', False):
            return True
        return self.echo_pending()

    # ---- BRLANE-13: the expected-echo ledger ---------------------------
    def _echo_channel_present(self):
        sc = getattr(self, 'sidecar', None)
        if sc is not None and getattr(sc, 'present', False):
            return True
        ap = getattr(self, 'ack_path', None)
        return bool(ap and os.path.exists(ap))

    @staticmethod
    def _op_line_facts(ln):
        """(op, pids, decoded destination zone) of one raw op line the
        AI emitted, or None for a non-op line (chat, !CALL, undo)."""
        import emitter as _em
        import log_replay as _lr
        if not isinstance(ln, str) or not ln.startswith('$'):
            return None
        op, f = _em.classify_line(_em.strip_tail(ln))
        if op is None or op in ('undo', 'chat', 'directive'):
            return None
        pids = []
        if isinstance(f.get('pieces'), list):
            pids = [str(p) for p in f['pieces']]
        elif isinstance(f.get('pieces'), str):
            # The native single/group send-to-box grammar returns a space-
            # separated PID string (Controls.281), unlike list-valued kills.
            # Track exact numeric identities only, never titles or hidden IDs.
            tokens = f['pieces'].split()
            if not tokens or any(not token.isascii() or not token.isdigit()
                                 for token in tokens):
                return None
            pids = tokens
        elif f.get('piece'):
            pids = [str(f['piece'])]
        elif f.get('die'):
            pids = [str(f['die'])]
        if not pids:
            return None
        zone = None
        to = f.get('to')
        if to:
            try:
                zone = str(_lr.decode_location(str(to)))
            except Exception:
                zone = str(to)
        if zone is None and op in ('kill', 'hand_exit_b', 'swap_place'):
            zone = 'Dead'
        return op, pids, zone

    def expect_echo(self, lines):
        """BRLANE-13: every `$nick` op line the AI emits is an EXPECTED
        ECHO until the client shows it (an ack record, or the feed's
        echo under the AI nick carrying the same pids).  Channel-gated:
        with neither a sidecar nor an ack file (offline suites, old
        jars) nothing is expected and nothing changes."""
        if getattr(self, 'spectator', False) \
                or not self._echo_channel_present():
            return
        for ln in lines or ():
            facts = self._op_line_facts(ln)
            if facts is None:
                continue
            op, pids, zone = facts
            deadline = self._tick_n + self._ECHO_DEADLINE_TICKS
            for p in pids:
                self.echo_expect[p] = {'zone': zone or '?',
                                       'deadline': deadline,
                                       'line': ln, 'expired': False}
            self.findings.append(('emission_echo_expected',
                                  '%s pid %s -> %s'
                                  % (op, ','.join(pids)[:40], zone)))

    def echo_pending(self):
        """True while any emitted line awaits its client echo (an
        EXPIRED one still holds: the self-repair ASK is the way out)."""
        return bool(getattr(self, 'echo_expect', None)
                    or getattr(self, '_post_commit_tail', ())
                    or self.forced_ai_rows_pending())

    def forced_ai_rows_pending(self):
        """An applied engine consequence not yet physically confirmed."""
        return bool(getattr(self, '_forced_ai_rows_holds', ()))

    def _clear_forced_ai_rows(self, held, why):
        """Release only this retained consequence's alarms and physical debt."""
        alarms = {entry['alarm'] for entry in held}
        owned_rows = []
        for entry in held:
            alarms.update(entry.get('refused_alarms', ()))
            alarms.update(entry.get('unacknowledged_alarms', ()))
            owned_rows.extend(entry.get('published_ledger', ()))
        owned_ids = {row['echo_id'] for row in owned_rows if 'echo_id' in row}
        self.echo_ledger[:] = [row for row in self.echo_ledger
                              if row.get('echo_id') not in owned_ids
                              and not any(row is owned for owned in owned_rows)]
        self.alarms[:] = [alarm for alarm in self.alarms if alarm not in alarms]
        self._forced_ai_rows_holds = ()
        self.findings.append(('forced_rows_recovered', why))

    def _confirm_forced_ai_rows(self, lines):
        """Successful native ACKs/echoes finish a published tail once."""
        held = tuple(getattr(self, '_forced_ai_rows_holds', ()))
        if not held:
            return
        context = (self.es.turn, self._cc_battle_serial, self.n_commits_total)
        if any(entry['context'] != context for entry in held):
            return
        for entry in held:
            known = set(entry.get('published_lines', ()))
            entry.setdefault('confirmed_lines', set()).update(known.intersection(lines))
        if all(entry.get('published_lines') and set(entry['published_lines'])
               <= entry.get('confirmed_lines', set()) for entry in held):
            self._clear_forced_ai_rows(held, 'published correction acknowledged by the client')

    def forced_ai_rows_wait(self):
        return ('Board correction required: my committed piece changes could not '
                'be completed. Correct the board and press RESYNC; the paid '
                'action and battle remain retained. %s' %
                str(self._forced_ai_rows_holds[0]['reason'])[:170])

    def human_physical_work_pending(self):
        """Committed human piece changes must be shown before play continues.

        These are outstanding physical removals, replacements or relocations,
        distinct from a still-unanswered engine choice. Claims decrement or
        remove the existing echo ledger entries; silence never satisfies them.
        """
        if getattr(self, '_lenient_board_wait', None) is not None:
            return True
        from .siege_seating import waiting as seating_waiting
        if seating_waiting(self):
            return True
        from .temporary_combat_figures import waiting
        if waiting(self.view,self.es):
            return True
        human = ('fp' if self.human_side.name == 'FREE_PEOPLES' else 'shadow')
        aliases = {'free_peoples': 'fp', 'sh': 'shadow'}
        return any(aliases.get(str(entry.get('side', '')).lower(),
                               str(entry.get('side', '')).lower()) == human
                   and int(entry.get('n', 1)) > 0
                   for entry in (getattr(self, 'echo_ledger', ()) or ()))

    def _release_echo(self, pids, why):
        """The client showed these pids (ack / feed echo): released.
        The BRJAVA ack consumer calls this too (why='ack')."""
        rel = [str(p) for p in (pids or ()) if str(p) in self.echo_expect]
        for p in rel:
            self.echo_expect.pop(p, None)
        if rel:
            self.findings.append(('emission_echo_released',
                                  '%s: pid %s' % (why, ' '.join(rel)[:60])))
        return rel

    def _clear_echo_expect(self, why, expired_only=False):
        gone = [p for p, e in self.echo_expect.items()
                if not expired_only or e.get('expired')]
        for p in gone:
            self.echo_expect.pop(p, None)
        if gone:
            self.findings.append(('echo_expect_cleared',
                                  '%s: pid %s' % (why, ' '.join(gone)[:60])))

    def _check_echo_expiry(self):
        """An emitted line lacks valid confirmation within the deadline:
        finding + sticky alarm + chat + the self-repair ASK (never a
        silent re-render, never an adoption)."""
        due = [p for p, e in self.echo_expect.items()
               if not e.get('expired')
               and self._tick_n >= int(e.get('deadline', 0))]
        if not due:
            return []
        first = dict(self.echo_expect[due[0]])
        # A deadline is evidence of missing confirmation, not completion.
        # Keep the barrier until a positive ACK/board echo or explicit recovery.
        for p in due:
            self.echo_expect[p]['expired'] = True
        msg = ('EMISSION UNACKNOWLEDGED: pid %s lacks valid confirmation at %s '
               'for line %r'
               % (' '.join(due)[:40], first.get('zone'),
                  str(first.get('line'))[:80]))
        self.findings.append(('emission_unacknowledged', msg[:160]))
        if msg not in self.alarms:
            self.alarms.append(msg)
        for entry in getattr(self, '_forced_ai_rows_holds', ()):
            published = set(entry.get('published_lines', ()))
            if all(str(self.echo_expect[p].get('line', '')).strip()
                   in published for p in due):
                entry.setdefault('unacknowledged_alarms', set()).add(msg)
        items = [('note', msg),
                 ('chat', 'prompt', chat_mod.emission_unacknowledged_note(
                     'to %s' % (first.get('zone') or '?')))]
        items.extend(self._propose_repair(msg[:140], clears='manual'))
        return items

    def _consume_inject_acks(self):
        """BRJAVA-3: the candidate jar's per-line ack file releases the
        expected echoes; a refused / unrendered / from-zone-mismatched
        line is a loud finding + alarm + the self-repair ASK."""
        ap = getattr(self, 'ack_path', None)
        if not ap or not os.path.exists(ap):
            return []
        from . import client_protocol as _cpa
        acks = _cpa.read_acks(ap)
        seen = int(getattr(self, '_acks_seen', 0) or 0)
        new = acks[seen:] if seen <= len(acks) else acks
        self._acks_seen = len(acks)
        items = []
        for rec in new:
            raw = str(rec.get('raw', '')).strip()
            from .ack_identity import matched
            pids = matched(self, rec)
            if pids is None:
                continue
            bad = _cpa.refused([rec])
            if bad:
                why = str(bad[0].get('why', ''))
                msg = 'EMISSION REFUSED: client acknowledgement validation failed: %s' % why[:200]
                self.findings.append(('emission_refused', why[:160]))
                if msg not in self.alarms:
                    self.alarms.append(msg)
                for entry in getattr(self, '_forced_ai_rows_holds', ()):
                    if raw in entry.get('published_lines', ()):
                        entry.setdefault('refused_alarms', set()).add(msg)
                items.append(('note', msg))
                # A failed injection is not evidence that the physical
                # board reached the committed state. Keep its existing
                # echo requirement until real confirmation or recovery.
                items.extend(self._propose_repair(msg[:140],
                                                  clears='manual'))
            else:
                self._release_echo(pids, 'ack')
                self._confirm_forced_ai_rows((raw,))
        return items

    # ---- BRLANE-7: the AI's own emitted lines (BL-26 re-assert) --------
    def note_own_emission(self, lines):
        """Every raw op line the server emits is kept (bounded) so a
        human undo that takes it back can be RE-ASSERTED."""
        led = self.__dict__.setdefault('_own_emitted', [])
        for ln in lines or ():
            facts = self._op_line_facts(ln)
            if facts is None:
                continue
            op, pids, zone = facts
            led.append({'raw': ln, 'op': op, 'pids': pids, 'to': zone,
                        'reasserted': False})
        del led[:-80]

    def _reassert_undone_ai_line(self, inner):
        """The emitted op line the undone rendered line `inner` came
        from (newest first, same kind / destination / piece count), or
        None."""
        from .undo_model import canonical_zone
        import rendered_replay as rr
        urec = live_parser.parse(inner)
        op = urec.get('op')
        if op not in ('move', 'kill'):
            return None
        if op == 'move':
            obj = str(urec.get('obj', '') or '')
            toks = (rr.split_group(obj[len('group ('):-1])
                    if obj.startswith('group (') and obj.endswith(')')
                    else [obj])
            n = 0
            for t in toks:
                pp = rr.parse_piece(t)
                n += pp[3] if pp is not None else 1
            want_to = canonical_zone(urec.get('to'))
        else:
            n = 0
            for t in urec.get('pieces', []) or []:
                pp = rr.parse_piece(t)
                n += pp[3] if pp is not None else 1
            want_to = None
        for ent in reversed(list(getattr(self, '_own_emitted', ()))):
            if ent.get('reasserted'):
                continue
            ek = ('kill' if ent['op'] in ('kill', 'hand_exit_b',
                                          'swap_place') else 'move')
            if ek != op or len(ent['pids']) != n:
                continue
            if op == 'move' and canonical_zone(ent.get('to')) != want_to:
                continue
            return ent
        return None

    # ---- BRLANE-2: retryable holds --------------------------------------
    def _retry_holds(self, why):
        """A HELD window is a first-class wait state: re-run the ONE
        gate+match body (_on_window) once its blocking condition
        cleared (the human's pending resolved / the AI's multi-step die
        closed), bounded to _HOLD_RETRY_MAX; a hold whose condition never
        clears expires LOUDLY into the self-repair ASK.  Never a silent
        age-out."""
        if getattr(self, 'spectator', False) or not self.holds \
                or getattr(self, '_retrying_holds', False):
            return []
        import ai as _ai
        self._retrying_holds = True
        items = []
        try:
            try:
                dp = _ai.describe_pending(self.es)
            except Exception:
                dp = None
            ai_half = (self.es.pending_army_move_die is not None
                       or getattr(self.es, 'pending_muster_die', None)
                       is not None)
            for w, res in list(self.holds):
                cond = getattr(w, '_hold_cond', None)
                if cond is None:
                    continue
                if cond == 'decision_endpoint':
                    # A completed transaction can precede its native export.
                    # Retry on new board evidence, not on elapsed time, and
                    # never turn transport latency into an Undo request.
                    revision = self._decision_endpoint_revision()
                    if revision == getattr(w, '_endpoint_revision', None):
                        continue
                    self.holds = [h for h in self.holds if h[0] is not w]
                    for record in w.records:
                        norm = record.get('norm', '')
                        if norm in self.uncommitted:
                            self.uncommitted.remove(norm)
                    w._hold_cond = None
                    items.extend(self._on_window(w) or [])
                    continue
                norms = [r.get('norm', '') for r in w.records]
                desc = (norms[1] if len(norms) > 1 else
                        (norms[0] if norms else '?'))[:80]
                cleared = ((cond == 'pending' and dp is None)
                           or (cond == 'die_identity' and not self._needs_staged_die_identity(w))
                           or (cond == 'ai_multistep' and not ai_half)
                           or (cond == 'divergence'
                               and not self._divergence_open()))
                if not cleared:
                    age = self._tick_n - int(getattr(w, '_hold_tick',
                                                     self._tick_n))
                    if age >= self._HOLD_EXPIRE_TICKS \
                            and not getattr(w, '_hold_expired', False):
                        w._hold_expired = True
                        self.findings.append(('hold_expired',
                                              '%s after %d ticks :: %s'
                                              % (cond, age, desc)))
                        msg = ('HELD WINDOW EXPIRED: %s never became '
                               'playable (%s still open after %d ticks) '
                               '- undo it and redo it' % (desc, cond, age))
                        if msg not in self.alarms:
                            self.alarms.append(msg)
                        items.append(('chat', 'prompt',
                                      chat_mod.held_window_expired(desc)))
                        items.extend(self._propose_repair(
                            'HELD WINDOW EXPIRED: %s' % desc,
                            clears='hold'))
                    continue
                retries = int(getattr(w, '_hold_retries', 0)) + 1
                w._hold_retries = retries
                self.holds = [h for h in self.holds if h[0] is not w]
                for n in norms:
                    if n in self.uncommitted:
                        self.uncommitted.remove(n)
                self.findings.append(('hold_retried',
                                      '%s cleared (%s) retry %d :: %s'
                                      % (cond, why, retries, desc)))
                if retries > self._HOLD_RETRY_MAX:
                    self.window_stats['none'] += 1
                    self.uncommitted.extend(norms)
                    self.findings.append(('hold_expired',
                                          'retry bound %d :: %s'
                                          % (self._HOLD_RETRY_MAX, desc)))
                    items.extend(self._refuse(w, gesture_match.MatchResult(
                        'none', note='your held gesture could not be '
                                     'replayed after %d retries - undo '
                                     'it and redo it'
                                     % self._HOLD_RETRY_MAX)))
                    continue
                w._hold_cond = None
                n_matched = self.window_stats['matched']
                items.extend(self._on_window(w) or [])
                if self.window_stats['matched'] > n_matched:
                    self.findings.append(('hold_retry_matched', desc))
        finally:
            self._retrying_holds = False
        return items

    def _tag_hold(self, w, cond):
        w._hold_cond = cond
        if getattr(w, '_hold_tick', None) is None:
            w._hold_tick = self._tick_n

    # ---- BRLANE-5: die-less human gestures ------------------------------
    def _wwsat_modal_expected_line(self, ln):
        """FLA-20 / FLH H2 (par5 g4#5): the client watchdog's 'modal
        dialog open: Confirm Random Discard' line is EXPECTED
        TELEMETRY while the engine is starving on the client's own
        WWSAT card_pick - it is the confirm the jar's yes-arm should
        have answered, not a client failure.  Consumed as evidence
        for the ONE wwsat re-send; every other modal line stays a
        halt."""
        t = str(ln)
        if 'modal dialog open' not in t \
                or 'Confirm Random Discard' not in t:
            return False
        try:
            p = self.rand.pending() if self.rand is not None else None
        except Exception:
            p = None
        if getattr(p, 'kind', None) != 'card_pick':
            return False
        import time as _t20
        self._wwsat_modal_ts = _t20.time()
        self.findings.append(('wwsat_modal_telemetry', t[:120]))
        return True

    def _dieless_gesture(self, rec):
        """A HUMAN board record with no die staged and no pending open:
        matched against the engine's DIE-LESS rows (tokens, dwarven
        rings) through the ONE matcher, or REFUSED (chat + one client-
        native undo press + finding).  Never a silent drop (BRLANE-5:
        the dropped drag was adopted without a die by the next census).
        Inert bookkeeping and dice bookkeeping stay engine-derived."""
        if getattr(self, 'spectator', False):
            return []
        if str(rec.get('player', '')) == self.nick:
            return []
        op = rec.get('op')
        if op not in ('move', 'kill'):
            return []
        norm = str(rec.get('norm', '') or '')[:80]
        if self._is_dice_bookkeeping(rec):
            if op == 'kill' and str(rec.get('zone', '')).strip().lower() \
                    == 'current action' and self.seg.window is None:
                self.findings.append(('orphan_die_kill', norm))
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'that die kill closed no action - stage the die, '
                    'act, then kill it'))]
            if op == 'move' and gesture_match.zone_key(
                    rec.get('to'))[0] == 'reinf':
                # SQF-6 (seqack_fp1 findings 6+7): the client's
                # recover parked a GRANTED, engine-UNSPENT die in a
                # reinforcements box (the frecover Controls.784
                # artifact) - the client pool then under-rolls.  Loud
                # + the counter-drag demanded by pid; a spent/unknown
                # die stays plain bookkeeping.
                from . import sidecar as _sc6
                _dmap6 = getattr(self.view, 'die_id_map', {}) or {}
                for _p6 in (_sc6.meta_pids(rec.get('meta')) or []):
                    _si6 = _dmap6.get(str(_p6))
                    if _si6 is None:
                        continue
                    try:
                        _d6 = self.es.dice[_si6[0]][int(_si6[1])]
                    except (KeyError, IndexError, TypeError,
                            ValueError):
                        continue
                    if not _d6.used:
                        self.findings.append(
                            ('die_parked_in_reinforcements',
                             '%s :: %s' % (_p6, norm)))
                        return [('chat', 'prompt',
                                 chat_mod.die_parked_note(_p6))]
            return []
        if self._rec_inert_here(rec):
            return []
        if self.seg.window is not None:
            return []
        import ai as _ai
        dieless = []
        # B-W2-05 (i) (COORD #2 2026-09-03, USER bridge go): the die-less
        # lane serves the HUMAN's own pending decision only.  While the
        # OPPONENT's combat pending is open (the AI attacker's
        # victor_advance after the human garrison withdrew, W2-025) no
        # human drag may decide it: describe_pending is the one answerer
        # for whose decision is open (the engine's W2-032 ownership
        # returns [] for the other side as well - belt and braces).
        _owned9 = True
        try:
            _dp9 = _ai.describe_pending(self.es)
            if _dp9 is not None and _dp9.get('side') is not self.human_side:
                _owned9 = False
        except Exception:
            _owned9 = True
        if self.phase == 'actions' and _owned9 \
                and not self._human_out_of_turn():
            try:
                if self.rand is not None:
                    with self.rand.pin():
                        menu = _ai.get_legal_actions(self.es,
                                                     self.human_side)
                else:
                    menu = _ai.get_legal_actions(self.es, self.human_side)
            except Exception:
                menu = []
            dieless = [a for a in menu
                       if (a.die_idx is None or a.die_idx < 0)
                       and a.action_type not in ('pass', 'elven_ring')]
        if dieless:
            shim = PhaseWindow()
            shim.records.append(rec)
            shim.combat_cards_open = self._combat_cards_open()
            try:
                res = gesture_match.match_window(
                    self.es, self.human_side, shim, menu=dieless,
                    pin=self._attempt_pin(), rng=self.rctl,
                    advance_tolerance=True, fs_prefilter=True)
            except Exception as e:
                res = gesture_match.MatchResult('none', note=repr(e)[:80])
            if res.status == 'match' and self._commit_chain(
                    res.chain, [norm], 'dieless', gesture=shim):
                self.findings.append(('dieless_commit',
                                      '%s :: %s' % (res.chain[0].action_type,
                                                    norm)))
                self.last_actor = self.human_side
                return []
            if self.rand is not None:
                self.rand.reset_feed()
        if self._divergence_open():
            # FLA-10 (par5 g3#6b): the census 'no' TELLS the human to
            # move the pieces to match the engine, and this lane then
            # refused exactly those drags ('stage an action die
            # first') and undid them - a repair catch-22.  While a
            # confirmed divergence stands a die-less drag/kill is
            # repair traffic: accepted client-side, engine untouched,
            # the next quiescent census re-judges the board.
            self.findings.append(('repair_drag_accepted',
                                  '%s :: %s' % (op, norm)))
            return [('note', 'repair drag accepted (divergence '
                             'open): %s' % norm)]
        zk = gesture_match.zone_key(rec.get('zone') if op == 'kill'
                                    else rec.get('to'))
        if op == 'kill' and zk[0] == 'hand':
            why = ('no discard is owed now - undo that card kill (a '
                   'hand-limit discard is asked for when your hand is '
                   'over 6)')
        elif op == 'kill':
            why = 'no casualty or removal is owed now - undo that removal'
        else:
            why = ('stage an action die first, then move - no die-less '
                   'action fits that drag')
        self.findings.append(('dieless_gesture_refused',
                              '%s :: %s' % (op, norm)))
        return self._refuse_phase_line(rec, chat_mod.refusal_illegal(why),
                                       'DIE-LESS GESTURE')

    # ---- item 16 / M7: the battle resolved by hand ----------------------
    def _battle_by_hand_check(self, dp, rec):
        """A record buffered into the HUMAN's open combat decision that
        moves the OPPONENT's pieces is the human resolving the battle by
        hand (USER 08-22 T4): say what the engine still owes and offer
        the self-repair AT ONCE - never a 25-tick silent stall, never
        an adoption."""
        if getattr(self, 'spectator', False) or rec.get('op') != 'move':
            return []
        kind = str(dp.get('kind', ''))
        if not kind.startswith('combat:') \
                or dp.get('side') is not self.human_side:
            return []
        key = (kind, self.n_commits_total)
        if getattr(self, '_battle_by_hand_said', None) == key:
            return []
        import ai as _ai
        import game_state as _gs
        import rendered_replay as rr
        obj = str(rec.get('obj', '') or '')
        toks = (rr.split_group(obj[len('group ('):-1])
                if obj.startswith('group (') and obj.endswith(')')
                else [obj])
        minions = set(rr.NAMED_MINIONS.values())
        foreign = False
        for t in toks:
            pp = rr.parse_piece(t)
            if pp is None:
                continue
            k = pp[0]
            if k in ('unit', 'leader'):
                nat = gesture_match._nation_enum(pp[1])
                if nat is not None \
                        and _ai._side_of_nation(nat) is self.ai_side:
                    foreign = True
            elif k == 'nazgul' and self.ai_side is _gs.Side.SHADOW:
                foreign = True
            elif k == 'named':
                is_minion = pp[1] in minions
                if (is_minion and self.ai_side is _gs.Side.SHADOW) or (
                        not is_minion
                        and self.ai_side is _gs.Side.FREE_PEOPLES):
                    foreign = True
        if not foreign:
            return []
        self._battle_by_hand_said = key
        what = ('combat card' if kind.endswith('_combat_card')
                else kind.split(':')[-1].replace('_', ' '))
        norm = str(rec.get('norm', '') or '')[:80]
        self.findings.append(('battle_resolved_by_hand',
                              '%s :: %s' % (kind, norm)))
        msg = ('BATTLE RESOLVED BY HAND: %s while the engine owes the '
               'human its %s decision' % (norm, what))
        if msg not in self.alarms:
            self.alarms.append(msg)
        items = [('chat', 'prompt', chat_mod.battle_by_hand_ask(
            what, self._pending_words(dp)))]
        items.extend(self._propose_repair(
            'BATTLE RESOLVED BY HAND: %s' % self.stall_text(),
            clears='pending'))
        return items

    # ---- BRLANE-12: the card target chatted over the play window -------
    def _apply_card_target_hint(self, hint):
        """The play committed and opened the human's card_option
        decision: the region chatted while the window was open answers
        it (the ONE chat resolver, match_choice)."""
        if not hint:
            return []
        import ai as _ai
        try:
            dp = _ai.describe_pending(self.es, self.human_side)
        except Exception:
            dp = None
        if dp is None or dp['side'] is not self.human_side \
                or str(dp['kind']) != 'card_option':
            return []
        act = chat_mod.match_choice(hint, dp['actions'])
        if act is None:
            self.findings.append(('card_target_hint_unmatched',
                                  str(hint)[:40]))
            return []
        items = self._commit_human(act, via='card-target-chat')
        self.findings.append(('card_target_from_chat', str(hint)[:40]))
        if self.chat_deferred is None:
            items.append(('chat', 'prompt', chat_mod.pick_committed_note(
                'card target: %s' % str(hint)[:40])))
        return items

    # ---- item 14 / M10: the card leaving the hand IS the play ----------
    def _card_identity_probe(self, w, side, res):
        """A window carrying ONE named hand exit of the human's side
        whose full match failed (not starved): the play_event row for
        that card on the staged face is the gesture - a MatchResult
        'match' carrying it (identity_fallback set), a starved hold when
        the play awaits client randomness, else None."""
        if getattr(self, 'spectator', False) or not self.enforce_human \
                or getattr(res, 'invalid_guards_recruit', False):
            return None
        try:
            facts = gesture_match.window_facts(w)
        except Exception:
            return None
        if len(facts.cards_played) != 1 or facts.unmapped:
            return None
        if facts.ring_moves:
            return None  # a ring's selected face must match too
        cid = gesture_match.title_to_cid(facts.cards_played[0])
        face = getattr(w, 'die_face', None)
        if cid is None or not face:
            return None
        import ai as _ai
        from . import randomness as rnd_mod
        try:
            with self.rand.pin():
                menu = _ai.get_legal_actions(self.es, side)
        except Exception:
            return None
        rows = [a for a in menu
                if a.action_type == 'play_event'
                and a.params.get('card_id') == cid
                and gesture_match._die_filter(self.es, side, face)(a)]
        if not rows:
            return None
        # the staged face's own row first (a W die is wild); same-face
        # dice are fungible (BL-108 rebinds the idx) - only genuinely
        # different option params make the play ambiguous
        dice = self.es.dice.get(side, [])

        def _letter(a):
            try:
                return gesture_match.FACE_LETTER.get(
                    getattr(dice[a.die_idx].current, 'name', ''), '?')
            except (IndexError, TypeError):
                return '?'
        exact = [a for a in rows if _letter(a) == face] or rows
        by_params = {}
        for a in exact:
            by_params.setdefault(
                repr(sorted((str(k), str(v))
                            for k, v in (a.params or {}).items())), a)
        if len(by_params) != 1:
            return None
        row = next(iter(by_params.values()))
        try:
            with self._attempt_pin()():
                post = _ai.apply_action(self.es, row)
        except rnd_mod.NeedClientCall as e:
            mr = gesture_match.MatchResult(
                'hold', starved=True,
                note='card identity %s awaits client %s'
                     % (cid, getattr(e, 'kind', '?')))
            mr.starve_call = e
            return mr
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return None
        mr = gesture_match.MatchResult(
            'match', chain=[row], post=post,
            note='card identity (%s: %s)' % (res.status, res.note or ''))
        mr.identity_fallback = (res.status, res.note)
        return mr

    def _explain_card_play_extras(self, w, res, pre):
        """After a card-identity commit: the window rows the engine did
        NOT produce are EXPLAINED in chat (finding), never refused."""
        cid = str((res.chain[0].params or {}).get('card_id', '?'))
        status, note = res.identity_fallback
        self.findings.append(('card_play_identity_commit',
                              '%s (%s: %s)' % (cid, status,
                                               str(note or '')[:60])))
        try:
            facts = gesture_match.window_facts(w)
            eng = gesture_match.delta_facts(pre, self.es)
            diff = [d for d in gesture_match._facts_differ(
                facts, eng, action=res.chain[0]) if d != 'the card played']
        except Exception as e:
            self.findings.append(('refusal_explain_failed', repr(e)[:80]))
            return []
        if not diff:
            return []
        self.findings.append(('card_play_extra_rows',
                              '%s :: %s' % (cid, ', '.join(diff)[:80])))
        return [('chat', 'prompt', chat_mod.card_play_extra_rows(
            chat_mod.nice_card(cid), ', '.join(diff[:3])))]

    # ---- EX-8: the human Shadow's physical table-kill payment ----------
    def _table_kill_payment_filter(self, dp, menu):
        """Bind both physical discards before selecting the table-card payment."""
        pco = getattr(self.es, 'pending_card_option', None)
        if not (isinstance(pco, dict) and pco.get('table_kill_payment')):
            return menu
        import ai as _ai
        import game_state as _gs
        from .sidecar import meta_pids
        buf = self.pending_buf
        if buf is None:
            return None
        buf._awaits_table_payment_identity = True
        records = buf.records
        if (not records or any(rec.get('op') != 'kill'
                              or gesture_match.zone_key(rec.get('zone')) != ('hand', 'SH')
                              for rec in records)):
            return None
        facts = gesture_match.window_facts(buf)
        if sum(facts.hand_discards.values()) != 2:
            return None
        if any(owner != 'SHADOW' or (deck is not None and count > 1)
               for (owner, deck), count in facts.hand_discards.items() if count):
            return None
        hand = list(self.es.hands[_gs.Side.SHADOW].hand)
        mapping = self.view.card_pid_map or {}
        inverse = {str(pid): str(cid) for cid, pid in mapping.items()}
        proven = set()
        complete = True
        for rec in records:
            pids = meta_pids(rec.get('meta'))
            if pids is not None:
                cids = [inverse.get(pid) for pid in pids]
                if any(cid not in hand for cid in cids):
                    return None  # Explicit conflicting identities cannot fall back to a dump.
                proven.update(cids)
            else:
                titles = [gesture_match.title_to_cid(title) for title in rec.get('pieces', ())]
                proven.update(cid for cid in titles if cid is not None)
                complete = complete and bool(titles) and all(cid is not None for cid in titles)
        if complete:
            gone = sorted(proven)
        else:
            # A complete but older export can still place newly drawn cards
            # in their decks. Absence from the hand alone is not a discard.
            try:
                if any(str(cid).startswith('HU_') or str(cid) not in mapping for cid in hand):
                    return None
                from .repair_evidence import dump_signature, same_dump
                from .client_protocol import dump_at_feed_position
                import copy
                evidence = dump_signature(self)
                header = evidence['header']
                if not header or int(header.get('session', 0)) <= 0:
                    return None
                self._check_dump_session(header)
                reader = copy.deepcopy(self.feed)
                reader.poll()
                if (reader._buf or self._feed_seq != len(reader.lines)
                        or not dump_at_feed_position(header, len(reader.lines))
                        or not self.view.refresh_dump() or not same_dump(self, evidence)):
                    return None
                locations = self.view.piece_region
                if any(str(mapping[cid]) not in locations for cid in hand):
                    return None
                gone = sorted(cid for cid in hand if str(locations[str(mapping[cid])]).lower()
                              in ('sa dead', 'sa casualties'))
                if not proven <= set(gone):
                    return None
            except (OSError, ValueError, TypeError):
                return None
        army, chars = _ai._table_kill_payment_cards(self.es)
        ga = [c for c in gone if c in army]
        gc = [c for c in gone if c in chars]
        if len(ga) > 1 or len(gc) > 1:
            self.findings.append(('table_kill_payment_ambiguous',
                                  '%s' % (gone[:4],)))
            self._out_queue.append(('chat', 'refusal',
                                    chat_mod.refusal_illegal(
                                        'the table-card kill is paid with '
                                        'exactly one army card and one '
                                        'character card - several left '
                                        'your hand; undo the extra kills')))
            return None
        if len(ga) == 1 and len(gc) == 1:
            want = {'army_card': ga[0], 'character_card': gc[0]}
            kept = [a for a in menu
                    if (getattr(a, 'params', None) or {}).get('choices')
                    == want]
            if kept:
                buf._awaits_table_payment_identity = False
                self.findings.append(('table_kill_payment_from_identity' if complete
                                      else 'table_kill_payment_from_dump',
                                      '%s + %s' % (ga[0][:40], gc[0][:40])))
                return kept
            self.findings.append(('table_kill_payment_ambiguous',
                                  'pair not offered: %s' % (want,)))
            self._out_queue.append(('chat', 'refusal',
                                    chat_mod.refusal_illegal(
                                        'that pair cannot pay the table-'
                                        'card kill - one army-type card '
                                        'and one character card')))
            return None
        return None            # the dump has not shown both kills yet

    # ---- BRLANE-14: a human-deck draw demand goes to chat ---------------
    def _ask_human_draw(self, p, items_):
        if (p is self.pending_random_call()
                and getattr(p, '_physical_request_owner', None) is self.rand
                and getattr(p, '_physical_base_state', None) is self.es
                and getattr(self.es, 'pending_combat', None)):
            stopped = getattr(p, '_physical_pre_request_state', None)
            if stopped is not None:
                staged = self._stage_human_casualty_prefix(stopped)
                staged.extend(self._stage_ai_casualty_prefix(stopped,
                    expressed=tuple(getattr(p, '_physical_expressed', ()))
                              + tuple(getattr(self.pending_buf, '_pre_rng_expressed', ()))))
                staged.extend(self._auto_wiped_figure_cleanup(expected_state=stopped))
                if staged or self.human_physical_work_pending() or self.echo_pending():
                    return staged
        key = ('draws', tuple(str(x) for x in items_))
        if getattr(self, '_human_draw_asked', None) == key:
            return []
        self._human_draw_asked = key
        if len(items_) == 1:
            sn, dk = items_[0]
            label = ('%s %s' % ('free' if str(sn) == 'FREE_PEOPLES'
                                else 'shadow', dk))
            self.findings.append(('human_draw_asked', label))
            return [('chat', 'prompt', chat_mod.draw_ask(label))]
        self.findings.append(('human_draw_asked',
                              'ambiguous: %s' % (list(items_)[:3],)))
        return [('chat', 'prompt', chat_mod.draw_ask_ambiguous())]

    def _chat_context_hint(self):
        """What IS accepted right now, for the not-a-command reply."""
        if self._human_window_open():
            return 'kill the staged die to complete your action'
        if self.phase == 'actions' and self._human_out_of_turn():
            return (self._TURN_OVER_DETAIL if self._turn_over_pending()
                    else 'it is not your action turn')
        if self.phase == 'actions':
            return 'stage an action die to act, or say pass'
        return 'the turn start is in progress (%s)' % self.phase

    # ------------------------------------------------------------------
    def _poll_routable_feed(self):
        """Join native paragraphs to their per-line metadata before routing.

        AiFeed flushes the two files separately. Once that channel is present,
        a missing record is unfinished delivery, not evidence of empty tags or
        identities. Keep the unread suffix in FeedReader and resume it exactly
        once. Plain chat and inert notices need no metadata; later board work
        cannot overtake an earlier unfinished native statement.
        """
        self.feed.poll()
        sc = self.sidecar
        if sc is not None:
            sc.poll()
            if sc.gen_change is not None:
                raise DriverHalt('FEED RE-ATTACH: native metadata generation changed')
        if getattr(self, '_grouped_die_identity_wait', None) is not None:
            return []
        rows = []
        previous = getattr(self, '_sidecar_wait_seq', None)
        self._sidecar_wait_seq = None
        for seq in range(self._feed_seq + 1, len(self.feed.lines) + 1):
            raw = self.feed.lines[seq - 1]
            if sc is not None and sc.present and sc.rec(seq) is None:
                op = live_parser.parse(raw).get('op')
                if op not in ('chat', 'system', 'noop', 'blank'):
                    self._sidecar_wait_seq = seq
                    if previous != seq:
                        self.findings.append(('sidecar_gap',
                                              'waiting for metadata at seq %d' % seq))
                    break
            rows.append((seq, raw))
        return rows

    def tick(self, raw_request_lines=()):
        """One cycle: ingest feed + the request's raw lines (sanctioned
        imports only), advance, return response items:
        [('line', s) | ('call', s) | ('chat', kind, text)]."""
        out = []
        self._raw_request_lines = list(raw_request_lines)
        # Native injection writes its acknowledgement before the next
        # request. Release those completed board updates before routing
        # human replies in that request (not after refusing their fp ok).
        out.extend(self._consume_inject_acks())
        # Prefer the record's explicit die channel. Native menu actions may
        # have an empty sidecar; their exact public raw command is a second
        # witness, paired against rendered faces and checked for conflicts.
        sc = self.sidecar
        if sc is not None:
            sc.poll()
            if sc.gen_change is not None:
                raise DriverHalt('FEED RE-ATTACH: native metadata generation changed')
        if raw_request_lines:
            from .native_die_identity import observe_raw as observe_die_moves
            observe_die_moves(self, raw_request_lines)
        from .staged_dice import retry_identity as retry_grouped_die_identity
        out.extend(retry_grouped_die_identity(self))
        if sc is not None and sc.present and raw_request_lines:
            self._queue_public_roll_witnesses(raw_request_lines)
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
            ready = self._poll_routable_feed()
        except FeedRestart as e:
            raise DriverHalt(
                'FEED RESTART: %s - client restarted or a new game '
                'began; restart the driver against the new feed.' % e)
        # ONE answerer for 'feed line number': the reader's own line
        # index (seq is 1-based over feed.lines).  Every line advances
        # it, routed or not, so the sidecar alignment survives any
        # per-line routing error.
        new = [raw for _seq, raw in ready]
        for seq, raw in ready:
            if seq <= self._feed_seq:
                continue  # A recovery in this batch retired its covered prefix.
            self._feed_seq = seq
            try:
                out.extend(self._ingest_line(raw, sc))
                from .human_draw_identity import publish as publish_draw_identity
                publish_draw_identity(self)
                if getattr(self, '_grouped_die_identity_wait', None) is not None:
                    break
            except NativeDieFaceMismatch:
                # This invalidates the whole physical action boundary. Let
                # the live handler restore its paired checkpoint immediately;
                # swallowing the error would label this partial route stable.
                raise
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
        # BRJAVA-1: a client complaint that reached the feed ('<game> AI
        # ERROR ...') is a LOUD halt + the self-repair ASK, never a
        # silent continue (one predicate, client_protocol.halt_verdict;
        # live_server scans the raw request log with the same one)
        if new and not getattr(self, 'spectator', False):
            from . import client_protocol as _cp1
            _halt1, _why1 = _cp1.halt_verdict(
                [l for l in new
                 if not self._wwsat_modal_expected_line(l)])
            if _halt1:
                self.findings.append(('client_error_line', _why1[:160]))
                if _why1 not in self.alarms:
                    self.alarms.append(_why1)
                self._out_queue.extend(self._propose_repair(
                    _why1[:140], clears='manual'))
                raise DriverHalt(_why1)
        if (getattr(self, '_sidecar_wait_seq', None) is not None
                or getattr(self, '_grouped_die_identity_wait', None) is not None):
            # Preserve the completed prefix's outgoing work. The server keeps
            # its open step and native requests until the same suffix resumes.
            self._out_queue = out + self._out_queue
            return []
        # A chat reply can reach AiFeed before the next native board export.
        # Retain an explicit phase-close request and retry on fresh evidence;
        # transport latency is not an illegal Fellowship declaration.
        from .discard_recovery import poll as poll_hand_discard_endpoint
        poll_hand_discard_endpoint(self, new)
        if getattr(self, '_fellowship_completion_requested', False):
            if self.phase in ('declare', 'guide'):
                out.extend(self._on_chat({'body': 'fp ok', 'nick': 'completion retry'}))
            else:
                self._fellowship_completion_requested = False
        from .allocation_roll import complete as complete_allocation_roll
        out.extend(complete_allocation_roll(self))
        if getattr(self, '_allocation_roll_wait', None) is None:
            self._complete_human_eye_allocation()
        # BRLANE-2: held windows whose blocking condition cleared are
        # retried BEFORE the queue drains (their items ride this tick)
        out.extend(self._retry_holds('tick'))
        from .lenient_board import poll as reconcile_lenient_board
        out.extend(reconcile_lenient_board(self))
        endpoint_buffer = self.pending_buf
        if getattr(endpoint_buffer, '_victor_close_record', None) is not None:
            self._try_pending()
        if (endpoint_buffer is not None
                and getattr(endpoint_buffer, '_hold_cond', None) == 'decision_endpoint'
                and self._decision_endpoint_revision() != getattr(endpoint_buffer, '_endpoint_revision', None)):
            self._try_pending()
        waiting_attack = getattr(self, '_chat_attack_identity_wait', None)
        if waiting_attack is not None:
            window, spec = waiting_attack
            if self.seg.window is not window:
                self._chat_attack_identity_wait = None
            elif not self._needs_staged_die_identity(window):
                self._chat_attack_identity_wait = None
                out.extend(self._human_chat_attack(spec))
        # A generic hand-card kill may reach the feed before the matching
        # native dump. Retry that SAME buffered transaction when its identity
        # witness was unavailable; no second human gesture is required.
        if (self.pending_buf is not None
                and getattr(self.pending_buf, '_awaits_discard_identity', False)):
            import ai as _discard_ai
            _discard_dp = _discard_ai.describe_pending(self.es)
            if (_discard_dp is not None and _discard_dp['kind'] == 'discard'
                    and self._pending_buffer_applies(_discard_dp)):
                self._try_pending()
        if (self.pending_buf is not None
                and getattr(self.pending_buf, '_awaits_table_payment_identity', False)):
            import ai as _payment_ai
            _payment_dp = _payment_ai.describe_pending(self.es)
            if (_payment_dp is not None and _payment_dp['kind'] == 'card_option'
                    and self._pending_buffer_applies(_payment_dp)):
                self._try_pending()
        # Announce an incomplete Hunt marker drag after the whole feed
        # batch has routed, so an already-completed split drag is silent.
        from .fellowship_reveal_gestures import retry as retry_reveal_placement
        retry_reveal_placement(self)
        out.extend(self._partial_hunt_corruption_note())
        from .casualty_gestures import live_completion_note
        out.extend(live_completion_note(self))
        out.extend(self._temporary_combat_figure_note())
        from .guide_completion import poll as complete_visible_guide
        complete_visible_guide(self)
        out.extend(self._retry_human_pending_prompt())
        if self._out_queue:
            # items minted by nested deferral retries (refusal undo
            # presses, holding notes) — surfaced before the AI acts
            out.extend(self._out_queue)
            self._out_queue = []
        # refusal-restore verification (armed-only, dump-read)
        self._check_restore_watch()
        # FLA-8: the repair-slot first-seen ledger accrues per tick
        if not getattr(self, 'spectator', False):
            self._note_repair_slots()
            # FLC H1 (par5 g1#12): the activation-map tripwire
            self._political_activation_witness()
        # R6: sidecar-present roll with no die channel -> alarm
        self._tick_n += 1
        self._check_roll_binding_starvation()
        # a pending buffer that stops moving is a JAM (a stall with no
        # counter is invisible): one loud note per 25 stalled ticks;
        # the census row keeps it on the books
        from .siege_withdrawal_gestures import retry as retry_siege_withdrawal
        siege_withdrawal_wait = retry_siege_withdrawal(self)
        if self.pending_buf is not None and self.pending_buf.records:
            import ai as _ai
            _dp = _ai.describe_pending(self.es)
            from .victor_completion import active as victor_completion_active
            victor_wait = (victor_completion_active(self, _dp)
                           and getattr(self.pending_buf, '_victor_close_record', None) is None)
            if siege_withdrawal_wait or victor_wait:
                self._buf_stall = None
            elif _dp is None and (
                    isinstance(getattr(self.es, 'pending_combat', None),
                               dict)
                    or self.human_cc_committed):
                # K1 reveal gap: dp is None because the battle is
                # parked on a hidden combat card; the buffer holds
                # battle-gap records for the decision the reveal opens
                # NEXT - not orphaned
                pass
            elif _dp is None:
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
            elif self._partial_hunt_corruption() is not None:
                # An identified partial allocation is waiting for the
                # player's remainder or undo, not a stalled matcher.
                self._buf_stall = None
            elif self._partial_declaration_wait(_dp):
                # Moving the pawn and resetting/flipping it are separate
                # native gestures. Thinking time between them is not a stall.
                self._buf_stall = None
            elif self._partial_casualty_wait() is not None:
                # A proved partial Elite payment is a human gesture still in
                # progress, not a matcher failure or permission to self-repair.
                self._buf_stall = None
            elif self._casualty_roll_context_only() is not None:
                # The round's consumed roll remains contextual evidence. The
                # player still owes a casualty, not a correction to that roll.
                self._buf_stall = None
            elif self._pending_human_roll_wait() is not None:
                # The exact physical transaction has already issued a human
                # roll request. Silence is not an illegal gesture; actual
                # rendered-but-unharvested results retain the diagnostic path.
                self._buf_stall = None
            elif self._card_completion_wait():
                # Terminal choices are deliberately withheld while the human
                # assembles an optional group. Only done/card/die disposal
                # closes it; idle time cannot turn that open boundary into
                # an unmatched-action failure or permission to repair it.
                self._buf_stall = None
            elif len(self.pending_buf.records) > 40:
                # BL-142: the stall counter keys on the record COUNT,
                # so a GROWING buffer is a WEDGE - hold the pending
                # open, release the records loudly (DIVERGENCE-class:
                # the seat stops), never an eternal silent fill
                n_w = len(self.pending_buf.records)
                self.alarms.append(
                    'PENDING BUFFER WEDGED: %d record(s) never matched '
                    'the open %s decision - the gesture does not equal '
                    'any legal option; %d record(s) released, RESYNC/'
                    'undo owed' % (n_w, _dp['kind'], n_w))
                self._expire_pending_live(_dp, self.pending_buf,
                                          'buffer wedged at %d' % n_w)
                self._buf_stall = None
                out.extend(self._propose_repair(
                    'PENDING BUFFER WEDGED: %d record(s) never matched '
                    'the open %s decision' % (n_w, _dp['kind'])))
            else:
                fp = (len(self.pending_buf.records),
                      self.n_commits_total)
                if getattr(self, '_buf_stall', None) \
                        and self._buf_stall[0] == fp:
                    self._buf_stall = (fp, self._buf_stall[1] + 1)
                    if self._buf_stall[1] % 25 == 0 \
                            and self._divergence_open():
                        # FLD-6c (par5b_g2 stale-line poisoning):
                        # while a divergence stands, the census-'no'
                        # contract has the human dragging REPAIR
                        # traffic - records that jam an open decision
                        # for a full stall window are that traffic,
                        # not answers (they matched nothing for 25
                        # ticks).  Release them client-side (engine
                        # untouched, the next quiescent census
                        # re-judges); the pending stays open for its
                        # real answer.
                        _n6c = len(self.pending_buf.records)
                        for _r6c in self.pending_buf.records:
                            self.findings.append(
                                ('repair_drag_accepted',
                                 'stalled-buffer release :: %s'
                                 % str(_r6c.get('norm', ''))[:80]))
                        self._unreveal(self.pending_buf)
                        self.pending_buf = None
                        self._buf_stall = None
                        out.append(('note',
                                    'divergence open: %d unmatched '
                                    'line(s) released as repair '
                                    'traffic - the %s decision is '
                                    'still open'
                                    % (_n6c, _dp['kind'])))
                    if self._buf_stall is not None \
                            and self._buf_stall[1] % 25 == 0:
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
                        # the FIRST stall signal offers self-repair;
                        # later ones only repeat the note (the offer
                        # stays on the status line, never re-chatted)
                        out.extend(self._propose_repair(
                            'PENDING BUFFER STALLED: %s'
                            % self.stall_text()))
                else:
                    self._buf_stall = (fp, 1)
        else:
            self._buf_stall = None
        out.extend(self._repair_tick())
        # BRLANE-13 / BRJAVA-3: the client's acks release the expected
        # echoes (refusals are loud); an echo past its deadline is loud
        out.extend(self._consume_inject_acks())
        out.extend(self._check_echo_expiry())
        out.extend(self._auto_wiped_figure_cleanup())
        out.extend(self._flush_post_commit_consequences())
        out.extend(self._auto_forced_siege_step())
        from .attack_chat import poll as poll_attack_nc
        out.extend(poll_attack_nc(self))
        out.extend(self._resume_uncommitted_card_selection())
        out.extend(self._ai_turn())
        # a deferred window / pending buffer starving on client
        # randomness surfaces its ask so live_server can ORDER it
        # (!CALL for the AI's own dice, a chat ask for the human's).
        # ONLY d6/tile/randomcomp asks are auto-ordered: rand.pending()
        # is the LAST starving candidate's call, not provably the
        # gesture's - an excess roll/draw is quarantined by the honesty
        # gate, but a wrong '!CALL draw_*' would draw a real card (CARD
        # kinds stay excluded; the randomcomp ask is USER-mandated chat
        # + a parser-inert builtin).  Other kinds surface as a loud
        # note.
        if self.deferred is not None or self.pending_buf is not None \
                or self.chat_deferred is not None:
            p = self.pending_random_call()
            if p is None and self.pending_buf is None and self.deferred is not None:
                # the global pending slot may be clobbered by an
                # interleaved attempt - the deferral's own captured
                # starve call is the surviving truth
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
                elif p.kind == 'draws' and _items:
                    # BRLANE-14 / F02: a draw the engine wants from the
                    # HUMAN's deck is asked in CHAT (once per demand),
                    # not only as a panel note
                    out.extend(self._ask_human_draw(p, _items))
                    if p.kind != self._last_blocked_kind:
                        out.append(('note',
                                    'blocked on client %s (%r) - not '
                                    'auto-ordered (candidate-starve '
                                    'ambiguity); resolve via the client'
                                    % (p.kind, p.kw)))
                elif p.kind != self._last_blocked_kind:
                    out.append(('note',
                                'blocked on client %s (%r) - not '
                                'auto-ordered (candidate-starve '
                                'ambiguity); resolve via the client'
                                % (p.kind, p.kw)))
                self._last_blocked_kind = p.kind
        else:
            self._last_blocked_kind = None
            self._human_draw_asked = None
        return out

    def _ingest_line(self, raw, sc):
        """Parse + sidecar-attach + segment + route ONE feed line (the
        per-line unit tick() isolates errors around)."""
        out = []
        rec = live_parser.parse(raw)
        if self._repair_consumes_echo(rec):
            return out               # our own self-repair kill, echoed
        self._bind_public_table_kill(rec)
        if sc is not None and sc.present:
            mrec = sc.rec(self._feed_seq)
            if mrec is not None:
                rec['meta'] = mrec
            else:
                self.findings.append(('sidecar_gap',
                                      'seq %d has no meta record'
                                      % self._feed_seq))
        # The same duplicate decision owns segmentation and raw identity
        # joining. A re-rendered move has no second native command to join.
        render_duplicate = self.seg.classify_render_duplicate(rec)
        if render_duplicate:
            return self._route('dropped_dup', rec)
        if 'meta' in rec:
            self._meta_track(rec)
        from .native_die_identity import observe_record as observe_die_record
        rec['_feed_seq'] = self._feed_seq
        observe_die_record(self, rec)
        from .human_draw_identity import observe_record as observe_draw_record
        observe_draw_record(self, rec)
        # Exact physical completion of an existing engine state is not an
        # effect of a still-unresolved multi-die selection. Both consumers
        # also see new native moves so their own Redo branches are retired.
        from .shadow_siege_slots import bind_record
        bind_record(self.view, self.es, rec)
        from .siege_seating import consume as consume_seating
        seated = consume_seating(self, rec)
        from .guide_seating import consume as consume_guide_seating
        guide_seated = consume_guide_seating(self, rec)
        if seated or guide_seated:
            if rec.get('op') not in ('undo', 'redo'):
                selection = getattr(self, '_grouped_die_selection', None)
                if selection is not None:
                    selection['redo'].clear()
                self.seg._clear_open_redo()
            return out
        from .staged_dice import consume as consume_staged_selection
        selection_items = consume_staged_selection(self, rec)
        if selection_items is not None:
            return selection_items
        from .fellowship_reveal_gestures import observe_record as observe_reveal_record
        observe_reveal_record(self, rec)
        from .temporary_combat_figures import consume_return
        if consume_return(self.view,self.es,rec):
            self.findings.append(('temporary_companion_returned',rec.get('norm','')))
            return out
        if self.echo_expect \
                and str(rec.get('player', '')) == self.nick:
            # BRLANE-13: the client's echo of our own line (same pids
            # under the AI nick) releases the expectation
            from . import sidecar as _sc13
            try:
                _pids13 = _sc13.meta_pids(rec.get('meta')) or []
            except Exception:
                _pids13 = []
            if _pids13:
                confirmed = {str(e.get('line', '')).strip()
                             for p, e in self.echo_expect.items() if p in _pids13
                             and all(q in _pids13 for q, other in self.echo_expect.items()
                                     if other.get('line') == e.get('line'))}
                self._release_echo(_pids13, 'echo')
                self._confirm_forced_ai_rows(confirmed)
        if rec.get('unknown'):
            self.unknown_lines.append(raw)
            self.alarms.append('UNKNOWN LINE: %r' % raw)
        if self.forced_ai_rows_pending() and rec.get('op') != 'chat':
            # Keep native corrections/ACKs observable, but no new roll, card,
            # Undo or action may cross an unrendered committed consequence.
            # Explicit Resync reads the complete corrected board below.
            from .casualty_gestures import buffer_early as buffer_blocked_casualties
            import ai as _blocked_casualty_ai
            if buffer_blocked_casualties(self, _blocked_casualty_ai.describe_pending(self.es), rec):
                return out
            self._consume_mirror_echo(rec)
            return out
        # Some players advance FSP before paying directly into the Hunt
        # Box (Space_Surgeon, 2023-01-04, first action).  Hold that partial
        # gesture only during an ordinary human FP action.  Pending hunt
        # resolutions and echoes of an already committed track move must
        # keep their existing phase routing.
        from .segmenter import fellowship_track_step
        step = fellowship_track_step(rec)
        allow_fellowship_prefix = False
        if step is not None and self.seg.window is None \
                and self.phase == 'actions' \
                and self.human_side.name == 'FREE_PEOPLES' \
                and not getattr(self, 'spectator', False) \
                and self.deferred is None \
                and step == (self.es.fellowship.progress,
                             self.es.fellowship.progress + 1):
            import ai
            allow_fellowship_prefix = ai.describe_pending(self.es) is None
        for ev, payload in self.seg.push(
                rec, allow_fellowship_prefix=allow_fellowship_prefix,
                render_duplicate=render_duplicate,
                open_redo_context=self._open_redo_context()):
            out.extend(self._route(ev, payload))
        out.extend(self._resume_uncommitted_card_selection())
        return out

    # ------------------------------------------------------------------
    def _open_redo_context(self):
        """Authorize only the same ordinary, unpaid human transaction."""
        window = self.seg.window or getattr(self.seg, '_redo_window', None)
        if (window is None or self.phase != 'actions'
                or getattr(self, 'spectator', False)
                or window.die_side != self.human_side.name
                or window.closed not in (None, 'undone')
                or window.awaiting_fellowship_die or window.fellowship_shortcut
                or self.deferred is not None or self.pending_buf is not None
                or getattr(self, 'open_step', None) is not None
                or getattr(self.rand, 'step_open', False)
                or window is getattr(self, '_staged_house_window', None)
                or window is getattr(self, '_staged_event_window', None)):
            return None
        import ai
        if ai.describe_pending(self.es) is not None:
            return None
        # A commit, engine replacement or applied-history Undo invalidates
        # saved physical evidence. Never reopen an already applied action.
        return (id(self.es), self.undo_epoch, self.n_commits_total,
                getattr(self.es, 'turn', None))

    def _staged_card_pending(self):
        if getattr(self, 'spectator', False):
            return False
        from .card_completion import needs_boundary
        if not needs_boundary(self.es):
            return False
        import ai
        pending = ai.describe_pending(self.es)
        return pending is not None and pending['side'] is self.human_side

    def _card_completion_wait(self):
        """An unfinished human transaction, not a verdict on its legality.

        The normal matcher validates the complete choice at its explicit
        boundary. A rejected completion also reopens this boundary after
        giving its correction, so the human can use Undo and assemble it again.
        """
        return (self.pending_buf is not None
                and not getattr(self.pending_buf, '_staged_card_closed', False)
                and self._staged_card_pending())

    @staticmethod
    def _copy_card_window(window):
        if window is None:
            return None
        import copy
        result = copy.copy(window)
        result.records = list(window.records)
        return result

    def _close_staged_card_buffer(self, close_record=None):
        if not self._staged_card_pending():
            return False
        if self.pending_buf is None:
            self.pending_buf = self._new_phase_window()
        if not getattr(self.pending_buf, '_staged_card_closed', False):
            self.pending_buf._staged_card_closed = True
            self.pending_buf._staged_card_close_record = close_record
            self.pending_buf._staged_card_event_window = self._copy_card_window(
                getattr(self, '_staged_event_window', None))
        self._try_pending()
        from .card_completion import closed_choice_correction
        correction = closed_choice_correction(
            self.es, self.pending_buf, getattr(self, '_last_pending_match', None),
            pending_rng=self.pending_random_call() is not None or self.deferred is not None)
        if correction:
            self._out_queue.append(('chat', 'refusal', correction))
        return True

    def _staged_card_close_event(self, ev, payload):
        if not self._staged_card_pending():
            return False
        if ev == 'closed' and getattr(payload, 'closed', None) == 'die_killed':
            if getattr(payload, 'die_side', None) != self.human_side.name:
                return False
            if self.pending_buf is None:
                self.pending_buf = self._new_phase_window()
            for rec in payload.records:
                if (rec.get('op') == 'move'
                        and gesture_match.zone_key(rec.get('from'))[0] == 'region'
                        and gesture_match.zone_key(rec.get('to'))[0] == 'region'
                        and not any(rec is prior for prior in self.pending_buf.records)):
                    self.pending_buf.records.append(rec)
            return self._close_staged_card_buffer(payload.records[-1] if payload.records else None)
        if (ev != 'phase' or payload.get('op') not in ('kill', 'move')
                or str(payload.get('player', '')) == self.nick):
            return False
        from .card_completion import closes_card
        if closes_card(self.es, payload):
            self._consume_mirror_echo(payload)
            return self._close_staged_card_buffer(payload)
        if payload.get('op') != 'kill':
            return False
        zone = str(payload.get('zone', '')).strip().lower()
        pieces = payload.get('pieces') or ()
        side_label = 'shadow action die' if self.human_side.name == 'SHADOW' else 'free action die'
        from .action_die_zones import temporary as temporary_die_zone
        if temporary_die_zone(zone) and any(side_label in str(p).lower() for p in pieces):
            self._close_staged_card_buffer(payload)
            self._consume_mirror_echo(payload)
            return True
        return False

    def _route(self, ev, payload):
        """Ported PHASE3-D routing shell around the event dispatch:
        phase board records defer behind a starving window / buffer
        into an open HUMAN-side pending; rng-bearing records retry
        whatever is blocked.  (The spectator's _route override layers
        its mirror + both-seat scoping on top and delegates here.)"""
        from .victor_completion import observe_close as observe_victor_close
        if observe_victor_close(self, ev, payload):
            return []
        if self._staged_card_close_event(ev, payload):
            if getattr(self, '_staged_event_window', None) is payload:
                self._staged_event_window = None
                self._promote_reconcile()
            return []
        if ev == 'open' and getattr(self, '_ring_gesture_records', None):
            # The player staged a die between the ring counter and its
            # face change. Match the complete ordered action window.
            payload._ring_prefix = list(self._ring_gesture_records)
            payload.records[:0] = payload._ring_prefix
            self._ring_gesture_records = None
        if ev == 'update' and not getattr(self, 'spectator', False):
            if getattr(self, '_staged_event_window', None) is payload:
                return self._route('phase', payload.records[-1])
            if getattr(self, '_staged_house_window', None) is payload:
                # The printed effect has committed. Discards and the
                # optional guide draw must resolve before the die is killed.
                return self._route('phase', payload.records[-1])
            staged = self._resolve_staged_house(payload)
            if staged is not None:
                return staged
            staged = self._resolve_staged_event(payload)
            if staged is not None:
                return staged
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
        if ev == 'update' and not getattr(self, 'spectator', False) \
                and getattr(payload, 'records', None):
            _rec4 = payload.records[-1]
            if isinstance(_rec4, dict) \
                    and _rec4.get('op') in ('move', 'kill') \
                    and self._consume_mirror_echo(_rec4):
                # FLA-4 (par5 g5#5): a record collected into an OPEN
                # window never reached the mirror-echo consume - the
                # echo stayed open all turn (census frozen 'owed:')
                # and the banked twin was re-performed (the duplicate
                # leader).  Echo coverage wins over window content
                # (BL-80: follow-through is never a fresh gesture).
                payload.records.pop()
                self.findings.append(
                    ('mirror_echo_in_window',
                     str(_rec4.get('norm', ''))[:80]))
                return []
        if ev == 'phase' \
                and payload.get('op') in ('move', 'kill') \
                and self._consume_mirror_echo(payload):
            # the client's mirror of an already-committed removal -
            # consumed, never a gesture, never pending-buffer content.
            # The AI's OWN forced-row line echoes here too (its nick):
            # the row banked at emission is claimed by that echo, so
            # echo_open carries only the human's rows
            return []
        if ev in ('phase', 'undo'):
            from .siege_withdrawal_gestures import consume_correction
            correction = consume_correction(self, payload)
            if correction is not None:
                return correction
        if ev == 'phase' and payload.get('op') == 'kill' \
                and not getattr(self, 'spectator', False) \
                and (getattr(self, '_discard_owed_pid', None)
                     or getattr(self, '_discard_recent62', None)):
            # BL-106: a DEMANDED discard hand-kill performed late is
            # the demand's ECHO - claim-once by pid, claimed BEFORE
            # pending-buffer routing ('demanded performances are never
            # pending answers', hand-zone kills included)
            from . import sidecar as _sc106
            import time as _t106
            try:
                _pids106 = [str(p) for p in
                            (_sc106.meta_pids(payload.get('meta'))
                             or [])]
            except Exception:
                _pids106 = []
            _dop106 = getattr(self, '_discard_owed_pid', None)
            if _dop106 and str(_dop106) in _pids106:
                self.findings.append(('discard_owed_claimed',
                                      str(_dop106)))
                self._discard_owed_pid = None
                return []
            # O2: the dump-confirm may release the owed pid BEFORE this
            # record routes - a hand-kill matching a RECENTLY released
            # discard pid is still that demand's echo
            _rec106 = getattr(self, '_discard_recent62', None) or {}
            for _p106 in _pids106:
                _ts106 = _rec106.get(_p106)
                if _ts106 and _t106.time() - _ts106 < 45.0:
                    _rec106.pop(_p106, None)
                    self.findings.append(
                        ('discard_owed_claimed_late', _p106))
                    return []
            # BL-106(b) instrument: a demanded kill passing the claim
            # UNCLAIMED says WHY
            self.findings.append(
                ('discard_claim_miss',
                 'owed=%s recent=%s rec_pids=%s'
                 % (_dop106, sorted(_rec106)[:3], _pids106[:3])))
        if ev == 'phase' and payload.get('op') == 'move' \
                and not getattr(self, 'spectator', False) \
                and str(payload.get('player', '')) != self.nick \
                and any(v == 'owed' for v in
                        (getattr(self, '_cc_reveal_owed', None)
                         or {}).values()):
            # FLA-12 (par5 g4#3): the demanded physical reveal of a
            # chat-committed combat card arrives as a PLAIN hand ->
            # card-slot drag (no '(chosen)') - claim it the moment it
            # lands, never buffer it behind the next unrelated
            # gesture, never re-nag a satisfied demand.
            _srcR = gesture_match.zone_key(payload.get('from'))
            _dstR = gesture_match.zone_key(payload.get('to'))
            if str(_srcR[1] if len(_srcR) > 1 else '').endswith(
                    'hand') and _dstR[0] == 'play':
                _itR = self._claim_cc_reveal(
                    str(payload.get('obj', '') or ''))
                if _itR is not None:
                    return _itR
        if ev == 'phase' and payload.get('op') == 'move' \
                and str(payload.get('player', '')) != self.nick \
                and not getattr(self, 'spectator', False) \
                and gesture_match.is_ring_track_move(
                    payload.get('obj'), payload.get('from'),
                    payload.get('to')):
            # a ring walked with NO die staged: the standalone ring
            # lane (commit against the elven_ring menu or refuse) -
            # never pending-buffer content, never a silent drop
            return self._on_ring_walk(payload)
        if ev == 'phase' and not getattr(self, 'spectator', False) \
                and str(payload.get('player', '')) != self.nick \
                and (payload.get('op') == 'die_change' or (
                    payload.get('op') == 'kill'
                    and len(payload.get('pieces') or ()) == 1
                    and str(payload['pieces'][0]).lower() in ('vilya', 'nenya', 'narya'))):
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
            elif (_src[0] == 'hand_chosen'
                    and not self._is_dice_bookkeeping(payload)
                    and not self._rec_inert_here(payload)):
                # Native reveal_chosen publishes both sides' cards under
                # the issuing AI's nick. The hand zone owns the reveal;
                # its existing handler distinguishes a physical human
                # commitment from an idempotent AI echo before nick-based
                # filtering can discard either acknowledgment.
                items = self._on_chosen_reveal(payload, _src[1])
                if items is not None:
                    return items
        from .siege_withdrawal_gestures import endpoint_record_needed as siege_endpoint_record
        if ev == 'phase' \
                and payload.get('op') in ('move', 'kill', 'flip') \
                and str(payload.get('player', '')) != self.nick \
                and not self._is_dice_bookkeeping(payload) \
                and (not self._rec_inert_here(payload) or siege_endpoint_record(self, payload)):
            # DICE bookkeeping records (recover drags, pool moves) are
            # never pending content: they reach the normal handlers (a
            # pending open at turn end must not eat the rollover).  Our
            # own emission echoes are never the human's pending
            # resolution (nick = SELF-echo detection only; the
            # attribution ruling forbids guessing the HUMAN's side from
            # a nick). Chosen-card reveals were handled above by hand
            # ownership. Spectator routing remains below.
            if self.deferred is not None:
                # a window is still starving: these records belong
                # AFTER its commit — queue, replay when it settles
                self.deferred_tail.append(payload)
                return []
            import ai
            guide_items = self._start_fellowship_guide_gesture(payload)
            if guide_items is not None:
                return guide_items
            # K2: a human FP's declaration records must find the
            # declare menu open (the AI's own turn-start actions may
            # have cleared the engine flag since it was armed)
            self._ensure_declare_armed()
            dp = ai.describe_pending(self.es)
            if (self.phase == 'guide' and self._declare_settled()
                    and self.es.fellowship.progress == 0
                    and payload.get('op') == 'move'
                    and str(payload.get('obj', '')).lower() == 'fsp'
                    and str(payload.get('to', '')).lower() == 'fellowship track 0'):
                # Native reveal already reset progress when declaration
                # committed. A second physical reset is the same result;
                # it neither nominates a Guide nor starts a die action.
                return []
            if (self.phase == 'guide' and self._declare_settled()
                    and dp is not None and dp['kind'] == 'guide_change'
                    and payload.get('op') == 'flip'
                    and str(payload.get('obj','')).lower() == 'the fellowship'
                    and self.es.fellowship.hidden):
                # Native declaration can end with a flip/unflip pair after
                # its pawn/progress gesture has already committed. Those
                # flips do not nominate a Guide and cannot enter its buffer.
                from board import sanitize
                import game_logic as gl
                if sanitize(payload.get('region','')) == sanitize(str(gl.fellowship_region(self.es))):
                    return self._route_event(ev,payload)
            from .casualty_gestures import buffer_early as buffer_early_casualties
            if buffer_early_casualties(self, dp, payload):
                return []
            if dp is not None and self._pending_buffer_applies(dp):
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf.records.append(payload)
                if not self._try_pending():
                    # item 16: the human resolving the AI's battle by
                    # hand is recognised AT ONCE, never a 25-tick stall
                    _it16 = self._battle_by_hand_check(dp, payload)
                    if _it16:
                        return _it16
                return []
            if dp is None \
                    and not getattr(self, 'spectator', False) \
                    and (isinstance(getattr(self.es, 'pending_combat',
                                            None), dict)
                         or self.human_cc_committed):
                # K1 REVEAL GAP: while a committed combat card is still
                # hidden the round parks with deciding_side None
                # (describe_pending is None for a full server
                # round-trip); a human board record here (a retreat
                # drag racing the reveal line) is HELD -
                # _poke_pending_buffer retries it the moment the reveal
                # opens the decision.  Never a bare drop.
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf.records.append(payload)
                self.findings.append(('battle_gap_buffered',
                                      str(payload.get('norm', ''))[:80]))
                return []
        out = self._route_event(ev, payload)
        # rng-bearing phase records may unblock the pending buffer or
        # a deferred window
        if ev == 'phase' and payload.get('op') in ('roll_num', 'draw',
                                                   'random_discard'):
            self._retry_blocked()
        return out

    def _resolve_staged_event(self, window):
        """Begin an interactive event when it leaves the hand, retaining its die.

        The action die is physical completion evidence, not the signal to
        begin a siege or another card's decisions. Immediate effects retain
        the ordinary complete-gesture matcher; any legal event that opens a
        pending decision uses the shared live pending lane.
        """
        import ai
        if (self.phase != 'actions' or self._human_out_of_turn()
                or window.die_side != self.human_side.name
                or ai.describe_pending(self.es) is not None
                or self.deferred is not None or self.chat_deferred is not None):
            return None
        window.combat_cards_open = False
        facts = gesture_match.window_facts(window)
        if len(facts.cards_played) != 1 or facts.unmapped or facts.ring_moves:
            return None
        cid = self.hands.reveal(facts.cards_played[0])
        if cid is None or self._needs_staged_die_identity(window):
            return None
        result = self._card_identity_probe(window, self.human_side,
                    gesture_match.MatchResult('none', note='staged event prefix'))
        if (result is None or result.status != 'match'
                or ai.describe_pending(result.post) is None):
            return None
        # A legal W-funded menu row is not evidence that a different staged
        # face paid for this card. Use the actual staged face, then bind its
        # native die identity through the existing payment machinery.
        action = result.chain[0]
        dice = self.es.dice.get(self.human_side, [])
        if (action.die_idx is None or not 0 <= action.die_idx < len(dice)
                or gesture_match.FACE_LETTER.get(getattr(
                    dice[action.die_idx].current, 'name', '')) != window.die_face):
            return None
        window.card_payment_staged = True
        self._rebind_chain_dice(result.chain, window)
        if not self._commit_chain(result.chain,
                [r.get('norm', '') for r in window.records],
                'staged-event', gesture=window):
            return []
        self._staged_event_window = window
        self.last_actor = self.human_side
        self.findings.append(('staged_event_started', cid))
        hint = getattr(self, '_card_target_hint', None)
        self._card_target_hint = None
        return self._apply_card_target_hint(hint)

    def _resolve_staged_house(self, window):
        """Commit House's complete printed effect while its die stays staged.

        Its two physical Strategy draws are the observable end of the
        printed effect. Later hand-limit and guide choices are separate
        decisions; waiting for the die kill would deadlock their input.
        """
        if self.phase != 'actions' or self._human_out_of_turn() \
                or self.human_side.name != 'FREE_PEOPLES' \
                or self.deferred is not None or self.chat_deferred is not None:
            return None
        records = getattr(window, 'records', ())
        if not records or records[-1].get('op') != 'draw' \
                or records[-1].get('what') != 'Free Strategy Card':
            return None
        if sum(r.get('op') == 'draw' and r.get('what') == 'Free Strategy Card'
               for r in records) != 2:
            return None
        import ai
        if ai.describe_pending(self.es) is not None:
            return None
        # Native CardBuffet play reveals from "Hand (chosen)". The open
        # prefix needs the same event/combat context as a closed window.
        window.combat_cards_open = self._combat_cards_open()
        facts = gesture_match.window_facts(window)
        if len(facts.cards_played) != 1 or gesture_match.title_to_cid(
                facts.cards_played[0]) != 'FREE_PEOPLES_Character_House_of_the_Stewards':
            return None
        # This is a complete printed-effect prefix, so it must match the
        # recorded recruitment and both draws. Card identity alone is not
        # enough to consume those physical effects.
        window._require_full_match = True
        window.card_payment_staged = True
        commits = self.n_commits_total
        out = self._on_window(window)
        if self.n_commits_total > commits:
            self._staged_house_window = window
        return out

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

    def _bind_public_table_kill(self, rec):
        """Bind public table-card disposal, whether killed or dragged away."""
        titles = None
        if rec.get('op') == 'kill' and gesture_match.zone_key(rec.get('zone'))[0] != 'hand':
            titles = rec.get('pieces') or ()
        elif (rec.get('op') == 'move'
              and gesture_match.zone_key(rec.get('from'))[0] not in ('hand', 'hand_chosen')
              and gesture_match.zone_key(rec.get('to'))[0] == 'dead'):
            import rendered_replay as rr
            obj = str(rec.get('obj', ''))
            titles = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
        if titles is not None:
            on_table = gesture_match.table_card_ids(self.es)
            ids = [gesture_match.title_to_cid(title)
                   for title in titles]
            # Kill's printed zone is the client's constant Current Action,
            # not the piece's origin. No hidden hand/dump inference is needed.
            rec['_table_discards'] = [cid for cid in ids if cid in on_table]

    def _meta_track(self, rec):
        """CLIENT-WAVE (c): pid-TRUE view bookkeeping from the
        sidecar's exact pids - piece positions stop depending on the
        fungible dump-intersection alone (the dump stays the WITNESS;
        this is tracking, never authority).  Redacted records update
        nothing (blinding fail-closed: meta_pids treats 'redacted' as
        absent - no bypass around the feed's blinding)."""
        from . import sidecar as sc_mod
        self._bind_public_table_kill(rec)
        pids = sc_mod.meta_pids(rec.get('meta'))
        # table-card seating is tracked even WITHOUT pids: a card
        # played straight from the hand carries `pids: redacted` (the
        # feed blinds a hidden-hand exit, correctly), but the card is
        # PUBLIC the moment it lands in a Board slot and the rendered
        # line NAMES it - the seat is booked by NAME and its pid
        # resolved from the client's own dump when needed (USER
        # 2026-08-20: "it should see where I placed the power of tom
        # bombadil from the rendered log, and delete it there").
        if rec.get('op') == 'move' and rec.get('to') and rec.get('obj'):
            self._track_table_seat(rec, pids)
        if not pids:
            return
        op = rec.get('op')
        if op == 'draw':
            # The client's random Companion draw physically parks its result
            # in a current-card area. Track that public result before emitting
            # its fate, just as for an ordinary move; do not remove it from
            # the Fellowship box a second time.
            import log_replay as lr
            import rendered_replay as rr
            tags = (rec.get('meta') or {}).get('tags') or ()
            piece = rr.parse_piece(str(rec.get('what', '')))
            if (len(pids) == 1 and str(pids[0]) in lr.COMPANIONS
                    and len(tags) == 5 and tags[:2] == ['Controls.637', 'Controls.638']
                    and tags[2] in ('Game.1998', 'Game.1999')
                    and tags[3] == 'Controls.639' and tags[4] in ('Game.2036', 'Game.2037')):
                names = tuple(lr.names_for_pid(str(pids[0])))
                if ((piece is not None and piece[0] == 'named' and piece[1] in names)
                        or str(rec.get('what', '')).strip().casefold() == 'gandalf' and str(pids[0]) == '240'):
                    self.view.piece_region[str(pids[0])] = lr.LOCATION_CODES[tags[4]]
                    rec['_native_companion_names'] = names
        elif op == 'move' and rec.get('to'):
            for p in pids:
                self.view.piece_region[str(p)] = str(rec['to'])
            # table-card seating is observed for EITHER player (USER
            # 2026-08-20: "you can see where the human player places
            # the table card, so just keep track of that location and
            # kill it there"); the mop-up kills the piece IN THE SLOT
            # we saw it go into
        elif op == 'kill':
            for p in pids:
                self._forget_table_seat_pid(p)
            # Keep Java's actual off-board destination, never pop: an
            # untracked pid falls back to its setup region. Shadow ordinary
            # pieces recycle and can be reused by a later casualty payment.
            import emitter
            for p in pids:
                self.view.piece_region[str(p)] = emitter._native_kill_zone(self.view, p)

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
            # A second chooser while a commitment already exists is not the
            # preamble that created it. A later old-card echo cannot withdraw
            # that existing choice merely because its deck happens to match.
            self._cc_choose_preamble = None
            w = getattr(self.seg, 'window', None)
            if (w is not None and w.die_side == self.human_side.name
                    and self.phase == 'actions'
                    and not getattr(self.es, 'pending_combat', None)):
                self._uncommitted_card_selection = {
                    'record': dict(rec), 'window': w,
                    'turn': getattr(self.es, 'turn', None)}
                return [('chat', 'prompt',
                         chat_mod.selected_card_intent_note())]
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
        old_cards = self._previous_cc_reveals(deck)
        items = self._attempt_chat_action(_commit, via='choose')
        self._cc_choose_preamble = (
            (self._cc_battle_serial, deck, tuple(old_cards))
            if old_cards and self.human_cc_committed else None)
        return items

    def _resume_uncommitted_card_selection(self):
        """Commit a held public chooser only after this die starts combat."""
        held = getattr(self, '_uncommitted_card_selection', None)
        if not held:
            return []
        w = held['window']
        current = getattr(self.seg, 'window', None)
        if (held['turn'] != getattr(self.es, 'turn', None)
                or (current is not None and current is not w)
                or (current is not w
                    and getattr(self, '_chat_attack_window', None) is not w)):
            self._uncommitted_card_selection = None
            return []
        facts = gesture_match.window_facts(w)
        if facts.cards_played or facts.combat_cards:
            self._uncommitted_card_selection = None
            return []
        if not self._human_cc_decision_open():
            return []
        self._uncommitted_card_selection = None
        return self._on_choose(held['record'])

    def _cancel_uncommitted_card_selection(self, rec):
        """A native deselection changes UI intent, never game resources."""
        held = getattr(self, '_uncommitted_card_selection', None)
        raw = str(rec.get('raw', ''))
        if not held or ' deselects a ' not in raw.lower():
            return []
        side, deck = self._deck_from_label(raw)
        expected_side, expected_deck = self._deck_from_label(
            held['record'].get('norm') or held['record'].get('raw'))
        # Native unchoose renders only "<chooser> deselects a card".
        # Its chooser identity links this UI cancellation to the held UI
        # selection; side ownership still came from the original deck label.
        chooser = held['record'].get('player')
        same_chooser = (side is None and deck is None and chooser
                        and raw.lstrip('$%').split(' deselects ', 1)[0] == chooser)
        if (side is expected_side and deck == expected_deck) or same_chooser:
            self._uncommitted_card_selection = None
            return [('note', 'Uncommitted card selection cancelled; action die remains available.')]
        return []

    def _previous_cc_reveals(self, deck=None):
        """Exact discarded identities whose physical reveal belongs to an
        earlier battle. Unknown '*' obligations cannot identify a later card.
        A newly held/reacquired card is not treated as a discarded old echo.
        """
        import game_logic as gl
        origins = getattr(self, '_cc_reveal_origins', {}) or {}
        current = getattr(self, '_cc_battle_serial', 0)
        return tuple(cid for cid, status in
                     (getattr(self, '_cc_reveal_owed', {}) or {}).items()
                     if cid != '*' and status in ('owed', 'claimed')
                     and origins.get(cid, current) != current
                     and self._cc_already_discarded(cid)
                     and (deck is None or gl.card_deck_of(cid) == deck))

    def _claim_previous_cc_preamble(self, cid):
        """Withdraw only the hidden choice inferred from this old reveal's
        native chooser preamble. Preserve the opponent's current commitment
        and any independently existing human commitment; never undo the real
        old card's physical reveal or replay its already resolved effect.
        """
        preamble = getattr(self, '_cc_choose_preamble', None)
        if not preamble or preamble[0] != self._cc_battle_serial \
                or cid not in preamble[2] or cid not in self._previous_cc_reveals():
            return []
        import ai
        pc = getattr(self.es, 'pending_combat', None) or {}
        role = 'atk' if pc.get('attacker_side') is self.human_side else 'def'
        if (pc.get(role + '_hidden_cc') or {}).get('deck') != preamble[1]:
            self._cc_choose_preamble = None
            return []
        human = self.human_side

        def cancel(state):
            ai.cancel_hidden_combat_card(state, human)
        cancel.action_type = 'combat_card_cancel'
        items = self._attempt_chat_action(cancel, via='cc_prior_reveal')
        if self.chat_deferred is not None and self.chat_deferred[0] is cancel:
            # Cancelling an unrevealed card must reopen its owner's choice,
            # not resolve a roll. Do not claim the old card or lose its retry
            # identity if that engine contract unexpectedly fails.
            self.chat_deferred = None
            self._chat_deferred_norms = ()
            raise DriverHalt('prior combat-card cancellation unexpectedly '
                             'requested randomness; the physical reveal '
                             'is still owed')
        self.cc_reveal_epoch += 1
        self.findings.append(('cc_prior_reveal', str(cid)))
        return items

    def _cc_already_discarded(self, cid):
        """Is `cid` out of the human hand AND in that side's face-up
        discard - a committed combat card the engine already played
        (ai._discard_played_combat_card)?  False for None."""
        if cid is None:
            return False
        import game_state as gs
        cid = str(cid)
        try:
            hand = list(self.es.hands[self.human_side].hand)
        except (KeyError, AttributeError):
            hand = []
        if cid in hand:
            return False
        pfx = 'fp' if self.human_side is gs.Side.FREE_PEOPLES else 'sh'
        dfu = getattr(self.es, 'discard_face_up', None) or {}
        return any(cid in (dfu.get('%s_%s' % (pfx, dk)) or ())
                   for dk in ('character', 'strategy'))

    def _claim_cc_reveal(self, title):
        """The BL-54/BL-81 claim-once body, shared by the '(chosen)'
        reveal route and the FLA-12 plain hand->card-slot drag route
        (par5 g4#3: the chat-committed card's physical reveal was
        never claimed - the nag re-fired 3-4x and the reveal only
        rode the next unrelated gesture).  Returns [] when the line
        claimed/duplicated an entry, None when no entry matches."""
        _led81 = getattr(self, '_cc_reveal_owed', None) or {}
        if not _led81:
            return None
        _rcid54 = gesture_match.title_to_cid(title)
        _k81 = None
        if _rcid54 is not None:
            if str(_rcid54) in _led81:
                _k81 = str(_rcid54)
            elif '*' in _led81:
                _k81 = '*'
        if _k81 is None:
            return None
        origins = getattr(self, '_cc_reveal_origins', {}) or {}
        current = getattr(self, '_cc_battle_serial', 0)
        prior = origins.get(_k81, current) != current
        if prior and (_k81 == '*' or _rcid54 not in self._previous_cc_reveals()):
            return None
        items = self._claim_previous_cc_preamble(_rcid54) if prior else []
        if _led81[_k81] == 'owed':
            _led81[_k81] = 'claimed'
            # the commit already discarded the played cid (face-up
            # pile): a hand swap here would trade a placeholder for
            # the PLAYED card and resurrect it in the engine hand.
            # Swap only while the identity is not yet in the discard
            # (the cid-unknown '*' commit is the one such case).
            if not self._cc_already_discarded(_rcid54):
                try:
                    self.hands.reveal(title)
                except Exception:
                    pass
            self.findings.append(('cc_reveal_physical',
                                  str(_rcid54)))
            self._poke_pending_buffer()
            return items
        self.findings.append(('cc_reveal_dup_echo', str(_rcid54)))
        return items

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
                    # the reveal may have opened the decision a
                    # battle-gap-held record answers (Scouts retreat)
                    self._poke_pending_buffer()
            return []
        if side is not self.human_side:
            return None
        # Revealing an uncommitted chooser is ordinary event-card play.
        # Do not later reuse its old chooser as a combat-card commitment.
        self._uncommitted_card_selection = None
        # BL-54: the CHAT-committed combat card's reveal - the commit
        # already APPLIED the card engine-side, so this line is the
        # PHYSICAL reveal only: swap the hand identity, let the table
        # mopup dispose the piece, never a second apply; a different
        # card than the committed one is a K1 integrity break.  BL-81:
        # a PER-BATTLE claim-once ledger - entries survive pending
        # advancement (the reveal is late-ordering tolerant), a claimed
        # cid's repeat consumes silently, the ledger resets at the NEXT
        # battle's start (post_commit_sweep).  FLA-12: the claim body
        # is shared with the plain hand->slot drag route.
        _out54 = self._claim_cc_reveal(title)
        if _out54 is not None:
            return _out54
        self._cc_choose_preamble = None
        _led81 = getattr(self, '_cc_reveal_owed', None) or {}
        # An older physical obligation is not a commitment in this round.
        # A different held current card must still reach the normal validator.
        origins = getattr(self, '_cc_reveal_origins', {}) or {}
        _led81 = {cid: status for cid, status in _led81.items()
                  if origins.get(cid, self._cc_battle_serial) == self._cc_battle_serial}
        if _led81:
            _rcid54 = gesture_match.title_to_cid(title)
            if _rcid54 is not None \
                    and any(v == 'owed' for v in _led81.values()):
                self.alarms.append(
                    'K1 INTEGRITY: revealed %r but the chat-committed '
                    'combat card is %r - reveal ignored'
                    % (str(_rcid54)[:40],
                       ','.join(k for k, v in _led81.items()
                                if v == 'owed')[:40]))
                return []
        if not self.human_cc_committed \
                and side not in self._cc_unrevealed():
            # the chooser EVENT play rides a die window (the play needs
            # its die) or an open pending that takes cards.  BL-27: a
            # BARE chooser play with neither is REFUSED + one
            # client-native press (the '<auto>' reveal is the client's
            # newest snapshot - the _withdraw_human_cc invariant);
            # never an orphaned card.
            w27 = getattr(self.seg, 'window', None)
            if w27 is not None and getattr(w27, 'die_side', None) \
                    == self.human_side.name:
                return None
            _dp27 = None
            try:
                if self.rand is not None:
                    with self.rand.pin():
                        _dp27 = ai.describe_pending(self.es)
                else:
                    _dp27 = ai.describe_pending(self.es)
            except Exception:
                _dp27 = None
            if _dp27 is not None:
                return None
            self.findings.append(('event_play_orphan_refused',
                                  title[:60]))
            from emitter import render as _r27
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'stage the action die first, then play the card - '
                'it goes back to your hand')),
                    ('line', _r27('undo', {'nick': self.nick},
                                  local=False))]
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
        _reveal.params = {'card_id': cid}
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
        self._poke_pending_buffer()
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
        if not self.human_cc_committed \
                and side not in self._cc_unrevealed():
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
            if payload.get('op') == 'noop':
                return self._cancel_uncommitted_card_selection(payload)
            return []
        if ev == 'phase':
            return self._on_phase(payload)
        if ev in ('open', 'update'):
            if ev == 'update' and payload.records:
                rec = payload.records[-1]
                if rec.get('op') == 'choose':
                    return self._on_choose(rec)
            return []
        if ev == 'closed':
            return self._on_window(payload)
        if ev == 'discarded':
            if getattr(payload, 'closed', None) == 'self_undo' \
                    and not getattr(self, 'spectator', False) \
                    and gesture_match.window_facts(payload).ring_moves:
                # Returning the staged die is not an undo of the paid
                # ring. Match the remaining ring/face facts with no die
                # consumption, retaining the client's real undo stack.
                return self._on_window(payload)
            # self-undo: LEGAL, SILENT (L1); the gesture is closed - a
            # parked reconcile arm is promoted, a chatted card target
            # dies with the window
            self._promote_reconcile()
            self._card_target_hint = None
            self._uncommitted_card_selection = None
            if getattr(self, '_staged_house_window', None) is payload:
                self._staged_house_window = None
            if getattr(self, '_staged_event_window', None) is payload:
                self._staged_event_window = None
            return []
        if ev == 'withdrawn':
            if (getattr(self, '_staged_house_window', None) is not None
                    or getattr(self, '_staged_event_window', None) is not None):
                # These records already resolved through the pending lane;
                # segmenter bookkeeping alone would leave the engine ahead.
                return self._on_undo('undo', {'inner': payload.get('norm', '')})
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
        return self._open_fellowship_guide('ai declare step closed')

    def _open_fellowship_guide(self, why):
        """Resolve declaration effects before entry and final Guide choice."""
        import game_logic as gl
        if self.phase == 'guide':
            return []
        if self.phase == 'fellowship_tail':
            return self._resume_fellowship_tail()
        self._close_declare(why)
        before = (self.es.fellowship.mordor_entered, self.es.fellowship.progress)
        gl.begin_fellowship_tail(self.es)
        return self._fellowship_tail_progress(before)

    def _resume_fellowship_tail(self):
        import game_logic as gl
        if self.phase != 'fellowship_tail':
            return []
        before = (self.es.fellowship.mordor_entered, self.es.fellowship.progress)
        gl.resume_fellowship_tail(self.es)
        return self._fellowship_tail_progress(before)

    def _fellowship_tail_progress(self, before):
        """Publish only transitions the core actually completed."""
        items = []
        if not before[0] and self.es.fellowship.mordor_entered:
            items.append(('mordor_entry', {'pre_progress': int(before[1] or 0)}))
        self.phase = ('fellowship_tail' if getattr(self.es, '_fellowship_tail_phase', False)
                      else 'guide')
        return items

    def _complete_fellowship_guide(self):
        """The selected (or uniquely forced) Guide is settled; eyes may begin."""
        import game_state as gs
        if self.phase != 'guide' or self.es.pending_guide_change is not None:
            return []
        self.phase = 'eyes'
        return [('ai_eyes', None)] if self.ai_side is gs.Side.SHADOW else []

    def _fellowship_close_blocked(self):
        import ai
        if getattr(self, '_turn_start_draws_open', False) or self.echo_pending():
            return True
        dp = ai.describe_pending(self.es)
        # With one eligible Guide the core opens eyes directly. The bridge
        # still waits for FP's phase-close signal; eyes are the NEXT phase,
        # not an unanswered Fellowship choice.
        return (bool(self.pending_buf and self.pending_buf.records)
                or (dp is not None and dp['kind'] != 'declare'
                    and not (dp['kind'] in ('guide_change', 'eye')
                             and self.phase == 'guide'
                             and getattr(self.es, '_eye_allocation_phase', False))))

    def _keep_fellowship_guide(self):
        """A human's phase-close signal keeps their current legal Guide."""
        import ai
        if self.es.pending_guide_change is None:
            return True
        current = self.es.fellowship.guide.name
        action = next((a for a in ai.get_legal_actions(self.es, self.human_side)
                       if a.action_type == 'choose_guide'
                       and a.params.get('companion') == current), None)
        if action is None:
            return False
        return self._commit_chain([action], [], 'fellowship-guide-keep')

    def _start_fellowship_guide_gesture(self, rec):
        """A human Guide-box arrival ends declaration before it is matched."""
        import ai
        import game_state as gs
        import game_logic as gl
        if (getattr(self, 'spectator', False) or self.phase != 'declare'
                or self.human_side is not gs.Side.FREE_PEOPLES
                or rec.get('op') != 'move'
                or _zone_key(rec.get('to')) not in _GUIDE_ZONES):
            return None
        if (self.es.pending_guide_change is not None
                and not getattr(self.es, '_eye_allocation_phase', False)):
            return None  # Existing compulsory replacements use their own menu.
        if self._fellowship_close_blocked():
            return self._refuse_phase_line(rec, chat_mod.refusal_illegal(
                'resolve the current Fellowship decision before appointing its Guide'),
                'GUIDE CHOICE')
        probe = ai.clone_state(self.es)
        gl.begin_fellowship_tail(probe)
        shim = self._new_phase_window()
        shim.records.append(rec)
        wanted = gesture_match.window_facts(shim).guide_set
        candidates = ((probe.pending_guide_change or {}).get('candidates')
                      or [probe.fellowship.guide.name])
        if getattr(probe, '_fellowship_tail_phase', False):
            # The intended phase close is accepted; the physical nomination
            # must wait for the declaration's card/Hunt effects.
            items = self._open_fellowship_guide('human Guide nomination')
            return items + self._refuse_phase_line(rec, chat_mod.refusal_illegal(
                'resolve the current Fellowship effect before appointing its Guide'),
                'GUIDE CHOICE')
        if wanted not in candidates:
            return self._refuse_phase_line(rec, chat_mod.refusal_illegal(
                'the Guide must be a highest-level Companion still in the Fellowship'),
                'GUIDE CHOICE')
        items = self._open_fellowship_guide('human Guide nomination')
        if self.es.pending_guide_change is not None:
            self.pending_buf = shim
            self._try_pending()
        else:
            items.extend(self._complete_fellowship_guide())
        return items

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
        if str(rec.get('nick', '')) == self.nick:
            return []               # our own chat echo
        control = (chat_mod.classify_command(rec.get('body', '')) or (None,))[0]
        if control == 'resume':
            return []  # server-owned rewind review control; no game decision
        receipts = getattr(self, '_raw_recovery_receipts', {})
        control_key = (control, str(rec.get('nick', '')))
        if receipts.get(control_key, 0):
            receipts[control_key] -= 1
            return []  # this exact operator command already escaped a held tick
        if control in ('reset', 'rewind') or (
                control == 'resync' and getattr(self, '_live_cycle_failure', False)):
            return [('recovery_request', control)]
        if (chat_mod.classify_command(rec.get('body', '')) or (None,))[0] == 'skip':
            # Explicit operator control precedes all game/gesture blockers.
            # The server executes it after this feed batch finishes routing.
            from .skip_recovery import capture
            return [('skip_request', capture(self))]
        if (self.forced_ai_rows_pending()
                and (chat_mod.classify_command(rec.get('body', '')) or (None,))[0]
                not in ('resync', 'adopt', 'yes', 'no', 'status', 'help')):
            return []  # the retained correction prompt already names the blocker
        if chat_mod.normalize_body(rec.get('body', '')).rstrip('.!') in ('done', 'finished'):
            from .guide_completion import request as request_guide_completion
            if request_guide_completion(self):
                return []  # Commit from the complete board at the feed boundary.
            if (self.phase == 'guide' and self.human_side.name == 'FREE_PEOPLES'
                    and self.es.pending_guide_change is None):
                return self._on_chat(dict(rec, body='fp ok'))
            from .casualty_gestures import complete_live_allocation
            completed = complete_live_allocation(self)
            if completed is not None:
                return completed
            if self._close_staged_card_buffer():
                return []
        cmd = chat_mod.classify_command(rec.get('body', ''))
        kind = cmd[0] if cmd else None
        if getattr(self, '_ring_gesture_records', None) \
                and kind not in ('resync', 'adopt', 'yes', 'no', 'status', 'help'):
            return [('chat', 'prompt', chat_mod.elven_ring_finish_note())]
        if self._repair is not None \
                and self._repair.get('phase') == 'offered' \
                and kind in ('yes', 'no'):
            if kind == 'yes':
                return self._repair_begin()
            _was_census = self._repair.get('kind') == 'census'
            self._repair = None
            # BRLANE-13: 'no' to an unacknowledged-emission offer drops
            # the expired expectations (loudly) so play can resume
            self._clear_echo_expect('declined', expired_only=True)
            if _was_census:
                # SQF-11 (seqack_fp1): the census decline keeps a
                # STANDING chat route (self-repair protocol: the ask
                # is one-shot, the door is not) - 'adopt' consents
                # later, 'resync' resets client-as-is
                self.findings.append(('census_declined_chat', ''))
                return [('chat', 'prompt',
                         chat_mod.census_adopt_declined_reply())]
            self.findings.append(('stall_repair_declined', ''))
            return [('chat', 'prompt', chat_mod.stall_repair_declined())]
        if kind == 'adopt' and not getattr(self, 'spectator', False):
            # SQF-11: the CLI seat's standing route to the adoption
            # the panel offers - consent at any time, one transaction
            r9a = self._repair
            if r9a is not None and r9a.get('kind') == 'census' \
                    and r9a.get('phase') == 'offered':
                return self._repair_begin()
            if r9a is not None:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'a self-repair is already %s - answer yes or no '
                    'to it first' % r9a.get('phase')))]
            _lq9a = getattr(self, '_last_quiescent_census', None) or {}
            _rows9a = [str(r) for r in (_lq9a.get('board') or ())]
            if not _rows9a:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'my board and your client agree - there is '
                    'nothing to adopt'))]
            self._repair = {'phase': 'offered', 'kind': 'census',
                            'reason': 'human adopt', 'log': None,
                            'kills': [], 'to_act': None,
                            'last_killed': None, 'kill_ticks': 0,
                            'rows': _rows9a, 'clears': 'board'}
            return self._repair_begin()
        if kind == 'pass' and self.phase in ('declare', 'guide'):
            import game_state as _gs24
            if self.human_side is _gs24.Side.FREE_PEOPLES:
                # BL-24 (2026-08-25): at the declare window the menu's
                # only row is 'pass', which never closed it - the
                # protocol then demanded the literal 'fp ok'.  A
                # human-FP 'pass' at declare IS the decline: alias it.
                kind = 'fp_ok'
        if kind == 'fp_ok' and self.phase in ('declare', 'guide'):
            if getattr(self, '_turn_start_draws_open', False):
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'finish the turn-start card draws and discards before saying fp ok'))]
            import game_logic as _draw_limits
            _draw_limits.open_hand_limit_discard(self.es)
        if kind == 'fp_ok' and self.phase in ('declare', 'guide') \
                and getattr(self.es, 'pending_discard',
                            None) is not None:
            # BL-106: the declaration window NEVER closes over an open
            # hand-limit discard - the learned decision is UNEXPIRABLE
            # and UNFORCEABLE (USER 2026-08-15: no path may auto-trim)
            self.findings.append(('fp_ok_held_for_discard',
                                  str(self.es.pending_discard)))
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                ('my hand-limit discard comes first - wait for me '
                 'to discard down to 6, then say fp ok')
                if self.es.pending_discard.get('side') is self.ai_side else
                ('your hand-limit discard comes first - discard down '
                 'to 6, then say fp ok')))]
        if kind == 'fp_ok' and self.phase in ('declare', 'guide'):
            # 'fp ok' closes the declaration opportunity: a human FP
            # who declared has already committed through the declare
            # menu; one who did not has declined (absence-is-the-
            # choice) - settled BEFORE the entry check reads the
            # position (K2)
            import game_state as gs
            if self.human_side is not gs.Side.FREE_PEOPLES:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'the Free Peoples must finish their Fellowship phase first'))]
            items = []
            if self.view.dump_path:
                # At the explicit completion boundary, a stable native board
                # can prove the legal declaration even if intermediate
                # flips/resets were redundant or arrived in another order.
                recovered, board_matches = self._recover_fellowship_choice(for_completion=True)
                items.extend(recovered)
                self._fellowship_completion_requested = board_matches is None
                if board_matches is None:
                    return items
                if not board_matches:
                    return items + [('chat', 'refusal', chat_mod.refusal_illegal(
                        'finish the declaration on the board: place the hidden Fellowship '
                        'at a legal destination and reset its progress, then say fp ok; '
                        'or say skip to abandon this interaction and use the board as it stands'))]
            if self._fellowship_close_blocked():
                return items + [('chat', 'refusal', chat_mod.refusal_illegal(
                    'resolve the current Fellowship decision before saying fp ok'))]
            # MORDOR ENTRY check (FINAL USER RULING 2026-08-16): 'fp
            # ok' closes the declaration opportunity, so the check
            # runs HERE, same turn start (ONE helper, shared with the
            # AI-as-FP declare path, R2)
            items.extend(self._open_fellowship_guide("'fp ok'"))
            if self.phase == 'fellowship_tail':
                return items + self._maybe_announce_pending()
            if not self._keep_fellowship_guide():
                return items + [('chat', 'refusal', chat_mod.refusal_illegal(
                    'choose a legal Guide before ending the Fellowship phase'))]
            items.extend(self._complete_fellowship_guide())
            return items
        if kind in ('eyes', 'fp_ok') and self.phase == 'fellowship_tail':
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'resolve the current Fellowship effect before ending the phase'))]
        if kind == 'eyes' and self.phase == 'guide':
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'the Free Peoples Guide choice comes before Hunt allocation'))]
        if kind == 'fp_ok' and self.phase == 'eyes':
            return []  # The preceding explicit Guide gesture already closed it.
        eye_done = chat_mod.normalize_body(rec.get('body', '')).rstrip('.!') in (
            'done', 'finished', 'roll', 'eyes done')
        if (self.human_side.name == 'SHADOW' and self.phase in ('declare', 'eyes')
                and (kind == 'eyes' or eye_done)):
            if self.phase == 'declare':
                if not self._eye_closes_declare():
                    return [('chat', 'refusal', chat_mod.refusal_illegal(
                        'the Free Peoples must finish their Fellowship phase first'))]
                self._close_declare('Hunt allocation completion')
                self.phase = 'eyes'
            # The number describes the manually placed total; it never moves
            # dice. A later count replaces it. "done" accepts the board even
            # if a previous count differs. Resolve after the entire feed batch.
            if kind == 'eyes' and int(cmd[1]) == 0:
                counts = self._eye_menu_counts()
                if counts and min(counts) > 0:
                    self.findings.append(('eye_chat_refused', '0e: minimum %d' % min(counts)))
                    return [('chat', 'refusal', chat_mod.refusal_illegal(
                        'at least %d eye must be allocated this turn; '
                        'place it on the board and say 1e or done' % min(counts)))]
            self._eye_completion_requested = (
                int(self.es.turn), None if eye_done else int(cmd[1]))
            return []
        if kind == 'resync':
            # witness FIRST (evidence), then - the USER's adopt-and-
            # keep-playing lever (C-12) - recovery when the report
            # found drift or expectation machinery is stuck
            self._clear_echo_expect('resync')        # BRLANE-13
            items = self._witness_report()
            drift = any('desync' in str(i[1]) for i in items
                        if i[0] == 'note')
            stuck = (self.deferred is not None
                     or self.chat_deferred is not None
                     or self.pending_buf is not None
                     or bool(self.uncommitted))
            if drift or stuck:
                items += self.resync_recover()
            # FLD-6e (par5b_g5#2): the chat verb completed SILENTLY -
            # ack in chat what actually happened (recovery run, or
            # nothing stuck)
            items.append(('chat', 'prompt',
                          chat_mod.resync_ack(bool(drift or stuck))))
            return items
        if kind == 'concede' and not getattr(self, 'spectator',
                                             False):
            # A-10: 'gg'/'concede'/'resign' ends the game - the
            # conceding HUMAN loses.  (Spectator: corpus 'gg' chats are
            # the game's own record, not a live concession -
            # byte-pinned census.)
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
        if dp is None:
            from .attack_chat import declaration, remember, poll
            attack_body = declaration(self, rec.get('body', ''))
            if attack_body is not None:
                before = self.n_commits_total
                output = self._on_chat(dict(rec, body=attack_body))
                if self.n_commits_total > before and self.es.pending_combat:
                    remember(self)
                    output.extend(poll(self))
                return output
        if ((dp is None or dp['kind'] == 'eye') and self.phase == 'guide'
                and self.human_side.name == 'FREE_PEOPLES'
                and not getattr(self, 'spectator', False)
                and self.es.pending_guide_change is None):
            # A sole eligible Guide is already selected by the rules. Human
            # confirmation of that same Guide is valid narration, not a new
            # decision or the signal to close the Fellowship phase.
            guide = getattr(self.es.fellowship, 'guide', None)
            name = getattr(guide, 'name', '')
            if name and chat_mod.normalize_body(rec.get('body', '')) == (
                    'guide ' + chat_mod.normalize_body(name)):
                return [('chat', 'prompt',
                         '%s remains the Guide; say fp ok when finished.' % name)]
        if (getattr(self, '_staged_event_window', None) is not None
                and kind in ('region', 'attack')):
            target = cmd[1][0] if isinstance(cmd[1], tuple) else cmd[1]
            resolved, _ = chat_mod.resolve_region(target, self._region_keys())
            combat = getattr(self.es, 'pending_combat', None) or {}
            if resolved and resolved == combat.get('defender_region'):
                # Repeating the event's already selected target does not
                # declare a second attack or answer the next combat choice.
                return [('note', 'event battle already targets %s' % resolved)]
        if cmd is not None and cmd[0] == 'help':
            # USER 2026-08-24: 'help' lists every chat the AI
            # understands; with an open decision, lead with what the
            # AI is waiting on
            out9 = []
            if dp is not None and dp['side'] is self.human_side:
                out9.append(('chat', 'prompt', chat_mod.waiting_on(
                    str(dp['kind']),
                    self._pending_words(dp))))
            out9.extend(('chat', 'prompt', l)
                        for l in chat_mod.HELP_LINES)
            return out9
        if dp is not None and dp['side'] is self.human_side:
            # BL-96: 'pick N' commits the ROW THE HUMAN READ - resolved
            # through the publication's stored identity map, never a
            # fresh enumeration's index; a stale identity refuses
            # LOUDLY; the commit VOICES what it did
            _pm96 = getattr(self, '_pick_pub96', None)
            if (cmd is not None and cmd[0] == 'magnitude' and _pm96
                    and int(cmd[1]) in _pm96
                    and not any(
                        a.action_type == 'combat_effect_choice'
                        and int((a.params or {}).get('count', -1)
                                or 0) == int(cmd[1])
                        for a in dp['actions'])):
                # FLD-10b (par5b_g5#1): the bare number IS the
                # advertised say_pick - it resolves through the SAME
                # published identity map as 'pick N' (the dp-order
                # index lane numbered rows differently from the
                # published menu and committed the wrong row with no
                # voice).  FLA-15 magnitude precedence kept above.
                cmd = ('pick', int(cmd[1]))
            if cmd is not None and cmd[0] == 'pick' and _pm96:
                _key96 = _pm96.get(int(cmd[1]))
                if _key96 is not None:
                    _hit96 = next(
                        (a for a in dp['actions']
                         if chat_mod.action_identity(a) == _key96),
                        None)
                    if _hit96 is None:
                        self.findings.append(
                            ('pick_identity_stale',
                             'pick %s: published row no longer in '
                             'the live menu' % cmd[1]))
                        return [('chat', 'refusal',
                                 chat_mod.refusal_illegal(
                                     'the menu changed since that '
                                     'pick was published - re-read '
                                     'the menu and pick again'))]
                    out96 = self._commit_human(_hit96, via='chat')
                    if self.chat_deferred is None and _hit96.action_type != 'siege_press':
                        # voice ONLY a landed commit (a deferred
                        # apply announces its own starvation); ONE
                        # description builder with the physical lane
                        # (SQF-5)
                        out96.append(('chat', 'prompt',
                                      chat_mod.pick_committed_note(
                                          self._pick_desc(_hit96))))
                    return out96
            _battle_chat = getattr(self.es, 'pending_combat', None) or {}
            if (_battle_chat.get('phase') in ('attacker_combat_card', 'defender_combat_card')
                    and chat_mod.matches_battle_target(rec.get('body', ''),
                                                      _battle_chat, self._region_keys())):
                # Repeating the same declared target while choosing a combat
                # card is narration of this battle, not a new paid action.
                return [('note', 'attack already declared against %s'
                         % _battle_chat['defender_region'])]
            act = chat_mod.match_choice(rec.get('body', ''),
                                        dp['actions'],
                                        combat=getattr(self.es, 'pending_combat', None),
                                        region_keys=self._region_keys())
            if act is not None:
                _via = ('chat-press' if act.action_type == 'siege_press'
                        and chat_mod.normalize_body(rec.get('body', '')) == 'press'
                        else 'chat')
                _outmc = self._commit_human(act, via=_via)
                if cmd is not None \
                        and cmd[0] in ('magnitude', 'pick') \
                        and self.chat_deferred is None and act.action_type != 'siege_press':
                    # FLD-10b: a number-resolved commit VOICES what
                    # it did (g5#1's silent lane) - ONE description
                    # builder with the pick lane (SQF-5)
                    _outmc.append(('chat', 'prompt',
                                   chat_mod.pick_committed_note(
                                       self._pick_desc(act))))
                return _outmc
            # 'discarding card' with several possible targets: name it
            _b9 = chat_mod.normalize_body(rec.get('body', ''))
            if _b9.startswith('discard'):
                _nds = [a for a in dp['actions']
                        if a.action_type == 'card_option'
                        and isinstance((a.params or {}).get('choices'),
                                       dict)
                        and (a.params or {}).get('choices').get(
                            'action') == 'discard_table_card']
                if len(_nds) > 1:
                    return [('chat', 'prompt',
                             chat_mod.discard_target_ambiguous())]
            # BL-23: an unmatched chat ANSWER while the human's
            # decision is open is refused LOUDLY, naming the input and
            # the words that work here (the no-silent-ignore invariant)
            _in23 = (_b9 or str(rec.get('body', '')))[:40]
            self.findings.append(('chat_answer_unmatched',
                                  '%s :: %s' % (dp['kind'], _in23)))
            from .decision_prompts import decision_label
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'that reply does not identify your %s; %s'
                % (decision_label(dp['kind']), self._pending_words(dp))))]
        # CARD-PLAY WINDOW GUARD (tester 2026-08-24): with an
        # un-terminated window holding the human's staged die AND a
        # card already dragged from hand, a region / attack chat is
        # NARRATION of the card's own target, never an attack
        # declaration - held with a note; the play resolves when the
        # die is killed (a single-siege card needs no chat; a
        # multi-choice pending accepts the region by name via
        # chat.match_choice)
        if dp is None and kind in ('attack', 'attack_from', 'region'):
            w9 = getattr(self.seg, 'window', None)
            if w9 is not None and getattr(w9, 'die_side', None) \
                    == self.human_side.name:
                try:
                    _wf9 = gesture_match.window_facts(w9)
                except Exception:
                    _wf9 = None
                if _wf9 is not None and _wf9.cards_played:
                    self.findings.append(
                        ('card_play_region_chat_held',
                         str(cmd[1])[:40]))
                    # BRLANE-12: the chatted region is the CARD's
                    # target - remembered, answered when the play
                    # opens its decision after the die kill (never
                    # matched against the plain-die attack menu)
                    _tgt12 = (cmd[1][0] if isinstance(cmd[1], tuple)
                              else cmd[1])
                    if self._is_region_name(str(_tgt12)):
                        self._card_target_hint = str(_tgt12)
                    # BRLANE-4: answered in CHAT, not a panel note only
                    return [('note', 'region chat held: a card-play '
                                     'window is open - kill the die '
                                     'to complete the play'),
                            ('chat', 'prompt',
                             chat_mod.card_window_region_held())]
        if kind == 'attack' and dp is None:
            return self._human_chat_attack(cmd[1])
        if kind == 'attack_from' and dp is None:
            ask = getattr(self, '_attack_ask', None)
            if ask is None:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'name the region to attack first'))]
            return self._human_chat_attack((ask, cmd[1]))
        if kind == 'region' and dp is None \
                and getattr(self, '_attack_ask', None) is not None:
            # F4: a bare region while my own from? ask is outstanding
            # IS the from-answer (the literal 'from x' form is not
            # required)
            ask4 = self._attack_ask
            return self._human_chat_attack((ask4, cmd[1]))
        if kind == 'region' and dp is None \
                and self._is_region_name(cmd[1]):
            return self._human_chat_attack((cmd[1], None))
        if kind == 'pass' and dp is None:
            # plain-action 'pass' with no pending open, during the
            # human's alternation turn: matched against the ENGINE
            # menu's own pass action (amendment 1.223 pass legality
            # lives IN the menu, never a driver counter); refused
            # otherwise
            return self._human_plain_pass()
        if kind in ('yes', 'no') \
                and not getattr(self, 'spectator', False) \
                and ((getattr(self, '_last_quiescent_census', None)
                      or {}).get('board')
                     or getattr(self, '_census_declined_sig', None)):
            if kind == 'yes':
                # FLD-6d (par5b_g2#28): a bare 'yes' while the
                # divergence stands IS the adoption consent - the ask
                # is one-shot and had already been consumed, and the
                # routes reminder read as a decline.  Same ONE lane
                # as the chat 'adopt' verb.
                _rows6d = [str(r) for r in
                           ((getattr(self, '_last_quiescent_census',
                                     None) or {}).get('board') or ())]
                if _rows6d and self._repair is None:
                    self._repair = {
                        'phase': 'offered', 'kind': 'census',
                        'reason': "human 'yes' (standing divergence)",
                        'log': None, 'kills': [], 'to_act': None,
                        'last_killed': None, 'kill_ticks': 0,
                        'rows': _rows6d, 'clears': 'board'}
                    return self._repair_begin()
            # SQF-11: no ask is open but a divergence stands - name
            # the standing routes, never a dead-end 'not a command'
            self.findings.append(('census_route_reminder', kind))
            return [('chat', 'prompt',
                     chat_mod.census_adopt_declined_reply())]
        # BRLANE-4 / M7 (USER 08-24: seven chats answered with nothing):
        # no lane owns this text - say so, naming what IS accepted now
        _body4 = str(rec.get('body', '') or '').strip()
        if not _body4:
            return []
        self.findings.append(('chat_unrecognised', _body4[:40]))
        return [('chat', 'refusal',
                 chat_mod.chat_not_a_command(self._chat_context_hint()))]

    @staticmethod
    def _pick_desc(action):
        """The pick-commit voice's description of one committed row -
        ONE builder for the chat pick lane and the physical pending
        lane (SQF-5: every pick commit is voiced)."""
        from .decision_prompts import action_description
        return action_description(action)

    def _pending_words(self, dp):
        """Physical instructions and accepted game words for the open choice."""
        from .decision_prompts import pending_instructions
        return pending_instructions(dp,self.es)

    def _temporary_combat_figure_note(self):
        from .temporary_combat_figures import waiting,return_zone
        owed=waiting(self.view,self.es)
        if not owed:
            self._temporary_figure_prompt=None
            return []
        key=tuple(sorted((row['pid'],row['turn'],row['rounds_fought'])for row in owed))
        if getattr(self,'_temporary_figure_prompt',None)==key:
            return []
        self._temporary_figure_prompt=key
        return [('chat','prompt',chat_mod._register(
            'The combat round is resolved: return %s from %s to %s before continuing.'%(
                row['name'],row['destination_zone'],return_zone(self.es,row))))for row in owed]

    def _auto_forced_siege_step(self):
        """Finish a compulsory siege step after its physical work is visible.

        Declaring an assault already supplies the human's intent. Likewise,
        no attacking Elite means there is no choice to continue the siege.
        Optional payments, casualty choices and other decisions remain open.
        """
        if self.forced_ai_rows_pending():
            return []
        pc = getattr(self.es, 'pending_combat', None) or {}
        expected = {'attacker_siege_decision': 'siege_assault',
                    'combat_round_result': 'siege_no_press'}.get(pc.get('phase'))
        if expected is None:
            return []
        if (getattr(self, 'spectator', False) or self.phase != 'actions'
                or self.deferred is not None or self.chat_deferred is not None
                or getattr(self, '_repair', None) is not None
                or (expected == 'siege_no_press' and self.echo_pending())
                or self.human_physical_work_pending()
                or (self.pending_buf is not None and self.pending_buf.records)):
            return []
        import ai
        if self.rand is not None:
            with self.rand.pin():
                dp = ai.describe_pending(self.es)
        else:
            dp = ai.describe_pending(self.es)
        if dp is None or dp['side'] is not self.human_side:
            return []
        actions = list(dp['actions'])
        if len(actions) != 1 or actions[0].action_type != expected:
            return []
        items = self._commit_human(actions[0], via='forced-siege')
        self.findings.append(('forced_siege_step', expected))
        return items

    def _retry_human_pending_prompt(self):
        """Announce a supported human choice after its physical barrier clears.

        A resumed combat can open another casualty allocation while an AI
        casualty is still rendering. The initial announcement is withheld;
        retry it on idle ticks without repeating partially paid choices.
        """
        if ((self.pending_buf is not None and self.pending_buf.records)
                or self.human_physical_work_pending() or self.echo_pending()):
            return []
        return self._maybe_announce_pending()

    def _pending_prompt(self, text):
        # Several decisions can finish in one feed batch. A queued question
        # belongs to this commit boundary, not the state reached afterward.
        return [('chat', 'prompt', text,
                 {'pending_at_commit': self.n_commits_total})]

    def _maybe_announce_pending(self):
        """One-shot chat announce for the next HUMAN-side card or combat
        effect choice, after preceding physical work completes (USER
        2026-08-24: 'the AI should say what its waiting on in the
        chat ... and give the options')."""
        if self.forced_ai_rows_pending():
            return []
        import ai
        try:
            dp = ai.describe_pending(self.es, self.human_side)
        except Exception:
            return []
        if dp is not None and dp['side'] is self.human_side:
            if dp['kind'] == 'guide_change':
                if (self.human_physical_work_pending() or self.echo_pending()
                        or (self.pending_buf is not None and self.pending_buf.records)):
                    return []
                from .guide_completion import prompt as guide_prompt
                instruction = guide_prompt(self)
                if instruction is None:
                    return []
                key = ('guide', self.n_commits_total,
                       tuple(chat_mod.action_identity(a) for a in dp['actions']))
                if getattr(self, '_announced_pending', None) == key:
                    return []
                self._announced_pending = key
                return self._pending_prompt(chat_mod.waiting_on(
                    dp['kind'], instruction))
            if self._siege_press_waiting(dp):
                return self._siege_press_reminder()
            import emitter as _cas_emitter
            context = _cas_emitter.casualty_context(self.es)
            if context is not None and context['side'] is self.human_side:
                # A pending choice and its later physical follow-through are
                # distinct: don't announce another allocation over owed pieces.
                if self.human_physical_work_pending() or self.echo_pending():
                    return []
                choices = []
                identities = {}
                verbs = {'remove_regular': 'remove one %s Regular (1 hit)',
                         'reduce_elite': 'reduce one %s Elite (1 hit)',
                         'remove_elite': 'remove one %s Elite entirely (2 hits)'}
                for i, action in enumerate(dp['actions'], 1):
                    params = _cas_emitter.casualty_action_params(action)
                    if params is None or params.get('operation') not in verbs:
                        return []
                    nation = params.get('nation')
                    label = verbs[params['operation']] % getattr(nation, 'value', str(nation))
                    choices.append((i, label))
                    identities[i] = chat_mod.action_identity(action)
                key = ('casualty', str(context['region']), context['hits'],
                       self.n_commits_total, tuple(identities.items()))
                if getattr(self, '_announced_pending', None) == key:
                    return []
                self._announced_pending = key
                self._pick_pub96 = identities
                return self._pending_prompt(chat_mod.casualty_choice_note(
                    str(context['region']), context['hits'], choices))
        if (dp is not None and dp['side'] is self.human_side
                and dp['kind'] == 'combat:combat_effect_choice'):
            from .combat_payment_prompt import payment_context, instructions
            payment = payment_context(dp, self.es)
            if payment is not None:
                if (self.human_physical_work_pending() or self.echo_pending()
                        or (self.pending_buf is not None and self.pending_buf.records)):
                    return []
                key = ('combat-payment', payment['effect'], payment['owner'],
                       payment['stage'], payment['region'], payment['hits'],
                       getattr(self.es, 'turn', None), self._cc_battle_serial,
                       (self.es.pending_combat or {}).get('rounds_fought'),
                       self.n_commits_total,
                       tuple(chat_mod.action_identity(a) for a in dp['actions']))
                if getattr(self, '_announced_pending', None) == key:
                    return []
                self._announced_pending = key
                return self._pending_prompt(chat_mod.waiting_on(
                    dp['kind'], instructions(payment)))
            effect = ((getattr(self.es, 'pending_combat', None) or {})
                      .get('effect_choice') or {})
            if effect.get('effect') in (
                    'mighty_attack', 'dread_despair', 'dread_and_despair',
                    'words_of_power', 'black_breath', 'heroic_death',
                    'relentless_assault', 'onslaught',
                    'fateful_strike', 'blade_of_westernesse'):
                if self.human_physical_work_pending() or self.echo_pending():
                    return []
                key = ('combat-effect', effect.get('effect'),
                       effect.get('stage'), self.n_commits_total,
                       tuple(chat_mod.action_identity(a) for a in dp['actions']))
                if getattr(self, '_announced_pending', None) == key:
                    return []
                self._announced_pending = key
                return self._pending_prompt(chat_mod.waiting_on(
                    dp['kind'], self._pending_words(dp)))
        if dp is None or dp['side'] is not self.human_side \
                or str(dp['kind']) != 'card_option':
            return []
        pco = getattr(self.es, 'pending_card_option', None) or {}
        key = (str(dp['kind']), str(pco.get('card_name')))
        if pco.get('card_id') == '__gandalf_draw__':
            key += (pco.get('source_card_id'), self.n_commits_total)
        if getattr(self, '_announced_pending', None) == key:
            return []
        self._announced_pending = key
        chs = [(a.params or {}).get('choices') or {}
               for a in dp['actions']]
        if pco.get('card_id') == '__gandalf_draw__':
            txt = chat_mod.gandalf_draw_choice_note(pco.get('deck', 'character'))
        elif any(isinstance(c, dict) and c.get('action') == 'hunt'
               for c in chs):
            txt = chat_mod.nazgul_strike_choice_note()
        else:
            txt = chat_mod.waiting_on(dp['kind'],self._pending_words(dp))
        return self._pending_prompt(txt)

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

    def _turn_over_pending(self):
        """BL-3 (2026-08-25): actions phase, nothing pending, and NO
        side can act (both pools spent). The server can request native
        dice recovery after all committed physical work is complete.
        The same native acknowledgment also accepts manual recovery."""
        import ai
        if self.phase != 'actions' or self.last_actor is None:
            return False
        if (getattr(self, '_chat_attack_window', None) is not None
                or getattr(self.seg, 'window', None) is not None
                or self.echo_pending() or self.human_physical_work_pending()):
            return False
        try:
            return (ai.describe_pending(self.es) is None
                    and all(d.used or getattr(d.current, 'name', None) == 'EYE'
                            for dice in self.es.dice.values() for d in dice))
        except Exception:
            return False

    _TURN_OVER_DETAIL = ('turn over - use the Recover Action Dice '
                         'menu, then draw')

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
                self._TURN_OVER_DETAIL if self._turn_over_pending()
                else 'it is not your action turn'))]
        w = self.seg.window
        if w is None or not getattr(w, 'die_face', None) \
                or w.die_side != self.human_side.name:
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
        # Only the staged face funds this declaration. The generic gesture
        # filter also accepts W-funded rows for other faces, but that would
        # spend an unstaged W die here when the actual M/P die cannot attack.
        import game_logic as gl
        waiver = gl.ring_waiver_active(self.es, self.human_side)
        dice = self.es.dice.get(self.human_side, [])
        def staged_funds(action):
            idx = action.die_idx
            return (idx is not None and 0 <= idx < len(dice)
                    and not dice[idx].used
                    and (waiver or gesture_match.FACE_LETTER.get(
                        getattr(dice[idx].current, 'name', '')) == w.die_face))
        cands = [a for a in menu if a.action_type == 'army_attack'
                 and sanitize(str(a.params.get('to', ''))).strip()
                 .lower() == tn
                 and staged_funds(a)]
        if sn is not None:
            cands = [a for a in cands
                     if sanitize(str(a.params.get('from', '')))
                     .strip().lower() == sn]
        if not cands:
            self._attack_ask = None
            # BRLANE-12: name the faces that DO carry an attack row on
            # that target (the human can re-stage the right die)
            _dice12 = self.es.dice.get(self.human_side, [])
            _faces12 = sorted({
                gesture_match.FACE_LETTER.get(
                    getattr(_dice12[a.die_idx].current, 'name', ''), '?')
                for a in menu
                if a.action_type == 'army_attack'
                and sanitize(str(a.params.get('to', ''))).strip()
                .lower() == tn
                and a.die_idx is not None
                and 0 <= a.die_idx < len(_dice12)})
            return [('chat', 'refusal', chat_mod.refusal_illegal(
                'no legal attack on %s%s with the staged %s die - %s'
                % (tn, (' from %s' % sn) if sn else '', w.die_face,
                   ('a die showing %s can attack it'
                    % '/'.join(_faces12)) if _faces12
                   else 'no die can attack it now')))]
        sources = sorted(set(sanitize(str(a.params.get('from', '')))
                             .strip().lower() for a in cands))
        if len(sources) > 1:
            # underspecified: ask, listing the legal sources; the
            # human answers 'from <region>'
            self._attack_ask = tn
            return [('chat', 'attack_from_ask',
                     chat_mod.attack_from_ask(sources))]
        self._attack_ask = None
        # BL-108: the face-deduped menu row names ONE same-face die;
        # the binding follows the CLIENT's physical choice - rebind to
        # the staged pid's idx (the window lane's own step) before the
        # commit, or the engine spends a die the client never kills
        if self._needs_staged_die_identity(w):
            self._chat_attack_identity_wait = (w, spec)
            return [('note','waiting for the exact native identity of the staged die')]
        self._rebind_chain_dice([cands[0]], w)
        items = self._commit_human(cands[0], via='chat-attack')
        if self.chat_deferred is None:
            self.last_actor = self.human_side
        self._chat_attack_window = w
        self.findings.append(('chat_attack', '%s from %s' % (tn, sn)))
        # Consume the already-declared assault before later records in this
        # same poll (such as "nc") are matched against the next decision.
        items.extend(self._auto_forced_siege_step())
        return items

    def _human_plain_pass(self):
        import ai

        def _un10(s):
            return sum(1 for d in self.es.dice.get(s, [])
                       if not d.used and d.current is not None)
        if self.phase != 'actions':
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass(
                'the action phase is not open'))]
        if self._human_out_of_turn():
            if self._turn_over_pending():
                # FLD-10d (par5b_g5#3): the TURN-OVER refusal prints
                # the two die counts too - every pass refusal names
                # the numbers it was judged on (the BRLANE-10 spec)
                return [('chat', 'refusal',
                         chat_mod.refusal_cannot_pass(
                             '%s (you have %d unused dice, i have %d)'
                             % (self._TURN_OVER_DETAIL,
                                _un10(self.human_side),
                                _un10(self.ai_side))))]
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass(
                'it is not your action turn'))]
        if self.rand is not None:
            with self.rand.pin():
                menu = ai.get_legal_actions(self.es, self.human_side)
        else:
            menu = ai.get_legal_actions(self.es, self.human_side)
        pa = next((a for a in menu if a.action_type == 'pass'), None)
        if pa is None:
            # BRLANE-10 / M11: the rule lives in ai.py (amendment
            # 1.223: unused ACTION dice only); the refusal prints the
            # numbers it was judged on so a rule reads as a rule
            _hunt10 = sum(1 for d in self.es.hunt_box
                          if getattr(d, 'side', None) is self.ai_side)
            return [('chat', 'refusal', chat_mod.refusal_cannot_pass_counts(
                _un10(self.human_side), _un10(self.ai_side), _hunt10))]
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
            if rec.get('_window_rng_fed'):
                return []
            # Rolling closes a physical Relentless Assault payment prefix.
            # A single removed Regular is otherwise still a possible prefix
            # of paying two hits, and must not start a speculative zero-cost
            # roll while the player is removing the next piece.
            pc = getattr(self.es, 'pending_combat', None) or {}
            ec = pc.get('effect_choice') or {}
            if (ec.get('effect') == 'relentless_assault'
                    and pc.get('deciding_side') is self.human_side):
                if self.pending_buf is None:
                    self.pending_buf = self._new_phase_window()
                self.pending_buf._relentless_roll_closed = True
                if not self.pending_buf.records:
                    self.pending_buf.records.append(rec)
            if self.rand is not None:
                from .pending_roll_wait import admit_human_roll
                values, surplus = admit_human_roll(self, rec)
                r = self.rand.feed('d6', values,
                                   side=self._d6_roll_side(rec))
                rec['_window_rng_fed'] = True
                if r.get('dropped') or surplus:
                    dropped = r.get('dropped', []) + surplus
                    self.findings.append(('excess_human_dice_ignored', tuple(dropped)))
                    if not r['kept']:
                        return [('chat', 'excess_dice',
                                 'roll already complete; ignoring extra dice: %s'
                                 % ' '.join(map(str, dropped)))]
                    return [('chat', 'excess_dice',
                             'using the first %d: %s (ignoring %s)'
                             % (len(r['kept']),
                                ' '.join(str(v) for v in r['kept']),
                                ' '.join(str(v)
                                         for v in dropped)))]
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
        if op == 'kill':
            # BRLANE-5: a human kill no lane claimed (no echo, no
            # pending) is matched or refused, never dropped
            return self._dieless_gesture(rec)
        if op in ('flip', 'die_change', 'random_discard'):
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
            # funnels (a deferred fellowship-move window settles on
            # exactly these records)
            self._tile_face_announce = None
            self._feed_draw_rec(rec)
            lbl = getattr(self, '_tile_face_announce', None)
            if lbl and not getattr(self, 'spectator', False):
                self._tile_face_announce = None
                return [('chat', 'tile_face', lbl)]
            return []
        deck = 'character' if 'character' in what else 'strategy'
        # A native AI-requested draw is printed under the local human's
        # nickname. The deck identifies the card owner, not that nickname's
        # dice side. In sided play, do not learn the human's roll ownership
        # from an AI-owned draw (including Witch-king bonus draws in combat).
        # Spectator replay retains its existing two-player inference.
        from . import sidecar as _sc_mod
        _mside = _sc_mod.meta_side(rec.get('meta'))
        if (getattr(self, 'spectator', False) or side is not self.ai_side
                or str(rec.get('player', '')) == self.nick):
            self._bind_roller_nick(rec.get('player'), _mside or side)
        if side is self.ai_side:
            return []               # our own draw echo (fed via decode)
        import game_logic as gl
        key = (side, deck)
        pco = getattr(self.es, 'pending_card_option', None) or {}
        gandalf_wrong_deck = (pco.get('card_id') == '__gandalf_draw__'
                              and pco.get('deck') != deck)
        # BL-132 L2: ROUTE BY OWED-NESS BEFORE WINDOW MEMBERSHIP.  A
        # draw supplied against a live funnel demand credits the
        # funnel/deferred directive and NEVER enters a gesture
        # window's fact bundle (fellowship_move expresses no draws).
        # The demand check consults BOTH the global pending and the
        # deferral's captured starve call (the global slot may be
        # clobbered by an interleaved attempt). The exact outstanding
        # side/deck demand IS the ownership (Palantir
        # precedent) and outranks any open window.
        try:
            _p117 = (self.rand.pending()
                     if self.rand is not None else None)
        except Exception:
            _p117 = None
        _sc132 = (getattr(self.deferred[0], '_starve_call', None)
                  if self.deferred is not None else None)
        # Only the missing side/deck pair owns this physical draw. Another
        # side's native draw, a different deck, or a random discard request
        # cannot authorize it. An already-fed receipt also cannot authorize a
        # second press while a retained transaction is waiting to commit.
        _p117 = next((call for call in (_p117, _sc132)
                      if self._draw_call_owns(call, side, deck)), None)
        if _p117 is not None and not gandalf_wrong_deck:
            # BL-132 L3: a draw consumed by the DEFERRAL'S OWN
            # starving call dies with that window's rollback - stamp
            # the ownership so _refuse never re-issues it
            if _sc132 is not None and (
                    _p117 is _sc132
                    or (getattr(_p117, 'kind', None)
                        == getattr(_sc132, 'kind', None)
                        and getattr(_p117, 'kw', None)
                        == getattr(_sc132, 'kw', None))):
                self._stamp_window_fed_draw(side, deck)
            self._feed_draw_rec(rec)
            self._retry_blocked()
            self.findings.append(('demanded_draw_fed',
                                  '%s %s' % (side.name, deck)))
            # BL-132 L1: ONE SETTLE POINT - the consuming lane
            # settles the draw in the same motion
            return self._settle_funnel_draw(side, deck)
        # BL-94(b): a refusal that popped a FED draw re-issues it - the
        # re-rendered line is the SAME draw the engine already granted
        # and consumed, never an un-owed human press; consumed
        # one-shot per (side, deck).
        _ri94 = getattr(self, '_reissued_draws94', None) or {}
        _tag94 = ('fp' if side.name == 'FREE_PEOPLES' else 'sa',
                  'char' if deck == 'character' else 'strat')
        if _ri94.get(_tag94, 0) > 0:
            # BL-107: an OPEN funnel demand OUTRANKS the echo token
            # (the one-answerer rule); tokens also TTL to the immediate
            # re-issue window (45s).
            import time as _t107
            _live107 = None
            try:
                _live107 = (self.rand.pending()
                            if self.rand is not None else None)
            except Exception:
                _live107 = None
            _fresh107 = (_t107.time()
                         - getattr(self, '_reissued_ts94', 0)) < 45.0
            if side is self.human_side:
                _fresh107 = (getattr(self, '_manual_reissued_context94', None)
                             == (self.es.turn, self.n_commits_total,
                                 len(self.es.hands[side].hand)))
            if (_live107 is not None
                    and getattr(_live107, 'kind', '')
                    in ('draws', 'card_pick')) or not _fresh107:
                if not _fresh107:
                    self._reissued_draws94 = {}
                    self.findings.append(('reissue_token_expired',
                                          '%s %s' % (side.name,
                                                     deck)))
            else:
                _ri94[_tag94] -= 1
                self._reissued_draws94 = _ri94
                # a pure ECHO: the engine granted and consumed this
                # draw before the refusal; the identity is already in
                # the engine hand and the pid binding survives in
                # card_pid_map - no engine effect, no funnel feed
                self.findings.append(('reissued_draw_consumed',
                                      '%s %s' % (side.name, deck)))
                return []
        # A matching physical draw is both the optional choice and its
        # one chance outcome. There is no owed-draw debt before choosing it.
        # Existing restored-draw echoes above must be consumed first.
        # Read the authoritative pending so a prior hand-limit discard,
        # another card, or the wrong deck cannot be mistaken for this choice.
        if self.rand is not None and self.chat_deferred is None \
                and self.deferred is None \
                and ((pco.get('card_id') == '__gandalf_draw__'
                      and pco.get('deck') == deck)
                     or pco.get('palantir_draw')):
            import ai
            dp = ai.describe_pending(self.es, side)
            if dp is not None and dp['side'] is side \
                    and dp['kind'] == 'card_option':
                take = (next((a for a in dp['actions']
                              if a.action_type == 'card_option'
                              and ((a.params or {}).get('choices') or {}).get('palantir_deck') == deck), None)
                        if pco.get('palantir_draw') else
                        chat_mod.match_choice('draw', dp['actions']))
                if take is not None:
                    self._feed_draw_rec(rec)
                    source = 'palantir_draw' if pco.get('palantir_draw') else 'gandalf_draw'
                    out = self._attempt_chat_action(
                        take, source, physical_norms=(rec.get('norm', ''),))
                    if pco.get('palantir_draw'):
                        self.findings.append(('palantir_draw_fed', '%s %s' % (side.name, deck)))
                    else:
                        self.findings.append(('gandalf_draw_fed', '%s %s' % (side.name, deck)))
                    return out + self._settle_funnel_draw(side, deck)
        n = self.turn_draws.get(key, 0)
        owed = (not gandalf_wrong_deck
                and (n == 0                        # turn-start draw
                     or gl.settle_owed_draw(self.es, side, deck)))
        if not owed:
            # a human mistake handled by the client-native restore is
            # a FINDING + notice, never an alarm (the alarm below is
            # for the un-restorable case only)
            self.findings.append(('unowed_draw_refused',
                                  '%s %s' % (side.name, deck)))
            items = [('chat', 'refusal',
                      chat_mod.refusal_illegal(
                          'Gandalf requires a Free %s card, not a %s draw'
                          % (pco.get('deck', 'character').title(), deck)
                          if gandalf_wrong_deck else 'un-owed %s draw' % deck))]
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
                if n is None or n > 1:
                    # BL-117(iii): SPAN-STRICT - the refusal reverts
                    # ONLY its own draw line (the newest line = exactly
                    # 1 press); n>1 means other lines sit above the
                    # draw and pressing through them would pop them
                    # too; hold-and-witness instead, never collateral
                    # pops.
                    if n is not None and n > 1:
                        self.findings.append(
                            ('unowed_draw_press_held',
                             '%d presses would pop other lines'
                             % n))
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
        self.hands.human_draw(deck, record=rec)
        # D4: the hand-limit discard resolves NOW through the engine's
        # own pending (which card = a learned/human decision)
        gl.open_hand_limit_discard(self.es)
        if self.phase == 'draws':
            waived = self._depleted_deck_draw_waivers(side)
            if waived is not None:
                return waived + self._ai_turn_start_draws()
        return []

    def _deck_depleted(self, side, deck):
        """Rulebook (text line 1763): a depleted deck is never drawn
        from again - its turn-start draw is not owed.  The engine deck
        list is the one count authority (conserved with the client by
        human_draw / adopt_decks)."""
        import game_state as gs
        attr = '%s_%s_deck' % (
            'fp' if side is gs.Side.FREE_PEOPLES else 'sh',
            'char' if deck == 'character' else 'strat')
        return not getattr(self.es, attr, None)

    def _depleted_deck_draw_waivers(self, side):
        """The turn-start hand-off gate for `side`'s draws: None while
        a STOCKED deck is still undrawn this turn (the hand-off waits
        for that draw line); otherwise the announce items - a panel
        note + finding per DEPLETED undrawn deck (empty when both
        decks were drawn)."""
        items = []
        for dk in ('character', 'strategy'):
            if self.turn_draws.get((side, dk)):
                continue
            if not self._deck_depleted(side, dk):
                return None
            self.findings.append(('draw_skipped_deck_depleted',
                                  '%s %s' % (side.name, dk)))
            items.append(('note', '%s %s deck is depleted - that '
                                  'turn-start draw is not owed '
                                  '(rulebook: a depleted deck is '
                                  'never drawn from again)'
                          % (side.name, dk)))
        return items

    def _draw_call_owns(self, call, side, deck):
        """A native deck draw answers only an exact, still-missing request."""
        if call is None or getattr(call, 'kind', None) != 'draws':
            return False
        pairs = (getattr(call, 'kw', None) or {}).get('items') or ()
        return (side.name, deck) in self.rand._missing_cards(pairs)

    def _settle_funnel_draw(self, side, deck):
        """BL-132 L1 (fpgame18, wave 26): ONE SETTLE POINT.  Every
        lane that consumes a physically-rendered human deck draw
        settles it in the same motion: one unit of owed-draw debt
        closes, the turn count advances, and the draws-phase trigger
        still fires.  The funnel lanes used to skip ALL of this - the
        T2 character draw left turn_draws at 0 (so the AI's
        turn-start draws never ran) and the debt open, and the engine
        entered the action phase owing a phantom (FP, character)
        demand that wedged a legal fellowship_move into a 4-undo
        cascade.  Hand/deck bookkeeping stays OUT: _feed_draw_rec
        swapped a placeholder into the fictional deck and the
        engine's own draw removes it (deck -1) - hands.human_draw
        here would double-count.  No _snapshot either: the deferred
        action's own commit owns this line in the undo ledger."""
        import game_logic as gl
        key = (side, deck)
        gl.settle_owed_draw(self.es, side, deck)
        self.turn_draws[key] = self.turn_draws.get(key, 0) + 1
        if self.phase == 'draws' \
                and self.turn_draws.get((side, 'character')) \
                and self.turn_draws.get((side, 'strategy')):
            return self._ai_turn_start_draws()
        return []

    def _stamp_window_fed_draw(self, side, deck):
        """BL-132 L3 (fpgame18): a draw fed against the OPEN
        deferral's own starving demand dies with that window's
        rollback - the refusal must NOT re-issue it (the client undo
        already restored the deck; the re-issue drew a FRESH physical
        card into a hand the engine had rolled back: pid 139, 'deck
        fp char: engine=21 client=20' settled).  Stamp the ownership
        on the window so _refuse can tell a window-owned fed draw
        from an externally-granted one (BL-94(b)'s original case,
        which DOES re-issue).  Ambiguity leaves no stamp - the
        re-issue default is the known-good BL-94(b) behavior."""
        w = (self.deferred[0] if self.deferred is not None
             else self.pending_buf)
        if w is None:
            return
        tag = ('fp' if side.name == 'FREE_PEOPLES' else 'sa',
               'char' if deck == 'character' else 'strat')
        d = getattr(w, '_fed_draws132', None)
        if d is None:
            d = w._fed_draws132 = {}
        d[tag] = d.get(tag, 0) + 1

    def _on_action_roll(self, rec):
        """Rendered letters roll: adopt faces onto the roller's pool in
        order (fungible until bound; pid binding rides the raw roll
        line, fact: rendered letters carry no pids)."""
        import game_state as gs
        import game_logic as gl
        faces = rec.get('faces', [])
        # BRLANE-15 / F05: the sidecar's die channel names the FAMILY
        # (the die_id_map is the one pid -> side registry) - faces and
        # counts decide only when the channel is absent
        from . import sidecar as _sc15
        _fam15 = None
        _mp15 = _sc15.roll_pids(rec.get('meta'))
        _raw15 = (None if rec.get('_allocation_closed') else
                  self._take_public_roll_witness(faces, _mp15))
        if not getattr(self, 'spectator', False) and self.phase in ('declare', 'fellowship_tail', 'guide'):
            return self._refuse_phase_line(rec, chat_mod.refusal_illegal(
                'finish the Fellowship phase and Hunt allocation before rolling action dice'),
                'EARLY ACTION ROLL')
        if _raw15 is not None and _mp15 is None:
            _mp15 = _raw15['pids']
            _fam15 = gs.Side[_raw15['side']]
            self.findings.append(('roll_identity_from_native_command',
                                  '%s: %s' % (_fam15.name, ' '.join(_mp15))))
        if _mp15:
            _sides15 = {self.view.die_id_map.get(str(p), (None, 0))[0]
                        for p in _mp15}
            _sides15.discard(None)
            if len(_sides15) == 1:
                _fam15 = next(iter(_sides15))
        _byface15 = (gs.Side.SHADOW if 'E' in faces
                     else gs.Side.FREE_PEOPLES if 'W' in faces else None)
        if _fam15 is not None:
            # the channel decided where the faces + counts could NOT
            # (both pools fit, or the heuristic said otherwise): on
            # the books; an agreeing verdict is silent
            _cnt15 = [s for s in (gs.Side.SHADOW, gs.Side.FREE_PEOPLES)
                      if len([d for d in self.es.dice.get(s, [])
                              if not d.used and d.current is None])
                      == len(faces)]
            _heur15 = (_byface15 if _byface15 is not None
                       else (_cnt15[0] if len(_cnt15) == 1 else None))
            if _heur15 is not _fam15:
                self.findings.append(('roll_side_from_sidecar',
                                      '%s: %s (faces/count said %s)'
                                      % (_fam15.name,
                                         ' '.join(_mp15)[:40],
                                         getattr(_heur15, 'name',
                                                 'ambiguous'))))
        side = _fam15 if _fam15 is not None else _byface15
        if (side is None and self.phase == 'eyes'
                and self.human_side is gs.Side.SHADOW
                and str(rec.get('player', '')) != self.nick
                and self.sidecar is not None and self.sidecar.present
                and not getattr(self, 'spectator', False)):
            # At this phase only the human Shadow can roll. Its public native
            # identities may arrive with the next request; preserve the result.
            side = gs.Side.SHADOW
        if side is None:
            # ambiguous small rolls: match by unused-dice count
            _match = [s for s in (gs.Side.SHADOW, gs.Side.FREE_PEOPLES)
                      if len([d for d in self.es.dice.get(s, [])
                              if not d.used and d.current is None])
                      == len(faces)]
            if len(_match) > 1:
                # BOTH pools fit and no Eye/Will face disambiguates:
                # Shadow first (unchanged), but never SILENT - the
                # action phase waits for BOTH rolls (pools_rolled), so
                # a wrong-side adoption leaves the real owner unrolled
                # forever.  Reachable when the eyes leave the Shadow
                # pool the same size as the FP pool AND the human rolls
                # before the AI's own !CALL lands.
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
        if side is gs.Side.SHADOW:
            from .allocation_roll import defer as defer_allocation_roll
            if defer_allocation_roll(self, rec, _mp15 or self._pending_roll_pids.get('SHADOW')):
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
        if rec.get('_allocation_roll_indices') is not None:
            idxs = rec['_allocation_roll_indices']
        if len(faces) > len(idxs) and idxs:
            # BL-39-5: a roll carrying MORE faces than dice awaiting
            # them is a RE-ROLL of settled dice: adopting it re-faces
            # settled dice and de-syncs both pools, and rolls are not
            # client-undoable - adopt NOTHING, say what is owed, alarm.
            self.alarms.append(
                'ROLL OVERSIZE: %d faces but only %d %s dice await a '
                'roll - NOTHING adopted (re-rolls corrupt settled '
                'dice)' % (len(faces), len(idxs), side.name))
            self.findings.append(('roll_oversize_refused',
                                  '%s %d>%d' % (side.name, len(faces),
                                                len(idxs))))
            out39 = [('chat', 'refusal', chat_mod.refusal_illegal(
                'only %d of your dice await a roll - select and roll '
                'just %s' % (len(idxs),
                             'that die' if len(idxs) == 1
                             else 'those dice')))]
            # BL-120: the refused roll's PHYSICAL consequences stand
            # client-side (its rolled-E faces walked into the hunt box;
            # dice boxes are un-adoptable) - count the refused roll's
            # Eyes and DEMAND the counter-drag (claim-once; the census
            # settles when the human drags them back).
            _n_e120 = sum(1 for f in faces if str(f).upper() == 'E')
            if _n_e120 > 0:
                self._roll_restore_owed = (
                    int(_n_e120),
                    __import__('time').time() + 60.0)
                out39.append(('chat', 'prompt',
                              chat_mod.roll_restore_note(_n_e120)))
                self.findings.append(('roll_restore_demanded',
                                      '%d eye(s)' % _n_e120))
            return out39
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
        if side is gs.Side.SHADOW:
            # Human Shadow may allocate eyes through several native drags.
            # Its roll is the close of that physical allocation window.
            self.es._eye_allocation_phase = False
        # bind the raw roll line's pid order to THE SAME index order
        # (sanctioned raw #2; the pool order changes across turns - a
        # static pre-seed mislabels every die after recover).  PARK
        # this adoption's index order; the bind fires now if the raw
        # line already arrived, or retroactively when it does.
        # BL-39c: park the ADOPTED indices only (the zip consumed the
        # first len(faces)).
        self._last_adopt_idxs[side.name] = (list(idxs)[:len(faces)],
                                            len(faces))
        # R6 / S10: remember WHEN this adoption parked - a
        # sidecar-present roll whose record carries no die channel
        # (segs[1]) is a LOUD alarm after a few ticks, never a silent
        # unbound pool
        self._adopt_parked_tick[side.name] = getattr(self, '_tick_n', 0)
        # Both explicit identity channels feed the same parking spot.
        # An empty per-record sidecar may use the correlated native command;
        # if both are present their exact PID order was checked above.
        from . import sidecar as _sc_mod
        _mp = _sc_mod.roll_pids(rec.get('meta'))
        if _mp is None and _raw15 is not None:
            _mp = _raw15['pids']
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
            # R2 / G11 (the SHADOW pool rolls FIRST): the human
            # Shadow's roll landed - the AI-as-FP now orders its OWN
            # pool roll (live_server -> '!CALL roll_action_dice')
            items.append(('ai_roll', None))
        if self.phase in ('roll', 'eyes') \
                and (getattr(self, 'spectator', False)
                     or self.pools_rolled()):
            self.phase = 'actions'
            # THE RULE (the certified AUTO machine: `active =
            # Side.FREE_PEOPLES` at every action-resolution start; USER
            # live specimen: 'shadow is not supposed to go first'): the
            # FREE PEOPLES side acts FIRST, side-based never
            # seat-based.  Mirrored through the ONE answerer: seeding
            # last_actor=SHADOW makes ai._next_acting_side answer FP.
            # An idle FP human surfaces via the stall instruments - the
            # AI WAITS.  (Spectator: byte-pinned census, unseeded.)
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
        if len(pids) != n_faces or len(pids) != len(idxs):
            # BL-39c: pids must match the ADOPTED INDEX COUNT too (a
            # re-roll's pid list zipped onto a 1-index adoption shifts
            # every published pid - the BL-25 positional inversion)
            self.alarms.append(
                'ROLL pid binding count mismatch: %d pid(s) vs %d '
                'rolled %s faces vs %d adopted idx(s) - bindings '
                'left as-is'
                % (len(pids), n_faces, side_name, len(idxs)))
            return
        side = (gs.Side.SHADOW if side_name == 'SHADOW'
                else gs.Side.FREE_PEOPLES)
        # pid[i] <-> the i-th adopted engine index (zip stops at the
        # engine roster when it is shorter - the partial-adoption
        # alarm already fired at adoption)
        for p, i in zip(pids, idxs):
            self.view.bind_die(p, side, i)
        # BL-25 coverage assert: after a roll bind this side's
        # registration slots must cover 0..n-1 with one pid each - a
        # collision or drop publishes a pid:null hole and wedges the
        # seat
        n_dice = len(self.es.dice.get(side, []))
        regs = sorted(i2 for _p2, (s2, i2)
                      in self.view.die_id_map.items() if s2 is side)
        if regs != list(range(n_dice)) and n_dice:
            self.alarms.append(
                'DIE BINDING COVERAGE GAP (%s): regs %s vs %d engine '
                'dice - the published dice table will hole/shift '
                '(BL-25)' % (side_name, regs, n_dice))

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
        """Attribute native AI rolls directly; otherwise use draw-zone binding."""
        player = str(rec.get('player', '') or '')
        if player == self.nick:
            return self.ai_side.name
        return self.nick_side.get(player)

    def _complete_human_eye_allocation(self):
        """Close an explicit human allocation after all received gestures land."""
        request = getattr(self, '_eye_completion_requested', None)
        if request is None:
            return
        turn, target = request
        if (self.phase != 'eyes' or self.human_side.name != 'SHADOW'
                or int(self.es.turn) != turn or getattr(self, 'spectator', False)):
            self._eye_completion_requested = None
            return
        if self._feed_seq != len(self.feed.lines) or getattr(self.feed, '_buf', ''):
            return
        have = int(getattr(self.es, 'eyes_allocated', 0) or 0)
        allocated = None
        if self.view.dump_path:
            # The native pool is authoritative at this explicit completion
            # boundary, including eyes the player returned after announcing.
            from . import client_board
            from .repair_evidence import dump_signature, same_dump
            from game_state import Side
            try:
                evidence = dump_signature(self)
                header = evidence['header']
                if header is None or self._dump_position_hold(header):
                    return
                self._check_dump_session(header)
                dump = client_board.read_board(self.view.dump_path)
                if not dump or not same_dump(self, evidence):
                    return
            except (OSError, ValueError):
                return
            dice = self.es.dice[Side.SHADOW]
            entries = sorted((int(index), str(pid))
                             for pid, (owner, index) in self.view.die_id_map.items()
                             if owner is Side.SHADOW)
            if [index for index, _ in entries] != list(range(len(dice))):
                return
            zones = [str(dump.get(pid, '')).strip().lower() for _, pid in entries]
            if any(zone not in ('sa dice', 'hunt box', 'the hunt for the ring') for zone in zones):
                return
            allocated = [index for (index, _), zone in zip(entries, zones) if zone != 'sa dice']
            have = len(allocated)
        if target is not None and have != target:
            return
        # Completion confirms the physical pool; it does not waive the
        # engine's allocation limits. Keep the request alive while the human
        # corrects the pool, so no second completion chat is required.
        counts = self._eye_menu_counts(refresh=True)
        if counts is not None and have not in counts:
            rejected = (turn, have, tuple(sorted(counts)))
            if rejected != getattr(self, '_eye_completion_rejected', None):
                self._eye_completion_rejected = rejected
                self.findings.append(('eye_completion_refused',
                                      'total=%d menu=%s' % (have, sorted(counts))))
                reason = ('at least %d eye must be allocated this turn; '
                          'place it on the board to finish Hunt allocation'
                          % min(counts) if counts and have < min(counts) else
                          'only %s eye(s) may go to the Hunt this turn; '
                          'correct the allocation on the board'
                          % ('/'.join(str(count) for count in sorted(counts)) or '0'))
                self._out_queue.append(
                    ('chat', 'refusal', chat_mod.refusal_illegal(reason)))
            return
        self._eye_completion_rejected = None
        if allocated is not None:
            import game_logic as gl
            # No faces have rolled during Hunt allocation. Reconcile only
            # Shadow's pre-roll pool; leave every physical piece untouched.
            sh_ids = {id(die) for die in dice}
            self.es.hunt_box = [die for die in self.es.hunt_box if id(die) not in sh_ids]
            for die in dice:
                die.used, die.current = False, None
            self.es.eyes_allocated = 0
            gl.allocate_eyes(self.es, have, indices=allocated)
        self._eye_completion_requested = None
        self.es._eye_allocation_phase = False
        self.phase = 'roll'

    def _on_eye(self, rec):
        own = str(rec.get('player', '')) == self.nick
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
                and str(rec.get('player', '')) != self.nick:
            # 'allocates an eye.' = ONE Shadow die flipped to Eye and
            # placed in the Hunt Box BEFORE the roll (D3); without it
            # the engine keeps 7 unset dice against a 6-face rendered
            # roll and the roller cannot be resolved
            import game_state as gs
            if self.human_side is gs.Side.SHADOW:
                self._allocate_one_eye(rec)
        return []

    def _eye_menu_counts(self, refresh=False):
        """The ENGINE's eye menu for this turn - the eye_allocate
        counts get_legal_actions offers under the allocation phase
        (p.17: at most one per Companion, at least one when the FP
        retrieved a die) - read on a clone BEFORE this turn's first
        eye lands and held for the turn.  ONE legality answerer for
        the human's allocation. Native board completion reconciles provisional
        gesture counts before the allocation is finalized."""
        import ai
        import game_state as gs
        turn = getattr(self.es, 'turn', None)
        if not refresh and getattr(self, '_eye_counts_turn', None) == turn \
                and getattr(self, '_eye_counts', None) is not None:
            return self._eye_counts
        clone = ai.clone_state(self.es)
        clone._eye_allocation_phase = True
        # the probe asks for the EYE menu alone: the FP declare flag
        # (K2, still armed while the AI-FP's step is queued) precedes
        # it in get_legal_actions and would answer instead
        clone._declare_phase = False
        # The menu describes the TOTAL allocation from the phase's starting
        # pool. A recovered board may already have marked some dice used before
        # this first lookup; those allocated dice still count toward its limit.
        for die in clone.dice.get(gs.Side.SHADOW, ()):
            die.used = False
            die.current = None
        clone.hunt_box = [die for die in clone.hunt_box if die.side is not gs.Side.SHADOW]
        clone.eyes_allocated = 0
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

    def _allocate_one_eye(self, rec=None, pid=None):
        """Engine-side Phase-3 allocation of ONE eye onto any unset
        unused Shadow die (dice are fungible until faces bind). With a
        native board, this is provisional bookkeeping: completion validates
        the final allocation after returned/corrected dice. Legacy feed-only
        input checks the engine menu per gesture. Returns True when booked.

        `pid`: the client die piece that carried the eye (the dragged
        die, or the piece named by the menu handler's Controls.372
        line).  BL-10 (2026-08-25): the allocated die never ROLLS, so
        the roll binding never names it - without an allocation-time
        bind the status table publishes its index with pid:null and
        the seat's fail-closed die map refuses EVERY action all turn
        (shgame2 turn 1)."""
        import game_logic as gl
        import game_state as gs
        if (not getattr(self, 'spectator', False)
                and self.phase in ('declare', 'fellowship_tail', 'guide')
                and self.human_side is gs.Side.SHADOW):
            if rec is not None:
                self._out_queue.extend(self._refuse_phase_line(
                    rec, chat_mod.refusal_illegal(
                        'the Free Peoples must finish their Guide choice before Hunt allocation'),
                    'EARLY HUNT ALLOCATION'))
            return False
        dice = self.es.dice.get(gs.Side.SHADOW, [])
        idx = next((i for i, d in enumerate(dice)
                    if not d.used and d.current is None), None)
        board_allocation = (self.phase == 'eyes' and self.view.dump_path
                            and not getattr(self, 'spectator', False))
        if idx is None:
            if board_allocation:
                # These are draft gestures; returned dice may have made this
                # incremental count obsolete. Completion reads the real pool.
                return False
            self.alarms.append('EYE ALLOCATION with no unset unused '
                               'Shadow die - roster desync')
            return False
        counts = (None if getattr(self, 'spectator', False)
                  or not self.enforce_human or board_allocation
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
        # BRLANE-6 / F05: ONE ledger entry per allocation LINE (the
        # client's undo of 'allocates an eye.' / the SA Dice -> Hunt
        # drag pops it: no phantom hunt-box eye); a multi-die drag is
        # one line = one entry
        if rec is not None and not getattr(self, 'spectator', False):
            _norm6 = str(rec.get('norm', '') or '')
            if _norm6 and getattr(self, '_eye_snapshot_seq', None) \
                    != self._feed_seq:
                self._snapshot([_norm6], 'eye')
                self._eye_snapshot_seq = self._feed_seq
        gl.allocate_eyes(self.es, 1, indices=[idx])
        if pid is None and rec is not None:
            # BL-25: the die pid rides the SIDECAR meta channel
            # (segs[1], the same contract as roll_pids); the raw-token
            # regex is the channel-absent fallback
            try:
                from . import sidecar as _sc25
                _ps25 = (_sc25.roll_pids(rec.get('meta'))
                         or _sc25.meta_pids(rec.get('meta')) or [])
                pid = _ps25[0] if _ps25 else None
            except Exception:
                pid = None
            if pid is None:
                import re as _re
                _m10 = _re.search(r'<~Controls\.372~>(\d+)',
                                  str(rec.get('raw', '') or ''))
                if _m10:
                    pid = _m10.group(1)
        if pid is not None:
            self.view.bind_die(str(pid), gs.Side.SHADOW, idx)
        elif getattr(self, 'live', True) \
                and not getattr(self, 'spectator', False):
            # no sidecar pid in LIVE play: a findings row - the ROLL-
            # time coverage assert (_try_bind_roll) is the loud alarm
            # if a real hole materializes.  Prefix replays and the
            # spectator carry no sidecars by construction (BL-25).
            self.findings.append(('eye_alloc_unbound',
                                  'no sidecar pid on the alloc line'))
        return True

    def _on_phase_move(self, rec):
        """Non-window board moves: recover-dice detection (D1), hunt
        pool maintenance, human eye-allocation drags (fact 12), the
        STANDALONE Elven-Ring walk (no die staged yet)."""
        to = str(rec.get('to', ''))
        frm = str(rec.get('from', ''))
        obj = str(rec.get('obj', ''))
        if (self.phase == 'draws' and not getattr(self.es, 'dice_rolled', False)
                and self._turn_start_recovery_leg(rec)):
            # Either pool may be recovered first. The first leg advances the
            # turn; all remaining physical pool returns are the same gesture.
            self.findings.append(('turn_start_recovery_leg', str(rec.get('norm', ''))))
            return []
        # CORPUS eye-allocation form (client fact 12): the plain drag
        # 'SA Dice -> The Hunt For The Ring' is the ONLY observed human
        # allocation gesture (0 of 17,693 human logs used
        # Controls.795).  The phase gate keeps post-roll drags of
        # ROLLED eyes out (adopt_rolled_faces already moved those).
        if to.strip().lower() == 'the hunt for the ring' \
                and frm.strip().lower() == 'sa dice' \
                and 'action die' in obj.lower() \
                and self.phase in ('draws', 'declare', 'eyes'):
            import game_state as gs
            if self.human_side is gs.Side.SHADOW \
                    or getattr(self, 'spectator', False):
                # the first Shadow eye gesture closes a HUMAN FP's
                # declaration opportunity (D3 order); an AI-FP's
                # declaration is its own LEARNED pending (standing
                # ruling: declare/reveal choices are FP decisions) -
                # the decline settles ONLY when the eye closes the
                # phase, the ONE predicate the menu-line lane _on_eye
                # uses (_eye_closes_declare), so the AI's queued
                # declare step keeps its menu and the Mordor entry
                # check (FINAL ruling 2026-08-16: same turn start,
                # after the declaration opportunity) still runs.
                # The spectator arms the flag in its own draw handler;
                # its settle stays unconditional.
                if getattr(self, 'spectator', False) \
                        or self._eye_closes_declare():
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
                # BL-10: the dragged die pids ride the RAW line - pass
                # them down so the allocated (never-rolled) dice keep
                # a client binding through the roll adoption
                import re as _re10
                _raw_pids = [
                    p for p in _re10.findall(
                        r'~>(\d{3})(?=<~)',
                        str(rec.get('raw', '') or ''))
                    if self.view.die_id_map.get(p, (None, 0))[0]
                    is gs.Side.SHADOW]
                for _i10 in range(n):
                    _p10 = (_raw_pids[_i10]
                            if _i10 < len(_raw_pids) else None)
                    if not self._allocate_one_eye(rec, pid=_p10):
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
            # Current Action holds both staged and spent dice. Returning
            # several unused staged pieces is selection cancellation, so
            # only engine exhaustion can establish a recovery boundary.
            import game_state as gs
            side = (gs.Side.FREE_PEOPLES if to == 'FP Dice'
                    else gs.Side.SHADOW)
            dice = self.es.dice.get(side, [])
            all_used = bool(dice) and all(d.used for d in dice)
            if (all_used
                    or (self._turn_over_pending() and self._turn_start_recovery_leg(rec))):
                return self._on_recover(rec)
            # D-T2-6: the self-undo shape is only right for a die that
            # is merely STAGED (open window, nothing in the engine).  A
            # die whose action is COMMITTED (the sidecar pid resolves
            # to a USED engine die) - or any die leaving the Hunt Box,
            # where a die sits only after a committed Fellowship move -
            # cannot be 'self-undone' by a drag: REFUSED like the eye
            # drag (client-native undo of this ONE line + a notice);
            # the engine stays whole.
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
            # Unused dice back in their pool do not advance the turn.
            self.findings.append(('die_self_undo',
                                  '%s from %s' % (obj[:40], frm[:30])))
            return []
        # BRLANE-5: every other human move (no die, no pending) is
        # matched or refused - never dropped for the census to adopt
        return self._dieless_gesture(rec)

    def _turn_start_recovery_leg(self, rec):
        """Only actual dice returned from their spent/Hunt boxes to own pool."""
        import rendered_replay as rr
        import game_state as gs
        import re
        destination = str(rec.get('to', '')).strip().lower()
        source = str(rec.get('from', '')).strip().lower()
        side = {'fp dice': gs.Side.FREE_PEOPLES, 'sa dice': gs.Side.SHADOW}.get(destination)
        if side is None or source not in ('current action', 'the hunt for the ring',
                                         'fp casualties', 'sa casualties', 'casualties'):
            return False
        obj = str(rec.get('obj', ''))
        tokens = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
        parsed = [rr.parse_piece(token) for token in tokens]
        label = 'Free' if side is gs.Side.FREE_PEOPLES else 'Shadow'
        return bool(parsed) and all(p is not None and p[0] == 'die'
                                    and re.match(r'^(?:\[\d+\]\s*)?'+label+r' Action Die\b', token, re.I)
                                    for p, token in zip(parsed, tokens))

    def _dragged_die_used(self, rec, side):
        """True if any identified moved die is used; False if all are unused.

        Rendered-only single-die records retain their existing fallback.
        """
        from .staged_dice import identities
        grouped = identities(self, rec, side)
        if grouped is not None:
            return any(used for _face, used in grouped.values())
        from . import sidecar as sc_mod
        from .native_die_identity import pids as public_die_pids
        pids = public_die_pids(rec)
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
        self._refusal_loop_check(str(notice))
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

    def _refusal_loop_check(self, text, limit=3, window_s=120.0):
        """BL-141 (shgame21 livelock): a seat retry with no budget met
        a driver refusal with no repeat limit, and the AI starved 4400
        ticks behind it.  The SAME refusal text `limit` times inside
        `window_s` is a loop - one loud DIVERGENCE-class alarm (the
        seat's stop signal), never an eternal politely-repeated
        refusal."""
        import time as _t141
        now = _t141.time()
        hist = getattr(self, '_refusal_hist141', None)
        if hist is None:
            hist = self._refusal_hist141 = {}
        key = str(text)[:120]
        stamps = [t for t in hist.get(key, []) if now - t < window_s]
        stamps.append(now)
        hist[key] = stamps
        if len(stamps) == limit:
            msg = ('REFUSAL LOOP: the same illegal gesture refused %d '
                   'times in %ds - the human path is wedged on "%s"; '
                   'held' % (limit, int(window_s), key[:80]))
            if msg not in self.alarms:
                self.alarms.append(msg)
            self.findings.append(('refusal_loop', key[:80]))

    def _on_ring_walk(self, rec):
        """Collect a physical ring payment and explicit selected die change.

        A counter alone does not identify either the die or its new face.
        Keep that gesture undoable until the client's face change supplies
        the missing choice; never mint an arbitrary engine menu row.
        """
        import ai
        import game_state as gs
        from .sidecar import meta_pids
        from emitter import die_piece_for
        why = None
        if self.phase != 'actions':
            why = 'the action phase is not open'
        elif self._human_out_of_turn():
            why = (self._TURN_OVER_DETAIL if self._turn_over_pending()
                   else 'it is not your action turn')
        elif ai.describe_pending(self.es) is not None:
            why = 'a decision is still open'
        if why:
            return self._refuse_ring_record(rec, why)
        with self.rand.pin():
            menu = [a for a in ai.get_legal_actions(self.es, self.human_side)
                    if a.action_type == 'elven_ring']
        if not menu:
            return self._refuse_ring_record(rec, 'no elven ring use is legal now (none left, or one already used this turn)')
        records = list(getattr(self, '_ring_gesture_records', None) or ())
        is_change = rec.get('op') == 'die_change'
        if any((r.get('op') == 'die_change') == is_change for r in records):
            return self._refuse_ring_record(rec, 'finish or undo the current ring choice before starting another')
        if is_change:
            if rec.get('side') != self.human_side.name or not rec.get('frm') or not rec.get('to'):
                return self._refuse_ring_record(rec, 'the ring needs your selected die and its old and new faces')
        else:
            frm = str(rec.get('from') or rec.get('zone') or '').lower().strip()
            to = str(rec.get('to') or '').lower().strip()
            fp_slot = lambda z: z.startswith('elven ring') or z in ('vilya slot', 'nenya slot', 'narya slot')
            sh_slot = lambda z: z.startswith(('dark ring', 'used ring'))
            good = (fp_slot(frm) and sh_slot(to) if self.human_side == gs.Side.FREE_PEOPLES
                    else sh_slot(frm) and (to in ('sa casualties', 'sa dead', 'dead') or rec.get('op') == 'kill'))
            if not good:
                return self._refuse_ring_record(rec, 'the ring must leave your available ring slot for the other side (FP) or casualties (Shadow)')
        records.append(rec)
        changed = next((r for r in records if r.get('op') == 'die_change'), None)
        if changed:
            old, new = changed.get('frm'), changed.get('to')
            menu = [a for a in menu
                    if gesture_match.FACE_LETTER.get(gs.ActionDieFace(a.params['new_face']).name) == new
                    and gesture_match.FACE_LETTER.get(self.es.dice[self.human_side][a.params['target_die_idx']].current.name) == old]
            pids = meta_pids(changed.get('meta'))
            if pids is not None:
                menu = [a for a in menu if die_piece_for(self.view, self.human_side.name,
                          a.params['target_die_idx']) in pids]
            if not menu:
                return self._refuse_ring_record(rec, 'that selected die or face change is not a legal Elven Ring use')
        self._ring_gesture_records = records
        if len(records) < 2 or len(menu) != 1:
            return [('chat', 'prompt', chat_mod.elven_ring_choice_note(
                face_changed=changed is not None))]
        act = menu[0]
        # Java's die-change command has no undo snapshot. The preceding
        # marker snapshot restores a later face change collaterally. If
        # the face changed first, undoing its later payment restores an
        # unresolved face-first prefix, not an invented second undo line.
        payment = next(r for r in records if r.get('op') != 'die_change')
        payment_index = records.index(payment)
        self._ring_gesture_records = records[:payment_index] or None
        if self._commit_chain([act], [payment.get('norm', '')], 'ring'):
            self._ring_gesture_records = None
            self.findings.append(('ring_standalone', str(act.params)))
            self.last_actor = self.human_side
        return []

    def _refuse_ring_record(self, rec, why):
        from .undo_model import presses_to_revert
        from emitter import render
        norm = str(rec.get('norm', '') or '')
        self.findings.append(('ring_refused', '%s :: %s' % (why, norm[:60])))
        items = [('chat', 'refusal', chat_mod.refusal_illegal(why))]
        if norm:
            self.uncommitted.append(norm)
        shim = PhaseWindow()
        prefix = list(getattr(self, '_ring_gesture_records', None) or ())
        if rec.get('op') == 'die_change' and prefix \
                and prefix[-1].get('op') != 'die_change':
            # The face command itself cannot be undone separately in
            # Java. Undo the pending marker and its collateral change.
            shim.records.extend(prefix)
            self._ring_refusal_collateral = norm
        shim.records.append(rec)
        n = presses_to_revert(self.feed.lines, self._window_span(shim), self.nick)
        if n is None:
            self.alarms.append('RING GESTURE REFUSED but undo-ineligible - corrective restore owed')
        else:
            items.extend(('line', render('undo', {'nick': self.nick}, local=False)) for _ in range(n))
        return items

    def _on_recover(self, rec):
        """Human recovered dice -> the turn rolled over (D1: trigger
        the AI's own recover, never wait for the marker)."""
        if self.game_over is not None:
            return []
        if (self.human_physical_work_pending() or self.echo_pending()
                or getattr(self, '_human_casualty_prefix', None)):
            return [('chat', 'prompt', chat_mod._register(
                'Finish the committed board changes and their pending resolution before recovering action dice.'))]
        import ai
        import game_logic as gl
        # Spending the last die opens no new turn until its whole action
        # finishes. Expiring a live pending retains it; resetting below would
        # still advance the turn and refund every die under that decision.
        unfinished = (not getattr(self, 'spectator', False)
                      and (ai.describe_pending(self.es) is not None
                           or self.combat_recovery_open()
                           or self.deferred is not None or self.chat_deferred is not None
                           or bool(self.pending_buf and self.pending_buf.records)
                           or self.action_completion_pending()))
        if unfinished:
            self.findings.append(('recover_held_for_decision', str(rec.get('norm', ''))))
            if not self.enforce_human:
                return [('chat', 'refusal', chat_mod.refusal_illegal(
                    'finish the current decision and its board changes before '
                    'recovering action dice'))]
        # BL-13: recover is the TURN-END gesture - with dice still
        # unspent on either side it is REFUSED (the action phase runs
        # until both pools are spent); client-native undo walks the
        # recovery legs back.
        import game_state as _gs13
        # BL-45: EYE dice never count as 'unspent' for this guard.
        _left = sum(1 for _s13 in (_gs13.Side.FREE_PEOPLES,
                                   _gs13.Side.SHADOW)
                    for _d13 in self.es.dice.get(_s13, [])
                    if not _d13.used
                    and getattr(_d13.current, 'name', None) != 'EYE')
        if unfinished or (_left and self.game_over is None):
            import time as _t49
            _now49 = _t49.time()
            _burst49 = (_now49 - getattr(self, '_recover_refusal_ts',
                                         0)) < 6.0
            self._recover_refusal_ts = _now49
            # BL-49: the recover press is a MULTI-LINE burst - one
            # refusal chat per burst, every leg undone
            reason = ('finish the current decision and its board changes before '
                      'recovering action dice' if unfinished else
                      'cannot recover - %d action %s still '
                      'unspent; the turn ends when both pools are '
                      'used' % (_left, 'die is' if _left == 1 else 'dice are'))
            items = ([] if _burst49 else
                     [('chat', 'refusal', chat_mod.refusal_illegal(reason))])
            from .undo_model import presses_to_revert
            from emitter import render
            _norm13 = str(rec.get('norm', '') or '')
            if _norm13:
                self.uncommitted.append(_norm13)
            shim = PhaseWindow()
            shim.records.append(rec)
            n13 = presses_to_revert(self.feed.lines,
                                    self._window_span(shim),
                                    self.nick)
            if n13 is None:
                # BL-49: undo-ineligible legs are REVERSED from the
                # line's own pids (the reverse drag is emitted)
                self.alarms.append('EARLY RECOVER undo-ineligible - '
                                   're-emitting the reverse drag')
                try:
                    import re as _re49
                    import emitter as _em49
                    from board import sanitize as _sz49
                    _fam49 = set(_em49._SH_DIE_IDS
                                 + _em49._FP_DIE_IDS)
                    _pids49 = [p for p in _re49.findall(
                        r'~>(\d{2,3})(?=<~)',
                        str(rec.get('raw', '') or ''))
                        if p in _fam49]
                    _frm49 = _em49.zone_code_for(
                        _sz49(str(rec.get('to', ''))))
                    _to49 = _em49.zone_code_for(
                        _sz49(str(rec.get('from', ''))))
                    if _pids49:
                        items.append(('line', render(
                            'move', {'nick': self.nick,
                                     'pieces': _pids49,
                                     'frm': _frm49, 'to': _to49},
                            local=False)))
                except Exception as _e49:
                    self.alarms.append(
                        'EARLY RECOVER reverse-drag failed: %r'
                        % (_e49,))
            else:
                items.extend(('line', render(
                    'undo', {'nick': self.nick}, local=False))
                    for _ in range(n13))
            self.findings.append(('early_recover_refused',
                                  'unfinished action' if unfinished else '%d unspent' % _left))
            return items
        # the turn is ending: whatever is still open resolves NOW or
        # expires LOUDLY, never a ride across the boundary
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
        # echo-ledger entries never claimed by a client mirror expire
        # at the boundary - LOUD rows, then a clean slate (a stale
        # entry could eat a NEXT turn's real gesture)
        for e in self.echo_ledger:
            self.findings.append(('mirror_echo_unclaimed',
                                  '%(side)s %(kind)s %(tkey)s x%(n)d '
                                  'at %(region)s' % e))
        self.echo_ledger = []
        self._post_commit_tail = ()
        self._human_casualty_prefix = None
        self._ai_casualty_prefix = None
        self._snapshot([rec.get('norm', '')], 'rollover')
        # Recovery concerns the dice that were in play in the ended turn;
        # new bonus dice are emitted separately after the reset.
        from .turn_bookkeeping import (capture_recovery_pools,
                                       restore_discarded_result_bindings)
        result_counts = {side.name: len(dice) for side, dice in self.es.dice.items()}
        recovery_pools = capture_recovery_pools(self)
        recovery_count = len(self.es.dice.get(self.ai_side, []))
        recovery_pids = sorted(str(pid) for pid, binding in self.view.die_id_map.items()
                               if binding[0] is self.ai_side and 0 <= binding[1] < recovery_count)
        # begin_turn_reset SETS state.turn to the passed number - no
        # increment afterwards
        _cor_pre = int(getattr(self.es.fellowship, 'corruption', 0)
                       or 0)
        ended = gl.begin_turn_reset(self.es, self.es.turn + 1,
                                    deal_cards=False,
                                    bonus_dice=False)
        # The boundary settles Mordor inactivity before military victory and
        # returns without resetting a terminal turn. Its real corruption delta
        # must reach the board even when that very delta ends the game.
        items = []
        _cor_post = int(getattr(self.es.fellowship, 'corruption', 0) or 0)
        if _cor_post != _cor_pre:
            import time as _time9
            from emitter import render as _crender
            items.append(('line', _crender(
                'move', {'nick': self.nick, 'pieces': ['252'],
                         'frm': 'Game.%d' % (2014 + _cor_pre),
                         'to': 'Game.%d' % (2014 + _cor_post)},
                local=False)))
            owed = getattr(self, '_track_owed', None)
            if owed is None:
                owed = self._track_owed = {}
            owed['252'] = (_cor_post, _time9.time() + 45.0)
            self.findings.append(('corruption_sweep_emitted',
                                  'rollover: %d -> %d' % (_cor_pre, _cor_post)))
        if ended:
            self.check_victory(end_of_turn=True)
            return items + [('game_over', dict(self.game_over))]
        self.phase = 'draws'
        self.turn_draws = {}
        self._turn_start_draws_open = False
        # J1 (A-11): the recover step is the ONE bonus-die granter (the
        # rollover passes bonus_dice=False - the RESET must not grant).
        # The grant runs HERE, on the LIVE state; the server renders
        # the AI's OWN new die pieces out of its reinforcement box
        # (('bonus_dice', pre_counts) item); the human drags their own
        # (forging human drags is forbidden).  calculate_bonus_dice
        # adjusts TO the roster size (idempotent; pinned).
        import game_state as gs
        pre_counts = {s.name: recovery_pools[s.name]['count']
                      for s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)}
        try:
            gl.calculate_bonus_dice(self.es)
        except Exception as e:
            self.alarms.append('J1 bonus-dice recalc FAILED: %r' % e)
        restore_discarded_result_bindings(self, recovery_pools, result_counts)
        # BL-39a: the item fires when EITHER side grew (a human-side
        # minion grants a die the client must be told about: announce
        # + family bind, else pid:null and a pool census split)
        if any(len(self.es.dice.get(_s39, []))
               > pre_counts.get(_s39.name, 0)
               for _s39 in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)):
            items.append(('bonus_dice', dict(pre_counts)))
        # FLD-10e (par5b_g5#20 GtW / g3#14a WK): the recalc SHRANK a
        # pool (a die-granting character died) - the loss side was
        # silent and unrendered: the client's own frecover re-rendered
        # the dead character's die and the next full-pool roll wedged
        # (engine=8 client=9 / phantom 5th die).
        # The old pool membership is retained in recover_ai below. The
        # bookkeeping scheduler returns these exact retired dice to their
        # Reinforcements before recovering the remaining pool; merely
        # killing them would let the native Recover Dice menu revive them.
        for _s10e in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            _post10e = len(self.es.dice.get(_s10e, []))
            _pre10e = pre_counts.get(_s10e.name, 0)
            if _post10e >= _pre10e:
                continue
            _gone10e = sorted(
                str(p) for p, si in
                (getattr(self.view, 'die_id_map', {}) or {}).items()
                if si and si[0] is _s10e and int(si[1]) >= _post10e)
            for _p10e in _gone10e:
                self.view.die_id_map.pop(_p10e, None)
            items.append(('chat', 'prompt',
                          chat_mod.die_removed_note(
                              _s10e.name, _pre10e - _post10e)))
            self.findings.append(
                ('die_removed_rendered',
                 '%s %d -> %d pids=%s'
                 % (_s10e.name, _pre10e, _post10e,
                    ' '.join(_gone10e) or '-')))
        items.append(('recover_ai', {'turn':int(self.es.turn),
                                     'pids':recovery_pids,'count':recovery_count,
                                     'pools':recovery_pools}))
        # BOTH human decks depleted: no draw line can ever arrive, so
        # the turn start hands off HERE (behind the AI's own recover)
        # instead of waiting at 'draws' forever
        if not getattr(self, 'spectator', False):
            _waived = self._depleted_deck_draw_waivers(self.human_side)
            if _waived is not None:
                items.extend(_waived)
                items.extend(self._ai_turn_start_draws())
        return items                    # live_server renders the drags

    # ---- windows -------------------------------------------------------
    def _on_window(self, w):
        if getattr(w, '_staged_die_pid', None) is not None:
            self._grouped_die_selection = None
        import ai
        import game_state as gs
        # our own emission echo: verify, never re-match
        stag = w.records[0] if w.records else {}
        if str(stag.get('player', '')) == self.nick:
            return []
        self._promote_reconcile()           # BRLANE-1: the window closed
        if getattr(self, '_staged_house_window', None) is w:
            self._staged_house_window = None
            # Only the player's physical cleanup remains. The play and
            # every later decision have their own existing commit entries.
            return []
        if getattr(self, '_staged_event_window', None) is w:
            pending = ai.describe_pending(self.es)
            if (pending and pending['kind'] == 'card_option'
                    and pending['side'] is self.human_side
                    and self.pending_buf is not None
                    and self.pending_buf.records):
                self._try_pending()
                result = getattr(self, '_last_pending_match', None)
                pending = ai.describe_pending(self.es)
                if (pending and pending['kind'] == 'card_option'
                        and self.pending_buf is not None
                        and getattr(result, 'status', None) == 'none'
                        and not getattr(result, 'starved', False)
                        and self.pending_random_call() is None
                        and self.deferred is None):
                    # Fixed-size card effects normally settle before cleanup.
                    # An explicit close with a completed NONE verdict must
                    # explain how to resume; it cannot silently discard the
                    # only physical completion evidence. Keep any supplied
                    # draw receipts on this same unresolved choice.
                    self.pending_buf._unmatched_card_cleanup = self._copy_card_window(w)
                    self._out_queue.append(('chat', 'refusal', chat_mod._register(
                        'The completed card effect does not match a legal choice. '
                        'Use native Undo to restore the card or die discard, then '
                        'return the incorrect pieces and perform the correct effect. '
                        'Already drawn cards remain reserved for this card effect.')))
            if self.commits and len(self.commits[-1]) > 5:
                completion = self.commits[-1][5]
                if isinstance(completion, dict) and completion.get('card_completion'):
                    known = set(self.commits[-1][1])
                    completion['cleanup'] = [r for r in w.records
                                             if r.get('op') == 'kill' and r.get('norm') not in known]
                    completion['event_window'] = self._copy_card_window(w)
            self._staged_event_window = None
            self.window_stats['matched'] += 1
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
        # still blocked is now attributable - retry once, then expire
        # visibly
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
        # a human window arriving while the AI's OWN multi-step die is
        # open (pending_army_move_die/muster) would be matched against
        # the SECOND-LEG-gated menu and mis-judged; the engine says it
        # is still the AI's turn - HOLD the window loudly instead of
        # matching against a poisoned menu
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
            self._tag_hold(w, 'ai_multistep')      # BRLANE-2: retryable
            self.holds.append((w, gesture_match.MatchResult(
                'hold', note='AI multi-step die still open - out of '
                             'turn')))
            self.alarms.append('WINDOW WHILE AI MULTI-STEP OPEN - '
                               'held (out of turn)')
            return out + [('note', 'window held: the AI is still '
                                   'resolving its die')]
        if self._human_out_of_turn() \
                and not getattr(self, 'spectator', False):
            # BL-70(1): the funnel's own awaited rolls may be riding
            # this about-to-be-refused window - EVERY d6 record feeds a
            # live pending FIRST (the client rolls ALL randomness under
            # the human's nick, so turn ownership is never roll
            # ownership)
            try:
                _p70 = self.rand.pending() if self.rand else None
            except Exception:
                _p70 = None
            if _p70 is not None \
                    and getattr(_p70, 'kind', '') == 'd6':
                for _r70 in [r for r in w.records
                             if r.get('op') == 'roll_num']:
                    self.rand.feed('d6', _r70.get('ints', []),
                                   side=self._d6_roll_side(_r70))
                    w.records.remove(_r70)
                    self.findings.append(
                        ('d6_fed_from_refused_window',
                         str(_r70.get('norm', ''))[:60]))
                if not w.records:
                    return out
            # rulebook p.20: the players ALTERNATE actions.  A second
            # human die window while the engine names the other side
            # (the AI's reply still in flight, or its pending open) is
            # REFUSED like the chat lanes refuse it (matching it would
            # re-seed last_actor and let the AI act twice in a row)
            w_norms = [r.get('norm', '') for r in w.records]
            self.uncommitted.extend(w_norms)
            self.window_stats['none'] += 1
            self.findings.append(('window_out_of_turn',
                                  'last_actor=%s :: %s'
                                  % (self.last_actor.name,
                                     (w_norms[1] if len(w_norms) > 1
                                      else w_norms)[:80])))
            res = gesture_match.MatchResult(
                'none', note=(self._TURN_OVER_DETAIL
                              if self._turn_over_pending()
                              else 'it is not your action turn'))
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
            self._tag_hold(w, 'pending')           # BRLANE-2: retryable
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
                # (_unreveal)
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

    # ---- rng-fed matching + deferral ------------------------------------
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
            if rec.get('_window_rng_fed'):
                continue
            op = rec.get('op')
            if op == 'roll_num':
                try:
                    self._out_queue.extend(self._on_phase(rec))
                except Exception as e:
                    self.findings.append(('rng_d6_feed', repr(e)))
            elif op == 'move':
                frm = str(rec.get('from', '')).lower()
                if 'hunt pool' in frm:
                    try:
                        self.rand.feed('tile', str(rec.get('obj', '')))
                        rec['_window_rng_fed'] = True
                    except Exception as e:
                        self.findings.append(('rng_tile_feed',
                                              repr(e)))
            elif op == 'draw':
                self._feed_draw_rec(rec)
                rec['_window_rng_fed'] = True

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
            # swap: one fictional REAL cid out, the placeholder in -
            # the engine draw then removes the placeholder (deck -1).
            # Never pop a placeholder an earlier draw of the same
            # gesture already swapped in (two draws in one play).
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
            from .human_draw_identity import slot
            slot(hm, deck, ph, rec)
            try:
                self.rand.feed('card', (side.name, deck, ph))
            except Exception as e:
                self.findings.append(('rng_card_feed', repr(e)))
            return
        if rec.get('_native_companion_names'):
            self.rand.feed('companion', rec['_native_companion_names'])
            return
        pp = rr.parse_piece(what)
        if pp is not None and pp[0] == 'tile':
            try:
                self.rand.feed('tile', what)
            except Exception as e:
                self.findings.append(('rng_tile_feed', repr(e)))
            # A-06 / corpus A7: the tile-face announce after an
            # AI-side draw - the AUTHORITATIVE face correction
            # ('1r'/'eye'; plain damage numbers excluded by the
            # grammar, they collide with hit counts).  Banked for the
            # caller (only the live phase-draw path emits;
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
        if self.forced_ai_rows_pending():
            return False
        import ai
        if (getattr(self, '_combat_recovery_active', False)
                and not getattr(self, '_resync_commit_transaction', False)):
            accepted, reason = self._run_resync_transaction(
                lambda: self._commit_chain(chain, w_norms, label, gesture=gesture))
            if not accepted:
                self._combat_recovery_failed = True
                self.findings.append(('resync_commit_rolled_back', str(reason)))
                self.alarms.append('recovery commit retained its original state: %s' % reason)
            return accepted
        press_count = sum(getattr(a, 'action_type', '') == 'siege_press' for a in chain)
        if (not getattr(self, 'spectator', False) and press_count
                and (press_count != 1 or self._siege_press_key() is None
                     or self._siege_press_waiting())):
            self._out_queue.extend(self._siege_press_reminder())
            return False
        from .temporary_combat_figures import prepare,register
        try:
            temporary_figures=prepare(self.view,self.es,self.human_side,gesture,chain)
        except ValueError as exc:
            self.alarms.append(str(exc))
            return False
        if any(getattr(a, 'action_type', None) == 'elven_ring' for a in chain):
            # 'changes' is collateral of the prior Java snapshot; it
            # never appears as the target of a separate native undo.
            w_norms = [n for n in w_norms if live_parser.parse(n).get('op') != 'die_change']
        recovery_rng = None
        if getattr(self, '_combat_recovery_active', False):
            from .combat_recovery import randomness_snapshot
            recovery_rng = randomness_snapshot(self.rctl)
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
            if recovery_rng is not None:
                from .combat_recovery import restore_randomness
                restore_randomness(self.rctl, recovery_rng)
                self._combat_recovery_failed = True
            else:
                self.rand.reset_feed()
            return False
        completion = None
        if getattr(gesture, '_staged_card_closed', False) and not getattr(self, 'spectator', False):
            from .card_completion import needs_boundary
            if needs_boundary(pre):
                close = getattr(gesture, '_staged_card_close_record', None)
                close_norm = (close or {}).get('norm')
                if close_norm and close_norm not in w_norms:
                    w_norms = list(w_norms) + [close_norm]
                completion = {'card_completion': True, 'buffer': self._copy_card_window(gesture),
                              'event_window': self._copy_card_window(
                                  getattr(gesture, '_staged_card_event_window', None)),
                              'cleanup': [], 'reversible': not any(self.rand._feed_pos.values())}
        self._snapshot(w_norms, label, post=post)
        if completion is not None:
            self.commits[-1] += (completion,)
        completion_output = len(self._out_queue)
        self.es = post
        register(self.view,temporary_figures)
        if any(getattr(a, 'action_type', '') == 'siege_press' for a in chain):
            self._siege_press_intent = None
        for hm in self.hand_models.values():
            hm.es = post
        # BL-82: the SAME marker writer as the chat lane (physical=
        # True: this commit carries client lines, so the no-client-
        # line demands are not owed - but the K1 flag must still
        # track hidden commits/reveals committed by gesture)
        for _a82 in chain:
            self._post_commit_markers(_a82, physical=True)
        self._paid_noop_check(pre, post, label, chain)
        # BL-150 D2 (2026-08-28): a discard_card leg the human never
        # performed (no hand-zone kill in the gesture) is the engine
        # inventing which card left the hand - LOUD, never silent
        _disc150 = [a for a in chain
                    if str(getattr(a, 'action_type', '')) == 'discard_card']
        if _disc150:
            _voiced150 = False
            try:
                _f150 = gesture_match.window_facts(gesture) \
                    if gesture is not None else None
                _voiced150 = bool(_f150 is not None
                                  and (_f150.hand_discards
                                       or _f150.cards_played))
            except Exception:
                _voiced150 = True          # unreadable: do not cry wolf
            if not _voiced150:
                _cids150 = [str((getattr(a, 'params', None) or {})
                                .get('card_id', '?'))[:50] for a in _disc150]
                self.findings.append(('unvoiced_discard_leg',
                                      '%s :: %s' % (label, _cids150)))
                self.alarms.append(
                    'UNVOICED DISCARD: the committed chain discarded %s '
                    'but your gesture removed no card from your hand - '
                    'the engine hand may now disagree with yours; '
                    'RESYNC if a card you hold becomes unplayable'
                    % ', '.join(_cids150))
        if label in ('pending:declare', 'declare-decline'):
            # this turn's declaration opportunity is spent (K2)
            self._declare_done_turn = getattr(post, 'turn', None)
        # G2: a combat round resolved inside this chain announces its
        # hit counts (engine statement)
        from . import emission as em_mod
        self._out_queue.extend(
            ('chat', k, t) for k, t in
            em_mod.round_hits_chat(post, self.human_side, pre=pre))
        # engine events the gesture did not express - forced
        # allocations of the AI's own units + settlement-capture
        # markers (live-only; no-op for the spectator)
        self._out_queue.extend(
            self.post_commit_sweep(pre, post, gesture=gesture,
                                   label=label, actions=chain))
        # SQF-5 (seqack_fp1 finding 5) + FLA-13 (par5 g2#6): a
        # card_option OR hunt decision answered PHYSICALLY (menu act /
        # board gesture) is voiced exactly like the chat pick lane -
        # every pick commit gets its voice line (the hunt
        # guide-sacrifice committed with no voice and the tester
        # re-asked)
        if str(label).startswith(('pending:card_option',
                                  'pending:hunt')) \
                and not getattr(self, 'spectator', False):
            _pk5 = next((a for a in chain
                         if str(getattr(a, 'action_type', ''))
                         in ('card_option', 'hunt_take_corruption',
                             'hunt_sacrifice_guide',
                             'hunt_sacrifice_random',
                             'hunt_guide_separate',
                             'hunt_gollum_reveal',
                             'hunt_gollum_no_reveal')), None)
            if _pk5 is not None:
                self._out_queue.append(
                    ('chat', 'prompt', chat_mod.pick_committed_note(
                        self._pick_desc(_pk5))))
        # USER 2026-08-24: after every committed gesture, monitor and
        # fix divergences.  FLAG ONLY here (adopting at the commit
        # instant reads a dump the client has not yet caught up to) -
        # the server reconciles on the next QUIET tick.
        self._reconcile_due = True
        # a commit that leaves a chat-only human pending open says so
        self._out_queue.extend(self._maybe_announce_pending())
        # BRLANE-2: a commit may have cleared what a held window waited on
        if completion is not None:
            completion['reversible'] = (completion['reversible']
                and not any(item[0] == 'line' for item in self._out_queue[completion_output:]))
        self._out_queue.extend(self._retry_holds('commit'))
        exc = self.rand.unconsumed()
        self.rand.step_commit()
        if any(exc.values()):
            self.findings.append(('unconsumed_rng',
                                  '%r after %s' % (
                                      {k: v for k, v in exc.items()
                                       if v}, label)))
            if getattr(self, 'ai_step_open', False):
                # same guard as the chat gate: never quarantine
                # randomness an open AI step is still owed
                self.findings.append(
                    ('unconsumed_rng_kept_for_open_step', label))
            else:
                self.rand.reset_feed()
        return True

    # ---- post-commit engine-event sweep ---------------------------------
    def _auto_wiped_figure_cleanup(self, expected_state=None):
        """Remove compulsory battle figures after the human shows unit losses."""
        from .combat_cleanup import unit_losses_visible
        from . import emission as _cleanup_emission
        prefix = getattr(self, '_human_casualty_prefix', None)
        if expected_state is None:
            expected_state = prefix[1] if prefix is not None and prefix[0] is self.es else self.es
        grouped = {}
        for row in self.echo_ledger:
            if (row.get('automatic_army_wipe') and not row.get('automatic_cleanup_sent')
                    and int(row.get('n', 0)) > 0):
                grouped.setdefault(row['region'], []).append(row)
        out = []
        for region, entries in grouped.items():
            if not unit_losses_visible(self, region, expected_state):
                continue
            rows = {'units': {}, 'naz': {'minus': [], 'plus': []},
                    'named': {'leaves': {}, 'joins': {}}, 'cas': {}, 'unexplained': []}
            for entry in entries:
                count = int(entry['n'])
                if entry['kind'] == 'unit':
                    rows['units'].setdefault(entry['tkey'], {'minus': [], 'plus': []})['minus'].append((region, count))
                    rows['cas'][entry['tkey']] = rows['cas'].get(entry['tkey'], 0) + int(entry.get('automatic_casualties', 0))
                elif entry['kind'] == 'naz':
                    rows['naz']['minus'].append((region, count))
                elif entry['kind'] == 'named':
                    rows['named']['leaves'].setdefault(entry['tkey'], []).extend([region] * count)
            emitted, _ = _cleanup_emission.render_forced_rows(
                self.view, self.nick, self.human_side, rows, expected_state,
                label='compulsory battle figure cleanup', post=expected_state)
            if emitted.hold:
                finding = ('forced_rows_hold', str(emitted.hold)[:200])
                if finding not in self.findings:
                    self.findings.append(finding)
                continue
            for entry in entries:
                entry['automatic_cleanup_sent'] = True
            self.expect_echo(emitted.lines)
            out.extend(('line', line) for line in emitted.lines)
        return out

    @staticmethod
    def _subtract_accounted_observation(rows, accounted):
        """Subtract only matching physical observations, never invent inverses."""
        from . import emission as em_mod
        if rows is None:
            return None
        result = em_mod.subtract_rows(rows, accounted)
        for key, slot in result['units'].items():
            for direction in ('minus', 'plus'):
                available = dict(rows.get('units', {}).get(key, {}).get(direction, ()))
                slot[direction] = [(region, min(count, available.get(region, 0)))
                                   for region, count in slot[direction] if available.get(region, 0) > 0]
        return result

    def _stage_ai_casualty_prefix(self, stopped, expressed=None):
        """Display exact real pre-draw AI casualties, never movement/capture."""
        from . import emission as em_mod
        import copy
        tag = 'shadow' if self.ai_side.name == 'SHADOW' else 'fp'
        prefix = getattr(self, '_ai_casualty_prefix', None)
        if prefix is not None and prefix[0] is not self.es:
            raise DriverHalt('Stopped AI casualty prefix belongs to another transaction')
        batches, own_ops, previous_external = prefix[1:] if prefix is not None else ((), (), ())
        # External d6-prefix operations were already displayed by the existing
        # chosen/proved casualty path. Remember them across later draw polls.
        from collections import Counter
        import json
        op_key = lambda op: json.dumps(op, sort_keys=True, default=str)
        external = list(previous_external)
        previous_counts = Counter(op_key(op) for op in previous_external)
        incoming_counts = Counter()
        for op in (expressed or ()):
            key = op_key(op)
            incoming_counts[key] += 1
            if incoming_counts[key] > previous_counts[key]:
                external.append(op)
        # Generic prefix emissions are dispatched as Driver items, not added
        # to those external lists. Keep their separate ownership: identical
        # kill operations can recur after the same fungible PID is replaced.
        rows = em_mod.removal_rows(self.es, stopped, tag)
        for batch in batches:
            rows = em_mod.subtract_rows(rows, batch)
        if external:
            observed, failed = em_mod.expressed_rows(self.view, external, tag)
            if failed:
                raise DriverHalt('Cannot identify prior actual AI casualty operations')
            rows = em_mod.subtract_rows(rows, observed)
        rows = copy.deepcopy(rows)
        for key in list(rows['units']):
            slot = rows['units'][key]
            if slot['minus'] and slot['plus']:
                del rows['units'][key]
            elif slot['plus']:
                elite = key.rsplit(':', 1)[0] + ':elite'
                if not (key.endswith(':regular') and rows['units'].get(elite, {}).get('minus')):
                    del rows['units'][key]
        if rows['naz']['plus']:
            rows['naz'] = {'minus': [], 'plus': []}
        for name in list(rows['named']['joins']):
            rows['named']['leaves'].pop(name, None)
        rows['named']['joins'] = {}
        if em_mod.rows_empty(rows):
            self._ai_casualty_prefix = (self.es, batches, own_ops, tuple(external))
            return []
        if getattr(self, 'echo_expect', None):
            return []
        emitted, ledger = em_mod.render_forced_rows(self.view, self.nick, self.ai_side,
            rows, self.es, label='stopped pre-draw AI casualties', post=stopped)
        if emitted.hold:
            raise DriverHalt('Stopped AI casualty prefix held: %s' % emitted.hold)
        self._ai_casualty_prefix = (self.es, batches + (rows,), own_ops + tuple(emitted.ops), tuple(external))
        self.echo_ledger.extend(ledger)
        self.expect_echo(emitted.lines)
        return [('line', line) for line in emitted.lines]

    def _stage_human_casualty_prefix(self, stopped, gesture=None):
        """Bank only proved real stopped-prefix losses; authoritative ES stays put.

        Callers must supply a stopped *actual* transaction, never a search
        simulation. Repeated callbacks share the exact base ES and account
        for previously staged losses instead of charging them again.
        """
        from . import emission as em_mod
        tag = 'fp' if self.human_side.name == 'FREE_PEOPLES' else 'shadow'
        prefix = getattr(self, '_human_casualty_prefix', None)
        if prefix is not None and prefix[0] is not self.es:
            raise DriverHalt('Stopped human casualty prefix belongs to another transaction')
        previous = prefix[1] if prefix is not None else self.es
        rows = em_mod.removal_rows(previous, stopped, tag)
        if gesture is not None:
            observed, blocked = self._gesture_rows(gesture, tag)
            if blocked:
                raise DriverHalt('Stopped human casualty prefix has unidentifiable physical evidence')
            if prefix is not None and observed is not None:
                accounted = em_mod.removal_rows(self.es, previous, tag)
                observed = self._subtract_accounted_observation(observed, accounted)
            rows = em_mod.subtract_rows(rows, observed)
        items = self._bank_human_physical_rows(previous, stopped, rows,
                                               label='stopped combat prefix')
        self._human_casualty_prefix = (self.es, stopped)
        return items

    def post_commit_sweep(self, pre, post, gesture=None, expressed=None,
                          label='', rebase_exempt=False, track_sources=None,
                          actions=None):
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
        from .crebain_gestures import observed_ai_progress
        for action in actions or ():
            crebain_progress = observed_ai_progress(self, pre, action, post)
            if crebain_progress is not None:
                expressed = list(expressed or ()) + crebain_progress
                break
        from .fellowship_reveal_gestures import observed_progress_reset
        reveal_progress = observed_progress_reset(self, pre, post, actions)
        if reveal_progress is not None:
            expressed = list(expressed or ()) + reveal_progress
        items = []
        if (self.phase == 'declare' and getattr(pre, '_declare_phase', False)
                and any(a.action_type in ('fellowship_declare','pass','declare_decline')
                        for a in (actions or ()))):
            # Keep declaration-triggered card choices inside the same tail.
            # Otherwise a later fp-ok could trigger stationary Balrog again.
            items.extend(self._open_fellowship_guide('declaration committed'))
        if self.phase == 'fellowship_tail':
            items.extend(self._resume_fellowship_tail())
            # AI-FP's declaration step owns its final phase-close message.
            # A human FP can proceed automatically when there is no choice.
            import game_state as gs
            if self.ai_side is gs.Side.SHADOW:
                items.extend(self._complete_fellowship_guide())
        # BL-55: the engine stamps a failed siege move-in - surface
        # it as an ALARM (the silent pass hid a no-siege state)
        _smf55 = getattr(post, '_siege_movein_failed', None)
        if _smf55:
            post._siege_movein_failed = None
            self.alarms.append('SIEGE MOVE-IN FAILED engine-side: %s '
                               '- NO siege established' % _smf55)
        # BL-136(i): the engine's siege move-in stamp (consumed once)
        _smi136 = getattr(post, '_siege_movein', None)
        if _smi136:
            post._siege_movein = None
        # BL-136(ii): every region this commit touched is a PAID
        # engine delta with a birth time - the reconcile eraser
        # witness reads it (an adoption that removes engine pieces
        # from a region touched seconds ago erased a paid delta)
        try:
            import time as _t136
            from board import sanitize as _san136
            _rd136 = getattr(self, '_recent_delta136', None)
            if _rd136 is None:
                _rd136 = self._recent_delta136 = {}
            for _reg136 in (client_board.board_delta(pre, post) or {}):
                if not str(_reg136).startswith('__'):
                    _rd136[_san136(str(_reg136))] = _t136.time()
        except Exception:
            pass
        # A new battle retires satisfied echoes but cannot erase an older
        # card still physically owed (fpgame15: Woodland reveal at Lorien).
        if getattr(pre, 'pending_combat', None) is None \
                and getattr(post, 'pending_combat', None) is not None:
            previous = self._cc_battle_serial
            self._cc_battle_serial += 1
            origins = getattr(self, '_cc_reveal_origins', {}) or {}
            self._cc_reveal_owed = {
                cid: status for cid, status in
                (getattr(self, '_cc_reveal_owed', {}) or {}).items()
                if status == 'owed'}
            self._cc_reveal_origins = {
                cid: origins.get(cid, previous) for cid in self._cc_reveal_owed}
            self._cc_choose_preamble = None
        ai_tag = 'shadow' if self.ai_side.name == 'SHADOW' else 'fp'
        hu_tag = 'fp' if ai_tag == 'shadow' else 'shadow'
        prefix = getattr(self, '_human_casualty_prefix', None)
        ai_prefix = getattr(self, '_ai_casualty_prefix', None)
        if any(item is not None and item[0] is not pre for item in (prefix, ai_prefix)):
            raise DriverHalt('Committed casualty prefix belongs to another transaction')
        try:
            rows_ai = em_mod.removal_rows(pre, post, ai_tag)
            human_pre = prefix[1] if prefix is not None else pre
            rows_hu = em_mod.removal_rows(human_pre, post, hu_tag)
            staged_human = em_mod.removal_rows(pre, human_pre, hu_tag) if prefix is not None else None
            # A proven transaction may already have displayed compulsory AI
            # losses before a post-casualty extra attack requested its dice.
            prior_ops = getattr(gesture, '_pre_rng_expressed', ()) if gesture is not None else ()
            if prior_ops:
                prior_rows, failed = em_mod.expressed_rows(self.view, prior_ops, ai_tag)
                if failed:
                    raise RuntimeError('Previously emitted pre-roll casualty identity lost')
                rows_ai = em_mod.subtract_rows(rows_ai, prior_rows)
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
            if ai_prefix is not None:
                for staged_ai in ai_prefix[1]:
                    # These operations belong only to the generic stopped
                    # prefix; external chosen/proved operations in sub_ai are
                    # separate payments, even when their unit types match.
                    rows_ai = em_mod.subtract_rows(rows_ai, staged_ai)
            res_ai = em_mod.subtract_rows(rows_ai, sub_ai)
            if staged_human is not None and sub_hu is not None:
                sub_hu = self._subtract_accounted_observation(sub_hu, staged_human)
            res_hu = em_mod.subtract_rows(rows_hu, sub_hu)
            self._human_casualty_prefix = None
            self._ai_casualty_prefix = None
            items.extend(self._bank_human_physical_rows(
                pre, post, res_hu, label=label, rebase_exempt=rebase_exempt,
                siege_movein=_smi136))
            # A same-commit relocation is continuation, not an ordinary loss.
            # Hold it with the marker tail until the human has paid, while
            # still allowing the AI's ordinary casualty cleanup immediately.
            relocation = (any(slot.get('minus') and slot.get('plus')
                              for slot in res_ai['units'].values())
                          or bool(res_ai['naz']['minus'] and res_ai['naz']['plus'])
                          or bool(set(res_ai['named']['leaves']) & set(res_ai['named']['joins'])))
            if not (relocation and self.human_physical_work_pending()):
                items.extend(self._render_post_commit_ai_rows(pre, post, res_ai, blocked, label))
                res_ai = None
        else:
            res_ai, blocked = None, False
        # Unit payments are engine-committed now, but their consequences must
        # not appear ahead of the human's physical removals/replacements. Keep
        # data, not a closure or pre-rendered marker lines: release uses the
        # current client view and only then books its own positive ACKs.
        # Marker-only recovery evidence must not replace the gesture's unit
        # observations in the casualty accounting above.
        from .choice_recovery import observed_tracks
        from .combat_recovery import observed_political_steps
        recovered_tracks = observed_tracks(self, pre, post, actions, gesture)
        recovered_tracks += observed_political_steps(self, pre, post, actions, gesture)
        if recovered_tracks:
            expressed = list(expressed or ()) + recovered_tracks
        payload = (pre, post, gesture, expressed, label, actions, track_sources, res_ai, blocked)
        if self.human_physical_work_pending() or getattr(self, '_post_commit_tail', ()):
            context = (getattr(post, 'turn', None), self._cc_battle_serial,
                       getattr(self, '_resync_ts', None))
            self._post_commit_tail = tuple(getattr(self, '_post_commit_tail', ())) + ((context, payload),)
        else:
            items.extend(self._post_commit_consequences(*payload))
        self.expect_echo([item[1] for item in items if item[0] == 'line'])
        return items

    def _flush_post_commit_consequences(self):
        if self.human_physical_work_pending() or getattr(self, 'echo_expect', None):
            return []
        queued = tuple(getattr(self, '_post_commit_tail', ()))
        if not queued:
            return []
        context, payload = queued[0]
        current = (getattr(self.es, 'turn', None), self._cc_battle_serial,
                   getattr(self, '_resync_ts', None))
        if context != current:
            message = 'Deferred physical consequences belong to another battle/turn; reset required'
            if message not in self.alarms:
                self.alarms.append(message)
            return []
        items = self._post_commit_consequences(*payload)
        self._post_commit_tail = queued[1:]
        self.expect_echo([item[1] for item in items if item[0] == 'line'])
        return items

    def _render_post_commit_ai_rows(self, pre, post, res_ai, blocked, label):
        from . import emission as em_mod
        items = []
        if not em_mod.rows_empty(res_ai):
            if blocked:
                items.extend(self._retain_forced_ai_rows(pre, post, res_ai, True,
                    label, 'expressed-row decode failed; existing output must not be repeated'))
            else:
                try:
                    em, ledger = em_mod.render_forced_rows(
                        self.view, self.nick, self.ai_side, res_ai,
                        pre, label=label, post=post)
                except Exception as error:
                    return self._retain_forced_ai_rows(pre, post, res_ai, False,
                        label, '%s: %s' % (type(error).__name__, error))
                if em.hold:
                    items.extend(self._retain_forced_ai_rows(
                        pre, post, res_ai, False, label, em.hold))
                else:
                    items.extend(('line', l) for l in em.lines)
                    for row in ledger:
                        self.echo_ledger.append(row)
                    if em.lines:
                        self.findings.append(
                            ('forced_rows_emitted',
                             '%s: %d line(s)' % (label,
                                                 len(em.lines))))
        return items

    def _retain_forced_ai_rows(self, pre, post, rows, blocked, label, reason):
        """Retain the exact failed tail without losing already emitted output."""
        import copy
        held = tuple(getattr(self, '_forced_ai_rows_holds', ()))
        if any(entry['pre'] is pre and entry['post'] is post
               and entry['label'] == label for entry in held):
            return []
        alarm = 'FORCED AI ROWS HOLD (%s): %s' % (label, reason)
        self._forced_ai_rows_holds = held + ({
            'pre': pre, 'post': post, 'rows': copy.deepcopy(rows),
            'blocked': bool(blocked), 'label': label, 'reason': str(reason),
            'alarm': alarm,
            'context': (self.es.turn, self._cc_battle_serial, self.n_commits_total)},)
        if alarm not in self.alarms:
            self.alarms.append(alarm)
        self.findings.append(('forced_rows_hold', str(reason)[:200]))
        return [] if held else [('chat', 'prompt', chat_mod._register(self.forced_ai_rows_wait()))]

    def _recover_forced_ai_rows(self, physical, evidence):
        """Cold recovery of retained output, never another engine decision.

        A complete current census may prove the work already performed. Else
        retry the exact retained rows on a copied native view, and publish only
        if their projected result matches the already committed public state.
        """
        import copy
        import ai
        from . import emission
        from .repair_evidence import same_dump
        held = tuple(self._forced_ai_rows_holds)
        markers = self._client_marker_states()

        def differences(dump):
            rows = client_board.diff_engine_vs_dump(self.es, dump,
                dynamic_types=self.view.dynamic_types, marker_states=markers, ai_module=ai)
            return None if rows is None else [r for r in rows if not client_board.is_staging_row(r)]

        context = (self.es.turn, self._cc_battle_serial, self.n_commits_total)
        if any(entry['context'] != context for entry in held):
            return [], ['retained consequence belongs to another commit; paid state retained for correction']
        remaining = differences(physical)
        published = {line for entry in held for line in entry.get('published_lines', ())}
        unrelated = [pid for pid, entry in self.echo_expect.items()
                     if str(entry.get('line', '')).strip() not in published]
        if remaining == [] and not unrelated and same_dump(self, evidence):
            # The full census proves the already committed piece result,
            # including fungible replacements made manually after refusal.
            # Only output owned by this retained tail receives that credit;
            # unrelated card/die expectations remain untouched.
            self._release_echo(list(self.echo_expect), 'retained correction census')
            self._clear_forced_ai_rows(held, 'committed correction proved by complete native census')
            self._last_resync_adoption = {
                'accepted': True, 'complete': True, 'retained_combat': True,
                'waiting_for': 'combat_step'}
            return [], ['committed board consequences are already physically complete; paid action retained']
        if self.echo_expect or published:
            return [], ['committed output still awaits native acknowledgement or complete board correction; retained correction not replayed']
        if any(entry['blocked'] for entry in held):
            return [], ['retained consequence cannot be safely rendered; correct the board to the committed state, then RESYNC']
        scratch = copy.copy(self.view)
        scratch.piece_region = dict(physical)
        ops, lines, ledger = [], [], []
        for entry in held:
            rendered, rows = emission.render_forced_rows(scratch, self.nick,
                self.ai_side, entry['rows'], entry['pre'], label=entry['label'], post=entry['post'])
            if rendered.hold:
                return [], ['committed board correction remains held: %s' % rendered.hold]
            ops.extend(rendered.ops)
            lines.extend(rendered.lines)
            ledger.extend(rows)
        projected = differences(scratch.piece_region)
        if projected is None or projected:
            return [], ['retained correction would not reconcile the complete board; no command emitted']
        if not same_dump(self, evidence):
            return [], ['board or feed changed during retained correction; no command emitted']
        def publish():
            emission._retrack_ops(self.view, ops)
            for row in ledger:
                self._echo_seq = getattr(self, '_echo_seq', 0) + 1
                row['echo_id'] = self._echo_seq
            self.echo_ledger.extend(ledger)
            self.expect_echo(lines)
            # Publishing is not physical completion. Retain the exact tail
            # through partial/refused delivery, preventing new battle choices
            # and preventing a retry from repeating an already applied kill.
            for entry in held:
                entry['published_lines'] = tuple(line.strip() for line in lines)
                entry['published_ledger'] = tuple(ledger)
                entry['confirmed_lines'] = set()
            return True
        accepted, reason = self._run_resync_transaction(publish)
        if not accepted:
            return [], ['retained correction publication failed: %s; obligation preserved' % reason]
        return [('line', line) for line in lines], [
            'retained committed correction emitted once; waiting for its native acknowledgement and board update']

    def _bank_human_physical_rows(self, pre, post, res_hu, label='',
                                  rebase_exempt=False, siege_movein=None):
        """Bank exact real human piece debts, without rendering consequences."""
        hu_tag = 'fp' if self.human_side.name == 'FREE_PEOPLES' else 'shadow'
        _smi136 = siege_movein
        items = []
        # Human unit casualties remain physical choices. Figures destroyed
        # with an army are compulsory follow-through, emitted only after
        # those unit losses are visible. A pure RELOCATION
        # (minus offset by an equal plus elsewhere, e.g. a leader
        # riding an army move the generic-leader subtraction
        # cannot attribute) is NOT a removal and never sits on the
        # ledger
        _banked43 = []
        _banked43_rows = []
        from .combat_cleanup import wiped_regions
        _wiped_regions = wiped_regions(pre, post, self.human_side)

        def _stamp63(row):
            # BL-63: identity-keyed banking - the seat performs
            # each entry ONCE by echo_id (fungible-piece re-kills
            # are the rolled-pids race with kills)
            self._echo_seq = getattr(self, '_echo_seq', 0) + 1
            row['echo_id'] = self._echo_seq
            from board import sanitize as _wipe_sanitize
            figure = (row.get('kind') in ('named', 'naz')
                      or (row.get('kind') == 'unit'
                          and str(row.get('tkey', '')).endswith(':leader')))
            if figure and _wipe_sanitize(str(row.get('region', ''))) in _wiped_regions:
                row['automatic_army_wipe'] = True
                if row.get('kind') == 'unit':
                    row['automatic_casualties'] = min(int(row.get('n', 0)),
                        int((res_hu.get('cas') or {}).get(row.get('tkey'), 0)))
            else:
                _banked43.append(str(row.get('kind', '')))
                _banked43_rows.append(dict(row))
            self.echo_ledger.append(row)

        # BL-136(i): a HUMAN attacker's siege move-in is a
        # relocation committed inside the DEFENDER's (AI) apply -
        # the engine's stamp opens the relocation lane for this
        # sweep (without it no demand named the move and
        # auto-reconcile adopted the un-moved client army back OUT
        # of the stronghold)
        _siege136 = bool(
            _smi136 and str(_smi136.get('side', '')).upper()
            .startswith('F' if hu_tag == 'fp' else 'S'))
        _frm136 = str((_smi136 or {}).get('from', ''))
        _to136 = str((_smi136 or {}).get('to', ''))

        def _bank_net(slot, kind, tkey):
            net = (sum(n for _, n in slot['minus'])
                   - sum(n for _, n in slot['plus']))
            # F2 (the identity lane): a pure RELOCATION committed
            # by pick/chat has NO physical gesture - bank the
            # paired MOVE for the seat to perform
            if (str(label).startswith('chat:') or _siege136) \
                    and slot['minus'] and slot['plus']:
                _mv8 = min(sum(n for _, n in slot['minus']),
                           sum(n for _, n in slot['plus']))
                if _mv8 > 0:
                    _row77 = {'kind': kind + '_move',
                              'side': hu_tag,
                              'region': slot['minus'][0][0],
                              'to_region': slot['plus'][0][0],
                              'tkey': tkey, 'n': _mv8}
                    # BL-77: bank only CLIENT-OWED deltas.  A row
                    # whose destination the client board already
                    # satisfies is the engine adopting the
                    # client's OWN drag (banked, it can never
                    # close and freezes reconcile).  A combined
                    # delta (uncommitted approach folded into the
                    # chat commit) names the PRE-approach source:
                    # rebase the demand to where the client pieces
                    # actually stand.
                    _f77 = self._move_row_fate(_row77)
                    if _f77 == 'satisfied':
                        self.findings.append(
                            ('echo_move_self_closed',
                             '%s %s->%s x%d' % (
                                 tkey, _row77['region'],
                                 _row77['to_region'], _mv8)))
                    else:
                        if isinstance(_f77, str) \
                                and _f77.startswith('rebase:'):
                            if rebase_exempt:
                                # BL-137: a demand born of an
                                # EXPLICIT decision (a card_option
                                # naming its regions) is never
                                # re-routed to where the client
                                # happens to hold pieces - bank
                                # as written; disagreement = hold
                                # + witness, never reroute
                                self.findings.append(
                                    ('echo_move_rebase_exempt',
                                     '%s: %s stays (client holds '
                                     'pieces at %s)'
                                     % (tkey, _row77['region'],
                                        _f77[7:])))
                            else:
                                self.findings.append(
                                    ('echo_move_rebased',
                                     '%s: %s -> %s (client facts)'
                                     % (tkey, _row77['region'],
                                        _f77[7:])))
                                _row77['region'] = _f77[7:]
                        _stamp63(_row77)
                        items.append(
                            ('chat', 'prompt',
                             chat_mod.echo_move_note(
                                 _row77['region'],
                                 _row77['to_region'])))
            left = max(0, net)
            for reg, n in slot['minus']:
                if left <= 0:
                    break
                take = min(n, left)
                left -= take
                _stamp63({'kind': kind, 'side': hu_tag,
                          'region': reg, 'tkey': tkey,
                          'n': take})
            # NET ARRIVALS bank too (kind '<k>_in'): the
            # forced-downgrade replacement regular the client board
            # must receive (pinned:
            # test_forced_downgrade_arrival_echo).  A pure
            # relocation (minus == plus) still banks NOTHING on
            # either side.
            left_in = max(0, -net)
            for reg, n in slot['plus']:
                if left_in <= 0:
                    break
                take = min(n, left_in)
                left_in -= take
                # BL-101: the dest-satisfied self-close covers
                # ARRIVALS too - a human who already dragged their
                # own recruits before the sweep must not get a
                # blind re-perform (a double recruit)
                _clA = None
                _enA = self._engine_tkey_count(post, tkey, reg)
                if _enA is not None:
                    try:
                        _cA = self._client_tkey_census(tkey)
                        _clA = (None if _cA is None
                                else _cA.get(reg, 0))
                    except Exception:
                        _clA = None
                if _clA is not None and _clA >= _enA:
                    self.findings.append(
                        ('echo_in_self_closed',
                         '%s %s x%d' % (tkey, reg, take)))
                    continue
                _rowI = {'kind': kind + '_in', 'side': hu_tag,
                         'region': reg, 'tkey': tkey, 'n': take}
                # FLC H2 (par5 g1#11/#9): the engine's downgrade
                # source stamp rides the row - a 'cas' arrival is
                # performed from the CASUALTY box, never refused
                # against an empty pool
                _srcsI = _dg_new9.get(
                    str(tkey).rsplit(':', 1)[0], None)
                if _srcsI and str(tkey).endswith(':regular'):
                    _rowI['src'] = _srcsI.pop(0)
                _stamp63(_rowI)
        _dg_new9 = {}
        try:
            _n0dg = len(getattr(pre, '_dg_src_log', None) or [])
            for _natD, _srcD in (getattr(post, '_dg_src_log',
                                         None) or [])[_n0dg:]:
                _dg_new9.setdefault(
                    str(getattr(_natD, 'value', _natD)).lower(),
                    []).append(str(_srcD))
        except Exception:
            _dg_new9 = {}
        for tkey, slot in res_hu['units'].items():
            _bank_net(slot, 'unit', tkey)
        _bank_net(res_hu['naz'], 'naz', 'nazgul')
        _post_fel9 = set()
        try:
            _post_fel9 = {str(getattr(c, 'name', c)) for c in
                          (getattr(post.fellowship, 'companions',
                                   ()) or ())}
            _g9f = getattr(getattr(post.fellowship, 'guide',
                                   None), 'name', None)
            if _g9f:
                _post_fel9.add(str(_g9f))
        except Exception:
            _post_fel9 = set()
        from .emission_card import _named_identity
        _post_fel9 = {_named_identity(name) for name in _post_fel9}
        for nm in sorted(set(res_hu['named']['leaves'])
                         | set(res_hu['named']['joins'])):
            lv = res_hu['named']['leaves'].get(nm, [])
            jn = res_hu['named']['joins'].get(nm, [])
            # SQF-10 + FLA-3 (par5 g3#6): a RESIDUAL named
            # relocation has no physical gesture left by
            # construction (the gesture's own rows were already
            # subtracted) - bank the paired move for the
            # seat/human to perform under ANY label (the
            # chat:-only gate dropped the window-label Return of
            # the Witch-king move and the census aged it into a
            # divergence); self-closed when the dump already
            # shows the piece at the destination.
            if len(lv) == 1 and len(jn) == 1:
                from board import sanitize as _snM
                from . import emission_card as _emcM
                _frmM = _snM(str(lv[0]))
                _toM = _snM(str(jn[0]))
                _pidM, _ = _emcM._named_pid_side(nm)
                _zM = _snM(str((getattr(self.view, 'piece_region',
                                        {}) or {}).get(
                    str(_pidM), '')))
                if _pidM is not None and _zM == _toM:
                    self.findings.append(
                        ('echo_move_self_closed',
                         '%s %s->%s x1' % (nm, _frmM, _toM)))
                else:
                    _stamp63({'kind': 'named_move', 'side': hu_tag,
                              'region': _frmM, 'to_region': _toM,
                              'tkey': nm, 'n': 1})
                    items.append(('chat', 'prompt',
                                  chat_mod.echo_move_note(
                                      _frmM, _toM)))
                continue
            # FLA-2 (par5 g5#2, the Gwaihir separation): a JOIN
            # with no leave is a figure ARRIVING from a
            # non-region zone - a separation leaving the
            # Fellowship box, or an out-of-play return.  The drag
            # from the client's own zone is owed; the
            # box->region mirror lane consumes it by identity.
            if len(jn) == 1 and not lv:
                from board import sanitize as _snJ
                from . import emission_card as _emcJ
                _toJ = _snJ(str(jn[0]))
                _pidJ, _ = _emcJ._named_pid_side(nm)
                _zJ = _snJ(str((getattr(self.view, 'piece_region',
                                        {}) or {}).get(
                    str(_pidJ), '')))
                if _pidJ is not None and _zJ == _toJ:
                    self.findings.append(
                        ('echo_move_self_closed',
                         '%s ->%s x1' % (nm, _toJ)))
                else:
                    # no dump / unknown zone: the Fellowship box
                    # is the one non-region home a human-side
                    # named JOIN normally leaves; a wrong source
                    # refuses loudly at the seat and the census
                    # keeps holding on the demand
                    _frmJ = _zJ or 'fellowship of the ring'
                    _stamp63({'kind': 'named_move',
                              'side': hu_tag, 'region': _frmJ,
                              'to_region': _toJ, 'tkey': nm,
                              'n': 1})
                    items.append(('chat', 'prompt',
                                  chat_mod.echo_move_note(
                                      _frmJ, _toJ)))
                continue
            # FLA-2: a LEAVE whose figure ENTERED the Fellowship
            # box is a move INTO the box, never a kill
            if lv and not jn and nm in _post_fel9:
                from board import sanitize as _snE
                _stamp63({'kind': 'named_move', 'side': hu_tag,
                          'region': _snE(str(lv[0])),
                          'to_region': 'fellowship of the ring',
                          'tkey': nm, 'n': 1})
                items.append(('chat', 'prompt',
                              chat_mod.echo_move_note(
                                  _snE(str(lv[0])),
                                  'fellowship of the ring')))
                continue
            for reg in lv[len(jn):]:
                _stamp63({'kind': 'named', 'side': hu_tag,
                          'region': reg, 'tkey': nm, 'n': 1})
            # BL-136(i): a NAMED figure riding the stamped siege
            # move-in (the Witch-king with the besiegers) is a
            # demanded relocation too
            if _siege136 and lv and jn:
                from board import sanitize as _sn136
                if any(_sn136(str(r)) == _sn136(_frm136)
                       for r in lv) \
                        and any(_sn136(str(r)) == _sn136(_to136)
                                for r in jn):
                    _stamp63({'kind': 'named_move', 'side': hu_tag,
                              'region': _sn136(_frm136),
                              'to_region': _sn136(_to136),
                              'tkey': nm, 'n': 1})
                    items.append(('chat', 'prompt',
                                  chat_mod.echo_move_note(
                                      _sn136(_frm136),
                                      _sn136(_to136))))
        if _siege136:
            # BL-136(i) POST-CONDITION: the stamped move-in MUST
            # have produced a demand (or a self-close) - silence
            # here is the defect
            from board import sanitize as _sn136b
            _seen136 = any(
                str(e.get('kind', '')).endswith('_move')
                and _sn136b(str(e.get('region', ''))) ==
                _sn136b(_frm136)
                and _sn136b(str(e.get('to_region', ''))) ==
                _sn136b(_to136)
                for e in self.echo_ledger)
            _closed136 = any(
                k == 'echo_move_self_closed'
                and _sn136b(_frm136) in str(v)
                for k, v in self.findings[-12:])
            if not _seen136 and not _closed136:
                self.alarms.append(
                    'SIEGE MOVE-IN UNDEMANDED: the engine moved '
                    'your army %s -> %s into the siege but no '
                    'drag demand was banked - drag it yourself; '
                    'RESYNC will show the drift'
                    % (_frm136, _to136))
                self.findings.append(('siege_movein_undemanded',
                                      '%s -> %s' % (_frm136,
                                                    _to136)))
        # P1: a FELLOWSHIP companion death inside a chat/pick
        # commit (hunt_sacrifice_guide above all) produces NO army
        # delta, so the removal machinery never banks it: bank the
        # named kill for the HUMAN-FP side (the seat's echo
        # performer kills the piece wherever the client holds it,
        # guide box included).  A companion who JOINED a post army
        # was separated, not killed - the named-join machinery
        # owns those.
        if not getattr(self, 'spectator', False):
            try:
                import log_replay as _lrP1
                _fc = {}
                for _st9, _tag9 in ((pre, 'pre'), (post, 'post')):
                    _s9 = {str(getattr(c, 'name', c)) for c in
                           (getattr(_st9.fellowship, 'companions',
                                    ()) or ())}
                    _g9 = getattr(getattr(_st9.fellowship,
                                          'guide', None),
                                  'name', None)
                    if _g9:
                        _s9.add(str(_g9))
                    _fc[_tag9] = _s9
                for _nmP1 in sorted(_fc['pre'] - _fc['post']):
                    if any(_nmP1 == str(getattr(ld, 'name', ld))
                           for _k9, _a9 in post.armies.items()
                           for ld in (getattr(_a9, 'leaders', ())
                                      or ())):
                        continue
                    _pidP1 = next(
                        (p for p in _lrP1.COMPANIONS
                         if _nmP1 in set(_lrP1.names_for_pid(
                             str(p)) or ())), None)
                    _zP1 = str((getattr(self.view, 'piece_region',
                                        {}) or {}).get(
                        str(_pidP1), ''))
                    if 'casualt' in _zP1.lower() \
                            or 'dead' in _zP1.lower():
                        continue    # client already shows it
                    if hu_tag == 'fp':
                        _stamp63({'kind': 'named', 'side': hu_tag,
                                  'region': 'fellowship',
                                  'tkey': _nmP1, 'n': 1})
                        continue
                    # T1 (physical lane): the AI is the FP and ITS
                    # companion died inside the HUMAN's commit
                    # (e.g. the client's random-companion draw
                    # parked the piece in 'SA Card Play'): the AI
                    # renders its own companion's death from
                    # wherever the client holds the piece
                    if _pidP1 is None:
                        self.alarms.append(
                            'AI COMPANION DEATH UNRENDERED: %s has '
                            'no client pid' % _nmP1)
                        continue
                    try:
                        import emitter as _emP1
                        from emitter import render as _rdP1
                        from board import sanitize as _snP1
                        _zcP1 = (_emP1.zone_code_for(
                            _snP1(_zP1).strip().lower())
                            if _zP1 else _emP1.FELLOWSHIP_BOX)
                    except LookupError:
                        _zcP1 = None
                    if _zcP1 is None:
                        self.alarms.append(
                            'AI COMPANION DEATH UNRENDERED: %s at '
                            'client zone %r (no zone code) - drag '
                            'the piece to the casualties box'
                            % (_nmP1, _zP1))
                        continue
                    items.append(('line', _rdP1(
                        'hand_exit_b',
                        {'nick': self.nick, 'pieces': [str(_pidP1)],
                         'frm': _zcP1}, local=False)))
                    self.findings.append(
                        ('ai_companion_death_emitted',
                         '%s from %s' % (_nmP1, _zP1 or 'box')))
            except Exception as _eP1:
                self.findings.append(('fellowship_death_bank_err',
                                      repr(_eP1)[:80]))
        # BL-43/D6: ANY commit that banks human board legs must
        # SAY so (silently banked demands expire unclaimed at
        # rollover)
        if _banked43:
            # J2 + FLA-3/D5a (par5 g3#4): the wording follows the
            # ROW TYPES.  *_move-only banks already said their own
            # per-row drag note - the kills wording here printed
            # 'your destroyed pieces are still on the board' after
            # a ZERO-casualty siege retirement.
            _note43 = chat_mod.echo_casualties_owed_note(_banked43_rows)
            if _note43 is not None:
                pass
            elif all(k.endswith('_move') for k in _banked43):
                _note43 = None
            elif str(label).startswith('chat:'):
                _note43 = chat_mod.echo_owed_note()
            elif all(k.endswith('_in') for k in _banked43):
                _note43 = chat_mod.echo_arrivals_note()
            else:
                _note43 = chat_mod.echo_kills_owed_note()
            if _note43 is not None:
                items.append(('chat', 'prompt', _note43))
        return items

    def _post_commit_consequences(self, pre, post, gesture=None, expressed=None,
                                  label='', actions=None, track_sources=None,
                                  ai_rows=None, ai_rows_blocked=False):
        """Render settled consequences only after the preceding physical debt."""
        from . import emission as em_mod
        items = []
        if ai_rows is not None:
            items.extend(self._render_post_commit_ai_rows(pre, post, ai_rows, ai_rows_blocked, label))
        items.extend(self._auto_wiped_figure_cleanup())
        attack_commit = any(getattr(a, 'action_type', '') == 'army_attack'
                            for a in (actions or ()))
        card_commit = any(getattr(a, 'action_type', '') in ('play_event', 'card_option')
                          for a in (actions or ()))
        recovered_political_sources = {
            str(fields.get('piece')): fields.get('box')
            for op, fields in (expressed or ())
            if op == 'recovered_political_source'}
        if recovered_political_sources or any(not pre.political.active.get(nation, False) and active
               for nation, active in post.political.active.items()):
            # Activation belongs to the committed transition, including an
            # ordinary invasion of an empty foreign region. Do not restrict
            # this to attack/card actions: army movement also activates nations.
            # Waiting for quiet reconciliation leaves its native counter
            # passive after the committed action. A flip is a toggle, so honor both
            # already performed gestures and the existing one-way ledger.
            from log_replay import POLITICAL_MARKERS
            from emitter import render as _attack_render
            walked = getattr(self, '_pol_face_walked', None)
            if walked is None:
                walked = self._pol_face_walked = set()
            marker_states = {}
            if getattr(self.view, 'dump_path', None):
                from .client_protocol import read_marker_states
                marker_states = read_marker_states(self.view.dump_path)
            for pid, nation in POLITICAL_MARKERS.items():
                pid = str(pid)
                was_active = pre.political.active.get(nation, False)
                if ((was_active and pid not in recovered_political_sources)
                        or not post.political.active.get(nation, False)
                        or pid in walked):
                    continue
                if was_active and marker_states.get(pid) != 0:
                    continue
                shown = any(op == 'flip' and pid in
                            [str(p) for p in (fields.get('pieces') or ())]
                            for op, fields in (expressed or ()))
                aliases = {str(nation.value).lower(),
                           str(nation.name).lower().replace('_', ' ')}
                shown = shown or any(
                    rec.get('op') == 'flip' and
                    str(rec.get('obj', '')).lower() in aliases
                    for rec in (getattr(gesture, 'records', None) or ())
                    if isinstance(rec, dict))
                # The dump may already include a human flip before its
                # feed record joins the declaration window.
                # Political br pieces report mask U for both faces; their
                # native H.k() state is 0 passive / 1 active.
                shown = shown or marker_states.get(pid) == 1
                walked.add(pid)
                if not shown:
                    items.append(('line', _attack_render(
                        'flip', {'nick': self.nick, 'pieces': [pid]},
                        local=False)))
                    if attack_commit:
                        self.findings.append(('attack_activation_emitted', nation.name))
                    elif card_commit:
                        self.findings.append(('card_activation_emitted', nation.name))
                    else:
                        self.findings.append(('political_activation_emitted', nation.name))
        # (2b) POLITICAL RIDER SWEEP (tester 2026-08-24): a card
        # effect's automatic political advance the human did NOT drag
        # is engine bookkeeping - _covers tolerates it on card-play
        # windows, and THIS leg walks the client marker for the
        # un-voiced steps (left unrendered, the next resync's
        # adopt_tracks would regress the ENGINE to the stale client
        # marker)
        voiced = None
        if gesture is not None:
            try:
                _gf9 = gesture_match.window_facts(gesture)
                # Every human window sweeps its unvoiced political
                # rider; the in-flight ledger guards double walks.
                voiced = dict(_gf9.political)
            except Exception:
                voiced = None
        elif str(label).startswith('chat:'):
            # FLC H3 (par5 g1#13): a CHAT-committed attack's
            # attacked-nation political advance never rendered at the
            # action - the client marker was only walked by the
            # rollover recon, which the tester read as a 'double step
            # for one attack'.  A chat commit voices nothing
            # physically, so every political delta sweeps here (the
            # BL-6 in-flight ledger still guards double-walks).
            voiced = {}
        elif str(label).startswith('ai:'):
            # Every AI transition can cause a political rider, including
            # victory advance/capture and automatic combat resolution. Render
            # it at this commit, subtracting marker steps already expressed.
            import log_replay as _attack_lr
            voiced = {}
            for op, fields in (expressed or ()):
                if op not in ('political_91', 'political_96', 'political_101'):
                    continue
                nation = _attack_lr.POLITICAL_MARKERS.get(
                    str(fields.get('piece', '')))
                if nation is not None:
                    key = nation.name
                    voiced[key] = voiced.get(key, 0) + 1
        if voiced is not None:
            try:
                import emitter as _emt
                from emitter import render as _prender
                # Cold recovery can prove the consequence on the native
                # board even when its original political feed row was lost.
                # Count it once alongside any surviving typed gesture.
                import log_replay as _recovered_lr
                recovered_politics = {}
                for op, fields in expressed or ():
                    if not fields.get('_recovered_political'):
                        continue
                    nation = _recovered_lr.POLITICAL_MARKERS.get(str(fields.get('piece', '')))
                    if nation is not None:
                        key = nation.name
                        recovered_politics[key] = recovered_politics.get(key, 0) + 1
                for key, count in recovered_politics.items():
                    voiced[key] = max(int(voiced.get(key, 0)), count)
                for nat, b1 in post.political.boxes.items():
                    b0 = pre.political.boxes.get(nat, b1)
                    key = getattr(nat, 'name', str(nat))
                    extra = (int(b1) - int(b0)) \
                        - int(voiced.get(key, 0))
                    base = int(b1) - extra
                    # BL-6: the rider consults the SAME in-flight
                    # ledger every other walk emitter feeds - a marker
                    # with a correction already in flight is never
                    # walked a second time
                    _infl6 = getattr(self, '_pol_recon_inflight',
                                     None)
                    if _infl6 is None:
                        _infl6 = self._pol_recon_inflight = {}
                    import log_replay as _lr6
                    _pid6 = next(
                        (str(p) for p, n6 in
                         _lr6.POLITICAL_MARKERS.items()
                         if getattr(n6, 'name', str(n6)) == key), None)
                    if _pid6 in recovered_political_sources:
                        source = recovered_political_sources[_pid6]
                        if source is None:
                            self.findings.append(('political_sweep_hold',
                                '%s: recovered marker has no measured source box' % key))
                            continue
                        base = int(source)
                        extra = int(b1) - base
                    elif extra <= 0:
                        continue
                    if extra == 0:
                        continue
                    _tgt6 = 'Game.%d' % (2000 + int(b1))
                    if _pid6 is not None \
                            and _infl6.get(_pid6) == _tgt6:
                        self.findings.append(
                            ('political_sweep_skipped_inflight', key))
                        continue
                    if extra < 0:
                        # A cold recovered consequence may find a marker
                        # beyond its rules target. Use the existing native
                        # correction form, never a fabricated forward step.
                        items.append(('line', _prender('move', {
                            'nick': self.nick, 'pieces': [_pid6],
                            'frm': 'Game.%d' % (2000 + base),
                            'to': _tgt6}, local=False)))
                        _infl6[_pid6] = _tgt6
                        self.findings.append(('political_sweep_emitted',
                                              '%s %d->%d' % (key, base, int(b1))))
                        continue
                    for step in range(extra):
                        try:
                            op9, f9 = _emt.plan_advance_political(
                                self.view, self.nick,
                                'fp' if self.human_side.name
                                == 'FREE_PEOPLES' else 'shadow',
                                {'nation': key, 'from': base + step})
                        except LookupError as _pe:
                            self.findings.append(
                                ('political_sweep_hold',
                                 '%s: %s' % (key, _pe)))
                            break
                        items.append(('line',
                                      _prender(op9, f9, local=False)))
                        self.findings.append(
                            ('political_sweep_emitted',
                             '%s %d->%d' % (key, base + step,
                                            base + step + 1)))
                    if _pid6 is not None:
                        _infl6[_pid6] = _tgt6
            except Exception as _pse:
                self.alarms.append('POLITICAL SWEEP FAILED (%s): %r'
                                   % (label, _pse))
        # (2c) CORRUPTION MARKER SWEEP (BL-2): an engine corruption
        # change that NO gesture or planner rendered walks marker 252
        # as the AI's own drag (corpus grammar: one Scribe drag on the
        # shared Track zones, multi-step attested); left unrendered,
        # the next adoption's adopt_tracks would regress the ENGINE to
        # the stale client marker.  The owed target arms the
        # adopt_tracks hold until the dump confirms.
        try:
            _c0 = int(getattr(pre.fellowship, 'corruption', 0) or 0)
            _c1 = int(getattr(post.fellowship, 'corruption', 0) or 0)
        except Exception:
            _c0 = _c1 = 0
        if _c1 != _c0:
            # A chat decision can finish a positively identified partial
            # marker drag. Its client source is the buffered endpoint,
            # while the engine's atomic pre-decision value is still _c0.
            _explicit252 = track_sources is not None and '252' in track_sources
            _source252 = int(track_sources['252']) if _explicit252 else _c0
            _voiced252 = False
            if gesture is not None:
                try:
                    _voiced252 = any(
                        '252' in str((r or {}).get('norm', '') if
                                     isinstance(r, dict) else r)
                        for r in (getattr(gesture, 'records', None)
                                  or ()))
                except Exception:
                    _voiced252 = False
            if expressed is not None and not _voiced252:
                _voiced252 = any(
                    '252' in str((f or {}).get('pieces', ''))
                    or str((f or {}).get('piece', '')) == '252'
                    for _op9, f in (expressed or ()))
            if not _voiced252 and not _explicit252:
                # BL-58: the human's OWN drag may already have walked
                # marker 252 as a separate record this tick - the feed
                # tail is the witness the dump cannot yet be
                _tail58 = [str(l).lower() for l in
                           getattr(self.feed, 'lines', [])[-40:]]
                if any('corruption' in l
                       and ('track %d' % _c1) in l
                       for l in _tail58):
                    _voiced252 = True
            if not _voiced252:
                import time as _time9
                from emitter import render as _crender
                items.append(('line', _crender(
                    'move', {'nick': self.nick, 'pieces': ['252'],
                             'frm': 'Game.%d' % (2014 + _source252),
                             'to': 'Game.%d' % (2014 + _c1)},
                    local=False)))
                owed = getattr(self, '_track_owed', None)
                if owed is None:
                    owed = self._track_owed = {}
                owed['252'] = (_c1, _time9.time() + 45.0)
                self.findings.append(('corruption_sweep_emitted',
                                      '%s: %d -> %d'
                                      % (label, _c0, _c1)))
        # (2c-bis) PROGRESS-MARKER SWEEP (F5): the declare's
        # progress-reset leg (marker 248) is engine bookkeeping under
        # an identity commit - unrendered, adopt_tracks regresses the
        # engine's reset to the stale client marker (the BL-2 pattern
        # on pid 248)
        try:
            _g0 = int(getattr(pre.fellowship, 'progress', 0) or 0)
            _g1 = int(getattr(post.fellowship, 'progress', 0) or 0)
        except Exception:
            _g0 = _g1 = 0
        if _g1 != _g0:
            # BL-101: coverage evidence must name the DESTINATION (a
            # source-only phrase would count an unrelated earlier walk
            # as covering the reset)
            _dst248 = 'to fellowship track %d' % _g1
            _v248 = False
            if gesture is not None:
                _v248 = any(
                    _dst248 in str((r or {}).get('norm', '')
                                   if isinstance(r, dict)
                                   else r).lower()
                    for r in (getattr(gesture, 'records',
                                      None) or ()))
            if gesture is not None and not _v248 and _g1 == 0:
                # Native reveal flips reset the progress marker themselves;
                # no separate FSP-move log line is produced.
                try:
                    _v248 = any(kind == 'reveal' for _region, kind in
                                gesture_match.window_facts(gesture).fel_flips)
                except Exception:
                    pass
            if expressed is not None and not _v248:
                _v248 = any(
                    ('248' in str((f or {}).get('pieces', ''))
                     or str((f or {}).get('piece', '')) == '248')
                    and str((f or {}).get('to', '')).endswith(
                        str(2014 + _g1))
                    for _o9, f in (expressed or ()))
            if not _v248:
                _t248 = [str(l).lower() for l in
                         getattr(self.feed, 'lines', [])[-40:]]
                if any(_dst248 in l for l in _t248):
                    _v248 = True
            if not _v248:
                import time as _tg9
                from emitter import render as _grender
                items.append(('line', _grender(
                    'track_marker', {'nick': self.nick,
                                     'piece': '248',
                                     'frm': 'Game.%d' % (2014 + _g0),
                                     'to': 'Game.%d' % (2014 + _g1)},
                    local=False)))
                owed = getattr(self, '_track_owed', None)
                if owed is None:
                    owed = self._track_owed = {}
                owed['248'] = (_g1, _tg9.time() + 45.0)
                self.findings.append(('progress_sweep_emitted',
                                      '%s: %d -> %d'
                                      % (label, _g0, _g1)))
        # (2d) FELLOWSHIP-PAWN SWEEP (F1): a committed effect's pawn
        # leg (Cruel Weather moves the fellowship) must render, or the
        # next adoption REGRESSES the engine to the stale client pawn
        # (the BL-2 pattern on pid 247).  fellowship_region changes
        # only on PUBLIC transitions (declare/reveal/effects), never on
        # hidden progress, so the render leaks nothing.
        try:
            import game_logic as _gl8
            from board import sanitize as _sz8
            _p0 = _sz8(str(_gl8.fellowship_region(pre)))
            _p1 = _sz8(str(_gl8.fellowship_region(post)))
        except Exception:
            _p0 = _p1 = ''
        if _p1 and _p1 != _p0 \
                and not getattr(post.fellowship, 'mordor_entered',
                                False):
            # The public client view records physical pawn movement even
            # when readable log text names 'The Fellowship', not PID247.
            _observed_pawn = (getattr(self.view, 'piece_region', {}) or {}).get('247')
            _v8 = (_observed_pawn is not None
                   and _sz8(str(_observed_pawn)) == _p1)
            if gesture is not None and not _v8:
                try:
                    _pawn_moves = gesture_match.window_facts(gesture).fel_moves
                    _v8 = bool(_pawn_moves and
                               _sz8(str(_pawn_moves[-1][1])) == _p1)
                except Exception:
                    pass
            if expressed is not None and not _v8:
                _v8 = any('247' in str((f or {}).get('pieces', ''))
                          or str((f or {}).get('piece', '')) == '247'
                          for _o8, f in (expressed or ()))
            if not _v8:
                try:
                    import emitter as _em8
                    from emitter import render as _r8
                    import time as _t8
                    items.append(('line', _r8(
                        'move', {'nick': self.nick,
                                 'pieces': ['247'],
                                 'frm': _em8.zone_code_for(_p0),
                                 'to': _em8.zone_code_for(_p1)},
                        local=False)))
                    self._pawn_owed = (_p1, _t8.time() + 45.0)
                    self.findings.append(('pawn_sweep_emitted',
                                          '%s -> %s' % (_p0, _p1)))
                except Exception as _pe8:
                    self.alarms.append('PAWN SWEEP FAILED: %r'
                                       % (_pe8,))
        # (2e) SIEGE SUB-BOX SWEEP (BL-76): a newly formed siege owes
        # the DEFENDERS' field -> Stronghold-box drags (unseated, the
        # dump shows field units, adoption re-imports them and
        # cleanup_sieges drops the siege it cannot see).  Human
        # defender: demand + publish; the AI's own withdrawals render
        # via its planner.
        try:
            _bs0 = set(getattr(pre, 'besieged_strongholds', set())
                       or set())
            _bs1 = set(getattr(post, 'besieged_strongholds', set())
                       or set())
        except Exception:
            _bs0 = _bs1 = set()
        _hu_tag76 = ('fp' if self.human_side.name == 'FREE_PEOPLES'
                     else 'shadow')
        for _reg76 in (_bs1 - _bs0):
            try:
                from combat import garrison_side_for as _gsf76
                from board import sanitize as _sz76
                _side76 = _gsf76(post, _sz76(str(_reg76)))
                # FLA-1 (par5 g2#4/#7, g5#7): compare by enum NAME -
                # str(Side.X) is 'Side.X', which never matched the
                # 'fp'/'sh' tags, so the chat-siege garrison seat was
                # never demanded (2/2 chat sieges diverged)
                _tag76 = ('fp' if getattr(_side76, 'name', '')
                          == 'FREE_PEOPLES' else 'shadow')
            except Exception:
                continue
            if not _tag76.startswith(_hu_tag76[:2]):
                continue
            from .siege_withdrawal_gestures import trim_before_seating_prompt
            if trim_before_seating_prompt(self, post, _reg76):
                # Retain the census obligation, but first consume the saved
                # legal trim. The human already seated the remaining five.
                self._siege_seat_owed = str(_reg76)
                continue
            from .shadow_siege_slots import garrison_seated
            self.view.refresh_dump()
            if garrison_seated(self.view, post, _reg76, _side76):
                self._siege_seat_owed = None
                continue
            self._siege_seat_owed = str(_reg76)
            items.append(('chat', 'prompt',
                          chat_mod.siege_seat_note()))
            self.findings.append(('siege_seat_demanded',
                                  str(_reg76)))
        # (3) capture markers — engine event, ledger-reconciled
        em2, notes = em_mod.render_capture_markers(
            self.view, self.nick, post, self.ctrl_markers,
            self.vp_marks)
        for n in notes:
            self.findings.append(('capture_marker', n))
            if 'EXHAUSTED' in n or 'failed' in n:
                self.alarms.append('CAPTURE MARKER: %s' % n)
        items.extend(('line', l) for l in em2.lines)
        # Every physical follow-through row needs confirmation, including
        # control/VP and political markers. Register the complete emission
        # once; acknowledging casualties alone must not release this commit.
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
        from .temporary_combat_figures import remove_park_fact
        remove_park_fact(self.view,gesture,facts)
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
                rows['named'][b].setdefault(emc._named_identity(nm), []).append(reg)
        # BL-152 (USER manual game 2026-08-28): window_facts.leaders
        # is COUNT-only (no nation); the RECORD tokens carry the nation
        # ('Rohan Leader'), so plain leaders are attributed from them
        # per region and only the residual the tokens do not explain
        # falls to the count-only fallback below (an FP residual
        # blocks the sweep)
        import rendered_replay as rr
        attributed = {}
        for rec in (getattr(gesture, 'records', None) or ()):
            op = rec.get('op')
            if op == 'move':
                obj = str(rec.get('obj', '') or '')
                toks = (rr.split_group(obj[len('group ('):-1])
                        if obj.startswith('group (') and obj.endswith(')')
                        else [obj])
                src = gesture_match.zone_key(rec.get('from'))
                dst = gesture_match.zone_key(rec.get('to'))
            elif op == 'kill':
                toks = list(rec.get('pieces') or ())
                src = gesture_match.zone_key(rec.get('zone'))
                dst = None
            else:
                continue
            for t in toks:
                pp = rr.parse_piece(t)
                if not pp or pp[0] != 'leader':
                    continue
                nat = gesture_match._nation_enum(pp[1])
                if nat is None:
                    continue
                key = '%s:leader' % str(getattr(nat, 'value', nat)).lower()
                if em_mod._row_side(key) != side_tag:
                    continue
                n = int(pp[3] or 1)
                slot = rows['units'].setdefault(
                    key, {'minus': [], 'plus': []})
                if src and src[0] in ('region', 'stronghold'):
                    slot['minus'].append((src[1], n))
                    attributed[src[1]] = attributed.get(src[1], 0) - n
                if dst and dst[0] in ('region', 'stronghold'):
                    slot['plus'].append((dst[1], n))
                    attributed[dst[1]] = attributed.get(dst[1], 0) + n
        blocked = False
        for reg, v in facts.leaders.items():
            v = v - attributed.get(reg, 0)      # what the tokens explained
            if not v:
                continue
            # generic leader rows (nazgul + plain leaders share the
            # client token space): attribute the SHADOW ones to nazgul
            # (the only fungible shadow leader); an FP generic-leader
            # residual the tokens could not attribute blocks the sweep
            # for safety
            if side_tag == 'shadow':
                b = 'plus' if v > 0 else 'minus'
                rows['naz'][b].append((reg, abs(v)))
            else:
                blocked = True
        return rows, blocked

    def _client_tkey_census(self, tkey):
        """BL-77 client facts: {region: count} of the client board's
        pieces matching a ledger tkey ('<nation>:regular|elite|leader'
        or 'nazgul'), read the exact way the census/adoption read the
        dump (adopt.client_units_by_region / client_board:360-376) so
        the namespaces cannot drift.  None = no dump to read."""
        dump = getattr(self.view, 'piece_region', None)
        if not dump:
            return None
        from . import adopt as ad
        from . import client_board as cb
        import board as bd
        out = {}
        if tkey == 'nazgul' or str(tkey).endswith(':leader'):
            want_ut = 'nazgul' if tkey == 'nazgul' else 'leader'
            want_nat = (None if tkey == 'nazgul'
                        else str(tkey).split(':')[0])
            for pid, zone in dump.items():
                base = cb._strip_sh(ad._norm(zone))
                if base not in bd.regions:
                    continue
                try:
                    nat, ut = cb.unit_type(pid,
                                           self.view.dynamic_types)
                except Exception:
                    continue
                if ut != want_ut:
                    continue
                if want_nat is not None and str(
                        getattr(nat, 'value', nat)).lower() != want_nat:
                    continue
                out[base] = out.get(base, 0) + 1
            return out
        for (reg, (nat, ut)), c in ad.client_units_by_region(
                dump, self.view.dynamic_types).items():
            k = '%s:%s' % (str(getattr(nat, 'value', nat)).lower(), ut)
            if k == tkey:
                out[reg] = out.get(reg, 0) + c
        return out

    def _engine_tkey_count(self, st9, tkey, region):
        """Engine-side count of tkey units at region (regular/elite/
        leader tkeys + 'nazgul'; None = not countable, callers must
        not self-close)."""
        try:
            from board import sanitize as _sanE
            if str(tkey) == 'nazgul' or str(tkey).endswith(':leader'):
                # FLA-5 (par5 g5#6 + g1#2): leader rows were 'not
                # countable', so the BL-101 arrival self-close never
                # ran and a client-placed leader was banked AGAIN
                # (engine=1 client=2).  ONE composition reader:
                # emission_card._leader_comp.
                from . import emission_card as _emcC
                naz9, plain9 = _emcC._leader_comp(
                    st9, _sanE(str(region)))
                if str(tkey) == 'nazgul':
                    return int(naz9)
                nat9 = str(tkey).rsplit(':', 1)[0]
                return int(sum(
                    n for nv, n in plain9.items()
                    if str(nv).lower() == nat9))
            nat, kind = str(tkey).rsplit(':', 1)
            if kind not in ('regular', 'elite'):
                return None
            a = next((ar for k, ar in st9.armies.items()
                      if _sanE(str(k)) == str(region)), None)
            if a is None:
                return 0
            return sum(
                1 for u in a.units
                if str(getattr(u.nation, 'value', u.nation)).lower()
                == nat
                and str(getattr(u.kind, 'value', u.kind)).lower()
                == kind)
        except Exception:
            return None

    def _move_row_fate(self, row):
        """BL-77 (shgame9 blocker): decide a to-be-banked relocation
        row's fate from CLIENT facts before it enters the ledger.
          'satisfied'       - the client board already shows the
                              pieces at the destination and none
                              stand at the named source: the 'move'
                              is the engine ADOPTING the client's own
                              drag; banking it is uncloseable forever
                              (shgame9: the approach leg banked while
                              the real debt was the siege move-in).
          'rebase:<region>' - the client holds the pieces at exactly
                              ONE other region (a combined delta
                              named the pre-approach source): demand
                              the drag from where the pieces actually
                              stand, keeping the matcher exact.
          None              - bank as written (incl. unreadable or
                              ambiguous client facts: the loud
                              unmatched paths downstream own those).
        """
        try:
            cnt = self._client_tkey_census(row['tkey'])
        except Exception:
            return None
        if cnt is None:
            return None
        n = int(row['n'])
        if cnt.get(row['region'], 0) >= n:
            return None
        alts = sorted(r for r, c in cnt.items()
                      if r not in (row['region'], row['to_region'])
                      and c >= n)
        if len(alts) == 1:
            return 'rebase:' + alts[0]
        if alts:
            return None
        if cnt.get(row['to_region'], 0) >= n:
            return 'satisfied'
        return None

    def _consume_mirror_echo(self, rec):
        """A non-self kill/removal record FULLY covered by open echo-
        ledger entries is the client-side mirror of an effect the
        engine already committed (the cbt10 race) — consume it as an
        echo, never a new gesture.  Partial/unknown coverage returns
        False (normal routing; loud paths there)."""
        if not self.echo_ledger or getattr(self, 'spectator', False):
            return False
        import rendered_replay as rr
        from .emission_card import _named_identity
        op = rec.get('op')
        suffix = ''      # WAVE-J: '' = removal rows, '_in' = arrivals
        to_reg = None    # BL-80: set for region->region relocations
        if op == 'kill':
            zk = gesture_match.zone_key(rec.get('zone'))
            zname = str(zk[1]) if len(zk) > 1 else ''
            if zk[0] in ('region', 'stronghold'):
                reg = zk[1]
            elif _zone_key(zname) in _GUIDE_ZONES \
                    or 'fellowship' in zname.lower() \
                    or 'guide' in zname.lower():
                # R2 / BL-149: a companion killed IN the
                # guide/fellowship box is the FELLOWSHIP-DEATH demand's
                # performance (banked with region='fellowship'); the
                # zone NAME (zk[1]) is what is tested
                reg = 'fellowship'
            else:
                # BL-149: a kill zone that names no region ('Current
                # Action' - the client's constant kill zone, 376 of
                # the corpus' human kill lines).  A NAMED piece is
                # unique, so its echo is identified by tkey alone;
                # fungible rows (units/leaders/nazgul) still need the
                # region and refuse below.
                reg = None
            toks = list(rec.get('pieces', []))
        elif op == 'move':
            src = gesture_match.zone_key(rec.get('from'))
            dst = gesture_match.zone_key(rec.get('to'))
            # FLA-2 (par5 g5#2): a single NAMED figure dragged from a
            # NON-region zone (Fellowship/Guide box, casualties,
            # reinforcements) into a region performs a banked
            # named_move by IDENTITY - the separation demand's close;
            # exact on name + to_region.
            if dst[0] in ('region', 'stronghold') \
                    and src[0] not in ('region', 'stronghold'):
                _objN = str(rec.get('obj', '') or '')
                if not _objN.startswith('group ('):
                    _ppN = rr.parse_piece(_objN)
                    if _ppN is not None and _ppN[0] == 'named':
                        _ledN = [dict(e) for e in self.echo_ledger]
                        _hitN = next(
                            (e for e in _ledN
                             if e['kind'] == 'named_move'
                             and _named_identity(e['tkey']) == _named_identity(_ppN[1])
                             and e['n'] > 0
                             and e.get('to_region') == dst[1]), None)
                        if _hitN is not None:
                            _hitN['n'] -= 1
                            self.echo_ledger = [
                                e for e in _ledN if e['n'] > 0]
                            self.findings.append(
                                ('mirror_kill_echo',
                                 'move :: %s' % str(
                                     rec.get('norm', ''))[:100]))
                            return True
            if src[0] in ('region', 'stronghold') \
                    and dst[0] in ('dead', 'reinf'):
                reg = src[1]
            elif src[0] in ('reinf', 'dead') \
                    and dst[0] in ('region', 'stronghold'):
                # the human mirrors a banked ARRIVAL (forced-downgrade
                # replacement) by dragging the piece in from a reserve
                # zone - echo-consumable against '<kind>_in' entries
                # ONLY (a region->region move is never an arrival echo)
                reg = dst[1]
                suffix = '_in'
            elif src[0] in ('region', 'stronghold') \
                    and dst[0] in ('region', 'stronghold'):
                # BL-80: the performance of a banked RELOCATION
                # ('<kind>_move' rows) - a record matching an open echo
                # entry is FOLLOW-THROUGH, never a fresh gesture and
                # never pending-buffer content (sibling of 'rolls are
                # never gestures').  Both endpoints must match the
                # entry (region AND to_region).
                reg = src[1]
                to_reg = dst[1]
                suffix = '_move'
            elif dst[0] in ('dead', 'reinf'):
                # BL-149: a removal dragged from a NON-region zone
                # (guide box / fellowship box / Current FP Card ->
                # casualties) - identity-keyed for named pieces
                reg = None
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
                if suffix == '_in':
                    return False   # named arrivals never banked
                # BL-136(i): 'named_move' rows ARE banked (a minion
                # riding the siege move-in); kills stay 'named'
                want.append(('named' + suffix, reg, _named_identity(a), 1))
            elif kind == 'tile':
                # BL-149: a spent hunt tile killed in the same group
                # as the casualty is client bookkeeping (gesture_match
                # ignores tiles in window facts too) - skipped, never
                # a veto
                continue
            else:
                return False
        if not want:
            return False        # tile-only bookkeeping: not an echo
        if reg is None and any(k.startswith(('unit', 'naz'))
                               for k, _r, _t, _n in want):
            return False        # fungible rows need the region
        # coverage check on a COPY; adopt only when every row covers
        ledger = [dict(e) for e in self.echo_ledger]
        for kind, reg, tkey, count in want:
            need = count
            for e in ledger:
                if need <= 0:
                    break
                if e['kind'] == kind \
                        and (reg is None or e['region'] == reg) \
                        and (_named_identity(e['tkey']) if kind.startswith('named') else e['tkey']) == tkey and e['n'] > 0 \
                        and (to_reg is None
                             or e.get('to_region') == to_reg):
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

    def _decision_endpoint_revision(self):
        import os
        path = getattr(self.view, 'dump_path', None)
        try:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size, self._feed_seq)
        except (OSError, TypeError):
            return (None, None, self._feed_seq)

    def _resolve_decision_endpoint(self, window, result):
        """Treat all gestures at this decision as one observed board change.

        The matcher supplies legal ways to reach the endpoint. Their internal
        order is not another choice for the player. Keep the actual successor
        and its engine continuation rather than adopting a board and losing
        the die, card, or remaining decision that produced it.
        """
        import ai
        if getattr(self, 'spectator', False) or result.status not in ('match', 'ambiguous'):
            return result
        from .decision_endpoint import completed, filter as filter_endpoint
        # A single parser match can be wrong too. At a completed native
        # interaction the board is authoritative for every candidate, not
        # just a tie-breaker after the parser happens to report ambiguity.
        if result.status == 'match' and not (
                completed(window) and getattr(self.view, 'dump_path', None)):
            return result
        candidates = getattr(result, 'candidates', ())
        if not candidates:
            return result
        attempted_revision = self._decision_endpoint_revision()
        resolved = filter_endpoint(self, window, candidates)
        if resolved.status == 'matched':
            chosen = resolved.candidates[0]
            selected = gesture_match.MatchResult(
                'match', chain=chosen['chain'], post=chosen['post'],
                applies=result.applies, note=resolved.reason)
            if hasattr(result, '_inferred_ring_target'):
                selected._inferred_ring_target = result._inferred_ring_target
            self.findings.append(('decision_endpoint_matched', resolved.reason))
            return selected
        held = gesture_match.MatchResult('hold', applies=result.applies,
                                        note=resolved.reason)
        held.endpoint_wait = True
        held.endpoint_unavailable = resolved.status == 'unavailable'
        # A candidate may have opened a Hunt/combat/card choice even though
        # the pre-action state had no pending decision. A board-only fallback
        # must not discard that unfinished effect on a temporary mismatch.
        held.endpoint_pending = any(ai.describe_pending(row['post']) is not None
                                    for row in candidates)
        held.endpoint_evidence = resolved.evidence
        # A newer export may have arrived during comparison. It still needs
        # to be examined on the next tick, even with no further human input.
        window._endpoint_revision = attempted_revision
        self._tag_hold(window, 'decision_endpoint')
        self.findings.append(('decision_endpoint_wait', resolved.reason))
        return held

    def _match_window_now(self, w, side):
        """Match (attempt-pinned) and settle one window.  Outcome items
        go through self._out_queue — this runs from nested retry sites
        whose returns are discarded, and refusal undo presses must
        still reach the client."""
        changes = [r for r in w.records if r.get('op') == 'die_change']
        if gesture_match.window_facts(w).ring_moves and len(changes) == 1:
            # Bind visible identity before outcome comparison. Resolving it
            # only after matching made distinct dice look falsely ambiguous.
            from .sidecar import meta_pids
            from emitter import die_piece_for
            rec = changes[0]
            pids = meta_pids(rec.get('meta'))
            w._explicit_ring_targets = {
                i for i, die in enumerate(self.es.dice[side])
                if not die.used and die.current is not None
                and gesture_match.FACE_LETTER.get(die.current.name) == rec.get('frm')
                and (pids is None or die_piece_for(self.view, side.name, i) in pids)}
        if (gesture_match.window_facts(w).ring_moves
                and not any(r.get('op') == 'die_change' for r in w.records)):
            # The ordinary client gesture does not flip the die. Bind the
            # implicit conversion to the physical die the player staged.
            from .native_die_identity import pids as public_die_pids
            from emitter import die_piece_for
            staged = None
            for record in w.records:
                pids = ([w._staged_die_pid] if getattr(w, '_staged_die_pid', None)
                        else public_die_pids(record) or [])
                matches = [i for i in range(len(self.es.dice[side]))
                           if die_piece_for(self.view, side.name, i) in pids]
                if matches:
                    staged = matches[0] if len(matches) == 1 else -1
                    break
            if staged is not None:
                w._inferred_ring_target = staged
        if self._needs_staged_die_identity(w):
            self._tag_hold(w, 'die_identity')
            if not any(held is w for held, _ in self.holds):
                self.holds.append((w, gesture_match.MatchResult('hold', note='awaiting exact staged die identity')))
            return [('note','waiting for the exact native identity of the staged die')]
        res = None
        if (not getattr(self, 'spectator', False) and side is self.human_side
                and getattr(side, 'name', '') == 'FREE_PEOPLES'):
            from .crebain_gestures import interposition_match
            with self._attempt_pin()():
                res = interposition_match(self.es, side, w)
        if res is None:
            res = gesture_match.match_window(
                self.es, side, w, pin=self._attempt_pin(), rng=self.rctl,
                advance_tolerance=not getattr(self, 'spectator', False),
                fs_prefilter=not getattr(self, 'spectator', False))
        res = self._resolve_decision_endpoint(w, res)
        # BRLANE-12: the region chatted over this window (consumed here
        # whatever the outcome - it never outlives the window)
        _hint12 = getattr(self, '_card_target_hint', None)
        self._card_target_hint = None
        if res.status != 'match' \
                and not getattr(res, 'endpoint_wait', False) \
                and not getattr(w, '_require_full_match', False) \
                and not (res.status == 'hold'
                         and getattr(res, 'starved', False)):
            # item 14 / M10: the card leaving the hand IS the play -
            # commit it by identity, explain the rest (never a refusal
            # of the play, never undo presses)
            _res14 = self._card_identity_probe(w, side, res)
            if _res14 is not None:
                res = _res14
        w_norms = [r.get('norm', '') for r in w.records]
        if res.status == 'match':
            ring_error = self._bind_ring_targets(res.chain, w, side)
            if ring_error:
                res = gesture_match.MatchResult('none', note=ring_error)
        if res.status == 'match':
            self._rebind_chain_dice(res.chain, w)
            _pre14 = self.es
            if self._commit_chain(res.chain, w_norms, 'window',
                                  gesture=w):
                self.window_stats['matched'] += 1
                self.last_actor = side      # alternation bookkeeping
                self._bury_dead_candidate_asks()
                if getattr(res, 'crebain_interposition', False):
                    from .crebain_gestures import remember_transport
                    remember_transport(self, w)
                if getattr(res, 'identity_fallback', None):
                    self._out_queue.extend(
                        self._explain_card_play_extras(w, res, _pre14))
                self._out_queue.extend(
                    self._apply_card_target_hint(_hint12))
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
            # The starve note rides the window (_expire_deferred picks
            # the engine-gap shim from it); the starving CALL itself
            # rides the window too - the global rand.pending() may be
            # CLEARED by an interleaved attempt before the tick's ask
            # gate reads it (wjg11), and the ask gate / waiting_on
            # fall back to it.
            w._starve_note = res.note
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
        if (res.status in ('hold', 'ambiguous') and not self.enforce_human
                and not getattr(res, 'endpoint_unavailable', False)
                and not getattr(res, 'endpoint_pending', False)):
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
            if not getattr(res, 'endpoint_wait', False):
                self.rand.reset_feed()
            return []
        # 'none': the live refusal doctrine (rule 5) — client-native
        # undo over the gesture window + chat notice
        self.window_stats['none'] += 1
        # EXPLAIN BEFORE THE ROLLBACK: _unreveal() swaps the played
        # card back out of the hand, so a menu derived after it can
        # never contain the play_event the human attempted - the
        # diagnosis is taken from the state the MATCHER judged
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
        window fails to commit, the engine hand must return to its
        pre-window TRUTH.  Two models (BL-100(ii), shgame11):
          LIVE HUMAN side (visible-hand ruling, USER 2026-08-25): the
          identity was decoded at the draw and the physical card is
          back in the client hand after the undo - the slot must END
          holding the REAL cid.  The old placeholder restore stranded
          Candles of Corpses off the legal surface: engine hand
          [placeholder], client hand [the card], sizes matching - an
          identity-space divergence invisible to the count census,
          and play_event rows build only from real cids.
          SPECTATOR / blinded sides: the placeholder restore stands
          (a stranded reveal consumed the deck's placeholder budget
          and poisoned every later menu build - game 2023-01-13:
          Kindred stayed in hand after its window failed, Celeborn's
          reveal then found no strategy placeholder left).
        ONE body, called by BOTH failure paths (the pending-buffer
        orphan and the window refusal)."""
        import game_logic as gl
        for r_side, cid in getattr(w, '_reveals', []):
            hm = self.hand_models.get(r_side)
            if hm is None:
                continue
            hand = self.es.hands[r_side].hand
            if not getattr(self, 'spectator', False) \
                    and r_side is self.human_side:
                if cid not in hand:
                    dk = gl.card_deck_of(cid) or 'strategy'
                    _w9 = 'CHAR' if dk == 'character' else 'STRAT'
                    ph = next((c for c in hand
                               if str(c).startswith('HU_')
                               and _w9 in str(c).upper()), None)
                    if ph is not None:
                        hand[hand.index(ph)] = cid
                        self.findings.append(
                            ('refused_reveal_restored', str(cid)))
                    else:
                        self.alarms.append(
                            'REFUSED REVEAL UNRESTORABLE: %s not in '
                            'the engine hand and no %s placeholder'
                            % (cid, dk))
                continue
            if cid in hand:
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

    def _bind_ring_targets(self, chain, w, side):
        """Bind ring targets before rebinding the paid die of the action.

        Two dice with the same old face are fungible in a board-count
        projection, but their physical identities must survive a change.
        """
        import game_state as gs
        from .sidecar import meta_pids
        from emitter import die_piece_for
        rings = [a for a in chain if a.action_type == 'elven_ring']
        if not rings:
            return None
        changes = [r for r in w.records if r.get('op') == 'die_change']
        if len(rings) == 1 and not changes:
            target = getattr(w, '_inferred_ring_target', None)
            if target is None:
                candidates = [i for i, die in enumerate(self.es.dice[side])
                              if not die.used and die.current is not None
                              and gesture_match.FACE_LETTER.get(die.current.name) == w.die_face]
                target = candidates[0] if len(candidates) == 1 else None
            paid = [a.die_idx for a in chain if a.die_idx is not None and a.die_idx >= 0]
            if (target is None or target < 0 or not paid
                    or any(index != target for index in paid)
                    or rings[0].params.get('target_die_idx') != target):
                return 'Elven Ring action could not be linked to the die you used'
            return None
        if len(rings) != 1 or len(changes) != 1:
            return 'Elven Ring needs one explicit selected die and new face; undo the incomplete gesture'
        rec, action = changes[0], rings[0]
        if rec.get('side') != side.name or rec.get('to') != gesture_match.FACE_LETTER.get(gs.ActionDieFace(action.params['new_face']).name):
            return 'Elven Ring face change does not match this action'
        indices = [i for i, d in enumerate(self.es.dice[side]) if not d.used
                   and d.current is not None and gesture_match.FACE_LETTER.get(d.current.name) == rec.get('frm')]
        pids = meta_pids(rec.get('meta'))
        if pids is not None:
            indices = [i for i in indices if die_piece_for(self.view, side.name, i) in pids]
        if len(indices) != 1:
            return 'Elven Ring target is ambiguous or invalid; undo and select one unused die with its client piece ID'
        action.params['target_die_idx'] = indices[0]
        return None

    def _needs_staged_die_identity(self, window):
        """Two equal faces are fungible in search, not in the native client."""
        if getattr(self, 'spectator', False) or self._staged_die_idx(window) is not None:
            return False
        from game_state import Side
        owner = getattr(window, 'die_side', None)
        if owner not in Side.__members__:
            return False
        face = getattr(window, 'die_face', None)
        return sum(not die.used and die.current is not None
                   and gesture_match.FACE_LETTER.get(die.current.name) == face
                   for die in self.es.dice.get(Side[owner], ())) > 1

    def _staged_die_idx(self, w):
        """BL-108: (Side, idx) of the die the client PHYSICALLY
        staged for this window (sidecar pid -> die_id_map), or
        None."""
        selected = getattr(w, '_staged_die_pid', None)
        if selected is not None:
            return self.view.die_id_map.get(selected)
        from . import sidecar as _sc108
        for r in (getattr(w, 'records', None) or ()):
            try:
                from .native_die_identity import pids as public_die_pids
                pids = public_die_pids(r) or []
            except Exception:
                pids = []
            for pid in pids:
                b = self.view.die_id_map.get(str(pid))
                if b is not None \
                        and getattr(b[0], 'name', str(b[0])) \
                        == str(getattr(w, 'die_side', '')):
                    return b
        return None

    def _rebind_chain_dice(self, chain, w):
        """BL-108 (fpgame14 L3/L2): the engine's face-dedup
        enumerates ONE row per face, so a matched candidate's die_idx
        may name a DIFFERENT same-face die than the one the client
        physically staged - the engine then spends idx1 while the
        client killed idx3's pid, the die identities drift for the
        rest of the turn (the rollover returned 3 dice), and the
        phantom guide-redraw read the wrong die's face through the
        same drift.  The binding follows the CLIENT'S PHYSICAL
        CHOICE: rewrite die_idx to the staged pid's registered idx
        whenever the faces agree (same-face fungibility is engine-
        legal by construction - that is why the dedup exists)."""
        b = self._staged_die_idx(w)
        if b is None:
            return
        side_b, idx_b = b
        dice = self.es.dice.get(side_b, [])
        if not (0 <= idx_b < len(dice)) or dice[idx_b].used:
            return
        face_b = dice[idx_b].current
        for a in (chain or ()):
            di = getattr(a, 'die_idx', None)
            if di is None or di < 0 or di == idx_b:
                continue
            if di < len(dice) and not dice[di].used \
                    and dice[di].current == face_b:
                a.die_idx = idx_b
                self.findings.append(
                    ('die_rebound_to_staged',
                     'idx %s -> %s (%s)'
                     % (di, idx_b, getattr(face_b, 'name', face_b))))

    def _discard_identity_filter(self, dp, menu):
        """BL-106(b) - the severity-1 class's LAST door (fpgame14 +
        fpgame17): a discard pending matches by CARD IDENTITY, never
        by deck label.  Buffered kill records' sidecar pids resolve
        through card_pid_map's inverse; resolvable pids restrict the
        candidate rows to their exact cids; a pid whose cid has NO
        row REFUSES loudly (fpgame17: the deck-label kill for an
        already-discarded card matched a DIFFERENT card's row and
        Athelas left the engine hand unasked). A generic caption needs
        a complete current hand snapshot. None means hold/refusal."""
        from . import sidecar as _scB
        buf = self.pending_buf
        if buf is not None:
            buf._awaits_discard_identity = False
        _hand_zone = ('hand', 'FP' if getattr(dp.get('side'), 'name', '')
                      == 'FREE_PEOPLES' else 'SH')
        if any(rec.get('op') == 'kill'
               and gesture_match.zone_key(rec.get('zone')) != _hand_zone
               for rec in (getattr(buf, 'records', None) or ())):
            return None
        pids = []
        for r in (getattr(buf, 'records', None) or ()):
            if r.get('op') != 'kill':
                continue
            try:
                pids.extend(str(p) for p in
                            (_scB.meta_pids(r.get('meta')) or ()))
            except Exception:
                pass

        def _hand106():
            try:
                return list(self.es.hands[dp.get('side')].hand)
            except Exception:
                return []

        def _refuse106(kind, detail, msg):
            self.findings.append((kind, str(detail)[:80]))
            self._out_queue.append(
                ('chat', 'refusal', chat_mod.refusal_illegal(msg)))
            self._refusal_loop_check(msg)
            return None
        if not pids:
            # A visible card title is already a public identity. Drawn
            # cards can still be HU_ placeholders while another named
            # card is discarded; their presence must not turn that exact
            # discard into every possible hand-count reduction. Physical
            # pids, when supplied above, retain authority over the title.
            named_cids = set()
            hand_side = ('FP' if str(getattr(dp.get('side'), 'name',
                                           dp.get('side'))) == 'FREE_PEOPLES'
                         else 'SH')
            for rec in (getattr(buf, 'records', None) or ()):
                if (rec.get('op') != 'kill'
                        or gesture_match.zone_key(rec.get('zone'))
                        != ('hand', hand_side)):
                    continue
                for title in rec.get('pieces') or ():
                    cid = gesture_match.title_to_cid(title)
                    if cid is not None:
                        named_cids.add(str(cid))
            if named_cids:
                named_menu = [a for a in menu
                              if str(getattr(a, 'action_type', ''))
                              != 'discard_card'
                              or str((getattr(a, 'params', None) or {})
                                     .get('card_id')) in named_cids]
                if not any(getattr(a, 'action_type', '') == 'discard_card'
                           for a in named_menu):
                    return _refuse106(
                        'discard_identity_stale', sorted(named_cids)[:2],
                        'that card is not discardable now - it may already '
                        'be discarded; pick from the current menu')
                self.findings.append(('discard_identity_from_title',
                                      ', '.join(sorted(named_cids))[:80]))
                return named_menu
            # the feed sidecar REDACTS the pids of the human's own
            # hand, so a hand-limit discard rendered as a deck back
            # carries no identity through the feed; the CLIENT'S OWN
            # DUMP is the legitimate identity source (USER: the engine
            # sees the human hand): the decoded hand pid that LEFT the
            # hand zone is the killed card.  Exactly one -> identity;
            # several -> loud refusal; none yet (dump not caught up)
            # -> hold quietly, the buffer retries.
            try:
                if any(str(card).startswith('HU_') or str(card) not in self.view.card_pid_map
                       for card in _hand106()):
                    raise ValueError('unresolved hand slots cannot identify a generic discard')
                from .repair_evidence import dump_signature, same_dump
                from .client_protocol import dump_at_feed_position
                import copy
                evidence = dump_signature(self)
                header = evidence['header']
                if not header or int(header.get('session', 0)) <= 0:
                    raise ValueError('no live board position')
                self._check_dump_session(header)
                reader = copy.deepcopy(self.feed)
                reader.poll()
                if (reader._buf or self._feed_seq != len(reader.lines)
                        or not dump_at_feed_position(header, len(reader.lines))
                        or not self.view.refresh_dump()
                        or not same_dump(self, evidence)):
                    raise ValueError('discard board is not at the routed event position')
                if any(str(self.view.card_pid_map[str(card)]) not in self.view.piece_region
                       for card in _hand106()):
                    raise ValueError('incomplete hand identity coverage in native export')
            except (OSError, ValueError, TypeError):
                buf._awaits_discard_identity = True
                from .discard_wait_status import remember as _remember_discard_wait
                _remember_discard_wait(self, dp)
                return None
            _pr106 = getattr(self.view, 'piece_region', {}) or {}
            _map106 = self.view.card_pid_map or {}
            _gone106 = sorted(
                str(c) for c in _hand106()
                if not str(c).startswith('HU_')
                and _map106.get(str(c)) is not None
                and str(_pr106.get(str(_map106.get(str(c))), '')).strip().lower()
                    == ('sa dead' if hand_side == 'SH' else 'fp dead'))
            _discard_facts = gesture_match.window_facts(buf)
            _discard_count = sum(_discard_facts.hand_discards.values())
            _only_hand_kills = all(
                rec.get('op') == 'kill'
                and gesture_match.zone_key(rec.get('zone')) == ('hand', hand_side)
                for rec in (getattr(buf, 'records', None) or ()))
            if len(_gone106) == 1 or (len(_gone106) > 1 and _only_hand_kills
                                      and len(_gone106) == _discard_count):
                self.findings.append(('discard_identity_from_dump',
                                      ','.join(_gone106)[:100]))
                pids = [str(_map106[cid]) for cid in _gone106]
            elif len(_gone106) > 1:
                return _refuse106(
                    'discard_identity_ambiguous', _gone106[:3],
                    'that discard names no card identity and the '
                    'board shows several hand cards gone - use the '
                    'menu discard row')
            else:
                # the dump has not shown the kill yet: hold, no chat,
                # no undo - the pending buffer retries; its 25-tick
                # expiry stays the loud path
                if not any(k == 'discard_identity_pending'
                           for k, _ in self.findings[-6:]):
                    self.findings.append(('discard_identity_pending',
                                          'no pid on the kill; dump '
                                          'shows no hand card gone yet'))
                buf._awaits_discard_identity = True
                from .discard_wait_status import remember as _remember_discard_wait
                _remember_discard_wait(self, dp)
                return None
        _inv106 = {str(v): str(k) for k, v in
                   (self.view.card_pid_map or {}).items()}
        cids = {c for c in (_inv106.get(p) for p in pids)
                if c is not None}
        if not cids:
            if any(str(c).startswith('HU_') for c in _hand106()):
                buf._awaits_discard_identity = True
                from .discard_wait_status import remember as _remember_discard_wait
                _remember_discard_wait(self, dp)
                return None
            return _refuse106(
                'discard_identity_unresolved', pids[:4],
                "that discard's card identity is unknown - use the "
                'menu discard row')
        kept = [a for a in menu
                if str(getattr(a, 'action_type', ''))
                != 'discard_card'
                or str((getattr(a, 'params', None)
                        or {}).get('card_id')) in cids]
        # A physical identity must also agree with the public deck/owner
        # caption. A stale dump or contradictory sidecar cannot authorize
        # another side's card or a Strategy card captioned as Character.
        _facts = gesture_match.window_facts(buf)
        _owner = getattr(dp.get('side'), 'name', '')
        _decks = {deck for (owner, deck), count in _facts.hand_discards.items()
                  if owner == _owner and count and deck is not None}
        if any(owner != _owner and count
               for (owner, deck), count in _facts.hand_discards.items()):
            return None
        if _decks:
            import game_logic as _discard_gl
            kept = [a for a in kept if a.action_type != 'discard_card'
                    or _discard_gl.card_deck_of(str(a.params.get('card_id'))) in _decks]
        if not any(str(getattr(a, 'action_type', ''))
                   == 'discard_card' for a in kept):
            return _refuse106(
                'discard_identity_stale', sorted(cids)[:2],
                'that card is not discardable now - it may already '
                'be discarded; pick from the current menu')
        return kept

    def _bury_dead_candidate_asks(self):
        """BL-104 (fpgame13 J3): a window COMMIT must bury the losing
        candidates' funnel requests as thoroughly as refusal/expiry
        do (BL-70(3) ran only on window DEATH).  A draws ask probed
        by a LOSING candidate survived the successful commit,
        waiting_on kept demanding it, and the next physical draw was
        ingested as that stale ask's fulfillment - an extra card in
        the hand (King Brand's Men needed TWO draws to clear one owed
        redraw).  Once buried, a further draw hits the un-owed guard
        and refuses - the correct end-state.  Scope: card-draw asks
        only, and NEVER while an AI step is open (its own deduped ask
        may be the pending - the fast-chat wedge class)."""
        try:
            p = self.rand.pending() if self.rand is not None else None
        except Exception:
            p = None
        if p is None or getattr(self, 'ai_step_open', False):
            return
        if getattr(p, 'kind', '') not in ('draws', 'card_pick'):
            return
        self.rand._pending = None
        self.findings.append(
            ('dead_candidate_ask_buried',
             '%s %r' % (getattr(p, 'kind', '?'),
                        (getattr(p, 'kw', {}) or {}).get('ctx', ''))))

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
                if getattr(w, 'closed', None) == 'die_killed':
                    # FLD-6a (par5b_g2 ghost die #1): the client
                    # PHYSICALLY consumed the staged die (the window
                    # closed on its kill) - leaving the ENGINE die
                    # unspent minted a 5th action from a 4-die pool
                    # (and the dice resync then re-created the client
                    # die).  Ruling 1.230: a paid play COMPLETES,
                    # never refunded - debit the die.
                    self._debit_expired_die(w, side)
        self.rand.reset_feed()

    def _debit_expired_die(self, w, side):
        """FLD-6a: spend the engine die a client-killed, expired
        deferred window consumed.  Face-matched, loud on a miss -
        never a silent force of a different die."""
        letter = str(getattr(w, 'die_face', '') or '')
        name = next((k for k, v in gesture_match.FACE_LETTER.items()
                     if v == letter), None)
        for d in self.es.dice.get(side, ()):
            if d.used or d.current is None:
                continue
            if name is not None \
                    and getattr(d.current, 'name', None) == name:
                d.used = True
                self.findings.append(
                    ('deferred_die_spent',
                     '%s die=%s (client killed it; expiry debits '
                     'the engine, 1.230)' % (side.name, letter)))
                return
        self.findings.append(
            ('deferred_die_spend_miss',
             '%s die=%s: no unused engine die of that face - dice '
             'already parted, resync owed' % (side.name, letter)))

    def _retry_blocked(self):
        """New client randomness arrived: give a deferred chat action,
        then the pending buffer, then a deferred window, another
        attempt."""
        if self.forced_ai_rows_pending():
            return
        import ai
        if self.chat_deferred is not None:
            # BEFORE the dp check: the starved chat action's OWN
            # pending (e.g. defender_combat_card) is still open, so
            # the dp branch below would swallow the retry forever
            action, via = self.chat_deferred
            self.chat_deferred = None
            physical_norms = getattr(self, '_chat_deferred_norms', ())
            self._chat_deferred_norms = ()
            items = self._attempt_chat_action(action, via, physical_norms=physical_norms)
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

    # ---- pending phase-buffer -------------------------------------------
    def _rec_inert_here(self, rec):
        """Already-resolved table removal is physical cleanup, not a new use."""
        pw = PhaseWindow()
        pw.records.append(rec)
        facts = gesture_match.window_facts(pw)
        ids = set(facts.table_activations) | set(facts.table_discards)
        if ids:
            gone = {cid for pile in (getattr(self.es, 'discard_face_up', {}) or {}).values()
                    for cid in pile}
            if ids <= gone and not ids & gesture_match.table_card_ids(self.es):
                facts.table_activations = []
                facts.table_discards = []
                facts.hunt_reduction_cards = []
        return facts.is_empty() and not facts.unmapped

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
        """The pending menu constrained by physical identity and intent."""
        menu = dp['actions']
        if (not getattr(self, 'spectator', False)
                and dp['kind'] == 'declare' and dp['side'] is self.human_side):
            # Correcting a pawn/track to its unchanged endpoint is not a
            # declaration decline. FP OK (or an explicit Guide nomination)
            # settles the legal pass through _settle_declare_decline.
            # This menu also fences prefix recovery of the same gestures.
            menu = [action for action in menu if action.action_type != 'pass']
        menu = self._pending_gesture_intent_menu(dp, menu)
        menu = self._relentless_physical_menu(dp, menu)
        if dp['kind'] == 'discard':
            menu = self._discard_identity_filter(dp, menu)
            if menu is None:
                return []       # identity refusal also fences expiry prefixes
        return self._drop_unevidenced_random(dp, menu)

    def _relentless_physical_menu(self, dp, menu):
        """Constrain the pre-roll count using the player's actual paid hits.

        The round cursor does not write casualties to the public army until
        later, so general speculative matching cannot distinguish the count
        before asking for dice. This filter only removes impossible choices;
        the usual full delta matcher still verifies payment composition.
        """
        if getattr(self, 'spectator', False) or dp['side'] is not self.human_side:
            return menu
        pc = getattr(self.es, 'pending_combat', None) or {}
        ec = pc.get('effect_choice') or {}
        if ec.get('effect') != 'relentless_assault':
            return menu
        buf = self.pending_buf
        if buf is None or not buf.records:
            return []
        from board import sanitize
        region = sanitize(str(pc.get('attacker_region' if ec.get('owner') == 'atk'
                                     else 'defender_region'))).lower()
        facts = gesture_match.window_facts(buf)
        # Native downgrade first renders "kills Elite", then places its
        # Regular. Its Controls.304 witness must not be mistaken for a
        # completed two-hit whole-Elite removal during that gap.
        required = {}
        replacements = {}
        for rec in buf.records:
            one = PhaseWindow()
            one.records = [rec]
            rows = gesture_match.window_facts(one).armies
            for (where, kind), delta in rows.items():
                if delta > 0 and kind.endswith(':regular'):
                    replacements[(where, kind)] = replacements.get((where, kind), 0) + delta
                if (delta < 0 and kind.endswith(':elite')
                        and 'Controls.304' in ((rec.get('meta') or {}).get('tags') or ())):
                    key = (where, kind[:-5] + 'regular')
                    required[key] = required.get(key, 0) - delta
        if any(replacements.get(key, 0) < count for key, count in required.items()):
            return []
        paid = 0
        for (where, kind), delta in facts.armies.items():
            if not delta:
                continue
            if where != region or ':' not in kind:
                return []
            unit = kind.rsplit(':', 1)[-1]
            if unit not in ('regular', 'elite'):
                return []
            paid -= delta * (2 if unit == 'elite' else 1)
        counts = [int(a.params.get('count', 0)) for a in menu]
        if not counts or paid not in counts:
            return []
        if paid < max(counts) and not getattr(buf, '_relentless_roll_closed', False):
            return []
        return [a for a in menu if int(a.params.get('count', 0)) == paid]

    def _pending_gesture_intent_menu(self, dp, menu):
        """An incidental board update cannot answer a human's open choice.

        Explicit chat/panel choices commit through their own lanes. Physical
        no-card and stand choices have no board delta, so they cannot be
        inferred from a reveal, track update, or unrelated piece movement.
        A compulsory single-row menu still needs its own physical witness.
        """
        if getattr(self, 'spectator', False) or dp['side'] is not self.human_side:
            return menu
        facts = (gesture_match.window_facts(self.pending_buf)
                 if self.pending_buf is not None else None)
        result = []
        for action in menu:
            kind = action.action_type
            if kind in ('combat_card_none', 'combat_no_retreat'):
                continue
            if kind == 'choose_guide':
                if facts is None or facts.guide_set != action.params.get('companion'):
                    continue
            if kind == 'discard_card':
                if facts is None or not (facts.hand_discards or facts.cards_played):
                    continue
            if kind == 'card_option':
                choices = action.params.get('choices') or {}
                if isinstance(choices, dict) and 'gandalf_draw' in choices:
                    if not choices['gandalf_draw'] or facts is None or not facts.draws:
                        continue
            result.append(action)
        return result

    def _partial_hunt_corruption(self):
        """A contiguous, incomplete corruption-only gesture for this Hunt.

        This recognizes a physical prefix; it never commits a choice or
        assumes a missing companion casualty. Mixed gestures stay with the
        normal matcher, including valid casualty/corruption combinations.
        """
        if getattr(self, 'spectator', False):
            return None
        import ai
        dp = ai.describe_pending(self.es)
        if dp is None or dp['kind'] != 'hunt' or dp['side'] is not self.human_side:
            return None
        if not any(a.action_type == 'hunt_take_corruption' for a in dp['actions']):
            return None
        ph = getattr(self.es, 'pending_hunt', None) or {}
        buf = self.pending_buf
        if ph.get('phase') != 'hunt_sacrifice' or not (buf and buf.records):
            return None
        start = int(self.es.fellowship.corruption)
        damage = int(ph.get('damage', 0))
        target = min(12, start + damage)
        current = start
        for rec in buf.records:
            if rec.get('op') != 'move' or str(rec.get('obj', '')).lower() != 'corruption':
                return None
            endpoints = []
            for key in ('from', 'to'):
                zone = str(rec.get(key, '')).lower()
                prefix = 'fellowship track '
                if not zone.startswith(prefix) or not zone[len(prefix):].isdigit():
                    return None
                endpoints.append(int(zone[len(prefix):]))
            if endpoints[0] != current or not current < endpoints[1] < target:
                return None
            current = endpoints[1]
        return {'start': start, 'current': current, 'target': target,
                'damage': damage, 'remaining': target - current,
                'buffer': buf}

    def _partial_hunt_corruption_note(self):
        progress = self._partial_hunt_corruption()
        if progress is None:
            self._partial_corruption_voiced = None
            return []
        key = (id(progress['buffer']), self.n_commits_total,
               progress['current'], progress['target'])
        if getattr(self, '_partial_corruption_voiced', None) == key:
            return []
        self._partial_corruption_voiced = key
        self.findings.append(('hunt_corruption_partial',
                              '%d of %d damage expressed' %
                              (progress['current'] - progress['start'], progress['damage'])))
        return [('chat', 'prompt', chat_mod.partial_hunt_corruption_note(
            progress['damage'], progress['current'], progress['target'],
            progress['remaining']))]

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
        the client when the action is applied. A removal alone cannot
        establish that a random draw occurred. The client's built-in
        draw must have rendered first; ClientRandomness then supplies
        that exact recorded result to the engine's companion funnel.

        Keep the entry only when both the draw and the departing
        companion are evidenced. Otherwise the FP must say it - the
        chat lane accepts 'kill the guide' / 'random'.
        """
        if dp.get('kind') != 'hunt':
            return menu
        evidenced = False
        gone = {}
        if self.pending_buf is not None:
            try:
                f = gesture_match.window_facts(self.pending_buf)
                gone = getattr(f, 'comps_gone', None) or {}
                evidenced = bool(gone)
            except Exception:
                evidenced = False
        # A reveal placement/flip does not choose a casualty. The
        # deterministic guide option needs the guide's actual removal,
        # just as a random sacrifice needs its recorded draw and removal.
        # Explicit chat choices use their own direct commit lane.
        guide = getattr(getattr(self.es, 'fellowship', None), 'guide', None)
        guide_name = str(getattr(guide, 'name', guide) or '')
        if not guide_name or not gone.get(guide_name, 0):
            menu = [a for a in menu
                    if a.action_type != 'hunt_sacrifice_guide']
        if not getattr(self, 'spectator', False):
            # Retained feed is the evidence, not its speculative cursor:
            # an ambiguous guide-tie attempt may have read the draw, but
            # begin_attempt must replay it until the chain commits.
            draws = (getattr(self.rand, '_feed', None) or {}).get('companion', ())
            if evidenced and draws:
                # A random draw can select the current guide. Preserve
                # that choice and consume its recorded outcome instead
                # of relabeling it as a deterministic guide sacrifice.
                return [a for a in menu if a.action_type != 'hunt_sacrifice_guide']
            evidenced = False
        if evidenced:
            return menu
        kept = [a for a in menu
                if a.action_type not in self._CLIENT_DRAWN_TYPES]
        if len(kept) != len(menu):
            # BL-144: PROTOCOL-INFO, not a hold FROM the human - the
            # client-drawn random option is WITHHELD from gesture
            # matching (matching it would execute a companion draw
            # before the FP spoke); the FP says random, guide or
            # corruption
            self.findings.append(
                ('random_casualty_withheld',
                 '%d client-drawn option(s) withheld from gesture '
                 'matching - say random, guide, or corruption'
                 % (len(menu) - len(kept))))
            # BL-21(1)/BL-50: the announce is DUE here but SAID by the
            # server tick, which re-verifies the pending is still open
            # right before printing; the key carries the combat round
            # so a new round's choice re-announces
            _pc21 = getattr(self.es, 'pending_combat', None)
            _key21 = (int(getattr(self.es, 'turn', 0) or 0),
                      str(dp.get('kind')),
                      getattr(dp.get('side'), 'name', ''),
                      int((_pc21 or {}).get('rounds_fought', 0) or 0)
                      if isinstance(_pc21, dict) else -1)
            self._held_choice_due = _key21
        return kept or menu

    def _poke_pending_buffer(self):
        """Battle-gap records held while NO decision was open (the K1
        reveal gap - see _route): retry them the moment the open
        pending became the human's (the reveal opened the retreat
        decision the held drag answers)."""
        import ai
        if self.pending_buf is None or not self.pending_buf.records:
            return
        dp = ai.describe_pending(self.es)
        if dp is not None and self._pending_buffer_applies(dp):
            self._try_pending()

    def _try_physical_hand_discard(self, dp, menu):
        """Commit all explicitly selected group members in one transaction."""
        from .discard_transaction import HandDiscardTransaction
        import ai
        if getattr(self, 'spectator', False) or dp['side'] is not self.human_side:
            return None
        buf = self.pending_buf
        facts = gesture_match.window_facts(buf)
        count = sum(facts.hand_discards.values())
        if count < 2:
            return None
        zone = ('hand', 'FP' if self.human_side.name == 'FREE_PEOPLES' else 'SH')
        if any(rec.get('op') != 'kill'
               or gesture_match.zone_key(rec.get('zone')) != zone
               for rec in buf.records):
            return False  # Other effects/randomness remain their own decisions.
        ids = tuple(sorted({str((a.params or {}).get('card_id', ''))
                            for a in menu if a.action_type == 'discard_card'}))
        if len(ids) != count or any(not cid or cid.startswith('HU_') for cid in ids):
            return False
        transaction = HandDiscardTransaction(self.human_side, ids)
        probe = ai.clone_state(self.es)
        try:
            transaction(probe)
        except ValueError as error:
            self.findings.append(('group_discard_refused', str(error)))
            key = (self.n_commits_total, tuple(ids), str(error))
            if getattr(self, '_group_discard_refusal', None) != key:
                self._group_discard_refusal = key
                self._out_queue.append(('chat', 'refusal', chat_mod.refusal_illegal(
                    'that group exceeds your required hand-limit discards or includes a card you do not hold - undo it and discard only the required cards')))
            return False
        if not gesture_match._covers(facts, gesture_match.delta_facts(self.es, probe)):
            return False
        norms = [rec.get('norm', '') for rec in buf.records]
        out = self._attempt_chat_action(transaction, 'physical-hand-discard-group',
                                        physical_norms=norms, record_action=True)
        self.pending_buf = None
        self._buf_stall = None
        self._last_pending_match = gesture_match.MatchResult('match', chain=[transaction],
                                                             post=self.es)
        self._out_queue.extend(out)
        self.findings.append(('group_discard_committed', ','.join(ids)))
        return True

    def _siege_press_key(self, dp=None):
        if getattr(self, 'spectator', False):
            return None
        pc = getattr(self.es, 'pending_combat', None) or {}
        if (not pc.get('is_siege') or pc.get('phase') != 'combat_round_result'
                or pc.get('attacker_side') is not self.human_side):
            return None
        if dp is None:
            import ai
            dp = ai.describe_pending(self.es)
        if not dp or dp['side'] is not self.human_side or not any(
                a.action_type == 'siege_press' for a in dp['actions']):
            return None
        return (getattr(self.es, 'turn', None), self._cc_battle_serial,
                pc.get('attacker_region'), pc.get('defender_region'), pc.get('rounds_fought', 0))

    def _siege_press_waiting(self, dp=None):
        key = self._siege_press_key(dp)
        return key is not None and getattr(self, '_siege_press_intent', None) != key

    def _siege_press_reminder(self):
        key = (self._siege_press_key(), self.n_commits_total)
        if getattr(self, '_siege_press_prompted', None) == key:
            return []
        self._siege_press_prompted = key
        return [('chat', 'prompt', chat_mod.siege_press_note())]

    def _pending_rng_signature(self):
        buf = self.pending_buf
        pc = getattr(self.es, 'pending_combat', None) or {}
        cursor = (pc.get('_round_resume') or {}).get('cursor')
        payment = (pc.get('effect_choice'), tuple(getattr(cursor, key, None) for key in
                   ('stage', 'pending', 'ra_remaining', 'ra_paid', 'pre_remaining', 'pre_paid')))
        return (id(self.es), self.n_commits_total,
                repr(tuple(pc.get(key) for key in
                           ('phase','rounds_fought','alloc_stage','atk_hits','def_hits'))),
                repr(payment),
                tuple(repr(record) for record in getattr(buf, 'records', ())))

    def _proven_pending_random_call(self):
        buf = self.pending_buf
        if (buf is not None and getattr(buf, '_proven_starve_signature', None)
                == self._pending_rng_signature()):
            return getattr(buf, '_proven_starve_call', None)
        return None

    def pending_random_call(self):
        """Outstanding request owned by the current physical transaction."""
        proven = self._proven_pending_random_call()
        if proven is not None:
            return proven
        if self.pending_buf is not None:
            pc = getattr(self.es, 'pending_combat', None) or {}
            if (pc.get('phase') == 'casualty_allocation'
                    or (pc.get('effect_choice') or {}).get('effect') in (
                        'onslaught_payment', 'relentless_assault_payment', 'pre_combat_casualty')):
                # A speculative casualty branch may demand dice even when
                # the observed payment is incomplete or has another meaning.
                return None
            retained = getattr(self.pending_buf, '_relentless_starve_call', None)
            if retained is not None:
                return retained
        pending = self.rand.pending() if self.rand is not None else None
        if pending is None and self.deferred is not None:
            pending = getattr(self.deferred[0], '_starve_call', None)
        return pending

    def _try_pending(self):
        """Match the buffered phase records against the open pending's
        menu; commit on full coverage.  Returns True when the pending
        cascade fully settled."""
        if self.forced_ai_rows_pending():
            return False
        import ai
        recovery_rng = None
        if getattr(self, '_combat_recovery_active', False):
            from .combat_recovery import randomness_snapshot
            recovery_rng = randomness_snapshot(self.rctl)
        if self.pending_buf is not None:
            self.pending_buf._proven_starve_call = None
            self.pending_buf._proven_starve_signature = None
            self.pending_buf._proven_starve_state = None
        dp = ai.describe_pending(self.es)
        from .siege_withdrawal_gestures import handle_pending as handle_siege_withdrawal
        siege_result = handle_siege_withdrawal(self, dp)
        if siege_result is not None:
            return siege_result
        from .victor_completion import handle_pending as handle_victor_completion
        victor_result = handle_victor_completion(self, dp)
        if victor_result is not None:
            return victor_result
        if dp is None:
            if self.pending_buf is not None and self.pending_buf.records \
                    and (isinstance(getattr(self.es, 'pending_combat',
                                            None), dict)
                         or self.human_cc_committed):
                # K1 reveal gap: the buffer holds battle-gap records
                # awaiting the reveal - do NOT discard them
                return False
            self.pending_buf = None
            return True
        from .card_completion import empty_close
        if self.pending_buf is None or (not self.pending_buf.records
                and not empty_close(self.es, self.pending_buf)):
            return False
        if self._siege_press_waiting(dp):
            self._last_pending_match = gesture_match.MatchResult('hold', note='say press before committing a siege continuation')
            self._out_queue.extend(self._siege_press_reminder())
            return False
        # Corruption alone below the full amount is incomplete. The
        # speculative matcher may otherwise explain the missing damage
        # by inventing a guide casualty the player has not expressed.
        if self._partial_hunt_corruption() is not None:
            return False
        # REVEAL BEFORE THE MENU IS READ (USER 2026-08-20: "scouts is
        # nonfunctional ... it still waits for my decision"): a HU_
        # placeholder has no identity, so describe_pending could only
        # offer combat_card_none for the human's combat-card decision
        # - swap placeholders for the buffer's named cards first (as
        # _on_window does for gesture windows); reveals are tracked on
        # the buffer so a failed match rolls them back (_unreveal).
        self.pending_buf.combat_cards_open = self._combat_cards_open()
        self._reveal_buffer_cards(self.pending_buf)
        if getattr(self.pending_buf, '_reveals', None):
            dp = ai.describe_pending(self.es) or dp
        _menu106b = self._pending_menu(dp)
        if dp['kind'] == 'discard':
            grouped = self._try_physical_hand_discard(dp, _menu106b)
            if grouped is not None:
                return grouped
        if str(dp.get('kind', '')) == 'card_option' \
                and not getattr(self, 'spectator', False):
            # EX-8: the table-kill payment pair by identity (dump)
            _menu106b = self._table_kill_payment_filter(dp, _menu106b)
            if _menu106b is None:
                return False
        from .crebain_gestures import pending_discard_window
        match_window = pending_discard_window(self, self.pending_buf)
        if match_window is None:
            self._last_pending_match = gesture_match.MatchResult(
                'hold', note='Flocks discard still needs its public card, Fellowship progress, and paying die evidence')
            return False
        if match_window is not self.pending_buf:
            # Its public card removal proves discard, before randomness can
            # be requested for the unchosen keep option's different Hunt.
            _menu106b = [a for a in _menu106b if a.action_type == 'card_option'
                         and (a.params.get('choices') or {}).get('crebain_discard') is True]
        from .fellowship_reveal_gestures import pending_window as reveal_placement_window
        match_window = reveal_placement_window(self, match_window)
        from .casualty_gestures import live_context
        casualty = live_context(self.es, dp)
        match_window._hold_ambiguous_elite_prefix = (
            not getattr(self, 'spectator', False)
            and dp['side'] is self.human_side
            and casualty is not None and casualty['side'] is self.human_side)
        res = gesture_match.match_window(
            self.es, dp['side'], match_window,
            menu=_menu106b, pin=self._attempt_pin(),
            rng=self.rctl,
            advance_tolerance=not getattr(self, 'spectator', False),
            fs_prefilter=not getattr(self, 'spectator', False))
        from .decision_endpoint import completed, recover_visible_choice
        if res.status in ('none', 'ambiguous') and completed(match_window):
            res = recover_visible_choice(self, dp, match_window) or res
        res = self._resolve_decision_endpoint(match_window, res)
        if match_window is not self.pending_buf and getattr(res, 'endpoint_wait', False):
            self.pending_buf._endpoint_revision = match_window._endpoint_revision
            self._tag_hold(self.pending_buf, 'decision_endpoint')
        if recovery_rng is not None and res.status != 'match' and not res.starved:
            # A rejected retry inside explicit recovery must not open a new
            # RNG step before the independently proved board choice runs.
            from .combat_recovery import restore_randomness
            restore_randomness(self.rctl, recovery_rng)
        self._last_pending_match = res
        from .partial_casualty_status import remember as _remember_partial_casualty
        _remember_partial_casualty(self, res)
        if (res.status == 'hold' and res.starved
                and not getattr(res, 'proven_starve_conflict', False)
                and getattr(res, 'proven_starve_call', None) is not None):
            # The matcher proved the COMPLETE physical payment and unanimous
            # continuation demand. Nested RNG backtracking may erase the
            # global pending slot; keep this request on its exact transaction.
            self.pending_buf._proven_starve_call = res.proven_starve_call
            self.pending_buf._proven_starve_signature = self._pending_rng_signature()
            self.pending_buf._proven_starve_state = getattr(res, 'proven_starve_state', None)
        if ((getattr(self.es, 'pending_combat', None) or {}).get(
                'effect_choice', {}).get('effect') == 'relentless_assault'):
            self.pending_buf._relentless_starve_call = (
                getattr(res, 'starve_call', None) if res.starved else None)
        if getattr(res, 'invalid_guards_recruit', False):
            key = (self.n_commits_total, res.note)
            if getattr(self, '_guards_recruit_refusal', None) != key:
                self._guards_recruit_refusal = key
                self.findings.append(('guards_recruit_refused', res.note))
                self._out_queue.append(('chat', 'refusal', res.note))
            return False
        if res.status != 'match':
            return False    # incomplete buffer / rng still owed: wait
        buf = self.pending_buf
        norms = [r.get('norm', '') for r in buf.records]
        if self._commit_chain(res.chain, norms,
                              'pending:%s' % dp['kind'],
                              gesture=buf):
            self.pending_stats['matched'] += 1
            self._bury_dead_candidate_asks()
            self.pending_buf = None
            if dp['kind'] == 'guide_change' and self.phase == 'guide':
                self._out_queue.extend(self._complete_fellowship_guide())
            self._out_queue.extend(self._retry_holds('pending resolved'))
            return True
        # the commit failed: the buffer's reveal-swaps must leave the
        # engine hand again, or the consumed placeholder budget
        # poisons every later menu build (see _unreveal)
        self._unreveal(buf)
        if getattr(self, '_combat_recovery_active', False) and getattr(self, '_combat_recovery_failed', False):
            from .combat_recovery import restore_randomness
            restore_randomness(self.rctl, recovery_rng)
            self.pending_buf = buf
            return False
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
        if not getattr(self, 'spectator', False):
            # Corpus reconstruction spans historical decisions. Live input
            # is one current interaction: never salvage an arbitrary record
            # prefix and discard the rest of the player's completed choice.
            return self._try_pending()
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
            if self._siege_press_waiting(dp):
                self._out_queue.extend(self._siege_press_reminder())
                return False
            if self._partial_hunt_corruption() is not None:
                return False
            recs = buf.records
            menu = self._pending_menu(dp)
            hit_k = None
            for k in range(len(recs), 0, -1):
                shim = PhaseWindow()
                shim.records = recs[:k]
                shim.combat_cards_open = getattr(
                    buf, 'combat_cards_open', None)
                from .casualty_gestures import live_context
                casualty = live_context(self.es, dp)
                shim._hold_ambiguous_elite_prefix = (
                    not getattr(self, 'spectator', False)
                    and dp['side'] is self.human_side
                    and casualty is not None and casualty['side'] is self.human_side)
                # A later action window is not confirmation of a whole-Elite
                # loss. "done" authenticated the original entire buffer only;
                # it cannot authorize a shorter casualty prefix or its tail.
                shim._casualty_closed = (k == len(recs)
                                        and getattr(buf, '_casualty_closed', False))
                res = gesture_match.match_window(
                    self.es, dp['side'], shim, menu=menu,
                    rng=self.rctl, stop_kinds=(),
                    fs_reset_relax=True,
                    advance_tolerance=not getattr(self, 'spectator',
                                                  False),
                    fs_prefilter=not getattr(self, 'spectator',
                                             False))
                if k == len(recs):
                    # Keep the full-buffer proof for status/idle handling.
                    # A shorter candidate neither describes all retained
                    # evidence nor replaces its unfinished-payment guard.
                    self._last_pending_match = res
                    from .partial_casualty_status import remember
                    remember(self, res)
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
            buf._casualty_closed = False
            buf._elite_completion_prompted = False
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
        if dp is not None and not (buf and buf.records) \
                and getattr(self, 'spectator', False):
            # CORPUS CENSUS ONLY (R5 / USER 2026-08-28 self-repair
            # doctrine): a LIVE human decision is never decided by
            # absence - it stays open, _expire_pending_live names it
            # and the window that arrived over it is HELD with the
            # owed-decision reminder (_on_window).  Spectator: no
            # gesture rendered for this pending = the pending's own
            # NO-OP - the unique candidate whose observable delta is
            # EMPTY (absence-is-the-choice, same class as the
            # 0-eye/declare-decline absences); committed only when
            # exactly ONE empty-delta candidate exists (never guess
            # between distinct no-ops).
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
            # grants or upstream cascade); left open it JAMS every
            # later window.  Force the least-wrong settle - a
            # PLACEHOLDER discard toward the client's reality - one
            # loud row per forced card, then REPLAY any hostage
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
            # pending: it will never resolve from records and left
            # open it JAMS every later window.  Force the first
            # commit-able option with draws STUBBED and d6 STUBBED to
            # natural 1s (the client resolved the card invisibly; the
            # force must not need randomness the log never carries) -
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
        if self._siege_press_waiting(dp):
            self._out_queue.extend(self._siege_press_reminder())
            return
        from .casualty_gestures import live_context
        casualty = live_context(self.es, dp)
        casualty_open = (n_buf and casualty is not None
                         and casualty['side'] is self.human_side)
        if self.combat_recovery_open() or casualty_open:
            label = 'CASUALTY' if casualty_open else 'COMBAT'
            message = ('%s DECISION HELD (%s): unmatched gestures and observed '
                       'rolls retained; complete the current choice or undo it' % (label, why))
            if message not in self.alarms:
                self.alarms.append(message)
                self.findings.append(('combat_pending_retained', message))
            return
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
        if dp['kind'] == 'guide_change':
            return self._maybe_announce_pending()
        key = (dp['kind'], dp['side'].name, self.n_commits_total)
        if getattr(self, '_reminded_pending', None) == key:
            return []
        self._reminded_pending = key
        kind = str(dp['kind'])
        try:
            text = chat_mod.waiting_on(kind,self._pending_words(dp))
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
        if (self.phase == 'guide' and dp['kind'] in ('guide_change', 'eye')
                and getattr(self.es, '_eye_allocation_phase', False)
                and self.human_side.name == 'FREE_PEOPLES'):
            # Finishing FP's Guide window must never skip Shadow's next
            # decision or leave the UI in guide with the core in eyes.
            items = self._on_chat({'body': 'fp ok', 'nick': 'panel'})
            return items, ('Fellowship phase complete' if self.phase == 'eyes'
                           else 'finish the current Fellowship board gesture first')
        if (dp['kind'] == 'declare' and self.pending_buf is not None
                and self.pending_buf.records):
            return [], ('a declaration is already on the board - finish it '
                        'with fp ok, or use Resync Board; it cannot be declined unseen')
        if dp['side'] is not self.human_side:
            return [], 'the AI owns this decision; there is no human choice to skip'
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
                        and (not any(bool(v) for v in ch.values())
                             or ch.get('action') == 'none'):
                    # wholly-falsy = the classic decline; an explicit
                    # {'action': 'none'} (Nazgul Strike either/or) is
                    # a decline too ('none' is a truthy STRING) - USER
                    # 2026-08-24
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
        '_eye_allocation_phase', '_reveal_place_due', '_hunt_reveal_pending',
        'pending_army_move_die', 'pending_muster_die',
    )

    def combat_recovery_open(self):
        """Include the uncommitted opening and physical tail of a battle."""
        if self.forced_ai_rows_pending():
            return True
        if isinstance(getattr(self.es, 'pending_combat', None), dict) \
                or getattr(self, 'human_cc_committed', False):
            return True
        if (self.human_physical_work_pending() or self.echo_pending()) and self.commits:
            if isinstance(getattr(self.commits[-1][0], 'pending_combat', None), dict):
                return True
        if self.rand is not None and self.rand.step_open:
            for key in self.rand._journal:
                kind = str(key[0]) if isinstance(key, (tuple, list)) and key else ''
                if kind == 'army_attack' or kind.startswith(('combat_', 'siege_', 'casualty_')):
                    return True
        return False

    def _run_resync_transaction(self, operation):
        """Roll back a recovery commit, including failed emission bookkeeping."""
        import copy
        from .combat_recovery import randomness_snapshot, restore_randomness
        # Keep service identities and the existing buffer; copy only driver
        # bookkeeping that a normal commit/sweep may change before returning.
        services = [self, self.es, self.view, self.feed, self.seg, self.rand,
                    self.rctl, self.chat, self.sidecar, self.pending_buf]
        # ScenarioDriver keeps the spectator parser module after its live
        # flip. It is an immutable collaborator, not recoverable game state.
        services.append(getattr(self, '_rr', None))
        services.extend(self.hand_models.values())
        services.extend(self.commits)   # existing ledger entries are immutable
        memo = {id(value): value for value in services if value is not None}
        saved = copy.deepcopy(vars(self), memo)
        view_saved = copy.deepcopy(vars(self.view), memo)
        hand_saved = [(model, copy.deepcopy(vars(model), memo))
                      for model in self.hand_models.values()]
        buffer = self.pending_buf
        buffer_saved = copy.deepcopy(vars(buffer), memo) if buffer is not None else None
        random_saved = randomness_snapshot(self.rctl)
        previous_guard = getattr(self, '_resync_commit_transaction', None)
        self._resync_commit_transaction = True
        try:
            if operation() is False:
                raise ValueError('the legal choice could not be committed')
            return True, None
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            vars(self).clear()
            vars(self).update(saved)
            vars(self.view).clear()
            vars(self.view).update(view_saved)
            if buffer is not None:
                vars(buffer).clear()
                vars(buffer).update(buffer_saved)
            for model, fields in hand_saved:
                vars(model).clear()
                vars(model).update(fields)
            restore_randomness(self.rctl, random_saved)
            return False, str(error)
        finally:
            if previous_guard is None:
                vars(self).pop('_resync_commit_transaction', None)
            else:
                self._resync_commit_transaction = previous_guard

    def _commit_recovered_combat(self, recovery, evidence, label='pending:combat-resync'):
        """Publish one recovered choice only while its board/feed proof holds."""
        from .repair_evidence import same_dump
        def commit():
            if not same_dump(self, evidence):
                raise ValueError('physical board or native feed changed before recovery commit')
            if recovery.status == 'pending':
                self.pending_buf = recovery.window
                settled = self._try_pending()
                if (not settled and (self._combat_recovery_failed
                        or self._proven_pending_random_call() is None)):
                    raise ValueError('recovered payment no longer proves its pending client result')
            else:
                import ai
                pending = ai.describe_pending(self.es)
                norms = [record.get('norm', '') for record in recovery.window.records]
                if not self._commit_chain(recovery.match.chain, norms,
                                          label, gesture=recovery.window):
                    raise ValueError('the legal combat choice could not be committed')
                self.pending_stats['matched'] += 1
                self.pending_buf = None
                self._bury_dead_candidate_asks()
                if pending and pending['kind'] == 'guide_change' and self.phase == 'guide':
                    self._out_queue.extend(self._complete_fellowship_guide())
            if not same_dump(self, evidence):
                raise ValueError('physical board or native feed changed during recovery commit')
            return True
        accepted, reason = self._run_resync_transaction(commit)
        return (True, recovery.note) if accepted else (
            False, 'combat recovery not committed: %s; original battle and evidence retained' % reason)

    def _recover_retained_combat(self):
        """Route available evidence, then reconstruct only proved missing moves."""
        from .combat_recovery import plan
        from .repair_evidence import dump_signature, same_dump
        items = []
        report = ['unfinished combat retained; cards, costs, rolls and action die remain committed']
        if getattr(self, '_combat_recovery_active', False):
            return items, report + ['combat recovery is already processing the current feed']
        if self._feed_seq < len(self.feed.lines):
            return items, report + ['prefetched client feed is still routing; no board-derived choice committed']
        self._combat_recovery_active = True
        self._combat_recovery_failed = False
        try:
            self._buf_stall = None
            self._last_blocked_kind = None
            self._last_pending_match = None
            self._announced_pending = None
            self._repair = None
            items.extend(self._consume_inject_acks())
            sc = self.sidecar
            if sc is not None:
                sc.poll()
            # Normal parse/sidecar/segment/route, preserving unmatched records.
            # A dump may already include a gesture still waiting in this file.
            ready = self._poll_routable_feed()
            route_failed = False
            for seq, raw in ready:
                self._feed_seq = seq
                try:
                    items.extend(self._ingest_line(raw, sc))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    route_failed = True
                    message = 'feed line %d: %s' % (self._feed_seq, error)
                    self.findings.append(('combat_recovery_route_error', message))
                    self.alarms.append(message)
            if getattr(self, '_sidecar_wait_seq', None) is not None:
                return items, report + ['native metadata is still arriving; combat evidence retained']
            if route_failed or self._combat_recovery_failed:
                return items, report + ['existing combat evidence could not be committed; buffered choices and rolls retained']
            self._retry_blocked()
            if self._combat_recovery_failed:
                return items, report + ['existing combat choice could not be committed; its buffer and rolls retained']
            if self.deferred is not None or self.chat_deferred is not None:
                return items, report + ['the retained combat transaction still needs its original client result']
            if not self.view.dump_path:
                return items, report + ['no stable board dump; combat retained without board adoption and still needs its physical choice']
            evidence = dump_signature(self)
            header = evidence['header']
            if header is None:
                return items, report + ['board dump has no feed-position header; no missing gesture inferred']
            self._check_dump_session(header)
            position_hold = self._dump_position_hold(header)
            if position_hold or self._feed_seq != len(self.feed.lines) or self.feed._buf:
                return items, report + [position_hold or 'client feed is incomplete; no missing gesture inferred']
            if not self.view.refresh_dump() or not same_dump(self, evidence):
                return items, report + ['board dump incomplete or changing; current combat choice retained']
            physical = dict(self.view.piece_region)
            from .recovery_board import validate as validate_recovery_board
            validate_recovery_board(self, physical)
            from .corsairs_presentation import projected as project_corsairs
            physical = project_corsairs(self, physical)
            if self.forced_ai_rows_pending():
                recovered, notes = self._recover_forced_ai_rows(physical, evidence)
                return items + recovered, report + notes
            recovery = plan(self, physical)
            if not same_dump(self, evidence) or self._feed_seq != len(self.feed.lines):
                return items, report + ['physical board or native feed changed while matching; no missing gesture committed']
            if recovery.status in ('match', 'pending'):
                accepted, note = self._commit_recovered_combat(recovery, evidence)
                report.append(note)
                self.findings.append(('combat_reset_recovered' if accepted else 'combat_reset_held', note))
                if not accepted:
                    return items, report
                if accepted and recovery.status == 'pending':
                    pending = self._proven_pending_random_call()
                    if pending is not None:
                        items.append(('rng_ask', pending))
            else:
                report.append(recovery.note)
            import ai
            markers = self._client_marker_states()
            rows = client_board.diff_engine_vs_dump(
                self.es, physical, dynamic_types=self.view.dynamic_types,
                marker_states=markers, ai_module=ai)
            remaining = [row for row in (rows or ()) if not client_board.is_staging_row(row)]
            pending = self._proven_pending_random_call()
            stopped = getattr(self.pending_buf, '_proven_starve_state', None)
            stopped_rows = (client_board.diff_engine_vs_dump(
                stopped, physical, dynamic_types=self.view.dynamic_types,
                marker_states=markers, ai_module=ai)
                if pending is not None and stopped is not None else None)
            # Report only the same measured board the recovery transaction
            # proved. An owed choice is different from an unreconciled board;
            # a stopped RNG continuation is still an uncommitted transaction.
            if not same_dump(self, evidence) or self._feed_seq != len(self.feed.lines):
                return items, report + ['physical board or feed changed before recovery verification']
            if stopped_rows is not None and not any(
                    not client_board.is_staging_row(row) for row in stopped_rows):
                self._last_resync_adoption = {
                    'accepted': True, 'complete': False, 'retained_combat': True,
                    'waiting_for': 'client_roll' if pending.kind == 'd6' else 'client_result'}
            elif rows is not None and not remaining and not (
                    self.pending_buf is not None and self.pending_buf.records):
                self._last_resync_adoption = {
                    'accepted': True, 'complete': True, 'retained_combat': True,
                    'waiting_for': 'combat_step'}
            if remaining:
                report.append('physical board still differs in %d place(s); first: %s' % (len(remaining), remaining[0]))
        except (KeyboardInterrupt, SystemExit, DriverHalt):
            raise
        except BaseException as error:
            report.append('combat recovery not committed: %s; current battle and choices retained' % error)
        finally:
            self._combat_recovery_active = False
            self._combat_recovery_failed = False
            if self._out_queue:
                items.extend(self._out_queue)
                self._out_queue = []
        return items, report

    def _recover_fellowship_choice(self, for_completion=False):
        """Use a complete current board to recover one visible pending choice."""
        from .choice_recovery import (plan, declaration_unchanged,
                                      unchanged_declaration_correction)
        from .repair_evidence import dump_signature, same_dump
        items = []
        board_matches = None
        def finish(rows):
            return (rows, board_matches) if for_completion else rows
        try:
            evidence = dump_signature(self)
            header = evidence['header']
            if header is None:
                return finish([('note', 'choice recovery needs a board/feed position header')])
            self._check_dump_session(header)
            hold = self._dump_position_hold(header)
            if hold or self._feed_seq != len(self.feed.lines) or self.feed._buf:
                return finish([('note', hold or 'choice recovery waits for complete native feed routing')])
            if not self.view.refresh_dump() or not same_dump(self, evidence):
                return finish([('note', 'choice recovery waits for a stable complete board')])
            from .recovery_board import validate as validate_recovery_board
            validate_recovery_board(self, self.view.piece_region)
            if for_completion:
                board_matches = declaration_unchanged(
                    self, self.view.piece_region, self._client_marker_states())
            result = plan(self, dict(self.view.piece_region), self._client_marker_states())
            if result.status == 'match':
                label = ('pending:declare' if any(action.action_type == 'fellowship_declare'
                          for action in result.match.chain) else
                         'pending:discard' if all(action.action_type == 'discard_card'
                          for action in result.match.chain) else 'pending:fellowship-resync')
                accepted, reason = self._commit_recovered_combat(
                    result, evidence, label=label)
                if accepted:
                    board_matches = True
                    if label == 'pending:discard':
                        self._ensure_declare_armed()
                items.append(('note', reason))
                self.findings.append(('fellowship_reset_recovered' if accepted
                                      else 'fellowship_reset_held', reason))
            elif (for_completion
                    and unchanged_declaration_correction(
                        self, self.view.piece_region, self._client_marker_states())):
                count = len(self.pending_buf.records)
                def retire_correction():
                    if not same_dump(self, evidence):
                        raise ValueError('physical board or native feed changed before correction acknowledgement')
                    self.pending_buf = None
                    self._last_pending_match = None
                    self._buf_stall = None
                    if not same_dump(self, evidence):
                        raise ValueError('physical board or native feed changed during correction acknowledgement')
                    return True
                accepted, reason = self._run_resync_transaction(retire_correction)
                if accepted:
                    board_matches = True
                    reason = 'acknowledged %d Fellowship pose correction record(s); declaration choice retained' % count
                    self.findings.append(('fellowship_pose_correction', count))
                else:
                    board_matches = None
                    reason = 'Fellowship correction retained: %s' % reason
                items.append(('note', reason))
            else:
                items.append(('note', 'choice recovery: %s' % result.note))
        except (KeyboardInterrupt, SystemExit, DriverHalt):
            raise
        except BaseException as error:
            items.append(('note', 'choice recovery unavailable: %s; pending retained' % error))
        finally:
            if self._out_queue:
                items.extend(self._out_queue)
                self._out_queue = []
        return finish(items)

    def reset_and_resync(self, politics_from_engine=False, hand_turn=False):
        """Retry unfinished combat; otherwise adopt a stable recovery board.

        A battle retains its current choice, observed rolls, paid cards/die and
        physical obligations. Existing evidence is retried without adopting
        casualties into an old hit budget. Outside combat, unresolved decisions
        retain their evidence; only a settled action permits board adoption.
        Returns (items, report), explicitly reporting retained or adopted state.
        """
        self._last_resync_adoption = {'accepted': False, 'complete': False}
        self._last_reset_preserved_combat = self.combat_recovery_open()
        if self._last_reset_preserved_combat:
            items, report = self._recover_retained_combat()
            self.findings.append(('combat_reset_retry', report[0]))
            return items + [('chat', 'prompt', chat_mod.combat_recovery_note())], report
        report = []
        items = []
        # Establish a usable recovery board before discarding any decision,
        # feed record or random result. Failed resync is a no-op for gameplay.
        try:
            items.extend(self.resync_recover(
                drain_feed=True, politics_from_engine=politics_from_engine))
        except DriverHalt:
            raise
        except Exception as error:
            report.append('resync failed: %r' % error)
            return items, report
        if not getattr(self, '_last_resync_adoption', {}).get('complete'):
            report.append('resync incomplete; pending decisions and recovery evidence retained')
            return items, report
        # Adoption may open a rules-owed guide or hand-limit choice. A matching
        # board does not answer it, so leave that decision to its normal owner.
        self._turn_start_draws_open = False
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
        self._staged_event_window = None
        self._uncommitted_card_selection = None
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
        report.append('resynced to the client board%s' %
                      (' (political track kept engine-side)'
                       if politics_from_engine else ''))
        # Board recovery preserves alternation. Only an explicitly requested
        # handoff may transfer the next action to the AI.
        if hand_turn and getattr(self, '_last_resync_adoption', {}).get('complete'):
            self.last_actor = self.human_side
            report.append('turn handed to the AI')
        else:
            report.append('turn order retained')
        return items, report

    @staticmethod
    def _effect_sig116(st):
        """BL-116(c): a cheap effect signature over every surface a
        paid action could touch (dice excluded).  None = unreadable
        (callers must not judge)."""
        try:
            from board import sanitize as _sn
            import game_state as _gs
            _pc = getattr(st, 'pending_combat', None)
            _po = getattr(st, 'pending_card_option', None)
            return (
                tuple(sorted((_sn(str(k)), len(a.units),
                              len(getattr(a, 'leaders', ()) or ()),
                              # USER manual game 2026-08-28: the
                              # Aragorn crowning changes a leader's
                              # IDENTITY (Strider -> Aragorn), not the
                              # count - the signature carries leader
                              # names
                              tuple(sorted(
                                  str(getattr(l, 'name', '') or '')
                                  for l in (getattr(a, 'leaders', ())
                                            or ()))))
                             for k, a in st.armies.items())),
                tuple(len(st.hands[s].hand)
                      for s in (_gs.Side.FREE_PEOPLES,
                                _gs.Side.SHADOW)),
                (int(getattr(st.fellowship, 'progress', 0) or 0),
                 int(getattr(st.fellowship, 'corruption', 0) or 0),
                 bool(getattr(st.fellowship, 'hidden', True)),
                 _sn(str(getattr(st.fellowship, 'region', '') or ''))),
                tuple(sorted((str(getattr(n, 'value', n)), int(v))
                             for n, v in (getattr(st.political,
                                                  'boxes', {})
                                          or {}).items())),
                tuple(sorted((str(getattr(n, 'value', n)), bool(v))
                             for n, v in (getattr(st.political,
                                                  'active', {})
                                          or {}).items())),
                len(getattr(st, 'hunt_box', ()) or ()),
                (str((_pc or {}).get('phase')),
                 (_pc or {}).get('rounds_fought')),
                str((_po or {}).get('card_id')),
            )
        except Exception:
            return None

    def _paid_noop_check(self, pre, post, label, actions=()):
        """BL-116(c) (the tracker's action-effect witness): any PAID
        commit (die-used count grew) whose effect signature is
        UNCHANGED is a paid no-op - it would have caught the
        pool-empty muster and the Celeborn silently-nothing play.
        'skip' and 'pass' are the two rules-legal paid no-ops."""
        if any(str(getattr(a, 'action_type', '')) in ('skip', 'pass')
               for a in (actions or ())):
            return
        try:
            _d0 = sum(1 for s in pre.dice
                      for d in pre.dice[s] if d.used)
            _d1 = sum(1 for s in post.dice
                      for d in post.dice[s] if d.used)
        except Exception:
            return
        if _d1 <= _d0:
            return
        s0 = self._effect_sig116(pre)
        s1 = self._effect_sig116(post)
        if s0 is not None and s0 == s1:
            self.alarms.append('PAID NO-OP: %s consumed a die with '
                               'no board effect (BL-116c)'
                               % str(label)[:60])
            self.findings.append(('paid_noop', str(label)[:60]))

    def _post_commit_markers(self, action, physical):
        """BL-82 (shgame10 E1): ONE writer for every action_type-keyed
        post-commit marker, called from BOTH commit routes - the
        chat/pick lane (_attempt_chat_action) and the physical chain
        route (_commit_chain, per chain action) - so no input lane can
        ever skip a demand.  `physical`: the commit carries client
        lines (a window / pending-buffer chain).  Markers that exist
        only because the lane left NO client line - the owed cc
        physical reveal (BL-41/54/81), the owed discard kill (BL-62) -
        are skipped for physical commits: the client board already
        shows those effects, and demanding them again would double
        them.  The K1 commitment flag is lane-independent."""
        _at = getattr(action, 'action_type', '')
        from .combat_card_cleanup import note_committed
        note_committed(self, action)
        if _at == 'combat_card_hidden_commit':
            self.human_cc_committed = True
            return
        if _at in ('combat_card_reveal', 'combat_card_cancel'):
            self.human_cc_committed = False
            self._cc_choose_preamble = None
            return
        if _at == 'play_event':
            # BL-121: a played card that landed on the ENGINE table
            # owes its Board-slot seating REGARDLESS of lane (the
            # client's play gesture leaves it in the transient
            # Current-Card zone); set the demand AT the commit; the
            # seat performs
            _cid121 = str(((getattr(action, 'params', None) or {})
                           .get('card_id')) or '')
            try:
                _pcs121 = getattr(self.es, 'persistent_cards', None)
                _onb121 = _cid121 and _pcs121 is not None and any(
                    str(getattr(c, 'card_id', '')) == _cid121
                    for c in (list(getattr(_pcs121, 'fp_cards', ()))
                              + list(getattr(_pcs121, 'sh_cards',
                                             ()))))
            except Exception:
                _onb121 = False
            if _onb121:
                _tp121 = self.view.card_pid_map.get(_cid121)
                _seated121 = (_tp121 is not None and _zone_key(
                    str(self.view.piece_region.get(str(_tp121), '')))
                    in _TABLE_SLOT_CODE)
                # A card reveal can temporarily overlay the view's location
                # during matching. The subsequent observed Board-slot move
                # remains independently recorded in the table-seat ledger.
                from cards import get_card
                _card121 = get_card(_cid121)
                _seat121 = self.table_seats.get(table_card_key(
                    _card121.card_name if _card121 else ''), {})
                _seated121 = _seated121 or (
                    _seat121.get('slot') in _TABLE_SLOT_NAME)
                if _seated121 and getattr(self, '_table_seat_owed', None) == str(_tp121):
                    self._table_seat_owed = None
                if _tp121 is not None and not _seated121:
                    self._table_seat_owed = str(_tp121)
                    _said121 = getattr(self, '_table_seat_said',
                                       set())
                    _said121.add(str(_tp121))
                    self._table_seat_said = _said121
                    self._out_queue.append(
                        ('chat', 'prompt',
                         chat_mod.table_seat_note()))
        if physical:
            return
        if _at == 'combat_card_play':
            # BL-41/BL-54: a CHAT-committed combat card has no
            # client-chosen piece - at reveal the human owes the
            # physical chooser play, consumed as a REVEAL ONLY (the
            # commit already applied the card; the stored cid
            # discriminates against a second apply).  BL-81: entries
            # live on a per-battle claim-once ledger ('*' = cid
            # unknown, matches any reveal once).
            _cid81 = ((getattr(action, 'params', None) or {})
                      .get('card_id'))
            _led81 = getattr(self, '_cc_reveal_owed', None) or {}
            _led81[str(_cid81) if _cid81 else '*'] = 'owed'
            self._cc_reveal_owed = _led81
            self._cc_reveal_origins[str(_cid81) if _cid81 else '*'] = self._cc_battle_serial
        elif _at == 'discard_card':
            # BL-62: an engine-side discard (pick/chat route) leaves
            # the PHYSICAL card in the client hand - demand + publish
            # the owed kill (the menu route kills first and never gets
            # here)
            _dcid62 = (getattr(action, 'params', None)
                       or {}).get('card_id')
            _dpid62 = self.view.card_pid_map.get(str(_dcid62))
            if _dpid62:
                self._discard_owed_pid = str(_dpid62)
                self._out_queue.append(
                    ('chat', 'prompt',
                     chat_mod.discard_board_note()))

    def _commit_human(self, action, via='gesture'):
        """One chat-committed human decision, under the SAME attempt/
        advance treatment as windows (WAVE-H): the old pin()-scoped
        apply ran at advance-depth 0 where the funnels are INERT - a
        defender's 'nc' that fires the combat round would have rolled
        ENGINE dice silently.  A starved apply defers (retried on each
        rng-bearing record; ask surfaced for live_server to order)."""
        if self.forced_ai_rows_pending():
            return []
        if (not getattr(self, 'spectator', False)
                and getattr(action, 'action_type', '') == 'siege_press'):
            # "Press" states continuation intent, not which eligible Elite
            # the human pays. Keep the real decision open until the physical
            # removal/replacement identifies that choice.
            key = self._siege_press_key()
            if via != 'chat-press' or key is None:
                return [('chat', 'prompt', chat_mod.siege_press_note())]
            self._siege_press_intent = key
            paid = bool(self.pending_buf and self.pending_buf.records and self._try_pending())
            return [('chat', 'prompt', chat_mod.siege_press_note(armed=True, paid=paid))]
        if (getattr(action, 'action_type', '') == 'siege_no_press'
                and self.pending_buf and self.pending_buf.records
                and gesture_match.window_facts(self.pending_buf).armies):
            return [('chat', 'prompt', chat_mod.siege_press_note(blocked_stop=True))]
        return self._attempt_chat_action(action, via)

    def _attempt_chat_action(self, action, via, physical_norms=(), record_action=False):
        """`action`: an engine Action (ai.apply_action on a clone) or
        a CALLABLE taking the clone to mutate (the K1 hidden-commit /
        reveal engine calls) - one transactional commit path."""
        if self.forced_ai_rows_pending():
            return []
        import ai
        from . import emission as em_mod
        from . import randomness as rnd_mod
        partial_corruption = (self._partial_hunt_corruption()
                              if getattr(action, 'action_type', '') == 'hunt_take_corruption'
                              else None)

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
            except rnd_mod.NeedClientCall as call:
                call._physical_base_state = self.es
                # the decision needs client randomness that has not
                # rendered yet (combat rolls...): defer, never invent
                self.chat_deferred = (action, via)
                self._chat_deferred_norms = physical_norms
                return [('note',
                         'chat action deferred (%s): awaiting client '
                         'randomness' % via)]
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                raise DriverHalt('human %s action failed to apply: %r'
                                 % (via, e))
        self.chat_deferred = None
        self._chat_deferred_norms = ()
        pre = self.es
        # Only this exact, successfully completed Hunt choice consumes
        # the prefix. Other buffered gestures must never be discarded.
        completed_prefix = (partial_corruption is not None
                            and post.fellowship.corruption == partial_corruption['target'])
        prefix_norms = ([r.get('norm', '') for r in partial_corruption['buffer'].records]
                        if completed_prefix else [])
        self._snapshot(prefix_norms + list(physical_norms), via, post=post,
                       action=action if record_action else None)
        self.es = post
        if completed_prefix:
            self.pending_buf = None
            self._buf_stall = None
            self._partial_corruption_voiced = None
            self.findings.append(('hunt_corruption_chat_completed',
                                  '%d -> %d' % (partial_corruption['current'],
                                                partial_corruption['target'])))
        for hm in self.hand_models.values():
            hm.es = post
        self.hands.es = post
        # K1 flag + owed-demand markers: BL-82's ONE writer (shared
        # with the chain route; a deferred retry lands here too)
        self._post_commit_markers(action, physical=False)
        self._paid_noop_check(pre, post, 'chat:%s' % via, [action])
        if self.rand is not None:
            exc = self.rand.unconsumed()
            self.rand.step_commit()
            if any(exc.values()):
                self.findings.append(
                    ('unconsumed_rng', '%r after chat:%s'
                     % ({k: v for k, v in exc.items() if v}, via)))
                if getattr(self, 'ai_step_open', False):
                    # an AI step is MID-FLIGHT (live_server maintains
                    # the flag): the 'unconsumed' randomness may be
                    # exactly what its outstanding directive is owed
                    # (the !CALL is deduped, never re-ordered) - keep
                    # the feed, note loudly
                    self.findings.append(
                        ('unconsumed_rng_kept_for_open_step', via))
                else:
                    self.rand.reset_feed()
        # N7: the search chain's NEGATIVE verdict is stated
        # (found-nothing must be distinguishable from effect-skipped)
        _ch7 = (getattr(action, 'params', None) or {}).get('choices')
        _cid7 = str((getattr(action, 'params', None)
                     or {}).get('card_id', '')).lower()
        if isinstance(_ch7, dict) and _ch7.get('_naz_done') \
                and 'search' in _cid7 \
                and bool(getattr(pre.fellowship, 'hidden', True)) \
                == bool(getattr(post.fellowship, 'hidden', True)):
            self._out_queue.append(('chat', 'prompt',
                                    chat_mod.search_no_find_note()))
        # G2: a round that resolved inside THIS apply announces its
        # hit counts (engine statement - see emission.round_hits_chat)
        out = [('chat', k, t) for k, t in
               em_mod.round_hits_chat(post, self.human_side, pre=pre)]
        # chat commits have NO gesture rows to subtract - the whole
        # delta is engine-resolved (forced allocations, capture
        # markers) and sweeps here
        out.extend(self.post_commit_sweep(
            pre, post, label='chat:%s' % via, actions=[action],
            track_sources=({'252': partial_corruption['current']}
                           if completed_prefix else None),
            # BL-137: decision-born demands (a card_option naming its
            # regions) are rebase-exempt
            rebase_exempt=(str(getattr(action, 'action_type', ''))
                           == 'card_option')))
        self._reconcile_due = True
        # Players may reveal and place the pawn before answering Hunt
        # damage in chat. That chat choice opens the placement decision;
        # retry the retained physical gesture now, against only that
        # decision. Do not infer a casualty from the earlier reveal or
        # discard its location when the damage decision changes kind.
        reveal_dp = ai.describe_pending(self.es)
        if (self.pending_buf is not None and self.pending_buf.records
                and reveal_dp is not None
                and reveal_dp.get('kind') == 'reveal_place'
                and reveal_dp.get('side') is self.human_side):
            reveal_facts = gesture_match.window_facts(self.pending_buf)
            if (reveal_facts.fel_moves
                    or any(kind == 'reveal' for _, kind in reveal_facts.fel_flips)):
                self._try_pending()
        elif getattr(action, 'action_type', '') in (
                'hunt_guide_separate', 'hunt_gollum_reveal', 'hunt_gollum_no_reveal'):
            # A Hobbit can already be on its chosen map region while the new
            # Guide's Hunt ability is unanswered. The reply now opens that
            # retained placement; do not require another physical drag.
            self._poke_pending_buffer()
        out.extend(self._maybe_announce_pending())
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

    def _political_recon(self):
        """ENGINE -> CLIENT political reconciler (USER trust ruling
        2026-08-24: "trust engine for political track, client
        otherwise").  Compares each nation's marker position in the
        client dump against the engine box and emits ONE corrective
        marker drag per mismatched nation.  An in-flight ledger keyed
        (pid -> target code) stops re-emission while the client has
        not yet applied a correction (dump staleness would otherwise
        double-walk the marker).  Requires a fresh view dump (the
        caller refreshed it)."""
        import re as _re
        from emitter import render
        from log_replay import POLITICAL_MARKERS
        items = []
        inflight = getattr(self, '_pol_recon_inflight', None)
        if inflight is None:
            inflight = self._pol_recon_inflight = {}
        pr = getattr(self.view, 'piece_region', {}) or {}
        for pid, nation in POLITICAL_MARKERS.items():
            zone = str(pr.get(str(pid), '') or '')
            m = _re.match(r'^Political (\d)$', zone)
            if not m:
                continue
            client_box = 3 - int(m.group(1))       # engine numbering
            eng_box = self.es.political.boxes.get(nation)
            if eng_box is None:
                continue
            tgt = 'Game.%d' % (2000 + int(eng_box))
            cur = 'Game.%d' % (2000 + client_box)
            if client_box == int(eng_box):
                inflight.pop(str(pid), None)
                continue
            if inflight.get(str(pid)) == tgt:
                continue                       # correction in flight
            # a recent feed line already moves this marker to the
            # target (the AI's own play emission, not yet applied in
            # the dump): the recon must not double-walk it
            _tail = getattr(self.feed, 'lines', [])[-40:]
            if any(str(pid) in l and tgt in l for l in _tail):
                continue
            inflight[str(pid)] = tgt
            items.append(('line', render(
                'move', {'nick': self.nick, 'pieces': [str(pid)],
                         'frm': cur, 'to': tgt}, local=False)))
            self.findings.append(
                ('political_recon',
                 '%s client box %d -> engine %d'
                 % (getattr(nation, 'value', nation), client_box,
                    int(eng_box))))
        # Native br counters report their actual active face through H.k().
        # The mask stays U on both faces. A pending toggle is released only
        # after an active-face dump witnesses application.
        from .client_protocol import read_marker_states
        dump_path = getattr(self.view, 'dump_path', None)
        faces = read_marker_states(dump_path) if dump_path else {}
        walked = getattr(self, '_pol_face_walked', None)
        if walked is None:
            walked = self._pol_face_walked = set()
        for pid, nation in POLITICAL_MARKERS.items():
            pid = str(pid)
            face = faces.get(pid)
            if face == 1:
                walked.discard(pid)
                continue
            if face != 0 or not self.es.political.active.get(nation, False):
                continue
            if pid in walked:
                continue
            walked.add(pid)
            items.append(('line', render('flip', {'nick': self.nick,
                                                  'pieces': [pid]}, local=False)))
            self.findings.append(('political_face_recon',
                                  '%s: passive native face -> engine active'
                                  % getattr(nation, 'value', nation)))
        return items

    def _guide_box_recon(self):
        """AI-FP guide-box reconciler (USER bug 2026-08-24: 'when AI
        fp loses their guide, the new guide does not get moved to the
        guide box').  When the AI plays Free Peoples and the engine's
        guide is not the piece sitting in the client's Guide box, emit
        the drag placing the right companion there (and the displaced
        one back into the Fellowship box).  Human-FP games are the
        human's own gesture space - untouched."""
        import log_replay as lr
        import game_state as gs
        from emitter import render
        if self.ai_side is not gs.Side.FREE_PEOPLES:
            return []
        g = getattr(self.es.fellowship, 'guide', None)
        if g is None:
            return []
        gname = str(getattr(g, 'name', ''))
        gpid = next((p for p in lr.COMPANIONS
                     if gname in set(lr.names_for_pid(str(p)) or ())),
                    None)
        if gpid is None:
            return []
        pr = getattr(self.view, 'piece_region', {}) or {}
        in_box = [str(p) for p, z in pr.items()
                  if _zone_key(z) in _GUIDE_ZONES]
        if str(gpid) in in_box and len(in_box) == 1:
            self._guide_recon_inflight = None      # box correct
            return []
        # BL-35: any guide-box pid with a just-emitted move/kill means
        # the dump has not settled - HOLD the whole recon (walking a
        # piece the client is mid-applying is the resurrection class)
        import time as _t35
        _rec35 = getattr(self, '_recent_emit_pids', {}) or {}
        # BL-119: the NEW guide's own pid counts too (its walk toward
        # the box may be rendered and not yet landed in the dump)
        if any(_t35.time() - _rec35.get(str(p), 0) <= 15.0
               for p in in_box + [str(gpid)]):
            return []
        # BL-68: the in-flight flag is a (pid, ts) TTL entry, never a
        # bare forever-flag (a swallowed render must not leave the
        # guide box empty all game); expiry alarms and re-emits
        _gin68 = getattr(self, '_guide_recon_inflight', None)
        if isinstance(_gin68, tuple) and _gin68[0] == str(gpid):
            if _t35.time() - _gin68[1] < 20.0:
                return []                          # correction in flight
            self.alarms.append('GUIDE RECON STALE: the %s walk never '
                               'confirmed - re-emitting' % gname)
        elif _gin68 == str(gpid):
            # legacy bare flag from an older code path: treat as stale
            self.alarms.append('GUIDE RECON STALE (bare flag) - '
                               're-emitting')
        # BL-35: only a LIVING fellowship member may be displaced
        # guide-box -> fellowship box (a dead companion's stale pid in
        # the guide zone must never be resurrected client-side)
        comps33 = set()
        try:
            for c33 in (getattr(self.es.fellowship, 'companions', ())
                        or ()):
                comps33.add(str(getattr(c33, 'name', c33)))
        except Exception:
            comps33 = set()
        items = []
        for p in in_box:
            if str(p) != str(gpid):
                _nm33 = set(lr.names_for_pid(str(p)) or ())
                if comps33 and not (_nm33 & comps33):
                    self.findings.append(('guide_recon_skip_dead',
                                          str(p)))
                    continue
                items.append(('line', render(
                    'move', {'nick': self.nick, 'pieces': [str(p)],
                             'frm': 'Game.1999', 'to': 'Game.1998'},
                    local=False)))
        frm = 'Game.1998'
        items.append(('line', render(
            'move', {'nick': self.nick, 'pieces': [str(gpid)],
                     'frm': frm, 'to': 'Game.1999'}, local=False)))
        # BL-68: in-flight is set only AFTER the lines are queued,
        # as a TTL entry - never before a render that could fail
        self._guide_recon_inflight = (str(gpid), _t35.time())
        self.findings.append(('guide_box_recon',
                              '%s -> Guide box' % gname))
        return items

    # ---- QUIESCENCE + CENSUS LANE (BRCMP design, 2026-08-29) ------------
    # A board dump may be JUDGED only when nothing the driver knows of
    # is in flight (design 1: ONE predicate, one owner - the witness
    # cadence and the quiet-tick reconcile both ask it).  A disagreement
    # becomes a DIVERGENCE only after it has held, identical, across
    # quiescent reads for the confirm window (design 2).  The lane then
    # LOGs and ASKs through the self-repair protocol (LOG -> ASK); it
    # NEVER adopts on its own (design 3: the USER's consent rule - the
    # human's 'yes', the panel resync (BL-147) and the lenient path are
    # the consent forms).
    _RENDER_TTL = 15.0        # s an emitted pid may lag in the dump (BL-35)
    _CLIENT_PACE_S = 1.6      # AiOpp poll period (ai.pace = 1600 ms)
    _CENSUS_CONFIRM_S = 10.0  # identical quiescent rows this long = real

    def _census_confirm_seconds(self):
        """K in seconds, derived from the client's poll pace: the
        whole reads that cover the confirm window, times the pace."""
        import math
        pace = float(getattr(self, 'client_pace_s', None)
                     or self._CLIENT_PACE_S)
        return max(self._CENSUS_CONFIRM_S,
                   math.ceil(self._CENSUS_CONFIRM_S / pace) * pace)

    @staticmethod
    def _demand_text(e):
        """One echo-ledger entry as the demand it stands for."""
        k = str(e.get('kind', '') or '')
        n = int(e.get('n', 1) or 1)
        what = e.get('tkey') or e.get('name') or k or 'piece'
        if k.endswith('_move'):
            # SQF-10: a banked RELOCATION demand names both ends
            return ('drag %d x %s %s -> %s (%s)'
                    % (n, what, e.get('region', '?'),
                       e.get('to_region', '?'), e.get('side', '?')))
        verb = 'drag in' if k.endswith('_in') else 'remove'
        return '%s %d x %s at %s (%s)' % (verb, n, what,
                                          e.get('region', '?'),
                                          e.get('side', '?'))

    def quiescence_holds(self, now=None, *, allow_engine_pending=False,
                         allow_recent_render=False):
        """[reason, ...] why the board dump may NOT be judged right
        now; [] = quiescent.  Every in-flight state the driver knows
        of: an open segmenter window, a deferred window / chat
        decision, a pending buffer, a refusal restore in flight, an AI
        line younger than the render TTL, an owed die/card kill, an
        echo-ledger demand of EITHER side, an owed track / pawn /
        siege-seat / table-seat / discard, an open battle, an open
        engine decision, a self-repair in progress, a resync this
        tick. Paired checkpoint capture may allow an idle engine decision
        and replace the render-age heuristic with a fresh, unchanged board
        proof and full census. Outstanding physical work still disqualifies it."""
        import time as _tq
        import ai as _aiq
        now = _tq.time() if now is None else now
        holds = []
        if getattr(self, '_allocation_roll_wait', None) is not None:
            holds.append('observed action-dice roll awaiting its native board export')
        if getattr(self, '_corsairs_arrival', None):
            holds.append('Corsairs arrival awaiting completed defense choice')
        from .temporary_combat_figures import waiting
        for binding in waiting(self.view,self.es):
            holds.append('owed: return temporarily parked '+binding['name'])
        if getattr(self.seg, 'window', None) is not None:
            holds.append('open window (die staged, not killed)')
        if any(getattr(window, '_hold_cond', None) == 'decision_endpoint'
               for window, _result in self.holds):
            holds.append('completed decision awaiting its board export')
        # the gesture / echo lanes' own published predicates (BRLANE
        # handoff 2026-08-29): read, never re-derived, when they exist
        _go = getattr(self, 'gesture_open', None)
        if callable(_go) and _go():
            holds.append('human gesture open (lane)')
        _ee = getattr(self, 'echo_expect', None) or {}
        if _ee:
            holds.append('owed: AI emission echo (pid %s)'
                         % ' '.join(sorted(str(p) for p in _ee)[:6]))
        if getattr(self, '_post_commit_tail', ()):
            holds.append('owed: deferred combat consequences awaiting physical completion')
        if self.forced_ai_rows_pending():
            holds.append('owed: committed AI board correction awaiting RESYNC')
        if self.deferred is not None:
            holds.append('deferred window (awaiting client randomness)')
        if self.chat_deferred is not None:
            holds.append('deferred chat decision (awaiting client '
                         'randomness)')
        if self.pending_buf is not None:
            holds.append('pending buffer open')
        rw = getattr(self, 'restore_watch', None)
        if rw is not None and getattr(rw, 'armed', False):
            holds.append('refusal restore in flight')
        rec = getattr(self, '_recent_emit_pids', None) or {}
        young = sorted(str(p) for p, ts in rec.items()
                       if now - float(ts or 0) <= self._RENDER_TTL)
        if young and not allow_recent_render:
            holds.append('AI line(s) rendering (pid %s)'
                         % ' '.join(young[:6]))
        if self.owed_die_kills:
            holds.append('owed: %d die kill(s) at chain close'
                         % len(self.owed_die_kills))
        if getattr(self, 'owed_card_kills', None):
            holds.append('owed: %d card discard(s) after round resolution'
                         % len(self.owed_card_kills))
        for e in list(self.echo_ledger or ()):
            holds.append('owed: %s' % self._demand_text(e))
        for pid, (tgt, _dl) in dict(getattr(self, '_track_owed', None)
                                    or {}).items():
            holds.append('owed: track marker %s -> Track %s' % (pid, tgt))
        po = getattr(self, '_pawn_owed', None)
        if po:
            holds.append('owed: fellowship pawn walk to %s' % (po[0],))
        sso = getattr(self, '_siege_seat_owed', None)
        if sso:
            holds.append('owed: seat the garrison in the %s Stronghold '
                         'box' % sso)
        tso = getattr(self, '_table_seat_owed', None)
        if tso:
            holds.append('owed: seat table card %s on a board slot' % tso)
        dop = getattr(self, '_discard_owed_pid', None)
        if dop:
            holds.append('owed: remove discarded card %s from the hand'
                         % dop)
        if (not allow_engine_pending
                and isinstance(getattr(self.es, 'pending_combat', None), dict)):
            holds.append('open battle')
        try:
            dp = _aiq.describe_pending(self.es)
        except Exception as e:
            dp = None
            holds.append('pending unreadable: %r' % (e,))
        if dp is not None and not allow_engine_pending:
            holds.append('open %s decision (%s)'
                         % (dp.get('kind'),
                            getattr(dp.get('side'), 'name', dp.get('side'))))
        r = self._repair
        if r is not None and not (r.get('kind') == 'census'
                                  and r.get('phase') == 'offered'):
            holds.append('self-repair in progress (%s)' % r.get('phase'))
        rs = float(getattr(self, '_resync_ts', 0) or 0)
        if rs and now - rs <= self._RENDER_TTL:
            holds.append('resync this tick (client rendering it)')
        _ph = self._dump_position_hold(self._dump_header())
        if _ph:
            holds.append(_ph)
        return holds

    def quiescent(self):
        """True when a board dump may be judged right now."""
        return not self.quiescence_holds()

    def _dump_header(self):
        """The candidate jar's '#SEQ' header of the current dump file
        (client_protocol, BRJAVA-2/7), or None: the shipped jar writes
        none (today's behaviour, unchanged) / no dump path."""
        from . import client_protocol as _cph
        dp = getattr(self.view, 'dump_path', None)
        return _cph.read_dump_header(dp) if dp else None

    def _dump_position_hold(self, hdr):
        """BRJAVA-2: a dump whose header is BEHIND or AHEAD of the feed
        lines this driver has consumed describes another instant - no
        verdict on it (a held reason, never a row).  None = at
        position or header-less."""
        from . import client_protocol as _cpp
        if hdr is None:
            return None
        n = len(getattr(self.feed, 'lines', ()) or ())
        d = _cpp.feed_position_delta(hdr, n)
        if d is None:
            return ('dump not at feed position: dump %s carries no feed '
                    'position' % (hdr.get('dump'),))
        if d != 0:
            return ('dump not at feed position: dump %s is %s the '
                    'consumed feed by %d line(s)'
                    % (hdr.get('dump'), 'behind' if d < 0 else 'ahead of',
                       abs(d)))
        return None

    def _check_dump_session(self, hdr):
        """BRJAVA-7: the first LIVE header fixes the session nonce; a
        later header from another session is a LOUD halt - no verdict,
        no adoption of another game's artefacts.  session == 0 = no
        live loop yet (the AiBoot boot dumps): accepted, never
        adopted as the nonce."""
        from . import client_protocol as _cps
        if hdr is None:
            return
        sess = int(hdr.get('session', 0) or 0)
        if sess <= 0:
            return
        nonce = getattr(self, 'session_nonce', None)
        if nonce is None:
            self.session_nonce = sess
            return
        if not _cps.session_matches(hdr, nonce):
            msg = ('client session %d != server session %d - a new game '
                   'was started under this server; restart Play_vs_AI.bat'
                   % (sess, int(nonce)))
            self.findings.append(('client_session_mismatch', msg[:120]))
            raise DriverHalt(msg)

    def _granted_die_pids(self, pr):
        """The die pids the engine has GRANTED: the base table plus as
        many bonus pids as the engine's per-side count says are in
        play (bound pids and pids in the pool / hunt / Current Action
        first, then the dead-zone ones), so a never-granted bonus die
        parked in a Dead/Reinf box is not read as spent (BRCMP-8: the
        comparator's shield, fed on the live path at last)."""
        import emitter as _emg
        import game_state as _gsg
        dmap = getattr(self.view, 'die_id_map', {}) or {}
        out = set()
        for side, ids, nbase in (
                (_gsg.Side.FREE_PEOPLES, _emg._FP_DIE_IDS, 4),
                (_gsg.Side.SHADOW, _emg._SH_DIE_IDS, 7)):
            base = [str(p) for p in ids[:nbase]]
            bonus = [str(p) for p in ids[nbase:]]
            out.update(base)
            extra = max(0, len(self.es.dice.get(side, [])) - nbase)
            granted = []
            for p in bonus:
                zk = _zone_key(str(pr.get(p, '') or ''))
                if p in dmap or zk in ('fpdice', 'sadice',
                                       'currentaction') or 'hunt' in zk:
                    granted.append(p)
            for p in bonus:
                if len(granted) >= extra:
                    break
                zk = _zone_key(str(pr.get(p, '') or ''))
                if p not in granted and zk in ('used', 'fpdead', 'sadead'):
                    granted.append(p)
            out.update(granted)
        return out

    def _client_marker_states(self):
        from .client_protocol import read_marker_states
        path = getattr(self.view, 'dump_path', None)
        return read_marker_states(path) if path else None

    def census_rows(self, extra=None):
        """ONE census read of the current dump against the engine ->
        (board_rows, staging_rows), the live shields fed: granted dice,
        revocations, the BOOT dump for PIECE CONSERVATION, the force
        pools.  Never mutates the engine.  Callers refresh the dump."""
        import ai as _aic
        pr = getattr(self.view, 'piece_region', {}) or {}
        from .corsairs_presentation import projected as project_corsairs
        pr = project_corsairs(self, pr)
        if getattr(self, '_initial_dump', None) is None \
                and len(pr) >= client_board.MIN_DUMP_PIECES:
            self._initial_dump = dict(pr)
        rows = client_board.diff_engine_vs_dump(
            self.es, pr, dynamic_types=self.view.dynamic_types,
            ai_module=_aic, marker_states=self._client_marker_states(),
            initial_dump=getattr(self, '_initial_dump', None),
            granted_pids=self._granted_die_pids(pr),
            pid_to_cid={str(pid): str(cid) for cid, pid in self.view.card_pid_map.items()},
            # no live writer: an honest empty set (BRCMP)
            revoked_pids=getattr(self, '_revoked_die_pids', None)) or []
        rows = [str(r) for r in rows] + [str(r) for r in (extra or ())]
        if self.phase == 'actions' and self.view.dump_path:
            from .client_protocol import read_action_die_faces
            from .action_die_zones import zone as die_zone
            try:
                faces = read_action_die_faces(self.view.dump_path)
            except OSError:
                faces = {}
            for pid, (side, index) in self.view.die_id_map.items():
                pool = 'fp dice' if side.name == 'FREE_PEOPLES' else 'sa dice'
                if die_zone(pr.get(str(pid), '')) != pool or str(pid) not in faces:
                    continue
                die = self.es.dice[side][index]
                actual = gesture_match.FACE_LETTER.get(getattr(die.current, 'name', None))
                if not die.used and actual is not None and actual != faces[str(pid)]:
                    rows.append('dice face %s: engine=%s client=%s' % (pid, actual, faces[str(pid)]))
        board = [r for r in rows if not client_board.is_staging_row(r)]
        staging = [r for r in rows if client_board.is_staging_row(r)]
        return board, staging

    def _census_row_lines(self, board, age):
        """Every row printed WITH its reason (BRCMP-6/-10):
        <family> <region/box> <side>: engine=<n> client=<n> pids=<...>
        | owed: <demand or none> | since <s>s.  FLA-7 (par5 g4#2): a
        row whose zone an OPEN demand explains names that demand -
        'owed: none' beside a live demand invited the adoption that
        erased the engine's paid recruits."""
        demands = [(str(e.get('region', '') or ''),
                    self._demand_text(e))
                   for e in (self.echo_ledger or ())]
        _sso = getattr(self, '_siege_seat_owed', None)
        if _sso:
            from board import sanitize as _snR
            demands.append((_snR(str(_sso)),
                            'seat the garrison in the %s Stronghold '
                            'box' % _sso))
        out = []
        for r in board:
            head = str(r).split(':', 1)[0]
            owed = [d for reg, d in demands if reg and reg in head]
            row = r if 'pids=' in r else r + ' pids=-'
            out.append('%s | owed: %s | since %ds'
                       % (row, '; '.join(owed[:2]) or 'none',
                          int(age)))
        return out

    def _census_retract(self, why):
        """A clean quiescent read RETRACTS every alarm the census /
        witness / flap detector raised (alarms are events with a
        retraction, never a counter) and withdraws an open adoption
        offer.  Findings stay as the record."""
        items = []
        for attr in ('_census_alarm', '_witness_alarm', '_census_aging_alarm'):
            a = getattr(self, attr, None)
            if a:
                if a in self.alarms:
                    self.alarms.remove(a)
                    items.append(('note', 'census: clean again - '
                                          'retracted: %s' % str(a)[:90]))
                setattr(self, attr, None)
        for a in list(getattr(self, '_flap_alarms', None) or ()):
            if a in self.alarms:
                self.alarms.remove(a)
                items.append(('note', 'census: clean again - retracted: '
                                      '%s' % str(a)[:90]))
        self._flap_alarms = []
        r = self._repair
        if r is not None and r.get('kind') == 'census' \
                and r.get('phase') == 'offered':
            self._repair = None
            self.findings.append(('census_offer_withdrawn', why))
            items.append(('note', 'census: the divergence cleared - '
                                  'adoption offer withdrawn'))
        self._census_offer = None
        self._census_declined_sig = None
        self._census_confirmed_sig = None
        self._census_said_sig = None
        # FLA-9: a window held on the divergence is retried the moment
        # the boards agree again
        items.extend(self._retry_holds('divergence cleared'))
        return items

    def _census_confirmed(self, sig, board, lines, now, age):
        """The same BOARD rows held across quiescent reads for the
        confirm window: alarm once (retractable), file the divergence,
        and LOG -> ASK once per row set.  A vanished offer with no
        adoption is the human's 'no': the engine is kept, the rows are
        re-issued as the demand, and the same rows are never asked
        about again."""
        items = []
        if getattr(self, '_census_confirmed_sig', None) != sig:
            self._census_confirmed_sig = sig
            old = getattr(self, '_census_alarm', None)
            if old and old in self.alarms:
                self.alarms.remove(old)
            msg = ('BOARD DIVERGENCE CONFIRMED: %d row(s) held %.0fs '
                   'across %d quiescent reads (first: %s)'
                   % (len(board), age,
                      int(getattr(self, '_recon_sig_reads', 0) or 0),
                      board[0][:100]))
            self.alarms.append(msg)
            self._census_alarm = msg
            self.findings.append(('census_divergence_confirmed',
                                  '%d row(s): %s'
                                  % (len(board), board[0][:100])))
        offer = getattr(self, '_census_offer', None)
        if offer is not None and self._repair is None:
            self._census_offer = None
            self._census_declined_sig = offer.get('sig')
            self.findings.append(('census_adopt_declined',
                                  '%d row(s)' % len(board)))
            items.append(('note', 'census: engine kept (you declined '
                                  'the adoption) - the client should '
                                  'show: ' + '; '.join(lines[:6])))
            if offer.get('sig') == sig:
                return items
        if self._repair is None \
                and getattr(self, '_census_declined_sig', None) != sig:
            items.extend(self._propose_repair(
                'BOARD DIVERGENCE: %d row(s) held %.0fs across quiescent '
                'reads' % (len(board), age), kind='census', rows=lines))
            if self._repair is not None:
                self._census_offer = {'sig': sig, 'ts': now}
        return items

    def census_tick(self, extra_rows=None, source='reconcile'):
        """THE census lane (BRCMP design 1-3, 5).  Judges the current
        dump only when quiescent (else publishes WHY, once per reason
        text, never a row), classifies rows BOARD / STAGING, records
        the streak of the identical BOARD row set, retracts healed
        alarms and - once the same rows have held for the confirm
        window - LOG -> ASKs through _propose_repair.  NEVER adopts by
        itself.  Returns driver items.  Callers refresh the dump."""
        import time as _tc
        now = _tc.time()
        items = []
        # BRJAVA-7 first: another session's artefacts halt loudly
        self._check_dump_session(self._dump_header())
        holds = self.quiescence_holds(now)
        if holds:
            text = 'census held: ' + '; '.join(holds[:4])
            if len(holds) > 4:
                text += ' (+%d more)' % (len(holds) - 4)
            self._census_held = text
            if getattr(self, '_census_held_said', None) != text:
                self._census_held_said = text
                items.append(('note', text))
            return items
        self._census_held = None
        self._census_held_said = None
        try:
            board, staging = self.census_rows(extra_rows)
        except Exception as _e_census:
            # DR-5: a broken witness is NOT a clean board: the
            # reconcile stays due (re-read on the next quiet tick),
            # the stability streak is left untouched, and the failure
            # is loud (alarm + census row once per distinct error)
            _msg_c = ('RECONCILE CENSUS FAILED: %r - the board is NOT '
                      'known clean; auto-reconcile stays due'
                      % (_e_census,))
            if _msg_c not in self.alarms:
                self.alarms.append(_msg_c)
                self.findings.append(('reconcile_census_failed',
                                      repr(_e_census)[:100]))
                items.append(('note', _msg_c))
            return items
        self._last_quiescent_census = {'t': now, 'board': list(board),
                                       'staging': list(staging),
                                       'source': source}
        if staging:
            ssig = tuple(sorted(staging))
            if ssig != getattr(self, '_census_staging_said', None):
                self._census_staging_said = ssig
                items.append(('note', 'census staging (dice / deck / hand '
                                      'size - never a divergence): '
                                      + '; '.join(staging[:4])))
        else:
            self._census_staging_said = None
        if not board:
            self._reconcile_due = False
            self._recon_prev_sig = None
            self._recon_sig_since = None
            self._recon_sig_reads = 0
            items.extend(self._census_retract('clean quiescent read'))
            return items
        sig = tuple(sorted(board))
        if sig != getattr(self, '_recon_prev_sig', None):
            self._recon_prev_sig = sig
            self._recon_sig_since = now
            self._recon_sig_reads = 1
            self._census_said_sig = None
            r = self._repair
            if r is not None and r.get('kind') == 'census' \
                    and r.get('phase') == 'offered':
                # the rows CHANGED under an open offer: stale question
                self._repair = None
                self._census_offer = None
                self.findings.append(('census_offer_withdrawn',
                                      'rows changed'))
                items.append(('note', 'census: the divergence changed - '
                                      'adoption offer withdrawn'))
        else:
            self._recon_sig_reads = int(getattr(self, '_recon_sig_reads',
                                                0) or 0) + 1
        age = now - float(getattr(self, '_recon_sig_since', None) or now)
        lines = self._census_row_lines(board, age)
        if getattr(self, '_census_said_sig', None) != sig:
            self._census_said_sig = sig
            items.append(('note', 'census: %d board row(s) disagree (not '
                                  'yet a divergence - judged again on the '
                                  'next quiet tick)' % len(board)))
            items.extend(('note', ln) for ln in lines)
        if age >= self._census_confirm_seconds() \
                and int(getattr(self, '_recon_sig_reads', 0) or 0) >= 2:
            items.extend(self._census_confirmed(sig, board, lines, now,
                                                age))
        return items

    def reconcile_board_quiet(self):
        """USER 2026-08-24: "after every die kill, monitor ... to ensure
        0 divergences."  Called by the SERVER on a tick after a commit
        flagged `_reconcile_due`: refresh the dump, release the
        dump-confirmed demands, then run the census lane
        (census_tick: quiescence gate -> BOARD/STAGING rows with
        reasons -> confirm window -> LOG -> ASK; NEVER an adoption on
        its own, BRCMP design 3) and, on a quiescent tick only, walk
        the political markers and the AI-FP guide box to the ENGINE
        state (ENGINE truth).  Returns driver items."""
        if getattr(self, 'spectator', False):
            self._reconcile_due = False
            return []
        if not self.view.refresh_dump():
            return []
        items0 = self._enforce_owed_tracks()
        # BL-67: a HUMAN-played persistent card parked in the
        # transient Current-Card zone owes its Board-slot seating;
        # quiet ticks only (no battle - the zone is legitimately busy
        # mid-round); demanded once per pid, the seat performs
        _cur67 = ('currentfpcard'
                  if self.human_side.name == 'FREE_PEOPLES'
                  else 'currentsacard')
        _pr67 = getattr(self.view, 'piece_region', {}) or {}
        _said67 = getattr(self, '_table_seat_said', set())
        for _p67, _z67 in _pr67.items():
            if _zone_key(_z67) != _cur67 or str(_p67) in _said67:
                continue
            _said67.add(str(_p67))
            self._table_seat_said = _said67
            self._table_seat_owed = str(_p67)
            items0.append(('chat', 'prompt',
                           chat_mod.table_seat_note()))
            break
        # FLD-2 release (par5b g1#5 / g4#13+#15 3-for-3 / g5#11): a
        # PERFORMED seat shows the pid out of the transient
        # Current-Card zone (a Board slot, or killed by the card's
        # own option) - without this release the owed row gated the
        # census blind for the rest of the session (NO release path
        # existed; the demand aged forever).
        _tso2 = getattr(self, '_table_seat_owed', None)
        if _tso2 and _pr67:
            _z2 = _zone_key(str(_pr67.get(str(_tso2), '')))
            if _z2 != _cur67:
                self._table_seat_owed = None
                self.findings.append(
                    ('table_seat_released',
                     'card %s -> %s' % (_tso2, _z2 or 'gone')))
        from .siege_seating import waiting as seating_waiting
        seating_waiting(self)
        # BL-62 release: the owed discard's pid left the hand zone
        _dop62 = getattr(self, '_discard_owed_pid', None)
        if _dop62:
            _z62 = _zone_key(str((getattr(self.view, 'piece_region',
                                          {}) or {}).get(_dop62, '')))
            if _z62 not in ('fphand', 'sahand'):
                # O2: remember the released pid - the dump-confirm can
                # land BEFORE the kill record routes, and the claim
                # gate must still recognise the echo
                import time as _t62
                _rec62 = getattr(self, '_discard_recent62', None)
                if _rec62 is None:
                    _rec62 = self._discard_recent62 = {}
                _rec62[str(_dop62)] = _t62.time()
                self._discard_owed_pid = None
        items = self.census_tick(source='reconcile')
        _held8 = getattr(self, '_census_held', None)
        _owed_only8 = False
        if _held8 is not None:
            # FLD-8 (par5b_g2 C3 batching): a census held ONLY by
            # owed PIECE demands (the frozen owed-seat class) starved
            # the engine-truth political walk for whole turns - the
            # marker then batch-jumped at rollover.  An owed piece
            # demand does not touch the political track; only an
            # in-flight TRACK marker (or any non-owed hold: open
            # gesture/decision/battle/resync) still gates (F17).
            try:
                _h8 = self.quiescence_holds()
            except Exception:
                _h8 = ['unreadable']
            _owed_only8 = bool(_h8) and all(
                h.startswith('owed:') and 'track marker' not in h
                for h in _h8)
        if _held8 is None or _owed_only8:
            # ENGINE-truth walks only on a QUIESCENT tick: never over an
            # open gesture (the F17 double-walk class)
            items.extend(self._political_recon())
            items.extend(self._guide_box_recon())
        return items0 + items

    def _enforce_owed_tracks(self):
        """BL-2 (2026-08-25): while an engine-side track change awaits
        its rendered marker move, that track is NOT adoptable
        client->engine (the owed dict feeds adopt_tracks' hold).  When
        the dump confirms the box, release; past the deadline, alarm
        LOUDLY and re-emit the corrective drag (engine truth walks the
        client - never a silent adoption of the stale marker).
        Requires a fresh dump (the caller refreshed it)."""
        import re as _re
        import time as _time9
        from emitter import render as _crender
        items = []
        pr = getattr(self.view, 'piece_region', {}) or {}
        # F1: the owed PAWN move confirms/expires like the owed tracks
        # (dump 247 at the target -> released; expiry -> alarm +
        # re-emit from engine truth)
        _po8 = getattr(self, '_pawn_owed', None)
        if _po8:
            import time as _tp8
            from board import sanitize as _szp8
            _tgt8, _dl8 = _po8
            _z8 = _szp8(str(pr.get('247', '') or '')).strip().lower()
            if _z8 and _szp8(_tgt8) in _z8:
                self._pawn_owed = None
            elif _tp8.time() > _dl8:
                _rec8 = getattr(self, '_recent_emit_pids', {}) or {}
                if _tp8.time() - _rec8.get('247', 0) > 15.0:
                    self.alarms.append(
                        'OWED PAWN MOVE STALE: 247 should show %s, '
                        'client shows %r - re-walking (engine truth)'
                        % (_tgt8, _z8))
                    try:
                        import emitter as _emp8
                        from emitter import render as _rp8
                        items.append(('line', _rp8(
                            'move', {'nick': self.nick,
                                     'pieces': ['247'],
                                     'frm': _emp8.zone_code_for(_z8)
                                     if _z8 else 'Game.1887',
                                     'to': _emp8.zone_code_for(
                                         _szp8(_tgt8))},
                            local=False)))
                    except Exception as _pe8b:
                        self.alarms.append('pawn re-walk failed: %r'
                                           % (_pe8b,))
                    self._pawn_owed = (_tgt8, _tp8.time() + 45.0)
        owed = getattr(self, '_track_owed', None)
        if not owed:
            return items
        for pid in list(owed):
            tgt, deadline = owed[pid]
            zone = str(pr.get(str(pid), '') or '')
            m = _re.match(r'^Track (\d+)$', zone)
            n = int(m.group(1)) if m else None
            if n == int(tgt):
                owed.pop(pid, None)         # dump confirms - released
                continue
            if _time9.time() > deadline:
                # BL-50b: never re-walk while our OWN marker line is
                # still in flight (a re-walk from a lagging box
                # double-steps the marker)
                _rec50 = getattr(self, '_recent_emit_pids', {}) or {}
                if _time9.time() - _rec50.get(str(pid), 0) <= 15.0:
                    continue
                self.alarms.append(
                    'OWED TRACK MARKER STALE: pid %s should show '
                    'Track %s, client shows %r - re-walking it '
                    '(engine truth; never adopted away)'
                    % (pid, tgt, zone))
                frm = ('Game.%d' % (2014 + n) if n is not None
                       else 'Game.2014')
                items.append(('line', _crender(
                    'move', {'nick': self.nick, 'pieces': [str(pid)],
                             'frm': frm,
                             'to': 'Game.%d' % (2014 + int(tgt))},
                    local=False)))
                owed[pid] = (int(tgt), _time9.time() + 45.0)
        return items

    def _resync_dice_pids(self):
        from emitter import die_piece_for
        return {(side, index): die_piece_for(self.view, side.name, index)
                for side, dice in self.es.dice.items()
                for index in range(len(dice))}

    def _adoption_outcome(self, report, pid_to_cid=None):
        import ai
        accepted = any(str(line).startswith('board adopted:') for line in report)
        rows = client_board.diff_engine_vs_dump(
            self.es, self.view.piece_region,
            dynamic_types=self.view.dynamic_types,
            marker_states=self._client_marker_states(), ai_module=ai,
            initial_dump=getattr(self, '_initial_dump', None),
            granted_pids=self._granted_die_pids(self.view.piece_region),
            pid_to_cid=(pid_to_cid if pid_to_cid is not None else
                        {str(pid): str(cid) for cid, pid in self.view.card_pid_map.items()}),
            revoked_pids=getattr(self, '_revoked_die_pids', None))
        remaining = None if rows is None else [str(row) for row in rows
                    if not client_board.is_staging_row(row)]
        return {'accepted': accepted, 'complete': accepted and remaining == [],
                'remaining': remaining}

    def _adopt_board_now(self, why, expected_evidence=None):
        """Adopt the client's board into the engine (the resync adopter:
        units / leaders / companions / guide / tracks; census-gated,
        transactional).  Returns note items."""
        items = []
        if self.forced_ai_rows_pending():
            return [('note', 'Committed AI board corrections are retained; use RESYNC to complete them without changing the paid action.')]
        self._last_board_adoption = {'accepted': False, 'complete': False}
        # BL-117(iv): NO adoption path, quiet or resync, runs over an
        # open battle - combat surfaces are engine-owned mid-battle
        if isinstance(getattr(self.es, 'pending_combat', None), dict):
            items.append(('note', '%s: battle open - board adoption '
                                  'SKIPPED (combat surfaces are '
                                  'engine-owned until the battle '
                                  'closes)' % why))
            self.findings.append(('adopt_blocked_mid_battle',
                                  str(why)[:60]))
            return items
        # FLA-7 (par5 g4#2): an adoption while OPEN owed: demands
        # explain the rows would erase the engine's paid delta (the
        # Orcs Multiplying recruits were deleted this way).  Auto and
        # consented adoptions DEFER until the demands close or expire
        # at rollover; the MANUAL panel resync stays client-as-is
        # (BL-147) and the lenient path keeps its contract.
        if why not in ('manual', 'lenient'):
            _dem7 = [self._demand_text(e)
                     for e in (self.echo_ledger or ())]
            _sso7 = getattr(self, '_siege_seat_owed', None)
            if _sso7:
                _dem7.append('seat the garrison in the %s Stronghold '
                             'box' % _sso7)
            if getattr(self, 'holds', None):
                # FLD-6b (par5b_g2 ghost #2, T7 Bree engine=2
                # client=1 elites): a HELD window's lines stand
                # DISPUTED on the client - adopting them bakes the
                # disputed gesture into the engine, and the gate's
                # later client-side undo leaves the engine
                # double-applied
                _dem7.append('held gesture window (gate dispute '
                             'open)')
            if _dem7:
                items.append(('note', '%s: adoption DEFERRED - open '
                                      'owed demand(s) explain the '
                                      'board: %s'
                              % (why, '; '.join(_dem7[:4]))))
                self.findings.append(('adopt_deferred_owed_demands',
                                      '; '.join(_dem7[:2])[:120]))
                return items
        if not self.view.refresh_dump():
            items.append(('note', '%s: no stable client dump - nothing '
                                  'adopted' % why))
            return items
        if expected_evidence is not None:
            from .repair_evidence import same_dump
            if not same_dump(self, expected_evidence):
                items.append(('note', '%s: physical board changed after consent measurement; '
                                      'nothing adopted, review the current board' % why))
                self.findings.append(('adopt_measurement_changed', why))
                return items
        # BL-136(ii): NO reconcile path adopts away a PAID engine delta
        # younger than the dump it reads (a dump written BEFORE the
        # last engine commit cannot know that commit's board).
        # Deferred, not refused: the next quiet tick re-reads; after
        # 60s the client has provably not rendered it and the adoption
        # proceeds (loudly, via the eraser witness below).
        if why != 'lenient':
            import os as _os136
            import time as _t136b
            _lc136 = float(getattr(self, '_last_commit_ts', 0) or 0)
            _dp136 = getattr(self.view, 'dump_path', None)
            try:
                _dm136 = (_os136.path.getmtime(_dp136)
                          if _dp136 else None)
            except OSError:
                _dm136 = None
            if _dm136 is not None and _lc136 and _dm136 < _lc136 \
                    and (_t136b.time() - _lc136) < 60.0:
                items.append(('note', '%s: dump older than the last '
                                      'engine commit - adoption '
                                      'DEFERRED (a paid delta the '
                                      'client has not rendered yet '
                                      'is never adopted away)' % why))
                self.findings.append(('adopt_deferred_stale_dump',
                                      str(why)[:40]))
                return items
        try:
            from . import adopt as _adopt
            import ai as _ai9
            _adopt_call = _adopt.adopt_client_board
            if expected_evidence is not None:
                from functools import partial
                from .repair_evidence import adopt_measured
                _adopt_call = partial(adopt_measured, self, expected_evidence)
            _rep, _nb, _na = _adopt_call(
                self.es, self.view.piece_region,
                dynamic_types=self.view.dynamic_types,
                ai_module=_ai9,
                guide_hook=self.reconcile_guide_from_client,
                dump_path=getattr(self.view, 'dump_path', None),
                owed_tracks={p: v for p, (v, _dl) in
                             (getattr(self, '_track_owed', None)
                              or {}).items()},
                owed_pawn=(getattr(self, '_pawn_owed', None)
                           or (None,))[0],
                # BRCMP design 3/4: the lenient path and the human's
                # 'yes' are CONSENT forms (rules flags + tracks follow
                # the board); any other call adopts PIECES only
                consent=(why in ('lenient', 'consented')),
                dice_pids=(self._resync_dice_pids() if why in ('consented', 'manual', 'lenient') else None),
                dice_phase=self.phase,
                pending_fields=self._PENDING_FIELDS)
            self._last_board_adoption = self._adoption_outcome(_rep)
            _flap110 = getattr(self, '_adopt_flap110', None)
            if _flap110 is None:
                _flap110 = self._adopt_flap110 = {}
            for _line in _rep:
                items.append(('note', '%s: %s' % (why, _line)))
                _lt110 = str(_line)[:120]
                # BL-110(b): a RULES-STATE-changing reconcile (a siege
                # dropped/formed) is never quiet
                if 'siege dropped' in _lt110 \
                        or 'siege formed' in _lt110:
                    self.alarms.append('RECONCILE CHANGED RULES '
                                       'STATE: %s' % _lt110)
                if 'DROPPED after adoption' in _lt110:
                    self.findings.append(('adopt_pending_dropped',
                                          _lt110[:100]))
                # BL-110(a): the FLAP detector - the same PIECE row
                # adopted 3+ times is the escalation.  PIECE rows only
                # (BRCMP-2): the summary line 'board adopted: N -> N'
                # repeats whenever the rows are of a class no leg
                # touches and is never a flap
                if not _lt110.startswith(('units ', 'leaders ', 'nazgul ',
                                          'companion ', 'minion ')):
                    continue
                _flap110[_lt110] = _flap110.get(_lt110, 0) + 1
                if _flap110[_lt110] == 3:
                    _fmsg = ('FLAPPING DIVERGENCE: adopted 3x, keeps '
                             'coming back - %s' % _lt110)
                    self.alarms.append(_fmsg)
                    self._flap_alarms = list(
                        getattr(self, '_flap_alarms', None) or []) \
                        + [_fmsg]
            # the next quiet read verifies the adoption from a fresh
            # streak (a clean read retracts what the census raised)
            self._reconcile_due = True
            self._recon_prev_sig = None
            self._recon_sig_since = None
            self._recon_sig_reads = 0
            # BL-97: this adopter is MODE-INDEPENDENT (the reconcile
            # lever); its finding kind must never read as the lenient
            # path
            self.findings.append(('board_adopt',
                                  'divergences %d -> %d' % (_nb, _na)))
            # BL-136(ii): the ERASER WITNESS - an adoption that removed
            # engine pieces from a region the engine touched seconds
            # ago erased a paid delta the client never rendered; loud,
            # never silent
            self._erase_witness136(_rep)
            # BL-136(iv): the armies-key invariant on the ADOPT path
            # (the apply-path witness never sees adoption writes)
            try:
                from board import sanitize as _sn136c
                _seen136c = {}
                for _k136 in list(self.es.armies):
                    _s136 = _sn136c(str(_k136))
                    if _s136 in _seen136c and _seen136c[_s136] != _k136:
                        self.alarms.append(
                            'ARMIES-KEY COLLISION after adoption: %r '
                            'vs %r sanitize equal - mixed key '
                            'conventions split a garrison' %
                            (_seen136c[_s136], _k136))
                        self.findings.append(
                            ('armies_key_collision_adopt',
                             '%r vs %r' % (_seen136c[_s136], _k136)))
                    _seen136c[_s136] = _k136
            except Exception:
                pass
        except Exception as _e9:
            items.append(('note', '%s: adopt FAILED (engine left '
                                  'untouched): %r' % (why, _e9)))
            self.alarms.append('BOARD ADOPT FAILED: %r' % (_e9,))
        return items

    def _erase_witness136(self, report, now=None):
        """BL-136(ii): alarm when an adoption report shows engine
        pieces REMOVED from a region the engine itself changed within
        the last 90s - the client never rendered that delta and the
        reconcile just deleted it.  Pure function of the report + the
        recent-delta ledger (pinned directly)."""
        import re as _re136
        import time as _t136c
        now = _t136c.time() if now is None else now
        recent = getattr(self, '_recent_delta136', None) or {}
        for line in report or ():
            m = _re136.match(r'^(units|nazgul|leaders?) (.+?)'
                             r'(?: (\S+/\S+))?: (\d+) -> (\d+)$',
                             str(line).strip())
            if not m:
                continue
            reg = m.group(2).strip().lower()
            a, b = int(m.group(4)), int(m.group(5))
            if b >= a:
                continue
            born = recent.get(reg)
            if born is None or (now - born) > 90.0:
                continue
            msg = ('RECONCILE ERASED A PAID ENGINE DELTA at %s: "%s" '
                   '%.0fs after the engine commit that made it - the '
                   'client never rendered it; the engine piece is '
                   'GONE, RESYNC/undo owed'
                   % (reg, str(line).strip()[:80], now - born))
            if msg not in self.alarms:
                self.alarms.append(msg)
            self.findings.append(('reconcile_erased_paid_delta',
                                  str(line).strip()[:80]))

    def _lenient_accept(self, w, why):
        """Reconcile a completed human action silently from its native board.

        Wait for complete export evidence before recording costs or handing
        play to the AI. Unresolved Hunt/combat choices retain their handlers.
        """
        why = str(why or 'gesture')[:120]
        # BL-97 mode-gate: under ENFORCEMENT no path may reach the
        # lenient accept - a caller that does is a routing bug and
        # must be loudly visible
        if getattr(self, 'enforce_human', False):
            self.alarms.append('LENIENT PATH REACHED UNDER '
                               'ENFORCEMENT (%s) - routing bug' % why)
        if 'expressed nothing' in why and w is not None \
                and getattr(w, 'die_face', None):
            # BL-12/BL-70(4): an EMPTY window is NEVER a die spend -
            # undo the stage instead of consuming the die for nothing
            from .undo_model import presses_to_revert
            from emitter import render as _r12
            items12 = [('note', 'empty window: die stage undone, '
                                'nothing spent (legality OFF)')]
            n12 = presses_to_revert(self.feed.lines,
                                    self._window_span(w), self.nick)
            if n12:
                items12.extend(('line', _r12(
                    'undo', {'nick': self.nick}, local=False))
                    for _ in range(n12))
            self.findings.append(('lenient_empty_window_unstaged',
                                  why[:80]))
            return items12
        from .lenient_board import accept
        return accept(self, w, why)

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

    def _divergence_open(self):
        """FLA-9/10 (par5 g5#9 + g3#6b): a CONFIRMED board divergence
        stands - the last quiescent census carried rows, the human
        declined an adoption over standing rows, or a census repair
        is open.  While it stands the engine's picture of the client
        is KNOWN wrong: the gate must not force-undo on its
        authority, and the human's die-less drags are repair
        traffic."""
        if getattr(self, '_census_declined_sig', None) \
                or getattr(self, '_census_confirmed_sig', None):
            return True
        r = self._repair
        if r is not None and r.get('kind') == 'census':
            return True
        lqc = getattr(self, '_last_quiescent_census', None) or {}
        return bool(lqc.get('board'))

    def _refuse(self, w, res, explain=None):
        if not self.enforce_human:
            return self._lenient_accept(
                w, getattr(res, 'note', None) or 'refused gesture')
        if self._divergence_open():
            # FLA-9 (par5 g5#9): the gate judged this gesture against
            # a state a CONFIRMED divergence says is wrong (the
            # impossible 'Attack Lorien from Lorien' closest-legal) -
            # never force-undo on that authority.  Hold + witness;
            # retried when the divergence clears (adopt / resync /
            # repair drags).
            self.findings.append(('gesture_hold_divergence_open',
                                  str(getattr(res, 'note', ''))[:80]))
            _msg9 = ('GESTURE HELD: a confirmed board divergence is '
                     'open - not judging or undoing your move against '
                     'a board I know is wrong; say adopt or resync')
            if _msg9 not in self.alarms:
                self.alarms.append(_msg9)
            self.holds.append((w, res))
            self._tag_hold(w, 'divergence')
            return [('chat', 'refusal',
                     chat_mod.divergence_gate_hold())]
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
            # BL-100(ii): the reveal swaps un-strand on EVERY failure
            # path (the pending-buffer orphan already did this)
            self._unreveal(w)
            self.holds.append((w, res))
            # R5 / D-03: the restore watch is ARMED on this path too.
            # DOCTRINE = HOLD-AND-WITNESS: the client-native undo
            # cannot reach the window (lines were pushed after it -
            # client fact 2), so the board stays unrestored; the watch
            # VERIFIES against the dump and fires the sticky RESTORE
            # WATCH alarm unless someone (human undo / manual edit +
            # resync) restores it.  No reversal is forged.
            expect, notat, _unver = expectations_from_window(
                [r.get('norm', '') for r in w.records],
                pids=self._record_pids(w.records))
            if expect or notat:
                self.restore_watch.arm(expect, notat)
            return out
        for _ in range(n):
            out.append(('line', render('undo', {'nick': self.nick},
                                       local=False)))
        from .human_draw_identity import cancel_records
        cancel_records(self, w.records)
        # BL-100(ii): the refused window's reveal swaps un-strand
        # (see _unreveal - live human slots END with the real cid)
        self._unreveal(w)
        # BL-94(b): PHYSICAL DRAWS ARE FACTS, NOT GESTURE LEGS.  The
        # undo stack is global, so reverting the window pops any
        # engine-demanded draw fed AFTER its terminator - but that
        # draw's identity is already in the engine hand (the grant was
        # consumed).  Re-issue each popped draw directive right after
        # the presses: the undo restored the deck order, so the client
        # re-draws the SAME physical card and both hands re-agree (the
        # unconsumed-rng quarantine owns the re-fed noise - census
        # correctness outranks it).
        import re as _re94
        _w094 = self._window_span(w)[0]
        _owned132 = dict(getattr(w, '_fed_draws132', None) or {})
        from collections import Counter as _DrawCounter
        _window_draws = _DrawCounter(str(record.get('raw', ''))
            for record in w.records if record.get('op') == 'draw')
        for _ln94 in (self.feed.lines[_w094:] if _w094 >= 0 else ()):
            _m94 = _re94.search(
                r'draws (Free|Shadow) (Character|Strategy) Card',
                str(_ln94))
            if _m94:
                if _window_draws.get(str(_ln94), 0):
                    _window_draws[str(_ln94)] -= 1
                    self.findings.append(('refusal_draw_not_reissued',
                                          'uncommitted window: %s' % str(_ln94)[:50]))
                    continue
                _tag94 = 'fp' if _m94.group(1) == 'Free' else 'sa'
                _dk94 = ('char' if _m94.group(2) == 'Character'
                         else 'strat')
                # BL-132 L3: a draw fed against THIS window's own
                # starving demand died with the rollback (the client
                # undo already restored the deck) - re-issue only the
                # genuinely-popped EXTERNAL grant (BL-94(b)'s case:
                # identity kept in the engine hand across the refusal)
                if _owned132.get((_tag94, _dk94), 0) > 0:
                    _owned132[(_tag94, _dk94)] -= 1
                    self.findings.append(
                        ('refusal_draw_not_reissued',
                         'window-owned: %s' % str(_ln94)[:50]))
                    continue
                if _tag94 == ('fp' if self.human_side.name == 'FREE_PEOPLES' else 'sa'):
                    out.append(('chat', 'prompt', chat_mod.draw_ask(
                        ('free' if _tag94 == 'fp' else 'shadow') + ' ' +
                        ('character' if _dk94 == 'char' else 'strategy'))))
                else:
                    out.append(('call', '!CALL draw_%s_%s'
                                % (_tag94, _dk94)))
                # arm the one-shot echo token: the re-rendered draw
                # line is THIS draw, never an un-owed human press
                _ri94 = getattr(self, '_reissued_draws94', None) or {}
                _ri94[(_tag94, _dk94)] = _ri94.get((_tag94, _dk94),
                                                   0) + 1
                self._reissued_draws94 = _ri94
                if _tag94 == ('fp' if self.human_side.name == 'FREE_PEOPLES' else 'sa'):
                    self._manual_reissued_context94 = (
                        self.es.turn, self.n_commits_total,
                        len(self.es.hands[self.human_side].hand))
                import time as _t107b
                self._reissued_ts94 = _t107b.time()   # BL-107 TTL
                self.findings.append(('refusal_draw_reissued',
                                      str(_ln94)[:60]))
        # ARM the restore watch on the refused window's own rendered
        # lines; tick() verifies against the client dump and the
        # sticky alarm fires if the restore never lands
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
        """Alarm once when neither explicit identity channel completes the
        parked adoption. Empty metadata alone is not an error when the exact
        native command has already supplied its matching die/face pairs."""
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
    def _withdraw_retained_window(self, w):
        """Withdraw one native tail line without leaving a retryable close.

        Reuse the segmenter's ordinary prefix grammar so both orders of a
        Fellowship gesture recover their correct remaining die obligation.
        Randomness already harvested from this window stays banked; only its
        now-invalid request is retired.
        """
        if self.seg.window is not None:
            self.alarms.append('UNDO of retained window while a newer die '
                               'window is open - holding')
            return []
        replay = Segmenter(nick=self.seg.nick)
        for record in w.records[:-1]:
            replay.push(record, allow_fellowship_prefix=True)
        self._unreveal(w)
        for record in w.records:
            norm = record.get('norm', '')
            if norm in self.uncommitted:
                self.uncommitted.remove(norm)
        self.holds = [h for h in self.holds if h[0] is not w]
        if self.deferred is not None and self.deferred[0] is w:
            self.deferred = None
        call = getattr(w, '_starve_call', None)
        if self.rand is not None and call is not None \
                and self.rand.pending() is call:
            self.rand._pending = None
            self.rand._pending_got = 0
        w.records.pop()
        for field in ('_starve_call', '_starve_note', '_hold_cond',
                      '_proven_starve_call', '_proven_starve_signature',
                      '_proven_starve_state', '_relentless_starve_call'):
            setattr(w, field, None)
        self.seg._last_move = None
        if replay.window is not None:
            for field in ('die_side', 'die_face', 'awaiting_fellowship_die',
                          'fellowship_shortcut'):
                setattr(w, field, getattr(replay.window, field))
            w.closed = None
            self.seg.window = w
        elif not w.records:
            w.closed = 'undone'
            return self._route_event('discarded', w)
        else:
            # A die-less refused prefix has no segmenter owner. Its remaining
            # lines are still uncommitted and can be unwound normally.
            w.closed = 'undone'
            self.uncommitted.extend(r.get('norm', '') for r in w.records)
        return []

    def _restore_card_event_window(self, saved, norm):
        window = self._copy_card_window(saved)
        if window is None:
            return
        if window.records and window.records[-1].get('norm') == norm:
            window.records.pop()
        window.closed = None
        self.seg.window = window
        self.seg._last_move = None
        self._staged_event_window = window

    def _card_completion_undo(self, norm):
        """Reopen a reversible physical choice without retracting its paid card."""
        def has_outcome():
            feed = self.rand._feed
            return any(feed[k] for k in ('d6', 'tiles', 'companion', 'card_pick')) or any(feed['cards'].values())

        def clear_close(buf):
            buf._staged_card_closed = False
            buf._staged_card_close_record = None
            buf._staged_card_request_nonce = None
            for field in ('_proven_starve_call', '_proven_starve_signature',
                          '_proven_starve_state', '_relentless_starve_call'):
                setattr(buf, field, None)
            buf._casualty_closed = False
            buf._elite_completion_prompted = False
            self._last_pending_match = None
            self._buf_stall = None

        def held():
            message = 'CARD CHOICE UNDO HELD: a roll or emitted consequence has already resolved; its outcome is retained'
            if message not in self.alarms:
                self.alarms.append(message)
            return [('note', message)]

        buf = self.pending_buf
        unmatched_cleanup = getattr(buf, '_unmatched_card_cleanup', None)
        if (unmatched_cleanup is not None and unmatched_cleanup.records
                and unmatched_cleanup.records[-1].get('norm') == norm):
            # This cleanup closed an invalid, still-uncommitted effect.
            # Undo restores only that physical boundary, preserving the
            # paid card and any randomness already supplied for its effect.
            self._restore_card_event_window(unmatched_cleanup, norm)
            buf._unmatched_card_cleanup = None
            self._last_pending_match = None
            self.undo_epoch += 1
            return []
        close = getattr(buf, '_staged_card_close_record', None)
        closed_tail = (buf is not None and getattr(buf, '_staged_card_closed', False)
                       and buf.records and buf.records[-1].get('norm') == norm)
        if (close and close.get('norm') == norm) or closed_tail:
            # Closing may have requested randomness without committing yet.
            # Cancel only an unanswered request; never discard a supplied roll.
            if has_outcome():
                return held()
            nonce = getattr(buf, '_staged_card_request_nonce', None)
            self._restore_card_event_window(getattr(buf, '_staged_card_event_window', None), norm)
            if closed_tail:
                buf.records.pop()
            clear_close(buf)
            self.rand.abort_step('native Undo of an unrolled card completion')
            self.undo_epoch += 1
            return [('cancel_card_request', nonce)] if nonce is not None else []
        index = len(self.commits) - 1
        while index >= 0 and not self.commits[index][1] and str(self.commits[index][2]).startswith('ai:'):
            index -= 1
        if index < 0 or len(self.commits[index]) < 6:
            return None
        entry = self.commits[index]
        completion = entry[5]
        if not isinstance(completion, dict) or not completion.get('card_completion'):
            return None
        cleanup = completion['cleanup']
        if cleanup and cleanup[-1].get('norm') == norm:
            if index != len(self.commits) - 1 or has_outcome():
                return held()
            cleanup.pop()
            self._restore_card_event_window(completion['event_window'], norm)
            completion['event_window'] = self._copy_card_window(self.seg.window)
            self.undo_epoch += 1
            return []
        if norm in entry[1] and (not completion['reversible'] or index != len(self.commits) - 1):
            return held()
        if not entry[1] or entry[1][-1] != norm:
            return None
        if has_outcome():
            return held()
        self.commits.pop()
        self.es = entry[0]
        for hand in self.hand_models.values():
            hand.es = self.es
        self.hands.es = self.es
        self._restore_turn_state(entry[3])
        buf = self._copy_card_window(completion['buffer'])
        if buf.records and buf.records[-1].get('norm') == norm:
            buf.records.pop()
        clear_close(buf)
        self.pending_buf = buf
        self._restore_card_event_window(completion['event_window'], norm)
        self.undo_epoch += 1
        self.findings.append(('card_completion_undone', norm[:100]))
        return []

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
            # BL-26: a redo restoring an AI line the human just
            # undid puts the board BACK in sync - consume it quietly
            _ri = str(rec.get('inner', '')).strip()
            if _ri and _ri == getattr(self, '_ai_undone', None):
                self._ai_undone = None
                self.findings.append(('undo_of_ai_line_redone',
                                      _ri[:80]))
                return [('note', 'redo restored my line - board back '
                                 'in sync')]
            # unmodelled: a sticky alarm (once), never a halt that
            # would drop the rest of the poll batch
            msg = 'REDO seen - unmodelled, holding; RESYNC if the ' \
                  'board looks wrong'
            if msg not in self.alarms:
                self.alarms.append(msg)
            return [('note', 'client REDO is not modelled - RESYNC')]
        inner = str(rec.get('inner', '')).strip()
        kind, norm = live_parser.normalize(inner)
        victor_close = getattr(self.pending_buf, '_victor_close_record', None)
        if victor_close and victor_close.get('norm') == norm:
            self.pending_buf._victor_close_record = None
            self._last_pending_match = None
            self.undo_epoch += 1
            return []
        completion_undo = self._card_completion_undo(norm)
        if completion_undo is not None:
            return completion_undo
        window = self.seg.window
        if (window is not None and not getattr(self, 'spectator', False)
                and window is not getattr(self, '_staged_house_window', None)
                and window is not getattr(self, '_staged_event_window', None)):
            # Native face changes push no Undo snapshot. Before this action
            # commits, undoing its latest ring payment also restores any
            # following die-face changes. Retire that uncommitted suffix;
            # the older staged die and the applied-action ledger stay intact.
            records = window.records
            index = len(records) - 1
            while index >= 0 and records[index].get('op') == 'die_change':
                index -= 1
            if 0 <= index < len(records) - 1:
                payment = records[index]
                ring_payment = (payment.get('op') == 'move' and
                    gesture_match.is_ring_track_move(payment.get('obj'),
                        payment.get('from'), payment.get('to'))) or (
                    payment.get('op') == 'kill' and
                    len(payment.get('pieces') or ()) == 1 and
                    str(payment['pieces'][0]).lower() in ('vilya', 'nenya', 'narya'))
                if ring_payment and payment.get('norm') == norm:
                    del records[index:]
                    self.seg._last_move = None
                    self.findings.append(('ring_prefix_undone', norm))
                    return []
        ring_records = getattr(self, '_ring_gesture_records', None)
        if ring_records and any(r.get('norm') == norm and r.get('op') != 'die_change' for r in ring_records):
            index = next(i for i, r in enumerate(ring_records) if r.get('norm') == norm)
            del ring_records[index:]
            if not ring_records:
                self._ring_gesture_records = None
            collateral = getattr(self, '_ring_refusal_collateral', None)
            if collateral in self.uncommitted:
                self.uncommitted.remove(collateral)
            self._ring_refusal_collateral = None
            return []
        # BL-26: the client's undo stack is GLOBAL - a human undo can
        # take back the AI'S OWN line while the engine keeps it; the
        # engine holds its truth, says so loudly, and asks for the
        # redo
        if live_parser.parse(inner).get('player') == self.nick:
            # BRLANE-7 / BL-26: the AI's own line is RE-ASSERTED (its raw
            # emitted line re-injected, echo expected again), never
            # adopted - the reconcile is NOT armed here (that arm was
            # the adoption path that erased the AI's move)
            self.findings.append(('undo_of_ai_line', inner[:80]))
            _ent7 = self._reassert_undone_ai_line(inner)
            if _ent7 is not None:
                _ent7['reasserted'] = True
                self.findings.append(('undo_of_ai_line_reasserted',
                                      inner[:80]))
                self.expect_echo([_ent7['raw']])
                return [('chat', 'prompt',
                         chat_mod.undo_ai_line_reasserted()),
                        ('line', _ent7['raw'])]
            self._ai_undone = inner
            _msg7 = ('UNDO OF AI LINE NOT RE-ASSERTABLE: %r - say '
                     'resync (the engine keeps its move)'
                     % inner[:80])
            if _msg7 not in self.alarms:
                self.alarms.append(_msg7)
            self.findings.append(('undo_of_ai_line_unreassertable',
                                  inner[:80]))
            return [('chat', 'prompt', chat_mod.undo_took_ai_line())]
        if self.deferred_tail and self.deferred_tail[-1].get('norm') == norm:
            self.deferred_tail.pop()
            return []
        if self.deferred is not None:
            w, _side = self.deferred
            if w.records and str(w.records[-1].get('norm', '')) == norm:
                if self.deferred_tail:
                    self.alarms.append('UNDO of deferred window before its '
                                       'newer queued gestures - holding')
                    return []
                return self._withdraw_retained_window(w)
        for w, _result in reversed(self.holds):
            if w.records and str(w.records[-1].get('norm', '')) == norm:
                return self._withdraw_retained_window(w)
        urec = live_parser.parse(inner)
        if urec.get('op') == 'move' and gesture_match.zone_key(
                urec.get('from'))[0] == 'hand_chosen':
            items = self._on_reveal_undone(urec)
            if items is not None:
                return items
        buf = self.pending_buf
        if (buf is not None and buf.records
                and str(buf.records[-1].get('norm', '')) == norm):
            # This native line is still uncommitted. Retract its evidence,
            # not the earlier chat decision or the AI's battle advance.
            from .casualty_gestures import live_context
            context = live_context(self.es)
            if (getattr(buf, '_hold_ambiguous_elite_prefix', False)
                    and context is not None and context['side'] is self.human_side
                    and self.rand is not None):
                # A probe request is hidden while this payment is unproved.
                # Withdrawing the last row must not expose it via the generic
                # RNG fallback. Keep supplied outcomes and the replay journal;
                # only cancel the request whose physical evidence was undone.
                self.rand._pending = None
                self.rand._pending_got = 0
            self._unreveal(buf)
            buf.records.pop()
            for field in ('_proven_starve_call', '_proven_starve_signature',
                          '_proven_starve_state', '_relentless_starve_call'):
                setattr(buf, field, None)
            buf._casualty_closed = False
            buf._elite_completion_prompted = False
            if not buf.records:
                self.pending_buf = None
            self._guards_recruit_refusal = None
            self._partial_corruption_voiced = None
            self._pending_amb_voiced = None
            self._last_pending_match = None
            self._buf_stall = None
            return []
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
                if label == 'eye':
                    self.findings.append(('eye_undone', norm[:80]))
                # BRLANE-2: the pop may have cleared what a hold waited on
                self._out_queue.extend(self._retry_holds('undo'))
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
            # chat kind is NOT in the sided vocabulary (ChatPolicyError
            # at the exit)
            return [('chat', 'prompt', chat_mod.undo_rebased())]
        return [('note', 'undo re-base incomplete - RESYNC')]

    # ---- AI turn --------------------------------------------------------
    def _ai_turn_start_draws(self):
        self.phase = 'declare'
        self._turn_start_draws_open = True
        import game_state as gs
        # K2: the Fellowship phase opens for WHICHEVER side is FP
        self._ensure_declare_armed()
        if self.ai_side is gs.Side.FREE_PEOPLES:
            # R2: the AI holds the Fellowship phase - its own
            # declaration decision (learned) follows its draws, then
            # the Mordor-entry check, then ITS 'fp ok' (live_server
            # queues the second step behind the first)
            return [('ai_draws', None), ('ai_declare', None)]
        return [('ai_draws', None)]     # live_server runs the AI draws

    def _ai_turn(self):
        """When the engine says the pending decision (or the active
        turn) is the AI's, decide and emit.  Emission module wires in
        live_server; here we only surface the fact."""
        if self.forced_ai_rows_pending():
            return []
        if self.phase == 'guide' or (self.phase == 'fellowship_tail'
                and self.ai_side.name == 'FREE_PEOPLES'):
            return []  # ai_declare owns AI guide choices; human gestures own theirs.
        if self._repair is not None \
                and self._repair.get('phase') in ('kill', 'resync'):
            return []                # never act mid self-repair
        if (getattr(self, '_turn_start_draws_open', False)
                and self.phase in ('draws', 'declare')):
            return [('ai_draws', None)]
        dp = self.ai_action_dp()
        if dp is not None:
            if dp['kind'] == 'eye':
                # Allocation completion also advances the bridge to rolls.
                # The generic action step does not own that continuation.
                return [('ai_eyes', None)] if self.phase == 'eyes' else []
            if dp['kind'] == 'declare':
                # the AI-FP's declaration is the 'ai_declare' step's
                # (queued by _ai_turn_start_draws), never a generic
                # decide item
                return []
            return [('ai_decide', dp)]
        return []

    def ai_action_dp(self, resuming_owed=()):
        """The AI's due decision, or None: an open engine pending owned
        by the AI, OR — PHASE3-B gap closed 2026-08-20 — the plain
        action-turn ALTERNATION after a committed action.  Whose-turn
        comes from the ONE engine answerer ai._next_acting_side
        (pendings, multi-step continuation, the battle-is-one-
        attacker-action rule, dice exhaustion -> None); the driver
        contributes only WHO JUST ACTED (its own commit bookkeeping),
        never a turn rule."""
        if self.forced_ai_rows_pending() or getattr(self, '_lenient_board_wait', None) is not None:
            return None
        import ai
        import game_state as gs
        if getattr(self, '_ring_gesture_records', None):
            return None
        if self.ai_side is gs.Side.FREE_PEOPLES:
            # K2: the AI-FP's declare menu stays armed through its own
            # turn-start draws/discard until its declare step closes
            self._ensure_declare_armed()
        # BL-21(2): gate on ANY open pending, not only the
        # AI-perspective view - a held FP casualty choice is invisible
        # under describe_pending(es, ai_side) and the alternation path
        # would let the AI play PAST it
        dp = ai.describe_pending(self.es)
        if dp is None:
            dp = ai.describe_pending(self.es, self.ai_side)
        if dp is not None:
            if dp['side'] is not self.ai_side:
                return None
            if dp['kind'] == 'action' and self.action_completion_pending(resuming_owed):
                return None
            return self._project_ai_dp(dp)
        if self.phase != 'actions' or self.last_actor is None:
            return None
        if self.action_completion_pending(resuming_owed):
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
        return self._project_ai_dp({'kind': 'action', 'side': self.ai_side,
                                   'actions': menu})

    def _project_ai_dp(self, dp):
        """AI conventions affect only the AI's published decision menu."""
        from ai_policies import project_ai_actions
        projected = project_ai_actions(self.es, self.ai_side, dp['actions'])
        if not projected.actions and dp['actions']:
            raise DriverHalt('AI policy removed every legal action')
        return dict(dp, actions=list(projected.actions),
                    ai_policy_forced_reason=projected.forced_reason,
                    ai_policy_reasons=projected.reasons)

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
    _TURN_STATE_FIELDS = ('turn_draws', '_turn_start_draws_open', 'phase', 'last_actor',
                          'owed_die_kills', 'owed_card_kills',
                          'human_cc_committed', '_human_combat_card_kills', '_ring_gesture_records',
                          '_cc_battle_serial', '_cc_reveal_owed',
                          '_cc_reveal_origins', '_cc_choose_preamble', '_siege_press_intent',
                          '_siege_seat_owed', '_siege_seat_receipts', '_siege_seat_redo',
                          '_guide_seat_receipts', '_guide_seat_redo', '_guide_seat_commit_epoch',
                          '_uncommitted_card_selection', '_post_commit_tail',
                          '_human_casualty_prefix', '_ai_casualty_prefix',
                          '_native_reveal_reset', '_allocation_roll_wait',
                          '_guide_completion_requested')

    def _turn_state(self):
        out = {}
        from .temporary_combat_figures import snapshot
        out['_view_temporary_combat_figures']=snapshot(self.view)
        out['_view_shadow_origins'] = dict(
            getattr(self.view, 'shadow_slot_origins', {}) or {})
        out['_view_shadow_departures'] = dict(
            getattr(self.view, 'shadow_slot_departures', {}) or {})
        out['_view_shadow_arrivals'] = dict(
            getattr(self.view, 'shadow_slot_arrivals', {}) or {})
        out['_view_shadow_strongholds'] = dict(
            getattr(self.view, 'shadow_stronghold_regions', {}) or {})
        for f in self._TURN_STATE_FIELDS:
            v = getattr(self, f, None)
            if isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, list):
                v = list(v)
            out[f] = v
        return out

    def _restore_turn_state(self, ts):
        from .temporary_combat_figures import restore
        restore(self.view,(ts or {}).get('_view_temporary_combat_figures'))
        self.view.shadow_slot_origins = dict(
            (ts or {}).get('_view_shadow_origins') or {})
        self.view._shadow_slot_restore = True
        self.view.shadow_slot_departures = dict(
            (ts or {}).get('_view_shadow_departures') or {})
        self.view.shadow_slot_arrivals = dict(
            (ts or {}).get('_view_shadow_arrivals') or {})
        self.view.shadow_stronghold_regions = dict(
            (ts or {}).get('_view_shadow_strongholds') or {})
        for f, v in (ts or {}).items():
            if f in ('_view_shadow_strongholds', '_view_shadow_arrivals', '_view_shadow_departures', '_view_shadow_origins','_view_temporary_combat_figures'):
                continue
            if isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, list):
                v = list(v)
            setattr(self, f, v)

    def _snapshot(self, line_norms, label, post=None, action=None):
        """Ledger entry: (engine PRE-state, client line norms, label,
        driver turn state, the committed engine action or None), with an
        optional sixth entry for a reversible card-completion window. The
        callers snapshot BEFORE mutating either the engine or the turn
        state.  `action` lets a lineless AI commit be RE-BASED when the
        human retracts the window beneath it (_on_undo)."""
        import ai
        import time as _t136s
        self.commits.append((ai.clone_state(self.es),
                             [n for n in line_norms if n], label,
                             self._turn_state(), action))
        self.n_commits_total += 1       # monotonic commit epoch
        self._guide_seat_commit_epoch = (getattr(self, '_guide_seat_commit_epoch', 0) or 0) + 1
        self._last_commit_ts = _t136s.time()    # BL-136(ii) freshness
        if len(self.commits) > 100:
            self.commits.pop(0)
        # A-09 / R1: every commit checks for a MID-ACTION ring victory
        # on the POST state the caller passes (every commit site
        # snapshots BEFORE `es = post`, so self.es here is still PRE);
        # military waits for the turn-end check in _on_recover (RULING
        # 1.212).  post=None = callers whose es already IS the current
        # state (human draw, rollover).
        self.check_victory(end_of_turn=False, state=post)

    def _queue_public_roll_witnesses(self, raw_lines):
        """Read new native action-roll commands, never infer a die order.

        AiFeed can provide an empty sidecar on a real native menu action.
        The raw command still explicitly pairs each die PID with its face.
        Records are consumed in command order against the rendered roll.
        """
        from log_replay import FP_DICE_VALUES, SHADOW_DICE_VALUES
        import emitter
        lines = list(raw_lines)
        seen = getattr(self, '_public_roll_raw_seen', 0)
        if seen and (len(lines) < seen or lines[seen-1] != getattr(self, '_public_roll_raw_tail', None)):
            raise DriverHalt('Native action-roll witness stream changed; identity history cannot be reused')
        queue = self.__dict__.setdefault('_public_roll_witnesses', [])
        new_witnesses = []
        for raw in lines[seen:]:
            match = RAW_ACTION_ROLL_RE.search(raw)
            if match is None:
                continue
            pids = match.group(1).split()
            tokens = raw.split('<~Controls.346~>', 1)[1].split()
            values = tokens[:len(pids)]
            side = ('FREE_PEOPLES' if set(pids) <= set(emitter._FP_DIE_IDS)
                    else 'SHADOW' if set(pids) <= set(emitter._SH_DIE_IDS) else None)
            if (not pids or len(set(pids)) != len(pids) or side is None
                    or len(values) != len(pids)
                    or any(not value.isdigit() or len(value)>2 for value in values)):
                raise DriverHalt('Malformed native action-roll identity/face witness')
            table = FP_DICE_VALUES if side == 'FREE_PEOPLES' else SHADOW_DICE_VALUES
            try:
                faces = [gesture_match.FACE_LETTER[table[int(value)].name] for value in values]
            except (KeyError, ValueError):
                raise DriverHalt('Native action-roll witness contains an invalid die face')
            new_witnesses.append({'side':side, 'pids':pids, 'faces':faces})
        queue.extend(new_witnesses)
        self._public_roll_raw_seen = len(lines)
        self._public_roll_raw_tail = lines[-1] if lines else None
        # The rendered feed can win the file-write race. In that case the
        # adopted faces/indices are already parked; pair them with the SAME
        # explicit command rather than waiting for another rendered roll.
        while queue and queue[0]['side'] in self._last_adopt_idxs:
            witness = queue[0]
            import game_state as gs
            owner = gs.Side[witness['side']]
            idxs, count = self._last_adopt_idxs[witness['side']]
            actual = [gesture_match.FACE_LETTER[self.es.dice[owner][i].current.name]
                      for i in idxs]
            if actual != witness['faces'] or count != len(witness['pids']):
                raise DriverHalt('Late native action-roll identities conflict with adopted faces')
            queue.pop(0)
            self._pending_roll_pids[witness['side']] = witness['pids']
            self._try_bind_roll(witness['side'])
            self.findings.append(('roll_identity_from_late_native_command',witness['side']))

    def _take_public_roll_witness(self, faces, sidecar_pids):
        queue = getattr(self, '_public_roll_witnesses', None)
        if not queue:
            return None
        witness = queue[0]
        if list(faces) != witness['faces']:
            raise DriverHalt('Rendered action-roll faces conflict with their native identity command')
        if sidecar_pids is not None and list(sidecar_pids) != witness['pids']:
            raise DriverHalt('Action-roll sidecar identities conflict with the native command')
        queue.pop(0)
        return witness

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

        Why the button could not fix it AT THE TIME (pre-2026-08-21):
        resync then adopted only into the VIEW.  Since the adopt.py
        ruling ("the resync is supposed to take the CLIENT state and
        give it to the engine"), resync_recover adopts the engine too
        and routes the guide through THIS function as
        adopt_client_board's guide_hook.

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

    def resync_recover(self, drain_feed=False, politics_from_engine=False):
        """Retry retained combat, or adopt a stable board outside combat.

        Combat recovery preserves pending decisions, rolls and physical work.
        Outside combat, recovery includes registered dice zones, rings, tracks, pieces,
        politics and available card/deck identities. Unknown human cards remain
        placeholders. The adopter rolls back if a leg fails or worsens the census.
        drain_feed skips the unmatched backlog after manual board edits.
        """
        if self.combat_recovery_open():
            items, report = self.reset_and_resync(hand_turn=False)
            return items + [('note', 'combat recovery: %s' % line) for line in report]
        import game_logic as gl
        import time as _tr
        items = [('note', 'resync-recovery: adopting the client '
                          'board into the view and (census-gated) '
                          'the engine; hidden info stays engine-side')]
        self._last_resync_adoption = {'accepted': False, 'complete': False}
        if getattr(self, '_turn_start_draws_open', False):
            # The AI's two compulsory draws commit together. A board snapshot
            # may include only the first native result, or neither result
            # while its issued request is still in flight. Adopting that
            # snapshot would abandon the draw journal and lose its remaining
            # draw. Let the normal scheduler finish with the same requests.
            items.append(('note', 'resync incomplete: waiting for the current '
                          'turn-start card draws to finish; issued requests and '
                          'observed cards retained'))
            return items
        from .repair_evidence import dump_signature, same_dump, adopt_measured
        from .recovery_epoch import prepare as prepare_epoch, commit as commit_epoch
        from .recovery_board import validate as validate_recovery_board
        evidence = None
        boundary = None
        if self.view.dump_path:
            try:
                evidence = dump_signature(self)
                boundary = prepare_epoch(self, evidence)
            except (OSError, ValueError) as error:
                items.append(('note', 'resync incomplete: native board unavailable: %s' % error))
                return items
        elif getattr(self, 'live', True):
            items.append(('note', 'resync incomplete: no current native board export'))
            return items
        if self.view.refresh_dump():
            self.findings.append(('resync_adopt',
                                  '%d piece zones adopted'
                                  % len(self.view.piece_region)))
        else:
            items.append(('note', 'resync-recovery: no stable dump - '
                                  'nothing adopted'))
            return items
        if evidence is not None:
            try:
                if not same_dump(self, evidence):
                    raise ValueError('native board or feed changed while reading recovery evidence')
                validate_recovery_board(self, self.view.piece_region)
            except ValueError as error:
                items.append(('note', 'resync incomplete: %s; engine retained' % error))
                return items
        if self.human_action_completion_pending():
            from .action_die_zones import temporary
            unpaid = [str(pid) for pid, (side, index) in self.view.die_id_map.items()
                      if side is self.human_side
                      and 0 <= index < len(self.es.dice.get(side, ()))
                      and not self.es.dice[side][index].used
                      and temporary(self.view.piece_region.get(str(pid), ''))]
            if unpaid:
                # A witnessed selection is still in progress. Current Action
                # alone cannot turn an unused die into payment; only the
                # committed rules action establishes that. Returned dice and
                # already paid dice still use ordinary measured recovery.
                items.append(('note', 'resync incomplete: unfinished staged action '
                              'retained (unused dice %s); finish the action, return '
                              'the dice, or use Rewind' % ' '.join(sorted(unpaid))))
                return items
        def unfinished():
            choices = [name for name in self._PENDING_FIELDS
                       if getattr(self.es, name, None)]
            import ai
            pending = ai.describe_pending(self.es)
            if (self.phase == 'declare' and pending is not None
                    and pending['kind'] == 'declare'):
                from .choice_recovery import unchanged_declaration_correction
                if unchanged_declaration_correction(
                        self, self.view.piece_region, self._client_marker_states(),
                        allow_empty=True):
                    # An unchanged Fellowship does not answer the declaration
                    # opportunity. Retain that phase while repairing the board.
                    # Keep its correction buffer until the whole adoption
                    # succeeds, so failed recovery retains all input evidence.
                    choices = [name for name in choices if name != '_declare_phase']
            if (self.phase in ('guide', 'eyes') and pending is not None
                    and pending['kind'] == 'eye'
                    and not (self.pending_buf and self.pending_buf.records)):
                # A clean Hunt-allocation window is a retained turn phase,
                # not an unresolved physical effect preventing board repair.
                choices = [name for name in choices if name != '_eye_allocation_phase']
            choices.extend(name for name in ('deferred', 'chat_deferred')
                           if getattr(self, name, None) is not None)
            return choices
        if unfinished():
            # A board image cannot answer an invisible choice or erase Hunt
            # damage. First let newly arrived native evidence resolve it via
            # the same routing and matcher used during ordinary play.
            if getattr(self, '_combat_recovery_active', False) \
                    or self._feed_seq < len(self.feed.lines):
                items.append(('note', 'resync incomplete: pending feed is still routing; '
                                      'unfinished decisions and turn retained'))
                return items
            self._combat_recovery_active = True
            self._combat_recovery_failed = False
            failed = False
            try:
                items.extend(self._consume_inject_acks())
                sc = self.sidecar
                if sc is not None:
                    sc.poll()
                ready = self._poll_routable_feed()
                for seq, raw in ready:
                    self._feed_seq = seq
                    try:
                        items.extend(self._ingest_line(raw, sc))
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as error:
                        failed = True
                        items.append(('note', 'resync feed line %d failed: %s'
                                      % (self._feed_seq, error)))
                if getattr(self, '_sidecar_wait_seq', None) is not None:
                    failed = True
                    items.append(('note', 'resync waits for native metadata; unfinished choices retained'))
                if not failed and not self._combat_recovery_failed:
                    self._retry_blocked()
                failed = failed or self._combat_recovery_failed
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                failed = True
                items.append(('note', 'resync evidence retry failed: %s' % error))
            finally:
                self._combat_recovery_active = False
                self._combat_recovery_failed = False
                if self._out_queue:
                    items.extend(self._out_queue)
                    self._out_queue = []
            remaining = unfinished()
            if (remaining and not failed and self.deferred is None
                    and self.chat_deferred is None):
                items.extend(self._recover_fellowship_choice())
                remaining = unfinished()
            if failed or remaining:
                reason = ', '.join(remaining) or 'the retained client result'
                items.append(('note', 'resync incomplete: %s still needs its legal '
                              'choice or native result; the board alone cannot '
                              'prove that choice, so decisions, rolls and turn retained'
                              % reason))
                return items
            if self.combat_recovery_open():
                recovered, report = self.reset_and_resync(hand_turn=False)
                return items + recovered + [('note', row) for row in report]
            # Routing may have changed projected positions. Read the complete
            # board again before performing the ordinary adoption transaction.
            if self.view.dump_path:
                try:
                    evidence = dump_signature(self)
                    boundary = prepare_epoch(self, evidence)
                except (OSError, ValueError) as error:
                    items.append(('note', 'resync incomplete: native board unavailable: %s' % error))
                    return items
            if not self.view.refresh_dump():
                items.append(('note', 'resync incomplete: board changed while '
                                      'settling retained evidence'))
                return items
            if evidence is not None:
                try:
                    if not same_dump(self, evidence):
                        raise ValueError('native board or feed changed while settling retained evidence')
                    validate_recovery_board(self, self.view.piece_region)
                except ValueError as error:
                    items.append(('note', 'resync incomplete: %s; engine retained' % error))
                    return items
        # USER RULING 2026-08-21: "the resync is supposed to take the
        # CLIENT state and give it to the engine, not the other way
        # around".  Adopting into the VIEW alone left every divergence
        # that lives in ENGINE state (guide, armies, companions)
        # unreachable, which is why the button never unstuck anything.
        # Transactional + census-gated: adopt.adopt_client_board writes
        # only when the divergence count improves, so a partial adopter
        # can decline but never corrupt.
        from . import adopt as _adopt
        recovery_snapshot = _adopt._snapshot_armies(self.es)
        absent = object()
        pending_snapshot = {key: getattr(self.es, key, absent)
                            for key in self._PENDING_FIELDS}
        import copy as _copy_recovery
        hand_snapshot = [(model, _copy_recovery.deepcopy(
            vars(model), {id(self.es): self.es}))
            for model in self.hand_models.values()]
        try:
            import ai as _ai9
            # Manual resync takes native positions and marker states as given.
            # Automatic self-repair can explicitly retain engine politics.
            _dp_r = getattr(self.view, 'dump_path', None)
            _inv_r = {str(v): str(k) for k, v in
                      (getattr(self.view, 'card_pid_map', None)
                       or {}).items()}
            if boundary is not None:
                _inv_r.update(boundary['card_ids'])
            adopter = (lambda *args, **kwargs: adopt_measured(self, evidence, *args, **kwargs)
                       if evidence is not None else _adopt.adopt_client_board(*args, **kwargs))
            _rep, _nb, _na = adopter(
                self.es, self.view.piece_region,
                dynamic_types=self.view.dynamic_types,
                ai_module=_ai9,
                guide_hook=self.reconcile_guide_from_client,
                dump_path=_dp_r,
                owed_tracks={p: v for p, (v, _dl) in
                             (getattr(self, '_track_owed', None)
                              or {}).items()},
                owed_pawn=(getattr(self, '_pawn_owed', None)
                           or (None,))[0],
                manual=True, pid_to_cid=_inv_r,
                mint=getattr(self.hands, '_mint', None),
                human_side=self.human_side,
                skip_political=politics_from_engine,
                pending_fields=self._PENDING_FIELDS,
                dice_pids=self._resync_dice_pids(), dice_phase=self.phase)
            outcome = self._adoption_outcome(_rep, pid_to_cid=_inv_r)
            if evidence is not None and not same_dump(self, evidence):
                raise ValueError('native board or feed changed during recovery adoption')
            if (outcome['complete'] and self.phase == 'declare'
                    and getattr(self.es, '_declare_phase', False)):
                outcome['waiting_for'] = 'declaration'
            self._last_resync_adoption = outcome
            if self._last_resync_adoption['complete']:
                from .table_recovery import cleanup_seats
                recovered_table_seats = cleanup_seats(
                    self.es, self.view.piece_region, _inv_r)
                self._turn_start_draws_open = False
                self.seg = Segmenter(nick=self.nick)
                self._ring_gesture_records = None
                self._staged_house_window = None
                self._staged_event_window = None
                # Accepted native faces cancel any abandoned activation toggle.
                self._pol_face_walked = set()
                # This successful, explicit manual transaction is a known
                # activation writer. Keep future unexplained-write detection.
                self._pol_active_prev = (
                    {getattr(n, 'name', str(n)): bool(value)
                     for n, value in self.es.political.active.items()},
                    int(getattr(self, 'n_commits_total', 0) or 0))
            for _line in _rep:
                items.append(('note', 'resync-adopt: %s' % _line))
                if 'DROPPED after adoption' in str(_line):
                    self.findings.append(('adopt_pending_dropped',
                                          str(_line)[:100]))
            self.findings.append(('resync_engine_adopted' if
                                  self._last_resync_adoption['accepted'] else
                                  'resync_engine_not_adopted',
                                  'divergences %d -> %d' % (_nb, _na)))
            if outcome['complete'] and boundary is not None:
                skipped = commit_epoch(self, boundary)
                if skipped:
                    self.findings.append(('resync_feed_drained', '%d covered line(s)' % skipped))
                    items.append(('note', 'resync-recovery: %d snapshot-covered feed line(s) retired' % skipped))
        except Exception as _e9:
            self._last_resync_adoption = {'accepted': False, 'complete': False}
            items.append(('note', 'resync-adopt FAILED: %r' % (_e9,)))
        if not self._last_resync_adoption.get('complete'):
            _adopt._restore_armies(self.es, recovery_snapshot)
            for key, value in pending_snapshot.items():
                if value is absent:
                    vars(self.es).pop(key, None)
                else:
                    setattr(self.es, key, value)
            for model, fields in hand_snapshot:
                vars(model).clear()
                vars(model).update(fields)
            self._last_resync_adoption['accepted'] = False
            items.append(('note', 'resync incomplete; original state and pending '
                          'evidence retained: %s' %
                          (self._last_resync_adoption.get('remaining') or
                           'client board was not adopted')))
            self.findings.append(('resync_recovery', 'incomplete; rolled back'))
            return items

        # The server uses this epoch to invalidate open AI steps and their
        # issued native requests. A failed attempt must retain those requests:
        # repeating an already issued draw can draw another physical card.
        from .table_recovery import rebase_cleanup
        rebase_cleanup(self, recovered_table_seats)
        self._resync_ts = _tr.time()
        # Only a successful board transaction retires its old evidence. Do not
        # unreveal buffered cards here: the adopted hand already reflects their
        # actual positions, so restoring the old hand would undo the recovery.
        self._uncommitted_card_selection = None
        self._grouped_die_selection = None
        self._grouped_die_identity_wait = None
        self._repair = None
        self._census_offer = None
        self.pending_buf = None
        self.deferred = None
        self.chat_deferred = None
        self.deferred_tail = []
        self._out_queue = []
        self._pol_recon_inflight = {}
        self._track_owed = {}
        self._pawn_owed = None
        self._reconcile_due = True
        self._recon_prev_sig = None
        self._recon_sig_since = None
        self._recon_sig_reads = 0
        self._census_declined_sig = None
        n_unc = len(self.uncommitted)
        if n_unc:
            self.findings.append(('resync_uncommitted_cleared',
                                  '%d line(s)' % n_unc))
            self.uncommitted = []
        for e in self.echo_ledger:
            self.findings.append(('mirror_echo_unclaimed',
                                  'resync-recovery: %r' % (e,)))
        self.echo_ledger = []
        self._post_commit_tail = ()
        self._human_casualty_prefix = None
        self._ai_casualty_prefix = None
        # a rules-owed hand-limit pending reopens (the client kept
        # playing; the engine still owns the rule)
        try:
            gl.open_hand_limit_discard(self.es)
        except Exception:
            pass
        # Explicit recovery must not overwrite the user's board with old
        # engine markers after adoption failed or left unresolved rows.
        if self._last_resync_adoption.get('accepted'):
            items.extend(self._political_recon())
            items.extend(self._guide_box_recon())
        if not self._last_resync_adoption.get('complete'):
            items.append(('note', 'resync incomplete: %s' %
                          (self._last_resync_adoption.get('remaining') or
                           'client board was not adopted')))
        self.findings.append(('resync_recovery', 'completed' if
                              self._last_resync_adoption.get('complete') else 'incomplete'))
        return items

    # ---- SELF-REPAIR (USER 2026-08-28) ---------------------------------
    def stall_description(self):
        """THE explicit answer to 'what exactly are you waiting on':
        the human side and pending kind the engine reports, how many
        legal options it has and of which action types, the buffered
        lines that match none of them, and the gesture/chat that would
        satisfy it.  One composer for the status line, the stall note,
        the stall log and the chat offer."""
        import ai
        d = {'side': None, 'kind': None, 'n_options': 0,
             'option_types': [], 'records': [], 'hint': '',
             'stalled_ticks': int((getattr(self, '_buf_stall', None)
                                   or (None, 0))[1])}
        pb = self.pending_buf
        d['records'] = [str(r.get('norm', ''))[:90]
                        for r in (getattr(pb, 'records', None) or [])]
        try:
            dp = ai.describe_pending(self.es, self.human_side)
        except Exception:
            dp = None
        if dp is not None:
            d['side'] = getattr(dp.get('side'), 'name', str(dp.get('side')))
            d['kind'] = str(dp.get('kind'))
            acts = list(dp.get('actions') or ())
            d['n_options'] = len(acts)
            d['option_types'] = sorted(set(
                str(getattr(a, 'action_type', '')) for a in acts))[:6]
            kl = d['kind'].lower()
            for key, hint in _PENDING_HINTS:
                if key in kl:
                    d['hint'] = hint
                    break
            if not d['hint']:
                d['hint'] = ('perform one of its %d legal options on the '
                             'board' % len(acts))
            if d['kind'] == 'combat:siege_trim':
                d['hint'] = self._pending_words(dp)
        return d

    def _pending_human_roll_wait(self):
        from .pending_roll_wait import waiting
        return waiting(self)

    def _partial_declaration_wait(self, pending):
        from .choice_recovery import declaration_in_progress
        return declaration_in_progress(self, pending)

    def _partial_casualty_wait(self):
        from .partial_casualty_status import waiting
        return waiting(self)

    def _casualty_roll_context_only(self):
        """Read-only classification of consumed dice while awaiting casualties."""
        pc = getattr(self.es, 'pending_combat', None) or {}
        records = getattr(self.pending_buf, 'records', None) or []
        if (pc.get('phase') != 'casualty_allocation' or not records
                or any(record.get('op') != 'roll_num' for record in records)
                or self.rand is None or self.rand.pending() is not None
                or self.rand.unconsumed().get('d6', 0)):
            return None
        import ai
        pending = ai.describe_pending(self.es, self.human_side)
        return pending if pending and pending['kind'] == 'combat:casualty_allocation' else None

    def _discard_identity_wait(self):
        from .discard_wait_status import waiting
        return waiting(self)

    def stall_text(self, d=None):
        d = d or self.stall_description()
        from .discard_wait_status import text as _discard_wait_text
        discard_wait = _discard_wait_text(self)
        if discard_wait is not None:
            return discard_wait
        from .partial_casualty_status import text as _partial_casualty_text
        partial = _partial_casualty_text(self)
        if partial is not None:
            return partial
        if d['kind'] == 'reveal_place':
            import ai
            from .decision_prompts import pending_instructions
            pending = ai.describe_pending(self.es)
            if pending and pending['side'] is self.human_side:
                return pending_instructions(pending, self.es)
        casualty = self._casualty_roll_context_only()
        if casualty is not None:
            from .decision_prompts import pending_instructions
            from emitter import casualty_context
            context = casualty_context(self.es)
            instruction = pending_instructions(casualty, self.es)
            if context is not None:
                return '%d hit%s to allocate at %s: %s' % (
                    context['hits'], '' if context['hits'] == 1 else 's',
                    context['region'], instruction)
            return instruction
        pending_human_roll = self._pending_human_roll_wait()
        if pending_human_roll is not None:
            count = int(pending_human_roll.kw['need'])
            return ('your physical choice is recorded; waiting for the requested combat dice: your requested roll of %d %s'
                    % (count, 'die' if count == 1 else 'dice'))
        proven = self._proven_pending_random_call()
        if proven is not None:
            label = {'d6':'combat dice','draws':'card draw','tile':'Hunt tile',
                     'randomcomp':'random Companion selection'}.get(proven.kind,proven.kind)
            return ('your physical choices are recorded; waiting for the requested %s' % label)
        pending_roll = getattr(self.pending_buf, '_relentless_starve_call', None)
        if pending_roll is not None:
            pc = getattr(self.es, 'pending_combat', None) or {}
            if ((pc.get('effect_choice') or {}).get('effect') == 'relentless_assault'
                    and getattr(pending_roll, 'kind', None) == 'd6'):
                return ('your physical Relentless Assault payment is recorded; '
                        'waiting for the requested combat dice')
        if self._card_completion_wait():
            return ('card movements are buffered; finish the remaining moves '
                    'and say done to resolve this card choice')
        recs = ' | '.join(d['records'][:2]) or '(none)'
        if d['kind'] is None:
            return ('no engine decision is open; %d line(s) buffered: %s'
                    % (len(d['records']), recs))
        return ("%s's %s decision: your %d line(s) [%s] match none of its "
                "%d legal options (%s); expected: %s; or undo the line"
                % (d['side'], d['kind'], len(d['records']), recs,
                   d['n_options'], ', '.join(d['option_types']) or 'n/a',
                   d['hint']))

    def _political_activation_witness(self):
        """Record activation changes and flag writes outside engine commits.

        Committed actions can legitimately activate nations. Those changes
        are informational; a change without a commit remains an error and
        raises the divergence alarm used to detect reconciliation leaks.
        """
        _pt1 = getattr(getattr(self.es, 'political', None),
                       'active', None)
        if not isinstance(_pt1, dict):
            return
        cur = {getattr(n, 'name', str(n)): bool(v)
               for n, v in _pt1.items()}
        prev = getattr(self, '_pol_active_prev', None)
        n_com = int(getattr(self, 'n_commits_total', 0) or 0)
        if prev is not None and cur != prev[0]:
            changed = sorted(k for k in set(cur) | set(prev[0])
                             if cur.get(k) != prev[0].get(k))
            labels = [str(c[2]) for c in
                      (self.commits or [])[-(n_com - prev[1]):]] \
                if n_com > prev[1] else []
            detail = '%s :: commits since last check: %s' % (
                ','.join(changed), ','.join(labels)[:80] or 'NONE')
            if n_com == prev[1]:
                self.findings.append(('political_activation_changed', detail))
                msg = ('POLITICAL ACTIVATION FLIPPED OUTSIDE ANY '
                       'COMMIT: %s - an off-engine writer '
                       '(probe/recon leak); do not trust the menu, '
                       'save this log' % ','.join(changed))
                if msg not in self.alarms:
                    self.alarms.append(msg)
            else:
                self.findings.append(('political_activation_committed', detail))
        self._pol_active_prev = (cur, n_com)

    def _note_repair_slots(self):
        """FLA-8 (par5 g4#6): first-seen TURN of every piece in a
        repair slot.  The self-repair kill list is scoped to pieces
        staged THIS turn - two T2 combat cards still sitting in
        Current Action at T4 were swept by the repair, and the journal
        replay died on the changed world.  A piece that leaves the
        slot and returns re-stamps."""
        pr = getattr(self.view, 'piece_region', None) or {}
        seen = getattr(self, '_repair_slot_turn', None)
        if seen is None:
            seen = self._repair_slot_turn = {}
        turn = int(getattr(self.es, 'turn', 0) or 0)
        now = {str(p) for p, z in pr.items()
               if str(z).strip().lower() in _REPAIR_SLOTS}
        for p in list(seen):
            if p not in now:
                seen.pop(p, None)
        for p in now:
            seen.setdefault(p, turn)

    def _repair_kill_targets(self):
        """[(pid, zone)] the repair empties: the staged die in Current
        Action and any card in the current card slots.  Table cards
        (Board slots) are legitimately in play and stay.  FLA-8: a
        piece first seen in a repair slot in an EARLIER turn is not
        this stall's staging - excluded, loudly."""
        pr = getattr(self.view, 'piece_region', {}) or {}
        self._note_repair_slots()
        seen = getattr(self, '_repair_slot_turn', None) or {}
        turn = int(getattr(self.es, 'turn', 0) or 0)
        out, stale = [], []
        for p, z in pr.items():
            if str(z).strip().lower() not in _REPAIR_SLOTS:
                continue
            if seen.get(str(p), turn) < turn:
                stale.append(str(p))
                continue
            out.append((str(p), str(z)))
        if stale:
            self.findings.append(
                ('repair_kill_scope_excluded',
                 'staged in an earlier turn, NOT killed: %s'
                 % ' '.join(sorted(stale))[:80]))
        return sorted(out)

    def _repair_turn_side(self, kills=None):
        """(to_act, last_killed) per the USER rule: the last die killed
        = the previous player's turn - a staged die the repair is about
        to kill counts as that kill; no die killed at all -> the Free
        Peoples act first."""
        import game_state as gs
        dmap = getattr(self.view, 'die_id_map', {}) or {}
        staged = [dmap[p][0] for p, z in (kills or ())
                  if p in dmap and 'action' in str(z).lower()]
        last = staged[-1] if staged \
            else side_of_last_die_kill(getattr(self.feed, 'lines', ()))
        if last is None:
            return gs.Side.FREE_PEOPLES, None
        return ((gs.Side.FREE_PEOPLES if last is gs.Side.SHADOW
                 else gs.Side.SHADOW), last)

    def _write_stall_log(self, reason, d, rows=None):
        """Step 1 of the repair, BEFORE anything is asked or touched:
        the whole game log so far plus everything the driver knows,
        in one file beside the feed, named for the moment.  `rows`:
        the census rows already read (the census lane passes its
        confirmed, reasoned rows); else one read here."""
        import os as _os
        import time as _time
        cd = _os.path.dirname(getattr(self.feed, 'path', '') or '') or '.'
        kind = re.sub(r'[^a-z0-9]+', '_',
                      str(d.get('kind') or 'nopending').lower())[:24]
        name = 'ai_stall_%s_T%s_%s.log' % (
            _time.strftime('%Y%m%d_%H%M%S'), getattr(self.es, 'turn', '?'),
            kind)
        path = _os.path.join(cd, name)
        if rows is not None:
            rows = [str(r) for r in rows]
        else:
            try:
                rows = client_board.diff_engine_vs_dump(
                    self.es, getattr(self.view, 'piece_region', {}) or {},
                    dynamic_types=getattr(self.view, 'dynamic_types', None),
                    marker_states=self._client_marker_states()) or []
            except Exception as e:
                rows = ['census unavailable: %r' % (e,)]
        feed_lines = list(getattr(self.feed, 'lines', ()) or ())
        body = ['STALL LOG  %s' % _time.strftime('%Y-%m-%d %H:%M:%S'),
                'reason: %s' % reason,
                'explicit: %s' % self.stall_text(d),
                'turn=%s phase=%s human=%s ai=%s last_actor=%s'
                % (getattr(self.es, 'turn', '?'), self.phase,
                   getattr(self.human_side, 'name', '?'),
                   getattr(self.ai_side, 'name', '?'),
                   getattr(self.last_actor, 'name', None)),
                '(the server console is ai_server_console.log beside '
                'this file)', '',
                '== engine pending ==', repr(d), '',
                '== buffered records (%d) ==' % len(d['records'])]
        body += d['records']
        body += ['', '== findings (last 60) ==']
        body += ['%s: %s' % (k, str(v)[:200])
                 for k, v in list(self.findings)[-60:]]
        body += ['', '== alarms ==']
        body += [str(a)[:240] for a in self.alarms]
        body += ['', '== census engine vs client (%d) ==' % len(rows)]
        body += [str(r)[:200] for r in rows]
        body += ['', '== FULL FEED (%d lines) ==' % len(feed_lines)]
        body += [str(l) for l in feed_lines]
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(body) + '\n')
        except OSError as e:
            self.alarms.append('STALL LOG NOT WRITTEN: %r' % (e,))
            return None
        self.findings.append(('stall_log_written', name))
        return path

    def _propose_repair(self, reason, kind='stall', rows=None, clears=None):
        """Offer correction only for a freshly measured quiet-board mismatch.

        Parser stalls, unacknowledged emissions and unfinished gestures retain
        their state and report diagnostics. They never authorize killing dice
        or dropping decisions. Every accepted offer uses explicit board consent.
        """
        import os as _os
        if self._repair is not None or getattr(self, 'spectator', False):
            return []
        from .repair_evidence import measure, diagnostic
        evidence = measure(self)
        if evidence['status'] != 'mismatch':
            return diagnostic(self, reason, evidence)
        # A parser/transport stall cannot authorize dropping an action or
        # killing its die. Only a measured board mismatch can offer the
        # existing explicit-consent census correction.
        kind = 'census'
        rows = self._census_row_lines(evidence['rows'], 0)
        clears = 'board'
        d = self.stall_description()
        log_path = self._write_stall_log(reason, d, rows=rows)
        log_name = _os.path.basename(log_path) if log_path else 'unsaved'
        if kind == 'census':
            rows = [str(r) for r in (rows or ())]
            self._repair = {'phase': 'offered', 'kind': 'census',
                            'reason': reason, 'log': log_path,
                            'kills': [], 'to_act': None,
                            'last_killed': None, 'kill_ticks': 0,
                            'rows': rows, 'evidence': evidence,
                            'clears': clears or 'board'}
            self.findings.append(('census_adopt_offered',
                                  '%d row(s) :: log=%s'
                                  % (len(rows), log_name)))
            items = [('note', 'BOARD DIVERGENCE: %d row(s) confirmed - '
                              'say yes to adopt the client board into '
                              'the engine, or no to keep the engine '
                              '(log %s)' % (len(rows), log_name))]
            _ask = getattr(chat_mod, 'census_adopt_ask', None)
            if _ask is not None:
                items.append(('chat', 'prompt', _ask(
                    len(rows), rows[0] if rows else '', log_name)))
            elif not getattr(self, '_census_ask_panel_only_said', False):
                # chat prose is builder-registered in chat.py; until
                # the builder lands the ask lives on the status line
                # and the panel (the human answers yes / no in chat)
                self._census_ask_panel_only_said = True
                self.findings.append(('census_ask_panel_only',
                                      'chat.census_adopt_ask missing - '
                                      'ask on the status line only'))
            return items
    def _repair_begin(self):
        """Recheck the physical board before executing explicit correction consent."""
        if self.forced_ai_rows_pending():
            self._repair = None
            items, report = self.reset_and_resync(hand_turn=False)
            return items + [('note', line) for line in report]
        from .repair_evidence import measure, diagnostic
        r = self._repair
        if r is None:
            return []
        if r.get('kind') != 'census':
            self._repair = None
            return diagnostic(self, 'old parser-stall repair withdrawn',
                              {'status': 'unavailable', 'reason': 'no confirmed physical-board correction', 'rows': []})
        evidence = measure(self)
        if evidence['status'] != 'mismatch':
            self._repair = None
            return diagnostic(self, 'board correction rechecked', evidence)
        previous = (r.get('evidence') or {}).get('rows')
        if previous is not None and sorted(previous) != sorted(evidence['rows']):
            self._repair = None
            return self._propose_repair('physical mismatch changed; review the current rows', kind='census')
        r['rows'] = self._census_row_lines(evidence['rows'], 0)
        r['evidence'] = evidence
        from emitter import render as _render
        import emitter as _em
        r = self._repair
        if r.get('kind') == 'census':
            # BRCMP design 3: consent -> the ONE transactional adoption
            # (pieces + the rules flags they imply); no kill, no turn
            # change - the board was quiescent when it was asked
            self._census_offer = None
            r['phase'] = 'adopt'
            n_rows = len(r.get('rows') or ())
            items = [('note', 'census: adopting the client board into '
                              'the engine (your consent, %d row(s))'
                      % n_rows)]
            items.extend(self._adopt_board_now('consented', expected_evidence=evidence))
            self.findings.append(('census_adopt_consented',
                                  '%d row(s)' % n_rows))
            self._repair = None
            self._census_declined_sig = None
            self._reconcile_due = True
            _done = getattr(chat_mod, 'census_adopt_done', None)
            if self._last_board_adoption.get('complete') and _done is not None:
                items.append(('chat', 'prompt', _done(n_rows)))
            else:
                remaining = self._last_board_adoption.get('remaining')
                items.append(('chat', 'prompt',
                              'board recovery is incomplete: %s; use RESYNC after checking the board' %
                              (remaining[0] if remaining else 'the board could not be adopted')))
            return items
        self.view.refresh_dump()
        r['kills'] = self._repair_kill_targets()
        r['to_act'], r['last_killed'] = self._repair_turn_side(r['kills'])
        r['phase'] = 'kill'
        items = [('note', 'self-repair: killing %d staged piece(s) %s'
                          % (len(r['kills']), r['kills']))]
        if r['kills']:
            pids = [p for p, _z in r['kills']]
            items.append(('line', _render(
                'kill', {'nick': self.nick, 'pieces': pids,
                         'to': _em.DICE_USED}, local=False)))
            self._repair_kill_echo = {'pids': set(pids),
                                      'until_seq': self._feed_seq + 400}
            dmap = getattr(self.view, 'die_id_map', {}) or {}
            for p in pids:
                si = dmap.get(p)
                if si is None:
                    continue
                try:
                    self.es.dice[si[0]][si[1]].used = True
                except (KeyError, IndexError, AttributeError):
                    pass
        return items

    def _repair_consumes_echo(self, rec):
        """Our own repair kill, echoed back by the client: consumed
        here, never a gesture."""
        rk = self._repair_kill_echo
        if not rk or rec.get('op') != 'kill' \
                or str(rec.get('player', '')) != self.nick:
            return False
        if self._feed_seq > rk.get('until_seq', 0):
            self._repair_kill_echo = None
            return False
        self._repair_kill_echo = None
        self.findings.append(('stall_repair_kill_echo',
                              str(rec.get('norm', ''))[:80]))
        return True

    def _repair_tick(self):
        r = self._repair
        if r is None:
            return []
        if r.get('kind') != 'census':
            self._repair = None
            from .repair_evidence import diagnostic
            return diagnostic(self, 'legacy parser-stall repair withdrawn',
                              {'status': 'unavailable', 'reason': 'retain the action; physical divergence has not been proven', 'rows': []})
        if r['phase'] == 'offered':
            # the offer is withdrawn when the condition it was made
            # for CLEARED (r['clears'], see _propose_repair)
            clears = r.get('clears') or 'buffer'
            if clears == 'board':
                # withdrawn by the census lane itself (a clean or a
                # changed quiescent read), never by the pending buffer
                return []
            if clears == 'pending':
                import ai as _aip
                try:
                    cleared = _aip.describe_pending(self.es) is None
                except Exception:
                    cleared = False
            elif clears == 'hold':
                cleared = not getattr(self, 'holds', None)
            elif clears == 'echo':
                _ep = getattr(self, 'echo_pending', None)
                cleared = not (callable(_ep) and _ep())
            elif clears == 'manual':
                # BRLANE-13 / BRJAVA-1: a refused or unrendered AI line
                # and a client complaint are bug reports - only the
                # human's yes / no / resync closes the offer
                cleared = False
            else:
                pb = self.pending_buf
                cleared = pb is None or not getattr(pb, 'records', None)
            if cleared:
                # the human cleared it themselves (undo / redo)
                self._repair = None
                self.findings.append(('stall_repair_withdrawn', clears))
                return [('note', 'self-repair offer withdrawn - the '
                                 'stall cleared (%s)' % clears)]
            return []
        if r['phase'] == 'kill':
            r['kill_ticks'] += 1
            self.view.refresh_dump()
            pr = getattr(self.view, 'piece_region', {}) or {}
            still = [p for p, _z in r['kills']
                     if str(pr.get(p, '')).strip().lower()
                     in _REPAIR_SLOTS]
            if still and r['kill_ticks'] < 50:
                return []
            out = []
            if still:
                self.findings.append(('stall_repair_kill_unconfirmed',
                                      '%d piece(s) after %d ticks: %s'
                                      % (len(still), r['kill_ticks'],
                                         still)))
                out.append(('note', 'self-repair: the client still shows '
                                    '%d staged piece(s) - continuing'
                            % len(still)))
            r['phase'] = 'resync'
            return out + self._repair_finish()
        return []

    def _repair_finish(self):
        """Steps 4-6: reset + resync client-as-is (political track from
        the engine, pushed to the client), the turn from the last die
        killed, announce."""
        import os as _os
        r = self._repair
        self._clear_echo_expect('self-repair')     # BRLANE-13
        items, report = self.reset_and_resync(politics_from_engine=True,
                                              hand_turn=False)
        to_act = r['to_act']
        complete = self._last_resync_adoption.get('complete')
        if complete:
            self.last_actor = (self.human_side if to_act is self.ai_side
                               else self.ai_side)
        self._buf_stall = None
        log_name = (_os.path.basename(r['log']) if r.get('log')
                    else 'unsaved')
        self.findings.append(('stall_repair_done' if complete else 'stall_repair_incomplete',
                              'log=%s kills=%d to_act=%s'
                              % (log_name, len(r['kills']), to_act.name)))
        self._repair = None
        out = list(items)
        out += [('note', 'self-repair: %s' % l) for l in report]
        if self._last_resync_adoption.get('complete'):
            out.append(('chat', 'prompt', chat_mod.stall_repair_done(
                to_act.name.replace('_', ' ').lower(), len(r['kills']),
                log_name)))
        else:
            out.append(('chat', 'prompt', 'recovery is incomplete; check the resync report before continuing'))
        return out

    def _witness_report(self):
        if not self.view.refresh_dump():
            return [('note', 'no stable board dump')]
        rows = client_board.diff_engine_vs_dump(
            self.es, self.view.piece_region,
            dynamic_types=self.view.dynamic_types,
            marker_states=self._client_marker_states())
        if rows:
            self.alarms.append('WITNESS: %d discrepancies' % len(rows))
            return [('note',
                     'desync: %d rows (first: %s)'
                     % (len(rows), rows[0]))]
        return [('note', 'engine and client agree')]
