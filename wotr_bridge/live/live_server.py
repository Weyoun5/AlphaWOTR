"""live_server — the sided-play entry point the client launches.

Wires the whole stack:  FileBridge transport  ->  LiveDriver (feed
router)  ->  nets (engine_play.load_nets + ai.mcts_action_with_policy)
->  emission (engine events -> gestures)  ->  ClientRandomness
(!CALL directives, deterministic retries)  ->  ChatOut (the single
whitelisted chat exit).

THE STEP PROTOCOL (PROTOCOL.md, deterministic-retry contract):
  * every AI decision applies on a CLONE (ai.apply_action) — a
    NeedClientCall leaves the driver's engine state UNTOUCHED;
  * the pick is journal-pinned (randomness.journal_pick), so a step
    retried across ticks repeats the previous attempt exactly;
  * emission goes out ONCE per step (the play_event staging must
    precede the effect's randomness — donor :15874-15900); directives
    go out once per distinct pending call;
  * the step commits (state adopted + feed prefixes dropped) only
    after a successful apply; the unconsumed() honesty gate runs
    first — excess client randomness is an alarm, never silence.

BOOT: a FRESH game per process (USER bug A1: never a prior session's
state).  Sims default treats exactly-1000 as 'default' (client fact
17: AiOpp hardcodes -Dai.sims=1000).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'src'))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import card_identity
from . import chat as chat_mod
from . import emission
from . import randomness as rnd_mod
from .driver import DriverHalt, LiveDriver


def _fresh_state(seed=None):
    import board as board_mod
    import game_logic
    import game_state as gs
    import random
    if seed is not None:
        random.seed(seed)
    st = gs.setup_initial_game(board_mod.load_board())
    game_logic.initialize_decks(st)
    return st


# ----------------------------------------------------------------------
# PHASE-3 status-menu publication (the ONE sanctioned cross-process
# answer to "what is legal for the human right now"): the engine's own
# menu (ai.describe_pending / ai.get_legal_actions) summarized into the
# panel status JSON via the transport's existing status_extra hook.
# This is I1 - the engine answering legality - published so a harness
# playing the HUMAN seat derives its board gestures from the CURRENT
# position (USER gesture ruling 2026-08-19: never from copied log
# lines, never from a "legal-looking" guess).  NO hand identities ride
# the file (design rule 8): card-identity params are redacted at the
# single exit below.
# ----------------------------------------------------------------------
_MENU_CAP = 900          # published army_move entries (counts stay
#                          exact; sized over the largest measured live
#                          menu - 861 Shadow army actions, gesture_match
#                          log).  The cap is PER TYPE on army_move only:
#                          every other action type is always published
#                          whole (a 1183-action turn-1 Shadow menu lost
#                          its advance_political entries to a flat cap)


def _json_safe(v):
    """Plain-JSON projection of an action param value (enums -> their
    value/name; containers recursed; anything else stringified)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_json_safe(x) for x in v]
    val = getattr(v, 'value', None)
    if isinstance(val, (str, int, float)):
        return val
    nm = getattr(v, 'name', None)
    if isinstance(nm, str):
        return nm
    return str(v)


def _tile_face(tile):
    """Normalized tile identity (donor ai_move_server._tile_face,
    verbatim semantics): special_name FIRST (HuntTile repr calls 'The
    Ring is Mine!' an Eye), accents/punctuation folded so engine
    names meet the corpus piece table."""
    import re as _re
    import unicodedata as _ud
    s = (getattr(tile, 'special_name', None) or
         getattr(tile, 'label', None) or
         getattr(tile, 'name', None) or
         getattr(tile, 'code', None) or tile)
    s = _ud.normalize('NFKD', str(s))
    s = s.encode('ascii', 'ignore').decode('ascii')
    return _re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def _menu_params_safe(params):
    """Param dict for publication: card identities NEVER ride the
    status file (rule 8 fail-closed - redacted even for placeholder
    ids, so no future param can leak by omission)."""
    out = {}
    for k, v in (params or {}).items():
        kl = str(k).lower()
        if 'card' in kl or kl == 'cid':
            out[str(k)] = '<hidden>'
        else:
            out[str(k)] = _json_safe(v)
    return out


class LiveServer(object):
    def __init__(self, client_dir, ai_side, net_tag='auto', sims=300,
                 seed=None, load_nets=True, scenario=None):
        # K-S1 scenario mode: the fixture's own prefix replay is the
        # REFERENCE the live flip is verified against (one lane,
        # scenario.replay_prefix, for the cutter and the server)
        self.scenario = scenario
        self._scenario_ref = None
        self._scenario_pid_cid = {}
        self._flip_due_since = None
        self._flip_note = None
        if scenario is not None:
            from . import scenario as scen_mod
            if scenario.ai_side_name != ai_side.name:
                raise DriverHalt(
                    'SCENARIO %s: the fixture has the AI playing %s but '
                    'this server was launched with --side for %s'
                    % (scenario.name, scenario.ai_side_name,
                       ai_side.name))
            if scenario.cut > 0:
                ref = scen_mod.replay_prefix(
                    scenario.feed_path, scenario.cut,
                    scenario.ai_side_name,
                    seed=(seed if seed is not None else 1234),
                    meta_path=scenario.meta_path,
                    expected=scenario.prefix)
                probs = ref.prefix_problems()
                if probs:
                    raise DriverHalt(
                        'SCENARIO %s: the fixture prefix does not replay '
                        'clean through the prefix lane: %s'
                        % (scenario.name, ' | '.join(probs[:6])))
                self._scenario_ref = scen_mod.position_summary(ref)
        # One-shot panel epochs stamped BEFORE this server existed are a
        # previous session's presses: adopt them as watermarks without
        # acting (a restart must never replay a stale RESET / RESYNC /
        # SKIP / whose-turn press).
        try:
            import json as _json0, os as _os0
            _cp0 = _os0.path.join(client_dir, 'ai_panel_control.json')
            _c0 = (_json0.load(open(_cp0, encoding='utf-8'))
                   if _os0.path.exists(_cp0) else {})
        except Exception:
            _c0 = {}
        # absent key = ON (the suites / a panel-less boot keep the gate);
        # the panel writes its checkbox state at ITS boot (default OFF)
        self._enforce_human_boot = bool(_c0.get('enforce_human', True))
        for _key, _attr in (('resync', '_resync_epoch'),
                            ('skip_pending', '_skip_epoch'),
                            ('reset_resync', '_reset_epoch'),
                            ('whose_turn_epoch', '_whose_turn_epoch')):
            try:
                _e = float(_c0.get(_key, 0) or 0)
            except (TypeError, ValueError):
                _e = 0.0
            if _e:
                setattr(self, _attr, _e)
        import game_state as gs
        self.client_dir = client_dir
        self.ai_side = ai_side
        self.human_side = (gs.Side.SHADOW
                           if ai_side is gs.Side.FREE_PEOPLES
                           else gs.Side.FREE_PEOPLES)
        self.nick = ('WOTR_AI_FP'
                     if ai_side is gs.Side.FREE_PEOPLES
                     else 'WOTR_AI_Shadow')
        self.sims = sims
        feed_path = os.path.join(client_dir, 'ai_feed.txt')
        dump_path = os.path.join(client_dir, 'ai_client_board.txt')
        self.rand = rnd_mod.ClientRandomness(
            human_side=self.human_side.name, seed=seed)
        self.rand.install()
        self.chat_out = chat_mod.ChatOut()
        if scenario is not None:
            from .scenario import ScenarioDriver
            self.driver = ScenarioDriver(
                _fresh_state(seed), ai_side, feed_path, self.rand,
                scenario.prefix, chat_out=self.chat_out,
                dump_path=dump_path, nick=self.nick)
        else:
            self.driver = LiveDriver(_fresh_state(seed), ai_side,
                                     feed_path, randomness=self.rand,
                                     chat_out=self.chat_out,
                                     dump_path=dump_path, nick=self.nick)
        # CLIENT-WAVE sidecar (WAVE-J adoption): the annotated meta
        # channel; absent file = channel off (old jars / suites) and
        # the heuristic lanes stay - the absence is NOTED once live
        from .sidecar import SidecarReader
        self.driver.sidecar = SidecarReader(feed_path + '.meta.jsonl')
        # USER 2026-08-22: legality enforcement for the human is the
        # panel's 'Enforce legal moves for HUMAN player (beta)' box -
        # absent from the control file = OFF (the board is taken as
        # given after each human action)
        self.driver.enforce_human = bool(
            getattr(self, '_enforce_human_boot', True))
        # sanctioned raw import #1: the AI's OWN draw identities via
        # the commitment decode (AiBoot claims the AI hand with 'ai')
        self.cards = card_identity.CardIdentity(self.ai_side.name)
        self.fp_net = self.sh_net = None
        if load_nets:
            import engine_play
            self.fp_net, self.sh_net = engine_play.load_nets(net_tag)
        self.notes = []             # panel-only chatter (never chat)
        self._em_sent = False       # this step's emission went out
        self._dirs_sent = set()     # pending-call identities sent
        # '!CALL' texts THIS STEP's emission already ordered - the
        # directive builder must not order them again (WAVE-H cbt1:
        # render_action draw_card's staged '!CALL draw_sa_strat' PLUS
        # _dir_lines answering the starved draw funnel = the client
        # drew TWO cards per action; honesty gate fired EXCESS
        # RANDOMNESS and the SA hand ran one card ahead of the engine)
        self._step_calls = {}       # directive text -> feed gen at order
        # WAVE-I: ops of a PRE-APPLY emission, kept across the step's
        # cross-tick retries for the post-commit sweep's subtraction
        self._step_expressed = []
        self._step_owed = []        # owed kills THIS step banked
        # K1: the AI's own combat card is CHOSEN (hidden) on the client
        # until the round's reveal directive; one reveal per round
        self._ai_cc_chosen = False
        self._reveal_round = None
        self._open_step = None      # a starved AI step retried per tick
        self._step_starved = 0      # ticks the open step has starved
        self._step_queue = []       # AI steps arriving while one is open
        # window/pending funnel asks already ordered, keyed by
        # (commit epoch, call identity): within one epoch a starving
        # ask is ordered ONCE (a re-sent roll the client DID execute
        # would roll twice); a NEW commit opens a new epoch so a later
        # identical ask (next turn's hunt) is not falsely deduped —
        # the AI-step _dirs_sent set stays step-scoped and separate.
        self._window_asks_sent = set()

    # ------------------------------------------------------------------
    def handle(self, request):
        """One transport cycle.  `request` is the transport's Request
        namedtuple (production: FileBridge.serve) or a dict with
        'lines' (the offline suites); either carries the raw client
        log lines (sanctioned imports only).  Returns RESP lines."""
        # SIDE-HEADER GUARD (manual-play delivery 2026-08-20).  The
        # client's 'Play vs AI...' dialog asks the HUMAN which side the
        # AI plays, and that click is INDEPENDENT of the launcher's
        # --side / -Dai.feed.aiside.  A contradicting click used to be
        # served silently (reproduced: server --side SA answered a
        # LIVE_FREE_PEOPLES request with '!CALL draw_sa_char'), and the
        # rule-8 fail-closed check does NOT catch it - every #HAND line
        # IS blinded, just the WRONG side's, so the human's hand would
        # be fully readable to the AI.  The header names the AI's side
        # (AiOpp.header()); disagreeing with it is fail-closed.
        _hdr = (request.get('side') if isinstance(request, dict)
                else getattr(request, 'side', None))
        if _hdr:
            _want = ('LIVE_SHADOW' if self.ai_side.name == 'SHADOW'
                     else 'LIVE_FREE_PEOPLES')
            _bare = _want[len('LIVE_'):]
            if str(_hdr) not in (_want, _bare, 'AUTO'):
                raise DriverHalt(
                    'SIDE MISMATCH: this server was launched with the '
                    'AI playing %s, but the client says the AI plays '
                    '%s (request header %s).  Playing on would blind '
                    'the AI to the wrong hand and expose yours.  Stop '
                    'the AI, close the client, and either re-run the '
                    'launcher for the other side or pick "AI plays '
                    '%s" in the Play vs AI dialog.'
                    % (self.ai_side.name, str(_hdr).replace('LIVE_', ''),
                       _hdr, self.ai_side.name.replace('_', ' ').title()))
        # a randomness LIVE HALT (deck/pool desync, journal mismatch,
        # tripwire) is STICKY: no cycle plays on past it until the
        # operator's RESET clears it (the flag was written and never
        # read - the next tick proceeded as if nothing had happened)
        if self.rand.sticky_halt \
                and not getattr(self, '_reset_requested', False):
            raise DriverHalt('%s  [sticky - press RESET + RESYNC to '
                             'clear]' % self.rand.sticky_halt)
        if isinstance(request, dict):
            lines = list(request.get('lines', ()) or ())
        else:
            # transport.Request namedtuple (Phase-3 seam fix: the old
            # request.get() died with AttributeError inside serve()'s
            # catch-all on EVERY real transport cycle - no RESP, ever)
            lines = list(getattr(request, 'log_lines', ()) or ())
        # WAVE-J wall-clock instrumentation (coordinator item 3): time
        # since the feed last GREW while any deferred/starved/open
        # state exists = 'stalled_seconds' (published; the leg fails a
        # game at a hard stall budget instead of sitting for an hour)
        import time as _time
        self._last_lines = list(lines)
        _now = _time.time()
        if len(lines) != getattr(self, '_feed_n', -1):
            self._feed_n = len(lines)
            self._feed_t = _now
        _open_state = (self._open_step is not None
                       or self.driver.deferred is not None
                       or self.driver.chat_deferred is not None
                       or self.driver.pending_buf is not None)
        self._stalled = ((_now - getattr(self, '_feed_t', _now))
                         if _open_state else 0.0)
        # WAVE-J (ledger F-06, design rule 8 FAIL-CLOSED): an
        # un-blinded #HAND line in the feed means the sided launch is
        # missing -Dai.feed.aiside - the human's whole hand would be
        # readable.  Refuse to play on, loudly.
        for ln in lines:
            if isinstance(ln, str) and ln.startswith('#HAND') \
                    and '<blinded>' not in ln:
                raise DriverHalt(
                    'BLINDING VIOLATION: un-blinded #HAND line in '
                    'the feed - the sided launch is missing '
                    '-Dai.feed.aiside (design rule 8, fail-closed)')
            if isinstance(ln, str) and ln.startswith('#BLIND '):
                # the agent attests its blinding policy once at premain:
                # '#BLIND aiside=<FP|SA|none> hands=<...>'.  The human
                # hand is blinded only when the attested hands include
                # the human's side - anything else is fail-closed.
                _human = 'FP' if self.ai_side is gs.Side.SHADOW else 'SA'
                _f = dict(t.split('=', 1) for t in ln.split()[1:]
                          if '=' in t)
                if _human not in _f.get('hands', 'none'):
                    raise DriverHalt(
                        'BLINDING VIOLATION: feed attests %r but this '
                        'server plays against a human %s whose hand '
                        'would be readable (design rule 8, fail-closed)'
                        % (ln.strip(), _human))
        # sanctioned raw import #1: decode the AI's OWN drawn card
        # identities from the commitment tables and feed them BEFORE
        # the tick, so a retried ai_draws step consumes them this
        # cycle.  An undecodable AI card is a LOUD halt, never a guess.
        _prefix_lane = (self.scenario is not None
                        and not getattr(self.driver, 'live', True))
        for d in self.cards.harvest(lines):
            if d['cid'] is None:
                raise DriverHalt(
                    'AI draw pid %s undecodable from the client '
                    'commitments (client name %r) - cannot bank an '
                    'unknown card' % (d['pid'], d['name']))
            if _prefix_lane:
                self._scenario_pid_cid[str(d['pid'])] = d['cid']
                continue
            self.rand.feed('card', (d['side'], d['deck'], d['cid'],
                                    d['pid']))
            self.driver.view.card_pid_map[d['cid']] = d['pid']
            self.driver.view.note_drawn(d['pid'])
        if _prefix_lane:
            # K-S1: no AI step, no human matching, no mop-up until the
            # prefix is verified and the driver flipped to live
            return self._scenario_prefix_cycle(lines)
        out = []
        # WAVE-J (C-11): MANUAL EDIT hold - the machine is a spectator
        # until the operator finishes and presses resync
        if getattr(self, '_manual_hold', False):
            return []
        # WAVE-J (C-12): a panel resync epoch runs the RECOVERY (adopt
        # + clear + reopen), draining any manual-edit backlog unmatched
        if getattr(self, '_resync_requested', False):
            self._resync_requested = False
            for it in self.driver.resync_recover(drain_feed=True):
                out.extend(self._dispatch(it))
        # SKIP DECISION (USER 2026-08-20).  Runs on the TICK, before
        # the feed is read, so the commit and its emission ride the
        # normal path.  The outcome is ANNOUNCED - the player must be
        # able to see which decision was skipped and with what.
        # RESET + RESYNC (USER 2026-08-20).  The driver clears the
        # ENGINE-side pendings and adopts the board; the AI-STEP state
        # lives here and must go too, or the server would resume a
        # half-finished decision it just erased the basis for.
        if getattr(self, '_reset_requested', False):
            self._reset_requested = False
            try:
                _items, _rep = self.driver.reset_and_resync()
                for it in _items:
                    out.extend(self._dispatch(it))
            except DriverHalt:
                raise
            except Exception as _re9:
                _rep = ['reset failed: %r' % (_re9,)]
            out.extend(self._abandon_step('panel reset'))
            self._step_queue = []
            self._window_asks_sent = set()
            # a LIVE HALT is sticky until the operator resets (K3: the
            # reset is the one lever that clears it)
            self.rand.sticky_halt = None
            _rep.append('AI step state cleared')
            for _line in _rep:
                self.notes.append('reset: %s' % _line)
                print('  [reset] %s' % _line)
        if getattr(self, '_skip_requested', False):
            self._skip_requested = False
            # an AI step MID-FLIGHT (pick journaled / emission sent) is
            # abandoned EXPLICITLY before the skip commits anything on
            # top of it (K3); a step merely waiting its turn behind a
            # human pending (ai_eyes/ai_declare) keeps waiting
            if self.rand.step_open:
                out.extend(self._abandon_step('panel skip'))
            try:
                _items, _msg = self.driver.skip_open_decision()
                for it in _items:
                    out.extend(self._dispatch(it))
            except DriverHalt:
                raise
            except Exception as _se:
                _msg = 'skip failed: %r' % (_se,)
                _items = []
            self.notes.append('panel: %s' % _msg)
            print('  [note] panel: %s' % _msg)
        n_feed0 = len(self.driver.feed.lines)
        # a sidecar attach-generation change = the agent re-emitted a
        # NEW chat document (new game) onto this feed: never routed
        # into the running game (rule 12)
        _sc = getattr(self.driver, 'sidecar', None)
        if _sc is not None:
            _sc.poll()
            if _sc.gen_change is not None:
                raise DriverHalt(
                    'FEED RE-ATTACH: sidecar gen %d -> %d at feed line '
                    '%d - the client began a new chat document (new '
                    'game); restart the server against the new feed'
                    % (_sc.gen_change[1], _sc.gen_change[2],
                       _sc.gen_change[0]))
        _undo0 = getattr(self.driver, 'undo_epoch', 0)
        try:
            items = self.driver.tick(lines)
        except DriverHalt as e:
            self.notes.append(str(e))
            raise
        if getattr(self.driver, 'undo_epoch', 0) != _undo0 \
                and self._open_step is not None:
            # K3: a human undo popped the engine state an open AI step
            # was deciding on - that step is stale (its pick, its
            # staged pieces); abandon it and decide afresh
            out.extend(self._abandon_step('human undo popped the '
                                          'engine state'))
        # the client's random-discard lines of this tick feed the
        # 'card_pick' funnel (W1#64) before the open step retries
        self._feed_random_discards(n_feed0)
        # K1 step 4: a human commit this tick may have completed the
        # round's card commitments - the reveal directive precedes any
        # roll ask the tick's items carry
        out.extend(self._combat_reveal_sweep())
        for item in items:
            if item[0] in ('ai_draws', 'ai_decide', 'ai_eyes',
                           'ai_declare'):
                # AI steps are singular and retried across ticks (the
                # deterministic-retry protocol).  A DIFFERENT step
                # arriving while one is open (boot blocker 3: 'fp ok'
                # during ai_draws) is QUEUED, never a clobber - the
                # open step's directives were already ordered and its
                # answers are still owed.
                if self._open_step is None:
                    self._open_step = item
                    self._step_starved = 0
                elif self._open_step[0] == item[0]:
                    self._open_step = item          # refresh payload
                elif all(q[0] != item[0] for q in self._step_queue):
                    self._step_queue.append(item)
                continue
            out.extend(self._dispatch(item))
        # WAVE-H cbt5: an AI-step DriverHalt must not DISCARD the lines
        # the tick already built (serve()'s catch drops the whole RESP;
        # the deferred window's '!CALL combat_roll' ask was built,
        # DEDUPED as sent, then lost with the cycle - starved forever).
        # The halt stays loud (notes + print, re-raised only when there
        # is nothing to deliver); the step stays open and re-halts each
        # tick until fixed - visible, never silent.
        try:
            if self.driver.game_over is not None:
                # WAVE-J (A-09): a won game stops the AI cold - no
                # further steps, no queue drain (the donor's attempt-27
                # played on and let FP heal out of a won game)
                self._abandon_step('game over')
                self._step_queue = []
            if self._open_step is not None:
                out.extend(self._run_open_step())
            while self._open_step is None and self._step_queue:
                self._open_step = self._step_queue.pop(0)
                self._step_starved = 0
                out.extend(self._run_open_step())
        except DriverHalt as e:
            msg = 'AI step halted: %s' % e
            if not self.notes or self.notes[-1] != msg:
                self.notes.append(msg)
            print('  ! AI step halted (cycle output preserved): %s'
                  % str(e)[:160])
            if not out:
                raise
        # K1 step 4 again: the AI's own commit may have completed the
        # round (its '!CALL choose_card' is in `out` already)
        out.extend(self._combat_reveal_sweep())
        out.extend(self._flush_owed_die_kills())
        out.extend(self._tile_mopup())
        out.extend(self._table_mopup())
        # WAVE-J (ledger rank 9): the continuous witness - every 30
        # cycles (~1 min at the 2s heartbeat), skipped once the game
        # is over
        self._cycles = getattr(self, '_cycles', 0) + 1
        if self._cycles % 30 == 0 and self.driver.game_over is None:
            self._witness_cadence()
        # WAVE-J (A-09/P5-3): the game-over closer goes out ONCE, as
        # the very last lines; every later cycle returns empty replies
        # so the client's autoplay terminates (fact 16 / C-13).  A
        # victory detected at a COMMIT (mid-action ring - no driver
        # item on that path) builds its closer here.
        if self.driver.game_over is not None \
                and getattr(self, '_victory_tail', None) is None:
            self._victory_tail = self._game_over_lines(
                dict(self.driver.game_over))
        vt = getattr(self, '_victory_tail', None)
        if vt is not None and not getattr(self, '_victory_sent',
                                          False):
            self._victory_sent = True
            out.extend(vt)
        elif self.driver.game_over is not None \
                and getattr(self, '_victory_sent', False):
            return []
        self._track_own_emission(out)
        return out

    # ---- K-S1 scenario mode -------------------------------------------
    # findings the prefix lane may raise LIVE that the offline cutter
    # never sees, benign here: the AiFeed sidecar lags the feed during
    # the fast replay (the meta record of a paragraph is a separate
    # flush), so a line whose seq the meta has not yet reached reports
    # a gap - the pid-true view bookkeeping it feeds is redundant in
    # the spectator prefix lane (both seats external, dump-matched)
    _SCENARIO_BENIGN_FINDINGS = ('sidecar_gap',)

    def _scenario_prefix_cycle(self, lines):
        """One transport cycle while the prefix lane replays the
        fixture: let the sidecar catch up to the feed (its meta lags),
        verify + ingest, then try the flip.  A real finding or alarm is
        a DESYNC (the cutter proved the fixture replays with none)."""
        import time as _t
        drv = self.driver
        # gate: the meta must cover every COMPLETE feed line the tick
        # will consume, or the fast-replay lag forges sidecar gaps
        sc = getattr(drv, 'sidecar', None)
        if sc is not None:
            feed_txt = ''
            try:
                with open(os.path.join(self.client_dir, 'ai_feed.txt'),
                          'rb') as _f:
                    feed_txt = _f.read().decode('utf-8', 'replace')
            except OSError:
                feed_txt = ''
            n_complete = feed_txt.count('\n')
            t0 = _t.time()
            while _t.time() - t0 < 3.0:
                sc.poll()
                if sc.n_records >= n_complete or not sc.present:
                    break
                _t.sleep(0.1)
        n_f, n_a = len(drv.findings), len(drv.alarms)
        drv.tick(lines)
        if getattr(drv, 'desync', None):
            raise DriverHalt(drv.desync)
        new_f = [(k, d) for k, d in drv.findings[n_f:]
                 if k not in self._SCENARIO_BENIGN_FINDINGS]
        for k, d in drv.findings[n_f:]:
            if k in self._SCENARIO_BENIGN_FINDINGS:
                print('  [scenario] benign %s: %s' % (k, str(d)[:120]))
        if new_f or len(drv.alarms) > n_a:
            probs = (['finding %s: %s' % (k, str(d)[:160])
                      for k, d in new_f]
                     + ['alarm: %s' % str(a)[:160]
                        for a in drv.alarms[n_a:]])
            raise DriverHalt('SCENARIO DESYNC (prefix lane, feed line %d '
                             'of %d): %s'
                             % (drv.prefix_seen, drv.prefix_n,
                                ' | '.join(probs[:4])))
        self._scenario_try_flip()
        return []

    def _scenario_try_flip(self):
        """Flip the driver to live once the client agrees with the
        prefix lane: the AI-side claim tables are consistent across the
        raw stream, the board dump's AI-hand pids all decode and match
        the engine's count, and the position equals the fixture's own
        reference replay.  Waits (bounded, visibly) for the dump / raw
        stream to catch up; anything else halts loudly."""
        import time as _t
        from . import client_board
        from . import scenario as scen_mod
        drv = self.driver
        if drv.live or not drv.flip_due:
            return
        now = _t.time()
        if self._flip_due_since is None:
            self._flip_due_since = now
        budget = float(self.scenario.timeouts.get('flip', 90) or 90)
        ai_name = self.ai_side.name

        def _wait(note):
            if note != self._flip_note:
                self._flip_note = note
                self.notes.append('scenario flip: %s' % note)
                print('  [scenario] flip waiting: %s' % note)
            if now - self._flip_due_since > budget:
                raise DriverHalt('SCENARIO FLIP FAILED after %ds: %s'
                                 % (int(budget), note))
        ok, n_claims = scen_mod.claim_tables_consistent(
            getattr(self, '_last_lines', ()) or (), ai_name)
        if not ok:
            raise DriverHalt(
                'SCENARIO CLAIM RACE: %d different %s commitment tables '
                'in the client log - the client claimed fresh tables '
                'before the fixture claim line replayed; the AI hand '
                'identities would disagree.  Rerun the scenario.'
                % (n_claims, ai_name))
        if n_claims == 0:
            return _wait('no %s claim line in the raw stream yet'
                         % ai_name)
        dump = client_board.read_board(
            os.path.join(self.client_dir, 'ai_client_board.txt'))
        if dump is None:
            return _wait('no stable client board dump yet')
        cids, card_pids, missing = scen_mod.ai_hand_from_dump(
            dump, ai_name, self._scenario_pid_cid)
        if missing:
            return _wait('AI-hand pid(s) %s not yet decoded from the '
                         'raw stream' % ','.join(missing))
        n_engine = len(drv.es.hands[self.ai_side].hand)
        if len(cids) != n_engine:
            return _wait('AI hand: engine %d card(s), client dump %d'
                         % (n_engine, len(cids)))
        cur = scen_mod.position_summary(drv)
        if self._scenario_ref is not None:
            diffs = scen_mod.summary_diff(self._scenario_ref, cur)
            if diffs:
                raise DriverHalt('SCENARIO DESYNC (position at the flip '
                                 'differs from the fixture reference): %s'
                                 % ' | '.join(diffs[:6]))
        drv.flip(cids, card_pids)
        self._flip_note = None
        for ln in drv.flip_report:
            self.notes.append('scenario live: %s' % ln)
            print('  [scenario] LIVE: %s' % ln)

    def _scenario_status(self):
        drv = self.driver
        if self.scenario is None:
            return None
        return {'name': self.scenario.name,
                'prefix_expected': int(getattr(drv, 'prefix_n', 0)),
                'prefix_seen': int(getattr(drv, 'prefix_seen', 0)),
                'live': bool(getattr(drv, 'live', True)),
                'flip_due': bool(getattr(drv, 'flip_due', False)),
                'note': self._flip_note,
                'skipped': len(getattr(drv, 'prefix_skipped', ()) or ())}

    def _run_open_step(self):
        item = self._open_step
        kind = item[0]
        # LOUD STALL COUNTER (instrument catalogue: a stall with no
        # counter is invisible; boot2 wedged EXACTLY here - the draw
        # ask was sent once, its answer was never harvested, and the
        # dedup made every later tick an empty reply with no witness).
        # Directives are deliberately NEVER auto-re-sent (a re-sent
        # draw the client DID execute would draw a second card); the
        # stall is surfaced instead.
        self._step_starved = getattr(self, '_step_starved', 0) + 1
        if self._step_starved % 25 == 0:
            msg = ('OPEN STEP STARVED: %r retried %d ticks without '
                   'the awaited client outcome (asks sent once, '
                   'never re-sent; pending=%r)'
                   % (kind, self._step_starved,
                      getattr(self.rand.pending(), 'kind', None)))
            self.notes.append(msg)
            print('  [stall] %s' % msg)
        if kind == 'ai_draws':
            return self._ai_draws()
        if kind == 'ai_eyes':
            return self._ai_eyes()
        if kind == 'ai_declare':
            return self._ai_declare()
        if kind == 'ai_decide':
            import ai
            # recompute the pending on the CURRENT state (the stored
            # dp's menu may be stale after new feed lines)
            dp = self.driver.ai_action_dp()
            if dp is None:
                return self._abandon_step('due decision vanished')
            return self._ai_decide(dp)
        return self._abandon_step('unknown step kind %r' % (kind,))

    _RANDDISC_LABEL_RE = None

    def _feed_random_discards(self, n0):
        """The feed producer for the 'card_pick' funnel: every client
        line '<nick> randomly discards a <deck label>.' consumed this
        tick (feed lines n0..) feeds ONE card_pick outcome.  The label
        names the deck (zone-derived, never the nick): 'Free ...
        Character Card' = the FP hand's Character cards.  Identity:
        the AI's OWN hand decodes the sidecar pid through the
        commitment map (never redacted for the AI side) - undecodable
        is a LOUD halt, never a guess; the HUMAN hand is blinded, so
        every Character card the engine models there (placeholders
        and revealed cids alike) is an acceptable id.  A deferred
        window/chat decision starving on the pick is retried at once."""
        import re as _re
        import game_state as gs
        from . import live_parser
        from .sidecar import meta_pids
        drv = self.driver
        feed_lines = drv.feed.lines
        if len(feed_lines) <= n0:
            return
        pat = LiveServer._RANDDISC_LABEL_RE
        if pat is None:
            pat = LiveServer._RANDDISC_LABEL_RE = _re.compile(
                r'randomly\s+discards\s+a\s+(.+?)\.?\s*$', _re.I)
        fed = 0
        for i in range(n0, len(feed_lines)):
            raw = str(feed_lines[i])
            try:
                rec = live_parser.parse(raw)
            except Exception:
                continue
            if rec.get('op') != 'random_discard':
                continue
            m = pat.search(raw)
            label = (m.group(1) if m else '').strip().lower()
            side = (gs.Side.FREE_PEOPLES if label.startswith('free')
                    else gs.Side.SHADOW if label.startswith('shadow')
                    else None)
            deck = ('character' if 'character' in label
                    else 'strategy' if 'strategy' in label else None)
            if side is None or deck is None:
                self.driver.alarms.append(
                    'RANDOM DISCARD with an unreadable deck label: %r'
                    % raw[:80])
                continue
            hand = drv.es.hands[side].hand
            if side is self.ai_side:
                sc = getattr(drv, 'sidecar', None)
                meta = sc.rec(i + 1) if (sc is not None
                                         and sc.present) else None
                pids = meta_pids(meta) or []
                inv = {str(v): k
                       for k, v in drv.view.card_pid_map.items()}
                cids = [inv[str(p)] for p in pids if str(p) in inv]
                cids = [c for c in cids if c in hand]
                if not cids:
                    raise DriverHalt(
                        'RANDOM DISCARD from the AI hand with no '
                        'decodable card identity (pids %r, line %r) - '
                        'cannot bank an unknown discard'
                        % (pids, raw[:80]))
            else:
                tag = 'HU_%s_%s' % (side.name[:2], deck[:4].upper())
                word = '_Character_' if deck == 'character' \
                    else '_Strategy_'
                cids = [c for c in hand if isinstance(c, str)
                        and (c.startswith(tag) or word in c)]
                if not cids:
                    self.driver.alarms.append(
                        'RANDOM DISCARD: the engine models no %s %s '
                        'card in the %s hand - hand desync (%r)'
                        % (side.name, deck, side.name, raw[:80]))
                    continue
            self.rand.feed('card_pick', cids)
            fed += 1
            self.notes.append('random discard fed: %s %s (%d candidate '
                              'id(s))' % (side.name, deck, len(cids)))
        if fed and (drv.deferred is not None
                    or drv.chat_deferred is not None
                    or drv.pending_buf is not None):
            retry = getattr(drv, '_retry_blocked', None)
            if retry is None:
                self.notes.append('random discard fed but the driver '
                                  'has no _retry_blocked - the deferred '
                                  'decision retries on its next record')
            else:
                retry()

    def _note_cc_chosen(self, em):
        """The AI's emission carried its hidden combat-card commit
        (the 'chooses a card' line, emitter op combat_card_pick): the
        card is chosen, not revealed, until the round's reveal
        directive (K1)."""
        if any('<~Controls.840~>' in str(ln) for ln in (em.lines or ())):
            self._ai_cc_chosen = True

    def _combat_reveal_sweep(self):
        """K1 step 4 (R4: cards are REVEALED TOGETHER).  Once BOTH
        sides of the open battle round have committed (a card - hidden
        on the client, or 'nc') and a chosen card is still unrevealed -
        the engine's answer (ai.combat_cards_unrevealed: the human's
        hidden commit) or the AI's own '!CALL choose_card' this round -
        ONE '!CALL reveal_chosen' goes out (it reveals every chosen
        card).  Invariants: exactly once per round (keyed on the
        battle + rounds_fought); the engine surface is LOUD when a
        hidden commit exists that it cannot answer for; the AI's
        chosen-state clears with the directive (the reveal lines then
        speak for themselves)."""
        import ai
        es = self.driver.es
        pc = getattr(es, 'pending_combat', None)
        if not isinstance(pc, dict):
            self._reveal_round = None
            self._ai_cc_chosen = False
            return []
        hidden = [k for k in ('atk_hidden_cc', 'def_hidden_cc')
                  if pc.get(k)]
        atk_done = 'atk_combat_card' in pc or 'atk_hidden_cc' in pc
        def_done = 'def_combat_card' in pc or 'def_hidden_cc' in pc
        # the human's commit may be DEFERRED on the round's roll (its
        # apply ran on a clone and starved): the decision is made, the
        # engine state just does not carry it yet
        # A side counts as committed once the engine carries its card
        # (atk_/def_combat_card or the hidden sentinel) OR its commit
        # is DEFERRED on the round's rolls: the human's chat commit ran
        # on a clone that starved (chat_deferred), and SYMMETRICALLY
        # the AI's OWN combat_card_play (choose_card emitted,
        # _ai_cc_chosen set) commits atk_/def_combat_card only when its
        # apply SUCCEEDS - that apply starves on the round's rolls
        # (NeedClientCall rolls back the post), so the engine state
        # does not yet carry the AI's card.  The deferred flag is
        # scoped to the side that owns it, so the AI-as-ATTACKER never
        # reveals before the human defender commits (the premature
        # reveal that _ai_cc_chosen alone would cause), and the
        # AI-as-DEFENDER reveals its chosen card once the human
        # attacker is done (else the client hand stays one card over
        # the engine's - witness escalation - and the card never
        # reaches the table).
        cd = getattr(self.driver, 'chat_deferred', None)
        cd_at = str(getattr(cd[0], 'action_type', '')) if cd else ''
        human_deferred = cd_at in ('combat_card_none',
                                   'combat_card_play',
                                   'combat_card_hidden_commit')
        ai_chosen = bool(getattr(self, '_ai_cc_chosen', False))
        ai_is_attacker = pc.get('attacker_side') is self.ai_side
        atk_committed = atk_done or (ai_chosen if ai_is_attacker
                                     else human_deferred)
        def_committed = def_done or (human_deferred if ai_is_attacker
                                     else ai_chosen)
        both = atk_committed and def_committed
        if not both:
            return []
        fn = getattr(ai, 'combat_cards_unrevealed', None)
        if fn is None:
            if hidden:
                raise DriverHalt('ENGINE SURFACE MISSING: ai.combat_'
                                 'cards_unrevealed (K1 step 1) while a '
                                 'hidden combat card is committed (%s)'
                                 % ', '.join(hidden))
            unrev = []
        else:
            unrev = list(fn(es) or ())
        if not unrev and not getattr(self, '_ai_cc_chosen', False):
            return []
        # once per round - and once more per REFUSED reveal (the
        # driver's cc_reveal_epoch: a withdrawn hidden commit re-opens
        # the window; its re-commit owes a fresh reveal)
        key = (str(pc.get('attacker_region')),
               str(pc.get('defender_region')),
               int(pc.get('rounds_fought', 0) or 0),
               int(getattr(self.driver, 'cc_reveal_epoch', 0) or 0))
        if getattr(self, '_reveal_round', None) == key:
            return []
        rfn = getattr(emission, 'render_combat_reveal', None)
        if rfn is None:
            raise DriverHalt('RENDERER MISSING: emission.render_combat_'
                             'reveal (K1 step 2) - chosen combat cards '
                             'cannot be revealed')
        em = rfn(self.driver.view, self.nick, self.ai_side)
        if em.hold:
            raise DriverHalt('EMISSION HOLD (combat reveal): %s'
                             % em.hold)
        self._reveal_round = key
        self._ai_cc_chosen = False
        self.notes.append('combat cards revealed together (round %d; '
                          'engine unrevealed: %s)'
                          % (key[2], ','.join(getattr(s, 'name', str(s))
                                              for s in unrev) or '-'))
        return self._emission_out(em)

    def _abandon_step(self, reason):
        """K3: drop the open AI step WITHOUT a commit.  Invariant: no
        step-scoped residue survives an abandon - the randomness step
        (journal, pending ask, cursors), the once-per-step emission
        flag, the directive dedup sets and the expressed-ops ledger
        all reset together, so the next decision renders and asks
        exactly as a first attempt would.  Idempotent when nothing is
        open; noted (panel) when something was."""
        had = (self._open_step[0] if self._open_step else None,
               bool(self.rand.step_open), bool(self._em_sent))
        out = []
        if had[2]:
            out.extend(self._unstage_abandoned())
        self.rand.abort_step(reason)
        self._open_step = None
        self._step_starved = 0
        self._em_sent = False
        self._dirs_sent.clear()
        self._step_calls.clear()
        self._step_expressed = []
        self._step_owed = []
        if had[0] is not None or had[1] or had[2]:
            msg = ('AI step abandoned (%s): step=%r rand_open=%r '
                   'emission_sent=%r - state cleared, next decision '
                   'starts fresh' % (reason, had[0], had[1], had[2]))
            self.notes.append(msg)
            print('  [note] %s' % msg)
        return out

    def _unstage_abandoned(self):
        """The abandoned step's PRE-APPLY emission staged pieces on the
        client that the engine never spent (the step never committed):
        the die it moved to Current Action, a card it moved to the
        Current-Card slot, any board op it rendered.  Invariants: the
        owed kills THIS step banked are dropped (killing them would
        remove pieces the engine still holds); nothing is dragged back
        by an invented line (the corpus has no 'Current Action -> pool'
        drag - humans UNDO; the client-native undo lane is the driver's)
        - the residue is an ALARM naming the pieces, for the operator's
        undo/RESYNC."""
        drv = self.driver
        names = []
        for ent in list(getattr(self, '_step_owed', ()) or ()):
            kind, key = ent[0], ent[1]
            if kind == 'die' and key in drv.owed_die_kills:
                drv.owed_die_kills.remove(key)
                names.append('%s die %s staged in Current Action'
                             % (key[0], key[1]))
                if len(key) > 2 and key[2] is not None:
                    names.append('card pid %s in the Current-Card slot'
                                 % key[2])
            elif kind == 'card' and key in drv.owed_card_kills:
                drv.owed_card_kills.remove(key)
                names.append('combat card pid %s' % key[1])
        n_ops = len(self._step_expressed or ())
        if names or n_ops:
            msg = ('ABANDONED STEP RESIDUE: the client still shows %s '
                   '(%d rendered op(s)) for a decision the engine never '
                   'committed - undo them on the client or RESYNC'
                   % ('; '.join(names) if names else 'rendered pieces',
                      n_ops))
            self.notes.append(msg)
            drv.alarms.append(msg)
        return []

    def _dispatch(self, item):
        kind = item[0]
        if kind == 'line' or kind == 'call':
            return [item[1]]
        if kind == 'chat':
            return [self._say(item[1], item[2])]
        if kind == 'note':
            self.notes.append(item[1])
            print('  [note] %s' % item[1])
            return []
        if kind == 'recover_ai':
            return self._recover()
        if kind == 'ai_roll':
            return self._order_own_roll()
        if kind == 'rng_ask':
            return self._window_ask_lines(item[1])
        if kind == 'bonus_dice':
            return self._bonus_dice_lines(item[1])
        if kind == 'mordor_entry':
            return self._mordor_entry_lines(item[1])
        if kind == 'game_over':
            # P5-3: the closer must be the LAST lines of the response
            # (after every cleanup/flush) - bank it; handle() appends
            if getattr(self, '_victory_tail', None) is None:
                self._victory_tail = self._game_over_lines(item[1])
            return []
        self.notes.append('unrouted driver item %r' % (kind,))
        return []

    def _bonus_dice_lines(self, pre_counts):
        """WAVE-J J1 (A-11): render the AI's OWN new bonus-die pieces
        out of its reinforcement box (never the human's - they drag
        their own; the witnesses report a forgotten drag).  Also
        pre-binds the base-table pids so die_piece_for answers before
        the next roll adoption."""
        import emitter
        import game_state as gs
        drv = self.driver
        s = self.ai_side
        base = (emitter._FP_DIE_IDS if s is gs.Side.FREE_PEOPLES
                else emitter._SH_DIE_IDS)
        n0 = int(pre_counts.get(s.name, 0) or 0)
        n1 = len(drv.es.dice.get(s, []))
        # also bind BOTH sides' new indices (the client-fact-11 static
        # family; the human's new die needs a status pid binding even
        # though the human drags the piece)
        for s2 in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            b2 = (emitter._FP_DIE_IDS if s2 is gs.Side.FREE_PEOPLES
                  else emitter._SH_DIE_IDS)
            for i in range(int(pre_counts.get(s2.name, 0) or 0),
                           len(drv.es.dice.get(s2, []))):
                if i < len(b2):
                    drv.view.die_id_map.setdefault(str(b2[i]),
                                                   (s2, i))
        try:
            import types
            pre = types.SimpleNamespace(dice={
                s2: [None] * (int(pre_counts.get(s2.name, 0) or 0)
                              if s2 is s
                              else len(drv.es.dice.get(s2, [])))
                for s2 in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)})
            lines = emission._pool_growth_lines(drv.view, self.nick,
                                                pre, drv.es)
        except LookupError as e:
            self.notes.append('J1 growth line unrenderable: %s' % e)
            return []
        self.notes.append('J1 bonus dice granted: %s pool %d -> %d'
                          % (s.name, n0, n1))
        return lines

    def _game_over_lines(self, vc):
        """WAVE-J (A-09 delivery): the end-of-game closer - the
        corruption marker walked to its final value first (attempt-71:
        the victory must be SHOWN by the marker that won it), then the
        LOSER's 'gg' (P5-3: the last line of the log; the AI can only
        speak for itself, so it says gg only when IT lost).  The full
        result rides the panel notes + status; the old 'game over:'
        status chat was whitelist-stripped by USER policy and is not
        re-introduced."""
        from emitter import render
        from board import sanitize
        out = []
        self.notes.append('GAME OVER: %s wins (%s) fp=%s sh=%s'
                          % (vc.get('victory'),
                             vc.get('victory_type'),
                             vc.get('free_peoples_vp', '?'),
                             vc.get('shadow_vp', '?')))
        if vc.get('victory_type') == 'ring_corruption':
            # walk marker 252 to its final slot (client-true source)
            try:
                from . import client_board
                _dp = getattr(self.driver.view, 'dump_path', None)
                dump = client_board.read_board(_dp) if _dp else None
                z = sanitize(str((dump or {}).get('252', '')))
                cor = int(getattr(self.driver.es.fellowship,
                                  'corruption', 12) or 12)
                dest = 'Game.%d' % (2014 + min(12, cor))
                if z.startswith('track '):
                    frm = 'Game.%d' % (2014 + int(z.split()[1]))
                    if frm != dest:
                        out.append(render('move', {
                            'nick': self.nick, 'pieces': ['252'],
                            'frm': frm, 'to': dest}, local=False))
            except Exception as e:
                self.notes.append('final corruption walk skipped: %r'
                                  % (e,))
        ai_name = ('shadow' if self.ai_side.name == 'SHADOW'
                   else 'free_peoples')
        if vc.get('victory') and vc.get('victory') != ai_name:
            out.append(self._say('concede', 'gg'))
        return out

    def _waiting_on(self):
        """WAVE-J (USER: the old panel said EXACTLY what the AI was
        waiting on; the lift carried the status SCHEMA but never the
        population).  The PRECISE blocker in USER-readable words,
        composed from the driver's actual state, priority-ordered;
        stalled_seconds rides along.  An empty answer while ANY
        starved/deferred/open state exists is a pinned test failure
        (the silent-stall disease on the display surface)."""
        drv = self.driver
        stall = getattr(self, '_stalled', 0.0)
        sfx = (' [stalled %ds]' % int(stall)) if stall >= 10 else ''
        if self.scenario is not None and not getattr(drv, 'live', True):
            return ('SCENARIO %s: replaying the prefix (%d/%d)%s'
                    % (self.scenario.name,
                       int(getattr(drv, 'prefix_seen', 0)),
                       int(getattr(drv, 'prefix_n', 0)),
                       (' - flip: %s' % self._flip_note)
                       if self._flip_note else ''))
        if drv.game_over is not None:
            vc = drv.game_over
            return ('GAME OVER: %s wins (%s)'
                    % (vc.get('victory'), vc.get('victory_type')))
        try:
            p = self.rand.pending() if self.rand is not None else None
        except Exception:
            p = None
        if p is not None:
            kw = getattr(p, 'kw', {}) or {}
            k = getattr(p, 'kind', '?')
            if k == 'd6':
                return ('waiting on: %s to roll %s dice (%s)%s'
                        % (kw.get('side', '?'), kw.get('need', '?'),
                           kw.get('reason', 'combat/hunt'), sfx))
            if k == 'tile':
                return 'waiting on: client hunt-tile draw%s' % sfx
            if k in ('card', 'draws'):
                return ('waiting on: client card draw (%s)%s'
                        % (kw.get('items', kw), sfx))
            return 'waiting on: client %s (%s)%s' % (k, kw, sfx)
        if drv.deferred is not None:
            _w = (drv.deferred[0]
                  if isinstance(drv.deferred, tuple)
                  and drv.deferred else None)
            _sc = getattr(_w, '_starve_call', None)
            if _sc is not None:
                return ('waiting on: client %s (%s) for the deferred '
                        'gesture window%s'
                        % (getattr(_sc, 'kind', '?'),
                           getattr(_sc, 'kw', {}), sfx))
            return ('waiting on: client randomness for the deferred '
                    'gesture window%s' % sfx)
        if drv.chat_deferred is not None:
            return ('waiting on: client randomness for the chat '
                    'decision%s' % sfx)
        if self._open_step is not None:
            return ('AI deciding: %s (retry %d)%s'
                    % (self._open_step[0],
                       getattr(self, '_step_starved', 0), sfx))
        if drv.pending_buf is not None \
                and getattr(drv.pending_buf, 'records', None):
            return ('waiting on: gesture match for %d buffered '
                    'record(s)%s'
                    % (len(drv.pending_buf.records), sfx))
        try:
            import ai
            with self.rand.pin():
                dp = ai.describe_pending(drv.es, self.human_side)
        except Exception:
            dp = None
        if dp is not None:
            return ("waiting on: %s's %s decision (board gesture "
                    "or chat)%s" % (dp['side'].name, dp['kind'], sfx))
        if drv.phase == 'actions':
            try:
                import ai
                nxt = (ai._next_acting_side(drv.es, drv.last_actor)
                       if drv.last_actor is not None else None)
            except Exception:
                nxt = None
            who = (nxt.name if nxt is not None
                   else self.human_side.name)
            return 'waiting on: %s to act%s' % (who, sfx)
        import game_state as _gs
        if self.ai_side is _gs.Side.FREE_PEOPLES:
            # WAVE-K R2: the seats swap - the AI holds the Fellowship
            # phase, the human Shadow the eyes + the first roll
            names = {'draws': 'human (Shadow) draws',
                     'declare': 'AI declaring (its Fellowship phase)',
                     'eyes': "human Shadow eye allocation ('Ne' or "
                             "the SA Dice -> Hunt box drag) + SA roll",
                     'roll': 'action-dice rolls (SA first, then the '
                             "AI's FP pool)"}
        else:
            names = {'draws': 'human draws',
                     'declare': "declaration window ('fp ok')",
                     'eyes': 'AI eye allocation',
                     'roll': 'action-dice rolls'}
        if str(drv.phase) == 'roll':
            # NAME THE POOL WE ARE ACTUALLY WAITING ON.  'action-dice
            # rolls' left the player with no idea it was THEIR roll
            # the game wanted (USER 2026-08-20) - and while their pool
            # sits unrolled the action phase cannot open at all.
            owed = [s for s in (self.human_side, self.ai_side)
                    if any(not d.used and d.current is None
                           for d in drv.es.dice.get(s, []))]
            if owed == [self.human_side]:
                n = len([d for d in drv.es.dice.get(self.human_side, [])
                         if not d.used and d.current is None])
                return ('waiting on: YOU to roll your %d %s action '
                        'dice%s' % (n, self.human_side.name.replace(
                            '_', ' ').title(), sfx))
            if owed == [self.ai_side]:
                return ("waiting on: the client to roll the AI's "
                        "pool%s" % sfx)
        return ('waiting on: %s%s'
                % (names.get(str(drv.phase), str(drv.phase)), sfx))

    def serve_handler(self, request):
        """The production transport handler: RESP lines + the
        waiting_on text the panel displays (transport writes it into
        the status every cycle)."""
        try:
            return self.handle(request), self._waiting_on()
        finally:
            self._write_mirror_snapshot()

    def _write_mirror_snapshot(self):
        """ENGINE MIRROR feed: the engine state + a status header for
        wotr_bridge\\live_view.py, written atomically.

        A DROPPED PORT (USER 2026-08-20: "the engine mirror is not
        connected").  The panel's ENGINE MIRROR button launches
        live_view.py, which polls <client_dir>/ai_live_state.pkl +
        ai_live_status.json; those files were written ONLY by the
        retired monolith (ai_move_server._write_sided_snapshot), so
        against the new stack the viewer opened on its placeholder
        state and waited forever.  Same filenames, same keys, so the
        viewer is unchanged.

        ENGINE TRUTH, labeled as such; 'ts' makes staleness visible.
        The pickle necessarily carries full hidden information - the
        VIEWER owns masking (mode + ai_side below), exactly as under
        the old server.  These files stay INSIDE the client dir and
        are cleared by the launcher between sessions.

        Throttled: pickling a GameState on every 0.2s poll is pure
        waste when nothing committed.  Writes when the commit count
        changes, or once a second so 'ts' stays fresh."""
        drv = getattr(self, 'driver', None)
        if drv is None or getattr(drv, 'es', None) is None:
            return
        import time as _t
        now = _t.time()
        n_commits = int(getattr(drv, 'n_commits_total', 0) or 0)
        last = getattr(self, '_mirror_last', (None, 0.0))
        if last[0] == n_commits and (now - last[1]) < 1.0:
            return
        self._mirror_last = (n_commits, now)
        import json as _json
        import pickle as _pickle
        cd = self.client_dir
        try:
            tmp = os.path.join(cd, 'ai_live_state.pkl.tmp')
            with open(tmp, 'wb') as f:
                _pickle.dump(drv.es, f,
                             protocol=_pickle.HIGHEST_PROTOCOL)
            # the viewer (live_view.py) holds the pickle open for
            # milliseconds while it loads; on Windows the swap then
            # fails with PermissionError - retry briefly instead of
            # skipping the snapshot
            _dst = os.path.join(cd, 'ai_live_state.pkl')
            for _try in range(5):
                try:
                    os.replace(tmp, _dst)
                    break
                except PermissionError:
                    if _try == 4:
                        raise
                    _t.sleep(0.02)
            findings = [('%s: %s' % (str(k), str(d)[:120]))
                        for k, d in list(
                            getattr(drv, 'findings', ()) or ())[-20:]]
            alarms = [str(a)[:200]
                      for a in list(getattr(drv, 'alarms', ())
                                    or ())[-20:]]
            mism = findings + alarms
            status = {
                'mode': 'sided',
                'ai_side': self.ai_side.name,
                'turn': int(getattr(drv.es, 'turn', 0)),
                'phase': str(getattr(drv, 'phase', 'sided')),
                'active': None,
                'last': str(self._waiting_on())[:200],
                'client_divergences': len(mism),
                'mismatches': mism[:20],
                'engine_vs_parser': 'n/a (sided: parser IS the view)',
                'engine_vs_client': (('%d issue(s): ' % len(mism))
                                     + '; '.join(mism[:3])
                                     if mism else 'OK'),
                'over': bool(getattr(drv, 'game_over', None)),
                'scenario': self._scenario_status(),
                'ts': now}
            tmp2 = os.path.join(cd, 'ai_live_status.json.tmp')
            with open(tmp2, 'w', encoding='utf-8') as f:
                _json.dump(status, f)
            os.replace(tmp2, os.path.join(cd, 'ai_live_status.json'))
        except Exception as e:
            # never let the mirror take the session down; skips are
            # COUNTED (a starving viewer is visible), one note per
            # distinct error and then every 20th skip
            self._mirror_skips = getattr(self, '_mirror_skips', 0) + 1
            msg = ('mirror snapshot skipped (%d so far): %r'
                   % (self._mirror_skips, e))
            if getattr(self, '_mirror_warned', None) != str(e)                     or self._mirror_skips % 20 == 0:
                self._mirror_warned = str(e)
                self.notes.append(msg)
                print('  [note] %s' % msg)

    def _track_own_emission(self, lines):
        """WAVE-J (wjg13 LIVE: the A28 fungible-pick class on OUR OWN
        lines) - the AI killed Isengard regular pid 002 at Orthanc and
        THREE SECONDS LATER advanced 'pid 002' with the army: the
        second render's fungible pick read a view the first render's
        kill had not yet reached (the echo lands a tick later; the
        AI's next decision fired first), so the client kept one real
        regular at Orthanc the engine had moved - the resync witness
        caught it.  ONE ANSWERER for our own effects: every raw op
        line WE emit updates the view at the emission exit (kills ->
        'Dead' - never popped: an untracked pid falls back to its
        SETUP zone in emitter._pieces_in and would be picked again;
        moves -> the decoded destination).  The sidecar echo later
        re-applies the same facts idempotently."""
        import emitter
        import log_replay as lr
        view = self.driver.view
        KILL_OPS = ('hand_exit_b', 'kill', 'swap_place')
        for ln in lines or ():
            if not isinstance(ln, str) or not ln.startswith('$'):
                continue
            for op, spec in emitter.OPS.items():
                m = spec['re'].match(ln)
                if not m:
                    continue
                d = m.groupdict()
                pids = []
                if d.get('pieces'):
                    pids = str(d['pieces']).split()
                elif d.get('piece'):
                    pids = [str(d['piece'])]
                elif d.get('die'):
                    pids = [str(d['die'])]
                if op in KILL_OPS:
                    for pid in pids:
                        view.piece_region[pid] = 'Dead'
                elif d.get('to'):
                    try:
                        name = lr.decode_location(str(d['to']))
                    except Exception:
                        name = None
                    if name:
                        for pid in pids:
                            view.piece_region[pid] = str(name)
                break

    def _tile_pool_preheal(self):
        """WAVE-J (ledger A-05, donor :12439-12456): a DRY client
        Hunt Pool before a tile directive is pre-healed - Used-zone
        tile pieces matching the ENGINE's own pool faces ride the
        corpus recycle drag Used(2045) -> Pool(2044) BEFORE the
        '!CALL hunt_tile' goes out (the attempt-22 hang: the client's
        draw button has nothing to draw and the directive no-ops
        forever)."""
        import log_replay as lr
        from board import sanitize
        from emitter import render
        from . import client_board
        drv = self.driver
        eng_pool = list(getattr(drv.es, 'hunt_pool', []) or [])
        if not eng_pool:
            return []
        _dp = getattr(drv.view, 'dump_path', None)
        dump = client_board.read_board(_dp) if _dp else None
        if dump is None:
            return []
        zones = {pid: sanitize(str(dump.get(pid, '')))
                 for pid in lr.HUNT_TILE_PIECES}
        if any(z == 'hunt pool' for z in zones.values()):
            return []                    # client pool not dry
        faces = {}
        for t in eng_pool:
            f = _tile_face(t)
            faces[f] = faces.get(f, 0) + 1
        back = []
        for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
            if zones.get(pid) != 'used hunt tiles':
                continue
            f = _tile_face(nm)
            if faces.get(f, 0) > 0:
                faces[f] -= 1
                back.append(pid)
        if not back:
            return []
        self.notes.append('tile pool pre-heal: %d piece(s) recycled '
                          'Used -> Pool' % len(back))
        return [render('move', {'nick': self.nick,
                                'pieces': sorted(back),
                                'frm': 'Game.2045',
                                'to': 'Game.2044'}, local=False)]

    def _window_ask_lines(self, call):
        """Order the randomness a DEFERRED human window / pending
        buffer is starving on (PHASE3-D port: the spectator reads a
        window's rng from the log; live must SOLICIT it — an AI-side
        defender roll would otherwise starve forever).  Epoch-keyed
        dedup (see __init__)."""
        # WAVE-J wjg4 LIVE REPRODUCTION: the epoch must advance on
        # every window SETTLEMENT, not commits only - a REFUSED window
        # (the illegal-probe class) consumes its ask without a commit,
        # and the next window's identical ask (the real fellowship
        # move's hunt roll) was falsely deduped: the deferral starved
        # forever while the server printed 'rng-starved' with no
        # '!CALL' ever going out.  Refusal count rides the key.
        ws = self.driver.window_stats
        epoch = (self.driver.n_commits_total,
                 int(ws.get('none', 0) or 0),
                 int(ws.get('hold', 0) or 0))
        ident = (epoch, call.kind, tuple(sorted(
            (k, str(v)) for k, v in call.kw.items())))
        # the RENDERED feed the driver tails is the stream a client
        # outcome renders in ('<game> X rolls 4', 'X draws Eye.'); the
        # raw request log carries the Controls grammar and never
        # matches the rendered check - the mark and the scan must
        # both read the feed
        lines_now = list(self.driver.feed.lines)
        meta = getattr(self, '_ask_meta', None)
        if meta is None:
            meta = self._ask_meta = {}
        if ident in self._window_asks_sent:
            # WAVE-J WATCHDOG (coordinator contract, the wjg4 silent-
            # wedge class): a deferred ask aged >= 25 ticks is judged
            # against the FEED - re-asking is SAFE exactly when no
            # result RENDERED since the ask (the client provably never
            # executed it); a rendered-but-unharvested result is a
            # LOUD STICKY HALT, never a re-ask (a re-send would
            # double-roll).
            m = meta.get(ident)
            if m is None:
                return []
            m['ticks'] += 1
            if m['ticks'] < 25:
                return []
            tail = lines_now[m['feed_mark']:]
            hit = self._rng_result_rendered(call.kind, tail)
            if hit is not None:
                msg = ('DEFERRAL WEDGE: a %s result RENDERED since '
                       'the ask but was never harvested (%.80s) - '
                       'refusing the re-ask (double-roll hazard); '
                       'resync/undo owed' % (call.kind, hit))
                if msg not in self.driver.alarms:
                    self.driver.alarms.append(msg)
                return []
            self.notes.append(
                'WATCHDOG re-ask: %s ask aged %d ticks, NO rendered '
                'result in the feed since it - the client provably '
                'never executed it' % (call.kind, m['ticks']))
            self._window_asks_sent.discard(ident)
        self._window_asks_sent.add(ident)
        meta[ident] = {'feed_mark': len(lines_now), 'ticks': 0}
        out = []
        for d in self.rand.directives_for(call):
            if d.kind == 'chat':
                out.append(self._say(
                    'random_ask' if d.text.endswith('?')
                    else 'roll_request', d.text))
            else:
                if d.text == '!CALL hunt_tile':
                    out.extend(self._tile_pool_preheal())
                out.append(d.text)
        return out

    @staticmethod
    def _rng_result_rendered(kind, tail):
        """The feed line proving the awaited randomness EXECUTED, or
        None - classified by the ONE live grammar (live_parser.parse,
        the driver's own): a d6 result is a 'roll_num' record
        ('<game> X rolls 4'); a hunt tile is a 'draw' record whose
        piece parses as a tile; a random companion is a 'draw' record
        naming a figure; a WWSAT discard is a 'random_discard'."""
        from . import live_parser
        import rendered_replay as rr
        for ln in tail:
            s = str(ln)
            try:
                rec = live_parser.parse(s)
            except Exception:
                continue
            op = rec.get('op')
            if kind == 'd6':
                if op == 'roll_num':
                    return s
            elif kind in ('tile', 'randomcomp'):
                if op == 'draw':
                    pp = rr.parse_piece(str(rec.get('what', '')
                                            or '').strip())
                    want = 'tile' if kind == 'tile' else 'named'
                    if pp is not None and pp[0] == want:
                        return s
            elif kind == 'card_pick':
                if op == 'random_discard':
                    return s
            elif kind == 'draws':
                # an AI-deck draw the ask ordered: the client renders
                # '<nick> draws <Deck back label>.' (a 'draw' record
                # whose piece is a card back / a card)
                if op == 'draw':
                    what = str(rec.get('what', '') or '').strip().lower()
                    if what.endswith(' card') and (
                            'strategy' in what or 'character' in what):
                        return s
        return None

    # ------------------------------------------------------------------
    def _say(self, kind, text):
        from emitter import render
        self.chat_out.say(kind, text)       # the ONE policy exit
        return render('chat', {'nick': self.nick, 'msg': text},
                      local=False)

    def _dir_lines(self, call):
        """Directives for a pending call, deduped per identity so a
        retried tick never re-orders the client, and deduped against
        the calls THIS STEP's emission already ordered (the cbt1
        double-draw class; see _step_calls).

        WAVE-K (wkg1 LIVE WEDGE, HARD STALL 121s): identity AND
        fulfilment GENERATION.  A step can legitimately need the SAME
        directive twice (the Palantir of Orthanc grants an EXTRA draw
        on every Event-die draw; 'draw N' effects; a second hunt
        tile): text-keyed dedup suppressed the second order forever -
        the retry re-consumed the one fed card, starved on the same
        key, and the ask was 'sent once, never re-sent'.  gen = how
        many client outcomes the call's channel has delivered THIS
        STEP; an identical ask re-orders only after the previous order
        was fulfilled (gen grew) - an idle retry (gen unchanged) still
        never re-orders (the cbt1 property)."""
        gen = self._call_gen(call)
        ident = (call.kind, tuple(sorted(
            (k, str(v)) for k, v in call.kw.items())), gen)
        if ident in self._dirs_sent:
            return []
        self._dirs_sent.add(ident)
        out = []
        for d in self.rand.directives_for(call):
            if d.kind == 'chat':
                out.append(self._say(
                    'random_ask' if d.text.endswith('?')
                    else 'roll_request', d.text))
            elif d.text in self._step_calls \
                    and self._step_calls[d.text] >= gen:
                self.notes.append('directive %r already ordered by '
                                  "this step's emission - not "
                                  're-ordered' % d.text)
            else:
                self._step_calls[d.text] = gen
                if d.text == '!CALL hunt_tile':
                    out.extend(self._tile_pool_preheal())
                out.append(d.text)
        return out

    def _text_gen(self, text):
        """Fulfilment generation of a directive TEXT: how many client
        outcomes its channel has delivered this step (per draw deck,
        the tile list, the companion list; AI-side d6 for combat_roll
        - untagged lines count, side-tagged only for the AI side).
        Unknown texts = 0 (never re-ordered: play_card, menu...)."""
        f = getattr(self.rand, '_feed', None) or {}
        t = str(text)
        if t.startswith('!CALL draw_'):
            tail = t[len('!CALL draw_'):]
            side_name = ('FREE_PEOPLES' if tail.startswith('fp')
                         else 'SHADOW')
            deck = 'character' if tail.endswith('char') else 'strategy'
            return len((f.get('cards') or {}).get((side_name, deck),
                                                  ()))
        if t == '!CALL hunt_tile':
            return len(f.get('tiles') or ())
        if t == '!CALL random_companion':
            return len(f.get('companion') or ())
        if t.startswith('!CALL combat_roll'):
            me = self.ai_side.name
            return sum(1 for tag, _v in (f.get('d6') or ())
                       if tag is None or tag == me)
        return 0

    def _call_gen(self, call):
        """Fulfilment generation of a pending NeedClientCall (see
        _text_gen) - the feed channel the call starves on."""
        f = getattr(self.rand, '_feed', None) or {}
        k = getattr(call, 'kind', None)
        kw = getattr(call, 'kw', {}) or {}
        if k == 'draws':
            cards = f.get('cards') or {}
            return sum(len(cards.get((str(sn), str(dk)), ()))
                       for sn, dk in (kw.get('items') or ()))
        if k == 'tile':
            return len(f.get('tiles') or ())
        if k == 'randomcomp':
            return len(f.get('companion') or ())
        if k == 'card_pick':
            return len(f.get('card_pick') or ())
        if k == 'd6':
            want = kw.get('side')
            want = getattr(want, 'name', want)
            want = str(want) if want is not None else None
            return sum(1 for tag, _v in (f.get('d6') or ())
                       if tag is None or want is None or tag == want)
        return 0

    def _emission_out(self, em):
        out = list(em.lines) + list(em.calls)
        for k, t in em.chat:
            out.append(self._say(k, t))
        return out

    # ------------------------------------------------------------------
    def _ai_decide(self, dp):
        """One journal-pinned, clone-applied, once-emitted AI step."""
        import ai
        self.rand.begin_attempt()

        def chooser(actions):
            if self.fp_net is None:
                return list(actions)        # netless harness mode
            act, _pol = ai.mcts_action_with_policy(
                self.driver.es, dp['side'], self.fp_net, self.sh_net,
                num_simulations=self.sims, temperature=0.0,
                add_noise=False)
            ordered = list(actions)
            if act is None:
                # the search returned nothing for a non-empty engine
                # menu: a degraded decision is never taken silently
                raise DriverHalt(
                    'MCTS returned no action for the %s %s decision '
                    '(%d legal) - refusing to fall back to menu[0]'
                    % (dp['side'].name, dp['kind'], len(ordered)))
            hit = next((a for a in ordered
                        if a.action_type == act.action_type
                        and a.die_idx == act.die_idx
                        and a.params == act.params), None)
            if hit is None:
                # the search picked an action the driver's menu does
                # not hold (menu drift between two legality probes):
                # LOUD - alarm + note - before the menu order decides
                msg = ('MCTS PICK NOT IN MENU: %s %s picked %s die=%r '
                       '%r but the engine menu (%d) lacks it - falling '
                       'back to the menu order, visibly'
                       % (dp['side'].name, dp['kind'],
                          act.action_type, act.die_idx,
                          _menu_params_safe(act.params), len(ordered)))
                self.driver.alarms.append(msg)
                self.notes.append(msg)
                print('  ! %s' % msg)
            else:
                ordered.remove(hit)
                ordered.insert(0, hit)
            return ordered

        picked = self.rand.journal_pick(dp['actions'], chooser)
        act = picked[0] if isinstance(picked, list) else picked
        try:
            with self.rand.advance():
                post = ai.apply_action(self.driver.es, act)
        except rnd_mod.NeedClientCall as e:
            out = []
            # card-effect emission is DELTA-derived (the pre/post the
            # server holds around the apply) -- there is no post yet on
            # this branch, so it is DEFERRED to the successful retry:
            # the client's own randomness precedes the board effect it
            # feeds, and _em_sent stays False so the post-apply render
            # still goes out exactly once.
            em = None
            if not self._em_sent and str(dp['kind']) not in (
                    'card_option', 'card_followup') \
                    and str(getattr(act, 'action_type', '')) \
                    != 'hunt_sacrifice_random':
                # (WAVE-K K7: the random-casualty decision's victim IS
                # the client's draw - nothing to render before it; the
                # post-apply render names the companion that left)
                em = self._render(dp, act, post=None)
                if em.hold:
                    raise DriverHalt('EMISSION HOLD (pre-apply): %s'
                                     % em.hold)
            # Invariant: the emission is booked as SENT only once the
            # whole cycle's lines exist.  The directive is built first
            # (its dedup reads the emission's own calls, booked
            # provisionally); a builder failure restores the books and
            # HOLDS the step loudly - nothing emitted, nothing lost, the
            # retry starts as a first attempt.
            calls_snap = dict(self._step_calls)
            dirs_snap = set(self._dirs_sent)
            if em is not None:
                for _c in em.calls:
                    self._step_calls[_c] = self._text_gen(_c)
            try:
                dir_out = self._dir_lines(e)
            except rnd_mod.NeedClientCall:
                raise
            except DriverHalt:
                self._step_calls = calls_snap
                self._dirs_sent = dirs_snap
                raise
            except Exception as _de:
                self._step_calls = calls_snap
                self._dirs_sent = dirs_snap
                raise DriverHalt('DIRECTIVE BUILD FAILED for client %s '
                                 '(%r): %s - step held, nothing emitted'
                                 % (e.kind, e.kw, _de))
            if em is not None and not (em.lines or em.calls or em.chat
                                       or em.ops):
                # an EMPTY pre-apply render (e.g. a Mordor-track
                # fellowship move whose tile the client still owes):
                # nothing was expressed, so the resumed attempt renders
                # the whole action from the post state
                em = None
            if em is not None:
                out.extend(self._emission_out(em))
                self._em_sent = True
                self._note_cc_chosen(em)
                # WAVE-I: the pre-apply emission's ops survive the
                # cross-tick retry so the post-commit sweep can
                # subtract the rows this emission already expressed
                self._step_expressed = list(em.ops)
                self._note_owed_die_kill(dp, act)
                self._note_owed_card_kill(dp, act)
            out.extend(dir_out)
            return out
        # success: honesty gate, then emit (if not already), commit
        exc = {k: v for k, v in self.rand.unconsumed().items() if v}
        if exc:
            self.driver.alarms.append('EXCESS RANDOMNESS at step end: '
                                      '%r' % exc)
        out = []
        expressed_ops = list(getattr(self, '_step_expressed', ()) or ())
        if not self._em_sent:
            em = self._render(dp, act, post=post)
            if em.hold:
                raise DriverHalt('EMISSION HOLD: %s' % em.hold)
            out.extend(self._emission_out(em))
            self._note_cc_chosen(em)
            for _c in em.calls:
                self._step_calls[_c] = self._text_gen(_c)
            expressed_ops.extend(em.ops)
            self._note_owed_die_kill(dp, act)
            self._note_owed_card_kill(dp, act)
        # WAVE-H cbt5: an IMMEDIATE-effect play (no pending chain
        # opened) applies its board effect engine-side with no gesture
        # (Orcs Multiplying Again placed 3 regulars; every later move
        # of that army held 'boards cannot supply the pieces').  The
        # bounded card-effect row renderer (the sanctioned rule-6
        # exception) covers exactly this delta; chains that DID open
        # keep rendering per decision as before.
        if str(getattr(act, 'action_type', '')) == 'play_event' \
                and not emission.chain_open(post) \
                and not emission.play_was_refunded(self.driver.es,
                                                  post, act):
            from . import emission_card
            # the play's OWN die spend is already rendered (staged
            # prologue) - mirror it onto the pre so the delta carries
            # only the CARD's effect (the row renderer rightly holds
            # on any other dice change)
            pre_fx = self.driver.es
            didx = getattr(act, 'die_idx', None)
            if didx is not None and int(didx) >= 0:
                pre_fx = ai.clone_state(pre_fx)
                try:
                    d_post = post.dice[dp['side']][int(didx)]
                    d_pre = pre_fx.dice[dp['side']][int(didx)]
                    d_pre.used = d_post.used
                    d_pre.current = d_post.current
                except (KeyError, IndexError):
                    pass
            em2 = emission_card.render_card_effect(
                self.driver.view, self.nick, dp['side'],
                'play_event:immediate', act,
                pre=pre_fx, post=post)
            if em2.hold:
                raise DriverHalt('EMISSION HOLD (card effect): %s'
                                 % em2.hold)
            out.extend(self._emission_out(em2))
            expressed_ops.extend(em2.ops)
        pre = self.driver.es
        self.driver._snapshot([], 'ai:%s' % dp['kind'], post=post,
                              action=act)
        self.driver.es = post
        self.driver.hands.es = post
        if str(dp['kind']) == 'action':
            # a plain die action alternates the turn (pendings do not)
            self.driver.last_actor = self.ai_side
        # a declined OPTIONAL card leg is stated in chat (the board
        # cannot show a choice NOT taken)
        try:
            _dl = chat_mod.declined_leg(act, dp.get("actions"))
        except Exception as _dle:
            _dl = None
            self.notes.append('declined-leg chat failed: %r' % (_dle,))
        if _dl:
            out.append(self._say('leg_decline', _dl))
        # G2 (WAVE-H): a combat round that resolved inside THIS apply
        # announces its hit counts (engine statement; one producer per
        # commit path - the driver's own commit paths mirror this)
        for k, t in emission.round_hits_chat(post,
                                             self.driver.human_side):
            out.append(self._say(k, t))
        # WAVE-I (one producer per commit path): forced allocations /
        # capture markers this apply resolved beyond the decision's
        # own emission (the expressed ops subtract first)
        for item in self.driver.post_commit_sweep(
                pre, post, expressed=expressed_ops,
                label='ai:%s' % dp['kind']):
            out.extend(self._dispatch(item))
        self.rand.step_commit()
        self._em_sent = False
        self._dirs_sent.clear()
        self._step_calls.clear()
        self._step_expressed = []
        self._step_owed = []
        self._open_step = None
        return out

    def _render(self, dp, act, post):
        k = str(dp['kind'])
        if k == 'declare' and str(getattr(act, 'action_type', '')) \
                == 'pass':
            # WAVE-K R2: the AI-as-FP DECLINES to declare - absence is
            # the choice (no board gesture); its 'fp ok' follows
            return emission.Emission()
        if k.startswith(('combat', 'hunt', 'guide')) \
                or k in ('discard', 'card_option', 'card_followup',
                         'reveal_place'):
            # pre = the driver's UNCOMMITTED engine state (the state
            # the decision was applied to; es is swapped to post only
            # after emission) -- the card-effect row renderer diffs
            # pre against post (emission_card, bounded rule-6
            # exception).
            return emission.render_decision(self.driver.view,
                                            self.nick, dp['side'], k,
                                            act, post=post,
                                            pre=self.driver.es)
        return emission.render_action(self.driver.view, self.nick,
                                      dp['side'], act, post=post)

    # ------------------------------------------------------------------
    def _ai_draws(self):
        """The AI's turn-start draws (one per deck), clone-applied so
        a starve leaves the state untouched and the retry re-runs both
        deterministically."""
        import ai
        import game_logic as gl
        import game_state as gs
        self.rand.begin_attempt()
        clone = ai.clone_state(self.driver.es)
        # WAVE-J (ledger rank 15, donor deck-empty guards
        # :16567-16580): an EMPTY deck skips that draw legitimately -
        # gl.draw_event_card raises on an empty deck and only
        # NeedClientCall is caught, so a late-game dry deck became a
        # per-tick cycle error forever.
        sh = self.ai_side is gs.Side.SHADOW
        decks = (('character', clone.sh_char_deck if sh
                  else clone.fp_char_deck),
                 ('strategy', clone.sh_strat_deck if sh
                  else clone.fp_strat_deck))
        try:
            with self.rand.advance():
                for dname, dlist in decks:
                    if not dlist:
                        self.notes.append(
                            'ai_draws: %s deck empty - draw skipped '
                            '(legitimate late-game)' % dname)
                        continue
                    gl.draw_event_card(clone, self.ai_side, dname)
        except rnd_mod.NeedClientCall as e:
            return self._dir_lines(e)
        self.driver._snapshot([], 'ai_draws', post=clone)
        self.driver.es = clone
        self.driver.hands.es = clone
        self.rand.step_commit()
        self._dirs_sent.clear()
        self._step_calls.clear()
        self._open_step = None
        return []

    def _ai_eyes(self):
        """The AI's (Shadow's) eye allocation, BEFORE any roll (D3):
        the engine's eye menu under the _eye_allocation_phase flag,
        decided by the nets, emitted via the planners ('0e' chat on a
        zero ride the emission).  On commit: the roll ask goes out —
        '!CALL roll_action_dice' for the AI's own pool, a chat ask for
        the human's (G1: each side rolls its own)."""
        import ai
        self.driver.es._eye_allocation_phase = True
        try:
            dp = ai.describe_pending(self.driver.es, self.ai_side)
            if dp is not None and dp['side'] is not self.ai_side:
                # a HUMAN-owned pending precedes the eye phase (cbt7:
                # the turn-3 hand-limit discard) - keep the step OPEN
                # and retry each tick; the loud stall counter keeps
                # the wait visible.  The old close-with-a-note skipped
                # the eye step and the roll ask FOREVER, silently.
                self._open_step = ('ai_eyes', None)
                return []
            if dp is None or not dp['actions']:
                self._open_step = None
                msg = 'ai_eyes: no eye menu (pool empty?)'
                self.notes.append(msg)
                print('  [note] %s' % msg)
                return []
            out = self._ai_decide(dp)
        finally:
            self.driver.es._eye_allocation_phase = False
            # the committed post state carries the flag copy too
            if getattr(self.driver.es, '_eye_allocation_phase', False):
                self.driver.es._eye_allocation_phase = False
        if self._open_step is None and str(dp['kind']) != 'eye':
            # the AI's OWN preceding pending (its hand-limit discard)
            # committed - the eye decision is still owed: re-open
            self._open_step = ('ai_eyes', None)
            return out
        if self._open_step is None:          # the eye step committed
            self.driver.phase = 'roll'
            # the FULL zone code ('Game.2043'): AiOpp.findZone matches
            # '<~'+code+'~>' against zone.e() - the bare '2043' form
            # missed every zone and the roll !CALL was a no-op (boot3
            # leg, 2026-08-19).  emitter owns the pool constants.
            from emitter import FP_DICE_POOL, SH_DICE_POOL
            zone = (SH_DICE_POOL if self.ai_side.name == 'SHADOW'
                    else FP_DICE_POOL)
            call = rnd_mod.NeedClientCall(
                'action_roll',
                sides={self.ai_side.name: (zone, [])})
            out.extend(self._dir_lines(call))
            # ...AND THE CHAT ASK FOR THE HUMAN'S OWN POOL (G1: each
            # side rolls its own).  This docstring promised it from
            # the start and the code never sent it: the human was
            # never told to roll, so their pool sat unrolled, and an
            # unrolled pool reads to the engine's alternation
            # answerer exactly like an exhausted one - the AI then
            # took the WHOLE turn (USER 2026-08-20: "it's not
            # respecting turn order at all").  driver.pools_rolled()
            # is the correctness gate; this is how the human learns
            # what the gate is waiting for.
            n_owed = len([d for d in self.driver.es.dice.get(
                              self.human_side, [])
                          if not d.used and d.current is None])
            if n_owed:
                out.append(self._say('roll_request',
                                     chat_mod.roll_request(n_owed)))
        else:
            self._open_step = ('ai_eyes', None)   # keep retrying HERE
        return out

    def _ai_declare(self):
        """WAVE-K R2 / K2: the AI-as-FP's OWN declaration decision at
        the turn start (learned, via the nets over the engine's declare
        menu), emitted through the generic planner route (emitter
        fellowship_declare; a decline renders nothing), committed, then
        the Mordor-entry check (driver.after_ai_declare - the ONE
        helper the inbound 'fp ok' path uses) and the AI's own 'fp ok'.

        Invariant (K2): this step never arms or clears the engine's
        declare flag - the driver arms it (game_logic.open_declare_phase)
        for whichever side is FP and closes it; the step only DECIDES
        when describe_pending says 'declare' for the AI side.  A
        human-owned pending preceding it (the Shadow's hand-limit
        discard) keeps the step OPEN and retried, like ai_eyes."""
        import ai
        import game_state as gs
        drv = self.driver
        if self.ai_side is not gs.Side.FREE_PEOPLES:
            return self._abandon_step('ai_declare on the Shadow seat')
        if drv.phase != 'declare':
            # the phase moved on (a recover/resync) - nothing to declare
            return self._abandon_step('ai_declare after the phase moved')
        with self.rand.pin():
            any_dp = ai.describe_pending(drv.es)
        if any_dp is not None and any_dp['side'] is not self.ai_side:
            self._open_step = ('ai_declare', None)
            return []
        # the driver's own answerer arms the menu for the FP seat and
        # returns the declare dp when it is open
        dp = drv.ai_action_dp()
        out = []
        if dp is not None and str(dp['kind']) != 'declare' \
                and dp['actions']:
            # the AI's OWN preceding pending (its hand-limit discard)
            # commits first; the declaration stays owed - retry here
            out = self._ai_decide(dp)
            self._open_step = ('ai_declare', None)
            return out
        if dp is not None and str(dp['kind']) == 'declare' \
                and dp['actions']:
            out = self._ai_decide(dp)
            if self._open_step is not None:
                # starved/held mid-step (should not happen: the
                # declaration has no randomness) - keep retrying
                self._open_step = ('ai_declare', None)
                return out
        else:
            self.notes.append('ai_declare: no declare menu (%r) - '
                              'declining by absence'
                              % (dp and dp['kind'],))
        for item in drv.after_ai_declare():
            out.extend(self._dispatch(item))
        out.append(self._say('fp_ok', 'fp ok'))
        self._open_step = None
        return out

    def _order_own_roll(self):
        """WAVE-K R2 (G11): the AI-as-FP orders its OWN pool roll once
        the human Shadow's roll has been adopted (driver item
        'ai_roll').  The directive text comes from the ONE answerer
        (randomness.directives_for); this fires once per turn by
        construction, so no step-scoped dedup applies."""
        import game_state as gs
        from emitter import FP_DICE_POOL, SH_DICE_POOL
        zone = (FP_DICE_POOL if self.ai_side is gs.Side.FREE_PEOPLES
                else SH_DICE_POOL)
        call = rnd_mod.NeedClientCall(
            'action_roll', sides={self.ai_side.name: (zone, [])})
        out = []
        for d in self.rand.directives_for(call):
            if d.kind == 'chat':
                out.append(self._say('roll_request', d.text))
            else:
                out.append(d.text)
        self.driver.phase = 'roll'
        return out

    def _recover(self):
        """WAVE-J (USER: the client turn counter NEVER TICKED - the
        hand-built bulk die moves bypassed the client's own handler).
        THE RULING 'ask the client what commands it has': the AI's
        recovery is the CLIENT'S OWN 'Recover Dice' menu command
        ('!CALL menu srecover' Shadow / 'frecover' FP - donor
        :17264-17274, live-proven there), whose handler emits the
        corpus recover groups from their TRUE zones (Controls.757/762
        / 726/731) AND bumps the client turn marker ('<~Game.7~>N')
        when the second side's recovery lands.  The AI orders ITS OWN
        side only; the human runs their own (harness: same menu
        route).  emission.render_recover (the reinvention) is retired
        from this path."""
        cmd = ('srecover' if self.ai_side.name == 'SHADOW'
               else 'frecover')
        self.notes.append('recover via the client menu: %s' % cmd)
        return ['!CALL menu %s' % cmd]

    # ------------------------------------------------------------------
    def _mordor_entry_lines(self, payload):
        """WAVE-J (ledger A-29): the Mordor entry CHOREOGRAPHY (donor
        _mordor_entry_lines :16467-16545) - pawn 247 to Mordor box 0,
        the progress marker home (Scribe drag from its pre-entry
        slot), the Used->Pool tile reshuffle for the faces the ENGINE
        returned, and the 'fsp enters mordor' announce."""
        import emitter
        import log_replay as lr
        from emitter import render
        from board import sanitize
        drv = self.driver
        out = [self._say('mordor_entry', 'fsp enters mordor')]
        # pawn: from its client zone (dump-true) to Mordor box 0
        frm = None
        try:
            from . import client_board
            _dp = getattr(drv.view, 'dump_path', None)
            dump = client_board.read_board(_dp) if _dp else None
            z = (dump or {}).get('247')
            if z:
                frm = emitter.zone_code_for(sanitize(str(z)))
        except Exception:
            frm = None
        mz = emitter.MORDOR_TRACK_ZONES
        if frm and frm != mz[0]:
            out.append(render('move', {'nick': self.nick,
                                       'pieces': ['247'],
                                       'frm': frm, 'to': mz[0]},
                              local=False))
        pre_p = int((payload or {}).get('pre_progress', 0) or 0)
        if pre_p > 0:
            out.append(render('move', {
                'nick': self.nick, 'pieces': ['248'],
                'frm': 'Game.%d' % (2014 + pre_p),
                'to': 'Game.2014'}, local=False))
        # tile reshuffle: client Used-zone tiles whose faces the
        # ENGINE returned to the pool ride the corpus drag 2045->2044
        try:
            pool_faces = {}
            for t in (getattr(drv.es, 'hunt_pool', []) or []):
                f = _tile_face(t)
                pool_faces[f] = pool_faces.get(f, 0) + 1
            back = []
            for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
                zz = sanitize(str((dump or {}).get(pid, '')))
                if zz != 'used hunt tiles':
                    continue
                f = _tile_face(nm)
                if pool_faces.get(f, 0) > 0:
                    pool_faces[f] -= 1
                    back.append(pid)
            if back:
                out.append(render('move', {
                    'nick': self.nick, 'pieces': sorted(back),
                    'frm': 'Game.2045', 'to': 'Game.2044'},
                    local=False))
        except Exception as e:
            self.notes.append('mordor reshuffle skipped: %r' % (e,))
        self.notes.append('MORDOR ENTRY rendered (pre_progress=%d)'
                          % pre_p)
        return out

    def _witness_cadence(self):
        """WAVE-J (ledger rank 9, W-01/W-02): the continuous
        engine-vs-client witness - every N cycles, diff + card census,
        persistence-deduped notes, a jsonl record, and a 3-strike
        ESCALATION into driver.alarms when the same non-empty row set
        settles (the donor's census_alert discipline)."""
        drv = self.driver
        # mid-window states legitimately lead/lag the engine (wjg4:
        # the deferred fellowship move's die sat in the hunt box
        # client-side while the engine apply waited on the hunt roll -
        # the cadence escalated a NON-desync).  The witness only
        # judges SETTLED states.
        if drv.deferred is not None or drv.chat_deferred is not None \
                or drv.pending_buf is not None \
                or self._open_step is not None \
                or drv.owed_die_kills:
            return
        try:
            if not drv.view.refresh_dump():
                return
            from . import client_board
            rows = client_board.diff_engine_vs_dump(
                drv.es, drv.view.piece_region,
                dynamic_types=drv.view.dynamic_types) or []
            # AI-side identity witness from the CLIENT's own facts:
            # pids in the AI hand zone decoded via the commitment map
            # (fact 10).  Only when EVERY zone pid decodes - a partial
            # decode compares apples to a subset.  The human side is
            # size-only (blinding: its identities are structurally
            # unknown; the wjg5 false-alarm class).
            from board import sanitize as _san
            inv = {str(v): k
                   for k, v in drv.view.card_pid_map.items()}
            zone = ('sa hand' if self.ai_side.name == 'SHADOW'
                    else 'fp hand')
            zpids = [p for p, z in drv.view.piece_region.items()
                     if _san(str(z)) == zone]
            hand_cids = None
            if zpids and all(str(p) in inv for p in zpids):
                hand_cids = {self.ai_side: [inv[str(p)]
                                            for p in zpids]}
            rows += client_board.card_census(
                drv.es, drv.view.piece_region,
                hand_cids=hand_cids) or []
        except Exception as e:
            self.notes.append('witness cadence error: %r' % (e,))
            return
        fp = tuple(sorted(str(r)[:120] for r in rows))
        prev = getattr(self, '_witness_fp', None)
        if fp != prev:
            self._witness_fp = fp
            self._witness_streak = 1
            if rows:
                self.notes.append('witness: %d row(s) (first: %s)'
                                  % (len(rows), rows[0]))
            elif prev:
                self.notes.append('witness: clean again')
        elif rows:
            self._witness_streak = getattr(self, '_witness_streak',
                                           1) + 1
            if self._witness_streak == 3:
                drv.alarms.append(
                    'WITNESS ESCALATION: %d row(s) settled over 3 '
                    'checks (first: %s)' % (len(rows), rows[0]))
        try:
            import json as _json
            import time as _time
            with open(os.path.join(self.client_dir,
                                   'ai_census.jsonl'), 'a',
                      encoding='utf-8') as f:
                f.write(_json.dumps({
                    't': round(_time.time(), 1),
                    'turn': int(getattr(drv.es, 'turn', 0)),
                    'rows': len(rows),
                    'first': (str(rows[0])[:160] if rows
                              else None)}) + '\n')
        except Exception:
            pass

    def on_control(self, c):
        """The panel's non-pause levers.  sims applies live; manual_edit
        holds the machine (spectator until the epoch clears + a resync);
        the one-shot epochs (resync / skip / reset / whose-turn) act
        once each (epochs present at boot were adopted as watermarks in
        __init__ and never replayed)."""
        try:
            # SIMS (USER 2026-08-20: "the mcts sim count doesn't update
            # / acknowledge my setting").  TWO defects here:
            #   1. `s != 1000` silently DISCARDED a setting of exactly
            #      1000.  That guard exists because AiOpp spawns the
            #      old server with a literal --sims 1000, so a control
            #      file echoing 1000 was treated as that default rather
            #      than a request - but the panel slider can sit on
            #      1000 too, and then the user's setting vanished.
            #      The new stack is ADOPTED, never spawned by AiOpp
            #      (launchServer: 'spawn SKIPPED'), so that default
            #      cannot reach us and the guard only ever discarded
            #      real requests.
            #   2. nothing ever published `sims` back, so the panel -
            #      which reads st['sims'] for its 'thinking (N sims)'
            #      line - could not acknowledge any value at all.
            #      Published in status_menu now.
            s = int(c.get('sims', 0) or 0)
            if s and s != self.sims:
                self.notes.append('panel: sims %d -> %d'
                                  % (self.sims, s))
                self.sims = s
            # SEARCH WIDTH (USER 2026-08-22 'cut those caps but add extra
            # sliders with those set as default'): MCTS children per
            # node / chance outcomes per node; 0 = no cap
            import ai as _ai_sw
            _mc = c.get('max_children'); _ck = c.get('chance_k')
            if _mc is not None or _ck is not None:
                _mc = None if _mc is None else int(_mc)
                _ck = None if _ck is None else int(_ck)
                _before = (_ai_sw.MCTS_MAX_CHILDREN, _ai_sw.CHANCE_K_MAX)
                _after = _ai_sw.set_search_limits(_mc, _ck)
                if _after != _before:
                    self.notes.append(
                        'panel: search width children %s -> %s, chance '
                        'outcomes %s -> %s (0 = no cap)'
                        % (_before[0], _after[0], _before[1], _after[1]))
            # ENFORCE LEGAL MOVES FOR THE HUMAN (beta) - USER 2026-08-22:
            # unchecked = the AI never countermands the human and takes
            # the board as given after each action
            en = bool(c.get('enforce_human', True))
            if en != bool(getattr(self.driver, 'enforce_human', True)):
                self.driver.enforce_human = en
                self.notes.append('panel: legality enforcement for the '
                                  'human %s' % ('ON (beta)' if en
                                                else 'OFF - human moves '
                                                     'taken as given'))
            me = bool(c.get('manual_edit', False))
            if me != getattr(self, '_manual_hold', False):
                self._manual_hold = me
                self.notes.append('panel: manual edit %s'
                                  % ('HOLD - machine is a spectator'
                                     if me else 'FINISHED'))
            rs = float(c.get('resync', 0) or 0)
            if rs > getattr(self, '_resync_epoch', 0):
                self._resync_epoch = rs
                self._resync_requested = True
                self.notes.append('panel: resync requested')
            # SKIP DECISION (USER 2026-08-20: "I want it to be able to
            # skip the decision").  Deferred to the next handle() the
            # way resync is - a commit belongs on the tick, not inside
            # the transport's control read.
            sk = float(c.get('skip_pending', 0) or 0)
            if sk > getattr(self, '_skip_epoch', 0):
                self._skip_epoch = sk
                self._skip_requested = True
                self.notes.append('panel: skip-decision requested')
            # RESET + RESYNC (USER 2026-08-20: "a button to resync and
            # reset and clear all pending decisions, so the AI
            # continues from there").  Same tick-deferred pattern.
            rr9 = float(c.get('reset_resync', 0) or 0)
            if rr9 > getattr(self, '_reset_epoch', 0):
                self._reset_epoch = rr9
                self._reset_requested = True
                self.notes.append('panel: RESET + RESYNC requested')
            # WHOSE-TURN OVERRIDE (USER 2026-08-20 panel buttons).  The
            # ENGINE still owns the turn RULE - ai._next_acting_side is
            # the one answerer and is not touched here.  All this sets
            # is the driver's own bookkeeping of WHO LAST ACTED, which
            # is the single input that answerer takes from us; when the
            # two have drifted (a gesture the matcher never saw, a
            # halted step) the game would otherwise be unplayable.
            # Epoch-stamped so one press acts once.
            wt = float(c.get('whose_turn_epoch', 0) or 0)
            if wt > getattr(self, '_whose_turn_epoch', 0):
                self._whose_turn_epoch = wt
                who = str(c.get('whose_turn', '')).lower()
                # AN OPEN DECISION IS NOT AN ALTERNATION PROBLEM (USER
                # 2026-08-20: "clear error -> your turn does not work,
                # it still waits for my decision").  These buttons move
                # last_actor, which only governs the PLAIN action
                # alternation; while the engine holds a pending owned
                # by a side, ai_action_dp consults describe_pending
                # FIRST and the AI correctly refuses to act.  Skipping
                # it here would mean answering a decision on the
                # player's behalf.  Say so instead of silently doing
                # nothing.
                try:
                    import ai as _ai_t
                    with self.rand.pin():
                        _dp = _ai_t.describe_pending(self.driver.es)
                except Exception:
                    _dp = None
                if _dp is not None and who in ('human', 'ai'):
                    msg = ('panel: turn override IGNORED - %s still '
                           'owes a %s decision; answer it on the board '
                           'or in chat, or press SKIP DECISION to '
                           'decline it (the turn buttons only fix '
                           'whose plain action is next)'
                           % (_dp['side'].name, _dp['kind']))
                    self.notes.append(msg)
                    print('  [note] %s' % msg)
                elif who in ('human', 'ai'):
                    # last_actor is the OTHER side, so the answerer
                    # names the side the user asked for
                    self.driver.last_actor = (
                        self.ai_side if who == 'human'
                        else self.human_side)
                    self.notes.append(
                        'panel: turn override -> %s acts next'
                        % ('the human' if who == 'human' else 'the AI'))
        except Exception as e:
            self.notes.append('panel control error: %r' % (e,))

    def _tile_mopup(self):
        """WAVE-J slot cleanup, tile half (the donor's _sided_mopup
        tile stratum re-landed for the new stack; USER B1/B2 family):

        * a RESOLVED hunt tile leaves the Current-Card display slot
          for Used Hunt Tiles Game.2045 (a face the engine holds in
          hunt_removed goes to Removed Game.2046 instead - Challenge
          of the King class);
        * a card-added SPECIAL tile (The Ring is Mine! class) parks
          from the Removed stash into Used - the corpus choreography
          (donor B1: Game.2046 -> Game.2045), covering ALL 8 special
          tiles by the piece table;

        The client's own hunt handler only ever ADDS tiles to the
        display and never clears them (donor t116), and the resync
        witnesses are BLIND to non-region zones (typed pieces only) -
        this reconciler is the one keeper.  Sweeps run ONLY while no
        hunt/card pending is open (the donor's t130 HOLD: a tile in
        decision limbo stays displayed); each sweep line is re-sent
        at most 3 times per piece, then a LOUD note."""
        drv = self.driver
        es = drv.es
        hd = list(getattr(es, 'hunt_discard', []) or [])
        hr = list(getattr(es, 'hunt_removed', []) or [])
        sig = (len(hd), len(hr))
        if sig != getattr(self, '_tile_sig', None):
            self._tile_sig = sig
            if sig != (0, 0):
                self._tile_dirty = True
        if not getattr(self, '_tile_dirty', False):
            return []
        for attr in ('pending_hunt', 'pending_card_option',
                     'pending_card_followup'):
            if getattr(es, attr, None) is not None:
                return []
        if getattr(es, '_reveal_place_due', False):
            return []
        from collections import Counter
        from board import sanitize
        import log_replay as lr
        from . import client_board
        from emitter import render
        _dp = getattr(drv.view, 'dump_path', None)
        dump = client_board.read_board(_dp) if _dp else None
        if dump is None:
            return []
        swept = getattr(self, '_tile_swept', None)
        if swept is None:
            swept = self._tile_swept = {}
        disp = {'fp card play', 'sa card play', 'current fp card',
                'current sa card', 'current action'}
        rm_faces = Counter(_tile_face(t) for t in hr)
        out = []
        work_left = False
        # (1) resolved display tiles -> Used (or Removed)
        for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
            z = sanitize(str(dump.get(pid, '')))
            if z not in disp:
                swept.pop(pid, None)
                continue
            work_left = True
            n = swept.get(pid, 0)
            if n >= 3:
                if n == 3:
                    swept[pid] = 4
                    self.notes.append(
                        'tile mop-up: piece %s still displayed after '
                        '3 sweep lines - client not applying them'
                        % pid)
                continue
            f = _tile_face(nm)
            dest = 'Game.2046' if rm_faces.get(f, 0) > 0 \
                else 'Game.2045'
            if dest == 'Game.2046':
                rm_faces[f] -= 1
            try:
                import emitter
                frm = emitter.zone_code_for(z)
            except LookupError:
                frm = 'Game.2036'
            out.append(render('move', {
                'nick': self.nick, 'pieces': [pid],
                'frm': frm, 'to': dest}, local=False))
            swept[pid] = n + 1
        # (2) B1 specials-in-play: engine discard holds a special the
        # client never drew - its piece sits in the Removed stash
        b1 = getattr(self, '_tile_b1', None)
        if b1 is None:
            b1 = self._tile_b1 = set()
        want_sp = Counter(_tile_face(t) for t in hd
                          if getattr(t, 'is_special', False))
        for f0 in list(want_sp):
            have = sum(1 for p in b1
                       if _tile_face(lr.HUNT_TILE_PIECES.get(p, ''))
                       == f0)
            need = want_sp[f0] - have
            for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
                if need <= 0:
                    break
                if pid in b1 or _tile_face(nm) != f0:
                    continue
                z = sanitize(str(dump.get(pid, '')))
                if z != 'removed hunt tiles':
                    continue
                out.append(render('move', {
                    'nick': self.nick, 'pieces': [pid],
                    'frm': 'Game.2046', 'to': 'Game.2045'},
                    local=False))
                b1.add(pid)
                need -= 1
            if need > 0:
                work_left = True
        if not work_left and not out:
            self._tile_dirty = False
        if out:
            self.notes.append('tile mop-up: %d sweep line(s)'
                              % len(out))
        return out

    def _note_owed_die_kill(self, dp, act):
        """USER invariant 'die -> action -> resolve everything -> kill
        die' (donor :15760-15829): army_attack / play_event / draw_card
        keep their die in Current Action until the chain closes.  Bank
        the owed kill at EMISSION time (the die-spend line is already
        on its way to the client either way)."""
        at = getattr(act, 'action_type', None)
        didx = getattr(act, 'die_idx', None)
        # discard_table_card BELONGS HERE (USER 2026-08-20: "it should
        # also kill its active die, which it neglected to do").  The
        # corpus ends the 1.149 table kill with `kills group (die,
        # card) in Current Action`; the card was banked in
        # _note_owed_card_kill but the DIE was not banked at all, so
        # the AI's spent die sat in Current Action for ever.
        if at in ('army_attack', 'play_event', 'draw_card',
                  'discard_table_card') \
                and didx is not None and int(didx) >= 0:
            import emitter
            # WAVE-J slot cleanup: a PLAYED EVENT CARD dies WITH its
            # die at chain close (corpus E1 group-kill shape) - bank
            # the card pid alongside so the flush can render ONE
            # group kill.  The die-only flush left every played card
            # stranded in the Current-Card slot (wjg1/wjg2 live).
            card_pid = None
            card_cid = None
            if at == 'discard_table_card':
                # ONE GROUP KILL (corpus 2023-01-20 L591): the die and
                # the table card die together.  The pid comes from the
                # seat we WATCHED, resolved against the client dump -
                # never a static id table.
                from .driver import table_card_key as _tck
                _p = getattr(self.driver, 'table_seat_pid', None)
                card_pid = (_p(_tck((getattr(act, 'params', {}) or {})
                                    .get('card_name') or ''))
                            if callable(_p) else None)
                card_cid = (getattr(act, 'params', {})
                            or {}).get('card_id')
            if at == 'play_event':
                card_cid = (getattr(act, 'params', {})
                            or {}).get('card_id')
                if card_cid is not None:
                    card_pid = self.driver.view.card_pid_map.get(
                        card_cid)
                if card_pid is None:
                    self.notes.append(
                        'play_event owed kill: no client pid for the '
                        'played card (%r) - the die kill will go out '
                        'alone; the card slot may strand'
                        % (str(card_cid)[:48],))
            ent = (emitter.canon_side(dp['side']), int(didx),
                   card_pid, card_cid)
            if all(e[:2] != ent[:2]
                   for e in self.driver.owed_die_kills):
                self.driver.owed_die_kills.append(ent)
                self._step_owed = (getattr(self, '_step_owed', None)
                                   or []) + [('die', ent)]

    def _note_owed_card_kill(self, dp, act):
        """WAVE-K (R4 / A-07 residue; wkg1 LIVE: pid 183 'The Shadow is
        Moving' stranded in SA Card Play at the turn-3 rollover): the
        AI's COMMITTED COMBAT CARD has no die to die with - the client's
        own '!CALL play_card' moved it Hand -> Current SA/FP Card and
        nothing ever killed it.  Corpus grammar (2023-01-02 L713/L736):
        the owner kills the card 'in Current Action' once its round is
        over.  Bank (side, pid, cid) at EMISSION; the flush fires when
        the combat pending has CLOSED (engine statement) and the chain
        is closed, at the same owed-kill flush site.  A table card used
        as a combat card (1.149b Balrog, from_table) is NOT banked - the
        table lane (_table_mopup) owns its death in its Board slot."""
        at = getattr(act, 'action_type', None)
        params = getattr(act, 'params', {}) or {}
        if at == 'discard_table_card':
            # the DIE lane owns this one: _note_owed_die_kill banks the
            # card pid alongside the die so the flush renders the ONE
            # group kill the corpus shows.  Banking it here as well
            # would kill the card twice (and the first cut resolved its
            # pid from a static table that named the WRONG card).
            return
        if at != 'combat_card_play':
            return
        if params.get('from_table'):
            return
        cid = params.get('card_id')
        pid = (self.driver.view.card_pid_map.get(cid)
               if cid is not None else None)
        if pid is None:
            # the render already HELD on a missing pid (no emission went
            # out); stay loud either way
            self.notes.append('combat card owed kill: no client pid for '
                              '%r - slot may strand' % (str(cid)[:48],))
            return
        import emitter
        ent = (emitter.canon_side(dp['side']), str(pid), cid)
        if ent not in self.driver.owed_card_kills:
            self.driver.owed_card_kills.append(ent)
            self._step_owed = (getattr(self, '_step_owed', None)
                               or []) + [('card', ent)]

    def _flush_owed_die_kills(self):
        """Flush owed die kills once the chain is CLOSED: no open AI
        step, no engine pending, nothing deferred.  (WAVE-H: nothing
        called render_die_kill - the battle/play die stranded in
        Current Action forever and, worse, its unclosed segmenter
        window swallowed the human's next records.)"""
        drv = self.driver
        owed_cards = getattr(drv, 'owed_card_kills', None) or []
        if (not drv.owed_die_kills and not owed_cards)                 or self._open_step is not None:
            return []
        if drv.deferred is not None or drv.pending_buf is not None \
                or drv.chat_deferred is not None:
            return []
        import ai
        if ai.describe_pending(drv.es) is not None:
            return []
        # an OPEN BATTLE is an unclosed chain even with no deciding
        # side (K1 reveal phase: hidden commits awaiting the client's
        # reveal / a refused reveal re-opening a window) - the battle
        # die stays staged until the engine closes pending_combat
        if isinstance(getattr(drv.es, 'pending_combat', None), dict):
            return []
        out = []
        still = []
        for ent in drv.owed_die_kills:
            # entries: (side, die_idx[, card_pid[, cid]]) - WAVE-J:
            # the played card dies WITH the die (one E1 group kill),
            # clearing the Current-Card slot.  EXCEPTION: a card the
            # engine holds in persistent_cards STAYS on the table -
            # it seats into a Board slot instead (donor table-card
            # wave grammar) and dies in its slot when the effect
            # expires (_table_mopup).
            side_name, die_idx = ent[0], ent[1]
            card_pid = ent[2] if len(ent) > 2 else None
            cid = ent[3] if len(ent) > 3 else None
            seat_line = None
            if card_pid is not None and cid is not None \
                    and self._cid_on_table(cid):
                seat_line = self._table_seat_line(side_name,
                                                  card_pid, cid)
                card_pid = None
            em = emission.render_die_kill(
                drv.view, self.nick, side_name, die_idx,
                extra_pids=([card_pid] if card_pid else ()))
            if em.hold:
                msg = 'owed die kill HOLD: %s' % em.hold
                if not self.notes or self.notes[-1] != msg:
                    self.notes.append(msg)
                still.append(ent)
            else:
                if seat_line:
                    out.append(seat_line)
                out.extend(em.lines)
        drv.owed_die_kills = still
        # WAVE-K (R4): the AI's committed combat card dies in the
        # Current-Card slot once the COMBAT PENDING HAS CLOSED (the
        # engine's own statement that the battle is over) - corpus form
        # 'kills <title> in Current Action.' (kill is by pid; the die
        # Used zone names the corpus zone, as the E1 group kill does).
        if owed_cards and getattr(drv.es, 'pending_combat', None) is None:
            from emitter import render
            import emitter
            for side_name, pid, cid in owed_cards:
                out.append(render('kill', {
                    'nick': self.nick, 'pieces': [str(pid)],
                    'to': emitter.DICE_USED}, local=False))
                self.notes.append('combat card %s (pid %s) killed in '
                                  'Current Action at combat close'
                                  % (str(cid)[:40], pid))
            drv.owed_card_kills = []
        return out

    # ---- WAVE-J table-card lane (donor _table_card_lines, bounded) --
    _TABLE_SLOTS_FP = ('Game.2029', 'Game.2030', 'Game.2031',
                       'Game.2032', 'Game.2033', 'Game.2034')

    def _cid_on_table(self, cid):
        pc = getattr(self.driver.es, 'persistent_cards', None)
        if pc is None:
            return False
        for cards in (getattr(pc, 'fp_cards', []) or [],
                      getattr(pc, 'sh_cards', []) or []):
            if any(getattr(c, 'card_id', None) == cid for c in cards):
                return True
        return False

    def _table_seat_line(self, side_name, card_pid, cid):
        """Drag a just-persistent card from the Current-Card slot to a
        free Board slot (it STAYS while the effect lasts)."""
        from emitter import render
        book = getattr(self, '_table_book', None)
        if book is None:
            book = self._table_book = {}
        used = {b[0] for b in book.values()}
        prefs = (self._TABLE_SLOTS_FP
                 if side_name == 'FREE_PEOPLES'
                 else tuple(reversed(self._TABLE_SLOTS_FP)))
        slot = next((s for s in prefs if s not in used), prefs[0])
        book[str(card_pid)] = (slot, cid)
        frm = ('Game.2037' if side_name == 'FREE_PEOPLES'
               else 'Game.2036')
        self.notes.append('table card seated: pid %s -> %s'
                          % (card_pid, slot))
        return render('move', {'nick': self.nick,
                               'pieces': [str(card_pid)],
                               'frm': frm, 'to': slot}, local=False)

    def _table_mopup(self):
        """Expired table cards die IN THEIR SLOT (corpus grammar,
        USER 2026-08-14 table-card wave: 'kills <title> in Board1.')."""
        from emitter import render
        out = []
        book = getattr(self, '_table_book', None) or {}
        for pid in list(book):
            slot, cid = book[pid]
            if not self._cid_on_table(cid):
                del book[pid]
                out.append(render('kill', {
                    'nick': self.nick, 'pieces': [pid],
                    'to': slot}, local=False))
                self.notes.append('table card expired: pid %s '
                                  'killed in %s' % (pid, slot))
        # SEATS WATCHED IN THE FEED - either player's (USER 2026-08-20:
        # "you can see where the human player places the table card, so
        # just keep track of that location and kill it there").
        # _table_book above only ever held the AI's OWN seats, so a card
        # the HUMAN played was never removed from their board when the
        # engine discarded it (measured: 14/14 AI-seated vs 0/14
        # human-seated).  driver.table_seats records the slot we watched
        # the piece go into; the card is matched by NAME because the
        # human's card has no commitment-decoded cid.
        seats = getattr(self.driver, 'table_seats', None)
        if seats:
            booked = {p for p in book}
            for key in list(seats):
                ent = seats[key]
                slot = ent.get('slot')
                armed = ent.get('armed')
                pid = self.driver.table_seat_pid(key)
                if pid is None or pid in booked:
                    # no pid yet (the dump has not placed it) or the AI
                    # lane owns it - never guess a piece
                    continue
                on_table = self._name_on_table(key)
                if not armed:
                    # ARM ONLY (never mop): the engine must be seen
                    # holding this card first.  Without this the seat
                    # was moppable the instant the piece landed, so a
                    # card the engine had not registered yet was killed
                    # on the next tick - USER 2026-08-20, Tom Bombadil
                    # "immediately killed ... clearly not a learned
                    # action".
                    if on_table:
                        ent['armed'] = True
                    continue
                if not on_table:
                    del seats[key]
                    out.append(render('kill', {
                        'nick': self.nick, 'pieces': [pid],
                        'to': slot}, local=False))
                    self.notes.append(
                        'table card expired: pid %s (%s) killed in %s'
                        % (pid, key, slot))
        return out

    def _name_on_table(self, key):
        """Is a play-on-table card with this normalised name still on
        the engine's table?  (The human's table cards carry no
        commitment-decoded cid, so the feed's title is the only handle
        we have on them.)"""
        from .driver import table_card_key
        pc = getattr(self.driver.es, 'persistent_cards', None)
        if pc is None:
            return False
        for cards in (getattr(pc, 'fp_cards', []) or [],
                      getattr(pc, 'sh_cards', []) or []):
            for c in cards:
                nm = getattr(c, 'card_name', None) or ''
                if table_card_key(nm) == key:
                    return True
        return False

    # ------------------------------------------------------------------
    def status_menu(self):
        """The status_extra payload (see module comment above the
        helpers): the HUMAN side's current legal menu summary + dice
        bindings + driver progress counters.  Menu published only when
        the human is to act: a pending decision the engine assigns to
        the human, or the open action phase (plain-action alternation
        is not engine-wired in this driver yet - Phase-3 item 4 - so
        the action phase publishes the human menu and SAYS so via
        'to_act').  Errors return loudly in the payload, never
        silently vanish (the transport's hook guard would eat them)."""
        try:
            import ai
            import emitter
            import game_state as gs
            drv = self.driver
            es = drv.es
            with self.rand.pin():
                dp = ai.describe_pending(es, self.human_side)
            menu = None
            pend = None
            to_act = None
            if dp is not None:
                pend = str(dp['kind'])
                to_act = dp['side'].name
                if dp['side'] is self.human_side:
                    menu = dp['actions']
            elif drv.phase == 'actions':
                to_act = self.human_side.name
                with self.rand.pin():
                    menu = ai.get_legal_actions(es, self.human_side)
            # WAVE-J (the wjg1 alternation deadlock): PUBLISH the
            # plain-action alternation from the ONE engine answerer
            # (ai._next_acting_side over the driver's last_actor
            # bookkeeping — the same computation ai_action_dp uses).
            # The harness's old our_turn heuristic re-derived this
            # from last_commit prefixes and wedged after a human
            # PENDING commit (last_commit='window': the leg waited
            # for an 'ai:' commit while the engine said FP acts —
            # both sides idle till the wall clock).  None = unknown
            # (no actor yet this turn / not the action phase).
            alternation = None
            if drv.phase == 'actions' and pend is None \
                    and drv.last_actor is not None:
                try:
                    _nx = ai._next_acting_side(es, drv.last_actor)
                    alternation = (_nx.name if _nx is not None
                                   else None)
                except Exception:
                    alternation = None
            out = {'side': self.human_side.name,
                   'alternation': alternation,
                   'phase': str(drv.phase),
                   'turn': int(getattr(es, 'turn', 0)),
                   'pending': pend, 'to_act': to_act,
                   'besieged': sorted(str(b) for b in
                                      (getattr(es,
                                               'besieged_strongholds',
                                               set()) or set()))}
            # WAVE-H (combat leg): the OPEN battle's public facts + the
            # type-swap ledger, so a harness playing the human seat can
            # render casualty gestures from the CURRENT position
            # (emitter's casualty planner reads pending_combat and
            # piece_types; battle state is public - no hidden info)
            pc = getattr(es, 'pending_combat', None)
            if isinstance(pc, dict):
                out['combat'] = {
                    k: _json_safe(pc.get(k)) for k in (
                        'phase', 'attacker_region', 'defender_region',
                        'attacker_side', 'defender_side',
                        'deciding_side', 'alloc_stage', 'atk_hits',
                        'def_hits', 'is_siege', 'rounds_fought')}
            # WAVE-J: the open HUNT's public facts (damage/phase) —
            # the harness's hunt-response gestures read the engine's
            # own numbers (corruption target = corruption + damage,
            # never a guessed default)
            ph = getattr(es, 'pending_hunt', None)
            if isinstance(ph, dict):
                out['hunt'] = {k: _json_safe(ph.get(k))
                               for k in ('phase', 'damage',
                                         'companion', 'level',
                                         'casualty_taken')}
            dt = getattr(drv.view, 'dynamic_types', None)
            if dt:
                out['dynamic_types'] = {str(k): _json_safe(v)
                                        for k, v in dt.items()}
            # WAVE-J (the wirF wedge): per-option RESOLVED ROWS for the
            # human's casualty menu — the ENGINE's own allocation
            # arithmetic (ai._apply_allocation on a clone, diffed by
            # emission.removal_rows).  The harness gestures EXACTLY one
            # published option's rows; nobody recomputes hits-to-pieces
            # (an elite absorbs 2 hits — the class that wedged wirF).
            cas_rows = None
            if pend == 'combat:casualty_allocation' \
                    and menu is not None and isinstance(pc, dict) \
                    and dp is not None \
                    and dp['side'] is self.human_side:
                from . import emission as _em
                _stage = pc.get('alloc_stage')
                if _stage == 'attacker':
                    _c_reg = pc.get('attacker_region')
                    _c_sd = pc.get('attacker_side')
                    _c_hits = int(pc.get('atk_hits', 0) or 0)
                else:
                    _c_reg = pc.get('defender_region')
                    _c_sd = pc.get('defender_side')
                    _c_hits = int(pc.get('def_hits', 0) or 0)
                _hu_tag = ('fp' if self.human_side is
                           gs.Side.FREE_PEOPLES else 'shadow')
                cas_rows = {}
                if _c_sd is self.human_side:
                    for a in menu:
                        if str(a.action_type) != 'casualty_allocate':
                            continue
                        _k = int(a.params.get('count', 0) or 0)
                        if _k in cas_rows:
                            continue
                        _cl = ai.clone_state(es)
                        ai._apply_allocation(_cl, _c_reg, _c_sd,
                                             _c_hits, _k)
                        _rows = _em.removal_rows(es, _cl, _hu_tag)
                        _ser = []
                        for _tk, _slot in _rows['units'].items():
                            for _r, _n in _slot['minus']:
                                _ser.append({'tkey': str(_tk),
                                             'region': str(_r),
                                             'd': -int(_n)})
                            for _r, _n in _slot['plus']:
                                _ser.append({'tkey': str(_tk),
                                             'region': str(_r),
                                             'd': int(_n)})
                        cas_rows[_k] = sorted(
                            _ser, key=lambda x: (x['tkey'],
                                                 x['region'],
                                                 x['d']))
            if menu is not None:
                acts = []
                by_type = {}
                for i, a in enumerate(menu):
                    at = str(a.action_type)
                    by_type[at] = by_type.get(at, 0) + 1
                    if at != 'army_move' or by_type[at] <= _MENU_CAP:
                        entry = {
                            'i': i, 'action_type': at,
                            'die_idx': (int(a.die_idx)
                                        if a.die_idx is not None
                                        else None),
                            'params': _menu_params_safe(a.params)}
                        if cas_rows is not None \
                                and at == 'casualty_allocate':
                            entry['rows'] = cas_rows.get(
                                int(a.params.get('count', 0) or 0))
                        acts.append(entry)
                out['n_actions'] = len(menu)
                out['by_type'] = by_type
                out['actions'] = acts
                out['truncated'] = len(menu) > len(acts)
            dice = []
            for i, d in enumerate(es.dice.get(self.human_side, [])):
                try:
                    pid = emitter.die_piece_for(
                        drv.view, self.human_side.name, i)
                except LookupError:
                    pid = None
                dice.append({'idx': i,
                             'face': (d.current.name
                                      if d.current is not None
                                      else None),
                             'used': bool(d.used), 'pid': pid})
            out['dice'] = dice
            # WAVE-J: the FELLOWSHIP block (engine statements the
            # harness's fellowship gestures render from).  FP-human
            # only: the true pawn region is FP-hidden information and
            # must never ride the status file for a SHADOW human.
            if self.human_side is gs.Side.FREE_PEOPLES:
                fs = getattr(es, 'fellowship', None)
                if fs is not None:
                    from board import sanitize as _san
                    out['fellowship'] = {
                        'progress': int(getattr(fs, 'progress', 0)
                                        or 0),
                        'corruption': int(getattr(fs, 'corruption',
                                                  0) or 0),
                        'hidden': bool(getattr(fs, 'hidden', True)),
                        'region': _san(str(
                            getattr(getattr(fs, 'region', None),
                                    'key', None)
                            or getattr(fs, 'region', '') or '')),
                        'mordor': bool(getattr(fs, 'mordor_entered',
                                               False)),
                        'mordor_step': int(getattr(fs, 'mordor_step',
                                                   0) or 0),
                        'guide': _json_safe(getattr(
                            getattr(fs, 'guide', None), 'name',
                            None)),
                        'companions': sorted(
                            str(getattr(c, 'name', c))
                            for c in (getattr(fs, 'companions', ())
                                      or ()))}
            # WAVE-J (mirror path): the OPEN echo-ledger entries — the
            # engine's statement of human-side removals a committed
            # apply resolved with no gesture yet.  The harness kills
            # EXACTLY these (never recomputed from hit chat).  The
            # AI's own forced rows share the ledger (claimed by their
            # self-echo one cycle later) but are never the human's to
            # gesture: only the human side's rows are published.
            hu_tag = ('fp' if self.human_side is gs.Side.FREE_PEOPLES
                      else 'shadow')
            out['echo_open'] = [
                {str(k): _json_safe(v) for k, v in e.items()}
                for e in drv.echo_ledger if e.get('side') == hu_tag]
            # n_commits = the MONOTONIC commit epoch (the ledger list
            # is a 100-entry undo ring; its length is not an epoch)
            prog = {'n_commits': int(drv.n_commits_total),
                    'last_commit': (drv.commits[-1][2]
                                    if drv.commits else None),
                    'n_alarms': len(drv.alarms),
                    'last_alarm': (str(drv.alarms[-1])[:200]
                                   if drv.alarms else None),
                    'n_holds': len(drv.holds),
                    'last_hold': (str(getattr(drv.holds[-1][1], 'note',
                                              drv.holds[-1][1]))[:200]
                                  if drv.holds else None),
                    'n_uncommitted': len(drv.uncommitted),
                    # WAVE-J: wedge-time visible (coordinator item 3)
                    'stalled_seconds': round(
                        getattr(self, '_stalled', 0.0), 1),
                    # WAVE-J (A-09): the game-end verdict, published
                    # so the harness/leg can detect the END (the old
                    # 'game over:' chat is whitelist-stripped)
                    'game_over': ({str(k): _json_safe(v)
                                   for k, v in drv.game_over.items()}
                                  if drv.game_over else None),
                    # WAVE-J: the SERVER's own statement that the AI
                    # owns an open step / staged die — replaces the
                    # harness's feed-scanning ai_die_open heuristic
                    # (rule 3: no consumer derives state from a log
                    # scan when the owner can answer)
                    'ai_open': bool(
                        self._open_step is not None
                        or self._step_queue
                        or getattr(drv, 'owed_die_kills', None)),
                    'open_step': (self._open_step[0]
                                  if self._open_step else None),
                    'owed_die_kills': len(
                        getattr(drv, 'owed_die_kills', ()) or ()),
                    # PHASE3-D ported mechanisms, visible to the legs
                    'deferred': drv.deferred is not None,
                    'pending_buf': (len(drv.pending_buf.records)
                                    if drv.pending_buf is not None
                                    else 0),
                    'n_findings': len(drv.findings),
                    # WAVE-I: the newest census rows, visible to the
                    # legs (kind + 120 chars — the leg journals quote
                    # them as evidence; identities never ride findings)
                    'last_findings': [
                        (str(k), str(d)[:120])
                        for k, d in drv.findings[-3:]],
                    'window_stats': dict(drv.window_stats)}
            # 'sims' is what the panel echoes back as its acknowledgement
            # ("thinking (N sims)"); nothing published it before, so the
            # slider looked ignored no matter what it was set to.
            import ai as _ai_pub
            return {'human_menu': out, 'progress': prog,
                    'sims': int(self.sims),
                    'enforce_human': bool(getattr(self.driver,
                                                  'enforce_human', True)),
                    'max_children': int(_ai_pub.MCTS_MAX_CHILDREN),
                    'chance_k': int(_ai_pub.CHANCE_K_MAX),
                    'scenario': self._scenario_status()}
        except Exception as e:
            return {'human_menu': {'error': repr(e)[:200]},
                    'scenario': self._scenario_status()}


def build(client_dir, side_str, sims=300, net_tag='auto', seed=None,
          load_nets=True, scenario=None):
    """The production wiring, factored out of main() so the composed
    transport seam is testable (test_live_server.test_transport_seam
    reproduced two seam defects main()'s wiring shipped with: the
    FileBridge owner_needle omission and the Request-namedtuple
    handle() mismatch)."""
    import game_state as gs
    from .transport import FileBridge
    side = (gs.Side.FREE_PEOPLES if side_str == 'FP'
            else gs.Side.SHADOW)
    srv = LiveServer(client_dir, side, net_tag=net_tag, sims=sims,
                     seed=seed, load_nets=load_nets, scenario=scenario)
    # owner_needle: the substring that PROVES a pid on the lock file
    # is a live server of THIS stack, matched against its command
    # line (client fact 18: never bare pid-exists).
    # status_extra: the Phase-3 human-menu publication (the engine
    # answering "what is legal" across the process boundary - I1).
    # control_cb (WAVE-J, ledger C-11): the panel's non-pause keys
    # (sims slider, manual-edit hold, panel resync) reach the server -
    # transport surfaced them since LIFT-C but build() never wired the
    # callback, so every panel lever except pause was silently ignored
    bridge = FileBridge(client_dir, owner_needle='live_server',
                        status_extra=srv.status_menu,
                        control_cb=srv.on_control)
    return srv, bridge


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--client-dir', required=True)
    ap.add_argument('--side', choices=('FP', 'SA'), required=True)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--net-tag', default='auto')
    ap.add_argument('--seed', type=int, default=None)
    # K-S1: a saved position (lab\scenarios\<name>) replayed onto the
    # client and verified before the AI owns its seat
    ap.add_argument('--scenario', default=None)
    a = ap.parse_args(argv)
    scenario = None
    if a.scenario:
        from .scenario import Scenario
        scenario = Scenario(a.scenario)
        print('  [scenario] %s: human %s, AI %s, prefix %d line(s)'
              % (scenario.name, scenario.human_side, scenario.ai_side,
                 scenario.cut))
    srv, bridge = build(a.client_dir, a.side, sims=a.sims,
                        net_tag=a.net_tag, seed=a.seed,
                        scenario=scenario)
    # serve_handler: RESP lines + the waiting_on blocker text (WAVE-J:
    # the panel's 'what am I waiting on' display - the old bare handle
    # left waiting_on EMPTY every tick)
    bridge.serve(srv.serve_handler)


if __name__ == '__main__':
    main()
