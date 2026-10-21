"""live_server â€” the sided-play entry point the client launches.

Wires the whole stack:  FileBridge transport  ->  LiveDriver (feed
router)  ->  nets (engine_play.load_nets + ai.mcts_action_with_policy)
->  emission (engine events -> gestures)  ->  ClientRandomness
(!CALL directives, deterministic retries)  ->  ChatOut (the single
whitelisted chat exit).

THE STEP PROTOCOL (PROTOCOL.md, deterministic-retry contract):
  * every AI decision applies on a CLONE (ai.apply_action) â€” a
    NeedClientCall leaves the driver's engine state UNTOUCHED;
  * the pick is journal-pinned (randomness.journal_pick), so a step
    retried across ticks repeats the previous attempt exactly;
  * emission goes out ONCE per step (the play_event staging must
    precede the effect's randomness â€” donor :15874-15900); directives
    go out once per distinct pending call;
  * the step commits (state adopted + feed prefixes dropped) only
    after a successful apply; the unconsumed() honesty gate runs
    first â€” excess client randomness is an alarm, never silence.

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
# (the published human_menu is UNCAPPED - USER ruling 2026-08-24
# "there should be no caps", extending the 2026-08-22
# no-legal-surface-caps ruling to the publication: every action of
# every type is always published whole)


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


# normalized tile identity: ONE implementation (randomness.tile_face)
_tile_face = rnd_mod.tile_face


def _menu_params_safe(params):
    """Param dict for DIAGNOSTIC surfaces (alarm/note text): card
    identities stay redacted there - an alarm can fire mid-decision
    and must never name the AI's own hidden card."""
    out = {}
    for k, v in (params or {}).items():
        kl = str(k).lower()
        if 'card' in kl or kl == 'cid':
            out[str(k)] = '<hidden>'
        else:
            out[str(k)] = _json_safe(v)
    return out


def _menu_params_pub(params):
    """Param dict for the published HUMAN menu (USER 2026-08-25: the
    engine sees the human hand, so the human's own card plays ride
    the status file with their identities - the menu is the human's
    own surface).  The old rule-8 redaction survives only on the
    diagnostic path above."""
    return {str(k): _json_safe(v) for k, v in (params or {}).items()}


class LiveServer(object):
    def __init__(self, client_dir, ai_side, net_tag='auto', sims=300,
                 seed=None, load_nets=True, scenario=None,
                 ai_password='ai', human_password=None, script=None,
                 script_ai_nick=None):
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
                            ('rewind_turn', '_rewind_epoch'),
                            ('resume_rewind', '_resume_rewind_epoch'),
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
        # the AiFeed meta sidecar; absent file = channel off (old jars
        # / suites) and the heuristic lanes stay - the absence is
        # NOTED once live
        from .sidecar import SidecarReader
        self.driver.sidecar = SidecarReader(feed_path + '.meta.jsonl')
        # BRJAVA-3 / BRLANE-13: the candidate jar's per-line ack file
        # (absent under the shipped jar = the feed-echo deadline lane)
        self.driver.ack_path = os.path.join(os.path.dirname(feed_path),
                                            'ai_inject_ack.jsonl')
        # USER 2026-08-22: legality enforcement for the human is the
        # panel's 'Enforce legal moves for HUMAN player (beta)' box -
        # absent from the control file = OFF (the board is taken as
        # given after each human action)
        self.driver.enforce_human = bool(
            getattr(self, '_enforce_human_boot', True))
        # sanctioned raw import #1: the AI's OWN draw identities via
        # the commitment decode (AiBoot claims the AI hand with 'ai')
        # CORPUS REPLAY (USER 2026-08-28, lab/corpus_replay/DESIGN.md):
        # the AI claims a RECORDED player's hand with that player's
        # password and the human seat's claim is the recorded one, so
        # both decode with the recorded passwords; --script pins every
        # AI decision to the recorded player's rendered lines.
        self.cards = card_identity.CardIdentity(
            self.ai_side.name, password=ai_password,
            human_password=human_password)
        self.script = None
        if script:
            try:
                from . import corpus_script as _cs
            except ImportError:
                import corpus_script as _cs
            self.script = _cs.ScriptChooser(script, script_ai_nick)
            print('  [script] AI decisions pinned to %s (%s): %d '
                  'decision line(s)' % (script, script_ai_nick,
                                        len(self.script.ai_lines)))
        self.fp_net = self.sh_net = self.own_net = None
        self.own_search = None
        if load_nets:
            from .own_model import load_own_net, OwnModelSearch
            self.own_net = load_own_net(net_tag, self.ai_side)
            if self.ai_side is gs.Side.FREE_PEOPLES:
                self.fp_net = self.own_net
            else:
                self.sh_net = self.own_net
            self.own_search = OwnModelSearch(self.ai_side, self.own_net, seed=seed,
                journal=os.path.join(client_dir, 'own_search_calls.jsonl'))
            print('  [search] %s; %s model only; %d simulations' %
                  (self.own_search.backend, self.ai_side.name, self.sims), flush=True)
        self.notes = []             # panel-only chatter (never chat)
        self._em_sent = False       # this step's emission went out
        self._dirs_sent = set()     # pending-call identities sent
        # '!CALL' texts THIS STEP's emission already ordered - the
        # directive builder must not order them again (a staged draw
        # !CALL plus a directive for the same starved funnel would
        # draw TWO cards)
        self._step_calls = {}       # directive text -> feed gen at order
        # ops of a PRE-APPLY emission, kept across the step's
        # cross-tick retries for the post-commit sweep's subtraction
        self._step_expressed = []
        self._step_dead_men_arrival = None
        self._step_casualty_prefix = None
        self._step_owed = []        # owed kills THIS step banked
        # K1: the AI's own combat card is CHOSEN (hidden) on the client
        # until the round's reveal directive; one reveal per round
        self._ai_cc_chosen = False
        self._reveal_round = None
        self._open_step = None      # a starved AI step retried per tick
        self._step_starved = 0      # ticks the open step has starved
        self._step_parked = False   # BRLANE-3: parked behind the human
        self._step_park_n = 0       # parks of the current open step
        self._raw_seen = 0          # BRJAVA-1: raw log lines scanned
        self._step_queue = []       # AI steps arriving while one is open
        # window/pending funnel asks already ordered, keyed by
        # (commit epoch, call identity): within one epoch a starving
        # ask is ordered ONCE (a re-sent roll the client DID execute
        # would roll twice); a NEW commit opens a new epoch so a later
        # identical ask (next turn's hunt) is not falsely deduped â€”
        # the AI-step _dirs_sent set stays step-scoped and separate.
        self._window_asks_sent = set()
        # BL-122: every human roll ask carries a NONCE in its published
        # text ('roll 2 dice #7') - same-text asks recur legitimately
        # round after round, so text is never identity; the seat
        # answers each nonce once.  BL-130: the entry carries the
        # demanded COUNT, so a rendered roll of the wrong width is a
        # named mismatch, never silent quarantine noise.  Live
        # (unanswered) entries publish in the status JSON ('asks');
        # closure = result rendered since feed_mark, or the ask's
        # commit epoch passed.
        self._ask_nonce_seq = 0
        self._live_asks = {}    # nonce -> {kind, need, text, feed_mark, epoch}

    # ------------------------------------------------------------------
    def handle(self, request):
        """One transport cycle - the BL-6 chokepoint wrapper: EVERY
        outgoing line (step-runner emissions included, which never
        pass _dispatch) registers its political marker walks on the
        recon in-flight ledger before leaving the process.  The
        dispatch-level registration alone missed the AI's own action
        renders and the double-walk recurred in both round-2
        sessions."""
        self._cycle_step_origin = None
        self._cycle_step_ref = None
        try:
            out = self._handle_inner(request)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            from .cycle_recovery import failed as recover_failed_cycle
            recovered = recover_failed_cycle(self, e)
            if recovered is not None:
                from .ack_identity import bind as bind_ack_operations
                bind_ack_operations(self.driver, recovered)
                return recovered
            raise
        self._cycle_err_sig = None
        self._cycle_err_n = 0
        out = list(out or ())
        from .idle_recovery import poll as recover_idle
        out = recover_idle(self, out)
        from .turn_rewind import status as rewind_status
        if (rewind_status(self)['phase'] == 'idle'
                and not any(str(line).startswith('!CALL turn_') for line in out)):
            # The paired snapshot protocol owns the whole response. Even a
            # previously owed report changes the native feed proof while the
            # client is capturing or restoring, so flush it after release.
            from .cycle_recovery import settled as settle_cycle_failure
            settle_cycle_failure(self)
            from .anduril_reports import committed as committed_anduril_reports
            from .fixed_forfeit_reports import committed as committed_fixed_forfeit_reports
            out = committed_fixed_forfeit_reports(self) + committed_anduril_reports(self) + out
        try:
            for _ln6 in (out or ()):
                self._note_political_emission(str(_ln6))
                self._note_guide_emission(str(_ln6))
        except Exception:
            pass
        # N6: ONE identical chat line per reply (several producers may
        # voice the same demand in one tick)
        try:
            _seen6 = set()
            _ded6 = []
            for _it6 in (out or ()):
                if isinstance(_it6, tuple) and len(_it6) == 3 \
                        and _it6[0] == 'chat':
                    _k6 = (_it6[1], str(_it6[2]))
                    if _k6 in _seen6:
                        continue
                    _seen6.add(_k6)
                _ded6.append(_it6)
            out = _ded6
        except Exception:
            pass
        from .ack_identity import bind as bind_ack_operations
        if not out and not getattr(self, '_cycle_failure', None):
            from .turn_rewind import maybe_capture
            out = maybe_capture(self) or out
        bind_ack_operations(self.driver, out)
        return out

    def _perform_reset(self):
        """Retry retained work or adopt a measured complete native board."""
        self._reset_requested = False
        self._recovery_ran_this_cycle = True
        out = []
        self._resync_quiet = True
        from .cycle_recovery import retained_retry, retry_started
        if retained_retry(self):
            # The failed step proved that authoritative state and commitments
            # were unchanged. Preserve its journal/requests; do not adopt the
            # board or decline an optional effect to make the error disappear.
            from .pending_roll_wait import retire_obsolete_roll
            for item in retire_obsolete_roll(self):
                out.extend(self._dispatch(item))
            self.rand.sticky_halt = None
            retry_started(self)
            self._panel_recovery_result = {
                'epoch': max(float(getattr(self, '_reset_epoch', 0)),
                             float(getattr(self, '_resync_epoch', 0))),
                'complete': False, 'summary': 'Retrying the retained AI step; committed results and requests kept.',
                'waiting_for': 'retained_step', 'commits': self.driver.n_commits_total,
            }
            return out
        if getattr(self, '_cycle_failure', None):
            # This state no longer matches the failed step's proven boundary.
            # A board adoption cannot prove that partial engine mutations were
            # rolled back; use the paired native/engine checkpoint instead.
            self._cycle_failure['rewind_attempted'] = True
            from .turn_rewind import recover
            return recover(self)
        _items = []
        _reset_failed = False
        try:
            _items, _rep = self.driver.reset_and_resync()
            for it in _items:
                out.extend(self._dispatch(it))
        except DriverHalt:
            raise
        except Exception as _re9:
            _reset_failed = True
            _rep = ['reset failed: %r' % (_re9,)]
            # Adoption can succeed before later reset bookkeeping fails.
            # The whole recovery is still incomplete; preserve queued AI
            # work and report failure rather than clearing it as success.
            self.driver._last_resync_adoption = dict(
                getattr(self.driver, '_last_resync_adoption', {}),
                complete=False)
        from .recovery_status import record as record_recovery_status
        record_recovery_status(self, _items, _rep)
        _combat_retry = (not _reset_failed and
                         getattr(self.driver, '_last_reset_preserved_combat', False))
        if _combat_retry:
            from .pending_roll_wait import retire_obsolete_roll
            for item in retire_obsolete_roll(self):
                out.extend(self._dispatch(item))
            # Keep the current action journal, observed outcomes, request
            # nonces and emitted-piece ledger. Reissuing their native calls
            # could reroll dice or charge a card/die a second time.
            self._step_starved = 0
            pending = self.rand.pending()
            if pending is not None:
                _rep.append('still waiting for the existing %s request; known '
                            'results were retained and no random call was repeated'
                            % pending.kind)
            _rep.append('AI step and outstanding physical acknowledgements retained')
        elif getattr(self.driver, '_last_resync_adoption', {}).get('complete'):
            out.extend(self._abandon_step('panel reset'))
            self._step_queue = []
            self._window_asks_sent = set()
            # The sequence keeps counting: never reuse a request nonce.
            self._live_asks = {}
            _rep.append('AI step state cleared')
        else:
            _rep.append('resync incomplete; AI step and outstanding requests retained')
        # a LIVE HALT is sticky until the operator resets (K3: the
        # reset is the one lever that clears it)
        if _combat_retry or getattr(self.driver, '_last_resync_adoption', {}).get('complete'):
            self.rand.sticky_halt = None
            from .cycle_recovery import retry_started
            retry_started(self)
        for _line in _rep:
            self.notes.append('reset: %s' % _line)
            print('  [reset] %s' % _line)
        return out

    def _take_recovery_controls(self, items):
        """Operator controls remain available before gameplay dispatch."""
        from .cycle_recovery import manual_retry, retry_skip
        keep = []
        for item in items:
            if item[0] == 'recovery_request':
                if item[1] == 'rewind':
                    self._rewind_requested = True
                    self._rewind_target = (None, None)
                elif not getattr(self, '_recovery_ran_this_cycle', False):
                    manual_retry(self)
            elif item[0] == 'skip_request' and retry_skip(self, item[1]):
                continue
            else:
                keep.append(item)
        return keep

    def _request_rewind(self):
        from .turn_rewind import request
        self._rewind_requested = False
        checkpoint_id, session = getattr(self, '_rewind_target', (None, None))
        self._rewind_target = (None, None)
        out = request(self, checkpoint_id=checkpoint_id, session=session)
        if not out:
            from .turn_rewind import status
            return [self._say('prompt', chat_mod._register(status(self)['note']))]
        return out

    def _note_guide_emission(self, line):
        """BL-119 (shgame13 M5): guide/fellowship renders stamp their
        pids on the shared in-flight ledger, so the guide-box recon
        never re-renders a walk the commit emission already sent (the
        doubled guide drags; the doubled FLIP is self-cancelling on a
        real board)."""
        try:
            s = str(line)
            if 'Guide Of The Fellowship' not in s \
                    and 'flips The Fellowship' not in s:
                return
            drv = self.driver
            import time as _t119
            rec = getattr(drv, '_recent_emit_pids', None)
            if rec is None:
                rec = drv._recent_emit_pids = {}
            if 'flips The Fellowship' in s:
                rec['247'] = _t119.time()
                return
            import log_replay as _lr119
            for p in _lr119.COMPANIONS:
                if any(nm and nm in s
                       for nm in (_lr119.names_for_pid(str(p))
                                  or ())):
                    rec[str(p)] = _t119.time()
                    return
        except Exception:
            pass

    def _new_raw_controls(self, lines):
        """Only unseen operator commands may escape a sticky gameplay halt.

        The request envelope is already removed. A changed cumulative history
        establishes a new watermark without replaying its existing controls;
        later appended commands and panel recovery remain available. Rendered
        commands already consumed by the driver own their eventual raw copies.
        """
        from collections import Counter
        import re

        def control(line):
            match = re.fullmatch(
                r'\s*<([^>]+)>\s*(skip(?:\s+decision)?[.!]?'
                r'|reset(?:\s*\+?\s*resync)?|resync|retry|resume'
                r'|rewind(?:\s+(?:turn|to\s+start\s+of\s+turn))?)[.!]?\s*',
                str(line), re.I)
            if match is None:
                return None
            kind = ('skip' if match[2].lower().startswith('skip') else
                    'resume' if match[2].lower().rstrip('.!') == 'resume' else
                    'rewind' if match[2].lower().startswith('rewind') else
                    'resync' if match[2].lower().rstrip('.!') == 'resync' else 'reset')
            return kind, match[1]

        current = tuple(lines)
        previous = getattr(self, '_raw_control_history', ())
        self._raw_control_history = current
        # Keep the old diagnostic cursors in the same envelope-free domain.
        self._skip_chat_seen = self._reset_chat_seen19 = len(current)
        if current == previous:
            return []
        if current[:len(previous)] != previous:
            self._raw_control_baseline = (
                Counter(control(line) for line in current),
                Counter(control(line) for line in
                        self.driver.feed.lines[:self.driver._feed_seq]))
            self.notes.append('raw control history changed: existing commands retired; '
                              'append a new command or use the panel for recovery')
            return []
        if not any(control(line) is not None for line in current[len(previous):]):
            return []
        consumed = Counter(control(line) for line in
                           self.driver.feed.lines[:self.driver._feed_seq])
        raw_base, feed_base = getattr(
            self, '_raw_control_baseline', (Counter(), Counter()))
        occurrences = Counter()
        fresh = []
        for index, line in enumerate(current):
            key = control(line)
            if key is None:
                continue
            occurrences[key] += 1
            if (index >= len(previous)
                    and occurrences[key] - raw_base[key] > consumed[key] - feed_base[key]):
                fresh.append(key)
        return fresh

    def _handle_inner(self, request):
        """One transport cycle.  `request` is the transport's Request
        namedtuple (production: FileBridge.serve) or a dict with
        'lines' (the offline suites); either carries the raw client
        log lines (sanctioned imports only).  Returns RESP lines."""
        # SIDE-HEADER GUARD: the client's 'Play vs AI...' click is
        # INDEPENDENT of the launcher's --side / -Dai.feed.aiside; a
        # contradicting click would blind the WRONG side's hand (every
        # #HAND line IS blinded, just the wrong side's, so the rule-8
        # check cannot catch it).  The header names the AI's side
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
        # All raw-history consumers use the same body, excluding its changing
        # per-request envelope. Validate session ownership before any control
        # can reset or skip a decision in this game.
        from . import client_protocol as _cp1
        wire_lines = (request.get('lines', ()) if isinstance(request, dict)
                      else getattr(request, 'log_lines', ())) or ()
        _seq_hdr1, lines = _cp1.split_seq_header(wire_lines)
        from .ack_identity import begin as begin_ack_response
        begin_ack_response(self.driver, _seq_hdr1)
        controls = self._new_raw_controls(lines)
        self._recovery_ran_this_cycle = False
        for kind, nick in controls:
            if nick == self.driver.nick:
                continue
            if kind == 'resume':
                from .turn_rewind import _state
                _state(self).review = False
            if kind in ('reset', 'rewind') or (kind == 'resync' and (
                    self.rand.sticky_halt or getattr(self, '_cycle_failure', None))):
                if kind == 'rewind':
                    self._rewind_requested = True
                    self._rewind_target = (None, None)
                else:
                    from .cycle_recovery import manual_retry
                    manual_retry(self)
                    self.notes.append('chat: RESET + RESYNC requested '
                                      '(clears the sticky halt)')
                from collections import Counter
                if not hasattr(self.driver, '_raw_recovery_receipts'):
                    self.driver._raw_recovery_receipts = Counter()
                self.driver._raw_recovery_receipts[(kind, nick)] += 1
        from . import turn_rewind
        if getattr(self, '_rewind_requested', False):
            return self._request_rewind()
        rewind_output = turn_rewind.poll(self)
        if rewind_output is not None:
            return rewind_output
        if turn_rewind._state(self).review:
            # Preserve the visible restored board. Queued automatic work must
            # not immediately play past the boundary the operator selected.
            if (getattr(self, '_skip_requested', False) or getattr(self, '_reset_requested', False)
                    or getattr(self, '_resync_requested', False)
                    or turn_rewind.review_control_waiting(self)):
                turn_rewind._state(self).review = False
            else:
                return []
        # a randomness LIVE HALT (deck/pool desync, journal mismatch,
        # tripwire) is STICKY: no cycle plays on past it until the
        # operator's RESET clears it.  FLA-19 (par5 g4#6): the halt
        # also killed the CHAT lane, so a CLI seat had NO recovery
        # (RESET+RESYNC was GUI-only) - a chat 'reset' / 'resync'
        # from the table now pulls the SAME panel lever (K3, one
        # implementation), and the halt text names the route.
        for _control_kind, _control_nick in controls:
            # The rendered feed is the normal single command owner. Raw chat
            # is only an escape from a sticky halt that prevents feed routing;
            # observing the same chat later in a request must not skip twice.
            if (self.rand.sticky_halt and _control_kind == 'skip'
                    and _control_nick != self.driver.nick):
                from .skip_recovery import arm as arm_skip
                arm_skip(self)
        if getattr(self, '_skip_requested', False):
            from .cycle_recovery import retry_skip
            retry_skip(self, getattr(self, '_skip_target', None))
        if self.rand.sticky_halt \
                and not getattr(self, '_skip_requested', False) \
                and not getattr(self, '_reset_requested', False):
            _hit19 = any(kind in ('reset', 'resync') and nick != self.driver.nick
                         for kind, nick in controls)
            if _hit19:
                self._reset_requested = True
                self.notes.append('chat: RESET + RESYNC requested '
                                  '(clears the sticky halt)')
            else:
                raise DriverHalt('%s  [sticky - press RESET + RESYNC '
                                 "in the panel, or say 'reset' in "
                                 'chat, to clear]'
                                 % self.rand.sticky_halt)
        # 'stalled_seconds' = time since the feed last GREW while any
        # deferred/starved/open state exists (published; the leg fails
        # a game at a hard stall budget)
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
        # design rule 8 FAIL-CLOSED: an un-blinded #HAND line means the
        # sided launch is missing -Dai.feed.aiside - refuse to play on,
        # loudly (CLC-1: the #BLIND verdict below reads gs - pinned by
        # test_audit0829_CLC)
        import game_state as gs
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
        # BRJAVA-1: the candidate jar's '#SEQ' request header is split
        # off; a client complaint in the NEW raw log lines ('<game> AI
        # ERROR ...') is a LOUD halt (the driver's feed scan is the same
        # predicate, client_protocol.halt_verdict) - the self-repair ASK
        # follows on the next cycle, never a silent continue
        self.driver._raw_request_lines = list(lines)
        _raw0 = int(getattr(self, '_raw_seen', 0) or 0)
        if _raw0 > len(lines):
            _raw0 = 0                       # the log was truncated
        # FLA-20 / FLH H2 (par5 g4#5): the watchdog's 'modal dialog
        # open: Confirm Random Discard' line is EXPECTED TELEMETRY
        # while the engine starves on the client's own WWSAT
        # card_pick - it is the confirm the jar's yes-arm should
        # answer, not a client failure (the halt here put 2 cycle
        # errors on a healthy wait).  One predicate on the driver;
        # every other complaint stays a halt.
        _scan1 = [l for l in lines[_raw0:]
                  if not self.driver._wwsat_modal_expected_line(l)]
        _halt1, _why1 = _cp1.halt_verdict(_scan1)
        self._raw_seen = len(lines)
        # FLA-19: the healthy-cycle cursor for the sticky-gate chat
        # scan (an OLD 'resync' line must never clear a later halt)
        self._reset_chat_seen19 = len(lines)
        if _halt1:
            self.driver.findings.append(('client_error_line', _why1[:160]))
            if _why1 not in self.driver.alarms:
                self.driver.alarms.append(_why1)
            self.driver._out_queue.extend(self.driver._propose_repair(
                _why1[:140], clears='manual'))
            raise DriverHalt(_why1)
        # sanctioned raw import #1: decode the AI's OWN drawn card
        # identities from the commitment tables and feed them BEFORE
        # the tick, so a retried ai_draws step consumes them this
        # cycle.  An undecodable AI card is a LOUD halt, never a guess.
        _prefix_lane = (self.scenario is not None
                        and not getattr(self.driver, 'live', True))
        for d in self.cards.harvest(lines):
            if d['side'] != self.ai_side.name:
                # USER 2026-08-25 "the engine should see the human
                # hand": the human's draw decodes too. Its observation
                # waits for the matching physical draw receipt and binds
                # as that draw's hand slot is created or committed.
                # Undecodable = LOUD alarm, placeholder kept (the
                # menu just lacks that card until it goes public).
                if d['cid'] is None:
                    self.driver.alarms.append(
                        'HUMAN draw pid %s undecodable from the '
                        'client commitments (name %r) - menu will '
                        'lack this card' % (d['pid'], d['name']))
                if _prefix_lane:
                    self._scenario_pid_cid[str(d['pid'])] = d['cid']
                    continue
                self._human_draw_ids = getattr(
                    self, '_human_draw_ids', [])
                self._human_draw_ids.append(d)
                continue
            if d['cid'] is None:
                raise DriverHalt(
                    'AI draw pid %s undecodable from the client '
                    'commitments (client name %r) - cannot bank an '
                    'unknown card' % (d['pid'], d['name']))
            if _prefix_lane:
                self._scenario_pid_cid[str(d['pid'])] = d['cid']
                continue
            self.rand.feed('card', (d['side'], d['deck'], d['cid'],
                                    d['pid']), raw_index=d['raw_index'])
            self.driver.view.card_pid_map[d['cid']] = d['pid']
            self.driver.view.note_drawn(d['pid'])
        if _prefix_lane:
            # K-S1: no AI step, no human matching, no mop-up until the
            # prefix is verified and the driver flipped to live
            output = self._scenario_prefix_cycle(lines)
            # Prefix cards were bound by the scenario's own identity lane.
            # They must not claim the first newly observed live draw.
            for model in self.driver.hand_models.values():
                model._draw_receipts = []
            return output
        # Resolve each receipt at slot creation, before subsequent mutations.
        from .human_draw_identity import settle as settle_human_draws
        settle_human_draws(self)
        out = []
        # MANUAL EDIT hold (C-11): the machine is a spectator until the
        # operator finishes and presses resync
        if getattr(self, '_manual_hold', False):
            return []
        # a panel resync epoch (C-12) runs the RECOVERY (adopt + clear
        # + reopen), draining any manual-edit backlog unmatched.  USER
        # 2026-08-24: resync means "reconcile engine and client board
        # state" and NOTHING else - it must not trigger the AI's turn;
        # the AI step machinery sits out this tick.
        if getattr(self, '_resync_requested', False):
            self._resync_requested = False
            if self.driver.combat_recovery_open():
                self._reset_requested = True
            else:
                _recovery_items = self.driver.resync_recover(drain_feed=True)
                from .recovery_status import record as record_recovery_status
                record_recovery_status(self, _recovery_items)
                for it in _recovery_items:
                    out.extend(self._dispatch(it))
                self._resync_quiet = True
                self.notes.append('resync: turn order retained; %s' %
                                  ('board reconciled' if getattr(self.driver,
                                   '_last_resync_adoption', {}).get('complete')
                                   else 'incomplete, existing action retained'))
        # SKIP DECISION (USER 2026-08-20).  Runs on the TICK, before
        # the feed is read, so the commit and its emission ride the
        # normal path.  The outcome is ANNOUNCED - the player must be
        # able to see which decision was skipped and with what.
        # Board recovery retains unfinished decisions and their AI step.
        # Clear a stale step only after the driver confirms complete adoption;
        # a retained combat retry keeps its requests and acknowledgements.
        if getattr(self, '_reset_requested', False):
            out.extend(self._perform_reset())
            if (turn_rewind.status(self)['phase'] != 'idle'
                    or self.rand.sticky_halt):
                # Reset may have started a paired physical restore. Its first
                # directive must reach the client before any more feed ingest
                # or gameplay can run. Failed recovery retains the same hold.
                return out
        n_feed0 = self.driver._feed_seq
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
        # the driver's honesty gates must know whether an AI step is
        # mid-flight (its owed randomness is never quarantined)
        self.driver.ai_step_open = self._open_step is not None
        try:
            items = self.driver.tick(lines)
        except DriverHalt as e:
            self.notes.append(str(e))
            if not getattr(self, '_skip_requested', False):
                raise
            # Skip is an explicit escape from a sticky interaction error too.
            # Adoption below still needs a complete current physical export.
            self.driver.feed.poll()
            self.driver._feed_seq = len(self.driver.feed.lines)
            items = []
        items = self._take_recovery_controls(items)
        if getattr(self, '_rewind_requested', False):
            self.driver._out_queue = items + self.driver._out_queue
            return out + self._request_rewind()
        if getattr(self, '_reset_requested', False):
            out.extend(self._perform_reset())
            if (turn_rewind.status(self)['phase'] != 'idle'
                    or self.rand.sticky_halt):
                self.driver._out_queue = items + self.driver._out_queue
                return out
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
        if (getattr(self.driver, '_sidecar_wait_seq', None) is not None
                or getattr(self.driver, '_grouped_die_identity_wait', None) is not None):
            # Metadata and rendered paragraphs are separately flushed native
            # channels. Keep the step, request nonces and queued replies while
            # the driver waits for their ordered join; status remains readable.
            from .skip_recovery import arm as arm_skip, run as run_skip
            for item in items:
                if item[0] == 'skip_request':
                    arm_skip(self, item[1])
            items = [item for item in items if item[0] != 'skip_request']
            if getattr(self, '_skip_requested', False):
                for item in run_skip(self):
                    out.extend(self._dispatch(item))
            self.driver._out_queue = items + self.driver._out_queue
            return out
        # BL-22 flush: the held 'nc' releases the moment the human's
        # own combat-card decision closed (committed, or the round
        # moved on) - the joint-reveal ordering
        if getattr(self, '_held_nc', None) is not None:
            if not self._nc_must_hold():
                out.append(self._say('nc', self._held_nc))
                self._held_nc = None
        # K1 step 4: a human commit this tick may have completed the
        # round's card commitments - the reveal directive precedes any
        # roll ask the tick's items carry
        self._append_combat_reveal_before_rolls(out)
        from .skip_recovery import arm as arm_skip
        for item in items:
            if item[0] == 'skip_request':
                arm_skip(self, item[1])
        items = [item for item in items if item[0] != 'skip_request']
        if getattr(self, '_skip_requested', False):
            from .skip_recovery import run as run_skip
            skipped = run_skip(self)
            if self._skip_replaced_tick_items:
                # Forced abandonment replaces this tick's old work. An expired
                # request must retain events belonging to the next decision.
                items, out = skipped, []
                self._resync_quiet = True
            else:
                items.extend(skipped)
        from .turn_bookkeeping import poll as poll_turn_bookkeeping
        items.extend(poll_turn_bookkeeping(self))
        if not out and not getattr(self, '_cycle_failure', None):
            # Preserve queued draw/recovery items inside the checkpoint. The
            # native capture handshake owns the cycle before they execute.
            queued = self.driver._out_queue
            self.driver._out_queue = items + queued
            checkpoint = turn_rewind.maybe_capture(self)
            if checkpoint:
                return checkpoint
            self.driver._out_queue = queued
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
                    # FLA-18: baseline the resync epoch at step OPEN
                    self._step_resync0 = float(
                        getattr(self.driver, '_resync_ts', 0) or 0)
                elif self._open_step[0] == item[0]:
                    self._open_step = item          # refresh payload
                elif all(q[0] != item[0] for q in self._step_queue):
                    self._step_queue.append(item)
                continue
            out.extend(self._dispatch(item))
        out.extend(self._flush_recovery_request())
        from .corsairs_presentation import poll as present_corsairs
        out.extend(present_corsairs(self))
        # Retry delayed identities and publish draws committed by step work.
        from .human_draw_identity import settle as settle_human_draws
        settle_human_draws(self)
        # BL-71: the witness cadence runs BEFORE the step machinery so
        # a wedged step cannot freeze the census (witnesses NOTICE,
        # the engine OWNS - read-only here)
        self._cycles = getattr(self, '_cycles', 0) + 1
        if self._cycles % 30 == 0 and self.driver.game_over is None:
            # the cadence returns the census lane's ITEMS (held notes,
            # reasoned rows, a LOG -> ASK) - routed through _dispatch
            # like every driver item (SQF-1: a raw tuple in `out`
            # breaks the transport's join and LOSES the note)
            for _wit1 in (self._witness_cadence() or []):
                out.extend(self._dispatch(_wit1))
        # Clear the previous round's slot before selecting another combat card.
        out.extend(self._flush_finished_combat_cards())
        # an AI-step DriverHalt must not DISCARD the lines the tick
        # already built (serve()'s catch drops the whole RESP, and a
        # deduped ask lost with the cycle starves forever).  The halt
        # stays loud (notes + print, re-raised only when there is
        # nothing to deliver); the step stays open and re-halts each
        # tick until fixed.
        try:
            if self.driver.game_over is not None:
                # A-09: a won game stops the AI cold - no further
                # steps, no queue drain
                self._abandon_step('game over')
                self._step_queue = []
            if getattr(self, '_resync_quiet', False):
                # USER 2026-08-24: a resync tick reconciles ONLY -
                # any open/queued AI step waits for the next tick
                # (and the next tick acts only if the AI is truly due)
                self._resync_quiet = False
            else:
                if self._open_step is not None:
                    out.extend(self._run_open_step())
                while self._open_step is None and self._step_queue:
                    self._open_step = self._step_queue.pop(0)
                    self._step_starved = 0
                    # FLA-18: baseline the resync epoch at step OPEN
                    self._step_resync0 = float(
                        getattr(self.driver, '_resync_ts', 0) or 0)
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
        self._append_combat_reveal_before_rolls(out)
        out.extend(self._owed_reveal_nag())
        out.extend(self._starved_randomness_watchdog())
        out.extend(self._flush_owed_die_kills())
        out.extend(self._tile_mopup())
        out.extend(self._table_mopup())
        # USER 2026-08-24 auto-reconcile: a committed gesture flags
        # _reconcile_due; the driver's census lane then judges the
        # board on a QUIESCENT tick (driver.quiescence_holds - the ONE
        # predicate, BRCMP design 1: open window, refusal restore, AI
        # render TTL, echo ledger of either side, owed demands, open
        # battle / decision, repair, resync) and publishes WHY when it
        # is held.  It never adopts by itself (design 3): a confirmed
        # divergence is LOGged and ASKed; engine truth still walks the
        # political markers + the AI-FP guide box on a quiescent tick.
        drv9 = self.driver
        # the HUMAN side's open echo demands (read by the BL-3 turn-over
        # announce below; the census gate reads the driver's predicate)
        _hu_echo8 = [e for e in getattr(drv9, 'echo_ledger', ())
                     if e.get('side') == ('fp' if self.human_side.name
                                          == 'FREE_PEOPLES'
                                          else 'shadow')]
        if getattr(drv9, '_reconcile_due', False) \
                and self._open_step is None:
            # BL-6: the recon runs INSIDE this tick - register the
            # batch's own political walks FIRST, or the recon re-walks
            # a marker whose line sits in `out` un-applied
            for _l6b in out:
                self._note_political_emission(str(_l6b))
            for it in drv9.reconcile_board_quiet():
                out.extend(self._dispatch(it))
        # BL-50: the held-choice announce prints only against a LIVE
        # pending
        _hcd50 = getattr(drv9, '_held_choice_due', None)
        if _hcd50 is not None \
                and getattr(self, '_held_choice_said', None) != _hcd50:
            import ai as _ai50
            try:
                _dp50 = _ai50.describe_pending(drv9.es)
            except Exception:
                _dp50 = None
            if _dp50 is not None \
                    and _dp50['side'] is self.human_side \
                    and str(_dp50.get('kind')) == str(_hcd50[1]):
                self._held_choice_said = _hcd50
                out.append(self._say('prompt',
                                     chat_mod.held_choice_note()))
            else:
                # F7: the pending changed KIND - the stale prompt must
                # never print
                drv9._held_choice_due = None
        # BL-106 witness: a LEARNED discard pending vanishing with no
        # new commit is action substitution - caught with a loud alarm
        _pd106 = getattr(drv9.es, 'pending_discard', None) is not None
        _n106 = len(getattr(drv9, 'commits', ()) or ())
        if getattr(self, '_had_discard106', False) and not _pd106 \
                and _n106 == getattr(self, '_commits106', -1):
            drv9.alarms.append(
                'DISCARD PENDING VANISHED WITHOUT A COMMIT - '
                'action-substitution suspect (BL-106)')
        self._had_discard106 = _pd106
        self._commits106 = _n106
        # BL-9a (2026-08-25): battle closed with the human's attack
        # die still staged - say the owed kill ONCE per window
        _event_window = self._finished_staged_event_window()
        _caw9 = getattr(drv9, '_chat_attack_window', None) or _event_window
        if _caw9 is not None \
                and not drv9.human_physical_work_pending() \
                and not drv9.echo_pending() \
                and getattr(drv9.es, 'pending_combat', None) is None \
                and getattr(self, '_die_kill_nagged', None) \
                is not _caw9:
            self._die_kill_nagged = _caw9
            out.append(self._say('prompt', chat_mod.resolved_event_die_kill()
                                 if _caw9 is _event_window else chat_mod.owed_die_kill()))
        # Announce automatic native recovery once after all decisions and
        # physical consequences of the ended turn have completed.
        if drv9.phase == 'actions' and drv9.game_over is None \
                and self._open_step is None and not self._step_queue \
                and drv9.deferred is None \
                and drv9.chat_deferred is None \
                and drv9.pending_buf is None \
                and getattr(drv9.es, 'pending_combat', None) is None \
                and getattr(drv9, '_chat_attack_window', None) is None \
                and getattr(drv9.seg, 'window', None) is None \
                and not drv9.echo_pending() \
                and not _hu_echo8 \
                and drv9.last_actor is not None \
                and getattr(self, '_turnover_said', None) \
                != int(getattr(drv9.es, 'turn', 0) or 0):
            # BL-124(iii) / R4: 'turn over' never speaks past an open
            # battle (the pc gate) or past the human's OPEN echo
            # demands (the _hu_echo8 gate)
            if getattr(drv9, '_turn_over_pending', None) is not None \
                    and drv9._turn_over_pending():
                self._turnover_said = int(getattr(drv9.es, 'turn', 0)
                                          or 0)
                # free-prose kind 'prompt': the text is builder-
                # registered (chat_mod.turn_over_recover)
                out.append(self._say('prompt',
                                     chat_mod.turn_over_recover()))
        # A-09/P5-3: the game-over closer goes out ONCE, as the very
        # last lines; every later cycle returns empty replies so the
        # client's autoplay terminates (fact 16 / C-13).  A victory
        # detected at a COMMIT (mid-action ring) builds its closer
        # here.
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

    def _resuming_owed_dice(self):
        """Exact staging owned by the emitted, journaled decision being retried."""
        if (self._open_step is None or self._open_step[0] != 'ai_decide'
                or not self._em_sent or not self.rand.step_open
                or not self.rand._journal):
            return ()
        return tuple(entry for kind, entry in self._step_owed if kind == 'die')

    def _run_open_step(self):
        item = self._open_step
        kind = item[0]
        drv = self.driver
        if getattr(drv, '_corsairs_arrival', None):
            self._step_parked = True
            return []
        if drv.human_physical_work_pending():
            self._step_parked = True
            return []
        # BRLANE-13 / BRJAVA-3: an AI line the client has not yet shown
        # (ack / echo outstanding) holds the step queue - the next
        # decision would otherwise be taken on a board the client has
        # not reached.  Not a starvation: the deadline lane is loud.
        _ep13 = getattr(drv, 'echo_pending', None)
        if callable(_ep13) and _ep13():
            if not getattr(self, '_echo_hold_said', False):
                self._echo_hold_said = True
                self.notes.append('AI step %r held: my last line(s) are '
                                  'not yet shown by the client (pid %s)'
                                  % (kind, ' '.join(sorted(
                                      getattr(drv, 'echo_expect', {})
                                  ))[:40]))
            return []
        self._echo_hold_said = False
        self._step_parked = False
        from .cycle_recovery import begin_step
        begin_step(self)
        # FLA-18 (par5 g4#6): a step opened BEFORE a resync would
        # replay its journal against a world that no longer exists -
        # the exact 'JOURNAL MISMATCH ... non-deterministic step
        # replay' sticky halt.  Abandon it; the next decision starts
        # fresh on the resynced board.
        _rs18 = float(getattr(drv, '_resync_ts', 0) or 0)
        if getattr(self, '_step_resync0', None) is None:
            self._step_resync0 = _rs18
        if _rs18 > self._step_resync0:
            self._step_resync0 = _rs18
            drv.findings.append(('step_abandoned_after_resync',
                                 str(kind)))
            return self._abandon_step('board resynced under the open '
                                      'step')
        try:
            if kind == 'ai_draws':
                out = self._ai_draws()
            elif kind == 'ai_eyes':
                out = self._ai_eyes()
            elif kind == 'ai_declare':
                out = self._ai_declare()
            elif kind == 'ai_decide':
                import ai
                # recompute the pending on the CURRENT state (the
                # stored dp's menu may be stale after new feed lines)
                resuming_owed = self._resuming_owed_dice()
                dp = self.driver.ai_action_dp(resuming_owed)
                if dp is None:
                    # Another unfinished physical transaction pauses this
                    # still-due action. Its own staged die is already excluded;
                    # phase/owner changes retain the ordinary abandonment path.
                    if (resuming_owed and drv.phase == 'actions'
                            and drv.action_completion_pending(resuming_owed)
                            and ai._next_acting_side(drv.es, drv.last_actor) is self.ai_side):
                        self._step_parked = True
                        return []
                    return self._abandon_step('due decision vanished')
                if dp['kind'] == 'eye':
                    # A queued discard can have settled before it runs.
                    # Eye allocation must retain its phase/roll callback.
                    self._open_step = ('ai_eyes', None)
                    out = self._ai_eyes()
                else:
                    out = self._ai_decide(dp)
            else:
                return self._abandon_step('unknown step kind %r'
                                          % (kind,))
        except RuntimeError as _jm18:
            if 'JOURNAL MISMATCH' not in str(_jm18):
                raise
            if drv.combat_recovery_open():
                # The client's already observed battle cannot be repaired by
                # choosing a fresh action and consuming its rolls differently.
                message = ('combat retry could not replay its recorded choice: %s; '
                           'battle, rolls and physical obligations retained'
                           % str(_jm18))
                self.rand.sticky_halt = message
                drv.findings.append(('combat_journal_mismatch', message))
                raise DriverHalt(message) from _jm18
            # FLA-18 (par5 g4#6): the replay's world changed under
            # the journal - the STEP is dead, not the game.  The
            # sticky halt froze every later cycle (chat lane dead,
            # cycle_errors 0->80).  Degrade: abandon the step (its
            # journal dies with it), stay LOUD, and on a repeat route
            # the self-repair ASK (LOG -> ASK, the standing
            # protocol).
            self.rand.sticky_halt = None
            drv.findings.append(('journal_mismatch_step_abandoned',
                                 '%s :: %s' % (kind,
                                               str(_jm18)[:120])))
            out = self._abandon_step('journal mismatch - '
                                     'non-deterministic replay')
            _msg18 = ('JOURNAL MISMATCH: AI step %r could not be '
                      'replayed deterministically - step abandoned, '
                      'retrying fresh' % (kind,))
            if _msg18 not in drv.alarms:
                drv.alarms.append(_msg18)
            _n18 = getattr(self, '_jm_count18', 0) + 1
            self._jm_count18 = _n18
            if _n18 >= 2:
                for _it18 in drv._propose_repair(
                        'JOURNAL MISMATCH x%d on AI step replay'
                        % _n18, clears='manual'):
                    out.extend(self._dispatch(_it18))
            return out
        # LOUD STALL COUNTER (a stall with no counter is invisible).
        # Directives are deliberately NEVER auto-re-sent (a re-sent
        # draw the client DID execute would draw a second card); the
        # stall is surfaced instead.  BRLANE-3 / M12: a tick PARKED
        # behind the HUMAN's own open decision is not a starvation.
        # FLA-20 / FLH H2: the watchdog PROVED the jar's yes-arm never
        # answered the WWSAT confirm (modal line seen while the
        # card_pick starves).  ONE evidence-gated re-send after the
        # jar's 10s arm window - the previous ask provably died
        # unanswered, so a re-send cannot double-discard.  Asks stay
        # never-re-sent everywhere else.
        _mts20 = getattr(drv, '_wwsat_modal_ts', None)
        if _mts20 and not getattr(self, '_wwsat_resent20', False):
            import time as _t20
            try:
                _p20 = self.rand.pending()
            except Exception:
                _p20 = None
            if getattr(_p20, 'kind', None) == 'card_pick' \
                    and _t20.time() - float(_mts20) > 10.0:
                self._wwsat_resent20 = True
                drv.findings.append(
                    ('wwsat_ask_resent',
                     'watchdog modal line proved the confirm '
                     'unanswered'))
                self.notes.append('re-sending !CALL wwsat once - the '
                                  'client watchdog reported the '
                                  'Confirm Random Discard dialog '
                                  'unanswered')
                out.append('!CALL wwsat')
        if self._open_step is not None and not self._step_parked:
            self._step_starved = getattr(self, '_step_starved', 0) + 1
            if self._step_starved % 25 == 0:
                msg = ('OPEN STEP STARVED: %r retried %d ticks without '
                       'the awaited client outcome (asks sent once, '
                       'never re-sent; pending=%r)'
                       % (kind, self._step_starved,
                          getattr(self.rand.pending(), 'kind', None)))
                self.notes.append(msg)
                print('  [stall] %s' % msg)
        return out

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
        feed_lines = drv.feed.lines[:drv._feed_seq]
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
        # A successful card selection can enter a learned round effect before
        # the native reveal. The engine moves BOTH choices (including nc)
        # into its round cursor at that transition; the card-phase keys no
        # longer carry the human's already committed choice.
        from .automatic_combat_reports import committed_round_cards
        if committed_round_cards(pc) is not None:
            atk_done = def_done = True
        # a side counts as committed once the engine carries its card
        # (atk_/def_combat_card or the hidden sentinel) OR its commit
        # is DEFERRED on the round's rolls: the human's chat commit ran
        # on a clone that starved (chat_deferred), and SYMMETRICALLY
        # the AI's OWN combat_card_play (choose_card emitted,
        # _ai_cc_chosen set) commits only when its apply SUCCEEDS.  The
        # deferred flag is scoped to the side that owns it, so the
        # AI-as-ATTACKER never reveals before the human defender
        # commits, and the AI-as-DEFENDER reveals once the human
        # attacker is done.
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
        if ai_is_attacker and ai_chosen and not def_committed:
            # A Grond/Fighting Uruk-hai round-one prohibition is settled
            # inside the AI's speculative card apply, which may already be
            # waiting for dice. Ask the core's card-phase helper on a separate
            # context whether the defender has a decision at all. The live
            # context must remain untouched until the action commits.
            probe_pc = dict(pc)
            probe_pc.setdefault('atk_combat_card', None)
            needs_choice = ai._maybe_combat_card_phase(es, probe_pc)
            def_committed = (not needs_choice
                             and 'def_combat_card' in probe_pc
                             and probe_pc['def_combat_card'] is None)
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
            # BL-41: a CHAT-committed human card with nothing
            # client-side for the reveal directive still owes its
            # PHYSICAL reveal - the prompt must not hide behind the
            # directive path.  BL-81: 'owed physical reveal' reads the
            # claim-once ledger (a claimed entry owes nothing).
            if any(v == 'owed' for v in
                   (getattr(self.driver, '_cc_reveal_owed', None)
                    or {}).values()):
                key41 = (str(pc.get('attacker_region')),
                         str(pc.get('defender_region')),
                         int(pc.get('rounds_fought', 0) or 0),
                         'chat-reveal')
                if getattr(self, '_reveal_round', None) != key41:
                    self._reveal_round = key41
                    # FLD-7: the nag defers to this say
                    self._reveal_said_at = int(getattr(self, '_cycles',
                                                       0) or 0)
                    return [self._say('prompt',
                                      chat_mod.cc_reveal_drag())]
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
        out = self._emission_out(em)
        # BL-41: a CHAT-committed human card has no client-chosen
        # piece for the reveal directive to flip - the human drags it
        # to the table now (the table mopup then disposes it)
        if any(v == 'owed' for v in
               (getattr(self.driver, '_cc_reveal_owed', None)
                or {}).values()):
            # BL-81: no clear here - only the PHYSICAL reveal claims
            # the entry (re-prompting is blocked by _reveal_round)
            # FLD-7: the nag defers to this say
            self._reveal_said_at = int(getattr(self, '_cycles', 0)
                                       or 0)
            out.append(self._say('prompt', chat_mod.cc_reveal_drag()))
        return out

    def _append_combat_reveal_before_rolls(self, out):
        """A speculative card apply may request dice in its same reply.

        Preserve the chosen-card command before the reveal, then expose the
        cards before any requested roll executes in the client.
        """
        reveal = self._combat_reveal_sweep()
        if not reveal:
            return
        index = next((i for i, line in enumerate(out)
                      if str(line).startswith('!CALL combat_roll ')
                      or ' - normal combat roll:' in str(line)
                      or ' - leader reroll:' in str(line)), len(out))
        out[index:index] = reveal

    def _starved_randomness_watchdog(self):
        """FLD-9 (par5b_g2 siege chain 5+2+2+1 dice with no voiced
        ask; the AI's own owed Shadow draw cleared only by a raw
        !CALL; g1#12 tile-draw starvation; g5#4/#5 fed-draw race): a
        randomness pending can OUTLIVE the event that should have
        voiced its ask (an ask item lost with a halted cycle, a lane
        that never dispatched one).  After 3 unchanged cycles: first
        the fed-randomness retry (the g5 race - the credit may
        already sit in the funnel), then the ONE ask lane
        (_window_ask_lines: nonce mint + dedupe + aged re-ask).
        card_pick stays with its FLA-20 lane (the WWSAT modal)."""
        p = self.driver.pending_random_call()
        if p is None or getattr(p, 'kind', '') \
                not in ('d6', 'tile', 'draws', 'card'):
            self._starved_ask = None
            return []
        key = (str(getattr(p, 'kind', '')),
               tuple(sorted((k, str(v)) for k, v in
                            (getattr(p, 'kw', {}) or {}).items())))
        st = getattr(self, '_starved_ask', None)
        cyc = int(getattr(self, '_cycles', 0) or 0)
        if st is None or st[0] != key:
            self._starved_ask = (key, cyc)
            return []
        if cyc - int(st[1]) < 3:
            return []
        self._starved_ask = (key, cyc)     # next attempt in 3 more
        self.driver._retry_blocked()       # loud: a failure surfaces
        p2 = self.driver.pending_random_call()
        if p2 is None:
            self._starved_ask = None
            return []
        return self._window_ask_lines(p2) or []

    def _owed_reveal_nag(self):
        """E1 (shgame10 blocker, reproduced by
        test_cc_pick_commit_reveal_prompt): the pick/chat-committed
        combat card's PHYSICAL reveal drag is owed REGARDLESS of round
        or battle state - the old prompt sat behind the current
        round's both-committed keys inside _combat_reveal_sweep, so a
        battle that advanced a round (keys cleared) or ended (pc None)
        never prompted, and the client hand kept the card forever
        ('hand size SH: engine=0 client=1').  Safe to say any time:
        the _cc_reveal_owed ledger is written ONLY by the public
        pick/chat lane (the card choice was already announced in
        chat and its effect applied) - the K1 hidden-commit lane
        never reaches this ledger.  Re-nags every ~30 cycles while
        unperformed; the BL-81 claim-once consume clears it."""
        _led = getattr(self.driver, '_cc_reveal_owed', None) or {}
        owed = sorted(k for k, v in _led.items() if v == 'owed')
        if not owed:
            self._reveal_nag_key = None
            return []
        # FLD-7 (par5b g4#2/#5 2-3x, g1#10 6x): the cycles//30 BUCKET
        # key re-printed at every bucket boundary (a demand said at
        # cycle 29 nagged again at cycle 30, back-to-back).  The
        # contract is ONCE, with a slow fallback: re-say only 30 full
        # cycles after the LAST say of this owed set - shared with
        # the sweep's prompt via _reveal_said_at.
        _cyc = int(getattr(self, '_cycles', 0) or 0)
        _prev = getattr(self, '_reveal_nag_key', None)
        if _prev is not None and _prev[0] == tuple(owed) \
                and _cyc - int(_prev[1]) < 30:
            return []
        _sw = getattr(self, '_reveal_said_at', None)
        if _sw is not None and _cyc - int(_sw) < 30 \
                and (_prev is None or _prev[0] == tuple(owed)):
            # the reveal sweep just said it this window
            self._reveal_nag_key = (tuple(owed), int(_sw))
            return []
        self._reveal_nag_key = (tuple(owed), _cyc)
        self._reveal_said_at = _cyc
        return [self._say('prompt', chat_mod.cc_reveal_drag())]

    def _battle_announce_if_new(self, pre, post, act):
        """FLC H5 (par5 g1#8): a card-initiated battle (The Fighting
        Uruk-hai opens pending_combat straight from the play) never
        got the army_attack lane's declaration chat and went straight
        to roll asks.  A battle newly opened by any NON-army_attack
        AI action announces the same way, once."""
        _pc5b = getattr(post, 'pending_combat', None)
        if not isinstance(_pc5b, dict) \
                or isinstance(getattr(pre, 'pending_combat', None),
                              dict) \
                or str(getattr(act, 'action_type', '')) \
                == 'army_attack':
            return []
        _to5b = str(_pc5b.get('defender_region', '') or '')
        _fr5b = str(_pc5b.get('attacker_region', '') or '')
        if not _to5b:
            return []
        _msg5b = (_to5b if _fr5b in ('', _to5b)
                  else '%s from %s' % (_to5b, _fr5b))
        self.driver.findings.append(('card_battle_announced', _msg5b))
        return [self._say('attack_declare', _msg5b)]

    def _abandon_step(self, reason):
        """K3: drop the open AI step WITHOUT a commit.  Invariant: no
        step-scoped residue survives an abandon - the randomness step
        (journal, pending ask, cursors), the once-per-step emission
        flag, the directive dedup sets and the expressed-ops ledger
        all reset together, so the next decision renders and asks
        exactly as a first attempt would.  Idempotent when nothing is
        open; noted (panel) when something was."""
        had = (self._open_step[0] if self._open_step else None,
               bool(self.rand.step_open), bool(self._em_sent)
               or getattr(self, '_step_dead_men_arrival', None) is not None)
        out = []
        if had[2]:
            out.extend(self._unstage_abandoned())
        self.rand.abort_step(reason)
        self._open_step = None
        self._step_starved = 0
        self._step_resync0 = None    # FLA-18: next step re-baselines
        self._step_dead_men_arrival = None
        self._step_casualty_prefix = None
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

    def _note_political_emission(self, ln):
        """BL-6 (2026-08-25): EVERY outgoing line that walks a
        political marker registers on the driver's recon in-flight
        ledger (cleared only when the client DUMP confirms the box).
        The old feed-tail check missed the AI's own already-applied
        emission, so the recon double-walked the marker against a
        stale dump (fpgame1: the Isengard advance rendered twice)."""
        # BL-35: EVERY outgoing line's piece pids enter the driver's
        # recent-emissions ledger - a recon must not touch a piece
        # whose emitted move/kill the dump has not shown yet
        try:
            import re as _re35
            import time as _t35
            _rec35 = getattr(self.driver, '_recent_emit_pids', None)
            if _rec35 is None:
                _rec35 = self.driver._recent_emit_pids = {}
            for _m35 in _re35.finditer(
                    r'(?<![\d.])(\d{2,3})(?![\d.])', ln):
                _rec35[_m35.group(1)] = _t35.time()
        except Exception:
            pass
        if '<~Game.20' not in ln:
            return
        try:
            import re as _re
            from log_replay import POLITICAL_MARKERS
        except Exception:
            return
        zones = _re.findall(r'<~(Game\.200[0-3])~>', ln)
        if not zones:
            return
        for pid in POLITICAL_MARKERS:
            if _re.search(r'(?<!\d)%s(?!\d)' % _re.escape(str(pid)),
                          ln):
                infl = getattr(self.driver, '_pol_recon_inflight',
                               None)
                if infl is None:
                    infl = self.driver._pol_recon_inflight = {}
                infl[str(pid)] = zones[-1]     # renderers put 'to' last
                break

    def _nc_must_hold(self):
        """BL-22 (K1 blinding): the AI's 'nc' is a commitment - it
        stays held while the ROUND's human card commitment is still
        outstanding (not merely while their decision window is open;
        fpgame5's nc printed before the window even existed).
        Released once the engine or a deferred chat carries the
        human's card/none, or the round is gone."""
        drv = self.driver
        pc = getattr(drv.es, 'pending_combat', None)
        if not isinstance(pc, dict):
            return False
        if drv.human_cc_committed:
            return False
        ai_is_atk = pc.get('attacker_side') is self.ai_side
        hkeys = (('def_combat_card', 'def_hidden_cc') if ai_is_atk
                 else ('atk_combat_card', 'atk_hidden_cc'))
        if any(k in pc for k in hkeys):
            return False
        if 'atk_hits' in pc or pc.get('alloc_stage'):
            return False        # the round is past the card stage
        cd = getattr(drv, 'chat_deferred', None)
        cd_at = str(getattr(cd[0], 'action_type', '')) if cd else ''
        if cd_at in ('combat_card_none', 'combat_card_play',
                     'combat_card_hidden_commit'):
            return False
        return True

    def _dispatch(self, item):
        kind = item[0]
        if self.driver.game_over is not None and kind in ('ai_roll', 'bonus_dice'):
            return []  # These start a later turn, unlike committed board cleanup.
        if kind == 'line' or kind == 'call':
            self._note_political_emission(str(item[1]))
            return [item[1]]
        if kind == 'chat':
            if (len(item) > 3 and isinstance(item[3], dict)
                    and 'pending_at_commit' in item[3]
                    and item[3]['pending_at_commit'] != self.driver.n_commits_total):
                return []  # The queued choice was resolved before delivery.
            if item[1] == 'nc' and self._nc_must_hold():
                # BL-22: held while the round's human card commitment
                # is outstanding (not merely while their decision
                # window is open)
                self._held_nc = item[2]
                self.notes.append('nc held until the human commits '
                                  '(K1 blinding)')
                return []
            return [self._say(item[1], item[2])]
        if kind == 'note':
            self.notes.append(item[1])
            print('  [note] %s' % item[1])
            return []
        if kind == 'recover_ai':
            return self._recover(item[1])
        if kind == 'skip_request':
            from .skip_recovery import arm as arm_skip
            arm_skip(self, item[1])
            return []
        if kind == 'recovery_request':
            self._take_recovery_controls([item])
            return []
        if kind == 'ai_roll':
            return self._order_own_roll()
        if kind == 'rng_ask':
            return self._window_ask_lines(item[1])
        if kind == 'cancel_card_request':
            # Native Undo reopened an optional card choice before any outcome.
            # Retire only that choice's issued nonce and window dedup entry.
            nonce = item[1]
            ask = self._live_asks.get(nonce)
            if ask is not None and self._rng_result_rendered(
                    ask['kind'], self.driver.feed.lines[ask['feed_mark']:]) is None:
                del self._live_asks[nonce]
                for ident, meta in list(getattr(self, '_ask_meta', {}).items()):
                    if meta.get('nonce') == nonce:
                        self._window_asks_sent.discard(ident)
                        del self._ask_meta[ident]
            return []
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
        from .turn_bookkeeping import _fresh_board, bind_unrolled_bonus_dice
        observed = _fresh_board(self)
        if observed is not None:
            bind_unrolled_bonus_dice(drv, observed)
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
        # BL-9b: the HUMAN pool grew too - only the AI's own die is
        # rendered ('they drag their own'), so it must be SAID or the
        # client pool runs one short forever
        hs = self.human_side
        h0 = int(pre_counts.get(hs.name, 0) or 0)
        h1 = len(drv.es.dice.get(hs, []))
        if h1 > h0:
            lbl = 'FP' if hs is gs.Side.FREE_PEOPLES else 'SA'
            # USER manual game 2026-08-28 (Aragorn crowning): the human
            # may already have pulled the new die out of Reinforcements
            # mid-turn - read the client (count the human's die pieces
            # NOT in Reinforcements); if the pool already accounts for
            # the new die, say what the board shows instead of the
            # drag prompt
            _hb = (emitter._FP_DIE_IDS if hs is gs.Side.FREE_PEOPLES
                   else emitter._SH_DIE_IDS)
            try:
                drv.view.refresh_dump()
            except (OSError, ValueError) as _rd_err:
                self.notes.append('J1 bonus-die dump read failed (%r) - '
                                  'judging from the last board read'
                                  % (_rd_err,))
            _pr = getattr(drv.view, 'piece_region', {}) or {}
            _on_table = sum(1 for p in _hb
                            if p in _pr
                            and 'reinf' not in str(_pr[p]).lower())
            if _on_table >= h1:
                self.notes.append('J1 human bonus die already on the '
                                  'table (%d of %d die pieces outside '
                                  'Reinforcements) - no drag prompt'
                                  % (_on_table, h1))
                lines.append(self._say(
                    'prompt', chat_mod.bonus_die_already_taken(lbl, h1)))
            else:
                lines.append(self._say('prompt', chat_mod.bonus_die_drag(
                    lbl, h1 - h0)))
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
                owed = (getattr(self.driver, '_track_owed', None) or {}).get('252')
                if z.startswith('track '):
                    frm = 'Game.%d' % (2014 + int(z.split()[1]))
                    if frm != dest and not (owed and owed[0] == min(12, cor)):
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

    def _finished_staged_event_window(self):
        """A revealed human event still owes its die only after all effects."""
        drv = self.driver
        window = getattr(drv, '_staged_event_window', None)
        if window is None or window is not getattr(drv.seg, 'window', None):
            return None
        import ai
        if (getattr(drv.es, 'pending_combat', None) is not None
                or ai.describe_pending(drv.es) is not None
                or drv.deferred is not None or drv.chat_deferred is not None
                or drv.pending_buf is not None or drv.pending_random_call() is not None
                or drv.human_physical_work_pending() or drv.echo_pending()):
            return None
        return window

    def _unanswered_human_choice(self, pending):
        """An offered human choice with no failed gesture or transport work.

        Use the caller's existing engine pending; elapsed player thinking time
        alone is neither a matcher failure nor a reason to offer repair.
        """
        drv = self.driver
        return bool(pending and pending['side'] is self.human_side
                    and pending.get('actions')
                    and not getattr(drv.pending_buf, 'records', None)
                    and drv.deferred is None and drv.chat_deferred is None
                    and not drv.uncommitted and not drv.holds and not drv.alarms
                    and drv._repair is None and not drv.echo_pending()
                    and not drv.echo_ledger and self.rand.pending() is None)

    def _waiting_on(self):
        """WAVE-J (USER: the old panel said EXACTLY what the AI was
        waiting on; the lift carried the status SCHEMA but never the
        population).  The PRECISE blocker in USER-readable words,
        composed from the driver's actual state, priority-ordered;
        stalled_seconds rides along.  An empty answer while ANY
        starved/deferred/open state exists is a pinned test failure
        (the silent-stall disease on the display surface)."""
        drv = self.driver
        from .turn_rewind import _state
        if _state(self).review:
            return 'Rewind complete: restored board is paused. Press RESUME or say resume.'
        if getattr(drv, '_sidecar_wait_seq', None) is not None:
            return 'waiting for native gesture metadata at feed line %d' % drv._sidecar_wait_seq
        if getattr(drv, '_grouped_die_identity_wait', None) is not None:
            return ('waiting for the native identities of your selected action dice; '
                    'selection retained. Panel RESET + RESYNC can retry board recovery')
        if drv.forced_ai_rows_pending():
            return drv.forced_ai_rows_wait()
        stall = getattr(self, '_stalled', 0.0)
        sfx = (' [stalled %ds]' % int(stall)) if (
            stall >= 10 and drv._partial_casualty_wait() is None
            and not drv._discard_identity_wait()
            and not drv._card_completion_wait()) else ''
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
        if getattr(self, '_skip_requested', False):
            return 'Skip requested: waiting for a complete current client board export; retrying automatically'
        if getattr(drv, '_fellowship_completion_requested', False):
            return 'Your fp ok is recorded; waiting for the client board export to finish the Fellowship phase'
        eye_request = getattr(drv, '_eye_completion_requested', None)
        if getattr(drv, '_allocation_roll_wait', None) is not None:
            return (drv._allocation_roll_wait.get('waiting') or
                    'Your action-dice roll is recorded; waiting for its native dice positions and identities. Do not roll again.')
        if eye_request is not None and drv.phase == 'eyes':
            if eye_request[1] is None:
                return 'Your done is recorded; waiting for the complete native Hunt allocation'
            return ('waiting on: your announced %d Hunt eyes on the board '
                    '(%d placed); say done to accept the current allocation'
                    % (eye_request[1], int(getattr(drv.es, 'eyes_allocated', 0) or 0)))
        if drv.human_physical_work_pending():
            if getattr(drv, '_siege_seat_owed', None):
                return ('waiting on: move every defending unit and leader into the '
                        '%s Stronghold box' % drv._siege_seat_owed)
            return ('waiting on: your committed piece removals, replacements '
                    'or moves to be completed on the board%s' % sfx)
        if getattr(self, '_recovery_request', None) is not None:
            return ('waiting on: action-dice recovery verification - %s%s' % (
                getattr(getattr(self, '_turn_bookkeeping', None), 'wait', '')
                or 'the native dice positions', sfx))
        try:
            # The ordinary matcher can stop after a proved physical payment
            # while the authoritative engine still shows its pre-payment army.
            # Use the same owned request that the emitted roll ask uses.
            p = drv.pending_random_call()
        except Exception:
            p = None
        if p is not None:
            kw = getattr(p, 'kw', {}) or {}
            k = getattr(p, 'kind', '?')
            if k == 'd6':
                # An issued, unanswered human roll is expected input time.
                # Keep the feed-age metric and every watchdog untouched;
                # only avoid presenting this proved wait as a stalled engine.
                from .pending_roll_wait import issued_waiting
                if (not drv.echo_pending() and not drv.alarms
                        and getattr(drv, '_repair', None) is None
                        and issued_waiting(self) is p):
                    sfx = ''
                return ('waiting on: %s%s'
                        % (chat_mod.roll_context(p, self.rand.human_side), sfx))
            if k == 'tile':
                return 'waiting on: client hunt-tile draw%s' % sfx
            if k in ('card', 'draws'):
                return ('waiting on: client card draw (%s)%s'
                        % (kw.get('items', kw), sfx))
            if k == 'card_pick':
                # FLA-20 / FLH H2: name the human recovery route -
                # the client's confirm dialog may be sitting open
                return ('waiting on: client card_pick (%s) - the '
                        "client's Confirm Random Discard dialog may "
                        'be open: press Continue in the client '
                        'window (seat: raw "!CALL wwsat")%s'
                        % (kw, sfx))
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
        # BRLANE-3 / M12: the HUMAN's open decision outranks a parked AI
        # step (shgame22 waited 184 s on 'AI deciding: ai_declare' while
        # the engine waited on the human's discard)
        if self._open_step is not None:
            try:
                import ai as _ai3
                with self.rand.pin():
                    _dp3 = _ai3.describe_pending(drv.es)
            except Exception:
                _dp3 = None
            if _dp3 is not None and _dp3['side'] is self.human_side:
                if _dp3['kind'] == 'guide_change':
                    from .guide_completion import status as guide_status
                    return guide_status(drv)
                from .decision_prompts import decision_label, pending_instructions
                if self._unanswered_human_choice(_dp3):
                    sfx = ''
                return ('waiting on your %s: %s%s' % (
                    decision_label(_dp3['kind']),
                    pending_instructions(_dp3, drv.es), sfx))
            _ep3 = getattr(drv, 'echo_pending', None)
            if callable(_ep3) and _ep3():
                return ('waiting on: your client to render my last '
                        'line(s) (pid %s)%s'
                        % (' '.join(sorted(getattr(drv, 'echo_expect',
                                                   {})))[:40], sfx))
        if self._open_step is not None:
            if self._open_step[0] == 'ai_draws':
                reason = getattr(getattr(self, '_turn_bookkeeping', None), 'wait', '')
                if reason:
                    return 'waiting on: turn-start cards - %s%s' % (reason, sfx)
            return ('AI deciding: %s (retry %d)%s'
                    % (self._open_step[0],
                       getattr(self, '_step_starved', 0), sfx))
        _rep = getattr(drv, '_repair', None)
        if _rep is not None:
            import os as _os_r
            if _rep.get('kind') == 'census':
                _rows_c = list(_rep.get('rows') or ())
                return ('BOARD DIVERGENCE - %d row(s) confirmed (first: '
                        '%s) | say yes to adopt the client board into '
                        'the engine, or no to keep the engine (log %s)%s'
                        % (len(_rows_c),
                           (_rows_c[0][:100] if _rows_c else '-'),
                           _os_r.path.basename(_rep.get('log') or '')
                           or 'unsaved', sfx))
            if _rep.get('phase') == 'offered':
                return ('STALLED - %s | say yes to self-repair (log %s '
                        'saved; %d staged piece(s) would be killed; %s '
                        'to act after) or no%s'
                        % (drv.stall_text(),
                           _os_r.path.basename(_rep.get('log') or '')
                           or 'unsaved', len(_rep.get('kills') or ()),
                           getattr(_rep.get('to_act'), 'name', '?'), sfx))
            if _rep.get('phase') == 'kill':
                return ('self-repairing: killed the staged pieces, '
                        'waiting for your client to show it%s' % sfx)
            return 'self-repairing: resyncing to your board%s' % sfx
        if drv.pending_buf is not None \
                and getattr(drv.pending_buf, 'records', None):
            partial = drv._partial_hunt_corruption()
            if partial is not None:
                # FileBridge keeps only 140 characters. Put the required
                # amount, marker target and confirmation before that limit.
                return ('waiting on: Hunt has %d damage: move corruption '
                        'from %d to %d or say corruption. Undo the partial '
                        'move before choosing a companion.%s'
                        % (partial['damage'], partial['current'],
                           partial['target'], sfx))
            # USER 2026-08-28: never a generic 'gesture match for N
            # buffered record(s)' - name the decision, the lines, the
            # options and what would satisfy it
            return 'waiting on: %s%s' % (drv.stall_text(), sfx)
        try:
            import ai
            with self.rand.pin():
                dp = ai.describe_pending(drv.es, self.human_side)
        except Exception:
            dp = None
        if (drv.phase == 'guide' and dp is not None and dp['kind'] == 'eye'
                and self.human_side.name == 'FREE_PEOPLES'):
            return 'waiting on: say fp ok to finish the Fellowship phase; then I allocate Hunt dice'
        if dp is not None:
            from .decision_prompts import decision_label, pending_instructions
            if dp['side'] is not self.human_side:
                return 'waiting on: AI %s%s' % (decision_label(dp['kind']), sfx)
            if dp['kind'] == 'guide_change':
                from .guide_completion import status as guide_status
                return guide_status(drv)
            if self._unanswered_human_choice(dp):
                sfx = ''
            return ('waiting on your %s: %s%s' % (
                decision_label(dp['kind']), pending_instructions(dp, drv.es), sfx))
        pc = getattr(drv.es, 'pending_combat', None)
        if pc is not None:
            # Hidden-card reveal is an external client event, so the engine
            # can have a live battle without a legal-action decision here.
            if pc.get('phase') == 'combat_card_reveal':
                return ('waiting on: reveal the selected combat cards in '
                        'the client before rolling%s' % sfx)
            return 'waiting on: complete the current battle on the board%s' % sfx
        if drv.phase == 'actions' and self._finished_staged_event_window() is not None:
            return ('waiting on: your action-die kill - the card effect '
                    'is complete; remove the staged die from Current Action%s' % sfx)
        if drv.phase == 'actions' \
                and getattr(drv, '_chat_attack_window', None) \
                is not None \
                and getattr(drv.es, 'pending_combat', None) is None:
            # BL-9a: the battle closed but the staged attack die was
            # never killed - the panel demands it (recovery misses a
            # die left in Current Action)
            return ('waiting on: your action-die kill - the battle '
                    'closed; drag the staged die out of Current '
                    'Action%s' % sfx)
        if drv.phase == 'actions':
            nxt_err = False
            try:
                import ai
                nxt = (ai._next_acting_side(drv.es, drv.last_actor)
                       if drv.last_actor is not None else None)
            except Exception:
                nxt = None
                nxt_err = True
            if nxt is None and drv.last_actor is not None \
                    and not nxt_err:
                if not drv._turn_over_pending():
                    return ('waiting on: finish the current action and '
                            'its outstanding board changes%s' % sfx)
                return ('turn over - automatically recovering dice and '
                        'drawing cards%s' % sfx)
            who = (nxt.name if nxt is not None
                   else self.human_side.name)
            return 'waiting on: %s to act%s' % (who, sfx)
        import game_state as _gs
        if self.ai_side is _gs.Side.FREE_PEOPLES:
            # R2: the seats swap - the AI holds the Fellowship phase,
            # the human Shadow the eyes + the first roll
            names = {'draws': 'automatic turn-start card draws',
                     'declare': 'AI declaring (its Fellowship phase)',
                     'eyes': 'human Shadow eye allocation (place eyes, '
                             'say 0e/1e/2e/etc. or done)',
                     'roll': 'automatic action-dice rolls'}
        else:
            names = {'draws': 'automatic turn-start card draws',
                     'declare': "declaration window ('fp ok')",
                     'eyes': 'AI eye allocation',
                     'roll': 'action-dice rolls'}
        if str(drv.phase) == 'roll':
            owed = [side.name.replace('_', ' ').title()
                    for side in (self.human_side, self.ai_side)
                    if any(not die.used and die.current is None
                           for die in drv.es.dice.get(side, []))]
            return ('waiting on: automatic %s action-dice rolls%s'
                    % (' and '.join(owed) or 'native', sfx))
        if drv.phase == 'draws':
            reason = getattr(getattr(self, '_turn_bookkeeping', None), 'wait', '')
            if reason:
                return 'waiting on: turn-start cards - %s%s' % (reason, sfx)
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
            # H1 (BL-148): the view's piece-type SWAP map is frozen
            # beside the pickle so the offline census
            # (test_live_session_census.py) decodes downgraded elites
            # correctly
            try:
                _dt = getattr(getattr(drv, 'view', None),
                              'dynamic_types', None)
                if _dt:
                    _dtp = os.path.join(cd, 'ai_dynamic_types.json')
                    _dttmp = _dtp + '.tmp'
                    with open(_dttmp, 'w', encoding='utf-8') as _f:
                        _json.dump({str(k): _json_safe(v)
                                    for k, v in _dt.items()}, _f)
                    os.replace(_dttmp, _dtp)
            except (OSError, TypeError, ValueError) as _e:
                self.notes.append('dynamic_types freeze skipped: %r' % _e)
            findings = [('%s: %s' % (str(k), str(d)[:120]))
                        for k, d in list(
                            getattr(drv, 'findings', ()) or ())[-20:]]
            alarms = [str(a)[:200]
                      for a in list(getattr(drv, 'alarms', ())
                                    or ())[-20:]]
            mism = findings + alarms
            # BL-37 three-class taxonomy (taxonomy.py, ONE taxonomy for
            # every surface): DIVERGENCE-class alone feeds the tester's
            # STOP rule; ERROR-class is its own counter (stop only when
            # wedged); everything else is protocol info.  The full
            # mism list stays published for diagnosis.
            from .taxonomy import partition as _partition37
            _div37, _err37, _info37 = _partition37(mism)
            # BRCMP-10 (2026-08-29): client_divergences is the count
            # of BOARD rows from the latest QUIESCENT census read,
            # nothing else; divergence-class events (retracting
            # alarms, wedge findings) are `stop_signals`
            _lq = getattr(drv, '_last_quiescent_census', None) or {}
            _board37 = list(_lq.get('board') or ())
            _stg37 = list(_lq.get('staging') or ())
            _held37 = getattr(drv, '_census_held', None)
            n_div = len(_board37)
            n_err = len(_err37)
            _since = getattr(drv, '_recon_sig_since', None)
            _age37 = (round(now - float(_since), 1)
                      if _board37 and _since else 0)
            # BL-122: sweep the live-ask registry - an ask closes when
            # its result RENDERED since the ask's feed mark, or when
            # its commit epoch passed.  BL-130 witness: a d6 result of
            # the wrong WIDTH is a named mismatch note.
            _la122 = getattr(self, '_live_asks', None) or {}
            for _nid in list(_la122):
                _ent = _la122[_nid]
                _tail = list(drv.feed.lines)[_ent['feed_mark']:]
                _hit = self._rng_result_rendered(_ent['kind'], _tail)
                if _hit is not None:
                    if _ent['kind'] == 'd6' and _ent.get('need'):
                        import re as _re122
                        _w = len(_re122.findall(r'\b\d+\b',
                                                str(_hit).split(
                                                    'rolls', 1)[-1]))
                        if _w != int(_ent['need']):
                            self.notes.append(
                                'ROLL COUNT MISMATCH (ask #%d): '
                                'asked %d, rendered %d (%.60s)'
                                % (_nid, int(_ent['need']), _w,
                                   str(_hit)))
                    del _la122[_nid]
                elif _ent['epoch'] != drv.n_commits_total:
                    del _la122[_nid]
            status = {
                'mode': 'sided',
                'ai_side': self.ai_side.name,
                'turn': int(getattr(drv.es, 'turn', 0)),
                'phase': str(getattr(drv, 'phase', 'sided')),
                'active': None,
                'asks': [dict(_la122[k], nonce=k)
                         for k in sorted(_la122)][:8],
                'last': str(self._waiting_on())[:200],
                'client_divergences': n_div,
                'stop_signals': len(_div37),
                'error_count': n_err,
                'census_rows': n_div,
                'census_first': (_board37[0][:200] if _board37
                                 else None),
                'census_staging': len(_stg37),
                'census_held': (str(_held37)[:200] if _held37
                                else None),
                'census_age': _age37,
                'mismatches': mism[:20],
                'engine_vs_parser': 'n/a (sided: parser IS the view)',
                # BL-145 audit residual: the summary string derives
                # from the SAME sources as the counters - the quiescent
                # census for divergences, the taxonomy partition for
                # stop signals / errors / notes
                'engine_vs_client': (
                    '; '.join(
                        ([('%d divergence(s) [quiescent census, %ss]: %s'
                           % (n_div, _age37,
                              '; '.join(r[:120] for r in _board37[:3])))]
                         if _board37 else [])
                        + ([str(_held37)[:160]] if _held37 else [])
                        + ([('%d stop signal(s): %s'
                             % (len(_div37), '; '.join(_div37[:3])))]
                           if _div37 else [])
                        + ([('%d error(s), %d note(s)'
                             % (len(_err37), len(_info37)))]
                           if (_err37 or _info37) else []))
                    or 'OK'),
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
            if getattr(self, '_mirror_warned', None) != str(e) \
                    or self._mirror_skips % 20 == 0:
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
        Java's casualty/reinforcement destination, never popped: an
        untracked pid falls back to its setup zone and could be picked again;
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
                        view.piece_region[pid] = emitter._native_kill_zone(view, pid)
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

    def _stage_human_costs_before_request(self, call, stopped=None, gesture=None):
        """Hold a real continuation request until its human losses are shown.

        The engine transaction is still open. Its stopped clone supplies only
        casualty evidence; capture and other consequences remain uncommitted.
        Speculative or obsolete callbacks cannot create physical debts.
        """
        drv = self.driver
        if call is not drv.pending_random_call():
            return None
        automatic_prefix = stopped is None
        if stopped is None:
            if (getattr(call, '_physical_request_owner', None) is not self.rand
                    or getattr(call, '_physical_base_state', None) is not drv.es):
                return None
            stopped = getattr(call, '_physical_pre_request_state', None)
        if stopped is None or not getattr(drv.es, 'pending_combat', None):
            return None
        items = drv._stage_human_casualty_prefix(stopped, gesture=gesture)
        if automatic_prefix:
            expressed = list(self._step_expressed or ())
            expressed.extend(getattr(drv.pending_buf, '_pre_rng_expressed', ()) or ())
            items.extend(drv._stage_ai_casualty_prefix(stopped, expressed=expressed))
        items.extend(drv._auto_wiped_figure_cleanup(expected_state=stopped))
        out = []
        for item in items:
            out.extend(self._dispatch(item))
        if out or drv.human_physical_work_pending() or drv.echo_pending():
            return out
        return None

    def _stage_proven_casualties_before_roll(self, call):
        """Display already-determined losses before their subsequent attack.

        The domain state remains at its transaction start. Only a unanimous
        stopped engine prefix may produce these rows, using the ordinary
        emitter and positive-acknowledgement barrier.
        """
        drv = self.driver
        if call is not drv._proven_pending_random_call():
            return None
        buf = drv.pending_buf
        stopped = getattr(buf, '_proven_starve_state', None)
        if stopped is None:
            raise DriverHalt('Post-casualty roll lacks its proved pre-roll state')
        staged = self._stage_human_costs_before_request(call, stopped, gesture=buf)
        if staged is not None:
            return staged
        from . import emission as em_mod
        tag = 'shadow' if self.ai_side.name == 'SHADOW' else 'fp'
        rows = em_mod.removal_rows(drv.es, stopped, tag)
        expressed = getattr(buf, '_pre_rng_expressed', ())
        if expressed:
            prior, failed = em_mod.expressed_rows(drv.view, expressed, tag)
            if failed:
                raise DriverHalt('Cannot identify previously emitted pre-roll casualties')
            rows = em_mod.subtract_rows(rows, prior)
        if not em_mod.rows_empty(rows):
            if drv.echo_pending():
                return []
            em, ledger = em_mod.render_forced_rows(
                drv.view, self.nick, self.ai_side, rows, drv.es,
                label='pre-extra-attack casualties', post=stopped)
            if em.hold:
                raise DriverHalt('Pre-extra-attack casualty emission held: %s' % em.hold)
            out = self._emission_out(em)
            if not out:
                raise DriverHalt('Pre-extra-attack casualties produced no native operation')
            buf._pre_rng_expressed = list(expressed) + list(em.ops)
            drv.echo_ledger.extend(ledger)
            return out
        if expressed and drv.echo_pending():
            return []
        return None

    def _window_ask_lines(self, call):
        """Order the randomness a DEFERRED human window / pending
        buffer is starving on (PHASE3-D port: the spectator reads a
        window's rng from the log; live must SOLICIT it â€” an AI-side
        defender roll would otherwise starve forever).  Epoch-keyed
        dedup (see __init__)."""
        if self.driver.game_over is not None or self.driver.forced_ai_rows_pending():
            return []
        staged = self._stage_proven_casualties_before_roll(call)
        if staged is not None:
            return staged
        staged = self._stage_human_costs_before_request(call)
        if staged is not None:
            return staged
        from .automatic_combat_reports import awaiting_selected_card_reveal
        if awaiting_selected_card_reveal(self, call):
            return []
        # A pending AI step owns its request until that step completes. The
        # generic starvation watchdog also arrives here, but must not mint
        # a second human roll request through a separate dedup registry.
        if self._open_step is not None:
            return self._dir_lines(call)
        # the epoch must advance on every window SETTLEMENT, not
        # commits only - a REFUSED window consumes its ask without a
        # commit, and the next window's identical ask would be falsely
        # deduped (the deferral would starve with no '!CALL' ever
        # going out).  Refusal + hold counts ride the key.
        ws = self.driver.window_stats
        epoch = (self.driver.n_commits_total,
                 int(ws.get('none', 0) or 0),
                 int(ws.get('hold', 0) or 0))
        # One window can legitimately need identical requests in sequence:
        # each failed Hunt die has its own reroll. The captured consumption
        # cursor permits the next request; idle retries, unconsumed outcomes
        # and unrelated-side outcomes retain the original request identity.
        ident = (epoch, call.kind, tuple(sorted(
            (k, str(v)) for k, v in call.kw.items())),
            getattr(call, '_request_cursor', None))
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
            # WATCHDOG: judge an aged request against the rendered feed.
            # A rendered-but-unharvested result is an error, never another
            # roll. An unanswered automated call retains its retry below;
            # an unanswered human roll simply remains pending.
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
            human_roll = (call.kind == 'd6'
                          and rnd_mod._side_name(call.kw.get('side'))
                          == rnd_mod._side_name(self.human_side))
            human_draw = (call.kind == 'draws' and call.kw.get('items')
                          and all(rnd_mod._side_name(side) == rnd_mod._side_name(self.human_side)
                                  for side, _ in call.kw['items']))
            if human_roll or human_draw:
                # A human may take arbitrarily long to use the native dice
                # control. Silence is not a failed client command: minting
                # another nonce asks them to roll the same dice again. Keep
                # the original request; a new consumed cursor identifies a
                # genuinely subsequent normal roll or leadership reroll.
                return []
            self.notes.append(
                'WATCHDOG re-ask: %s ask aged %d ticks, NO rendered '
                'result in the feed since it - the client provably '
                'never executed it' % (call.kind, m['ticks']))
            self._window_asks_sent.discard(ident)
        self._window_asks_sent.add(ident)
        meta[ident] = {'feed_mark': len(lines_now), 'ticks': 0}
        from .automatic_combat_reports import before_roll
        out = before_roll(self, call)
        from .fixed_forfeit_reports import before_roll as fixed_before_roll
        out.extend(fixed_before_roll(self, call))
        from .anduril_reports import before_roll as anduril_before_roll
        out.extend(anduril_before_roll(self, call))
        if call.kind == 'd6':
            out.append(self._say('prompt', chat_mod.roll_context(
                call, self.rand.human_side)))
        dirs = list(self.rand.directives_for(call))
        for d in dirs:
            if d.kind == 'chat':
                txt = d.text
                if call.kind == 'draws':
                    out.append(self._say('prompt', txt))
                    continue
                if not d.text.endswith('?'):
                    # BL-122: the roll ask carries a NONCE - the seat
                    # answers each nonce exactly once, a re-nagged SAME
                    # ask is never double-answered, a
                    # legitimately-recurring same-text ask is never
                    # falsely deduped. Human d6 requests never expire into
                    # a fresh nonce merely through waiting. BL-130: the
                    # demanded count rides the entry.
                    txt = self._nonce_ask(
                        d.text, str(getattr(call, 'kind', '')),
                        int((getattr(call, 'kw', {}) or {})
                            .get('need', 0) or 0),
                        feed_mark=len(lines_now))
                    meta[ident]['nonce'] = self._ask_nonce_seq
                    buf = self.driver.pending_buf
                    if getattr(buf, '_staged_card_closed', False):
                        buf._staged_card_request_nonce = self._ask_nonce_seq
                out.append(self._say(
                    'random_ask' if d.text.endswith('?')
                    else 'roll_request', txt))
            else:
                if d.text == '!CALL hunt_tile':
                    out.extend(self._tile_pool_preheal())
                    # BL-34: a SECOND tile draw in one commit epoch is
                    # the stronghold-reveal rule firing - say so, or it
                    # reads as a double-draw
                    _ep34 = int(getattr(self.driver,
                                        'n_commits_total', 0) or 0)
                    if getattr(self, '_tile_ask_epoch', None) \
                            == _ep34:
                        out.append(self._say(
                            'prompt',
                            chat_mod.stronghold_tile_note()))
                    self._tile_ask_epoch = _ep34
                out.append(d.text)
        # BL-30: a HUMAN-owed draw is asked out loud, like the rolls
        # (a silent starve is never acceptable)
        if not dirs and str(getattr(call, 'kind', '')) \
                in ('card', 'draws'):
            kw30 = getattr(call, 'kw', {}) or {}
            try:
                _s30, _d30 = list(kw30.get('items') or [])[0]
                lbl = '%s %s' % (str(_s30).replace('_', ' ').lower(),
                                 _d30)
            except Exception:
                lbl = 'card'
            out.append(self._say('prompt', chat_mod.draw_ask(lbl)))
        from .pending_roll_wait import remember_issued
        remember_issued(self, call, out)
        return out

    def _nonce_ask(self, text, kind, need, feed_mark=None):
        """BL-122: mint the next ask nonce, register the live ask
        (kind, demanded count, feed mark, commit epoch) and return
        the published text ('roll 2 dice #7').  ONE minting point
        for every human roll ask - the funnel lane and the turn-start
        pool ask (fpgame19: the pool ask went out un-nonced)."""
        self._ask_nonce_seq = int(getattr(self, '_ask_nonce_seq',
                                          0)) + 1
        nid = self._ask_nonce_seq
        txt = '%s #%d' % (text, nid)
        la = getattr(self, '_live_asks', None)
        if la is None:
            la = self._live_asks = {}
        la[nid] = {'kind': str(kind), 'need': int(need or 0),
                   'text': txt,
                   'feed_mark': (len(self.driver.feed.lines)
                                 if feed_mark is None else feed_mark),
                   'epoch': self.driver.n_commits_total}
        return txt

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
            elif kind == 'action_roll':
                # the pool roll renders as LETTERS ('X rolls W P H H
                # .'); a d6 number line is never a pool result
                if op == 'roll_dice':
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
        staged = self._stage_human_costs_before_request(call)
        if staged is not None:
            return staged
        # The watchdog can reach this lane without re-entering _ai_decide.
        # Its request must respect the same complete physical cost barrier.
        if getattr(self, '_step_casualty_prefix', None) is not None and self.driver.echo_pending():
            return []
        from .automatic_combat_reports import awaiting_selected_card_reveal
        if awaiting_selected_card_reveal(self, call):
            return []
        gen = self._call_gen(call)
        ident = (call.kind, tuple(sorted(
            (k, str(v)) for k, v in call.kw.items())), gen)
        if ident in self._dirs_sent:
            return []
        self._dirs_sent.add(ident)
        from .automatic_combat_reports import before_roll
        out = before_roll(self, call)
        from .fixed_forfeit_reports import before_roll as fixed_before_roll
        out.extend(fixed_before_roll(self, call))
        from .anduril_reports import before_roll as anduril_before_roll
        out.extend(anduril_before_roll(self, call))
        if call.kind == 'd6':
            out.append(self._say('prompt', chat_mod.roll_context(
                call, self.rand.human_side)))
        for d in self.rand.directives_for(call):
            if d.kind == 'chat':
                txt = d.text
                if call.kind == 'draws':
                    out.append(self._say('prompt', txt))
                    continue
                if not d.text.endswith('?'):
                    # FLC H5 (par5 g4#4 + g1#8): THIS lane (the open
                    # AI step's directives - siege assaults, card
                    # battles) never minted a nonce, so its asks
                    # printed 'roll 3 dice ' while every window ask
                    # carried '#N'.  One minting point: _nonce_ask.
                    txt = self._nonce_ask(
                        d.text, str(getattr(call, 'kind', '')),
                        int((getattr(call, 'kw', {}) or {})
                            .get('need', 0) or 0))
                out.append(self._say(
                    'random_ask' if d.text.endswith('?')
                    else 'roll_request', txt))
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
        from .pending_roll_wait import remember_issued
        remember_issued(self, call, out)
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
        # Declare a paid siege continuation before moving its paying Elite.
        # The public log must distinguish that cost from ordinary casualties.
        out = [self._say(k, t) for k, t in em.chat if k == 'press']
        out.extend(em.lines)
        out.extend(em.calls)
        for k, t in em.chat:
            if k != 'press':
                out.append(self._say(k, t))
        # BRLANE-7 (BL-26 re-assert) + BRLANE-13 (expected echo): every
        # op line the AI emits is remembered and awaited
        self.driver.note_own_emission(em.lines)
        self.driver.expect_echo(em.lines)
        return out

    # ------------------------------------------------------------------
    def _ai_decide(self, dp):
        """One journal-pinned, clone-applied, once-emitted AI step."""
        if self.driver.forced_ai_rows_pending() or self.driver.human_physical_work_pending():
            return []
        if dp.get('kind') in ('combat_effect_choice','combat:combat_effect_choice'):
            from .automatic_combat_reports import current_ai_card_requires_reveal
            if current_ai_card_requires_reveal(self):
                return []
        if dp['kind'] == 'action' and self.driver.action_completion_pending(
                self._resuming_owed_dice()):
            return []
        import ai
        from ai_policies import project_ai_actions, action_policy_key
        arrival_step = getattr(self, '_step_dead_men_arrival', None)
        if arrival_step is not None and self.driver.echo_pending():
            return []
        if getattr(self, '_step_casualty_prefix', None) is not None and self.driver.echo_pending():
            return []
        self.rand.begin_attempt()
        # Rebuild the full legal menu here: a reduced caller-supplied menu
        # must not hide an available Elite or bonus draw from the policy.
        with self.rand.pin():
            legal = list(ai.get_legal_actions(self.driver.es, dp['side']))
        supplied = list(dp['actions'])
        projected = project_ai_actions(self.driver.es, dp['side'], legal + supplied)
        survivors = {id(a) for a in projected.actions}
        actions = [a for a in supplied if id(a) in survivors]
        if not actions:
            raise DriverHalt('AI policy rejects every supplied action')
        dp = dict(dp, actions=actions)

        def policy_permits(action):
            result = project_ai_actions(self.driver.es, dp['side'], legal + [action])
            return any(a is action for a in result.actions)

        def chooser(actions):
            if self.script is not None:
                # corpus replay: the recorded player's line decides
                return self.script.rank(self, dp, actions)
            if self.own_net is None:
                return list(actions)        # netless harness mode
            if len(actions) == 1:
                # A reduced caller menu is not proof of a forced choice. Only
                # bypass search when the complete legal policy menu agrees.
                complete = project_ai_actions(self.driver.es, dp['side'], legal).actions
                only_key = action_policy_key(actions[0])
                if complete and all(action_policy_key(row) == only_key for row in complete):
                    return list(actions)
            publish = getattr(self, '_publish_status', None)
            if callable(publish):
                from .decision_prompts import decision_label
                publish(_enrich=False, state='thinking',
                        waiting_on='AI choosing %s, %d simulations' % (
                            decision_label(dp['kind']), int(self.sims)),
                        human_menu={'side': self.human_side.name,
                                    'phase': self.driver.phase,
                                    'turn': int(getattr(self.driver.es, 'turn', 0)),
                                    'pending': str(dp['kind']), 'to_act': dp['side'].name,
                                    'alternation': dp['side'].name,
                                    'n_actions': 0, 'actions': [], 'by_type': {},
                                    'truncated': False})
            # Retain the actual pre-choice state and scores; the live mirror
            # is overwritten after every move and cannot explain a past pick.
            from . import search_receipts
            try:
                search_before = search_receipts.capture(self.driver.es)
            except Exception as error:
                search_before = None
                self.notes.append('search receipt capture failed: %s' % error)
            act, _pol = self.own_search.choose(
                self.driver.es, dp['side'], num_simulations=self.sims)
            ordered = list(actions)
            if act is None:
                # the search returned nothing for a non-empty engine
                # menu: a degraded decision is never taken silently
                raise DriverHalt(
                    'MCTS returned no action for the %s %s decision '
                    '(%d legal) - refusing to fall back to menu[0]'
                    % (dp['side'].name, dp['kind'], len(ordered)))
            if search_before is not None:
                try:
                    search_receipts.save(
                        self.client_dir, self.driver.n_commits_total,
                        dp['side'], dp['kind'], self.sims,
                        search_before, act, _pol)
                except Exception as error:
                    self.notes.append('search receipt write failed: %s' % error)
            if not policy_permits(act):
                raise DriverHalt('MCTS returned an action forbidden by the AI policy')
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
        if not policy_permits(act) or action_policy_key(act) not in {
                action_policy_key(a) for a in dp['actions']}:
            raise DriverHalt('AI journal returned an action outside the current policy menu')
        if arrival_step is not None and arrival_step['action'] != action_policy_key(act):
            raise DriverHalt('Dead Men arrival cannot resume a different chosen option')
        try:
            with self.rand.advance():
                from .casualty_randomness import apply_observed_casualty, is_round_unit_payment
                if act.action_type == 'casualty_allocate' or is_round_unit_payment(self.driver.es, act):
                    post = apply_observed_casualty(self.driver.es, act)
                else:
                    post = ai.apply_action(self.driver.es, act)
        except rnd_mod.NeedClientCall as e:
            e._physical_base_state = self.driver.es
            e._physical_expressed = tuple(self._step_expressed or ())
            out = []
            staged = self._stage_chosen_casualties_before_roll(dp, act, e)
            if staged is not None:
                return staged
            if e.kind == 'd6' and arrival_step is None:
                try:
                    prologue = emission.dead_men_arrival_prologue(
                        self.driver.view, self.nick, dp['side'], act)
                except (LookupError, ValueError) as error:
                    raise DriverHalt('EMISSION HOLD (Dead Men arrival): %s' % error)
                if prologue is not None:
                    arrival, arrival_emission = prologue
                    self._step_dead_men_arrival = {
                        'action': action_policy_key(act), 'pre': arrival}
                    self._step_expressed = list(arrival_emission.ops)
                    # Send only movement/politics now. The ordinary echo gate
                    # must acknowledge the complete group before the retry
                    # requests dice; the journal keeps the same chosen option.
                    return self._emission_out(arrival_emission)
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
                # K7: the random-casualty decision's victim IS the
                # client's draw - nothing to render before it; the
                # post-apply render names the companion that left
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
                # the pre-apply emission's ops survive the cross-tick
                # retry so the post-commit sweep can subtract the rows
                # this emission already expressed
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
        from .crebain_gestures import resumed_transport
        crebain_transport = resumed_transport(self.driver, act, post)
        if not self._em_sent:
            if crebain_transport is not None:
                em = self._render(dp, act, post=post, pre_override=crebain_transport['pre'])
            else:
                em = self._render(dp, act, post=post)
            if em.hold:
                if str(dp['kind']) in ('card_option', 'card_followup'):
                    # FLA-17 / BL-105 family (par5 g3#12, the Entmoot
                    # Huorns loop): an unrenderable AI card_option
                    # must not re-raise every tick (DriverHalt loop,
                    # cycle_errors climbing, rollover wedged).
                    # Degrade: commit with NO lines, SAY the pick,
                    # and let post_commit_sweep render/demand the
                    # residual board rows (its holds are loud, never
                    # a halt loop).
                    self.driver.findings.append(
                        ('emission_hold_degraded',
                         '%s :: %s' % (dp['kind'],
                                       str(em.hold)[:140])))
                    self.notes.append('EMISSION HOLD degraded to '
                                      'say-pick: %s'
                                      % str(em.hold)[:140])
                    out.append(self._say(
                        'prompt', chat_mod.pick_committed_note(
                            self.driver._pick_desc(act))))
                    em = emission.Emission()
                else:
                    raise DriverHalt('EMISSION HOLD: %s' % em.hold)
            from .corsairs_presentation import defer as defer_corsairs
            defer_corsairs(self.driver, act, self.driver.es, post, em)
            out.extend(self._emission_out(em))
            self._note_cc_chosen(em)
            for _c in em.calls:
                self._step_calls[_c] = self._text_gen(_c)
            expressed_ops.extend(em.ops)
            self._note_owed_die_kill(dp, act)
            self._note_owed_card_kill(dp, act)
        # an IMMEDIATE-effect play (no pending chain opened) applies
        # its board effect engine-side with no gesture: the bounded
        # card-effect row renderer (the sanctioned rule-6 exception)
        # covers exactly this delta; chains that DID open keep
        # rendering per decision
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
        out.extend(self._battle_announce_if_new(pre, post, act))
        # a declined OPTIONAL card leg is stated in chat (the board
        # cannot show a choice NOT taken)
        try:
            _dl = chat_mod.declined_leg(act, dp.get("actions"))
        except Exception as _dle:
            _dl = None
            self.notes.append('declined-leg chat failed: %r' % (_dle,))
        if _dl:
            out.append(self._say('leg_decline', _dl))
        # outbound symmetry (USER 2026-08-24): the AI's OWN Nazgul
        # Strike either/or pick is said with the same words the human
        # would use
        _ch9 = (getattr(act, 'params', {}) or {}).get('choices')
        if isinstance(_ch9, dict):
            if _ch9.get('action') == 'hunt':
                out.append(self._say('hunt_roll', 'rolling for hunt'))
            elif _ch9.get('action') == 'discard_table_card':
                out.append(self._say('discarding', 'discarding card'))
        # G2: a combat round that resolved inside THIS apply announces
        # its hit counts (engine statement; one producer per commit
        # path)
        for k, t in emission.round_hits_chat(post,
                                             self.driver.human_side, pre=pre):
            out.append(self._say(k, t))
        # one producer per commit path: forced allocations / capture
        # markers this apply resolved beyond the decision's own
        # emission (the expressed ops subtract first)
        for item in self.driver.post_commit_sweep(
                pre, post, expressed=expressed_ops,
                gesture=crebain_transport['gesture'] if crebain_transport else None,
                label='ai:%s' % dp['kind'], actions=[act]):
            out.extend(self._dispatch(item))
        if crebain_transport is not None:
            self.driver._crebain_human_transport = None
        # A completed ordinary action may already have rendered its die kill.
        # Keep that exact line after automatic consequences (for example North
        # activation on entering an empty Old Forest Road), before relinquishing
        # the action. Pending/combat kills remain owned by their existing gates.
        if act.die_idx is not None and int(act.die_idx) >= 0:
            die_cleanup = emission.render_die_kill(
                self.driver.view, self.nick, dp['side'], int(act.die_idx))
            if not die_cleanup.hold:
                for line in die_cleanup.lines:
                    if line in out:
                        out.remove(line)
                        out.append(line)
        # USER 2026-08-24 auto-reconcile: the AI's own commit can move
        # engine-truth surfaces (its guide dies, politics advance) -
        # flag the quiet-tick reconciler here too
        self.driver._reconcile_due = True
        self.rand.step_commit()
        self._em_sent = False
        self._dirs_sent.clear()
        self._step_calls.clear()
        self._step_expressed = []
        self._step_dead_men_arrival = None
        self._step_casualty_prefix = None
        self._step_owed = []
        self._open_step = None
        return out

    def _is_table_card(self, a):
        """The engine's own persistent-card classifier
        (persistent_effects.create_persistent_card): a play-on-table
        card has a PersistentCardType; everything else is a one-shot
        event that leaves the slot with its die."""
        try:
            from persistent_effects import create_persistent_card
            p = getattr(a, 'params', None) or {}
            cid = str(p.get('card_id') or '')
            name = str(p.get('card_name') or '')
            if not name and cid:
                name = cid.split('_', 2)[-1].replace('_', ' ')
            if not name:
                return False
            pc = create_persistent_card(cid, name, self.human_side)
            return getattr(pc, 'card_type', None) is not None
        except Exception:
            return False

    def _gesture_reply_for(self, drv, es, menu):
        """USER 2026-08-25 (verbatim): "I want the tester AI to move
        pieces manually just like a human would."  The seat writes
        <client>/ai_gesture_request.json {token, i}; this answers with
        the client lines that row's decision renders to - the AI's own
        emitter (render_decision / render_action) on a PROBE apply
        (attempt-scoped randomness, never step-committed; nothing
        changes engine-side).  The seat performs the lines with its
        own nick; the driver's pending buffer matches them exactly as
        it would a human's drags.  A row that needs client randomness
        first, or that the emitter cannot render, answers with `hold`
        - the seat refuses loudly (a harness gap to build, never a
        'pick')."""
        import json as _j
        p = os.path.join(self.client_dir, 'ai_gesture_request.json')
        if not os.path.exists(p):
            return getattr(self, '_gesture_reply', None)
        try:
            with open(p, encoding='utf-8') as f:
                req = _j.load(f)
        except Exception as _re:
            req = {'_bad': repr(_re)[:120]}
        try:
            os.remove(p)
        except OSError:
            pass
        if not isinstance(req, dict) or '_bad' in req:
            # never silent: a request the server could not read is a
            # reply with a hold
            rep = {'token': None, 'i': None, 'lines': [], 'calls': [],
                   'chat': [], 'hold': 'unreadable gesture request: %r'
                   % ((req or {}).get('_bad') if isinstance(req, dict)
                      else req)}
            self.notes.append('gesture request unreadable: %s'
                              % rep['hold'])
            self._gesture_reply = rep
            return rep
        token = str(req.get('token'))
        rep = {'token': token, 'i': req.get('i'), 'lines': [],
               'calls': [], 'chat': [], 'hold': None}
        try:
            import ai
            i = int(req.get('i'))
            if not (0 <= i < len(menu)):
                raise IndexError('row %d not in the %d-row menu'
                                 % (i, len(menu)))
            a = menu[i]
            try:
                with self.rand.pin():
                    dp = (ai.describe_pending(es, self.human_side)
                          or ai.describe_pending(es))
            except Exception:
                dp = None
            k = str((dp or {}).get('kind', 'action'))
            self.rand.begin_attempt()
            try:
                with self.rand.advance():
                    post = ai.apply_action(es, a)
            except rnd_mod.NeedClientCall as e:
                rep['hold'] = ('needs client randomness first (%s) - '
                               'no physical form before it renders'
                               % getattr(e, 'kind', '?'))
                post = None
            if post is not None:
                em = self._render({'kind': k, 'side': self.human_side},
                                  a, post)
                rep['lines'] = list(em.lines or [])
                rep['calls'] = list(em.calls or [])
                rep['chat'] = [list(c) for c in (em.chat or [])]
                rep['hold'] = em.hold
                if not rep['lines'] and not rep['hold'] \
                        and not rep['chat']:
                    rep['hold'] = ('the emitter renders no client '
                                   'line for %s - no physical form'
                                   % getattr(a, 'action_type', '?'))
        except Exception as e:
            rep['hold'] = 'render failed: %r' % (e,)
        self.notes.append('gesture reply for row %s: %s'
                          % (req.get('i'),
                             rep['hold'] or '%d line(s)'
                             % len(rep['lines'])))
        self._gesture_reply = rep
        return rep

    def _render(self, dp, act, post, pre_override=None):
        k = str(dp['kind'])
        if k == 'declare' and str(getattr(act, 'action_type', '')) \
                == 'pass':
            # R2: the AI-as-FP DECLINES to declare - absence is the
            # choice (no board gesture); its 'fp ok' follows
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
                                            pre=(pre_override if pre_override is not None else
                                                 (getattr(self, '_step_dead_men_arrival', None)
                                                  or {}).get('pre', self.driver.es)))
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
        from .turn_bookkeeping import draws_ready
        if not draws_ready(self):
            return []
        self.rand.begin_attempt()
        clone = ai.clone_state(self.driver.es)
        # an EMPTY deck skips that draw legitimately (gl.draw_event_
        # card raises on an empty deck and only NeedClientCall is
        # caught)
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
        # Both turn-start draws precede declaration. draw_event_card only
        # marks needs_discard; open the actual decision before handing off.
        gl.open_hand_limit_discard(clone)
        self.driver._snapshot([], 'ai_draws', post=clone)
        self.driver.es = clone
        self.driver.hands.es = clone
        self.driver._turn_start_draws_open = False
        self.rand.step_commit()
        self._dirs_sent.clear()
        self._step_calls.clear()
        self._open_step = None
        due = ai.describe_pending(clone)
        if due is not None and due['kind'] == 'discard' and due['side'] is self.ai_side:
            if all(step[0] != 'ai_decide' for step in self._step_queue):
                self._step_queue.insert(0, ('ai_decide', due))
        if self.ai_side is gs.Side.FREE_PEOPLES and self.driver.phase == 'declare':
            if all(step[0] != 'ai_declare' for step in self._step_queue):
                self._step_queue.append(('ai_declare', None))
        return []

    def _ai_eyes(self):
        """Commit Shadow's chosen eyes; bookkeeping rolls after native acknowledgment."""
        if self.driver.phase != 'eyes':
            return self._abandon_step('eye allocation before Fellowship phase completion')
        import ai
        dp = ai.describe_pending(self.driver.es)
        if dp is not None and dp['side'] is not self.ai_side:
            # Preserve the existing timing: an earlier compulsory Guide
            # replacement is not the optional end-Fellowship nomination.
            self._open_step = ('ai_eyes', None)
            self._step_parked = True
            return []
        if dp is None:
            self.driver.es._eye_allocation_phase = True
            dp = ai.describe_pending(self.driver.es)
        if dp is None or not dp['actions']:
            self._open_step = None
            msg = 'ai_eyes: no eye menu (pool empty?)'
            self.notes.append(msg)
            print('  [note] %s' % msg)
            return []
        out = self._ai_decide(dp)
        if self._open_step is None and str(dp['kind']) != 'eye':
            # the AI's OWN preceding pending (its hand-limit discard)
            # committed - the eye decision is still owed: re-open
            self._open_step = ('ai_eyes', None)
            return out
        if self._open_step is None:          # the eye step committed
            self.driver.phase = 'roll'
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
        if drv.phase not in ('declare', 'fellowship_tail', 'guide'):
            # the phase moved on (a recover/resync) - nothing to declare
            return self._abandon_step('ai_declare after the phase moved')
        if drv.phase == 'fellowship_tail':
            for item in drv._resume_fellowship_tail():
                self.driver._out_queue.append(item)
        if drv.phase == 'guide' and drv.es.pending_guide_change is None:
            drv._complete_fellowship_guide()
            self._open_step = None
            return [self._say('fp_ok', 'fp ok')]
        with self.rand.pin():
            any_dp = ai.describe_pending(drv.es)
        if any_dp is not None and any_dp['side'] is not self.ai_side:
            # BRLANE-3 / M12: the park is LOUD - chat once per
            # (pending, epoch), a finding, the status line ranks the
            # human's decision first (_waiting_on). An unanswered choice
            # remains an ordinary wait; inconsistent gesture/transport work
            # retains its existing diagnostic and repair path.
            self._open_step = ('ai_declare', None)
            self._step_parked = True
            self._step_park_n = getattr(self, '_step_park_n', 0) + 1
            out = []
            _key3 = (str(any_dp['kind']), any_dp['side'].name,
                     drv.n_commits_total)
            if getattr(self, '_declare_park_said', None) != _key3:
                self._declare_park_said = _key3
                self._step_park_n = 1
                _what3 = str(any_dp['kind']).split(':')[-1].replace('_',
                                                                    ' ')
                drv.findings.append(('ai_declare_parked',
                                     '%s %s' % (any_dp['side'].name,
                                                any_dp['kind'])))
                self.notes.append('ai_declare parked: the %s\'s %s '
                                  'decision comes first'
                                  % (any_dp['side'].name, any_dp['kind']))
                out.append(self._say('prompt',
                                     chat_mod.pending_reminder(_what3)))
            if (self._step_park_n >= 25 and drv._repair is None
                    and not self._unanswered_human_choice(any_dp)):
                for it in drv._propose_repair(
                        'AI DECLARE PARKED %d ticks behind the %s\'s %s '
                        'decision: %s' % (self._step_park_n,
                                          any_dp['side'].name,
                                          any_dp['kind'],
                                          drv.stall_text()),
                        clears='pending'):
                    out.extend(self._dispatch(it))
            return out
        # the driver's own answerer arms the menu for the FP seat and
        # returns the declare dp when it is open
        dp = drv.ai_action_dp()
        out = []
        if drv.phase == 'guide':
            if dp is None or dp['kind'] != 'guide_change':
                self._open_step = ('ai_declare', None)
                return []
            out = self._ai_decide(dp)
            if self._open_step is not None or drv.es.pending_guide_change is not None:
                self._open_step = ('ai_declare', None)
                return out
            drv._complete_fellowship_guide()
            out.append(self._say('fp_ok', 'fp ok'))
            self._open_step = None
            return out
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
        if drv.phase == 'fellowship_tail' or drv.es.pending_guide_change is not None:
            self._open_step = ('ai_declare', None)
        else:
            drv._complete_fellowship_guide()
            out.append(self._say('fp_ok', 'fp ok'))
            self._open_step = None
        return out

    def _order_own_roll(self):
        """Join automatic pool bookkeeping; one owner prevents duplicate rolls."""
        if self.driver.phase in ('eyes', 'roll'):
            self.driver.es._eye_allocation_phase = False
            self.driver.phase = 'roll'
        return []

    def _recover(self, request=None):
        """Recover through native menus, observing either player's manual work."""
        from .turn_bookkeeping import request_recovery
        return request_recovery(self, request)

    def _flush_recovery_request(self):
        from .turn_bookkeeping import flush_recovery
        return flush_recovery(self)

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
            # Only the missing part of the pool returned on entry. Already
            # present duplicates do not authorize moving used numbered tiles.
            for pid, nm in lr.HUNT_TILE_PIECES.items():
                if sanitize(str((dump or {}).get(pid, ''))) == 'hunt pool':
                    f = _tile_face(nm)
                    pool_faces[f] = pool_faces.get(f, 0) - 1
            from game_state import SPECIAL_HUNT_TILES
            returned_faces = {_tile_face(t) for t in SPECIAL_HUNT_TILES.values()}
            returned_faces.add('eye')
            back = []
            for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
                zz = sanitize(str((dump or {}).get(pid, '')))
                if zz != 'used hunt tiles':
                    continue
                f = _tile_face(nm)
                if f in returned_faces and pool_faces.get(f, 0) > 0:
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

    def _census_aging_tick(self, why):
        """BL-125(1) (fpgame15 O4): the instrument that watches the
        instruments - a witness gated too long means the divergence
        counter is judging on stale data (the frozen census timestamp
        under a persisting counter).  Instrument health (error-class,
        BRCMP-9), never a divergence."""
        drv = self.driver
        from .pending_roll_wait import issued_waiting
        # Deferred consequences also block continuation through echo_pending,
        # but are not commands awaiting a native acknowledgement. A tail held
        # behind human casualties must retain the ordinary human-wait exemption.
        ai_ack_pending = bool(getattr(drv, 'echo_expect', None))
        if not ai_ack_pending and issued_waiting(self) is not None:
            # A deferred human card reveal can own the real roll request even
            # with no AI step open. Its unanswered nonce is a normal wait.
            self._census_gated125 = 0
            return
        if not ai_ack_pending and drv.human_physical_work_pending():
            # _run_open_step deliberately parks even an existing AI
            # continuation until the person finishes committed casualties.
            # Its presence does not transfer ownership of this physical wait
            # to the AI. An unacknowledged AI command still ages below.
            self._census_gated125 = 0
            return
        if self._open_step is None and not ai_ack_pending:
            import ai
            try:
                decision = ai.describe_pending(drv.es)
                human_choice = (decision is not None
                                and decision.get('side') is self.human_side)
                human_window = (getattr(drv.seg, 'window', None) is not None
                                and drv.deferred is None
                                and drv.chat_deferred is None)
                if (human_choice or human_window
                        or drv.human_physical_work_pending()):
                    # Census is intentionally held during human choices and
                    # partial gestures. Time spent making that choice is not
                    # evidence that the board reader has stopped working.
                    self._census_gated125 = 0
                    return
            except Exception:
                # An unreadable decision is not proof of a normal human wait.
                pass
        self._census_gated125 = getattr(self, '_census_gated125', 0) + 1
        if self._census_gated125 == 10:
            message = ('CENSUS AGING: the witness has not measured for '
                '10 cadences (%s) - the divergence counter may be '
                'stale' % why)
            old = getattr(drv, '_census_aging_alarm', None)
            if old in drv.alarms:
                drv.alarms.remove(old)
            drv.alarms.append(message)
            drv._census_aging_alarm = message
            drv.findings.append(('census_aging', 'gated x10: %s' % why))

    def _witness_cadence(self):
        """WAVE-J (ledger rank 9, W-01/W-02): the continuous
        engine-vs-client witness - every N cycles, the driver's census
        lane (quiescence gate -> BOARD/STAGING rows with reasons ->
        confirm window -> LOG -> ASK) plus the hand-identity / card /
        rules-invariant extras, a jsonl record, and a 3-strike
        ESCALATION alarm on the same non-empty BOARD row set (the
        donor's census_alert discipline) that RETRACTS when the board
        heals (BRCMP-3/-9).  Returns the lane's items."""
        drv = self.driver
        if self._open_step is not None:
            call = self.rand.pending()
            if (call is not None and call.kind == 'd6'
                    and rnd_mod._side_name(call.kw.get('side'))
                    == rnd_mod._side_name(self.human_side)):
                # A person may take minutes to roll. The speculative AI
                # action cannot be censused against the board until that
                # result arrives; this is an explicit wait, not failed
                # instrumentation. Never escalate it into a repair prompt.
                self._census_gated125 = 0
                self._census_jsonl({'rows': None, 'first': None,
                                    'held': 'awaiting human dice roll'})
                return []
            self._census_aging_tick('AI step open')
            return []
        try:
            if not drv.view.refresh_dump():
                return []
            from . import client_board
            extras = []
            # BL-100 + BL-102 witnesses: the IDENTITY-SPACE hand census
            # (count rows are blind to identity drift at equal sizes).
            # BL-100: a decoded client-hand cid missing from an engine
            # hand that holds a placeholder = STRANDED REVEAL, loud
            # alarm once per cid.  BL-102: cid-level census rows ride
            # the settle/escalation machinery.
            if not getattr(drv, 'spectator', False):
                _hu100 = self.human_side
                _hand100 = drv.es.hands[_hu100].hand
                _hz100 = ('fp hand'
                          if _hu100.name == 'FREE_PEOPLES'
                          else 'sa hand')
                _cl100 = {str(p) for p, z in
                          (drv.view.piece_region or {}).items()
                          if str(z).strip().lower() == _hz100}
                # BL-131: ONE hand-truth answerer - the BL-95 menu
                # witness reads THIS snapshot, never its own read
                _map100 = drv.view.card_pid_map or {}
                _has_ph = any(str(c).startswith('HU_')
                              for c in _hand100)
                _seen100 = getattr(self, '_strand_alarmed100', set())
                for _cid100, _pid100 in _map100.items():
                    if str(_pid100) in _cl100 \
                            and _cid100 not in _hand100:
                        if _has_ph and _cid100 not in _seen100:
                            _seen100.add(_cid100)
                            drv.alarms.append(
                                'STRANDED REVEAL: %s decoded in the '
                                'client hand (pid %s) but the '
                                'engine hand holds a placeholder - '
                                'the card is off the legal surface'
                                % (_cid100, _pid100))
                        extras.append(
                            'hand cid %s: client holds it (pid %s), '
                            'engine hand does not'
                            % (_cid100, _pid100))
                self._strand_alarmed100 = _seen100
                if _cl100:
                    for _c102 in _hand100:
                        if str(_c102).startswith('HU_'):
                            continue
                        _p102 = _map100.get(str(_c102))
                        if _p102 is not None \
                                and str(_p102) not in _cl100:
                            extras.append(
                                'hand cid %s: engine holds it, '
                                'client pid %s not in the hand zone'
                                % (_c102, _p102))
            # AI-side identity witness from the CLIENT's own facts:
            # pids in the AI hand zone decoded via the commitment map
            # (fact 10), only when EVERY zone pid decodes.  The human
            # side is size-only (blinding: its identities are
            # structurally unknown).
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
            extras += client_board.card_census(
                drv.es, drv.view.piece_region,
                hand_cids=hand_cids) or []
            # BL-21(4): rules-invariant rows - agreed-wrong states
            # the piece diff cannot see still trip the witness
            extras += client_board.rules_invariant_rows(drv.es) or []
        except Exception as e:
            self.notes.append('witness cadence error: %r' % (e,))
            return []
        # the ONE census lane: quiescence gate, BOARD/STAGING split,
        # streak, retraction, LOG -> ASK (driver.census_tick)
        items = drv.census_tick(extra_rows=extras, source='witness')
        held = getattr(drv, '_census_held', None)
        if held:
            self._census_aging_tick(held)
            self._census_jsonl({'rows': None, 'first': None,
                                'held': str(held)[:160]})
            return items
        self._census_gated125 = 0
        lq = getattr(drv, '_last_quiescent_census', None) or {}
        rows = list(lq.get('board') or ())
        staging = list(lq.get('staging') or ())
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
                _old = getattr(drv, '_witness_alarm', None)
                if _old and _old in drv.alarms:
                    drv.alarms.remove(_old)
                _msg = ('WITNESS ESCALATION: %d row(s) settled over 3 '
                        'checks (first: %s)' % (len(rows), rows[0]))
                drv.alarms.append(_msg)
                # retracted by the census lane on a clean read
                drv._witness_alarm = _msg
        # the CURRENT census (rows, first) rides the status JSON so
        # the panel can say 'census: N row(s) diverging' even before
        # the 3-check escalation (USER 2026-08-25: 'CLEAN' beside
        # 'divergences 3 -> 3')
        self._last_census = (len(rows), str(rows[0]) if rows else None)
        self._census_jsonl({'rows': len(rows),
                            'first': (str(rows[0])[:160] if rows
                                      else None),
                            'staging': len(staging)})
        return items

    def _census_jsonl(self, rec):
        """One ai_census.jsonl record (t, turn + the caller's keys);
        a held read carries rows=None and the reason - never a fake
        clean 0."""
        drv = self.driver
        try:
            import json as _json
            import time as _time
            body = {'t': round(_time.time(), 1),
                    'turn': int(getattr(drv.es, 'turn', 0))}
            body.update(rec)
            with open(os.path.join(self.client_dir,
                                   'ai_census.jsonl'), 'a',
                      encoding='utf-8') as f:
                f.write(_json.dumps(body) + '\n')
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
            # / acknowledge my setting"): any value >= 1 is a request
            # (an 'exactly 1000 = default' guard would discard real
            # settings); `sims` is published back in status_menu so
            # the panel can acknowledge it.
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
                if getattr(self, '_cycle_failure', None):
                    from .cycle_recovery import manual_retry
                    manual_retry(self)
                else:
                    self._resync_requested = True
                self.notes.append('panel: resync requested')
            # SKIP DECISION (USER 2026-08-20: "I want it to be able to
            # skip the decision").  Deferred to the next handle() the
            # way resync is - a commit belongs on the tick, not inside
            # the transport's control read.
            sk = float(c.get('skip_pending', 0) or 0)
            if sk > getattr(self, '_skip_epoch', 0):
                self._skip_epoch = sk
                from .skip_recovery import arm as arm_skip
                arm_skip(self)
                self.notes.append('panel: skip-decision requested')
            # RESET + RESYNC (USER 2026-08-20: "a button to resync and
            # reset and clear all pending decisions, so the AI
            # continues from there").  Same tick-deferred pattern.
            rr9 = float(c.get('reset_resync', 0) or 0)
            if rr9 > getattr(self, '_reset_epoch', 0):
                self._reset_epoch = rr9
                from .cycle_recovery import manual_retry
                manual_retry(self)
                self.notes.append('panel: RESET + RESYNC requested')
            rw = float(c.get('rewind_turn', 0) or 0)
            if rw > getattr(self, '_rewind_epoch', 0):
                self._rewind_epoch = rw
                self._rewind_requested = True
                self._rewind_target = (c.get('rewind_checkpoint'), c.get('rewind_session'))
                self.notes.append('panel: turn rewind requested')
            resume = float(c.get('resume_rewind', 0) or 0)
            if resume > getattr(self, '_resume_rewind_epoch', 0):
                self._resume_rewind_epoch = resume
                from .turn_rewind import _state
                _state(self).review = False
            # WHOSE-TURN OVERRIDE (USER 2026-08-20 panel buttons): the
            # ENGINE owns the turn RULE (ai._next_acting_side,
            # untouched); this sets only the driver's bookkeeping of
            # WHO LAST ACTED, the single input that answerer takes from
            # us.  Epoch-stamped so one press acts once.
            wt = float(c.get('whose_turn_epoch', 0) or 0)
            if wt > getattr(self, '_whose_turn_epoch', 0):
                self._whose_turn_epoch = wt
                who = str(c.get('whose_turn', '')).lower()
                # AN OPEN DECISION IS NOT AN ALTERNATION PROBLEM (USER
                # 2026-08-20: "clear error -> your turn does not work,
                # it still waits for my decision"): these buttons move
                # last_actor, which governs only the PLAIN action
                # alternation; while the engine holds a pending,
                # ai_action_dp consults describe_pending FIRST and the
                # AI correctly refuses to act.  Say so instead of
                # silently doing nothing.
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
        * a card-added SPECIAL tile moves from Removed into Used before
          Mordor entry, or directly into Pool after entry, matching the
          engine's actual destination and the native physical inventory;

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
        hp = list(getattr(es, 'hunt_pool', []) or [])
        sig = tuple(tuple(sorted(_tile_face(t) for t in pile)) for pile in (hd, hr, hp))
        if sig != getattr(self, '_tile_sig', None):
            self._tile_sig = sig
            if hd or hr or hp:
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
        # Card-added specials stage in Used before Mordor, directly in Pool
        # afterward. Bind each move to a deficit and the actual native zone;
        # a historic "already staged" flag cannot establish current location.
        special_swept = getattr(self, '_tile_special_swept', None)
        if special_swept is None:
            special_swept = self._tile_special_swept = {}
        for pile, dest_name, dest_code in ((hd, 'used hunt tiles', 'Game.2045'),
                                           (hp, 'hunt pool', 'Game.2044')):
            want_sp = Counter(_tile_face(t) for t in pile if getattr(t, 'is_special', False))
            for f0, wanted in want_sp.items():
                have = sum(1 for pid, nm in lr.HUNT_TILE_PIECES.items()
                           if _tile_face(nm) == f0 and sanitize(str(dump.get(pid, ''))) == dest_name)
                need = wanted - have
                for pid, nm in sorted(lr.HUNT_TILE_PIECES.items()):
                    if need <= 0:
                        break
                    if _tile_face(nm) != f0:
                        continue
                    z = sanitize(str(dump.get(pid, '')))
                    allowed = {'removed hunt tiles'}
                    if dest_name == 'hunt pool' and es.fellowship.mordor_entered:
                        allowed.add('used hunt tiles')
                    if z not in allowed:
                        continue
                    key = (pid, z, dest_name)
                    n = special_swept.get(key, 0)
                    if n >= 3:
                        if n == 3:
                            self.notes.append('tile mop-up: special %s still unstaged after 3 sweep lines' % pid)
                            special_swept[key] = 4
                        continue
                    out.append(render('move', {'nick': self.nick, 'pieces': [pid],
                        'frm': 'Game.2046' if z == 'removed hunt tiles' else 'Game.2045',
                        'to': dest_code}, local=False))
                    special_swept[key] = n + 1
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
        # also kill its active die, which it neglected to do"): the
        # corpus ends the 1.149 table kill with `kills group (die,
        # card) in Current Action`
        if at in ('army_attack', 'play_event', 'draw_card',
                  'discard_table_card') \
                and didx is not None and int(didx) >= 0:
            import emitter
            # a PLAYED EVENT CARD dies WITH its die at chain close
            # (corpus E1 group-kill shape) - bank the card pid
            # alongside so the flush renders ONE group kill
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
        over. Bank (side, pid, cid) at emission; flush after its round's
        casualty and card effects resolve, before the next card selection.
        An authenticated table Balrog transfers to this combat cleanup lane:
        its native reveal relocates it from Board to Current SA Card."""
        at = getattr(act, 'action_type', None)
        params = getattr(act, 'params', {}) or {}
        if at == 'discard_table_card':
            # the DIE lane owns this one: _note_owed_die_kill banks the
            # card pid alongside the die (banking it here too would
            # kill the card twice)
            return
        if at != 'combat_card_play':
            return
        cid = params.get('card_id')
        if params.get('from_table'):
            from .automatic_combat_reports import table_balrog_commit
            pc = getattr(self.driver.es,'pending_combat',None) or {}
            role = 'atk' if pc.get('attacker_side') is dp['side'] else 'def'
            if not table_balrog_commit(self,cid,getattr(dp['side'],'name',''),
                    str(pc.get('attacker_region','')).casefold(),
                    str(pc.get('defender_region','')).casefold(),role):
                return
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

    def _flush_finished_combat_cards(self):
        """Discard resolved combat cards between rounds, with physical ACKs.

        Combat dice remain staged for the whole battle. A card stays until its
        casualty and post-casualty effects finish, even if rolls already ended.
        """
        drv = self.driver
        if (drv.human_physical_work_pending() or drv.echo_pending()
                or drv.pending_buf is not None or drv.deferred is not None
                or drv.chat_deferred is not None or self._open_step is not None):
            return []
        pc = getattr(drv.es, 'pending_combat', None)
        if pc is not None:
            completed = {'attacker_continue', 'defender_retreat', 'combat_round_result',
                         'attacker_combat_card', 'defender_combat_card', 'fp_shine',
                         'victor_advance'}
            if (int(pc.get('rounds_fought', 0)) < 1 or pc.get('phase') not in completed
                    or pc.get('_round_resume') or pc.get('_p9') or pc.get('effect_choice')):
                return []
        active = {pc.get(key) for key in ('atk_combat_card', 'def_combat_card')} if pc else set()
        from .combat_card_cleanup import ready_human_cards, retire_table_claim
        human_cards = ready_human_cards(self, active)
        out, retained = [], []
        import emitter
        for entry in dict.fromkeys(list(drv.owed_card_kills) + human_cards):
            side_name, pid, cid = entry
            if cid in active:
                retained.append(entry)
                continue
            retire_table_claim(self, pid, cid)
            line = emitter.render('kill', {'nick': self.nick, 'pieces': [str(pid)],
                                           'to': emitter.DICE_USED}, local=False)
            out.append(line)
            self.notes.append('resolved combat card %s (pid %s) discarded' % (str(cid)[:40], pid))
        drv.owed_card_kills = retained
        if out:
            drv.expect_echo(out)
        return out

    def _flush_owed_die_kills(self):
        """Flush owed die kills once the chain is CLOSED: no open AI
        step, no engine pending, nothing deferred.  (WAVE-H: nothing
        called render_die_kill - the battle/play die stranded in
        Current Action forever and, worse, its unclosed segmenter
        window swallowed the human's next records.)"""
        drv = self.driver
        out = []
        if drv.forced_ai_rows_pending() or drv.human_physical_work_pending():
            return out
        # TABLE SEAT AT COMMIT (2026-08-24): the Board-slot seat must
        # NOT wait for chain closure (a long-lived pending would keep
        # it banked silently); seat as soon as the engine files the
        # card in persistent_cards and the AI step is closed; only the
        # DIE kill stays deferred (die -> play -> resolve -> kill)
        if self._open_step is None and drv.owed_die_kills:
            _seated = []
            for ent in drv.owed_die_kills:
                if len(ent) > 3 and ent[2] is not None \
                        and ent[3] is not None \
                        and self._cid_on_table(ent[3]):
                    out.append(self._table_seat_line(ent[0], ent[2],
                                                     ent[3]))
                    ent = (ent[0], ent[1])
                _seated.append(ent)
            drv.owed_die_kills = _seated
        out.extend(self._flush_finished_combat_cards())
        owed_cards = getattr(drv, 'owed_card_kills', None) or []
        if (not drv.owed_die_kills and not owed_cards) \
                or self._open_step is not None:
            return out
        if drv.deferred is not None or drv.pending_buf is not None \
                or drv.chat_deferred is not None:
            return out
        import ai
        if ai.describe_pending(drv.es) is not None:
            return out
        # an OPEN BATTLE is an unclosed chain even with no deciding
        # side (K1 reveal phase: hidden commits awaiting the client's
        # reveal / a refused reveal re-opening a window) - the battle
        # die stays staged until the engine closes pending_combat
        if isinstance(getattr(drv.es, 'pending_combat', None), dict):
            return out
        still = []
        for ent in drv.owed_die_kills:
            # entries: (side, die_idx[, card_pid[, cid]]): the played
            # card dies WITH the die (one E1 group kill),
            # clearing the Current-Card slot.  EXCEPTION: a card the
            # engine holds in persistent_cards STAYS on the table -
            # it seats into a Board slot at COMMIT (sweep above) and
            # dies in its slot when the effect expires (_table_mopup).
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
        if out:
            drv.expect_echo(out)
        return out

    def _stage_chosen_casualties_before_roll(self, dp, act, call):
        """Settle the complete deterministic prefix of the actual AI choice.

        The chosen casualty can trigger further forced losses in the same
        engine apply. Its stopped clone contains those costs before the real
        post-casualty roll or draw; rendering just the selected primitive
        without that clone also loses its Elite replacement source stamp.
        A genuine intervening choice returns normally from the engine and
        never enters this path.
        """
        if dp.get('side') is not self.ai_side:
            return None
        # A late Sorcerer draw can interrupt the final casualty or paid press
        # before apply returns. Only the current real request, owned by this
        # journal and based on this exact state, may supply its stopped clone.
        draw_prefix = (call.kind == 'draws'
                       and call is self.driver.pending_random_call()
                       and getattr(call, '_physical_request_owner', None) is self.rand
                       and getattr(call, '_physical_base_state', None) is self.driver.es)
        stopped = getattr(call, '_casualty_pre_request_state', None)
        if stopped is None and draw_prefix:
            stopped = getattr(call, '_physical_pre_request_state', None)
        normal_casualty = (act.action_type == 'casualty_allocate'
                          and str(call.kw.get('ctx', '')).startswith('post-casualty '))
        round_payment = bool(getattr(call, '_round_payment_prefix', False))
        draw_cost = (draw_prefix and
                     (act.action_type in ('casualty_allocate', 'siege_press',
                                          'casualty_preserve_units', 'casualty_preserve_elites')
                      or round_payment))
        if (stopped is None
                or not ((call.kind == 'd6' and (normal_casualty or round_payment)) or draw_cost)):
            return None
        staged = self._stage_human_costs_before_request(call, stopped)
        if staged is not None:
            return staged
        from ai_policies import action_policy_key
        from . import emission as em_mod
        marker = getattr(self, '_step_casualty_prefix', None)
        key = action_policy_key(act)
        if marker is not None and (marker['pre'] is not self.driver.es or marker['action'] != key):
            raise DriverHalt('Pre-roll casualty prefix cannot resume a different AI choice')
        tag = 'shadow' if self.ai_side.name == 'SHADOW' else 'fp'
        rows = em_mod.removal_rows(self.driver.es, stopped, tag)
        expressed = list(self._step_expressed or ())
        if expressed:
            prior, failed = em_mod.expressed_rows(self.driver.view, expressed, tag)
            if failed:
                raise DriverHalt('Cannot identify the AI step\'s preceding casualty emissions')
            rows = em_mod.subtract_rows(rows, prior)
        if not em_mod.rows_empty(rows):
            if self.driver.echo_pending():
                return []
            em, ledger = em_mod.render_forced_rows(self.driver.view, self.nick, self.ai_side,
                rows, self.driver.es, label='chosen and forced pre-roll casualties', post=stopped)
            if em.hold:
                raise DriverHalt('AI pre-roll casualties held: %s' % em.hold)
            if act.action_type == 'siege_press':
                em.chat.insert(0, ('press', 'press'))
            out = self._emission_out(em)
            if not out:
                raise DriverHalt('AI pre-roll casualty prefix rendered nothing')
            self._step_expressed = expressed + list(em.ops)
            self._step_casualty_prefix = {'pre':self.driver.es, 'action':key}
            self._em_sent = True
            self.driver.echo_ledger.extend(ledger)
            return out
        if marker is not None:
            if self.driver.echo_pending():
                return []
            return self._dir_lines(call)
        return None

    # ---- table-card lane (bounded) -----------------------------------
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
        from .table_recovery import sync_cleanup
        sync_cleanup(self)
        book = getattr(self, '_table_book', None)
        if book is None:
            book = self._table_book = {}
        used = {b[0] for b in book.values()}
        used.update(entry.get('slot') for entry in
                    getattr(self.driver, 'table_seats', {}).values())
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
        from .table_recovery import sync_cleanup
        sync_cleanup(self)
        out = []
        book = getattr(self, '_table_book', None) or {}
        combat_balrog_pids = {str(pid) for _,pid,cid in self.driver.owed_card_kills
                             if cid == 'SHADOW_Character_Balrog_of_Moria'}
        from .combat_card_cleanup import human_balrog_pids
        combat_balrog_pids.update(human_balrog_pids(self.driver))
        for pid in list(book):
            slot, cid = book[pid]
            if cid == 'SHADOW_Character_Balrog_of_Moria' and str(pid) in combat_balrog_pids:
                continue
            if not self._cid_on_table(cid):
                del book[pid]
                out.append(render('kill', {
                    'nick': self.nick, 'pieces': [pid],
                    'to': slot}, local=False))
                self.notes.append('table card expired: pid %s '
                                  'killed in %s' % (pid, slot))
        # SEATS WATCHED IN THE FEED - either player's (USER 2026-08-20:
        # "you can see where the human player places the table card, so
        # just keep track of that location and kill it there"):
        # _table_book holds only the AI's OWN seats;
        # driver.table_seats records the slot we watched a piece go
        # into (matched by NAME - the human's card has no
        # commitment-decoded cid)
        seats = getattr(self.driver, 'table_seats', None)
        if seats:
            booked = {p for p in book}
            for key in list(seats):
                ent = seats[key]
                slot = ent.get('slot')
                armed = ent.get('armed')
                pid = self.driver.table_seat_pid(key)
                if str(pid) in combat_balrog_pids:
                    continue
                if pid is None or pid in booked:
                    # no pid yet (the dump has not placed it) or the AI
                    # lane owns it - never guess a piece
                    continue
                on_table = self._name_on_table(key)
                if not armed:
                    # ARM ONLY (never mop): the engine must be seen
                    # holding this card first (USER 2026-08-20, Tom
                    # Bombadil "immediately killed ... clearly not a
                    # learned action": a card the engine had not
                    # registered yet must never be killed on the next
                    # tick)
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
            # FLA-16 (par5 g1#5 + g2#8): the AI's own turn-start
            # action (a hand-limit discard above all) clears the
            # engine's _declare_phase flag, and the menu then
            # published n_total=null while the declaration window was
            # still open.  Re-arm (idempotent, K2's one answerer)
            # before reading the pending.
            drv._ensure_declare_armed()
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
                elif drv.phase == 'guide' and dp['kind'] == 'eye':
                    # Core eyes are ready, but the human FP still owns the
                    # explicit phase-close signal; do not label it AI's turn.
                    to_act = gs.Side.FREE_PEOPLES.name
                    pend = None
                    menu = []
            elif drv.phase == 'actions':
                to_act = self.human_side.name
                with self.rand.pin():
                    menu = ai.get_legal_actions(es, self.human_side)
            # PUBLISH the plain-action alternation from the ONE engine
            # answerer (ai._next_acting_side over the driver's
            # last_actor - the same computation ai_action_dp uses).
            # None = unknown (no actor yet this turn / not the action
            # phase).  A consumer must never re-derive it from commit
            # labels.
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
            # the OPEN battle's public facts + the type-swap ledger, so
            # a harness playing the human seat renders casualty
            # gestures from the CURRENT position (battle state is
            # public - no hidden info)
            pc = getattr(es, 'pending_combat', None)
            if isinstance(pc, dict):
                out['combat'] = {
                    k: _json_safe(pc.get(k)) for k in (
                        'phase', 'attacker_region', 'defender_region',
                        'attacker_side', 'defender_side',
                        'deciding_side', 'alloc_stage', 'atk_hits',
                        'def_hits', 'is_siege', 'rounds_fought')}
            # the open HUNT's public facts (damage/phase) - the harness
            # reads the engine's own numbers, never a guessed default
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
            # per-option RESOLVED ROWS for the human's casualty menu -
            # the ENGINE's own allocation arithmetic
            # (exact operation plus forced continuation on a clone, diffed by
            # emission.removal_rows); nobody recomputes hits-to-pieces
            cas_rows = None
            _cas_context = emitter.casualty_context(es)
            if _cas_context is not None \
                    and menu is not None \
                    and dp is not None \
                    and dp['side'] is self.human_side:
                from . import emission as _em
                _c_reg, _c_sd, _c_hits = (_cas_context['region'],
                    _cas_context['side'], _cas_context['hits'])
                if _cas_context['kind'] == 'card':
                    out['card_casualty'] = {'region': str(_c_reg), 'hits': _c_hits}
                _hu_tag = ('fp' if self.human_side is
                           gs.Side.FREE_PEOPLES else 'shadow')
                cas_rows = {}
                if _c_sd is self.human_side:
                    for a in menu:
                        _params = emitter.casualty_action_params(a)
                        if _params is None:
                            continue
                        _identity = chat_mod.action_identity(a)
                        _cl = ai.clone_state(es)
                        _cost = ai._apply_casualty_operation(
                            _cl, _c_reg, _c_sd, _c_hits, _params)
                        ai._forced_casualty_allocation(
                            _cl, _c_reg, _c_sd, _c_hits - _cost)
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
                        cas_rows[_identity] = sorted(
                            _ser, key=lambda x: (x['tkey'],
                                                 x['region'],
                                                 x['d']))
            if menu is not None:
                # UNCAPPED (USER 2026-08-24 "there should be no
                # caps"): every action of every type is published
                acts = []
                by_type = {}
                _seen_rows6 = {}
                _pickmap96 = {}
                for i, a in enumerate(menu):
                    at = str(a.action_type)
                    by_type[at] = by_type.get(at, 0) + 1
                    entry = {
                        'i': i, 'action_type': at,
                        'die_idx': (int(a.die_idx)
                                    if a.die_idx is not None
                                    else None),
                        'params': _menu_params_pub(a.params)}
                    if at in ('card_option', 'combat_effect_choice') \
                            or (entry['die_idx'] in (None, -1)
                                and out.get('pending')):
                        # BL-43/BL-65: every PENDING row is pickable -
                        # the row IS the answer (identity lane); the
                        # driver's demand/echo machinery renders the
                        # physical follow-through (one answerer)
                        entry['params']['say_pick'] = i + 1
                        # BL-96: the pick number binds the ACTION THE
                        # HUMAN READ - the chat resolver commits by
                        # this identity, never by a re-enumeration's
                        # index
                        _pickmap96[i + 1] = chat_mod.action_identity(a)
                    if at == 'card_option':
                        # BL-86: the chain TERMINATOR is a card_option
                        # row whose choices carry a *_done key - marked
                        # legibly; the say-word stays 'done'
                        _ch86 = (a.params or {}).get('choices')
                        if isinstance(_ch86, dict) and any(
                                _ch86.get(k) for k in
                                chat_mod._LEG_TERMINATORS):
                            entry['params']['say_done'] = True
                    if at == 'elven_ring':
                        # BL-51: accepted-but-ignored is misleading -
                        # the waiver is side-wide by the USER's
                        # verbatim 2026-08-18 rule; no die is targeted
                        entry['params'].pop('target_die_idx', None)
                    if at in ('play_event', 'discard_card',
                              'combat_card_play'):
                        # the client pid rides along so the seat can
                        # drive the client's own chooser for the play
                        _cp9 = drv.view.card_pid_map.get(
                            str((a.params or {}).get('card_id')))
                        if _cp9 is not None:
                            entry['params']['pid'] = str(_cp9)
                    if at == 'play_event':
                        # USER 2026-08-25 ("cards are not getting
                        # cleaned up, they are remaining in the active
                        # card slot"): a played event card leaves the
                        # Current-Card slot WITH its die (corpus E1
                        # group kill) - except a play-on-table card,
                        # which is SEATED (BL-121).  The engine's own
                        # persistent classifier decides; the seat
                        # reads the flag.
                        entry['params']['table'] = bool(
                            self._is_table_card(a))
                    if cas_rows is not None \
                            and emitter.casualty_action_params(a) is not None:
                        entry['rows'] = cas_rows.get(chat_mod.action_identity(a))
                    # F1: byte-identical twins are ANNOTATED, never
                    # dropped (USER no-caps ruling); the SEAT collapses
                    # dup_of rows for display
                    _rk6 = (entry['action_type'], entry['die_idx'],
                            repr(sorted(
                                (entry.get('params') or {}).items(),
                                key=lambda kv: str(kv[0]))))
                    if _rk6 in _seen_rows6:
                        entry['dup_of'] = _seen_rows6[_rk6]
                    else:
                        _seen_rows6[_rk6] = i
                    acts.append(entry)
                out['n_actions'] = len(menu)
                out['by_type'] = by_type
                out['actions'] = acts
                out['truncated'] = False    # kept for readers; never true
                # USER 2026-08-25 PHYSICAL-ANSWER LANE: the seat asks
                # for the drags a pending row implies and performs
                # them as the human; the driver then matches them as a
                # real gesture.  Rendered by the SAME emitter the AI
                # uses for its own decisions, on a probe clone.
                try:
                    out['gesture_reply'] = self._gesture_reply_for(
                        drv, es, menu)
                except Exception as _ge:
                    out['gesture_reply'] = {'hold': 'gesture lane '
                                            'error %r' % (_ge,)}
                # BL-96: the published pick numbering is the CONTRACT
                # - stored for the driver's chat-time resolver (only
                # when this publication carried pick rows)
                if _pickmap96:
                    drv._pick_pub96 = _pickmap96
                # BL-95 witness: every published card
                # pid must be a member of the CLIENT hand - a menu
                # naming a card the hand does not hold is an
                # impossible state that must never survive one tick
                # silently
                try:
                    # Use the same current board projection as the census.
                    # A retained hand-only cache can outlive a manual resync
                    # while the census is held for another physical effect.
                    _hz95 = ('fp hand' if self.human_side.name
                             == 'FREE_PEOPLES' else 'sa hand')
                    _hand95 = {str(p) for p, z in
                               (drv.view.piece_region or {}).items()
                               if str(z).strip().lower() == _hz95}
                    # BL-103: TWO-READ persistence - a witness that
                    # can race a stale dump files only when seen on
                    # two consecutive publications
                    _pend95 = getattr(self, '_pid95_pending', set())
                    _now95 = set()
                    if _hand95:
                        for _e95 in acts:
                            _p95 = (_e95.get('params')
                                    or {}).get('pid')
                            if _p95 and str(_p95) not in _hand95:
                                _tok95 = (_e95['action_type'],
                                          str(_p95))
                                _now95.add(_tok95)
                                if _tok95 in _pend95:
                                    _r95 = ('menu_pid_not_in_hand',
                                            '%s pid=%s (persisted)'
                                            % _tok95)
                                    if _r95 not in drv.findings[-20:]:
                                        drv.findings.append(_r95)
                    self._pid95_pending = _now95
                except Exception:
                    pass
            # BL-30b: while a window/chat decision is rng-starved the
            # rest of this publication is STALE - say so
            out['deferred'] = bool(
                drv.deferred is not None
                or drv.chat_deferred is not None)
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
            # the FELLOWSHIP block (engine statements the
            # harness's fellowship gestures render from).  FP-human
            # only: the true pawn region is FP-hidden information and
            # must never ride the status file for a SHADOW human.
            # BL-38: the waiver state is published so the seat can
            # tell why the menu just widened (and which rows ride it)
            try:
                import game_logic as _gl38
                out['ring_waiver'] = bool(
                    _gl38.ring_waiver_active(es, self.human_side))
            except Exception:
                pass
            # F3: the eye minimum publishes so the seat can GATE its
            # roll
            try:
                if self.human_side is gs.Side.SHADOW \
                        and str(drv.phase) in ('draws', 'declare',
                                               'eyes'):
                    _cts3 = drv._eye_menu_counts()
                    out['eye_minimum'] = (min(_cts3) if _cts3 else 0)
                    out['eyes_allocated'] = int(
                        getattr(es, 'eyes_allocated', 0) or 0)
            except Exception:
                pass
            if self.human_side is not gs.Side.FREE_PEOPLES:
                # BL-36: the Shadow seat gets the PUBLIC
                # fellowship facts (corruption/guide/hidden are open
                # information on the physical board); the pawn's true
                # region stays FP-hidden
                fs36 = getattr(es, 'fellowship', None)
                if fs36 is not None:
                    out['fellowship'] = {
                        'corruption': int(getattr(fs36, 'corruption',
                                                  0) or 0),
                        'hidden': bool(getattr(fs36, 'hidden', True)),
                        'guide': _json_safe(getattr(
                            getattr(fs36, 'guide', None), 'name',
                            None)),
                        'progress': int(getattr(fs36, 'progress', 0)
                                        or 0)}
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
            # the OPEN echo-ledger entries - the engine's statement of
            # human-side removals a committed apply resolved with no
            # gesture yet.  The harness kills EXACTLY these; only the
            # human side's rows are published (the AI's own forced
            # rows share the ledger and are claimed by their
            # self-echo).
            hu_tag = ('fp' if self.human_side is gs.Side.FREE_PEOPLES
                      else 'shadow')
            out['echo_open'] = [
                {str(k): _json_safe(v) for k, v in e.items()}
                for e in drv.echo_ledger if e.get('side') == hu_tag]
            # BL-62: the engine-committed discard's owed hand kill
            _dop62 = getattr(drv, '_discard_owed_pid', None)
            if _dop62:
                out['discard_owed_pid'] = str(_dop62)
            # BL-120/Q2: the refused oversize roll's eye
            # drag-back, published so the seat can PERFORM it (the
            # demand chat alone had no performer and the game froze)
            _rro120 = getattr(drv, '_roll_restore_owed', None)
            if _rro120:
                import time as _t120
                if _t120.time() > float(_rro120[1]):
                    drv._roll_restore_owed = None
                else:
                    out['roll_restore_owed'] = int(_rro120[0])
            # BL-67: the human table card owed its Board seating
            _tso67 = getattr(drv, '_table_seat_owed', None)
            if _tso67:
                out['table_seat_owed'] = str(_tso67)
            # BL-76: the formed siege's owed garrison drags
            _sso76 = getattr(drv, '_siege_seat_owed', None)
            if _sso76:
                out['siege_seat_owed'] = str(_sso76)
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
                    # wedge-time visible
                    'stalled_seconds': round(
                        getattr(self, '_stalled', 0.0), 1),
                    # A-09: the game-end verdict, published
                    # so the harness/leg can detect the END (the old
                    # 'game over:' chat is whitelist-stripped)
                    'game_over': ({str(k): _json_safe(v)
                                   for k, v in drv.game_over.items()}
                                  if drv.game_over else None),
                    # the SERVER's own statement that the AI owns an
                    # open step / staged die (rule 3: no consumer
                    # derives state from a log scan when the owner can
                    # answer)
                    'ai_open': bool(
                        self._open_step is not None
                        or self._step_queue
                        or getattr(drv, 'owed_die_kills', None)),
                    'open_step': (self._open_step[0]
                                  if self._open_step else None),
                    'owed_die_kills': len(
                        getattr(drv, 'owed_die_kills', ()) or ()),
                    # deferral / pending-buffer state, visible to the legs
                    'deferred': drv.deferred is not None,
                    'pending_buf': (len(drv.pending_buf.records)
                                    if drv.pending_buf is not None
                                    else 0),
                    'n_findings': len(drv.findings),
                    # the newest census rows, visible to the legs (kind
                    # + 120 chars; identities never ride findings)
                    'last_findings': [
                        (str(k), str(d)[:120])
                        for k, d in drv.findings[-3:]],
                    'window_stats': dict(drv.window_stats)}
            # 'sims' is what the panel echoes back as its acknowledgement
            # ("thinking (N sims)"); nothing published it before, so the
            # slider looked ignored no matter what it was set to.
            import ai as _ai_pub
            from .recovery_status import current as current_recovery_status
            from .turn_rewind import status as rewind_status
            return {'human_menu': out, 'progress': prog,
                    'recovery': current_recovery_status(self),
                    'turn_rewind': rewind_status(self),
                    'sims': int(self.sims),
                    'enforce_human': bool(getattr(self.driver,
                                                  'enforce_human', True)),
                    'max_children': int(_ai_pub.MCTS_MAX_CHILDREN),
                    'chance_k': int(_ai_pub.CHANCE_K_MAX),
                    'scenario': self._scenario_status()}
        except Exception as e:
            from .recovery_status import current as current_recovery_status
            from .turn_rewind import status as rewind_status
            return {'human_menu': {'error': repr(e)[:200]},
                    'recovery': current_recovery_status(self),
                    'turn_rewind': rewind_status(self),
                    'scenario': self._scenario_status()}


def build(client_dir, side_str, sims=300, net_tag='auto', seed=None,
          load_nets=True, scenario=None, ai_password='ai',
          human_password=None, script=None, script_ai_nick=None):
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
                     seed=seed, load_nets=load_nets, scenario=scenario,
                     ai_password=ai_password, human_password=human_password,
                     script=script, script_ai_nick=script_ai_nick)
    # owner_needle: the substring that PROVES a pid on the lock file
    # is a live server of THIS stack (client fact 18: never bare
    # pid-exists).  status_extra: the human-menu publication (the
    # engine answering "what is legal" across the process boundary -
    # I1).  control_cb: the panel's non-pause keys (sims slider,
    # manual-edit hold, panel resync) reach the server.
    bridge = FileBridge(client_dir, owner_needle='live_server',
                        status_extra=srv.status_menu,
                        control_cb=srv.on_control)
    srv._publish_status = bridge.write_status
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
    # corpus replay (lab/corpus_replay/DESIGN.md): recorded passwords
    # + the recorded player's rendered log as the AI's decision script
    ap.add_argument('--ai-password', default='ai')
    ap.add_argument('--human-password', default=None)
    ap.add_argument('--script', default=None)
    ap.add_argument('--script-ai-nick', default=None)
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
                        scenario=scenario, ai_password=a.ai_password,
                        human_password=a.human_password,
                        script=a.script, script_ai_nick=a.script_ai_nick)
    # serve_handler: RESP lines + the waiting_on blocker text the
    # panel displays
    bridge.serve(srv.serve_handler)


if __name__ == '__main__':
    main()
