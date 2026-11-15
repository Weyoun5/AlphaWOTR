# -*- coding: ascii -*-
"""engine_play.py -- standalone playable WOTR console interface driven
DIRECTLY by the project's own rules engine (no Java client).

ONE-IMPLEMENTATION CONTRACT (USER directive 2026-08-14): this file is a
DRIVER, not a fork.  Every rule decision is delegated to the existing
engine surfaces -- nothing here re-implements rules:

    board.load_board / gs.setup_initial_game / game_logic.initialize_decks
    game_logic.roll_action_dice / allocate_eyes / draw_event_card
    ai.get_legal_actions              (the ONLY source of offered actions)
    ai._traced_execute -> ai._execute_action   (the ONLY mutator)
    ai._pending_decision_side / ai._turn_owner_after /
        ai._active_after_action   (side routing -- the same single-source-
        of-truth routers self-play and MCTS use; C02: alternation runs
        from the turn owner, ai.search_turn_owner scopes the AI's search)
    ai._auto_enter_mordor / ai._balrog_stationary_turn_start
    ai._safe_next_turn                (turn boundary)
    turn_start.steps                  (THE turn-start phases: discards ->
        declare -> Fellowship tail -> Guide -> eyes; the GUI watch mode
        and the legacy GUI run the same driver)
    game_logic.check_all_victory_conditions / should_game_end
    wotr_bridge/live/own_model.OwnModelSearch   (the AI's search: the
        live AI's own contract, one per AI seat; R17-GUI-ENGINEPLAY)
    ai.mcts_action_with_policy        (search='legacy_puct' only: the
        per-decision helper ai._run_single_episode uses internally;
        arena/eval config)
    ai.load_value_net                 (the running generation's nets:
                                       net_tags.auto_tags)

The turn/phase skeleton in _game_gen() mirrors ai._run_single_episode
block-for-block (each block is annotated with its source lines); the
only difference is that each decision is YIELDED to a controller (human
menu or MCTS) instead of always calling MCTS inline.  Updates to the
engine automatically transfer: there is no copied rules logic here.

Enforcement: the interface makes illegal play impossible --
  * only ai.get_legal_actions output is ever offered;
  * EngineSession.submit() refuses wrong-side submissions, bad indexes
    and stale actions (menus are regenerated from the live state, so a
    consumed die is never offered again);
  * side alternation is the engine's own: a side acts twice in a row
    only when the rules say so (opponent out of dice, pending decision
    routed to it, multi-step die actions, elven-ring free action,
    battle-attacker attribution) -- all decided by ai.py's routers.

Runs unchanged in the production tree and the AlphaWOTR release
bundle: everything resolves relative to this file's own location
(ai.py's _PROJECT_ROOT = parent of src/).
"""

import os
import sys

# Env BEFORE any torch import (Windows gotchas: '' makes torch report
# cuda with 0 devices -- use '-1'; KMP duplicate lib crash).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

HERE = os.path.dirname(os.path.abspath(__file__))
# insert only when absent (one copy per process; src/ module names are unique on sys.path)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import argparse
import random

import numpy as np

import board
import game_state as gs
from game_state import Side
import game_logic
import game_trace
import ai
import turn_start


FP = Side.FREE_PEOPLES
SH = Side.SHADOW


def _side_name(side):
    return 'FREE PEOPLES' if side is FP else 'SHADOW'


def _p(text=''):
    """ASCII-safe print (Windows console is cp1252; transcripts must be
    byte-stable under redirection)."""
    try:
        s = str(text)
    except Exception:
        s = repr(text)
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)


# A submission the engine refuses (wrong side / bad index / stale
# action).  State is guaranteed untouched when this is raised.  ONE
# class with the engine's own refusal (ai.IllegalPlay) so a driver
# catches every engine refusal under one name.
IllegalPlay = ai.IllegalPlay


class EngineHalted(RuntimeError):
    """P1-03: an exception raised INSIDE the game generator while it ran
    a submitted decision (an execution failure re-raised by _execute,
    'unfinished_turn' at the turn boundary, a Fellowship-tail error, an
    engine IllegalPlay).  The generator is closed and the session is
    HALTED (session.halt_error holds the original, chained as
    __cause__).  Deliberately NOT an IllegalPlay: the state is not
    untouched and the decision cannot be retried."""


# R17-GUI-ENGINEPLAY (row 75): the AI's search contract.
#   'own_value'   (default) the live AI's own search: one
#                 wotr_bridge/live/own_model.OwnModelSearch per AI seat
#                 (own_search.active, 2 reply panels x 2 belief worlds, only
#                 the seat's own model, temperature 0, no root noise, on the
#                 backend WOTR_LIVE_SEARCH_BACKEND names) - the AI a person
#                 plays or watches here is the one Play_vs_AI.bat serves.
#   'legacy_puct' the arena/eval route: ai.mcts_action_with_policy with the
#                 module's WOTR_OPPONENT_POLICY default (opponent replies
#                 from the other seat's net) at `temperature`.
SEARCH_CONTRACTS = ('own_value', 'legacy_puct')


def _live_own_model():
    """wotr_bridge/live/own_model.py under the package name the live server
    imports it by (python -m live.live_server, from wotr_bridge/).  Appended
    to sys.path, so no module already resolvable from src/ changes."""
    bridge = os.path.join(os.path.dirname(HERE), 'wotr_bridge')
    if bridge not in sys.path:
        sys.path.append(bridge)
    from live import own_model
    return own_model


def _sims_for_kind(kind, base):
    """Per-decision-class search budget -- the SAME scaling
    _run_single_episode applies (discard sims//4, declare sims//3,
    full sims otherwise).  Single implementation: the game generator
    uses it to stamp DecisionPoints and ai_pick re-evaluates it against
    the live base (sims_override) so a mid-game strength change takes
    effect at the NEXT AI decision."""
    base = int(base)
    if kind == 'discard':
        return max(20, base // 4)
    if kind in ('declare', 'guide', 'fellowship_tail'):
        return max(30, base // 3)
    return base


class DecisionPoint(object):
    """One decision the engine is waiting on.

    kind: 'discard' | 'declare' | 'eye' | 'action'
    side: the Side that owns the decision (engine-routed)
    actions: the FULL legal menu (ai.get_legal_actions output)
    note: display-only context (pending battle/hunt/card phase)
    sims/temp/noise: the exact search config _run_single_episode uses
        for this decision class (consumed by EngineSession.ai_pick; on
        the default 'own_value' route the session stamps temp 0 / no
        noise, the only values the live AI's search uses).
    turn_owner: the side whose die action is being resolved (C02, round
        18: ai._turn_owner_after); ai_pick scopes the search with it
        (ai.search_turn_owner) so a pending answered at the root routes
        on from that side.  None outside the action phase.
    """

    def __init__(self, kind, side, actions, note='', sims=100,
                 temp=0.0, noise=False, turn=0, turn_owner=None):
        self.kind = kind
        self.side = side
        self.actions = actions
        self.note = note
        self.sims = sims
        self.temp = temp
        self.noise = noise
        self.turn = turn
        self.turn_owner = turn_owner


class _ConsoleAnnotator(object):
    """Duck-typed TraceRecorder stand-in for ai._traced_execute: engine
    roll/tile sites (game_trace.emit_annotation) narrate to the console
    through the EXISTING describe_step renderer instead of snapshotting.
    MCTS lookahead never reaches it (ai._traced_execute scopes
    ACTIVE_TRACER to the single real execution)."""

    buffering = False

    def __init__(self, log, viewer=None):
        self._log = log
        # P1-01: the seat these lines are shown to (EngineSession's
        # human_side); None = spectator (AI vs AI), nothing masked
        self._viewer = viewer
        # P2-03: TraceRecorder's meta dict.  ai._traced_execute writes
        # meta['failure_action'] before re-raising an execution error;
        # without it that write raised AttributeError, which replaced
        # the real engine exception (log, VIOLATIONS, GUI/console).
        self.meta = {}
        # R2-S4-ENGINE-AI-TOOLS (rescan v067): lines held while a masked
        # combat-card commit executes (EngineSession._execute), so the
        # reveal is narrated before a round fought inside that execution
        self._held = None

    def begin_action_buffer(self):
        pass  # print immediately; console order = execution order

    def hold(self):
        self._held = []

    def release(self):
        held, self._held = (self._held or []), None
        return held

    def _emit(self, line):
        if self._held is not None:
            self._held.append(line)
        else:
            self._log(line)

    def record_action(self, state, side, action, note='', **kw):
        try:
            params = dict(action.params or {})
            if (action.action_type == 'card_drawn'
                    and self._viewer is not None
                    and params.get('side')
                    != game_trace._side_tag(self._viewer)):
                # P1-01 (1.261): a drawn card is its drawer's private
                # knowledge - the human sees the deck, never the card
                self._emit('    . T%s %s: card_drawn (%s deck, identity '
                           'hidden)' % (state.turn, params.get('side', '?'),
                                        params.get('deck', '?')))
                return
            step = {'kind': 'action', 'turn': state.turn,
                    'side': game_trace._side_tag(side),
                    'action_type': action.action_type,
                    'params': params}
            self._emit('    . ' + game_trace.describe_step(step))
        except Exception:
            pass


# ---------------------------------------------------------------------
# Action -> human-readable menu text (display only; reuses
# game_trace._card_display_name for card identity)
# ---------------------------------------------------------------------

def _card_name(params):
    try:
        return game_trace._card_display_name(params)
    except Exception:
        return str(params.get('card_id', '?'))


def _die_prefix(state, side, action):
    di = getattr(action, 'die_idx', -1)
    if di is None or di < 0:
        return ''
    try:
        face = state.dice[side][di].current
        letter = game_trace._face_letter(face) if face is not None else '?'
    except Exception:
        letter = '?'
    return '[die %d:%s] ' % (di, letter)


def _deck_phrase(card_id):
    deck = game_logic.card_deck_of(card_id)
    return '%s card' % deck if deck else 'card'


def _masked_private_body(state, side, action, viewer):
    """P1-01: the display body of `side`'s action when it names a card
    the `viewer` seat may not see; None when nothing is private to it.
    viewer None = spectator (AI vs AI): never masked.  Private:
    face-down hand discards and the 1.149 table-kill payment pair
    (1.261), and the attacker's combat card while it is committed face
    down (1.246 / USER 2026-08-16: identity hidden, type visible, until
    the defender's own commit reveals both - EngineSession logs that
    reveal).  The defender's commit reveals both, so it is not masked."""
    if viewer is None or side == viewer:
        return None
    t = action.action_type
    p = action.params or {}
    if t == 'discard_card':
        return 'Discard a %s (face down)' % _deck_phrase(p.get('card_id'))
    if t == 'combat_card_play' and not p.get('from_table'):
        pc = getattr(state, 'pending_combat', None) or {}
        if (pc.get('phase') == 'attacker_combat_card'
                and pc.get('attacker_side') == side):
            # R2-S4-ENGINE-AI-TOOLS (rescan v067): with no defender card
            # window (Grond / Fighting Uruk-hai round 1, no Companion) no
            # defender commit ever comes - the wording holds in both paths
            return ('Commit a combat card face down (%s; revealed before '
                    'the round is fought)'
                    % _deck_phrase(p.get('card_id')))
        return None
    if t == 'card_option':
        ch = p.get('choices') or {}
        if 'army_card' in ch and 'character_card' in ch:
            return ('Card option [%s]: pay with one Army card + one '
                    'Character card (both face down)' % _card_name(p))
    return None


def army_move_split_text(p):
    """The composition suffix of an army_move row's label, e.g.
    ' (2r/0e, +1 leader(s): Boromir, 1 nazgul)', or '' when the row names
    none.  ONE builder for describe_action and the legacy GUI's
    _describe_action (R3-COSMETIC-BRIDGE-TEXT, R2-E5-01-SURFACE FU3): since
    D5 (2026-10-22) one menu holds a row per named figure set, so a label
    without the split shows several different moves as the same line."""
    comp = []
    if p.get('n_regular') is not None or p.get('n_elite') is not None:
        comp.append('%sr/%se' % (p.get('n_regular', 0),
                                 p.get('n_elite', 0)))
    if p.get('take_leaders') and p.get('n_leaders') is not None:
        # R2-E5-01-SURFACE (D5 2026-10-22): a which-figure row names
        # its leader figures ('leader_set'); the count alone would show
        # several rows with one label
        _ls9 = p.get('leader_set')
        comp.append('+%s leader(s)%s' % (
            p['n_leaders'], (': ' + ', '.join(
                ('%s x%d' % (r[1], r[3]) if r[3] > 1 else str(r[1]))
                for r in _ls9)) if _ls9 else ''))
    elif p.get('take_leaders'):
        comp.append('+leaders')
    if p.get('n_nazgul'):
        comp.append('%s nazgul' % p.get('n_nazgul'))
    return (' (%s)' % ', '.join(comp)) if comp else ''


def die_discard_text(p):
    """The label of a die-discard 'skip' row (USER A2-06, to be ruling
    1.276; ai._die_discard_rows: params face=<ActionDieFace name>, side):
    R3-COSMETIC-BRIDGE-TEXT (R2-DICE-ENGINE FU6)."""
    face = str(p.get('face', '?'))
    member = getattr(gs.ActionDieFace, face, None)
    return 'Discard the %s die for no effect' % (
        member.value if member is not None else face)


def describe_action(state, side, action, viewer=None):
    """Compact human-readable line for a legal action (display only).
    `viewer`: the seat the line is shown to (None = spectator); card
    identities private to `side` are masked for any other seat
    (_masked_private_body)."""
    masked = _masked_private_body(state, side, action, viewer)
    if masked is not None:
        return _die_prefix(state, side, action) + masked
    t = action.action_type
    p = action.params or {}
    body = None
    if t == 'pass':
        body = 'Pass (no die spent; legal while holding fewer dice)'
    elif t == 'army_move':
        body = ('Move army %s -> %s' % (p.get('from', '?'), p.get('to', '?'))
                + army_move_split_text(p))
        if p.get('mouth_muster'):
            body += ' (Mouth: muster die as army)'
    elif t == 'skip':
        # R3-COSMETIC-BRIDGE-TEXT (R2-DICE-ENGINE FU6): the die-discard row
        # (USER A2-06, to be ruling 1.276; ai._die_discard_rows)
        body = die_discard_text(p)
    elif t == 'army_attack':
        body = 'Attack %s from %s' % (p.get('to', '?'), p.get('from', '?'))
    elif t == 'advance_political':
        n = p.get('nation')
        body = 'Advance %s politically' % getattr(n, 'value', n)
    elif t == 'muster_recruit':
        body = 'Muster %s in %s' % (p.get('type', '?'), p.get('region', '?'))
    elif t == 'muster_elite':
        body = 'Upgrade to elite in %s' % p.get('region', '?')
    elif t == 'muster_minion':
        body = 'Muster %s at %s' % (p.get('minion', '?'), p.get('region', '?'))
    elif t == 'draw_card':
        body = 'Draw %s card' % p.get('deck', '?')
    elif t == 'play_event':
        body = 'Play event: %s' % _card_name(p)
    elif t == 'discard_card':
        body = 'Discard %s' % _card_name(p)
    elif t == 'discard_table_card':
        body = 'Discard table card %s' % _card_name(p)
    elif t == 'fellowship_move':
        body = 'Move the Fellowship (hunt may trigger)'
    elif t == 'fellowship_hide':
        body = 'Hide the Fellowship'
        if p.get('strider_ability'):
            body += ' (Strider guide ability)'
    elif t == 'fellowship_declare':
        body = 'Declare the Fellowship at %s' % p.get('region', '?')
    elif t == 'eye_allocate':
        body = 'Allocate %s die/dice to the Hunt Box' % p.get('count', '?')
    elif t == 'companion_separate':
        who = p.get('companions') or p.get('companion', '?')
        body = 'Separate/place %s -> %s' % (who, p.get('to_region', '?'))
    elif t == 'companion_move':
        who = p.get('companions') or p.get('companion', '?')
        body = 'Move companion(s) %s -> %s' % (who, p.get('to_region',
                                                          p.get('to', '?')))
    elif t == 'nazgul_move':
        body = 'Move Nazgul -> %s' % p.get('to_region', p.get('to', '?'))
    elif t == 'mouth_move':
        body = 'Move the Mouth of Sauron -> %s' % p.get('to_region',
                                                        p.get('to', '?'))
    elif t == 'elven_ring':
        body = ('Use Elven Ring: die %s -> %s (free action)'
                % (p.get('target_die_idx', '?'), p.get('new_face', '?')))
    elif t == 'dwarven_ring':
        body = 'Use Dwarven Ring (%s)' % (p,)
    elif t == 'use_token':
        body = 'Use action token (%s)' % (p.get('kind', p),)
    elif t == 'crown_aragorn':
        body = 'Crown Aragorn (Will of the West)'
    elif t == 'transform_gandalf':
        body = 'Transform Gandalf the White at %s' % p.get('region', '?')
    elif t == 'choose_guide':
        body = 'Choose guide: %s' % p.get('companion', '?')
    elif t == 'card_option':
        ch = p.get('choices') or {}
        bits = ', '.join('%s=%s' % (k, v) for k, v in sorted(ch.items())
                         if not str(k).startswith('_'))
        body = 'Card option [%s]: %s' % (_card_name(p), bits or 'resolve')
    elif t == 'combat_card_play':
        body = 'Play combat card: %s' % _card_name(p)
    elif t == 'combat_card_none':
        body = 'Decline combat card'
    elif t == 'combat_effect_choice':
        body = ('Combat-card effect: count=%s (%s)'
                % (p.get('count', '?'), _card_name(p)))
        if p.get('target') is not None:
            # 1.254: WHICH figure the card effect takes
            body += ' target=%s' % (p.get('target'),)
    elif t == 'combat_continue':
        body = 'Continue the battle (another round)'
    elif t == 'combat_cease':
        body = 'Cease the attack'
    elif t == 'combat_retreat':
        body = 'Retreat to %s' % p.get('to', '?')
    elif t == 'combat_no_retreat':
        body = 'Stand and fight (no retreat)'
    elif t == 'retreat_into_siege':
        body = 'Retreat into the stronghold (siege)'
    elif t == 'siege_assault':
        body = 'Assault the stronghold (elites fight at 5+)'
    elif t == 'siege_press':
        body = 'Press the siege (spend another die to continue)'
    elif t == 'siege_no_press':
        body = 'Do not press the siege'
    elif t == 'siege_trim':
        body = 'Trim besieged army to 5 units (%s)' % (p,)
    elif t == 'victor_advance':
        body = 'Advance victorious army into %s' % p.get('to',
                                                         p.get('region', '?'))
        # R2-E5-01-SURFACE (D5 2026-10-22): the composition, incl. the
        # named figures / nation mix a which-figure row carries
        _cv9 = []
        if p.get('n_regular') is not None or p.get('n_elite') is not None:
            _cv9.append('%sr/%se' % (p.get('n_regular', 0), p.get('n_elite', 0)))
        if p.get('unit_set'):
            _cv9.append('units: ' + ', '.join(
                '%d %s %s' % (r[3], r[1], r[2].lower()) for r in p['unit_set']))
        if p.get('n_leaders') is not None:
            _cv9.append('+%s leader(s)%s' % (p['n_leaders'], (': ' + ', '.join(
                str(r[1]) for r in p['leader_set'])) if p.get('leader_set') else ''))
        elif p.get('take_leaders'):
            _cv9.append('+leaders')
        if p.get('n_nazgul'):
            _cv9.append('%s nazgul' % p.get('n_nazgul'))
        if _cv9:
            body += ' (%s)' % ', '.join(_cv9)
    elif t == 'victor_stay':
        body = 'Stay (do not advance after victory)'
    elif t == 'casualty_allocate':
        operation = {'remove_regular': 'Remove Regular',
                     'reduce_elite': 'Reduce Elite',
                     'remove_elite': 'Remove whole Elite'}.get(p.get('operation'), 'Invalid allocation')
        body = '%s (%s)' % (operation, getattr(p.get('nation'), 'value', p.get('nation', '?')))
    elif t == 'casualty_preserve_units':
        body = 'Casualty policy: preserve unit count (downgrade elites)'
    elif t == 'casualty_preserve_elites':
        body = 'Casualty policy: preserve elites (remove regulars)'
    elif t == 'shine':
        body = 'Reveal Ring-bearers to cancel hits (shine)'
    elif t == 'shine_none':
        body = 'Do not shine'
    elif t == 'rearguard_none':
        body = 'No rearguard'
    elif t == 'hunt_gollum_reveal':
        body = 'Gollum reveals himself (takes the hunt damage)'
    elif t == 'hunt_gollum_no_reveal':
        body = 'Gollum stays hidden (Fellowship takes the hunt)'
    elif t == 'hunt_take_corruption':
        body = 'Take the hunt damage as corruption'
    elif t == 'hunt_sacrifice_guide':
        body = 'Sacrifice the guide to absorb the hunt'
    elif t == 'hunt_sacrifice_random':
        body = 'Random companion absorbs the hunt'
    elif t == 'hunt_guide_separate':
        body = ('Separate guide %s to reduce the hunt'
                % p.get('companion', '?'))
    elif t == 'followup_done':
        body = 'Done (end card follow-up)'
    elif t == 'end_turn':
        body = 'End turn'
    if body is None:
        bits = ' '.join('%s=%s' % (k, getattr(v, 'value', v))
                        for k, v in sorted(p.items(),
                                           key=lambda kv: str(kv[0]))
                        if v is not None and not str(k).startswith('_'))
        body = '%s %s' % (t, bits) if bits else t
    return _die_prefix(state, side, action) + body


# ---------------------------------------------------------------------
# EngineSession: the enforced game state machine
# ---------------------------------------------------------------------

class EngineSession(object):
    """Engine-driven WOTR game with an enforced submission interface.

    The internal generator (_game_gen) mirrors ai._run_single_episode's
    turn structure; each decision is yielded as a DecisionPoint.  The
    ONLY way to act is submit(index, side), which refuses anything that
    is not the engine-routed side picking an index into the live menu.
    """

    def __init__(self, seed=None, mcts_sims=300, max_turns=40,
                 shadow_vp=10, temperature=None, add_noise=False,
                 selfplay_temps=False, log=None, human_side=None,
                 max_turn_decisions=10000, search='own_value'):
        # human_side: the seat a HUMAN plays (None = watch / AI-vs-AI).
        # Player-owned learned decisions on that seat are never taken
        # by a self-play hardcode (the gateway auto-declare).
        self.human_side = human_side
        if seed is None:
            seed = int.from_bytes(os.urandom(4), 'big')
        self.seed = int(seed)
        self.sims = int(mcts_sims)
        self.max_turns = int(max_turns)
        if type(max_turn_decisions) is not int or max_turn_decisions < 1:
            raise ValueError('max_turn_decisions must be a positive integer')
        self.max_turn_decisions = max_turn_decisions
        # R17-GUI-ENGINEPLAY (row 75): see SEARCH_CONTRACTS.  The live
        # AI's search always runs at temperature 0 with no root noise, so
        # 'own_value' refuses any other setting instead of stamping
        # DecisionPoints with values its search never uses.
        if search not in SEARCH_CONTRACTS:
            raise ValueError('unknown search contract %r (one of %s)'
                             % (search, ', '.join(SEARCH_CONTRACTS)))
        self.search = search
        if search == 'own_value':
            if ((temperature is not None and float(temperature) != 0.0)
                    or add_noise or selfplay_temps):
                raise ValueError(
                    "search='own_value' is the live AI's search: temperature "
                    "0, no root noise and no self-play temperature schedule "
                    "(search='legacy_puct' takes those)")
            temperature = 0.0
        elif temperature is None:
            temperature = 0.02
        self._seat_searches = {}    # Side -> OwnModelSearch ('own_value')
        self.temperature = float(temperature)
        self.add_noise = bool(add_noise)
        self.selfplay_temps = bool(selfplay_temps)
        self.log = log if log is not None else (lambda s: None)
        # P1-01: the log is shown to the human seat - mask the AI's
        # private card identities (spectator: human_side None, unmasked)
        self._annotator = _ConsoleAnnotator(self.log, viewer=human_side)
        # (side, card_id) of an AI attacker's combat card masked at its
        # face-down commit, until the reveal is logged (_execute)
        self._masked_commit = None
        self.result_info = None
        # P1-03: the exception that closed the game generator (None while
        # the game runs).  A halt is not a game result: result_info stays
        # None and current() is None.
        self.halt_error = None
        self.decisions = 0
        self.action_log = []      # compact executed-action lines
        self._nets = None         # (fp_net, sh_net) once attached
        # live strength control (GUI slider): when set, ai_pick uses
        # this as the BASE sims for the next decision (per-kind scaling
        # preserved via _sims_for_kind).  None = the constructed --sims.
        self.sims_override = None
        self.last_pick_sims = None  # sims actually used by the last ai_pick

        # Determinism: seed every stream the engine/search draws from
        # (run with PYTHONHASHSEED=0 for full determinism, see the .bat)
        random.seed(self.seed)
        np.random.seed(self.seed % (2 ** 32))
        if getattr(ai, 'HAS_TORCH', False):
            ai.torch.manual_seed(self.seed)
            try:
                ai.torch.set_num_threads(1)
            except Exception:
                pass

        # Episode-boundary hygiene, exactly as _run_single_episode:
        # the single-shot tree-reuse cache must never carry a previous
        # game's subtree into this one (ai.py lines ~14633).
        ai._TREE_CACHE['child'] = None
        ai._TREE_CACHE['sig'] = None

        game_logic.SHADOW_VP_VICTORY = int(shadow_vp)

        # Game setup -- the exact _run_single_episode sequence
        # (ai.py ~14643-14654): board, initial state, decks, and the
        # turn-1 draw of 1 Character + 1 Strategy card per player.
        b = board.load_board()
        self.state = gs.setup_initial_game(b)
        game_logic.initialize_decks(self.state)
        for _side in (FP, SH):
            for _deck in ('character', 'strategy'):
                try:
                    game_logic.draw_event_card(self.state, _side, _deck)
                except Exception:
                    pass

        self._gen = self._game_gen()
        try:
            self._current = next(self._gen)
        except StopIteration:
            self._current = None

    # -- public interface ---------------------------------------------

    def current(self):
        """The DecisionPoint the engine is waiting on (None = game over)."""
        return self._current

    def game_over(self):
        return self._current is None

    def submit(self, index, side):
        """Submit a choice for the CURRENT decision.  Refuses (raises
        IllegalPlay, state untouched) anything but the engine-routed
        side picking a valid index into the live legal menu."""
        dp = self._current
        if dp is None:
            if self.halt_error is not None:
                raise IllegalPlay(
                    'game halted on an engine error (%s: %s) - no '
                    'decision is open' % (type(self.halt_error).__name__,
                                          self.halt_error))
            raise IllegalPlay('game is over - no decision is open')
        if side is not dp.side:
            raise IllegalPlay(
                'not your turn: this decision belongs to %s (a side may '
                'only act when the engine routes the decision to it)'
                % _side_name(dp.side))
        if isinstance(index, bool) or not isinstance(index, int):
            raise IllegalPlay('action index must be an integer, got %r'
                              % (index,))
        if index < 0 or index >= len(dp.actions):
            raise IllegalPlay(
                'illegal action index %d: the legal menu has %d entries '
                '(only actions from get_legal_actions are playable)'
                         % (index, len(dp.actions)))
        current_actions = ai.get_legal_actions(ai.clone_state(self.state), side)
        if dp.actions[index] not in current_actions:
            raise IllegalPlay('selected action is stale for the current state')
        self.decisions += 1
        try:
            self._current = self._gen.send(index)
        except StopIteration:
            self._current = None
        except BaseException as err:
            # P1-03: the exception closed the generator.  Its last
            # DecisionPoint must not stay open: a resubmit would hit
            # StopIteration and 'end' the game with result_info None.
            # Halt the session and fail loudly (never an IllegalPlay -
            # the state is not untouched and nothing can be retried).
            self._current = None
            self.halt_error = err
            if not isinstance(err, Exception):
                raise
            raise EngineHalted('engine error - session halted: %s: %s'
                               % (type(err).__name__, err)) from err

    def abandon(self):
        """Close the game generator (quit mid-game)."""
        try:
            self._gen.close()
        except Exception:
            pass
        self._current = None
        if self.result_info is None:
            self.result_info = {'winner': 'abandoned',
                                'victory_type': 'abandoned',
                                'turns': self.state.turn,
                                'message': 'game abandoned'}

    # -- AI controller ------------------------------------------------

    def attach_nets(self, fp_net, sh_net):
        # INV-RING-1 per seat (R2-MODEL-CANDIDATES, rescan v087): at an AI
        # root every learned leaf value comes from that seat's net
        # (ai._expand_node -> _mcts_eval_state), so a seat net whose final
        # value layer is all zero makes every leaf the same constant.
        # Refuse it for every seat the AI plays (watch mode: both); the
        # human's seat net only gives opponent priors and is never checked
        # (human-Shadow vs AI-FP still loads).  None nets are skipped.
        for _seat9, _net9 in ((FP, fp_net), (SH, sh_net)):
            if _net9 is not None and _seat9 is not self.human_side:
                ai.assert_state_dependent_value_head(
                    _net9, getattr(_net9, 'checkpoint_path', None)
                    or 'for the AI %s seat' % _side_name(_seat9))
        self._nets = (fp_net, sh_net)
        self._seat_searches = {}    # searches of the previous nets retire

    def prepare_search(self):
        """Build the search of every AI seat now, as the live server builds
        its OwnModelSearch at startup: a seat net the live search refuses
        (no net, a profile mismatch, trained self-play modules on the native
        route) or a native box that does not load fails before play starts,
        not at the first AI decision.  ai_pick builds a seat's search on
        demand when this was not called.  No-op for 'legacy_puct'."""
        if self.search != 'own_value':
            return
        for side in (FP, SH):
            if side is not self.human_side:
                self._seat_search(side)

    def _seat_search(self, side):
        """The seat's OwnModelSearch (the live AI's own contract), seeded
        from the session seed exactly as the live AI vs AI seats are
        (own_model.seat_seed), so a --seed game replays."""
        search = self._seat_searches.get(side)
        if search is None:
            if self._nets is None:
                raise RuntimeError('no nets attached (attach_nets first)')
            net = self._nets[0 if side is FP else 1]
            if net is None:
                raise RuntimeError('no net attached for the AI %s seat'
                                   % _side_name(side))
            own_model = _live_own_model()
            search = own_model.OwnModelSearch(
                side, net, seed=own_model.seat_seed(self.seed, side))
            self._seat_searches[side] = search
        return search

    def sims_for_current(self):
        """The simulation count ai_pick would use RIGHT NOW: the live
        base (sims_override when set, else the constructed --sims) with
        the current decision's per-kind scaling.  Single source for the
        GUI's 'sims=N' display/log."""
        dp = self._current
        if dp is None:
            return None
        base = (self.sims_override if self.sims_override is not None
                else self.sims)
        return _sims_for_kind(dp.kind, base)

    def ai_pick(self):
        """MCTS choice for the current decision.  'own_value' (default):
        the live AI's own search (the seat's OwnModelSearch.choose, as
        live_server calls it); 'legacy_puct': ai.mcts_action_with_policy
        with the per-phase temperature/noise _run_single_episode applies.
        The sims budget is re-read from sims_for_current() at call time,
        so a live sims_override takes effect at the NEXT decision.
        Returns a menu index for submit()."""
        dp = self._current
        if dp is None:
            raise RuntimeError('game is over')
        if self._nets is None:
            raise RuntimeError('no nets attached (attach_nets first)')
        fp_net, sh_net = self._nets
        n_sims = self.sims_for_current()
        self.last_pick_sims = n_sims
        with ai.search_turn_owner(dp.turn_owner):
            if self.search == 'own_value':
                action, _visit = self._seat_search(dp.side).choose(
                    self.state, dp.side, num_simulations=n_sims)
            else:
                action, _visit = ai.mcts_action_with_policy(
                    self.state, dp.side, fp_net, sh_net,
                    num_simulations=n_sims, temperature=dp.temp,
                    add_noise=dp.noise)
        idx = None
        if action is not None:
            for i, a in enumerate(dp.actions):
                if (a.action_type == action.action_type
                        and a.die_idx == action.die_idx
                        and a.params == action.params):
                    idx = i
                    break
            if idx is None:
                # live_server's B4-08 identity (R17-GUI-ENGINEPLAY): a row
                # differing only in optional UI labels (card_name, a card
                # option's idx) is the same choice - the native search
                # returns decoded rows that may lack those labels
                from ai_policies import action_policy_key
                key = action_policy_key(action)
                idx = next((i for i, a in enumerate(dp.actions)
                            if action_policy_key(a) == key), None)
        if idx is None:
            raise IllegalPlay('Search did not return a legal current decision')
        return idx

    # -- internals ----------------------------------------------------

    def _execute(self, action, side):
        """Execute the selected public action; errors stop the session."""
        st = self.state
        # P1-01 (1.246): an AI attacker's face-down combat card is masked
        # for the human; the defender's commit reveals both cards, so
        # name it as that commit is made (before the round's rolls)
        masks_commit = (action.action_type == 'combat_card_play'
                        and _masked_private_body(st, side, action,
                                                 self.human_side)
                        is not None)
        if (self._masked_commit is not None
                and action.action_type in ('combat_card_play',
                                           'combat_card_none')
                and (st.pending_combat or {}).get('phase')
                == 'defender_combat_card'):
            self._log_commit_reveal()
        if masks_commit:
            # R2-S4-ENGINE-AI-TOOLS (rescan v067): with no defender window
            # the round is fought INSIDE this execution; its rolls are held
            # and narrated after the reveal line below
            self._annotator.hold()
        try:
            ai._traced_execute(self.state, action, self._annotator)
        except Exception as err:
            for line in self._annotator.release():
                self.log(line)
            ai._record_exec_failure(self.state, action, err)
            self.log('! execution failed (%s: %s)' % (type(err).__name__, err))
            raise
        held = self._annotator.release() if masks_commit else []
        if masks_commit:
            self._masked_commit = (side, (action.params or {}).get('card_id'))
        if (self._masked_commit is not None
                and ai._hidden_commit_card(st, self.human_side) is None):
            # no defender window opened (e.g. a forced decline): the
            # round was fought, so the card is public now
            self._log_commit_reveal()
        for line in held:
            self.log(line)
        return action

    def _log_commit_reveal(self):
        side, cid = self._masked_commit
        self._masked_commit = None
        self.log("  %s attacker's combat card revealed: %s"
                 % ('FP' if side is FP else 'SH',
                    _card_name({'card_id': cid})))

    def _note_pending(self):
        """Display-only context line for the open pending phase."""
        st = self.state
        pc = st.pending_combat
        if pc is not None:
            reg = pc.get('region') or pc.get('defender_region') or '?'
            return ('battle at %s - phase %s'
                    % (reg, pc.get('phase', '?')))
        if st.pending_guide_change is not None:
            return 'choose a new Fellowship guide'
        if st.pending_hunt is not None:
            return ('hunt - phase %s (damage %s)'
                    % (st.pending_hunt.get('phase', '?'),
                       st.pending_hunt.get('damage', '?')))
        pco = getattr(st, 'pending_card_option', None)
        if pco is not None:
            return 'card choice: %s' % _card_name(pco)
        pcf = getattr(st, 'pending_card_followup', None)
        if pcf is not None:
            return 'card follow-up: %s' % _card_name(pcf)
        if getattr(st, 'pending_army_move_die', None) is not None:
            return 'second army movement of the Army die'
        if getattr(st, 'pending_muster_die', None) is not None:
            return 'second recruitment of the Muster die'
        return ''

    def _record(self, side, action):
        line = ('T%d %s %s' % (self.state.turn,
                               'FP' if side is FP else 'SH',
                               _compact_action(action)))
        self.action_log.append(line)

    def _finish(self, winner, vtype, message, turns):
        self.result_info = {'winner': winner, 'victory_type': vtype,
                            'turns': turns, 'message': message or ''}

    def _game_gen(self):
        """Turn/phase driver -- mirrors ai._run_single_episode
        (ai.py 14713-15061) with each decision yielded out.  Every
        structural predicate is the engine's own helper; comments cite
        the mirrored block."""
        state = self.state
        sims = self.sims
        game_over = False
        turn_count = 0

        while not game_over and turn_count < self.max_turns:
            turn_count += 1
            # Temperature schedule (episode 14716-14723): the arena/
            # eval override path is the default here (temperature_override)
            if self.selfplay_temps:
                if turn_count <= 3:
                    fp_temp = sh_temp = 0.6
                elif turn_count <= 8:
                    fp_temp = sh_temp = 0.3
                else:
                    fp_temp = sh_temp = 0.15
            else:
                fp_temp = sh_temp = self.temperature

            self.log('')
            self.log('========== TURN %d ==========' % state.turn)

            # Phases 1b-3 (episode 14727-14876): THE shared turn-start
            # driver (R3-LEGACY-GUI-TURN; the GUI watch mode and the
            # legacy GUI run the same one) - hand-limit discards ->
            # gateway auto-declare (self-play hardcode: never on a human
            # FP seat, whose declare DecisionPoint already offers the
            # gateway declare) -> the declare menu under the engine's own
            # _declare_phase flag (1.142c noise-free) -> Fellowship tail
            # (stationary Balrog, Almanac C17; turn-start Mordor entry)
            # -> Guide -> Shadow eye allocation under
            # _eye_allocation_phase (1.142 noise-free).  Each Choose is
            # yielded as a DecisionPoint; each Execute runs through
            # self._execute.  The driver is closed on every exit, so a
            # session closed mid-decision clears the phase flag.
            _ts = turn_start.steps(state,
                                   auto_declare=self.human_side is not FP)
            _ts_out = None
            _reply = None
            try:
                while True:
                    try:
                        req = _ts.send(_reply)
                    except StopIteration as _stop:
                        _ts_out = _stop.value
                        break
                    _reply = None
                    if isinstance(req, turn_start.Choose):
                        k, side = req.kind, req.side
                        side_temp = fp_temp if side is FP else sh_temp
                        if k == 'declare':
                            dp = DecisionPoint(
                                'declare', side, req.actions,
                                note='Fellowship phase - declare or pass '
                                     '(noise-free, argmax: 1.142c)',
                                sims=_sims_for_kind('declare', sims),
                                temp=0.0, noise=False, turn=state.turn)
                        elif k == 'eye':
                            dp = DecisionPoint(
                                'eye', side, req.actions,
                                note='allocate eyes to the Hunt Box before '
                                     'rolling (noise-free: 1.142)',
                                sims=sims, temp=0.0, noise=False,
                                turn=state.turn)
                        elif k == 'guide':
                            dp = DecisionPoint(
                                'guide', side, req.actions,
                                note='End Fellowship phase - keep or '
                                     'choose a highest-level Guide',
                                sims=_sims_for_kind('guide', sims),
                                temp=side_temp, noise=self.add_noise,
                                turn=state.turn)
                        elif k == 'discard':
                            dp = DecisionPoint(
                                'discard', side, req.actions,
                                note='hand over 6 cards - discard down',
                                sims=_sims_for_kind('discard', sims),
                                temp=side_temp, noise=self.add_noise,
                                turn=state.turn)
                        else:   # 'fellowship_tail' / an open 'decision'
                            dp = DecisionPoint(
                                k, side, req.actions,
                                note=self._note_pending(),
                                sims=_sims_for_kind(k, sims),
                                temp=side_temp, noise=self.add_noise,
                                turn=state.turn)
                        idx = yield dp
                        _reply = req.actions[idx]
                    elif isinstance(req, turn_start.Execute):
                        act = self._execute(req.action, req.side)
                        self._record(req.side, act)
                    elif req.kind == 'auto_declared':
                        self.log('  Fellowship declared at the gateway')
                    elif req.kind == 'mordor_entered':
                        self.log('  Fellowship enters Mordor (turn start)')
                    elif req.kind == 'eyes_allocated':
                        self.log('  Shadow allocates %d eye(s) to the '
                                 'Hunt Box' % req.detail['count'])
            finally:
                _ts.close()
            if _ts_out == 'game_over':
                game_over = True
                break

            # Phase 3: roll action dice (episode 14878-14879)
            game_logic.roll_action_dice(state, pool=FP)
            game_logic.roll_action_dice(state, pool=SH)
            self.log('  Dice: FP [%s]  SH [%s]  (hunt box: %d)'
                     % (_dice_faces(state, FP), _dice_faces(state, SH),
                        len(state.hunt_box)))

            def _unused(side):
                return sum(1 for d in state.dice.get(side, [])
                           if not d.used and d.current is not None)

            # Action resolution uses a diagnostic cap, never a victory trigger.
            # Long combat/casualty chains remain in this same action phase.
            active = FP
            # C02 (round 18): the side whose die action is being resolved
            # (episode twin _turn_owner9); answers to its pendings keep it
            owner = None
            _empty9 = set()   # MC6: sides whose menu came back empty
            for _ in range(self.max_turn_decisions):
                has_pending = (
                    state.pending_combat is not None
                    or state.pending_hunt is not None
                    or state.pending_guide_change is not None
                    or getattr(state, 'pending_card_option',
                               None) is not None
                    or getattr(state, 'pending_discard',
                               None) is not None
                    or getattr(state, 'pending_card_followup',
                               None) is not None
                    # USER RULING 2026-08-16: open reveal placement
                    # is an FP pending
                    or getattr(state, '_reveal_place_due', False))

                fp_left = _unused(FP)
                sh_left = _unused(SH)
                if fp_left == 0 and sh_left == 0 and not has_pending:
                    break

                if not has_pending:
                    if active is FP and fp_left == 0:
                        active = SH
                    elif active is SH and sh_left == 0:
                        active = FP
                    if _unused(active) == 0:
                        break

                # Single source of truth for pending routing (v13)
                _ps0 = ai._pending_decision_side(state)
                if _ps0 is not None:
                    active = _ps0

                actions = ai.get_legal_actions(state, active)
                if not actions:
                    # MC6 (rulebook 1681-1683, ai.py episode twin): a
                    # side with no legal action left hands the rest of
                    # the round to its opponent; the round ends only
                    # when NEITHER side can act
                    _oth9 = SH if active is FP else FP
                    _empty9.add(active)
                    if (not has_pending and _oth9 not in _empty9
                            and _unused(_oth9) > 0):
                        active = _oth9
                        continue
                    break
                _empty9.clear()

                temp = fp_temp if active is FP else sh_temp
                idx = yield DecisionPoint(
                    'action', active, actions,
                    note=self._note_pending(), sims=sims, temp=temp,
                    noise=self.add_noise, turn=state.turn,
                    turn_owner=owner)
                action = actions[idx]

                self.log('  %s: %s' % ('FP' if active is FP else 'SH',
                                       describe_action(
                                           state, active, action,
                                           viewer=self.human_side)))
                # C02: the turn owner is read off the PRE-action state
                owner = ai._turn_owner_after(state, active, owner)
                action = self._execute(action, active)
                self._record(active, action)

                # Victory timing (episode 15035-15040): ring victories
                # end the game immediately; military waits for the
                # end-of-turn Victory Check (rulebook p.17)
                vc = game_logic.check_all_victory_conditions(state)
                if game_logic.should_game_end(vc, end_of_turn=False):
                    game_over = True
                    break

                # Turn-order routing (the episode's ONE post-action router,
                # ai._active_after_action): ring use keeps the seat, a
                # multi-step die action its owner, a pending goes to its
                # decider, otherwise alternation from the TURN OWNER (C02)
                active = ai._active_after_action(state, action, active, owner)

            if game_over:
                break
            blockers = ai._turn_end_blockers(state)
            if blockers:
                raise RuntimeError('unfinished_turn: ' + ', '.join(blockers))
            if ai._safe_next_turn(state):
                # Resolve inactivity before the military Victory Check.
                game_over = True
                break

        # Outcome (episode 15081+, labels only -- no training targets)
        result = game_logic.check_all_victory_conditions(state)
        if result.get('victory') in ('free_peoples', 'shadow'):
            self._finish(result['victory'],
                         result.get('victory_type', '?'),
                         result.get('message', ''), turn_count)
        else:
            # 'timeout' stays as the MACHINE field (tests/tools key on
            # it); every human-facing string must say TURN CAP instead
            # (USER 2026-08-14: a bare 'timeout' banner from a capped
            # smoke read as a malfunction).
            self._finish('timeout', 'timeout',
                         'TURN CAP REACHED (--max-turns %d) - game '
                         'stopped undecided, NOT a real game ending'
                         % self.max_turns, turn_count)
        self.log('')
        if self.result_info['winner'] == 'timeout':
            self.log('=== GAME STOPPED: TURN CAP REACHED '
                     '(--max-turns %d) after %d turn(s) - no victory '
                     'decided ===' % (self.max_turns, turn_count))
        else:
            self.log('=== GAME OVER: %s (%s) after %d turn(s) ==='
                     % (self.result_info['winner'].upper(),
                        self.result_info['victory_type'], turn_count))
        if self.result_info['message']:
            self.log('    %s' % self.result_info['message'])


def _compact_action(action):
    p = action.params or {}
    bits = ' '.join(
        '%s=%s' % (k, getattr(v, 'value', v))
        for k, v in sorted(p.items(), key=lambda kv: str(kv[0]))
        if v is not None and not str(k).startswith('_')
        and k not in ('choices',))
    ch = p.get('choices')
    if isinstance(ch, dict) and ch:
        bits += ' choices={%s}' % ','.join(
            '%s=%s' % (k, ch[k]) for k in sorted(ch, key=str)
            if ch[k] is not None)
    return ('%s[d%s] %s' % (action.action_type, action.die_idx, bits)
            ).strip()


def _dice_faces(state, side):
    out = []
    for d in state.dice.get(side, []):
        if not d.used and d.current is not None:
            try:
                out.append(game_trace._face_letter(d.current))
            except Exception:
                out.append('?')
    return ' '.join(out) if out else '-'


# ---------------------------------------------------------------------
# Console front-end
# ---------------------------------------------------------------------

def _fellowship_line(state, show_secret):
    f = state.fellowship
    guide = getattr(f.guide, 'name', '?') if f.guide else 'none'
    if f.mordor_entered:
        where = 'MORDOR step %d/5' % f.mordor_step
    elif show_secret or not f.hidden:
        try:
            where = 'at %s' % game_logic.fellowship_region(state)
        except Exception:
            where = 'at ?'
    else:
        where = 'position hidden'
    return ('Fellowship: %s, %s, progress %d, corruption %d/12, '
            'guide %s, %d companions'
            % ('hidden' if f.hidden else 'REVEALED', where, f.progress,
               f.corruption, guide, len(f.companions)))


def print_header(state, human_side=None):
    vp = game_logic.calculate_victory_points(state)
    _p('-' * 66)
    _p('Turn %d | VP: Shadow %d/%d, FP %d/%d | Hunt box: %d'
       % (state.turn, vp['shadow'], game_logic.SHADOW_VP_VICTORY,
          vp['free_peoples'], game_logic.FP_VP_VICTORY,
          len(state.hunt_box)))
    _p('FP dice: [%s]   SH dice: [%s]   Elven rings: %d   '
       % (_dice_faces(state, FP), _dice_faces(state, SH),
          state.fp_elven_rings))
    # watch mode (no human) sees everything; a human playing SHADOW
    # must not be shown the hidden Fellowship's true position
    show_secret = (human_side is None) or (human_side is FP)
    _p(_fellowship_line(state, show_secret))
    _p('-' * 66)


def print_board_summary(state):
    _p('--- armies (r=regular e=elite L=leaders/nazgul) ---')
    for key in sorted(state.armies, key=lambda k: str(k)):
        a = state.armies[key]
        if not a.units and not a.leaders:
            continue
        fp_r = sum(1 for u in a.units
                   if ai._side_of_nation(u.nation) is FP
                   and u.kind is gs.UnitKind.REGULAR)
        fp_e = sum(1 for u in a.units
                   if ai._side_of_nation(u.nation) is FP
                   and u.kind is gs.UnitKind.ELITE)
        sh_r = sum(1 for u in a.units
                   if ai._side_of_nation(u.nation) is SH
                   and u.kind is gs.UnitKind.REGULAR)
        sh_e = sum(1 for u in a.units
                   if ai._side_of_nation(u.nation) is SH
                   and u.kind is gs.UnitKind.ELITE)
        nl = len(a.leaders)
        parts = []
        if fp_r or fp_e:
            parts.append('FP %dr/%de' % (fp_r, fp_e))
        if sh_r or sh_e:
            parts.append('SH %dr/%de' % (sh_r, sh_e))
        if nl:
            parts.append('%dL' % nl)
        _p('  %-28s %s' % (str(key)[:28], ' '.join(parts)))
    _p('--- political track (box 0-3, 3=at war; * = active) ---')
    for n in gs.Nation:
        try:
            box = state.political.boxes.get(n)
            if box is None:
                continue
            act = '*' if state.political.active.get(n) else ' '
            _p('  %-22s %d%s' % (getattr(n, 'value', n), box, act))
        except Exception:
            pass


def print_hand(state, side):
    hand = state.hands[side].hand
    _p('--- your hand (%d) ---' % len(hand))
    for cid in hand:
        _p('  %s  [%s]' % (_card_name({'card_id': cid}), cid))


def print_menu(state, dp):
    _p('')
    _p('DECISION for %s  (%s)%s'
       % (_side_name(dp.side), dp.kind,
          ('  -- ' + dp.note) if dp.note else ''))
    for i, a in enumerate(dp.actions):
        _p('  [%2d] %s' % (i + 1, describe_action(state, dp.side, a)))


HELP_TEXT = """  commands:
    <number>        play that menu entry
    t <action_type> play the first entry of that type (e.g. t fellowship_move)
    s               show board summary (armies, political track)
    h               show your hand
    ?               this help
    q               quit (asks to confirm)"""


def _human_choose(session, dp, human_side):
    """Read one choice from stdin.  Every prompt consumes exactly one
    input line (scripts stay aligned); invalid input re-prompts.
    Returns an index, or None on quit/EOF."""
    state = session.state
    print_header(state, human_side)
    print_menu(state, dp)
    while True:
        _p('your choice [1-%d] (? for help):' % len(dp.actions))
        try:
            raw = input().strip()
        except EOFError:
            _p('(stdin closed - abandoning game)')
            return None
        if not raw:
            continue
        low = raw.lower()
        if low == 'q':
            _p('really quit? [y/n]')
            try:
                conf = input().strip().lower()
            except EOFError:
                return None
            if conf == 'y':
                return None
            continue
        if low == '?':
            _p(HELP_TEXT)
            continue
        if low == 's':
            print_board_summary(state)
            continue
        if low == 'h':
            print_hand(state, dp.side)
            continue
        if low.startswith('t '):
            want = low[2:].strip()
            hit = None
            for i, a in enumerate(dp.actions):
                if a.action_type == want:
                    hit = i
                    break
            if hit is None:
                _p('REFUSED: no legal action of type %r in this menu'
                   % want)
                continue
            _p('  -> %s' % describe_action(state, dp.side,
                                           dp.actions[hit]))
            return hit
        try:
            k = int(raw)
        except ValueError:
            _p('REFUSED: %r is not a menu number (? for help)' % raw)
            continue
        if k < 1 or k > len(dp.actions):
            _p('REFUSED: %d is outside the legal menu [1-%d] - only '
               'listed legal actions can be played' % (k,
                                                       len(dp.actions)))
            continue
        return k - 1


def _resolve_net_tag(tag):
    if tag != 'auto':
        return tag
    # the deployed pair of the running generation first (a checkpoint
    # loads only under its own WOTR_ACT_GEN), then the legacy chain
    import net_tags
    for cand in net_tags.auto_tags(getattr(ai, '_OBSERVATION_GENERATION',
                                           '')):
        if os.path.exists(ai._resolve_model_path('wotr_fp_%s.pt'
                                                 % cand)):
            return cand
    return None


def load_nets(tag='auto'):
    """Load the packaged FP/SH nets via ai.load_value_net ('auto' =
    net_tags.auto_tags of the running generation).
    Fails LOUDLY -- no silent fallback policy."""
    tag = _resolve_net_tag(tag)
    if tag is None:
        raise SystemExit('no nets found next to the project root - '
                         'expected wotr_{fp,sh}_<tag>.pt for one of '
                         'the tags of this generation (src/net_tags.py)')
    fp_net = ai.load_value_net('wotr_fp_%s.pt' % tag, 'fp')
    sh_net = ai.load_value_net('wotr_sh_%s.pt' % tag, 'sh')
    if fp_net is None or sh_net is None:
        raise SystemExit('net load failed for tag %r (file missing or '
                         'torch unavailable)' % tag)
    # R2-MODEL-CANDIDATES: the per-seat INV-RING-1 refusal in
    # EngineSession.attach_nets names the file it refuses
    fp_net.checkpoint_path = ai._resolve_model_path('wotr_fp_%s.pt' % tag)
    sh_net.checkpoint_path = ai._resolve_model_path('wotr_sh_%s.pt' % tag)
    from reward_contract import current_contract, PROFILE, validate_net_profile
    if current_contract() == PROFILE:
        # Validate the complete pair before replacing an existing objective.
        profiles = {}
        for side, net in (('fp', fp_net), ('sh', sh_net)):
            profile = getattr(net, 'reward_profile', None)
            validate_net_profile(net, side, profile)
            profiles[side] = profile
        ai.PLAYER_PROFILES.update(profiles)
    _p('nets loaded: wotr_{fp,sh}_%s.pt' % tag)
    return fp_net, sh_net


def run_game(mode='hvai', human='fp', seed=None, sims=300, max_turns=40,
             shadow_vp=10, net_tag='auto', selfplay_temps=False,
             dump_actions=None, search='own_value'):
    human_side = None
    if mode == 'hvai':
        human_side = FP if human == 'fp' else SH
    session = EngineSession(seed=seed, mcts_sims=sims,
                            max_turns=max_turns, shadow_vp=shadow_vp,
                            selfplay_temps=selfplay_temps, log=_p,
                            human_side=human_side, search=search)
    _p('WOTR engine-play console -- engine is the sole arbiter')
    _p('mode: %s%s | seed: %d%s | sims: %d | max turns: %d'
       % (mode,
          (' (human=%s)' % _side_name(human_side)) if human_side else '',
          session.seed,
          ' (fixed via --seed)' if seed is not None
          else ' (randomly drawn this run)',
          sims, max_turns))
    fp_net, sh_net = load_nets(net_tag)
    session.attach_nets(fp_net, sh_net)
    session.prepare_search()

    while not session.game_over():
        dp = session.current()
        if human_side is not None and dp.side is human_side:
            idx = _human_choose(session, dp, human_side)
            if idx is None:
                session.abandon()
                break
            try:
                session.submit(idx, dp.side)
            except IllegalPlay as e:
                # unreachable through the menu loop (defense in depth)
                _p('REFUSED: %s' % e)
        else:
            _p('')
            _p('T%d %s thinking (%s, %d sims)...'
               % (dp.turn, _side_name(dp.side), dp.kind, dp.sims))
            if dp.note:
                _p('  context: %s' % dp.note)
            idx = session.ai_pick()
            _p('  AI picks: %s'
               % describe_action(session.state, dp.side,
                                 dp.actions[idx], viewer=human_side))
            session.submit(idx, dp.side)

    res = session.result_info or {}
    _p('')
    _p('RESULT: winner=%s type=%s turns=%s decisions=%d'
       % (res.get('winner'), res.get('victory_type'),
          res.get('turns'), session.decisions))
    if dump_actions:
        with open(dump_actions, 'w', encoding='ascii',
                  errors='replace') as fh:
            fh.write('\n'.join(session.action_log) + '\n')
        _p('action log written: %s (%d actions)'
           % (dump_actions, len(session.action_log)))
    return session


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Standalone WOTR console play over the rules '
                    'engine (no Java client).')
    ap.add_argument('--mode', choices=('hvai', 'watch'), default='hvai',
                    help='hvai = human vs AI (default); watch = AI vs '
                         'AI, every action printed')
    ap.add_argument('--human', choices=('fp', 'sh'), default='fp',
                    help='which side the human plays in hvai mode')
    ap.add_argument('--sims', type=int, default=300,
                    help='MCTS simulations per AI decision (default 300)')
    ap.add_argument('--seed', type=int, default=None,
                    help='RNG seed for reproducible games (also set '
                         'PYTHONHASHSEED=0, the .bat does)')
    ap.add_argument('--max-turns', type=int, default=40,
                    help='turn cap (default 40)')
    ap.add_argument('--shadow-vp', type=int, default=10,
                    help='Shadow military victory threshold (default 10)')
    ap.add_argument('--net', default='auto',
                    help='net tag (default auto: the deployed pair '
                         'of WOTR_ACT_GEN, src/net_tags.py)')
    ap.add_argument('--search', choices=SEARCH_CONTRACTS,
                    default='own_value',
                    help='AI search contract (default own_value: the live '
                         'AI\'s own search, as Play_vs_AI.bat serves it; '
                         'legacy_puct: the arena/eval search)')
    ap.add_argument('--selfplay-temps', action='store_true',
                    help='use the self-play temperature schedule '
                         '(0.6/0.3/0.15) instead of eval temp 0.02 '
                         '(--search legacy_puct only)')
    ap.add_argument('--dump-actions', default=None,
                    help='write the executed-action log to this file')
    a = ap.parse_args(argv)
    if a.selfplay_temps and a.search == 'own_value':
        ap.error('--selfplay-temps needs --search legacy_puct (the live '
                 'AI\'s own search always runs at temperature 0)')
    run_game(mode=a.mode, human=a.human, seed=a.seed, sims=a.sims,
             max_turns=a.max_turns, shadow_vp=a.shadow_vp,
             net_tag=a.net, selfplay_temps=a.selfplay_temps,
             dump_actions=a.dump_actions, search=a.search)


if __name__ == '__main__':
    main()
