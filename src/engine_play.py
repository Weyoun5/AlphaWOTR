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
    ai._pending_decision_side / ai._post_action_active   (side routing --
        the same single-source-of-truth routers self-play and MCTS use)
    ai._auto_enter_mordor / ai._balrog_stationary_turn_start
    ai._safe_next_turn                (turn boundary)
    game_logic.check_all_victory_conditions / should_game_end
    ai.mcts_action_with_policy        (the per-decision helper
        ai._run_single_episode uses internally; arena/eval config)
    ai.load_value_net                 (current-generation nets:
                                       corpus_live / v41 chain)

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
        for this decision class (consumed by EngineSession.ai_pick).
    """

    def __init__(self, kind, side, actions, note='', sims=100,
                 temp=0.0, noise=False, turn=0):
        self.kind = kind
        self.side = side
        self.actions = actions
        self.note = note
        self.sims = sims
        self.temp = temp
        self.noise = noise
        self.turn = turn


class _ConsoleAnnotator(object):
    """Duck-typed TraceRecorder stand-in for ai._traced_execute: engine
    roll/tile sites (game_trace.emit_annotation) narrate to the console
    through the EXISTING describe_step renderer instead of snapshotting.
    MCTS lookahead never reaches it (ai._traced_execute scopes
    ACTIVE_TRACER to the single real execution)."""

    buffering = False

    def __init__(self, log):
        self._log = log

    def begin_action_buffer(self):
        pass  # print immediately; console order = execution order

    def record_action(self, state, side, action, note='', **kw):
        try:
            step = {'kind': 'action', 'turn': state.turn,
                    'side': game_trace._side_tag(side),
                    'action_type': action.action_type,
                    'params': dict(action.params or {})}
            self._log('    . ' + game_trace.describe_step(step))
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


def describe_action(state, side, action):
    """Compact human-readable line for a legal action (display only)."""
    t = action.action_type
    p = action.params or {}
    body = None
    if t == 'pass':
        body = 'Pass (no die spent; legal while holding fewer dice)'
    elif t == 'army_move':
        body = 'Move army %s -> %s' % (p.get('from', '?'), p.get('to', '?'))
        comp = []
        if p.get('n_regular') is not None or p.get('n_elite') is not None:
            comp.append('%sr/%se' % (p.get('n_regular', 0),
                                     p.get('n_elite', 0)))
        if p.get('take_leaders'):
            comp.append('+leaders')
        if p.get('n_nazgul'):
            comp.append('%s nazgul' % p.get('n_nazgul'))
        if comp:
            body += ' (%s)' % ', '.join(comp)
        if p.get('mouth_muster'):
            body += ' (Mouth: muster die as army)'
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
                 shadow_vp=10, temperature=0.02, add_noise=False,
                 selfplay_temps=False, log=None, human_side=None,
                 max_turn_decisions=10000):
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
        self.temperature = float(temperature)
        self.add_noise = bool(add_noise)
        self.selfplay_temps = bool(selfplay_temps)
        self.log = log if log is not None else (lambda s: None)
        self._annotator = _ConsoleAnnotator(self.log)
        self.result_info = None
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
        self._nets = (fp_net, sh_net)

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
        """MCTS choice for the current decision -- the SAME per-decision
        helper self-play uses (ai.mcts_action_with_policy), with the
        per-phase search config _run_single_episode applies.  The sims
        budget is re-read from sims_for_current() at call time, so a
        live sims_override takes effect at the NEXT decision.  Returns
        a menu index for submit()."""
        dp = self._current
        if dp is None:
            raise RuntimeError('game is over')
        if self._nets is None:
            raise RuntimeError('no nets attached (attach_nets first)')
        fp_net, sh_net = self._nets
        n_sims = self.sims_for_current()
        self.last_pick_sims = n_sims
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
            raise IllegalPlay('Search did not return a legal current decision')
        return idx

    # -- internals ----------------------------------------------------

    def _execute(self, action, side):
        """Execute the selected public action; errors stop the session."""
        try:
            ai._traced_execute(self.state, action, self._annotator)
            return action
        except Exception as err:
            ai._record_exec_failure(self.state, action, err)
            self.log('! execution failed (%s: %s)' % (type(err).__name__, err))
            raise

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

            # Phase 1b: hand-limit discards (episode 14727-14756)
            _d_guard = 0
            while (getattr(state, 'pending_discard', None) is not None
                   and _d_guard < 12):
                _d_guard += 1
                d_side = state.pending_discard['side']
                acts = ai.get_legal_actions(state, d_side)
                if not acts or getattr(state, 'pending_discard',
                                       None) is None:
                    continue
                d_temp = fp_temp if d_side is FP else sh_temp
                idx = yield DecisionPoint(
                    'discard', d_side, acts,
                    note='hand over 6 cards - discard down',
                    sims=_sims_for_kind('discard', sims), temp=d_temp,
                    noise=self.add_noise, turn=state.turn)
                act = self._execute(acts[idx], d_side)
                self._record(d_side, act)

            # Declaration finishes before begin_fellowship_tail exposes
            # the stationary Balrog opportunity (Almanac C17).

            # Gateway auto-declare, TURN-START only (episode twin).
            # Self-play hardcode: never on a human FP seat - the declare
            # DecisionPoint below already offers the gateway declare.
            _auto_declared = False
            if not state.fellowship.mordor_entered \
                    and self.human_side is not FP:
                _auto_declared = ai._auto_enter_mordor(state)
                if _auto_declared:
                    self.log('  Fellowship declared at the gateway')

            # Phase 2: Fellowship declare phase (episode 14784-14824).
            # Same gate: only offered when a declare is actually
            # available; the menu is get_legal_actions under the
            # engine's own _declare_phase flag (declares + pass).
            if (not _auto_declared and state.fellowship.progress >= 0
                    and not state.fellowship.mordor_entered
                    and ai._get_declare_actions(state,
                                                allow_in_place=True)):
                state._declare_phase = True
                try:
                    acts = ai.get_legal_actions(state, FP)
                    idx = yield DecisionPoint(
                        'declare', FP, acts,
                        note='Fellowship phase - declare or pass '
                             '(noise-free, argmax: 1.142c)',
                        sims=_sims_for_kind('declare', sims),
                        temp=0.0, noise=False, turn=state.turn)
                finally:
                    state._declare_phase = False
                act = acts[idx]
                if act.action_type == 'fellowship_declare':
                    act = self._execute(act, FP)
                    self._record(FP, act)

            was_mordor = state.fellowship.mordor_entered
            game_logic.begin_fellowship_tail(state)
            for _tail_step in range(128):
                if not getattr(state, '_fellowship_tail_phase', False):
                    break
                owner = ai._pending_decision_side(state)
                if owner is None:
                    raise RuntimeError('Unresolved Fellowship tail has no decision owner')
                acts = ai.get_legal_actions(state, owner)
                if not acts:
                    raise RuntimeError('Required Fellowship effect has no legal choice')
                idx = yield DecisionPoint(
                    'fellowship_tail', owner, acts, note=self._note_pending(),
                    sims=_sims_for_kind('fellowship_tail', sims),
                    temp=fp_temp if owner is FP else sh_temp,
                    noise=self.add_noise, turn=state.turn)
                act = self._execute(acts[idx], owner)
                self._record(owner, act)
                game_logic.resume_fellowship_tail(state)
            if getattr(state, '_fellowship_tail_phase', False):
                raise RuntimeError('Fellowship tail exceeded 128 decisions')
            if game_logic.should_game_end(game_logic.check_all_victory_conditions(state), end_of_turn=False):
                game_over = True
                break
            if state.fellowship.mordor_entered and not was_mordor:
                self.log('  Fellowship enters Mordor (turn start)')
            if state.pending_guide_change is not None:
                acts = ai.get_legal_actions(state, FP)
                if not acts or any(a.action_type != 'choose_guide' for a in acts):
                    raise RuntimeError('Required Fellowship Guide menu is unavailable')
                idx = yield DecisionPoint(
                    'guide', FP, acts,
                    note='End Fellowship phase - keep or choose a highest-level Guide',
                    sims=_sims_for_kind('guide', sims), temp=fp_temp,
                    noise=self.add_noise, turn=state.turn)
                act = self._execute(acts[idx], FP)
                self._record(FP, act)
                if state.pending_guide_change is not None:
                    raise RuntimeError('Fellowship Guide selection did not resolve')

            # Phase 3: Shadow eye allocation, pre-roll (episode
            # 14827-14876); menu/executor are the engine's own.
            shadow_unused = [d for d in state.dice.get(SH, [])
                             if not d.used]
            if len(shadow_unused) > 0:
                state._eye_allocation_phase = True
                try:
                    acts = ai.get_legal_actions(state, SH)
                    idx = yield DecisionPoint(
                        'eye', SH, acts,
                        note='allocate eyes to the Hunt Box before '
                             'rolling (noise-free: 1.142)',
                        sims=sims, temp=0.0, noise=False,
                        turn=state.turn)
                finally:
                    state._eye_allocation_phase = False
                act = acts[idx]
                n_eyes = act.params.get('count', 0)
                act = self._execute(act, SH)
                self._record(SH, act)
                self.log('  Shadow allocates %d eye(s) to the Hunt Box'
                         % n_eyes)

            state._eye_allocation_phase = False
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
                    noise=self.add_noise, turn=state.turn)
                action = actions[idx]
                other = SH if active is FP else FP

                self.log('  %s: %s' % ('FP' if active is FP else 'SH',
                                       describe_action(state, active,
                                                       action)))
                action = self._execute(action, active)
                self._record(active, action)

                # Victory timing (episode 15035-15040): ring victories
                # end the game immediately; military waits for the
                # end-of-turn Victory Check (rulebook p.17)
                vc = game_logic.check_all_victory_conditions(state)
                if game_logic.should_game_end(vc, end_of_turn=False):
                    game_over = True
                    break

                # Turn-order routing (episode 15042-15052)
                if action.action_type == 'elven_ring':
                    pass  # ring use is NOT an action - same side acts
                elif (state.pending_army_move_die is not None
                        or state.pending_muster_die is not None):
                    pass  # same side continues (multi-step die action)
                else:
                    _ps9 = ai._pending_decision_side(state,
                                                     fallback=active)
                    active = (_ps9 if _ps9 is not None
                              else ai._post_action_active(state, active,
                                                          other))

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
    # CURRENT generation first (v41 = brain everywhere, USER
    # 2026-08-22; corpus_live IS the v41 pair the live head runs),
    # then the retired v37 chain for old checkouts.
    for cand in ('corpus_live', 'corpus_v41jt', 'corpus_v41td',
                 'corpus_v41', 'corpus_v37jt', 'corpus_v37td',
                 'corpus_v37'):
        if os.path.exists(ai._resolve_model_path('wotr_fp_%s.pt'
                                                 % cand)):
            return cand
    return None


def load_nets(tag='auto'):
    """Load the packaged FP/SH nets via ai.load_value_net (current
    generation first: corpus_live / v41, then the retired v37 chain).
    Fails LOUDLY -- no silent fallback policy."""
    tag = _resolve_net_tag(tag)
    if tag is None:
        raise SystemExit('no nets found next to the project root - '
                         'expected wotr_{fp,sh}_corpus_live.pt (the '
                         'v41 pair) or wotr_fp_corpus_v41*/v37*.pt')
    fp_net = ai.load_value_net('wotr_fp_%s.pt' % tag, 'fp')
    sh_net = ai.load_value_net('wotr_sh_%s.pt' % tag, 'sh')
    if fp_net is None or sh_net is None:
        raise SystemExit('net load failed for tag %r (file missing or '
                         'torch unavailable)' % tag)
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
             dump_actions=None):
    human_side = None
    if mode == 'hvai':
        human_side = FP if human == 'fp' else SH
    session = EngineSession(seed=seed, mcts_sims=sims,
                            max_turns=max_turns, shadow_vp=shadow_vp,
                            selfplay_temps=selfplay_temps, log=_p,
                            human_side=human_side)
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
                                 dp.actions[idx]))
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
                    help='net tag (default auto: corpus_live, then '
                         'the v41/v37 chains)')
    ap.add_argument('--selfplay-temps', action='store_true',
                    help='use the self-play temperature schedule '
                         '(0.6/0.3/0.15) instead of eval temp 0.02')
    ap.add_argument('--dump-actions', default=None,
                    help='write the executed-action log to this file')
    a = ap.parse_args(argv)
    run_game(mode=a.mode, human=a.human, seed=a.seed, sims=a.sims,
             max_turns=a.max_turns, shadow_vp=a.shadow_vp,
             net_tag=a.net, selfplay_temps=a.selfplay_temps,
             dump_actions=a.dump_actions)


if __name__ == '__main__':
    main()
