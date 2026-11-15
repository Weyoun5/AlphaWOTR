# -*- coding: ascii -*-
"""turn_start.py -- THE turn-start driver (R3-LEGACY-GUI-TURN).

One implementation of a turn's pre-roll phases, block for block in the
order the self-play episode loop (ai._run_single_episode_impl) runs them:

  1b  open engine decisions: the hand-limit discards opened by the
      turn-start draws (any other decision a GUI left open is driven
      here too, before the Fellowship phase)
  2   Fellowship phase: the gateway auto-declare (the sanctioned
      self-play hardcode - never on a human Free Peoples seat), else the
      engine's declare menu under its own _declare_phase flag (offered at
      progress >= 0: in-place declaration is legal; 1.142c noise-free)
  2b  begin/resume_fellowship_tail: stationary Balrog, declaration Hunt
      effects, turn-start Mordor entry + refill (FINAL ruling
      2026-08-16); every open choice goes to its owner
  -   ring victory: a terminal Fellowship phase ends the game, no roll
  3   the end-of-phase highest-level Guide choice
  4   the Shadow Hunt allocation on the engine's eye menu under
      _eye_allocation_phase (companion maximum, ruling 1.271 minimum;
      1.142 noise-free)

Run by engine_play's console driver (_game_gen), the GUI watch mode
(WotrGUI._selfplay_turn_start) and the legacy hot-seat / --ai GUI
(WotrGUI._legacy_fellowship_phase), so the three cannot drift.

The driver owns ORDER, ENGINE PHASE FLAGS, MENUS, GUARDS and the
structural checks.  It never chooses and never executes: it is a
generator of requests, answered by the caller.

  Choose(kind, side, actions) -> send back the chosen Action (an element
      of `actions`, or a search pick from the AI policy menu).  For
      'declare' and 'eye' the engine phase flag is SET while the request
      is outstanding, so a search the caller runs sees the real menu.
  Execute(kind, side, action) -> execute it with the caller's own
      executor (ai._execute_action or a wrapper of it).
  Event(kind, detail)         -> informational: 'auto_declared',
      'fellowship_tail_done', 'game_over' ({'result': ...}),
      'mordor_entered', 'eyes_allocated' ({'count': n}).

Kinds: 'discard', 'decision' (an open engine decision that is not the
hand-limit discard), 'declare', 'fellowship_tail', 'guide', 'eye'.

The generator returns (StopIteration.value) 'ready' - the roll may
follow - or 'game_over' - a ring victory during the Fellowship phase;
nothing more is asked and the dice must not be rolled.  Every structural
failure raises TurnStartError naming the step (loud: nothing is skipped,
defaulted or substituted).  Closing the generator (gen.close(), which
run() always does) clears a phase flag a pending request still holds.
"""

from collections import namedtuple

import ai
import game_logic
from game_state import Side

FP, SH = Side.FREE_PEOPLES, Side.SHADOW

# The episode loop's guards (ai._run_single_episode_impl): at most 12
# turn-start discards, at most 128 Fellowship-tail decisions.
OPEN_DECISION_GUARD = 12
TAIL_GUARD = 128

DECLARE_TYPES = ('fellowship_declare', 'pass', 'declare_decline')
PHASE_FLAGS = ('_declare_phase', '_eye_allocation_phase',
               '_fellowship_tail_phase')

Choose = namedtuple('Choose', 'kind side actions')
Execute = namedtuple('Execute', 'kind side action')
Event = namedtuple('Event', 'kind detail')


class TurnStartError(RuntimeError):
    """A turn-start step cannot continue (no owner, no legal menu, an
    off-menu or missing selection, a guard exceeded)."""


def side_label(side):
    return 'Free Peoples' if side is FP else 'Shadow'


def _checked(state, req, act, types=None):
    """The chosen action, or TurnStartError.  `types`: the action types
    the step accepts (the episode loop's type checks); otherwise the pick
    must be on the offered menu or on the AI policy's menu for the same
    decision (the menu ai.mcts_search searches)."""
    what = '%s %s' % (side_label(req.side), req.kind)
    if act is None:
        raise TurnStartError('%s has no selection' % what)
    if types is not None:
        if act.action_type not in types:
            raise TurnStartError('%s has no legal selection (got %s)'
                                 % (what, act.action_type))
        return act
    if act in req.actions:
        return act
    if act in ai.get_ai_actions(ai.clone_state(state), req.side):
        return act
    raise TurnStartError('%s has no legal selection (got %r)' % (what, act))


def steps(state, auto_declare=True, declare=True, fellowship=True,
          eyes=True, fallback=None):
    """The turn-start generator (see the module docstring).

    auto_declare -- run the gateway auto-declare (an AI Free Peoples).
    declare      -- offer the declare menu (an AI Free Peoples, or any
                    seat whose caller asks it, e.g. engine_play's human).
    fellowship   -- run 1b + 2 + 2b.  The legacy GUI passes False once
                    this turn's tail has run (a second stationary-Balrog
                    opportunity would be wrong); the victory check, the
                    Guide choice and the eyes still run.
    eyes         -- run 4.  The legacy GUI allocates eyes itself
                    (ROLL DICE / Alloc Eyes); the flag is still cleared.
    fallback     -- ai._pending_decision_side's fallback (the legacy
                    GUI's current side, for a battle with no
                    deciding_side).
    """
    if state.dice_rolled:
        raise TurnStartError('the turn-start phase must precede the '
                             'action dice roll')
    if fellowship:
        # 1b: open engine decisions (the turn-start hand-limit discards)
        for _ in range(OPEN_DECISION_GUARD):
            kind, side = ai._pending_decision_kind_side(state, fallback)
            if kind is None or kind == 'declare':
                # a declare flag is a phase, not an open decision;
                # begin_fellowship_tail closes it
                break
            if kind == 'eye':
                raise TurnStartError('a stale Hunt-allocation flag is open '
                                     'ahead of the Fellowship phase')
            if side is None:
                raise TurnStartError('the open %s decision has no owner'
                                     % kind)
            acts = ai.get_legal_actions(state, side)
            if not acts:
                raise TurnStartError('the open %s decision (%s) has no '
                                     'legal choice' % (kind, side_label(side)))
            req = Choose('discard' if kind == 'discard' else 'decision',
                         side, acts)
            act = _checked(state, req, (yield req),
                           ('discard_card',) if kind == 'discard' else None)
            yield Execute(req.kind, side, act)
        else:
            kind, _side = ai._pending_decision_kind_side(state, fallback)
            if kind not in (None, 'declare'):
                raise TurnStartError('turn-start decisions exceeded %d'
                                     % OPEN_DECISION_GUARD)

        # 2: Fellowship phase.  A declared tail is never re-offered.
        if (not state.fellowship.mordor_entered
                and not getattr(state, '_fellowship_tail_phase', False)):
            auto_declared = False
            if auto_declare:
                auto_declared = ai._auto_enter_mordor(state)
                if auto_declared:
                    yield Event('auto_declared', {
                        'region': game_logic.fellowship_region(state)})
            if (declare and not auto_declared
                    and state.fellowship.progress >= 0
                    and not state.fellowship.mordor_entered
                    and ai._get_declare_actions(state, allow_in_place=True)):
                state._declare_phase = True
                try:
                    acts = ai.get_legal_actions(state, FP)
                    req = Choose('declare', FP, acts)
                    dec = _checked(state, req, (yield req), DECLARE_TYPES)
                finally:
                    state._declare_phase = False
                if dec.action_type == 'fellowship_declare':
                    yield Execute('declare', FP, dec)

        # 2b: declaration effects, stationary Balrog, turn-start Mordor
        # entry + refill; each open choice goes to its owner
        was_mordor = state.fellowship.mordor_entered
        game_logic.begin_fellowship_tail(state)
        for _ in range(TAIL_GUARD):
            if not getattr(state, '_fellowship_tail_phase', False):
                break
            owner = ai._pending_decision_side(state, fallback)
            if owner is None:
                raise TurnStartError(
                    'open Fellowship effect has no decision owner (%s)'
                    % ', '.join(game_logic.fellowship_tail_pending(state)))
            acts = ai.get_legal_actions(state, owner)
            if not acts:
                raise TurnStartError('required Fellowship effect (%s) has '
                                     'no legal choice' % side_label(owner))
            req = Choose('fellowship_tail', owner, acts)
            act = _checked(state, req, (yield req))
            yield Execute('fellowship_tail', owner, act)
            game_logic.resume_fellowship_tail(state)
        if getattr(state, '_fellowship_tail_phase', False):
            raise TurnStartError('Fellowship effects exceeded %d decisions'
                                 % TAIL_GUARD)
        yield Event('fellowship_tail_done', {})
    else:
        was_mordor = state.fellowship.mordor_entered

    result = game_logic.check_all_victory_conditions(state)
    if game_logic.should_game_end(result, end_of_turn=False):
        yield Event('game_over', {'result': result})
        return 'game_over'
    if state.fellowship.mordor_entered and not was_mordor:
        yield Event('mordor_entered', {})

    # 3: end of the Fellowship phase - the highest-level Guide choice
    if state.pending_guide_change is not None:
        acts = ai.get_legal_actions(state, FP)
        if not acts or any(a.action_type != 'choose_guide' for a in acts):
            raise TurnStartError('the end-of-phase Guide menu is '
                                 'unavailable')
        req = Choose('guide', FP, acts)
        act = _checked(state, req, (yield req), ('choose_guide',))
        yield Execute('guide', FP, act)
        if state.pending_guide_change is not None:
            raise TurnStartError('the Guide choice did not resolve')

    # 4: the Shadow Hunt allocation on the engine's eye menu
    if eyes and any(not d.used for d in state.dice.get(SH, [])):
        state._eye_allocation_phase = True
        try:
            acts = ai.get_legal_actions(state, SH)
            if not acts or any(a.action_type != 'eye_allocate'
                               for a in acts):
                raise TurnStartError(
                    'the engine eye menu is unavailable (served %s)'
                    % sorted({a.action_type for a in acts}))
            req = Choose('eye', SH, acts)
            eye = _checked(state, req, (yield req), ('eye_allocate',))
            counts = [a.params.get('count') for a in acts]
            if eye.params.get('count') not in counts:
                raise TurnStartError('Shadow eye has no legal selection '
                                     '(menu %s, got %r)' % (counts, eye))
        finally:
            state._eye_allocation_phase = False
        yield Execute('eye', SH, eye)
        yield Event('eyes_allocated', {'count': eye.params.get('count', 0)})
    state._eye_allocation_phase = False
    return 'ready'


def run(gen, choose, execute, on_event=None):
    """Drive a steps() generator with callbacks, for callers that are not
    generators themselves: choose(req) -> Action, execute(req),
    on_event(ev).  Returns the generator's outcome ('ready' /
    'game_over').  The generator is always closed, so a phase flag held
    by an outstanding request is cleared even when a callback raises
    (the exception propagates unchanged)."""
    reply = None
    try:
        while True:
            try:
                req = gen.send(reply)
            except StopIteration as stop:
                return stop.value
            reply = None
            if isinstance(req, Choose):
                reply = choose(req)
            elif isinstance(req, Execute):
                execute(req)
            elif on_event is not None:
                on_event(req)
    finally:
        gen.close()
