"""Single-threaded Python reference environment around existing rules functions.

Independent game RNG streams permit interleaved games and batched inference.
All actual decisions, including upkeep, use the same prepared-menu interface.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import random

import numpy as np

from .contracts import EpisodeEnd
from .policy import ActorInput
from .probabilities import encode_probabilities


def derived_seed(seed: int, domain: str) -> int:
    return int.from_bytes(hashlib.sha256(f"actor_critic_rng_v1:{seed}:{domain}".encode()).digest()[:8], "big")


class GameRandom:
    """Swapped only around rules calls; never shared concurrently by threads."""
    def __init__(self, seed):
        self.python = random.Random(derived_seed(seed, "game_python")).getstate()
        self.numpy = np.random.RandomState(derived_seed(seed, "game_numpy") % (2 ** 32)).get_state()

    @contextmanager
    def scope(self):
        previous_python, previous_numpy = random.getstate(), np.random.get_state()
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        try:
            yield
        finally:
            self.python, self.numpy = random.getstate(), np.random.get_state()
            random.setstate(previous_python)
            np.random.set_state(previous_numpy)


@dataclass(frozen=True)
class PreparedDecision:
    side: str
    phase: str
    turn: int
    actions: tuple
    inputs: ActorInput
    fixed_reason: str | None


class PythonEnvironment:
    """Owns one actual game. No MCTS nodes, hypothetical hands, or reward shaping."""
    def __init__(self, ai, episode_id: str, seed: int, *, state=None, active=None,
                 max_turns=100, max_transitions=10000):
        if max_turns < 1 or max_transitions < 1:
            raise ValueError("Episode limits must be positive")
        self.ai, self.gs, self.gl = ai, ai.gs, ai.game_logic
        self.episode_id, self.seed = episode_id, seed
        self.rng = GameRandom(seed)
        self.samplers = {s: np.random.default_rng(derived_seed(seed, "policy_" + s)) for s in ("fp", "sh")}
        self.sides = {"fp": self.gs.Side.FREE_PEOPLES, "sh": self.gs.Side.SHADOW}
        self.tags = {v: k for k, v in self.sides.items()}
        self.max_turns, self.max_transitions = max_turns, max_transitions
        self.committed = 0
        self.end = None
        # Ordinary action owner, retained while the other player answers a
        # reaction. prepare() routes pending choices separately from this seat.
        self.active = active
        self.stage = "actions" if state is not None and state.dice_rolled else "opening"
        if state is not None and not state.dice_rolled:
            if getattr(state, "_declare_phase", False):
                self.stage = "declaration"
            elif (getattr(state, "_fellowship_tail_phase", False)
                  or getattr(state, "_eye_allocation_phase", False)):
                self.stage = "preaction"
        with self.rng.scope():
            if state is None:
                state = self.gs.setup_initial_game(ai.board.load_board())
                self.gl.initialize_decks(state)
                for side in self.sides.values():
                    for deck in ("character", "strategy"):
                        self.gl.draw_event_card(state, side, deck)
            self.state = state
            self.first_turn = state.turn
            if self.stage == "actions" and self.active is None:
                self.active = ai._turn_start_side(state)

    def finish(self, status, *, reason=None, victory=None):
        self.end = EpisodeEnd(self.episode_id, status, transition_count=self.committed,
            reason=reason, winner=({"free_peoples": "fp", "shadow": "sh"}[victory["victory"]] if victory else None),
            victory_route=(victory or {}).get("victory_type"))

    @contextmanager
    def _engine_scope(self, *, action=False):
        with self.rng.scope():
            replay = getattr(self, "replay", None)
            if replay is None:
                yield
            else:
                with replay.capture(self, action=action):
                    yield

    def _victory(self, end_of_turn=False):
        if end_of_turn:
            # A limit may stop us before the next turn begins. Preview only
            # the end-of-turn settlement; nonterminal rollover below owns
            # the real penalty and reset. Terminal settlement must precede
            # military victory and must not draw the next turn's cards.
            if not self.gl._settle_end_of_turn(self.ai.clone_state(self.state)):
                return False
            self.ai._safe_next_turn(self.state)
        result = self.gl.check_all_victory_conditions(self.state)
        if self.gl.should_game_end(result, end_of_turn=end_of_turn):
            self.finish("terminated", victory=result)
            return True
        return False

    def _frame(self, side):
        policy = self.ai.get_ai_policy(self.state, side)
        actions, reason = tuple(policy.actions), policy.forced_reason
        if self.stage == "declaration" and self.state.fellowship.hidden and self.state.fellowship.progress > 0:
            # Preserve the existing sanctioned gateway-declaration convention,
            # while recording its actual player transition instead of hiding it.
            for gateway in ("morannon", "minas morgul"):
                forced = next((a for a in actions if a.action_type == "fellowship_declare"
                               and a.params.get("region") == gateway), None)
                if forced is not None:
                    actions, reason = (forced,), "mordor_gateway_declaration"
                    break
        if not actions:
            raise RuntimeError(f"No legal action for {self.tags[side]} in {self.stage}")
        # Reuse the existing vectorized encoder; no search menu pruning or caps.
        encodings, regions = self.ai._encode_actions_batch_ordinary(actions, self.state, side, {})
        inputs = ActorInput(self.ai.encode_state(self.state, side), encodings, regions,
                            encode_probabilities(self.ai, self.state, side, actions),
                            kinds=tuple(a.action_type for a in actions))
        return PreparedDecision(self.tags[side], self.stage, self.state.turn, actions, inputs, reason)

    def prepare(self):
        with self._engine_scope():
            while self.end is None:
                if self._victory():
                    return None
                if self.committed >= self.max_transitions:
                    # An actual end-turn win on the final allowed decision is
                    # still a completed game. Open reactions must settle first.
                    if (self.stage == "actions" and self.active is None
                            and self.ai._pending_decision_side(self.state, fallback=self.sides["fp"]) is None
                            and self._victory(end_of_turn=True)):
                        return None
                    self.finish("truncated", reason="transition_limit")
                    return None
                state = self.state
                if self.stage == "post_declaration":
                    self.gl.begin_fellowship_tail(state)
                    self.stage = "preaction"
                elif self.stage == "post_allocation":
                    self._roll()
                if self.stage == "opening":
                    pending = self.ai._pending_decision_side(state)
                    if pending is not None:
                        return self._frame(pending)
                    if not state.fellowship.mordor_entered and self.ai._get_declare_actions(state, allow_in_place=True):
                        state._declare_phase = True
                        self.stage = "declaration"
                    else:
                        self.gl.begin_fellowship_tail(state)
                        self.stage = "preaction"
                if self.stage == "declaration":
                    return self._frame(self.sides["fp"])
                if self.stage == "preaction":
                    self.gl.resume_fellowship_tail(state)
                    pending = self.ai._pending_decision_side(state)
                    if pending is not None:
                        return self._frame(pending)
                    self._roll()
                if self.stage == "actions":
                    pending = self.ai._pending_decision_side(state, fallback=self.active)
                    if pending is not None:
                        return self._frame(pending)
                    if self.active is not None:
                        return self._frame(self.active)
                    if self._victory(end_of_turn=True):
                        return None
                    if state.turn - self.first_turn + 1 >= self.max_turns:
                        self.finish("truncated", reason="turn_limit")
                        return None
                    self.ai._safe_next_turn(state)
                    self.stage = "opening"
            return None

    def _roll(self):
        self.state._eye_allocation_phase = False
        for side in self.sides.values():
            self.gl.roll_action_dice(self.state, pool=side)
        self.stage = "actions"
        self.active = self.ai._turn_start_side(self.state)

    def apply(self, frame: PreparedDecision, selected: int):
        with self._engine_scope(action=True):
            action = frame.actions[selected]
            self.ai._execute_action(self.state, action)
            # Count a successful player action before any subsequent automatic
            # operation: a failed roll/upkeep must not erase a committed choice.
            self.committed += 1
            self.after_action(frame, action)

    def after_action(self, frame: PreparedDecision, action):
        """Stage and seat bookkeeping once frame's `action` has been applied to self.state.

        apply() runs it after the engine call; a caller that builds the successor
        state elsewhere (chance.kernel_outcomes: the engine combat kernel's states)
        must run it too, or the world keeps the parent's seat and stage."""
        if frame.phase == "declaration":
            self.stage = "post_declaration"
        elif frame.phase == "preaction":
            if action.action_type == "eye_allocate":
                self.stage = "post_allocation"
        elif frame.phase == "actions":
            owner = self.active if self.active is not None else self.sides[frame.side]
            if self.ai._pending_decision_side(self.state, fallback=owner) is None:
                self.active = self.ai._next_acting_side(self.state, owner)
