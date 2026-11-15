"""Chance-event logging for the chance control variate (luck subtraction).

Review of 2026-10-11 (ChatGPT, agreed): for every CHANCE EVENT of a game record the position before it, the exact
outcome distribution and the realised outcome, so that a frozen critic u can price how lucky the draw was,
    L_j = u(z_j, c_j) - sum_c p(c | z_j) u(z_j, c),
a term of zero conditional mean that is subtracted from the return of every decision taken before the event
(luck.py). The game itself is untouched: the live environment applies every move with its own RNG exactly as
without logging; the alternatives are enumerated on CLONES (chance.expand: exact by the fork driver, else the
engine's combat kernel, else sampled clones), so a logged collection is the same game, seed for seed, as an
unlogged one. The realised outcome's position is the live state after the apply; its probability bookkeeping
(`matched`) is diagnostic only: L_j is zero-mean whether or not the realised leaf is identified, because the
live engine draws from the same distribution the enumeration describes. For sampled expansions (no enumerator)
the expectation is the mean over K independent clones, which keeps the zero mean at a higher variance.

The action-dice roll after the eye allocation (far too many outcomes to enumerate) is logged from K sampled
rolls of the same position (chance.roll_outcomes); the realised position is the next decision's. Its luck term is
zero-mean only averaged over its own K samples (the expectation is a K-sample estimate, not the exact one), so it
carries more variance than an exact event; raising --chance-roll-samples trades time for variance.

NOT LOGGED (GPT-6 review 2026-10-11): chance that the engine resolves during automatic preparation outside an
apply (turn-start upkeep, the Fellowship tail's own draws when no decision is pending) is not enumerated;
only the chance inside applies (stored and forced decisions) and the post-allocation roll are. This loses some
variance reduction, never the zero mean: an unlogged event simply stays in the return as it was.

Arrays stored in arrays.npz (all events of the game in order):
  chance_after      int32  [E]      rows stored before the event: the event follows decision chance_after - 1
  chance_kind       str    [E]      exact | kernel | sampled | roll
  chance_action     str    [E]      the action type that triggered it (or "roll")
  chance_offsets    int64  [E + 1]  ragged layout of the alternatives
  chance_probs      float32 [total] probabilities (sum to 1 per event)
  chance_terminal   float32 [total] the game's result for the Shadow at that alternative if the game is decided there, else NaN
  chance_obs_sh / chance_obs_fp   float16 [total, obs]  both views of every alternative
  chance_live_sh / chance_live_fp float16 [E, obs]      both views of the realised position
  chance_live_terminal float32 [E] the Shadow's result if the realised position decides the game, else NaN
  chance_matched    int32  [E]      index of the alternative identical to the realised position (-1 if none)
"""
import numpy as np

from .chance import Unsupported, expand, roll_outcomes

KINDS = ("exact", "kernel", "sampled", "roll")


def terminal_value(env_like):
    """The Shadow's result (+1 / -1) if the position decides the game now, else NaN (no side effect)."""
    gl = env_like.gl
    try:
        result = gl.check_all_victory_conditions(env_like.state)
        if not gl.should_game_end(result, end_of_turn=False):
            return float("nan")
    except Exception:
        return float("nan")
    return 1. if result["victory"] == "shadow" else -1.


class ChanceLogger:
    def __init__(self, env, ai, seed, *, roll_samples=8, max_walks=64, samples=8):
        self.env, self.ai = env, ai
        self.roll_samples, self.max_walks, self.samples = int(roll_samples), int(max_walks), int(samples)
        self.rng = np.random.default_rng((int(seed) * 0x9E3779B1 + 0x7F4A7C15) % (2 ** 63))
        self.events = []
        self.pending = None
        self.skipped = dict(unsupported=0, errors=0)

    def _views(self, world):
        state = world.state
        return (np.asarray(self.ai.encode_state(state, world.sides["sh"]), np.float16),
                np.asarray(self.ai.encode_state(state, world.sides["fp"]), np.float16))

    def _record(self, leaves, how, action, rows_stored):
        probs, obs_sh, obs_fp, terminal = [], [], [], []
        for probability, world in leaves:
            sh, fp = self._views(world)
            probs.append(float(probability))
            obs_sh.append(sh)
            obs_fp.append(fp)
            terminal.append(terminal_value(world))
        self.pending = dict(after=int(rows_stored), kind=how, action=str(action), probs=np.asarray(probs, np.float32),
                            obs_sh=np.stack(obs_sh), obs_fp=np.stack(obs_fp), terminal=np.asarray(terminal, np.float32))

    def before_apply(self, frame, selected, rows_stored):
        """Enumerate the chance of applying frame.actions[selected] on clones; nothing is recorded when the apply
        holds no chance (one exact outcome)."""
        self.pending = None
        try:
            leaves, how = expand(self.env, frame, selected, max_walks=self.max_walks, samples=self.samples,
                                 seed=int(self.rng.integers(0, 2 ** 31)))
        except Unsupported:
            self.skipped["unsupported"] += 1
            return
        except Exception:
            self.skipped["errors"] += 1
            return
        if how == "exact" and len(leaves) == 1:
            return
        self._record(leaves, how, frame.actions[selected].action_type, rows_stored)

    def after_apply(self):
        """The realised position: the live environment right after the apply."""
        if self.pending is None:
            return
        self._finish_live(self.env)

    def _finish_live(self, env_like, live_views=None):
        event = self.pending
        self.pending = None
        sh, fp = self._views(env_like) if live_views is None else live_views
        event["live_sh"], event["live_fp"] = sh, fp
        event["live_terminal"] = terminal_value(env_like)
        same = np.nonzero(np.all(event["obs_sh"] == sh[None], axis=1) & np.all(event["obs_fp"] == fp[None], axis=1))[0]
        event["matched"] = int(same[0]) if len(same) else -1
        self.events.append(event)

    def before_prepare(self, rows_stored):
        """The action-dice roll that follows the eye allocation: K sampled rolls of the same position."""
        if self.pending is not None:            # an apply's event never got its realised position: drop it
            self.pending = None
        if self.env.end is None and self.env.stage == "post_allocation":
            try:
                leaves = roll_outcomes(self.env, self.roll_samples, int(self.rng.integers(0, 2 ** 31)))
            except Exception:
                self.skipped["errors"] += 1
                return
            self._record(leaves, "roll", "roll", rows_stored)

    def on_frame(self, mover_tag, mover_obs, other_obs):
        """The realised position of a pending roll event: the next decision's views."""
        if self.pending is None or self.pending["kind"] != "roll":
            return
        sh, fp = (mover_obs, other_obs) if mover_tag == "sh" else (other_obs, mover_obs)
        self._finish_live(self.env, (np.asarray(sh, np.float16), np.asarray(fp, np.float16)))

    def arrays(self, obs_dim):
        events = self.events
        if not events:
            return dict(chance_after=np.zeros(0, np.int32), chance_kind=np.array([], dtype="U8"), chance_action=np.array([], dtype="U32"),
                        chance_offsets=np.zeros(1, np.int64), chance_probs=np.zeros(0, np.float32), chance_terminal=np.zeros(0, np.float32),
                        chance_obs_sh=np.zeros((0, obs_dim), np.float16), chance_obs_fp=np.zeros((0, obs_dim), np.float16),
                        chance_live_sh=np.zeros((0, obs_dim), np.float16), chance_live_fp=np.zeros((0, obs_dim), np.float16),
                        chance_live_terminal=np.zeros(0, np.float32), chance_matched=np.zeros(0, np.int32))
        counts = [len(e["probs"]) for e in events]
        return dict(chance_after=np.asarray([e["after"] for e in events], np.int32),
                    chance_kind=np.asarray([e["kind"] for e in events]), chance_action=np.asarray([e["action"] for e in events]),
                    chance_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                    chance_probs=np.concatenate([e["probs"] for e in events]),
                    chance_terminal=np.concatenate([e["terminal"] for e in events]),
                    chance_obs_sh=np.concatenate([e["obs_sh"] for e in events]), chance_obs_fp=np.concatenate([e["obs_fp"] for e in events]),
                    chance_live_sh=np.stack([e["live_sh"] for e in events]), chance_live_fp=np.stack([e["live_fp"] for e in events]),
                    chance_live_terminal=np.asarray([e["live_terminal"] for e in events], np.float32),
                    chance_matched=np.asarray([e["matched"] for e in events], np.int32))

    def summary(self):
        kinds = {}
        for e in self.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        return dict(events=len(self.events), kinds=kinds, alternatives=int(sum(len(e["probs"]) for e in self.events)),
                    matched=int(sum(e["matched"] >= 0 for e in self.events)), **self.skipped)
