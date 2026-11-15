"""Actor-conditioned hypothetical root hands; reference helper, not a policy.

Reuses the engine's _SimOverlay bytecode with a private RNG namespace instead
of rebinding process globals. Callers must aggregate action values across these
worlds before selecting one action for the actor's information set.
"""
from dataclasses import dataclass
import copy
import json
import random
import types


@dataclass(frozen=True)
class BeliefWorld:
    state: object
    weight: float
    sample_index: int
    rng_before: tuple
    rng_after: tuple


def _local_overlay_type(ai, rng):
    namespace = dict(ai.__dict__)
    namespace.update(random=rng, OVERLAY_STATS=dict(ai.OVERLAY_STATS),
                     _COMBAT_FORECAST_OBSERVER=None)
    methods = {'__slots__': ()}
    for name, method in vars(ai._SimOverlay).items():
        if not isinstance(method, types.FunctionType):
            continue
        if method.__globals__ is not ai.__dict__:
            raise ValueError('Overlay method does not belong to the supplied engine')
        rebound = types.FunctionType(method.__code__, namespace, name,
                                     method.__defaults__, method.__closure__)
        rebound.__kwdefaults__ = method.__kwdefaults__
        methods[name] = rebound
    return type('_PrivateRandomOverlay', (ai._SimOverlay,), methods)


def _blind_commitment(ai, state, actor):
    """Mirror the engine's root-only nonce/physical-slot preparation.

    The existing overlay still owns candidate legality, type filtering, sampling,
    physical-card conservation, and the eventual committed identity replacement.
    """
    if ai._hidden_commit_card(state, actor) is None:
        return
    pending = state.pending_combat
    cid = pending.get('atk_combat_card')
    # Existing nonce/slot may describe this exact current hidden window. Locate
    # the actual physical entry first; a hidden protocol card has no public slot.
    slot = None
    for deck, cards in (getattr(state, 'discard_face_up', None) or {}).items():
        for index in range(len(cards) - 1, -1, -1):
            if cards[index] == cid:
                cards[index] = '_committed_face_down_'
                slot = (deck, index)
                break
        if slot is not None:
            break
    if slot is not None:
        state._commit_slot = slot
    pending['_commit_nonce'] = True


def _canonicalize_unknown_order(ai, state, actor):
    """Forget only orders the source sampler already treats as unknown.

    The source overlay sorts the other side's unseen card pool, but shuffles
    the actor's draw decks and hunt bag directly. Canonicalize their input so
    an identical RNG sample describes the same world across outer shuffles.
    Preserve the actor's known hand/discards and every public/pending field.
    Full tile properties break ties without dropping display or special data.
    """
    tag = 'fp' if actor == ai.Side.FREE_PEOPLES else 'sh'
    for kind in ('char', 'strat'):
        cards = getattr(state, tag + '_' + kind + '_deck', None)
        if cards is not None:
            cards.sort()
    pool = getattr(state, 'hunt_pool', None)
    if pool is not None:
        pool.sort(key=lambda tile: json.dumps(
            vars(tile), sort_keys=True, separators=(',', ':'), allow_nan=False))


def sample_worlds(ai, state, actor, *, count=4, seed=None, rng_states=None):
    """Return independent equally weighted worlds from the actor's knowledge.

    Exactly one of seed or rng_states is required. Explicit RNG states identify
    each sample and must have length=count. Neither game RNG nor search RNG nor
    engine telemetry is consumed. Actor must be the side whose choice is being
    considered; its own sampled hand is preserved, while the other side's true
    hand is forgotten within the engine's existing exchangeable-pool belief.
    """
    if actor not in (ai.Side.FREE_PEOPLES, ai.Side.SHADOW):
        raise ValueError('Actor must be a canonical engine Side')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if (seed is None) == (rng_states is None):
        raise ValueError('Supply exactly one of seed or rng_states')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError('seed must be an integer')
    if rng_states is not None:
        states = tuple(rng_states)
        if len(states) != count:
            raise ValueError('One RNG state is required for each sample')
        for value in states:
            random.Random().setstate(value)
    else:
        stream = random.Random(seed)
        states = tuple(random.Random(stream.getrandbits(256)).getstate()
                       for _ in range(count))
    result = []
    for index, before in enumerate(states):
        rng = random.Random()
        rng.setstate(before)
        world = copy.deepcopy(state)
        _canonicalize_unknown_order(ai, world, actor)
        _blind_commitment(ai, world, actor)
        overlay = _local_overlay_type(ai, rng)(world, actor)
        overlay.patch(world)
        result.append(BeliefWorld(world, 1.0 / count, index, before, rng.getstate()))
    return tuple(result)
