"""Versioned learning semantics; old buffers must never acquire new labels.

The win/loss contract predicts the result from a player's actual decision
state. Intermediate rewards and shaping are zero. TD(lambda) propagates
the result backwards through that player's decisions; policy learns search
targets from both winning and losing games.
"""
import os

LEGACY = 'legacy_profile_v1'
WINLOSS = 'predecision_winloss_v1'
PROFILE = 'predecision_profile_v1'
PREDECISION = frozenset((WINLOSS, PROFILE))
_ROUTES = {'fp': frozenset(('military', 'ring_destroyed')),
           'sh': frozenset(('military', 'ring_corruption'))}
_TARGET = {('fp', 'ring'): 'ring_destroyed', ('fp', 'military'): 'military',
           ('sh', 'corruption'): 'ring_corruption', ('sh', 'military'): 'military'}


def current_contract():
    name = os.environ.get('WOTR_REWARD_CONTRACT', LEGACY)
    if name not in (LEGACY, WINLOSS, PROFILE):
        raise ValueError('Unknown WOTR_REWARD_CONTRACT: %r' % name)
    return name


def uses_winloss():
    return current_contract() == WINLOSS


def uses_predecision(contract=None):
    return (current_contract() if contract is None else contract) in PREDECISION


def validate_profile(side, profile):
    if side not in _ROUTES or (profile != 'mixed' and (side, profile) not in _TARGET):
        raise ValueError('Invalid profile for %s: %r' % (side, profile))


def validate_net_profile(net, side, profile):
    """A trained profile head cannot silently play another objective.

    Unstamped corpus nets are permitted as explicit warm-start inputs.
    """
    validate_profile(side, profile)
    recorded = getattr(net, 'reward_profile', None)
    if recorded is not None and recorded != profile:
        raise ValueError('%s checkpoint profile %r differs from assigned %r'
                         % (side, recorded, profile))


def profile_terminal(side, profile, victory_type, won):
    """Fixed versioned mission weights; losses remain worse than any win.

    The off-target win weights preserve the existing diagnostic objectives.
    No environment-dependent shaping, microdecision cost, closeness bonus,
    or turn decay is part of this contract.
    """
    validate_profile(side, profile)
    winner = side if won else ('sh' if side == 'fp' else 'fp')
    if victory_type not in _ROUTES[winner]:
        raise ValueError('Invalid victory route for winner %s: %r' % (winner, victory_type))
    if not won:
        return -1.0
    if profile == 'mixed' or victory_type == _TARGET.get((side, profile)):
        return 1.0
    return 0.0 if side == 'fp' else 0.2


def trace_contract(trace):
    return (trace.get('meta') or {}).get('reward_contract', LEGACY)


def terminal(winner, side, *, profile=None, victory_type=None, contract=None):
    if winner not in ('fp', 'sh') or side not in ('fp', 'sh'):
        raise ValueError('A decided game and valid side are required')
    contract = current_contract() if contract is None else contract
    if contract == PROFILE:
        return profile_terminal(side, profile, victory_type, winner == side)
    if contract != WINLOSS:
        raise ValueError('Use the legacy reward implementation for %r' % contract)
    return 1.0 if winner == side else -1.0
