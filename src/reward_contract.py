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
# Off-target win weights (1.225b, USER 2026-08-03 "zero off mission for fp but keep it for shadow").
OFF_TARGET = {'fp': 0.0, 'sh': 0.2}
# A league net is searched by its own reward table (hpc league lgcommon.OBJECTIVES; USER 2026-10-30 "SH-CORRUPTION:
# +1 corruption win, 0 military win, -1 FP win", "FP-MILITARY: +1 FP military win, 0 Ring destroyed, -1 SH win"):
# the off-target win weight of the objective its checkpoint's 'league' record names (SEARCHCOR 2026-11-01).
LEAGUE_OFF_TARGET = {'fp_military': 0.0, 'sh_corruption': 0.0}


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


def league_off_target(side, league):
    """The off-target win weight of a league checkpoint's reward table (its 'league' record), or None (not a
    league net, or an objective whose off-target route is on-target: the contract weights apply)."""
    objective = league.get('objective') if isinstance(league, dict) else None
    if objective not in LEAGUE_OFF_TARGET:
        return None
    if not objective.startswith(side + '_'):
        raise ValueError('League objective %r does not belong to side %r' % (objective, side))
    return LEAGUE_OFF_TARGET[objective]


def profile_terminal(side, profile, victory_type, won, off_target=None):
    """Fixed versioned mission weights; losses remain worse than any win.

    The off-target win weights preserve the existing diagnostic objectives.
    No environment-dependent shaping, microdecision cost, closeness bonus,
    or turn decay is part of this contract.  off_target: the searching net's
    own off-target win weight (league_off_target); None keeps OFF_TARGET.
    """
    validate_profile(side, profile)
    winner = side if won else ('sh' if side == 'fp' else 'fp')
    if victory_type not in _ROUTES[winner]:
        raise ValueError('Invalid victory route for winner %s: %r' % (winner, victory_type))
    if not won:
        return -1.0
    if profile == 'mixed' or victory_type == _TARGET.get((side, profile)):
        return 1.0
    if off_target is None:
        return OFF_TARGET[side]
    off_target = float(off_target)
    if not -1.0 < off_target <= 1.0:
        raise ValueError('Off-target win weight must lie in (-1, 1]: %r' % (off_target,))
    return off_target


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
