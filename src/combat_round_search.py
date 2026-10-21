"""Search identity for a partially resolved, publicly observed combat round."""
from enum import Enum


def round_signature(payload):
    """Preserve cursor/figure aliases while excluding the enclosing GameState.

    Rolls, payments, surviving targets and the next stage all affect future
    choices. A board-only signature would incorrectly merge those branches.
    """
    if payload is None:
        return None
    seen = {}

    def freeze(value):
        if isinstance(value, Enum):
            return (type(value).__module__, type(value).__name__, value.name)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if callable(value):
            return ('callable', value.__module__, value.__qualname__)
        ident = id(value)
        if ident in seen:
            return ('ref', seen[ident])
        seen[ident] = len(seen)
        if isinstance(value, dict):
            return ('dict', tuple((freeze(k), freeze(v)) for k, v in
                                  sorted(value.items(), key=lambda kv: str(kv[0]))
                                  if k != '_live_state'))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(freeze(v) for v in value))
        if isinstance(value, (set, frozenset)):
            return ('set', tuple(freeze(v) for v in sorted(value, key=str)))
        if hasattr(value, '__dict__'):
            return (type(value).__module__, type(value).__name__, freeze(vars(value)))
        raise TypeError('Unsupported combat signature value: ' + type(value).__name__)

    return freeze(payload)


ADAPTIVE_CARDS = frozenset({
    'dread and despair', 'relentless assault', 'mighty attack',
    'fateful strike', 'blade of westernesse', 'heroic death', 'black breath',
    'charge', 'sudden strike', "durin's bane",
})


def requires_phase_sampling(parent_state, probe_state):
    """The legacy whole-round kernel cannot cross an unresolved owner choice."""
    if (getattr(parent_state, 'pending_combat', None) or {}).get('_round_resume'):
        return True
    if (getattr(probe_state, 'pending_combat', None) or {}).get('_round_resume'):
        return True
    info = getattr(probe_state, '_last_round_call', None) or {}
    if not (info.get('choices') or {}).get('defer_round_choices'):
        return False
    from cards import get_card
    for key in ('atk_card', 'def_card'):
        cid = info.get(key)
        if cid:
            card = get_card(cid)
            if card and card.combat_name.strip().lower() in ADAPTIVE_CARDS:
                return True
    return False
