"""Validated public reveal/declaration event and conditional stronghold order."""
import numpy as np
from hunt_protection_context import STRONGHOLDS

KINDS = ('reveal', 'declare')
TAIL_WIDTH = 67


def event_context(state):
    event = getattr(state, '_fellowship_route_event', None)
    if event is None:
        return None
    if (not isinstance(event, dict) or
            set(event) != {'kind', 'balrog_due', 'conditional_strongholds'} or
            event['kind'] not in KINDS or type(event['balrog_due']) is not bool):
        raise ValueError('Malformed Fellowship route event')
    queue = event['conditional_strongholds']
    if (not isinstance(queue, (tuple, list)) or len(queue) > len(STRONGHOLDS) or
            any(not isinstance(region, str) or region not in STRONGHOLDS for region in queue) or
            len(set(queue)) != len(queue)):
        raise ValueError('Malformed conditional stronghold order')
    return event['kind'], event['balrog_due'], tuple(queue)


def clone_event(state):
    event = event_context(state)
    if event is None:
        return None
    return {'kind': event[0], 'balrog_due': event[1], 'conditional_strongholds': event[2]}


def observation_tail(state):
    tail = np.zeros(TAIL_WIDTH, dtype=np.float32)
    event = event_context(state)
    if event is None:
        return tail
    kind, due, queue = event
    tail[KINDS.index(kind)] = 1
    tail[2] = float(due)
    for position, region in enumerate(queue):
        tail[3 + position * 8 + STRONGHOLDS.index(region)] = 1
    return tail
