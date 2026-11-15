"""Named figures that have already moved on the current Character die."""
import numpy as np

NAMES = ('Gandalf the Grey', 'Strider', 'Legolas', 'Gimli', 'Boromir',
         'Meriadoc', 'Peregrin', 'Aragorn', 'Gandalf the White')
TAIL_WIDTH = 1 + len(NAMES)

def movement_tail(state):
    tail = np.zeros(TAIL_WIDTH, dtype=np.float32)
    pending = getattr(state, 'pending_card_option', None) or {}
    if pending.get('card_id') != '__companion_moves__':
        return tail
    if not pending.get('direct'):
        raise ValueError('Companion continuation lacks direct movement provenance')
    options = pending.get('options')
    if not isinstance(options, (list, tuple)) or not options:
        raise ValueError('Companion continuation has no public options')
    histories = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError('Malformed Companion movement option')
        if option.get('_comp_done'):
            continue
        history = option.get('_comp_moved')
        if (not isinstance(history, (list, tuple)) or not history or
                any(not isinstance(name, str) or name not in NAMES for name in history) or
                len(set(history)) != len(history)):
            raise ValueError('Missing or malformed already-moved Companion names')
        histories.append(frozenset(history))
    if not histories or any(history != histories[0] for history in histories):
        raise ValueError('Companion continuation has missing or conflicting histories')
    tail[0] = 1
    for name in histories[0]:
        tail[1 + NAMES.index(name)] = 1
    return tail
