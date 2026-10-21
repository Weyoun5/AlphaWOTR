"""Public per-figure movement history needed while a Nazgul chain is open."""
import numbers
import numpy as np

FLAGS = ('__wk_ok__', '__wk__', '__mo_ok__', '__mo__')


def movement_tail(state, region_index):
    """Active, per-region already-flown Nazgul counts /9, four Minion flags.

    Locations and Minion identities remain in the original board observation.
    This tail records which figures may still move during this paid action.
    A terminator has no arrivals payload; every actual flight must agree.
    """
    result = np.zeros(1 + len(region_index) + len(FLAGS), dtype=np.float32)
    pending = getattr(state, 'pending_card_option', None) or {}
    if pending.get('card_name') != '__nazgul_moves__' and pending.get('card_id') != '__nazgul_moves__':
        return result
    if not pending.get('direct'):
        raise ValueError('Nazgul continuation lacks direct movement context')
    options = pending.get('options')
    if not isinstance(options, (list, tuple)) or not options:
        raise ValueError('Nazgul continuation has no public options')
    flights = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError('Invalid Nazgul continuation option')
        if option.get('_naz_done'):
            continue
        if not isinstance(option.get('_naz_arr'), dict):
            raise ValueError('Nazgul flight lacks public arrival provenance')
        flights.append(option['_naz_arr'])
    if not flights:
        # Existing executors automatically finish exhausted flight chains.
        raise ValueError('Nazgul continuation has no remaining flight')
    history = flights[0]
    if any(value != history for value in flights[1:]):
        raise ValueError('Conflicting Nazgul arrival histories')
    result[0] = 1.
    for key, value in history.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError('Invalid Nazgul history count')
        if key in FLAGS:
            if value not in (0, 1):
                raise ValueError('Invalid Nazgul Minion eligibility flag')
            result[1 + len(region_index) + FLAGS.index(key)] = value
        else:
            if key not in region_index or not 0 <= value <= 9:
                raise ValueError('Invalid Nazgul arrival region or count')
            result[1 + region_index[key]] = value / 9.
    if history.get('__wk__') and not history.get('__wk_ok__'):
        raise ValueError('Moved Witch-king was not eligible for this chain')
    if history.get('__mo__') and not history.get('__mo_ok__'):
        raise ValueError('Moved Mouth was not eligible for this chain')
    return result
