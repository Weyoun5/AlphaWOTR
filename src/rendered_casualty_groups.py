"""Equivalent first choices within a simultaneous group of Regular losses.

Only same-line one-hit Regular removals commute here. Elite swaps, supplies,
different physical lines and different owners retain their existing checks.
"""
import copy


def positions(events, cursor):
    first = events[cursor]
    if first[2] != 'loss_regular':
        return [cursor]
    result = []
    for index in range(cursor, len(events)):
        if events[index][0] != first[0] or events[index][2] != 'loss_regular':
            break
        result.append(index)
    return result


def consume(events, cursor, witness, matches):
    if cursor >= len(events):
        return False
    for index in positions(events, cursor):
        if matches(events[index], witness):
            events[cursor], events[index] = events[index], events[cursor]
            return True
    return False


def signature(action):
    return (action.action_type, action.die_idx, action.params)


def annotate(state, legal, chosen, events, cursor, *, card=False):
    group = positions(events, cursor)
    if len(group) < 2:
        return
    nations = {events[index][1] for index in group}
    alternatives = []
    for action in legal:
        params = (action.params.get('choices') or {}) if card else action.params
        operation = params.get('casualty_operation' if card else 'operation')
        nation = params.get('nation')
        if (action.action_type == ('card_option' if card else 'casualty_allocate')
                and operation == 'remove_regular' and getattr(nation, 'name', nation) in nations):
            alternatives.append(copy.deepcopy(signature(action)))
    if len(alternatives) >= 2:
        if signature(legal[chosen]) not in alternatives:
            raise ValueError('Grouped casualty target does not match its legal menu')
        state._rx_grouped_casualty_targets = alternatives


def indices(state, legal, chosen, phase):
    targets = getattr(state, '_rx_grouped_casualty_targets', None)
    if not targets:
        return []
    if phase not in ('casualty_alloc', 'card_option'):
        raise ValueError('Grouped casualty targets attached to unrelated phase')
    found = [i for i, action in enumerate(legal) if signature(action) in targets]
    if chosen not in found or len(found) != len(targets):
        raise ValueError('Grouped casualty targets no longer match the legal menu')
    return found
