"""Exact, factored army cargo choices for Shadow movement Events.

Figures of the same kind are exchangeable. Routes and each available figure
category are selected separately, avoiding a Cartesian product of all subsets.
Nothing moves until the last category has been selected.
"""
from collections import Counter
from board import sanitize
from game_state import Side, Nazgul


def key(figure, unit=False):
    if unit:
        return ('unit', figure.nation.name, figure.kind.name)
    if isinstance(figure, Nazgul):
        return ('nazgul', 'SAURON', '')
    name = getattr(figure, 'name', None)
    if name:
        return ('named', str(name), '')
    return ('leader', figure.nation.name, '')


def figures(state, source):
    import game_logic as gl
    from combat import side_of_nation
    army = next((army for region, army in state.armies.items()
                 if sanitize(region) == sanitize(source)), None)
    if army is None:
        raise ValueError('Cargo source has no army')
    units = [u for u in army.units if side_of_nation(u.nation) == Side.SHADOW]
    leaders = [l for l in army.leaders if gl._leader_side(l) == Side.SHADOW
               and not gl._is_saruman(l)]
    return units, leaders


def inventory(state, source):
    units, leaders = figures(state, source)
    return Counter([key(u, True) for u in units] + [key(l) for l in leaders])


def freeze(counts):
    return tuple((*k, count) for k, count in sorted(counts.items()) if count)


def decode(cargo):
    if not isinstance(cargo, (tuple, list)):
        raise ValueError('Cargo must be a sequence')
    result = Counter()
    for row in cargo:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError('Malformed cargo row')
        kind, name, subtype, count = row
        if any(not isinstance(v, str) for v in row[:3]) or type(count) is not int or count <= 0:
            raise ValueError('Malformed cargo count')
        category = kind, name, subtype
        if category in result:
            raise ValueError('Duplicate cargo category')
        result[category] = count
    return result


def selected(state, source, cargo):
    wanted = decode(cargo)
    available = inventory(state, source)
    if not any(k[0] == 'unit' and n for k, n in wanted.items()):
        raise ValueError('Army cargo must contain an army unit')
    if any(n > available[k] for k, n in wanted.items()):
        raise ValueError('Cargo exceeds source inventory')
    units, leaders = figures(state, source)
    selected_units, selected_leaders = [], []
    for items, out, is_unit in ((units, selected_units, True), (leaders, selected_leaders, False)):
        for item in items:
            category = key(item, is_unit)
            if wanted[category]:
                out.append(item)
                wanted[category] -= 1
    return selected_units, selected_leaders


def stage(state, card, choices, source):
    """Return completed choices, or None after opening a cargo count menu.

Legacy direct replay callers retain their explicit aggregate contract. The
public pending path always passes through the factored exact selection.
"""
    if not choices.get('_from_pending') or choices.get('_cargo_complete'):
        return choices
    available = inventory(state, source)
    categories = sorted(available, key=lambda k: (k[0] != 'unit', k))
    if not any(k[0] == 'unit' for k in categories):
        raise ValueError('No movable army cargo')
    if '_cargo_index' in choices:
        index = choices['_cargo_index']
        cargo = decode(choices.get('cargo', ()))
        if type(index) is not int or not 0 <= index < len(categories):
            raise ValueError('Invalid cargo selection stage')
        category = categories[index]
        if tuple(choices.get('_cargo_key', ())) != category:
            raise ValueError('Cargo category changed')
        count = choices.get('count')
        if type(count) is not int or not 0 <= count <= available[category] or cargo[category] != count:
            raise ValueError('Invalid selected cargo count')
        if any(k not in categories[:index + 1] or n > available[k] for k, n in cargo.items()):
            raise ValueError('Invalid accumulated cargo')
        continuation = dict(choices['_cargo_continue'])
        index += 1
    else:
        index, cargo = 0, Counter()
        continuation = dict(choices)
    while index < len(categories):
        category = categories[index]
        minimum = 0
        if category[0] == 'unit' and not any(k[0] == 'unit' for k in categories[index + 1:]):
            minimum = int(not any(k[0] == 'unit' and n for k, n in cargo.items()))
        options = []
        for count in range(minimum, available[category] + 1):
            next_cargo = cargo.copy()
            next_cargo[category] = count
            from shadow_join_rules import cargo_possible
            if not cargo_possible(state, card.card_name, continuation,
                                  next_cargo, categories[index + 1:]):
                continue
            options.append(dict(continuation, cargo=freeze(next_cargo), count=count,
                                _cargo_key=category, _cargo_index=index,
                                _cargo_continue=continuation))
        if not options:
            raise ValueError('Army cargo cannot cross political borders')
        if len(options) > 1:
            state.pending_card_option = {
                'card_id': card.card_id, 'card_name': card.card_name,
                'side': 'sh', 'decider': 'sh', 'direct': True,
                'options': options, 'cargo_source': sanitize(source)}
            return None
        cargo = decode(options[0]['cargo'])
        index += 1
    selected(state, source, freeze(cargo))
    return dict(continuation, cargo=freeze(cargo), _cargo_complete=True)
