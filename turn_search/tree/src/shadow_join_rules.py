"""Political borders for the two Shadow Events that join existing armies."""
from board import sanitize
from game_state import Side, Nation

CARDS = {'Shadows Gather': 3, 'The Shadow Lengthens': 2}


def paths(state, source, target, steps):
    from event_effects import _army_reachable_free, _besieged_garrison_source
    source, target = sanitize(source), sanitize(target)
    if source == target or _besieged_garrison_source(state, source, Side.SHADOW):
        return []
    result, queue = [], [(source, ())]
    while queue:
        current, route = queue.pop(0)
        if len(route) >= steps:
            continue
        for region in sorted(_army_reachable_free(state, current, 1, Side.SHADOW)):
            if region == source or region in route:
                continue
            walked = (*route, region)
            if region == target:
                result.append(walked)
            elif region not in state.besieged_strongholds:
                queue.append((region, walked))
    return result


def may_follow(state, nation, route):
    from ai import _region_nation
    return state.political.is_at_war(nation) or all(
        _region_nation(region) in (None, nation) for region in route)


def select(state, source, target, steps, units, exact=False):
    """Choose one common legal path; passive figures may remain at source."""
    best, mids = [], None
    for route in paths(state, source, target, steps):
        eligible = [u for u in units if may_follow(state, u.nation, route)]
        if exact and len(eligible) != len(units):
            continue
        if len(eligible) > len(best):
            best, mids = eligible, list(route[:-1])
    return best, mids


def cargo_possible(state, card_name, choices, cargo, remaining=()):
    """A partial cargo must fit one route and be completable with an army unit."""
    if card_name not in CARDS:
        return True
    nations = {Nation[k[1]] for k, n in cargo.items() if k[0] == 'unit' and n}
    later = {Nation[k[1]] for k in remaining if k[0] == 'unit'}
    return any(all(may_follow(state, n, route) for n in nations)
               and (bool(nations) or any(may_follow(state, n, route) for n in later))
               for route in paths(state, choices['from_region'], choices['to_region'], CARDS[card_name]))
