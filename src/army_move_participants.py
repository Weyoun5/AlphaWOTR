"""Exact ordinary-movement participants, including the second Army-die leg."""
from collections import Counter

from army_card_cargo import key, freeze, decode
from game_state import Army, Nazgul, UnitKind


def card_followup_qualifies(army, kind):
    """The moving army must carry the card's required Nazgul/Witch-king."""
    import game_logic as gl
    return any(gl.is_nazgul_figure(leader)
               and (kind != 'wk_army' or not isinstance(leader, Nazgul))
               for leader in army.leaders)


def inventory(army, side):
    """Immutable categories preserve unmoved figures across clones and saves."""
    import game_logic as gl
    from combat import side_of_nation
    if army is None:
        return ()
    return freeze(Counter(
        [key(u, True) for u in army.units if side_of_nation(u.nation) == side]
        + [key(l) for l in army.leaders if gl._leader_side(l) == side]))


def preexisting(army, snapshot):
    """Read-only view containing only figures present before the first leg.

    Old count-only snapshots cannot identify safe movers. Their destination
    offers no second move, while other armies and the pass remain available.
    """
    due = decode(snapshot.get('figures', ()))
    units, leaders = [], []
    for figures, output, unit in ((army.units, units, True), (army.leaders, leaders, False)):
        for figure in figures:
            category = key(figure, unit)
            if due[category] > 0:
                output.append(figure)
                due[category] -= 1
    if any(due.values()):
        raise ValueError('A preexisting army figure is missing')
    return Army(region=army.region, units=units, leaders=leaders)


def selected(army, side, params, nation_ok=None):
    """Select the same nation-filtered, counted figures as ordinary movement."""
    import game_logic as gl
    from combat import side_of_nation
    units = [u for u in army.units if side_of_nation(u.nation) == side
             and (nation_ok is None or nation_ok(u.nation))]
    chosen_units = []
    for kind, name in ((UnitKind.ELITE, 'n_elite'), (UnitKind.REGULAR, 'n_regular')):
        pool = [u for u in units if u.kind == kind]
        count = params.get(name)
        count = len(pool) if count is None else int(count)
        if count < 0 or count > len(pool):
            raise ValueError('Movement exceeds eligible army units')
        chosen_units.extend(pool[:count])
    if not chosen_units:
        raise ValueError('Army movement requires an eligible unit')
    leaders = [l for l in army.leaders if gl._leader_side(l) == side
               and not gl._is_saruman(l)
               and (nation_ok is None or not gl.leader_must_follow(l) or nation_ok(l.nation))]
    nazgul_count = params.get('n_nazgul')
    count = params.get('n_leaders')
    count = int(count) if count is not None else (len(leaders) if params.get('take_leaders', True) else 0)
    if count < 0 or (nazgul_count is not None and int(nazgul_count) < 0):
        raise ValueError('Negative movement leader count')
    ordered = [l for l in leaders if gl.leader_must_follow(l)] + [
        l for l in leaders if not gl.leader_must_follow(l)]
    if nazgul_count is None:
        chosen = {id(l) for l in ordered[:count]}
        chosen_leaders = [l for l in leaders if id(l) in chosen]
    else:
        nazgul = [l for l in leaders if isinstance(l, Nazgul)][:int(nazgul_count)]
        chosen = {id(l) for l in ordered if not isinstance(l, Nazgul)}
        others = [l for l in ordered if id(l) in chosen][:count]
        chosen = {id(l) for l in others}
        chosen_leaders = nazgul + [l for l in leaders if id(l) in chosen]
    return Army(region=army.region, units=chosen_units, leaders=chosen_leaders)
