"""Exact ordinary-movement participants, including the second Army-die leg."""
from collections import Counter
from itertools import product

from army_card_cargo import key, freeze, decode
from game_state import Army, Nazgul, UnitKind


def figure_sets(figures, size, unit=False):
    """E5-01 (D5, 2026-10-22): every DISTINCT set of `size` figures drawn
    from `figures`, keyed by army_card_cargo.key (named figures distinct,
    same-nation plain Leaders / same nation+kind units interchangeable), as
    army_card_cargo.freeze tuples.  Deterministic order: categories sorted
    by key, counts enumerated like itertools.product (first category
    slowest, each count ascending).  The native menus
    (native_backend/*/cpp army_move_menu.inc, combat_phase_menu.inc)
    enumerate the same order."""
    counts = Counter(key(f, unit) for f in figures)
    cats = sorted(counts)
    out = []
    for combo in product(*(range(counts[c] + 1) for c in cats)):
        if sum(combo) == size:
            out.append(tuple((*c, n) for c, n in zip(cats, combo) if n))
    return out


def pick(figures, cargo, unit=False):
    """The figures named by the frozen cargo `cargo`, first matches in list
    order.  Loud ValueError when the source cannot supply the set."""
    due = decode(cargo)
    if any((k[0] == 'unit') != unit or k[0] == 'nazgul' for k in due):
        raise ValueError('Chosen figure set names the wrong figure class')
    out = []
    for figure in figures:
        category = key(figure, unit)
        if due[category] > 0:
            out.append(figure)
            due[category] -= 1
    if any(due.values()):
        raise ValueError('A chosen army figure is not available to move')
    return out


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
    if params.get('unit_set') is not None:
        # L1104-W06 (C06/C12): the row NAMES its units (army_card_cargo keys,
        # freeze format) - a nation-mixed army's partial move (1 Gondor + 1
        # Rohan Regular; 1 S&E + 4 Sauron Regulars out of Erebor).  Drawn
        # from the same nation-filtered pool; a set the pool cannot supply,
        # or one disagreeing with the counts, is refused loudly.
        named = pick(units, params['unit_set'], unit=True)
        if Counter(u.kind for u in named) != Counter(u.kind for u in chosen_units):
            raise ValueError('unit_set disagrees with the unit counts')
        chosen_units = named
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
    leader_set = params.get('leader_set')
    if leader_set is not None:
        # E5-01 (D5): the row NAMES its non-Nazgul figures (army_card_cargo
        # keys); Nazgul stay on n_nazgul.  A set the source cannot supply,
        # or one disagreeing with n_leaders, is refused loudly.
        named = pick([l for l in leaders if not isinstance(l, Nazgul)], leader_set)
        if params.get('n_leaders') is None or len(named) != count:
            raise ValueError('leader_set disagrees with n_leaders')
        chosen = {id(l) for l in named}
        nazgul = ([] if nazgul_count is None else
                  [l for l in leaders if isinstance(l, Nazgul)][:int(nazgul_count)])
        chosen_leaders = nazgul + [l for l in leaders if id(l) in chosen]
    elif nazgul_count is None:
        chosen = {id(l) for l in ordered[:count]}
        chosen_leaders = [l for l in leaders if id(l) in chosen]
    else:
        nazgul = [l for l in leaders if isinstance(l, Nazgul)][:int(nazgul_count)]
        chosen = {id(l) for l in ordered if not isinstance(l, Nazgul)}
        others = [l for l in ordered if id(l) in chosen][:count]
        chosen = {id(l) for l in others}
        chosen_leaders = nazgul + [l for l in leaders if id(l) in chosen]
    return Army(region=army.region, units=chosen_units, leaders=chosen_leaders)
