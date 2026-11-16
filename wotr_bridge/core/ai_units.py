"""WHICH units an AI seat takes, where its step names a COUNT of units and its army holds several nations.

The search and the nets count units; they do not see nations: "move 1 Regular" takes the first Regular of the
army's roster, and the Elite that pays for a pressed siege is the first one that can.  Wherever the AI's own option
NAMES the nation (a casualty, a removal over the stacking limit, the trim of a garrison, the victor's advance, a
card's army by kind) that is its choice and is not touched.  For the two steps that only count, the units are named
here, AFTER the step is chosen, the way a person's own pieces reach the engine (menu.realise): an army move gets the
engine's own parameter for the units that go (`unit_set`), a pressed siege the nation of the Elite that pays
(`nation`).  The engine draws from the same pool as before (the units that may cross the border, those a first move
of the die has not brought in); a set it would refuse is never named.

THE SHADOW.  Owner: "try to keep 2 sauron regulars for hill trolls. always keep at least 1 isengard unit of any
type. for elites, isengard > southron > sauron units".  An army that is split has a main body - the group its
Leaders go with, or with none the larger one (the one that stays, when equal) - and the other group gets the units
the rule gives up first (give_up), counted over that army:
  1. never the army's last Isengard unit, while another nation's unit of that kind can go ("always");
  2. not a Sauron Regular that leaves the main body fewer than two ("try to": the card Hill-trolls upgrades two
     Sauron Regulars, wherever they stand);
  3. else by what is kept longest - Regulars: Sauron, then Isengard, then Southrons & Easterlings; Elites: Isengard,
     then Southrons & Easterlings, then Sauron.
The Elite that pays for a pressed siege is reduced, not lost: only 3 speaks.

THE FREE PEOPLES.  Owner: "for free people prefer nations at war".  The units that GO with a move are those of
nations At War first; among nations of one standing the roster's order stays as it was.
A step that leaves no such choice, or that the rule would fill as the roster does, is returned as it came.
"""
import collections

SHADOW = ('SAURON', 'ISENGARD', 'SOUTHRON_EASTERLINGS')
PAIR = 2            # the Sauron Regulars the card Hill-trolls upgrades
GIVEN_FIRST = {'REGULAR': ('SOUTHRON_EASTERLINGS', 'ISENGARD', 'SAURON'),
               'ELITE': ('SAURON', 'SOUTHRON_EASTERLINGS', 'ISENGARD')}


def give_up(pool, n_regular, n_elite, lost=True):
    """Of the Shadow's `pool` ({(NATION, KIND): how many}) the units that go: n_regular Regulars and n_elite
    Elites, one at a time by THE RULE (the Isengard units and the Sauron Regulars counted in the pool as it
    shrinks).  lost False: nothing leaves (an Elite is reduced).  -> Counter."""
    left, gone = collections.Counter(pool), collections.Counter()
    for kind, n in (('REGULAR', n_regular), ('ELITE', n_elite)):
        for _ in range(int(n or 0)):
            nations = [x for x in GIVEN_FIRST[kind] if left[(x, kind)] > 0]
            if not nations:
                break
            isengard = sum(v for (x, _k), v in left.items() if x == 'ISENGARD')
            pick = min(nations, key=lambda x: (
                lost and x == 'ISENGARD' and isengard <= 1,
                lost and kind == 'REGULAR' and x == 'SAURON' and left[('SAURON', 'REGULAR')] <= PAIR,
                GIVEN_FIRST[kind].index(x)))
            left[(pick, kind)] -= 1
            gone[(pick, kind)] += 1
    return gone


def detached(pool, n_regular, n_elite, main_goes):
    """An army of the Shadow (`pool`) is split, n_regular Regulars and n_elite Elites go: -> the units that GO,
    the main body keeping what THE RULE keeps longest."""
    if not main_goes:
        return give_up(pool, n_regular, n_elite)
    regulars = sum(v for (_n, k), v in pool.items() if k == 'REGULAR')
    stay = give_up(pool, regulars - n_regular, sum(pool.values()) - regulars - n_elite)
    return collections.Counter(pool) - stay


def at_war_first(units, n_regular, n_elite, at_war):
    """Of the Free Peoples' `units` (in the roster's order) those that GO: for each kind the nations At War
    first.  at_war(nation) -> bool.  -> Counter {(NATION, KIND): how many}."""
    gone = collections.Counter()
    for kind, n in (('REGULAR', n_regular), ('ELITE', n_elite)):
        of_kind = [u for u in units if u.kind.name == kind]
        for u in sorted(of_kind, key=lambda u: not at_war(u.nation))[:int(n or 0)]:      # (a stable sort)
            gone[_key(u)] += 1
    return gone


def _key(unit):
    return unit.nation.name, unit.kind.name


def _move(ai, state, side, act):
    """An army move by counts out of an army of several nations -> the same move naming the units that go."""
    import army_move_participants as engine
    p = act.params or {}
    if act.action_type != 'army_move' or p.get('unit_set') is not None or not p.get('from') or not p.get('to') \
            or (p.get('n_regular') is None and p.get('n_elite') is None):
        return None
    army = ai._armies_lookup(state, p['from'])
    if army is None:
        return None
    # the pool the engine's own executor draws from: the border predicate of the WHOLE army, and - for the
    # second move of a die - only the units its first move did not bring in
    here = ai.sanitize(p['from']) == ai.sanitize(p['to'])
    may = None if here else ai._border_pred9(
        state, [u for u in army.units if ai._side_of_nation(u.nation) == side], army.leaders, side, ai.sanitize(p['to']))
    source = army
    if state.pending_army_move_die is not None and ai.sanitize(p['from']) == ai.sanitize(state.pending_army_move_from or ''):
        source = engine.preexisting(army, state.pending_army_move_preexist or {})
    try:
        as_listed = engine.selected(source, side, p, may)
    except ValueError:
        return None
    units = [u for u in source.units if ai._side_of_nation(u.nation) == side and (may is None or may(u.nation))]
    going = collections.Counter(u.kind.name for u in as_listed.units)
    n_r, n_e = going['REGULAR'], going['ELITE']
    if not 0 < n_r + n_e < len(units):
        return None                     # (the whole army goes, or none of its units)
    if side.name == 'SHADOW':
        pool = collections.Counter(_key(u) for u in units if u.nation.name in SHADOW)
        if sum(pool.values()) != len(units):
            return None
        leaders = [x for x in source.leaders if ai.game_logic._leader_side(x) == side]
        main_goes = bool(as_listed.leaders) if leaders else n_r + n_e > len(units) - n_r - n_e
        named = detached(pool, n_r, n_e, main_goes)
    else:
        named = at_war_first(units, n_r, n_e, state.political.is_at_war)
    if named == collections.Counter(_key(u) for u in as_listed.units):
        return None                     # (what the roster's order takes anyway)
    unit_set = tuple(('unit', n, k, v) for (n, k), v in sorted(named.items()) if v)      # (army_card_cargo.freeze)
    better = type(act)(act.die_idx, act.action_type, dict(p, unit_set=unit_set))
    try:
        engine.selected(source, side, better.params, may)
    except ValueError:
        return None
    return better


def _press(ai, state, side, act):
    """The Shadow presses a siege: the Elite that pays is the one THE RULE gives up (it is reduced, not lost)."""
    p, battle = act.params or {}, getattr(state, 'pending_combat', None) or {}
    if act.action_type != 'siege_press' or side.name != 'SHADOW' or p.get('nation') is not None \
            or p.get('no_reduce') or not battle.get('attacker_region') or battle.get('attacker_side') is not side:
        return None
    region = ai.sanitize(battle['attacker_region'])
    elites = [u for u in ai._alloc_side_units(state, region, side, ai.UnitKind.ELITE) if u.nation.name in SHADOW]
    if len({u.nation for u in elites}) < 2:
        return None
    nation = next(iter(give_up(collections.Counter(_key(u) for u in elites), 0, 1, lost=False)))[0]
    payer = next(u for u in elites if u.nation.name == nation)
    if payer is elites[0]:
        return None                     # (the one the roster's order takes anyway)
    return type(act)(act.die_idx, act.action_type, dict(p, nation=payer.nation))


def pick(ai, state, side, act):
    """An AI seat's chosen step `act` with the units its rule names (the module's text); any other step as it
    came.  ai: the engine's module (the caller has it)."""
    for reader in (_move, _press):
        better = reader(ai, state, side, act)
        if better is not None:
            return better
    return act
