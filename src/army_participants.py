"""Deterministic attack formation and frozen battle participant views.

RB28/35: peaceful Army units and plain Leaders cannot attack. Every rear
Army must contain an Army unit. Companions, Nazgul and Minions are already
belligerent. The user's AI convention commits all eligible figures except
the minimum necessary guard (Regular before Elite, stable nation tie-break).

This module does not decide adjacency, card eligibility, siege access,
movement/advance, payment, or human voluntary composition. In particular,
Saruman's inability to advance does not prevent his attack (Almanac8).
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import game_state as gs

FORMATION_VERSION = 'forced_rearguard_v1'
_SHADOW_NATIONS = frozenset((gs.Nation.ISENGARD, gs.Nation.SAURON,
                           gs.Nation.SOUTHRON_EASTERLINGS))
_MINION_NAMES = frozenset(('Saruman', 'Witch King', 'Mouth of Sauron'))


@dataclass(frozen=True)
class AttackFormation:
    """Immutable containers of the ORIGINAL figure references.

    No figure is copied, moved or mutated. ``rearguard_signature`` is a
    captured, immutable count representation suitable for later persistence;
    source region and side must be recorded by the eventual battle caller.
    Figure references themselves remain mutable game objects, not snapshots.
    """
    attacking_units: tuple
    attacking_leaders: tuple
    rearguard_units: tuple
    rearguard_leaders: tuple
    bystander_units: tuple
    bystander_leaders: tuple
    guard_added: bool
    character_action_ok: bool
    rearguard_signature: tuple
    unavailable_reason: Optional[str] = None

    @property
    def available(self):
        return self.unavailable_reason is None


def _nation_side(nation):
    if not isinstance(nation, gs.Nation):
        raise ValueError('Army figure has an unknown nation: %r' % (nation,))
    return gs.Side.SHADOW if nation in _SHADOW_NATIONS else gs.Side.FREE_PEOPLES


def _is_minion(figure):
    # Actual board Minion is game_logic.Minion, a Leader subclass. Importing
    # game_logic here would recreate its ai/combat dependency cycle. Its three
    # canonical board names distinguish it from a plain Leader; tracker-only
    # ShadowMinion/Gandalf/Aragorn objects are not board figures.
    return isinstance(figure, gs.Leader) \
        and getattr(figure, 'name', None) in _MINION_NAMES


def _leader_side(figure):
    if isinstance(figure, gs.Companion):
        return gs.Side.FREE_PEOPLES
    if isinstance(figure, gs.Nazgul) or _is_minion(figure):
        return gs.Side.SHADOW
    if isinstance(figure, gs.Leader) and not getattr(figure, 'name', None):
        return _nation_side(figure.nation)
    raise ValueError('Unsupported board leader/character: %r' % (figure,))


def _at_war(state, nation):
    _nation_side(nation)  # validate the domain before reading political state
    try:
        box = state.political.boxes[nation]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError('Missing political position for %s' % nation.name) from exc
    if type(box) is not int or not 0 <= box <= 3:
        raise ValueError('Invalid political position for %s: %r' % (nation.name, box))
    return box == 3


def _saruman_present(state):
    return any(_is_minion(figure) and figure.name == 'Saruman'
               for army in getattr(state, 'armies', {}).values()
               for figure in army.leaders)


def _rear_signature(units, leaders):
    unit_counts = Counter((u.nation.name, u.kind.name) for u in units)
    # Only plain political Leaders can be forced to remain by these rules.
    leader_counts = Counter(l.nation.name for l in leaders)
    return (FORMATION_VERSION,
            tuple((nation, kind, n) for (nation, kind), n in sorted(unit_counts.items())),
            tuple(sorted(leader_counts.items())))


def select_forced_attack(state, army, side, *, require_leader=False):
    """Select the user's maximal eligible attacking force without mutation.

    ``available`` concerns formation only. The engine must still validate
    the target, the action's actual die/card payment, and other attack rules.
    ``require_leader`` applies the Character attack requirement to the
    participating force, including Saruman's eligible Isengard Elites.

    Enemy figures in a shared siege entry are bystanders to THIS partition.
    A stable nation-name order chooses among otherwise equivalent guards;
    identical copies have no distinct rules identity and keep input order.
    """
    if not isinstance(side, gs.Side):
        raise ValueError('Attack formation requires an actual Side')
    if type(require_leader) is not bool:
        raise ValueError('require_leader must be a bool')
    if army is None:
        raise ValueError('Attack formation requires an Army')
    attacking_units, rear_units, bystander_units = [], [], []
    attacking_leaders, rear_leaders, bystander_leaders = [], [], []
    for unit in army.units:
        if not isinstance(unit, gs.Unit) or not isinstance(unit.kind, gs.UnitKind):
            raise ValueError('Unsupported Army unit: %r' % (unit,))
        if _nation_side(unit.nation) != side:
            bystander_units.append(unit)
        elif _at_war(state, unit.nation):
            attacking_units.append(unit)
        else:
            rear_units.append(unit)
    for leader in army.leaders:
        if _leader_side(leader) != side:
            bystander_leaders.append(leader)
        elif isinstance(leader, (gs.Companion, gs.Nazgul)) or _is_minion(leader) \
                or _at_war(state, leader.nation):
            attacking_leaders.append(leader)
        else:
            rear_leaders.append(leader)

    guard_added = False
    if rear_leaders and not rear_units and attacking_units:
        guard_idx = min(range(len(attacking_units)), key=lambda i: (
            attacking_units[i].kind != gs.UnitKind.REGULAR,
            attacking_units[i].nation.name))
        rear_units.append(attacking_units.pop(guard_idx))
        guard_added = True

    has_leader = bool(attacking_leaders) or (
        any(u.nation == gs.Nation.ISENGARD and u.kind == gs.UnitKind.ELITE
            for u in attacking_units) and _saruman_present(state))
    character_ok = bool(attacking_units) and has_leader
    if not attacking_units:
        unavailable = ('no_attacking_unit_after_guard' if guard_added
                       else 'no_eligible_attacking_unit')
    elif require_leader and not has_leader:
        unavailable = 'no_participating_leader_for_character_action'
    else:
        unavailable = None
    return AttackFormation(
        tuple(attacking_units), tuple(attacking_leaders),
        tuple(rear_units), tuple(rear_leaders),
        tuple(bystander_units), tuple(bystander_leaders),
        guard_added, character_ok, _rear_signature(rear_units, rear_leaders),
        unavailable)


def freeze_attack(state, context, formation=None):
    """Record formation once, before the battle changes politics or figures.

    The payload contains only immutable tuples/strings/counts. It survives
    clone/trace round trips without references to the original Army objects.
    An empty rearguard is recorded too: it must not be recalculated later.
    """
    if context.get('attacker_rearguard') is not None:
        return context['attacker_rearguard']
    from board import sanitize
    region = sanitize(str(context['attacker_region']))
    side = context['attacker_side']
    army = next((a for k, a in state.armies.items()
                 if sanitize(str(k)) == region), None)
    if formation is None:
        if context.get('corsairs_relief_garrison'):
            units, leaders, _, _ = partition_battle_army(state, army, region, side, context)
            army = gs.Army(region=army.region, units=list(units), leaders=list(leaders))
        formation = select_forced_attack(state, army, side)
    if not formation.available:
        raise RuntimeError('Cannot form attacking army: ' + formation.unavailable_reason)
    payload = (region, side.name, formation.rearguard_signature)
    context['attacker_rearguard'] = payload
    return payload


def partition_battle_army(state, army, region, side, context=None):
    """Return participating units/leaders and all protected/bystander figures.

    Reads the captured counts, never the CURRENT political position. The
    original region remains the rear's home after the attackers move in.
    Identical unit copies are fungible; one frozen count remains reserved
    through casualties and Elite downgrades. No lists or figures are mutated.
    """
    if army is None:
        return (), (), (), ()
    from board import sanitize
    pc = context if context is not None else getattr(state, 'pending_combat', None)
    payload = (pc or {}).get('attacker_rearguard')
    units_due, leaders_due = Counter(), Counter()
    reserve_due = Counter()
    if ((pc or {}).get('corsairs_relief_garrison')
            and side == (pc or {}).get('attacker_side')
            and sanitize(str(region)) == sanitize(str(pc.get('defender_region')))):
        from army_card_cargo import decode, key as cargo_key
        reserve_due = decode(pc['corsairs_relief_garrison'])
    if payload is not None:
        origin, owner, signature = payload
        if origin == sanitize(str(region)) and owner == side.name:
            version, units, leaders = signature
            if version != FORMATION_VERSION:
                raise ValueError('Unknown attack formation version: %r' % (version,))
            units_due.update({(n, k): count for n, k, count in units})
            leaders_due.update(dict(leaders))
    participants, leaders, rest_units, rest_leaders = [], [], [], []
    for unit in army.units:
        key = (unit.nation.name, unit.kind.name)
        if _nation_side(unit.nation) != side:
            rest_units.append(unit)
        elif reserve_due and reserve_due[cargo_key(unit, True)] > 0:
            reserve_due[cargo_key(unit, True)] -= 1
            rest_units.append(unit)
        elif units_due[key] > 0:
            units_due[key] -= 1
            rest_units.append(unit)
        else:
            participants.append(unit)
    for leader in army.leaders:
        if _leader_side(leader) != side:
            rest_leaders.append(leader)
        elif reserve_due and reserve_due[cargo_key(leader)] > 0:
            reserve_due[cargo_key(leader)] -= 1
            rest_leaders.append(leader)
        elif (isinstance(leader, gs.Leader) and not _is_minion(leader)
              and leaders_due[leader.nation.name] > 0):
            leaders_due[leader.nation.name] -= 1
            rest_leaders.append(leader)
        else:
            leaders.append(leader)
    if any(n > 0 for n in units_due.values()) or any(n > 0 for n in leaders_due.values()):
        raise RuntimeError('A protected rearguard figure is missing at %s' % region)
    if any(n > 0 for n in reserve_due.values()):
        raise RuntimeError('A protected Corsairs garrison figure is missing at %s' % region)
    return tuple(participants), tuple(leaders), tuple(rest_units), tuple(rest_leaders)


def battle_army_view(state, army, region, side, context=None):
    """Read-only roster view; figures are shared, containers are independent."""
    if army is None:
        return None
    units, leaders, _, _ = partition_battle_army(
        state, army, region, side, context)
    return gs.Army(region=army.region, units=list(units), leaders=list(leaders))
