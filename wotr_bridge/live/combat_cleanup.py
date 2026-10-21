"""Physical completion of compulsory figure deaths after an army is destroyed."""
from collections import Counter


def wiped_regions(pre, post, side):
    import ai
    from board import sanitize
    pc = getattr(pre, 'pending_combat', None)
    if not isinstance(pc, dict):
        return set()
    result = set()
    for role in ('attacker', 'defender'):
        if pc.get(role + '_side') is not side:
            continue
        region = pc.get(role + '_region')
        if (region and ai._alloc_side_units(pre, region, side, pc=pc)
                and not ai._alloc_side_units(post, region, side, pc=pc)):
            result.add(sanitize(str(region)))
    return result


def unit_losses_visible(driver, region, expected_state=None):
    """Require the human's physical units to match the already-committed state.

    Comparing surviving units also handles compulsory rearguards and sieges;
    it does not require the region to be completely empty.
    """
    import ai
    import emitter
    from board import sanitize
    region = sanitize(str(region))
    seen = Counter()
    for pid in emitter._pieces_in(driver.view, region):
        nation, kind = emitter._unit_type(pid, driver.view)
        if kind in ('regular', 'elite') and ai._side_of_nation(nation) is driver.human_side:
            seen[nation, kind] += 1
    army = ai._armies_lookup(driver.es if expected_state is None else expected_state, region)
    expected = Counter((u.nation, u.kind.name.lower()) for u in (army.units if army else ())
                       if ai._side_of_nation(u.nation) is driver.human_side)
    return seen == expected
