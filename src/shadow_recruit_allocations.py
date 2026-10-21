"""Regional allocation of scarce Regulars for multi-region recruitment cards.

Almanac p35 lets the owner distribute a short reinforcement supply up to each
printed region limit. Complete-supply card choices keep their existing shape.
"""
from itertools import product

from game_state import Nation, Side, UnitKind


def allocation_spec(state, name, choices=None):
    from ai import figures_available
    from card_options import _free_for_side
    from event_effects import _free_sauron_strongholds, _free_se_settlements
    choices = choices or {}
    key, nation, maximum_regions = "regions", Nation.SAURON, 3
    if name == "Orcs Multiplying Again":
        limits = {"dol guldur": 3, "mount gundabad": 3}
    elif name == "Monsters Roused":
        limits = {"angmar": 1, "ettenmoors": 1, "weather hills": 1}
    elif name == "A New Power is Rising":
        nation = Nation.ISENGARD
        elites = min(2, figures_available(state, nation, "elite"))
        if "unit_combo" in choices:
            if choices["unit_combo"] not in ("2regular", "mixed", "2elite"):
                raise ValueError("Unknown Orthanc recruitment combination")
            elites = {"2regular": 0, "mixed": 1, "2elite": 2}[choices["unit_combo"]]
            if elites > figures_available(state, nation, "elite"):
                raise ValueError("Orthanc combination exceeds the Elite supply")
        limits = {"north dunland": 2, "orthanc": 2 - elites, "south dunland": 2}
    elif name == "Many Kings to the Service of Mordor":
        key, nation = "settlements", Nation.SOUTHRON_EASTERLINGS
        limits = dict.fromkeys(_free_se_settlements(state), 2)
    elif name == "Pits of Mordor":
        key = "strongholds"
        limits = dict.fromkeys(_free_sauron_strongholds(state), 2)
    else:
        return None
    limits = {r: n for r, n in sorted(limits.items())
              if n and _free_for_side(state, r, Side.SHADOW)}
    capacity = sum(sorted(limits.values(), reverse=True)[:maximum_regions])
    available = figures_available(state, nation, "regular")
    return key, nation, limits, min(available, capacity), capacity, maximum_regions


def allocation_options(state, name, choices=None):
    """Return canonical shortage options, or None for the established path."""
    spec = allocation_spec(state, name, choices)
    if spec is None:
        return None
    key, _, limits, total, capacity, maximum_regions = spec
    if total == 0 or total >= capacity:
        return None
    regions = list(limits)
    options = []
    for counts in product(*(range(limits[r] + 1) for r in regions)):
        if sum(counts) != total or sum(n > 0 for n in counts) > maximum_regions:
            continue
        options.append({
            key: [r for r, n in zip(regions, counts) if n],
            "recruit_counts": [n for n in counts if n],
        })
    return options


def apply_allocation(state, name, choices):
    """Apply one checked allocation; the card wrapper owns overstack timing."""
    from event_effects import recruit_unit
    expected = allocation_options(state, name, choices)
    spec = allocation_spec(state, name, choices)
    if expected is None or spec is None:
        raise ValueError("Recruitment allocation is not a current shortage choice")
    key, nation = spec[:2]
    actual = {key: choices.get(key), "recruit_counts": choices.get("recruit_counts")}
    counts = actual["recruit_counts"]
    if not isinstance(counts, list) or any(type(n) is not int for n in counts):
        raise ValueError("Recruitment allocation counts must be integers")
    if actual not in expected:
        raise ValueError("Recruitment allocation must place the maximum legal supply")
    if name == "A New Power is Rising" and "unit_combo" in choices:
        from ai import figures_available
        from card_options import _free_for_side
        default = allocation_spec(state, name)
        if _free_for_side(state, "orthanc", Side.SHADOW):
            chosen_elites = {"2regular": 0, "mixed": 1, "2elite": 2}[choices["unit_combo"]]
            default_elites = min(2, figures_available(state, nation, "elite"))
            if spec[3] + chosen_elites < default[3] + default_elites:
                raise ValueError("Orthanc override would leave recruitable units unused")
    result = {"effects": []}
    for region, count in zip(actual[key], actual["recruit_counts"]):
        for _ in range(count):
            if recruit_unit(state, region, nation, UnitKind.REGULAR) is None:
                raise RuntimeError("Validated recruitment allocation could not place its unit")
        result["effects"].append(f"Recruited {count} {nation.value} Regular(s) in {region}")
    if name == "Monsters Roused":
        if recruit_unit(state, "trollshaws", Nation.SAURON, UnitKind.ELITE):
            result["effects"].append("Recruited Sauron Elite in Trollshaws")
    elif name == "A New Power is Rising":
        from ai import figures_available
        elite_count = min(2, figures_available(state, nation, "elite"))
        if choices.get("unit_combo") in ("2regular", "mixed", "2elite"):
            elite_count = {"2regular": 0, "mixed": 1, "2elite": 2}[choices["unit_combo"]]
        for _ in range(elite_count):
            if recruit_unit(state, "orthanc", nation, UnitKind.ELITE):
                result["effects"].append("Recruited Isengard Elite in Orthanc")
    return result
