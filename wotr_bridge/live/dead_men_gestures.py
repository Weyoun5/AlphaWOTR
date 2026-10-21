"""The human arrival does not perform Dead Men's forced Shadow aftermath."""
from collections import Counter
import copy


def forced_shadow_coverage(pre, post, side, chain, coverage, observed):
    """Leave only unvoiced AI-owned consequences for the normal commit sweep.

    This is one already-legal companion choice, never a casualty or retreat
    choice. Its engine apply either performs a unique consequence or stops at
    the Shadow's own menu. Human arrivals and every voiced row remain exact.
    """
    import game_state as gs
    cid = 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow'
    if (side is not gs.Side.FREE_PEOPLES or len(chain) != 1
            or (getattr(pre, 'pending_card_option', None) or {}).get('card_id') != cid):
        return coverage
    action = chain[0]
    params = getattr(action, 'params', None) or {}
    choices = params.get('choices') or {}
    if (getattr(action, 'action_type', None) != 'card_option'
            or params.get('card_id') != cid or not choices.get('target_region')
            or not choices.get('companions')):
        return coverage
    from board import sanitize
    from combat import side_of_nation
    from .gesture_match import _named_side_tag, _shadow_tkey

    def generic_shadow_leaders(state):
        result = Counter()
        for region, army in state.armies.items():
            for leader in army.leaders:
                nation = getattr(leader, 'nation', None)
                if (not getattr(leader, 'name', None) and nation is not None
                        and side_of_nation(nation) is gs.Side.SHADOW):
                    result[sanitize(str(region)).strip().lower()] += 1
        return result

    out = copy.copy(coverage)
    out.armies = Counter({key: count for key, count in coverage.armies.items()
                          if observed.armies.get(key) or not _shadow_tkey(key[1])})
    out.named = Counter({key: count for key, count in coverage.named.items()
                         if observed.named.get(key) or _named_side_tag(key[0]) != 'shadow'})
    before, after = generic_shadow_leaders(pre), generic_shadow_leaders(post)
    out.leaders = coverage.leaders.copy()
    for region in before.keys() | after.keys():
        if not observed.leaders.get(region):
            out.leaders[region] -= after[region] - before[region]
    return out
