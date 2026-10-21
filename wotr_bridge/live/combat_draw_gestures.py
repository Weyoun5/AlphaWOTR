"""Coverage for physically consumed draws during one pending transaction."""
from collections import Counter
from copy import copy

# Real combat endpoints may settle a bonus draw after losses or continuation.
# Membership only enables the exact consumed-card proof below; combat_continue
# merely opens defender_retreat and does not itself settle the Sorcerer draw.
COMBAT_CHOICES = frozenset(('combat_effect_choice', 'combat_card_play', 'combat_card_none',
                            'shine', 'shine_none', 'casualty_allocate',
                            'siege_press', 'siege_no_press', 'siege_continue',
                            'siege_assault', 'combat_no_retreat', 'combat_cease',
                            'combat_retreat', 'retreat_into_siege'))


def draw_cursor(rng):
    rand = getattr(rng, 'r', None)
    if rand is None or not getattr(rand, '_step_open', False):
        return None
    return dict(rand._feed_pos.get('cards', {}))


def consumed_draw_coverage(pre, post, chain, coverage, facts, rng, start):
    """Engine outcome AND this attempt's consumed physical card IDs must agree.

    A pending request, merely queued card, wrong deck or a prior attempt's draw
    never licenses a missing physical row. This only covers the already
    consumed client randomness; the driver retains its normal echo/census gates.
    """
    combat = (getattr(pre, 'pending_combat', None)
              and all(a.action_type in COMBAT_CHOICES for a in chain))
    pending_card = getattr(pre, 'pending_card_option', None) or {}
    card_choice = (pending_card.get('card_id') and chain
                   and all(a.action_type == 'card_option' for a in chain)
                   and (getattr(chain[0], 'params', None) or {}).get('card_id')
                   == pending_card['card_id'])
    # A staged play can commit its card/payment before the player's selected
    # recruitment arrives. Its effect then consumes the demanded draw receipts
    # inside card_option, so those receipts are no longer gesture rows. The
    # same exact transaction proof applies; unrelated choices and queued draws
    # still cannot fill missing coverage.
    if start is None or not chain or not (combat or card_choice):
        return coverage
    rand = getattr(rng, 'r', None)
    end = draw_cursor(rng)
    if rand is None or end is None:
        return coverage
    from .randomness import _CARD_ATTR
    from game_state import Side
    proven = {}
    for key, amount in coverage.draws.items():
        if not amount:
            continue
        before, after = start.get(key, 0), end.get(key, 0)
        if amount < 0 or after - before != amount or key not in _CARD_ATTR:
            return coverage
        rows = rand._feed['cards'].get(key, ())
        if not 0 <= before <= after <= len(rows):
            return coverage
        consumed = Counter(str(cid) for cid, pid in rows[before:after])
        attr = _CARD_ATTR[key]
        lost = Counter(map(str, getattr(pre, attr))) - Counter(map(str, getattr(post, attr)))
        if consumed != lost:
            return coverage
        side = Side[key[0]]
        acquired = Counter(map(str, post.hands[side].hand)) - Counter(map(str, pre.hands[side].hand))
        if consumed != acquired:
            return coverage
        if facts.draws.get(key, 0) < 0 or facts.draws.get(key, 0) > amount:
            return coverage
        proven[key] = facts.draws.get(key, 0)
    if not proven or any(end.get(k, 0) - start.get(k, 0) != coverage.draws.get(k, 0)
                         for k in set(start) | set(end)):
        return coverage
    result = copy(coverage)
    result.draws = Counter(coverage.draws)
    for key, count in proven.items():
        result.draws[key] = count
    return result
