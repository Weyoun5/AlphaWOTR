"""Three canonical commitment slots for simultaneous Shadow army cards."""
import ai,numpy as np
from game_state import UnitKind
from event_effects import SHADOW_NATIONS
from board import sanitize
TSL_CARD_ID='SHADOW_Strategy_The_Shadow_Lengthens'

SLOT_WIDTH=108
MAX_COMMITTED=3
TAIL_WIDTH=324
TSIM_CARD_ID='SHADOW_Strategy_The_Shadow_is_Moving'

def exact_commitments(state):
    """Canonical source/target/cargo for completed original-army selections."""
    from army_card_cargo import freeze, inventory, selected, key
    from collections import Counter
    pending = getattr(state, 'pending_card_option', None) or {}
    card_id = pending.get('card_id')
    if card_id not in (TSL_CARD_ID, TSIM_CARD_ID):
        return ()
    options = pending.get('options') or []
    history_key = '_tsl_moves' if card_id == TSL_CARD_ID else '_tsim_moves'
    histories = [option.get(history_key, ()) for option in options]
    if not histories:
        return ()
    # Lists and tuples are both JSON transport forms of the same public plan.
    histories = [tuple(tuple(move) for move in history) for history in histories]
    if any(history != histories[0] for history in histories):
        raise ValueError('Conflicting army-card commitment histories')
    history = histories[0]
    limit = 1 if card_id == TSL_CARD_ID else MAX_COMMITTED
    if not isinstance(history, (list, tuple)) or len(history) > limit:
        raise ValueError('Invalid committed army count')
    if history and not pending.get('direct'):
        raise ValueError('Committed army requires direct continuation')
    moves = []
    for move in history:
        permitted = (2, 4, 5) if card_id == TSL_CARD_ID else (2, 3)
        if len(move) not in permitted:
            raise ValueError('Malformed staged army commitment')
        source, target = move[:2]
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError('Commitment regions must be strings')
        source, target = sanitize(source), sanitize(target)
        if source not in ai.REGION_INDEX or target not in ai.REGION_INDEX or source == target:
            raise ValueError('Invalid commitment regions')
        explicit = move[4] if len(move) == 5 else move[2] if card_id == TSIM_CARD_ID and len(move) == 3 else None
        if explicit is not None:
            units, leaders = selected(state, source, explicit)
            counts = Counter([key(u, True) for u in units] + [key(l) for l in leaders])
        else:
            counts = inventory(state, source)
            if card_id == TSL_CARD_ID and len(move) >= 4:
                # Historical aggregate choices remain readable without claiming
                # nation selection beyond the old deterministic figure order.
                army = ai._armies_lookup(state, source)
                limits = dict(zip((UnitKind.REGULAR, UnitKind.ELITE), move[2:4]))
                counts = Counter({k:n for k,n in counts.items() if k[0] != 'unit'})
                for unit in army.units:
                    if unit.nation not in SHADOW_NATIONS:
                        continue
                    left = limits[unit.kind]
                    if left is None or left > 0:
                        counts[key(unit,True)] += 1
                        if left is not None:limits[unit.kind] -= 1
                if any(v is not None and (type(v) is not int or v < 0) for v in move[2:4]):
                    raise ValueError('Invalid historical cargo counts')
                if any(v is not None and v > 0 for v in limits.values()):
                    raise ValueError('Historical cargo exceeds available figures')
        if not any(k[0] == 'unit' for k in counts):
            raise ValueError('Committed army must have combat units')
        moves.append((source, target, freeze(counts)))
    if len({move[0] for move in moves}) != len(moves):
        raise ValueError('Duplicate original source commitment')
    return tuple(sorted(moves))


def commitments(state):
    result = []
    for source, target, cargo in exact_commitments(state):
        regular = sum(row[3] for row in cargo if row[0] == 'unit' and row[2] == 'REGULAR')
        elite = sum(row[3] for row in cargo if row[0] == 'unit' and row[2] == 'ELITE')
        result.append((source,target,regular,elite))
    return tuple(result)


def committed_move(state):
    if (getattr(state,'pending_card_option',None) or {}).get('card_id') != TSL_CARD_ID:
        return None
    moves = commitments(state)
    return moves[0] if moves else None


def commitment_tail(state):
    tail=np.zeros(TAIL_WIDTH,dtype=np.float32)
    for index,(source,target,regular,elite) in enumerate(commitments(state)):
        offset=index*SLOT_WIDTH
        tail[offset]=1
        tail[offset+1+ai.REGION_INDEX[source]]=1
        tail[offset+1+ai.REGION_INDEX[target]]=-1
        tail[offset+106:offset+108]=(regular/10,elite/10)
    return tail
