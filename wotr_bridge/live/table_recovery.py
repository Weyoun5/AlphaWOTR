"""Recover persistent membership from authenticated card IDs and native zones."""
from collections import Counter
import re


def _is_table(zone):
    return re.fullmatch(r'board[1-9]\d*', str(zone).strip().lower()) is not None


def _is_card_pid(pid):
    return str(pid).isdigit() and 106 <= int(pid) <= 209


def _cards(state):
    table = getattr(state, 'persistent_cards', None)
    return list(getattr(table, 'fp_cards', ())) + list(getattr(table, 'sh_cards', ()))


def _already_played(state, cid, card):
    """A paid one-shot card may stay visible to mark its ongoing effect.

    Grond and Fighting Uruk-hai commonly move along Board slots to count
    siege rounds. Their effect already lives in the retained combat state;
    their printed card is not an additional persistent rule. Accept this
    only from the committed inventory, never by interpreting its position
    as proof that an unplayed event happened.
    """
    key = ('fp' if card.side.name == 'FREE_PEOPLES' else 'sh') + '_' + card.deck.lower()
    if list(state.discard_face_up.get(key, ())).count(cid) != 1:
        return False
    if sum(list(pile).count(cid) for pile in state.discard_face_up.values()) != 1:
        return False
    if any(cid in hand.hand for hand in state.hands.values()):
        return False
    return not any(cid in getattr(state, name, ()) for name in (
        'fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'))


def _inspect(state, dump, pid_to_cid):
    """Return observed known membership, removals and unresolved evidence."""
    from cards import get_card
    from persistent_effects import create_persistent_card
    known = {str(pid): str(cid) for pid, cid in (pid_to_cid or {}).items()}
    inverse = {}
    errors = []
    for pid, cid in known.items():
        inverse.setdefault(cid, []).append(pid)
    active = {card.card_id: card for card in _cards(state)}
    observed = {}
    for pid, zone in dump.items():
        pid = str(pid)
        if not _is_card_pid(pid) or not _is_table(zone):
            continue
        cid = known.get(pid)
        if cid is None:
            errors.append('table card PID %s: public identity unavailable' % pid)
            continue
        if len(inverse[cid]) != 1:
            errors.append('table card %s: conflicting physical identities' % cid)
            continue
        card = get_card(cid)
        try:
            if card is None:
                raise ValueError('not a supported persistent card')
            if 'play on the table' not in (card.event_req + ' ' + card.event_effect).lower():
                if cid not in active and _already_played(state, cid, card):
                    continue
                raise ValueError('not a supported persistent card')
            template = create_persistent_card(cid, card.card_name, card.side)
        except ValueError:
            errors.append('table card %s: persistent effect or required context unavailable' % cid)
            continue
        observed[cid] = (pid, template)
    removed = {}
    for cid, card in active.items():
        if cid in observed:
            continue
        pids = inverse.get(cid, [])
        if len(pids) != 1 or pids[0] not in dump:
            errors.append('table card %s: complete physical identity/location unavailable' % cid)
            continue
        pid = pids[0]
        zone = str(dump[pid]).strip().lower()
        prefix = 'fp' if card.side.name == 'FREE_PEOPLES' else 'sa'
        destinations = {prefix + ' hand': 'hand',
                        prefix + ' dead': 'discard',
                        prefix + ' casualties': 'discard'}
        deck = 'character' if '_Character_' in cid else 'strategy'
        destinations[prefix + (' char deck' if deck == 'character' else ' strat deck')] = 'deck'
        if zone not in destinations:
            errors.append('table card %s: unresolved native location %s' % (cid, dump[pid]))
            continue
        removed[cid] = (card, destinations[zone])
    return observed, removed, errors


def rows(state, dump, pid_to_cid=None):
    """Optional identity-aware census; never infer card names from native PIDs."""
    if pid_to_cid is None:
        return []
    observed, removed, errors = _inspect(state, dump, pid_to_cid)
    counts = Counter(card.card_id for card in _cards(state))
    result = list(errors)
    for cid, (pid, template) in observed.items():
        if counts[cid] != 1:
            result.append('table card %s: engine=%d client=1 pids=%s'
                          % (cid, counts[cid], pid))
    for cid in removed:
        result.append('table card %s: engine=%d client=0' % (cid, counts[cid]))
    return result


def adopt(state, dump, pid_to_cid):
    """Adopt only membership and inventory, never replay an event's effects."""
    from game_state import Side
    observed, removed, errors = _inspect(state, dump, pid_to_cid)
    if errors:
        raise ValueError('; '.join(errors))
    table = getattr(state, 'persistent_cards', None)
    if table is None:
        raise ValueError('persistent card state unavailable')
    old = _cards(state)
    existing = {card.card_id: card for card in old}
    tracked = set(observed) | set(removed)
    for hand in state.hands.values():
        hand.hand[:] = [cid for cid in hand.hand if cid not in tracked]
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        cards = getattr(state, name)
        cards[:] = [cid for cid in cards if cid not in tracked]
    for cards in state.discard_face_up.values():
        cards[:] = [cid for cid in cards if cid not in tracked]
    table.fp_cards[:] = []
    table.sh_cards[:] = []
    report = []
    for cid, (_, template) in observed.items():
        # Retain any already-known contextual payload. The recognized
        # constructor is used only for a new context-free persistent entry.
        table.add_card(existing.get(cid, template))
        if cid not in existing:
            report.append('table card adopted: %s' % cid)
    for cid, (card, destination) in removed.items():
        prefix = 'fp' if card.side is Side.FREE_PEOPLES else 'sh'
        deck = 'character' if '_Character_' in cid else 'strategy'
        if destination == 'deck':
            name = prefix + ('_char_deck' if deck == 'character' else '_strat_deck')
            getattr(state, name).append(cid)
        elif destination == 'discard':
            state.discard_face_up.setdefault(prefix + '_' + deck, []).append(cid)
        # adopt_hands follows, using the same physical PID identity map.
        report.append('table card removed: %s -> %s' % (cid, destination))
    return report


def cleanup_seats(state, dump, pid_to_cid):
    """Prepare the physical cleanup ledger for a completely adopted table."""
    from .driver import table_card_key, _TABLE_SLOT_CODE, _zone_key
    observed, removed, errors = _inspect(state, dump, pid_to_cid)
    active = Counter(card.card_id for card in _cards(state))
    if errors or removed or active != Counter({cid: 1 for cid in observed}):
        raise ValueError('persistent cleanup needs complete adopted table membership')
    seats = {}
    for cid, (pid, card) in observed.items():
        slot = _TABLE_SLOT_CODE.get(_zone_key(dump[pid]))
        if slot is None:
            raise ValueError('persistent cleanup needs a recognized native Board slot')
        seats[table_card_key(card.card_name)] = {
            'slot': slot, 'pid': pid, 'armed': True,
        }
    return seats


def rebase_cleanup(driver, seats):
    """Publish only after acceptance; snapshot-covered placements need no replay."""
    driver.table_seats.clear()
    driver.table_seats.update(seats)
    driver._table_cleanup_epoch = int(getattr(driver, '_table_cleanup_epoch', 0)) + 1


def sync_cleanup(server):
    """Retire the server's pre-recovery claims before it seats or cleans cards."""
    epoch = int(getattr(server.driver, '_table_cleanup_epoch', 0))
    if epoch != int(getattr(server, '_table_cleanup_epoch', 0)):
        # Both factions' adopted cards now belong to driver.table_seats.
        # A prior server claim may name a card returned to its hand or deck.
        server._table_book = {}
        server._table_cleanup_epoch = epoch
