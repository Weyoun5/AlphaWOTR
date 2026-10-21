"""Physical cleanup ownership for successfully committed human combat cards."""


def note_committed(driver, action):
    """The shared human commit boundary owns both chooser and chat plays.

    Store identities, not inferred slot contents. A later event play of the
    same physical card retires any older combat obligation before its effects.
    """
    kind = getattr(action, 'action_type', '')
    cid = (getattr(action, 'params', None) or {}).get('card_id')
    if not cid or kind not in ('combat_card_play', 'combat_card_reveal', 'play_event'):
        return
    from cards import get_card
    card = get_card(cid)
    if card is None or card.side is not driver.human_side:
        return
    owed = dict(getattr(driver, '_human_combat_card_kills', None) or {})
    if kind == 'play_event':
        owed.pop(cid, None)
    else:
        owed[cid] = driver.view.card_pid_map.get(cid)
    driver._human_combat_card_kills = owed


def human_balrog_pids(driver):
    """The accepted table-to-combat transfer owns its old table seat too."""
    cid = 'SHADOW_Character_Balrog_of_Moria'
    owed = getattr(driver, '_human_combat_card_kills', None) or {}
    if cid not in owed:
        return set()
    pid = owed[cid] or driver.view.card_pid_map.get(cid)
    return {str(pid)} if pid is not None else set()


def retire_table_claim(server, pid, cid):
    """A transferred Balrog cannot retain a second disposal in its old slot."""
    if cid != 'SHADOW_Character_Balrog_of_Moria':
        return
    book = getattr(server, '_table_book', None) or {}
    if str(pid) in book and book[str(pid)][1] == cid:
        del book[str(pid)]
    from .driver import table_card_key
    key = table_card_key('Balrog of Moria')
    seats = getattr(server.driver, 'table_seats', {})
    if key in seats and str(server.driver.table_seat_pid(key)) == str(pid):
        del seats[key]


def ready_human_cards(server, active):
    """Return exact committed PIDs in a current native card slot, once.

    The caller owns the round/effect/physical-work gates. A chat commitment
    may precede its physical reveal; neither a hidden hand nor a Board slot
    is a cleanup target. The same-session export must cover the routed feed.
    """
    driver = server.driver
    owed = dict(getattr(driver, '_human_combat_card_kills', None) or {})
    if not owed or not driver.view.dump_path:
        return []
    if driver._feed_seq != len(driver.feed.lines) or getattr(driver.feed, '_buf', ''):
        return []
    from . import client_board
    from .repair_evidence import dump_signature, same_dump
    try:
        evidence = dump_signature(driver)
        header = evidence['header']
        if header is None or driver._dump_position_hold(header):
            return []
        driver._check_dump_session(header)
        dump = client_board.read_board(driver.view.dump_path)
        if dump is None or not same_dump(driver, evidence):
            return []
    except (OSError, ValueError):
        return []
    import emitter
    shadow = driver.human_side.name == 'SHADOW'
    slots = {'current sa card', 'sa card play'} if shadow else {'current fp card', 'fp card play'}
    dead = {'sa dead', 'sa casualties'} if shadow else {'fp dead', 'fp casualties'}
    ready = []
    for cid, original_pid in list(owed.items()):
        if cid in active or server._cid_on_table(cid):
            continue
        pid = original_pid or driver.view.card_pid_map.get(cid)
        if pid is None:
            continue
        zone = str(dump.get(str(pid), '')).strip().casefold()
        if zone in dead or cid in driver.es.hands[driver.human_side].hand:
            # Manual disposal / recovery fulfilled or retired the old claim.
            retire_table_claim(server, pid, cid)
            del owed[cid]
        elif zone in slots and driver._cc_already_discarded(cid):
            ready.append((emitter.canon_side(driver.human_side), str(pid), cid))
            del owed[cid]
    driver._human_combat_card_kills = owed
    return ready
