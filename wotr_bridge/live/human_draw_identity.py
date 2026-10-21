"""Bind a decoded human card only to the slot created by its physical draw.

Rendered draws and private commitment observations arrive on separate channels.
Their order within each deck is shared; a staged draw need not have entered the
engine hand yet. Keep that association through matching, refusal and native Undo.
"""


def receipt(model, deck, record=None):
    entries = getattr(model, '_draw_receipts', None)
    if entries is None:
        entries = model._draw_receipts = []
    token = record.get('_human_draw_token') if record is not None else None
    if token is not None:
        return next((entry for entry in entries if entry['token'] == token), None)
    model._draw_serial = getattr(model, '_draw_serial', 0) + 1
    token = model._draw_serial
    if record is not None:
        record['_human_draw_token'] = token
    entry = {'token': token, 'deck': deck, 'placeholder': None,
             'norm': record.get('norm') if record is not None else None,
             'feed_seq': record.get('_feed_seq', 0) if record is not None else 0,
             'cancelled': False, 'undone': False, 'resolved': False,
             'observation': None}
    entries.append(entry)
    _associate(model)
    return entry


def slot(model, deck, placeholder, record=None):
    entry = receipt(model, deck, record)
    if entry is not None:
        entry['placeholder'] = placeholder
        _resolve(model, entry)


def _associate(model):
    waiting = []
    for decoded in getattr(model, '_draw_observations', ()):
        entry = next((row for row in getattr(model, '_draw_receipts', ())
                      if row['observation'] is None and row['deck'] == decoded['deck']), None)
        if entry is None:
            waiting.append(decoded)
        else:
            entry['observation'] = decoded
            entry['raw_index'] = decoded.get('raw_index')
    model._draw_observations = waiting


def _resolve(model, entry):
    decoded = entry['observation']
    if (decoded is None or entry['undone'] or entry['cancelled']
            or entry['resolved'] or decoded['cid'] is None
            or entry['placeholder'] is None):
        return
    if model.reveal(decoded['cid'], deck=entry['deck'], placeholder=entry['placeholder']) is not None:
        entry['resolved'] = True


def observe_record(driver, record):
    if getattr(driver, 'spectator', False) or not getattr(driver, 'live', True):
        return
    if record.get('op') == 'draw':
        what = str(record.get('what', '')).lower()
        owner = 'free' if driver.human_side.name == 'FREE_PEOPLES' else 'shadow'
        if what.startswith(owner) and what.endswith('card'):
            receipt(driver.hands, 'character' if 'character' in what else 'strategy', record)
    elif record.get('op') == 'undo':
        from . import live_parser
        inner = str(record.get('inner', '')).strip()
        _kind, norm = live_parser.normalize(inner)
        undone = live_parser.parse(inner)
        owner = 'free' if driver.human_side.name == 'FREE_PEOPLES' else 'shadow'
        if (undone.get('op') == 'draw' and str(undone.get('what', '')).lower().startswith(owner)
                and str(undone.get('what', '')).lower().endswith('card')):
            driver.hands._native_draw_undo_seen = True
        for entry in reversed(getattr(driver.hands, '_draw_receipts', ())):
            if entry['norm'] == norm and not entry['undone']:
                entry['cancelled'] = True
                entry['undone'] = True
                break


def cancel_records(driver, records):
    tokens = {record.get('_human_draw_token') for record in records}
    for entry in getattr(driver.hands, '_draw_receipts', ()):
        if entry['token'] in tokens:
            entry['cancelled'] = True


def publish(driver):
    for entry in getattr(driver.hands, '_draw_receipts', ()):
        # A staged draw's placeholder enters the hand when its action commits.
        # Resolve that slot before a later record in the current feed batch.
        _resolve(driver.hands, entry)
        if entry['resolved'] and not entry['cancelled'] and not entry['undone']:
            decoded = entry['observation']
            driver.view.card_pid_map[decoded['cid']] = decoded['pid']
            if not entry.get('published'):
                driver.view.note_drawn(decoded['pid'])
                entry['published'] = True


def settle(server):
    model = server.driver.hands
    observations = model.__dict__.setdefault('_draw_observations', [])
    observations.extend(getattr(server, '_human_draw_ids', ()))
    server._human_draw_ids = []
    _associate(model)
    entries = getattr(model, '_draw_receipts', [])
    remaining = []
    for entry in entries:
        decoded = entry['observation']
        if decoded is None:
            remaining.append(entry)  # The raw identity channel may lag the feed.
            continue
        if entry['undone']:
            continue
        # Keep an ordered receipt until native Undo has passed it, including
        # a refusal already scheduled for Undo. Otherwise identical generic
        # draw text could withdraw an older, still undecoded legal grant.
        if entry['cancelled'] or decoded['cid'] is None:
            remaining.append(entry)
            continue
        _resolve(model, entry)
        if not entry['resolved']:
            remaining.append(entry)
            continue
        remaining.append(entry)
    model._draw_receipts = remaining
    publish(server.driver)
    from .hand_redraw import reconcile
    reconcile(server.driver)
