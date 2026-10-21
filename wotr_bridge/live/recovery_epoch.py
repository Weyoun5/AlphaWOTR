"""A recovery retires only evidence covered by its accepted native export."""
import copy


def prepare(driver, evidence):
    from . import client_protocol
    from .repair_evidence import same_dump
    header = evidence.get('header')
    if not header or int(header.get('session', 0)) <= 0:
        raise ValueError('recovery needs a live session and feed-position header')
    driver._check_dump_session(header)
    reader = copy.deepcopy(driver.feed)
    reader.poll()  # Preview without consuming the real reader on a refused adoption.
    if reader._buf or not client_protocol.dump_at_feed_position(header, len(reader.lines)):
        raise ValueError('recovery board does not cover the complete current native feed')
    raw = list(getattr(driver, '_raw_request_lines', ()))
    count = header.get('raw_lines')
    if not isinstance(count, int) or count < 0 or count > len(raw):
        raise ValueError('recovery needs the raw history covered by the board export')
    book = getattr(driver, '_native_die_evidence', {})
    seen = book.get('seen', 0)
    if seen and (seen > len(raw) or raw[seen - 1] != book.get('tail')):
        raise ValueError('native die history changed before recovery')
    if not same_dump(driver, evidence):
        raise ValueError('native evidence changed while establishing recovery boundary')
    card_ids = {}
    for model in driver.hand_models.values():
        observations = list(getattr(model, '_draw_observations', ()))
        observations += [entry['observation'] for entry in getattr(model, '_draw_receipts', ())
                         if entry.get('observation')]
        for row in observations:
            if row.get('cid') is not None and 0 <= row.get('raw_index', -1) < count:
                card_ids[str(row['pid'])] = row['cid']
    return {'reader': reader, 'raw': raw, 'raw_count': count, 'card_ids': card_ids,
            'feed_count': len(reader.lines), 'session': header['session'],
            'dump': header.get('dump')}


def commit(driver, boundary):
    """No new I/O: an append after validation remains unread for the next tick."""
    from .native_die_identity import rebase
    old = int(getattr(driver, '_feed_seq', 0))
    rebase(driver, boundary['raw'], boundary['raw_count'], boundary['feed_count'])
    vars(driver.feed).update(vars(boundary['reader']))
    driver._feed_seq = boundary['feed_count']
    if (getattr(driver, '_sidecar_wait_seq', None) is not None
            and driver._sidecar_wait_seq <= boundary['feed_count']):
        driver._sidecar_wait_seq = None
    driver._evidence_epoch = int(getattr(driver, '_evidence_epoch', 0)) + 1
    driver._evidence_boundary = {key: boundary[key] for key in
                                 ('session', 'dump', 'feed_count', 'raw_count')}
    driver.view.card_pid_map.update({cid: pid for pid, cid in boundary['card_ids'].items()})
    for model in driver.hand_models.values():
        model._draw_receipts = [entry for entry in getattr(model, '_draw_receipts', ())
                               if entry.get('feed_seq', 0) > boundary['feed_count']]
        model._draw_observations = [entry for entry in getattr(model, '_draw_observations', ())
                                   if entry.get('raw_index', -1) >= boundary['raw_count']]
    if driver.rand is not None:
        driver.rand.retire_cards_before(boundary['raw_count'])
    return max(0, boundary['feed_count'] - old)
