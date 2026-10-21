"""Physical follow-through for a Guide already assigned by the engine."""
import re


_FELLOWSHIP = frozenset(('fellowshipofthering', 'fellowshipbox'))
_GUIDE = frozenset(('guideofthefellowship', 'guidebox', 'guideofthefellowshipbox'))


def consume(driver, record):
    """Acknowledge exact box bookkeeping, without choosing or adopting a Guide.

    Declaration and pending Guide choices retain their normal decision path.
    Native history can retract/reinstate only a witnessed physical receipt in
    this same engine commit; it never replays the inner text as a new action.
    """
    from game_state import Side
    from . import live_parser

    es = driver.es
    guide = getattr(es.fellowship, 'guide', None)
    name = getattr(guide, 'name', None)
    if (getattr(driver, 'spectator', False)
            or driver.human_side is not Side.FREE_PEOPLES
            or driver.phase == 'declare'
            or getattr(es, '_declare_phase', False)
            or es.pending_guide_change is not None
            or not name
            or not any(c.name == name for c in es.fellowship.companions)):
        return False
    # This epoch belongs to the reversible commit snapshot. The driver's
    # audit counter stays monotonic after Undo and cannot identify the
    # restored physical history preceding an undone action.
    context = (es.turn, name, getattr(driver, '_guide_seat_commit_epoch', 0) or 0)
    receipts = list(getattr(driver, '_guide_seat_receipts', None) or ())
    redo = list(getattr(driver, '_guide_seat_redo', None) or ())
    op = record.get('op')
    if op in ('undo', 'redo'):
        _, norm = live_parser.normalize(str(record.get('inner', '')).strip())
        source = receipts if op == 'undo' else redo
        if not source or source[-1][:2] != (context, norm):
            driver._guide_seat_redo = []
            return False
        receipt = source.pop()
        (redo if op == 'undo' else receipts).append(receipt)
        driver._guide_seat_receipts = receipts[-32:]
        driver._guide_seat_redo = redo[-32:]
        driver.findings.append(('guide_seat_' + ('undone' if op == 'undo' else 'redone'), norm))
        return True
    if op in ('move', 'kill', 'flip', 'draw', 'alloc_eye', 'roll_dice', 'roll_num', 'choose'):
        driver._guide_seat_redo = []
    if op != 'move' or str(record.get('player', '')) == driver.nick:
        return False
    source, target = [re.sub(r'[^a-z0-9]', '', str(record.get(key, '')).lower())
                      for key in ('from', 'to')]
    forward = source in _FELLOWSHIP and target in _GUIDE
    inverse = source in _GUIDE and target in _FELLOWSHIP
    if not forward and not inverse:
        return False
    if inverse and (not receipts or receipts[-1][0] != context or not receipts[-1][2]):
        return False
    import rendered_replay as rr
    obj = str(record.get('obj', ''))
    tokens = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    if len(tokens) != 1:
        return False
    piece = rr.parse_piece(tokens[0])
    if piece is None or piece[0] != 'named' or piece[1] != name or piece[3] != 1:
        return False
    from .sidecar import meta_pids
    pids = meta_pids(record.get('meta')) or ()
    if pids:
        from .client_board import unit_type
        if len(pids) != 1 or unit_type(str(pids[0]), driver.view.dynamic_types) != (name, 'companion'):
            return False
    receipts.append((context, record.get('norm', ''), forward))
    driver._guide_seat_receipts = receipts[-32:]
    driver.findings.append(('guide_seat_performed', record.get('norm', '')))
    return True
