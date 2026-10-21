"""Public group staging is a selection until exactly one unused die remains."""
from collections import Counter
import re
from .action_die_zones import temporary


def pending(driver):
    """A physical group remains human work until its last die is returned."""
    if getattr(driver, '_grouped_die_identity_wait', None) is not None:
        return True
    book = getattr(driver, '_grouped_die_selection', None)
    return bool(book and book['active'] and book['turn'] == driver.es.turn
                and (book['window'] is None or driver.seg.window is book['window']))


def _printed_faces(record, side):
    import rendered_replay as rr
    obj = str(record.get('obj', ''))
    tokens = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    printed = Counter()
    for token in tokens:
        match = re.fullmatch(r'(?:\[(\d+)\]\s*)?(Free|Shadow) Action Die \(([ACEHMPW])\)', token)
        if match is None or (match[2] == 'Free') != (side.name == 'FREE_PEOPLES'):
            return None
        printed[match[3]] += int(match[1] or 1)
    return printed


def identities(driver, record, side):
    from .native_die_identity import pids
    from .gesture_match import FACE_LETTER
    printed = _printed_faces(record, side)
    if not printed:
        return None
    physical = pids(record)
    if not physical or len(physical) != len(set(physical)) or len(physical) != sum(printed.values()):
        return None
    found = {}
    for pid in physical:
        binding = driver.view.die_id_map.get(str(pid))
        if binding is None or binding[0] is not side:
            return None
        dice = driver.es.dice.get(side, ())
        if not 0 <= binding[1] < len(dice):
            return None
        die = dice[binding[1]]
        face = FACE_LETTER.get(getattr(die.current, 'name', None))
        if face is None:
            return None
        found[str(pid)] = (face, bool(die.used))
    return found if Counter(face for face, _used in found.values()) == printed else None


def consume(driver, record):
    """Return routed items, or None to leave the ordinary record path intact."""
    if getattr(driver, 'spectator', False):
        return None
    from .segmenter import _norm_text
    book = getattr(driver, '_grouped_die_selection', None)
    if book and (book['turn'] != driver.es.turn or driver.phase != 'actions'
                 or (book['window'] is not None and driver.seg.window is not book['window'])):
        driver._grouped_die_selection = book = None
    if record.get('op') == 'undo' and book and book['records']:
        norm = _norm_text(str(record.get('inner', '')))
        last = book['records'][-1]
        window = book['window']
        if (norm == last.get('norm') and (window is None or window.records[-1] is last)):
            if window is not None:
                driver.seg.window = None
            book['window'] = None
            book['redo'].append(book['records'].pop())
            _rebuild_selection(driver, book)
            return []
    if record.get('op') == 'redo' and book and book['redo']:
        if _norm_text(str(record.get('inner', ''))) == book['redo'][-1].get('norm'):
            book['records'].append(book['redo'].pop())
            return _rebuild_selection(driver, book)
    if book and record.get('op') not in ('undo', 'redo', 'chat', 'system', 'blank'):
        book['redo'].clear()
    if (book and len(book['active']) > 1 and book['window'] is None
            and record.get('op') in ('move', 'kill', 'flip')
            and str(record.get('player', '')) != driver.nick
            and not re.search(r'\b(?:Free|Shadow) Action Die \(',
                              str(record.get('obj', '')) + ' '.join(record.get('pieces', ())))):
        # Selecting several dice is still an unfinished human action. Retain
        # its board gestures until the player narrows the selection or undoes
        # them; they must not enter the die-less adoption/refusal lane.
        record['_grouped_effect'] = True
        book['records'].append(record)
        return []
    if (record.get('op') == 'kill'
            and str(record.get('player', '')) != driver.nick
            and getattr(driver.seg.window, '_staged_die_pid', None) is not None):
        from .native_die_identity import pids, kill_faces
        if kill_faces(record) and not pids(record):
            # A narrowed group still needs its selected physical die. Native
            # Kill can render before the request containing that public ID.
            driver._grouped_die_identity_wait = record
            return []
    if record.get('op') != 'move' or str(record.get('player', '')) == driver.nick:
        return None
    side = driver.human_side
    pool = 'fp dice' if side.name == 'FREE_PEOPLES' else 'sa dice'
    source, target = str(record.get('from', '')).lower(), str(record.get('to', '')).lower()
    # Rendered feed can grow after the request's raw-history snapshot. Keep
    # this ordered selection record until its public native IDs arrive; a
    # group has no single-die window that could otherwise retain its work.
    selection_move = ((source == pool and temporary(target))
                      or (book and temporary(source)
                          and (temporary(target) or target == pool)))
    if driver.phase == 'actions' and selection_move:
        from .native_die_identity import pids
        printed = _printed_faces(record, side)
        if printed and (book or sum(printed.values()) > 1) and not pids(record):
            import ai
            if book or (driver.seg.window is None and ai.describe_pending(driver.es) is None):
                driver._grouped_die_identity_wait = record
                return []
    if source == pool and temporary(target) and driver.phase == 'actions':
        if book and (book['window'] is None
                     or (not _has_effects(book)
                         and len(book['window'].records) == len(book['records']))):
            members = identities(driver, record, side)
            if (members and not any(used for _face, used in members.values())
                    and not set(members).intersection(book['active'])):
                # Restaging a returned die, or adding another candidate, is
                # still selection. Keep every staged PID until it is narrowed.
                record['_grouped_add_pids'] = members
                book['records'].append(record)
                book['active'].update(members)
                driver.seg.window = book['window'] = None
                return _open_remaining(driver, book)
        if driver.seg.window is not None or record.get('die_stage'):
            driver._grouped_die_selection = None
            return None
        members = identities(driver, record, side)
        if members is None or len(members) < 2 or any(used for _face, used in members.values()):
            return None
        import ai
        if ai.describe_pending(driver.es) is not None:
            return None
        driver._grouped_die_selection = {
            'turn': driver.es.turn, 'commits': driver.n_commits_total,
            'original': members, 'active': dict(members), 'records': [record],
            'redo': [], 'window': None}
        return []
    if book is not None and temporary(source) and temporary(target):
        members = identities(driver, record, side)
        if (members is not None and set(members) <= set(book['active'])
                and (book['window'] is None
                     or (not _has_effects(book)
                         and len(book['window'].records) == len(book['records'])))):
            record['_grouped_return_pids'] = ()
            book['records'].append(record)
            if book['window'] is not None:
                book['window'].records.append(record)
            return []
    if not temporary(source) or target != pool or book is None:
        if book and not book['active']:
            driver._grouped_die_selection = None
        return None
    members = identities(driver, record, side)
    if (members is None or any(used for _face, used in members.values())
            or not set(members) <= set(book['active'])):
        return None
    if book['window'] is not None:
        # Once action effects have begun, ordinary window cancellation owns
        # their undo; this helper only narrows the initial dice selection.
        if _has_effects(book) or len(book['window'].records) != len(book['records']):
            return None
        driver.seg.window = None
    record['_grouped_return_pids'] = tuple(members)
    book['records'].append(record)
    for pid in members:
        book['active'].pop(pid)
    book['window'] = None
    return _open_remaining(driver, book)


def _has_effects(book):
    return any(record.get('_grouped_effect') for record in book['records'])


def retry_identity(driver):
    """Resume the same already joined paragraph, before routing its suffix."""
    record = getattr(driver, '_grouped_die_identity_wait', None)
    if record is None:
        return []
    from .native_die_identity import pids
    if not pids(record):
        return []
    driver._grouped_die_identity_wait = None
    result = consume(driver, record)
    if result is not None:
        return result
    # Invalid/changed selection evidence still follows the normal route.
    result = []
    for event, payload in driver.seg.push(record, render_duplicate=False):
        result.extend(driver._route(event, payload))
    return result


def _rebuild_selection(driver, book):
    if book['window'] is not None:
        driver.seg.window = None
    book['window'] = None
    book['active'] = dict(book['original']) if book['records'] else {}
    for previous in book['records'][1:]:
        book['active'].update(previous.get('_grouped_add_pids', {}))
        for pid in previous.get('_grouped_return_pids', ()):
            book['active'].pop(pid)
    return _open_remaining(driver, book)


def _open_remaining(driver, book):
    from .segmenter import Window
    if len(book['active']) != 1:
        return []
    pid, (face, _used) = next(iter(book['active'].items()))
    window = Window(driver.human_side.name, face, book['records'][0])
    window.records = list(book['records'])
    window._staged_die_pid = pid
    driver.seg.window = book['window'] = window
    return driver._route('open', window)
