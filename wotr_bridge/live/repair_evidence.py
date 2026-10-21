"""A parser stall is not evidence that the physical game state diverged."""
import os
import hashlib
import tempfile


def dump_signature(driver):
    path = driver.view.dump_path
    before = os.stat(path).st_mtime_ns
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if before != os.stat(path).st_mtime_ns:
        raise ValueError('physical board changed while reading its fingerprint')
    feed_stat = os.stat(driver.feed.path)
    return {'mtime_ns': before, 'sha256': digest, 'header': driver._dump_header(),
            'feed_stat': (feed_stat.st_size, feed_stat.st_mtime_ns)}


def same_dump(driver, evidence):
    try:
        return dump_signature(driver) == {key: evidence.get(key) for key in ('mtime_ns', 'sha256', 'header', 'feed_stat')}
    except (OSError, ValueError):
        return False


def adopt_measured(driver, evidence, *args, **kwargs):
    """Give every adopter read one immutable export; reject a moving live board."""
    from . import adopt
    def refused():
        return (['ADOPTION DECLINED: physical board or native feed changed after '
                 'measurement; engine retained, review the current board'], 0, 0)
    if not same_dump(driver, evidence):
        return refused()
    with open(driver.view.dump_path, 'rb') as stream:
        content = stream.read()
    if hashlib.sha256(content).hexdigest() != evidence['sha256'] or not same_dump(driver, evidence):
        return refused()
    snapshot = adopt._snapshot_armies(driver.es)
    pending = {name: getattr(driver.es, name) for name in driver._PENDING_FIELDS if hasattr(driver.es, name)}
    with tempfile.TemporaryDirectory(prefix='wotr_measured_board_') as folder:
        path = os.path.join(folder, 'board.txt')
        with open(path, 'wb') as stream:
            stream.write(content)
        kwargs['dump_path'] = path
        result = adopt.adopt_client_board(*args, **kwargs)
    if not same_dump(driver, evidence):
        adopt._restore_armies(driver.es, snapshot)
        for name, value in pending.items():
            setattr(driver.es, name, value)
        for name in driver._PENDING_FIELDS:
            if name not in pending and hasattr(driver.es, name):
                delattr(driver.es, name)
        return refused()
    return result


def measure(driver):
    """Read a complete current client dump; never infer or adopt board state."""
    path = getattr(driver.view, 'dump_path', None)
    if not path or not os.path.isfile(path):
        return {'status': 'unavailable', 'reason': 'no physical board dump', 'rows': []}
    if int(getattr(driver, '_feed_seq', 0)) < len(getattr(driver.feed, 'lines', ())):
        return {'status': 'in_flight', 'reason': 'prefetched client feed has not finished routing', 'rows': []}
    holds = driver.quiescence_holds()
    if holds:
        return {'status': 'in_flight', 'reason': '; '.join(holds[:4]), 'rows': []}
    try:
        signature = dump_signature(driver)
        before = signature['mtime_ns']
        header = signature['header']
        if header is None:
            return {'status': 'unavailable', 'reason': 'board dump has no feed-position header', 'rows': []}
        driver._check_dump_session(header)
        position = driver._dump_position_hold(header)
        if position:
            return {'status': 'unavailable', 'reason': position, 'rows': []}
        if not driver.view.refresh_dump():
            return {'status': 'unavailable', 'reason': 'board dump incomplete or changing', 'rows': []}
        rows, staging = driver.census_rows()
        if not same_dump(driver, signature):
            return {'status': 'unavailable', 'reason': 'board dump changed during comparison', 'rows': []}
        return {'status': 'mismatch' if rows else 'synced', 'rows': list(rows),
                'staging': list(staging), **signature,
                'reason': rows[0] if rows else 'physical board matches the engine'}
    except Exception as exc:
        return {'status': 'unavailable', 'reason': 'board comparison failed: %s' % exc, 'rows': []}


def diagnostic(driver, reason, evidence):
    """Report the failure while retaining every action, die and open decision."""
    sig = (str(reason), evidence['status'], evidence['reason'])
    if getattr(driver, '_repair_measurement_note', None) == sig:
        return []
    driver._repair_measurement_note = sig
    driver.findings.append(('repair_not_offered', {'trigger': str(reason), **evidence}))
    return [('note', 'Action retained: %s. No self-repair offered: %s.'
             % (reason, evidence['reason']))]
