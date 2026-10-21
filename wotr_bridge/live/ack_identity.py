"""Bind acknowledgements to the exact operation published in one response."""
from . import client_protocol


def begin(driver, header):
    driver._response_identity = None
    if header and header.get('session', 0) > 0 and header.get('req', -1) >= 0:
        driver._check_dump_session(header)
        driver._response_identity = (header['session'], header['req'])
    if not getattr(driver, '_ack_watermark_ready', False):
        path = getattr(driver, 'ack_path', None)
        driver._acks_seen = len(client_protocol.read_acks(path)) if path else 0
        driver._ack_watermark_ready = True


def bind(driver, lines):
    identity = getattr(driver, '_response_identity', None)
    if identity is None:
        return
    pending = driver.__dict__.setdefault('_ack_operations', {})
    for index, line in enumerate(lines or ()):
        facts = driver._op_line_facts(line)
        if facts is None:
            continue
        key = identity + (index,)
        raw = str(line).strip()
        pending[key] = raw
        for pid in facts[1]:
            entry = driver.echo_expect.get(pid)
            if entry is not None and str(entry.get('line', '')).strip() == raw:
                entry['operation'] = key


def matched(driver, record):
    key = (record.get('session'), record.get('req'), record.get('i'))
    pending = getattr(driver, '_ack_operations', {})
    raw = str(record.get('raw', '')).strip()
    if key not in pending or pending[key] != raw:
        return None
    del pending[key]
    return [pid for pid, entry in driver.echo_expect.items()
            if entry.get('operation') == key]
