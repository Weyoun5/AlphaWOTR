"""Native request/response identities for isolated bridge regression fixtures."""
from pathlib import Path
import json
from live import ack_identity


def request_header(driver, session=9911):
    driver._fixture_request = getattr(driver, '_fixture_request', 0) + 1
    return dict(session=session, req=driver._fixture_request)


def publish(driver, lines, session=9911):
    header = request_header(driver, session)
    ack_identity.begin(driver, header)
    ack_identity.bind(driver, lines)
    return header


def stamp_acks(driver, records):
    """Use the actual published operation index, never infer it from PID order."""
    pending = getattr(driver, '_ack_operations', {})
    result = []
    for record in records:
        keys = [key for key, raw in pending.items() if raw == record['raw'].strip()]
        if len(keys) != 1:
            raise AssertionError(('ACK fixture needs one published operation', record['raw'], keys))
        session, req, index = keys[0]
        result.append(dict(record, session=session, req=req, i=index))
    return result


def snapshot_header(driver, path=None):
    """Bind saved board-body fixture data to this test's actual empty/raw feed."""
    path = Path(path or driver.view.dump_path)
    feed = Path(driver.feed.path)
    if not feed.read_text(encoding='utf8'):
        feed.write_text('#BLIND ai_side=%s\n' % driver.ai_side.name, encoding='utf8')
    body = [row for row in path.read_text(encoding='utf8').splitlines() if not row.startswith('#SEQ ')]
    raw = list(getattr(driver, '_raw_request_lines', ()))
    driver._raw_request_lines = raw
    count = len(feed.read_text(encoding='utf8').splitlines()) - 1
    path.write_text('#SEQ dump=1 session=9911 raw_lines=%d feed_paragraphs=%d\n%s\n' %
                    (len(raw), count, '\n'.join(body)), encoding='utf8')
