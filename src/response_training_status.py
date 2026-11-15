"""Bounded status for between-batch owned response-policy training."""
from pathlib import Path
import json
import time


def read(path):
    path = Path(path)
    if not path.exists():
        return {}
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Response status exceeds size bound')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Response status is not an object')
    return value


def observe(loop):
    """Return None outside an active update; never infer process liveness."""
    loop = Path(loop).resolve()
    home = loop / 'response_policy'
    if not home.exists():
        return None
    try:
        marker = read(loop / '.batch_journal/committed.json')
        generation = marker.get('generation')
        if type(generation) is not int or generation < 0:
            return None
        folder = home / ('generation_%06d' % generation)
        status = read(folder / 'status.json')
        if not status or status.get('status') == 'complete':
            return None
        owner = status.get('active_owner')
        command = status.get('command', [])
        progress = {}
        if '--output' in command:
            target = Path(command[command.index('--output') + 1]).resolve()
            if not target.is_relative_to(folder):
                raise ValueError('Response progress path escapes update directory')
            progress = read(target / 'progress.json')
        stamp = status.get('updated_at')
        age = time.time() - stamp if type(stamp) in (int, float) else None
        error = status.get('error')
        detail = '%s response policy, root generation %d' % (str(owner or 'preparing').upper(), generation)
        if progress.get('active_epoch') is not None:
            detail += ' | epoch %s | %s rows' % (progress['active_epoch'], format(progress.get('rows_seen', 0), ','))
        if error:
            detail += ' | ' + str(error)
        stale = status.get('status') == 'training' and (age is None or age > 90)
        if stale:
            detail += ' | response-training heartbeat missing or stale'
        return {'stage': 'response_training', 'status': status.get('status'),
                'summary': 'TRAINING RESPONSE POLICIES — ' + str(status.get('status')),
                'detail': detail, 'updated_at': stamp, 'heartbeat_age_seconds': age,
                'owner': owner, 'root_generation': generation,
                'error': error, 'needs_attention': bool(error or stale),
                'rows_seen': progress.get('rows_seen'), 'epoch': progress.get('active_epoch')}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return {'stage': 'response_training', 'status': 'unavailable',
                'summary': 'RESPONSE TRAINING STATUS UNREADABLE',
                'detail': str(error), 'error': str(error), 'needs_attention': True}
