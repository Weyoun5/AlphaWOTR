"""Single-writer dashboard publication after the training journal commits.

The feed is a derived view, never an input to training. An interrupted replace
leaves the previous complete feed; retry publishes the same generation once.
"""
import json
import os
from pathlib import Path
from batch_recovery import digest, _json, _publish

VERSION = 'committed_served_games_v1'
CONTRACT = 'dashboard_feed_contract.json'


def configure(journal):
    path = journal.path(CONTRACT)
    expected = {'version': VERSION, 'publication': 'committed_only'}
    if path.exists():
        if _json(path) != expected:
            raise RuntimeError('Dashboard feed contract changed')
    else:
        feed = journal.path('all_games.jsonl')
        if feed.exists() and feed.stat().st_size:
            raise RuntimeError('Committed dashboard feed refuses legacy feed adoption')
        _publish(path, expected)


def _rows(path):
    with open(path, encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError('Dashboard source row is not an object')
    return rows


def publish(journal, attempt):
    """Validate the committed batch bytes, then atomically publish its rows."""
    configure(journal)
    marker = journal._marker()
    index = attempt['index']
    if marker['generation'] != index:
        raise RuntimeError('Dashboard publication requires the committed generation')
    record_path = journal.path('.batch_journal/txn_%06d/record.json' % index)
    if digest(record_path) != marker.get('record_sha256'):
        raise RuntimeError('Dashboard committed record hash mismatch')
    record = _json(record_path)
    batch_name = 'batch_%03d' % index
    batch = journal.path(batch_name)
    if (record.get('index') != index or record.get('controls') != journal.controls
            or journal._batch_inventory(batch) != record.get('inputs')):
        raise RuntimeError('Dashboard committed input hashes changed')
    phase = _json(batch / 'phase.json').get('phase_id')
    if not isinstance(phase, str) or not phase:
        raise RuntimeError('Dashboard requires a served phase identity')
    attempts = _rows(batch / 'attempts.jsonl')
    games = _rows(batch / 'games.jsonl')
    by_game = {}
    for row in attempts:
        key = (row.get('worker_id'), row.get('episode'))
        if key in by_game or row.get('phase_id') != phase:
            raise RuntimeError('Dashboard attempt identity is ambiguous')
        by_game[key] = row
    published, seen = [], set()
    for game in games:
        key = (game.get('worker_id'), game.get('episode'))
        source_attempt = by_game.get(key)
        if (key in seen or any(type(value) is not int or value < 0 for value in key)
                or source_attempt is None or source_attempt.get('status') != 'completed'):
            raise RuntimeError('Dashboard completed game has no unique attempt')
        seen.add(key)
        # The archive already validated coverage. Recheck exact shared fields
        # here so publication cannot silently attach different result metadata.
        if any(source_attempt.get(name) != value for name, value in game.items()):
            raise RuntimeError('Dashboard game differs from its served attempt')
        basename = Path(str(game.get('trace_path', ''))).name
        trace = journal.path(batch_name + '/' + basename)
        if not basename or not basename.endswith('.trace.gz') or not trace.is_file():
            raise RuntimeError('Dashboard committed trace is missing')
        row = dict(game)
        row.update(batch=batch_name, trace_path=str(trace), trace_sha256=digest(trace),
                   phase_id=phase, recovery_attempt_id=attempt['attempt'],
                   feed_contract=VERSION, committed_generation=index,
                   committed_record_sha256=marker['record_sha256'])
        published.append(row)
    if not published:
        raise RuntimeError('Dashboard committed batch has no completed games')
    feed = journal.path('all_games.jsonl')
    prior = _rows(feed) if feed.exists() else []
    retained, current = [], []
    prior_keys = set()
    for row in prior:
        generation = row.get('committed_generation')
        key = (generation, row.get('worker_id'), row.get('episode'))
        if (row.get('feed_contract') != VERSION or type(generation) is not int
                or generation < 0 or generation > index or key in prior_keys):
            raise RuntimeError('Dashboard prior feed identity is invalid')
        prior_keys.add(key)
        (current if generation == index else retained).append(row)
    if current and current != published:
        raise RuntimeError('Dashboard generation was published with different content')
    payload = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in retained + published)
    if feed.exists() and feed.read_text(encoding='utf-8') == payload:
        return
    temporary = journal.path('all_games.jsonl.tmp')
    temporary.write_text(payload, encoding='utf-8', newline='\n')
    os.replace(temporary, feed)
