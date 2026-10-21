"""Optional, lossless, read-only second-level cache for combat_dist.

The existing AI memo remains the first level. Workers never create/update
tables. Incompatible/corrupt tables are disabled, with an exact-kernel fallback.
Configure WOTR_COMBAT_TABLE with an absolute path before starting workers.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import warnings
import zlib

SCHEMA = 'wotr-exchange-distribution-v1'
DATA_FILES = ('card_list.csv', 'wotr_map.txt', 'board200.jpg')


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return {'enum': type(value).__name__, 'value': value.value}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def tupled(value):
    return tuple(tupled(v) for v in value) if isinstance(value, list) else value


def context_bytes(ctx):
    # No saturation, roster sorting, card normalization, or field omission.
    return raw(plain(ctx))


def context_from_plain(value):
    import combat_dist as cd
    from game_state import Nation
    def army(a):
        a = dict(a)
        for key in ('nation', 'elite_nation', 'elite2_nation'):
            if a[key] is not None:
                a[key] = Nation(a[key]['value'])
        leaders = []
        for leader in a['leaders']:
            leader = dict(leader)
            if leader['nation'] is not None:
                leader['nation'] = Nation(leader['nation']['value'])
            leaders.append(cd.LeaderSpec(**leader))
        a['leaders'] = tuple(leaders)
        return cd.ArmySpec(**a)
    value = dict(value)
    value['attacker'], value['defender'] = army(value['attacker']), army(value['defender'])
    value['choices'] = tupled(value['choices'])
    return cd.ExchangeContext(**value)


def encode(dist):
    return zlib.compress(raw({
        'outcomes': list(dist.outcomes.items()),
        'hits_joint': list(dist.hits_joint.items()),
        'witness': (None if dist.witness is None else list(dist.witness.items()))}), 6)


def decode(payload):
    import combat_dist as cd
    value = json.loads(zlib.decompress(payload))
    dist = cd.ExchangeDist()
    dist.outcomes = {tupled(k): p for k, p in value['outcomes']}
    dist.hits_joint = {tupled(k): p for k, p in value['hits_joint']}
    dist.witness = (None if value['witness'] is None else
                    {tupled(k): tupled(p) for k, p in value['witness']})
    if abs(dist.total() - 1.0) > 1e-10 or any(p < 0 for p in dist.outcomes.values()):
        raise ValueError('Invalid stored combat probability mass')
    if dist.witness is not None and set(dist.witness) != set(dist.outcomes):
        raise ValueError('Stored witness coverage differs from outcomes')
    return dist


def runtime_manifest(source_dir=None):
    src = Path(source_dir or Path(__file__).resolve().parent).resolve()
    # Conservative dependency closure: no stale table after any Python runtime
    # source or combat data changes. Compute once per reader, never per lookup.
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(src.glob('*.py'))}
    if not {'ai.py', 'combat.py', 'combat_dist.py', 'combat_lookup.py'} <= sources.keys():
        raise ValueError('Incomplete combat runtime source directory')
    data = {n: hashlib.sha256((src.parent / 'data' / n).read_bytes()).hexdigest()
            for n in DATA_FILES}
    return {'schema': SCHEMA, 'sources': sources, 'data': data}


def fingerprint(manifest):
    return hashlib.sha256(raw(manifest)).hexdigest()


class Reader:
    def __init__(self, path, source_dir=None):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Combat table must be an existing absolute file path')
        self.pid = os.getpid()
        self.path = path.resolve()
        self.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            self.connection.execute('PRAGMA query_only=ON')
            meta_rows = self.connection.execute('SELECT value FROM metadata').fetchall()
            if len(meta_rows) != 1:
                raise ValueError('Combat table must have exactly one metadata row')
            meta = json.loads(meta_rows[0][0])
            actual = runtime_manifest(source_dir)
            if meta.get('schema') != SCHEMA or meta.get('runtime') != actual \
                    or meta.get('fingerprint') != fingerprint(actual):
                raise ValueError('Combat table runtime/schema fingerprint mismatch')
            self.metadata = meta
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def lookup(self, ctx, want_witness=False):
        if self.pid != os.getpid():
            raise RuntimeError('Combat table reader must be reopened after fork')
        key_raw = context_bytes(ctx)
        row = self.connection.execute(
            'SELECT context,payload,payload_sha FROM entries WHERE key=?',
            (hashlib.sha256(key_raw).digest(),)).fetchone()
        if row is None:
            return None
        context, payload, digest = row
        if context != key_raw or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Combat table entry identity/checksum mismatch')
        dist = decode(payload)
        return None if want_witness and dist.witness is None else dist


_READER = None
_PID = None
_INITIALIZED = False
STATS = {'hits': 0, 'misses': 0, 'disabled': 0}


def lookup(ctx, want_witness=False):
    """Optional lookup; any cache problem leaves exact computation available."""
    global _READER, _PID, _INITIALIZED
    pid = os.getpid()
    if pid != _PID:
        if _READER is not None:
            _READER.close()
        _READER, _PID, _INITIALIZED = None, pid, False
    try:
        if not _INITIALIZED:
            _INITIALIZED = True
            path = os.environ.get('WOTR_COMBAT_TABLE')
            if path:
                _READER = Reader(path)
        if _READER is None:
            return None
        dist = _READER.lookup(ctx, want_witness)
        STATS['hits' if dist is not None else 'misses'] += 1
        return dist
    except Exception as exc:
        if _READER is not None:
            _READER.close()
        _READER = None
        STATS['disabled'] += 1
        warnings.warn('Combat table disabled; using exact kernel: %s' % exc,
                      RuntimeWarning, stacklevel=2)
        return None
