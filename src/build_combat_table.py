"""Offline, single-writer exact table builder. Never invoked by AI workers."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import random
import sqlite3
import time

import combat_dist as cd
import combat_lookup as lookup
from game_state import Nation, Side


def army(r, e, ei, leadership, shadow):
    nation = Nation.SAURON if shadow else Nation.GONDOR
    return cd.ArmySpec(r, e, nation, nation,
                       (cd.LeaderSpec('nazgul' if shadow else 'leader'),) * leadership,
                       ei, Nation.ISENGARD)


def small_catalog(max_units=5, max_leadership=3):
    """Exact literal contexts, no semantic canonicalization or aliases."""
    fp_counts = [(r, e, 0) for r in range(max_units + 1)
                 for e in range(max_units + 1 - r) if r + e]
    sh_counts = [(r, e, ei) for r in range(max_units + 1)
                 for e in range(max_units + 1 - r)
                 for ei in range(max_units + 1 - r - e) if r + e + ei]
    for f, s, lf, ls, terrain, fp_attacks, saruman in itertools.product(
            fp_counts, sh_counts, range(max_leadership + 1),
            range(max_leadership + 1), ('field', 'city', 'siege'), (False, True), (False, True)):
        fp, sh = army(*f, lf, False), army(*s, ls, True)
        a, d = (fp, sh) if fp_attacks else (sh, fp)
        if terrain == 'siege' and d.regulars + d.elites + d.elites2 > 5:
            continue
        yield cd.ExchangeContext(a, d, is_siege=terrain == 'siege', city=terrain == 'city',
                                 defer_casualties=True, saruman_in_play=saruman)


def representative_cards():
    import cert_combat_dist as cert
    from cards import get_cards
    names = {side: sorted({c.combat_name for c in get_cards().values()
                          if c.side == side and c.combat_name and c.combat_name != 'Scouts'})
             for side in Side}
    fp = dataclasses.replace(army(5, 5, 0, 0, False), leaders=(
        cert.C('Aragorn', 3, 2), cert.C('Legolas', 2, 1), cert.C('Gimli', 2, 1),
        cert.C('Gandalf the White', 3, 1), cd.LeaderSpec('leader')))
    sh = dataclasses.replace(army(5, 3, 2, 5, True), leaders=cert.NZ(5) + (cert.M('Witch King'),))
    for terrain, fp_attacks in itertools.product(('field', 'city', 'siege'), (False, True)):
        a, d = (fp, sh) if fp_attacks else (sh, fp)
        if terrain == 'siege':
            d = dataclasses.replace(d, regulars=2, elites=3, elites2=0)
        for ca, cb in itertools.product(
                [None] + names[Side.FREE_PEOPLES if fp_attacks else Side.SHADOW],
                [None] + names[Side.SHADOW if fp_attacks else Side.FREE_PEOPLES]):
            yield cd.ExchangeContext(a, d, ca, cb, is_siege=terrain == 'siege', city=terrain == 'city',
                                     shine=False, defer_casualties=True, saruman_in_play=True)


def build(output, contexts, description):
    output = Path(output).resolve()
    partial = output.with_suffix(output.suffix + '.building')
    if output.exists() or partial.exists():
        raise FileExistsError('Fresh table output required')
    before = lookup.runtime_manifest()
    conn = sqlite3.connect(partial)
    conn.execute('CREATE TABLE metadata(value TEXT NOT NULL)')
    conn.execute('CREATE TABLE entries(key BLOB PRIMARY KEY,context BLOB NOT NULL,'
                 'payload BLOB NOT NULL,payload_sha TEXT NOT NULL) WITHOUT ROWID')
    start = time.perf_counter()
    catalog_hash = hashlib.sha256()
    count = attempts = full_outcomes = payload_bytes = 0
    rs = random.getstate()
    try:
        for ctx in contexts:
            attempts += 1
            if not ctx.defer_casualties:
                raise ValueError('This catalog requires deferred-loss witness contexts')
            key_raw = lookup.context_bytes(ctx)
            key = hashlib.sha256(key_raw).digest()
            if conn.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone():
                continue
            dist = cd.exact_distribution(ctx, want_witness=True)
            payload = lookup.encode(dist)
            decoded = lookup.decode(payload)
            if (decoded.outcomes, decoded.hits_joint, decoded.witness) != (
                    dist.outcomes, dist.hits_joint, dist.witness):
                raise ValueError('Lossless codec verification failed')
            conn.execute('INSERT INTO entries VALUES(?,?,?,?)',
                         (key, key_raw, payload, hashlib.sha256(payload).hexdigest()))
            catalog_hash.update(key)
            count += 1
            full_outcomes += len(dist.outcomes)
            payload_bytes += len(payload)
            if count % 10000 == 0:
                conn.commit()
                print(json.dumps({'contexts': count, 'seconds': time.perf_counter() - start}), flush=True)
        if lookup.runtime_manifest() != before:
            raise ValueError('Runtime changed while building table')
        meta = {'schema': lookup.SCHEMA, 'fingerprint': lookup.fingerprint(before),
                'runtime': before, 'catalog': description, 'context_order_sha256': catalog_hash.hexdigest(),
                'entries': count, 'attempts': attempts, 'full_outcomes': full_outcomes,
                'payload_bytes': payload_bytes, 'build_seconds': time.perf_counter() - start,
                'all_outcomes_and_witnesses_preserved': True}
        conn.execute('INSERT INTO metadata VALUES(?)', (lookup.raw(meta).decode('utf-8'),))
        conn.commit()
        if conn.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ValueError('SQLite integrity check failed')
        conn.close()
        partial.rename(output)
        meta['database_bytes'] = output.stat().st_size
        with output.open('rb') as stream:
            meta['database_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        output.with_suffix(output.suffix + '.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return meta
    finally:
        random.setstate(rs)
        conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--catalog-jsonl', help='Exact plain ExchangeContext objects, one per line')
    p.add_argument('--max-units', type=int, default=5)
    p.add_argument('--max-leadership', type=int, default=3)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if args.catalog_jsonl:
        source = Path(args.catalog_jsonl).resolve()
        description = {'kind': 'explicit-contexts', 'source': str(source),
                       'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        contexts = (lookup.context_from_plain(json.loads(line))
                    for line in source.read_text(encoding='utf-8').splitlines() if line.strip())
    else:
        if not 1 <= args.max_units <= 10 or not 0 <= args.max_leadership <= 9:
            raise ValueError('Bounded catalog requires units 1..10, leadership 0..9')
        import cert_combat_dist as cert
        description = {'kind': 'small-army-no-card-plus-representative-cards',
                       'max_units': args.max_units, 'max_generic_leadership': args.max_leadership,
                       'terrains': ['field', 'city', 'siege'], 'both_orientations': True,
                       'saruman_states': [False, True], 'small_choices': [], 'small_shine': None,
                       'card_contexts': 2268, 'certification_contexts': len(cert.build_cases()),
                       'scope': 'Small-army no-card grid; one rich roster/card grid; certification examples. '
                                'Other named rosters, choices, leadership/order variants and larger armies use exact fallback.'}
        contexts = itertools.chain(small_catalog(args.max_units, args.max_leadership),
                                   representative_cards(),
                                   (dataclasses.replace(c, defer_casualties=True) for _, c in cert.build_cases()))
    meta = build(output, contexts, description)
    print(json.dumps({k: v for k, v in meta.items() if k != 'runtime'}), flush=True)


if __name__ == '__main__':
    main()
