"""Explicit receipt-checked import of completed games into a fresh served run.

No partial trace or error attempt is imported. Originals remain untouched.
Only isolated episode RNG permits skipping previous completed identities.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError('Completed-game reuse refused: ' + message)


def load(args, paths, seed):
    """Validate an explicit reviewed manifest without changing any files."""
    manifest_path = getattr(args, 'reuse_completed_manifest', None)
    hashes = (getattr(args, 'reuse_completed_sha256', None),
              getattr(args, 'reuse_review', None), getattr(args, 'reuse_review_sha256', None))
    if manifest_path is None:
        require(not any(hashes), 'partial reuse arguments')
        return None
    require(all(hashes), 'manifest/review require exact hashes')
    pins = {}
    def read(ref, lines=False):
        p = Path(ref['path']).resolve()
        expected = ref['sha256']
        require(isinstance(expected, str) and len(expected) == 64, 'missing evidence hash')
        raw = p.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'changed evidence: ' + str(p))
        pins[str(p)] = expected
        text = raw.decode('utf-8-sig')
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)
    manifest = read({'path': manifest_path, 'sha256': hashes[0]})
    review = read({'path': hashes[1], 'sha256': hashes[2]})
    require(manifest.get('schema') == 'served_completed_reuse_v1', 'unknown manifest schema')
    require(review.get('schema') == 'served_completed_reuse_review_v1' and review.get('passed') is True
            and review.get('manifest_sha256') == hashes[0], 'independent review mismatch')
    from rng_streams import current_contract, ISOLATED, served_episode_seed
    require(current_contract() == ISOLATED, 'legacy worker RNG cannot skip completed episodes')
    target = manifest['target']
    expected = {'games': args.games, 'workers': args.workers, 'sims': args.sims,
                'arch': args.arch, 'worker_seed_base': seed, 'rng_contract': ISOLATED,
                'profiles': {side: os.environ.get('WOTR_SP_' + side.upper() + '_PROFILES') for side in ('fp', 'sh')},
                'checkpoint_hashes': {side: digest(path) for side, path in paths.items()}}
    require(target == expected and all(',' not in str(p) and p for p in expected['profiles'].values()),
            'target seed/checkpoint/profile/budget differs')
    launch = read(manifest['source_launch'])
    attempts = read(manifest['source_attempts'], lines=True)
    games = read(manifest['source_games'], lines=True)
    stop = read(manifest['stop_proof'])
    require(stop.get('passed') is True and stop.get('stage_pid_absent') is True
            and stop.get('user_cancelled') is False, 'old attempt quiescence not proved')
    preserved = {str(Path(v['path']).resolve()): v['sha256'] for v in stop.get('preserved_inputs', [])}
    for field in ('source_launch', 'source_attempts', 'source_games'):
        ref = manifest[field]
        require(preserved.get(str(Path(ref['path']).resolve())) == ref['sha256'], 'stop proof does not bind ' + field)
    for field in ('games', 'workers', 'sims', 'arch', 'worker_seed_base', 'rng_contract'):
        require(launch.get(field) == target[field], 'original launch differs: ' + field)
    require({side: item['sha256'] for side, item in launch['checkpoints'].items()} == target['checkpoint_hashes'],
            'original model pair differs')
    def index(rows):
        indexed = {}
        for row in rows:
            identity = (row.get('worker_id'), row.get('episode'))
            require(all(type(n) is int and n >= 0 for n in identity) and identity not in indexed,
                    'invalid or duplicate source identity')
            indexed[identity] = row
        return indexed
    indexed_attempts, indexed_games = index(attempts), index(games)
    selected = manifest.get('games')
    require(isinstance(selected, list) and 0 < len(selected) <= args.games, 'requires a nonempty completed set within the planned game budget')
    imported, seen = [], set()
    approved = review.get('trace_hashes')
    approved_sidecars = review.get('sidecar_hashes')
    require(isinstance(approved, dict) and isinstance(approved_sidecars, dict),
            'missing independently reviewed trace/sidecar hashes')

    def relocated_sidecar_origin(game, trace_hash, sidecar_hash, final_trace):
        """Follow only previously approved immutable imports, retaining every pin."""
        prior_game = game
        seen_manifests = set()
        chain = []
        for _ in range(64):
            previous = prior_game.get('reused_from')
            require(isinstance(previous, dict), 'relocated sidecar lacks prior import provenance')
            old_manifest = read({'path': previous['manifest_path'], 'sha256': previous['manifest_sha256']})
            old_review = read({'path': previous['review_path'], 'sha256': previous['review_sha256']})
            require(previous['manifest_sha256'] not in seen_manifests, 'cyclic prior import provenance')
            seen_manifests.add(previous['manifest_sha256'])
            require(old_manifest.get('schema') == 'served_completed_reuse_v1'
                    and old_manifest.get('target') == target
                    and old_review.get('schema') == 'served_completed_reuse_review_v1'
                    and old_review.get('passed') is True
                    and old_review.get('manifest_sha256') == previous['manifest_sha256'],
                    'prior import review/target differs')
            old_trace = str(Path(previous['trace_path']).resolve())
            old_sidecar = str(Path(previous['sidecar_path']).resolve())
            require(old_sidecar == old_trace + '.combat_eval.jsonl'
                    and previous['trace_sha256'] == trace_hash
                    and previous['sidecar_sha256'] == sidecar_hash
                    and old_review.get('trace_hashes', {}).get(old_trace) == trace_hash
                    and old_review.get('sidecar_hashes', {}).get(old_sidecar) == sidecar_hash,
                    'prior import trace/sidecar hashes differ')
            selected_prior = [r for r in old_manifest.get('games', [])
                              if (r.get('worker_id'), r.get('episode')) == (game['worker_id'], game['episode'])]
            require(len(selected_prior) == 1 and selected_prior[0].get('trace_path') == old_trace
                    and selected_prior[0].get('trace_sha256') == trace_hash
                    and selected_prior[0].get('sidecar') == {'path':old_sidecar,'sha256':sidecar_hash},
                    'prior manifest did not admit the same game')
            for path, value in ((old_trace, trace_hash), (old_sidecar, sidecar_hash)):
                require(digest(path) == value, 'prior imported artifact changed')
                pins[path] = value
            old_games = index(read(old_manifest['source_games'], lines=True))
            old_attempts = index(read(old_manifest['source_attempts'], lines=True))
            key = (game['worker_id'], game['episode'])
            prior_game = old_games.get(key, {})
            prior_attempt = old_attempts.get(key, {})
            original_trace = selected_prior[0].get('source_trace_path', old_trace)
            if original_trace != old_trace:
                require(old_review.get('relocations', {}).get(old_trace) == original_trace,
                        'prior archived trace relocation is not reviewed')
            require(prior_game.get('status') == prior_attempt.get('status') == 'completed'
                    and prior_game.get('trace_path') == prior_attempt.get('trace_path') == original_trace
                    and prior_game.get('checkpoint_ids') == game.get('checkpoint_ids')
                    and prior_game.get('profiles') == game.get('profiles'),
                    'prior source receipts do not describe the same completed game')
            chain.append({'manifest_path':previous['manifest_path'],'manifest_sha256':previous['manifest_sha256'],
                          'review_path':previous['review_path'],'review_sha256':previous['review_sha256'],
                          'trace_path':old_trace,'sidecar_path':old_sidecar})
            if Path(original_trace) == final_trace:
                return chain
        raise RuntimeError('Completed-game reuse refused: prior import chain exceeds64 links')

    for item in selected:
        identity = (item.get('worker_id'), item.get('episode'))
        require(all(type(n) is int and n >= 0 for n in identity) and identity not in seen, 'duplicate/invalid import identity')
        wid, episode = identity
        require(wid < min(args.workers, args.games)
                and episode < args.games // args.workers + (wid < args.games % args.workers), 'out-of-budget identity')
        seen.add(identity)
        game, attempt = indexed_games.get(identity, {}), indexed_attempts.get(identity, {})
        require(game.get('status') == attempt.get('status') == 'completed', 'source attempt was not completed')
        require(game.get('profiles') == attempt.get('profiles') == target['profiles'], 'source profiles differ')
        require(game.get('checkpoint_ids') == attempt.get('checkpoint_ids'), 'source checkpoint identities differ')
        trace = Path(item['trace_path']).resolve()
        source_trace = Path(item.get('source_trace_path', str(trace))).resolve()
        require(source_trace == Path(game.get('trace_path', '')).resolve()
                == Path(attempt.get('trace_path', '')).resolve(), 'trace identity differs from source receipts')
        if source_trace != trace:
            require(review.get('relocations', {}).get(str(trace)) == str(source_trace),
                    'archived trace relocation requires an exact review')
        require(digest(trace) == item['trace_sha256'] and approved.get(str(trace)) == item['trace_sha256'],
                'changed or unreviewed completed trace')
        pins[str(trace)] = item['trace_sha256']
        sidecar = Path(str(trace) + '.combat_eval.jsonl')
        sidecar_ref = item.get('sidecar', {})
        require(sidecar_ref.get('path') == str(sidecar)
                and approved_sidecars.get(str(sidecar)) == sidecar_ref.get('sha256'),
                'missing or unreviewed evaluation sidecar')
        events = read(sidecar_ref, lines=True)
        starts = [r for r in events if r.get('kind') == 'episode_start']
        ends = [r for r in events if r.get('kind') == 'episode_end']
        finals = [r for r in events if r.get('kind') == 'final_trace']
        require(len(starts) == len(ends) == len(finals) == 1 and events[0] is starts[0]
                and events[-1] is ends[0] and ends[0].get('success') is True,
                'sidecar does not describe the same successful completed game')
        final_trace = Path(finals[0]['trace_path']).resolve()
        origin_chain = [] if final_trace == source_trace else relocated_sidecar_origin(
            game, item['trace_sha256'], sidecar_ref['sha256'], final_trace)
        # Reuse never waives the ordinary predecision/schema/rules admission.
        import game_trace
        from trace_admission import validate_trace
        from reward_contract import current_contract as reward_contract
        from trace_decisions import current_learning_context
        from selector_contract import training_identity
        tr = game_trace.load_trace(str(trace))
        admission = validate_trace(tr, decision_contract=reward_contract(),
                                   **dict(current_learning_context(), **training_identity()))
        require(admission.get('accepted') is True, 'completed trace fails ordinary admission')
        meta = tr.get('meta', {})
        rng = meta.get('rng_episode', {})
        require(meta.get('rng_contract') == ISOLATED and rng.get('contract') == ISOLATED
                and rng.get('episode_identity') == {'run_seed': seed, 'episode_index': wid + episode * args.workers}
                and rng.get('episode_seed') == served_episode_seed(seed, wid + episode * args.workers),
                'trace isolated episode RNG identity differs')
        require(meta.get('profiles') == target['profiles']
                and meta.get('checkpoint_ids') == game.get('checkpoint_ids')
                and all(meta.get(k) == game.get(k) for k in ('winner', 'victory_type')),
                'trace outcome/profile/checkpoint metadata differs')
        imported.append({'record': game, 'trace': str(trace), 'sha256': item['trace_sha256'],
                         'sidecar': str(sidecar), 'sidecar_sha256': sidecar_ref['sha256'], 'admission': admission,
                         'sidecar_origin_chain': origin_chain})
    require(set(approved) == {item['trace'] for item in imported}, 'reviewed/imported trace population differs')
    require(set(approved_sidecars) == {item['sidecar'] for item in imported},
            'reviewed/imported sidecar population differs')
    require(review.get('relocations', {}) == {
        str(Path(item['trace_path']).resolve()): str(Path(item['source_trace_path']).resolve())
        for item in selected if item.get('source_trace_path')
        and Path(item['source_trace_path']).resolve() != Path(item['trace_path']).resolve()},
        'reviewed/imported relocation population differs')
    for path, expected_sha in pins.items():
        require(digest(path) == expected_sha, 'evidence changed during validation')
    return {'manifest_path': str(Path(manifest_path).resolve()), 'manifest_sha256': hashes[0],
            'review_path': str(Path(hashes[1]).resolve()), 'review_sha256': hashes[2],
            'pins': pins, 'games': imported}


def materialize(reuse, output):
    """Copy proved complete traces, retaining source provenance in each receipt."""
    if reuse is None:
        return []
    output = Path(output).resolve()
    trace_dir = output / 'traces'
    trace_dir.mkdir(exist_ok=True)
    records = []
    for item in reuse['games']:
        row = dict(item['record'])
        target = trace_dir / Path(item['trace']).name
        require(not target.exists(), 'import target already exists')
        shutil.copyfile(item['trace'], target)
        require(digest(target) == item['sha256'], 'trace copy changed')
        sidecar_target = trace_dir / Path(item['sidecar']).name
        require(not sidecar_target.exists(), 'sidecar import target already exists')
        shutil.copyfile(item['sidecar'], sidecar_target)
        require(digest(sidecar_target) == item['sidecar_sha256'], 'sidecar copy changed')
        row.update(trace_path=str(target), reused_from={'manifest_path': reuse['manifest_path'],
                   'manifest_sha256': reuse['manifest_sha256'], 'trace_path': item['trace'],
                   'trace_sha256': item['sha256'], 'review_path': reuse['review_path'],
                   'review_sha256': reuse['review_sha256'], 'sidecar_path': item['sidecar'],
                   'sidecar_sha256': item['sidecar_sha256']})
        if item['record'].get('reused_from'):
            row['reused_from']['prior_reuse'] = item['record']['reused_from']
        if item.get('sidecar_origin_chain'):
            row['reused_from']['sidecar_origin_chain'] = item['sidecar_origin_chain']
        records.append(row)
    for path, expected_sha in reuse['pins'].items():
        require(digest(path) == expected_sha, 'reuse evidence changed during copy')
    receipt = {k: reuse[k] for k in ('manifest_path', 'manifest_sha256', 'review_path', 'review_sha256', 'pins')}
    receipt['imported_identities'] = [[r['worker_id'], r['episode']] for r in records]
    (output / 'completed_reuse.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return records
