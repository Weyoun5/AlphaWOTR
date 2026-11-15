"""Content-bound, atomically published frozen-tower feature stores."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .contracts import CONTRACT, ContractError, canonical_json
from .durability import digest_file, digest_json, file_lock
from .runtime_io import write_json


def tower_identity(critic):
    """Hash the actual in-memory state, including buffers and feature semantics."""
    import torch
    towers = {}
    for side, tower in sorted(critic.towers.items()):
        h = hashlib.sha256()
        for name, tensor in sorted(tower.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            h.update(canonical_json([name, str(tensor.dtype), list(tensor.shape)]).encode('ascii'))
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        towers[side] = dict(sha256=h.hexdigest(), training=bool(tower.training),
                            modules=[[name, type(module).__module__, type(module).__qualname__]
                                     for name, module in tower.named_modules()])
    return dict(towers=towers, hidden=critic.hidden, metadata=critic.meta.get('towers', {}))


def feature_request(run, critic, folders):
    episodes = []
    for folder in folders:
        folder = Path(folder).resolve()
        episodes.append(dict(path=str(folder),
            files={name: digest_file(folder / name) if (folder / name).exists() else None
                   for name in ('episode.json', 'arrays.npz')}))
    request = dict(schema='both_views_features_v2', feature_spec='sh_h_fp_h_sh_u_neg_fp_u_float16_v1',
                contract=CONTRACT.to_dict(), run=str(Path(run).resolve()),
                towers=tower_identity(critic), episodes=episodes)
    return json.loads(canonical_json(request))


def _load(folder, request):
    receipt = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    if receipt.get('request') != request or receipt.get('schema') != 'both_views_feature_completion_v1':
        raise ContractError('Critic feature cache receipt does not match its inputs')
    if set(receipt.get('files', {})) != {'features.f16', 'labels.npz'}:
        raise ContractError('Incomplete critic feature cache receipt')
    for name, sha in receipt['files'].items():
        if digest_file(folder / name) != sha:
            raise ContractError('Modified or incomplete critic feature cache: ' + name)
    rows, width = int(receipt['rows']), int(receipt['width'])
    if rows < 0 or width != 2 * request['towers']['hidden'] + 2:
        raise ContractError('Invalid critic feature cache shape')
    with np.load(folder / 'labels.npz', allow_pickle=False) as archive:
        labels = {key: archive[key] for key in ('z', 'turn', 'seed', 'sh_row')}
    if any(values.shape != (rows,) for values in labels.values()):
        raise ContractError('Critic feature labels do not match the completed rows')
    if (folder / 'features.f16').stat().st_size != max(1, int(receipt['allocated_rows'])) * width * 2:
        raise ContractError('Critic feature store has an invalid byte length')
    store = (np.memmap(folder / 'features.f16', np.float16, 'r', shape=(rows, width))
             if rows else np.empty((0, width), np.float16))
    return store, labels, int(receipt['games'])


def read_completed_store(output, run):
    """Read one unambiguous completed store for this exact run.

    A specific completed store directory can be supplied to select among tower
    generations. Never choose a generation by mtime. Historical flat stores are
    readable only when no current-format store for this run is present.
    """
    output, run = Path(output).resolve(), Path(run).resolve()
    paths = ([output / 'complete.json'] if (output / 'complete.json').is_file()
             else sorted((output / 'features').glob('*/complete.json')))
    candidates, matching_run = [], False
    for path in paths:
        receipt = json.loads(path.read_text(encoding='utf-8'))
        request = receipt.get('request') or {}
        if request.get('run') != str(run):
            continue
        matching_run = True
        if all((Path(episode['path']) / name).is_file() and
               digest_file(Path(episode['path']) / name) == sha
               for episode in request.get('episodes', [])
               for name, sha in episode['files'].items()):
            candidates.append((path, request))
    if len(candidates) > 1:
        raise ContractError('Multiple current critic feature stores; pass the intended completed store directory explicitly')
    if candidates:
        path, request = candidates[0]
        store, labels, games = _load(path.parent, request)
        return store, labels, games, dict(schema='completed_critic_features_v1',
            path=str(path.parent), completion_sha256=digest_file(path))
    if matching_run:
        raise ContractError('Critic feature store episodes changed; rebuild the features')
    # Preserve explicit historical consumers without presenting their files as
    # newly certified tower outputs. Bind these exact bytes in the derived index.
    labels_path = output / 'features' / (run.name + '.labels.npz')
    feature_path = output / 'features' / (run.name + '.f16')
    if not labels_path.is_file() or not feature_path.is_file():
        raise ContractError('No completed critic feature store for ' + str(run))
    with np.load(labels_path, allow_pickle=False) as archive:
        rows = int(archive['rows'])
        labels = {key: archive[key] for key in ('z', 'turn', 'seed', 'sh_row')}
    if rows < 0 or any(values.shape != (rows,) for values in labels.values()):
        raise ContractError('Invalid historical critic feature labels')
    width = 514  # The historical format was fixed at two 256-wide towers.
    if feature_path.stat().st_size != rows * width * 2:
        raise ContractError('Invalid historical critic feature byte length')
    store = (np.memmap(feature_path, np.float16, 'r', shape=(rows, width))
             if rows else np.empty((0, width), np.float16))
    identity = dict(schema='historical_flat_critic_features_v1', path=str(feature_path),
                    features_sha256=digest_file(feature_path), labels_sha256=digest_file(labels_path))
    return store, labels, len(np.unique(labels['seed'])), identity


def feature_store(output, run, critic, folders, *, workers=6, chunksize=4, log_every=2000):
    from . import full_state_critic as fc
    folders = list(folders)
    request = feature_request(run, critic, folders)
    root = Path(output) / 'features'
    root.mkdir(parents=True, exist_ok=True)
    name = Path(run).name + '-' + digest_json(request)
    final = root / name
    with file_lock(root / (name + '.lock')):
        if (final / 'complete.json').exists():
            result = _load(final, request)
            if feature_request(run, critic, folders) != request:
                raise ContractError('Critic feature inputs changed during cache admission')
            return result
        stage = root / ('.' + name + '.staged-' + uuid.uuid4().hex)
        stage.mkdir()
        counts = [fc.stored_rows(folder) for folder in folders]
        keep = [folder for folder, count in zip(folders, counts) if count > 0]
        rows, width = int(sum(counts)), 2 * critic.hidden + 2
        if rows < 0:
            raise ContractError('Negative announced critic feature row count')
        store = np.memmap(stage / 'features.f16', np.float16, 'w+', shape=(max(rows, 1), width))
        labels = dict(z=np.zeros(rows, np.float32), turn=np.zeros(rows, np.int16),
                      seed=np.zeros(rows, np.int64), sh_row=np.zeros(rows, bool))
        at, done = 0, 0

        def consume(chunks):
            nonlocal at, done
            for chunk in chunks:
                for game in chunk:
                    if game is None:
                        continue
                    n = len(game['turn'])
                    if at + n > rows:
                        raise ContractError('More stored rows than episode.json announced')
                    h = critic.tower_features(game['sh_view'], game['fp_view'])
                    store[at:at+n] = h.half().cpu().numpy()
                    del h
                    for key in ('z', 'turn', 'seed'):
                        labels[key][at:at+n] = game[key]
                    labels['sh_row'][at:at+n] = game['side'] == 'sh'
                    at += n
                    done += 1
                    game.clear()
                    if log_every and done % log_every == 0:
                        store.flush()
                        print(f'  features of {done} games ({at} rows)', flush=True)

        try:
            if workers == 0:  # Explicit bounded serial use; normal CLI retains parallel readers.
                consume(fc.read_games(keep[start:start+chunksize]) for start in range(0, len(keep), chunksize))
            else:
                with fc.ProcessPoolExecutor(workers) as pool:
                    consume(fc.bounded_map(pool, fc.read_games, keep, chunksize=chunksize, inflight=2*workers))
            store.flush()
        finally:
            store._mmap.close()  # Windows directory publication must have no open writer mapping.
        labels = {key: values[:at] for key, values in labels.items()}
        np.savez(stage / 'labels.npz', **labels)
        if feature_request(run, critic, folders) != request:
            raise ContractError('Critic weights or episode contents changed while computing features')
        write_json(stage / 'complete.json', dict(schema='both_views_feature_completion_v1', request=request,
            rows=at, games=done, width=width, allocated_rows=rows,
            files={key: digest_file(stage / key) for key in ('features.f16', 'labels.npz')}))
        if final.exists():
            final.rename(root / (name + '.incomplete-' + uuid.uuid4().hex))
        os.replace(stage, final)
        return _load(final, request)
