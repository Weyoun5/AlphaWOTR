"""Opt-in single-writer, process-interruption journal for the existing loop.

No legacy adoption, power-loss guarantee, or atomic external pair visibility.
Required backups and ready inputs are never overwritten. Recovery validates
all paths/hashes before touching owned state. Arena is outside initial scope.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import re
import shutil
import stat

VERSION = 'single_writer_journal_v1'
# Environment keys a reviewed_runtime_source_transitions_v1 envelope may carry across a
# committed generation through a `kind: environment` hop (package LBP, USER 2026-09-11:
# production self-play at WOTR_LEAF_BATCH=K).  A search-SETTING change is a legitimate,
# identity-changing decision: it stays inside `controls.environment` (never excluded) and
# is accepted only through an explicit, hash-bound approval that names the changed keys.
# Any key not listed here keeps the original refusal.  Extending this tuple is a src
# edit on purpose (test_runtime_source_transition pins it).
REVIEWED_ENVIRONMENT_TRANSITION_KEYS = ('WOTR_LEAF_BATCH',)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    with open(path, encoding='utf8') as source:
        return json.load(source)


def _publish(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2), encoding='utf8')
    os.replace(tmp, path)


def _rng_capture():
    import numpy as np
    import torch
    ns = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _rng_restore(value):
    import numpy as np
    import torch
    random.setstate(value['python'])
    ns = value['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if len(value['cuda']) != torch.cuda.device_count():
            raise RuntimeError('Recovery GPU RNG device count changed')
        torch.cuda.set_rng_state_all(value['cuda'])


class BatchJournal:
    def __init__(self, workspace, live_paths, keys, controls):
        supplied = Path(workspace).absolute()
        self._no_links(supplied)
        self.root = supplied.resolve()
        self.keys = ['%s_%s' % tuple(k) for k in keys]
        if len(self.keys) != 2 or len(set(self.keys)) != 2:
            raise RuntimeError('Recovery requires one explicit pair')
        self.live = []
        for path in live_paths:
            target = Path(path).absolute()
            self._no_links(target)
            try:
                rel = target.resolve().relative_to(self.root).as_posix()
            except ValueError:
                raise RuntimeError('Live model is outside loop workspace') from None
            if '/' in rel or not re.fullmatch(r'live_[a-z]+_[a-z]+\.pt', rel):
                raise RuntimeError('Unsupported live model path')
            self.live.append(rel)
        if len(self.live) != 2 or len(set(self.live)) != 2:
            raise RuntimeError('Live model paths must be distinct')
        self.controls = controls
        self.meta = self.path('.batch_journal')
        self.marker = self.path('.batch_journal/committed.json')
        self.pending = self.path('.batch_journal/pending.json')
        self.pruning = self.path('.batch_journal/pruning.json')

    @staticmethod
    def _no_links(path):
        for p in [path, *path.parents]:
            if p.exists() or p.is_symlink():
                info = p.lstat()
                if p.is_symlink() or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
                    raise RuntimeError('Recovery refuses symbolic links/reparse paths: %s' % p)

    def path(self, rel):
        if not isinstance(rel, str) or '\\' in rel or ':' in rel:
            raise RuntimeError('Invalid recovery relative path')
        p = Path(rel)
        if p.is_absolute() or any(part in ('.', '..') for part in rel.split('/')):
            raise RuntimeError('Invalid recovery relative path')
        target = self.root / p
        self._no_links(target)
        if not target.resolve().is_relative_to(self.root):
            raise RuntimeError('Recovery path escaped workspace')
        return target

    def _owned(self, rel):
        if rel in self.live + ['loop_state.json', 'probe_cache.json']:
            return True
        if re.fullmatch(r'probe_after_batch_\d+\.json', rel):
            return True
        parts = rel.split('/')
        return (len(parts) == 3 and parts[0] == 'buffer' and parts[1] in self.keys
                and re.fullmatch(r'b\d+\.pkl\.gz(?:\.corrupt)?', parts[2]) is not None)

    def inventory(self):
        names = set(self.live + ['loop_state.json', 'probe_cache.json'])
        names.update(p.relative_to(self.root).as_posix() for p in self.root.glob('probe_after_batch_*.json'))
        for key in self.keys:
            folder = self.path('buffer/' + key)
            if folder.exists():
                for p in folder.iterdir():
                    rel = p.relative_to(self.root).as_posix()
                    if rel.endswith('.tmp') and self._owned(rel[:-4]):
                        continue
                    if not self._owned(rel) or not p.is_file():
                        raise RuntimeError('Unexpected replay inventory: ' + rel)
                    names.add(rel)
        result = {}
        for rel in sorted(names):
            p = self.path(rel)
            if p.exists():
                if not p.is_file():
                    raise RuntimeError('Owned artifact is not a file')
                result[rel] = digest(p)
        return result

    def _check_inventory(self, expected):
        if any(not self._owned(rel) for rel in expected):
            raise RuntimeError('Invalid owned inventory path')
        if self.inventory() != expected:
            raise RuntimeError('Committed training inventory differs from its marker')

    def fixed_inputs(self):
        names = [name.replace('live_', 'baseline_', 1) for name in self.live]
        names += ['probe_ref_seed.json', 'learning_contract.json', 'baseline_manifest.json']
        return {name: digest(self.path(name)) for name in names if self.path(name).is_file()}

    def bootstrap(self):
        if self.marker.exists():
            return
        if self.pending.exists() or any(self.root.glob('batch_*')) or any(self.root.glob('buffer/*/*')):
            raise RuntimeError('Recovery refuses legacy or incomplete loop migration; use a fresh directory')
        state = self.path('loop_state.json')
        if state.exists() and any(v.get('history') for v in _json(state).values()):
            raise RuntimeError('Recovery refuses existing training history')
        if not all(self.path(p).is_file() for p in self.live):
            raise RuntimeError('Recovery needs initialized model pair')
        self.meta.mkdir(parents=True, exist_ok=True)
        _publish(self.marker, {'version': VERSION, 'generation': -1,
                              'inventory': self.inventory(), 'controls': self.controls,
                              'fixed_inputs': self.fixed_inputs()})

    def _marker(self):
        marker = _json(self.marker)
        if marker.get('version') != VERSION or type(marker.get('generation')) is not int or marker['generation'] < -1:
            raise RuntimeError('Invalid committed marker')
        return marker

    def _record(self):
        pointer = _json(self.pending)
        idx = pointer.get('index')
        if type(idx) is not int or idx < 0 or set(pointer) != {'index', 'sha256'}:
            raise RuntimeError('Invalid pending pointer')
        folder = '.batch_journal/txn_%06d' % idx
        path = self.path(folder + '/record.json')
        if digest(path) != pointer['sha256']:
            raise RuntimeError('Pending record hash mismatch')
        record = _json(path)
        if record.get('version') != VERSION or record.get('index') != idx:
            raise RuntimeError('Invalid pending record')
        for rel, sha in record['before'].items():
            if not self._owned(rel) or digest(self.path(folder + '/backup/' + rel)) != sha:
                raise RuntimeError('Recovery backup invalid: ' + rel)
        if digest(self.path(folder + '/rng.pt')) != record['rng_sha256']:
            raise RuntimeError('Recovery RNG hash mismatch')
        return record, folder

    def recover(self):
        """Idempotent disk rollback; keep pending/backup for exact ready retry."""
        if not self.marker.exists():
            if self.pending.exists():
                raise RuntimeError('Pending transaction has no committed parent')
            return
        marker = self._marker()
        # A different experiment must never initiate rollback or cleanup of
        # another experiment's interrupted artifacts.
        if self.fixed_inputs() != marker['fixed_inputs']:
            raise RuntimeError('Committed baseline/probe/contract input hashes changed')
        if marker['controls'] != self.controls:
            raise RuntimeError('Recovery controls/source identity changed')
        if self.pending.exists():
            record, folder = self._record()
            if record.get('controls') != self.controls:
                raise RuntimeError('Pending training controls/source identity changed')
            if marker['generation'] == record['index']:
                if marker.get('record_sha256') != digest(self.path(folder + '/record.json')):
                    raise RuntimeError('Committed transaction identity mismatch')
                self._check_inventory(marker['inventory'])
                self.pending.unlink()
            else:
                if marker['generation'] != record['index'] - 1 or record['before'] != marker['inventory']:
                    raise RuntimeError('Pending parent generation mismatch')
                # Validate all current names before any removal or restoration.
                current = self.inventory()
                for rel in current.keys() - record['before'].keys():
                    self.path(rel).unlink()
                for rel, sha in record['before'].items():
                    target = self.path(rel)
                    if target.exists() and digest(target) == sha:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.path(rel + '.tmp')
                    shutil.copyfile(self.path(folder + '/backup/' + rel), tmp)
                    os.replace(tmp, target)
                # Include interrupted writer sidecars, but only exact owned names.
                for p in self.root.rglob('*.tmp'):
                    rel = p.relative_to(self.root).as_posix()
                    if self._owned(rel[:-4]):
                        self.path(rel).unlink()
                self._check_inventory(record['before'])
        self._check_inventory(marker['inventory'])
        self.prune_backups()

    def _transaction_tree(self, idx):
        """Inspect a canonical journal directory without following links."""
        folder = '.batch_journal/txn_%06d' % idx
        files, directories = {}, []
        def visit(rel):
            path = self.path(rel)
            if not path.exists():
                return
            if not path.is_dir():
                raise RuntimeError('Journal transaction is not a directory')
            directories.append(rel)
            for child in sorted(path.iterdir()):
                name = child.relative_to(self.root).as_posix()
                target = self.path(name)
                if target.is_dir():
                    visit(name)
                elif target.is_file():
                    files[name] = digest(target)
                else:
                    raise RuntimeError('Unsupported journal transaction entry')
        visit(folder)
        return files, directories

    def _prune_path(self, rel, generation):
        """Deletion capability restricted to obsolete transaction artifacts."""
        self.path(rel)
        match = re.fullmatch(r'\.batch_journal/txn_(\d{6,})/(.+)', rel)
        if not match:
            raise RuntimeError('Invalid journal pruning path')
        idx = int(match[1])
        if match[1] != '%06d' % idx or not 0 <= idx < generation:
            raise RuntimeError('Pruning would touch a protected transaction')
        tail = match[2]
        if tail not in ('record.json', 'rng.pt') and not (
                tail.startswith('backup/') and self._owned(tail[7:])):
            raise RuntimeError('Pruning path is not an owned backup artifact')
        return idx

    def _historical_controls_match(self, controls):
        """Accept a committed backup's original source identity after an upgrade.

        This is used only for retained/obsolete committed records. Pending
        training, rollback and ready-input checks still require exact current
        controls. Historical transaction records are never rewritten.
        """
        if controls == self.controls:
            return True
        path = self.path('.batch_journal/runtime_source_transitions.json')
        if not path.is_file():
            return False
        envelope = _json(path)
        if envelope.get('schema') != 'reviewed_runtime_source_transitions_v1':
            raise RuntimeError('Unknown runtime source transition schema')
        transitions = envelope.get('transitions')
        if not isinstance(transitions, list) or not 0 < len(transitions) <= 32:
            raise RuntimeError('Invalid runtime source transition history')
        def controls_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        cursor = controls
        for entry in transitions:
            before, after = entry['before'], entry['after']
            kind = entry.get('kind', 'sources')
            if kind == 'sources':
                # A runtime (search implementation) upgrade: identical learning
                # controls, different sources.
                if ({k:v for k,v in before.items() if k != 'sources'}
                        != {k:v for k,v in after.items() if k != 'sources'}
                        or before.get('sources') == after.get('sources')):
                    raise RuntimeError('Runtime upgrade attempted to change learning controls')
                schema = 'reviewed_runtime_source_transition_v1'
            elif kind == 'environment':
                # A reviewed search-SETTING change (REVIEWED_ENVIRONMENT_TRANSITION_KEYS):
                # every control except `environment` identical, and the environment
                # differs EXACTLY in the keys the entry names, each of them listed.
                # Reward, selector, contracts, data, globals, probes and sources keep
                # the refusal.  The approval must name the same keys.
                keys = entry.get('environment_keys')
                before_env = before.get('environment') or {}
                after_env = after.get('environment') or {}
                changed = {k for k in set(before_env) | set(after_env)
                           if before_env.get(k) != after_env.get(k)}
                if ({k:v for k,v in before.items() if k != 'environment'}
                        != {k:v for k,v in after.items() if k != 'environment'}
                        or not isinstance(keys, list) or not keys or len(set(keys)) != len(keys)
                        or any(k not in REVIEWED_ENVIRONMENT_TRANSITION_KEYS for k in keys)
                        or changed != set(keys)):
                    raise RuntimeError('Runtime environment transition attempted to change unapproved controls')
                schema = 'reviewed_runtime_environment_transition_v1'
            else:
                raise RuntimeError('Unknown runtime transition kind')
            approval = entry['approval']
            approved_path = Path(approval['path'])
            if not approved_path.is_absolute():
                raise RuntimeError('Runtime source approval must have an absolute path')
            self._no_links(approved_path)
            if digest(approved_path) != approval['sha256']:
                raise RuntimeError('Runtime source approval changed')
            proof = _json(approved_path)
            if (proof.get('schema') != schema
                    or proof.get('passed') is not True
                    or proof.get('before_controls_sha256') != controls_hash(before)
                    or proof.get('after_controls_sha256') != controls_hash(after)
                    or proof.get('committed_history_preserved') is not True
                    or (kind == 'environment'
                        and proof.get('changed_environment_keys') != sorted(keys))):
                raise RuntimeError('Runtime source transition is unapproved' if kind == 'sources'
                                   else 'Runtime environment transition is unapproved')
            if before == cursor:
                cursor = after
        return cursor == self.controls

    def prune_backups(self):
        """Keep last commit/pending; resume interrupted obsolete-file cleanup.

        A persisted deletion plan permits missing already-deleted files, never
        unexpected files or changed hashes. No live/replay/batch paths qualify.
        """
        marker = self._marker()
        generation = marker['generation']
        if marker['controls'] != self.controls or self.fixed_inputs() != marker['fixed_inputs']:
            raise RuntimeError('Pruning experiment identity changed')
        self._check_inventory(marker['inventory'])
        if generation < 0:
            if self.pruning.exists():
                raise RuntimeError('Pruning plan has no committed generation')
            return
        current = self.path('.batch_journal/txn_%06d/record.json' % generation)
        if digest(current) != marker.get('record_sha256'):
            raise RuntimeError('Pruning committed record binding mismatch')
        record = _json(current)
        if (record.get('version') != VERSION or record.get('index') != generation
                or not self._historical_controls_match(record.get('controls'))):
            raise RuntimeError('Pruning committed record identity mismatch')
        kept_folder = '.batch_journal/txn_%06d' % generation
        for name, value in record['before'].items():
            if not self._owned(name):
                raise RuntimeError('Invalid retained backup path')
            backup = self.path(kept_folder + '/backup/' + name)
            if not backup.is_file() or digest(backup) != value:
                raise RuntimeError('Retained transaction backup is missing or changed')
        rng = self.path(kept_folder + '/rng.pt')
        if not rng.is_file() or digest(rng) != record['rng_sha256']:
            raise RuntimeError('Retained transaction RNG is missing or changed')
        if self.pending.exists():
            pending, _folder = self._record()
            if (pending['index'] not in (generation, generation + 1)
                    or pending['controls'] != self.controls
                    or (pending['index'] == generation + 1 and pending['before'] != marker['inventory'])):
                raise RuntimeError('Pruning pending transaction identity mismatch')
        if self.pruning.exists():
            plan = _json(self.pruning)
            if (set(plan) != {'version', 'generation', 'controls', 'files'}
                    or plan['version'] != 'journal_backup_retention_v1'
                    or plan['generation'] != generation or plan['controls'] != self.controls
                    or not isinstance(plan['files'], dict)):
                raise RuntimeError('Invalid journal pruning plan')
        else:
            selected = {}
            for child in sorted(self.meta.iterdir()):
                match = re.fullmatch(r'txn_(\d{6,})', child.name)
                if not match:
                    continue
                idx = int(match[1])
                if child.name != 'txn_%06d' % idx or idx >= generation:
                    continue
                files, directories = self._transaction_tree(idx)
                folder = '.batch_journal/txn_%06d' % idx
                previous = _json(self.path(folder + '/record.json'))
                if (previous.get('version') != VERSION or previous.get('index') != idx
                        or not self._historical_controls_match(previous.get('controls'))):
                    raise RuntimeError('Obsolete transaction identity mismatch')
                expected = {folder + '/record.json': digest(self.path(folder + '/record.json')),
                            folder + '/rng.pt': previous['rng_sha256']}
                expected.update({folder + '/backup/' + name: value
                                 for name, value in previous['before'].items()})
                for name in expected:
                    self._prune_path(name, generation)
                allowed_directories = set()
                for name in expected:
                    parent = Path(name).parent
                    while parent != Path('.batch_journal'):
                        allowed_directories.add(parent.as_posix())
                        parent = parent.parent
                if files != expected or any(name not in allowed_directories for name in directories):
                    raise RuntimeError('Obsolete transaction backup inventory mismatch')
                selected.update(expected)
            if not selected:
                return
            plan = {'version': 'journal_backup_retention_v1', 'generation': generation,
                    'controls': self.controls, 'files': selected}
            # Fully validated before a single old backup can be deleted.
            _publish(self.pruning, plan)
        indices = set()
        expected_directories = set()
        for name in plan['files']:
            idx = self._prune_path(name, generation)
            indices.add(idx)
            parent = Path(name).parent
            limit = Path('.batch_journal/txn_%06d' % idx)
            while parent != limit.parent:
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        # Validate the entire remaining cleanup inventory before any mutation.
        for idx in indices:
            files, directories = self._transaction_tree(idx)
            if (any(name not in plan['files'] or plan['files'][name] != value
                    for name, value in files.items())
                    or any(name not in expected_directories for name in directories)):
                raise RuntimeError('Interrupted pruning inventory changed')
        for name in sorted(plan['files']):
            path = self.path(name)
            if path.exists():
                path.unlink()
        for name in sorted(expected_directories, key=lambda value: (-value.count('/'), value)):
            path = self.path(name)
            if path.exists():
                path.rmdir()
        self.pruning.unlink()

    def next_index(self):
        self.recover()
        return self._marker()['generation'] + 1

    def ready(self, idx):
        if idx != self.next_index():
            raise RuntimeError('Batch index is not next committed generation')
        if not self.pending.exists():
            if any(self.root.glob('batch_%03d*' % idx)):
                raise RuntimeError('Incomplete played batch has no ready manifest; retained for review')
            return None
        record, _folder = self._record()
        if record['controls'] != self.controls:
            raise RuntimeError('Ready batch controls/source identity changed')
        batch = self.path('batch_%03d' % idx)
        if self._batch_inventory(batch) != record['inputs']:
            raise RuntimeError('Ready batch input hashes changed')
        return str(batch)

    def _batch_inventory(self, batch):
        result = {}
        for path in batch.rglob('*'):
            rel = path.relative_to(self.root).as_posix()
            self.path(rel)
            if path.is_file():
                result[path.relative_to(batch).as_posix()] = digest(path)
        if not result:
            raise RuntimeError('Ready batch is empty')
        return result

    def begin(self, idx, batch):
        import torch
        expected = self.path('batch_%03d' % idx)
        if Path(batch).resolve() != expected:
            raise RuntimeError('Played batch is outside expected workspace path')
        if self.pending.exists():
            record, folder = self._record()
            # Extractor can rewrite deterministic admission output; it must match.
            if self._batch_inventory(expected) != record['inputs']:
                raise RuntimeError('Re-extracted ready batch changed')
            _rng_restore(torch.load(self.path(folder + '/rng.pt'), weights_only=True))
            return
        marker = self._marker()
        if idx != marker['generation'] + 1:
            raise RuntimeError('Cannot begin out-of-order batch')
        self._check_inventory(marker['inventory'])
        folder = '.batch_journal/txn_%06d' % idx
        location = self.path(folder)
        # Orphan preparation is retained and refused, never overwritten.
        location.mkdir(exist_ok=False)
        before = self.inventory()
        for rel, sha in before.items():
            target = self.path(folder + '/backup/' + rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(rel), target)
            if digest(target) != sha:
                raise RuntimeError('Backup copy verification failed')
        rng = self.path(folder + '/rng.pt')
        torch.save(_rng_capture(), rng)
        record = {'version': VERSION, 'index': idx, 'before': before,
                  'controls': self.controls, 'inputs': self._batch_inventory(expected),
                  'rng_sha256': digest(rng)}
        _publish(self.path(folder + '/record.json'), record)
        _publish(self.pending, {'index': idx, 'sha256': digest(self.path(folder + '/record.json'))})

    def commit(self, idx):
        record, folder = self._record()
        parent = self._marker()
        if idx != record['index'] or parent['generation'] != idx - 1:
            raise RuntimeError('Commit generation mismatch')
        if self.fixed_inputs() != parent['fixed_inputs']:
            raise RuntimeError('Fixed inputs changed during training')
        if self.controls != parent['controls'] or record['controls'] != self.controls:
            raise RuntimeError('Commit controls/source identity changed')
        if not all(self.path(name).is_file() for name in self.live):
            raise RuntimeError('Commit requires both live model files')
        if self._batch_inventory(self.path('batch_%03d' % idx)) != record['inputs']:
            raise RuntimeError('Ready batch input hashes changed before commit')
        _publish(self.marker, {'version': VERSION, 'generation': idx,
                              'inventory': self.inventory(), 'controls': self.controls,
                              'fixed_inputs': parent['fixed_inputs'],
                              'record_sha256': digest(self.path(folder + '/record.json'))})
        # Publication above is authoritative even if this cleanup is interrupted.
        self.pending.unlink()
        self.prune_backups()
