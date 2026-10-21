"""Preserve and retry proved-quiescent served attempts before ready publication.

Opt-in process-interruption recovery, not a power-loss or concurrent-writer
contract. Unknown/live descendants are refused. No persisted PID is signalled.
Pending training transactions remain owned exclusively by BatchJournal.
"""
import json
import os
import re
import uuid
from batch_recovery import digest, _json, _publish

VERSION = 'served_quiescent_v1'


class PreReadyRecovery:
    def __init__(self, journal, publish_committed=None):
        self.journal = journal
        self.active = journal.path('.batch_journal/pre_ready.json')
        self.publish_committed = publish_committed

    def _read(self):
        value = _json(self.active)
        if (value.get('version') != VERSION
                or not re.fullmatch(r'[0-9a-f]{32}', str(value.get('attempt')))
                or type(value.get('index')) is not int or value['index'] < 0
                or value.get('stage') not in ('prepared', 'launching', 'quiescent', 'archiving')):
            raise RuntimeError('Invalid pre-ready attempt record')
        marker = self.journal._marker()
        if (value.get('parent_generation') != value['index'] - 1
                or value.get('controls') != self.journal.controls
                or value.get('fixed_inputs') != self.journal.fixed_inputs()
                or value.get('before') != marker['inventory']
                and marker['generation'] < value['index']):
            raise RuntimeError('Pre-ready experiment/parent identity changed')
        if value['index'] not in (marker['generation'], marker['generation'] + 1):
            raise RuntimeError('Pre-ready attempt generation is not current')
        return value

    @staticmethod
    def _roots(index):
        return ['incoming_served_b%03d' % index, 'batch_%03d' % index,
                '.batch_journal/txn_%06d' % index]

    def start(self, index):
        if self.active.exists() or self.journal.pending.exists():
            raise RuntimeError('A pre-ready or ready attempt already exists')
        marker = self.journal._marker()
        if index != marker['generation'] + 1:
            raise RuntimeError('Pre-ready index is not next generation')
        self.journal._check_inventory(marker['inventory'])
        roots = self._roots(index)
        if any(self.journal.path(name).exists() for name in roots):
            raise RuntimeError('Unowned pre-ready artifacts retained; no automatic adoption')
        _publish(self.active, {'version': VERSION, 'attempt': uuid.uuid4().hex,
            'index': index, 'parent_generation': marker['generation'],
            'controls': self.journal.controls, 'fixed_inputs': self.journal.fixed_inputs(),
            'before': marker['inventory'], 'stage': 'prepared'})

    def launching(self, index, output, workers, games):
        value = self._read()
        root = self.journal.path(self._roots(index)[0])
        # Both unsharded output and the single configured shard belong to this
        # fresh reserved incoming root. Arbitrary nested directories do not.
        supplied = self.journal.path(os.path.relpath(output, self.journal.root).replace(os.sep, '/'))
        relative = supplied.relative_to(root).as_posix()
        if relative != '.' and not re.fullmatch(r'shard_\d{6,}', relative):
            raise RuntimeError('Unexpected served output path')
        if (value['index'] != index or value['stage'] != 'prepared'
                or type(workers) is not int or workers < 1
                or type(games) is not int or games < 1
                or root.exists()):
            raise RuntimeError('Invalid served pre-launch state')
        value.update(stage='launching', output=supplied.relative_to(self.journal.root).as_posix(),
                     expected_workers=min(workers, games), planned_games=games)
        # Publication precedes ProcessTree construction. Death in that small
        # interval conservatively refuses retry rather than guessing no spawn.
        _publish(self.active, value)

    def quiescent(self, process, returncode, supervisor_cleanup=None):
        value = self._read()
        observed = process.poll()
        if (value['stage'] != 'launching' or type(returncode) is not int
                or type(observed) is not int or observed != returncode):
            raise RuntimeError('Served root has not been joined')
        output = value['output']
        launch_path = self.journal.path(output + '/launch.json')
        phase_path = self.journal.path(output + '/phase.json')
        if supervisor_cleanup is not None:
            # The live owning ProcessTree itself must have produced this
            # receipt. Never certify a persisted/arbitrary PID or caller dict.
            from process_tree import ProcessTree
            if (type(process) is not ProcessTree
                    or process.termination_receipt != supervisor_cleanup
                    or supervisor_cleanup.get('pid') != process.pid
                    or supervisor_cleanup.get('root_returncode') != returncode
                    or (supervisor_cleanup.get('platform'),supervisor_cleanup.get('method'))
                        not in (('nt','taskkill_tree'),('posix','owned_process_group'))):
                raise RuntimeError('Owned process-tree termination receipt is unproved')
            evidence={name: digest(path) if path.is_file() else None
                for name,path in (('launch',launch_path),('phase',phase_path))}
            value.update(stage='quiescent', certificate={
                'method':'joined_owned_process_tree_termination',
                'root_returncode':returncode,'root_pid':process.pid,
                'termination':supervisor_cleanup,'evidence':evidence})
            _publish(self.active,value)
            return
        launch, phase = _json(launch_path), _json(phase_path)
        cleanup = phase.get('cleanup') or {}
        exits = cleanup.get('worker_exitcodes')
        expected = {str(w) for w in range(value['expected_workers'])}
        if (phase.get('finished') is not True or cleanup.get('alive') != []
                or not isinstance(exits, dict) or set(exits) != expected
                or any(type(code) is not int for code in exits.values())
                or type(cleanup.get('server_exitcode')) is not int
                or launch.get('games') != value['planned_games']
                or min(launch.get('workers', 0), launch.get('games', 0)) != value['expected_workers']):
            raise RuntimeError('Served descendant quiescence is unproved; retain and refuse')
        # The newly created output cannot contain an earlier attempt's phase.
        # The pinned runner joins these owned children before writing cleanup;
        # concrete exit codes and alive=[] are required even after a failure.
        value.update(stage='quiescent', certificate={
            'launch_sha256': digest(launch_path), 'phase_sha256': digest(phase_path),
            'root_returncode': returncode, 'root_pid': process.pid,
            'method': 'joined_root_and_served_owned_child_cleanup'})
        _publish(self.active, value)

    def _tree(self, relative):
        path = self.journal.path(relative)
        if not path.exists():
            return None
        if not path.is_dir():
            raise RuntimeError('Pre-ready owned root is not a directory')
        files, dirs = {}, []
        def visit(folder):
            dirs.append(folder.relative_to(path).as_posix())
            for child in sorted(folder.iterdir()):
                checked = self.journal.path(child.relative_to(self.journal.root).as_posix())
                if checked.is_dir():
                    visit(checked)
                elif checked.is_file():
                    files[checked.relative_to(path).as_posix()] = digest(checked)
                else:
                    raise RuntimeError('Unsupported pre-ready evidence entry')
        visit(path)
        return {'files': files, 'directories': dirs}

    def recover(self):
        if not self.active.exists():
            return
        value = self._read()
        marker = self.journal._marker()
        self.journal._check_inventory(marker['inventory'])
        if marker['generation'] == value['index']:
            if self.publish_committed is not None:
                self.publish_committed(self.journal, value)
            # Successful publication is authoritative. Preserve the attempt
            # metadata without relocating its committed training inputs.
            target = self.journal.path('.batch_journal/completed_' + value['attempt'] + '.json')
            if target.exists():
                raise RuntimeError('Completed attempt destination exists')
            os.replace(self.active, target)
            return
        if self.journal.pending.exists():
            # Never abandon or re-play published ready input. The existing
            # hash-checked rollback/RNG restoration path is authoritative.
            self.journal._record()
            return
        if value['stage'] == 'launching':
            raise RuntimeError('Interrupted play has no descendant quiescence proof; retained, no PID signalled')
        if value['stage'] == 'prepared':
            if any(self.journal.path(name).exists() for name in self._roots(value['index'])):
                raise RuntimeError('Prepared attempt unexpectedly has output; retained')
        if value['stage'] == 'quiescent':
            certificate = value.get('certificate', {})
            if certificate.get('method') == 'joined_owned_process_tree_termination':
                receipt=certificate.get('termination',{})
                if (type(certificate.get('root_pid')) is not int or certificate['root_pid']<=0
                        or receipt.get('pid')!=certificate['root_pid']
                        or type(certificate.get('root_returncode')) is not int
                        or receipt.get('root_returncode')!=certificate['root_returncode']
                        or (receipt.get('platform'),receipt.get('method'))
                            not in (('nt','taskkill_tree'),('posix','owned_process_group'))
                        or set(certificate.get('evidence',{}))!={'launch','phase'}):
                    raise RuntimeError('Invalid owned-tree quiescence certificate')
                for name,expected in certificate['evidence'].items():
                    path=self.journal.path(value['output']+'/'+name+'.json')
                    if (digest(path) if path.is_file() else None)!=expected:
                        raise RuntimeError('Served quiescence evidence changed')
            else:
                for name in ('launch', 'phase'):
                    if digest(self.journal.path(value['output'] + '/' + name + '.json')) != certificate.get(name + '_sha256'):
                        raise RuntimeError('Served quiescence evidence changed')
        if value['stage'] != 'archiving':
            archive = '.batch_journal/abandoned/' + value['attempt']
            if self.journal.path(archive).exists():
                raise RuntimeError('Abandoned attempt destination already exists')
            trees = {name: self._tree(name) for name in self._roots(value['index'])}
            # pending.json.tmp is an unpublished pointer, never a ready input.
            # Preserve it with preparation evidence if publication was killed.
            sidecar = self.journal.path('.batch_journal/pending.json.tmp')
            sidecar_sha = digest(sidecar) if sidecar.exists() else None
            value.update(stage='archiving', archive=archive, trees=trees,
                         pending_tmp_sha256=sidecar_sha)
            _publish(self.active, value)
        archive = '.batch_journal/abandoned/' + value['attempt']
        if value.get('archive') != archive or set(value.get('trees', {})) != set(self._roots(value['index'])):
            raise RuntimeError('Invalid abandonment plan')
        locations = []
        for name, tree in value['trees'].items():
            destination = archive + '/' + name.replace('.batch_journal/', 'journal/')
            original, moved = self._tree(name), self._tree(destination)
            if tree is None:
                if original is not None or moved is not None:
                    raise RuntimeError('Unexpected artifact after abandonment plan')
            elif (original, moved) not in ((tree, None), (None, tree)):
                raise RuntimeError('Abandonment inventory changed')
            if tree is not None and os.name == 'nt':
                # Some Windows hosts disable long-path support. Validate all
                # destinations before moving, so preserved files stay readable.
                for child in list(tree['files']) + tree['directories']:
                    full = self.journal.root / destination / ('' if child == '.' else child)
                    if len(str(full)) >= 248:
                        raise RuntimeError('Abandonment destination exceeds conservative Windows path limit')
            locations.append((name, destination, original is not None))
        source_tmp = self.journal.path('.batch_journal/pending.json.tmp')
        target_tmp = self.journal.path(archive + '/pending.json.tmp')
        sha = value.get('pending_tmp_sha256')
        observed = (digest(source_tmp) if source_tmp.exists() else None,
                    digest(target_tmp) if target_tmp.exists() else None)
        if (sha is None and observed != (None, None)
                or sha is not None and observed not in ((sha, None), (None, sha))):
            raise RuntimeError('Unpublished pending pointer changed')
        # Validate the entire move plan before moving anything. Rename only
        # these exact private workspace paths; preserve every byte and empty dir.
        for source, destination, move in locations:
            if move:
                target = self.journal.path(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(self.journal.path(source), target)
        if source_tmp.exists():
            target_tmp.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_tmp, target_tmp)
        target = self.journal.path(archive + '/attempt.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError('Abandoned attempt manifest already exists')
        os.replace(self.active, target)


def for_loop(journal, backend):
    mode = os.environ.get('WOTR_PRE_READY_RECOVERY', '')
    if not mode:
        return None
    if mode != VERSION or journal is None or backend != 'served_v1':
        raise RuntimeError('Pre-ready recovery requires served_quiescent_v1, served_v1 and batch journal')
    from committed_game_feed import configure, publish
    configure(journal)
    return PreReadyRecovery(journal, publish_committed=publish)
