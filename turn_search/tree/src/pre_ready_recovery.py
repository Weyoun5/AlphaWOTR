"""Preserve and retry proved-quiescent served attempts before ready publication.

Opt-in process-interruption recovery, not a power-loss or concurrent-writer
contract. Unknown/live descendants are refused. No persisted PID is signalled.
Pending training transactions remain owned exclusively by BatchJournal.

R2-SL-28 (USER: "broadly speaking, we need to prevent all softlocks"): an
attempt may no longer wedge in 'launching'. A joined runner root that never
wrote its spawn marker is certified 'joined_root_pre_spawn'; an interrupted
attempt whose runner lock is released (runner gone) and whose children sat in
the runner's kill-on-close job (or were never started) is certified by
recover() itself (runner_lifetime_guard), and so is one killed before its
runner took the lock once the root identity play_batch recorded (pid + OS
creation time) names no live process. Anything still unprovable is refused
with the exact operator escape, run_td_v33loop.py --abandon-pre-ready ATTEMPT,
which checks for live runner/worker processes before archiving by move.
"""
import json
import os
import re
import time
import uuid
from pathlib import Path
from batch_recovery import digest, _json, _publish
from runner_lifetime_guard import (PROTOCOL as RUNNER_PROTOCOL, LOCK_NAME, GUARD_NAME, SPAWN_MARKER,
                                   SCHEMA as GUARD_SCHEMA, JOB_KIND, probe_lock)

VERSION = 'served_quiescent_v1'
# Certificates whose evidence is the digests of launch.json, phase.json and the spawn marker.
_EVIDENCE_METHODS = ('joined_root_pre_spawn', 'runner_lock_released_pre_spawn', 'root_exited_pre_spawn',
                     'runner_lock_released_kill_on_close_job', 'operator_abandon_v1')
_PRE_SPAWN_METHODS = ('joined_root_pre_spawn', 'runner_lock_released_pre_spawn', 'root_exited_pre_spawn')


def _process_create_time(pid):
    """OS creation time of live process pid, or None when it cannot be read (psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:            # NoSuchProcess (already exited), AccessDenied
        return None


def _root_state(root):
    """'exited', 'alive' or 'unknown' for a recorded runner root {pid, create_time}.

    The pair names one process: a live process with that pid but another
    creation time is a later process that reused the pid. Nothing is signalled."""
    if (not isinstance(root, dict) or type(root.get('pid')) is not int or root['pid'] <= 0
            or type(root.get('create_time')) is not float):
        return 'unknown'
    try:
        import psutil
    except ImportError:
        return 'unknown'
    try:
        observed = psutil.Process(root['pid']).create_time()
    except psutil.NoSuchProcess:
        return 'exited'
    except psutil.Error:
        return 'unknown'
    # Within a second counts as the same process (a refusal, never a false
    # 'exited'): a pid is not reused that soon after its process was created.
    return 'alive' if abs(observed - root['create_time']) < 1.0 else 'exited'


def _live_processes():
    """Rows {pid, ppid, create_time, cmdline} of every visible live process (psutil)."""
    import psutil
    rows = []
    for process in psutil.process_iter(['pid', 'ppid', 'create_time', 'cmdline']):
        rows.append(dict(process.info))
    return rows


def _pid_exists(pid):
    import psutil
    return psutil.pid_exists(pid)


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
                or value.get('stage') not in ('prepared', 'launching', 'unproved', 'quiescent', 'archiving')):
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
        # Publication precedes ProcessTree construction. 'launching' is left by
        # quiescent() (joined root), by recover() from the runner's lifetime
        # evidence (R2-SL-28: runner.lock released + spawn marker absent or
        # kill-on-close job; or root_started()'s root identity, once that exact
        # process is gone, when no runner.lock was taken), or by the operator
        # escape abandon(). It is NOT only "death in a small interval": an
        # outside kill of a batch that runs for hours also leaves it, so never
        # guess "no spawn".
        _publish(self.active, value)

    def root_started(self, process):
        """play_batch, inside its cleanup try, right after ProcessTree(): record the root's identity.

        R2-SL-28: {pid, OS creation time} of the owning ProcessTree's own child
        (the runner). If loop and runner are killed together before the runner
        takes runner.lock, recover() can later tell, without signalling
        anything, that this exact process is gone. Only a real ProcessTree is
        recorded; an unreadable creation time is recorded as None (no proof)."""
        value = self._read()
        if value['stage'] != 'launching' or 'root' in value:
            raise RuntimeError('Served root identity recorded out of order')
        from process_tree import ProcessTree
        if type(process) is not ProcessTree or type(process.pid) is not int or process.pid <= 0:
            return
        value['root'] = {'pid': process.pid, 'create_time': _process_create_time(process.pid)}
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
        if not self.journal.path(output + '/' + SPAWN_MARKER).exists() and self._declares_spawn_marker(launch_path):
            # R2-SL-28: the joined root is the runner, and the runner writes its
            # spawn marker (fsynced) before it starts any served child. Its exit
            # without that marker proves no served child ever existed: a typo
            # in an operational knob, an import error, or any other pre-spawn
            # failure is certified here instead of wedging 'launching'.
            value.update(stage='quiescent', certificate={
                'method': 'joined_root_pre_spawn', 'root_returncode': returncode,
                'root_pid': process.pid, 'evidence': self._evidence(output)})
            _publish(self.active, value)
            return
        if not launch_path.is_file() or not phase_path.is_file():
            raise RuntimeError('Served descendant quiescence is unproved (the spawned runner left no %s); '
                               'retain and refuse' % ('launch.json' if not launch_path.is_file() else 'phase.json'))
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

    def unproved(self, reason):
        """play_batch: quiescent() refused. Record why, instead of a bare 'launching'.

        recover() treats 'unproved' exactly like 'launching' (runner lifetime
        evidence, else the operator escape); quiescent() never accepts it."""
        value = self._read()
        if value['stage'] not in ('launching', 'unproved'):
            return
        text = ('%s: %s' % (type(reason).__name__, reason)) if isinstance(reason, BaseException) else str(reason)
        value.update(stage='unproved', unproved={'reason': text[:4000], 'at': time.time()})
        _publish(self.active, value)

    def _evidence(self, output):
        evidence = {}
        for name, filename in (('launch', 'launch.json'), ('phase', 'phase.json'), ('spawning', SPAWN_MARKER)):
            path = self.journal.path(output + '/' + filename)
            evidence[name] = digest(path) if path.is_file() else None
        return evidence

    @staticmethod
    def _declares_spawn_marker(launch_path):
        """True when no served child of this output can predate its spawn marker.

        Every runner version writes launch.json in _prepare_run, before run()
        can start a child, so without launch.json nothing was spawned. A runner
        that wrote launch.json must also declare the marker protocol; for an
        older runner the absence of a marker proves nothing."""
        if not launch_path.exists():
            return True
        try:
            launch = _json(launch_path)
        except (OSError, ValueError):
            return False
        return isinstance(launch, dict) and launch.get('lifetime_guard') == RUNNER_PROTOCOL

    @staticmethod
    def _guard_state(path):
        if not path.is_file():
            return None
        try:
            state = _json(path)
        except (OSError, ValueError):
            return None
        return state if isinstance(state, dict) else None

    def _interrupted_proof(self, value):
        """(certificate, None) from the dead runner's own lifetime evidence, else (None, reason)."""
        output = value['output']
        lock = probe_lock(self.journal.path(output + '/' + LOCK_NAME))
        if lock == 'absent':
            reason = ('the runner left no %s (it never started, died before taking it, or predates it)'
                      % LOCK_NAME)
            # R2-SL-28: loop and runner killed together before the runner took
            # its lock. The pre-spawn rule applies once the recorded root (that
            # exact process) is gone: the runner takes runner.lock before it
            # writes its spawn marker, and the marker precedes any served child.
            root = value.get('root')
            state = _root_state(root)
            if state == 'alive':
                return None, '%s, and its root process (pid %d) is still alive' % (reason, root['pid'])
            if state == 'unknown':
                return None, '%s, and its root process identity is %s' % (
                    reason, 'unrecorded' if root is None else 'unreadable')
            launch_path = self.journal.path(output + '/launch.json')
            if (self.journal.path(output + '/' + SPAWN_MARKER).exists()
                    or not self._declares_spawn_marker(launch_path)):
                return None, ('%s; its root process exited, but the output does not prove that no served '
                              'child was started' % reason)
            return {'method': 'root_exited_pre_spawn', 'lock': lock, 'root': dict(root),
                    'evidence': self._evidence(output)}, None
        if lock == 'held':
            return None, 'a live runner holds %s; let it finish or stop it, then start the loop again' % LOCK_NAME
        launch_path = self.journal.path(output + '/launch.json')
        if not launch_path.is_file() or not self._declares_spawn_marker(launch_path):
            return None, 'launch.json does not declare the runner lifetime protocol'
        evidence = self._evidence(output)
        guard_path = self.journal.path(output + '/' + GUARD_NAME)
        state = self._guard_state(guard_path)
        if evidence['spawning'] is None:
            # The runner is gone (lock released) and never reached its spawn marker.
            method = 'runner_lock_released_pre_spawn'
        else:
            job = state.get('job') if state is not None and isinstance(state.get('job'), dict) else {}
            if not (state is not None and state.get('schema') == GUARD_SCHEMA
                    and type(state.get('runner_pid')) is int and state['runner_pid'] > 0
                    and job.get('kind') == JOB_KIND and job.get('active') is True):
                return None, ('served children were started without an active kill-on-close job (%s)'
                              % (job.get('reason') or 'no valid ' + GUARD_NAME))
            # The kernel killed the job's children when the runner's handle
            # closed. Defence in depth: no live child of that runner may remain.
            survivors = self._runner_children(state)
            if survivors:
                return None, ('processes started by the dead runner (pid %d) are still alive: %s'
                              % (state['runner_pid'], ', '.join(survivors)))
            method = 'runner_lock_released_kill_on_close_job'
        return {'method': method, 'lock': lock, 'evidence': evidence,
                'guard_sha256': digest(guard_path) if guard_path.is_file() else None}, None

    @staticmethod
    def _runner_children(state):
        try:
            rows = _live_processes()
        except ImportError:
            return ['(psutil unavailable: cannot confirm that no child of the runner is alive)']
        runner = state['runner_pid']
        started = state.get('started_at') if isinstance(state.get('started_at'), (int, float)) else 0.0
        return ['pid %s' % row.get('pid') for row in rows
                if row.get('ppid') == runner and isinstance(row.get('create_time'), (int, float))
                and row['create_time'] >= started - 1.0]

    def _live_blockers(self, value):
        """Reasons the attempt's runner or one of its workers may still be alive."""
        output = value['output']
        blockers = []
        if probe_lock(self.journal.path(output + '/' + LOCK_NAME)) == 'held':
            blockers.append('%s is held by a live runner' % LOCK_NAME)
        state = self._guard_state(self.journal.path(output + '/' + GUARD_NAME)) or {}
        runner = state.get('runner_pid') if type(state.get('runner_pid')) is int else None
        started = state.get('started_at') if isinstance(state.get('started_at'), (int, float)) else 0.0
        base = self._roots(value['index'])[0]
        target = os.path.normcase(str(self.journal.root / base))
        try:
            rows = _live_processes()
        except ImportError:
            raise RuntimeError('psutil is required to check for live runner and worker processes') from None
        for row in rows:
            pid = row.get('pid')
            if pid == os.getpid():
                continue
            argv = [str(part) for part in (row.get('cmdline') or [])]
            text = ' '.join(argv)
            named = False
            for part in argv:
                if base.lower() in part.lower():
                    candidate = part.rsplit('=', 1)[-1]
                    try:
                        candidate = str(Path(candidate).resolve())
                    except (OSError, ValueError):
                        pass
                    candidate = os.path.normcase(candidate)
                    named = named or candidate == target or candidate.startswith(target + os.sep)
            if named:
                blockers.append('pid %s names the attempt output (%s)' % (pid, text[:300]))
            elif (runner is not None and row.get('ppid') == runner
                    and isinstance(row.get('create_time'), (int, float)) and row['create_time'] >= started - 1.0):
                blockers.append('pid %s is a child of the runner (pid %d)' % (pid, runner))
            elif '--multiprocessing-fork' in argv:
                if os.name == 'nt':
                    match = re.search(r'parent_pid=(\d+)', text)
                    parent = int(match.group(1)) if match else None
                    orphan = parent is None or parent == runner or not _pid_exists(parent)
                else:
                    orphan = row.get('ppid') == 1
                if orphan:
                    blockers.append('pid %s is an orphaned multiprocessing child (%s)' % (pid, text[:300]))
        return blockers

    def _escape(self, value, reason):
        return ('Interrupted play has no descendant quiescence proof (%s); retained, no PID signalled. '
                'Once no loop, runner or worker process of this attempt is alive, run '
                '"python src/run_td_v33loop.py --abandon-pre-ready %s" in the loop\'s environment: it checks '
                'for live processes and archives the attempt by move to .batch_journal/abandoned/%s'
                % (reason, value['attempt'], value['attempt']))

    def abandon(self, attempt):
        """Operator escape (R2-SL-28) for an interrupted attempt recover() cannot prove quiescent.

        Called by run_td_v33loop.py --abandon-pre-ready ATTEMPT, which first
        refuses while a loop process owns the workspace. Refuses while the
        attempt's runner lock is held or a live process names its output, is a
        child of its runner, or is an orphaned multiprocessing child. Never
        signals a process; archives by move through recover()."""
        if not self.active.exists():
            raise RuntimeError('No pre-ready attempt exists; nothing to abandon')
        value = self._read()
        if value['attempt'] != attempt:
            raise RuntimeError('The pre-ready attempt is %s, not %s' % (value['attempt'], attempt))
        marker = self.journal._marker()
        if (value['stage'] not in ('launching', 'unproved') or marker['generation'] == value['index']
                or self.journal.pending.exists()):
            raise RuntimeError('Attempt %s (stage %s) is recovered by an ordinary loop start; not abandoned'
                               % (attempt, value['stage']))
        blockers = self._live_blockers(value)
        if blockers:
            raise RuntimeError('Refusing to abandon pre-ready attempt %s: %s' % (attempt, '; '.join(blockers)))
        value.update(stage='quiescent', certificate={
            'method': 'operator_abandon_v1', 'checked_at': time.time(), 'previous_stage': value['stage'],
            'checks': ['runner.lock not held', 'no live process names the output',
                       'no live child of the runner', 'no orphaned multiprocessing child'],
            'evidence': self._evidence(value['output'])})
        _publish(self.active, value)
        self.recover()

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
        if value['stage'] in ('launching', 'unproved'):
            # R2-SL-28: certify from the dead runner's own lifetime evidence, or
            # refuse naming the operator escape; never a wedge with no exit.
            certificate, reason = self._interrupted_proof(value)
            if certificate is None:
                raise RuntimeError(self._escape(value, reason))
            value.update(stage='quiescent', certificate=certificate)
            _publish(self.active, value)
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
            elif certificate.get('method') in _EVIDENCE_METHODS:
                evidence = certificate.get('evidence')
                if (not isinstance(evidence, dict) or set(evidence) != {'launch', 'phase', 'spawning'}
                        or certificate['method'] in _PRE_SPAWN_METHODS and evidence['spawning'] is not None):
                    raise RuntimeError('Invalid interrupted-attempt quiescence certificate')
                if self._evidence(value['output']) != evidence:
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
