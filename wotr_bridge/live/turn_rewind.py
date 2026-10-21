"""Verified, same-session Java + engine checkpoints with rollback and release.

The private native deck graph remains in client-local storage. This coordinator saves
the existing engine/driver state in compressed local files, publishes only public provenance, and
never adopts a board as a substitute for restoring the corresponding engine.
The native rendered feed stays append-only; abandoned lines are explicitly
retired only after a verified restore. No checkpoint survives a process restart.
"""
import hashlib
import json
import os
import io
import pickle
import re
import sys
import zlib
from pathlib import Path
import random
import time
import types
import uuid

SCHEMA = 'wotr_turn_checkpoint_v1'
TIMEOUT = 20.0
_DRIVER_LIVE = ('_feed_seq', '_acks_seen', 'enforce_human', '_raw_recovery_receipts',
                '_response_identity', '_ack_operations', '_ack_watermark_ready',
                'session_nonce')

# These are collaborators and live transport/operator identities, not gameplay.
_SERVICES = ('driver', 'rand', 'cards', 'chat_out', 'script', 'fp_net', 'sh_net',
             'own_net', 'own_search', 'scenario')
_LIVE_SERVER = frozenset(_SERVICES + (
    'client_dir', 'ai_side', 'human_side', 'nick', 'sims', 'notes', '_turn_rewind',
    '_cycle_failure', '_cycle_err_sig', '_cycle_err_n', '_cycles', '_feed_n', '_feed_t',
    '_reset_requested', '_rewind_requested', '_rewind_target', '_resync_requested', '_manual_hold',
    '_enforce_human_boot', '_raw_control_history', '_raw_control_baseline',
    '_reset_chat_seen19', '_skip_chat_seen', '_mirror_last', '_mirror_skips',
    '_skip_requested', '_skip_target', '_skip_message',
    '_mirror_warned', '_recovery_status', '_recovery_attempt', '_automatic_recovery',
    '_recovery_ran_this_cycle', '_ask_nonce_seq',
    '_idle_recovery_watch', '_idle_recovery_attempts',
))


def _live_key(key):
    return (key in _LIVE_SERVER or key.endswith('_epoch')
            or key.startswith('_cycle_')
            or key in ('_last_cycle_failure', '_live_cycle_failure'))


class StateSnapshot:
    """Compressed local checkpoint with references to stable services only.

    Models/native controller handles are persistent references, never copied or
    serialized. Commit ledgers are serialized by value: native Undo mutates
    their line lists, and a former pre-state can become the live state. The
    native client checkpoint and this file are scoped to the current session.
    """
    def __init__(self, server, identifier=None):
        driver = server.driver
        services = [server] + [getattr(server, name, None) for name in _SERVICES]
        services += [driver.feed, driver.sidecar, driver.view, driver.seg, driver.rctl]
        services += list(driver.hand_models.values())
        services += [getattr(driver, '_rr', None)]
        registry = _state(server).references
        for value in services:
            if value is not None:
                registry[id(value)] = value
        self.references = registry
        self.targets = [driver]
        roots = [{k: v for k, v in vars(driver).items()
                  if k not in _DRIVER_LIVE}]
        for value in [driver.view, driver.seg, driver.rctl, server.rand, server.cards,
                      server.chat_out] + list(driver.hand_models.values()):
            if value is not None and value not in self.targets:
                self.targets.append(value)
                roots.append(vars(value))
        own = getattr(server, 'own_search', None)
        search = ({name: getattr(own, name) for name in ('seed', 'index')}
                  if own is not None else None)
        numpy = sys.modules.get('numpy')
        numpy_state = numpy.random.get_state() if numpy is not None else None
        payload = (roots, {k: v for k, v in vars(server).items() if not _live_key(k)},
                   random.getstate(), numpy_state, search)
        stream = io.BytesIO()
        class Writer(pickle.Pickler):
            def persistent_id(self, value):
                if isinstance(value, (types.FunctionType, types.MethodType, types.ModuleType)):
                    registry[id(value)] = value
                return ('reference', id(value)) if id(value) in registry else None
        Writer(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(payload)
        data = zlib.compress(stream.getvalue())
        directory = Path(server.client_dir) / 'ai_turn_checkpoints' / str(driver.session_nonce)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('engine-' + (identifier or uuid.uuid4().hex) + '.bin')
        self.digest = hashlib.sha256(data).hexdigest()
        _atomic_bytes(self.path, data)

    def restore(self, server):
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.digest:
            raise ValueError('engine checkpoint checksum mismatch')
        registry = self.references
        class Reader(pickle.Unpickler):
            def persistent_load(self, identity):
                if identity[0] != 'reference' or identity[1] not in registry:
                    raise ValueError('checkpoint service reference unavailable')
                return registry[identity[1]]
        fields, server_fields, random_state, numpy_state, search = Reader(
            io.BytesIO(zlib.decompress(data))).load()
        driver_live = {key: getattr(server.driver, key) for key in
                       _DRIVER_LIVE
                       if hasattr(server.driver, key)}
        for target, values in zip(self.targets, fields):
            vars(target).clear()
            vars(target).update(values)
        vars(server.driver).update(driver_live)
        for key in list(vars(server)):
            if not _live_key(key):
                delattr(server, key)
        vars(server).update(server_fields)
        random.setstate(random_state)
        if numpy_state is not None:
            sys.modules['numpy'].random.set_state(numpy_state)
        if search is not None:
            for key, value in search.items():
                setattr(server.own_search, key, value)


class Checkpoint:
    def __init__(self):
        self.phase = 'idle'
        self.token = self.transaction = self.session = None
        self.candidate = self.rollback = None
        self.proof = None
        self.started = 0.0
        self.turn = self.stage = None
        self.note = 'No verified checkpoint is available yet.'
        self.checkpoints = {}
        self.entries = []
        self.references = {}
        self.turn_starts = {}
        self.last_signature = None
        self.current_id = self.restored_id = None
        self.latest_id = None
        self.event_cursor = 0
        self.attempted_signature = None
        self.failed_phase = None
        self.last_receipt = None
        self.abandoned_feed_end = None
        self.rollback_feed_seq = None
        self.current_feed_end = None
        self.publication = None
        self.review_after_restore = False
        self.review = False


def _state(server):
    state = getattr(server, '_turn_rewind', None)
    if state is None:
        state = server._turn_rewind = Checkpoint()
    return state


def review_control_waiting(server):
    """Peek new rendered operator chat while gameplay routing is paused.

    Native raw action history can omit chat. Leave the feed cursor untouched:
    the normal driver will consume the command and all preceding gestures.
    """
    from . import chat, live_parser
    driver = server.driver
    driver.feed.poll()
    for line in driver.feed.lines[driver._feed_seq:]:
        record = live_parser.parse(line)
        if record.get('op') != 'chat' or record.get('nick') == driver.nick:
            continue
        control = (chat.classify_command(record.get('body', '')) or (None,))[0]
        if control in ('resume', 'skip', 'rewind', 'reset', 'resync'):
            return True
    return False


def _atomic_bytes(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def _read(server, filename):
    try:
        data = json.loads((Path(server.client_dir) / filename).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def supported(server):
    capability = _read(server, 'ai_turn_checkpoint_capability.json')
    return (capability.get('schema') == SCHEMA
            and capability.get('session') == getattr(server.driver, 'session_nonce', None)
            and isinstance(capability.get('session'), int) and capability['session'] > 0)


def _proof(server):
    from .client_protocol import parse_seq_header, feed_position_delta
    path = Path(server.driver.view.dump_path)
    before = path.stat().st_mtime_ns
    raw = path.read_text(encoding='utf-8')
    if before != path.stat().st_mtime_ns:
        raise ValueError('board export changed while reading')
    header_line, body = raw.split('\n', 1)
    header = parse_seq_header(header_line)
    if (header is None or header.get('session') != server.driver.session_nonce
            or feed_position_delta(header, len(server.driver.feed.lines)) != 0
            or getattr(server.driver.feed, '_buf', b'')):
        raise ValueError('board export does not match the complete feed/session')
    return dict(session=header['session'], feed_paragraphs=header['feed_paragraphs'],
                raw_lines=header['raw_lines'],
                board_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest())


def _events(lines):
    result = []
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        if re.search(r'\bdraws?\b|\bhand\b|\bpassword\b|\bchosen\s+(?:combat\s+)?card'
                     r'|\bchoos\w*.*\bcard\b|\b(?:claim|commit)\w*.*\b(?:card|deck)\b', line, re.I):
            line = '[Private card/deck event: details hidden]'
        result.append({'public_text': line})
    return result


def _persist(server):
    state = _state(server)
    entries = list(state.entries)
    tail = _events(server.driver.feed.lines[state.event_cursor:])
    if tail:
        entries.append(dict(checkpoint_id=None, turn=int(server.driver.es.turn),
                            label='Events after the last verified boundary (not restorable yet)',
                            restorable=False, events=tail))
    data = dict(schema='rewind_timeline_v1', session=str(server.driver.session_nonce),
                current_checkpoint_id=state.current_id,
                turn_start_checkpoint_id=state.turn_starts.get(int(server.driver.es.turn)),
                restored_checkpoint_id=state.restored_id,
                turn=int(server.driver.es.turn), entries=entries)
    _atomic_bytes(Path(server.client_dir) / 'ai_rewind_timeline.json',
                  (json.dumps(data, sort_keys=True) + '\n').encode('utf-8'))
    manifest = dict(schema=SCHEMA, checkpoint_id=state.token, transaction=state.transaction,
                    phase=state.phase, proof=state.proof, process=os.getpid(), restart_recoverable=False)
    _atomic_bytes(Path(server.client_dir) / 'ai_turn_checkpoint_manifest.json',
                  (json.dumps(manifest, sort_keys=True) + '\n').encode('utf-8'))
    state.publication = (len(server.driver.feed.lines), state.phase, state.current_id,
                         state.restored_id, len(state.entries))


def _command(state, operation):
    state.started = time.monotonic()
    state.last_receipt = None
    return ['!CALL turn_%s %s %s' % (operation, state.token, state.transaction)]


def status(server):
    state = _state(server)
    available = supported(server)
    if state.current_id is not None:
        checkpoint = state.checkpoints.get(state.current_id, {})
        marker = (int(server.driver.es.turn), server.driver.phase,
                  server.driver.n_commits_total, len(server.driver.commits))
        if (state.current_feed_end != len(server.driver.feed.lines)
                or checkpoint.get('engine_marker') != marker):
            state.current_id = None
    publication = (len(server.driver.feed.lines), state.phase, state.current_id,
                   state.restored_id, len(state.entries))
    if available and publication != state.publication:
        try:
            _persist(server)
        except OSError:
            state.note = 'Rewind history could not be published; existing checkpoint state is retained.'
    return dict(available=bool(state.checkpoints and available), supported=available,
                turn_available=bool(available and state.turn_starts.get(int(server.driver.es.turn))),
                request_epoch=getattr(server, '_rewind_epoch', 0),
                session=str(getattr(server.driver, 'session_nonce', '')),
                phase=state.phase, turn=state.turn, stage=state.stage,
                note=(state.note if available else
                      'Turn rewind requires the updated client and a new paired session. '
                      'This running game has no native checkpoint; use Retry/Resync.'),
                current_checkpoint_id=state.current_id, restored_checkpoint_id=state.restored_id,
                restart_recoverable=False, review=state.review)


def maybe_capture(server):
    """Save each distinct synchronized stable boundary, including turn opening.

    Items queued by the caller remain in driver._out_queue; only passive engine
    continuations are eligible, never an already issued physical/random request.
    """
    from . import turn_bookkeeping
    state = _state(server)
    driver = server.driver
    die_evidence = getattr(driver, '_native_die_evidence', {})
    if (state.phase != 'idle' or state.review or not supported(server)
            or getattr(server, '_open_step', None) is not None
            or server.rand.step_open or server.rand.pending() is not None
            or driver.echo_pending() or driver.human_physical_work_pending()
            or driver.pending_buf is not None or driver.deferred is not None
            or driver.chat_deferred is not None
            or driver._feed_seq != len(driver.feed.lines)
            or getattr(server, '_cycle_failure', None)
            or any(die_evidence.get(kind) for kind in
                   ('commands', 'records', 'kill_commands', 'kill_records'))):
        return []
    try:
        proof = _proof(server)
    except (OSError, ValueError):
        return []
    signature = (int(driver.es.turn), driver.phase, len(driver.commits),
                 proof['feed_paragraphs'], proof['raw_lines'], proof['board_sha256'])
    if signature in (state.last_signature, state.attempted_signature):
        return []
    state.current_id = None
    _persist(server)
    opening = _opening_stage(server)
    # Fresh matching board/feed proofs below supersede the render-age timer.
    # Otherwise automatic turn draws outrun that timer and the pre-draw
    # opening is never saved, even though every recovery already landed.
    # Actual outstanding echoes, gestures, randomness and repairs still hold.
    if driver.quiescence_holds(allow_engine_pending=True, allow_recent_render=True):
        return []
    if not driver.view.refresh_dump():
        return []
    rows, staging = driver.census_rows()
    if rows or staging:
        return []
    try:
        if _proof(server) != proof:
            return []
    except (OSError, ValueError):
        return []
    return begin_capture(server, opening=opening, signature=signature)


def _opening_stage(server):
    """Prove a pre-draw opening, never infer one from a midturn attachment."""
    from . import turn_bookkeeping
    driver = server.driver
    turn = int(driver.es.turn)
    if (turn in _state(server).turn_starts or driver.phase != 'draws'
            or driver.turn_draws or getattr(driver, '_turn_start_draws_open', False)):
        return False
    recovery = getattr(getattr(server, '_turn_bookkeeping', None), 'recovery', None)
    if recovery is not None and recovery.turn == turn and recovery.complete:
        stage = 'turn opening after dice recovery, before turn draws'
    elif (turn == 1 and not driver.commits and driver.n_commits_total == 0
          and server.scenario is None and not getattr(driver, '_last_resync_adoption', None)
          and any(re.match(r'^<game> .+ has started a new War of the Ring game\.', line)
                  for line in driver.feed.lines)):
        stage = 'initial turn before turn draws'
    else:
        return False
    # This boundary is a synchronized recovered board, independent of whether
    # the next automatic draw can start. A draw-side error must not remove the
    # user's route back to this position; maybe_capture checks board parity.
    return stage


def begin_capture(server, *, opening=False, signature=None):
    state = _state(server)
    if state.phase != 'idle' or not supported(server):
        return []
    identifier = uuid.uuid4().hex
    try:
        proof = _proof(server)
        candidate = StateSnapshot(server, identifier)
    except Exception:
        state.note = 'Checkpoint postponed: engine/native state could not be preserved.'
        return []
    state.token = identifier
    state.transaction = uuid.uuid4().hex
    state.session = proof['session']
    state.proof = proof
    state.turn = int(server.driver.es.turn)
    state.stage = (opening if isinstance(opening, str) else
                   'turn opening after dice recovery; draws as captured' if opening else
                   'verified %s boundary after the listed events' % server.driver.phase)
    state.candidate = dict(snapshot=candidate, proof=proof, turn=state.turn, stage=state.stage,
                           opening=opening, signature=signature, feed_end=len(server.driver.feed.lines),
                           turn_starts=dict(state.turn_starts),
                           engine_marker=(state.turn, server.driver.phase,
                                          server.driver.n_commits_total, len(server.driver.commits)))
    state.attempted_signature = signature
    state.phase = 'capturing'
    state.note = 'Saving a matching native client and engine checkpoint.'
    return _command(state, 'checkpoint')


def request(server, checkpoint_id=None, session=None, *, review=True):
    state = _state(server)
    if session is not None and str(session) != str(getattr(server.driver, 'session_nonce', None)):
        state.note = 'Rewind selection belongs to another session; refresh the history window.'
        return []
    if checkpoint_id is not None and session is None:
        state.note = 'Manual rewind requires the current session identity.'
        return []
    if state.phase == 'failed':
        # A late native watchdog/release receipt may already settle this
        # transaction. Consume it before sending anything that overwrites it.
        settled = poll(server)
        if state.phase != 'failed':
            return settled or []
        state.phase = 'querying'
        state.note = 'Checking the native outcome before retrying the recovery handshake.'
        return _command(state, 'status')
    if state.phase != 'idle':
        state.note = 'A checkpoint transaction is already pending.'
        return []
    chosen = (state.turn_starts.get(int(server.driver.es.turn))
              if checkpoint_id is None else checkpoint_id)
    checkpoint = state.checkpoints.get(chosen)
    if checkpoint is None or not supported(server):
        state.note = ('No verified checkpoint matches that selection in this session. '
                      'Retry/Resync retains the current game.')
        return []
    try:
        rollback = StateSnapshot(server)
    except Exception:
        state.note = 'Rewind could not preserve the current engine; the game was retained.'
        return []
    state.rollback = rollback
    state.review_after_restore = review
    state.review = False
    state.rollback_feed_seq = server.driver._feed_seq
    state.token = chosen
    state.transaction = uuid.uuid4().hex
    state.session = checkpoint['proof']['session']
    state.proof = checkpoint['proof']
    state.turn, state.stage = checkpoint['turn'], checkpoint['stage']
    state.phase = 'restoring'
    state.note = 'Restoring the selected native checkpoint; waiting for exact board verification.'
    return _command(state, 'restore')


def recover(server):
    """Request the most recent verified boundary; caller owns retry limits."""
    state = _state(server)
    if not state.checkpoints:
        return request(server, review=False)
    selected = state.latest_id
    if selected is None:
        return request(server, review=False)
    return request(server, checkpoint_id=selected,
                   session=str(getattr(server.driver, 'session_nonce', '')), review=False)


def _abort(server, note):
    state = _state(server)
    if state.rollback is not None:
        state.rollback.restore(server)
        server.driver._feed_seq = state.rollback_feed_seq
    state.phase = 'aborting'
    state.note = note + ' Restoring the pre-rewind client position.'
    return _command(state, 'abort')


def _capture_failed(state):
    state.candidate = None
    state.phase = 'idle'
    state.note = 'New checkpoint was not verified; previous verified checkpoints remain available.'


def _finish_abort(server):
    state = _state(server)
    if state.rollback is not None:
        state.rollback.restore(server)
        server.driver._feed_seq = state.rollback_feed_seq
    state.rollback = None
    state.phase = 'idle'
    state.current_id = None
    state.note = 'Rewind cancelled; the pre-rewind client and engine were restored. Retry/Resync is available.'
    _persist(server)
    return []


def poll(server):
    """None permits normal play; [] pauses ordinary ticks during the handshake."""
    state = _state(server)
    if state.phase == 'idle':
        return None
    receipt = _read(server, 'ai_turn_checkpoint.json')
    matches = (receipt.get('schema') == SCHEMA and receipt.get('token') == state.transaction
               and receipt.get('checkpoint_id') == state.token
               and receipt.get('session') == state.session)
    phase = receipt.get('phase') if matches else None
    if phase == 'aborted' and state.phase != 'capturing':
        return _finish_abort(server)
    if phase == 'rollback_failed':
        state.failed_phase, state.phase = 'aborting', 'failed'
        state.note = 'Native rollback could not be verified. Rewind retries that rollback; original checkpoint is retained.'
        return []
    if state.phase in ('failed', 'querying'):
        if phase == 'released':
            state.phase = 'releasing'
            state.started = time.monotonic()
        elif state.phase == 'querying' and phase in ('accepted', 'restored'):
            if state.failed_phase == 'releasing' and phase == 'accepted':
                state.phase = 'releasing'
                return _command(state, 'release')
            return _abort(server, 'Unconfirmed restore cancelled; original engine retained.')
        elif state.phase == 'failed':
            return []
    if time.monotonic() - state.started > TIMEOUT:
        if state.phase == 'capturing':
            _capture_failed(state)
            return []
        if state.phase not in ('aborting', 'releasing', 'querying'):
            return _abort(server, 'Rewind timed out; its engine change was rolled back.')
        if state.phase != 'querying':
            state.failed_phase = state.phase
        state.phase = 'failed'
        state.note = 'Native recovery acknowledgment is unconfirmed. Rewind retries the same handshake.'
        return []
    if not matches:
        return []
    if phase == 'failed':
        if state.phase == 'capturing':
            _capture_failed(state)
            return []
        if state.phase == 'aborting':
            # The previous failure receipt is not a new abort result. Keep the
            # original deadline; Java's independent watchdog also restores it.
            return []
        if state.phase in ('releasing', 'querying'):
            return []
        return _abort(server, 'Native rewind refused or failed; original engine retained.')
    expected = {'capturing': 'captured', 'restoring': 'restored',
                'accepting': 'accepted', 'releasing': 'released', 'aborting': 'aborted',
                'querying': 'status_pending'}[state.phase]
    if phase != expected:
        return []
    if state.phase in ('capturing', 'restoring', 'accepting'):
        try:
            if state.phase != 'capturing':
                server.driver.feed.poll()
            proof = _proof(server)
            for key in ('session', 'board_sha256', 'raw_lines'):
                if receipt.get(key) != state.proof[key] or proof[key] != state.proof[key]:
                    raise ValueError('native checkpoint fingerprint mismatch')
            if receipt.get('feed_paragraphs') != proof['feed_paragraphs']:
                return []
            if state.phase == 'capturing' and proof != state.proof:
                raise ValueError('native feed changed during capture')
        except (OSError, ValueError) as error:
            if state.phase == 'capturing':
                _capture_failed(state)
                return []
            if 'fingerprint mismatch' not in str(error):
                return []
            return _abort(server, 'Rewind verification failed; original engine retained.')
    if state.phase == 'capturing':
        checkpoint = state.candidate
        state.checkpoints[state.token] = checkpoint
        state.candidate = None
        state.last_signature = checkpoint['signature']
        state.current_id = state.token
        state.latest_id = state.token
        state.current_feed_end = checkpoint['feed_end']
        if checkpoint['opening']:
            state.turn_starts[checkpoint['turn']] = state.token
        checkpoint['turn_starts'] = dict(state.turn_starts)
        state.entries.append(dict(checkpoint_id=state.token, turn=state.turn,
                                  label=state.stage, restorable=True,
                                  events=_events(server.driver.feed.lines[state.event_cursor:checkpoint['feed_end']])))
        state.event_cursor = checkpoint['feed_end']
        state.phase = 'idle'
        state.note = 'Checkpoint verified after the listed public events.'
        _persist(server)
        return []
    if state.phase == 'restoring':
        try:
            state.checkpoints[state.token]['snapshot'].restore(server)
            state.abandoned_feed_end = len(server.driver.feed.lines)
            server.driver._feed_seq = state.abandoned_feed_end
            if server.driver.sidecar is not None:
                server.driver.sidecar.poll()
            state.phase = 'accepting'
            state.note = 'Native board and engine restored; awaiting client acceptance.'
            return _command(state, 'accept')
        except Exception:
            return _abort(server, 'Engine rewind failed; original engine retained.')
    if state.phase == 'accepting':
        state.phase = 'releasing'
        return _command(state, 'release')
    if state.phase == 'releasing':
        from .cycle_recovery import rewind_completed
        rewind_completed(server)
        state.rollback = None
        state.phase = 'idle'
        state.current_id = state.restored_id = state.token
        state.latest_id = state.token
        state.current_feed_end = state.abandoned_feed_end
        # Each retained branch owns its own turn-opening ancestry. Do not let
        # an abandoned future turn's opening become this branch's fallback.
        state.turn_starts = dict(state.checkpoints[state.token]['turn_starts'])
        state.last_signature = state.checkpoints[state.token]['signature']
        state.note = 'Restored exactly the selected checkpoint: turn %s, %s.' % (state.turn, state.stage)
        state.review = state.review_after_restore
        if state.review:
            state.note += ' Play is paused here; press RESUME or say resume when ready.'
        state.entries.append(dict(checkpoint_id=None, restored_checkpoint_id=state.token, turn=state.turn,
                                  label='Restored this checkpoint; intervening events were rewound',
                                  restorable=False,
                                  events=_events(server.driver.feed.lines[state.event_cursor:state.abandoned_feed_end])))
        state.event_cursor = state.abandoned_feed_end
        _persist(server)
        from . import chat
        return [server._say('prompt', chat._register(state.note))]
    return []
