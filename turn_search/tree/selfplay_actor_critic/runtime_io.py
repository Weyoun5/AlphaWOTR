"""Atomic control files and nonfatal status publishing; no background process."""
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import warnings
import zlib

from .contracts import CONTRACT, ContractError
from .environment import PythonEnvironment


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        for delay in (.01, .05, .1, .25, .5, .5, None):
            try:
                os.replace(name, path)
                return
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def atomic_bytes(path, data):
    _atomic_write(path, lambda stream: stream.write(data))


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))


def publish_status(path, value):
    try:
        write_json(path, value)
        return True
    except OSError as error:
        warnings.warn(f"Status publication failed; control checkpoints are separate: {error}")
        return False


SESSION_MAGIC = b"WOTR_SESSION_ZLIB_V1\n"
SESSION_CHUNK_BYTES = 1024 * 1024


class _CompressedPickleWriter:
    """Accept pickle buffers without materializing an uncompressed checkpoint."""
    def __init__(self, stream):
        self.stream = stream
        self.compressor = zlib.compressobj(level=1)
        self.digest = hashlib.sha256()

    def _emit(self, compressed):
        if compressed:
            self.digest.update(compressed)
            view = memoryview(compressed)
            for start in range(0, len(view), SESSION_CHUNK_BYTES):
                self.stream.write(view[start:start + SESSION_CHUNK_BYTES])

    def write(self, data):
        view = memoryview(data).cast("B")
        for start in range(0, len(view), SESSION_CHUNK_BYTES):
            self._emit(self.compressor.compress(view[start:start + SESSION_CHUNK_BYTES]))
        return len(view)

    def finish(self):
        self._emit(self.compressor.flush())
        return self.digest.digest()


class _DecompressedPickleReader(io.RawIOBase):
    """Bound intermediate decompression memory while pickle rebuilds objects."""
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.decompressor = zlib.decompressobj()
        self.pending = b""
        self.finished = False

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.finished:
            return 0
        target = memoryview(buffer).cast("B")
        if not target:
            return 0
        while True:
            compressed = self.pending or self.stream.read(SESSION_CHUNK_BYTES)
            if not compressed:
                raise ContractError("Session checkpoint modified or incomplete: truncated compression stream")
            output = self.decompressor.decompress(compressed, min(len(target), SESSION_CHUNK_BYTES))
            self.pending = self.decompressor.unconsumed_tail
            if self.decompressor.eof:
                if self.decompressor.unused_data or self.stream.read(1):
                    raise ContractError("Session checkpoint modified: trailing compressed data")
                self.finished = True
            if output:
                target[:len(output)] = output
                return len(output)
            if self.finished:
                return 0


def save_session(path, environments, rollouts, traces):
    # Trusted local pickle, never an interchange format for downloaded files.
    # Modules cannot be pickled; all actual game/controller/RNG data is retained.
    states = [{k: v for k, v in vars(e).items() if k not in ("ai", "gs", "gl")} for e in environments]
    payload = dict(schema="actor_critic_session_v2", contract_id=CONTRACT.identity,
                   environments=states, rollouts=rollouts, traces=traces)

    def write(stream):
        # Hash and payload share ONE atomic envelope. Only compressed chunks
        # reach disk; the original live checkpoint remains until replacement.
        stream.write(SESSION_MAGIC + bytes(32))
        compressed = _CompressedPickleWriter(stream)
        pickle.dump(payload, compressed, protocol=5)
        digest = compressed.finish()
        stream.seek(len(SESSION_MAGIC))
        stream.write(digest)

    _atomic_write(path, write)


def _verify_stream(stream, expected):
    digest = hashlib.sha256()
    while chunk := stream.read(SESSION_CHUNK_BYTES):
        digest.update(chunk)
    if len(expected) != 32 or digest.digest() != expected:
        raise ContractError("Session checkpoint modified or incomplete")


def load_session(ai, path):
    # Verify the complete on-disk payload BEFORE unpickling. Both formats
    # stream through bounded buffers; legacy raw sessions remain resumable.
    with Path(path).open("rb") as stream:
        compressed = stream.read(len(SESSION_MAGIC)) == SESSION_MAGIC
        stream.seek(len(SESSION_MAGIC) if compressed else 0)
        expected = stream.read(32)
        payload_offset = stream.tell()
        _verify_stream(stream, expected)
        stream.seek(payload_offset)
        try:
            if compressed:
                with io.BufferedReader(_DecompressedPickleReader(stream), buffer_size=SESSION_CHUNK_BYTES) as reader:
                    payload = pickle.load(reader)
                    if reader.read(1):
                        raise ContractError("Session checkpoint modified: trailing pickle data")
            else:
                payload = pickle.load(stream)
        except (zlib.error, pickle.UnpicklingError, EOFError) as error:
            raise ContractError("Session checkpoint modified or incomplete") from error
    if payload["schema"] != "actor_critic_session_v2" or payload["contract_id"] != CONTRACT.identity:
        raise ContractError("Unsupported session checkpoint")
    environments = []
    for state in payload["environments"]:
        env = PythonEnvironment.__new__(PythonEnvironment)
        vars(env).update(state)
        env.ai, env.gs, env.gl = ai, ai.gs, ai.game_logic
        environments.append(env)
    return environments, payload["rollouts"], payload["traces"]


class RunLock:
    """Process-lifetime OS lock; automatically released even after a crash."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another controller owns this run directory")
        return self

    def __exit__(self, *args):
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()
