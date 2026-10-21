"""One synchronous deadline for duplex multiprocessing request/reply I/O.

No helper thread owns the connection. Stream sockets retain multiprocessing's
length-prefixed wire format; Windows named pipes retain message boundaries.
The caller exclusively owns the connection and invalidates it on failure.
"""
import math
import multiprocessing.connection as connections
from multiprocessing.reduction import ForkingPickler
import os
import select
import socket
import struct
import time


def _note(original, label, error):
    if hasattr(original, 'add_note'):
        original.add_note('%s: %r' % (label, error))


def _cancel_pending(conn, ov, original):
    """Cancel while retaining the actual I/O buffer until Windows releases it.

    Normal cancellation completes promptly. If the OS cannot cancel or report
    completion, safe release cannot also promise a bounded return: this cleanup
    retains local ownership and waits. There is no background thread or global
    buffer quarantine, and cleanup reporting errors do not replace `original`.
    """
    import _winapi
    try:
        ov.cancel()
    except Exception as error:
        _note(original, 'Inference I/O cancellation failed', error)
        # This pipe is exclusively owned by the synchronous client. If the
        # per-operation wrapper fails, cancel all I/O on that same handle before
        # attempting to close it. CloseHandle alone does not establish that an
        # in-flight overlapped operation has released its buffer.
        try:
            import ctypes
            cancel = ctypes.WinDLL('kernel32', use_last_error=True).CancelIoEx
            cancel.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
            cancel.restype = ctypes.c_int
            if not cancel(conn.fileno(), None) and ctypes.get_last_error() != 1168:
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception as fallback_error:
            _note(original, 'Inference handle cancellation failed', fallback_error)
    reported = set()
    while True:
        try:
            _, error = ov.GetOverlappedResult(False)
            if error != 996:  # ERROR_IO_INCOMPLETE is not exported by _winapi.
                return
        except Exception as error:
            label = ('Inference I/O reaping failed', repr(error))
            if label not in reported:
                reported.add(label)
                _note(original, label[0], error)
            try:
                if _winapi.WaitForMultipleObjects([ov.event], False, 0) == _winapi.WAIT_OBJECT_0:
                    return
            except Exception as wait_error:
                label = ('Inference I/O cleanup wait failed', repr(wait_error))
                if label not in reported:
                    reported.add(label)
                    _note(original, label[0], wait_error)
        time.sleep(.001)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Inference transport deadline expired')
    return remaining


def _stream_exchange(conn, payload, deadline):
    # A duplex multiprocessing Connection owns a socket, on both POSIX and
    # Windows. Borrow it only while exchanging this frame; detach on every exit
    # so that InferClient remains the single owner responsible for close().
    sock = socket.socket(fileno=conn.fileno())
    previous_blocking = os.get_blocking(sock.fileno()) if os.name != 'nt' else True
    try:
        sock.setblocking(False)

        def wait(writing):
            remaining = _remaining(deadline)
            ready = select.select([] if writing else [sock],
                                  [sock] if writing else [], [], remaining)
            if not (ready[1] if writing else ready[0]):
                raise TimeoutError('Inference transport deadline expired')

        def send(data):
            view = memoryview(data)
            while view:
                wait(True)
                try:
                    count = sock.send(view)
                except (BlockingIOError, InterruptedError):
                    continue
                if count == 0:
                    raise EOFError('Inference connection closed during send')
                view = view[count:]

        def receive(size):
            chunks = []
            left = size
            while left:
                wait(False)
                try:
                    part = sock.recv(min(left, 65536))
                except (BlockingIOError, InterruptedError):
                    continue
                if not part:
                    raise EOFError('Inference connection closed during response')
                chunks.append(part)
                left -= len(part)
            return b''.join(chunks)

        size = len(payload)
        header = (struct.pack('!i', size) if size <= 0x7fffffff else
                  struct.pack('!iQ', -1, size))
        if size <= 16384:
            # Match multiprocessing's small-frame coalescing: avoid an isolated
            # tiny header interacting with TCP Nagle/delayed-ACK behavior.
            send(header + bytes(payload))
        else:
            send(header)
            send(payload)
        size, = struct.unpack('!i', receive(4))
        if size == -1:
            size, = struct.unpack('!Q', receive(8))
        elif size < 0:
            raise ValueError('Invalid inference response frame length')
        return receive(size)
    finally:
        try:
            sock.setblocking(previous_blocking)
        finally:
            sock.detach()


def _pipe_exchange(conn, payload, deadline):
    import _winapi

    handle = conn.fileno()

    def operation(data, writing):
        _remaining(deadline)
        ov, error = (_winapi.WriteFile(handle, data, overlapped=True) if writing
                     else _winapi.ReadFile(handle, data, overlapped=True))
        try:
            if error == _winapi.ERROR_IO_PENDING:
                # Round up to avoid an early timeout from millisecond truncation.
                milliseconds = min(0xfffffffe, math.ceil(_remaining(deadline) * 1000))
                status = _winapi.WaitForMultipleObjects([ov.event], False, milliseconds)
                if status == _winapi.WAIT_TIMEOUT:
                    raise TimeoutError('Inference transport deadline expired')
                if status != _winapi.WAIT_OBJECT_0:
                    raise OSError('Unexpected inference pipe wait result: %s' % status)
            count, error = ov.GetOverlappedResult(False)
        except BaseException as original:
            # Cancel and reap before releasing the buffer or the pipe handle.
            # Closing a handle alone is not a substitute for owning pending I/O.
            _cancel_pending(conn, ov, original)
            # Release the completed operation's buffer export before retaining
            # this frame in the propagated exception traceback.
            ov = None
            raise
        if error not in (0, _winapi.ERROR_MORE_DATA):
            raise OSError(error, 'Inference pipe operation failed')
        return count, error, None if writing else ov.getbuffer()

    written, error, _ = operation(payload, True)
    if error != 0 or written != len(payload):
        raise OSError('Incomplete inference request write')
    chunks = []
    while True:
        _, error, data = operation(65536, False)
        chunks.append(data)
        if error == 0:
            return b''.join(chunks)


def exchange(conn, message, deadline):
    """Exchange one object using the connection's normal multiprocessing format.

    The deadline covers blocking transport, including partial frames. Pickling
    and unpickling run in this thread, with deadline checks around that work.
    Windows cancellation must complete before its pending buffer can be freed;
    a hard OS cancellation/completion failure has no bounded cleanup guarantee.
    Only real duplex multiprocessing connections are supported.
    """
    _remaining(deadline)
    payload = ForkingPickler.dumps(message)
    _remaining(deadline)
    pipe_type = getattr(connections, 'PipeConnection', ())
    if pipe_type and isinstance(conn, pipe_type):
        response = _pipe_exchange(conn, payload, deadline)
    elif isinstance(conn, connections.Connection):
        response = _stream_exchange(conn, payload, deadline)
    else:
        raise TypeError('Inference requires a duplex multiprocessing Connection')
    _remaining(deadline)
    reply = ForkingPickler.loads(response)
    _remaining(deadline)
    return reply
