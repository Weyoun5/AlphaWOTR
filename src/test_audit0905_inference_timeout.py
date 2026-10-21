"""Real response-unavailable and malformed-response controls; no game labels."""
import multiprocessing as mp
import os
import time
import unittest
from unittest.mock import patch

import numpy as np
from infer_server import InferClient


def peer(connection, control, mode):
    control.send('ready')
    request = connection.recv()
    control.send('received')
    if mode == 'malformed':
        connection.send((request[1] + 1, np.zeros(1)))
    else:
        control.recv()
        try:
            connection.send((request[1], np.zeros(1)))
        except (EOFError, OSError):
            pass
    connection.close()
    control.close()


class ResponseTimeoutControls(unittest.TestCase):
    def case(self, mode):
        context = mp.get_context('spawn')
        pipe, control = context.Pipe(), context.Pipe()
        process = context.Process(target=peer, args=(pipe[0], control[0], mode))
        process.start()
        pipe[0].close()
        control[0].close()
        client = InferClient(pipe[1], response_timeout=.2)
        try:
            self.assertTrue(control[1].poll(5))
            self.assertEqual(control[1].recv(), 'ready')
            started = time.monotonic()
            error = ValueError if mode == 'malformed' else TimeoutError
            with self.assertRaises(error):
                client.eval('fp', np.zeros(1, np.float32))
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 2)
            if mode != 'malformed':
                self.assertGreaterEqual(elapsed, .15)
            second = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                client.eval('fp', np.zeros(1))
            self.assertLess(time.monotonic() - second, .1)
            if mode != 'malformed':
                control[1].send('release')
            process.join(timeout=3)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
        finally:
            client.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            control[1].close()

    def test_absent_response_times_out_and_late_response_cannot_be_reused(self):
        self.case('no_reply')

    def test_wrong_request_id_invalidates_connection(self):
        self.case('malformed')

    def test_cleanup_error_preserves_original_timeout(self):
        class BrokenClosePipe:
            close_calls = 0

            def send(self, message):
                self.message = message

            def poll(self, seconds):
                return False

            def close(self):
                self.close_calls += 1
                raise OSError('literal cleanup failure')

        pipe = BrokenClosePipe()
        client = InferClient(pipe, response_timeout=.1)
        # This unit case isolates a close() error; real stalled I/O has separate coverage.
        with patch('infer_server._exchange', side_effect=TimeoutError('literal deadline')), \
                self.assertRaisesRegex(TimeoutError, 'eval request 1') as caught:
            client.eval('fp', np.zeros(1))
        if hasattr(caught.exception, 'add_note'):
            self.assertIn('literal cleanup failure', str(caught.exception.__notes__))
        client.close()
        self.assertEqual(pipe.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.eval('fp', np.zeros(1))

    def test_explicit_close_is_idempotent_and_refuses_future_requests(self):
        class CountingPipe:
            close_calls = 0

            def close(self):
                self.close_calls += 1

            def send(self, message):
                raise AssertionError('A closed client must never send')

        pipe = CountingPipe()
        client = InferClient(pipe)
        client.close()
        client.close()
        self.assertEqual(pipe.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.eval('fp', np.zeros(1))

    def test_invalid_timeouts_refused(self):
        for value in (0, -1, float('nan'), float('inf'), True, 'garbage'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InferClient(None, response_timeout=value)

    def test_environment_default_and_override(self):
        with patch.dict(os.environ, {'WOTR_INFER_TIMEOUT_SEC': '1.5'}):
            self.assertEqual(InferClient(None).response_timeout, 1.5)
            self.assertEqual(InferClient(None, response_timeout=2).response_timeout, 2)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(InferClient(None).response_timeout, 60)


if __name__ == '__main__':
    unittest.main(verbosity=2)
