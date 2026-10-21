"""Real FileBridge regressions on disposable files; no live game or UI."""
from pathlib import Path
import io
import builtins
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / 'wotr_bridge'), str(ROOT / 'src')]
from live import transport


class ResponseIdentity(unittest.TestCase):
    def bridge(self, **kwargs):
        directory = tempfile.TemporaryDirectory(prefix='response_identity_')
        self.addCleanup(directory.cleanup)
        bridge = transport.FileBridge(directory.name, 'response_identity', poll=.001, **kwargs)
        self.addCleanup(bridge.release_lock)
        return bridge

    def request(self, bridge, req=7, session=5, side='LIVE_FREE_PEOPLES'):
        transport.atomic_write(bridge.req, '%s\n#SEQ req=%d session=%d\npublic probe' %
                               (side, req, session))

    def reply(self, bridge):
        return Path(bridge.resp).read_text(encoding='utf8').splitlines()

    def test_success_is_tagged_once_and_handler_runs_once(self):
        bridge = self.bridge()
        self.request(bridge)
        calls = []
        bridge.serve(lambda request: calls.append(request) or (['OP_ONE', '!CALL d6 1'], 'waiting'),
                     max_cycles=6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'OP_ONE', '!CALL d6 1'])
        self.assertEqual(calls[0].log_lines[0], '#SEQ req=7 session=5')

    def test_empty_paused_and_error_all_complete_identified_request(self):
        for mode in ('empty', 'paused', 'error', 'invalid_side'):
            with self.subTest(mode=mode):
                bridge = self.bridge()
                self.request(bridge, side='INVALID' if mode == 'invalid_side' else 'LIVE_SHADOW')
                if mode == 'paused':
                    transport.atomic_write(bridge.control_path, json.dumps({'paused': True}))
                calls = []
                def handler(request):
                    calls.append(request)
                    if mode == 'error':
                        raise ValueError('fixture handler failure')
                    return []
                bridge.serve(handler, max_cycles=1)
                self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5'])
                self.assertEqual(len(calls), 0 if mode in ('paused', 'invalid_side') else 1)

    def test_legacy_requests_keep_plain_reply_and_req_only_echo(self):
        for body, expected in [('public probe', ['OP']),
                               ('#SEQ req=9\npublic probe', ['#SEQ req=9', 'OP'])]:
            with self.subTest(body=body):
                bridge = self.bridge()
                transport.atomic_write(bridge.req, 'AUTO\n' + body)
                bridge.serve(lambda request: ['OP'], max_cycles=1)
                self.assertEqual(self.reply(bridge), expected)

    def test_old_handler_response_keeps_old_session_when_new_request_arrives(self):
        bridge = self.bridge()
        self.request(bridge, 11, 100)
        entered, release = threading.Event(), threading.Event()
        calls = []
        def handler(request):
            calls.append(request)
            entered.set()
            if not release.wait(3):
                raise TimeoutError('fixture release missing')
            return ['OLD_OPERATION']
        worker = threading.Thread(target=lambda: bridge.serve(handler, max_cycles=1))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.request(bridge, 12, 200)
        finally:
            release.set()
            worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=11 session=100', 'OLD_OPERATION'])
        self.assertIn('req=12 session=200', Path(bridge.req).read_text())

    def test_failed_publication_retains_exact_payload_without_replaying_handler(self):
        bridge = self.bridge()
        self.request(bridge)
        real_write = transport.atomic_write
        attempts, calls = [], []
        def write(path, text):
            if path == bridge.resp:
                attempts.append(text)
                if len(attempts) <= 3:
                    raise PermissionError('fixture locked RESP')
            return real_write(path, text)
        def handler(request):
            calls.append(request)
            return ['COMMITTED_OPERATION', '!CALL d6 1']
        with patch.object(transport, 'atomic_write', side_effect=write):
            bridge.serve(handler, max_cycles=5)
        expected = '#SEQ req=7 session=5\nCOMMITTED_OPERATION\n!CALL d6 1'
        self.assertEqual(len(calls), 1)
        self.assertEqual(attempts, [expected] * 4)
        self.assertEqual(self.reply(bridge), expected.splitlines())

    def test_repeated_publication_failure_blocks_next_handler_and_is_visible(self):
        bridge = self.bridge()
        self.request(bridge)
        real_write = transport.atomic_write
        calls = []
        def handler(request):
            calls.append(request)
            self.request(bridge, 8, 5)
            return ['FIRST_OPERATION']
        def write(path, text):
            if path == bridge.resp:
                raise PermissionError('fixture retained RESP blocked')
            return real_write(path, text)
        with patch.object(transport, 'atomic_write', side_effect=write):
            bridge.serve(handler, max_cycles=4)
        self.assertEqual(len(calls), 1)
        self.assertIn('req=8', Path(bridge.req).read_text())
        status = json.loads(Path(bridge.status_path).read_text())
        self.assertEqual(status['state'], 'error')
        self.assertIn('retained RESP blocked', status['note'])
        bridge.serve(handler, max_cycles=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'FIRST_OPERATION'])

    def test_status_failure_after_publish_cannot_overwrite_response(self):
        bridge = self.bridge()
        self.request(bridge)
        real_status = bridge.write_status
        def status(**kwargs):
            if kwargs.get('waiting_on') == 'after publish':
                raise OSError('fixture panel status failure')
            return real_status(**kwargs)
        with patch.object(bridge, 'write_status', side_effect=status):
            bridge.serve(lambda request: (['COMMITTED_OPERATION'], 'after publish'), max_cycles=1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'COMMITTED_OPERATION'])

    def test_idle_status_or_control_error_cannot_replace_unconsumed_response(self):
        for failure in ('status', 'control'):
            with self.subTest(failure=failure):
                bridge = self.bridge()
                self.request(bridge)
                real_status, real_control = bridge.write_status, bridge.read_control
                def status(**kwargs):
                    if Path(bridge.resp).exists() and kwargs == dict(connected=True, state='idle'):
                        raise OSError('fixture idle status failure')
                    return real_status(**kwargs)
                def control():
                    if Path(bridge.resp).exists() and not Path(bridge.req).exists():
                        raise OSError('fixture idle control failure')
                    return real_control()
                with patch.object(bridge, 'write_status', side_effect=status if failure == 'status' else real_status), \
                     patch.object(bridge, 'read_control', side_effect=control if failure == 'control' else real_control):
                    bridge.serve(lambda request: ['COMMITTED_OPERATION'], max_cycles=2)
                self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'COMMITTED_OPERATION'])

    def test_claimed_request_read_failure_retries_before_new_request(self):
        bridge = self.bridge()
        self.request(bridge)
        real_open = builtins.open
        reads, calls = [], []
        def open_file(path, *args, **kwargs):
            if str(path) == bridge.req + '.processing':
                reads.append(str(path))
                if len(reads) == 1:
                    self.request(bridge, 8, 6)
                    raise PermissionError('fixture claimed request locked')
            return real_open(path, *args, **kwargs)
        def handler(request):
            calls.append(request)
            return ['FIRST_OPERATION']
        with patch.object(builtins, 'open', side_effect=open_file):
            bridge.serve(handler, max_cycles=2)
        self.assertEqual(len(reads), 2)
        self.assertEqual(len(calls), 1)
        self.assertIn('req=7 session=5', calls[0].raw)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'FIRST_OPERATION'])
        self.assertIn('req=8 session=6', Path(bridge.req).read_text())
        bridge.serve(lambda request: calls.append(request) or ['SECOND_OPERATION'], max_cycles=1)
        self.assertEqual(len(calls), 2)
        self.assertIn('req=8 session=6', calls[1].raw)
        self.assertEqual(self.reply(bridge), ['#SEQ req=8 session=6', 'SECOND_OPERATION'])

    def test_repeated_claim_read_failure_is_visible_and_recoverable(self):
        bridge = self.bridge()
        self.request(bridge)
        real_open = builtins.open
        calls = []
        def open_file(path, *args, **kwargs):
            if str(path) == bridge.req + '.processing':
                raise OSError('fixture unread request retained')
            return real_open(path, *args, **kwargs)
        with patch.object(builtins, 'open', side_effect=open_file):
            bridge.serve(lambda request: calls.append(request) or ['OP'], max_cycles=3)
        self.assertEqual(calls, [])
        self.assertFalse(Path(bridge.resp).exists())
        status = json.loads(Path(bridge.status_path).read_text())
        self.assertEqual(status['state'], 'error')
        self.assertIn('unread request retained', status['note'])
        bridge.serve(lambda request: calls.append(request) or ['OP'], max_cycles=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'OP'])

    def test_processing_disposal_failure_never_replays_read_request(self):
        bridge = self.bridge()
        self.request(bridge)
        real_remove = transport.os.remove
        calls = []
        def remove(path):
            if str(path) == bridge.req + '.processing':
                raise PermissionError('fixture processing deletion locked')
            return real_remove(path)
        with patch.object(transport.os, 'remove', side_effect=remove):
            bridge.serve(lambda request: calls.append(request) or ['OP'], max_cycles=6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.reply(bridge), ['#SEQ req=7 session=5', 'OP'])
        self.assertTrue(Path(bridge.req + '.processing').exists())
        self.assertFalse(Path(bridge.req).exists())


if __name__ == '__main__':
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ResponseIdentity))
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
