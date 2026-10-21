import json
from pathlib import Path
import tempfile
import time
import unittest
from response_training_status import observe


class ResponseStatus(unittest.TestCase):
    def test_training_completion_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(observe(root))
            folder = root / 'response_policy/generation_000000'
            attempt = folder / 'fp_attempt_000'
            attempt.mkdir(parents=True)
            marker = root / '.batch_journal/committed.json'
            marker.parent.mkdir()
            marker.write_text(json.dumps({'generation': 0}))
            progress = attempt / 'progress.json'
            progress.write_text(json.dumps({'active_epoch': 1, 'rows_seen': 128}))
            status = {'status': 'training', 'active_owner': 'fp', 'updated_at': time.time(),
                      'command': ['python', '--output', str(attempt)]}
            path = folder / 'status.json'
            path.write_text(json.dumps(status))
            result = observe(root)
            self.assertEqual((result['owner'], result['rows_seen']), ('fp', 128))
            self.assertFalse(result['needs_attention'])
            status['updated_at'] -= 120
            path.write_text(json.dumps(status))
            self.assertTrue(observe(root)['needs_attention'])
            status.update(status='error', error='training failed')
            path.write_text(json.dumps(status))
            self.assertIn('training failed', observe(root)['detail'])
            status['status'] = 'complete'
            path.write_text(json.dumps(status))
            self.assertIsNone(observe(root))

    def test_bad_status_and_escaped_progress_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'response_policy/generation_000000'
            folder.mkdir(parents=True)
            marker = root / '.batch_journal/committed.json'
            marker.parent.mkdir()
            marker.write_text('{"generation": 0}')
            status = folder / 'status.json'
            status.write_text('{broken')
            self.assertTrue(observe(root)['needs_attention'])
            status.write_text(json.dumps({'status': 'training', 'command': ['--output', str(root)]}))
            self.assertIn('escapes', observe(root)['error'])


if __name__ == '__main__':
    unittest.main()
