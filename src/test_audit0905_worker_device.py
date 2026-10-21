"""Worker device selection must precede loading and never silently fall back."""
import os
import queue
import unittest
from unittest.mock import patch
import torch
import ai
import run_td_v33sp as producer


class WorkerDeviceTests(unittest.TestCase):
    def run_worker(self, **kw):
        messages = queue.Queue()
        producer._worker(0, 0, 4, 'unused', 1, 'fp', 'sh', messages, **kw)
        return messages.get_nowait()

    def test_unavailable_requested_gpu_fails_before_model_load(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5'), \
             patch.object(ai, 'DEVICE', torch.device('cpu')), \
             patch.object(torch.cuda, 'is_available', return_value=False), \
             patch.object(producer, '_load_net') as loader:
            result = self.run_worker(local_device='cuda')
            self.assertEqual(result[0], 'e')
            self.assertIn('no assigned CUDA device', result[2])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            loader.assert_not_called()

    def test_gpu_selected_before_load_and_assigned_mask_preserved(self):
        def observe_load(*args, **kw):
            self.assertEqual(ai.DEVICE.type, 'cuda')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            raise RuntimeError('stop after observing device')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5'), \
             patch.object(ai, 'DEVICE', torch.device('cpu')), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(producer, '_load_net', side_effect=observe_load) as loader:
            result = self.run_worker(local_device='cuda')
            self.assertIn('stop after observing device', result[2])
            loader.assert_called_once()

    def test_available_runtime_with_zero_assigned_devices_refuses(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5'), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=0), \
             patch.object(producer, '_load_net') as loader:
            result = self.run_worker(local_device='cuda')
            self.assertIn('no assigned CUDA device', result[2])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            loader.assert_not_called()

    def test_default_worker_stays_cpu(self):
        def observe_load(*args, **kw):
            self.assertEqual(ai.DEVICE.type, 'cpu')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')
            raise RuntimeError('default CPU observed')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5'), \
             patch.object(ai, 'DEVICE', torch.device('cuda')), \
             patch.object(producer, '_load_net', side_effect=observe_load):
            result = self.run_worker()
            self.assertIn('default CPU observed', result[2])

    def test_invalid_device_reports_error_without_loading(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='5'), \
             patch.object(producer, '_load_net') as loader:
            result = self.run_worker(local_device='typo')
            self.assertIn('Unsupported local worker device', result[2])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
