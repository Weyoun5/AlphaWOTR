"""Scheduler-assigned GPU visibility must survive inference-server startup.

Literal CUDA/ROCm masks model Flux per-task assignments. The server body is
replaced only to inspect its launch environment, without allocating a GPU.
"""
import os
import sys
import types
import unittest
from unittest.mock import patch

import run_sp_served


class ServerAffinity(unittest.TestCase):
    def launch_environment(self, assigned):
        seen = []
        stub = types.ModuleType('infer_server')
        stub.server_main = lambda *args: seen.append((dict(os.environ), args))
        with patch.dict(os.environ, assigned, clear=True), \
                patch.dict(sys.modules, {'infer_server': stub}):
            run_sp_served._server_entry(['pipe'], {'fp': 'fp.pt'}, 'base')
            self.assertEqual(dict(os.environ), assigned)
        self.assertEqual(seen[0][1], (['pipe'], {'fp': 'fp.pt'}, 'base'))
        return seen[0][0]

    def test_flux_task_gpu_two_stays_visible_as_assigned(self):
        self.assertEqual(self.launch_environment({'CUDA_VISIBLE_DEVICES': '2'})
                         ['CUDA_VISIBLE_DEVICES'], '2')

    def test_ordered_subset_is_not_replaced_by_all_devices(self):
        self.assertEqual(self.launch_environment({'CUDA_VISIBLE_DEVICES': '3,1'})
                         ['CUDA_VISIBLE_DEVICES'], '3,1')

    def test_rocm_and_cuda_masks_remain_intact_together(self):
        masks = {'CUDA_VISIBLE_DEVICES': '0', 'ROCR_VISIBLE_DEVICES': '3',
                 'HIP_VISIBLE_DEVICES': '0'}
        self.assertEqual(self.launch_environment(masks), masks)

    def test_absent_gpu_mask_remains_absent(self):
        self.assertEqual(self.launch_environment({}), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
