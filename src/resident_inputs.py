"""Exact, graph-local host mirrors and minimal dirty-prefix upload plans.

Arrays hold the literal capture input, including padding. No model arithmetic,
graph shape, reduction order, or ownership is changed. Callers must apply every
returned upload before replay and invalidate the graph if an upload fails.
"""
import numpy as np


def same_bits(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return np.array_equal(np.ascontiguousarray(left).view(np.uint8),
                          np.ascontiguousarray(right).view(np.uint8))


class ResidentInputs:
    def __init__(self, batch, bucket, dim, adim):
        self.arrays = {
            'x': np.zeros((batch, dim), np.float32),
            'enc': np.zeros((batch, bucket, adim), np.float32),
            'region': np.zeros((batch, bucket), np.int64),
            'count': np.zeros(batch, np.int64),
        }
        self.arrays['enc'][:, :, 33] = -1
        self.upload_bytes = 0
        self.full_upload_bytes = 0
        self.updates = 0
        self.reused_action_rows = 0

    def update(self, items):
        # GraphPV performs complete validation before entering this function.
        uploads = []
        arrays = self.arrays
        for i, (vector, actions, regions) in enumerate(items):
            count = len(actions)
            old_count = int(arrays['count'][i])
            if not same_bits(arrays['x'][i], vector):
                arrays['x'][i] = vector
                uploads.append(('x', i, arrays['x'][i]))
            if same_bits(arrays['enc'][i, :count], actions):
                self.reused_action_rows += count
                changed_actions = False
            else:
                arrays['enc'][i, :count] = actions
                changed_actions = True
            if count < old_count:
                arrays['enc'][i, count:old_count] = 0
                arrays['enc'][i, count:old_count, 33] = -1
                changed_actions = True
            if changed_actions:
                extent = max(count, old_count)
                index = (i, slice(0, extent))
                uploads.append(('enc', index, arrays['enc'][index]))
            # Region requests may use any integer width; the old staging path
            # converted them to int64. Match that conversion before comparing.
            regions = np.asarray(regions, dtype=np.int64)
            changed_regions = not same_bits(arrays['region'][i, :count], regions)
            if changed_regions:
                arrays['region'][i, :count] = regions
            if count < old_count:
                arrays['region'][i, count:old_count] = 0
                changed_regions = True
            if changed_regions:
                extent = max(count, old_count)
                index = (i, slice(0, extent))
                uploads.append(('region', index, arrays['region'][index]))
            if count != old_count:
                arrays['count'][i] = count
                index = slice(i, i + 1)
                uploads.append(('count', index, arrays['count'][index]))
        self.upload_bytes += sum(view.nbytes for _, _, view in uploads)
        self.full_upload_bytes += sum(value.nbytes for value in arrays.values())
        self.updates += 1
        return uploads

    def stats(self):
        return dict(upload_bytes=self.upload_bytes,
                    full_upload_bytes=self.full_upload_bytes,
                    reused_action_rows=self.reused_action_rows,
                    host_bytes=sum(v.nbytes for v in self.arrays.values()),
                    updates=self.updates)
