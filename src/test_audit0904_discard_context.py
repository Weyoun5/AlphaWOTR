"""A turn-start discard cannot observe later Hunt allocation or action rolls.

The hand is an explicitly authored midgame observation. The literal log
records establish ordering; neither a legal menu nor an executor supplies
the expected empty pre-allocation dice/Hunt context.
"""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ai
import rendered_extract as rx
import rendered_replay as rr

CID = 'FREE_PEOPLES_Character_Athelas'


def fixture():
    lines = [
        '%<replay> Turn 2',
        '%Player draws Free Character Card.',
        '%Player kills Athelas/Andúril in Free Peoples Hand.',
        '%Opponent allocates an eye.',
        '%Opponent rolls E E M M M M .',
        '%Opponent adds 2 eyes to the hunt.',
        '%Player rolls P C C C .',
        '%Player moves Free Action Die (P) from FP Dice to Current Action.',
        '%Player draws Free Character Card.',
        '%Player kills Free Action Die (P) in Current Action.',
        '%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.',
    ]
    with tempfile.TemporaryDirectory(prefix='discard-observation-') as folder:
        path = Path(folder) / 'literal.rendered.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        parsed = rr.parse_game(str(path))
    batches = parsed['batches']
    rx._annotate_remaining_pools(batches, rx._roll_pools(batches), {2: 1})
    return batches


class DiscardContextTests(unittest.TestCase):
    def test_record_before_hunt_allocation_has_no_future_used_eye(self):
        batches = fixture()
        record = next(r for b in batches for r in b['records'] if r['line'] == 3)
        self.assertEqual(record['_rx_pools'], {
            'FREE_PEOPLES': {'unused': [], 'used': []},
            'SHADOW': {'unused': [], 'used': []}})

    def test_later_record_preserves_roll_and_spent_event_die(self):
        batches = fixture()
        record = next(r for b in batches for r in b['records'] if r['line'] == 9)
        self.assertEqual(record['_rx_pools']['FREE_PEOPLES'],
                         {'unused': ['C', 'C', 'C'], 'used': ['P']})
        self.assertEqual(record['_rx_pools']['SHADOW']['unused'], ['M'] * 4)

    def test_emitted_turn_start_discard_has_no_future_hunt_or_dice(self):
        batches = fixture()
        ledger = rx._HandLedger()
        fillers = [c for c in ai._get_card_idx_map()
                   if c.startswith('FREE_PEOPLES_') and c != CID][:6]
        ledger._fp = [CID] + fillers
        ledger._record(2)
        ledger.discard_events = [('fp', 3, CID)]
        captured = []
        encode = ai.encode_state

        def observe(state, side):
            captured.append({
                'hunt_count': len(state.hunt_box),
                'allocated': state.eyes_allocated,
                'faces': {s.name: [(d.current.name if d.current else None, d.used)
                                  for d in state.dice[s]] for s in state.dice}})
            return encode(state, side)

        stats = defaultdict(Counter)
        with patch.object(rx, '_LEDGER', ledger), patch.object(rx, '_TABLE', None), \
                patch.object(rx, '_RX_RINGS_CUR', None), \
                patch.object(rx, '_RX_LINE_CUR', None), \
                patch.object(rx, '_RX_EYEALLOC', {2: 1}), \
                patch.object(rx, '_RX_EYEROLL', {2: 2}), \
                patch.object(ai, 'encode_state', side_effect=observe):
            samples = rx._emit_discards(batches, stats)
        self.assertEqual(len(samples), 1, dict(stats))
        self.assertEqual(captured, [{
            'hunt_count': 0, 'allocated': 0,
            'faces': {'FREE_PEOPLES': [], 'SHADOW': []}}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
