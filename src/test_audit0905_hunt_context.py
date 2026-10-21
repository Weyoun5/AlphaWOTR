"""Public rolls with different known damage must reach distinct observations."""
from pathlib import Path
import os
import pickle
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
os.environ.update(WOTR_ACT_GEN='v48',WOTR_TIMING_GENERATION='v48',WOTR_NATIVE='0',CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0,str(SOURCE))
import test_audit0905_mithril_return as f
import hunt_protection_context as context
import numpy as np
assert Path(f.ai.__file__).resolve().parent==SOURCE

def different_public_rolls(staff=False):
    states=[]
    for rolls in ([6,1],[6,6]):
        state=f.setup(original='Eye',replacement='1')
        if staff:
            state,_=f.add_protection_through_public_play(state,"Wizard's Staff")
        state.dice[f.SH][0].current=f.gs.ActionDieFace.EYE
        state.dice[f.SH][0].used=True
        state.hunt_box.append(state.dice[f.SH][0])
        with patch.object(f.gl,'roll_d6_batch',side_effect=lambda n,*args:list(rolls)):
            state=f.choose(state,'fellowship_move')
        states.append(state)
    return states

class HuntProtectionContextTests(unittest.TestCase):
    def test_public_success_counts_distinguish_both_protection_windows(self):
        for staff in (False,True):
            states=different_public_rolls(staff)
            for side in (f.FP,f.SH):
                with self.subTest(staff=staff,side=side):
                    arrays=[f.ai.encode_state(s,side) for s in states]
                    self.assertEqual(arrays[0].shape,(4216,))
                    np.testing.assert_array_equal(arrays[0][:4191],arrays[1][:4191])
                    for n,encoded in enumerate(arrays,1):
                        expected=np.zeros(25,np.float32)
                        start=9 if staff else 0
                        expected[start]=1
                        expected[start+7]=n/5.
                        np.testing.assert_array_equal(encoded[4191:],expected)
                    self.assertFalse(np.array_equal(arrays[0].astype(np.float16),arrays[1].astype(np.float16)))

    def test_search_keeps_different_wizard_success_counts_separate(self):
        a,b=different_public_rolls(staff=True)
        self.assertNotEqual(f.ai._state_sig(a),f.ai._state_sig(b))
        self.assertNotEqual(f.ai._chance_sig(a),f.ai._chance_sig(b))

    def test_stronghold_source_and_resolved_region_have_literal_positions(self):
        state=f.setup(original='1R',replacement='3')
        state.fellowship.region=state.board.region('moria')
        state=f.choose(f.redraw(f.move(state),yes=False),'hunt_take_corruption')
        # Almanac p27: placement precedes the starting stronghold draw.
        state=f.place(state,'dimrill dale')
        expected=np.zeros(25,np.float32)
        expected[6]=1  # Mithril: stronghold extra draw
        # v48's seen-region mask applies only while placement is due. The
        # figure is already placed here; the draw source remains explicit.
        self.assertFalse(state._reveal_place_due)
        self.assertEqual(state._sh_tile_drawn_regions,{'moria'})
        np.testing.assert_array_equal(context.observation_tail(state),expected)
        before=pickle.dumps(state)
        for copy in (f.ai.clone_state(state),pickle.loads(before)):
            np.testing.assert_array_equal(context.observation_tail(copy),expected)
        self.assertEqual(pickle.dumps(state),before)

    def test_original_waiting_for_return_remains_explicit(self):
        state=f.redraw(f.move(f.setup()))
        expected=np.zeros(25,np.float32)
        expected[0]=1
        expected[7]=.2
        expected[8]=1
        np.testing.assert_array_equal(context.observation_tail(state),expected)

    def test_unrelated_or_finished_state_has_no_stale_context(self):
        state=f.setup()
        np.testing.assert_array_equal(context.observation_tail(state),np.zeros(25))
        state=f.redraw(f.move(state))
        state=f.place(f.choose(state,'hunt_take_corruption'),'dimrill dale')
        np.testing.assert_array_equal(context.observation_tail(state),np.zeros(25))

    def test_map_identity_and_invalid_continuations_fail_loudly(self):
        actual=tuple(r for r,meta in sorted(f.board.regions.items())
                     if meta.get('is_stronghold') and f.gl.is_shadow_region(r))
        self.assertEqual(actual,context.STRONGHOLDS)
        for key,value in [('successes',6),('successes',True),('apply_kind','unknown')]:
            state=different_public_rolls()[0]
            state._mithril_stash[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                f.ai.encode_state(state,f.FP)
        state=f.setup()
        state._reveal_place_due=True
        state._sh_tile_drawn_regions={'not a stronghold'}
        with self.assertRaises(ValueError):
            f.ai.encode_state(state,f.FP)

if __name__=='__main__':
    unittest.main(verbosity=2)
