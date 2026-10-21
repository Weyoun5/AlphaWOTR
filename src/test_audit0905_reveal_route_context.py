"""Public human routes can owe different future draws at the same destination."""
import os
import pickle
import unittest

os.environ.update(WOTR_ACT_GEN='v49',WOTR_TIMING_GENERATION='v49',WOTR_TRANSIT_GENERATION='v49')
import test_audit0905_reveal_transit as t
import numpy as np
import fellowship_reveal_route as route

def different_public_routes():
    states=[]
    for middle in ('dol guldur','western brown lands'):
        state=t.crossing(progress=5,extra_two=True)
        state=t.human_route(state,'eastern brown lands',
            ['hollin','moria','dimrill dale','south anduin vale',middle,'eastern brown lands'])
        states.append(state)
    return states

class RouteContextTests(unittest.TestCase):
    def test_public_routes_distinguish_owed_draw_in_both_inputs_and_search(self):
        a,b=different_public_routes()
        expected=np.zeros(64,np.float32)
        expected[1]=1 # Dol Guldur is the next required extra draw.
        for side in (t.f.FP,t.f.SH):
            left,right=[t.f.ai.encode_state(s,side) for s in (a,b)]
            self.assertEqual(left.shape,(4280,))
            np.testing.assert_array_equal(left[:4216],right[:4216])
            np.testing.assert_array_equal(left[4216:],expected)
            np.testing.assert_array_equal(right[4216:],np.zeros(64))
            self.assertFalse(np.array_equal(left.astype(np.float16),right.astype(np.float16)))
        self.assertNotEqual(t.f.ai._state_sig(a),t.f.ai._state_sig(b))
        self.assertNotEqual(t.f.ai._chance_sig(a),t.f.ai._chance_sig(b))

    def test_clone_pickle_and_complete_route_keep_and_clear_only_the_owed_draw(self):
        state=different_public_routes()[0]
        before=pickle.dumps(state)
        for branch in (t.f.ai.clone_state(state),pickle.loads(before)):
            np.testing.assert_array_equal(route.observation_tail(branch),route.observation_tail(state))
            branch=t.f.choose(t.f.redraw(branch,yes=False),'hunt_take_corruption')
            self.assertEqual(repr(branch._mithril_stash['tile']),'2')
            np.testing.assert_array_equal(route.observation_tail(branch),np.zeros(64))
            branch=t.f.choose(t.f.redraw(branch,yes=False),'hunt_take_corruption')
            self.assertEqual(branch.fellowship.corruption,6)
            np.testing.assert_array_equal(route.observation_tail(branch),np.zeros(64))
        self.assertEqual(pickle.dumps(state),before)

    def test_invalid_queue_and_duplicates_are_not_silently_encoded(self):
        for queue in ('moria',('unknown',),('moria','moria'),tuple(['moria']*9)):
            state=different_public_routes()[0]
            state._pending_reveal_strongholds=queue
            with self.subTest(queue=queue),self.assertRaises(ValueError):
                t.f.ai.encode_state(state,t.f.FP)
            with self.assertRaises(ValueError):route.pending_draws(state)
            # The existing search API signals an invalid fingerprint with None.
            self.assertIsNone(t.f.ai._state_sig(state))

if __name__=='__main__':unittest.main(verbosity=2)
