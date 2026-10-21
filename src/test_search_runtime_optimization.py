"""Literal rules and mutation witnesses for the optimized search consumers."""
import copy, pickle, unittest
from unittest.mock import patch
import numpy as np
import ai, board, game_state as gs, search_action_encoding, action_encoding_backend

class SearchRuntimeOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_key=staticmethod(ai._action_cache_key)
        cls.reference_identity=staticmethod(ai._current_action_keys)
        state=gs.setup_initial_game(board.load_board())
        ai._encode_actions_batch([ai.Action(-1,'pass',{})],state,gs.Side.FREE_PEOPLES,{})

    def test_native_cache_and_identity_are_really_active(self):
        self.assertEqual(ai._action_cache_key.__module__,'wotr_action_encoding')
        self.assertEqual(ai._current_action_keys.__module__,'wotr_action_encoding')

    def test_identity_memo_is_local_and_duplicates_preserve_one_call(self):
        first=ai.Action(-1,'pass',{});second=ai.Action(-1,'pass',{})
        calls=[]
        def key(action):
            calls.append(action)
            return tuple(action.params.items())
        self.assertEqual(ai._current_action_keys([first,second,first],key),((),(),()))
        self.assertEqual(len(calls),2)
        first.params['changed']=2
        self.assertEqual(ai._current_action_keys([first],key),((('changed',2),),))

    def test_cached_rows_follow_literal_state_and_parameter_changes(self):
        state=gs.setup_initial_game(board.load_board());side=gs.Side.FREE_PEOPLES
        for action in [ai.Action(-1,'companion_separate',{'companion':'Meriadoc','to_region':'rivendell'}),
                       ai.Action(-1,'army_move',{'from':'minas tirith','to':'osgiliath','n_regular':1,'n_elite':0,'take_leaders':True,'side':'fp'}),
                       ai.Action(-1,'pass',{}),ai.Action(-1,'card_option',{'choices':{'region':'dale','nested':[1,2]}})]:
            for target in ('osgiliath','pelargir'):
                if action.action_type=='army_move':action.params['to']=target
                for progress in (0,3):
                    state.fellowship.progress=progress
                    self.assertEqual(self.reference_key(action,state,side,{}),ai._action_cache_key(action,state,side,{}))
                    np.testing.assert_array_equal(ai.encode_action(action,state,side),ai._encode_action_cached(action,state,side,{}))

    def test_snapshot_keeps_aliases_and_excludes_only_derived_fields(self):
        shared=['a',1];action=ai.Action(-1,'literal',{'one':shared,'two':shared})
        action._lattice=True;action.extra={'origin':7}
        fields=action_encoding_backend.prepare_snapshot(action)
        self.assertIsNotNone(fields)
        frozen=pickle.dumps(fields,protocol=5);shared.append(9)
        decoded=pickle.loads(frozen)
        self.assertIs(decoded['params']['one'],decoded['params']['two'])
        self.assertEqual(decoded['params']['one'],['a',1]);self.assertNotIn('_lattice',decoded)
        self.assertEqual(decoded['extra'],{'origin':7})

    def test_snapshot_cycle_and_custom_types_fall_back_safely(self):
        loop=[];loop.append(loop)
        self.assertIsNone(action_encoding_backend.prepare_snapshot(ai.Action(-1,'literal',{'loop':loop})))
        self.assertIsNone(action_encoding_backend.prepare_snapshot(ai.Action(-1,'literal',{'value':object()})))

    def test_separation_retains_every_group_in_place_and_rechecks_next_call(self):
        state=gs.setup_initial_game(board.load_board());here=ai.game_logic.fellowship_region(state)
        separable=[c for c in state.fellowship.companions if c.name!='Gollum' and c.level>0]
        actions=ai._get_companion_separate_actions(state,0)
        groups={tuple(sorted(a.params.get('companions') or [a.params['companion']])) for a in actions if a.params['to_region']==here}
        self.assertEqual(len(groups),2**len(separable)-1)
        state.fellowship.mordor_entered=True
        self.assertEqual(ai._get_companion_separate_actions(state,0),[])

    def test_hunt_hobbit_identity_does_not_share_answers(self):
        state=gs.setup_initial_game(board.load_board())
        for companion in ('Meriadoc','Peregrin'):
            state.pending_hunt={'phase':'hobbit_place','companion':companion,'level':1}
            actions=ai._get_companion_separate_actions(state,0)
            self.assertTrue(actions)
            self.assertTrue(all(a.params.get('companion')==companion for a in actions))

    def test_shadow_field_army_remains_a_companion_destination(self):
        from test_adv_companion_terminus_sibling import test_hunt_placed_hobbit_may_end_on_shadow_army
        test_hunt_placed_hobbit_may_end_on_shadow_army()

if __name__=='__main__':unittest.main()
