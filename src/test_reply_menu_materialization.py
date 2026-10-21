"""Reply menus retain semantic order and seeded outcomes without eager rows."""
from collections import Counter
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai
import ai_policies
import game_state as gs
import reply_evaluation as replies
import search_lattice as lattice
import test_own_search_forced_probe as fixture


def reference_menu(engine, state, actor):
    policy=engine.get_ai_policy(state,actor)
    actions={}
    for action in policy.actions:
        actions.setdefault(ai_policies.action_policy_key(action),action)
    return actions,policy.forced_reason


class ReplyMenuTests(unittest.TestCase):
    def setUp(self):
        token=lattice.SCOPE.set(lattice.Scope())
        self.addCleanup(lattice.SCOPE.reset,token)

    def test_real_menu_order_and_keys_equal_without_materializing_blocks(self):
        state=fixture.army_state()
        projected=ai._segment_policy(SimpleNamespace(state=state,side=gs.Side.FREE_PEOPLES))
        self.assertTrue(isinstance(projected.actions,lattice.Menu),type(projected.actions))
        block_rows=sum(len(s) for s in projected.actions.segments if isinstance(s,lattice.Block))
        self.assertGreater(block_rows,0)
        engine=SimpleNamespace(get_ai_policy=lambda *args:projected)
        expected,reason=reference_menu(engine,state,gs.Side.FREE_PEOPLES)
        before=lattice.STATS['materialized']
        actual,actual_reason=replies.menu(engine,state,gs.Side.FREE_PEOPLES)
        self.assertEqual(tuple(actual),tuple(expected))
        self.assertEqual(actual_reason,reason)
        self.assertEqual(len(actual),len(expected))
        self.assertTrue(all(key in actual for key in expected))
        self.assertEqual(lattice.STATS['materialized'],before)
        chosen=next(s for s in projected.actions.segments if isinstance(s,lattice.Block)).keys()[0]
        self.assertEqual(vars(actual[chosen]),vars(expected[chosen]))
        self.assertEqual(lattice.STATS['materialized'],before+1)
        self.assertIs(actual[chosen],actual[chosen])
        self.assertEqual(lattice.STATS['materialized'],before+1)
        with self.assertRaises(KeyError):actual[('absent',-1,())]

    def test_semantic_duplicates_keep_first_row_across_block_boundaries(self):
        first=ai.Action(0,'army_move',{'from':'orthanc','to':'fords of isen','card_name':'first'})
        duplicate=ai.Action(0,'army_move',{'from':'orthanc','to':'fords of isen','card_name':'second'})
        other=ai.Action(0,'army_move',{'from':'orthanc','to':'dunland'})
        block=lattice.LatticeBlock([duplicate,other],'fixture',0,(),ai_policies.action_policy_key)
        for segments in (([first],block),(block,[first])):
            rows=lattice.Menu(ai.Action)
            for segment in segments:rows.extend(segment)
            policy=ai_policies.PolicyProjection(rows)
            engine=SimpleNamespace(get_ai_policy=lambda *args:policy)
            expected,_=reference_menu(engine,None,None)
            actual,_=replies.menu(engine,None,None)
            self.assertEqual(list(actual),list(expected))
            self.assertEqual(len(actual),2)
            self.assertEqual([vars(a) for a in actual.values()],
                             [vars(a) for a in expected.values()])
            if isinstance(segments[0],list):self.assertIs(actual[next(iter(actual))],first)

    def test_dynamic_mutations_are_rekeyed_on_each_call(self):
        action=ai.Action(0,'move',{'path':['orthanc']})
        rows=lattice.Menu(ai.Action);rows.extend([action,action])
        engine=SimpleNamespace(get_ai_policy=lambda *args:ai_policies.PolicyProjection(rows))
        old,_=replies.menu(engine,None,None)
        action.params['path'].append('fords of isen')
        new,_=replies.menu(engine,None,None)
        expected,_=reference_menu(engine,None,None)
        self.assertEqual(list(new),list(expected))
        self.assertNotEqual(list(new),list(old))
        self.assertIs(new[next(iter(new))],action)

    def test_both_sides_and_captured_siege_keep_full_ordered_policy_menu(self):
        ordinary=fixture.army_state()
        faces=[gs.ActionDieFace.ARMY,gs.ActionDieFace.CHARACTER,
               gs.ActionDieFace.MUSTER,gs.ActionDieFace.EVENT]
        ai.game_logic.adopt_rolled_faces(ordinary,[(gs.Side.SHADOW,i,face)
                                                 for i,face in enumerate(faces)])
        siege=ai.clone_state(ordinary)
        ai.game_logic.get_or_create_army(siege,'morannon').units=[
            gs.Unit(gs.Nation.GONDOR,gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.GONDOR,gs.UnitKind.ELITE),
            gs.Unit(gs.Nation.SAURON,gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.SAURON,gs.UnitKind.REGULAR)]
        siege.enemy_controlled[gs.Side.FREE_PEOPLES].add('morannon')
        siege.besieged_strongholds.add('morannon')
        for state in (ordinary,siege):
            for side in (gs.Side.FREE_PEOPLES,gs.Side.SHADOW):
                with self.subTest(siege=state is siege,side=side):
                    old,reason=reference_menu(ai,state,side)
                    new,new_reason=replies.menu(ai,state,side)
                    self.assertEqual(tuple(old),tuple(new))
                    self.assertEqual(reason,new_reason)

    def test_real_forced_policy_projection_and_seeded_successor_parity(self):
        ordinary=fixture.army_state()
        forced=ai.clone_state(ordinary)
        forced.pending_card_option={
            'card_id':'__gandalf_draw__','card_name':'Gandalf guide draw',
            'side':'fp','decider':'fp','direct':True,'gandalf_draw':True,
            'source_card_id':'FREE_PEOPLES_Character_House_of_the_Stewards',
            'deck':'character','die_face':'EVENT',
            'options':[{'gandalf_draw':True},{'gandalf_draw':False}]}
        for state in (ordinary,forced):
            old,reason=reference_menu(ai,state,gs.Side.FREE_PEOPLES)
            new,new_reason=replies.menu(ai,state,gs.Side.FREE_PEOPLES)
            self.assertEqual(tuple(old),tuple(new))
            self.assertEqual(reason,new_reason)
            self.assertEqual(bool(reason),state is forced)
            for seed in (17,941):
                keys=random.Random(seed).sample(list(old),min(12,len(old)))
                for key in keys:
                    with self.subTest(forced=bool(reason),seed=seed,action=key[0]):
                        old_fields=dict(vars(old[key]));new_fields=dict(vars(new[key]))
                        # The existing segmented policy stores this private
                        # encoding-only destination sequence as immutable tuple.
                        for fields in (old_fields,new_fields):
                            if '_block_dests' in fields:
                                fields['_block_dests']=tuple(fields['_block_dests'])
                        self.assertEqual(old_fields,new_fields)
                        outcomes=[]
                        for action in (old[key],new[key]):
                            with replies.transition_rng(ai,seed):
                                parent=ai.clone_state(state)
                                successor=ai.apply_action(parent,action)
                                ai._resume_preaction_for_search(parent,successor,action)
                                outcomes.append((pickle.dumps(successor),random.random(),
                                                 float(ai.np.random.random())))
                        self.assertEqual(*outcomes)


if __name__=='__main__':unittest.main(verbosity=2)
