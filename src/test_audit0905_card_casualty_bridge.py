"""Player-like card casualty gestures and exact source-stamped rendering."""
from pathlib import Path
import sys, tempfile, unittest
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src', ROOT/'wotr_bridge'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import ai, emitter, game_state as gs
import test_live_emission as TE
from test_audit0905_card_casualties import played, FP, SH, I, S, G, R, E
from test_audit0905_casualty_bridge_operations import Window
from wotr_bridge.live import gesture_match as gm, emission
from wotr_bridge.live.live_server import LiveServer


def choice(state, owner, operation, nation):
    return next(a for a in ai.get_legal_actions(state, owner)
                if a.params.get('choices', {}).get('casualty_operation') == operation
                and a.params['choices'].get('nation') == nation)


class CardCasualtyBridge(unittest.TestCase):
    def test_human_whole_elite_retains_all_sauron_regulars(self):
        state = played('The Spirit of Mordor', [6,5,1,2,3])
        result = gm.match_window(state, SH, Window('$Wey kills Isengard Elite in Orthanc.'), depth=0)
        self.assertEqual(result.status, 'match', result)
        self.assertEqual([a.params['choices']['casualty_operation'] for a in result.chain], ['remove_elite'])
        self.assertEqual(Counter((u.nation,u.kind) for u in result.post.armies['orthanc'].units),
                         Counter({(S,R):5}))

    def test_grouped_mixed_nation_reduction_and_loss(self):
        state = played('The Spirit of Mordor', [6,5,1,2,3])
        result = gm.match_window(state, SH, Window(
            '$Wey kills Isengard Elite in Orthanc.',
            '$Wey moves Isengard Regular from SA Reinforcements to Orthanc.',
            '$Wey kills Sauron Regular in Orthanc.'), depth=0)
        self.assertEqual(result.status, 'match', result)
        self.assertEqual(Counter((u.nation,u.kind) for u in result.post.armies['orthanc'].units),
                         Counter({(S,R):4,(I,R):1}))

    def test_split_removal_keeps_card_budget_until_next_choice(self):
        state = played('The Spirit of Mordor', [6,5,1,2,3])
        result = gm.match_window(state, SH, Window('$Wey kills Sauron Regular in Orthanc.'))
        self.assertEqual(result.status, 'match', result)
        self.assertEqual(result.post.pending_card_option['cas_hits'],1)
        self.assertEqual(result.post.pending_card_option['card_name'],'The Spirit of Mordor')
        self.assertEqual(result.post.pending_card_option['cas_side'],SH)

    def test_fp_whole_elite_and_explicit_replacement_death_remain_distinct(self):
        for replacement, expected_regular_casualties in ((False,0),(True,1)):
            with self.subTest(replacement=replacement):
                state=played('Dreadful Spells',[5,6])
                lines=['$Wey kills Gondor Elite in Fords of Isen.']
                if replacement:
                    lines += ['$Wey moves Gondor Regular from FP Reinforcements to Fords of Isen.',
                              '$Wey kills Gondor Regular in Fords of Isen.']
                result=gm.match_window(state,FP,Window(*lines),depth=0)
                self.assertEqual(result.status,'match',result)
                self.assertEqual(result.post.fp_casualties[G]['elite'],1)
                self.assertEqual(result.post.fp_casualties[G]['regular'],expected_regular_casualties)

    def test_wrong_nation_over_budget_and_incomplete_swap_never_match(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        for lines in [('$Wey kills Gondor Elite in Orthanc.',),
                      ('$Wey kills group ([3]Sauron Regular) in Orthanc.',),
                      ('$Wey moves Isengard Regular from SA Reinforcements to Orthanc.',)]:
            with self.subTest(lines=lines):
                self.assertNotEqual(gm.match_window(state,SH,Window(*lines),depth=0).status,'match')

    def test_whole_elite_card_emits_exact_piece(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        action=choice(state,SH,'remove_elite',I)
        view=TE.make_view([state])
        result=emission.render_decision(view,'Wey',SH,'card_option',action,
                                        pre=state,post=ai.apply_action(state,action))
        self.assertFalse(result.hold,result.hold)
        self.assertEqual(result.ops,[('hand_exit_b',{'nick':'Wey','pieces':['006'],'frm':'Game.1919'})])

    def test_card_reduction_requires_correct_source_stamp(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        action=choice(state,SH,'reduce_elite',I)
        view=TE.make_view([state])
        missing=emission.render_decision(view,'Wey',SH,'card_option',action,pre=state)
        self.assertTrue(missing.hold)
        self.assertEqual(missing.lines,[])
        result=emission.render_decision(view,'Wey',SH,'card_option',action,
                                        pre=state,post=ai.apply_action(state,action))
        self.assertFalse(result.hold,result.hold)
        self.assertEqual([op for op,fields in result.ops],['swap_place','swap_remove'])
        self.assertEqual(result.ops[0][1]['pieces'],['006'])
        pid=result.ops[1][1]['pieces'][0]
        self.assertEqual(emitter._unit_type(pid,view,side_hint='shadow'),(I,'regular'))

    def test_unrelated_card_identity_does_not_use_casualty_emitter(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        action=choice(state,SH,'remove_elite',I)
        wrong=ai.Action(-1,'card_option',dict(action.params,card_id='unrelated'))
        result=emission.render_decision(TE.make_view([state]),'Wey',SH,'card_option',wrong,
                                        pre=state,post=state)
        self.assertTrue(result.hold)
        self.assertEqual(result.lines,[])

    def test_card_preview_rows_keep_identity_and_remaining_hits(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        with tempfile.TemporaryDirectory() as folder:
            server=LiveServer(folder,FP,load_nets=False,seed=31)
            server.driver.es=state
            server.driver.hands.es=state
            server.driver.phase='actions'
            menu=server.status_menu()['human_menu']
            self.assertNotIn('error',menu,menu)
            self.assertEqual(menu['card_casualty'],{'region':'orthanc','hits':2})
            rows=menu['actions']
            self.assertEqual(len(rows),3)
            self.assertEqual(len({repr(row['rows']) for row in rows}),3)
            self.assertEqual(len({row['params']['say_pick'] for row in rows}),3)

    def test_card_budget_and_identity_change_matcher_context(self):
        state=played('The Spirit of Mordor',[6,5,1,2,3])
        altered=ai.clone_state(state)
        altered.pending_card_option['cas_hits']=1
        self.assertNotEqual(gm._pending_ctx_tag(state,None),gm._pending_ctx_tag(altered,None))
        self.assertFalse(gm._casualty_budget_decreased(state,altered,
                        ai.Action(-1,'card_option',{'choices':{'unit_type':'elite'}}),SH))


if __name__=='__main__':
    unittest.main(verbosity=2)
