"""Human-selected Dead Men groups and automatic attack politics, literal oracles."""
from pathlib import Path
import importlib.util,os,sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'wotr_bridge')]
HERE=ROOT/'audit/2026-09-04/decision_pass/dead_men_subset_review'
spec=importlib.util.spec_from_file_location('dead_men_fixture',HERE/'test_dead_men.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
import ai,event_effects as ee
from live import gesture_match as gm,live_parser,emission
from live.client_view import ClientView

if os.environ.get('WOTR_AUDIT_ORIGINAL_DEAD_MEN')=='1':
    spec=importlib.util.spec_from_file_location('live._original_dead_men_matcher',HERE/'bridge_original/gesture_match.py')
    gm=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gm;spec.loader.exec_module(gm)

class Window:
    die_side=None;die_face=None;closed='pending'
    def __init__(self,*lines):self.records=[live_parser.parse(line) for line in lines]
    def body_records(self):return self.records

class DeadMenBridge(unittest.TestCase):
    def match(self,lines,live=True):
        state=fixture.opened(enemy=True)
        with patch.object(ee,'_d6_batch',return_value=[1]):
            result=gm.match_window(state,fixture.FP,Window(*lines),advance_tolerance=live)
        return state,result

    def test_strider_only_commits_before_shadow_casualty_choice(self):
        before,result=self.match(['$AuditPlayer moves Strider from Westemnet to Lamedon.'])
        self.assertEqual(result.status,'match',result)
        self.assertEqual(len(result.chain),1)
        self.assertEqual(result.chain[0].params['choices']['companions'],['Strider'])
        self.assertEqual(fixture.locations(result.post),{'Gimli':'westemnet','Strider':'lamedon'})
        self.assertEqual(result.post.pending_card_option['decider'],'sh')
        self.assertEqual(result.post.pending_card_option['cas_hits'],1)
        self.assertEqual(len(result.post.armies['lamedon'].units),3)

    def test_literal_native_group_drag_selects_only_the_named_companions(self):
        before,result=self.match(['$AuditPlayer moves group (Strider, Gimli) from Westemnet to Lamedon.'])
        self.assertEqual(result.status,'match',result)
        self.assertEqual(set(result.chain[0].params['choices']['companions']),{'Strider','Gimli'})
        self.assertEqual(fixture.locations(result.post),{'Strider':'lamedon','Gimli':'lamedon'})

    def test_unselected_or_remote_companion_cannot_be_inferred(self):
        for name in ('Gimli','Boromir'):
            with self.subTest(name=name):
                before,result=self.match([f'$AuditPlayer moves {name} from Westemnet to Lamedon.'])
                self.assertNotEqual(result.status,'match')

    def test_voiced_wrong_political_advance_still_refuses(self):
        before,result=self.match(['$AuditPlayer moves Strider from Westemnet to Lamedon.',
                                 '$AuditPlayer moves Isengard from Political Track 1 to Political Track 3.'])
        self.assertNotEqual(result.status,'match')

    def test_spectator_stays_strict_until_automatic_marker_rows_arrive(self):
        lines=['$AuditPlayer moves Strider from Westemnet to Lamedon.']
        before,result=self.match(lines,live=False);self.assertNotEqual(result.status,'match')
        before,result=self.match(lines+[
            '$AuditPlayer moves Isengard from Political Track 1 to Political Track 0.',
            '$AuditPlayer moves Sauron from Political Track 1 to Political Track 0.'],live=False)
        self.assertEqual(result.status,'match',result)

    def test_initial_ai_emission_moves_group_before_attack_marker_legs(self):
        state=fixture.opened(enemy=True)
        action=next(a for a in ai.get_legal_actions(state,fixture.FP)
                    if a.params['choices']['target_region']=='lamedon'
                    and a.params['choices']['companions']==['Strider'])
        with patch.object(ee,'_d6_batch',return_value=[1]):post=ai.apply_action(state,action)
        view=ClientView(lambda:state);view.piece_region.update({'241':'Westemnet','243':'Westemnet'})
        rendered=emission.render_decision(view,'WOTR_AI_FP',fixture.FP,'card_option',action,post=post)
        self.assertFalse(rendered.hold)
        self.assertEqual(rendered.lines,[
            '$WOTR_AI_FP<~Scribe.141~>241 <~Scribe.142~><~Game.1925~><~Scribe.143~><~Game.1924~><~Scribe.144~>',
            '$WOTR_AI_FP<~HighlightLabel.101~>263<~HighlightLabel.102~><~Game.2002~><~HighlightLabel.103~><~Game.2003~>',
            '$WOTR_AI_FP<~HighlightLabel.101~>262<~HighlightLabel.102~><~Game.2002~><~HighlightLabel.103~><~Game.2003~>'])

if __name__=='__main__':unittest.main(verbosity=2)
