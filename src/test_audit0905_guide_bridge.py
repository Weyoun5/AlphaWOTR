"""Player-shaped Guide nomination/keep boundaries; engine menu and timing."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
import ai,game_state as gs,game_logic as gl,emitter
from live.live_server import LiveServer
from live import live_parser
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
GANDALF='Gandalf the Grey'

class GuideBridge(unittest.TestCase):
    def server(self,side=SH):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        server=LiveServer(temp.name,side,load_nets=False,seed=77)
        d=server.driver;d.phase='declare';d._ensure_declare_armed()
        # This fixture routes individual parsed records without a Java export.
        # Complete-board completion/recovery is covered by the native-file suite.
        d.view.dump_path=None
        d.view.piece_region.update({'240':'Guide Of The Fellowship','241':'Fellowship Of The Ring',
                                    '244':'Fellowship Of The Ring'})
        return server
    def route(self,driver,line):
        driver._feed_seq=getattr(driver,'_feed_seq',0)+1
        rec=live_parser.parse(line)
        return driver._route('phase',rec)

    def balrog_server(self, side=FP, progress=0):
        import event_effects as ee
        s=self.server(side);d=s.driver;state=d.es
        gl.close_declare_phase(state)
        state.fellowship.region=state.board.region('moria')
        state.fellowship.progress=progress
        card=ee._card_by_name('Balrog of Moria')
        state.hands[SH].hand=[card.card_id]
        for dice in state.dice.values():
            for die in dice:die.used=True
        state.dice[SH][0].current=gs.ActionDieFace.EVENT
        state.dice[SH][0].used=False
        play=next(a for a in ai.get_legal_actions(state,SH)
                  if a.action_type=='play_event' and a.params['card_id']==card.card_id)
        d.es=ai.apply_action(state,play)
        d.hands.es=d.es
        for hm in d.hand_models.values():hm.es=d.es
        for dice in d.es.dice.values():
            for die in dice:die.used=False;die.current=None
        d.es.dice_rolled=False
        d._ensure_declare_armed()
        return s,card.card_id
    def test_literal_native_initial_strider_swap_commits_before_eyes(self):
        s=self.server();d=s.driver
        self.route(d,'$AuditPlayer moves Gandalf the Grey from Guide Of The Fellowship to Fellowship Of The Ring.')
        self.assertEqual(d.es.fellowship.guide.name,GANDALF)
        self.route(d,'$AuditPlayer moves Strider from Fellowship Of The Ring to Guide Of The Fellowship.')
        self.assertEqual(d.es.fellowship.guide.name,'Strider')
        self.assertEqual(d.phase,'eyes')
        self.assertIsNone(d.pending_buf)
        self.assertIsNone(d.es.pending_guide_change)
        self.assertTrue(d.es._eye_allocation_phase)
        self.assertEqual(d.es.eyes_allocated,0)
        self.assertFalse(d.es.dice_rolled)
        self.assertEqual([x[0] for x in d._out_queue].count('ai_eyes'),1)
        self.assertEqual(d._on_chat(live_parser.parse('<AuditPlayer> fp ok')),[])
    def test_fp_ok_explicitly_keeps_current_guide_and_opens_eyes_once(self):
        for current in (GANDALF,'Strider'):
            with self.subTest(current=current):
                s=self.server();d=s.driver;gl.change_guide(d.es,chosen=current)
                before=d.n_commits_total
                items=d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
                self.assertEqual(d.es.fellowship.guide.name,current)
                self.assertIsNone(d.es.pending_guide_change)
                self.assertEqual(d.phase,'eyes')
                self.assertEqual([x[0] for x in items].count('ai_eyes'),1)
                self.assertEqual(d.n_commits_total-before,2) # Declaration decline + keep.
                self.assertTrue(d.es._eye_allocation_phase)
    def test_lower_level_nomination_does_not_close_declaration(self):
        s=self.server();d=s.driver
        items=self.route(d,'$AuditPlayer moves Boromir from Fellowship Of The Ring to Guide Of The Fellowship.')
        self.assertEqual(d.es.fellowship.guide.name,GANDALF)
        self.assertEqual(d.phase,'declare')
        self.assertFalse(getattr(d.es,'_eye_allocation_phase',False))
        self.assertTrue(items)
    def test_fp_ok_cannot_skip_compulsory_replacement(self):
        s=self.server();d=s.driver
        d.es.pending_guide_change={'candidates':[GANDALF,'Strider']}
        before=d.n_commits_total
        items=d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
        self.assertEqual(d.n_commits_total,before)
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertFalse(getattr(d.es,'_eye_allocation_phase',False))
        self.assertTrue(items)
        self.route(d,'$AuditPlayer moves Strider from Fellowship Of The Ring to Guide Of The Fellowship.')
        self.assertEqual(d.es.fellowship.guide.name,'Strider')
        self.assertEqual(d.phase,'declare')
        self.assertIsNone(d.es.pending_guide_change)
    def test_ai_choose_and_keep_finish_before_fp_ok_and_rolls(self):
        for selected in ('Strider',GANDALF):
            with self.subTest(selected=selected):
                s=self.server(FP);d=s.driver;choices=[]
                class Chooser:
                    def rank(self,server,dp,actions):
                        desired=[a for a in actions if (dp['kind']=='declare' and a.action_type=='pass') or
                                 (dp['kind']=='guide_change' and a.action_type=='choose_guide' and a.params['companion']==selected)]
                        assert len(desired)==1,(dp['kind'],actions)
                        choices.append(desired[0].action_type);return desired
                s.script=Chooser();s._open_step=('ai_declare',None)
                first=s._ai_declare()
                self.assertFalse(any('fp ok' in line for line in first),first)
                self.assertEqual(d.phase,'guide')
                self.assertTrue(d.es._eye_allocation_phase)
                second=s._ai_declare()
                self.assertEqual(choices,['pass','choose_guide'])
                self.assertEqual(d.es.fellowship.guide.name,selected)
                self.assertEqual(d.phase,'eyes')
                self.assertIsNone(s._open_step)
                self.assertEqual(sum('fp ok' in line for line in second),1)
                self.assertFalse(any('roll_action_dice' in line for line in first+second))
                moves=[emitter.classify_line(line) for line in second if '<~Scribe.141~>' in line]
                self.assertEqual([fields['pieces'] for op,fields in moves], [['240'],['241']] if selected=='Strider' else [])
    def test_early_shadow_eye_cannot_skip_ai_guide(self):
        s=self.server(FP);d=s.driver
        d._open_fellowship_guide('test declaration complete')
        d._on_chat(live_parser.parse('<AuditPlayer> 0e'))
        self.assertFalse(d._allocate_one_eye())
        self.assertEqual(d.es.eyes_allocated,0)
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertEqual(d.phase,'guide')
    def test_parked_ai_eyes_preserve_phase_fence(self):
        s=self.server();d=s.driver
        d._open_fellowship_guide('test declaration complete')
        s._open_step=('ai_eyes',None)
        self.assertEqual(s._ai_eyes(),[])
        self.assertTrue(d.es._eye_allocation_phase)
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertFalse(d.es.dice_rolled)

    def test_parked_ai_eyes_do_not_relabel_compulsory_guide(self):
        s=self.server();d=s.driver
        gl.close_declare_phase(d.es)
        d.es.pending_guide_change={'candidates':[GANDALF,'Strider']}
        d.es._eye_allocation_phase=False
        self.assertFalse(d.es._eye_allocation_phase)
        s._open_step=('ai_eyes',None)
        self.assertEqual(s._ai_eyes(),[])
        self.assertFalse(d.es._eye_allocation_phase)
        self.assertEqual(d.es.pending_guide_change['candidates'],[GANDALF,'Strider'])

    def test_later_turn_and_already_mordor_still_allow_nomination(self):
        for mordor in (False,True):
            with self.subTest(mordor=mordor):
                s=self.server();d=s.driver;d.es.turn=4
                d.es.fellowship.mordor_entered=mordor
                self.route(d,'$AuditPlayer moves Strider from Fellowship Of The Ring to Guide Of The Fellowship.')
                self.assertEqual(d.es.fellowship.guide.name,'Strider')
                self.assertEqual(d.phase,'eyes')
                self.assertIsNone(d.es.pending_guide_change)
                self.assertTrue(d.es._eye_allocation_phase)
                self.assertFalse(d.es.dice_rolled)

    def test_single_eligible_guide_needs_no_artificial_keep_action(self):
        s=self.server();d=s.driver
        d.es.fellowship.companions=[c for c in d.es.fellowship.companions if c.name!='Strider']
        before=d.n_commits_total
        items=d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
        self.assertEqual(d.es.fellowship.guide.name,GANDALF)
        self.assertEqual(d.n_commits_total-before,1) # Only declaration decline.
        self.assertIsNone(d.es.pending_guide_change)
        self.assertEqual(d.phase,'eyes')
        self.assertEqual([it[0] for it in items],['ai_eyes'])

    def test_action_roll_cannot_bypass_pending_guide(self):
        s=self.server(FP);d=s.driver
        d._open_fellowship_guide('test declaration complete')
        before=[(die.current,die.used) for die in d.es.dice[SH]]
        result=d._on_action_roll({'op':'action_roll','faces':['E','A','A','P','C','M','W'],
                                 'norm':'$AuditPlayer rolls Shadow Action Dice.'})
        self.assertTrue(result)
        self.assertEqual([(die.current,die.used) for die in d.es.dice[SH]],before)
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertEqual(d.phase,'guide')

    def test_public_balrog_stationary_close_waits_before_guide(self):
        s,cid=self.balrog_server(FP);d=s.driver
        class Chooser:
            def rank(self,server,dp,actions):
                return [a for a in actions if a.action_type=='pass']
        s.script=Chooser();s._open_step=('ai_declare',None)
        out=s._ai_declare()
        self.assertEqual(d.phase,'fellowship_tail')
        self.assertEqual(d.es.pending_card_option['card_id'],cid)
        self.assertIsNone(d.es.pending_guide_change)
        self.assertFalse(getattr(d.es,'_eye_allocation_phase',False))
        self.assertFalse(any('fp ok' in line or 'roll_action_dice' in line for line in out))
        decline=next(a for a in ai.get_legal_actions(d.es,SH)
                     if a.params.get('choices',{}).get('balrog_declare_draw') is False)
        d._commit_human(decline,via='test public Balrog decline')
        self.assertEqual(d.phase,'guide')
        self.assertFalse(d.es._fellowship_tail_phase)
        self.assertIsNone(d.es.pending_card_option)
        self.assertEqual(d.es.pending_guide_change['candidates'],[GANDALF,'Strider'])
        self.assertFalse(d.es.dice_rolled)

    def test_public_balrog_draw_waits_for_hunt_before_guide(self):
        s,cid=self.balrog_server(FP);d=s.driver
        d._open_fellowship_guide('test stationary declaration close')
        draw=next(a for a in ai.get_legal_actions(d.es,SH)
                  if a.params.get('choices',{}).get('balrog_declare_draw') is True)
        d.es.hunt_pool=[gs.HuntTile(2)]
        with patch.object(gl,'pop_hunt_tile',side_effect=lambda state:state.hunt_pool.pop()):
            d._commit_human(draw,via='test public Balrog draw')
        self.assertEqual(d.phase,'fellowship_tail')
        self.assertEqual(d.es.pending_hunt['damage'],2)
        self.assertIsNone(d.es.pending_guide_change)
        self.assertFalse(getattr(d.es,'_eye_allocation_phase',False))
        self.assertEqual(len(d.es.hunt_discard),1)
        # The Free Peoples now owns the real damage menu, despite Shadow
        # having selected the preceding Balrog card option.
        dp=ai.describe_pending(d.es)
        self.assertIs(dp['side'],FP)
        class Chooser:
            def rank(self,server,pending,actions):
                return [a for a in actions if a.action_type=='hunt_take_corruption']
        s.script=Chooser();s._open_step=('ai_declare',None)
        s._ai_declare()
        self.assertEqual(d.es.fellowship.corruption,2)
        self.assertIsNone(d.es.pending_hunt)
        self.assertEqual(d.phase,'guide')
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertFalse(d.es.dice_rolled)

    def test_public_gateway_declaration_balrog_precedes_mordor_entry(self):
        s,cid=self.balrog_server(FP,progress=8);d=s.driver
        d.view.piece_region.update({'247':'Moria','248':'Track 8'})
        class Chooser:
            def rank(self,server,dp,actions):
                return [a for a in actions if a.action_type=='fellowship_declare'
                        and a.params.get('region')=='morannon']
        s.script=Chooser();s._open_step=('ai_declare',None)
        out=s._ai_declare()
        self.assertEqual(d.phase,'fellowship_tail')
        self.assertFalse(d.es.fellowship.mordor_entered)
        self.assertEqual(d.es.pending_card_option['card_id'],cid)
        self.assertIsNone(d.es.pending_guide_change)
        decline=next(a for a in ai.get_legal_actions(d.es,SH)
                     if a.params.get('choices',{}).get('balrog_declare_draw') is False)
        items=d._commit_human(decline,via='test gateway Balrog decline')
        self.assertTrue(d.es.fellowship.mordor_entered)
        self.assertEqual(d.phase,'guide')
        self.assertEqual([kind for kind,*_ in items].count('mordor_entry'),1)
        self.assertFalse(d.es.dice_rolled)
        self.assertTrue(d.es._eye_allocation_phase)
        self.assertEqual(d._resume_fellowship_tail(),[])

    def test_human_close_ai_balrog_decline_opens_human_guide(self):
        s,cid=self.balrog_server(SH);d=s.driver
        d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
        self.assertEqual(d.phase,'fellowship_tail')
        self.assertFalse(getattr(d.es,'_eye_allocation_phase',False))
        class Chooser:
            def rank(self,server,dp,actions):
                return [a for a in actions if a.params.get('choices',{}).get('balrog_declare_draw') is False]
        s.script=Chooser();s._open_step=('ai_decide',None)
        s._ai_decide(d.ai_action_dp())
        self.assertEqual(d.phase,'guide')
        self.assertIsNotNone(d.es.pending_guide_change)
        self.assertEqual(d.es.eyes_allocated,0)
        self.assertFalse(d.es.dice_rolled)
        d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
        self.assertEqual(d.phase,'eyes')
        self.assertIsNone(d.es.pending_guide_change)

    def test_human_actual_moria_declaration_cannot_trigger_balrog_twice(self):
        s,cid=self.balrog_server(SH);d=s.driver
        declare=next(a for a in ai.get_legal_actions(d.es,FP)
                     if a.action_type=='fellowship_declare' and a.params['region']=='moria')
        d._commit_human(declare,via='test actual Moria declaration')
        self.assertEqual(d.phase,'fellowship_tail')
        class Chooser:
            def rank(self,server,dp,actions):
                return [a for a in actions if a.params.get('choices',{}).get('balrog_declare_draw') is False]
        s.script=Chooser();s._open_step=('ai_decide',None)
        s._ai_decide(d.ai_action_dp())
        self.assertEqual(d.phase,'guide')
        self.assertIsNone(d.es.pending_card_option)
        d._on_chat(live_parser.parse('<AuditPlayer> fp ok'))
        self.assertEqual(d.phase,'eyes')
        self.assertIsNone(d.es.pending_card_option)

if __name__=='__main__':unittest.main(verbosity=2)
