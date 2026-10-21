"""Literal native Hunt Guide death and Shadow-owned dice regressions."""
from pathlib import Path
import importlib.util
import os
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
import ai,board,game_logic as gl,game_state as gs
from live.client_view import ClientView
from live import emission,randomness
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW

if os.environ.get('WOTR_AUDIT_ORIGINAL_HUNT')=='1':
    original=ROOT/'audit/2026-09-04/decision_pass/guide_phase_review/hunt_native/original_runtime'
    def load(name):
        spec=importlib.util.spec_from_file_location('live._audit_original_'+name,original/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
    emission,randomness=load('emission'),load('randomness')


class HuntBridge(unittest.TestCase):
    def position(self,tie=False):
        state=gs.setup_initial_game(board.load_board());gl.initialize_decks(state)
        state._declare_phase=False;state.dice_rolled=True
        if tie:state.fellowship.companions=[c for c in state.fellowship.companions if c.name!='Strider']
        state.pending_hunt={'phase':'hunt_sacrifice','damage':2}
        state.fellowship.progress=1
        view=ClientView(lambda:state)
        view.piece_region.update({'240':'Guide Of The Fellowship','241':'Fellowship Of The Ring',
                                 '242':'Fellowship Of The Ring','243':'Fellowship Of The Ring',
                                 '244':'Fellowship Of The Ring','245':'Fellowship Of The Ring','246':'Fellowship Of The Ring'})
        return state,view

    def execute(self,state,view,kind):
        action=next(a for a in ai.get_legal_actions(state,FP) if a.action_type==kind)
        post=ai.apply_action(state,action)
        return post,emission.render_decision(view,'WOTR_AI_FP',FP,'hunt',action,post=post)

    def test_gandalf_sacrifice_seats_unique_strider_after_death(self):
        state,view=self.position();post,em=self.execute(state,view,'hunt_sacrifice_guide')
        self.assertEqual(post.fellowship.guide.name,'Strider')
        self.assertEqual(post.fellowship.corruption,0)
        self.assertNotIn('Gandalf the Grey',[c.name for c in post.fellowship.companions])
        self.assertIsNone(post.pending_guide_change);self.assertIsNone(post.pending_hunt)
        self.assertFalse(em.hold)
        self.assertEqual(em.lines,[
            '$WOTR_AI_FP<~Controls.281~>240 <~Controls.282~><~Game.1999~><~Controls.283~><~Game.2039~>',
            '$WOTR_AI_FP<~Scribe.141~>241 <~Scribe.142~><~Game.1998~><~Scribe.143~><~Game.1999~><~Scribe.144~>'])

    def test_tied_survivors_wait_for_the_actual_guide_choice(self):
        state,view=self.position(tie=True);post,em=self.execute(state,view,'hunt_sacrifice_guide')
        self.assertEqual(set(post.pending_guide_change['candidates']),{'Legolas','Gimli','Boromir'})
        self.assertEqual(em.lines,['$WOTR_AI_FP<~Controls.281~>240 <~Controls.282~><~Game.1999~><~Controls.283~><~Game.2039~>'])

    def test_corruption_choice_keeps_gandalf_and_emits_no_replacement(self):
        state,view=self.position();post,em=self.execute(state,view,'hunt_take_corruption')
        self.assertEqual(post.fellowship.guide.name,'Gandalf the Grey')
        self.assertEqual(post.fellowship.corruption,2)
        self.assertFalse(any('241' in line or '240' in line for line in em.lines))

    def test_already_seated_unique_guide_does_not_duplicate_drag(self):
        state,view=self.position();view.piece_region['241']='Guide Of The Fellowship'
        post,em=self.execute(state,view,'hunt_sacrifice_guide')
        self.assertEqual(len(em.lines),1)
        self.assertNotIn('241',em.lines[0])

    def test_wrong_physical_replacement_source_holds(self):
        state,view=self.position();view.piece_region['241']='Bree'
        post,em=self.execute(state,view,'hunt_sacrifice_guide')
        self.assertIn('forced Guide',em.hold or '')

    def test_real_separation_card_seats_forced_guide_after_departure(self):
        import event_effects as ee
        state,view=self.position();state.pending_hunt=None
        card=ee._card_by_name('I Will Go Alone')
        state.hands[FP].hand=[card.card_id]
        for dice in state.dice.values():
            for die in dice:die.used=True
        state.dice[FP][0].used=False;state.dice[FP][0].current=gs.ActionDieFace.CHARACTER
        play=next(a for a in ai.get_legal_actions(state,FP) if a.action_type=='play_event' and a.params['card_id']==card.card_id)
        state=ai.apply_action(state,play)
        view=ClientView(lambda:state);view.piece_region.update({'240':'Guide Of The Fellowship','241':'Fellowship Of The Ring'})
        action=next(a for a in ai.get_legal_actions(state,FP)
                    if a.params.get('choices',{}).get('companions')==['Gandalf the Grey']
                    and a.params['choices'].get('destination')=='bree')
        post=ai.apply_action(state,action)
        rendered=emission.render_decision(view,'WOTR_AI_FP',FP,'card_option',action,post=post)
        self.assertEqual(post.fellowship.guide.name,'Strider')
        self.assertFalse(rendered.hold)
        self.assertEqual(rendered.lines,[
            '$WOTR_AI_FP<~Scribe.141~>240 <~Scribe.142~><~Game.1999~><~Scribe.143~><~Game.1902~><~Scribe.144~>',
            '$WOTR_AI_FP<~Scribe.141~>241 <~Scribe.142~><~Game.1998~><~Scribe.143~><~Game.1999~><~Scribe.144~>'])

    def starve(self,human,context):
        owner=randomness.ClientRandomness(human_side=human.name,seed=77);owner.install();owner.begin_attempt()
        if context=='hunt reroll':owner.reroll_decline_shim=False
        with owner.advance():
            with self.assertRaises(randomness.NeedClientCall) as caught:
                gl.roll_d6_batch(1,context)
        return owner,caught.exception

    def test_human_shadow_is_asked_to_roll_real_hunt_dice(self):
        for context in ('hunt roll','hunt reroll'):
            with self.subTest(context=context):
                owner,call=self.starve(SH,context)
                self.assertIs(call.kw['side'],SH)
                out=owner.directives_for(call)
                self.assertEqual([(d.kind,d.text) for d in out],[('chat','roll 1 die')])

    def test_ai_shadow_executes_its_own_hunt_roll(self):
        owner,call=self.starve(FP,'hunt roll')
        self.assertIs(call.kw['side'],SH)
        self.assertEqual([(d.kind,d.text) for d in owner.directives_for(call)],[('call','!CALL combat_roll 1')])

    def test_unrelated_unowned_roll_keeps_existing_owner_contract(self):
        owner,call=self.starve(SH,'card die roll')
        self.assertIsNone(call.kw['side'])
        self.assertEqual([(d.kind,d.text) for d in owner.directives_for(call)],[('call','!CALL combat_roll 1')])


if __name__=='__main__':unittest.main(verbosity=2)
