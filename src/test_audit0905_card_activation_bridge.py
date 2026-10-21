"""Literal TABA political activation at the completed native card boundary."""
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
from live.live_server import LiveServer
import ai,game_logic as gl,game_state as gs
CARD='FREE_PEOPLES_Character_There_and_Back_Again'

class CardActivation(unittest.TestCase):
 def position(self,captured=False):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);Path(temp.name,'ai_feed.txt').write_text('')
  server=LiveServer(temp.name,gs.Side.SHADOW,load_nets=False,seed=61);driver=server.driver;pre=driver.es
  gimli=next(c for c in pre.fellowship.companions if c.name=='Gimli')
  pre.fellowship.companions=[];pre.fellowship.guide=gs.Companion('Gollum',0,0)
  gl.get_or_create_army(pre,'erebor').leaders.append(gimli)
  pre.political.active[gs.Nation.DWARVES]=True
  if captured:pre.enemy_controlled[gs.Side.SHADOW].add('erebor')
  pre._declare_phase=False;pre.dice_rolled=True;pre.hands[gs.Side.FREE_PEOPLES].hand=[CARD]
  if CARD in pre.fp_char_deck:pre.fp_char_deck.remove(CARD)
  pre.dice[gs.Side.FREE_PEOPLES]=[gs.ActionDie([gs.ActionDieFace.CHARACTER],gs.Side.FREE_PEOPLES)]
  pre.dice[gs.Side.FREE_PEOPLES][0].current=gs.ActionDieFace.CHARACTER
  driver.phase='actions';driver.view.piece_region.update({'256':'Political 3','257':'Political 3','258':'Political 3','364':'Guide Box','243':'Erebor'})
  action=next(a for a in ai.get_legal_actions(pre,gs.Side.FREE_PEOPLES) if a.action_type=='play_event' and a.params.get('card_id')==CARD)
  post=ai.apply_action(pre,action)
  return driver,pre,post,action
 def lines(self,items):return [it[1] for it in items if it[0]=='line']
 def sweep(self,driver,pre,post,action,**kwargs):return self.lines(driver.post_commit_sweep(pre,post,label='chat:play_event',actions=[action],**kwargs))
 def test_actual_taba_card_activates_before_three_advances(self):
  driver,pre,post,action=self.position();lines=self.sweep(driver,pre,post,action)
  flip='$WOTR_AI_Shadow<~Controls.323~>258'
  self.assertEqual(lines[0],flip)
  self.assertEqual(set(lines[1:]),{f'$WOTR_AI_Shadow<~HighlightLabel.91~>{pid}<~HighlightLabel.92~><~Game.2000~><~HighlightLabel.93~><~Game.2001~>' for pid in ('256','257','258')})
  self.assertEqual(len(lines),4)
 def test_repeat_commit_does_not_flip_or_advance_twice(self):
  driver,pre,post,action=self.position();self.sweep(driver,pre,post,action)
  self.assertEqual(self.sweep(driver,pre,post,action),[])
 def test_captured_region_has_no_activation_or_advance(self):
  driver,pre,post,action=self.position(captured=True)
  self.assertEqual(self.sweep(driver,pre,post,action),[])
 def test_already_expressed_human_flip_is_not_toggled_back(self):
  driver,pre,post,action=self.position()
  lines=self.sweep(driver,pre,post,action,expressed=[('flip',{'pieces':['258']})])
  self.assertFalse(any('Controls.323' in line for line in lines));self.assertEqual(len(lines),3)
 def test_nested_card_option_also_emits_activation(self):
  driver,pre,post,action=self.position()
  wrapped=ai.Action(die_idx=-1,action_type='card_option',params={'card_id':CARD,'choices':{'_taba_done':True}})
  lines=self.sweep(driver,pre,post,wrapped)
  self.assertEqual(lines[0],'$WOTR_AI_Shadow<~Controls.323~>258')

if __name__=='__main__':unittest.main(verbosity=2)
