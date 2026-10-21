"""Native-reproduced Dead Men arrival timing and once-only retry boundaries."""
from pathlib import Path
import copy,importlib.util,json,os,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
import ai,board,event_effects as ee,game_logic as gl,game_state as gs
from live import emission
from live.live_server import LiveServer,DriverHalt
if os.environ.get('WOTR_AUDIT_ORIGINAL_DEAD_MEN_TIMING') == '1':
    path=ROOT/'audit/2026-09-04/decision_pass/dead_men_subset_review/native/timing_original/live_server.py'
    spec=importlib.util.spec_from_file_location('live._original_dead_men_server',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    LiveServer,DriverHalt=original.LiveServer,original.DriverHalt
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
CID='FREE_PEOPLES_Character_Dead_Men_of_Dunharrow'

class DeadMenTiming(unittest.TestCase):
    def position(self,group=('Strider','Gimli')):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        Path(temp.name,'ai_feed.txt').write_text('')
        Path(temp.name,'ai_inject_ack.jsonl').write_text('')
        server=LiveServer(temp.name,FP,load_nets=False,seed=71)
        state=server.driver.es;state.armies={}
        source=gl.get_or_create_army(state,'westemnet')
        for name in ('Strider','Gimli'):
            comp=next(c for c in state.fellowship.companions if c.name==name)
            state.fellowship.companions.remove(comp);source.leaders.append(comp)
        enemy=gl.get_or_create_army(state,'erech')
        enemy.units=[gs.Unit(gs.Nation.SAURON,gs.UnitKind.REGULAR),gs.Unit(gs.Nation.SAURON,gs.UnitKind.ELITE)]
        state.political.boxes[gs.Nation.SAURON]=3
        state._declare_phase=False;state.dice_rolled=True
        for dice in state.dice.values():
            for die in dice:die.used=True
        state.dice[FP][0].used=False;state.dice[FP][0].current=gs.ActionDieFace.CHARACTER
        state.hands[FP].hand=[CID]
        play=next(a for a in ai.get_legal_actions(state,FP) if a.action_type=='play_event' and a.params['card_id']==CID)
        state=ai.apply_action(state,play)
        server.driver.es=state;server.driver.hands.es=state;server.driver.phase='actions';server.driver.last_actor=FP
        server.driver.view.piece_region.update({'241':'Westemnet','243':'Westemnet','011':'Erech','035':'Erech','262':'At War'})
        dp=ai.describe_pending(state)
        action=next(a for a in dp['actions'] if a.params.get('choices')=={'target_region':'erech','companions':list(group)})
        # The live bridge loads only its own seat and calls own_search.choose.
        # Setting the old pair alone silently exercised netless menu order.
        server.own_net=object()
        server.own_search=SimpleNamespace(choose=None)
        server._open_step=('ai_decide',dp)
        return server,dp,action

    def test_literal_arrival_precedes_roll_and_all_echoes_gate_retry(self):
        server,dp,action=self.position();before=ai.clone_state(server.driver.es)
        with patch.object(server.own_search,'choose',return_value=(action,None)) as select:
            first=server._ai_decide(dp)
            self.assertEqual(first,[
                '$WOTR_AI_FP<~Scribe.141~>243 <~Scribe.142~><~Game.1925~><~Scribe.143~><~Game.1913~><~Scribe.144~>',
                '$WOTR_AI_FP<~Scribe.141~>241 <~Scribe.142~><~Game.1925~><~Scribe.143~><~Game.1913~><~Scribe.144~>'])
            self.assertEqual(server._ai_decide(dp),[])
            self.assertEqual(server._run_open_step(),[])
            self.assertEqual(set(server.driver.echo_expect),{'241','243'})
            from bridge_evidence_fixture import publish
            header=publish(server.driver, first, session=1)
            ack_path=Path(server.driver.ack_path)
            def acknowledge(pid,index):
                raw='$WOTR_AI_FP<~Scribe.141~>'+pid+' <~Scribe.142~><~Game.1925~><~Scribe.143~><~Game.1913~><~Scribe.144~>'
                with ack_path.open('a',encoding='utf-8') as out:
                    out.write(json.dumps(dict(header,i=first.index(raw),kind='line',raw=raw,ok=True,paragraphs_before=1,paragraphs_after=2))+'\n')
                server.driver._consume_inject_acks()
            acknowledge('241',0)
            self.assertEqual(server._run_open_step(),[])
            acknowledge('241',0)  # duplicate receipt cannot release Gimli
            self.assertEqual(server._run_open_step(),[])
            acknowledge('243',1)
            self.assertEqual(server._ai_decide(dp),[
                '<WOTR_AI_FP> Free Peoples (AI) - event card roll: 1 die ',
                '!CALL combat_roll 1'])
            self.assertEqual(server._ai_decide(dp),[])
            self.assertEqual(select.call_count,1,'the pinned option must not be reselected')
            server.rand.feed('d6',[1])
            remaining=server._ai_decide(dp)
            self.assertFalse(any('241' in line or '243' in line for line in remaining),remaining)
            self.assertFalse(any('210' in line or 'play_card' in line for line in first+remaining))
            self.assertEqual(select.call_count,1)
        post=server.driver.es
        self.assertEqual([c.name for c in post.armies['erech'].leaders],['Strider','Gimli'])
        self.assertEqual(len(post.armies['erech'].units),2)
        self.assertEqual(ai.describe_pending(post)['side'],SH)
        self.assertTrue(post.pending_card_option['card_casualty'])
        self.assertEqual(post.pending_card_option['cas_hits'],1)
        self.assertIsNone(server._step_dead_men_arrival)
        self.assertEqual(post.dice[FP][0].used,before.dice[FP][0].used)

    def test_wrong_physical_source_refuses_before_dice(self):
        server,dp,action=self.position();server.driver.view.piece_region['243']='Bree'
        with patch.object(server.own_search,'choose',return_value=(action,None)):
            with self.assertRaisesRegex(DriverHalt,'Dead Men arrival'):
                server._ai_decide(dp)
        self.assertEqual(server.driver.view.piece_region['241'],'Westemnet')
        self.assertFalse(server.driver.echo_expect)

    def test_ambiguous_engine_identity_refuses(self):
        server,dp,action=self.position()
        other=gl.get_or_create_army(server.driver.es,'bree')
        other.leaders.append(copy.deepcopy(server.driver.es.armies['westemnet'].leaders[0]))
        with self.assertRaisesRegex(LookupError,'unique source'):
            emission.dead_men_arrival_prologue(server.driver.view,server.nick,FP,action)

    def test_strider_only_preserves_optional_companion(self):
        server,dp,action=self.position(('Strider',))
        arrival,rendered=emission.dead_men_arrival_prologue(server.driver.view,server.nick,FP,action)
        self.assertEqual([c.name for c in arrival.armies['westemnet'].leaders],['Gimli'])
        self.assertEqual([c.name for c in server.driver.es.armies['westemnet'].leaders],['Strider','Gimli'])
        self.assertEqual(len(rendered.lines),1);self.assertIn('241 ',rendered.lines[0])

    def test_other_card_never_uses_the_prologue(self):
        server,dp,action=self.position();action=copy.deepcopy(action)
        action.params['card_id']='FREE_PEOPLES_Character_Ents_Awake'
        self.assertIsNone(emission.dead_men_arrival_prologue(server.driver.view,server.nick,FP,action))

    def test_defender_politics_are_emitted_with_arrival_before_dice(self):
        server,dp,action=self.position(('Strider',))
        server.driver.es.political.boxes[gs.Nation.SAURON]=2
        arrival,rendered=emission.dead_men_arrival_prologue(server.driver.view,server.nick,FP,action)
        self.assertEqual(rendered.lines,[
            '$WOTR_AI_FP<~Scribe.141~>241 <~Scribe.142~><~Game.1925~><~Scribe.143~><~Game.1913~><~Scribe.144~>',
            '$WOTR_AI_FP<~HighlightLabel.101~>262<~HighlightLabel.102~><~Game.2002~><~HighlightLabel.103~><~Game.2003~>'])
        self.assertEqual(arrival.political.boxes[gs.Nation.SAURON],3)
        self.assertEqual(server.driver.es.political.boxes[gs.Nation.SAURON],2)
        self.assertFalse(rendered.calls)

    def test_abandon_reports_physical_residue_and_clears_phase(self):
        server,dp,action=self.position()
        with patch.object(server.own_search,'choose',return_value=(action,None)):
            server._ai_decide(dp)
        server._abandon_step('test explicit resync')
        self.assertIsNone(server._step_dead_men_arrival)
        self.assertTrue(any('ABANDONED STEP RESIDUE' in item for item in server.driver.alarms))

if __name__=='__main__':unittest.main()
