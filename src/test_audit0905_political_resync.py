from pathlib import Path
import json,sys,tempfile,unittest,shutil
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
from live import adopt,client_board,client_protocol
from live.driver import LiveDriver
from game_state import Nation,PoliticalTrack
import log_replay
NATIVE=ROOT/'audit/2026-09-04/decision_pass/attack_politics_native_review/final/native/neural_human_fp/gameplay_candidate_d75c799fda/client/final.board.txt'

class NativePoliticalFace(unittest.TestCase):
 def state(self):
  return SimpleNamespace(political=PoliticalTrack(boxes={},active={}))
 def test_actual_active_marker_survives(self):
  state=self.state();state.political.active[Nation.ROHAN]=True
  dump=client_board.read_board(NATIVE);faces=client_protocol.read_marker_states(NATIVE)
  self.assertEqual(faces['259'],1);self.assertEqual(client_board.read_board_flags(NATIVE)['259'],'U')
  adopt.adopt_political(state,dump,marker_states=faces)
  self.assertTrue(state.political.active[Nation.ROHAN]);self.assertEqual(state.political.boxes[Nation.ROHAN],1)
 def test_actual_passive_marker_overrides_incorrect_engine_active(self):
  state=self.state();state.political.active[Nation.DWARVES]=True
  faces=client_protocol.read_marker_states(NATIVE);self.assertEqual(faces['257'],0)
  adopt.adopt_political(state,client_board.read_board(NATIVE),marker_states=faces)
  self.assertFalse(state.political.active[Nation.DWARVES])
 def test_fixed_active_single_face_markers(self):
  state=self.state();faces=client_protocol.read_marker_states(NATIVE)
  for pid in ['256','261','262','263']:
   self.assertNotIn(pid,faces);state.political.active[log_replay.POLITICAL_MARKERS[pid]]=False
  adopt.adopt_political(state,client_board.read_board(NATIVE),marker_states=faces)
  for pid in ['256','261','262','263']:self.assertTrue(state.political.active[log_replay.POLITICAL_MARKERS[pid]])
 def test_legacy_masks_never_guess_activation(self):
  state=self.state();state.political.active[Nation.ROHAN]=True;state.political.active[Nation.DWARVES]=False
  notes=adopt.adopt_political(state,{'259':'Political 2','257':'Political 3'},flags={'259':'U','257':'-'})
  self.assertTrue(state.political.active[Nation.ROHAN]);self.assertFalse(state.political.active[Nation.DWARVES])
  self.assertEqual(sum('unavailable' in note for note in notes),2)
 def driver(self):
  state=self.state();dump=client_board.read_board(NATIVE)
  adopt.adopt_political(state,dump,marker_states=client_protocol.read_marker_states(NATIVE))
  driver=LiveDriver.__new__(LiveDriver);driver.es=state;driver.view=SimpleNamespace(piece_region=dump,dump_path=NATIVE)
  driver.feed=SimpleNamespace(lines=[]);driver.nick='WOTR_AI_Shadow';driver.findings=[]
  return driver
 def test_reconcile_does_not_toggle_actual_active_counter(self):
  driver=self.driver();self.assertEqual(driver._political_recon(),[])
 def test_reconcile_passive_counter_once_until_actual_ack(self):
  driver=self.driver();driver.es.political.active[Nation.DWARVES]=True
  self.assertEqual(driver._political_recon(),[('line','$WOTR_AI_Shadow<~Controls.323~>257')])
  self.assertEqual(driver._political_recon(),[])
 def test_reconcile_missing_native_face_does_not_guess(self):
  driver=self.driver();driver.es.political.active[Nation.DWARVES]=True
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'legacy.txt';path.write_text('\n'.join('\t'.join(line.split('\t')[:4]) for line in NATIVE.read_text().splitlines()))
   driver.view.dump_path=path;self.assertEqual(driver._political_recon(),[])
 def test_seen_active_releases_inflight_toggle(self):
  driver=self.driver();driver._pol_face_walked={'259'}
  self.assertEqual(driver._political_recon(),[]);self.assertNotIn('259',driver._pol_face_walked)
 def test_public_manual_adoption_is_a_known_activation_writer(self):
  from live.live_server import LiveServer
  from game_state import Side
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp);(path/'ai_feed.txt').write_text('')
   driver=LiveServer(str(path),Side.SHADOW,load_nets=False,seed=61).driver
   shutil.copy2(NATIVE,path/'ai_client_board.txt');driver.view.dump_path=path/'ai_client_board.txt'
   from bridge_evidence_fixture import snapshot_header
   snapshot_header(driver)
   driver.es.political.active[Nation.ROHAN]=False;driver.es.political.active[Nation.DWARVES]=True
   driver._pol_active_prev=None;driver._political_activation_witness()
   driver._pol_face_walked={'257'}
   driver.alarms.append('pre-existing sentinel alarm')
   items=driver.resync_recover();driver._political_activation_witness()
   self.assertTrue(any('board adopted:' in str(it) for it in items))
   self.assertTrue(driver.es.political.active[Nation.ROHAN]);self.assertFalse(driver.es.political.active[Nation.DWARVES])
   self.assertNotIn('257',driver._pol_face_walked)
   self.assertEqual(driver.alarms,['pre-existing sentinel alarm'])
   # The authorized transaction must not disable detection of a later leak.
   driver.es.political.active[Nation.GONDOR]=True;driver._political_activation_witness()
   self.assertEqual(len(driver.alarms),2);self.assertIn('GONDOR',driver.alarms[-1])

if __name__=='__main__':unittest.main(verbosity=2)
