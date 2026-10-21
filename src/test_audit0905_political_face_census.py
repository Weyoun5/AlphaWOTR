from pathlib import Path
import sys,tempfile,unittest,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
from live import client_board,adopt,client_protocol
from live.live_server import LiveServer
from game_state import Side,Nation
NATIVE=ROOT/'audit/2026-09-04/decision_pass/attack_politics_native_review/next_issue/native_final_r3/neural_human_fp/gameplay_candidate_482c206f1a/client/final.board.txt'

class ActivationCensus(unittest.TestCase):
 def position(self):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
  path=Path(temp.name);(path/'ai_feed.txt').write_text('')
  driver=LiveServer(str(path),Side.SHADOW,load_nets=False,seed=61).driver
  shutil.copy2(NATIVE,path/'ai_client_board.txt');driver.view.dump_path=path/'ai_client_board.txt'
  from bridge_evidence_fixture import snapshot_header
  snapshot_header(driver)
  driver.resync_recover();return driver
 def drift(self,driver):
  driver.es.political.active[Nation.ROHAN]=False;driver.es.political.active[Nation.DWARVES]=True
 def face_rows(self,driver,states):
  return [r for r in client_board.diff_engine_vs_dump(driver.es,driver.view.piece_region,marker_states=states) if r.startswith('political activation ')]
 def expected(self):return ['political activation dwarves: engine=True client=False pids=257@political 3','political activation rohan: engine=False client=True pids=259@political 2']
 def test_native_faces_add_exact_two_independent_rows(self):
  driver=self.position();faces=client_protocol.read_marker_states(NATIVE)
  self.assertEqual(self.face_rows(driver,faces),[])
  self.drift(driver);self.assertEqual(self.face_rows(driver,faces),self.expected())
 def test_legacy_dump_unknown_faces_are_not_guessed(self):
  driver=self.position();self.drift(driver)
  self.assertEqual(self.face_rows(driver,None),[]);self.assertEqual(self.face_rows(driver,{}),[])
 def test_witness_and_live_census_receive_native_faces(self):
  driver=self.position();self.drift(driver)
  board,_=driver.census_rows();self.assertEqual([r for r in board if r.startswith('political activation ')],self.expected())
  notes=driver._witness_report();self.assertTrue(any('desync:' in str(note) for note in notes))
 def test_manual_transaction_corrects_activation_and_reduces_census(self):
  driver=self.position();self.drift(driver)
  before=self.face_rows(driver,driver._client_marker_states());self.assertEqual(len(before),2)
  driver.resync_recover()
  self.assertEqual(self.face_rows(driver,driver._client_marker_states()),[])
  self.assertTrue(driver.es.political.active[Nation.ROHAN]);self.assertFalse(driver.es.political.active[Nation.DWARVES])
 def test_fixed_active_markers_have_no_false_passive_inference(self):
  driver=self.position();faces=client_protocol.read_marker_states(NATIVE)
  for pid in ('256','261','262','263'):self.assertNotIn(pid,faces);self.assertTrue(client_board.political_marker_active(pid,faces))
  self.assertEqual(self.face_rows(driver,faces),[])

if __name__=='__main__':unittest.main(verbosity=2)
