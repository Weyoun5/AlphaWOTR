"""Captured human Save watchdog telemetry must not halt native gameplay."""
from pathlib import Path
import shutil,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge'),str(ROOT)]
from live import client_protocol,live_parser
from live.driver import DriverHalt
from live.live_server import LiveServer
from test_bridge_oct12_dead_men_prefix_live import Rig,fresh_state,FP,SH
SAVE="<game> AI: modal dialog open: title='Save' text='Save In: File Name: Files of Type:' (injecting=false)"
class HumanSaveDialog(unittest.TestCase):
 def test_actual_save_line_is_benign_protocol_telemetry(self):
  self.assertEqual(client_protocol.client_error_lines([SAVE]),[])
  self.assertEqual(client_protocol.halt_verdict([SAVE]),(False,''))
  self.assertIsNotNone(live_parser.parse(SAVE))
 def test_driver_save_keeps_state_and_no_alarms(self):
  rig=Rig(fresh_state(),FP,phase='actions');self.addCleanup(shutil.rmtree,rig.td)
  original=rig.drv.es
  rig.feed(SAVE)
  self.assertIs(rig.drv.es,original);self.assertEqual(rig.drv.n_commits_total,0)
  self.assertEqual(rig.drv.alarms,[])
  self.assertFalse(any(f[0]=='client_error_line' for f in rig.drv.findings))
 def test_server_raw_request_save_no_halt(self):
  rig=Rig(fresh_state(),FP,phase='actions');self.addCleanup(shutil.rmtree,rig.td)
  server=LiveServer(rig.td,FP,load_nets=False,seed=7)
  server.handle({'lines':[SAVE]})
  self.assertEqual(server.driver.alarms,[])
 def test_server_injection_and_actual_error_still_halt(self):
  for line in (SAVE.replace('(injecting=false)','(injecting=true)'),
               SAVE.replace("text='Save In: File Name: Files of Type:'","text='Error saving file'")):
   with self.subTest(line=line):
    rig=Rig(fresh_state(),FP,phase='actions');self.addCleanup(shutil.rmtree,rig.td)
    server=LiveServer(rig.td,FP,load_nets=False,seed=7)
    with self.assertRaises(DriverHalt):server.handle({'lines':[line]})
    self.assertTrue(server.driver.alarms)
 def test_error_and_injecting_dialogs_still_alert(self):
  lines=[SAVE.replace('(injecting=false)','(injecting=true)'),
   SAVE.replace("title='Save'","title='Error'"),
   SAVE.replace("text='Save In: File Name: Files of Type:'","text='Error saving file'"),
   "<game> AI: modal dialog open: title='Choose' text='Unknown choice' (injecting=false)"]
  for line in lines:
   with self.subTest(line=line):
    self.assertTrue(client_protocol.halt_verdict([line])[0])
    rig=Rig(fresh_state(),FP,phase='actions');self.addCleanup(shutil.rmtree,rig.td)
    with self.assertRaises(DriverHalt):rig.feed(line)
    self.assertTrue(rig.drv.alarms)
if __name__=='__main__':unittest.main()
