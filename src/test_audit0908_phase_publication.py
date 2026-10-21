from pathlib import Path
import ast,ctypes,json,os,sys,tempfile,threading,time,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
ROOT=Path(os.environ.get('WOTR_PHASE_PROJECT_ROOT',str(ROOT))).resolve()
SOURCE=Path(os.environ.get('WOTR_PHASE_SOURCE',str(ROOT/'src/run_td_v33sp.py'))).resolve()
sys.path.insert(0,str(ROOT/'src'))
def writer():
    nodes=[n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ('_write_phase_summary','_publish_phase_summary')]
    scope={'os':os,'time':time,'json':json}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope['_write_phase_summary']
class Publication(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'phase.json';self.path.write_text('{"old":true}')
    def publish(self):return writer()(str(self.path),'phase-identity','served-test',25,[],[],str(self.path.parent/'attempts.jsonl'))
    def test_transient_sharing_denial_retries_same_complete_document(self):
        actual=os.replace;calls=[]
        def replace(src,dst):
            calls.append((src,dst));self.assertEqual(json.loads(Path(src).read_text())['phase_id'],'phase-identity')
            if len(calls)<3:
                self.assertEqual(json.loads(self.path.read_text()),{'old':True})
                raise PermissionError('literal sharing denial')
            return actual(src,dst)
        with patch.object(os,'replace',side_effect=replace),patch.object(time,'sleep') as sleep:result=self.publish()
        self.assertEqual(len(calls),3);self.assertEqual(sleep.call_count,2)
        self.assertEqual(json.loads(self.path.read_text()),result);self.assertEqual(result['unreported'],25)
        self.assertFalse(Path(str(self.path)+'.tmp').exists())
    def test_permanent_denial_has_finite_attempts_and_keeps_both_artifacts(self):
        error=PermissionError('permanent denial')
        with patch.object(os,'replace',side_effect=error) as replace,patch.object(time,'sleep') as sleep:
            with self.assertRaises(PermissionError) as got:self.publish()
        self.assertIs(got.exception,error);self.assertEqual(replace.call_count,25);self.assertEqual(sleep.call_count,24)
        self.assertEqual(json.loads(self.path.read_text()),{'old':True})
        self.assertEqual(json.loads(Path(str(self.path)+'.tmp').read_text())['planned'],25)
    def test_other_io_error_propagates_immediately(self):
        error=OSError(28,'literal disk full')
        with patch.object(os,'replace',side_effect=error) as replace,patch.object(time,'sleep') as sleep:
            with self.assertRaises(OSError) as got:self.publish()
        self.assertIs(got.exception,error);self.assertEqual(replace.call_count,1);sleep.assert_not_called()
    @unittest.skipUnless(os.name=='nt','Actual Windows sharing semantics')
    def test_actual_windows_reader_temporarily_denies_delete_sharing(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p]
        kernel.CreateFileW.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.CreateFileW(str(self.path),0x80000000,3,None,3,0,None)
        self.assertNotIn(handle,(None,ctypes.c_void_p(-1).value))
        released=threading.Event()
        def release():time.sleep(.09);kernel.CloseHandle(handle);released.set()
        thread=threading.Thread(target=release);thread.start();denials=[];actual=os.replace
        def replace(src,dst):
            try:return actual(src,dst)
            except PermissionError as exc:denials.append(getattr(exc,'winerror',None));raise
        try:
            with patch.object(os,'replace',side_effect=replace):result=self.publish()
        finally:thread.join(2)
        self.assertTrue(released.is_set());self.assertTrue(denials)
        self.assertEqual(json.loads(self.path.read_text()),result)
if __name__=='__main__':unittest.main()
