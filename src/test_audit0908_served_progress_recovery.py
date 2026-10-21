from pathlib import Path
import ast,ctypes,json,multiprocessing as mp,os,queue,random,subprocess,sys,tempfile,time,types,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
ROOT=Path(os.environ.get('WOTR_PROGRESS_PROJECT_ROOT',str(ROOT))).resolve()
SOURCE=Path(os.environ.get('WOTR_PROGRESS_SOURCE',str(ROOT/'src'))).resolve()
sys.path[:0]=[str(SOURCE),str(ROOT/'src')]
import served_progress as sp,process_tree as pt,pre_ready_recovery as pr,batch_recovery as br

def producer(q):
    emitter=sp.ProgressEmitter(0,q,interval=0)
    for _ in range(5):emitter.completed('mcts_search');time.sleep(.04)
    time.sleep(5)

class ProgressTests(unittest.TestCase):
    def loop_scope(self,process,recovery=None):
        source=SOURCE/'run_td_v33loop.py';tree=ast.parse(source.read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='play_batch')
        scope=dict(os=os,subprocess=subprocess,json=json,ROOT=str(ROOT),SRC=str(ROOT/'src'),LOOP='unused',TRACE_DIR='unused',GAMES_JSONL='unused',NET_KEYS=[('fp','ring'),('sh','military')],PY=sys.executable,
            BATCH_WALL_SEC=1,ProcessTree=lambda *a,**kw:process,_PRE_READY_ATTEMPT=recovery,_frozen_cfg=lambda:None,
            _served_batch_launch=lambda *a:(['literal'],str(HERE/'never_created')),
            _archive_served_batch=lambda *a:a,log_line=lambda text:None)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
        return scope['play_batch']
    def test_actual_served_loop_waits_without_total_wall_limit(self):
        waits=[];p=types.SimpleNamespace(wait=lambda **kw:waits.append(kw) or 0)
        with patch.dict(os.environ,WOTR_LOOP_SELFPLAY='served_v1'):
            result=self.loop_scope(p)(0,25,4,1000)
        self.assertEqual(waits,[{'timeout':None}]);self.assertFalse(result[3])
    def test_loop_interruption_passes_actual_cleanup_to_recovery_before_reraise(self):
        calls=[];receipt={'root_returncode':1,'pid':77}
        def wait(**kw):raise KeyboardInterrupt('literal interruption')
        p=types.SimpleNamespace(wait=wait,terminate=lambda:calls.append('cleanup') or receipt)
        r=types.SimpleNamespace(launching=lambda *a:None,quiescent=lambda *a:calls.append(a))
        with patch.dict(os.environ,WOTR_LOOP_SELFPLAY='served_v1'),self.assertRaises(KeyboardInterrupt):
            self.loop_scope(p,r)(0,25,4,1000)
        self.assertEqual(calls,['cleanup',(p,1,receipt)])
    def test_twelve_hour_healthy_batch_survives_while_every_worker_advances(self):
        now=[0.];watch=sp.ProgressWatch(1800,clock=lambda:now[0])
        for wid in range(4):watch.started(wid)
        for sequence in range(1,38):
            now[0]+=1200
            for wid in range(4):watch.accept(wid,{'sequence':sequence,'stage':'mcts_search'})
            watch.check(range(4))
        self.assertGreater(now[0],8*3600)
    def test_active_peers_do_not_hide_stalled_worker(self):
        now=[0.];watch=sp.ProgressWatch(1800,clock=lambda:now[0])
        watch.started(0);watch.started(1)
        now[0]=1800;watch.accept(1,{'sequence':9,'stage':'_traced_execute'})
        with self.assertRaisesRegex(RuntimeError,'worker 0'):watch.check([0,1])
        watch.check([1])
    def test_repeated_sequence_and_malformed_payload_do_not_reset_deadline(self):
        now=[0.];watch=sp.ProgressWatch(1800,clock=lambda:now[0]);watch.started(0)
        watch.accept(0,{'sequence':1,'stage':'mcts_search'});now[0]=1801
        for payload in ({'sequence':1,'stage':'mcts_search'},{'sequence':True,'stage':'x'}):
            with self.assertRaises(ValueError):watch.accept(0,payload)
        with self.assertRaises(RuntimeError):watch.check([0])
    def test_only_completed_root_search_and_action_are_wrapped_and_rng_unchanged(self):
        messages=[];q=types.SimpleNamespace(put=messages.append)
        def work(value):return value+1
        fake=types.SimpleNamespace(mcts_search=work,_traced_execute=work,_get_priors_and_value=work)
        now=[0.];emitter=sp.ProgressEmitter(3,q,clock=lambda:now[0],interval=10)
        emitter.install(fake);rng=random.getstate()
        self.assertIs(fake._get_priors_and_value,work)
        self.assertEqual(fake.mcts_search(4),5)
        self.assertEqual(fake._traced_execute(5),6)
        self.assertEqual(len(messages),1)
        now[0]=11;fake.mcts_search(9);self.assertEqual(messages[-1][2]['sequence'],3)
        self.assertEqual(random.getstate(),rng)
        with self.assertRaises(TypeError):fake.mcts_search(None)
        self.assertEqual(emitter.sequence,3)
    def test_production_timeout_cannot_be_lowered_below_thirty_minutes(self):
        for value in ('0','nan','inf','1799'):
            with patch.dict(os.environ,WOTR_SERVED_STALL_SEC=value),self.assertRaises(ValueError):sp.stall_seconds()
        with patch.dict(os.environ,WOTR_SERVED_STALL_SEC='3600'):self.assertEqual(sp.stall_seconds(),3600)
    def test_snapshot_is_phase_bound_and_throttled(self):
        with tempfile.TemporaryDirectory() as td:
            now=[0.];watch=sp.ProgressWatch(1800,clock=lambda:now[0]);watch.started(2)
            path=Path(td)/'progress.json';watch.publish(path,'literal-phase',[2]);before=path.read_bytes()
            now[0]=1;watch.accept(2,{'sequence':4,'stage':'mcts_search'});watch.publish(path,'literal-phase',[2]);self.assertEqual(path.read_bytes(),before)
            now[0]=10;watch.publish(path,'literal-phase',[2]);row=json.loads(path.read_text())
            self.assertEqual(row['phase_id'],'literal-phase');self.assertEqual(row['workers']['2']['sequence'],4)
    def test_real_worker_progress_then_stall(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();child=ctx.Process(target=producer,args=(q,));child.start()
        watch=sp.ProgressWatch(.18);watch.started(0);messages=0
        try:
            while messages<5:
                msg=q.get(timeout=5);watch.accept(msg[1],msg[2]);watch.check([0]);messages+=1
            time.sleep(.20)
            with self.assertRaisesRegex(RuntimeError,'no concrete'):watch.check([0])
        finally:
            child.terminate();child.join(5);self.assertFalse(child.is_alive());q.close()

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'audit',prefix='pg_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for side in ('fp_ring','sh_military'):(self.root/('live_'+side+'.pt')).write_bytes(('literal-'+side).encode())
        self.j=br.BatchJournal(self.root,[self.root/'live_fp_ring.pt',self.root/'live_sh_military.pt'],[('fp','ring'),('sh','military')],{'test':'fixed'})
        self.j.bootstrap();self.r=pr.PreReadyRecovery(self.j);self.r.start(0)
        self.out=self.root/'incoming_served_b000';self.r.launching(0,str(self.out),1,2);self.out.mkdir()
        (self.out/'launch.json').write_text(json.dumps({'games':2,'workers':1}))
        (self.out/'phase.json').write_text(json.dumps({'finished':False,'completed':1}))
        (self.out/'completed.trace.gz').write_bytes(b'preserved completed bytes')
    def process(self):
        p=pt.ProcessTree([sys.executable,'-B',str(Path(__file__).resolve()),'--tree-child',str(self.root)],cwd=ROOT,env=dict(os.environ,WOTR_PROGRESS_SOURCE=str(SOURCE)))
        deadline=time.monotonic()+10
        while not (self.root/'child.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue((self.root/'child.json').exists())
        self.addCleanup(lambda:p.terminate() if p.poll() is None else None)
        return p
    def test_real_tree_termination_certificate_preserves_completed_game_and_recovers(self):
        p=self.process();child=json.loads((self.root/'child.json').read_text())['pid']
        handle=None
        if os.name=='nt':
            kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x100000,False,child);self.assertTrue(handle)
            kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        receipt=p.terminate(grace_seconds=.02)
        if handle:
            try:self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
            finally:kernel.CloseHandle(handle)
        # The same actual joined tree is refused by the original code because
        # forced shutdown cannot publish the runner's own finished phase.
        import importlib.util
        spec=importlib.util.spec_from_file_location('old_pre_ready',ROOT/'audit/2026-09-04/performance_pass/served_progress_recovery_r1/baseline/pre_ready_recovery.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        with self.assertRaisesRegex(RuntimeError,'quiescence is unproved'):
            old.PreReadyRecovery(self.j).quiescent(p,receipt['root_returncode'])
        self.r.quiescent(p,receipt['root_returncode'],receipt)
        certificate=json.loads(self.r.active.read_text())['certificate']
        self.assertEqual(certificate['termination'],receipt)
        self.r.recover()
        copies=list((self.root/'.batch_journal/abandoned').glob('*/incoming_served_b000/completed.trace.gz'))
        self.assertEqual(len(copies),1);self.assertEqual(copies[0].read_bytes(),b'preserved completed bytes')
        self.assertFalse(self.r.active.exists())
    def test_forged_unowned_receipt_refused(self):
        fake=types.SimpleNamespace(pid=123,poll=lambda:1,termination_receipt={'pid':123})
        with self.assertRaisesRegex(RuntimeError,'receipt is unproved'):self.r.quiescent(fake,1,{'pid':123})
        self.assertEqual(json.loads(self.r.active.read_text())['stage'],'launching')
    def test_changed_receipt_refused_and_property_copy_cannot_change_owner(self):
        p=self.process();receipt=p.terminate(grace_seconds=.02);bad=dict(receipt,pid=receipt['pid']+1)
        owned=p.termination_receipt;owned['pid']+=1
        self.assertEqual(p.termination_receipt,receipt)
        with self.assertRaisesRegex(RuntimeError,'receipt is unproved'):self.r.quiescent(p,receipt['root_returncode'],bad)
        self.r.quiescent(p,receipt['root_returncode'],receipt)
        (self.out/'phase.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'evidence changed'):self.r.recover()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--tree-child':
        root=Path(sys.argv[2]);child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (root/'child.json').write_text(json.dumps({'pid':child.pid}));time.sleep(60)
    else:unittest.main()
