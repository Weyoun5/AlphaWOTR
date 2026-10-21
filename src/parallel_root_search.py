"""Opt-in independent root trees sharing the existing batched inference server.

This is a diagnostic coordinator, not a shared-tree MCTS implementation or a
new default selector. It never repairs, drops or retries a failed worker tree.
"""
import copy
import hashlib
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import time
import traceback
import numpy as np

CONTRACT='independent_root_trees_v1'

def split_budget(total, workers):
    if type(total) is not int or type(workers) is not int or workers < 1 or total < workers:
        raise ValueError('Each worker needs at least one simulation; total must be an integer')
    q,r=divmod(total,workers)
    return [q+int(i<r) for i in range(workers)]

def aggregate(records, legal_actions):
    """Pool visits and backup sufficient statistics using semantic actions."""
    from ai_policies import action_policy_key
    legal={action_policy_key(a):a for a in legal_actions}
    if len(legal)!=len(legal_actions):
        raise ValueError('Duplicate semantic root actions')
    entries={}
    required=('visit_counts','q_values','q_var','term_frac','priors_clean','proven',
              'proven_value','forced','conf','phi','action_encs','region_indices')
    for record in records:
        actions=record.get('root_actions')
        if actions is None or not actions:
            raise ValueError('Worker omitted semantic root actions')
        if any(len(record[key])!=len(actions) for key in required):
            raise ValueError('Worker root statistic lengths differ')
        seen=set()
        for i,action in enumerate(actions):
            key=action_policy_key(action)
            if key not in legal or key in seen:
                raise ValueError('Worker returned illegal or duplicated semantic action')
            seen.add(key)
            values={name:float(record[name][i]) for name in required if name not in ('action_encs','region_indices')}
            if not all(np.isfinite(x) for x in values.values()):
                raise ValueError('Nonfinite worker root statistics')
            n=values['visit_counts']
            if (n<0 or n!=int(n) or values['q_var']<0 or not 0<=values['term_frac']<=1
                    or values['priors_clean']<0 or not 0<=values['forced']<=n
                    or values['forced']!=int(values['forced']) or values['proven'] not in (-1,0,1)):
                raise ValueError('Invalid worker visits/probabilities')
            enc=np.asarray(record['action_encs'][i],dtype=np.float32)
            if not np.isfinite(enc).all():raise ValueError('Nonfinite action encoding')
            item=entries.setdefault(key,dict(action=legal[key],n=0,qsum=0,second=0,terminal=0,
                prior=0,forced=0,conf=0,phi=0,proofs=[],enc=enc,region=int(record['region_indices'][i])))
            if not np.array_equal(item['enc'],enc) or item['region']!=int(record['region_indices'][i]):
                raise ValueError('Identical root action encoded differently across workers')
            item['n']+=int(n);item['qsum']+=n*values['q_values']
            item['second']+=n*(values['q_var']+values['q_values']**2)
            item['terminal']+=n*values['term_frac'];item['prior']+=values['priors_clean']
            item['forced']+=int(values['forced']);item['conf']+=n*values['conf'];item['phi']+=n*values['phi']
            item['proofs'].append((int(values['proven']),values['proven_value']))
    items=[entries[key] for key in legal if key in entries]
    if not items or sum(x['n'] for x in items)<=0:
        raise ValueError('Parallel trees produced no root visits')
    n=np.array([x['n'] for x in items],dtype=np.int64)
    q=np.array([x['qsum']/x['n'] if x['n'] else 0 for x in items])
    p=np.array([x['prior'] for x in items]);p/=p.sum() if p.sum()>0 else 1
    if not p.any():raise ValueError('Parallel trees produced no prior mass')
    proof=[]
    for x in items:
        ps=x['proofs']
        # A proof is retained only when every independent tree reports exactly
        # the same proof/value for this action. Missing or conflicting evidence
        # stays unproved; no proof is fabricated from the averaged Q.
        proof.append(ps[0] if len(ps)==len(records) and all(y==ps[0] for y in ps) else (0,0.0))
    return dict(root_actions=tuple(x['action'] for x in items),action_encs=np.stack([x['enc'] for x in items]),
        region_indices=np.array([x['region'] for x in items],dtype=np.int32),visit_counts=n,
        visits=n/n.sum(),q_values=q,q_var=np.maximum(0,np.array([x['second']/x['n'] if x['n'] else 0 for x in items])-q*q),
        term_frac=np.array([x['terminal']/x['n'] if x['n'] else 0 for x in items]),priors_clean=p,
        forced=np.array([x['forced'] for x in items],dtype=np.int64),
        conf=np.array([x['conf']/x['n'] if x['n'] else 1 for x in items]),
        phi=np.array([x['phi']/x['n'] if x['n'] else 0 for x in items]),
        proven=np.array([x[0] for x in proof],dtype=np.int8),proven_value=np.array([x[1] for x in proof]))

def _server(connections, paths, arch, device, environment):
    os.environ.update(environment)
    import infer_server
    infer_server.server_main(connections,paths,arch,device=device)

def _worker(wid, commands, inference, environment):
    os.environ.update(environment)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    try:
        import torch
        torch.set_num_threads(1)
        import ai,infer_server,rng_streams
        ai.DEVICE=torch.device('cpu')
        handles=infer_server.attach_client(inference,['fp','sh'],require_metadata=True)
        client=handles['fp']._client
        original_roundtrip=client._roundtrip
        inference_totals={'seconds':0.0}
        def timed_roundtrip(*args,**kwargs):
            started=time.perf_counter()
            try:return original_roundtrip(*args,**kwargs)
            finally:inference_totals['seconds']+=time.perf_counter()-started
        client._roundtrip=timed_roundtrip
        for side,net in handles.items():
            if getattr(net,'reward_profile',None) is not None:
                ai.PLAYER_PROFILES[side]=net.reward_profile
        from trace_decisions import current_learning_context
        from selector_contract import current_identity
        commands.send(('ready',wid,{'checkpoints':{key:net.checkpoint_id for key,net in handles.items()},
            'learning':current_learning_context(),'selector':current_identity()}))
        while True:
            job=commands.recv()
            if job is None:break
            jid,state,side,budget,seed,options=job
            rng_streams.start_episode({'rng_episode_seed':seed,'rng_episode_identity':{'parallel_job':jid,'worker':wid}})
            torch.manual_seed(seed % (2**63))
            ai._TREE_CACHE['child']=None
            prior_requests=client._rid
            prior_wait=inference_totals['seconds']
            prior_kernel={key:value for key,value in ai.KERNEL_STATS.items() if type(value) is int}
            start=time.perf_counter()
            result=ai.mcts_search(state,side,handles['fp'],handles['sh'],num_simulations=budget,
                bonus_sims=0,temperature=0,add_noise=False,**options)
            if result is None or result[1] is None:
                raise ValueError('Non-forced parallel search returned no statistics')
            diagnostic={'inference_requests':client._rid-prior_requests,
                'inference_roundtrip_seconds':inference_totals['seconds']-prior_wait,
                'kernel_counter_delta':{key:value-prior_kernel.get(key,0) for key,value in ai.KERNEL_STATS.items() if type(value) is int}}
            commands.send(('result',jid,wid,result[1],time.perf_counter()-start,budget,seed,diagnostic))
    except BaseException:
        try:commands.send(('error',wid,traceback.format_exc()))
        except (OSError,EOFError):pass
    finally:
        commands.close();inference.close()

class ParallelRootSearch:
    """Persistent CPU tree workers and exactly one existing inference server.

    Use as a context manager. Calls are synchronous and never overlap. A worker
    error/timeout poisons and closes the coordinator; no partial result escapes.
    """
    def __init__(self,fp_path,sh_path,*,workers=2,device='cuda:0',arch='base',environment=None,timeout=120):
        split_budget(workers,workers)
        if not np.isfinite(timeout) or timeout<=0:raise ValueError('Invalid timeout')
        self.paths={side:str(Path(path).resolve()) for side,path in (('fp',fp_path),('sh',sh_path))}
        if any(not Path(p).is_file() for p in self.paths.values()):raise ValueError('Checkpoint does not exist')
        self.hashes={side:hashlib.sha256(Path(p).read_bytes()).hexdigest() for side,p in self.paths.items()}
        self.workers=workers;self.device=device;self.arch=arch;self.timeout=float(timeout)
        self.environment=dict(environment or {})
        self.environment.update(WOTR_NET_ARCH=arch,WOTR_RNG_CONTRACT='isolated_game_search_v1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self.processes=[];self.commands=[];self.server=None;self.control=None
        self.job=0;self.closed=False;self.startup_seconds=None

    def __enter__(self):
        if self.closed or self.server is not None:raise RuntimeError('Coordinator cannot be reused after close/start')
        ctx=mp.get_context('spawn');inference=[];servers=[];children=[]
        self.control,control_child=ctx.Pipe()
        start=time.perf_counter()
        try:
            for wid in range(self.workers):
                parent,child=ctx.Pipe();client,server=ctx.Pipe()
                self.commands.append(parent);children.append(child);inference.append(client);servers.append(server)
            self.server=ctx.Process(target=_server,args=(servers+[control_child],self.paths,self.arch,self.device,self.environment))
            self.server.start()
            for wid in range(self.workers):
                p=ctx.Process(target=_worker,args=(wid,children[wid],inference[wid],self.environment))
                p.start();self.processes.append((wid,p))
            for conn in children+inference+servers+[control_child]:conn.close()
            messages=self._receive('ready')
            from trace_decisions import current_learning_context
            from selector_contract import current_identity
            expected={'checkpoints':self.hashes,'learning':current_learning_context(),'selector':current_identity()}
            if any(message[2]!=expected for message in messages):raise ValueError('Worker checkpoint/learning/selector identity differs')
            self.startup_seconds=time.perf_counter()-start
            return self
        except BaseException:
            self.close()
            for conn in children+inference+servers+[control_child]:conn.close()
            raise

    def _receive(self,kind):
        pending=set(range(self.workers));messages=[];deadline=time.monotonic()+self.timeout
        while pending:
            if self.server is None or not self.server.is_alive():raise RuntimeError('Inference server exited')
            left=deadline-time.monotonic()
            if left<=0:raise TimeoutError('Parallel root workers timed out')
            ready=wait([self.commands[i] for i in pending],timeout=min(left,.2))
            for conn in ready:
                message=conn.recv()
                if message[0]!='ready' and message[0]!='result':raise RuntimeError('Parallel worker failed: '+str(message))
                if message[0]!=kind:raise RuntimeError('Unexpected parallel worker protocol')
                wid=message[1] if kind=='ready' else message[2]
                if wid not in pending or conn is not self.commands[wid]:raise RuntimeError('Duplicate/foreign worker response')
                pending.remove(wid);messages.append(message)
            for wid,p in self.processes:
                if wid in pending and not p.is_alive():raise RuntimeError('Parallel worker exited without result')
        return sorted(messages,key=lambda m:m[1] if kind=='ready' else m[2])

    def search(self,state,side,*,simulations=64,seed=1,temperature=0,determinize=True,c_puct=None):
        if self.closed or self.server is None:raise RuntimeError('Coordinator is not running')
        budgets=split_budget(simulations,self.workers)
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('Seed must be a uint64')
        import ai
        from rng_streams import derive_seed
        policy=ai.get_ai_policy(state,side)
        if policy.forced_reason:return policy.actions[0],None
        self.job+=1;start=time.perf_counter()
        try:
            options={'determinize':determinize}
            if c_puct is not None:options['c_puct']=c_puct
            seeds=[derive_seed(seed,'root-tree:%d'%i) for i in range(self.workers)]
            for wid,conn in enumerate(self.commands):conn.send((self.job,state,side,budgets[wid],seeds[wid],options))
            responses=self._receive('result')
            if any(r[1]!=self.job or r[5]!=budgets[r[2]] or r[6]!=seeds[r[2]] for r in responses):
                raise RuntimeError('Stale worker search/budget/seed response')
            info=aggregate([r[3] for r in responses],policy.actions)
            # The existing selector is run once on pooled sufficient stats.
            # Its temporary sampling RNG cannot perturb the caller's game RNG.
            old_rng=np.random.get_state()
            try:
                np.random.seed(derive_seed(seed,'root-selection',bits=32))
                idx,pi,diag=ai.select_root_action(dict(p=info['priors_clean'],q=info['q_values'],n=info['visit_counts'],
                    tf=info['term_frac'],pv=info['proven'],pvv=info['proven_value'],fn=info['forced'],cf=info['conf'],ph=info['phi']),
                    temperature,cfg=copy.deepcopy(ai.SELECT_CFG))
            finally:np.random.set_state(old_rng)
            from selector_contract import make_evidence
            from reward_contract import current_contract
            info.update(make_evidence(ai.SELECT_CFG,temperature,pi,eligible=diag['elig'],reward_contract=current_contract()))
            info.update(played=idx,q_hat=diag['q_hat'],sigma=diag['sigma'],regret=diag['regret'],
                parallel_root={'contract':CONTRACT,'workers':self.workers,'requested_simulations':simulations,
                    'budgets':budgets,'seeds':seeds,'actual_root_visits':int(info['visit_counts'].sum()),
                    'startup_seconds':self.startup_seconds,'search_wall_seconds':time.perf_counter()-start,
                    'worker_search_seconds':[r[4] for r in responses],'checkpoint_sha256':self.hashes,
                    'worker_diagnostics':[r[7] for r in responses],
                    'tree_reuse':False,'root_noise':False})
            return info['root_actions'][idx],info
        except BaseException:
            self.close();raise

    def close(self):
        if self.closed:return
        self.closed=True
        for conn in self.commands:
            try:conn.send(None)
            except (OSError,EOFError):pass
        from served_lifecycle import cleanup
        self.cleanup_report=cleanup(self.processes,self.server,self.control,join_seconds=2,kill_seconds=2)
        for conn in self.commands+([self.control] if self.control is not None else []):conn.close()

    def __exit__(self,*args):self.close()
