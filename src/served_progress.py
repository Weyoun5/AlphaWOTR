"""Throttled concrete worker progress; no synthetic heartbeat or total wall cap."""
import functools
import math
import os
import time
import json


def stall_seconds():
    value=float(os.environ.get('WOTR_SERVED_STALL_SEC','1800'))
    if not math.isfinite(value) or value<1800:
        raise ValueError('WOTR_SERVED_STALL_SEC must be finite and at least 1800 seconds')
    return value


class ProgressEmitter:
    def __init__(self,worker,queue,*,clock=time.monotonic,interval=10.0):
        self.worker,self.queue,self.clock,self.interval=worker,queue,clock,interval
        self.sequence=0;self.last=None

    def completed(self,stage,*,force=False):
        self.sequence+=1
        now=self.clock()
        if force or self.last is None or now-self.last>=self.interval:
            self.queue.put(('p',self.worker,{'sequence':self.sequence,'stage':stage}))
            self.last=now

    def install(self,ai):
        # Only successful work advances the sequence. No RNG, inference or
        # helper thread is introduced by these observation-only wrappers.
        originals={}
        for name in ('mcts_search','_traced_execute'):
            fn=getattr(ai,name);originals[name]=fn
            def wrap(fn=fn,name=name):
                @functools.wraps(fn)
                def run(*args,**kwargs):
                    result=fn(*args,**kwargs)
                    self.completed(name)
                    return result
                return run
            setattr(ai,name,wrap())
        return originals


class ProgressWatch:
    def __init__(self,timeout,*,clock=time.monotonic):
        if not math.isfinite(timeout) or timeout<=0:raise ValueError('Invalid stall timeout')
        self.timeout,self.clock=timeout,clock
        self.workers={}
        self.last_publish=None

    def started(self,wid):
        if wid in self.workers:raise ValueError('Worker progress registered twice')
        self.workers[wid]={'sequence':0,'time':self.clock(),'stage':'initializing'}

    def accept(self,wid,payload):
        if type(wid) is not int or wid not in self.workers or not isinstance(payload,dict):raise ValueError('Unknown worker progress')
        seq=payload.get('sequence');stage=payload.get('stage')
        if type(seq) is not int or seq<1 or not isinstance(stage,str) or not stage:
            raise ValueError('Malformed worker progress')
        old=self.workers[wid]
        if seq<=old['sequence']:raise ValueError('Worker progress did not advance')
        self.workers[wid]={'sequence':seq,'time':self.clock(),'stage':stage}

    def check(self,active):
        now=self.clock()
        for wid in active:
            row=self.workers[wid]
            if now-row['time']>=self.timeout:
                raise RuntimeError('Served worker %d made no concrete search/action progress for %.1fs (last=%s, sequence=%d)' %
                    (wid,now-row['time'],row['stage'],row['sequence']))

    def publish(self,path,phase_id,active,*,force=False):
        now=self.clock()
        if not force and self.last_publish is not None and now-self.last_publish<10:return
        payload={'schema':'served_worker_progress_v1','phase_id':phase_id,
            'updated_at':time.time(),'stall_seconds':self.timeout,
            'workers':{str(w):dict(sequence=r['sequence'],stage=r['stage'],
                seconds_since_progress=max(0,now-r['time']),active=w in active)
                for w,r in self.workers.items()}}
        with open(str(path)+'.tmp','w',encoding='utf-8') as stream:json.dump(payload,stream)
        for attempt in range(25):
            try:os.replace(str(path)+'.tmp',path);break
            except PermissionError:
                if attempt==24:raise
                time.sleep(.02)
        self.last_publish=now
