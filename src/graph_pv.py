"""Private bounded CUDA Graph PV cache, exact inputs and full valid menus."""
from collections import OrderedDict
import time
import numpy as np
import torch
from resident_inputs import ResidentInputs


class GraphCacheBudgetExceeded(ValueError):
    pass


class GraphPV:
    def __init__(self,net,*,max_graphs=32,max_padded_rows=32768,max_cached_bytes=512*1024*1024):
        self.net=net;self.device=next(net.parameters()).device
        if self.device.type!='cuda':raise ValueError('CUDA graph requires CUDA net')
        if net.arch_kind!='gnn' or net.arch_name!='base':raise ValueError('Only reviewed base GNN supported')
        if torch.are_deterministic_algorithms_enabled():raise RuntimeError('Installed deterministic scatter_add does not support graph capture')
        self.math=self._math_settings()
        if max_graphs<1 or max_padded_rows<1:raise ValueError('Positive bounds required')
        self.max_graphs=max_graphs;self.max_rows=max_padded_rows
        self.max_bytes=int(max_cached_bytes);self.cached_bytes=0;self.peak_cached_bytes=0
        self.capture_seconds=0.;self.valid_rows=0;self.padded_rows=0
        if self.max_bytes<=0:raise ValueError('Positive memory budget required')
        self.graphs=OrderedDict();self.captures=self.replays=self.evictions=0
        self.parameters=tuple(net.parameters());self.buffers=tuple(net.buffers())
        self.versions=tuple(t._version for t in self.parameters+self.buffers)
        self.storage=tuple(t.data_ptr() for t in self.parameters+self.buffers)
        self.layouts=tuple((module,tuple(module._parameters.items()),tuple(module._buffers.items()),
                            tuple(module._modules.items())) for module in net.modules())
        if any(t.dtype!=torch.float32 for t in self.parameters):raise TypeError('Float32 weights required')

    def guard(self):
        if any(m.training for m in self.net.modules()):raise RuntimeError('Eval only')
        if self._math_settings()!=self.math or torch.is_autocast_enabled():raise RuntimeError('Capture math settings changed')
        for module,parameters,buffers,children in self.layouts:
            for current,expected in ((module._parameters,parameters),(module._buffers,buffers),(module._modules,children)):
                if len(current)!=len(expected) or any(current.get(name) is not value for name,value in expected):
                    raise RuntimeError('Model parameter/buffer/module identity changed; recreate CUDA graphs')
        params=tuple(self.net.parameters());bufs=tuple(self.net.buffers())
        if (len(params)!=len(self.parameters) or len(bufs)!=len(self.buffers)
                or any(a is not b for a,b in zip(params+bufs,self.parameters+self.buffers))
                or tuple(t._version for t in params+bufs)!=self.versions
                or tuple(t.data_ptr() for t in params+bufs)!=self.storage):
            raise RuntimeError('Model revision changed; recreate CUDA graphs')

    @staticmethod
    def _math_settings():
        return (torch.are_deterministic_algorithms_enabled(),torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,torch.get_float32_matmul_precision())

    def close(self):
        self.graphs.clear();self.cached_bytes=0

    def stats(self):
        return dict(captures=self.captures,replays=self.replays,evictions=self.evictions,
                    capture_seconds=self.capture_seconds,live_graphs=len(self.graphs),
                    accounted_cached_bytes=self.cached_bytes,peak_accounted_bytes=self.peak_cached_bytes,
                    max_cached_bytes=self.max_bytes,max_graphs=self.max_graphs,
                    valid_rows=self.valid_rows,padded_rows=self.padded_rows,
                    resident_input_stats=[entry['resident_inputs'].stats() for entry in self.graphs.values()])

    def _evict(self):
        _,old=self.graphs.popitem(last=False)
        self.cached_bytes-=old['accounted_bytes'];self.evictions+=1

    @staticmethod
    def _estimated_bytes(batch,bucket):
        # Conservative base-model tensor/workspace allowance, not a claim that
        # CUDA allocator reservation exactly equals live graph tensor bytes.
        return 1024*1024 + batch*4*1024*1024 + batch*bucket*8192

    def _capture(self,bucket,batch,dim,adim,delta):
        import ai
        device=self.device
        x=torch.zeros((batch,dim),device=device)
        enc=torch.zeros((batch,bucket,adim),device=device)
        enc[:,:,33]=-1
        region=torch.zeros((batch,bucket),device=device,dtype=torch.long)
        count=torch.zeros(batch,device=device,dtype=torch.long)
        owner=torch.arange(batch,device=device).repeat_interleave(bucket)
        column=torch.arange(bucket,device=device).unsqueeze(0)
        card_pad=torch.tensor(ai.CARD_PAD_IDX,device=device)
        def calculate():
            # Identical base _process_input arithmetic, with its CPU-created
            # padding scalar hoisted out of capture. No graph/model weight edits.
            spatial=x[:,:self.net._spatial_dim].reshape(batch,ai.NUM_REGIONS,ai.NODE_FEAT_DIM)
            gnn,nodes=self.net.board_gnn(spatial)
            raw=x[:,ai._CARD_IDX_START:ai._CARD_IDX_END]
            cidx=raw.long();cidx=torch.where(cidx<0,card_pad,cidx).clamp(0,ai.CARD_PAD_IDX)
            embedded=self.net.card_embedding(cidx)
            mask=(raw>=0).unsqueeze(-1).float()
            card_out=self.net.card_proj((embedded*mask).sum(1)/mask.sum(1).clamp(min=1.0))
            globals_=torch.cat([x[:,self.net._spatial_dim:ai._CARD_IDX_START],x[:,ai._CARD_IDX_END:]],dim=1)
            bb=self.net.backbone(torch.cat([gnn,card_out,globals_],dim=1))
            value=self.net.value_head(bb).squeeze(-1)
            ae=enc.reshape(-1,adim);ri=region.reshape(-1)
            features=torch.cat([ae[:,:ai._ACTION_CARD_IDX_DIM],ae[:,ai._ACTION_CARD_IDX_DIM+1:]],dim=-1)
            cidx=ae[:,ai._ACTION_CARD_IDX_DIM].long()
            cidx=torch.where(cidx<0,card_pad,cidx).clamp(0,ai.CARD_PAD_IDX)
            cards=self.net.card_embedding(cidx)
            mean=nodes.mean(dim=1)
            gathered=torch.where((ri<0).unsqueeze(-1),mean[owner],nodes[owner,ri.clamp(min=0)])
            combined=torch.cat([bb[owner],gathered,features,cards],dim=1)
            logits=self.net.spatial_action_scorer(combined).squeeze(-1)
            if delta:logits=logits+self.net.policy_delta(combined).squeeze(-1)
            logits=logits.reshape(batch,bucket)
            valid=column<count.unsqueeze(1)
            masked=logits.masked_fill(~valid,float('-inf'))
            # Empty menus use a finite dummy softmax; all dummy outputs are zero.
            masked=torch.where((count==0).unsqueeze(1),torch.zeros_like(masked),masked)
            probs=torch.softmax(masked,dim=1).masked_fill(~valid,0.)
            return torch.cat([value,probs.reshape(-1)])
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):calculate()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph),torch.no_grad():output=calculate()
        self.captures+=1
        return dict(graph=graph,x=x,enc=enc,region=region,count=count,output=output,
                    owner=owner,column=column,card_pad=card_pad,
                    resident_inputs=ResidentInputs(batch,bucket,dim,adim))

    def eval_values(self, vectors):
        """Raw values using the same graph with zero valid action rows.

        Reuses reviewed capture/masking/lifetime logic. No value shaping here.
        """
        vectors=np.asarray(vectors)
        if vectors.dtype!=np.float32 or vectors.ndim!=2 or vectors.shape[1]!=self.net.input_dim:
            raise ValueError('Value batch dimensions or dtype')
        if not np.isfinite(vectors).all():raise ValueError('Nonfinite value batch')
        if len(vectors)==0:
            self.guard()
            return np.empty(0,dtype=np.float32)
        adim=(self.net.spatial_action_scorer[0].in_features
              -self.net.backbone[4].out_features
              -self.net.board_gnn.layers[-1].linear.out_features
              -self.net.card_embedding.embedding_dim+1)
        empty=np.empty((0,adim),dtype=np.float32)
        regions=np.empty(0,dtype=np.int64)
        results=self.pv_many([(vec,empty,regions) for vec in vectors],with_delta=False)
        return np.asarray([value for _priors,value in results],dtype=np.float32)

    def pv_many(self,items,with_delta=True):
        self.guard()
        if not items:return []
        counts=[len(a) for v,a,r in items]
        batch=len(items);maximum=max(counts)
        bucket=1 << max(0,maximum-1).bit_length()
        if batch*bucket>self.max_rows:raise ValueError('Graph padded row budget exceeded')
        dim=self.net.input_dim
        # Action layout uses the frozen model's embedding convention.
        actual_adim=int(np.asarray(items[0][1]).shape[1])
        expected_adim=(self.net.spatial_action_scorer[0].in_features
                       -self.net.backbone[4].out_features
                       -self.net.board_gnn.layers[-1].linear.out_features
                       -self.net.card_embedding.embedding_dim+1)
        if actual_adim!=expected_adim:raise ValueError('Action encoding width')
        for i,(v,a,r) in enumerate(items):
            v=np.asarray(v);a=np.asarray(a);r=np.asarray(r)
            if v.dtype!=np.float32 or a.dtype!=np.float32:raise TypeError('Float32 request required')
            if v.shape!=(dim,) or a.shape!=(counts[i],actual_adim) or r.shape!=(counts[i],):raise ValueError('Request dimensions')
            if r.dtype.kind not in 'iu' or np.any(r < -1) or np.any(r >= self.net.board_gnn.num_nodes):raise ValueError('Invalid region')
            if not np.isfinite(v).all() or not np.isfinite(a).all():raise ValueError('Nonfinite request')
        key=(batch,bucket,dim,actual_adim,bool(with_delta))
        entry=self.graphs.get(key)
        if entry is None:
            estimate=self._estimated_bytes(batch,bucket)
            if estimate>self.max_bytes:raise GraphCacheBudgetExceeded('Single graph exceeds memory budget')
            while self.graphs and (len(self.graphs)>=self.max_graphs or self.cached_bytes+estimate>self.max_bytes):self._evict()
            allocated=torch.cuda.memory_allocated(self.device)
            reserved=torch.cuda.memory_reserved(self.device)
            start=time.perf_counter()
            entry=self._capture(bucket,batch,dim,actual_adim,bool(with_delta))
            torch.cuda.synchronize(self.device)
            self.capture_seconds+=time.perf_counter()-start
            accounted=max(estimate,torch.cuda.memory_allocated(self.device)-allocated,
                          torch.cuda.memory_reserved(self.device)-reserved)
            if accounted>self.max_bytes:
                del entry
                raise GraphCacheBudgetExceeded('Captured graph exceeds memory budget')
            while self.graphs and self.cached_bytes+accounted>self.max_bytes:self._evict()
            entry['accounted_bytes']=accounted
            self.graphs[key]=entry;self.cached_bytes+=accounted
            self.peak_cached_bytes=max(self.peak_cached_bytes,self.cached_bytes)
        self.valid_rows+=sum(counts);self.padded_rows+=batch*bucket
        self.graphs.move_to_end(key)
        with torch.no_grad():
            try:
                for field,index,value in entry['resident_inputs'].update(items):
                    entry[field][index].copy_(torch.from_numpy(value),non_blocking=True)
            except BaseException:
                # A partially uploaded mirror must never be trusted again.
                try:
                    torch.cuda.synchronize(self.device)
                finally:
                    self.graphs.pop(key);self.cached_bytes-=entry['accounted_bytes']
                raise
            entry['graph'].replay();self.replays+=1
            host=entry['output'].cpu().numpy()
        return [(host[batch+i*bucket:batch+i*bucket+n].copy(),float(host[i])) for i,n in enumerate(counts)]
