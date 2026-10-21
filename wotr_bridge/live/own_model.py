"""Live root-only search using the established own-value reply reference."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import time


class OwnModelSearch:
    def __init__(self, side, net, *, seed=None, journal=None, backend=None):
        import ai
        from reply_evaluation import validate_root_context
        self.side=side;self.net=net;self.profile=net.reward_profile
        with __import__('own_search').own_value_context(ai,side,net,profile=self.profile):
            self.tag=validate_root_context(ai,side,net)
        self.seed=secrets.randbits(64) if seed is None else int(seed) % (1 << 64)
        self.index=0;self.journal=Path(journal) if journal else None
        self.backend=backend or os.environ.get('WOTR_LIVE_SEARCH_BACKEND','python')
        self.native=None;self.backend_identity={'backend':self.backend}
        if self.backend=='native':
            from .native_search import load
            self.native,self.backend_identity=load(ai)
        elif self.backend!='python':
            raise ValueError('Unknown live search backend: '+self.backend)

    def choose(self, state, side, *, num_simulations=300):
        import ai,own_search,rng_streams
        if side is not self.side:
            raise ValueError('Own-model live search belongs to its configured seat')
        index=self.index;self.index+=1
        seed=rng_streams.derive_seed(self.seed,'bridge-own-reply:%d' % index)
        record={'schema':'root_only_search_call_v1','contract':own_search.MODE,
                'root_side':self.tag,'profile':self.profile,'search_index':index,
                'reply_seed':seed,'reply_panels':2,'belief_worlds':2,
                'leaf_batch':1,'tree_reuse':False,'simulations_requested':int(num_simulations),
                'model_sha256':getattr(self.net,'checkpoint_sha256',None),
                'backend':self.backend,
                'controller_binary_sha256':self.backend_identity.get('controller_binary_sha256'),
                'opponent_model_loaded':False,'passed':False}
        from .inference_cache import cached_inference
        from .policy_thread_scope import policy_thread_scope
        started=time.perf_counter();context=None;inference_stats={};policy_stats={}
        try:
            with cached_inference(self.net,enabled=os.environ.get('WOTR_LIVE_INFERENCE_CACHE','0')=='1') as inference_stats, policy_thread_scope(self.net) as policy_stats, own_search.active(ai,side,self.net,profile=self.profile,seed=seed,
                                   reply_panels=2,belief_worlds=2) as context:
                fp,sh=ai._SEARCH_NETS
                # Retain the live engine's advancing search RNG. The standalone
                # audit reference_rng reset is deliberately absent here.
                if self.native is None:
                    result=ai.mcts_action_with_policy(state,side,fp,sh,
                                num_simulations=num_simulations,temperature=0.0,add_noise=False)
                else:
                    context.native_seed=seed
                    result=self.native.native_mcts_search(ai,state,side,fp,sh,
                                num_simulations=num_simulations,temperature=0.0,add_noise=False)
                    if result is None:
                        result=(None,None)
                record['passed']=True
            return result
        except BaseException as error:
            record['error_type']=type(error).__name__
            raise
        finally:
            record['seconds']=time.perf_counter()-started
            record['inference_cache']=dict(inference_stats)
            record['policy_inference']=dict(policy_stats)
            record['counts']=dict(context.stats) if context is not None else {}
            if record['passed'] and result is not None and result[1] is not None:
                if self.native is not None:
                    record['counts']=dict(result[1].get('own_value_search',{}).get('stats',{}))
                result[1]['own_value_search']=dict(record)
            if self.journal:
                self.journal.parent.mkdir(parents=True,exist_ok=True)
                with self.journal.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,sort_keys=True)+'\n')


def load_own_net(tag, side):
    """Open only the configured seat's file; never overwrite stored seat metadata."""
    import ai
    threads=os.environ.get('WOTR_LIVE_INFERENCE_THREADS')
    if threads is not None:
        ai.torch.set_num_threads(int(threads))
    from reward_contract import validate_net_profile,validate_profile
    if side not in (ai.Side.FREE_PEOPLES,ai.Side.SHADOW):raise ValueError('Invalid AI seat')
    seat='fp' if side is ai.Side.FREE_PEOPLES else 'sh'
    choices=(tag,) if tag!='auto' else ('corpus_live','corpus_v41jt','corpus_v41td','corpus_v41','corpus_v37jt','corpus_v37td','corpus_v37')
    path=next((Path(ai._resolve_model_path('wotr_%s_%s.pt' % (seat,choice)))
               for choice in choices if Path(ai._resolve_model_path('wotr_%s_%s.pt' % (seat,choice))).is_file()),None)
    if path is None:raise FileNotFoundError('No checkpoint for the configured AI seat '+seat)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    net=ai.load_value_net(str(path),side=None)
    if net is None:raise RuntimeError('AI checkpoint did not load')
    if net.side!=seat:raise ValueError('Checkpoint stored seat differs from configured AI seat')
    validate_profile(seat,net.reward_profile);validate_net_profile(net,seat,net.reward_profile)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError('Checkpoint changed during load')
    net.checkpoint_sha256=before;net.checkpoint_path=str(path)
    ai.PLAYER_PROFILES[seat]=net.reward_profile
    return net
