"""Live root-only search using the established own-value reply reference."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import time


def _one_line(value, limit=600):
    return ' '.join(str(value).split())[:limit]


def search_failure_summary(ai, error, *, strict=False):
    """R2-SL-27: a bounded, JSON-safe digest of ai._SEARCH_FAILURE_DIAGNOSTIC.

    That packet is what ai._search_failure_scope captures when a search
    raises; the native route runs under the same decorator chain. The digest
    keeps the error, the frame list and capture errors only: no states, no
    sampled hidden overlay, no tree. strict: only a packet whose error text is
    this error's, for a caller that did not clear the packet before the search
    (an older search's failure must never be attached to this one).
    """
    packet = getattr(ai, '_SEARCH_FAILURE_DIAGNOSTIC', None)
    if not isinstance(packet, dict):
        return None
    text = packet.get('error')
    if strict and text != '%s: %s' % (type(error).__name__, error):
        return None
    out = {'schema': str(packet.get('schema')),
           'error': _one_line(text) if text is not None else None,
           'frames': ['%s:%s' % (frame.get('function'), frame.get('line'))
                      for frame in (packet.get('frames') or ()) if isinstance(frame, dict)][-24:],
           'capture_errors': [_one_line(item, 200) for item in (packet.get('capture_errors') or ())][:8]}
    node = packet.get('node')
    if isinstance(node, dict):
        out['node'] = {str(key): str(value)[:60] for key, value in node.items()}
    action = packet.get('action')
    if isinstance(action, dict):
        out['action_type'] = str(action.get('action_type'))[:60]
    return out


def traceback_failure_summary(error):
    """R3-LIVE-SEARCH-ROOT-CAUSE: the same bounded digest from the exception's
    own traceback, for a failure raised before the search's failure scope
    opened. The native root is packed by orchestrator._resolve_root
    (snapshot.pack_snapshot) before the decorated search runs, so a refused
    root (e.g. an unregistered field) leaves no ai._SEARCH_FAILURE_DIAGNOSTIC.
    Frames are 'file:function:line', innermost last; no locals, no states."""
    import traceback
    frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ is not None else []
    return {'schema': 'own_search_failure_traceback_v1',
            'error': _one_line('%s: %s' % (type(error).__name__, error)),
            'frames': ['%s:%s:%s' % (Path(frame.filename).name, frame.name, frame.lineno)
                       for frame in frames][-24:],
            'outside_search_scope': True}


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
            # R2-MODEL-CANDIDATES (X4-02): only numeric arrays cross the native
            # inference ABI (no action kinds, no chance rows), so the native
            # route cannot apply a net's trained self-play policy modules.
            # Refuse at startup instead of serving the projection (or failing
            # at the first inference); the Python backend applies them.
            modules=ai.selfplay_policy_modules(net)
            if modules:
                raise ValueError('The native live search cannot apply this net\'s trained self-play policy modules %s; '
                                 'use WOTR_LIVE_SEARCH_BACKEND=python or a checkpoint without them' % (list(modules),))
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
                'native_boundary_sha256':self.backend_identity.get('boundary_identity_sha256'),
                'opponent_model_loaded':False,'passed':False}
        # VNETINSTALL: the seat's leaf values and PUCT constant on every receipt.  A seat with a bound value net passes
        # its own c_puct to the search; None = the call exactly as before (the engine default, ai._C_PUCT)
        c_puct=getattr(self.net,'search_c_puct',None)
        cpuct_kw={} if c_puct is None else {'c_puct':c_puct}
        record['search_value']=search_value_record(self.net)
        record['c_puct']=float(ai._C_PUCT) if c_puct is None else float(c_puct)
        from .inference_cache import cached_inference
        from .policy_thread_scope import policy_thread_scope
        started=time.perf_counter();context=None;inference_stats={};policy_stats={}
        # VALUENET: a bound value model gets its own board cache scope and a receipt record
        vm=getattr(self.net,'value_model',None);vm_cache={}
        vm_before=(vm.calls,vm.rows) if vm is not None else None
        # R2-SL-27: a failure packet present after this call is this call's
        # (the search scope also clears it on entry; a raise before entry
        # must not pick up an older search's packet).
        ai._SEARCH_FAILURE_DIAGNOSTIC=None
        try:
            with cached_inference(self.net,enabled=os.environ.get('WOTR_LIVE_INFERENCE_CACHE','0')=='1') as inference_stats, cached_inference(vm.model if vm is not None else None,enabled=vm is not None and os.environ.get('WOTR_LIVE_INFERENCE_CACHE','0')=='1') as vm_cache, policy_thread_scope(self.net) as policy_stats, own_search.active(ai,side,self.net,profile=self.profile,seed=seed,
                                   reply_panels=2,belief_worlds=2) as context:
                fp,sh=ai._SEARCH_NETS
                # Retain the live engine's advancing search RNG. The standalone
                # audit reference_rng reset is deliberately absent here.
                if self.native is None:
                    result=ai.mcts_action_with_policy(state,side,fp,sh,
                                num_simulations=num_simulations,temperature=0.0,add_noise=False,**cpuct_kw)
                else:
                    context.native_seed=seed
                    result=self.native.native_mcts_search(ai,state,side,fp,sh,
                                num_simulations=num_simulations,temperature=0.0,add_noise=False,**cpuct_kw)
                    if result is None:
                        result=(None,None)
                record['passed']=True
            return result
        except BaseException as error:
            record['error_type']=type(error).__name__
            # R2-SL-27: the cause itself survives console rotation (the native
            # controller/orchestrator error is this exception's text).
            record['error_message']=_one_line(error)
            try:
                summary=search_failure_summary(ai,error)
                if summary is None:
                    # R3-LIVE-SEARCH-ROOT-CAUSE: raised before the search scope
                    # (native root packing): digest this exception's traceback
                    summary=traceback_failure_summary(error)
            except Exception as summary_error:
                summary={'summary_error':'%s: %s' % (type(summary_error).__name__,_one_line(summary_error,200))}
            if summary is not None:
                record['search_failure']=summary
            raise
        finally:
            record['seconds']=time.perf_counter()-started
            record['inference_cache']=dict(inference_stats)
            record['policy_inference']=dict(policy_stats)
            record['counts']=dict(context.stats) if context is not None else {}
            if vm is not None:
                record['value_model']={'sha256':vm.sha256,'name':vm.name,'calls':vm.calls-vm_before[0],
                                       'rows':vm.rows-vm_before[1],'board_cache':dict(vm_cache)}
            if record['passed'] and result is not None and result[1] is not None:
                if self.native is not None:
                    record['counts']=dict(result[1].get('own_value_search',{}).get('stats',{}))
                result[1]['own_value_search']=dict(record)
            if self.journal:
                self.journal.parent.mkdir(parents=True,exist_ok=True)
                with self.journal.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,sort_keys=True)+'\n')


def seat_seed(seed, side):
    """R3-AI-VS-AI-SERVER: one seat's search seed in AI vs AI, derived from
    the server seed and the seat name (None keeps OwnModelSearch's own
    secrets.randbits per seat)."""
    if seed is None:
        return None
    import rng_streams
    return rng_streams.derive_seed(int(seed) % (1 << 64),
                                   'live-seat-search:%s' % side.name)


class SeatSearches:
    """R3-AI-VS-AI-SERVER (USER X1-07 2026-10-23 "AI vs AI watch should work
    properly."): the AI vs AI server's searches, one per seat.

    choose() routes by the side the ENGINE names as deciding (the caller's
    describe_pending / ai_action_dp side, never a request header) to that
    seat's own search; each seat's search runs exactly the sided route
    (OwnModelSearch -> own_search.active(ai, side, ...)), so its belief worlds
    come from that side's information set only (1.261).  `seed` / `index`
    read and restore every seat's counters as {side name: value} (the
    turn_rewind snapshot treats the server's own_search like one seat's)."""

    def __init__(self, searches):
        self.searches = dict(searches)
        names = sorted(getattr(side, 'name', side) for side in self.searches)
        if names != ['FREE_PEOPLES', 'SHADOW']:
            raise ValueError('AI vs AI needs one search per seat, got %r' % (names,))
        for side, search in self.searches.items():
            configured = getattr(search, 'side', side)
            if configured is not side:
                raise ValueError('The %s seat search is configured for %r'
                                 % (side.name, configured))

    def __getitem__(self, side):
        return self.searches[side]

    @property
    def backend(self):
        return ', '.join('%s %s' % (side.name, getattr(search, 'backend', '?'))
                         for side, search in sorted(self.searches.items(),
                                                    key=lambda item: item[0].name))


    def choose(self, state, side, *, num_simulations=300):
        search = self.searches.get(side)
        if search is None:
            raise ValueError('AI vs AI search has no seat %r' % (side,))
        return search.choose(state, side, num_simulations=num_simulations)

    def _counters(self, name):
        return {side.name: getattr(search, name)
                for side, search in self.searches.items()}

    def _restore(self, name, value):
        for side, search in self.searches.items():
            setattr(search, name, value[side.name])

    seed = property(lambda self: self._counters('seed'),
                    lambda self, value: self._restore('seed', value))
    index = property(lambda self: self._counters('index'),
                     lambda self, value: self._restore('index', value))


def load_own_net(tag, side):
    """Open only the configured seat's file; never overwrite stored seat metadata."""
    import ai
    threads=os.environ.get('WOTR_LIVE_INFERENCE_THREADS')
    if threads is not None:
        ai.torch.set_num_threads(int(threads))
    from reward_contract import validate_net_profile,validate_profile
    if side not in (ai.Side.FREE_PEOPLES,ai.Side.SHADOW):raise ValueError('Invalid AI seat')
    seat='fp' if side is ai.Side.FREE_PEOPLES else 'sh'
    # 'auto': the deployed pair of the running generation first (src/net_tags.py)
    from net_tags import auto_tags
    choices=(tag,) if tag!='auto' else auto_tags(getattr(ai,'_OBSERVATION_GENERATION',''))
    path=next((Path(ai._resolve_model_path('wotr_%s_%s.pt' % (seat,choice)))
               for choice in choices if Path(ai._resolve_model_path('wotr_%s_%s.pt' % (seat,choice))).is_file()),None)
    if path is None:raise FileNotFoundError('No checkpoint for the configured AI seat '+seat)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    net=ai.load_value_net(str(path),side=None)
    if net is None:raise RuntimeError('AI checkpoint did not load')
    if net.side!=seat:raise ValueError('Checkpoint stored seat differs from configured AI seat')
    validate_profile(seat,net.reward_profile);validate_net_profile(net,seat,net.reward_profile)
    # INV-RING-1: an all-zero final value layer makes every search leaf the
    # same constant (the shipped Shadow net: every root q exactly 0, so the
    # seat just plays its top prior). Refuse it; never serve a dead critic.
    # (R2-MODEL-CANDIDATES: one shared helper, also used by the engine-only
    # EngineSession AI seats.)
    ai.assert_state_dependent_value_head(net,path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError('Checkpoint changed during load')
    net.checkpoint_sha256=before;net.checkpoint_path=str(path)
    ai.PLAYER_PROFILES[seat]=net.reward_profile
    # VNETINSTALL: FP-WIN / SH-WIN get their value net and c_puct 1.5 (refused if the file is missing or changed)
    bind_search_value(net,seat)
    return net


# VNETINSTALL (2026-11-03; kit claude_bugscan_2026-10-21/vnetinstall_1103).  The search of the FP-WIN and SH-WIN nets
# reads every non-terminal leaf value from a separately trained value network (hpc/remote/league/valuelab, run
# valuelab_20261102: the full WOTRValueNet trained on FP-WIN c200 vs SH-WIN c200 game outcomes, i.e. the WIN objective)
# and searches with c_puct 1.5 instead of the engine default 4.0.  Paired cluster games at 300 simulations (runs vnet3 and
# vnet4, whole seeds resampled): FP +44 Elo [+26, +61], SH +37 [+20, +55] over the old search.  The binding is keyed by
# the seat checkpoint's sha256, so it follows the net, never the tag.  FP-MILITARY, SH-CORRUPTION and every other net
# (D6, corpus) keep their own value head and the default c_puct: the value nets learnt the win objective, not theirs.
# Priors always come from the seat net.  The switch is ai.bind_search_value_model (VALUENET) plus net.search_c_puct.
# LF050 (2026-11-03; kit claude_bugscan_2026-10-21/lfinstall_1103): the full-network league_full c050 WIN nets (FP
# 776d5d5e, SH 2eb37451; every layer trained, 500K learning games) bind the SAME value nets at c_puct 1.5 - the
# configuration the cluster evaluated (runs/league_full_20261103 s300: lf c050 vs lm c200 FP +29 Elo [-5, +63], SH +38
# [+4, +75]).  The c200 entries stay, so the lg200_* pairs bind exactly as before.
# LF201 (2026-11-04; kit claude_bugscan_2026-10-21/lfinstall_1104): the same league trained on to cycle 201 (2M
# learning games; FP 17dcfcc2, SH d5f7a332) binds the SAME value nets at c_puct 1.5, as the cluster evaluated it
# (runs/league_full_20261103 eval s300: lf c201 vs lm c200 FP +14 Elo [-17, +44], SH +77 [+42, +118]; vs lf c050
# FP +1 [-34, +32], SH +40 [+9, +75]).  The c050 and c200 entries stay, so their pairs bind exactly as before.
SEARCH_VALUE_NETS={
    'fp':{'e87e0c9bb8bd3654c9b0ba9c6840a0fcdfdfece9f6161949b3e967131467c956':
          {'net':'FP-WIN','file':'value_nets/fp_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'61a84901822582812d59c015d371f0baeea9141eb35d5e8ee5bd55d5905d102e'},
          '776d5d5e0bc732c6682dd319b89a9e0105feeaf6c9f3705748ceafa2883adf5c':
          {'net':'FP-WIN full c050','file':'value_nets/fp_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'61a84901822582812d59c015d371f0baeea9141eb35d5e8ee5bd55d5905d102e'},
          '17dcfcc2c9b789c248bcb2a12a8a57eada0d068a3c6d53656e3907b357f450a8':
          {'net':'FP-WIN full c201','file':'value_nets/fp_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'61a84901822582812d59c015d371f0baeea9141eb35d5e8ee5bd55d5905d102e'},
          'd67c7072d7ea704f0ad19b710ae2abd47ecc02863ed74facd4b1793455394751':
          {'net':'FP-WIN search-trained c008','file':'value_nets/fp_st008_judge.pt','c_puct':1.5,
           'sha256':'3061e54a4f8d6d99c06ddc5ad81c05cf0e54d7e93297b57416b3bbfc74b6d883'},
          # the same net as the file it is published as (every tensor identical; the record of where it was trained
          # names no folder of the training machine): another file, so another key
          '3c13d2892f6fb2c6240ce5ddcf20bfea7ada2dfacc23ac0f3d4d25059d300f6b':
          {'net':'FP-WIN search-trained c008','file':'value_nets/fp_st008_judge.pt','c_puct':1.5,
           'sha256':'3061e54a4f8d6d99c06ddc5ad81c05cf0e54d7e93297b57416b3bbfc74b6d883'}},
    'sh':{'785f199185ca9a80ea51c0f6375c0ce8e9e4eed2d1bf5e9679fc73f924309109':
          {'net':'SH-WIN','file':'value_nets/sh_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'4c2afe87e180bd852469b5902d116598f173bf9659a344ac95e35137cf8aff35'},
          '2eb3745163662b6b89639878b047853421470db93c287e3fd8294bf79b6487df':
          {'net':'SH-WIN full c050','file':'value_nets/sh_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'4c2afe87e180bd852469b5902d116598f173bf9659a344ac95e35137cf8aff35'},
          'd5f7a3322e9384e3eefb1e7fa5a8605f201d495915257f5123a033811bd15b8c':
          {'net':'SH-WIN full c201','file':'value_nets/sh_full_fresh_mc.pt','c_puct':1.5,
           'sha256':'4c2afe87e180bd852469b5902d116598f173bf9659a344ac95e35137cf8aff35'},
          '0fe9c149c7af0e4e808ba2a8350fa06f040036f69aa1a64379a50f33fd798a37':
          {'net':'SH-WIN search-trained c008','file':'value_nets/sh_st008_judge.pt','c_puct':1.5,
           'sha256':'410d6cf6d807df909bc90ade4fb5078143742218e6fc68fbf0281fcce847901b'},
          'bb3bbcbd44e53eeb3a41252404dc97f792e1a6fdad0c2bbaf0ef5a1b4635f982':       # (as published: see 'fp')
          {'net':'SH-WIN search-trained c008','file':'value_nets/sh_st008_judge.pt','c_puct':1.5,
           'sha256':'410d6cf6d807df909bc90ade4fb5078143742218e6fc68fbf0281fcce847901b'}},
}
SEARCH_VALUE_NOTE='value nets are bound to FP-WIN and SH-WIN only (they learnt the win objective)'


def search_value_record(net):
    """The receipt record of what a net's search reads its leaf values from, and its c_puct."""
    import ai
    record=getattr(net,'search_value',None)
    if record is not None:
        return dict(record)
    vm=getattr(net,'value_model',None)
    if vm is not None:      # bound outside load_own_net (a test harness): described as found
        c_puct=getattr(net,'search_c_puct',None)
        return {'source':'value net','net':None,'value_net':vm.path,'value_sha256':vm.sha256,'value_name':vm.name,
                'c_puct':float(ai._C_PUCT) if c_puct is None else float(c_puct)}
    return {'source':'own value head','c_puct':float(ai._C_PUCT),'value_net':None,'note':SEARCH_VALUE_NOTE}


def bind_search_value(net,seat):
    """Bind SEARCH_VALUE_NETS[seat][the net's checkpoint sha256] (value net + c_puct) to this net, or record that it
    searches with its own head.  A bound net whose value file is missing or has another sha256 is refused: no fallback."""
    import ai
    spec=SEARCH_VALUE_NETS.get(seat,{}).get(getattr(net,'checkpoint_sha256',None))
    label=os.path.basename(getattr(net,'checkpoint_path','') or '') or seat
    if spec is None:
        net.search_c_puct=None
        net.search_value={'source':'own value head','c_puct':float(ai._C_PUCT),'value_net':None,'note':SEARCH_VALUE_NOTE}
        print('  [search value] %s: its own value head, c_puct %g (%s)' % (label,ai._C_PUCT,SEARCH_VALUE_NOTE),flush=True)
        return dict(net.search_value)
    path=Path(__file__).resolve().parents[2]/spec['file']
    if not path.is_file():
        raise FileNotFoundError('The %s search value net is missing: %s (the install is incomplete: roll it back or '
                                'reinstall; there is no fallback to the old value head)' % (spec['net'],path))
    model=ai.load_search_value_model(path,seat,expected_sha256=spec['sha256'])
    ai.bind_search_value_model(net,model)
    net.search_c_puct=float(spec['c_puct'])
    net.search_value={'source':'value net','net':spec['net'],'value_net':spec['file'],'value_sha256':model.sha256,
                      'value_name':model.name,'c_puct':net.search_c_puct}
    print('  [search value] %s = %s: leaf values from the value net %s (sha256 %s), c_puct %g'
          % (label,spec['net'],spec['file'],model.sha256[:16],net.search_c_puct),flush=True)
    return dict(net.search_value)
