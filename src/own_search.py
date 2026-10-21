"""Isolated Python reference: own-policy MCTS with belief-valued replies.

Version zero deliberately runs sequential leaves and does not reuse old trees.
The outer engine owns game chance and root-hand-conditioned backups. This
module owns a separate, bounded store of opponent-information reply values.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import copy
import random

from reply_evaluation import (evaluate_panel, menu, own_value_context, root_value,
                              transition_rng, validate_root_context)
from reply_selection import ReplySelector
from root_hand import sample_worlds
from portable_reply_identity import (NodeTokens, PolicyKeyEncoder, actions_bytes,
                                    logical_charge, worlds_bytes)

MODE='own_value_reply_v0'


@contextmanager
def reference_rng(ai, seed):
    """Seed a standalone diagnostic search without altering an active episode.

    The engine decorator swaps persistent search streams, so seeding only the
    process RNG would be insufficient under isolated_game_search_v1. Reuse its
    actual seed derivation and scope machinery, restoring all owned stream state.
    Production episode integration will keep its advancing episode search stream.
    """
    import rng_streams
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
        raise ValueError('Reference search seed must be an unsigned 64-bit integer')
    if rng_streams._depth:
        raise RuntimeError('Standalone reference search cannot replace an active RNG scope')
    names=('_episode','_search_python','_search_numpy','_depth','_owner_thread','_search_calls')
    saved={name:getattr(rng_streams,name) for name in names}
    with transition_rng(ai,seed):
        try:
            if rng_streams.current_contract()==rng_streams.ISOLATED:
                rng_streams.start_episode({'rng_episode_seed':seed,
                    'rng_episode_identity':{'purpose':MODE,'standalone_search':True}})
            yield
        finally:
            for name,value in saved.items():setattr(rng_streams,name,value)


class SearchContext:
    def __init__(self, ai, side, net, *, seed, reply_panels, belief_worlds):
        for value in (reply_panels,belief_worlds):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('Reply panels and belief worlds must be positive integers')
        self.ai=ai;self.side=side;self.net=net
        self.panels=reply_panels;self.world_count=belief_worlds
        self.stream=random.Random(seed);self.world_seed=self.stream.getrandbits(128)
        self.cache=OrderedDict();self.cache_bytes=0;self.stats=Counter()
        self.node_tokens=NodeTokens()
        self.action_encoder=PolicyKeyEncoder()

    def value(self, state, previous_actor):
        # Match MCTS's mandatory single-policy closure before its next learned
        # leaf. All transitions remain the real executor and resume functions.
        for _ in range(128):
            result=self.ai.game_logic.check_all_victory_conditions(state)
            if (getattr(state,'_search_endturn_victory',None) is not None
                    or self.ai.game_logic.should_game_end(result,end_of_turn=False)):
                break
            actor=self.ai._next_acting_side(state,previous_actor)
            if actor is None:break
            policy=self.ai.get_ai_policy(state,actor)
            if not policy.forced_reason:break
            # Ordinary leaves need only the forced-policy flag. Building full
            # semantic keys for their often-large menus is discarded immediately.
            # Keep the existing first-row dedup and validation for forced menus.
            from ai_policies import action_policy_key
            actions={}
            for candidate in policy.actions:
                actions.setdefault(action_policy_key(candidate),candidate)
            if len(actions)!=1:raise RuntimeError('Fixed policy requires exactly one action')
            action=next(iter(actions.values()))
            successor=self.ai.apply_action(state,action)
            self.ai._resume_preaction_for_search(state,successor,action)
            state=successor;previous_actor=actor;self.stats['forced_reply_transitions']+=1
        else:raise RuntimeError('Reply fixed-policy closure exceeded 128 transitions')
        self.stats['reply_value_evaluations']+=1
        return root_value(self.ai,state,self.side,self.net)

    def choose(self, state, actor, *, node=None, allowed_keys=None):
        if actor==self.side:raise ValueError('Reply selector requires the opposing actor')
        # A deterministic representative identifies the information set. It is
        # not the Monte Carlo sample used to accumulate this reply's value.
        worlds=sample_worlds(self.ai,state,actor,count=1,seed=self.world_seed)
        self.stats['belief_key_worlds']+=1
        first=None;forced=None
        for world in worlds:
            actions,reason=menu(self.ai,copy.deepcopy(world.state),actor)
            keys=tuple(actions)
            if allowed_keys is not None:
                keys=tuple(key for key in keys if key in allowed_keys)
            if first is None:first=keys;forced=reason
            elif keys!=first or reason!=forced:
                raise ValueError('Actor-conditioned reply worlds disagree on legal choices')
        if not first:return None
        if forced:
            if len(first)!=1:raise RuntimeError('Fixed policy requires exactly one reply')
            return first[0]
        # RGK1 preserves complete reachable graphs and aliases. Fixed sampling
        # forgets root secrets; fresh value panels below remain independent.
        # Monotonic weak node tokens prohibit cross-node reuse without relying
        # on CPython address allocation. Retain bytes and statistics, no worlds.
        worldwire=worlds_bytes((world.state,world.weight) for world in worlds)
        actionwire=actions_bytes(first,encoder=self.action_encoder)
        key=(self.node_tokens.token(node),actor,worldwire,actionwire)
        entry=self.cache.get(key)
        if entry is not None:
            selector,size=entry;self.cache.move_to_end(key);self.stats['reply_cache_hits']+=1
        else:
            selector=ReplySelector(first,seed=self.stream.getrandbits(128))
            size=logical_charge(len(worldwire),len(actionwire),len(first))
            # Shared portable logical charge, explicitly separate from RSS.
            if size<=8*1024*1024:
                while self.cache and (len(self.cache)>=64 or self.cache_bytes+size>8*1024*1024):
                    _,old=self.cache.popitem(last=False);self.cache_bytes-=old[1]
                self.cache[key]=(selector,size);self.cache_bytes+=size
            self.stats['reply_cache_misses']+=1
        for _ in range(self.panels):
            action_key=selector.select()
            try:
                samples=sample_worlds(self.ai,state,actor,count=self.world_count,
                                     seed=self.stream.getrandbits(128))
                self.stats['belief_worlds_sampled']+=len(samples)
                values,weights=evaluate_panel(self.ai,samples,action_key,actor=actor,
                    root_side=self.side,own_net=self.net,seed=self.stream.getrandbits(128),
                    value_fn=lambda ai,s,root,net:self.value(s,actor),
                    menu_contract=(first,forced),allowed_keys=allowed_keys,
                    _owned_worlds=True)
                selector.complete(action_key,values,weights)
            except BaseException:
                selector.cancel();raise
            self.stats['reply_panels']+=1
        self.stats['reply_cache_estimated_bytes']=self.cache_bytes
        self.stats['reply_choices']+=1
        return selector.best()

    def child(self, node, overlay):
        if overlay is None:raise RuntimeError('Own-value replies require the outer hidden-hand sampler')
        overlay.patch(node.state)
        actions,forced=menu(self.ai,node.state,node.side)
        if not actions:
            if (self.ai._pending_decision_side(node.state) is not None
                    or getattr(node.state,'_ring_free_side',None) is not None
                    or node.state.pending_army_move_die is not None
                    or node.state.pending_muster_die is not None):
                raise RuntimeError('Empty sampled menu during required decision')
            other=self.ai.Side.SHADOW if node.side==self.ai.Side.FREE_PEOPLES else self.ai.Side.FREE_PEOPLES
            end=next((c for c in node.children if c.action is not None and c.action.action_type=='end_turn'),None)
            if end is not None and not self.ai.get_ai_actions(node.state,other):return end
            self.stats['empty_opponent_menus']+=1
            return None
        key=self.choose(node.state,node.side,node=node)
        if key not in actions:raise RuntimeError('Belief reply is absent from actual actor menu')
        # Retain/materialize only the chosen row. The existing catalog keeps
        # state/visits for that semantic edge; no full sampled menu is pinned.
        from search_edge_catalog import choose
        return choose(node,[actions[key]],(1.0,),(key,),True,self.ai.Action,self.ai.MCTSNode)

    def turn_action(self,state,actor,actions):
        from ai_policies import action_policy_key
        keys={action_policy_key(action):action for action in actions}
        key=self.choose(state,actor,allowed_keys=set(keys))
        self.stats['turn_reply_choices']+=1
        return keys[key] if key is not None else None


@contextmanager
def active(ai, side, own_net, *, profile, seed, reply_panels=2, belief_worlds=2):
    if not hasattr(ai,'_OWN_VALUE_CONTEXT'):
        raise ValueError('Use the prepared own-value reference engine')
    if ai._OWN_VALUE_CONTEXT is not None:raise RuntimeError('Nested own-value searches are unsupported')
    names=('_OWN_VALUE_CONTEXT','_OPPONENT_POLICY_MODE','_TREE_REUSE','_LEAF_BATCH',
           '_SEARCH_NETS','_SEARCH_ROOT_SIDE','_SIM_OVERLAY',
           '_LAST_LEAF_CONF','_LAST_LEAF_PHI','_CONF_ROOT_C','_CONF_MEAN')
    saved={name:getattr(ai,name) for name in names}
    tree=dict(ai._TREE_CACHE)
    with own_value_context(ai,side,own_net,profile=profile):
        context=SearchContext(ai,side,own_net,seed=seed,reply_panels=reply_panels,belief_worlds=belief_worlds)
        try:
            ai._CONF_MEAN=list(saved['_CONF_MEAN'])
            ai._OWN_VALUE_CONTEXT=context;ai._OPPONENT_POLICY_MODE=MODE
            ai._TREE_REUSE=False;ai._LEAF_BATCH=1;ai._TREE_CACHE['child']=None
            ai._SEARCH_NETS=(own_net,None) if side==ai.Side.FREE_PEOPLES else (None,own_net)
            ai._SEARCH_ROOT_SIDE=side
            yield context
        finally:
            for name,value in saved.items():setattr(ai,name,value)
            ai._TREE_CACHE.clear();ai._TREE_CACHE.update(tree)


def search(ai,state,side,own_net,*,profile,simulations,seed,reply_panels=2,belief_worlds=2):
    """Call the existing MCTS with only the searching side's model handle."""
    with active(ai,side,own_net,profile=profile,seed=seed,reply_panels=reply_panels,belief_worlds=belief_worlds) as ctx:
        with reference_rng(ai,seed):
            fp,sh=ai._SEARCH_NETS
            result=ai.mcts_search(state,side,fp,sh,num_simulations=simulations,
                                  temperature=0,add_noise=False,determinize=True)
        if result is not None and result[1] is not None:
            result[1]['own_value_search']={'contract':MODE,'reply_identity':'RGK1','reply_cache_charge':'wire_plus_512_per_row_plus_4096',
                'reply_panels_per_visit':reply_panels,
                'belief_worlds_per_panel':belief_worlds,'leaf_batch':1,'tree_reuse':False,'stats':dict(ctx.stats)}
        return result
