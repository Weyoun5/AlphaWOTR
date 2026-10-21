"""Real-engine reference reply evaluation using only the searching side's value.

Used by the isolated own-value MCTS reference, which runs sequentially:
the legacy Python engine uses module RNGs, temporarily restored around each
transition. A native port must instead use its owned RNG/state handles.
"""
from contextlib import contextmanager
from collections.abc import Mapping
from types import SimpleNamespace
import copy
import random

from reply_selection import ReplySelector


def root_tag(ai, root_side):
    if root_side == ai.Side.FREE_PEOPLES:return 'fp'
    if root_side == ai.Side.SHADOW:return 'sh'
    raise ValueError('Own-value search requires a valid root Side')


def validate_root_context(ai, root_side, own_net=None, *, require_model=True):
    from reward_contract import validate_profile, validate_net_profile
    tag=root_tag(ai,root_side)
    profile=ai.PLAYER_PROFILES.get(tag)
    validate_profile(tag,profile)
    if require_model:
        if own_net is None or getattr(own_net,'side',None)!=tag:
            raise ValueError('Own-value search requires a model bound to root side '+tag)
        validate_net_profile(own_net,tag,profile)
    return tag


@contextmanager
def own_value_context(ai, root_side, own_net, *, profile):
    """Explicit profile binding for a sequential reference worker.

    The handle must carry its loaded side; a remote key alone is not evidence
    of model identity. Existing reward-contract warm-start rules still apply.
    """
    tag=root_tag(ai,root_side);previous=ai.PLAYER_PROFILES
    try:
        ai.PLAYER_PROFILES={**previous,tag:profile}
        validate_root_context(ai,root_side,own_net)
        yield
    finally:
        ai.PLAYER_PROFILES=previous


def load_root_model(ai, path, root_side, *, profile):
    """Reuse the existing loader without overriding checkpoint side metadata."""
    from reward_contract import validate_profile
    validate_profile(root_tag(ai,root_side),profile)
    net=ai.load_value_net(str(path),side=None,profile=profile)
    with own_value_context(ai,root_side,net,profile=profile):
        return net


@contextmanager
def transition_rng(ai, seed):
    before_py=random.getstate();before_np=ai.np.random.get_state()
    try:
        random.setstate(random.Random(seed).getstate())
        ai.np.random.seed(seed % (1<<32))
        yield
    finally:
        random.setstate(before_py);ai.np.random.set_state(before_np)


class _IndexedActions(Mapping):
    """One menu's semantic keys; materialize immutable rows only on lookup.

    First-row order and duplicate handling match dict.setdefault. Dynamic
    Actions are still keyed afresh for every menu; nothing crosses calls.
    """
    def __init__(self, rows, key_function):
        self._rows=rows;self._positions={};self._materialized={}
        for index,key in enumerate(rows.action_keys(key_function)):
            self._positions.setdefault(key,index)

    def __iter__(self):return iter(self._positions)
    def __len__(self):return len(self._positions)
    def __contains__(self,key):return key in self._positions
    def __getitem__(self,key):
        if key not in self._materialized:
            self._materialized[key]=self._rows[self._positions[key]]
        return self._materialized[key]


def menu(ai, state, actor):
    from ai_policies import action_policy_key
    from search_lattice import Menu
    segmented=getattr(ai,'_segment_policy',None)
    policy=(segmented(SimpleNamespace(state=state,side=actor)) if segmented is not None
            else ai.get_ai_policy(state,actor))
    if isinstance(policy.actions,Menu):
        # Immutable blocks already own the same policy keys. Iterating them
        # recreates and recursively freezes every action before picking one.
        return _IndexedActions(policy.actions,action_policy_key),policy.forced_reason
    actions={}
    for action in policy.actions:
        # Existing engine semantic action keys identify equivalent duplicate rows.
        actions.setdefault(action_policy_key(action),action)
    return actions,policy.forced_reason


def root_value(ai, state, root_side, own_net):
    validate_root_context(ai,root_side,require_model=False)
    terminal=getattr(state,'_search_endturn_victory',None)
    if terminal is None:
        result=ai.game_logic.check_all_victory_conditions(state)
        if ai.game_logic.should_game_end(result,end_of_turn=False):terminal=result
    if terminal is not None:return ai._search_terminal(terminal,root_side,state)
    validate_root_context(ai,root_side,own_net)
    fp=own_net if root_side==ai.Side.FREE_PEOPLES else None
    sh=own_net if root_side==ai.Side.SHADOW else None
    return ai._mcts_eval_state(state,root_side,fp,sh)


def evaluate_panel(ai, worlds, key, *, actor, root_side, own_net, seed, value_fn=None,
                   menu_contract=None, allowed_keys=None, _owned_worlds=False):
    from belief_context import belief_confidence
    validate_root_context(ai,root_side,own_net)
    root_tag(ai,actor)
    if actor==root_side:
        raise ValueError('Reply actor must be the opponent of the searching side')
    stream=random.Random(seed);values=[];weights=[]
    evaluate=value_fn if value_fn is not None else root_value
    saved={name:getattr(ai,name) for name in ('_LAST_LEAF_CONF','_LAST_LEAF_PHI') if hasattr(ai,name)}
    try:
        for world in worlds:
            # The fresh-panel caller transfers its independently sampled worlds.
            # Other callers retain the historical no-mutation contract.
            state=world.state if _owned_worlds else copy.deepcopy(world.state)
            with transition_rng(ai,stream.getrandbits(128)):
                actions,reason=menu(ai,state,actor)
                if menu_contract is not None:
                    keys=tuple(k for k in actions if allowed_keys is None or k in allowed_keys)
                    if (keys,reason)!=menu_contract:
                        raise ValueError('Fresh belief sample changed actor-visible legal choices')
                if key not in actions:
                    raise ValueError('Reply is not legal throughout the actor information set')
                action=actions[key]
                with belief_confidence(ai,state,root_side):
                    successor=ai.apply_action(state,action)
                    ai._resume_preaction_for_search(state,successor,action)
                    values.append(float(evaluate(ai,successor,root_side,own_net)))
            weights.append(world.weight)
        return tuple(values),tuple(weights)
    finally:
        for name,value in saved.items():setattr(ai,name,value)


def plan_reply(ai, worlds, *, actor, root_side, own_net, budget, seed, value_fn=None):
    """Bounded information-set reply search; returns a semantic action key.

    No actual private root hand or partner model is accepted. Belief worlds must
    come from the actor-conditioned sampler. This version evaluates successors
    one decision ahead. The MCTS adapter also uses evaluate_panel at opponent
    nodes; batched inference is a separate optional component.
    """
    worlds=tuple(worlds)
    if not worlds or isinstance(budget,bool) or not isinstance(budget,int) or budget<1:
        raise ValueError('Nonempty belief worlds and positive integer budget required')
    if actor==root_side:
        raise ValueError('Reply actor must differ from root side')
    validate_root_context(ai,root_side,own_net)
    root_tag(ai,actor)
    first=None;forced=None
    for world in worlds:
        actions,reason=menu(ai,copy.deepcopy(world.state),actor)
        keys=tuple(actions)
        if first is None:first=keys;forced=reason
        elif set(keys)!=set(first) or reason!=forced:
            raise ValueError('Belief worlds disagree on actor-visible legal choices')
    if not first:return dict(action_key=None,reason='no_legal_action',panels=0,evaluations=0,statistics=[])
    if forced:
        if len(first)!=1:raise ValueError('Forced AI policy must contain exactly one action')
        return dict(action_key=first[0],reason=forced,panels=0,evaluations=0,statistics=[])
    selector=ReplySelector(first,seed=seed);stream=random.Random(seed)
    for _ in range(budget):
        key=selector.select()
        values,weights=evaluate_panel(ai,worlds,key,actor=actor,root_side=root_side,
            own_net=own_net,seed=stream.getrandbits(128),value_fn=value_fn)
        selector.complete(key,values,weights)
    return dict(action_key=selector.best(),reason='adversarial_root_value',panels=budget,
                evaluations=budget*len(worlds),statistics=selector.report())
