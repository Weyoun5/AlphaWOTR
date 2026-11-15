"""Private army segments: no per-option Python objects before edge admission."""
from bisect import bisect_right
from collections import Counter, OrderedDict
from contextvars import ContextVar
import itertools
import pickle
import random
import numpy as np
STATS = Counter()
NATIVE = None

class Block:
    __slots__ = ('_keys', 'die')

    def __setattr__(self, *args):
        raise TypeError('Immutable companion segment')

class Menu:
    __slots__ = ('segments', 'ends', 'size', 'Action')

    def __init__(self, Action):
        self.segments = []
        self.ends = []
        self.size = 0
        self.Action = Action

    def extend(self, values):
        if isinstance(values, Menu):
            for segment in values.segments:
                self.extend(segment)
            return
        if isinstance(values, Block):
            if len(values):
                self.segments.append(values)
                self.size += len(values)
                self.ends.append(self.size)
            return
        values = list(values)
        if not values:
            return
        self.size += len(values)
        if self.segments and isinstance(self.segments[-1], list):
            self.segments[-1].extend(values)
            self.ends[-1] = self.size
        else:
            self.segments.append(values)
            self.ends.append(self.size)

    def append(self, action):
        if isinstance(action, Block):
            return self.extend(action)
        self.size += 1
        if self.segments and isinstance(self.segments[-1], list):
            self.segments[-1].append(action)
            self.ends[-1] = self.size
        else:
            self.segments.append([action])
            self.ends.append(self.size)

    def __len__(self):
        return self.size

    def __iter__(self):
        STATS['full_iteration'] += 1
        for segment in self.segments:
            if isinstance(segment, Block):
                for index in range(len(segment)):
                    yield segment.materialize(index, self.Action)
            else:
                yield from segment

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        which = bisect_right(self.ends, index)
        offset = index - (self.ends[which - 1] if which else 0)
        segment = self.segments[which]
        return segment.materialize(offset, self.Action) if isinstance(segment, Block) else segment[offset]

    def dynamic(self):
        for segment in self.segments:
            if not isinstance(segment, Block):
                yield from segment

    def action_keys(self, ordinary_key):
        keys = []
        identity = {}
        for segment in self.segments:
            if isinstance(segment, Block):
                keys.extend(segment.keys())
            else:
                for action in segment:
                    ident = id(action)
                    entry = identity.get(ident)
                    if entry is None:
                        entry = (action, ordinary_key(action))
                        identity[ident] = entry
                    keys.append(entry[1])
        return tuple(keys)

def allowed(ai, state):
    if state is None or ai.combat.FINAL_HITS_HOOK is not None:
        return False
    if any((v is not None and (not (k == 'pending_hunt_tiles' and v == [])) for k, v in vars(state).items() if k.startswith('pending_'))):
        return False
    mouth = getattr(state, 'mouth_of_sauron', None)
    return mouth is None or not mouth.in_play

def project(ai, state, side, menu):
    from ai_policies import project_ai_actions, PolicyProjection
    if not isinstance(menu, Menu):
        return project_ai_actions(state, side, menu)
    if not menu.segments:
        STATS['flat_projection_menus'] += 1
        return project_ai_actions(state, side, [])
    if len(menu.segments) == 1 and type(menu.segments[0]) is list:
        rows = menu.segments[0]
        STATS['flat_projection_menus'] += 1
        STATS['flat_projection_rows'] += len(rows)
        return project_ai_actions(state, side, rows)
    dynamic = list(menu.dynamic())
    p = project_ai_actions(state, side, dynamic)
    keep = Counter((id(a) for a in p.actions))
    result = Menu(ai.Action)
    for segment in menu.segments:
        if isinstance(segment, Block):
            result.extend(segment)
        else:
            chosen = []
            for a in segment:
                if keep[id(a)]:
                    chosen.append(a)
                    keep[id(a)] -= 1
            result.extend(chosen)
    return PolicyProjection(result, p.forced_reason if len(result) == 1 else None, p.reasons)

def encode(ai, menu, state, side, memo, ordinary):
    if not isinstance(menu, Menu):
        return ordinary(menu, state, side, memo)
    output = np.empty((len(menu), ai.ACTION_ENCODING_DIM), dtype=np.float32)
    regions = np.empty(len(menu), dtype=np.int64)
    dynamic = []
    positions = []
    offset = 0
    for segment in menu.segments:
        if isinstance(segment, LatticeBlock):
            a, r = segment.encoding(ai, state, side, memo, ordinary)
            output[offset:offset + len(segment)] = a
            regions[offset:offset + len(segment)] = r
            STATS['lattice_encoded_rows'] += len(segment)
        else:
            dynamic.extend(segment)
            positions.extend(range(offset, offset + len(segment)))
        offset += len(segment)
    if dynamic:
        a, r = ordinary(dynamic, state, side, memo)
        output[positions] = a
        regions[positions] = r
        STATS['dynamic_encoded_rows'] += len(dynamic)
    return (output, regions)

class Scope:

    def __init__(self):
        self.blocks = OrderedDict()
        self.rows = 0
        self.encodings = OrderedDict()
        self.bytes = 0
        self.catalogs = {}
SCOPE = ContextVar('private_lattice_scope', default=None)

class LatticeBlock(Block):
    __slots__ = ('raw', 'snapshots', 'destinations', 'block_key', '_encoding_actions')
    action_type = 'companion_separate'

    def __init__(self, raw, key, die, destinations, policy_key):
        object.__setattr__(self, 'raw', raw)
        from search_edge_catalog import snapshot_fields
        object.__setattr__(self, 'snapshots', tuple((pickle.dumps(snapshot_fields(a), protocol=5) for a in raw)))
        object.__setattr__(self, '_encoding_actions', None)
        object.__setattr__(self, '_keys', tuple((policy_key(a) for a in raw)))
        object.__setattr__(self, 'destinations', tuple(sorted(destinations)))
        object.__setattr__(self, 'block_key', key)
        object.__setattr__(self, 'die', die)
        STATS['block_created'] += 1
        STATS['block_snapshot_rows'] += len(raw)

    def __len__(self):
        return len(self.snapshots)

    def keys(self):
        return self._keys

    def materialize(self, index, Action):
        STATS['materialized'] += 1
        fields = pickle.loads(self.snapshots[index])
        action = Action(fields.pop('die_idx'), fields.pop('action_type'), fields.pop('params'))
        action.__dict__.update(fields)
        action._lattice = True
        action._block_key = self.block_key
        action._block_dests = self.destinations
        return action

    def _encoding_fallback(self, ai, state, side, memo, ordinary):
        scope = SCOPE.get()
        dice = state.dice.get(side, [])
        face = dice[self.die].current.name if 0 <= self.die < len(dice) and dice[self.die].current is not None else None
        hunt = getattr(state, 'pending_hunt', None)
        hplace = isinstance(hunt, dict) and hunt.get('phase') == 'hobbit_place'
        enemy = []
        for dest in self.destinations:
            mk = ('__enemy__', dest)
            value = memo.get(mk) if memo is not None else None
            if value is None:
                value = ai._enemy_comp_at(state, side, dest)
                if memo is not None:
                    memo[mk] = value
            enemy.append((dest, value))
        key = (self, side, face, hplace, tuple(enemy))
        result = scope.encodings.get(key) if scope is not None else None
        if result is not None:
            STATS['encoding_hit_rows'] += len(self)
            scope.encodings.move_to_end(key)
            return result
        if self._encoding_actions is None:
            object.__setattr__(self, '_encoding_actions', tuple((self.materialize(i, ai.Action) for i in range(len(self)))))
        result = ordinary(self._encoding_actions, state, side, memo)
        if scope is not None:
            result = tuple((np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape) for a in result))
            size = sum((a.nbytes for a in result))
            if size <= 64 * 1024 * 1024:
                while scope.encodings and scope.bytes + size > 64 * 1024 * 1024:
                    _, removed = scope.encodings.popitem(last=False)
                    scope.bytes -= sum((a.nbytes for a in removed))
                scope.encodings[key] = result
                scope.bytes += size
        STATS['encoding_miss_rows'] += len(self)
        return result

    def encoding(self, ai, state, side, memo, ordinary):
        scope = SCOPE.get()
        if scope is None or len(self) < 32 or side != ai.Side.FREE_PEOPLES:
            return self._encoding_fallback(ai, state, side, memo, ordinary)
        dice = state.dice.get(side, [])
        face = dice[self.die].current.name if 0 <= self.die < len(dice) and dice[self.die].current is not None else None
        hunt = getattr(state, 'pending_hunt', None)
        hplace = isinstance(hunt, dict) and hunt.get('phase') == 'hobbit_place'
        enemy = []
        for dest in self.destinations:
            mk = ('__enemy__', dest)
            value = memo.get(mk) if memo is not None else None
            if value is None:
                value = ai._enemy_comp_at(state, side, dest)
                if memo is not None:
                    memo[mk] = value
            enemy.append((dest, value))
        key = (((self.block_key, self.die, face),), len(self), hplace, tuple(enemy))
        matrix = ai._LATTICE_ENC_CACHE.get(key)
        if matrix is None or matrix.shape != (len(self), ai.ACTION_ENCODING_DIM):
            token = SCOPE.set(None)
            try:
                return self._encoding_fallback(ai, state, side, memo, ordinary)
            finally:
                SCOPE.reset(token)
        region_cache = getattr(scope, 'regions', None)
        if region_cache is None:
            scope.regions = region_cache = {}
        regions = region_cache.get(self)
        if regions is None:
            regions = np.frombuffer(np.asarray([ai._get_action_region_index(self.materialize(i, ai.Action)) for i in range(len(self))], dtype=np.int64).tobytes(), dtype=np.int64)
            region_cache[self] = regions
        STATS['original_matrix_direct_rows'] += len(self)
        return (matrix, regions)

def wrapper(policy_key):

    def wrap(raw, key, die, destinations):
        scope = SCOPE.get()
        ident = id(raw)
        if scope is not None:
            block = scope.blocks.get(ident)
            if block is not None and block.raw is raw:
                scope.blocks.move_to_end(ident)
                STATS['block_reuse_rows'] += len(block)
                return block
        block = LatticeBlock(raw, key, die, destinations, policy_key)
        if scope is not None and len(block) <= 100000:
            while scope.blocks and (len(scope.blocks) >= 256 or scope.rows + len(block) > 100000):
                _, removed = scope.blocks.popitem(last=False)
                scope.rows -= len(removed)
            scope.blocks[ident] = block
            scope.rows += len(block)
        return block
    return wrap

def category_rows(self):
    for segment in self.segments:
        yield from (itertools.repeat(segment, len(segment)) if isinstance(segment, Block) else segment)
Menu.category_rows = category_rows

def topk(values, indices, count):
    if count <= 0:
        return indices[:0]
    if len(indices) <= count:
        chosen = indices
    else:
        current = values[indices]
        threshold = np.partition(current, len(current) - count)[len(current) - count]
        high = indices[current > threshold]
        equal = indices[current == threshold]
        chosen = np.concatenate((high, equal[-(count - len(high)):]))
    return chosen[np.lexsort((chosen, values[chosen]))[::-1]]

def prune(ai, actions, priors, max_children=40, free_guarantee=None, ordinary=None):
    if type(actions) is list or type(actions) is tuple:
        view = Menu(ai.Action)
        # Borrow during this synchronous call; do not mutate or retain the source.
        view.segments = [actions]
        view.ends = [len(actions)]
        view.size = len(actions)
        STATS['borrowed_pruning_rows'] += len(actions)
        actions = view
    if isinstance(actions, Menu) and (type(max_children) is not int or type(ai._FREE_CANDS_N) is not int or ai._FREE_CANDS_N < 0):
        return ordinary(actions, priors, max_children, free_guarantee)
    if isinstance(actions, Menu) and type(priors) is list and (len(priors) == len(actions)) and all((type(value) is float for value in priors)):
        values = np.asarray(priors, dtype=np.float64)
        if np.isfinite(values).all():
            STATS['numeric_list_pruning_rows'] += len(priors)
            priors = values
    if not isinstance(actions, Menu):
        return ordinary(actions, priors, max_children, free_guarantee)
    if not (type(priors) is np.ndarray and priors.ndim == 1 and (len(priors) == len(actions)) and (priors.dtype.kind in 'fiu') and np.isfinite(priors).all()):
        return ordinary(actions, priors, max_children, free_guarantee)
    free_guarantee = ai._FREE_CANDS if free_guarantee is None else free_guarantee
    categories = np.empty(len(actions), dtype=np.int64)
    free = []
    offset = 0
    other = ai._ACTION_TYPE_TO_IDX['other']
    for segment in actions.segments:
        if isinstance(segment, Block):
            cat = ai._ACTION_TYPE_TO_IDX.get(ai._ATYPE_TO_CAT.get(segment.action_type, 'other'), other)
            categories[offset:offset + len(segment)] = cat
            if free_guarantee and segment.action_type in ai._FREE_ACTION_TYPES:
                free.extend(range(offset, offset + len(segment)))
        else:
            for i, action in enumerate(segment):
                categories[offset + i] = ai._ACTION_TYPE_TO_IDX.get(ai._ATYPE_TO_CAT.get(action.action_type, 'other'), other)
                if free_guarantee and action.action_type in ai._FREE_ACTION_TYPES:
                    free.append(offset + i)
        offset += len(segment)
    selected = set()
    for cat in np.unique(categories):
        selected.update(map(int, topk(priors, np.flatnonzero(categories == cat), 3)))
    if free_guarantee and free:
        selected.update(map(int, topk(priors, np.asarray(free, dtype=np.int64), ai._FREE_CANDS_N)))
    if len(selected) < max_children:
        keep = np.ones(len(actions), dtype=bool)
        if selected:
            keep[list(selected)] = False
        selected.update(map(int, topk(priors, np.flatnonzero(keep), max_children - len(selected))))
    return sorted(selected)

def search_scope(search):
    from functools import wraps

    @wraps(search)
    def scoped(*args, **kwargs):
        token = SCOPE.set(Scope())
        try:
            return search(*args, **kwargs)
        finally:
            SCOPE.reset(token)
    return scoped
