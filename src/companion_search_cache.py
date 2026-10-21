"""Invocation-scoped reuse of canonical companion geometry and answers.

Board topology/metadata are immutable under Board.adjacent's existing contract.
Mutable siege/control sets, Fellowship origin/progress and all separable names
and levels are keyed explicitly. Neither Actions nor tree statistics are shared.
"""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps

_CURRENT = ContextVar('companion_search_cache', default=None)
MAX_REACH_ENTRIES = 1024
MAX_ANSWER_CONTEXTS = 256


class _Scope:
    __slots__ = ('reach', 'answers')

    def __init__(self):
        self.reach = OrderedDict()
        self.answers = OrderedDict()


def search_scope(search):
    """Dispose of both tables after every search, including failed searches."""
    @wraps(search)
    def scoped(*args, **kwargs):
        token = _CURRENT.set(_Scope())
        try:
            return search(*args, **kwargs)
        finally:
            _CURRENT.reset(token)
    return scoped


def _geometry_key(ai, state, start, max_steps):
    return (state.board, start, max_steps,
            frozenset(getattr(state, 'besieged_strongholds', set())),
            frozenset(state.enemy_controlled.get(ai.Side.FREE_PEOPLES, set())),
            frozenset(state.enemy_controlled.get(ai.Side.SHADOW, set())))


def cache_reachable(ai):
    """Keep the original BFS as the only implementation of reachability."""
    def decorate(original):
        @wraps(original)
        def reachable(state, start, max_steps):
            scope = _CURRENT.get()
            if (scope is None or type(state.board) is not ai.board.Board
                    or type(start) is not str or type(max_steps) is not int):
                return original(state, start, max_steps)
            try:
                key = _geometry_key(ai, state, start, max_steps)
            except TypeError:
                return original(state, start, max_steps)
            cache = scope.reach
            value = cache.get(key)
            if value is not None:
                cache.move_to_end(key)
                return value.copy()
            value = original(state, start, max_steps)
            if len(cache) >= MAX_REACH_ENTRIES:
                cache.popitem(last=False)
            cache[key] = value.copy()
            return value
        return reachable
    return decorate


def answer_table(ai, state, origin, progress, separable):
    """Reuse only the original generator's canonical answer table.

The generator keeps its existing distinct/present-name and literal-type guards,
and bypasses this for identity-sensitive hunt placement. Its Mordor early exit
is unchanged. All unknown contexts receive a fresh invocation-local dictionary.
"""
    scope = _CURRENT.get()
    if (scope is None or type(state.board) is not ai.board.Board
            or type(origin) is not str or type(progress) is not int):
        return {}
    try:
        key = (_geometry_key(ai, state, origin, 0), progress,
               tuple((companion.name, companion.level) for companion in separable))
    except TypeError:
        return {}
    cache = scope.answers
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
        return value
    if len(cache) >= MAX_ANSWER_CONTEXTS:
        cache.popitem(last=False)
    value = {}
    cache[key] = value
    return value
