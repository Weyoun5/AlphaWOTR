"""Invocation-local encoding acceleration; never attach this to a GameState.

The caller creates one instance for one read-only _encode_actions_batch call.
Exact raw army keys take priority. Normalized aliases retain the first match
in insertion order, including a first match whose value is None.
"""

ENCODING_BATCH_KEY = object()
MISSING_BATCH = object()


def active_batch(comp_memo):
    return comp_memo.get(ENCODING_BATCH_KEY) if comp_memo is not None else None


class EncodingBatch:
    __slots__ = ('state', 'sanitize', 'aliases', 'compositions')

    def __init__(self, state, sanitize):
        self.state = state
        self.sanitize = sanitize
        self.aliases = None
        self.compositions = {}

    def armies_lookup(self, state, key):
        if state is not self.state:
            raise ValueError('EncodingBatch used with a different state')
        if key is None:
            return None
        army = state.armies.get(key)
        if army is not None:
            return army
        normalized = self.sanitize(str(key))
        aliases = self.aliases
        if aliases is None:
            aliases = {}
            for stored_key, value in state.armies.items():
                aliases.setdefault(self.sanitize(stored_key), value)
            self.aliases = aliases
        return aliases.get(normalized)
