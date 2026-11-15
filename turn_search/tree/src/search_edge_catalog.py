"""Immutable search-owned opponent edge snapshots; public Actions stay mutable."""
import pickle
from action_encoding_backend import prepare_snapshot as _native_fields
import random
from collections import Counter
from enum import Enum
STATS = Counter()
DERIVED_FIELDS = frozenset(('_lattice', '_encs', '_block_key', '_block_dests', '_ri'))

def snapshot_fields(action):
    return {k: v for k, v in vars(action).items() if k not in DERIVED_FIELDS}

def value_payload(value):
    if type(value) in (str, int, float, bool, bytes, type(None)):
        return True
    if isinstance(value, Enum):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all((value_payload(v) for v in value))
    if type(value) is dict:
        return all((value_payload(k) and value_payload(v) for k, v in value.items()))
    return False

_GENERIC_FIELDS = object()

def _prepare_payload(action, Action):
    fields = _native_fields(action) if type(action) is Action else None
    if fields is not None:
        return fields
    # Keep side-effect-capable/custom values on the original validation and
    # second snapshot-read sequence. Only pure built-in payloads are fused.
    return _GENERIC_FIELDS if value_payload(snapshot_fields(action)) else None

class Edge:
    __slots__ = ('payload', 'key', 'prior', 'child')

    def __init__(self, action, key, prior, child=None, payload=None):
        self.payload = pickle.dumps(snapshot_fields(action) if payload is None or payload is _GENERIC_FIELDS else payload, protocol=5)
        self.key = key
        self.prior = prior
        self.child = child

class Catalog:
    __slots__ = ('records', 'physical', 'index', 'disabled')

    def __init__(self):
        self.records = []
        self.physical = []
        self.index = {}
        self.disabled = False

def _ordinary_materialize(edge, node, Action, Node):
    fields = pickle.loads(edge.payload)
    action = Action(fields.pop('die_idx'), fields.pop('action_type'), fields.pop('params'))
    action.__dict__.update(fields)
    if edge.child is None:
        edge.child = Node(None, node.side, parent=node, action=action, prior=edge.prior)
        STATS['materialized'] += 1
    else:
        edge.child.action = action
    return edge.child

def flush(node, Action, Node):
    catalog = getattr(node, '_opponent_edge_catalog', None)
    if catalog is not None and (not catalog.disabled):
        node.children = [materialize(edge, node, Action, Node) for edge in catalog.records]
        catalog.physical = list(node.children)
        catalog.disabled = True
        STATS['catalog_fallbacks'] += 1

def _ordinary_choose(node, actions, probabilities, action_keys, forced, Action, Node):
    catalog = node._opponent_edge_catalog
    if catalog is None:
        catalog = node._opponent_edge_catalog = Catalog()
    if catalog.disabled:
        return full_choice(node, actions, probabilities, action_keys, forced, Action, Node)
    if len(node.children) < len(catalog.physical) or any((a is not b for a, b in zip(node.children, catalog.physical))):
        raise RuntimeError('Snapshot edge catalog requires append-only child list')
    new_children = node.children[len(catalog.physical):]
    if any((c.action is None or not value_payload(snapshot_fields(c.action)) for c in new_children)):
        pending = list(new_children)
        flush(node, Action, Node)
        node.children.extend(pending)
        return full_choice(node, actions, probabilities, action_keys, forced, Action, Node)
    from ai_policies import action_policy_key
    for child in new_children:
        if child.action is not None:
            edge = Edge(child.action, action_policy_key(child.action), child.prior, child)
            catalog.records.append(edge)
            catalog.index[edge.key] = edge
    catalog.physical = list(node.children)
    for action, prior, key in zip(actions, probabilities, action_keys):
        if key not in catalog.index:
            payload = _prepare_payload(action, Action)
            if payload is None:
                flush(node, Action, Node)
                return full_choice(node, actions, probabilities, action_keys, forced, Action, Node)
            edge = Edge(action, key, prior, payload=payload)
            catalog.records.append(edge)
            catalog.index[key] = edge
            STATS['snapshots_created'] += 1
    index = 0 if forced else random.choices(range(len(actions)), weights=probabilities, k=1)[0]
    edge = catalog.index[action_keys[index]]
    fresh = edge.child is None
    child = materialize(edge, node, Action, Node)
    if fresh:
        node.children.append(child)
        catalog.physical.append(child)
    STATS['selections'] += 1
    return child

def full_choice(node, actions, probabilities, action_keys, forced, Action, Node):
    from ai_policies import action_policy_key
    existing = {action_policy_key(c.action): c for c in node.children if c.action is not None}
    eligible = []
    for action, prior, key in zip(actions, probabilities, action_keys):
        child = existing.get(key)
        if child is None:
            child = Node(None, node.side, parent=node, action=action, prior=prior)
            node.children.append(child)
            existing[key] = child
        eligible.append(child)
    return eligible[0] if forced else random.choices(eligible, weights=probabilities, k=1)[0]

from search_lattice import Block,Menu,SCOPE,STATS as _SEGMENT_STATS

def materialize(edge, node, Action, Node):
    payload = edge.payload
    if not (type(payload) is tuple and len(payload) == 2 and isinstance(payload[0], Block)):
        return _ordinary_materialize(edge, node, Action, Node)
    action = payload[0].materialize(payload[1], Action)
    if edge.child is None:
        edge.child = Node(None, node.side, parent=node, action=action, prior=edge.prior)
        STATS['materialized'] += 1
    else:
        edge.child.action = action
    return edge.child

def choose(node, actions, probabilities, action_keys, forced, Action, Node):
    if not isinstance(actions, Menu):
        return _ordinary_choose(node, actions, probabilities, action_keys, forced, Action, Node)
    cat = node._opponent_edge_catalog
    if cat is None:
        cat = node._opponent_edge_catalog = Catalog()
    if cat.disabled:
        return _ordinary_choose(node, list(actions), probabilities, action_keys, forced, Action, Node)
    if len(node.children) < len(cat.physical) or any((a is not b for a, b in zip(node.children, cat.physical))):
        raise RuntimeError('Snapshot edge catalog requires append-only child list')
    new_children = node.children[len(cat.physical):]
    if any((c.action is None or not value_payload(snapshot_fields(c.action)) for c in new_children)):
        pending = list(new_children)
        flush(node, Action, Node)
        node.children.extend(pending)
        return full_choice(node, list(actions), probabilities, action_keys, forced, Action, Node)
    from ai_policies import action_policy_key
    for child in new_children:
        if child.action is not None:
            edge = Edge(child.action, action_policy_key(child.action), child.prior, child)
            cat.records.append(edge)
            cat.index[edge.key] = edge
    cat.physical = list(node.children)
    scope = SCOPE.get()
    admitted = scope.catalogs.setdefault(cat, set()) if scope is not None else set()
    offset = 0
    for segment in actions.segments:
        if isinstance(segment, Block):
            if segment in admitted:
                _SEGMENT_STATS['catalog_block_skipped_rows'] += len(segment)
                offset += len(segment)
                continue
            for i in range(len(segment)):
                key = action_keys[offset + i]
                if key not in cat.index:
                    edge = object.__new__(Edge)
                    edge.payload = (segment, i)
                    edge.key = key
                    edge.prior = probabilities[offset + i]
                    edge.child = None
                    cat.records.append(edge)
                    cat.index[key] = edge
                    STATS['snapshots_created'] += 1
            admitted.add(segment)
        else:
            for i, action in enumerate(segment):
                key = action_keys[offset + i]
                if key not in cat.index:
                    payload = _prepare_payload(action, Action)
                    if payload is None:
                        flush(node, Action, Node)
                        return full_choice(node, list(actions), probabilities, action_keys, forced, Action, Node)
                    edge = Edge(action, key, probabilities[offset + i], payload=payload)
                    cat.records.append(edge)
                    cat.index[key] = edge
                    STATS['snapshots_created'] += 1
        offset += len(segment)
    index = 0 if forced else random.choices(range(len(actions)), weights=probabilities, k=1)[0]
    edge = cat.index[action_keys[index]]
    fresh = edge.child is None
    child = materialize(edge, node, Action, Node)
    if fresh:
        node.children.append(child)
        cat.physical.append(child)
    STATS['selections'] += 1
    return child
