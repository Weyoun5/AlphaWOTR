"""Lossless migration storage for whitelisted engine objects, owned by C++.

This is a typed value graph, not pickle storage and not yet a native rules
implementation. Ordered containers and object/dictionary aliases are retained.
Python construction occurs only at import/export boundaries.
"""
from dataclasses import dataclass
from enum import Enum
import ctypes as C
import importlib
import threading
from functools import lru_cache

import numpy as np

import bridge
import snapshot_schema

NONE, BOOL, INT, FLOAT, STRING, BYTES, LIST, TUPLE, DICT, SET, FROZENSET, OBJECT, ENUM, BIGINT = range(14)


class Value(C.Structure):
    _fields_ = [('tag', C.c_uint32), ('flags', C.c_uint32),
                ('integer', C.c_int64), ('real', C.c_double),
                ('offset', C.c_uint64), ('count', C.c_uint64)]


@dataclass
class Graph:
    values: object
    edges: object
    data: bytes
    root: int


def _identity(cls):
    return cls.__module__ + '.' + cls.__qualname__


def _classes():
    result = {}
    for name in (*snapshot_schema.CLASS_FIELDS, *snapshot_schema.ENUM_TYPES):
        module, attribute = name.rsplit('.', 1)
        cls = getattr(importlib.import_module(module), attribute)
        if _identity(cls) != name:
            raise ValueError('Schema class identity mismatch: ' + name)
        result[name] = cls
    return result


def _check_attributes(name, attributes):
    if type(attributes) is not dict:
        raise ValueError('Object attributes must be a plain dictionary: ' + name)
    unknown = attributes.keys() - set(snapshot_schema.CLASS_FIELDS[name])
    if unknown:
        raise ValueError('Unregistered fields for ' + name + ': ' + repr(unknown))


_GLOBAL_LISTS = None
_GLOBAL_LIST_LOCK = threading.RLock()


def _global_list_bindings(classes):
    """Validate and pin the two external lists borrowed by original dice code.

    Native storage still owns all six values. Only reconstruction reuses these
    registered globals, matching clone_state and later bonus-die creation.
    """
    global _GLOBAL_LISTS
    with _GLOBAL_LIST_LOCK:
        current = {}
        enum = classes['game_state.ActionDieFace']
        for name, members in snapshot_schema.GLOBAL_LIST_BINDINGS.items():
            module, attribute = name.rsplit('.', 1)
            value = getattr(importlib.import_module(module), attribute)
            expected = tuple(enum[member] for member in members)
            if type(value) is not list or len(value) != len(expected) or any(
                    actual is not wanted for actual, wanted in zip(value, expected)):
                raise ValueError('Registered global list contents changed: ' + name)
            if _GLOBAL_LISTS is not None and value is not _GLOBAL_LISTS[name]:
                raise ValueError('Registered global list identity changed: ' + name)
            current[name] = value
        if _GLOBAL_LISTS is None:
            _GLOBAL_LISTS = current
        return _GLOBAL_LISTS


def encode_graph(ai, state):
    """Encode all registered state; reject unsupported objects rather than drop them."""
    classes = _classes()
    global_lists = _global_list_bindings(classes)
    binding_by_id = {id(value): name for name, value in global_lists.items()}
    values, edges, data = [], [], bytearray()
    memo, strings = {}, {}

    def visit(obj):
        typ = type(obj)
        if typ is str and obj in strings:
            return strings[obj]
        compound = typ in (list, tuple, dict, set, frozenset) or isinstance(obj, Enum) or _identity(typ) in snapshot_schema.CLASS_FIELDS
        if compound and id(obj) in memo:
            return memo[id(obj)]
        index = len(values)
        node = Value()
        values.append(node)
        if compound:
            memo[id(obj)] = index
        if obj is None:
            node.tag = NONE
        elif typ is bool:
            node.tag, node.integer = BOOL, int(obj)
        elif typ is int and -(1 << 63) <= obj < (1 << 63):
            node.tag, node.integer = INT, obj
        elif typ is float:
            node.tag, node.real = FLOAT, obj
        elif typ in (str, bytes, int):
            node.tag = {str: STRING, bytes: BYTES, int: BIGINT}[typ]
            raw = obj.encode('utf-8', 'surrogatepass') if typ is str else str(obj).encode('ascii') if typ is int else obj
            node.offset, node.count = len(data), len(raw)
            data.extend(raw)
            if typ is str:
                strings[obj] = index
        else:
            if isinstance(obj, Enum):
                name = _identity(typ)
                if name not in snapshot_schema.ENUM_TYPES or classes[name] is not typ:
                    raise ValueError('Unregistered enum: ' + name)
                node.tag = ENUM
                children = [visit(name), visit(obj.name)]
            elif typ in (list, tuple, set, frozenset):
                node.tag = {list: LIST, tuple: TUPLE, set: SET, frozenset: FROZENSET}[typ]
                if typ is list and id(obj) in binding_by_id:
                    node.flags = 1
                    node.integer = visit(binding_by_id[id(obj)])
                children = [visit(item) for item in obj]
            elif typ is dict:
                node.tag = DICT
                children = []
                for key, value in obj.items():
                    children.extend((visit(key), visit(value)))
            else:
                name = _identity(typ)
                if name not in snapshot_schema.CLASS_FIELDS or classes[name] is not typ:
                    raise ValueError('Unregistered snapshot object: ' + name)
                node.tag = OBJECT
                children = [visit(name)]
                if hasattr(obj, '__dict__'):
                    attributes = vars(obj)
                    _check_attributes(name, attributes)
                    # Preserve aliases to the real __dict__, not just values.
                    children.extend((visit('__dict__'), visit(attributes)))
                for slot in snapshot_schema.CLASS_SLOTS.get(name, ()):
                    if hasattr(obj, slot):
                        children.extend((visit(slot), visit(getattr(obj, slot))))
            node.offset, node.count = len(edges), len(children)
            edges.extend(children)
        return index

    root = visit(state)
    return Graph((Value * len(values))(*values), (C.c_uint64 * len(edges))(*edges), bytes(data), root)


def decode_graph(ai, graph, *, metadata=False):
    """Reconstruct whitelisted classes without invoking their constructors."""
    # Trace rows usually contain only scalars and containers. They need all
    # graph validation below, but no engine class imports or external-list
    # reconstruction. Object-bearing metadata retains the ordinary path.
    plain = metadata and all(not node.flags and node.tag not in (OBJECT, ENUM) for node in graph.values)
    classes = {} if plain else _classes()
    global_lists = {} if plain else _global_list_bindings(classes)
    memo = {}
    bindings_seen = set()
    for node in graph.values:
        if not node.flags:
            continue
        if node.tag != LIST or node.flags != 1:
            raise ValueError('Unsupported graph value flags')
        index = int(node.integer)
        if not 0 <= index < len(graph.values) or graph.values[index].tag != STRING:
            raise ValueError('Global list binding requires a STRING node')
        name_node = graph.values[index]
        offset, count = int(name_node.offset), int(name_node.count)
        if offset > len(graph.data) or count > len(graph.data) - offset:
            raise ValueError('Global list binding name outside graph storage')
        name = graph.data[offset:offset + count].decode('utf-8', 'surrogatepass')
        if name not in global_lists:
            raise ValueError('Unregistered global list binding: ' + name)
        if name in bindings_seen:
            raise ValueError('Duplicate global list binding: ' + name)
        bindings_seen.add(name)

    def build(index, depth=0):
        if index in memo:
            return memo[index]
        if depth > 400:
            raise ValueError('Malformed immutable cycle or unsupported graph nesting')
        if not 0 <= index < len(graph.values):
            raise ValueError('Graph reference outside node array')
        node = graph.values[index]
        tag = node.tag
        if node.flags and not (tag == LIST and node.flags == 1):
            raise ValueError('Unsupported graph value flags')
        if tag not in (INT, BOOL) and not node.flags and node.integer != 0:
            raise ValueError('Unused graph integer field must be zero')
        offset, count = int(node.offset), int(node.count)
        if tag in (STRING, BYTES, BIGINT):
            if offset > len(graph.data) or count > len(graph.data) - offset:
                raise ValueError('Graph byte range outside storage')
            raw = graph.data[offset:offset + count]
            result = raw.decode('utf-8', 'surrogatepass') if tag == STRING else int(raw.decode('ascii')) if tag == BIGINT else raw
        elif tag in (NONE, BOOL, INT, FLOAT):
            result = None if tag == NONE else bool(node.integer) if tag == BOOL else int(node.integer) if tag == INT else float(node.real)
        else:
            if offset > len(graph.edges) or count > len(graph.edges) - offset:
                raise ValueError('Graph edge range outside storage')
            refs = graph.edges[offset:offset + count]
            if tag == ENUM:
                if count != 2:
                    raise ValueError('Malformed enum')
                name, member = (build(ref, depth + 1) for ref in refs)
                if name not in snapshot_schema.ENUM_TYPES:
                    raise ValueError('Unregistered enum: ' + str(name))
                result = classes[name][member]
            elif tag == OBJECT:
                if count < 1 or count % 2 != 1:
                    raise ValueError('Unsupported object representation')
                name = build(refs[0], depth + 1)
                if name not in snapshot_schema.CLASS_FIELDS:
                    raise ValueError('Unregistered object: ' + str(name))
                result = object.__new__(classes[name])
                memo[index] = result
                seen = set()
                for k in range(1, count, 2):
                    field = build(refs[k], depth + 1)
                    if field in seen:
                        raise ValueError('Duplicate object field: ' + str(field))
                    seen.add(field)
                    value = build(refs[k + 1], depth + 1)
                    if field == '__dict__':
                        _check_attributes(name, value)
                    elif field not in snapshot_schema.CLASS_SLOTS.get(name, ()):
                        raise ValueError('Unregistered object slot: ' + str(field))
                    object.__setattr__(result, field, value)
            elif tag == DICT:
                if count % 2:
                    raise ValueError('Malformed dictionary')
                result = {}
                memo[index] = result
                for k in range(0, count, 2):
                    result[build(refs[k], depth + 1)] = build(refs[k + 1], depth + 1)
            elif tag == LIST and node.flags == 1:
                binding_index = int(node.integer)
                if not 0 <= binding_index < len(graph.values) or graph.values[binding_index].tag != STRING:
                    raise ValueError('Global list binding requires a STRING node')
                name = build(binding_index, depth + 1)
                if name not in global_lists:
                    raise ValueError('Unregistered global list binding: ' + str(name))
                result = global_lists[name]
                if count != len(result):
                    raise ValueError('Global list binding has incorrect length: ' + name)
                # Do not memoize or return the external list until all owned
                # members prove this exact pinned source representation.
                items = [build(ref, depth + 1) for ref in refs]
                if any(actual is not wanted for actual, wanted in zip(items, result)):
                    raise ValueError('Global list binding contents differ: ' + name)
            elif tag in (LIST, SET):
                result = [] if tag == LIST else set()
                memo[index] = result
                for ref in refs:
                    item = build(ref, depth + 1)
                    result.append(item) if tag == LIST else result.add(item)
            elif tag in (TUPLE, FROZENSET):
                items = [build(ref, depth + 1) for ref in refs]
                # Like deepcopy: a mutable child may have completed this tuple
                # recursively while closing a tuple/list cycle.
                result = memo[index] if index in memo else tuple(items) if tag == TUPLE else frozenset(items)
            else:
                raise ValueError('Unsupported graph value tag: ' + str(tag))
        memo[index] = result
        return result

    return build(graph.root)


_LIB = None
_LIB_LOCK = threading.Lock()


def library():
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        lib = bridge.library()
        lib.wotr_snapshot_abi_version.argtypes = []
        lib.wotr_snapshot_abi_version.restype = C.c_uint32
        if lib.wotr_snapshot_abi_version() != 1:
            raise RuntimeError('Snapshot ABI mismatch')
        lib.wotr_snapshot_last_error.argtypes = []
        lib.wotr_snapshot_last_error.restype = C.c_char_p
        arrays = [C.POINTER(Value), C.c_size_t, C.POINTER(C.c_uint64), C.c_size_t, C.POINTER(C.c_uint8), C.c_size_t]
        lib.wotr_snapshot_create.argtypes = arrays + [C.c_uint64]
        lib.wotr_snapshot_create.restype = C.c_void_p
        lib.wotr_snapshot_clone.argtypes = [C.c_void_p]
        lib.wotr_snapshot_clone.restype = C.c_void_p
        lib.wotr_snapshot_action_clone.argtypes = [C.c_void_p]
        lib.wotr_snapshot_action_clone.restype = C.c_void_p
        lib.wotr_snapshot_action_clone_cache_info.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        lib.wotr_snapshot_action_clone_cache_info.restype = C.c_int
        lib.wotr_snapshot_victory.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32, C.POINTER(C.c_int64), C.c_size_t]
        lib.wotr_snapshot_victory.restype = C.c_int
        lib.wotr_snapshot_political_ring_actions.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
        lib.wotr_snapshot_political_ring_actions.restype = C.c_void_p
        lib.wotr_snapshot_muster_actions.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
        lib.wotr_snapshot_muster_actions.restype = C.c_void_p
        for name in ('army_move_actions', 'saruman_muster_actions', 'companion_political_actions', 'discard_card_actions', 'fellowship_declare_actions', 'full_actions', 'draw_card_actions', 'mouth_move_actions', 'minion_muster_actions'):
            function = getattr(lib, 'wotr_snapshot_' + name)
            function.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
            function.restype = C.c_void_p
        lib.wotr_snapshot_apply_saruman_muster_action.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_apply_saruman_muster_action.restype = C.c_void_p
        lib.wotr_snapshot_apply_companion_political_action.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.c_char_p]
        lib.wotr_snapshot_apply_companion_political_action.restype = C.c_void_p
        lib.wotr_snapshot_apply_discard_card.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_char_p]
        lib.wotr_snapshot_apply_discard_card.restype = C.c_void_p
        lib.wotr_snapshot_apply_descriptor.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_int64]
        lib.wotr_snapshot_apply_descriptor.restype = C.c_void_p
        lib.wotr_snapshot_project_policy.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_project_policy.restype = C.c_void_p
        lib.wotr_snapshot_next_actor.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_next_actor.restype = C.c_int
        lib.wotr_snapshot_pending_actor.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_pending_actor.restype = C.c_void_p
        lib.wotr_snapshot_combat_post_stage.argtypes = [C.c_void_p, C.c_int64]
        lib.wotr_snapshot_combat_post_stage.restype = C.c_void_p
        lib.wotr_snapshot_fellowship_tail.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_void_p), C.POINTER(C.c_int32)]
        lib.wotr_snapshot_fellowship_tail.restype = C.c_void_p
        lib.wotr_snapshot_apply_fellowship_declare.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int32]
        lib.wotr_snapshot_apply_fellowship_declare.restype = C.c_void_p
        lib.wotr_snapshot_guide_actions.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_guide_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_guide_choice.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p]
        lib.wotr_snapshot_apply_guide_choice.restype = C.c_void_p
        lib.wotr_snapshot_hunt_params.argtypes = [C.c_void_p, C.c_void_p]
        lib.wotr_snapshot_hunt_params.restype = C.c_void_p
        lib.wotr_snapshot_hunt_roll.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64, C.POINTER(C.c_void_p), C.POINTER(C.c_int64), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_hunt_roll.restype = C.c_void_p
        lib.wotr_snapshot_combat_pre_stage.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_pre_stage.restype = C.c_void_p
        lib.wotr_snapshot_combat_round_stage.argtypes = lib.wotr_snapshot_combat_pre_stage.argtypes
        lib.wotr_snapshot_combat_round_stage.restype = C.c_void_p
        lib.wotr_snapshot_apply_draw_card.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_apply_draw_card.restype = C.c_void_p
        lib.wotr_snapshot_hunt_tile.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_int32, C.c_int32, C.c_int32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_hunt_tile.restype = C.c_void_p
        lib.wotr_snapshot_combat_publish_paused.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_publish_paused.restype = C.c_void_p
        lib.wotr_snapshot_apply_fellowship_move.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_fellowship_move.restype = C.c_void_p
        lib.wotr_snapshot_eye_allocation_actions.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_eye_allocation_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_eye_allocation.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        lib.wotr_snapshot_apply_eye_allocation.restype = C.c_void_p
        lib.wotr_snapshot_roll_action_dice.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_roll_action_dice.restype = C.c_void_p
        lib.wotr_snapshot_combat_reroll_stage.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_reroll_stage.restype = C.c_void_p
        lib.wotr_snapshot_combat_primary_stage.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_primary_stage.restype = C.c_void_p
        lib.wotr_snapshot_draw_event_card.argtypes = [C.c_void_p, C.c_int32, C.c_char_p, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_draw_event_card.restype = C.c_void_p
        lib.wotr_snapshot_open_hand_limit_discard.argtypes = [C.c_void_p, C.POINTER(C.c_int32)]
        lib.wotr_snapshot_open_hand_limit_discard.restype = C.c_void_p
        lib.wotr_snapshot_begin_turn_reset.argtypes = [C.c_void_p, C.c_int64, C.c_int32, C.c_int32, C.POINTER(C.c_int32), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_begin_turn_reset.restype = C.c_void_p
        if hasattr(lib, 'wotr_snapshot_begin_turn_reset_settled'):
            lib.wotr_snapshot_begin_turn_reset_settled.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32, C.c_int32, C.POINTER(C.c_int32), C.POINTER(C.c_void_p)]
            lib.wotr_snapshot_begin_turn_reset_settled.restype = C.c_void_p
        lib.wotr_snapshot_next_turn.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_int32), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_next_turn.restype = C.c_void_p
        lib.wotr_snapshot_enforce_control.argtypes = [C.c_void_p, C.c_void_p]
        lib.wotr_snapshot_enforce_control.restype = C.c_void_p
        lib.wotr_snapshot_apply_mouth_move.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.c_char_p]
        lib.wotr_snapshot_apply_mouth_move.restype = C.c_void_p
        lib.wotr_snapshot_combat_casualty_menu.argtypes = [C.c_void_p, C.c_int32, C.c_char_p, C.c_int64]
        lib.wotr_snapshot_combat_casualty_menu.restype = C.c_void_p
        lib.wotr_snapshot_combat_casualty_operation.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_char_p, C.c_int64, C.c_char_p, C.c_char_p]
        lib.wotr_snapshot_combat_casualty_operation.restype = C.c_void_p
        lib.wotr_snapshot_hunt_menu.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
        lib.wotr_snapshot_hunt_menu.restype = C.c_void_p
        lib.wotr_snapshot_hunt_decision.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_hunt_decision.restype = C.c_void_p
        lib.wotr_snapshot_fellowship_actions.argtypes = [C.c_void_p, C.c_int64]
        lib.wotr_snapshot_fellowship_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_fellowship_hide.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32]
        lib.wotr_snapshot_apply_fellowship_hide.restype = C.c_void_p
        lib.wotr_snapshot_apply_minion_muster.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.c_char_p]
        lib.wotr_snapshot_apply_minion_muster.restype = C.c_void_p
        lib.wotr_snapshot_companion_actions.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32]
        lib.wotr_snapshot_companion_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_companion.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int32, C.c_char_p, C.c_char_p, C.POINTER(C.c_char_p), C.c_size_t]
        lib.wotr_snapshot_apply_companion.restype = C.c_void_p
        lib.wotr_snapshot_combat_deferred_post.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_deferred_post.restype = C.c_void_p
        lib.wotr_snapshot_cleanup_sieges.argtypes = lib.wotr_snapshot_enforce_control.argtypes
        lib.wotr_snapshot_cleanup_sieges.restype = C.c_void_p
        lib.wotr_snapshot_safe_next_turn.argtypes = lib.wotr_snapshot_next_turn.argtypes
        lib.wotr_snapshot_safe_next_turn.restype = C.c_void_p
        lib.wotr_snapshot_combat_allocation_settle.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_allocation_settle.restype = C.c_void_p
        lib.wotr_snapshot_apply_descriptor_random.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_descriptor_random.restype = C.c_void_p
        lib.wotr_snapshot_combat_resume_settle.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_combat_resume_settle.restype = C.c_void_p
        lib.wotr_snapshot_resume_preaction.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_resume_preaction.restype = C.c_void_p
        lib.wotr_snapshot_resume_preaction_with_events.argtypes = lib.wotr_snapshot_resume_preaction.argtypes + [C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_resume_preaction_with_events.restype = C.c_void_p
        lib.wotr_snapshot_hunt_protection_actions.argtypes = [C.c_void_p]
        lib.wotr_snapshot_hunt_protection_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_hunt_protection.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_hunt_protection.restype = C.c_void_p
        lib.wotr_snapshot_play_event_menu.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_play_event_menu.restype = C.c_void_p
        lib.wotr_snapshot_apply_simple_event.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_apply_simple_event.restype = C.c_void_p
        lib.wotr_snapshot_gandalf_draw_actions.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_gandalf_draw_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_gandalf_draw.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32]
        lib.wotr_snapshot_apply_gandalf_draw.restype = C.c_void_p
        lib.wotr_snapshot_apply_combat_round_choice.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_combat_round_choice.restype = C.c_void_p
        for name in ('apply_combat_casualty', 'apply_combat_onslaught', 'apply_combat_onslaught_payment', 'combat_round_begin', 'apply_combat_shine', 'apply_combat_pre_target', 'apply_hunt_event_choice', 'apply_combat_advance', 'apply_lure_event_choice'):
            fn = getattr(lib, 'wotr_snapshot_' + name)
            fn.argtypes = lib.wotr_snapshot_apply_combat_round_choice.argtypes
            fn.restype = C.c_void_p
        lib.wotr_snapshot_apply_combat_card.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.c_int32,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_combat_card.restype=C.c_void_p
        lib.wotr_snapshot_combat_entry_actions.argtypes=[C.c_void_p,C.c_void_p,C.c_int32]
        lib.wotr_snapshot_combat_entry_actions.restype=C.c_void_p
        lib.wotr_snapshot_combat_phase_actions.argtypes=lib.wotr_snapshot_combat_entry_actions.argtypes
        lib.wotr_snapshot_combat_phase_actions.restype=C.c_void_p
        lib.wotr_snapshot_apply_combat_phase.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.c_int64,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_combat_phase.restype=C.c_void_p
        lib.wotr_snapshot_combat_relocate.argtypes=[C.c_void_p,C.c_void_p,C.c_char_p,C.c_char_p,C.c_int32,C.c_int64,C.c_int64,C.c_int64,C.c_int64,C.c_int32,C.c_int32]
        lib.wotr_snapshot_combat_relocate.restype=C.c_void_p
        for name in ('apply_hunt_event','apply_misc_event','apply_lure_event','apply_hunt_table_event','apply_recruit_event_traced','apply_companion_event','apply_shadow_lengthens_event','apply_nazgul_hunt_event','apply_faramir_event','apply_gwaihir_event'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)];fn.restype=C.c_void_p
        lib.wotr_snapshot_hunt_event_choice_actions.argtypes=[C.c_void_p]
        lib.wotr_snapshot_hunt_event_choice_actions.restype=C.c_void_p
        lib.wotr_snapshot_lure_choice_actions.argtypes=[C.c_void_p]
        lib.wotr_snapshot_lure_choice_actions.restype=C.c_void_p
        lib.wotr_snapshot_misc_event_choice_actions.argtypes=[C.c_void_p,C.c_int32]
        lib.wotr_snapshot_misc_event_choice_actions.restype=C.c_void_p
        lib.wotr_snapshot_apply_misc_event_choice.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64]
        lib.wotr_snapshot_apply_misc_event_choice.restype=C.c_void_p
        for name in ('apply_recruit_event_choice','apply_overstack_choice','apply_hill_trolls_choice','apply_target_recruit_choice','apply_shadows_gather_choice','apply_power_payment','apply_half_orcs_choice','apply_pits_choice'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=lib.wotr_snapshot_apply_misc_event_choice.argtypes;fn.restype=C.c_void_p
        lib.wotr_snapshot_apply_recruit_event.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_char_p]
        lib.wotr_snapshot_apply_recruit_event.restype=C.c_void_p
        lib.wotr_snapshot_apply_hill_trolls_event.argtypes=lib.wotr_snapshot_apply_recruit_event.argtypes
        lib.wotr_snapshot_apply_hill_trolls_event.restype=C.c_void_p
        for name in ('apply_target_recruit_event','apply_shadow_recruit_event','apply_shadows_gather_event','apply_power_event','apply_power_kill','apply_half_orcs_event','apply_pits_event'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=lib.wotr_snapshot_apply_recruit_event.argtypes;fn.restype=C.c_void_p
        for name in ('token_actions','recruit_event_choice_actions','hill_trolls_choice_actions','target_recruit_choice_actions','shadows_gather_choice_actions','power_kill_actions','power_payment_actions','half_orcs_choice_actions','faramir_choice_actions','card_casualty_actions','pits_choice_actions'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int32];fn.restype=C.c_void_p
        for name in ('dwarven_ring_actions','overstack_choice_actions','worn_choice_actions','crebain_choice_actions','companion_event_choice_actions','shadow_lengthens_choice_actions','nazgul_hunt_choice_actions','gwaihir_choice_actions','palantir_draw_actions'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=[C.c_void_p,C.c_int32];fn.restype=C.c_void_p
        lib.wotr_snapshot_apply_combat_retreat.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_combat_retreat.restype=C.c_void_p
        lib.wotr_snapshot_apply_token.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int32,C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_token.restype=C.c_void_p
        lib.wotr_snapshot_apply_dwarven_ring.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_dwarven_ring.restype=C.c_void_p
        lib.wotr_snapshot_apply_worn_choice.argtypes=lib.wotr_snapshot_apply_dwarven_ring.argtypes
        lib.wotr_snapshot_apply_worn_choice.restype=C.c_void_p
        for name in ('apply_crebain_choice','apply_companion_event_choice','apply_shadow_lengthens_choice','apply_nazgul_hunt_choice','apply_faramir_choice','apply_card_casualty_choice','apply_gwaihir_choice','apply_palantir_draw'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=lib.wotr_snapshot_apply_worn_choice.argtypes;fn.restype=C.c_void_p
        for name in ('nazgul_move_actions','will_special_actions'):
            fn=getattr(lib,'wotr_snapshot_'+name);fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int32];fn.restype=C.c_void_p
        lib.wotr_snapshot_nazgul_chain_actions.argtypes=[C.c_void_p,C.c_int32]
        lib.wotr_snapshot_nazgul_chain_actions.restype=C.c_void_p
        lib.wotr_snapshot_apply_nazgul_move.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_char_p,C.c_char_p,C.c_int64,C.c_int32]
        lib.wotr_snapshot_apply_nazgul_move.restype=C.c_void_p
        lib.wotr_snapshot_apply_nazgul_chain.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64]
        lib.wotr_snapshot_apply_nazgul_chain.restype=C.c_void_p
        lib.wotr_snapshot_apply_will_special.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_int32,C.c_char_p]
        lib.wotr_snapshot_apply_will_special.restype=C.c_void_p
        lib.wotr_snapshot_army_attack_menu.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_int32]
        lib.wotr_snapshot_army_attack_menu.restype=C.c_void_p
        lib.wotr_snapshot_apply_army_attack.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int32,C.c_int64,C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_army_attack.restype=C.c_void_p
        lib.wotr_snapshot_combat_setup.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_int32, C.c_int32]
        lib.wotr_snapshot_combat_setup.restype = C.c_void_p
        lib.wotr_snapshot_apply_dice_event.argtypes = lib.wotr_snapshot_apply_simple_event.argtypes
        lib.wotr_snapshot_apply_dice_event.restype = C.c_void_p
        lib.wotr_snapshot_dice_event_choice_actions.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_dice_event_choice_actions.restype = C.c_void_p
        lib.wotr_snapshot_apply_dice_event_choice.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64]
        lib.wotr_snapshot_apply_dice_event_choice.restype = C.c_void_p
        lib.wotr_snapshot_fellowship_declare_with_rng.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_fellowship_declare_with_rng.restype = C.c_void_p
        lib.wotr_snapshot_fellowship_tail_with_events.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_void_p), C.POINTER(C.c_int32), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_fellowship_tail_with_events.restype = C.c_void_p
        lib.wotr_snapshot_apply_special_tile_event.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.wotr_snapshot_apply_special_tile_event.restype = C.c_void_p
        lib.wotr_snapshot_apply_army_move.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.c_char_p, C.c_int64, C.c_int64, C.c_int64, C.c_int64, C.c_int32, C.c_int32]
        lib.wotr_snapshot_apply_army_move.restype = C.c_void_p
        lib.wotr_snapshot_apply_muster_action.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.c_char_p, C.c_char_p]
        lib.wotr_snapshot_apply_muster_action.restype = C.c_void_p
        lib.wotr_snapshot_apply_political_action.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_apply_political_action.restype = C.c_void_p
        lib.wotr_snapshot_apply_elven_ring_action.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int64, C.c_char_p]
        lib.wotr_snapshot_apply_elven_ring_action.restype = C.c_void_p
        lib.wotr_snapshot_next_quiet_actor.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
        lib.wotr_snapshot_next_quiet_actor.restype = C.c_int
        lib.wotr_snapshot_destroy.argtypes = [C.c_void_p]
        lib.wotr_snapshot_destroy.restype = None
        for name in ('node_count', 'edge_count', 'byte_count', 'root'):
            fn = getattr(lib, 'wotr_snapshot_' + name)
            fn.argtypes = [C.c_void_p]
            fn.restype = C.c_uint64 if name == 'root' else C.c_size_t
        lib.wotr_snapshot_export.argtypes = [C.c_void_p] + arrays
        lib.wotr_snapshot_export.restype = C.c_int
        lib.wotr_snapshot_replace_integer.argtypes = [C.c_void_p, C.c_uint64, C.c_int64, C.c_int64]
        lib.wotr_snapshot_replace_integer.restype = C.c_int
        for name in ('can_advance_political', 'advance_political'):
            fn = getattr(lib, 'wotr_snapshot_' + name)
            fn.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
            fn.restype = C.c_int
        lib.wotr_snapshot_encode_prefix.argtypes = [C.c_void_p, C.c_int32,
            C.POINTER(C.c_char_p), C.c_size_t, C.POINTER(C.c_char_p), C.c_size_t,
            C.POINTER(C.c_float), C.c_size_t]
        lib.wotr_snapshot_encode_prefix.restype = C.c_int
        lib.wotr_snapshot_encode_state.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_float), C.c_size_t]
        lib.wotr_snapshot_encode_state.restype = C.c_int
        lib.wotr_snapshot_prepare_observation_tables.argtypes = [C.c_void_p]
        lib.wotr_snapshot_prepare_observation_tables.restype = C.c_int
        lib.wotr_snapshot_use_elven_ring.argtypes = [C.c_void_p, C.c_int32, C.c_int64, C.c_char_p, C.POINTER(C.c_char), C.c_size_t]
        lib.wotr_snapshot_use_elven_ring.restype = C.c_int
        lib.wotr_snapshot_ring_waiver_active.argtypes = [C.c_void_p, C.c_int32]
        lib.wotr_snapshot_ring_waiver_active.restype = C.c_int
        _LIB = lib
        return lib


def _error(lib):
    return RuntimeError((lib.wotr_snapshot_last_error() or b'Native snapshot operation failed').decode('utf-8', 'replace'))


@lru_cache(maxsize=4)
def _encoding_tables(regions, nations):
    for value in (*regions, *nations):
        if type(value) is not str or not value or '\0' in value:
            raise ValueError('Encoding table names must be nonempty strings without NUL')
    return ((C.c_char_p * len(regions))(*(key.encode('ascii') for key in regions)),
            (C.c_char_p * len(nations))(*(name.encode('ascii') for name in nations)))


class NativeSnapshot:
    """Owns complete registered graph data and full observations; limited rules."""
    native_rules = False

    def __init__(self, lib, pointer):
        if not pointer:
            raise _error(lib)
        self._lib, self._pointer, self._lock = lib, pointer, threading.RLock()
        self.counts = tuple(int(getattr(lib, 'wotr_snapshot_' + name)(pointer)) for name in ('node_count', 'edge_count', 'byte_count'))
        self.root = int(lib.wotr_snapshot_root(pointer))

    @classmethod
    def from_graph(cls, graph):
        if type(graph.root) is not int or not 0 <= graph.root < len(graph.values):
            raise ValueError('Invalid graph root')
        if not isinstance(graph.values, C.Array) or graph.values._type_ is not Value:
            raise ValueError('Expected an ABI Value array')
        if not isinstance(graph.edges, C.Array) or graph.edges._type_ is not C.c_uint64 or type(graph.data) is not bytes:
            raise ValueError('Expected uint64 edge array and bytes')
        lib = library()
        data = (C.c_uint8 * len(graph.data)).from_buffer_copy(graph.data)
        pointer = lib.wotr_snapshot_create(graph.values, len(graph.values), graph.edges, len(graph.edges), data, len(data), graph.root)
        return cls(lib, pointer)

    def _require_open(self):
        if not self._pointer:
            raise RuntimeError('Native snapshot is closed')

    def export_graph(self):
        with self._lock:
            self._require_open()
            nv, ne, nb = self.counts
            values, edges, data = (Value * nv)(), (C.c_uint64 * ne)(), (C.c_uint8 * nb)()
            if self._lib.wotr_snapshot_export(self._pointer, values, nv, edges, ne, data, nb):
                raise _error(self._lib)
            return Graph(values, edges, bytes(data), self.root)

    def unpack(self, ai):
        return decode_graph(ai, self.export_graph())

    def unpack_metadata(self, ai):
        return decode_graph(ai, self.export_graph(), metadata=True)

    def clone(self):
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_clone(self._pointer))

    def action_clone(self):
        """Ordinary ai.clone_state semantics, including defaults and aliases."""
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_action_clone(self._pointer))

    def action_clone_cache_info(self):
        """Per-handle hit/miss counters and retained native graph payload bytes."""
        with self._lock:
            self._require_open()
            hits, misses, size = C.c_uint64(), C.c_uint64(), C.c_uint64()
            if self._lib.wotr_snapshot_action_clone_cache_info(self._pointer, C.byref(hits), C.byref(misses), C.byref(size)):
                raise _error(self._lib)
            return dict(hits=hits.value, misses=misses.value, bytes=size.value)

    def victory(self, ai, *, end_of_turn=False, search=False):
        """Native victory conditions, with provisional and terminal results distinct."""
        if type(end_of_turn) is not bool or type(search) is not bool:
            raise ValueError('Victory mode flags must be bool')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        output = (C.c_int64 * 5)()
        with self._lock:
            self._require_open()
            if self._lib.wotr_snapshot_victory(self._pointer, tables._pointer, end_of_turn, search, output, 5):
                raise _error(self._lib)
        return dict(winner=None if output[0] < 0 else output[0],
                    kind=(None, 'military', 'ring_destroyed', 'ring_corruption')[output[1]],
                    fp_vp=output[2], sh_vp=output[3], ended=bool(output[4]))

    def winner(self, ai, end_of_turn=False):
        return self.victory(ai, end_of_turn=end_of_turn, search=True)['winner']

    def political_ring_actions(self, ai, side):
        """Native raw legal-menu subset; decode descriptors only at this boundary."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_political_ring_actions(
                    self._pointer, tables._pointer, side)) as menu:
                return menu.unpack(ai)

    def roll_action_dice(self, rng, side):
        """Return a rolled child, advanced RNG child and ordered enum results."""
        import random_stream
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Dice rolling requires an owned NativeRandom stream')
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        random_out, faces_out = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:
                raise RuntimeError('Native random stream is closed')
            pointer = self._lib.wotr_snapshot_roll_action_dice(
                self._pointer, rng._pointer, side, C.byref(random_out), C.byref(faces_out))
            if not pointer:
                raise _error(self._lib)
        child = NativeSnapshot(self._lib, pointer)
        child_rng = random_stream.NativeRandom._adopt(rng._lib, random_out.value)
        try:
            with NativeSnapshot(self._lib, faces_out.value) as faces:
                result = faces.unpack(bridge.load_reference())
            return child, child_rng, result
        except BaseException:
            child.close(); child_rng.close()
            raise

    def combat_reroll_stage(self, rng):
        """Run only paused combat stage 9→10, returning state/RNG children."""
        import combat_context
        combat_context.require_native_context()
        import random_stream
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Combat rerolls require an owned NativeRandom stream')
        output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:
                raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_reroll_stage(
                self._pointer, rng._pointer, C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def combat_primary_stage(self, rng, target=None):
        """Advance stages 5–8, stopping at an unresolved learned Might choice."""
        import combat_context
        combat_context.require_native_context()
        import random_stream
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Combat primary rolls require an owned NativeRandom stream')
        if target is not None and (type(target) is not str or '\0' in target):
            raise ValueError('Might target must be a string without NUL or None')
        output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:
                raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_primary_stage(
                self._pointer, rng._pointer, None if target is None else target.encode('utf-8'), C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def combat_pre_stage(self, rng, selection_index=None, *, battle_context=False):
        return self._combat_advance(rng, selection_index, battle_context, 'pre_stage')

    def combat_round_stage(self, rng, selection_index=None, *, battle_context=False):
        """Advance an inner round to its next decision or deferred casualty boundary."""
        return self._combat_advance(rng, selection_index, battle_context, 'round_stage')

    def _combat_advance(self, rng, selection_index, battle_context, stage):
        import combat_context, random_stream
        combat_context.require_native_context()
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Pre-combat requires an owned NativeRandom stream')
        if selection_index is not None and (type(selection_index) is not int or not 0 <= selection_index < (1 << 63)):
            raise ValueError('Combat choice must be nonnegative int64 or None')
        if type(battle_context) is not bool:
            raise ValueError('Battle context flag must be bool')
        output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_combat_' + stage)(
                self._pointer, rng._pointer, -1 if selection_index is None else selection_index,
                battle_context, C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def combat_deferred_post(self, ai, rng):
        import combat_context, random_stream, observation_tables
        combat_context.require_native_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Deferred combat requires an owned NativeRandom stream')
        tables = observation_tables.native_tables(ai);output, info = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_deferred_post(
                self._pointer, tables._pointer, rng._pointer, C.byref(output), C.byref(info)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, info.value) as payload:details = payload.unpack(ai)
            return child, advanced, details
        except BaseException:
            child.close();advanced.close();raise

    def combat_allocation_settle(self, ai, rng, enter=False):
        import combat_context, random_stream, observation_tables
        combat_context.require_native_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Allocation settle requires an owned NativeRandom stream')
        if type(enter) is not bool:raise ValueError('Allocation entry flag must be bool')
        tables = observation_tables.native_tables(ai);output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_allocation_settle(
                self._pointer, tables._pointer, rng._pointer, enter, C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def combat_resume_settle(self, ai, rng, index=None):
        import combat_context, random_stream, observation_tables
        combat_context.require_native_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Combat resume requires an owned NativeRandom stream')
        if index is not None and (type(index) is not int or not 0 <= index < (1 << 63)):raise ValueError('Choice must be nonnegative int64 or None')
        tables = observation_tables.native_tables(ai);output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_resume_settle(
                self._pointer, tables._pointer, rng._pointer, -1 if index is None else index, C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def resume_preaction(self, ai, rng, before, kind, *, with_events=False):
        import combat_context, random_stream, observation_tables
        from contextlib import ExitStack
        combat_context.require_native_hunt_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Preaction requires an owned NativeRandom stream')
        if type(before) is not NativeSnapshot or before._lib is not self._lib:raise ValueError('Previous state must use the same native library')
        if type(kind) is not str or '\0' in kind:raise ValueError('Action type must be text without NUL')
        if type(with_events) is not bool:raise ValueError('Event output mode must be Boolean')
        tables = observation_tables.native_tables(ai);output,events = C.c_void_p(),C.c_void_p()
        with ExitStack() as stack:
            for owner in sorted({id(x):x for x in (self, before)}.values(),key=id):stack.enter_context(owner._lock)
            stack.enter_context(rng._lock);self._require_open();before._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_resume_preaction_with_events(
                before._pointer, self._pointer, tables._pointer, rng._pointer, kind.encode('utf-8'), C.byref(output), C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai) if with_events else None
            return (child,advanced,annotations) if with_events else (child,advanced)
        except BaseException:child.close();advanced.close();raise

    def protection_actions(self, ai):
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_protection_actions(self._pointer)) as result:return result.unpack(ai)

    def apply_hunt_protection(self, ai, rng, index):
        import random_stream, observation_tables, combat_context
        combat_context.require_native_event_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Hunt protection requires an owned NativeRandom stream')
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Choice index must be nonnegative int64')
        tables = observation_tables.native_tables(ai);output, events = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_hunt_protection(
                self._pointer, tables._pointer, rng._pointer, index, C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:child.close();advanced.close();raise

    def play_event_menu(self, ai, side, index, category):
        import observation_tables
        self._event_arguments(side,index,category)
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_play_event_menu(self._pointer,tables._pointer,side,index,category.encode('utf-8')))

    @staticmethod
    def _event_arguments(side,index,text):
        if type(side) is not int or side not in (0,1):raise ValueError('Side must be FP or SH')
        if type(index) is not int or not -(1 << 63) <= index < (1 << 63):raise ValueError('Die index must be int64')
        if type(text) is not str or '\0' in text:raise ValueError('Event parameter must be text without NUL')

    def apply_simple_event(self, ai, side, index, card_id):
        import observation_tables
        self._event_arguments(side,index,card_id)
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_simple_event(self._pointer,tables._pointer,side,index,card_id.encode('utf-8')))

    def apply_gandalf_draw(self, ai, side, draw):
        import observation_tables
        if type(side) is not int or side not in (0,1):raise ValueError('Side must be FP or SH')
        if type(draw) is not bool:raise ValueError('Gandalf choice must be bool')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_gandalf_draw(self._pointer,tables._pointer,side,draw))

    def apply_combat_round_choice(self, ai, rng, index):
        return self._apply_combat_integer_action(ai, rng, index, 'apply_combat_round_choice')

    def apply_combat_casualty(self, ai, rng, index):
        return self._apply_combat_integer_action(ai, rng, index, 'apply_combat_casualty')

    def apply_combat_onslaught(self, ai, rng, count):
        return self._apply_combat_integer_action(ai, rng, count, 'apply_combat_onslaught')

    def apply_combat_onslaught_payment(self, ai, rng, index):
        return self._apply_combat_integer_action(ai, rng, index, 'apply_combat_onslaught_payment')

    def combat_round_begin(self, ai, rng, is_siege=None):
        if is_siege is not None and type(is_siege) is not bool:raise ValueError('Siege override must be Boolean or None')
        return self._apply_combat_integer_action(ai,rng,0 if is_siege is None else 1+int(is_siege),'combat_round_begin')

    def apply_combat_shine(self, ai, rng, shine):
        if type(shine) is not bool:raise ValueError('Shine choice must be Boolean')
        return self._apply_combat_integer_action(ai,rng,int(shine),'apply_combat_shine')

    def apply_combat_pre_target(self, ai, rng, index):
        return self._apply_combat_integer_action(ai,rng,index,'apply_combat_pre_target')

    def apply_combat_advance(self, ai, rng, index):
        return self._apply_combat_integer_action(ai,rng,index,'apply_combat_advance')

    def apply_combat_phase(self, ai, rng, kind, *, elites_back=0, nation=None):
        import combat_context,random_stream,observation_tables
        combat_context.require_native_action_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Combat phase requires owned RNG')
        if type(kind) is not str or '\0' in kind or type(elites_back) is not int or not 0<=elites_back<(1<<63):raise ValueError('Invalid combat phase arguments')
        if nation is not None:
            if isinstance(nation,ai.Nation):nation=nation.value
            if kind!='siege_press' or type(nation) is not str or '\0' in nation:raise ValueError('Explicit siege payment requires a nation string')
            _kind,side=self.pending_actor(ai)
            with NativeSnapshot.from_graph(encode_graph(ai,[(-1,kind,{'nation':nation})])) as menu:
                return self.apply_descriptor_random(ai,rng,side,menu,0)
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_combat_phase(self._pointer,tables._pointer,rng._pointer,kind.encode('utf8'),elites_back,C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def combat_phase_actions(self, ai, side):
        import observation_tables
        if type(side) is not int or side not in (0,1):raise ValueError('Combat actor must be FP or SH')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib,self._lib.wotr_snapshot_combat_phase_actions(self._pointer,tables._pointer,side)) as result:return result.unpack(ai)

    def apply_hunt_event_choice(self, ai, rng, index):
        return self._apply_combat_integer_action(ai,rng,index,'apply_hunt_event_choice')

    def apply_hunt_table_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_hunt_table_event')

    def apply_combat_retreat(self, ai, rng, destination):
        if type(destination) is not str or '\0' in destination:raise ValueError('Retreat destination must be text without NUL')
        return self._apply_named_rng_action(ai,rng,'apply_combat_retreat',(destination.encode('utf8'),),'combat')

    def apply_token(self, ai, rng, side, kind, value):
        if type(side) is not int or side not in (0,1):raise ValueError('Token side must be FP or SH')
        if any(type(v) is not str or '\0' in v for v in (kind,value)):raise ValueError('Invalid token kind/value')
        return self._apply_named_rng_action(ai,rng,'apply_token',(side,kind.encode('utf8'),value.encode('utf8')),'bid')

    def apply_dwarven_ring(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid Dwarven Ring actor/die')
        return self._apply_named_rng_action(ai,rng,'apply_dwarven_ring',(side,index),'bid')

    def _apply_named_rng_action(self, ai, rng, operation, arguments, guard):
        import combat_context,random_stream,observation_tables
        if guard=='combat':combat_context.require_native_action_context()
        else:combat_context.require_native_bid_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Native action requires owned RNG')
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,getattr(self._lib,'wotr_snapshot_'+operation)(self._pointer,tables._pointer,rng._pointer,*arguments,C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def apply_target_recruit_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_target_recruit_event')

    def apply_shadow_recruit_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_shadow_recruit_event')

    def apply_shadows_gather_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_shadows_gather_event')

    def apply_target_recruit_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_target_recruit_choice')

    def apply_shadows_gather_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_shadows_gather_choice')

    def apply_crebain_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid event actor/index')
        return self._apply_named_rng_action(ai,rng,'apply_crebain_choice',(side,index),'event')

    def apply_companion_event_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid event actor/index')
        return self._apply_named_rng_action(ai,rng,'apply_companion_event_choice',(side,index),'event')

    def apply_companion_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_companion_event')

    def apply_shadow_lengthens_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_shadow_lengthens_event')

    def apply_shadow_lengthens_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid event choice actor/index')
        return self._apply_named_rng_action(ai,rng,'apply_shadow_lengthens_choice',(side,index),'event')

    def apply_palantir_draw(self, ai, rng, side, index):
        """Resolve the learned mandatory Palantir deck choice."""
        return self._apply_named_rng_action(ai,rng,'apply_palantir_draw',(side,index),'event')

    def apply_nazgul_hunt_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_nazgul_hunt_event')

    def apply_nazgul_hunt_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid event choice actor/index')
        return self._apply_named_rng_action(ai,rng,'apply_nazgul_hunt_choice',(side,index),'event')

    def apply_faramir_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_faramir_event')

    def apply_faramir_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid Faramir choice')
        return self._apply_named_rng_action(ai,rng,'apply_faramir_choice',(side,index),'event')

    def apply_card_casualty_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid card casualty choice')
        return self._apply_named_rng_action(ai,rng,'apply_card_casualty_choice',(side,index),'event')

    def apply_power_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_power_event')

    def apply_power_kill(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_power_kill')

    def apply_power_payment(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_power_payment')

    def apply_half_orcs_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_half_orcs_event')

    def apply_half_orcs_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_half_orcs_choice')

    def apply_pits_event(self, ai, side, index, card_id):
        return self._apply_native_card_event(ai,side,index,card_id,'apply_pits_event')

    def apply_pits_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_pits_choice')

    def apply_gwaihir_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_gwaihir_event')

    def apply_gwaihir_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid Gwaihir choice')
        return self._apply_named_rng_action(ai,rng,'apply_gwaihir_choice',(side,index),'event')

    def _apply_native_card_event(self, ai, side, index, card_id, operation):
        import observation_tables
        self._event_arguments(side,index,card_id);tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,getattr(self._lib,'wotr_snapshot_'+operation)(self._pointer,tables._pointer,side,index,card_id.encode('utf8')))

    def apply_recruit_event(self, ai, side, index, card_id):
        import observation_tables
        self._event_arguments(side,index,card_id);tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_recruit_event(self._pointer,tables._pointer,side,index,card_id.encode('utf8')))

    def apply_recruit_event_traced(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_recruit_event_traced')

    def apply_hill_trolls_event(self, ai, side, index, card_id):
        import observation_tables
        self._event_arguments(side,index,card_id);tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_hill_trolls_event(self._pointer,tables._pointer,side,index,card_id.encode('utf8')))

    def apply_hill_trolls_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_hill_trolls_choice')

    def apply_worn_choice(self, ai, rng, side, index):
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid Worn actor/index')
        return self._apply_named_rng_action(ai,rng,'apply_worn_choice',(side,index),'event')

    def apply_recruit_event_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_recruit_event_choice')

    def apply_overstack_choice(self, ai, side, index):
        return self._apply_side_index_choice(ai,side,index,'apply_overstack_choice')

    def _apply_side_index_choice(self, ai, side, index, operation):
        import observation_tables
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid choice actor/index')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,getattr(self._lib,'wotr_snapshot_'+operation)(self._pointer,tables._pointer,side,index))

    def apply_lure_event_choice(self, ai, rng, index):
        return self._apply_combat_integer_action(ai,rng,index,'apply_lure_event_choice')

    def apply_lure_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_lure_event')

    def lure_choice_actions(self, ai):
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib,self._lib.wotr_snapshot_lure_choice_actions(self._pointer)) as result:return result.unpack(ai)

    def combat_entry_actions(self, ai, side):
        import observation_tables
        if type(side) is not int or side not in (0,1):raise ValueError('Combat actor must be FP or SH')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib,self._lib.wotr_snapshot_combat_entry_actions(self._pointer,tables._pointer,side)) as result:return result.unpack(ai)

    def hunt_event_choice_actions(self, ai):
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib,self._lib.wotr_snapshot_hunt_event_choice_actions(self._pointer)) as result:return result.unpack(ai)

    def apply_combat_card(self, ai, rng, card_id=None, from_table=False):
        import combat_context,random_stream,observation_tables
        combat_context.require_native_action_context()
        if type(rng) is not random_stream.NativeRandom or type(from_table) is not bool:raise ValueError('Combat card requires owned RNG and Boolean source')
        if card_id is not None and (type(card_id) is not str or '\0' in card_id):raise ValueError('Card ID must be text without NUL or None')
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_combat_card(self._pointer,tables._pointer,rng._pointer,None if card_id is None else card_id.encode('utf8'),from_table,C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def combat_relocate(self, ai, source, destination, side, regular=None, elite=None, leaders=None, nazgul=None, battle_context=False, defer_stacking=False):
        import combat_context as context,observation_tables
        context.require_native_context()
        counts=tuple(-1 if value is None else value for value in (regular,elite,leaders,nazgul))
        if type(side) is not int or side not in (0,1) or any(type(v) is not int or not -1<=v<(1<<63) for v in counts):raise ValueError('Invalid relocation actor/counts')
        if any(type(v) is not str or '\0' in v for v in (source,destination)) or type(battle_context) is not bool or type(defer_stacking) is not bool:raise ValueError('Invalid relocation regions/mode')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_combat_relocate(self._pointer,tables._pointer,source.encode('utf8'),destination.encode('utf8'),side,*counts,battle_context,defer_stacking))

    def apply_hunt_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_hunt_event')

    def apply_misc_event(self, ai, rng, side, index, card_id):
        return self._apply_rng_card_event(ai,rng,side,index,card_id,'apply_misc_event')

    def _apply_rng_card_event(self, ai, rng, side, index, card_id, operation):
        import combat_context,random_stream,observation_tables
        combat_context.require_native_event_context();self._event_arguments(side,index,card_id)
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Event requires owned NativeRandom')
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,getattr(self._lib,'wotr_snapshot_'+operation)(self._pointer,tables._pointer,rng._pointer,side,index,card_id.encode('utf8'),C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def apply_misc_event_choice(self, ai, side, index):
        import observation_tables
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0<=index<(1<<63):raise ValueError('Invalid event choice actor/index')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_misc_event_choice(self._pointer,tables._pointer,side,index))

    def army_attack_menu(self, ai, side, index, *, require_leader=False):
        import combat_context,observation_tables
        combat_context.require_native_context()
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0 <= index < (1<<63) or type(require_leader) is not bool:raise ValueError('Invalid attack menu arguments')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_army_attack_menu(self._pointer,tables._pointer,side,index,require_leader))

    def apply_army_attack(self, ai, rng, side, index, source, destination):
        import combat_context,random_stream,observation_tables
        combat_context.require_native_action_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Attack requires owned NativeRandom')
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0 <= index < (1<<63):raise ValueError('Invalid attack payment arguments')
        if any(type(v) is not str or '\0' in v for v in (source,destination)):raise ValueError('Attack regions must be strings without NUL')
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_army_attack(self._pointer,tables._pointer,rng._pointer,side,index,source.encode('utf8'),destination.encode('utf8'),C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def apply_nazgul_move(self, ai, side, index, source, destination, count=-1, witch=False):
        import observation_tables
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0 <= index < (1<<63):raise ValueError('Invalid Nazgul payment arguments')
        if type(count) is not int or not -1 <= count < (1<<63) or type(witch) is not bool:raise ValueError('Invalid Nazgul selection')
        if any(type(v) is not str or '\0' in v for v in (source,destination)):raise ValueError('Regions must be strings without NUL')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_nazgul_move(self._pointer,tables._pointer,side,index,source.encode('utf8'),destination.encode('utf8'),count,witch))

    def apply_nazgul_chain(self, ai, side, index):
        import observation_tables
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0 <= index < (1<<63):raise ValueError('Invalid Nazgul continuation arguments')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_nazgul_chain(self._pointer,tables._pointer,side,index))

    def apply_will_special(self, ai, side, index, gandalf, region=''):
        import observation_tables
        if type(side) is not int or side not in (0,1) or type(index) is not int or not 0 <= index < (1<<63) or type(gandalf) is not bool:raise ValueError('Invalid promotion arguments')
        if type(region) is not str or '\0' in region:raise ValueError('Promotion region must be a string without NUL')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_will_special(self._pointer,tables._pointer,side,index,gandalf,region.encode('utf8')))

    def _apply_combat_integer_action(self, ai, rng, index, operation):
        import combat_context, random_stream, observation_tables
        if operation in ('apply_hunt_event_choice','apply_lure_event_choice'):combat_context.require_native_event_context()
        else:combat_context.require_native_action_context()
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Combat action requires an owned NativeRandom stream')
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Choice index must be nonnegative int64')
        tables = observation_tables.native_tables(ai);output, events = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_' + operation)(
                self._pointer, tables._pointer, rng._pointer, index, C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:child.close();advanced.close();raise

    def combat_setup(self, ai, attacker, defender, attacker_card=None, defender_card=None, is_siege=False, attacker_side=None):
        import combat_context, observation_tables
        combat_context.require_native_context()
        if type(is_siege) is not bool:raise ValueError('Siege must be Boolean')
        if attacker_side is not None and (type(attacker_side) is not int or attacker_side not in (0,1)):raise ValueError('Attacker side must be FP, SH or None')
        strings=(attacker,defender,'' if attacker_card is None else attacker_card,'' if defender_card is None else defender_card)
        if any(type(value) is not str or '\0' in value for value in strings):raise ValueError('Combat regions and card IDs must be strings without NUL')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_combat_setup(self._pointer,tables._pointer,*(v.encode('utf8') for v in strings),is_siege,-1 if attacker_side is None else attacker_side))

    def apply_dice_event(self, ai, side, index, card_id):
        import observation_tables
        self._event_arguments(side,index,card_id);tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_dice_event(self._pointer,tables._pointer,side,index,card_id.encode('utf8')))

    def apply_dice_event_choice(self, ai, side, index):
        import observation_tables
        if type(side) is not int or side not in (0,1):raise ValueError('Event owner must be FP or SH')
        if type(index) is not int or not 0 <= index < (1<<63):raise ValueError('Choice index must be nonnegative int64')
        tables=observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib,self._lib.wotr_snapshot_apply_dice_event_choice(self._pointer,tables._pointer,side,index))

    def apply_fellowship_declare_with_rng(self, ai, rng, destination='', pass_declaration=False):
        import random_stream,observation_tables
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Declaration requires an owned NativeRandom stream')
        if type(destination) is not str or '\0' in destination or type(pass_declaration) is not bool:raise ValueError('Invalid declaration arguments')
        tables=observation_tables.native_tables(ai);output,events=C.c_void_p(),C.c_void_p()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,self._lib.wotr_snapshot_fellowship_declare_with_rng(self._pointer,tables._pointer,rng._pointer,destination.encode('utf8'),pass_declaration,C.byref(output),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,annotations
        except BaseException:child.close();advanced.close();raise

    def tail_with_events(self, ai, rng, begin=True):
        import random_stream,observation_tables
        if type(rng) is not random_stream.NativeRandom or type(begin) is not bool:raise ValueError('Tail requires an owned RNG and Boolean mode')
        tables=observation_tables.native_tables(ai);output,events,finished=C.c_void_p(),C.c_void_p(),C.c_int32()
        with self._lock,rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child=NativeSnapshot(self._lib,self._lib.wotr_snapshot_fellowship_tail_with_events(self._pointer,tables._pointer,rng._pointer,begin,C.byref(output),C.byref(finished),C.byref(events)))
        advanced=random_stream.NativeRandom._adopt(rng._lib,output.value)
        try:
            with NativeSnapshot(self._lib,events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child,advanced,bool(finished.value),annotations
        except BaseException:child.close();advanced.close();raise

    def apply_special_tile_event(self, ai, rng, side, index, card_id):
        import random_stream, observation_tables
        self._event_arguments(side,index,card_id)
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Tile event requires an owned NativeRandom stream')
        tables = observation_tables.native_tables(ai);output, events = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_special_tile_event(
                self._pointer, tables._pointer, rng._pointer, side, index, card_id.encode('utf-8'), C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations=payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:child.close();advanced.close();raise

    def combat_casualty_menu(self, ai, side, region, hits):
        import combat_context
        combat_context.require_native_context()
        self._casualty_arguments(side, region, hits)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_casualty_menu(
                    self._pointer, side, region.encode('utf-8'), hits)) as result:
                return result.unpack(ai)

    @staticmethod
    def _casualty_arguments(side, region, hits):
        if type(side) is not int or side not in (0, 1):raise ValueError('Casualty side must be FP or SH')
        if type(region) is not str or '\0' in region:raise ValueError('Region must be text without NUL')
        if type(hits) is not int or not 0 <= hits < (1 << 63):raise ValueError('Hits must be nonnegative int64')

    def combat_casualty_operation(self, ai, side, region, hits, nation, operation):
        import combat_context, observation_tables
        combat_context.require_native_context()
        self._casualty_arguments(side, region, hits)
        if any(type(value) is not str or '\0' in value for value in (nation, operation)):
            raise ValueError('Casualty nation/operation must be text without NUL')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_casualty_operation(
                self._pointer, tables._pointer, side, region.encode('utf-8'), hits, nation.encode('utf-8'), operation.encode('utf-8')))

    def hunt_menu_handle(self, ai, family='damage'):
        import observation_tables
        families = {'damage':0, 'hobbit':1, 'reveal':2}
        if type(family) is not str or family not in families:raise ValueError('Unknown Hunt menu family')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_menu(self._pointer, tables._pointer, families[family]))

    def hunt_actions(self, ai, family='damage'):
        with self.hunt_menu_handle(ai, family) as result:return result.unpack(ai)

    def apply_hunt_choice(self, ai, rng, kind):
        return self._hunt_decision(ai, rng, 0, kind)

    def apply_hunt_guide_choice(self, ai, rng, name):
        return self._hunt_decision(ai, rng, 1, name)

    def apply_hobbit_placement(self, ai, rng, destination):
        return self._hunt_decision(ai, rng, 2, destination)

    def apply_fellowship_reveal(self, ai, rng, destination):
        return self._hunt_decision(ai, rng, 3, destination)

    def _hunt_decision(self, ai, rng, mode, selection):
        import random_stream, observation_tables
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Hunt decision requires an owned NativeRandom stream')
        if type(selection) is not str or '\0' in selection:raise ValueError('Hunt selection must be text without NUL')
        tables = observation_tables.native_tables(ai)
        output, events = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_decision(
                self._pointer, tables._pointer, rng._pointer, mode, selection.encode('utf-8'), C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations = payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:
            child.close();advanced.close();raise

    def fellowship_actions(self, ai, index):
        if type(index) is not int or not -(1 << 63) <= index < (1 << 63):raise ValueError('Die index must be int64')
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_fellowship_actions(self._pointer, index)) as result:
                return result.unpack(ai)

    def apply_fellowship_hide(self, ai, index, strider_ability=False):
        import observation_tables
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Die index must be nonnegative int64')
        if type(strider_ability) is not bool:raise ValueError('Strider ability must be bool')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_fellowship_hide(
                self._pointer, tables._pointer, index, strider_ability))

    def hunt_params(self, ai):
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_params(self._pointer, tables._pointer)) as result:
                return result.unpack(ai)

    def combat_publish_paused(self, rng):
        import combat_context, random_stream
        combat_context.require_native_context()
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Combat publication requires an owned NativeRandom stream')
        output = C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_publish_paused(
                self._pointer, rng._pointer, C.byref(output)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value)

    def apply_fellowship_move(self, ai, rng, index):
        import os, random_stream, observation_tables
        if os.environ.get('MORDOR_STOP_KEEPS_STEP'):
            raise ValueError('Native movement requires standard Mordor stop behavior')
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Fellowship move requires an owned NativeRandom stream')
        if type(index) is not int or not 0 <= index < (1 << 63):
            raise ValueError('Fellowship die index must be nonnegative int64')
        tables = observation_tables.native_tables(ai)
        output, events = C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_fellowship_move(
                self._pointer, tables._pointer, rng._pointer, index, C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations = payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:
            child.close();advanced.close();raise

    def resolve_hunt_tile(self, ai, rng, successes=1, mordor=False, track=False, eye_no_effect=False):
        import random_stream, observation_tables
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Hunt tile requires an owned NativeRandom stream')
        if type(successes) is not int or not -(1 << 63) <= successes < (1 << 63):
            raise ValueError('Hunt successes must be int64')
        if any(type(flag) is not bool for flag in (mordor, track, eye_no_effect)):
            raise ValueError('Hunt tile flags must be bool')
        tables = observation_tables.native_tables(ai)
        output, value, events = C.c_void_p(), C.c_void_p(), C.c_void_p()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_tile(
                self._pointer, tables._pointer, rng._pointer, successes, mordor, track, eye_no_effect,
                C.byref(output), C.byref(value), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, value.value) as result, NativeSnapshot(self._lib, events.value) as annotations:
                return child, advanced, result.unpack(ai), annotations.unpack(ai)
        except BaseException:
            child.close();advanced.close();raise

    def hunt_roll(self, ai, rng, count, threshold=6, fp_modifier=0):
        import random_stream, observation_tables
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Hunt requires an owned NativeRandom stream')
        for value in (count, threshold, fp_modifier):
            if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
                raise ValueError('Hunt parameters must be int64')
        tables = observation_tables.native_tables(ai)
        output, events, successes = C.c_void_p(), C.c_void_p(), C.c_int64()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_hunt_roll(
                self._pointer, tables._pointer, rng._pointer, count, threshold, fp_modifier,
                C.byref(output), C.byref(successes), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations = payload.unpack_metadata(ai)
            return child, advanced, successes.value, annotations
        except BaseException:
            child.close();advanced.close();raise

    def combat_post_stage(self, selection_index=None):
        """Advance post-reroll cards/choices; immediate casualties remain separate."""
        import combat_context
        combat_context.require_native_context()
        if selection_index is not None and (type(selection_index) is not int or not 0 <= selection_index < (1 << 63)):
            raise ValueError('Combat choice index must be nonnegative int64 or None')
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_combat_post_stage(
                self._pointer, -1 if selection_index is None else selection_index))

    def begin_fellowship_tail(self, ai, rng):
        return self._fellowship_tail(ai, rng, True)

    def resume_fellowship_tail(self, ai, rng):
        return self._fellowship_tail(ai, rng, False)

    def _fellowship_tail(self, ai, rng, begin):
        import random_stream
        if type(rng) is not random_stream.NativeRandom:
            raise ValueError('Fellowship tail requires an owned NativeRandom stream')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        output, finished = C.c_void_p(), C.c_int32()
        with self._lock, rng._lock:
            self._require_open()
            if not rng._pointer:
                raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_fellowship_tail(
                self._pointer, tables._pointer, rng._pointer, begin, C.byref(output), C.byref(finished)))
        return child, random_stream.NativeRandom._adopt(rng._lib, output.value), bool(finished.value)

    def draw_event_card(self, ai, side, deck):
        """Return child, drawn ID, and trace descriptor for boundary emission."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(deck) is not str or '\0' in deck:
            raise ValueError('Deck must be a string without NUL')
        output = C.c_void_p()
        with self._lock:
            self._require_open()
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_draw_event_card(
                self._pointer, side, deck.encode('utf-8'), C.byref(output)))
        try:
            with NativeSnapshot(self._lib, output.value) as payload:
                card, annotation = payload.unpack(ai)
            return child, card, annotation
        except BaseException:
            child.close(); raise

    def open_hand_limit_discard(self, ai):
        """Return child and whether a learned hand-limit discard is pending."""
        output = C.c_int32()
        with self._lock:
            self._require_open()
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_open_hand_limit_discard(
                self._pointer, C.byref(output)))
        return child, bool(output.value)

    def enforce_settlement_control(self, ai):
        import combat_context, observation_tables
        combat_context.require_native_context()
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_enforce_control(self._pointer, tables._pointer))

    def next_turn(self, ai):
        """Repair settlement control, reset dice and draw, preserving pending hand choices."""
        return self._next_turn(ai, 'next_turn')

    def safe_next_turn(self, ai):
        """Apply the distinct self-play/search rollover, including early terminal stop."""
        return self._next_turn(ai, 'safe_next_turn')

    def cleanup_sieges(self, ai):
        import combat_context, observation_tables
        combat_context.require_native_context()
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_cleanup_sieges(self._pointer, tables._pointer))

    def _next_turn(self, ai, operation):
        import combat_context, observation_tables, game_state as gs
        combat_context.require_native_context()
        fp = ('CHARACTER','CHARACTER','MUSTER','EVENT','MUSTER_ARMY','WILL_OF_WEST')
        sh = ('CHARACTER','ARMY','MUSTER','EVENT','MUSTER_ARMY','EYE')
        if tuple(face.name for face in gs.FP_FACES) != fp or tuple(face.name for face in gs.SH_FACES) != sh:
            raise ValueError('Next turn requires the pinned standard global face lists')
        tables = observation_tables.native_tables(ai)
        won, output = C.c_int32(), C.c_void_p()
        with self._lock:
            self._require_open()
            child = NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_' + operation)(
                self._pointer, tables._pointer, C.byref(won), C.byref(output)))
        try:
            with NativeSnapshot(self._lib, output.value) as payload:annotations = payload.unpack_metadata(ai)
            return child, bool(won.value), annotations
        except BaseException:
            child.close();raise

    def begin_turn_reset(self, ai, new_turn, *, deal_cards=True, bonus_dice=True):
        """Return reset child, Mordor-penalty victory flag and draw annotations.

        Matches begin_turn_reset; next_turn's control reconciliation is separate.
        Trace descriptors are returned for the caller to emit at its boundary.
        """
        if type(new_turn) is not int or not -(1 << 63) <= new_turn < (1 << 63):
            raise ValueError('Turn must be int64')
        if type(deal_cards) is not bool or type(bonus_dice) is not bool:
            raise ValueError('Turn reset flags must be bool')
        if bonus_dice:
            import game_state as gs
            fp = ('CHARACTER','CHARACTER','MUSTER','EVENT','MUSTER_ARMY','WILL_OF_WEST')
            sh = ('CHARACTER','ARMY','MUSTER','EVENT','MUSTER_ARMY','EYE')
            if tuple(face.name for face in gs.FP_FACES) != fp or tuple(face.name for face in gs.SH_FACES) != sh:
                raise ValueError('Bonus dice require the pinned standard global face lists')
        if not hasattr(self._lib, 'wotr_snapshot_begin_turn_reset_settled'):
            raise RuntimeError('Native begin_turn_reset requires the settled-turn export; rebuild the selected native library')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        won, output = C.c_int32(), C.c_void_p()
        with self._lock:
            self._require_open()
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_begin_turn_reset_settled(
                self._pointer, tables._pointer, new_turn, deal_cards, bonus_dice, C.byref(won), C.byref(output)))
        try:
            with NativeSnapshot(self._lib, output.value) as payload:
                annotations = payload.unpack_metadata(ai)
            return child, bool(won.value), annotations
        except BaseException:
            child.close(); raise

    def eye_allocation_actions(self, ai, side):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_eye_allocation_actions(self._pointer, side)) as menu:
                return menu.unpack(ai)

    def apply_eye_allocation(self, ai, count):
        if type(count) is not int or not 0 <= count < (1 << 63):
            raise ValueError('Eye count must be a nonnegative int64')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native eye actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_eye_allocation(self._pointer, tables._pointer, count))

    def army_move_actions(self, ai, side):
        """Raw reference rows, not a complete validated native legal menu.

        Execution may explicitly reject an unported effect or a reference rule
        inconsistency. A driver must validate transitions before using this
        subset; exact row parity does not certify every row as executable.
        """
        return self._table_menu(ai, side, 'army_move_actions')

    def saruman_muster_actions(self, ai, side):
        return self._table_menu(ai, side, 'saruman_muster_actions')

    def companion_political_actions(self, ai, side):
        return self._table_menu(ai, side, 'companion_political_actions')

    def discard_card_actions(self, ai, side):
        return self._table_menu(ai, side, 'discard_card_actions')

    def fellowship_declare_actions(self, ai, side):
        return self._table_menu(ai, side, 'fellowship_declare_actions')

    def guide_actions(self, ai, side):
        with self.menu_handle(ai, side, 'guide') as menu:
            return menu.unpack(ai)

    def apply_guide_choice(self, ai, name):
        if type(name) is not str or '\0' in name:
            raise ValueError('Guide name must be a string without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native guide choice requires ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_guide_choice(
                self._pointer, tables._pointer, name.encode('utf-8')))

    def apply_fellowship_declare(self, ai, region):
        return self._declare(ai, region, False)

    def pass_fellowship_declare(self, ai):
        return self._declare(ai, '', True)

    def _declare(self, ai, region, passing):
        if type(region) is not str or '\0' in region:
            raise ValueError('Declaration region must be a string without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native declaration requires ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_fellowship_declare(
                self._pointer, tables._pointer, region.encode('utf-8'), passing))

    def apply_companion_political_action(self, ai, side, die_index, nation, companion):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Companion die index must be a nonnegative int64')
        for value in (nation, companion):
            if type(value) is not str or '\0' in value:
                raise ValueError('Companion and nation must be strings without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native companion actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_companion_political_action(
                self._pointer, tables._pointer, side, die_index, nation.encode('utf-8'), companion.encode('utf-8')))

    def apply_discard_card(self, ai, side, card_id):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(card_id) is not str or '\0' in card_id:
            raise ValueError('Card ID must be a string without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native discard actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_discard_card(
                self._pointer, tables._pointer, side, card_id.encode('utf-8')))

    def apply_mouth_move(self, ai, side, index, source, destination):
        import observation_tables
        if type(side) is not int or side not in (0, 1):raise ValueError('Side must be FP or SH')
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Die index must be nonnegative int64')
        if any(type(value) is not str or '\0' in value for value in (source, destination)):
            raise ValueError('Movement region must be text without NUL')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_mouth_move(
                self._pointer, tables._pointer, side, index, source.encode('utf-8'), destination.encode('utf-8')))

    def apply_minion_muster(self, ai, side, index, name, region):
        import observation_tables
        if type(side) is not int or side not in (0, 1):raise ValueError('Side must be FP or SH')
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Die index must be nonnegative int64')
        if any(type(value) is not str or '\0' in value for value in (name, region)):raise ValueError('Minion name/region must be text without NUL')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_minion_muster(
                self._pointer, tables._pointer, side, index, name.encode('utf-8'), region.encode('utf-8')))

    def companion_separation_actions(self, ai, index):return self._companion_actions(ai, index, 0)
    def companion_move_actions(self, ai, index):return self._companion_actions(ai, index, 1)
    def companion_chain_actions(self, ai):return self._companion_actions(ai, -1, 2)

    def _companion_actions(self, ai, index, mode):
        import observation_tables
        if type(index) is not int or not -(1 << 63) <= index < (1 << 63):raise ValueError('Index must be int64')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_companion_actions(self._pointer, tables._pointer, index, mode)) as result:
                return result.unpack(ai)

    def apply_companion_separation(self, ai, index, names, destination):
        return self._apply_companion(ai, index, 0, '', destination, names)

    def apply_companion_move(self, ai, index, source, destination, names=None):
        return self._apply_companion(ai, index, 1, source, destination, names)

    def apply_companion_move_option(self, ai, index):
        return self._apply_companion(ai, index, 2, '', '', None)

    def _apply_companion(self, ai, index, mode, source, destination, names):
        import observation_tables
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Index must be nonnegative int64')
        if names is None:names=[]
        if type(names) not in (list, tuple):raise ValueError('Companion names must be a sequence')
        if any(type(value) is not str or '\0' in value for value in (source, destination, *names)):
            raise ValueError('Companion names/regions must be text without NUL')
        encoded=(C.c_char_p*len(names))(*(name.encode('utf-8') for name in names))
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_companion(
                self._pointer, tables._pointer, index, mode, source.encode('utf-8'), destination.encode('utf-8'), encoded, len(names)))

    def draw_card_actions(self, ai, side):
        return self._table_menu(ai, side, 'draw_card_actions')

    def apply_draw_card(self, ai, side, index, deck):
        import observation_tables
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(index) is not int or not 0 <= index < (1 << 63):
            raise ValueError('Draw die index must be nonnegative int64')
        if type(deck) is not str or deck not in ('character', 'strategy'):
            raise ValueError('Draw deck must be character or strategy')
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_draw_card(
                self._pointer, tables._pointer, side, index, deck.encode('utf-8')))

    def menu_handle(self, ai, side, family):
        """Own a resident raw descriptor subset for native execution.

        Families are explicitly partial. This never certifies the complete
        legal action set, and individual rows can require unported effects.
        """
        names = {'political_ring':'political_ring_actions', 'muster':'muster_actions',
                 'saruman_muster':'saruman_muster_actions', 'army_move':'army_move_actions',
                 'companion_political':'companion_political_actions',
                 'eye_allocation':'eye_allocation_actions', 'discard_card':'discard_card_actions',
                 'fellowship_declare':'fellowship_declare_actions', 'guide':'guide_actions', 'full':'full_actions', 'draw_card':'draw_card_actions', 'mouth_move':'mouth_move_actions', 'muster_minion':'minion_muster_actions', 'gandalf_draw':'gandalf_draw_actions', 'dice_event_choice':'dice_event_choice_actions'}
        names.update(nazgul_move='nazgul_move_actions',nazgul_chain='nazgul_chain_actions',will_special='will_special_actions',misc_event_choice='misc_event_choice_actions')
        names.update(token='token_actions',dwarven_ring='dwarven_ring_actions',recruit_event_choice='recruit_event_choice_actions',overstack_choice='overstack_choice_actions')
        names.update(worn_choice='worn_choice_actions',hill_trolls_choice='hill_trolls_choice_actions')
        names.update(target_recruit_choice='target_recruit_choice_actions',shadows_gather_choice='shadows_gather_choice_actions',crebain_choice='crebain_choice_actions',companion_event_choice='companion_event_choice_actions')
        names.update(palantir_draw='palantir_draw_actions')
        names.update(shadow_lengthens_choice='shadow_lengthens_choice_actions',nazgul_hunt_choice='nazgul_hunt_choice_actions',power_kill='power_kill_actions',power_payment='power_payment_actions')
        names.update(half_orcs_choice='half_orcs_choice_actions')
        names.update(faramir_choice='faramir_choice_actions',card_casualty='card_casualty_actions')
        names.update(pits_choice='pits_choice_actions',gwaihir_choice='gwaihir_choice_actions')
        if type(family) is not str or family not in names:
            raise ValueError('Unknown native menu family')
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if family in ('eye_allocation', 'guide', 'gandalf_draw', 'dice_event_choice', 'nazgul_chain', 'misc_event_choice', 'dwarven_ring', 'overstack_choice', 'worn_choice', 'crebain_choice', 'companion_event_choice', 'shadow_lengthens_choice', 'nazgul_hunt_choice', 'gwaihir_choice', 'palantir_draw'):
            with self._lock:
                self._require_open()
                return NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_' + names[family])(self._pointer, side))
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_' + names[family])(
                self._pointer, tables._pointer, side))

    def apply_descriptor_random(self, ai, rng, side, menu, index):
        """Execute a resident descriptor and preserve exact independently owned RNG."""
        import os, combat_context, random_stream, observation_tables
        from contextlib import ExitStack
        combat_context.require_native_bid_context()
        if os.environ.get('MORDOR_STOP_KEEPS_STEP'):raise ValueError('Native actions require standard Mordor stop behavior')
        if type(rng) is not random_stream.NativeRandom:raise ValueError('Action needs an owned NativeRandom stream')
        if type(side) is not int or side not in (0, 1):raise ValueError('Side must be FP or SH')
        if type(index) is not int or not 0 <= index < (1 << 63):raise ValueError('Action row must be nonnegative int64')
        if type(menu) is not NativeSnapshot or menu._lib is not self._lib:raise ValueError('Menu must use the same native library')
        tables = observation_tables.native_tables(ai);output, events = C.c_void_p(), C.c_void_p()
        with ExitStack() as stack:
            for owner in sorted({id(x):x for x in (self, menu)}.values(), key=id):stack.enter_context(owner._lock)
            stack.enter_context(rng._lock)
            self._require_open();menu._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            child = NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_descriptor_random(
                self._pointer, tables._pointer, menu._pointer, rng._pointer, side, index, C.byref(output), C.byref(events)))
        advanced = random_stream.NativeRandom._adopt(rng._lib, output.value)
        try:
            with NativeSnapshot(self._lib, events.value) as payload:annotations = payload.unpack_metadata(ai)
            return child, advanced, annotations
        except BaseException:
            child.close();advanced.close();raise

    def apply_descriptor(self, ai, side, menu, index):
        """Execute one resident descriptor with current-state legality checks."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(index) is not int or not 0 <= index < (1 << 63):
            raise ValueError('Descriptor row index must be a nonnegative int64')
        if type(menu) is not NativeSnapshot or menu._lib is not self._lib:
            raise ValueError('Menu must be an owned snapshot from the same native library')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        first, second = sorted((self, menu), key=id)
        with first._lock, second._lock:
            self._require_open(); menu._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_descriptor(
                self._pointer, tables._pointer, menu._pointer, side, index))

    def project_policy(self, ai, side, menu):
        """Project supplied rows, returning resident selected rows and reasons.

        Global forcedness is meaningful only when the caller supplies the
        complete current legal menu. Raw subsets remain explicitly partial.
        """
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(menu) is not NativeSnapshot or menu._lib is not self._lib:
            raise ValueError('Menu must be an owned snapshot from the same native library')
        output = C.c_void_p();first, second = sorted((self, menu), key=id)
        with first._lock, second._lock:
            self._require_open();menu._require_open()
            selected = NativeSnapshot(self._lib, self._lib.wotr_snapshot_project_policy(
                self._pointer, menu._pointer, side, C.byref(output)))
        try:
            with NativeSnapshot(self._lib, output.value) as meta:
                forced, reasons = meta.unpack(ai)
            return selected, forced, reasons
        except BaseException:
            selected.close();raise

    def _table_menu(self, ai, side, name):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, getattr(self._lib, 'wotr_snapshot_' + name)(
                    self._pointer, tables._pointer, side)) as menu:
                return menu.unpack(ai)

    def apply_saruman_muster_action(self, ai, side, die_index, mode):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Saruman die index must be a nonnegative int64')
        if type(mode) is not str or '\0' in mode:
            raise ValueError('Saruman mode must be a string without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native Saruman actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_saruman_muster_action(
                self._pointer, tables._pointer, side, die_index, mode.encode('utf-8')))

    def muster_actions(self, ai, side):
        """Native raw ordinary recruitment menu, preserving source row order."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_muster_actions(
                    self._pointer, tables._pointer, side)) as menu:
                return menu.unpack(ai)

    def apply_army_move(self, ai, side, die_index, from_region, to_region, *,
                        n_regular=None, n_elite=None, take_leaders=True,
                        n_nazgul=None, n_leaders=None):
        return self._army_move(ai, side, die_index, from_region, to_region,
                              n_regular, n_elite, n_leaders, n_nazgul, take_leaders, False)

    def pass_army_move(self, ai, side):
        return self._army_move(ai, side, 0, '', '', None, None, None, None, True, True)

    def _army_move(self, ai, side, die_index, from_region, to_region,
                   regular, elite, leaders, nazgul, take_leaders, passing):
        """Native ordinary move; unsupported effects refuse without mutation."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Army die index must be a nonnegative int64')
        if type(take_leaders) is not bool:
            raise ValueError('take_leaders must be bool')
        for value in (from_region, to_region):
            if type(value) is not str or '\0' in value:
                raise ValueError('Army regions must be strings without NUL')
        counts = []
        for value in (regular, elite, leaders, nazgul):
            if value is not None and (type(value) is not int or not 0 <= value < (1 << 63)):
                raise ValueError('Army counts must be nonnegative int64 or None')
            counts.append(-1 if value is None else value)
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native army actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_army_move(
                self._pointer, tables._pointer, side, die_index, from_region.encode('utf-8'),
                to_region.encode('utf-8'), *counts, take_leaders, passing))

    def apply_muster_action(self, ai, side, die_index, nation, region, kind):
        """Ordinary elite or paired regular/leader recruitment in supported states."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Muster die index must be a nonnegative int64')
        for value in (nation, region, kind):
            if type(value) is not str or '\0' in value:
                raise ValueError('Muster parameters must be strings without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native muster actions require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_muster_action(
                self._pointer, tables._pointer, side, die_index, nation.encode('utf-8'), region.encode('utf-8'), kind.encode('utf-8')))

    def finish_muster(self, ai, side):
        return self.apply_muster_action(ai, side, 0, '', '', 'pass')

    def apply_political_action(self, ai, side, die_index, nation):
        """Return a native child for a supported complete quiet political action.

        Unsupported contexts raise explicitly; this never calls Python gameplay.
        """
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Political die index must be a nonnegative int64')
        if type(nation) is not str or '\0' in nation:
            raise ValueError('Nation must be an enum member name without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native political actions currently require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_political_action(
                self._pointer, tables._pointer, side, die_index, nation.encode('utf-8')))

    def next_actor(self, just_acted):
        """Exact pending-owner priority and ordinary alternation, read-only."""
        if type(just_acted) is not int or just_acted not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        with self._lock:
            self._require_open()
            result = self._lib.wotr_snapshot_next_actor(self._pointer, just_acted)
            if result == -2:
                raise _error(self._lib)
            return None if result < 0 else result

    def pending_actor(self, ai, fallback=None):
        if fallback is not None and (type(fallback) is not int or fallback not in (0, 1)):
            raise ValueError('Fallback must be FP/SH integer or None')
        with self._lock:
            self._require_open()
            with NativeSnapshot(self._lib, self._lib.wotr_snapshot_pending_actor(
                    self._pointer, -1 if fallback is None else fallback)) as result:
                return result.unpack(ai)

    def next_quiet_actor(self, ai, just_acted):
        """Native next-owner routing for settled ordinary action states."""
        if type(just_acted) is not int or just_acted not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            result = self._lib.wotr_snapshot_next_quiet_actor(self._pointer, tables._pointer, just_acted)
            if result < 0:
                raise _error(self._lib)
            return None if result == 2 else result

    def apply_elven_ring_action(self, ai, side, die_index, new_face):
        """Return a child after complete quiet ring conversion and actor routing."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not 0 <= die_index < (1 << 63):
            raise ValueError('Elven Ring die index must be a nonnegative int64')
        if type(new_face) is not str or '\0' in new_face:
            raise ValueError('Face must be an enum member name without NUL')
        import combat
        if combat.FINAL_HITS_HOOK is not None:
            raise ValueError('Native Elven Ring actions currently require ordinary live/self-play mode')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            return NativeSnapshot(self._lib, self._lib.wotr_snapshot_apply_elven_ring_action(
                self._pointer, tables._pointer, side, die_index, new_face.encode('utf-8')))

    def replace_integer(self, node, expected, replacement):
        if type(node) is not int or not 0 <= node < self.counts[0]:
            raise ValueError('Invalid scalar node index')
        for value in (expected, replacement):
            if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
                raise ValueError('Replacement values must fit int64')
        with self._lock:
            self._require_open()
            if self._lib.wotr_snapshot_replace_integer(self._pointer, node, expected, replacement):
                raise _error(self._lib)

    def _political_rule(self, name, nation, method):
        for value in (nation, method):
            if type(value) is not str or '\0' in value:
                raise ValueError('Native rule parameters must be strings without NUL')
        with self._lock:
            self._require_open()
            result = getattr(self._lib, 'wotr_snapshot_' + name)(self._pointer, nation.encode('utf-8'), method.encode('utf-8'))
            if result < 0:
                raise _error(self._lib)
            if name == 'advance_political':
                self.counts = tuple(int(getattr(self._lib, 'wotr_snapshot_' + key)(self._pointer)) for key in ('node_count', 'edge_count', 'byte_count'))
            return result

    def can_advance_political(self, nation, method='muster_die'):
        """Native game_logic rule; does not check action-die payment or actor."""
        return bool(self._political_rule('can_advance_political', nation, method))

    def advance_political(self, nation, method='muster_die'):
        """Native political rule plus table-card discard, without die payment."""
        return self._political_rule('advance_political', nation, method)

    def encode_prefix_into(self, ai, side, output):
        """Encode regions/politics directly from resident complete native state."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        regions, nations = _encoding_tables(tuple(ai.ALL_REGION_KEYS), tuple(n.name for n in ai.ALL_NATIONS))
        size = len(regions) * (14 + 2 * len(nations)) + 4 * len(nations)
        if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != (size,) or not output.flags.c_contiguous or not output.flags.writeable or not output.flags.aligned:
            raise ValueError('Output must be writable aligned contiguous float32 with exact size')
        with self._lock:
            self._require_open()
            if self._lib.wotr_snapshot_encode_prefix(self._pointer, side, regions, len(regions), nations, len(nations), output.ctypes.data_as(C.POINTER(C.c_float)), output.size):
                raise _error(self._lib)
        return output

    def encode_prefix(self, ai, side):
        size = len(ai.ALL_REGION_KEYS) * (14 + 2 * len(ai.ALL_NATIONS)) + 4 * len(ai.ALL_NATIONS)
        return self.encode_prefix_into(ai, side, np.empty(size, dtype=np.float32))

    def encode_state_into(self, ai, side, output):
        """Full v52 network input, computed entirely from resident native data."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != (4647,) or not output.flags.c_contiguous or not output.flags.writeable or not output.flags.aligned:
            raise ValueError('Output must be writable aligned contiguous float32 with exact size')
        import observation_tables
        tables = observation_tables.native_tables(ai)
        with self._lock:
            self._require_open()
            if self._lib.wotr_snapshot_encode_state(self._pointer, tables._pointer, side, output.ctypes.data_as(C.POINTER(C.c_float)), output.size):
                raise _error(self._lib)
        return output

    def encode_state(self, ai, side):
        return self.encode_state_into(ai, side, np.empty(4647, dtype=np.float32))

    def ring_waiver_active(self, side):
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        with self._lock:
            self._require_open()
            result = self._lib.wotr_snapshot_ring_waiver_active(self._pointer, side)
            if result < 0:
                raise _error(self._lib)
            return bool(result)

    def use_elven_ring(self, side, die_index, new_face):
        """Native physical die conversion; excludes the AI action envelope."""
        if type(side) is not int or side not in (0, 1):
            raise ValueError('Side must be integer 0=FP or 1=SH')
        if type(die_index) is not int or not -(1 << 63) <= die_index < (1 << 63):
            raise ValueError('Die index must fit int64')
        if type(new_face) is not str or '\0' in new_face:
            raise ValueError('Face must be an enum member name without NUL')
        message = C.create_string_buffer(1024)
        with self._lock:
            self._require_open()
            if self._lib.wotr_snapshot_use_elven_ring(self._pointer, side, die_index, new_face.encode('utf-8'), message, len(message)):
                raise _error(self._lib)
            self.counts = tuple(int(getattr(self._lib, 'wotr_snapshot_' + key)(self._pointer)) for key in ('node_count', 'edge_count', 'byte_count'))
            return message.value.decode('utf-8')

    def close(self):
        with self._lock:
            if self._pointer:
                self._lib.wotr_snapshot_destroy(self._pointer)
                self._pointer = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, '_pointer', None):
            self.close()


def pack_snapshot(ai, state):
    if _identity(type(state)) != 'game_state.GameState':
        raise ValueError('A complete snapshot requires a GameState root')
    return NativeSnapshot.from_graph(encode_graph(ai, state))
