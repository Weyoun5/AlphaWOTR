"""Freeze immutable observation metadata from the existing reference engine.

These tables are built once at the Python boundary, then read by native code.
Shapely drawing polygons are deliberately excluded; authoritative adjacency,
settlement flags, faction mapping and the existing distance approximations remain.
"""
import threading

import snapshot

_CACHE = {}
_LOCK = threading.Lock()
# The frozen v52 observation contract the native encode_state implements; a
# v53 observation is this vector ++ the encoding_v53 state tail.
V52_PREFIX_CONTRACT = 'v52_observed_combat_round_v3'


class _ImmutableTables:
    """Keep a native owner alive without exposing mutation or close operations.

    Readers retain this object locally across a GIL-releasing native call. No
    shared mutex is needed for immutable native bytes; independent positions
    can encode concurrently. The private owner is destroyed only after its
    last Python reference, including in-flight readers, has gone away.
    """
    def __init__(self, owner):
        self.__owner = owner
        self._pointer = owner._pointer

    def export_graph(self):
        return self.__owner.export_graph()


def table_data(ai):
    import board
    import cards
    import event_effects
    import game_state as gs
    gl = ai.game_logic
    flags = ('_CARD_CONTEXT_ON', '_COMPANION_CONTEXT_ON', '_FELLOWSHIP_DIE_CONTEXT_ON',
             '_HUNT_PROTECTION_CONTEXT_ON', '_REVEAL_ROUTE_CONTEXT_ON', '_ROUTE_EVENT_CONTEXT_ON',
             '_BREAKING_CONTEXT_ON', '_COMBAT_ROUND_CONTEXT_ON')
    if not all(getattr(ai, flag, False) for flag in flags):
        raise ValueError('Native full observation currently requires the complete v52 contract')
    # R2-D5-V53-NATIVE: under generation v53 (USER D5 2026-10-22) the tables
    # are the complete v52 tables - 'observation_contract' stays the v52
    # PREFIX contract the untouched native encode_state checks, because the
    # v53 observation is that v52 vector ++ the encoding_v53 state tail -
    # plus appended v53 keys that only wotr_snapshot_encode_state_v53 reads.
    v53 = bool(getattr(ai, '_V53_CONTEXT_ON', False))
    if v53:
        import encoding_v53
        if ai.OBSERVATION_CONTRACT != encoding_v53.OBSERVATION_CONTRACT:
            raise ValueError('Native observation contract does not match this encoder')
    elif ai.OBSERVATION_CONTRACT != V52_PREFIX_CONTRACT:
        raise ValueError('Native observation contract does not match this encoder')
    vp = gl._vp_settlement_tables()
    # Entry/capture source helpers use this faction spelling map, which differs
    # from the observation region-nation helper for neutral/border regions.
    movement_nations = dict(gondor=gs.Nation.GONDOR, rohan=gs.Nation.ROHAN,
        elves=gs.Nation.ELVES, dwarves=gs.Nation.DWARVES,
        north=gs.Nation.THE_NORTH, isengard=gs.Nation.ISENGARD,
        sauron=gs.Nation.SAURON, southrons=gs.Nation.SOUTHRON_EASTERLINGS,
        southron=gs.Nation.SOUTHRON_EASTERLINGS)
    movement_nations.update({'the north': gs.Nation.THE_NORTH,
        'southrons and easterlings': gs.Nation.SOUTHRON_EASTERLINGS})
    data = {
        'schema': 'wotr_native_observation_tables_v1',
        'observation_contract': V52_PREFIX_CONTRACT,
        'feature_count': 4647,
        'region_keys': list(ai.ALL_REGION_KEYS),
        'nation_names': [nation.name for nation in ai.ALL_NATIONS],
        'card_indices': dict(ai._get_card_idx_map()),
        'cards': cards.get_cards(),
        'special_hunt_tiles': gs.SPECIAL_HUNT_TILES,
        'board_regions': {key: {field: value for field, value in info.items() if field != 'poly'}
                          for key, info in board.regions.items()},
        'board_adj': board.adj,
        'distance_to_mordor': ai._DIST_TO_MORDOR,
        'max_mordor_distance': ai._MAX_MORDOR_DIST,
        'nation_max_figures': ai.NATION_MAX_FIGURES,
        'action_die_face_names': [face.name for face in gs.ActionDieFace],
        'vp_settlements': vp,
        'live_vp_fp_settlements': vp[0],
        'live_vp_sh_settlements': vp[1],
        'fp_vp_targets': ai._FP_VP_TARGETS,
        'sh_vp_targets': ai._SH_VP_TARGETS,
        'dist_to_fp_target': ai._DIST_TO_FP_TARGET,
        'dist_to_sh_target': ai._DIST_TO_SH_TARGET,
        'fp_settle_neighbors_1': ai._FP_SETTLE_NEIGHBORS_1,
        'fp_settle_neighbors_2': ai._FP_SETTLE_NEIGHBORS_2,
        'sh_settle_neighbors_1': ai._SH_SETTLE_NEIGHBORS_1,
        'sh_settle_neighbors_2': ai._SH_SETTLE_NEIGHBORS_2,
        'region_nations': {key: ai._region_nation(key) for key in ai.ALL_REGION_KEYS},
        'shadow_strongholds': gl._get_shadow_strongholds(None),
        'fp_region_keys': {key for key in ai.ALL_REGION_KEYS if gl.is_fp_region(key)},
        'shadow_region_keys': {key for key in ai.ALL_REGION_KEYS if gl.is_shadow_region(key)},
        'event_fp_nation_regions': {key for key in ai.ALL_REGION_KEYS if event_effects._is_fp_nation_region(key)},
        'event_region_metadata': {key: event_effects._get_region_metadata(key) for key in ai.ALL_REGION_KEYS},
        'combat_fp_nation_regions': {key for key in ai.ALL_REGION_KEYS if gl.is_fp_nation_region(key)},
        'board_region_parent_names': {key: board.fid_to_name.get(meta.get('parent_id')) for key,meta in board.regions.items()},
        'movement_region_nations': {key: movement_nations.get(
            board.fid_to_name.get(info.get('parent_id'), '').lower())
            if info.get('parent_id') else None for key, info in board.regions.items()},
        'companion_nations': gl.COMPANION_NATIONS,
        'fp_vp_victory': gl.FP_VP_VICTORY,
        'shadow_vp_victory': gl.SHADOW_VP_VICTORY,
    }
    if v53:   # appended after the unchanged v52 body (graph key order preserved)
        data['observation_contract_v53'] = encoding_v53.OBSERVATION_CONTRACT
        data['feature_count_v53'] = 4647 + encoding_v53.STATE_TAIL_DIM
        data['v53_state_tail_dim'] = encoding_v53.STATE_TAIL_DIM
    return data


def native_tables(ai):
    """One owned, immutable table handle per reference module AND victory thresholds.

    game_logic.FP_VP_VICTORY / SHADOW_VP_VICTORY are module globals the Python
    engine reads live; episodes set SHADOW_VP_VICTORY from config['shadow_vp']
    and engine_play from --shadow-vp.  The native terminal check and the
    vp_limit observation feature read them from this graph, so the cache key
    carries the current values: a handle built at another threshold is never
    returned (Q1-03).  Callers must therefore re-resolve per use, not pin one.
    """
    gl = ai.game_logic
    key = (ai, gl.FP_VP_VICTORY, gl.SHADOW_VP_VICTORY)
    with _LOCK:
        result = _CACHE.get(key)
        if result is None:
            data = table_data(ai)
            if (ai, data['fp_vp_victory'], data['shadow_vp_victory']) != key:
                raise RuntimeError('Victory thresholds changed while native observation tables were built')
            owner = snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai, data))
            try:
                if owner._lib.wotr_snapshot_prepare_observation_tables(owner._pointer):
                    raise snapshot._error(owner._lib)
                result = _ImmutableTables(owner)
            except BaseException:
                owner.close()
                raise
            _CACHE[key] = result
        return result
