"""Physical completeness checks for explicit recovery, never search legality."""


def validate(driver, dump):
    """A missing figure is not a casualty; reserve boxes retain ownership.

    Native replacement figures may add IDs and change an ID's unit type. The
    base native pieces still have physical locations, including dead/reserve
    boxes. Require those locations without imposing a fixed total piece count.
    """
    import ai
    import game_state as gs
    import log_replay as lr
    from board import sanitize
    from piece_table_authoritative import AUTH_ARMY_PIECES
    from .client_board import unit_type

    dynamic = driver.view.dynamic_types
    required = set(AUTH_ARMY_PIECES) | set(lr.COMPANIONS) | set(lr.MINION_PIECES)
    for pid in getattr(driver, '_initial_dump', {}) or {}:
        if unit_type(pid, dynamic)[1] in ('regular', 'elite', 'leader', 'nazgul', 'companion', 'minion'):
            required.add(str(pid))
    missing = sorted(required - set(dump))
    if missing:
        raise ValueError('incomplete native board: missing figure PID(s) %s; no casualty inferred'
                         % ', '.join(missing[:12]))

    boxes = {
        'fp reinf': gs.Side.FREE_PEOPLES, 'fp reinforcements': gs.Side.FREE_PEOPLES,
        'fp dead': gs.Side.FREE_PEOPLES, 'fp casualties': gs.Side.FREE_PEOPLES,
        'sa reinf': gs.Side.SHADOW, 'sa reinforcements': gs.Side.SHADOW,
        'sa dead': gs.Side.SHADOW, 'sa casualties': gs.Side.SHADOW,
    }
    for pid, zone in dump.items():
        nation, kind = unit_type(pid, dynamic)
        if kind not in ('regular', 'elite', 'leader', 'nazgul'):
            continue
        owner = gs.Side.SHADOW if kind == 'nazgul' else ai._side_of_nation(nation)
        box_owner = boxes.get(sanitize(str(zone)))
        if box_owner is not None and box_owner is not owner:
            raise ValueError('native figure %s belongs to %s but is in the other side\'s box: %s'
                             % (pid, owner.name, zone))
