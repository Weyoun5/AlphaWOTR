"""Recover visible Fellowship choices through the existing pending matcher."""
import copy
import re

from .combat_recovery import Recovery, _preserve_randomness


_ACTIONS = frozenset(('choose_guide', 'hunt_take_corruption',
                      'hunt_sacrifice_guide', 'hunt_gollum_reveal',
                      'hunt_gollum_no_reveal', 'fellowship_reveal_place',
                      'fellowship_declare'))
_BOXES = frozenset(('fellowship box', 'fellowship of the ring',
                    'guide box', 'guide of the fellowship'))
_GUIDE = frozenset(('guide box', 'guide of the fellowship'))
_DEAD = frozenset(('fp dead', 'fp casualties', 'dead'))


def declaration_in_progress(driver, pending):
    """A legal pawn placement awaiting its physical reset is normal input."""
    if driver.phase != 'declare' or pending.get('kind') != 'declare':
        return False
    records = getattr(driver.pending_buf, 'records', ()) or ()
    if not records or any(r.get('op') not in ('move', 'flip') or
            str(r.get('obj', '')).lower() not in ('the fellowship', 'fsp') for r in records):
        return False
    from board import sanitize
    from .gesture_match import window_facts
    facts = window_facts(driver.pending_buf)
    if (not facts.fel_moves or facts.fs_progress > 0
            or driver.es.fellowship.progress + facts.fs_progress < 0):
        return False
    destination = facts.fel_moves[-1][1]
    return any(a.action_type == 'fellowship_declare'
               and sanitize(str(a.params.get('region', ''))) == destination
               for a in pending['actions'])


def observed_tracks(driver, pre, post, actions, gesture):
    """Credit typed, already measured track drags during explicit recovery."""
    if (not getattr(driver, '_resync_commit_transaction', False) or not actions
            or any(action.action_type not in _ACTIONS for action in actions)
            or gesture is None):
        return []
    import game_logic as gl
    from board import sanitize
    from . import client_board, client_protocol, gesture_match as gm
    path = getattr(driver.view, 'dump_path', None)
    dump = client_board.read_board(path) if path else None
    if not dump:
        return []
    facts = gm.window_facts(gesture)
    covered = []
    for pid, field, token in (('248', 'progress', 'FSP'), ('252', 'corruption', 'Corruption')):
        source, target = getattr(pre.fellowship, field), getattr(post.fellowship, field)
        delta = facts.fs_progress if field == 'progress' else facts.corruption
        if source == target or delta != target - source or _track(dump, pid) != target:
            continue
        if not any(record.get('_combat_recovery') and record.get('op') == 'move'
                   and str(record.get('obj', '')).lower() == token.lower()
                   and str(record.get('to', '')).lower() == 'fellowship track %d' % target
                   for record in gesture.records):
            continue
        if field == 'progress':
            markers = client_protocol.read_marker_states(path)
            reveal = (getattr(pre, '_reveal_place_due', False)
                      and any(a.action_type == 'fellowship_reveal_place' for a in actions))
            declare = (getattr(pre, '_declare_phase', False)
                       and any(a.action_type == 'fellowship_declare' for a in actions))
            if (not (reveal or declare) or target != 0
                    or client_board.fellowship_marker_hidden(markers) is not post.fellowship.hidden
                    or sanitize(str(dump.get('247', ''))) != sanitize(str(gl.fellowship_region(post)))):
                continue
        covered.append(('track_marker', {'piece': pid,
                        'frm': 'Game.%d' % (2014 + source), 'to': 'Game.%d' % (2014 + target)}))
    return covered


def _track(dump, pid):
    match = re.fullmatch(r'Track (\d+)', str(dump.get(pid, '')), re.I)
    return int(match.group(1)) if match else None


def _companions(state, dump):
    import log_replay as lr
    from board import sanitize
    locations = {}
    names = {str(c.name) for c in state.fellowship.companions}
    if state.fellowship.guide is not None:
        names.add(str(state.fellowship.guide.name))
    for pid in lr.COMPANIONS:
        if str(pid) not in dump:
            continue
        matches = set(lr.names_for_pid(str(pid)) or ()) & names
        if len(matches) == 1:
            locations[next(iter(matches))] = sanitize(str(dump[str(pid)]))
    return locations


def _physical_matches(post, initial, dump, markers, guide, *, compare_armies=True):
    """Every supported visible surface must agree with the legal successor."""
    import game_logic as gl
    from board import sanitize, regions
    from .client_board import fellowship_marker_hidden
    from .combat_recovery import _engine_counts, _dump_counts
    try:
        if compare_armies:
            counts, labels, owners = _engine_counts(post)
            if counts != _dump_counts(dump, initial.view.dynamic_types, labels, owners):
                return False
    except (KeyError, ValueError):
        return False
    live = {str(c.name) for c in post.fellowship.companions}
    # Gollum guides the Fellowship after the companion list becomes empty.
    # His visible Guide-box piece is not a casualty or an absent companion.
    if getattr(post.fellowship.guide, 'name', None) == 'Gollum':
        live.add('Gollum')
    for name, zone in _companions(initial.es, dump).items():
        if (name in live and zone not in _BOXES) or (name not in live and zone not in _DEAD):
            return False
    if guide is not None and getattr(post.fellowship.guide, 'name', None) != guide:
        return False
    for pid, field in (('248', 'progress'), ('252', 'corruption')):
        value = _track(dump, pid)
        if value is not None and value != getattr(post.fellowship, field):
            return False
    physical_hidden = fellowship_marker_hidden(markers)
    if physical_hidden is None and post.fellowship.hidden != initial.es.fellowship.hidden:
        return False
    if physical_hidden is not None and physical_hidden != post.fellowship.hidden:
        return False
    pawn = sanitize(str(dump.get('247', '')))
    if pawn in regions and pawn != sanitize(str(gl.fellowship_region(post))):
        return False
    return True


def declaration_unchanged(driver, dump, markers):
    """A measured unchanged board supports declining declaration at fp ok."""
    import game_logic as gl
    from board import sanitize, regions
    from .client_board import fellowship_marker_hidden
    state = driver.es
    guide = getattr(state.fellowship.guide, 'name', None)
    locations = _companions(state, dump)
    names = {str(c.name) for c in state.fellowship.companions}
    if guide is not None:
        names.add(guide)
    return (set(locations) == names
            and [name for name, zone in locations.items() if zone in _GUIDE] == [guide]
            and sanitize(str(dump.get('247', ''))) in regions
            and sanitize(str(dump.get('247', ''))) == sanitize(str(gl.fellowship_region(state)))
            and _track(dump, '248') is not None and _track(dump, '252') is not None
            and fellowship_marker_hidden(markers) is not None
            # An unrelated army discrepancy is not a Fellowship declaration.
            # Its normal census/recovery lane remains responsible for it.
            and _physical_matches(state, driver, dump, markers, guide, compare_armies=False))


def unchanged_declaration_correction(driver, dump, markers, *, allow_empty=False):
    """Prove that a declaration buffer only repairs the current physical pose.

    This is a pure predicate. The caller must validate a complete, current,
    stable board/feed export and explicitly request completion or recovery
    before retiring the buffer. It authorizes no engine action or phase close.
    Recovery may also validate an empty buffer with ``allow_empty=True``.
    """
    import ai
    pending = ai.describe_pending(driver.es)
    records = getattr(driver.pending_buf, 'records', ()) or ()
    if (driver.phase != 'declare' or not pending
            or pending['kind'] != 'declare' or pending['side'] is not driver.human_side
            or not any(action.action_type == 'pass' for action in pending['actions'])
            or (not records and not allow_empty) or any(record.get('op') not in ('move', 'flip')
                or str(record.get('obj', '')).lower() not in ('the fellowship', 'fsp')
                for record in records)):
        return False
    # Declaration can precede a card continuation in the legal-menu cascade;
    # matching its pose must not dispose of that continuation's evidence.
    if (any(getattr(driver.es, name, None) is not None for name in (
                'pending_card_option', 'pending_card_followup', 'pending_discard',
                'pending_army_move_die', 'pending_muster_die', '_gandalf_draw_pending'))
            or any(getattr(driver.es, name, False) for name in (
                '_pending_hunt_damage', '_reveal_place_due', '_hunt_reveal_pending',
                '_fellowship_tail_phase', '_eye_allocation_phase'))
            or getattr(driver, '_turn_start_draws_open', False)
            or getattr(driver, 'ai_step_open', False)
            or driver.deferred is not None or driver.chat_deferred is not None
            or driver.echo_pending()):
        return False
    if driver.rand is not None and (driver.rand.pending() is not None
            or any(driver.rand.unconsumed().values())):
        return False
    if driver.rand is not None:
        # A failed gesture probe opens an empty retry step. That alone is not
        # an unresolved decision, but recorded picks or consumed results are.
        positions = driver.rand._feed_pos
        if (driver.rand._journal
                or any(positions[name] for name in ('d6', 'tiles', 'companion', 'card_pick'))
                or any(positions['cards'].values())):
            return False
    return declaration_unchanged(driver, dump, markers)


def plan(driver, dump, markers):
    """Return only a unique legal choice supported by visible physical pieces."""
    import ai
    import game_logic as gl
    from board import sanitize, regions
    from . import gesture_match as gm
    from .client_board import fellowship_marker_hidden
    state = driver.es
    pending = ai.describe_pending(state)
    if pending and pending['side'] is driver.human_side and pending['kind'] == 'discard':
        from .discard_recovery import plan as recover_discard
        return recover_discard(driver, dump, pending)
    if (not pending or pending['side'] is not driver.human_side
            or pending['kind'] not in ('declare', 'guide_change', 'hunt', 'reveal_place')):
        from .decision_endpoint import recover_visible_choice
        direct = recover_visible_choice(driver, pending)
        if direct is not None:
            old = driver.pending_buf
            window = copy.copy(old) if old is not None else driver._new_phase_window()
            window.records = list(getattr(old, 'records', ()) or ())
            return Recovery('match', 'completed board proves the retained choice',
                            window=window, match=direct)
        return Recovery('hold', 'the current board does not yet identify the pending choice')
    if driver.rand is not None and driver.rand.unconsumed().get('companion', 0):
        return Recovery('hold', 'a retained random companion result must resolve through its original choice')
    if pending['kind'] == 'declare':
        return _declaration(driver, dump, markers, pending)
    old = driver.pending_buf
    window = copy.copy(old) if old is not None else driver._new_phase_window()
    window.records = list(getattr(old, 'records', ()) or ())
    facts = gm.window_facts(window)
    added = []
    locations = _companions(state, dump)
    expected_names = {str(companion.name) for companion in state.fellowship.companions}
    if state.fellowship.guide is not None:
        expected_names.add(str(state.fellowship.guide.name))
    if set(locations) != expected_names:
        return Recovery('hold', 'the board does not locate every retained Fellowship companion')
    guides = [name for name, zone in locations.items() if zone in _GUIDE]
    if len(guides) != 1:
        return Recovery('hold', 'the complete board must identify exactly one Fellowship guide')
    guide = guides[0]
    current_guide = getattr(state.fellowship.guide, 'name', None)
    if (pending['kind'] == 'guide_change' or guide != current_guide) and facts.guide_set is None:
        added.append(dict(op='move', obj=guide, **{
            'from': 'Fellowship Box', 'to': 'Guide Of The Fellowship'}))
    for name, zone in locations.items():
        if zone in _DEAD and not facts.comps_gone.get(name):
            if name != current_guide:
                return Recovery('hold', 'a non-guide casualty requires its recorded random choice')
            added.append(dict(op='kill', pieces=[name], zone='Guide Of The Fellowship'))
        elif zone not in _BOXES and zone not in _DEAD:
            return Recovery('hold', 'a companion placement needs its original legal gesture')
    for pid, field, token in (('252', 'corruption', 'Corruption'), ('248', 'progress', 'FSP')):
        target = _track(dump, pid)
        if target is None:
            return Recovery('hold', 'the complete board must expose the Fellowship tracks')
        expressed = facts.corruption if field == 'corruption' else facts.fs_progress
        source = getattr(state.fellowship, field) + expressed
        if source != target:
            added.append(dict(op='move', obj=token, **{
                'from': 'Fellowship Track %d' % source,
                'to': 'Fellowship Track %d' % target}))
    hidden = fellowship_marker_hidden(markers)
    pawn = sanitize(str(dump.get('247', '')))
    current = sanitize(str(gl.fellowship_region(state)))
    if pending['kind'] == 'reveal_place' and hidden is not False:
        return Recovery('hold', 'Hunt reveal placement needs the native revealed Fellowship face')
    if hidden is not None and hidden != state.fellowship.hidden and not facts.fel_flips:
        if hidden:
            return Recovery('hold', 'an unfinished Hunt cannot authorize hiding the Fellowship')
        added.append(dict(op='flip', obj='The Fellowship', zone=pawn,
                          region=pawn, revealed=True, prev_track=0))
    if pawn in regions and pawn != current and not facts.fel_moves:
        added.append(dict(op='move', obj='The Fellowship', **{'from': current, 'to': pawn}))
    if not added:
        return Recovery('hold', 'the stable board supplies no missing Fellowship choice')
    for record in added:
        record.update(norm='', _combat_recovery=True)
    window.records.extend(added)
    menu = [action for action in pending['actions'] if action.action_type in _ACTIONS]
    reveal_routes = [action for action in menu
                     if action.action_type == 'fellowship_reveal_place'
                     and sanitize(str(action.params.get('region', ''))) == pawn]
    if len(reveal_routes) > 1:
        return Recovery('hold', 'the pawn destination does not distinguish the legal reveal routes')
    if not menu:
        return Recovery('hold', 'the pending legal menu has no board-proved Fellowship choice')
    with _preserve_randomness(driver.rctl):
        result = gm.match_window(state, pending['side'], window, menu=menu,
                                 pin=driver._attempt_pin(), rng=driver.rctl,
                                 advance_tolerance=True, fs_prefilter=True)
    if result.status != 'match' or any(action.action_type not in _ACTIONS for action in result.chain):
        return Recovery('hold', result.note or 'the board does not prove a unique legal Fellowship choice')
    if not _physical_matches(result.post, driver, dump, markers, guide):
        return Recovery('hold', 'the legal Fellowship choice leaves visible board differences unresolved')
    return Recovery('match', 'recovered missing physical %s choice' % pending['kind'],
                    window=window, match=result, added=len(added))


def _declaration(driver, dump, markers, pending):
    """Match a finished declaration by its public result, never by flip count."""
    import game_logic as gl
    from board import sanitize, regions
    from . import gesture_match as gm
    from .client_board import fellowship_marker_hidden
    state = driver.es
    pawn = sanitize(str(dump.get('247', '')))
    current = sanitize(str(gl.fellowship_region(state)))
    if (driver.phase != 'declare' or not state.fellowship.hidden
            or fellowship_marker_hidden(markers) is not True
            or pawn not in regions or _track(dump, '248') != 0):
        return Recovery('hold', 'a declaration needs a hidden Fellowship at a legal destination with progress zero')
    records = list(getattr(driver.pending_buf, 'records', ()) or ())
    # Only declaration choreography can be replaced by its final board result.
    # Casualties, cards, companion moves and random results retain their lane.
    if any(record.get('op') not in ('move', 'flip') or
           str(record.get('obj', '')).lower() not in ('the fellowship', 'fsp')
           for record in records):
        return Recovery('hold', 'the pending window also contains a different unfinished gesture')
    if pawn == current and state.fellowship.progress == 0 and not records:
        return Recovery('hold', 'the unchanged board does not prove an in-place declaration')
    locations = _companions(state, dump)
    names = {str(c.name) for c in state.fellowship.companions}
    guide = getattr(state.fellowship.guide, 'name', None)
    if guide is not None:
        names.add(guide)
    if (set(locations) != names or
            [name for name, zone in locations.items() if zone in _GUIDE] != [guide]
            or _track(dump, '252') is None):
        return Recovery('hold', 'the complete board must locate the retained Fellowship and its Guide and tracks')
    window = driver._new_phase_window()
    window.records = [dict(op='move', obj='The Fellowship',
                          **{'from': current, 'to': pawn}),
                      dict(op='move', obj='FSP', **{
                          'from': 'Fellowship Track %d' % state.fellowship.progress,
                          'to': 'Fellowship Track 0'})]
    for record in window.records:
        record.update(norm='', _combat_recovery=True)
    menu = [a for a in pending['actions'] if a.action_type == 'fellowship_declare'
            and sanitize(str(a.params.get('region', ''))) == pawn]
    if not menu:
        return Recovery('hold', 'the completed board is not a legal Fellowship declaration')
    with _preserve_randomness(driver.rctl):
        result = gm.match_window(state, pending['side'], window, menu=menu,
                                 pin=driver._attempt_pin(), rng=driver.rctl,
                                 advance_tolerance=True, fs_prefilter=True)
    if (result.status != 'match' or
            not _physical_matches(result.post, driver, dump, markers, guide)):
        return Recovery('hold', 'the legal declaration does not yet match the complete board')
    return Recovery('match', 'recovered completed Fellowship declaration',
                    window=window, match=result, added=len(window.records))
