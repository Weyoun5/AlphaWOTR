"""Account for the native reveal flip's already performed progress reset."""
import copy


def _signature(state):
    import game_logic as gl
    import game_state as gs
    from board import sanitize
    fellowship = state.fellowship
    if (not getattr(state, '_reveal_place_due', False)
            or fellowship.hidden or fellowship.mordor_entered
            or fellowship.progress <= 0):
        return None
    return (state.turn, sanitize(str(gl.fellowship_region(state))),
            int(fellowship.progress),
            tuple((die.current, die.used) for die in state.dice[gs.Side.FREE_PEOPLES]))


def observe_record(driver, record):
    """Keep an actual own reveal echo, bound to its unfinished placement."""
    import game_state as gs
    from board import sanitize
    if getattr(driver, 'spectator', False) or driver.human_side is not gs.Side.FREE_PEOPLES:
        return
    signature = _signature(driver.es)
    previous = getattr(driver, '_native_reveal_reset', None)
    if previous is not None and record.get('op') == 'undo':
        from . import live_parser
        _, inner = live_parser.normalize(str(record.get('inner', '')))
        if inner == previous['record'].get('norm'):
            driver._native_reveal_reset = None
            return
    if previous is not None and previous['signature'] != signature:
        driver._native_reveal_reset = None
    if (signature is None or record.get('op') != 'flip'
            or record.get('player') != driver.nick
            or str(record.get('obj', '')).lower() != 'the fellowship'
            or record.get('revealed') is not True
            or record.get('prev_track') != signature[2]
            or sanitize(str(record.get('region', ''))) != signature[1]):
        return
    driver._native_reveal_reset = {'signature': signature,
                                   'record': copy.deepcopy(record)}
    driver._native_reveal_retry_signature = None


def _observed_reset(driver, state):
    """A complete native endpoint proves the reset, regardless of who flipped."""
    from board import sanitize
    from . import client_board, client_protocol
    witness = getattr(driver, '_native_reveal_reset', None)
    signature = _signature(state)
    if signature is None:
        return None
    if witness is None or witness['signature'] != signature:
        # The client may reset progress separately or the human may perform
        # the reveal before its AI echo arrives. No particular flip paragraph
        # is required when the complete current board already proves it.
        from .decision_endpoint import read_complete
        try:
            dump, markers, _ = read_complete(driver)
        except (OSError, ValueError):
            return None
        if (sanitize(str(dump.get('248', ''))) in ('track 0', 'fellowship track 0')
                and client_board.fellowship_marker_hidden(markers) is False):
            return signature[2]
        return None
    path = getattr(driver.view, 'dump_path', None)
    if not path:
        return None
    dump = client_board.read_board(path)
    if not dump or sanitize(str(dump.get('248', ''))) not in ('track 0', 'fellowship track 0'):
        return None
    markers = client_protocol.read_marker_states(path)
    if client_board.fellowship_marker_hidden(markers) is not False:
        return None
    return witness['signature'][2]


def pending_window(driver, window):
    """Add only an observed reset to a copy; the human still names the place."""
    import ai
    from . import gesture_match as gm
    pending = ai.describe_pending(driver.es)
    if (not pending or pending['kind'] != 'reveal_place'
            or pending['side'] is not driver.human_side):
        return window
    facts = gm.window_facts(window)
    # A human reveal/reset already expresses this delta. Never count it twice,
    # nor select a destination based only on the engine-produced reveal echo.
    if not facts.fel_moves or facts.fs_progress or facts.fel_flips:
        return window
    progress = _observed_reset(driver, driver.es)
    if progress is None:
        return window
    result = copy.copy(window)
    result.records = list(window.records) + [dict(
        op='move', obj='FSP', norm='', _native_reveal_reset=True,
        **{'from': 'Fellowship Track %d' % progress,
           'to': 'Fellowship Track 0'})]
    return result


def observed_progress_reset(driver, pre, post, actions):
    """The same witnessed reset satisfies the placement commit's sweep."""
    if not any(action.action_type == 'fellowship_reveal_place' for action in actions or ()):
        return None
    progress = _observed_reset(driver, pre)
    driver._native_reveal_reset = None
    if progress is None or post.fellowship.progress != 0 or post.fellowship.hidden:
        return None
    return [('track_marker', {'piece': '248',
                             'frm': 'Game.%d' % (2014 + progress),
                             'to': 'Game.2014'})]


def retry(driver):
    """A later native export can supply the reset missing on first receipt."""
    window = driver.pending_buf
    if window is None:
        return False
    if pending_window(driver, window) is window:
        return False
    key = (id(window), tuple(record.get('norm', '') for record in window.records),
           _signature(driver.es), driver._decision_endpoint_revision())
    if key == getattr(driver, '_native_reveal_retry_signature', None):
        return False
    driver._native_reveal_retry_signature = key
    return driver._try_pending()
