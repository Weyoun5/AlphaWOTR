"""Complete a human's optional Guide choice from the occupied native box."""

GUIDE_BOXES = frozenset(('guide box', 'guide of the fellowship'))
INSTRUCTION = 'place your Guide in the Guide box, then say done'


def _pending(driver):
    if (getattr(driver, 'spectator', False)
            or getattr(driver.es, 'pending_guide_change', None) is None):
        return None
    import ai
    pending = ai.describe_pending(driver.es)
    if (pending and pending['side'] is driver.human_side
            and pending['kind'] == 'guide_change'):
        return pending
    return None


def _key(driver, pending):
    return (driver.es.turn, driver.n_commits_total,
            tuple(a.params.get('companion') for a in pending['actions']))


def request(driver):
    """Retain `done` for this Guide choice only, including a late board export."""
    pending = _pending(driver)
    if pending is None:
        return False
    driver._guide_completion_requested = _key(driver, pending)
    return True


def box_state(driver):
    """Observe an authenticated complete export; never assume an absent box."""
    from board import sanitize
    import log_replay
    from .decision_endpoint import read_complete
    try:
        dump, markers, evidence = read_complete(driver)
    except (OSError, ValueError):
        return 'unavailable', None
    occupants = [str(pid) for pid, zone in dump.items()
                 if sanitize(str(zone)) in GUIDE_BOXES
                 and str(pid) in log_replay.COMPANIONS]
    state = 'empty' if not occupants else 'occupied' if len(occupants) == 1 else 'multiple'
    return state, (dump, markers, evidence)


def prompt(driver):
    state, _ = box_state(driver)
    return INSTRUCTION if state == 'empty' else None


def status(driver):
    state, _ = box_state(driver)
    if state == 'empty':
        return 'waiting on your Fellowship Guide: ' + INSTRUCTION
    if state == 'multiple':
        return 'waiting on: leave just one eligible companion in the Guide box, then say done'
    if state == 'unavailable':
        return 'waiting on: the current board export to confirm the Guide box'
    if driver.phase == 'guide':
        return 'waiting on: say fp ok or done when your Fellowship phase is finished'
    return 'waiting on: confirmation of the Guide shown on the board'


def poll(driver):
    """An occupied legal box keeps/selects its Guide after an action closes.

    At turn start the human still owns the entire Fellowship phase; only an
    explicit completion signal may end it. No guide is inferred from a stale
    export, a half-finished action, or the engine's provisional default.
    """
    pending = _pending(driver)
    if pending is None:
        driver._guide_completion_requested = None
        return False
    requested = getattr(driver, '_guide_completion_requested', None)
    key = _key(driver, pending)
    if requested is not None and requested != key:
        driver._guide_completion_requested = None
        requested = None
    if (driver.phase not in ('actions', 'fellowship_tail') and requested != key
            or driver.deferred is not None or driver.chat_deferred is not None
            or driver.seg.window is not None or driver.echo_pending()
            or driver.human_physical_work_pending() or driver.forced_ai_rows_pending()):
        return False
    state, proof = box_state(driver)
    if state != 'occupied':
        return False
    dump, markers, evidence = proof
    from .choice_recovery import plan
    recovered = plan(driver, dump, markers)
    if (recovered.status != 'match' or len(recovered.match.chain) != 1
            or recovered.match.chain[0].action_type != 'choose_guide'):
        return False
    accepted, _ = driver._commit_recovered_combat(
        recovered, evidence, label='pending:guide_change')
    if accepted:
        driver._guide_completion_requested = None
        driver._buf_stall = None
        driver._last_pending_match = recovered.match
    return accepted
