"""Reconcile an unrecognized completed gesture from the current native board.

No chat warning, guessed die spend, or turn handoff precedes a successful
adoption. A late export retries automatically; unresolved rule choices retain
their normal gesture and randomness handlers.
"""
import copy


def accept(driver, window, reason):
    if getattr(driver, '_lenient_board_wait', None) is None:
        driver._lenient_board_wait = {
            'window': window, 'reason': str(reason),
            'epoch': (id(driver.es), driver.undo_epoch, driver.es.turn),
        }
        driver.findings.append(('lenient_board_queued', str(reason)))
    return poll(driver)


def poll(driver):
    waiting = getattr(driver, '_lenient_board_wait', None)
    if waiting is None:
        return []
    if waiting['epoch'] != (id(driver.es), driver.undo_epoch, driver.es.turn):
        # A real commit, Undo, Skip or Resync supersedes the old observation.
        driver._lenient_board_wait = None
        return []
    import ai
    if (driver.forced_ai_rows_pending()
            or driver.pending_random_call() is not None
            or isinstance(getattr(driver.es, 'pending_combat', None), dict)
            or ai.describe_pending(driver.es) is not None):
        return []  # Never replace a Hunt, battle or other unfinished effect.
    from .decision_endpoint import read_complete
    from .repair_evidence import same_dump
    try:
        dump, _markers, evidence = read_complete(driver)
    except (OSError, ValueError):
        return []
    original = driver.es
    window = waiting['window']
    window_fields = copy.deepcopy(vars(window)) if window is not None else None
    models = list({id(m): m for m in [driver.hands, *driver.hand_models.values()]}.values())
    hand_fields = [(m, copy.deepcopy(vars(m), {id(original): original})) for m in models]
    driver.es = ai.clone_state(original)
    for model in models:
        model.es = driver.es
    driver.view.piece_region = dump
    items = []
    complete = False
    try:
        if window is not None:
            driver._unreveal(window)
            items.extend(driver._lenient_hand_facts(window))
        items.extend(driver._adopt_board_now('lenient', expected_evidence=evidence))
        complete = bool(driver._last_board_adoption.get('complete')) and same_dump(driver, evidence)
        if complete:
            post = driver.es
    finally:
        driver.es = original
        if not complete:
            for model, fields in hand_fields:
                vars(model).clear()
                vars(model).update(fields)
            if window is not None:
                vars(window).clear()
                vars(window).update(window_fields)
    if not complete:
        return []
    norms = [r.get('norm', '') for r in getattr(window, 'records', ())]
    driver._snapshot(norms, 'board_action', post=post)
    driver.es = post
    for model in models:
        model.es = post
    if window is not None and getattr(window, 'die_face', None):
        driver.last_actor = driver.human_side
    driver._lenient_board_wait = None
    for norm in norms:
        if norm in driver.uncommitted:
            driver.uncommitted.remove(norm)
    from .undo_model import RestoreWatch
    driver.restore_watch = RestoreWatch()
    if driver.rand is not None:
        driver.rand.reset_feed()
    driver.findings.append(('lenient_board_reconciled', waiting['reason']))
    return items
