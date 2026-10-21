"""Describe an existing matcher-proved unfinished payment, without accepting it."""


def key(driver):
    from .casualty_gestures import live_context
    context = live_context(driver.es)
    if context is None or context['side'] is not driver.human_side:
        return None
    return (driver._pending_rng_signature(),
            tuple((name, str(context.get(name))) for name in
                  ('kind', 'side', 'region', 'stage', 'hits', 'card_id')))


def remember(driver, result):
    if (getattr(result, 'status', None) == 'hold'
            and getattr(result, 'waiting_casualty_completion', False)
            and not getattr(result, 'starved', False)):
        result._partial_casualty_status_key = key(driver)


def waiting(driver):
    from .casualty_gestures import early_context
    early = early_context(driver)
    if early is not None and getattr(driver.pending_buf, '_early_casualty_key', None) is not None:
        return early
    result = getattr(driver, '_last_pending_match', None)
    proof = getattr(result, '_partial_casualty_status_key', None)
    if (proof is None or getattr(result, 'status', None) != 'hold'
            or not getattr(result, 'waiting_casualty_completion', False)
            or getattr(result, 'starved', False)
            or driver.pending_buf is None or not driver.pending_buf.records
            or proof != key(driver)):
        return None
    from .casualty_gestures import live_context
    return live_context(driver.es)


def text(driver):
    context = waiting(driver)
    if context is None:
        return None
    return ('casualty payment in progress in %s: finish the %d-hit allocation '
            'by removing or reducing your pieces. An Elite removed entirely '
            'takes 2 hits; an Elite reduced with its same-nation Regular '
            'replacement takes 1 hit. Finish any Regular placements required '
            'by the available supply; the completed board records the allocation.'
            % (context['region'], context['hits']))
