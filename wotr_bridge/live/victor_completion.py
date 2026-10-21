"""Finish a human victory advance from the board when the action die is killed."""


def active(driver, pending=None):
    if getattr(driver, 'spectator', False):
        return False
    if pending is None:
        import ai
        pending = ai.describe_pending(driver.es)
    return bool(pending and pending['kind'] == 'combat:victor_advance'
                and pending['side'] is driver.human_side)


def observe_close(driver, event, payload):
    if not active(driver):
        return
    if event == 'closed' and getattr(payload, 'closed', None) == 'die_killed':
        if payload.die_side != driver.human_side.name:
            return
        record = payload.records[-1] if payload.records else None
    elif event == 'phase' and payload.get('op') in ('kill', 'move'):
        record = payload
    else:
        return
    from .action_die_zones import temporary, zone, casualties
    label = 'free action die' if driver.human_side.name == 'FREE_PEOPLES' else 'shadow action die'
    if not record:
        return
    if record.get('op') == 'move':
        disposed = (temporary(record.get('from', ''))
                    and zone(record.get('to', '')) == casualties(driver.human_side.name)
                    and label in str(record.get('obj', '')).lower())
    else:
        disposed = (temporary(record.get('zone', ''))
                    and any(label in str(piece).lower() for piece in record.get('pieces', ())))
    if not disposed:
        return
    if driver.pending_buf is None:
        driver.pending_buf = driver._new_phase_window()
    driver.pending_buf._victor_close_record = record
    return event == 'phase'


def handle_pending(driver, pending):
    """No movement means stay only at completion, never on an idle poll.

    Candidate transitions come from the normal legal engine menu. Comparing
    complete native figure counts also handles a relieved garrison returning
    from its stronghold box without calling that an advancing army.
    """
    if not active(driver, pending):
        return None
    buf = driver.pending_buf
    close = getattr(buf, '_victor_close_record', None)
    if close is None:
        return False
    from .decision_endpoint import read_complete, filter as filter_endpoint
    from .combat_recovery import _engine_counts, _dump_counts, _preserve_randomness
    from . import gesture_match
    import ai
    try:
        dump, _markers, _signature = read_complete(driver)
        candidates = []
        with _preserve_randomness(driver.rctl):
            for action in pending['actions']:
                if action.action_type not in ('victor_advance', 'victor_stay'):
                    continue
                post = ai.apply_action(driver.es, action)
                counts, labels, owners = _engine_counts(post)
                if counts != _dump_counts(dump, driver.view.dynamic_types, labels, owners):
                    continue
                candidates.append(dict(chain=[action], post=post, execution=()))
        resolved = filter_endpoint(driver, buf, candidates)
    except (OSError, ValueError, KeyError) as error:
        driver._last_pending_match = gesture_match.MatchResult('hold', note=str(error))
        return False
    if resolved.status != 'matched':
        driver._last_pending_match = gesture_match.MatchResult('hold', note=resolved.reason)
        return False
    chosen = resolved.candidates[0]
    norms = [record.get('norm', '') for record in buf.records]
    if close.get('norm') not in norms:
        norms.append(close.get('norm', ''))
    if not driver._commit_chain(chosen['chain'], norms, 'pending:combat:victor_advance', gesture=buf):
        return False
    driver.pending_stats['matched'] += 1
    driver.pending_buf = None
    driver._buf_stall = None
    driver.findings.append(('victor_endpoint_completed', chosen['chain'][0].action_type))
    driver._out_queue.extend(driver._retry_holds('victory advance completed on board'))
    return True
