"""Panel acknowledgement for an explicit board recovery request."""


def record(server, items, report=()):
    outcome = getattr(server.driver, '_last_resync_adoption', {})
    complete = bool(outcome.get('complete'))
    notes = [str(item[1]) for item in items if item[0] == 'note'] + list(report)
    details = next((row for row in notes if any(word in row.lower()
                    for word in ('unavailable:', 'failed:', 'waits for', 'needs a'))), '')
    summary = ('RESYNC complete: board reconciled; turn order retained.' if complete
               else 'RESYNC incomplete: the bridge could not finish recovery.')
    waiting = outcome.get('waiting_for') if outcome.get('accepted') else None
    if complete and outcome.get('retained_combat'):
        summary = ('RESYNC complete: board reconciled; combat and turn order retained. '
                   'Continue the pending combat step.')
    elif complete and waiting == 'declaration':
        summary = ('RESYNC complete: board reconciled; Fellowship declaration remains open. '
                   'Say fp ok when finished.')
    elif waiting in ('client_roll', 'client_result'):
        summary = ('RESYNC: physical payment retained; waiting for the existing client %s. '
                   'Combat and turn order retained.'
                   % ('roll' if waiting == 'client_roll' else 'result'))
    elif not complete and details:
        # Exceptions can contain hidden card identities. The panel gets a
        # useful category; the existing diagnostic log retains the details.
        if 'cannot pickle' in details:
            summary += ' Recovery could not save its rollback state.'
        elif 'waits for' in details or 'needs a' in details:
            summary += ' Waiting for a complete, stable board and game log.'
        else:
            summary += ' Recovery could not finish; the diagnostic log records the cause.'
    server._panel_recovery_result = {
        'epoch': max(float(getattr(server, '_reset_epoch', 0)),
                     float(getattr(server, '_resync_epoch', 0))),
        'complete': complete, 'summary': summary,
        'waiting_for': waiting,
        'commits': server.driver.n_commits_total,
    }


def current(server):
    result = getattr(server, '_panel_recovery_result', None)
    if result and result['commits'] == server.driver.n_commits_total:
        return {k: v for k, v in result.items() if k != 'commits'}
    return {}
