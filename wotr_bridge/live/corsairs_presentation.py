"""Keep a declared Corsairs landing visible in Umbar until defense is chosen.

The rules state has already landed the army (Almanac p.34). This is only a
client presentation debt, tied to exact physical pieces and the same battle.
"""
from board import sanitize


def defer(driver, action, pre, post, emission):
    pc = getattr(post, 'pending_combat', None) or {}
    cid = str((action.params or {}).get('card_id', ''))
    if (emission.hold or cid != 'SHADOW_Strategy_Corsairs_of_Umbar'
            or pc.get('phase') != 'defender_siege_choice'
            or pc.get('defender_side') is not driver.human_side
            or pc.get('attacker_side') is not driver.ai_side):
        return
    import log_replay
    from . import emission as em, chat
    target = sanitize(str(pc['defender_region']))
    deferred, positions = [], {}
    for op, fields in emission.ops:
        source = log_replay.decode_location(str(fields.get('frm', '')))
        destination = log_replay.decode_location(str(fields.get('to', '')))
        if sanitize(str(source or '')) != 'umbar' or sanitize(str(destination or '')) != target:
            continue
        ids = list(fields.get('pieces') or ()) or [fields.get('piece')]
        if not ids or any(pid is None for pid in ids):
            continue
        deferred.append((op, fields))
        positions.update({str(pid): (str(source), str(destination)) for pid in ids})
    if not deferred:
        return
    lines = [em.render(op, fields, local=False) for op, fields in deferred]
    if any(line not in emission.lines for line in lines):
        raise ValueError('Corsairs movement is missing its native rendering')
    for line in lines:
        emission.lines.remove(line)
    # The renderer provisionally tracked these moves. Keep its view honest
    # until the delayed operations are actually published.
    for pid, (source, _destination) in positions.items():
        driver.view.piece_region[pid] = source
    driver._corsairs_arrival = dict(target=target, positions=positions,
                                   ops=deferred, lines=lines)
    # Preserve full ops in the effect's expressed ledger: the ordinary forced
    # row sweep must not publish the very movement we deliberately deferred.
    from collections import Counter
    import emitter
    counts = Counter(emitter._unit_type(pid, driver.view, side_hint='shadow') for pid in positions)
    names = []
    for (nation, kind), count in sorted(counts.items(), key=lambda row: str(row[0])):
        if kind in ('minion', 'companion'):
            names.append(str(nation))
        else:
            names.append('%d %s %s%s' % (count, getattr(nation, 'value', nation) or '',
                                        kind, '' if count == 1 else 's'))
    emission.chat.append(('prompt', chat._register('Corsairs attack %s from Umbar with %s. '
        'Choose stand or siege; I will move the arriving pieces after your defense is settled.'
        % (target, ', '.join(names)))))


def projected(driver, dump):
    """Account only for these declared moves; retain all other discrepancies."""
    debt = getattr(driver, '_corsairs_arrival', None)
    if not debt or not dump:
        return dump
    result = dict(dump)
    for pid, (source, target) in debt['positions'].items():
        if sanitize(str(result.get(pid, ''))) == sanitize(source):
            result[pid] = target
    return result


def poll(server):
    driver = server.driver
    debt = getattr(driver, '_corsairs_arrival', None)
    if not debt:
        return []
    pc = getattr(driver.es, 'pending_combat', None) or {}
    if pc.get('phase') == 'defender_siege_choice':
        return []
    # A spoken siege selection is not complete until the human's defenders
    # are in their stronghold box. Never overlay them during that transfer.
    if driver.human_physical_work_pending():
        return []
    from .turn_bookkeeping import _fresh_board
    physical = _fresh_board(server)
    if physical is None:
        return []
    from . import emission as em
    ops = []
    for op, fields in debt['ops']:
        ids = list(fields.get('pieces') or ()) or [fields.get('piece')]
        missing = []
        for pid in ids:
            source, target = debt['positions'][str(pid)]
            zone = sanitize(str(physical.get(str(pid), '')))
            if zone == sanitize(source):
                missing.append(pid)
            elif zone != sanitize(target):
                return []  # Retain evidence; a different piece cannot pay it.
        if missing:
            fields = dict(fields)
            if 'pieces' in fields: fields['pieces'] = missing
            else: fields['piece'] = missing[0]
            ops.append((op, fields))
    rendered = em._ops_to_emission(driver.view, driver.nick, 'corsairs_arrival', ops)
    lines = server._emission_out(rendered)
    driver._corsairs_arrival = None
    driver.findings.append(('corsairs_arrival_presented', debt['target']))
    return lines
