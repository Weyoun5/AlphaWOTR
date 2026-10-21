"""Observe fixed Nazgul costs on the real trajectory, without adding choices."""
from .automatic_combat_reports import region_key


def snapshot(combat):
    return {role: (getattr(combat, role + '_leadership'),
                   getattr(combat, role + '_dice_mod'),
                   getattr(combat, role + '_reroll_mod'))
            for role in ('attacker', 'defender')}


def retained(combat):
    saved = getattr(combat, '_bridge_fixed_forfeit_reports', None)
    if not saved or saved[0] != getattr(combat, 'siege_round', None):
        return ()
    return saved[1]


def remember(combat, observed):
    reports = list(retained(combat))
    for observation in observed:
        if observation not in reports:
            reports.append(observation)
    combat._bridge_fixed_forfeit_reports = (combat.siege_round, tuple(reports))


def observations(combat, before, effect):
    result = []
    for role, short in (('attacker', 'atk'), ('defender', 'def')):
        card = getattr(combat, role + '_card', None)
        if (card is None or getattr(combat, role + '_card_cancelled', False)
                or str(card.combat_name).casefold() != effect.casefold()):
            continue
        cost = before[role][0] - getattr(combat, role + '_leadership')
        modifier = ('reroll_mod' if effect == 'They are Terrible' else 'dice_mod')
        index = 2 if modifier == 'reroll_mod' else 1
        if (cost != (1 if effect == 'They are Terrible' else 2)
                or getattr(combat, role + '_' + modifier) - before[role][index] != 1):
            continue
        result.append((str(card.card_id), int(cost),
                       getattr(getattr(combat, role + '_side'), 'name', ''),
                       region_key(combat.attacker_region),
                       region_key(combat.defender_region), short))
    return result


def before_roll(server, call):
    if call.kind != 'd6':
        return []
    from .automatic_combat_reports import observed_card_context
    from . import chat
    from combat_forfeit_text import fixed_forfeit_report
    out = []
    for card_id, cost, owner, origin, destination, role in getattr(
            call, '_automatic_forfeit_reports', ()):
        origin,destination = region_key(origin),region_key(destination)
        context = observed_card_context(server, card_id, owner, origin, destination, role, allow_human=True)
        if context is None:
            continue
        card = context['card']
        text = '%s: forfeit %d Nazgul leadership' % (card.combat_name, cost)
        if fixed_forfeit_report(text) is None:
            continue
        key = (server.driver.es.turn, context['pc'].get('_battle_serial'),
               origin, destination, context['reveal'], card_id, cost)
        sent = getattr(server, '_automatic_forfeit_reports_sent', None)
        if sent is None:
            sent = server._automatic_forfeit_reports_sent = set()
        if key not in sent:
            out.append(server._say('prompt', chat.fixed_leadership_forfeit_note(card_id, cost)))
            sent.add(key)
    return out


def committed(server):
    """Publish proved costs even when all rolls were already available."""
    from types import SimpleNamespace
    from .anduril_reports import committed_proxy
    saved = committed_proxy(server, '_bridge_fixed_forfeit_reports')
    if saved is None:
        return []
    proxy,reports = saved
    if not hasattr(server, '_automatic_forfeit_reports_sent'):
        server._automatic_forfeit_reports_sent = set()
    proxy._automatic_forfeit_reports_sent = server._automatic_forfeit_reports_sent
    return before_roll(proxy, SimpleNamespace(kind='d6', _automatic_forfeit_reports=reports))
