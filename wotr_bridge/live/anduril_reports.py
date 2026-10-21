"""Public accounting for automatic Anduril conversions actually executed."""
import re
from .automatic_combat_reports import region_key


def result_context(state):
    pc = getattr(state, 'pending_combat', None) or {}
    context = {key: pc[key] for key in ('attacker_region', 'defender_region', 'attacker_side',
            'defender_side', '_battle_serial', 'rounds_fought') if key in pc}
    context['_bridge_turn'] = getattr(state, 'turn', None)
    # Alias identity survives normal state deepcopy/pickle. Optional numeric
    # battle serials are absent in some legitimate live states.
    context['_bridge_pending_combat'] = getattr(state, 'pending_combat', None)
    return context


def attach_result(result, context, reports, attribute='_bridge_anduril_reports'):
    # This metadata alone never publishes. Only the exact result installed
    # on the committed driver state is eligible at the server boundary.
    if reports and context and not getattr(result, 'round_paused', False):
        setattr(result, attribute, (dict(context), tuple(dict.fromkeys(reports))))


def committed_proxy(server, attribute):
    result = getattr(server.driver.es, '_last_battle_result', None)
    saved = getattr(result, attribute, None)
    if not saved or getattr(result, 'round_paused', False):
        return None
    context, reports = saved
    pc = getattr(server.driver.es, 'pending_combat', None) or {}
    if (context.get('_bridge_turn') != server.driver.es.turn
            or (pc and pc is not context.get('_bridge_pending_combat'))
            or (pc and (pc.get('_battle_serial') != context.get('_battle_serial')
                        or pc.get('rounds_fought', 0) > context.get('rounds_fought', 0) + 1
                        or any(region_key(pc.get(key)) != region_key(context.get(key)) for key in
                               ('attacker_region', 'defender_region'))
                        or any(pc.get(key) != context.get(key) for key in ('attacker_side', 'defender_side'))))):
        return None
    from copy import copy
    # Use the committed result's captured owning battle after normal round
    # completion has removed its cursor/card keys (or ended the battle).
    driver = copy(server.driver)
    driver.es = copy(server.driver.es)
    driver.es.pending_combat = dict(context)
    for report in reports:
        driver.es.pending_combat[report[-1] + '_combat_card'] = report[0]
    proxy = copy(server)
    proxy.driver = driver
    return proxy, reports


def committed(server):
    from types import SimpleNamespace
    saved = committed_proxy(server, '_bridge_anduril_reports')
    if saved is None:
        return []
    proxy, reports = saved
    # Share the exact same round/reveal dedup registry as earlier d6 reports.
    if not hasattr(server, '_automatic_anduril_reports_sent'):
        server._automatic_anduril_reports_sent = set()
    proxy._automatic_anduril_reports_sent = server._automatic_anduril_reports_sent
    return before_roll(proxy, SimpleNamespace(kind='d6', _automatic_anduril_reports=reports))


def retained(combat):
    saved = getattr(combat, '_bridge_anduril_reports', None)
    return saved[1] if saved and saved[0] == getattr(combat, 'siege_round', None) else ()


def observe(combat, before, rolls, threshold, modifier, attacker, converted):
    role = 'attacker' if attacker else 'defender'
    card = getattr(combat, role + '_card', None)
    if (not card or str(card.combat_name).casefold() not in ('anduril', 'andúril')
            or getattr(combat, role + '_card_cancelled') or converted not in (1, 2)):
        return None
    paid = list(getattr(combat, '_round_paid_leadership', ()))
    if paid[:len(before[1])] != before[1] or len(paid) != len(before[1]) + 1:
        return None
    payer = paid[-1]
    figures = [c for c in combat.companions_in_battle if c.name == payer]
    if len(figures) != 1 or not any(n in payer.casefold() for n in ('aragorn', 'strider')):
        return None
    figure = figures[0]
    cost = int(figure.leadership)
    misses = sum(r != 6 and r + modifier < threshold for r in rolls)
    maximum = 2 if getattr(figure, 'crowned', False) or cost >= 2 else 1
    if (cost not in (1, 2) or before[0] - getattr(combat, role + '_leadership') != cost
            or converted != min(maximum, misses)):
        return None
    observation = (str(card.card_id), payer, cost, int(converted),
                   getattr(getattr(combat, role + '_side'), 'name', ''),
                   region_key(combat.attacker_region), region_key(combat.defender_region),
                   'atk' if attacker else 'def')
    reports = list(retained(combat))
    if observation not in reports:
        reports.append(observation)
    combat._bridge_anduril_reports = (combat.siege_round, tuple(reports))
    return observation


def parse_report(text):
    """A resolved public fact, never a combat-effect choice or action."""
    match = re.fullmatch(r"Anduril: forfeit (Aragorn|Strider)'s ([12]) leadership; convert ([12]) normal-roll miss(?:es)? into ([12]) hit(?:s)?", text)
    if not match:
        return None
    name, cost, misses, hits = match.groups()
    if misses != hits or int(hits) > int(cost) or int(cost) != (2 if name == 'Aragorn' else 1):
        return None
    return {'figure': name, 'leadership': int(cost), 'converted_misses': int(misses), 'hits': int(hits)}


def before_roll(server, call):
    if call.kind != 'd6':
        return []
    from .automatic_combat_reports import observed_card_context
    from .chat import anduril_conversion_note
    out = []
    for cid, payer, cost, converted, owner, origin, destination, role in getattr(call, '_automatic_anduril_reports', ()):
        origin,destination = region_key(origin),region_key(destination)
        context = observed_card_context(server, cid, owner, origin, destination, role, allow_human=True)
        if context is None:
            continue
        text = "Anduril: forfeit %s's %d leadership; convert %d normal-roll %s into %d %s" % (
            payer, cost, converted, 'miss' if converted == 1 else 'misses', converted,
            'hit' if converted == 1 else 'hits')
        if parse_report(text) is None:
            continue
        key = (server.driver.es.turn, context['pc'].get('_battle_serial'), origin, destination,
               context['reveal'], cid, payer, cost, converted)
        sent = getattr(server, '_automatic_anduril_reports_sent', None)
        if sent is None:
            sent = server._automatic_anduril_reports_sent = set()
        if key not in sent:
            out.append(server._say('prompt', anduril_conversion_note(text)))
            sent.add(key)
    return out
