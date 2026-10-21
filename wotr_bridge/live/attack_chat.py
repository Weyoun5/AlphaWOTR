"""An attack declaration may include the attacker's first-round 'nc'."""


def declaration(driver, body):
    from . import chat
    text = chat.normalize_body(body)
    if not text.endswith(' nc'):
        return None
    text = text[:-3].strip()
    parsed = chat.classify_command(text)
    if not parsed or parsed[0] not in ('region', 'attack'):
        return None
    target = parsed[1][0] if parsed[0] == 'attack' else parsed[1]
    region, _ = chat.resolve_region(target, driver._region_keys())
    return text if region else None


def remember(driver):
    combat = driver.es.pending_combat or {}
    if combat.get('attacker_side') is driver.human_side:
        driver._declared_attack_nc = (
            driver.es.turn, driver._cc_battle_serial,
            combat.get('attacker_region'), combat.get('defender_region'),
            int(combat.get('rounds_fought', 0)))


def poll(driver):
    intent = getattr(driver, '_declared_attack_nc', None)
    if intent is None:
        return []
    combat = driver.es.pending_combat or {}
    current = (driver.es.turn, driver._cc_battle_serial,
               combat.get('attacker_region'), combat.get('defender_region'),
               int(combat.get('rounds_fought', 0)))
    if current != intent or combat.get('atk_card_committed'):
        driver._declared_attack_nc = None
        return []
    if (driver.deferred is not None or driver.chat_deferred is not None
            or driver.forced_ai_rows_pending() or driver.human_physical_work_pending()):
        return []
    import ai
    pending = ai.describe_pending(driver.es)
    if not pending or pending['side'] is not driver.human_side:
        return []  # the defender may still be deciding field battle or siege
    action = next((a for a in pending['actions'] if a.action_type == 'combat_card_none'), None)
    if action is None:
        return []
    driver._declared_attack_nc = None
    return driver._commit_human(action, via='chat')
