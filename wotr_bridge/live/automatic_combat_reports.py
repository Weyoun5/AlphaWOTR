"""Public reports of forced results observed on a real combat advance."""


def region_key(value):
    """Use the engine's board identity for display and cursor spellings."""
    from board import sanitize
    return sanitize(str(value or ''))


def words_observation(combat, before, choices):
    target = getattr(combat, '_wop_cancelled', None)
    companions = list(getattr(combat, 'companions_in_battle', ()) or ())
    if target is None or target is before or len(companions) != 1 or companions[0] is not target:
        return None
    if not any(target is leader for role in ('attacker', 'defender')
               for leader in getattr(getattr(combat, role + '_army'), 'leaders', ())):
        return None
    for role, short in (('attacker', 'atk'), ('defender', 'def')):
        card = getattr(combat, role + '_card', None)
        if (card is None or getattr(combat, role + '_card_cancelled', False)
                or str(card.combat_name).casefold() != 'words of power'
                or (choices or {}).get(short + '_effect_target') is not None):
            continue
        return (str(card.card_id), str(target.name),
                getattr(getattr(combat, role + '_side'), 'name', ''),
                region_key(combat.attacker_region), region_key(combat.defender_region), short)
    return None


def selected_step_card(server):
    """The actual journaled AI card choice, never a merely available card."""
    step = getattr(server, '_open_step', None)
    journal = getattr(server.rand, '_journal', ())
    if not step or step[0] != 'ai_decide' or not journal or not server.rand.step_open:
        return None
    pending = step[1]
    owner = getattr(pending.get('side'), 'name', '')
    if owner != getattr(server.ai_side, 'name', ''):
        return None
    pc = getattr(server.driver.es, 'pending_combat', None) or {}
    for role, phase in (('atk','attacker_combat_card'), ('def','defender_combat_card')):
        if pending.get('kind') != 'combat:'+phase or pc.get('phase') != phase:
            continue
        from .randomness import act_key
        matches = [action for action in pending.get('actions', ())
                   if action.action_type == 'combat_card_play' and act_key(action) == journal[0]]
        if len(matches) != 1:
            return None
        return (str(matches[0].params.get('card_id')), owner,
                region_key(pc.get('attacker_region')), region_key(pc.get('defender_region')), role)
    return None


def committed_round_cards(pc):
    """Both actual choices retained by the current paused engine round.

    A None in this validated tuple is a committed no-card choice. An absent
    role-card key in the outer pending dictionary does not mean that choice
    was never made: the engine moves both cards into its resumable cursor.
    """
    cards = pc.get('_round_cards')
    resume = pc.get('_round_resume')
    effect = pc.get('effect_choice')
    if (pc.get('phase') != 'combat_effect_choice' or not isinstance(cards,(tuple,list))
            or len(cards) != 2 or any(card is not None and not isinstance(card,str) for card in cards)
            or not isinstance(resume,dict) or resume.get('schema') != 'combat_battle_resume_v1'
            or resume.get('done') is not False or not isinstance(effect,dict)):
        return None
    cursor = resume.get('cursor')
    battle = getattr(cursor,'combat',None)
    pending = getattr(cursor,'pending',None)
    if (getattr(cursor,'schema',None) != 'combat_round_cursor_v1' or battle is None
            or not isinstance(pending,dict) or effect.get('stage') != 'round'
            or effect.get('round_stage') != getattr(cursor,'stage',None)
            or any(effect.get(key) != pending.get(key) for key in ('effect','owner','options'))):
        return None
    args = resume.get('args')
    if not isinstance(args,(tuple,list)) or len(args) != 7 or tuple(args[2:4]) != tuple(cards):
        return None
    for index,role in enumerate(('attacker','defender')):
        region = region_key(pc.get(role+'_region'))
        if (region_key(getattr(battle,role+'_region',None)) != region
                or region_key(args[index]) != region
                or getattr(battle,role+'_side',None) != pc.get(role+'_side')):
            return None
        card = getattr(battle,role+'_card',None)
        if (getattr(card,'card_id',None) if card is not None else None) != cards[index]:
            return None
    return tuple(cards)


def table_balrog_commit(server, card_id, owner, origin, destination, role):
    """Exact table-origin payment from the selected or committed real action.

    Balrog is the only engine combat card with this origin. The live round
    must match the journal PRE-state and that state must actually own the
    persistent Balrog. A card caption or current table location is insufficient.
    """
    if card_id != 'SHADOW_Character_Balrog_of_Moria' or owner != 'SHADOW':
        return False
    driver = server.driver
    pc = getattr(driver.es, 'pending_combat', None) or {}
    def paid_from_table(state, action):
        params = getattr(action, 'params', {}) or {}
        before = getattr(state, 'pending_combat', None) or {}
        from game_state import PersistentCardType
        persistent = getattr(state, 'persistent_cards', None)
        table = persistent.get_by_type(PersistentCardType.BALROG_OF_MORIA) if persistent else None
        return (getattr(action,'action_type',None) == 'combat_card_play'
                and params.get('card_id') == card_id and params.get('from_table') is True
                and table is not None and table.card_id == card_id
                and getattr(table.side,'name','') == owner
                and before.get('phase') == ('attacker_combat_card' if role == 'atk' else 'defender_combat_card')
                and before.get('_battle_serial') == pc.get('_battle_serial')
                and before.get('rounds_fought',0) == pc.get('rounds_fought',0)
                and getattr(state,'turn',None) == getattr(driver.es,'turn',None)
                and region_key(before.get('attacker_region')) == region_key(origin)
                and region_key(before.get('defender_region')) == region_key(destination)
                and before.get('attacker_side') == pc.get('attacker_side')
                and before.get('defender_side') == pc.get('defender_side'))
    if (getattr(server,'_open_step',None) is not None
            and selected_step_card(server) == (card_id,owner,origin,destination,role)):
        from .randomness import act_key
        pending = server._open_step[1]
        selected = [a for a in pending.get('actions',()) if act_key(a) == server.rand._journal[0]]
        if len(selected) == 1 and paid_from_table(driver.es, selected[0]):
            return True
    for entry in reversed(getattr(driver,'commits',())):
        if (len(entry) >= 5 and isinstance(entry[3],dict)
                and isinstance(getattr(driver,'_cc_battle_serial',None),int)
                and entry[3].get('_cc_battle_serial') == driver._cc_battle_serial
                and paid_from_table(entry[0],entry[4])):
            return True
    return False


def observed_card_context(server, card_id, owner, origin, destination, role, require_reveal=True,
                          allow_human=False):
    """Bind the real effect to this battle's committed or selected card."""
    driver = server.driver
    pc = getattr(driver.es, 'pending_combat', None) or {}
    origin,destination = region_key(origin),region_key(destination)
    owners = {getattr(server.ai_side, 'name', '')}
    if allow_human:
        owners.add(getattr(server.human_side, 'name', ''))
    if (role not in ('atk','def') or owner not in owners
            or region_key(pc.get('attacker_region')) != origin
            or region_key(pc.get('defender_region')) != destination
            or getattr(pc.get('attacker_side' if role == 'atk' else 'defender_side'), 'name', '') != owner):
        return None
    from cards import get_card
    committed = pc.get(role + '_combat_card')
    round_cards = committed_round_cards(pc)
    if committed is None and round_cards is not None:
        committed = round_cards[0 if role == 'atk' else 1]
    if committed != card_id:
        if committed is not None or selected_step_card(server) != (card_id,owner,origin,destination,role):
            return None
    card = get_card(card_id) if card_id else None
    if card is None:
        return None
    # A native reveal is public and precedes this response. A title merely
    # in the hand, a chooser request, or a previous round's cleanup is not.
    hand,slot = ('Shadow Hand','Current SA Card') if owner == 'SHADOW' else ('Free Peoples Hand','Current FP Card')
    from .card_identity import name_to_cid, _norm
    from .live_parser import parse
    deck = ('SA' if owner == 'SHADOW' else 'FP') + ' ' + str(card.deck).title()
    def same_card(title):
        # Use the bridge's existing native-title resolver, including its
        # known client aliases. The combat face must also be the actual face
        # of this card, not just an event-name prefix that resolves to its ID.
        return (isinstance(title,str) and '/' in title
                and _norm(title.rsplit('/',1)[1]) == _norm(card.combat_name)
                and (name_to_cid(title.rsplit('/',1)[0],deck)
                     or name_to_cid(title,deck)) == card_id)
    ai_nick = str(getattr(server,'nick',getattr(driver,'nick','')))
    reveal_actors = {'auto'}
    if owner == getattr(server.ai_side, 'name', ''):
        reveal_actors.add(ai_nick)
    elif allow_human:
        reveal_actors.update(nick for nick, side in driver.nick_side.items()
                             if getattr(side, 'name', str(side)) == owner)
    table_origin = table_balrog_commit(server,card_id,owner,origin,destination,role)
    def table_slot(zone):
        import re
        match = re.fullmatch(r'(Board[1-9]\d*)(?: \(chosen\))?',str(zone or ''))
        return match[1] if match else None
    last_table_slot = None
    reveal = None
    for i, line in enumerate(driver.feed.lines):
        if ' moves ' not in line and ' kills ' not in line:
            continue
        record = parse(line)
        if record.get('op') == 'move' and same_card(record.get('obj')):
            source_board = table_slot(record.get('from'))
            target_board = table_slot(record.get('to'))
            if target_board is not None:
                last_table_slot = target_board
                reveal = None
            elif (table_origin and source_board is not None and source_board == last_table_slot
                    and record.get('to') == slot and record.get('player') in reveal_actors):
                reveal = i
                last_table_slot = None
            elif (not table_origin and record.get('from') == hand+' (chosen)' and record.get('to') == slot
                    and record.get('player') in reveal_actors):
                reveal = i
            elif record.get('from') in (slot,'Current Action'):
                reveal = None
            elif source_board is not None:
                last_table_slot = None
        elif (record.get('op') == 'move' and record.get('from') in (slot,'Current Action')
              and str(record.get('obj','')).startswith('group (')
              and str(record.get('obj','')).endswith(')')):
            from rendered_replay import split_group
            if any(same_card(piece) for piece in split_group(record['obj'][7:-1])):
                reveal = None
        elif (record.get('op') == 'kill' and record.get('zone') in (slot,'Current Action')
              and any(same_card(piece) for piece in record.get('pieces',()))):
            reveal = None
        elif (record.get('op') == 'kill' and table_slot(record.get('zone')) == last_table_slot
              and any(same_card(piece) for piece in record.get('pieces',()))):
            last_table_slot = None
    if reveal is None and require_reveal:
        return None
    return {'pc':pc,'card':card,'reveal':reveal}


def current_ai_card_requires_reveal(server):
    """Any selected/committed current AI card needs its actual public reveal."""
    pc = getattr(server.driver.es,'pending_combat',None) or {}
    selected = selected_step_card(server)
    candidates = [selected] if selected is not None else []
    round_cards = committed_round_cards(pc)
    for index,(role,prefix) in enumerate((('atk','attacker'),('def','defender'))):
        owner = getattr(pc.get(prefix+'_side'),'name','')
        card = pc.get(role+'_combat_card')
        if card is None and round_cards is not None:
            card = round_cards[index]
        if owner == getattr(server.ai_side,'name','') and card is not None:
            candidates.append((card,owner,region_key(pc.get('attacker_region')),
                               region_key(pc.get('defender_region')),role))
    for candidate in candidates:
        context = observed_card_context(server,*candidate,require_reveal=False)
        if context is not None and context['reveal'] is None:
            return True
    return False


def awaiting_selected_card_reveal(server, call):
    """Do not ask dice before the current AI combat card is publicly revealed."""
    return call.kind == 'd6' and current_ai_card_requires_reveal(server)


def before_roll(server, call):
    """Only publish an observed forced AI target after its exact public reveal."""
    if call.kind != 'd6':
        return []
    observations = getattr(call, '_automatic_words_reports', ())
    if len(observations) != 1:
        return []
    card_id, target, owner, origin, destination, role = observations[0]
    origin,destination = region_key(origin),region_key(destination)
    context = observed_card_context(server,card_id,owner,origin,destination,role)
    if context is None or str(context['card'].combat_name).casefold() != 'words of power':
        return []
    driver,pc,reveal = server.driver,context['pc'],context['reveal']
    key = (driver.es.turn, pc.get('_battle_serial'), origin, destination,
           reveal, card_id, target)
    sent = getattr(server, '_automatic_words_reports_sent', None)
    if sent is None:
        sent = server._automatic_words_reports_sent = set()
    if key in sent:
        return []
    from . import chat
    text = chat.combat_effect_choice_note({'card_id': card_id, 'target': target})
    line = server._say('prompt', text)
    sent.add(key)
    return [line]
