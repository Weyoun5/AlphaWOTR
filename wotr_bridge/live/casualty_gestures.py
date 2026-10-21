"""Separate a shown casualty operation from compulsory consequences of it."""
from collections import Counter


def _battle_key(driver):
    pc = getattr(driver.es, 'pending_combat', None) or {}
    return (driver.es.turn, getattr(driver, '_cc_battle_serial', None),
            *(pc.get(key) for key in ('attacker_region', 'defender_region',
                                      'attacker_side', 'defender_side', 'rounds_fought')))


def early_context(driver):
    """Human defender losses already rolled while the AI pays its own hits."""
    pc = getattr(driver.es, 'pending_combat', None) or {}
    if (getattr(driver, 'spectator', False) or pc.get('phase') != 'casualty_allocation'
            or pc.get('alloc_stage') != 'attacker'
            or pc.get('deciding_side') is not driver.ai_side
            or pc.get('attacker_side') is not driver.ai_side
            or pc.get('defender_side') is not driver.human_side
            or type(pc.get('def_hits')) is not int or pc['def_hits'] <= 0):
        return None
    return dict(side=driver.human_side, region=pc['defender_region'], hits=pc['def_hits'])


def buffer_early(driver, pending, record):
    """Retain only this round's human casualties received during AI payment."""
    import ai
    from . import gesture_match as gm
    context = early_context(driver)
    if (context is None and pending is not None and pending.get('side') is driver.human_side
            and (driver.echo_pending() or driver.forced_ai_rows_pending())):
        context = live_context(driver.es, pending)
    if (context is None or pending is None or context['side'] is not driver.human_side
            or str(record.get('player', '')) == driver.nick):
        return False
    if record.get('op') not in ('kill', 'move'):
        return False
    if record['op'] == 'kill':
        tokens = record.get('pieces') or ()
    else:
        obj = str(record.get('obj', ''))
        tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    # Ordinary leaders can accompany a whole army's loss. Unrelated named
    # figures/cards/dice never become casualty evidence merely by grouping.
    for token in tokens:
        piece = gm.rr.parse_piece(token)
        if (piece is None or piece[0] not in ('unit', 'leader')
                or ai._side_of_nation(gm._nation_enum(piece[1])) is not driver.human_side):
            return False
    facts = gm.window_facts(gm._PrefixShim([record], None, None))
    region = gm.sanitize(str(context['region'])).lower()
    if not facts.armies or any(
            where != region or not kind.endswith((':regular', ':elite'))
            or gm._shadow_tkey(kind) != (driver.human_side.name == 'SHADOW')
            for (where, kind), count in facts.armies.items() if count):
        return False
    window = driver.pending_buf
    key = _battle_key(driver)
    if window is not None and window.records and getattr(window, '_early_casualty_key', None) != key:
        return False
    if window is None:
        window = driver.pending_buf = driver._new_phase_window()
    if not window.records:
        window._early_casualty_units, window._early_casualty_leaders = _human_counts(driver, region)
        window._early_casualty_hits = context['hits']
        window._early_casualty_region = region
    window._early_casualty_key = key
    window.records.append(record)
    driver.findings.append(('early_defender_casualty_buffered', str(record.get('norm', ''))))
    return True


def _human_counts(driver, region):
    import ai
    from . import gesture_match as gm
    units, leaders = Counter(), 0
    for where, army in driver.es.armies.items():
        if gm.sanitize(str(where)).lower() != region:
            continue
        units.update(gm._unit_type_key(unit.nation, unit.kind) for unit in army.units
                     if ai._side_of_nation(unit.nation) is driver.human_side)
        leaders += sum(1 for leader in army.leaders if getattr(leader, 'nation', None) is not None
                       and ai._side_of_nation(leader.nation) is driver.human_side)
    return units, leaders


def _already_applied(driver, window):
    """A forced defender allocation may finish inside the AI's last payment."""
    from . import gesture_match as gm
    expected = window._early_casualty_units.copy()
    facts = gm.window_facts(window)
    region = window._early_casualty_region
    before_hp = sum(count * (2 if kind.endswith(':elite') else 1) for kind, count in expected.items())
    paid = 0
    for (where, kind), count in facts.armies.items():
        if where != region:
            return False
        expected[kind] += count
        paid -= count * (2 if kind.endswith(':elite') else 1)
    if paid != min(window._early_casualty_hits, before_hp) or any(n < 0 for n in expected.values()):
        return False
    current, leaders = _human_counts(driver, region)
    if (+expected != current or (facts.leaders and
            window._early_casualty_leaders + facts.leaders.get(region, 0) != leaders)):
        return False
    for record in window.records:
        driver._consume_mirror_echo(record)
    driver.findings.append(('early_defender_casualty_already_applied', len(window.records)))
    driver.pending_buf = None
    return True


def _retry_early(driver):
    window = driver.pending_buf
    key = getattr(window, '_early_casualty_key', None)
    if key is None:
        return
    if key != _battle_key(driver):
        # A rewind/skip/new round cannot turn old casualties into new payment.
        driver.uncommitted.extend(record.get('norm', '') for record in window.records)
        driver.findings.append(('early_defender_casualty_expired', key))
        driver.pending_buf = None
        return
    if early_context(driver) is not None:
        return
    context = live_context(driver.es)
    if context is None and _already_applied(driver, window):
        return
    if driver.echo_pending() or driver.forced_ai_rows_pending():
        return
    if context is not None and context['side'] is driver.human_side:
        driver._try_pending()


def _native_allocation_complete(driver, context, window):
    """A current full board confirms all hits, without inferring a replacement."""
    import ai
    import emitter
    import game_state as gs
    from . import client_board, decision_endpoint, gesture_match as gm
    region = gm.sanitize(str(context['region'])).lower()
    flows, required = observed_unit_flow(window.records, driver.human_side)
    replacements = Counter()
    for flow in flows:
        for ((where, kind), sign), count in flow.items():
            if sign > 0 and kind.endswith(':regular'):
                replacements[(where, kind[:-7] + 'elite')] += count
    if not native_reductions_covered(required, replacements):
        return False
    facts = gm.window_facts(window)
    changes = Counter()
    for (where, kind), count in facts.armies.items():
        if not count:
            continue
        if (where != region or not kind.endswith((':regular', ':elite'))
                or gm._shadow_tkey(kind) != (driver.human_side.name == 'SHADOW')):
            return False
        changes[kind] += count
    paid = -sum(count * (2 if kind.endswith(':elite') else 1) for kind, count in changes.items())
    expected, _leaders = _human_counts(driver, region)
    remaining_hp = sum(count * (2 if kind.endswith(':elite') else 1) for kind, count in expected.items())
    due = min(context['hits'], remaining_hp) if context.get('kind') in ('combat', 'card') else context['hits']
    if paid <= 0 or paid != due:
        return False
    try:
        dump, _markers, _signature = decision_endpoint.read_complete(driver)
    except (OSError, ValueError):
        return False
    expected.update(changes)
    if any(count < 0 for count in expected.values()):
        return False
    zone = ('stronghold' if emitter.garrison_side_of(driver.es, region) == driver.human_side.name
            else 'region', region)
    shown = Counter()
    for pid, where in dump.items():
        if gm.zone_key(where) != zone:
            continue
        nation, kind = client_board.unit_type(pid, driver.view.dynamic_types)
        if kind in ('regular', 'elite') and ai._side_of_nation(nation) is driver.human_side:
            shown[gm._unit_type_key(nation, gs.UnitKind[kind.upper()])] += 1
    return +expected == shown


def live_context(state, pending=None):
    """The current casualty or cursor-owned combat unit payment."""
    import ai
    import emitter
    from .combat_payment_prompt import payment_context
    context = emitter.casualty_context(state)
    if context is not None:
        return context
    pending = ai.describe_pending(state) if pending is None else pending
    context = payment_context(pending, state) if pending else None
    if context is not None:
        context = dict(context, kind='combat_payment',
                       card_id=(state.pending_combat.get('effect_choice') or {}).get('card_id'))
    return context


def complete_live_allocation(driver):
    """An explicit finish disambiguates a whole-Elite loss from a reduction."""
    import ai
    from . import chat
    if getattr(driver, 'spectator', False):
        return None
    pending = ai.describe_pending(driver.es)
    context = live_context(driver.es, pending)
    window = driver.pending_buf
    if (not pending or pending['side'] is not driver.human_side
            or context is None or context['side'] is not driver.human_side
            or window is None or not window.records):
        return None
    window._casualty_closed = True
    if driver._try_pending():
        return []
    if getattr(getattr(driver, '_last_pending_match', None), 'starved', False):
        return []  # The confirmed allocation still owns its native RNG request.
    # The finish only closes existing evidence. It cannot choose a missing
    # casualty, waive an unplaced Regular, or authorize subsequent gestures.
    if driver.pending_buf is window:
        window._casualty_closed = False
    return [('chat', 'prompt', chat._register(
        'The casualty allocation is not complete yet. Finish the required '
        'unit removals and Regular replacements, then say done.'))]


def live_completion_note(driver):
    """Announce only a still-open ambiguity after the entire feed batch."""
    from . import chat
    from .partial_casualty_status import waiting
    _retry_early(driver)
    window = driver.pending_buf
    if (window is None
            or not getattr(getattr(driver, '_last_pending_match', None),
                           'ambiguous_elite_prefix', False)
            or waiting(driver) is None):
        return []
    context = live_context(driver.es)
    if (context is not None and context['side'] is driver.human_side
            and not driver.echo_pending() and not driver.forced_ai_rows_pending()
            and _native_allocation_complete(driver, context, window)):
        window._casualty_closed = True
        if driver._try_pending() or getattr(getattr(driver, '_last_pending_match', None), 'starved', False):
            return []
        if driver.pending_buf is window:
            window._casualty_closed = False
    if getattr(window, '_elite_completion_prompted', False):
        return []
    window._elite_completion_prompted = True
    return [('chat', 'prompt', chat._register(
        'Finish your casualty changes on the board. An Elite removed entirely '
        'takes 2 hits; reducing it takes 1 hit and requires its same-nation '
        'Regular replacement. The completed board records the allocation.'))]


def prefix_matches(events, records):
    """Independent figure kinds commute; preserve each supply pool's order.

    Grouped Elite removals can precede grouped Regular replacements. Regular
    removal/placement order remains significant: it determines whether a later
    replacement comes from casualties or the reinforcement pool.
    """
    remaining = [record.copy() for record in records]
    def owner(event):
        (region, unit_type), _sign = event
        return region, unit_type
    for event in events:
        for row in remaining:
            if row[event] > 0:
                row[event] -= 1
                break
            if any(count > 0 and owner(other) == owner(event)
                   for other, count in row.items()):
                return False
        else:
            return False
    return True


def observed_unit_flow(records, side=None, undo_drafts=True):
    """Physical loss evidence after reversing draft removals in this choice.

    A unit removed from the battle and returned to the same place has not
    paid a hit. Keep replacement-pool provenance: placing a Regular and then
    losing it is still two operations, as is returning it to a different pool.
    Only an earlier outward transfer followed by its exact physical inverse
    cancels. Grouped and separate transfers use the same fungible unit counts.
    """
    from . import gesture_match as gm
    records = list(records)
    flows = []
    for record in records:
        facts = gm.window_facts(gm._PrefixShim([record], None, None))
        flows.append(Counter({(key, 1 if value > 0 else -1): abs(value)
                              for key, value in facts.armies.items() if value and
                              (side is None or gm._shadow_tkey(key[1]) ==
                               (side.name == 'SHADOW'))}))
    outward = {}
    if undo_drafts:
        for index, record in enumerate(records):
            if record.get('op') == 'move':
                source, target = gm.zone_key(record.get('from')), gm.zone_key(record.get('to'))
                obj = str(record.get('obj', ''))
                tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
            elif record.get('op') == 'kill':
                source, target = gm.zone_key(record.get('zone')), ('dead',)
                tokens = record.get('pieces') or ()
            else:
                continue
            for token in tokens:
                piece = gm.rr.parse_piece(token)
                if piece is None or piece[0] != 'unit' or piece[3] <= 0 or source == target:
                    continue
                _kind, nation, rank, count = piece
                typed = (nation, rank)
                one = dict(record)
                if record.get('op') == 'move':
                    one['obj'] = token
                else:
                    one['pieces'] = [token]
                facts = gm.window_facts(gm._PrefixShim([one], None, None))
                event = Counter({(key, 1 if value > 0 else -1): abs(value) // count
                                 for key, value in facts.armies.items() if value and
                                 (side is None or gm._shadow_tkey(key[1]) ==
                                  (side.name == 'SHADOW'))})
                if not event:
                    continue
                reverse = outward.get((typed, target, source), [])
                remaining = count
                while reverse and remaining:
                    old = reverse[-1]
                    canceled = min(remaining, old[1])
                    for key, amount in event.items():
                        flows[index][key] -= canceled * amount
                    for key, amount in old[2].items():
                        flows[old[0]][key] -= canceled * amount
                    remaining -= canceled
                    old[1] -= canceled
                    if not old[1]:
                        reverse.pop()
                if remaining and source[0] in ('region', 'stronghold', 'mordor'):
                    outward.setdefault((typed, source, target), []).append([index, remaining, event])
    flows = [+flow for flow in flows]
    required = Counter()
    for record, flow in zip(records, flows):
        if 'Controls.304' in ((record.get('meta') or {}).get('tags') or ()):
            for (key, sign), count in flow.items():
                if sign < 0 and key[1].endswith(':elite'):
                    required[key] += count
    return flows, required


def native_reductions_covered(required, performed):
    """An explicit downgrade remains a reduction even after done or recovery."""
    return all(performed[key] >= count for key, count in required.items())


def payment_reductions(before, action, side):
    """Count the validated unit-payment primitive without replaying the round."""
    from .round_payment_gestures import is_action, EFFECTS
    params = getattr(action, 'params', None) or {}
    pc = getattr(before, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    if (not is_action(action) or params.get('kind') != 'elite'
            or params.get('operation') != 'downgrade'
            or effect.get('effect') not in EFFECTS | {'onslaught_payment'}
            or effect.get('owner') not in ('atk', 'def')
            or pc.get('deciding_side') is not side):
        return Counter()
    import game_state as gs
    from . import gesture_match as gm
    role = 'attacker' if effect['owner'] == 'atk' else 'defender'
    if pc.get(role + '_side') is not side:
        return Counter()
    key = (gm.sanitize(str(pc[role + '_region'])).lower(),
           gm._unit_type_key(gs.Nation[params['nation']], gs.UnitKind.ELITE))
    return Counter({key: 1})


def equivalence_state(post, chain):
    """Compare casualty results without per-apply physical witness history.

    Keep the actual committed state/witnesses untouched. Those witnesses tell
    the renderer how this representative path happened, not a future rule.
    Replacement-source order remains significant within each nation.
    """
    import copy
    import emitter
    from .onslaught_gestures import is_payment_action
    from .round_payment_gestures import is_action as round_action, EFFECTS
    payload = (getattr(post, 'pending_combat', None) or {}).get('_round_resume') or {}
    cursor = payload.get('cursor')
    completed_round_payment = False
    if cursor is None and chain and all(round_action(action) for action in chain):
        from cards import get_card
        faces = {'relentless assault', 'charge', 'sudden strike', "durin's bane"}
        completed_round_payment = bool(getattr(post, '_last_round_phase_call', False)) and all(
            get_card(action.params['card_id']) is not None
            and get_card(action.params['card_id']).combat_name.casefold() in faces for action in chain)
    if completed_round_payment:
        result = copy.deepcopy(post)
        battle_result = getattr(result, '_last_battle_result', None)
        if battle_result is not None:
            for role in ('attacker', 'defender'):
                getattr(battle_result, role+'_casualties').sort(key=lambda unit:unit.nation.name)
        result._casualty_operations = []
        if hasattr(result, '_dg_src_log'):
            result._dg_src_log.sort(key=lambda row:str(getattr(row[0], 'name', row[0])))
        return result
    if chain and all(round_action(action) for action in chain) and cursor is not None:
        effect = ((getattr(post, 'pending_combat', None) or {}).get('effect_choice') or {}).get('effect')
        if effect in EFFECTS and getattr(cursor.combat, '_live_state', None) is post:
            # The back-reference is an ownership invariant, not a second game
            # state. Check it before removing it from this equality-only copy.
            from . import gesture_match as gm
            result = copy.deepcopy(post)
            copied_cursor = result.pending_combat['_round_resume']['cursor']
            combat = copied_cursor.combat
            combat._live_state = None
            if copied_cursor.pending is None and copied_cursor.selection is None:
                # A blocked roll follows the last paid primitive before the
                # outer dispatcher can replace its old option descriptor.
                # The cursor proves that descriptor is no longer a choice.
                result.pending_combat['effect_choice'].pop('options', None)
            for role in ('attacker', 'defender'):
                casualties = getattr(combat, role+'_casualties')
                # Only different nations commute; retain each nation's order.
                casualties.sort(key=lambda unit:unit.nation.name)
                army = getattr(combat, role+'_army')
                army.units.sort(key=lambda unit:repr(gm._typed_value_key(unit)))
                army.leaders.sort(key=lambda leader:repr(gm._typed_value_key(leader)))
            result._casualty_operations = []
            if hasattr(result, '_dg_src_log'):
                result._dg_src_log.sort(key=lambda row:str(getattr(row[0], 'name', row[0])))
            return result
    if not chain or not (all(emitter.casualty_action_params(a) is not None for a in chain)
                         or all(is_payment_action(a) for a in chain)):
        return post
    result = copy.copy(post)
    result._casualty_operations = []
    if hasattr(post, '_dg_src_log'):
        result._dg_src_log = sorted(post._dg_src_log,
                                    key=lambda row: str(getattr(row[0], 'name', row[0])))
    return result


def unfinished_elite_removal(before, post, chain, side):
    """An Elite kill may be the first half of a downgrade until more is shown."""
    import emitter
    if not chain or not all(emitter.casualty_action_params(a) is not None for a in chain):
        return False
    old, new = emitter.casualty_context(before), emitter.casualty_context(post)
    if old is None or new is None or old['side'] is not side or new['side'] is not side:
        return False
    if any(old[key] != new[key] for key in ('kind', 'stage', 'region', 'card_id')):
        return False
    return (0 < new['hits'] < old['hits']
            and any((emitter.casualty_action_params(a) or {}).get('operation') == 'remove_elite'
                    for a in chain))


def event_flow(event, type_key, sanitize):
    import game_state as gs
    region = sanitize(str(event['region'])).lower()
    operation = event['operation']
    kind = gs.UnitKind.REGULAR if operation == 'remove_regular' else gs.UnitKind.ELITE
    rows = [((region, type_key(event['nation'], kind)), -1)]
    if operation == 'reduce_elite' and event.get('replacement_source') is not None:
        rows.append(((region, type_key(event['nation'], gs.UnitKind.REGULAR)), 1))
    return rows


def observed_operations(before, post, action, side, prior_flow, prior_events,
                        observed_flow, records, type_key, sanitize, prefix_matches):
    """Allow only an entire observed prefix of this action's fresh witnesses.

    The public engine apply resets its operation witnesses and applies exactly
    one selected operation followed by compulsory operations. A later actual
    choice stays pending in the returned state. Never inspect prior witnesses,
    infer a missing replacement, or choose another casualty operation here.
    """
    import emitter
    events = list(getattr(post, '_casualty_operations', None) or [])
    own = [event for event in events if event.get('side') is side]
    if getattr(action, 'action_type', '') != 'casualty_allocate' or not own:
        return own, Counter()
    context = emitter.casualty_context(before)
    params = action.params or {}
    first = own[0]
    if (context is None or context['side'] is not side
            or first.get('nation') != params.get('nation')
            or first.get('operation') != params.get('operation')
            or first.get('remaining_before') != context['hits']
            or sanitize(str(first.get('region'))).lower()
               != sanitize(str(context['region'])).lower()):
        return own, Counter()
    flow = prior_flow.copy()
    history = list(prior_events)
    for index, event in enumerate(own):
        addition = event_flow(event, type_key, sanitize)
        candidate = flow.copy(); candidate.update(addition)
        if (any(value > observed_flow[key] for key, value in candidate.items())
                or not prefix_matches(history + addition, records)):
            if index > 0 and flow == observed_flow:
                withheld = Counter()
                for later in own[index:]:
                    for key, sign in event_flow(later, type_key, sanitize):
                        withheld[key] += sign
                return own[:index], withheld
            return own, Counter()
        flow = candidate
        history.extend(addition)
    return own, Counter()


def physical_coverage(engine_facts, withheld):
    """Coverage describes shown operations; the committed engine state stays full."""
    if not withheld:
        return engine_facts
    import copy
    facts = copy.copy(engine_facts)
    facts.armies = engine_facts.armies.copy()
    for key, value in withheld.items():
        facts.armies[key] -= value
    return facts


def awaiting_replacement(expected, observed, events, records, prefix_matches):
    """A legal downgrade's only missing evidence is its Regular placement."""
    missing = expected - observed
    # Several Elite kills can arrive before any of their replacements. Extra
    # removals cannot disprove this legal unfinished reduction. This is only a
    # reason to retain the decision; it never supplies evidence for a commit.
    extra = observed - expected
    if not missing or any(sign > 0 for (_key, sign) in extra):
        return False
    if any(not unit_type.endswith(':regular')
           for ((_region, unit_type), _sign) in missing):
        return False
    # A last Elite can be reduced and its replacement compulsorily removed
    # by the remaining hit. That entire unshown Regular lifetime still starts
    # with a missing replacement; it cannot authenticate a whole-Elite kill.
    if not any(sign == 1 for _key, sign in missing):
        return False
    if any(sign < 0 and amount > missing[(key, 1)]
           for (key, sign), amount in missing.items()):
        return False
    shown = observed.copy()
    ordered = []
    for event in events:
        if shown[event]:
            ordered.append(event)
            shown[event] -= 1
    return bool(ordered) and prefix_matches(ordered, records)
