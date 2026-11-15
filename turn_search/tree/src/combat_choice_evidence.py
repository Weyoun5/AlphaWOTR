"""Conservative chronological labels for observed, paused combat rounds.

Consumes the existing rendered record schema, not model suggestions or guessed
rolls. Missing gestures, ambiguous magnitudes, and cleanup deaths abstain.
"""
import unicodedata


def admission_refusal(state, evidence, action=None):
    """Require real phase provenance before v52 effect rows can be retained."""
    pc = getattr(state, 'pending_combat', None) or {}
    ec = pc.get('effect_choice') or {}
    effect = ec.get('effect')
    if effect == 'words_of_power' and ec.get('stage') == 'pre':
        return _words_admission_refusal(state, evidence, action)
    if not isinstance(evidence, dict) or evidence.get('schema') != 'observed_round_v1':
        return 'combat_observed_evidence_missing'
    if (evidence.get('effect') != effect or type(evidence.get('line')) is not int
            or evidence['line'] <= 0 or not evidence.get('reason')):
        return 'combat_observed_evidence_invalid'
    if ec.get('stage') in ('post', 'onslaught_payment') and effect in ('onslaught', 'onslaught_payment'):
        completed = pc.get('_observed_round_completed') or {}
        if (pc.get('rounds_fought', 0) < 1 or completed.get('schema') != 'observed_round_v1'
                or completed.get('round') != pc['rounds_fought'] - 1
                or set(evidence.get('observed_rolls', ())) != {'combat', 'reroll', 'ordinary_casualties'}
                or evidence.get('round') != pc['rounds_fought'] - 1):
            return 'combat_onslaught_observation_unbound'
        return None
    cursor = (pc.get('_round_resume') or {}).get('cursor')
    if ec.get('stage') != 'round' or cursor is None or cursor.pending is None:
        return 'combat_observed_cursor_missing'
    if (cursor.pending.get('effect') != effect or cursor.pending.get('owner') != ec.get('owner')
            or cursor.pending.get('stage') != ec.get('round_stage')
            or evidence.get('round') != pc.get('rounds_fought', 0)
            or set(evidence.get('observed_rolls', ())) != set(cursor.observed_rolls)):
        return 'combat_observed_stage_mismatch'
    seen = set(cursor.observed_rolls)
    if effect in ('mighty_attack',):
        role = 'attacker' if ec.get('owner') == 'atk' else 'defender'
        valid = (('combat', role) in cursor.observed_roll_sides
                 and ('reroll', role) not in cursor.observed_roll_sides)
    elif effect in ('fateful_strike', 'blade_of_westernesse', 'heroic_death', 'black_breath'):
        valid = {'combat', 'reroll'} <= seen
    elif effect in ('pre_combat_casualty', 'relentless_assault', 'relentless_assault_payment', 'dread_and_despair'):
        valid = 'combat' not in seen
    else:
        valid = False
    return None if valid else 'combat_observed_timing_mismatch'


def _norm(value):
    value = ''.join(c for c in unicodedata.normalize('NFKD', str(value or ''))
                    if not unicodedata.combining(c)).lower().replace('-', ' ').strip()
    return {'meriadoc brandybuck': 'meriadoc', 'peregrin took': 'peregrin'}.get(value, value)


def _named_choice(state, effect, owner, round_items, chats, lower, upper):
    """Only the revealed card's owner can announce its current choice."""
    from combat_decision_text import parse_choice
    from cards import get_card
    pc = state.pending_combat
    cursor = (pc.get('_round_resume') or {}).get('cursor')
    card = ((cursor.combat.attacker_card if owner == 'atk' else cursor.combat.defender_card)
            if cursor is not None else get_card(pc.get('atk_combat_card' if owner == 'atk' else 'def_combat_card')))
    # Onslaught survives the ordinary round cleanup: the engine deliberately
    # transfers its revealed card identity into the deferred effect/payment.
    if card is None and effect == 'onslaught':
        pending = pc.get('_onslaught_payment') or pc.get('effect_choice') or {}
        if pending.get('owner') == owner:
            card = get_card(pending.get('card_id'))
    players = {_norm(c.get('player')) for c, actor, _ in round_items
               if actor == ('A' if owner == 'atk' else 'D') and c.get('player')}
    if len(players) != 1:
        return None, None, None
    rows = [(int(row[0]), parse_choice(row[1], card=card)) for row in chats if len(row) == 3
            and lower < int(row[0]) < upper and _norm(row[2]) in players]
    rows = [(line, parsed) for line, parsed in rows if parsed and parsed['effect'] == effect]
    if not rows:
        return None, None, None
    if card is None or _norm(card.combat_name) != effect.replace('_', ' '):
        return None, None, 'named_choice_card_unbound'
    if any(parsed != rows[0][1] for _, parsed in rows[1:]):
        return None, None, 'conflicting_named_choices'
    return rows[0][1], rows[0][0], None


def match_round_choice(state, batch, legal, round_items, lower, upper,
                       after_line=0, chats=()):
    """Return (legal index, evidence line, reason); index None means abstain.

    The observation comes from the real continuation. round_items is the
    existing parser's (chunk, A/D, pre/main/re/post) segmentation. Evidence is
    restricted to the printed timing window and actual participating region.
    Silence alone is unknown. A physical victim plus a completed, unsaturated
    casualty ledger can prove Heroic cancellation; explicit card-specific chat
    can prove a decline when bound to the actual owner and timing window.
    """
    import rendered_replay as rr
    pc = getattr(state, 'pending_combat', None) or {}
    ec = pc.get('effect_choice') or {}
    payload = pc.get('_round_resume') or {}
    cursor = payload.get('cursor')
    if ec.get('stage') != 'round' or cursor is None:
        return None, None, 'missing_observed_continuation'
    effect, owner = ec.get('effect'), ec.get('owner')
    if owner not in ('atk', 'def'):
        return None, None, 'unknown_owner'
    observed = set(cursor.observed_rolls)
    mains = [int(c['line']) for c, _, phase in round_items if phase == 'main']
    rerolls = [int(c['line']) for c, _, phase in round_items if phase == 're']
    pre = [int(c['line']) for c, _, phase in round_items if phase == 'pre']
    no_roll_ra = not mains and effect in ('relentless_assault', 'relentless_assault_payment')
    if not mains and effect != 'pre_combat_casualty' and not no_roll_ra:
        return None, None, 'missing_recorded_combat_rolls'
    lower = max(int(lower), int(after_line))
    upper = int(upper)
    round_upper = upper
    if effect == 'pre_combat_casualty':
        if 'combat' in observed:
            return None, None, 'precasualty_after_combat'
        attacking_role = 'defender' if owner == 'atk' else 'attacker'
        if ('pre_combat', attacking_role) not in cursor.observed_roll_sides:
            return None, None, 'missing_preattack_observation'
        tag = 'D' if owner == 'atk' else 'A'
        own_pre = [int(c['line']) for c, actor, phase in round_items if phase == 'pre' and actor == tag]
        if len(own_pre) != 1:
            return None, None, 'ambiguous_preattack_chunk'
        lower = max(lower, own_pre[0])
        upper = min([upper] + mains + [n for n in pre if n > own_pre[0]])
    elif effect in ('relentless_assault', 'relentless_assault_payment', 'dread_and_despair'):
        if 'combat' in observed:
            return None, None, 'choice_after_combat_roll'
        lower = max([lower] + pre)
        upper = min([upper] + mains)
    elif effect == 'mighty_attack':
        role = 'attacker' if owner == 'atk' else 'defender'
        tag = 'A' if owner == 'atk' else 'D'
        if (('combat', role) not in cursor.observed_roll_sides
                or ('reroll', role) in cursor.observed_roll_sides):
            return None, None, 'missing_combat_only_observation'
        own_mains = [int(c['line']) for c, actor, phase in round_items
                     if actor == tag and phase == 'main']
        own_rerolls = [int(c['line']) for c, actor, phase in round_items
                       if actor == tag and phase == 're']
        if len(own_mains) != 1:
            return None, None, 'missing_unique_owner_combat_roll'
        # Recorded owner/phase attribution is independent of whether the
        # physical game rolled both mains first or finished one side first.
        lower = max(lower, own_mains[0])
        upper = min([upper] + own_rerolls)
    elif effect in ('fateful_strike', 'blade_of_westernesse', 'heroic_death', 'black_breath'):
        if not {'combat', 'reroll'} <= observed:
            return None, None, 'missing_completed_roll_observation'
        lower = max([lower] + mains + rerolls)
    else:
        return None, None, 'unsupported_effect'

    records = sorted((r for r in batch.get('records', ())
                      if lower < int(r.get('line', 0)) < upper), key=lambda r: int(r['line']))
    battle = cursor.combat
    own = battle.attacker_army if owner == 'atk' else battle.defender_army
    enemy = battle.defender_army if owner == 'atk' else battle.attacker_army
    # Reconstructed armies may carry a board.Region object; the public battle
    # keys are the authoritative locations, including split siege participants.
    own_region = rr.sanitize(pc['attacker_region' if owner == 'atk' else 'defender_region'])
    enemy_region = rr.sanitize(pc['defender_region' if owner == 'atk' else 'attacker_region'])
    own_side = pc['attacker_side' if owner == 'atk' else 'defender_side'].name
    expected_slot = 'current fp card' if own_side == 'FREE_PEOPLES' else 'current sa card'
    options = [(i, {k: v for k, v in a.params.items() if k != 'card_id'})
               for i, a in enumerate(legal) if a.action_type == 'combat_effect_choice']

    def region(zone):
        kind, key = rr.zone_kind(zone)
        return rr.sanitize(key) if kind in ('region', 'stronghold') else None

    def pieces(record):
        if record.get('op') == 'kill':
            tokens = record.get('pieces') or ()
        else:
            # Reuse the rendered group parser, including internal card commas.
            from rendered_extract import _group_tokens
            tokens = _group_tokens(record.get('obj', ''))
        return [p for p in (rr.parse_piece(t) for t in tokens) if p]

    def loss_pieces(record):
        """A native kill and a drag to the casualty pool record the same loss.

        Card-slot parking and ordinary movement are not deaths. Only Shadow
        army units and ordinary Nazgul recycle into reinforcements; named
        figures and Free Peoples casualties do not.
        """
        if record.get('op') == 'kill':
            return pieces(record)
        if record.get('op') != 'move' or region(record.get('from', '')) is None:
            return []
        destination = str(record.get('to', '')).strip().lower()
        kind, _ = rr.zone_kind(destination)
        if kind not in ('cas', 'reinf'):
            return []
        from rendered_extract import _unit_side_str
        losses = []
        for piece in pieces(record):
            category, name, _, _ = piece
            if category in ('unit', 'leader'):
                side = _unit_side_str(name)
            elif category == 'nazgul' or (category == 'named' and name in rr.NAMED_MINIONS.values()):
                side = 'SHADOW'
            elif category == 'named' and name in rr.COMP_ALIAS.values():
                side = 'FREE_PEOPLES'
            else:
                continue
            if ((destination.startswith(('fp ', 'free peoples ')) and side != 'FREE_PEOPLES')
                    or (destination.startswith(('sa ', 'shadow ')) and side != 'SHADOW')):
                continue
            if kind == 'reinf' and not (side == 'SHADOW' and category in ('unit', 'nazgul')):
                continue
            losses.append(piece)
        return losses

    def loss_region(record):
        return region(record.get('zone' if record.get('op') == 'kill' else 'from', ''))

    named, named_line, named_error = _named_choice(state, effect, owner, round_items, chats, lower, upper)
    if named_error:
        return None, None, named_error
    explicit = _explicit_effect_decline(effect, owner, options, round_items,
                                        chats, lower, upper)
    if named and named.get('decline') and explicit is None:
        declines = [i for i, opt in options if opt == {'count': 0}]
        if len(declines) == 1:
            explicit = (declines[0], named_line, 'explicit_effect_decline_chat')
    if named and named.get('decline') and explicit is None:
        return None, None, 'named_decline_not_legal'
    if named and not named.get('decline') and explicit is not None:
        return None, None, 'conflicting_named_choices'
    if explicit is not None:
        conflicting_types = ('unit',) if effect == 'relentless_assault' else ('named', 'leader', 'nazgul')
        if any(any(p[0] in conflicting_types for p in loss_pieces(r))
               for r in records):
            return None, None, 'explicit_decline_conflicts_with_loss_gesture'
        if effect == 'relentless_assault' and any(
                r.get('op') == 'move' and region(r.get('from', '')) == own_region
                and any(p[0] == 'unit' for p in pieces(r)) for r in records):
            return None, None, 'explicit_decline_conflicts_with_payment_gesture'
        return explicit

    if effect in ('relentless_assault', 'relentless_assault_payment', 'pre_combat_casualty'):
        total = None if effect == 'relentless_assault' else (cursor.pre_total if effect == 'pre_combat_casualty' else cursor.ra_paid + cursor.ra_remaining)
        paid = cursor.pre_paid if effect == 'pre_combat_casualty' else cursor.ra_paid
        result = _match_unit_sequence(state, batch, options, own_side, own_region, lower, upper,
                                    total, paid, effect == 'relentless_assault', cleanup_upper=round_upper,
                                    wipe_army=own if no_roll_ra else None)
        if named and result[0] is not None and effect == 'relentless_assault':
            actual = dict(options)[result[0]]
            if actual != {'count': named.get('count')}:
                return None, None, 'named_count_conflicts_with_payment'
        return result

    if effect == 'mighty_attack':
        parked, lines, park_groups = [], [], []
        for record in records:
            if record.get('op') != 'move':
                continue
            if str(record.get('from', '')).strip().lower() == expected_slot:
                return None, None, 'park_correction_or_return'
            if region(record.get('from', '')) != own_region or str(record.get('to', '')).strip().lower() != expected_slot:
                continue
            group = pieces(record)
            parked.extend(group); lines.append(int(record['line']))
            park_groups.append((group, int(record['line'])))
        if not parked:
            return None, None, 'missing_park_gesture'
        targets = [_norm(p[1]) for p in parked if p[0] == 'named' and p[3] == 1]
        if len(targets) != 1 or len(parked) != 1:
            return None, None, 'ambiguous_payer_gesture'
        found = [i for i, opt in options if _norm(opt.get('target')) == targets[0]]
        return (found[0], max(lines), 'park_gesture') if len(found) == 1 else (None, None, 'unmatched_park_gesture')

    if effect == 'dread_and_despair':
        from combat_forfeit_text import dread_forfeit_count
        card = battle.attacker_card if owner == 'atk' else battle.defender_card
        if card is None or _norm(card.combat_name) != 'dread and despair':
            return None, None, 'ff_card_unbound'
        tag = 'A' if owner == 'atk' else 'D'
        players = {_norm(c.get('player')) for c, actor, _ in round_items
                   if actor == tag and c.get('player')}
        if len(players) != 1:
            return None, None, 'ff_owner_unbound'
        evidence = [(int(row[0]), dread_forfeit_count(row[1]))
                    for row in chats if len(row) == 3 and _norm(row[2]) in players
                    and lower < int(row[0]) < upper]
        evidence = [(line, count) for line, count in evidence if count is not None]
        if len(evidence) != 1:
            return None, None, 'missing_or_ambiguous_ff_chat'
        line, count = evidence[0]
        found = [i for i, opt in options if opt == {'count': count}]
        return (found[0], line, 'explicit_ff_chat') if len(found) == 1 else (None, None, 'ff_count_not_legal')

    target_region = own_region if effect == 'heroic_death' else enemy_region
    opposite_card = battle.defender_card if owner == 'atk' else battle.attacker_card
    if effect == 'black_breath' and not named and opposite_card and _norm(opposite_card.combat_name) == 'heroic death':
        # A sole FP death could equally mean BB declined and HD sacrificed
        # that figure. Do not assign it to the earlier optional card.
        deaths = 0
        for record in records:
            parsed = loss_pieces(record)
            if any(p[0] == 'unit' for p in parsed): break
            if loss_region(record) == target_region:
                deaths += sum(p[3] for p in parsed if p[0] in ('named', 'leader'))
        if deaths < 2:
            return None, None, 'black_breath_heroic_attribution_ambiguous'
    candidates = []
    for record in records:
        parsed = loss_pieces(record)
        # After any ordinary Army casualty, later figure deaths may just be
        # mandatory army-elimination cleanup. They cannot label a card target.
        if any(p[0] == 'unit' for p in parsed):
            break
        if loss_region(record) != target_region:
            continue
        identities = {(p[0], p[1]) for p in parsed if p[0] in ('named', 'leader', 'nazgul')}
        if any(later.get('op') == 'move'
               and int(later['line']) > int(record['line'])
               and rr.zone_kind(later.get('from', ''))[0] in ('cas', 'reinf')
               and region(later.get('to', '')) in rr.board.regions
               and any((p[0], p[1]) in identities for p in pieces(later))
               for later in records):
            # Returning the figure makes the loss unproven. Ordinary leaders
            # and Nazgul have no identity in rendered logs, so do not guess
            # whether a replacement token was the one just removed.
            return None, None, 'figure_loss_reverted'
        targets = []
        for p in parsed:
            if p[3] != 1:
                return None, None, 'multiple_figure_deaths'
            if p[0] == 'named':
                targets.append(_norm(p[1]))
            elif p[0] in ('leader', 'nazgul'):
                targets.append(p[0])
        for target in targets:
            indices = [i for i, opt in options if _norm(opt.get('target')) == target]
            if named:
                indices = [i for i in indices if _norm(named.get('target')) == target
                           and ('count' not in named or dict(options)[i].get('count') == named['count'])]
                if not indices:
                    return None, None, 'named_target_conflicts_with_figure_gesture'
                if effect == 'heroic_death':
                    balance = _heroic_casualty_balance(state, batch, options, target, own_side,
                                                     own_region, lower, upper, int(record['line']), round_items)
                    if balance is not None and balance[0] not in indices:
                        return None, None, 'named_count_conflicts_with_casualty_balance'
            if len(indices) > 1:
                if effect == 'heroic_death':
                    magnitude = _heroic_casualty_balance(
                        state, batch, options, target, own_side, own_region,
                        lower, upper, int(record['line']), round_items)
                    if magnitude is not None:
                        return magnitude
                return None, None, 'sacrifice_magnitude_ambiguous'
            if len(indices) == 1:
                candidates.append((indices[0], int(record['line'])))
        if candidates:
            # The first unconsumed, correctly located figure gesture belongs
            # to this initiative-ordered pause. A later gesture may belong to
            # the next card (e.g. Black Breath then Heroic Death).
            break
    if len(candidates) != 1:
        return None, None, 'missing_or_ambiguous_figure_gesture'
    return candidates[0][0], candidates[0][1], 'figure_gesture'


def _explicit_effect_decline(effect, owner, options, round_items, chats, lower, upper):
    """Never reinterpret generic 'nc', 'pass', or an opponent's suggestion."""
    import re
    aliases = {'heroic_death': r'(?:heroic death|hd)',
               'black_breath': r'(?:black breath|bb)',
               'relentless_assault': r'(?:relentless assault|ra)',
               'onslaught': r'onslaught'}
    if effect not in aliases or not any(opt == {'count': 0} for _, opt in options):
        return None
    tag = 'A' if owner == 'atk' else 'D'
    players = {_norm(chunk.get('player')) for chunk, actor, _ in round_items
               if actor == tag and chunk.get('player')}
    if len(players) != 1:
        return None
    name = aliases[effect]
    pattern = re.compile(rf'\s*(?:(?:no|skip|decline)\s+{name}|{name}\s*(?:0|no|skip|decline))\s*[.!]?\s*', re.I)
    from combat_decision_text import parse_choice
    matches = [(int(row[0]), str(row[1])) for row in chats if len(row) == 3
               and lower < int(row[0]) < upper and _norm(row[2]) in players
               and (pattern.fullmatch(str(row[1])) or
                    parse_choice(row[1]) == {'effect': effect, 'decline': True, 'count': 0})]
    if len(matches) != 1:
        return None
    found = [index for index, opt in options if opt == {'count': 0}]
    return (found[0], matches[0][0], 'explicit_effect_decline_chat') if len(found) == 1 else None


def _heroic_casualty_balance(state, batch, options, target, side, region,
                            lower, upper, victim_line, round_items=()):
    """Recover cancellation only from a closed, surviving Army loss ledger.

    A wiped Army saturates its observable losses, and an Elite without a
    replacement can have multiple hit costs. Both remain unknown.
    """
    import rendered_extract as rx
    import rendered_replay as rr
    pc = state.pending_combat
    cursor = pc['_round_resume']['cursor'];battle = cursor.combat
    own_role, enemy_role = ('attacker', 'defender') if pc['effect_choice']['owner'] == 'atk' else ('defender', 'attacker')
    army = getattr(battle, own_role + '_army')
    records = [r for r in batch.get('records', ()) if lower < int(r.get('line', 0)) < upper]
    # Require the committed card's physical disposal, not an arbitrary end of
    # file or silence following a figure death.
    closures = [int(r['line']) for r in records if r.get('op') == 'kill'
                and 'current' in str(r.get('zone', '')).lower()
                and any('/heroic death' in _norm(t) for t in r.get('pieces', ()))]
    if not closures or max(closures) < victim_line:
        return None
    # Unit movement between regions can hide losses or mix in reinforcements.
    # A final march after the card's disposal is outside this ledger.
    end = max(closures) + 1
    ledger_upper = upper
    incoming = max(0, getattr(battle, enemy_role + '_hits')
                   + getattr(battle, enemy_role + '_bonus_hits')
                   - getattr(battle, own_role + '_hits_cancelled'))
    enemy_card = getattr(battle, enemy_role + '_card')
    enemy_cancelled = getattr(battle, enemy_role + '_card_cancelled')
    if enemy_card is not None and not enemy_cancelled:
        enemy_effect = _norm(enemy_card.combat_name)
        if enemy_effect == 'great host':
            # Its automatic hit has no public roll to mark the end of ordinary
            # damage. A card-disposal ledger can mix both damage windows and
            # falsely infer a smaller Heroic sacrifice. Explicit owner-bound
            # count + figure evidence remains usable by the caller.
            # Preserve the simple provable no-trigger case: even with no
            # cancellation each incoming hit removes at most one Army unit,
            # and ordinary casualties cannot increase the Shadow unit count.
            minimum_own_units = max(0, len(army.units) - incoming)
            enemy_units = len(getattr(battle, enemy_role + '_army').units)
            if enemy_units >= 2 * minimum_own_units:
                return None
        if enemy_effect in ('onslaught', 'we come to kill'):
            enemy_tag = 'D' if own_role == 'attacker' else 'A'
            post = [int(chunk['line']) for chunk, actor, phase in round_items
                    if actor == enemy_tag and phase == 'post']
            if len(post) != 1 or not victim_line < post[0] < end:
                return None
            # Production round classification supplies this boundary. The
            # additional attack follows ordinary casualties; losses after its
            # roll cannot determine the earlier Heroic cancellation amount.
            end = post[0]
            ledger_upper = end
    for record in records:
        if int(record['line']) >= end or record.get('op') != 'move':
            continue
        origin, dest = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
        if (origin[0] in ('region', 'stronghold') and dest[0] in ('region', 'stronghold')
                and region in (origin[1], dest[1]) and origin[1] != dest[1]):
            if any(p and p[0] == 'unit' and rx._unit_side_str(p[1]) == side
                   for p in (rr.parse_piece(t) for t in rx._group_tokens(record.get('obj', '')))):
                return None
    normalized, _ = _regular_group_evidence(batch, side)
    try:
        losses = rx._casualty_gesture_operations(normalized, side, {region}, lower + 1, ledger_upper)
        if any(line >= end for line, _, _ in losses):
            return None
    except rx._CasualtyGestureError:
        return None
    if any(kind not in ('loss_regular', 'reduce_elite') for _, _, kind in losses):
        return None
    # Each admitted operation independently costs one hit. Require surviving
    # capacity, so several cancellation amounts cannot yield the same wipe.
    hp = sum(2 if unit.kind.name == 'ELITE' else 1 for unit in army.units)
    if len(losses) >= hp:
        return None
    cancelled = incoming - len(losses)
    found = [index for index, opt in options
             if _norm(opt.get('target')) == target and opt.get('count') == cancelled]
    if cancelled <= 0 or len(found) != 1:
        return None
    return found[0], victim_line, 'heroic_closed_casualty_balance'



def _regular_group_evidence(batch, owner, *, elite_removals=False):
    """Preserve loss totals without treating token order as a human decision.

    Elite groups are admitted only for payment accounting with no replacement
    of an involved nation anywhere in the supplied window. The caller still
    checks replacement supply and exact hit cost for every removed Elite.
    """
    import rendered_replay as rr
    import rendered_extract as rx
    import copy
    result, ambiguous = [], set()
    for record in batch.get('records', ()):
        tokens = (record.get('pieces') or ()) if record.get('op') == 'kill' else rx._group_tokens(record.get('obj', ''))
        parsed = [rr.parse_piece(t) for t in tokens]
        own = [p for p in parsed if p and p[0] == 'unit' and rx._unit_side_str(p[1]) == owner]
        if len({(p[1], p[2]) for p in own}) <= 1:
            result.append(record)
            continue
        if any(p[2] != 'REGULAR' for p in own):
            nations = {p[1] for p in own if p[2] == 'ELITE'}
            replacements = any(
                later.get('op') == 'move'
                and int(later.get('line', 0)) > int(record['line'])
                and rr.zone_kind(later.get('from', ''))[0] in ('cas', 'reinf')
                and rr.zone_kind(later.get('to', ''))[0] in ('region', 'stronghold')
                and rr.zone_kind(later.get('to', ''))[1] == rr.zone_kind(record.get('zone', ''))[1]
                and any(p and p[0] == 'unit' and p[1] in nations and p[2] == 'REGULAR'
                        for p in (rr.parse_piece(t) for t in rx._group_tokens(later.get('obj', ''))))
                for later in batch.get('records', ()))
            if (not elite_removals or replacements or record.get('op') != 'kill'
                    or len(own) != len(parsed)):
                result.append(record)
                continue
        ambiguous.add(int(record['line']))
        for token in tokens:
            item = dict(record)
            if record.get('op') == 'kill':
                item['pieces'] = [token]
            else:
                item['obj'] = token
            result.append(item)
    copied = dict(batch)
    copied['records'] = result
    return copied, ambiguous


def _late_replacement(batch, owner, region, nation, upper, cleanup_upper):
    """A later replacement makes an apparent two-hit removal unproven."""
    import rendered_replay as rr
    import rendered_extract as rx
    for r in batch.get('records', ()):
        if r.get('op') != 'move' or not upper <= int(r.get('line', 0)) < (cleanup_upper or upper):
            continue
        origin, dest = rr.zone_kind(r.get('from', '')), rr.zone_kind(r.get('to', ''))
        if origin[0] not in ('cas', 'reinf') or dest[0] not in ('region', 'stronghold') or dest[1] != region:
            continue
        if any(p and p[0] == 'unit' and p[1] == nation and p[2] == 'REGULAR'
               for p in (rr.parse_piece(t) for t in rx._group_tokens(r.get('obj', '')))):
            return True
    return False

def _match_unit_sequence(state, batch, options, owner, region, lower, upper,
                         total, paid, count_choice=False, cleanup_upper=None, wipe_army=None):
    """Existing physical loss/replacement parser, with exact hit cost evidence.

    A missing replacement cannot distinguish an Elite's two-hit removal from
    a one-hit downgrade with empty supply. Such histories abstain.
    """
    import rendered_extract as rx
    import ai
    from game_state import Nation
    from combat import _cas_box_regulars
    normalized, ambiguous_lines = _regular_group_evidence(batch, owner, elite_removals=True)
    try:
        events = rx._casualty_gesture_operations(normalized, owner, {region}, lower + 1, upper,
                                                with_evidence_end=True, cleanup_hi=cleanup_upper)
    except rx._CasualtyGestureError as exc:
        return None, None, 'payment_' + str(exc)
    if not events:
        return None, None, 'missing_payment_gesture'
    sequence = []
    accumulated = 0
    for line, nation, kind, evidence_end in events:
        if total is not None and accumulated == total:
            break
        if kind == 'loss_regular':
            cost, operation, unitkind = 1, 'remove', 'regular'
        elif kind == 'reduce_elite':
            cost, operation, unitkind = 1, 'downgrade', 'elite'
        else:
            if _late_replacement(batch, owner, region, nation, upper, cleanup_upper):
                return None, None, 'replacement_after_payment_window'
            available = (_cas_box_regulars(state, Nation[nation]) > 0 or
                         ai.figures_available(state, Nation[nation], 'regular') > 0)
            if not available:
                return None, None, 'elite_shortage_hit_cost_ambiguous'
            cost, operation, unitkind = 2, 'remove', 'elite'
        sequence.append((evidence_end, {'nation': nation, 'kind': unitkind, 'operation': operation}, cost))
        accumulated += cost
    observed_total = sum(cost for _, _, cost in sequence)
    if wipe_army is not None:
        from collections import Counter
        remaining, consumed = [], 0
        for line, option, cost in sequence:
            if consumed >= paid:
                remaining.append(option)
            consumed += cost
        live = Counter((u.nation.name, u.kind.name.lower()) for u in wipe_army.units)
        removed = Counter((o['nation'], o['kind']) for o in remaining if o['operation'] == 'remove')
        if live != removed or any(o['operation'] != 'remove' for o in remaining):
            return None, None, 'no_roll_payment_does_not_prove_wipe'
    if count_choice:
        found = [i for i, opt in options if opt == {'count': observed_total}]
        return (found[0], sequence[-1][0], 'physical_self_hits') if len(found) == 1 else (None, None, 'self_hit_count_not_legal')
    if observed_total != total:
        return None, None, 'payment_total_mismatch'
    consumed = 0
    for line, option, cost in sequence:
        if consumed == paid:
            if line in ambiguous_lines:
                return None, None, 'payment_group_order_ambiguous'
            found = [i for i, opt in options if opt == option]
            return (found[0], line, 'physical_payment') if len(found) == 1 else (None, None, 'payment_not_legal')
        consumed += cost
    return None, None, 'payment_progress_mismatch'


def match_onslaught_payment(state, batch, legal, lower, upper, round_items=(),
                             self_loss_after_line=None, chats=()):
    import rendered_replay as rr
    pc = state.pending_combat or {}
    payment = pc.get('_onslaught_payment') or {}
    ec = pc.get('effect_choice') or {}
    owner = payment.get('owner', ec.get('owner'))
    if owner not in ('atk', 'def'):
        return None, None, 'missing_onslaught_payment'
    side = pc['attacker_side' if owner == 'atk' else 'defender_side']
    region = pc['attacker_region' if owner == 'atk' else 'defender_region']
    options = [(i, {k: v for k, v in a.params.items() if k != 'card_id'})
               for i, a in enumerate(legal) if a.action_type == 'combat_effect_choice']
    tag = 'A' if owner == 'atk' else 'D'
    post = [c for c, actor, phase in round_items if actor == tag and phase == 'post']
    completed = [int(c['line']) for c, _, phase in round_items if phase in ('main', 're')]
    choice_lower = max([int(lower)] + completed)
    choice_upper = min([int(upper)] + [int(c['line']) for c in post])
    named, named_line, named_error = _named_choice(state, 'onslaught', owner, round_items,
                                                 chats, choice_lower, choice_upper)
    if named_error:
        return None, None, named_error
    if named and named.get('decline') and (post or payment):
        return None, None, 'named_decline_conflicts_with_payment'
    if not payment and not post:
        completed_rolls = [int(chunk['line']) for chunk, _, phase in round_items
                           if phase in ('main', 're')]
        if completed_rolls:
            boundary = max([int(lower)] + completed_rolls)
            if self_loss_after_line is not None:
                boundary = max(boundary, int(self_loss_after_line))
            explicit = _explicit_effect_decline('onslaught', owner, options,
                round_items, chats, boundary, upper)
            if named and named.get('decline') and explicit is None:
                declines = [i for i, opt in options if opt == {'count': 0}]
                if len(declines) == 1:
                    explicit = (declines[0], named_line, 'explicit_effect_decline_chat')
            if named and not named.get('decline') and explicit is not None:
                return None, None, 'conflicting_named_choices'
            if explicit is not None:
                import rendered_extract as rx
                check_after = int(self_loss_after_line) if self_loss_after_line is not None else explicit[1]
                try:
                    losses = rx._casualty_gesture_operations(batch, side.name,
                        {rr.sanitize(region)}, check_after + 1, upper)
                except rx._CasualtyGestureError:
                    return None, None, 'onslaught_decline_loss_unproven'
                if losses:
                    return None, None, 'onslaught_decline_conflicts_with_payment'
                return explicit
    if len(post) != 1:
        return None, None, 'missing_unique_onslaught_roll'
    post_line = int(post[0]['line'])
    slot = 'current fp card' if side.name == 'FREE_PEOPLES' else 'current sa card'
    parks = [int(r['line']) for r in batch.get('records', ())
             if r.get('op') == 'move' and lower < int(r.get('line', 0)) < post_line
             and rr.zone_kind(r.get('from', ''))[1] == rr.sanitize(region)
             and str(r.get('to', '')).strip().lower() == slot]
    if self_loss_after_line is not None:
        # Only the driver which replayed all ordinary casualties may provide
        # this boundary. It is not inferred from a post-roll or a death alone.
        if type(self_loss_after_line) is not int or not lower <= self_loss_after_line < post_line:
            return None, None, 'onslaught_self_loss_boundary_invalid'
        boundary = self_loss_after_line
    elif parks:
        boundary = min(parks) - 1
    else:
        return None, None, 'onslaught_self_loss_boundary_unproven'
    total = len(post[0]['ints'])
    if named and named.get('count') != total:
        return None, None, 'named_count_conflicts_with_onslaught_roll'
    if payment and total != payment['count']:
        return None, None, 'onslaught_roll_payment_mismatch'
    result = _match_unit_sequence(state, batch, options, side.name, rr.sanitize(region),
        boundary, post_line, payment.get('count'), payment.get('paid', 0),
        count_choice=not payment, cleanup_upper=upper)
    if not payment and result[0] is not None and options[result[0]][1].get('count') != total:
        return None, None, 'onslaught_roll_self_hit_mismatch'
    return result


def match_unordered_onslaught_removal_group(state, batch, legal, lower, upper,
                                           round_items=(), self_loss_after_line=None):
    """A terminal atomic loss set, represented by equivalent legal positives.

    Regular removals cost one and proved full Elite removals cost two hits.
    No replacement, choice, attack or figure cleanup can intervene before the
    final payment. Emit only the initial alternative set; later canonical steps
    are replay, not labels. Missing replacement supply remains ambiguous.
    """
    from collections import Counter
    import rendered_replay as rr
    import rendered_extract as rx
    import ai
    pc = state.pending_combat or {}
    payment = pc.get('_onslaught_payment') or {}
    if not payment or int(payment.get('remaining', 0)) < 2:
        return None
    result = match_onslaught_payment(state, batch, legal, lower, upper, round_items,
                                     self_loss_after_line=self_loss_after_line)
    if result[2] != 'payment_group_order_ambiguous':
        return None
    owner = payment['owner']
    role, tag = ('attacker', 'A') if owner == 'atk' else ('defender', 'D')
    region = rr.sanitize(pc[role + '_region'])
    side = pc[role + '_side']
    post = [c for c, actor, phase in round_items if actor == tag and phase == 'post']
    if len(post) != 1:
        return None
    postline = int(post[0]['line'])
    groups = []
    for record in batch.get('records', ()):
        line = int(record.get('line', 0))
        zone = rr.zone_kind(record.get('zone', ''))
        if record.get('op') != 'kill' or not lower < line < postline or zone[1] != region:
            continue
        pieces = [rr.parse_piece(t) for t in record.get('pieces', ())]
        if not pieces or any(not p or p[0] != 'unit' or p[2] not in ('REGULAR', 'ELITE')
                             or rx._unit_side_str(p[1]) != side.name for p in pieces):
            continue
        counts = Counter()
        for p in pieces:
            counts[(p[1], p[2])] += int(p[3])
        if len(counts) > 1 and sum(count * (2 if kind == 'ELITE' else 1)
                                  for (_, kind), count in counts.items()) == payment['remaining']:
            groups.append((line, counts))
    if len(groups) != 1:
        return None
    line, counts = groups[0]
    from game_state import Nation
    from combat import _cas_box_regulars
    for nation, kind in counts:
        if kind == 'ELITE' and (
                _late_replacement(batch, side.name, region, nation, postline, upper)
                or not (_cas_box_regulars(state, Nation[nation]) > 0
                        or ai.figures_available(state, Nation[nation], 'regular') > 0)):
            return None
    # The group must be the final physical operation before the attack.
    if any(line < int(r.get('line', 0)) < postline for r in batch.get('records', ())):
        return None
    # RNG lives separately from physical records. Check both the segmented
    # round view and the complete batch so an unassigned random event cannot
    # be hidden by segmentation. Only the final Onslaught roll is permitted.
    if (any(line < int(chunk.get('line', 0)) < postline for chunk, _, _ in round_items)
            or any(line < int(chunk.get('line', 0)) < postline
                   for chunk in batch.get('rng', ()))):
        return None
    available = Counter((u.nation.name, u.kind.name)
                        for u in ai._alloc_side_units(state, pc[role + '_region'], side))
    if any(available[n] < count for n, count in counts.items()):
        return None
    options = []
    indices = []
    for nation, kind in sorted(counts):
        option = {'nation': nation, 'kind': kind.lower(), 'operation': 'remove'}
        found = [i for i,a in enumerate(legal) if a.action_type == 'combat_effect_choice'
                 and {k:v for k,v in a.params.items() if k != 'card_id'} == option]
        if len(found) != 1:
            return None
        indices.append(found[0])
        options.extend([option] * counts[(nation, kind)])
    return {'legal_indices': indices, 'canonical_options': options, 'line': line,
            'reason': 'unordered_commuting_payment_group',
            'physical_regular_losses': {nation: count for (nation, kind), count in sorted(counts.items())
                                        if kind == 'REGULAR'},
            'physical_unit_losses': [dict(nation=nation, kind=kind.lower(), count=count)
                                     for (nation, kind), count in sorted(counts.items())],
            'emit_intermediate_choices': False}


# Existing audit callers retain the original name; production uses the broader
# name because the proof now includes full Elite removals as well as Regulars.
match_unordered_onslaught_regular_group = match_unordered_onslaught_removal_group


# Corpus-only user convention. This never projects the live/self-play menu.
_WOP_CORPUS_POLICY = 'white_unless_anduril_aragorn_captain_hobbit_v1'


def _words_context(state, legal):
    import ai
    import combat
    from cards import get_card, is_hobbit_identity
    pc = state.pending_combat or {}
    ec = pc.get('effect_choice') or {}
    if (pc.get('phase') != 'combat_effect_choice' or ec.get('stage') != 'pre'
            or ec.get('effect') != 'words_of_power' or ec.get('owner') not in ('atk','def')
            or any(k not in pc for k in ('atk_combat_card','def_combat_card'))):
        raise ValueError('missing_live_words_context')
    owner = ec['owner']
    role = 'attacker' if owner == 'atk' else 'defender'
    if pc[role+'_side'] != ai.Side.SHADOW or pc.get('deciding_side') != ai.Side.SHADOW:
        raise ValueError('words_owner_not_shadow')
    pair = {k:pc[k] for k in ('atk_combat_card','def_combat_card')}
    if pair[owner+'_combat_card'] != ec.get('card_id'):
        raise ValueError('words_card_not_current')
    card = get_card(ec['card_id'])
    if not card or _norm(card.combat_name) != 'words of power':
        raise ValueError('words_card_identity')
    clone = ai.clone_state(state)
    battle = combat.setup_combat(clone, pc['attacker_region'],pc['defender_region'],
                                  is_siege=pc.get('is_siege',False),attacker_side=pc['attacker_side'])
    figures = {c.name:c for c in battle.companions_in_battle}
    targets=[]
    for action in legal:
        if action.action_type != 'combat_effect_choice' or action.params.get('card_id') != ec['card_id']:
            raise ValueError('non_words_menu')
        target=action.params.get('target')
        if target not in figures or set(action.params) != {'card_id','target'}:
            raise ValueError('nonparticipant_words_target')
        targets.append(target)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('ambiguous_words_menu')
    fp_owner='def' if owner=='atk' else 'atk'
    fp_card=get_card(pair[fp_owner+'_combat_card']) if pair[fp_owner+'_combat_card'] else None
    anduril=bool(fp_card and _norm(fp_card.combat_name)=='anduril')
    # Reconstruction leaves Companion.captain_of_the_west at its default.
    # The actual combat calculator uses these canonical names; participation
    # and side legality were already established by setup_combat/the menu.
    traits={name:{'captain':name in combat.CAPTAINS_OF_THE_WEST,
                  'hobbit':bool(is_hobbit_identity(name))} for name in sorted(targets)}
    binding={'turn':int(state.turn),'round':int(pc.get('rounds_fought',0)),'owner':owner,
             'cards':pair,'attacker_region':str(pc['attacker_region']),
             'defender_region':str(pc['defender_region']),'is_siege':bool(pc.get('is_siege',False)),
             'legal_targets':sorted(targets),'traits':traits,'fp_anduril':anduril}
    ranked=[]
    for name in targets:
        norm=_norm(name)
        rank=(0 if norm=='gandalf the white' and not anduril else
              1 if norm=='aragorn' else 2 if traits[name]['captain'] else
              3 if traits[name]['hobbit'] else None)
        if rank is not None:ranked.append((rank,norm,name))
    return binding, min(ranked)[2] if ranked else None


def _words_chat_target(text, targets, card=None):
    import re
    from combat_decision_text import parse_choice
    parsed = parse_choice(text, card=card)
    match=re.fullmatch(r'\s*cancel\s+(.+?)\s*[.!]?\s*',str(text),re.I)
    if parsed and parsed['effect'] == 'words_of_power' and parsed.get('target'):
        named = _norm(parsed['target'])
    elif match:
        named = _norm(match.group(1))
    else:
        return None
    matches=[name for name in targets if _norm(name)==named or
             (named=='gandalf' and _norm(name) in ('gandalf the white','gandalf the grey'))]
    return matches[0] if len(matches)==1 else None


def match_words_of_power(state, batch, legal, lower, upper, round_items=(), chats=()):
    """Observed owner cancellation wins; otherwise use the authorized BC prior."""
    try:binding,priority=_words_context(state,legal)
    except (ValueError,KeyError,TypeError) as exc:return None,None,str(exc)
    from cards import get_card
    card = get_card(binding['cards'][binding['owner']+'_combat_card'])
    lower,upper=int(lower),int(upper)
    rolls=[int(chunk['line']) for chunk,_,_ in round_items]
    if not rolls or not 0 <= lower < upper <= min(rolls):
        return None,None,'words_window_unbound'
    tag='A' if binding['owner']=='atk' else 'D'
    players={str(chunk['player']).strip().casefold() for chunk,actor,_ in round_items
             if actor==tag and chunk.get('player')}
    observed=[]
    if len(players)==1:
        for row in chats:
            if len(row)!=3 or not lower < int(row[0]) < upper or str(row[2]).strip().casefold() not in players:
                continue
            target=_words_chat_target(row[1],binding['legal_targets'],card)
            from combat_decision_text import parse_choice
            parsed = parse_choice(row[1],card=card)
            if parsed and parsed['effect'] == 'words_of_power' and target is None:
                return None,None,'words_observed_target_not_legal'
            if target is not None:observed.append((int(row[0]),str(row[1]),str(row[2]),target))
    if len({row[3] for row in observed})>1:return None,None,'words_conflicting_cancel_targets'
    target=observed[0][3] if observed else priority
    if target is None:return None,None,'words_priority_has_no_covered_target'
    index=next(i for i,a in enumerate(legal) if a.params['target']==target)
    evidence={'schema':'corpus_words_of_power_v1','policy_version':_WOP_CORPUS_POLICY,
              'effect':'words_of_power','binding':binding,'target':target,
              'action_card_id':legal[index].params['card_id'],'action_die_idx':legal[index].die_idx,
              'round':binding['round'],'owner':binding['owner'],
              'window_start':lower,'window_end':upper,'owner_players':sorted(players),
              'provenance':'observed' if observed else 'user_authorized_inferred',
              'reason':'observed_cancel_chat' if observed else 'user_authorized_wop_priority',
              'line':observed[0][0] if observed else lower,
              'chat':{'text':observed[0][1],'speaker':observed[0][2]} if observed else None}
    return index,evidence,evidence['reason']


def _words_admission_refusal(state,evidence,action):
    import ai
    refusal='combat_words_of_power_target_unproven'
    if (not isinstance(evidence,dict) or evidence.get('schema')!='corpus_words_of_power_v1'
            or evidence.get('policy_version')!=_WOP_CORPUS_POLICY or action is None):return refusal
    try:
        legal=ai.get_legal_actions(state,state.pending_combat['deciding_side'])
        binding,priority=_words_context(state,legal)
        if (evidence.get('binding')!=binding or evidence.get('round')!=binding['round']
                or evidence.get('owner')!=binding['owner'] or evidence.get('effect')!='words_of_power'
                or action.action_type!='combat_effect_choice'
                or action.params!={'card_id':evidence.get('action_card_id'),'target':evidence.get('target')}
                or action.die_idx!=evidence.get('action_die_idx')
                or not any(a.action_type==action.action_type and a.params==action.params and a.die_idx==action.die_idx for a in legal)):
            return refusal
        lo,hi,line=evidence['window_start'],evidence['window_end'],evidence['line']
        if any(type(n) is not int for n in (lo,hi,line)) or not 0<=lo<hi:return refusal
        if evidence.get('provenance')=='user_authorized_inferred':
            return None if (evidence.get('reason')=='user_authorized_wop_priority' and priority is not None
                            and evidence['target']==priority and evidence.get('chat') is None and line==lo) else refusal
        chat=evidence.get('chat') or {}
        players=evidence.get('owner_players') or []
        if (evidence.get('provenance')!='observed' or evidence.get('reason')!='observed_cancel_chat'
                or not lo<line<hi or len(players)!=1 or str(chat.get('speaker','')).strip().casefold()!=players[0]
                or _words_chat_target(chat.get('text',''),binding['legal_targets'],evidence['action_card_id'])!=evidence['target']):return refusal
        return None
    except (KeyError,TypeError,ValueError,AttributeError):return refusal
