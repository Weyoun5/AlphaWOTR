"""Player-facing decision language; no engine identifiers or menu indices."""

_LABELS = {
    'guide_change':'Fellowship Guide', 'hunt':'Hunt resolution',
    'reveal_place':'Fellowship reveal', 'discard':'hand-limit discard',
    'declare':'Fellowship declaration', 'eye':'Hunt allocation',
    'card_followup':'card effect', 'card_option':'card effect',
    'attacker_rearguard':'attacking army', 'defender_siege_choice':'stronghold defense',
    'attacker_continue':'continuing the battle', 'defender_retreat':'defending army',
    'victor_advance':'advance after battle', 'attacker_siege_decision':'siege assault',
    'combat_round_result':'continuing the siege', 'attacker_card_retreat':'retreat',
    'attacker_casualty_policy':'army casualties', 'defender_casualty_policy':'army casualties',
    'fp_shine':'Gandalf in battle', 'attacker_combat_card':'combat card',
    'defender_combat_card':'combat card', 'combat_card_reveal':'combat-card reveal',
    'casualty_allocation':'army casualties', 'siege_trim':'stronghold army limit',
    'combat_effect_choice':'combat-card effect', 'combat_setup':'battle setup',
    'combat card':'combat card', 'no open decision':'current action',
}


def decision_label(kind):
    key=str(kind or '').split(':')[-1].strip()
    return _LABELS.get(key,_LABELS.get(key.replace(' ','_'),'current action'))


def _enemy_figure_instructions(dp, state):
    """Name only targets in the current FP card's actual legal choice menu."""
    import game_state as gs
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    title = {'fateful_strike': 'Fateful Strike',
             'blade_of_westernesse': 'Blade of Westernesse'}.get(effect.get('effect'))
    cursor = (pc.get('_round_resume') or {}).get('cursor')
    owner = effect.get('owner')
    if (title is None or pc.get('phase') != 'combat_effect_choice'
            or str(dp.get('kind', '')).split(':')[-1] != 'combat_effect_choice'
            or effect.get('stage') != 'round' or owner not in ('atk', 'def')
            or dp.get('side') is not gs.Side.FREE_PEOPLES
            or pc.get('attacker_side' if owner == 'atk' else 'defender_side') is not gs.Side.FREE_PEOPLES
            or cursor is None):
        return None
    pending = getattr(cursor, 'pending', None) or {}
    if (any(pending.get(k) != effect.get(k) for k in ('effect', 'owner'))
            or pending.get('stage') != effect.get('round_stage')
            or getattr(cursor, 'stage', None) != effect.get('round_stage')):
        return None
    targets = []
    for action in dp.get('actions') or ():
        params = action.params or {}
        target = params.get('target')
        if (action.action_type != 'combat_effect_choice'
                or params.get('card_id') != effect.get('card_id')
                or not isinstance(target, str) or not target
                or {'target': target} not in pending.get('options', ())
                or {'target': target} not in effect.get('options', ())):
            return None
        targets.append('Nazgul' if target == 'nazgul' else target)
    if targets:
        cost = ('this figure loss is additional to the hits against the Shadow Army'
                if effect.get('effect') == 'fateful_strike' else
                'use one hit from your Leader re-roll to eliminate that Minion instead of applying that hit to the Shadow Army')
        return (title + ': name one eligible enemy figure to eliminate ('
                + ', '.join(dict.fromkeys(targets))
                + '); ' + cost)
    return None


def pending_instructions(dp,state=None):
    from .combat_payment_prompt import payment_context, instructions
    payment = payment_context(dp, state)
    if payment is not None:
        return instructions(payment)
    enemy_figure = _enemy_figure_instructions(dp, state)
    if enemy_figure is not None:
        return enemy_figure
    actions=dp.get('actions') or ()
    kinds={a.action_type for a in actions}
    kind=str(dp.get('kind','')).split(':')[-1].replace(' ','_')
    if kind=='siege_trim' or kinds=={'siege_trim'}:
        import game_logic
        pc=getattr(state,'pending_combat',None) or {}
        amounts={a.params.get('excess') for a in actions
                 if a.action_type=='siege_trim' and type(a.params.get('excess')) is int
                 and a.params['excess']>0}
        count=next(iter(amounts)) if len(amounts)==1 else None
        units=('the excess army units' if count is None else
               '%d excess army unit%s'%(count,'' if count==1 else 's'))
        side=dp.get('side') or pc.get('defender_side')
        destination={'FREE_PEOPLES':'FP Reinforcements','SHADOW':'SA Reinforcements'}.get(
            getattr(side,'name',side),'your reinforcements')
        region=pc.get('trim_region')
        location=' from %s'%region if isinstance(region,str) and region else ''
        return ('return %s alive to %s%s, leaving %d army units inside the stronghold; '
                'Elites return as Elites')%(units,destination,location,game_logic.MAX_SIEGE_UNITS)
    effect=((getattr(state,'pending_combat',None) or {}).get('effect_choice') or {})
    effect_name=effect.get('effect')
    if kind=='combat_effect_choice' and effect_name=='mighty_attack':
        targets=sorted({a.params['target'] for a in actions if isinstance(a.params.get('target'),str)})
        if targets:
            return ('Mighty Attack: temporarily move your chosen participating Companion to Current FP Card '
                    'or name the participating Companion ('+', '.join(targets)+'); '
                    'forfeit their leadership for this round before your Leader re-roll')
    if kind=='combat_effect_choice' and effect_name in ('dread_despair','dread_and_despair'):
        amounts=sorted({a.params['count'] for a in actions if type(a.params.get('count')) is int})
        if amounts:
            return ('Dread and Despair: say "Dread and Despair: forfeit %d Nazgul leadership", '
                    'or use another legal amount (%s); resolve this before combat rolls')%(
                        amounts[0],', '.join(map(str,amounts)))
    if kind=='combat_effect_choice' and effect_name=='relentless_assault':
        amounts=sorted({a.params['count'] for a in actions if type(a.params.get('count')) is int})
        if amounts:
            return ('Relentless Assault: state the number of hits you choose to take (%s) before combat rolls; '
                    'then remove or reduce the paying units on the board')%(', '.join(map(str,amounts)))
    if kind=='combat_effect_choice' and effect_name in ('words_of_power','black_breath'):
        targets=[a for a in actions if isinstance(a.params.get('target'),str)]
        if targets:
            title='Words of Power' if effect_name=='words_of_power' else 'Black Breath'
            instruction=title+(': name the Companion you choose ' if effect_name=='words_of_power'
                               else ': name the Companion or Leader you choose ')
            instruction+=('to cancel for this round before combat rolls' if effect_name=='words_of_power'
                          else 'as the casualty from the Leader re-roll')
            if any(a.params.get('count')==0 and not a.params.get('target')for a in actions):
                instruction+='; or say 0 to decline'
            return instruction
    if kind=='combat_effect_choice' and effect_name=='heroic_death':
        targets=[a for a in actions if isinstance(a.params.get('target'),str)]
        if targets:
            instruction=("Heroic Death: state the chosen figure's name followed by the amount of hits to cancel; "
                         'then remove that figure on the board before allocating the remaining hits')
            if any(a.params.get('count')==0 and not a.params.get('target')for a in actions):
                instruction+='; or say 0 to decline'
            return instruction
    if kind=='combat_effect_choice' and effect.get('stage')=='post' and effect.get('effect')=='onslaught':
        amounts=sorted({a.params['count'] for a in actions if type(a.params.get('count')) is int})
        if amounts:
            return ('say onslaught followed by the number of additional hits to take, from %d to %d '
                    '(0 declines); then remove or reduce the paying units on the board')%(amounts[0],amounts[-1])
    if 'casualty_allocate' in kinds or kind in ('casualty_allocation',
            'attacker_casualty_policy','defender_casualty_policy'):
        return ('remove the lost units on the board; a whole Elite takes 2 hits; '
                'to reduce an Elite for 1 hit, remove it and place its Regular replacement')
    if kinds.intersection(('combat_card_play','combat_card_none')) or kind.endswith('_combat_card'):
        return 'select your combat card in the client, or say nc to play no card'
    if kind=='combat_card_reveal':
        return 'reveal your selected combat card in the client before rolling'
    if 'victor_stay' in kinds or 'victor_advance' in kinds:
        return ('move any advancing units into the conquered region, or leave them where they are; '
                'then kill the action die to finish')
    words=[]
    for action,phrase in (
        ('siege_assault','say assault'),('siege_no_press','say stop'),
        ('siege_press','say press, then reduce the Elite you choose on the board'),
        ('combat_continue','say press'),('combat_cease','say stop'),
        ('combat_no_retreat','say field to fight in the field'),('retreat_into_siege','move your defenders into the stronghold or say siege'),
        ('combat_retreat','move your retreating army on the board'),
        ('victor_advance','move the advancing units into the conquered region'),
        ('victor_stay','say stand to remain in place'),
        ('shine','say shine'),('shine_none','say no shine'),
        ('hunt_take_corruption','move the corruption marker or say corruption'),
        ('hunt_sacrifice_guide','remove the Guide on the board or say guide'),
        ('hunt_sacrifice_random','say random'),
        ('hunt_gollum_reveal','say reveal'),('hunt_gollum_no_reveal','say no reveal')):
        if action in kinds:words.append(phrase)
    if words:return '; or '.join(words)
    if 'discard_card' in kinds or kind=='discard':
        return 'discard the excess cards from your hand in the client, or say discard followed by the card name'
    if kind=='guide_change':
        from .guide_completion import INSTRUCTION
        return INSTRUCTION
    if kind=='reveal_place':
        regions=list(dict.fromkeys(str(a.params['region']) for a in actions
                     if a.action_type=='fellowship_reveal_place' and a.params.get('region')))
        destination=('one of: '+', '.join(regions)) if regions else 'a legal revealed region'
        return ('place the Fellowship in '+destination+
                '; leave it revealed with progress 0. You can correct its location without flipping again')
    if kind=='declare':return 'place the Fellowship declaration on the board, or say fp ok to continue without declaring'
    if kind=='eye':return 'place Hunt eyes manually and say the total (0e, 1e, 2e, etc.); or say done to accept the current board and roll automatically'
    pco=getattr(state,'pending_card_option',None) or {}
    choices=[(a.params or {}).get('choices') or {} for a in actions if a.action_type=='card_option']
    if pco.get('card_id')=='SHADOW_Character_Flocks_of_Crebain':
        options=[]
        if any(c.get('crebain_discard') is False for c in choices):
            options.append('say keep flocks to leave Flocks of Crebain in play')
        if any(c.get('crebain_discard') is True for c in choices):
            options.append('say discard flocks to use its bonus for this Hunt')
        if options:return '; or '.join(options)
    if pco.get('card_id')=='__gandalf_draw__':
        return 'draw the available bonus card in the client, or say draw or decline'
    from .card_completion import needs_boundary
    if kind in ('card_option', 'card_followup') and needs_boundary(state):
        return ('finish the chosen group movement or recruitment on the board, then say done; '
                'keep the action die staged while any remaining card effects are resolved')
    if any(c.get('action')=='hunt' for c in choices):
        return 'say rolling for hunt, discarding card followed by its name, or none'
    if any(c.get('_naz_done') for c in choices):
        return 'move the Nazgul on the board, then say done'
    if any(c.get('_comp_done') or c.get('_ffm_done') for c in choices):
        return 'move the Companions on the board, then say done'
    if any(c.get('_ents_chain_done') for c in choices):
        return 'play another available Character Event card without using an action die, or say done to finish'
    if 'combat_effect_choice' in kinds:
        targeted=[a for a in actions if isinstance((a.params or {}).get('target'),str)]
        if targeted:
            if any(type((a.params or {}).get('count')) is int for a in targeted):
                instruction="state the chosen figure's name followed by the amount of hits to cancel"
            else:
                instruction='name the Companion or Leader you choose for this card effect'
            declines=[a for a in actions if a.params.get('count')==0 and not a.params.get('target')]
            if len(declines)==1:instruction+='; or say 0 to decline'
            return instruction
        if actions and all(type((a.params or {}).get('count')) is int for a in actions):
            if len({a.params['count'] for a in actions})==len(actions):
                return 'state the amount you choose for this combat-card effect, then show any required payment on the board'
        return ('the board does not yet identify the figure or payment for your combat card; '
                'show its required piece changes, or use its choice control in the AI panel')
    if kind=='hunt':return 'finish the Hunt damage or reveal on the board; say help for the supported Hunt commands'
    if kind in ('card_option','card_followup'):
        return ('finish the card effect on the board; if its choice has no visible piece or card change, '
                'use its choice control in the AI panel')
    return ('the board does not yet identify a complete legal action; finish the intended piece and card changes '
            'or undo the incomplete action; the AI panel retains the error details')


def action_description(action):
    kind=str(getattr(action,'action_type',''))
    p=getattr(action,'params',None) or {}
    fixed={'combat_card_none':'no combat card','combat_continue':'continue the battle',
        'combat_cease':'stop the attack','siege_assault':'assault the stronghold',
        'siege_press':'continue the siege','siege_no_press':'stop the siege assault',
        'combat_no_retreat':'fight in the field','retreat_into_siege':'withdraw into the stronghold',
        'victor_stay':'remain in place','hunt_take_corruption':'take corruption',
        'hunt_sacrifice_guide':'sacrifice the Guide','hunt_sacrifice_random':'a random companion casualty',
        'hunt_gollum_reveal':'reveal the Fellowship','hunt_gollum_no_reveal':'keep the Fellowship hidden',
        'shine':'Gandalf shines','shine_none':'Gandalf does not shine','pass':'pass'}
    if kind in fixed:return fixed[kind]
    if kind=='siege_trim':
        count=p.get('excess')
        units=('%d excess army unit%s'%(count,'' if count==1 else 's')
               if type(count) is int and count>0 else 'the excess army units')
        return 'return %s alive to reinforcements'%units
    if kind=='casualty_allocate':
        nation=p.get('nation');nation=str(getattr(nation,'value',nation) or 'army')
        operation={'remove_regular':'remove a %s Regular','remove_elite':'remove a %s Elite',
                   'reduce_elite':'reduce a %s Elite'}.get(p.get('operation'))
        if operation:return operation%nation
    if kind=='combat_effect_choice':
        target=p.get('target')
        if isinstance(target,str) and '_' not in target:
            return ('card effect on '+target+(' for %d hits'%p['count'] if type(p.get('count')) is int else ''))
        if type(p.get('count')) is int:return 'combat-card amount: %d'%p['count']
    if kind=='choose_guide':
        companion=p.get('companion')
        if isinstance(companion,str) and '_' not in companion:return 'appoint '+companion+' as Guide'
    if kind in ('combat_retreat','victor_advance'):
        destination=p.get('to_region') or p.get('to')
        if isinstance(destination,str) and '_' not in destination:
            return ('retreat to ' if kind=='combat_retreat' else 'advance into ')+destination
    if kind=='discard_card':return 'discard the chosen card'
    if kind=='combat_card_play':return 'play the selected combat card'
    if kind=='card_option':
        ch=p.get('choices') or {}
        if p.get('card_id')=='SHADOW_Character_Flocks_of_Crebain':
            if ch.get('crebain_discard') is True:return 'discard Flocks of Crebain for this Hunt'
            if ch.get('crebain_discard') is False:return 'keep Flocks of Crebain in play'
        if ch.get('gandalf_draw') is True:return 'take the bonus draw'
        if ch.get('gandalf_draw') is False:return 'decline the bonus draw'
        return 'the selected card effect'
    return 'the action you selected'
