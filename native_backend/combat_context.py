"""Refuse native combat when Python-only forcing/context is active.

This gate never executes gameplay. Expected code comes from the pinned source,
not from whichever callable happens to be installed when first invoked.
"""
from pathlib import Path
import tokenize
import types
import sys
import bridge

_compiled={}
_module_codes={}
_bindings={}
_runtime=None
_modules=None


def _codes(module):
    # The reference loader already pins source content. Keep canonicalization
    # at first use of a module/path, rather than issuing Windows filesystem
    # queries on every resident action. Callable/code/default checks remain live.
    key=(module,module.__file__)
    cached=_module_codes.get(key)
    if cached is not None:return cached
    path=Path(module.__file__).resolve()
    if path not in _compiled:
        with tokenize.open(path) as handle:source=handle.read()
        code=compile(source,str(path),'exec',dont_inherit=True)
        _compiled[path]=tuple(value for value in code.co_consts if isinstance(value,types.CodeType))
    result=_compiled[path];_module_codes[key]=result
    return result


def _function(module,name,defaults,kwdefaults=None):
    function=getattr(module,name,None);key=(module.__name__,name)
    if key in _bindings:
        original,code=_bindings[key]
        valid=function is original and function.__code__ is code
    else:
        valid=(type(function) is types.FunctionType and function.__globals__ is vars(module)
               and any(function.__code__==code and function.__code__.co_filename==code.co_filename for code in _codes(module) if code.co_name==name))
        if valid:_bindings[key]=(function,function.__code__)
    valid=valid and function.__defaults__==defaults and function.__kwdefaults__==kwdefaults
    if not valid:raise RuntimeError('Native combat refuses active Python override: '+module.__name__+'.'+name)
    return function


def _class_method(module,class_name,name):
    """Pin state-container methods used by the original post-card operation."""
    owner=getattr(module,class_name,None);function=getattr(owner,name,None)
    key=(module.__name__,class_name+'.'+name)
    if key in _bindings:
        original,code=_bindings[key]
        valid=function is original and function.__code__ is code
    else:
        classes=[code for code in _codes(module) if code.co_name==class_name]
        methods=[method for code in classes for method in code.co_consts if isinstance(method,types.CodeType) and method.co_name==name]
        valid=(type(function) is types.FunctionType and function.__globals__ is vars(module)
               and any(function.__code__==code and function.__code__.co_filename==code.co_filename for code in methods))
        if valid:_bindings[key]=(function,function.__code__)
    if not valid or function.__defaults__ is not None or function.__kwdefaults__ is not None:
        raise RuntimeError('Native combat refuses overridden '+module.__name__+'.'+class_name+'.'+name)


def require_native_context():
    """Allow only ordinary explicitly-owned RNG, outside Python forcing scopes."""
    global _runtime,_modules
    if _runtime is None:
        runtime=bridge.load_reference()
        import combat,combat_round_resume,random,game_logic,game_state,persistent_effects,army_participants,cards
        _modules=(combat,combat_round_resume,random,game_logic,game_state,persistent_effects,army_participants,cards)
        _runtime=runtime
    if sys.modules.get('ai') is not _runtime or any(
            sys.modules.get(name) is not module for name,module in zip(('combat','combat_round_resume','random','game_logic','game_state','persistent_effects','army_participants','cards'),_modules)):
        raise RuntimeError('Native combat refuses replaced reference modules')
    import combat as cb
    import combat_round_resume as resume
    import random
    for name in ('FINAL_HITS_HOOK','_ACTIVE_STATE','_BATTLE_DG','_DG_AVAIL0','_WIPE_CARD_CONTEXT'):
        if getattr(cb,name,None) is not None:
            raise RuntimeError('Native combat requires explicit native context for '+name)
    functions={
        'roll_d6_batch':('',None),'combat_roll':(0,None,None),
        'leader_reroll':(0,False,None,0,None),'calc_combat_strength':(None,None),
        'side_of_army':None,'side_of_nation':None,'_combat_leader_side':None,
        'apply_confusion_effect':None,'apply_reroll_cancellation':None,'apply_they_are_terrible':None,
        'mighty_attack_payers':None,'apply_mighty_attack':(None,),'_pick_named_companion':None,
        '_get_card_initiative':None,'_defender_first_initiative':None,
        'apply_bonus_hits':None,'_apply_bonus_hits_for_side':None,'apply_casualty_prevention':(None,),
        'fateful_strike_targets':None,'combat_effect_targets':(None,),'heroic_death_choices':None,
        'apply_fateful_strike':(None,),'apply_blade_of_westernesse':(None,),'apply_black_breath':(None,),
        '_heroic_death_sacrifice':(None,),'_heroic_death_for_side':(None,),
        '_card_kill_minion_bookkeeping':None,'_handle_eliminated_minions':None,
        'apply_cancel_effects':None,'apply_words_of_power':(None,),
        'apply_dice_modifiers':None,'apply_dice_count_modifiers':None,'_companions_present':None,
        'apply_dread_and_despair':(None,),'relentless_assault_choices':None,
        '_apply_pre_combat_for_side':None,'count_elite_units':None,
        '_consume_downgrade_source':(None,),'_downgrade_source_of':(None,),
        '_capture_reg_baseline':None,'_live_reg_avail':None,'_cas_box_regulars':None,'_dg_stamp':None,
        '_publish_round_casualties':None,'_publish_paused_combat':None,'_track_fp_casualties':None,
        'book_eliminated_characters':None,'_track_fp_leader_deaths':None,'_handle_eliminated_companions':None,
        'resume_deferred_post_casualty':(None,None,False,False,False,None,None),
        '_apply_post_casualty_for_side':(None,),'_apply_post_casualty_effects':(None,),
        '_write_back_side_units':None,'garrison_side_for':None,'_army_entry':None,'_battle_side_units':None,
        'classify_wipe_characters':(None,None),'wipe_spares_characters':(None,),
        'setup_combat':(None,None,False,None),'resume_combat':None,
        '_resolve_combat_body':(None,None),'_emit_round_annotations':None,
        '_check_settlement_capture':None,'_wipe_captures_in_place':None,
        'apply_onslaught_choice':('preserve_elites','preserve_elites',True,False),
        'apply_casualties':('preserve_elites',),'calc_leadership':(None,False),
        '_recompute_round_leadership':None,
        'resolve_combat':(None,None,False,1,False,None),
    }
    for name,defaults in functions.items():_function(cb,name,defaults)
    _function(cb,'eliminate_army_characters',(None,),{'card_name':None})
    _function(cb,'reconcile_destroyed_army',(None,),{'card_name':None,'army':None})
    for name in ('_combat_roll','_primary_side','_reroll_setup','_mighty','_rerolls','_reroll_side','_defender_roll_and_mighty','_choose','_card','_name',
                 '_fateful','_blade','_bonuses','_initiative6','_finish',
                 '_begin','_pre_attacks','_modifiers','_dread','_relentless','_payment'):_function(resume,name,None)
    import game_logic,game_state,persistent_effects
    import cards
    _function(cards,'get_card',None)
    _function(cards,'get_cards',None)
    _function(cards,'can_play_combat_card',(None,))
    _function(cards,'is_hobbit_identity',None)
    if cards._HOBBIT_IDENTITIES!=frozenset(('meriadoc','meriadoc brandybuck','peregrin','peregrin took',
            'frodo','frodo baggins','sam','samwise','sam gamgee','samwise gamgee')):
        raise RuntimeError('Native combat refuses modified Hobbit identities')
    _class_method(cards,'Card','to_combat_card')
    _function(game_logic,'_handle_saruman_elimination_discards',None)
    for name,defaults in (('_on_companion_casualty',(True,)),('check_companion_persistent_discards',None),
                          ('card_deck_of',None),('_wwsat_resolve',(None,)),('pick_random_card',('',))):_function(game_logic,name,defaults)
    for name in ('check_saruman_eliminated_discards','discard_table_card','check_companion_casualty_effect','can_fp_play_combat_card'):_function(persistent_effects,name,None)
    for name in ('get_by_type','remove_card','check_companion_requirements','get_fp_character_cards','has_card_type'):_class_method(game_state,'PersistentCardState',name)
    import army_participants
    for name,defaults in (('partition_battle_army',(None,)),('battle_army_view',(None,)),
                          ('_nation_side',None),('_leader_side',None),('_is_minion',None)):_function(army_participants,name,defaults)
    for name,defaults in (('freeze_attack',(None,)),('_at_war',None),('_rear_signature',None),('_saruman_present',None)):
        _function(army_participants,name,defaults)
    _function(army_participants,'select_forced_attack',None,{'require_leader':False})
    _function(game_logic,'battle',(None,None,False,1,False,None))
    _function(game_logic,'move_army',None,dict(move_units=None,move_elite=None,move_leaders=None,
        move_nazgul=None,side=None,nation_ok=None,combat_context=None,movement_roster=None,defer_stacking=False))
    for name in ('army_stack_cap','_check_political_activation_for_entry','_check_companion_activates_nation',
                 'remove_orphaned_leaders','leader_must_follow','_is_saruman','_leader_side',
                 'fellowship_region','is_fp_nation_region','get_or_create_army'):
        _function(game_logic,name,None)
    import board
    for name in ('region','adjacent'):_class_method(board,'Board',name)
    for name,defaults in (('_alloc_side_units',(None,None)),('_battle_army',(None,)),
                          ('_casualty_operations',None),('_apply_casualty_operation',None),
                          ('_forced_casualty_allocation',None),('_enter_casualty_allocation',None),
                          ('_complete_allocated_round',None),('_set_post_round_phase',None),
                          ('_finish_battle',None),('_open_deferred_onslaught',None),('_maybe_open_palantir_draw',None),
                          ('_run_combat_round',(None,)),('_maybe_shine_phase',None),('_maybe_casualty_policy_phase',None),
                          ('_maybe_combat_card_phase',None),('_accept_combat_round_result',None),('_resolve_wk_sorcerer',None),
                          ('combat_cards_unrevealed',None),('_maybe_combat_effect_choice_phase',None),('_combat_target_kind',None),
                          ('_pending_battle_wiped',None),('_finish_wiped_battle',None),
                          ('_commit_combat_card',(False,)),('_discard_played_combat_card',None),
                          ('_get_combat_card_actions',None),('_combat_card_check_ctx',None),('_combat_card_playable',None),
                          ('_extra_combat_card_ok',None),('_battle_map_companions',None),('_battle_has_shadow_minion',None),
                          ('_begin_siege',None),('_retreat_army',(None,)),('_register_move_overstack',None),('_defender_controls_stronghold',None),
                          ('_get_retreat_actions',None),('_settlement_controller',None),('_stronghold_controller',None),
                          ('_get_pending_combat_actions',None),
                          ('_count_deployed_all',None),('figures_available',(None,))):_function(_runtime,name,defaults)
    if _runtime.ALL_NATIONS!=list(game_state.Nation) or _runtime._CASUALTY_COST!={'remove_regular':1,'reduce_elite':1,'remove_elite':2}:
        raise RuntimeError('Native combat refuses modified casualty ordering/cost rules')
    if army_participants.FORMATION_VERSION!='forced_rearguard_v1':
        raise RuntimeError('Native combat refuses modified formation schema')
    if _runtime._WIPE_GUARD_PHASES!=('attacker_combat_card','defender_combat_card',
            'combat_round_result','attacker_continue','defender_retreat','defender_siege_choice','attacker_siege_decision'):
        raise RuntimeError('Native combat refuses modified stale-battle phases')
    import event_effects
    _function(event_effects,'_stack_cap',None)
    _function(event_effects,'resolve_overstacks',(True,))
    for name in ('grant_draw','settle_owed_draw','draw_event_card','open_hand_limit_discard','is_nazgul_figure'):_function(game_logic,name,None)
    _function(resume,'advance',(None,))
    stages=resume.STAGES
    if type(stages) is not tuple or len(stages)!=14 or stages[5] is not resume._combat_roll or stages[6] is not resume._reroll_setup or stages[8] is not resume._defender_roll_and_mighty or stages[9] is not resume._rerolls or stages[10] is not resume._bonuses or stages[13] is not resume._finish:
        raise RuntimeError('Native combat refuses modified Python stage schedule')
    if any(stages[index] is not getattr(resume,name) for index,name in enumerate(('_begin','_pre_attacks','_modifiers','_dread','_relentless'))):
        raise RuntimeError('Native combat refuses modified pre-combat stage schedule')
    lambdas=[code for code in _codes(resume) if code.co_name=='<lambda>']
    for index,attacker,helper in ((7,True,'_mighty'),(11,True,'_initiative6'),(12,False,'_initiative6')):
        function=stages[index]
        if (type(function) is not types.FunctionType or function.__globals__ is not vars(resume)
            or function.__defaults__ is not None or function.__kwdefaults__ is not None
            or function.__code__.co_consts!=(None,attacker)
            or function.__code__.co_names!=(helper,)
            or not any(function.__code__==code for code in lambdas)):
            raise RuntimeError('Native combat refuses modified Mighty stage dispatch')
    expected={'Gandalf the Grey','Strider','Legolas','Gimli','Boromir','Aragorn'}
    if type(cb.CAPTAINS_OF_THE_WEST) is not set or cb.CAPTAINS_OF_THE_WEST!=expected:
        raise RuntimeError('Native combat refuses modified Captain rules')
    # These native handles intentionally do not call a monkeypatched Python
    # Random method. Refuse such forcing instead of silently drawing afresh.
    if cb.random is not random or type(random._inst) is not random.Random:
        raise RuntimeError('Native combat refuses replaced Python random owner')
    import _random
    for name in ('random','getrandbits'):
        if getattr(random.Random,name) is not getattr(_random.Random,name):
            raise RuntimeError('Native combat refuses overridden Random.'+name)
    expected_class=next(code for code in _codes(random) if code.co_name=='Random')
    for name,code_name,defaults in (('randint','randint',None),('randrange','randrange',(None,1)),('choice','choice',None),
                                    ('_randbelow','_randbelow_with_getrandbits',None)):
        function=getattr(random.Random,name)
        expected=next(code for code in expected_class.co_consts if isinstance(code,types.CodeType) and code.co_name==code_name)
        if (type(function) is not types.FunctionType or function.__globals__ is not vars(random)
                or function.__code__!=expected or function.__defaults__!=defaults or function.__kwdefaults__ is not None):
            raise RuntimeError('Native combat refuses overridden Random.'+name)
    for name in ('randint','randrange','getrandbits','random','choice'):
        method=getattr(random,name)
        if getattr(method,'__self__',None) is not random._inst or getattr(method,'__func__',None) is not getattr(random.Random,name,None) and name in ('randint','randrange','choice'):
            raise RuntimeError('Native combat refuses overridden Python random.'+name)


def require_native_hunt_context():
    """Boundary guard for resident actions that can draw Hunt dice/tiles/victims."""
    require_native_context()
    import os,game_logic,random
    if game_logic.BALROG_TILE_GATE is not None:
        raise RuntimeError('Native Hunt requires explicit native context for BALROG_TILE_GATE')
    if os.environ.get('MORDOR_STOP_KEEPS_STEP'):
        raise RuntimeError('Native Hunt requires standard Mordor stop behavior')
    for name,defaults in (('roll_d6_batch',('',None)),('pick_hunt_tile',None),('pop_hunt_tile',None),('pick_random_companion',('',)),('pick_random_card',('',))):
        _function(game_logic,name,defaults)
    _class_method(random,'Random','shuffle')
    method=random.shuffle
    if getattr(method,'__self__',None) is not random._inst or getattr(method,'__func__',None) is not random.Random.shuffle:
        raise RuntimeError('Native Hunt refuses overridden Python random.shuffle')


def require_native_event_context():
    """Event RNG funnels may be externally forced independently of Hunt RNG."""
    require_native_hunt_context()
    import event_effects
    for name,defaults in (('_tile_choice',None),('_tile_pop',None),('_d6_batch',('card effect',))):
        _function(event_effects,name,defaults)


def require_native_bid_context():
    """Bid compensation retains the frozen draw funnel and reroll classes."""
    require_native_event_context()
    import game_logic as gl
    for name in ('has_action_token','can_use_dwarven_ring','grant_draw','settle_owed_draw'):
        _function(gl,name,None)
    _function(gl,'use_action_token',None,{'nation':None,'deck':'strategy'})
    _function(gl,'use_dwarven_ring',None,{'forced_face':None})
    expected={'CHARACTER':{'character'},'ARMY':{'army'},'MUSTER':{'muster'},'MUSTER_ARMY':{'muster','army'},'EVENT':{'event'},'EYE':{'eye'},'WILL_OF_WEST':{'will'}}
    actual={face.name:set(classes) for face,classes in gl._DWARVEN_FACE_CLASSES.items()}
    if actual!=expected:raise RuntimeError('Native bid action refuses modified reroll face classes')


def require_native_action_context():
    """Full action entry also owns clone, stable-point upkeep and trace rows."""
    require_native_hunt_context()
    for name in ('apply_action','_execute_action','_execute_action_inner','_dedupe_armies',
                 '_check_hand_limits','_recheck_last_battle','_offer_wwsat_discard',
                 '_maybe_open_gandalf_draw','open_reveal_placement'):
        _function(_runtime,name,None)
    import game_trace
    _function(game_trace,'emit_annotation',(None,))
    import onslaught_choices
    import combat
    _function(combat,'onslaught_max_sacrifice',None)
    for name in ('_roles','_options','_advance','begin','pay'):
        _function(onslaught_choices,name,None)
