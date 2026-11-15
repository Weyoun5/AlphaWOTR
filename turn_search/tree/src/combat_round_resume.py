"""Serializable combat-round stage cursor for live owner decisions.

Advances to the next actual owner choice without rerolling previous phases.
The enclosing battle still owns physical writeback and once-only bookkeeping.
"""
from dataclasses import dataclass,field
import copy
import combat as cb

@dataclass
class RoundCursor:
    combat: object
    choices: dict = field(default_factory=dict)
    stage: int = 0
    pending: object = None
    selection: object = None
    observed_rolls: set = field(default_factory=set)
    observed_roll_sides: set = field(default_factory=set)
    paid_leadership_targets: list = field(default_factory=list)
    ra_remaining: object = None
    ra_paid: int = 0
    ra_attacker: object = None
    pre_attack_index: int = 0
    pre_remaining: object = None
    pre_victim_attacker: object = None
    pre_total: int = 0
    pre_paid: int = 0
    schema: str = 'combat_round_cursor_v1'

    @property
    def stage_name(self):
        names=('begin','preattacks','modifiers','dread','relentless','combat_rolls',
               'reroll_setup','mighty_attacker','mighty_defender','rerolls',
               'bonuses','initiative6_first','initiative6_second','finish')
        if self.stage==4 and self.ra_remaining:
            return 'relentless_payment'
        if self.stage==1 and self.pre_remaining:
            return 'precombat_payment'
        return names[self.stage] if self.stage<len(names) else 'complete'

def _card(c,attacker):
    card=c.attacker_card if attacker else c.defender_card
    cancelled=c.attacker_card_cancelled if attacker else c.defender_card_cancelled
    return None if cancelled else card
def _name(c,attacker):
    card=_card(c,attacker);return card.combat_name.lower() if card else ''
def _choose(cur,effect,attacker,options):
    descriptor={'stage':cur.stage,'effect':effect,'owner':'atk' if attacker else 'def','options':options}
    if cur.selection is None:return descriptor
    if cur.selection not in options:raise ValueError('Choice is not legal at the observed round stage')
    choice=cur.selection;cur.selection=None
    role='atk' if attacker else 'def'
    for key in ('effect_count','effect_target'):cur.choices.pop(role+'_'+key,None)
    if 'count' in choice:cur.choices[role+'_effect_count']=choice['count']
    if 'target' in choice:cur.choices[role+'_effect_target']=choice['target']
    return None

def _begin(cur):
    c=cur.combat;c.attacker_final_hits=c.defender_final_hits=0
    c._round_paid_leadership=[]
    c._round_nazgul_present = c.nazgul_in_battle
    cb.apply_cancel_effects(c);cb.apply_words_of_power(c,cur.choices)
def _pre_attacks(cur):
    c=cur.combat
    order=(False,True) if cb._defender_first_initiative(c) else (True,False)
    while cur.pre_attack_index<len(order):
        attacker=order[cur.pre_attack_index]
        role='attacker' if attacker else 'defender'
        victim='defender' if attacker else 'attacker'
        army=getattr(c,victim+'_army')
        if cur.pre_remaining is None:
            before=getattr(c,role+'_pre_combat_hits')
            cb._apply_pre_combat_for_side(c,attacker)
            cur.pre_remaining=getattr(c,role+'_pre_combat_hits')-before
            cur.pre_total=cur.pre_remaining;cur.pre_paid=0
            cur.pre_victim_attacker=not attacker
            cur.observed_rolls.add('pre_combat')
            cur.observed_roll_sides.add(('pre_combat',role))
        while cur.pre_remaining and army.units:
            pending,cost=_payment(cur,'pre_combat_casualty',not attacker,cur.pre_remaining)
            if pending:return pending
            cur.pre_remaining-=cost
            cur.pre_paid+=cost
        if not army.units:
            setattr(c,victim+'_destroyed',True);c.battle_over=True
            return
        cur.pre_attack_index+=1;cur.pre_remaining=None


def _payment(cur,effect,attacker,remaining):
    """One chosen casualty operation, shared by prior attacks and self-hits."""
    c=cur.combat;role='attacker' if attacker else 'defender';army=getattr(c,role+'_army')
    options=[]
    for unit in army.units:
        kind='elite' if unit.kind==cb.UnitKind.ELITE else 'regular'
        operations=['remove'] if kind=='regular' else ['downgrade']
        if kind=='elite' and remaining>=2:operations.append('remove')
        for operation in operations:
            option={'nation':unit.nation.name,'kind':kind,'operation':operation}
            if option not in options:options.append(option)
    if cur.selection is None:
        return ({'stage':cur.stage,'effect':effect,'owner':'atk' if attacker else 'def',
                 'options':options},0)
    option=cur.selection
    if option not in options:raise ValueError('Illegal observed casualty payment')
    cur.selection=None
    unit=next(u for u in army.units if u.nation.name==option['nation']
              and ('elite' if u.kind==cb.UnitKind.ELITE else 'regular')==option['kind'])
    dgc=cb._BATTLE_DG if cb._BATTLE_DG is not None else {}
    av0=None if cb._DG_AVAIL0 is not None else cb._capture_reg_baseline(cb._ACTIVE_STATE)
    army.units.remove(unit)
    cb._return_shadow_regular(unit,dgc)
    cost=2 if option['kind']=='elite' and option['operation']=='remove' else 1
    if option['operation']=='downgrade':
        if cb._consume_downgrade_source(cb._ACTIVE_STATE,unit.nation,dgc,av0):
            army.units.append(cb.Unit(unit.nation,cb.UnitKind.REGULAR))
            unit._downgraded=True
    getattr(c,role+'_casualties').append(unit)
    if (unit.kind==cb.UnitKind.ELITE and unit.nation==cb.Nation.ISENGARD
            and getattr(c,'_saruman_in_play',False)):
        setattr(c,role+'_leadership',max(0,getattr(c,role+'_leadership')-1))
    return None,cost
def _modifiers(cur):
    c=cur.combat
    cb.apply_dice_modifiers(c);cb.apply_dice_count_modifiers(c)

def _dread(cur):
    c=cur.combat
    for attacker in (True,False):
        if _name(c,attacker)=='dread and despair' and getattr(c,'_shine_companion',None) is None:
            # Almanac: the cost is mandatory and may exceed the FP strength.
            # The one-die floor limits the result, not legal human payments.
            maximum=max(0,int(c.nazgul_in_battle))
            if maximum==0:
                continue
            options=[{'count':n} for n in range(1,maximum+1)]
            pending=_choose(cur,'dread_and_despair',attacker,options)
            if pending:return pending
    cb.apply_dread_and_despair(c,cur.choices)
def _relentless(cur):
    c=cur.combat
    for attacker in (True,False):
        if _name(c,attacker)=='relentless assault':
            role='attacker' if attacker else 'defender'
            short='atk' if attacker else 'def'
            army=getattr(c,role+'_army')
            if cur.ra_remaining is None:
                options=[{'count':k} for k in cb.relentless_assault_choices(c,attacker)]
                pending=_choose(cur,'relentless_assault',attacker,options)
                if pending:return pending
                cur.ra_remaining=cur.choices[short+'_effect_count']
                cur.ra_attacker=attacker
            while cur.ra_remaining and army.units:
                pending,cost=_payment(cur,'relentless_assault_payment',attacker,cur.ra_remaining)
                if pending:return pending
                cur.ra_remaining-=cost;cur.ra_paid+=cost
                setattr(c,role+'_dice_mod',getattr(c,role+'_dice_mod')+cost)
            if not army.units:
                setattr(c,role+'_destroyed',True);c.battle_over=True

def _combat_roll(cur):
    c=cur.combat
    c._resume_atk_threshold=6 if c.is_siege or (c.city_defense and c.siege_round==1) else 5
    c._resume_def_threshold=5
    cb.apply_confusion_effect(c)
    _primary_side(cur,True)

def _primary_side(cur,attacker):
    c=cur.combat;role='attacker' if attacker else 'defender';short='atk' if attacker else 'def'
    if ('combat',role) in cur.observed_roll_sides:return
    cancelled=getattr(c,'_wop_cancelled',None);cancelled_atk=getattr(c,'_wop_cancelled_is_attacker',False)
    strength=cb.calc_combat_strength(getattr(c,role+'_army'),cancelled_captain=cancelled if cancelled_atk==attacker else None)-getattr(c,role+'_dice_reduction',0)
    rolls,hits=cb.combat_roll(strength,getattr(c,'_resume_'+short+'_threshold'),getattr(c,role+'_dice_mod'),getattr(c,role+'_dice_cap'),side=getattr(c,role+'_side'))
    setattr(c,role+'_combat_rolls',rolls)
    if getattr(c,role+'_confusion'):
        enemy='defender' if attacker else 'attacker';setattr(c,enemy+'_hits',getattr(c,enemy+'_hits')+rolls.count(1))
    setattr(c,role+'_hits',getattr(c,role+'_hits')+max(0,hits))
    cur.observed_rolls.add('combat')
    cur.observed_roll_sides.add(('combat',role))
def _reroll_setup(cur):
    cb.apply_reroll_cancellation(cur.combat);cb.apply_they_are_terrible(cur.combat)
def _mighty(cur,attacker):
    c=cur.combat;role='attacker' if attacker else 'defender';short='atk' if attacker else 'def'
    rolls=getattr(c,role+'_combat_rolls');threshold=getattr(c,'_resume_'+short+'_threshold');modifier=getattr(c,role+'_dice_mod')
    targets=cb.mighty_attack_payers(c,attacker,rolls,threshold,modifier)
    if targets:
        pending=_choose(cur,'mighty_attack',attacker,[{'target':t} for t in targets])
        if pending:return pending
    auto=cb.apply_mighty_attack(c,rolls,threshold,modifier,attacker,choices=cur.choices)
    cur.paid_leadership_targets=list(getattr(c,'_round_paid_leadership',[]))
    setattr(c,'_resume_'+short+'_auto',auto);setattr(c,role+'_hits',getattr(c,role+'_hits')+auto)
def _reroll_side(cur,role,short):
    c=cur.combat
    if ('reroll',role) in cur.observed_roll_sides:return
    hits=0
    if not getattr(c,role+'_reroll_cancelled') and getattr(c,role+'_leadership')>0:
        rolls,hits=cb.leader_reroll(getattr(c,role+'_combat_rolls'),getattr(c,'_resume_'+short+'_threshold'),getattr(c,role+'_leadership'),getattr(c,role+'_reroll_mod'),confused=getattr(c,role+'_confusion'),miss_modifier=getattr(c,role+'_dice_mod'),exclude_misses=getattr(c,'_resume_'+short+'_auto'),side=getattr(c,role+'_side'))
        setattr(c,role+'_rerolls',rolls);setattr(c,role+'_hits',getattr(c,role+'_hits')+hits)
    setattr(c,'_resume_'+short+'_reroll_hits',hits)
    cur.observed_rolls.add('reroll')
    cur.observed_roll_sides.add(('reroll',role))

def _defender_roll_and_mighty(cur):
    # User-selected sequential convention: finish the attacker's dice before
    # exposing any defender roll. Observed-side guards preserve exact resume.
    _reroll_side(cur,'attacker','atk')
    _primary_side(cur,False)
    return _mighty(cur,False)

def _rerolls(cur):
    for role,short in (('attacker','atk'),('defender','def')):
        _reroll_side(cur,role,short)
def _fateful(cur,attacker):
    c=cur.combat;rh=getattr(c,'_resume_'+('atk' if attacker else 'def')+'_reroll_hits')
    targets=cb.fateful_strike_targets(c,attacker,rh)
    if targets:
        pending=_choose(cur,'fateful_strike',attacker,[{'target':t} for t in targets])
        if pending:return pending
    cb.apply_fateful_strike(c,rh,attacker,choices=cur.choices)
def _blade(cur,attacker):
    c=cur.combat;rh=getattr(c,'_resume_'+('atk' if attacker else 'def')+'_reroll_hits')
    if _name(c,attacker)=='blade of westernesse' and rh:
        targets=cb.combat_effect_targets(c,attacker,rh)
        if targets:
            pending=_choose(cur,'blade_of_westernesse',attacker,[{'target':t} for t in targets])
            if pending:return pending
    cb.apply_blade_of_westernesse(c,rh,attacker,choices=cur.choices)
def _bonuses(cur):
    cb.apply_bonus_hits(cur.combat);cb.apply_casualty_prevention(cur.combat,cur.choices)
def _initiative6(cur,first):
    c=cur.combat;attacker=not cb._defender_first_initiative(c) if first else cb._defender_first_initiative(c)
    role='attacker' if attacker else 'defender';short='atk' if attacker else 'def'
    enemy='defender' if attacker else 'attacker';rh=getattr(c,'_resume_'+short+'_reroll_hits')
    if _name(c,attacker)=='fateful strike':
        return _fateful(cur,attacker)
    if _name(c,attacker)=='blade of westernesse':
        return _blade(cur,attacker)
    if _name(c,attacker)=='black breath' and rh and c._round_nazgul_present:
        targets=cb.combat_effect_targets(c,attacker,rh)
        if targets:
            pending=_choose(cur,'black_breath',attacker,[{'count':0}]+[{'target':t} for t in targets])
            if pending:return pending
    if _name(c,attacker)=='heroic death':
        hits=max(0,getattr(c,enemy+'_hits')+getattr(c,enemy+'_bonus_hits')-getattr(c,role+'_hits_cancelled'))
        options=[dict(count=n,**({'target':t} if t is not None else {})) for n,t in cb.heroic_death_choices(c,attacker,hits)]
        pending=_choose(cur,'heroic_death',attacker,options)
        if pending:return pending
    before=list(c.attacker_army.leaders)+list(c.defender_army.leaders)
    cb.apply_black_breath(c,rh,attacker,choices=cur.choices);cb._heroic_death_for_side(c,attacker,cur.choices)
    after=c.attacker_army.leaders+c.defender_army.leaders
    dead=getattr(c,'_round_card_fp_dead',None)
    if dead is None:dead=c._round_card_fp_dead=[]
    dead.extend(figure for figure in before
                if cb._combat_leader_side(figure)==cb.Side.FREE_PEOPLES
                and not any(figure is remaining for remaining in after))
def _finish(cur):
    c=cur.combat
    ah=max(0,c.attacker_hits+c.attacker_bonus_hits-c.defender_hits_cancelled)
    dh=max(0,c.defender_hits+c.defender_bonus_hits-c.attacker_hits_cancelled)
    if cb.FINAL_HITS_HOOK is not None:ah,dh=cb.FINAL_HITS_HOOK(c,ah,dh)
    c.attacker_final_hits=ah;c.defender_final_hits=dh
    if cur.choices.get('defer_casualties'):
        c.casualties_deferred=True;c.deferred_hits_on_defender=ah;c.deferred_hits_on_attacker=dh;c.battle_over=True
        return
    c.attacker_casualties.extend(cb.apply_casualties(c.attacker_army,dh,cur.choices.get('atk_casualty_policy','preserve_elites')))
    c.defender_casualties.extend(cb.apply_casualties(c.defender_army,ah,cur.choices.get('def_casualty_policy','preserve_elites')))
    c.attacker_destroyed=cb.is_army_destroyed(c.attacker_army);c.defender_destroyed=cb.is_army_destroyed(c.defender_army)
    c.battle_over=c.attacker_destroyed or c.defender_destroyed
    cb._apply_post_casualty_effects(c,cur.choices)

STAGES=(_begin,_pre_attacks,_modifiers,_dread,_relentless,_combat_roll,_reroll_setup,
        lambda c:_mighty(c,True),_defender_roll_and_mighty,_rerolls,
        _bonuses,
        lambda c:_initiative6(c,True),lambda c:_initiative6(c,False),_finish)

def advance(cur,decision=None):
    """Return next observed-state decision descriptor or None when complete."""
    if cur.schema!='combat_round_cursor_v1':raise ValueError('Unknown round cursor schema')
    if decision is not None:
        if cur.pending is None or decision not in cur.pending['options']:
            raise ValueError('No matching pending round choice')
        cur.selection=copy.deepcopy(decision);cur.pending=None
    elif cur.pending is not None:
        return copy.deepcopy(cur.pending)
    while cur.stage<len(STAGES):
        if cur.combat.battle_over:
            cur.stage=len(STAGES);return None
        pending=STAGES[cur.stage](cur)
        if pending is not None:
            cur.pending=pending;return copy.deepcopy(pending)
        cur.stage+=1
    return None
