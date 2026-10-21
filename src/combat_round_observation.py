"""Append-only public combat continuation features; no hidden or future rolls."""
import re
import numpy as np

OBSERVATION_CONTRACT = 'v52_observed_combat_round_v3'
# v8 preserves Help Unlooked For's attacking source in the existing source
# region lane. Width is unchanged; v7 training rows cannot recover that fact.
ACTION_CONTRACT = 'observed_combat_choices_v8'
STATE_TAIL_DIM = 131
ACTION_TAIL_DIM = 28
SOURCE_STATE_DIM = 4350
SOURCE_ACTION_DIM = 105
from army_cargo_state import TAIL_DIM as CARGO_STATE_TAIL_DIM
TARGET_STATE_DIM = SOURCE_STATE_DIM + STATE_TAIL_DIM + CARGO_STATE_TAIL_DIM
from card_cargo_observation import TAIL_DIM as CARGO_TAIL_DIM
TARGET_ACTION_DIM = SOURCE_ACTION_DIM + ACTION_TAIL_DIM + CARGO_TAIL_DIM
STAGES = ('begin', 'preattacks', 'modifiers', 'dread', 'relentless',
          'combat_rolls', 'reroll_setup', 'mighty_attacker', 'mighty_defender',
          'rerolls', 'bonuses', 'initiative6_first', 'initiative6_second',
          'finish', 'relentless_payment', 'complete')
EFFECTS = ('dread_and_despair', 'relentless_assault', 'mighty_attack',
           'fateful_strike', 'blade_of_westernesse', 'black_breath',
           'heroic_death', 'relentless_assault_payment')
COMPANIONS = ('Gandalf the Grey', 'Strider', 'Legolas', 'Gimli', 'Boromir',
              'Meriadoc', 'Peregrin', 'Aragorn', 'Gandalf the White', 'Gollum')
TARGETS = COMPANIONS + ('Witch King', 'Saruman', 'Mouth of Sauron', 'leader', 'nazgul')
# Nation identity uses four binary columns; all eight nations remain distinct.
PAYMENT_NATIONS = ('GONDOR', 'ROHAN', 'ELVES', 'DWARVES', 'THE_NORTH',
                   'ISENGARD', 'SAURON', 'SOUTHRON_EASTERLINGS')

def _normal(value):
    return re.sub('[^a-z0-9]', '', str(value).lower())

_TARGET_INDEX = {_normal(name): i for i, name in enumerate(TARGETS)}
for alias, name in (('merry', 'Meriadoc'), ('pippin', 'Peregrin'),
                    ('Meriadoc Brandybuck', 'Meriadoc'), ('Peregrin Took', 'Peregrin'),
                    ('Witch-king of Angmar', 'Witch King'), ('witchking', 'Witch King')):
    _TARGET_INDEX[_normal(alias)] = TARGETS.index(name)

def _cursor(state):
    pending = getattr(state, 'pending_combat', None) or {}
    payload = pending.get('_round_resume') or {}
    cursor = payload.get('cursor')
    if cursor is not None and getattr(cursor, 'schema', None) != 'combat_round_cursor_v1':
        raise ValueError('Unknown combat continuation schema')
    return cursor

def encode_state_tail(state, side=None):
    """Use attacker/defender coordinates, visible equally to both players."""
    out = np.zeros(STATE_TAIL_DIM, dtype=np.float32)
    cursor = _cursor(state)
    if cursor is None:
        pc = getattr(state, 'pending_combat', None) or {}
        effect = pc.get('effect_choice') or {}
        if effect.get('effect') == 'onslaught' and effect.get('stage') == 'post':
            out[125] = out[127] = 1
        elif effect.get('stage') == 'onslaught_payment':
            payment = pc.get('_onslaught_payment') or {}
            out[126] = out[127] = 1
            for index,key in ((128,'remaining'),(129,'paid'),(130,'count')):
                out[index] = float(payment[key]) / 10
        if out[127]:
            out[24 + ('atk', 'def').index(effect['owner'])] = 1
        return out
    c = cursor.combat
    name = cursor.stage_name
    if name == 'precombat_payment':
        out[120] = 1
    elif name not in STAGES:
        raise ValueError('Unknown combat continuation stage: ' + str(name))
    else:
        out[STAGES.index(name)] = 1
    pending = cursor.pending or {}
    if pending:
        if pending['effect'] == 'pre_combat_casualty':
            out[121] = 1
            out[122] = float(cursor.pre_remaining) / 10
            out[123 if cursor.pre_victim_attacker else 124] = 1
        else:
            out[16 + EFFECTS.index(pending['effect'])] = 1
        out[24 + ('atk', 'def').index(pending['owner'])] = 1
    observed = set(cursor.observed_rolls)
    if not observed <= {'pre_combat', 'combat', 'reroll'}:
        raise ValueError('Unknown observed combat roll phase')
    if 'reroll' in observed and 'combat' not in observed:
        raise ValueError('Noncausal observed combat roll phases')
    observed_sides = set(cursor.observed_roll_sides)
    if not observed_sides <= {(phase,role) for phase in ('pre_combat','combat','reroll') for role in ('attacker','defender')}:
        raise ValueError('Unknown observed combat roll owner')
    fields = ('pre_combat_rolls', 'combat_rolls', 'rerolls')
    for r, role in enumerate(('attacker', 'defender')):
        for phase, (visible, field) in enumerate(zip(('pre_combat', 'combat', 'reroll'), fields)):
            slot = r * 3 + phase
            if (visible,role) not in observed_sides:
                continue  # Future arrays are never inspected or encoded.
            out[26 + slot] = 1
            for die in getattr(c, role + '_' + field):
                if type(die) is not int or not 1 <= die <= 6:
                    raise ValueError('Invalid observed d6')
                out[32 + 6 * slot + die - 1] += .2
    attrs = ('combat_strength', 'leadership', 'dice_mod', 'reroll_mod',
             'hits', 'pre_combat_hits', 'bonus_hits', 'hits_cancelled', 'final_hits')
    for r, (role, short) in enumerate((('attacker', 'atk'), ('defender', 'def'))):
        for j, attr in enumerate(attrs):
            value = getattr(c, role + '_' + attr, 0)
            # Hit arithmetic is public only after its corresponding phase.
            if attr == 'pre_combat_hits' and 'pre_combat' not in observed:
                value = 0
            if attr in ('hits', 'bonus_hits', 'hits_cancelled', 'final_hits') and 'combat' not in observed:
                value = 0
            out[68 + r * 10 + j] = float(value) / 10
        if 'reroll' in observed:
            out[68 + r * 10 + 9] = float(getattr(c, '_resume_' + short + '_reroll_hits', 0)) / 5
    leaders = list(c.attacker_army.leaders) + list(c.defender_army.leaders)
    leaders += list(getattr(c, 'companions_in_battle', []))
    suppressed = getattr(c, '_wop_cancelled', None)
    shined = getattr(c, '_shined_companion', None)
    paid = {_TARGET_INDEX.get(_normal(x)) for x in cursor.paid_leadership_targets}
    forfeited = getattr(c, '_leadership_forfeited', set())
    for leader in leaders:
        idx = _TARGET_INDEX.get(_normal(getattr(leader, 'name', '')))
        if idx is None or idx >= len(COMPANIONS):
            continue
        out[88 + idx] = 1
        out[98 + idx] = float(leader is suppressed)
        out[108 + idx] = float(leader is shined or leader.name in forfeited or idx in paid)
    out[118] = float(getattr(cursor, 'ra_remaining', 0) or 0) / 10
    out[119] = float(getattr(cursor, 'ra_paid', 0)) / 10
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite public combat observation')
    return out

def encode_action_tail(action, state):
    out = np.zeros(ACTION_TAIL_DIM, dtype=np.float32)
    cursor = _cursor(state)
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    onslaught = effect.get('stage') == 'onslaught_payment' or (effect.get('stage') == 'post' and effect.get('effect') == 'onslaught')
    if (cursor is None and not onslaught) or action.action_type != 'combat_effect_choice':
        return out
    p = action.params or {}
    out[0] = 1
    target = p.get('target')
    out[1] = float(p.get('count') == 0 and not target)
    out[2] = float(p.get('count', 0)) / 10
    if target:
        name = _normal(target)
        if name not in _TARGET_INDEX:
            raise ValueError('Unknown combat target: ' + str(target))
        out[3 + _TARGET_INDEX[name]] = 1
    if 'nation' in p:
        nation = getattr(p['nation'], 'name', p['nation'])
        if nation == 'NORTH':
            nation = 'THE_NORTH'
        code = PAYMENT_NATIONS.index(nation) + 1
        # Four binary identity columns cover all eight nations exactly; two
        # additional reserved zero columns preserve this declared tail width.
        out[18:22] = [(code >> bit) & 1 for bit in range(4)]
    if 'kind' in p:
        out[24 + ('regular', 'elite').index(p['kind'])] = 1
    if 'operation' in p:
        out[26 + ('remove', 'downgrade').index(p['operation'])] = 1
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite combat action')
    return out
