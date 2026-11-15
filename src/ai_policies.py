"""User-selected AI conventions, independent of human game legality.

Project a COMPLETE legal menu. Never generate actions, mutate a state,
or spend resources. Runtime projection returns supplied Action objects,
never copies. The combat-victim and Mighty-payer projections keep ONE row,
chosen by menu index (the first of equal copies), so over `legal + supplied`
the supplied copy of the kept row is dropped: match a supplied action by
action_policy_key, not by object identity (live_server._ai_decide,
R17-LIVE-AIDECIDE). The other projections keep every copy of a kept row.
Supervised recruitment projection separately records both the literal human
action and its fixed-kind training target; replay still executes the human action.
Rule timing belongs to the engine.
"""
from dataclasses import dataclass
import os
from typing import Optional

RECRUIT_ELITE = 'recruit_elite_if_available'
BONUS_DRAW = 'bonus_draw_always_take'
COMBAT_VICTIM = 'combat_victim_priority'
MIGHTY_PAYER = 'mighty_payer_lowest_leadership'
DREAD_USEFUL = 'dread_no_excess_forfeit'
# Move the existing useful-count pruning out of human legality without changing
# the learned meaning of a count or invalidating the trained policy contract.
POLICY_VERSION_V52 = 'elite_bonus_draw_combat_victim_mighty_v3'
# D6P-POLICY-VERSION (D6 plan 4d / DP-F, 2026-10-23): the A2-05 Ring-twin
# projection (below) changed the AI menu projection, so the v53 generation
# carries its own version.  Like the observation/action contracts it is keyed
# by the encoding generation chosen once from WOTR_ACT_GEN at import (ai.py,
# action_schema.py and selfplay_actor_critic/contracts.py read the same
# variable; trace_decisions.current_learning_context refuses a skew).  Every
# other generation keeps the v3 stamp, so the deployed v52 nets, corpora and
# checkpoints load unchanged; under v52 the A2-05 dedupe still runs with the
# v3 stamp, as it has since 2026-10-21.
POLICY_VERSION_V53 = 'elite_bonus_draw_combat_victim_mighty_ring_twin_v4'


def policy_version_for_generation(generation):
    """AI policy contract stamped under one WOTR_ACT_GEN value."""
    return POLICY_VERSION_V53 if str(generation).strip().lower() == 'v53' else POLICY_VERSION_V52


POLICY_VERSION = policy_version_for_generation(os.environ.get('WOTR_ACT_GEN', ''))


def _dread_useful_actions(state, actions):
    """Prune wasteful AI payments without restricting human or corpus legality."""
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    if pc.get('phase') != 'combat_effect_choice' or effect.get('effect') != 'dread_and_despair':
        return None
    battle = getattr((pc.get('_round_resume') or {}).get('cursor'), 'combat', None)
    owner = effect.get('owner')
    if battle is None or owner not in ('atk', 'def'):
        return None
    from combat import calc_combat_strength
    enemy = battle.defender_army if owner == 'atk' else battle.attacker_army
    maximum = max(1, calc_combat_strength(enemy, cancelled_captain=getattr(battle, '_wop_cancelled', None)) - 1)
    if any(a.action_type != 'combat_effect_choice' or set(a.params) != {'card_id', 'count'}
           or a.params.get('card_id') != effect.get('card_id')
           or type(a.params.get('count')) is not int for a in actions):
        return None
    selected = tuple(a for a in actions if 1 <= a.params['count'] <= maximum)
    return selected or None


def _mighty_payer_priority(state, actions):
    """Pay the lowest leadership among the already legal Companion targets.

    Playing Mighty Attack commits its cost. This projection chooses its
    payer only; the complete human menu and the effect timing stay intact.
    """
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    if pc.get('phase') != 'combat_effect_choice' or effect.get('effect') != 'mighty_attack':
        return None
    cursor = (pc.get('_round_resume') or {}).get('cursor')
    battle = getattr(cursor, 'combat', None)
    figures = {c.name: c for c in getattr(battle, 'companions_in_battle', ())}
    candidates = []
    for index, action in enumerate(actions):
        target = (action.params or {}).get('target')
        if action.action_type != 'combat_effect_choice' or target not in figures:
            # Do not collapse a future menu containing a separate use/decline.
            return None
        figure = figures[target]
        name = str(target).strip().casefold()
        candidates.append((int(figure.leadership), name == 'aragorn', name, index))
    return min(candidates)[-1] if candidates else None


def _combat_victim_priority(state, actions):
    """Select only among legal, already-timed victim choices."""
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    if (pc.get('phase') != 'combat_effect_choice'
            or effect.get('effect') not in ('blade_of_westernesse', 'fateful_strike')):
        return None
    ranks = {'witchking': 0, 'mouthofsauron': 1, 'saruman': 2, 'nazgul': 3}
    candidates = []
    for index, action in enumerate(actions):
        target = ''.join(c for c in str((action.params or {}).get('target', '')).lower()
                         if c.isalnum())
        if action.action_type == 'combat_effect_choice' and target in ranks:
            candidates.append((ranks[target], index))
    return min(candidates)[1] if candidates else None


@dataclass(frozen=True)
class PolicyProjection:
    actions: tuple
    forced_reason: Optional[str] = None
    reasons: tuple = ()


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    return value


def _recruit_choice(choices):
    """Remove only known recruitment-kind fields; retain all other choices.

    Equal-card, equal-location/nation/count/mode actions form one group.
    Shared-pool alternatives with equal maximal Elite counts remain choices
    (which location receives the scarce Elite is not fixed by this policy).
    """
    if not isinstance(choices, dict):
        return None
    rest, score, found = {}, 0, False
    for key, value in choices.items():
        if (key == 'unit_type' or key.endswith('_unit_type')) \
                and value in ('regular', 'elite'):
            rest[key] = '<regular-or-elite>'
            score += int(value == 'elite')
            found = True
        elif (key == 'unit_combo' or key.endswith('_unit_combo')) \
                and value in ('2regular', 'mixed', '2elite'):
            rest[key] = '<two-recruits>'
            score += {'2regular': 0, 'mixed': 1, '2elite': 2}[value]
            found = True
        else:
            rest[key] = value
    return (_freeze(rest), score) if found else None


def action_policy_key(action):
    """Semantic identity for validating supplied actions against a full menu.

    Optional UI labels are not game choices. Source/deck remain significant
    for synthetic Gandalf actions, and all execution-bearing fields remain.
    """
    params = dict(action.params or {})
    params.pop('card_name', None)
    if 'choices' in params:
        params.pop('idx', None)
    return (action.action_type, action.die_idx, _freeze(params))


# A2-05 (fix wave 2026-10-21): Elven/Dwarven Ring rows are generated once per
# unused die, because the live bridge binds the human's physical die by that
# index (human legality keeps every row).  The dice of one pool are
# interchangeable and encode_action leaves the die index unencoded (col 22
# dead), so rows differing ONLY in that index are one choice: repeated, each
# copy got the same logit (the group's prior times the face multiplicity) and
# search split visits and candidate slots across the copies.  The AI keeps the
# lowest die index offered per group - the face dedupe every other die-funded
# family already gets in get_legal_actions.  Lowest index, not first seen, so
# `legal + supplied` copies of the kept row both survive on this path (the
# combat-victim and Mighty-payer projections return before it and keep ONE row
# by menu index; see the module docstring); a group with one row is untouched.
# Mirrored in both native ai_policy_projection.inc trees.
_RING_DIE_FIELD = {'elven_ring': 'target_die_idx', 'dwarven_ring': 'die_idx'}


def ring_twin_key(action):
    """``(group_key, die_index)`` for a die-targeted Ring row, else None."""
    field = _RING_DIE_FIELD.get(action.action_type)
    if field is None:
        return None
    params = action.params or {}
    die = params.get(field)
    if type(die) is not int:
        return None
    rest = {k: v for k, v in params.items() if k != field}
    return (action.action_type, action.die_idx, _freeze(rest)), die


def _ring_twin_lowest(actions):
    lowest = {}
    for action in actions:
        twin = ring_twin_key(action)
        if twin is not None and (twin[0] not in lowest or twin[1] < lowest[twin[0]]):
            lowest[twin[0]] = twin[1]
    return lowest


def ring_twin_index(legal_actions, human_index, projection):
    """Menu index of the RETAINED twin of a Ring row the projection dropped.

    The twin has the same faces on an interchangeable die and the same
    encoding, so it is the same choice (no recruit-style provenance). None
    when the row is not a dropped Ring twin.
    """
    actions = tuple(legal_actions)
    twin = ring_twin_key(actions[human_index])
    if twin is None:
        return None
    retained_ids = {id(action) for action in projection.actions}
    if id(actions[human_index]) in retained_ids:
        return None
    for index, action in enumerate(actions):
        other = ring_twin_key(action)
        if id(action) in retained_ids and other is not None and other[0] == twin[0]:
            return index
    return None


def project_recruitment_target(legal_actions, human_index, projection):
    """Preserve independent card targets under the user's fixed recruit kind.

    Returns ``(original_menu_index, provenance)`` only for a unique retained
    card-option differing solely in recognized Regular/Elite recruit fields.
    Never map ordinary Muster actions: their point costs and continuations
    differ. The caller must apply the original human action to replay state.
    """
    import copy
    from enum import Enum
    actions = tuple(legal_actions)
    source = actions[human_index]
    retained_ids = {id(action) for action in projection.actions}
    if (source.action_type != 'card_option' or projection.forced_reason is not None
            or len(projection.actions) < 2 or id(source) in retained_ids):
        return None
    source_recruit = _recruit_choice((source.params or {}).get('choices'))
    if source_recruit is None:
        return None

    def key(action):
        params = dict(action.params or {})
        recruit = _recruit_choice(params.pop('choices', None))
        if recruit is None:
            return None
        params.pop('idx', None)
        params.pop('card_name', None)
        return (action.action_type, action.die_idx, _freeze(params), recruit[0])

    wanted = key(source)
    matches = [(i, action) for i, action in enumerate(actions)
               if id(action) in retained_ids
               and key(action) == wanted
               and _recruit_choice(action.params.get('choices'))[1] > source_recruit[1]]
    if len(matches) != 1:
        return None
    target_index, target = matches[0]

    def plain(value):
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return copy.deepcopy(value)

    def record(action):
        return {'action_type': action.action_type, 'die_idx': action.die_idx,
                'params': plain(action.params or {})}

    return target_index, {
        'schema': 'fixed_recruit_target_projection_v1',
        'transformation': 'maximum_elites_same_non_kind_choices',
        'source_human_action': record(source),
        'projected_action': record(target),
        'literal_human_kind_claimed': False,
        'replay_action': 'source_human_action',
    }


def project_ai_actions(state, side, legal_actions):
    """Return the AI subset of a complete legal menu, with forced provenance.

    A forced reason denotes a pending transition with no remaining choice,
    not an ordinary one-action turn. Such transitions still execute and must
    remain in traces, but do not produce learnable policy/value samples.
    Ordinary muster prefers an Elite at the SAME nation/region/die even
    though this spends both points. A one-point follow-up cannot recruit an
    Elite. Leaders and Saruman's different recruitment/upgrade modes remain.
    """
    actions = tuple(legal_actions)
    dread = _dread_useful_actions(state, actions)
    if dread is not None:
        return PolicyProjection(dread, None, (DREAD_USEFUL,))
    victim = _combat_victim_priority(state, actions)
    if victim is not None:
        return PolicyProjection((actions[victim],), COMBAT_VICTIM, (COMBAT_VICTIM,))
    payer = _mighty_payer_priority(state, actions)
    if payer is not None:
        return PolicyProjection((actions[payer],), MIGHTY_PAYER, (MIGHTY_PAYER,))
    keep = [True] * len(actions)
    families = {}
    tags = {}
    elite_musters = set()
    for action in actions:
        p = action.params or {}
        if action.action_type == 'muster_elite' and not p.get('saruman'):
            elite_musters.add((action.die_idx, p.get('nation'), p.get('region')))
    for i, action in enumerate(actions):
        p = action.params or {}
        if action.action_type == 'muster_recruit' \
                and p.get('type') == 'regular' and not p.get('saruman'):
            if (action.die_idx, p.get('nation'), p.get('region')) in elite_musters:
                keep[i] = False
                tags[i] = RECRUIT_ELITE
        if action.action_type in ('card_option', 'play_event'):
            recruit = _recruit_choice(p.get('choices'))
            if recruit is not None:
                rest, score = recruit
                # Preserve payment/card/mode identity; display name is optional.
                outer = {k: v for k, v in p.items()
                         if k not in ('choices', 'card_name', 'side', 'idx',
                                      'source_card_id', 'deck')}
                key = (action.action_type, action.die_idx, _freeze(outer), rest)
                families.setdefault(key, []).append((i, score))
                tags[i] = RECRUIT_ELITE
    for members in families.values():
        best = max(score for _, score in members)
        for i, score in members:
            if score < best:
                keep[i] = False
    # A2-05: one row per Ring twin group (see _RING_DIE_FIELD).
    ring_lowest = _ring_twin_lowest(actions)
    for i, action in enumerate(actions):
        twin = ring_twin_key(action)
        if twin is not None and twin[1] != ring_lowest[twin[0]]:
            keep[i] = False

    pending = getattr(state, 'pending_card_option', None) or {}
    draw_rows = []
    if pending.get('gandalf_draw'):
        for i, action in enumerate(actions):
            p = action.params or {}
            choice = p.get('choices') or {}
            if action.action_type == 'card_option' \
                    and p.get('card_id') == '__gandalf_draw__' \
                    and type(choice.get('gandalf_draw')) is bool:
                draw_rows.append((i, choice['gandalf_draw']))
        if any(take for _, take in draw_rows):
            for i, take in draw_rows:
                keep[i] = take
                tags[i] = BONUS_DRAW
    elif pending.get('palantir_draw'):
        # Palantir already requires a draw. Preserve all available deck choices.
        for i, action in enumerate(actions):
            if action.action_type == 'card_option' \
                    and 'palantir_deck' in (action.params.get('choices') or {}):
                tags[i] = BONUS_DRAW

    selected = tuple(a for i, a in enumerate(actions) if keep[i])
    reasons = tuple(sorted(set(tags.values())))
    forced = None
    if len(selected) == 1 and selected[0].action_type == 'card_option':
        idx = next(i for i, flag in enumerate(keep) if flag)
        forced = tags.get(idx)
    return PolicyProjection(selected, forced, reasons)
