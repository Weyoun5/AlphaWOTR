"""Small, cached chance features. No action execution, search or hand sampling.

Roll PMFs are components, not final damage / battle-win distributions. They
condition on current rosters and explicit modifiers, omit unknown cards and
unresolved choices, and expose those conditions in separate columns. In
particular pre-combat losses are NOT silently allocated to produce a forecast.
"""
from collections import Counter
from copy import copy
from functools import lru_cache

import numpy as np

from .probability_schema import INDEX as I, WIDTH


@lru_cache(maxsize=4096)
def dice_pmf(dice, hit_faces, reroll_faces, leadership, confused=False, observed=None,
             rerolled=None, converted=0):
    """Exact ordinary-hit component, including capped re-rolls of misses.

    observed = (normal hits, natural ones). Rerolled is an observed hit count.
    This deliberately excludes card conversions not already chosen/applied,
    cross-side Confusion damage, and post-roll bonuses/prevention.
    """
    from combat_dist import _binom_terms, _trinom_terms
    pmf = [0.] * 6
    if observed is None:
        outcomes = _trinom_terms(dice, hit_faces / 6., 1 / 6.)
    else:
        outcomes = [(observed, 1.)]
    for (hits, ones), probability in outcomes:
        misses = max(0, dice - hits - (ones if confused else 0) - converted)
        count = min(5, max(0, leadership), misses)
        rolls = _binom_terms(count, reroll_faces / 6.) if rerolled is None else [(rerolled, 1.)]
        for more, weight in rolls:
            pmf[hits + converted + more] += probability * weight
    return tuple(pmf)


@lru_cache(maxsize=256)
def plain_pmf(dice, faces):
    from combat_dist import _binom_terms
    out = [0.] * 6
    for hits, weight in _binom_terms(dice, faces / 6.):
        out[hits] = weight
    return tuple(out)


def _put(out, name, values):
    start = I[name]
    out[start:start + len(values)] = values


def _active_card(c, role):
    return None if getattr(c, role + "_card_cancelled") else getattr(c, role + "_card")


def _name(card):
    return card.combat_name.lower() if card else ""


def _detached(c):
    """Copy the small parameter object; never clone a GameState or apply losses."""
    c = copy(c)
    c._live_state = None
    # Deterministic appliers only change scalar fields and these marker sets.
    c._leadership_forfeited = set(getattr(c, "_leadership_forfeited", ()))
    c._round_paid_leadership = list(getattr(c, "_round_paid_leadership", ()))
    return c


class _CombatContext:
    """Read-only game view with a candidate's formation; no GameState clone."""
    def __init__(self, state, context):
        self._state = state
        self.pending_combat = context

    def __getattr__(self, name):
        return getattr(self._state, name)


def _base_combat(state, pc):
    import combat as cb
    view = state if getattr(state, "pending_combat", None) is pc else _CombatContext(state, pc)
    c = cb.setup_combat(view, pc["attacker_region"], pc["defender_region"],
        is_siege=pc.get("is_siege", False), attacker_side=pc["attacker_side"])
    c._live_state = None
    if pc.get("rounds_fought", 0):
        c.city_defense = False
    c.attacker_dice_reduction = pc.get("atk_dice_reduction", 0)
    c.defender_dice_reduction = pc.get("def_dice_reduction", 0)
    return c


def _cards_and_parameters(base, pc, action, side, cursor):
    import combat as cb
    from cards import get_card
    c = _detached(base)
    own = "attacker" if c.attacker_side == side else "defender"
    enemy = "defender" if own == "attacker" else "attacker"
    short = "atk" if own == "attacker" else "def"
    choosing_card = action.action_type in ("combat_card_play", "combat_card_none")
    unknown = choosing_card  # simultaneous cards: even a committed opposing card is private
    choices = dict(cursor.choices) if cursor else {k: pc[k] for k in
        ("atk_effect_count", "def_effect_count", "atk_effect_target", "def_effect_target")
        if pc.get(k) is not None}
    if choosing_card:
        c.attacker_card = c.defender_card = None
        if action.action_type == "combat_card_play":
            setattr(c, own + "_card", get_card(action.params["card_id"]).to_combat_card())
        choices = {}  # future effect choices must not leak from the pending payload
    elif cursor is None:
        # Effect choices occur after both commitments are revealed.
        for role, tag in (("attacker", "atk"), ("defender", "def")):
            cid = pc.get(tag + "_combat_card")
            setattr(c, role + "_card", get_card(cid).to_combat_card() if cid else None)
    if action.action_type == "combat_effect_choice":
        for field in ("count", "target"):
            if field in action.params:
                choices[short + "_effect_" + field] = action.params[field]
    stage = cursor.stage if cursor else -1
    # An observed pre-combat roll is not a settled roster: the owner may still
    # owe casualties before the ordinary roll. Candidate rows describe the
    # current army until the actual payment is made.
    unresolved = bool(cursor is not None and cursor.pre_remaining)
    if stage < 0:
        cb.apply_cancel_effects(c)
        # Words of Power has an owner target choice. Never invoke its fallback.
        wop = [(role, tag) for role, tag in (("attacker", "atk"), ("defender", "def"))
               if _name(_active_card(c, role)) == "words of power" and c.companions_in_battle]
        if all(choices.get(tag + "_effect_target") is not None for role, tag in wop):
            cb.apply_words_of_power(c, choices)
        else:
            unresolved = True
    if stage <= 2:
        cb.apply_dice_modifiers(c)
        cb.apply_dice_count_modifiers(c)
    for role, tag in (("attacker", "atk"), ("defender", "def")):
        name = _name(_active_card(c, role))
        if name == "dread and despair" and stage <= 3:
            if choices.get(tag + "_effect_count") is None:
                unresolved = True
            else:
                cb.apply_dread_and_despair(c, choices)
        if name == "relentless assault" and stage <= 4:
            # A count alone does not select the casualty composition. Show the
            # requested modifier on CURRENT rosters and mark the unresolved cost.
            count = choices.get(tag + "_effect_count")
            if cursor is not None and cursor.ra_remaining is not None:
                count = cursor.ra_remaining  # prior payments already modified the live cursor
            if count is not None:
                setattr(c, role + "_dice_mod", getattr(c, role + "_dice_mod") + count)
            unresolved = True
    if stage <= 5:
        cb.apply_confusion_effect(c)
    if stage <= 6:
        cb.apply_reroll_cancellation(c)
        cb.apply_they_are_terrible(c)
    return c, own, enemy, unknown, unresolved, choices


def _side_features(out, prefix, c, role, cursor, choices):
    import combat as cb
    from combat_dist import _hit_faces, _mk_side_params
    attacker = role == "attacker"
    short = "atk" if attacker else "def"
    card = _active_card(c, role)
    effect = card.combat_effect.lower() if card else ""
    cancelled = getattr(c, "_wop_cancelled", None) if getattr(c, "_wop_cancelled_is_attacker", False) == attacker else None
    dice = cb.calc_combat_strength(getattr(c, role + "_army"), cancelled_captain=cancelled)
    dice -= getattr(c, role + "_dice_reduction", 0)
    cap = getattr(c, role + "_dice_cap")
    dice = max(1, min(dice, cap) if cap is not None else dice)
    threshold = 6 if attacker and (c.is_siege or (c.city_defense and c.siege_round == 1)) else 5
    modifier = getattr(c, role + "_dice_mod")
    faces = _hit_faces(threshold, modifier)
    rr_faces = _hit_faces(threshold, getattr(c, role + "_reroll_mod"))
    leadership = 0 if getattr(c, role + "_reroll_cancelled") else getattr(c, role + "_leadership")
    observed, rerolled, converted, ones = None, None, 0, dice / 6.
    visible = cursor.observed_roll_sides if cursor else ()
    if ("combat", role) in visible:
        rolls = getattr(c, role + "_combat_rolls")
        dice = len(rolls)
        ones = rolls.count(1)
        hits = sum(r != 1 and (r == 6 or r + modifier >= threshold) for r in rolls)
        observed = (hits, ones)
        # This marker is public only after this side's conversion phase.
        conversion_done = cursor.stage > (7 if attacker else 8) or ("reroll", role) in visible
        if conversion_done:
            converted = getattr(c, "_resume_" + short + "_auto", 0)
    if ("reroll", role) in visible:
        rerolled = sum(r != 1 and (r == 6 or r + getattr(c, role + "_reroll_mod") >= threshold)
                      for r in getattr(c, role + "_rerolls"))
    confused = getattr(c, role + "_confusion")
    pmf = dice_pmf(dice, faces, rr_faces, leadership, confused, observed, rerolled, converted)
    values = dict(dice_over_5=dice / 5, leadership_over_10=leadership / 10,
        hit_probability=faces / 6, reroll_hit_probability=rr_faces / 6,
        roll_observed=float(observed is not None), confusion_self_hits_over_5=(ones / 5 if confused else 0),
        initiative_over_10=(card.combat_value / 10 if card else 0))
    # Descriptive components, not an allocation/target choice on the owner's behalf.
    params = _mk_side_params(c, attacker, choices, bool(c.attacker_confusion or c.defender_confusion))
    values["conversion_limit_over_2"] = params.ma_max / 2 if not params.ma_blocked else 0
    values["conversion_cost_over_5"] = params.ma_pay / 5 if not params.ma_blocked else 0
    if "score one additional hit" in effect and "more total hits" not in effect:
        values["bonus_per_success_over_2"] = .5
    if "score two additional hits" in effect:
        values["bonus_per_success_over_2"] = 1.
    later = ("after" in effect or params.ma_kind is not None or "retreat" in effect)
    values["later_effect"] = float(later)
    pre_dice, pre_threshold = 0, threshold
    if params.pre_kind == "ss":
        pre_dice = min(5, getattr(c, role + "_leadership"))
    elif params.pre_kind == "charge":
        pre_dice = min(5, cb.count_elite_units(getattr(c, role + "_army")))
    elif params.pre_kind == "db":
        pre_dice, pre_threshold = 3, 4
    pre = plain_pmf(pre_dice, _hit_faces(pre_threshold, 0))
    if ("pre_combat", role) in visible:
        # Use only actual observed faces, never cached future hit fields.
        hits = sum(r != 1 and (r == 6 or r >= pre_threshold) for r in getattr(c, role + "_pre_combat_rolls"))
        pre = tuple(float(i == hits) for i in range(6))
    for name, value in values.items():
        out[I[prefix + "_" + name]] = value
    _put(out, prefix + "_roll_hits_0", pmf)
    _put(out, prefix + "_pre_hits_0", pre)
    return effect, pmf, pre_dice > 0 and ("pre_combat", role) not in visible


def combat_features(base, pc, action, side, cursor=None):
    out = np.zeros(WIDTH, np.float32)
    out[I["combat_available"]] = 1
    c, own, enemy, unknown, unresolved, choices = _cards_and_parameters(base, pc, action, side, cursor)
    out[I["opponent_card_unknown"]] = unknown
    out[I["unresolved_choices"]] = unresolved
    results = []
    for prefix, role in (("own", own), ("enemy", enemy)):
        out[I[prefix + "_card_present"]] = getattr(c, role + "_card") is not None
        out[I[prefix + "_card_cancelled"]] = getattr(c, role + "_card_cancelled")
        results.append(_side_features(out, prefix, c, role, cursor, choices))
    out[I["current_roster_condition"]] = float(unresolved or any(item[2] for item in results))
    for i, prefix in enumerate(("own", "enemy")):
        if "two or more hits" in results[i][0] and "cancel one hit" in results[i][0]:
            out[I[prefix + "_shield_expected_cancel"]] = sum(results[1 - i][1][2:])
    return out


@lru_cache(maxsize=1024)
def _tile_components(bag, hits, mordor, eyes):
    damage = [0.] * 19  # -2..15, then >=16; raw values BEFORE mitigation/healing caps
    reveal = stop = 0.
    total = sum(count for spec, count in bag)
    if not total:
        damage[2] = 1.
        return tuple(damage), reveal, stop
    for (amount, eye, revealing, stopping, d6), count in bag:
        weight = count / total
        outcomes = ((eyes if mordor else hits, 1.),) if eye else (
            tuple((face, 1 / 6.) for face in range(1, 7)) if d6 and mordor else ((amount, 1.),))
        for value, probability in outcomes:
            damage[min(18, value + 2)] += weight * probability
        reveal += weight * revealing
        stop += weight * stopping
    return tuple(damage), reveal, stop


def hunt_features(ai, state):
    out = np.zeros(WIDTH, np.float32)
    mordor = bool(state.fellowship.mordor_entered)
    out[I["hunt_available"]] = out[I["raw_before_mitigation"]] = 1
    out[I["hunt_mordor"]] = mordor
    if mordor:
        hits = (1., 0., 0., 0., 0., 0.)  # no Hunt roll; draw probability is separately 1
    else:
        p = ai.game_logic.get_hunt_params(state)
        modifier = p["fp_modifier"] + getattr(state, "_crebain_hunt_bonus", 0)
        # Reuse the cached binomial/reroll component. Hunt has no natural-six
        # override, and the currently spent die is added AFTER this Hunt.
        faces = sum(r > 1 and r + modifier >= p["threshold"] for r in range(1, 7))
        hits = dice_pmf(p["num_dice"], faces, faces, len(p["reroll_sources"]))
    _put(out, "hunt_hits_0", hits)
    # Bag composition is public; order is never inspected or included in keys.
    pool = state.hunt_pool or [t for t in state.hunt_discard if not t.is_special]
    bag = tuple(sorted(Counter((t.damage, t.is_eye, t.reveal, t.stops_movement, t.is_dice_roll)
                               for t in pool).items()))
    out[I["tile_available"]] = bool(pool)
    distribution = np.zeros(19, np.float64)
    if not mordor:
        distribution[2] = hits[0]
    for successes, probability in (((0, 1.),) if mordor else enumerate(hits[1:], 1)):
        pmf, reveal, stop = _tile_components(bag, successes, mordor, len(state.hunt_box))
        distribution += probability * np.asarray(pmf)
        out[I["tile_reveal_icon_probability"]] += probability * reveal
        out[I["tile_stop_icon_probability"]] += probability * stop
    out[I["tile_draw_probability"]] = (1. if mordor else 1 - hits[0]) if pool else 0
    _put(out, "raw_damage_-2", distribution)
    return out


def encode_probabilities(ai, state, side, actions):
    """One row per legal candidate. No state cloning, RNG, network, or transition."""
    out = np.zeros((len(actions), WIDTH), np.float32)
    pc = getattr(state, "pending_combat", None) or {}
    cursor = (pc.get("_round_resume") or {}).get("cursor")
    base = None
    hunt = None
    attacks = {}
    for i, action in enumerate(actions):
        if action.action_type in ("fellowship_move", "mordor_move"):
            if hunt is None:
                hunt = hunt_features(ai, state)
            out[i] = hunt
        elif action.action_type in ("army_attack", "siege_assault"):
            if action.action_type == "army_attack":
                source, target = action.params["from"], action.params["to"]
                # A co-located garrison SORTIE is a field battle.
                siege = bool(action.params.get("siege")) and ai._stronghold_controller(state, target) != side
                context = dict(attacker_region=source, defender_region=target,
                    attacker_side=side, defender_side=ai.Side.SHADOW if side == ai.Side.FREE_PEOPLES else ai.Side.FREE_PEOPLES,
                    is_siege=siege)
            else:
                context = {key: value for key, value in pc.items()
                           if key not in ("atk_combat_card", "def_combat_card", "atk_effect_count",
                                          "def_effect_count", "atk_effect_target", "def_effect_target")}
                context["is_siege"] = True
                source, target, siege = pc["attacker_region"], pc["defender_region"], True
            key = source, target, siege
            if key not in attacks:
                if action.action_type == "army_attack":
                    # Political nonparticipants and their mandatory guard do
                    # not contribute dice/leadership, even before declaration.
                    # freeze_attack only writes this local immutable roster
                    # descriptor; it neither pays a die nor moves a figure.
                    from army_participants import freeze_attack
                    freeze_attack(state, context)
                row = combat_features(_base_combat(state, context), context, action, side)
                row[I["opponent_card_unknown"]] = 1
                row[I["unresolved_choices"]] = 1  # cards/retreat/shine have not been chosen
                attacks[key] = row
            out[i] = attacks[key]
        elif action.action_type in ("combat_card_play", "combat_card_none", "combat_effect_choice") and pc:
            ec = pc.get("effect_choice") or {}
            if ec.get("stage") in ("post", "onslaught_payment"):
                # This is an actual post-casualty choice, with a known hit cost.
                if ec.get("effect") == "onslaught":
                    count = action.params["count"]
                    out[i, I["combat_available"]] = 1
                    _put(out[i], "own_additional_hits_0", plain_pmf(count, 3))
                continue
            if base is None:
                base = cursor.combat if cursor else _base_combat(state, pc)
            out[i] = combat_features(base, pc, action, side, cursor)
    return out
