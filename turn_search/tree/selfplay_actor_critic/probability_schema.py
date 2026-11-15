"""Names and units for conditional chance components (no engine imports)."""

VERSION = "visible_chance_components_rosters_conditions_v3"
BASE_NAMES = (
    "combat_available", "opponent_card_unknown", "own_card_present", "enemy_card_present",
    "own_card_cancelled", "enemy_card_cancelled", "unresolved_choices", "current_roster_condition",
)
SIDE_NAMES = (
    "dice_over_5", "leadership_over_10", "hit_probability", "reroll_hit_probability",
    "roll_observed", "confusion_self_hits_over_5", "bonus_per_success_over_2",
    "shield_expected_cancel", "conversion_limit_over_2", "conversion_cost_over_5",
    "initiative_over_10", "later_effect",
) + tuple(f"roll_hits_{i}" for i in range(6)) + tuple(f"pre_hits_{i}" for i in range(6)) + tuple(f"additional_hits_{i}" for i in range(6))
HUNT_NAMES = (
    "hunt_available", "hunt_mordor", "raw_before_mitigation", "tile_draw_probability",
    "tile_reveal_icon_probability", "tile_stop_icon_probability", "tile_available",
) + tuple(f"hunt_hits_{i}" for i in range(6)) + tuple(f"raw_damage_{i}" for i in range(-2, 16)) + ("raw_damage_16_plus",)
NAMES = BASE_NAMES + tuple(f"{side}_{name}" for side in ("own", "enemy") for name in SIDE_NAMES) + HUNT_NAMES
WIDTH = len(NAMES)
INDEX = {name: i for i, name in enumerate(NAMES)}
