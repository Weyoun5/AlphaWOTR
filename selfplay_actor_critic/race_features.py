"""Where the race scalars live in the v52 observation (src/ai.py encode_state).

The vector is side-relative (my/enemy slots) with no side flag, so each
side gets its own RaceLayout whose scalar NAMES are absolute (fp_vp, sh_vp,
...) while the INDICES pick the right slot for that side. Army strengths
per region use the absolute per-nation counts inside each region block
(f14+2i regulars /15, f15+2i elites /5, nation order ELVES, DWARVES, GONDOR,
THE_NORTH, ROHAN, ISENGARD, SAURON, SOUTHRON_EASTERLINGS), which are the same
numbers in both views. Settlement tables come from the board (city 1 VP,
stronghold 2 VP) with the same FP/Shadow ownership predicate the engine's
victory-point count uses, so the race model scores exactly the settlements
that decide the military clocks. Verified against the encoder by mutation
probes on 2026-10-03 (scratch report race_feature_map.md).
"""
from .race_critic import RaceLayout

REGION_STRIDE = 30
REGION_BASE = 0
NATION_COUNT = 8
FP_NATIONS = (0, 1, 2, 3, 4)
SH_NATIONS = (5, 6, 7)
POLITICAL_BASE = 3150
FELLOWSHIP = dict(fs_progress=3182, corruption=3183, fs_hidden=3184, mordor_entered=3185, mordor_step=3186,
                  companions_in_fs=3187, guide_level=3188, mordor_proximity=3189)
HUNT = dict(eyes_allocated=3192, rolled_eyes_in_box=3193, fp_dice_in_box=3194, hunt_pool_tiles=3195,
            hunt_pool_eyes=3196, hunt_pool_sh_special=3197, hunt_pool_fp_special=3198, hunt_pool_stops=3199,
            hunt_pool_mean_damage=3200, mordor_pool=3201)
DIE_FACES = ("character", "army", "muster", "event", "muster_army", "eye", "will_of_west")
DECKS = dict(deck_fp_char=3222, deck_fp_strat=3223, deck_sh_char=3224, deck_sh_strat=3225)
TURN = 3339
NATIONS_AT_WAR = dict(fp_nations_at_war=3340, sh_nations_at_war=3341)
COMPANIONS = ("gandalf_grey", "strider", "legolas", "gimli", "boromir", "merry", "pippin", "gollum", "aragorn",
              "gandalf_white")
COMPANION_BASE = 3342
UNITS_ON_BOARD_BASE = 3385          # 8 x (regulars/max, elites/max)
REINFORCEMENT_BASE = 3421           # 8 x (regulars, elites, leaders remaining/max)
ENTER_PLAY_FLAGS = 3447             # 5 flags
PENDING = dict(pending_battle=3675, i_attack=3677, pending_hunt=3702, eye_allocation_phase=3717, declare_phase=3718)
# Side-relative slot pairs: (slot in the owner's own view, slot in the other side's view).
MY_ENEMY = dict(vp=(3190, 3191), dice_unused=(3202, 3203), hand=(3218, 3219), vp_capturable=(3445, 3446))
MY_FACES, ENEMY_FACES = 3204, 3211
RINGS = dict(fp=(3333, 3336), sh=(3336, 3333))          # (fp rings slot, sh rings slot) per viewing side
SPECIALS = dict(fp=dict(gandalf_white=3328, aragorn=3329, witch_king=3330, saruman=3331, mouth=3332),
                sh=dict(witch_king=3328, saruman=3329, mouth=3330, gandalf_white=3331, aragorn=3332))
# Captured flags inside a region block: f7 = captured by MY side, f8 = captured by the ENEMY.
CAPTURED_BY_ME, CAPTURED_BY_ENEMY = 7, 8


def _mine(side, owner):
    """Pick the (my, enemy) slot pair element for an absolute owner."""
    return 0 if side == owner else 1


def race_scalars(side: str) -> list:
    """[(name, index), ...] absolute names, side-specific indices."""
    other = "sh" if side == "fp" else "fp"
    scalars = [("turn", TURN)]
    scalars += list(FELLOWSHIP.items())
    for key, pair in MY_ENEMY.items():
        scalars.append((f"fp_{key}", pair[_mine(side, "fp")]))
        scalars.append((f"sh_{key}", pair[_mine(side, "sh")]))
    scalars += list(HUNT.items())
    for k, face in enumerate(DIE_FACES):
        scalars.append((f"fp_face_{face}", (MY_FACES if side == "fp" else ENEMY_FACES) + k))
        scalars.append((f"sh_face_{face}", (MY_FACES if side == "sh" else ENEMY_FACES) + k))
    scalars += list(DECKS.items())
    for i in range(NATION_COUNT):
        scalars += [(f"political_pos_{i}", POLITICAL_BASE + 4 * i), (f"political_active_{i}", POLITICAL_BASE + 4 * i + 1),
                    (f"political_at_war_{i}", POLITICAL_BASE + 4 * i + 2)]
    scalars += list(NATIONS_AT_WAR.items())
    for j, name in enumerate(COMPANIONS):
        scalars += [(f"{name}_alive", COMPANION_BASE + 4 * j), (f"{name}_in_fs", COMPANION_BASE + 4 * j + 1),
                    (f"{name}_level", COMPANION_BASE + 4 * j + 2)]
    scalars += [("fp_elven_rings", RINGS[side][0]), ("sh_elven_rings", RINGS[side][1])]
    scalars += [(name, index) for name, index in SPECIALS[side].items()]
    for i in range(NATION_COUNT):
        scalars += [(f"units_regulars_{i}", UNITS_ON_BOARD_BASE + 2 * i), (f"units_elites_{i}", UNITS_ON_BOARD_BASE + 2 * i + 1)]
        scalars += [(f"reinforce_regulars_{i}", REINFORCEMENT_BASE + 3 * i), (f"reinforce_elites_{i}", REINFORCEMENT_BASE + 3 * i + 1),
                    (f"reinforce_leaders_{i}", REINFORCEMENT_BASE + 3 * i + 2)]
    scalars += [(f"enter_play_{k}", ENTER_PLAY_FLAGS + k) for k in range(5)]
    scalars += list(PENDING.items())
    del other
    return scalars


def settlement_table(ai, side: str):
    """Settlements in region-index order with VP, owner (0 FP / 1 Shadow), the
    side-specific captured-flag index and the neighbouring region indices."""
    import game_logic
    board_module = ai.board
    board = board_module.Board()
    keys = list(ai.ALL_REGION_KEYS)
    index = {k: i for i, k in enumerate(keys)}
    rows = []
    for key in keys:
        info = board_module.regions[key]
        is_city, is_stronghold = bool(info.get("is_city", False)), bool(info.get("is_stronghold", False))
        if not is_city and not is_stronghold:
            continue
        if game_logic.is_fp_region(key):
            owner = 0
        elif game_logic.is_shadow_region(key):
            owner = 1
        else:
            continue
        r = index[key]
        captured_by = "sh" if owner == 0 else "fp"
        field = CAPTURED_BY_ME if captured_by == side else CAPTURED_BY_ENEMY
        neighbors = tuple(sorted(index[board_module.sanitize(n)] for n in board.adjacent(key)
                                 if board_module.sanitize(n) in index and board_module.sanitize(n) != key))
        rows.append(dict(key=key, region=r, vp=2 if is_stronghold else 1, owner=owner,
                         captured_index=REGION_BASE + REGION_STRIDE * r + field, neighbors=neighbors))
    return rows


def build_layout(ai, side: str) -> RaceLayout:
    if side not in ("fp", "sh"):
        raise ValueError("side must be 'fp' or 'sh'")
    from .contracts import CONTRACT
    scalars = race_scalars(side)
    settlements = settlement_table(ai, side)
    fp_strength = tuple((14 + 2 * i, 15.) for i in FP_NATIONS) + tuple((15 + 2 * i, 10.) for i in FP_NATIONS)
    sh_strength = tuple((14 + 2 * i, 15.) for i in SH_NATIONS) + tuple((15 + 2 * i, 10.) for i in SH_NATIONS)
    return RaceLayout(
        width=int(CONTRACT.observation_width),
        scalar_names=tuple(name for name, _ in scalars),
        scalar_index=tuple(int(i) for _, i in scalars),
        scalar_scale=tuple(1. for _ in scalars),
        region_base=REGION_BASE, region_count=int(ai.NUM_REGIONS), region_stride=REGION_STRIDE,
        fp_strength=fp_strength, sh_strength=sh_strength,
        settlement_region=tuple(s["region"] for s in settlements),
        settlement_vp=tuple(s["vp"] for s in settlements),
        settlement_owner=tuple(s["owner"] for s in settlements),
        settlement_captured_index=tuple(s["captured_index"] for s in settlements),
        settlement_neighbors=tuple(s["neighbors"] for s in settlements))
