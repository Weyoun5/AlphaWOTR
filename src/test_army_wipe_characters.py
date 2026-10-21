#!/usr/bin/env python3
"""ARMY-DESTRUCTION CHARACTER REGRESSION  (USER RULING 2026-08-15).

    "if the army is wiped out, leaders/minions die with it.  the only
     exception is if the army dies due to spirit of mordor or dreadful
     spells, in which case companions/nazgul/minions survive, but free
     people leaders die."

Every path where an army's units reach zero must go through the ONE
reconciler (combat.reconcile_destroyed_army / eliminate_army_characters
/ classify_wipe_characters).  The rule fires on a DESTRUCTION EVENT,
never on the mere absence of units: a separated Companion, a lone Nazgul
and a lone named Minion are LEGAL standing figures and must survive the
standing p.8 sweep (game_logic.remove_orphaned_leaders) untouched.

Run:  python test_army_wipe_characters.py     (exit 0 = all pass)
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai                                    # noqa: E402
import board                                 # noqa: E402
import combat as cb                          # noqa: E402
import event_effects as ee                   # noqa: E402
import game_logic                            # noqa: E402
import game_state as gs                      # noqa: E402
from board import sanitize                   # noqa: E402
from cards import get_card, get_cards        # noqa: E402

Side = gs.Side
Nation = gs.Nation
Kind = gs.UnitKind
Minion = game_logic.Minion


# ---------------------------------------------------------------------------
# Tiny runner (same contract as the other src/test_*.py files)
# ---------------------------------------------------------------------------

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def army_at(st, region_key):
    return game_logic.get_or_create_army(st, region_key)


def entry(st, region_key):
    """The live state.armies entry for region_key, or None."""
    san = sanitize(str(region_key))
    for k, a in st.armies.items():
        if sanitize(k) == san:
            return a
    return None


def clear(st, region_key):
    a = army_at(st, region_key)
    a.units.clear()
    a.leaders.clear()
    return a


def force_d6(value):
    """Force BOTH dice sources (combat rolls and card rolls) to `value`.
    Returns a restore() callable."""
    o_cb, o_ee = cb.roll_d6_batch, ee._d6_batch
    cb.roll_d6_batch = lambda n, ctx='', **kw: [value] * n
    ee._d6_batch = lambda n, ctx='card effect': [value] * n

    def restore():
        cb.roll_d6_batch = o_cb
        ee._d6_batch = o_ee
    return restore


def kinds_at(st, region_key):
    """Multiset of character kinds standing at region_key."""
    a = entry(st, region_key)
    out = []
    for l in (a.leaders if a is not None else []):
        if isinstance(l, gs.Companion):
            out.append('companion:' + str(l.name))
        elif isinstance(l, gs.Nazgul):
            out.append('nazgul')
        elif type(l).__name__ == 'Minion':
            out.append('minion:' + str(getattr(l, 'name', '')))
        else:
            out.append('leader')
    return sorted(out)


def companion_on_board(st, name):
    return any(isinstance(l, gs.Companion) and l.name == name
               for a in st.armies.values() for l in a.leaders)


def arm_minions(st):
    """Put the three named Minions in play so their elimination is
    OBSERVABLE through the character-sheet flags."""
    for attr, nm in (('witch_king', 'Witch King'), ('saruman', 'Saruman'),
                     ('mouth_of_sauron', 'Mouth of Sauron')):
        m = getattr(st, attr, None)
        if m is None:
            m = gs.ShadowMinion(name=nm, level=3, leadership=2)
            setattr(st, attr, m)
        m.in_play = True


# ---------------------------------------------------------------------------
# 1. The classifier contract (the ruling, in one table)
# ---------------------------------------------------------------------------

@test
def test_classifier_contract():
    figs = [gs.Leader(Nation.GONDOR),
            gs.Companion(name='Legolas', level=2, leadership=1),
            gs.Nazgul(Nation.SAURON),
            Minion(nation=Nation.SAURON, name='Witch King')]

    # BATTLE (no card): every kind dies with the army
    dead, alive = cb.classify_wipe_characters(figs, None, None)
    assert len(dead) == 4 and alive == [], \
        "battle wipe kills every character kind: %r" % (alive,)

    # The SPARING cards: only the plain Leader dies.  USER rulings
    # 2026-08-15 — Spirit of Mordor + Dreadful Spells (first ruling),
    # then case-by-case: "Dead men of dunharrow and ents kill
    # nazgul/minions. Return to valinor and faramir's rangers does
    # not" -> the latter two spare as well.
    for nm in ('The Spirit of Mordor', 'Dreadful Spells',
               'the spirit of mordor', 'DREADFUL SPELLS',
               'Return to Valinor', "Faramir's Rangers",
               "FARAMIR'S RANGERS"):
        dead, alive = cb.classify_wipe_characters(figs, None, nm)
        assert [type(d).__name__ for d in dead] == ['Leader'], \
            "%s: only the plain Leader dies, got %r" % (nm, dead)
        assert len(alive) == 3, \
            "%s: companion/nazgul/minion survive, got %r" % (nm, alive)

    # Cards that KILL characters when their casualties wipe the army
    for nm in ('Dead Men of Dunharrow', 'The Ents Awake',
               'The Ents Awake: Huorns'):
        dead, alive = cb.classify_wipe_characters(figs, None, nm)
        assert len(dead) == 4 and alive == [], \
            "%s is NOT a sparing card: %r" % (nm, alive)

    # dying_side filter: a co-located ENEMY figure is never collateral
    dead, alive = cb.classify_wipe_characters(figs, Side.SHADOW, None)
    assert sorted(type(d).__name__ for d in dead) == ['Minion', 'Nazgul'], \
        "only the dying side's figures die: %r" % (dead,)
    assert len(alive) == 2, "the FP figures stand: %r" % (alive,)

    # published context == explicit card name
    prev = cb.set_wipe_card_context('Dreadful Spells')
    try:
        dead, alive = cb.classify_wipe_characters(figs, None, None)
        assert [type(d).__name__ for d in dead] == ['Leader'], \
            "the published wipe context selects the exception"
    finally:
        cb.set_wipe_card_context(prev)
    dead, _ = cb.classify_wipe_characters(figs, None, None)
    assert len(dead) == 4, "the context is restored (battle rules again)"


# ---------------------------------------------------------------------------
# 2. Battle wipe kills ALL character kinds
# ---------------------------------------------------------------------------

@test
def test_battle_wipe_kills_all_character_kinds():
    st = setup_game()
    arm_minions(st)
    st.fp_casualties = {}

    # FP defender: 1 regular + a plain Leader + a separated Companion
    d = clear(st, 'osgiliath')
    d.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    d.leaders.append(gs.Leader(Nation.GONDOR))
    d.leaders.append(gs.Companion(name='Legolas', level=2, leadership=1))

    # Shadow attacker: enough dice to wipe it, with a Nazgul + the
    # Witch-king riding along
    a = clear(st, 'north ithilien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.leaders.append(gs.Nazgul(Nation.SAURON))

    restore = force_d6(6)
    try:
        game_logic.battle(st, 'north ithilien', 'osgiliath')
    finally:
        restore()

    assert kinds_at(st, 'osgiliath') == [], \
        "a battle wipe leaves NO character standing: %r" \
        % (kinds_at(st, 'osgiliath'),)
    assert not companion_on_board(st, 'Legolas'), \
        "the Companion died with the destroyed army"
    assert st.fp_casualties.get(Nation.GONDOR, {}).get('leader', 0) >= 1, \
        "the FP plain Leader is booked as a PERMANENT casualty (p.30)"


@test
def test_battle_wipe_kills_shadow_characters_and_books_minions():
    st = setup_game()
    arm_minions(st)

    d = clear(st, 'gap of rohan')
    d.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    d.leaders.append(gs.Nazgul(Nation.SAURON))
    d.leaders.append(Minion(nation=Nation.SAURON, name='Witch King'))
    d.leaders.append(Minion(nation=Nation.SAURON,
                            name='Mouth of Sauron'))

    a = clear(st, 'fords of isen')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.ROHAN, Kind.REGULAR))

    restore = force_d6(6)
    try:
        game_logic.battle(st, 'fords of isen', 'gap of rohan')
    finally:
        restore()

    assert kinds_at(st, 'gap of rohan') == [], \
        "Nazgul and named Minions die with the destroyed army: %r" \
        % (kinds_at(st, 'gap of rohan'),)
    assert st.witch_king.in_play is False, \
        "the Witch-king flips OUT of play (no phantom action die)"
    assert st.mouth_of_sauron.in_play is False, \
        "the Mouth flips OUT of play (no phantom action die)"


@test
def test_battle_wipe_fires_companion_casualty_triggers():
    """A Companion eliminated with an army is a real CASUALTY: 'Worn
    with Sorrow and Toil' must fire exactly as on a hunt casualty."""
    hand_card = next((cid for cid in get_cards()
                      if cid.startswith('FREE') and '_Character_' in cid),
                     None)
    assert hand_card is not None, "an FP Character card id resolves"

    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Character_Worn',
        card_name='Worn with Sorrow and Toil', side=Side.SHADOW,
        card_type=gs.PersistentCardType.WORN_WITH_SORROW))
    st.hands[Side.FREE_PEOPLES].hand = [hand_card]     # single source

    d = clear(st, 'osgiliath')
    d.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    d.leaders.append(gs.Companion(name='Legolas', level=2, leadership=1))
    a = clear(st, 'north ithilien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))

    restore = force_d6(6)
    try:
        game_logic.battle(st, 'north ithilien', 'osgiliath')
    finally:
        restore()

    assert not companion_on_board(st, 'Legolas'), "the Companion is dead"
    assert hand_card not in st.hands[Side.FREE_PEOPLES].hand, \
        "Worn with Sorrow and Toil fired on the army-wipe companion death"


# ---------------------------------------------------------------------------
# 3. The two exception cards
# ---------------------------------------------------------------------------

@test
def test_spirit_of_mordor_wipe_spares_characters():
    CID = next((c for c in get_cards()
                if getattr(get_card(c), 'card_name', '')
                == 'The Spirit of Mordor'), None)
    assert CID is not None, "The Spirit of Mordor card id resolves"

    st = setup_game()
    arm_minions(st)
    a = clear(st, 'dol guldur')
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))   # 2+ nations
    a.leaders.append(gs.Nazgul(Nation.SAURON))
    a.leaders.append(Minion(nation=Nation.SAURON, name='Witch King'))

    restore = force_d6(5)                                    # 5+ = hit
    try:
        ee._effect_spirit_of_mordor(
            st, get_card(CID), {'target_region': sanitize('dol guldur')})
    finally:
        restore()

    left = entry(st, 'dol guldur')
    assert left is not None and not left.units, \
        "the target army is wiped out"
    assert kinds_at(st, 'dol guldur') == ['minion:Witch King', 'nazgul'], \
        "Spirit of Mordor SPARES nazgul/minions: %r" \
        % (kinds_at(st, 'dol guldur'),)
    assert st.witch_king.in_play is True, \
        "a spared Witch-king is NOT booked out of play"


@test
def test_dreadful_spells_wipe_spares_companion_kills_fp_leader():
    CID = next((c for c in get_cards()
                if getattr(get_card(c), 'card_name', '')
                == 'Dreadful Spells'), None)
    assert CID is not None, "Dreadful Spells card id resolves"

    st = setup_game()
    st.fp_casualties = {}
    f = clear(st, 'osgiliath')
    f.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    f.leaders.append(gs.Leader(Nation.GONDOR))
    f.leaders.append(gs.Companion(name='Legolas', level=2, leadership=1))

    s = clear(st, 'north ithilien')
    s.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    for _ in range(3):
        s.leaders.append(gs.Nazgul(Nation.SAURON))

    restore = force_d6(5)
    try:
        ee._effect_dreadful_spells(st, get_card(CID), {
            'target_region': sanitize('north ithilien'),
            'target_fp_army': sanitize('osgiliath')})
    finally:
        restore()

    left = entry(st, 'osgiliath')
    assert left is not None and not left.units, "the FP army is wiped out"
    assert kinds_at(st, 'osgiliath') == ['companion:Legolas'], \
        "Dreadful Spells kills the plain FP Leader and SPARES the " \
        "Companion: %r" % (kinds_at(st, 'osgiliath'),)
    assert st.fp_casualties.get(Nation.GONDOR, {}).get('leader', 0) >= 1, \
        "the dead FP Leader is booked as a PERMANENT casualty"


# ---------------------------------------------------------------------------
# 4. Faramir's Rangers SPARES (USER case-by-case ruling 2026-08-15)
# ---------------------------------------------------------------------------

@test
def test_faramirs_rangers_wipe_spares_characters():
    CID = next((c for c in get_cards()
                if getattr(get_card(c), 'card_name', '')
                == "Faramir's Rangers"), None)
    assert CID is not None, "Faramir's Rangers card id resolves"

    st = setup_game()
    arm_minions(st)
    a = clear(st, 'south ithilien')
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.leaders.append(gs.Nazgul(Nation.SAURON))
    a.leaders.append(Minion(nation=Nation.SAURON, name='Witch King'))

    restore = force_d6(5)
    try:
        ee._effect_faramirs_rangers(
            st, get_card(CID),
            {'target_region': sanitize('south ithilien')})
    finally:
        restore()

    # USER: "Return to valinor and faramir's rangers does not [kill
    # nazgul/minions]" — the units die, the characters stand.
    left = kinds_at(st, 'south ithilien')
    assert any('nazgul' in k for k in left) \
        and any('minion' in k for k in left), \
        "Faramir's Rangers SPARES characters (user ruling): %r" % (left,)
    assert st.witch_king.in_play is True, \
        "the spared Witch-king stays in play"


# ---------------------------------------------------------------------------
# 5. Standing characters are NEVER touched (no destruction event)
# ---------------------------------------------------------------------------

@test
def test_separated_companion_alone_is_never_killed():
    st = setup_game()
    comp = gs.Companion(name='Gimli', level=2, leadership=1)
    game_logic.place_separated_companion(st, comp, 'carrock')
    assert kinds_at(st, 'carrock') == ['companion:Gimli']

    for _ in range(3):
        game_logic.remove_orphaned_leaders(st)
        game_logic.enforce_settlement_control(st)
        game_logic.cleanup_sieges(st)
    assert kinds_at(st, 'carrock') == ['companion:Gimli'], \
        "a separated Companion standing alone survives every cleanup"

    # ... and an unrelated battle elsewhere does not reach him
    d = clear(st, 'osgiliath')
    d.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    a = clear(st, 'north ithilien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    restore = force_d6(6)
    try:
        game_logic.battle(st, 'north ithilien', 'osgiliath')
    finally:
        restore()
    assert kinds_at(st, 'carrock') == ['companion:Gimli'], \
        "an unrelated wipe never reaches a standing Companion"


@test
def test_lone_nazgul_and_minion_are_never_killed():
    st = setup_game()
    arm_minions(st)
    a = clear(st, 'carrock')
    a.leaders.append(gs.Nazgul(Nation.SAURON))
    b = clear(st, 'rhosgobel')
    b.leaders.append(Minion(nation=Nation.SAURON, name='Mouth of Sauron'))

    for _ in range(3):
        game_logic.remove_orphaned_leaders(st)
        game_logic.enforce_settlement_control(st)
        game_logic.cleanup_sieges(st)

    assert kinds_at(st, 'carrock') == ['nazgul'], \
        "a lone Nazgul stands (it is not a destruction event)"
    assert kinds_at(st, 'rhosgobel') == ['minion:Mouth of Sauron'], \
        "a lone named Minion stands"
    assert st.mouth_of_sauron.in_play is True, \
        "a standing Minion is never booked out of play"


@test
def test_colocated_enemy_character_survives_a_wipe():
    """A separated Companion sharing a region with the Shadow army that
    gets destroyed there is NOT collateral (dying_side filter)."""
    st = setup_game()
    d = clear(st, 'gap of rohan')
    d.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    d.leaders.append(gs.Nazgul(Nation.SAURON))
    d.leaders.append(gs.Companion(name='Gimli', level=2, leadership=1))

    a = clear(st, 'fords of isen')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.ROHAN, Kind.REGULAR))

    restore = force_d6(6)
    try:
        game_logic.battle(st, 'fords of isen', 'gap of rohan')
    finally:
        restore()

    assert kinds_at(st, 'gap of rohan') == ['companion:Gimli'], \
        "the Shadow figures die, the FP Companion stands: %r" \
        % (kinds_at(st, 'gap of rohan'),)


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ARMY-DESTRUCTION CHARACTER REGRESSION (USER RULING 2026-08-15)")
    print("=" * 70)
    npass = nfail = 0
    for fn in TESTS:
        try:
            fn()
            print("  [PASS] %s" % fn.__name__)
            npass += 1
        except Exception:
            print("  [FAIL] %s" % fn.__name__)
            for line in traceback.format_exc().splitlines()[-6:]:
                print("         " + line)
            nfail += 1
    print("-" * 70)
    print("TOTAL %d  |  PASS %d  FAIL %d" % (npass + nfail, npass, nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
